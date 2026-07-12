// the training run.json serde_json::json! literal outgrew the default macro
// recursion limit when the S41 aug_copies key landed (the macro recurses per
// token, not per key — this is a compile-time-only knob)
#![recursion_limit = "256"]

pub mod audio;
pub mod commands;
pub mod download;
pub mod inference;
pub mod logging;
pub mod models;
pub mod pyenv;
pub mod separation;
pub mod training;
pub mod util;

use std::sync::Arc;
use tauri::{Emitter, Manager};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::Layer;

#[derive(thiserror::Error, Debug)]
pub enum UtaiError {
    #[error("Inference error: {0}")]
    Inference(String),
    #[error("Audio processing error: {0}")]
    Audio(String),
    #[error("Project error: {0}")]
    Project(String),
    #[error("Training error: {0}")]
    Training(String),
    #[error("Model error: {0}")]
    Model(String),
    #[error("Download error: {0}")]
    Download(String),
    #[error("Runtime pack error: {0}")]
    Pyenv(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

impl serde::Serialize for UtaiError {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(self.to_string().as_ref())
    }
}

pub type Result<T> = std::result::Result<T, UtaiError>;

pub struct AppState {
    pub inference: inference::InferenceManager,
    pub training: training::TrainingManager,
    pub models: models::ModelRegistry,
    pub separation: separation::SeparationManager,
    pub log_buffer: Arc<logging::LogBuffer>,
    pub cache_dir: std::path::PathBuf,
    pub app_dir: std::path::PathBuf,
    pub msst_models_dir: std::path::PathBuf,
    /// Long tasks currently running (stable ids → `close.task_<id>` labels), so the close-flow's
    /// in-progress warning can LIST what's running. Registered via `begin_task` (RAII). Training +
    /// separation are queried directly in `running_tasks`, not stored here.
    pub active_tasks: Arc<parking_lot::Mutex<std::collections::HashMap<String, usize>>>,
}

impl AppState {
    pub fn new(
        app_dir: std::path::PathBuf,
        cache_dir: std::path::PathBuf,
        models_dir: std::path::PathBuf,
        log_buffer: Arc<logging::LogBuffer>,
    ) -> Self {
        let msst_models_dir = models_dir.join("msst");
        Self {
            inference: inference::InferenceManager::new(),
            training: training::TrainingManager::new(app_dir.clone()),
            models: models::ModelRegistry::new(models_dir),
            separation: separation::SeparationManager::new(app_dir.clone()),
            log_buffer,
            cache_dir,
            app_dir,
            msst_models_dir,
            active_tasks: Arc::new(parking_lot::Mutex::new(std::collections::HashMap::new())),
        }
    }

    /// Register a long-running task so the close-flow's in-progress warning can list it. Returns a guard
    /// that unregisters on drop (panic-safe). Use a stable id with a matching `close.task_<id>` locale key.
    pub fn begin_task(&self, id: &str) -> TaskGuard {
        *self.active_tasks.lock().entry(id.to_string()).or_insert(0) += 1;
        TaskGuard {
            tasks: Arc::clone(&self.active_tasks),
            id: id.to_string(),
        }
    }
}

/// RAII guard from `AppState::begin_task` — decrements the task's refcount when dropped (removing the id at
/// 0), so N concurrent same-id tasks all stay listed until the LAST finishes. Clears on normal return, `?`
/// early-return, OR panic.
pub struct TaskGuard {
    tasks: Arc<parking_lot::Mutex<std::collections::HashMap<String, usize>>>,
    id: String,
}

impl Drop for TaskGuard {
    fn drop(&mut self) {
        let mut tasks = self.tasks.lock();
        if let Some(n) = tasks.get_mut(&self.id) {
            *n -= 1;
            if *n == 0 {
                tasks.remove(&self.id);
            }
        }
    }
}

/// Stop Windows from popping a MODAL "DLL not found" dialog that BLOCKS the process when a DLL
/// dependency can't be resolved — let LoadLibrary fail fast instead. MUST run before the first
/// ort DLL load (run() does; standalone harnesses like `cargo test` must call it themselves,
/// otherwise a missing CUDA dependency hangs the process at 0 CPU with an invisible dialog).
pub fn suppress_windows_dll_error_dialogs() {
    #[cfg(windows)]
    {
        extern "system" {
            fn SetErrorMode(u_mode: u32) -> u32;
        }
        const SEM_FAILCRITICALERRORS: u32 = 0x0001;
        const SEM_NOOPENFILEERRORBOX: u32 = 0x8000;
        unsafe {
            SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX);
        }
    }
}

/// Which ORT BUILD init_ort_runtime actually committed ("CUDA" / "DirectML" / a
/// dev-cache/system path). Recorded so later startup lines can state the FACT of
/// what loaded instead of re-announcing the Auto chain — the S22 logging rule
/// ("log the actual hardware, not the intent") applied to the startup path.
pub static ORT_LOADED_BUILD: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// Find and initialize the ORT runtime DLL.
/// Picks CUDA or DirectML DLL based on user's saved device preference.
/// pub so out-of-app harnesses (integration tests) can reuse the exact app init path.
pub fn init_ort_runtime(app_dir: &std::path::Path) {
    let dll_name = "onnxruntime.dll";
    let runtime_dir = app_dir.join("runtime").join("ort");

    // Decide which ORT BUILD to load. ORT bundles ONE provider set per DLL: the DirectML build has
    // no CUDA provider and vice-versa. Explicit "cuda" → CUDA build; "directml"/"cpu" → DirectML
    // build. Auto/unset → load the CUDA build ONLY when this machine actually has CUDA (Toolkit cudart
    // present) AND the CUDA build is downloaded — that's what lets Auto's "CUDA → … → CPU" chain
    // actually light up CUDA. On a non-CUDA machine we keep the DirectML build, because loading the
    // CUDA build there has no DirectML provider and would drop Auto straight to CPU.
    let cuda_dll = runtime_dir.join("cuda").join(dll_name);
    // Auto/unset → load the CUDA build when this machine actually has CUDA (Toolkit cudart present) AND
    // the CUDA build is downloaded — the whole point of Auto: light up CUDA without a manual pick. The
    // CUDA build MUST be the version matching the ort crate's API (1.24.x / API 24); a mismatch deadlocks.
    let prefer_cuda = match read_device_preference(app_dir).as_deref() {
        Some("cuda") => true,
        Some("directml") | Some("cpu") => false,
        _ => cuda_dll.exists() && crate::commands::settings::is_cuda_available(),
    };

    let mut search_paths: Vec<std::path::PathBuf> = Vec::new();

    if prefer_cuda {
        if cuda_dll.exists() {
            // PRELOAD CUDA + cuDNN dylibs in the top-level loader context BEFORE ort's init_from loads the
            // CUDA build. Without this, ort's nested load of providers_cuda (under the loader-lock) hangs.
            // VERIFIED with the version-matched 1.24.x build: preload + init_from + CUDA session all succeed.
            let cuda_bin = std::env::var("CUDA_PATH").ok().map(|p| std::path::PathBuf::from(p).join("bin"));
            let cudnn_dir = app_dir.join("runtime").join("cuda");
            let _ = ort::ep::cuda::preload_dylibs(cuda_bin.as_deref(), Some(&cudnn_dir));
            search_paths.push(cuda_dll);
            tracing::info!("CUDA available — preloaded CUDA dylibs + loading CUDA ORT build");
        } else {
            tracing::warn!("CUDA preferred but runtime/ort/cuda/ missing — using DirectML build");
        }
    }

    // Default: DirectML DLL
    search_paths.push(runtime_dir.join(dll_name));

    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            search_paths.push(dir.join(dll_name));
        }
    }

    // Dev mode: check ort-sys download cache
    #[cfg(windows)]
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        let cache_base = std::path::PathBuf::from(&local).join("ort.pyke.io").join("dfbin");
        if cache_base.exists() {
            if let Ok(entries) = std::fs::read_dir(&cache_base) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.is_dir() {
                        let dll = p.join(dll_name);
                        if dll.exists() {
                            search_paths.push(dll);
                        }
                    }
                }
            }
        }
    }

    for path in &search_paths {
        if path.exists() {
            tracing::info!("Loading ORT from: {}", path.display());
            match ort::init_from(path) {
                Ok(builder) => {
                    builder.commit();
                    quiet_ort_logging();
                    // Classify what ACTUALLY loaded (fact, not intent) for later log lines.
                    let build = if path.components().any(|c| c.as_os_str() == "cuda") {
                        "CUDA".to_string()
                    } else if *path == runtime_dir.join(dll_name) {
                        "DirectML".to_string()
                    } else {
                        format!("dev/system ({})", path.display())
                    };
                    tracing::info!("ORT runtime loaded successfully ({build} build)");
                    let _ = ORT_LOADED_BUILD.set(build);
                    return;
                }
                Err(e) => {
                    tracing::warn!("ORT init from '{}' failed: {}", path.display(), e);
                }
            }
        }
    }

    tracing::warn!("No local ORT DLL found, trying system PATH...");
    ort::init().commit();
    quiet_ort_logging();
    let _ = ORT_LOADED_BUILD.set("system PATH".to_string());
}

/// Silence ORT's per-op VERBOSE logging AT THE SOURCE. The `ort` crate's tracing bridge creates
/// the ORT environment hardcoded at `ORT_LOGGING_LEVEL_VERBOSE`, so ORT invokes the Rust log
/// callback for EVERY tensor allocation ("block in memory pattern…") — thousands per inference,
/// crossing the FFI boundary + formatting a string each time even when the subscriber later drops
/// it. Raising the env level to Warning makes ORT skip the callback entirely for verbose/info
/// messages (real warnings + errors still surface). Belt-and-suspenders with the file-log filter.
fn quiet_ort_logging() {
    if let Ok(env) = ort::environment::current() {
        env.set_log_level(ort::logging::LogLevel::Warning);
    }
}

/// Read the saved device string ("cuda"/"directml"/"cpu"/"auto") from config.json, if any.
/// Handles both the unit form ("auto"/"cpu") and the struct form ({"cuda":{"device_id":0}}).
fn read_device_preference(app_dir: &std::path::Path) -> Option<String> {
    let cfg_path = app_dir.join("config.json");
    let content = std::fs::read_to_string(&cfg_path).ok()?;
    let val: serde_json::Value = serde_json::from_str(&content).ok()?;
    let device = val.get("device")?;
    if let Some(s) = device.as_str() {
        return Some(s.to_string());
    }
    device.as_object().and_then(|o| o.keys().next().cloned())
}

/// Add CUDA runtime DLL directories to PATH so ORT CUDA EP can find cuDNN/cublas.
/// pub because out-of-app harnesses (integration tests) must replicate this too:
/// the cudnn 9 shim resolves its sub-DLLs (cudnn_graph/engines_*) via PATH at
/// graph-build time — preload_dylibs alone leaves a bare `cargo test` failing
/// with CUDNN_BACKEND_API_FAILED at the first Conv (S39: masqueraded as an
/// environment drift until the app-vs-test PATH difference was isolated).
pub fn setup_cuda_dll_paths(app_dir: &std::path::Path) {
    #[cfg(windows)]
    {
        let mut dirs_to_add: Vec<std::path::PathBuf> = Vec::new();

        // 1. App-local CUDA runtime dir (downloaded via settings panel — holds cuDNN)
        let cuda_dir = app_dir.join("runtime").join("cuda");
        if cuda_dir.exists() {
            dirs_to_add.push(cuda_dir);
        }

        // 2. System CUDA Toolkit bin (cudart / cublas / cublasLt) — providers_cuda.dll needs these and
        //    runtime/cuda only ships cuDNN. Without this the CUDA build can't resolve cudart and the EP
        //    fails to register (or, before SetErrorMode, hung). Prefer CUDA_PATH from the Toolkit installer.
        if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
            let bin = std::path::PathBuf::from(&cuda_path).join("bin");
            if bin.exists() {
                dirs_to_add.push(bin);
            }
        }

        // 3. Python site-packages nvidia dirs (dev convenience) — only if nothing else was found.
        if dirs_to_add.is_empty() {
            if let Ok(local) = std::env::var("LOCALAPPDATA") {
                let python_base = std::path::PathBuf::from(&local)
                    .join("Programs")
                    .join("Python");
                if let Ok(entries) = std::fs::read_dir(&python_base) {
                    for entry in entries.flatten() {
                        let site_pkgs = entry.path().join("Lib").join("site-packages").join("nvidia");
                        if site_pkgs.exists() {
                            for sub in ["cudnn", "cublas"] {
                                let bin = site_pkgs.join(sub).join("bin");
                                if bin.exists() {
                                    dirs_to_add.push(bin);
                                }
                            }
                            break;
                        }
                    }
                }
            }
        }

        if !dirs_to_add.is_empty() {
            let current_path = std::env::var("PATH").unwrap_or_default();
            let additions: Vec<String> = dirs_to_add.iter()
                .map(|d| d.to_string_lossy().to_string())
                .collect();
            let new_path = format!("{};{}", additions.join(";"), current_path);
            std::env::set_var("PATH", &new_path);
            // S60c: ALSO register via AddDllDirectory — cudnn 9's FRONTEND path lazily loads
            // its engine DLLs (cudnn_engines_tensor_ir64_9.dll etc.) with LOAD_LIBRARY_SEARCH_*
            // semantics that IGNORE PATH once the process is in default-dirs mode; the PATH
            // prepend alone left GAME's convs failing CUDNN_FE on CUDA while the classic-API
            // convs (voice models) kept working.
            extern "system" {
                fn AddDllDirectory(new_directory: *const u16) -> *mut std::ffi::c_void;
            }
            for dir in &dirs_to_add {
                let wide: Vec<u16> = dir
                    .as_os_str()
                    .to_string_lossy()
                    .encode_utf16()
                    .chain(std::iter::once(0))
                    .collect();
                unsafe {
                    AddDllDirectory(wide.as_ptr());
                }
                tracing::info!("Added to PATH+DllDirectory for CUDA: {}", dir.display());
            }
        }
    }
}

/// Startup sweep of the on-disk cache (`local_data/cache`). Two policies:
///  1. `audio_cache/` — playback WAV copies of non-WAV imports are rewritten on next load, so they
///     cost nothing to regenerate: wipe the whole folder.
///  2. `cache/<segment_id>/` — per-segment workflow outputs (separation/voice-conversion) are
///     expensive to regenerate and may be referenced by saved projects, so keep recent ones and
///     enforce a budget: drop dirs older than MAX_AGE, then, if still over MAX_TOTAL_BYTES, drop
///     oldest-first (LRU by newest-file mtime) until under budget.
/// Only touches `local_data/cache`; autosaves and models live in sibling dirs.
fn cleanup_cache_on_startup(cache_dir: &std::path::Path) {
    const MAX_AGE_SECS: u64 = 30 * 24 * 60 * 60; // 30 days
    const MAX_TOTAL_BYTES: u64 = 3 * 1024 * 1024 * 1024; // 3 GB

    let audio_cache = cache_dir.join("audio_cache");
    if audio_cache.exists() {
        match std::fs::remove_dir_all(&audio_cache) {
            Ok(()) => tracing::info!("Cache: cleared audio_cache"),
            Err(e) => tracing::warn!("Cache: failed to clear audio_cache: {}", e),
        }
    }

    let entries = match std::fs::read_dir(cache_dir) {
        Ok(e) => e,
        Err(_) => return, // no cache dir yet
    };

    let now = std::time::SystemTime::now();
    let mut dirs: Vec<(std::path::PathBuf, std::time::SystemTime, u64)> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        match path.file_name().and_then(|n| n.to_str()) {
            Some("audio_cache") => continue, // already handled
            // usp_work = the open .usp's extracted media — lifecycle is owned by open_project_archive/
            // prune_usp_work + the recovery-aware removal in setup; the age/budget sweep must never eat
            // a pending recovery's files.
            Some("usp_work") => continue,
            _ => {}
        }
        // Nested per-RUN output dirs (cache/<segId>/r*/ — every render writes a fresh one so re-runs
        // never overwrite a split sibling's referenced audio): age-prune the stale ones individually,
        // or a long-lived segment's dead old runs would count against the byte budget forever while
        // its newest file keeps the whole segment dir "recent".
        if let Ok(runs) = std::fs::read_dir(&path) {
            for run in runs.flatten() {
                let rp = run.path();
                if !rp.is_dir() {
                    continue;
                }
                let (rm, _) = dir_mtime_and_size(&rp);
                if now.duration_since(rm).map(|a| a.as_secs() > MAX_AGE_SECS).unwrap_or(false) {
                    let _ = std::fs::remove_dir_all(&rp);
                    tracing::info!("Cache: pruned aged run dir {}", rp.display());
                }
            }
        }
        let (mtime, size) = dir_mtime_and_size(&path);
        // Age prune first.
        if now.duration_since(mtime).map(|a| a.as_secs() > MAX_AGE_SECS).unwrap_or(false) {
            let _ = std::fs::remove_dir_all(&path);
            tracing::info!("Cache: pruned aged segment dir {}", path.display());
            continue;
        }
        dirs.push((path, mtime, size));
    }

    // Budget prune: oldest-first until under the total-size cap.
    let mut total: u64 = dirs.iter().map(|(_, _, s)| *s).sum();
    if total > MAX_TOTAL_BYTES {
        dirs.sort_by_key(|(_, m, _)| *m); // oldest first
        for (path, _, size) in &dirs {
            if total <= MAX_TOTAL_BYTES {
                break;
            }
            let _ = std::fs::remove_dir_all(path);
            total = total.saturating_sub(*size);
            tracing::info!("Cache: budget-pruned segment dir {} ({} bytes)", path.display(), size);
        }
    }
}

/// Recursively compute a directory's total byte size and its newest contained-file mtime
/// (used as a "last activity" proxy for LRU). Returns (UNIX_EPOCH, 0) for an empty/unreadable dir.
fn dir_mtime_and_size(dir: &std::path::Path) -> (std::time::SystemTime, u64) {
    fn walk(d: &std::path::Path, newest: &mut std::time::SystemTime, size: &mut u64) {
        if let Ok(rd) = std::fs::read_dir(d) {
            for e in rd.flatten() {
                let Ok(md) = e.metadata() else { continue };
                if md.is_dir() {
                    walk(&e.path(), newest, size);
                } else {
                    *size += md.len();
                    if let Ok(m) = md.modified() {
                        if m > *newest {
                            *newest = m;
                        }
                    }
                }
            }
        }
    }
    let mut newest = std::time::SystemTime::UNIX_EPOCH;
    let mut size = 0u64;
    walk(dir, &mut newest, &mut size);
    (newest, size)
}

/// App root = the directory holding `converter/`. Resolved from the EXECUTABLE's location first
/// (stable regardless of launch context: a release exe sits next to converter/, a dev exe in
/// src-tauri/target/debug walks up to the repo root), with the old CWD probe as fallback. The
/// final fallback is the exe dir — NOT the CWD — so the data/models root can never silently move
/// with whatever directory the app happened to be launched from.
fn resolve_app_dir() -> std::path::PathBuf {
    let has_converter = |d: &std::path::Path| d.join("converter").join("convert.py").exists();

    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()));
    if let Some(dir) = &exe_dir {
        let mut cur = Some(dir.as_path());
        while let Some(d) = cur {
            if has_converter(d) {
                return d.to_path_buf();
            }
            cur = d.parent();
        }
    }

    let cwd = std::env::current_dir().unwrap_or_default();
    if has_converter(&cwd) {
        return cwd;
    }
    if let Some(parent) = cwd.parent() {
        if has_converter(parent) {
            return parent.to_path_buf();
        }
    }

    exe_dir.unwrap_or(cwd)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Observed: Auto loading the CUDA build hung on a missing cudart/cublas without this —
    // fail fast so init_ort_runtime falls back to the DirectML build instead of hanging.
    suppress_windows_dll_error_dialogs();
    let log_dir = logging::get_log_dir();
    let _ = std::fs::create_dir_all(&log_dir);

    let file_appender = tracing_appender::rolling::daily(&log_dir, "utai.log");
    let (non_blocking, _file_guard) = tracing_appender::non_blocking(file_appender);

    let log_buffer = Arc::new(logging::LogBuffer::new(2000));

    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "utai=info".into());

    // The FILE log MUST be filtered too. Without a filter it captures EVERY target at EVERY
    // level — including ORT's per-tensor-allocation VERBOSE/TRACE (`execution_frame.cc` "block
    // in memory pattern…" fires thousands of times per inference). Unfiltered, a single voice
    // run wrote 80k+ lines that (a) flooded the log file and (b) backed up in the non-blocking
    // appender's in-RAM buffer to multi-GB, dragging inference to a crawl. utai stays at debug
    // for post-crash forensics; ort/symphonia/everything else only surfaces WARN+ (real problems,
    // never the per-op spam). RUST_LOG overrides both layers for deep debugging.
    let file_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "warn,utai=debug".into());

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .with_filter(env_filter),
        )
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(non_blocking)
                .with_ansi(false)
                .with_filter(file_filter),
        )
        .with(logging::BufferLayer::new(Arc::clone(&log_buffer)))
        .init();

    tracing::info!("UTAI v2 starting — logs: {}", log_dir.display());

    let app_dir_early = resolve_app_dir();
    setup_cuda_dll_paths(&app_dir_early);
    init_ort_runtime(&app_dir_early);

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        // Exclude VISIBLE from restore so the plugin doesn't show() the window EARLY (before the frontend
        // registers onCloseRequested) — that would re-open the startup close-race that visible:false closes.
        // Size/position/maximized still restore; the frontend's show() (after the listener) is the sole reveal.
        .plugin(
            tauri_plugin_window_state::Builder::new()
                .with_state_flags(
                    tauri_plugin_window_state::StateFlags::all() & !tauri_plugin_window_state::StateFlags::VISIBLE,
                )
                .build(),
        )
        .setup(move |app| {
            // Big growable files (models + cache) live under the DATA ROOT — default app_dir/data (next
            // to the program, NOT C: AppData; these reach tens of GB). Resolved from config BEFORE
            // deriving the dirs (so a user-set drive takes effect); changing it requires a restart.
            let mut data_dir = commands::settings::resolve_data_dir(&app_dir_early);
            // Legacy fallback: existing installs kept data under AppData. ONLY when the user hasn't set a
            // custom dir (data_dir is still the default app_dir/data) and it has no models yet but the old
            // AppData location does, fall back to AppData so upgraders don't "lose" their models — the
            // settings UI then prompts them to migrate off C:.
            if data_dir == app_dir_early.join("data") && !data_dir.join("models").exists() {
                if let Ok(appdata) = app.path().app_local_data_dir() {
                    if appdata.join("models").exists() {
                        tracing::warn!("Using legacy AppData data dir ({}); migrate via Settings", appdata.display());
                        data_dir = appdata;
                    }
                }
            }
            let cache_dir = data_dir.join("cache");
            let models_dir = data_dir.join("models");
            let _ = std::fs::create_dir_all(&cache_dir);
            let _ = std::fs::create_dir_all(&models_dir);
            // Embedded Python runtime packs live under <data_root>/runtimes (S42) —
            // rooted on the SAME resolved data dir (incl. the legacy AppData fallback)
            // so a user data-dir migration moves the packs' home along with models/cache.
            pyenv::init_runtime_root(&data_dir);
            // Sweep stale on-disk caches on startup (background thread — don't block window show).
            // This is also the crash/power-loss recovery path: a startup sweep always runs, unlike a
            // shutdown hook that wouldn't fire on a hard exit.
            {
                let cd = cache_dir.clone();
                // usp_work (extracted .usp archives) is NOT wiped in-session anymore (a failed open must
                // never destroy the still-open project's media — see open_project_archive). Reclaim it
                // here instead, but ONLY when no autosave recovery is pending: a crash-recovered
                // .usp-opened project's media lives in usp_work and must survive until recovery resolves.
                let recovery_pending = app_dir_early.join("autosave.json").exists();
                std::thread::spawn(move || {
                    if !recovery_pending {
                        let usp_work = cd.join("usp_work");
                        if usp_work.exists() {
                            match std::fs::remove_dir_all(&usp_work) {
                                Ok(()) => tracing::info!("Cache: cleared usp_work (no recovery pending)"),
                                Err(e) => tracing::warn!("Cache: failed to clear usp_work: {}", e),
                            }
                        }
                    }
                    cleanup_cache_on_startup(&cd);
                    // Reclaim torn runtime-pack installs / deferred deletes (and
                    // RECOVER a pack stranded mid-reinstall) — nothing else GCs
                    // <data>/runtimes/.staging (S42 audit).
                    pyenv::sweep_staging();
                });
            }
            // Model avatars render via the asset protocol (asset.localhost). The models dir is
            // user-movable (data-root setting), so the static tauri.conf.json scope can't know it —
            // extend the scope at runtime once the real dir is resolved.
            if let Err(e) = app.asset_protocol_scope().allow_directory(&models_dir, true) {
                tracing::warn!("Failed to add models dir to asset protocol scope: {}", e);
            }
            let state = Arc::new(AppState::new(app_dir_early, cache_dir, models_dir, Arc::clone(&log_buffer)));
            commands::settings::load_and_apply_config(&state);
            // Idle-release sweeper: free GPU sessions (+ the resident ORT CUDA arena) after a stretch of
            // no inference, so VRAM returns to the driver once the user stops working. A new run reloads
            // on demand (reload-on-miss). An in-flight separation keeps refreshing last_activity per chunk,
            // so this never fires mid-job.
            {
                let st = Arc::clone(&state);
                std::thread::spawn(move || loop {
                    std::thread::sleep(std::time::Duration::from_secs(30));
                    st.inference.engine.release_if_idle(std::time::Duration::from_secs(60));
                });
            }
            app.manage(state);

            // System tray — minimize-to-tray + a Show/Quit menu. Menu/click events route to the FRONTEND,
            // which owns the close-flow (in-progress-work + unsaved-changes prompts). Labels follow the UI
            // language via `set_tray_labels` on mount; the English here is just the pre-mount fallback.
            let show_i = tauri::menu::MenuItem::with_id(app, "show", "Show Window", true, None::<&str>)?;
            let quit_i = tauri::menu::MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
            let tray_menu = tauri::menu::Menu::with_items(app, &[&show_i, &quit_i])?;
            tauri::tray::TrayIconBuilder::with_id("main")
                .icon(app.default_window_icon().unwrap().clone())
                .tooltip("UTAI")
                .menu(&tray_menu)
                .show_menu_on_left_click(false)
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "show" => commands::window::show_main(app),
                    "quit" => {
                        commands::window::show_main(app);
                        let _ = app.emit("tray-quit", ());
                    }
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    if let tauri::tray::TrayIconEvent::Click {
                        button: tauri::tray::MouseButton::Left,
                        button_state: tauri::tray::MouseButtonState::Up,
                        ..
                    } = event
                    {
                        commands::window::show_main(tray.app_handle());
                    }
                })
                .build(app)?;

            // Disable WebView2's browser-accelerator keys (Ctrl+F find / Ctrl+P print / F5·Ctrl+R reload /
            // F7 caret / Ctrl+±·0 zoom / Alt+Arrow nav). These are handled by the WebView2 HOST, above the
            // DOM, so a page-level keydown preventDefault can't stop them — only this native setting does.
            // The app's OWN shortcuts (Ctrl+S/O/N/Z/Y + the vocal-editor keys) are plain DOM handlers and
            // stay unaffected. Fail-soft: any COM step erroring just leaves the accelerators enabled.
            #[cfg(windows)]
            if let Some(win) = app.get_webview_window("main") {
                let _ = win.with_webview(|webview| unsafe {
                    use webview2_com::Microsoft::Web::WebView2::Win32::ICoreWebView2Settings3;
                    use windows::core::Interface;
                    if let Ok(core) = webview.controller().CoreWebView2() {
                        if let Ok(settings) = core.Settings() {
                            if let Ok(s3) = settings.cast::<ICoreWebView2Settings3>() {
                                let _ = s3.SetAreBrowserAcceleratorKeysEnabled(false);
                            }
                        }
                    }
                });
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::inference::run_rvc,
            commands::inference::run_sovits,
            commands::inference::render_vocal_segment,
            commands::inference::validate_lyrics,
            commands::inference::cancel_voice,
            commands::inference::detect_f0,
            commands::inference::get_default_vocoder_info,
            commands::import::import_score_file,
            commands::export_score::export_score_files,
            commands::export_audio::export_audio_pcm_begin,
            commands::export_audio::export_audio_pcm_chunk,
            commands::export_audio::export_audio_encode,
            commands::export_audio::export_audio_discard,
            commands::project::save_project_archive,
            commands::project::open_project_archive,
            commands::project::prune_usp_work,
            commands::project::path_exists,
            commands::project::write_autosave,
            commands::project::read_autosave,
            commands::project::clear_autosave,
            commands::training::start_training,
            commands::training::stop_training,
            commands::training::force_stop_training,
            commands::training::get_training_status,
            commands::training::get_training_history,
            commands::training::reset_training_display,
            commands::training::check_training_workspace,
            commands::training::get_training_workspace_info,
            commands::audition::render_model_audition,
            commands::audition::render_candidate_scale,
            commands::audition::set_candidate_vocal_range,
            commands::audition::render_audition_voice,
            commands::audition::render_audition_vocoder,
            commands::audition::render_audition_diffusion,
            commands::audition::audition_active,
            commands::models::list_models,
            commands::models::import_model,
            commands::models::delete_model,
            commands::models::check_model_exists,
            commands::models::set_model_avatar,
            commands::models::set_model_vocal_range,
            commands::models::attach_diffusion,
            commands::audio::load_audio_file,
            commands::audio::probe_audio_duration,
            commands::audio::transpose_audio,
            commands::audio::save_temp_audio,
            commands::audio::ensure_cache_dir,
            commands::audio::save_binary_file,
            commands::audio::analyze_segment_tempo,
            commands::audio::stretch_segment_audio,
            commands::storage::get_storage_report,
            commands::storage::cleanup_render_cache,
            commands::storage::delete_training_workspace,
            commands::storage::cleanup_audition_caches,
            commands::storage::cleanup_logs,
            commands::midi_extract::extract_midi_from_audio,
            commands::midi_extract::cancel_midi_extract,
            commands::midi_extract::midi_extract_status,
            commands::midi_extract::download_game_package,
            commands::midi_extract::delete_game_package,
            commands::separation::run_msst_separation,
            commands::separation::get_separation_status,
            commands::separation::cancel_separation,
            commands::msst_models::get_msst_models_dir,
            commands::msst_models::list_msst_models,
            commands::msst_models::download_msst_model,
            commands::msst_models::delete_msst_model,
            commands::msst_models::import_local_msst_model,
            commands::msst_models::convert_msst_model,
            commands::logs::get_recent_logs,
            commands::logs::get_logs_since,
            commands::logs::log_message,
            commands::logs::get_log_file_path,
            commands::settings::get_hardware_info,
            commands::settings::set_device_preference,
            commands::settings::get_device_preference,
            commands::settings::get_data_dir,
            commands::settings::migrate_data_dir,
            commands::settings::is_cuda_runtime_ready,
            commands::settings::download_cuda_runtime,
            commands::pyenv::get_runtime_env_info,
            commands::pyenv::download_runtime_pack,
            commands::pyenv::install_runtime_pack_local,
            commands::pyenv::cancel_runtime_install,
            commands::pyenv::delete_runtime_pack,
            commands::pyenv::run_pack_envtest,
            commands::pyenv::test_download_source,
            commands::window::quit_app,
            commands::window::running_tasks,
            commands::window::set_tray_labels,
        ])
        // Window close + app exit are driven ENTIRELY by the frontend now (App.tsx onCloseRequested →
        // minimize-to-tray / quit decision → in-progress + unsaved prompts → invoke("quit_app")). No
        // Rust-side close/exit guard: the frontend confirms before quit_app, which exits unconditionally.
        .build(tauri::generate_context!())
        .expect("Failed to build UTAI")
        .run(|_app, _event| {});
}
