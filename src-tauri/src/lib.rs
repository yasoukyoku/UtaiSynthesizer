// the training run.json serde_json::json! literal outgrew the default macro
// recursion limit when the S41 aug_copies key landed (the macro recurses per
// token, not per key — this is a compile-time-only knob)
#![recursion_limit = "256"]

pub mod audio;
pub mod commands;
pub mod crashlog;
pub mod download;
pub mod gpu;
pub mod inference;
pub mod logging;
pub mod models;
pub mod portable;
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

    /// True while at least one task with this id is registered — for PRE-FLIGHT interlocks (S64c:
    /// begin_task is a refcount for the close-flow listing, NOT a mutex; single-flight commands
    /// must check this before registering).
    pub fn task_active(&self, id: &str) -> bool {
        self.active_tasks.lock().contains_key(id)
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

    /// Single-flight + heavy-job interlock for MODEL CONVERSION (S66 user rule): a torch-export
    /// python eats gigabytes of RAM, so conversions never run in parallel with each other NOR
    /// start while another resource-heavy job runs. The exclusion is deliberately one-way for
    /// separation/renders (those may START while a conversion runs); training and the audition
    /// family are excluded BOTH ways (see start_training / audition's ensure_no_convert).
    ///
    /// Atomicity (review S66): convert-vs-convert is compare-and-register under ONE
    /// active_tasks lock (a check-then-begin_task pair leaves a both-pass window — begin_task
    /// is a refcount, NOT a mutex, the S64c rule). The cross-checks run register-then-verify:
    /// the slot is visible BEFORE we verify training/audition/render idle, and those peers
    /// check task_active("convert") before committing — one side always sees the other.
    pub fn acquire_convert_slot(&self) -> std::result::Result<TaskGuard, String> {
        let guard = {
            let mut tasks = self.active_tasks.lock();
            if tasks.contains_key("convert") {
                return Err("CONVERT_BUSY".into());
            }
            *tasks.entry("convert".to_string()).or_insert(0) += 1;
            TaskGuard {
                tasks: Arc::clone(&self.active_tasks),
                id: "convert".to_string(),
            }
        };
        // Register-then-verify: dropping `guard` on a violation unregisters the slot.
        if commands::audition::AUDITION_IN_FLIGHT.load(std::sync::atomic::Ordering::SeqCst) {
            // Generic on purpose: the flag's holder may be an audition, an export or a cleanup.
            return Err(commands::audition::BUSY_RETRY_MSG.into());
        }
        if commands::inference::voice_render_active() {
            return Err("CONVERT_RENDER_BUSY".into());
        }
        if self.training.is_active() {
            return Err("TRAINING_ACTIVE".into());
        }
        if matches!(
            self.separation.status().state,
            separation::SeparationState::LoadingModel | separation::SeparationState::Separating
        ) {
            return Err("SEPARATION_BUSY".into());
        }
        Ok(guard)
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
        // Auto needs the provider's FULL dependency set resolvable, not just cudart (S64c audit: a
        // PARTIAL wheel install would otherwise pick the CUDA build — which has no DirectML provider —
        // and silently drop a DirectML-capable NVIDIA box to CPU every session).
        // S68b: an Auto-mode preferred NON-NVIDIA GPU must load the DirectML build for the
        // same reason — the CUDA build can't honor the pick and Auto would land on CPU.
        _ => {
            let picked_non_nvidia = read_auto_gpu(app_dir)
                .and_then(crate::gpu::adapter_vendor)
                .is_some_and(|v| v != "nvidia");
            !picked_non_nvidia
                && cuda_dll.exists()
                && crate::commands::settings::cuda_provider_deps_resolvable(app_dir)
        }
    };

    let mut search_paths: Vec<std::path::PathBuf> = Vec::new();

    if prefer_cuda {
        if cuda_dll.exists() {
            // PRELOAD CUDA + cuDNN dylibs in the top-level loader context BEFORE ort's init_from loads the
            // CUDA build. Without this, ort's nested load of providers_cuda (under the loader-lock) hangs.
            // VERIFIED with the version-matched 1.24.x build: preload + init_from + CUDA session all succeed.
            let cuda_bin = std::env::var("CUDA_PATH").ok().map(|p| std::path::PathBuf::from(p).join("bin"));
            let cudnn_dir = app_dir.join("runtime").join("cuda");
            // SEPARATE preload calls (S64c audit): preload_dylibs aborts on its FIRST failed load and
            // runs the CUDA loop before the cuDNN loop — one combined call with a wrong-major Toolkit
            // on CUDA_PATH (e.g. 13) would silently skip pinning OUR runtime/cuda copies. Ours load
            // FIRST (first-load-by-basename wins); each lane independent; failures logged, not swallowed.
            if let Err(e) = ort::ep::cuda::preload_dylibs(None, Some(&cudnn_dir)) {
                tracing::warn!("cuDNN preload (runtime/cuda) incomplete: {e}");
            }
            // CUDA core libs from OUR dir — only where the self-contained download placed them
            // (a pre-S64c runtime/cuda holds only cuDNN; attempting would just warn-spam dev boxes).
            if cudnn_dir.join("cudart64_12.dll").exists() {
                if let Err(e) = ort::ep::cuda::preload_dylibs(Some(&cudnn_dir), None) {
                    tracing::warn!("CUDA preload (runtime/cuda) incomplete: {e}");
                }
            }
            if let Some(bin) = cuda_bin.as_deref() {
                if let Err(e) = ort::ep::cuda::preload_dylibs(Some(bin), None) {
                    tracing::warn!("CUDA preload (CUDA_PATH) incomplete: {e}");
                }
            }
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

/// S68b: the Auto-mode preferred GPU (config.json `auto_gpu`), read pre-tauri like
/// read_device_preference — the ORT build pick needs it before AppState exists.
fn read_auto_gpu(app_dir: &std::path::Path) -> Option<u32> {
    let cfg_path = app_dir.join("config.json");
    let content = std::fs::read_to_string(&cfg_path).ok()?;
    let val: serde_json::Value = serde_json::from_str(&content).ok()?;
    val.get("auto_gpu")?.as_u64().map(|v| v as u32)
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

/// Window-state flags shared by the plugin registration AND the pre-update-install flush
/// (commands/update.rs — the updater exits via process::exit, skipping the plugin's own exit-time
/// save; the two sites must never drift, S64 audit).
pub fn window_state_flags() -> tauri_plugin_window_state::StateFlags {
    tauri_plugin_window_state::StateFlags::all() & !tauri_plugin_window_state::StateFlags::VISIBLE
}

/// App root = the directory holding `converter/`. Resolved from the EXECUTABLE's location first
/// (stable regardless of launch context: a release exe sits next to converter/, a dev exe in
/// src-tauri/target/debug walks up to the repo root), with the old CWD probe as fallback. The
/// final fallback is the exe dir — NOT the CWD — so the data/models root can never silently move
/// with whatever directory the app happened to be launched from.
fn resolve_app_dir() -> std::path::PathBuf {
    let has_converter = |d: &std::path::Path| d.join("converter").join("convert.py").exists();

    // Dev builds pin to the repo root (compile-time known). S64: bundle.resources now copies
    // converter/convert.py NEXT TO THE DEBUG EXE on `tauri dev` — without this pin, the walk below
    // would hit target/debug first and silently move the dev data root there (the exact
    // "data dir drift" class the resolver exists to prevent).
    #[cfg(debug_assertions)]
    {
        if let Some(root) = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent() {
            if has_converter(root) {
                return root.to_path_buf();
            }
        }
    }

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

/// S68e: `<exe dir>\webview` as the WebView2 user data folder (WebView2 itself appends
/// an `EBWebView` subdir inside it — the legacy default did the same under the
/// identifier dir). ONE-TIME migration: when the merged dir doesn't exist yet and the
/// legacy profile does, copy it over (pid-suffixed staging dir + rename — two racing
/// first-boot instances each stage privately and the loser just cleans up; the profile
/// holds localStorage — UI language + toggles — that must survive the move; ~170 MB, a
/// one-off few seconds). EVERY failure returns None = the webview stays on the LEGACY
/// default profile (nothing lost, retried next boot) — the merged dir is only created
/// once the migration (or a fresh install with no legacy profile) succeeds, and it is
/// writability-probed before use: a read-only dir handed to WebView2 fails environment
/// creation, which would otherwise take the whole window (and boot) down (review S68e).
#[cfg(not(debug_assertions))]
fn webview_data_dir() -> Option<std::path::PathBuf> {
    let dir = std::env::current_exe().ok()?.parent()?.join("webview");
    if !dir.exists() {
        if let Ok(local) = std::env::var("LOCALAPPDATA") {
            let legacy = std::path::PathBuf::from(local)
                .join("com.utaisynthesizer.app")
                .join("EBWebView");
            if legacy.is_dir() {
                // Disk preflight (S68d posture): stay on the legacy profile rather
                // than half-copy on a tight volume; retried once space frees up.
                let needed =
                    commands::settings::migrate_tree_needed(&legacy, &dir.join("EBWebView"), false);
                if let Some(free) = util::free_bytes_at(dir.parent()?) {
                    if free < needed.saturating_mul(2) {
                        tracing::warn!(
                            "webview profile migration deferred: {} MB needed, {} MB free",
                            needed / 1_000_000,
                            free / 1_000_000
                        );
                        return None;
                    }
                }
                let staging = dir.with_extension(format!("staging{}", std::process::id()));
                let _ = std::fs::remove_dir_all(&staging);
                let copied =
                    commands::settings::copy_dir_all(&legacy, &staging.join("EBWebView"), false);
                match copied {
                    Ok(()) => {
                        if std::fs::rename(&staging, &dir).is_ok() {
                            // Baseline for the S68e.1 reclaim veto. A FILE'S CONTENT —
                            // not filesystem timestamps: copying/carrying the install
                            // folder refreshes CreationTime/mtimes wholesale (review
                            // S68e.1), which would silently reset a metadata-based
                            // baseline and un-veto a live dev profile.
                            stamp_webview_merge_marker(&dir);
                        } else {
                            let _ = std::fs::remove_dir_all(&staging);
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            "webview profile migration failed ({e}) — staying on the legacy profile (retried next boot)"
                        );
                        let _ = std::fs::remove_dir_all(&staging);
                        return None;
                    }
                }
                if !dir.exists() {
                    // Lost the staging race AND the winner hasn't landed either —
                    // stay on the legacy profile this boot.
                    return None;
                }
            }
        }
    } else {
        // Any boot AFTER the migration boot: the merged profile is authoritative,
        // the legacy copy is ~170 MB of dead weight — reclaim it (S68e.1, §user:
        // "why is com.utaisynthesizer.app still there").
        reclaim_legacy_webview_profile(&dir);
    }
    std::fs::create_dir_all(&dir).ok()?;
    let probe = dir.join(".writable");
    std::fs::write(&probe, b"").ok()?;
    let _ = std::fs::remove_file(&probe);
    Some(dir)
}

/// The S68e.1 reclaim-veto baseline, stamped INSIDE the merged webview dir at
/// migration time. Epoch seconds as file CONTENT — content survives folder copies,
/// while CreationTime/mtimes get refreshed by exactly the carry-the-folder moves this
/// feature exists for.
#[cfg(not(debug_assertions))]
const WEBVIEW_MERGE_MARKER: &str = ".merged-at";

#[cfg(not(debug_assertions))]
fn stamp_webview_merge_marker(dir: &std::path::Path) {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let _ = std::fs::write(dir.join(WEBVIEW_MERGE_MARKER), secs.to_string());
}

/// S68e.1: delete the legacy identifier-dir WebView2 profile once the MERGED profile
/// exists (any boot after the migration boot), then drop the identifier dir itself if
/// that emptied it. Guards, in order:
///   1. A LIVE pre-merge instance (its sentinel sits in the legacy LOGS dir) vetoes
///      the whole reclaim: its profile is only partially lock-protected — a
///      remove_dir_all would tear out closed files (leveldb tables, Preferences)
///      before the first locked one stops it, corrupting the live session.
///   2. Baseline = the .merged-at marker's CONTENT (missing marker — 0.8.0-migrated
///      installs — is stamped NOW and judged from the next boot on).
///   3. Anything under the legacy profile touched AFTER the baseline = something
///      (a dev build, an occasionally-run old copy) still uses it → keep, forever
///      (every Chromium session bumps Local State/Preferences via atomic same-dir
///      saves, so real use is always visible on the probes). Fail-safe = keep.
#[cfg(not(debug_assertions))]
fn reclaim_legacy_webview_profile(merged: &std::path::Path) {
    let Ok(local) = std::env::var("LOCALAPPDATA") else { return };
    let root = std::path::PathBuf::from(local).join("com.utaisynthesizer.app");
    let legacy = root.join("EBWebView");
    if !legacy.is_dir() {
        return;
    }
    if logging::foreign_live_session_in(&root.join("logs")) {
        return; // a pre-merge copy is running right now — its profile is live
    }
    let marker = merged.join(WEBVIEW_MERGE_MARKER);
    let baseline = std::fs::read_to_string(&marker)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|secs| std::time::UNIX_EPOCH + std::time::Duration::from_secs(secs));
    let Some(baseline) = baseline else {
        // Pre-marker merge (0.8.0) or a wiped marker: stamp now, judge next boot.
        stamp_webview_merge_marker(merged);
        return;
    };
    // Newest signal of post-baseline use, from the dirs Chromium's atomic saves touch.
    let touched_after = [
        legacy.clone(),
        legacy.join("Default"),
        legacy.join("Default").join("Local Storage"),
        legacy.join("Default").join("Local Storage").join("leveldb"),
    ]
    .iter()
    .filter_map(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok())
    .any(|m| m > baseline);
    if touched_after {
        return; // someone (a dev build / an old copy in occasional use) still uses it
    }
    match std::fs::remove_dir_all(&legacy) {
        Ok(()) => {
            let _ = std::fs::remove_dir(&root); // only succeeds once fully empty
            tracing::info!("legacy webview profile reclaimed ({})", legacy.display());
        }
        Err(e) => tracing::warn!("legacy webview profile not reclaimed ({e}) — retried next boot"),
    }
}

/// Dev builds keep the tauri default profile (identifier dir) — the dev app root is
/// the repo, and a repo-local webview/ would pollute the working tree.
#[cfg(debug_assertions)]
fn webview_data_dir() -> Option<std::path::PathBuf> {
    None
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Observed: Auto loading the CUDA build hung on a missing cudart/cublas without this —
    // fail fast so init_ort_runtime falls back to the DirectML build instead of hanging.
    suppress_windows_dll_error_dialogs();
    let log_dir = logging::get_log_dir();
    let _ = std::fs::create_dir_all(&log_dir);
    // S68e: move the legacy identifier-dir logs (incl. crash sentinels) into the
    // merged home BEFORE the appender opens today's file — a same-day upgrade then
    // keeps appending to the migrated file instead of forking a second one.
    logging::migrate_legacy_logs(&log_dir);

    // S67: the offset is captured once and drives BOTH the per-line timestamps and
    // the daily file roll (LocalDailyFile — tracing-appender's own rolling::daily is
    // hardwired to UTC dates/boundaries). UTC is the honest fallback.
    let tz_offset = time::UtcOffset::current_local_offset().unwrap_or(time::UtcOffset::UTC);
    let mut file_appender =
        logging::LocalDailyFile::new(log_dir.clone(), logging::LOG_PREFIX, tz_offset);
    // S67c: process-start divider, FILE ONLY. Same-day launches APPEND to one file, so a
    // crashed process's last line and the next launch's first line were visually
    // indistinguishable (the 07-16 community log interleaves six runs across three app
    // versions). A raw write here — BEFORE the non_blocking wrap and registry init — is
    // single-threaded (ownership then moves into NonBlocking: later direct access cannot
    // even compile), lands ahead of the banner, skips the lossy channel, and is
    // structurally invisible to the UI panel (BufferLayer only sees tracing events, and
    // the panel pipeline must never change — S67).
    {
        use std::io::Write;
        let _ = writeln!(
            file_appender,
            "\n======================== process start: UtaiSynthesizer {} (pid {}) ========================",
            env!("CARGO_PKG_VERSION"),
            std::process::id()
        );
    }
    let (non_blocking, file_guard) = tracing_appender::non_blocking(file_appender);
    // S68b: park the worker guard where quit paths can drop it (= drain the lossy
    // channel). app.exit(0) never runs Drop, so held-in-run() it was dead code.
    *crashlog::LOG_GUARD.lock() = Some(file_guard);

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

    // S67: file/stdout timestamps in LOCAL time with the UTC offset printed on every
    // line — users could never match the log PANEL's local times against the file's
    // old UTC-Z lines. Fixed 6-digit micros keep the old line width. S68b: the format
    // string moved to logging::LINE_TIME_FORMAT (shared with the panic hook's raw
    // append). NOTE: the panel's own timestamps (logging.rs BufferLayer, GetLocalTime +
    // lexicographic `since`) are untouched. File names/roll boundaries follow the same
    // offset via LocalDailyFile above.
    let timer = tracing_subscriber::fmt::time::OffsetTime::new(tz_offset, logging::LINE_TIME_FORMAT);

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .with_timer(timer.clone())
                .with_filter(env_filter),
        )
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(non_blocking)
                .with_ansi(false)
                .with_timer(timer)
                .with_filter(file_filter),
        )
        .with(logging::BufferLayer::new(Arc::clone(&log_buffer)))
        .init();

    // UtcOffset's Display prints "+08:00:00" (seconds included) — format hh:mm ourselves
    // so the banner matches the per-line "(UTC+08:00)" style.
    let offset_label = format!(
        "{}{:02}:{:02}",
        if tz_offset.is_negative() { '-' } else { '+' },
        tz_offset.whole_hours().abs(),
        tz_offset.minutes_past_hour().abs()
    );
    tracing::info!(
        "UtaiSynthesizer {} starting (pid {}) — logs: {} (timestamps local, UTC{})",
        env!("CARGO_PKG_VERSION"),
        std::process::id(),
        log_dir.display(),
        offset_label
    );

    // S68b crash forensics: panics reach the log file even under panic=abort (release
    // has no hook otherwise — a Rust panic died stderr-only, indistinguishable from an
    // OS kill), and an unclean previous exit triggers a Windows-Event-Log autopsy
    // (Application Error 1000 / display-reset 4101 disambiguate the silent-death
    // classes our own log structurally cannot).
    crashlog::install_panic_hook(log_dir.clone(), tz_offset);
    crashlog::spawn_autopsy(crashlog::rotate_session_sentinel(&log_dir));

    // S68d fullportable: if this installed copy was MOVED, point the installer's
    // registry memory (and the Add/Remove entry) at where it actually lives — manual
    // installer runs then land here instead of resurrecting on C:. Heavily guarded
    // (dead-location proof etc.) — see portable.rs; warn-only. On a BACKGROUND thread:
    // the dead-location probe stats the recorded dir, and a stale record pointing at
    // an offline network share would otherwise stall boot for the SMB connect timeout
    // (review S68d). Nothing at boot depends on these registry values.
    std::thread::spawn(portable::heal_install_registry);

    let app_dir_early = resolve_app_dir();
    setup_cuda_dll_paths(&app_dir_early);
    init_ort_runtime(&app_dir_early);

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        // Exclude VISIBLE from restore so the plugin doesn't show() the window EARLY (before the frontend
        // registers onCloseRequested) — that would re-open the startup close-race that visible:false closes.
        // Size/position/maximized still restore; the frontend's show() (after the listener) is the sole reveal.
        .plugin(
            tauri_plugin_window_state::Builder::new()
                .with_state_flags(window_state_flags())
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
            // S68c: a completed data-dir migration leaves the OLD tree marked for reclaim — finish
            // it now, on the first boot that runs on the NEW root (this process holds zero handles
            // into the old tree at this point; the worker delta-syncs stragglers before deleting).
            commands::settings::spawn_pending_data_dir_delete(app_dir_early.clone(), data_dir.clone());
            let cache_dir = data_dir.join("cache");
            let models_dir = data_dir.join("models");
            let _ = std::fs::create_dir_all(&cache_dir);
            let _ = std::fs::create_dir_all(&models_dir);
            // S64b migration: the shared-model dir used to be `models/aux` — a reserved Windows
            // device name most systems can't even create (beta testers: os error 267/1200); only
            // relaxed Win11 builds (the dev machine) ever had one. Rename it where it exists.
            {
                let old_aux = models_dir.join("aux");
                let new_aux = models_dir.join(models::AUX_DIR_NAME);
                if old_aux.is_dir() && !new_aux.exists() {
                    match std::fs::rename(&old_aux, &new_aux) {
                        Ok(()) => tracing::info!("migrated models/aux -> models/{}", models::AUX_DIR_NAME),
                        Err(e) => tracing::warn!("models/aux migration failed: {e}"),
                    }
                }
            }
            // S64b cleanup: pre-0.1.2 CUDA downloads copied the four CUDA ORT DLLs next to the exe
            // (a dev-only convenience that leaked into release, polluting the install root AND
            // risking shadowing). The set is only ever OURS when providers_cuda sits there too.
            #[cfg(not(debug_assertions))]
            if let Ok(exe) = std::env::current_exe() {
                if let Some(exe_dir) = exe.parent() {
                    if exe_dir.join("onnxruntime_providers_cuda.dll").exists() {
                        for n in [
                            "onnxruntime.dll",
                            "onnxruntime_providers_cuda.dll",
                            "onnxruntime_providers_shared.dll",
                            "onnxruntime_providers_tensorrt.dll",
                        ] {
                            let _ = std::fs::remove_file(exe_dir.join(n));
                        }
                        tracing::info!("removed stray CUDA ORT DLLs from the install root (S64b)");
                    }
                }
            }
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

            // S68e fullportable: the main window is created IN CODE (tauri.conf.json no
            // longer declares windows[]) so the WebView2 profile — localStorage: UI
            // language + toggles; previously %LOCALAPPDATA%\com.utaisynthesizer.app\
            // EBWebView — can live inside the install folder (<exe>\webview). The S64
            // startup close-race contract is preserved VERBATIM and needs all three
            // legs: visible(false) here + the frontend's show()-after-close-listener +
            // the window-state plugin registered with !VISIBLE flags (see
            // project_v2_tray_close_flow) — do not touch any of them in isolation.
            // Property values transcribed 1:1 from the removed conf entry.
            {
                let build_main = |data_dir: Option<std::path::PathBuf>| {
                    let wb =
                        tauri::WebviewWindowBuilder::new(app, "main", tauri::WebviewUrl::default())
                            .title("UtaiSynthesizer")
                            .inner_size(1400.0, 900.0)
                            .min_inner_size(1024.0, 700.0)
                            .resizable(true)
                            .decorations(true)
                            .visible(false);
                    let wb = match data_dir {
                        Some(dir) => wb.data_directory(dir),
                        None => wb,
                    };
                    wb.build()
                };
                match webview_data_dir() {
                    Some(dir) => {
                        if let Err(e) = build_main(Some(dir)) {
                            // Last resort (review S68e): a poisoned merged profile dir
                            // must never keep the app from booting — fall back to the
                            // default profile location and try once more.
                            tracing::warn!(
                                "window build with merged webview profile failed ({e}) — retrying with the default profile"
                            );
                            build_main(None)?;
                        }
                    }
                    None => {
                        build_main(None)?;
                    }
                }
            }

            // System tray — minimize-to-tray + a Show/Quit menu. Menu/click events route to the FRONTEND,
            // which owns the close-flow (in-progress-work + unsaved-changes prompts). Labels follow the UI
            // language via `set_tray_labels` on mount; the English here is just the pre-mount fallback.
            let show_i = tauri::menu::MenuItem::with_id(app, "show", "Show Window", true, None::<&str>)?;
            let quit_i = tauri::menu::MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
            let tray_menu = tauri::menu::Menu::with_items(app, &[&show_i, &quit_i])?;
            tauri::tray::TrayIconBuilder::with_id("main")
                .icon(app.default_window_icon().unwrap().clone())
                .tooltip("UtaiSynthesizer")
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
            commands::inference::run_autotune,
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
            commands::audition::get_candidate_vocal_range,
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
            commands::logs::open_log_dir,
            commands::settings::get_hardware_info,
            commands::settings::list_inference_gpus,
            commands::settings::set_device_preference,
            commands::settings::get_device_preference,
            commands::settings::get_data_dir,
            commands::settings::get_data_dir_issue,
            commands::settings::migrate_data_dir,
            commands::settings::is_cuda_runtime_ready,
            commands::settings::download_cuda_runtime,
            commands::settings::cancel_cuda_download,
            commands::settings::get_cuda_mem_limit,
            commands::settings::set_cuda_mem_limit,
            commands::settings::install_cuda_runtime_local,
            commands::settings::cuda_runtime_paths,
            commands::settings::fetch_mirror_list,
            commands::assets::asset_pack_status,
            commands::assets::download_asset_pack,
            commands::assets::cancel_asset_pack_download,
            commands::pyenv::get_runtime_env_info,
            commands::pyenv::training_env_ready,
            commands::pyenv::converter_env_ready,
            commands::training::training_required_assets,
            commands::pyenv::download_runtime_pack,
            commands::pyenv::install_runtime_pack_local,
            commands::pyenv::cancel_runtime_install,
            commands::pyenv::delete_runtime_pack,
            commands::pyenv::run_pack_envtest,
            commands::pyenv::test_download_source,
            commands::update::update_check,
            commands::update::update_install,
            commands::update::update_cancel,
            commands::settings::bundled_integrity_report,
            commands::settings::migrate_pending_restart,
            commands::window::quit_app,
            commands::window::restart_app,
            commands::window::running_tasks,
            commands::window::set_tray_labels,
        ])
        // Window close + app exit are driven ENTIRELY by the frontend now (App.tsx onCloseRequested →
        // minimize-to-tray / quit decision → in-progress + unsaved prompts → invoke("quit_app")). No
        // Rust-side close/exit guard: the frontend confirms before quit_app, which exits unconditionally.
        .build(tauri::generate_context!())
        .expect("Failed to build UtaiSynthesizer")
        .run(|_app, _event| {});
}
