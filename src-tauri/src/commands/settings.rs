use std::sync::Arc;
use tauri::{Emitter, State};

use crate::inference::engine::DeviceConfig;
use crate::AppState;

#[derive(serde::Serialize)]
pub struct HardwareInfo {
    pub gpu_name: String,
    pub cuda_available: bool,
    pub directml_available: bool,
    pub current_device: String,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct AppConfig {
    #[serde(default)]
    pub device: DeviceConfig,
    /// User-set data root for the BIG growable files (models + cache). Empty/None → app_dir/data (next
    /// to the program, NOT C: AppData — those files reach tens of GB). See `resolve_data_dir`.
    #[serde(default)]
    pub data_dir: Option<String>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            device: DeviceConfig::Auto,
            data_dir: None,
        }
    }
}

#[tauri::command]
pub fn get_hardware_info(state: State<'_, Arc<AppState>>) -> Result<HardwareInfo, String> {
    let current = state.inference.engine.device();
    let current_str = match &current {
        DeviceConfig::Cpu => "cpu".to_string(),
        DeviceConfig::DirectMl { .. } => "directml".to_string(),
        DeviceConfig::Cuda { .. } => "cuda".to_string(),
        DeviceConfig::Auto => "auto".to_string(),
    };

    Ok(HardwareInfo {
        gpu_name: detect_gpu_name(),
        cuda_available: is_cuda_available(),
        directml_available: cfg!(windows),
        current_device: current_str,
    })
}

#[tauri::command]
pub fn set_device_preference(
    state: State<'_, Arc<AppState>>,
    device: String,
) -> Result<(), String> {
    let config = match device.as_str() {
        "cuda" => DeviceConfig::Cuda { device_id: 0 },
        "directml" => DeviceConfig::DirectMl { device_id: 0 },
        "cpu" => DeviceConfig::Cpu,
        _ => DeviceConfig::Auto,
    };

    state.inference.engine.set_device(config.clone());

    // Persist — load-then-update so we never clobber the rest of the config (esp. data_dir).
    let mut cfg = load_config(&state.app_dir).unwrap_or_default();
    cfg.device = config;
    if let Err(e) = save_config(&state.app_dir, &cfg) {
        tracing::warn!("Failed to save config: {}", e);
    }

    Ok(())
}

#[tauri::command]
pub fn get_device_preference(state: State<'_, Arc<AppState>>) -> Result<String, String> {
    let current = state.inference.engine.device();
    Ok(match current {
        DeviceConfig::Cpu => "cpu".to_string(),
        DeviceConfig::DirectMl { .. } => "directml".to_string(),
        DeviceConfig::Cuda { .. } => "cuda".to_string(),
        DeviceConfig::Auto => "auto".to_string(),
    })
}

pub fn load_and_apply_config(state: &AppState) {
    if let Some(cfg) = load_config(&state.app_dir) {
        tracing::info!("Loaded device preference: {:?}", cfg.device);
        state.inference.engine.set_device(cfg.device);
    } else {
        tracing::info!("No config found, using Auto (CUDA → DirectML → CPU)");
    }
}

fn config_path(app_dir: &std::path::Path) -> std::path::PathBuf {
    app_dir.join("config.json")
}

fn save_config(app_dir: &std::path::Path, cfg: &AppConfig) -> std::io::Result<()> {
    let path = config_path(app_dir);
    let json = serde_json::to_string_pretty(cfg).unwrap_or_default();
    // Temp + rename so a crash mid-write can't truncate config.json (losing device pref + data_dir).
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, json)?;
    std::fs::rename(&tmp, &path)
}

fn load_config(app_dir: &std::path::Path) -> Option<AppConfig> {
    let path = config_path(app_dir);
    let content = std::fs::read_to_string(path).ok()?;
    match serde_json::from_str(&content) {
        Ok(cfg) => Some(cfg),
        Err(e) => {
            // A corrupt config silently falling back to defaults would look like lost settings.
            tracing::warn!("config.json exists but failed to parse ({}); using defaults", e);
            None
        }
    }
}

/// Data root for the big growable files (models + cache). User-set in config.json's `data_dir`; else
/// `app_dir/data` — NEXT TO THE PROGRAM, never C: AppData (those files reach tens of GB). Derived at
/// startup; changing it takes effect on restart.
pub fn resolve_data_dir(app_dir: &std::path::Path) -> std::path::PathBuf {
    if let Some(cfg) = load_config(app_dir) {
        if let Some(d) = cfg.data_dir {
            let d = d.trim();
            if !d.is_empty() {
                return std::path::PathBuf::from(d);
            }
        }
    }
    app_dir.join("data")
}

/// The data root ACTUALLY in use this session — parent of cache_dir (cache_dir = data_root/cache,
/// models = data_root/models). May differ from `resolve_data_dir`: startup can pick the legacy
/// AppData fallback for upgraders (see lib.rs setup).
fn effective_data_root(state: &AppState) -> &std::path::Path {
    state.cache_dir.parent().unwrap_or(state.cache_dir.as_path())
}

/// Current data dir (for the settings UI).
#[tauri::command]
pub fn get_data_dir(state: State<'_, Arc<AppState>>) -> Result<String, String> {
    Ok(effective_data_root(&state).to_string_lossy().to_string())
}

/// Recursively copy a directory's contents into `dst` (creating it). Cross-drive safe (copy, not rename).
fn copy_dir_all(src: &std::path::Path, dst: &std::path::Path) -> std::io::Result<()> {
    if !src.exists() {
        return Ok(());
    }
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if from.is_dir() {
            copy_dir_all(&from, &to)?;
        } else {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

/// One-click migrate: copy the CURRENT models + cache into `new_dir`, then persist it as the data dir.
/// Takes effect on restart. Old data is LEFT in place (rollback = revert the setting / don't restart);
/// the user deletes the old copy manually once the new location is confirmed working.
#[tauri::command]
pub async fn migrate_data_dir(state: State<'_, Arc<AppState>>, new_dir: String) -> Result<(), String> {
    let new = std::path::PathBuf::from(new_dir.trim());
    if new.as_os_str().is_empty() {
        return Err("Empty target directory".into());
    }
    let data_root = effective_data_root(&state).to_path_buf();
    let target = new.clone();
    // The copy reaches tens of GB — run it off the event loop so the UI stays responsive.
    tauri::async_runtime::spawn_blocking(move || -> Result<(), String> {
        std::fs::create_dir_all(&target).map_err(|e| format!("Create target: {e}"))?;
        // Refuse a target nested inside the data root (or vice versa) — copying a tree into itself
        // recurses forever.
        let canon_target = std::fs::canonicalize(&target).map_err(|e| format!("Resolve target: {e}"))?;
        let canon_root = std::fs::canonicalize(&data_root).unwrap_or_else(|_| data_root.clone());
        if canon_target.starts_with(&canon_root) || canon_root.starts_with(&canon_target) {
            return Err("Target directory overlaps the current data directory".into());
        }
        copy_dir_all(&data_root.join("models"), &target.join("models")).map_err(|e| format!("Copy models: {e}"))?;
        copy_dir_all(&data_root.join("cache"), &target.join("cache")).map_err(|e| format!("Copy cache: {e}"))
    })
    .await
    .map_err(|e| format!("Copy task failed: {e}"))??;
    let mut cfg = load_config(&state.app_dir).unwrap_or_default();
    cfg.data_dir = Some(new.to_string_lossy().to_string());
    save_config(&state.app_dir, &cfg).map_err(|e| format!("Save config: {e}"))?;
    tracing::info!("Migrated data dir → {} (restart to apply)", new.display());
    Ok(())
}

fn detect_gpu_name() -> String {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let output = std::process::Command::new("powershell")
            .args(["-NoProfile", "-Command",
                   "(Get-CimInstance -ClassName Win32_VideoController).Name -join ', '"])
            .creation_flags(crate::util::CREATE_NO_WINDOW)
            .output();
        if let Ok(out) = output {
            let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !text.is_empty() {
                return text;
            }
        }
    }
    "Unknown GPU".to_string()
}

/// Whether CUDA is ACTUALLY usable, not just "files downloaded". Verifies that the CUDA ORT build is
/// present AND that the CUDA major it was built for (read from providers_cuda.dll's imports) matches a
/// cudart + cuDNN actually resolvable on this machine. This is what stops the old false "Ready" when a
/// CUDA-11-built ORT (1.21.x) sat on a CUDA-12 system — it now correctly reports NOT ready.
#[tauri::command]
pub fn is_cuda_runtime_ready(state: State<'_, Arc<AppState>>) -> Result<bool, String> {
    let cuda_dir = state.app_dir.join("runtime").join("ort").join("cuda");
    let ort_cuda_dll = cuda_dir.join("onnxruntime.dll");
    let providers = cuda_dir.join("onnxruntime_providers_cuda.dll");
    if !ort_cuda_dll.exists() || !providers.exists() {
        return Ok(false);
    }
    // Which CUDA major does this build actually need? (1.21.x wrongly needs 11 → unusable on a 12 box.)
    let major = cuda_build_major(&providers).unwrap_or(0);
    if major < 12 {
        return Ok(false); // CUDA 11 build (or unreadable) — treat as not ready
    }
    let cudnn_dir = state.app_dir.join("runtime").join("cuda");
    let cudnn_ok = dll_on_path_or_dir("cudnn64_9.dll", &cudnn_dir);
    let cudart_ok = dll_on_path("cudart64_12.dll");
    Ok(cudnn_ok && cudart_ok)
}

/// Scan a providers_cuda.dll for its imported `cudart64_NNN.dll` string to learn the CUDA MAJOR it was
/// built against (110 → 11, 12 → 12, 118 → 11). Reads the whole DLL once; fine for an on-demand check.
fn cuda_build_major(providers_cuda: &std::path::Path) -> Option<u32> {
    use std::collections::HashMap;
    use std::sync::Mutex;
    // Cache keyed by (path, mtime, len) so repeated Settings opens don't re-read the DLL, while a
    // re-download replacing it in-session (do_download_cuda_runtime) is picked up without a restart.
    type CacheKey = (std::path::PathBuf, Option<std::time::SystemTime>, u64);
    static CACHE: Mutex<Option<HashMap<CacheKey, Option<u32>>>> = Mutex::new(None);
    let meta = std::fs::metadata(providers_cuda).ok();
    let key: CacheKey = (
        providers_cuda.to_path_buf(),
        meta.as_ref().and_then(|m| m.modified().ok()),
        meta.as_ref().map(|m| m.len()).unwrap_or(0),
    );
    if let Some(m) = CACHE.lock().unwrap().as_ref() {
        if let Some(v) = m.get(&key) {
            return *v;
        }
    }
    let result = scan_cuda_major(providers_cuda);
    CACHE
        .lock()
        .unwrap()
        .get_or_insert_with(HashMap::new)
        .insert(key, result);
    result
}

fn scan_cuda_major(providers_cuda: &std::path::Path) -> Option<u32> {
    use std::io::Read;
    // The "cudart64_NNN.dll" import string lives near the PE header, not in the hundreds-of-MB CUDA
    // kernel blob — read only the first 64MB instead of slurping the whole DLL into RAM.
    let mut data = Vec::new();
    std::fs::File::open(providers_cuda)
        .ok()?
        .take(64 * 1024 * 1024)
        .read_to_end(&mut data)
        .ok()?;
    let needle = b"cudart64_";
    let mut i = 0usize;
    while i + needle.len() + 1 < data.len() {
        if &data[i..i + needle.len()] == needle {
            let mut j = i + needle.len();
            let mut digits = String::new();
            while j < data.len() && data[j].is_ascii_digit() && digits.len() < 4 {
                digits.push(data[j] as char);
                j += 1;
            }
            if let Ok(n) = digits.parse::<u32>() {
                return Some(if n >= 100 { n / 10 } else { n });
            }
        }
        i += 1;
    }
    None
}

/// True if `name` is found on PATH or in the system CUDA Toolkit bin (CUDA_PATH may not be on PATH here).
fn dll_on_path(name: &str) -> bool {
    if let Ok(path) = std::env::var("PATH") {
        if std::env::split_paths(&path).any(|d| d.join(name).exists()) {
            return true;
        }
    }
    if let Ok(cuda) = std::env::var("CUDA_PATH") {
        if std::path::Path::new(&cuda).join("bin").join(name).exists() {
            return true;
        }
    }
    false
}

fn dll_on_path_or_dir(name: &str, extra: &std::path::Path) -> bool {
    extra.join(name).exists() || dll_on_path(name)
}

/// Download CUDA ORT DLLs + cuDNN DLLs for CUDA EP support.
/// Emits `cuda-download-progress` events with {stage, progress, message}.
#[tauri::command]
pub async fn download_cuda_runtime(
    app_handle: tauri::AppHandle,
    state: State<'_, Arc<AppState>>,
) -> Result<(), String> {
    let _task = state.begin_task("cuda_download"); // listed in the close-flow's in-progress warning
    let app_dir = state.app_dir.clone();
    let handle = app_handle.clone();

    let result = tokio::task::spawn_blocking(move || {
        let rt = tokio::runtime::Handle::current();
        rt.block_on(async {
            do_download_cuda_runtime(&app_dir, &handle).await
        })
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))?;

    // Surface the outcome into the tracing pipeline (log panel + file) — a failed download used to be
    // invisible there (only shown under the button), which is exactly what the user hit.
    match &result {
        Ok(()) => tracing::info!("CUDA runtime download complete"),
        Err(e) => tracing::error!("CUDA runtime download failed: {}", e),
    }
    result.map_err(|e| e.to_string())
}

async fn do_download_cuda_runtime(
    app_dir: &std::path::Path,
    handle: &tauri::AppHandle,
) -> crate::Result<()> {

    let emit = |stage: &str, progress: f32, msg: &str| {
        let _ = handle.emit("cuda-download-progress", serde_json::json!({
            "stage": stage, "progress": progress, "message": msg,
        }));
    };

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .map_err(|e| crate::UtaiError::Audio(format!("HTTP client: {}", e)))?;

    // ── Stage 1: Download CUDA ORT DLLs from NuGet ──
    emit("ort", 0.0, "Downloading CUDA ORT runtime...");
    let ort_cuda_dir = app_dir.join("runtime").join("ort").join("cuda");
    // Wipe any previous (possibly wrong-CUDA) DLLs first so a re-download REPLACES them cleanly.
    let _ = std::fs::remove_dir_all(&ort_cuda_dir);
    std::fs::create_dir_all(&ort_cuda_dir)?;

    // 1.24.4 MUST match the ORT API version the `ort` crate (2.0-rc.12) targets — API 24 — AND the
    // bundled DirectML build (1.24.4). A mismatched CUDA build (e.g. 1.20.1 = API 20) makes ort's
    // init_from of the CUDA build DEADLOCK (ort calls API-24 ABI against an API-20 DLL). 1.24.4's
    // providers_cuda imports cudart64_12 / cublas64_12 / cudnn64_9 (correct CUDA-12 + cuDNN-9).
    // AVOID 1.21.x (mis-built against CUDA 11). Gpu.Windows has the actual DLLs.
    let ort_url = "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.Gpu.Windows/1.24.4";
    let ort_nupkg = app_dir.join("runtime").join("ort_gpu.nupkg.zip");

    download_file(&client, ort_url, &ort_nupkg, |p| emit("ort", p * 0.4, "Downloading CUDA ORT...")).await?;
    emit("ort", 0.4, "Extracting CUDA ORT DLLs...");

    crate::util::extract_zip_dlls(&ort_nupkg, &ort_cuda_dir, |n| n.starts_with("runtimes/win-x64/native"))?;
    let _ = std::fs::remove_file(&ort_nupkg);

    // ── Stage 2: cuDNN — SKIP if already present. The user may already have it (runtime/cuda is kept
    //    across re-downloads), and re-fetching from PyPI can fail on a flaky/blocked network (e.g. CN)
    //    even though Stage 1 + the existing cuDNN are all that's needed. Don't fail the whole install. ──
    let cuda_dir = app_dir.join("runtime").join("cuda");
    std::fs::create_dir_all(&cuda_dir)?;
    if cuda_dir.join("cudnn64_9.dll").exists() {
        emit("cudnn", 0.85, "cuDNN already present — skipping");
        tracing::info!("CUDA download: cuDNN already present, skipping re-download");
    } else {
        emit("cudnn", 0.5, "Downloading cuDNN...");
        let cudnn_url = "https://files.pythonhosted.org/packages/f2/a4/045f8d0ce6b99726d88e76bbb8ee147123f55e80111d89262762d8149abb/nvidia_cudnn_cu12-9.22.0.52-py3-none-win_amd64.whl";
        let cudnn_whl = app_dir.join("runtime").join("cudnn.whl.zip");
        download_file(&client, cudnn_url, &cudnn_whl, |p| emit("cudnn", 0.5 + p * 0.35, "Downloading cuDNN...")).await?;
        emit("cudnn", 0.85, "Extracting cuDNN DLLs...");
        crate::util::extract_zip_dlls(&cudnn_whl, &cuda_dir, |n| n.contains("nvidia/cudnn/bin"))?;
        let _ = std::fs::remove_file(&cudnn_whl);
    }

    // ── Stage 3: Also copy to exe dir for dev mode ──
    emit("copy", 0.95, "Finalizing...");
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let target_debug = exe_dir;
            // Copy CUDA ORT DLLs next to exe for dev convenience
            for entry in std::fs::read_dir(&ort_cuda_dir).into_iter().flatten().flatten() {
                let name = entry.file_name();
                let dest = target_debug.join(&name);
                // Overwrite unconditionally — a stale wrong-CUDA copy here would shadow the new one.
                let _ = std::fs::copy(entry.path(), &dest);
            }
        }
    }

    emit("done", 1.0, "CUDA runtime ready. Restart to activate.");
    tracing::info!("CUDA runtime download complete: ORT={}, cuDNN={}", ort_cuda_dir.display(), cuda_dir.display());
    Ok(())
}

async fn download_file(
    client: &reqwest::Client,
    url: &str,
    dest: &std::path::Path,
    progress_cb: impl Fn(f32),
) -> crate::Result<()> {
    let resp = client.get(url).send().await
        .map_err(|e| crate::UtaiError::Audio(format!("Download failed: {}", e)))?;

    if !resp.status().is_success() {
        return Err(crate::UtaiError::Audio(format!("HTTP {}: {}", resp.status(), url)));
    }

    let total = resp.content_length().unwrap_or(0);
    use futures_util::StreamExt;
    let mut stream = resp.bytes_stream();
    let mut file = std::fs::File::create(dest)?;
    let mut downloaded: u64 = 0;

    use std::io::Write;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| crate::UtaiError::Audio(format!("Download stream: {}", e)))?;
        file.write_all(&chunk)?;
        downloaded += chunk.len() as u64;
        if total > 0 {
            progress_cb(downloaded as f32 / total as f32);
        }
    }
    Ok(())
}

// extract_nupkg_dlls / extract_wheel_dlls moved to crate::util::extract_zip_dlls
// (callers pass a starts_with / contains closure for the path match).

pub(crate) fn is_cuda_available() -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // Check CUDA toolkit's standard install location first (fast)
        if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
            let bin = std::path::Path::new(&cuda_path).join("bin");
            if bin.exists() {
                if let Ok(entries) = std::fs::read_dir(&bin) {
                    for entry in entries.flatten() {
                        let name = entry.file_name().to_string_lossy().to_lowercase();
                        if name.starts_with("cudart64_") && name.ends_with(".dll") {
                            return true;
                        }
                    }
                }
            }
        }
        // Fallback: check if nvcc is on PATH (lightweight — just runs one command)
        if let Ok(output) = std::process::Command::new("where")
            .arg("nvcc.exe")
            .creation_flags(crate::util::CREATE_NO_WINDOW)
            .output()
        {
            if output.status.success() {
                return true;
            }
        }
    }
    false
}
