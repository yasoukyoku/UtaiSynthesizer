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
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            device: DeviceConfig::Auto,
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

    // Persist to config file
    let cfg = AppConfig { device: config };
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
    std::fs::write(path, json)
}

fn load_config(app_dir: &std::path::Path) -> Option<AppConfig> {
    let path = config_path(app_dir);
    let content = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

fn detect_gpu_name() -> String {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let output = std::process::Command::new("powershell")
            .args(["-NoProfile", "-Command",
                   "(Get-CimInstance -ClassName Win32_VideoController).Name -join ', '"])
            .creation_flags(0x08000000) // CREATE_NO_WINDOW
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

/// Check if CUDA ORT runtime + cuDNN are already downloaded.
#[tauri::command]
pub fn is_cuda_runtime_ready(state: State<'_, Arc<AppState>>) -> Result<bool, String> {
    let ort_cuda_dll = state.app_dir.join("runtime").join("ort").join("cuda").join("onnxruntime.dll");
    let cuda_dir = state.app_dir.join("runtime").join("cuda");
    let has_cudnn = cuda_dir.exists() && std::fs::read_dir(&cuda_dir)
        .map(|entries| entries.flatten().any(|e| {
            let name = e.file_name().to_string_lossy().to_lowercase();
            name.starts_with("cudnn") && name.ends_with(".dll")
        }))
        .unwrap_or(false);
    Ok(ort_cuda_dll.exists() && has_cudnn)
}

/// Download CUDA ORT DLLs + cuDNN DLLs for CUDA EP support.
/// Emits `cuda-download-progress` events with {stage, progress, message}.
#[tauri::command]
pub async fn download_cuda_runtime(
    app_handle: tauri::AppHandle,
    state: State<'_, Arc<AppState>>,
) -> Result<(), String> {
    let app_dir = state.app_dir.clone();
    let handle = app_handle.clone();

    tokio::task::spawn_blocking(move || {
        let rt = tokio::runtime::Handle::current();
        rt.block_on(async {
            do_download_cuda_runtime(&app_dir, &handle).await
        })
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))?
    .map_err(|e| format!("{}", e))
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
    std::fs::create_dir_all(&ort_cuda_dir)?;

    // Gpu.Windows has the actual DLLs (Gpu is just a meta-package).
    let ort_url = "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.Gpu.Windows/1.21.1";
    let ort_nupkg = app_dir.join("runtime").join("ort_gpu.nupkg.zip");

    download_file(&client, ort_url, &ort_nupkg, |p| emit("ort", p * 0.4, "Downloading CUDA ORT...")).await?;
    emit("ort", 0.4, "Extracting CUDA ORT DLLs...");

    extract_nupkg_dlls(&ort_nupkg, &ort_cuda_dir, "runtimes/win-x64/native")?;
    let _ = std::fs::remove_file(&ort_nupkg);

    // ── Stage 2: Download cuDNN from PyPI wheel ──
    emit("cudnn", 0.5, "Downloading cuDNN...");
    let cuda_dir = app_dir.join("runtime").join("cuda");
    std::fs::create_dir_all(&cuda_dir)?;

    let cudnn_url = "https://files.pythonhosted.org/packages/f2/a4/045f8d0ce6b99726d88e76bbb8ee147123f55e80111d89262762d8149abb/nvidia_cudnn_cu12-9.22.0.52-py3-none-win_amd64.whl";
    let cudnn_whl = app_dir.join("runtime").join("cudnn.whl.zip");

    download_file(&client, cudnn_url, &cudnn_whl, |p| emit("cudnn", 0.5 + p * 0.35, "Downloading cuDNN...")).await?;
    emit("cudnn", 0.85, "Extracting cuDNN DLLs...");

    extract_wheel_dlls(&cudnn_whl, &cuda_dir, "nvidia/cudnn/bin")?;
    let _ = std::fs::remove_file(&cudnn_whl);

    // ── Stage 3: Also copy to exe dir for dev mode ──
    emit("copy", 0.95, "Finalizing...");
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let target_debug = exe_dir;
            // Copy CUDA ORT DLLs next to exe for dev convenience
            for entry in std::fs::read_dir(&ort_cuda_dir).into_iter().flatten().flatten() {
                let name = entry.file_name();
                let dest = target_debug.join(&name);
                if !dest.exists() {
                    let _ = std::fs::copy(entry.path(), &dest);
                }
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

fn extract_nupkg_dlls(
    zip_path: &std::path::Path,
    dest_dir: &std::path::Path,
    prefix: &str,
) -> crate::Result<()> {
    let file = std::fs::File::open(zip_path)?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| crate::UtaiError::Audio(format!("Zip open: {}", e)))?;

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)
            .map_err(|e| crate::UtaiError::Audio(format!("Zip entry: {}", e)))?;
        let name = entry.name().to_string();
        if name.starts_with(prefix) && name.ends_with(".dll") {
            let filename = name.rsplit('/').next().unwrap_or(&name);
            let dest = dest_dir.join(filename);
            let mut out = std::fs::File::create(&dest)?;
            std::io::copy(&mut entry, &mut out)?;
            tracing::info!("Extracted: {}", dest.display());
        }
    }
    Ok(())
}

fn extract_wheel_dlls(
    zip_path: &std::path::Path,
    dest_dir: &std::path::Path,
    prefix: &str,
) -> crate::Result<()> {
    let file = std::fs::File::open(zip_path)?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| crate::UtaiError::Audio(format!("Zip open: {}", e)))?;

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)
            .map_err(|e| crate::UtaiError::Audio(format!("Zip entry: {}", e)))?;
        let name = entry.name().to_string();
        if name.contains(prefix) && name.ends_with(".dll") {
            let filename = name.rsplit('/').next().unwrap_or(&name);
            let dest = dest_dir.join(filename);
            let mut out = std::fs::File::create(&dest)?;
            std::io::copy(&mut entry, &mut out)?;
            tracing::info!("Extracted: {}", dest.display());
        }
    }
    Ok(())
}

fn is_cuda_available() -> bool {
    #[cfg(windows)]
    {
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
        if let Ok(output) = std::process::Command::new("where").arg("nvcc.exe").output() {
            if output.status.success() {
                return true;
            }
        }
    }
    false
}
