pub mod audio;
pub mod commands;
pub mod inference;
pub mod logging;
pub mod models;
pub mod project;
pub mod separation;
pub mod training;

use std::sync::Arc;
use parking_lot::RwLock;
use tauri::Manager;
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
    pub project: RwLock<Option<project::Project>>,
    pub training: training::TrainingManager,
    pub models: models::ModelRegistry,
    pub separation: separation::SeparationManager,
    pub log_buffer: Arc<logging::LogBuffer>,
    pub cache_dir: std::path::PathBuf,
    pub app_dir: std::path::PathBuf,
    pub msst_models_dir: std::path::PathBuf,
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
            project: RwLock::new(None),
            training: training::TrainingManager::new(),
            models: models::ModelRegistry::new(models_dir),
            separation: separation::SeparationManager::new(app_dir.clone()),
            log_buffer,
            cache_dir,
            app_dir,
            msst_models_dir,
        }
    }
}

/// Find and initialize the ORT runtime DLL.
/// Picks CUDA or DirectML DLL based on user's saved device preference.
fn init_ort_runtime(app_dir: &std::path::Path) {
    let dll_name = "onnxruntime.dll";
    let runtime_dir = app_dir.join("runtime").join("ort");

    // Read device preference early to decide which DLL to load
    let prefer_cuda = read_early_device_preference(app_dir);

    let mut search_paths: Vec<std::path::PathBuf> = Vec::new();

    if prefer_cuda {
        // Try CUDA DLL first, then fall back to default
        let cuda_dll = runtime_dir.join("cuda").join(dll_name);
        if cuda_dll.exists() {
            search_paths.push(cuda_dll);
            tracing::info!("CUDA runtime requested — trying CUDA ORT DLL");
        } else {
            tracing::warn!("CUDA requested but runtime/ort/cuda/ not found — falling back to DirectML");
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
                    tracing::info!("ORT runtime loaded successfully");
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
}

/// Read device preference from config.json without full AppState setup.
fn read_early_device_preference(app_dir: &std::path::Path) -> bool {
    let cfg_path = app_dir.join("config.json");
    if let Ok(content) = std::fs::read_to_string(&cfg_path) {
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
            if let Some(device) = val.get("device").and_then(|d| d.as_str()) {
                return device == "cuda";
            }
            // Also handle the object form {"cuda": {"device_id": 0}}
            if val.get("device").and_then(|d| d.get("cuda")).is_some() {
                return true;
            }
        }
    }
    false
}

/// Add CUDA runtime DLL directories to PATH so ORT CUDA EP can find cuDNN/cublas.
fn setup_cuda_dll_paths(app_dir: &std::path::Path) {
    #[cfg(windows)]
    {
        let mut dirs_to_add: Vec<std::path::PathBuf> = Vec::new();

        // 1. App-local CUDA runtime dir (downloaded via settings panel)
        let cuda_dir = app_dir.join("runtime").join("cuda");
        if cuda_dir.exists() {
            dirs_to_add.push(cuda_dir);
        }

        // 2. Python site-packages nvidia dirs (dev convenience)
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
            for dir in &dirs_to_add {
                tracing::info!("Added to PATH for CUDA: {}", dir.display());
            }
        }
    }
}

fn resolve_app_dir() -> std::path::PathBuf {
    let cwd = std::env::current_dir().unwrap_or_default();
    if cwd.join("converter").join("convert.py").exists() {
        return cwd;
    }
    if let Some(parent) = cwd.parent() {
        if parent.join("converter").join("convert.py").exists() {
            return parent.to_path_buf();
        }
    }
    cwd
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let log_dir = logging::get_log_dir();
    let _ = std::fs::create_dir_all(&log_dir);

    let file_appender = tracing_appender::rolling::daily(&log_dir, "utai.log");
    let (non_blocking, _file_guard) = tracing_appender::non_blocking(file_appender);

    let log_buffer = Arc::new(logging::LogBuffer::new(2000));

    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "utai=info".into());

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .with_filter(env_filter),
        )
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(non_blocking)
                .with_ansi(false),
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
        .plugin(tauri_plugin_window_state::Builder::new().build())
        .setup(move |app| {
            let local_data = app.path().app_local_data_dir()
                .unwrap_or_else(|_| std::env::current_dir().unwrap_or_default().join(".utai_data"));
            let cache_dir = local_data.join("cache");
            let models_dir = local_data.join("models");
            let state = Arc::new(AppState::new(app_dir_early, cache_dir, models_dir, Arc::clone(&log_buffer)));
            commands::settings::load_and_apply_config(&state);
            app.manage(state);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::inference::run_rvc,
            commands::inference::run_sovits,
            commands::inference::detect_f0,
            commands::inference::run_s2h,
            commands::project::new_project,
            commands::project::open_project,
            commands::project::save_project,
            commands::project::get_project_state,
            commands::training::start_training,
            commands::training::stop_training,
            commands::training::get_training_status,
            commands::training::check_can_close,
            commands::models::list_models,
            commands::models::import_model,
            commands::models::delete_model,
            commands::models::check_model_exists,
            commands::models::set_model_avatar,
            commands::audio::load_audio_file,
            commands::audio::process_effects,
            commands::audio::save_temp_audio,
            commands::audio::ensure_cache_dir,
            commands::audio::export_audio,
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
            commands::logs::get_log_file_path,
            commands::settings::get_hardware_info,
            commands::settings::set_device_preference,
            commands::settings::get_device_preference,
            commands::settings::is_cuda_runtime_ready,
            commands::settings::download_cuda_runtime,
        ])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                let state = window.state::<Arc<AppState>>();
                if state.training.is_active() {
                    api.prevent_close();
                    let _ = window.minimize();
                }
            }
        })
        .build(tauri::generate_context!())
        .expect("Failed to build UTAI")
        .run(|app, event| {
            if let tauri::RunEvent::ExitRequested { api, .. } = event {
                let state = app.state::<Arc<AppState>>();
                if state.training.is_active() {
                    api.prevent_exit();
                }
            }
        });
}
