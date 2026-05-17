pub mod audio;
pub mod commands;
pub mod inference;
pub mod models;
pub mod project;
pub mod training;

use std::sync::Arc;
use parking_lot::RwLock;
use tauri::Manager;

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
}

impl AppState {
    pub fn new() -> Self {
        Self {
            inference: inference::InferenceManager::new(),
            project: RwLock::new(None),
            training: training::TrainingManager::new(),
            models: models::ModelRegistry::new(),
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "utai=info".into()),
        )
        .init();

    let state = Arc::new(AppState::new());

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_window_state::Builder::new().build())
        .manage(state)
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
            commands::audio::process_effects,
            commands::audio::export_audio,
        ])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                let state = window.state::<Arc<AppState>>();
                if state.training.is_active() {
                    api.prevent_close();
                    let _ = window.hide();
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
