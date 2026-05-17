use std::sync::Arc;
use tauri::State;

use crate::training::{TrainingConfig, TrainingStatus};
use crate::AppState;

#[tauri::command]
pub async fn start_training(
    state: State<'_, Arc<AppState>>,
    config: TrainingConfig,
) -> Result<(), String> {
    state.training.start(config).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn stop_training(state: State<'_, Arc<AppState>>) -> Result<(), String> {
    state.training.stop().map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_training_status(
    state: State<'_, Arc<AppState>>,
) -> Result<TrainingStatus, String> {
    Ok(state.training.status())
}

#[tauri::command]
pub async fn check_can_close(state: State<'_, Arc<AppState>>) -> Result<bool, String> {
    Ok(!state.training.is_active())
}
