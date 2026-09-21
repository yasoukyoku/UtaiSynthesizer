use std::sync::Arc;
use tauri::State;

use crate::separation::{SeparationConfig, SeparationStatus};
use crate::AppState;

#[tauri::command]
pub async fn run_msst_separation(
    state: State<'_, Arc<AppState>>,
    config: SeparationConfig,
) -> Result<(), String> {
    state.separation.clear_completed();
    state
        .separation
        .start(config, &state.inference.engine)
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_separation_status(
    state: State<'_, Arc<AppState>>,
) -> Result<SeparationStatus, String> {
    Ok(state.separation.status())
}

#[tauri::command]
pub async fn cancel_separation(
    state: State<'_, Arc<AppState>>,
) -> Result<(), String> {
    state.separation.cancel().map_err(|e| e.to_string())
}
