use std::path::PathBuf;
use std::sync::Arc;
use tauri::State;

use crate::training::{StartTrainingRequest, StepPoint, TrainingSnapshot};
use crate::AppState;

fn data_root(state: &AppState) -> PathBuf {
    // data root = parent of the models dir (data/models -> data/)
    state
        .models
        .models_dir()
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| state.app_dir.join("data"))
}

#[tauri::command]
pub async fn start_training(
    app: tauri::AppHandle,
    state: State<'_, Arc<AppState>>,
    request: StartTrainingRequest,
) -> Result<(), String> {
    let data_dir = data_root(&state);
    state
        .training
        .start(app, data_dir, request)
        .map_err(|e| e.to_string())?;
    // AFTER a successful launch: torch needs the VRAM — every ORT GPU session goes
    // (CPU aux stays warm; reload-on-miss restores them later). Doing this on the
    // failure path would evict the whole fleet for nothing.
    state.inference.engine.release_gpu_sessions_except(&[]);
    Ok(())
}

#[tauri::command]
pub async fn stop_training(state: State<'_, Arc<AppState>>) -> Result<(), String> {
    state.training.stop().map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn force_stop_training(state: State<'_, Arc<AppState>>) -> Result<(), String> {
    state.training.force_stop().map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_training_status(
    state: State<'_, Arc<AppState>>,
) -> Result<TrainingSnapshot, String> {
    Ok(state.training.status())
}

/// Clear the finished run's DISPLAY state (snapshot + loss history) back to idle.
/// Files are untouched — the workspace/checkpoints stay resumable.
#[tauri::command]
pub async fn reset_training_display(state: State<'_, Arc<AppState>>) -> Result<(), String> {
    state.training.reset_display().map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_training_history(
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<StepPoint>, String> {
    Ok(state.training.history())
}

/// Whether a training WORKSPACE for this name exists (checkpoints the registry
/// doesn't know about yet) — the retrain-wipes-everything confirm must fire for
/// these too, not only for imported models.
#[tauri::command]
pub async fn check_training_workspace(
    state: State<'_, Arc<AppState>>,
    name: String,
) -> Result<bool, String> {
    let ws = crate::training::workspace_path(&data_root(&state), &name);
    Ok(ws.join("config.json").exists() || ws.join("weights").exists())
}

/// Structured workspace facts (S39): the main-model retrain dialog must warn
/// when the wipe would also destroy diffusion training progress, and the
/// 浅扩散 card phrases its own dialog by resume-vs-cache-reuse.
#[tauri::command]
pub async fn get_training_workspace_info(
    state: State<'_, Arc<AppState>>,
    name: String,
) -> Result<crate::training::WorkspaceInfo, String> {
    Ok(crate::training::workspace_info(&data_root(&state), &name))
}
