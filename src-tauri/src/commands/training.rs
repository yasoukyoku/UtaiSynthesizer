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
    // S41 audition interlock (red-team R4/A2; 审查修复 S41-INT-4): HOLD the
    // audition flag for the whole start sequence — a conversion subprocess may
    // be writing into <workspace>/audition and its ONNX sessions hold Windows
    // file locks; a mere load() check would leave a check-then-act window for
    // an audition to slip in mid-start. The frontend disables the button too;
    // this guard is the authoritative gate.
    let _audition_lock = crate::commands::audition::FlightGuard::acquire(
        "试听渲染进行中，请等待完成后再开始训练",
    )?;
    let data_dir = data_root(&state);
    let audition_dir =
        crate::training::workspace_path(&data_dir, &request.model_name).join("audition");
    // BEFORE manager.start(): drop every audition session (file locks) so the
    // fresh-wipe path inside try_start cannot trip over them. Non-destructive —
    // an evicted session reloads on miss.
    state.inference.engine.unload_paths_with_prefix(&audition_dir);
    state
        .training
        .start(app, data_dir, request)
        .map_err(|e| e.to_string())?;
    // AFTER a successful launch (never on guard-rejected starts, red-team R10 —
    // a rejected start must not cost the user their audition cache): the new
    // run's candidate list supersedes the old one.
    if audition_dir.exists() {
        if let Err(e) = std::fs::remove_dir_all(&audition_dir) {
            tracing::warn!("audition dir cleanup failed (non-fatal): {}", e);
        }
    }
    // torch needs the VRAM — every ORT GPU session goes (CPU aux stays warm;
    // reload-on-miss restores them later). Doing this on the failure path
    // would evict the whole fleet for nothing.
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
/// Files are untouched — the workspace/checkpoints stay resumable. S41: the
/// audition cache dir IS removed (清空结果 = giving up this run's archive entry
/// points, user decision 52588f8) — and the workspace path must be read from
/// the snapshot BEFORE reset clears it (red-team F19/R10).
#[tauri::command]
pub async fn reset_training_display(state: State<'_, Arc<AppState>>) -> Result<(), String> {
    // held for the whole clear (S41-INT-4 — same rationale as start_training)
    let _audition_lock = crate::commands::audition::FlightGuard::acquire(
        "试听渲染进行中，请等待完成后再清空结果",
    )?;
    let workspace = state.training.status().workspace;
    if !workspace.is_empty() {
        let audition_dir = std::path::Path::new(&workspace).join("audition");
        if audition_dir.exists() {
            state.inference.engine.unload_paths_with_prefix(&audition_dir);
            if let Err(e) = std::fs::remove_dir_all(&audition_dir) {
                tracing::warn!("audition dir cleanup failed (non-fatal): {}", e);
            }
        }
    }
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
