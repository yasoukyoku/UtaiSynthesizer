use std::path::PathBuf;
use std::sync::Arc;
use tauri::State;

use crate::models::{ImportOutcome, ModelEntry, ModelType};
use crate::AppState;

fn parse_voice_type(model_type: &str) -> Option<ModelType> {
    match model_type {
        "rvc" => Some(ModelType::Rvc),
        "sovits" => Some(ModelType::SoVits),
        _ => None,
    }
}

#[tauri::command]
pub async fn list_models(
    state: State<'_, Arc<AppState>>,
    model_type: Option<String>,
) -> Result<Vec<ModelEntry>, String> {
    // Explicit rescan (not just the registry's lazy one): the manager UI calls this after
    // imports/deletes and must reflect on-disk reality.
    state.models.scan().map_err(|e| e.to_string())?;

    match model_type.as_deref() {
        Some("rvc") => Ok(state.models.list_by_type(&ModelType::Rvc)),
        Some("sovits") => Ok(state.models.list_by_type(&ModelType::SoVits)),
        Some("s2h") => Ok(state.models.list_by_type(&ModelType::S2H)),
        Some("f0") => Ok(state.models.list_by_type(&ModelType::F0)),
        Some("nsf_hifigan") => Ok(state.models.list_by_type(&ModelType::NsfHifigan)),
        _ => Ok(state.models.list()),
    }
}

/// Returns the created entry PLUS non-fatal warnings (failed index conversion, synthesized
/// sidecar config, avatar problems) — the frontend must surface these, not just "success".
#[tauri::command]
pub async fn import_model(
    state: State<'_, Arc<AppState>>,
    name: String,
    path: String,
    model_type: String,
    index_path: Option<String>,
    diffusion_path: Option<String>,
    diffusion_config_path: Option<String>,
    avatar_path: Option<String>,
) -> Result<ImportOutcome, String> {
    let mt = parse_voice_type(&model_type)
        .ok_or_else(|| format!("Unsupported model type: {}", model_type))?;

    // A same-name re-import REPLACES the model on disk — drop any live inference session first,
    // or it would keep serving the stale ONNX (and leak the old RvcIndex RAM).
    state.inference.unload_voice(&name);

    let idx = index_path.map(PathBuf::from);
    let diff = diffusion_path.map(PathBuf::from);
    let diff_cfg = diffusion_config_path.map(PathBuf::from);
    let avatar = avatar_path.map(PathBuf::from);
    state
        .models
        .import_file(
            &name,
            &PathBuf::from(path),
            mt,
            &state.app_dir,
            idx.as_deref(),
            diff.as_deref(),
            diff_cfg.as_deref(),
            avatar.as_deref(),
        )
        .map_err(|e| {
            tracing::error!("Model import failed: {}", e);
            e.to_string()
        })
}

#[tauri::command]
pub async fn set_model_avatar(
    state: State<'_, Arc<AppState>>,
    name: String,
    avatar_path: String,
) -> Result<Option<String>, String> {
    state
        .models
        .set_avatar(&name, &PathBuf::from(avatar_path))
        .map(|p| p.map(|x| x.to_string_lossy().to_string()))
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn delete_model(
    state: State<'_, Arc<AppState>>,
    name: String,
) -> Result<(), String> {
    // Unload BEFORE removing files: a loaded session would keep serving the deleted model (and
    // on Windows can hold the .onnx file open, blocking removal).
    state.inference.unload_voice(&name);
    state.models.delete(&name).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn check_model_exists(
    state: State<'_, Arc<AppState>>,
    name: String,
    model_type: String,
) -> Result<bool, String> {
    match parse_voice_type(&model_type) {
        Some(mt) => Ok(state.models.exists(&name, &mt)),
        None => Ok(false),
    }
}
