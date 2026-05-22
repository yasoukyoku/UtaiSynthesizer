use std::path::PathBuf;
use std::sync::Arc;
use tauri::State;

use crate::models::{ModelEntry, ModelType};
use crate::AppState;

#[tauri::command]
pub async fn list_models(
    state: State<'_, Arc<AppState>>,
    model_type: Option<String>,
) -> Result<Vec<ModelEntry>, String> {
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

#[tauri::command]
pub async fn import_model(
    state: State<'_, Arc<AppState>>,
    name: String,
    path: String,
    model_type: String,
    index_path: Option<String>,
    avatar_path: Option<String>,
) -> Result<ModelEntry, String> {
    let mt = match model_type.as_str() {
        "rvc" => ModelType::Rvc,
        "sovits" => ModelType::SoVits,
        _ => return Err(format!("Unsupported model type: {}", model_type)),
    };

    let idx = index_path.map(PathBuf::from);
    let avatar = avatar_path.map(PathBuf::from);
    state
        .models
        .import_pth(&name, &PathBuf::from(path), mt, &state.app_dir, idx.as_ref(), avatar.as_ref())
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
    state.models.delete(&name).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn check_model_exists(
    state: State<'_, Arc<AppState>>,
    name: String,
    model_type: String,
) -> Result<bool, String> {
    let mt = match model_type.as_str() {
        "rvc" => ModelType::Rvc,
        "sovits" => ModelType::SoVits,
        _ => return Ok(false),
    };
    Ok(state.models.exists(&name, &mt))
}
