use std::path::PathBuf;
use std::sync::Arc;
use tauri::State;

use crate::models::{ImportOutcome, ModelEntry, ModelType};
use crate::AppState;

fn parse_voice_type(model_type: &str) -> Option<ModelType> {
    match model_type {
        "rvc" => Some(ModelType::Rvc),
        "sovits" => Some(ModelType::SoVits),
        // S40: the vocoder RESOURCE class (fine-tuned / imported NSF-HiFiGAN
        // vocoders under models/nsf_hifigan/); the aux default vocoder stays
        // aux-resolved and outside the registry
        "vocoder" => Some(ModelType::NsfHifigan),
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
        // S40 alias — the frontend voice store speaks "vocoder" everywhere
        // (import/delete via parse_voice_type do too)
        Some("vocoder") => Ok(state.models.list_by_type(&ModelType::NsfHifigan)),
        _ => Ok(state.models.list()),
    }
}

/// Returns the created entry PLUS non-fatal warnings (failed index conversion, synthesized
/// sidecar config, avatar problems) — the frontend must surface these, not just "success".
#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn import_model(
    state: State<'_, Arc<AppState>>,
    name: String,
    path: String,
    model_type: String,
    index_path: Option<String>,
    diffusion_path: Option<String>,
    diffusion_config_path: Option<String>,
    avatar_path: Option<String>,
    vocoder_config_path: Option<String>,
) -> Result<ImportOutcome, String> {
    let mt = parse_voice_type(&model_type)
        .ok_or_else(|| format!("Unsupported model type: {}", model_type))?;

    // A same-name re-import REPLACES the model on disk — drop any live inference session first,
    // or it would keep serving the stale ONNX (and leak the old RvcIndex RAM).
    state.inference.unload_voice(&name);
    // Vocoder resources are cached BY PATH (engine session + mel filterbank
    // npy), not by voice name — evict them too before the files are replaced
    // (设计红队 A18; a live session also holds a Windows file lock).
    if matches!(mt, ModelType::NsfHifigan) {
        if let Some(old) = state.models.get_by_type(&name, &mt) {
            state.inference.unload_model_file(&old.path);
        }
    }

    let idx = index_path.map(PathBuf::from);
    let diff = diffusion_path.map(PathBuf::from);
    let diff_cfg = diffusion_config_path.map(PathBuf::from);
    let avatar = avatar_path.map(PathBuf::from);
    let voc_cfg = vocoder_config_path.map(PathBuf::from);
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
            voc_cfg.as_deref(),
        )
        .map_err(|e| {
            tracing::error!("Model import failed: {}", e);
            e.to_string()
        })
}

/// Attach a TRAINED shallow-diffusion checkpoint (model_<step>.pt with its
/// config.yaml auto-resolved next to it) to an installed SoVITS model (S39).
/// Conversion + validation run into a temp dir BEFORE the model's live
/// sessions are dropped — a failure leaves an existing attachment untouched
/// and still loaded; the swap itself is rename-based with rollback.
#[tauri::command]
pub async fn attach_diffusion(
    state: State<'_, Arc<AppState>>,
    name: String,
    ckpt_path: String,
    config_path: Option<String>,
) -> Result<ModelEntry, String> {
    let cfg = config_path.map(PathBuf::from);
    let tmp = state
        .models
        .prepare_diffusion_attachment(
            &name,
            &PathBuf::from(&ckpt_path),
            cfg.as_deref(),
            &state.app_dir,
        )
        .map_err(|e| {
            tracing::error!("Diffusion attach (prepare) failed: {}", e);
            e.to_string()
        })?;
    // sessions hold Windows file handles on the OLD attachment — drop them
    // only now, after everything that can fail has succeeded
    state.inference.unload_voice(&name);
    state
        .models
        .commit_diffusion_attachment(&name, &tmp)
        .map_err(|e| {
            tracing::error!("Diffusion attach (commit) failed: {}", e);
            e.to_string()
        })?;
    state
        .models
        .get(&name)
        .ok_or_else(|| format!("找不到模型「{}」", name))
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
    model_type: Option<String>,
) -> Result<(), String> {
    // Type-scoped when the caller knows the type (设计红队 A5): an untyped
    // first-match delete of a vocoder named after its singer would remove the
    // SINGER MODEL's files (scan order rvc→sovits→…→nsf_hifigan).
    let mt = model_type.as_deref().and_then(parse_voice_type);
    // Unload BEFORE removing files: a loaded session would keep serving the deleted model (and
    // on Windows can hold the .onnx file open, blocking removal).
    state.inference.unload_voice(&name);
    if let Some(ModelType::NsfHifigan) = mt {
        if let Some(old) = state.models.get_by_type(&name, &ModelType::NsfHifigan) {
            state.inference.unload_model_file(&old.path);
        }
    }
    state.models.delete(&name, mt.as_ref()).map_err(|e| e.to_string())
}

/// S60-2: persist a model's tested vocal range into its sidecar (the frontend-orchestrated
/// range test writes this; the render layer reads it back via vocal_range::speaker_range).
#[tauri::command]
pub async fn set_model_vocal_range(
    state: State<'_, Arc<AppState>>,
    name: String,
    model_type: String,
    record: serde_json::Value,
) -> Result<(), String> {
    let mt = parse_voice_type(&model_type).ok_or("RANGE_BAD_TYPE")?;
    if !record.is_object() {
        return Err("RANGE_BAD_RECORD".to_string());
    }
    state
        .models
        .set_config_extra_key(&name, &mt, "vocal_range", record)
        .map_err(|e| e.to_string())
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
