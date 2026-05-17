use std::path::PathBuf;
use std::sync::Arc;
use tauri::State;

use crate::audio::effects::{Effect, VocoderChoice};
use crate::audio::export::{ExportConfig, ExportFormat};
use crate::audio::AudioBuffer;
use crate::AppState;

#[derive(serde::Deserialize)]
pub struct EffectRequest {
    pub audio_path: String,
    pub effects: Vec<Effect>,
    pub output_path: Option<String>,
}

#[derive(serde::Serialize)]
pub struct EffectResult {
    pub output_path: String,
    pub duration_secs: f64,
}

#[tauri::command]
pub async fn process_effects(
    state: State<'_, Arc<AppState>>,
    request: EffectRequest,
) -> Result<EffectResult, String> {
    let mut buffer = crate::audio::load_wav(&PathBuf::from(&request.audio_path))
        .map_err(|e| e.to_string())?;

    let nsf_session: Option<&str> = None; // TODO: load NSF-HiFiGAN session if needed

    for effect in &request.effects {
        buffer = crate::audio::effects::apply_effect(
            &buffer,
            effect,
            &state.inference.engine,
            nsf_session.as_deref(),
        )
        .map_err(|e| e.to_string())?;
    }

    let output_path = request
        .output_path
        .unwrap_or_else(|| {
            let input = PathBuf::from(&request.audio_path);
            let stem = input.file_stem().unwrap_or_default().to_string_lossy();
            input
                .with_file_name(format!("{}_processed.wav", stem))
                .to_string_lossy()
                .to_string()
        });

    crate::audio::save_wav(&PathBuf::from(&output_path), &buffer)
        .map_err(|e| e.to_string())?;

    Ok(EffectResult {
        output_path,
        duration_secs: buffer.duration_secs(),
    })
}

#[tauri::command]
pub async fn export_audio(
    state: State<'_, Arc<AppState>>,
    _output_path: String,
    _format: String,
    _sample_rate: u32,
    _normalize: bool,
) -> Result<(), String> {
    let proj = state.project.read();
    let _project = proj
        .as_ref()
        .ok_or_else(|| "No project open".to_string())?;

    // TODO: Collect all track audio, mixdown, and export
    // This requires rendering all tracks (which needs the full pipeline)
    // For now, return an error indicating the feature is pending S2H integration
    Err("Export requires rendered tracks — pending full pipeline integration".to_string())
}
