use std::path::PathBuf;
use std::sync::Arc;
use tauri::State;

use crate::inference::{ConvertOptions, SynthesisResult, VoiceBackendType};
use crate::AppState;

#[tauri::command]
pub async fn run_rvc(
    state: State<'_, Arc<AppState>>,
    voice_name: String,
    model_path: String,
    audio_path: String,
    options: ConvertOptions,
) -> Result<SynthesisResult, String> {
    let path = PathBuf::from(&model_path);

    let model_entry = state
        .models
        .get(&voice_name)
        .ok_or_else(|| format!("Model '{}' not found in registry", voice_name))?;

    state
        .inference
        .load_voice(&voice_name, &path, VoiceBackendType::Rvc, model_entry.sample_rate, model_entry.index_path.as_ref())
        .map_err(|e| e.to_string())?;

    let audio_buf = crate::audio::load_audio(&PathBuf::from(&audio_path))
        .map_err(|e| e.to_string())?;

    let f0_model = state
        .models
        .list_by_type(&crate::models::ModelType::F0)
        .first()
        .cloned()
        .ok_or_else(|| "No F0 model available".to_string())?;

    let f0_session = state
        .inference
        .ensure_f0_loaded(&f0_model.path)
        .map_err(|e| e.to_string())?;

    let f0_result = crate::inference::f0::detect(
        &state.inference.engine,
        &f0_session,
        &audio_buf.samples,
        audio_buf.sample_rate,
    )
    .map_err(|e| e.to_string())?;

    let n_frames = f0_result.f0.len();
    let features = ndarray::Array2::zeros((n_frames, 768));

    let result = state
        .inference
        .convert(&voice_name, &features, &f0_result.f0, &options)
        .map_err(|e| e.to_string())?;

    Ok(result)
}

#[tauri::command]
pub async fn run_sovits(
    state: State<'_, Arc<AppState>>,
    voice_name: String,
    model_path: String,
    audio_path: String,
    options: ConvertOptions,
    shallow_diffusion: bool,
) -> Result<SynthesisResult, String> {
    let path = PathBuf::from(&model_path);

    let model_entry = state
        .models
        .get(&voice_name)
        .ok_or_else(|| format!("Model '{}' not found in registry", voice_name))?;

    state
        .inference
        .load_voice(
            &voice_name,
            &path,
            VoiceBackendType::SoVits { shallow_diffusion },
            model_entry.sample_rate,
            None,
        )
        .map_err(|e| e.to_string())?;

    let audio_buf = crate::audio::load_audio(&PathBuf::from(&audio_path))
        .map_err(|e| e.to_string())?;

    let f0_model = state
        .models
        .list_by_type(&crate::models::ModelType::F0)
        .first()
        .cloned()
        .ok_or_else(|| "No F0 model available".to_string())?;

    let f0_session = state
        .inference
        .ensure_f0_loaded(&f0_model.path)
        .map_err(|e| e.to_string())?;

    let f0_result = crate::inference::f0::detect(
        &state.inference.engine,
        &f0_session,
        &audio_buf.samples,
        audio_buf.sample_rate,
    )
    .map_err(|e| e.to_string())?;

    let n_frames = f0_result.f0.len();
    let features = ndarray::Array2::zeros((n_frames, 768));

    let result = state
        .inference
        .convert(&voice_name, &features, &f0_result.f0, &options)
        .map_err(|e| e.to_string())?;

    Ok(result)
}

#[tauri::command]
pub async fn detect_f0(
    state: State<'_, Arc<AppState>>,
    audio_path: String,
) -> Result<Vec<f32>, String> {
    let audio_buf = crate::audio::load_audio(&PathBuf::from(&audio_path))
        .map_err(|e| e.to_string())?;

    let f0_model = state
        .models
        .list_by_type(&crate::models::ModelType::F0)
        .first()
        .cloned()
        .ok_or_else(|| "No F0 model available".to_string())?;

    let f0_session = state
        .inference
        .ensure_f0_loaded(&f0_model.path)
        .map_err(|e| e.to_string())?;

    let f0_result = crate::inference::f0::detect(
        &state.inference.engine,
        &f0_session,
        &audio_buf.samples,
        audio_buf.sample_rate,
    )
    .map_err(|e| e.to_string())?;

    Ok(f0_result.f0)
}

#[tauri::command]
pub async fn run_s2h(
    state: State<'_, Arc<AppState>>,
    phonemes: Vec<i64>,
    durations: Vec<i64>,
    pitches: Vec<f32>,
) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), String> {
    let s2h_model = state
        .models
        .list_by_type(&crate::models::ModelType::S2H)
        .first()
        .cloned()
        .ok_or_else(|| "No S2H model available".to_string())?;

    let session_id = state
        .inference
        .engine
        .load_model(&s2h_model.path)
        .map_err(|e| e.to_string())?;

    let score = crate::inference::s2h::ScoreInput {
        phonemes,
        durations,
        pitches,
    };

    let output = match crate::inference::s2h::infer(&state.inference.engine, &session_id, &score) {
        Ok(o) => {
            state.inference.engine.unload_model(&session_id);
            o
        }
        Err(e) => {
            state.inference.engine.unload_model(&session_id);
            return Err(e.to_string());
        }
    };

    let hubert: Vec<Vec<f32>> = output
        .hubert_features
        .rows()
        .into_iter()
        .map(|r| r.to_vec())
        .collect();
    let contentvec: Vec<Vec<f32>> = output
        .contentvec_features
        .rows()
        .into_iter()
        .map(|r| r.to_vec())
        .collect();

    Ok((hubert, contentvec))
}
