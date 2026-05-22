pub mod engine;
pub mod f0;
pub mod nsf_hifigan;
pub mod rvc;
pub mod s2h;
pub mod sovits;

use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::PathBuf;

use crate::Result;

pub struct InferenceManager {
    pub engine: engine::OnnxEngine,
    loaded_voices: RwLock<HashMap<String, LoadedVoice>>,
    cached_f0_session: RwLock<Option<(PathBuf, String)>>,
}

#[derive(Clone, Debug)]
pub enum VoiceBackendType {
    Rvc,
    SoVits { shallow_diffusion: bool },
}

struct LoadedVoice {
    backend_type: VoiceBackendType,
    _model_path: PathBuf,
    session_id: String,
    sample_rate: u32,
    index: Option<rvc::RvcIndex>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConvertOptions {
    pub f0_shift: f32,
    pub speaker_id: Option<u32>,
    pub index_ratio: f32,
    pub protect_voiceless: f32,
    pub l2_normalize: bool,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            f0_shift: 0.0,
            speaker_id: None,
            index_ratio: 0.6,
            protect_voiceless: 0.33,
            l2_normalize: false,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SynthesisResult {
    pub audio: Vec<f32>,
    pub sample_rate: u32,
}

impl InferenceManager {
    pub fn new() -> Self {
        Self {
            engine: engine::OnnxEngine::new(),
            loaded_voices: RwLock::new(HashMap::new()),
            cached_f0_session: RwLock::new(None),
        }
    }

    pub fn ensure_f0_loaded(&self, f0_model_path: &PathBuf) -> Result<String> {
        {
            let cached = self.cached_f0_session.read();
            if let Some((path, sid)) = cached.as_ref() {
                if path == f0_model_path && self.engine.is_loaded(sid) {
                    return Ok(sid.clone());
                }
            }
        }
        let sid = self.engine.load_model(f0_model_path)?;
        *self.cached_f0_session.write() = Some((f0_model_path.clone(), sid.clone()));
        tracing::info!("F0 model cached: {}", f0_model_path.display());
        Ok(sid)
    }

    pub fn is_voice_loaded(&self, name: &str) -> bool {
        let voices = self.loaded_voices.read();
        if let Some(voice) = voices.get(name) {
            self.engine.is_loaded(&voice.session_id)
        } else {
            false
        }
    }

    pub fn load_voice(
        &self,
        name: &str,
        model_path: &PathBuf,
        backend_type: VoiceBackendType,
        sample_rate: u32,
        index_path: Option<&PathBuf>,
    ) -> Result<()> {
        if self.is_voice_loaded(name) {
            return Ok(());
        }
        self.unload_voice(name);
        let session_id = self.engine.load_model(model_path)?;

        let index = match (&backend_type, index_path) {
            (VoiceBackendType::Rvc, Some(path)) if path.exists() => {
                match rvc::RvcIndex::load(path) {
                    Ok(idx) => Some(idx),
                    Err(e) => {
                        tracing::warn!("Failed to load index, continuing without: {}", e);
                        None
                    }
                }
            }
            _ => None,
        };

        let voice = LoadedVoice {
            backend_type,
            _model_path: model_path.clone(),
            session_id,
            sample_rate,
            index,
        };
        self.loaded_voices.write().insert(name.to_string(), voice);
        Ok(())
    }

    pub fn convert(
        &self,
        voice_name: &str,
        features: &ndarray::Array2<f32>,
        f0: &[f32],
        options: &ConvertOptions,
    ) -> Result<SynthesisResult> {
        let voices = self.loaded_voices.read();
        let voice = voices
            .get(voice_name)
            .ok_or_else(|| crate::UtaiError::Inference(format!("Voice '{}' not loaded", voice_name)))?;

        match &voice.backend_type {
            VoiceBackendType::Rvc => {
                rvc::infer(&self.engine, &voice.session_id, features, f0, options, voice.index.as_ref(), voice.sample_rate)
            }
            VoiceBackendType::SoVits { shallow_diffusion } => {
                sovits::infer(
                    &self.engine,
                    &voice.session_id,
                    features,
                    f0,
                    options,
                    *shallow_diffusion,
                    voice.sample_rate,
                )
            }
        }
    }

    pub fn unload_voice(&self, name: &str) {
        let mut voices = self.loaded_voices.write();
        if let Some(voice) = voices.remove(name) {
            self.engine.unload_model(&voice.session_id);
        }
    }
}

pub fn apply_pitch_shift(f0: &[f32], semitones: f32) -> Vec<f32> {
    if semitones.abs() < 0.001 {
        return f0.to_vec();
    }
    let ratio = 2.0f32.powf(semitones / 12.0);
    f0.iter()
        .map(|&x| if x > 0.0 { x * ratio } else { 0.0 })
        .collect()
}
