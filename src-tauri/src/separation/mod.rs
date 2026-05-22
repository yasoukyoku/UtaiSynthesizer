pub mod pipeline;
pub mod sidecar;
pub mod stft;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::inference::engine::OnnxEngine;
use crate::{Result, UtaiError};

pub struct SeparationManager {
    script_path: PathBuf,
    venv_python: PathBuf,
    fallback_python: PathBuf,
    active: Mutex<Option<ActiveJob>>,
    native_status: Arc<Mutex<SeparationStatus>>,
    cancel_flag: Arc<AtomicBool>,
}

enum ActiveJob {
    Sidecar(sidecar::SeparationSidecar),
    NativeHandle(std::thread::JoinHandle<()>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SeparationConfig {
    pub audio_path: String,
    pub model_path: String,
    pub output_dir: String,
    #[serde(default = "default_device")]
    pub device: String,
}

fn default_device() -> String {
    "cpu".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeparationStatus {
    pub state: SeparationState,
    pub stems: Option<Vec<StemOutput>>,
    pub progress: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SeparationState {
    Idle,
    LoadingModel,
    Separating,
    Completed,
    Error(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StemOutput {
    pub label: String,
    pub path: String,
}

impl SeparationManager {
    pub fn new(app_dir: PathBuf) -> Self {
        let msst_dir = app_dir.join("python").join("msst");
        Self {
            script_path: msst_dir.join("separate.py"),
            venv_python: msst_dir.join(".venv").join("Scripts").join("python.exe"),
            fallback_python: PathBuf::from("python"),
            active: Mutex::new(None),
            native_status: Arc::new(Mutex::new(SeparationStatus {
                state: SeparationState::Idle,
                stems: None,
                progress: 0.0,
            })),
            cancel_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Auto-detects native vs sidecar:
    /// - .onnx file with matching .json config → native Rust pipeline
    /// - Otherwise → Python sidecar fallback
    pub fn start(&self, config: SeparationConfig, engine: &OnnxEngine) -> Result<()> {
        let mut active = self.active.lock();
        if active.is_some() {
            return Err(UtaiError::Audio(
                "Separation already in progress".to_string(),
            ));
        }

        let model_path = PathBuf::from(&config.model_path);
        let has_onnx_config = model_path.extension().map_or(false, |ext| ext == "onnx")
            && model_path.with_extension("json").exists();

        if has_onnx_config {
            self.start_native(&mut active, config, engine, &model_path)
        } else {
            self.start_sidecar(&mut active, config)
        }
    }

    fn start_native(
        &self,
        active: &mut Option<ActiveJob>,
        config: SeparationConfig,
        engine: &OnnxEngine,
        model_path: &Path,
    ) -> Result<()> {
        let pipe = pipeline::NativePipeline::new(engine, model_path)?;
        let sample_rate = pipe.config().sample_rate;

        {
            let mut s = self.native_status.lock();
            s.state = SeparationState::LoadingModel;
            s.stems = None;
            s.progress = 0.0;
        }

        self.cancel_flag.store(false, Ordering::Relaxed);

        let audio_path = PathBuf::from(&config.audio_path);
        let output_dir = PathBuf::from(&config.output_dir);
        let status = Arc::clone(&self.native_status);
        let cancel = Arc::clone(&self.cancel_flag);

        let handle = std::thread::spawn(move || {
            let audio = match load_audio_for_separation(&audio_path, sample_rate) {
                Ok(a) => a,
                Err(e) => {
                    let msg = format!("Failed to load audio: {}", e);
                    tracing::error!("{}", msg);
                    status.lock().state = SeparationState::Error(msg);
                    return;
                }
            };

            status.lock().state = SeparationState::Separating;

            if cancel.load(Ordering::Relaxed) {
                status.lock().state = SeparationState::Error("Cancelled".into());
                return;
            }

            let status_cb = Arc::clone(&status);
            let cancel_cb = Arc::clone(&cancel);

            match pipe.separate(&audio, &|p| {
                status_cb.lock().progress = p;
                !cancel_cb.load(Ordering::Relaxed)
            }) {
                Ok(stems) => {
                    let _ = std::fs::create_dir_all(&output_dir);
                    let mut stem_outputs = Vec::new();

                    for stem in &stems {
                        let filename = format!("{}.wav", stem.label);
                        let stem_path = output_dir.join(&filename);
                        if let Err(e) = pipeline::save_wav(&stem_path, stem, sample_rate) {
                            let msg = format!("Failed to save stem: {}", e);
                            tracing::error!("{}", msg);
                            status.lock().state = SeparationState::Error(msg);
                            return;
                        }
                        stem_outputs.push(StemOutput {
                            label: stem.label.clone(),
                            path: stem_path.to_string_lossy().to_string(),
                        });
                    }

                    let mut s = status.lock();
                    s.state = SeparationState::Completed;
                    s.stems = Some(stem_outputs);
                    s.progress = 1.0;
                }
                Err(e) => {
                    let msg = format!("Separation failed: {}", e);
                    tracing::error!("{}", msg);
                    status.lock().state = SeparationState::Error(msg);
                }
            }
        });

        *active = Some(ActiveJob::NativeHandle(handle));
        tracing::info!("Native MSST separation started: {}", model_path.display());
        Ok(())
    }

    fn start_sidecar(
        &self,
        active: &mut Option<ActiveJob>,
        config: SeparationConfig,
    ) -> Result<()> {
        let sidecar_config = serde_json::json!({
            "audioPath": config.audio_path,
            "modelName": config.model_path,
            "outputDir": config.output_dir,
            "device": config.device,
        });
        let config_json = serde_json::to_string(&sidecar_config)?;

        let python = if self.venv_python.exists() {
            &self.venv_python
        } else {
            &self.fallback_python
        };

        let sidecar =
            sidecar::SeparationSidecar::spawn(python, &self.script_path, &config_json)?;
        *active = Some(ActiveJob::Sidecar(sidecar));
        tracing::info!("Sidecar MSST separation started: {}", config.model_path);
        Ok(())
    }

    pub fn status(&self) -> SeparationStatus {
        let active = self.active.lock();
        match active.as_ref() {
            Some(ActiveJob::Sidecar(s)) => s.status(),
            Some(ActiveJob::NativeHandle(_)) => self.native_status.lock().clone(),
            None => {
                let s = self.native_status.lock();
                if matches!(
                    s.state,
                    SeparationState::Completed | SeparationState::Error(_)
                ) {
                    return s.clone();
                }
                SeparationStatus {
                    state: SeparationState::Idle,
                    stems: None,
                    progress: 0.0,
                }
            }
        }
    }

    pub fn cancel(&self) -> Result<()> {
        self.cancel_flag.store(true, Ordering::Relaxed);
        let mut active = self.active.lock();
        match active.take() {
            Some(ActiveJob::Sidecar(s)) => {
                s.cancel()?;
            }
            Some(ActiveJob::NativeHandle(_handle)) => {
                self.native_status.lock().state =
                    SeparationState::Error("Cancelled".to_string());
            }
            None => {}
        }
        Ok(())
    }

    pub fn clear_completed(&self) {
        let mut active = self.active.lock();
        let should_clear = match active.as_ref() {
            Some(ActiveJob::Sidecar(s)) => {
                let st = s.status();
                matches!(
                    st.state,
                    SeparationState::Completed | SeparationState::Error(_)
                )
            }
            Some(ActiveJob::NativeHandle(h)) => h.is_finished(),
            None => false,
        };
        if should_clear {
            *active = None;
        }
    }
}

/// Load audio at a specific sample rate for separation.
/// Uses ffmpeg -ar to resample during decode (much faster than post-load resampling).
fn load_audio_for_separation(path: &Path, target_sr: u32) -> crate::Result<pipeline::AudioData> {
    let buf = load_at_sample_rate(path, target_sr)?;
    let channels = buf.channels as usize;
    let num_frames = buf.samples.len() / channels;

    let (left, right) = if channels >= 2 {
        let mut l = Vec::with_capacity(num_frames);
        let mut r = Vec::with_capacity(num_frames);
        for i in 0..num_frames {
            l.push(buf.samples[i * channels]);
            r.push(buf.samples[i * channels + 1]);
        }
        (l, r)
    } else {
        (buf.samples, vec![])
    };

    Ok(pipeline::AudioData {
        left,
        right,
        channels: channels.min(2),
        sample_rate: target_sr,
    })
}

/// Load audio file, resampling to target_sr via ffmpeg if needed.
fn load_at_sample_rate(path: &Path, target_sr: u32) -> crate::Result<crate::audio::AudioBuffer> {
    let buf = crate::audio::load_audio(path)?;
    if buf.sample_rate == target_sr {
        return Ok(buf);
    }
    tracing::info!("Resampling via ffmpeg: {}Hz → {}Hz", buf.sample_rate, target_sr);
    crate::audio::load_audio_at_rate(path, target_sr)
}

