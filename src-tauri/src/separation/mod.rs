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
    /// The CURRENT (or most recent) native job's status slot. Each start_native installs a FRESH Arc:
    /// cancel is cooperative (the worker only polls between chunks), so a cancelled worker can outlive
    /// its job by up to a chunk — writing to its own now-orphaned slot, where its late progress/stems
    /// can never masquerade as the NEXT job's result (they used to share one slot: cancel → restart
    /// could deposit the OLD job's stems as the new node's output).
    native_status: Mutex<Arc<Mutex<SeparationStatus>>>,
}

enum ActiveJob {
    Sidecar(sidecar::SeparationSidecar),
    NativeHandle {
        handle: std::thread::JoinHandle<()>,
        /// This job's OWN cancel flag. A shared manager-level flag was reset to `false` by the next
        /// start() — un-cancelling the still-running previous worker, which then RESUMED to completion.
        cancel: Arc<AtomicBool>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SeparationConfig {
    pub audio_path: String,
    pub model_path: String,
    pub output_dir: String,
    #[serde(default = "default_device")]
    pub device: String,
    #[serde(default)]
    pub normalize: bool,
    /// UI override for MSST overlap-add window count (None → keep the model JSON default).
    #[serde(default)]
    pub num_overlap: Option<usize>,
    /// Test-time augmentation: average original / polarity-flip / channel-swap passes.
    #[serde(default)]
    pub use_tta: bool,
    /// HTDemucs random time-shift passes (0 = off).
    #[serde(default)]
    pub shifts: usize,
    /// Inference batch size (UI). None → single-chunk. Effective only on dynamic-batch models.
    #[serde(default)]
    pub batch: Option<usize>,
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
            native_status: Mutex::new(Arc::new(Mutex::new(SeparationStatus {
                state: SeparationState::Idle,
                stems: None,
                progress: 0.0,
            }))),
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
        // Evict all other cached sessions first — see OnnxEngine::release_others. Running two
        // separation nodes back-to-back otherwise keeps the previous model's multi-GB CUDA arena
        // resident while the next one loads, and 12 GB cards fall into WDDM paging.
        engine.release_others(model_path);
        let mut pipe = pipeline::NativePipeline::new(engine, model_path)?;
        pipe.set_normalize(config.normalize);
        if let Some(n) = config.num_overlap {
            pipe.set_num_overlap(n);
        }
        pipe.set_use_tta(config.use_tta);
        pipe.set_shifts(config.shifts);
        if let Some(b) = config.batch {
            pipe.set_batch(b);
        }
        // Report the EP actually backing this run so the user can confirm what hardware ran (and
        // whether Auto / an explicit pick ended up on GPU or fell back).
        if let Some(dev) = engine.resolved_device(pipe.session_id()) {
            tracing::info!("MSST inference backend: {}", dev);
        }
        let sample_rate = pipe.config().sample_rate;

        // Fresh per-JOB status slot + cancel flag (see the field docs): the previous job's detached
        // worker keeps its own pair, so it stays cancelled and its late writes stay invisible.
        let status = Arc::new(Mutex::new(SeparationStatus {
            state: SeparationState::LoadingModel,
            stems: None,
            progress: 0.0,
        }));
        *self.native_status.lock() = Arc::clone(&status);
        let cancel = Arc::new(AtomicBool::new(false));

        let audio_path = PathBuf::from(&config.audio_path);
        let output_dir = PathBuf::from(&config.output_dir);

        let cancel_job = Arc::clone(&cancel);
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

        *active = Some(ActiveJob::NativeHandle { handle, cancel: cancel_job });
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
            Some(ActiveJob::NativeHandle { handle, .. }) => {
                // Read liveness BEFORE the status (not after): a worker that writes its terminal state
                // and exits between the two reads must never be seen as (non-terminal, finished) — that
                // combination is reported as a crash. is_finished() is an acquire load set after the
                // closure returns, so once it's true the closure's final status write is visible below.
                let finished = handle.is_finished();
                let slot = self.native_status.lock().clone();
                let s = slot.lock().clone();
                // If the worker thread has ended but never reached a terminal state, it
                // panicked or was killed (e.g. OOM). Surface an error immediately instead
                // of letting the frontend poll until its timeout.
                if finished
                    && matches!(
                        s.state,
                        SeparationState::LoadingModel | SeparationState::Separating
                    )
                {
                    let msg = "Separation worker exited unexpectedly (possible crash or out of memory) — check logs".to_string();
                    tracing::error!("{}", msg);
                    return SeparationStatus {
                        state: SeparationState::Error(msg),
                        stems: None,
                        progress: s.progress,
                    };
                }
                s
            }
            None => {
                let slot = self.native_status.lock().clone();
                let s = slot.lock().clone();
                if matches!(
                    s.state,
                    SeparationState::Completed | SeparationState::Error(_)
                ) {
                    return s;
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
        let mut active = self.active.lock();
        match active.take() {
            Some(ActiveJob::Sidecar(s)) => {
                s.cancel()?;
                // Mark the (native) slot terminal too: status()'s None branch reads it, and a PREVIOUS
                // native job's Completed+stems left there would otherwise settle as this sidecar job's
                // result on the frontend's post-cancel re-poll.
                self.native_status.lock().lock().state =
                    SeparationState::Error("Cancelled".to_string());
            }
            Some(ActiveJob::NativeHandle { cancel, .. }) => {
                // Flip THIS job's flag (the worker polls it between chunks) and mark its slot terminal
                // for the frontend. The detached worker keeps only its own flag + slot: the next start()
                // installs fresh ones, so it can neither un-cancel this worker nor see its late writes.
                // A slot ALREADY Completed is kept — the run finished just before the cancel landed, and
                // the frontend's post-cancel re-poll deliberately accepts a completed result.
                cancel.store(true, Ordering::Relaxed);
                let slot = self.native_status.lock().clone();
                let mut s = slot.lock();
                if !matches!(s.state, SeparationState::Completed) {
                    s.state = SeparationState::Error("Cancelled".to_string());
                }
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
            Some(ActiveJob::NativeHandle { handle, .. }) => handle.is_finished(),
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

