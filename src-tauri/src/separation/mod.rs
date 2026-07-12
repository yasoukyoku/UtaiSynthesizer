pub mod pipeline;
// stft moved to the utai-dsp sub-crate (dev opt-3); re-export keeps the public path
// `utai_lib::separation::stft::*` (tests) working unchanged.
pub use utai_dsp::stft;
// The python (audio-separator) fallback sidecar was removed in S42: its venv never shipped
// (python/msst/.venv did not exist — the fallback was effectively dead since S31 made the
// native ONNX pipeline the sole production path), and the S42 embedded-runtime work
// deliberately does NOT carry the audio-separator dependency family. Un-converted models
// now fail loudly with a "convert first" message instead of silently limping into python.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::inference::engine::OnnxEngine;
use crate::{Result, UtaiError};

pub struct SeparationManager {
    active: Mutex<Option<ActiveJob>>,
    /// The CURRENT (or most recent) native job's status slot. Each start_native installs a FRESH Arc:
    /// cancel is cooperative (the worker only polls between chunks), so a cancelled worker can outlive
    /// its job by up to a chunk — writing to its own now-orphaned slot, where its late progress/stems
    /// can never masquerade as the NEXT job's result (they used to share one slot: cancel → restart
    /// could deposit the OLD job's stems as the new node's output).
    native_status: Mutex<Arc<Mutex<SeparationStatus>>>,
}

enum ActiveJob {
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
    /// Inference precision: "fp16" selects the `<stem>.fp16.onnx` sibling when it exists
    /// (falls back to fp32 with a warning otherwise). None/"fp32" → the fp32 model.
    /// fp16 halves VRAM+size and is ~2x faster; verified 53-59 dB vs fp32 for both roformers —
    /// and REQUIRED for MelBand inst_v2 on 12GB cards (fp32 saturates VRAM into WDDM paging).
    #[serde(default)]
    pub precision: Option<String>,
    /// UVR VR arch only: aggressiveness (−100..100, UVR default 5).
    #[serde(default)]
    pub aggression: Option<i32>,
    /// UVR VR arch only: post-process (merge_artifacts) toggle + threshold (0.1/0.2/0.3).
    #[serde(default)]
    pub post_process: Option<bool>,
    #[serde(default)]
    pub post_process_threshold: Option<f32>,
}

fn default_device() -> String {
    "cpu".to_string()
}

/// `x.onnx` → `x.fp16.onnx` (same dir). The fp16 variant shares the fp32 model's `.json`
/// (see `pipeline::model_config_path`) so the two precisions can never drift apart.
pub fn fp16_sibling(p: &Path) -> PathBuf {
    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or_default();
    p.with_file_name(format!("{stem}.fp16.onnx"))
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
    pub fn new(_app_dir: PathBuf) -> Self {
        Self {
            active: Mutex::new(None),
            native_status: Mutex::new(Arc::new(Mutex::new(SeparationStatus {
                state: SeparationState::Idle,
                stems: None,
                progress: 0.0,
            }))),
        }
    }

    /// Native Rust pipeline only (.onnx + matching .json config). Un-converted models are
    /// rejected with a "convert first" error — the python fallback was removed in S42.
    pub fn start(&self, config: SeparationConfig, engine: &OnnxEngine) -> Result<()> {
        let mut active = self.active.lock();
        if active.is_some() {
            // Stable CODE (i18n rule): the frontend pre-flights via get_separation_status before
            // dispatching a run, so this is the TOCTOU backstop — engine.ts maps it to a localized
            // toast/node error.
            return Err(UtaiError::Audio("SEPARATION_BUSY".to_string()));
        }

        let mut model_path = PathBuf::from(&config.model_path);
        if config.precision.as_deref() == Some("fp16") {
            let fp16 = fp16_sibling(&model_path);
            if fp16.exists() {
                tracing::info!("Using fp16 model variant: {}", fp16.display());
                model_path = fp16;
            } else {
                tracing::warn!(
                    "fp16 precision requested but {} not found — falling back to fp32",
                    fp16.display()
                );
            }
        } else if !model_path.exists() {
            // Download-time precision choice may have installed ONLY the fp16 variant — the
            // frontend always addresses models by their fp32 path.
            let fp16 = fp16_sibling(&model_path);
            if fp16.exists() {
                tracing::info!("fp32 model absent, using installed fp16 variant: {}", fp16.display());
                model_path = fp16;
            }
        }
        let has_onnx_config = model_path.extension().map_or(false, |ext| ext == "onnx")
            && pipeline::model_config_path(&model_path).exists();

        if has_onnx_config {
            self.start_native(&mut active, config, engine, &model_path)
        } else {
            // CODE + detail suffix (i18n rule) — engine.ts localizes the text and appends the path.
            Err(UtaiError::Audio(format!(
                "MSST_MODEL_NOT_CONVERTED: {}",
                model_path.display()
            )))
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
        if let Some(a) = config.aggression {
            pipe.set_aggression(a);
        }
        if let Some(pp) = config.post_process {
            pipe.set_post_process(pp, config.post_process_threshold);
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

    pub fn status(&self) -> SeparationStatus {
        let active = self.active.lock();
        match active.as_ref() {
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

