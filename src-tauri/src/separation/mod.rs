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
    // ⛔ A `device: String` field (default "cpu") lived here from S31 until S170. NOTHING in
    // src-tauri ever read it — the execution provider comes from the GLOBAL OnnxEngine::device()
    // preference that Settings writes, never from a per-node string — while engine.ts sent it on
    // every dispatch. A field the caller sets and the backend ignores is worse than no field at
    // all. The `every_separation_config_field_is_actually_read` gate below keeps it, and anything
    // like it, from growing back.
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
        // Reap a finished predecessor inside the SAME critical section that admits the new job.
        // Until S170 this was a two-lock check-then-act: commands/separation.rs called
        // clear_completed() (lock, look, unlock) and only THEN start() (lock again, look again).
        let prev = self.reap_finished(&mut active);
        if active.is_some() {
            // Reaching here now means exactly ONE thing: a worker is genuinely mid-run. Before the
            // reap above, this same red ALSO fired for a run that had already FINISHED — one CODE
            // and one user-facing sentence for two different situations, which is the failure this
            // project's first rule names. It is logged too: the 2026-09-09 report had to be
            // reconstructed from the ABSENCE of a second "Native MSST separation started" line,
            // because the throw site wrote nothing at all.
            tracing::warn!(
                "SEPARATION_BUSY: refusing the dispatch, a separation is still in flight (slot state {:?})",
                prev
            );
            // Stable CODE (i18n rule): the frontend pre-flights via get_separation_status before
            // dispatching a run, so this is the TOCTOU backstop — engine.ts maps it to a localized
            // toast/node error.
            return Err(UtaiError::Audio("SEPARATION_BUSY".to_string()));
        }

        let fp32_path = PathBuf::from(&config.model_path);
        let mut model_path = fp32_path.clone();
        let mut using_fp16 = false;
        if config.precision.as_deref() == Some("fp16") {
            let fp16 = fp16_sibling(&model_path);
            if fp16.exists() {
                tracing::info!("Using fp16 model variant: {}", fp16.display());
                model_path = fp16;
                using_fp16 = true;
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
                using_fp16 = true;
            }
        }

        // The CPU execution provider has no fp16 kernels. The repo already depends on that from
        // the other side — the fp16 gate rule on MSST_FP16_ARCHS in src/lib/models/msst-catalog.ts
        // says fp16 verification MUST run on the CUDA EP because "the CPU EP emulates fp16 in
        // fp32 and false-passes". ⚠ That quoted sentence is the whole of what we have measured:
        // the exact emulation shape, and what it costs in time or memory, is NOT measured here and
        // must not be asserted. What follows from it alone is enough: an fp16 graph on CPU buys
        // nothing the fp32 graph does not already give — and until S170
        // this whole function had no device test at all: on 2026-09-09 a user whose DirectML run
        // had died with DXGI_ERROR_DEVICE_HUNG switched Settings to CPU, the fp16 model was picked
        // anyway, the run entered its inference loop and the process died without another log line.
        //
        // This keys on the EXPLICIT Cpu preference only. Auto can also RESOLVE to CPU (no usable
        // GPU), but that verdict exists only inside build_session_auto; start_native logs the
        // resolved backend once the session is built, which is where that case surfaces today.
        if using_fp16 && matches!(engine.device(), crate::inference::engine::DeviceConfig::Cpu) {
            if fp32_path.exists() {
                tracing::warn!(
                    "Device preference is CPU and the CPU provider has no fp16 kernels — using the \
                     fp32 model {} instead of {}",
                    fp32_path.display(),
                    model_path.display()
                );
                model_path = fp32_path.clone();
            } else {
                // CODE + detail suffix (i18n rule) — engine.ts localizes the text and appends the
                // path. Refuse rather than start: the remedy (convert the fp32 variant) is one
                // click in the model manager, and an unattended fp16-on-CPU run is exactly what
                // produced a silent process death with nothing in the log to act on.
                tracing::error!(
                    "Refusing a CPU separation: only the fp16 variant of {} is installed",
                    fp32_path.display()
                );
                return Err(UtaiError::Audio(format!(
                    "MSST_FP16_ON_CPU: {}",
                    fp32_path.display()
                )));
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
        tracing::info!(
            "Native MSST separation started: {} ({})",
            model_path.display(),
            crate::inference::engine::memory_stamp()
        );
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

    /// Drop the previous job's slot if that job is DONE. The caller must already hold `active`.
    /// Returns the previous job's slot state (None when there was no job at all) so the caller can
    /// say WHICH situation it refused instead of reporting two different things as one red.
    ///
    /// ⛔ Why the predicate is "terminal slot", not "thread finished" — the 2026-09-09 field bug:
    /// the worker writes `Completed` into its slot INSIDE its closure, and `JoinHandle::is_finished()`
    /// only flips AFTER that closure returns, with the decoded audio and the stem buffers dropped in
    /// between. "Slot says Completed, thread not joined yet" is therefore not a race that MIGHT
    /// happen — it is a window that exists on every single run. How often the next dispatch actually
    /// LANDS in it is a different question and was observed exactly once. Meanwhile the frontend pre-flight (engine.ts rejectIfSeparationBusy)
    /// counts only LoadingModel/Separating as busy, so it structurally cannot see Completed and had
    /// already dispatched the next node. Field log 2026-09-09, quoted as what it SHOWS: `separate_spectral TOTAL: 254247.1 ms`
    /// at 22:10:11.512 (that is the spectral pass returning — `Completed` is written later, after
    /// both stems are saved), then `SEPARATION_BUSY` at 22:10:12.159, and the whole 6-node run
    /// failed. The log does NOT say which dispatch was refused; that it was the run's second
    /// separation node is inferred from the absence of a second `Native MSST separation started`
    /// line. This predicate is now the frontend's.
    ///
    /// Reaping early is safe precisely because each job owns its status slot and its cancel flag
    /// (see the field docs above): a predecessor that is still unwinding holds its OWN Arc, so it can
    /// neither write into the next job's slot nor be un-cancelled by it. It also holds no engine
    /// session — NativePipeline keeps a session id, not the Session.
    ///
    /// Lock order is this file's existing one — `active` (held by the caller) → `native_status` →
    /// the job's own slot — and `native_status` is released before the slot is locked, exactly as
    /// `status()` and `cancel()` do it. The worker thread only ever takes its own slot, so it can
    /// never block this.
    fn reap_finished(&self, active: &mut Option<ActiveJob>) -> Option<SeparationState> {
        let thread_exited = match active.as_ref() {
            Some(ActiveJob::NativeHandle { handle, .. }) => handle.is_finished(),
            None => return None,
        };
        let slot = self.native_status.lock().clone();
        let state = slot.lock().state.clone();
        let terminal = matches!(
            state,
            SeparationState::Completed | SeparationState::Error(_)
        );
        if terminal || thread_exited {
            *active = None;
            // Two reasons, one line, both named: a run that ENDED normally vs a worker that left
            // without ever reaching a terminal state (crash / OOM — `status()` synthesises the error
            // for that one). From the outside they used to be indistinguishable.
            tracing::info!(
                "Reaped the previous separation job before dispatch (slot state {:?}, worker thread exited {})",
                state,
                thread_exited
            );
        }
        Some(state)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::engine::DeviceConfig;
    use std::sync::mpsc;

    fn scratch_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir()
            .join("utai_separation_gate")
            .join(format!("{}_{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).expect("scratch dir");
        d
    }

    /// Build the config through SERDE, exactly the way `run_msst_separation` receives it from
    /// engine.ts. Keeps compiling across field changes, and doubles as the proof that those three
    /// camelCase keys are the only ones a dispatch must carry.
    fn config_for(dir: &Path, model: &Path, precision: Option<&str>) -> SeparationConfig {
        let mut v = serde_json::json!({
            "audioPath": dir.join("in.wav").to_string_lossy(),
            "modelPath": model.to_string_lossy(),
            "outputDir": dir.join("out").to_string_lossy(),
        });
        if let Some(p) = precision {
            v["precision"] = serde_json::Value::String(p.to_string());
        }
        serde_json::from_value(v).expect("SeparationConfig deserialises from the frontend shape")
    }

    /// Install a job whose worker has REACHED `state` and then PARKS — alive, `is_finished()`
    /// false — until the returned sender is dropped.
    ///
    /// ⛔ No sleeps and no deadlines anywhere in this module: the hand-off window is held open by a
    /// channel handshake, so every assertion below is a statement about the CODE, not about how
    /// fast this machine happens to be. (Three synthetic fixtures timed this same window at
    /// 4.4 / 11.4 / 17.9 ms — 4.2x apart — which is why not one millisecond appears here.)
    fn park_worker_at(mgr: &SeparationManager, state: SeparationState) -> mpsc::Sender<()> {
        let status = Arc::new(Mutex::new(SeparationStatus {
            state: SeparationState::LoadingModel,
            stems: None,
            progress: 0.0,
        }));
        *mgr.native_status.lock() = Arc::clone(&status);
        let (reached_tx, reached_rx) = mpsc::channel::<()>();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let handle = std::thread::spawn(move || {
            status.lock().state = state;
            reached_tx.send(()).expect("the test is still listening");
            // The only way out, so the worker can never finish early and turn a red into a flake.
            let _ = release_rx.recv();
        });
        *mgr.active.lock() = Some(ActiveJob::NativeHandle {
            handle,
            cancel: Arc::new(AtomicBool::new(false)),
        });
        reached_rx.recv().expect("worker reached its state");
        release_tx
    }

    #[test]
    fn a_completed_run_is_reaped_so_the_next_dispatch_is_not_refused() {
        let dir = scratch_dir("reap_completed");
        let mgr = SeparationManager::new(dir.clone());
        let engine = OnnxEngine::new();
        let _release = park_worker_at(&mgr, SeparationState::Completed);

        // What the frontend sees at this instant: rejectIfSeparationBusy counts only
        // LoadingModel/Separating as busy, so it would dispatch the next node right now.
        let seen = mgr.status().state;
        assert!(
            matches!(seen, SeparationState::Completed),
            "FIXTURE BROKEN (not the code under test): the slot must read Completed, got {:?}",
            seen
        );

        let err = mgr
            .start(config_for(&dir, &dir.join("absent.onnx"), None), &engine)
            .expect_err("the model file does not exist, so this dispatch must fail somewhere")
            .to_string();

        assert!(
            !err.contains("SEPARATION_BUSY"),
            "a FINISHED separation must not make the next dispatch busy. 2026-09-09 field bug: the \
             worker writes Completed INSIDE its closure and JoinHandle::is_finished() only flips \
             AFTER that closure returns, so an is_finished()-only reaper always sits inside a window \
             where it refuses the next node and throws a finished run away. Got: {}",
            err
        );
        assert!(
            err.contains("MSST_MODEL_NOT_CONVERTED"),
            "the dispatch must get PAST the guard and reach model resolution — a guard that always \
             returned some OTHER error would satisfy the assertion above while still blocking every \
             run. Got: {}",
            err
        );
    }

    #[test]
    fn an_errored_run_is_reaped_too() {
        let dir = scratch_dir("reap_errored");
        let mgr = SeparationManager::new(dir.clone());
        let engine = OnnxEngine::new();
        let _release = park_worker_at(&mgr, SeparationState::Error("boom".to_string()));

        let err = mgr
            .start(config_for(&dir, &dir.join("absent.onnx"), None), &engine)
            .expect_err("the model file does not exist")
            .to_string();
        assert!(
            !err.contains("SEPARATION_BUSY"),
            "Error is a terminal slot state too — a failed run must not wedge the next dispatch. \
             Got: {}",
            err
        );
    }

    #[test]
    fn a_run_that_is_genuinely_in_flight_is_still_refused() {
        // The control that stops "fix it by deleting the guard" from passing. Green before AND
        // after the reap change; only a removed or weakened single-slot guard turns it red.
        for state in [SeparationState::LoadingModel, SeparationState::Separating] {
            let tag = format!("{:?}", state);
            let dir = scratch_dir(&format!("busy_{}", tag));
            let mgr = SeparationManager::new(dir.clone());
            let engine = OnnxEngine::new();
            let _release = park_worker_at(&mgr, state);

            let err = mgr
                .start(config_for(&dir, &dir.join("absent.onnx"), None), &engine)
                .expect_err("a worker is mid-run")
                .to_string();
            assert!(
                err.contains("SEPARATION_BUSY"),
                "the single-slot guard must still hold while a worker is genuinely mid-run ({}); \
                 got: {}",
                tag,
                err
            );
        }
    }

    #[test]
    fn cpu_preference_refuses_an_fp16_only_model() {
        let dir = scratch_dir("cpu_fp16_only");
        let fp32 = dir.join("base.onnx"); // deliberately NOT created
        std::fs::write(fp16_sibling(&fp32), b"stub").expect("write the fp16 stub");
        let mgr = SeparationManager::new(dir.clone());
        let engine = OnnxEngine::new();
        engine.set_device(DeviceConfig::Cpu);

        let err = mgr
            .start(config_for(&dir, &fp32, Some("fp16")), &engine)
            .expect_err("fp16 on the CPU provider must not start")
            .to_string();
        assert!(
            err.contains("MSST_FP16_ON_CPU"),
            "the CPU provider has no fp16 kernels; refuse with an actionable CODE instead of \
             entering the run that died silently on 2026-09-09. Got: {}",
            err
        );
    }

    #[test]
    fn a_gpu_preference_still_takes_the_fp16_only_model() {
        // Aetiology control: proves the refusal above keys on the DEVICE, not on the missing .json.
        let dir = scratch_dir("gpu_fp16_only");
        let fp32 = dir.join("base.onnx");
        std::fs::write(fp16_sibling(&fp32), b"stub").expect("write the fp16 stub");
        let mgr = SeparationManager::new(dir.clone());
        let engine = OnnxEngine::new();
        engine.set_device(DeviceConfig::DirectMl { device_id: 0 });

        let err = mgr
            .start(config_for(&dir, &fp32, Some("fp16")), &engine)
            .expect_err("there is no .json next to the stub")
            .to_string();
        assert!(!err.contains("MSST_FP16_ON_CPU"), "a GPU run must not be refused: {}", err);
        assert!(
            err.contains("base.fp16.onnx"),
            "and it must really have picked the fp16 file: {}",
            err
        );
    }

    #[test]
    fn cpu_preference_falls_back_to_the_fp32_sibling_when_it_exists() {
        let dir = scratch_dir("cpu_fp16_downgrade");
        let fp32 = dir.join("base.onnx");
        std::fs::write(&fp32, b"stub").expect("write the fp32 stub");
        std::fs::write(fp16_sibling(&fp32), b"stub").expect("write the fp16 stub");
        let mgr = SeparationManager::new(dir.clone());
        let engine = OnnxEngine::new();
        engine.set_device(DeviceConfig::Cpu);

        let err = mgr
            .start(config_for(&dir, &fp32, Some("fp16")), &engine)
            .expect_err("neither variant has a .json")
            .to_string();
        // The CODE's payload is the model the run WOULD have used, so it reports the choice made.
        assert!(
            !err.contains(".fp16.onnx"),
            "with an fp32 sibling on disk a CPU run must silently prefer it — neither refuse nor \
             keep fp16. Got: {}",
            err
        );
        assert!(err.contains("base.onnx"), "expected the fp32 path in the error: {}", err);
        // ⛔ Without these two, the case cannot tell a silent DOWNGRADE from a loud REFUSAL: flip the
        // two arms of the `fp32_path.exists()` branch and the assertions above still pass, because
        // `MSST_FP16_ON_CPU: <dir>ase.onnx` satisfies both of them. (Caught by the adversarial
        // review of this very test — the probe it shipped with was half vacuous.)
        assert!(
            !err.contains("MSST_FP16_ON_CPU"),
            "with the fp32 sibling present this must DOWNGRADE, not refuse: {}",
            err
        );
        assert!(
            err.contains("MSST_MODEL_NOT_CONVERTED"),
            "and it must get past the device check into model resolution: {}",
            err
        );
    }

    /// Every field of `SeparationConfig` must be read somewhere in this file. `device` was not.
    ///
    /// ⚠ Honest boundary (this repo's own wiring-gate rule): this proves the identifier appears in
    /// CODE, not that the read is on a reachable path. Its job is narrow — stop a field that
    /// silently ignores what the caller asked for from growing back.
    #[test]
    fn every_separation_config_field_is_actually_read() {
        // ⛔ Cut the test module off FIRST. This file's own fixtures mention `config.device`-shaped
        // identifiers in string literals, and a wiring gate must read production code only — the
        // same fail-closed cut `models/mod.rs` and `commands/audition.rs` already use, with the
        // reason written out there: without it any future fixture silently turns this gate green.
        let src = include_str!("mod.rs");
        let prod = src
            .split_once("#[cfg(test)]")
            .expect("GATE BROKEN: the test-module marker moved — fix this gate, do not delete it")
            .0;
        let code = crate::wiring_gate::strip_comments_for_wiring(prod);
        let decl = code
            .split_once("pub struct SeparationConfig {")
            .expect("GATE BROKEN: the struct header moved — fix this gate, do not delete it")
            .1
            .split_once("\n}")
            .expect("GATE BROKEN: unterminated struct body")
            .0;
        let fields: Vec<String> = decl
            .lines()
            .filter_map(|l| l.trim().strip_prefix("pub "))
            .filter_map(|l| l.split_once(':'))
            .map(|(n, _)| n.trim().to_string())
            .collect();
        assert!(
            fields.len() >= 10 && fields.iter().any(|f| f == "precision"),
            "GATE BROKEN: the field scanner found {:?}",
            fields
        );
        let unread: Vec<&String> = fields
            .iter()
            .filter(|f| !code.contains(&format!("config.{}", f)))
            .collect();
        assert!(
            unread.is_empty(),
            "SeparationConfig declares fields nothing in this file reads: {:?}. A field the caller \
             sets and the backend ignores is a silent lie — delete it or wire it.",
            unread
        );
    }
}
