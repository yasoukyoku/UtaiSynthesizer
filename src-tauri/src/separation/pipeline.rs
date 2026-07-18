use std::path::{Path, PathBuf};

use ndarray::Array3;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::stft::{self, StftProcessor};
use utai_dsp::{
    add_stems_into, assemble_cac, build_msst_window, compute_audio_mean_std, compute_residual,
    deinterleave_cac, demucs_ispec, demucs_spec, denormalize_stem, msst_chunk_audio,
    normalize_audio, prepare_padded_audio, shift_left, shift_right, sum_freq_time,
    AudioAccumulator,
};
use utai_dsp::mdx::{mdx_pad_mix, np_hanning, zero_low_bins};
use utai_dsp::vr;
// Re-export so the established external paths (pipeline::AudioData etc.) stay valid.
pub use utai_dsp::{AudioData, StemAudio};
use crate::inference::engine::{InputTensor, OnnxEngine};
use crate::{Result, UtaiError};

// ─── Per-model config JSON — written by the Python converter ─────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    #[serde(rename = "type")]
    pub model_type: String,
    pub sample_rate: u32,
    pub stereo: bool,
    pub num_stems: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    #[serde(default)]
    pub freq_bins: usize,
    #[serde(default)]
    pub dim_f: usize,
    #[serde(default)]
    pub num_subbands: usize,
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,
    #[serde(default = "default_num_overlap")]
    pub num_overlap: usize,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    /// Model was exported with a DYNAMIC batch axis (new converter marker). Old models lack this → the
    /// batched path is gated OFF and they keep the single-chunk loop (a batch-pinned ONNX hard-fails on B>1).
    #[serde(default)]
    pub dynamic_batch: bool,
    /// Runtime batch-size override (UI). Effective only on dynamic_batch models; else forced to 1.
    #[serde(default)]
    pub batch: usize,
    #[serde(default)]
    pub normalize: bool,
    /// Test-time augmentation (averaged augmented passes) — runtime override, not from model JSON.
    #[serde(default)]
    pub use_tta: bool,
    /// HTDemucs random time-shift passes — runtime override.
    #[serde(default)]
    pub shifts: usize,
    #[serde(default)]
    pub segment_samples: usize,
    #[serde(default)]
    pub processing_mode: ProcessingMode,
    /// Labels of the model's DIRECT outputs in training order (from the original yaml's
    /// training.instruments / target_instrument). Single-stem instrumental-target models
    /// (e.g. melband inst_v2) output the INSTRUMENTAL — without this the old heuristic
    /// labeled stem 0 "vocals" and content-swapped the two stems.
    #[serde(default)]
    pub stem_names: Option<Vec<String>>,
    /// Label of the mix-minus-stem residual for num_stems==1 models ("vocals" for
    /// instrumental-target models; "instrumental" otherwise).
    #[serde(default)]
    pub residual_name: Option<String>,
    // Legacy field — ignored if num_overlap is set
    #[serde(default)]
    pub overlap: usize,

    // ── UVR VR arch (type == "uvr_vr"; converter architectures/uvr_vr.py) ──
    /// Model window in STFT frames (crops fed to the net; T is STATIC in the export).
    #[serde(default)]
    pub window_size: usize,
    /// predict_mask time crop per side (v5.0: 128, v5.1: 64) — baked into the ONNX
    /// output width (window_size − 2·offset = roi).
    #[serde(default)]
    pub offset: usize,
    /// Combined multiband bin count (model input freq dim = bins + 1).
    #[serde(default)]
    pub bins: usize,
    #[serde(default)]
    pub is_v51: bool,
    #[serde(default)]
    pub pre_filter_start: i64,
    #[serde(default)]
    pub pre_filter_stop: i64,
    /// v5.0 GLOBAL waveform-domain channel transforms (6_HP uses mid_side_b2).
    #[serde(default)]
    pub reverse: bool,
    #[serde(default)]
    pub mid_side: bool,
    #[serde(default)]
    pub mid_side_b2: bool,
    /// Multiband table (lowest→highest); presence gates the VR path's sanity check.
    #[serde(default)]
    pub bands: Option<Vec<VrBandConfig>>,
    /// Aggressiveness split row = band 1's crop_stop (exponent differs below/above).
    #[serde(default)]
    pub aggr_split_bin: usize,
    /// Primary stem ∈ UVR's NON_ACCOM_STEMS (flips the aggressiveness exponent).
    #[serde(default)]
    pub primary_non_accom: bool,
    /// Runtime: UI aggression (−100..100, UVR default 5 → mask exponent tweak).
    #[serde(default)]
    pub aggression: Option<i32>,
    /// Runtime: UVR post-process (merge_artifacts) toggle + threshold (0.1/0.2/0.3).
    #[serde(default)]
    pub post_process: bool,
    #[serde(default)]
    pub post_process_threshold: Option<f32>,

    // ── Legacy MDX-Net (type == "mdx_net"; converter architectures/mdx_net.py) ──
    /// Static time frames of the graph (256 for the KARA models).
    #[serde(default)]
    pub dim_t: usize,
    /// UVR per-model output gain (applied to the primary stem BEFORE the residual).
    #[serde(default)]
    pub compensate: Option<f32>,
}

/// One VR band's DSP params (converter json `bands` entry; mirrors
/// `utai_dsp::vr::VrBandParam`). Filter keys are absent where the reference
/// JSON omits them (band 1 has no hpf, the top band no lpf).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VrBandConfig {
    pub sr: u32,
    pub hl: usize,
    pub n_fft: usize,
    pub crop_start: usize,
    pub crop_stop: usize,
    #[serde(default)]
    pub hpf_start: Option<i64>,
    #[serde(default)]
    pub hpf_stop: Option<i64>,
    #[serde(default)]
    pub lpf_start: Option<i64>,
    #[serde(default)]
    pub lpf_stop: Option<i64>,
    #[serde(default)]
    pub convert_channels: Option<String>,
}

fn default_chunk_size() -> usize { 131584 }
fn default_num_overlap() -> usize { 2 }
fn default_batch_size() -> usize { 1 }

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessingMode {
    #[default]
    Spectral,
    Waveform,
    Hybrid,
}

pub struct NativePipeline {
    engine: *const OnnxEngine,
    session_id: String,
    config: ModelConfig,
}

unsafe impl Send for NativePipeline {}
unsafe impl Sync for NativePipeline {}

/// Config JSON for a model file. fp16 variants (`<stem>.fp16.onnx`) share the fp32 model's
/// `<stem>.json` — ONE json per model so the two precisions can never drift apart.
pub fn model_config_path(model_path: &Path) -> PathBuf {
    let direct = model_path.with_extension("json"); // x.onnx → x.json; x.fp16.onnx → x.fp16.json
    if direct.exists() {
        return direct;
    }
    if let Some(name) = model_path.file_name().and_then(|n| n.to_str()) {
        if let Some(base) = name.strip_suffix(".fp16.onnx") {
            return model_path.with_file_name(format!("{base}.json"));
        }
    }
    direct
}

impl NativePipeline {
    pub fn new(engine: &OnnxEngine, model_path: &Path) -> Result<Self> {
        let config_path = model_config_path(model_path);
        if !config_path.exists() {
            return Err(UtaiError::Audio(format!(
                "Model config not found: {}", config_path.display()
            )));
        }
        let config_text = std::fs::read_to_string(&config_path)?;
        let config: ModelConfig = serde_json::from_str(&config_text)?;
        let t_load = std::time::Instant::now();
        let session_id = engine.load_model(&model_path.to_path_buf())?;
        tracing::debug!("[perf] session build (load_model): {:.1} ms", t_load.elapsed().as_secs_f64() * 1e3);
        Ok(Self { engine: engine as *const OnnxEngine, session_id, config })
    }

    fn engine(&self) -> &OnnxEngine { unsafe { &*self.engine } }
    pub fn config(&self) -> &ModelConfig { &self.config }
    pub fn session_id(&self) -> &str { &self.session_id }

    /// Override the normalize flag (UI checkbox takes precedence over the model JSON default).
    pub fn set_normalize(&mut self, v: bool) {
        self.config.normalize = v;
    }

    /// Override the overlap-add window count (UI). Clamped to ≥1.
    pub fn set_num_overlap(&mut self, n: usize) {
        self.config.num_overlap = n.max(1);
    }

    /// Override the inference batch size (UI). Only takes effect on dynamic-batch models; clamped ≥1.
    pub fn set_batch(&mut self, n: usize) {
        self.config.batch = n.max(1);
    }

    /// Effective batch size for inference: the runtime override on dynamic-batch models, else 1 (old
    /// models pin batch=1 in the ONNX and would hard-fail on B>1).
    fn effective_batch(&self) -> usize {
        if self.config.dynamic_batch { self.config.batch.max(1) } else { 1 }
    }

    /// Enable test-time augmentation (UI).
    pub fn set_use_tta(&mut self, v: bool) {
        self.config.use_tta = v;
    }

    /// Set HTDemucs random time-shift passes (UI).
    pub fn set_shifts(&mut self, n: usize) {
        self.config.shifts = n;
    }

    /// VR aggressiveness (UI, −100..100; UVR default 5). No-op for other archs.
    pub fn set_aggression(&mut self, v: i32) {
        self.config.aggression = Some(v);
    }

    /// VR post-process (merge_artifacts) toggle + threshold (UI). No-op for other archs.
    pub fn set_post_process(&mut self, on: bool, threshold: Option<f32>) {
        self.config.post_process = on;
        if threshold.is_some() {
            self.config.post_process_threshold = threshold;
        }
    }

    /// Run separation. Returns (stem_label, audio_data) pairs.
    /// `progress_cb` returns `true` to continue, `false` to cancel.
    ///
    /// When `normalize` is on, the input is mean/std-normalized before inference and the
    /// stems are denormalized afterwards (MSST convention) — keeps quiet/loud inputs in the
    /// model's expected range.
    pub fn separate(
        &self,
        audio: &AudioData,
        progress_cb: &dyn Fn(f32) -> bool,
    ) -> Result<Vec<StemAudio>> {
        // Mono input to a stereo model: duplicate to stereo, matching original MSST
        // (msst_infer.py:278-284). Without this every stereo path's is_stereo gate would
        // feed a mono-shaped tensor into a stereo graph and fail at inference. Done at
        // THIS single point so normalize / TTA / residual all see the same stereo mix.
        let mono_duplicated;
        let audio = if self.config.stereo && audio.channels == 1 {
            tracing::info!("Mono input to a stereo model — duplicating to stereo (MSST semantics)");
            mono_duplicated = AudioData {
                left: audio.left.clone(),
                right: audio.left.clone(),
                channels: 2,
                sample_rate: audio.sample_rate,
            };
            &mono_duplicated
        } else {
            audio
        };

        // htdemucs (hybrid demucs) is NOT mean/std-normalized in MSST, and its node UI hides the
        // Normalize toggle — so a stale `normalize=true` carried over from a spectral model on the
        // same node must not apply here. Guarding at this single point keeps the UI and engine honest.
        // uvr_vr normalizes its own magnitude globally (reference semantics) and mdx_net has no
        // input normalization at all — both must ignore a stale flag the same way.
        let skip_normalize = matches!(self.config.model_type.as_str(), "htdemucs" | "uvr_vr" | "mdx_net");
        let mut stems = if self.config.normalize && !skip_normalize {
            let (mean, std) = compute_audio_mean_std(audio);
            tracing::info!("MSST normalize ON (mean={:.6}, std={:.6})", mean, std);
            let normed = normalize_audio(audio, mean, std);
            let mut stems = self.run_augmented(&normed, progress_cb)?;
            for stem in &mut stems {
                denormalize_stem(stem, mean, std);
            }
            stems
        } else {
            self.run_augmented(audio, progress_cb)?
        };

        // Single-stem models: derive the residual (mix - stem) HERE — against the ORIGINAL
        // un-normalized mix, AFTER denormalization and TTA averaging. Computing it inside the
        // per-path functions against the normalized mix and then denormalizing left a +mean DC
        // offset (stem + residual summed to mix + mean). Hybrid (htdemucs) never derives a
        // residual — unchanged. Empty stems = empty input (0 chunks), nothing to derive.
        if self.config.num_stems == 1
            && !matches!(self.config.processing_mode, ProcessingMode::Hybrid)
            && !stems.is_empty()
        {
            stems.push(compute_residual(audio, &stems[0], &residual_label(&self.config)));
        }
        Ok(stems)
    }

    /// Test-time augmentation wrapper. Dispatches over a set of input transforms (identity +
    /// optional polarity-flip / channel-swap for `use_tta`, + optional time-shifts for HTDemucs
    /// `shifts`), inverts each result back to the original frame, and averages them.
    ///
    /// With NO augmentation enabled (no TTA, no shifts) `augs` is a single identity pass and this
    /// calls `dispatch` directly — byte-for-byte the previous behavior (zero regression risk).
    fn run_augmented(
        &self,
        audio: &AudioData,
        progress_cb: &dyn Fn(f32) -> bool,
    ) -> Result<Vec<StemAudio>> {
        let stereo = self.config.stereo && audio.channels == 2;
        let mut augs: Vec<Aug> = vec![Aug::Identity];
        // uvr_vr implements the ORIGINAL VR TTA (a half-window-shifted second mask pass,
        // averaged in mask domain) inside separate_vr — the generic polarity/channel-swap
        // passes here are NOT what UVR's TTA option means and must not stack on top.
        if self.config.use_tta && self.config.model_type != "uvr_vr" {
            augs.push(Aug::Polarity);
            if stereo {
                augs.push(Aug::ChannelSwap);
            }
        }
        if self.config.model_type == "htdemucs" && self.config.shifts > 0 {
            let max_shift = (audio.sample_rate as usize / 2).max(1); // ~0.5s (MSST demucs convention)
            let n = self.config.shifts;
            for i in 0..n {
                // Deterministic, evenly-spaced offsets in (0, max_shift) — reproducible (vs random).
                let off = (max_shift * (i + 1) / (n + 1)).max(1);
                augs.push(Aug::Shift(off));
            }
        }

        let num_passes = augs.len();
        if num_passes == 1 {
            return self.dispatch(audio, progress_cb);
        }
        tracing::info!("MSST TTA: averaging {} augmented passes", num_passes);

        // Per-sample contribution count, NOT a flat 1/num_passes. A Shift pass's inverse (shift_left)
        // zero-fills the last `off` output samples, so that pass contributes nothing there; dividing
        // those samples by num_passes would silently attenuate the stem's tail (up to ~0.5s). Each
        // pass adds 1.0 over the region it actually fills; we divide by that per-sample count.
        let orig_len = audio.left.len();
        let mut weights = vec![0.0f32; orig_len];
        let mut acc: Option<Vec<StemAudio>> = None;
        for (i, aug) in augs.iter().enumerate() {
            let transformed = aug.apply(audio);
            let base = i as f32;
            let span = num_passes as f32;
            let sub_cb = |p: f32| progress_cb((base + p) / span);
            let mut stems = self.dispatch(&transformed, &sub_cb)?;
            for stem in &mut stems {
                aug.invert_stem(stem);
            }
            let valid = aug.valid_len(orig_len).min(orig_len);
            for w in weights[..valid].iter_mut() {
                *w += 1.0;
            }
            match acc.as_mut() {
                None => acc = Some(stems),
                Some(a) => add_stems_into(a, &stems),
            }
        }

        let mut result = acc.expect("num_passes >= 1");
        for stem in &mut result {
            for (x, &w) in stem.left.iter_mut().zip(weights.iter()) {
                if w > 0.0 {
                    *x /= w;
                }
            }
            for (x, &w) in stem.right.iter_mut().zip(weights.iter()) {
                if w > 0.0 {
                    *x /= w;
                }
            }
        }
        Ok(result)
    }

    fn dispatch(
        &self,
        audio: &AudioData,
        progress_cb: &dyn Fn(f32) -> bool,
    ) -> Result<Vec<StemAudio>> {
        match self.config.processing_mode {
            ProcessingMode::Spectral => match self.config.model_type.as_str() {
                "mdx23c" => self.separate_mdx23c(audio, progress_cb),
                "uvr_vr" => self.separate_vr(audio, progress_cb),
                "mdx_net" => self.separate_mdx_net(audio, progress_cb),
                _ => self.separate_spectral(audio, progress_cb),
            },
            ProcessingMode::Waveform => self.separate_waveform(audio, progress_cb),
            ProcessingMode::Hybrid => self.separate_hybrid(audio, progress_cb),
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Spectral separation (BSRoformer, MelBandRoformer) — MSST-compatible
    // ═══════════════════════════════════════════════════════════════
    fn separate_spectral(
        &self,
        audio: &AudioData,
        progress_cb: &dyn Fn(f32) -> bool,
    ) -> Result<Vec<StemAudio>> {
        let proc = StftProcessor::new(stft::StftConfig {
            n_fft: self.config.n_fft,
            hop_length: self.config.hop_length,
            win_length: self.config.win_length,
        });

        let is_stereo = self.config.stereo && audio.channels == 2;
        let freq_dim = if is_stereo { proc.freq_bins() * 2 } else { proc.freq_bins() };

        // ── MSST-compatible chunking with border padding ──
        let chunk_size = self.config.chunk_size;
        let num_overlap = self.config.num_overlap.max(1);
        let step = chunk_size / num_overlap;
        let fade_size = chunk_size / 10;

        let (padded_left, padded_right, border, orig_len) =
            prepare_padded_audio(audio, chunk_size, step);

        let chunks = msst_chunk_audio(&padded_left, &padded_right, is_stereo, chunk_size, step);
        let total_chunks = chunks.len();
        if total_chunks == 0 { return Ok(vec![]); }

        let window = build_msst_window(chunk_size, fade_size);

        let t_total = std::time::Instant::now();
        // ── Phase 1: Parallel STFT on all chunks ──
        tracing::info!("STFT: {} chunks (parallel)...", total_chunks);
        let t_stft = std::time::Instant::now();
        let spectrograms: Vec<(Array3<f32>, usize)> = chunks.par_iter().map(|chunk| {
            let spec = if is_stereo {
                proc.stft_stereo(&chunk.left, &chunk.right)
            } else {
                proc.stft(&chunk.left)
            };
            let num_frames = spec.shape()[1];
            (spec, num_frames)
        }).collect();
        tracing::debug!("[perf] STFT phase: {:.1} ms ({} chunks)", t_stft.elapsed().as_secs_f64() * 1e3, total_chunks);

        if !progress_cb(0.05) {
            return Err(UtaiError::Audio("Stopped by user".into()));
        }

        // ── Phase 2: GPU inference + immediate iSTFT + overlap-add (merged) ──
        // Official MSST does STFT→model→iSTFT all on GPU, then CPU just accumulates.
        // We do STFT on CPU (Phase 1), but merge inference+iSTFT so there's no
        // separate CPU phase after GPU finishes — iSTFT runs between GPU calls.
        tracing::info!("Inference + iSTFT: {} chunks...", total_chunks);
        let padded_len = padded_left.len();
        let mut stem_accumulators: Vec<AudioAccumulator> = (0..self.config.num_stems)
            .map(|_| AudioAccumulator::new(padded_len, is_stereo))
            .collect();

        // Inference in batches of `b` chunks (b=1 on old/batch-pinned models → exact single-chunk path).
        // All chunks are padded to chunk_size → identical num_frames, so a batch stacks rectangularly.
        let b = self.effective_batch();
        tracing::info!(
            "MSST spectral inference: batch={} (dynamic_batch={}, config.batch={}), {} chunks",
            b, self.config.dynamic_batch, self.config.batch, total_chunks
        );
        let mut chunk_idx = 0;
        // Phase timing accumulators — summarized once per run (tracing) for perf regressions.
        let mut perf_assemble = 0.0f64;
        let mut perf_infer = 0.0f64;
        let mut perf_infer_min = f64::MAX;
        let mut perf_infer_max = 0.0f64;
        let mut perf_runs = 0usize;
        let mut perf_post = 0.0f64;
        while chunk_idx < total_chunks {
            let bg = b.min(total_chunks - chunk_idx); // group size, remainder-safe

            let t_asm = std::time::Instant::now();
            // Stack bg spectrograms into one [bg, freq_dim, num_frames, 2] input.
            let num_frames = spectrograms[chunk_idx].1;
            let chunk_floats = freq_dim * num_frames * 2;
            let mut input_data: Vec<f32> = Vec::with_capacity(bg * chunk_floats);
            for j in 0..bg {
                input_data.extend(spectrograms[chunk_idx + j].0.iter().copied());
            }
            let input = InputTensor::F32 {
                data: input_data,
                shape: vec![bg as i64, freq_dim as i64, num_frames as i64, 2],
            };
            perf_assemble += t_asm.elapsed().as_secs_f64();
            let t_run = std::time::Instant::now();
            let outputs = self.engine().run(&self.session_id, vec![("stft_repr", input)])?;
            let dt_run = t_run.elapsed().as_secs_f64();
            perf_infer += dt_run;
            perf_infer_min = perf_infer_min.min(dt_run);
            perf_infer_max = perf_infer_max.max(dt_run);
            perf_runs += 1;
            let t_post = std::time::Instant::now();
            let output_data = outputs.into_iter().next().ok_or_else(||
                UtaiError::Audio("Model produced no output".into()))?;

            // Unbind: output is [bg, num_stems, freq_dim, num_frames, 2]; item j stem s starts at
            // (j*num_stems + s) * chunk_floats. At bg==1 this reduces to stem_idx*chunk_floats (old path).
            for j in 0..bg {
                let ci = chunk_idx + j;
                let (spec, nf) = &spectrograms[ci];
                let offset = chunks[ci].offset;
                let chunk_len = chunks[ci].left.len();

                let mut win = window.clone();
                if ci == 0 {
                    for i in 0..fade_size.min(win.len()) { win[i] = 1.0; }
                }
                if ci == total_chunks - 1 {
                    let wl = win.len();
                    for i in 0..fade_size.min(wl) { win[wl - 1 - i] = 1.0; }
                }

                for stem_idx in 0..self.config.num_stems {
                    let base = (j * self.config.num_stems + stem_idx) * chunk_floats;
                    let mask = Array3::from_shape_vec(
                        (freq_dim, *nf, 2),
                        output_data[base..base + chunk_floats].to_vec(),
                    ).map_err(|e| UtaiError::Audio(format!("Mask reshape: {}", e)))?;

                    let masked = stft::apply_complex_mask(spec, &mask);

                    if is_stereo {
                        let (left, right) = proc.istft_stereo(&masked, chunk_len);
                        stem_accumulators[stem_idx].add_windowed_stereo(&left, &right, offset, &win);
                    } else {
                        let mono = proc.istft(&masked, chunk_len);
                        stem_accumulators[stem_idx].add_windowed_mono(&mono, offset, &win);
                    }
                }
            }

            perf_post += t_post.elapsed().as_secs_f64();

            chunk_idx += bg;
            let progress = 0.05 + 0.90 * chunk_idx as f32 / total_chunks as f32;
            if !progress_cb(progress) {
                return Err(UtaiError::Audio("Stopped by user".into()));
            }
        }
        // One summary line per run. avg-vs-max spread flags the CUDA-arena/WDDM-paging slow mode
        // (S31 perf investigation: same config runs at ~890 ms or 2.6-3.7 s per chunk when VRAM fills).
        tracing::info!(
            "[perf] inference: {:.1} ms total over {} runs (avg {:.1} / min {:.1} / max {:.1} ms per run, batch={})",
            perf_infer * 1e3, perf_runs, perf_infer * 1e3 / perf_runs.max(1) as f64,
            perf_infer_min * 1e3, perf_infer_max * 1e3, b
        );
        tracing::debug!("[perf] input assemble: {:.1} ms total", perf_assemble * 1e3);
        tracing::debug!("[perf] iSTFT+mask+overlap-add (post): {:.1} ms total", perf_post * 1e3);

        progress_cb(1.0);

        let t_fin = std::time::Instant::now();
        let stems: Vec<StemAudio> = stem_accumulators.into_iter().enumerate().map(|(i, acc)| {
            let label = stem_label(&self.config, i);
            acc.finalize_with_border(label, border, orig_len)
        }).collect();
        // Residual for num_stems==1 is derived in separate() against the un-normalized mix.
        tracing::debug!("[perf] finalize: {:.1} ms", t_fin.elapsed().as_secs_f64() * 1e3);
        tracing::info!(
            "[perf] separate_spectral TOTAL: {:.1} ms ({})",
            t_total.elapsed().as_secs_f64() * 1e3,
            crate::inference::engine::memory_stamp()
        );
        Ok(stems)
    }

    // ═══════════════════════════════════════════════════════════════
    // MDX23C — CaC spectral, MSST-compatible phased pipeline
    // ═══════════════════════════════════════════════════════════════
    fn separate_mdx23c(
        &self,
        audio: &AudioData,
        progress_cb: &dyn Fn(f32) -> bool,
    ) -> Result<Vec<StemAudio>> {
        let proc = StftProcessor::new(stft::StftConfig {
            n_fft: self.config.n_fft,
            hop_length: self.config.hop_length,
            win_length: self.config.win_length,
        });

        let dim_f = self.config.dim_f;
        let chunk_size = self.config.chunk_size;
        let num_overlap = self.config.num_overlap.max(1);
        let step = chunk_size / num_overlap;
        let fade_size = chunk_size / 10;

        let (padded_left, padded_right, border, orig_len) =
            prepare_padded_audio(audio, chunk_size, step);

        let chunks = msst_chunk_audio(&padded_left, &padded_right, true, chunk_size, step);
        let total_chunks = chunks.len();
        if total_chunks == 0 { return Ok(vec![]); }

        let window = build_msst_window(chunk_size, fade_size);

        let t_total = std::time::Instant::now();
        // Phase 1: Parallel STFT (L/R separate for CaC)
        tracing::info!("MDX23C phase 1: {} STFTs (parallel)...", total_chunks);
        let t_stft = std::time::Instant::now();
        let stft_pairs: Vec<(Array3<f32>, Array3<f32>, usize)> = chunks.par_iter().map(|chunk| {
            let spec_l = proc.stft(&chunk.left);
            let spec_r = proc.stft(&chunk.right);
            let num_frames = spec_l.shape()[1];
            (spec_l, spec_r, num_frames)
        }).collect();
        tracing::debug!("[perf] STFT phase: {:.1} ms ({} chunks)", t_stft.elapsed().as_secs_f64() * 1e3, total_chunks);

        if !progress_cb(0.05) {
            return Err(UtaiError::Audio("Stopped by user".into()));
        }

        // Phase 2: GPU inference ‖ post-processing (double-buffered). The main thread
        // assembles CaC inputs and runs inference; a worker thread consumes each chunk's
        // output IN ORDER (sync_channel(1) FIFO, bounded memory) and does the reshape +
        // iSTFT + overlap-add, with the independent stems processed in parallel (their
        // accumulators are disjoint &mut via par_iter_mut). Each stem's accumulation
        // sequence is unchanged (chunk order preserved by the channel, stem-local adds)
        // → BIT-EXACT vs the old serial loop (S32 md5 gate); wall ≈ max(GPU, CPU-post)
        // instead of their sum. Peak RAM stays O(one chunk in flight per side).
        tracing::info!("MDX23C phase 2: inference ‖ iSTFT on {} chunks...", total_chunks);
        let padded_len = padded_left.len();
        let num_stems = self.config.num_stems;

        let mut perf_assemble = 0.0f64;
        let mut perf_infer = 0.0f64;
        let mut perf_infer_min = f64::MAX;
        let mut perf_infer_max = 0.0f64;
        let (stem_accumulators, perf_post) =
            std::thread::scope(|s| -> Result<(Vec<AudioAccumulator>, f64)> {
                let proc_ref = &proc;
                let window_ref = &window;
                // (chunk_idx, f, num_frames, offset, chunk_len, output_data)
                type Item = (usize, usize, usize, usize, usize, Vec<f32>);
                let (tx, rx) = std::sync::mpsc::sync_channel::<Item>(1);
                let worker = s.spawn(move || {
                    let mut accs: Vec<AudioAccumulator> =
                        (0..num_stems).map(|_| AudioAccumulator::new(padded_len, true)).collect();
                    let mut perf_post = 0.0f64;
                    for (chunk_idx, f, num_frames, offset, chunk_len, output_data) in rx {
                        let t_post = std::time::Instant::now();
                        let stem_size = 4 * f * num_frames;

                        let mut win = window_ref.clone();
                        if chunk_idx == 0 {
                            for i in 0..fade_size.min(win.len()) { win[i] = 1.0; }
                        }
                        if chunk_idx == total_chunks - 1 {
                            let wl = win.len();
                            for i in 0..fade_size.min(wl) { win[wl - 1 - i] = 1.0; }
                        }

                        accs.par_iter_mut().enumerate().for_each(|(stem_idx, acc)| {
                            let stem_off = stem_idx * stem_size;
                            // Full freq_bins output (nyquist row stays zero) — the model
                            // emits only f rows.
                            let (spec_out_l, spec_out_r) = deinterleave_cac(
                                &output_data, stem_off, f, num_frames, proc_ref.freq_bins());
                            let left = proc_ref.istft(&spec_out_l, chunk_len);
                            let right = proc_ref.istft(&spec_out_r, chunk_len);
                            acc.add_windowed_stereo(&left, &right, offset, &win);
                        });
                        perf_post += t_post.elapsed().as_secs_f64();
                    }
                    (accs, perf_post)
                });

                for (chunk_idx, (spec_l, spec_r, num_frames)) in stft_pairs.into_iter().enumerate() {
                    let t_asm = std::time::Instant::now();
                    let f = dim_f.min(spec_l.shape()[0]);
                    let cac_data = assemble_cac(&spec_l, &spec_r, f, num_frames);
                    // STFT pair consumed — free it before inference (the model outputs the full
                    // spectrum directly, so unlike the mask-based spectral path it's not needed again).
                    drop(spec_l);
                    drop(spec_r);

                    let input = InputTensor::F32 {
                        data: cac_data,
                        shape: vec![1, 4, f as i64, num_frames as i64],
                    };
                    perf_assemble += t_asm.elapsed().as_secs_f64();
                    let t_run = std::time::Instant::now();
                    let outputs = self.engine().run(&self.session_id, vec![("stft_repr", input)])?;
                    let dt_run = t_run.elapsed().as_secs_f64();
                    perf_infer += dt_run;
                    perf_infer_min = perf_infer_min.min(dt_run);
                    perf_infer_max = perf_infer_max.max(dt_run);
                    let output_data = outputs.into_iter().next().ok_or_else(||
                        UtaiError::Audio("Model produced no output".into()))?;

                    let offset = chunks[chunk_idx].offset;
                    let chunk_len = chunks[chunk_idx].left.len();
                    if tx.send((chunk_idx, f, num_frames, offset, chunk_len, output_data)).is_err() {
                        break; // worker died — its panic resurfaces at the join below
                    }

                    let progress = 0.05 + 0.85 * (chunk_idx + 1) as f32 / total_chunks as f32;
                    if !progress_cb(progress) {
                        // tx drops on return → worker drains its ≤1 buffered item and exits;
                        // the scope joins it on the way out.
                        return Err(UtaiError::Audio("Stopped by user".into()));
                    }
                }
                drop(tx); // close the channel so the worker drains and exits
                worker.join().map(Ok).unwrap_or_else(|_| {
                    Err(UtaiError::Audio("MDX23C post worker panicked".into()))
                })
            })?;

        progress_cb(1.0);

        tracing::info!(
            "[perf] inference: {:.1} ms total over {} runs (avg {:.1} / min {:.1} / max {:.1} ms per run)",
            perf_infer * 1e3, total_chunks, perf_infer * 1e3 / total_chunks.max(1) as f64,
            perf_infer_min * 1e3, perf_infer_max * 1e3
        );
        tracing::debug!("[perf] CaC assemble: {:.1} ms total", perf_assemble * 1e3);
        tracing::debug!("[perf] reshape+iSTFT+overlap-add (post, worker ‖ per-stem par, overlaps inference): {:.1} ms total", perf_post * 1e3);

        let t_fin = std::time::Instant::now();
        let stems: Vec<StemAudio> = stem_accumulators.into_iter().enumerate().map(|(i, acc)| {
            let label = stem_label(&self.config, i);
            acc.finalize_with_border(label, border, orig_len)
        }).collect();
        // Residual for num_stems==1 is derived in separate() against the un-normalized mix.
        tracing::debug!("[perf] finalize: {:.1} ms", t_fin.elapsed().as_secs_f64() * 1e3);
        tracing::info!("[perf] separate_mdx23c TOTAL: {:.1} ms ({} chunks)", t_total.elapsed().as_secs_f64() * 1e3, total_chunks);
        Ok(stems)
    }

    // ═══════════════════════════════════════════════════════════════
    // HTDemucs hybrid — MSST uses simple averaging (no windowing)
    // ═══════════════════════════════════════════════════════════════
    fn separate_hybrid(
        &self,
        audio: &AudioData,
        progress_cb: &dyn Fn(f32) -> bool,
    ) -> Result<Vec<StemAudio>> {
        let proc = StftProcessor::new(stft::StftConfig {
            n_fft: self.config.n_fft,
            hop_length: self.config.hop_length,
            win_length: self.config.win_length,
        });
        let freq_bins = self.config.n_fft / 2;
        let segment_samples = self.config.segment_samples;
        if segment_samples == 0 {
            return Err(UtaiError::Audio("HTDemucs requires segment_samples in config".into()));
        }

        let orig_len = audio.left.len();
        let hop = self.config.hop_length;
        let le = (segment_samples + hop - 1) / hop;

        // MSST: HTDemucs uses num_overlap for step, simple counter accumulation
        let num_overlap = self.config.num_overlap.max(1);
        let step = (segment_samples / num_overlap).max(1);

        // MSST demix (utils/utils.py:144-157) visits EVERY step start while pos < len —
        // including tail starts whose window crosses EOF (those chunks are zero-padded to
        // segment_samples and averaged by the accumulator counter) — so the segment count
        // is ceil(len / step), never "1 if it fits".
        let total_segments = ((orig_len + step - 1) / step).max(1);
        let mut seg_idx = 0;
        let mut pos = 0;
        let num_stems = self.config.num_stems;

        let t_total = std::time::Instant::now();
        // GPU inference ‖ post-processing (double-buffered), same construction as the
        // mdx23c path: main thread does demucs_spec×2 + input assembly + inference; the
        // worker consumes segments IN ORDER (sync_channel(1) FIFO) and runs the per-stem
        // reshape + demucs_ispec×2 + branch-sum + accumulate with stems in parallel
        // (disjoint per-stem accumulators). Per-stem accumulation order is unchanged →
        // BIT-EXACT vs the old serial loop (S32 CPU-EP md5 gate; GPU md5 is NOT a valid
        // gate for htdemucs — its CUDA runs are nondeterministic run-to-run).
        let mut perf_pre = 0.0f64;
        let mut perf_infer = 0.0f64;
        let mut perf_infer_min = f64::MAX;
        let mut perf_infer_max = 0.0f64;

        let (stem_accumulators, perf_post) =
            std::thread::scope(|s| -> Result<(Vec<AudioAccumulator>, f64)> {
                let proc_ref = &proc;
                // (pos, chunk_len, f, num_frames, outputs=[freq_data, time_data])
                type Item = (usize, usize, usize, usize, Vec<Vec<f32>>);
                let (tx, rx) = std::sync::mpsc::sync_channel::<Item>(1);
                let worker = s.spawn(move || {
                    let mut accs: Vec<AudioAccumulator> =
                        (0..num_stems).map(|_| AudioAccumulator::new(orig_len, true)).collect();
                    let mut perf_post = 0.0f64;
                    for (pos, chunk_len, f, num_frames, outputs) in rx {
                        let t_post = std::time::Instant::now();
                        let freq_data = &outputs[0];
                        let time_data = &outputs[1];
                        let freq_stem_size = 4 * f * num_frames;
                        let time_stem_size = 2 * segment_samples;

                        accs.par_iter_mut().enumerate().for_each(|(stem_idx, acc)| {
                            let foff = stem_idx * freq_stem_size;
                            let (spec_out_l, spec_out_r) =
                                deinterleave_cac(freq_data, foff, f, num_frames, f);
                            // demucs `_ispec` — re-adds the nyquist row + cropped edge
                            // frames as zeros.
                            let freq_l = demucs_ispec(proc_ref, &spec_out_l, segment_samples);
                            let freq_r = demucs_ispec(proc_ref, &spec_out_r, segment_samples);

                            let toff = stem_idx * time_stem_size;
                            let time_l = &time_data[toff..toff + segment_samples];
                            let time_r = &time_data[toff + segment_samples..toff + time_stem_size];

                            let stem_l = sum_freq_time(&freq_l, time_l, chunk_len);
                            let stem_r = sum_freq_time(&freq_r, time_r, chunk_len);

                            // MSST: HTDemucs uses simple accumulation + counter (no windowing)
                            acc.add_simple_stereo(&stem_l, &stem_r, pos);
                        });
                        perf_post += t_post.elapsed().as_secs_f64();
                    }
                    (accs, perf_post)
                });

                while pos < orig_len {
                    let end = (pos + segment_samples).min(orig_len);
                    let chunk_len = end - pos;

                    let mut left_chunk = vec![0.0f32; segment_samples];
                    let mut right_chunk = vec![0.0f32; segment_samples];
                    left_chunk[..chunk_len].copy_from_slice(&audio.left[pos..end]);
                    if audio.channels == 2 && audio.right.len() >= end {
                        right_chunk[..chunk_len].copy_from_slice(&audio.right[pos..end]);
                    }

                    let t_pre = std::time::Instant::now();
                    // demucs `_spec` convention — NOT a plain centered STFT (see demucs_spec;
                    // feeding proc.stft directly costs ~80 dB of stem SNR, measured 6.22 vs 86.22 dB).
                    let spec_l = demucs_spec(&proc, &left_chunk);
                    let spec_r = demucs_spec(&proc, &right_chunk);
                    debug_assert_eq!(spec_l.shape()[1], le);
                    let num_frames = spec_l.shape()[1].min(le);
                    let f = freq_bins.min(spec_l.shape()[0]);

                    let cac_data = assemble_cac(&spec_l, &spec_r, f, num_frames);

                    let mut mix_data = Vec::with_capacity(2 * segment_samples);
                    mix_data.extend_from_slice(&left_chunk);
                    mix_data.extend_from_slice(&right_chunk);

                    let cac_input = InputTensor::F32 {
                        data: cac_data,
                        shape: vec![1, 4, f as i64, num_frames as i64],
                    };
                    let mix_input = InputTensor::F32 {
                        data: mix_data,
                        shape: vec![1, 2, segment_samples as i64],
                    };

                    perf_pre += t_pre.elapsed().as_secs_f64();
                    let t_run = std::time::Instant::now();
                    let outputs = self.engine().run(
                        &self.session_id,
                        vec![("cac_spec", cac_input), ("mix", mix_input)],
                    )?;
                    let dt_run = t_run.elapsed().as_secs_f64();
                    perf_infer += dt_run;
                    perf_infer_min = perf_infer_min.min(dt_run);
                    perf_infer_max = perf_infer_max.max(dt_run);

                    if tx.send((pos, chunk_len, f, num_frames, outputs)).is_err() {
                        break; // worker died — its panic resurfaces at the join below
                    }

                    seg_idx += 1;
                    if !progress_cb(seg_idx as f32 / total_segments as f32) {
                        // tx drops on return → worker drains its ≤1 buffered item and exits;
                        // the scope joins it on the way out.
                        return Err(UtaiError::Audio("Stopped by user".into()));
                    }

                    // No early break: MSST demix visits every step start < orig_len (tail chunks
                    // were zero-padded above; the accumulator counter averages the overlap).
                    pos += step;
                }
                drop(tx); // close the channel so the worker drains and exits
                worker.join().map(Ok).unwrap_or_else(|_| {
                    Err(UtaiError::Audio("HTDemucs post worker panicked".into()))
                })
            })?;

        tracing::info!(
            "[perf] inference: {:.1} ms total over {} segments (avg {:.1} / min {:.1} / max {:.1} ms per run)",
            perf_infer * 1e3, seg_idx, perf_infer * 1e3 / (seg_idx.max(1)) as f64,
            perf_infer_min * 1e3, perf_infer_max * 1e3
        );
        tracing::debug!("[perf] demucs_spec×2+assemble (pre, serial): {:.1} ms total", perf_pre * 1e3);
        tracing::debug!(
            "[perf] reshape+demucs_ispec×{}+sum (post, worker ‖ per-stem par, overlaps inference): {:.1} ms total",
            self.config.num_stems * 2, perf_post * 1e3
        );

        let t_fin = std::time::Instant::now();
        let stems: Vec<StemAudio> = stem_accumulators.into_iter().enumerate().map(|(i, acc)| {
            let label = stem_label(&self.config, i);
            acc.finalize(label)
        }).collect();
        tracing::debug!("[perf] finalize: {:.1} ms", t_fin.elapsed().as_secs_f64() * 1e3);
        tracing::info!("[perf] separate_hybrid TOTAL: {:.1} ms ({} segments)", t_total.elapsed().as_secs_f64() * 1e3, seg_idx);

        Ok(stems)
    }

    // ═══════════════════════════════════════════════════════════════
    // UVR VR arch — whole-song multiband magnitude masking
    // (reference: vr_separator.py + spec_utils.py; see utai_dsp::vr)
    // ═══════════════════════════════════════════════════════════════
    fn separate_vr(
        &self,
        audio: &AudioData,
        progress_cb: &dyn Fn(f32) -> bool,
    ) -> Result<Vec<StemAudio>> {
        let bands_cfg = self.config.bands.as_ref().ok_or_else(|| {
            UtaiError::Audio("uvr_vr model json is missing the `bands` table".into())
        })?;
        if bands_cfg.is_empty() || self.config.bins == 0 || self.config.window_size == 0 {
            return Err(UtaiError::Audio("uvr_vr model json has invalid VR params".into()));
        }
        let params = vr::VrParams {
            bins: self.config.bins,
            pre_filter_start: self.config.pre_filter_start,
            pre_filter_stop: self.config.pre_filter_stop,
            is_v51: self.config.is_v51,
            reverse: self.config.reverse,
            mid_side: self.config.mid_side,
            mid_side_b2: self.config.mid_side_b2,
            bands: bands_cfg
                .iter()
                .map(|b| vr::VrBandParam {
                    sr: b.sr,
                    hl: b.hl,
                    n_fft: b.n_fft,
                    crop_start: b.crop_start,
                    crop_stop: b.crop_stop,
                    hpf_start: b.hpf_start,
                    hpf_stop: b.hpf_stop,
                    lpf_start: b.lpf_start,
                    lpf_stop: b.lpf_stop,
                    convert_channels: b.convert_channels.clone(),
                })
                .collect(),
            window_size: self.config.window_size,
            offset: self.config.offset,
            aggr_split_bin: self.config.aggr_split_bin,
            primary_non_accom: self.config.primary_non_accom,
        };
        let orig_len = audio.left.len();
        if orig_len == 0 {
            return Ok(vec![]);
        }

        let t_total = std::time::Instant::now();
        // ── Phase A: multiband analysis (CPU) ──
        let t_ana = std::time::Instant::now();
        let combined = vr::vr_analyze(&params, &audio.left, &audio.right);
        let (mag, phase) = vr::vr_mag_phase(&combined);
        drop(combined); // masking reconstructs from mag·e^{iφ}, the complex spec isn't reused
        tracing::debug!("[perf] VR analysis: {:.1} ms", t_ana.elapsed().as_secs_f64() * 1e3);
        let n_frame = mag.shape()[2];
        if n_frame == 0 {
            return Ok(vec![]);
        }
        if !progress_cb(0.05) {
            return Err(UtaiError::Audio("Stopped by user".into()));
        }

        // ── Phase B: window-batched mask inference ──
        let (pad_l, pad_r, roi) = vr::make_padding(n_frame, params.window_size, params.offset);
        let t_inf = std::time::Instant::now();
        let mask_full = {
            let padded = vr::pad_and_normalize_mag(&mag, pad_l, pad_r);
            let p1 = if self.config.use_tta { 0.40 } else { 0.72 };
            self.vr_execute_mask(&padded, roi, &params, 0.05, p1, progress_cb)?
        };
        // Crop to n_frame; TTA = a second pass shifted by roi/2 (original VR semantics),
        // averaged with the first IN MASK DOMAIN.
        let mut mask = ndarray::Array3::<f32>::zeros((2, params.bins + 1, n_frame));
        for ch in 0..2 {
            for f in 0..params.bins + 1 {
                for t in 0..n_frame {
                    mask[[ch, f, t]] = mask_full[[ch, f, t]];
                }
            }
        }
        drop(mask_full);
        if self.config.use_tta {
            let padded = vr::pad_and_normalize_mag(&mag, pad_l + roi / 2, pad_r + roi / 2);
            let mask_tta = self.vr_execute_mask(&padded, roi, &params, 0.40, 0.72, progress_cb)?;
            let shift = roi / 2;
            for ch in 0..2 {
                for f in 0..params.bins + 1 {
                    for t in 0..n_frame {
                        mask[[ch, f, t]] = (mask[[ch, f, t]] + mask_tta[[ch, f, shift + t]]) * 0.5;
                    }
                }
            }
        }
        tracing::debug!("[perf] VR mask inference: {:.1} ms", t_inf.elapsed().as_secs_f64() * 1e3);

        // ── Phase C: mask post-processing + apply ──
        // UI aggression (UVR default 5) → value = aggression/100; exponent flip for
        // non-accompaniment primaries (none of the shipped 9, but registry-driven).
        let aggression = self.config.aggression.unwrap_or(5);
        vr::adjust_aggr(
            &mut mask,
            self.config.primary_non_accom,
            aggression as f64 / 100.0,
            params.aggr_split_bin,
        );
        if self.config.post_process {
            let thres = self.config.post_process_threshold.unwrap_or(0.2);
            tracing::info!("VR post-process (merge_artifacts) ON, threshold {}", thres);
            vr::merge_artifacts(&mut mask, thres, 64, 32);
        }
        let (y_spec, v_spec) = vr::vr_apply_mask(&mask, &mag, &phase);
        drop(mask);
        drop(mag);
        drop(phase);
        if !progress_cb(0.75) {
            return Err(UtaiError::Audio("Stopped by user".into()));
        }

        // ── Phase D: per-stem multiband synthesis (CPU; the two stems are independent) ──
        let t_syn = std::time::Instant::now();
        let ((yl, yr), (vl, vr_)) = rayon::join(
            || vr::vr_synthesize(&params, &y_spec),
            || vr::vr_synthesize(&params, &v_spec),
        );
        tracing::debug!("[perf] VR synthesis: {:.1} ms", t_syn.elapsed().as_secs_f64() * 1e3);
        progress_cb(1.0);

        // Reference output is hl_top·(frames−1) samples (≤ input by < one hop);
        // zero-pad the tail so stems align sample-exact with the DAW's source lane.
        let fit = |mut v: Vec<f32>| {
            v.resize(orig_len, 0.0);
            v
        };
        let stems = vec![
            StemAudio {
                label: stem_label(&self.config, 0),
                left: fit(yl),
                right: fit(yr),
                channels: 2,
            },
            StemAudio {
                label: stem_label(&self.config, 1),
                left: fit(vl),
                right: fit(vr_),
                channels: 2,
            },
        ];
        tracing::info!(
            "[perf] separate_vr TOTAL: {:.1} ms ({} frames; {})",
            t_total.elapsed().as_secs_f64() * 1e3, n_frame,
            crate::inference::engine::memory_stamp()
        );
        Ok(stems)
    }

    /// One full masking pass over the padded magnitude: slide `window_size`-wide crops
    /// at stride `roi`, batch them, run the ONNX mask net (output width = roi — the
    /// offset crop is baked into the graph), concat along time.
    fn vr_execute_mask(
        &self,
        padded: &ndarray::Array3<f32>,
        roi: usize,
        params: &vr::VrParams,
        p0: f32,
        p1: f32,
        progress_cb: &dyn Fn(f32) -> bool,
    ) -> Result<ndarray::Array3<f32>> {
        let bins1 = padded.shape()[1];
        let total_t = padded.shape()[2];
        let ws = params.window_size;
        let patches = (total_t - 2 * params.offset) / roi;
        if patches == 0 {
            return Err(UtaiError::Audio("VR window error: input too short".into()));
        }
        let mut mask = ndarray::Array3::<f32>::zeros((2, bins1, patches * roi));
        let b = self.effective_batch();
        let win_floats = 2 * bins1 * ws;
        let out_floats = 2 * bins1 * roi;
        let mut done = 0usize;
        while done < patches {
            let bg = b.min(patches - done);
            let mut input_data = vec![0.0f32; bg * win_floats];
            for j in 0..bg {
                vr::copy_mag_window(
                    padded,
                    (done + j) * roi,
                    ws,
                    &mut input_data[j * win_floats..(j + 1) * win_floats],
                );
            }
            let input = InputTensor::F32 {
                data: input_data,
                shape: vec![bg as i64, 2, bins1 as i64, ws as i64],
            };
            let outputs = self.engine().run(&self.session_id, vec![("mag", input)])?;
            let out = outputs.into_iter().next().ok_or_else(|| {
                UtaiError::Audio("VR model produced no output".into())
            })?;
            if out.len() < bg * out_floats {
                return Err(UtaiError::Audio(format!(
                    "VR mask output too small: {} < {}", out.len(), bg * out_floats
                )));
            }
            for j in 0..bg {
                let t0 = (done + j) * roi;
                for ch in 0..2 {
                    for f in 0..bins1 {
                        let base = ((j * 2 + ch) * bins1 + f) * roi;
                        for i in 0..roi {
                            mask[[ch, f, t0 + i]] = out[base + i];
                        }
                    }
                }
            }
            done += bg;
            let p = p0 + (p1 - p0) * done as f32 / patches as f32;
            if !progress_cb(p) {
                return Err(UtaiError::Audio("Stopped by user".into()));
            }
        }
        Ok(mask)
    }

    // ═══════════════════════════════════════════════════════════════
    // Legacy MDX-Net (UVR KARA models) — direct-spectrogram nets
    // (reference: UVR SeperateMDX / audio-separator mdx_separator.py)
    // ═══════════════════════════════════════════════════════════════
    fn separate_mdx_net(
        &self,
        audio: &AudioData,
        progress_cb: &dyn Fn(f32) -> bool,
    ) -> Result<Vec<StemAudio>> {
        let proc = StftProcessor::new(stft::StftConfig {
            n_fft: self.config.n_fft,
            hop_length: self.config.hop_length,
            win_length: self.config.win_length,
        });
        let dim_f = self.config.dim_f;
        let dim_t = self.config.dim_t;
        if dim_f == 0 || dim_t == 0 {
            return Err(UtaiError::Audio("mdx_net model json needs dim_f/dim_t".into()));
        }
        let hop = self.config.hop_length;
        let trim = self.config.n_fft / 2;
        // chunk = hop·(dim_t−1): the graph's T axis is static, so the geometry is fixed.
        let chunk_size = hop * (dim_t - 1);
        let gen_size = chunk_size - 2 * trim;
        let num_overlap = self.config.num_overlap.max(1);
        let step = (chunk_size / num_overlap).max(1);
        let compensate = self.config.compensate.unwrap_or(1.0);
        let orig_len = audio.left.len();
        if orig_len == 0 {
            return Ok(vec![]);
        }

        let t_total = std::time::Instant::now();
        let (pl, pr) = mdx_pad_mix(&audio.left, &audio.right, trim, gen_size);
        let padded_len = pl.len();
        let mut acc = AudioAccumulator::new(padded_len, true);
        let starts: Vec<usize> = (0..padded_len).step_by(step).collect();
        let total = starts.len();
        let mut perf_infer = 0.0f64;

        for (i, &start) in starts.iter().enumerate() {
            let end = (start + chunk_size).min(padded_len);
            let actual = end - start;
            let mut cl = vec![0.0f32; chunk_size];
            let mut cr = vec![0.0f32; chunk_size];
            cl[..actual].copy_from_slice(&pl[start..end]);
            cr[..actual].copy_from_slice(&pr[start..end]);

            let spec_l = proc.stft(&cl);
            let spec_r = proc.stft(&cr);
            debug_assert_eq!(spec_l.shape()[1], dim_t);
            let mut cac = assemble_cac(&spec_l, &spec_r, dim_f, dim_t);
            zero_low_bins(&mut cac, dim_f, dim_t); // spek[:, :, :3, :] *= 0 (both references)

            let input = InputTensor::F32 {
                data: cac,
                shape: vec![1, 4, dim_f as i64, dim_t as i64],
            };
            let t_run = std::time::Instant::now();
            let outputs = self.engine().run(&self.session_id, vec![("input", input)])?;
            perf_infer += t_run.elapsed().as_secs_f64();
            let out = outputs.into_iter().next().ok_or_else(|| {
                UtaiError::Audio("MDX-Net model produced no output".into())
            })?;

            // Output IS the primary stem's spectrogram (no mask). Rows ≥ dim_f
            // (up to nyquist) stay zero, matching the reference zero-pad.
            let (sl, sr_) = deinterleave_cac(&out, 0, dim_f, dim_t, proc.freq_bins());
            let left = proc.istft(&sl, chunk_size);
            let right = proc.istft(&sr_, chunk_size);

            // hanning(actual) OLA — result/divider semantics via the accumulator.
            let window = np_hanning(actual);
            acc.add_windowed_stereo(&left[..actual], &right[..actual], start, &window);

            if !progress_cb((i + 1) as f32 / total as f32) {
                return Err(UtaiError::Audio("Stopped by user".into()));
            }
        }

        // result/divider → strip the leading trim pad → crop to input length,
        // then UVR's compensate gain on the primary (the residual in separate()
        // is computed AFTER this — mix − compensated primary, UVR semantics).
        let mut stem = acc.finalize_with_border(stem_label(&self.config, 0), trim, orig_len);
        for x in stem.left.iter_mut() {
            *x *= compensate;
        }
        for x in stem.right.iter_mut() {
            *x *= compensate;
        }
        tracing::info!(
            "[perf] separate_mdx_net TOTAL: {:.1} ms ({} chunks, inference {:.1} ms)",
            t_total.elapsed().as_secs_f64() * 1e3, total, perf_infer * 1e3
        );
        Ok(vec![stem])
    }

    fn separate_waveform(
        &self,
        audio: &AudioData,
        progress_cb: &dyn Fn(f32) -> bool,
    ) -> Result<Vec<StemAudio>> {
        let is_stereo = self.config.stereo && audio.channels == 2;
        let chunk_size = self.config.chunk_size;
        let num_overlap = self.config.num_overlap.max(1);
        let step = chunk_size / num_overlap;
        let fade_size = chunk_size / 10;

        let (padded_left, padded_right, border, orig_len) =
            prepare_padded_audio(audio, chunk_size, step);

        let chunks = msst_chunk_audio(&padded_left, &padded_right, is_stereo, chunk_size, step);
        let total_chunks = chunks.len();

        let window = build_msst_window(chunk_size, fade_size);

        let mut stem_accumulators: Vec<AudioAccumulator> = (0..self.config.num_stems)
            .map(|_| AudioAccumulator::new(padded_left.len(), is_stereo))
            .collect();

        for (chunk_idx, chunk) in chunks.iter().enumerate() {
            let channels = if is_stereo { 2i64 } else { 1i64 };
            let chunk_len = chunk.left.len() as i64;

            let input_data = if is_stereo {
                let mut data = Vec::with_capacity(chunk.left.len() * 2);
                data.extend_from_slice(&chunk.left);
                data.extend_from_slice(&chunk.right);
                data
            } else {
                chunk.left.clone()
            };

            let input = InputTensor::F32 {
                data: input_data,
                shape: vec![1, channels, chunk_len],
            };
            let outputs = self.engine().run(&self.session_id, vec![("audio", input)])?;
            let output_data = &outputs[0];
            let samples_per_stem = (channels as usize) * (chunk_len as usize);

            let mut win = window.clone();
            if chunk_idx == 0 {
                for i in 0..fade_size.min(win.len()) { win[i] = 1.0; }
            }
            if chunk_idx == total_chunks - 1 {
                let wl = win.len();
                for i in 0..fade_size.min(wl) { win[wl - 1 - i] = 1.0; }
            }

            for stem_idx in 0..self.config.num_stems {
                let stem_offset = stem_idx * samples_per_stem;
                if is_stereo {
                    let left = &output_data[stem_offset..stem_offset + chunk_len as usize];
                    let right = &output_data[stem_offset + chunk_len as usize..stem_offset + samples_per_stem];
                    stem_accumulators[stem_idx].add_windowed_stereo(left, right, chunk.offset, &win);
                } else {
                    let mono = &output_data[stem_offset..stem_offset + chunk_len as usize];
                    stem_accumulators[stem_idx].add_windowed_mono(mono, chunk.offset, &win);
                }
            }

            if !progress_cb((chunk_idx + 1) as f32 / total_chunks as f32) {
                return Err(UtaiError::Audio("Stopped by user".into()));
            }
        }

        let stems: Vec<StemAudio> = stem_accumulators.into_iter().enumerate().map(|(i, acc)| {
            let label = stem_label(&self.config, i);
            acc.finalize_with_border(label, border, orig_len)
        }).collect();
        // Residual for num_stems==1 is derived in separate() against the un-normalized mix.
        Ok(stems)
    }

    pub fn unload(&self) { self.engine().unload_model(&self.session_id); }
}

// NOTE: NativePipeline intentionally does NOT unload its session on Drop. Sessions are
// cached + LRU-bounded by OnnxEngine so that repeated separations of the same model reuse
// the already-optimized session instead of reloading from disk every run. Call `unload()`
// or OnnxEngine::clear_sessions() to free explicitly.

// ─── Test-time augmentation (TTA) ────────────────────────────────
//
// Each pass transforms the input, runs the model, then inverts the transform on the OUTPUT so all
// passes line up before averaging. Polarity/channel-swap are general (any architecture); shifts are
// HTDemucs-only (matching MSST). Only the model's DIRECT outputs are averaged here — the derived
// mix-minus-stem residual is computed once in separate(), against the original mix, after averaging
// and denormalization (per-sample: avg over passes of (mix - stem) == mix - avg(stem), since every
// pass's inverted mix contribution is the mix itself over the region it fills).

enum Aug {
    Identity,
    Polarity,
    ChannelSwap,
    Shift(usize),
}

impl Aug {
    /// Transform the input for this pass (length preserved).
    fn apply(&self, a: &AudioData) -> AudioData {
        match self {
            Aug::Identity => a.clone(),
            Aug::Polarity => AudioData {
                left: a.left.iter().map(|x| -x).collect(),
                right: a.right.iter().map(|x| -x).collect(),
                channels: a.channels,
                sample_rate: a.sample_rate,
            },
            Aug::ChannelSwap => AudioData {
                left: a.right.clone(),
                right: a.left.clone(),
                channels: a.channels,
                sample_rate: a.sample_rate,
            },
            Aug::Shift(off) => AudioData {
                left: shift_right(&a.left, *off),
                right: shift_right(&a.right, *off),
                channels: a.channels,
                sample_rate: a.sample_rate,
            },
        }
    }

    /// Undo the transform on the model's OUTPUT so every pass aligns before averaging.
    fn invert_stem(&self, s: &mut StemAudio) {
        match self {
            Aug::Identity => {}
            Aug::Polarity => {
                for x in s.left.iter_mut() {
                    *x = -*x;
                }
                for x in s.right.iter_mut() {
                    *x = -*x;
                }
            }
            Aug::ChannelSwap => std::mem::swap(&mut s.left, &mut s.right),
            Aug::Shift(off) => {
                s.left = shift_left(&s.left, *off);
                s.right = shift_left(&s.right, *off);
            }
        }
    }

    /// Number of leading output samples this pass actually fills. A Shift's inverse (shift_left)
    /// zero-fills the last `off` samples, which must be excluded from the TTA average.
    fn valid_len(&self, n: usize) -> usize {
        match self {
            Aug::Shift(off) => n.saturating_sub(*off),
            _ => n,
        }
    }
}

/// Stem label for the model's direct output `stem_idx`: the config's `stem_names` (from the
/// original training yaml) when present, else the legacy per-arch heuristic below. The heuristic
/// is WRONG for instrumental-target single-stem models and for MSST-trained htdemucs (which
/// trains [vocals, other], not the official [drums, bass, other, vocals]) — new conversions
/// always carry stem_names.
fn stem_label(config: &ModelConfig, stem_idx: usize) -> String {
    if let Some(names) = &config.stem_names {
        if let Some(name) = names.get(stem_idx) {
            return name.clone();
        }
    }
    default_stem_label(&config.model_type, stem_idx, config.num_stems)
}

/// Label for the derived mix-minus-stem residual (num_stems==1 models only).
fn residual_label(config: &ModelConfig) -> String {
    config.residual_name.clone().unwrap_or_else(|| "instrumental".to_string())
}

fn default_stem_label(model_type: &str, stem_idx: usize, num_stems: usize) -> String {
    match model_type {
        "bs_roformer" | "mel_band_roformer" | "mdx23c" => {
            if num_stems == 1 {
                match stem_idx { 0 => "vocals".to_string(), _ => format!("stem_{}", stem_idx) }
            } else {
                match stem_idx { 0 => "vocals".to_string(), 1 => "instrumental".to_string(), _ => format!("stem_{}", stem_idx) }
            }
        }
        "htdemucs" => match stem_idx {
            0 => "drums".to_string(), 1 => "bass".to_string(),
            2 => "other".to_string(), 3 => "vocals".to_string(),
            _ => format!("stem_{}", stem_idx),
        },
        _ => format!("stem_{}", stem_idx),
    }
}

// ─── WAV I/O ─────────────────────────────────────────────────────

pub fn load_wav(path: &Path) -> Result<AudioData> {
    let reader = hound::WavReader::open(path)
        .map_err(|e| UtaiError::Audio(format!("Failed to open WAV: {}", e)))?;

    let spec = reader.spec();
    let channels = spec.channels as usize;
    let sample_rate = spec.sample_rate;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader.into_samples::<i32>().filter_map(|s| s.ok()).map(|s| s as f32 / max_val).collect()
        }
        hound::SampleFormat::Float => {
            reader.into_samples::<f32>().filter_map(|s| s.ok()).collect()
        }
    };

    let num_samples = samples.len() / channels;
    let (left, right) = if channels >= 2 {
        let mut l = Vec::with_capacity(num_samples);
        let mut r = Vec::with_capacity(num_samples);
        for i in 0..num_samples {
            l.push(samples[i * channels]);
            r.push(samples[i * channels + 1]);
        }
        (l, r)
    } else {
        (samples, vec![])
    };

    Ok(AudioData { left, right, channels: channels.min(2), sample_rate })
}

pub fn save_wav(path: &Path, stem: &StemAudio, sample_rate: u32) -> Result<()> {
    // 32-bit float (not 16-bit int) so the separated stems keep FULL precision — the model works in
    // f32 and the source may be 24/32-bit; quantizing to 16-bit here would add audible noise to a result
    // the user may process further. No clamping: preserve any inter-sample overshoot for downstream gain.
    //
    // S68c: non-finite samples ARE scrubbed to 0.0 (finite values pass through bit-exact — this is
    // NOT the clamping ruled out above). fp16 models on true-fp16 GPU kernels (DirectML) can emit
    // NaN/Inf, nothing upstream checks, and a poisoned stem wav nuked the voice pipeline behind it
    // (the 0.5.0 20% abort). Scrubbing at THE stem-write funnel keeps every consumer clean —
    // playback, chained MSST nodes, RVC/SoVITS.
    let spec = hound::WavSpec {
        channels: stem.channels as u16,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| UtaiError::Audio(format!("Failed to create WAV: {}", e)))?;

    let mut bad = 0usize;
    let mut scrub = |s: f32| {
        if s.is_finite() {
            s
        } else {
            bad += 1;
            0.0
        }
    };
    if stem.channels == 2 {
        for i in 0..stem.left.len() {
            writer.write_sample(scrub(stem.left[i]))
                .map_err(|e| UtaiError::Audio(format!("WAV write: {}", e)))?;
            writer.write_sample(scrub(stem.right[i]))
                .map_err(|e| UtaiError::Audio(format!("WAV write: {}", e)))?;
        }
    } else {
        for &s in &stem.left {
            writer.write_sample(scrub(s))
                .map_err(|e| UtaiError::Audio(format!("WAV write: {}", e)))?;
        }
    }
    if bad > 0 {
        tracing::warn!(
            "separation stem {} contained {} non-finite sample(s) (NaN/Inf) — zeroed on write (fp16 GPU numeric fault upstream?)",
            path.display(),
            bad
        );
    }

    writer.finalize().map_err(|e| UtaiError::Audio(format!("WAV finalize: {}", e)))?;
    Ok(())
}
