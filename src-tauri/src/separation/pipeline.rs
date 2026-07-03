use std::path::{Path, PathBuf};

use ndarray::Array3;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::stft::{self, StftProcessor};
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
        // htdemucs (hybrid demucs) is NOT mean/std-normalized in MSST, and its node UI hides the
        // Normalize toggle — so a stale `normalize=true` carried over from a spectral model on the
        // same node must not apply here. Guarding at this single point keeps the UI and engine honest.
        let mut stems = if self.config.normalize && self.config.model_type != "htdemucs" {
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
        if self.config.use_tta {
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
            ProcessingMode::Spectral => {
                if self.config.model_type == "mdx23c" {
                    self.separate_mdx23c(audio, progress_cb)
                } else {
                    self.separate_spectral(audio, progress_cb)
                }
            }
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
        tracing::info!("[perf] separate_spectral TOTAL: {:.1} ms", t_total.elapsed().as_secs_f64() * 1e3);
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

        // Phase 1: Parallel STFT (L/R separate for CaC)
        tracing::info!("MDX23C phase 1: {} STFTs (parallel)...", total_chunks);
        let stft_pairs: Vec<(Array3<f32>, Array3<f32>, usize)> = chunks.par_iter().map(|chunk| {
            let spec_l = proc.stft(&chunk.left);
            let spec_r = proc.stft(&chunk.right);
            let num_frames = spec_l.shape()[1];
            (spec_l, spec_r, num_frames)
        }).collect();

        if !progress_cb(0.05) {
            return Err(UtaiError::Audio("Stopped by user".into()));
        }

        // Phase 2: GPU inference + immediate iSTFT + overlap-add (merged — same pattern as
        // separate_spectral). Each chunk's inference output AND its STFT pair are dropped as
        // soon as they're consumed, so peak RAM is O(one chunk) instead of retaining every
        // inference output (+ all STFT pairs) until a separate phase 3.
        tracing::info!("MDX23C phase 2: inference + iSTFT on {} chunks...", total_chunks);
        let padded_len = padded_left.len();
        let mut stem_accumulators: Vec<AudioAccumulator> = (0..self.config.num_stems)
            .map(|_| AudioAccumulator::new(padded_len, true))
            .collect();

        for (chunk_idx, (spec_l, spec_r, num_frames)) in stft_pairs.into_iter().enumerate() {
            let f = dim_f.min(spec_l.shape()[0]);
            let mut cac_data = Vec::with_capacity(4 * f * num_frames);
            for ch_spec in [&spec_l, &spec_r] {
                for ri in 0..2 {
                    for freq in 0..f {
                        for t in 0..num_frames {
                            cac_data.push(ch_spec[[freq, t, ri]]);
                        }
                    }
                }
            }
            // STFT pair consumed — free it before inference (the model outputs the full
            // spectrum directly, so unlike the mask-based spectral path it's not needed again).
            drop(spec_l);
            drop(spec_r);

            let input = InputTensor::F32 {
                data: cac_data,
                shape: vec![1, 4, f as i64, num_frames as i64],
            };
            let outputs = self.engine().run(&self.session_id, vec![("stft_repr", input)])?;
            let output_data = outputs.into_iter().next().ok_or_else(||
                UtaiError::Audio("Model produced no output".into()))?;

            // iSTFT + windowed overlap-add for this chunk (identical numerics to the old
            // phase 3 — same window variants for first/last chunk, same accumulation).
            let offset = chunks[chunk_idx].offset;
            let chunk_len = chunks[chunk_idx].left.len();
            let chan_size = f * num_frames;
            let stem_size = 4 * chan_size;

            let mut win = window.clone();
            if chunk_idx == 0 {
                for i in 0..fade_size.min(win.len()) { win[i] = 1.0; }
            }
            if chunk_idx == total_chunks - 1 {
                let wl = win.len();
                for i in 0..fade_size.min(wl) { win[wl - 1 - i] = 1.0; }
            }

            for stem_idx in 0..self.config.num_stems {
                let stem_off = stem_idx * stem_size;
                let freq_bins = proc.freq_bins();
                let mut spec_out_l = Array3::<f32>::zeros((freq_bins, num_frames, 2));
                let mut spec_out_r = Array3::<f32>::zeros((freq_bins, num_frames, 2));

                for freq in 0..f {
                    for t in 0..num_frames {
                        let ft = freq * num_frames + t;
                        spec_out_l[[freq, t, 0]] = output_data[stem_off + ft];
                        spec_out_l[[freq, t, 1]] = output_data[stem_off + chan_size + ft];
                        spec_out_r[[freq, t, 0]] = output_data[stem_off + 2 * chan_size + ft];
                        spec_out_r[[freq, t, 1]] = output_data[stem_off + 3 * chan_size + ft];
                    }
                }

                let left = proc.istft(&spec_out_l, chunk_len);
                let right = proc.istft(&spec_out_r, chunk_len);
                stem_accumulators[stem_idx].add_windowed_stereo(&left, &right, offset, &win);
            }

            let progress = 0.05 + 0.85 * (chunk_idx + 1) as f32 / total_chunks as f32;
            if !progress_cb(progress) {
                return Err(UtaiError::Audio("Stopped by user".into()));
            }
        }

        progress_cb(1.0);

        let stems: Vec<StemAudio> = stem_accumulators.into_iter().enumerate().map(|(i, acc)| {
            let label = stem_label(&self.config, i);
            acc.finalize_with_border(label, border, orig_len)
        }).collect();
        // Residual for num_stems==1 is derived in separate() against the un-normalized mix.
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

        let mut stem_accumulators: Vec<AudioAccumulator> = (0..self.config.num_stems)
            .map(|_| AudioAccumulator::new(orig_len, true))
            .collect();

        // MSST demix (utils/utils.py:144-157) visits EVERY step start while pos < len —
        // including tail starts whose window crosses EOF (those chunks are zero-padded to
        // segment_samples and averaged by the accumulator counter) — so the segment count
        // is ceil(len / step), never "1 if it fits".
        let total_segments = ((orig_len + step - 1) / step).max(1);
        let mut seg_idx = 0;
        let mut pos = 0;

        while pos < orig_len {
            let end = (pos + segment_samples).min(orig_len);
            let chunk_len = end - pos;

            let mut left_chunk = vec![0.0f32; segment_samples];
            let mut right_chunk = vec![0.0f32; segment_samples];
            left_chunk[..chunk_len].copy_from_slice(&audio.left[pos..end]);
            if audio.channels == 2 && audio.right.len() >= end {
                right_chunk[..chunk_len].copy_from_slice(&audio.right[pos..end]);
            }

            // demucs `_spec` convention — NOT a plain centered STFT (see demucs_spec below;
            // feeding proc.stft directly costs ~80 dB of stem SNR, measured 6.22 vs 86.22 dB).
            let spec_l = demucs_spec(&proc, &left_chunk);
            let spec_r = demucs_spec(&proc, &right_chunk);
            debug_assert_eq!(spec_l.shape()[1], le);
            let num_frames = spec_l.shape()[1].min(le);
            let f = freq_bins.min(spec_l.shape()[0]);

            let mut cac_data = Vec::with_capacity(4 * f * num_frames);
            for ch_spec in [&spec_l, &spec_r] {
                for ri in 0..2 {
                    for freq in 0..f {
                        for t in 0..num_frames {
                            cac_data.push(ch_spec[[freq, t, ri]]);
                        }
                    }
                }
            }

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

            let outputs = self.engine().run(
                &self.session_id,
                vec![("cac_spec", cac_input), ("mix", mix_input)],
            )?;

            let freq_data = &outputs[0];
            let time_data = &outputs[1];

            let freq_stem_size = 4 * f * num_frames;
            let time_stem_size = 2 * segment_samples;

            for stem_idx in 0..self.config.num_stems {
                let foff = stem_idx * freq_stem_size;
                let chan_size = f * num_frames;
                let mut spec_out_l = Array3::<f32>::zeros((f, num_frames, 2));
                let mut spec_out_r = Array3::<f32>::zeros((f, num_frames, 2));
                for freq in 0..f {
                    for t in 0..num_frames {
                        let ft = freq * num_frames + t;
                        spec_out_l[[freq, t, 0]] = freq_data[foff + ft];
                        spec_out_l[[freq, t, 1]] = freq_data[foff + chan_size + ft];
                        spec_out_r[[freq, t, 0]] = freq_data[foff + 2 * chan_size + ft];
                        spec_out_r[[freq, t, 1]] = freq_data[foff + 3 * chan_size + ft];
                    }
                }
                // demucs `_ispec` — re-adds the nyquist row + cropped edge frames as zeros.
                let freq_l = demucs_ispec(&proc, &spec_out_l, segment_samples);
                let freq_r = demucs_ispec(&proc, &spec_out_r, segment_samples);

                let toff = stem_idx * time_stem_size;
                let time_l = &time_data[toff..toff + segment_samples];
                let time_r = &time_data[toff + segment_samples..toff + time_stem_size];

                let mut stem_l = Vec::with_capacity(chunk_len);
                let mut stem_r = Vec::with_capacity(chunk_len);
                for i in 0..chunk_len {
                    let fl = if i < freq_l.len() { freq_l[i] } else { 0.0 };
                    let fr = if i < freq_r.len() { freq_r[i] } else { 0.0 };
                    stem_l.push(fl + time_l[i]);
                    stem_r.push(fr + time_r[i]);
                }

                // MSST: HTDemucs uses simple accumulation + counter (no windowing)
                stem_accumulators[stem_idx].add_simple_stereo(&stem_l, &stem_r, pos);
            }

            seg_idx += 1;
            if !progress_cb(seg_idx as f32 / total_segments as f32) {
                return Err(UtaiError::Audio("Stopped by user".into()));
            }

            // No early break: MSST demix visits every step start < orig_len (tail chunks
            // were zero-padded above; the accumulator counter averages the overlap).
            pos += step;
        }

        let stems: Vec<StemAudio> = stem_accumulators.into_iter().enumerate().map(|(i, acc)| {
            let label = stem_label(&self.config, i);
            acc.finalize(label)
        }).collect();

        Ok(stems)
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

// ─── demucs `_spec` / `_ispec` STFT convention (HTDemucs) ────────
//
// HTDemucs' spectral branch does NOT consume a plain centered STFT. demucs
// (demucs4ht.py::_spec) reflect-pads the chunk by pad = hop/2*3 on the left and
// pad + (le*hop - length) on the right (le = ceil(length/hop)) BEFORE the centered
// STFT, then keeps frames [2 : 2+le] and drops the nyquist bin. Net effect: output
// frame j is centered at j*hop + hop/2 — whereas a plain centered STFT centers frame
// j at j*hop. This half-hop shift is what keeps the freq and time branches aligned
// inside the model; feeding a plain STFT costs ~80 dB of stem SNR (measured 6.22 dB
// vs 86.22 dB against the original torch pipeline).
//
// Scale invariant: demucs runs torch.stft(normalized=True), i.e. a 1/sqrt(n_fft)
// factor. That factor is deliberately OMITTED here: the exported graph normalizes the
// spec in-graph by its own mean/std and denormalizes its output, so a constant input
// scale k scales mean/std by k and the model output by k — which our matching
// unnormalized iSTFT then inverts exactly. Proven numerically (our spec == demucs
// _spec × sqrt(n_fft) bit-exact; end-to-end 86.22 dB). Do NOT add the factor.

/// demucs `_spec`: signal [T] → [n_fft/2 bins, ceil(T/hop) frames, 2] (nyquist dropped).
fn demucs_spec(proc: &StftProcessor, signal: &[f32]) -> Array3<f32> {
    let n_fft = proc.config().n_fft;
    let hop = proc.config().hop_length;
    debug_assert_eq!(n_fft, hop * 4, "demucs convention requires hop == n_fft/4");
    let t = signal.len();
    let le = (t + hop - 1) / hop;
    let pad = hop / 2 * 3;
    // Torch-exact asymmetric reflect pad (demucs pad1d); the extra le*hop - t on the
    // right rounds the signal up to a whole number of hops.
    let padded = reflect_pad_lr(signal, pad, pad + le * hop - t);
    // proc.stft additionally center-pads by n_fft/2 internally (stft.rs) — exactly like
    // torch.stft(center=True) inside demucs' spectro(). The demucs pad sits OUTSIDE that
    // center pad, so the padded signal yields exactly le + 4 frames.
    let full = proc.stft(&padded);
    debug_assert_eq!(full.shape()[1], le + 4);
    let bins = n_fft / 2; // drop the nyquist row
    let mut out = Array3::<f32>::zeros((bins, le, 2));
    for b in 0..bins {
        for fr in 0..le {
            out[[b, fr, 0]] = full[[b, fr + 2, 0]];
            out[[b, fr, 1]] = full[[b, fr + 2, 1]];
        }
    }
    out
}

/// demucs `_ispec`: spec [n_fft/2 bins, ceil(length/hop) frames, 2] → signal [length].
/// Re-adds a zero nyquist row + 2 zero frames per side (undoing the `[2 : 2+le]` crop),
/// runs the plain centered iSTFT over le*hop + 2*pad samples, then slices
/// [pad : pad+length]. The edge frames it re-inserts are zeros, so the outer pad region
/// is reconstructed from incomplete overlap-add — same as demucs; chunk overlap covers it.
fn demucs_ispec(proc: &StftProcessor, spec: &Array3<f32>, length: usize) -> Vec<f32> {
    let hop = proc.config().hop_length;
    let full_bins = proc.freq_bins(); // n_fft/2 + 1
    let bins = spec.shape()[0].min(full_bins);
    let frames = spec.shape()[1];
    debug_assert_eq!((length + hop - 1) / hop, frames, "frames must equal ceil(length/hop)");
    let pad = hop / 2 * 3;
    let mut padded = Array3::<f32>::zeros((full_bins, frames + 4, 2));
    for b in 0..bins {
        for fr in 0..frames {
            padded[[b, fr + 2, 0]] = spec[[b, fr, 0]];
            padded[[b, fr + 2, 1]] = spec[[b, fr, 1]];
        }
    }
    let le_len = hop * ((length + hop - 1) / hop) + 2 * pad;
    let x = proc.istft(&padded, le_len);
    let end = (pad + length).min(x.len());
    x[pad.min(end)..end].to_vec()
}

// ─── Input normalization (MSST-style mean/std, optional) ─────────

fn compute_audio_mean_std(audio: &AudioData) -> (f32, f32) {
    let n = audio.left.len();
    if n == 0 {
        return (0.0, 1.0);
    }
    let stereo = audio.channels >= 2 && audio.right.len() == n;
    let mono = |i: usize| if stereo { (audio.left[i] + audio.right[i]) * 0.5 } else { audio.left[i] };
    let mut sum = 0.0f64;
    for i in 0..n {
        sum += mono(i) as f64;
    }
    let mean = (sum / n as f64) as f32;
    let mut var = 0.0f64;
    for i in 0..n {
        let d = (mono(i) - mean) as f64;
        var += d * d;
    }
    let std = ((var / n as f64).sqrt() as f32).max(1e-8);
    (mean, std)
}

fn normalize_audio(audio: &AudioData, mean: f32, std: f32) -> AudioData {
    let inv = 1.0 / std;
    AudioData {
        left: audio.left.iter().map(|&x| (x - mean) * inv).collect(),
        right: audio.right.iter().map(|&x| (x - mean) * inv).collect(),
        channels: audio.channels,
        sample_rate: audio.sample_rate,
    }
}

fn denormalize_stem(stem: &mut StemAudio, mean: f32, std: f32) {
    for x in stem.left.iter_mut() {
        *x = *x * std + mean;
    }
    for x in stem.right.iter_mut() {
        *x = *x * std + mean;
    }
}

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

/// Shift right by `off` samples (zero-fill the front), length preserved.
fn shift_right(x: &[f32], off: usize) -> Vec<f32> {
    if off == 0 || x.is_empty() {
        return x.to_vec();
    }
    let n = x.len();
    let off = off.min(n);
    let mut out = vec![0.0f32; n];
    out[off..].copy_from_slice(&x[..n - off]);
    out
}

/// Shift left by `off` samples (zero-fill the tail) — exact inverse of `shift_right`.
fn shift_left(x: &[f32], off: usize) -> Vec<f32> {
    if off == 0 || x.is_empty() {
        return x.to_vec();
    }
    let n = x.len();
    let off = off.min(n);
    let mut out = vec![0.0f32; n];
    out[..n - off].copy_from_slice(&x[off..]);
    out
}

/// Accumulate `b` into `a` element-wise. Same stem count / length is guaranteed: identical model and
/// identical (length-preserving) input across passes → identical output shape.
fn add_stems_into(a: &mut [StemAudio], b: &[StemAudio]) {
    for (sa, sb) in a.iter_mut().zip(b.iter()) {
        for (x, y) in sa.left.iter_mut().zip(sb.left.iter()) {
            *x += *y;
        }
        for (x, y) in sa.right.iter_mut().zip(sb.right.iter()) {
            *x += *y;
        }
    }
}

// ─── Audio Data Types ────────────────────────────────────────────

#[derive(Clone)]
pub struct AudioData {
    pub left: Vec<f32>,
    pub right: Vec<f32>,
    pub channels: usize,
    pub sample_rate: u32,
}

pub struct StemAudio {
    pub label: String,
    pub left: Vec<f32>,
    pub right: Vec<f32>,
    pub channels: usize,
}

struct AudioChunk {
    left: Vec<f32>,
    right: Vec<f32>,
    offset: usize,
}

// ─── MSST-compatible Chunking ────────────────────────────────────

/// Add border reflect padding before chunking (MSST `generic` mode).
/// Returns (padded_left, padded_right, border_samples, original_length).
fn prepare_padded_audio(
    audio: &AudioData,
    chunk_size: usize,
    step: usize,
) -> (Vec<f32>, Vec<f32>, usize, usize) {
    let orig_len = audio.left.len();
    let border = chunk_size - step;
    let is_stereo = audio.channels >= 2;

    if orig_len > 2 * border && border > 0 {
        let padded_left = reflect_pad(&audio.left, border);
        let padded_right = if is_stereo {
            reflect_pad(&audio.right, border)
        } else {
            vec![0.0; orig_len + 2 * border]
        };
        (padded_left, padded_right, border, orig_len)
    } else {
        let padded_right = if is_stereo { audio.right.clone() } else { vec![0.0; orig_len] };
        (audio.left.clone(), padded_right, 0, orig_len)
    }
}

/// Reflect padding matching `torch.nn.functional.pad(x, (pad_left, pad_right), mode='reflect')`.
/// Requires pad_left < signal.len() and pad_right < signal.len() (torch's own constraint).
fn reflect_pad_lr(signal: &[f32], pad_left: usize, pad_right: usize) -> Vec<f32> {
    let len = signal.len();
    assert!(
        pad_left < len && pad_right < len,
        "reflect_pad requires pads ({}, {}) < signal length ({})", pad_left, pad_right, len
    );
    let mut out = Vec::with_capacity(pad_left + len + pad_right);
    // Left pad: signal[pad_left], signal[pad_left-1], ..., signal[1] (edge sample excluded)
    for i in (1..=pad_left).rev() {
        out.push(signal[i]);
    }
    out.extend_from_slice(signal);
    // Right pad: signal[len-2], signal[len-3], ..., signal[len-1-pad_right]
    for i in 0..pad_right {
        out.push(signal[len - 2 - i]);
    }
    out
}

/// Symmetric reflect pad (MSST `generic` border padding) — thin wrapper over
/// `reflect_pad_lr`, the single source of truth for torch-exact reflect padding.
fn reflect_pad(signal: &[f32], pad: usize) -> Vec<f32> {
    reflect_pad_lr(signal, pad, pad)
}

/// MSST-style chunking: step = chunk_size / num_overlap.
/// Last chunk padded to chunk_size (edge-excluded reflect if filled past C/2 + 1
/// samples — MSST `length > C // 2 + 1` — zero otherwise).
fn msst_chunk_audio(
    left: &[f32],
    right: &[f32],
    is_stereo: bool,
    chunk_size: usize,
    step: usize,
) -> Vec<AudioChunk> {
    let total_len = left.len();
    if total_len == 0 { return vec![]; }

    let mut chunks = Vec::new();
    let mut pos = 0;

    while pos < total_len {
        let end = (pos + chunk_size).min(total_len);
        let actual_len = end - pos;

        let (chunk_left, chunk_right) = if actual_len == chunk_size {
            (left[pos..end].to_vec(),
             if is_stereo { right[pos..end].to_vec() } else { vec![0.0; chunk_size] })
        } else {
            // Pad last chunk to chunk_size
            let mut cl = vec![0.0f32; chunk_size];
            let mut cr = vec![0.0f32; chunk_size];
            cl[..actual_len].copy_from_slice(&left[pos..end]);
            if is_stereo {
                cr[..actual_len].copy_from_slice(&right[pos..end]);
            }

            // Reflect pad, matching original MSST (utils.py): threshold `length > C // 2 + 1`,
            // torch 'reflect' mode which EXCLUDES the edge sample — first pad value is
            // x[len-2], not a duplicate of x[len-1]. The threshold guarantees
            // need <= actual_len - 3, so `actual_len - 2 - i` never underflows;
            // saturating_sub is a defensive guard for degenerate tiny chunk_size.
            if actual_len > chunk_size / 2 + 1 {
                let need = chunk_size - actual_len;
                for i in 0..need {
                    let src = pos + actual_len.saturating_sub(2 + i);
                    cl[actual_len + i] = left[src];
                    if is_stereo {
                        cr[actual_len + i] = right[src];
                    }
                }
            }
            (cl, cr)
        };

        chunks.push(AudioChunk { left: chunk_left, right: chunk_right, offset: pos });

        if end >= total_len { break; }
        pos += step;
    }
    chunks
}

/// Residual = mix - stem
fn compute_residual(mix: &AudioData, stem: &StemAudio, label: &str) -> StemAudio {
    let left: Vec<f32> = mix.left.iter().zip(stem.left.iter()).map(|(m, s)| m - s).collect();
    let right = if mix.channels == 2 && stem.channels == 2 {
        mix.right.iter().zip(stem.right.iter()).map(|(m, s)| m - s).collect()
    } else { vec![] };
    StemAudio { label: label.to_string(), left, right, channels: mix.channels }
}

// ─── Overlap-add Accumulator ─────────────────────────────────────

struct AudioAccumulator {
    left: Vec<f32>,
    right: Vec<f32>,
    weights: Vec<f32>,
    is_stereo: bool,
}

impl AudioAccumulator {
    fn new(length: usize, is_stereo: bool) -> Self {
        Self {
            left: vec![0.0; length],
            right: if is_stereo { vec![0.0; length] } else { vec![] },
            weights: vec![0.0; length],
            is_stereo,
        }
    }

    fn add_windowed_mono(&mut self, data: &[f32], offset: usize, window: &[f32]) {
        for i in 0..data.len() {
            let pos = offset + i;
            if pos >= self.left.len() { break; }
            let w = if i < window.len() { window[i] } else { 1.0 };
            self.left[pos] += data[i] * w;
            self.weights[pos] += w;
        }
    }

    fn add_windowed_stereo(&mut self, left: &[f32], right: &[f32], offset: usize, window: &[f32]) {
        for i in 0..left.len() {
            let pos = offset + i;
            if pos >= self.left.len() { break; }
            let w = if i < window.len() { window[i] } else { 1.0 };
            self.left[pos] += left[i] * w;
            self.right[pos] += right[i] * w;
            self.weights[pos] += w;
        }
    }

    /// HTDemucs: simple accumulation with counter=1.0 (MSST demucs mode)
    fn add_simple_stereo(&mut self, left: &[f32], right: &[f32], offset: usize) {
        for i in 0..left.len() {
            let pos = offset + i;
            if pos >= self.left.len() { break; }
            self.left[pos] += left[i];
            self.right[pos] += right[i];
            self.weights[pos] += 1.0;
        }
    }

    fn finalize(mut self, label: String) -> StemAudio {
        let tiny = 1e-8f32;
        for i in 0..self.left.len() {
            if self.weights[i] > tiny {
                self.left[i] /= self.weights[i];
                if self.is_stereo { self.right[i] /= self.weights[i]; }
            }
        }
        let channels = if self.is_stereo { 2 } else { 1 };
        StemAudio { label, left: self.left, right: self.right, channels }
    }

    /// Finalize and strip border padding added by prepare_padded_audio.
    fn finalize_with_border(mut self, label: String, border: usize, orig_len: usize) -> StemAudio {
        let tiny = 1e-8f32;
        for i in 0..self.left.len() {
            if self.weights[i] > tiny {
                self.left[i] /= self.weights[i];
                if self.is_stereo { self.right[i] /= self.weights[i]; }
            }
        }

        // Remove border padding
        let start = border;
        let end = (start + orig_len).min(self.left.len());
        let left = self.left[start..end].to_vec();
        let right = if self.is_stereo {
            self.right[start..end].to_vec()
        } else { vec![] };

        let channels = if self.is_stereo { 2 } else { 1 };
        StemAudio { label, left, right, channels }
    }
}

/// MSST-style fade window matching `_getWindowingArray()`:
///   fadein  = torch.linspace(0, 1, fade_size)  — includes both endpoints
///   fadeout = torch.linspace(1, 0, fade_size)
fn build_msst_window(window_size: usize, fade_size: usize) -> Vec<f32> {
    let mut win = vec![1.0f32; window_size];
    if fade_size <= 1 || window_size <= fade_size * 2 {
        return win;
    }
    let denom = (fade_size - 1) as f32;
    for i in 0..fade_size {
        let t = i as f32 / denom; // linspace(0, 1, fade_size)
        win[i] = t;
        win[window_size - 1 - i] = t; // linspace(1, 0, fade_size) reversed
    }
    win
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
    let spec = hound::WavSpec {
        channels: stem.channels as u16,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| UtaiError::Audio(format!("Failed to create WAV: {}", e)))?;

    if stem.channels == 2 {
        for i in 0..stem.left.len() {
            writer.write_sample(stem.left[i])
                .map_err(|e| UtaiError::Audio(format!("WAV write: {}", e)))?;
            writer.write_sample(stem.right[i])
                .map_err(|e| UtaiError::Audio(format!("WAV write: {}", e)))?;
        }
    } else {
        for &s in &stem.left {
            writer.write_sample(s)
                .map_err(|e| UtaiError::Audio(format!("WAV write: {}", e)))?;
        }
    }

    writer.finalize().map_err(|e| UtaiError::Audio(format!("WAV finalize: {}", e)))?;
    Ok(())
}

// ─── HTDemucs `_spec` convention tests ───────────────────────────
// These lock the demucs frame convention (frames centered at j*hop + hop/2, nyquist
// dropped, [2 : 2+le] crop) against future regressions. Verified against the torch
// reference numerically (86.22 dB end-to-end); the tests below are self-contained.
#[cfg(test)]
mod htdemucs_spec_tests {
    use super::*;

    const N_FFT: usize = 4096;
    const HOP: usize = 1024;
    const SR: usize = 44100;

    fn make_proc() -> StftProcessor {
        StftProcessor::new(stft::StftConfig {
            n_fft: N_FFT,
            hop_length: HOP,
            win_length: N_FFT,
        })
    }

    /// Deterministic test signal: sines + band-limited noise. The 2-tap averager on the
    /// noise nulls the response at nyquist, so the convention's dropped-nyquist-bin error
    /// stays negligible and the roundtrip bound below is meaningful.
    fn synth_signal(len: usize) -> Vec<f32> {
        let mut state = 0x1234_5678u32;
        let mut noise = Vec::with_capacity(len + 1);
        for _ in 0..=len {
            // xorshift32 — deterministic, no rand dependency
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            noise.push((state as f32 / u32::MAX as f32) - 0.5);
        }
        (0..len)
            .map(|i| {
                let t = i as f32 / SR as f32;
                let tau = 2.0 * std::f32::consts::PI;
                (tau * 440.0 * t).sin() * 0.4
                    + (tau * 1234.5 * t).sin() * 0.25
                    + (tau * 3210.0 * t).sin() * 0.15
                    + 0.05 * (noise[i] + noise[i + 1]) * 0.5
            })
            .collect()
    }

    #[test]
    fn reflect_pad_lr_matches_torch() {
        // torch.nn.functional.pad([0,1,2,3,4], (3,2), mode='reflect') == [3,2,1,0,1,2,3,4,3,2]
        let x = [0.0f32, 1.0, 2.0, 3.0, 4.0];
        assert_eq!(
            reflect_pad_lr(&x, 3, 2),
            vec![3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]
        );
    }

    /// Alignment property: demucs_spec frame j is centered at j*hop + hop/2, so it must
    /// equal frame j of a PLAIN centered STFT of the signal advanced by hop/2 samples
    /// (whose frame j is centered at j*hop in ITS coordinates = j*hop + hop/2 in ours) —
    /// sample-exactly on interior frames, where neither transform's window touches any
    /// padded region (the two paths then window bit-identical samples).
    #[test]
    fn demucs_spec_frame_alignment() {
        let proc = make_proc();
        let t = 3 * SR; // 3 s
        let x = synth_signal(t);
        let le = (t + HOP - 1) / HOP;

        let ds = demucs_spec(&proc, &x);
        assert_eq!(ds.shape(), &[N_FFT / 2, le, 2]);

        let advanced = &x[HOP / 2..];
        let plain = proc.stft(advanced);

        // Interior frames: window [j*hop + hop/2 - n_fft/2, j*hop + hop/2 + n_fft/2)
        // must lie inside [0, t - hop/2) so BOTH transforms window pure signal samples.
        let j_lo = (N_FFT / 2 - HOP / 2 + HOP - 1) / HOP; // ceil -> 2
        let j_hi = (t - HOP - N_FFT / 2) / HOP; // floor, inclusive
        assert!(j_hi > j_lo + 8, "test signal too short for interior frames");
        assert!(j_hi < le && j_hi < plain.shape()[1]);

        let mut max_diff = 0.0f32;
        for j in j_lo..=j_hi {
            for b in 0..N_FFT / 2 {
                for ri in 0..2 {
                    max_diff = max_diff.max((ds[[b, j, ri]] - plain[[b, j, ri]]).abs());
                }
            }
        }
        assert!(max_diff < 1e-6, "interior frame mismatch: {max_diff}");
    }

    /// demucs_spec → demucs_ispec must round-trip to identity on the interior. The first
    /// and last pad = 3*hop/2 samples are excluded: the `[2 : 2+le]` crop discards the
    /// analysis frames covering the outer reflect pad and _ispec re-inserts them as ZERO
    /// frames, so the pad region is reconstructed from incomplete overlap-add (exactly as
    /// in demucs — chunk overlap covers it in real use).
    #[test]
    fn demucs_spec_ispec_roundtrip() {
        let proc = make_proc();
        let t = 3 * SR;
        let x = synth_signal(t);
        let le = (t + HOP - 1) / HOP;

        let spec = demucs_spec(&proc, &x);
        let rec = demucs_ispec(&proc, &spec, t);
        assert_eq!(rec.len(), t);

        let pad = HOP / 2 * 3;
        let hi = (le * HOP - pad).min(t);
        let mut max_err = 0.0f32;
        for i in pad..hi {
            max_err = max_err.max((rec[i] - x[i]).abs());
        }
        assert!(max_err < 1e-4, "roundtrip interior error: {max_err}");
    }
}
