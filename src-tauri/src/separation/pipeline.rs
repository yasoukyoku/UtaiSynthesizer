use std::path::Path;

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

impl NativePipeline {
    pub fn new(engine: &OnnxEngine, model_path: &Path) -> Result<Self> {
        let config_path = model_path.with_extension("json");
        if !config_path.exists() {
            return Err(UtaiError::Audio(format!(
                "Model config not found: {}", config_path.display()
            )));
        }
        let config_text = std::fs::read_to_string(&config_path)?;
        let config: ModelConfig = serde_json::from_str(&config_text)?;
        let session_id = engine.load_model(&model_path.to_path_buf())?;
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
        if self.config.normalize && self.config.model_type != "htdemucs" {
            let (mean, std) = compute_audio_mean_std(audio);
            tracing::info!("MSST normalize ON (mean={:.6}, std={:.6})", mean, std);
            let normed = normalize_audio(audio, mean, std);
            let mut stems = self.run_augmented(&normed, progress_cb)?;
            for stem in &mut stems {
                denormalize_stem(stem, mean, std);
            }
            Ok(stems)
        } else {
            self.run_augmented(audio, progress_cb)
        }
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

        // ── Phase 1: Parallel STFT on all chunks ──
        tracing::info!("STFT: {} chunks (parallel)...", total_chunks);
        let spectrograms: Vec<(Array3<f32>, usize)> = chunks.par_iter().map(|chunk| {
            let spec = if is_stereo {
                proc.stft_stereo(&chunk.left, &chunk.right)
            } else {
                proc.stft(&chunk.left)
            };
            let num_frames = spec.shape()[1];
            (spec, num_frames)
        }).collect();

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
        while chunk_idx < total_chunks {
            let bg = b.min(total_chunks - chunk_idx); // group size, remainder-safe

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
            let outputs = self.engine().run(&self.session_id, vec![("stft_repr", input)])?;
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

            chunk_idx += bg;
            let progress = 0.05 + 0.90 * chunk_idx as f32 / total_chunks as f32;
            if !progress_cb(progress) {
                return Err(UtaiError::Audio("Stopped by user".into()));
            }
        }

        progress_cb(1.0);

        let mut stems: Vec<StemAudio> = stem_accumulators.into_iter().enumerate().map(|(i, acc)| {
            let label = default_stem_label(&self.config.model_type, i, self.config.num_stems);
            acc.finalize_with_border(label, border, orig_len)
        }).collect();

        if self.config.num_stems == 1 {
            stems.push(compute_residual(audio, &stems[0], "instrumental"));
        }
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

        // Phase 2: GPU inference
        tracing::info!("MDX23C phase 2: GPU inference on {} chunks...", total_chunks);
        let mut inference_outputs: Vec<Vec<f32>> = Vec::with_capacity(total_chunks);

        for (chunk_idx, (spec_l, spec_r, num_frames)) in stft_pairs.iter().enumerate() {
            let f = dim_f.min(spec_l.shape()[0]);
            let mut cac_data = Vec::with_capacity(4 * f * num_frames);
            for ch_spec in [spec_l, spec_r] {
                for ri in 0..2 {
                    for freq in 0..f {
                        for t in 0..*num_frames {
                            cac_data.push(ch_spec[[freq, t, ri]]);
                        }
                    }
                }
            }

            let input = InputTensor::F32 {
                data: cac_data,
                shape: vec![1, 4, f as i64, *num_frames as i64],
            };
            let outputs = self.engine().run(&self.session_id, vec![("stft_repr", input)])?;
            inference_outputs.push(outputs.into_iter().next().ok_or_else(||
                UtaiError::Audio("Model produced no output".into()))?);

            let progress = 0.05 + 0.85 * (chunk_idx + 1) as f32 / total_chunks as f32;
            if !progress_cb(progress) {
                return Err(UtaiError::Audio("Stopped by user".into()));
            }
        }

        // Phase 3: iSTFT + windowed overlap-add
        tracing::info!("MDX23C phase 3: iSTFT + overlap-add...");
        let padded_len = padded_left.len();
        let mut stem_accumulators: Vec<AudioAccumulator> = (0..self.config.num_stems)
            .map(|_| AudioAccumulator::new(padded_len, true))
            .collect();

        for (chunk_idx, output_data) in inference_outputs.iter().enumerate() {
            let (spec_l, _spec_r, num_frames) = &stft_pairs[chunk_idx];
            let offset = chunks[chunk_idx].offset;
            let chunk_len = chunks[chunk_idx].left.len();
            let f = dim_f.min(spec_l.shape()[0]);
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
                let mut spec_out_l = Array3::<f32>::zeros((freq_bins, *num_frames, 2));
                let mut spec_out_r = Array3::<f32>::zeros((freq_bins, *num_frames, 2));

                for freq in 0..f {
                    for t in 0..*num_frames {
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
        }

        progress_cb(1.0);

        let mut stems: Vec<StemAudio> = stem_accumulators.into_iter().enumerate().map(|(i, acc)| {
            let label = default_stem_label(&self.config.model_type, i, self.config.num_stems);
            acc.finalize_with_border(label, border, orig_len)
        }).collect();

        if self.config.num_stems == 1 {
            stems.push(compute_residual(audio, &stems[0], "instrumental"));
        }
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
        let step = segment_samples / num_overlap;

        let mut stem_accumulators: Vec<AudioAccumulator> = (0..self.config.num_stems)
            .map(|_| AudioAccumulator::new(orig_len, true))
            .collect();

        let total_segments = if orig_len <= segment_samples {
            1
        } else {
            (orig_len + step - 1) / step
        };
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

            let spec_l = proc.stft(&left_chunk);
            let spec_r = proc.stft(&right_chunk);
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
                let full_bins = proc.freq_bins();
                let mut spec_out_l = Array3::<f32>::zeros((full_bins, num_frames, 2));
                let mut spec_out_r = Array3::<f32>::zeros((full_bins, num_frames, 2));
                for freq in 0..f {
                    for t in 0..num_frames {
                        let ft = freq * num_frames + t;
                        spec_out_l[[freq, t, 0]] = freq_data[foff + ft];
                        spec_out_l[[freq, t, 1]] = freq_data[foff + chan_size + ft];
                        spec_out_r[[freq, t, 0]] = freq_data[foff + 2 * chan_size + ft];
                        spec_out_r[[freq, t, 1]] = freq_data[foff + 3 * chan_size + ft];
                    }
                }
                let freq_l = proc.istft(&spec_out_l, segment_samples);
                let freq_r = proc.istft(&spec_out_r, segment_samples);

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

            if pos + segment_samples >= orig_len { break; }
            pos += step;
        }

        let stems: Vec<StemAudio> = stem_accumulators.into_iter().enumerate().map(|(i, acc)| {
            let label = default_stem_label(&self.config.model_type, i, self.config.num_stems);
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

        let mut stems: Vec<StemAudio> = stem_accumulators.into_iter().enumerate().map(|(i, acc)| {
            let label = default_stem_label(&self.config.model_type, i, self.config.num_stems);
            acc.finalize_with_border(label, border, orig_len)
        }).collect();

        if self.config.num_stems == 1 {
            stems.push(compute_residual(audio, &stems[0], "instrumental"));
        }
        Ok(stems)
    }

    pub fn unload(&self) { self.engine().unload_model(&self.session_id); }
}

// NOTE: NativePipeline intentionally does NOT unload its session on Drop. Sessions are
// cached + LRU-bounded by OnnxEngine so that repeated separations of the same model reuse
// the already-optimized session instead of reloading from disk every run. Call `unload()`
// or OnnxEngine::clear_sessions() to free explicitly.

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
// HTDemucs-only (matching MSST). compute_residual uses the per-pass (transformed) mix, so the
// derived "instrumental" stem inverts correctly too.

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

/// Reflect padding matching `torch.nn.functional.pad(x, (pad, pad), mode='reflect')`.
/// Requires pad < signal.len() (enforced by caller via `orig_len > 2 * border` check).
fn reflect_pad(signal: &[f32], pad: usize) -> Vec<f32> {
    let len = signal.len();
    assert!(pad < len, "reflect_pad requires pad ({}) < signal length ({})", pad, len);
    let mut out = Vec::with_capacity(pad + len + pad);
    // Left pad: signal[pad], signal[pad-1], ..., signal[1]
    for i in (1..=pad).rev() {
        out.push(signal[i]);
    }
    out.extend_from_slice(signal);
    // Right pad: signal[len-2], signal[len-3], ..., signal[len-1-pad]
    for i in 0..pad {
        out.push(signal[len - 2 - i]);
    }
    out
}

/// MSST-style chunking: step = chunk_size / num_overlap.
/// Last chunk padded to chunk_size (reflect if >50% filled, zero otherwise).
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

            // Reflect pad if >50% filled
            if actual_len > chunk_size / 2 {
                let need = chunk_size - actual_len;
                for i in 0..need.min(actual_len) {
                    cl[actual_len + i] = left[end - 1 - i.min(end - pos - 1)];
                    if is_stereo {
                        cr[actual_len + i] = right[end - 1 - i.min(end - pos - 1)];
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
