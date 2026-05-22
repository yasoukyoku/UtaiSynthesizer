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
    #[serde(default)]
    pub normalize: bool,
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

    /// Run separation. Returns (stem_label, audio_data) pairs.
    /// `progress_cb` returns `true` to continue, `false` to cancel.
    pub fn separate(
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

        for chunk_idx in 0..total_chunks {
            let (spec, num_frames) = &spectrograms[chunk_idx];
            let input_data: Vec<f32> = spec.iter().copied().collect();
            let input = InputTensor::F32 {
                data: input_data,
                shape: vec![1, freq_dim as i64, *num_frames as i64, 2],
            };
            let outputs = self.engine().run(&self.session_id, vec![("stft_repr", input)])?;
            let output_data = outputs.into_iter().next().ok_or_else(||
                UtaiError::Audio("Model produced no output".into()))?;

            // Immediately iSTFT + accumulate
            let offset = chunks[chunk_idx].offset;
            let chunk_len = chunks[chunk_idx].left.len();

            let mut win = window.clone();
            if chunk_idx == 0 {
                for i in 0..fade_size.min(win.len()) { win[i] = 1.0; }
            }
            if chunk_idx == total_chunks - 1 {
                let wl = win.len();
                for i in 0..fade_size.min(wl) { win[wl - 1 - i] = 1.0; }
            }

            for stem_idx in 0..self.config.num_stems {
                let stem_offset = stem_idx * freq_dim * num_frames * 2;
                let mask = Array3::from_shape_vec(
                    (freq_dim, *num_frames, 2),
                    output_data[stem_offset..stem_offset + freq_dim * num_frames * 2].to_vec(),
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

            let progress = 0.05 + 0.90 * (chunk_idx + 1) as f32 / total_chunks as f32;
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

impl Drop for NativePipeline {
    fn drop(&mut self) { self.engine().unload_model(&self.session_id); }
}

// ─── Audio Data Types ────────────────────────────────────────────

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
    let spec = hound::WavSpec {
        channels: stem.channels as u16,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| UtaiError::Audio(format!("Failed to create WAV: {}", e)))?;

    let to_i16 = |s: f32| -> i16 { (s.clamp(-1.0, 1.0) * 32767.0) as i16 };

    if stem.channels == 2 {
        for i in 0..stem.left.len() {
            writer.write_sample(to_i16(stem.left[i]))
                .map_err(|e| UtaiError::Audio(format!("WAV write: {}", e)))?;
            writer.write_sample(to_i16(stem.right[i]))
                .map_err(|e| UtaiError::Audio(format!("WAV write: {}", e)))?;
        }
    } else {
        for &s in &stem.left {
            writer.write_sample(to_i16(s))
                .map_err(|e| UtaiError::Audio(format!("WAV write: {}", e)))?;
        }
    }

    writer.finalize().map_err(|e| UtaiError::Audio(format!("WAV finalize: {}", e)))?;
    Ok(())
}
