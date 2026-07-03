//! utai-dsp — the separation pipeline's hot DSP kernels (STFT, overlap-add, chunking,
//! demucs conventions, CaC layout), extracted into a dedicated crate so DEV builds run
//! them at opt-level 3 while the frequently-edited app crate keeps opt-0 fast compiles
//! (`[profile.dev.package.utai-dsp]` in the workspace root).
//!
//! WHY a separate crate (and not `[profile.dev.package.ndarray]`): ndarray's generic
//! indexing (`spec[[f, t, ri]]`) monomorphizes in the CALLING crate, so a dependency
//! opt-override never optimizes these loops while they live in an opt-0 crate. S32
//! measured the difference at 10-30x wall time on the conv-arch pipelines.
//!
//! Numerics are LOCKED: every function here was moved verbatim from the app crate
//! (same float op order = bit-exact) and is covered by the unit tests here plus the
//! app-side bitwise A/B gate (converter/verify/README.md).

pub mod demucs;
pub mod stft;

pub use demucs::{demucs_ispec, demucs_spec};

use ndarray::Array3;

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

pub struct AudioChunk {
    pub left: Vec<f32>,
    pub right: Vec<f32>,
    pub offset: usize,
}

// ─── Input normalization (MSST-style mean/std, optional) ─────────

pub fn compute_audio_mean_std(audio: &AudioData) -> (f32, f32) {
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

pub fn normalize_audio(audio: &AudioData, mean: f32, std: f32) -> AudioData {
    let inv = 1.0 / std;
    AudioData {
        left: audio.left.iter().map(|&x| (x - mean) * inv).collect(),
        right: audio.right.iter().map(|&x| (x - mean) * inv).collect(),
        channels: audio.channels,
        sample_rate: audio.sample_rate,
    }
}

pub fn denormalize_stem(stem: &mut StemAudio, mean: f32, std: f32) {
    for x in stem.left.iter_mut() {
        *x = *x * std + mean;
    }
    for x in stem.right.iter_mut() {
        *x = *x * std + mean;
    }
}

// ─── Sample-shift / stem accumulation helpers (TTA) ──────────────

/// Shift right by `off` samples (zero-fill the front), length preserved.
pub fn shift_right(x: &[f32], off: usize) -> Vec<f32> {
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
pub fn shift_left(x: &[f32], off: usize) -> Vec<f32> {
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
pub fn add_stems_into(a: &mut [StemAudio], b: &[StemAudio]) {
    for (sa, sb) in a.iter_mut().zip(b.iter()) {
        for (x, y) in sa.left.iter_mut().zip(sb.left.iter()) {
            *x += *y;
        }
        for (x, y) in sa.right.iter_mut().zip(sb.right.iter()) {
            *x += *y;
        }
    }
}

// ─── MSST-compatible Chunking ────────────────────────────────────

/// Add border reflect padding before chunking (MSST `generic` mode).
/// Returns (padded_left, padded_right, border_samples, original_length).
pub fn prepare_padded_audio(
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
pub fn reflect_pad_lr(signal: &[f32], pad_left: usize, pad_right: usize) -> Vec<f32> {
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
pub fn reflect_pad(signal: &[f32], pad: usize) -> Vec<f32> {
    reflect_pad_lr(signal, pad, pad)
}

/// MSST-style chunking: step = chunk_size / num_overlap.
/// Last chunk padded to chunk_size (edge-excluded reflect if filled past C/2 + 1
/// samples — MSST `length > C // 2 + 1` — zero otherwise).
pub fn msst_chunk_audio(
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
pub fn compute_residual(mix: &AudioData, stem: &StemAudio, label: &str) -> StemAudio {
    let left: Vec<f32> = mix.left.iter().zip(stem.left.iter()).map(|(m, s)| m - s).collect();
    let right = if mix.channels == 2 && stem.channels == 2 {
        mix.right.iter().zip(stem.right.iter()).map(|(m, s)| m - s).collect()
    } else { vec![] };
    StemAudio { label: label.to_string(), left, right, channels: mix.channels }
}

// ─── Overlap-add Accumulator ─────────────────────────────────────

pub struct AudioAccumulator {
    left: Vec<f32>,
    right: Vec<f32>,
    weights: Vec<f32>,
    is_stereo: bool,
}

impl AudioAccumulator {
    pub fn new(length: usize, is_stereo: bool) -> Self {
        Self {
            left: vec![0.0; length],
            right: if is_stereo { vec![0.0; length] } else { vec![] },
            weights: vec![0.0; length],
            is_stereo,
        }
    }

    pub fn add_windowed_mono(&mut self, data: &[f32], offset: usize, window: &[f32]) {
        for i in 0..data.len() {
            let pos = offset + i;
            if pos >= self.left.len() { break; }
            let w = if i < window.len() { window[i] } else { 1.0 };
            self.left[pos] += data[i] * w;
            self.weights[pos] += w;
        }
    }

    pub fn add_windowed_stereo(&mut self, left: &[f32], right: &[f32], offset: usize, window: &[f32]) {
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
    pub fn add_simple_stereo(&mut self, left: &[f32], right: &[f32], offset: usize) {
        for i in 0..left.len() {
            let pos = offset + i;
            if pos >= self.left.len() { break; }
            self.left[pos] += left[i];
            self.right[pos] += right[i];
            self.weights[pos] += 1.0;
        }
    }

    pub fn finalize(mut self, label: String) -> StemAudio {
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
    pub fn finalize_with_border(mut self, label: String, border: usize, orig_len: usize) -> StemAudio {
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
pub fn build_msst_window(window_size: usize, fade_size: usize) -> Vec<f32> {
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

// ─── CaC layout helpers (shared by the mdx23c and htdemucs paths) ─

/// Assemble the CaC model input planes [L.re, L.im, R.re, R.im], each `f`×`num_frames`
/// row-major — the exact element order of the former per-path quad loops (they had
/// drifted into two identical copies; this is the single source of truth).
pub fn assemble_cac(spec_l: &Array3<f32>, spec_r: &Array3<f32>, f: usize, num_frames: usize) -> Vec<f32> {
    let mut cac_data = Vec::with_capacity(4 * f * num_frames);
    for ch_spec in [spec_l, spec_r] {
        for ri in 0..2 {
            for freq in 0..f {
                for t in 0..num_frames {
                    cac_data.push(ch_spec[[freq, t, ri]]);
                }
            }
        }
    }
    cac_data
}

/// Inverse of the CaC output layout for ONE stem: 4 planes at `stem_off` →
/// (spec_l, spec_r), each shaped (out_bins, num_frames, 2) with rows ≥ `f` left zero
/// (mdx23c passes the full freq_bins so the nyquist row stays zero; htdemucs passes f).
pub fn deinterleave_cac(
    data: &[f32],
    stem_off: usize,
    f: usize,
    num_frames: usize,
    out_bins: usize,
) -> (Array3<f32>, Array3<f32>) {
    let chan_size = f * num_frames;
    let mut spec_out_l = Array3::<f32>::zeros((out_bins, num_frames, 2));
    let mut spec_out_r = Array3::<f32>::zeros((out_bins, num_frames, 2));
    for freq in 0..f {
        for t in 0..num_frames {
            let ft = freq * num_frames + t;
            spec_out_l[[freq, t, 0]] = data[stem_off + ft];
            spec_out_l[[freq, t, 1]] = data[stem_off + chan_size + ft];
            spec_out_r[[freq, t, 0]] = data[stem_off + 2 * chan_size + ft];
            spec_out_r[[freq, t, 1]] = data[stem_off + 3 * chan_size + ft];
        }
    }
    (spec_out_l, spec_out_r)
}

/// HTDemucs freq-branch + time-branch sum: out[i] = freq[i] (0 past freq's end) + time[i].
pub fn sum_freq_time(freq: &[f32], time: &[f32], len: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let fl = if i < freq.len() { freq[i] } else { 0.0 };
        out.push(fl + time[i]);
    }
    out
}
