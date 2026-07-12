//! RMVPE f0 detection — Rust side of the e2e export contract
//! (converter\verify\voice\RMVPE_CONTRACT.md; THE doc — read it before touching this).
//!
//! The ONNX graph (rmvpe_e2e.onnx) contains pad-to-32 + DeepUnet+BiGRU + decode, so the
//! ONLY DSP here is waveform → log-mel:
//!   reflect-pad 512 both ends → frames of 1024 @ hop 160 → periodic hann (f64-tabulated)
//!   → rFFT-1024 magnitude → mel = filters[128,513] @ mag → ln(clamp(mel, 1e-5)).
//! Output f0 is Hz @ 100 fps, T = 1 + N/160, unvoiced frames are EXACT 0.0.
//!
//! Also hosts the so-vits RMVPEF0Predictor.post_process port (f0 resize + uv + gap
//! interpolation) — reference vectors generated from the ORIGINAL code (gen_refs.py).

use ndarray::Array2;
use rustfft::{num_complex::Complex, FftPlanner};

use super::engine::{InputTensor, OnnxEngine};
use super::features::{np_interp, reflect_pad_np, torch_interp_nearest};
use crate::{Result, UtaiError};

pub const RMVPE_SR: u32 = 16000;
pub const RMVPE_HOP: usize = 160;
const N_FFT: usize = 1024;
const N_MELS: usize = 128;
const FREQ_BINS: usize = N_FFT / 2 + 1; // 513
const MEL_CLAMP: f32 = 1e-5;
/// Reflect padding needs N ≥ 513 — shorter inputs are zero-padded up to this first (contract).
const MIN_INPUT: usize = FREQ_BINS;

/// RVC pipeline.py uses thred=0.03 for rmvpe, fixed.
pub const RVC_RMVPE_THRESHOLD: f32 = 0.03;
/// so-vits infer() cr_threshold default (0.05) → RMVPEF0Predictor(threshold=0.05).
pub const SOVITS_RMVPE_THRESHOLD: f32 = 0.05;

/// Validate the mel filter bank loaded from rmvpe_mel_filters.npy ([128, 513] f32).
pub fn validate_mel_filters(filters: &Array2<f32>) -> Result<()> {
    if filters.nrows() != N_MELS || filters.ncols() != FREQ_BINS {
        return Err(UtaiError::Model(format!(
            "RMVPE_MEL_SHAPE: expected [{}, {}], got [{}, {}]",
            N_MELS,
            FREQ_BINS,
            filters.nrows(),
            filters.ncols()
        )));
    }
    Ok(())
}

/// mono 16 kHz f32 waveform (RAW, no normalization) → f0[Hz] @ 100 fps, T = 1 + N/160.
/// Unvoiced frames come back as exact 0.0 (safe to test with == 0.0).
pub fn rmvpe_detect(
    engine: &OnnxEngine,
    session_id: &str,
    mel_filters: &Array2<f32>,
    wav16k: &[f32],
    threshold: f32,
) -> Result<Vec<f32>> {
    validate_mel_filters(mel_filters)?;
    if wav16k.is_empty() {
        return Err(UtaiError::Audio("F0_EMPTY_INPUT".into()));
    }
    // contract: N ≥ 513 (reflect pad undefined below) — caller-side zero pad
    let zero_padded: Vec<f32>;
    let x: &[f32] = if wav16k.len() < MIN_INPUT {
        let mut v = wav16k.to_vec();
        v.resize(MIN_INPUT, 0.0);
        zero_padded = v;
        &zero_padded
    } else {
        wav16k
    };

    let n = x.len();
    let t_frames = 1 + n / RMVPE_HOP;
    let padded = reflect_pad_np(x, N_FFT / 2, N_FFT / 2);

    // periodic hann, tabulated in f64 then cast (== torch.hann_window default)
    let window: Vec<f32> = (0..N_FFT)
        .map(|k| {
            (0.5 - 0.5 * (2.0 * std::f64::consts::PI * k as f64 / N_FFT as f64).cos()) as f32
        })
        .collect();

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut buffer = vec![Complex::new(0.0f32, 0.0f32); N_FFT];

    // magnitude spectrogram [513, T]
    let mut mag = Array2::<f32>::zeros((FREQ_BINS, t_frames));
    for t in 0..t_frames {
        let start = t * RMVPE_HOP;
        for k in 0..N_FFT {
            buffer[k] = Complex::new(padded[start + k] * window[k], 0.0);
        }
        fft.process(&mut buffer);
        for bin in 0..FREQ_BINS {
            mag[[bin, t]] = (buffer[bin].re * buffer[bin].re + buffer[bin].im * buffer[bin].im)
                .sqrt();
        }
    }

    // mel[128,T] = filters[128,513] @ mag[513,T]; log-mel = ln(clamp(mel, 1e-5))
    let mut mel = mel_filters.dot(&mag);
    mel.mapv_inplace(|v| v.max(MEL_CLAMP).ln());

    let mel_data: Vec<f32> = mel
        .as_standard_layout()
        .as_slice()
        .expect("mel standard layout")
        .to_vec();
    let inputs = vec![
        (
            "mel",
            InputTensor::F32 {
                data: mel_data,
                shape: vec![1, N_MELS as i64, t_frames as i64],
            },
        ),
        (
            "threshold",
            InputTensor::F32 {
                data: vec![threshold],
                shape: vec![1],
            },
        ),
    ];
    let outputs = engine.run(session_id, inputs)?;
    let f0 = outputs
        .into_iter()
        .next()
        .ok_or_else(|| UtaiError::Inference("RMVPE_NO_OUTPUT".into()))?;
    if f0.len() != t_frames {
        return Err(UtaiError::Inference(format!(
            "RMVPE_FRAMES_MISMATCH: expected {}, got {}",
            t_frames,
            f0.len()
        )));
    }
    Ok(f0)
}

/// so-vits RMVPEF0Predictor.post_process port (modules\F0Predictor\RMVPEF0Predictor.py):
///   1. f0 = repeat_expand(f0, pad_to)              # F.interpolate mode='nearest'
///   2. uv = (f0 > 0) as f32                        # 1.0 = voiced
///   3. drop zeros; np.interp over frame TIMES (hop/sr · index) with edge fill
///      (0 nonzero → all-zero f0; 1 nonzero → constant fill)
/// The caller (compute_f0_uv) short-circuits an ALL-zero rmvpe result to zeros(pad_to)
/// BEFORE this — mirror that check at the call site.
/// Returns (f0, uv), both length pad_to; f0 interpolated (f64 internally, like np.interp).
pub fn sovits_f0_postprocess(
    f0: &[f32],
    pad_to: usize,
    hop_size: usize,
    sample_rate: u32,
) -> (Vec<f32>, Vec<f32>) {
    let resized = torch_interp_nearest(f0, pad_to);
    let uv: Vec<f32> = resized
        .iter()
        .map(|&v| if v > 0.0 { 1.0 } else { 0.0 })
        .collect();

    let nz: Vec<usize> = resized
        .iter()
        .enumerate()
        .filter(|(_, &v)| v != 0.0)
        .map(|(i, _)| i)
        .collect();
    if nz.is_empty() {
        return (vec![0.0; pad_to], uv);
    }
    if nz.len() == 1 {
        return (vec![resized[nz[0]]; pad_to], uv);
    }

    let hop_over_sr = hop_size as f64 / sample_rate as f64;
    let time_org: Vec<f64> = nz.iter().map(|&i| hop_over_sr * i as f64).collect();
    let values: Vec<f64> = nz.iter().map(|&i| resized[i] as f64).collect();
    let time_frame: Vec<f64> = (0..pad_to).map(|i| i as f64 * hop_over_sr).collect();
    let interp = np_interp(&time_frame, &time_org, &values);
    (interp.into_iter().map(|v| v as f32).collect(), uv)
}

#[cfg(test)]
mod tests {
    use super::*;

    // References generated by scratchpad gen_refs.py: the ORIGINAL
    // RMVPEF0Predictor.post_process (ast-exec, stub hop_length=512) run on three f0
    // vectors with pad_to 16/16/8, sampling_rate=44100.
    #[test]
    fn sovits_f0_postprocess_matches_original() {
        // case A: 9 frames → 16 (upsample), gaps interpolated, edges filled
        const F0_IN_A: &[f32] = &[0.0, 0.0, 220.0, 230.0, 0.0, 0.0, 240.0, 0.0, 0.0];
        const F0_OUT_A: &[f64] = &[220.0, 220.0, 220.0, 220.0, 220.0, 220.0, 230.0, 230.0, 232.5, 235.0, 237.5, 240.0, 240.0, 240.0, 240.0, 240.0];
        const UV_OUT_A: &[f32] = &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let (f0, uv) = sovits_f0_postprocess(F0_IN_A, 16, 512, 44100);
        for (i, (g, w)) in f0.iter().zip(F0_OUT_A.iter()).enumerate() {
            assert!((*g as f64 - w).abs() < 1e-3, "A f0[{}]: {} vs {}", i, g, w);
        }
        assert_eq!(uv, UV_OUT_A, "A uv");

        // case B: 25 frames → 16 (downsample, the real 100fps→86fps direction)
        const F0_IN_B: &[f32] = &[0.0, 0.0, 0.0, 210.0, 215.0, 0.0, 0.0, 0.0, 0.0, 225.0, 228.0, 231.0, 0.0, 0.0, 300.0, 0.0, 0.0, 0.0, 190.0, 0.0, 0.0, 188.0, 0.0, 0.0, 0.0];
        const F0_OUT_B: &[f64] = &[210.0, 210.0, 210.0, 215.0, 2.18333333333333343e+02, 2.21666666666666657e+02, 225.0, 228.0, 264.0, 300.0, 2.63333333333333314e+02, 2.26666666666666629e+02, 190.0, 189.0, 188.0, 188.0];
        const UV_OUT_B: &[f32] = &[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let (f0b, uvb) = sovits_f0_postprocess(F0_IN_B, 16, 512, 44100);
        for (i, (g, w)) in f0b.iter().zip(F0_OUT_B.iter()).enumerate() {
            assert!((*g as f64 - w).abs() < 1e-3, "B f0[{}]: {} vs {}", i, g, w);
        }
        assert_eq!(uvb, UV_OUT_B, "B uv");

        // case C: single voiced frame → constant fill
        const F0_IN_C: &[f32] = &[0.0, 0.0, 0.0, 111.5, 0.0];
        const UV_OUT_C: &[f32] = &[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0];
        let (f0c, uvc) = sovits_f0_postprocess(F0_IN_C, 8, 512, 44100);
        assert!(f0c.iter().all(|&v| (v - 111.5).abs() < 1e-4), "C constant fill");
        assert_eq!(uvc, UV_OUT_C, "C uv");
    }

    #[test]
    fn sovits_f0_postprocess_all_zero_yields_zeros() {
        let (f0, uv) = sovits_f0_postprocess(&[0.0; 10], 6, 512, 44100);
        assert_eq!(f0, vec![0.0; 6]);
        assert_eq!(uv, vec![0.0; 6]);
    }
}
