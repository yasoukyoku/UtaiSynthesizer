//! NSF-HiFiGAN mel extraction — faithful port of so-vits-svc vdecoder/nsf_hifigan/nvSTFT.py
//! STFT.get_mel, specialized to keyshift=0 / speed=1 / center=False (the ONLY way the
//! enhancer (enhancer.py:105) and the diffusion vocoder (diffusion/vocoder.py:74, inference
//! always passes keyshift=0) call it). Chain, with original line refs:
//!   pad_left=(win-hop)//2, pad_right=max((win-hop+1)//2, win-len-pad_left)   (nvSTFT.py:100-101)
//!   reflect pad if pad_right < len else constant-0 (BOTH sides)              (nvSTFT.py:102-107)
//!   torch.stft n_fft/hop/win, periodic hann, center=False                    (nvSTFT.py:109-110)
//!   mag = sqrt(re^2 + im^2 + 1e-9)                                           (nvSTFT.py:112)
//!   mel = filters @ mag;  ln(clamp(mel, min=1e-5))                           (nvSTFT.py:121-123)
//! Output [n_mels, n_frames]; n_frames = len//hop when len is a hop multiple (>= hop).
//!
//! The slaney mel filterbank ([128,1025] for the nsf_hifigan config) ships as
//! aux/nsf_hifigan_mel.npy from the converter — its correctness is the converter gate's
//! job; the unit tests here prove the DSP chain with an analytic filterbank.
//!
//! NOTE: deliberately NOT built on utai-dsp stft.rs (center=True torch lineage) nor
//! vr.rs (librosa zero-pad lineage) — nvSTFT uses center=False with (win-hop)/2 reflect
//! padding, a third framing scheme. Follows the f0.rs RMVPE front-end precision
//! decisions: f64-tabulated periodic hann cast to f32, f32 rustfft, f32 ndarray dot.

use ndarray::Array2;
use rustfft::{num_complex::Complex, FftPlanner};

use super::features::reflect_pad_np;

/// nsf_hifigan geometry (pretrain/nsf_hifigan/config.json): n_fft=2048, win_size=2048,
/// hop_size=512 @ 44100 / 128 mels. Enhancer and diffusion mel both use exactly this.
pub const NSF_N_FFT: usize = 2048;
pub const NSF_WIN_SIZE: usize = 2048;
pub const NSF_HOP: usize = 512;

/// dynamic_range_compression_torch clip_val (nvSTFT.py:51-52).
const MEL_CLAMP: f32 = 1e-5;
/// epsilon added INSIDE the magnitude sqrt (nvSTFT.py:112) — unlike the RMVPE mel (no eps).
const MAG_EPS: f32 = 1e-9;

/// mono f32 waveform @ model sr (44.1k) → ln-mel `[n_mels, n_frames]` with the
/// NSF-HiFiGAN geometry (2048/2048/512). `filters` = mel filterbank `[n_mels, 1025]`
/// (aux/nsf_hifigan_mel.npy). This is the mel fed to the diffusion q_sample (via
/// norm_spec) and to the enhancer's vocoder.
pub fn nsf_mel(samples: &[f32], filters: &Array2<f32>) -> Array2<f32> {
    mel_spectrogram(samples, filters, NSF_N_FFT, NSF_WIN_SIZE, NSF_HOP)
}

/// STFT.get_mel(y, keyshift=0, speed=1, center=False) for an arbitrary
/// (n_fft, win_size, hop) geometry. `filters` must be `[n_mels, n_fft/2+1]`.
pub fn mel_spectrogram(
    samples: &[f32],
    filters: &Array2<f32>,
    n_fft: usize,
    win_size: usize,
    hop: usize,
) -> Array2<f32> {
    let freq_bins = n_fft / 2 + 1;
    // torch.stft rejects win_length > n_fft too; window is zero-padded to n_fft below.
    assert!(
        win_size <= n_fft && hop <= win_size && hop > 0,
        "mel_spectrogram 几何参数非法: n_fft={} win={} hop={}",
        n_fft,
        win_size,
        hop
    );
    assert_eq!(
        filters.ncols(),
        freq_bins,
        "mel 滤波器形状与 n_fft 不匹配：期望 [n_mels, {}]，得到 [{}, {}]",
        freq_bins,
        filters.nrows(),
        filters.ncols()
    );

    let n = samples.len();
    // nvSTFT.py:100-101 — pad_right's second arm can go negative; the first arm is >= 0,
    // so clamping the subtraction at 0 is equivalent to the python max().
    let pad_left = (win_size - hop) / 2;
    let pad_right = ((win_size - hop + 1) / 2).max(win_size.saturating_sub(n + pad_left));
    // nvSTFT.py:102-107 — F.pad applies ONE mode to BOTH sides. When reflect is chosen,
    // pad_left <= pad_right < n, so the single-bounce torch reflect == reflect_pad_np
    // (multi-bounce only differs for pad > n-1, unreachable here).
    let padded: Vec<f32> = if pad_right < n {
        reflect_pad_np(samples, pad_left, pad_right)
    } else {
        let mut v = vec![0.0f32; pad_left + n + pad_right];
        v[pad_left..pad_left + n].copy_from_slice(samples);
        v
    };

    // center=False framing: frame t covers [t*hop, t*hop + n_fft). The padding above
    // guarantees padded.len() >= win_size; win_size == n_fft for every shipped config.
    let padded_len = padded.len();
    assert!(
        padded_len >= n_fft,
        "填充后长度 {} 仍小于 n_fft {}（win_size < n_fft 的短输入）",
        padded_len,
        n_fft
    );
    let n_frames = 1 + (padded_len - n_fft) / hop;

    // periodic hann of win_size, f64-tabulated then cast (torch.hann_window default),
    // centered into the n_fft frame exactly like torch.stft pads short windows.
    let mut window = vec![0.0f32; n_fft];
    let w_off = (n_fft - win_size) / 2;
    for k in 0..win_size {
        window[w_off + k] =
            (0.5 - 0.5 * (2.0 * std::f64::consts::PI * k as f64 / win_size as f64).cos()) as f32;
    }

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut buffer = vec![Complex::new(0.0f32, 0.0f32); n_fft];

    // magnitude spectrogram [freq_bins, n_frames], eps inside the sqrt (nvSTFT.py:112)
    let mut mag = Array2::<f32>::zeros((freq_bins, n_frames));
    for t in 0..n_frames {
        let start = t * hop;
        for k in 0..n_fft {
            buffer[k] = Complex::new(padded[start + k] * window[k], 0.0);
        }
        fft.process(&mut buffer);
        for bin in 0..freq_bins {
            mag[[bin, t]] =
                (buffer[bin].re * buffer[bin].re + buffer[bin].im * buffer[bin].im + MAG_EPS)
                    .sqrt();
        }
    }

    // mel = filters @ mag; ln(clamp(mel, 1e-5)) (nvSTFT.py:121-123)
    let mut mel = filters.dot(&mag);
    mel.mapv_inplace(|v| v.max(MEL_CLAMP).ln());
    mel
}

#[cfg(test)]
mod tests {
    use super::*;

    // All reference vectors below were generated by scratchpad gen_mel_refs.py run with
    // D:\MyDev\Utai_v2-dev\converter\.venv\Scripts\python.exe against the ORIGINAL
    // vdecoder/nsf_hifigan/nvSTFT.py STFT.get_mel (torch 2.12.0+cpu, numpy 2.2.6), with
    // the analytic 8-row gaussian filterbank below injected via the class's own
    // mel_basis cache. Signal/filterbank formulas MUST stay in lockstep with that script.

    /// analytic filterbank: filt[m][k] = exp(-((k - (20+70m))/60)^2), m=0..7, k=0..1024,
    /// computed in f64 then cast — identical formula in gen_mel_refs.py.
    fn analytic_filters() -> Array2<f32> {
        let bins = NSF_N_FFT / 2 + 1;
        let mut f = Array2::<f32>::zeros((8, bins));
        for m in 0..8 {
            let center = 20.0 + 70.0 * m as f64;
            for k in 0..bins {
                f[[m, k]] = (-((k as f64 - center) / 60.0).powi(2)).exp() as f32;
            }
        }
        f
    }

    /// 0.5*sin(2π(100 t + ((10000-100)/(2 dur)) t²)), t = i/44100, f64 → f32.
    fn chirp(n: usize) -> Vec<f32> {
        let dur = n as f64 / 44100.0;
        let half_k = (10000.0 - 100.0) / (2.0 * dur);
        (0..n)
            .map(|i| {
                let t = i as f64 / 44100.0;
                (0.5 * (2.0 * std::f64::consts::PI * (100.0 * t + half_k * t * t)).sin()) as f32
            })
            .collect()
    }

    /// LCG noise: x = x*1664525 + 1013904223 (mod 2^32); s = 0.3*((x/2^32)*2 - 1).
    fn lcg_noise(n: usize, seed: u32) -> Vec<f32> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x = x.wrapping_mul(1664525).wrapping_add(1013904223);
                (0.3 * ((x as f64 / 4294967296.0) * 2.0 - 1.0)) as f32
            })
            .collect()
    }

    // same comparator pattern as features.rs tests (module-local by the file-ownership
    // rule of the S36 split; scale floor 1.0 → tol is absolute for |want| < 1, relative above)
    fn assert_close(got: &[f32], want: &[f32], tol: f32, label: &str) {
        assert_eq!(got.len(), want.len(), "{}: length {} vs {}", label, got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let scale = w.abs().max(1.0);
            assert!(
                (g - w).abs() <= tol * scale,
                "{}: idx {} got {} want {} (tol {})",
                label, i, g, w, tol
            );
        }
    }

    // t1: chirp len 5120 = 512*10, reflect branch (pad_right=768 < 5120) → 10 frames.
    #[test]
    fn nsf_mel_matches_original_chirp() {
        const REF_T1_FRAME0: &[f32] = &[7.711158752e+00, 7.250881195e+00, 4.918555260e+00, 1.521614313e+00, 5.561922193e-01, 1.497659236e-01, -1.635755152e-01, -4.121512473e-01];
        const REF_T1_SPREAD: &[f32] = &[7.711158752e+00, 4.918555260e+00, 1.497659236e-01, -4.121512473e-01, 6.232388496e+00, 7.633544922e+00, -3.632454395e+00, -5.692244530e+00, 1.315827370e+00, 7.464926720e+00, 3.498716116e+00, -5.623843670e+00, -5.742986202e+00, 4.330679893e+00, 7.170151234e+00, 1.684875786e-01, 1.721770883e+00, 2.260149956e+00, 7.795177460e+00, 5.770388603e+00, 3.478411436e+00, 4.007862568e+00, 7.095374107e+00, 6.806504726e+00];

        let filters = analytic_filters();
        let mel = nsf_mel(&chirp(5120), &filters);
        assert_eq!(mel.dim(), (8, 10), "t1 shape");

        let frame0: Vec<f32> = (0..8).map(|m| mel[[m, 0]]).collect();
        assert_close(&frame0, REF_T1_FRAME0, 1e-4, "t1 frame0");

        // frames (0,2,4,6,8,9) × bins (0,2,5,7), frame-major — 24 spread points
        let mut spread = Vec::with_capacity(24);
        for f in [0usize, 2, 4, 6, 8, 9] {
            for b in [0usize, 2, 5, 7] {
                spread.push(mel[[b, f]]);
            }
        }
        assert_close(&spread, REF_T1_SPREAD, 1e-4, "t1 spread");
    }

    // t2: short inputs. len 1500 is STILL the reflect branch (pad_right=768 < 1500);
    // the constant branch needs pad_right >= len, i.e. len <= 768 for this geometry:
    // len 600 (pad_right=768 arm 1) and len 300 (pad_right=980 via the win-len-pad_left arm).
    #[test]
    fn nsf_mel_short_input_branches() {
        const REF_T2A_ALL: &[f32] = &[5.711702824e+00, 6.035090446e+00, 6.138385773e+00, 6.103619576e+00, 6.199264526e+00, 6.216928482e+00, 6.080991268e+00, 6.002516747e+00, 5.862903595e+00, 6.102591038e+00, 6.110278606e+00, 6.070018768e+00, 6.200645447e+00, 6.179111958e+00, 6.054619789e+00, 6.016012192e+00];
        const REF_T2B_ALL: &[f32] = &[5.451765537e+00, 5.878322124e+00, 5.912583351e+00, 6.044648170e+00, 6.023201942e+00, 5.896439552e+00, 5.972767353e+00, 5.944054604e+00];
        const REF_T2C_ALL: &[f32] = &[4.973464012e+00, 5.333301067e+00, 5.444479465e+00, 5.598187923e+00, 5.531678677e+00, 5.641640663e+00, 5.628134727e+00, 5.589754105e+00];

        let filters = analytic_filters();

        let mel = nsf_mel(&lcg_noise(1500, 123456789), &filters);
        assert_eq!(mel.dim(), (8, 2), "t2a shape");
        let mut all = Vec::with_capacity(16);
        for f in 0..2 {
            for b in 0..8 {
                all.push(mel[[b, f]]);
            }
        }
        assert_close(&all, REF_T2A_ALL, 1e-4, "t2a (len 1500, reflect)");

        let mel = nsf_mel(&lcg_noise(600, 42), &filters);
        assert_eq!(mel.dim(), (8, 1), "t2b shape");
        let col: Vec<f32> = (0..8).map(|m| mel[[m, 0]]).collect();
        assert_close(&col, REF_T2B_ALL, 1e-4, "t2b (len 600, constant)");

        let mel = nsf_mel(&lcg_noise(300, 7), &filters);
        assert_eq!(mel.dim(), (8, 1), "t2c shape");
        let col: Vec<f32> = (0..8).map(|m| mel[[m, 0]]).collect();
        assert_close(&col, REF_T2C_ALL, 1e-4, "t2c (len 300, constant tail-arm)");
    }

    // t3: len exactly hop*k → exactly k frames (verified against the original for k=1,2,8;
    // gen_mel_refs.py asserts the same counts python-side).
    #[test]
    fn nsf_mel_frame_count_on_hop_multiples() {
        let filters = analytic_filters();
        for k in [1usize, 2, 8] {
            let mel = nsf_mel(&chirp(NSF_HOP * k), &filters);
            assert_eq!(mel.dim(), (8, k), "len {} → {} frames", NSF_HOP * k, k);
            assert!(mel.iter().all(|v| v.is_finite()), "non-finite mel at k={}", k);
        }
    }
}
