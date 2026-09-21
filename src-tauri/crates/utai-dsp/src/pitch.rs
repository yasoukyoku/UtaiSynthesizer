//! Phase 5-2: DSP pitch tracker (NCCF via FFT autocorrelation).
//!
//! Analysis-only (零损伤): feeds `f0Curve` visualization and the harmonicity
//! metrics' `f0_hz` argument. Deliberately NOT the rmvpe ML path (`inference/f0.rs`):
//! that one needs a 16 kHz mono model feed and exists for the voice pipeline —
//! the analysis nodes must run offline with zero models and fixed, reproducible
//! parameters (plan §Phase 5: analysis STFT/params are self-contained, never
//! mixed with the inference-side conventions).
//!
//! Method: per-frame normalized autocorrelation (NCCF) computed through the
//! FFT power theorem (O(N log N) per frame instead of O(N·lags)), lags in
//! [60, 1200] Hz, parabolic peak interpolation, one octave-halving guard
//! (a subharmonic lag scoring ≥ 0.9 of the peak wins — classic NCCF fix).

use rustfft::{num_complex::Complex, FftPlanner};

/// Analysis frame length (samples). 2048 @ 44.1 kHz ≈ 46 ms.
pub const FRAME: usize = 2048;
/// Hop between frames. 512 → ~86 fps @ 44.1 kHz.
pub const HOP: usize = 512;
/// FFT size for the autocorrelation (≥ 2·FRAME keeps the linear ACF alias-free).
const FFT_SIZE: usize = 4096;
/// Pitch search bounds (Hz).
const F0_MIN: f32 = 60.0;
const F0_MAX: f32 = 1200.0;
/// Voiced threshold on the NCCF peak (classical 0.5).
const VOICED_CLARITY: f32 = 0.5;
/// Silence gate: frames below this RMS are unvoiced regardless of clarity.
const RMS_GATE: f32 = 1e-4;
/// A lag scoring this close to the peak is treated as the same pitch (octave guard).
const OCTAVE_GUARD: f32 = 0.9;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PitchFrame {
    /// Frame-center time in seconds.
    pub time_secs: f32,
    /// Estimated f0 in Hz; 0.0 = unvoiced.
    pub f0_hz: f32,
    /// NCCF peak value (0..1]; 0.0 when unvoiced.
    pub clarity: f32,
}

/// Track pitch over the whole signal. Never panics on short input — returns an
/// empty Vec when there is less than one frame.
pub fn track_pitch(x: &[f32], sample_rate: u32) -> Vec<PitchFrame> {
    let sr = sample_rate as f32;
    let sr = if sr <= 0.0 { 44100.0 } else { sr };
    if x.len() < FRAME {
        return Vec::new();
    }
    let min_lag = ((sr / F0_MAX).floor() as usize).max(2);
    let max_lag = ((sr / F0_MIN).ceil() as usize).min(FRAME - 1);
    if min_lag >= max_lag {
        return Vec::new();
    }

    let mut planner = FftPlanner::<f32>::new();
    let fwd = planner.plan_fft_forward(FFT_SIZE);
    let inv = planner.plan_fft_inverse(FFT_SIZE);
    let window = hann(FRAME);

    let n_frames = (x.len() - FRAME) / HOP + 1;
    let mut out = Vec::with_capacity(n_frames);
    let mut buf = vec![Complex::new(0.0f32, 0.0f32); FFT_SIZE];
    // Energies via suffix sums: e_τ = Σ_{i=τ}^{FRAME-1} wx[i]² — O(FRAME) per
    // frame total instead of O(FRAME·lags) (a 3-min song has ~15k frames).
    let mut energy = vec![0.0f32; FRAME];
    let mut wx = vec![0.0f32; FRAME];

    for f in 0..n_frames {
        let start = f * HOP;
        for (i, w) in wx.iter_mut().enumerate() {
            *w = x[start + i] * window[i];
        }
        let rms = (wx.iter().map(|v| v * v).sum::<f32>() / FRAME as f32).sqrt();
        let time_secs = (start + FRAME / 2) as f32 / sr;
        if rms < RMS_GATE {
            out.push(PitchFrame { time_secs, f0_hz: 0.0, clarity: 0.0 });
            continue;
        }

        for (i, b) in buf.iter_mut().enumerate() {
            *b = Complex::new(if i < FRAME { wx[i] } else { 0.0 }, 0.0);
        }
        fwd.process(&mut buf);
        for b in buf.iter_mut() {
            let p = b.re * b.re + b.im * b.im;
            *b = Complex::new(p, 0.0);
        }
        inv.process(&mut buf);
        let scale = 1.0 / FFT_SIZE as f32;
        // Linear autocorrelation: acf[τ] = Σ_{n<FRAME-τ} wx[n]·wx[n+τ].
        let acf = |lag: usize| buf[lag].re * scale;

        let mut suffix = 0.0f32;
        for i in (0..FRAME).rev() {
            suffix += wx[i] * wx[i];
            energy[i] = suffix;
        }
        let e0 = energy[0];

        // Global NCCF peak among INTERIOR local maxima: edge lags can be
        // spuriously hot for low pitches (cos(2π·lag/τ₀) ≈ 0.9 at the min-lag
        // boundary while the true period peak is window-damped), which would
        // octave-error bass notes into "unvoiced".
        let nccf = |tau: usize| acf(tau) / (e0 * energy[tau] + 1e-12).sqrt();
        let mut best_lag = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for tau in (min_lag + 1)..max_lag {
            let v = nccf(tau);
            if v >= nccf(tau - 1) && v > nccf(tau + 1) && v > best_val {
                best_val = v;
                best_lag = tau;
            }
        }
        if best_lag == 0 {
            // f0 pinned at a search edge — fall back to the plain maximum.
            for tau in min_lag..=max_lag {
                let v = nccf(tau);
                if v > best_val {
                    best_val = v;
                    best_lag = tau;
                }
            }
        }
        // Octave guard: prefer the shorter lag when it scores nearly as high.
        let mut lag = best_lag;
        let mut val = best_val;
        while lag >= 2 * min_lag {
            let half = lag / 2;
            let v_half = nccf(half);
            if v_half >= OCTAVE_GUARD * val {
                lag = half;
                val = v_half;
            } else {
                break;
            }
        }
        if val < VOICED_CLARITY {
            out.push(PitchFrame { time_secs, f0_hz: 0.0, clarity: 0.0 });
            continue;
        }

        // Parabolic interpolation around the peak (sub-sample lag).
        let y_m = acf(lag - 1) / (e0 * energy[lag - 1] + 1e-12).sqrt();
        let y_0 = val;
        let y_p = acf(lag + 1) / (e0 * energy[lag + 1] + 1e-12).sqrt();
        let denom = y_m - 2.0 * y_0 + y_p;
        let delta = if denom.abs() > 1e-12 {
            (0.5 * (y_m - y_p) / denom).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let lag_f = lag as f32 + delta;
        let f0 = if lag_f > 0.0 { sr / lag_f } else { 0.0 };
        let f0 = if (F0_MIN..=F0_MAX).contains(&f0) { f0 } else { 0.0 };
        out.push(PitchFrame { time_secs, f0_hz: f0, clarity: y_0.clamp(0.0, 1.0) });
    }
    out
}

/// Median f0 across voiced frames (None when the whole clip is unvoiced).
pub fn median_f0(frames: &[PitchFrame]) -> Option<f32> {
    let mut v: Vec<f32> = frames.iter().map(|f| f.f0_hz).filter(|f| *f > 0.0).collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = v.len() / 2;
    Some(if v.len() % 2 == 1 { v[mid] } else { (v[mid - 1] + v[mid]) * 0.5 })
}

fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(sr: u32, secs: f32, f0: f32) -> Vec<f32> {
        let n = (sr as f32 * secs) as usize;
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * f0 * i as f32 / sr as f32).sin())
            .collect()
    }

    #[test]
    fn tracks_a_220hz_sine() {
        let sr = 44100;
        let frames = track_pitch(&sine(sr, 1.0, 220.0), sr);
        assert!(frames.len() > 50, "1 s @ hop 512 must give ~85 frames, got {}", frames.len());
        let voiced = frames.iter().filter(|f| f.f0_hz > 0.0).count();
        assert!(voiced as f32 / frames.len() as f32 > 0.9, "sine must be voiced almost everywhere");
        let med = median_f0(&frames).unwrap();
        assert!((med - 220.0).abs() / 220.0 < 0.02, "median f0 {med} vs 220");
    }

    #[test]
    fn tracks_an_880hz_sine_without_octave_error() {
        let sr = 44100;
        let med = median_f0(&track_pitch(&sine(sr, 1.0, 880.0), sr)).unwrap();
        assert!((med - 880.0).abs() / 880.0 < 0.02, "median f0 {med} vs 880");
    }

    #[test]
    fn tracks_a_low_82hz_sine() {
        let sr = 44100;
        let med = median_f0(&track_pitch(&sine(sr, 1.5, 82.4), sr)).unwrap();
        assert!((med - 82.4).abs() / 82.4 < 0.03, "median f0 {med} vs 82.4");
    }

    #[test]
    fn silence_is_unvoiced_everywhere() {
        let sr = 44100;
        let frames = track_pitch(vec![0.0f32; sr as usize].as_slice(), sr);
        assert!(!frames.is_empty());
        assert!(frames.iter().all(|f| f.f0_hz == 0.0 && f.clarity == 0.0));
        assert!(median_f0(&frames).is_none());
    }

    #[test]
    fn short_input_returns_empty_without_panicking() {
        assert!(track_pitch(&[0.1f32; 10], 44100).is_empty());
        assert!(track_pitch(&[], 44100).is_empty());
    }

    #[test]
    fn vibrato_stays_within_tolerance() {
        let sr = 44100;
        // 220 Hz ± 3 % vibrato at 5 Hz — proper phase accumulation (a plain
        // sin(2π·f(t)·t) sweeps its instantaneous frequency unintentionally).
        let n = sr;
        let mut phase = 0.0f32;
        let x: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                let f = 220.0 * (1.0 + 0.03 * (2.0 * std::f32::consts::PI * 5.0 * t).sin());
                phase += 2.0 * std::f32::consts::PI * f / sr as f32;
                phase.sin()
            })
            .collect();
        let med = median_f0(&track_pitch(&x, sr)).unwrap();
        assert!((med - 220.0).abs() / 220.0 < 0.05, "vibrato median {med} vs 220");
    }
}
