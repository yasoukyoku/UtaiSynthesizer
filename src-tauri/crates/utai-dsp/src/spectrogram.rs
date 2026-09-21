//! Phase 5-1: offline spectrogram for the `spectrogram` analysis node.
//!
//! Fixed, self-contained analysis parameters (plan §Phase 5: do NOT mix these
//! with `vr.rs`'s librosa-style inference STFT — 5-5 spectralCompare compares
//! A/B through THESE parameters, so changing them invalidates old comparisons).
//! dB floor is a hard -120 dB; the color map is an inferno approximation.

use rustfft::{num_complex::Complex, FftPlanner};

/// Analysis FFT size (fixed contract for 5-1/5-5).
pub const N_FFT: usize = 2048;
/// Analysis hop (fixed contract for 5-1/5-5).
pub const HOP: usize = 512;
/// Hard dB floor (silence renders as the map's zero color, not -inf).
pub const MIN_DB: f32 = -120.0;
/// Max rendered columns (wider inputs are max-pooled column-wise so
/// transients survive the downsample).
pub const MAX_COLS: usize = 1024;

#[derive(Debug, Clone)]
pub struct Spectrogram {
    /// [freq_bin][frame] dB matrix; bins = N_FFT/2 + 1.
    pub db: Vec<Vec<f32>>,
    pub sample_rate: u32,
    pub peak_db: f32,
}

/// Linear-magnitude STFT [bin][frame] — shared by spectrogram/timbre/compare
/// so every analysis node sees byte-identical frames.
pub fn magnitude_frames(x: &[f32]) -> Vec<Vec<f32>> {
    let bins = N_FFT / 2 + 1;
    let win = hann(N_FFT);
    if x.is_empty() {
        return vec![Vec::new(); bins];
    }
    // Center-pad with zeros (analysis-grade; no reflect padding needed here).
    let pad = N_FFT / 2;
    let total = pad + x.len() + pad;
    let n_frames = total.saturating_sub(N_FFT) / HOP + 1;
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut buf = vec![Complex::new(0.0f32, 0.0f32); N_FFT];
    let mut out = vec![vec![0.0f32; n_frames]; bins];
    for f in 0..n_frames {
        let start = f * HOP;
        for (i, b) in buf.iter_mut().enumerate() {
            let s = start + i;
            let v = if s >= pad && s < pad + x.len() { x[s - pad] } else { 0.0 };
            *b = Complex::new(v * win[i], 0.0);
        }
        fft.process(&mut buf);
        for (bin, row) in out.iter_mut().enumerate() {
            let c = buf[bin];
            row[f] = (c.re * c.re + c.im * c.im).sqrt();
        }
    }
    out
}

/// dB spectrogram with the analysis contract (floor -120 dB).
pub fn spectrogram_db(x: &[f32], sample_rate: u32) -> Spectrogram {
    let mag = magnitude_frames(x);
    let mut peak_db = MIN_DB;
    let db: Vec<Vec<f32>> = mag
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|m| {
                    let d = (20.0 * (m + 1e-10).log10()).max(MIN_DB);
                    if d > peak_db {
                        peak_db = d;
                    }
                    d
                })
                .collect()
        })
        .collect();
    Spectrogram { db, sample_rate, peak_db }
}

/// Render to 8-bit RGB (row-major, top row = highest frequency — the
/// conventional spectrogram orientation). Returns (pixels, width, height).
pub fn to_rgb(spec: &Spectrogram, max_cols: usize) -> (Vec<u8>, usize, usize) {
    let rows = spec.db.len();
    let cols = spec.db.first().map(|r| r.len()).unwrap_or(0);
    if rows == 0 || cols == 0 {
        return (Vec::new(), 0, 0);
    }
    let max_cols = max_cols.max(1).min(MAX_COLS);
    // Max-pool column groups (preserves transient peaks that averaging would smear).
    let group = ((cols + max_cols - 1) / max_cols).max(1);
    let w = (cols + group - 1) / group;
    let range = (spec.peak_db - MIN_DB).max(1e-6);
    let mut pixels = vec![0u8; w * rows * 3];
    for (wcol, out_col) in (0..cols).step_by(group).zip(0..w) {
        let end = (wcol + group).min(cols);
        for (r, row) in spec.db.iter().enumerate() {
            let mut m = MIN_DB;
            for c in wcol..end {
                m = m.max(row[c]);
            }
            let v = ((m - MIN_DB) / range).clamp(0.0, 1.0);
            let (rr, gg, bb) = inferno(v);
            let o = ((rows - 1 - r) * w + out_col) * 3;
            pixels[o] = rr;
            pixels[o + 1] = gg;
            pixels[o + 2] = bb;
        }
    }
    (pixels, w, rows)
}

/// Inferno-approximation gradient (5 stops, linear interpolation).
fn inferno(v: f32) -> (u8, u8, u8) {
    const STOPS: [(f32, [u8; 3]); 5] = [
        (0.00, [0, 0, 4]),
        (0.25, [87, 16, 110]),
        (0.50, [188, 55, 84]),
        (0.75, [249, 142, 9]),
        (1.00, [252, 255, 164]),
    ];
    let v = v.clamp(0.0, 1.0);
    let mut i = 0;
    while i < STOPS.len() - 2 && v > STOPS[i + 1].0 {
        i += 1;
    }
    let (a, ca) = STOPS[i];
    let (b, cb) = STOPS[i + 1];
    let t = if b > a { (v - a) / (b - a) } else { 0.0 };
    let mix = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round().clamp(0.0, 255.0) as u8;
    (mix(ca[0], cb[0]), mix(ca[1], cb[1]), mix(ca[2], cb[2]))
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
            .map(|i| (2.0 * std::f32::consts::PI * f0 * i as f32 / sr as f32).sin() * 0.5)
            .collect()
    }

    #[test]
    fn sine_peaks_at_the_right_bin() {
        let sr = 44100u32;
        let spec = spectrogram_db(&sine(sr, 0.5, 1000.0), sr);
        assert_eq!(spec.db.len(), N_FFT / 2 + 1);
        let frames = spec.db[0].len();
        assert!(frames > 30, "0.5 s @ hop 512 must give ~42 frames, got {frames}");
        // Sum energy per bin, find the max.
        let (best_bin, _) = spec
            .db
            .iter()
            .enumerate()
            .map(|(i, row)| (i, row.iter().sum::<f32>()))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        let expect = (1000.0 / (sr as f32 / N_FFT as f32)).round() as usize;
        assert!((best_bin as i32 - expect as i32).abs() <= 2, "bin {best_bin} vs {expect}");
        // Raw-STFT magnitude scale (rustfft does not normalize): a 0.5-amp
        // hann-windowed sine peaks near amp·coherent_gain·N/4 ≈ 256 → ~48 dB.
        assert!(spec.peak_db > 40.0 && spec.peak_db < 60.0, "peak_db {}", spec.peak_db);
    }

    #[test]
    fn silence_floors_at_min_db() {
        let spec = spectrogram_db(&vec![0.0f32; 44100], 44100);
        assert!(spec.db.iter().all(|r| r.iter().all(|&d| d == MIN_DB)));
        assert_eq!(spec.peak_db, MIN_DB);
    }

    #[test]
    fn rgb_dimensions_and_orientation() {
        let sr = 44100u32;
        let spec = spectrogram_db(&sine(sr, 0.3, 440.0), sr);
        let (px, w, h) = to_rgb(&spec, 256);
        assert_eq!(w, 256.min(spec.db[0].len()));
        assert_eq!(h, N_FFT / 2 + 1);
        assert_eq!(px.len(), w * h * 3);
        // 440 Hz maps to bin 20 of 1025 → near the BOTTOM of the flipped image.
        let bin = (440.0 / (sr as f32 / N_FFT as f32)).round() as usize;
        let bottom_sum: u32 = px[((h - 1 - bin) * w) * 3..((h - 1 - bin) * w + 1) * 3]
            .iter()
            .map(|&b| b as u32)
            .sum();
        let top_sum: u32 = px[0..3].iter().map(|&b| b as u32).sum();
        assert!(bottom_sum > top_sum, "high-energy row (440 Hz) must be near the image bottom");
    }

    #[test]
    fn empty_and_tiny_inputs_do_not_panic() {
        let spec = spectrogram_db(&[], 44100);
        let (px, w, h) = to_rgb(&spec, 256);
        assert_eq!((px.len(), w, h), (0, 0, 0));
        let spec = spectrogram_db(&[0.5f32; 37], 44100);
        let (_, w, h) = to_rgb(&spec, 256);
        assert!(w > 0 && h == N_FFT / 2 + 1);
    }
}
