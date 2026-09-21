//! Phase 5-3: timbre metrics (spectral centroid / 85 % rolloff / ZCR / MFCC)
//! for the `timbreMetrics` analysis node, plus the log-mel feature frames that
//! Phase 5-6's DTW consumes.
//!
//! All spectral work rides `spectrogram::magnitude_frames` so 5-1/5-3/5-5/5-6
//! see byte-identical STFT frames (plan §Phase 5: analysis params are a fixed,
//! self-contained contract — never mixed with the inference-side STFT).

use crate::spectrogram::{magnitude_frames, N_FFT};

/// Mel bands for MFCC / DTW features (HTK-style triangles).
pub const N_MELS: usize = 40;
/// MFCC coefficients kept (c0..c12).
pub const N_MFCC: usize = 13;
/// Rolloff percentile (librosa's default).
const ROLLOFF_PCT: f32 = 0.85;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct TimbreReport {
    /// Median per-frame spectral centroid (Hz).
    pub centroid_hz: f32,
    /// Median per-frame 85 % spectral rolloff (Hz).
    pub rolloff85_hz: f32,
    /// Zero-crossing rate over the whole clip.
    pub zcr: f32,
    /// Mean MFCC vector across frames (length N_MFCC).
    pub mfcc_mean: Vec<f32>,
    /// Std-dev MFCC vector across frames (length N_MFCC).
    pub mfcc_std: Vec<f32>,
    /// Analysis frames processed.
    pub frames: usize,
}

/// HTK mel scale.
pub fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// HTK mel scale, inverse.
pub fn mel_to_hz(m: f32) -> f32 {
    700.0 * (10.0f32.powf(m / 2595.0) - 1.0)
}

/// Sparse triangular mel filterbank: one (bin, weight) list per band.
fn mel_filterbank(n_mels: usize, n_fft: usize, sample_rate: u32) -> Vec<Vec<(usize, f32)>> {
    let bins = n_fft / 2 + 1;
    let m_min = hz_to_mel(0.0);
    let m_max = hz_to_mel(sample_rate as f32 / 2.0);
    let pts: Vec<f32> = (0..n_mels + 2)
        .map(|i| mel_to_hz(m_min + (m_max - m_min) * i as f32 / (n_mels + 1) as f32))
        .collect();
    let bin_hz = sample_rate as f32 / n_fft as f32;
    (0..n_mels)
        .map(|m| {
            let (lo, mid, hi) = (pts[m], pts[m + 1], pts[m + 2]);
            (0..bins)
                .filter_map(|b| {
                    let f = b as f32 * bin_hz;
                    let w = if f >= lo && f <= mid && mid > lo {
                        (f - lo) / (mid - lo)
                    } else if f > mid && f <= hi && hi > mid {
                        (hi - f) / (hi - mid)
                    } else {
                        0.0
                    };
                    (w > 0.0).then_some((b, w))
                })
                .collect()
        })
        .collect()
}

/// Log-mel energies from an existing magnitude STFT (no second FFT pass).
fn log_mel_from_mag(mag: &[Vec<f32>], sample_rate: u32) -> Vec<Vec<f32>> {
    let frames = mag.first().map(|r| r.len()).unwrap_or(0);
    let fb = mel_filterbank(N_MELS, N_FFT, sample_rate);
    (0..frames)
        .map(|f| {
            fb.iter()
                .map(|band| {
                    let e: f32 = band
                        .iter()
                        .map(|&(b, w)| {
                            let m = mag[b][f];
                            m * m * w
                        })
                        .sum();
                    (e + 1e-10).ln()
                })
                .collect()
        })
        .collect()
}

/// Log-mel feature frames [frame][N_MELS] — DTW's input (Phase 5-6).
pub fn log_mel_frames(x: &[f32], sample_rate: u32) -> Vec<Vec<f32>> {
    log_mel_from_mag(&magnitude_frames(x), sample_rate)
}

/// Orthonormal DCT-II (what classic MFCC pipelines apply to log-mel).
fn dct_ii(v: &[f32], n_out: usize) -> Vec<f32> {
    let n = v.len();
    (0..n_out)
        .map(|k| {
            let s: f32 = v
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    x * (std::f32::consts::PI / n as f32 * (i as f32 + 0.5) * k as f32).cos()
                })
                .sum();
            let norm = if k == 0 { (1.0 / n as f32).sqrt() } else { (2.0 / n as f32).sqrt() };
            s * norm
        })
        .collect()
}

fn median(v: &mut [f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = v.len() / 2;
    if v.len() % 2 == 1 { v[mid] } else { (v[mid - 1] + v[mid]) * 0.5 }
}

/// Full timbre report. Never panics on empty input (zeroed report).
pub fn analyze_timbre(x: &[f32], sample_rate: u32) -> TimbreReport {
    let mag = magnitude_frames(x);
    let frames = mag.first().map(|r| r.len()).unwrap_or(0);
    let bin_hz = sample_rate as f32 / N_FFT as f32;

    let mut centroids = Vec::with_capacity(frames);
    let mut rolloffs = Vec::with_capacity(frames);
    for f in 0..frames {
        let (mut num, mut den, mut total) = (0.0f32, 0.0f32, 0.0f32);
        for (b, row) in mag.iter().enumerate() {
            let m = row[f];
            num += b as f32 * bin_hz * m;
            den += m;
            total += m;
        }
        centroids.push(if den > 1e-10 { num / den } else { 0.0 });
        let target = ROLLOFF_PCT * total;
        let mut r = 0.0f32;
        let mut cum = 0.0f32;
        for (b, row) in mag.iter().enumerate() {
            cum += row[f];
            if cum >= target {
                r = b as f32 * bin_hz;
                break;
            }
        }
        rolloffs.push(r);
    }

    let zcr = if x.len() < 2 {
        0.0
    } else {
        let crossings = x.windows(2).filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0)).count();
        crossings as f32 / (x.len() - 1) as f32
    };

    let coefs: Vec<Vec<f32>> =
        log_mel_from_mag(&mag, sample_rate).iter().map(|v| dct_ii(v, N_MFCC)).collect();
    let mut mfcc_mean = vec![0.0f32; N_MFCC];
    let mut mfcc_std = vec![0.0f32; N_MFCC];
    if !coefs.is_empty() {
        for k in 0..N_MFCC {
            let col: Vec<f32> = coefs.iter().map(|c| c[k]).collect();
            let mean = col.iter().sum::<f32>() / col.len() as f32;
            let var = col.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / col.len() as f32;
            mfcc_mean[k] = mean;
            mfcc_std[k] = var.sqrt();
        }
    }

    TimbreReport {
        centroid_hz: median(&mut centroids),
        rolloff85_hz: median(&mut rolloffs),
        zcr,
        mfcc_mean,
        mfcc_std,
        frames,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(sr: u32, secs: f32, f0: f32, amp: f32) -> Vec<f32> {
        let n = (sr as f32 * secs) as usize;
        (0..n)
            .map(|i| amp * (2.0 * std::f32::consts::PI * f0 * i as f32 / sr as f32).sin())
            .collect()
    }

    #[test]
    fn centroid_and_rolloff_rank_by_brightness() {
        let sr = 44100u32;
        let low = analyze_timbre(&sine(sr, 0.4, 220.0, 0.5), sr);
        let high = analyze_timbre(&sine(sr, 0.4, 2000.0, 0.5), sr);
        assert!(
            low.centroid_hz > 150.0 && low.centroid_hz < 400.0,
            "220 Hz centroid {}",
            low.centroid_hz
        );
        assert!(
            high.centroid_hz > 1500.0 && high.centroid_hz < 2600.0,
            "2 kHz centroid {}",
            high.centroid_hz
        );
        assert!(high.rolloff85_hz > low.rolloff85_hz);
        assert!(high.zcr > low.zcr, "2 kHz ZCR {} vs 220 Hz {}", high.zcr, low.zcr);
    }

    #[test]
    fn silence_reports_zeroes_without_nan() {
        let sr = 44100u32;
        let r = analyze_timbre(&vec![0.0f32; sr as usize], sr);
        assert_eq!(r.centroid_hz, 0.0);
        assert_eq!(r.rolloff85_hz, 0.0);
        assert_eq!(r.zcr, 0.0);
        assert!(r.frames > 0);
        assert!(r.mfcc_mean.iter().all(|v| v.is_finite()));
        assert!(r.mfcc_std.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn mfcc_c0_tracks_loudness() {
        let sr = 44100u32;
        let loud = analyze_timbre(&sine(sr, 0.3, 440.0, 0.5), sr);
        let quiet = analyze_timbre(&sine(sr, 0.3, 440.0, 0.05), sr);
        assert_eq!(loud.mfcc_mean.len(), N_MFCC);
        assert_eq!(loud.mfcc_std.len(), N_MFCC);
        assert!(
            loud.mfcc_mean[0] > quiet.mfcc_mean[0],
            "c0 {} vs {}",
            loud.mfcc_mean[0],
            quiet.mfcc_mean[0]
        );
    }

    #[test]
    fn log_mel_shapes_match_contract() {
        let sr = 44100u32;
        let mels = log_mel_frames(&sine(sr, 0.2, 1000.0, 0.5), sr);
        assert!(mels.len() > 10, "0.2 s @ hop 512 must give ~17 frames");
        assert!(mels.iter().all(|f| f.len() == N_MELS));
        let mean: Vec<f32> = (0..N_MELS)
            .map(|k| mels.iter().map(|f| f[k]).sum::<f32>() / mels.len() as f32)
            .collect();
        let (bi, _) = mean.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
        assert!((7..=12).contains(&bi), "1 kHz must peak in a mid-low mel band, got {bi}");
    }

    #[test]
    fn empty_and_tiny_inputs_are_safe() {
        let r = analyze_timbre(&[], 44100);
        assert_eq!(r.frames, 0);
        assert_eq!(r.centroid_hz, 0.0);
        let r = analyze_timbre(&[0.3f32; 37], 44100);
        assert!(r.mfcc_mean.iter().all(|v| v.is_finite()));
        assert!(r.frames > 0);
    }
}
