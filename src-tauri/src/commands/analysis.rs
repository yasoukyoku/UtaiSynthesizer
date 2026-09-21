//! Phase 5: analysis / visualization IPC commands (分析可视化节点).
//!
//! Six commands over the utai-dsp analysis kernels — all zero-damage (they
//! decode, analyze and report; only `analyze_spectrogram` writes a file, and
//! that file IS the artifact: the inferno PNG):
//!   - `analyze_harmonicity(input, f0_hz)` — 谐波健康报告 (5-4 wiring; the
//!     kernels existed since S163, this is their workflow-node face)
//!   - `track_f0(input)` — full NCCF pitch track for f0Curve (5-2)
//!   - `analyze_spectrogram(input, output)` — PNG + geometry report (5-1)
//!   - `analyze_timbre(input)` — centroid / rolloff / ZCR / MFCC (5-3)
//!   - `compare_spectra(input_a, input_b)` — per-band dB diff + log-mel (5-5)
//!   - `dtw_compare(input_a, input_b, band)` — log-mel DTW (5-6)
//!
//! Analysis STFT params are the fixed contract inside utai-dsp's
//! `spectrogram`/`timbre` (deliberately NOT the vr.rs inference conventions —
//! plan §Phase 5). Mono-side kernels ride a plain channel-average down-mix;
//! two-input commands polyphase-resample B to A's rate when rates differ.

use std::path::Path;

use serde::Serialize;
use utai_dsp::dtw::{self, DtwResult};
use utai_dsp::harmonicity;
use utai_dsp::pitch::{self, PitchFrame};
use utai_dsp::spectrogram::{self, Spectrogram};
use utai_dsp::timbre;

use crate::audio::{load_audio, AudioBuffer};
use crate::{Result, UtaiError};

macro_rules! require_non_empty {
    ($buf:expr) => {
        if $buf.samples.is_empty() {
            return Err(UtaiError::Audio("Empty audio file".into()));
        }
    };
}

/// Channel-average down-mix (every analysis kernel wants mono).
fn downmix_mono(buf: &AudioBuffer) -> Vec<f32> {
    let ch = buf.channels.max(1) as usize;
    buf.samples
        .chunks(ch)
        .map(|fr| fr.iter().sum::<f32>() / fr.len().max(1) as f32)
        .collect()
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Polyphase-resample `x` from `from` Hz to `to` Hz (no-op when equal).
fn resample_to(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let g = gcd(from, to);
    utai_dsp::vr::resample_poly_f32(x, (to / g) as usize, (from / g) as usize)
}

// ─── 5-4 harmonicity ─────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct HarmonicityReport {
    /// Auto-detected f0 (median of the NCCF track). None = no voiced frames.
    pub detected_f0_hz: Option<f32>,
    /// The f0 the metrics actually used (requested if given, else detected).
    pub f0_hz_used: Option<f32>,
    pub energy_fraction_db: Option<f32>,
    pub peak_width_pct: Option<f32>,
    pub comb_depth_db: Option<f32>,
    pub upper_harmonic_sag_db: Option<f32>,
    pub upper_harmonic_level_db: Option<f32>,
    pub second_harmonic_db: Option<f32>,
    pub shimmer_db: Option<f32>,
    /// Only in requested mode: cents between the requested target and the
    /// actual pitch.
    pub pitch_error_cents: Option<f32>,
    pub channels: u16,
    pub sample_rate: u32,
    pub duration_secs: f64,
}

/// 谐波健康检查 (5-4 接线)。Pass `f0_hz` to evaluate against a known target
/// (also enables `pitch_error_cents`); omit it to auto-detect via the NCCF
/// tracker. Metrics are `null` when the kernel's guards reject the input
/// (too short, no in-band energy, …) — that is data, not an error.
#[tauri::command]
pub fn analyze_harmonicity(input: String, f0_hz: Option<f64>) -> Result<HarmonicityReport> {
    let buf = load_audio(Path::new(&input))?;
    require_non_empty!(buf);
    let mono = downmix_mono(&buf);

    let detected = pitch::median_f0(&pitch::track_pitch(&mono, buf.sample_rate));
    let requested = f0_hz.map(|v| v as f32);
    let used = requested.or(detected);

    let m = |f: fn(&[f32], u32, f32) -> Option<f32>| used.and_then(|f0| f(&mono, buf.sample_rate, f0));

    Ok(HarmonicityReport {
        detected_f0_hz: detected,
        f0_hz_used: used,
        energy_fraction_db: m(harmonicity::harmonic_energy_fraction_db),
        peak_width_pct: m(harmonicity::harmonic_peak_width_pct),
        comb_depth_db: m(harmonicity::comb_depth_db),
        upper_harmonic_sag_db: m(harmonicity::upper_harmonic_sag_db),
        upper_harmonic_level_db: m(harmonicity::upper_harmonic_level_db),
        second_harmonic_db: m(harmonicity::second_harmonic_level_db),
        shimmer_db: m(harmonicity::shimmer_db),
        pitch_error_cents: match (requested, used) {
            (Some(_), Some(f0)) => harmonicity::pitch_error_cents(&mono, buf.sample_rate, f0),
            _ => None,
        },
        channels: buf.channels,
        sample_rate: buf.sample_rate,
        duration_secs: buf.duration_secs(),
    })
}

// ─── 5-2 f0 track ────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct F0FrameDto {
    /// Frame-center time (s).
    pub t: f32,
    /// f0 in Hz; 0 = unvoiced.
    pub f0: f32,
    /// NCCF peak (0..1].
    pub clarity: f32,
}

#[derive(Debug, Serialize)]
pub struct F0TrackReport {
    pub median_f0_hz: Option<f32>,
    /// Voiced frames / total frames (0 when the clip is shorter than one frame).
    pub voiced_ratio: f32,
    /// Full-resolution track (~86 fps @ 44.1 kHz) — the f0Curve payload.
    pub frames: Vec<F0FrameDto>,
    pub sample_rate: u32,
    pub duration_secs: f64,
}

/// f0Curve payload (5-2).
#[tauri::command]
pub fn track_f0(input: String) -> Result<F0TrackReport> {
    let buf = load_audio(Path::new(&input))?;
    require_non_empty!(buf);
    let mono = downmix_mono(&buf);
    let track: Vec<PitchFrame> = pitch::track_pitch(&mono, buf.sample_rate);
    let voiced = track.iter().filter(|f| f.f0_hz > 0.0).count();
    Ok(F0TrackReport {
        median_f0_hz: pitch::median_f0(&track),
        voiced_ratio: if track.is_empty() {
            0.0
        } else {
            voiced as f32 / track.len() as f32
        },
        frames: track
            .into_iter()
            .map(|f| F0FrameDto { t: f.time_secs, f0: f.f0_hz, clarity: f.clarity })
            .collect(),
        sample_rate: buf.sample_rate,
        duration_secs: buf.duration_secs(),
    })
}

// ─── 5-1 spectrogram PNG ─────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct SpectrogramReport {
    pub png_path: String,
    /// Rendered PNG size in pixels (top row = highest frequency).
    pub width: u32,
    pub height: u32,
    /// Loudest analysis peak (dB, raw-STFT scale — see spectrogram.rs).
    pub peak_db: f32,
    pub bins: usize,
    pub frames: usize,
    pub sample_rate: u32,
    pub duration_secs: f64,
}

/// Spectrogram artifact (5-1): renders the fixed-contract analysis STFT as
/// an inferno PNG and returns its geometry. The dB matrix itself is NOT
/// shipped over IPC — the PNG is the artifact the node previews inline.
#[tauri::command]
pub fn analyze_spectrogram(input: String, output: String) -> Result<SpectrogramReport> {
    let buf = load_audio(Path::new(&input))?;
    require_non_empty!(buf);
    let mono = downmix_mono(&buf);

    let spec = spectrogram::spectrogram_db(&mono, buf.sample_rate);
    let (rgb, w, h) = spectrogram::to_rgb(&spec, spectrogram::MAX_COLS);

    let mut img = image::RgbImage::new(w as u32, h as u32);
    for (i, px) in rgb.chunks_exact(3).enumerate() {
        img.put_pixel((i % w) as u32, (i / w) as u32, image::Rgb([px[0], px[1], px[2]]));
    }
    img.save(&output).map_err(|e| UtaiError::Audio(format!("PNG encode failed: {e}")))?;

    Ok(SpectrogramReport {
        png_path: output,
        width: w as u32,
        height: h as u32,
        peak_db: spec.peak_db,
        bins: spec.db.len(),
        frames: spec.db.first().map_or(0, |r| r.len()),
        sample_rate: buf.sample_rate,
        duration_secs: buf.duration_secs(),
    })
}

// ─── 5-3 timbre ──────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct TimbreMetricsReport {
    pub centroid_hz: f32,
    pub rolloff85_hz: f32,
    pub zcr: f32,
    pub mfcc_mean: Vec<f32>,
    pub mfcc_std: Vec<f32>,
    pub frames: usize,
    pub channels: u16,
    pub sample_rate: u32,
    pub duration_secs: f64,
}

/// Timbre metrics (5-3).
#[tauri::command]
pub fn analyze_timbre(input: String) -> Result<TimbreMetricsReport> {
    let buf = load_audio(Path::new(&input))?;
    require_non_empty!(buf);
    let mono = downmix_mono(&buf);
    let t = timbre::analyze_timbre(&mono, buf.sample_rate);
    Ok(TimbreMetricsReport {
        centroid_hz: t.centroid_hz,
        rolloff85_hz: t.rolloff85_hz,
        zcr: t.zcr,
        mfcc_mean: t.mfcc_mean,
        mfcc_std: t.mfcc_std,
        frames: t.frames,
        channels: buf.channels,
        sample_rate: buf.sample_rate,
        duration_secs: buf.duration_secs(),
    })
}

// ─── 5-5 spectral compare ────────────────────────────────────────

/// Octave-band edges (Hz). The top edge gets clamped to 0.45·sr.
const BAND_EDGES_HZ: [f32; 9] =
    [60.0, 120.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 20000.0];

#[derive(Debug, Serialize)]
pub struct BandDiff {
    pub label: String,
    pub lo_hz: f32,
    pub hi_hz: f32,
    pub mean_db_a: f32,
    pub mean_db_b: f32,
    /// mean_db_a − mean_db_b (positive = A louder in this band).
    pub diff_db: f32,
}

#[derive(Debug, Serialize)]
pub struct SpectraCompareReport {
    pub bands: Vec<BandDiff>,
    /// Cosine similarity of the mean log-mel vectors (1 = same timbre mix).
    pub logmel_cosine: f32,
    /// Euclidean distance between the mean log-mel vectors.
    pub logmel_l2: f32,
    pub mean_db_a: f32,
    pub mean_db_b: f32,
    /// B was resampled to A's rate (the inputs' rates differed).
    pub resampled: bool,
    pub sample_rate: u32,
    pub duration_a_secs: f64,
    pub duration_b_secs: f64,
}

fn hz_label(hz: f32) -> String {
    if hz >= 1000.0 {
        format!("{}k", hz / 1000.0)
    } else {
        format!("{}", hz as u32)
    }
}

fn band_label(lo: f32, hi: f32, top: bool) -> String {
    if top {
        format!("{}+", hz_label(lo))
    } else {
        format!("{}-{}", hz_label(lo), hz_label(hi))
    }
}

/// Mean dB of one frequency band of a `spectrogram_db` matrix.
fn band_mean_db(spec: &Spectrogram, sample_rate: u32, lo: f32, hi: f32) -> Option<f32> {
    let bin_hz = sample_rate as f32 / spectrogram::N_FFT as f32;
    let k0 = ((lo / bin_hz).ceil() as usize).max(1);
    let k1 = ((hi / bin_hz).floor() as usize).min(spec.db.len() - 1);
    if k0 > k1 {
        return None;
    }
    let (mut sum, mut n) = (0.0f32, 0usize);
    for row in &spec.db[k0..=k1] {
        for &v in row {
            sum += v;
            n += 1;
        }
    }
    (n > 0).then(|| sum / n as f32)
}

fn mean_all_db(spec: &Spectrogram) -> f32 {
    let (mut sum, mut n) = (0.0f32, 0usize);
    for row in &spec.db {
        for &v in row {
            sum += v;
            n += 1;
        }
    }
    if n == 0 { 0.0 } else { sum / n as f32 }
}

fn mean_logmel(frames: &[Vec<f32>]) -> Option<Vec<f32>> {
    let dim = frames.first()?.len();
    let mut out = vec![0.0f32; dim];
    for f in frames {
        for (o, &v) in out.iter_mut().zip(f) {
            *o += v;
        }
    }
    let n = frames.len() as f32;
    for o in out.iter_mut() {
        *o /= n;
    }
    Some(out)
}

/// (cosine similarity, L2 distance) of the two mean log-mel vectors.
fn mean_logmel_compare(a: &[Vec<f32>], b: &[Vec<f32>]) -> (f32, f32) {
    match (mean_logmel(a), mean_logmel(b)) {
        (Some(ma), Some(mb)) => {
            let l2 = ma
                .iter()
                .zip(&mb)
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt();
            (1.0 - dtw::cosine_distance(&ma, &mb), l2)
        }
        _ => (0.0, 0.0),
    }
}

/// A/B spectral compare (5-5): octave-band dB diffs over the fixed analysis
/// STFT plus a length-agnostic log-mel summary. B is resampled to A's rate
/// when needed.
#[tauri::command]
pub fn compare_spectra(input_a: String, input_b: String) -> Result<SpectraCompareReport> {
    let buf_a = load_audio(Path::new(&input_a))?;
    require_non_empty!(buf_a);
    let buf_b = load_audio(Path::new(&input_b))?;
    require_non_empty!(buf_b);

    let mono_a = downmix_mono(&buf_a);
    let mono_b = downmix_mono(&buf_b);
    let resampled = buf_b.sample_rate != buf_a.sample_rate;
    let mono_b = resample_to(&mono_b, buf_b.sample_rate, buf_a.sample_rate);
    let sr = buf_a.sample_rate;

    let spec_a = spectrogram::spectrogram_db(&mono_a, sr);
    let spec_b = spectrogram::spectrogram_db(&mono_b, sr);

    let nyq = sr as f32 * 0.45;
    let mut bands = Vec::new();
    for i in 0..BAND_EDGES_HZ.len() - 1 {
        let lo = BAND_EDGES_HZ[i];
        let hi = BAND_EDGES_HZ[i + 1].min(nyq);
        if hi <= lo {
            continue;
        }
        if let (Some(ma), Some(mb)) =
            (band_mean_db(&spec_a, sr, lo, hi), band_mean_db(&spec_b, sr, lo, hi))
        {
            bands.push(BandDiff {
                label: band_label(lo, hi, i == BAND_EDGES_HZ.len() - 2),
                lo_hz: lo,
                hi_hz: hi,
                mean_db_a: ma,
                mean_db_b: mb,
                diff_db: ma - mb,
            });
        }
    }

    let feats_a = timbre::log_mel_frames(&mono_a, sr);
    let feats_b = timbre::log_mel_frames(&mono_b, sr);
    let (logmel_cosine, logmel_l2) = mean_logmel_compare(&feats_a, &feats_b);

    Ok(SpectraCompareReport {
        mean_db_a: mean_all_db(&spec_a),
        mean_db_b: mean_all_db(&spec_b),
        bands,
        logmel_cosine,
        logmel_l2,
        resampled,
        sample_rate: sr,
        duration_a_secs: buf_a.duration_secs(),
        duration_b_secs: buf_b.duration_secs(),
    })
}

// ─── 5-6 DTW ─────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct DtwCompareReport {
    pub cost: f32,
    /// cost / path_len — the length-normalized similarity score.
    pub mean_step_cost: f32,
    pub path_len: usize,
    pub a_frames: usize,
    pub b_frames: usize,
    /// Sakoe–Chiba band used (0 = unrestricted).
    pub band: usize,
    pub resampled: bool,
    pub sample_rate: u32,
    pub duration_a_secs: f64,
    pub duration_b_secs: f64,
}

/// DTW over log-mel sequences (5-6). `band` = max frame drift
/// (Sakoe–Chiba); 0 / omitted = unrestricted.
#[tauri::command]
pub fn dtw_compare(
    input_a: String,
    input_b: String,
    band: Option<usize>,
) -> Result<DtwCompareReport> {
    let buf_a = load_audio(Path::new(&input_a))?;
    require_non_empty!(buf_a);
    let buf_b = load_audio(Path::new(&input_b))?;
    require_non_empty!(buf_b);

    let mono_a = downmix_mono(&buf_a);
    let mono_b = downmix_mono(&buf_b);
    let resampled = buf_b.sample_rate != buf_a.sample_rate;
    let mono_b = resample_to(&mono_b, buf_b.sample_rate, buf_a.sample_rate);
    let sr = buf_a.sample_rate;

    let feats_a = timbre::log_mel_frames(&mono_a, sr);
    let feats_b = timbre::log_mel_frames(&mono_b, sr);
    let band_used = band.unwrap_or(0);
    let r: DtwResult = dtw::dtw_align(&feats_a, &feats_b, band_used).ok_or_else(|| {
        UtaiError::Audio("dtw_compare: one side produced no analysis frames".into())
    })?;

    Ok(DtwCompareReport {
        cost: r.cost,
        mean_step_cost: r.mean_step_cost,
        path_len: r.path_len,
        a_frames: r.a_frames,
        b_frames: r.b_frames,
        band: band_used,
        resampled,
        sample_rate: sr,
        duration_a_secs: buf_a.duration_secs(),
        duration_b_secs: buf_b.duration_secs(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::save_wav_f32;
    use std::env::temp_dir;

    fn write_wav(path: &Path, samples: Vec<f32>, sr: u32, ch: u16) {
        save_wav_f32(
            path,
            &AudioBuffer { samples, sample_rate: sr, channels: ch },
        )
        .unwrap();
    }

    fn sine(sr: u32, secs: f32, f0: f64) -> Vec<f32> {
        let n = (sr as f32 * secs) as usize;
        (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * f0 * i as f64 / sr as f64).sin() as f32 * 0.5)
            .collect()
    }

    /// f0 + 4 decaying harmonics — the only in-band energy is harmonic.
    fn harmonic_stack(sr: u32, secs: f32, f0: f64) -> Vec<f32> {
        let n = (sr as f32 * secs) as usize;
        (0..n)
            .map(|i| {
                let t = i as f64 / sr as f64;
                let w = 2.0 * std::f64::consts::PI * f0;
                let v = 0.5 * (w * t).sin()
                    + 0.25 * (2.0 * w * t).sin()
                    + 0.15 * (3.0 * w * t).sin()
                    + 0.10 * (4.0 * w * t).sin()
                    + 0.05 * (5.0 * w * t).sin();
                v as f32
            })
            .collect()
    }

    #[test]
    fn harmonicity_auto_f0_and_requested() {
        let dir = temp_dir();
        let input = dir.join("utai_harm_in.wav");
        // 880 Hz, not 220: second_harmonic_level_db verifies the found peaks sit
        // within 1.5% of f0/2f0, and the 4096-pt FFT's ~10.8 Hz bin resolution is
        // only fine enough for that above ~700 Hz.
        write_wav(&input, harmonic_stack(44100, 0.5, 880.0), 44100, 1);

        let auto = analyze_harmonicity(input.to_str().unwrap().into(), None).unwrap();
        let f0 = auto.detected_f0_hz.expect("harmonic stack must be voiced");
        assert!((f0 - 880.0).abs() < 10.0, "detected {f0}");
        assert_eq!(auto.f0_hz_used, Some(f0));
        let frac = auto.energy_fraction_db.expect("harmonics present");
        assert!(frac > -4.0, "band energy is essentially all harmonic: {frac}");
        assert!(auto.second_harmonic_db.is_some());
        assert!(auto.pitch_error_cents.is_none(), "no target → no cents");

        let req = analyze_harmonicity(input.to_str().unwrap().into(), Some(880.0)).unwrap();
        assert_eq!(req.f0_hz_used, Some(880.0));
        let cents = req.pitch_error_cents.unwrap();
        assert!(cents.abs() < 25.0, "tone on target: {cents}");
    }

    #[test]
    fn harmonicity_rejects_empty() {
        let dir = temp_dir();
        let input = dir.join("utai_harm_empty.wav");
        write_wav(&input, vec![], 44100, 1);
        assert!(analyze_harmonicity(input.to_str().unwrap().into(), None).is_err());
    }

    #[test]
    fn f0_track_finds_440() {
        let dir = temp_dir();
        let input = dir.join("utai_f0_in.wav");
        write_wav(&input, sine(44100, 0.5, 440.0), 44100, 1);
        let r = track_f0(input.to_str().unwrap().into()).unwrap();
        assert!(r.frames.len() > 20, "{} frames", r.frames.len());
        assert!(r.voiced_ratio > 0.8, "{}", r.voiced_ratio);
        let med = r.median_f0_hz.unwrap();
        assert!((med - 440.0).abs() < 3.0, "median {med}");
        assert!((r.duration_secs - 0.5).abs() < 1e-3);
    }

    #[test]
    fn spectrogram_png_written_and_decodable() {
        let dir = temp_dir();
        let input = dir.join("utai_spec_in.wav");
        let png = dir.join("utai_spec_out.png");
        write_wav(&input, sine(44100, 1.0, 1000.0), 44100, 1);
        let r = analyze_spectrogram(
            input.to_str().unwrap().into(),
            png.to_str().unwrap().into(),
        )
        .unwrap();
        assert_eq!(r.bins, spectrogram::N_FFT / 2 + 1);
        assert!(r.frames > 80, "{}", r.frames);
        assert!(
            r.peak_db > 40.0 && r.peak_db < 60.0,
            "raw-STFT scale sine peak: {}",
            r.peak_db
        );
        let bytes = std::fs::read(&png).expect("png written");
        let img = image::load_from_memory(&bytes).unwrap();
        assert_eq!((img.width(), img.height()), (r.width, r.height));
        assert_eq!(img.height(), spectrogram::N_FFT as u32 / 2 + 1);
    }

    #[test]
    fn timbre_metrics_on_pure_sine() {
        let dir = temp_dir();
        let input = dir.join("utai_timbre_in.wav");
        // Proper interleaved stereo — a mono vector declared as 2ch would halve
        // the frame count and double the perceived pitch.
        let stereo: Vec<f32> =
            sine(44100, 0.5, 1000.0).iter().flat_map(|&v| [v, v]).collect();
        write_wav(&input, stereo, 44100, 2);
        let r = analyze_timbre(input.to_str().unwrap().into()).unwrap();
        assert_eq!(r.channels, 2);
        assert!(r.frames > 0);
        assert!(
            r.centroid_hz > 800.0 && r.centroid_hz < 1300.0,
            "centroid {}",
            r.centroid_hz
        );
        let zcr_expected = 2.0 * 1000.0 / 44100.0;
        assert!(
            (r.zcr - zcr_expected).abs() < 0.015,
            "{} vs {zcr_expected}",
            r.zcr
        );
        assert_eq!(r.mfcc_mean.len(), timbre::N_MFCC);
        assert_eq!(r.mfcc_std.len(), timbre::N_MFCC);
    }

    #[test]
    fn spectra_compare_same_and_different() {
        let dir = temp_dir();
        let a = dir.join("utai_cmp_a.wav");
        let b = dir.join("utai_cmp_b.wav");
        write_wav(&a, sine(44100, 0.5, 1000.0), 44100, 1);
        write_wav(&b, sine(44100, 0.5, 200.0), 44100, 1);

        let same =
            compare_spectra(a.to_str().unwrap().into(), a.to_str().unwrap().into()).unwrap();
        assert!(!same.resampled);
        assert!(same.logmel_cosine > 0.999, "{}", same.logmel_cosine);
        assert!(same.logmel_l2 < 1e-6, "{}", same.logmel_l2);
        for band in &same.bands {
            assert!(band.diff_db.abs() < 0.5, "{}: {}", band.label, band.diff_db);
        }

        let diff =
            compare_spectra(a.to_str().unwrap().into(), b.to_str().unwrap().into()).unwrap();
        // Log-mel floors (ln 1e-10 ≈ -23) dominate most bins, keeping cosine
        // similarity high even for far-apart tones — assert the relative
        // ordering and the L2 magnitude instead of an absolute ceiling.
        assert!(
            diff.logmel_cosine < same.logmel_cosine - 0.05,
            "{} vs {}",
            diff.logmel_cosine,
            same.logmel_cosine
        );
        assert!(diff.logmel_l2 > 1.0, "{}", diff.logmel_l2);
        let high = diff.bands.iter().find(|x| x.label == "1k-2k").unwrap();
        assert!(high.diff_db > 20.0, "1k sine louder up there: {}", high.diff_db);
        let low = diff.bands.iter().find(|x| x.label == "120-250").unwrap();
        assert!(low.diff_db < -20.0, "200 Hz louder down there: {}", low.diff_db);
    }

    #[test]
    fn dtw_same_near_zero_and_resample_branch() {
        let dir = temp_dir();
        let a = dir.join("utai_dtw_a.wav");
        let c = dir.join("utai_dtw_c.wav");
        write_wav(&a, sine(44100, 0.5, 1000.0), 44100, 1);
        write_wav(&c, sine(22050, 0.5, 1000.0), 22050, 1);

        let same =
            dtw_compare(a.to_str().unwrap().into(), a.to_str().unwrap().into(), None).unwrap();
        assert_eq!(same.a_frames, same.b_frames);
        assert_eq!(same.path_len, same.a_frames, "identical inputs walk the diagonal");
        assert!(same.mean_step_cost < 0.05, "{}", same.mean_step_cost);

        let rs =
            dtw_compare(a.to_str().unwrap().into(), c.to_str().unwrap().into(), None).unwrap();
        assert!(rs.resampled, "22050 → 44100");
        assert!(
            rs.mean_step_cost < 0.3,
            "same tone after resample: {}",
            rs.mean_step_cost
        );
    }

    #[test]
    fn dtw_distant_inputs_score_high() {
        let dir = temp_dir();
        let a = dir.join("utai_dtw_x.wav");
        let b = dir.join("utai_dtw_y.wav");
        write_wav(&a, sine(44100, 0.5, 1000.0), 44100, 1);
        write_wav(&b, sine(44100, 0.5, 200.0), 44100, 1);
        let r =
            dtw_compare(a.to_str().unwrap().into(), b.to_str().unwrap().into(), Some(0)).unwrap();
        // Log-mel floors cap the max cosine distance for even far-apart tones;
        // > 0.1 still separates cleanly from the identical-input floor (< 0.05).
        assert!(r.mean_step_cost > 0.1, "{}", r.mean_step_cost);
        assert_eq!(r.band, 0);
    }

    #[test]
    fn empty_inputs_rejected() {
        let dir = temp_dir();
        let e = dir.join("utai_empty.wav");
        write_wav(&e, vec![], 44100, 1);
        assert!(track_f0(e.to_str().unwrap().into()).is_err());
        assert!(analyze_timbre(e.to_str().unwrap().into()).is_err());
        assert!(
            analyze_spectrogram(
                e.to_str().unwrap().into(),
                dir.join("utai_no.png").to_str().unwrap().into()
            )
            .is_err()
        );
    }
}
