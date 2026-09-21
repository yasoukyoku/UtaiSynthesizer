//! Phase 1: ITU-R BS.1770 loudness (LUFS) measurement + normalization.
//!
//! Two commands:
//!   - `measure_loudness(path)` → integrated LUFS + true peak (dBTP)
//!   - `normalize_to_lufs(input, target_lufs, output)` → gain-adjusted WAV
//!
//! Uses the `ebur128` crate (Rust port of libebur128, the reference BS.1770
//! implementation). True peak uses its built-in 4× oversampled peak meter —
//! no separate resample needed.

use std::path::Path;

use ebur128::{EbuR128, Mode};
use serde::Serialize;

use crate::audio::{load_audio, save_wav, AudioBuffer};
use crate::{Result, UtaiError};

#[derive(Debug, Serialize)]
pub struct LoudnessReport {
    /// Integrated loudness in LUFS (negative, e.g. -16.0).
    pub integrated_lufs: f64,
    /// True peak in dBTP (per channel; we report the worst/peak channel).
    pub true_peak_dbtp: f64,
    pub channels: u16,
    pub sample_rate: u32,
    pub duration_secs: f64,
}

/// Measure integrated LUFS + true peak of an audio file.
#[tauri::command]
pub fn measure_loudness(path: String) -> Result<LoudnessReport> {
    let buf = load_audio(Path::new(&path))?;
    measure_loudness_buffer(&buf)
}

/// Core BS.1770 measurement over an already-decoded buffer (shared by
/// `measure_loudness`, `normalize_to_lufs` and the compliance checker so the
/// ebur128 setup stays in exactly one place).
pub(crate) fn measure_loudness_buffer(buf: &AudioBuffer) -> Result<LoudnessReport> {
    let sr = buf.sample_rate;
    let ch = buf.channels as u32;
    if buf.samples.is_empty() {
        return Err(UtaiError::Audio("Empty audio file".into()));
    }

    // Mode::I = integrated loudness; Mode::TRUE_PEAK = oversampled true peak.
    let mut ebur = EbuR128::new(ch, sr, Mode::I.union(Mode::TRUE_PEAK))
        .map_err(|e| UtaiError::Audio(format!("ebur128 init failed: {e:?}")))?;
    ebur.add_frames_f32(&buf.samples)
        .map_err(|e| UtaiError::Audio(format!("ebur128 add_frames failed: {e:?}")))?;

    let integrated = ebur
        .loudness_global()
        .map_err(|e| UtaiError::Audio(format!("loudness_global failed: {e:?}")))?;

    // True peak is per-channel; report the worst (max).
    // NOTE: ebur128 returns the LINEAR peak — convert to dBTP via 20*log10.
    let mut tp = 0.0_f64;
    for c in 0..ch {
        let p = ebur
            .true_peak(c)
            .map_err(|e| UtaiError::Audio(format!("true_peak failed: {e:?}")))?;
        if p > tp {
            tp = p;
        }
    }
    let true_peak_dbtp = if tp > 0.0 {
        20.0 * tp.log10()
    } else {
        f64::NEG_INFINITY
    };

    Ok(LoudnessReport {
        integrated_lufs: integrated,
        true_peak_dbtp,
        channels: buf.channels,
        sample_rate: sr,
        duration_secs: buf.duration_secs(),
    })
}

/// Normalize an audio file to a target integrated LUFS and write a 32-bit float
/// WAV. The gain is `10^((target - current)/20)`. No limiter is applied — if
/// the gain would push true peak above 0 dBTP the caller should reduce the
/// target or apply limiting separately.
#[tauri::command]
pub fn normalize_to_lufs(input: String, target_lufs: f64, output: String) -> Result<LoudnessReport> {
    let buf = load_audio(Path::new(&input))?;
    if buf.samples.is_empty() {
        return Err(UtaiError::Audio("Empty audio file".into()));
    }
    let sr = buf.sample_rate;
    let ch = buf.channels as u32;

    let mut ebur = EbuR128::new(ch, sr, Mode::I.union(Mode::TRUE_PEAK))
        .map_err(|e| UtaiError::Audio(format!("ebur128 init failed: {e:?}")))?;
    ebur.add_frames_f32(&buf.samples)
        .map_err(|e| UtaiError::Audio(format!("ebur128 add_frames failed: {e:?}")))?;
    let current = ebur
        .loudness_global()
        .map_err(|e| UtaiError::Audio(format!("loudness_global failed: {e:?}")))?;

    // current can be -inf (silence). Guard against that.
    if !current.is_finite() {
        return Err(UtaiError::Audio(
            "Cannot normalize silence (current loudness is -inf LUFS)".into(),
        ));
    }
    let gain = 10.0_f64.powf((target_lufs - current) / 20.0);

    let mut normalized = AudioBuffer {
        samples: buf.samples.iter().map(|&s| (s as f64 * gain) as f32).collect(),
        sample_rate: sr,
        channels: buf.channels,
    };

    // Clamp to [-1, 1] to avoid NaN/Inf propagation (honest clipping).
    for s in normalized.samples.iter_mut() {
        *s = s.clamp(-1.0, 1.0);
    }

    save_wav(Path::new(&output), &normalized)?;

    // Re-measure the normalized file so the returned report reflects the output.
    let report = measure_loudness(output.clone())?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::save_wav;
    use std::env::temp_dir;

    /// A 1 kHz sine at a known level: peak = 0.5 → -6 dBFS. RMS of sine = peak/√2
    /// → -9 dBFS ≈ -9 LUFS (close enough for a mono single-frequency signal).
    fn make_sine_wav(path: &Path, peak: f32) {
        let sr = 44100u32;
        let dur = 2.0;
        let n = (sr as f64 * dur) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| peak * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / sr as f32).sin())
            .collect();
        let buf = AudioBuffer { samples, sample_rate: sr, channels: 1 };
        save_wav(path, &buf).unwrap();
    }

    #[test]
    fn sine_at_half_scale_lufs_around_minus_nine() {
        let path = temp_dir().join("utai_loudness_test_sine.wav");
        make_sine_wav(&path, 0.5);
        let r = measure_loudness(path.to_str().unwrap().into()).unwrap();
        // A 1 kHz sine at 0.5 peak has RMS 0.5/√2 ≈ 0.354 → -9 dBFS.
        // BS.1770 K-weighting has ~unity gain at 1 kHz, so LUFS ≈ -9.
        assert!(
            r.integrated_lufs < -6.0 && r.integrated_lufs > -12.0,
            "LUFS {} should be near -9 for -6 dBFS sine",
            r.integrated_lufs
        );
        // True peak of a sine at 0.5 peak ≈ -6 dBTP.
        assert!(
            r.true_peak_dbtp < -4.0 && r.true_peak_dbtp > -8.0,
            "true peak {} should be near -6 dBTP",
            r.true_peak_dbtp
        );
    }

    #[test]
    fn normalize_reaches_target() {
        let in_path = temp_dir().join("utai_norm_in.wav");
        let out_path = temp_dir().join("utai_norm_out.wav");
        make_sine_wav(&in_path, 0.5); // ~-9 LUFS
        let report = normalize_to_lufs(in_path.to_str().unwrap().into(), -16.0, out_path.to_str().unwrap().into()).unwrap();
        // After normalization to -16 LUFS, the integrated loudness should be ~-16.
        assert!(
            (report.integrated_lufs - (-16.0)).abs() < 1.0,
            "normalized LUFS {} should be near -16",
            report.integrated_lufs
        );
    }

    #[test]
    fn silent_file_errors_on_normalize() {
        let path = temp_dir().join("utai_silent.wav");
        let buf = AudioBuffer { samples: vec![0.0f32; 44100], sample_rate: 44100, channels: 1 };
        save_wav(&path, &buf).unwrap();
        let out = temp_dir().join("utai_silent_out.wav");
        let res = normalize_to_lufs(path.to_str().unwrap().into(), -16.0, out.to_str().unwrap().into());
        assert!(res.is_err(), "normalizing silence should error");
    }
}
