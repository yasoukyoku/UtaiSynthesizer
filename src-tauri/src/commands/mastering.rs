//! Phase 4: mastering completion commands (配方 F ②③④⑤ 节点).
//!
//!   - `apply_bus_eq(input, bands, output)` — link-stereo parametric bus EQ
//!     (HPF / LPF / peaking / low shelf / high shelf, any band count)
//!   - `apply_stereo_width(input, width, output)` — M/S width, capped at 1.5
//!   - `apply_saturate(input, drive, output)` — peak-normalized tanh
//!     saturation (drive 0..1, default 0.05)
//!   - `apply_phase_rotate(input, strength, output)` — cascaded all-pass
//!     rotation as a peak-headroom tool (NOT an anti-detection device)
//!   - `remove_dc(input, output)` — per-channel mean subtraction (配方 F ①,
//!     kernel existed since Phase 1-4; this exposes it as a workflow node)
//!
//! The DSP kernels live in utai-dsp (`bus_eq`, `stereo_width`, `saturate`,
//! `phase_rotate`, `dc_offset`); this file is only the IPC shell, following
//! the compliance.rs pattern: decode → per-plane processing → float WAV.

use std::path::Path;

use serde::{Deserialize, Serialize};
use utai_dsp::bus_eq::{self, BandKind, EqBand};
use utai_dsp::dc_offset::{measure_dc_offset, remove_dc_offset};
use utai_dsp::phase_rotate;
use utai_dsp::saturate;
use utai_dsp::stereo_width;

use crate::audio::{load_audio, save_wav_f32, AudioBuffer};
use crate::{Result, UtaiError};

/// Wire format of one EQ band (matches the TS node param object).
#[derive(Debug, Clone, Deserialize)]
pub struct BandDto {
    #[serde(rename = "type")]
    pub kind: String,
    pub freq: f64,
    #[serde(default)]
    pub gain_db: f64,
    #[serde(default = "default_q")]
    pub q: f64,
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_q() -> f64 {
    1.0
}
fn default_true() -> bool {
    true
}

impl BandDto {
    fn to_band(&self) -> Option<EqBand> {
        BandKind::from_tag(&self.kind).map(|kind| EqBand {
            kind,
            freq: self.freq,
            gain_db: self.gain_db,
            q: self.q,
            enabled: self.enabled,
        })
    }
}

/// Split an interleaved buffer into (left, Option<right>).
fn deinterleave(buf: &AudioBuffer) -> (Vec<f32>, Option<Vec<f32>>) {
    let fc = buf.frame_count();
    if buf.channels == 1 {
        (buf.samples.clone(), None)
    } else {
        let l: Vec<f32> = (0..fc).map(|f| buf.samples[f * 2]).collect();
        let r: Vec<f32> = (0..fc).map(|f| buf.samples[f * 2 + 1]).collect();
        (l, Some(r))
    }
}

/// Re-interleave planes into the buffer's channel layout.
fn reinterleave(left: &[f32], right: Option<&[f32]>, channels: u16) -> Vec<f32> {
    if channels == 1 {
        left.to_vec()
    } else {
        left.iter()
            .zip(right.unwrap_or_default().iter())
            .flat_map(|(a, b)| [*a, *b])
            .collect()
    }
}

fn peak(samples: &[f32]) -> f32 {
    samples.iter().fold(0.0f32, |a, &v| a.max(v.abs()))
}

macro_rules! require_non_empty {
    ($buf:expr) => {
        if $buf.samples.is_empty() {
            return Err(UtaiError::Audio("Empty audio file".into()));
        }
    };
}

// ─── Reports ─────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct BusEqReport {
    /// Bands actually in circuit (disabled/invalid ones dropped).
    pub bands_applied: usize,
    pub peak_in: f32,
    pub peak_out: f32,
    pub channels: u16,
    pub sample_rate: u32,
    pub duration_secs: f64,
}

#[derive(Debug, Serialize)]
pub struct StereoWidthReport {
    /// Width actually applied (after clamping to [0, 1.5]).
    pub width_applied: f64,
    /// Requested width was outside [0, 1.5] and got clamped.
    pub capped: bool,
    /// false for mono input (no side channel to widen).
    pub applied: bool,
    pub peak_in: f32,
    pub peak_out: f32,
    pub channels: u16,
    pub sample_rate: u32,
    pub duration_secs: f64,
}

#[derive(Debug, Serialize)]
pub struct SaturateReport {
    /// Drive actually used (clamped to [0, 1]); 0 = transparent bypass.
    pub drive_applied: f64,
    pub peak_in: f32,
    pub peak_out: f32,
    pub channels: u16,
    pub sample_rate: u32,
    pub duration_secs: f64,
}

#[derive(Debug, Serialize)]
pub struct PhaseRotateReport {
    /// All-pass sections engaged (0..=8).
    pub sections: usize,
    pub peak_in: f32,
    /// True peak can only shift via phase interaction of summed tones —
    /// magnitude response is untouched. Compare with peak_in to see the
    /// headroom effect.
    pub peak_out: f32,
    pub channels: u16,
    pub sample_rate: u32,
    pub duration_secs: f64,
}

#[derive(Debug, Serialize)]
pub struct DcRemoveReport {
    /// |DC| per channel before removal (max across channels).
    pub dc_before: f64,
    /// |DC| after — expected ≈ 0.
    pub dc_after: f64,
    pub peak_in: f32,
    pub peak_out: f32,
    pub channels: u16,
    pub sample_rate: u32,
    pub duration_secs: f64,
}

// ─── Commands ────────────────────────────────────────────────────

/// Link-stereo parametric bus EQ (配方 F ②).
#[tauri::command]
pub fn apply_bus_eq(input: String, bands: Vec<BandDto>, output: String) -> Result<BusEqReport> {
    let buf = load_audio(Path::new(&input))?;
    require_non_empty!(buf);

    let bands: Vec<EqBand> = bands.iter().filter_map(|b| b.to_band()).collect();
    let (mut left, mut right) = deinterleave(&buf);
    let peak_in = peak(&left).max(right.as_deref().map(peak).unwrap_or(0.0));
    let bands_applied =
        bus_eq::apply_eq(&mut left, right.as_deref_mut(), buf.sample_rate as f64, &bands);
    let peak_out = peak(&left).max(right.as_deref().map(peak).unwrap_or(0.0));

    let out = AudioBuffer {
        samples: reinterleave(&left, right.as_deref(), buf.channels),
        sample_rate: buf.sample_rate,
        channels: buf.channels,
    };
    save_wav_f32(Path::new(&output), &out)?;

    Ok(BusEqReport {
        bands_applied,
        peak_in,
        peak_out,
        channels: buf.channels,
        sample_rate: buf.sample_rate,
        duration_secs: buf.duration_secs(),
    })
}

/// M/S stereo width (配方 F ③). Width is clamped to [0, 1.5].
#[tauri::command]
pub fn apply_stereo_width(input: String, width: f64, output: String) -> Result<StereoWidthReport> {
    let buf = load_audio(Path::new(&input))?;
    require_non_empty!(buf);

    let (mut left, mut right) = deinterleave(&buf);
    let peak_in = peak(&left).max(right.as_deref().map(peak).unwrap_or(0.0));
    let applied = right.is_some();
    let capped = stereo_width::apply_stereo_width(&mut left, right.as_deref_mut(), width);
    let width_applied = width.clamp(0.0, stereo_width::MAX_WIDTH);
    let peak_out = peak(&left).max(right.as_deref().map(peak).unwrap_or(0.0));

    let out = AudioBuffer {
        samples: reinterleave(&left, right.as_deref(), buf.channels),
        sample_rate: buf.sample_rate,
        channels: buf.channels,
    };
    save_wav_f32(Path::new(&output), &out)?;

    Ok(StereoWidthReport {
        width_applied,
        capped,
        applied,
        peak_in,
        peak_out,
        channels: buf.channels,
        sample_rate: buf.sample_rate,
        duration_secs: buf.duration_secs(),
    })
}

/// Peak-normalized tanh saturation (配方 F ④). Drive clamped to [0, 1].
#[tauri::command]
pub fn apply_saturate(input: String, drive: f64, output: String) -> Result<SaturateReport> {
    let buf = load_audio(Path::new(&input))?;
    require_non_empty!(buf);

    let (mut left, mut right) = deinterleave(&buf);
    let peak_in = peak(&left).max(right.as_deref().map(peak).unwrap_or(0.0));
    let drive_applied = saturate::apply_saturation(&mut left, drive);
    if let Some(r) = right.as_deref_mut() {
        saturate::apply_saturation(r, drive);
    }
    let peak_out = peak(&left).max(right.as_deref().map(peak).unwrap_or(0.0));

    let out = AudioBuffer {
        samples: reinterleave(&left, right.as_deref(), buf.channels),
        sample_rate: buf.sample_rate,
        channels: buf.channels,
    };
    save_wav_f32(Path::new(&output), &out)?;

    Ok(SaturateReport {
        drive_applied,
        peak_in,
        peak_out,
        channels: buf.channels,
        sample_rate: buf.sample_rate,
        duration_secs: buf.duration_secs(),
    })
}

/// Cascaded all-pass phase rotation (配方 F ④′) — peak-headroom tool only.
#[tauri::command]
pub fn apply_phase_rotate(
    input: String,
    strength: f64,
    output: String,
) -> Result<PhaseRotateReport> {
    let buf = load_audio(Path::new(&input))?;
    require_non_empty!(buf);

    let (mut left, mut right) = deinterleave(&buf);
    let peak_in = peak(&left).max(right.as_deref().map(peak).unwrap_or(0.0));
    let sections = phase_rotate::apply_phase_rotation(
        &mut left,
        right.as_deref_mut(),
        buf.sample_rate as f64,
        strength,
    );
    let peak_out = peak(&left).max(right.as_deref().map(peak).unwrap_or(0.0));

    let out = AudioBuffer {
        samples: reinterleave(&left, right.as_deref(), buf.channels),
        sample_rate: buf.sample_rate,
        channels: buf.channels,
    };
    save_wav_f32(Path::new(&output), &out)?;

    Ok(PhaseRotateReport {
        sections,
        peak_in,
        peak_out,
        channels: buf.channels,
        sample_rate: buf.sample_rate,
        duration_secs: buf.duration_secs(),
    })
}

/// Per-channel DC offset removal (配方 F ①) by mean subtraction.
#[tauri::command]
pub fn remove_dc(input: String, output: String) -> Result<DcRemoveReport> {
    let buf = load_audio(Path::new(&input))?;
    require_non_empty!(buf);

    let (mut left, mut right) = deinterleave(&buf);
    let (dc_l, dc_r) = match right.as_deref() {
        None => measure_dc_offset(&left, None),
        Some(r) => measure_dc_offset(&left, Some(r)),
    };
    let peak_in = peak(&left).max(right.as_deref().map(peak).unwrap_or(0.0));

    remove_dc_offset(&mut left, right.as_deref_mut());
    let (dc_l2, dc_r2) = match right.as_deref() {
        None => measure_dc_offset(&left, None),
        Some(r) => measure_dc_offset(&left, Some(r)),
    };
    let peak_out = peak(&left).max(right.as_deref().map(peak).unwrap_or(0.0));

    let out = AudioBuffer {
        samples: reinterleave(&left, right.as_deref(), buf.channels),
        sample_rate: buf.sample_rate,
        channels: buf.channels,
    };
    save_wav_f32(Path::new(&output), &out)?;

    Ok(DcRemoveReport {
        dc_before: dc_l.abs().max(dc_r.abs()),
        dc_after: dc_l2.abs().max(dc_r2.abs()),
        peak_in,
        peak_out,
        channels: buf.channels,
        sample_rate: buf.sample_rate,
        duration_secs: buf.duration_secs(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    fn write_wav(path: &Path, samples: Vec<f32>, sr: u32, ch: u16) {
        save_wav_f32(
            path,
            &AudioBuffer { samples, sample_rate: sr, channels: ch },
        )
        .unwrap();
    }

    fn stereo_sine(sr: u32, frames: usize, f0: f64) -> Vec<f32> {
        (0..frames)
            .flat_map(|i| {
                let v = (2.0 * std::f64::consts::PI * f0 * i as f64 / sr as f64).sin() as f32 * 0.5;
                [v, v * 0.5]
            })
            .collect()
    }

    #[test]
    fn bus_eq_boosts_and_reports() {
        let dir = temp_dir();
        let input = dir.join("utai_eq_in.wav");
        let out = dir.join("utai_eq_out.wav");
        write_wav(&input, stereo_sine(44100, 4410, 1000.0), 44100, 2);
        let r = apply_bus_eq(
            input.to_str().unwrap().into(),
            vec![
                BandDto { kind: "peaking".into(), freq: 1000.0, gain_db: 12.0, q: 1.0, enabled: true },
                BandDto { kind: "peaking".into(), freq: 100.0, gain_db: 6.0, q: 1.0, enabled: false },
            ],
            out.to_str().unwrap().into(),
        )
        .unwrap();
        assert_eq!(r.bands_applied, 1, "disabled band is dropped");
        assert!(r.peak_out > r.peak_in, "12 dB boost must raise the peak");
        assert_eq!(r.channels, 2);
        let loaded = load_audio(&out).unwrap();
        assert_eq!(loaded.samples.len(), 8820);
    }

    #[test]
    fn bus_eq_rejects_empty_file() {
        let dir = temp_dir();
        let input = dir.join("utai_eq_empty.wav");
        write_wav(&input, vec![], 44100, 1);
        let res = apply_bus_eq(
            input.to_str().unwrap().into(),
            vec![],
            dir.join("utai_eq_empty_out.wav").to_str().unwrap().into(),
        );
        assert!(res.is_err());
    }

    #[test]
    fn stereo_width_caps_and_reports() {
        let dir = temp_dir();
        let input = dir.join("utai_w_in.wav");
        let out = dir.join("utai_w_out.wav");
        // L=0.6, R=0.2 everywhere → M=0.4, S=0.2.
        write_wav(&input, vec![0.6f32, 0.2].repeat(64), 44100, 2);
        let r = apply_stereo_width(
            input.to_str().unwrap().into(),
            2.0,
            out.to_str().unwrap().into(),
        )
        .unwrap();
        assert!(r.capped && r.applied);
        assert!((r.width_applied - 1.5).abs() < 1e-9);
        let loaded = load_audio(&out).unwrap();
        assert!((loaded.samples[0] - 0.7).abs() < 1e-6, "w=1.5 → L=0.7, got {}", loaded.samples[0]);
        assert!((loaded.samples[1] - 0.1).abs() < 1e-6, "w=1.5 → R=0.1, got {}", loaded.samples[1]);
    }

    #[test]
    fn stereo_width_mono_is_noop() {
        let dir = temp_dir();
        let input = dir.join("utai_w_mono.wav");
        let out = dir.join("utai_w_mono_out.wav");
        write_wav(&input, vec![0.4f32; 64], 44100, 1);
        let r = apply_stereo_width(
            input.to_str().unwrap().into(),
            1.4,
            out.to_str().unwrap().into(),
        )
        .unwrap();
        assert!(!r.applied, "mono has no side channel");
        let loaded = load_audio(&out).unwrap();
        assert!((loaded.samples[0] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn saturate_is_transparent_at_zero_and_bounded() {
        let dir = temp_dir();
        let input = dir.join("utai_sat_in.wav");
        let out0 = dir.join("utai_sat_out0.wav");
        let out1 = dir.join("utai_sat_out1.wav");
        write_wav(&input, stereo_sine(44100, 4410, 220.0), 44100, 2);

        let r0 = apply_saturate(
            input.to_str().unwrap().into(),
            0.0,
            out0.to_str().unwrap().into(),
        )
        .unwrap();
        assert_eq!(r0.drive_applied, 0.0);
        assert!((r0.peak_out - r0.peak_in).abs() < 1e-6, "drive 0 = bypass");

        let r1 = apply_saturate(
            input.to_str().unwrap().into(),
            1.0,
            out1.to_str().unwrap().into(),
        )
        .unwrap();
        assert_eq!(r1.drive_applied, 1.0);
        assert!(r1.peak_out <= 1.0 + 1e-4, "tanh output stays bounded");
        assert!(r1.peak_out > r1.peak_in, "normalized tanh lifts 0.5-peak sine");
    }

    #[test]
    fn phase_rotate_reports_sections_and_headroom() {
        let dir = temp_dir();
        let input = dir.join("utai_pr_in.wav");
        let out = dir.join("utai_pr_out.wav");
        // Broadband multi-tone: phase rotation shifts the summed peak.
        let n = 44100;
        let s: Vec<f32> = (0..n)
            .flat_map(|i| {
                let t = i as f64 / 44100.0;
                let v = 0.4 * (2.0 * std::f64::consts::PI * 220.0 * t).sin()
                    + 0.3 * (2.0 * std::f64::consts::PI * 1310.0 * t).sin()
                    + 0.2 * (2.0 * std::f64::consts::PI * 5230.0 * t).sin();
                [v as f32, v as f32]
            })
            .collect();
        write_wav(&input, s, 44100, 2);
        let expected_len = 2 * n;
        let r = apply_phase_rotate(
            input.to_str().unwrap().into(),
            1.0,
            out.to_str().unwrap().into(),
        )
        .unwrap();
        assert_eq!(r.sections, 8);
        assert!(r.peak_out > 0.0 && r.peak_out < 1.0);
        let loaded = load_audio(&out).unwrap();
        assert_eq!(loaded.samples.len(), expected_len, "length preserved");
    }

    #[test]
    fn remove_dc_measures_before_and_after() {
        let dir = temp_dir();
        let input = dir.join("utai_dc_in.wav");
        let out = dir.join("utai_dc_out.wav");
        // 0.05 DC + sine, stereo identical.
        let s: Vec<f32> = (0..4410)
            .flat_map(|i| {
                let v = 0.05 + 0.3 * (i as f32 * 0.02).sin();
                [v, v]
            })
            .collect();
        write_wav(&input, s, 44100, 2);
        let r = remove_dc(input.to_str().unwrap().into(), out.to_str().unwrap().into()).unwrap();
        assert!(r.dc_before > 0.04, "dc_before ≈ 0.05, got {}", r.dc_before);
        assert!(r.dc_after < 1e-4, "dc_after ≈ 0, got {}", r.dc_after);
        assert!(r.peak_out < r.peak_in, "removing positive DC lowers the peak");
        let loaded = load_audio(&out).unwrap();
        let mid: f32 = (loaded.samples[100] + loaded.samples[101]) * 0.5;
        assert!(mid.abs() <= 0.35 + 1e-3, "sample is now the bare sine");
    }
}
