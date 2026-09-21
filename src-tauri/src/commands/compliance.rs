//! Phase 1: mastering compliance commands.
//!
//! Three commands:
//!   - `mix_audio_files(paths, output)` — sum N inputs into one 32-bit float
//!     WAV (mono upmixed, short inputs padded with silence, headroom preserved)
//!   - `compliance_check(path)` — one-shot mastering health report: LUFS,
//!     true peak, clipping runs, stereo phase correlation, head/tail silence,
//!     DC offset
//!   - `apply_dither(input, dither_type, output)` — dithered 16-bit WAV for
//!     the final bit-depth reduction
//!
//! The DSP kernels live in utai-dsp (`dither`, `clip_detect`, `silence`,
//! `dc_offset`, `phase_correlation`); this file is only the IPC shell.

use std::path::Path;

use serde::Serialize;
use utai_dsp::clip_detect::detect_clipping;
use utai_dsp::dc_offset::measure_dc_offset;
use utai_dsp::dither::{apply_dither_16bit, DitherType};
use utai_dsp::phase_correlation::measure_phase_correlation;
use utai_dsp::silence::{measure_silence, DEFAULT_SILENCE_DB};

use crate::audio::{load_audio, save_wav_f32, save_wav_i16, AudioBuffer};
use crate::commands::loudness::measure_loudness_buffer;
use crate::{Result, UtaiError};

#[derive(Debug, Serialize)]
pub struct MixReport {
    /// Number of inputs actually summed (≥ 2).
    pub inputs: usize,
    pub channels: u16,
    pub sample_rate: u32,
    pub duration_secs: f64,
    /// Peak absolute sample of the sum (linear). Can exceed 1.0 — the mix is
    /// written as 32-bit float WAV, so headroom is preserved for downstream
    /// nodes (lufsNormalize / dither decide the final ceiling).
    pub peak: f32,
    /// How many mono inputs were upmixed to the output channel count.
    pub upmixed: usize,
}

#[derive(Debug, Serialize)]
pub struct ComplianceReport {
    pub integrated_lufs: f64,
    pub true_peak_dbtp: f64,
    /// Clipping run count (≥ 3 consecutive samples at |s| ≥ 0.999),
    /// worst channel.
    pub clipped_runs: usize,
    /// Stereo mono-compatibility (Pearson L/R, [-1, 1]). `None` for mono.
    pub phase_correlation: Option<f64>,
    pub head_silence_secs: f64,
    pub tail_silence_secs: f64,
    /// Max |DC offset| across channels, fraction of full scale.
    pub dc_offset: f64,
    pub channels: u16,
    pub sample_rate: u32,
    pub duration_secs: f64,
}

#[derive(Debug, Serialize)]
pub struct DitherReport {
    /// Which dither was applied ("None" | "Tpdf" | "TpdfShaped").
    pub dither_type: String,
    pub channels: u16,
    pub sample_rate: u32,
    pub duration_secs: f64,
    /// Peak of the quantized output in dBFS (0 = full scale).
    pub peak_dbfs: f64,
}

/// Sum N audio files into one float WAV. Inputs must share a sample rate;
/// mono inputs are upmixed by duplication, shorter inputs are padded with
/// silence. The sum is written unclamped (f32 WAV) — the report carries the
/// peak so callers can decide on limiting.
#[tauri::command]
pub fn mix_audio_files(paths: Vec<String>, output: String) -> Result<MixReport> {
    if paths.len() < 2 {
        return Err(UtaiError::Audio(
            "mix_audio_files needs at least 2 inputs (a single input needs no mix)".into(),
        ));
    }
    let bufs: Vec<AudioBuffer> = paths
        .iter()
        .map(|p| load_audio(Path::new(p)))
        .collect::<Result<Vec<_>>>()?;

    let sr = bufs[0].sample_rate;
    if bufs.iter().any(|b| b.sample_rate != sr) {
        return Err(UtaiError::Audio(
            "mix inputs must share one sample rate — resample first".into(),
        ));
    }
    let channels = bufs.iter().map(|b| b.channels).max().unwrap();
    if channels > 2 {
        return Err(UtaiError::Audio(
            "only mono/stereo mixing is supported".into(),
        ));
    }

    let max_frames = bufs.iter().map(|b| b.frame_count()).max().unwrap();
    if max_frames == 0 {
        return Err(UtaiError::Audio("all mix inputs are empty".into()));
    }

    let ch = channels as usize;
    let mut mix = vec![0.0f32; max_frames * ch];
    let mut upmixed = 0usize;
    for b in &bufs {
        let bch = b.channels as usize;
        if bch == 1 && ch == 2 {
            upmixed += 1;
        }
        let fc = b.frame_count();
        for f in 0..fc {
            for c in 0..ch {
                let s = if bch == 1 {
                    b.samples[f]
                } else {
                    b.samples[f * 2 + c]
                };
                mix[f * ch + c] += s;
            }
        }
    }

    let peak = mix.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let out = AudioBuffer {
        samples: mix,
        sample_rate: sr,
        channels,
    };
    save_wav_f32(Path::new(&output), &out)?;

    Ok(MixReport {
        inputs: bufs.len(),
        channels,
        sample_rate: sr,
        duration_secs: out.duration_secs(),
        peak,
        upmixed,
    })
}

/// One-shot mastering health report over a decoded file.
#[tauri::command]
pub fn compliance_check(path: String) -> Result<ComplianceReport> {
    let buf = load_audio(Path::new(&path))?;
    let loud = measure_loudness_buffer(&buf)?;

    let chans = deinterleave(&buf);

    // Clipping: count runs per channel, report the worst.
    let clipped_runs = chans
        .iter()
        .map(|c| detect_clipping(c).len())
        .max()
        .unwrap_or(0);

    // Phase correlation only makes sense with two channels.
    let phase_correlation = if chans.len() == 2 {
        Some(measure_phase_correlation(&chans[0], &chans[1]))
    } else {
        None
    };

    // Silence measurement wants mono (utai-dsp convention: down-mix stereo).
    let mono = downmix_mono(&buf);
    let (head_silence_secs, tail_silence_secs) =
        measure_silence(&mono, buf.sample_rate, DEFAULT_SILENCE_DB);

    // DC offset per channel; report the worst magnitude.
    let (dc_l, dc_r) = match chans.len() {
        1 => measure_dc_offset(&chans[0], None),
        _ => measure_dc_offset(&chans[0], Some(&chans[1])),
    };
    let dc_offset = dc_l.abs().max(dc_r.abs());

    Ok(ComplianceReport {
        integrated_lufs: loud.integrated_lufs,
        true_peak_dbtp: loud.true_peak_dbtp,
        clipped_runs,
        phase_correlation,
        head_silence_secs,
        tail_silence_secs,
        dc_offset,
        channels: buf.channels,
        sample_rate: buf.sample_rate,
        duration_secs: buf.duration_secs(),
    })
}

/// Apply dither and write a 16-bit integer WAV. `dither_type`: 0 = None
/// (plain truncation), 1 = TPDF, 2 = TPDF + noise shaping. The seed is fixed
/// so the same input always produces a byte-identical output file.
#[tauri::command]
pub fn apply_dither(input: String, dither_type: u8, output: String) -> Result<DitherReport> {
    let dt = match dither_type {
        0 => DitherType::None,
        1 => DitherType::Tpdf,
        2 => DitherType::TpdfShaped,
        other => {
            return Err(UtaiError::Audio(format!(
                "Unknown dither type {other} (0=None, 1=Tpdf, 2=TpdfShaped)"
            )))
        }
    };
    let buf = load_audio(Path::new(&input))?;
    if buf.samples.is_empty() {
        return Err(UtaiError::Audio("Empty audio file".into()));
    }

    let quantized = apply_dither_16bit(&buf.samples, dt, 0);
    save_wav_i16(Path::new(&output), &quantized, buf.sample_rate, buf.channels)?;

    let peak = quantized
        .iter()
        .fold(0i32, |a, &v| a.max((v as i32).abs())) as f64;
    let peak_dbfs = if peak > 0.0 {
        20.0 * (peak / 32767.0).log10()
    } else {
        f64::NEG_INFINITY
    };

    Ok(DitherReport {
        dither_type: format!("{dt:?}"),
        channels: buf.channels,
        sample_rate: buf.sample_rate,
        duration_secs: buf.duration_secs(),
        peak_dbfs,
    })
}

/// Split an interleaved buffer into per-channel planes.
fn deinterleave(buf: &AudioBuffer) -> Vec<Vec<f32>> {
    let ch = buf.channels as usize;
    let fc = buf.frame_count();
    (0..ch)
        .map(|c| (0..fc).map(|f| buf.samples[f * ch + c]).collect())
        .collect()
}

/// Average all channels per frame — the mono view the silence check wants.
fn downmix_mono(buf: &AudioBuffer) -> Vec<f32> {
    let ch = buf.channels as usize;
    if ch == 1 {
        return buf.samples.clone();
    }
    buf.samples
        .chunks_exact(ch)
        .map(|frame| frame.iter().sum::<f32>() / ch as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    /// Fixtures are written as 32-bit float WAV: the mix math under test is exact
    /// float arithmetic, and 16-bit fixtures would inject ±1 LSB quantization
    /// noise that breaks the 1e-6 assertions.
    fn write_wav(path: &Path, samples: Vec<f32>, sr: u32, ch: u16) {
        save_wav_f32(
            path,
            &AudioBuffer {
                samples,
                sample_rate: sr,
                channels: ch,
            },
        )
        .unwrap();
    }

    #[test]
    fn mix_sums_two_monos_and_pads() {
        let dir = temp_dir();
        let a = dir.join("utai_mix_a.wav");
        let b = dir.join("utai_mix_b.wav");
        let out = dir.join("utai_mix_out.wav");
        write_wav(&a, vec![0.25f32; 4410], 44100, 1); // 0.1 s
        write_wav(&b, vec![0.5f32; 2205], 44100, 1); // 0.05 s
        let r = mix_audio_files(
            vec![a.to_str().unwrap().into(), b.to_str().unwrap().into()],
            out.to_str().unwrap().into(),
        )
        .unwrap();
        assert_eq!(r.inputs, 2);
        assert_eq!(r.channels, 1);
        assert_eq!(r.upmixed, 0);
        assert!(
            (r.duration_secs - 0.1).abs() < 1e-3,
            "longer input wins: {}",
            r.duration_secs
        );
        let mixed = load_audio(&out).unwrap();
        // During overlap: 0.25 + 0.5 = 0.75.
        assert!((mixed.samples[0] - 0.75).abs() < 1e-6);
        // After b ends, only a remains (padding works).
        assert!((mixed.samples[3000] - 0.25).abs() < 1e-6);
        assert!((r.peak - 0.75).abs() < 1e-6);
    }

    #[test]
    fn mix_upmixes_mono_into_stereo() {
        let dir = temp_dir();
        let a = dir.join("utai_mix_up_a.wav");
        let b = dir.join("utai_mix_up_b.wav");
        let out = dir.join("utai_mix_up_out.wav");
        write_wav(&a, vec![0.3f32; 100], 44100, 1);
        // Stereo L=0.1, R=-0.1 (interleaved).
        let stereo: Vec<f32> = (0..50).flat_map(|_| [0.1f32, -0.1f32]).collect();
        write_wav(&b, stereo, 44100, 2);
        let r = mix_audio_files(
            vec![a.to_str().unwrap().into(), b.to_str().unwrap().into()],
            out.to_str().unwrap().into(),
        )
        .unwrap();
        assert_eq!(r.channels, 2);
        assert_eq!(r.upmixed, 1);
        let mixed = load_audio(&out).unwrap();
        assert_eq!(mixed.samples.len(), 200); // 100 frames × 2ch
        assert!((mixed.samples[0] - 0.4).abs() < 1e-6); // 0.3+0.1
        assert!((mixed.samples[1] - 0.2).abs() < 1e-6); // 0.3+(-0.1)
    }

    #[test]
    fn mix_rejects_sample_rate_mismatch() {
        let dir = temp_dir();
        let a = dir.join("utai_mix_sr_a.wav");
        let b = dir.join("utai_mix_sr_b.wav");
        let out = dir.join("utai_mix_sr_out.wav");
        write_wav(&a, vec![0.1f32; 100], 44100, 1);
        write_wav(&b, vec![0.1f32; 100], 48000, 1);
        let res = mix_audio_files(
            vec![a.to_str().unwrap().into(), b.to_str().unwrap().into()],
            out.to_str().unwrap().into(),
        );
        assert!(res.is_err(), "44.1k + 48k must error, not silently resample");
    }

    #[test]
    fn mix_rejects_single_input() {
        let dir = temp_dir();
        let a = dir.join("utai_mix_one.wav");
        write_wav(&a, vec![0.1f32; 100], 44100, 1);
        let res = mix_audio_files(
            vec![a.to_str().unwrap().into()],
            dir.join("utai_mix_one_out.wav").to_str().unwrap().into(),
        );
        assert!(res.is_err());
    }

    #[test]
    fn compliance_flags_all_issues() {
        let dir = temp_dir();
        let path = dir.join("utai_compliance.wav");
        let sr = 44100u32;
        let head = sr as usize / 10; // 0.1 s of leading silence
        let n = head + sr as usize;
        // Identical channels (correlation = 1) with a DC offset, plus a
        // 5-sample clipped run right after the silence.
        let mut s = vec![0.0f32; n];
        for i in head..n {
            s[i] = 0.01 + 0.5 * ((i - head) as f32 * 0.01).sin();
        }
        for i in head..head + 5 {
            s[i] = 1.0;
        }
        // Stereo = L duplicated.
        let mut stereo = Vec::with_capacity(n * 2);
        for &v in &s {
            stereo.push(v);
            stereo.push(v);
        }
        write_wav(&path, stereo, sr, 2);

        let rep = compliance_check(path.to_str().unwrap().into()).unwrap();
        assert_eq!(rep.clipped_runs, 1, "one 5-sample run at 1.0");
        assert!(
            rep.dc_offset > 0.005 && rep.dc_offset < 0.012,
            "DC should be ≈0.01 over the non-silent part, got {}",
            rep.dc_offset
        );
        assert!(
            rep.head_silence_secs > 0.08 && rep.head_silence_secs < 0.12,
            "head silence ≈0.1 s, got {}",
            rep.head_silence_secs
        );
        let corr = rep.phase_correlation.expect("stereo file has correlation");
        assert!((corr - 1.0).abs() < 1e-6, "identical L/R → 1.0, got {corr}");
    }

    #[test]
    fn apply_dither_writes_16bit_and_reports_peak() {
        let dir = temp_dir();
        let input = dir.join("utai_dither_in.wav");
        let out = dir.join("utai_dither_out.wav");
        write_wav(&input, vec![0.5f32; 4410], 44100, 1);
        let r = apply_dither(
            input.to_str().unwrap().into(),
            1,
            out.to_str().unwrap().into(),
        )
        .unwrap();
        assert_eq!(r.dither_type, "Tpdf");
        assert!(
            (r.peak_dbfs - (-6.02)).abs() < 0.1,
            "peak of 0.5 ≈ -6.02 dBFS, got {}",
            r.peak_dbfs
        );
        let reader = hound::WavReader::open(&out).unwrap();
        assert_eq!(reader.spec().bits_per_sample, 16);
        assert_eq!(reader.spec().sample_format, hound::SampleFormat::Int);
        let first: i16 = reader.into_samples::<i16>().next().unwrap().unwrap();
        assert!(
            (16380..=16388).contains(&first),
            "0.5*32767 ≈ 16384 ± 1 LSB of dither, got {first}"
        );
    }

    #[test]
    fn apply_dither_rejects_unknown_type() {
        let dir = temp_dir();
        let input = dir.join("utai_dither_bad_in.wav");
        write_wav(&input, vec![0.5f32; 100], 44100, 1);
        let res = apply_dither(
            input.to_str().unwrap().into(),
            9,
            dir.join("utai_dither_bad_out.wav").to_str().unwrap().into(),
        );
        assert!(res.is_err(), "type 9 must be rejected");
    }
}
