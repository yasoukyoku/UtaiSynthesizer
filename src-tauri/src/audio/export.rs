use std::path::Path;

use super::AudioBuffer;
use crate::{Result, UtaiError};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExportConfig {
    pub format: ExportFormat,
    pub sample_rate: u32,
    pub normalize: bool,
    pub normalize_target_db: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ExportFormat {
    Wav16,
    Wav32Float,
    Flac,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            format: ExportFormat::Wav32Float,
            sample_rate: 44100,
            normalize: true,
            normalize_target_db: -1.0,
        }
    }
}

pub fn mixdown(tracks: &[TrackAudio]) -> Result<AudioBuffer> {
    if tracks.is_empty() {
        return Ok(AudioBuffer::new_mono(vec![], 44100));
    }

    let sample_rate = tracks[0].buffer.sample_rate;
    let max_len = tracks
        .iter()
        .map(|t| {
            let offset_samples = (t.offset_secs * sample_rate as f64) as usize;
            offset_samples + t.buffer.samples.len()
        })
        .max()
        .unwrap_or(0);

    let mut mix = vec![0.0f32; max_len];

    for track in tracks {
        if track.muted {
            continue;
        }

        let offset = (track.offset_secs * sample_rate as f64) as usize;
        let vol = 10.0f32.powf(track.volume_db / 20.0);
        let gain_l = vol * track.pan_gain_l();
        let gain_r = vol * track.pan_gain_r();
        let channels = track.buffer.channels as usize;

        let frames = track.buffer.samples.len() / channels;
        for frame in 0..frames {
            let idx = offset + frame;
            if idx >= mix.len() {
                break;
            }
            if channels == 1 {
                let s = track.buffer.samples[frame];
                mix[idx] += s * (gain_l + gain_r) * 0.5;
            } else {
                let l = track.buffer.samples[frame * channels];
                let r = track.buffer.samples[frame * channels + 1];
                mix[idx] += l * gain_l + r * gain_r;
            }
        }
    }

    Ok(AudioBuffer::new_mono(mix, sample_rate))
}

pub fn export(buffer: &AudioBuffer, path: &Path, config: &ExportConfig) -> Result<()> {
    let mut output = buffer.clone();

    if output.sample_rate != config.sample_rate {
        output = super::resample::resample(&output, config.sample_rate)?;
    }

    if config.normalize {
        output = super::effects::normalize(&output, config.normalize_target_db);
    }

    match config.format {
        ExportFormat::Wav16 => export_wav_16(path, &output),
        ExportFormat::Wav32Float => super::save_wav(path, &output),
        ExportFormat::Flac => {
            // FLAC export would require a FLAC encoder crate
            // For now, fall back to WAV
            super::save_wav(path, &output)
        }
    }
}

fn export_wav_16(path: &Path, buffer: &AudioBuffer) -> Result<()> {
    let spec = hound::WavSpec {
        channels: buffer.channels,
        sample_rate: buffer.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| UtaiError::Audio(format!("Create WAV: {}", e)))?;

    for &sample in &buffer.samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let int_sample = (clamped * 32767.0) as i16;
        writer
            .write_sample(int_sample)
            .map_err(|e| UtaiError::Audio(format!("Write: {}", e)))?;
    }

    writer
        .finalize()
        .map_err(|e| UtaiError::Audio(format!("Finalize: {}", e)))?;

    Ok(())
}

#[derive(Debug, Clone)]
pub struct TrackAudio {
    pub buffer: AudioBuffer,
    pub offset_secs: f64,
    pub volume_db: f32,
    pub pan: f32,
    pub muted: bool,
}

impl TrackAudio {
    fn pan_gain_l(&self) -> f32 {
        // Constant-power pan law: pan in [-1, 1], center=0
        ((self.pan + 1.0) * 0.5 * std::f32::consts::FRAC_PI_2).cos()
    }

    fn pan_gain_r(&self) -> f32 {
        ((self.pan + 1.0) * 0.5 * std::f32::consts::FRAC_PI_2).sin()
    }
}
