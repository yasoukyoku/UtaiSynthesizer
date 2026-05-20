use crate::inference::engine::OnnxEngine;
use crate::inference::nsf_hifigan::{self, VocoderMode};
use crate::{Result, UtaiError};

use super::AudioBuffer;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum VocoderChoice {
    World,
    NsfHifigan,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Effect {
    PitchShift {
        semitones: f32,
        vocoder: VocoderChoice,
    },
    FormantShift {
        ratio: f32,
        vocoder: VocoderChoice,
    },
    AudioEnhance,
    Volume {
        gain_db: f32,
    },
    FadeIn {
        duration_ms: u32,
    },
    FadeOut {
        duration_ms: u32,
    },
    Normalize {
        target_db: f32,
    },
}

pub fn apply_effect(
    buffer: &AudioBuffer,
    effect: &Effect,
    engine: &OnnxEngine,
    nsf_session_id: Option<&str>,
) -> Result<AudioBuffer> {
    match effect {
        Effect::PitchShift { semitones, vocoder } => match vocoder {
            VocoderChoice::World => pitch_shift_world(buffer, *semitones),
            VocoderChoice::NsfHifigan => {
                let sid = nsf_session_id.ok_or_else(|| {
                    UtaiError::Audio("NSF-HiFiGAN model not loaded".to_string())
                })?;
                pitch_shift_nsf(buffer, *semitones, engine, sid)
            }
        },
        Effect::FormantShift { ratio, vocoder } => match vocoder {
            VocoderChoice::World => formant_shift_world(buffer, *ratio),
            VocoderChoice::NsfHifigan => {
                let sid = nsf_session_id.ok_or_else(|| {
                    UtaiError::Audio("NSF-HiFiGAN model not loaded".to_string())
                })?;
                formant_shift_nsf(buffer, *ratio, engine, sid)
            }
        },
        Effect::AudioEnhance => {
            let sid = nsf_session_id.ok_or_else(|| {
                UtaiError::Audio("NSF-HiFiGAN model not loaded".to_string())
            })?;
            audio_enhance(buffer, engine, sid)
        }
        Effect::Volume { gain_db } => Ok(apply_gain(buffer, *gain_db)),
        Effect::FadeIn { duration_ms } => Ok(apply_fade_in(buffer, *duration_ms)),
        Effect::FadeOut { duration_ms } => Ok(apply_fade_out(buffer, *duration_ms)),
        Effect::Normalize { target_db } => Ok(normalize(buffer, *target_db)),
    }
}

fn pitch_shift_world(buffer: &AudioBuffer, semitones: f32) -> Result<AudioBuffer> {
    // WORLD-based pitch shifting (DSP approach)
    // Uses analysis-modification-synthesis:
    // 1. Extract F0, spectral envelope, aperiodicity (DIO + CheapTrick + D4C)
    // 2. Modify F0 by semitone ratio
    // 3. Resynthesize with WORLD synthesizer

    // TODO: Implement WORLD C bindings via world-sys crate
    // For now, simple time-domain pitch shift as placeholder
    let ratio = 2.0f32.powf(semitones / 12.0);
    let new_len = (buffer.samples.len() as f32 / ratio) as usize;
    let mut output = Vec::with_capacity(new_len);

    for i in 0..new_len {
        let src_pos = i as f32 * ratio;
        let idx = src_pos as usize;
        let frac = src_pos - idx as f32;

        if idx + 1 < buffer.samples.len() {
            let sample = buffer.samples[idx] * (1.0 - frac) + buffer.samples[idx + 1] * frac;
            output.push(sample);
        }
    }

    Ok(AudioBuffer::new_mono(output, buffer.sample_rate))
}

fn pitch_shift_nsf(
    buffer: &AudioBuffer,
    semitones: f32,
    engine: &OnnxEngine,
    session_id: &str,
) -> Result<AudioBuffer> {
    let mel = nsf_hifigan::audio_to_mel(&buffer.samples, buffer.sample_rate)?;

    // Extract F0 for the buffer (simplified — in production use RMVPE)
    let n_frames = mel.shape()[0];
    let f0 = estimate_f0_simple(&buffer.samples, buffer.sample_rate, n_frames);

    let result = nsf_hifigan::synthesize(
        engine,
        session_id,
        &mel,
        &f0,
        &VocoderMode::PitchShift { semitones },
    )?;

    Ok(AudioBuffer::new_mono(result.audio, result.sample_rate))
}

fn formant_shift_world(buffer: &AudioBuffer, _ratio: f32) -> Result<AudioBuffer> {
    // WORLD-based formant shifting
    // Modifies spectral envelope while keeping F0 unchanged
    // TODO: Implement with WORLD C bindings
    Ok(buffer.clone())
}

fn formant_shift_nsf(
    buffer: &AudioBuffer,
    ratio: f32,
    engine: &OnnxEngine,
    session_id: &str,
) -> Result<AudioBuffer> {
    let mel = nsf_hifigan::audio_to_mel(&buffer.samples, buffer.sample_rate)?;
    let n_frames = mel.shape()[0];
    let f0 = estimate_f0_simple(&buffer.samples, buffer.sample_rate, n_frames);

    let result = nsf_hifigan::synthesize(
        engine,
        session_id,
        &mel,
        &f0,
        &VocoderMode::FormantShift { ratio },
    )?;

    Ok(AudioBuffer::new_mono(result.audio, result.sample_rate))
}

fn audio_enhance(
    buffer: &AudioBuffer,
    engine: &OnnxEngine,
    session_id: &str,
) -> Result<AudioBuffer> {
    let mel = nsf_hifigan::audio_to_mel(&buffer.samples, buffer.sample_rate)?;
    let n_frames = mel.shape()[0];
    let f0 = estimate_f0_simple(&buffer.samples, buffer.sample_rate, n_frames);

    let result = nsf_hifigan::synthesize(
        engine,
        session_id,
        &mel,
        &f0,
        &VocoderMode::AudioEnhance,
    )?;

    Ok(AudioBuffer::new_mono(result.audio, result.sample_rate))
}

fn apply_gain(buffer: &AudioBuffer, gain_db: f32) -> AudioBuffer {
    let multiplier = 10.0f32.powf(gain_db / 20.0);
    let samples: Vec<f32> = buffer.samples.iter().map(|&s| s * multiplier).collect();
    AudioBuffer {
        samples,
        sample_rate: buffer.sample_rate,
        channels: buffer.channels,
    }
}

fn apply_fade_in(buffer: &AudioBuffer, duration_ms: u32) -> AudioBuffer {
    let fade_samples = (buffer.sample_rate as f64 * duration_ms as f64 / 1000.0) as usize;
    let mut samples = buffer.samples.clone();

    for i in 0..fade_samples.min(samples.len()) {
        let gain = i as f32 / fade_samples as f32;
        samples[i] *= gain;
    }

    AudioBuffer {
        samples,
        sample_rate: buffer.sample_rate,
        channels: buffer.channels,
    }
}

fn apply_fade_out(buffer: &AudioBuffer, duration_ms: u32) -> AudioBuffer {
    let fade_samples = (buffer.sample_rate as f64 * duration_ms as f64 / 1000.0) as usize;
    let mut samples = buffer.samples.clone();
    let total = samples.len();

    for i in 0..fade_samples.min(total) {
        let idx = total - 1 - i;
        let gain = i as f32 / fade_samples as f32;
        samples[idx] *= 1.0 - gain;
    }

    AudioBuffer {
        samples,
        sample_rate: buffer.sample_rate,
        channels: buffer.channels,
    }
}

pub fn normalize(buffer: &AudioBuffer, target_db: f32) -> AudioBuffer {
    let peak = buffer
        .samples
        .iter()
        .map(|s| s.abs())
        .fold(0.0f32, f32::max);

    if peak < 1e-10 {
        return buffer.clone();
    }

    let target_linear = 10.0f32.powf(target_db / 20.0);
    let gain = target_linear / peak;
    let samples: Vec<f32> = buffer.samples.iter().map(|&s| s * gain).collect();

    AudioBuffer {
        samples,
        sample_rate: buffer.sample_rate,
        channels: buffer.channels,
    }
}

fn estimate_f0_simple(audio: &[f32], sample_rate: u32, n_frames: usize) -> Vec<f32> {
    // Simplified F0 estimation via zero-crossing rate
    // In production, use RMVPE through the inference engine
    let hop = audio.len() / n_frames.max(1);
    let mut f0 = Vec::with_capacity(n_frames);

    for frame in 0..n_frames {
        let start = frame * hop;
        let end = (start + hop).min(audio.len());
        let frame_audio = &audio[start..end];

        let mut crossings = 0u32;
        for i in 1..frame_audio.len() {
            if (frame_audio[i] >= 0.0) != (frame_audio[i - 1] >= 0.0) {
                crossings += 1;
            }
        }

        let freq = crossings as f32 * sample_rate as f32 / (2.0 * (end - start) as f32);
        f0.push(if freq > 50.0 && freq < 1200.0 { freq } else { 0.0 });
    }

    f0
}
