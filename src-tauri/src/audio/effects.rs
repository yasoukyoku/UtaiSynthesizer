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
    // WORLD C bindings not yet available — use STFT-based pitch shift as interim.
    // Shifts frequency bins in the spectrogram, preserves duration.
    let channels = buffer.channels as usize;
    if channels == 0 {
        return Ok(buffer.clone());
    }
    let frame_count = buffer.samples.len() / channels;
    let ratio = 2.0f32.powf(semitones / 12.0);

    let mut all_samples = Vec::with_capacity(buffer.samples.len());

    for ch in 0..channels {
        let mono: Vec<f32> = (0..frame_count)
            .map(|i| buffer.samples[i * channels + ch])
            .collect();

        let n_fft = 2048;
        let hop = 512;
        let win: Vec<f32> = (0..n_fft).map(|i| {
            0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n_fft as f32).cos())
        }).collect();

        let n_frames = if mono.len() >= n_fft { (mono.len() - n_fft) / hop + 1 } else { 0 };
        let freq_bins = n_fft / 2 + 1;

        // Forward STFT
        use std::f32::consts::PI;
        let mut mag = vec![vec![0.0f32; n_frames]; freq_bins];
        let mut phase = vec![vec![0.0f32; n_frames]; freq_bins];

        for t in 0..n_frames {
            let offset = t * hop;
            for k in 0..freq_bins {
                let mut re = 0.0f32;
                let mut im = 0.0f32;
                for n in 0..n_fft {
                    if offset + n < mono.len() {
                        let angle = -2.0 * PI * k as f32 * n as f32 / n_fft as f32;
                        re += mono[offset + n] * win[n] * angle.cos();
                        im += mono[offset + n] * win[n] * angle.sin();
                    }
                }
                mag[k][t] = (re * re + im * im).sqrt();
                phase[k][t] = im.atan2(re);
            }
        }

        // Shift frequency bins
        let mut shifted_mag = vec![vec![0.0f32; n_frames]; freq_bins];
        for k in 0..freq_bins {
            let src = k as f32 / ratio;
            let lo = src as usize;
            let frac = src - lo as f32;
            if lo + 1 < freq_bins {
                for t in 0..n_frames {
                    shifted_mag[k][t] = mag[lo][t] * (1.0 - frac) + mag[lo + 1][t] * frac;
                }
            } else if lo < freq_bins {
                for t in 0..n_frames {
                    shifted_mag[k][t] = mag[lo][t] * (1.0 - frac);
                }
            }
        }

        // Inverse STFT (Griffin-Lim-like: use original phase)
        let out_len = (n_frames - 1) * hop + n_fft;
        let mut output = vec![0.0f32; out_len];
        let mut window_sum = vec![0.0f32; out_len];

        for t in 0..n_frames {
            let offset = t * hop;
            for n in 0..n_fft {
                if offset + n >= out_len { break; }
                let mut sample = 0.0f32;
                for k in 0..freq_bins {
                    let angle = 2.0 * PI * k as f32 * n as f32 / n_fft as f32;
                    sample += shifted_mag[k][t] * (phase[k][t] + angle).cos();
                }
                sample *= 2.0 / n_fft as f32;
                output[offset + n] += sample * win[n];
                window_sum[offset + n] += win[n] * win[n];
            }
        }

        for i in 0..out_len.min(frame_count) {
            if window_sum[i] > 1e-8 {
                output[i] /= window_sum[i];
            }
        }

        output.truncate(frame_count);
        all_samples.extend_from_slice(&output);
    }

    // Re-interleave channels
    let mut interleaved = vec![0.0f32; buffer.samples.len()];
    for ch in 0..channels {
        for i in 0..frame_count {
            interleaved[i * channels + ch] = all_samples[ch * frame_count + i];
        }
    }

    Ok(AudioBuffer {
        samples: interleaved,
        sample_rate: buffer.sample_rate,
        channels: buffer.channels,
    })
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

fn formant_shift_world(buffer: &AudioBuffer, ratio: f32) -> Result<AudioBuffer> {
    if (ratio - 1.0).abs() < 0.001 {
        return Ok(buffer.clone());
    }
    Err(UtaiError::Audio(
        "Formant shift with WORLD vocoder is not yet implemented. Use NSF-HiFiGAN vocoder instead.".into()
    ))
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
    let channels = buffer.channels as usize;
    let fade_frames = (buffer.sample_rate as f64 * duration_ms as f64 / 1000.0) as usize;
    let total_frames = buffer.samples.len() / channels;
    let mut samples = buffer.samples.clone();

    for frame in 0..fade_frames.min(total_frames) {
        let gain = frame as f32 / fade_frames as f32;
        for ch in 0..channels {
            samples[frame * channels + ch] *= gain;
        }
    }

    AudioBuffer {
        samples,
        sample_rate: buffer.sample_rate,
        channels: buffer.channels,
    }
}

fn apply_fade_out(buffer: &AudioBuffer, duration_ms: u32) -> AudioBuffer {
    let channels = buffer.channels as usize;
    let fade_frames = (buffer.sample_rate as f64 * duration_ms as f64 / 1000.0) as usize;
    let total_frames = buffer.samples.len() / channels;
    let mut samples = buffer.samples.clone();

    for i in 0..fade_frames.min(total_frames) {
        let frame = total_frames - 1 - i;
        let gain = i as f32 / fade_frames as f32;
        for ch in 0..channels {
            samples[frame * channels + ch] *= gain;
        }
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
