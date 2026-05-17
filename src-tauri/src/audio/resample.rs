use rubato::{FftFixedIn, Resampler};

use super::AudioBuffer;
use crate::{Result, UtaiError};

pub fn resample(buffer: &AudioBuffer, target_rate: u32) -> Result<AudioBuffer> {
    if buffer.sample_rate == target_rate {
        return Ok(buffer.clone());
    }

    let channels = buffer.channels as usize;
    let frames_per_channel = buffer.samples.len() / channels;

    let mut channel_data: Vec<Vec<f32>> = Vec::with_capacity(channels);
    for ch in 0..channels {
        let data: Vec<f32> = buffer
            .samples
            .iter()
            .skip(ch)
            .step_by(channels)
            .copied()
            .collect();
        channel_data.push(data);
    }

    let mut resampler = FftFixedIn::<f32>::new(
        buffer.sample_rate as usize,
        target_rate as usize,
        frames_per_channel,
        1,
        channels,
    )
    .map_err(|e| UtaiError::Audio(format!("Resampler init: {}", e)))?;

    let output = resampler
        .process(&channel_data, None)
        .map_err(|e| UtaiError::Audio(format!("Resampling: {}", e)))?;

    let out_frames = output[0].len();
    let mut interleaved = Vec::with_capacity(out_frames * channels);
    for i in 0..out_frames {
        for ch in 0..channels {
            interleaved.push(output[ch][i]);
        }
    }

    Ok(AudioBuffer {
        samples: interleaved,
        sample_rate: target_rate,
        channels: buffer.channels,
    })
}

pub fn to_mono(buffer: &AudioBuffer) -> AudioBuffer {
    if buffer.channels == 1 {
        return buffer.clone();
    }

    let channels = buffer.channels as usize;
    let frames = buffer.samples.len() / channels;
    let mut mono = Vec::with_capacity(frames);

    for i in 0..frames {
        let sum: f32 = (0..channels)
            .map(|ch| buffer.samples[i * channels + ch])
            .sum();
        mono.push(sum / channels as f32);
    }

    AudioBuffer::new_mono(mono, buffer.sample_rate)
}
