pub mod effects;
pub mod export;
pub mod resample;

use crate::Result;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AudioBuffer {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

impl AudioBuffer {
    pub fn new_mono(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            samples,
            sample_rate,
            channels: 1,
        }
    }

    pub fn duration_secs(&self) -> f64 {
        self.samples.len() as f64 / (self.sample_rate as f64 * self.channels as f64)
    }

    pub fn frame_count(&self) -> usize {
        self.samples.len() / self.channels as usize
    }
}

pub fn load_wav(path: &std::path::Path) -> Result<AudioBuffer> {
    let reader = hound::WavReader::open(path)
        .map_err(|e| crate::UtaiError::Audio(format!("Failed to open WAV: {}", e)))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .filter_map(|s| s.ok())
            .collect(),
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_val)
                .collect()
        }
    };

    Ok(AudioBuffer {
        samples,
        sample_rate,
        channels,
    })
}

pub fn save_wav(path: &std::path::Path, buffer: &AudioBuffer) -> Result<()> {
    let spec = hound::WavSpec {
        channels: buffer.channels,
        sample_rate: buffer.sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| crate::UtaiError::Audio(format!("Failed to create WAV: {}", e)))?;

    for &sample in &buffer.samples {
        writer
            .write_sample(sample)
            .map_err(|e| crate::UtaiError::Audio(format!("Write error: {}", e)))?;
    }

    writer
        .finalize()
        .map_err(|e| crate::UtaiError::Audio(format!("Finalize error: {}", e)))?;

    Ok(())
}
