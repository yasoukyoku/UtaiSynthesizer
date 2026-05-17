use ndarray::Array2;

use super::engine::OnnxEngine;
use crate::{Result, UtaiError};

const RMVPE_SAMPLE_RATE: u32 = 16000;
const RMVPE_HOP_SIZE: usize = 160;

pub struct F0Result {
    pub f0: Vec<f32>,
    pub voiced: Vec<bool>,
    pub hop_size: usize,
    pub sample_rate: u32,
}

pub fn detect(
    engine: &OnnxEngine,
    session_id: &str,
    audio: &[f32],
    sample_rate: u32,
) -> Result<F0Result> {
    let resampled = if sample_rate != RMVPE_SAMPLE_RATE {
        resample_for_f0(audio, sample_rate, RMVPE_SAMPLE_RATE)?
    } else {
        audio.to_vec()
    };

    let frame_count = resampled.len() / RMVPE_HOP_SIZE;
    let input_len = frame_count * RMVPE_HOP_SIZE;
    let trimmed = &resampled[..input_len];

    // Input shape: [1, input_len]
    let input_shape = vec![1usize, input_len];

    let inputs: Vec<(&str, &[f32], &[usize])> = vec![
        ("audio", trimmed, &input_shape),
    ];

    let outputs = engine.run_f32(session_id, &inputs)?;
    let f0_raw = outputs.into_iter().next().unwrap_or_default();

    let voiced: Vec<bool> = f0_raw.iter().map(|&x| x > 50.0).collect();
    let f0: Vec<f32> = f0_raw
        .iter()
        .zip(voiced.iter())
        .map(|(&f, &v)| if v { f } else { 0.0 })
        .collect();

    Ok(F0Result {
        f0,
        voiced,
        hop_size: RMVPE_HOP_SIZE,
        sample_rate: RMVPE_SAMPLE_RATE,
    })
}

fn resample_for_f0(audio: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    use rubato::{FftFixedIn, Resampler};

    let mut resampler = FftFixedIn::<f32>::new(
        from_rate as usize,
        to_rate as usize,
        audio.len(),
        1,
        1,
    )
    .map_err(|e| UtaiError::Audio(e.to_string()))?;

    let input = vec![audio.to_vec()];
    let output = resampler
        .process(&input, None)
        .map_err(|e| UtaiError::Audio(e.to_string()))?;

    Ok(output.into_iter().next().unwrap_or_default())
}
