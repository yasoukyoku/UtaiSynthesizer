use ndarray::{Array2, Axis};

use super::engine::OnnxEngine;
use super::{ConvertOptions, SynthesisResult};
use crate::{Result, UtaiError};

const SOVITS_SAMPLE_RATE: u32 = 44100;

pub fn infer(
    engine: &OnnxEngine,
    session_id: &str,
    contentvec_features: &Array2<f32>,
    f0: &[f32],
    options: &ConvertOptions,
    shallow_diffusion: bool,
) -> Result<SynthesisResult> {
    let f0_shifted = apply_pitch_shift(f0, options.f0_shift);

    // c: [1, T, 768]
    let features = contentvec_features.clone().insert_axis(Axis(0));
    let c_shape = vec![
        features.shape()[0],
        features.shape()[1],
        features.shape()[2],
    ];
    let c_data = features.as_slice().ok_or_else(|| {
        UtaiError::Inference("Features array not contiguous".to_string())
    })?;

    // f0: [1, T]
    let f0_shape = vec![1usize, f0_shifted.len()];

    // sid: [1] — speaker id as f32
    let speaker_id = options.speaker_id.unwrap_or(0);
    let sid_data = vec![speaker_id as f32];
    let sid_shape = vec![1usize];

    let mut inputs: Vec<(&str, &[f32], &[usize])> = vec![
        ("c", c_data, &c_shape),
        ("f0", &f0_shifted, &f0_shape),
        ("sid", &sid_data, &sid_shape),
    ];

    // shallow diffusion: n_steps as f32
    let n_steps_data = vec![20.0f32];
    let n_steps_shape = vec![1usize];
    if shallow_diffusion {
        inputs.push(("n_steps", &n_steps_data, &n_steps_shape));
    }

    let outputs = engine.run_f32(session_id, &inputs)?;
    let audio = outputs.into_iter().next().unwrap_or_default();

    Ok(SynthesisResult {
        audio,
        sample_rate: SOVITS_SAMPLE_RATE,
    })
}

pub fn infer_blended(
    engine: &OnnxEngine,
    session_id: &str,
    contentvec_features: &Array2<f32>,
    f0: &[f32],
    options: &ConvertOptions,
    speaker_weights: &[(u32, f32)],
    shallow_diffusion: bool,
) -> Result<SynthesisResult> {
    let f0_shifted = apply_pitch_shift(f0, options.f0_shift);

    // c: [1, T, 768]
    let features = contentvec_features.clone().insert_axis(Axis(0));
    let c_shape = vec![
        features.shape()[0],
        features.shape()[1],
        features.shape()[2],
    ];
    let c_data = features.as_slice().ok_or_else(|| {
        UtaiError::Inference("Features array not contiguous".to_string())
    })?;

    // f0: [1, T]
    let f0_shape = vec![1usize, f0_shifted.len()];

    // blend_ids: [N] — as f32
    let blend_ids_data: Vec<f32> = speaker_weights.iter().map(|(id, _)| *id as f32).collect();
    let blend_ids_shape = vec![speaker_weights.len()];

    // blend_weights: [N]
    let blend_weights_data: Vec<f32> = speaker_weights.iter().map(|(_, w)| *w).collect();
    let blend_weights_shape = vec![speaker_weights.len()];

    let mut inputs: Vec<(&str, &[f32], &[usize])> = vec![
        ("c", c_data, &c_shape),
        ("f0", &f0_shifted, &f0_shape),
        ("blend_ids", &blend_ids_data, &blend_ids_shape),
        ("blend_weights", &blend_weights_data, &blend_weights_shape),
    ];

    let n_steps_data = vec![20.0f32];
    let n_steps_shape = vec![1usize];
    if shallow_diffusion {
        inputs.push(("n_steps", &n_steps_data, &n_steps_shape));
    }

    let outputs = engine.run_f32(session_id, &inputs)?;
    let audio = outputs.into_iter().next().unwrap_or_default();

    Ok(SynthesisResult {
        audio,
        sample_rate: SOVITS_SAMPLE_RATE,
    })
}

fn apply_pitch_shift(f0: &[f32], semitones: f32) -> Vec<f32> {
    if semitones.abs() < 0.001 {
        return f0.to_vec();
    }
    let ratio = 2.0f32.powf(semitones / 12.0);
    f0.iter().map(|&x| if x > 0.0 { x * ratio } else { 0.0 }).collect()
}
