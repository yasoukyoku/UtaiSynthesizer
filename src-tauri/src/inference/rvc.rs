use ndarray::{Array1, Array2, Axis};

use super::engine::OnnxEngine;
use super::{ConvertOptions, SynthesisResult};
use crate::{Result, UtaiError};

const RVC_SAMPLE_RATE: u32 = 40000;

pub fn infer(
    engine: &OnnxEngine,
    session_id: &str,
    hubert_features: &Array2<f32>,
    f0: &[f32],
    options: &ConvertOptions,
) -> Result<SynthesisResult> {
    let features = apply_index_retrieval(hubert_features, options.index_ratio);
    let f0_shifted = apply_pitch_shift(f0, options.f0_shift);

    // phone: [1, T, 768]
    let phone = features.clone().insert_axis(Axis(0));
    let phone_shape = vec![
        phone.shape()[0],
        phone.shape()[1],
        phone.shape()[2],
    ];
    let phone_data = phone.as_slice().ok_or_else(|| {
        UtaiError::Inference("Phone array not contiguous".to_string())
    })?;

    // phone_lengths: [1] — but run_f32 only takes f32, so encode length as f32
    let phone_lengths_data = vec![phone.shape()[1] as f32];
    let phone_lengths_shape = vec![1usize];

    // pitch: [1, T] — coarse pitch values as f32
    let pitch_data: Vec<f32> = f0_shifted.iter().map(|&x| f0_to_coarse(x) as f32).collect();
    let pitch_shape = vec![1usize, f0_shifted.len()];

    // pitchf: [1, T]
    let pitchf_shape = vec![1usize, f0_shifted.len()];

    let inputs: Vec<(&str, &[f32], &[usize])> = vec![
        ("phone", phone_data, &phone_shape),
        ("phone_lengths", &phone_lengths_data, &phone_lengths_shape),
        ("pitch", &pitch_data, &pitch_shape),
        ("pitchf", &f0_shifted, &pitchf_shape),
    ];

    let outputs = engine.run_f32(session_id, &inputs)?;
    let audio = outputs.into_iter().next().unwrap_or_default();

    Ok(SynthesisResult {
        audio,
        sample_rate: RVC_SAMPLE_RATE,
    })
}

fn apply_pitch_shift(f0: &[f32], semitones: f32) -> Vec<f32> {
    if semitones.abs() < 0.001 {
        return f0.to_vec();
    }
    let ratio = 2.0f32.powf(semitones / 12.0);
    f0.iter().map(|&x| if x > 0.0 { x * ratio } else { 0.0 }).collect()
}

fn f0_to_coarse(f0: f32) -> i64 {
    if f0 < 1.0 {
        return 0;
    }
    let mel = 1127.0 * (1.0 + f0 / 700.0).ln();
    let coarse = (mel / 10.0).round() as i64;
    coarse.clamp(1, 255)
}

fn apply_index_retrieval(features: &Array2<f32>, _ratio: f32) -> Array2<f32> {
    // TODO: KNN index retrieval against .npy index file
    // For now, pass features through unchanged
    features.clone()
}
