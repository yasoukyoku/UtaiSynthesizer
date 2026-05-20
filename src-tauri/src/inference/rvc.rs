use ndarray::Array2;

use super::engine::{InputTensor, OnnxEngine};
use super::{apply_pitch_shift, ConvertOptions, SynthesisResult};
use crate::Result;

pub fn infer(
    engine: &OnnxEngine,
    session_id: &str,
    hubert_features: &Array2<f32>,
    f0: &[f32],
    options: &ConvertOptions,
    sample_rate: u32,
) -> Result<SynthesisResult> {
    let features = apply_index_retrieval(hubert_features, options.index_ratio);
    let f0_shifted = apply_pitch_shift(f0, options.f0_shift);

    let t = features.shape()[0];

    // phone: [1, T, 768]
    let phone_data: Vec<f32> = features.iter().copied().collect();
    let phone = InputTensor::F32 {
        data: phone_data,
        shape: vec![1, t as i64, 768],
    };

    // phone_lengths: [1]
    let phone_lengths = InputTensor::I64 {
        data: vec![t as i64],
        shape: vec![1],
    };

    // pitch: [1, T] — coarse pitch (i64 for embedding lookup)
    let pitch_data: Vec<i64> = f0_shifted.iter().map(|&x| f0_to_coarse(x)).collect();
    let pitch = InputTensor::I64 {
        data: pitch_data,
        shape: vec![1, t as i64],
    };

    // pitchf: [1, T] — continuous F0 in Hz
    let pitchf = InputTensor::F32 {
        data: f0_shifted.clone(),
        shape: vec![1, t as i64],
    };

    // sid: [1] — speaker ID
    let sid = InputTensor::I64 {
        data: vec![options.speaker_id.unwrap_or(0) as i64],
        shape: vec![1],
    };

    let inputs = vec![
        ("phone", phone),
        ("phone_lengths", phone_lengths),
        ("pitch", pitch),
        ("pitchf", pitchf),
        ("sid", sid),
    ];

    let outputs = engine.run(session_id, inputs)?;
    let audio = outputs.into_iter().next().unwrap_or_default();

    Ok(SynthesisResult {
        audio,
        sample_rate,
    })
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
    features.clone()
}
