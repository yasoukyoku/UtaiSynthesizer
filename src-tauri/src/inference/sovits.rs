use ndarray::Array2;

use super::engine::{InputTensor, OnnxEngine};
use super::{apply_pitch_shift, ConvertOptions, SynthesisResult};
use crate::Result;

pub fn infer(
    engine: &OnnxEngine,
    session_id: &str,
    contentvec_features: &Array2<f32>,
    f0: &[f32],
    options: &ConvertOptions,
    _shallow_diffusion: bool,
    sample_rate: u32,
) -> Result<SynthesisResult> {
    let f0_shifted = apply_pitch_shift(f0, options.f0_shift);
    let t = contentvec_features.shape()[0];
    let ssl_dim = contentvec_features.shape()[1] as i64;

    // c: [1, T, ssl_dim]
    let c_data: Vec<f32> = contentvec_features.iter().copied().collect();
    let c = InputTensor::F32 {
        data: c_data,
        shape: vec![1, t as i64, ssl_dim],
    };

    // f0: [1, T]
    let f0_input = InputTensor::F32 {
        data: f0_shifted.clone(),
        shape: vec![1, t as i64],
    };

    // uv: [1, T] — voiced/unvoiced flags
    let uv_data: Vec<i64> = f0_shifted.iter().map(|&x| if x > 0.0 { 1i64 } else { 0i64 }).collect();
    let uv = InputTensor::I64 {
        data: uv_data,
        shape: vec![1, t as i64],
    };

    // sid: [1]
    let sid = InputTensor::I64 {
        data: vec![options.speaker_id.unwrap_or(0) as i64],
        shape: vec![1],
    };

    let inputs = vec![
        ("c", c),
        ("f0", f0_input),
        ("uv", uv),
        ("sid", sid),
    ];

    let outputs = engine.run(session_id, inputs)?;
    let audio = outputs.into_iter().next().unwrap_or_default();

    Ok(SynthesisResult {
        audio,
        sample_rate,
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
    sample_rate: u32,
) -> Result<SynthesisResult> {
    // Voice blending requires a custom ONNX model that accepts blend_ids + blend_weights
    // For now, use the primary speaker from the first weight
    let primary_sid = speaker_weights.first().map(|(id, _)| *id).unwrap_or(0);
    let mut opts = options.clone();
    opts.speaker_id = Some(primary_sid);
    infer(engine, session_id, contentvec_features, f0, &opts, shallow_diffusion, sample_rate)
}
