use ndarray::Array2;

use super::engine::{InputTensor, OnnxEngine};
use crate::{Result, UtaiError};

#[derive(Debug, Clone)]
pub struct S2HOutput {
    pub hubert_features: Array2<f32>,
    pub contentvec_features: Array2<f32>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScoreInput {
    pub phonemes: Vec<i64>,
    pub durations: Vec<i64>,
    pub pitches: Vec<f32>,
}

pub fn infer(
    engine: &OnnxEngine,
    session_id: &str,
    score: &ScoreInput,
) -> Result<S2HOutput> {
    let seq_len = score.phonemes.len();

    let phonemes = InputTensor::I64 {
        data: score.phonemes.clone(),
        shape: vec![1, seq_len as i64],
    };

    let durations = InputTensor::I64 {
        data: score.durations.clone(),
        shape: vec![1, seq_len as i64],
    };

    let pitches = InputTensor::F32 {
        data: score.pitches.clone(),
        shape: vec![1, seq_len as i64],
    };

    let inputs = vec![
        ("phonemes", phonemes),
        ("durations", durations),
        ("pitches", pitches),
    ];

    let outputs = engine.run(session_id, inputs)?;

    let hubert_features = reshape_features(&outputs, 0)?;
    let contentvec_features = reshape_features(&outputs, 1)?;

    Ok(S2HOutput {
        hubert_features,
        contentvec_features,
    })
}

fn reshape_features(outputs: &[Vec<f32>], index: usize) -> Result<Array2<f32>> {
    let data = outputs.get(index).ok_or_else(|| {
        UtaiError::Inference(format!("Missing S2H output at index {}", index))
    })?;

    if data.is_empty() {
        return Ok(Array2::zeros((0, 768)));
    }

    let feature_dim = 768usize;
    let t = data.len() / feature_dim;
    if t * feature_dim != data.len() {
        return Err(UtaiError::Inference(format!(
            "S2H output at index {} has length {} which is not divisible by {}",
            index, data.len(), feature_dim
        )));
    }

    Array2::from_shape_vec((t, feature_dim), data.clone())
        .map_err(|e| UtaiError::Inference(e.to_string()))
}
