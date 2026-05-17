use ndarray::Array2;

use super::engine::OnnxEngine;
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

    // Encode i64 phonemes and durations as f32 for run_f32 interface
    let phonemes_f32: Vec<f32> = score.phonemes.iter().map(|&x| x as f32).collect();
    let durations_f32: Vec<f32> = score.durations.iter().map(|&x| x as f32).collect();

    // phonemes: [1, seq_len]
    let phonemes_shape = vec![1usize, seq_len];
    // durations: [1, seq_len]
    let durations_shape = vec![1usize, seq_len];
    // pitches: [1, seq_len]
    let pitches_shape = vec![1usize, seq_len];

    let inputs: Vec<(&str, &[f32], &[usize])> = vec![
        ("phonemes", &phonemes_f32, &phonemes_shape),
        ("durations", &durations_f32, &durations_shape),
        ("pitches", &score.pitches, &pitches_shape),
    ];

    let outputs = engine.run_f32(session_id, &inputs)?;

    // Dual-head output: [0] = HuBERT-base features, [1] = ContentVec features
    // Each output is a flat Vec<f32> representing [1, T, 768] → reshape to [T, 768]
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
        // Placeholder: return empty array when engine returns stub data
        return Ok(Array2::zeros((0, 768)));
    }

    // Assume feature dim is 768; derive T from total length
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
