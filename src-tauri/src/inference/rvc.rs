use std::path::Path;

use ndarray::{Array1, Array2};

use super::engine::{InputTensor, OnnxEngine};
use super::{apply_pitch_shift, ConvertOptions, SynthesisResult};
use crate::{Result, UtaiError};

const TOP_K: usize = 8;

pub struct RvcIndex {
    raw: Array2<f32>,
    normalized: Array2<f32>,
}

impl RvcIndex {
    pub fn load(path: &Path) -> Result<Self> {
        let raw: Array2<f32> = ndarray_npy::read_npy(path)
            .map_err(|e| UtaiError::Model(format!("Failed to load index '{}': {}", path.display(), e)))?;

        let mut normalized = raw.clone();
        l2_normalize_rows(&mut normalized);

        tracing::info!("Loaded RVC index: {} vectors x {} dim", raw.nrows(), raw.ncols());
        Ok(Self { raw, normalized })
    }
}

pub fn infer(
    engine: &OnnxEngine,
    session_id: &str,
    hubert_features: &Array2<f32>,
    f0: &[f32],
    options: &ConvertOptions,
    index: Option<&RvcIndex>,
    sample_rate: u32,
) -> Result<SynthesisResult> {
    let mut features = if options.index_ratio > 0.0 {
        if let Some(idx) = index {
            apply_index_retrieval(hubert_features, idx, options.index_ratio)
        } else {
            hubert_features.clone()
        }
    } else {
        hubert_features.clone()
    };

    if options.l2_normalize {
        l2_normalize_rows(&mut features);
    }
    let f0_shifted = apply_pitch_shift(f0, options.f0_shift);

    let t = features.shape()[0];

    let phone_data: Vec<f32> = features.iter().copied().collect();
    let phone = InputTensor::F32 {
        data: phone_data,
        shape: vec![1, t as i64, 768],
    };

    let phone_lengths = InputTensor::I64 {
        data: vec![t as i64],
        shape: vec![1],
    };

    let pitch_data: Vec<i64> = f0_shifted.iter().map(|&x| f0_to_coarse(x)).collect();
    let pitch = InputTensor::I64 {
        data: pitch_data,
        shape: vec![1, t as i64],
    };

    let pitchf = InputTensor::F32 {
        data: f0_shifted.clone(),
        shape: vec![1, t as i64],
    };

    let sid = InputTensor::I64 {
        data: vec![options.speaker_id.unwrap_or(0) as i64],
        shape: vec![1],
    };

    let noise_scale = InputTensor::F32 {
        data: vec![1.0 - options.protect_voiceless],
        shape: vec![1],
    };

    let inputs = vec![
        ("phone", phone),
        ("phone_lengths", phone_lengths),
        ("pitch", pitch),
        ("pitchf", pitchf),
        ("sid", sid),
        ("noise_scale", noise_scale),
    ];

    let outputs = engine.run(session_id, inputs)?;
    let audio = outputs
        .into_iter()
        .next()
        .ok_or_else(|| UtaiError::Inference("RVC model returned no output tensors".into()))?;

    Ok(SynthesisResult {
        audio,
        sample_rate,
    })
}

fn f0_to_coarse(f0: f32) -> i64 {
    let f0_mel = 1127.0_f32 * (1.0 + f0 / 700.0).ln();
    if f0_mel <= 0.0 {
        return 1;
    }
    // Original RVC normalizes mel range [f0_min=50Hz, f0_max=1100Hz] → [1, 255]
    // f0_mel_min = 1127 * ln(1 + 50/700) ≈ 77.74
    // f0_mel_max = 1127 * ln(1 + 1100/700) ≈ 1064.42
    let normalized = (f0_mel - 77.74) / (1064.42 - 77.74) * 254.0 + 1.0;
    (normalized.round() as i64).clamp(1, 255)
}

fn l2_normalize_rows(features: &mut Array2<f32>) {
    for mut row in features.rows_mut() {
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        row.iter_mut().for_each(|x| *x /= norm);
    }
}

/// Brute-force top-K KNN retrieval with cosine similarity.
///
/// For each frame in `features`, finds the K nearest neighbors in the index,
/// then returns a weighted blend of the original features and the retrieved vectors.
fn apply_index_retrieval(
    features: &Array2<f32>,
    index: &RvcIndex,
    ratio: f32,
) -> Array2<f32> {
    let t = features.nrows();
    let dim = features.ncols();
    let n = index.raw.nrows();

    if n == 0 || dim != index.raw.ncols() {
        return features.clone();
    }

    let mut query_norm = features.clone();
    l2_normalize_rows(&mut query_norm);

    // similarity: [T, N] = query_norm @ index_norm.T
    let similarity = query_norm.dot(&index.normalized.t());

    let mut result = Array2::zeros((t, dim));

    for frame in 0..t {
        let sim_row = similarity.row(frame);

        let top_k = top_k_indices(sim_row.as_slice().unwrap(), TOP_K.min(n));

        let mut weight_sum = 0.0f32;
        let mut retrieved = Array1::zeros(dim);

        for &(idx, score) in &top_k {
            let w = score.max(0.0);
            retrieved += &(&index.raw.row(idx) * w);
            weight_sum += w;
        }

        if weight_sum > 1e-12 {
            retrieved /= weight_sum;
        }

        let orig = features.row(frame);
        for j in 0..dim {
            result[[frame, j]] = orig[j] * (1.0 - ratio) + retrieved[j] * ratio;
        }
    }

    result
}

/// Returns top-K indices with their similarity scores, sorted descending.
fn top_k_indices(scores: &[f32], k: usize) -> Vec<(usize, f32)> {
    use std::cmp::Ordering;

    // Min-heap of size K: (score, index). Keeps the K largest scores.
    let mut heap: Vec<(usize, f32)> = Vec::with_capacity(k + 1);

    for (i, &score) in scores.iter().enumerate() {
        if heap.len() < k {
            heap.push((i, score));
            if heap.len() == k {
                heap.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            }
        } else if score > heap[0].1 {
            heap[0] = (i, score);
            heap.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        }
    }

    heap.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    heap
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn f0_to_coarse_matches_original_rvc() {
        // Exact values verified against Python original:
        // f0_mel_min = 1127*ln(1+50/700), f0_mel_max = 1127*ln(1+1100/700)
        // coarse = round((mel - mel_min) / (mel_max - mel_min) * 254 + 1), clamp [1, 255]
        assert_eq!(f0_to_coarse(0.0), 1);
        assert_eq!(f0_to_coarse(50.0), 1);
        assert_eq!(f0_to_coarse(100.0), 20);
        assert_eq!(f0_to_coarse(220.0), 60);
        assert_eq!(f0_to_coarse(440.0), 122);
        assert_eq!(f0_to_coarse(880.0), 217);
        assert_eq!(f0_to_coarse(1100.0), 255);
        assert_eq!(f0_to_coarse(2000.0), 255);

        // Monotonicity
        assert!(f0_to_coarse(220.0) < f0_to_coarse(440.0));
        assert!(f0_to_coarse(440.0) < f0_to_coarse(880.0));
    }

    #[test]
    fn l2_normalize_produces_unit_vectors() {
        let mut features = array![[3.0, 4.0], [0.0, 5.0], [1.0, 0.0]];
        l2_normalize_rows(&mut features);

        for row in features.rows() {
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-6, "Row norm should be 1.0, got {}", norm);
        }

        // [3, 4] → [0.6, 0.8]
        assert!((features[[0, 0]] - 0.6).abs() < 1e-6);
        assert!((features[[0, 1]] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn top_k_finds_largest() {
        let scores = vec![0.1, 0.9, 0.5, 0.8, 0.3, 0.7];
        let result = top_k_indices(&scores, 3);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].0, 1); // 0.9
        assert_eq!(result[1].0, 3); // 0.8
        assert_eq!(result[2].0, 5); // 0.7
    }

    #[test]
    fn top_k_handles_small_input() {
        let scores = vec![0.5, 0.3];
        let result = top_k_indices(&scores, 8);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, 0);
        assert_eq!(result[1].0, 1);
    }

    #[test]
    fn index_retrieval_blends_correctly() {
        // 2 index vectors, 2-dim for simplicity
        let raw = array![[1.0, 0.0], [0.0, 1.0]];
        let mut normalized = raw.clone();
        l2_normalize_rows(&mut normalized);
        let index = RvcIndex { raw, normalized };

        // Query: [1, 0] should match index[0] perfectly
        let query = array![[1.0, 0.0]];
        let result = apply_index_retrieval(&query, &index, 1.0);

        // ratio=1.0 → fully retrieved → should be close to [1, 0] (the nearest neighbor)
        assert!((result[[0, 0]] - 1.0).abs() < 0.1, "Expected ~1.0, got {}", result[[0, 0]]);
        assert!(result[[0, 1]].abs() < 0.1, "Expected ~0.0, got {}", result[[0, 1]]);

        // ratio=0.0 → fully original
        let result0 = apply_index_retrieval(&query, &index, 0.0);
        // ratio=0 is handled by the caller (skips retrieval), but if called:
        // (1-0)*orig + 0*retrieved = orig
        assert!((result0[[0, 0]] - 1.0).abs() < 1e-6);
    }
}
