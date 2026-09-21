//! Phase 5-6: Dynamic Time Warping for the `dtwAlign` analysis node.
//!
//! Compares two clips' log-mel feature sequences (from `timbre::log_mel_frames`)
//! with cosine frame distance and an optional Sakoe–Chiba band (band = 0 →
//! unrestricted). Frame counts are internally capped so the DP matrix stays
//! ≤ 2048×2048 (≈ 16 MB) regardless of clip length.

/// Max frames per side after striding (2048² f32 ≈ 16 MB DP matrix).
const MAX_FRAMES: usize = 2048;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DtwResult {
    /// Total accumulated path cost.
    pub cost: f32,
    /// cost / path_len — length-normalized so short/long clips compare fairly.
    pub mean_step_cost: f32,
    /// Warping path length in steps.
    pub path_len: usize,
    /// Frames used from A (after striding).
    pub a_frames: usize,
    /// Frames used from B (after striding).
    pub b_frames: usize,
}

/// Cosine distance in [0, 2]; zero-norm frames get the neutral 1.0.
pub fn cosine_distance(u: &[f32], v: &[f32]) -> f32 {
    let (mut dot, mut nu, mut nv) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..u.len().min(v.len()) {
        dot += u[i] * v[i];
        nu += u[i] * u[i];
        nv += v[i] * v[i];
    }
    if nu <= 1e-12 || nv <= 1e-12 {
        return 1.0;
    }
    1.0 - dot / (nu.sqrt() * nv.sqrt())
}

/// Uniform stride down to ≤ MAX_FRAMES (keeps the coarse temporal structure).
fn stride(feats: &[Vec<f32>]) -> Vec<Vec<f32>> {
    if feats.len() <= MAX_FRAMES {
        return feats.to_vec();
    }
    let k = (feats.len() + MAX_FRAMES - 1) / MAX_FRAMES;
    (0..feats.len()).step_by(k).map(|i| feats[i].clone()).collect()
}

/// Align `a` against `b`. Returns None only when either side is empty.
/// `band` = max |i − j·na/nb| deviation in A-frames; 0 disables the band.
pub fn dtw_align(a: &[Vec<f32>], b: &[Vec<f32>], band: usize) -> Option<DtwResult> {
    let a = stride(a);
    let b = stride(b);
    let (na, nb) = (a.len(), b.len());
    if na == 0 || nb == 0 {
        return None;
    }

    let inf = f32::INFINITY;
    let mut dp = vec![vec![inf; nb]; na];
    for i in 0..na {
        let (j0, j1) = if band == 0 {
            (0usize, nb - 1)
        } else {
            let lo = (((i as f32 - band as f32) * nb as f32 / na as f32).floor() as isize).max(0);
            let hi =
                (((i as f32 + band as f32) * nb as f32 / na as f32).ceil() as isize)
                    .min(nb as isize - 1);
            (lo as usize, hi as usize)
        };
        for j in j0..=j1 {
            let d = cosine_distance(&a[i], &b[j]);
            let best = if i == 0 && j == 0 {
                d
            } else {
                let up = if i > 0 { dp[i - 1][j] } else { inf };
                let left = if j > 0 { dp[i][j - 1] } else { inf };
                let diag = if i > 0 && j > 0 { dp[i - 1][j - 1] } else { inf };
                let m = up.min(left).min(diag);
                if m.is_finite() { d + m } else { inf }
            };
            dp[i][j] = best;
        }
    }
    let cost = dp[na - 1][nb - 1];
    if !cost.is_finite() {
        return None;
    }

    // Backtrack one optimal path; every finite cell was written from a finite
    // predecessor, so the chain always reaches (0, 0).
    let (mut i, mut j) = (na - 1, nb - 1);
    let mut path_len = 1usize;
    while i > 0 || j > 0 {
        let mut best = inf;
        let mut pi = i;
        let mut pj = j;
        if i > 0 && j > 0 && dp[i - 1][j - 1] < best {
            best = dp[i - 1][j - 1];
            pi = i - 1;
            pj = j - 1;
        }
        if i > 0 && dp[i - 1][j] < best {
            best = dp[i - 1][j];
            pi = i - 1;
            pj = j;
        }
        if j > 0 && dp[i][j - 1] < best {
            best = dp[i][j - 1];
            pi = i;
            pj = j - 1;
        }
        debug_assert!(best.is_finite(), "backtrack lost the finite path");
        i = pi;
        j = pj;
        path_len += 1;
    }
    Some(DtwResult {
        cost,
        mean_step_cost: cost / path_len as f32,
        path_len,
        a_frames: na,
        b_frames: nb,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn axis_line(n: usize) -> Vec<Vec<f32>> {
        // Constant [3, 4] frames: norms are exact (5.0), so identical
        // sequences score d = 0.0 bit-exactly and ties resolve to diag.
        vec![vec![3.0f32, 4.0]; n]
    }

    #[test]
    fn identical_sequences_cost_zero() {
        let a = axis_line(50);
        let r = dtw_align(&a, &a, 0).unwrap();
        assert_eq!(r.cost, 0.0, "identical frames are bit-exact zero distance");
        assert_eq!(r.path_len, 50);
        assert_eq!(r.mean_step_cost, 0.0);
        assert_eq!((r.a_frames, r.b_frames), (50, 50));
    }

    #[test]
    fn shifted_aligns_cheaper_than_mismatch() {
        // Cyclic 2-D phase features: b follows a with a 3-frame lag (stalled
        // at the end so the warp stays monotone); the mismatch alternates axes.
        let a: Vec<Vec<f32>> = (0..40)
            .map(|i| {
                let t = 2.0 * std::f32::consts::PI * i as f32 / 20.0;
                vec![t.sin(), t.cos()]
            })
            .collect();
        let shifted: Vec<Vec<f32>> = (0..40).map(|j| a[(j + 3).min(39)].clone()).collect();
        let mismatch: Vec<Vec<f32>> = (0..40)
            .map(|j| if j % 2 == 0 { vec![1.0, 0.0] } else { vec![0.0, 1.0] })
            .collect();
        let r_s = dtw_align(&a, &shifted, 0).unwrap();
        let r_m = dtw_align(&a, &mismatch, 0).unwrap();
        assert!(r_s.cost < r_m.cost, "shifted {} vs mismatch {}", r_s.cost, r_m.cost);
        assert!(r_s.mean_step_cost < r_m.mean_step_cost);
    }

    #[test]
    fn empty_inputs_return_none() {
        let a = axis_line(5);
        assert!(dtw_align(&[], &a, 0).is_none());
        assert!(dtw_align(&a, &[], 0).is_none());
        assert!(dtw_align(&[], &[], 0).is_none());
    }

    #[test]
    fn band_is_a_pure_restriction() {
        // a runs x-axis then y-axis; b runs the reverse order — realignment
        // needs large deviation, so a tight band can only cost more.
        let mut a: Vec<Vec<f32>> = (0..8).map(|_| vec![1.0, 0.0]).collect();
        a.extend((0..8).map(|_| vec![0.0, 1.0]));
        let mut b: Vec<Vec<f32>> = (0..8).map(|_| vec![0.0, 1.0]).collect();
        b.extend((0..8).map(|_| vec![1.0, 0.0]));
        let free = dtw_align(&a, &b, 0).unwrap();
        let wide = dtw_align(&a, &b, 1000).unwrap();
        assert_eq!(free.cost, wide.cost, "a wide band must equal the free DP");
        let tight = dtw_align(&a, &b, 1).unwrap();
        assert!(tight.cost >= free.cost - 1e-4, "tight {} vs free {}", tight.cost, free.cost);
    }

    #[test]
    fn long_inputs_are_strided_not_exploded() {
        let a = axis_line(5000);
        let r = dtw_align(&a, &a, 0).unwrap();
        assert!(r.a_frames <= MAX_FRAMES, "a_frames {}", r.a_frames);
        assert_eq!(r.a_frames, r.b_frames);
        assert!(r.cost < 1e-3);
    }
}
