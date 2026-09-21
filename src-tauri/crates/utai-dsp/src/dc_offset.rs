//! DC offset detection and removal.
//!
//! DC offset (mean ≠ 0) eats headroom and can trip peak limiters. Distribution
//! spec: |DC| < 0.001 (0.1% of full scale). Removal is a single-pole high-pass
//! at ~10 Hz or, simpler and phase-linear, subtraction of the per-channel mean.

/// Returns (dc_left, dc_right) as fractions of full scale. For mono input the
/// right value equals the left.
pub fn measure_dc_offset(left: &[f32], right: Option<&[f32]>) -> (f64, f64) {
    let mean = |x: &[f32]| -> f64 {
        if x.is_empty() {
            return 0.0;
        }
        let sum: f64 = x.iter().map(|&s| s as f64).sum();
        sum / x.len() as f64
    };
    let l = mean(left);
    let r = right.map(mean).unwrap_or(l);
    (l, r)
}

/// Remove DC offset in place by subtracting the per-channel mean.
/// This is a linear, phase-preserving operation (pure gain = -mean offset).
pub fn remove_dc_offset(left: &mut [f32], right: Option<&mut [f32]>) {
    let (l, r) = measure_dc_offset(left, right.as_deref());
    for s in left.iter_mut() {
        *s -= l as f32;
    }
    if let Some(rch) = right {
        for s in rch.iter_mut() {
            *s -= r as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measures_known_offset() {
        let l = vec![0.05f32; 1000];
        let (dc, _) = measure_dc_offset(&l, None);
        assert!((dc - 0.05).abs() < 1e-6, "DC should be 0.05, got {dc}");
    }

    #[test]
    fn removal_brings_dc_to_zero() {
        let mut l: Vec<f32> = (0..1000).map(|i| -0.03 + (i as f32 * 0.02).sin()).collect();
        let mut r: Vec<f32> = (0..1000).map(|i| 0.07 + (i as f32 * 0.02).cos()).collect();
        remove_dc_offset(&mut l, Some(&mut r));
        let (dc_l, dc_r) = measure_dc_offset(&l, Some(&r));
        assert!(dc_l.abs() < 1e-6);
        assert!(dc_r.abs() < 1e-6);
    }

    #[test]
    fn empty_is_zero() {
        let (dc_l, dc_r) = measure_dc_offset(&[], None);
        assert_eq!(dc_l, 0.0);
        assert_eq!(dc_r, 0.0);
    }
}
