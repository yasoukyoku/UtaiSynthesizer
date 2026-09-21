//! Stereo phase-correlation (mono-compatibility) check.
//!
//! Returns the Pearson correlation coefficient between L and R in [-1, 1]:
//!   1.0 = perfectly in phase, 0.0 = uncorrelated, -1.0 = fully out of phase.
//! Below ~0.7 the mono down-mix starts to audibly cancel; below 0 the centre
//! image collapses on mono playback.

pub fn measure_phase_correlation(left: &[f32], right: &[f32]) -> f64 {
    let n = left.len().min(right.len());
    if n == 0 {
        return 1.0; // nothing to correlate → treat as coherent
    }
    let mut sum_l = 0.0f64;
    let mut sum_r = 0.0f64;
    let mut sum_lr = 0.0f64;
    let mut sum_ll = 0.0f64;
    let mut sum_rr = 0.0f64;
    for i in 0..n {
        let l = left[i] as f64;
        let r = right[i] as f64;
        sum_l += l;
        sum_r += r;
        sum_lr += l * r;
        sum_ll += l * l;
        sum_rr += r * r;
    }
    let mean_l = sum_l / n as f64;
    let mean_r = sum_r / n as f64;
    let cov = sum_lr - n as f64 * mean_l * mean_r;
    let var_l = sum_ll - n as f64 * mean_l * mean_l;
    let var_r = sum_rr - n as f64 * mean_r * mean_r;
    let denom = (var_l * var_r).sqrt();
    if denom < 1e-12 {
        return 1.0; // both channels effectively silent → coherent
    }
    (cov / denom).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_channels_correlation_one() {
        let s: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.03).sin()).collect();
        let r = measure_phase_correlation(&s, &s);
        assert!((r - 1.0).abs() < 1e-6, "identical L/R should correlate 1.0, got {r}");
    }

    #[test]
    fn opposite_phase_correlation_minus_one() {
        let l: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.03).sin()).collect();
        let r: Vec<f32> = l.iter().map(|&x| -x).collect();
        let c = measure_phase_correlation(&l, &r);
        assert!((c + 1.0).abs() < 1e-6, "opposite-phase should correlate -1.0, got {c}");
    }

    #[test]
    fn uncorrelated_is_near_zero() {
        // Two independent sine waves at incommensurate frequencies → ~0.
        let l: Vec<f32> = (0..10000).map(|i| (i as f32 * 0.0123).sin()).collect();
        let r: Vec<f32> = (0..10000).map(|i| (i as f32 * 0.0791).sin()).collect();
        let c = measure_phase_correlation(&l, &r);
        assert!(c.abs() < 0.1, "uncorrelated signals should be near 0, got {c}");
    }

    #[test]
    fn empty_input_returns_one() {
        assert_eq!(measure_phase_correlation(&[], &[]), 1.0);
    }
}
