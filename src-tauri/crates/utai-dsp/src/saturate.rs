//! tanh soft saturation (Phase 4-3, 配方 F ④).
//!
//! `y = tanh(drive·x) / tanh(drive)` with the drive input clamped to
//! [-1, 1] first — peak-normalized soft saturation. Properties:
//!   * `y(±1) = ±1` for every drive, and the input clamp guarantees the
//!     output NEVER exceeds full scale — the downstream limiter/LUFS stage
//!     sees a strictly bounded signal;
//!   * slope at 0 is `drive / tanh(drive)` ≈ 1 + drive²/3 → near-transparent
//!     at the distribution default drive 5%, warming up as drive rises;
//!   * `drive = 0` is an exact bypass (short-circuited, no per-sample work).
//!
//! `drive` is clamped to [0, 1] by the kernel; the command layer reports the
//! applied value and the in/out peaks.

/// Apply tanh saturation in place. Returns the clamped drive actually used.
pub fn apply_saturation(samples: &mut [f32], drive: f64) -> f64 {
    let d = drive.clamp(0.0, 1.0);
    if d < 1e-4 || samples.is_empty() {
        return 0.0; // transparent
    }
    let d32 = d as f32;
    let norm = d32.tanh();
    for s in samples.iter_mut() {
        *s = (d32 * s.clamp(-1.0, 1.0)).tanh() / norm;
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f64 = 44100.0;
    const EPS: f32 = 1e-5;

    #[test]
    fn zero_drive_is_bypass() {
        let mut x: Vec<f32> = (0..100).map(|i| (i as f32 * 0.05).sin()).collect();
        let expect = x.clone();
        apply_saturation(&mut x, 0.0);
        assert_eq!(x, expect);
    }

    #[test]
    fn full_scale_maps_to_full_scale() {
        // y(1) = tanh(d)/tanh(d) = 1 for any drive.
        for d in [0.05f64, 0.3, 1.0] {
            let mut x = vec![1.0f32, -1.0];
            apply_saturation(&mut x, d);
            assert!((x[0] - 1.0).abs() < EPS, "drive {d}: y(1) = {}", x[0]);
            assert!((x[1] + 1.0).abs() < EPS, "drive {d}: y(-1) = {}", x[1]);
        }
    }

    #[test]
    fn output_never_exceeds_full_scale() {
        let mut x: Vec<f32> = (0..1000)
            .map(|i| 1.5 * (i as f32 * 0.037).sin()) // intentionally over 1.0
            .collect();
        apply_saturation(&mut x, 0.6);
        assert!(x.iter().all(|&v| v.abs() <= 1.0 + EPS));
    }

    #[test]
    fn small_drive_is_nearly_transparent() {
        let mut x = vec![0.5f32];
        apply_saturation(&mut x, 0.05);
        // slope ≈ d/tanh(d) ≈ 1.0008 → y(0.5) ≈ 0.5000416
        assert!((x[0] - 0.5).abs() < 1e-3, "5% drive should be transparent, got {}", x[0]);
    }

    #[test]
    fn high_drive_warms_mid_levels() {
        // Mid-level signal gets lifted toward the ceiling (soft compression).
        let mut x = vec![0.5f32];
        apply_saturation(&mut x, 1.0);
        let expected = (0.5f32).tanh() / 1.0f32.tanh();
        assert!((x[0] - expected).abs() < EPS);
        assert!(x[0] > 0.6, "drive 1 should lift 0.5 toward the ceiling, got {}", x[0]);
    }

    #[test]
    fn drive_is_clamped() {
        let mut x = vec![0.5f32];
        let applied = apply_saturation(&mut x, 42.0);
        assert_eq!(applied, 1.0, "over-range drive clamps to 1");
        let applied_neg = apply_saturation(&mut vec![0.5f32], -3.0);
        assert_eq!(applied_neg, 0.0, "negative drive clamps to bypass");
    }

    #[test]
    fn odd_symmetry_no_dc() {
        // tanh is odd and normalized by a constant → zero DC on symmetric
        // input. Use an integer number of cycles (1000 Hz @ 44.1 kHz over
        // 4410 samples = exactly 100 cycles) so the sample mean is exact.
        let mut x: Vec<f32> = (0..4410)
            .map(|i| 0.9 * (2.0 * std::f64::consts::PI * 1000.0 * i as f64 / SR).sin() as f32)
            .collect();
        apply_saturation(&mut x, 0.8);
        let dc: f64 = x.iter().map(|&v| v as f64).sum::<f64>() / x.len() as f64;
        assert!(dc.abs() < 1e-4, "saturation must not introduce DC, got {dc}");
    }
}
