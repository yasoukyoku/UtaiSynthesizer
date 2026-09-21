//! M/S stereo width (Phase 4-2, 配方 F ③).
//!
//! Mid = (L+R)/2, Side = (L−R)/2; out L = M + w·S, out R = M − w·S.
//! w = 1.0 is the identity, 0 folds to mono, >1 widens. The distribution
//! spec caps width at [`MAX_WIDTH`] (1.5) — beyond that mono compatibility
//! collapses (side content dominates and out-of-phase cancellation shows up
//! on mono playback). The kernel clamps and reports whether the cap was hit
//! so the UI can surface a warning instead of silently failing.

/// Distribution cap for the width control.
pub const MAX_WIDTH: f64 = 1.5;

/// Apply M/S width in place. Returns `true` when the requested width was
/// clamped to [`MAX_WIDTH`] (or below 0).
///
/// Mono input (`right == None`) is a no-op: there is no side channel.
pub fn apply_stereo_width(left: &mut [f32], right: Option<&mut [f32]>, width: f64) -> bool {
    let Some(r) = right else { return false };
    let capped = !(0.0..=MAX_WIDTH).contains(&width);
    let w = width.clamp(0.0, MAX_WIDTH) as f32;
    if (w - 1.0).abs() < 1e-9 {
        return false; // identity — skip the copy loop entirely
    }
    for (l, rr) in left.iter_mut().zip(r.iter_mut()) {
        let m = (*l + *rr) * 0.5;
        let s = (*l - *rr) * 0.5;
        *l = m + w * s;
        *rr = m - w * s;
    }
    capped
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    #[test]
    fn width_one_is_identity() {
        let mut l: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut r: Vec<f32> = (0..64).map(|i| (i as f32 * 0.13).cos()).collect();
        let (le, re) = (l.clone(), r.clone());
        assert!(!apply_stereo_width(&mut l, Some(&mut r), 1.0));
        assert_eq!(l, le);
        assert_eq!(r, re);
    }

    #[test]
    fn width_zero_folds_to_mono() {
        let mut l = vec![0.8f32; 32];
        let mut r = vec![-0.2f32; 32];
        apply_stereo_width(&mut l, Some(&mut r), 0.0);
        for (lv, rv) in l.iter().zip(r.iter()) {
            assert!((lv - 0.3).abs() < EPS, "L should equal mid 0.3, got {lv}");
            assert!((rv - 0.3).abs() < EPS, "R should equal mid 0.3, got {rv}");
        }
    }

    #[test]
    fn width_two_clamps_to_max_and_reports() {
        let mut l = vec![0.6f32; 16];
        let mut r = vec![0.2f32; 16];
        let capped = apply_stereo_width(&mut l, Some(&mut r), 2.0);
        assert!(capped, "width 2.0 must be reported as capped");
        // w=1.5: M=0.4, S=0.2 → L=0.4+0.3=0.7, R=0.4-0.3=0.1
        assert!((l[0] - 0.7).abs() < EPS, "got {}", l[0]);
        assert!((r[0] - 0.1).abs() < EPS, "got {}", r[0]);
    }

    #[test]
    fn negative_width_clamps_to_mono() {
        let mut l = vec![0.6f32; 8];
        let mut r = vec![0.2f32; 8];
        let capped = apply_stereo_width(&mut l, Some(&mut r), -0.5);
        assert!(capped);
        assert!((l[0] - 0.4).abs() < EPS && (r[0] - 0.4).abs() < EPS);
    }

    #[test]
    fn within_cap_is_not_reported() {
        let mut l = vec![0.6f32; 8];
        let mut r = vec![0.2f32; 8];
        assert!(!apply_stereo_width(&mut l, Some(&mut r), 1.5));
        assert!((l[0] - 0.7).abs() < EPS);
    }

    #[test]
    fn mono_input_is_noop() {
        let mut l = vec![0.5f32; 8];
        let before = l.clone();
        assert!(!apply_stereo_width(&mut l, None, 1.4));
        assert_eq!(l, before);
    }

    #[test]
    fn width_round_trip_preserves_mid() {
        // (M + wS, M − wS) at w then at 1/w keeps M and restores S sign only
        // when w == 1; the invariant that MUST hold for any w is M equality.
        let mut l: Vec<f32> = (0..32).map(|i| (i as f32 * 0.21).sin()).collect();
        let mut r: Vec<f32> = (0..32).map(|i| (i as f32 * 0.07).cos()).collect();
        let mid_before: Vec<f32> = l.iter().zip(r.iter()).map(|(a, b)| (a + b) * 0.5).collect();
        apply_stereo_width(&mut l, Some(&mut r), 1.3);
        let mid_after: Vec<f32> = l.iter().zip(r.iter()).map(|(a, b)| (a + b) * 0.5).collect();
        for (a, b) in mid_before.iter().zip(mid_after.iter()) {
            assert!((a - b).abs() < 1e-5, "mid must be invariant: {a} vs {b}");
        }
    }
}
