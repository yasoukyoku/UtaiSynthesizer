//! All-pass phase rotation (Phase 4-5, 配方 F ④′).
//!
//! Cascaded 2nd-order all-pass sections with log-spaced pole frequencies.
//! An all-pass changes ONLY phase — the per-frequency magnitude response is
//! mathematically untouched — so a multi-tone signal's crest factor (and
//! therefore true peak) shifts while the timbre stays identical.
//!
//! Purpose per the master plan: a **peak headroom optimization** tool placed
//! before the limiter. It is NOT an anti-detection / fingerprint-avoidance
//! device and the UI must never present it as one.
//!
//! `strength` ∈ [0, 1] selects how many of the 8 sections are active
//! (0 → bypass, 1 → all 8). Section placement is deterministic so the same
//! input always yields the same output.

/// Number of cascaded all-pass sections at full strength.
pub const SECTION_COUNT: usize = 8;
/// Pole-frequency sweep bounds (Hz), geometric spacing.
const F_LO: f64 = 60.0;
const F_HI: f64 = 9000.0;
/// Section Q — under-damped enough to rotate phase meaningfully per stage.
const Q: f64 = 0.7;

/// One 2nd-order all-pass section (RBJ cookbook, Direct Form I).
struct AllPass {
    a1: f64,
    a2: f64,
    b0: f64,
    b1: f64,
    b2: f64,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl AllPass {
    fn design(sr: f64, f0: f64, q: f64) -> AllPass {
        let w0 = 2.0 * std::f64::consts::PI * f0 / sr;
        let (s, c) = w0.sin_cos();
        let alpha = s / (2.0 * q);
        let a0 = 1.0 + alpha;
        AllPass {
            b0: (1.0 - alpha) / a0,
            b1: (-2.0 * c) / a0,
            b2: 1.0,
            a1: (-2.0 * c) / a0,
            a2: (1.0 - alpha) / a0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

/// Number of sections engaged for a given strength.
pub fn sections_for(strength: f64) -> usize {
    ((strength.clamp(0.0, 1.0) * SECTION_COUNT as f64).round() as usize).min(SECTION_COUNT)
}

/// Apply phase rotation in place. Returns the number of sections engaged.
/// Mono and stereo channels are processed with IDENTICAL section settings
/// (fresh state per channel) so the stereo image is preserved.
pub fn apply_phase_rotation(left: &mut [f32], right: Option<&mut [f32]>, sr: f64, strength: f64) -> usize {
    let n = sections_for(strength);
    if n == 0 || left.is_empty() {
        return 0;
    }
    let mk_chain = || -> Vec<AllPass> {
        (0..n)
            .map(|k| {
                let f0 = F_LO * (F_HI / F_LO).powf(k as f64 / (SECTION_COUNT - 1) as f64);
                AllPass::design(sr, f0, Q)
            })
            .collect()
    };
    let mut lchain = mk_chain();
    for s in left.iter_mut() {
        let mut v = *s as f64;
        for f in lchain.iter_mut() {
            v = f.process(v);
        }
        *s = v as f32;
    }
    if let Some(r) = right {
        let mut rchain = mk_chain();
        for s in r.iter_mut() {
            let mut v = *s as f64;
            for f in rchain.iter_mut() {
                v = f.process(v);
            }
            *s = v as f32;
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f64 = 44100.0;

    #[test]
    fn zero_strength_is_bypass() {
        let mut l: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let before = l.clone();
        assert_eq!(apply_phase_rotation(&mut l, None, SR, 0.0), 0);
        assert_eq!(l, before);
    }

    #[test]
    fn strength_maps_to_sections() {
        assert_eq!(sections_for(0.0), 0);
        assert_eq!(sections_for(0.5), 4);
        assert_eq!(sections_for(1.0), SECTION_COUNT);
        assert_eq!(sections_for(2.0), SECTION_COUNT, "clamped");
        assert_eq!(sections_for(-1.0), 0, "clamped");
    }

    #[test]
    fn steady_state_sine_rms_is_preserved() {
        // All-pass leaves per-frequency magnitude untouched → the RMS of a
        // settled sine must match the input RMS to high precision.
        let n = (SR * 0.2) as usize;
        let skip = (SR * 0.05) as usize;
        let mk = |freq: f64| -> Vec<f32> {
            (0..n)
                .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / SR).sin() as f32 * 0.5)
                .collect()
        };
        for &freq in &[100.0f64, 440.0, 3000.0] {
            let mut x = mk(freq);
            apply_phase_rotation(&mut x, None, SR, 1.0);
            let rms = |s: &[f32]| {
                (s.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / s.len() as f64).sqrt()
            };
            let (ri, ro) = (rms(&mk(freq)[skip..]), rms(&x[skip..]));
            assert!(
                (ri - ro).abs() / ri < 1e-3,
                "{freq} Hz RMS changed: {ri} → {ro}"
            );
        }
    }

    #[test]
    fn output_differs_but_stays_bounded() {
        // Broadband signal: phase must change (≠ input) while samples stay
        // within a sane bound (all-pass magnitude is exactly 1 per frequency,
        // so transient overshoot beyond ~2× input peak cannot occur).
        let mut l: Vec<f32> = (0..4410)
            .map(|i| {
                let t = i as f64 / SR;
                (0.4 * (2.0 * std::f64::consts::PI * 220.0 * t).sin()
                    + 0.3 * (2.0 * std::f64::consts::PI * 1310.0 * t).sin()
                    + 0.2 * (2.0 * std::f64::consts::PI * 5230.0 * t).sin()) as f32
            })
            .collect();
        let before = l.clone();
        let n = apply_phase_rotation(&mut l, None, SR, 1.0);
        assert_eq!(n, SECTION_COUNT);
        assert!(
            l.iter().zip(before.iter()).any(|(a, b)| (a - b).abs() > 1e-3),
            "phase rotation must alter the waveform"
        );
        assert!(l.iter().all(|&v| v.abs() < 1.0), "stays within full scale");
    }

    #[test]
    fn stereo_planes_use_identical_sections() {
        // Identical L/R → still identical after rotation (no image smear).
        let mk = || -> Vec<f32> {
            (0..4410)
                .map(|i| (2.0 * std::f64::consts::PI * 330.0 * i as f64 / SR).sin() as f32 * 0.5)
                .collect()
        };
        let (mut l, mut r) = (mk(), mk());
        apply_phase_rotation(&mut l, Some(&mut r), SR, 0.75);
        for (a, b) in l.iter().zip(r.iter()) {
            assert!((a - b).abs() < 1e-6, "correlated channels must stay correlated");
        }
    }
}
