//! voice_realism.rs — the Phase-7 "vocal realism" micro-texture kernels (master plan §8.7 recipe G
//! steps ①②③; step ④ harmonicityCheck already lives in harmonicity.rs and is wired at the engine
//! layer). Three ADD-ON stages applied to the finished vocal render (post-loop, after every carving
//! stage, before peak normalization), in this fixed order:
//!
//! ① `apply_hf_excitation` — "air": a soft-saturated copy of the ≥8 kHz band mixed back in at a low
//!    level. Adds breathiness/edge without touching the body of the tone. DC passes BIT-EXACT (the
//!    high-pass sees x − x_prev = 0 for constant input), so silent/tail regions are untouched.
//! ② `apply_formant_jitter` — a slow (0.3–1.2 Hz) LFO'd cascade of 5 first-order all-passes. Moves
//!    the poles slightly without changing pitch or level (all-pass = flat magnitude by
//!    construction), which reads as the natural micro-drift of a real vocal tract. The LFO uses the
//!    ABSOLUTE sample index, so a base pass and its donor pass (S147) land on the same LFO phase —
//!    splice-consistent.
//! ③ `synth_breath` — a procedural inhale (band-limited noise + inhale envelope + slow shimmer),
//!    which the caller adds into detected SP-only gaps. Deterministic per seed.
//!
//! All kernels are std-only, length-preserving, and BIT-EXACT no-ops at zero mix/depth, so the
//! production defaults (0 / 0) leave existing renders byte-identical.

use std::f32::consts::TAU;

/// ① Exciter crossover (Hz): the "air" band lives above this. 8 kHz keeps the formant body
/// untouched and only brightens sibilance/breath noise.
const HF_CUTOFF_HZ: f32 = 8000.0;
/// ① tanh drive before saturation — gentle density at mix 5–15% without audible fizz.
const HF_DRIVE: f32 = 3.0;

/// ② All-pass cascade depth (master plan: 4–6 stages).
const JITTER_STAGES: usize = 5;
/// ② Per-stage nominal all-pass coefficient a0 (staggered to decorrelate the pole motion).
const JITTER_A0: [f32; JITTER_STAGES] = [0.55, 0.60, 0.62, 0.65, 0.58];
/// ② Per-stage LFO rate (Hz) — inside the master-plan band 0.3–1.2 Hz, staggered so the stages
/// never sync up into an audible wobble.
const JITTER_LFO_HZ: [f32; JITTER_STAGES] = [0.37, 0.53, 0.71, 0.89, 1.07];
/// ② Per-stage LFO phase (rad), spread over [0, 2π).
const JITTER_PHASE: [f32; JITTER_STAGES] = [0.0, 1.3, 2.1, 4.4, 5.5];
/// ② Max |depth| this kernel accepts. a = a0 + depth·sin must stay < 1 for all-pass stability
/// (max a0 = 0.65 + 0.3 → 0.95). Production clamps at 0.08 anyway; this is the kernel ceiling.
const JITTER_DEPTH_MAX: f32 = 0.3;

/// ③ Inhale band (Hz): rumble filtered below, hiss filtered above.
const BREATH_HP_HZ: f32 = 350.0;
const BREATH_LP_HZ: f32 = 3800.0;
/// ③ Peak amplitude the shaped breath is normalized to — audible under the vocal, never loud.
const BREATH_PEAK: f32 = 0.16;
/// ③ Slow shimmer (breath turbulence is non-stationary).
const BREATH_SHIMMER_HZ: f32 = 4.3;
const BREATH_SHIMMER_DEPTH: f32 = 0.15;
/// ③ Envelope silhouette: smoothstep attack over the first 25%, hold flat to 55%, smoothstep
/// release over the rest — fade-in, body, fade-out, zero clicks at either end.
const BREATH_ATTACK_FRAC: f32 = 0.25;
const BREATH_HOLD_FRAC: f32 = 0.55;

/// ① Mix a soft-saturated copy of the ≥8 kHz band back into `samples` at `mix` level (0 = exact
/// no-op, sane range 0.05–0.15). Length preserved. One-pole RC high-pass + tanh; the filter state
/// is seeded from `samples[0]` so a DC input passes bit-exact.
pub fn apply_hf_excitation(samples: &mut [f32], sample_rate: u32, mix: f32) {
    let mix = if mix.is_finite() { mix.clamp(0.0, 1.0) } else { 0.0 };
    if mix <= 0.0 || samples.is_empty() || sample_rate == 0 {
        return;
    }
    let dt = 1.0 / sample_rate as f32;
    let rc = 1.0 / (TAU * HF_CUTOFF_HZ);
    let alpha = rc / (rc + dt);
    let mut x_prev = samples[0];
    let mut hp_prev = 0.0f32;
    for s in samples.iter_mut() {
        let x = *s;
        let hp = alpha * (hp_prev + x - x_prev);
        let e = (HF_DRIVE * hp).tanh();
        *s = x + mix * e;
        x_prev = x;
        hp_prev = hp;
    }
}

/// ② Cascade `JITTER_STAGES` first-order all-passes whose coefficients wobble slowly with `depth`
/// (0 = exact no-op; sane range 0.03–0.08; kernel ceiling `JITTER_DEPTH_MAX`). y = −a·x + x[n−1] +
/// a·y[n−1] with a = a0 + depth·sin(2π·f_lfo·t + φ) at the ABSOLUTE sample index t — so two renders
/// of the same timeline (base / donor) share one LFO phase. Length and level preserved (all-pass is
/// flat-magnitude by construction), pitch untouched.
pub fn apply_formant_jitter(samples: &mut [f32], sample_rate: u32, depth: f32) {
    let depth = if depth.is_finite() { depth.clamp(0.0, JITTER_DEPTH_MAX) } else { 0.0 };
    if depth <= 0.0 || samples.is_empty() || sample_rate == 0 {
        return;
    }
    let fs = sample_rate as f32;
    let mut x_prev = [0.0f32; JITTER_STAGES];
    let mut y_prev = [0.0f32; JITTER_STAGES];
    x_prev[0] = samples[0];
    for (i, s) in samples.iter_mut().enumerate() {
        let t = i as f32 / fs;
        let mut v = *s;
        for st in 0..JITTER_STAGES {
            let a = JITTER_A0[st]
                + depth * (TAU * JITTER_LFO_HZ[st] * t + JITTER_PHASE[st]).sin();
            let y = -a * v + x_prev[st] + a * y_prev[st];
            x_prev[st] = v;
            y_prev[st] = y;
            v = y;
        }
        *s = v;
    }
}

/// ③ Synthesize one inhale: xorshift64* white noise → 350 Hz HP → 3.8 kHz LP → inhale envelope
/// (smoothstep attack 25%, hold to 55%, smoothstep release) → slow 4.3 Hz shimmer AM → normalize
/// the peak to `BREATH_PEAK`. Deterministic for a given `seed`; length is exactly `len` samples.
pub fn synth_breath(sample_rate: u32, len: usize, seed: u64) -> Vec<f32> {
    if len == 0 || sample_rate == 0 {
        return Vec::new();
    }
    let fs = sample_rate as f32;
    let dt = 1.0 / fs;
    let rc_hp = 1.0 / (TAU * BREATH_HP_HZ);
    let hp_alpha = rc_hp / (rc_hp + dt);
    let rc_lp = 1.0 / (TAU * BREATH_LP_HZ);
    let lp_beta = dt / (rc_lp + dt);

    // xorshift64* — tiny, dependency-free, deterministic. The golden-ratio XOR decorrelates
    // caller-supplied seeds (0 included) so the stream never collapses to all-zeros.
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    if state == 0 {
        state = 0x2545_F491_4F6C_DD1D;
    }
    let mut next_unit = move || -> f32 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let r = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (r >> 11) as f32 / (1u64 << 53) as f32
    };

    // Shimmer phase from the seed's high half — deterministic per breath, varies across gaps.
    let shimmer_phi = ((seed >> 32) as f64 / u32::MAX as f64 * TAU as f64) as f32;

    let mut out = Vec::with_capacity(len);
    let mut x_prev = 0.0f32;
    let mut hp_prev = 0.0f32;
    let mut lp = 0.0f32;
    for i in 0..len {
        let n = next_unit() * 2.0 - 1.0;
        let hp = hp_alpha * (hp_prev + n - x_prev);
        lp += lp_beta * (hp - lp);
        let t = i as f32 / fs;
        let tf = i as f32 / len as f32;
        let env = if tf < BREATH_ATTACK_FRAC {
            smoothstep(tf / BREATH_ATTACK_FRAC)
        } else if tf <= BREATH_HOLD_FRAC {
            1.0
        } else {
            smoothstep(1.0 - (tf - BREATH_HOLD_FRAC) / (1.0 - BREATH_HOLD_FRAC))
        };
        let am = 1.0 + BREATH_SHIMMER_DEPTH * (TAU * BREATH_SHIMMER_HZ * t + shimmer_phi).sin();
        out.push(lp * env * am);
        x_prev = n;
        hp_prev = hp;
    }

    // Normalize the body to BREATH_PEAK (guard the degenerate all-zero case).
    let peak = out.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if peak > 1e-9 {
        let scale = BREATH_PEAK / peak;
        for v in out.iter_mut() {
            *v *= scale;
        }
    }
    out
}

/// Element-wise `dst += src` (truncating at the shorter length) — the overlay primitive the caller
/// uses to drop a synthesized breath into a detected SP-only gap.
pub fn add_into(dst: &mut [f32], src: &[f32]) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d += s;
    }
}

fn smoothstep(s: f32) -> f32 {
    let s = s.clamp(0.0, 1.0);
    s * s * (3.0 - 2.0 * s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f32, sr: f32, n: usize) -> Vec<f32> {
        (0..n).map(|i| (TAU * freq * i as f32 / sr).sin()).collect()
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
    }

    #[test]
    fn zero_mix_and_zero_depth_are_bit_exact_noops() {
        let sr = 44100.0;
        let x: Vec<f32> = (0..8192)
            .map(|i| {
                if i < 4096 {
                    (TAU * 440.0 * i as f32 / sr).sin()
                } else {
                    (TAU * 6000.0 * i as f32 / sr).sin()
                }
            })
            .collect();
        let mut a = x.clone();
        apply_hf_excitation(&mut a, 44100, 0.0);
        assert_eq!(a, x, "mix=0 must be a bit-exact no-op");
        let mut b = x.clone();
        apply_formant_jitter(&mut b, 44100, 0.0);
        assert_eq!(b, x, "depth=0 must be a bit-exact no-op");
    }

    #[test]
    fn hf_dc_input_is_bit_exact() {
        let mut x = vec![0.3f32; 1024];
        apply_hf_excitation(&mut x, 44100, 0.15);
        assert!(x.iter().all(|&v| v == 0.3), "DC must pass through bit-exact");
    }

    #[test]
    fn hf_exciter_brightens_hf_leaves_lf() {
        let sr = 44100.0;
        let hf = sine(12000.0, sr, 16384);
        let mut y = hf.clone();
        apply_hf_excitation(&mut y, 44100, 0.15);
        assert!(y.iter().all(|v| v.is_finite()));
        assert!(
            rms(&y) > rms(&hf) * 1.05,
            "HF band must gain energy: {} -> {}",
            rms(&hf),
            rms(&y)
        );

        let lf = sine(220.0, sr, 16384);
        let mut z = lf.clone();
        apply_hf_excitation(&mut z, 44100, 0.15);
        let ratio = rms(&z) / rms(&lf);
        assert!(
            (ratio - 1.0).abs() < 0.05,
            "LF band must stay nearly untouched: ratio={ratio}"
        );
    }

    #[test]
    fn jitter_is_level_preserving_and_finite() {
        let sr = 44100.0;
        let x = sine(440.0, sr, 16384);
        let mut y = x.clone();
        apply_formant_jitter(&mut y, 44100, 0.05);
        assert_eq!(y.len(), x.len());
        assert!(y.iter().all(|v| v.is_finite()));
        let ratio = rms(&y) / rms(&x);
        assert!(
            (ratio - 1.0).abs() < 0.05,
            "all-pass must preserve level: ratio={ratio}"
        );
    }

    #[test]
    fn jitter_dc_converges_to_input() {
        let mut x = vec![0.3f32; 4800];
        apply_formant_jitter(&mut x, 44100, 0.05);
        let err: f32 = x[4096..].iter().map(|v| (v - 0.3).abs()).fold(0.0, f32::max);
        assert!(err < 1e-4, "DC should converge to the input after warm-up, err={err}");
    }

    #[test]
    fn breath_length_peak_and_fades() {
        let (sr, len) = (44100u32, 22050usize);
        let b = synth_breath(sr, len, 7);
        assert_eq!(b.len(), len);
        assert!(b.iter().all(|v| v.is_finite()));
        assert_eq!(b[0], 0.0, "attack must start from silence");
        let peak = b.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        assert!((peak - BREATH_PEAK).abs() < 1e-3, "peak={peak}");
        assert!(
            b[len - 1].abs() < 0.01 * BREATH_PEAK,
            "release must end near silence, last={}",
            b[len - 1]
        );
        assert!(rms(&b) > 0.001, "body must not be silence");
    }

    #[test]
    fn breath_is_deterministic_per_seed() {
        let a = synth_breath(44100, 4096, 42);
        let b = synth_breath(44100, 4096, 42);
        assert_eq!(a, b);
        let c = synth_breath(44100, 4096, 43);
        assert!(a.iter().zip(&c).any(|(x, y)| x != y), "different seeds must differ");
        assert!(synth_breath(44100, 0, 1).is_empty());
        assert_eq!(synth_breath(44100, 1, 1), vec![0.0]);
    }

    #[test]
    fn add_into_sums_elementwise() {
        let mut dst = vec![1.0, 2.0, 3.0];
        add_into(&mut dst, &[10.0, 20.0, 30.0, 40.0]);
        assert_eq!(dst, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn full_realism_chain_stays_bounded() {
        let sr = 44100.0;
        let mut x: Vec<f32> = (0..44100)
            .map(|i| {
                let t = i as f32 / sr;
                0.3 * (TAU * 220.0 * t).sin() + 0.2 * (TAU * 880.0 * t).sin()
            })
            .collect();
        apply_hf_excitation(&mut x, 44100, 0.10);
        apply_formant_jitter(&mut x, 44100, 0.05);
        let breath = synth_breath(44100, x.len(), 12345);
        add_into(&mut x, &breath);
        assert!(x.iter().all(|v| v.is_finite()));
        let peak = x.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        assert!(peak < 1.0, "chain output must stay bounded, peak={peak}");
    }
}
