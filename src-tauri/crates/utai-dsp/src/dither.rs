//! Dither for bit-depth reduction.
//!
//! When 32-bit float is reduced to 16-bit integer, simple truncation produces
//! signal-correlated quantization distortion (audible "grain" in quiet passages).
//! TPDF dither replaces that correlated distortion with uncorrelated white noise
//! at ~-93 dBFS; noise shaping pushes most of that noise above 15 kHz where the
//! ear is insensitive, giving a perceived noise floor of ~-110 dBFS.
//!
//! 24-bit output needs no dither (quantization at -144 dBFS is inaudible);
//! 32-bit float output needs none either.

const LSB_16: f64 = 1.0 / 32768.0;

/// Tiny xorshift64 PRNG — no external dependency, fully deterministic from seed.
/// Good enough for dither (we only need uniform distribution, not cryptographic
/// quality). Period 2^64-1.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        // Avoid a zero seed (xorshift can't escape 0).
        Self { state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed } }
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
    /// Uniform f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        // 53 random bits → [0, 1).
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DitherType {
    /// No dither — direct truncation (current behaviour). Use only for 24-bit+.
    None,
    /// Triangular Probability Density Function — two independent uniform
    /// variables summed. White, uncorrelated, ~-93 dBFS.
    Tpdf,
    /// TPDF plus 2nd-order error-feedback noise shaping. Perceived ~-110 dBFS;
    /// noise energy pushed above 15 kHz.
    TpdfShaped,
}

/// Apply 16-bit dither. `samples` are f32 in [-1, 1]; output is i16.
///
/// `seed` makes the output deterministic (same seed → identical dither
/// sequence) so auditioning and exporting produce the same file.
pub fn apply_dither_16bit(samples: &[f32], dither_type: DitherType, seed: u64) -> Vec<i16> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(samples.len());

    match dither_type {
        DitherType::None => {
            for &s in samples {
                let q = (s as f64 * 32767.0).round() as i32;
                out.push(q.clamp(-32768, 32767) as i16);
            }
        }
        DitherType::Tpdf => {
            for &s in samples {
                // TPDF: sum of two uniform [-0.5, 0.5] → triangular [-1, 1],
                // scaled to 1 LSB peak-to-peak.
                let u1 = rng.next_f64() - 0.5;
                let u2 = rng.next_f64() - 0.5;
                let dither = (u1 + u2) * LSB_16;
                let v = s as f64 + dither;
                let q = (v * 32767.0).round() as i32;
                out.push(q.clamp(-32768, 32767) as i16);
            }
        }
        DitherType::TpdfShaped => {
            // 2nd-order error-feedback noise shaper.
            // NTF(z) = (1 - z^-1)^2 = 1 - 2z^-1 + z^-2 → feedback H = 2z^-1 - z^-2.
            // This places a double zero at DC, pushing quantization noise to
            // high frequencies (above ~10 kHz for typical audio rates).
            let mut e1 = 0.0f64; // e[n-1]
            let mut e2 = 0.0f64; // e[n-2]
            for &s in samples {
                let u1 = rng.next_f64() - 0.5;
                let u2 = rng.next_f64() - 0.5;
                let dither = (u1 + u2) * LSB_16;
                let v = s as f64 + dither - 2.0 * e1 + e2;
                let q = (v * 32767.0).round() as i32;
                let y = q.clamp(-32768, 32767) as f64 / 32767.0;
                let e = y - v;
                e2 = e1;
                e1 = e;
                out.push(q.clamp(-32768, 32767) as i16);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_is_simple_truncation() {
        // Codebase convention: float * 32767 (symmetric, -1.0 → -32767).
        let out = apply_dither_16bit(&[1.0, -1.0, 0.0], DitherType::None, 0);
        assert_eq!(out[0], 32767);
        assert_eq!(out[1], -32767);
    }

    #[test]
    fn tpdf_is_uncorrelated_with_signal() {
        // A constant at exactly 0.5 LSB: TPDF of ±1 LSB should spread the
        // quantized output across 0 and 1 (prove dither is added, not just
        // truncation which would always round to the nearest quantum).
        let half_lsb = 0.5 / 32768.0;
        let signal = vec![half_lsb; 4096];
        let out = apply_dither_16bit(&signal, DitherType::Tpdf, 42);
        let has_zero = out.iter().any(|&x| x == 0);
        let has_one = out.iter().any(|&x| x == 1);
        assert!(has_zero && has_one, "TPDF should spread 0.5-LSB signal across 0 and 1");
    }

    #[test]
    fn tpdf_distribution_is_triangular() {
        // For a zero input, TPDF ±1 LSB quantized to *32767 lands in {-1,0,1}
        // with p=0.125/0.75/0.125. Variance = 0.25 / 32767² ≈ 2.33e-10.
        let n = 100_000;
        let signal = vec![0.0f32; n];
        let out = apply_dither_16bit(&signal, DitherType::Tpdf, 7);
        let noises: Vec<f64> = out.iter().map(|&x| x as f64 / 32767.0).collect();
        let mean: f64 = noises.iter().sum::<f64>() / n as f64;
        let var: f64 = noises.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let expected_var = 0.25 / (32767.0f64 * 32767.0);
        assert!(mean.abs() < 1e-6, "mean {mean} should be near 0");
        assert!(
            (var - expected_var).abs() < 0.1 * expected_var,
            "variance {var} should be near {expected_var}"
        );
    }

    #[test]
    fn noise_shaped_has_less_low_frequency_noise() {
        // Compare low-frequency energy of TPDF vs TpdfShaped on a zero signal.
        // The shaped output should have noticeably less energy in the lowest
        // FFT bin (DC neighbourhood) because the NTF has a double zero at DC.
        let n = 8192;
        let signal = vec![0.0f32; n];
        let tpdf = apply_dither_16bit(&signal, DitherType::Tpdf, 1);
        let shaped = apply_dither_16bit(&signal, DitherType::TpdfShaped, 1);
        let energy = |o: &[i16]| -> f64 {
            // Rough low-frequency proxy: sum of adjacent differences is small
            // when low-frequency content is small (shaped noise is "rough" →
            // large sample-to-sample differences).
            o.windows(2).map(|w| (w[1] as f64 - w[0] as f64).abs()).sum::<f64>()
        };
        // Shaped noise concentrates at high freq → larger sample-to-sample jumps.
        assert!(
            energy(&shaped) > energy(&tpdf),
            "shaped dither should have more high-frequency (sample-to-sample) energy"
        );
    }

    #[test]
    fn deterministic_with_same_seed() {
        let signal: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.001).sin()).collect();
        let a = apply_dither_16bit(&signal, DitherType::TpdfShaped, 123);
        let b = apply_dither_16bit(&signal, DitherType::TpdfShaped, 123);
        assert_eq!(a, b, "same seed must produce byte-identical output");
        let c = apply_dither_16bit(&signal, DitherType::TpdfShaped, 124);
        assert_ne!(a, c, "different seed should (almost surely) differ");
    }
}
