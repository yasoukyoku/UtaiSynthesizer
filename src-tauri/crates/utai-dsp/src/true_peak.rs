//! True-peak (ITU-R BS.1770-4) measurement.
//!
//! "True peak" = the peak of the reconstructed *analogue* waveform, which can
//! exceed the highest PCM sample because the DAC interpolates between samples.
//! The standard mandates 4× oversampling before taking max(|x|).
//!
//! We upsample by 4× using a windowed-sinc (Kaiser) low-pass filter, then take
//! the per-channel max. Returns dBTP = 20·log10(true_peak_linear).

use crate::vr; // reuse the existing kaiser-windowed sinc resampler

/// Measure true peak in dBTP for an interleaved or per-channel buffer.
/// `left` and `right` are mono channel buffers; pass `right = None` for mono.
/// Returns (left_dbtp, right_dbtp).
pub fn measure_true_peak_db(left: &[f32], right: Option<&[f32]>) -> (f64, f64) {
    let peak = |ch: &[f32]| -> f64 {
        if ch.is_empty() {
            return f64::NEG_INFINITY;
        }
        let up = upsample_4x(ch);
        let mut m = 0.0f32;
        for &s in &up {
            let a = s.abs();
            if a > m {
                m = a;
            }
        }
        if m < 1e-10 {
            return f64::NEG_INFINITY;
        }
        20.0 * (m as f64).log10()
    };
    let l = peak(left);
    let r = right.map(peak).unwrap_or(l);
    (l, r)
}

/// 4× upsample via the existing `vr::resample_poly_f32` (kaiser-windowed sinc
/// polyphase, scipy-equivalent, proven in this codebase).
fn upsample_4x(x: &[f32]) -> Vec<f32> {
    if x.is_empty() {
        return vec![];
    }
    vr::resample_poly_f32(x, 4, 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_scale_sine_true_peak_near_zero() {
        // A full-scale sine sampled well below Nyquist: true peak ≈ 0 dBTP.
        let sr = 44100u32;
        let s: Vec<f32> = (0..sr as usize)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / sr as f32).sin())
            .collect();
        let (l, _) = measure_true_peak_db(&s, None);
        // Sine peaks at 1.0 → 0 dBTP. Allow small interpolation error.
        assert!(l > -0.5, "full-scale sine true peak {l} should be near 0 dBTP");
        assert!(l < 0.1, "should not exceed 0 dBTP");
    }

    #[test]
    fn silent_is_neg_infinity() {
        let s = vec![0.0f32; 1000];
        let (l, _) = measure_true_peak_db(&s, None);
        assert!(l.is_infinite() && l < 0.0);
    }

    #[test]
    fn inter_sample_peak_exceeds_sample_peak() {
        // Construct a signal whose sample peak is < 1 but whose true peak > sample peak.
        // A 1/4-Nyquist sine (fs/4) has samples at 0, 1, 0, -1 → true peak ≈ sample peak.
        // Instead, two closely-spaced impulses create inter-sample overshoot.
        let mut s = vec![0.0f32; 64];
        s[30] = 0.9;
        s[31] = 0.9;
        let sample_peak = s.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let (tpl, _) = measure_true_peak_db(&s, None);
        let tp_linear = 10.0_f64.powf(tpl / 20.0) as f32;
        assert!(
            tp_linear >= sample_peak,
            "true peak {tp_linear} must be ≥ sample peak {sample_peak}"
        );
    }
}
