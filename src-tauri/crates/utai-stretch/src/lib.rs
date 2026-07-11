//! utai-stretch — offline time-stretch (tempo change, pitch preserved) via the vendored
//! Signalsmith Stretch C++ library (MIT; vendor/signalsmith-stretch/). Chosen over WSOLA per
//! the S59 engine research: phase-vocoder variant with energy-weighted phase blending, quality
//! on full mixes ≈ Rubber Band R3 tier, official sweet spot 0.75–1.5× — exactly the Tempo
//! Slider's range. The Tempo Slider feeds it whole sources/stems; output length is exactly
//! round(input_len × factor) per channel (the upstream exact-length recipe).

/// `time_factor` = output duration / input duration (>1 = slower/longer). Pitch is unchanged.
pub fn stretch_interleaved(
    input: &[f32],
    channels: usize,
    sample_rate: u32,
    time_factor: f64,
) -> Result<Vec<f32>, String> {
    if channels == 0 || input.len() % channels != 0 {
        return Err("STRETCH_BAD_INPUT".into());
    }
    if !(time_factor.is_finite() && time_factor > 0.0) {
        return Err("STRETCH_RATIO_RANGE".into());
    }
    let in_samples = input.len() / channels;
    if in_samples == 0 {
        return Ok(Vec::new());
    }
    let out_samples = ((in_samples as f64) * time_factor).round().max(1.0) as usize;
    let mut output = vec![0.0f32; out_samples * channels];
    let rc = unsafe {
        utai_stretch_exact(
            input.as_ptr(),
            in_samples as i32,
            channels as i32,
            sample_rate as f32,
            time_factor,
            output.as_mut_ptr(),
            out_samples as i32,
        )
    };
    if rc != 0 {
        return Err(format!("STRETCH_ENGINE_FAILED: {rc}"));
    }
    Ok(output)
}

extern "C" {
    fn utai_stretch_exact(
        input: *const f32,
        in_samples: i32,
        channels: i32,
        sample_rate: f32,
        time_factor: f64,
        output: *mut f32,
        out_samples: i32,
    ) -> i32;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const SR: u32 = 44100;

    fn sine_stereo(freq: f32, n: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(n * 2);
        for i in 0..n {
            let s = (2.0 * PI * freq * i as f32 / SR as f32).sin() * 0.5;
            v.push(s); // L
            v.push(s * 0.8); // R (correlated but distinct)
        }
        v
    }

    /// Autocorrelation fundamental period over a mono slice (mirror of formant.rs's checker).
    fn est_period(x: &[f32], min_lag: usize, max_lag: usize) -> usize {
        let mut best = min_lag;
        let mut best_val = f32::MIN;
        for lag in min_lag..=max_lag.min(x.len() - 1) {
            let mut acc = 0.0f32;
            for i in 0..x.len() - lag {
                acc += x[i] * x[i + lag];
            }
            if acc > best_val {
                best_val = acc;
                best = lag;
            }
        }
        best
    }

    fn mono_left(inter: &[f32]) -> Vec<f32> {
        inter.chunks_exact(2).map(|f| f[0]).collect()
    }

    #[test]
    fn exact_output_length() {
        let x = sine_stereo(440.0, 2 * SR as usize);
        for factor in [0.75f64, 1.0, 1.25, 1.5] {
            let y = stretch_interleaved(&x, 2, SR, factor).expect("stretch");
            let expected = ((2 * SR as usize) as f64 * factor).round() as usize * 2;
            assert_eq!(y.len(), expected, "factor={factor}");
            assert!(y.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn preserves_pitch_while_stretching() {
        let f0 = 220.0f32;
        let x = sine_stereo(f0, 3 * SR as usize);
        let expected = (SR as f32 / f0).round() as usize; // ~200 samples
        for factor in [0.8f64, 1.3] {
            let y = stretch_interleaved(&x, 2, SR, factor).expect("stretch");
            let mono = mono_left(&y);
            // measure well inside the output (skip edges)
            let mid = &mono[mono.len() / 4..mono.len() * 3 / 4];
            let p = est_period(mid, expected - 20, expected + 20);
            assert!(
                (p as i32 - expected as i32).abs() <= 3,
                "factor={factor} moved pitch: period {p} vs {expected}"
            );
        }
    }

    #[test]
    fn energy_is_sane() {
        let x = sine_stereo(330.0, 2 * SR as usize);
        let y = stretch_interleaved(&x, 2, SR, 1.2).expect("stretch");
        let ex = x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / x.len() as f64;
        let ey = y.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / y.len() as f64;
        assert!(ey > ex * 0.25 && ey < ex * 4.0, "mean energy in={ex} out={ey}");
    }

    #[test]
    fn survives_inputs_shorter_than_engine_latency() {
        // The engine's output latency at 44.1k is ~a few thousand samples; a 50ms clip stretched
        // DOWN yields out_samples < out_lat — the fold-back loop must stay in bounds (audit MAJOR).
        for n in [64usize, 441, 2205] {
            let x = sine_stereo(440.0, n);
            for factor in [0.75f64, 1.0, 1.4] {
                let y = stretch_interleaved(&x, 2, SR, factor).expect("short stretch");
                assert_eq!(y.len(), ((n as f64) * factor).round().max(1.0) as usize * 2);
                assert!(y.iter().all(|v| v.is_finite()));
            }
        }
    }

    #[test]
    fn rejects_bad_args() {
        assert!(stretch_interleaved(&[0.0; 10], 3, SR, 1.2).is_err()); // not divisible
        assert!(stretch_interleaved(&[0.0; 8], 2, SR, f64::NAN).is_err());
        assert!(stretch_interleaved(&[0.0; 8], 2, SR, 0.0).is_err());
    }
}
