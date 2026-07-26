//! utai-stretch — offline time-stretch (tempo change, pitch preserved) via the vendored
//! Signalsmith Stretch C++ library (v1.3.2 + signalsmith-linear 0.3.1, MIT;
//! vendor/signalsmith-stretch/VENDOR.md records the exact upstream commits). Chosen over WSOLA
//! per the S59 engine research: phase-vocoder variant with energy-weighted phase blending,
//! quality on full mixes ≈ Rubber Band R3 tier, official sweet spot 0.75–1.5× — exactly the
//! Tempo Slider's range. The Tempo Slider feeds it whole sources/stems; output length is exactly
//! round(input_len × factor) per channel (the upstream `exact()` recipe).

/// Engine-native formant control for one stretch call. `semitones` places the formant
/// envelope relative to the INPUT spectrum (0 = keep the source timbre under any transpose).
/// `base_hz` optionally feeds the engine the KNOWN fundamental instead of its per-block
/// auto-detector: the detector chases noise on voiceless consonants and the resulting
/// per-block correction jitter is audible as pops (S82, ear-confirmed on real renders) —
/// pass the fed f0's median whenever the caller knows it (the vocal inverse does; the
/// instrumental Signalsmith node does not and leaves the detector on).
#[derive(Debug, Clone, Copy)]
pub struct FormantPin {
    pub semitones: f64,
    pub base_hz: Option<f64>,
}

/// `time_factor` = output duration / input duration (>1 = slower/longer).
/// `transpose_semitones` = spectral-domain pitch shift (0 = pitch unchanged); tonality-aware
/// (~8 kHz limit inside the shim) so full mixes keep natural highs.
/// `formant` = `None` → formants follow the transpose (the classic full-spectrum shift, zero
/// extra cost); `Some(pin)` → engine-native `setFormantSemitones` with pitch compensation
/// (see FormantPin). Consumers: the Signalsmith node (follow/offset knobs) and the
/// range-extension inverse (κ policy); the Tempo Slider passes `None`.
pub fn stretch_interleaved(
    input: &[f32],
    channels: usize,
    sample_rate: u32,
    time_factor: f64,
    transpose_semitones: f64,
    formant: Option<FormantPin>,
) -> Result<Vec<f32>, String> {
    if channels == 0 || input.len() % channels != 0 {
        return Err("STRETCH_BAD_INPUT".into());
    }
    if !(time_factor.is_finite() && time_factor > 0.0) {
        return Err("STRETCH_RATIO_RANGE".into());
    }
    if !transpose_semitones.is_finite() {
        return Err("TRANSPOSE_RANGE".into());
    }
    if let Some(pin) = formant {
        if !pin.semitones.is_finite() {
            return Err("TRANSPOSE_FORMANT_RANGE".into());
        }
        if let Some(b) = pin.base_hz {
            if !(b.is_finite() && b > 0.0) {
                return Err("TRANSPOSE_FORMANT_RANGE".into());
            }
        }
    }
    let in_samples = input.len() / channels;
    if in_samples == 0 {
        return Ok(Vec::new());
    }
    let out_samples = ((in_samples as f64) * time_factor).round().max(1.0) as usize;
    // i32 FFI boundary: a colossal input × factor (hours of audio at 4×) would wrap `as i32`
    // into a small positive count and come back rc=0 with mostly-zero audio (S82 review) —
    // refuse loudly instead. Same CODE as the factor validation: it IS a ratio/size problem.
    if in_samples > i32::MAX as usize || out_samples > i32::MAX as usize {
        return Err("STRETCH_RATIO_RANGE".into());
    }
    let mut output = vec![0.0f32; out_samples * channels];
    let rc = unsafe {
        utai_stretch_exact(
            input.as_ptr(),
            in_samples as i32,
            channels as i32,
            sample_rate as f32,
            time_factor,
            transpose_semitones,
            formant.map_or(0.0, |p| p.semitones),
            i32::from(formant.is_some()),
            formant.and_then(|p| p.base_hz).unwrap_or(0.0),
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
        transpose_semitones: f64,
        formant_semitones: f64,
        formant_active: i32,
        formant_base_hz: f64,
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

    fn mid(x: &[f32]) -> &[f32] {
        &x[x.len() / 4..x.len() * 3 / 4]
    }

    /// Spectral centroid via a coarse 40 Hz-step DFT up to 10 kHz — no FFT dependency, just
    /// enough resolution to rank "brighter vs darker" for the formant test.
    fn centroid(x: &[f32], sr: u32) -> f32 {
        let n = x.len().min(8192);
        let x = &x[..n];
        let (mut num, mut den) = (0.0f64, 0.0f64);
        let mut f = 40.0f32;
        while f < 10_000.0 {
            let w = 2.0 * std::f64::consts::PI * f as f64 / sr as f64;
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (i, s) in x.iter().enumerate() {
                let ph = w * i as f64;
                re += *s as f64 * ph.cos();
                im -= *s as f64 * ph.sin();
            }
            let mag = (re * re + im * im).sqrt();
            num += mag * f as f64;
            den += mag;
            f += 40.0;
        }
        (num / den.max(1e-9)) as f32
    }

    #[test]
    fn exact_output_length() {
        let x = sine_stereo(440.0, 2 * SR as usize);
        for factor in [0.75f64, 1.0, 1.25, 1.5] {
            let y = stretch_interleaved(&x, 2, SR, factor, 0.0, None).expect("stretch");
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
            let y = stretch_interleaved(&x, 2, SR, factor, 0.0, None).expect("stretch");
            let mono = mono_left(&y);
            // measure well inside the output (skip edges)
            let p = est_period(mid(&mono), expected - 20, expected + 20);
            assert!(
                (p as i32 - expected as i32).abs() <= 3,
                "factor={factor} moved pitch: period {p} vs {expected}"
            );
        }
    }

    #[test]
    fn transpose_shifts_pitch_and_keeps_length() {
        let f0 = 220.0f32;
        let x = sine_stereo(f0, 3 * SR as usize);
        for semis in [-5.0f64, 4.0, 12.0] {
            let y = stretch_interleaved(&x, 2, SR, 1.0, semis, None).expect("transpose");
            // pure transpose: sample-exact same length
            assert_eq!(y.len(), x.len(), "semis={semis}");
            let shifted = f0 * (2.0f32).powf(semis as f32 / 12.0);
            let expected = (SR as f32 / shifted).round() as usize;
            let mono = mono_left(&y);
            let p = est_period(mid(&mono), expected.saturating_sub(20).max(8), expected + 20);
            // spectral shift tolerance: within ~1.5% of the target period
            let tol = (expected as f32 * 0.015).ceil() as i32 + 1;
            assert!(
                (p as i32 - expected as i32).abs() <= tol,
                "semis={semis} period {p} vs {expected} (tol {tol})"
            );
        }
    }

    #[test]
    fn formant_preservation_keeps_the_source_timbre() {
        // Harmonic-rich tone (1/k rolloff, 20 harmonics of 220 Hz): a plain +12 st transpose
        // drags the whole envelope up an octave (much brighter); with formant=Some(0) the
        // engine re-imposes the source envelope, so the centroid must stay near the original.
        let f0 = 220.0f32;
        let n = 2 * SR as usize;
        let mut x = Vec::with_capacity(n * 2);
        for i in 0..n {
            let t = i as f32 / SR as f32;
            let mut s = 0.0f32;
            for k in 1..=20u32 {
                s += (2.0 * PI * f0 * k as f32 * t).sin() / k as f32;
            }
            s *= 0.2;
            x.push(s);
            x.push(s);
        }
        let follow = stretch_interleaved(&x, 2, SR, 1.0, 12.0, None).expect("follow");
        let pin = |base_hz: Option<f64>| FormantPin { semitones: 0.0, base_hz };
        let pres = stretch_interleaved(&x, 2, SR, 1.0, 12.0, Some(pin(None))).expect("preserve");
        // known-f0 hint (the S82 anti-pop path) must not break the formant contract
        let pres_base =
            stretch_interleaved(&x, 2, SR, 1.0, 12.0, Some(pin(Some(f0 as f64)))).expect("base");
        // all must still transpose the fundamental up an octave
        let expected = (SR as f32 / (f0 * 2.0)).round() as usize;
        for (name, y) in [("follow", &follow), ("preserved", &pres), ("preserved+base", &pres_base)] {
            let mono = mono_left(y);
            let p = est_period(mid(&mono), expected.saturating_sub(15).max(8), expected + 15);
            assert!(
                (p as i32 - expected as i32).abs() <= 3,
                "{name}: period {p} vs {expected}"
            );
        }
        let c_orig = centroid(mid(&mono_left(&x)), SR);
        let c_follow = centroid(mid(&mono_left(&follow)), SR);
        assert!(
            c_follow > c_orig * 1.2,
            "transpose alone should brighten: orig {c_orig} follow {c_follow}"
        );
        for (name, y) in [("preserved", &pres), ("preserved+base", &pres_base)] {
            let c_pres = centroid(mid(&mono_left(y)), SR);
            assert!(
                (c_pres - c_orig).abs() < (c_follow - c_orig).abs() * 0.8,
                "{name} did not hold the envelope: orig {c_orig} follow {c_follow} got {c_pres}"
            );
        }
    }

    #[test]
    fn energy_is_sane() {
        let x = sine_stereo(330.0, 2 * SR as usize);
        let y = stretch_interleaved(&x, 2, SR, 1.2, 0.0, None).expect("stretch");
        let ex = x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / x.len() as f64;
        let ey = y.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / y.len() as f64;
        assert!(ey > ex * 0.25 && ey < ex * 4.0, "mean energy in={ex} out={ey}");
    }

    #[test]
    fn survives_inputs_shorter_than_engine_latency() {
        // exact() refuses inputs shorter than its seek length (~130 ms at 44.1k); the shim
        // front-pads those with silence and trims the stretched pad back off, so a 50 ms clip
        // must still return the exact caller-computed length with finite samples.
        for n in [64usize, 441, 2205] {
            let x = sine_stereo(440.0, n);
            for factor in [0.75f64, 1.0, 1.4] {
                let y = stretch_interleaved(&x, 2, SR, factor, 0.0, None).expect("short stretch");
                assert_eq!(y.len(), ((n as f64) * factor).round().max(1.0) as usize * 2);
                assert!(y.iter().all(|v| v.is_finite()));
            }
        }
    }

    #[test]
    fn rejects_bad_args() {
        assert!(stretch_interleaved(&[0.0; 10], 3, SR, 1.2, 0.0, None).is_err()); // not divisible
        assert!(stretch_interleaved(&[0.0; 8], 2, SR, f64::NAN, 0.0, None).is_err());
        assert!(stretch_interleaved(&[0.0; 8], 2, SR, 0.0, 0.0, None).is_err());
        assert!(stretch_interleaved(&[0.0; 8], 2, SR, 1.0, f64::NAN, None).is_err());
        let bad_semi = FormantPin { semitones: f64::NAN, base_hz: None };
        assert!(stretch_interleaved(&[0.0; 8], 2, SR, 1.0, 0.0, Some(bad_semi)).is_err());
        let bad_base = FormantPin { semitones: 0.0, base_hz: Some(-5.0) };
        assert!(stretch_interleaved(&[0.0; 8], 2, SR, 1.0, 0.0, Some(bad_base)).is_err());
    }
}
