//! TD-PSOLA pitch shifting guided by a KNOWN f0 track (S60-2 音域扩展的反变换引擎).
//!
//! Why TD-PSOLA: the S41 augmentation engine-selection battle (4 rounds of A/B + literature)
//! established that time-domain PSOLA is the formant-preserving shifter for singing voice
//! (measured 0.00 st envelope drift vs "exactly the transpose amount" for resample/spectral
//! engines), with a quality knee around ±3–4 st. The training side uses praat's PSOLA
//! (python-only); this is the runtime Rust engine, A/B-gated against praat with rmvpe-blooded
//! metrics (parselmouth is systematically BLIND to PSOLA glitches — S41).
//!
//! Design (the caller ALWAYS knows f0 — cover path: rmvpe track; vocal path: the parametric
//! f0 the render was fed), which makes analysis-mark placement a guided search instead of
//! blind epoch detection (praat's pitch marks are internal — its external pitch-tier drive
//! was falsified in S41; here WE own the marks):
//!   - voiced islands (contiguous f0>0 frames) are processed independently; unvoiced audio
//!     passes through UNTOUCHED (dry), voiced↔unvoiced seams blend by the synthesis window
//!     coverage (no hard splice);
//!   - analysis marks: island-polarity-adaptive peak picking, next mark searched in
//!     [prev + 0.7·T, prev + 1.4·T] with T = sr/f0 at the previous mark;
//!   - synthesis: marks advance by T_local/ratio; each copies a 2·T Hann-windowed grain
//!     centered on the NEAREST analysis mark; overlap-add with a normalization buffer;
//!   - output length == input length exactly (pitch shift only, no time change).
//!
//! Ratio semantics: `ratio[i]` = output_f0 / input_f0 for frame i (hop `hop` samples);
//! 1.0 = identity. The range-extension tiers bypass this entirely for in-comfort audio —
//! ratio==1 calls are the CALLER's responsibility to skip (bit-exactness by construction).

/// Practical f0 clamp for period computation (mirrors the RMVPE-era plumbing: below 50 Hz
/// periods get too long to window locally; above 1100 Hz is out of the trained range).
const F0_MIN_HZ: f32 = 50.0;
const F0_MAX_HZ: f32 = 1100.0;

/// Islands shorter than this many periods can't carry a full analysis window — passed dry.
const MIN_ISLAND_PERIODS: f32 = 2.5;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PsolaParams {
    /// Samples per f0/ratio frame (e.g. 441 for 100 fps @ 44.1 kHz).
    pub hop: usize,
}

/// Pitch-shift `x` by the per-frame `ratio`, guided by the known per-frame `f0_hz`
/// (0 = unvoiced). `f0_hz.len() == ratio.len()`; frame i covers samples
/// [i·hop, (i+1)·hop). Returns a buffer of exactly `x.len()` samples.
pub fn psola_shift(x: &[f32], sr: u32, f0_hz: &[f32], ratio: &[f32], params: PsolaParams) -> Vec<f32> {
    assert_eq!(f0_hz.len(), ratio.len(), "f0/ratio track length mismatch");
    let n = x.len();
    let hop = params.hop.max(1);
    if n == 0 || f0_hz.is_empty() {
        return x.to_vec();
    }
    let sr_f = sr as f32;
    let frame_of = |s: usize| (s / hop).min(f0_hz.len() - 1);
    let f0_at = |s: usize| f0_hz[frame_of(s)];
    let ratio_at = |s: usize| ratio[frame_of(s)];
    let period_at = |s: usize| {
        let f = f0_at(s).clamp(F0_MIN_HZ, F0_MAX_HZ);
        (sr_f / f).max(8.0)
    };

    // ── voiced islands in SAMPLE space (frame-aligned) ──
    let mut islands: Vec<(usize, usize)> = Vec::new();
    {
        let mut start: Option<usize> = None;
        for f in 0..f0_hz.len() {
            let s0 = f * hop;
            if s0 >= n {
                break;
            }
            let voiced = f0_hz[f] > 0.0;
            match (voiced, start) {
                (true, None) => start = Some(s0),
                (false, Some(a)) => {
                    islands.push((a, s0));
                    start = None;
                }
                _ => {}
            }
        }
        if let Some(a) = start {
            islands.push((a, n));
        }
    }

    let mut wet = vec![0.0f64; n];
    let mut norm = vec![0.0f64; n];

    for &(a, b) in &islands {
        let b = b.min(n);
        if b <= a {
            continue;
        }
        let t_first = period_at(a) as usize;
        if (b - a) as f32 / period_at(a) < MIN_ISLAND_PERIODS {
            continue; // too short to window — stays dry via the blend below
        }

        // island polarity: do glottal pulses peak positive or negative here?
        let probe_end = (a + 4 * t_first).min(b);
        let max_v = x[a..probe_end].iter().cloned().fold(f32::MIN, f32::max);
        let min_v = x[a..probe_end].iter().cloned().fold(f32::MAX, f32::min);
        let sign = if max_v.abs() >= min_v.abs() { 1.0f32 } else { -1.0f32 };

        // ── analysis marks: guided peak picking ──
        let mut marks: Vec<usize> = Vec::new();
        {
            // first mark = strongest (signed) peak within the first period
            let w_end = (a + t_first).min(b);
            let mut best = a;
            let mut best_v = f32::MIN;
            for i in a..w_end {
                let v = x[i] * sign;
                if v > best_v {
                    best_v = v;
                    best = i;
                }
            }
            marks.push(best);
            loop {
                let prev = *marks.last().unwrap();
                let t = period_at(prev);
                let lo = prev + (0.7 * t) as usize;
                let hi = (prev + (1.4 * t) as usize).min(b);
                if lo + 1 >= hi {
                    break;
                }
                let mut best = lo;
                let mut best_v = f32::MIN;
                for i in lo..hi {
                    let v = x[i] * sign;
                    if v > best_v {
                        best_v = v;
                        best = i;
                    }
                }
                marks.push(best);
                if best + (0.7 * period_at(best)) as usize >= b {
                    break;
                }
            }
        }
        if marks.len() < 2 {
            continue;
        }

        // local analysis period per mark = distance to the neighboring marks (more faithful
        // than sr/f0 under vibrato); ends take their single neighbor's distance.
        let local_t = |k: usize| -> usize {
            if marks.len() < 2 {
                return period_at(marks[k]) as usize;
            }
            if k == 0 {
                marks[1] - marks[0]
            } else if k == marks.len() - 1 {
                marks[k] - marks[k - 1]
            } else {
                ((marks[k + 1] - marks[k - 1]) / 2).max(1)
            }
        };

        // ── synthesis: advance by T/ratio, grain from the nearest analysis mark ──
        let mut pos = marks[0] as f64;
        let island_end = b as f64;
        // guard: ratio ≤ 0 or non-finite would loop forever — clamp to a sane band
        let safe_ratio = |s: usize| {
            let r = ratio_at(s);
            if r.is_finite() { r.clamp(0.25, 4.0) } else { 1.0 }
        };
        let mut nearest = 0usize; // marks are monotonic; advance a cursor instead of bisecting
        while pos < island_end {
            let pi = pos as usize;
            while nearest + 1 < marks.len()
                && marks[nearest + 1].abs_diff(pi) <= marks[nearest].abs_diff(pi)
            {
                nearest += 1;
            }
            let m = marks[nearest];
            let t_a = local_t(nearest).clamp(8, 2048);
            // 2·T Hann grain centered on the analysis mark, OLA'd centered at `pos`
            let center = pos.round() as isize;
            for j in -(t_a as isize)..=(t_a as isize) {
                let src = m as isize + j;
                let dst = center + j;
                if src < 0 || src >= n as isize || dst < 0 || dst >= n as isize {
                    continue;
                }
                let w = 0.5 + 0.5 * ((std::f32::consts::PI * j as f32) / t_a as f32).cos();
                wet[dst as usize] += (x[src as usize] * w) as f64;
                norm[dst as usize] += w as f64;
            }
            let step = local_t(nearest) as f64 / safe_ratio(pi) as f64;
            pos += step.max(1.0);
        }
    }

    // ── blend. INSIDE an island the window sum legitimately dips (down-shift: grain
    // spacing T/r > T ⇒ min coverage ≈ 0.5) — those samples must be PURE normalized wet;
    // mixing dry there leaks original-pitch content (caught by the praat A/B: -5 st p90
    // blew to 68¢). Dry only fades in below NORM_FLOOR, i.e. across the window tails at
    // island edges — which keeps voiced↔unvoiced seams smooth without explicit crossfades. ──
    const NORM_FLOOR: f64 = 0.25;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let nv = norm[i];
        if nv >= NORM_FLOOR {
            out.push((wet[i] / nv) as f32);
        } else if nv > 1e-6 {
            let a = nv / NORM_FLOOR;
            out.push((a * (wet[i] / nv) + (1.0 - a) * x[i] as f64) as f32);
        } else {
            out.push(x[i]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 44100;
    const HOP: usize = 441; // 100 fps

    /// Synthetic "voice": harmonic sawtooth-ish tone with mild vibrato + amplitude envelope.
    fn synth_tone(f0_base: f32, secs: f32, vibrato_cents: f32) -> (Vec<f32>, Vec<f32>) {
        let n = (SR as f32 * secs) as usize;
        let mut x = vec![0.0f32; n];
        let mut f0 = Vec::new();
        let mut phase = 0.0f64;
        for i in 0..n {
            let vib = (vibrato_cents / 1200.0) * (2.0 * std::f32::consts::PI * 5.5 * i as f32 / SR as f32).sin();
            let f = f0_base * 2f32.powf(vib);
            if i % HOP == 0 {
                f0.push(f);
            }
            phase += f as f64 / SR as f64;
            let mut s = 0.0f32;
            for h in 1..=8 {
                s += ((2.0 * std::f64::consts::PI * phase * h as f64).sin() as f32) / h as f32;
            }
            let env = (i.min(n - i) as f32 / (0.01 * SR as f32)).min(1.0); // 10ms fade in/out
            x[i] = 0.3 * s * env;
        }
        (x, f0)
    }

    /// Median f0 (Hz) of a buffer via normalized autocorrelation over 40 ms windows.
    fn measure_f0_median(x: &[f32], lo_hz: f32, hi_hz: f32) -> f32 {
        let win = (SR as f32 * 0.04) as usize;
        let lag_min = (SR as f32 / hi_hz) as usize;
        let lag_max = (SR as f32 / lo_hz) as usize;
        let mut readings = Vec::new();
        let mut start = win;
        while start + win + lag_max < x.len() {
            let seg = &x[start..start + win + lag_max];
            let corr = |lag: usize| -> f32 {
                let mut num = 0.0f32;
                let mut d0 = 0.0f32;
                let mut d1 = 0.0f32;
                for i in 0..win {
                    num += seg[i] * seg[i + lag];
                    d0 += seg[i] * seg[i];
                    d1 += seg[i + lag] * seg[i + lag];
                }
                num / (d0.sqrt() * d1.sqrt() + 1e-9)
            };
            let rs: Vec<f32> = (lag_min..=lag_max).map(corr).collect();
            let best = rs.iter().cloned().fold(f32::MIN, f32::max);
            // octave disambiguation: a perfectly periodic tone peaks equally at T and 2T —
            // take the smallest lag that is a LOCAL correlation peak within 95% of the best
            // (a bare 95% threshold lands a few samples early on the broad peak = tens of cents)
            let best_lag = (1..rs.len().saturating_sub(1)).find(|&k| {
                rs[k] >= 0.95 * best && rs[k] >= rs[k - 1] && rs[k] >= rs[k + 1]
            });
            if best > 0.5 {
                if let Some(k) = best_lag {
                    readings.push(SR as f32 / (lag_min + k) as f32);
                }
            }
            start += win;
        }
        readings.sort_by(|a, b| a.partial_cmp(b).unwrap());
        readings[readings.len() / 2]
    }

    fn cents(a: f32, b: f32) -> f32 {
        1200.0 * (a / b).log2()
    }

    #[test]
    fn shift_up_3st_accurate() {
        let (x, f0) = synth_tone(220.0, 1.2, 30.0);
        let r = 2f32.powf(3.0 / 12.0);
        let ratio = vec![r; f0.len()];
        let y = psola_shift(&x, SR, &f0, &ratio, PsolaParams { hop: HOP });
        assert_eq!(y.len(), x.len());
        let f_in = measure_f0_median(&x, 100.0, 500.0);
        let f_out = measure_f0_median(&y, 100.0, 600.0);
        let err = cents(f_out, f_in * r).abs();
        assert!(err < 25.0, "3st up: {err:.1} cents off (in {f_in:.1} Hz out {f_out:.1} Hz)");
    }

    #[test]
    fn shift_down_4st_accurate() {
        let (x, f0) = synth_tone(330.0, 1.2, 25.0);
        let r = 2f32.powf(-4.0 / 12.0);
        let ratio = vec![r; f0.len()];
        let y = psola_shift(&x, SR, &f0, &ratio, PsolaParams { hop: HOP });
        let f_in = measure_f0_median(&x, 150.0, 600.0);
        let f_out = measure_f0_median(&y, 100.0, 600.0);
        let err = cents(f_out, f_in * r).abs();
        assert!(err < 25.0, "4st down: {err:.1} cents off (in {f_in:.1} out {f_out:.1})");
    }

    #[test]
    fn unvoiced_passthrough_bit_exact() {
        // noise with an all-zero f0 track must come back bit-identical (pure dry path)
        let mut x = vec![0.0f32; SR as usize];
        let mut seed = 0x12345678u32;
        for v in x.iter_mut() {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (seed as f32 / u32::MAX as f32 - 0.5) * 0.2;
        }
        let f0 = vec![0.0f32; x.len() / HOP + 1];
        let ratio = vec![1.5f32; f0.len()];
        let y = psola_shift(&x, SR, &f0, &ratio, PsolaParams { hop: HOP });
        assert_eq!(x, y);
    }

    #[test]
    fn mixed_voiced_unvoiced_no_click_and_length() {
        // tone in the middle, silence+noise around it; sanity: length exact, no NaN, no
        // sample exceeding a sane bound (window normalization keeps gain ~1)
        let (tone, f0_tone) = synth_tone(260.0, 0.5, 20.0);
        let pad = SR as usize / 2;
        let mut x = vec![0.0f32; pad];
        x.extend_from_slice(&tone);
        x.extend(vec![0.0f32; pad]);
        let mut f0 = vec![0.0f32; pad / HOP];
        f0.extend_from_slice(&f0_tone);
        f0.extend(vec![0.0f32; pad / HOP + 2]);
        let ratio = vec![2f32.powf(2.0 / 12.0); f0.len()];
        let y = psola_shift(&x, SR, &f0, &ratio, PsolaParams { hop: HOP });
        assert_eq!(y.len(), x.len());
        assert!(y.iter().all(|v| v.is_finite()));
        let peak_in = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let peak_out = y.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(peak_out < peak_in * 1.8, "gain blew up: {peak_in} -> {peak_out}");
    }

    #[test]
    fn time_varying_ratio_tracks() {
        // first half shift +2st, second half -2st — both halves must land on target
        let (x, f0) = synth_tone(240.0, 2.0, 0.0);
        let up = 2f32.powf(2.0 / 12.0);
        let dn = 2f32.powf(-2.0 / 12.0);
        let ratio: Vec<f32> = (0..f0.len()).map(|i| if i < f0.len() / 2 { up } else { dn }).collect();
        let y = psola_shift(&x, SR, &f0, &ratio, PsolaParams { hop: HOP });
        let half = y.len() / 2;
        let margin = SR as usize / 5;
        let f_a = measure_f0_median(&y[..half - margin], 100.0, 600.0);
        let f_b = measure_f0_median(&y[half + margin..], 100.0, 600.0);
        assert!(cents(f_a, 240.0 * up).abs() < 30.0, "first half {f_a:.1} Hz");
        assert!(cents(f_b, 240.0 * dn).abs() < 30.0, "second half {f_b:.1} Hz");
    }
}
