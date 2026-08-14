//! TD-PSOLA — pitch shift with the formant envelope preserved BY CONSTRUCTION.
//!
//! Why this exists (and why it is written the way it is): on 2026-07-26 we deleted a Rust PSOLA
//! because "PSOLA sounds dirty", and the gate that guarded it (`psola_ab.rs`) **only measured
//! pitch**. The repo already recorded, at that time, that praat's TD-PSOLA is transparent on our
//! own material (ΔHNR −0.12..+2.68 dB) while *our* implementation lost 5.15–8.57 dB — i.e. the
//! thing that lost was our implementation, not the algorithm. The old file's own comment (see
//! `git show b231e4e^:src-tauri/src/inference/vocal_range.rs`, the paragraph above the engine
//! choice) is even more damning: it measured **the same 5–9 dB loss at ratio 1.000**, no shift at
//! all — the damage was a flat tax on passing through the resynthesis. A correct TD-PSOLA at
//! ratio 1.0 is the IDENTITY, and that is the cheapest, least-fakeable gate this module has.
//!
//! Every design decision below was made by a gate going red, not by taste (S146):
//!
//! 1. **Target pulses come from the analysis marks, not from an independent phase integration.**
//!    Treat the marks as a phase function Φ(m_k) = k; synthesis mark `s_j = Φ⁻¹(j/r)`, source mark
//!    `k = round(j/r)`. Copy/discard then needs no branch, |s_j − m_k| ≤ half a period always, the
//!    time axis is the identity (so the exact-length contract is structural, not patched
//!    afterwards), and at r = 1 it degenerates to `s_j = m_j` exactly. Integrating a pitch tier
//!    instead leaves every grain displaced by up to half a period even at r = 1 (measured: the
//!    identity gate went red at max |Δ| = 1.05).
//! 2. **Marks are found on the DC-removed signal; grains are cut from the ORIGINAL signal.**
//!    Mixing the two shifts the whole output by the input's DC (identity gate: max == median ==
//!    2.277e-3 == mean(x)).
//! 3. **No mark is ever rejected.** praat drops low-correlation pulses because its PointProcess is
//!    a standalone object; our Φ(m_k) = k needs *every* period to carry a mark — one missing mark
//!    makes that stretch synthesize at double the period. Measured gap rate 1.1%, and turning
//!    rejection on breaks the identity gate (2598 samples over tolerance).
//! 4. **The bell width is the TARGET neighbour distance, clipped by the SOURCE neighbour distance.**
//!    Clipping is what actually makes downward shifts change the pitch: without it a down-shifted
//!    grain spans two source periods and the overlap-add reproduces the input. Measured at −12 st:
//!    output bit-identical to input, while the envelope / correlation / HNR rulers all read
//!    "perfect" — only the pitch ruler (the "necessary but not sufficient" one) caught it at
//!    +1200 cents. Clipping costs exact COLA on the way down; that cost is *reported*
//!    (`PsolaDiagnostics::cola_gap_frac`) rather than hidden.
//! 5. **Dry signal (copyFlat) is only ever mixed in OUTSIDE the first..last target pulse.** Inside
//!    that span a window-sum shortfall is a defect; covering it with the un-shifted input is how
//!    the 2026-07 implementation produced beating at syllable edges.
//! 6. **The correlation window is 3 periods** (praat's own source uses 1). Measured on real
//!    material at +6 st: 1 period ⇒ ΔHNR −3.03, 2 ⇒ −1.77, **3 ⇒ −1.67**, 4 ⇒ −2.41, 6 ⇒ −2.54,
//!    against a ceiling of −1.66 measured by feeding praat's own pulse train into this same
//!    synthesizer. Three periods reaches that ceiling.
//!
//! Reference readings on the S146 material (炉心融解 bars 28-44 × 东雪莲, the two rescued phrases;
//! `scripts/range_rulers/`): at +6 st this implementation reads envelope shift +0.50 st /
//! ΔHNR −1.66 dB against praat's +0.30 / −1.44; at −6, −12 and +12 it is *better* than praat on
//! ΔHNR. ⛔ praat is NOT a valid reference above about +6 st — its pitch ceiling (1400 Hz) is
//! breached and its own arm degrades (envelope +2.70, peak correlation 0.661 at +12).

/// What the shift could not do cleanly, reported instead of hidden.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PsolaDiagnostics {
    /// Voiced islands that carried at least three marks.
    pub islands: usize,
    /// Total analysis marks placed.
    pub marks: usize,
    /// Fraction of covered samples whose overlap-add window sum fell below 0.9. Structurally ~0
    /// when shifting up; on the way down the source-width clipping (see 4 above) makes it large —
    /// that is the amplitude ripple, and it is the number to watch when a down-shift sounds wobbly.
    pub cola_gap_frac: f32,
    /// Median window sum over the covered span (1.0 = exact COLA).
    pub cola_w_median: f32,
}

const MAX_PERIOD_SECONDS: f64 = 0.02;
/// See design note 6. Changing this changes the quality readings — re-run
/// `scripts/range_rulers/compare.py` before and after.
const CORR_WIN_PERIODS: f64 = 3.0;
const SEARCH_LO: f64 = 0.8;
const SEARCH_HI: f64 = 1.25;
const MIN_ISLAND_SECONDS: f64 = 0.02;

/// Sub-sample peak of a parabola through three samples.
fn parabolic(l: f64, m: f64, r: f64) -> f64 {
    let d = 2.0 * m - l - r;
    if d.abs() > 1e-30 {
        0.5 * (r - l) / d
    } else {
        0.0
    }
}

/// f0 at a sample position, linearly interpolated INSIDE a voiced run; 0 = unvoiced.
/// Never interpolates across a voiced/unvoiced boundary — a blended 0 would invent a period.
fn f0_at(f0: &[f32], hop: usize, i: f64) -> f64 {
    if hop == 0 || f0.is_empty() || i < 0.0 {
        return 0.0;
    }
    let u = i / hop as f64;
    let k = u as usize;
    if k >= f0.len() {
        return 0.0;
    }
    let a = f64::from(f0[k]);
    if !(a > 0.0) {
        return 0.0;
    }
    let b = if k + 1 < f0.len() { f64::from(f0[k + 1]) } else { a };
    if !(b > 0.0) {
        a
    } else {
        a + (b - a) * (u - k as f64)
    }
}

/// Contiguous voiced sample ranges, probed on a coarse grid (the boundaries only need to be
/// good enough to bracket the mark recursion — the marks themselves come from the waveform).
fn voiced_islands(f0: &[f32], hop: usize, n: usize, min_samples: usize) -> Vec<(usize, usize)> {
    const STRIDE: usize = 32;
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < n {
        if !(f0_at(f0, hop, i as f64) > 0.0) {
            i += STRIDE;
            continue;
        }
        let a = i;
        let mut j = i;
        while j + STRIDE < n && f0_at(f0, hop, (j + STRIDE) as f64) > 0.0 {
            j += STRIDE;
        }
        let b = (j + STRIDE).min(n);
        if b - a >= min_samples {
            out.push((a, b));
        }
        i = b + STRIDE;
    }
    out
}

/// Absolute extremum in `[i0, i1)`, polarity-independent, refined to sub-sample.
fn find_extremum(x: &[f32], i0: f64, i1: f64) -> f64 {
    let lo = (i0.max(1.0)) as usize;
    let hi = (i1.min((x.len() as f64) - 1.0)).max(0.0) as usize;
    if hi <= lo {
        return lo as f64;
    }
    let mut k = lo;
    let mut best = -1.0f64;
    for i in lo..hi {
        let v = f64::from(x[i]).abs();
        if v > best {
            best = v;
            k = i;
        }
    }
    let s = if x[k] >= 0.0 { 1.0 } else { -1.0 };
    k as f64
        + parabolic(
            s * f64::from(x[k - 1]),
            s * f64::from(x[k]),
            s * f64::from(x[k + 1]),
        )
}

/// Normalized cross-correlation of the window at `t1` against every integer position in
/// `[lo, hi]`, with the correlation peak refined to sub-sample. Returns (position, correlation);
/// on any out-of-range condition it returns `t1` unchanged — callers MUST enforce forward
/// progress themselves (an earlier version looped forever here).
fn max_correlation(x: &[f32], t1: f64, period: f64, lo: f64, hi: f64) -> (f64, f64) {
    let n = x.len();
    let h = (period * 0.5 * CORR_WIN_PERIODS).round() as isize;
    if h < 2 {
        return (t1, 0.0);
    }
    let h = h as usize;
    let a0 = t1.round() as isize - h as isize;
    if a0 < 0 || a0 as usize + 2 * h > n {
        return (t1, 0.0);
    }
    let a = &x[a0 as usize..a0 as usize + 2 * h];
    let na: f64 = a.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
    if na <= 1e-30 {
        return (t1, 0.0);
    }
    let lo_i = lo.floor() as isize;
    let hi_i = hi.ceil() as isize;
    let mut best_c = -1.0f64;
    let mut best_i = lo_i;
    let mut prev = -1.0f64;
    let mut at_best_prev = -1.0f64;
    let mut at_best_next = -1.0f64;
    let mut pending_next = false;
    for c in lo_i..=hi_i {
        let b0 = c - h as isize;
        let v = if b0 < 0 || b0 as usize + 2 * h > n {
            -1.0
        } else {
            let b = &x[b0 as usize..b0 as usize + 2 * h];
            let mut dot = 0.0f64;
            let mut nb = 0.0f64;
            for (p, q) in a.iter().zip(b.iter()) {
                let (p, q) = (f64::from(*p), f64::from(*q));
                dot += p * q;
                nb += q * q;
            }
            if nb > 1e-30 {
                dot / (na * nb).sqrt()
            } else {
                -1.0
            }
        };
        if pending_next {
            at_best_next = v;
            pending_next = false;
        }
        if v > best_c {
            best_c = v;
            best_i = c;
            at_best_prev = prev;
            pending_next = true;
        }
        prev = v;
    }
    let off = if best_i > lo_i && best_i < hi_i && at_best_prev >= -1.0 && at_best_next >= -1.0 {
        parabolic(at_best_prev, best_c, at_best_next)
    } else {
        0.0
    };
    (best_i as f64 + off, best_c.max(0.0))
}

/// Pitch marks inside one voiced island: seed at the island's midpoint extremum, then recurse
/// outward maximizing correlation with the previous mark's neighbourhood.
fn analysis_marks(x: &[f32], sample_rate: u32, f0: &[f32], hop: usize, a: usize, b: usize) -> Vec<f64> {
    let sr = f64::from(sample_rate);
    let mid = (a + b) as f64 * 0.5;
    let f_mid = f0_at(f0, hop, mid);
    if !(f_mid > 0.0) {
        return Vec::new();
    }
    let t0 = sr / f_mid;
    let seed = find_extremum(x, mid - 0.5 * t0, mid + 0.5 * t0);
    let mut marks = vec![seed];
    for dir in [1.0f64, -1.0] {
        let mut cur = seed;
        loop {
            let f = f0_at(f0, hop, cur);
            if !(f > 0.0) {
                break;
            }
            let per = sr / f;
            let (lo, hi) = if dir > 0.0 {
                (cur + SEARCH_LO * per, cur + SEARCH_HI * per)
            } else {
                (cur - SEARCH_HI * per, cur - SEARCH_LO * per)
            };
            if lo < a as f64 - per || hi > b as f64 + per {
                break;
            }
            let (pos, _corr) = max_correlation(x, cur, per, lo, hi);
            // Forward progress is mandatory: max_correlation returns `cur` unchanged at the
            // buffer edges, and without this the loop never terminates.
            if dir > 0.0 && pos < cur + 0.5 * per {
                break;
            }
            if dir < 0.0 && pos > cur - 0.5 * per {
                break;
            }
            if pos < 0.0 || pos >= x.len() as f64 {
                break;
            }
            marks.push(pos);
            cur = pos;
        }
    }
    marks.sort_by(f64::total_cmp);
    marks
}

/// One grain: rising half-cosine into `t_pos`, falling half out of it, cut from `x` around
/// `s_pos`. Transport is by whole samples — praat does the same, and fractional delay would
/// destroy the ratio-1.0 identity.
fn add_bell(
    x: &[f32],
    acc: &mut [f64],
    wsum: &mut [f64],
    s_pos: f64,
    t_pos: f64,
    lw: f64,
    rw: f64,
) {
    let n = x.len() as isize;
    let d = t_pos.round() as isize - s_pos.round() as isize;
    for (w0, w1, rise) in [(-lw, 0.0, true), (0.0, rw, false)] {
        let i0 = (s_pos + w0).round() as isize;
        let i1 = (s_pos + w1).round() as isize;
        if i1 <= i0 {
            continue;
        }
        let len = (i1 - i0) as f64;
        for i in i0..i1 {
            let ti = i + d;
            if i < 0 || i >= n || ti < 0 || ti >= n {
                continue;
            }
            let ph = ((i - i0) as f64 + 0.5) / len * std::f64::consts::PI;
            let w = if rise {
                0.5 * (1.0 - ph.cos())
            } else {
                0.5 * (1.0 + ph.cos())
            };
            acc[ti as usize] += f64::from(x[i as usize]) * w;
            wsum[ti as usize] += w;
        }
    }
}

/// Shift `x` by `semitones` (positive = up) keeping duration and formants. `f0_hz` is the
/// per-frame fundamental of `x` itself (0 = unvoiced), `f0_hop` its stride in samples.
/// Output length always equals input length.
pub fn psola_shift(
    x: &[f32],
    sample_rate: u32,
    semitones: f64,
    f0_hz: &[f32],
    f0_hop: usize,
) -> Vec<f32> {
    psola_shift_diag(x, sample_rate, semitones, f0_hz, f0_hop).0
}

/// As [`psola_shift`], plus what it could not do cleanly.
pub fn psola_shift_diag(
    x: &[f32],
    sample_rate: u32,
    semitones: f64,
    f0_hz: &[f32],
    f0_hop: usize,
) -> (Vec<f32>, PsolaDiagnostics) {
    let n = x.len();
    let mut diag = PsolaDiagnostics::default();
    // NOTE: deliberately no `semitones == 0 => return x` shortcut. That shortcut would make the
    // ratio-1.0 identity gate vacuously true, which is the exact shape of gate that let the 2026-07
    // implementation through. The caller (`vocal_range::apply_inverse`) owns the shift==0 fast path.
    if n == 0 || !semitones.is_finite() || sample_rate == 0 || f0_hop == 0 {
        return (x.to_vec(), diag);
    }
    let sr = f64::from(sample_rate);
    let ratio = 2f64.powf(semitones / 12.0);
    if !(ratio.is_finite() && ratio > 0.0) {
        return (x.to_vec(), diag);
    }
    let mean = x.iter().map(|v| f64::from(*v)).sum::<f64>() / n as f64;
    let dc_free: Vec<f32> = x.iter().map(|v| (f64::from(*v) - mean) as f32).collect();

    let mut acc = vec![0.0f64; n];
    let mut wsum = vec![0.0f64; n];
    let mut covered = vec![false; n];
    let max_period = MAX_PERIOD_SECONDS * sr;

    for (a, b) in voiced_islands(f0_hz, f0_hop, n, (MIN_ISLAND_SECONDS * sr) as usize) {
        let src = analysis_marks(&dc_free, sample_rate, f0_hz, f0_hop, a, b);
        if src.len() < 3 {
            continue;
        }
        diag.islands += 1;
        diag.marks += src.len();
        let last = (src.len() - 1) as f64;
        let count = (last * ratio) as usize;
        let mut tgt: Vec<f64> = Vec::with_capacity(count + 1);
        let mut ks: Vec<usize> = Vec::with_capacity(count + 1);
        for j in 0..=count {
            let u = j as f64 / ratio;
            if u > last {
                break;
            }
            let lo = u as usize;
            let hi = (lo + 1).min(src.len() - 1);
            tgt.push(src[lo] + (src[hi] - src[lo]) * (u - lo as f64));
            ks.push(u.round() as usize);
        }
        if tgt.len() < 3 {
            continue;
        }
        let c0 = tgt[0].round().max(0.0) as usize;
        let c1 = (tgt[tgt.len() - 1].round().max(0.0) as usize).min(n);
        for s in covered.iter_mut().take(c1).skip(c0) {
            *s = true;
        }
        for i in 0..tgt.len() {
            let tm = tgt[i];
            let tl = if i > 0 { tgt[i - 1] } else { tm - (tgt[1] - tm) };
            let tr = if i + 1 < tgt.len() {
                tgt[i + 1]
            } else {
                tm + (tm - tgt[tgt.len() - 2])
            };
            let k = ks[i].min(src.len() - 1);
            let src_l = if k > 0 { src[k] - src[k - 1] } else { tm - tl };
            let src_r = if k + 1 < src.len() { src[k + 1] - src[k] } else { tr - tm };
            let lw = (tm - tl).min(src_l);
            let rw = (tr - tm).min(src_r);
            if lw <= 1.0 || rw <= 1.0 || lw > max_period || rw > max_period {
                continue;
            }
            add_bell(x, &mut acc, &mut wsum, src[k], tm, lw, rw);
        }
    }

    let mut gap = 0usize;
    let mut cov_n = 0usize;
    let mut ws: Vec<f64> = Vec::new();
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let w = wsum[i].clamp(0.0, 1.0);
        if covered[i] {
            cov_n += 1;
            if w < 0.9 {
                gap += 1;
            }
            ws.push(w);
            out[i] = acc[i] as f32;
        } else {
            // copyFlat: outside the synthesized span the un-shifted input rides the window-sum
            // ramp, which IS the crossfade. Inside it, a shortfall is a defect and must not be
            // papered over with un-shifted audio (that is beating, not repair).
            out[i] = (acc[i] + (1.0 - w) * f64::from(x[i])) as f32;
        }
    }
    diag.cola_gap_frac = if cov_n > 0 { gap as f32 / cov_n as f32 } else { 0.0 };
    diag.cola_w_median = if ws.is_empty() {
        1.0
    } else {
        ws.sort_by(f64::total_cmp);
        ws[ws.len() / 2] as f32
    };
    (out, diag)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A voiced test signal: harmonics of `f0` under a fixed formant envelope.
    /// ⛔ Synthetic periodic material systematically flatters/frames PSOLA-class algorithms
    /// (S81, three times) — these tests assert STRUCTURE (identity, length, that the pitch
    /// actually moved), never quality. Quality lives in `scripts/range_rulers/` on real renders.
    fn voiced(sr: u32, secs: f64, f0: f64) -> Vec<f32> {
        let n = (f64::from(sr) * secs) as usize;
        let mut y = vec![0.0f32; n];
        let formants = [(700.0, 90.0, 1.0), (1200.0, 110.0, 0.6), (2600.0, 160.0, 0.35)];
        let mut k = 1.0;
        while k * f0 < f64::from(sr) * 0.45 {
            let f = k * f0;
            let mut e = 1e-4;
            for (fc, bw, amp) in formants {
                e += amp / (1.0 + ((f - fc) / bw).powi(2));
            }
            for (i, s) in y.iter_mut().enumerate() {
                *s += (e * (2.0 * std::f64::consts::PI * f * i as f64 / f64::from(sr)).cos()) as f32;
            }
            k += 1.0;
        }
        let peak = y.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-9);
        y.iter().map(|v| v / peak * 0.9).collect()
    }

    fn flat_f0(n: usize, hop: usize, hz: f32) -> Vec<f32> {
        vec![hz; n / hop + 2]
    }

    /// Dominant period by NORMALIZED autocorrelation, in samples — a coarse but independent
    /// pitch readout. It must be normalized: dividing only by the overlap length biases the
    /// score toward long lags and this helper then reports the search ceiling (it did, at 735).
    fn dominant_period(x: &[f32], lo: usize, hi: usize) -> usize {
        let mut best = lo;
        let mut best_v = f64::NEG_INFINITY;
        let mut scores: Vec<f64> = Vec::new();
        for lag in lo..hi.min(x.len() / 2) {
            let (mut dot, mut ea, mut eb) = (0.0f64, 0.0f64, 0.0f64);
            for i in 0..x.len() - lag {
                let (a, b) = (f64::from(x[i]), f64::from(x[i + lag]));
                dot += a * b;
                ea += a * a;
                eb += b * b;
            }
            let v = if ea > 1e-30 && eb > 1e-30 { dot / (ea * eb).sqrt() } else { -1.0 };
            scores.push(v);
            if v > best_v {
                best_v = v;
                best = lag;
            }
        }
        // Octave guard: multiples of the true period score just as high on stationary material
        // (it reported 147 for a 74-sample output). Take the smallest lag that is BOTH a local
        // maximum and within 10% of the best — "smallest above threshold" alone lands on the
        // shoulder of the peak and reads 71 for a true 73.5.
        for i in 1..scores.len() - 1 {
            if scores[i] >= best_v * 0.9 && scores[i] >= scores[i - 1] && scores[i] >= scores[i + 1] {
                best = lo + i;
                break;
            }
        }
        best
    }

    #[test]
    fn ratio_one_is_the_identity() {
        // THE gate. The implementation we deleted in 2026-07 lost 5-9 dB of HNR at ratio 1.000 —
        // the damage was the resynthesis itself, and only this assertion sees that.
        let sr = 44_100;
        let x = voiced(sr, 0.5, 220.0);
        let hop = sr as usize / 200;
        let (y, _) = psola_shift_diag(&x, sr, 0.0, &flat_f0(x.len(), hop, 220.0), hop);
        assert_eq!(y.len(), x.len());
        let worst = x
            .iter()
            .zip(y.iter())
            .enumerate()
            .map(|(i, (a, b))| ((a - b).abs(), i))
            .fold((0.0f32, 0usize), |m, v| if v.0 > m.0 { v } else { m });
        assert!(
            worst.0 < 1e-5,
            "ratio 1.0 must reproduce the input; worst |Δ| = {} at sample {}",
            worst.0,
            worst.1
        );
    }

    #[test]
    fn the_length_contract_holds_in_both_directions() {
        let sr = 44_100;
        let x = voiced(sr, 0.4, 300.0);
        let hop = sr as usize / 200;
        let f0 = flat_f0(x.len(), hop, 300.0);
        for st in [-24.0, -12.0, -6.0, -1.0, 1.0, 6.0, 12.0, 24.0] {
            let (y, _) = psola_shift_diag(&x, sr, st, &f0, hop);
            assert_eq!(y.len(), x.len(), "length changed at {st} st");
            assert!(y.iter().all(|v| v.is_finite()), "non-finite output at {st} st");
        }
    }

    #[test]
    fn the_period_readout_itself_is_calibrated() {
        // Positive control for the measurement, not for PSOLA: the helper is the only thing
        // standing between "the pitch moved" and "I believe the pitch moved", so it gets its own
        // known answer first. 44100/300 = 147, 44100/600 = 73.5.
        let sr = 44_100;
        for (f0, want) in [(300.0, 147.0), (600.0, 73.5), (150.0, 294.0)] {
            let got = dominant_period(&voiced(sr, 0.3, f0), 40, 800);
            assert!(
                (got as f64 - want).abs() / want < 0.03,
                "readout says {got} for {f0} Hz, expected ≈{want}"
            );
        }
    }

    #[test]
    fn the_pitch_actually_moves_up_and_down() {
        // The −12 case is the one that matters: an earlier version returned the input unchanged
        // there (the envelope / correlation / HNR rulers all read "perfect"; only pitch caught it).
        let sr = 44_100;
        let f0 = 300.0;
        let x = voiced(sr, 0.5, f0);
        let hop = sr as usize / 200;
        let f0t = flat_f0(x.len(), hop, f0 as f32);
        let base = dominant_period(&x, 40, 800);
        for (st, want) in [(12.0, 0.5f64), (-12.0, 2.0), (6.0, 0.5f64.sqrt()), (-6.0, 2f64.sqrt())] {
            let (y, _) = psola_shift_diag(&x, sr, st, &f0t, hop);
            let got = dominant_period(&y[sr as usize / 10..y.len() - sr as usize / 10], 40, 800);
            let expect = base as f64 * want;
            assert!(
                (got as f64 - expect).abs() / expect < 0.08,
                "{st} st: period {got} samples, expected ≈{expect:.0} (input {base})"
            );
        }
    }

    #[test]
    fn an_all_unvoiced_input_passes_through_untouched() {
        // Fricatives and silence must not be overlap-added at all (that is how consonants get
        // metallic). No marks ⇒ nothing covered ⇒ copyFlat everywhere.
        let sr = 44_100;
        let mut x = vec![0.0f32; sr as usize / 2];
        let mut seed = 12345u32;
        for s in x.iter_mut() {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *s = (seed >> 8) as f32 / 8_388_608.0 - 1.0;
        }
        let hop = sr as usize / 200;
        let (y, d) = psola_shift_diag(&x, sr, 6.0, &vec![0.0f32; x.len() / hop + 2], hop);
        assert_eq!(d.islands, 0);
        assert_eq!(y, x, "unvoiced material must be copied verbatim");
    }

    #[test]
    fn shifting_up_keeps_exact_cola_and_shifting_down_reports_its_ripple() {
        // Not a quality claim — an attribution instrument. Upward the bells tile exactly; downward
        // the source-width clipping (which is what makes the pitch actually drop) leaves gaps, and
        // that ripple must be VISIBLE rather than silently filled with un-shifted audio.
        let sr = 44_100;
        let x = voiced(sr, 0.5, 300.0);
        let hop = sr as usize / 200;
        let f0 = flat_f0(x.len(), hop, 300.0);
        let (_, up) = psola_shift_diag(&x, sr, 6.0, &f0, hop);
        let (_, down) = psola_shift_diag(&x, sr, -6.0, &f0, hop);
        assert!(up.cola_gap_frac < 0.02, "upward gap {} too high", up.cola_gap_frac);
        assert!((up.cola_w_median - 1.0).abs() < 0.02, "upward window sum {}", up.cola_w_median);
        assert!(
            down.cola_gap_frac > 0.2,
            "downward ripple must be reported, got {}",
            down.cola_gap_frac
        );
    }

    /// Run THIS implementation on a real render so `scripts/range_rulers/compare.py` can put it
    /// under the same four rulers as praat and the current production engine. Structural tests
    /// above cannot answer "is it clean" — synthetic periodic material systematically flatters
    /// PSOLA-class algorithms (S81, three times).
    ///
    /// ```powershell
    /// $env:UTAI_PSOLA_IN="…\arm_raw.wav"; $env:UTAI_PSOLA_F0="…\f0.f32"   # raw LE f32, one per hop
    /// $env:UTAI_PSOLA_HOP="220"; $env:UTAI_PSOLA_ST="6"; $env:UTAI_PSOLA_OUT="…\arm_rust.wav"
    /// cargo test -p utai-dsp psola_probe -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "probe: needs a wav + f0 track on disk (set UTAI_PSOLA_*)"]
    fn psola_probe() {
        let path = std::env::var("UTAI_PSOLA_IN").expect("UTAI_PSOLA_IN");
        let out = std::env::var("UTAI_PSOLA_OUT").expect("UTAI_PSOLA_OUT");
        let f0p = std::env::var("UTAI_PSOLA_F0").expect("UTAI_PSOLA_F0");
        let hop: usize = std::env::var("UTAI_PSOLA_HOP").expect("UTAI_PSOLA_HOP").parse().unwrap();
        let st: f64 = std::env::var("UTAI_PSOLA_ST").expect("UTAI_PSOLA_ST").parse().unwrap();

        let mut rd = hound::WavReader::open(&path).expect("open in");
        let spec = rd.spec();
        let x: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => rd.samples::<f32>().map(|s| s.unwrap()).collect(),
            hound::SampleFormat::Int => rd
                .samples::<i32>()
                .map(|s| s.unwrap() as f32 / (1i32 << (spec.bits_per_sample - 1)) as f32)
                .collect(),
        };
        let x: Vec<f32> = if spec.channels > 1 {
            x.chunks(spec.channels as usize).map(|c| c.iter().sum::<f32>() / c.len() as f32).collect()
        } else {
            x
        };
        let raw = std::fs::read(&f0p).expect("read f0");
        let f0: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let (y, d) = psola_shift_diag(&x, spec.sample_rate, st, &f0, hop);
        println!(
            "psola_probe: {} samples @{} Hz, {st:+} st, f0 frames {} hop {hop}\n  \
             islands {} marks {} cola_gap {:.1}% w_median {:.3}",
            x.len(), spec.sample_rate, f0.len(), d.islands, d.marks,
            d.cola_gap_frac * 100.0, d.cola_w_median
        );
        assert_eq!(y.len(), x.len(), "exact-length contract");

        let peak = y.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-9);
        let g = 0.92 / peak; // same normalization the S145 arms used, so levels are comparable
        let mut w = hound::WavWriter::create(
            &out,
            hound::WavSpec {
                channels: 1,
                sample_rate: spec.sample_rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .expect("create out");
        for v in &y {
            w.write_sample((((v * g).clamp(-1.0, 1.0)) * 32767.0).round() as i16).unwrap();
        }
        w.finalize().unwrap();
        println!("  -> {out}");
    }

    #[test]
    fn degenerate_inputs_are_returned_rather_than_panicking() {
        let sr = 44_100;
        let hop = sr as usize / 200;
        assert!(psola_shift(&[], sr, 6.0, &[], hop).is_empty());
        let tiny = vec![0.1f32; 32];
        assert_eq!(psola_shift(&tiny, sr, 6.0, &flat_f0(32, hop, 300.0), hop), tiny);
        let x = voiced(sr, 0.2, 300.0);
        assert_eq!(psola_shift(&x, sr, 6.0, &[], hop), x, "no f0 ⇒ no marks ⇒ passthrough");
        assert_eq!(psola_shift(&x, sr, f64::NAN, &flat_f0(x.len(), hop, 300.0), hop), x);
        assert_eq!(psola_shift(&x, sr, 6.0, &flat_f0(x.len(), hop, 300.0), 0), x);
    }
}
