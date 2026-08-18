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
    /// Window-sum p01 / p99 over the covered span, and the fraction above 1.05.
    ///
    /// ⛔ Why these exist: the three fields above were all computed from a **clamped** window sum,
    /// so `cola_w_median` was structurally ≤ 1.000 and the overlap SURPLUS — the thing that only
    /// appears when shifting UP, i.e. the production direction — could not be read at all. A
    /// diagnostic that cannot express the failure it is watching for is an empty criterion
    /// (S129). The clamp is still applied where it is *used* (the dry-fill gain below); only the
    /// statistics moved to the raw sum, so the audio is bit-identical.
    pub cola_w_p01: f32,
    pub cola_w_p99: f32,
    pub cola_over_frac: f32,
    /// RMS(样本) of the sub-sample transport residual **that was discarded**.
    /// ⭐ 0.0000 at ratio 1.0 · ≈0.41 for whole-sample transport at any other ratio ·
    /// 0 once `frac_transport` carries it. See `add_bell`.
    pub transport_residual_rms: f32,
    /// S150 — how many analysis marks the phase lock actually moved.
    /// ⛔ Same reason as `wsola_moved`: "the arm is on" and "the arm did something" are two
    /// different facts, and only the second one is visible in the audio.
    pub marks_locked: usize,
    /// S148 — how many grains the WSOLA search actually moved off `src[k]`.
    /// ⛔ It is the difference between "the arm is on" and "the arm did something": a search that
    /// never moves a grain produces byte-identical audio, which is indistinguishable from the arm
    /// being off unless this number is printed. (S147 shipped a change whose benefit was silently
    /// halved and the only thing that exposed it was a count in a log line.)
    pub wsola_moved: usize,
    /// S151 — the fraction of the analysed source span that **no grain ever reads**.
    ///
    /// Each grain reads `[s_pos − lw, s_pos + rw)`, and on an UP-shift both half-widths collapse
    /// to the *target* spacing (`:1027-1028` takes `min(target, source)`), i.e. `T_src / ratio`.
    /// So the read windows around consecutive source marks stop touching the moment
    /// **`ratio > 2`, which is exactly `|shift| > 12` semitones**, and from there a slice of every
    /// pitch period — the low-energy middle, i.e. the formant ring-down, since the locked marks
    /// sit on the energy peaks — is simply never used. Computed from the real mark train
    /// (61523 marks, goose donor −7): +7 → 0.00 %, +11 → 0.00 %, **+12 → 0.00 %**, +13 → 4.9 %,
    /// **+14 → 10.2 %**, +16 → 20.0 %.
    ///
    /// ⛔ Why it had to become a field rather than a note: **nothing else can see this.**
    /// `cola_gap_frac` / `cola_w_*` are computed in the OUTPUT domain where the half-windows equal
    /// the target spacing by construction (measured 0.00 % / 1.000 at every shift from +7 to +16);
    /// the in-note envelope-depth ruler reads p50 −0.01 dB at +14; the per-note octave gate reads
    /// 0.00 % with its positive control at 100 %; and every mark-layer ruler is **ratio-invariant**
    /// by construction — `analysis_marks` and `lock_phase` do not take `ratio` at all. S148 and
    /// S150 calibrated this engine only over `|shift| ≤ 7`; the user's 2026-08-18 run reached −14.
    /// ⚠ It is a coverage number, not an audibility one — there is no ear datum on this axis yet.
    pub src_uncovered_frac: f32,
    /// S152 — the share of the OUTPUT's energy that sits below ~50 Hz, **before** any removal.
    ///
    /// ⭐ This process **manufactures** it. The donor going in carries 0.001 % there; the output
    /// carries, per note (goose +7 × akiko, 174 notes with ≥0.3 s of body, median):
    /// −9 st (ratio 1.68) **10.8 %** · −12 (2.00) **25.5 %** · −14 (2.24) **33.7 %**
    /// (p90 39.5 / 64.6 / 72.1 %). On the production render the notes that went through a rescue
    /// group read p50 0.558 % against p50 0.002 % for the ones that did not — a 279× separation.
    ///
    /// **Mechanism** — `the_manufactured_baseline_is_the_narrowed_grain_window_s_own_mean`:
    /// on an up-shift both half-widths collapse to `T_src / ratio` (`:1027-1028` takes the min of
    /// the target and source spacings), so every grain reads a window **narrower than one period,
    /// centred on the mark** — and S150's phase lock puts the marks on the energy peaks. The
    /// bell-weighted mean of a sub-period window centred on a peak is not zero; `wsum ≈ 1` then
    /// lays that same mean down under every grain. The gate predicts the baseline from that window
    /// alone and matches the measured value within 3× at +4/+8/+12/+16 st.
    ///
    /// ⛔ **A wrong mechanism was written here first and the gate killed it on its first run.**
    /// The claim was "it needs an ASYMMETRIC waveform (a glottal pulse); a symmetric source
    /// produces none" — measured: a plain sine at +4 st injects **more** than the asymmetric pulse
    /// train (0.195 vs 0.104 by |mean|/RMS). Symmetry is not the variable; *being centred on a
    /// peak with a sub-period window* is the whole of it. ⚠ Keep this paragraph: this file has
    /// history with confidently-worded wrong comments (one of them kept TD-PSOLA out for four
    /// months).
    ///
    /// ⚠⚠ **Two collars that must travel with this number, or it reads as an emergency:**
    /// 1. **It is inaudible.** 97 % of it is below 5 Hz; 20-50 Hz holds ≤0.26 %.
    /// 2. **It does not eat headroom.** Removing everything under 50 Hz moves the whole-song peak
    ///    by **−0.008 dB** (a locally-measured note peak moves ~0.34 dB).
    ///
    /// ⭐ What it *does* cost: it is inside every RMS-domain ruler we own, so S150/S151's `rms` /
    /// `depth` / `ripple` readings on rescued notes carry up to a third of their energy in
    /// infrasound; and it is the shape behind the user's "波形甚是诡异" (registry `user_coordinates`
    /// points at `[685]`, the note whose waveform rides a wandering baseline).
    pub infrasonic_frac: f32,
    /// S152 — how much of that the removal arm actually took out (`before − after`), 0.0 when off.
    /// ⛔ Same reason as `wsola_moved` / `marks_locked`: "the arm is on" and "the arm did
    /// something" are different facts and only the second one is visible in the audio.
    pub infrasonic_removed: f32,
}

const MAX_PERIOD_SECONDS: f64 = 0.02;
/// See design note 6. Changing this changes the quality readings — re-run
/// `scripts/range_rulers/compare.py` before and after.
const CORR_WIN_PERIODS: f64 = 3.0;
const SEARCH_LO: f64 = 0.8;
const SEARCH_HI: f64 = 1.25;
const MIN_ISLAND_SECONDS: f64 = 0.02;

/// S150 — phase locking. Half-width of the energy window used to find the pulse, in periods.
/// Measured on the akiko donor at +7 (Σ|depth − upper bound| over the 5 registered notes):
/// T/16 → 1.88 · **T/8 → 1.23** · T/4 → 0.83 · T/2 → 12.03. The T/2 collapse is the tell that
/// this is a real optimum and not a flat knob: a window that wide stops resolving the pulse.
const LOCK_ENERGY_HALF: f64 = 0.125;
/// Loop gain of the phase-locked loop in [`lock_phase`]: how much of each period's measured phase
/// error is applied. 1.0 = jump straight onto the detected pulse; small = track it slowly.
///
/// ⛔⛔ **Two earlier formulations each shipped their own audible artifact**, and the shape of the
/// failure is what picked this structure — not taste:
/// * **Correct the marks afterwards** (what S150 shipped first): the correction is bounded by the
///   search radius while the underlying drift is not, so the correction runs to the edge of the
///   window and then traverses the *whole* window back — a sawtooth. Measured on the trajectory:
///   p05..p95 of the correction is exactly ±0.45 periods, i.e. the full search range, and the
///   envelope modulation spectrum grows a **coherent 10 Hz peak, 44-74×** the median. The user
///   heard this immediately and named it precisely: the spindles became "one short seam after
///   another". ⭐ Their reading of *why* was also right — the un-locked engine's scattered phase
///   was **dithering** that seam, which is why it sounded like slow wobble instead of seams.
/// * **Snap greedily inside the walk** (jump fully onto the pulse each step): no accumulation, so
///   no sawtooth — but with a nearly degenerate peak choice (runner-up/winner 0.72-0.99) it
///   **alternates** between sub-features. Measured: spacing-deviation lag-1 −0.538 (baseline
///   −0.259) and transient flux 50.4 at the click coordinates (baseline 20.6).
///
/// A PLL avoids both by construction: it references the *absolute* pulse every step (so the error
/// cannot accumulate ⇒ no wrap) and applies only a fraction of it (so a flip-flopping detection is
/// low-passed ⇒ no alternation). Measured against the two failures and against the gold standard:
///
/// | | Σ\|depth−upper\| | spacing jitter p99 | flux at the seams | 10 Hz coherence | worst lag-1 |
/// |---|---|---|---|---|---|
/// | untouched engine | 18.94 | 0.1377 | 20.6 | 28.6× @279 Hz | −0.259 |
/// | correct-afterwards | 2.02 | 0.1382 | 22.1 | **44.5× @10 Hz** | −0.243 |
/// | greedy in-loop | 0.54 | 0.1888 | **50.4** | 10.2× @350 Hz | **−0.538** |
/// | **this, β = 0.1** | **0.63** | 0.1393 | 24.8 | 16.0× @350 Hz | −0.287 |
/// | praat's own marks | 0.02 | 0.0632 | 25.0 | 7.9× @356 Hz | −0.155 |
const LOCK_BETA: f64 = 0.1;

/// Half-width of the windowed-sinc used to carry the sub-sample transport residual.
///
/// ⚠ 32 taps is what S146g measured with; the cost is real but bounded (37.1 s of audio went
/// 159 ms → 1592 ms = still 23× realtime), and the recovered quality is ~80-85% of the whole
/// fixed toll. ⛔ **Never substitute linear interpolation here**: it reads BETTER on the HNR
/// ruler (+0.43…+2.23 dB) purely because it low-passes, and a same-construction control that
/// only low-passes reads +0.40 on its own — i.e. the ruler is being paid, not the ear.
const TRANSPORT_SINC_HALF: isize = 16;

/// Read `x` at a fractional position with a Blackman-windowed sinc.
/// Integer `pos` returns the sample itself to within f64 rounding (sinc(k) = 0 for k ≠ 0).
fn sinc_read(x: &[f32], pos: f64) -> f64 {
    let n = x.len() as isize;
    let c = pos.floor() as isize;
    let frac = pos - c as f64;
    let mut acc = 0.0;
    for k in (-TRANSPORT_SINC_HALF + 1)..=TRANSPORT_SINC_HALF {
        let idx = c + k;
        if idx < 0 || idx >= n {
            continue;
        }
        let t = k as f64 - frac; // distance from the tap to the read position
        let s = if t.abs() < 1e-12 {
            1.0
        } else {
            let pt = std::f64::consts::PI * t;
            pt.sin() / pt
        };
        // Blackman window over the tap span, so the kernel dies smoothly at the edges.
        let ph = std::f64::consts::PI * (t / (TRANSPORT_SINC_HALF as f64) + 1.0);
        let w = 0.42 - 0.5 * ph.cos() + 0.08 * (2.0 * ph).cos();
        acc += f64::from(x[idx as usize]) * s * w;
    }
    acc
}

/// Running RMS of the transport residual that was **discarded** (0 when it is carried).
/// ⭐ This is a direct, non-vacuous readout of the quantity the fix is about: it is exactly
/// 0.0000 at ratio 1.0, ≈0.41 samples for whole-sample transport at any other ratio, and 0 once
/// the residual is carried. A diagnostic that reports the defect itself cannot go quietly stale.
#[derive(Default)]
struct ResidualStat {
    sq: f64,
    n: usize,
}

impl ResidualStat {
    fn push(&mut self, v: f64) {
        self.sq += v * v;
        self.n += 1;
    }
    fn rms(&self) -> f32 {
        if self.n == 0 {
            0.0
        } else {
            (self.sq / self.n as f64).sqrt() as f32
        }
    }
}

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

/// S148 — per-grain trace for the "spindle" investigation (test-only, env-gated, no production
/// path). The user named the defect from the waveform: a sustained rescued note breaks into a
/// string of lens shapes. `cola_w_median` is 1.000 there, so it is not a window-sum gap — it has
/// to be coherent addition between overlapping grains. What decides that is how far each grain's
/// own pulse sits from where it is being placed:
///
/// ```text
/// u = j / ratio ;  tgt[j] = interp(src, u)   (smooth)
///                  ks[j]  = round(u)          (quantised)  <- the only lossy step
///                  delta  = tgt[j] - src[round(u)]
/// ```
///
/// ⛔ Do not derive the modulation rate from `ratio` on paper — I tried, and the arithmetic
/// matched the −7 arm (~0.68 s) while missing the −5 arm by 3.5×. Dump it and measure.
#[cfg(test)]
static GRAIN_TRACE: std::sync::Mutex<Vec<[f64; 8]>> = std::sync::Mutex::new(Vec::new());

#[cfg(test)]
fn grain_trace_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("UTAI_PSOLA_GRAIN_DUMP").is_ok())
}

/// S148 — pick the SOURCE read position for one grain by waveform similarity with what has
/// already accumulated (WSOLA). Returns `s0` unchanged unless some offset in `±radius` beats it.
///
/// The comparison window is the grain's LEFT half — that is exactly the span where this grain will
/// overlap the previous one, i.e. the only place where a phase mismatch can cancel. `acc` there is
/// the previous grain's windowed right half; both sides are normalised, so the previous grain's
/// window taper does not bias the score.
///
/// ⛔ `MARGIN` is not cosmetic: at ratio 1.0 the accumulator **is** the windowed input at this very
/// position, so offset 0 is the true maximum — but the correlation surface is flat around it and
/// floating-point noise could hand the win to a neighbour, which would break the identity gate
/// (`ratio_one_is_the_identity`) that this whole module is built on. Requiring a strict improvement
/// keeps 0 the winner whenever nothing is actually better.
fn wsola_pick(x: &[f32], acc: &[f64], s0: f64, tm: f64, lw: f64, radius: f64) -> f64 {
    const MARGIN: f64 = 1e-4;
    let r = radius.round() as isize;
    let n = lw.round() as usize;
    if r < 1 || n < 8 {
        return s0;
    }
    let ti = tm.round() as isize - n as isize;
    if ti < 0 || ti as usize + n > acc.len() {
        return s0;
    }
    let a = &acc[ti as usize..ti as usize + n];
    let anorm = a.iter().map(|v| v * v).sum::<f64>().sqrt();
    if anorm <= 0.0 {
        return s0;
    }
    let score = |off: isize| -> f64 {
        let si = s0.round() as isize - n as isize + off;
        if si < 0 || si as usize + n > x.len() {
            return f64::NEG_INFINITY;
        }
        let b = &x[si as usize..si as usize + n];
        let bn = b.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>().sqrt();
        if bn <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let dot: f64 = a.iter().zip(b).map(|(p, q)| p * f64::from(*q)).sum();
        dot / (anorm * bn)
    };
    let mut best = 0isize;
    let mut bestv = score(0);
    if !bestv.is_finite() {
        return s0;
    }
    for off in -r..=r {
        if off == 0 {
            continue;
        }
        let v = score(off);
        if v > bestv + MARGIN {
            bestv = v;
            best = off;
        }
    }
    s0 + best as f64
}

/// Energy in `[c-h, c+h)`, clipped at the buffer edges. A shrinking window at the edges is fine:
/// it only ever competes against its own neighbours a sample away.
fn window_energy(x: &[f32], c: isize, h: isize) -> f64 {
    let n = x.len();
    let lo = c.saturating_sub(h).max(0) as usize;
    let hi = ((c + h).max(0) as usize).min(n);
    if hi <= lo {
        return 0.0;
    }
    x[lo..hi].iter().map(|v| f64::from(*v) * f64::from(*v)).sum()
}

/// Pull ONE position onto the nearest pulse: integer argmax of the local energy within
/// `±radius_periods · t`, refined to sub-sample with the same parabola the rest of this module
/// uses. Returns `pos` unchanged when the radius is zero or the search cannot run.
///
/// ⭐ Why local ENERGY and not something cleverer: measured on this material, praat's own marks sit
/// at the argmax of `E(T/8)` (or, indistinguishably, of `|x|`) for **62%** of marks against a
/// random floor of **9.4%**, while ours manage 35%. The textbook glottal-closure detectors do
/// *worse*: LPC residual 26%, `|dx/dt|` 19%. So the cheap feature is also the right one here —
/// don't port the literature over that measurement.
/// ⚠ The energy argmax of a one-sided (sharp onset, decaying ring) pulse sits *after* the onset by
/// roughly the window half-width — a constant bias, which is harmless here because what the
/// synthesis needs is a consistent phase reference, not the glottal closure instant itself.
fn snap_to_pulse(x: &[f32], pos: f64, t: f64, radius_periods: f64) -> f64 {
    if !(radius_periods > 0.0) || !(t > 2.0) || !pos.is_finite() {
        return pos;
    }
    let r = ((radius_periods * t).round() as isize).max(1);
    let h = ((LOCK_ENERGY_HALF * t).round() as isize).max(2);
    let c = pos.floor() as isize;
    let (mut best, mut best_v, mut best_i) = (pos, f64::NEG_INFINITY, c);
    let (mut vm1, mut vp1, mut last) = (f64::NAN, f64::NAN, f64::NAN);
    let mut have = false;
    for i in (c - r)..=(c + r) {
        if i < 0 || i as usize >= x.len() {
            last = f64::NAN;
            continue;
        }
        let v = window_energy(x, i, h);
        if have && i == best_i + 1 {
            vp1 = v;
        }
        if !have || v > best_v {
            have = true;
            best_v = v;
            best = i as f64;
            best_i = i;
            vm1 = last;
            vp1 = f64::NAN;
        }
        last = v;
    }
    if !have {
        return pos;
    }
    if vm1.is_finite() && vp1.is_finite() {
        let d = parabolic(vm1, best_v, vp1);
        if d.is_finite() {
            best += d.clamp(-1.0, 1.0);
        }
    }
    best
}

/// S150 — **phase locking**: pull each analysis mark onto the nearest glottal pulse.
/// Returns how many marks actually moved. `radius_periods == 0` ⇒ no-op, marks untouched.
///
/// ## Why this exists (S148 root cause, measured — do not re-derive)
///
/// The walk in [`analysis_marks`] steps by *correlation*, so its **period is right and its phase
/// is not**: the step size is accurate (measured spacing median 120.00 samples against praat's
/// 119.75, and our spacing is actually *smoother* — 0.0013 vs 0.0021 relative variation) while
/// nothing ever re-anchors the marks to the waveform after the seed. The phase error therefore
/// accumulates: our marks scatter **±0.42 of a period** around praat's and land where the local
/// energy is **2.3–4.4 dB lower**.
///
/// That costs exactly what the user named by eye ("a string of lens shapes" in a sustained
/// rescued note): a mark that sits off the pulse makes the `T_out`-wide grain window cut a
/// different slice of the pulse every period, which writes an envelope modulation that was not in
/// the input. ⭐ The decisive single-variable experiment: swapping **only** the marks for praat's
/// reproduced praat's readings on 5 notes × 2 metrics to within 0.02 dB (`[785]` 9.49 → 3.45
/// against praat's own 3.47) ⇒ the synthesis rules are already correct; 100% of the injected
/// modulation comes from mark placement. ⭐ And the note that never grows the defect at any shift
/// (`[86]`) is the one whose marks are already accurate (0.33 dB) — the strongest internal
/// consistency this chain has.
///
/// ## Contract (each clause is a defect that was measured, not a preference)
///
/// * **Every mark survives.** Design note 3: Φ(m_k) = k needs one mark per period. Locking only
///   ever *moves* marks — never adds, drops or merges them.
/// * **The radius is the LOCAL step, not the island's median.** An island can run 2580 marks
///   (≈7 s) with the pitch moving inside it, so a median-derived radius can exceed half of a
///   locally shorter period and snap onto the *neighbouring* pulse — which is how you manufacture
///   period doubling. The code therefore uses `t = steps[i - 1]`, the walk's own local step.
///   ⛔ This is not hypothetical caution: S148's WSOLA arm was killed by a blind test precisely
///   for manufacturing an exact −1200 cent period doubling.
///   ⛔⛔ **S151 correction — the previous sentence here was wrong, and wrong in the direction
///   that matters.** It claimed the radius used `min(gap_left, gap_right)` and that this made
///   overshoot "structurally impossible". Only the LEFT gap is read (`:601`), and the geometry
///   does not close: snapping one period late needs `0.30·t ≥ 2T − t`, i.e. `t ≥ 1.538·T`, while
///   the walk's own band (`SEARCH_LO`/`SEARCH_HI` = 0.8/1.25) only bounds consecutive steps to
///   `t/T ≤ 1.5625`. That is a **1.6 % overlap, not a proof.** What actually makes a bad snap
///   harmless is [`LOCK_BETA`]: a mark that snaps to the wrong pulse still moves only
///   `0.1 · 0.30 · t` = **3 % of a period**. Measured end to end on the shipped arm (the two mark
///   dumps S150 left in `TESTING\s150_marks\work`, 61523 marks / 272 islands): per-island span
///   ratio locked/unlocked p50 0.999819 = **−0.31 cents**, p05/p95 −3.7/+2.9 cents, and exactly
///   **1 island of 272** past ±10 cents. ⇒ the claim (no octave manufacture) survives; the reason
///   given for it did not. Written down because a wrong-but-confident comment in this very file
///   is what kept TD-PSOLA shut out for four months (`scripts/range_rulers/README.md`).
/// * **Marks may not cross or collapse** (the same guard, enforced sequentially).
/// * **The correction is a first-order loop, not a smoother.** `cur = pred + β·(snap − pred)`,
///   `β =` [`LOCK_BETA`]. ⛔ **S151 correction:** this bullet used to describe a **median smoother
///   over `LOCK_SMOOTH` marks`** — that constant has not existed since the PLL replaced that arm
///   (`grep LOCK_SMOOTH` hits this comment and nothing else), and the arm it describes is one of
///   the two the user's ear rejected. Its readings are kept below, **labelled as the rejected
///   arm**, because the comparison is what forced this shape.
/// * ⛔ **"Agreement with praat's marks" is NOT a criterion** — S146 measured an `absmax`
///   detector with the *highest* agreement (67%) and the *worst* ΔHNR (−5.94) and voiced survival
///   (52%). The criteria are the rulers in `scripts/range_rulers/` plus f0.
///
/// ## Why local energy, and why post-hoc — both measured, neither chosen by taste
///
/// **The feature.** praat's own marks sit at the argmax of `E(T/8)` — or, indistinguishably, of
/// `|x|` — for **62%** of marks against a **9.4%** random floor (jitter praat's marks ±0.5 T and
/// every feature collapses to 9-10%). Ours manage 35%. The textbook glottal-closure detectors do
/// *worse*: LPC residual **26%**, `|dx/dt|` **19%**. ⇒ the cheap feature is the right one here;
/// do not port the literature over that measurement. (praat also actively *avoids* zero crossings:
/// 4.2% against a 19.7% floor.)
///
/// **The shape.** The error is **cumulative, not per-step noise**: per-step median 0.0024 of a
/// period against an accumulated median of 0.0695 (p90 0.40, worst 3.31), lag-1 autocorrelation
/// +0.9897, growing monotonically with distance from the seed (0.009 within 2 steps → 0.177 beyond
/// 80). praat is not cleverer — its search is *weaker* than ours (1-period window vs our 3) — it
/// re-anchors on every voiced interval while we anchor **once per island**, and our islands run to
/// 2580 marks. ⇒ That argues for locking *inside* the walk, and it was implemented and measured:
///
/// | arm | marks | landing energy | spacing var | Σ\|depth − upper\| | w_p01 at −7 |
/// |---|---|---|---|---|---|
/// | today | 61523 | −11.83 | 0.0013 | 18.94 | 0.2915 |
/// | in-loop α=0.15 | 61518 | −10.91 | 0.0061 | 0.54 | 0.2851 |
/// | in-loop α=0.45 | **57077** | −10.79 | 0.0062 | 0.54 | 0.2554 |
/// | post-hoc α=0.45 + smoothing (**rejected by ear**) | **61523** | −10.87 | **0.0028** | **0.45** | **0.2882** |
/// | upper bound (praat's marks) | 58777 | −10.98 | 0.0021 | 0.02 | 0.2905 |
///
/// ⛔ **None of the three candidate rows above is what ships** — the shipped arm is the PLL, and
/// its own numbers are in [`LOCK_BETA`]. This table is kept because it is the evidence that killed
/// the two obvious formulations: in-loop fights the walk's mandatory forward-progress check and
/// **changes the mark count** (−4446 = −7.2% at 0.45), a design-note-3 violation — those stretches
/// synthesize at double the period; and the post-hoc row, which looked best here on every
/// instrument, was the one the user's ear returned as **clicks** (v1) and then as a **short seam**
/// every ~100 ms (v2). ⇒ ⭐ this table IS the "instruments all green, ear says no" sample.
///
/// **The negative control that had to be run.** "Depth went down" does not by itself mean "the
/// marks found the pulses" — any consistent absolute anchor stops the accumulation. So the same
/// mechanism was run snapping to the energy **minimum**: Σ 18.94 → **9.56** (it does help) but
/// landing energy collapses to **−22.12 dB** (against −10.87 here and −10.98 for praat) and
/// spacing jitter quadruples. ⇒ the ruler separates "anchored" from "anchored on the pulse" by
/// 20×, and the claim survives its own control.
///
/// ## ⛔ How to check this for octave errors, and how NOT to
///
/// The in-loop arm above does not merely change the mark count — on some notes it lays down an
/// **alternating long-short spacing**, which is period doubling written into the mark train
/// itself. Two things about finding it:
///
/// * **Per note, never per song.** The same criterion on the same arm reads 34.6% over the two
///   broken notes, 3.4% over 23 notes and **0.78% over all voiced frames** — diluted 44×. A
///   whole-song f0 average cannot see a defect that lives on 2 notes out of 23. (S148 already
///   paid for this once: the r4 positive control only reads its 24.25% inside the right window.)
/// * **The mark train tells you before the audio does**, and without trusting any pitch tracker —
///   which matters, because on those notes pyworld and praat disagree. Take the local spacing
///   ratio `d[i] / median(d[i±10])` and correlate its deviation with itself at lag 1: alternating
///   long-short shows up as a strongly NEGATIVE lag-1. ⚠ A plain "bad spacing rate" with ±0.5 T
///   thresholds is structurally blind here — the alternation is 0.7 T / 1.3 T, i.e. *inside* the
///   band.
///
/// Measured, worst note per arm (lag-1 of the spacing deviation / worst-note low-octave rate from
/// a per-note dio+stonemask pass, positive control = drop every other mark ⇒ 100%):
/// baseline **−0.259 / 0.00%** · **this arm −0.345 / 0.00%** · praat's marks −0.369 / 0.00% ·
/// in-loop 0.15 **−0.562** · in-loop 0.45 **−0.654**. ⇒ this arm's worst note sits between the
/// baseline and the praat-marks arm, and no arm here except the deliberate positive control shows
/// an octave error at +7 or +12. ⚠ Honest cost: spacing jitter p90 0.054 against praat's 0.012.
fn lock_phase(x: &[f32], marks: &mut [f64], radius_periods: f64) -> usize {
    if radius_periods <= 0.0 || marks.len() < 3 {
        return 0;
    }
    // The walk's PERIODS are already right (spacing median 120.00 samples against praat's 119.75);
    // only the phase is wrong. So the loop predicts from the walk's own step and corrects phase.
    let steps: Vec<f64> = marks.windows(2).map(|w| w[1] - w[0]).collect();
    let mut moved = 0usize;
    for i in 1..marks.len() {
        let t = steps[i - 1];
        if !(t > 2.0) {
            marks[i] = marks[i - 1] + t.max(1.0);
            continue;
        }
        let pred = marks[i - 1] + t;
        let snapped = snap_to_pulse(x, pred, t, radius_periods);
        // ⭐ Only a fraction of the measured error — see LOCK_BETA for what each of the two
        // extremes (correct-afterwards, and beta = 1) sounded like.
        let next = pred + LOCK_BETA * (snapped - pred);
        // Strictly increasing is load-bearing for Phi(m_k) = k, so enforce rather than assume.
        let next = if next > marks[i - 1] + 1.0 { next } else { marks[i - 1] + t };
        if next != marks[i] {
            moved += 1;
        }
        marks[i] = next;
    }
    moved
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
/// `s_pos`.
///
/// ⛔⛔ **The comment that used to live here was wrong, and it was load-bearing.** It read
/// "Transport is by whole samples — praat does the same, and fractional delay would destroy the
/// ratio-1.0 identity", and it kept the single largest quality fix out of this file for four
/// months (S146 → S146g). The identity is **structural**, not a consequence of integer transport:
/// at ratio 1.0 the target pulses ARE the source marks, so `t_pos == s_pos` exactly and the
/// residual is identically zero however it is transported. Measured with both `frac == 0`
/// short-circuits removed: worst |Δ| = 5.5e-18 on synthetic and on real material.
///
/// ⭐ What integer transport actually costs (S146g, three independent measurements agreeing on
/// one signature): `d` is a difference of TWO independent roundings, so every grain discards a
/// sub-sample residual δ. RMS(δ) is **exactly 0.0000 at ratio 1.0** and jumps to **0.41 samples**
/// the instant the ratio leaves 1.0 — then stays there regardless of depth (+1 → 0.4139,
/// +6 → 0.4129, +8 → 0.4121). That is precisely the shape of the toll we could not explain:
/// −0.59 dB HNR for entering the process once, and only −0.37 more from +7 to +8. Depth changes
/// the grain COUNT, not the rate.
/// ⚠ The discriminating control: at +12 (ratio 2.0) half the grains land on integer deltas, so
/// jitter halves while duplicate grains rise to 50%. "Duplicate grains are the problem" predicts
/// the toll keeps worsening; "jitter is the problem" predicts a plateau. Measured: **plateau**.
///
/// `frac_transport` = carry that residual with a windowed-sinc read instead of dropping it.
///
/// ⛔⛔ **BLIND TEST SAYS NO (2026-08-16). Leave it OFF and do not reopen this without new
/// material.** Three packages, **7 load-bearing pairs + 3 blank controls**, level-matched to
/// ±0.000 dB, both arms fed the SAME rendered wav: the user could not tell them apart anywhere —
/// 东雪莲 at +6, akiko at +7, and the two spots they had themselves named as still-improvable
/// (bars 169 「たらああ」 at −7, the deepest shift in the song, and bars 189-192 「いだあああ」).
/// ⭐ The null is load-bearing, not a shrug: the same listener and the same protocol **got both
/// load-bearing groups right** when S146 swapped the engine, so the setup is demonstrably
/// sensitive to a real perceptual difference.
///
/// What that means, stated so nobody re-derives it: **removing this jitter is worth ~+1.05 dB
/// ΔHNR and f0 p90 132→24 cents, and NONE of it is audible on this material.** The measurement
/// was right; the inference "better number ⇒ better sound" was not. Keep the arm — the defect it
/// removes is real and may matter for other material (deeper shifts, other languages, the cover
/// lane) — but the burden of proof for turning it on is a blind test that PASSES, not a ruler.
///
/// ⚠ Residual difference between the arms, measured: −22.2 dB relative, correlation 0.9970,
/// worst |Δ| 0.19 — i.e. **an order of magnitude smaller than re-rendering the same command
/// twice** (worst |Δ| 1.47, SVC is not bit-reproducible). Predicting "inaudible" from that alone
/// would have been fair; the blind test is what makes it a fact.
#[allow(clippy::too_many_arguments)]
/// S152 — half-width of the moving average used to isolate the infrasonic baseline, in ms.
/// Two passes of a box of this length = a triangular window, whose first null sits at
/// `1000 / INFRASONIC_MA_MS` Hz and whose stop-band then falls as 1/f².
///
/// ⛔ 20 ms is **measured, not chosen**: on a real production output (goose +7 × akiko, −14 st,
/// the worst ratio on the score) the resulting high-pass reads
/// 0-5 Hz **−41.3 dB** · 5-20 −19.3 · 20-50 −2.9 · 50-100 **−0.20** · 100-200 **−0.08** ·
/// ≥200 Hz **−0.01 dB**. That is the whole point of the two-pass form: a single box leaves
/// −13 dB side-lobes that would put ±0.4 dB of ripple on the fundamental, and the fundamental of
/// a donor is 150-500 Hz — the one band this must not touch.
const INFRASONIC_MA_MS: f64 = 20.0;

/// Box filter with a running prefix sum; the window shrinks at the two ends rather than
/// zero-padding (zero-padding would manufacture a step exactly where the buffer starts).
fn box_average(x: &[f64], half: usize) -> Vec<f64> {
    let n = x.len();
    let mut pre = Vec::with_capacity(n + 1);
    let mut s = 0.0f64;
    pre.push(0.0);
    for v in x {
        s += *v;
        pre.push(s);
    }
    (0..n)
        .map(|i| {
            let a = i.saturating_sub(half);
            let b = (i + half + 1).min(n);
            (pre[b] - pre[a]) / (b - a) as f64
        })
        .collect()
}

/// The infrasonic baseline of `x` — two box passes = a triangular low-pass. See
/// [`INFRASONIC_MA_MS`] for the measured response.
fn infrasonic_baseline(x: &[f32], sample_rate: u32) -> Vec<f64> {
    let half = (((f64::from(sample_rate) * INFRASONIC_MA_MS / 1000.0) as usize) / 2).max(1);
    let v: Vec<f64> = x.iter().map(|s| f64::from(*s)).collect();
    box_average(&box_average(&v, half), half)
}

fn add_bell(
    x: &[f32],
    acc: &mut [f64],
    wsum: &mut [f64],
    s_pos: f64,
    t_pos: f64,
    lw: f64,
    rw: f64,
    formant_rate: f64,
    frac_transport: bool,
    residual: &mut ResidualStat,
) {
    let n = x.len() as isize;
    // The true displacement, and the part of it the integer index can express.
    let delta = t_pos - s_pos;
    // ⚠ `delta.round()` vs `round(t) − round(s)` on the carrying arm is **conditioning, not
    // correctness**: either choice leaves `si = i − frac` at the same source position, it only
    // changes how big `frac` gets (≤0.5 vs ≤1.0) and therefore how far off-centre the sinc read
    // sits. Don't write a test asserting this line — it would be asserting a preference.
    let d = if frac_transport {
        delta.round() as isize
    } else {
        t_pos.round() as isize - s_pos.round() as isize
    };
    // What is left on the table. `frac_transport` applies it below, so it only accumulates into
    // the diagnostic when we are actually throwing it away.
    let frac = delta - d as f64;
    residual.push(if frac_transport { 0.0 } else { delta - (t_pos.round() - s_pos.round()) });
    for (w0, w1, rise) in [(-lw, 0.0, true), (0.0, rw, false)] {
        let i0 = (s_pos + w0).round() as isize;
        let i1 = (s_pos + w1).round() as isize;
        if i1 <= i0 {
            continue;
        }
        let len = (i1 - i0) as f64;
        for i in i0..i1 {
            let ti = i + d;
            if ti < 0 || ti >= n {
                continue;
            }
            let ph = ((i - i0) as f64 + 0.5) / len * std::f64::consts::PI;
            let w = if rise {
                0.5 * (1.0 - ph.cos())
            } else {
                0.5 * (1.0 + ph.cos())
            };
            // κ = 0 (formant_rate == 1) keeps the whole-sample path: no interpolation at all, so
            // the ratio-1.0 identity stays bit-exact. Only a non-zero formant move reads the
            // source at a stride, which is what scales the spectral envelope by that stride.
            // The source position that maps onto output `ti`. Integer transport reads `i`;
            // carrying the residual reads `i - frac` (⇒ `frac == 0` collapses to the same read,
            // which is why ratio 1.0 stays bit-exact either way).
            let si = i as f64 - if frac_transport { frac } else { 0.0 };
            let v = if formant_rate == 1.0 {
                if frac == 0.0 || !frac_transport {
                    // ⚠ Fast path kept for bit-exactness, NOT for speed. Removing it leaves the
                    // identity intact to 5.5e-18 (measured), but `assert_eq!` is a stronger gate
                    // than an epsilon and this line is what lets us keep it.
                    if i < 0 || i >= n {
                        continue;
                    }
                    f64::from(x[i as usize])
                } else {
                    sinc_read(x, si)
                }
            } else {
                // κ ≠ 1 scales the read stride about `s_pos`, which is what moves the spectral
                // envelope. The residual composes INSIDE that map — correcting it afterwards
                // would be corrected by the wrong factor.
                let sp = s_pos + (si - s_pos) * formant_rate;
                if sp < 0.0 || sp >= (n - 1) as f64 {
                    continue;
                }
                let k = sp.floor();
                let f = sp - k;
                let k = k as usize;
                f64::from(x[k]) * (1.0 - f) + f64::from(x[k + 1]) * f
            };
            acc[ti as usize] += v * w;
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
    psola_shift_formant(x, sample_rate, semitones, 0.0, f0_hz, f0_hop)
}

/// As [`psola_shift_diag`], but the formant envelope is additionally moved by
/// `formant_semitones` **relative to the input** — the same convention the Signalsmith arm used
/// (`FormantPin::semitones`), so the κ slider keeps its meaning across the engine change:
/// `formant_semitones = κ · semitones`, κ=0 keeps the source timbre (the default, and the arm
/// the user A/B'd), κ=1 makes the formants follow the pitch (the plain transpose / chipmunk).
///
/// Keeping the whole κ range inside ONE engine is deliberate: routing κ>0 to a second engine
/// would put a cliff in the middle of a user-facing slider and two engines on a shared surface
/// (`apply_inverse` serves score and cover, S85: "fixing A ≠ leaving B unharmed").
pub fn psola_shift_formant(
    x: &[f32],
    sample_rate: u32,
    semitones: f64,
    formant_semitones: f64,
    f0_hz: &[f32],
    f0_hop: usize,
) -> (Vec<f32>, PsolaDiagnostics) {
    psola_shift_opts(x, sample_rate, semitones, formant_semitones, f0_hz, f0_hop, false)
}

/// Same, with the S146g sub-sample transport switch.
/// ⚠ `frac_transport = false` is byte-for-byte the pre-S146g behaviour — production still runs
/// that arm until a blind test settles it (the rulers cannot: `TRANSPORT_SINC_HALF`).
#[allow(clippy::too_many_arguments)]
pub fn psola_shift_opts(
    x: &[f32],
    sample_rate: u32,
    semitones: f64,
    formant_semitones: f64,
    f0_hz: &[f32],
    f0_hop: usize,
    frac_transport: bool,
) -> (Vec<f32>, PsolaDiagnostics) {
    psola_shift_wsola(x, sample_rate, semitones, formant_semitones, f0_hz, f0_hop, frac_transport, 0.0)
}

/// S148 — additive: an optional **bounded waveform-similarity search on the SOURCE side** (WSOLA).
///
/// ## What it is for
///
/// Measured on real material at +7 st (akiko donor, the production caliber): **3.9–4.3 % of voiced
/// frames come out more than 4 dB below the input**, while praat on the *same input at the same
/// ratio* does that on **0.2 %** — a 20× gap, so it is not inherent to TD-PSOLA. Five candidate
/// causes were each killed by measurement: the input's own properties (f0-matched controls: every
/// difference ≈ 0), the analysis marks (99.82 % of spacings within 0.7–1.3 × the f0 period, **0.0 %
/// anomaly rate at the notch frames**), the grain displacement δ (`corr(Δlevel, |δ|) = +0.07`), the
/// fed-f0 source (score-parametric vs measured: 3.91 % vs 4.29 % — measured is no better), and the
/// island count (272 vs 88 islands, same notch rate).
///
/// What the diagnostics say is that **the window sum is intact where the notches are**
/// (`cola_gap_frac` = 0.0 %, `cola_w_median` = 1.000 on both arms) ⇒ the level is not lost to a
/// COLA hole, it is lost to **grain-to-grain signal cancellation**.
///
/// And there is a structural asymmetry behind that: [`max_correlation`] is used **only in the
/// analysis pass** (to place the marks). The synthesis pass places every grain *blindly* at
/// `src[k]` — nothing ever checks that the grain about to be added is in phase with what has
/// already accumulated. `wsola_frac > 0` adds exactly that check.
///
/// ## Contract
///
/// * `wsola_frac` is the search radius **as a fraction of the grain's left half-width**, so it is
///   always < one period and cannot alias onto the neighbouring pitch pulse.
/// * ⛔ **Only the SOURCE read position moves.** The synthesis pulse `tm` is untouched, so the
///   output pitch and the exact-length contract are structurally unaffected — moving `tm` instead
///   would jitter the pitch by the search radius.
/// * The search must **beat the unshifted position by a margin** to move, so ratio 1.0 (where the
///   accumulator already *is* the windowed input at that position) keeps `s == src[k]` and the
///   identity gate stays honest. `wsola_frac = 0.0` is byte-for-byte the pre-S148 behaviour.
#[allow(clippy::too_many_arguments)]
pub fn psola_shift_wsola(
    x: &[f32],
    sample_rate: u32,
    semitones: f64,
    formant_semitones: f64,
    f0_hz: &[f32],
    f0_hop: usize,
    frac_transport: bool,
    wsola_frac: f64,
) -> (Vec<f32>, PsolaDiagnostics) {
    psola_shift_locked(
        x, sample_rate, semitones, formant_semitones, f0_hz, f0_hop, frac_transport, wsola_frac, 0.0,
    )
}

/// S150 — additive: **phase-lock the analysis marks** onto the glottal pulses before synthesis.
///
/// `phase_lock` is the search radius in periods; **0.0 = off = byte-for-byte the pre-S150 arm**,
/// which is what production still runs until a blind test says otherwise (S146 protocol: blind
/// test first, flip after — and S148's WSOLA is why that protocol is not negotiable).
///
/// ## What it buys, measured
///
/// The defect is the one the user named from the waveform, and S148 traced it to a single input:
/// our marks have the right period and the wrong phase (see [`lock_phase`]). Locking them at
/// `0.45` closes essentially the whole gap to the "upper bound" arm (= our synthesis fed praat's
/// marks, the arm that won a blind group in S148's u1):
///
/// | | `[785]` | `[685]` | `[800]` | `[791]` | `[86]` |
/// |---|---|---|---|---|---|
/// | today | 9.49 | 5.38 | 5.25 | 4.99 | 1.14 |
/// | **locked 0.45** | **3.51** | **0.80** | **0.85** | **1.26** | 0.95 |
/// | upper bound (praat's marks) | 3.45 | 0.83 | 0.69 | 1.20 | 1.13 |
///
/// Population, not just the 5 registered notes — all 23 non-rest notes ≥0.8 s, "modulation this
/// process ADDED to the input", median/p90: today **+2.09 / +5.94 dB**, locked **+0.00 / +0.26**,
/// praat's marks **+0.02 / +0.35**. It holds across the whole shift range in both directions
/// (−7 −5 −2 +1 +3 +5 +7) and on a second material (the registered 东雪莲 fixture at +6, where
/// praat *is* a valid reference: injection +2.86 → +0.88 against praat's own +1.49).
///
/// The four registered rulers agree (goose +7, all 23 windows): envelope shift +0.30 → +0.20
/// (praat +0.20), peak correlation 0.976 → **0.981** (praat 0.979), voiced survival 87.4% →
/// **89.4%** (praat 89.5%), ΔHNR −1.58 → **−1.34** (praat −0.94), >4 kHz share unchanged.
///
/// ⚠ What is NOT settled: **whether it is audible**. The depth ruler has exactly one audibility
/// data point (S148 u1: ~2.7 dB heard, ≤0.46 dB not), from a single load-bearing group.
#[allow(clippy::too_many_arguments)]
pub fn psola_shift_locked(
    x: &[f32],
    sample_rate: u32,
    semitones: f64,
    formant_semitones: f64,
    f0_hz: &[f32],
    f0_hop: usize,
    frac_transport: bool,
    wsola_frac: f64,
    phase_lock: f64,
) -> (Vec<f32>, PsolaDiagnostics) {
    psola_shift_infra(
        x, sample_rate, semitones, formant_semitones, f0_hz, f0_hop, frac_transport, wsola_frac,
        phase_lock, false,
    )
}

/// S152 — additive: optionally **subtract the infrasonic baseline this process manufactures**.
///
/// `remove_infrasonic = false` is byte-for-byte the pre-S152 arm and is what production runs
/// until a blind test settles it (S146 protocol; S148's WSOLA is why that protocol is not
/// negotiable — it read 4.80 % → 0.38 % on its own ruler and was 3/3 rejected by ear).
///
/// ## What it is for
///
/// See [`PsolaDiagnostics::infrasonic_frac`] for the measurement and the mechanism. Short form:
/// an up-shift narrows every grain's read window to less than one period while keeping it centred
/// on a (phase-locked, i.e. peak-aligned) mark, that window's own mean is not zero, and
/// `wsum ≈ 1` lays it down as a wandering baseline — up to a third of a rescued note's energy at
/// −14 st. ⚠ It is NOT a property of asymmetric waveforms; a sine does it too (see the gate).
///
/// ## ⚠ What this is NOT
///
/// It is **not** an audibility fix: 97 % of the injected energy is below 5 Hz and removing all of
/// it moves the whole-song peak by −0.008 dB. Its two real payoffs are that every RMS-domain
/// ruler we own stops carrying it, and that the output stops riding a wandering baseline (which
/// is the shape the user named "波形甚是诡异" on `[685]`).
///
/// ## Contract
///
/// * The removal is a **linear, zero-phase, band-limited** subtraction — see
///   [`INFRASONIC_MA_MS`] for the measured response (≥100 Hz moves by ≤0.08 dB).
/// * It runs on the whole buffer, covered and pass-through alike, so it cannot manufacture a
///   discontinuity at the covered/uncovered boundary.
/// * ⛔ It is **not** bit-exact at ratio 1.0 — a linear filter never is. That is exactly why it
///   is off by default: `ratio_one_is_the_identity` is the cheapest non-self-certifying gate on
///   this whole line (it killed three designs in S146 that "looked right"), and it must keep
///   asserting `assert_eq!` on the production arm. The honest gate for the arm being ON is
///   `the_infrasonic_arm_leaves_everything_above_the_fundamental_alone`.
#[allow(clippy::too_many_arguments)]
pub fn psola_shift_infra(
    x: &[f32],
    sample_rate: u32,
    semitones: f64,
    formant_semitones: f64,
    f0_hz: &[f32],
    f0_hop: usize,
    frac_transport: bool,
    wsola_frac: f64,
    phase_lock: f64,
    remove_infrasonic: bool,
) -> (Vec<f32>, PsolaDiagnostics) {
    let n = x.len();
    let mut diag = PsolaDiagnostics::default();
    let mut residual = ResidualStat::default();
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
    if !formant_semitones.is_finite() {
        return (x.to_vec(), diag);
    }
    // Exactly 1.0 for κ=0 so the whole-sample (bit-exact) grain path is taken.
    let formant_rate = if formant_semitones == 0.0 {
        1.0
    } else {
        2f64.powf(formant_semitones / 12.0)
    };
    if !(formant_rate.is_finite() && formant_rate > 0.0) {
        return (x.to_vec(), diag);
    }
    let mean = x.iter().map(|v| f64::from(*v)).sum::<f64>() / n as f64;
    let dc_free: Vec<f32> = x.iter().map(|v| (f64::from(*v) - mean) as f32).collect();

    let mut acc = vec![0.0f64; n];
    let mut wsum = vec![0.0f64; n];
    let mut covered = vec![false; n];
    let max_period = MAX_PERIOD_SECONDS * sr;
    // S151 源覆盖率的累加器(见 `PsolaDiagnostics::src_uncovered_frac`)。
    let (mut uncovered, mut span) = (0.0f64, 0.0f64);

    for (a, b) in voiced_islands(f0_hz, f0_hop, n, (MIN_ISLAND_SECONDS * sr) as usize) {
        let mut src = analysis_marks(&dc_free, sample_rate, f0_hz, f0_hop, a, b);
        if src.len() < 3 {
            continue;
        }
        // S150 — marks are found on the DC-free signal, so they are locked on it too.
        diag.marks_locked += lock_phase(&dc_free, &mut src, phase_lock);
        diag.islands += 1;
        diag.marks += src.len();
        let last = (src.len() - 1) as f64;
        let count = (last * ratio) as usize;
        let (mut island_first, mut cover_end) = (f64::NAN, f64::NAN);
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
            // S148 WSOLA(默认 0 = 关,生产逐位不变):只挪【源】读点,不挪合成脉冲 tm。
            let s_pos = if wsola_frac > 0.0 && i > 0 {
                wsola_pick(x, &acc, src[k], tm, lw, wsola_frac * lw)
            } else {
                src[k]
            };
            if s_pos != src[k] {
                diag.wsola_moved += 1;
            }
            #[cfg(test)]
            if grain_trace_on() {
                let t_src = if k + 1 < src.len() { src[k + 1] - src[k] } else { src_l };
                GRAIN_TRACE.lock().unwrap().push([
                    tm,                       // 这颗粒被放在哪(样本)
                    s_pos,                    // 从哪读的
                    tm - s_pos,               // δ:自身脉冲点与放置点的偏差
                    t_src,                    // 该处的源周期
                    (tm - s_pos) / t_src,     // δ 归一到周期 = 相位误差
                    lw,
                    rw,                       // ⚠ 重放 OLA 必须要它:窗是 [s−lw, s) ∪ [s, s+rw)
                    k as f64,
                ]);
            }
            // S151 —— **源覆盖率**:这一颗粒从源上读的是 `[s_pos − lw, s_pos + rw)`。上移时
            // `lw = rw = T_src / ratio`(上面那两个 `min` 取的是目标邻距),所以一旦
            // `ratio > 2`(= |位移| > 12 半音),相邻两个读窗之间就留下一段**永远不进任何颗粒**
            // 的源波形。见 `SRC_UNCOVERED` 的注释:这是唯一直接看得见它的读数。
            let (rs, re) = (s_pos - lw, s_pos + rw);
            if cover_end.is_nan() {
                island_first = rs;
                cover_end = rs;
            }
            if rs > cover_end {
                uncovered += rs - cover_end;
            }
            cover_end = cover_end.max(re);
            add_bell(
                x, &mut acc, &mut wsum, s_pos, tm, lw, rw, formant_rate, frac_transport,
                &mut residual,
            );
        }
        if !cover_end.is_nan() {
            span += (cover_end - island_first).max(0.0);
        }
    }

    let mut gap = 0usize;
    let mut cov_n = 0usize;
    let mut ws: Vec<f64> = Vec::new();
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        // ⛔ Statistics read the RAW sum; the clamp below is only what the dry-fill gain needs.
        // Clamping first made the surplus (w > 1) unreadable — see PsolaDiagnostics.
        let raw = wsum[i];
        let w = raw.clamp(0.0, 1.0);
        if covered[i] {
            cov_n += 1;
            if raw < 0.9 {
                gap += 1;
            }
            ws.push(raw);
            out[i] = acc[i] as f32;
        } else {
            // copyFlat: outside the synthesized span the un-shifted input rides the window-sum
            // ramp, which IS the crossfade. Inside it, a shortfall is a defect and must not be
            // papered over with un-shifted audio (that is beating, not repair).
            // ⚠ The clamp on `w` here is DEFENSIVE, not load-bearing: measured across every
            // fixture in this file, zero uncovered samples ever carry wsum > 1 (a surplus needs
            // overlapping bells, and overlapping bells mean covered). Feeding the raw sum instead
            // is bit-identical on real input — so do not expect a test to catch that swap.
            out[i] = (acc[i] + (1.0 - w) * f64::from(x[i])) as f32;
        }
    }
    // S152 —— **读数无条件算,修法才由旋钮控**。这样今天的生产日志里就能看见它,而输出逐位不变。
    // ⛔ 这一条是从 S147 那次「收益静默减半」学来的:一个只在改动打开时才存在的读数,
    // 没法用来判断「改动之前是什么样」。
    {
        /// 输出里 <~50 Hz 的能量占比。
        fn infra_frac(y: &[f32], sample_rate: u32) -> f64 {
            let lf = infrasonic_baseline(y, sample_rate);
            let (mut e_lf, mut e_all) = (0.0f64, 0.0f64);
            for (o, l) in y.iter().zip(lf.iter()) {
                e_all += f64::from(*o) * f64::from(*o);
                e_lf += l * l;
            }
            if e_all > 0.0 {
                e_lf / e_all
            } else {
                0.0
            }
        }
        let before = infra_frac(&out, sample_rate);
        diag.infrasonic_frac = before as f32;
        if remove_infrasonic {
            let lf = infrasonic_baseline(&out, sample_rate);
            for (o, l) in out.iter_mut().zip(lf.iter()) {
                *o = (f64::from(*o) - *l) as f32;
            }
            // 报「真的拿掉了多少」而不是「开着」—— 与 `wsola_moved` / `marks_locked` 同一条规矩。
            diag.infrasonic_removed = (before - infra_frac(&out, sample_rate)) as f32;
        }
    }
    diag.transport_residual_rms = residual.rms();
    diag.src_uncovered_frac = if span > 0.0 { (uncovered / span) as f32 } else { 0.0 };
    diag.cola_gap_frac = if cov_n > 0 { gap as f32 / cov_n as f32 } else { 0.0 };
    if ws.is_empty() {
        diag.cola_w_median = 1.0;
        diag.cola_w_p01 = 1.0;
        diag.cola_w_p99 = 1.0;
    } else {
        diag.cola_over_frac = ws.iter().filter(|&&w| w > 1.05).count() as f32 / ws.len() as f32;
        ws.sort_by(f64::total_cmp);
        let at = |q: f64| ws[(((ws.len() - 1) as f64) * q).round() as usize] as f32;
        diag.cola_w_median = at(0.50);
        diag.cola_w_p01 = at(0.01);
        diag.cola_w_p99 = at(0.99);
    }
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

    /// S151 —— 上移超过一个八度时,源波形有一整段**从来不被任何颗粒读到**,而仓里
    /// 在这条判据之前**没有任何东西看得见它**:`cola_*` 是在输出域算的(半窗按构造等于目标
    /// 邻距,实测 +7..+16 全是 0.00% / 1.000),标记层的尺子全部 ratio 不变量。
    /// 阈值不是估的:读窗半宽 = `T_src/ratio`,相邻源标记相距 `T_src` ⇒ `ratio > 2` 才留缝。
    #[test]
    fn nothing_reads_part_of_the_source_once_the_shift_passes_an_octave() {
        let sr = 44_100;
        let f0 = 220.0;
        let x = voiced(sr, 0.5, f0);
        let hop = sr as usize / 200;
        let f0t = flat_f0(x.len(), hop, f0 as f32);
        let frac = |st: f64| {
            let (_, d) = psola_shift_diag(&x, sr, st, &f0t, hop);
            assert!(d.islands > 0, "夹具必须真的是浊音");
            d.src_uncovered_frac
        };
        // 阴性对照(两条,缺一条这条判据就可能只是「一个恒为正的数」):
        // ⚠ 判据用 1e-9 而不是 == 0.0:ratio ≤ 2 时相邻读窗**正好相接**,读数是浮点噪声
        // (实测 0 / 0 / 1.06e-15),不是真的缝。阈值仍然比阳性那一档小 8 个数量级。
        for st in [0.0, 7.0, 12.0] {
            let f = frac(st);
            assert!(f < 1e-9, "{st:+} st 上读窗必须铺满源,读到 {f}");
        }
        // 阳性:用户 2026-08-18 实机真的跑到了 −14 ⇒ 逆变换 +14。
        let f14 = frac(14.0);
        assert!(
            (0.05..0.20).contains(&f14),
            "+14 上必须读出一段没人读过的源(实测真素材 10.2%),读到 {f14}"
        );
        assert!(frac(16.0) > f14, "越深漏得越多");
    }

    #[test]
    fn the_window_sum_diagnostics_can_see_a_surplus_not_only_a_shortfall() {
        // ⛔ The empty criterion this replaces: the stats were computed from a **clamped** window
        // sum, so `cola_w_median` was structurally ≤ 1.000 and the overlap SURPLUS — which only
        // occurs when shifting UP, i.e. the direction production actually runs — could not be
        // expressed at all. A diagnostic that cannot represent the failure it watches for tells
        // you nothing when it reads "fine".
        let sr = 44_100;
        let f0 = 220.0;
        let x = voiced(sr, 0.5, f0);
        let hop = sr as usize / 200;
        let f0t = flat_f0(x.len(), hop, f0 as f32); // 空 f0 轨 ⇒ 零 island,那样测不到任何东西
        let (_, up) = psola_shift_diag(&x, sr, 7.0, &f0t, hop);
        assert!(up.islands > 0, "the fixture must actually be voiced");

        // ⛔ STRICTLY ordered. `p01 <= median <= p99` survives a p99 that has silently collapsed
        // onto the median (mutation-checked: that version went green on the loose form), so the
        // three readouts have to be shown to be three readouts.
        assert!(
            up.cola_w_p01 < up.cola_w_median && up.cola_w_median < up.cola_w_p99,
            "p01/median/p99 = {}/{}/{} — these must be three distinct readouts",
            up.cola_w_p01,
            up.cola_w_median,
            up.cola_w_p99
        );
        assert!(
            up.cola_w_p99 > 1.0,
            "p99 {} — a clamped statistic can never exceed 1.0, which is exactly the bug",
            up.cola_w_p99
        );
        assert!(up.cola_w_p99 < 2.0, "…but a surplus this large would be a real defect");
        assert!((0.0..=1.0).contains(&up.cola_over_frac));

        // And the audio is untouched by this: identity still holds bit-for-bit.
        let (id, _) = psola_shift_diag(&x, sr, 0.0, &f0t, hop);
        assert_eq!(id, x, "ratio 1.0 must remain the identity");
    }

    #[test]
    fn the_discarded_transport_residual_is_zero_at_ratio_one_and_flat_everywhere_else() {
        // ⭐ S146g's load-bearing measurement, as a criterion. Whole-sample transport takes
        // `round(t) − round(s)` — a difference of TWO independent roundings — so every grain
        // drops a sub-sample residual. Its RMS is the shape of the toll we could not explain:
        // exactly 0 at ratio 1.0, and then a CONSTANT ≈0.41 samples at any other ratio.
        // That constant is why entering the process once costs −0.59 dB while +7→+8 adds only
        // −0.37: depth changes the grain count, not the rate.
        let sr = 44_100;
        let f0 = 220.0;
        let x = voiced(sr, 0.5, f0);
        let hop = sr as usize / 200;
        let f0t = flat_f0(x.len(), hop, f0 as f32);

        let (_, id) = psola_shift_diag(&x, sr, 0.0, &f0t, hop);
        assert_eq!(
            id.transport_residual_rms, 0.0,
            "ratio 1.0 discards nothing — the identity is structural, not a short-circuit"
        );

        // ⚠ The residual is `(t − round(t)) − (s − round(s))` — the difference of two independent
        // rounding errors. If those were uniform and independent the RMS would be the triangular
        // value √(2/12) = 0.408, which is exactly what S146g read on the registered material
        // (0.4139 / 0.4129 / 0.4121 at +1 / +6 / +8). This synthetic fixture has a near-constant
        // period, so its two fractional parts are correlated and it reads higher (~0.52-0.60).
        // ⇒ **The criterion is the SHAPE, not the constant.** Pinning the constant here would be
        // over-fitting to one fixture; pinning the shape is what the diagnosis rests on.
        let mut seen = vec![];
        for st in [1.0, 3.0, 6.0, 8.0] {
            let (_, d) = psola_shift_diag(&x, sr, st, &f0t, hop);
            seen.push(d.transport_residual_rms);
            assert!(
                (0.25..0.75).contains(&d.transport_residual_rms),
                "{st} st: a two-rounding residual has to land near √(2/12)=0.41, got {}",
                d.transport_residual_rms
            );
        }
        // ⛔ FLAT, not growing. A depth-proportional residual would mean the toll compounds, and
        // then both the diagnosis and the fix are aimed at the wrong thing. (+8 vs +1 measures
        // 1.14× here; anything approaching proportionality would be several ×.)
        let (lo, hi) = seen.iter().fold((f32::MAX, 0.0f32), |(a, b), &v| (a.min(v), b.max(v)));
        assert!(hi / lo < 1.5, "residual must not track depth, got {seen:?}");
    }

    #[test]
    fn carrying_the_residual_removes_it_without_touching_the_identity() {
        // ⛔⛔ THE criterion the recon insisted on, and the reason it exists: the existing
        // identity gate is **structurally blind** to this code. At ratio 1.0 the fractional
        // branch executes ZERO times (delta ≡ 0 ⇒ the fast path takes it), so
        // `ratio_one_is_the_identity` would stay green on a completely broken interpolator —
        // the same shape as putting a gate on `apply_inverse`, which returns early at shift 0.
        // ⇒ assert the interpolator itself, and assert the residual is actually gone.
        let sr = 44_100;
        let f0 = 220.0;
        let x = voiced(sr, 0.5, f0);
        let hop = sr as usize / 200;
        let f0t = flat_f0(x.len(), hop, f0 as f32);

        // ⑴ The interpolator at an integer position IS the sample (this is what makes the
        //    identity survive even with the fast path removed — measured 5.5e-18 by S146g).
        for i in [40usize, 137, 4096, 9999] {
            assert!(
                (sinc_read(&x, i as f64) - f64::from(x[i])).abs() < 1e-9,
                "sinc_read at integer {i} must return the sample itself"
            );
        }

        // ⑵ Carrying the residual zeroes the thing it is meant to zero…
        for st in [1.0, 6.0, 8.0] {
            let (_, off) = psola_shift_opts(&x, sr, st, 0.0, &f0t, hop, false);
            let (_, on) = psola_shift_opts(&x, sr, st, 0.0, &f0t, hop, true);
            assert!(off.transport_residual_rms > 0.25, "{st} st: baseline must drop a residual");
            assert_eq!(on.transport_residual_rms, 0.0, "{st} st: carried ⇒ nothing discarded");
            assert_eq!(on.islands, off.islands, "{st} st: the mark layer must not move");
        }

        // ⑶ …and ratio 1.0 stays BIT-exact on both arms.
        for frac in [false, true] {
            let (y, _) = psola_shift_opts(&x, sr, 0.0, 0.0, &f0t, hop, frac);
            assert_eq!(y, x, "ratio 1.0 must be the identity with frac_transport = {frac}");
        }
    }

    #[test]
    fn the_fractional_arm_is_opt_in_and_production_is_byte_for_byte_unchanged() {
        // ⚠ Additive, per S146's protocol: the rulers cannot settle whether this SOUNDS better
        // (S146g measured ΔHNR ranking the praat gold standard BELOW two arms already condemned
        // by ear), so nothing the user hears may move until a blind test says so.
        let sr = 44_100;
        let f0 = 260.0;
        let x = voiced(sr, 0.4, f0);
        let hop = sr as usize / 200;
        let f0t = flat_f0(x.len(), hop, f0 as f32);
        for st in [-6.0, -1.0, 1.0, 6.0, 8.0] {
            let (legacy, _) = psola_shift_opts(&x, sr, st, 0.0, &f0t, hop, false);
            let (via_public, _) = psola_shift_diag(&x, sr, st, &f0t, hop);
            assert_eq!(legacy, via_public, "{st} st: the default entry must be the legacy arm");
            let (frac, _) = psola_shift_opts(&x, sr, st, 0.0, &f0t, hop, true);
            assert_ne!(frac, legacy, "{st} st: …and the opt-in arm must actually differ");
        }
    }


    /// |mean| / RMS — an **independent** proxy for the manufactured baseline.
    /// ⛔ Deliberately not `infrasonic_baseline`: measuring a filter with itself is the shape of a
    /// criterion that cannot fail. On a stationary fixture the injection is essentially pure DC,
    /// so the plain mean reads it exactly and owes nothing to the implementation.
    /// ⚠ An earlier version of this helper used the RMS of 20 ms block means and had a **24 %
    /// floor** on a 220 Hz fixture (a block holds a non-integer number of periods), i.e. it could
    /// not have failed for the right reason. Kept as a note because that floor looked like a
    /// finding for about a minute.
    fn baseline_rms(x: &[f32], _sr: u32) -> f64 {
        (x.iter().map(|v| f64::from(*v)).sum::<f64>() / x.len().max(1) as f64).abs()
    }

    /// A glottal-pulse-like **asymmetric** source whose every period integrates to exactly zero:
    /// an instantaneous jump followed by an exponential decay, minus that period's own mean.
    /// ⇒ any baseline in the OUTPUT was manufactured by the process, not carried in.
    fn asym_pulses(sr: u32, secs: f64, period: usize) -> Vec<f32> {
        let p = period.max(8);
        // ⛔ Whole periods only. 44100 samples of a 200-sample period is 220.5 of them, and that
        // half period puts a mean of 0.0014 into the FIXTURE — which is exactly the quantity the
        // test is about, so it has to be zero by construction, not by luck.
        let n = ((f64::from(sr) * secs) as usize / p) * p;
        let mut cycle: Vec<f64> =
            (0..p).map(|i| (-(i as f64) / (p as f64 * 0.15)).exp()).collect();
        let m = cycle.iter().sum::<f64>() / p as f64;
        for v in cycle.iter_mut() {
            *v -= m;
        }
        (0..n).map(|i| cycle[i % p] as f32).collect()
    }

    fn rms(x: &[f32]) -> f64 {
        (x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / x.len().max(1) as f64).sqrt()
    }

    /// Magnitude of a single frequency (a Goertzel-style projection), normalised by length.
    fn tone_mag(x: &[f32], sr: u32, f: f64) -> f64 {
        let w = 2.0 * std::f64::consts::PI * f / f64::from(sr);
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for (i, v) in x.iter().enumerate() {
            let p = w * i as f64;
            re += f64::from(*v) * p.cos();
            im += f64::from(*v) * p.sin();
        }
        (re * re + im * im).sqrt() / x.len() as f64
    }

    /// A pure sine — the SYMMETRIC control. Every period integrates to zero and so does every
    /// window centred anywhere in it, so this source must NOT produce the baseline.
    fn sine(sr: u32, secs: f64, f0: f64) -> Vec<f32> {
        let n = (f64::from(sr) * secs) as usize;
        (0..n)
            .map(|i| {
                (2.0 * std::f64::consts::PI * f0 * i as f64 / f64::from(sr)).sin() as f32
            })
            .collect()
    }

    #[test]
    fn the_infrasonic_arm_is_opt_in_and_the_default_arm_is_byte_for_byte_unchanged() {
        // S146 protocol, same as `frac_transport` / `wsola` / `phase_lock` before it: nothing the
        // user hears moves until a blind test says so. ⚠ Both directions — a criterion that only
        // covers the up-shift is how a −12 st arm once returned its input bit-for-bit while four
        // rulers read "perfect" (S146).
        let sr = 44_100;
        let f0 = 220.0;
        let (x, _) = pulses(sr, 1.0, |_| f0, |_| 1.0);
        let hop = sr as usize / 200;
        let f0t = flat_f0(x.len(), hop, f0 as f32);
        for st in [-12.0, -7.0, -1.0, 1.0, 7.0, 12.0, 14.0] {
            let (legacy, dl) = psola_shift_locked(&x, sr, st, 0.0, &f0t, hop, false, 0.0, 0.30);
            let (via_deep, dd) =
                psola_shift_infra(&x, sr, st, 0.0, &f0t, hop, false, 0.0, 0.30, false);
            assert_eq!(legacy, via_deep, "{st} st: the 9-arg entry must stay the legacy arm");
            assert_eq!(dl, dd, "{st} st: …diagnostics included");
            // ⛔ And the readout must EXIST on the arm that is off — otherwise "what does it look
            // like today" is unanswerable without shipping the change (S147's silent-halving).
            assert!(
                dl.infrasonic_frac.is_finite(),
                "{st} st: the readout must be computed unconditionally"
            );
            assert_eq!(dl.infrasonic_removed, 0.0, "{st} st: nothing removed while off");

            let (on, don) = psola_shift_infra(&x, sr, st, 0.0, &f0t, hop, false, 0.0, 0.30, true);
            assert_ne!(on, legacy, "{st} st: …and the opt-in arm must actually differ");
            assert!(
                don.infrasonic_removed > 0.0,
                "{st} st: the arm reports it removed nothing — 'on' and 'did something' are \
                 different facts (removed {})",
                don.infrasonic_removed
            );
        }
    }

    #[test]
    fn the_manufactured_baseline_is_the_narrowed_grain_window_s_own_mean() {
        // ⭐ The mechanism, on synthetic material where the answer is computable in closed form.
        //
        // ⛔ This test started life asserting something ELSE — "it needs an asymmetric pulse, a
        // symmetric source produces none" — and killed it on the first run: a plain sine at the
        // same ratio injects **more** (0.195 vs 0.104). Asymmetry is not the variable. What is:
        // on an up-shift both half-widths collapse to `T_src / ratio` (`:1027-1028` takes the
        // min of target and source spacing), so every grain reads a window **narrower than one
        // period, centred on the mark** — and the marks are phase-locked onto the energy peaks.
        // The bell-weighted mean of a sub-period window centred on a peak is not zero, `wsum ≈ 1`
        // lays that same mean down under every grain, and the result is a baseline. Symmetric or
        // not is irrelevant; being centred on a peak is the whole of it.
        let sr = 44_100;
        let period = 200usize; // 220.5 Hz — an integer period so "zero mean per period" is exact
        let f0 = f64::from(sr) / period as f64;
        let hop = sr as usize / 200;
        let asym = asym_pulses(sr, 1.0, period);
        let sym: Vec<f32> = sine(sr, 1.0, f0)[..asym.len()].to_vec();
        let f0t = flat_f0(asym.len(), hop, f0 as f32);

        // Ratio 1.0 is the identity ⇒ the window IS a whole period ⇒ nothing added.
        let (id, _) = psola_shift_locked(&asym, sr, 0.0, 0.0, &f0t, hop, false, 0.0, 0.30);
        assert_eq!(id, asym, "ratio 1.0 must still be the identity");
        let src_ratio = baseline_rms(&asym, sr) / rms(&asym);
        assert!(
            src_ratio < 1e-5,
            "the fixture itself carries a baseline ({src_ratio:.7}) — then nothing below can be              attributed to the process"
        );

        for (name, x, peak) in [("asym", &asym, 0usize), ("sine", &sym, period / 4)] {
            let mut last = 0.0f64;
            for st in [4.0f64, 8.0, 12.0, 16.0] {
                let (out, diag) = psola_shift_locked(x, sr, st, 0.0, &f0t, hop, false, 0.0, 0.30);
                let got = baseline_rms(&out, sr) / rms(&out);
                assert!(
                    got > last,
                    "{name} +{st} st: the injection must grow with the ratio ({last:.5} → {got:.5})"
                );
                last = got;
                assert!(
                    diag.infrasonic_frac > 0.001,
                    "{name} +{st} st: the readout should see it (read {})",
                    diag.infrasonic_frac
                );

                // ⭐ THE mechanism gate: predict the baseline from the narrowed window alone.
                // Half-width `T_src / ratio`, Hann rise on the left of the mark and Hann fall on
                // the right — exactly `add_bell`'s window — over the source centred on the peak.
                let half = period as f64 / 2f64.powf(st / 12.0);
                let h = half.round() as isize;
                let (mut num, mut den) = (0.0f64, 0.0f64);
                for j in -h..h {
                    let len = half;
                    let ph = ((if j < 0 { j + h } else { j } as f64) + 0.5) / len
                        * std::f64::consts::PI;
                    let w = if j < 0 { 0.5 * (1.0 - ph.cos()) } else { 0.5 * (1.0 + ph.cos()) };
                    let idx = (peak as isize + j).rem_euclid(period as isize) as usize;
                    num += f64::from(x[idx]) * w;
                    den += w;
                }
                let want = (num / den).abs() / rms(x);
                assert!(
                    (got / want.max(1e-9)).max(want.max(1e-9) / got) < 3.0,
                    "{name} +{st} st: measured baseline {got:.5} vs the grain window's own mean                      {want:.5} — off by more than 3× means the cause is NOT the narrowed window                      and the note on `infrasonic_frac` is wrong"
                );
            }
        }
    }

    #[test]
    fn the_infrasonic_arm_removes_the_baseline_without_touching_the_fundamental() {
        // The honest gate for the arm being ON. It cannot be `assert_eq!` at ratio 1.0 — a linear
        // filter is never bit-exact — so the contract it must meet instead is stated as a bound
        // on where it is allowed to act: **below the fundamental and nowhere else.**
        let sr = 44_100;
        let f0 = 220.0;
        let hop = sr as usize / 200;
        let (x, _) = pulses(sr, 1.0, |_| f0, |_| 1.0);
        let f0t = flat_f0(x.len(), hop, f0 as f32);
        for st in [-12.0f64, -7.0, 7.0, 12.0, 14.0] {
            let (off, doff) = psola_shift_locked(&x, sr, st, 0.0, &f0t, hop, false, 0.0, 0.30);
            let (on, don) = psola_shift_infra(&x, sr, st, 0.0, &f0t, hop, false, 0.0, 0.30, true);

            let b_off = baseline_rms(&off, sr) / rms(&off);
            let b_on = baseline_rms(&on, sr) / rms(&on);
            assert!(
                b_on < b_off * 0.25,
                "{st} st: the baseline should be mostly gone ({b_off:.5} → {b_on:.5})"
            );
            assert!(
                don.infrasonic_frac <= doff.infrasonic_frac + 1e-6,
                "{st} st: the BEFORE readout must not depend on the arm"
            );

            // ⛔ THE bound that makes this safe: the fundamental and its first harmonics are the
            // one band a low-cut must not reach, and a donor's f0 is 150-500 Hz — close enough to
            // 50 Hz that "obviously fine" is not an argument.
            let out_f0 = f0 * 2f64.powf(st / 12.0);
            for h in [1.0f64, 2.0, 3.0] {
                let f = out_f0 * h;
                if f >= f64::from(sr) / 2.0 {
                    continue;
                }
                let (a, b) = (tone_mag(&off, sr, f), tone_mag(&on, sr, f));
                let d = 20.0 * (b.max(1e-12) / a.max(1e-12)).log10();
                assert!(
                    d.abs() < 0.15,
                    "{st} st: harmonic {h} at {f:.0} Hz moved {d:+.3} dB — the removal is \
                     supposed to be band-limited well below it"
                );
            }
        }
    }

    #[test]
    fn the_interpolator_reproduces_a_known_signal_between_samples() {
        // ⛔ THE gate that stops `sinc_read` being quietly swapped for linear interpolation —
        // which S146g specifically warned about, because linear reads BETTER on the HNR ruler
        // purely by low-passing (+0.43…+2.23 dB, and a pure-low-pass control gets +0.40 on its
        // own). Ground truth is analytic, so no ruler and no taste is involved.
        let sr = 44_100u32;
        for fq in [1000.0f64, 5000.0, 8000.0] {
            let x: Vec<f32> = (0..4096)
                .map(|i| (2.0 * std::f64::consts::PI * fq * f64::from(i) / f64::from(sr)).sin() as f32)
                .collect();
            let mut worst = 0.0f64;
            for i in 100..3900 {
                for f in [0.25, 0.5, 0.75] {
                    let want =
                        (2.0 * std::f64::consts::PI * fq * (f64::from(i) + f) / f64::from(sr)).sin();
                    worst = worst.max((sinc_read(&x, f64::from(i) + f) - want).abs());
                }
            }
            // Measured 1.1e-5 (1k) · 1.0e-5 (5k) · 3.1e-5 (8k). Linear interpolation at 8 kHz is
            // off by ~1e-1 — three to four orders of magnitude, so this bound is not delicate.
            assert!(worst < 1e-3, "{fq} Hz: interpolation error {worst:.6} is not band-limited");
        }
    }

    #[test]
    fn the_grain_is_actually_read_at_the_fractional_position() {
        // ⛔⛔ The wiring gate. Without it, `si = i as f64` (residual computed, interpolator
        // present, and simply not used) stays GREEN on every other criterion here — measured:
        // the output still differs from the legacy arm, because `d` changes too. "Different"
        // is not "correct", and this is the shape that mutation exposed.
        //
        // One bell maps each source index to exactly one output index, so `acc/wsum` at a covered
        // output sample IS the value that was read for it — comparable against ground truth.
        let x: Vec<f32> = (0..600)
            .map(|i| (f64::from(i) * 0.037).sin() as f32 * 0.5 + (f64::from(i) * 0.31).cos() as f32 * 0.2)
            .collect();
        let (s_pos, t_pos) = (300.0f64, 300.5f64);
        let delta = t_pos - s_pos;

        for frac_transport in [false, true] {
            let mut acc = vec![0.0f64; x.len()];
            let mut wsum = vec![0.0f64; x.len()];
            let mut res = ResidualStat::default();
            add_bell(&x, &mut acc, &mut wsum, s_pos, t_pos, 12.0, 12.0, 1.0, frac_transport, &mut res);

            let ti = 305usize; // inside the bell, away from its zero-weight edges
            assert!(wsum[ti] > 1e-3, "the probe index must be covered");
            let got = acc[ti] / wsum[ti];
            if frac_transport {
                let want = sinc_read(&x, f64::from(ti as u32) - delta);
                assert!(
                    (got - want).abs() < 1e-9,
                    "carrying arm must read the source at {} — got {got}, want {want}",
                    f64::from(ti as u32) - delta
                );
                // …and that read must NOT be the whole-sample one, or the arm does nothing.
                assert!(
                    (got - f64::from(x[ti - 1])).abs() > 1e-6,
                    "the fractional offset was computed and then ignored"
                );
            } else {
                assert!(
                    (got - f64::from(x[ti - 1])).abs() < 1e-9,
                    "legacy arm must read whole samples — got {got}"
                );
            }
        }
    }

    /// A pulse train with a controllable per-pulse gain: pulse `i` gets `gain(i)`.
    /// ⛔ Deliberately NOT the `voiced()` fixture — that one is a sum of cosines whose marks are
    /// already on the pulses, so it cannot express "the mark is off the pulse", which is the whole
    /// subject here. Ground truth (where the pulses are) is by construction.
    /// `f0(k)` lets the period MOVE inside the island — a constant-period fixture cannot tell a
    /// local radius from an island-wide one, and a mutation swapping them then goes green.
    fn pulses(
        sr: u32,
        secs: f64,
        f0: impl Fn(usize) -> f64,
        gain: impl Fn(usize) -> f64,
    ) -> (Vec<f32>, Vec<f64>) {
        let n = (f64::from(sr) * secs) as usize;
        let mut y = vec![0.0f32; n];
        let mut at = Vec::new();
        let mut c = f64::from(sr) / f0(0) * 0.5;
        let mut k = 0usize;
        loop {
            let p = f64::from(sr) / f0(k);
            if c as usize + 2 * p as usize >= n {
                break;
            }
            at.push(c);
            let g = gain(k);
            // A damped ring evaluated at the FRACTIONAL distance from the pulse, so the true peak
            // sits between samples. ⛔ An integer-aligned fixture makes the sub-sample refinement
            // invisible (mutation ⑤ went green on one).
            for i in 0..(p * 0.9) as usize {
                let idx = c as usize + i;
                let t = idx as f64 - c;
                if t < 0.0 || idx >= n {
                    continue;
                }
                let v = g * (-t / (p * 0.12)).exp() * (2.0 * std::f64::consts::PI * 3.5 * t / p).sin();
                y[idx] += v as f32;
            }
            c += p;
            k += 1;
        }
        (y, at)
    }

    #[test]
    fn the_phase_lock_pulls_marks_onto_the_pulses_and_keeps_every_one() {
        // ⭐ THE criterion for S150. The defect it fixes is not "the period is wrong" (ours is
        // right: spacing median 120.00 against praat's 119.75) but "the marks sit at the wrong
        // PHASE inside the period" (measured scatter ±0.42 of a period, landing energy 2.3-4.4 dB
        // below praat's). So the assertion has to be about phase, against ground truth.
        let sr = 44_100u32;
        let (x, truth) = pulses(sr, 0.5, |_| 300.0, |_| 1.0);
        let p = f64::from(sr) / 300.0;

        // Marks with the right PERIOD and a drifting PHASE — exactly the shape the correlation
        // walk produces (it steps by one period and never re-anchors).
        // ⚠ The drift RATE is the measured one, not a round number: the walk's per-step phase error
        // is **0.0024 of a period** (median), which is why the accumulated error reaches half a
        // period only after hundreds of steps. That number is load-bearing here, because a
        // first-order loop has a steady-state lag of drift/β for a ramp — an invented 0.02/step
        // fixture demands 8× the loop bandwidth reality does, and fails a correct implementation.
        let mut marks: Vec<f64> = truth
            .iter()
            .enumerate()
            .map(|(i, t)| t + p * (0.40 - 0.0024 * i as f64).clamp(-0.42, 0.42))
            .collect();
        let before: Vec<f64> = marks.clone();

        // ⛔ Negative control FIRST: radius 0 must be a no-op, and must say so.
        let mut untouched = marks.clone();
        assert_eq!(lock_phase(&x, &mut untouched, 0.0), 0, "radius 0 must move nothing");
        assert_eq!(untouched, before, "radius 0 must leave the marks bit-identical");

        let moved = lock_phase(&x, &mut marks, 0.45);
        assert!(moved > truth.len() / 2, "the lock must actually move marks, moved {moved}");
        assert_eq!(marks.len(), before.len(), "design note 3: no mark may be added or dropped");
        assert!(marks.windows(2).all(|w| w[1] > w[0]), "marks must stay strictly increasing");

        // ⚠ The criterion is the SPREAD of the phase, not its distance to the pulse onset.
        // The energy argmax of a one-sided pulse sits a constant ~T/8 after the onset (measured),
        // and that bias is harmless — every mark gets the same one, and what the synthesis needs is
        // a *consistent* phase reference. Asserting distance-to-truth instead fails a correct
        // implementation for having the bias, which is the same mistake as the first version of the
        // sub-sample assertion below.
        // ⚠ …and measured over the TRACKING regime, not the acquisition transient: a loop with gain
        // β needs ~3/β marks to pull in from a cold start (here 30), and the seed mark is by
        // construction wherever the walk left it. Skipping the first 40 is not a convenience — a
        // criterion that includes the pull-in measures the initial offset, which nothing controls.
        const SKIP: usize = 40;
        let spread = |m: &[f64]| -> f64 {
            let mut e: Vec<f64> =
                m.iter().zip(&truth).skip(SKIP).map(|(a, b)| (a - b) / p).collect();
            e.sort_by(f64::total_cmp);
            e[(e.len() as f64 * 0.9) as usize % e.len()] - e[(e.len() as f64 * 0.1) as usize % e.len()]
        };
        let (b0, a0) = (spread(&before), spread(&marks));
        assert!(b0 > 0.15, "the fixture must actually drift to begin with, got {b0}");
        // Measured: 0.220 → 0.024 of a period — and the residual IS the loop's steady-state lag for
        // a ramp (drift/β = 0.0024/0.1 = 0.024), i.e. the number is predicted, not fitted.
        assert!(a0 < 0.05, "phase must stop drifting: p10..p90 spread {b0:.3} -> {a0:.3} periods");

        // ⭐ Sub-sample resolution, asserted as SPREAD rather than as distance-to-truth.
        // ⚠ Written this way after the naive version failed at 18.7 samples: the feature is local
        // ENERGY, and a glottal-like pulse is one-sided (sharp onset, decaying ring), so the
        // energy-maximising window centre sits *after* the onset by roughly its own half-width.
        // That is a BIAS, and a constant bias is harmless here — every mark gets the same one, and
        // what this module needs is a consistent phase reference, not the glottal closure instant.
        // What is NOT harmless is per-mark scatter, because that lands straight in the synthesis
        // pulse positions. The fixture's period is 44100/313 = 140.9 samples, so the pulses fall
        // between samples and an integer-grid argmax must scatter by ~±0.5.
        let (xf, tf) = pulses(sr, 0.4, |_| 313.0, |_| 1.0);
        let mut mf: Vec<f64> = tf.iter().map(|t| t + 6.0).collect();
        lock_phase(&xf, &mut mf, 0.45);
        let mut res: Vec<f64> = mf.iter().zip(&tf).map(|(a, b)| a - b).collect();
        res.sort_by(f64::total_cmp);
        let mid = res[res.len() / 2];
        let mut dev: Vec<f64> = res.iter().map(|v| (v - mid).abs()).collect();
        dev.sort_by(f64::total_cmp);
        // ⚠ The bound is MEASURED, not guessed, and re-measured when the loop replaced the
        // correct-afterwards scheme: with the refinement the scatter is **0.039** samples, with it
        // removed **0.075**. 0.055 sits between the two regimes. (Pre-PLL it was 0.005 / 0.045 —
        // the loop low-passes the quantisation too, which is why the bound had to move.)
        assert!(
            dev[dev.len() / 2] < 0.055,
            "residual scatter {:.3} samples around its median — that is integer-grid quantisation, \
             the sub-sample refinement is not doing its job",
            dev[dev.len() / 2]
        );
    }

    #[test]
    fn the_phase_lock_cannot_reach_the_neighbouring_pulse() {
        // ⛔⛔ THE structural gate, and the reason the radius is derived from the LOCAL period.
        // Alternating loud/quiet pulses is the classic period-doubling bait: a per-mark "snap to
        // the biggest thing nearby" would drag every quiet-pulse mark onto its loud neighbour,
        // halving the mark count in effect and writing an exact octave-down subharmonic. That is
        // not a hypothetical — S148's WSOLA arm was killed by a blind test for manufacturing
        // exactly that (−1200 cents, 1.47 s, "it even sounds solid"), and its four acceptance
        // criteria contained no pitch measurement at all.
        //
        // ⚠ The fixture has to be hostile on BOTH counts or the guard goes untested: alternating
        // gains supply the bait, and a **glissando** (250 → 450 Hz across the island) is what
        // separates a LOCAL radius from an island-wide one. On a constant-period fixture the two
        // are numerically identical, and a mutation swapping them went green.
        let sr = 44_100u32;
        let (x, truth) = pulses(
            sr,
            0.4,
            |k| 250.0 + 4.0 * k as f64,
            |i| if i % 2 == 0 { 1.0 } else { 0.25 },
        );
        let mut marks = truth.clone();
        lock_phase(&x, &mut marks, 0.45);

        assert!(marks.windows(2).all(|w| w[1] > w[0]), "marks must stay strictly increasing");
        // Ground truth is per-pulse now, so the tolerance is per-pulse too.
        let mut worst = 0.0f64;
        for i in 1..truth.len() - 1 {
            let p = (truth[i + 1] - truth[i]).min(truth[i] - truth[i - 1]);
            worst = worst.max((marks[i] - truth[i]).abs() / p);
        }
        assert!(worst < 0.5, "no mark may travel to a neighbouring pulse, worst {worst:.3} periods");
        // …and the spacing must not have collapsed anywhere (that is what doubling looks like in
        // the mark train itself, before any audio is synthesized).
        for i in 1..marks.len() {
            let want = truth[i] - truth[i - 1];
            let got = marks[i] - marks[i - 1];
            assert!(
                got > 0.5 * want && got < 1.5 * want,
                "spacing at {i} is {got:.1} against a true period of {want:.1}"
            );
        }
    }

    #[test]
    fn the_phase_lock_may_not_jitter_the_spacing_when_the_peak_choice_is_ambiguous() {
        // ⛔⛔ THE gate this file was missing, and the user paid for that: the first S150 arm was
        // shipped with every criterion green and it **clicked**. The mechanism, once localised:
        // real voiced material has a second energy lobe inside each period (formant ringing), the
        // two are nearly tied (measured runner-up/winner ratio p50 0.72-0.99), so a per-mark argmax
        // flips between them for a *run* of marks. Consecutive grains then sit at 0.7 T / 1.3 T
        // instead of T, stop lining up, and each mismatch is a broadband transient.
        // ⚠ Note what did NOT catch it: mark count (unchanged), `cola_gap` (0.003%), the identity
        // gate, the depth ruler (it got *better*), the per-note octave gate, and the whole-song f0
        // gate. The defect lives in the SPACING, so the criterion has to be the spacing.
        let sr = 44_100u32;
        let p = f64::from(sr) / 260.0;
        // Two lobes per period, with gains that cross over every few periods ⇒ the argmax winner
        // genuinely alternates. This is the fixture the naive version fails on.
        let (main, truth) = pulses(sr, 0.5, |_| 260.0, |_| 1.0);
        let (second, _) = pulses(sr, 0.5, |_| 260.0, |i| 0.92 + 0.16 * ((i as f64) / 3.0).sin());
        let shift = (0.30 * p) as usize;
        let mut x = main.clone();
        for i in shift..x.len() {
            x[i] += second[i - shift];
        }

        let mut marks: Vec<f64> = truth.iter().map(|t| t + 0.12 * p).collect();
        let jitter = |m: &[f64]| -> f64 {
            let d: Vec<f64> = m.windows(2).map(|w| w[1] - w[0]).collect();
            let mut s = d.clone();
            s.sort_by(f64::total_cmp);
            let med = s[s.len() / 2];
            let mut r: Vec<f64> = d.iter().map(|v| (v / med - 1.0).abs()).collect();
            r.sort_by(f64::total_cmp);
            r[(r.len() as f64 * 0.99) as usize % r.len()]
        };
        let before = jitter(&marks);
        lock_phase(&x, &mut marks, 0.45);
        let after = jitter(&marks);
        // Calibrated by taking the two defences out, one at a time (fixture jitter p99):
        //   (recalibrated for the PLL; the pre-PLL numbers are in the commit history)
        //   median + slew               0.0200   (the slew alone holds it)
        //   low-pass, no slew           0.0095   (the low-pass alone holds it)
        //   **median, no slew**         **0.2931** ← the arm that shipped the clicks, 31× worse
        // ⇒ the two are redundant on purpose, and this bound sits between the two regimes.
        // ⚠ On real material the same statistic reads 0.1377 (untouched engine) / 0.3457 (clicky
        // arm) / 0.1385 (shipping) — the fixture reproduces the magnitude, not just the direction.
        assert!(
            after < 0.05,
            "locking jittered the spacing: p99 of |d/median − 1| went {before:.4} → {after:.4} — \
             that is what a click sounds like"
        );
    }

    #[test]
    fn two_marks_may_not_collapse_onto_the_same_pulse() {
        // ⛔ The crossing guard needs its own fixture: the two tests above cannot go red for it,
        // because with evenly-spaced marks nothing ever converges. The failure it prevents is
        // subtle and silent — two marks landing on ONE pulse leaves the count intact (so design
        // note 3 still "passes") and the monotonic fixup at the end turns the collapse into a
        // 1e-6 gap, i.e. it looks fine everywhere except in the audio, where that stretch now
        // synthesizes at double the period.
        // ⇒ assert the SPACING, not the ordering. (Written after noticing the guard was
        // uncovered; the ordering assert alone went green on a version with the guard removed.)
        let sr = 44_100u32;
        let (x, truth) = pulses(sr, 0.3, |_| 250.0, |_| 1.0);
        let p = f64::from(sr) / 250.0;
        // Adversarial: a pair of marks only 0.6 periods apart, both within reach of one pulse.
        let mut marks: Vec<f64> = Vec::new();
        for (i, t) in truth.iter().enumerate() {
            if i % 3 == 0 {
                marks.push(t - 0.25 * p);
                marks.push(t + 0.35 * p);
            } else {
                marks.push(*t);
            }
        }
        marks.sort_by(f64::total_cmp);
        marks.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        let n_before = marks.len();
        lock_phase(&x, &mut marks, 0.45);
        assert_eq!(marks.len(), n_before, "no mark may be dropped");
        let min_gap = marks
            .windows(2)
            .map(|w| w[1] - w[0])
            .fold(f64::MAX, f64::min);
        assert!(
            min_gap > 0.4 * p,
            "two marks collapsed onto one pulse: min gap {min_gap:.1} samples vs period {p:.1}"
        );
    }

    #[test]
    fn the_phase_lock_is_opt_in_and_the_default_arm_is_byte_for_byte_unchanged() {
        // ⚠ Additive, per the S146 protocol: nothing the user hears may move until a blind test
        // says so. S148's WSOLA is why that is not negotiable — it read better on the ruler it was
        // built for and was 3/3 rejected by ear.
        let sr = 44_100;
        let x = voiced(sr, 0.5, 300.0);
        let hop = sr as usize / 200;
        let f0 = flat_f0(x.len(), hop, 300.0);

        for st in [-6.0, -1.0, 1.0, 6.0] {
            let (base, d0) = psola_shift_diag(&x, sr, st, &f0, hop);
            let (off, d1) =
                psola_shift_locked(&x, sr, st, 0.0, &f0, hop, false, 0.0, 0.0);
            assert_eq!(base, off, "{st} st: phase_lock 0.0 must be the legacy arm, bit for bit");
            assert_eq!(d0.marks_locked, 0, "…and it must report that it moved nothing");
            assert_eq!(d1.marks_locked, 0);

            let (on, d2) = psola_shift_locked(&x, sr, st, 0.0, &f0, hop, false, 0.0, 0.45);
            // ⛔ "the arm is on" and "the arm did something" are two different facts — a lock that
            // never moved a mark would produce byte-identical audio and be indistinguishable from
            // off unless this count is asserted (S147: a change whose benefit was silently halved).
            assert!(d2.marks_locked > 0, "the lock must actually move marks when enabled");
            assert_eq!(d2.marks, d0.marks, "locking must not change the mark COUNT (design note 3)");
            assert_eq!(d2.islands, d0.islands);
            assert_ne!(on, base, "{st} st: …and the opt-in arm must actually change the audio");
        }
    }

    #[test]
    fn ratio_one_stays_the_identity_with_the_phase_lock_on() {
        // The cheapest, least fakeable gate this module has — and it must be re-asserted for every
        // arm, because it is what caught three "obviously correct" designs in S146.
        // ⚠ Stated honestly, and now MEASURED rather than reasoned: this gate is **structurally
        // blind to where the marks are**. Replacing `analysis_marks` wholesale with a 137-sample
        // uniform grid — no f0, no waveform, nothing — still produces a whole-song ST=0 output
        // that is **bit-identical** to the baseline (sha256 1565ff95…). Every mark set whose
        // spacings land in (1.0, 882] samples passes, because at r=1 `tgt[j] == src[j]` ⇒ `d == 0`
        // and the rising/falling half-cosines sum to exactly 1 on the same span.
        // ⇒ It proves the lock did not break the synthesis path; it proves **nothing** about
        // placement. Before this commit, NO test in this file could see a mark-placement change:
        // a realistic phase lock altered the +7 fixture audio by max |Δ| = 0.978 (peak 0.9) and
        // all 14 tests stayed green. That is what the four gates above exist for.
        let sr = 44_100;
        let x = voiced(sr, 0.5, 220.0);
        let hop = sr as usize / 200;
        let f0 = flat_f0(x.len(), hop, 220.0);
        for lock in [0.0, 0.25, 0.45] {
            let (y, d) = psola_shift_locked(&x, sr, 0.0, 0.0, &x_f0(&f0), hop, false, 0.0, lock);
            assert_eq!(y, x, "ratio 1.0 must be the identity with phase_lock = {lock}");
            if lock > 0.0 {
                assert!(d.marks_locked > 0, "…and the lock must have been live while proving it");
            }
        }
    }

    /// Identity helper so the call above reads as one line (the f0 track is not the subject).
    fn x_f0(f0: &[f32]) -> Vec<f32> {
        f0.to_vec()
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
    fn the_formant_knob_is_a_no_op_at_zero_and_moves_the_spectrum_without_the_pitch() {
        // κ is a user-facing slider (0..1). Two things must hold or the slider is a lie:
        //   κ=0 must be BIT-identical to the plain path (otherwise the default arm — the only one
        //        the user actually A/B'd — silently changed), and
        //   a non-zero formant move must actually change the audio while leaving the pitch alone
        //        (an "it compiles" wiring would satisfy neither).
        let sr = 44_100;
        let x = voiced(sr, 0.5, 300.0);
        let hop = sr as usize / 200;
        let f0 = flat_f0(x.len(), hop, 300.0);

        let plain = psola_shift(&x, sr, 6.0, &f0, hop);
        let kappa0 = psola_shift_formant(&x, sr, 6.0, 0.0, &f0, hop).0;
        assert_eq!(plain, kappa0, "κ=0 must be bit-identical to the plain shift");

        // formant-only: pitch must not move, timbre must.
        let warped = psola_shift_formant(&x, sr, 0.0, 6.0, &f0, hop).0;
        assert_eq!(warped.len(), x.len());
        let inner = sr as usize / 10..x.len() - sr as usize / 10;
        assert_eq!(
            dominant_period(&x[inner.clone()], 40, 800),
            dominant_period(&warped[inner.clone()], 40, 800),
            "a formant-only move must not touch the pitch"
        );
        let diff = x
            .iter()
            .zip(warped.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(diff > 1e-3, "a formant move of +6 st must change the audio, got max |Δ| {diff}");

        // and the spectral centre of mass must rise (that IS the formant move)
        let centroid = |v: &[f32]| -> f64 {
            let seg = &v[inner.clone()];
            let mut num = 0.0f64;
            let mut den = 0.0f64;
            for (i, w) in seg.windows(2).enumerate() {
                let d = f64::from(w[1] - w[0]).abs(); // crude HF-weighted energy proxy
                num += d * i as f64;
                den += d;
            }
            let _ = num;
            den / seg.len() as f64 // mean |Δ| ∝ spectral centroid × amplitude
        };
        assert!(
            centroid(&warped) > centroid(&x) * 1.05,
            "formants moved up ⇒ the high-frequency content must rise ({} vs {})",
            centroid(&warped),
            centroid(&x)
        );
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
    /// S148 —— 把**分析标记**倒出来,因为外面看不见它。
    ///
    /// 为什么要这个:6 半音上我们在浊音帧的 **2.4%** 上挖出 >4 dB 的电平陷波,而 praat 在同一段
    /// 输入、同一个比值上只有 **0.2%**;那些位置在 5/6/7 半音之间**高度稳定**(三者共有 57 帧),
    /// 而**输入自身的性质一条都分不开它们**(f0 匹配对照之后:周期性 NCC 差 −0.001、f0 抖动
    /// +0.001、电平斜率 −1.1 dB/s、到最近清音帧 +15 帧)。
    /// ⇒ 排除法把病因指向**我们自己的内部状态**,而唯一看不见的内部状态就是标记。
    ///
    /// ⚠ 这里**只重跑分析一路**(`voiced_islands` + `analysis_marks`,两个纯函数),
    /// 合成一行都不碰 —— 它们不吃 ratio,所以任何深度下标记都是同一套。
    /// 输出:每行 `island_a island_b mark_sample`(制表分隔),给 `UTAI_PSOLA_MARKS` 指路。
    #[test]
    #[ignore = "probe: dumps analysis marks (set UTAI_PSOLA_IN/F0/HOP/MARKS)"]
    fn psola_marks_dump() {
        let path = std::env::var("UTAI_PSOLA_IN").expect("UTAI_PSOLA_IN");
        let f0p = std::env::var("UTAI_PSOLA_F0").expect("UTAI_PSOLA_F0");
        let out = std::env::var("UTAI_PSOLA_MARKS").expect("UTAI_PSOLA_MARKS");
        let hop: usize = std::env::var("UTAI_PSOLA_HOP").expect("UTAI_PSOLA_HOP").parse().unwrap();

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

        // 与 psola_shift_opts 同一条前处理:标记跑在去 DC 的信号上(设计注 2)。
        let n = x.len();
        let sr = f64::from(spec.sample_rate);
        let mean = x.iter().map(|v| f64::from(*v)).sum::<f64>() / n as f64;
        let dc_free: Vec<f32> = x.iter().map(|v| (f64::from(*v) - mean) as f32).collect();

        // S150 —— `UTAI_PSOLA_LOCK=<periods>` 打开相位锁定(默认 0 = 关,与生产同一套标记)。
        // ⭐ 有了它,「Rust 的标记」与「Python 原型的标记」可以**逐个对拍**,
        //    候选在离线上量到的每一个数才真的属于将要上线的那份实现。
        let lock: f64 =
            std::env::var("UTAI_PSOLA_LOCK").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
        let mut s = String::new();
        let mut islands = 0usize;
        let mut marks = 0usize;
        let mut locked = 0usize;
        for (a, b) in voiced_islands(&f0, hop, n, (MIN_ISLAND_SECONDS * sr) as usize) {
            let mut src = analysis_marks(&dc_free, spec.sample_rate, &f0, hop, a, b);
            if src.len() < 3 {
                continue;
            }
            locked += lock_phase(&dc_free, &mut src, lock);
            islands += 1;
            marks += src.len();
            for m in &src {
                // ⛔ 全精度,不是 `{m:.4}`。S150 实测:4 位小数的截断会让 12 个颗粒的
                // `round(tm) − round(s_pos)`(或窗端点 `round(s±w)`)翻到 .5 的另一侧,
                // 于是拿这份 dump 离线重放出来的波形与 Rust 自己渲的差最多 **3814 LSB**。
                // ⚠ 12 段位置是**事前**从「到 .5 边界的距离 < 5e-5」预测出来的,12/12 命中 ——
                // 所以这是精度问题,不是合成路径问题。f64 的 `{}` 是最短往返表示。
                s.push_str(&format!("{a}\t{b}\t{m}\n"));
            }
        }
        std::fs::write(&out, s).expect("write marks");
        eprintln!("[mg] marks: {islands} islands, {marks} marks, lock {lock} moved {locked} -> {out}");
    }

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

        // S148:`UTAI_PSOLA_WSOLA=<frac>` 打开源侧波形相似度搜索(默认 0 = 关,逐位同旧)。
        let wsola: f64 =
            std::env::var("UTAI_PSOLA_WSOLA").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
        // S148 —— `frac_transport` 以前在这里写死成 false,于是 `UTAI_PSOLA_FRAC` 对这条探针**完全无效**:
        // 开与不开的输出 sha256 逐位相同。我差点把那读成「亚样本搬运对包络起伏没用」。
        // ⛔「臂开着」与「臂做了事」是两件事 —— 现在它由 env 控,并且把实际取值打出来。
        let frac = std::env::var("UTAI_PSOLA_FRAC").ok().is_some_and(|v| v != "0" && v != "false");
        // S150 —— `UTAI_PSOLA_LOCK=<periods>`(默认 0 = 关)。
        let lock: f64 =
            std::env::var("UTAI_PSOLA_LOCK").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
        let (y, d) =
            psola_shift_locked(&x, spec.sample_rate, st, 0.0, &f0, hop, frac, wsola, lock);
        println!("  arms: frac_transport={frac} wsola={wsola} phase_lock={lock}");
        println!(
            "psola_probe: {} samples @{} Hz, {st:+} st, f0 frames {} hop {hop}\n  \
             islands {} marks {} cola_gap {:.1}% w_median {:.3} wsola {wsola} moved {} \
             lock {lock} moved {}",
            x.len(), spec.sample_rate, f0.len(), d.islands, d.marks,
            d.cola_gap_frac * 100.0, d.cola_w_median, d.wsola_moved, d.marks_locked
        );
        assert_eq!(y.len(), x.len(), "exact-length contract");

        // S148 —— 逐颗粒轨迹(只在 `UTAI_PSOLA_GRAIN_DUMP` 设了的时候写)。
        // ⛔ 空文件与「没开」必须分得开:开了就一定有行,一行都没有说明循环根本没跑到,
        //    那是「跑不起来」不是「测出来没有」。
        if let Ok(p) = std::env::var("UTAI_PSOLA_GRAIN_DUMP") {
            let rows = GRAIN_TRACE.lock().unwrap();
            assert!(!rows.is_empty(), "grain dump 开着却一行都没有 —— 合成循环没跑到,读数无效");
            let mut s = String::from("tm\tsrc\tdelta\tt_src\tphase\tlw\trw\tk\n");
            for r in rows.iter() {
                s.push_str(&format!(
                    "{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.6}\t{:.4}\t{:.4}\t{}\n",
                    r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7] as i64
                ));
            }
            std::fs::write(&p, s).expect("write grain dump");
            println!("  grain dump: {} 行 -> {p}", rows.len());
        }

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
