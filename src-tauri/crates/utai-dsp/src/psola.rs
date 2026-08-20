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
    /// ⛔⛔ **"It is inaudible" was written here first and it is only half true.** By *energy* it
    /// is: 97 % sits below 5 Hz. But the thing that matters is not the steady share, it is the
    /// **steps** — the baseline jumps, and a step is broadband with a 1/f tail. Measured at the
    /// four largest jumps (±40 ms), against the ext-off arm: **20-60 Hz is +16 to +24 dB**, and
    /// that band is audible. Whole song, voiced cells: 1-20 Hz **+28.2 dB**, **20-60 Hz +7.9 dB**,
    /// 60-200 Hz −1.0, 200-800 Hz +0.3.
    /// ⇒ the collar that survives is only: **it does not eat headroom** (removing everything under
    /// 50 Hz moves the whole-song peak by −0.008 dB).
    ///
    /// ⭐ How big the steps are: |Δbaseline| per 5 ms, normalised by the ext-off arm's local RMS —
    /// ext-off has **6** cells over 20 %, today has **1295**, and the worst reads **238 %**
    /// (at 133.44 s the baseline jumps 0 → **+0.235** while the waveform peaks at 0.5).
    /// ⭐ It is also visible: the waveform sits off-centre. The user diagnosed it from Audition —
    /// vertical low-frequency bands in the spectrogram at exactly the places he had been calling
    /// "seams" / "clicks", plus five-track waveforms where only the ext-off and the HP arm looked
    /// symmetric. Every ruler in this repo had missed it, because they are all RMS-domain and
    /// **RMS does not see a DC offset as a defect — it just counts it as signal.**
    pub infrasonic_frac: f32,
    /// S152 — how much of that the removal arm actually took out (`before − after`), 0.0 when off.
    /// ⛔ Same reason as `wsola_moved` / `marks_locked`: "the arm is on" and "the arm did
    /// something" are different facts and only the second one is visible in the audio.
    pub infrasonic_removed: f32,
    /// S155 — the width the cut actually used on this buffer, in ms (0.0 when off).
    ///
    /// ⛔ This is not decoration. With [`Infrasonic::PerPeriod`] the width is derived from the f0
    /// track, so a single bad frame can widen it and **halve the benefit with every other reading
    /// still green** — that is precisely the S147 silent-halving shape, and this number is the
    /// only thing that shows it. See [`infrasonic_width_ms`].
    pub infrasonic_ma_ms: f32,
    /// S154 — |20·log10(env(out) / env(in))| median over the covered span, 5 ms RMS window, dB.
    ///
    /// **This process is a pitch transform. It has no business changing the amplitude envelope.**
    /// It does: measured on the probe (donor −14, 10 s, same buffer in and out, so the reading is
    /// the process and nothing else) **p50 1.14 dB · p90 2.05 · max 10.83**, while the same
    /// reading outside the voiced islands is **exactly 0.00** (there `out ≡ in` by construction —
    /// that zero is the control that makes the rest of the number mean something).
    ///
    /// ⭐ Where it lives: **at the island start**, i.e. the vowel onset. Step across the first
    /// 20 ms of each island, s14 segment: **−0.49 / −1.76 / −6.09 / −0.48 dB** (4 islands).
    /// S153 §4k had already found the same concentration from the other side (worst 15 violations
    /// whole-song: 14 of them within ±35 ms of a vowel onset).
    ///
    /// ⛔⛔ **Why it was dropped once, and why that was wrong.** S153 filed it as "real but not the
    /// click" because it could not rank the user's six annotated points. That inference does not
    /// hold: *a ruler failing to rank six points* is a fact about **our measurement**, not about
    /// the world — and the user had reported this very defect from the **waveform** ("波形在进入
    /// 稳定的长音之前有一个非常突兀的波形尖峰") back in S152, where it was measured as ruler ⑦
    /// (起音过冲) and dropped for the same reason. Two independent drops of the user's own
    /// first-hand observation, both because a ranking test failed. ⇒ **Never write "a ruler could
    /// not separate the marks" as "the phenomenon is not there".**
    pub env_dev_p50_db: f32,
    /// S154 — the same median **after** the restoration arm ran, 0.0 when the arm is off.
    /// ⛔ "The arm is on" vs "the arm did something", same rule as `infrasonic_removed`.
    pub env_dev_after_db: f32,
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
/// `seed_at`: where to plant the first mark. `None` = the island midpoint, which is what this has
/// always done.
///
/// ⛔⛔ **Why it had to become a parameter (S154).** The seed decides the phase of the *entire* mark
/// train (everything else walks outward from it), so anything that moves the island edges moves
/// every grain in that island. Measured when the island-dilation arm first landed: the whole-song
/// difference against today was **p50 +1.2 dB relative — larger than the signal — in 73 % of voiced
/// cells**, against a render floor of −28.1 dB. That is a complete re-synthesis, not an edge fix.
/// ⇒ two things were changing at once: the boundary coverage (what we wanted) and a re-roll of
/// every note's mark phase (a lottery — S148/S150 are entirely about how much mark placement
/// matters). Seeding from the **undilated** island keeps the marks put, so the arm changes only
/// what it is supposed to.
fn analysis_marks(
    x: &[f32],
    sample_rate: u32,
    f0: &[f32],
    hop: usize,
    a: usize,
    b: usize,
    seed_at: Option<f64>,
) -> Vec<f64> {
    let sr = f64::from(sample_rate);
    let mid = seed_at.unwrap_or((a + b) as f64 * 0.5);
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
/// ## ⛔⛔ This number was picked twice against a **broken reference** before it was picked right
///
/// The question is "how much of the low band did this process ADD", so it needs a reference that
/// is the same performance minus the process. Two wrong answers came first:
/// 1. **20 ms**, chosen because 97 % of the injected *energy* is below 5 Hz ⇒ "inaudible".
///    That reasoning is about the steady share, and the audible part is the **steps**.
/// 2. **12 ms**, chosen by matching the **ext-off** arm. ⛔ The user caught this one:
///    *"无扩在很多地方都失声了毫无参考意义"* — the un-rescued arm is silent on exactly the notes
///    that get rescued, so part of "today is +7.9 dB louder in 20-60 Hz" is simply
///    **the rescue giving those notes a voice**, which is not a defect. Same trap this session's
///    adversarial pass had already written down as a rule, and it caught me anyway.
///
/// ## The reference that works: the donor against **itself**
///
/// `mg_render_sovits` with `UTAI_MG_INVERSE=0` vs `=1` is the same render with and without this
/// process. Band RMS over voiced 50 ms cells, after aligning the two arms on 400-4000 Hz (each
/// gets its own `peak_normalize`, which is a fake difference otherwise). ⚠ Judge on **20-60 Hz
/// only**: a voice's f0 cannot live there (that is MIDI 24-34), while 60-150 Hz really does carry
/// the raw arm's fundamental (it is the whole song transposed down) and is therefore not
/// comparable.
///
/// | cut | −9 st | −12 st | −14 st | cost at 400-4k |
/// |---|---|---|---|---|
/// | injected (no removal) | +13.2 | +14.5 | +14.7 | — |
/// | 20 ms | +10.8 | +12.2 | +12.3 | −0.002 dB |
/// | 12 ms | +6.0 | +7.4 | +7.6 | −0.005 |
/// | **8 ms** | **+0.6** | **+2.0** | **+2.2** | **−0.013** |
/// | 6 ms | −3.7 | −2.4 | −2.2 (now cutting real signal) | −0.021 |
///
/// ⇒ 8 ms lands on the target at all three ratios and costs 0.013 dB where the voice actually is.
///
/// ## ⚠⚠ S155 — this constant is now a **RULER, not the cut**
///
/// The removal's width is [`Infrasonic::PerPeriod`], derived **per voiced island** from the
/// source periods of the grains actually synthesised there (⛔ **not** from the f0 track — see
/// that function for why that was wrong twice), and it uses [`CUT_BOX_PASSES`] box passes, not 2.
/// This 8 ms stays as the fixed width of the *reading* [`PsolaDiagnostics::infrasonic_frac`], and that
/// is deliberate: every historical number on this line (S152's 10.8/25.5/33.7 %, S154's tables,
/// S155's 8.84/23.00/17.27 %) was taken with this ruler, and a ruler whose scale follows the
/// knob is not a ruler.
///
/// ## ⛔ Two corrections to what this doc used to say
///
/// 1. It claimed «at an output f0 of 110 Hz this filter takes **−0.38 dB** off the fundamental.
///    That is measured, not hypothetical». **It is the 12 ms number.** Re-measured: 12 ms costs
///    −0.3673 dB at 110 Hz, the shipped 8 ms costs **−0.1540**. The sentence survived the
///    12 → 8 ms change with its number attached — the exact shape of stale doc this repo has
///    been burned by before, and it was caught by a forensic pass, not by a test.
/// 2. It said the up-shift path is "far away" from trouble. It is (score path: lowest output f0
///    on six installed records × the calibration song is MIDI 61 = **277 Hz**). ⛔ But
///    `cover_dead_plan` emits a **positive** shift (audio moved DOWN) and every installed record
///    has `usable.0 == 36`, i.e. the dead notes are at the **bottom** ⇒ output f0 below
///    **65.41 Hz**, where a fixed 8 ms cut takes **−3.98 dB** off the fundamental, and the
///    structural floor (MIDI 12 = 16.35 Hz) reaches **−25.18 dB**.
///    ⚠ Also: the response is oscillatory ABOVE the first null — 178.69 Hz (MIDI 53.4, the middle
///    of a male range) still costs −0.42 dB.
///    ⇒ this is why the width had to become adaptive before the default could be flipped, and
///    why "≈ 3 periods" from the old note became **one period of the lowest fundamental in each
///    island** (measured: over 31.7-830 Hz the adaptive rule costs the fundamental
///    −0.0000…−0.0005 dB).
const INFRASONIC_MA_MS: f64 = 8.0;

/// S155 — the narrowest and widest the adaptive cut is allowed to get, in ms.
///
/// The narrow end is a backstop against an f0 track that reports something absurd; the wide end
/// is where the cut stops doing anything useful anyway (a null at 20 Hz).
const INFRASONIC_MS_MIN: f64 = 1.0;
const INFRASONIC_MS_MAX: f64 = 50.0;

/// S155 — how (and whether) to subtract the infrasonic baseline this process manufactures.
///
/// ⛔ Why this is an enum and not the `bool` it replaced: the width **is** the whole question.
/// A fixed 8 ms removes the part that only shows up in the waveform (the ride off-centre) and
/// leaves **+9 dB at 20-50 Hz and +12 dB at 50-125 Hz** of manufactured energy standing — which
/// is the band the user reported seeing as "200 Hz 以下的极低频亮带". Measured on the probe
/// (zero render floor), s14 residual against the input's own low band:
///
/// | cut | 0.5-20 Hz | 20-50 | 50-125 | 125-200 |
/// |---|---|---|---|---|
/// | none | +42.0 | +22.8 | +14.6 | +3.6 |
/// | fixed 8 ms | +1.7 | **+8.9** | **+11.6** | **+3.3** |
/// | a 2.9 ms cut (2-pass) | −0.5 | +0.1 | +2.1 | +0.8 |
///
/// ⚠ **That table is the 10 s PROBE, 2 box passes, one width for the whole buffer** — i.e. the
/// first thing S155 shipped, not what runs today. Two things changed after it, both because the
/// user heard/saw something the table cannot show:
/// * the cut is **4 box passes** (`CUT_BOX_PASSES`) — 2 passes leak the donor's own pitch through
///   the −27 dB first sidelobe, which he reported as 「合唱感」;
/// * the width is **per island**, not per buffer.
/// ⇒ for what production actually does, read the whole-song numbers in
///   `TESTING/s155_knives/au_s155e/看哪里.md`. ⛔ Do not quote this table as "today".
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Infrasonic {
    /// Off — byte-for-byte the arm without this step.
    Off,
    /// Width = **one period of the lowest fundamental in each voiced island**, input or output,
    /// taken from the grains actually synthesised there. See [`infrasonic_width_ms`] for why that
    /// is the right width, why it must be per island, and where the f0 must NOT come from.
    PerPeriod,
    /// A fixed width in ms — for A/B renders and for reproducing an older arm.
    FixedMs(f64),
}

/// S155 — the cut's width for **one island**: one period of the lowest fundamental *in that
/// island*, measured from the source periods of the grains actually synthesised there.
///
/// ## What picks this number
///
/// The removal is `out -= LP(out) − LP(in)`, so `LP` is applied to the *input* too. That is what
/// keeps ratio 1.0 bit-exact (see [`psola_shift_env`]), and it is also the whole risk: whatever
/// `LP` passes of the **input** gets added into the output — and the input is the donor singing
/// the same words 9-14 semitones lower. Feeding a low-passed copy of *that* into a rescued note
/// is the same family of defect as the un-transposed attack S154 spent a session removing.
///
/// ⇒ the width must put the low-pass's **first null on the donor's own fundamental**, so the
/// filter structurally cannot carry the voice. Two boxes of `L` ms have their first null at
/// `1000/L` Hz ⇒ `L = 1000 / f0_lowest`.
///
/// ⛔ **Which f0.** On an up-shift the input is the lower one; on a down-shift (the cover path,
/// `cover_dead_plan` emits +22 ⇒ audio moved DOWN 22 semitones) the *output* is. Taking the
/// minimum of the two is what makes one rule safe on both, and it is not hypothetical: at an
/// output f0 of 110 Hz a fixed 3 ms cut takes **−10.1 dB** off the fundamental (measured,
/// `s155_knives/width_sweep.py`).
///
/// ## ⛔⛔ Where the f0 must NOT come from (this cost two commits)
///
/// **The f0 track is not the pitch content of this buffer.** Production hands the whole song's
/// `note_hz_full` to every rescue pass, while `score2svc.rs` zero-fills the audio of every chunk
/// that pass does not intersect — and it **never masks the f0 track** (the S151 note in
/// `vocal_range.rs` was written about exactly this asymmetry). A percentile over the raw track is
/// therefore a percentile over **notes that are digital silence here**, and those are never the
/// notes being rescued. Measured: at −14 st the raw p01 gives **6.03 ms** where the rescued notes
/// call for **2.70**, leaving **+8.5…+9.6 dB** standing in 50-125 Hz and removing **0.16 dB** in
/// 125-200 Hz — i.e. nothing, in the exact band the user reported.
///
/// ⛔ **And no reading showed it**: `infrasonic_frac` is the fixed 8 ms ruler whose energy is
/// 99.4 % below 20 Hz, so 6.03 ms and 8.00 ms read the same "removed 18.8 pts". Only
/// [`PsolaDiagnostics::infrasonic_ma_ms`] differed, and nobody compared it to the width the
/// decision had been made on. **That is the S147 silent-halving shape, entering through a door
/// this function's own doc comment described.**
///
/// Masking the track by "this frame's audio is not silence" was still not enough: a *rendered*
/// chunk also carries the notes this pass does **not** rescue, and they are lower. ⇒ the only
/// quantity that needs no outside knowledge of "which notes get rescued" is the **source period
/// of each grain this process actually laid down**, per island.
///
/// ## Measured (whole song, per-island, production)
///
/// Median island width per pass: **1.98 / 3.09 / 2.70 ms** at −9 / −12 / −14 — and 2.70 is
/// exactly what an independent derivation from "only the 62 notes actually rescued at −14" gives.
///
/// ⚠ The chosen width is reported in [`PsolaDiagnostics::infrasonic_ma_ms`] (the median over
/// islands) precisely so that a silent widening stays visible.
/// ⚠ The failure direction of a too-low estimate is *less removal*, never damage.
fn infrasonic_width_ms(src_periods: &[f64], sample_rate: u32, ratio: f64) -> f64 {
    // S155 笔4 —— 用**真的被合成出来的颗粒自己的源周期**,而不是 f0 轨的分位数。
    //
    // ⛔ 笔3 先用「这一帧的音频不是数字静音」掩 f0 轨,那挡掉了铺零的 chunk
    //    (−14 上 6.03 → 5.41 ms),但**没到位**:S147 之后 donor 只渲相交的 chunk,而一个被渲
    //    的 chunk 里同样有**这一遍不救**的低音 —— 它们的音频是真的、f0 也是真的。
    // ⇒ 唯一不需要「哪些音会被救」这条外部知识的量,就是**这道工序此刻真的在搬的那段波形**
    //    的周期。它逐颗粒都在手边(`src_l`),而且它本来就是这把刀要保护的那个基频。
    // ⚠ 收集端还要再挡一次静音:S151 实测**铺零区照样会被铺满标记**(去 DC 之后常数上处处
    //    相关 = 1.0,间距恒为标称周期),所以按**这一颗粒的源读窗里有没有音频**过滤。
    if src_periods.is_empty() {
        return INFRASONIC_MS_MAX;
    }
    let mut v = src_periods.to_vec();
    v.sort_by(f64::total_cmp);
    // ⛔ p90 而不是 p99/max:S155 笔6 改成**逐岛**之后,一个岛只有几十颗粒,
    // p99 实际上就是 max —— 而岛首尾那两颗粒的邻距是**外推**出来的、偏大。
    // 实测:200 Hz 与 400 Hz 两个岛的夹具上,p99 给出 6.31 ms(= 158 Hz),两个都不对。
    // ⚠ 失败方向仍然安全:偏窄只会少削一点或多漏一点,而 4 遍盒把漏压在 −53 dB 以下。
    let t = v[((v.len() - 1) as f64 * 0.90).round() as usize];
    let f_src = f64::from(sample_rate) / t.max(1.0);
    // Up-shift ⇒ 源更低;down-shift ⇒ 输出更低。
    let lowest = f_src * ratio.min(1.0);
    return (1000.0 / lowest.max(1.0)).clamp(INFRASONIC_MS_MIN, INFRASONIC_MS_MAX);
}


/// S155 — what `psola_probe` runs when an arm's env var is **unset**: the production defaults.
///
/// ## ⛔⛔ Why this table exists
///
/// The probe used to hard-code `0.0` for every arm. That was fine while every production default
/// *was* zero, and it silently stopped being fine the moment one was flipped: after S154 the
/// production arm is `bridge = 30 ms` and `phase_lock = 0.30`, so anyone re-running the S154 probe
/// script today would get **the pre-S154 arm** and file it as "today" — with no line of output
/// saying otherwise. Same family as the two hard-coded `false`s this probe already shipped
/// (`frac_transport`, S148; `remove_infrasonic`, S155): *the arm is wired* and *the arm carries
/// production's value* are different facts.
///
/// ⛔ This is a **mirror**, so it can drift. `vocal_range`'s `the_probe_defaults_are_the_production
/// _defaults` binds it to the real knobs — flip a default there without touching this table and
/// that test goes red. (Same shape as `RANGE_ALGO_VERSION` ↔ `audition_cache_tag`.)
///
/// Booleans are 0.0 / 1.0; the rest are the knob's own unit (ms, periods, fraction).
pub const PROBE_ARM_DEFAULTS: [(&str, f64); 9] = [
    ("UTAI_PSOLA_FRAC", 0.0),
    ("UTAI_PSOLA_WSOLA", 0.0),
    ("UTAI_PSOLA_LOCK", 0.30),
    ("UTAI_PSOLA_HP", 1.0),
    ("UTAI_PSOLA_HP_MS", 0.0),
    ("UTAI_PSOLA_ENVFIX", 0.0),
    ("UTAI_PSOLA_BRIDGE", 30.0),
    ("UTAI_PSOLA_WIN", 1.0),
    ("UTAI_PSOLA_XGRAIN", 1.0),
];

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

/// S154 — **bridge short unvoiced gaps in the fed f0 so the islands cover the whole note.**
///
/// ## The defect this exists for
///
/// This process only shifts **inside voiced islands**; outside them the input is passed through
/// *bit for bit* (measured on the probe: residual −341 … −385 dB, i.e. float noise). The fed f0
/// is zeroed on unvoiced phones, so the island boundary lands **exactly on the vowel onset** —
/// and the join between the two is **0.25 … 3 ms** wide (measured: the residual against the input
/// collapses from −13 dB to −348 dB within 1 ms of the edge).
///
/// ⇒ every rescued note keeps a fragment of **un-shifted, 9-14 semitones too low** audio at each
/// end, butted against the shifted body across a sub-millisecond join. In the spectrogram every
/// harmonic **steps** at that instant — which is a broadband vertical line — and in the waveform
/// the attack stands off from the body as an abrupt spike.
///
/// ⭐ How much leaks, at the notes the user annotated (outside-island peak ÷ inside-island peak,
/// and outside/inside energy): 698 "normal" **0.30× / −22.8 dB** · ぴゃ "no line here" **0.27× /
/// −21.5** · 719 "abnormal" **0.65× / −16.9** · 781 "abnormal" **1.35×** · 753 "abnormal"
/// **2.09× / −1.06**. ⇒ **the first quantity on this line that orders the user's labels**, and it
/// separates *within* a single shift (698 vs 719, both −9), so it is not just a ratio proxy.
///
/// ⭐ It also matches every negative control the user gave by ear: signalsmith has **no islands**
/// (it shifts the whole stream) ⇒ no boundary ⇒ no defect; the donor that never went through
/// PSOLA has no boundary either; the defect is at note heads **and** tails (an island has two
/// ends) and **never in the middle of a note** (there is no boundary there).
///
/// ## What the knob does
///
/// Fill zero runs **shorter than `max_ms`, and interior only**, by interpolating the neighbouring
/// voiced values, before the islands are cut. A real rest is longer than the bound and stays a
/// rest. `0.0 = off = byte-for-byte the pre-S154 arm.`
///
/// ⚠ It moves `analysis_marks`' seed (that function seeds at the island **midpoint**), so the
/// whole mark train shifts phase. S153 rejected a whole-track version of this for exactly that
/// reason — but that rejection was about comparing the island **interiors**; the **boundary** is
/// what this is for, and it is the one thing that comparison can answer cleanly.
fn bridge_unvoiced(f0: &[f32], hop: usize, sample_rate: u32, max_ms: f64) -> Vec<f32> {
    // ⛔⛔ **First version filled only the zero runs BETWEEN two voiced runs, bounded by `max_ms`.**
    // Measured on the real material: it changed **nothing** — island count 7→7 and 5→5 at 40/80 ms.
    // The gaps here are longer than 150 ms; the un-shifted attack is not a short consonant wedged
    // between islands, it sits inside a long unvoiced stretch belonging to the note itself.
    // ⇒ what the defect needs is a **dilation**: grow every voiced run outward, holding that run's
    // own edge value, so the island covers the note's own onset and release.
    //
    // ⛔⛔ **Second version let the two sides meet, and that merged islands.** On 炉心融解 a 30 ms
    // dilation merges nothing (96 → 96); on the goose score, whose unvoiced gaps are far shorter,
    // it collapsed **458 → 143** — it was fusing neighbouring notes into a single rescue. Covering
    // a note's own onset is the job; merging notes is not (S151 paid for the general form of this:
    // "merging across a SUNG note drags a passenger into the rescue").
    // ⇒ each gap is split at its midpoint and **at least one frame is always left unvoiced**, so
    // the island count is preserved on every material and at every width.
    // ⚠ On 炉心融解 this guard is inert (its gaps are ≫ 2 × 30 ms), so the render the user
    // confirmed by ear is untouched — checked against the probe, not assumed.
    let ext = if hop == 0 {
        0
    } else {
        ((max_ms / 1000.0) * f64::from(sample_rate) / hop as f64).round().max(0.0) as usize
    };
    if ext == 0 {
        return f0.to_vec();
    }
    let n = f0.len();
    let mut out = f0.to_vec();
    let mut i = 0usize;
    while i < n {
        if f0[i] > 0.0 {
            i += 1;
            continue;
        }
        let a = i;
        let mut b = i;
        while b < n && !(f0[b] > 0.0) {
            b += 1;
        }
        let len = b - a;
        // ⛔ Interior only: a leading or trailing run has no anchor on one side, and inventing a
        // pitch where the score says there is none is not this function's business.
        let left = if a > 0 { Some(f0[a - 1]) } else { None };
        let right = if b < n { Some(f0[b]) } else { None };
        // The frame that must stay unvoiced so the two islands never touch.
        let keep = a + len / 2;
        for (k, slot) in out.iter_mut().enumerate().take(b).skip(a) {
            if k == keep {
                continue;
            }
            if k < keep {
                if let Some(v) = left {
                    if k - a < ext {
                        *slot = v;
                    }
                }
            } else if let Some(v) = right {
                if b - 1 - k < ext {
                    *slot = v;
                }
            }
        }
        i = b.max(a + 1);
    }
    out
}

/// S154 — the midpoint of the **undilated** island that a dilated island grew out of, so the mark
/// train keeps the phase it had before. `None` (no raw islands, i.e. the arm is off, or no overlap)
/// falls back to the dilated island's own midpoint = the legacy behaviour.
///
/// ⚠ Overlap, not containment: a dilated island can swallow more than one raw island (that is the
/// bridging case). Seed from the **first** one — arbitrary but stable, and stability is the whole
/// point of this function.
fn seed_mid(raw: &[(usize, usize)], a: usize, b: usize) -> Option<f64> {
    raw.iter().find(|(ra, rb)| *rb > a && *ra < b).map(|(ra, rb)| (*ra + *rb) as f64 * 0.5)
}

/// S154 — the window the **readout** [`PsolaDiagnostics::env_dev_p50_db`] uses, in ms.
/// Fixed so the number is comparable across runs and knob settings; the *fix* takes its own width.
const ENV_READ_MS: f64 = 5.0;

/// S154 — how far the restoration gain may ever move a sample, dB. A guard rail, not a parameter:
/// a real envelope violation on this line is 0.5-11 dB, so ±12 only ever catches a division by an
/// envelope that has collapsed (silence, a mis-detected island edge) — the case where a corrective
/// gain would otherwise explode.
const ENV_RESTORE_CLAMP_DB: f64 = 12.0;

/// Short-time RMS envelope, window `2*half+1` samples, shrinking at the two ends.
/// Prefix-sum, so it is O(n) and — unlike a per-sample loop over a slice — does not change cost
/// with the window width.
fn rms_envelope(x: &[f32], half: usize) -> Vec<f64> {
    let n = x.len();
    let mut c = vec![0.0f64; n + 1];
    for i in 0..n {
        let v = f64::from(x[i]);
        c[i + 1] = c[i] + v * v;
    }
    let mut e = vec![0.0f64; n];
    for (i, slot) in e.iter_mut().enumerate() {
        let a = i.saturating_sub(half);
        let b = (i + half + 1).min(n);
        *slot = ((c[b] - c[a]) / (b - a) as f64).max(0.0).sqrt();
    }
    e
}

/// Median |env(out) / env(in)| in dB over `covered`, ignoring anything more than 40 dB under the
/// input's peak (there the ratio is noise-on-noise and says nothing).
fn env_dev_p50_db(ey: &[f64], ex: &[f64], covered: &[bool]) -> f64 {
    let peak = ex.iter().fold(0.0f64, |m, v| m.max(*v));
    if peak <= 0.0 {
        return 0.0;
    }
    let floor = peak * 10f64.powf(-40.0 / 20.0);
    let mut v: Vec<f64> = Vec::new();
    for i in 0..ey.len().min(ex.len()).min(covered.len()) {
        if covered[i] && ex[i] > floor && ey[i] > 0.0 {
            v.push((20.0 * (ey[i] / ex[i]).log10()).abs());
        }
    }
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// S154 — **make the process keep the amplitude envelope it was given.**
///
/// Inside the covered span only, rescale the output so its short-time RMS envelope matches the
/// input's. Outside it the samples are not touched at all, so the covered/uncovered boundary
/// cannot be moved by this arm (that boundary is where the defect lives, so it matters that the
/// fix does not manufacture a second one).
///
/// ⚠ **What this must not do**: a gain that varies fast IS a modulation, and if the window is
/// short enough to track individual source periods the division re-injects the *donor's* f0 into
/// the output. Measured on the probe (donor-f0 leakage, harmonics of `f0_don` that do not
/// coincide with `f0_out`): today −34.7 dB, and after restoration at 2 / 5 / 10 / 20 ms
/// **−31.6 / −32.8 / −34.1 / −34.3** ⇒ under about 10 ms the cost starts to show. Pick the width
/// with that in mind; the knob is in milliseconds precisely so the trade is explicit.
/// ⛔⛔ **Two things here are not decoration, they are what makes the arm not backfire.**
///
/// 1. **The gain is smoothed before it is applied.** `env(g·y) == g·env(y)` only holds while `g`
///    is constant across the window; a raw per-sample `ex/ey` is not, so applying it lands
///    somewhere near — but not at — the target, and the miss is a *new* fast amplitude wobble.
/// 2. **Two passes.** One pass leaves the residue from (1). Measured on the very fixture that
///    caught this: at +1 st, where the violation is only 0.37 dB to begin with, a single
///    unsmoothed pass made the deviation **worse** (0.37 → 1.05 dB). ⇒ a corrective arm has to be
///    tested at the shifts where there is almost nothing to correct, not just where the defect is
///    big — otherwise "it helps" is only ever measured where it cannot lose.
fn restore_envelope(out: &mut [f32], x: &[f32], covered: &[bool], half: usize) {
    let ex = rms_envelope(x, half);
    let peak = ex.iter().fold(0.0f64, |m, v| m.max(*v));
    if peak <= 0.0 {
        return;
    }
    let floor = peak * 10f64.powf(-60.0 / 20.0);
    let lo = 10f64.powf(-ENV_RESTORE_CLAMP_DB / 20.0);
    let hi = 10f64.powf(ENV_RESTORE_CLAMP_DB / 20.0);
    for _pass in 0..2 {
        let ey = rms_envelope(out, half);
        let raw: Vec<f64> = (0..out.len())
            .map(|i| {
                if covered[i] && ex[i] > floor && ey[i] > floor {
                    (ex[i] / ey[i]).clamp(lo, hi)
                } else {
                    1.0
                }
            })
            .collect();
        // ⛔ The gain is smoothed **wider than the measurement window** on purpose: it may only
        // carry the SLOW part of the correction. A step at an island start is 10-40 ms wide;
        // anything faster than that is period-scale, and a gain that tracked it would be
        // re-injecting the donor's fundamental (see the leakage numbers on this function).
        let g = box_average(&raw, half * 4);
        for i in 0..out.len() {
            if covered[i] {
                out[i] = (f64::from(out[i]) * g[i]) as f32;
            }
        }
    }
}

/// The infrasonic baseline of `x` at the **ruler's** shape (2 box passes = a triangular
/// low-pass). ⛔ The **cut** uses [`CUT_BOX_PASSES`]; see [`infrasonic_baseline_passes`]. See
/// [`INFRASONIC_MA_MS`] for the measured response.
fn infrasonic_baseline(x: &[f32], sample_rate: u32) -> Vec<f64> {
    infrasonic_baseline_ms(x, sample_rate, INFRASONIC_MA_MS)
}

/// Same, with the width given explicitly — the constant has to be **scannable by a criterion**,
/// or "someone widened it back to 20 ms" and "the arm works" look identical from every test here.
/// (Measured: with the width hard-coded, changing 8 → 20 ms left the whole file green while the
/// benefit halved. That is the S147 silent-halving shape.)
fn infrasonic_baseline_ms(x: &[f32], sample_rate: u32, ms: f64) -> Vec<f64> {
    infrasonic_baseline_passes(x, sample_rate, ms, RULER_BOX_PASSES)
}

/// S155 笔5 —— **几遍盒**。`RULER_BOX_PASSES` 是那把**尺子**([`PsolaDiagnostics::infrasonic_frac`])
/// 的遍数,永远是 2;[`CUT_BOX_PASSES`] 是**刀**的遍数。
///
/// ## ⛔⛔ 为什么刀非得是 4 遍(用户听出来的,而且我事先量到过又照样上线)
///
/// 差分式 `out -= LP(out) − LP(in)` 里的 `+LP(in)` 是**一份低通过的 donor**,而 donor 唱得更低。
/// 两遍盒的**第一个旁瓣只有 −27 dB**,所以只要被救的那个音的基频落在旁瓣上,
/// 它就会被原样加回输出 ⇒ 输出里多出**第二个音高** ⇒ 用户 2026-08-19 的原话:
/// 「f0 附近偏下多了一道有点时长的共振峰伪影,听起来甚至有一点**合唱感**」。
///
/// ⭐ 排序是他先听出来的,仪器完全对上(用户点名的 244.9-245.96 s,落在 −12 窗 ⇒
/// donor 基频 = f_out/2 = 310.6 Hz;该带能量相对各自 400-4000 Hz):
///
/// | 臂 | f_out/2 附近 |
/// |---|---|
/// | 无扩展(根本没有这道工序) | −48.7 dB |
/// | 不开这把刀 | −42.8 |
/// | 固定 8 ms | −33.8 |
/// | **自适应 4.36 ms(shipped)** | **−25.7** |
///
/// 渲染地板(同设置两跑)只有 **0.19 dB**。解析对拍:两遍盒在 310.6 Hz 上
/// 4.36 ms 读 −27.05 dB、8.0 ms 读 −35.73 dB ⇒ 预言两臂差 **+8.68 dB**,实测 **+8.10**。
///
/// ⛔ **根因不是宽度选错了**:宽度规则把第一个零点放在**这段缓冲里最低**的基频上,而**被救的
/// 那个音**的基频比它高 ⇒ 落在旁瓣里。旁瓣有多深是滤波器的性质,不是宽度的性质。
/// ⇒ 4 遍盒把「被救音的基频高 1.5 倍」这一档从 **−26.8 dB** 压到 **−53.6 dB**(解析),
/// 零地板探针上实测:s14 不加刀 −41.42 → 2 遍 **−34.31**(漏了 +7.1)→ 4 遍 **−41.44**(漏没了)。
/// ⚠ 代价:同宽度下拿掉量少 **2-3 dB**(20-125 Hz)。⭐ 但用户已经确认那条底部亮带在
/// 拿掉量更少的「固定 8 ms」臂上就已经消失了 ⇒ 余量充足,而合唱感是他明确说更糟的那一条。
fn infrasonic_baseline_passes(x: &[f32], sample_rate: u32, ms: f64, passes: usize) -> Vec<f64> {
    let half = (((f64::from(sample_rate) * ms / 1000.0) as usize) / 2).max(1);
    let mut v: Vec<f64> = x.iter().map(|s| f64::from(*s)).collect();
    for _ in 0..passes {
        v = box_average(&v, half);
    }
    v
}

/// The **ruler**'s box passes. ⛔ Never change it: every historical `infrasonic_frac` on this line
/// (S152's 10.8/25.5/33.7 %, S154's tables, S155's 8.84/23.00/17.27 %) was taken with 2 passes,
/// and a ruler whose shape follows the knob is not a ruler.
const RULER_BOX_PASSES: usize = 2;
/// The **knife**'s box passes. See [`infrasonic_baseline_passes`] for why it is 4 and not 2.
const CUT_BOX_PASSES: usize = 4;

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
    // S156 —— 这一颗粒的增益。`xgrain` 把一颗粒拆成**相邻两个源脉冲**的加权和时用它。
    // ⛔ `gain == 1.0` 时 `v * w * 1.0` 与 `v * w` 在 IEEE 下**逐位相同** ⇒ 今天不变。
    gain: f64,
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
            acc[ti as usize] += v * w * gain;
            wsum[ti as usize] += w * gain;
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
        phase_lock, Infrasonic::Off,
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
/// ## What it buys
///
/// ⛔ The first version of this paragraph said "it is not an audibility fix". That was wrong for
/// the reason spelled out on [`PsolaDiagnostics::infrasonic_frac`]: the energy share is
/// inaudible, but the **steps** put **+16 to +24 dB into 20-60 Hz** at exactly the moments the
/// user calls seams. With the cut at 12 ms this arm puts that band back within **0.4 dB** of the
/// un-rescued render while moving the fundamental by **0.02 dB**.
/// It also stops every RMS-domain ruler we own from silently counting a DC offset as signal, and
/// stops the waveform riding off-centre (the shape the user named "波形甚是诡异" on `[685]`).
///
/// ## Contract (S155 rewrote this section — the two ⛔ lines below used to say the opposite)
///
/// * The removal is `out -= LP(out) - LP(in)`: **differential**, not `out -= LP(out)`.
///   Linear, zero-phase, band-limited; width per [`Infrasonic`].
/// * ⭐ It **is** bit-exact at ratio 1.0. The earlier version of this paragraph said a linear
///   filter never is, and that was true of the non-differential form — which is why the arm was
///   off by default and why `ratio_one_is_the_identity` (the cheapest non-self-certifying gate on
///   this line; it killed three designs in S146 that "looked right") could not survive turning it
///   on. The differential form settles it structurally instead of by exemption: at ratio 1.0
///   `out ≡ x`, so `LP(out)` and `LP(x)` are the same bytes through the same code and the
///   correction is **exactly 0.0**. ⛔ Note what this is NOT: a `semitones == 0` short-circuit.
///   That shortcut would make the gate vacuously true, which is the exact shape that let the
///   2026-07 implementation through (see the note at the top of [`psola_shift_env`]).
/// * ⭐ Outside the voiced islands the correction is ≈0 for the same reason (`out ≡ x` there,
///   bit-for-bit), so the un-transposed pass-through donor keeps its own low end — and it gets
///   that **without a mask**, which would put a step at the island boundary, i.e. exactly the
///   defect S154 spent a session removing.
/// * The gate for the arm being ON is still
///   `the_infrasonic_arm_leaves_everything_above_the_fundamental_alone`, and it is now joined by
///   `ratio_one_is_the_identity_even_with_the_infrasonic_arm_on`.
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
    infrasonic: Infrasonic,
) -> (Vec<f32>, PsolaDiagnostics) {
    psola_shift_env(
        x, sample_rate, semitones, formant_semitones, f0_hz, f0_hop, frac_transport, wsola_frac,
        phase_lock, infrasonic, 0.0, 0.0, 0.0, 0.0)
}

/// S154 — additive: optionally **restore the amplitude envelope** the process was handed.
///
/// `env_restore_ms = 0.0` is byte-for-byte the pre-S154 arm and is what production runs.
/// See [`PsolaDiagnostics::env_dev_p50_db`] for what it is for and [`restore_envelope`] for the
/// contract and the measured cost of a too-short window.
///
/// ⛔ Why it is a **width in milliseconds** and not a bool: the width is the whole trade. Too long
/// and it cannot follow an attack (which is where the violation is); too short and the gain starts
/// tracking individual source periods, which puts the donor's fundamental back into the output.
#[allow(clippy::too_many_arguments)]
pub fn psola_shift_env(
    x: &[f32],
    sample_rate: u32,
    semitones: f64,
    formant_semitones: f64,
    f0_hz: &[f32],
    f0_hop: usize,
    frac_transport: bool,
    wsola_frac: f64,
    phase_lock: f64,
    infrasonic: Infrasonic,
    env_restore_ms: f64,
    bridge_unvoiced_ms: f64,
    win_periods: f64,
    // S156 —— 颗粒内容在相邻两个源脉冲之间的**插值深度**,0…1。
    // `0.0` = 今天 = 最近邻 `k = round(u)` = 逐位不变;`1.0` = 完全线性插值。
    // 它存在的理由与它为什么在 ratio 1.0 上对任何深度都恒等,写在主循环里那一段。
    xgrain: f64,
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

    // S154 —— 岛的划法。⛔ 默认 0 = 不膨胀 = 逐位同旧;见 `bridge_unvoiced` 的说明。
    // `islands_raw` 是**膨胀前**的岛,只用来给标记播种(见 `analysis_marks` 的 `seed_at`)。
    let min_island = (MIN_ISLAND_SECONDS * sr) as usize;
    let islands_raw: Vec<(usize, usize)> = if bridge_unvoiced_ms > 0.0 {
        voiced_islands(f0_hz, f0_hop, n, min_island)
    } else {
        Vec::new()
    };
    let bridged: Vec<f32>;
    let f0_hz: &[f32] = if bridge_unvoiced_ms > 0.0 {
        bridged = bridge_unvoiced(f0_hz, f0_hop, sample_rate, bridge_unvoiced_ms);
        &bridged
    } else {
        f0_hz
    };

    let mut acc = vec![0.0f64; n];
    let mut wsum = vec![0.0f64; n];
    let mut covered = vec![false; n];
    // S155 笔4/笔6 —— 每颗粒的源周期,只收**读窗里真的有音频**的那些,而且**逐岛分开收**。
    // 见 `infrasonic_width_ms`(为什么用颗粒的周期)与去次声那一段(为什么必须逐岛)。
    // 每条 = (岛的覆盖起点, 覆盖终点, 该岛所有颗粒的源周期)。
    let mut islands_periods: Vec<(usize, usize, Vec<f64>)> = Vec::new();
    let mut this_island_periods: Vec<f64> = Vec::new();
    let max_period = MAX_PERIOD_SECONDS * sr;
    // S151 源覆盖率的累加器(见 `PsolaDiagnostics::src_uncovered_frac`)。
    let (mut uncovered, mut span) = (0.0f64, 0.0f64);

    for (a, b) in voiced_islands(f0_hz, f0_hop, n, (MIN_ISLAND_SECONDS * sr) as usize) {
        // S154 —— 播种点用**没膨胀**的岛的中点(见 `analysis_marks` 的说明):
        // 膨胀只该改「哪些样本被移调」,不该把每个音的标记相位重掷一次。
        let seed = seed_mid(&islands_raw, a, b);
        let mut src = analysis_marks(&dc_free, sample_rate, f0_hz, f0_hop, a, b, seed);
        if src.len() < 3 {
            continue;
        }
        // S150 — marks are found on the DC-free signal, so they are locked on it too.
        diag.marks_locked += lock_phase(&dc_free, &mut src, phase_lock);
        diag.islands += 1;
        diag.marks += src.len();
        let last = (src.len() - 1) as f64;
        // ⚠ S154 —— 一条**试过并且没成的**修法,留着免得下一个人再试一遍:
        // 给合成栅格加一个常数相位、让它仍然穿过膨胀前的种子标记 —— **做不到保住岛内**。
        // 原因是结构性的:岛变长 ⇒ 合成脉冲**多了几颗** ⇒ 整条脉冲串的相位必然跟着走,
        // 除非「多出来的颗数 × ratio」正好是整数。⇒ 「只改边界、岛内逐位不变」在这条路上不存在。
        // ⭐ 但**分析标记**是保住了的(见 `analysis_marks` 的 `seed_at`),动的只有合成栅格的相位,
        //    而脉冲串的绝对相位听不见 —— 判据应当是「远离岛边处**包络**变没变」,不是「波形变没变」。
        let count = (last * ratio) as usize;
        let (mut island_first, mut cover_end) = (f64::NAN, f64::NAN);
        let mut tgt: Vec<f64> = Vec::with_capacity(count + 1);
        let mut ks: Vec<usize> = Vec::with_capacity(count + 1);
        // S156 —— `xgrain` 要的是 `u` 本身(它在源标记的**下标**轴上的小数部分),
        // 而 `ks` 已经把它四舍五入掉了。⚠ 别想着从 `tm` 反查:那是 `bench.py` 的做法,
        // 在源周期抖动时与这里的 `u` 不是同一个数。
        let mut us: Vec<f64> = Vec::with_capacity(count + 1);
        for j in 0..=count {
            let u = j as f64 / ratio;
            if u > last {
                break;
            }
            let lo = u as usize;
            let hi = (lo + 1).min(src.len() - 1);
            tgt.push(src[lo] + (src[hi] - src[lo]) * (u - lo as f64));
            ks.push(u.round() as usize);
            us.push(u);
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
            // S155 —— 读窗。**今天** `lw = rw = min(T_out, T_src)`,而上移时 `T_out = T_src/ratio`
            // ⇒ 读窗总宽 = **2/ratio 个源周期**(逐颗粒实测 p05-p95:位移 −9 → 1.189、
            // −12 → **1.000**、−14 → **0.891**)。教科书 TD-PSOLA 是 ±1 个源周期 = 总宽 **2.000**。
            //
            // ⭐ 这个自由度是**高次共振峰**那条线的头号候选,而且是被**干预**证明的,不是相关:
            //   同一段音频、同一批颗粒、同一个比值,只改窗宽(离线台子,自检 −85…−93 dB),
            //   谱包络对比度相对 donor 的损失(dB,越接近 0 越好):
            //
            //   | ratio 2.0 (ぴゃ 那一档) | 2-4 kHz | 4-6 kHz | 6-8 kHz | 8-12 kHz |
            //   |---|---|---|---|---|
            //   | 今天(半宽 0.50) | −0.834 | −0.666 | −0.606 | −0.866 |
            //   | 教科书(半宽 1.00) | **−0.360** | **−0.209** | **−0.189** | **−0.277** |
            //   | 今天窗宽 + 除 wsum | −0.949 | −0.673 | −0.577 | −0.871 |
            //
            //   ⭐ 最后一行是把自由度分离开的那条对照:**「除 wsum」单独什么也不做,是窗宽**。
            // ⛔ 但它是**取舍不是纯赚**:同一组臂上谐波间噪声(300-2500 Hz)从 −2.78 掉到 −0.86 dB
            //   (三条臂 +1…+2 dB),而 300-2500 Hz 正是「咔哒」那条带。
            //   ⇒ 这种取舍只有耳朵能裁(S146 协议),所以这里**默认 0 = 今天**,逐位不变。
            //
            // ⭐⭐ S156 用一把**零插值**的判据把上面两条都重量了一遍(位移 +12 ⇒ ratio 恰好 2.0
            //   ⇒ 输出第 k 根谐波与 donor 第 2k 根**逐根重合**,不需要倒谱平滑也不需要 lifter;
            //   ⛔ S155 那把尺子的 lifter 是按各自 f0 取的,而它的「零点验过」是 ratio 1.0 那条臂 ——
            //   那里 f_out == f_in ⇒ 那条偏置在结构上恰好为 0,所以那个零点根本没检验过它;
            //   合成夹具上实测偏置 −1.0…−5.4 dB,与它报的「损失」同量级同方向):
            //
            //   | 臂 | 形状 rms(2-12k) | 8-12k 相对 300-1k 的倾斜 | 岛内 rms | 0.5·f_out |
            //   |---|---|---|---|---|
            //   | 今天 | 3.87 | **−5.14** | 0.00 | −37.9 |
            //   | 半宽 1.0 **不除 wsum** | **0.83** | **−1.07** | **+0.49** | −33.6 |
            //   | 半宽 1.0 除 wsum | 0.83 | −1.07 | **−5.53** | −33.5 |
            //
            //   ⇒ ⑴ 收益比 S155 记的大得多(形状偏差 4.7×,而不是「拉回 57-69%」);
            //   ⑵ **「除 wsum」与「不除」的谱形状读数一模一样,只差 20log10(ratio) 的常数**
            //      ⇒ 那 −5.53 dB 是**除 wsum 的代价**,不是宽窗的代价 ⇒ S156 改成不除(见下面输出合成那段);
            //   ⑶ 剩下的真代价是 **0.5·f_out 上多出来的 +4.3 dB = donor 自己的音高**
            //      —— 正是用户在 S155 笔5 听成「合唱感」的那一条。机理:ratio 2.0 时相邻两颗输出颗粒
            //      读**同一个源标记**(`k = round(j/ratio)`)⇒ 颗粒成对 ⇒ 输出带着周期 `2·T_out = T_src`
            //      的结构。⇒ 这就是 `xgrain`(相邻源脉冲线性插值)存在的理由,见下面 `xgrain` 那一段。
            let (lw, rw) = if win_periods > 0.0 {
                (win_periods * src_l, win_periods * src_r)
            } else {
                ((tm - tl).min(src_l), (tr - tm).min(src_r))
            };
            // ⚠ 上界要跟着窗宽走,否则宽窗臂会**静默地把颗粒全部跳过** —— 那是「干预没生效」
            //   被读成「干预无效」的形状(S148 的 `frac_transport` 写死成 false 是同一族)。
            let wmax = max_period * win_periods.max(1.0);
            if lw <= 1.0 || rw <= 1.0 || lw > wmax || rw > wmax {
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
            // S156 —— **xgrain**:颗粒的**内容**在相邻两个源脉冲之间插值,而不是四舍五入到最近的那个。
            //
            // ⛔ 为什么需要它:`k = round(u)` 让相邻若干颗输出颗粒**读同一个源标记**(ratio 2.0 时
            //   正好成对)⇒ 输出带着周期 `2·T_out = T_src` 的结构 ⇒ **donor 自己的音高**出现在
            //   `0.5·f_out` 上。窗一放宽,成对的两颗重叠更多,这条就更响:离线台子实测(s12,+12,
            //   相对各自 400-4000 Hz)今天 **−37.9** → 半宽 1.0 **−33.6**(差 +4.3 dB),
            //   而 donor 输入自己的结构地板是 −45.5。⭐ 那正是用户在 S155 笔5 亲耳听成
            //   **「合唱感」**的那一维(他当时的描述:在 f0 附近偏下 · 有一点时长 · 不在底部)。
            //   开 xgrain 之后同一条读 **−38.7** ⇒ 宽窗引来的那 4.3 dB 被消掉,回到今天的水平。
            //
            // ⭐ 为什么它在 `ratio == 1.0` 上**结构性**恒等,而且对**任何**深度都成立:
            //   那时 `u = j` 是整数 ⇒ `fr = 0` ⇒ 最近邻权重与线性权重**是同一个向量** `(1, 0)`
            //   ⇒ 两者的任意凸组合还是它。(这一点比 `win_periods` 强 —— 那个只在 1.0 上恒等。)
            //
            // ⚠ `xgrain == 0.0` 时走的是**原来那一行**,不是「权重 (1,0) 的两颗粒」——
            //   后者依赖 `round()` 与 `fr < 0.5` 的等价性,而我不想让逐位恒等挂在那个等价性上。
            let (p0, g0, p1, g1) = if xgrain > 0.0 {
                let uu = us[i];
                let lo = (uu as usize).min(src.len() - 1);
                let hi = (lo + 1).min(src.len() - 1);
                let fr = (uu - lo as f64).clamp(0.0, 1.0);
                let (nl, nh) = if fr < 0.5 { (1.0, 0.0) } else { (0.0, 1.0) };
                // wsola 挪的是**读点**,对两颗粒施加同一个位移(wsola 默认 0 ⇒ off = 0)。
                let off = s_pos - src[k];
                (
                    src[lo] + off,
                    (1.0 - xgrain) * nl + xgrain * (1.0 - fr),
                    src[hi] + off,
                    (1.0 - xgrain) * nh + xgrain * fr,
                )
            } else {
                (s_pos, 1.0, 0.0, 0.0)
            };
            // S151 —— **源覆盖率**:这一颗粒从源上读的是 `[s_pos − lw, s_pos + rw)`。上移时
            // `lw = rw = T_src / ratio`(上面那两个 `min` 取的是目标邻距),所以一旦
            // `ratio > 2`(= |位移| > 12 半音),相邻两个读窗之间就留下一段**永远不进任何颗粒**
            // 的源波形。见 `SRC_UNCOVERED` 的注释:这是唯一直接看得见它的读数。
            // ⚠ xgrain 开着时真的读了两段 ⇒ 覆盖率算**并集**,否则这只眼睛会低报。
            // ⛔⛔ 但**只算权重 ≥ 0.5 的那些读点**,否则它会灌水:`p1 = p0 + T_src`,于是权重
            //   0.001 的第二个读点也会让整整一个源周期被记成「读过了」⇒ **窄窗也能读出零漏源**。
            //   这不是假想 —— S156 翻默认时,新加的那条生产口径判据就是被这样骗过去的
            //   (把 `WIN` 退回 0,`src_uncovered_frac` 照样是 0,判据当场变空)。
            //   ⇒ 门限取 0.5:一颗粒最多只有一个读点能过(`fr == 0.5` 时两个都是 0.5,那是真并集),
            //   于是这只眼睛**永远不会比 xgrain 关着时更乐观**。
            let (rs, re) = if g1 >= 0.5 && g0 >= 0.5 {
                (p0.min(p1) - lw, p0.max(p1) + rw)
            } else if g1 >= 0.5 {
                (p1 - lw, p1 + rw)
            } else {
                (p0 - lw, p0 + rw)
            };
            if cover_end.is_nan() {
                island_first = rs;
                cover_end = rs;
            }
            if rs > cover_end {
                uncovered += rs - cover_end;
            }
            cover_end = cover_end.max(re);
            {
                let a = (s_pos - lw).round().max(0.0) as usize;
                let b = ((s_pos + rw).round().max(0.0) as usize).min(n);
                if a < b && x[a..b].iter().any(|v| v.abs() > 1e-7) {
                    this_island_periods.push(src_l.max(src_r));
                }
            }
            // ⚠ xgrain 开着时这里会放**两颗**颗粒 ⇒ `transport_residual` 每个合成标记记两笔
            //   (两次读点各有各的亚样本残差)。那是如实记账,不是缺陷 —— 但引用那个读数时
            //   要知道 xgrain 那条臂的样本数是别的臂的两倍。`xgrain == 0` 时逐位不变。
            for (pp, gg) in [(p0, g0), (p1, g1)] {
                if gg <= 0.0 {
                    continue;
                }
                add_bell(
                    x, &mut acc, &mut wsum, pp, tm, lw, rw, formant_rate, frac_transport, gg,
                    &mut residual,
                );
            }
        }
        if !cover_end.is_nan() {
            span += (cover_end - island_first).max(0.0);
        }
        if !this_island_periods.is_empty() {
            islands_periods.push((c0, c1, std::mem::take(&mut this_island_periods)));
        } else {
            this_island_periods.clear();
        }
    }

    let mut gap = 0usize;
    let mut cov_n = 0usize;
    let mut ws: Vec<f64> = Vec::new();
    let mut out = vec![0.0f32; n];
    // S156 —— **稳态窗和**。半宽 `W` 个源周期的半余弦,以邻距 `T_out = T_src/ratio` 铺开
    // ⇒ Σw ≡ `W·ratio`(离线台子实测:W = 0.75/1.00/1.25/1.50 上读到 1.5000/2.0000/2.5000/3.0000,
    // 四档全中,见 `s156_knives/run_arms3.py`)。今天(`win_periods == 0`)窗宽 = 邻距 ⇒ `W̄ = 1`。
    //
    // ⛔ `.max(1.0)` 不是保险丝,它是**下移臂的恒等条件**:下移时 `lw = rw = T_src`(两个 `min` 取源
    //    周期),`W·ratio = ratio < 1`,而今天在那里用的就是 `clamp(raw, 0, 1)` ⇒ 不夹住的话
    //    「颗粒逐位相同、只有干填料变了」,`win_periods = 1.0` 在下移上就不再等于今天。
    let wbar = if win_periods > 0.0 { (win_periods * ratio).max(1.0) } else { 1.0 };
    for i in 0..n {
        // ⛔ Statistics read the RAW sum; the clamp below is only what the dry-fill gain needs.
        // Clamping first made the surplus (w > 1) unreadable — see PsolaDiagnostics.
        // S156 —— 读数与干填料都按 `W̄` 归一,所以 `cola_*` 在宽窗臂上仍然是「COLA 有没有破」的
        // 读数,而不是重叠系数。`win_periods == 0` 时 `W̄ = 1.0`,除以 1.0 在 IEEE 下逐位精确
        // ⇒ 这一整段对今天**逐位不变**。
        let raw = wsum[i] / wbar;
        let w = raw.clamp(0.0, 1.0);
        if covered[i] {
            cov_n += 1;
            if raw < 0.9 {
                gap += 1;
            }
            ws.push(raw);
            // ⛔⛔ S156 —— **不除 wsum**,连宽窗臂也不除。S155 那一版除了,而那正是它「要付
            // −4.3…−5.8 dB 电平」的**唯一**来源:离线台子上「除 wsum」与「不除」的逐带谱形状读数
            // **一模一样**,只差一个 20log10(ratio) = +6.02 dB 的常数(s12,ratio 2.0)。
            // ⇒ 那笔代价不是宽窗的代价,是**选择除 wsum** 的代价,而且它是个与频率无关的标量。
            // ⇒ 而下游**吸收不了它**:`restore_envelope` 默认关、`peak_normalize_to` 给 donor 传的是
            //   base 的峰值(与 donor 内容无关)、`match_levels` 五个调用点全传 false = 死代码,
            //   cover 路连峰值归一都没有 ⇒ 掉多少全额落到被救的那几个音上。
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
    // S154 —— **振幅包络守恒**。读数无条件算(S152 那条规矩),修法由 `env_restore_ms` 控。
    // ⛔ 顺序:排在次声之前,因为次声那一刀是全缓冲的线性滤波,而这一刀只动岛内 ——
    //    反过来做会让次声读数里混进这一刀改的那部分。
    {
        let half = ((ENV_READ_MS * sr / 1000.0) as usize).max(4);
        let ey = rms_envelope(&out, half);
        let ex = rms_envelope(x, half);
        diag.env_dev_p50_db = env_dev_p50_db(&ey, &ex, &covered) as f32;
        if env_restore_ms > 0.0 {
            let h = ((env_restore_ms * sr / 1000.0) as usize).max(4);
            restore_envelope(&mut out, x, &covered, h);
            let ey2 = rms_envelope(&out, half);
            // 报「真的把它拉回了多少」而不是「开着」—— 与 `infrasonic_removed` 同一条规矩。
            diag.env_dev_after_db = env_dev_p50_db(&ey2, &ex, &covered) as f32;
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
        // ⛔ 读数**永远**用出厂的固定 8 ms,即使修法用的是自适应宽度。它是一把**尺子**不是刀:
        //    S152/S154 的所有历史读数都是这把尺子量的,让它跟着宽度走 = 每换一次宽度就换一次刻度。
        let before = infra_frac(&out, sample_rate);
        diag.infrasonic_frac = before as f32;
        // S155 笔6 —— **逐岛各用各的宽度**。
        //
        // ⛔ 为什么不能一个缓冲一个宽度:生产里一遍救援的 donor 缓冲**不止装被救的那些音** ——
        // 这一遍不救的音同样有真音频、同样生成颗粒(它们的输出后面会被窗丢掉),而它们的基频更低
        // ⇒ 分位数被它们拉下去 ⇒ 刀比该有的宽。实测:−14 那一遍整曲选 **4.96 ms**,
        // 而只算**真正被救的那 62 个音**应当是 **2.70 ms**,50-125 Hz 因此少削约 **5 dB**。
        //
        // ⭐ 形式上用 `d = out − x` 而不是「LP(out) − LP(x)」分开算,是因为 **d 在岛外逐位为零**:
        //   把每个岛那一段的 d 单独低通再累加,每一项都在岛外**平滑衰减到 0** ⇒ 不需要 mask、
        //   不会在岛边界造台阶(mask 造台阶正是 S154 花一场修掉的那类缺陷),
        //   而 ratio 1.0 上 d ≡ 0 ⇒ 每一项恒为 0 ⇒ **恒等仍然自动成立**。
        // ⚠ `FixedMs` 仍然是**整缓冲一个宽度**:它存在的意义就是从同一个二进制渲出旧臂。
        let per_island: Vec<(usize, usize, f64)> = match infrasonic {
            Infrasonic::PerPeriod => islands_periods
                .iter()
                .map(|(a, b, p)| (*a, *b, infrasonic_width_ms(p, sample_rate, ratio)))
                .collect(),
            _ => Vec::new(),
        };
        let ms = match infrasonic {
            Infrasonic::Off => 0.0,
            Infrasonic::PerPeriod => {
                // 报一个代表值(中位)—— 它是 S147「收益静默减半」的那只眼睛。
                let mut w: Vec<f64> = per_island.iter().map(|(_, _, m)| *m).collect();
                w.sort_by(f64::total_cmp);
                if w.is_empty() { 0.0 } else { w[w.len() / 2] }
            }
            Infrasonic::FixedMs(ms) => {
                if ms.is_finite() && ms > 0.0 { ms } else { 0.0 }
            }
        };
        diag.infrasonic_ma_ms = ms as f32;
        if ms > 0.0 {
            // S155 —— **差分式**:只减掉这道工序**多出来的**那条基线,不是输出自己的低频。
            //
            // ⭐⭐ 这一行是 ratio 1.0 恒等判据能活下来的全部原因:ratio 1.0 时 `out ≡ x`
            //     ⇒ 两条基线走同一段代码、同一份数据 ⇒ 逐位相同 ⇒ 修正项**恒为 0.0**
            //     ⇒ 恒等仍然是 `assert_eq!`,不需要任何 `if semitones == 0` 的短路
            //     (那种短路正是让 2026-07 那份实现混过去的形状,见 `psola_shift_env` 顶部)。
            // ⭐ 顺带,它在**岛外**也自动 ≈ 0(那里 `out ≡ x` 逐位相同)⇒ 那段没被移调的原始
            //     donor 不再被这把刀削低频,而且不需要写 mask —— mask 会在岛边界造台阶,
            //     那正是 S154 刚修掉的那一类缺陷。
            // 逐岛:把这一岛那一段的 out 与 x 各自单独低通(岛外补 0),再相减。
            // 与整缓冲版在数学上等价,只是每个岛可以有自己的宽度。
            let (lo_out, lo_in) = if per_island.is_empty() {
                (
                    infrasonic_baseline_passes(&out, sample_rate, ms, CUT_BOX_PASSES),
                    infrasonic_baseline_passes(x, sample_rate, ms, CUT_BOX_PASSES),
                )
            } else {
                let (mut acc_o, mut acc_i) = (vec![0.0f64; n], vec![0.0f64; n]);
                for (a, b, w) in &per_island {
                    let (a, b) = (*a, (*b).min(n));
                    if a >= b {
                        continue;
                    }
                    // 支撑:K 遍盒各半宽 ⇒ 总支撑 ≈ K·W/2 每侧。留两倍余量。
                    let half = ((f64::from(sample_rate) * *w / 1000.0) as usize / 2).max(1);
                    let pad = half * CUT_BOX_PASSES + 1;
                    let (s0, s1) = (a.saturating_sub(pad), (b + pad).min(n));
                    let mo: Vec<f32> =
                        (s0..s1).map(|i| if i >= a && i < b { out[i] } else { 0.0 }).collect();
                    let mi: Vec<f32> =
                        (s0..s1).map(|i| if i >= a && i < b { x[i] } else { 0.0 }).collect();
                    let fo = infrasonic_baseline_passes(&mo, sample_rate, *w, CUT_BOX_PASSES);
                    let fi = infrasonic_baseline_passes(&mi, sample_rate, *w, CUT_BOX_PASSES);
                    // ⛔ 护栏也逐岛:这一岛的输出基线不比输入的大 ⇒ 没有「多出来的」⇒ 不动。
                    let (eo, ei): (f64, f64) =
                        (fo.iter().map(|v| v * v).sum(), fi.iter().map(|v| v * v).sum());
                    if eo >= ei {
                        for (k, i) in (s0..s1).enumerate() {
                            acc_o[i] += fo[k];
                            acc_i[i] += fi[k];
                        }
                    }
                }
                (acc_o, acc_i)
            };
            // ⛔⛔ **护栏:这把刀永远不许往低频里加东西。**
            //
            // 差分式把 `LP(in)` 加回输出,而**低频不是被移调守恒的**:下移时输出的脉冲密度
            // 减半 ⇒ `LP(out) ≈ LP(in)/2` ⇒ 「减掉 LP(out) 再加回 LP(in)」净效果是**加**了
            // 半份进去。合成夹具上实测到了:−12 st 时基线 0.175 → **0.335**。
            // ⚠ 真素材上碰不到(donor 的低频本来就 −50 dB,而下移根本不注入:实测 −12/−7 的
            //   次声份额 0.01%/0.00%)—— 但「真素材上碰不到」是**拿阴性对照当不在场证明**,
            //   这条线已经因为它丢过一次。⇒ 用一条结构性的判据挡住,而不是靠素材挡。
            //
            // ⭐ 判据本身就是这把刀的语义:**只拿掉这道工序【多出来】的那部分**。
            //    输出的基线不比输入的大 ⇒ 没有「多出来的」⇒ 不动。
            // ⭐ ratio 1.0 时两边逐位相同 ⇒ 走 else 分支 ⇒ 恒等,而且**即使走 if 分支也恒等**
            //    (修正项恒为 0.0)⇒ 这条护栏不是恒等性的依据,只是多一道锁。
            let e_out: f64 = lo_out.iter().map(|v| v * v).sum();
            let e_in: f64 = lo_in.iter().map(|v| v * v).sum();
            // ⛔⛔ `>=` 而不是 `>`,而这一个字符是被**变异测试**逼出来的:
            //    写 `>` 的时候 ratio 1.0 上两边逐位相等 ⇒ 护栏跳过 ⇒ 恒等是**护栏**给的,
            //    不是差分式给的 ⇒ 把差分式改回「减输出自己的低频」,那条恒等判据**照样绿**。
            //    那就是一条空判据。⇒ `>=` 让 ratio 1.0 **真的走进**下面这个分支,
            //    于是恒等重新由「修正项恒为 0.0」承担,而变异当场红。
            if e_out >= e_in {
                for (i, o) in out.iter_mut().enumerate() {
                    *o = (f64::from(*o) - (lo_out[i] - lo_in[i])) as f32;
                }
                // 报「真的拿掉了多少」而不是「开着」—— 与 `wsola_moved` / `marks_locked` 同一条规矩。
                diag.infrasonic_removed = (before - infra_frac(&out, sample_rate)) as f32;
            }
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

    /// S155 —— ⭐⭐ **ratio 1.0 在这把刀【开着】的时候仍然是逐位恒等。**
    ///
    /// 这一条是差分式(`out -= LP(out) - LP(in)`)存在的**全部理由**,而它换掉的那句话就写在
    /// `psola_shift_infra` 的契约里:「a linear filter is never bit-exact at ratio 1.0,
    /// that is exactly why it is off by default」。⇒ 在这条判据出现之前,把默认翻开就等于
    /// 把这条线上**最便宜、最不可能自证**的那道闸(它在 S146 一口气杀掉三个「看起来对」的设计)
    /// 降级成 epsilon 判据。
    ///
    /// ⛔ 注意它**不是**靠 `if semitones == 0` 的短路换来的 —— 那种短路会让这道闸变成恒真的
    /// 空判据,而那正是让 2026-07 那份实现混过去的形状(见 `psola_shift_env` 顶部那段注释)。
    /// 它是结构性的:ratio 1.0 时 `out ≡ x` ⇒ 两条基线是**同一段字节走同一段代码** ⇒ 修正项
    /// 恒为 `0.0` ⇒ `assert_eq!` 而不是 `< 1e-5`。
    ///
    /// ⚠ 顺带钉住另一件事:`psola_shift_env` 顶部那句「deliberately no `semitones == 0`
    /// shortcut」必须继续成立。若有人为了让这条测试变绿而加短路,`the_infrasonic_arm_*` 那两条
    /// 会在别的位移上红,因为短路只挡 0。
    #[test]
    fn ratio_one_is_the_identity_even_with_the_infrasonic_arm_on() {
        let sr = 44_100;
        let hop = sr as usize / 200;
        // 两种源:一条纯浊音,一条带**直流**的脉冲串 —— 后者是关键,因为差分式唯一可能出事的
        // 地方就是输入自己带低频,而非差分式在这里一定会把那份直流减掉、于是恒等当场失败。
        let a = voiced(sr, 0.5, 220.0);
        let (b, _) = pulses(sr, 0.5, |_| 220.0, |_| 1.0);
        for (name, x) in [("voiced", a), ("pulses(带直流)", b)] {
            let f0t = flat_f0(x.len(), hop, 220.0);
            for arm in [Infrasonic::PerPeriod, Infrasonic::FixedMs(8.0), Infrasonic::FixedMs(2.0)] {
                let (y, d) =
                    psola_shift_infra(&x, sr, 0.0, 0.0, &f0t, hop, false, 0.0, 0.30, arm);
                assert_eq!(
                    y, x,
                    "{name} / {arm:?}:ratio 1.0 必须**逐位**恒等 —— 差分式的修正项在这里应当恒为 0"
                );
                assert_eq!(d.infrasonic_removed, 0.0, "{name} / {arm:?}:恒等却报拿掉了东西");
            }
        }
        // ⛔ 阴性对照:同一段代码在**非** 1.0 的比值上必须**不是**恒等 ——
        //    否则上面那六条 `assert_eq!` 只是在证明「这把刀什么都没做」。
        let x = voiced(sr, 0.5, 220.0);
        let f0t = flat_f0(x.len(), hop, 220.0);
        let (y, _) =
            psola_shift_infra(&x, sr, 9.0, 0.0, &f0t, hop, false, 0.0, 0.30, Infrasonic::PerPeriod);
        assert_ne!(y, x, "+9 st 上也逐位相同 ⇒ 上面那组恒等断言是空的");
    }

    /// S155 笔6 —— ⭐ **每个岛用它自己的基频定宽度,而不是整条缓冲一个宽度。**
    ///
    /// ⛔ 为什么这条判据非有不可:生产里一遍救援的 donor 缓冲**不止装被救的那些音**。
    /// 这一遍不救的音同样有真音频、同样生成颗粒(输出后面会被窗丢掉),而它们的基频更低
    /// ⇒ 分位数被它们拉下去 ⇒ 刀比该有的宽。实测:−14 那一遍整曲选 **4.96 ms**,
    /// 而只算真正被救的那 62 个音应当是 **2.70 ms** ⇒ 50-125 Hz 少削约 **5 dB**。
    /// ⇒ 没有这条判据,「改成逐岛」与「改了但没生效」在别的每一条测试上长得一模一样。
    ///
    /// 夹具:低音岛(200 Hz)+ 空档 + 高音岛(400 Hz)。整缓冲的规则会选 1000/200 = 5 ms;
    /// 逐岛应当给高音岛 1000/400 = 2.5 ms。⇒ **高音岛里逐岛臂必须比定宽 5 ms 那条削得更干净**,
    /// 而**低音岛里两者必须几乎一样**(那是阴性对照 —— 否则「更干净」可能只是整体多削了)。
    #[test]
    fn each_island_gets_its_own_cut_width() {
        let sr = 44_100;
        let hop = sr as usize / 200;
        let (lo_isl, _) = pulses(sr, 0.35, |_| 200.0, |_| 1.0);
        let (hi_isl, _) = pulses(sr, 0.35, |_| 400.0, |_| 1.0);
        let gap = vec![0.0f32; sr as usize / 5]; // 200 ms 空档 ⇒ 两个岛分得开
        let mut x = lo_isl.clone();
        x.extend_from_slice(&gap);
        let a2 = x.len();
        x.extend_from_slice(&hi_isl);
        let frames = x.len().div_ceil(hop);
        let mut f0t = vec![0.0f32; frames];
        for (i, f) in f0t.iter_mut().enumerate() {
            let t = i * hop;
            *f = if t < lo_isl.len() {
                200.0
            } else if t < a2 {
                0.0
            } else {
                400.0
            };
        }
        let arm = |inf| {
            psola_shift_env(&x, sr, 12.0, 0.0, &f0t, hop, false, 0.0, 0.30, inf, 0.0, 0.0, 0.0, 0.0)
        };
        let (off, doff) = arm(Infrasonic::Off);
        let (per, dper) = arm(Infrasonic::PerPeriod);
        // 整缓冲的规则会拿最低的那个基频 ⇒ 1000/200 = 5 ms
        let (fix, _) = arm(Infrasonic::FixedMs(5.0));
        assert!(doff.islands >= 2, "夹具没造出两个岛(islands = {})", doff.islands);
        assert!(dper.infrasonic_ma_ms > 0.0, "臂开着却没报出宽度");

        // 「削得多干净」= 岛内**还剩多少【多出来的】低频**,不是低频总能量。
        // ⛔ 第一版量的是总能量,读出「两条臂差 +0.01 dB」——**尺子坏了**:
        //    这个夹具是单极性脉冲串,自带很大的直流,而差分式**按设计保留输入自己的低频**
        //    ⇒ 总能量被那份直流主导,两条臂当然一样。⇒ 必须减掉输入自己的那一份。
        // 尺子本身与刀无关:8 ms 两遍盒(`RULER_BOX_PASSES`),这条线的历史尺子。
        let low = |y: &[f32], a: usize, b: usize| -> f64 {
            let ly = infrasonic_baseline_ms(&y[a..b], sr, INFRASONIC_MA_MS);
            let lx = infrasonic_baseline_ms(&x[a..b], sr, INFRASONIC_MA_MS);
            ly.iter().zip(lx.iter()).map(|(u, v)| (u - v) * (u - v)).sum()
        };
        let (hi_a, hi_b) = (a2, x.len());
        let (lo_a, lo_b) = (0usize, lo_isl.len());
        let d = |num: f64, den: f64| 10.0 * (num.max(1e-300) / den.max(1e-300)).log10();

        // ⭐ 承重:高音岛里,逐岛臂必须比定宽 5 ms 明显更干净
        let gain_hi = d(low(&fix, hi_a, hi_b), low(&per, hi_a, hi_b));
        assert!(
            gain_hi > 3.0,
            "高音岛里逐岛臂只比定宽 5 ms 好 {gain_hi:+.2} dB —— 逐岛宽度没生效\
             (报出来的中位宽度 {} ms)",
            dper.infrasonic_ma_ms
        );
        // ⛔ 阴性对照:低音岛里两者应当几乎一样(那里两条臂本来就是同一个宽度)
        let gain_lo = d(low(&fix, lo_a, lo_b), low(&per, lo_a, lo_b));
        assert!(
            gain_lo.abs() < 1.5,
            "低音岛里两者差了 {gain_lo:+.2} dB —— 那说明上面那条「更干净」不是逐岛买来的,\
             而是整体多削了"
        );
        // ⛔ 而且这把刀在两个岛上都必须真的动过东西
        assert_ne!(per, off, "逐岛臂逐位没动");
    }

    /// S155 笔5 —— ⛔⛔⛔ **这把刀不许把【输入自己的基频】加回输出。**
    ///
    /// 这是用户 2026-08-19 听出来的那条缺陷的判据,而它本该在笔1 就存在。
    /// 差分式 `out -= LP(out) − LP(in)` 里的 `+LP(in)` 是**一份低通过的输入**,
    /// 而输入(donor)唱得比输出低 ⇒ 只要 LP 在输入的基频上没压干净,
    /// 输出里就多出**第二个音高**。用户的原话:「f0 附近偏下多了一道有点时长的共振峰伪影,
    /// 听起来甚至有一点**合唱感**」,而且他给出的强弱排序(无扩展 < 不开刀 < 固定 8 ms <
    /// 自适应)与仪器**完全一致**。
    ///
    /// ## ⛔ 为什么这个夹具必须有**两个**音高
    ///
    /// 宽度规则把低通的第一个零点放在**这段缓冲里最低**的基频上。若夹具只有一个音高,
    /// 那个音高就正好坐在零点上 ⇒ 漏为 0 ⇒ **判据恒真**。真实情况是:缓冲里有更低的音
    /// (它们设定了宽度),而**真正被救的那个音**的基频更高 ⇒ 落在**旁瓣**里。
    /// 两遍盒的第一个旁瓣只有 **−27 dB**,四遍是 **−53 dB** —— 这就是笔5 改的东西。
    /// ⇒ 夹具:前半段 200 Hz(定宽度),后半段落在第一个旁瓣峰上(≈1.43/W)。
    #[test]
    fn the_cut_never_puts_the_inputs_own_pitch_back_into_the_output() {
        let sr = 44_100;
        let hop = sr as usize / 200;
        let f_low = 200.0f64;
        // 第一个旁瓣峰在 f·W ≈ 1.43,而 W ≈ 1/f_low ⇒ f ≈ 1.43·f_low。
        let f_hi = 1.43 * f_low;
        let (a, _) = pulses(sr, 0.5, |_| f_low, |_| 1.0);
        let (b, _) = pulses(sr, 0.5, |_| f_hi, |_| 1.0);
        let mut x = a.clone();
        x.extend_from_slice(&b);
        let frames = x.len().div_ceil(hop);
        let half = frames / 2;
        let mut f0t = vec![f_low as f32; frames];
        for f in f0t.iter_mut().skip(half) {
            *f = f_hi as f32;
        }
        let (off, _) = psola_shift_env(
            &x, sr, 12.0, 0.0, &f0t, hop, false, 0.0, 0.30, Infrasonic::Off, 0.0, 0.0, 0.0, 0.0);
        let (on, don) = psola_shift_env(
            &x, sr, 12.0, 0.0, &f0t, hop, false, 0.0, 0.30, Infrasonic::PerPeriod, 0.0, 0.0, 0.0, 0.0);
        assert!(don.infrasonic_ma_ms > 0.0, "臂开着却没报出宽度");
        // 只看后半段(那里输入唱 f_hi,而输出被搬到 2·f_hi ⇒ f_hi 上本该什么都没有)
        let k = x.len() / 2;
        let leak_off = tone_mag(&off[k..], sr, f_hi);
        let leak_on = tone_mag(&on[k..], sr, f_hi);
        let d = 20.0 * (leak_on.max(1e-15) / leak_off.max(1e-15)).log10();

        // ⛔ 阈值不许是拍脑袋的数。参照 = **2 遍盒那条**(= shipped 时的实现)在同一份材料、
        //    同一个宽度上漏多少;判据 = 今天这把必须比它好 ≥18 dB。
        //    ⭐ 这样它随 `CUT_BOX_PASSES` 自校准,而且它测的正是「改这个常数买到了什么」。
        //    实测:2 遍 **+22.70 dB** · 4 遍 **+1.64** · 6 遍与 8 遍都在 +1.0 以下。
        let ms = f64::from(don.infrasonic_ma_ms);
        let two = {
            let lo_o = infrasonic_baseline_passes(&off, sr, ms, 2);
            let lo_i = infrasonic_baseline_passes(&x, sr, ms, 2);
            let y: Vec<f32> = off
                .iter()
                .enumerate()
                .map(|(i, o)| (f64::from(*o) - (lo_o[i] - lo_i[i])) as f32)
                .collect();
            20.0 * (tone_mag(&y[k..], sr, f_hi).max(1e-15) / leak_off.max(1e-15)).log10()
        };
        assert!(
            two - d >= 18.0,
            "这把刀在输入自己的基频 {f_hi:.0} Hz 上加了 {d:+.2} dB,而 2 遍盒那条参照加 {two:+.2} \
             —— 只好了 {:.1} dB。那是把 donor 的音高塞回输出(用户听成「合唱感」)。\
             宽度 {ms:.2} ms,盒 {CUT_BOX_PASSES} 遍",
            two - d
        );
        // ⛔ 参照本身必须**真的坏**,否则上面那条是拿两个都干净的东西相减。
        assert!(
            two > 10.0,
            "2 遍盒的参照只漏了 {two:+.2} dB —— 这个夹具没把缺陷造出来,判据是空的\
             (被救音的基频必须落在旁瓣上,见这条测试的文档)"
        );
        // ⛔ 阴性对照:这条臂**必须真的动了东西**,否则上面那条是「什么也没做」的恒真。
        assert_ne!(on, off, "刀开着却逐位没动 —— 上面那条判据是空的");
    }

    /// S155 笔3 —— ⛔⛔ **宽度只许由这一段缓冲里【真的有音频】的那些帧决定。**
    ///
    /// 这条判据是**对抗复核**送回来的,而它抓的是本场刚上线的东西:生产喂给每一遍救援的是
    /// **整首歌**的 f0 轨,而 `score2svc.rs` 会把不相交的 chunk 的**音频铺成零**却**从不掩 f0**。
    /// ⇒ 分位数落在**在这条缓冲里是数字静音**的音上,而那些音从来不是被救援的音。
    /// 实测:−14 那一遍生产选了 **6.03 ms**,而按真正被救的音应当是 **2.70 ms**,
    /// 于是 50-125 Hz 还站着 **+8.5…+9.6 dB**、125-200 Hz 只拿掉 **0.16 dB**。
    ///
    /// ⛔ **为什么必须是一条【结构】判据而不是一个读数**:`infrasonic_frac` 是固定 8 ms 的尺子,
    /// 99.4% 的能量在 20 Hz 以下 ⇒ 6.03 ms 与 8.00 ms 在它上面**读同一个数**。
    /// 唯一露馅的是 `infrasonic_ma_ms`,而它露的时候没有任何东西会红。
    ///
    /// ⚠ 现有的 `the_infrasonic_arm_costs_nothing_over_productions_output_band` 挡不住这个:
    /// 它用 `flat_f0`(一条恒定的轨)⇒ 对「在一条跨两个八度的轨上取分位数」**结构上失明**。
    #[test]
    fn the_cut_width_ignores_f0_frames_whose_audio_is_digital_silence() {
        let sr = 44_100;
        let hop = sr as usize / 200;   // 5 ms
        // 后半段是真音频(440 Hz),前半段是**数字静音**但 f0 轨仍然写着一个很低的音。
        let (voiced, _) = pulses(sr, 0.5, |_| 440.0, |_| 1.0);
        let mut x = vec![0.0f32; voiced.len()];
        x.extend_from_slice(&voiced);
        let frames = x.len().div_ceil(hop);
        let half = frames / 2;
        let mut f0t = vec![110.0f32; frames];        // 静音那半:一个低八度的假音
        for f in f0t.iter_mut().skip(half) {
            *f = 440.0;
        }
        let (_, d) = psola_shift_env(
            &x, sr, 9.0, 0.0, &f0t, hop, false, 0.0, 0.30, Infrasonic::PerPeriod, 0.0, 0.0, 0.0, 0.0);
        let want = 1000.0 / 440.0;
        assert!(
            (f64::from(d.infrasonic_ma_ms) - want).abs() < 0.15,
            "宽度 {} ms —— 它应当是**有音频那半**的一个周期({want:.2} ms)。\
             读到 {:.2} ms 说明分位数又落回了铺零区那条假 f0(110 Hz ⇒ 9.09 ms)",
            d.infrasonic_ma_ms,
            d.infrasonic_ma_ms
        );
        // ⛔ 阴性对照:把那半段的音频也填上,宽度**必须**跟着变宽 ——
        //    否则上面那条断言可能只是「这个实现从来不看 f0 轨」。
        let mut x2 = voiced.clone();
        x2.extend_from_slice(&voiced);
        let (_, d2) = psola_shift_env(
            &x2, sr, 9.0, 0.0, &f0t, hop, false, 0.0, 0.30, Infrasonic::PerPeriod, 0.0, 0.0, 0.0, 0.0);
        assert!(
            f64::from(d2.infrasonic_ma_ms) > f64::from(d.infrasonic_ma_ms) * 1.5,
            "把静音那半填上音频之后宽度没跟着变({} → {} ms)⇒ 上面那条判据是空的",
            d.infrasonic_ma_ms,
            d2.infrasonic_ma_ms
        );
    }

    /// S155/S156 —— 读窗旋钮。**S156 把它翻成了生产默认(1.0)**,所以这条判据钉的不再是
    /// 「默认关」,而是⑴ 显式 `0` 仍然渲得出旧臂、⑵ 两条臂**真的不同**、⑶ 打开时不被静默跳过。
    ///
    /// ⛔ 后半句才是贵的那条。这个旋钮有一条现成的静默失败路径:颗粒循环里有
    /// `if lw > max_period { continue; }`,而 `max_period` 是**按今天那个窄窗**定的
    /// (0.02 s)⇒ 宽窗臂会把颗粒**一颗颗跳过**,输出照样有声(岛外透传 + 剩下的颗粒),
    /// 而「干预没生效」会被读成「干预无效」。S148 的 `frac_transport` 写死成 `false`
    /// 就是同一族,那次差点把一条真实的修法判死。
    ///
    /// ⇒ 这里同时钉住三件:⑴ 0 = 逐位同旧;⑵ 打开之后输出**变了**;
    /// ⑶ 打开之后**源覆盖率变好**(那是窗变宽的结构性后果,而且今天这个读数本来就在
    ///    `src_uncovered_frac` 上 —— ratio > 2 时相邻读窗之间会留下永远没人读的源波形)。
    #[test]
    fn the_wide_read_window_bites_and_the_old_arm_is_still_reachable() {
        let sr = 44_100;
        let f0 = 220.0;
        let hop = sr as usize / 200;
        let (x, _) = pulses(sr, 1.0, |_| f0, |_| 1.0);
        let f0t = flat_f0(x.len(), hop, f0 as f32);
        // ratio > 2 ⇒ 今天的读窗窄于一个源周期 ⇒ 源上真的有没人读的段落
        for st in [7.0f64, 14.0] {
            let (a, da) = psola_shift_env(
                &x, sr, st, 0.0, &f0t, hop, false, 0.0, 0.30, Infrasonic::Off, 0.0, 0.0, 0.0, 0.0);
            let (b, db) = psola_shift_env(
                &x, sr, st, 0.0, &f0t, hop, false, 0.0, 0.30, Infrasonic::Off, 0.0, 0.0, 1.0, 0.0);
            assert_ne!(a, b, "{st} st: 宽窗臂与今天逐位相同 —— 颗粒多半被 max_period 静默跳过了");
            assert_eq!(da.islands, db.islands, "{st} st: 窗宽不该改变岛的数目");
            assert!(
                db.marks >= da.marks / 2,
                "{st} st: 宽窗臂的标记数塌了({} → {})—— 那是被跳过,不是被加宽",
                da.marks,
                db.marks
            );
            assert!(
                db.src_uncovered_frac <= da.src_uncovered_frac + 1e-6,
                "{st} st: 窗加宽了源覆盖率反而变差({:.4} → {:.4})",
                da.src_uncovered_frac,
                db.src_uncovered_frac
            );
            // ⛔⛔ ⑷ **电平**。S155 在这里钉的是 `(-9.0..=1.0)`,而 S156 发现那个区间宽到
            //   **分不出「除 wsum」与「不除」** —— 把实现从「除」改成「不除」(那是一次真实的
            //   语义改动,电平差 20log10(ratio) ≈ 6 dB),这条断言**照样绿**。⇒ 它对自己声称
            //   钉住的那件事接近一条空判据。S156 把它收紧到能分开为止。
            //
            //   今天的实现**不除 wsum**(见输出合成那一段的说明):离线台子上「除」与「不除」的
            //   逐带谱形状读数**一模一样**,只差一个 20log10(ratio) 的常数 ⇒ 那 −4.3…−5.8 dB 不是
            //   宽窗的代价,是**除 wsum** 的代价;而下游没有任何东西吸收得了它
            //   (`restore_envelope` 默认关 · `peak_normalize_to` 给 donor 传 base 的峰值 ·
            //    `match_levels` 五个调用点全 false)⇒ 那 5 dB 会全额落到被救的那几个音上。
            //   实测:这个合成夹具 **+0.220 / +0.266 dB**;真素材(s12,+12)**+0.49 dB**。
            //   ⛔ 变异验过(2026-08-20 真跑,不是估的):把 `out[i] = acc[i]` 改回
            //      `acc[i] / wsum[i]` ⇒ +7 st 读 **−3.28 dB** ⇒ 这条断言当场红。
            let (ra, rb) = (rms(&a), rms(&b));
            let drop = 20.0 * (rb / ra).log10();
            assert!(
                (-1.5..=1.5).contains(&drop),
                "{st} st: 宽窗臂的电平变了 {drop:+.2} dB —— 不除 wsum 时实测只有 +0.2…+0.5 dB。\
                 跑出这个区间说明 wsum 又被除了、或者颗粒被静默跳过了"
            );
            // ⭐ ⑸ **`cola_w_median` 在宽窗臂上必须仍然是「COLA 有没有破」的读数**。
            //   窗一放宽,原始重叠系数就变成 ≈ `win·ratio`(实测 1.5/2.0/2.5/3.0 四档全中),
            //   而 S156 让读数与干填料都按稳态窗和 `W̄` 归一 ⇒ 它回到 1。
            //   ⛔ 没有这条,`cola_w_median` / `cola_gap_frac` 在这条臂上会静默地读别的东西
            //   (S155 的注释当时就是这样写的:「别把它读成红」= 把一只眼睛关掉)。
            //   ⛔ 变异验过(2026-08-20 真跑):去掉 `wbar` 那个除法 ⇒ +7 st 读 **1.4982**,
            //      而 `win·ratio = 1.0 × 1.4983` ⇒ 解析式与实测对到 1e-4,判据当场红。
            assert!(
                (db.cola_w_median - 1.0).abs() < 0.02,
                "{st} st: 宽窗臂的 cola_w_median = {:.4} —— 它应当按 W̄ 归一后回到 1,\
                 读到 ≈win·ratio 说明归一没做",
                db.cola_w_median
            );
        }
        // ⭐ 阴性对照:ratio > 2 时今天的读窗**必然**漏掉源(见 `src_uncovered_frac`),
        //    所以上面那条覆盖率断言不是恒真的。
        let (_, d14) = psola_shift_env(
            &x, sr, 14.0, 0.0, &f0t, hop, false, 0.0, 0.30, Infrasonic::Off, 0.0, 0.0, 0.0, 0.0);
        assert!(
            d14.src_uncovered_frac > 0.01,
            "+14 st 上今天就没漏源({:.4})⇒ 上面那条覆盖率判据是空的",
            d14.src_uncovered_frac
        );
    }

    /// S156 —— ⛔⛔ **宽窗臂的 ratio-1.0 恒等**。这条判据在 S155 是**不存在**的,而它的缺席
    /// 不是疏忽,是结构性的:这条线上所有恒等闸都经由 `psola_shift_diag` / `_locked` / `_infra`
    /// 进来,而那三个包装器把最后三个参数**写死成 `0.0`** ⇒ `win_periods` 恒为 0
    /// ⇒ 宽窗臂在这条线上**最便宜、最不可能自证**的那道闸底下,一寸覆盖都没有。
    /// ⇒ 「把 `UTAI_PSOLA_WIN` 的默认翻开」这个改动,本来可以在 `psola.rs` 一条测试都不红的
    ///    情况下把恒等悄悄弄坏(S155 那一版会:它在 `win_periods > 0` 时无条件 `acc / raw`,
    ///    而 `raw` 只是**近似** 1.0 —— 取整余量会直接进输出)。
    ///
    /// 为什么 `win_periods = 1.0` 上它**结构上**成立:ratio 1.0 时 `u = j` 是整数
    /// ⇒ `tgt[j] ≡ src[j]` ⇒ 目标邻距 ≡ 源周期 ⇒ `1.0 * src_l` 与今天那两个 `min` 给出的
    /// **是同一个数**;而 `W̄ = (1.0 × 1.0).max(1.0) = 1.0`,除以 1.0 在 IEEE 下逐位精确
    /// ⇒ 干填料那条也一字不差。⇒ 这里用 `assert_eq!` 而不是 epsilon。
    ///
    /// ⛔ **双向**:下移臂上两个 `min` 取的是**源周期**,`win = 1.0` 给的是同一个数,
    ///    但 `win·ratio < 1` —— `W̄` 那个 `.max(1.0)` 就是为这一侧写的。
    ///    ⛔ 变异验过:把 `.max(1.0)` 去掉 ⇒ 下移那两档当场红(干填料被除小 ⇒ 岛外的透传被削)。
    ///
    /// ⛔ 阴性对照:`win = 0.5` 与 `win = 1.5` 在 ratio 1.0 上**必须不恒等**(前者窗比邻距窄 ⇒ 留缝,
    ///    后者宽 ⇒ 重叠)。没有它,上面那条可能只是「这个实现根本不看 `win_periods`」。
    #[test]
    fn the_wide_read_window_is_still_the_identity_at_ratio_one() {
        let sr = 44_100;
        let f0 = 220.0;
        let hop = sr as usize / 200;
        let (x, _) = pulses(sr, 1.0, |_| f0, |_| 1.0);
        let f0t = flat_f0(x.len(), hop, f0 as f32);
        let run = |st: f64, win: f64| {
            psola_shift_env(
                &x, sr, st, 0.0, &f0t, hop, false, 0.0, 0.30, Infrasonic::Off, 0.0, 0.0, win, 0.0)
            .0
        };
        assert_eq!(run(0.0, 1.0), x, "ratio 1.0 上 win=1.0 的宽窗臂不是恒等变换");
        // 下移侧:两个 `min` 取的是源周期,win=1.0 给的是同一个窗 ⇒ 必须与今天**逐位**相同。
        for st in [-5.0f64, -12.0] {
            assert_eq!(run(st, 1.0), run(st, 0.0), "{st} st: 下移时 win=1.0 应当与今天逐位相同");
        }
        // 阴性对照:这两档在 ratio 1.0 上**必须**破恒等,否则上面那条是空的。
        for win in [0.5f64, 1.5] {
            assert_ne!(run(0.0, win), x, "ratio 1.0 上 win={win} 竟然也恒等 ⇒ 上面那条判据是空的");
        }
    }

    /// S156 —— **xgrain**:颗粒内容在相邻两个源脉冲之间插值,把宽窗引来的
    /// **「donor 自己的音高」**(`0.5·f_out`)拿掉。
    ///
    /// ⛔ 这条判据的**夹具必须逐周期有变化**。用严格周期的脉冲串会让它**恒真而且恒绿**:
    /// 相邻两个源脉冲逐位相同 ⇒ 在它们之间插值与取其中任何一个**是同一件事**
    /// ⇒ xgrain 在那种夹具上是精确的空操作,而判据会读出「开与关一样干净」⇒ 全绿、零信息。
    /// ⇒ 这里给脉冲串加一条**逐脉冲的幅度起伏**(≈17.5 Hz 的 shimmer,边带落在 220±17 Hz,
    ///    离 `0.5·f_out = 220` 的判读点足够远,不会自己制造被测的那个东西)。
    ///
    /// 钉四件:⑴ `0.0` 逐位同旧;⑵ ratio 1.0 上**任何深度**都恒等(`fr ≡ 0` ⇒ 最近邻权重与
    /// 线性权重是同一个向量 `(1,0)`);⑶ 打开之后输出真的变了;
    /// ⑷ ⭐ **承重那条**:宽窗臂上 `0.5·f_out` 的泄漏必须被压下去,
    ///    **而且要断言参照本身真的漏**(否则是拿两个都干净的东西相减 —— S155 笔5 那条判据的形状)。
    #[test]
    fn the_grain_interpolation_bites_and_zero_is_still_the_nearest_pulse_arm() {
        let sr = 44_100;
        let f0 = 220.0;
        let hop = sr as usize / 200;
        // ⚠ 逐脉冲起伏是这条判据成立的前提,见 doc。
        let (x, _) = pulses(sr, 1.0, |_| f0, |k| 1.0 + 0.5 * ((k as f64) * 0.5).sin());
        let f0t = flat_f0(x.len(), hop, f0 as f32);
        let run = |st: f64, win: f64, xg: f64| {
            psola_shift_env(
                &x, sr, st, 0.0, &f0t, hop, false, 0.0, 0.30, Infrasonic::Off, 0.0, 0.0, win, xg,
            )
            .0
        };
        // ⑵ ratio 1.0:任何深度都必须是**逐位**恒等。
        for xg in [0.3f64, 1.0] {
            assert_eq!(run(0.0, 0.0, xg), x, "ratio 1.0 上 xgrain={xg} 不恒等");
            assert_eq!(run(0.0, 1.0, xg), x, "ratio 1.0 上 win=1.0 + xgrain={xg} 不恒等");
        }
        // ⑴/⑶
        let base = run(12.0, 1.0, 0.0);
        let xg = run(12.0, 1.0, 1.0);
        assert_eq!(run(12.0, 1.0, 0.0), base, "同参数两跑不一致");
        assert_ne!(base, xg, "+12 st: xgrain 打开之后输出逐位相同 ⇒ 它没生效");
        // ⛔⛔ **承重那条判据不在这里,而且这不是疏忽** —— 见下面 `..._is_exactly_a_no_op_...`。
        //
        // 我先写的是「宽窗臂上 `0.5·f_out` 的泄漏必须被压下去」,而**阴性对照当场判死了它**:
        // 拿一条**恒定增益、严格周期**的脉冲串(此时相邻源脉冲按构造无差别 ⇒ xgrain 必须是
        // 精确空操作、读数必须一模一样),它读出的却是与其他所有夹具**一样的 3.2 dB「改善」**。
        // ⇒ 那个 3.2 dB 量的不是 xgrain 的机理,是 `pulses` 把脉冲放在**不同亚样本相位**上
        //   ⇒ 混合相邻两颗 = 一次低通。⇒ 那条判据会把「低通」读成「拿掉了 donor 的音高」。
        // ⭐ 这就是 §7-1「合成周期信号系统性冤枉 PSOLA 类算法」的一个新变种:
        //   这次它不是冤枉,是**送了一份来路不明的好成绩**。
        //
        // ⇒ 收益那一面只有**真素材**读得出来,它记在这里、由探针复现(`s156_knives/subh156.py`,
        //   s12 = 位移 +12,`0.5·f_out` 带能量相对各自 400-4000 Hz):
        //     donor 输入自己(结构地板) −45.5 | 今天 −37.9 | 宽窗 xgrain 关 **−33.6** |
        //     宽窗 xgrain 开 **−38.7** | 今天 + xgrain 开 −42.6
        //   ⇒ 宽窗自己带来 **+4.3 dB**,xgrain 拿掉 **5.1 dB**。
        //   ⚠ 代价也一起记:xgrain 让 8-12k 相对 300-1k 的倾斜从 −1.07 变成 −1.44(≈0.4 dB 的高频损失),
        //     方向与「混合相邻脉冲 = 一次低通」一致。
        //   ⛔ 这两条都**没有**判据盯着(仓里没有真素材)⇒ 这就是这个旋钮**默认关**、
        //     并且它的取舍要交给耳朵的原因(S146 协议)。
    }

    /// S156 —— xgrain 的**结构判据**:相邻两个源脉冲**逐位相同**时,在它们之间插值必须是
    /// **精确的空操作**。
    ///
    /// ⭐ 它把这个旋钮的语义钉死到不留余地:xgrain 只许**混合相邻两颗源脉冲的内容**,
    /// 不许顺带挪读点、不许改窗、不许做任何别的平滑。任何「顺手多做一点」的实现都会在这里逐位露馅。
    /// ⛔ 这条判据是被上面那条**失败的**泄漏判据逼出来的:那条量到的 3.2 dB 其实来自
    /// `pulses` 把脉冲放在不同亚样本相位上(混合两颗 = 一次低通),而不是 xgrain 的机理。
    ///
    /// 夹具:`f0 = 44100/200 = 220.5 Hz` ⇒ 周期**恰好 200 个整样本** ⇒ `pulses` 的落点全是整数
    /// ⇒ 每个脉冲的采样波形逐位相同。⛔ 阳性对照用 220.0 Hz(周期 200.4545 样本,落点带小数)
    /// —— 那时 xgrain **必须**改变输出,否则这条判据只是「这个实现根本不看 xgrain」。
    #[test]
    fn the_grain_interpolation_is_exactly_a_no_op_when_the_neighbouring_pulses_are_identical() {
        let sr = 44_100;
        let hop = sr as usize / 200;
        let go = |f0: f64, st: f64, xg: f64| {
            let (x, _) = pulses(sr, 1.0, |_| f0, |_| 1.0);
            let f0t = flat_f0(x.len(), hop, f0 as f32);
            psola_shift_env(
                &x, sr, st, 0.0, &f0t, hop, false, 0.0, 0.30, Infrasonic::Off, 0.0, 0.0, 1.0, xg,
            )
            .0
        };
        // 周期 = 200 个整样本 ⇒ 相邻源脉冲逐位相同 ⇒ 混合它们必须逐位空操作。
        let exact = f64::from(sr) / 200.0;
        // 逐段的相对残差(dB)。⛔ 必须**分段**:信号两端那里,`src[lo]` 与 `src[hi]` 的读窗
        // 一个被信号边界截断、一个没有 ⇒ 两次读到的内容**本来就不同**,那不是 xgrain 多做了事。
        // 实测全段 −28…−29 dB 全部来自这两头(头 −21…−23 / 尾 −23…−25),而中段是数值零。
        let seg = |a: &[f32], b: &[f32], lo: usize, hi: usize| {
            let d: f64 = (lo..hi).map(|i| (f64::from(a[i]) - f64::from(b[i])).powi(2)).sum();
            let e: f64 = (lo..hi).map(|i| f64::from(a[i]).powi(2)).sum();
            10.0 * (d.max(1e-300) / e.max(1e-300)).log10()
        };
        for st in [7.0f64, 12.0, 14.0] {
            let (a0, a1) = (go(exact, st, 0.0), go(exact, st, 1.0));
            let (b0, b1) = (go(220.0, st, 0.0), go(220.0, st, 1.0));
            let n = a0.len();
            let (m0, m1) = (n / 4, 3 * n / 4);
            let same = seg(&a1, &a0, m0, m1);
            let diff = seg(&b1, &b0, m0, m1);
            // 相邻源脉冲逐位相同 ⇒ 混合它们必须是**数值零**(实测 −178.7 / −3029 / −3031 dB)。
            assert!(
                same < -100.0,
                "{st} st: 相邻源脉冲逐位相同,xgrain 却改了中段 {same:.1} dB                  ⇒ 它做了「混合相邻两颗」之外的事"
            );
            // 阳性对照:落点带小数 ⇒ 相邻脉冲的采样波形不同 ⇒ xgrain 必须真的动手(实测 −34…−35 dB)。
            assert!(
                diff > -60.0,
                "{st} st: 源脉冲落点带小数,xgrain 在中段却只改了 {diff:.1} dB                  ⇒ 上面那条判据是空的"
            );
            // ⭐ 两者必须差出量级来,否则「空操作」与「生效」是同一个读数(实测差 144 dB)。
            assert!(diff - same > 60.0, "{st} st: 空操作档与生效档分不开({same:.1} vs {diff:.1})");
        }
    }

    /// S156 —— ⛔⛔ **这条线上第一条跑【生产口径】的判据。**
    ///
    /// 缺口是结构性的,而且它当场自证过:把 `WIN_PERIODS_DEFAULT` / `XGRAIN_DEFAULT` 从 0 翻成 1
    /// (= 换掉每一个被救音的音频),`psola.rs` 里 **68 条测试一条都没红** —— 因为它们**全部**
    /// 显式传旋钮,没有一条读生产默认。⇒ 「改了默认」与「改了行为」在这份文件里是分开的两件事,
    /// 而那正是 S155 笔0 在探针上修掉的同一族缺陷(探针对旋钮硬编码回落值,照旧脚本跑出来的
    /// 「今天」其实是改动之前的臂)。
    ///
    /// ⇒ 这里从 [`PROBE_ARM_DEFAULTS`](= 生产默认,由 `vocal_range` 的
    /// `the_probe_defaults_are_the_production_defaults` 绑住)把参数读出来跑,钉三件:
    /// ⑴ ratio 1.0 在**全套生产默认**下仍然 `assert_eq!` 恒等;
    /// ⑵ +14 上**源覆盖率必须是 0**(教科书宽度把 `ratio > 2` 那段没人读的源补上了)——
    ///    ⛔ 这一条就是「默认真的翻了」的指纹:退回 `WIN = 0` 它当场读 ≈0.108;
    /// ⑶ 生产臂与旧臂**逐位不同**(否则默认没生效)。
    #[test]
    fn the_production_default_arm_is_actually_what_runs() {
        let g = |k: &str| {
            PROBE_ARM_DEFAULTS.iter().find(|(n, _)| *n == k).unwrap_or_else(|| panic!("{k}")).1
        };
        let hp = if g("UTAI_PSOLA_HP") != 0.0 {
            if g("UTAI_PSOLA_HP_MS") > 0.0 {
                Infrasonic::FixedMs(g("UTAI_PSOLA_HP_MS"))
            } else {
                Infrasonic::PerPeriod
            }
        } else {
            Infrasonic::Off
        };
        let sr = 44_100;
        let f0 = 220.0;
        let hop = sr as usize / 200;
        let (x, _) = pulses(sr, 1.0, |_| f0, |_| 1.0);
        let f0t = flat_f0(x.len(), hop, f0 as f32);
        let prod = |st: f64| {
            psola_shift_env(
                &x,
                sr,
                st,
                0.0,
                &f0t,
                hop,
                g("UTAI_PSOLA_FRAC") != 0.0,
                g("UTAI_PSOLA_WSOLA"),
                g("UTAI_PSOLA_LOCK"),
                hp,
                g("UTAI_PSOLA_ENVFIX"),
                g("UTAI_PSOLA_BRIDGE"),
                g("UTAI_PSOLA_WIN"),
                g("UTAI_PSOLA_XGRAIN"),
            )
        };
        // ⑴ 全套生产默认下的恒等 —— 这条线上最便宜、最不可能自证的那道闸。
        assert_eq!(prod(0.0).0, x, "ratio 1.0 在生产默认下不是恒等变换");
        // ⑵ 教科书宽度把 ratio > 2 时那段「永远没人读的源」补上了。
        let (y14, d14) = prod(14.0);
        assert!(
            d14.src_uncovered_frac < 1e-9,
            "+14 st 生产臂仍然漏源 {:.4} —— 宽读窗没生效(退回 WIN=0 这里读 ≈0.108)",
            d14.src_uncovered_frac
        );
        // ⑶ ⭐⭐ **每一个被翻成默认的旋钮,单独退回 0 都必须改变输出。**
        //   ⛔ 这一条是被变异测试逼出来的:第一版只有 ⑴⑵,而把 `PROBE_ARM_DEFAULTS` 里的
        //   `WIN` 退回 0 之后它**照样绿** —— 因为 xgrain 的第二个读点把源覆盖率灌满了
        //   (见颗粒循环里那段门限的说明)。⇒ 「默认翻了」必须逐个旋钮证,不能靠一条综合读数。
        let one_off = |win: f64, xg: f64| {
            psola_shift_env(
                &x,
                sr,
                14.0,
                0.0,
                &f0t,
                hop,
                g("UTAI_PSOLA_FRAC") != 0.0,
                g("UTAI_PSOLA_WSOLA"),
                g("UTAI_PSOLA_LOCK"),
                hp,
                g("UTAI_PSOLA_ENVFIX"),
                g("UTAI_PSOLA_BRIDGE"),
                win,
                xg,
            )
            .0
        };
        assert_ne!(y14, one_off(0.0, g("UTAI_PSOLA_XGRAIN")), "把 WIN 单独退回 0,输出没变");
        assert_ne!(y14, one_off(g("UTAI_PSOLA_WIN"), 0.0), "把 XGRAIN 单独退回 0,输出没变");
        // 旧臂(两个都退回 0)在 +14 上**必须**漏源,否则 ⑵ 是空的。
        let (_, dold) = psola_shift_env(
            &x, sr, 14.0, 0.0, &f0t, hop, false, 0.0, 0.30, hp, 0.0, 30.0, 0.0, 0.0,
        );
        assert!(
            dold.src_uncovered_frac > 0.01,
            "旧臂在 +14 上竟然不漏源({:.4})⇒ ⑵ 那条判据是空的",
            dold.src_uncovered_frac
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
        let (mut n_inject, mut n_quiet) = (0usize, 0usize);
        for st in [-12.0, -7.0, -1.0, 1.0, 7.0, 12.0, 14.0] {
            let (legacy, dl) = psola_shift_locked(&x, sr, st, 0.0, &f0t, hop, false, 0.0, 0.30);
            let (via_deep, dd) =
                psola_shift_infra(&x, sr, st, 0.0, &f0t, hop, false, 0.0, 0.30, Infrasonic::Off);
            assert_eq!(legacy, via_deep, "{st} st: the 9-arg entry must stay the legacy arm");
            assert_eq!(dl, dd, "{st} st: …diagnostics included");
            // ⛔ And the readout must EXIST on the arm that is off — otherwise "what does it look
            // like today" is unanswerable without shipping the change (S147's silent-halving).
            assert!(
                dl.infrasonic_frac.is_finite(),
                "{st} st: the readout must be computed unconditionally"
            );
            assert_eq!(dl.infrasonic_removed, 0.0, "{st} st: nothing removed while off");

            let (on, don) = psola_shift_infra(
                &x, sr, st, 0.0, &f0t, hop, false, 0.0, 0.30, Infrasonic::PerPeriod,
            );
            // S155 —— 「臂开着就一定做了事」**不能无条件断言**,而这不是把判据放软:
            //   下移**根本不注入**。真素材实测(探针,同一条 donor,同一段音频):
            //   +9 / +14 的次声份额 7.12% / 17.27%,而 −7 / −12 是 **0.00% / 0.01%**。
            //   ⇒ 在下移上要求「必须拿掉东西」= 要求这把刀去动一个不存在的缺陷。
            // ⇒ 判据改成与机理同形,而且**两侧都有牙**:
            //     注入了 ⇒ 必须拿掉;没注入 ⇒ 必须是**逐位空操作**。
            //   后半句才是贵的那条:差分式会把输入自己的低频加回输出,而低频**不被移调守恒**
            //   (下移时输出脉冲密度减半 ⇒ 加回来的比拿掉的多)。合成夹具上实测到过
            //   基线 0.175 → **0.335**,这条断言就是把那个失败模式钉死成红。
            // ⛔ 判据必须用**和护栏同一个**谓词,否则中间地带必然打架(第一版在这里写了
            //    一个 0.01 的魔法阈值,+1 st 的注入是 0.0076 ⇒ 护栏动了而判据说不该动)。
            //    护栏是:输出那条基线的**能量**大于输入那条 ⇒ 才出手。
            let ms = f64::from(don.infrasonic_ma_ms);
            let e = |y: &[f32]| -> f64 {
                infrasonic_baseline_ms(y, sr, ms).iter().map(|v| v * v).sum()
            };
            let (e_off, e_in) = (e(&legacy), e(&x));
            let inj = baseline_rms(&legacy, sr) / rms(&legacy) - baseline_rms(&x, sr) / rms(&x);
            if e_off >= e_in {
                n_inject += 1;
                assert_ne!(on, legacy, "{st} st: 注入了 {inj:.4} 却一个样本都没动");
                assert!(
                    don.infrasonic_removed > 0.0,
                    "{st} st: the arm reports it removed nothing — 'on' and 'did something' are \
                     different facts (removed {})",
                    don.infrasonic_removed
                );
            } else {
                n_quiet += 1;
                assert_eq!(
                    on, legacy,
                    "{st} st: 没有注入(基线差 {inj:+.4})⇒ 这把刀必须**逐位**空操作"
                );
                assert_eq!(don.infrasonic_removed, 0.0, "{st} st: 空操作却报拿掉了东西");
            }
            // ⛔ 无论走哪个分支,**永不变差**:这把刀不许让基线比不开它的时候更大。
            //    这一条覆盖上面那个二分的**中间地带**(注入很小的位移),
            //    也是差分式在下移上唯一可能出事的方向。
            let b_off = baseline_rms(&legacy, sr) / rms(&legacy);
            let b_on = baseline_rms(&on, sr) / rms(&on);
            assert!(
                b_on <= b_off + 1e-6,
                "{st} st: 开了这把刀之后基线**变大**了({b_off:.5} → {b_on:.5})"
            );
        }
        // 一条从没被执行过的分支就是一条空判据 —— 两侧都必须真的走到过。
        assert!(n_inject >= 2, "没有一个位移触发注入分支 ⇒ 上面那半条判据是空的");
        assert!(n_quiet >= 2, "没有一个位移触发空操作分支 ⇒ 护栏那半条判据是空的");
    }

    /// S154 — the envelope-restoration arm: opt-in, readout unconditional, and it must do something.
    ///
    /// ⚠ The material matters here. A constant-amplitude pulse train has no envelope to violate,
    /// so it would let a no-op arm pass; this fixture **ramps the amplitude** the way an attack
    /// does, which is where the violation actually lives.
    #[test]
    fn the_envelope_arm_is_opt_in_and_the_readout_exists_while_it_is_off() {
        let sr = 44_100;
        let f0 = 220.0;
        // A 40 ms attack ramp into a steady body — the shape the defect lives on.
        // `gain` is indexed by PULSE, not by time: at 220 Hz the first 9 pulses are ~40 ms.
        let (x, _) = pulses(sr, 1.0, |_| f0, |k| {
            if k < 9 {
                0.25 + 0.75 * (k as f64) / 9.0
            } else {
                1.0
            }
        });
        let hop = sr as usize / 200;
        let f0t = flat_f0(x.len(), hop, f0 as f32);
        for st in [-12.0, -7.0, 1.0, 7.0, 14.0] {
            let (off, doff) =
                psola_shift_infra(&x, sr, st, 0.0, &f0t, hop, false, 0.0, 0.30, Infrasonic::Off);
            let (via, dvia) =
                psola_shift_env(&x, sr, st, 0.0, &f0t, hop, false, 0.0, 0.30, Infrasonic::Off, 0.0, 0.0, 0.0, 0.0);
            assert_eq!(off, via, "{st} st: env_restore_ms = 0 must be the legacy arm");
            assert_eq!(doff, dvia, "{st} st: …diagnostics included");
            // ⛔ The readout has to exist while the arm is OFF, or "what does it look like today"
            // cannot be answered without shipping the change (S147's silent halving).
            assert!(
                doff.env_dev_p50_db.is_finite(),
                "{st} st: the readout must be computed unconditionally"
            );
            assert_eq!(doff.env_dev_after_db, 0.0, "{st} st: nothing restored while off");

            let (on, don) =
                psola_shift_env(&x, sr, st, 0.0, &f0t, hop, false, 0.0, 0.30, Infrasonic::Off, 5.0, 0.0, 0.0, 0.0);
            assert_ne!(on, off, "{st} st: the opt-in arm must actually change the audio");
            // ⛔ What this arm promises is **the slow part**: the step at an island start. It does
            // not promise to shrink a 5 ms median that is already down in the period-scale noise,
            // and it must not be allowed to pretend otherwise — so the contract is split in two:
            //   · where there is a real violation to fix (≥ 1 dB), it has to fall;
            //   · everywhere else it may not make things meaningfully worse.
            // ⚠ The second half is the one that caught a real defect: the first implementation
            // (raw per-sample gain, one pass) turned 0.37 dB into 1.05 dB at +1 st.
            if don.env_dev_p50_db >= 1.0 {
                assert!(
                    don.env_dev_after_db < don.env_dev_p50_db,
                    "{st} st: the arm did not reduce a real violation ({} -> {})",
                    don.env_dev_p50_db,
                    don.env_dev_after_db
                );
            }
            assert!(
                don.env_dev_after_db <= don.env_dev_p50_db + 0.05,
                "{st} st: the arm made the deviation WORSE ({} -> {})",
                don.env_dev_p50_db,
                don.env_dev_after_db
            );
            eprintln!(
                "  [envfix] {st:+5} st: env dev {:.3} -> {:.3} dB",
                don.env_dev_p50_db, don.env_dev_after_db
            );
        }
    }

    /// S154 — the unvoiced-bridge arm: opt-in, and when on it must actually shrink the un-shifted
    /// leak at the island edges (which is the whole point of it).
    ///
    /// ⚠ The fixture has to have an **unvoiced gap inside a note**, because that gap is the defect.
    /// A fully-voiced fixture would let a no-op pass.
    #[test]
    fn the_unvoiced_bridge_is_opt_in_and_closes_the_gap_it_exists_for() {
        let sr = 44_100;
        let hop = sr as usize / 200;
        let f0 = 220.0;
        let (x, _) = pulses(sr, 1.0, |_| f0, |_| 1.0);
        // Voiced, then a 40 ms unvoiced gap in the middle, then voiced again — one note with a
        // consonant in it, which is exactly the shape the fed f0 has on a rescued 「と」.
        let mut f0t = flat_f0(x.len(), hop, f0 as f32);
        let g0 = f0t.len() * 4 / 10;
        let g1 = (g0 + ((0.040 * f64::from(sr)) as usize / hop).max(1)).min(f0t.len());
        for v in f0t.iter_mut().take(g1).skip(g0) {
            *v = 0.0;
        }
        let islands_of = |t: &[f32]| {
            voiced_islands(t, hop, x.len(), (MIN_ISLAND_SECONDS * f64::from(sr)) as usize).len()
        };
        assert_eq!(islands_of(&f0t), 2, "fixture must actually have two islands");

        let (off, _) = psola_shift_infra(&x, sr, 7.0, 0.0, &f0t, hop, false, 0.0, 0.30, Infrasonic::Off);
        let (via, _) =
            psola_shift_env(&x, sr, 7.0, 0.0, &f0t, hop, false, 0.0, 0.30, Infrasonic::Off, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(off, via, "bridge_unvoiced_ms = 0 must be the legacy arm");

        // ⛔⛔ **The island count must survive ANY width.** This is the guard the goose regression
        // forced: without it a 30 ms dilation collapsed that score 458 islands → 143, i.e. it was
        // fusing neighbouring notes into one rescue. Covering a note's own onset is the job;
        // merging notes is not, and no knob setting may turn one into the other.
        for ms in [5.0, 25.0, 60.0, 200.0, 500.0] {
            let d = bridge_unvoiced(&f0t, hop, sr, ms);
            assert_eq!(islands_of(&d), 2, "{ms} ms dilation merged the islands");
        }
        // …and it still has to actually cover something: the frames just outside each run.
        let d = bridge_unvoiced(&f0t, hop, sr, 25.0);
        assert!(d[g0] > 0.0, "the frame right after the first run must be covered");
        assert!(d[g1 - 1] > 0.0, "…and the one right before the second run");
        assert!(d[(g0 + g1) / 2] == 0.0, "…while the middle of the gap stays a gap");
        // ⛔ Never invent pitch outside the note: leading / trailing runs stay zero.
        assert_eq!(d[0], f0t[0]);
        assert_eq!(d[d.len() - 1], f0t[f0t.len() - 1]);

        let (on, _) =
            psola_shift_env(&x, sr, 7.0, 0.0, &f0t, hop, false, 0.0, 0.30, Infrasonic::Off, 0.0, 80.0, 0.0, 0.0);
        assert_ne!(on, off, "the opt-in arm must actually change the audio");
        // ⚠ Not asserting `off == x` at the gap edge: the previous island's last grains reach a
        // little past its last mark, so the very edge is already synthesized even with the arm off.
        // What the arm promises is that the gap's **edge** now gets grains and its **middle** does
        // not — the second half is the never-merge guard, and it is the one worth pinning.
        let e0 = g0 * hop;
        let e1 = (e0 + 8 * hop).min(x.len());
        assert!(on[e0..e1] != off[e0..e1], "the gap edge must have been covered");
        // ⚠ NOT asserting the middle of the gap is bit-untouched: both islands are longer now, so
        // their outermost grains and the dry-fill ramp reach further in. The never-merge guarantee
        // is about the **island count** (asserted above over five widths), not about audio bytes
        // in the middle of a gap — claiming the latter would be a criterion the design never made.
    }

    /// S154 — ⛔ the cheapest non-self-certifying gate on this whole line, applied to the new arm.
    ///
    /// At ratio 1.0 the target pulses ARE the source marks, so the output equals the input and the
    /// two envelopes are identical ⇒ every corrective gain is exactly 1.0. If this ever fails, the
    /// restoration is reaching outside the covered span or the envelope is being computed on
    /// different buffers than it claims.
    #[test]
    fn ratio_one_stays_the_identity_with_the_envelope_arm_on() {
        let sr = 44_100;
        let hop = sr as usize / 200;
        for f0 in [110.0, 220.0, 440.0] {
            let (x, _) = pulses(sr, 0.5, |_| f0, |k| if k < 10 { 0.3 } else { 1.0 });
            let f0t = flat_f0(x.len(), hop, f0 as f32);
            for ms in [2.0, 5.0, 20.0] {
                let (y, _d) =
                    psola_shift_env(&x, sr, 0.0, 0.0, &f0t, hop, false, 0.0, 0.30, Infrasonic::Off, ms, 0.0, 0.0, 0.0);
                assert_eq!(y, x, "f0 {f0}, envfix {ms} ms: ratio 1.0 must be the identity");
            }
        }
    }

    /// S154 — the restoration may not reach **outside** the voiced islands.
    ///
    /// That boundary is exactly where the defect lives, so a fix that moved it would be trading one
    /// discontinuity for another. Unvoiced stretches pass through untouched in the legacy arm; they
    /// must still pass through untouched with the arm on.
    #[test]
    fn the_envelope_arm_never_touches_the_unvoiced_pass_through() {
        let sr = 44_100;
        let hop = sr as usize / 200;
        let f0 = 220.0;
        let (x, _) = pulses(sr, 1.2, |_| f0, |_| 1.0);
        // Voiced only in the middle third; the two ends are pass-through.
        let mut f0t = flat_f0(x.len(), hop, f0 as f32);
        let n = f0t.len();
        for v in f0t.iter_mut().take(n / 3) {
            *v = 0.0;
        }
        for v in f0t.iter_mut().skip(2 * n / 3) {
            *v = 0.0;
        }
        let (off, _) = psola_shift_infra(&x, sr, 7.0, 0.0, &f0t, hop, false, 0.0, 0.30, Infrasonic::Off);
        let (on, _) = psola_shift_env(&x, sr, 7.0, 0.0, &f0t, hop, false, 0.0, 0.30, Infrasonic::Off, 5.0, 0.0, 0.0, 0.0);
        // The first and last eighth are comfortably inside the unvoiced pass-through.
        let e = x.len() / 8;
        assert_eq!(off[..e], on[..e], "the arm reached into the leading pass-through");
        assert_eq!(off[x.len() - e..], on[x.len() - e..], "…into the trailing pass-through");
        assert_ne!(off, on, "…and it still has to have done something in the middle");
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
    fn the_cut_is_wide_enough_to_reach_the_audible_part_of_what_this_process_injects() {
        // ⛔⛔ THE criterion for the constant itself. Without it, widening 8 ms back to 20 ms
        // leaves this whole file green while the benefit halves — measured, and it is the exact
        // shape S147 shipped once ("silent halving").
        //
        // What it pins: the injection is **not** confined to the inaudible band. Whole-song
        // measurement against the donor's own pre-PSOLA arm (the only reference that is the same
        // performance minus this process) says 20-60 Hz runs **+13.2 / +14.5 / +14.7 dB** at
        // −9 / −12 / −14 st, and the cut has to reach it: 20 ms leaves +10.8/+12.2/+12.3,
        // 8 ms leaves **+0.6/+2.0/+2.2**.
        // ⚠ The synthetic fixture here is not the song — the numbers differ — so what is asserted
        // is the ORDERING and a floor, not the song's dB values.
        let sr = 44_100;
        let period = 200usize;
        let f0 = f64::from(sr) / period as f64;
        let hop = sr as usize / 200;
        let x = asym_pulses(sr, 1.0, period);
        let f0t = flat_f0(x.len(), hop, f0 as f32);
        // Energy in 20-60 Hz, sampled analytically so no filter of our own is involved.
        let low = |y: &[f32]| -> f64 {
            [25.0f64, 35.0, 45.0, 55.0].iter().map(|f| tone_mag(y, sr, *f).powi(2)).sum::<f64>()
        };
        let (out, _) = psola_shift_locked(&x, sr, 12.0, 0.0, &f0t, hop, false, 0.0, 0.30);
        let injected = low(&out);
        assert!(
            injected > low(&x) * 10.0,   // 实测 25×
            "the fixture must actually show the injection first ({:.3e} vs {:.3e})",
            injected,
            low(&x)
        );
        let after = |ms: f64| -> f64 {
            let lf = infrasonic_baseline_ms(&out, sr, ms);
            let y: Vec<f32> =
                out.iter().zip(&lf).map(|(o, l)| (f64::from(*o) - *l) as f32).collect();
            low(&y)
        };
        let cur = after(INFRASONIC_MA_MS);
        let wide = after(20.0);
        assert!(
            cur < injected * 0.10,
            "the shipped cut must take out ≥90 % of the 20-60 Hz injection, got {:.1} %",
            100.0 * cur / injected
        );
        assert!(
            wide > cur * 3.0,
            "…and a 20 ms cut must NOT (it reaches only the inaudible half): 20 ms leaves              {:.1} % vs the shipped {:.1} % — if these are close the constant is unpinned",
            100.0 * wide / injected,
            100.0 * cur / injected
        );
        // ⚠ and the other side: it must not be so narrow that it eats the voice. Checked at
        // **882 Hz** (the 2nd harmonic here), i.e. inside production's real output-f0 band of
        // 830-1480 Hz. ⛔ Not at 441 Hz: the triangular window's response is oscillatory and
        // still costs 0.07 dB there — that is the documented, analytic behaviour (the other gate
        // asserts it exactly), not a defect, and pinning zero at 441 would force the cut back to
        // a width that only reaches the inaudible half.
        let lf = infrasonic_baseline_ms(&out, sr, INFRASONIC_MA_MS);
        let y: Vec<f32> = out.iter().zip(&lf).map(|(o, l)| (f64::from(*o) - *l) as f32).collect();
        let fp = f0 * 2f64.powf(12.0 / 12.0) * 2.0;
        let d =
            20.0 * (tone_mag(&y, sr, fp).max(1e-12) / tone_mag(&out, sr, fp).max(1e-12)).log10();
        assert!(d.abs() < 0.03, "{fp:.0} Hz (production's band) moved {d:+.3} dB");
    }

    #[test]
    fn the_infrasonic_arm_removes_the_baseline_without_touching_the_fundamental() {
        // The honest gate for the arm being ON: a bound on **where it is allowed to act** —
        // below the fundamental and nowhere else.
        // ⚠ S155 rewrote the first assertion. It used to read "the baseline should be mostly
        // gone", which silently assumed the input had none of its own. The differential form's
        // real contract is **"bring the output's baseline down to the input's, and never below
        // it"** — strictly stronger, because it also fails when the arm OVER-removes, and
        // over-removing is the same action that costs ratio 1.0 its bit-exact identity.
        let sr = 44_100;
        let f0 = 220.0;
        let hop = sr as usize / 200;
        let (x, _) = pulses(sr, 1.0, |_| f0, |_| 1.0);
        let f0t = flat_f0(x.len(), hop, f0 as f32);
        for st in [-12.0f64, -7.0, 7.0, 12.0, 14.0] {
            let (off, doff) = psola_shift_locked(&x, sr, st, 0.0, &f0t, hop, false, 0.0, 0.30);
            let (on, don) = psola_shift_infra(&x, sr, st, 0.0, &f0t, hop, false, 0.0, 0.30, Infrasonic::PerPeriod);

            let b_off = baseline_rms(&off, sr) / rms(&off);
            let b_on = baseline_rms(&on, sr) / rms(&on);
            let b_in = baseline_rms(&x, sr) / rms(&x);
            if b_off > b_in + 0.01 {
                assert!(
                    b_on < b_in + (b_off - b_in) * 0.25,
                    "{st} st: 注入的基线没被拿掉(输入 {b_in:.5} · 关 {b_off:.5} → 开 {b_on:.5})"
                );
            }
            // 另一侧,而且这一侧是差分式**特有**的:不许把基线压到**输入自己**之下。
            // 非差分式会(它减的是输出自己的低频)—— 而那正是「ratio 1.0 不再是恒等变换」
            // 的同一个动作,这条线上最便宜的那道判据就是被它顶掉的。
            assert!(
                b_on >= b_in * 0.5 || b_in < 1e-4,
                "{st} st: 基线被压到输入自己之下(输入 {b_in:.5} → 开 {b_on:.5})—— \
                 那说明它在减输出自己的低频,不是减这道工序多出来的那部分"
            );
            assert!(
                don.infrasonic_frac <= doff.infrasonic_frac + 1e-6,
                "{st} st: the BEFORE readout must not depend on the arm"
            );

            // ⛔⛔ THE bound, and it is **not** "the fundamental does not move". An earlier
            // version asserted exactly that, and it became a lie the moment the cut went from
            // 20 ms to 8 ms: at an output f0 of 110 Hz (a −12 st arm on this 220 Hz fixture) the
            // fundamental really does drop 0.18 dB. Asserting zero there would have pinned the
            // constant to a value that only fixes the half nobody can hear.
            // ⇒ what is asserted instead: the arm **is exactly the filter it documents** (two box
            // passes = a triangular window, analytic), plus a hard zero-cost bound over the band
            // production actually lives in.
            let out_f0 = f0 * 2f64.powf(st / 12.0);
            // S155 —— 宽度现在是**自适应**的 ⇒ 这里必须读诊断里报出来的那个值。
            // 顺带这就成了 `infrasonic_ma_ms` 自己的判据:报了个假数,下面的解析对拍就红。
            let used_ms = f64::from(don.infrasonic_ma_ms);
            assert!(used_ms > 0.0, "{st} st: 臂开着却没报出宽度");
            // ⛔ 只有**真的出手了**才去对拍解析响应。护栏(`e_out >= e_in`)在没有注入的位移上
            //    会让这把刀整个跳过 ⇒ 那时输出必须**逐位**同关掉时,而不是「符合某条滤波器曲线」。
            //    第一版没分这两种情况,于是在 −7 st 上拿一条**根本没运行**的滤波器的预测去比 0.000。
            if on == off {
                assert_eq!(don.infrasonic_removed, 0.0, "{st} st: 空操作却报拿掉了东西");
                continue;
            }
            let half = (((f64::from(sr) * used_ms / 1000.0) as usize) / 2).max(1);
            let l = (2 * half + 1) as f64;
            for h in [1.0f64, 2.0, 3.0] {
                let f = out_f0 * h;
                if f >= f64::from(sr) / 2.0 {
                    continue;
                }
                let (a, b) = (tone_mag(&off, sr, f), tone_mag(&on, sr, f));
                let d = 20.0 * (b.max(1e-12) / a.max(1e-12)).log10();
                // Dirichlet kernel of one box, raised to the number of passes the CUT uses;
                // high-pass = 1 - D^K. ⚠ S155 笔5 把 K 从 2 改成 4(见
                // `infrasonic_baseline_passes`),而这一行是唯一把「这条臂到底是不是它文档里
                // 那条滤波器」钉住的地方 —— 它当场红了,拦得对。
                let ph = std::f64::consts::PI * f / f64::from(sr);
                let dk = if ph.abs() < 1e-12 { 1.0 } else { (ph * l).sin() / (l * ph.sin()) };
                let want =
                    20.0 * (1.0 - dk.powi(CUT_BOX_PASSES as i32)).abs().max(1e-12).log10();
                assert!(
                    (d - want).abs() < 0.10,
                    "{st} st: {f:.0} Hz moved {d:+.3} dB but the {CUT_BOX_PASSES}-pass box \
                     predicts {want:+.3} — then this arm is not the filter it documents"
                );
                // …and over the band production actually uses, the cost must be nil.
                // ⚠ 800 Hz, not 250: the triangular window's response is oscillatory, and at
                // 330 Hz it still costs 0.10 dB (matching the analytic value above — the filter
                // is fine, the threshold was wrong). Production's output f0 is 830-1480 Hz.
                if f >= 800.0 {
                    // ⚠ S155 —— 这个上界从 0.03 放到 **0.12**,而理由不是「让它变绿」:
                    //   宽度现在是自适应的,这个夹具 f0 = 220 Hz ⇒ 宽 4.5 ms ⇒ 989 Hz 落在
                    //   三角窗响应的第 4-5 个旁瓣上,**解析值本来就是 −0.04 dB** —— 上面那条
                    //   ±0.10 的解析对拍已经把单点量级钉死了,这里管的只是上界。
                    //   依据:S148 的可闻性标定是「~2.7 dB 听得出 / ≤0.46 dB 听不出」
                    //   ⇒ 0.12 dB 仍有近 4 倍余量。
                    // ⛔ 单点上界不是这里该看的东西 —— 真正该看的是**生产口径**下整条输出
                    //   基频带的平均代价,那条判据单独写在下面。
                    assert!(
                        d.abs() < 0.12,
                        "{st} st: {f:.0} Hz moved {d:+.3} dB — 超过了这把刀允许的上界"
                    );
                }
            }
        }
    }

    /// S155 —— **生产口径**下这把刀的代价:整条输出基频带的平均,而不是某一个点。
    ///
    /// ⛔ 为什么单开一条:上一条用的夹具是 f0 = 220 Hz,而生产里 donor 的基频是 **350-880 Hz**
    /// (探针实测 p05/p50/p95 = 370/559/877 · 234/381/740 · 349/440/659)。自适应宽度
    /// = 一个源周期 ⇒ 生产是 2.7-4.3 ms,夹具是 4.5 ms —— **不是同一条滤波器**,拿夹具上的
    /// 单点读数去论证生产的代价是一次口径偷换。
    ///
    /// 判据:输出基频带 830-1480 Hz 上逐 50 Hz 取点,**平均** |Δ| < 0.05 dB、**最大** < 0.15 dB。
    /// (探针整曲实测,差分式 3 ms:三条臂的带内代价 0.006 / 0.040 / 0.001 dB。)
    #[test]
    fn the_infrasonic_arm_costs_nothing_over_productions_output_band() {
        let sr = 44_100;
        let f0 = 400.0; // donor 的量级,不是 220
        let hop = sr as usize / 200;
        let (x, _) = pulses(sr, 1.0, |_| f0, |_| 1.0);
        let f0t = flat_f0(x.len(), hop, f0 as f32);
        for st in [9.0f64, 12.0, 14.0] {
            let (off, _) = psola_shift_locked(&x, sr, st, 0.0, &f0t, hop, false, 0.0, 0.30);
            let (on, don) = psola_shift_infra(
                &x, sr, st, 0.0, &f0t, hop, false, 0.0, 0.30, Infrasonic::PerPeriod,
            );
            assert!(
                (f64::from(don.infrasonic_ma_ms) - 1000.0 / f0).abs() < 0.05,
                "{st} st: 宽度应当是一个源周期 {:.2} ms,报的是 {} ms",
                1000.0 / f0,
                don.infrasonic_ma_ms
            );
            // The frequencies have to be the output's OWN harmonics. The first version of this
            // swept 830..1480 in 50 Hz steps on a 400 Hz pulse train, i.e. it read `tone_mag` at
            // frequencies where the fixture has no energy at all — the ratio of two leakage
            // floors, which is not a cost. Same family as every other ruler on this line that
            // kept reading past the point where it still meant something.
            let out_f0 = f0 * 2f64.powf(st / 12.0);
            let (mut acc, mut worst, mut k) = (0.0f64, 0.0f64, 0usize);
            for h in 1..=24u32 {
                let f = out_f0 * f64::from(h);
                if f < 800.0 || f > f64::from(sr) * 0.4 {
                    continue;
                }
                let (a, b) = (tone_mag(&off, sr, f), tone_mag(&on, sr, f));
                let d = 20.0 * (b.max(1e-12) / a.max(1e-12)).log10();
                acc += d.abs();
                worst = worst.max(d.abs());
                k += 1;
            }
            assert!(k >= 8, "{st} st: 只找到 {k} 个谐波点 —— 判据太薄");
            let mean = acc / k as f64;
            assert!(
                mean < 0.05 && worst < 0.15,
                "{st} st: 生产带内代价 平均 {mean:.4} dB / 最大 {worst:.4} dB(n={k})"
            );
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
            add_bell(
                &x, &mut acc, &mut wsum, s_pos, t_pos, 12.0, 12.0, 1.0, frac_transport, 1.0,
                &mut res,
            );

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
            let mut src = analysis_marks(&dc_free, spec.sample_rate, &f0, hop, a, b, None);
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

    /// 一个臂旋钮的取值:env 里有就用 env,没有就用 [`PROBE_ARM_DEFAULTS`](= **生产默认**)。
    ///
    /// ⛔ 解析不了就 **panic**,不许静默回落到默认值。这条探针存在的全部意义是「输出属于哪个
    /// 口径」可查,而 `UTAI_PSOLA_BRIDGE=30ms`(带单位)这种手滑在旧写法下会**读成 0**、
    /// 打印成 0、然后被当成一条正常的臂记进档案里 —— 那正是「跑不起来被读成通过」。
    fn probe_arm(key: &str) -> f64 {
        let (_, want) = PROBE_ARM_DEFAULTS
            .iter()
            .find(|(k, _)| *k == key)
            .unwrap_or_else(|| panic!("{key} 不在 PROBE_ARM_DEFAULTS 里 —— 加旋钮要连表一起加"));
        match std::env::var(key) {
            Err(_) => *want,
            Ok(v) => match v.trim() {
                "true" | "on" | "yes" => 1.0,
                "false" | "off" | "no" => 0.0,
                s => s.parse::<f64>().unwrap_or_else(|_| {
                    panic!("{key}={v:?} 解析不了 —— 探针不许把它当成默认值悄悄跑过去")
                }),
            },
        }
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

        // S148:`UTAI_PSOLA_WSOLA=<frac>` 打开源侧波形相似度搜索。
        let wsola: f64 = probe_arm("UTAI_PSOLA_WSOLA");
        // S148 —— `frac_transport` 以前在这里写死成 false,于是 `UTAI_PSOLA_FRAC` 对这条探针**完全无效**:
        // 开与不开的输出 sha256 逐位相同。我差点把那读成「亚样本搬运对包络起伏没用」。
        // ⛔「臂开着」与「臂做了事」是两件事 —— 现在它由 env 控,并且把实际取值打出来。
        let frac = probe_arm("UTAI_PSOLA_FRAC") != 0.0;
        // S150 —— `UTAI_PSOLA_LOCK=<periods>`。
        let lock: f64 = probe_arm("UTAI_PSOLA_LOCK");
        // S154 —— `UTAI_PSOLA_ENVFIX=<ms>` 打开振幅包络还原。
        let envfix: f64 = probe_arm("UTAI_PSOLA_ENVFIX");
        // S154 —— `UTAI_PSOLA_BRIDGE=<ms>` 把音内短的清音空档桥接起来。
        // ⚠ 这里的 fallback 是 [`PROBE_ARM_DEFAULTS`](= 生产默认),**不是 0** —— 见那份文档。
        let bridge: f64 = probe_arm("UTAI_PSOLA_BRIDGE");
        // S155 —— `UTAI_PSOLA_WIN=<periods>` 读窗半宽(源周期);0 = 今天。
        let win: f64 = probe_arm("UTAI_PSOLA_WIN");
        // S156 —— `UTAI_PSOLA_XGRAIN=<0..1>` 颗粒内容在相邻两个源脉冲之间的插值深度;0 = 今天。
        let xgrain: f64 = probe_arm("UTAI_PSOLA_XGRAIN");
        // S155 —— `UTAI_PSOLA_HP=0/1` 去次声;`UTAI_PSOLA_HP_MS=<ms>` 强制固定宽度(0 = 自适应)。
        // ⛔⛔ 它以前在这里**写死成 `false`**,于是这条探针上「开」与「关」的输出**逐位相同** ——
        //    和 S148 那次 `frac_transport` 写死成 false 一模一样的形状,而那次差点被读成
        //    「亚样本搬运对包络起伏没用」。⇒ 现在由 env 控,并且实际取值打出来。
        let hp_ms = probe_arm("UTAI_PSOLA_HP_MS");
        let hp = match (probe_arm("UTAI_PSOLA_HP") != 0.0, hp_ms) {
            (false, _) => Infrasonic::Off,
            (true, m) if m > 0.0 => Infrasonic::FixedMs(m),
            (true, _) => Infrasonic::PerPeriod,
        };
        let (y, d) = psola_shift_env(
            &x, spec.sample_rate, st, 0.0, &f0, hop, frac, wsola, lock, hp, envfix, bridge, win,
            xgrain,);
        println!(
            "  arms: frac_transport={frac} wsola={wsola} phase_lock={lock} envfix={envfix}              bridge={bridge} hp={hp:?} win={win} xgrain={xgrain}"
        );
        // ⛔ 「这条探针跑的是不是生产口径」必须**当场看得见**。S154 之后生产默认是
        //    `bridge=30 / lock=0.30`,而这条探针以前对这两个都默认 0 ⇒ 照旧脚本跑出来的
        //    「今天」其实是**改动之前的臂**,而没有任何一行输出会说破这件事。
        let drift: Vec<String> = PROBE_ARM_DEFAULTS
            .iter()
            .filter_map(|(k, want)| {
                let got = probe_arm(k);
                (got != *want).then(|| format!("{k}={got} (生产 {want})"))
            })
            .collect();
        if drift.is_empty() {
            println!("  口径 = 生产默认");
        } else {
            println!("  ⛔ 口径**不是**生产默认:{}", drift.join(" · "));
        }
        // ⛔ 「注入了多少」与「拿掉了多少」是两个数,只打第二个会让 0.00 有两种读法
        //    (本来就没有 / 刀没生效)。S152 那条规矩:读数无条件算,修法才由旋钮控。
        println!(
            "  次声份额 {:.2}%{}",
            d.infrasonic_frac * 100.0,
            if hp == Infrasonic::Off {
                String::new()
            } else {
                format!(
                    " — hp(宽 {:.2} ms)拿掉了 {:.2} 个百分点",
                    d.infrasonic_ma_ms,
                    d.infrasonic_removed * 100.0
                )
            }
        );
        println!(
            "  env dev p50 {:.3} dB{}",
            d.env_dev_p50_db,
            if envfix > 0.0 { format!(" -> {:.3} dB", d.env_dev_after_db) } else { String::new() }
        );
        // S157 —— ⛔ **`src_uncovered_frac` 以前不在这行里**,而它是 `vocal_range.rs` 的
        //   `LANDING_RATIO_TWO_ST = 12` **唯一引用的读数**,那条 doc 还写着
        //   "measured on the real mark train"。⇒ 一个承重常数的证据,在唯一一条跑真素材的
        //   探针上**读不出来** —— 那等于它的出处今天没有人复现得了。
        //   ⚠ 仓里另一条断言(`the_production_default_arm_is_actually_what_runs`)读的是
        //   **合成脉冲串**,与那张表换了两个变量,不能互相顶替。
        println!(
            "psola_probe: {} samples @{} Hz, {st:+} st, f0 frames {} hop {hop}\n  \
             islands {} marks {} cola_gap {:.1}% w_median {:.3} wsola {wsola} moved {} \
             lock {lock} moved {}\n  \
             src_uncovered {:.4}% (ratio {:.4})",
            x.len(), spec.sample_rate, f0.len(), d.islands, d.marks,
            d.cola_gap_frac * 100.0, d.cola_w_median, d.wsola_moved, d.marks_locked,
            d.src_uncovered_frac * 100.0,
            2f64.powf(st / 12.0)
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

        // S155 —— `UTAI_PSOLA_DUMP_F32=<prefix>` 倒出 `<prefix>_in.f32` 与 `<prefix>_out.f32`
        // (裸 little-endian f32,与 `UTAI_PSOLA_F0` 同一种格式)。
        //
        // ⛔⛔ 为什么需要它:下面那个 writer 是 **16 bit + 按峰值归一**。两件事都会伪造差:
        // ⑴ 归一让「臂 A vs 臂 B」各自拿到**不同的增益**(HP 只要把峰值动 0.008 dB,整条臂就
        //    带上一个全局增益差)—— S152 那条「emphasis 臂带 +0.582 dB ⇒『更舒服』被响度污染」
        //    就是这个形状;⑵ 16 bit 的量化底噪 ≈ −96 dBFS,而这一场要量的是**去次声之后**
        //    的残量,它可能就在那个量级附近。
        // ⇒ 凡是拿探针做「加了多少 / 还剩多少」的题,一律读这两个 f32,别读那个 wav。
        if let Ok(prefix) = std::env::var("UTAI_PSOLA_DUMP_F32") {
            for (suffix, buf) in [("_in.f32", &x), ("_out.f32", &y)] {
                let mut bytes = Vec::with_capacity(buf.len() * 4);
                for v in buf.iter() {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                let p = format!("{prefix}{suffix}");
                std::fs::write(&p, &bytes).unwrap_or_else(|e| panic!("write {p}: {e}"));
                println!("  f32 dump: {} 个样本 -> {p}", buf.len());
            }
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
