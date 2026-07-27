//! S60-2 音域扩展 — vocal-range records + the v1 three-tier shift decision.
//!
//! A model's tested range lives in its sidecar json as an `extra` field (survives the
//! `#[serde(flatten)]` round-trip untyped):
//!
//! ```json5
//! "vocal_range": { "speakers": { "0": {
//!     "usable":  [48, 84],   // MIDI, inclusive — f0 err <100¢ & voiced >50% (v1 criteria)
//!     "comfort": [52, 79],   // f0 err <50¢ & voiced >80%; user-adjustable within usable
//!     "comfort_auto": [52, 79],           // the detected value (Reset target)
//!     "semitones": { "48": [err_cents, voiced_ratio], ... },  // raw scan, for re-derive/UI
//!     "tested_at": "2026-07-12"
//! } } }
//! ```
//!
//! Tier decision (v1 session20, verbatim semantics):
//!   1. everything inside COMFORT  → shift 0 (byte-identical render);
//!   2. everything inside USABLE   → shift 0 (render as-is, SKIP the inverse — never pay a
//!      DSP pass that isn't needed);
//!   3. outside USABLE             → minimal INTEGER translation into comfort; a range wider
//!      than the comfort zone gets centered best-effort (v1's compression tier is deliberately
//!      not ported — real material rarely exceeds a 2-octave comfort span; logged when hit).

use crate::models::ModelConfig;

/// Minimum span (semitones) a comfort zone must offer to be USED as the shift target.
/// A degenerate zone (S60d: the '哑音→把上限往下拖' spiral committed comfort=[42,42])
/// otherwise centers EVERY part toward a single point — the observed -27/-33 st renders.
/// Mirrored by MIN_COMFORT_SPAN in src/lib/vocal/rangeTest.ts (UI slider constraint).
pub const MIN_COMFORT_SPAN: f32 = 5.0;
/// Absurdity brake on any tier decision: no material legitimately needs more than ±2
/// octaves of translation — past that a stale/garbage record is doing the deciding.
pub const MAX_RANGE_SHIFT: i64 = 24;
/// Real singing behaves worse at the measured ceiling than the sustained-vowel scale probe
/// (consonants, dynamics) — whenever a shift happens anyway, land the top this far BELOW
/// c_hi instead of hugging the boundary (S60d2: -9 st landed the song top exactly on 70
/// and the climax still muted).
pub const CEILING_MARGIN: f32 = 2.0;
/// Frames an out-of-usable violation must SUSTAIN to count as musical content (≈250 ms on the
/// 100 fps f0 grid). rmvpe reads breaths/sibilance an octave UP for a few frames at a time;
/// those phantom islands (a) defeated the all-inside-usable byte-identical short-circuit with a
/// single frame and (b) accumulated enough 3×-weighted mass that rescuing THEM dragged
/// whole-song shifts of -6/-9 st on a healthy record + in-range song (S62b field case,
/// lengv2.3 — the shift magnitude tracked the octave-doubled spikes, not the melody). A real
/// climax is seconds long and sails through. Shorter runs are DELETED from the analysis.
pub const MIN_VIOLATION_MS: f32 = 250.0;
/// Largest ORIGINAL-index step a violation run may take and still count as ONE sustained run,
/// in milliseconds. Bridges rmvpe's blips inside a held note; a real breath/consonant gap
/// (100-300 ms) must break the run. See phantom_kept_mask.
pub const GAP_TOL_MS: f32 = 30.0;

/// Both thresholds are expressed in TIME, not frames: the cover path judges on a 100 fps grid
/// and the score path on the DAW's 50 fps grid, so a frame-count constant would silently mean
/// 250 ms on one and 500 ms on the other.
fn frames_for(ms: f32, fps: f32) -> usize {
    ((ms * fps / 1000.0).round() as usize).max(1)
}
/// Weight of a frame inside PROVEN usable but above the margined comfort ceiling. Tier-2
/// semantics ("inside usable renders untouched") mean these must never trigger a shift on
/// their own — but when a shift happens anyway they still nudge the optimizer to land the
/// material under the margin. Mirror band below comfort weighs less (fry degrades softer).
const BOUNDARY_WEIGHT_TOP: f32 = 0.3;
const BOUNDARY_WEIGHT_BOTTOM: f32 = 0.1;

/// Lowest / highest MIDI note the range test scans (mirrors RANGE_MIDI_LO/HI in rangeTest.ts).
pub const DAMAGE_LO_MIDI: usize = 36;
pub const DAMAGE_SLOTS: usize = 61; // 36..=96
/// Damage is quantized into a u8 so SpeakerRange stays `Copy` (it is passed BY VALUE all the
/// way into sovits/rvc run_pipeline; a Vec here would cascade through every signature).
const DAMAGE_MAX: f32 = 3.0;

/// S85 dead-only slot flags (bit set = criterion PASSES at that slot).
/// f0-axes ONLY, deliberately: the timbre axes misjudged real models in BOTH directions
/// (chika_v2's fake-healthy top / 东雪莲's low_ratio-killed comfort), while err/voiced
/// tracked the user's ear on "can it sing here at all" — see memory S85 三轮.
const SLOT_SINGABLE: u8 = 1 << 0; // usable-grade: err < 100¢ && voiced > 0.5
const SLOT_LANDING: u8 = 1 << 1; // landing-grade: err ≤ 50¢ && voiced ≥ 0.9

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeakerRange {
    /// MIDI bounds, inclusive.
    pub usable: (f32, f32),
    pub comfort: (f32, f32),
    /// S81 (E): per-semitone damage derived from the record's RAW scan, MIDI 36..=96,
    /// quantized 0..DAMAGE_MAX. `None` = the record predates the scan (or carries none) — the
    /// decision then falls back to the pre-S81 four-step usable/comfort ladder verbatim.
    ///
    /// Why this exists: `usable` is a "can produce a pitch at all" verdict (err<100 cents,
    /// voiced>50%) yet BOTH paths were treating it as an untouchable ceiling, so a model whose
    /// record says [36,77] never moved material anywhere inside 41 semitones — including the
    /// top few that measurably sing badly. A continuous curve lets the optimizer prefer the
    /// genuinely good part of the range instead of "anything not literally rejected".
    pub damage: Option<[u8; DAMAGE_SLOTS]>,
    /// S85: per-slot f0-axes flags (SLOT_*), MIDI 36..=96, for the score path's dead-only
    /// plan. `None` = no raw scan — dead_only_plan then falls back to usable/comfort bounds.
    /// Fixed array keeps the struct `Copy` (same reasoning as `damage`).
    pub slot_flags: Option<[u8; DAMAGE_SLOTS]>,
}

impl SpeakerRange {
    /// Bounds-only record (no raw scan) — the pre-S81 shape.
    pub fn bounds(usable: (f32, f32), comfort: (f32, f32)) -> Self {
        Self { usable, comfort, damage: None, slot_flags: None }
    }

    /// S85 dead-only: can the model produce a pitch AT ALL at this (integer) MIDI slot?
    /// With a raw scan = the usable-grade f0 criterion per slot (窗外/未测 = false — "never
    /// tested" must never read as "fine"); bounds-only records fall back to the usable bounds.
    fn slot_singable(&self, midi: i64) -> bool {
        match &self.slot_flags {
            Some(f) => {
                let slot = midi - DAMAGE_LO_MIDI as i64;
                (0..DAMAGE_SLOTS as i64).contains(&slot)
                    && f[slot as usize] & SLOT_SINGABLE != 0
            }
            None => (self.usable.0..=self.usable.1).contains(&(midi as f32)),
        }
    }

    /// S85 dead-only: is this slot a trustworthy LANDING for a rescued dead note
    /// (comfort-grade f0: err ≤ 50¢ && voiced ≥ 0.9)? Bounds-only → comfort bounds.
    fn slot_landing_ok(&self, midi: i64) -> bool {
        match &self.slot_flags {
            Some(f) => {
                let slot = midi - DAMAGE_LO_MIDI as i64;
                (0..DAMAGE_SLOTS as i64).contains(&slot) && f[slot as usize] & SLOT_LANDING != 0
            }
            None => (self.comfort.0..=self.comfort.1).contains(&(midi as f32)),
        }
    }

    /// Damage at a (possibly fractional) MIDI pitch, linearly interpolated between slots.
    /// Outside the scanned window there is NO evidence, so it reads as fully damaged — the
    /// optimizer must never treat "never tested" as "fine".
    fn damage_at(&self, midi: f32) -> Option<f32> {
        let d = self.damage.as_ref()?;
        let lo = DAMAGE_LO_MIDI as f32;
        let hi = lo + (DAMAGE_SLOTS - 1) as f32;
        if !(lo..=hi).contains(&midi) {
            return Some(DAMAGE_MAX);
        }
        let x = midi - lo;
        let i = x.floor() as usize;
        let f = x - i as f32;
        let a = d[i] as f32 * (DAMAGE_MAX / 255.0);
        let b = d[(i + 1).min(DAMAGE_SLOTS - 1)] as f32 * (DAMAGE_MAX / 255.0);
        Some(a + (b - a) * f)
    }
}

/// What a shift COSTS, independent of how much material it rescues.
///
/// The pre-S81 penalty was `0.003/st` and nothing else, which is not a cost model — rescuing 4%
/// of the frames scored 3×0.04 = 0.12 against a 9-semitone penalty of 0.027, so the optimizer
/// happily recoloured 100% of the audio to save 4% of it. The missing term is that a non-zero
/// shift puts EVERY sample through the inverse transform, whatever the shift size:
///   - SHIFT_FIXED_COST — the toll for passing the whole render through resynthesis at all
///     (measured on real material: the shipped engine costs several dB of harmonic-to-noise
///     ratio on 100% of the signal the moment the shift is non-zero);
///   - SHIFT_PER_ST — the marginal formant displacement, which under a formant-preserving
///     inverse is exactly |s| semitones of mismatch against the pitch the listener hears.
///
/// The magnitudes are NOT free parameters: they are the centre of the feasible region left by
/// three optimizer tests whose expected answers the user adjudicated BY EAR (phantom bursts
/// must not move / the S60d2 climax must still reach -7 / low material must still reach +22).
/// Stated as an invariant: a whole-render recolour needs at least ~2% of the loudness-weighted
/// material to be genuinely damaged before it pays for itself.
const SHIFT_FIXED_COST: f32 = 0.06;
const SHIFT_PER_ST: f32 = 0.005;
fn shift_cost(s: i64) -> f32 {
    if s == 0 {
        0.0
    } else {
        SHIFT_FIXED_COST + SHIFT_PER_ST * s.abs() as f32
    }
}

/// Per-semitone damage from one raw scan entry `[err_cents, voiced_ratio]`.
///
/// PERCEPTUAL, not linear-in-the-metric: a scan's cents error wanders 1-37 cents across a
/// model's whole healthy range, and none of that is audible. Charging `err/100` for it would
/// make the optimizer chase measurement noise and recolor songs that were fine. So each term
/// stays at zero until it means something, then ramps to full damage at the point the old
/// binary criterion used to reject:
///   - pitch:   free below 25 cents, full damage at 100 (the old `usable` cutoff);
///   - voicing: free above 0.95, full damage at 0.50 (ditto).
/// A rejected semitone (err 9999) saturates on the pitch term alone.
///
/// S81 F1 closes the blind spot the f0 pair could never see: both of those are derived from f0,
/// and f0 is an explicit conditioning INPUT to net_g, so a semitone whose TIMBRE has collapsed
/// still reports perfect pitch and full voicing (akiko MIDI 80 measured -7.7 dB with 88.7% of
/// its energy below 1.5*f0 while the record stored err=2 cents / voiced=1.00). `timbre` carries
/// `(rms_db, low_ratio)` when the scan has them — a 2-tuple record from before S81 passes None
/// and scores exactly as it did.
///   - low_ratio: free below 0.55 (a healthy vowel spreads across harmonics), full damage at
///     0.95 (a bare fundamental). THE term that catches "sings the note, has no voice left".
///   - rms_db: free down to -6 dB relative to the scale's loudest note, full damage at -18.
///     Graded, never a gate — a model is simply quieter at the edges of its range.
///
/// ★S85b: the S83 quiet CAP (1.0) + escape valve are REVERTED — this function is byte-identical
/// to v0.11.0 again. They were added for a SCORE symptom (chika_v2's broken climax at shift 0),
/// but the score path no longer consults this optimizer at all (dead-only, memory S85), so the
/// pair only governed COVER/audition — where they flipped 东雪莲/鹅妈妈 from the ear-proven -6
/// into a catastrophic -24 whole-song recolour (user log A/B, 2026-07-27 23:24 vs 23:44).
/// Known cost of the revert: the loudness-tilted-scale mis-shift the cap also fixed (a quiet
/// low range reading as damage) is back for COVER — queued as part of cover dead-only 化.
fn damage_from_scan(err_cents: f64, voiced: f64, timbre: Option<(f64, f64)>) -> f32 {
    let pitch = (((err_cents - 25.0) / 75.0).clamp(0.0, 1.0) as f32) * DAMAGE_MAX;
    let voicing = (((0.95 - voiced) / 0.45).clamp(0.0, 1.0) as f32) * DAMAGE_MAX;
    let (thin, quiet) = match timbre {
        Some((rms_db, low_ratio)) => (
            (((low_ratio - 0.55) / 0.40).clamp(0.0, 1.0) as f32) * DAMAGE_MAX,
            (((-6.0 - rms_db) / 12.0).clamp(0.0, 1.0) as f32) * DAMAGE_MAX,
        ),
        None => (0.0, 0.0),
    };
    (pitch + voicing + thin + quiet).min(DAMAGE_MAX)
}

/// Parse the sidecar `vocal_range` record for one speaker id. EXACT id only — an untested
/// speaker of a multi-speaker model must read as "no record" (no shift; the resource manager
/// offers per-speaker 补做), NOT silently borrow speaker 0's range and transpose by the wrong
/// singer's zone (audit S60). Single-speaker models resolve to id 0 exactly anyway.
///
/// S60d read-side healing: a comfort narrower than MIN_COMFORT_SPAN falls back to
/// comfort_auto, then to usable — a poisoned sidecar stops causing disasters the moment
/// this ships, without a disk migration. usable itself below the minimum ⇒ no record.
pub fn speaker_range(config: &ModelConfig, speaker_id: u32) -> Option<SpeakerRange> {
    let rec = config.extra.get("vocal_range")?;
    let speakers = rec.get("speakers")?;
    let sp = speakers.get(speaker_id.to_string())?;
    let pair = |key: &str| -> Option<(f32, f32)> {
        let v = sp.get(key)?.as_array()?;
        let lo = v.first()?.as_f64()? as f32;
        let hi = v.get(1)?.as_f64()? as f32;
        (lo <= hi).then_some((lo, hi))
    };
    let usable = pair("usable")?;
    if usable.1 - usable.0 < MIN_COMFORT_SPAN {
        return None;
    }
    let comfort = [pair("comfort"), pair("comfort_auto"), Some(usable)]
        .into_iter()
        .flatten()
        .find(|c| c.1 - c.0 >= MIN_COMFORT_SPAN && c.0 >= usable.0 && c.1 <= usable.1)?;
    // S81 (E): fold the raw per-semitone scan into a damage curve. Absent/garbage scan ⇒ None
    // ⇒ every consumer falls back to the pre-S81 ladder (old records keep working untouched).
    // S85: the same pass derives per-slot f0-axes flags for the score dead-only plan.
    let mut flags = [0u8; DAMAGE_SLOTS]; // untested slot = no flags = dead & unlandable
    let damage = sp.get("semitones").and_then(|m| m.as_object()).and_then(|m| {
        let mut d = [255u8; DAMAGE_SLOTS]; // untested slot = fully damaged, never "fine"
        let mut seen = 0usize;
        for (k, v) in m {
            let Ok(midi) = k.parse::<i64>() else { continue };
            let slot = midi - DAMAGE_LO_MIDI as i64;
            if !(0..DAMAGE_SLOTS as i64).contains(&slot) {
                continue;
            }
            let Some(a) = v.as_array() else { continue };
            let (Some(err), Some(voiced)) = (
                a.first().and_then(|x| x.as_f64()),
                a.get(1).and_then(|x| x.as_f64()),
            ) else {
                continue;
            };
            // 4-tuple = S81 scan (err, voiced, rms_db, low_ratio); 2-tuple = pre-S81, scored
            // exactly as before. Partial/garbage tuples fall back to the f0-only pair.
            let timbre = match (a.get(2).and_then(|x| x.as_f64()), a.get(3).and_then(|x| x.as_f64())) {
                (Some(rms_db), Some(low_ratio)) => Some((rms_db, low_ratio)),
                _ => None,
            };
            d[slot as usize] = (damage_from_scan(err, voiced, timbre) / DAMAGE_MAX * 255.0)
                .round()
                .clamp(0.0, 255.0) as u8;
            flags[slot as usize] = (if err < 100.0 && voiced > 0.5 { SLOT_SINGABLE } else { 0 })
                | (if err <= 50.0 && voiced >= 0.9 { SLOT_LANDING } else { 0 });
            seen += 1;
        }
        (seen >= 2).then_some(d)
    });
    let slot_flags = damage.is_some().then_some(flags);
    Some(SpeakerRange { usable, comfort, damage, slot_flags })
}

/// Structural write-side gate for a full `vocal_range` record (`{ speakers: { id: {...} } }`).
/// Rejects shapes no honest tester or clamped UI could produce (unordered bounds, comfort
/// escaping usable). Deliberately does NOT enforce MIN_COMFORT_SPAN — a narrow auto-test
/// result is honest data worth persisting; the read side above decides applicability.
pub fn validate_range_record(record: &serde_json::Value) -> Result<(), String> {
    let speakers = record
        .get("speakers")
        .and_then(|s| s.as_object())
        .ok_or("RANGE_INVALID")?;
    for sp in speakers.values() {
        let pair = |key: &str| -> Option<(f64, f64)> {
            let v = sp.get(key)?.as_array()?;
            Some((v.first()?.as_f64()?, v.get(1)?.as_f64()?))
        };
        let (u_lo, u_hi) = pair("usable").ok_or("RANGE_INVALID")?;
        let (c_lo, c_hi) = pair("comfort").ok_or("RANGE_INVALID")?;
        if !(u_lo <= u_hi && c_lo <= c_hi && c_lo >= u_lo && c_hi <= u_hi) {
            return Err("RANGE_INVALID".to_string());
        }
    }
    Ok(())
}

// NOTE: "which speaker governs a blend" = the existing ①c `crate::inference::dominant_speaker`
// (max-weight entry, else speaker_id) — reused, NOT re-implemented here (NO-dup).


/// Per-frame RMS of `x` on a hop grid (frame i = samples [i·hop, (i+1)·hop)) — the loudness
/// track for the shift decision's energy weighting. Frames past the end read 0.
pub fn frame_rms(x: &[f32], hop: usize, frames: usize) -> Vec<f32> {
    let hop = hop.max(1);
    (0..frames)
        .map(|i| {
            let lo = (i * hop).min(x.len());
            let hi = ((i + 1) * hop).min(x.len());
            if hi <= lo {
                return 0.0;
            }
            let s: f32 = x[lo..hi].iter().map(|v| v * v).sum();
            (s / (hi - lo) as f32).sqrt()
        })
        .collect()
}

/// Median-of-5 over the voiced MIDI sequence (edge windows clamp) — kills the classic 1-2
/// frame rmvpe octave flips before any range judgement sees them.
fn median5(seq: &[f32]) -> Vec<f32> {
    let n = seq.len();
    (0..n)
        .map(|i| {
            let lo = i.saturating_sub(2);
            let hi = (i + 3).min(n);
            let mut w: Vec<f32> = seq[lo..hi].to_vec();
            w.sort_by(|a, b| a.total_cmp(b));
            w[w.len() / 2]
        })
        .collect()
}

/// Phantom-island mask: a violation run shorter than MIN_VIOLATION_MS is detector noise
/// (octave-read breaths, fry blips), not singing — it must neither defeat the byte-identical
/// short-circuit nor add rescue mass to the optimizer. In-usable frames are always kept;
/// sustained violations (a real climax) are always kept.
///
/// S81: `orig_idx[k]` is the ORIGINAL frame index of `seq[k]`. `seq` is the voiced-only
/// compaction, so two high phrases separated by a breath sit side by side in it — before this,
/// several 200 ms bursts with breaths between them fused into one long run and sailed through
/// the filter (the run test was measuring "voiced frames" where it meant "elapsed time").
/// A gap wider than `gap_tol` now breaks the run.
fn phantom_kept_mask(
    seq: &[f32],
    orig_idx: &[usize],
    u_lo: f32,
    u_hi: f32,
    min_run: usize,
    gap_tol: usize,
) -> Vec<bool> {
    let mut kept = vec![true; seq.len()];
    let mut i = 0;
    while i < seq.len() {
        let out = seq[i] < u_lo || seq[i] > u_hi;
        let mut j = i + 1;
        while j < seq.len()
            && ((seq[j] < u_lo || seq[j] > u_hi) == out)
            && orig_idx[j] <= orig_idx[j - 1] + gap_tol
        {
            j += 1;
        }
        if out && j - i < min_run {
            kept[i..j].fill(false);
        }
        i = j;
    }
    kept
}

/// Per-DAW-frame Hz track for the SCORE path, so 「自己唱」 can be judged by the same optimizer
/// the cover path uses instead of two bare min/max numbers (S81 A).
///
/// Length == Σ`frames` == the length of the Option-A f0 / loudness lanes, so the caller can hand
/// the loudness envelope straight in as the energy weight. With a DAW f0 curve the track is that
/// curve (unvoiced ⇒ 0); without one it is the note pitches held for their durations — which
/// incidentally makes a long note weigh more than a grace note, exactly as it should.
pub fn score_frame_hz(
    note_nums: &[i64],
    frames: &[i64],
    f0_cents: &[f32],
    f0_voiced: &[u8],
    transpose: i64,
) -> Vec<f32> {
    let midi_to_hz = |m: f32| 440.0 * 2f32.powf((m - 69.0) / 12.0);
    if !f0_cents.is_empty() {
        return f0_cents
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                if f0_voiced.get(i).copied().unwrap_or(0) == 0 {
                    0.0
                } else {
                    midi_to_hz(c / 100.0 + transpose as f32)
                }
            })
            .collect();
    }
    let mut out = Vec::new();
    for (i, &n) in note_nums.iter().enumerate() {
        let len = frames.get(i).copied().unwrap_or(0).max(0) as usize;
        let hz = if n > 0 { midi_to_hz(n as f32 + transpose as f32) } else { 0.0 };
        out.extend(std::iter::repeat(hz).take(len));
    }
    out
}

/// S85 dead-only plan for the SCORE path (三轮耳判定案 — memory S85).
///
/// The whole-piece shift is structurally wrong for the score path: it moves EVERY note off its
/// written pitch to rescue the few that are broken, and both the render (per-音素×落点 craters)
/// and the shift+inverse round trip (audible tax growing with |shift|) charge for ALL of it —
/// the user's ear verdict was "开了不如不开" at every tested depth beyond -1, while the
/// dead-only arms passed. So: rest-delimited phrases containing at least one DEAD note (the
/// model cannot produce a pitch there at all — f0-axes verdict, see slot_singable) get the
/// minimal |shift| that lands every dead note on a landing-grade slot; every other note in the
/// piece stays at its written pitch, bit-identical to extension-off. The sung OUTPUT pitch is
/// always the written pitch (donor renders shifted, the inverse undoes it — 乐谱内容是底线).
///
/// Direction: dead-above-usable phrases move down, dead-below move up; a phrase dead on both
/// sides (or dead strictly inside the range — direction ambiguous) is unfixable by translation
/// and is skipped, as is one with no landing within ±MAX_RANGE_SHIFT (caller logs). A dragged
/// healthy note must stay singable after the shift — rescuing の85 must not push う73 into a
/// crater.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeadGroup {
    /// Score triple indices, inclusive — the rest-delimited phrase.
    pub start: usize,
    pub end: usize,
    /// Semitones the phrase RENDERS at (negative = down); the inverse undoes it.
    pub shift: i64,
}

/// Returns `(plan, unfixable)` — `unfixable` carries the (start, end) triple range of every
/// dead-containing phrase with NO landing (dead on both sides, or nothing within
/// ±MAX_RANGE_SHIFT); the caller logs each LOUDLY so a skipped broken climax never reads as
/// "handled" (审查 S85: positions, not just a count — cover 侧富审计的对等物).
pub fn dead_only_plan(
    note_nums: &[i64],
    transpose: i64,
    range: &SpeakerRange,
) -> (Vec<DeadGroup>, Vec<(usize, usize)>) {
    let eff = |n: i64| (n + transpose).clamp(1, 127); // mirror transpose_note_pitch's clamp
    let mut out = Vec::new();
    let mut unfixable = Vec::new();
    let mut i = 0usize;
    while i < note_nums.len() {
        if note_nums[i] <= 0 {
            i += 1;
            continue;
        }
        let mut j = i;
        while j + 1 < note_nums.len() && note_nums[j + 1] > 0 {
            j += 1;
        }
        let sung: Vec<i64> = (i..=j).map(|k| eff(note_nums[k])).collect();
        let dead: Vec<i64> = sung.iter().copied().filter(|&p| !range.slot_singable(p)).collect();
        if !dead.is_empty() {
            let above = dead.iter().any(|&p| p as f32 > range.usable.1);
            let below = dead.iter().any(|&p| (p as f32) < range.usable.0);
            // Candidate order: single-sided dead searches its own direction by growing |s|;
            // INTERIOR dead (a bridged-weak slot inside usable — a legal record form the write
            // side produces on purpose, rangeTest.ts longestRun) has no inherent direction and
            // tries both, down first at each magnitude. Dead on BOTH sides is untranslatable.
            let candidates: Vec<i64> = if above && below {
                Vec::new()
            } else if above {
                (1..=MAX_RANGE_SHIFT).map(|m| -m).collect()
            } else if below {
                (1..=MAX_RANGE_SHIFT).collect()
            } else {
                (1..=MAX_RANGE_SHIFT).flat_map(|m| [-m, m]).collect()
            };
            let found = candidates.into_iter().find(|&s| {
                dead.iter().all(|&p| range.slot_landing_ok(p + s))
                    && sung.iter().all(|&p| range.slot_singable(p + s))
            });
            match found {
                Some(shift) => out.push(DeadGroup { start: i, end: j, shift }),
                None => unfixable.push((i, j)),
            }
        }
        i = j + 1;
    }
    (out, unfixable)
}

/// S85: dead-group 短语窗(50fps 帧域)——短语区间的帧窗向两侧休止扩展(pre ≤4 帧吃借帧
/// 辅音、post ≤2 帧吃释放,各以半个间隙为上限=与相邻唱段/拼接窗永不重叠)。返回
/// (shift, 起帧, 止帧);采样换算与交叉淡化在音频域(score2svc::apply_dead_only_windows)。
pub fn dead_group_windows(
    note_nums: &[i64],
    frames: &[i64],
    plan: &[DeadGroup],
) -> Vec<(i64, i64, i64)> {
    let mut cum = Vec::with_capacity(frames.len() + 1);
    let mut acc = 0i64;
    cum.push(0);
    for &f in frames {
        acc += f.max(0);
        cum.push(acc);
    }
    plan.iter()
        .map(|g| {
            let mut k = g.start;
            while k > 0 && note_nums[k - 1] <= 0 {
                k -= 1;
            }
            let gap_prev = cum[g.start] - cum[k];
            let mut k = g.end + 1;
            while k < note_nums.len() && note_nums[k] <= 0 {
                k += 1;
            }
            let gap_next = cum[k] - cum[g.end + 1];
            (g.shift, cum[g.start] - 4.min(gap_prev / 2), cum[g.end + 1] + 2.min(gap_next / 2))
        })
        .collect()
}

/// Whole-signal shift decision for the COVER/audition path (S60d2 — frame-mass optimizer;
/// S62b — spike hygiene + usable-aware weighting).
///
/// The previous p02/p98-bounds version had two blind spots the user could HEAR: the top 2%
/// of frames (= seconds of the climax on a full song) stayed above the ceiling by
/// construction, and the minimal translation parked the material top exactly ON c_hi with
/// zero headroom. Instead: brute-force the integer translation in ±MAX_RANGE_SHIFT that
/// minimizes the sung-outside-the-zone frame mass, where
///   - frames above USABLE weigh 3× frames below (top overflow = saturation/mute, the audible
///     disaster; bottom overflow = fry, degrades softer),
///   - frames still inside usable but past the margined comfort boundary weigh only
///     BOUNDARY_WEIGHT_* — the model provably sings them (tier-2), so they can never justify
///     recoloring the whole render by themselves, yet still steer WHERE a real shift lands
///     (S62b: counting proven 76-77 frames as full violations recolored in-range songs),
///   - the shift cost (SHIFT_FIXED_COST + SHIFT_PER_ST·|s|, S81-B) prices what a non-zero
///     shift really does — 100% of the audio passes through the inverse resynthesis — and
///     breaks near-ties toward the smallest coloration.
/// rmvpe spike hygiene BEFORE any judgement (S62b): median-5, then phantom violation islands
/// (< MIN_VIOLATION_RUN) are deleted — a handful of octave-doubled breath frames used to both
/// defeat the all-inside-usable short-circuit and out-mass the shift penalty (3×0.6% of a full
/// song beats 0.003×6), so the whole song chased the spikes down -6/-9 st.
///
/// LOUDNESS weighting (S62c): `energy` = optional per-frame RMS on the same grid (frame_rms).
/// A separated stem reads high for SECONDS on reverb tails / harmony bleed / breathy passages —
/// sustained enough to pass the run filter — but those frames are far below the lead's level.
/// Violation mass is therefore weighted by loudness (normalized to the piece's p95, floored),
/// so what drives a whole-render recolor is what a listener would actually HEAR out-of-range;
/// two different models both "muffled with extension ON" over in-range songs was this (§user).
/// A piece entirely inside USABLE (after hygiene) renders untouched (tiers 1/2, byte-identical).
pub fn piece_range_shift(
    f0_hz: &[f32],
    energy: Option<&[f32]>,
    range: &SpeakerRange,
    fps: f32,
) -> i64 {
    // S81 (H): a length-mismatched energy track must fail SAFE to "unweighted", not to
    // "every frame we couldn't measure gets FULL weight" — the latter hands silence the same
    // authority as the lead vocal. Unreachable today (all three call sites pass exact lengths).
    let energy = match energy {
        Some(e) if e.len() != f0_hz.len() => {
            tracing::warn!(
                "range-extend: energy track {} frames != f0 {} — decision falls back to unweighted",
                e.len(),
                f0_hz.len()
            );
            None
        }
        other => other,
    };
    let mut midis: Vec<f32> = Vec::with_capacity(f0_hz.len());
    let mut weights: Vec<f32> = Vec::with_capacity(f0_hz.len());
    // S81 (D): the ORIGINAL frame index of each kept sample — phantom_kept_mask needs it to tell
    // "one held note" from "several bursts separated by breaths" (the compaction hides gaps).
    let mut orig_idx: Vec<usize> = Vec::with_capacity(f0_hz.len());
    for (i, &v) in f0_hz.iter().enumerate() {
        if v <= 0.0 {
            continue;
        }
        midis.push(69.0 + 12.0 * (v / 440.0).log2());
        weights.push(energy.and_then(|e| e.get(i)).copied().unwrap_or(1.0));
        orig_idx.push(i);
    }
    if midis.len() < 10 {
        return 0; // too little voiced material to judge — render untouched
    }
    let (u_lo, u_hi) = range.usable;
    let (c_lo, c_hi) = range.comfort;
    if c_hi - c_lo <= 0.0 {
        return 0; // degenerate comfort never a target (speaker_range heals; defensive)
    }
    // normalize loudness to the piece's p95 (robust "lead level"), floor so nothing zeroes out
    if energy.is_some() {
        let mut sorted = weights.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let p95 = sorted[(sorted.len() as f32 * 0.95) as usize % sorted.len()].max(1e-6);
        for w in &mut weights {
            *w = (*w / p95).clamp(0.02, 1.0);
        }
    }
    let filtered = median5(&midis);
    let kept = phantom_kept_mask(
        &filtered,
        &orig_idx,
        u_lo,
        u_hi,
        frames_for(MIN_VIOLATION_MS, fps),
        frames_for(GAP_TOL_MS, fps) + 1, // a step of k skips k-1 frames
    );
    let dropped = kept.iter().filter(|&&k| !k).count();
    let voiced: Vec<(f32, f32)> = filtered
        .iter()
        .zip(&weights)
        .zip(&kept)
        .filter(|(_, &k)| k)
        .map(|((&m, &w), _)| (m, w))
        .collect();
    if voiced.len() < 10 {
        // armed-always-audits (S83): a near-silent piece is a verdict too.
        tracing::info!(
            "range-extend: too little voiced material ({} frames after hygiene) — rendering untouched",
            voiced.len()
        );
        return 0;
    }
    // Bounds-only records keep the pre-S81 short-circuit verbatim (byte-identical). A record
    // WITH a scan deliberately does not get it: "inside usable" is exactly the verdict S81
    // found untrustworthy, and the cost function below already refuses to move a clean piece
    // (a zero-damage piece cannot beat SHIFT_FIXED_COST).
    if range.damage.is_none() && voiced.iter().all(|&(m, _)| m >= u_lo && m <= u_hi) {
        // armed-always-audits (S83): this short-circuit used to print only when phantoms were
        // dropped — the common bounds-only in-range render was a silent verdict.
        tracing::info!(
            "range-extend: piece in-range for a bounds-only record (frames={}, phantom-dropped={dropped}) — rendering untouched",
            voiced.len()
        );
        return 0; // tiers 1/2 — the whole piece sits in the proven zone
    }
    // S81 (C): the margin exists because the sustained-vowel probe is more forgiving than real
    // singing, so it must be measured from the PROVEN top (u_hi). When the record already
    // reports c_hi < u_hi the comfort criterion has charged for that gap once — subtracting the
    // margin from c_hi as well double-counts and pushes the shift further than the evidence
    // warrants. (comfort == usable, the common case, is unchanged.)
    let top = c_hi.min(u_hi - CEILING_MARGIN);
    let n: f32 = voiced.iter().map(|&(_, w)| w).sum();
    let frame_mass = |sf: f32| -> f32 {
        let mut mass = 0f32;
        for &(m, w) in &voiced {
            let p = m + sf;
            if let Some(d) = range.damage_at(p) {
                // S81 (E): measured per-semitone evidence replaces the four-step LADDER…
                // …but NOT the user's comfort zone. Those are different kinds of statement:
                // `damage` is what the probe measured, `comfort` is what the user decided they
                // want ("above X this singer sounds bad to me" — a legitimate, guard-railed
                // action per S60d2). Letting the scan swallow the comfort term made every
                // manual comfort adjustment a no-op on any model that had a scan, i.e. all of
                // them — a regression introduced with the damage curve earlier this session.
                let wish = if p > top {
                    BOUNDARY_WEIGHT_TOP
                } else if p < c_lo {
                    BOUNDARY_WEIGHT_BOTTOM
                } else {
                    0.0
                };
                mass += (d + wish) * w;
                continue;
            }
            if p > u_hi {
                mass += 3.0 * w;
            } else if p > top {
                mass += BOUNDARY_WEIGHT_TOP * w;
            } else if p < u_lo {
                mass += w;
            } else if p < c_lo {
                mass += BOUNDARY_WEIGHT_BOTTOM * w;
            }
        }
        mass / n
    };
    // ★S85b: the S83 escape valve (dead-mass FIXED-toll waiver) is REVERTED — the loop below is
    // byte-identical to v0.11.0. Its motivating case (score climax broken at shift 0) is now
    // handled by the score path's dead-only mechanism, which never consults this optimizer;
    // here it only pushed COVER renders into ear-rejected deep recolours (东雪莲 -6 → -24).
    let mut best_cost = f32::MAX;
    let mut best_shift = 0i64;
    for s in -MAX_RANGE_SHIFT..=MAX_RANGE_SHIFT {
        let sf = s as f32;
        let cost = frame_mass(sf) + shift_cost(s);
        if cost < best_cost {
            best_cost = cost;
            best_shift = s;
        }
    }
    // One-line decision audit — ALWAYS, including "armed but chose 0" (S83 promise kept: a
    // shift-0 line makes "why didn't it move" one grep instead of a forensics session).
    let above0: f32 = voiced.iter().filter(|&&(m, _)| m > u_hi).map(|&(_, w)| w).sum::<f32>() / n * 100.0;
    let below0: f32 = voiced.iter().filter(|&&(m, _)| m < u_lo).map(|&(_, w)| w).sum::<f32>() / n * 100.0;
    tracing::info!(
        "range-extend optimizer: shift {best_shift:+} st (frames={}, phantom-dropped={dropped}, loudness-weighted at 0: {above0:.1}% above-usable / {below0:.1}% below; cost {:.4} -> {best_cost:.4})",
        voiced.len(),
        frame_mass(0.0)
    );
    best_shift
}

/// κ — how much of the inverse's pitch move the FORMANTS follow:
///   κ=0  formants stay where the model put them — the source timbre, dark/covered on big
///        down-shifts (engine-native `setFormantSemitones` with pitch compensation);
///   κ=1  formants move with the pitch (the plain spectral transpose) — bright/chipmunk.
/// Default 0 is the configuration the user A/B'd on real songs and accepted
/// ("干净了…共振腔相对来讲保持的甚至还挺好"); κ=1 is audibly a chipmunk. Since the 1.3.2
/// vendor upgrade the whole policy runs INSIDE Signalsmith — no external formant_warp pass,
/// no 918 Hz lifter ceiling, no overshoot guard — and κ=1 skips the formant machinery
/// entirely (zero extra cost).
pub const DEFAULT_FORMANT_KAPPA: f32 = 0.0;

/// Sticky ~100 ms formant-base schedule from a fed-f0 track (S82b/S82c streaming base): per
/// window the voiced (> 20 Hz) median, UNquantized — an earlier semitone quantization (meant
/// to merge windows into coarse runs) lost the user's 8-vs-9 A/B to the smooth track, whose
/// per-window values make the shim slice at every ~100 ms boundary (negligible cost — the
/// engine re-reads the base every internal block anyway) while the analysis width follows
/// the melody continuously instead of in semitone stairs. Unvoiced windows carry the
/// previous value (a mid-stream 0 would re-enable the engine's noise-chasing auto-detector —
/// the exact jitter this schedule exists to kill, S82 ear-confirmed); leading unvoiced
/// windows backfill from the first voiced one. All-unvoiced ⇒ None (auto-detect). Callers
/// pass the SHIFTED track — the inverse's input is the render at the shifted pitch.
/// Returns (track, step_samples).
pub fn formant_base_track(
    f0_hz: &[f32],
    hop_samples: usize,
    sample_rate: u32,
) -> Option<(Vec<f32>, usize)> {
    if f0_hz.is_empty() || hop_samples == 0 {
        return None;
    }
    let step_frames =
        ((sample_rate as f32 * 0.1 / hop_samples as f32).round() as usize).max(1);
    let mut track: Vec<f32> = Vec::with_capacity(f0_hz.len() / step_frames + 1);
    for win in f0_hz.chunks(step_frames) {
        let mut voiced: Vec<f32> = win.iter().copied().filter(|&v| v > 20.0).collect();
        if voiced.len() < (win.len() / 4).max(2) {
            track.push(0.0); // filled below
            continue;
        }
        voiced.sort_by(|a, b| a.total_cmp(b));
        track.push(voiced[voiced.len() / 2]);
    }
    // sticky forward-fill, then backfill the leading unvoiced stretch
    let first_voiced = track.iter().position(|&v| v > 0.0)?;
    let head = track[first_voiced];
    for v in track[..first_voiced].iter_mut() {
        *v = head;
    }
    for i in first_voiced + 1..track.len() {
        if track[i] <= 0.0 {
            track[i] = track[i - 1];
        }
    }
    Some((track, hop_samples * step_frames))
}

/// THE single execution point of the inverse shift — shared by the score, cover and audition
/// paths so engine policy can never drift between them. Shifts `audio` back by `-shift`
/// semitones through the Signalsmith engine (time_factor 1.0 = pure transpose); `kappa` sets
/// the formant policy (see DEFAULT_FORMANT_KAPPA); `fed_f0` = (the SHIFTED f0 the model was
/// fed, its hop in output samples) — folded into a sticky base schedule
/// (formant_base_track) so the formant analysis follows the audio's local fundamental
/// instead of its noise-chasing auto-detector (S82/S82b anti-pop). Output length == input
/// length exactly (every caller's trim/pad arithmetic depends on that).
///
/// A stretch failure is LOUD: there is no second engine to fall back to, and returning the
/// un-inverted render would be a silently wrong-pitched result. The Err carries the engine's
/// stable CODE (STRETCH_*) for the frontend error funnel.
pub fn apply_inverse(
    audio: Vec<f32>,
    sample_rate: u32,
    shift: i64,
    kappa: f32,
    fed_f0: Option<(&[f32], usize)>,
) -> Result<Vec<f32>, String> {
    if shift == 0 || audio.is_empty() {
        return Ok(audio);
    }
    // One-time engine anchor so an A/B can prove what actually ran (a stale render cache
    // otherwise reads as "no difference").
    static ENGINE_LOG: std::sync::Once = std::sync::Once::new();
    ENGINE_LOG.call_once(|| {
        tracing::info!("range-extend: inverse engine = signalsmith-stretch 1.3.2 (native formant control, streaming base)");
    });
    let k = kappa.clamp(0.0, 1.0);
    let semis = -(shift as f64); // semitones the AUDIO moves = inverse of the model-side shift
    let schedule =
        fed_f0.and_then(|(f0, hop)| formant_base_track(f0, hop, sample_rate));
    // κ=1 is the plain transpose: skip the formant machinery entirely (zero extra cost).
    let formant = ((1.0 - k) > 1e-3).then(|| utai_stretch::FormantPin {
        semitones: f64::from(k) * semis,
        base_hz: schedule.as_ref().map_or(&[][..], |(t, _)| t.as_slice()),
        base_step: schedule.as_ref().map_or(0, |(_, s)| *s),
    });
    tracing::debug!(
        "range-extend: inverse {semis:+.0} st, formant kappa {k:.2}, base schedule {} pts",
        schedule.as_ref().map_or(0, |(t, _)| t.len())
    );
    let n = audio.len();
    let mut y = utai_stretch::stretch_interleaved(&audio, 1, sample_rate, 1.0, semis, formant)?;
    if y.len() != n {
        // exact-length contract guard, not a fix
        tracing::warn!(
            "range-extend: stretch returned {} samples for {n} — padded/truncated to contract",
            y.len()
        );
        y.resize(n, 0.0);
    }
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn range() -> SpeakerRange {
        SpeakerRange::bounds((48.0, 84.0), (52.0, 79.0))
    }

    /// S81 (A): the score path's bare min/max tiers are gone — both paths now run
    /// `piece_range_shift`, so what used to be `compute_range_shift`'s contract is re-asserted
    /// here through the optimizer, on a 50 fps track built by `score_frame_hz`.
    fn score_track(notes: &[(i64, i64)], transpose: i64) -> Vec<f32> {
        let nn: Vec<i64> = notes.iter().map(|n| n.0).collect();
        let fr: Vec<i64> = notes.iter().map(|n| n.1).collect();
        score_frame_hz(&nn, &fr, &[], &[], transpose)
    }

    #[test]
    fn score_path_leaves_in_range_material_untouched() {
        // The tier-1/2 contract survives the unification: a part comfortably inside the
        // record renders with no shift (and therefore no inverse pass at all).
        let t = score_track(&[(60, 200), (67, 200), (70, 200)], 0);
        assert_eq!(piece_range_shift(&t, None, &range(), 50.0), 0);
    }

    #[test]
    fn score_path_rescues_a_sustained_high_part() {
        // …while genuinely out-of-range material still moves. usable=(48,84) c=(52,79) ⇒
        // ceiling = min(79, 84-2) = 79; a 2 s note at 88 has to come down 9.
        let t = score_track(&[(60, 200), (88, 100)], 0);
        assert_eq!(piece_range_shift(&t, None, &range(), 50.0), -9);
    }

    #[test]
    fn score_path_ignores_a_single_overshooting_frame() {
        // THE S81 (A) defect: the old min/max judged from the extremes, so one portamento
        // overshoot frame transposed the entire part. 20 ms cannot do that any more.
        let mut cents: Vec<f32> = vec![6000.0; 2000];
        cents[900] = 9500.0; // one frame an octave+ above, as an overshoot reads
        let voiced = vec![1u8; cents.len()];
        let t = score_frame_hz(&[], &[], &cents, &voiced, 0);
        assert_eq!(piece_range_shift(&t, None, &range(), 50.0), 0);
    }

    #[test]
    fn score_frame_hz_prefers_the_curve_and_honours_rests() {
        // With a DAW curve the track IS the curve (unvoiced ⇒ 0); without one the note
        // pitches are held for their durations, so a long note outweighs a grace note.
        let cents = vec![6000.0f32, 6000.0, 6000.0];
        let t = score_frame_hz(&[69], &[3], &cents, &[1, 0, 1], 0);
        assert_eq!(t.len(), 3);
        assert_eq!(t[1], 0.0, "unvoiced frame is a hole, not a pitch");
        let t2 = score_track(&[(69, 2), (0, 3), (81, 1)], 0);
        assert_eq!(t2.len(), 6);
        assert!((t2[0] - 440.0).abs() < 0.01);
        assert_eq!(t2[2], 0.0, "rest note (num <= 0) is unvoiced");
        assert!((t2[5] - 880.0).abs() < 0.02);
        // transpose folds into the OUTPUT Hz
        let t3 = score_track(&[(69, 1)], 12);
        assert!((t3[0] - 880.0).abs() < 0.02);
    }

    #[test]
    fn sidecar_parse_and_speaker_fallback() {
        let json = serde_json::json!({
            "vocal_range": { "speakers": { "0": {
                "usable": [48, 84], "comfort": [52, 79]
            } } }
        });
        // every field is #[serde(default)]-tolerant — an empty object is a valid config;
        // `extra` is the flattened Value, so the whole object lands there verbatim
        let mut config: ModelConfig = serde_json::from_str("{}").unwrap();
        config.extra = json;
        let r = speaker_range(&config, 0).unwrap();
        assert_eq!(r.comfort, (52.0, 79.0));
        // an UNTESTED speaker must read as no-record (no silent speaker-0 borrow — audit S60)
        assert_eq!(speaker_range(&config, 3), None);
    }

    fn config_with(sp: serde_json::Value) -> ModelConfig {
        let mut config: ModelConfig = serde_json::from_str("{}").unwrap();
        config.extra = serde_json::json!({ "vocal_range": { "speakers": { "0": sp } } });
        config
    }

    // ── S85 dead-only plan(三轮耳判定案的钉子;memory S85)──

    /// 真实东雪莲 sidecar 的缩影:36..=79 f0 健康(err 1-3/voiced 1),80 半浊(0.67),
    /// 81 近死(0.19),82+ 全死;comfort 被 low_ratio 门砍到 52 —— 三轮的核心教训是
    /// f0 判据贴耳朵、comfort/timbre 判据两个方向都骗人,dead-only 必须只信前者。
    fn dxl_like() -> SpeakerRange {
        let mut semis = serde_json::Map::new();
        for m in 36..=79i64 {
            semis.insert(m.to_string(), serde_json::json!([2, 1.0]));
        }
        semis.insert("80".into(), serde_json::json!([1, 0.67]));
        semis.insert("81".into(), serde_json::json!([6, 0.19]));
        for m in 82..=96i64 {
            semis.insert(m.to_string(), serde_json::json!([9999, 0.0]));
        }
        speaker_range(
            &config_with(serde_json::json!({
                "usable": [36, 80], "comfort": [36, 52], "semitones": serde_json::Value::Object(semis)
            })),
            0,
        )
        .unwrap()
    }

    #[test]
    fn dead_only_rescues_the_climax_note_minimally() {
        // のう(85,73) between rests: 85 dead → minimal landing 79(80 半浊不合格)⇒ -6;
        // 前面的健康 73 短语原位不动 —— 用户耳判通过的 t23 臂逐字。
        let nn = [0, 73, 73, 0, 85, 73, 0];
        let (plan, unfix) = dead_only_plan(&nn, 0, &dxl_like());
        assert!(unfix.is_empty());
        assert_eq!(plan, vec![DeadGroup { start: 4, end: 5, shift: -6 }]);
    }

    #[test]
    fn dead_only_ignores_out_of_comfort_but_singable_material() {
        // 三轮核心:comfort [36,52] 说 64-78 全「越界」,但 f0 判据说能唱 → 一律原位。
        // (二轮 per-note 把整段搬走 = 用户判「灾难」的那台机器,这里钉死不再回来。)
        let nn = [0, 64, 66, 0, 73, 78, 0];
        let (plan, unfix) = dead_only_plan(&nn, 0, &dxl_like());
        assert!(plan.is_empty() && unfix.is_empty());
    }

    #[test]
    fn dead_only_transpose_folds_into_the_verdict() {
        let nn = [0, 73, 0]; // written 73 + transpose 12 = 85 → 同款 -6 救援
        let (plan, _) = dead_only_plan(&nn, 12, &dxl_like());
        assert_eq!(plan, vec![DeadGroup { start: 1, end: 1, shift: -6 }]);
    }

    #[test]
    fn dead_only_dragged_neighbours_must_stay_singable() {
        // 短语 [85, 40]:-6 把 40 拖到 34(窗外=死)→ 无解,必须响亮计数而非静默跳过。
        let nn = [0, 85, 40, 0];
        let (plan, unfix) = dead_only_plan(&nn, 0, &dxl_like());
        assert!(plan.is_empty());
        assert_eq!(unfix, vec![(1, 2)], "无解组带位置(审查 S85:取证要「在哪」)");
    }

    #[test]
    fn dead_only_bounds_record_falls_back_to_bounds() {
        // 无扫描旧记录:dead=出 usable,落点=进 comfort。88→79 = -9;40→52 = +12。
        let r = SpeakerRange::bounds((48.0, 84.0), (52.0, 79.0));
        let (plan, unfix) = dead_only_plan(&[0, 88, 0], 0, &r);
        assert_eq!(plan, vec![DeadGroup { start: 1, end: 1, shift: -9 }]);
        assert!(unfix.is_empty());
        let (plan, _) = dead_only_plan(&[0, 40, 0], 0, &r);
        assert_eq!(plan, vec![DeadGroup { start: 1, end: 1, shift: 12 }]);
    }

    #[test]
    fn dead_only_interior_bridged_weak_slot_rescues_both_ways() {
        // 写侧 longestRun 桥接会把 usable 界内的孤立弱槽留在记录里(合法形态,审查 S85):
        // 78 死于 usable [36,80] 内部 → 无固有方向 → 双向按幅度搜索,-1 落 77 即救。
        let mut semis = serde_json::Map::new();
        for m in 36..=80i64 {
            semis.insert(m.to_string(), serde_json::json!([2, 1.0]));
        }
        semis.insert("78".into(), serde_json::json!([9999, 0.0]));
        let r = speaker_range(
            &config_with(serde_json::json!({
                "usable": [36, 80], "comfort": [36, 80],
                "semitones": serde_json::Value::Object(semis)
            })),
            0,
        )
        .unwrap();
        let (plan, unfix) = dead_only_plan(&[0, 78, 0], 0, &r);
        assert!(unfix.is_empty());
        assert_eq!(plan, vec![DeadGroup { start: 1, end: 1, shift: -1 }]);
    }

    #[test]
    fn dead_group_windows_extend_into_rests_without_overlap() {
        // cum=[0,5,9,19,32,37,45];前间隙 9 帧→pre=4,后间隙 5 帧→post=2(半间隙封顶)。
        let nn = [0, 0, 73, 85, 0, 73];
        let fr = [5i64, 4, 10, 13, 5, 8];
        let plan = [DeadGroup { start: 2, end: 3, shift: -6 }];
        assert_eq!(dead_group_windows(&nn, &fr, &plan), vec![(-6, 5, 34)]);
    }

    #[test]
    fn degenerate_comfort_heals_from_auto_then_usable() {
        // the S60d field case verbatim: slider spiral committed [42,42], auto intact
        let r = speaker_range(
            &config_with(serde_json::json!({
                "usable": [42, 70], "comfort": [42, 42], "comfort_auto": [42, 70]
            })),
            0,
        )
        .unwrap();
        assert_eq!(r.comfort, (42.0, 70.0));
        // auto degenerate too → usable
        let r = speaker_range(
            &config_with(serde_json::json!({
                "usable": [42, 70], "comfort": [42, 42], "comfort_auto": [50, 52]
            })),
            0,
        )
        .unwrap();
        assert_eq!(r.comfort, (42.0, 70.0));
        // usable itself below the minimum span → the record is unusable for shifting
        assert_eq!(
            speaker_range(
                &config_with(serde_json::json!({ "usable": [42, 45], "comfort": [42, 45] })),
                0
            ),
            None
        );
    }

    #[test]
    fn shift_is_bounded_and_degenerate_comfort_never_moves() {
        // Far above the range but still rescuable: the optimizer takes the SMALLEST shift that
        // clears the proven ceiling (105 - 21 = 84 = u_hi) rather than the largest that fits.
        let t = score_track(&[(105, 300)], 0);
        let s = piece_range_shift(&t, None, &range(), 50.0);
        assert_eq!(s, -21);
        assert!(s.abs() <= MAX_RANGE_SHIFT, "the ±{MAX_RANGE_SHIFT} brake is structural");
        // Beyond rescue (115 would need -31): every reachable shift leaves it out of range, so
        // paying for a whole-render recolour buys nothing — the new cost model declines. The
        // pre-S81 code clamped to -24 and recoloured the render for no benefit at all.
        let hopeless = score_track(&[(115, 300)], 0);
        assert_eq!(piece_range_shift(&hopeless, None, &range(), 50.0), 0);
        // degenerate comfort (defensive; speaker_range normally heals first) → never a target
        let degenerate = SpeakerRange::bounds((48.0, 84.0), (52.0, 52.0));
        assert_eq!(piece_range_shift(&t, None, &degenerate, 50.0), 0);
    }

    fn hz(midi: f32) -> f32 {
        440.0 * 2f32.powf((midi - 69.0) / 12.0)
    }

    #[test]
    fn piece_optimizer_counts_the_climax_mass() {
        // the S60d2 field case shape: 96% of frames at 67, 4% climax at 75 — a p98-bounds
        // decision ignored the climax entirely; the optimizer must land it under the
        // margined ceiling (70-2=68): 75-7=68 ⇒ exactly -7, no more (|shift| penalty)
        let r = SpeakerRange::bounds((42.0, 70.0), (42.0, 70.0));
        let mut f0 = vec![hz(67.0); 960];
        f0.extend(vec![hz(75.0); 40]);
        assert_eq!(piece_range_shift(&f0, None, &r, 100.0), -7);
    }

    #[test]
    fn piece_optimizer_ignores_isolated_spikes_and_proven_zone() {
        let r = SpeakerRange::bounds((42.0, 70.0), (42.0, 70.0));
        // 3 spike frames at 90 can't justify dragging 1000 frames below the floor → 0
        let mut f0 = vec![hz(60.0); 1000];
        f0.extend(vec![hz(90.0); 3]);
        assert_eq!(piece_range_shift(&f0, None, &r, 100.0), 0);
        // entirely inside usable → untouched (tiers 1/2 preserved, byte-identical path)
        assert_eq!(piece_range_shift(&vec![hz(65.0); 500], None, &r, 100.0), 0);
    }

    #[test]
    fn piece_optimizer_ignores_phantom_octave_bursts() {
        // S62b field case shape (lengv2.3, HEALTHY record [36,77]): melody well in range,
        // sustained boundary chorus at 76.5 (PROVEN usable), plus scattered short bursts of
        // octave-doubled breath frames at 81. The whole song must render UNTOUCHED — the old
        // code let one phantom frame defeat the tier-2 short-circuit and then chased the
        // burst mass down -6 st, recoloring the entire render.
        let r = SpeakerRange::bounds((36.0, 77.0), (36.0, 77.0));
        let mut f0: Vec<f32> = Vec::new();
        for i in 0..20000 {
            f0.push(hz(55.0 + (i % 17) as f32)); // 55..71 melody mass
        }
        f0.extend(vec![hz(76.5); 800]); // boundary chorus — inside usable
        for _ in 0..5 {
            f0.extend(vec![hz(81.0); 10]); // phantom octave burst (~100 ms)
            f0.extend(vec![hz(65.0); 200]);
        }
        assert_eq!(piece_range_shift(&f0, None, &r, 100.0), 0);
        // the minimal version: a SINGLE spike frame must not defeat the short-circuit either
        let mut f1 = vec![hz(60.0); 5000];
        f1.push(hz(84.0));
        assert_eq!(piece_range_shift(&f1, None, &r, 100.0), 0);
    }

    #[test]
    fn piece_optimizer_still_rescues_sustained_true_high() {
        // A genuinely out-of-range SUSTAINED section (4 s continuous at 80 > u_hi 77) still
        // triggers the rescue — spike hygiene must not neuter the feature. 80 + s ≤ top (75)
        // ⇒ exactly -5 (the |shift| penalty stops there).
        let r = SpeakerRange::bounds((36.0, 77.0), (36.0, 77.0));
        let mut f0 = vec![hz(60.0); 1000];
        f0.extend(vec![hz(80.0); 400]);
        assert_eq!(piece_range_shift(&f0, None, &r, 100.0), -5);
    }

    #[test]
    fn piece_optimizer_shifts_low_material_up() {
        let r = SpeakerRange::bounds((48.0, 84.0), (52.0, 79.0));
        // 50-frame plateaus stepping 30..40 — real f0 is smooth at the 100 fps grid, so the
        // material must be median-filter-stable (a per-frame sawtooth is not a voice).
        let f0: Vec<f32> = (0..550).map(|i| hz(30.0 + ((i / 50) % 11) as f32)).collect();
        // lowest frames at 30 must clear c_lo 52 → +22; the |shift| penalty stops there
        assert_eq!(piece_range_shift(&f0, None, &r, 100.0), 22);
    }

    #[test]
    fn piece_optimizer_discounts_quiet_phantom_sustains() {
        // S62c field case: a separated stem's reverb tail / harmony bleed reads high for
        // SECONDS (defeats the run filter) but sits far below the lead's level — loudness
        // weighting must keep the piece untouched…
        let r = SpeakerRange::bounds((36.0, 77.0), (36.0, 77.0));
        let mut f0 = vec![hz(60.0); 5000];
        let mut en = vec![0.3f32; 5000];
        f0.extend(vec![hz(81.0); 200]); // 2 s sustained phantom above usable
        en.extend(vec![0.01f32; 200]); // …at tail energy (~-30 dB vs lead)
        assert_eq!(piece_range_shift(&f0, Some(&en), &r, 100.0), 0);
        // …while a LOUD sustained true-high still rescues (same shape as the None test).
        let mut f1 = vec![hz(60.0); 1000];
        let mut e1 = vec![0.25f32; 1000];
        f1.extend(vec![hz(80.0); 400]);
        e1.extend(vec![0.35f32; 400]);
        assert_eq!(piece_range_shift(&f1, Some(&e1), &r, 100.0), -5);
    }

    /// Build a record whose raw scan is clean up to `good_hi` and badly voiced above it.
    fn scanned(usable: (f32, f32), good_hi: i64) -> SpeakerRange {
        let mut semis = serde_json::Map::new();
        for midi in 36..=96i64 {
            let entry = if midi <= good_hi {
                serde_json::json!([5.0, 1.0]) // healthy: 5 cents, fully voiced
            } else {
                serde_json::json!([8.0, 0.60]) // pitch still perfect, voicing collapsing
            };
            semis.insert(midi.to_string(), entry);
        }
        speaker_range(
            &config_with(serde_json::json!({
                "usable": [usable.0, usable.1],
                "comfort": [usable.0, usable.1],
                "semitones": serde_json::Value::Object(semis),
            })),
            0,
        )
        .unwrap()
    }

    #[test]
    fn damage_from_scan_is_perceptual_not_linear() {
        // S81 (E): a healthy semitone must cost NOTHING — a scan's 1-37 cent wander across a
        // model's good range is inaudible, and charging for it would make the optimizer chase
        // measurement noise and recolour songs that were fine.
        assert_eq!(damage_from_scan(2.0, 1.00, None), 0.0);
        assert_eq!(damage_from_scan(24.0, 0.96, None), 0.0);
        // a rejected semitone saturates on pitch alone
        assert_eq!(damage_from_scan(9999.0, 0.0, None), 3.0);
        // …and a semitone that keeps perfect pitch while losing voicing is still damaged
        assert!(damage_from_scan(6.0, 0.63, None) > 2.0);
    }

    #[test]
    fn the_timbre_dimension_catches_what_f0_cannot() {
        // THE case the whole F1 change exists for, using the numbers measured off the probe wav
        // on disk: akiko MIDI 80 stores err=2 cents / voiced=1.00 — a perfect score on both f0
        // axes — while measuring -7.7 dB with 88.7% of its energy below 1.5*f0.
        assert_eq!(damage_from_scan(2.0, 1.00, None), 0.0, "f0-only is blind here");
        assert!(
            damage_from_scan(2.0, 1.00, Some((-7.7, 0.887))) > 2.0,
            "with the audio measured, the same semitone reads as badly damaged"
        );
        // a healthy note is still free WITH the dimension present (akiko MIDI 74)
        assert_eq!(damage_from_scan(5.0, 1.00, Some((0.0, 0.109))), 0.0);
        // lengv2.3's near-pure-sine 75 (0.983) vs its healthy 74 (0.467)
        assert!(damage_from_scan(6.0, 1.00, Some((-1.2, 0.983))) > 2.0);
        assert_eq!(damage_from_scan(8.0, 1.00, Some((-8.0, 0.467))), 0.5, "quiet but voiced = graded, not rejected");
        // (S85b: the S83 quiet-cap anchors + escape-valve test were removed with their
        // mechanisms — decision layer back to v0.11.0; the broken-climax case they served is
        // now the score dead-only plan's job, tested in the dead_only_* group below.)
    }

    #[test]
    fn damage_curve_moves_material_off_a_nominally_usable_but_bad_top() {
        // THE S81 defect: `usable` says [36,77] so the pre-S81 tier-2 short-circuit rendered
        // anything inside those 41 semitones untouched — including a top that the record's OWN
        // raw scan shows singing badly. With the scan read as a damage curve, material sitting
        // on the bad top is moved down to the good part instead.
        let r = scanned((36.0, 77.0), 72);
        let mut f0 = vec![hz(60.0); 1000];
        f0.extend(vec![hz(75.0); 100]); // inside `usable`, but the scan says it is damaged
        assert_eq!(piece_range_shift(&f0, None, &r, 100.0), -3, "material should land on the proven-good 72");
    }

    #[test]
    fn a_narrowed_comfort_still_steers_a_scanned_record() {
        // The user lowering the comfort ceiling is a judgement the scan cannot make for them
        // ("above this it sounds bad TO ME"). When the damage curve arrived it briefly replaced
        // the comfort term instead of adding to it, which silently turned every manual comfort
        // adjustment into a no-op. Same scan, two comfort zones, different answers.
        let wide = scanned((36.0, 77.0), 77);
        let mut narrowed = wide;
        narrowed.comfort = (36.0, 62.0); // user says: nothing above 62 on this singer
        let f0 = vec![hz(70.0); 1500];
        assert_eq!(piece_range_shift(&f0, None, &wide, 100.0), 0);
        assert!(
            piece_range_shift(&f0, None, &narrowed, 100.0) < 0,
            "material above the user's comfort ceiling must still be pulled down"
        );
    }

    #[test]
    fn a_clean_piece_inside_a_scanned_range_still_renders_untouched() {
        // The mirror, and the thing that must NOT regress: reading the scan may never start
        // recolouring songs that sit in the genuinely healthy part of the range.
        let r = scanned((36.0, 77.0), 77);
        assert_eq!(piece_range_shift(&vec![hz(60.0); 1500], None, &r, 100.0), 0);
    }

    #[test]
    fn a_tiny_violation_no_longer_buys_a_whole_render_recolour() {
        // S81 (B): 1% of the material out of range used to out-vote a 5-semitone shift because
        // the only cost of shifting was 0.003/st — while the REAL cost is that 100% of the
        // audio goes through the inverse transform. Now it does not pay for itself.
        let r = SpeakerRange::bounds((36.0, 77.0), (36.0, 77.0));
        let mut f0 = vec![hz(60.0); 3000];
        f0.extend(vec![hz(80.0); 30]); // 300 ms — long enough to survive the phantom filter
        assert_eq!(piece_range_shift(&f0, None, &r, 100.0), 0);
    }

    #[test]
    fn ceiling_margin_is_not_charged_twice() {
        // S81 (C): the margin models "the probe was more forgiving than real singing", so it is
        // measured from the PROVEN top. When the record already reports comfort strictly inside
        // usable, that gap has been charged once by the comfort criterion — landing at
        // c_hi - MARGIN would charge it again. Ceiling = min(c_hi, u_hi - MARGIN) = 79, so a
        // 90 top needs -11; the pre-S81 formula (c_hi - MARGIN = 77) pushed it to -13.
        let r = SpeakerRange::bounds((48.0, 83.0), (48.0, 79.0));
        let mut f0 = vec![hz(70.0); 1000];
        f0.extend(vec![hz(90.0); 400]);
        assert_eq!(piece_range_shift(&f0, None, &r, 100.0), -11, "two semitones less push than before");
    }

    #[test]
    fn phantom_islands_separated_by_breaths_stay_phantom() {
        // S81 (D): the run filter used to measure "consecutive VOICED frames", so five 100 ms
        // bursts with breaths between them fused into one 50-frame run and sailed through —
        // then out-massed the shift penalty and recolored the whole song. Each burst is
        // individually phantom; the breaths must keep them that way.
        let r = SpeakerRange::bounds((36.0, 77.0), (36.0, 77.0));
        let mut f0 = vec![hz(60.0); 2000];
        for _ in 0..5 {
            f0.extend(vec![hz(85.0); 10]); // 100 ms above usable
            f0.extend(vec![0.0f32; 30]); // 300 ms breath — unvoiced
        }
        assert_eq!(piece_range_shift(&f0, None, &r, 100.0), 0);
    }

    #[test]
    fn a_run_split_by_a_short_dropout_stays_one_run() {
        // …and the mirror: a genuinely sustained high note with a 2-frame rmvpe dropout in the
        // middle must NOT be chopped into two phantoms (that would neuter the whole feature).
        let r = SpeakerRange::bounds((36.0, 77.0), (36.0, 77.0));
        let mut f0 = vec![hz(60.0); 1000];
        f0.extend(vec![hz(80.0); 200]);
        f0.extend(vec![0.0f32; 2]); // dropout inside the held note
        f0.extend(vec![hz(80.0); 200]);
        assert_ne!(piece_range_shift(&f0, None, &r, 100.0), 0, "sustained high note must still be rescued");
    }

    #[test]
    fn mismatched_energy_length_falls_back_to_unweighted() {
        // S81 (H): a short energy track must read as "no energy information", never as
        // "the frames I could not measure are as loud as the lead".
        let r = SpeakerRange::bounds((36.0, 77.0), (36.0, 77.0));
        let mut f0 = vec![hz(60.0); 1000];
        f0.extend(vec![hz(80.0); 400]);
        let short = vec![0.3f32; 10];
        assert_eq!(piece_range_shift(&f0, Some(&short), &r, 100.0), piece_range_shift(&f0, None, &r, 100.0));
    }

    #[test]
    fn the_inverse_honours_the_exact_length_contract() {
        // Every caller's trim/pad arithmetic depends on len(out) == len(in); shift 0 must be
        // the untouched passthrough (tier 1/2 bit-parity).
        let sr = 44100u32;
        let n = sr as usize; // 1 s
        let mut x = vec![0.0f32; n];
        for (i, v) in x.iter_mut().enumerate() {
            let t = i as f32 / sr as f32;
            *v = (0.4 * (2.0 * std::f32::consts::PI * 220.0 * t).sin())
                + (0.2 * (2.0 * std::f32::consts::PI * 440.0 * t).sin());
        }
        let untouched =
            apply_inverse(x.clone(), sr, 0, DEFAULT_FORMANT_KAPPA, None).expect("shift 0");
        assert_eq!(untouched, x);
        let fed: Vec<f32> = vec![220.0; 101];
        for (shift, kappa, fed_f0) in [
            (-3i64, 0.0f32, None),
            (5, 0.0, Some((fed.as_slice(), sr as usize / 100))),
            (-7, 1.0, None),
        ] {
            let y = apply_inverse(x.clone(), sr, shift, kappa, fed_f0).expect("inverse");
            assert_eq!(y.len(), x.len(), "shift={shift} kappa={kappa}");
            assert!(y.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn the_base_schedule_is_sticky_smooth_and_voiced_only() {
        // 100 fps hop at 48 kHz ⇒ 10 frames per 100 ms window. Layout: 1 s unvoiced lead,
        // 1 s of 220 Hz, 1 s unvoiced (must CARRY 220 — a mid-stream 0 would re-enable the
        // auto-detector), 1 s of 465 Hz (passes through UNquantized — S82c user A/B).
        let hop = 480usize;
        let mut f0 = vec![0.0f32; 100];
        f0.extend(vec![220.0; 100]);
        f0.extend(vec![0.0; 100]);
        f0.extend(vec![465.0; 100]);
        let (track, step) = formant_base_track(&f0, hop, 48000).expect("schedule");
        assert_eq!(step, 4800); // 10 frames × 480 samples
        assert_eq!(track.len(), 40);
        assert!(track.iter().all(|&v| v > 0.0), "no mid-stream zeros: {track:?}");
        assert!((track[0] - 220.0).abs() < 0.01, "leading unvoiced backfills: {}", track[0]);
        assert!((track[15] - 220.0).abs() < 0.01);
        assert!((track[25] - 220.0).abs() < 0.01, "unvoiced stretch carries: {}", track[25]);
        assert!((track[35] - 465.0).abs() < 0.01, "smooth (no quantize): {}", track[35]);
        // all-unvoiced ⇒ None (auto-detect), never a garbage schedule
        assert!(formant_base_track(&vec![0.0f32; 400], hop, 48000).is_none());
    }

    #[test]
    fn record_validation() {
        let ok = serde_json::json!({ "speakers": { "0": { "usable": [42, 70], "comfort": [45, 60] } } });
        assert!(validate_range_record(&ok).is_ok());
        // narrow-but-honest comfort is accepted at write time (read side decides applicability)
        let narrow = serde_json::json!({ "speakers": { "0": { "usable": [42, 70], "comfort": [50, 50] } } });
        assert!(validate_range_record(&narrow).is_ok());
        for bad in [
            serde_json::json!({}),
            serde_json::json!({ "speakers": { "0": { "usable": [70, 42], "comfort": [45, 60] } } }),
            serde_json::json!({ "speakers": { "0": { "usable": [42, 70], "comfort": [40, 60] } } }),
            serde_json::json!({ "speakers": { "0": { "usable": [42, 70], "comfort": [45, 75] } } }),
            serde_json::json!({ "speakers": { "0": { "usable": [42, 70] } } }),
        ] {
            assert!(validate_range_record(&bad).is_err(), "accepted: {bad}");
        }
    }
}
