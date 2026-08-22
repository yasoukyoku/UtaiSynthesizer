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
/// Frames a DEAD region must SUSTAIN to count as musical content (≈250 ms on the 100 fps
/// grid). rmvpe reads breaths/sibilance an octave UP for a few frames at a time; those
/// phantom islands must never trigger a recolour (S62b field case, lengv2.3 — a handful of
/// octave-doubled spikes once dragged whole-song shifts; the dead-only world keeps the same
/// hygiene: a real climax is seconds long and sails through, shorter runs are ignored).
pub const MIN_VIOLATION_MS: f32 = 250.0;
/// Largest ORIGINAL-index step a dead run may take and still count as ONE sustained region,
/// in milliseconds. Bridges rmvpe's blips + the voiceless consonants inside a held climax;
/// a real breath/phrase gap (100-300 ms) must break the region.
pub const GAP_TOL_MS: f32 = 30.0;

/// Both thresholds are expressed in TIME, not frames: the cover path judges on a 100 fps grid
/// and the score path on the DAW's 50 fps grid, so a frame-count constant would silently mean
/// 250 ms on one and 500 ms on the other.
fn frames_for(ms: f32, fps: f32) -> usize {
    ((ms * fps / 1000.0).round() as usize).max(1)
}

/// Context each windowed donor slice keeps on BOTH sides of its dead window (S85e). Covers the
/// pipeline warm-up at slice edges (rmvpe/ContentVec context, RVC t_pad, chunk seams) with the
/// window interior + 10 ms crossfades sitting deep inside; past ~300 ms edge effects are gone,
/// so 1.5 s is generous. Cost scales with it linearly — a dead 2% of a song renders as ~15%.
pub const DONOR_PAD_SECONDS: f32 = 1.5;

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
    /// MIDI bounds, inclusive. ⚠ S146f: this is the USER's line — "which score notes should be
    /// rescued" — and it is NOT a bound on what the model may be asked to sing. See
    /// `reach` below and `slot_reachable`.
    pub usable: (f32, f32),
    pub comfort: (f32, f32),
    /// S146f: the SCAN's own usable bounds (record key `usable_auto`; absent ⇒ = `usable`,
    /// which is a statement of fact for pre-S146e records — nothing could edit `usable` then).
    ///
    /// ⛔ Why this is a separate field, measured rather than assumed: `usable` was being used for
    /// two jobs whose correct response to the user narrowing it is OPPOSITE.
    ///   ⑴ "is this score note dead (⇒ rescue it)" — narrowing MUST bite; that is the knob.
    ///   ⑵ "after moving the phrase by s, can every note still be voiced" — this asks about the
    ///      MODEL's physical ability, not the user's intent, and narrowing must NOT bite.
    /// Sharing one predicate made ⑵ tighten with ⑴, which pushed every existing landing deeper:
    /// measured on the user's own model and song (akiko, 炉心融解), dropping the ceiling 79→77
    /// moved every rescue from −2/−5/−7 to −3/−6/−8, and 79→74 took it to −6/−9/−11 (dose
    /// 251 → 523 semitone·seconds) — while the group count and rescued seconds stayed IDENTICAL.
    /// The user heard that as "把可用上限往下调 反而是负效果". With the split, the same sweep
    /// holds the landings at −1/−2/−5/−7 all the way down (dose 256-258) and the knob becomes
    /// monotone: lowering it only ADDS rescues, it never makes an existing one worse.
    /// ⇒ 用户 2026-08-15 拍板:「可用上限」只管「哪些音要救」。
    pub reach: (f32, f32),
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
    /// S151: the scan's RAW `low_ratio` per slot (×255), i.e. how much of the note's energy sits
    /// below 1.5·f0 — "sings the note, has no voice left".
    ///
    /// ⛔ Why it cannot be read off `damage`: `damage_from_scan` gives `low_ratio` a **free zone
    /// below 0.55**, so on akiko the six slots 73..78 (0.100 / 0.121 / 0.181 / 0.211 / 0.276 /
    /// 0.388) are all exactly 0 damage — the landing rule is structurally blind to a 4× spread in
    /// the one axis that has ever agreed with the user's ear. Measured on the user's own two
    /// renders: notes the model was actually asked to sing at **77-78** develop an in-note
    /// envelope wobble **30 %** of the time against **7 %** at ≤76 (Fisher p = 0.0026), and inside
    /// one fixed shift the vowel collapses −2.74 dB at landings > 73 against −0.95 at ≤73
    /// (p = 4.2e-10). See [`landing_extra_depth`].
    pub thin: Option<[u8; DAMAGE_SLOTS]>,
}

/// Level (dB below the probe scale's own loudest note) at which `damage_from_scan` starts
/// counting loss, and — since S146f — the cut at which the onset probe vetoes a LANDING.
///
/// ⚠ Deliberately ONE constant for both, and deliberately not a new number: `rms_db` is measured
/// relative to each probe's own peak (`commands/inference.rs:1210-1215`), so the two passes are
/// on the same scale and the threshold that already means "damage" for one means it for the
/// other. Calibration on the user's own record (akiko, 「あ」 pass over its usable band):
/// 72 −0.7 · 73 −0.5 · 76 −3.0 · 77 −1.4 · 78 −0.3 · 79 −1.3 · **80 −6.7** — only the slot that
/// is actually collapsing crosses it.
const RMS_FREE_DB: f64 = -6.0;

impl SpeakerRange {
    /// Bounds-only record (no raw scan) — the pre-S81 shape.
    pub fn bounds(usable: (f32, f32), comfort: (f32, f32)) -> Self {
        Self { usable, comfort, reach: usable, damage: None, slot_flags: None, thin: None }
    }

    /// S85 dead-only: can the model produce a pitch AT ALL at this (integer) MIDI slot?
    /// With a raw scan = the usable-grade f0 criterion per slot (窗外/未测 = false — "never
    /// tested" must never read as "fine"); bounds-only records fall back to the usable bounds.
    pub(crate) fn slot_singable(&self, midi: i64) -> bool {
        // S146e: the `usable` bounds are ANDed in on BOTH arms — they are the user's knob, and
        // until now the raw-scan arm ignored them completely. That is the bug the user reported
        // as "调节那个可用范围没用,比如我想让高音用舒适的办法去唱也做不到": narrowing the upper
        // bound in the model manager changed the stored record, `rangeRecordSig` correctly
        // invalidated the render — and the render came back IDENTICAL, because every predicate
        // downstream read `slot_flags` only.
        //
        // ⚠ Direction check: this can only ever REMOVE slots from singable, i.e. mark MORE notes
        // dead ⇒ more phrases handed to the rescue. It cannot silently start singing something.
        //
        // ⚠ Zero-change today, live tomorrow: measured across all eight installed records
        // (scratchpad/knob_feasibility.py, 炉心融解 803 triples), `∧usable` moves nothing — the
        // scan derives `usable` from the same per-slot data, so the bounds are already implied by
        // the flags. The AND only bites once the user drags the slider, which is exactly the
        // property wanted: it cannot regress today's output, and it gives the knob teeth.
        if !(self.usable.0..=self.usable.1).contains(&(midi as f32)) {
            return false;
        }
        self.slot_voiceable(midi)
    }

    /// S146f: can the MODEL voice this slot at all — the scan's verdict, with the scan's own
    /// bounds. Deliberately blind to the user's `usable` line.
    ///
    /// ⛔ This is the predicate for "after moving the phrase, does every note still come out",
    /// which is a question about the model, not about what the user wants rescued. Folding the
    /// user's line into it is what made the ceiling knob non-monotone in quality (see `reach`).
    fn slot_reachable(&self, midi: i64) -> bool {
        (self.reach.0..=self.reach.1).contains(&(midi as f32)) && self.slot_voiceable(midi)
    }

    /// The scan's per-slot f0 verdict alone (no bounds of any kind).
    fn slot_voiceable(&self, midi: i64) -> bool {
        match &self.slot_flags {
            Some(f) => {
                let slot = midi - DAMAGE_LO_MIDI as i64;
                (0..DAMAGE_SLOTS as i64).contains(&slot)
                    && f[slot as usize] & SLOT_SINGABLE != 0
            }
            None => true, // bounds-only record: the caller's bounds check IS the whole predicate
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

    /// S146e: the PREFERRED landing — landing-grade *and* inside the user's comfort band.
    ///
    /// ⛔ Deliberately NOT folded into `slot_landing_ok` as a hard AND. Measured on the eight
    /// installed records: 东雪莲's stored comfort is [36,52] while the phrase lives at 75-85, so a
    /// hard AND takes it from "10 groups rescued / 0 unsolvable" to "0 rescued / 10 unsolvable" —
    /// i.e. the knob would silently switch range extension OFF for that model. `comfort` is a
    /// *preference* (where the model sounds good), `usable` is a *bound* (where it can sing at
    /// all); only the latter can be load-bearing. See `minimal_rescue_shift`'s two passes.
    fn slot_landing_preferred(&self, midi: i64) -> bool {
        self.slot_landing_ok(midi) && (self.comfort.0..=self.comfort.1).contains(&(midi as f32))
    }

    /// S151: the scan's raw `low_ratio` at an integer slot. `None` = no scan, or off the scanned
    /// window — and OUTSIDE the window it must read as the WORST possible value, never as "fine"
    /// (the same rule `damage_at` already follows).
    pub(crate) fn thinness(&self, midi: i64) -> Option<f32> {
        let t = self.thin.as_ref()?;
        let slot = midi - DAMAGE_LO_MIDI as i64;
        Some(match (0..DAMAGE_SLOTS as i64).contains(&slot) {
            true => t[slot as usize] as f32 / 255.0,
            false => 1.0,
        })
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
            (((RMS_FREE_DB - rms_db) / 12.0).clamp(0.0, 1.0) as f32) * DAMAGE_MAX,
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
    // S146f: the scan's own bounds. Absent ⇒ = `usable` (a fact for pre-S146e records: nothing
    // could edit it then). ⚠ A hand-poisoned sidecar could hold a `usable_auto` NARROWER than
    // `usable`; union them so the split can only ever widen what the drag check accepts, never
    // secretly narrow it below what today's build already allows.
    let reach = pair("usable_auto")
        .map(|a| (a.0.min(usable.0), a.1.max(usable.1)))
        .unwrap_or(usable);
    if usable.1 - usable.0 < MIN_COMFORT_SPAN {
        return None;
    }
    // S146f: comfort is bounded by `reach`, NOT by the user's `usable`. Before the split those
    // were the same number and "the landing band lives inside the usable band" was right; after
    // it, `usable` is a score-side rescue line while comfort describes LANDINGS, which `reach`
    // governs. Keeping the old bound had a concrete cost: narrowing the ceiling to 74 dragged
    // the user's comfort from 79 down to 74 (the UI clamp), after which no landing could ever
    // reach it and the knob silently stopped doing anything — every group logged the fallback.
    let comfort = [pair("comfort"), pair("comfort_auto"), Some(usable)]
        .into_iter()
        .flatten()
        .find(|c| c.1 - c.0 >= MIN_COMFORT_SPAN
            && band_fits((c.0 as f64, c.1 as f64), (reach.0 as f64, reach.1 as f64)))?;
    // S81 (E): fold the raw per-semitone scan into a damage curve. Absent/garbage scan ⇒ None
    // ⇒ every consumer falls back to the pre-S81 ladder (old records keep working untouched).
    // S85: the same pass derives per-slot f0-axes flags for the score dead-only plan.
    let mut flags = [0u8; DAMAGE_SLOTS]; // untested slot = no flags = dead & unlandable
    let mut thin = [255u8; DAMAGE_SLOTS]; // untested slot = maximally thin, never "fine"
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
            if let Some((_, low_ratio)) = timbre {
                thin[slot as usize] = (low_ratio * 255.0).round().clamp(0.0, 255.0) as u8;
            }
            flags[slot as usize] = (if err < 100.0 && voiced > 0.5 { SLOT_SINGABLE } else { 0 })
                | (if err <= 50.0 && voiced >= 0.9 { SLOT_LANDING } else { 0 });
            seen += 1;
        }
        (seen >= 2).then_some(d)
    });
    // S146b: a SECOND probe pass, shaped like real singing (short note + voiceless onset),
    // may narrow the LANDING band. Why this exists: the scan's only probe is a 400 ms 「あ」
    // with no onset consonant (rangeTest.ts), and that is the easiest thing a model ever has
    // to sing. Measured on akiko at MIDI 80 — same pitch, same duration, only the onset
    // differs: 「ま」(voiced onset) −0.2 dB / voiced 1.00, 「か」(voiceless onset) −7.9 dB /
    // voiced **0.33**. The song agrees at scale (unrescued MIDI 80, n=45: voiceless onset 64%
    // unvoiced, voiced onset 23%, no onset 8%; the whole low register 0/548). So a rescue that
    // lands a dead note on a slot the 「あ」 probe called comfortable can still come out mute —
    // which is exactly what the user heard: 「おもうううう」 WAS rescued to 79 and う@85→79 still
    // measured voiced 0.17.
    //
    // ⛔ This can only ever CLEAR the bit, never set it: the extra probe is allowed to say
    // "that landing is not safe after all", never "…is safe after all". Consequences:
    //   * the set of DEAD notes is untouched (SLOT_SINGABLE is not read here) ⇒ this change
    //     rescues exactly the same notes as before, just lands them somewhere the model can
    //     actually phonate. Zero extra notes dragged through the inverse — which matters,
    //     because widening the dead set is a move the user has already rejected by ear
    //     (S145: 「反而没那么自然」), and it measured 1 real rescue per 15 healthy notes dragged.
    //   * a record without this key behaves byte-identically to before (old records keep working).
    if let Some(m) = sp.get("semitones_onset").and_then(|m| m.as_object()) {
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
            // S146f: the level column, when the record carries it. Why this had to be added:
            // S81 spent a whole session establishing that the f0 pair CANNOT see a timbre/level
            // collapse — and S146b then built this second probe on exactly that pair. Measured
            // counterexample straight off disk: akiko's 「か」 probe at MIDI 80 renders at
            // **rms −12.27 dB** (its own scale's peak as 0) with HNR 12.44 against a 24.34
            // neighbour mean — i.e. all but mute — and the stored tuple was `[3, 1]`: perfect
            // pitch, perfect voicing, LANDING left set.
            //
            // ⚠ Only the LEVEL column is consumed. `low_ratio` is stored from this pass too, but
            // vetoing on it needs a distribution nobody has yet (no record on disk has ever
            // carried onset-pass timbre columns), and akiko's 「あ」 pass already sits at 0.629 at
            // slot 79 — above `damage_from_scan`'s 0.55 free point — so a naive AND there would
            // move real landings on a guess. Decide it from data after the first re-scan.
            let onset_rms = a.get(2).and_then(|x| x.as_f64());
            let level_ok = onset_rms.map_or(true, |r| r >= RMS_FREE_DB);
            if !(err <= 50.0 && voiced >= 0.9 && level_ok) {
                flags[slot as usize] &= !SLOT_LANDING;
            }
        }
    }
    let slot_flags = damage.is_some().then_some(flags);
    Some(SpeakerRange { usable, comfort, reach, damage, slot_flags, thin: damage.map(|_| thin) })
}

/// 一个区间在参照系里站不站得住 —— **读侧愈合与写侧闸共用的唯一判据**。
///
/// ⛔ `within` 永远是 `reach`(扫描量出来的带),不是用户的 `usable`:S146f 起两个边界正交,
/// `usable` 说「哪些音要救」而目标范围说「落点去哪」,后者归扫描管。把这条写成两份的代价
/// 已经兑现过一次 —— 前端那份镜像(`rangeBounds.ts::fitsIn`)当时漏改,用户存进去 79、
/// 读回来 74。⚠ 那份镜像仍然存在(前端要在写盘前预览后端会不会收),改这条**必须两边一起改**。
fn band_fits(band: (f64, f64), within: (f64, f64)) -> bool {
    band.0 <= band.1 && band.0 >= within.0 && band.1 <= within.1
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
        // S146f: comfort is bounded by the SCAN's band (`usable_auto` ∪ `usable`), not by the
        // user's rescue line. `usable` narrows to say "rescue more notes"; landings are governed
        // by what the model can voice, so a comfort target above the rescue line is a legal and
        // useful thing to ask for ("land no higher than 79 even though I want 74+ rescued").
        // Records written before S146e carry no `usable_auto`, so this stays the old check there.
        let (r_lo, r_hi) = pair("usable_auto").map_or((u_lo, u_hi), |(a, b)| (a.min(u_lo), b.max(u_hi)));
        if !(u_lo <= u_hi && band_fits((c_lo, c_hi), (r_lo, r_hi))) {
            return Err("RANGE_INVALID".to_string());
        }
    }
    Ok(())
}

// NOTE: "which speaker governs a blend" = the existing ①c `crate::inference::dominant_speaker`
// (max-weight entry, else speaker_id) — reused, NOT re-implemented here (NO-dup).


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
///
/// `frames` is the per-triple duration on the score's 50 fps grid (same slice the splicer's
/// `dead_group_windows` reads) — it exists only for the trim block inside
/// [`dead_only_plan_with`](the `ms(..)` closure), which needs to know how much sung material a
/// cut would actually free. ⚠ S158:这里原来指向一个叫 `trim_freed_ms` 的函数,**全仓没有**
/// —— doc 指向一个不存在的符号,读的人只会以为是自己没找到。
pub fn dead_only_plan(
    note_nums: &[i64],
    frames: &[i64],
    transpose: i64,
    range: &SpeakerRange,
) -> (Vec<DeadGroup>, Vec<(usize, usize)>) {
    dead_only_plan_with(note_nums, frames, transpose, range, RescueTuning::from_env())
}

/// The two S151 knobs, passed by value so **no test ever reaches the process environment**
/// (a test that does changes answer depending on what the machine has exported, and it fails in
/// the direction that hurts — it passes SILENTLY; S150 paid for that on the phase lock).
/// [`RescueTuning::today`] is by definition the shipped behaviour.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RescueTuning {
    /// `(head_ms, tail_ms)` a cut must free before it is worth its seam; `None` = no trimming.
    pub trim: Option<(f32, f32)>,
    /// `None` = today's landing rule verbatim. `Some(n)` = the S151 arm: the depth budget widens
    /// to `shallowest + n` **and** the ranking gains a `low_ratio` tier.
    /// ⛔ 两件事必须由**同一个开关**控制:只放宽预算而不换排序,等于让「最浅优先」把死音停得更
    /// 靠上;只换排序而不放宽预算,在 akiko 上也已经会改今天的落点(78 → 77)。
    pub landing: Option<i64>,
    /// S159 —— [`LANDING_RATIO_TWO_ST`] 的**可扫版本**(出厂 = 那个常量本身)。
    ///
    /// ⛔ 它**不是**新旋钮:`new()` / `today()` / `from_env()` 一律填常量,生产上没有第二条路。
    /// 它存在的唯一理由是 S157 就登记、至今没做的那件事 —— **给 14 重新定价**:
    /// 那个常量是私有 `const`、**没有任何运行期缝**,扫它只能「改常量 + 重编译」,一个值一遍,
    /// 而且一改 `changing_a_production_default_forces_a_paired_version_bump` 当场红。
    /// ⇒ 加一个**只有计划台会动**的自由度(`with_cap`),让一次 0.04 s 的跑出整张表。
    /// ⚠ 要动它当默认,改的仍然是 [`LANDING_RATIO_TWO_ST`]。
    pub cap: i64,
    /// S159zi —— [`SPLIT_MIN_COST_DEFAULT`] 的**可扫版本**(出厂 = 那个常量本身)。
    ///
    /// ⛔ 与 [`Self::cap`] 同一个形状、同一条理由,**不是新旋钮**:`new()`/`today()`/`from_env()`
    /// 一律填常量,生产上没有第二条路。它存在是因为那个门槛**是我拍的、不是量出来的**,
    /// 而用户报的两处(S159ze 的 480 ms vs 门 500;S159zi 的 5760 vs 门 6000)**连着两次
    /// 卡在门外一点点** —— 那是「门的数值没有出处」的指纹,不是巧合。
    /// ⇒ 给计划台一个自由度,让一次跑出整张定价表(`with_split_cost`)。
    ///
    /// ⭐ 它还是**判据的隔离手段**:一级拆([`SPLIT_MIN_INTERIOR_NOTES`])那几条判据的夹具
    /// 正好也是二级那一刀的形状,两级同时开火时它们的阴性对照会失效。⇒ 那些判据显式写
    /// `with_split_cost(f32::INFINITY)` 把二级关掉 —— **隔离,而不是照新结果改期望值**。
    pub split_cost: f32,
}

impl RescueTuning {
    /// S159 —— 只换深度上限的那一条臂(计划台专用)。
    pub fn with_cap(self, cap: i64) -> Self {
        Self { cap, ..self }
    }

    /// S159zi —— 只换拆组门槛的那一条臂(计划台与判据隔离用,见 [`Self::split_cost`])。
    pub fn with_split_cost(self, split_cost: f32) -> Self {
        Self { split_cost, ..self }
    }
}

impl RescueTuning {
    /// Exactly what ships today.
    ///
    /// ⛔⛔ S157:`landing` 这里原来写的是**字面量 `None`**,而不是 [`LANDING_DEFAULT`]。
    /// 两个值今天恰好相同,所以它是一条**没有人核验过的 doc**:哪一天翻了 `LANDING_DEFAULT`,
    /// 经它取臂的那一批判据会**继续静默地测旧臂**,而「Exactly what ships today」这句话
    /// 一个字都不用改就变成假的。
    /// ⭐ 这正是 S156 在 `psola.rs` 上抓到的那条形状(「改了默认」与「改了行为」在测试文件里
    /// 可以是完全分开的两件事),换了个文件又长了一次。
    /// ⇒ 现在两个字段都从常量取,并由
    /// `the_today_tuning_is_literally_the_shipped_defaults` 钉住。
    /// ⚠ 要「S151 之前那条臂」的判据请显式写 `RescueTuning::new(None, None)`,别借 `today()`。
    pub fn today() -> Self {
        Self {
            trim: TRIM_DEFAULT,
            landing: LANDING_DEFAULT,
            cap: LANDING_RATIO_TWO_ST,
            split_cost: SPLIT_MIN_COST_DEFAULT,
        }
    }

    pub fn new(trim: Option<(f32, f32)>, landing: Option<i64>) -> Self {
        Self { trim, landing, cap: LANDING_RATIO_TWO_ST, split_cost: SPLIT_MIN_COST_DEFAULT }
    }

    pub fn from_env() -> Self {
        Self {
            trim: parse_trim(std::env::var("UTAI_RANGE_TRIM").ok().as_deref()),
            landing: parse_landing(std::env::var("UTAI_RANGE_LANDING").ok().as_deref()),
            // ⛔ 深度上限与拆组门槛**都没有** env 缝:生产各只有一条路。
            cap: LANDING_RATIO_TWO_ST,
            split_cost: SPLIT_MIN_COST_DEFAULT,
        }
    }
}

/// [`dead_only_plan`] with the passenger-trim thresholds passed in instead of read from the
/// environment — the shape `apply_inverse`/`apply_inverse_with` already uses here.
///
/// ⛔ Why the split exists: a test that reaches the env is a test that changes answer depending on
/// what the *machine* has exported, and it fails in the direction that hurts — it passes SILENTLY
/// (S150 paid for this on the phase lock). Every test in this file pins its arm explicitly;
/// `None` is by definition the pre-S151 behaviour.
pub fn dead_only_plan_with(
    note_nums: &[i64],
    frames: &[i64],
    transpose: i64,
    range: &SpeakerRange,
    tune: RescueTuning,
) -> (Vec<DeadGroup>, Vec<(usize, usize)>) {
    let trim = tune.trim;
    let eff = |n: i64| (n + transpose).clamp(1, 127); // mirror transpose_note_pitch's clamp
    let ms = |from: usize, to: usize| -> f32 {
        let f: i64 = (from..to).map(|k| frames.get(k).copied().unwrap_or(0).max(0)).sum();
        f as f32 * 1000.0 / super::score2svc::CV_FPS as f32
    };
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
        let dead_at: Vec<usize> =
            (i..=j).filter(|&k| !range.slot_singable(eff(note_nums[k]))).collect();
        if !dead_at.is_empty() {
            // S151: the rescued span. Default = the whole phrase (pre-S151, bit-for-bit); with the
            // knob on, each SIDE is cut back to the dead run when doing so frees enough sung
            // material to be worth the seam it creates. ⛔ The DEAD set is untouched by
            // construction — every dead note lies inside [first_dead, last_dead] — so this cannot
            // change which notes get rescued, only who rides along.
            let dead: Vec<i64> = dead_at.iter().map(|&k| eff(note_nums[k])).collect();
            let whole: Vec<i64> = (i..=j).map(|k| eff(note_nums[k])).collect();
            // ⛔⛔ WHICH phrases get rescued is decided on the WHOLE phrase, before any trimming,
            // i.e. exactly as it was decided before S151. Dropping passengers also drops their
            // `slot_reachable` constraints, so a phrase that has no landing today CAN acquire one
            // the moment its passengers leave — measured, not hypothesised: `[85, 40]` against
            // dxl_like is unfixable as a phrase (−6 puts 40 outside the scanned window) and
            // rescuable the moment the 40 is dropped. That is a real improvement and it is NOT
            // this knife: it changes which notes get sung differently, so it needs its own
            // decision and its own blind test. Bundling it would make any A/B unattributable.
            // Pinned by `trimming_may_not_turn_an_unfixable_phrase_into_a_rescue`.
            let Some(whole_shift) =
                minimal_rescue_shift_capped(&dead, &whole, range, tune.landing, tune.cap)
            else {
                unfixable.push((i, j));
                i = j + 1;
                continue;
            };
            // ── S159z —— 先把这一句的死音切成**簇**:相邻两个死音之间夹着
            // ≥ [`SPLIT_MIN_INTERIOR_NOTES`] 个可唱音,就在那里断开(定价见那份 doc)。
            // ⛔ 「这一句救不救」仍然由**整句**的 `whole_shift` 决定(上面那条 `else` 分支),
            //    拆簇只改**谁陪着走多深**,不改**哪些音被救** —— 由
            //    `splitting_never_changes_which_notes_are_rescued` 钉住。
            let clusters: Vec<(usize, usize)> = {
                let mut v = Vec::new();
                let (mut cs, mut prev) = (dead_at[0], dead_at[0]);
                for &d in &dead_at[1..] {
                    // 同一句内两个死音之间全是可唱音(休止会先把句子断开)⇒ 直接数下标差。
                    if d - prev - 1 >= SPLIT_MIN_INTERIOR_NOTES {
                        v.push((cs, prev));
                        cs = d;
                    }
                    prev = d;
                }
                v.push((cs, prev));
                // ── S159zi —— 第二级:按**深度差**再拆(见 [`SPLIT_MIN_COST_DEFAULT`])。
                // 一级只看「唱不唱得动」,而 3:16 那一处三个音**全是死音**,它看不见。
                let need = |cf: usize, cl: usize| -> Option<i64> {
                    let d: Vec<i64> = (cf..=cl)
                        .filter(|&k| !range.slot_singable(eff(note_nums[k])))
                        .map(|k| eff(note_nums[k]))
                        .collect();
                    let all: Vec<i64> = (cf..=cl).map(|k| eff(note_nums[k])).collect();
                    minimal_rescue_shift_capped(&d, &all, range, tune.landing, tune.cap)
                };
                let mut todo: Vec<(usize, usize)> = v;
                todo.reverse();
                let mut fine: Vec<(usize, usize)> = Vec::new();
                while let Some((cf, cl)) = todo.pop() {
                    // 这一簇里的死音下标 —— 断点只许落在**它们之间**(见那份 doc 的不变量)。
                    let ds: Vec<usize> = (cf..=cl)
                        .filter(|&k| !range.slot_singable(eff(note_nums[k])))
                        .collect();
                    let here = match need(cf, cl) {
                        Some(h) if ds.len() >= 2 => h,
                        _ => {
                            fine.push((cf, cl));
                            continue;
                        }
                    };
                    let mut best: Option<(usize, usize, f32)> = None;
                    for w in 1..ds.len() {
                        let (p, q) = (ds[w - 1], ds[w]);
                        let (Some(ls), Some(rs)) = (need(cf, p), need(q, cl)) else { continue };
                        // ⛔ 只算两侧,不算夹心 —— 夹心是一级那一刀的账(doc)。
                        let gain = ms(cf, p + 1) * (here.abs() - ls.abs()).max(0) as f32
                            + ms(q, cl + 1) * (here.abs() - rs.abs()).max(0) as f32;
                        if best.is_none_or(|(_, _, g)| gain > g) {
                            best = Some((p, q, gain));
                        }
                    }
                    match best {
                        Some((p, q, gain)) if gain >= tune.split_cost => {
                            tracing::info!(
                                "range: cluster notes[{cf}..={cl}] at {here:+} st split at \
                                 [{p}|{q}] — {gain:.0} ms·st of chaperone depth freed"
                            );
                            todo.push((q, cl));
                            todo.push((cf, p));
                        }
                        _ => fine.push((cf, cl)),
                    }
                }
                fine.sort_unstable();
                fine
            };
            let n_clusters = clusters.len();
            for (ci, &(cluster_first, cluster_last)) in clusters.iter().enumerate() {
            // 这一簇可以占到多宽。⛔⛔ **内部的那一侧一格都不许往夹心里伸** ——
            // 第一版写成「伸到邻簇的死音前一格」,那样相邻两簇会把夹心**同时认领**,
            // 拼接器会把同一段材料贴两次。而且伸进去本来就与这一刀的意图相反:
            // 夹心整段留在 base 才是目的。
            // ⇒ 只有**句首那一侧**和**句尾那一侧**保留 S151 的裁剪逻辑;
            //   只有一簇时 `(lo, hi) == (i, j)`,与 S159z 之前逐位相同。
            let lo = if ci == 0 { i } else { cluster_first };
            let hi = if ci + 1 == n_clusters { j } else { cluster_last };
            let dead: Vec<i64> = (cluster_first..=cluster_last)
                .filter(|&k| !range.slot_singable(eff(note_nums[k])))
                .map(|k| eff(note_nums[k]))
                .collect();
            let (first_dead, last_dead) = (cluster_first, cluster_last);
            let (mut a, mut b) = (lo, hi);
            if let Some((head_ms, tail_ms)) = trim {
                // ⛔⛔ S158 —— **一刀只有在它自己造出来的那条缝落得下去的时候才许下。**
                //
                // 机理:`dead_group_windows` 的 `GUARD_FRAMES` 会让窗**倒着伸进被卸掉的
                // 那个乘客** `GUARD_FRAMES` 帧,好让 10 ms 交叉淡化压在它身上而不是压在
                // 被救音的起音上。那 40 ms 因此是**由 donor 渲的**(在 `shift` 位),
                // 可它从来没过 `minimal_rescue_shift` 的 `slot_reachable` —— 那个检查只看
                // `all`(裁剪前)或 `kept`(裁剪后),两边都不含窗外那一个音。
                //
                // 今天这是死代码(`TRIM_DEFAULT` 之前是 `None`);S158 翻默认的那一秒它就活了。
                // ⚠ 实测:炉心融解 +7 × akiko,13 条 trim 臂里被拖进窗的乘客一共 39 个,
                // **0 个够不着**(reach = [36,80],最深的 −14 也只把 MIDI 60 拖到 46)。
                // ⇒ 这一条今天**一个字节都不改输出**,它挡的是别的模型/别的谱。
                //
                // 退法选「不裁那一侧」而不是「不伸护栏」:后者会把淡化重新压回被救音的起音,
                // 而那正是护栏存在的理由 —— 用一个已知的坏去换另一个已知的坏是没有意义的。
                // ⛔⛔ S159ze —— 它**必须响**。今天它静默返回 false,于是「护栏拦下的一刀」与
                // 「本来就不该裁的一刀」在输出上**一模一样**(铁律:一条闸的红必须能被归因)。
                // ⚠ 而且它的承重刚刚变大了:下面那条按代价的门**恰恰只在大 |s| 上加刀**,
                //    而 `eff(n)+s` 掉出 `reach` 的概率随 |s| **单调上升**。
                let guard_ok = |k: Option<usize>, s: i64| -> bool {
                    match k.and_then(|k| note_nums.get(k).copied()) {
                        Some(n) if n > 0 => {
                            let ok = range.slot_reachable(eff(n) + s);
                            if !ok {
                                tracing::warn!(
                                    "range: guard blocked a trim — note[{}] eff {} at {s:+} st is \
                                     outside the scanned window; keeping it aboard",
                                    k.unwrap_or(0),
                                    eff(n),
                                );
                            }
                            ok
                        }
                        _ => true, // 没有邻音 / 那边是休止 ⇒ 护栏伸不进任何唱音
                    }
                };
                // ⛔⛔⛔ S159f / S159g —— **别再往这里加「按接点音程决定裁不裁」的规矩。**
                //
                // S159f 加过一条(`TRIM_MIN_JOINT_INTERVAL = 5`:裁剪暴露出来的接点音程 < 5 半音
                // 就不裁那一侧)。用户实机一听**更糟**:同一句里能听出缝的音从 1 个变成 3-4 个。
                // 撤掉之前把现场量清楚了,四条读数:
                //
                // ⑴ 用户听到的「缝」**不是拼接缝**,是 **donor 那一路自己在音符交界处的塌陷** ——
                //    5 ms 包络上一个 ~40 ms 宽、电平掉 2-4 dB、**谱心塌掉 20-30%** 的坑。
                //    同一条交界([799]→[800])在「裁了」与「没裁」两遍里读数**完全相同**
                //    (1.85 dB / 28.7%):拼接点落在它前面约 20 ms,坑的两侧本来就都取自 donor。
                // ⑵ 它**随位移深度走**(同一批音、同一份谱、只有位移不同的几遍真渲染):
                //    [801] −7 → 1.42 dB,−8 → 2.94 dB;[761] −9 → 3.36,−10 → 4.47;
                //    [762] −9 → 0.38,−10 → 2.48。base↔base 的元音交界地板是 p50 0.9 dB / 8%。
                // ⑶ ⛔ **不是 PSOLA 把干净音频弄坏的**:同一段 base 音频过一遍生产口径的
                //    `psola_shift_env` +8,交界塌陷 1.11-1.43 dB —— 与没过 PSOLA 的原始几乎相同。
                // ⑷ ⛔ 也**不是「喂进去的是阶梯基频、音频却是滑音」**(交界处两者差到 146 音分):
                //    换成从音频自己测出来的滑音轨再跑一遍,逐个交界读数几乎不动(1.12 / 1.26 / 1.46)。
                //
                // ⇒ 裁剪的真实作用是**减少落在 donor 里的音符交界数**。挡掉裁剪 = 把 4 个乘客
                //    连同它们的 3 个交界一起拖进 donor ⇒ 多出 2 个可闻的坑。**方向是反的。**
                // ⇒ 今天:该裁就裁,不看接点音程。真正要修的是 ⑴ 那个塌陷本身(还没定位到层)。
                let (freed_head, freed_tail) = (ms(lo, first_dead), ms(last_dead + 1, hi + 1));
                // S159ze —— 第二条门:**按代价**(见 [`TRIM_MIN_COST_DEFAULT`])。
                // ⛔ OR,不是替换:它只**加刀**,永远不撤掉今天在裁的任何一刀。
                // ⛔⛔ `bar.is_finite()` 不是保险丝,它是**语义**:`+inf` 的意思是
                // 「**这一侧永不裁**」(见 [`TRIM_HEAD_MS`] 的 doc 与 `parse_trim` 对 `inf` 的处理)。
                // 少了它,按代价的支路会**绕过那个显式关闭** —— 判据
                // `the_landing_rule_never_changes_WHICH_notes_get_rescued` 当场把它抓了出来
                // (那条用 `(INFINITY, 500.0)` 当「只裁尾」的臂,span 从 (0,4) 被改成 (2,4))。
                let worth = |freed: f32, bar: f32| {
                    bar.is_finite()
                        && (freed >= bar
                            || freed * whole_shift.unsigned_abs() as f32 >= TRIM_MIN_COST_DEFAULT)
                };
                if worth(freed_head, head_ms) && guard_ok(first_dead.checked_sub(1), whole_shift) {
                    a = first_dead;
                }
                if worth(freed_tail, tail_ms) && guard_ok(Some(last_dead + 1), whole_shift) {
                    b = last_dead;
                }
                if (a, b) != (lo, hi) {
                    tracing::info!(
                        "range: phrase notes[{lo}..={hi}] rescued as [{a}..={b}] — dropped {:.2}s of \
                         passengers at the head, {:.2}s at the tail",
                        if a > lo { freed_head } else { 0.0 } / 1000.0,
                        if b < hi { freed_tail } else { 0.0 } / 1000.0,
                    );
                }
            }
            // ⛔⛔ S159z —— 条件里的 `n_clusters == 1` 不是装饰:原来写的是「只有裁剪动过
            // 边界才重求落点」,于是一个**本来就等于自己死音段**的簇(`(a, b) == (lo, hi)`)
            // 会直接继承**整句**的 `whole_shift` —— 正好把这一刀的意义抵消掉。
            // 判据 `an_interior_run_of_three_singable_notes_splits_the_phrase` 当场读到
            // 「−12 vs −12」把它抓了出来。
            // ⇒ 只有「一整句就是一簇、而且没裁过」才走继承那条路(那条路与 S159z 之前逐位相同)。
            let shift = if n_clusters == 1 && (a, b) == (lo, hi) {
                whole_shift
            } else {
                // The landing is re-solved against the notes that ACTUALLY ride along — the
                // dropped ones no longer constrain it. Measured across the four installed records
                // × 炉心融解 (and its +7 stress case): this moves **no** group. Audited rather
                // than assumed, because a moved landing is a change nobody has heard.
                let kept: Vec<i64> = (a..=b).map(|k| eff(note_nums[k])).collect();
                let s = minimal_rescue_shift_capped(&dead, &kept, range, tune.landing, tune.cap)
                    .unwrap_or(whole_shift);
                if s != whole_shift {
                    tracing::info!(
                        "range: notes[{a}..={b}] landing moved {whole_shift:+} → {s:+} st once its \
                         passengers were dropped"
                    );
                }
                s
            };
            if n_clusters > 1 {
                tracing::info!(
                    "range: phrase notes[{i}..={j}] split into {n_clusters} groups — this one is                      [{a}..={b}] at {shift:+} st (the whole phrase would have been {whole_shift:+})"
                );
            }
            out.push(DeadGroup { start: a, end: b, shift });
            }
        }
        i = j + 1;
    }
    (out, unfixable)
}

/// S159z —— **句内拆组**:一段夹在两个死音【之间】的可唱音,至少这么多个音才值得把组拆开。
///
/// ## ⛔ 为什么需要它:裁剪只卸得掉两头的乘客,卸不掉夹心
///
/// [`dead_only_plan_with`] 按休止分乐句,整句取一个位移(`worst()` 对「死音 ∪ 乘客」取 max)。
/// S151 的卸乘客把**两头**的乘客切掉了,但一段可唱音如果**夹在两个死音中间**,无论怎么裁都跟着走。
/// 用户 2026-08-21 点名的那一处正是这个形状(炉心融解 +7 × yachiyo,`notes[685..=693]`,位移 −15):
///
/// | 音 | 原谱 | +7 后 | 它自己需要 | 实际被拖 | 白丢的高频 |
/// |---|---|---|---|---|---|
/// | `[685]` | 83 | 90 | 15 | 15 | 0 dB |
/// | `[686]` | 73 | 80 | 5 | 15 | 13.1 dB |
/// | `[687]` | 71 | 78 | 3 | 15 | 15.7 dB |
/// | **`[688]`** | **68** | **75** | **0** | **15** | **19.7 dB** |
/// | **`[689..692]`** | **61-64** | **68-71** | **0** | **15** | **19.7 dB** |
/// | `[693]` | 73 | 80 | 5 | 15 | 13.1 dB |
///
/// ⇒ 组里**只有第一个音真的需要救**,后面五个音本来就在音域内,却陪着下潜 15 个半音。
/// 用户听到的是「ぴゃ 中间整个糊成一团」。
///
/// ## ⭐⭐⭐ 那 19.7 dB 是量出来的,不是估的(S159z 的三方对照)
///
/// 同一批音、同一个模型、同一个音高,唯一差别是走没走扩展(`native` / `donor pre` / `donor post`;
/// 离线台子在没被碰的素材上**逐位相同** ⇒ 噪声地板 = 0):
///
/// | 频段 | 总成本 | 其中 PSOLA | 其中 donor 自己 |
/// |---|---|---|---|
/// | 5-8 kHz | −13.32 dB | −2.84 | **−10.48** |
/// | 8-16 kHz | −12.51 | −1.93 | **−10.58** |
/// | 60-150 Hz | +19.5 | +4.07 | **+15.41** |
///
/// 而「donor 自己」那一栏又被**整曲移调 −8 的原生渲染**钉死:`donor − low8 = **+0.00 dB**`
/// (每一档都是零)⇒ donor 渲染路径没有损伤任何东西(只渲相交 chunk 那套优化是严格恒等的),
/// **那 10.5 dB 完全是「模型低唱 8 度」的内禀代价**。
/// ⇒ **定价:每往低渲 1 个半音 ≈ 高频 −1.31 dB + 次基频 +2.93 dB。**
///
/// ## 为什么门限是 3 个音,而且不是品味
///
/// 拆开会**新增 2 条边**,同时把该段自己的 `n−1` 个音符交界**移出 donor** ——
/// 而 S159g 量到的那种 ~40 ms、−2…−4 dB、谱心塌 20-30% 的塌陷,正是发生在 donor 内部的交界上。
/// ⇒ 判据 = 「移出去的交界 ≥ 新增的交界」⇔ `n − 1 ≥ 2` ⇔ **n ≥ 3**。实测(HF 按 1.31 dB/半音折算):
///
/// | 门限 | yachiyo +7 | akiko +7 | yachiyo 原 key |
/// |---|---|---|---|
/// | 音数 ≥2 | 20 处 / 189 dB·s / 交界 **−24** | 16 / 107 / −17 | 13 / 47 / **+7** |
/// | **音数 ≥3** | **12 处 / 153 dB·s / −32** | **11 / 86 / −22** | **6 / 25 / ±0** |
/// | 音数 ≥4 | 10 / 131 / −32 | 8 / 81 / −22 | 0 / 0 / ±0 |
///
/// ⇒ ≥3 在三种配置上**两条轴同时改善**(≥2 会让原 key 的交界变多,≥4 白白少收 22 dB·s)。
/// ⭐ 顺带解释了「原 key 干净、+7 炸」:陪绑量随移调量放大(原 key 25 dB·s vs +7 153)。
///
/// ## ⛔⛔ 它翻掉了一条已登记的决定,原话抄在这里
///
/// [`parse_trim`] 的 doc 里写着:「⇒ **下一步是【缝处的局部电平匹配】,不是调门限、
/// 不是换裁哪一侧,更不是先去拆句。**」—— 那句话是在**深度还没有价格**的时候写的:
/// 当时秤上只有「拆句新增的电平缝 p50 3.023 dB」,看不见「不拆要白丢 19.7 dB 高频、
/// 而且 donor 里的交界反而**多 32 条**」。⇒ 证据变了,结论跟着变;
/// ⚠ **那条「缝处局部电平匹配」仍然是对的、仍然欠着**(根因 = base 与 donor 只共用一个
/// 【全曲】归一标量),它和这一刀不冲突 —— 做完它,这一刀只会更划算。
///
/// ⛔ **别在这里加「按接点音程决定拆不拆」的规矩**:S159f 在裁剪上加过一条同族的,
/// 用户实机一听更糟(见 [`dead_only_plan_with`] 里那段 ⛔⛔⛔)。机理是一样的 ——
/// 拦住拆分 = 把更多材料留在 donor 里,**方向是反的**。
const SPLIT_MIN_INTERIOR_NOTES: usize = 3;

/// ⚙ 出厂默认 = 3000.0 —— 拆簇的**第二级**门:按**深度差**而不是按「唱不唱得动」(ms·半音)
///
/// ## ⛔ 一级拆看不见的那一半
///
/// [`SPLIT_MIN_INTERIOR_NOTES`] 断在「两个死音之间夹着 ≥3 个**可唱音**」的地方。
/// ⇒ 一整串**全是死音**、但各自需要的深度差着十几度的音,它**结构上看不见**。
///
/// 用户 2026-08-22 报的不为人所知的鹅妈妈童谣 +7 × 东雪莲(`usable = [36, 79]`)3:16.060-3:16.461
/// 就是这个形状 —— ⚠ 而且我一开始把坐标读成了前一个音,拿五把尺子量了一个**干净的音**:
///
/// | 音 | +7 后 | 自己需要 | 同簇被拖 | 白丢 |
/// |---|---|---|---|---|
/// | `[1033]う` | 92 | **−13** | −14 | 1 |
/// | `[1034]た` | 80(高出 usable 顶 **1** 度)| **−1** | **−14** | **13** |
/// | `[1035]で` | 80 | **−1** | **−14** | **13** |
///
/// 三个**都是死音** ⇒ 一级不断 ⇒ `worst()` 取 max ⇒ 后两个多渲低 13 度。
/// 按 S159z 的定价(每半音 ≈ 高频 −1.31 dB / 次基频 +2.93 dB)那是 **高频 −17 dB**。
/// 实测对上了:`[1034]` 的**紧贴谐波裙边**(±0.06…0.15·f0 相对谐波峰)= **−6.4 dB**,
/// 是全曲 MIDI 80 / −14 那一档 19 个音里**最脏的一个**(该档中位 −17.0)。
///
/// ## 判据:`Σ 时长 × (簇深 − 自己深)`,单位 ms·半音,门 = **3000(量出来的,见下表)**
///
/// ⛔ **只算两侧、不算夹心。**夹心(被断开之后整段留在 base)省下的其实更多,但那是
/// [`SPLIT_MIN_INTERIOR_NOTES`] 那一刀的账;两边都算会让同一份收益被记两次,而
/// **这是一个门不是一本账**(同 [`TRIM_MIN_COST_DEFAULT`] 的措辞,理由也一样)。
///
/// ⛔ 断点只许落在**两个死音之间** ⇒ 每一簇的两端仍然是死音,`(first_dead, last_dead)`
/// 与 `lo`/`hi` 那两行的不变量一个字不用改。
///
/// ⚠ 收益用「子串自己重解出来的位移」估,而**最终**位移要等裁剪定完 `a..b` 之后才算得出来
/// (见下面那条 `let shift = if n_clusters == 1 …`)。⇒ 这个估计**只会高估**(裁剪只会让
/// |s| 更浅),和 [`TRIM_MIN_COST_DEFAULT`] 一样往「少拆」那侧偏。
///
/// ## ⛔ 与原 key 的关系:**这一条我没有构造性论证,只有实测**
///
/// [`TRIM_MIN_COST_DEFAULT`] 那条能证明原 key 够不着(它的新支路要求 `d > 12`,而四份装机
/// 记录在原 key 上最深只有 −10)。**这一条不能**:它的门只要求「某一侧 ≥600 ms 且深度差
/// ≥10」,原 key 的 −10 理论上够得着。⇒ **改完必须逐份 dump 原 key 的计划对拍**,
/// 不许拿上一条的论证套过来。⚠ 这正是 S159z 血训「一条只在窄区间上采样的曲线,
/// 读出来的单调性是【那个区间】的性质」的同族:**别把一条论证的适用域悄悄扩大。**
///
/// ## ⭐⭐⭐ 3000 是**夹**出来的,不是拍的
///
/// ⛔ 我第一版拍了 6000(= 500 ms × 12,借 [`TRIM_MIN_COST_DEFAULT`] 的形状),而用户点名的
/// 那一处实测 `480 ms × 12 st = 5760` —— **差 4% 没够着**。⚠ 那是**第二次**:S159ze 卸乘客
/// 那条门是 500 ms 而乘客总长 **480.0 ms**,差整整一帧。
/// ⭐ **「用户报的病例总是差一点点够不着」是【门槛没有出处】的指纹,不是巧合。**
/// ⇒ 加了 [`RescueTuning::split_cost`],一次跑出整张表(`mg_dump_plan_arms` 的 `split_scan`,
/// 秒级、不碰 ONNX)。**8 份装机记录 × 6 张谱 × 移调 {0,2,5,7} = 192 份**,每份内部扫 10 档:
///
/// | 门 | t+0(原 key)| t+2 | t+5 | t+7 | 用户点名的 `[1034][1035]` |
/// |---|---|---|---|---|---|
/// | 12000 | 0/48 | 0/48 | 0/48 | 1/48 | −14(没够着)|
/// | 6000 | 0/48 | 0/48 | 3/48 | 7/48 | **−14 —— 没够着** |
/// | 4500 | 0/48 | 1/48 | 4/48 | 11/48 | dx/ak 脱出,**yachiyo 仍 −14** |
/// | **3000(出厂)** | **0/48** | 1/48 | 4/48 | 14/48 | **三份记录全部脱出** |
/// | 2000 | **1/48** | 2/48 | 6/48 | 15/48 | 脱出 |
///
/// (表里的分数 = 「计划与**不拆**那一臂不同」的份数。)
///
/// ⇒ **下界**(原 key 逐字节不变;东雪莲 × 鹅妈妈在 2000 上 23→29 组)= **≥ 3000**;
///    **上界**(必须打中用户点名的病例;yachiyo_v2 要到 3000 才开火)= **≤ 3000**。
///    ⇒ **只有 3000 同时满足。**
/// ⚠ 这不是「调参调到过」:两条边界来自**两个不同的判据**,而它们正好在这一点相交。
/// ⛔ 哪天任一装机记录变了,这个数**必须重扫** —— 别把它当常量供起来。
/// ⚠ t+2 上有 1/48 变了(东雪莲 × 鹅妈妈 54→60 组);那一处也是净赚
/// (陪绑 101.9 → 70.3 s·半音 = **−31%**,代价 12 条新缝)。
///
/// ## 代价:句内拆**必然**造缝,而这些缝几乎是免费的(实测,而且推翻了我自己的预期)
///
/// 东雪莲 +7,窗边落在唱音内的条数 67 →(门 6000)105 →(门 3000)127。⚠ 我一开始把这当成
/// 这一刀的主要代价。**实测把它推翻了**(谱形跳变 = 边两侧各 20 ms、500-8000 Hz 对数谱
/// 去均值 RMS 差):
///
/// | | 谱形跳变中位 | 电平台阶 dB 中位 |
/// |---|---|---|
/// | 拆组**前**就有的 67 条老缝 | 11.74 | 1.90 |
/// | **拆组新增的 38 条** | **9.77** | 1.87 |
/// | ⭐ **同样是音-音交界、但没有缝**(972 处,关键对照)| **9.03** | **2.90** |
/// | 音内随机位置(阴性对照,2082 处)| 8.19 | 0.89 |
///
/// ⇒ 新缝落在**音与音的交界**上(乐句内部没有休止可放),而那里本来就有一个 9.03 dB 的跳变
/// (S159zi 量到起音比稳态脏 ~11 dB,连没进扩展的音也一样)⇒ **缝只加了 0.74 dB,
/// 而且电平台阶反而比普通交界还小。**
/// ⛔ 别把这条读成「缝不要紧」:**老缝**比交界高 **2.71 dB**,而那笔账(「缝处局部电平匹配」,
/// 根因 = `apply_dead_only_windows_with` 的 `match_levels` 只给**每个位移一个全曲标量**)
/// **仍然欠着**。
///
/// ⭐ 造对照臂用字段而不是 `git stash`:S143 在 `stash` 的假信号上给自己的代码定过一次罪。
/// ⛔⛔ 也别用「改源码 + 重编译」的循环脚本扫参数 —— S159zi 那样干了一次,脚本**扛过了 kill**、
/// 在我跑测试与渲染的同时把这个常量来回改,最后还把整份源文件**还原**掉了。
/// 抓出它的是 `every_shipped_default_is_declared_at_the_top_of_its_own_doc`(doc↔常量一致性闸),
/// 而它当初瞄的根本不是这件事。
const SPLIT_MIN_COST_DEFAULT: f32 = 3000.0;

/// ⚙ 出厂默认 = 6000.0 —— 卸乘客的**第二条**门:按**代价**而不是按时长(单位 ms·半音)
///
/// ## ⛔ 为什么需要它:今天的判据里没有任何一项含【位移深度】
///
/// [`TRIM_HEAD_MS`] / [`TRIM_TAIL_MS`] 只问「回收多少**毫秒**」。⇒ **480 ms 被拖 3 度和被拖 17 度,
/// 在那行代码眼里一模一样。**
///
/// 用户 2026-08-22 报的那一处(不为人所知的鹅妈妈童谣 +7,3:16.060-3:16.461)正是这个形状:
///
/// | 音 | 原谱 | +7 后 | 自己需要 | 实际被拖 | 陪绑 |
/// |---|---|---|---|---|---|
/// | `[1033]う` | 85 | **92** | 12-16 | 14-17 | 0-2(**顶音,真需要**)|
/// | `[1034]た` | 73 | 80 | **0**(东雪莲)/ 3-4 | **14-17** | **10-14** |
/// | `[1035]で` | 73 | 80 | **0** / 3-4 | **14-17** | **10-14** |
///
/// 而尾部乘客总时长 = 140 + 340 = **480.0 ms 整**(7 + 17 = 24 帧 × 20 ms,不是浮点舍入),
/// 门是 **500.0** ⇒ **差整整一帧没触发**。⇒ 480 ms 的音白白多渲低 10-14 个半音,
/// 按 S159z 的定价(每半音 ≈ 高频 −1.31 dB)那是 **13-18 dB 的高频**。
///
/// ## 判据的形状:`freed_ms × |位移|`,单位 ms·半音
///
/// ⭐ 能塌成这么简单是因为**被裁掉的音自己需要 0 半音**:被裁的只可能落在 `lo..first_dead` 或
/// `last_dead+1..=hi`,而 `dead_at` 的定义保证这两段里**每个音都过了 [`SpeakerRange::slot_singable`]**。
/// ⇒ 「省下的陪绑」= `freed_ms × |whole_shift|`,不需要逐音求解。
///
/// ⛔⛔ **必须用 `whole_shift` 而不是重解之后的 `shift`** —— 后者要到边界定完之后才算得出来。
/// ⚠ 这会**高估**收益(重解只会让 |s| 变浅),而 `guard_ok` 也用 `whole_shift` ⇒ 两边都往安全侧偏。
/// **别把这个式子当精确,它是个门不是账。**
///
/// ## 为什么 6000,以及为什么原 key 逐位不变(**构造性论证,不是测量**)
///
/// 新支路要生效必须同时满足 `freed < 500`(否则老支路已经命中)与 `freed × d ≥ 6000`
/// ⇒ **`d > 12`**。而 [`parse_landing`] 的 doc 里记着四份装机记录在**原 key** 上的全曲最深位移:
/// akiko −9 · yachiyo −10 · 东雪莲 −7 · yuyuko −7 ⇒ **原 key 这条支路一次都够不着**
/// ⇒ 用户 S158f 已经耳判背书过的那条臂**逐位不变**(S121 的 additive-then-flip)。
/// ⭐ 6000 = 500 ms × 12 半音:**它就是「老门限在临界深度上的等价物」**,不是调出来的。
///
/// ⛔ **只许 OR,不许替换掉老门限。**纯替换的等效门限是 `T/d`,会随深度滑动,而 S151 那次
/// 平台读数(`(0.38, 1.08] s` 给同一刀集)是在**当时的刀集**上量的,S157c 的落点与 S159z 的拆组之后
/// 平台可能已经漂了 —— 那份 doc 自己也写着「⚠ +7 谱是 threshold-sensitive,Re-measure before moving」。
const TRIM_MIN_COST_DEFAULT: f32 = 6000.0;

/// ⚙ 出厂默认 = Some((TRIM_HEAD_MS, TRIM_TAIL_MS)) —— 卸乘客 = **头尾都裁 500 ms**(S158f;`=0` 渲旧臂)
/// S151 卸乘客 —— 一刀要**回收多少毫秒**的活音才值得它造出来的那条缝,`(裁头, 裁尾)`。
/// **S158 起这是出厂默认(只裁尾)**;`UTAI_RANGE_TRIM=0` 关(渲旧臂)· `=1` 用下面两个常量 ·
/// `=<head_ms>:<tail_ms>` 扫参数。
///
/// **Why this knife exists at all.** S148 measured the toll: a passenger pays it for *entering*
/// the process, not for how deep it goes — `corr(Δrms, |shift|) = −0.172` across 154 passengers,
/// while merely passing through PSOLA costs p50 0.7-2.0 dB against a 0.001 floor. S150 then found
/// a second, independent bill on the vocoder side, and that one IS dose-shaped: the non-harmonic
/// 4-11 kHz fraction of in-range notes goes 24.0 % (untouched) → 29.2 % (−3) → 39.2 % (−7).
/// ⇒ the only two ways to cut the bill are "let fewer notes in" (this) and "make the toll
/// smaller" (S150's phase lock). The user's verdict on the direction: 「乘客卸载我觉得可以先做了」.
///
/// **Why it is conditional rather than always-on.** The blind test that endorsed the aggressive
/// version (S148 r3, arm D = cut every group to its dead run) only came back clearly better on
/// the ONE segment where the recovery was large (three notes getting back 5.36/5.33/3.70 dB); the
/// listener called the other two 「区别不是特别大」. And every cut buys its material with a new
/// seam in the middle of a phrase, which is not free: Δripple at a cut **head** is p50 0.258 /
/// p90 1.452 dB against a 0.060 floor, at a cut **tail** 0.033 / 0.407 — an **8×** asymmetry.
///
/// ⚠⚠ **S158 更正了这段的两处**(实测,见 `TESTING\s158_knives\读数_渲染之后.md`):
/// ⑴ 那个 8× 是在**没有 `GUARD_FRAMES` 的 S148 二进制**上量的,而且样本被污染
///    (21 头 / 13 尾里有 5 个是同一个音,被裁成单音组两侧同时开缝)。
/// ⑵ ⛔ **「因为裁头把 10 ms 淡入压在音的【起音】上」这句归因解释不了今天的读数。**
///    今天护栏在 22/22 个头刀与 17/17 个尾刀上**都拿满 2 帧 = 40 ms**(一次都没有被
///    前后邻音的时长封顶)⇒ 淡入结构上落在**被卸掉的那个乘客**的最后 40 ms 里,
///    离被救音的起音还有 30 ms —— 而代价**原样还在**:出厂二进制上、被暴露的那个
///    被救音的 |Δripple| p50 **头 0.267 / 尾 0.005**(同处地板 0.004 / 0.006),
///    最坏两条头缝 `[543]け 2.312` · `[156]で 2.004` dB,已经顶到这条轴唯一的
///    可闻刻度门口(S148 u1:~2.7 dB 听得出 / ≤0.46 dB 听不出)。
/// ⇒ ⭐⭐⭐ **S158b 找到了真原因,写在 [`parse_trim`] 的 doc 里:base 与 donor 只共用一个
///    【全曲】归一标量,所以任何**局部**都可以差几 dB;窗边落在休止里时没有代价,
///    而卸乘客/拆句做的事就是把窗边搬到乐句中间。头缝与尾缝**都**是几 dB 的电平台阶
///    + 谱跳变(原 key 尾缝台阶 p50 **3.023 dB**,地板 0.005)。
///    ⇒ **下一步是【缝处的局部电平匹配】,不是调门限、不是换裁哪一侧,更不是先去拆句。**
///
/// **The numbers.** Offline sweep over the four installed records × 炉心融解 (the user's own
/// project, 803 triples), reproduced from the production plans in the 2026-08-18 log to the group
/// (`TESTING\s151_knives\gate_fidelity.py`). Recoverable material per cut on akiko is bimodal —
/// {0.18, 0.36, 0.38} then {1.08, 1.10, 1.28, 1.44, 1.46, 1.82, 2.56, 3.44, 4.36} seconds — so
/// **every threshold in (0.38, 1.08] picks exactly the same 10 head + 7 tail cuts**, and these
/// two values sit inside that plateau rather than on its edge. What it buys on akiko: 154 → 46
/// passengers if both sides always cut (33.0 s of passenger audio), 28.6 s of that at these
/// thresholds, for 17 new seams instead of 34. ⭐ The one cut the user has already endorsed by
/// ear (S148 r1, `notes[753..=762]`, 3/3) frees **1.82 s** and is comfortably inside.
/// ⚠ Not a plateau everywhere: on the +7 stress case the same records give 14 head / 13 tail at
/// 750 ms and 5 / 9 at 1000 ms, i.e. that song IS threshold-sensitive. Re-measure before moving.
///
/// ⛔⛔⛔ **S158 在这个默认上翻了三次(开 → 撤 → 开)。三次的理由都留在这里,因为**
/// **中间那次撤销正好是这条线上最典型的一种错误,而它差点把一把好刀关掉。**
///
/// ## 翻的时候我拿的是什么(每个数都是真的,而且都不相干)
///
/// 整曲四条臂(+7 压力谱,同一二进制,`render_guard` 量过身份),`note_delta.py` **逐音**配对:
/// 被卸掉的 60 个音 |Δrms| p50 **0.667** dB(同批地板 0.004),而**其余五类全部落在渲染地板里**
/// —— 包括我当时叫做「缝旁边那个被救音」的那 16 个(|Δripple| p50 **0.005**,地板 0.006)。
/// 于是我写下「它新造出来的唯一一样东西(缝)量不出来」。
///
/// ## ⛔ 那句话是错的:**缝量不出来,是因为那把尺子结构上看不见它**
///
/// 用户当场指出该看的不是音、是**「扩展段↔未扩展段」那一次切换**,而且原 key 上更暴露。
/// 换成**边界尺子**(`TESTING\s158_knives\seam158.py`:同一时间位置、两条臂相减,
/// 跳过 10 ms 淡化本身,窗 40 ms)之后,同一批边读出来是:
///
/// | 只裁尾造出的句内新边 | n | 电平台阶 p50/p90 | 24 带谱跳变 p50/p90 | 同位置地板 |
/// |---|---|---|---|---|
/// | 原 key | 7 | **3.023 / 3.609 dB** | 7.863 / 12.429 dB | 0.005 / 0.020 |
/// | +7 压力谱 | 16 | **0.929 / 4.810 dB** | 6.515 / 7.858 dB | 0.005 / 0.029 |
///
/// 阴性对照两条都读地板(没动过的休止边 0.034;被救音内部 0.028),阳性对照(头缝)读得出来
/// ⇒ **尺子不瞎,也不是在量「任何边」。**
/// ⭐⭐⭐ **而 +7 上它本来就在** —— 我只是没量到:一个 180 ms 的音,末尾 40 ms 有 3 dB 台阶,
/// 整个音的 ripple 统计几乎不动。**「读数在地板上」与「这把尺子看不见它」在输出上一模一样。**
/// (与 S148 那条同族:「不实」是音**内部**的形状,整音统计量结构上看不见它。)
///
/// ## ⭐⭐⭐ 顺带定死了「缝为什么贵」——不是淡化落在起音上
///
/// 两份谱、头缝与尾缝**都**是几 dB 的**电平台阶 + 谱跳变**,而 `GUARD_FRAMES` 把淡化挪到
/// 乘客身上并不改变它。真机理是:**base 与 donor 只共用一个【全曲】归一标量**
/// (S147 笔1),所以它们在任何**局部**都可以差几 dB。窗边落在休止里时这没有代价
/// (两侧都是静音);**卸乘客/拆句做的事,恰恰就是把窗边从休止里搬到乐句中间。**
/// ⇒ 普查实测:翻默认之前,**两份谱上每一条窗边都落在休止里**(原 key 50/50、+7 126/126),
/// 一次真正的切换都没有;只裁尾把它变成 7 条(原 key)/ 16 条(+7)。
/// ⇒ ⭐ **这条线的下一步是【缝处的局部电平匹配】,不是调门限、也不是换裁哪一侧。**
/// ⚠ S147 笔1 判死过三种「自然」的做法(窗内对齐 / 保留区对齐 / donor 不归一),但那三种
/// 对齐的是**整条窗**;这里要的是**只在缝那几十毫秒上**对齐,是另一件事,没人试过。
///
/// ## ⛔⛔⛔ 然后用户听了,而我那个「代价」是**自己贴上去的符号**
///
/// 用户听完原 key 那四条臂:「**卸掉乘客的那个版本我没听出来什么特别奇怪的东西,反倒是低音那边
/// 因为被卸了反而还能更自然一点**」。⇒ 我把那 3 dB 叫做「代价」,而 S148 血训第 55 条逐字写着
/// **「Δ 量的是【变化】不是【好坏】,给它贴符号要另有依据」** —— 我没有那个依据。
///
/// 结构上一看就明白,而我当时没看:**那 7 条边【全部落在音符边界上】**
/// (`[180]み→[181]い` 这一类),而音符边界本来就该有电平与音色的变化;
/// 而且换过去的那一侧是 **base = 那个音本来正确的渲染**。
/// ⇒ 「跨音符边界的电平关系变了 3 dB」**不是接缝瑕疵**,那就是唱歌。
///
/// ## ⭐⭐⭐ 而收益那一侧,我用的栏目**结构上分不出好坏**
///
/// 我第一轮那份分析脚本报的是 **|Δ| 绝对值**(p50/p90)。绝对值把方向抹掉了 ⇒
/// 「被卸掉的音动了 100 倍地板」这句话里,**没有一个字说的是【往哪边动】**。
/// 换成**有符号中位数**、并换到这条线上唯一被盲测背书过、且与耳朵同向的那条轴
/// (`low_ratio`,S148 r1),原 key 上:
///
/// | 类 | n | Δlow_ratio 中位(有符号) | 同批地板 |
/// |---|---|---|---|
/// | **被卸掉的** | 26 | **−0.1254** | +0.0000 |
/// | 仍在车上的 | 213 | −0.0000 | −0.0000 |
/// | 从没被碰过(阴性) | 467 | +0.0000 | +0.0000 |
///
/// 逐音:`[231]た 0.445→0.098` · `[442]と 0.565→0.249` · `[686]か 0.584→0.272` ·
/// `[693]に 0.973→0.728`。⭐ **S148 那次盲测里用户听到「破音」的那一格正是 `low_ratio`
/// 0.388 → 0.628** ⇒ 这批音是在**反方向跨过同一条带,而且跨得更多**。
/// ⇒ **耳朵与这条线上最有背书的那把尺子指同一个方向**,而它们都不同意我那条「缝的代价」。
///
/// ⇒ ⭐ **默认 = 只裁尾**(S158d 翻回来)。旋钮 `UTAI_RANGE_TRIM=0` 仍能渲旧臂。
/// ⛔ **头裁仍然关**:S151 盲测判负,而 S158 的边界尺子在两份谱上也读出它更贵
/// (原 key 头缝台阶 p50 1.019 / p90 7.257,尾缝 3.023 / 3.609 —— p90 差 2 倍)。
///
/// ## ⚠ 那把边界尺子没有作废,但它的**结论口径**要写清楚
///
/// `TESTING\s158_knives\seam158.py` 量到的是**真的**:这一刀确实把窗边从休止搬进乐句中间,
/// 并在那里留下几 dB 的电平/谱台阶(而 base 与 donor 只共用一个**全曲**归一标量,
/// S147 笔1 ⇒ 局部本来就能差几 dB)。它**不能**回答那一下听不听得出来。
/// ⭐⭐ 而这一轮顺带给了这条轴**第一个可闻性刻度**:
/// **落在音符边界上的 3.0 dB 电平台阶 + 7.9 dB 谱跳变:用户仔细找**能在频谱上看到它**,
/// 但听感上「不奇怪到可以忽略」。**⇒ 刻度是「可忽略」,不是「不存在」——别把它记成后者。
/// ⇒ 将来做「拆句」时,这个刻度是可以用的 —— 但要注意拆句的边**不一定落在音符边界上**。
/// The env parse as a pure function, so it can be asserted without touching process state
/// (reading the real environment in a test both races the other tests and passes SILENTLY on a
/// machine where someone exported the variable — S150 paid for that lesson on `parse_phase_lock`).
fn parse_trim(v: Option<&str>) -> Option<(f32, f32)> {
    // ⛔⛔ S159ze —— `+inf` **必须放行**,因为 [`TRIM_HEAD_MS`] 的 doc 就是这么教人关掉单侧的
    // (「想关回去:`UTAI_RANGE_TRIM=inf:500`」)。而 `is_finite()` 会把它判成垃圾 ⇒ 落到下面那条
    // `_ => TRIM_DEFAULT` ⇒ **拿到的是头裁【开着】的出厂臂**,而且**没有任何一行输出会说破**。
    //
    // ⇒ 这正是本文件反复警告的那一族:「臂开着」与「臂做了事」是两件事
    // (S148 的 `frac_transport`、S155 的 `parse_infrasonic_hp` 都在这上面栽过)。
    // ⚠ 而它比那两次更隐蔽:那两次是**写死成 false**,这次是**doc 教的用法本身就无效** ——
    //   照着 doc 渲出来的「关掉头裁」对照臂,其实是出厂臂。**凡引用过 `inf:` 臂的读数一律作废。**
    //
    // 语义上 `+inf` 正好就是「这一侧永不裁」:门是 `freed_ms >= head_ms`,而 `freed >= inf` 恒假。
    // ⇒ 放行它不需要任何额外分支。⛔ 但 `-inf` 与 `NaN` 仍然是垃圾(前者会让门恒真 = 静默全裁)。
    let ok = |x: f32| (x.is_finite() || x == f32::INFINITY) && x >= 0.0;
    match v.map(str::trim) {
        None | Some("") => TRIM_DEFAULT,
        Some("0") => None,
        Some("1") => Some((TRIM_HEAD_MS, TRIM_TAIL_MS)),
        Some(s) => {
            let mut it = s.split(':');
            match (it.next().map(str::trim), it.next().map(str::trim), it.next()) {
                (Some(h), Some(t), None) => match (h.parse::<f32>(), t.parse::<f32>()) {
                    (Ok(h), Ok(t)) if ok(h) && ok(t) => Some((h, t)),
                    _ => TRIM_DEFAULT,
                },
                _ => TRIM_DEFAULT,
            }
        }
    }
}

/// ⚙ 出厂默认 = Some(3) —— 落点可以往深里多看 3 个半音(S157c 翻)
/// S151 —— `UTAI_RANGE_LANDING=<semitones>`:落点排序**可以往深里多看几个半音**。
/// 缺省 = [`LANDING_MAX_EXTRA_DEPTH`] = 今天那条臂。
///
/// 存在的理由在 `minimal_rescue_shift` 里 `LANDING_THIN_EPS` 那段注释(两把独立的尺子同时
/// 指认「落点音高」是这条线上的主杠杆)。⛔ 它**不是**放开预算:排序仍然先过 damage,
/// 预算仍然由这个数封顶 —— S148 记着无上限时东雪莲会一路走到 **−24**,那是用户耳判过的灾难。
/// ⚠ 只接**谱面路**;cover/audition 那一路仍然用常数(它的素材与判据是另一套)。
fn parse_landing(v: Option<&str>) -> Option<i64> {
    match v.map(str::trim) {
        None | Some("") => LANDING_DEFAULT,
        Some("0") => None, // 显式关得掉 —— 抱怨某条臂时要能用同一个二进制渲旧臂
        Some(x) => match x.parse::<i64>() {
            Ok(n) if (1..=MAX_RANGE_SHIFT).contains(&n) => Some(n),
            _ => LANDING_DEFAULT,
        },
    }
}

/// ⛔ 与 [`TRIM_DEFAULT`] 同理:仪器不许给它翻默认,盲测过了才翻,翻的时候
/// `RANGE_ALGO_VERSION` ↔ `audition_cache_tag` 必须成对 bump。
///
/// **翻的时候用 3**,理由是离线扫的(四份装机记录 × 炉心融解,`TESTING\s151_knives`,
/// 计划已过实机忠实度闸)——「死音落在 77-78 的比例 / 全曲最深位移」:
///
/// | 记录 | 今天 | +2 | **+3** | +4 |
/// |---|---|---|---|---|
/// | akiko(用户的模型) | 51 % / −7 | 48 % / −8 | **11 % / −9** | 11 % / −9 |
/// | akiko 谱面 +7 | 30 % / −14 | 26 % / −15 | **12 % / −16** | 12 % / −16 |
/// | yachiyo(RVC) | 59 % / −7 | 0 % / −9 | 0 % / −10 | 0 % / −10 |
/// | 东雪莲 | 40 % / −7 | 40 % / −7 | 40 % / −7 | 4 % / −10 |
/// | yuyuko(RVC) | 53 % / −7 | 53 % / −7 | 53 % / −7 | 6 % / −10 |
///
/// ⚠ 两条要一起读:⑴ **东雪莲在这条轴上救不了** —— 它的落点 `low_ratio` p50 是 **0.865**,
/// 上面那一片全是薄的,所以这一刀在它身上只会换来更深而换不到更干净;
/// ⑵ 深度的代价是**真的**(S150:乘客的 4-11 kHz 非谐波 24.0 → 29.2 → 39.2 % @ 0/−3/−7)
/// ⇒ 这一刀必须和**卸乘客**一起看,不能单独放大预算。
const LANDING_DEFAULT: Option<i64> = Some(3);

/// ⛔⛔ **S158 翻成「只裁尾」,当天又撤回 `None`** —— 完整的账在 [`parse_trim`] 的 doc 里:
/// 翻的时候用的是**逐音**尺子(它结构上看不见缝),换成**边界**尺子之后,这一刀造出来的
/// 句内新边在原 key 上是 **3.023 dB 的电平台阶**(同位置地板 0.005)。
/// ⇒ 前置变成「**缝处的局部电平匹配**」,做完再谈翻默认。
/// 翻它必须成对 bump `RANGE_ALGO_VERSION` ↔ `audition_cache_tag`
/// (S150:漏掉一个不是错误,是用户听到一条陈缓存)。⚠ S158 撤回时把标签也退回 `s157a` ——
/// **标签跟的是音频不是场次**,撤完之后输出与 s157a 逐位相同。
/// ⛔ 开着它渲:`UTAI_RANGE_TRIM=1`,同一个二进制。
/// ⚠ 只影响**谱面轨**:`cover_dead_plan`(音频轨/audition)是另一份分组逻辑,里面没有裁剪。
const TRIM_DEFAULT: Option<(f32, f32)> = Some((TRIM_HEAD_MS, TRIM_TAIL_MS));
/// ⭐ **S158f 起头裁也开着(500 ms)。**这个值在 S151-S158e 之间一直是 `f32::INFINITY`
/// (= 关),下面是它被关掉、又被打开的完整理由 —— 两边都留着,因为它们**都是真的**。
///
/// ## 关掉它的那次(S151 r1,5 组 × 2 文件,`level_match: none`,两个对照都答对)
/// * `[685..=693]` **尾裁**,放掉 2.56 s ⇒ 用户选**改动版** ✅
/// * `[796..=802]` **头裁**,放掉 4.36 s(全曲最大的一刀)⇒ 用户选**今天** ❌
/// * `[612..=624]` **头裁**,放掉 3.44 s ⇒ 「区别不是很大」⚪
/// ⚠ **每侧只有 1 个数据点**,纯掷硬币 p = 0.5 —— 当时就写着「这是机理 + 一个同向数据点,
/// 不是判决」。而它当时靠的那条机理(「8 倍,因为 10 ms 淡入压在**起音**上」)
/// **已经被 S158b 证伪**:护栏把淡入挪到被卸乘客身上并不改变代价,见 [`parse_trim`]。
///
/// ## 打开它的那次(S158f,用户听完 `TESTING\s158_knivesu_s158k0_头尾都裁.wav`)
/// > 「我是没听出来什么区别来,或者至少『头尾都裁』应该是没让它变坏,
/// >   而换来的更多能卸掉的那应该是收益」
/// 边界尺子在**原 key** 上读到的也同向:头边电平台阶 p50 **1.019** dB,而尾边是 **3.023**
/// ——**中位数上头边反而更便宜**,只有最坏那几条更贵(p90 7.257 vs 3.609)。
/// 账:原 key 上头乘客是最大的一块(**76 个 / 20.36 s**),而只裁尾只够得到 26 个。
///
/// ⚠ 仍然登记着的风险:p90 那几条(−8/−9/−7 的深位移头刀)只过了一次耳朵,没有单独承重。
/// 想关回去:`UTAI_RANGE_TRIM=inf:500` 或直接 `=0`。
const TRIM_HEAD_MS: f32 = 500.0;
const TRIM_TAIL_MS: f32 = 500.0;

/// THE single landing search for both tracks (score phrases / cover regions): the minimal |s|
/// that lands every DEAD pitch on a landing-grade slot while every dragged pitch stays
/// singable. Candidate order: single-sided dead searches its own direction by growing |s|;
/// INTERIOR dead (a bridged-weak slot inside usable — a legal record form the write side
/// produces on purpose, rangeTest.ts longestRun) has no inherent direction and tries both,
/// down first at each magnitude. Dead on both sides is untranslatable ⇒ None.
fn minimal_rescue_shift(
    dead: &[i64],
    all: &[i64],
    range: &SpeakerRange,
    landing: Option<i64>,
) -> Option<i64> {
    minimal_rescue_shift_capped(dead, all, range, landing, LANDING_RATIO_TWO_ST)
}

/// S159 —— [`minimal_rescue_shift`],但深度上限由参数给。⛔ 生产永远传 [`LANDING_RATIO_TWO_ST`];
/// 这条门存在的唯一理由是让计划台一次跑出「上限 × 预算」那张二维表(见 `RescueTuning::cap`)。
fn minimal_rescue_shift_capped(
    dead: &[i64],
    all: &[i64],
    range: &SpeakerRange,
    landing: Option<i64>,
    ratio_two_cap: i64,
) -> Option<i64> {
    let above = dead.iter().any(|&p| p as f32 > range.usable.1);
    let below = dead.iter().any(|&p| (p as f32) < range.usable.0);
    let candidates: Vec<i64> = if above && below {
        Vec::new()
    } else if above {
        (1..=MAX_RANGE_SHIFT).map(|m| -m).collect()
    } else if below {
        (1..=MAX_RANGE_SHIFT).collect()
    } else {
        (1..=MAX_RANGE_SHIFT).flat_map(|m| [-m, m]).collect()
    };
    let qualifying: Vec<i64> = candidates
        .into_iter()
        .filter(|&s| {
            // ⛔ `slot_reachable`, NOT `slot_singable`: this asks whether the MODEL can voice the
            // moved phrase, so it must read the scan's bounds — never the user's rescue line
            // (S146f; using `slot_singable` here made every landing walk deeper as the user
            // lowered the ceiling, which is the regression they reported by ear).
            dead.iter().all(|&p| range.slot_landing_ok(p + s))
                && all.iter().all(|&p| range.slot_reachable(p + s))
        })
        .collect();
    // Bounds-only record (no raw scan) ⇒ there is nothing to rank by; keep the historical
    // "shallowest that qualifies" verbatim.
    if range.damage.is_none() {
        return qualifying.into_iter().next();
    }
    // S146c: among the shifts that QUALIFY, land where the record says the model is actually
    // fine — not on the first slot that merely passes the gate.
    //
    // Why this changed: LANDING is a binary gate (err ≤ 50¢ ∧ voiced ≥ 0.9), so "shallowest that
    // qualifies" lands, by construction, on the WORST slot that still passes. Measured on the
    // user's own model and phrase (akiko, notes[186..=191] = [75,76,85,83,81,80]): damage is
    // 0.000 flat across 66-78, 0.592 at 79, saturated 3.000 at 80+. Today's −6 puts the top dead
    // note on **79** — the last slot before the cliff — and the render came back voiced 0.17
    // there. One semitone deeper lands the whole group at damage 0.000.
    //
    // The trade this reverses is explicit: depth was minimised because every extra semitone cost
    // audible chipmunk (S85). S146 replaced the inverse engine with TD-PSOLA and that cost fell
    // from 2.00 semitones of formant leak to 0.30 — so "as shallow as possible" is no longer the
    // right objective. ⛔ It is NOT "always deeper" either: the record decides, and S145 measured
    // 东雪莲's slot 78 getting 3.8 dB WORSE one semitone down.
    //
    // ⚠ This cannot rescue an extra note: `qualifying` is computed with the untouched predicates,
    // so the dead set and the set of rescued groups are exactly what they were.
    let worst = |s: i64| -> f32 {
        dead.iter()
            .chain(all.iter())
            .map(|&p| range.damage_at((p + s) as f32).unwrap_or(DAMAGE_MAX))
            .fold(0.0f32, f32::max)
    };
    // ⛔ The ranking is bounded to a neighbourhood of the shallowest qualifying shift. Without
    // this bound the rule walks as deep as the damage curve rewards, and on a model whose curve
    // is poor through its MIDDLE register that is very deep indeed: measured on
    // Sovits4.1东雪莲主模型 (15 slots at low_ratio > 0.7 scattered over 53-79) the unbounded rule
    // picks −10..−24 where it used to pick −6/−4 — and a −24 whole-song recolour is the exact
    // outcome the user identified by ear as a catastrophe (S85b, dev log A/B 23:24 vs 23:44).
    // S146f — THE TARGET RANGE (`comfort` on disk; 「目标范围」 in the UI). Rescued notes land
    // inside it, and the search spends whatever depth that costs.
    //
    // ⭐ 用户 2026-08-15 拍板的语义,而且是简化掉一层启发式换来的:「识别出来的范围就是
    // 【还原】的目标,然后如果手动调过了就听手动的」。An earlier draft gated this on "did the
    // user actually move it", to protect against the editor's own clamp having dragged the value
    // down — but S146f 笔 3 removed the clamp that produced the artefact, so the gate was
    // maintaining a heuristic for a problem that can no longer occur. A heuristic is exactly the
    // thing that silently misjudges later; the stored value is now simply believed.
    //
    // ⛔ What must NOT be simplified away: it DEGRADES rather than vetoes. An unreachable band
    // falls through to the normal budget below, audited — 东雪莲 stores [36,52] against a phrase
    // living at 75-85, and a hard veto would take it from 10 groups rescued to 0, i.e. the knob
    // would be an off switch for range extension on that model.
    //
    // Sweep on the user's own song (akiko, usable 74): target 79/78 → −1/−2/−5/−7, 77 →
    // −1/−3/−6/−8, 76 → −1/−2/−4/−7/−9, 74 → −1/−4/−6/−9/−11. Monotone and legible, which is the
    // whole point of putting it on a slider. Across the eight installed records this moves
    // exactly one that nobody has re-tuned: yuyuko, whose detected target is [37,79] while
    // today's rule lands its rescues on 82 — outside the band its own scan measured.
    let anchor: Option<i64> = {
        let reachable: Vec<i64> = qualifying
            .iter()
            .copied()
            .filter(|&s| dead.iter().all(|&p| range.slot_landing_preferred(p + s)))
            .collect();
        if reachable.is_empty() {
            tracing::info!(
                "range: dead {:?} cannot reach the comfort band [{:.0},{:.0}] the user set at any \
                 depth — falling back to the normal budget",
                dead,
                range.comfort.0,
                range.comfort.1
            );
            None
        } else {
            reachable.into_iter().min_by_key(|s| s.abs())
        }
    };
    let shallowest = anchor.or_else(|| qualifying.iter().copied().min_by_key(|s| s.abs()))?;
    // S151 —— **可选的深度不许花到「合成开始丢源」那条线之外**。
    // 笔 3 量到:上移时颗粒读窗半宽塌成 `T_src/ratio`,`ratio > 2`(= |位移| > 12 半音)之后
    // 每个基音周期都有一段**永远不进任何颗粒**(+14 实测 10.2%,+16 约 20%)。
    // 于是:**必须**走那么深才够得着落点的组照走不误(那是救与不救的问题),
    // 但**为了更干净的落点而多花的那几个半音**,到 `LANDING_RATIO_TWO_ST` 为止。
    // ⛔ 没有这一条,这一刀在「谱面写高 7」那种压力工况上会把最深从 −14 推到 −16。
    let budget = match landing {
        None => LANDING_MAX_EXTRA_DEPTH,
        Some(extra) => {
            // ⛔ **护栏只封「多花的」那部分,永远不许比今天更小** —— 第一版写成 `min(room)`,
            // 于是已经越过那条线的组连今天本来就有的 +1 都被收走了,落到 damage 更差的一格上
            // (实测:谱面 +7 的深组从 −14 变成 −13)。护栏是上限,不是替代品。
            let room = (ratio_two_cap - shallowest.abs()).max(0);
            extra.max(0).min(room).max(LANDING_MAX_EXTRA_DEPTH)
        }
    };
    let mut pool: Vec<i64> = qualifying
        .into_iter()
        .filter(|s| s.abs() <= shallowest.abs() + budget)
        .collect();

    // S146e, the comfort knob: inside the depth budget already fixed above, prefer landings that
    // sit in the user's comfort band. This is the "让高音用舒适的办法去唱" request.
    //
    // ⛔ It is a preference INSIDE the budget, never a new budget — and that placement is the
    // whole design, measured rather than assumed. The obvious implementation (restrict the
    // qualifying set to comfort, THEN take the shallowest) recomputes `shallowest` over a smaller
    // set, so the depth cap starts counting from a deeper anchor: on yuyuko (usable [36,82],
    // comfort [37,79]) it moves the real 炉心融解 rescues from −3/−1 to −7/−5/−4 — a four-semitone
    // recolour nobody has heard, which is precisely the failure S146c-hotfix just removed.
    // As written, all eight installed records render byte-identically today
    // (scratchpad/twopass.py, 803 triples × 8 models), and the knob bites the moment it is moved.
    //
    // ⛔ And it must degrade, not veto: 东雪莲 stores comfort [36,52] for a phrase living at
    // 75-85. A hard AND would take it from "10 groups rescued" to "0 rescued / 10 unsolvable" —
    // the knob would be a kill switch for range extension on that model.
    let in_comfort: Vec<i64> = pool
        .iter()
        .copied()
        .filter(|&s| dead.iter().all(|&p| range.slot_landing_preferred(p + s)))
        .collect();
    if in_comfort.is_empty() {
        // Audited per group. A silent relaxation of the user's own setting is exactly the
        // "验证本身是空的" shape — the knob would look connected and quietly not be.
        tracing::info!(
            "range: dead {:?} has no landing inside comfort [{:.0},{:.0}] within the depth budget \
             (shallowest {shallowest}, +{budget}) — using landing-grade slots {:?}",
            dead,
            range.comfort.0,
            range.comfort.1,
            pool
        );
    } else {
        pool = in_comfort;
    }

    // S151 —— **在 damage 打平的那一批里,按扫描的原始 `low_ratio` 排**,平了才取最浅。
    //
    // ⛔ 为什么这一层非加不可:`damage_from_scan` 给 `low_ratio` 留了 **0.55 以下全免费**,
    // 于是 akiko 的 73..78 六格(0.100 / 0.121 / 0.181 / 0.211 / 0.276 / 0.388)在目标函数里
    // **完全等价**,而「最浅优先」就把死音停在这六格里最差的那一格。实测(用户 2026-08-18
    // 那两条实机渲染,逐音):+0 那跑 85 个死音里 **43 个(51%)落在 77-78**;而 donor 被要求
    // 唱在 77-78 时音内包络起伏率 **30%**,≤76 只有 **7%**(Fisher p = 0.0026)。
    // 另一把独立的尺子给出同向读数:**同一个位移**下,落点 >73 的元音塌 −2.74 dB,
    // ≤73 只塌 −0.95(p = 4.2e-10)。⇒ 这一格是这条线上第一个被两把独立尺子同时指认的杠杆。
    //
    // ⛔ 但它**只在 damage 已经打平的候选之间**做选择,而且预算仍然由 `extra` 封顶 ——
    // S148 记着无上限的 damage 排序会把东雪莲一路走到 **−24**,那是用户耳判过的灾难。
    let thinner = |s: i64| -> f32 {
        dead.iter().map(|&p| range.thinness(p + s).unwrap_or(1.0)).fold(0.0f32, f32::max)
    };

    // ⛔⛔ 默认臂(`landing == None`)到此为止,**与 S151 之前逐字相同**:先按
    // `worst`(死音 ∪ 乘客)取最小,再在容差内取最浅。下面 S157 那一支只有旋钮开着才走,
    // 而 `cover_dead_plan` 硬传 `None`(⚠ 别写行号:那句原来指 `:1065`,今天那个函数在 `:1190`
    //  —— 行号引用会随着任何一次编辑变成假的,而它看起来和真的一模一样)
    // ⇒ **cover 轨结构上不受任何影响**。
    if landing.is_none() {
        let best = pool.iter().map(|&s| worst(s)).fold(f32::INFINITY, f32::min);
        return pool
            .into_iter()
            .filter(|&s| worst(s) <= best + LANDING_DAMAGE_EPS)
            .min_by_key(|s| s.abs());
    }

    // S157 —— **键序改成「死音 damage → 死音 low_ratio → 乘客 damage → 最浅」。**
    //
    // ⛔ 今天的 `worst` 对 **死音 ∪ 乘客** 取 max,于是**一个乘客的 damage 可以否决死音的落点**。
    // 这不是假想:用户 2026-08-20 报的 ぴゃ(notes[685],MIDI 90,组 685..=693)就卡在这里 ——
    // 走 −14 时三个死音落在 76 / 66 / 64(damage **全 0**,而且 76 的 `low_ratio` 0.211 比
    // 今天落点 78 的 0.388 好将近一倍),可两个**乘客**落到 MIDI 61 与 54,
    // 它们的 `low_ratio` 是 **0.572 / 0.573** —— **刚刚越过 `damage_from_scan` 那条 0.55 免费线**
    // ⇒ damage 0.165 / 0.1725 > [`LANDING_DAMAGE_EPS`](0.05)⇒ −14 被踢出 `tied` ⇒ 停在 −12。
    // ⇒ 用一个**中音区乘客的微小瑕疵**,否决掉**顶音本身**的一次大改善。
    //
    // ⭐ 为什么是「挪一把键」而不是「加一个权重」:这里**没有引进任何新常数**。
    // `thin` 是这条轴上**唯一有耳证**的一维(上面那两个 p 值),而「乘客落点的 damage」
    // 一条耳证也没有 —— 它今天却排在 `thin` **上面**。这一笔只是把它挪到下面,
    // 让它继续当**平局的最后一把尺子**。
    //
    // ⚠ 它与「卸乘客」(`trim`)是**两笔独立的账**,S156 交接把它们混成了一件:
    // 那把刀省的是**过路费**(每个乘客进一次 PSOLA 的 0.7-2.0 dB),要靠裁剪/拆句、会造新缝;
    // 而这一笔省的是**落点**,一条缝都不造。离线穷举(63 组 × 四份装机记录 × 炉心融解+7,
    // 复刻件对生产计划 63/63 对拍):把乘客从落点约束里**整个**拿掉,今天的臂与 `landing=3`
    // 都是 **0 / 63 组会动** —— **卸乘客一个落点都改不了**。
    //
    // 实测这一笔 +(护栏 14)之后,薄区(落点 77-79)的死音占比:
    // akiko 30% → **12%** · yachiyo 34% → **3%** · 东雪莲 30% → 30%(**一个字节不动**)·
    // yuyuko 36% → 36%(同上)⇒ 拿不到好处的两份记录**也拿不到坏处**,
    // 而四份记录的**最深位移全部仍然是 14**(= 今天的最深)。
    let worst_dead = |s: i64| -> f32 {
        dead.iter()
            .map(|&p| range.damage_at((p + s) as f32).unwrap_or(DAMAGE_MAX))
            .fold(0.0f32, f32::max)
    };
    let best_dead = pool.iter().map(|&s| worst_dead(s)).fold(f32::INFINITY, f32::min);
    let t1: Vec<i64> =
        pool.into_iter().filter(|&s| worst_dead(s) <= best_dead + LANDING_DAMAGE_EPS).collect();
    let best_thin = t1.iter().copied().map(thinner).fold(f32::INFINITY, f32::min);
    let t2: Vec<i64> =
        t1.into_iter().filter(|&s| thinner(s) <= best_thin + LANDING_THIN_EPS).collect();
    // 乘客的 damage 仍然在,只是降到最后一把:同样干净的落点之间,选对乘客更友善的那个。
    let best_all = t2.iter().copied().map(worst).fold(f32::INFINITY, f32::min);
    t2.into_iter().filter(|&s| worst(s) <= best_all + LANDING_DAMAGE_EPS).min_by_key(|s| s.abs())
}

/// **可选**深度的上限,单位半音。⛔ 它封的只是「为了更干净的落点而多花的那几个半音」——
/// 真的非走那么深才够得着落点的组照走不误(那是救与不救的问题,见 `budget` 那个 `match`)。
///
/// ## ⛔⛔ S157:它原来的理由**已经死了**,这个数是按**新证据**重新定价的
///
/// 原文写的是「`ratio = 2` —— 合成开始漏读源波形的那条线
/// (`src_uncovered_frac`:+12 → 0.00 % / +14 → 10.2 % / +16 → 20.0 %,**real mark train**)」。
/// 而 S156 把 `UTAI_PSOLA_WIN` 翻成了生产默认 1.0,读窗半宽从此 = `win_periods * src_l`
/// = **源侧邻距**(`utai-dsp/src/psola.rs:1849-1851`),**与 ratio 无关** ⇒ 结构上不再留缝。
///
/// ⭐ 这不是推理,是量出来的(S157,`TESTING\s157_knives\probe_ratio2.sh`,**真标记序列**、
/// **三份素材**、生产默认口径 —— 而且为此先给 `psola_probe` 补上了这个读数,
/// 它以前**根本不打**,即「一个承重常数的出处没有人复现得了」):
///
/// | 位移 | ratio | 旧臂 `WIN=0` | **今天(生产默认)** |
/// |---|---|---|---|
/// | +12 | 2.000 | 0.00 % | 0.00 % |
/// | +14 | 2.245 | **10.69 / 10.78 / 10.71 %** | **0.0000 %** |
/// | +16 | 2.520 | **20.38 / 20.48 / 20.40 %** | **0.0000 %** |
///
/// 阳性对照(旧臂)精确复现了原文那张表;阴性对照(+12 两条臂都读 0)证明
/// 「新臂读 0」不是一个恒为 0 的空读数。
///
/// ## ⭐ 而**另一个从没被写进这条 doc 的理由是活的**,这个数按它定
///
/// **donor 自己的音高**漏进 `0.5·f_out` —— 就是用户 S155 笔5 亲耳听成**「合唱感」**的那一维。
/// 同一批探针在 +12…+16 上读出**单调上升**(三份素材:+12 −40.2 / −38.6 / −35.1 →
/// +16 −36.1 / −34.4 / −30.3),看起来像「每深一个半音约 +1 dB」。
///
/// ⛔⛔ **而那条外推当天就被整曲数据推翻了,留着是因为它是这条线上最容易再犯一次的错。**
/// 生产口径整曲(318 个死音,按位移分组,相对各自 400-4000 Hz 的 p50):
///
/// | 位移 | −2 | −3 | −5 | **−9** | **−10** | −11 | **−12** | −13 | **−14** |
/// |---|---|---|---|---|---|---|---|---|---|
/// | ratio | 1.12 | 1.19 | 1.33 | 1.68 | 1.78 | 1.89 | **2.000** | 2.12 | 2.24 |
/// | 亚谐波（**旧出厂，`FRAC` 关**）| −48.3 | −49.7 | −50.1 | **−37.8** | **−34.0** | −43.3 | **−44.9** | −46.0 | **−35.4** |
///
/// ⇒ **它不是剂量,是有峰有谷**:峰在 ratio ≈ 1.68-1.78 与 ≈ 2.245,
/// 而**恰好 ratio = 2.000 那一档反而干净**(−44.9)。
/// ⛔ 后半句同时**推翻了 `psola.rs:1845-1848` 写的机理**(「ratio 2.0 时相邻两颗颗粒读同一个
/// 源标记 ⇒ 成对 ⇒ 0.5·f_out 上有结构」)—— 在**生产默认**(xgrain 开着)下那一档最干净。
/// [假说,未证:ratio 恰好 2 时 `u` 的小数部分不是 0 就是 0.5,而 0.5 正是 xgrain 把相邻
/// 两颗**等权混合**的那一点。]
/// ⚠ 内部一致性:−14 在两条独立渲染的臂上(n=62 / 72)读数都是 **−35.4** ⇒ 这个量由**位移**
/// 决定、不由臂决定,尺子也是稳的。
/// ⚠ 探针为什么会读出「单调」:它只采了 +12…+16,而那正好是从谷爬向峰的一段。
/// ⭐ **通用形状:一条只在窄区间上采样的曲线,读出来的单调性是那个区间的性质,不是曲线的。**
///
/// **为什么是 14。**四份装机记录 × 炉心融解+7 离线穷举(复刻件对生产计划 63/63 对拍):
/// 14 是让最需要它的那一组(ぴゃ,`notes[685..=693]`,`|shallowest| = 11`)够得着落点 76
/// 的**最小值**,而且四份记录在 14 上的**最深位移仍然全部是 14 = 今天的最深**
/// ⇒ **不新增任何一个比今天更深的音,亚谐波零新增暴露**。
/// 放到 16 / 24 只多买 0-3 个百分点的薄区落点,却把最深推到 **16**。
/// ⭐ 而 14 这个值**恰好落在上面那张表的谷与峰之间**:−13(−46.0)是谷,−14(−35.4)是峰,
/// 但 −14 那一档**今天本来就有 62 个音**在上面 ⇒ 这一刀不往那里新增暴露的音,
/// 只是把已经在 −12 的一小批挪过去。⚠ 这条是**取舍**不是物理常数,重定价要拿新数据。
///
/// ⚠⚠ **S157c 起它【生效了】**:`budget` 那个 `match` 只在 `landing == Some(_)` 时读它,
/// 而 [`LANDING_DEFAULT`] 今天是 `Some(3)` ⇒ 这条护栏正在生产里封顶。
/// (这一行以前写的是「今天不生效」,S157c 翻默认时漏改 —— S158 修。)
///
/// ## ⛔⛔ S157c 更正：上面那张表是 **`FRAC` 关着**时量的，已经不描述今天这条臂
///
/// `frac_transport` 翻成默认之后在**新出厂臂**上重量（同一批 318 个死音、同一把尺子）：
/// 全曲 p50 **−35.6 → −39.3（好 3.7 dB）**，而且**深位移那一头改善最大** ——
/// −14 那一档 **−31.3 → −40.7（好 9.4 dB）**；−9 那 129 个音搬到 −10 之后 −33.9 → −38.0。
/// ⇒ **「亚谐波在深位移上更响」这条理由基本不成立了** ——
/// 那大半是「颗粒被放到整数样本上」的表现，而那件事已经修掉
/// （见 `psola.rs` 的 `frac_transport` 与 `add_bell` 里 `d` 那两行取整）。
/// ⇒ ⭐⭐⭐ **通用形状：一个常数的定价表，必须和它被定价时的【那条臂】一起记
///    —— 换了默认，那张表就不再描述今天。**这一场在**同一个常数**上栽了两次：
///    原文那条源覆盖率的理由被 S156 的宽读窗打掉，
///    而我自己换上的这条被 S157c 的亚样本搬运打掉。
///
/// ## ✅ S159 —— **重新定价做了,结论是【14 不动】**(它推翻了上一段自己写的期待)
///
/// 上一段写着「**放宽它的理由现在比之前更强** —— 下一次动这条线时值得重扫」。扫过了,不成立。
///
/// 台子 = `mg_dump_plan_arms` 的 `[cap]` 二维表(`cap ∈ {12,13,14,15,16,18,20,24}` ×
/// `extra ∈ 1..=5`,**走生产函数** `dead_only_plan_with`,0.04 s 一跑,不碰 ONNX)。
/// ⛔ **必须是二维**:`budget = clamp(extra, LANDING_MAX_EXTRA_DEPTH, max(0, cap − |最浅|))`,
/// 出厂 `extra = 3` ⇒ **`|最浅| ≤ cap − 3` 的组根本碰不到 cap**。只扫 cap 一维会读出一张
/// 「几乎全平」的表,并把「14 没问题」写成结论 —— 那正是 S157 记的「一条只在窄区间上采样的
/// 曲线,读出来的单调性是【那个区间】的性质,不是曲线的」。判据
/// [`tests::the_depth_cap_only_bites_once_the_shallowest_landing_is_already_deep`] 钉住这条算术。
///
/// **原 key 炉心融解**(25 组):`cap` 从 12 到 24,**每一行逐字相同** —— 组数 25 / donor 遍数 /
/// Σ|位移| / lr_worst / lr_p50 / 位移集合全部不变。⇒ 用户实际用的那份谱上,这个常数**是惰性的**。
///
/// **+7 压力谱**(93 组,`|最浅|` 能到 21,cap 真的咬得到)—— `extra = 3` 那一列:
///
/// | cap | donor 遍 | Σ\|位移\| | lr_worst | lr_p50 | 最深 |
/// |---|---|---|---|---|---|
/// | 12–16 | 15 | 7888–7896 | 0.627 | 0.212 | −21 |
/// | 18 | 15 | 8098 | 0.627 | 0.212 | −21(新引入 −17)|
/// | 20 | 15 | 8106 | 0.627 | 0.212 | −21(新引入 −20)|
/// | **24** | **17** | 8239 | **0.573** | 0.212 | **−23** |
///
/// ⇒ ⭐ **12–16 之间没有任何可测的差别**;18/20 **改了计划却在 `low_ratio` 上一分钱没买到**,
/// 只是把剂量抬了 2.6%;24 买到 lr_worst **0.054**,代价是 **+2 遍 donor** 与一个 **−23**
/// —— 正好**违反 S157 当初定 14 时用的那条准入判据**(「不新增任何一个比今天更深的音」)。
/// ⇒ **这个区间里没有哪个值在任何一条量得到的轴上比 14 好。**
///
/// ⚠⚠ 三条限制,别把上面读成比它本身更强的结论:
/// 1. 这是**计划层**的读数,它**一个字都没听**。它说的是「计划不会朝任何仪器叫得出好的方向变」,
///    不是「深一点不会更好听」。
/// 2. 只扫了 akiko 这一份 sidecar 的两份谱。别的歌手/谱上 `|最浅|` 的分布不同,cap 可能咬得到。
/// 3. ⭐ **S159 顺带改变了它的经济账**:窗内逆变换之后一遍 donor 的逆变换从 ~25 s 掉到 2-6 s,
///    ⇒ 「多一遍」这个代价小了很多。**即便如此**,24 那一档买到的东西仍然是 0.054。
///
/// ⇒ **14 不动;真正在动的那一维是 [`LANDING_DEFAULT`](`extra`),不是这个上限。**
const LANDING_RATIO_TWO_ST: i64 = 14;

/// How much better a deeper landing's raw `low_ratio` must be before it is preferred over a
/// shallower one. `low_ratio` is stored quantized to a u8 over 0..1, so one stored step is
/// 1/255 ≈ 0.004 — this is ten of those, i.e. wide enough that the rule never chases quantization
/// noise, narrow enough that akiko's 0.388 → 0.211 (78 → 76) still counts as an improvement.
/// ⚠ Deliberately not a tuning knob: on the four installed records every value from 0.01 to 0.10
/// picks the same landing, because the steps that matter on this axis are 0.06-0.24 wide.
const LANDING_THIN_EPS: f32 = 0.04;

/// How far past the shallowest qualifying shift the damage ranking may look. **One semitone.**
///
/// Chosen from the eight installed records, not from taste — what each value does to the rescue
/// depths on 炉心融解:
///   * 0  = the pre-S146c rule (always shallowest) — stops ON the cliff, which is the thing the
///          user heard as broken (akiko lands 85 on 79, damage 0.592, rendered voiced 0.17);
///   * 1  = akiko −6→−7 (the depth the user confirmed by ear as a clear improvement) and
///          东雪莲 −6→−7; **the other six models do not move at all**;
///   * ≥2 = starts walking yuyuko deeper (−3→−5→−6) with no measured benefit;
///   * ∞  = 东雪莲 −24/−16/−13/−11/−10, yachiyo gains a −10. See the ⛔ above.
/// The structural reason 1 is enough: the cliff at the top of a model's range is ONE slot wide
/// (akiko: damage 0.000 flat to 78, 0.592 at 79, saturated at 80). Clearing it needs one step.
const LANDING_MAX_EXTRA_DEPTH: i64 = 1;

/// How much worse than the best reachable landing still counts as "the same". `damage` is stored
/// quantized to a u8 over 0..=DAMAGE_MAX, so one stored step is 3/255 ≈ 0.0118 — this is four of
/// those, i.e. wide enough that the rule never chases quantization noise into a deeper shift.
/// ⚠ Deliberately NOT a tuning knob: on the measured record every value from 0.00 to 0.20 picks
/// the same shift, because the damage cliff at the top of a model's range is ~0.6 tall. If a
/// future model makes this constant matter, that is the signal to look at the record, not to
/// tune the constant.
const LANDING_DAMAGE_EPS: f32 = 0.05;

/// S159k —— cover 区段**最小救援深度**(半音)。比它浅的区段干脆不救。
///
/// ## ⛔ 这是量出来的,不是品味
/// 用户 2026-08-21 的真素材(+7 SV 渲染 × yachiyo RVC,107 段),开/关两臂同一二进制:
/// 区段内 2613 个浊音帧的谐波间噪声,按**该段的位移深度**分组看逐帧改善(负 = 变干净):
///
/// | 深度 | Δ 中位 | 该档里**变脏**(> +3 dB)的帧 |
/// |---|---|---|
/// | **−2 st** | **−1.02 dB** | **21.7%** |
/// | −3 st | −5.55 | 8.4% |
/// | −4 st | −7.06 | 9.5% |
/// | −5…−10 st | −12…−44 | 1-4% |
///
/// ⇒ **−2 那一档几乎什么也没买到,却照样付两条边界的代价**(见 [`cover_dead_plan`] 的
/// 「边界」那一段:每段两条边,而边界台阶正是用户听到的破音)。⇒ 门槛设在 3。
/// ⚠ −3/−4 那两档买到的是 5-7 dB,不算大但是正的,而且变脏比例已经掉到个位数 ⇒ 留着。
const COVER_MIN_RESCUE_DEPTH: i64 = 3;

/// S159k —— 把区段的边往外找清音帧时,**最多找这么远**(毫秒)。找不到就**不外扩**。
///
/// ## ⛔ 上限是量出来的,而且「找不到就不动」是它的另一半
/// 第一版没有上限 —— 判据当场抓住了:一条 20 s 的连唱会被整条吞进 donor
/// (`cover_plan_rescues_a_sustained_dead_climax_locally` / `..._bridges_consonant_gaps_...`)。
/// 于是去量用户那份真素材(292 s,214 条边,100 fps 网格上判有声):
///
/// | | |
/// |---|---|
/// | 有声连段时长 | p50 **135 ms** · p90 765 · max 4340 |
/// | 边到最近清音帧 | p25 **0 ms** · p50 **10** · p75 158 · p90 866 · p95 1758 |
/// | 上限 100 / 150 / 200 ms | 能挪进清音的边 72.4% / 74.8% / **78.5%** |
/// | 上限 **300** / 500 / 800 / 1200 ms | **82.2%** / 85.5% / 89.3% / 92.1% |
///
/// ⇒ 300 ms 之后**收益递减**(上表就是那半的读数)。
/// ⚠ 另一半「代价随外扩长度涨」是**推的不是量的**:每一帧外扩都是被拖进 donor 的乘客,
/// 而「把低处拖进 donor 会脏 25 dB」有读数(区段里 MIDI 72-74 的低处帧 −11.52 vs 同音高
/// 非扩展 −36.89),但**「涨得多快」没有量过** —— 别把它当成量过的东西引用。
/// ⛔ **失败方向:找不到清音就把边留在原处**(= 今天的行为,不更差),
/// 而不是「扩到上限为止」—— 那只会把缝挪个位置,还白搭一堆乘客。
const COVER_EDGE_SEEK_MS: f32 = 300.0;

/// ⚙ 出厂默认 = 800.0 —— S159n:**相邻两段位移相同**且间隔不超过这么久 ⇒ 合并成一段。
///
/// ⛔ 为什么只合并**同位移**:合并本身能直接减少边的条数(边才是用户听到的那个台阶),
/// 而「位移相同」这一条让它**深度上一分不花** —— 没有任何一段会被拖得比它自己需要的更深。
/// ⚠⚠ **S159p 收窄**:上面那句对**段**成立,对**两段之间那截被吞进来的材料**不成立 ——
/// 它们从来没有被任何谓词看过。⇒ 合并现在还要过 `bridge_ok`(中间不含无解区间 +
/// 中间每个浊帧移位后仍 `slot_reachable`),详见 `cover_dead_plan` 里那一段。
/// ⚠ 允许深度差之后能收掉的边多得多(实测:差 ≤1 度 / 0.5 s ⇒ 107 → 82 段),
/// 但那要付「把浅段拖深」的钱,而**那笔钱没有量过** ⇒ 不在这里花。
///
/// ## 定价(用户 2026-08-21 的真素材,107 段)
/// | 门限 | 段数 | 边数 | 覆盖 | 被拖深 |
/// |---|---|---|---|---|
/// | 0(今天) | 107 | 214 | 69.6 s | — |
/// | 0.5 s | 101 | 202 | 71.0 | **0** |
/// | **0.8 s** | **99** | **198** | **72.1** | **0** |
/// | 1.2 s 及以上 | 99 | 198 | 72.1 | 0 |
/// ⇒ 0.8 s 之后曲线就平了(再远的同位移邻段本来就不存在)⇒ 停在 0.8。
const COVER_MERGE_SAME_SHIFT_MS: f32 = 800.0;

/// S159n —— `UTAI_COVER_PHRASE_GAP_MS=<ms>`:**按乐句整段救**(0 = 关 = 出厂默认)。
///
/// ## ⛔ 为什么它今天是旋钮而不是默认
/// 它是唯一能让**台阶不存在**的路子(一个乐句内部不再有 native↔donor 的切换),而且
/// 边数直接腰斩:实测 214 → **132**(门限 100 ms)/ **76**(150 ms),而且每条边按构造
/// 都落在一条 ≥ 门限的真空档里。
/// ⚠ 但代价是**结构性的**:覆盖从 23.8% 涨到 **58-62%**,donor 里的最低音从 MIDI 70.7
/// 掉到 **60.9**(150 ms 时 59.6)。⭐ 好消息:实测**没有任何乐句会掉到 MIDI 48 以下**
/// (该模型的下界)⇒ 不会出现唱不出来的帧。
/// ⛔⛔ **净效果预测不了**:今天的区段里根本没有 MIDI 70 以下的素材,那一档的开/关差
/// 从来没量过。⇒ 先挂旋钮、渲一遍量,**别在量之前翻默认**。
fn cover_phrase_gap_ms() -> f32 {
    std::env::var("UTAI_COVER_PHRASE_GAP_MS")
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0 && *v <= 5000.0)
        .unwrap_or(0.0)
}

/// S85 七轮:COVER(音频轨/audition)的 dead-only 计划 — `dead_only_plan` 的帧域版,同一
/// 死亡判据(slot_singable)与同一落点搜索(minimal_rescue_shift),两轨哲学统一:整曲平移
/// 退役,只有模型「连音高都发不出」的**持续**区域被局部救援;深度由该区域自己的最小落点
/// 决定(真需要 -24 就 -24——只染那一段,不再有整曲代价权衡)。
///
/// One dead-only splice job: render a donor at `shift` and paste frames `[start, end)` back.
/// ★命名字段而非裸元组(S85d 实机翻车纪念:两个生产者曾用不同元组顺序,拼接器把帧号 7512
/// 当移调渲了 donor——「+7512 st」进了日志。编译器从此守约。)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeadJob {
    /// Semitones the donor RENDERS at (negative = down); the inverse undoes it.
    pub shift: i64,
    /// Window start, in probe frames (inclusive).
    pub start: i64,
    /// Window end, in probe frames (exclusive).
    pub end: i64,
}

/// `f0_hz` = 整段探测 f0(**未 pad 网格、含用户移调**=模型将要唱的音高,与输出时间轴对齐)。
/// 区域在原始帧号上构建:GAP_TOL_MS 桥接高潮内的清辅音/换气微隙;浊死帧数 ≥ MIN_VIOLATION_MS
/// 起判(S62b 幻影岛铁律:rmvpe 倍频误读绝不触发染色)。区内被拖拽的浊帧必须保持 singable。
/// 返回 `(Vec<DeadJob>, Vec<无解区域(起,止)>)` — caller 恒审计带位置。
pub fn cover_dead_plan(
    f0_hz: &[f32],
    fps: f32,
    range: &SpeakerRange,
) -> (Vec<DeadJob>, Vec<(i64, i64)>) {
    let min_run = frames_for(MIN_VIOLATION_MS, fps);
    let gap_tol = frames_for(GAP_TOL_MS, fps) + 1;
    let mut idx: Vec<usize> = Vec::new();
    let mut midi: Vec<f32> = Vec::new();
    for (i, &v) in f0_hz.iter().enumerate() {
        if v > 0.0 {
            idx.push(i);
            midi.push(69.0 + 12.0 * (v / 440.0).log2());
        }
    }
    let midi = median5(&midi); // 倍频闪烁卫生,与旧决策同款
    let is_dead = |m: f32| !range.slot_singable(m.round() as i64);
    // 死帧原始帧号 → gap 桥接分组 → 时长门。★门量的是「浊死帧数」而非帧号跨度(审查 S85d:
    // 跨度门会让桥接隙+夹层活帧凑数——5 个 3 帧幻影爆点跨 250ms 就能成区;S62b 铁律
    // 「幻影岛绝不触发染色」要求死亡本身够长。桥接仍跨活帧/清音隙=真高潮里的短落坑不劈区,
    // 由死帧数门兜假区)。
    let mut groups: Vec<(usize, usize, usize)> = Vec::new(); // (起, 止, 浊死帧数)
    for (&i, &m) in idx.iter().zip(midi.iter()) {
        if !is_dead(m) {
            continue;
        }
        match groups.last_mut() {
            Some((_, e, c)) if i <= *e + gap_tol => {
                *e = i;
                *c += 1;
            }
            _ => groups.push((i, i, 1)),
        }
    }
    groups.retain(|&(_, _, c)| c >= min_run);

    // ── S159k —— **边界只许落在清音帧上。**用户 2026-08-21 的真病例(cover 开扩展「高音破音/炸」)。
    //
    // ⛔ 机理是量出来的,而且**第一个假说被自己的数据打掉了**:先怀疑「一个瞬时峰把整段拖下去」,
    //    量 107 段的「过度压低」= `|位移| − (最高音 − 可用顶)` ⇒ 中位 **−0.2 半音**,只有 1 段 > 4,
    //    **没有任何一段的中位音高落在可用范围内** ⇒ 上面那段分组逻辑是对的,不是它的错。
    //
    // ⭐⭐ 真正的缺陷:分组是**逐帧按 f0 过线**做的,所以区段的边**必然落在「音高穿过那条线」的地方**
    //    —— 而那正是一个长音的中间。实测 214 条边里 **125 条(58%)两侧都在唱**;同一批边界位置上的
    //    音色台阶(开/关两臂,唯一变量是扩展):**关 p50 15.38 / p90 30.31 / >30 dB 共 11 条**
    //    → **开 p50 19.18 / p90 37.78 / max 55.26 / 共 32 条**(大台阶**翻三倍**)。
    //    最大的几条是 **−8 → −62 dB** 再切回来 = 一个长音中间,嗓子在「模型硬唱(脏)」与
    //    「救援(干净)」之间来回切,一段两次、107 段。**这就是用户听到的「炸」。**
    //
    // ⭐ 而扩展本身是**大幅净赚**的(同一批帧:谐波间噪声 −26.35 → −46.49 dB,2068 帧变干净、
    //    只有 188 帧变脏;MIDI 81-83 改善 42.9 dB)⇒ ⛔ **别把「关掉扩展」当解法。**
    //
    // ⇒ 把谱面轨的**结构性质**搬过来(它按休止分乐句,边天然落在休止里):一帧过线,就把**整条浊音岛**
    //   一起收进来。⭐ 这在**深度上是免费的** —— 外扩不会抬高该段的最高音。
    // ⚠ 但它**不是**零代价:外扩会把岛上低的那一头也拖进 donor,而实测区段里 MIDI 72-74 的低处帧
    //   读 −11.52 dB、同音高的非扩展帧读 −36.89(拖低会脏 25 dB)。⇒ 代价由 `minimal_rescue_shift`
    //   对「死音 ∪ 乘客」取 max 那条自己兜:乘客太低就找不到落点,那时**退回未外扩的区段**(见下)。
    let voiced = |i: usize| matches!(f0_hz.get(i), Some(v) if *v > 0.0);
    let seek = frames_for(COVER_EDGE_SEEK_MS, fps) as usize;
    // 往外找 ≤ `seek` 帧内的第一个清音帧;⛔ 找不到就**原地不动**(见 `COVER_EDGE_SEEK_MS`)。
    let back = |a: usize| -> usize {
        let stop = a.saturating_sub(seek);
        let mut i = a;
        while i > stop && voiced(i - 1) {
            i -= 1;
        }
        if i > 0 && voiced(i - 1) {
            a
        } else {
            i
        }
    };
    let fwd = |b: usize| -> usize {
        let stop = (b + seek).min(f0_hz.len().saturating_sub(1));
        let mut i = b;
        while i < stop && voiced(i + 1) {
            i += 1;
        }
        if i + 1 < f0_hz.len() && voiced(i + 1) {
            b
        } else {
            i
        }
    };
    // S159n —— 乐句模式(旋钮):区段直接取**包住它的那条乐句**(被 ≥ 门限的清音连段分开的
    // 有声区间)⇒ 边按构造落在真空档里。⛔ 死帧长度门(`MIN_VIOLATION_MS`)照旧先过一遍,
    // 幻影岛铁律(S62b)不因为换了分段方式就失效。
    let phrase_gap = cover_phrase_gap_ms();
    let phrase_of = |i: usize| -> (usize, usize) {
        let g = frames_for(phrase_gap, fps) as usize;
        let (mut a, mut b) = (i, i);
        // 往回:走到一条长度 ≥ g 的清音连段的**右端**为止。
        while a > 0 {
            if !voiced(a - 1) {
                let mut k = a - 1;
                let mut run = 0usize;
                while k > 0 && !voiced(k) {
                    run += 1;
                    k -= 1;
                }
                if !voiced(k) {
                    run += 1;
                }
                if run >= g {
                    break;
                }
            }
            a -= 1;
        }
        while b + 1 < f0_hz.len() {
            if !voiced(b + 1) {
                let mut k = b + 1;
                let mut run = 0usize;
                while k < f0_hz.len() && !voiced(k) {
                    run += 1;
                    k += 1;
                }
                if run >= g {
                    break;
                }
            }
            b += 1;
        }
        (a, b)
    };
    // (外扩起, 外扩止, 它是由哪些原始组扩出来的)
    let mut spans: Vec<(usize, usize, Vec<(usize, usize)>)> = Vec::new();
    for &(a, b, _) in &groups {
        let (ea, eb) = if phrase_gap > 0.0 {
            let (pa, _) = phrase_of(a);
            let (_, pb) = phrase_of(b);
            (pa, pb)
        } else {
            (back(a), fwd(b))
        };
        // ⛔ 外扩之后两段可能撞上(同一条岛上有两处过线)—— 不合并的话拼接器会把同一段贴两次。
        match spans.last_mut() {
            Some((_, pe, orig)) if ea <= *pe + 1 => {
                *pe = (*pe).max(eb);
                orig.push((a, b));
            }
            _ => spans.push((ea, eb, vec![(a, b)])),
        }
    }

    let mut out = Vec::new();
    let mut unfixable = Vec::new();
    let collect = |a: usize, b: usize| -> (Vec<i64>, Vec<i64>) {
        let pitches: Vec<i64> = idx
            .iter()
            .zip(midi.iter())
            .filter(|(&i, _)| i >= a && i <= b)
            .map(|(_, &m)| m.round() as i64)
            .collect();
        let dead: Vec<i64> =
            pitches.iter().copied().filter(|&p| !range.slot_singable(p)).collect();
        (pitches, dead)
    };
    // ⛔ 深度门与「无解」是**两件事**,报法必须分开(S129 铁律:一条红要能被归因)。
    let mut push = |s: i64, a: usize, b: usize| {
        if s.abs() >= COVER_MIN_RESCUE_DEPTH {
            out.push(DeadJob { shift: s, start: a as i64, end: (b + 1) as i64 });
        }
    };
    for (ea, eb, orig) in spans {
        let (pitches, dead) = collect(ea, eb);
        match minimal_rescue_shift(&dead, &pitches, range, None) {
            Some(s) => push(s, ea, eb),
            None => {
                // 外扩把乘客拖得太低 ⇒ 这一整条岛没有落点。**退回未外扩的原始组**:边会落回
                // 长音中间(今天的行为),但至少不比今天差。⛔ 必须响 —— 这是降级不是正常路径。
                tracing::warn!(
                    "range-extend(cover): island [{ea},{eb}] has no landing once extended to \
                     unvoiced edges — falling back to {} un-extended region(s) (audible seams likely)",
                    orig.len()
                );
                for &(a, b) in &orig {
                    let (p, d) = collect(a, b);
                    match minimal_rescue_shift(&d, &p, range, None) {
                        Some(s) => push(s, a, b),
                        None => unfixable.push((a as i64, (b + 1) as i64)),
                    }
                }
            }
        }
    }

    // ── S159n ⑴ —— **同位移的相邻段合并**(见 `COVER_MERGE_SAME_SHIFT_MS`)。
    // ⛔ 放在最后做:合并要看的是**落点算完之后**的位移,而不是分组时的形状。
    //
    // ⛔⛔⛔ **S159p:第一版漏了护栏,而谱面轨的同款函数本来就有。**
    // `merge_same_shift_across_rests`(本文件)只在**中间全是休止**时才合并
    // (`((pe + 1)..g.start).all(|k| note_nums[k] <= 0)`)—— 它靠「只跨休止」躲开了下面这件事。
    // cover 没有休止的概念,于是第一版只比了位移与帧距,**把两段之间那截材料直接吞进 donor
    // 而从不重算一次落点**。那截材料**从来没有被任何谓词看过**,后果有两种,都被独立审计
    // 逐行核验过:
    // ⑴ 中间那截的乘客移位之后可能掉出 `slot_reachable` —— 合并做出了**计划器本人会判「无解」
    //    的窗**。反例(fps=100,`bounds((48,84),(52,79))`):
    //    `[0]*20 + hz(88)*40 + hz(50)*30 + hz(88)*40 + [0]*20` ⇒ 两组各 −9、相距 30 帧 ⇒ 合并成
    //    `{-9, 20, 130}`,而中间 30 帧 MIDI 50 被渲成 **41**,usable 底是 48。
    // ⑵ 中间那截可能正是一段刚被判 `unfixable` 的区间。被盖住之后,审计还在打
    //    「has NO landing … rendered broken as-is」,而拼接器实际按邻居的位移渲了它 ——
    //    **日志那句话是假的**。⛔ 这正是「一条闸的红必须能被归因 / 跑不起来不许被读成通过」要挡的形状。
    // ⚠ 顺带:`back`/`fwd` 在同一份材料上是**明确拒绝**吞掉这截的(seek 上限 + 够不着就不动),
    //    而第一版的合并无条件吞掉 800 ms —— 两把刀的方向是相反的。
    // ⇒ 合并前必须过两道:中间不许含无解区间,且中间每一个浊帧移位后都得 `slot_reachable`。
    let merge_gap = frames_for(COVER_MERGE_SAME_SHIFT_MS, fps) as i64;
    out.sort_by_key(|j| (j.start, j.end));
    let bridge_ok = |p_end: i64, j_start: i64, shift: i64| -> bool {
        // ⑵ 中间盖住了一段「无解」⇒ 不许合并(否则那条审计行就是假的)。
        if unfixable.iter().any(|&(ua, ub)| ua < j_start && p_end < ub) {
            return false;
        }
        // ⑴ 中间每一个浊帧,移位之后都必须仍在模型够得着的范围里 —— 与
        //    `minimal_rescue_shift` 对窗内材料用的是**同一条**谓词。
        idx.iter()
            .zip(midi.iter())
            .filter(|(&i, _)| (i as i64) >= p_end && (i as i64) < j_start)
            .all(|(_, &m)| range.slot_reachable(m.round() as i64 + shift))
    };
    let mut merged: Vec<DeadJob> = Vec::with_capacity(out.len());
    for j in out {
        match merged.last_mut() {
            Some(p)
                if p.shift == j.shift
                    && j.start - p.end <= merge_gap
                    && bridge_ok(p.end, j.start, j.shift) =>
            {
                p.end = p.end.max(j.end);
            }
            _ => merged.push(j),
        }
    }
    (merged, unfixable)
}

/// S85: dead-group 短语窗(50fps 帧域)——短语区间的帧窗向两侧休止扩展(pre ≤4 帧吃借帧
/// 辅音、post ≤2 帧吃释放,各以半个间隙为上限=与相邻唱段/拼接窗永不重叠)。
/// 采样换算与交叉淡化在音频域(本文件 apply_dead_only_windows)。
pub fn dead_group_windows(
    note_nums: &[i64],
    frames: &[i64],
    plan: &[DeadGroup],
) -> Vec<DeadJob> {
    let mut cum = Vec::with_capacity(frames.len() + 1);
    let mut acc = 0i64;
    cum.push(0);
    for &f in frames {
        acc += f.max(0);
        cum.push(acc);
    }
    let raw: Vec<DeadJob> = plan
        .iter()
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
            // S151 护栏 —— 一条窗边落在**唱音**上时,把窗多伸进那个乘客,好让拼接器的 10 ms
            // 交叉淡化压在**它**身上而不是压在被救的死音上。见 `GUARD_FRAMES`。
            // ⚠ 有休止可用时一个字不改(`gap_prev/gap_next > 0` 走原来那条路),所以
            // **今天的计划逐帧不变** —— 这条只有裁剪/拆组产生的乐句内部边界才会走到。
            let pre = if gap_prev > 0 {
                4.min(gap_prev / 2)
            } else if g.start > 0 {
                GUARD_FRAMES.min((cum[g.start] - cum[g.start - 1]) / 2)
            } else {
                0
            };
            let post = if gap_next > 0 {
                2.min(gap_next / 2)
            } else if g.end + 1 < note_nums.len() {
                GUARD_FRAMES.min((cum[g.end + 2] - cum[g.end + 1]) / 2)
            } else {
                0
            };
            DeadJob { shift: g.shift, start: cum[g.start] - pre, end: cum[g.end + 1] + post }
        })
        .collect();
    merge_same_shift_across_rests(note_nums, plan, raw)
}

/// 能被桥接的休止上限,50 fps 帧(= 0.5 s)。缺陷本身只有 60 ms;这个数是「够宽到覆盖所有
/// 乐句间的短休止,窄到不会把窗撑过整段间奏」——实测把 63 个窗合成 50 个(覆盖 +12.6 s),
/// 而**渲染耗时与逐遍 skipped 一个数都不变**。
const MERGE_BRIDGE_FRAMES: i64 = 25;

/// S151d —— **位移相同、中间只隔休止的两个窗必须合并成一个。**
///
/// ⛔ 这是用户 2026-08-18 在 46 秒「ま」与「さ」之间听到的那个杂音,取证如下(炉心融解 +7,
/// 逐 10 ms RMS):窗在 **46.40 s** 到期,而那一刻 donor 的「ま」**还在 −18 dB 上响着**;
/// 交叉淡化把它在 30 ms 内砍到 **−56 dB**(= 同处 base 的电平,那是休止的数字静音),
/// 60 ms 之后下一个窗再淡进来。⇒ **音的收尾被硬切**,而且切它的理由根本不存在 ——
/// 两侧位移**一样**,那 60 ms 本来就在**同一条 donor** 上,是我们自己把它挖掉换成 base 的。
///
/// 判据:`两个同位移的窗之间只有休止时必须合成一个窗`。⚠ 只在**中间全是休止**时合并:
/// 跨过一个唱音去合并,等于把一个乘客悄悄拖进救援(那是另一把刀的事,不许在这里发生)。
fn merge_same_shift_across_rests(
    note_nums: &[i64],
    plan: &[DeadGroup],
    raw: Vec<DeadJob>,
) -> Vec<DeadJob> {
    let mut out: Vec<DeadJob> = Vec::with_capacity(raw.len());
    let mut prev_end_note: Option<usize> = None;
    for (g, j) in plan.iter().zip(raw) {
        let mergeable = match (out.last(), prev_end_note) {
            (Some(last), Some(pe)) => {
                last.shift == j.shift
                    // ⛔⛔ S158 —— **先证明这一对在音符序上真的是「前一组之后」的。**
                    // 少了它,非升序的计划会让下面两条守卫**同时**退化成恒真:`g.start <= pe`
                    // 时 `((pe + 1)..g.start)` 是**空区间**而空区间的 `.all()` 是 `true`
                    // (于是「中间只隔休止」通过了,尽管中间隔着前面一整段歌),同时
                    // `j.start - last.end` 变成**负数**而永远 ≤ 上限 ⇒ 合并发生,
                    // 而 `last.end = j.end.max(last.end)` 把这一条救援**整个丢掉**。
                    // S158 实测:降序喂 `[(3,3,-6),(1,1,-6)]` ⇒ `(6,31)` 那条窗整个消失,
                    // 而组数日志、`plan.json`、单测**全绿**。⚠ 这条路今天**从 env 探针够不着**
                    // (`mg_parse_plan_override` 自 S148 起断言升序不重叠),它只对代码里的
                    // 下一个 planner 开火。
                    //
                    // ⚠ 今天的唯一生产者 `dead_only_plan_with` 由构造升序 ⇒ 这一条**逐位不改
                    // 今天的输出**;它挡的是下一个 planner(裁剪/拆句会重排组)。
                    // ⛔ 只加这一条,**不许**再加「窗不许重叠」:乐句**内部**拆出来的相邻两组,
                    // 窗由构造就是重叠的(护栏从同一条边界向两侧各伸 `GUARD_FRAMES`),而
                    // 「同位移拆分被这里重新合并成逐位相同的窗」正是拆句那把刀现成的阴性对照。
                    // 判据:`merging_never_deletes_a_rescue_*` 的第三段。
                    && g.start > pe
                    && ((pe + 1)..g.start).all(|k| note_nums.get(k).copied().unwrap_or(0) <= 0)
                    // ⛔ 只桥**短**休止:我修的是一个 60 ms 的洞,而不设上限时这条规则会跨过
                    // 长休止把窗撑到 18.6 s(实测),那是**把结论推到证据之外**。
                    // 桥不过去的长休止本来也没有「收尾被切」的问题 —— 那里两条臂都是静音。
                    && (j.start - last.end) <= MERGE_BRIDGE_FRAMES
            }
            _ => false,
        };
        if mergeable {
            let last = out.last_mut().expect("checked");
            // ⚠ S158 变异实测:`.max()` 这一半**杀不掉**(去掉它整个模块 91 条全绿)——
            // 有了上面的 `g.start > pe`,输入按音符序升序,而窗随音符序单调 ⇒ `j.end >= last.end`
            // 由构造成立。留着是防御,**别把它的绿读成「被覆盖了」**。
            last.end = j.end.max(last.end);
        } else {
            out.push(j);
        }
        prev_end_note = Some(g.end);
    }
    out
}

/// How far a rescue window may reach into a **sung** neighbour so the splicer's 10 ms cross-fade
/// lands on the passenger instead of on the rescued note. Two frames = 40 ms at the score's
/// 50 fps grid, i.e. 4× the fade, and the same roll today's rests already get.
///
/// ⛔ Why this is not cosmetic. `apply_dead_only_windows` draws the fade **inside** the window:
/// at the window's first and last sample the donor weight is 0, so those 10 ms are (mostly) BASE
/// — and on a rescued note base is precisely the broken render this whole feature exists to
/// replace. As long as every window edge sat in a rest (pre ≤4 / post ≤2 frames of silence) that
/// was harmless. Trimming and splitting move edges into the middle of a phrase, and without this
/// the trim would hand back up to 10 ms of the defect at **both ends of every cut group** — while
/// every counter (group count, passenger count, donor passes, shift set) and every unit test
/// stayed green. Measured cost of the guard itself: 40 ms of one passenger is rendered from the
/// donor instead of base, at the same written pitch (the inverse already put it back).
const GUARD_FRAMES: i64 = 2;

/// S85e windowed donors: the merged, padded, clamped OUTPUT-sample spans one shift's jobs
/// need rendered. `spf` MUST be the same samples-per-frame map the splicer uses
/// (base.len()/total_frames) so windows land inside their slices; `pad` (samples, from
/// [`DONOR_PAD_SECONDS`]) is the per-side context; overlapping/adjacent spans merge so no
/// audio renders twice. The caller renders each span, inverts it, and embeds it at its span
/// offset in a BASE-SNAPSHOT buffer (not zeros — a slice render can undershoot its span by a
/// frame-grid remainder, and an end-clamped window would then splice digital zeros; base fill
/// restores the pre-S85e "short donor keeps base" fallback per sample). The splicer contract
/// (full-length donor) holds unchanged; windows sit ≥pad inside their span.
pub fn donor_slice_spans(
    jobs: &[DeadJob],
    shift: i64,
    spf: f64,
    out_len: usize,
    pad: usize,
) -> Vec<(usize, usize)> {
    let mut spans: Vec<(usize, usize)> = jobs
        .iter()
        .filter(|j| j.shift == shift)
        .map(|j| {
            let a = (j.start.max(0) as f64 * spf) as usize;
            let b = (j.end.max(0) as f64 * spf) as usize;
            (a.saturating_sub(pad), (b + pad).min(out_len))
        })
        .filter(|(a, b)| b > a)
        .collect();
    spans.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (a, b) in spans {
        match merged.last_mut() {
            Some((_, e)) if a <= *e => *e = (*e).max(b),
            _ => merged.push((a, b)),
        }
    }
    merged
}

/// S159 —— 一条窗在 donor 上对应的**样本**区间,含余量。**帧→样本的唯一公式。**
///
/// `spf` = `base.len() / total_frames`(线性,端点精确;SoVITS hop 网格的取整漂移被它吸收)。
/// `margin` 用哪一个由调用方给,而今天只有两个合法值,各有各的理由:
/// * [`MERGE_BRIDGE_FRAMES`](25) —— **拼接层**真的会从 donor 上切走的那一段;
/// * [`crate::inference::score2svc::DONOR_WINDOW_MARGIN_FRAMES`](29 = 25 + 4)——
///   **渲染侧**的余量,取自「`join_rests` 名义上能把窗边挪到 `l.end + gap + min(gap,4)`」。
///
/// ⭐ S159 实测更正一句会被引用错的话:**拼接层能读到的硬上界其实是 25 帧,不是 29。**
/// `join_rests` 的候选切换点还要能在**左右两条 `kept` 片段**里各取到一个 5 ms 的读窗,
/// 而片段只有 ±25 帧 ⇒ `rms_db` 在越界时返回 `None`、那些候选被 `continue` 掉;
/// 实测(gap 取满 25 时)`hi` 名义上是 `l.end + 29` 帧,而真正能落到的最远处**正好是**
/// `l.end + 25` 帧。⇒ 「窗内逆变换」用 29 是**有余量的超集**,不是刚好够。
/// ⛔ 但仍然用 29 而不是 25:它是仓里已经登记的那个「渲染侧必须覆盖拼接侧够得到的最远处」
/// 的常数(`score2svc.rs:74`,doc 明写「这两个数字从此是一对」),而这一刀是**第三个**
/// 吃同一个余量的消费者。少一帧的代价是「拼进一段没被逆变换过的 donor」——
/// 那是比目标高/低 N 个半音的音频,而组数/乘客数/donor 遍数/位移集**全绿**。
fn donor_read_span(j: &DeadJob, spf: f64, n: usize, margin_frames: i64) -> Option<(usize, usize)> {
    let margin = (margin_frames.max(0) as f64 * spf) as usize;
    let a = ((j.start.max(0) as f64 * spf) as usize).min(n);
    let b = ((j.end.max(0) as f64 * spf) as usize).min(n);
    let (lo, hi) = (a.saturating_sub(margin), (b + margin).min(n));
    (hi > lo).then_some((lo, hi))
}

/// S159 —— 交给 [`apply_inverse_windowed`] 的保留区间:这一遍**自己**的窗,按渲染侧余量取,
/// 合并重叠。空 = 整条缓冲 = 今天。
///
/// `own_windows` 必须是 [`apply_dead_only_windows`] 的闭包收到的**第二个参数** ——
/// 那是全仓唯一一处按 `shift` 过滤的地方(S148 把它收成一处的理由写在那个函数的 doc 里)。
/// ⛔ 别在调用点自己再写一遍 `jobs.iter().filter(|j| j.shift == s)`:那是 S147 那次
/// 「渲多了但拼对了 = 功能正确、收益静默减半」的形状,而它当时只被一个指纹抓到。
pub fn donor_keep_samples(
    own_windows: &[(i64, i64)],
    base_len: usize,
    total_frames: i64,
) -> Vec<(usize, usize)> {
    donor_keep_samples_with(own_windows, base_len, total_frames, windowed_inverse())
}

/// ⚙ 出厂默认 = true —— 开
///
/// S159 —— **窗内逆变换**:donor 遍的 TD-PSOLA 只跑「会被拼回歌里」的那几段
/// (见 [`donor_keep_samples`] 与 `utai_dsp::psola::psola_shift_win`)。
/// `UTAI_RANGE_WINDOWED_INVERSE=0` 关掉 = 整条缓冲照跑 = S158 及以前的行为。
///
/// ## ⛔ 它为什么**不**跟着 `RANGE_ALGO_VERSION` 一起 bump
/// 这条线上每一个旋钮到今天为止都是「换音质」的,所以「翻默认必须成对 bump 缓存版本」
/// 是条铁律。**这一个不是**:窗内的输出与整条臂**逐位相同**,而窗外的样本拼接层一个也读不到
/// ⇒ 出厂音频一个字节不变,变的只有秒数。
/// ⇒ bump 它只会让每一条已经烤好的救援白重渲一遍 —— `audition_cache_tag` 的 doc 明写
/// 那是「用户直接叫 bug 的那一件事」。
/// ⚠ **但这条结论是被证明的,不是被假设的**,它整个挂在下面这几条上:
/// `psola.rs::the_window_keeps_every_sample_inside_it_bit_identical`(引擎侧逐位)·
/// [`tests::the_inverse_window_covers_every_sample_the_splice_reads`](拼接侧只读窗内)·
/// [`tests::the_window_reaches_the_engine_and_an_empty_window_is_not_an_error`](这条路真的通)。
/// 哪一条塌了,这个「不 bump」的决定就跟着塌。
///
/// ## 它存在的第二个理由
/// 速度 A/B 必须**同一个二进制**(S146:两条臂不同路时,量到的每一个读数里都混着路线差)。
pub fn windowed_inverse() -> bool {
    parse_windowed_inverse(std::env::var("UTAI_RANGE_WINDOWED_INVERSE").ok().as_deref())
}

fn parse_windowed_inverse(v: Option<&str>) -> bool {
    match v {
        Some(s) if !s.trim().is_empty() => !matches!(s.trim(), "0" | "false" | "off" | "no"),
        _ => WINDOWED_INVERSE_DEFAULT,
    }
}

const WINDOWED_INVERSE_DEFAULT: bool = true;

/// [`donor_keep_samples`],但开关由参数给 —— ⛔ 判据不许读进程环境(S151 笔1:
/// 一个读 env 的纯函数在 CI 上与在别人 export 过变量的机器上会给出不同答案)。
pub fn donor_keep_samples_with(
    own_windows: &[(i64, i64)],
    base_len: usize,
    total_frames: i64,
    enabled: bool,
) -> Vec<(usize, usize)> {
    if !enabled || own_windows.is_empty() || base_len == 0 || total_frames <= 0 {
        return Vec::new();
    }
    let spf = base_len as f64 / total_frames as f64;
    let mut v: Vec<(usize, usize)> = own_windows
        .iter()
        .filter_map(|&(start, end)| {
            donor_read_span(
                &DeadJob { shift: 0, start, end },
                spf,
                base_len,
                crate::inference::score2svc::DONOR_WINDOW_MARGIN_FRAMES,
            )
        })
        .collect();
    v.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(v.len());
    for (a, b) in v {
        match out.last_mut() {
            Some((_, e)) if a <= *e => *e = (*e).max(b),
            _ => out.push((a, b)),
        }
    }
    out
}

/// Active RMS (50 ms windows above -40 dBFS mean) — the dead-only donor level match. Silence /
/// no active window ⇒ None (caller skips matching rather than amplifying noise).
fn active_rms(x: &[f32], sample_rate: u32) -> Option<f32> {
    let win = (sample_rate as usize / 20).max(1);
    let mut sum = 0f64;
    let mut n = 0usize;
    for c in x.chunks_exact(win) {
        let r = (c.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / win as f64).sqrt();
        if r > 0.01 {
            sum += r;
            n += 1;
        }
    }
    (n > 0).then(|| (sum / n as f64) as f32)
}

/// S85 dead-only 拼接编排(**两轨单一源**:score 短语窗 / cover 死区窗共用;S85 七轮从
/// score2svc.rs 搬家至此)。base = 无移调的完整渲染;每个 distinct shift 渲一个完整 donor
/// (闭包内部完成「移调渲染+逆变换回原音高」,与 base 同构=耳判验证过的拼接口径);
/// jobs=(shift, 起帧, 止帧)@任意帧率网格,10ms 余弦交叉淡化(窗短于双淡化时收缩淡化宽度,
/// 绝不静默丢弃)。
///
/// 电平匹配(审查 S85 major;`match_levels`):**只在两渲各自归一时开**——score 的
/// render_score_* 每渲各自 peak_normalize(0.92),全曲 active-RMS 比例对齐消归一台阶
/// (±12dB 安全笼)。cover 管线无逐渲归一,电平差=模型对移调的真实响应,不该被全局
/// 拉平——且窗口化 donor 的缓冲大半为零,全曲 RMS 对比会拿「死区邻域响度」冒充全曲
/// (高潮区必偏响 → 误衰减),所以 cover 恒 false。
/// 帧→样本映射 = base.len()/total_frames 实测每帧样本数(端点精确;SoVITS hop 网格取整的
/// 累积漂移被线性吸收,RVC 网格下与名义值相等)。
///
/// ⛔⛔ S148:闭包的第二个参数是**这一遍自己的窗**,而不是全部窗 —— 这个参数存在的唯一理由
/// 是 S147 的那次 hotfix。当时两个调用点各自在闭包里写了一遍
/// `jobs.iter().filter(|j| j.shift == s)`,而第一版**漏掉了那个 filter**、把全部位移的窗的并集
/// 传给每一遍 ⇒ 四个 donor 渲同一批 chunk,**功能正确、收益静默减半**;单元测试、长度契约、
/// 音频秒数全部正常,唯一暴露它的是「四个不同位移的 `skipped` 完全相同」这个指纹。
/// ⇒ 那个 filter 现在**只存在于这里一处**,调用点拿到什么就用什么,没有再写错的余地。
/// 判据见 `donor_render_gets_only_its_own_windows`(附「同样的窗给同样答案」的阴性对照)。
pub fn apply_dead_only_windows(
    base: &mut [f32],
    sample_rate: u32,
    total_frames: i64,
    jobs: &[DeadJob],
    match_levels: bool,
    donor_render: impl FnMut(i64, &[(i64, i64)]) -> crate::Result<Vec<f32>>,
) -> crate::Result<()> {
    apply_dead_only_windows_with(
        base,
        sample_rate,
        total_frames,
        jobs,
        match_levels,
        join_rests_enabled(),
        donor_render,
    )
}

/// 同上,但「异位移短休止接上」由参数给 —— ⛔ 判据不许读进程环境(`dead_only_plan_with`
/// 同一个模式)。**这条路径存在的理由不是好看**:走 env 的那一版让「片段余量不够 ⇒ 静默
/// 退回今天的行为」这条分支在变异里全绿,也就是收益可以被悄悄砍掉而没有任何东西会红。
#[allow(clippy::too_many_arguments)]
pub fn apply_dead_only_windows_with(
    base: &mut [f32],
    sample_rate: u32,
    total_frames: i64,
    jobs: &[DeadJob],
    match_levels: bool,
    join_enabled: bool,
    mut donor_render: impl FnMut(i64, &[(i64, i64)]) -> crate::Result<Vec<f32>>,
) -> crate::Result<()> {
    if jobs.is_empty() || base.is_empty() || total_frames <= 0 {
        return Ok(());
    }
    let spf = base.len() as f64 / total_frames as f64;
    let xf = (sample_rate as usize / 100).max(2); // 10 ms
    let base_rms = if match_levels { active_rms(base, sample_rate) } else { None };
    let mut shifts: Vec<i64> = jobs.iter().map(|j| j.shift).collect();
    shifts.sort_unstable();
    shifts.dedup();
    // S152 —— 每一遍 donor 在**它自己的窗 ± 余量**上的样本,留到全部渲完之后再拼。
    // ⛔ 为什么要留:窗边该放在休止的哪一点,只有**同时看得见两侧那两条 donor** 才决定得了
    // (见 `join_rests`)。留的是片段不是整条:全曲窗覆盖约 56 %,实测这首歌约 30 MB。
    let mut kept: Vec<(i64, usize, Vec<f32>)> = Vec::new();
    for s in shifts {
        // 这一遍**自己**的窗。⛔ 同一个 filter 下面 :896 还要用一次(音频域拼接),两处必须
        // 是同一个谓词 —— 那正是 S147 hotfix 的形状:闭包那一侧漏了它,拼接这一侧没漏,
        // 于是「渲多了但拼对了」= 功能正确、收益减半。
        let own: Vec<(i64, i64)> =
            jobs.iter().filter(|j| j.shift == s).map(|j| (j.start, j.end)).collect();
        let mut donor = donor_render(s, &own)?;
        if donor.len().abs_diff(base.len()) > spf.ceil() as usize {
            tracing::warn!(
                "range-extend(dead-only): donor {} samples vs base {} — windows clamped",
                donor.len(),
                base.len()
            );
        }
        if let (Some(br), Some(dr)) = (base_rms, active_rms(&donor, sample_rate)) {
            let g = (br / dr).clamp(0.25, 4.0); // ±12dB 安全笼(比值失真时别放大灾难)
            if (g - 1.0).abs() > 1e-3 {
                donor.iter_mut().for_each(|v| *v *= g);
            }
        }
        let n = base.len().min(donor.len());
        // S152 —— 留片段,拼接推迟到全部 donor 渲完(见 `kept` 的注释)。余量 = 桥接上限,
        // 因为窗边最远只会挪到相邻休止的另一头。
        // S159 —— 区间的算法搬进 [`donor_read_span`],因为「窗内逆变换」要用**同一个**公式:
        // 那一刀保住的样本区间必须 ⊇ 这里切走的片段,而两处各写一遍 = 一处改了另一处不改。
        for (ji, j) in jobs.iter().enumerate().filter(|(_, j)| j.shift == s) {
            if let Some((lo, hi)) = donor_read_span(j, spf, n, MERGE_BRIDGE_FRAMES) {
                kept.push((ji as i64, lo, donor[lo..hi].to_vec()));
            }
        }
    }
    splice_kept(base, sample_rate, spf, jobs, &kept, xf, join_enabled)
}

/// S152 —— 拼接层,从 `apply_dead_only_windows` 拆出来:它现在拿到的是**全部**位移的 donor
/// 片段,所以窗边可以由音频决定(`join_rests`),而不是只能由帧数规则决定。
fn splice_kept(
    base: &mut [f32],
    sample_rate: u32,
    spf: f64,
    jobs: &[DeadJob],
    kept: &[(i64, usize, Vec<f32>)],
    xf: usize,
    join_enabled: bool,
) -> crate::Result<()> {
    // ⛔ `join_enabled` 是参数不是 env —— 判据不许读进程环境(S151 笔1)。
    let n = base.len();
    // ⛔ 时间序,不是位移序。窗互不重叠时两者等价(实测这首歌 0 对重叠),但**接上之后
    // 相邻两窗会重叠一个淡化宽度**,而那一次淡化的意义正是「从上一条 donor 淡到下一条」——
    // 顺序错了就会淡回 base,也就是这一刀本来要消灭的那个洞。
    let mut order: Vec<usize> = (0..jobs.len()).collect();
    order.sort_by_key(|&i| (jobs[i].start, jobs[i].end));
    let join = join_rests(base, sample_rate, spf, jobs, kept, &order, xf, join_enabled);
    for (oi, &ji) in order.iter().enumerate() {
        let j = &jobs[ji];
        // ⛔ 片段一次找完,不许放进逐样本的循环 —— 那是 53 × 12.8 M 次线性扫。
        let Some((_, seg_lo, seg)) = kept.iter().find(|(i, _, _)| *i == ji as i64) else {
            continue;
        };
        let (fa, fb) = (j.start, j.end);
        let mut a = ((fa.max(0) as f64 * spf) as usize).min(n);
        let mut b = ((fb.max(0) as f64 * spf) as usize).min(n);
        // 被接上的一对:左窗伸到切换点(不淡出),右窗从切换点【往前】一个淡化宽度开始
        // ⇒ 那一次淡化两侧都是 donor,base 一个样本都不参与。
        let mut hard_end = false;
        if let Some(t) = join.get(&oi).copied() {
            b = t.min(n);
            hard_end = true;
        }
        if oi > 0 {
            if let Some(t) = join.get(&(oi - 1)).copied() {
                // ⛔ 赋值而不是 `min` —— 切换点可能**晚于**今天的窗起点(46 s 那处的正解就是),
                // 只往前挪的话那一半永远够不着。
                a = t.saturating_sub(xf).min(n);
            }
        }
            // 窗短于双淡化 → 收缩淡化宽度;完全空窗才放弃,响亮。
            // 写入范围必须落在留下来的片段里。今天由构造一定成立(余量 = 桥接上限),
            // 但「窗边被挪出片段」是一条会静默出错的路 ⇒ 夹紧并响亮。
            let (sa, sb) = (*seg_lo, *seg_lo + seg.len());
            if a < sa || b > sb {
                tracing::warn!(
                    "range-extend(dead-only): window {a}..{b} escapes its donor segment {sa}..{sb} — clamped"
                );
            }
            let a = a.max(sa);
            let b = b.min(sb);
            let xfw = xf.min((b.saturating_sub(a)) / 2);
            if b <= a || xfw == 0 {
                tracing::warn!(
                    "range-extend(dead-only): window frames {fa}..{fb} degenerate after clamp ({a}..{b} samples) — NOT rescued"
                );
                continue;
            }
            // S151 —— **只在另一侧真的有东西可以淡回去的时候才淡**。A fade exists to hide the
            // donor↔base discontinuity; at the very edge of the buffer there is no continuation
            // to hide, and fading there simply hands the last 10 ms back to base. That is not
            // hypothetical: on every score whose final phrase is rescued the last group gets
            // `gap_next == 0` ⇒ `post == 0` ⇒ the fade-out lands ON the last rescued note
            // (measured on 炉心融解/akiko: group `[796..=802]`, note 802 = MIDI 81, dead).
            // ⚠ 这一条改的是**今天的**输出(结尾那 10 ms),所以它跟着 `RANGE_ALGO_VERSION` 一起 bump。
            let fade_in = a > 0;
            let fade_out = b < n && !hard_end;
            // S151d ⛔ **一条被干预判负的假说,记下来免得下一个人再走一遍**:
            // 用户报「咚」之后我怀疑这里的等增益淡化 —— `base` 与 `donor` 是两次独立渲染,
            // 相位不相关,等增益在淡化中点保幅度不保功率,理论上掉 3 dB,而 10 ms 的幅度凹坑
            // 正落在 40-150 Hz。**换成 √w/√(1−w) 重渲了一遍整曲,读数一个字没动**
            // (窄缝处 Δ低频 +8.44 → +8.60 dB,用户点名的三处逐点差 ≤0.1 dB)⇒ **不是它。**
            // ⚠ 同时纠正我自己的一个量错:「每条窗边注入 +5~18 dB 低频」是**尺子的混淆** ——
            // ±100 ms 的窗跨在边上,里面大半是被救的音频,而同处的 base 是那个又轻又破的高音,
            // 差的是**响度**不是低频。按「窗边 vs 同一条臂的窗心」重量:相对低频只差 **−0.48 dB**。
            for k in a..b {
                let w = if fade_in && k < a + xfw {
                    0.5 - 0.5 * (std::f32::consts::PI * (k - a) as f32 / xfw as f32).cos()
                } else if fade_out && k >= b - xfw {
                    0.5 - 0.5 * (std::f32::consts::PI * (b - k) as f32 / xfw as f32).cos()
                } else {
                    1.0
                };
                base[k] = base[k] * (1.0 - w) + seg[k - *seg_lo] * w;
            }
        }
    Ok(())
}

/// S152 —— **窗边该放在休止的哪一点,只有音频知道。**
///
/// ## 缺陷
/// 两个相邻、**位移不同**、中间只隔一段休止的窗,今天被 `dead_group_windows` 的
/// `pre = min(4, gap/2)` / `post = min(2, gap/2)` 切成「donor_L / base / donor_R」三段,
/// 两侧各留一个电平台阶。S151 笔5 只合并**同位移**的窗,异位移这一族被规则**按设计放过了**
/// —— 全曲还剩 **31 条**。
///
/// ## ⛔⛔ 这把刀**没有耳朵背书,而且它最初瞄的那个靶子是个非事件**
/// 我一开始把它当成「用户 2026-08-18 在 46 秒『ま』与『さ』之间听到的那个杂音」的解药,
/// 对抗核验把这条打掉了,两条都要记住:
/// * **46.041 s 那条桥的桥内是 −240 dBFS 的数字静音**(base / donor_L / donor_R 三条全是)
///   ⇒ 那里根本没有洞,它在我第一版排序里排第 2 是**尺子的问题**;
/// * 用户耳朵指的那一处是 **46.401 s**(合并**之前**那一版的同位移桥,在 41 条桥里
///   `Δnotch` 排 **1/41**),而 S151 笔5 已经把它合并掉了 —— 可用户说三条整曲臂**都还听得到**,
///   ⇒ **那个杂音的真凶仍然没找到**,别把这一刀读成「46 s 修好了」。
/// ⚠ 现存证据只支持一句话:**它把接缝从「在 −29 dBFS 处切换」挪到「在 −240 / −52 dBFS 处切换」**,
/// 那是一个客观更安全的位置,仅此而已。翻默认前必须盲测。
///
/// ## 为什么必须由音频决定(离线扫过,`edge_sweep.py`)
/// 31 对这样的窗,把切换点放在休止里逐 5 ms 扫,台阶(两侧 20 ms 电平差):
///
/// | 方案(切换点两侧 20 ms 的电平差) | p50 | p90 | max |
/// |---|---|---|---|
/// | 今天(中间留 base) | 6.59 | 12.97 | 17.25 dB |
/// | 接上,切换点取**中点** | 2.67 | **24.27** | **165.65 dB** |
/// | 接上,切换点**按音频搜** | **0.03** | **0.77** | 6.41 dB |
///
/// ⇒ ⛔ **固定常数比今天还差**,而按音频搜几乎把台阶消灭。最优点离中点的偏移从 −100 到
/// +95 ms 都有 ⇒ 没有任何固定偏移能替代搜索。这就是这一刀不能写成「post 从 2 改成 4」的原因。
///
/// ## ⛔ 判据本身被自己的分辨率骗过一次
/// 第一版按「两侧电平**差**最小」搜、窗 20 ms。46 s 那段休止里其实有 **15 ms 的真数字静音**
/// (base / donor_L / donor_R 三条全是 −240 dBFS)—— 在那里切换是零风险的,而
/// **10 / 20 / 40 ms 的窗全部错过它**(分别落在 46.040 / 46.102 / 46.115),只有 5 ms 窗找得到。
/// ⇒ 判据改成「**两条里较响的那条最安静**」+ 5 ms 窗:它自动同时满足「都安静」与「差不多」,
/// 并且在有静音的休止里直接落到 −240。31 对里 **19 对**的休止有这样的静音。
///
/// ## 返回
/// `{order 下标 → 切换点样本}` —— 键是**时间序**里的位置,因为拼接就是按那个顺序做的。
fn join_rests(
    base: &[f32],
    sample_rate: u32,
    spf: f64,
    jobs: &[DeadJob],
    kept: &[(i64, usize, Vec<f32>)],
    order: &[usize],
    xf: usize,
    enabled: bool,
) -> std::collections::HashMap<usize, usize> {
    // ⛔ `enabled` 是参数不是 `std::env` —— 判据不许读进程环境(S151 笔1 的规矩:
    // 一个读 env 的纯函数在 CI 上和在别人 export 过变量的机器上会给出不同答案)。
    let mut out = std::collections::HashMap::new();
    if !enabled {
        return out;
    }
    // ⛔ 5 ms,不是 20 ms。第一版用 20 ms 的窗找「两侧电平最接近的点」,在 46 s 那处**漏掉了
    // 正确答案**:那段休止里有 15 ms 的**真数字静音**(三条信号全 −240 dBFS),5 ms 窗找得到,
    // 10 / 20 / 40 ms 窗全部错过(分别落在 46.040 / 46.102 / 46.115)。
    let w = (sample_rate as usize / 200).max(8); // 5 ms
    let step = (sample_rate as usize / 400).max(1); // 2.5 ms
    let seg = |ji: usize| kept.iter().find(|(i, _, _)| *i == ji as i64);
    let rms_db = |buf: &[f32], lo: usize, a: usize, b: usize| -> Option<f32> {
        let (a, b) = (a.checked_sub(lo)?, b.checked_sub(lo)?);
        if b <= a || b > buf.len() {
            return None;
        }
        let m: f64 = buf[a..b].iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>()
            / (b - a) as f64;
        Some(20.0 * (m.sqrt().max(1e-12)).log10() as f32)
    };
    let cell = (sample_rate as usize / 100).max(8); // 10 ms —— 打分格
    for oi in 0..order.len().saturating_sub(1) {
        let (li, ri) = (order[oi], order[oi + 1]);
        let (l, r) = (&jobs[li], &jobs[ri]);
        // 同位移的那一族是 S151 笔5 的事(它在计划层就合并了),这里只管异位移。
        if l.shift == r.shift || r.start <= l.end {
            continue;
        }
        let gap = r.start - l.end;
        if gap <= 0 || gap > MERGE_BRIDGE_FRAMES {
            continue;
        }
        let (Some((_, llo, lbuf)), Some((_, rlo, rbuf))) = (seg(li), seg(ri)) else { continue };
        // ⭐ 搜索范围是**整段休止**,不只是今天那两条边之间。今天的窗把休止切成
        // `post(≤2 帧) / gap / pre(≤4 帧)`,所以整段休止 = `[l.end − min(2,gap), r.start + min(4,gap)]`
        // (`min` 保证任何休止长度下都不会越进唱音)。
        // ⛔ 必须允许 T **晚于**今天的右窗起点 —— 46 s 那处的正解就在那边:今天窗在音前 80 ms
        // 就开了,而右 donor 在其后 40 ms 里是静音,于是 base 的辅音预卷被换成了一个 26 dB 的洞。
        let f2s = |f: i64| (f.max(0) as f64 * spf) as usize;
        let lo = f2s(l.end - gap.min(2)).max(*llo + w).max(w);
        let hi = f2s(r.start + gap.min(4))
            .min(base.len().saturating_sub(w))
            .min(rlo + rbuf.len() - w);
        if hi <= lo {
            continue;
        }
        // 整段休止上的 base 电平(10 ms 一格)—— 「洞」就是相对它挖下去的那部分。
        let cells: Vec<(usize, f32)> = (lo..hi)
            .step_by(cell)
            .filter_map(|u| rms_db(base, 0, u, u + cell).map(|v| (u, v)))
            .collect();
        // ⭐ 参照 = **今天**在同一段休止上挖出来的洞。今天的分法是「L 到 l.end / base / R 从
        // r.start 起」,所以它自己也有一个 dip;这一刀只有**比今天更浅**才值得动。
        // ⛔ 用绝对门限是错的:今天在两条边之间放的就是 base(dip 恒 0),绝对门限会把
        // 「今天已经很好」的那些也一起接上(离线扫的 128.9 s 就是这一类)。
        let e_l = f2s(l.end);
        let e_r = f2s(r.start);
        // ⛔ dip 只在**两种分法真的不同**的那一段上算 —— `[min(t,e_l), max(t,e_r)]`。
        // 第一版在整段休止上算,于是「左窗内部那几格」(两种分法完全相同)把最大值钉死,
        // 今天与候选读出**一模一样的 24.44**,判据当场变空。
        let dip_in = |a: usize, b: usize, pick: &dyn Fn(usize) -> Option<f32>| -> f32 {
            let mut d = 0.0f32;
            for &(u, bl) in &cells {
                if u < a || u + cell > b {
                    continue;
                }
                if let Some(c) = pick(u) {
                    d = d.max(bl - c);
                }
            }
            d
        };
        let today_pick = |u: usize| {
            if u + cell <= e_l {
                rms_db(lbuf, *llo, u, u + cell)
            } else if u >= e_r {
                rms_db(rbuf, *rlo, u, u + cell)
            } else {
                rms_db(base, 0, u, u + cell) // 今天这一段就是 base ⇒ 自己减自己 = 0
            }
        };
        let mut best: Option<((f32, f32), usize)> = None;
        let mut t = lo;
        while t <= hi {
            let (Some(la), Some(rb)) =
                (rms_db(lbuf, *llo, t - w, t), rms_db(rbuf, *rlo, t, t + w))
            else {
                t += step;
                continue;
            };
            // ⭐ 主判据 = **拼出来的信号相对 base 最深挖下去多少 dB**(只算负的那一侧)。
            // 用户听到的就是这个洞;而「最安静的切换点」那条判据会把洞**加深**
            // (46 s 那处它选 46.05,洞 24.5 dB;选 46.16 时只有 2.1 dB)。
            // ⛔ 区域**固定**为整段休止,不许随候选缩放。第一版按 `[min(t,e_l), max(t,e_r)]`
            // 取,于是候选把「切换点之后的那个洞」挤出了区域 —— 它靠**藏起自己的缺陷**拿高分。
            let dip = dip_in(lo, hi, &|u: usize| {
                if u + cell <= t {
                    rms_db(lbuf, *llo, u, u + cell)
                } else if u >= t {
                    rms_db(rbuf, *rlo, u, u + cell)
                } else {
                    None // 跨切换点那一格不算,两侧各半没有意义
                }
            });
            // 收益 = 今天在**同一段**上的洞 − 候选的洞。⇒ 判据自校准,不需要绝对门限。
            let gain = dip_in(lo, hi, &today_pick) - dip;
            // 平局(比如两侧都安静的休止)时退回「切在最安静处」。
            let sc = (-gain, la.max(rb));
            if best.is_none_or(|(bs, _): ((f32, f32), usize)| {
                sc.0 < bs.0 - 0.05 || ((sc.0 - bs.0).abs() <= 0.05 && sc.1 < bs.1)
            }) {
                best = Some((sc, t));
            }
            t += step;
        }
        if let Some(((neg_gain, _q), t)) = best {
            // 只有**比今天浅一大截**才动。⛔ 不设这一条,「今天已经很好」的那些也会被接上,
            // 而离线扫里正好有一个反例(128.9 s:今天 1.12 dB,强行接上是 6.41)。
            if -neg_gain >= JOIN_MIN_GAIN_DB && t > xf {
                out.insert(oi, t);
            }
        }
    }
    out
}

/// 切换点必须安静到这个程度才接 —— 找不到就别动,今天那条边(中间留 base)照旧。
///
/// ⛔ **门限是从分布里长出来的,不是挑的**(`edge_sweep.py`,31 对真窗、5 ms 窗):
/// * 今天切换处 `max(左, 右)` 的 p50 = −54.5、**p90 = −29.1**、最坏 −21.8 dBFS;
/// * 按这个判据搜到的最优 p50 = **−240**(19/31 对的休止里有一段真数字静音)、p90 = −52.0;
/// * 落在 −50 dBFS 以下的有 **29/31**。
/// ⇒ −50 落在两个分布之间那 20 dB 宽的平台中间。
const JOIN_QUIET_DBFS: f32 = -50.0;

/// 要动这条边,新的洞必须比**今天**的浅至少这么多 dB。
///
/// ⛔ 6 dB 是从**用户点名的那一处**读出来的,不是挑的:46 s 那段休止里,今天(以及用户的实机
/// 渲染)在 46.11-46.14 有一个 **26 dB** 的洞 —— donor −9 在那 40 ms 里是静音,而 base 有
/// −36 dB 的 /m/ 预卷,窗却在音前 80 ms(`pre = 4` 帧)就开了。把切换点挪到 46.16 之后,
/// 同一段的最大负偏差只剩 **2.1 dB**。⇒ 收益 24 dB,门限取 6 是保守的两倍余量。
/// ⚠⚠ 这个洞**三条整曲探针臂 + 用户的实机渲染上全都有**(实测 46.13 处:实机 −62.6 dB vs
/// 扩展全关 −36.4),而在 S152 之前**没有任何判据看得见它** —— 包括我自己这一场先写的
/// 「电平台阶」与「最安静切换点」两把,它们都会把这个洞**加深**。
const JOIN_MIN_GAIN_DB: f32 = 6.0;

/// ⚙ 出厂默认 = false —— 关
/// `UTAI_RANGE_JOIN=1` 打开「异位移短休止按音频接上」。**默认关 ⇒ 生产逐位不变。**
/// ⛔ 翻它必须成对 bump `RANGE_ALGO_VERSION` 与 `audition_cache_tag`,而且要盲测过
/// (S146 protocol;⚠ 改窗集合会让每条 donor 的 chunk 选择变、整条换一个相位实现,
/// 所以那次 A/B **必须带同臂两跑的地板**)。
pub fn join_rests_enabled() -> bool {
    parse_join_rests(std::env::var("UTAI_RANGE_JOIN").ok().as_deref())
}

fn parse_join_rests(v: Option<&str>) -> bool {
    match v {
        Some(s) => matches!(s.trim(), "1" | "true" | "on" | "yes"),
        None => JOIN_RESTS_DEFAULT,
    }
}

const JOIN_RESTS_DEFAULT: bool = false;

/// κ — how much of the inverse's pitch move the FORMANTS follow:
///   κ=0  formants stay where the model put them — the source timbre. Under TD-PSOLA this is
///        free: the algorithm cannot move the spectral envelope at all, it only re-spaces
///        pitch periods.
///   κ=1  formants move with the pitch (the plain spectral transpose) — bright/chipmunk.
/// Default 0 is the configuration the user A/B'd on real songs and accepted
/// ("干净了…共振腔相对来讲保持的甚至还挺好").
///
/// ⚠ The old note here claimed κ=0's cost was "dark/covered". That was backwards, and it cost a
/// session: on the S142 real-song render the user heard the opposite — κ=0 was audibly BRIGHT,
/// a chipmunk. S145 found why: Signalsmith's "envelope" is a morphological closing whose
/// smoothing width is set solely by the base f0 we feed it (`signalsmith-stretch.h:985`), so the
/// higher the note the coarser the estimate and the closer the compensation gets to the identity
/// — and range extension only ever rescues high notes. On real material κ=0 leaked +2.40
/// semitones of formant rise out of a possible 6.00. That is the whole reason the engine changed.
///
/// ⚠ Until S146 this constant had ZERO readers on the production paths: `RvcOptions::default()`
/// and `SovitsOptions::default()` each wrote a bare `0.0`, so "the default κ" had four
/// independent definitions (here, those two, and `voiceDefaults.ts`) and nothing tying them
/// together. The two Rust ones now read this constant. `voiceDefaults.ts` still declares its own
/// — the frontend cannot see a Rust const, and inventing a codegen step for one number is worse
/// than the drift it prevents; but it IS a second source of truth and it is written down here.
pub const DEFAULT_FORMANT_KAPPA: f32 = 0.0;

/// Which engine executes the inverse.
///
/// TD-PSOLA is the default since S146: on the user's own material (炉心融解 bars 28-44 ×
/// 东雪莲, the two rescued phrases) it leaks +0.30 semitones of formant rise where Signalsmith
/// κ=0 leaks +2.40 — and the user picked it in a blind A/B (two versions × two copies, "sort
/// these into two pairs", plus a blank control that came back "sounds the same", correctly).
///
/// The Signalsmith arm stays reachable so the A/B can be re-rendered from one tree
/// (`UTAI_RANGE_ENGINE=signalsmith`). ⛔ That env var is a **rendering affordance for
/// comparisons, not a product setting** — "a temporary measure must never ship as the default"
/// is a rule this line already broke once (S81's D-group external warp).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InverseEngine {
    /// utai_dsp::psola — formants preserved by construction.
    Psola,
    /// The 1.3.2 phase vocoder with engine-native formant compensation (the pre-S146 default).
    Signalsmith,
}

/// `UTAI_RANGE_ENGINE=signalsmith` re-selects the old arm; anything else (including unset) is
/// TD-PSOLA.
pub fn inverse_engine() -> InverseEngine {
    match std::env::var("UTAI_RANGE_ENGINE").as_deref() {
        Ok("signalsmith") => InverseEngine::Signalsmith,
        _ => InverseEngine::Psola,
    }
}

/// ⚙ 出厂默认 = true —— 亚样本搬运 = 开(S157c 翻)
/// S146g — carry the sub-sample transport residual instead of dropping it.
///
/// The measurement is unambiguous (whole-sample transport discards a residual whose RMS is
/// exactly 0 at ratio 1.0 and a flat ≈0.41 samples everywhere else — the shape of the fixed toll
/// we could not explain), and carrying it recovers ~80-85% of that toll on the production
/// caliber. What is NOT settled is whether it sounds better: on the registered fixture ΔHNR
/// ranks the praat gold standard BELOW two arms the user already rejected by ear, and the
/// carrying arm reads ABOVE gold — ΔHNR > 0 means "more periodic than the input", which is what
/// WORLD bought by collapsing unvoiced plosives. ⇒ blind test first, flip after (S146 protocol).
///
/// ⚠⚠ **S157c 翻成了默认 `true`,而上面那段是翻之前写的 —— 两件事都成立,别读串:**
/// S146g 的盲测(8 对承重 + 3 个空白对照,一对都听不出)**是真的**,而且它顺带标定了
/// 一个量级:**~1 dB 级的 ΔHNR 在这条线上不构成可闻收益**。S157c 能重开**不是因为那次判错**,
/// 是因为【量的轴与量级都换了】:那次测 ~1 dB 的 ΔHNR、在窄读窗 + 浅位移的年代;
/// 这次是**基频附近的谐波间噪声**上的 10-16 dB、在 ratio 2.2449 + 宽读窗上,
/// 而且用户先用眼睛报了症状。⇒ ⭐ **「我们试过 X 输了」要连【哪条轴、多大量级、什么条件】
/// 一起记;重开时要证明的是「这一次不在那次的覆盖面里」,不是「那次错了」。**
pub fn frac_transport() -> bool {
    parse_frac_transport(std::env::var("UTAI_PSOLA_FRAC").ok().as_deref())
}

/// The env parse, as a pure function so it can be asserted without touching process state
/// (and so `the_probe_defaults_are_the_production_defaults` can read the default off it).
fn parse_frac_transport(v: Option<&str>) -> bool {
    // ⛔⛔ S157c 翻默认时**必须同时改这里**:旧写法是 `matches!(v, Some("1"))`,任何看不懂的值
    //    都读成 `false`。默认关的时候那是对的(垃圾不许**静默打开**一个未经验的臂),
    //    默认开之后它变成「垃圾值**静默关掉**一个已经上线的修法」—— 用户会拿到一条自己没要求的
    //    旧臂,而且没有任何一行输出会说破。⇒ 三分:明确开、明确关、其余一律**退回默认**。
    //    (S155 在 `parse_infrasonic_hp` 上原样踩过一次。)
    match v.map(str::trim) {
        Some("1" | "true" | "on" | "yes") => true,
        Some("0" | "false" | "off" | "no") => false,
        _ => FRAC_TRANSPORT_DEFAULT,
    }
}

/// ⛔ S157c 翻成 **true**。翻它必须成对 bump `RANGE_ALGO_VERSION` ↔ `audition_cache_tag`。
const FRAC_TRANSPORT_DEFAULT: bool = true;

/// S148 — `UTAI_PSOLA_WSOLA=<frac>` turns on the source-side waveform-similarity search in
/// `utai_dsp::psola`. **Default 0.0 = off = byte-for-byte the pre-S148 arm.**
///
/// Why it exists: measured at +7 st on the akiko donor (production caliber), **4.80 % of voiced
/// frames come out more than 4 dB below the input**, against praat's **0.80 %** on the same input
/// at the same ratio. The window sum is intact where those notches are (`cola_gap` 0.0 %,
/// `cola_w_median` 1.000) ⇒ the level is lost to **grain-to-grain phase cancellation**, and the
/// structural reason is that `max_correlation` is used only when placing the *marks* — the
/// synthesis pass adds every grain blindly. Turning this on takes the notch rate to **0.38 %**
/// (radius 0.15), i.e. below the gold standard.
///
/// ⛔ **Do not flip the default on that number.** The knob is a monotone trade, not a fix:
/// 0 / 0.06 / 0.10 / 0.15 / 0.30 read notch 4.80/3.81/1.92/0.38/0.01 % against ΔHNR
/// −0.15/−1.12/−1.48/−2.23/−4.70 dB, while **praat sits off that curve entirely** (0.80 % *and*
/// +0.27 dB). So there is ~2 dB of headroom that this particular knob cannot reach, and switching
/// it on trades a 4.8 % notch rate for a 2 dB ΔHNR loss — two known quantities whose relative
/// audibility nobody has measured. That is exactly the shape only ears can settle (S146 protocol:
/// blind test first, flip after), and the notch axis has never had an audibility calibration at all.
pub fn wsola_frac() -> f64 {
    parse_wsola_frac(std::env::var("UTAI_PSOLA_WSOLA").ok().as_deref())
}

/// The env parse, as a pure function — same reason as [`parse_frac_transport`].
/// ⚠ Unlike `parse_phase_lock`, an explicit 0 is indistinguishable from "off" here, which is fine
/// only because the default *is* off.
fn parse_wsola_frac(v: Option<&str>) -> f64 {
    v.and_then(|v| v.parse().ok()).filter(|v: &f64| *v > 0.0).unwrap_or(0.0)
}

/// ⚙ 出厂默认 = 0.30 —— 相位锁定 = 开(S150 盲测通过之后翻)
/// S150 — `UTAI_PSOLA_LOCK=<periods>` phase-locks the analysis marks onto the glottal pulses.
/// **Default 0.0 = off = byte-for-byte the pre-S150 arm.**
///
/// This is the fix for the defect the user named from the waveform ("the second half turns into a
/// string of lens shapes" in sustained rescued notes). S148 traced it to a single input: our marks
/// have the **right period and the wrong phase** — spacing median 120.00 samples against praat's
/// 119.75, but scattered ±0.42 of a period inside it, landing where the local energy is 2.3-4.4 dB
/// lower. Feeding praat's marks into our own synthesis reproduced praat's readings to 0.02 dB on
/// 5 notes × 2 metrics, so mark placement is 100 % of the gap. See `utai_dsp::psola::lock_phase`.
///
/// Measured with the lock at **0.30** (goose donor +7, all 23 non-rest notes ≥0.8 s — "modulation
/// this process ADDED", median): today **+2.09 dB**, locked **−0.01**, praat's own marks +0.02.
/// Holds at −7 −5 −2 +1 +3 +5 +7 and on the registered 东雪莲 fixture at +6. The four registered
/// rulers all move toward praat (peak correlation 0.976 → 0.981, ΔHNR −1.58 → −1.34, voiced
/// survival 87.4 → 89.4 %, >4 kHz unchanged).
///
/// ⛔⛔ **The value is a loop gain, not a correction — and that took two rejected arms to learn.**
/// The user's ear killed both of the obvious formulations, each with its own artifact:
/// *correcting the marks afterwards* clicks (the bounded correction sawtooths across the search
/// window: coherent 10 Hz modulation peak, 44-74× the median), and *snapping greedily* roughens
/// (the near-tied peak choice alternates: spacing lag-1 −0.538, transient flux 50.4 vs 20.6).
/// ⭐ The user also diagnosed *why* the un-locked engine sounded smoother than either: its
/// scattered phase was **dithering** the seam. See `utai_dsp::psola::LOCK_BETA`.
///
/// ⛔ **Why the rulers alone could not promote it**(S150 之前;它已经在盲测通过之后翻成默认)。
/// The rulers cannot promote it — that is the whole lesson
/// of S148: WSOLA read 4.80 % → 0.38 % on the ruler it was built for and was 3/3 rejected by ear
/// (it was manufacturing an octave-down subharmonic that the ruler counted as a repair). The
/// audibility scale for THIS axis has exactly one data point (S148 u1: ~2.7 dB heard, ≤0.46 dB
/// not, from a single load-bearing group, p = 0.5). ⇒ blind test first, flip after (S146 protocol).
pub fn phase_lock() -> f64 {
    parse_phase_lock(std::env::var("UTAI_PSOLA_LOCK").ok().as_deref())
}

/// The env parse, as a pure function so it can be asserted without touching process state.
/// ⚠ Unlike `wsola_frac`, an explicit **0 turns it OFF** — the A/B that promoted this default
/// needs a way to render the old arm from the same binary, and "the knob only goes on" is how you
/// end up unable to reproduce the arm a user is complaining about.
fn parse_phase_lock(v: Option<&str>) -> f64 {
    v.and_then(|v| v.parse().ok())
        .filter(|v: &f64| v.is_finite() && *v >= 0.0)
        .unwrap_or(PHASE_LOCK_DEFAULT)
}

/// S150 — **ON by default since the blind test passed** (user 2026-08-17: "翻吧").
///
/// The protocol this satisfies is S146's, and it is the only thing that may promote a default on
/// this line: **a blind test that passes, not a ruler**. Three rounds, same listener, same
/// protocol, and the first two are why this number is 0.30 and not something else:
/// * **v1** — correct-the-marks-afterwards with a *median* smoother: rejected, "clicks".
/// * **v2** — same shape with a real low-pass: rejected, "continuous now, but short seams", and
///   the listener also named the cause (the un-locked engine's scattered phase was *dithering*
///   the seam). Measured: a coherent 10 Hz sawtooth from the bounded correction.
/// * **v3** — this arm (a phase-locked loop, β = 0.1): **2/2 load-bearing groups preferred it,
///   both controls called correctly (one of them bit-identical), and both artifacts gone by ear.**
///
/// ⛔ Honest strength: two load-bearing groups is p = 0.25 by coin flip. The weight comes from
/// four things agreeing — both groups, both controls, a mechanism chain measured end to end, and
/// instruments that agree the two introduced artifacts are gone. ⚠ One prediction did NOT land:
/// the group the depth ruler expected to differ most (Q3) came back "no obvious difference" ⇒
/// **a bigger depth delta does not imply audibility**; do not use depth as a linear proxy.
/// ⚠ Still only measured on ONE model (akiko) and one song; yachiyo remains untested (S148 §7③).
const PHASE_LOCK_DEFAULT: f64 = 0.30;

/// ⚙ 出厂默认 = true —— 去次声 = 开(S155 翻)
/// S152 — `UTAI_PSOLA_HP=0` turns OFF the removal of the infrasonic baseline TD-PSOLA
/// manufactures. **ON by default since S155**; `UTAI_PSOLA_HP_MS=<ms>` forces a fixed width
/// instead of the adaptive one.
///
/// See `utai_dsp::psola::PsolaDiagnostics::infrasonic_frac` for the measurement and the
/// mechanism, and `utai_dsp::psola::Infrasonic` for what the width buys.
///
/// ## Why it was off, and what changed (S155)
///
/// Two things blocked it, and neither was taste:
/// 1. ⛔ **It cost the ratio-1.0 identity gate.** `out -= LP(out)` is a linear filter, and a
///    linear filter is never bit-exact — so turning it on would have downgraded
///    `ratio_one_is_the_identity` (the cheapest non-self-certifying gate on this line; it killed
///    three designs in S146) to an epsilon test. S155 changed the removal to the **differential**
///    form `out -= LP(out) − LP(in)`, which is exactly 0 at ratio 1.0 by construction ⇒
///    `ratio_one_is_the_identity_even_with_the_infrasonic_arm_on` asserts `assert_eq!`.
/// 2. ⛔ **The fixed 8 ms width only removed the half nobody can hear.** Measured on the probe
///    (zero render floor), residual against the input's own low band at −14 st: 8 ms leaves
///    **+8.9 dB at 20-50 Hz and +11.6 dB at 50-125 Hz** standing, with the residual peaking at
///    **71 Hz**. That leftover is what the user reported on 2026-08-19 as
///    「很多地方的 200 以下还造出了极低频的伪影」, with his own negative control attached
///    (the donor singing the same words 14 semitones lower is clean down there).
///    S155's width is one period of the lowest fundamental **in each voiced island** ⇒ on the
///    whole song the median widths are 1.98 / 3.09 / 2.70 ms at −9 / −12 / −14.
/// 3. ⛔ **The low-pass itself was too sloppy.** Two box passes have a **−27 dB first sidelobe**,
///    and the rescued note's own fundamental lands in it ⇒ the differential form's `+LP(in)` put
///    **the donor's pitch back into the output**, which the user reported as 「合唱感」.
///    Four passes (−53 dB) fixed it; the width rule did not change.
///
/// ## What promotes it
///
/// ⛔ Not a ruler — this line's rule is that only ears may promote a *quality* default. This one
/// is promoted as a **correctness** change plus an explicit instruction: the user named it on
/// 2026-08-19 (「次声应该得去」) on evidence he read off the **waveform** — it is the only arm
/// that puts the waveform back near the centre line, and the arms without it drift visibly at
/// e.g. 4:05. The measurements above say what it does; they are not what says to ship it.
pub fn infrasonic() -> utai_dsp::psola::Infrasonic {
    use utai_dsp::psola::Infrasonic;
    if !parse_infrasonic_hp(std::env::var("UTAI_PSOLA_HP").ok().as_deref()) {
        return Infrasonic::Off;
    }
    match parse_infrasonic_ms(std::env::var("UTAI_PSOLA_HP_MS").ok().as_deref()) {
        ms if ms > 0.0 => Infrasonic::FixedMs(ms),
        _ => Infrasonic::PerPeriod,
    }
}

/// ⚙ 出厂默认 = 0.0 —— 0 = 用自适应宽度,不写死
/// S155 — `UTAI_PSOLA_HP_MS=<ms>` pins the cut to a fixed width. **0 = adaptive** (the default).
///
/// ⛔ It exists so an older arm can be rendered from the same binary — "the knob only goes one
/// way" is how you end up unable to reproduce the arm a user is complaining about (S150).
pub fn infrasonic_fixed_ms() -> f64 {
    parse_infrasonic_ms(std::env::var("UTAI_PSOLA_HP_MS").ok().as_deref())
}

/// The env parse, as a pure function so it can be asserted without touching process state.
fn parse_infrasonic_ms(v: Option<&str>) -> f64 {
    v.and_then(|v| v.trim().parse().ok())
        .filter(|v: &f64| v.is_finite() && *v >= 0.0 && *v <= 200.0)
        .unwrap_or(INFRASONIC_MS_DEFAULT)
}

/// The env parse, as a pure function so it can be asserted without touching process state.
///
/// ⛔⛔ **翻默认把一个失败方向翻了过来,而旧的写法没跟着翻。**旧版是
/// `matches!(s.trim(), "1"|"true"|"on"|"yes")`,任何看不懂的值都读成 `false`:
/// 默认关的时候那是对的(垃圾不许**静默打开**一个未经耳判的臂),默认开之后它变成了
/// 「垃圾**静默关掉**一个已经上线的修法」—— 用户会拿到一条自己没要求的旧臂,而且没有任何
/// 一行输出会说破。⇒ 现在三分:明确的开、明确的关、其余一律**退回默认**。
fn parse_infrasonic_hp(v: Option<&str>) -> bool {
    match v.map(str::trim) {
        Some("1" | "true" | "on" | "yes") => true,
        Some("0" | "false" | "off" | "no") => false,
        _ => INFRASONIC_HP_DEFAULT,
    }
}

/// ⛔ Changing this changes the audio ⇒ it must bump `RANGE_ALGO_VERSION` **and**
/// `audition_cache_tag` in the same commit (S150: missing one of the pair makes the user hear a
/// stale cache and read it as "the change did nothing"). S155 flipped it false → **true** and
/// bumped `s154a` → `s155a` in the same commit.
const INFRASONIC_HP_DEFAULT: bool = true;

/// 0 = the adaptive width (one period of the lowest fundamental **per island**).
/// See [`infrasonic_fixed_ms`].
const INFRASONIC_MS_DEFAULT: f64 = 0.0;

/// ⚙ 出厂默认 = 1.0 —— 颗粒插值 = 开(S156 翻)
/// S155 — `UTAI_PSOLA_WIN=<periods>` widens TD-PSOLA's **read window** to that many source
/// periods per side. **0 = off = byte-for-byte the pre-S155 arm**, which reads
/// `min(T_out, T_src)` per side, i.e. `2/ratio` source periods in total (measured per grain:
/// −9 st → 1.189, −12 → **1.000**, −14 → **0.891**). Textbook TD-PSOLA is ±1 period = 2.000.
///
/// ## What it is for
///
/// The user's second defect, reported 2026-08-19: 「高音(ぴゃ)的高次共振峰坍塌」.
/// TD-PSOLA's whole selling point is that formants survive by construction — it moves time-domain
/// segments — so the falsifiable claim is **the output's spectral envelope should equal the
/// donor's**. It does not, and the loss is in exactly the bands he named. Cepstrally-smoothed
/// envelope contrast, measured against the donor (⭐ the ruler's zero is verified: on the ratio-1.0
/// arm, where `out ≡ x` bit-for-bit, it reads 0.000 in every band):
///
/// | | 2-4 kHz | 4-6 kHz | 6-8 kHz | 8-12 kHz |
/// |---|---|---|---|---|
/// | today, ratio 2.0 | −0.834 | −0.666 | −0.606 | −0.866 |
/// | half-width 1.0 period | **−0.360** | **−0.209** | **−0.189** | **−0.277** |
/// | today's width + divide by wsum | −0.949 | −0.673 | −0.577 | −0.871 |
///
/// ⭐ That last row isolates the degree of freedom: **dividing by wsum does nothing on its own —
/// it is the width.** And it is an intervention on the same audio, the same grains and the same
/// ratio, so it is a mechanism, not the correlation `formant.py` first found across arms.
///
/// ## ⛔ Why it is off by default
///
/// It is a **trade, not a win**: on the same three arms the inter-harmonic noise in 300-2500 Hz
/// goes from −7.77 / −2.78 / −6.40 dB to −7.05 / **−0.86** / −5.33, i.e. 1-2 dB worse — and
/// 300-2500 Hz is the band the "click" lived in. S154 measured the same direction independently.
/// ⇒ this is exactly the shape only ears can settle, and this line's rule is that only a blind
/// test (or the user's own spectral analysis on a whole-song render) may promote such a default.
/// S156 —— `UTAI_PSOLA_XGRAIN=<0..1>`:颗粒**内容**在相邻两个源脉冲之间的插值深度。
/// **0 = 今天 = 最近邻 `k = round(u)` = 逐位不变**;1 = 完全线性插值。
///
/// ## 它修的是什么
///
/// `k = round(u)` 是一条**阶梯**:上移时相邻若干颗输出颗粒读**同一个源标记**(ratio 2.0 时
/// 正好成对)⇒ 输出带着周期 `2·T_out = T_src` 的结构 ⇒ **donor 自己的音高**出现在 `0.5·f_out`。
/// 那正是用户在 S155 笔5 亲耳听成**「合唱感」**的那一维。
/// 探针实测(s12 = 位移 +12,`0.5·f_out` 带能量相对各自 400-4000 Hz):
/// donor 输入自己(结构地板)−45.5 · 今天 −37.9 · **宽读窗 −33.6** · 宽读窗 + xgrain **−38.7** ·
/// 今天 + xgrain −42.6。⇒ 宽读窗自己带来 +4.3 dB,而这个旋钮拿掉 5.1 dB。
///
/// ## ⛔ 它是一笔取舍(⚠ S156 已经翻成默认开 —— 下面这段是翻之前写的理由,保留)
///
/// 它是**取舍**:混合相邻两颗源脉冲同时是一次「跨周期的低通」,同一组读数里
/// 8-12 kHz 相对 300-1 kHz 的倾斜从 −1.07 变成 −1.44(≈0.4 dB 的高频损失)。
/// ⇒ 这种取舍只有耳朵能裁(S146 协议),而这条线的裁法是**整曲渲染交给用户自己做频谱分析**。
/// ⛔ 收益那一面**没有判据盯着**(仓里没有真素材);`psola.rs` 里那条
/// `the_grain_interpolation_is_exactly_a_no_op_when_the_neighbouring_pulses_are_identical`
/// 钉的是**语义**(只许混合、不许多做),不是收益。理由写在那条判据的 doc 里。
pub fn xgrain() -> f64 {
    parse_xgrain(std::env::var("UTAI_PSOLA_XGRAIN").ok().as_deref())
}

/// The env parse, as a pure function so it can be asserted without touching process state.
fn parse_xgrain(v: Option<&str>) -> f64 {
    v.and_then(|v| v.trim().parse().ok())
        .filter(|v: &f64| v.is_finite() && *v >= 0.0 && *v <= 1.0)
        .unwrap_or(XGRAIN_DEFAULT)
}

/// ⛔ Changing this changes the audio ⇒ pair-bump `RANGE_ALGO_VERSION` and `audition_cache_tag`.
///
/// **S156:翻成 1.0**(同上)。它存在的理由是抵消宽读窗带来的 `0.5·f_out` 泄漏,
/// ⚠ 而它的代价落在**所有**被救音上 —— 见 [`xgrain`] 的 doc 末尾那条登记。
const XGRAIN_DEFAULT: f64 = 1.0;

/// ⚙ 出厂默认 = 0 —— LP-PSOLA = 关(S157b 判负,旋钮留着)
/// S157b —— `UTAI_PSOLA_LPC=<order>`:**LP-PSOLA** —— 颗粒搬运挪进**残差域**。
/// `0` = 关 = 今天,逐位不变。
///
/// ## 为什么(机理 + 实测的余量)
///
/// `ratio` 不是整数时相邻两颗输出颗粒读**同一个源标记**却放在不同相位上 ⇒ 被复制的不是一个
/// 脉冲,而是「脉冲 ⊛ 声道冲激响应」的一整条长尾 ⇒ 尾巴之间非相干叠加 ⇒ **谐波之间出现噪声**。
/// 残差几乎就是一串脉冲,复制它是良性的。全文在 `utai_dsp::psola` 的 LP-PSOLA 那一段。
///
/// ⭐ 余量是量出来的(S157b,**真 ぴゃ donor**,f0 659 Hz,同一段音频只改 ratio,
/// 1000-2100 Hz 上 **PSOLA 自己加的**谐波间噪声):ratio 2.0000 **+7.68 dB** ·
/// 2.1189 +10.28 · 2.2449 **+12.53** · 2.3784 +15.40 ⇒ **每个半音约 +2.5 dB**。
/// 而同一场的 2×2 取证:模型在 MIDI 76 上给的 donor 比 78 那档干净 **9.1 dB**(同一带),
/// 这道工序却在 ratio 2.2449 上加了 **12.75 dB** ⇒ **把模型给的好处全还回去了**
/// —— 这正是用户 2026-08-20 看到的「合唱感又回来了、而且不止一条」。
///
/// ⚠ **阶数不是常数是旋钮**:Roebel & Rodet 2005 明写「一旦移调不是整数倍,变换后的声音就带
/// whistling artifacts」,根因是**高音上谐波稀疏 ⇒ 全极点去拟合谐波而不是包络**
/// ⇒ 阶数要按素材扫,别写死。
pub fn lpc_order() -> usize {
    parse_lpc_order(std::env::var("UTAI_PSOLA_LPC").ok().as_deref())
}

/// The env parse, as a pure function — same reason as [`parse_frac_transport`].
/// ⚠ 上限 64:阶数超过它就不再是「包络」了,而且格型的每样本代价与阶数成正比。
fn parse_lpc_order(v: Option<&str>) -> usize {
    match v.map(str::trim) {
        None | Some("") => LPC_ORDER_DEFAULT,
        Some("0") => 0, // 显式关得掉 —— 抱怨某条臂时要能用同一个二进制渲旧臂
        Some(x) => match x.parse::<usize>() {
            Ok(n) if (1..=64).contains(&n) => n,
            _ => LPC_ORDER_DEFAULT,
        },
    }
}

/// ⛔ 见 [`lpc_order`]。翻它必须成对 bump `RANGE_ALGO_VERSION` ↔ `audition_cache_tag`。
const LPC_ORDER_DEFAULT: usize = 0;

/// ⚙ 出厂默认 = false —— `UTAI_PSOLA_EDGEFILL=1` 把**岛边那段交叉淡化补完**。
///
/// ## 它修什么(S159zj 实测,鹅妈妈 +7 × 东雪莲,全曲 **1212 条岛边**)
///
/// `psola.rs` 的 `covered` 边界钉在**第一颗/最后一颗合成标记**上,而窗和要再爬约
/// `win_periods × T_src` 才满。合成分支于是在 `i = c0` 上**突然把干填料整项丢掉**:
/// 岛外 `out = acc + (1−w)·carry`,岛内 `out = acc`。
///
/// | 逆变换 | 岛边条数 | 台阶 `1−w` p50 | ≈ |
/// |---|---|---|---|
/// | +2 | **742** | 0.080 | −22 dB |
/// | +7 | 200 | 0.160 | −16 dB |
/// | +12 | 20 | 0.498 | **−6 dB** |
/// | +14 | 50 | 0.538 | −5.4 dB |
///
/// ⭐ 它是 S156 把 [`WIN_PERIODS_DEFAULT`] 翻成 1.0 带进来的:`win_periods == 0` 时
/// `W̄ = 1` ⇒ 这个台阶**解析地恒为 0**。⇒ 「音头音尾竖直条纹」这一族在 S156 之前
/// 结构上不存在,这是它的一条**可证伪的**出处。
///
/// ## ⛔ 它**不是**「岛内短缺也糊上去」
///
/// `psola.rs` 文件头第 5 条与合成分支旁边的注释都写着:**岛内的窗和短缺是真缺陷,
/// 拿未移调音频盖住它是拍频不是修复。**这一刀只动两端那段由**窗宽定义**造成的爬坡 ——
/// 那一段的语义本来就是交叉淡化,岛外那半边已经在这么做,这里只是把它做完。
/// 三条硬门(`edge_fill` / `win_periods > 0` / **`ratio > 1`**)见 `psola_shift_edge`。
///
/// ## ⚠ 为什么默认先关
///
/// 合成夹具上**证不出它更好**:`pulses` 那种脉冲串里 `carry` 自己就有巨大的样本间跳变,
/// 补进去反而让局部一阶差变大(实测 0.107 vs 0.094)—— **公式连续 ≠ 信号连续**。
/// 换成平滑浊音夹具才读到变小。⇒ 合成夹具只能钉**结构**(逐位不变的三条门 + 改动只落在
/// 两端),**好不好**必须在真素材上量,而那一步用的是
/// 「岛边 vs 岛内的单样本一阶差 / 局部 RMS」,逆变换**前**的 donor 当阴性对照
/// (S159zi 在那把尺子上读到岛边 +2.52 dB 而 donor 侧只有 +0.17)。
/// ⛔ 翻默认要成对 bump [`RANGE_ALGO_VERSION`] 与 `audition_cache_tag`。
pub fn edge_fill() -> bool {
    parse_edge_fill(std::env::var("UTAI_PSOLA_EDGEFILL").ok().as_deref())
}

/// The env parse, as a pure function so it can be asserted without touching process state.
fn parse_edge_fill(v: Option<&str>) -> bool {
    match v.map(str::trim) {
        Some("1" | "true" | "on" | "yes") => true,
        Some("0" | "false" | "off" | "no") => false,
        _ => EDGE_FILL_DEFAULT,
    }
}

/// ⛔ Changing this changes the audio ⇒ pair-bump `RANGE_ALGO_VERSION` and `audition_cache_tag`.
const EDGE_FILL_DEFAULT: bool = false;

/// ⚙ 出厂默认 = 1.0 —— 教科书宽读窗 = 开(S156 翻)
///
/// `UTAI_PSOLA_WIN=<周期数>` —— 颗粒**读**窗的半宽是几个**源**周期。
/// ⛔ 显式 `0` 仍然渲得出旧臂(S156 之前那条 `2/ratio` 的窄窗)。
/// 机理、收益与那笔电平代价见 [`WIN_PERIODS_DEFAULT`] 与 `utai_dsp::psola` 的 `win_periods`。
pub fn win_periods() -> f64 {
    parse_win_periods(std::env::var("UTAI_PSOLA_WIN").ok().as_deref())
}

/// The env parse, as a pure function so it can be asserted without touching process state.
fn parse_win_periods(v: Option<&str>) -> f64 {
    v.and_then(|v| v.trim().parse().ok())
        .filter(|v: &f64| v.is_finite() && *v >= 0.0 && *v <= 4.0)
        .unwrap_or(WIN_PERIODS_DEFAULT)
}

/// ⛔ Changing this changes the audio ⇒ pair-bump `RANGE_ALGO_VERSION` and `audition_cache_tag`.
///
/// **S156:翻成 1.0 = 教科书宽度**(用户 2026-08-20 听完整曲五条臂之后拍板)。
/// 收益与代价见 [`win_periods`] 的 doc;⛔ 显式 `UTAI_PSOLA_WIN=0` 仍然能渲出旧臂。
const WIN_PERIODS_DEFAULT: f64 = 1.0;

/// ⚙ 出厂默认 = 0.0 —— **关**(S159i 曾翻成 3.0,**S159zc 因用户对照 S158 判定为退化而退回**)
/// 非零时那个数是**下限**,真实窗宽逐岛按 donor 周期定(`utai_dsp::psola` 的 `ENV_RESTORE_PERIODS` = 1.5 个周期)。
///
/// ## ⛔⛔ S159zc —— 它为什么被退回(用户 2026-08-22 拿 S158 的产物当对照)
///
/// 用户点名 `TESTING\s158_knivesu_s158_头尾都裁.wav`(同素材:炉心融解 +7 × akiko × 291.1 s)
/// 「听起来比我们现在的产出还自然」。逐项对拍之后:
///
/// | 全曲长时平均谱(按 rms 归一 ⇒ 只比形状),相对 S158 | 60-150 | 150-300 | 3-5k | 5-8k | 8-16k |
/// |---|---|---|---|---|---|
/// | 今天出厂(envfix 3 ms) | **+4.16** | **+5.61** | **+3.13** | +1.99 | +1.79 |
/// | **`UTAI_PSOLA_ENVFIX=0`** | **+0.04** | **+0.32** | **+0.18** | +0.18 | +0.12 |
///
/// ⇒ **关掉它,全曲谱形状逐档回到 S158 的 ±0.32 dB 以内**;开着它是一条真实的全曲染色
/// (用户早先的原话:「像被上了一个很奇怪的 EQ……不饱满,有点电话声」)。
/// ⭐ 频谱图上更直接:用户点名的面状伪影(基频以下 0-1 kHz 的一片亮区)在**开着时存在、关掉时不存在**,
/// 而 S158 那一版也不存在。⇒ **那片「面状伪影」是这把刀造出来的。**
/// ⚠ 机理:它在浊音岛内**跑两遍**、每遍夹 ±[`ENV_RESTORE_CLAMP_DB`] = 12 dB ⇒ 最多把塌陷处抬 **24 dB**,
/// 而被抬的是一段**结构已经坏掉**的信号 —— 连同它的次基频残留与毛刺一起放大。
/// 实测:用户点名的 5 处中位电平 出厂 **−16.62** vs S158 **−29.07** vs `ENVFIX=0` **−29.08 dB**。
///
/// ⛔⛔ **血训:我一度用「关掉它电平掉了 16-22 dB ⇒ 那只是变安静,不算修好」把它筛掉了。**
/// 但用户耳朵认可的 S158 **就在那个更安静的电平上** —— **「更安静」在这里正是【正确】,不是【回避】。**
/// ⇒ 「变安静不算修好」这条规矩只在**没有参照臂**时成立;**一旦有一条耳朵认可的参照,就以参照为准**。
///
/// ⚠ 它原本买到的东西(S159i 登记):音符交界处的塌陷 pre 1.00 → post 4.63 dB,被它按回 0.93/0.95;
/// 实测关掉之后那条轴从 9.85 恶化到 13.21 dB。**这笔代价是真的,但用户耳判压过它**(交界塌陷用户从未点名)。
/// S154 — `UTAI_PSOLA_ENVFIX=<ms>` makes the inverse **keep the amplitude envelope it was given**,
/// inside the voiced islands only. **0 = off = byte-for-byte the pre-S154 arm.**
///
/// ## What it is for
///
/// A pitch transform has no business changing the amplitude envelope, and this one does — see
/// `utai_dsp::psola::PsolaDiagnostics::env_dev_p50_db` for the numbers and the control that makes
/// them mean something (outside the islands the deviation is *exactly* 0.00 dB, because there the
/// output is the input).
///
/// ⭐ The user reported this defect from the **waveform**, twice, before any of our rulers found
/// it: *"波形在进入稳定的长音之前有一个非常突兀的波形尖峰"*. On the probe it is a step at the
/// island start — −0.49 / −1.76 / −6.09 / −0.48 dB across the four islands of the −14 segment —
/// and restoring the envelope offline removes the step from the waveform (median |deviation|
/// 1.14 → 0.24 dB).
///
/// ⛔⛔ **It had been measured and dropped twice** (S152 ruler ⑦ 起音过冲, S153 §4k), both times
/// because the reading could not rank the user's six annotated points. That inference is invalid
/// and it cost this line two sessions: *a ruler failing to rank* is a fact about the measurement,
/// not about the world.
///
/// ⚠ **It is not the whole defect.** The user's position, and the honest one: the envelope step is
/// a *symptom*; it does not by itself explain why the spectrum is messy in the same places. This
/// arm exists so that an arm with the envelope violation removed can be rendered and looked at —
/// if the spectrum is still messy, the two are cleanly separated.
///
/// ⚠ Width is the whole trade-off (see `restore_envelope`): under ~10 ms the corrective gain
/// starts tracking individual source periods and puts the donor's fundamental back in
/// (measured leakage −34.7 dB today → −31.6 / −32.8 / −34.1 / −34.3 at 2 / 5 / 10 / 20 ms).
pub fn env_restore_ms() -> f64 {
    parse_env_restore_ms(std::env::var("UTAI_PSOLA_ENVFIX").ok().as_deref())
}

/// The env parse, as a pure function so it can be asserted without touching process state.
fn parse_env_restore_ms(v: Option<&str>) -> f64 {
    v.and_then(|v| v.trim().parse().ok())
        .filter(|v: &f64| v.is_finite() && *v >= 0.0 && *v <= 200.0)
        .unwrap_or(ENV_RESTORE_MS_DEFAULT)
}

/// ⛔⛔⛔ **S159za 把它退回 0.0(默认关)。S159i 翻开它是错的,理由逐条写在这里。**
///
/// ## 翻开它时我拿的读数,以及那个读数为什么不管用
///
/// S159i 看的是渲染日志里的 `env dev p50`(PSOLA 对包络改动的**中位**):
/// 位移 +6 → 0.789 dB · +7 → 0.912 · +10 → 1.770 · +12 → 2.696 · +15 → 3.277 · **+17 → 3.622**,
/// 而 envfix 把它拉回 **0.113-0.271 dB**。数字很漂亮,方向也对。
///
/// ⛔ 但用户 2026-08-22 听到的是**尾巴**不是中位:唱音内部 **20-70 dB** 的凹陷。
///    而这把刀在结构上**够不着那条尾巴**,三条硬理由都在 `utai_dsp::psola::restore_envelope` 里:
///    ⑴ [`ENV_RESTORE_CLAMP_DB`] = **12.0** —— 校正增益最多只能挪 ±12 dB;
///    ⑵ `ey[i] > floor`(floor = 输入峰值 −60 dB)—— 输出塌得最深的样本**直接跳过、增益取 1.0**;
///    ⑶ `box_average(&raw, half * 4)` —— 校正被平滑到 12-28 ms,而凹陷宽 **2-40 ms** ⇒ 被抹平。
///    ⇒ 它修不了那些坑,却在坑**附近**把增益抬上去(最多 12 dB),把残留的毛刺一起放大。
///
/// ## 实测(S159za 消融扫描,炉心融解 +7 × yachiyo,只改这一个旋钮)
///
/// 判据是两把**在用户 ground truth 上验过阳性**的尺子(`scripts/range_rulers/`):
///
/// | | 用户点名 6 处合计 | 深窗内候选 >6 dB | 泄漏 深窗 p90 | ⛔ 窗外对照 |
/// |---|---|---|---|---|
/// | 出厂(envfix 3 ms) | 92.6 | 149 | −22.98 dB | 1.34 / −21.94 |
/// | **envfix 关** | **76.6** | **143** | **−27.12** | 1.35 / −22.04(**不动**) |
///
/// ⇒ **两族同时改善,而阴性对照纹丝不动** —— 这是这一轮里唯一一条干净的。
/// ⚠ 对照:`VALLEY=0` 的点读更低(54.1),但它把**窗外**也动了(1.34 → 1.18)⇒ 那不是这个缺陷,不算数。
/// ⚠ 对照:`XGRAIN=0` 让深窗候选 149 → **235**、泄漏恶化 **8.2 dB** ⇒ `xgrain` 在扛大梁,别碰。
///
/// ⭐ **留着这把刀本身**(它仍然由旋钮可开、判据仍然钉着它的逐岛性质);
///    退的只是**默认值**。要重新翻开它,先解决上面 ⑴⑵⑶ 那三条结构限制,
///    并且拿**尾巴**(不是中位)的判据重新量。
/// ⚙ 出厂默认 = 0.0(**S159za 退回**;S159i 曾翻成 3.0)—— 那 3 ms 是**下限**,真实窗宽由 donor 周期定
///
/// S159i 起它从「窗宽」降级成「开关 + 下限」:引擎逐岛取
/// `max(这个下限, 1.5 个 donor 周期)`(`utai_dsp::psola` 的 `ENV_RESTORE_PERIODS`)。
/// ⛔ 别再把它当宽度调:真素材上 donor 基频低到 123 Hz(周期 8.1 ms),一个固定 5 ms
/// 在那儿只有 0.62 个周期 —— 正落在把 donor 基频漏回来的那一档。
///
/// ## 为什么翻开(宽度曲线与泄漏读数在引擎那边 `ENV_RESTORE_PERIODS` 的 doc 里)
/// donor 那一路在音符交界处会塌谱形,而 PSOLA 把它**放大成电平坑**:真素材上
/// pre 1.00 dB → post 4.63 dB。逐岛包络还原把它按回 **0.93 / 0.95 dB** = donor 自己那一档,
/// 而 donor 基频泄漏与关掉时**同档**(−48.8 dB),「本来就没坑」的那条交界纹丝不动(0.50 → 0.52)。
///
/// ⛔ Same pairing rule as `INFRASONIC_HP_DEFAULT`: making this non-zero changes the audio ⇒ it
/// must bump `RANGE_ALGO_VERSION` **and** `audition_cache_tag` in the same commit.
const ENV_RESTORE_MS_DEFAULT: f64 = 0.0;

/// ⚙ 出厂默认 = 30.0 —— 桥接清音 30 ms
/// S154 — `UTAI_PSOLA_BRIDGE=<ms>` bridges short unvoiced gaps in the fed f0 so the voiced islands
/// cover the whole rescued note. **0 = off = byte-for-byte the pre-S154 arm.**
///
/// See `utai_dsp::psola::bridge_unvoiced` for the mechanism, the measured leak at the user's
/// annotated notes, and the five independent things it accounts for. Short form: PSOLA only shifts
/// **inside** the islands and passes everything else through **bit for bit**, the fed f0 is zeroed
/// on unvoiced phones, so the island boundary sits exactly on the vowel onset and every rescued
/// note keeps a fragment of un-shifted, 9-14 semitones too low audio at each end, joined across
/// **0.25-3 ms**. That join is a step in every harmonic — a broadband vertical line.
///
/// ⚠ This is the *root* candidate; `UTAI_PSOLA_ENVFIX` addresses the same boundary's **symptom**
/// (the amplitude step) without removing the pitch discontinuity underneath it. If the boundary is
/// really the cause, this knob should do what that one could not.
pub fn bridge_unvoiced_ms() -> f64 {
    parse_bridge_unvoiced_ms(std::env::var("UTAI_PSOLA_BRIDGE").ok().as_deref())
}

/// The env parse, as a pure function so it can be asserted without touching process state.
fn parse_bridge_unvoiced_ms(v: Option<&str>) -> f64 {
    v.and_then(|v| v.trim().parse().ok())
        .filter(|v: &f64| v.is_finite() && *v >= 0.0 && *v <= 500.0)
        .unwrap_or(BRIDGE_UNVOICED_MS_DEFAULT)
}

/// S154 — **ON by default at 30 ms since the user confirmed it (2026-08-19).**
///
/// > 「无论是 30ms 还是 60ms 都无论是听起来还是看频谱都把问题解决了(而且这次也和你那边看到的
/// >  结果对上了);**岛外扩是对的**」
///
/// This is the first fix on this line confirmed by ear **and** on the spectrogram, and it closes a
/// defect the user had been reporting since S152 (vertical line at note onsets / abrupt waveform
/// spike / click). ⭐ It was found by taking his waveform observation literally after two earlier
/// sessions had filed the same observation away as "a ruler could not rank the six marks".
///
/// ## Why 30 and not 60
///
/// Both were confirmed indistinguishable by ear, so the tie is broken on risk, measured whole-song
/// (96 islands, −9 and −14):
/// * **30 ms merges no islands at all** (smallest inter-island gap 100.1 → 20.3 ms); 60 ms merges
///   2; 100 ms collapses 96 → **43**, which would drag neighbouring notes into one rescue.
/// * Residual un-shifted leak in the 60 ms before the worst annotated onset: **14.7 % → 2.8 %**
///   (30) → 0.2 % (60) ⇒ 30 removes ~80 % of it for none of the merge risk.
/// * Away from island edges the change is **phase, not quality**: the waveform is completely
///   different (−0.6 … +5.0 dB relative) while the short-time envelope moves by only
///   **0.02-0.52 dB p50**. ⇒ judge this axis on the ENVELOPE; a waveform residual reads "broken"
///   for something inaudible (it cost two false alarms while this was being built).
///
/// ⚠ Only measured on **akiko × 炉心融解 +7**. yachiyo / 东雪莲 / goose are untested on this arm.
/// ⚠ 45-60 ms is the next notch if the line ever comes back — it removes the remaining ~20 %.
const BRIDGE_UNVOICED_MS_DEFAULT: f64 = 30.0;

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
    apply_inverse_with(inverse_engine(), audio, sample_rate, shift, kappa, fed_f0)
}

/// S159 —— [`apply_inverse`],但只保证 `keep` 那几段**样本**是逆变换过的;窗外原样透传。
///
/// `keep` 空 = 整条 = [`apply_inverse`] = **逐位同今天**。⛔ 默认必须是「整条」而不是「什么都不做」:
/// 一个漏传窗的调用点在前者上只是慢一点,在后者上会**静默产出一整条没被搬回原音高的音频**,
/// 而长度契约、有限性、秒数、每一个计数器全部正常(S147「收益静默减半」的同一形状,只是更贵)。
///
/// ## 谁可以用它
/// 只有**窗外必然被丢掉**的消费者:谱面轨的 donor 遍(`apply_dead_only_windows` 只从 donor 上
/// 切走「窗 ± 余量」几段,`match_levels` 全传 false,归一标量来自 base)。
/// ⛔ 探针那两条(`mg_cvfix_inverse` / `mg_render_cover`)整条输出就是拿去听的东西 —— 不许给窗。
///
/// 传什么见 [`donor_keep_samples`];引擎侧的保证与前置条件见 `utai_dsp::psola::psola_shift_win`。
pub fn apply_inverse_windowed(
    audio: Vec<f32>,
    sample_rate: u32,
    shift: i64,
    kappa: f32,
    fed_f0: Option<(&[f32], usize)>,
    keep: &[(usize, usize)],
) -> Result<Vec<f32>, String> {
    apply_inverse_windowed_with(inverse_engine(), audio, sample_rate, shift, kappa, fed_f0, keep)
}

/// [`apply_inverse`] with the engine named explicitly — the A/B arm and the tests take this door.
/// (Selecting the engine through the process environment inside a test would race the other
/// tests in the same binary.)
pub fn apply_inverse_with(
    engine: InverseEngine,
    audio: Vec<f32>,
    sample_rate: u32,
    shift: i64,
    kappa: f32,
    fed_f0: Option<(&[f32], usize)>,
) -> Result<Vec<f32>, String> {
    apply_inverse_windowed_with(engine, audio, sample_rate, shift, kappa, fed_f0, &[])
}

/// [`apply_inverse_windowed`] with the engine named explicitly — **the single execution point**.
#[allow(clippy::too_many_arguments)]
pub fn apply_inverse_windowed_with(
    engine: InverseEngine,
    audio: Vec<f32>,
    sample_rate: u32,
    shift: i64,
    kappa: f32,
    fed_f0: Option<(&[f32], usize)>,
    keep: &[(usize, usize)],
) -> Result<Vec<f32>, String> {
    if shift == 0 || audio.is_empty() {
        return Ok(audio);
    }
    let k = kappa.clamp(0.0, 1.0);
    let semis = -(shift as f64); // semitones the AUDIO moves = inverse of the model-side shift
    let n = audio.len();
    // One-time engine anchor so an A/B can prove what actually ran (a stale render cache
    // otherwise reads as "no difference").
    static ENGINE_LOG: std::sync::Once = std::sync::Once::new();
    ENGINE_LOG.call_once(|| match engine {
        InverseEngine::Psola => tracing::info!(
            "range-extend: inverse engine = utai-dsp TD-PSOLA (formants preserved by construction)"
        ),
        InverseEngine::Signalsmith => tracing::info!(
            "range-extend: inverse engine = signalsmith-stretch 1.3.2 (native formant control, streaming base) [UTAI_RANGE_ENGINE override]"
        ),
    });

    let mut y = match engine {
        InverseEngine::Psola => {
            // The fed f0 goes in RAW. ⛔ Not through `formant_base_track`: that schedule exists
            // for Signalsmith's per-block analysis (100 ms medians, sticky through unvoiced
            // stretches because a mid-stream 0 would restart its noise-chasing auto-detector).
            // Every one of those properties is wrong here — PSOLA needs the 0s to tell voiced
            // from unvoiced, and it needs the per-frame resolution to seed the mark search.
            let (f0, hop) = fed_f0.ok_or_else(|| "RANGE_INVERSE_NO_PITCH".to_string())?;
            if hop == 0 || !f0.iter().any(|v| *v > 0.0) {
                // Silently handing back un-inverted audio is the one outcome this whole function
                // exists to prevent: it is a wrong-pitched render that nothing downstream can see.
                return Err("RANGE_INVERSE_NO_PITCH".into());
            }
            let frac = frac_transport();
            let wsola = wsola_frac();
            let lock = phase_lock();
            let hp = infrasonic();
            let envfix = env_restore_ms();
            let bridge = bridge_unvoiced_ms();
            let win = win_periods();
            let xg = xgrain();
            let fill = edge_fill();
            let lpc = lpc_order();
            let (out, diag) = utai_dsp::psola::psola_shift_edge(
                &audio,
                sample_rate,
                semis,
                f64::from(k) * semis,
                f0,
                hop,
                frac,
                wsola,
                lock,
                hp,
                envfix,
                bridge,
                win,
                xg,
                lpc,
                keep,
                fill,
            );
            // ⛔⛔ S159 —— 判据是 `islands_seen`(窗过滤**之前**的候选岛数),不是 `islands`。
            // 加了窗之后 `islands == 0` 多了一个**正常**的来源:这一遍的窗全落在休止里。
            // 两者若报成同一条红,「窗算错了」与「模型没给 f0」就分不开 —— 而这条线上
            // 同一条红被判成「假红」已经出过两次(S129 铁律)。
            if diag.islands_seen == 0 {
                return Err("RANGE_INVERSE_NO_PITCH".into());
            }
            if diag.keep_ignored {
                // 降级必须**响**:静默降级 = 收益静默归零而每一个读数都正常(S147 那个形状)。
                tracing::warn!(
                    "range-extend: inverse {semis:+.0} st — WINDOW IGNORED (one of lpc {lpc} / \
                     wsola {wsola} / envfix {envfix} is non-zero, so skipping islands would no \
                     longer be sample-exact inside the window) — running the whole buffer; this \
                     pass gets none of the windowed-inverse speedup"
                );
            }
            if diag.islands == 0 {
                // 不是错误:这一遍的窗没碰到任何浊音岛 ⇒ 没东西要救。原样返回,并且**说出来**。
                tracing::info!(
                    "range-extend: inverse {semis:+.0} st — the window ({:.1}% of the buffer) \
                     touches no voiced island; all {} candidates skipped, returning this pass \
                     unchanged (this is NOT the no-pitch failure)",
                    f64::from(diag.keep_frac) * 100.0,
                    diag.islands_seen
                );
                return Ok(audio);
            }
            // ⭐ info!, not debug!: when a bad render is reported, these are the numbers that say
            // whether the inverse did its job — and at debug level they were absent from every
            // log we have ever been handed. `transport_residual` is the S146g readout: 0.000 at
            // ratio 1.0, ≈0.41 for whole-sample transport, 0 once it is carried.
            // ⚠⚠ S151 —— **这一行的统计样本大半可能来自数字静音**,别把它读成「被救的那段音频健康」。
            // S147 之后 donor 只渲相交 chunk、其余铺零(`score2svc.rs` 的 `audio.resize(.., 0.0)`);
            // 去 DC 把零变成常数偏置,而 `analysis_marks` 的相关搜索在常数上处处相关 = 1.0,
            // 走位照样一路铺标记。实测(S151 侦察,`pymarks.py` 与 Rust 逐位等价):把一个 4.06 s
            // 的浊音岛整段铺零,该岛标记数 1363 → **1439**(比真音频还密),间距恒为 90.00 样本。
            // ⇒ `marks` / `cola_*` / `w_p01` 在 `skipped` 高的那几遍里主要是零区的统计。
            // ⚠ **音频本身没错**(颗粒仍从原始 x 上切,零区仍出零)—— 坏的是仪器。
            // 修它要么在零区跳过铺标记(会动输出),要么把两类样本分开计数;两者都还没做,
            // 所以先把这条限制写在读数旁边,免得下一个人拿它当健康证明。
            // ⭐ `src uncovered` 不吃这条限制:零区的读窗照样相接,贡献 0。
            // S159 —— ⛔ 「省了多少」必须打出来。一个算错的窗(比如只省了 10% 而不是 90%)与一个
            // 算对的窗在**音频上逐位相同**,只有这几个数看得见差别 —— 那正是 S147 B2 那次
            // 「功能正确、收益静默减半」唯一被抓到的方式(当时是 `skipped`)。
            // ⚠ 同时:`islands` / `marks` / `cola_*` / `src uncovered` 现在统计的是**被保留的那批岛**。
            // 它们会变小、会「变好看」—— 那是少做了工序,不是修好了什么,别当健康证明。
            tracing::info!(
                "range-extend: inverse {semis:+.0} st, formant kappa {k:.2}, psola {} islands / \
                 {} marks ({}/{} islands in window, keep {:.1}%{}), cola gap {:.2}% \
                 (w p01/median/p99 {:.3}/{:.3}/{:.3}, over 1.05 {:.2}%), \
                 edge step p50/p90 {:.3}/{:.3} over {} island edges, \
                 src uncovered {:.2}%, infrasonic {:.2}%{}, env dev p50 {:.3} dB{}, \
                 transport residual {:.4}{}, hp gate {:+.1} dB",
                diag.islands,
                diag.marks,
                diag.islands_seen - diag.islands_skipped,
                diag.islands_seen,
                f64::from(diag.keep_frac) * 100.0,
                if diag.keep_ignored { " WINDOW IGNORED" } else { "" },
                diag.cola_gap_frac * 100.0,
                diag.cola_w_p01,
                diag.cola_w_median,
                diag.cola_w_p99,
                diag.cola_over_frac * 100.0,
                // S159zj —— **岛边的干填料台阶**(见 `PsolaDiagnostics::edge_step_p50`)。
                // ⛔ 无条件打:它盯的缺陷今天**就在出厂臂上**,而上面那几个 `cola_*` 是整遍
                //    聚合的分位数,岛边样本被岛长稀释掉 ⇒ 结构上看不见它。
                diag.edge_step_p50,
                diag.edge_step_p90,
                diag.island_edges,
                diag.src_uncovered_frac * 100.0,
                // S152 —— 这一项**无条件**算,所以「今天是什么样」在生产日志里就能读到,
                // 不必等改动打开。打开时再补一句「真的拿掉了多少」。
                diag.infrasonic_frac * 100.0,
                if hp == utai_dsp::psola::Infrasonic::Off {
                    String::new()
                } else {
                    // ⛔ 宽度也要打出来:它现在是从 f0 轨推出来的,一个坏帧就能把它变宽、
                    //    于是收益静默减半而别的读数全绿(S147 那个形状)。
                    format!(
                        " (hp {:.2} ms — removed {:.2} pts)",
                        diag.infrasonic_ma_ms,
                        diag.infrasonic_removed * 100.0
                    )
                },
                // S154 —— 同样**无条件**算:这道工序改了多少振幅包络,今天的生产日志里就读得到。
                diag.env_dev_p50_db,
                if envfix > 0.0 {
                    format!(" (envfix {envfix} ms — pulled back to {:.3} dB)", diag.env_dev_after_db)
                } else if bridge > 0.0 {
                    format!(" (bridge {bridge} ms)")
                } else {
                    String::new()
                },
                diag.transport_residual_rms,
                // ⛔ 打出**真的移了几个颗粒**,不只是「开着」:一个从不移动的搜索会产出逐位
                // 相同的音频,与关掉不可分辨(S147 那次「收益静默减半」的同族)。
                if wsola > 0.0 || lock > 0.0 {
                    // ⛔ 同一条规矩:打出**真的动了几个**,不只是「开着」。
                    format!(
                        " (wsola {wsola} — moved {} grains; phase lock {lock} — moved {} marks)",
                        diag.wsola_moved, diag.marks_locked
                    )
                } else {
                    String::new()
                },
                // S159 —— 去次声总闸的余量。它是「窗内逆变换」唯一一条不结构性的耦合
                // (全缓冲能量比,抽掉几个岛原则上能把它推翻面),所以离翻面有多远必须看得见。
                diag.infrasonic_gate_db
            );
            out
        }
        InverseEngine::Signalsmith => {
            let schedule = fed_f0.and_then(|(f0, hop)| formant_base_track(f0, hop, sample_rate));
            // κ=1 is the plain transpose: skip the formant machinery entirely (zero extra cost).
            let formant = ((1.0 - k) > 1e-3).then(|| utai_stretch::FormantPin {
                semitones: f64::from(k) * semis,
                base_hz: schedule.as_ref().map_or(&[][..], |(t, _)| t.as_slice()),
                base_step: schedule.as_ref().map_or(0, |(_, s)| *s),
            });
            // The base-schedule count is only meaningful when a pin was actually built — printing
            // it under κ=1 (where `formant` is None and the schedule never reaches the engine)
            // made the audit line claim work that did not happen (S145).
            tracing::info!(
                "range-extend: inverse {semis:+.0} st, formant kappa {k:.2}, base schedule {}",
                match (formant.is_some(), schedule.as_ref()) {
                    (true, Some((t, _))) => format!("{} pts", t.len()),
                    (true, None) => "auto-detect".to_string(),
                    (false, _) => "not used (kappa=1)".to_string(),
                }
            );
            utai_stretch::stretch_interleaved(&audio, 1, sample_rate, 1.0, semis, formant)?
        }
    };
    if y.len() != n {
        // Exact-length contract guard, not a fix. ⚠ Because it is here, every downstream
        // "the output length equals the input length" assertion is structurally true — the real
        // length gate is `utai_dsp::psola`'s own unit test, which has no such net under it.
        tracing::warn!(
            "range-extend: engine returned {} samples for {n} — padded/truncated to contract",
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

    /// One second per triple on the score's 50 fps grid. `frames` only ever feeds the
    /// passenger-trim threshold, and every test below that is not ABOUT the trim pins its arm to
    /// `None`, so the value is deliberately uniform — a varied one would look load-bearing.
    fn secs(n: usize) -> Vec<i64> {
        vec![50; n]
    }

    // ── S85 七轮: cover_dead_plan(帧域 dead-only;旧整曲优化器的耳锚精神迁移在此)──

    fn hz(m: f32) -> f32 {
        440.0 * 2f32.powf((m - 69.0) / 12.0)
    }

    #[test]
    fn cover_plan_leaves_singable_material_untouched() {
        // 旧 tier-1/2 契约的继承者:整段都唱得动 ⇒ 零区域 ⇒ 逐位不动、无逆变换。
        let mut f0 = vec![hz(60.0); 2000];
        f0.extend(vec![hz(70.0); 2000]);
        let (jobs, unfix) = cover_dead_plan(&f0, 100.0, &range());
        assert!(jobs.is_empty() && unfix.is_empty());
    }

    #[test]
    fn cover_plan_ignores_phantom_islands() {
        // S62b 耳锚精神:一两帧的倍频误读绝不触发染色(旧机器曾被它拖走 -6/-9)。
        let mut f0 = vec![hz(60.0); 2000];
        f0[900] = hz(95.0);
        f0[901] = hz(95.0);
        let (jobs, unfix) = cover_dead_plan(&f0, 100.0, &range());
        assert!(jobs.is_empty() && unfix.is_empty(), "2 frames << MIN_VIOLATION_MS");
    }

    #[test]
    fn cover_plan_rescues_a_sustained_dead_climax_locally() {
        // 旧「climax 必须得救」耳锚的 dead-only 形态:88 超出 usable(48,84) 的 1 秒高潮
        // 成为一个区域,最小落点=comfort 顶 79(bounds 记录)⇒ 该区域 -9;其余素材零触碰。
        let mut f0 = vec![hz(60.0); 2000];
        f0.extend(vec![hz(88.0); 100]);
        let (jobs, unfix) = cover_dead_plan(&f0, 100.0, &range());
        assert!(unfix.is_empty());
        assert_eq!(jobs, vec![DeadJob { shift: -9, start: 2000, end: 2100 }]);
    }

    #[test]
    fn cover_plan_moves_a_dominantly_low_piece_up() {
        // 旧「+22」耳锚精神:整段低于音域=一个大区域,深移上来是唯一出路,深度理所应当。
        let f0 = vec![hz(30.0); 3000];
        let (jobs, unfix) = cover_dead_plan(&f0, 100.0, &range());
        assert!(unfix.is_empty());
        assert_eq!(jobs.len(), 1);
        let DeadJob { shift: s, start: a, end: b } = jobs[0];
        assert_eq!((a, b), (0, 3000));
        assert_eq!(s, 22, "30 → comfort 底 52 = +22,bounds 落点语义");
    }

    #[test]
    fn cover_plan_bridges_consonant_gaps_inside_one_climax() {
        // 高潮里的清辅音/换气微隙(≤GAP_TOL_MS)不许把一个区域劈成两个。
        let mut f0 = vec![hz(60.0); 1000];
        f0.extend(vec![hz(88.0); 40]);
        f0.extend(vec![0.0; 2]); // 20ms 无声隙
        f0.extend(vec![hz(88.0); 40]);
        let (jobs, _) = cover_dead_plan(&f0, 100.0, &range());
        assert_eq!(jobs.len(), 1, "one bridged region, not two");
        assert_eq!(jobs[0].start, 1000);
        assert_eq!(jobs[0].end, 1082);
    }

    /// S159k ⑴ —— ⛔⛔ **区段的边只许落在清音帧上。**用户 2026-08-21 的真病例
    /// (cover 开扩展「高音破音/炸」)。机理与读数写在 `cover_dead_plan` 里那一段注释上。
    ///
    /// ⚠ 注意上面那 8 条老判据**碰不到这条规则** —— 它们的夹具全程有声,外扩按设计不会发生。
    /// (第一版没有上限时它们**红过**,正是它们抓出「会吞掉整段连唱」。)
    #[test]
    fn a_cover_region_edge_walks_out_to_the_nearest_unvoiced_frame() {
        // 50 帧静音 │ 20 帧 60 │ **5 帧清音** │ 25 帧 60 │ 40 帧 88(死)│ 静音
        let mut f0 = vec![0.0f32; 50];
        f0.extend(vec![hz(60.0); 20]);
        f0.extend(vec![0.0; 5]);
        f0.extend(vec![hz(60.0); 25]);
        f0.extend(vec![hz(88.0); 40]);
        f0.extend(vec![0.0; 50]);
        let (jobs, unfix) = cover_dead_plan(&f0, 100.0, &range());
        assert!(unfix.is_empty());
        assert_eq!(jobs.len(), 1, "{jobs:?}");
        // 死区本来是 [100,140);边往回走 25 帧撞到 74 那条清音隙 ⇒ 起点 75。
        assert_eq!(jobs[0].start, 75, "边没有走到清音处");
        assert_eq!(jobs[0].end, 140, "尾边本来就贴着静音,不该动");
        // ⭐ 承重:边的**外侧**必须是清音 —— 这才是「缝落在听不见的地方」那句话的内容。
        assert_eq!(f0[jobs[0].start as usize - 1], 0.0, "起点外侧不是清音");
        assert_eq!(f0[jobs[0].end as usize], 0.0, "终点外侧不是清音");
        // ⛔ 阴性对照:外扩不许改变**深度**(它不抬高该段最高音)。
        assert_eq!(jobs[0].shift, -9, "外扩改了落点 —— 那就不是「深度上免费」了");
    }

    /// S159k ⑴ 的另一半 —— **够不着清音就原地不动**(失败方向 = 今天的行为,不更差)。
    #[test]
    fn a_cover_region_edge_stays_put_when_no_gap_is_within_reach() {
        // 60 帧 60(= 600 ms,远超 COVER_EDGE_SEEK_MS 的 300 ms)紧贴死区,中间没有清音。
        let mut f0 = vec![0.0f32; 50];
        f0.extend(vec![hz(60.0); 60]);
        f0.extend(vec![hz(88.0); 40]);
        f0.extend(vec![0.0; 50]);
        let (jobs, _) = cover_dead_plan(&f0, 100.0, &range());
        assert_eq!(jobs.len(), 1, "{jobs:?}");
        assert_eq!(jobs[0].start, 110, "够不着清音时不许外扩(那只会白搭一堆乘客)");
        assert_eq!(jobs[0].end, 150);
    }

    /// S159k ⑵ —— **只需要一两个半音的区段,不救。**
    ///
    /// 读数在 [`COVER_MIN_RESCUE_DEPTH`] 的 doc 里:−2 那一档逐帧改善只有 −1.02 dB,
    /// 而该档里 **21.7% 的帧反而变脏**,却照样付两条边界的代价。
    /// ⛔ 光钉「浅的不救」会被「干脆全都不救」满足 ⇒ 阴性对照必须钉「够深的照救」。
    #[test]
    fn a_cover_region_that_only_needs_one_or_two_semitones_is_left_alone() {
        // usable 顶 = comfort 顶 = 84 ⇒ 一个 85 的死区只需要 −1。
        let r = SpeakerRange::bounds((48.0, 84.0), (48.0, 84.0));
        let mut shallow = vec![0.0f32; 20];
        shallow.extend(vec![hz(85.0); 40]);
        shallow.extend(vec![0.0; 20]);
        let (jobs, unfix) = cover_dead_plan(&shallow, 100.0, &r);
        assert!(jobs.is_empty(), "只需要 1 个半音的区段不该进工序:{jobs:?}");
        assert!(unfix.is_empty(), "⛔「太浅所以不救」不是「无解」—— 两者必须分开报");
        // ⛔ 阴性对照:同一份记录、同一个夹具形状,够深的照救。
        let mut deep = vec![0.0f32; 20];
        deep.extend(vec![hz(88.0); 40]);
        deep.extend(vec![0.0; 20]);
        let (jobs2, _) = cover_dead_plan(&deep, 100.0, &r);
        assert_eq!(jobs2.len(), 1, "够深的区段被门槛误伤了:{jobs2:?}");
        assert_eq!(jobs2[0].shift, -4);
        // ⛔ **恰好等于门槛的那一格**:87 需要 −3,必须算「够深」(`>=` 不是 `>`)。
        //    变异实测:少了这一条,把 `>=` 改成 `>` 全绿。
        let mut edge = vec![0.0f32; 20];
        edge.extend(vec![hz(87.0); 40]);
        edge.extend(vec![0.0; 20]);
        let (jobs3, _) = cover_dead_plan(&edge, 100.0, &r);
        assert_eq!(jobs3.len(), 1, "恰好 3 个半音必须照救:{jobs3:?}");
        assert_eq!(jobs3[0].shift, -3);
    }

    /// S159k ⑴ 的边界情形 —— **同一条岛上的两个死区,外扩之后撞上了必须合并。**
    /// ⛔ 不合并的话拼接器会把同一段贴两次(`apply_dead_only_windows` 按 job 逐个贴)。
    #[test]
    fn two_cover_regions_that_extend_into_the_same_island_are_merged() {
        // 静音 │ 30 帧 88(死)│ 5 帧 60 │ 25 帧 88(死)│ 静音 —— 死帧间隔 5 > GAP_TOL ⇒ 本是两组。
        let mut f0 = vec![0.0f32; 50];
        f0.extend(vec![hz(88.0); 30]);
        f0.extend(vec![hz(60.0); 5]);
        f0.extend(vec![hz(88.0); 25]);
        f0.extend(vec![0.0; 50]);
        let (jobs, _) = cover_dead_plan(&f0, 100.0, &range());
        assert_eq!(jobs.len(), 1, "外扩之后撞上的两段没有合并 ⇒ 拼接器会贴两次:{jobs:?}");
        assert_eq!((jobs[0].start, jobs[0].end), (50, 110));
    }

    /// S159n ⑴ —— **位移相同的相邻段合并**(深度免费)。读数在 [`COVER_MERGE_SAME_SHIFT_MS`] 的 doc 里。
    ///
    /// ⛔ 三件一起钉,少一件就能被「干脆全合并」或者「根本没合并」满足:
    /// ⑴ 同位移 + 间隔够近 ⇒ **合并**;⑵ **位移不同 ⇒ 绝不合并**(否则就是在把浅段拖深,
    ///    而那笔钱没量过);⑶ 同位移但**隔得太远 ⇒ 不合并**(否则门限形同虚设)。
    #[test]
    fn adjacent_cover_regions_with_the_same_shift_are_merged() {
        let r = SpeakerRange::bounds((48.0, 84.0), (48.0, 84.0));
        // 两段都需要 −4(88 → 84),中间隔 40 帧 = 400 ms < 800 ms 门限。
        let mut same = vec![0.0f32; 20];
        same.extend(vec![hz(88.0); 40]);
        same.extend(vec![0.0; 40]);
        same.extend(vec![hz(88.0); 40]);
        same.extend(vec![0.0; 20]);
        let (jobs, _) = cover_dead_plan(&same, 100.0, &r);
        assert_eq!(jobs.len(), 1, "同位移的两段没有合并:{jobs:?}");
        assert_eq!((jobs[0].start, jobs[0].end, jobs[0].shift), (20, 140, -4));

        // ⑵ ⛔ 阴性对照:位移不同(−4 与 −8)⇒ 必须还是两段。
        let mut diff = vec![0.0f32; 20];
        diff.extend(vec![hz(88.0); 40]);
        diff.extend(vec![0.0; 40]);
        diff.extend(vec![hz(92.0); 40]);
        diff.extend(vec![0.0; 20]);
        let (jobs2, _) = cover_dead_plan(&diff, 100.0, &r);
        assert_eq!(jobs2.len(), 2, "位移不同的两段被合并了 —— 那是在把浅段拖深:{jobs2:?}");
        assert_eq!((jobs2[0].shift, jobs2[1].shift), (-4, -8));

        // ⑶ ⛔ 同位移但隔了 1.2 s(> 门限)⇒ 不合并。
        let mut far = vec![0.0f32; 20];
        far.extend(vec![hz(88.0); 40]);
        far.extend(vec![0.0; 120]);
        far.extend(vec![hz(88.0); 40]);
        far.extend(vec![0.0; 20]);
        let (jobs3, _) = cover_dead_plan(&far, 100.0, &r);
        assert_eq!(jobs3.len(), 2, "隔了 1.2 s 还合并 ⇒ 门限形同虚设:{jobs3:?}");
    }

    /// S159p —— ⛔⛔⛔ **合并不许把「没人看过的材料」吞进 donor。**
    ///
    /// 独立审计逐行核出来的两条(我写第一版时漏了护栏,而**谱面轨的同款
    /// `merge_same_shift_across_rests` 本来就有** —— 它靠「只跨休止合并」躲开了这件事):
    /// ⑴ 中间那截的乘客移位之后可能掉出 `slot_reachable` ⇒ 合并做出**计划器本人会判无解的窗**;
    /// ⑵ 中间那截可能正是一段刚被判 `unfixable` 的区间 ⇒ 审计还在打「rendered broken as-is」,
    ///    而拼接器实际按邻居的位移渲了它。
    ///
    /// ⚠ **上面那条 `adjacent_cover_regions_with_the_same_shift_are_merged` 抓不到这两件事** ——
    /// 它的夹具中间是**静音**,只测了安全的那一侧。这条补的就是危险的那一侧。
    #[test]
    fn a_cover_merge_never_swallows_material_no_predicate_has_looked_at() {
        let r = range();
        // ⑴ 中间是**唱得动**但移位后够不着的材料:88 需要 −9,而 50 − 9 = 41 < usable 底 48。
        let mut low = vec![0.0f32; 20];
        low.extend(vec![hz(88.0); 40]);
        low.extend(vec![hz(50.0); 30]);
        low.extend(vec![hz(88.0); 40]);
        low.extend(vec![0.0; 20]);
        let (jobs, _) = cover_dead_plan(&low, 100.0, &r);
        assert_eq!(jobs.len(), 2, "把移位后够不着的乘客吞进了 donor:{jobs:?}");
        for j in &jobs {
            assert!(j.end <= 60 || j.start >= 90, "job {j:?} 盖住了中间那 30 帧");
        }
        // ⛔ 阴性对照:同样的形状、同样的间距,中间换成**移位后够得着**的材料 ⇒ 必须合并。
        //    (没有这一条,上面那条可以由「护栏一刀切、根本不合并」满足。)
        let mut okm = vec![0.0f32; 20];
        okm.extend(vec![hz(88.0); 40]);
        okm.extend(vec![hz(70.0); 30]);
        okm.extend(vec![hz(88.0); 40]);
        okm.extend(vec![0.0; 20]);
        let (jobs2, _) = cover_dead_plan(&okm, 100.0, &r);
        assert_eq!(jobs2.len(), 1, "够得着的乘客也不合并 ⇒ 护栏过紧:{jobs2:?}");

        // ⑵ 中间是一段**无解**区间(96 太高、30 太低,同一组里上下都死 ⇒ 没有落点)。
        //
        // ⚠⚠ **如实登记:这一半没有独立覆盖到「无解」那道检查。**变异「把 `unfixable` 那一问
        //    去掉」在这个夹具上是**绿**的 —— 因为中间那 30 帧在邻居的 −9 上是**够不着**的
        //    (96 − 9 = 87 > usable 顶 84),所以先被 ⑴ 那道可达性拦住了,轮不到 ⑵。
        // ⛔ 试过造「**可达但无解**」的局面(邻居 −6 · 中间 88+56):被**死帧长度门**
        //    (`MIN_VIOLATION_MS` = 25 帧)与 `median5` 一起挡住 —— 要让低音落在死音组**内部**,
        //    它就得短到 ≤3 帧(`GAP_TOL_MS` 的桥接上限),而那个长度会被中值抹掉。
        // ⇒ 「无解」那道检查今天是**防御**,不是活路径;它守的是**审计行的真实性**
        //    (不许一边打『rendered broken as-is』一边把它渲进 donor),一行的代价,留着。
        //    ⛔ 别把这条读成「已经验过了」。
        let mut unf = vec![0.0f32; 20];
        unf.extend(vec![hz(88.0); 40]);
        unf.extend(vec![0.0; 10]);
        unf.extend(vec![hz(96.0); 15]);
        unf.extend(vec![hz(30.0); 15]);
        unf.extend(vec![0.0; 10]);
        unf.extend(vec![hz(88.0); 40]);
        unf.extend(vec![0.0; 20]);
        let (jobs3, unfix3) = cover_dead_plan(&unf, 100.0, &r);
        assert_eq!(unfix3.len(), 1, "夹具没造出无解区间,这一半是空的:{unfix3:?}");
        let (ua, ub) = unfix3[0];
        assert_eq!(jobs3.len(), 2, "合并盖住了一段刚被判无解的区间:{jobs3:?}");
        for j in &jobs3 {
            assert!(
                j.end <= ua || j.start >= ub,
                "job {j:?} 盖住了无解区间 ({ua},{ub}) —— 审计行会说『原样渲坏』而实际不是"
            );
        }
    }

    /// S159n ⑶ —— **按乐句整段救**(`UTAI_COVER_PHRASE_GAP_MS`,出厂 = 0 = 关)。
    ///
    /// ⛔ 这条判据只钉**旋钮关着时行为不变**这一件 —— 那是它今天唯一的承诺。
    /// 旋钮开着的形状(边落在真空档里)由离线台子在真素材上量,不在这里断言:
    /// 净效果**还没量**(今天的区段里根本没有 MIDI 70 以下的素材),
    /// ⛔ 在量出来之前谁也不许把它翻成默认。
    #[test]
    fn the_phrase_mode_knob_is_off_by_default_and_inert() {
        // ⛔ 不读进程 env(会随机器上导出了什么改答案,S150 在 `parse_phase_lock` 上付过这个学费)——
        //    这里钉的是**函数的出厂值**。
        assert_eq!(cover_phrase_gap_ms(), 0.0, "乐句模式的出厂默认必须是关");
    }

    #[test]
    fn cover_plan_counts_unfixable_regions_loudly() {
        // 拖拽守卫:死亡高潮里混着够不着的低音 ⇒ 无解 ⇒ 响亮报位置而非静默跳过。
        let mut f0 = vec![hz(88.0); 30];
        f0.extend(vec![hz(30.0); 30]); // 同一 gap 桥接域内:高低两侧都死
        let (jobs, unfix) = cover_dead_plan(&f0, 100.0, &range());
        assert!(jobs.is_empty());
        assert_eq!(unfix, vec![(0, 60)], "无解区域带位置(审查 S85d:取证要「在哪」)");
    }

    #[test]
    fn cover_plan_jobs_feed_the_splicer_with_shifts_not_frames() {
        // S85d 实机翻车的钉子:两个生产者曾用不同元组顺序,拼接器把死区起始帧(7512)当
        // 移调渲 donor(日志「+7512 st」,被 S82 FFI 守卫响亮拦截)。计划→拼接直连断言。
        let mut f0 = vec![hz(60.0); 2000];
        f0.extend(vec![hz(88.0); 100]);
        let (jobs, _) = cover_dead_plan(&f0, 100.0, &range());
        assert_eq!(jobs.len(), 1);
        let mut base = vec![0.5f32; 1_008_000]; // 2100 帧 @48k
        let mut seen = Vec::new();
        apply_dead_only_windows(&mut base, 48000, 2100, &jobs, false, |s, _own| {
            seen.push(s);
            Ok(vec![0.5f32; 1_008_000])
        })
        .unwrap();
        assert_eq!(seen, vec![-9], "donor 闭包收到的必须是移调半音,绝不是帧号");
    }

    #[test]
    fn cover_plan_min_run_counts_dead_frames_not_span() {
        // 审查 S85d:5 个 3 帧幻影爆点以 ≤40ms 间隙相连 = 跨度 ≥250ms 但浊死帧仅 150ms
        // ——门量死帧数,假区必死;真高潮(死帧本身 ≥250ms)不受影响。
        let mut f0 = vec![hz(60.0); 1000];
        for _ in 0..5 {
            f0.extend(vec![hz(95.0); 3]);
            f0.extend(vec![hz(60.0); 4]); // 活帧桥接域内夹层
        }
        f0.extend(vec![hz(60.0); 1000]);
        let (jobs, unfix) = cover_dead_plan(&f0, 100.0, &range());
        assert!(jobs.is_empty() && unfix.is_empty(), "phantom bursts must never recolour");
    }

    // ── 共享拼接器(两轨+audition 唯一执行点;审查 S85d:搬家丢测已补钉)──

    #[test]
    fn dead_only_splice_blends_only_the_windows() {
        // 窗外逐位不动、窗心=donor、缘上 10ms 余弦 ramp 半程=0.5。
        // base 全零 ⇒ active_rms=None ⇒ 电平匹配跳过(不许把噪声放大成信号)。
        let mut base = vec![0.0f32; 48000];
        let donor = vec![1.0f32; 48000];
        let mut calls = 0usize;
        apply_dead_only_windows(&mut base, 48000, 50, &[DeadJob { shift: -6, start: 10, end: 20 }], true, |s, _own| {
            calls += 1;
            assert_eq!(s, -6);
            Ok(donor.clone())
        })
        .unwrap();
        assert_eq!(calls, 1, "one donor render per DISTINCT shift");
        let a = (10.0 / 50.0 * 48000.0) as usize;
        assert_eq!(base[0], 0.0);
        assert_eq!(base[a - 1], 0.0, "窗外逐位不动");
        assert!((base[a + 480] - 1.0).abs() < 1e-6, "ramp 结束=donor");
        assert!((base[a + 240] - 0.5).abs() < 0.01, "10ms 余弦半程");
        assert!((base[(0.3 * 48000.0) as usize] - 1.0).abs() < 1e-6, "窗心=donor");
        assert_eq!(base[48000 / 2], 0.0, "窗后回到 base");
    }

    /// ⛔⛔ S147 那次 hotfix 欠的判据 —— 它当时**一条都没有**,所以缺陷是靠我顺手打进 `[perf]`
    /// 的一个数字(「四个位移的 `skipped` 完全相同」)才暴露的。
    ///
    /// 缺陷形状:donor 打洞的第一版把**全部位移的窗的并集**传给每一遍 ⇒ 四个 donor 渲同一批
    /// chunk。**功能正确、收益静默减半**;单元测试、长度契约、音频秒数**全部正常**。
    /// 那个 filter 现在只存在于 `apply_dead_only_windows` 内部一处,而这条判据守住它。
    ///
    /// 两条断言缺一不可:
    /// ⑴ 不同的位移必须拿到**不同**的窗 —— 否则「传并集」那种写法照样过;
    /// ⑵ **阴性对照**:窗本来就相同的两个位移必须拿到**相同**的答案 —— 否则 ⑴ 有可能是靠
    ///    「总是给点不一样的东西」蒙过去的,而那种实现同样是错的。
    /// ⚠ 期望值写**字面量**,不是从 `jobs` 现算(S146f 一场犯四次:把被测的东西写进断言 = 恒真)。
    /// 拼接的**独立参照实现**(照 S151 的语义直写:位移序、窗内 10 ms 余弦、
    /// 缓冲区边界不淡化)。⛔ 它存在的唯一理由是 S152 把拼接拆成了两阶段 ——
    /// 「重构没改行为」这句话必须由一份**不是从被测代码抄来**的实现来证。
    fn reference_splice(
        base: &mut [f32],
        spf: f64,
        xf: usize,
        jobs: &[DeadJob],
        donor_of: impl Fn(i64) -> Vec<f32>,
    ) {
        let mut shifts: Vec<i64> = jobs.iter().map(|j| j.shift).collect();
        shifts.sort_unstable();
        shifts.dedup();
        for s in shifts {
            let donor = donor_of(s);
            let n = base.len().min(donor.len());
            for j in jobs.iter().filter(|j| j.shift == s) {
                let a = ((j.start.max(0) as f64 * spf) as usize).min(n);
                let b = ((j.end.max(0) as f64 * spf) as usize).min(n);
                let xfw = xf.min(b.saturating_sub(a) / 2);
                if b <= a || xfw == 0 {
                    continue;
                }
                let (fi, fo) = (a > 0, b < n);
                for k in a..b {
                    let w = if fi && k < a + xfw {
                        0.5 - 0.5 * (std::f32::consts::PI * (k - a) as f32 / xfw as f32).cos()
                    } else if fo && k >= b - xfw {
                        0.5 - 0.5 * (std::f32::consts::PI * (b - k) as f32 / xfw as f32).cos()
                    } else {
                        1.0
                    };
                    base[k] = base[k] * (1.0 - w) + donor[k] * w;
                }
            }
        }
    }

    #[test]
    fn the_two_phase_splice_is_byte_for_byte_the_one_pass_one_it_replaced() {
        // ⭐ S152 的承重判据。拼接从「渲一条→立刻拼→丢掉」改成「全部渲完→统一拼」,
        // 唯一的理由是窗边要由**两侧的 donor** 一起决定;而在那把刀关着的时候,
        // 输出必须**逐位**不变。⛔ 参照实现是独立写的,不是从被测代码抄的。
        let sr = 48_000u32;
        let total = 400i64;
        let n = 400 * 480usize; // spf = 480
        let mk = |seed: u32, f: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let t = i as f32 / sr as f32;
                    ((t * f + seed as f32 * 0.37).sin() * 0.4 + (t * f * 2.1).sin() * 0.2) as f32
                })
                .collect()
        };
        let jobs = vec![
            DeadJob { shift: -14, start: 20, end: 60 },
            DeadJob { shift: -9, start: 63, end: 120 },
            DeadJob { shift: -2, start: 150, end: 200 },
            DeadJob { shift: -9, start: 260, end: 330 },
            DeadJob { shift: -14, start: 333, end: 399 },
        ];
        let donor_of = |s: i64| mk((s.unsigned_abs() as u32) * 7 + 1, 300.0 + s as f32 * 11.0);

        let mut got = mk(0, 220.0);
        apply_dead_only_windows(&mut got, sr, total, &jobs, false, |s, _own| Ok(donor_of(s)))
            .unwrap();

        let mut want = mk(0, 220.0);
        reference_splice(&mut want, n as f64 / total as f64, (sr as usize / 100).max(2), &jobs, donor_of);

        assert_eq!(got.len(), want.len());
        let bad = got.iter().zip(&want).position(|(a, b)| a != b);
        assert!(
            bad.is_none(),
            "两阶段拼接与参照实现在样本 {:?} 就分开了(got {:?} want {:?})",
            bad,
            bad.map(|i| got[i]),
            bad.map(|i| want[i])
        );
    }

    /// ⭐ 46 s 那处的**真形状**,三条判据共用一份。
    ///
    /// 谱:窗 L = 帧 20..100(样本 9600..48000)· 休止 · 窗 R = 帧 110..380(52800..)。
    /// 休止在 base 里是 47040..54000(= `l.end − post`,窗本来就伸进休止 2 帧)(⚠ **比 R 的窗起点还长** —— 今天 `pre = 4 帧` 让窗在
    /// 音前 80 ms 就开了,这正是缺陷的几何)。
    /// * `base` 在休止里有内容(残响 / 辅音预卷,−26 dBFS),别处很响。
    ///   ⚠ 它取**负值**:RMS 不看符号,但「淡化淡回了 base」会在输出里留下负号
    ///   —— 那是「接上时仍然淡出到 base」这条变异唯一抓得住的把柄(正值夹具上它全绿)。
    /// * **右 donor 在 53760 之前是静音** ⇒ 今天 52800 开窗就把 base 那 960 样本(20 ms)
    ///   换成一个洞;
    /// * 左 donor 在整段休止里都有内容(−24 dBFS)⇒ 把切换点挪到 53760 之后,洞就没了。
    /// ⚠ 53760 落在打分格(10 ms)的边界上,这是**故意**的:洞若不占满整格,判据的分辨率
    ///   就读不出它 —— 那条限制本身写在 `the_rest_join_moves_...` 的断言里。
    fn join_fixture(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let base = (0..n)
            .map(|i| if (47_040..54_000).contains(&i) { -0.05f32 } else { -1.0 })
            .collect();
        let left = (0..n)
            .map(|i| if (47_000..56_000).contains(&i) { 0.06f32 } else { 0.5 })
            .collect();
        let right = (0..n)
            .map(|i| if (46_000..53_760).contains(&i) { 0.0f32 } else { 0.25 })
            .collect();
        (base, left, right)
    }

    fn join_jobs() -> Vec<DeadJob> {
        // ⛔ 左窗位移 −9、右窗 −14 —— 时间序与位移序**相反**。同向的话「拼接按位移序」
        // 那条变异是等价的(实测全绿),而顺序错了正好会让淡化淡回 base。
        vec![
            DeadJob { shift: -9, start: 20, end: 100 },
            DeadJob { shift: -14, start: 110, end: 380 },
        ]
    }

    #[test]
    fn the_rest_join_moves_the_switch_to_where_the_hole_stops_and_refuses_when_there_is_no_hole() {
        // ⭐ 这一刀的核心判据。⛔ 判据本身被自己的两版口径判负过两次,都记在 `join_rests` 的
        // 文档里:①「两侧电平差最小」+ 20 ms 窗 —— 漏掉了 46 s 那 15 ms 的真数字静音;
        // ②「切在最安静处」—— 它会把洞**加深**(46 s 那处它选 46.05,洞 24.5 dB)。
        let sr = 48_000u32;
        let total = 400i64;
        let n = 400 * 480usize;
        let spf = spf_of(n, total);
        let xf = (sr as usize / 100).max(2);
        let jobs = join_jobs();
        let (base, left, right) = join_fixture(n);
        let kept: Vec<(i64, usize, Vec<f32>)> = vec![(0, 0, left.clone()), (1, 0, right.clone())];

        // 关着 ⇒ 一条都不接(默认,也是「生产逐位不变」的来源)
        assert!(
            join_rests(&base, sr, spf, &jobs, &kept, &[0, 1], xf, false).is_empty(),
            "默认必须什么也不做"
        );

        let on = join_rests(&base, sr, spf, &jobs, &kept, &[0, 1], xf, true);
        let t = *on.get(&0).expect("今天在休止里挖了个洞,挪一挪就能补上 ⇒ 必须接");
        // ⚠ 打分格是 10 ms,所以分辨率到此为止:它挪过洞的主体(52800→53280)而不是精确
        // 落在 53500。这条限制写在这里,免得下一个人把 `t == 53500` 当成期望值。
        // ⚠ 分辨率:打分格 10 ms,而跨切换点那一格不计分 ⇒ 它挪过洞的主体而不是精确到边界。
        // 这条限制写在这里,免得下一个人把某个精确值当期望。
        assert!(
            t >= 53_280,
            "切换点 {t} 必须挪过那个洞的主体(今天是 52800 就开窗,洞是 52800-53760)"
        );

        // ⛔ 阴性对照 ①:右 donor 在整段休止里都有内容 ⇒ 今天就没有洞 ⇒ **不许动**。
        let ok_right: Vec<f32> = (0..n).map(|i| if i < 46_000 { 0.25f32 } else { 0.06 }).collect();
        assert!(
            join_rests(&base, sr, spf, &jobs, &[(0, 0, left.clone()), (1, 0, ok_right)],
                       &[0, 1], xf, true).is_empty(),
            "今天已经没有洞了就不许动(离线扫的 128.9 s 就是这一类:今天 1.12 dB,接上反而 6.41)"
        );
        // ⛔ 阴性对照 ②:左 donor 在休止里也是静音 ⇒ 挪过去补不上 ⇒ **不许动**。
        let dead_left: Vec<f32> = (0..n).map(|i| if i < 47_000 { 0.5f32 } else { 0.0 }).collect();
        assert!(
            join_rests(&base, sr, spf, &jobs, &[(0, 0, dead_left), (1, 0, right)],
                       &[0, 1], xf, true).is_empty(),
            "两条 donor 都补不上那个洞时必须放弃"
        );
    }

    #[test]
    fn joining_a_rest_puts_no_base_in_the_crossfade() {
        // 接上之后那一次淡化的两侧**都是 donor**;base 一个样本都不许参与 ——
        // 否则这一刀又把它本来要消灭的那个洞画了回去(只是窄了一点)。
        let sr = 48_000u32;
        let total = 400i64;
        let n = 400 * 480usize;
        let spf = spf_of(n, total);
        let xf = (sr as usize / 100).max(2);
        let jobs = join_jobs();
        let (base, left, right) = join_fixture(n);
        let kept: Vec<(i64, usize, Vec<f32>)> = vec![(0, 0, left), (1, 0, right)];

        let mut off = base.clone();
        splice_kept(&mut off, sr, spf, &jobs, &kept, xf, false).unwrap();
        let mut on = base.clone();
        splice_kept(&mut on, sr, spf, &jobs, &kept, xf, true).unwrap();

        // ⓐ 关着 ⇒ 那个洞必须还在:52800 开窗之后是右 donor 的静音。
        assert!(
            off[53_300..53_700].iter().all(|v| v.abs() < 1e-6),
            "关着的时候这一段必须是右 donor 的静音(那就是用户听到的洞)"
        );
        // ⓑ ⭐ 打开 ⇒ 今天放 base 的那一段改由**左 donor** 铺满(0.06 而不是 0.05)。
        assert!(
            off[49_000..52_700].iter().all(|v| (*v + 0.05).abs() < 1e-6),
            "阴性对照:关着的时候这一段是 base"
        );
        let bad: Vec<usize> =
            (49_000..52_900).filter(|&k| (on[k] - 0.06).abs() > 1e-3).collect();
        assert!(
            bad.is_empty(),
            "这一段还剩 {} 个样本不是左 donor(第一个在 {:?},值 {:?})",
            bad.len(),
            bad.first(),
            bad.first().map(|&k| on[k])
        );
        // ⓒ 洞必须**变短**(不必消失:切换点受打分格 10 ms 的分辨率限制;
        // 这份夹具实测 481 → 361 个静音样本,即 10 ms 的 960 样本洞缩到 7.5 ms)。
        let zeros = |x: &[f32]| x[52_800..53_760].iter().filter(|v| v.abs() < 1e-6).count();
        assert!(
            zeros(&on) + 100 < zeros(&off),
            "洞没怎么变短:关 {} 个静音样本 → 开 {} 个",
            zeros(&off),
            zeros(&on)
        );
        // ⓓ base 不许出现在切换区里 —— 它是负的,两条 donor 都是正的。
        assert!(
            on[48_500..53_900].iter().all(|v| *v > -1e-6),
            "切换区里出现了 base(负值)⇒ 淡化淡回了 base"
        );
        // ⓔ 而且窗外仍是 base、窗内是 donor(证明上面几条不是恒真的)。
        assert_eq!(on[0], -1.0, "窗外必须仍然是 base");
        assert!((on[30_000] - 0.5).abs() < 1e-6, "左窗内必须是左 donor");
        assert!((on[60_000] - 0.25).abs() < 1e-6, "右窗内必须是右 donor");
    }

    #[test]
    fn the_join_refuses_the_three_things_it_is_not_for() {
        // ⛔ 三条**错误分支**,写完必须真触发一次 —— 一条从没被执行过的分支就是一条空判据
        // (S129)。变异实测:不补这三条,「同位移也接」「不限桥接上限」「越界不夹紧」全绿。
        let sr = 48_000u32;
        let total = 400i64;
        let n = 400 * 480usize;
        let spf = spf_of(n, total);
        let xf = (sr as usize / 100).max(2);
        let (base, left, right) = join_fixture(n);
        let kept: Vec<(i64, usize, Vec<f32>)> = vec![(0, 0, left.clone()), (1, 0, right.clone())];
        let diff = join_jobs();

        // 先钉住阴性对照:这份夹具在异位移下**必须**接得上,否则下面三条都是恒真的。
        assert!(
            !join_rests(&base, sr, spf, &diff, &kept, &[0, 1], xf, true).is_empty(),
            "阴性对照:这份夹具本来就该接得上"
        );

        // ⓐ **同位移**:那是 S151 笔5 在计划层合并的事,这里必须放过。
        let same = vec![
            DeadJob { shift: -9, start: 20, end: 100 },
            DeadJob { shift: -9, start: 110, end: 380 },
        ];
        assert!(
            join_rests(&base, sr, spf, &same, &kept, &[0, 1], xf, true).is_empty(),
            "同位移的一对不归这一刀管"
        );

        // ⓑ **长休止**:桥接上限之外不许接。
        // ⛔ 夹具必须是「不设守卫就一定会接」的那种 —— 否则这条守卫是等价变异(实测全绿)。
        let far = vec![
            DeadJob { shift: -9, start: 20, end: 100 },
            DeadJob { shift: -14, start: 100 + MERGE_BRIDGE_FRAMES + 1, end: 380 },
        ];
        let far_base: Vec<f32> = (0..n)
            .map(|i| if (47_040..62_000).contains(&i) { -0.05f32 } else { -1.0 })
            .collect();
        let far_kept: Vec<(i64, usize, Vec<f32>)> = vec![
            (0, 0, (0..n).map(|i| if (47_000..63_000).contains(&i) { 0.06f32 } else { 0.5 }).collect()),
            (1, 0, (0..n).map(|i| if (46_000..61_000).contains(&i) { 0.0f32 } else { 0.25 }).collect()),
        ];
        // 先证它「本来会接」:把同一份夹具的间隙缩到上限之内。
        let near = vec![
            DeadJob { shift: -9, start: 20, end: 100 },
            DeadJob { shift: -14, start: 100 + MERGE_BRIDGE_FRAMES, end: 380 },
        ];
        assert!(
            !join_rests(&far_base, sr, spf, &near, &far_kept, &[0, 1], xf, true).is_empty(),
            "阴性对照:同一份夹具在上限之内必须接得上"
        );
        assert!(
            join_rests(&far_base, sr, spf, &far, &far_kept, &[0, 1], xf, true).is_empty(),
            "休止比桥接上限还长 ⇒ 不许接"
        );

        // ⓒ **片段不够长**:右片段从 53_000 才开始 ⇒ 右窗要从 `t − xf` 起,越出片段 ⇒
        // 必须响亮夹紧,不许越界索引(不夹紧 ⇒ `seg[k - seg_lo]` 下溢 panic,变异实测)。
        let short: Vec<(i64, usize, Vec<f32>)> = vec![
            (0, 0, left[..56_000].to_vec()),
            (1, 53_000, right[53_000..(380 * 480)].to_vec()),
        ];
        let mut out = base.clone();
        splice_kept(&mut out, sr, spf, &diff, &short, xf, true).unwrap();
        assert_eq!(out[0], -1.0, "窗外仍是 base");
        assert!((out[30_000] - 0.5).abs() < 1e-6, "左窗内仍是左 donor");
    }

    #[test]
    fn the_donor_render_margin_covers_everything_the_join_can_reach() {
        // ⛔⛔ 这两个常量是**一对**,而它们住在两个模块里 —— 正是「改一个忘了另一个」最容易
        // 发生的形状,而且后果是**静默的**:窗边被挪进一个没渲的 chunk ⇒ 拼进去的是铺零 ⇒
        // 一个新的洞,形状与这一刀要修的那个一模一样,任何现存判据都看不见。
        //
        // 这一刀最远能把窗边挪到 `l.end + gap + min(gap, 4)`,而 `gap ≤ MERGE_BRIDGE_FRAMES`。
        let reach = MERGE_BRIDGE_FRAMES + 4;
        assert!(
            crate::inference::score2svc::DONOR_WINDOW_MARGIN_FRAMES >= reach,
            "donor 只渲了窗 ±{} 帧,而拼接层够得到 ±{reach} 帧 —— 中间那段会是铺零",
            crate::inference::score2svc::DONOR_WINDOW_MARGIN_FRAMES
        );
    }

    #[test]
    fn the_join_survives_the_whole_pipeline_and_needs_the_segment_margin() {
        // ⭐ 端到端:从 `apply_dead_only_windows_with` 进,donor 由闭包给。
        // ⛔ 它守的是**收益不许被静默砍掉** —— 变异实测:把 `margin` 改成 0 之后,每条 donor
        // 的片段只覆盖窗本身,搜索窗左右两半就永远凑不齐,这一刀悄悄什么也不做,而所有
        // 只用手工 `kept` 的单元测试**全绿**。S147 那次「收益静默减半」是同一个形状。
        // ⚠ 这条测试本身被我删过一次(重写三条 join 判据时连带删掉),而 `margin` 的变异
        //   当场就变绿了 —— 留着这句,提醒下一个人别再删。
        let sr = 48_000u32;
        let total = 400i64;
        let n = 400 * 480usize;
        let jobs = join_jobs();
        let (base0, left, right) = join_fixture(n);
        let run = |join: bool| {
            let mut b = base0.clone();
            apply_dead_only_windows_with(&mut b, sr, total, &jobs, false, join, |s, _| {
                Ok(if s == -9 { left.clone() } else { right.clone() })
            })
            .unwrap();
            b
        };
        let off = run(false);
        let on = run(true);
        assert_ne!(off, on, "打开与关闭必须真的不同");
        assert!(
            off[49_000..52_700].iter().all(|v| (*v + 0.05).abs() < 1e-6),
            "关着的时候那一段仍是 base"
        );
        let bad: Vec<usize> =
            (49_000..52_900).filter(|&k| (on[k] - 0.06).abs() > 1e-3).collect();
        assert!(
            bad.is_empty(),
            "走完整条管线之后那一段还剩 {} 个样本不是左 donor(第一个 {:?})",
            bad.len(),
            bad.first()
        );
    }

    fn spf_of(n: usize, total: i64) -> f64 {
        n as f64 / total as f64
    }

    /// S159 —— ⭐⭐⭐ **拼接层读到的每一个 donor 样本,都必须落在交给逆变换的保留区间里。**
    ///
    /// 这是「窗内逆变换」在**接线层**的承重判据,而且它是**行为**判据不是文本判据:
    /// donor 闭包按 [`donor_keep_samples`] 把缓冲染成两种值(窗内 +1 / 窗外 **−1**),
    /// 拼完之后 base 里只要出现负数,就说明拼接读到了一段**没有被逆变换过**的音频。
    ///
    /// ## ⛔ 它挡的是什么样的错
    /// 生产里那段「窗外」是 donor 在 `shift` 位渲出来的原始音频 —— 比目标高/低好几个半音。
    /// 少给一点余量,它会顺着 10 ms 交叉淡化混进被救乐句的边上;而**组数、乘客数、donor 遍数、
    /// 位移集、音频秒数、长度契约全部正常**,耳朵在整曲里也几乎抓不到。
    /// 形状与 S152 修掉的那个洞、以及 S147 那次「渲多了但拼对了」完全同族。
    ///
    /// ## 两条臂都要跑
    /// `join_rests` **关**着是今天的出厂,但它一开就会把窗边挪到相邻休止的另一头
    /// (最远 `MERGE_BRIDGE_FRAMES + 4` 帧)—— 那正是余量必须取 29 而不是 25 的理由。
    /// ⇒ 只跑默认臂的话,这条余量在判据上是**空的**。
    ///
    /// ⛔ 变异:`donor_keep_samples` 的余量改成 `MERGE_BRIDGE_FRAMES` 或 0 ⇒ 红(见下面的阴性对照,
    /// 它证明这条断言真的分得出「够」与「不够」)。
    #[test]
    fn the_inverse_window_covers_every_sample_the_splice_reads() {
        const SR: u32 = 48_000;
        const TOTAL_FRAMES: i64 = 600; // 12 s @ 50 fps
        let base_len = (TOTAL_FRAMES as usize) * (SR as usize) / 50;
        // 三组、两个位移。⛔ 前两组**必须**异位移、间隔 `0 < gap ≤ MERGE_BRIDGE_FRAMES`,
        // 否则 `join_rests` 的那个 `continue` 会让它一次都不执行 ⇒ 余量在判据上是空的。
        // ⚠ gap 取 10 而不是取满 25:切换点还要能在**左片段**里取到读窗,而左片段只到
        //   `l.end + 25` 帧 —— gap 取满时那正好等于 `r.start`,一个候选都过不去(实测)。
        let jobs = [
            DeadJob { shift: -6, start: 80, end: 140 },
            DeadJob { shift: -9, start: 150, end: 210 }, // gap = 10 帧
            DeadJob { shift: -6, start: 420, end: 470 },
        ];
        let spf = base_len as f64 / TOTAL_FRAMES as f64;
        // ⛔ `join_rests` 有一条 `JOIN_MIN_GAIN_DB = 6` 的门限:只有「今天那条边挖了个洞、
        // 而换个切换点能填上」时才动。常数信号上它**一次都不开火** —— 所以夹具要照 S152 那个
        // 真实病例造:base 在休止里有微弱预卷,而右 donor 在自己窗口的头 60 ms 是**静音**。
        let hole = ((jobs[1].start as f64 * spf) as usize, ((jobs[1].start as f64 * spf) as usize) + SR as usize * 60 / 1000);
        // 窗内 +1 / 窗外 **−1** 的 donor;`margin` 由参数给,好做阴性对照。
        let paint = |own: &[(i64, i64)], margin: i64, shift: i64| -> Vec<f32> {
            let mut d = vec![-1.0f32; base_len];
            for &(start, end) in own {
                let j = DeadJob { shift: 0, start, end };
                if let Some((a, b)) = donor_read_span(&j, spf, base_len, margin) {
                    for v in d[a..b].iter_mut() {
                        *v = 1.0;
                    }
                }
            }
            if shift == jobs[1].shift {
                for v in d[hole.0..hole.1.min(base_len)].iter_mut() {
                    *v = 0.0; // 洞在**窗内**,所以它不会被误读成「读到了保留区之外」
                }
            }
            d
        };
        for join in [false, true] {
            let mut base = vec![0.02f32; base_len]; // −34 dBFS 的底,好让「洞」量得出来
            apply_dead_only_windows_with(&mut base, SR, TOTAL_FRAMES, &jobs, false, join, |s, own| {
                // ⭐ 生产走的就是 `donor_keep_samples` 这一个函数,判据不许自己再拼一遍公式。
                let keep = donor_keep_samples_with(own, base_len, TOTAL_FRAMES, true);
                let mut d = vec![-1.0f32; base_len];
                for &(a, b) in &keep {
                    for v in d[a..b].iter_mut() {
                        *v = 1.0;
                    }
                }
                if s == jobs[1].shift {
                    for v in d[hole.0..hole.1.min(base_len)].iter_mut() {
                        *v = 0.0;
                    }
                }
                Ok(d)
            })
            .expect("splice");
            let bad = base.iter().position(|v| *v < 0.0);
            assert!(
                bad.is_none(),
                "join={join}:拼接读到了保留区间之外的 donor 样本(第 {} 个)—— \
                 生产里那一段是没被搬回原音高的音频",
                bad.unwrap_or(0)
            );
            // ⛔ 阴性对照:拼接**真的**发生过(否则上面那条只是「什么都没拼」)。
            assert!(base.iter().any(|v| *v > 0.5), "join={join}:一个 donor 样本都没拼进来");

            // ⛔ 阴性对照 ②:余量给不够时会怎样 —— 而**两条臂的答案不一样,那正是这条判据的重点**。
            let mut tight = vec![0.02f32; base_len];
            apply_dead_only_windows_with(
                &mut tight, SR, TOTAL_FRAMES, &jobs, false, join,
                |s, own| Ok(paint(own, 0, s)),
            )
            .expect("splice");
            let leaks = tight.iter().any(|v| *v < 0.0);
            if join {
                assert!(
                    leaks,
                    "join 开着时余量取 0 竟然也没漏 —— 那说明 `DONOR_WINDOW_MARGIN_FRAMES` 这 29 帧\
                     在判据上是**空的**,而它正是这条余量存在的全部理由"
                );
            } else {
                // ⭐⭐ 这不是「判据弱」,是一条要记住的事实:**今天的出厂默认(join 关)下,
                // 拼接只读窗本身** —— `splice_kept` 写的就是 `[start·spf, end·spf]`,
                // 那 25/29 帧余量一个样本都用不上。
                // ⇒ 只跑默认臂的话,余量取 0 / 25 / 29 **产出逐位相同的歌**,这条余量结构上测不到。
                // 那正是为什么上面要多跑一条 `join = true`;也是为什么翻 `JOIN_RESTS_DEFAULT`
                // 之前必须回来看这条判据。
                assert!(
                    !leaks,
                    "join 关着时拼接竟然读到了窗外 —— `splice_kept` 的读取范围变了,\
                     上面那句「今天只读窗本身」的注释已经过期"
                );
            }
        }
    }

    /// S159 —— **接线闸**:谱面轨的 donor 遍必须真的要窗。
    ///
    /// ⛔ 为什么需要一条这么笨的判据:这一刀「不接」与「接了」在**音频上逐位相同**,
    /// 上面那些判据一条都不会红,渲染照样出得来、秒数照样有 —— 只是慢一倍。
    /// S142 记过同一族的形状:整段代价提示当时可以删光而全仓零红 ⇒ 那种地方**第一条判据是
    /// 存在性/接线闸**。运行期的那只眼睛是 `[perf]` 旁边那行 `窗内 N/M 岛 · keep x%`。
    ///
    /// ⚠ 它只钉「有没有接」,不钉「接得对不对」—— 后者是上面那两条行为判据的事。
    #[test]
    fn the_score_donor_passes_actually_ask_for_the_window() {
        let cmd = include_str!("../commands/inference.rs");
        assert_eq!(
            cmd.matches(concat!("donor_keep", "_samples(")).count(),
            2,
            "谱面轨的两条臂(SoVITS / RVC)必须各自算一次保留区间 —— 少一条 = 那条臂静默退回全曲逆变换"
        );
        let s2s = include_str!("score2svc.rs");
        assert!(
            s2s.contains(concat!("d.keep", "_samples")),
            "`apply_range_inverse` 没有把 `DonorCtx::keep_samples` 传下去 —— 窗算了却没人用"
        );
        // base 遍(`donor == None`)必须拿到空窗 = 整条 = 今天。⛔ 反过来会让 base 遍
        // 一个岛都不处理,而 base 遍本来就 `range_shift == 0`、根本不进逆变换 —— 那种错
        // 在音频上看不见,只会在将来某次重构里变成真的。
        assert!(
            s2s.contains("donor.map_or(&[][..], |d| d.keep_samples)"),
            "base 遍的默认必须是「整条」而不是「什么都不做」"
        );
    }

    /// S159 —— 渲染侧的余量必须**严格**盖过拼接侧真正切走的那一段(29 > 25),
    /// 而且两者用的是**同一个**帧→样本公式。
    ///
    /// ⛔ 这条与上面那条是两件事:上面钉「拼接读到的都在保留区内」,这条钉「保留区比拼接层
    /// 自己留的片段还宽」—— 因为 `join_rests` 能把窗边挪出片段(`splice_kept` 那时只会打一条
    /// warn 然后夹紧),而被夹掉的那一段今天不会有任何判据看见。
    #[test]
    fn the_render_side_margin_strictly_covers_the_splice_side_slice() {
        const SR_FRAMES: i64 = 600;
        let base_len = 600usize * 960;
        let spf = base_len as f64 / SR_FRAMES as f64;
        let j = DeadJob { shift: -6, start: 200, end: 260 };
        let splice = donor_read_span(&j, spf, base_len, MERGE_BRIDGE_FRAMES).unwrap();
        let render = donor_keep_samples_with(&[(j.start, j.end)], base_len, SR_FRAMES, true);
        assert_eq!(render.len(), 1);
        let (ra, rb) = render[0];
        assert!(ra <= splice.0 && rb >= splice.1, "渲染侧 {ra}..{rb} 没盖住拼接侧 {splice:?}");
        assert!(ra < splice.0 && rb > splice.1, "两侧余量一样宽 —— join_rests 挪出去那一段没人管");
        // 边界写字面量(⛔ 不许拿被测的常量算期望值):25 帧 = 24000 样本,29 帧 = 27840 样本。
        assert_eq!(splice, (200 * 960 - 24_000, 260 * 960 + 24_000));
        assert_eq!((ra, rb), (200 * 960 - 27_840, 260 * 960 + 27_840));
        // 空窗 ⇒ 空保留区 ⇒ 整条缓冲 = 今天(⛔ 失败方向必须是「多做」)。
        assert!(donor_keep_samples_with(&[], base_len, SR_FRAMES, true).is_empty());
        // 相邻/重叠的窗必须合并(引擎那边每个岛都要扫一遍这张表)。
        let merged = donor_keep_samples_with(&[(100, 160), (170, 220)], base_len, SR_FRAMES, true);
        assert_eq!(merged.len(), 1, "隔 10 帧(< 2×29)的两条窗没有被合并:{merged:?}");
    }

    #[test]
    fn donor_render_gets_only_its_own_windows() {
        let jobs = [
            DeadJob { shift: -7, start: 10, end: 20 },
            DeadJob { shift: -2, start: 30, end: 40 },
            DeadJob { shift: -7, start: 44, end: 48 },
        ];
        let mut base = vec![0.0f32; 48000];
        let mut seen: Vec<(i64, Vec<(i64, i64)>)> = Vec::new();
        apply_dead_only_windows(&mut base, 48000, 50, &jobs, false, |s, own| {
            seen.push((s, own.to_vec()));
            Ok(vec![1.0f32; 48000])
        })
        .unwrap();
        seen.sort_by_key(|(s, _)| *s);
        assert_eq!(
            seen,
            vec![(-7i64, vec![(10i64, 20i64), (44, 48)]), (-2i64, vec![(30i64, 40i64)])],
            "每一遍只该拿到属于它自己那个位移的窗(传并集 ⇒ 这里两行会相同)"
        );
        assert_ne!(seen[0].1, seen[1].1, "不同的位移必须给出不同的保留集");

        // 阴性对照:同样的窗 ⇒ 同样的答案。
        let same = [
            DeadJob { shift: -7, start: 10, end: 20 },
            DeadJob { shift: -2, start: 10, end: 20 },
        ];
        let mut base2 = vec![0.0f32; 48000];
        let mut seen2: Vec<Vec<(i64, i64)>> = Vec::new();
        apply_dead_only_windows(&mut base2, 48000, 50, &same, false, |_, own| {
            seen2.push(own.to_vec());
            Ok(vec![1.0f32; 48000])
        })
        .unwrap();
        assert_eq!(
            seen2,
            vec![vec![(10i64, 20i64)], vec![(10i64, 20i64)]],
            "窗本来就相同时两遍必须拿到相同的答案 —— 否则上一条只是在测「总是不同」"
        );
    }

    #[test]
    fn dead_only_splice_matches_donor_level_to_base() {
        // 审查 S85 major 的钉子:base/donor 独立归一 → 拼接窗响度台阶;donor 按全曲
        // active-RMS 对齐 base。base 恒 0.5、donor 恒 0.25 ⇒ g=2 ⇒ 窗心 ≈0.5。
        let mut base = vec![0.5f32; 48000];
        let donor = vec![0.25f32; 48000];
        apply_dead_only_windows(&mut base, 48000, 50, &[DeadJob { shift: -6, start: 10, end: 20 }], true, |_, _| Ok(donor.clone()))
            .unwrap();
        assert!((base[(0.3 * 48000.0) as usize] - 0.5).abs() < 1e-3, "donor 缩放到 base 电平");
        assert!((base[0] - 0.5).abs() < 1e-6, "窗外不动");
    }

    #[test]
    fn dead_only_splice_rescues_a_single_frame_window() {
        // 1 帧窗收缩淡化宽度仍拿到 donor 内容,绝不静默丢弃(曾静默 continue+审计谎报)。
        let mut base = vec![0.0f32; 48000];
        let donor = vec![1.0f32; 48000];
        apply_dead_only_windows(&mut base, 48000, 50, &[DeadJob { shift: -6, start: 10, end: 11 }], true, |_, _| Ok(donor.clone()))
            .unwrap();
        let mid = (10.5 / 50.0 * 48000.0) as usize;
        assert!(base[mid] > 0.9, "1 帧窗仍拿到 donor 内容(微淡化)");
        assert_eq!(base[0], 0.0);
        assert_eq!(base[48000 / 2], 0.0);
    }

    #[test]
    fn dead_only_splice_skips_level_match_for_cover() {
        // S85e:cover 无逐渲归一,match_levels=false 时 donor 电平原样(模型对移调的
        // 真实响应)——窗口化 donor 缓冲大半为零,全曲 RMS 对比会误衰减,必须可关。
        let mut base = vec![0.5f32; 48000];
        let donor = vec![0.25f32; 48000];
        apply_dead_only_windows(&mut base, 48000, 50, &[DeadJob { shift: -6, start: 10, end: 20 }], false, |_, _| Ok(donor.clone()))
            .unwrap();
        assert!((base[(0.3 * 48000.0) as usize] - 0.25).abs() < 1e-6, "false ⇒ donor 电平不动");
    }

    // ── S85e 窗口化 donor 切片跨度 ──

    #[test]
    fn donor_spans_map_pad_and_clamp() {
        // 帧窗 → 样本跨度 ±pad,曲首/曲尾夹紧;spf 与拼接器同一映射。
        let jobs = [
            DeadJob { shift: -6, start: 2, end: 4 },
            DeadJob { shift: -6, start: 90, end: 100 },
        ];
        let spans = donor_slice_spans(&jobs, -6, 480.0, 48000, 2000);
        assert_eq!(spans, vec![(0, 3920), (41200, 48000)], "首端 0 夹紧、尾端 len 夹紧");
    }

    #[test]
    fn donor_spans_merge_neighbours_and_filter_shift() {
        // 同 shift 补白后重叠的切片合并(绝不重复渲染);异 shift 不掺和。
        let jobs = [
            DeadJob { shift: -6, start: 10, end: 12 },
            DeadJob { shift: -6, start: 14, end: 16 },
            DeadJob { shift: -2, start: 40, end: 42 },
        ];
        let spans = donor_slice_spans(&jobs, -6, 100.0, 100_000, 500);
        assert_eq!(spans, vec![(500, 2100)], "两窗补白重叠 → 单一合并切片");
        let spans2 = donor_slice_spans(&jobs, -2, 100.0, 100_000, 500);
        assert_eq!(spans2, vec![(3500, 4700)]);
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
        // ⛔ S158:这条钉的是**裁剪之前**那条臂,所以 trim 必须写死成 `None`。
        //    以前它借 `RescueTuning::today()`,而 S158 把 `TRIM_DEFAULT` 翻成「只裁尾」的
        //    那一秒它就换了臂(实测当场红:`{4,5}` 变 `{4,4}`)。⭐ 这次它红得响,
        //    但同一族里 `trimming_can_never_lose_a_rescue` 那种「off 与 on 必须一致」的判据
        //    换臂之后是**照绿**的 —— 所以借 `today()` 这件事本身就不许再有。
        let nn = [0, 73, 73, 0, 85, 73, 0];
        let land = RescueTuning::today().landing;
        let (plan, unfix) =
            dead_only_plan_with(&nn, &secs(nn.len()), 0, &dxl_like(), RescueTuning::new(None, land));
        assert!(unfix.is_empty());
        assert_eq!(plan, vec![DeadGroup { start: 4, end: 5, shift: -6 }]);
        // ⭐ **旋钮开着**时它把那个尾乘客放掉,而死音与落点一个字不变。
        let (knob_on, _) =
            dead_only_plan_with(&nn, &secs(nn.len()), 0, &dxl_like(), trim_arms(Some((f32::INFINITY, 500.0))).1);
        assert_eq!(knob_on, vec![DeadGroup { start: 4, end: 4, shift: -6 }]);
        // ⭐ 而出厂默认(S158d 起 = 只裁尾)必须**就是**那一档 —— 这一行是「默认往哪边」的判据,
        //    上面那一行是「这一刀做什么」的判据,两件事分开钉。
        let (shipped, _) =
            dead_only_plan_with(&nn, &secs(nn.len()), 0, &dxl_like(), RescueTuning::today());
        assert_eq!(shipped, knob_on, "出厂默认 = 只裁尾 ⇒ 与显式开着那条臂逐字相同");
    }

    #[test]
    fn dead_only_ignores_out_of_comfort_but_singable_material() {
        // 三轮核心:comfort [36,52] 说 64-78 全「越界」,但 f0 判据说能唱 → 一律原位。
        // (二轮 per-note 把整段搬走 = 用户判「灾难」的那台机器,这里钉死不再回来。)
        let nn = [0, 64, 66, 0, 73, 78, 0];
        let (plan, unfix) = dead_only_plan_with(&nn, &secs(nn.len()), 0, &dxl_like(), RescueTuning::today());
        assert!(plan.is_empty() && unfix.is_empty());
    }

    #[test]
    fn dead_only_transpose_folds_into_the_verdict() {
        let nn = [0, 73, 0]; // written 73 + transpose 12 = 85 → 同款 -6 救援
        let (plan, _) = dead_only_plan_with(&nn, &secs(nn.len()), 12, &dxl_like(), RescueTuning::today());
        assert_eq!(plan, vec![DeadGroup { start: 1, end: 1, shift: -6 }]);
    }

    #[test]
    fn dead_only_dragged_neighbours_must_stay_singable() {
        // 短语 [85, 40]:-6 把 40 拖到 34(窗外=死)→ 无解,必须响亮计数而非静默跳过。
        let nn = [0, 85, 40, 0];
        let (plan, unfix) = dead_only_plan_with(&nn, &secs(nn.len()), 0, &dxl_like(), RescueTuning::today());
        assert!(plan.is_empty());
        assert_eq!(unfix, vec![(1, 2)], "无解组带位置(审查 S85:取证要「在哪」)");
    }

    #[test]
    fn dead_only_bounds_record_falls_back_to_bounds() {
        // 无扫描旧记录:dead=出 usable,落点=进 comfort。88→79 = -9;40→52 = +12。
        let r = SpeakerRange::bounds((48.0, 84.0), (52.0, 79.0));
        let (plan, unfix) = dead_only_plan_with(&[0, 88, 0], &secs(3), 0, &r, RescueTuning::today());
        assert_eq!(plan, vec![DeadGroup { start: 1, end: 1, shift: -9 }]);
        assert!(unfix.is_empty());
        let (plan, _) = dead_only_plan_with(&[0, 40, 0], &secs(3), 0, &r, RescueTuning::today());
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
        let (plan, unfix) = dead_only_plan_with(&[0, 78, 0], &secs(3), 0, &r, RescueTuning::today());
        assert!(unfix.is_empty());
        assert_eq!(plan, vec![DeadGroup { start: 1, end: 1, shift: -1 }]);
    }

    // ── S151 卸乘客 ─────────────────────────────────────────────────────────────────────
    // ⛔ 阈值一律写**字面量**:引用被测的常量会让判据永远不可能红(这一区 S146c 的血训)。

    /// S158 —— 这一族 A/B 的两条臂,**只许差一个自由度**。
    ///
    /// ⛔⛔ 它们原来一边写 `RescueTuning::today()`、另一边写
    /// `RescueTuning::new(Some((1000.0, 500.0)), None)` ⇒ trim **和** landing 同时不同,
    /// 每一条 A/B 的归因都是坏的。今天读不出差别,**只因为** `dxl_like()` 的 `low_ratio`
    /// 是平的 ⇒ landing 臂在它上面选不出别的落点 —— 也就是说这几条判据的「单变量」
    /// 是**夹具的巧合**,不是设计。
    ///
    /// ⛔ 而更糟的一半是:关臂借的是 `today()`。S158 把 `TRIM_DEFAULT` 翻成「只裁尾」的
    /// 那一秒,那一边会**静默变成开臂**,`trimming_can_never_lose_a_rescue` 这种
    /// 「off 与 on 必须一致」的判据会**照绿**,而它此后什么都没在测。
    ///
    /// ⇒ 两条臂的 landing 取**同一个值**,而且从 `today()` 取(不写字面量 ⇒ 将来翻 landing
    /// 也不会让这一族悄悄变成双变量);trim 那一维由调用方给,是**唯一**的自变量。
    fn trim_arms(trim: Option<(f32, f32)>) -> (RescueTuning, RescueTuning) {
        let land = RescueTuning::today().landing;
        (RescueTuning::new(None, land), RescueTuning::new(trim, land))
    }

    /// 机理夹具用的**有限**头门限。⚠ 出厂那一档的头门限是 `f32::INFINITY`(= 永不裁头,
    /// 盲测判负 + S158 实测复现),所以「头裁到底怎么算」只能用这个**从未出厂**的值来测机理;
    /// 出厂那一档由 `the_shipped_trim_only_ever_cuts_the_tail` 单独钉。
    const MECH_HEAD_MS: f32 = 1000.0;

    /// 乐句 = 3 个健康音 · 1 个死音 · 2 个健康音。头 1.5 s、尾 0.8 s 都够本 ⇒ 两边都裁掉。
    #[test]
    fn a_rescue_group_sheds_the_passengers_that_pay_for_their_own_seam() {
        let nn = [0, 73, 73, 73, 85, 73, 73, 0];
        let fr = [10, 25, 25, 25, 40, 20, 20, 10]; // 头 75 帧 = 1.50 s;尾 40 帧 = 0.80 s
        let (off_arm, on_arm) = trim_arms(Some((MECH_HEAD_MS, 500.0)));
        let (whole, _) = dead_only_plan_with(&nn, &fr, 0, &dxl_like(), off_arm);
        assert_eq!(
            whole,
            vec![DeadGroup { start: 1, end: 6, shift: -6 }],
            "关掉时必须是 S150 之前那条整句臂"
        );
        let (trimmed, unfix) = dead_only_plan_with(&nn, &fr, 0, &dxl_like(), on_arm);
        assert!(unfix.is_empty());
        assert_eq!(
            trimmed,
            vec![DeadGroup { start: 4, end: 4, shift: -6 }],
            "5 个乘客白白过一遍 PSOLA + 一遍移调声码器,而它们值一条缝"
        );
    }

    /// 同样的形状,但乘客只有 0.60 s / 0.28 s —— 造一条缝去救这么点材料是亏的。
    #[test]
    fn a_cut_that_frees_almost_nothing_is_not_worth_the_seam_it_makes() {
        let nn = [0, 73, 73, 73, 85, 73, 73, 0];
        let fr = [10, 10, 10, 10, 40, 7, 7, 10]; // 头 30 帧 = 0.60 s;尾 14 帧 = 0.28 s
        let (plan, _) = dead_only_plan_with(&nn, &fr, 0, &dxl_like(), trim_arms(Some((MECH_HEAD_MS, 500.0))).1);
        assert_eq!(
            plan,
            vec![DeadGroup { start: 1, end: 6, shift: -6 }],
            "回收量在门限之下 ⇒ 整句不动,与关掉时一模一样"
        );
    }

    /// 头尾各 0.70 s:**尾裁、头不裁**。S148 实测裁头的缝 Δripple p50 0.258 / p90 1.452 dB,
    /// 裁尾 0.033 / 0.407(地板 0.060)—— 8 倍,因为 10 ms 淡入压在**起音**上。
    #[test]
    fn the_head_cut_has_to_buy_more_than_the_tail_cut() {
        let nn = [0, 73, 73, 73, 85, 73, 73, 0];
        let fr = [10, 12, 12, 11, 40, 18, 17, 10]; // 头 35 帧 = 0.70 s;尾 35 帧 = 0.70 s
        let (plan, _) = dead_only_plan_with(&nn, &fr, 0, &dxl_like(), trim_arms(Some((MECH_HEAD_MS, 500.0))).1);
        assert_eq!(
            plan,
            vec![DeadGroup { start: 1, end: 4, shift: -6 }],
            "同样的回收量:尾边够本、头边不够本"
        );
    }

    /// ⭐ 这一刀的**定义域**:它只决定谁陪着走,永远不许改「哪些音被救」。
    /// (数学上也成立:死音集合与裁剪无关,而裁掉被拖拽音只会**放松** `minimal_rescue_shift`
    /// 的约束 ⇒ 合格集合只增不减。这条判据把那个方向钉在实测上。)
    #[test]
    fn trimming_can_never_lose_a_rescue() {
        let nn = [0, 73, 73, 85, 73, 0, 60, 62, 0, 73, 85, 83, 73, 73, 0, 85, 40, 0];
        let fr = vec![30i64; nn.len()];
        let r = dxl_like();
        let dead_of = |plan: &[DeadGroup]| -> Vec<usize> {
            let mut v: Vec<usize> = plan
                .iter()
                .flat_map(|g| g.start..=g.end)
                .filter(|&k| nn[k] > 0 && !r.slot_singable(nn[k]))
                .collect();
            v.sort_unstable();
            v
        };
        let (off_arm, on_arm) = trim_arms(Some((MECH_HEAD_MS, 500.0)));
        let (off, unfix_off) = dead_only_plan_with(&nn, &fr, 0, &r, off_arm);
        let (on, unfix_on) = dead_only_plan_with(&nn, &fr, 0, &r, on_arm);
        assert_eq!(dead_of(&off), dead_of(&on), "被救的死音一个不许多、一个不许少");
        assert_eq!(unfix_off, unfix_on, "无解的乐句也不许因为裁剪而改变");
        assert_eq!(unfix_on, vec![(15, 16)], "…而这份夹具里确实有一个(85 拖着 40 出窗)");
        assert!(on.iter().zip(&off).all(|(a, b)| a.shift == b.shift), "本夹具上落点不动");
        assert!(on.iter().zip(&off).any(|(a, b)| (a.start, a.end) != (b.start, b.end)),
                "…但至少有一组真的被裁过(否则上面三条都是空的)");
    }

    /// ⭐ **实测发现的一格,不是设计出来的**:`[85, 40]` 这句今天无解(−6 把 40 拖到 34,
    /// 出了扫描窗),而**一旦把乘客 40 卸掉,它就有落点了** —— 挡路的正是那个乘客。
    /// 那是一条真的改善,但它改的是【哪些音被救】而不是【谁陪着走】⇒ **不许搭这一刀的车**:
    /// 混在一起,任何 A/B 的结果都归因不到人。留作单独一笔 + 单独一次盲测。
    #[test]
    fn trimming_may_not_turn_an_unfixable_phrase_into_a_rescue() {
        let nn = [0, 85, 40, 0];
        let fr = [10, 40, 100, 10]; // 尾部乘客 2.0 s —— 回收量远超门限,唯一拦住它的只能是规则
        let (off_arm, on_arm) = trim_arms(Some((MECH_HEAD_MS, 500.0)));
        let (off, unfix_off) = dead_only_plan_with(&nn, &fr, 0, &dxl_like(), off_arm);
        let (on, unfix_on) = dead_only_plan_with(&nn, &fr, 0, &dxl_like(), on_arm);
        assert!(off.is_empty() && on.is_empty(), "两条臂都必须放弃这一句");
        assert_eq!(unfix_off, vec![(1, 2)]);
        assert_eq!(unfix_on, vec![(1, 2)], "卸乘客不许把无解句变成被救句");
        // 阳性对照:同一句,乘客换成一个不挡路的音 ⇒ 两条臂都救,且开着的那条真的裁了。
        let nn = [0, 85, 73, 0];
        let (off, _) = dead_only_plan_with(&nn, &fr, 0, &dxl_like(), off_arm);
        let (on, _) = dead_only_plan_with(&nn, &fr, 0, &dxl_like(), on_arm);
        assert_eq!(off, vec![DeadGroup { start: 1, end: 2, shift: -6 }]);
        assert_eq!(on, vec![DeadGroup { start: 1, end: 1, shift: -6 }]);
    }

    /// akiko 的形状(照盘上那份记录简化):73..79 的 `low_ratio` 从 0.10 一路爬到 0.63,
    /// 而 `damage_from_scan` 对 0.55 以下**全免费** ⇒ 73..78 六格在今天的目标函数里**完全等价**。
    /// ⛔ 数值写字面量,不引用被测的常量。
    fn akiko_like() -> SpeakerRange {
        let mut semis = serde_json::Map::new();
        for m in 36..=72i64 {
            semis.insert(m.to_string(), serde_json::json!([1, 1.0, -1.0, 0.20]));
        }
        for (m, lr) in [(73, 0.10), (74, 0.12), (75, 0.18), (76, 0.21), (77, 0.28), (78, 0.39), (79, 0.63)] {
            semis.insert(m.to_string(), serde_json::json!([1, 1.0, -1.0, lr]));
        }
        semis.insert("80".into(), serde_json::json!([3, 0.67, -6.7, 0.93]));
        for m in 81..=96i64 {
            semis.insert(m.to_string(), serde_json::json!([9999, 0.0, -21.0, 0.80]));
        }
        speaker_range(
            &config_with(serde_json::json!({
                "usable": [36, 76], "usable_auto": [36, 80], "comfort": [36, 79],
                "semitones": serde_json::Value::Object(semis)
            })),
            0,
        )
        .unwrap()
    }

    /// S151 主刀:落点排序在 damage 打平之后**再看一眼扫描的原始 `low_ratio`**。
    /// 出处是两把独立的尺子(用户 2026-08-18 那两条实机渲染):donor 被要求唱在 77-78 时
    /// 音内包络起伏率 30% vs ≤76 的 7%(p=0.0026);同一位移下落点 >73 的元音塌 −2.74 dB
    /// vs ≤73 的 −0.95(p=4.2e-10)。
    #[test]
    fn the_landing_arm_lands_where_the_scan_says_the_voice_is_still_there() {
        let nn = [0, 80, 0];
        let fr = secs(3);
        let r = akiko_like();
        // ⛔ S157c:`today()` 已经翻成 `Some(3)` ⇒ 要「S151 之前那条臂」必须**显式**写。
        let (today, _) = dead_only_plan_with(&nn, &fr, 0, &r, RescueTuning::new(None, None));
        assert_eq!(
            today,
            vec![DeadGroup { start: 1, end: 1, shift: -2 }],
            "今天:预算 +1 ⇒ 停在 78(low_ratio 0.39),因为 73..78 在 damage 上全是 0"
        );
        let (deep, _) = dead_only_plan_with(&nn, &fr, 0, &r, RescueTuning::new(None, Some(3)));
        assert_eq!(
            deep,
            vec![DeadGroup { start: 1, end: 1, shift: -4 }],
            "开着:同样的合格集合里挑 low_ratio 更低的那一格(76 = 0.21),而不是最浅的"
        );
        // ⛔ 预算仍然封顶 —— 无上限的排序正是 S148 记着的那台「东雪莲 −24」的机器。
        let (cap, _) = dead_only_plan_with(&nn, &fr, 0, &r, RescueTuning::new(None, Some(1)));
        assert_eq!(cap[0].shift, -2, "预算 +1 时它够不到 76,必须老老实实停在 78");
        // ⭐ 这一对是**变异逼出来的**:上面那三条在「排序被默认打开」时**全是绿的**,
        // 因为在那个夹具上 damage 已经把候选筛到只剩一个。要让「默认臂没被动过」这件事
        // 有判据,必须构造一个**默认预算内就有两个 damage 打平、但 low_ratio 不同**的局面:
        // 死音 79 ⇒ 落点 78(0.39)与 77(0.28),两个 damage 都是 0。
        let (near_today, _) =
            dead_only_plan_with(&[0, 79, 0], &fr, 0, &r, RescueTuning::new(None, None));
        assert_eq!(near_today[0].shift, -1, "默认臂:damage 打平就取最浅 ⇒ 停在 78");
        let (near_on, _) = dead_only_plan_with(&[0, 79, 0], &fr, 0, &r, RescueTuning::new(None, Some(1)));
        assert_eq!(near_on[0].shift, -2, "开着:同一个预算内也要挑 low_ratio 更低的 77");
        // 阴性对照:一个 low_ratio 平坦的记录上,开与关必须给出**同一个**落点。
        let flat = dxl_like();
        let (a, _) = dead_only_plan_with(&[0, 85, 0], &fr, 0, &flat, RescueTuning::new(None, None));
        let (b, _) = dead_only_plan_with(&[0, 85, 0], &fr, 0, &flat, RescueTuning::new(None, Some(3)));
        assert_eq!(a, b, "没有 low_ratio 可排时,这一刀必须什么也不做");
    }

    /// ⛔⛔ S158d —— **cover / audition 那一轨拿不到谱面轨的旋钮,而这件事到今天为止
    /// 没有任何判据钉着。**S157c 翻了 `LANDING_DEFAULT`、S158d 翻了 `TRIM_DEFAULT`,
    /// **两次都只到了谱面轨**:`cover_dead_plan` 里没有裁剪那一段,`landing` 也是硬传 `None`。
    /// ⇒ 用户在音频轨/试听上听到的救援,与谱面轨渲出来的**已经是两条不同的规则**,
    /// 而没有一行输出说破。
    ///
    /// ⚠⚠ **S159k 改掉了这条 doc 原来的一半,原文照录以便对账**:
    /// 「cover 轨从来就只覆盖死音区间本身(按死帧成区 + `GAP_TOL_MS` 桥接),它**根本不拖乘客**
    ///   —— 也就是说它一直是『头尾都裁』的。S158d 这一刀是让谱面轨往它那边走了半步(只裁尾)。」
    /// 那句话当时还写着「这条判据**不主张『应该接上』** —— 那是另一笔**要单独定价的账**」。
    /// ⇒ S159k **把那笔账定价了**,然后接上了半步:cover 现在会把边**外扩到最近的清音帧**
    /// (上限 `COVER_EDGE_SEEK_MS`),因此**它开始拖乘客了**。定价见 `cover_dead_plan` 里
    /// 那一段注释(用户真病例:边界大台阶 11 条 → 32 条,最大 −8 → −62 dB)。
    ///
    /// ⛔ **仍然故意不给的**:谱面轨的 `landing` / `trim` 旋钮。这条判据钉的就是这一件 ——
    /// 先用阳性对照证明旋钮在谱面轨上真的咬人,再钉住 cover 轨给的是**旋钮之前**那个答案。
    /// 哪天有人把它接上,这条会红,而那时候红的是「你改了一条用户听得见的规则」,不是「测试碍事」。
    #[test]
    fn the_cover_lane_deliberately_does_not_get_the_score_lane_knobs() {
        let r = akiko_like();
        let fr = secs(3);
        // ⓐ 阳性对照:同一个死音,`landing` 在**谱面轨**上必须真的换落点。
        //    (没有这一条,下面那条断言可能只是「这个夹具上旋钮本来就不咬人」。)
        let (off, _) = dead_only_plan_with(&[0, 80, 0], &fr, 0, &r, RescueTuning::new(None, None));
        let (on, _) = dead_only_plan_with(&[0, 80, 0], &fr, 0, &r, RescueTuning::today());
        assert_eq!(off[0].shift, -2, "旋钮之前:预算 +1 ⇒ 停在 78");
        assert_eq!(on[0].shift, -4, "阳性对照:出厂默认在谱面轨上把它带到 76");
        // ⓑ 而 cover 轨对同一个音高给出的是**旋钮之前**那个答案。
        // ⚠ S159k:原来这里用的是 80,而 80 只需要 −2 —— 已被 `COVER_MIN_RESCUE_DEPTH` 挡掉。
        //    换成 83(够深,门槛碰不到它),ⓐ 的阳性对照也跟着换成同一个音高。
        let (off3, _) = dead_only_plan_with(&[0, 83, 0], &fr, 0, &r, RescueTuning::new(None, None));
        let (on3, _) = dead_only_plan_with(&[0, 83, 0], &fr, 0, &r, RescueTuning::today());
        assert_eq!(off3[0].shift, -5, "旋钮之前:83 停在 78");
        assert_eq!(on3[0].shift, -7, "阳性对照:出厂默认在谱面轨上把它带到 76");
        let f0 = vec![hz(83.0); 200]; // 2 s @ 100 fps,远超 MIN_VIOLATION_MS
        let (jobs, unfix) = cover_dead_plan(&f0, 100.0, &r);
        assert!(unfix.is_empty());
        assert_eq!(jobs.len(), 1, "一整段 83 应当成一个区域");
        assert_eq!(
            jobs[0].shift, -5,
            "cover 轨拿不到 LANDING_DEFAULT —— 若这里变成 -7,说明有人把谱面轨的旋钮接到了              音频轨上,那是用户听得见的改动,必须是有意的并且要单独定价"
        );
        // ⓒ **S159k 之后**:cover 会把边外扩到最近的清音帧,但**上限 `COVER_EDGE_SEEK_MS`**,
        //    而且**够不着就原地不动**。这里用「前后都是唱得动的材料、中间一段死音」钉住那个上限:
        //    前后各 200 帧(2 s)全程有声、中间没有任何清音隙 ⇒ **一帧都不许外扩**。
        //    ⛔ 这一条是 S159k 的护栏:第一版没有上限,它会把整整 6 秒连唱吞进 donor。
        let mut f0b = vec![hz(60.0); 200];
        f0b.extend(vec![hz(83.0); 200]);
        f0b.extend(vec![hz(60.0); 200]);
        let (jobs2, _) = cover_dead_plan(&f0b, 100.0, &r);
        assert_eq!(jobs2.len(), 1);
        assert_eq!(
            (jobs2[0].start, jobs2[0].end),
            (200, 400),
            "够不着清音时 cover 的边必须原地不动 —— 没有上限的话这里会变成 (0, 600)"
        );
        // ⛔ 阴性对照:同一份材料,只要**够得着**清音,边就必须真的走过去
        //    (否则上面那条可以由「外扩根本没接上」满足)。
        // ⚠ 第一版这条对照写错过:只在死区**前面**放了清音,却去断言**尾边**会走 —— 当场红,
        //    而红的是对照不是代码。⇒ 清音要放在**被断言的那一侧**。
        let mut f0c = vec![hz(60.0); 200];
        f0c.extend(vec![hz(83.0); 200]);
        f0c.extend(vec![hz(60.0); 20]);
        f0c.push(0.0); // 死区之后 20 帧(200 ms < 上限)有一条清音隙
        f0c.extend(vec![hz(60.0); 179]);
        let (jobs3, _) = cover_dead_plan(&f0c, 100.0, &r);
        assert_eq!(jobs3[0].start, 200, "头边够不着清音 ⇒ 原地不动");
        assert_eq!(jobs3[0].end, 420, "尾边够得着 ⇒ 必须走到那条清音隙");
        assert_eq!(f0c[420], 0.0, "断言里的 420 得真的是那条清音隙");
    }

    /// 与卸乘客同一条定义域:它只决定**落在哪**,不许改**哪些音被救**。
    #[test]
    fn the_landing_arm_never_changes_which_notes_get_rescued() {
        let nn = [0, 73, 80, 73, 0, 60, 0, 85, 40, 0];
        let fr = secs(nn.len());
        let r = akiko_like();
        let (a, ua) = dead_only_plan_with(&nn, &fr, 0, &r, RescueTuning::new(None, None));
        let (b, ub) = dead_only_plan_with(&nn, &fr, 0, &r, RescueTuning::new(None, Some(3)));
        let dead_of = |p: &[DeadGroup]| {
            let mut v: Vec<usize> = p.iter().flat_map(|g| g.start..=g.end)
                .filter(|&k| nn[k] > 0 && !r.slot_singable(nn[k])).collect();
            v.sort_unstable();
            v
        };
        assert_eq!(dead_of(&a), dead_of(&b), "被救的死音一个不许多、一个不许少");
        assert_eq!(ua, ub, "无解的乐句也不许变");
        assert!(a.iter().zip(&b).any(|(x, y)| x.shift != y.shift), "…但落点必须真的动过");
    }

    /// ⛔ 可选的深度有上限([`LANDING_RATIO_TWO_ST`]);**必须**走那么深才够得着的组照走不误
    /// —— 那是「救不救得了」,不是「落得干不干净」。
    ///
    /// ⚠ S157 更正:这条判据原来的题头写的是「不许推过 `ratio = 2`,那之后每个基音周期都有
    /// 一段永远不被读到」。**那个理由已经死了**(S156 把宽读窗翻成默认之后,
    /// +12/+14/+16 的 `src_uncovered_frac` 实测**全是 0.0000%**,三份素材、带阳性阴性对照)。
    /// 今天这个上限按**另一条**证据定价,全文在 [`LANDING_RATIO_TWO_ST`] 的 doc 里。
    /// ⚠ 这条判据只钉「上限存在且会咬」,**钉不住它的值** —— 值由
    /// `the_optional_depth_cap_is_worth_exactly_what_the_ear_line_paid_for` 钉。
    #[test]
    fn optional_depth_stops_where_the_synthesis_starts_dropping_the_source() {
        let mut semis = serde_json::Map::new();
        for m in 36..=79i64 {
            // 76 以下明显不薄 ⇒ 排序想往下走;⛔ 台阶要比 `LANDING_THIN_EPS` 大得多,
            // 否则「打平」会把这条判据变成一句空话(第一版就是这么写坏的:台阶 0.008 < eps)。
            let lr = if m >= 77 { 0.50 } else if m == 76 { 0.30 } else { 0.20 };
            semis.insert(m.to_string(), serde_json::json!([1, 1.0, -1.0, lr]));
        }
        for m in 80..=96i64 {
            semis.insert(m.to_string(), serde_json::json!([9999, 0.0, -21.0, 0.90]));
        }
        let r = speaker_range(
            &config_with(serde_json::json!({
                "usable": [36, 79], "usable_auto": [36, 79], "comfort": [36, 79],
                "semitones": serde_json::Value::Object(semis)
            })),
            0,
        )
        .unwrap();
        // 够得着的组(最浅 −4)⇒ 可以多花到 −7
        let (near, _) = dead_only_plan_with(&[0, 83, 0], &secs(3), 0, &r, RescueTuning::new(None, Some(3)));
        assert_eq!(near[0].shift, -7, "还没到 ratio=2,那三个半音可以花");
        // 已经在线上的组(最浅 −13)⇒ 一个半音都不许多花
        let (deep_today, _) =
            dead_only_plan_with(&[0, 92, 0], &secs(3), 0, &r, RescueTuning::new(None, None));
        let (deep_on, _) = dead_only_plan_with(&[0, 92, 0], &secs(3), 0, &r, RescueTuning::new(None, Some(3)));
        assert_eq!(deep_today[0].shift, -13, "这个音本来就要 −13 才够得着");
        assert_eq!(
            deep_on[0].shift, -13,
            "已经越过 ratio=2 的组:救援照做,但**不许为了更干净的落点再往下花**"
        );
    }

    /// S157 的夹具 —— **用户 2026-08-20 报的那个音的微缩版**,逐格照 akiko 盘上的记录抄。
    ///
    /// 乐句 = `notes[685..=693]` 的真实音高 `[90, 80, 78, 75, 68, 71, 70, 71, 80]`
    /// (usable 上限 76 ⇒ 死音 = 90 / 80 / 78 / 80,中间五个是乘客)。三件必须照抄的细节:
    /// * **MIDI 54 与 61 的 `low_ratio` = 0.573 / 0.572** —— 刚越过 `damage_from_scan`
    ///   那条 0.55 免费线 ⇒ damage 0.1725 / 0.165。它们正是**两个乘客在 −14 上的落点**,
    ///   而在 −12 上乘客落到 63 / 56 / 59 / 58(全都干净)⇒ 这就是「乘客否决落点」的全部机理。
    ///   ⛔ 不许把整条中音区都写成薄的:那样两条臂都被否决,判据当场变空(第一版就是)。
    /// * **MIDI 65 的 onset 通道电平 −9.9 dB**(< [`RMS_FREE_DB`])⇒ 它的 LANDING 位被清掉
    ///   ⇒ −13 不合格。少了这一格,今天的臂会停在 −13 而不是生产实测的 −12。
    /// * 73..79 的 `low_ratio` 阶梯 0.10/0.12/0.18/**0.21**/0.28/**0.39**/0.63 —— 落点
    ///   78 与 76 的差(0.39 → 0.21)就是这一刀要买的东西。
    fn pya_like() -> SpeakerRange {
        let mut semis = serde_json::Map::new();
        for m in 36..=72i64 {
            let lr = match m {
                54 => 0.573, // ⭐ 乘客 68 在 −14 上的落点
                61 => 0.572, // ⭐ 乘客 75 在 −14 上的落点
                _ => 0.20,
            };
            semis.insert(m.to_string(), serde_json::json!([1, 1.0, -1.0, lr]));
        }
        for (m, lr) in
            [(73, 0.10), (74, 0.12), (75, 0.18), (76, 0.21), (77, 0.28), (78, 0.39), (79, 0.63)]
        {
            semis.insert(m.to_string(), serde_json::json!([1, 1.0, -1.0, lr]));
        }
        semis.insert("80".into(), serde_json::json!([3, 0.67, -6.7, 0.93]));
        for m in 81..=96i64 {
            semis.insert(m.to_string(), serde_json::json!([9999, 0.0, -21.0, 0.80]));
        }
        speaker_range(
            &config_with(serde_json::json!({
                "usable": [36, 76], "usable_auto": [36, 80], "comfort": [36, 79],
                "semitones": serde_json::Value::Object(semis),
                // 盘上的原值是 [5, 1, -9.9, 0.73] —— 只有电平那一列会否决落点。
                "semitones_onset": { "65": [5, 1.0, -9.9, 0.73] }
            })),
            0,
        )
        .unwrap()
    }

    /// ⭐⭐⭐ S159zi —— **一整串全是死音、但各自需要的深度差很多时,也要拆开**。
    ///
    /// [`an_interior_run_of_three_singable_notes_splits_the_phrase`] 钉的是**一级**拆
    /// (夹心是**可唱音**);这一条钉的是**二级**(见 [`SPLIT_MIN_COST_DEFAULT`]):
    /// 用户 2026-08-22 报的 3:16 那一处三个音**全是死音**,一级结构上看不见。
    ///
    /// ⛔ 三条阴性对照,少一条这条判据就可能是「恒真」:
    /// ⑴ **深度差小就不许拆**(`[83, 81, 81]`:差 2 度 × 2000 ms = 4000 < 6000)——
    ///    门限真的在起作用,而不是「只要有死音就拆」;
    /// ⑵ **同一批音,只把时长改短就不许拆**(2 × 100 ms × 11 度 = 2200 < 6000)——
    ///    判据真的是 **ms·半音**,而不是偷偷退化成「只看深度差」;
    /// ⑶ **被救的死音一个不许多、一个不许少**,而且拆出来的组不许重叠、不许比整句更深。
    ///
    /// ⛔ 变异(写这条判据时逐个**真跑过**,下面记的是**实测读数**——
    /// ⚠ 其中两条我先写下的预测**是错的**,按实测改了过来:
    /// 「去掉门」与「取最差断点」我都以为会**少拆**,实测两条都**多拆**成了三组):
    /// * [`SPLIT_MIN_COST_DEFAULT`] 改成 30000 ⇒ 读 `[1..3] −13` 一组,**红**;
    /// * 把 `gain >= SPLIT_MIN_COST_DEFAULT` 换成 `>= 0.0` ⇒ 读
    ///   `[1..1] −13, [2..2] −2, [3..3] −2` —— 连**没有任何收益**的那一刀也切,**红**
    ///   (⇒ 那条门挡的不只是「不划算」,还有「切完两侧位移一模一样」的纯浪费);
    /// * `need(q, cl)` 改成 `need(cf, cl)`(右侧永远拿整簇的深度)⇒ `gain` 恒 0 ⇒
    ///   读一组,**红**;
    /// * `best` 的比较从 `gain > g` 改成 `gain < g`(取最差的断点)⇒ 读三组,**红**
    ///   —— 机理不是我以为的「gain 0 所以不拆」,而是**先切在最差处、剩下的再递归切开**。
    #[test]
    fn a_run_of_dead_notes_splits_where_the_depth_requirement_drops() {
        let r = dxl_like(); // usable [36,80];81 起是死音,落点只到 79 ⇒ 81 要 −2、92 要 −13
        let plan = |p: &[i64], f: &[i64]| dead_only_plan_with(p, f, 0, &r, RescueTuning::today()).0;

        // ⑷ ⭐ 主臂 —— 3:16 的形状:顶音要 −13,后面两个只高出 usable 顶 1 度、只要 −2。
        let deep = [0i64, 92, 81, 81, 0];
        let g = plan(&deep, &secs(deep.len()));
        assert_eq!(
            g,
            vec![
                DeadGroup { start: 1, end: 1, shift: -13 },
                DeadGroup { start: 2, end: 3, shift: -2 },
            ],
            "全是死音也要按深度拆开(读到 {g:?})"
        );
        // ⭐⭐⭐ 这一刀买到的就是这 11 个半音 × 2 s:不拆的话那两个音会跟着走 −13,
        // 按 S159z 的定价 ≈ 每个白丢 **高频 14.4 dB**。
        assert_eq!(g[0].shift.abs() - g[1].shift.abs(), 11, "省下来的正是这 11 个半音");

        // ⑴ 阴性对照 A —— 深度差小就不许拆(**1** 度 × 2000 ms = 2000 < 3000)。
        // ⚠ S159zi:门从 6000 降到 3000(定价见 [`SPLIT_MIN_COST_DEFAULT`])之后,原来那个
        //    `[83, 81, 81]`(差 2 度 = 4000)**够得着了** ⇒ 阴性对照会当场失效。
        //    ⛔ 正确的改法是把夹具挪到新门限的下方(83 → 82),而不是把期望值改成「拆」——
        //    这条对照要证的是「门限真的在起作用」,改期望值就把它证没了。
        let shallow = [0i64, 82, 81, 81, 0];
        let gs = plan(&shallow, &secs(shallow.len()));
        assert_eq!(
            gs,
            vec![DeadGroup { start: 1, end: 3, shift: -3 }],
            "深度差只有 1 度 ⇒ 不值一条新缝,不许拆(读到 {gs:?})"
        );

        // ⑵ 阴性对照 B —— **同一批音高**,只把浅的那一侧改成 100 ms ⇒ 2200 ms·半音 < 6000。
        let short = plan(&deep, &[50, 50, 5, 5, 50]);
        assert_eq!(
            short,
            vec![DeadGroup { start: 1, end: 3, shift: -13 }],
            "判据是 ms·半音:同样 11 度的差,只有 200 ms 就不许拆(读到 {short:?})"
        );

        // ⑸ 断点必须落在**深度真的掉下去**的那一处,不是第一处可断的地方。
        let four = [0i64, 92, 92, 81, 81, 0];
        let g4 = plan(&four, &secs(four.len()));
        assert_eq!(
            g4,
            vec![
                DeadGroup { start: 1, end: 2, shift: -13 },
                DeadGroup { start: 3, end: 4, shift: -2 },
            ],
            "断点要断在 [2|3] 而不是 [1|2](读到 {g4:?})"
        );

        // ⑶ 被救的死音一个不许多、一个不许少;不许重叠;不许比整句更深。
        for (p, gg) in [(&deep[..], &g), (&four[..], &g4)] {
            let got: Vec<usize> = gg
                .iter()
                .flat_map(|d| d.start..=d.end)
                .filter(|&k| p[k] > 0 && !r.slot_singable(p[k]))
                .collect();
            let want: Vec<usize> = (0..p.len()).filter(|&k| p[k] > 0 && !r.slot_singable(p[k])).collect();
            assert_eq!(got, want, "拆组只改「谁陪着走多深」,不许改「哪些音被救」");
            for w in gg.windows(2) {
                assert!(w[0].end < w[1].start, "拆出来的组不许重叠:{gg:?}");
            }
            let whole: Vec<i64> = p.iter().copied().filter(|&x| x > 0).collect();
            let dead: Vec<i64> = whole.iter().copied().filter(|&x| !r.slot_singable(x)).collect();
            let ws = minimal_rescue_shift_capped(
                &dead, &whole, &r, RescueTuning::today().landing, LANDING_RATIO_TWO_ST,
            )
            .expect("整句本来就有落点");
            for d in gg {
                assert!(d.shift.abs() <= ws.abs(), "{d:?} 比整句({ws:+})还深 —— 拆组只许变浅或不变");
            }
        }
    }

    /// ⭐⭐⭐ S159z 主刀 —— **夹在两个死音中间的可唱段,够长就把组拆开**。
    ///
    /// 机理与定价在 [`SPLIT_MIN_INTERIOR_NOTES`] 的 doc(用户点名的 `notes[685..=693]`:
    /// 组里只有第一个音真的需要救,后面五个音在音域内却陪着下潜 15 个半音 = 每个白丢 19.7 dB 高频)。
    ///
    /// ⛔ 三条阴性对照,少一条这条判据就可能是「恒真」:
    /// ⑴ **夹心 2 个音必须不拆**(移出去的交界 1 < 新增的边 2)—— 门限真的在起作用;
    /// ⑵ **被救的死音一个不许多、一个不许少** —— 拆组只改「谁陪着走多深」;
    /// ⑶ **拆出来的两组不许重叠** —— 第一版就是在这里错的(邻簇同时认领夹心 ⇒ 贴两次)。
    ///
    /// ⛔ 变异(写这条判据时逐个真跑过):
    /// * `SPLIT_MIN_INTERIOR_NOTES` 改成 4 ⇒ ⑵ 读出一组,**红**;
    /// * `lo`/`hi` **两侧一起**改回第一版的越界写法(`clusters[ci ∓ 1]` 那两行)
    ///   ⇒ ⑶ 的**裁剪关掉那一臂**读到 `[1..6]` 与 `[4..7]` 重叠,**红**。
    ///   ⚠ 只改 `hi` 一侧不会重叠,而且**只在出厂臂上断言的话连两侧一起改也是绿的**
    ///   —— 出厂裁剪会把边界收回去。这条判据一开始就是那样写的,是个空判据;
    /// * 把 `dead` 从簇改回整句 ⇒ 第二组读 −15 而不是 −5,深度那条断言**红**。
    #[test]
    fn an_interior_run_of_three_singable_notes_splits_the_phrase() {
        let r = pya_like();
        // ⚠⚠ S159zi —— **显式关掉二级拆**([`SPLIT_MIN_COST_DEFAULT`])。这条判据钉的是**一级**
        // (夹心是可唱音),而 ⑴ 那条阴性对照(夹心 2 个音 ⇒ 不许拆)会被二级拆开
        // ⇒ 它就不再是「一级的门限真的在起作用」的探针了。
        // ⛔ 隔离要靠关掉另一把刀,不是照新结果改期望值,也不是去改夹具的时长
        //    (那是绕路,而且下次一调门槛它还会再断一次)。
        let one_level = RescueTuning::today().with_split_cost(f32::INFINITY);
        let plan = |p: &[i64]| {
            let f = secs(p.len());
            dead_only_plan_with(p, &f, 0, &r, one_level).0
        };

        // ⑴ 阴性对照:夹心只有 2 个音 ⇒ 不许拆。
        let two = [0, 90, 80, 78, 75, 68, 80, 0];
        assert_eq!(plan(&two).len(), 1, "夹心 2 个音:移出去的交界(1)< 新增的边(2)⇒ 不许拆");

        // ⭐ 夹心 3 个音 ⇒ 拆成两组,而且**夹心一个音都不在任何一组里**。
        let three = [0, 90, 80, 78, 75, 68, 71, 80, 0];
        let g = plan(&three);
        assert_eq!(g.len(), 2, "夹心 3 个音 ⇒ 拆成两组(读到 {g:?})");
        assert_eq!((g[0].start, g[0].end), (1, 3), "第一组 = 前三个死音");
        assert_eq!((g[1].start, g[1].end), (7, 7), "第二组 = 最后那个死音");
        for k in 4..=6 {
            assert!(
                !g.iter().any(|d| (d.start..=d.end).contains(&k)),
                "夹心音 {k} 必须留在 base —— 它本来就唱得动"
            );
        }

        // ⑶ ⛔⛔ **不许重叠**(第一版就是在这里错的:邻簇同时认领夹心 ⇒ 拼接器贴两次)。
        //
        // ⚠ 这一条**必须在裁剪关掉的臂上**量。写这条判据时我先只在出厂臂上断言,
        // 拿第一版那两行去做变异 —— **绿的**:出厂裁剪(头尾各 500 ms)会把两簇的边界
        // 各自收回自己的死音段,正好把重叠盖住 ⇒ 那条断言当时是**空判据**。
        // ⇒ 关掉裁剪,越界写法就直接露出来。
        assert!(g[0].end < g[1].start, "拆出来的组不许重叠:{g:?}");
        let untrimmed = dead_only_plan_with(
            &three,
            &secs(three.len()),
            0,
            &r,
            RescueTuning::new(None, None).with_split_cost(f32::INFINITY),
        )
        .0;
        assert_eq!(untrimmed.len(), 2, "裁剪关掉也要拆成两组(读到 {untrimmed:?})");
        assert!(
            untrimmed[0].end < untrimmed[1].start,
            "裁剪关掉时更不许重叠(内部那一侧本来就一格都不许伸):{untrimmed:?}"
        );
        assert_eq!(
            (untrimmed[0].start, untrimmed[0].end, untrimmed[1].start, untrimmed[1].end),
            (1, 3, 7, 7),
            "裁剪关掉时,内部簇的边界就是它自己的死音段"
        );

        // ⑵ 被救的死音一个不许多、一个不许少 —— 与不拆的那一版比对同一批下标。
        let dead_of = |p: &[DeadGroup], src: &[i64]| {
            let mut v: Vec<usize> = p
                .iter()
                .flat_map(|d| d.start..=d.end)
                .filter(|&k| src[k] > 0 && !r.slot_singable(src[k]))
                .collect();
            v.sort_unstable();
            v
        };
        assert_eq!(dead_of(&g, &three), vec![1, 2, 3, 7], "四个死音全都还在,而且没多出别的");

        // ⭐ 拆开之后每一簇**各自**求落点 ⇒ 只需要浅救的那一簇不再被顶音拖下去。
        //
        // ⛔ 这里**不能**直接断言「第二组更浅」:落点旋钮(出厂 `Some(3)`)本来就允许
        // 为了更干净的落点往深里多看几个半音,而这个夹具的 `low_ratio` 恰好偏深 ——
        // 写这条判据时我先写成 `g[1] < g[0]`,真跑读到 **−14 vs −14**,是**判据错了不是代码错了**。
        // ⇒ 要量的是「**需求**上的差」,所以把落点旋钮关掉再看一次:那一档
        // `minimal_rescue_shift` 取**最浅的合格位移**,两簇的差就是纯粹的需求差。
        let bare = dead_only_plan_with(
            &three,
            &secs(three.len()),
            0,
            &r,
            RescueTuning::new(Some((TRIM_HEAD_MS, TRIM_TAIL_MS)), None).with_split_cost(f32::INFINITY),
        )
        .0;
        assert_eq!(bare.len(), 2, "落点关掉也要拆成两组(读到 {bare:?})");
        assert!(
            bare[1].shift.abs() < bare[0].shift.abs(),
            "顶音 80 那一簇的**需求**必须明显浅于顶音 90 那一簇,读到 {} vs {}",
            bare[1].shift,
            bare[0].shift
        );
        // ⛔ 78 而不是 76:76 只有**开着落点旋钮**才够得着(见
        // `a_passenger_may_not_veto_the_landing_the_dead_note_needs` 的 ⑴/⭐ 两臂),
        // 这一档旋钮是关的 ⇒ 两簇都停在最浅的合格落点 78。写这条判据时我先写成 76,
        // 真跑读到 `(78, 78)` —— 又是**判据错了不是代码错了**。
        assert_eq!(
            (90 + bare[0].shift, 80 + bare[1].shift),
            (78, 78),
            "两簇各自停在最浅的合格落点(78)—— 边界写死成字面量,不许引用被测的常量"
        );
        // ⭐⭐⭐ 这一刀买到的就是这两个数之差:顶音 90 那一簇要 −12,而只有一个 80 的那一簇
        // 只要 **−2**。不拆的话后者会跟着走 −12 ⇒ 白白多渲低 **10 个半音 = 高频 −13.1 dB**
        // (定价见 [`SPLIT_MIN_INTERIOR_NOTES`])。
        assert_eq!((bare[0].shift, bare[1].shift), (-12, -2), "省下来的正是这 10 个半音");
        // 而且**没有任何一组**比整句一起救还深 —— 拆组只会让深度变浅或不变。
        let whole: Vec<i64> = (1..=7).map(|k| three[k]).collect();
        let dead: Vec<i64> = whole.iter().copied().filter(|&p| !r.slot_singable(p)).collect();
        let whole_shift =
            minimal_rescue_shift_capped(&dead, &whole, &r, RescueTuning::today().landing, LANDING_RATIO_TWO_ST)
                .expect("整句本来就有落点");
        for d in &g {
            assert!(
                d.shift.abs() <= whole_shift.abs(),
                "拆出来的组 {d:?} 比整句({whole_shift:+})还深 —— 拆组只许让深度变浅或不变"
            );
        }
    }

    /// ⭐⭐ S157 主刀 —— **一个乘客的 damage 不许否决死音的落点**。
    ///
    /// 今天的 `worst` 对 **死音 ∪ 乘客** 取 max,于是用户听出问题的那个音卡在落点 78:
    /// 走 −14 时三个死音落在 76/66/64(damage 全 0,而 76 的 `low_ratio` 0.21 比 78 的 0.39
    /// 好将近一倍),可两个乘客落到 MIDI 61 / 54(`low_ratio` 0.572 / 0.573,**刚过 0.55**)
    /// ⇒ damage 0.165 / 0.1725 > [`LANDING_DAMAGE_EPS`] ⇒ −14 被踢出候选。
    ///
    /// ⛔ **三条阴性对照,少一条这条判据就可能是「恒真」而不是「有牙」**:
    /// ⑴ 默认臂(`landing == None`)必须**仍然是 −12** —— 键序那一笔不许泄进今天的臂;
    /// ⑵ 被救的音一个不许多、一个不许少;
    /// ⑶ 把两个乘客换成落点干净的音高之后,开与关必须给出**同一个**答案
    ///    (否则「赢」可能来自这一刀之外的任何东西)。
    ///
    /// ⛔ 变异(写这条判据时逐个真跑过):
    /// * 键序改回「`worst`(死音∪乘客)排在 `thinner` 前面」⇒ 读 −12,**红**;
    /// * [`LANDING_RATIO_TWO_ST`] 退回 12 或 13 ⇒ 读 −12 / −13,**红**;
    /// * 把 MIDI 54/61 的 `low_ratio` 降到 0.20(= 拿掉被测机理)⇒ 开与关都读 −14,
    ///   ⑶ 那条阴性对照当场**红** ⇒ 说明它真的在盯着「乘客否决」这件事。
    /// ⚠⚠ **S159z 把这个夹具的「夹心」从 5 个音缩到 2 个 —— 不是为了让它变绿。**
    ///
    /// [`SPLIT_MIN_INTERIOR_NOTES`] 落地之后,原来那个 `[.., 75, 68, 71, 70, 71, ..]`
    /// 正好是**要被拆开**的形状 ⇒ 五个乘客会被整段留在 base,而这条判据钉的
    /// 「**乘客**否决落点」在那个夹具上**结构上不会再发生** ⇒ 照着新结果改期望值
    /// 会把它变成一条**空判据**(S129 那一族血训)。
    /// ⇒ 正确的改法是把夹心缩到 2 个音(不触发拆组),机理原样保留:
    /// 那两个乘客在 −14 上仍然落到 MIDI **61 / 54**(`low_ratio` 0.572 / 0.573),
    /// 仍然是唯一能否决 −14 的东西 —— ⑶ 那三条阴性对照逐条仍然有牙。
    /// ⭐ 拆组本身另有判据:[`an_interior_run_of_three_singable_notes_splits_the_phrase`]。
    #[test]
    fn a_passenger_may_not_veto_the_landing_the_dead_note_needs() {
        let phrase = [0, 90, 80, 78, 75, 68, 80, 0];
        let fr = secs(phrase.len());
        // ⚠⚠ S159zi —— **显式关掉二级拆**([`SPLIT_MIN_COST_DEFAULT`])。
        // 这个夹具正好也是二级那一刀的形状(顶音 90 要 −12,后面五个只要 −4)⇒ 两级同时开火时
        // 这条判据钉的「**乘客**否决落点」**结构上不会再发生**(乘客被拆到自己那一组里去了)
        // ⇒ 照新结果改期望值 = 又一条空判据。⛔ 正确的隔离是**关掉另一把刀**。
        // ⭐ 二级本身另有判据:[`a_run_of_dead_notes_splits_where_the_depth_requirement_drops`]。
        let iso = |t: RescueTuning| t.with_split_cost(f32::INFINITY);
        let r = pya_like();

        // ⑴ 默认臂 = 生产实测的那一档(用户听到的就是它):落点 78。
        // ⛔ S157c:`today()` 已翻成 `Some(3)` ⇒ 「S151 之前那条臂」必须显式写。
        let (today, _) = dead_only_plan_with(&phrase, &fr, 0, &r, iso(RescueTuning::new(None, None)));
        assert_eq!(
            today,
            vec![DeadGroup { start: 1, end: 6, shift: -12 }],
            "默认臂必须逐字是今天:顶音 90 停在落点 78"
        );

        // ⭐ 旋钮开着:同样一批合格落点里,死音自己的 `low_ratio` 说话 ⇒ 顶音落到 76。
        let (on, _) = dead_only_plan_with(&phrase, &fr, 0, &r, iso(RescueTuning::new(None, Some(3))));
        assert_eq!(
            on,
            vec![DeadGroup { start: 1, end: 6, shift: -14 }],
            "开着:顶音必须够得着 76(low_ratio 0.21),而不是被两个乘客的 0.17 damage 挡在 78"
        );
        assert_eq!(90 + on[0].shift, 76, "落点写死成字面量 —— 边界不许引用被测的常量");

        // ⑵ 这一刀只决定**落在哪**,不许改**哪些音被救**。
        let dead_of = |p: &[DeadGroup]| {
            let mut v: Vec<usize> = p
                .iter()
                .flat_map(|g| g.start..=g.end)
                .filter(|&k| phrase[k] > 0 && !r.slot_singable(phrase[k]))
                .collect();
            v.sort_unstable();
            v
        };
        assert_eq!(dead_of(&today), dead_of(&on), "被救的死音一个不许多、一个不许少");
        assert_eq!(dead_of(&on), vec![1, 2, 3, 6], "而且就是那四个(90 / 80 / 78 / 80)");

        // ⑶ ⛔⛔ **夹具有效性** —— 少了这一段,上面那个 −14 可能只是「−12 从来没被谁挡过」。
        //    这里直接把被测机理量出来:同一个 `damage_at`,同一批音高。
        let worst = |set: &[i64], s: i64| -> f32 {
            set.iter().map(|&p| r.damage_at((p + s) as f32).unwrap_or(DAMAGE_MAX)).fold(0.0, f32::max)
        };
        let pax = [75i64, 68];
        let dead = [90i64, 80, 78, 80];
        assert_eq!(worst(&pax, -12), 0.0, "−12 上两个乘客必须全都干净");
        assert!(
            worst(&pax, -14) > LANDING_DAMAGE_EPS,
            "−14 上乘客的 damage 必须真的越过容差(读到 {:.4},容差 {LANDING_DAMAGE_EPS}),\
             否则这条判据钉的不是「乘客否决」",
            worst(&pax, -14)
        );
        assert_eq!(worst(&dead, -12), worst(&dead, -14), "死音在两档上一样干净 ⇒ 差别只来自乘客");
        assert!(
            r.thinness(76).unwrap() < r.thinness(78).unwrap() - LANDING_THIN_EPS,
            "而顶音自己的 low_ratio 必须真的更好(76 = {:.3} vs 78 = {:.3}),否则没什么可赢的",
            r.thinness(76).unwrap(),
            r.thinness(78).unwrap()
        );

        // ⑷ ⛔ **乘客那把键只是降级了,没有被删掉。**
        //    构造一个「死音 damage 与 low_ratio 在几档上全都打平、只有乘客不同」的局面:
        //    死音 84(落点 79..76 的 low_ratio 在这个夹具里是阶梯,所以换一个平坦的记录),
        //    见 `the_passenger_key_still_breaks_a_tie_it_is_just_no_longer_the_first_key`。
    }

    /// ⛔ 与 [`a_passenger_may_not_veto_the_landing_the_dead_note_needs`] 成对:那一笔把
    /// 「乘客 damage」这把键从 `thin` 上面挪到了下面,**而不是删掉它**。少了这一条,
    /// 「降级」与「删掉」在仓里读出来是同一个东西。
    ///
    /// 局面:死音的 `low_ratio` 在候选落点上**完全平坦**(每一档都一样干净)⇒ 前两把键全部打平
    /// ⇒ 只剩乘客那把说话。它必须选**对乘客更友善**的那一档,而不是最浅的那一档。
    /// ⛔ 变异:把最后那一层 `worst` 过滤删掉 ⇒ 读 −5(最浅),**红**。
    #[test]
    fn the_passenger_key_still_breaks_a_tie_it_is_just_no_longer_the_first_key() {
        let mut semis = serde_json::Map::new();
        for m in 36..=79i64 {
            // 死音的落点(79..76)与绝大多数音高一样干净;只有 **55** 是薄的,
            // 而 55 恰好是那个乘客在**最浅**那一档上的落点。
            let lr = if m == 55 { 0.60 } else { 0.20 };
            semis.insert(m.to_string(), serde_json::json!([1, 1.0, -1.0, lr]));
        }
        for m in 80..=96i64 {
            semis.insert(m.to_string(), serde_json::json!([9999, 0.0, -21.0, 0.90]));
        }
        let r = speaker_range(
            &config_with(serde_json::json!({
                "usable": [36, 79], "usable_auto": [36, 79], "comfort": [36, 79],
                "semitones": serde_json::Value::Object(semis)
            })),
            0,
        )
        .unwrap();
        // 死音 84 ⇒ 最浅 −5(落点 79);预算 3 ⇒ 候选 −5..−8,死音那两把键在四档上全平。
        // 乘客 60:−5 落到 **55**(薄),−6/−7/−8 落到 54/53/52(干净)。
        let (on, _) =
            dead_only_plan_with(&[0, 84, 60, 0], &secs(4), 0, &r, RescueTuning::new(None, Some(3)));
        assert_eq!(
            on[0].shift, -6,
            "死音两把键全平之后,乘客那把仍然要说话:−5 把乘客扔在薄槽 55 上 ⇒ 走 −6"
        );
        // 阴性对照:把乘客挪到一个在四档上都不薄的音高 ⇒ 必须回到最浅那一档。
        let (flat, _) =
            dead_only_plan_with(&[0, 84, 61, 0], &secs(4), 0, &r, RescueTuning::new(None, Some(3)));
        assert_eq!(flat[0].shift, -5, "乘客不薄时三把键全平 ⇒ 取最浅,否则上面那条不是乘客给的");
    }

    /// ⭐ S157 —— [`LANDING_RATIO_TWO_ST`] 的**值**,上下两侧都钉。
    ///
    /// ⛔ 为什么单独一条:`optional_depth_stops_where_the_synthesis_starts_dropping_the_source`
    /// 只钉「上限存在」,把这个常数从 12 一路改到 15 它**全绿**(S157 扫过)。而这个数刚被
    /// 重新定过价(12 → 14),再让它躺在一条读不出它的判据下面,就等于没有记录。
    ///
    /// 下侧(≥14 才咬得住)由 `a_passenger_may_not_veto_the_landing_the_dead_note_needs` 钉;
    /// 这里钉**上侧**:一个 `|最浅| = 13` 的组,预算只剩 `14 − 13 = 1`,
    /// 于是哪怕 `low_ratio` 一路奖励更深,它也只许多花一个半音。
    /// ⛔ 变异:常数改成 16 ⇒ 预算变 3 ⇒ 读 −16,**红**。
    #[test]
    fn the_optional_depth_cap_is_worth_exactly_what_the_ear_line_paid_for() {
        let mut semis = serde_json::Map::new();
        for m in 36..=79i64 {
            // 越低越不薄 ⇒ 排序想一路往下走;台阶远大于 `LANDING_THIN_EPS`,否则「打平」
            // 会把这条判据变成一句空话(S151 就这么写坏过一次)。
            let lr = match m {
                m if m >= 79 => 0.60, // 唯一一格越过 0.55 ⇒ 它是 damage 上就该被跳过的那一档
                78 => 0.45,
                77 => 0.35,
                _ => 0.20,
            };
            semis.insert(m.to_string(), serde_json::json!([1, 1.0, -1.0, lr]));
        }
        for m in 80..=96i64 {
            semis.insert(m.to_string(), serde_json::json!([9999, 0.0, -21.0, 0.90]));
        }
        let r = speaker_range(
            &config_with(serde_json::json!({
                "usable": [36, 79], "usable_auto": [36, 79], "comfort": [36, 79],
                "semitones": serde_json::Value::Object(semis)
            })),
            0,
        )
        .unwrap();
        // 死音 92,落点上限 79 ⇒ 最浅 −13;预算 = 14 − 13 = 1 ⇒ 只许走到 −14。
        let (on, _) = dead_only_plan_with(&[0, 92, 0], &secs(3), 0, &r, RescueTuning::new(None, Some(3)));
        assert_eq!(on[0].shift, -14, "|最浅| = 13 时预算只剩 1 ⇒ 到 −14 为止(常数 16 会读 −16)");
        assert_eq!(92 + on[0].shift, 78, "落点写死成字面量");
        // 阳性对照:同一个记录、同一个旋钮,一个最浅只有 −4 的组必须真的花掉那三个半音,
        // 否则上面那条「只走到 −14」可能只是「预算从来没起过作用」。
        let (shallow, _) =
            dead_only_plan_with(&[0, 83, 0], &secs(3), 0, &r, RescueTuning::new(None, Some(3)));
        assert_eq!(shallow[0].shift, -7, "够得着的组要花满 3 个半音,不然预算这条线是空的");
    }

    /// S159 —— ⛔⛔ **深度上限只在「最浅落点本来就很深」时才咬 —— 所以只扫它一维必然读出一张平表。**
    ///
    /// `budget = clamp(extra, LANDING_MAX_EXTRA_DEPTH, max(0, cap − |最浅|))`。
    /// 出厂 `extra = 3` ⇒ 只要 `|最浅| ≤ cap − 3`,`room ≥ extra`,**cap 一个字节都不改结果**。
    /// 这就是 S159 给 `LANDING_RATIO_TWO_ST` 重新定价时那张表的形状:
    /// **原 key 炉心融解上 cap 从 12 扫到 24 每一行逐字相同**(那份谱的 `|最浅|` 全 ≤ 11)。
    ///
    /// ⇒ ⛔ **别把「扫了没变」读成「这个值是对的」** —— 要先证明那个区间里它**够得着**。
    /// 这与 S157 记的那条同族:「一条只在窄区间上采样的曲线,读出来的单调性是**那个区间**的性质」。
    ///
    /// ⛔ 变异:`room` 那一行改成 `(ratio_two_cap - shallowest.abs()).max(3)`(= 让 cap 永不咬)
    /// ⇒ 下面第二组断言红;把 `.max(LANDING_MAX_EXTRA_DEPTH)` 去掉 ⇒ 第三组红。
    #[test]
    fn the_depth_cap_only_bites_once_the_shallowest_landing_is_already_deep() {
        let mut semis = serde_json::Map::new();
        for m in 36..=79i64 {
            // 一路往下越来越不薄 ⇒ 排序永远想走到预算的最深处 ⇒ 读出来的就是预算本身。
            semis.insert(m.to_string(), serde_json::json!([1, 1.0, -1.0, 0.60 - 0.01 * (79 - m) as f64]));
        }
        for m in 80..=110i64 {
            semis.insert(m.to_string(), serde_json::json!([9999, 0.0, -21.0, 0.90]));
        }
        let r = speaker_range(
            &config_with(serde_json::json!({
                "usable": [36, 79], "usable_auto": [36, 79], "comfort": [36, 79],
                "semitones": serde_json::Value::Object(semis)
            })),
            0,
        )
        .unwrap();
        let depth = |note: i64, cap: i64| -> i64 {
            let t = RescueTuning::new(None, Some(3)).with_cap(cap);
            dead_only_plan_with(&[0, note, 0], &secs(3), 0, &r, t).0[0].shift
        };
        // ⑴ **够不着 cap 的那一档:换 cap 一个字节都不动。**死音 83 ⇒ 最浅 −4,`room = cap − 4 ≥ 3`。
        for cap in [12i64, 14, 16, 20, 24] {
            assert_eq!(depth(83, cap), -7, "cap {cap}:|最浅| = 4 时 cap 不该起作用");
        }
        // ⑵ **咬得到的那一档:落点随 cap 一格一格走。**死音 92 ⇒ 最浅 −13。
        //    边界写实测字面量(⛔ 不许拿被测的常量算期望值 —— 那是恒真)。
        assert_eq!(depth(92, 14), -14, "cap 14 ⇒ room 1");
        assert_eq!(depth(92, 15), -15, "cap 15 ⇒ room 2");
        assert_eq!(depth(92, 16), -16, "cap 16 ⇒ room 3(= extra,到此为止)");
        assert_eq!(depth(92, 20), -16, "cap 再大也被 extra = 3 封住");
        // ⑶ **护栏是下限不是替代品**:`|最浅| ≥ cap` 时仍然保留 `LANDING_MAX_EXTRA_DEPTH` 那一格。
        assert_eq!(depth(92, 13), -14, "cap 13 < |最浅| ⇒ room 0,但护栏保底 1 格");
        assert_eq!(depth(92, 5), -14, "cap 荒谬地小也不许比今天更浅");
    }

    /// ⛔⛔ S157 —— `RescueTuning::today()` 的 doc 说「Exactly what ships today」,
    /// 而它的 `landing` 原来是**字面量 `None`**,不是 [`LANDING_DEFAULT`]。两个值今天恰好相同,
    /// 所以那是一条**没有人核验过的话**:翻默认的那天,经它取臂的十几条判据会**继续静默地测旧臂**。
    /// ⭐ 与 S156 在 `psola.rs` 上抓到的是同一条形状,换了个文件又长了一次。
    /// ⚠ 这条判据今天**读不出差别**(两个常量都是 `None`)—— 它是一条**机械联系**:
    ///    谁把 `today()` 再写死一次,它当场红。
    #[test]
    fn the_today_tuning_is_literally_the_shipped_defaults() {
        assert_eq!(RescueTuning::today().trim, TRIM_DEFAULT, "trim 必须从常量取");
        assert_eq!(RescueTuning::today().landing, LANDING_DEFAULT, "landing 必须从常量取");
        assert_eq!(RescueTuning::today(), RescueTuning::new(TRIM_DEFAULT, LANDING_DEFAULT));
    }

    /// ⛔⛔ S157c —— **改了生产默认却没 bump 版本,在这之前是【零红】**,
    /// 而那不是一个错误、是用户听到一条陈缓存(S150 的原话)。
    /// ⇒ 把「今天的生产默认」压成一个指纹字符串写死在这里,并且**同一条判据**去核对
    /// 两个版本字面量(`RANGE_ALGO_VERSION` 在 TS 里、`audition_cache_tag` 在 Rust 里)。
    /// 谁动了任何一个默认,这条当场红,而红的措辞直接告诉他要改哪三处。
    ///
    /// ⭐ 用 `include_str!` 读那两份源码 —— 与本仓已有的那几处零漂移技巧同款
    /// (`commands/inference.rs` 读 `vocalNotes.ts`、`commands/settings.rs` 读 `Settings.tsx`)。
    #[test]
    fn changing_a_production_default_forces_a_paired_version_bump() {
        let fp = format!(
            "trim={:?} landing={:?} ratio2={} depth={} frac={} win={} xgrain={} lpc={}              hp={} hp_ms={} envfix={} bridge={} lock={} kappa={} join={} wininv={}",
            TRIM_DEFAULT,
            LANDING_DEFAULT,
            LANDING_RATIO_TWO_ST,
            LANDING_MAX_EXTRA_DEPTH,
            parse_frac_transport(None),
            parse_win_periods(None),
            parse_xgrain(None),
            parse_lpc_order(None),
            parse_infrasonic_hp(None),
            parse_infrasonic_ms(None),
            parse_env_restore_ms(None),
            parse_bridge_unvoiced_ms(None),
            parse_phase_lock(None),
            DEFAULT_FORMANT_KAPPA,
            parse_join_rests(None),
            // S159 —— ⚠ 这一个进指纹,但它**不该**触发版本 bump:它不改音频,只改秒数
            //    (理由与那三条承重判据写在 `windowed_inverse()` 的 doc 里)。
            //    进指纹的意义是「下一个人翻它的时候必须来这里改一行,于是不得不读那段 doc」。
            parse_windowed_inverse(None),
        );
        assert_eq!(
            fp,
            "trim=Some((500.0, 500.0)) landing=Some(3) ratio2=14 depth=1 frac=true win=1 xgrain=1 lpc=0              hp=true hp_ms=0 envfix=0 bridge=30 lock=0.3 kappa=0 join=false wininv=true",
            "⛔ 生产默认变了。必须同时改三处:①这条判据里的指纹              ②`src/lib/vocal/vocalRender.ts` 的 `RANGE_ALGO_VERSION`              ③`src-tauri/src/commands/audition.rs` 的 `_sNNNx_` cache tag ——              漏掉后两个不是错误,是用户听到一条陈缓存(S150)。"
        );
        const TAG: &str = "s159g";
        let ts = include_str!("../../../src/lib/vocal/vocalRender.ts");
        assert!(
            ts.contains(&format!("RANGE_ALGO_VERSION = \"{TAG}\"")),
            "vocalRender.ts 的 RANGE_ALGO_VERSION 没跟着 bump 到 {TAG}"
        );
        let au = include_str!("../commands/audition.rs");
        assert!(
            au.contains(&format!("\"_{TAG}_ru")),
            "audition.rs 的 cache tag 没跟着 bump 到 _{TAG}_"
        );
    }

    /// ⛔⛔ S158 —— **每一个出厂默认,必须在它自己那段 doc 的【第一行】写明白。**
    ///
    /// 起因是一次盘点:三个已经翻过的默认,它们**常量那一侧的 doc 改对了**,而**访问函数
    /// 那一侧的 doc 留在原地说反话** —— 而后者才是读代码的人第一眼看到的东西:
    /// * `frac_transport()` 写着「**Default off.** … blind test first, flip after」,
    ///   而 `FRAC_TRANSPORT_DEFAULT = true`(S157c 翻的);
    /// * `xgrain()` 写着「## ⛔ **为什么它默认关**」,而 `XGRAIN_DEFAULT = 1.0`(S156 翻的);
    /// * `phase_lock()` 写着「⛔ **Why it is still off by default.**」,而
    ///   `PHASE_LOCK_DEFAULT = 0.30`(S150 翻的)。
    /// 三次翻默认,三次同样的漏 ⇒ 这不是手滑,是**没有任何东西盯着散文**。
    ///
    /// ⛔ 而 `changing_a_production_default_forces_a_paired_version_bump` 结构上看不见它:
    /// 那条钉的是**值**(指纹 + 两个版本字面量),散文改不改它都绿。
    ///
    /// ⇒ 这条判据要求每个 `*_DEFAULT` 的访问函数,doc 的**第一行**是
    /// `⚙ 出厂默认 = <常量初始化式逐字>`(后面可以接人话)。它不保证下面的散文不过期,
    /// 但它把「翻默认」和「改那段 doc」**焊在同一次编辑里**,而且把结论顶到第一行 ——
    /// 底下万一还有陈货,读者第一眼就看得见矛盾。
    /// ⚠ 同时它要求这张表**覆盖本文件里每一个 `*_DEFAULT`**,新加一个默认却不登记会红。
    #[test]
    fn every_shipped_default_is_declared_at_the_top_of_its_own_doc() {
        const PAIRS: &[(&str, &str)] = &[
            ("LANDING_DEFAULT", "fn parse_landing("),
            ("TRIM_DEFAULT", "fn parse_trim("),
            ("TRIM_MIN_COST_DEFAULT", "const TRIM_MIN_COST_DEFAULT"),
            ("SPLIT_MIN_COST_DEFAULT", "const SPLIT_MIN_COST_DEFAULT"),
            ("JOIN_RESTS_DEFAULT", "pub fn join_rests_enabled("),
            ("FRAC_TRANSPORT_DEFAULT", "pub fn frac_transport("),
            ("PHASE_LOCK_DEFAULT", "pub fn phase_lock("),
            ("INFRASONIC_HP_DEFAULT", "pub fn infrasonic("),
            ("INFRASONIC_MS_DEFAULT", "pub fn infrasonic_fixed_ms("),
            ("XGRAIN_DEFAULT", "pub fn xgrain("),
            ("LPC_ORDER_DEFAULT", "pub fn lpc_order("),
            ("WIN_PERIODS_DEFAULT", "pub fn win_periods("),
            ("EDGE_FILL_DEFAULT", "pub fn edge_fill("),
            ("ENV_RESTORE_MS_DEFAULT", "pub fn env_restore_ms("),
            ("BRIDGE_UNVOICED_MS_DEFAULT", "pub fn bridge_unvoiced_ms("),
            ("WINDOWED_INVERSE_DEFAULT", "pub fn windowed_inverse("),
        ];
        let src = include_str!("vocal_range.rs");
        let lines: Vec<&str> = src.lines().collect();

        // ⓐ 覆盖面:本文件顶层的每一个 `const *_DEFAULT` 都必须登记在上表里。
        let mut found: Vec<&str> = Vec::new();
        for l in &lines {
            if let Some(rest) = l.strip_prefix("const ") {
                if let Some(name) = rest.split(':').next().map(str::trim) {
                    if name.ends_with("_DEFAULT") {
                        found.push(name);
                    }
                }
            }
        }
        for name in &found {
            assert!(
                PAIRS.iter().any(|(c, _)| c == name),
                "新加了默认 `{name}` 却没登记到 `every_shipped_default_is_declared_*` 的表里 \
                 —— 那条默认此后没有任何东西盯着它的 doc"
            );
        }
        assert_eq!(found.len(), PAIRS.len(), "表与文件里的默认数量对不上:{found:?}");

        // ⓑ 逐条:doc 第一行必须逐字带上常量的初始化式。
        for (konst, sig) in PAIRS {
            let decl = lines
                .iter()
                .find(|l| l.starts_with(&format!("const {konst}")))
                .unwrap_or_else(|| panic!("找不到 `const {konst}`"));
            let expr = decl
                .split_once('=')
                .and_then(|(_, r)| r.rsplit_once(';'))
                .map(|(v, _)| v.trim())
                .unwrap_or_else(|| panic!("`const {konst}` 的初始化式解析不出来"));
            let at = lines
                .iter()
                .position(|l| l.starts_with(sig))
                .unwrap_or_else(|| panic!("找不到 `{sig}` —— 表里的锚点过期了"));
            // 往上收一段连续的 doc(跳过属性行)
            let mut top = at;
            while top > 0 {
                let p = lines[top - 1].trim_start();
                if p.starts_with("///") || p.starts_with("#[") {
                    top -= 1;
                } else {
                    break;
                }
            }
            let first = lines[top..at]
                .iter()
                .find(|l| l.trim_start().starts_with("///"))
                .unwrap_or_else(|| panic!("`{sig}` 头上一行 doc 都没有"));
            let want = format!("/// ⚙ 出厂默认 = {expr}");
            assert!(
                first.trim_start().starts_with(&want),
                "`{sig}` 的 doc 第一行必须是 `{want}`(后面可以接人话),实际是:\n  {first}\n\
                 ⇒ 翻默认时那段 doc 没跟着改。三个已经翻过的默认都在这一处漏过(见本判据的 doc)。"
            );
        }
    }

    #[test]
    fn the_landing_knob_is_off_by_default_and_parses() {
        // S157c —— **默认已经翻成 Some(3)**(整曲实测:薄区落点 30%→12%、ぴゃ 落点 78→76)。
        assert_eq!(parse_landing(None), LANDING_DEFAULT, "空环境必须给出默认");
        assert_eq!(LANDING_DEFAULT, Some(3), "边界写字面量,别引用被测的常量");
        assert_eq!(parse_landing(Some("")), LANDING_DEFAULT);
        assert!(parse_landing(Some("0")).is_none(), "显式关得掉 —— 抱怨时要能用同一个二进制渲旧臂");
        // ⛔⛔ 翻默认之后「垃圾值往哪边倒」必须跟着翻:垃圾不许**静默关掉**一个已上线的修法。
        assert_eq!(parse_landing(Some("3")), Some(3));
        assert_eq!(parse_landing(Some(" 4 ")), Some(4));
        for junk in ["x", "-1", "999", "1.5", ""] {
            assert_eq!(parse_landing(Some(junk)), LANDING_DEFAULT, "垃圾 {junk} 必须回落到默认");
        }
    }

    /// 出厂那一档到底给出什么(⚠ **不是**常量等于常量):构造一个头尾都够本的乐句,
    /// 断言它**两侧都切**。⭐ S158f 之前这条叫 `the_shipped_trim_only_ever_cuts_the_tail`,
    /// 期望是 `{1, 4}`(只切尾)—— 用户听完 `04_头尾都裁` 之后头裁也翻开了,理由在 `TRIM_HEAD_MS`。
    #[test]
    fn the_shipped_trim_cuts_both_sides() {
        let on = parse_trim(Some("1")).expect("`1` = 出厂那一档");
        let nn = [0, 73, 73, 73, 85, 73, 73, 0];
        let fr = [10, 100, 100, 100, 40, 100, 100, 10]; // 头 6.0 s、尾 4.0 s,都远超任何门限
        let (plan, _) = dead_only_plan_with(&nn, &fr, 0, &dxl_like(), trim_arms(Some(on)).1);
        assert_eq!(
            plan,
            vec![DeadGroup { start: 4, end: 4, shift: -6 }],
            "出厂档:两侧的乘客都放掉,只留死音本身"
        );
    }

    /// 旋钮本身:**默认开 = 头尾都裁 500 ms**(S158f;当中翻过、撤过、又翻回来、再放开头裁,
    /// 全过程在 [`TRIM_DEFAULT`] 与 [`TRIM_HEAD_MS`] 的 doc 里),而且解析是纯函数
    /// (测试里读进程环境既会被并行污染,又会在别人 export 了变量时**静默通过** ——
    /// S150 在 `parse_phase_lock` 上付过学费)。
    #[test]
    fn the_passenger_trim_is_on_by_default_and_the_knob_parses() {
        // ⛔ 边界写**字面量**:原来写的 `assert_eq!(on, (TRIM_HEAD_MS, TRIM_TAIL_MS))` 是
        //    「常量等于自己」⇒ 把 500 改成 100 或 3000 都不会红(S158 盘点时点名的空判据)。
        assert_eq!(parse_trim(Some("1")), Some((500.0, 500.0)), "`1` 档 = 头 500 / 尾 500 ms");
        assert_eq!(parse_trim(None), parse_trim(Some("1")), "⭐ 默认必须**就是**出厂那一档");
        assert_eq!(parse_trim(Some("")), parse_trim(Some("1")));
        assert!(parse_trim(Some("0")).is_none(), "显式关得掉 —— 抱怨时要能用同一个二进制渲旧臂");
        // ⛔⛔ 默认**开**之后,「垃圾往哪边倒」必须跟着翻:默认关的年代「垃圾 ⇒ 关」是对的
        //    (不许静默打开一个没验过的臂);默认开之后同一行变成「垃圾 ⇒ **静默关掉**一条
        //    已上线的修法」,而用户拿到的是一条他没要的旧臂,且没有任何一行输出会说破。
        //    (S155 在 `parse_infrasonic_hp` 上原样踩过。)
        // ⛔ S159ze —— `-inf` 必须当垃圾:门是 `freed_ms >= head_ms`,`freed >= -inf` **恒真**
        //    ⇒ 放行它等于**静默把那一侧全裁**,方向与 `inf`(永不裁)正好相反。
        for junk in ["x", "800", "800:", "800:x", "-1:0", "1:2:3", "nan:1", "-inf:500", "inf:nan"] {
            assert_eq!(parse_trim(Some(junk)), parse_trim(Some("1")),
                       "垃圾 {junk} 必须回落到**默认**,不是静默关掉");
        }
        // ⛔⛔ S159ze —— `inf` 是 [`TRIM_HEAD_MS`] 的 doc 教的「关掉单侧」写法,**必须真的关掉**。
        //    在这条断言之前它落到 `_ => TRIM_DEFAULT` ⇒ 拿到头裁**开着**的出厂臂,而且一行都不响。
        //    ⚠ 变异:把 `|| x == f32::INFINITY` 去掉 ⇒ 这两条当场红。
        assert_eq!(parse_trim(Some("inf:500")), Some((f32::INFINITY, 500.0)), "inf = 这一侧永不裁");
        assert_eq!(parse_trim(Some("500:inf")), Some((500.0, f32::INFINITY)));
        assert_ne!(parse_trim(Some("inf:500")), TRIM_DEFAULT, "它绝不许静默回落到出厂臂");
        assert_eq!(parse_trim(Some("800:300")), Some((800.0, 300.0)), "扫参数用");
        assert_eq!(parse_trim(Some(" 800 : 300 ")), Some((800.0, 300.0)));
    }

    /// S151d:同位移、中间只隔休止的两个窗必须合成一个 —— 否则我们会在**同一条 donor** 上
    /// 挖一个洞、把 base 填进去,而 donor 的收尾正响着(实测 46.40 s:−18 dB 被 30 ms 内砍到 −56)。
    #[test]
    fn two_windows_at_the_same_shift_with_only_a_rest_between_them_become_one() {
        let nn = [0, 85, 0, 85, 0];
        let fr = [10i64, 20, 3, 20, 10];
        let plan = [
            DeadGroup { start: 1, end: 1, shift: -6 },
            DeadGroup { start: 3, end: 3, shift: -6 },
        ];
        let w = dead_group_windows(&nn, &fr, &plan);
        assert_eq!(w.len(), 1, "同一条 donor 上不许挖洞");
        assert_eq!(w[0].shift, -6);
        assert_eq!((w[0].start, w[0].end), (6, 55), "合并后的窗要盖住两段与中间那个休止");
        // ⛔ 位移不同 ⇒ 必须**不**合并(那两侧本来就是两条不同的 donor)。
        let plan2 = [
            DeadGroup { start: 1, end: 1, shift: -6 },
            DeadGroup { start: 3, end: 3, shift: -7 },
        ];
        assert_eq!(dead_group_windows(&nn, &fr, &plan2).len(), 2, "位移不同不许合并");
        // ⛔ 中间夹着一个**唱音** ⇒ 必须不合并(合了等于把乘客偷偷拖进救援)。
        let nn3 = [0, 85, 73, 85, 0];
        let plan3 = [
            DeadGroup { start: 1, end: 1, shift: -6 },
            DeadGroup { start: 3, end: 3, shift: -6 },
        ];
        assert_eq!(dead_group_windows(&nn3, &fr, &plan3).len(), 2, "跨过唱音不许合并");
        // ⛔ 长休止不许桥:缺陷只有 60 ms,不设上限时窗会被撑到 18.6 s(实测)。
        let fr_long = [10i64, 20, 200, 20, 10];
        assert_eq!(dead_group_windows(&nn, &fr_long, &plan).len(), 2, "4 秒的休止不许桥");
    }

    /// ⛔⛔ S158:**这一刀在非升序的计划上会【静默吞掉一整条救援】。**
    ///
    /// 取证(S158,本判据第一版跑出来的实测值):降序喂 `[(3,3,-6),(1,1,-6)]` ⇒ 输出只剩
    /// `[DeadJob { shift: -6, start: 32, end: 55 }]`,`(6,31)` 那条窗**整个消失**,
    /// 而组数日志、`plan.json`、以及上面那一条单测**全绿** —— 因为它们量的都是
    /// 「合并该不该发生」,没有一条量「合并之后还剩几条救援」。
    ///
    /// ⚠ S157 交接把出处记成了 `UTAI_MG_PLAN="685:693:-12,174:184:-12"`,而那条路
    /// **走不通**:`mg_parse_plan_override` 从 S148 起就断言「组必须按下标升序且互不重叠」
    /// (`score2svc_mg.rs`,`prev.end < start`)⇒ 那个计划在解析层就 panic 了。
    /// ⇒ 这个缺陷今天**从 env 探针够不着**,它只对**代码里的下一个 planner** 开火 ——
    /// 这正是它能一直躺着的原因,也是它必须由单测(而不是探针)钉住的原因。
    ///
    /// 机理是两条守卫**同时**在降序上退化成恒真:
    /// * `((pe + 1)..g.start)` 在 `g.start <= pe` 时是**空区间**,而空区间的 `.all()` 是 `true`
    ///   ⇒ 「中间只隔休止」这一条通过了,尽管中间隔着的是**前面一整段歌**;
    /// * `j.start - last.end` 变成**负数** ⇒ 「只桥短休止」那条上限也通过了。
    ///
    /// ⇒ 判据钉的是**不许丢**:合并只允许发生在「按音符序真的紧邻、且窗不重叠」的一对上。
    /// ⚠ 今天的唯一生产者 [`dead_only_plan_with`] 由构造升序不重叠 ⇒ 这条修法**逐位不改**
    /// 今天的输出;它挡的是**下一个 planner**(裁剪/拆句会重排组)。
    #[test]
    fn merging_never_deletes_a_rescue_whatever_order_the_plan_arrives_in() {
        let nn = [0, 85, 0, 85, 0];
        let fr = [10i64, 20, 3, 20, 10];
        let asc = [
            DeadGroup { start: 1, end: 1, shift: -6 },
            DeadGroup { start: 3, end: 3, shift: -6 },
        ];
        // 升序:这两条**本来就该**合并成一条 —— 上面那条测试钉的就是它,这里只取它的窗。
        let merged = dead_group_windows(&nn, &fr, &asc);
        assert_eq!(merged, vec![DeadJob { shift: -6, start: 6, end: 55 }]);

        // ⭐ 同一批组,**降序**喂进来。两段音频还在原处,所以救援也必须还是两条。
        let desc = [
            DeadGroup { start: 3, end: 3, shift: -6 },
            DeadGroup { start: 1, end: 1, shift: -6 },
        ];
        let got = dead_group_windows(&nn, &fr, &desc);
        assert_eq!(
            got.len(),
            2,
            "降序输入把一条救援吞掉了 —— 每条组都必须在输出里留下自己的窗(拿到的是 {got:?})"
        );
        let mut spans: Vec<(i64, i64)> = got.iter().map(|j| (j.start, j.end)).collect();
        spans.sort();
        assert_eq!(
            spans,
            vec![(6, 31), (32, 55)],
            "两条窗必须各自还在原来的位置上(合并只许发生在真正紧邻的一对上)"
        );

        // ⛔ 阴性对照 ①:**位移不同**的降序对同样不许合并 —— 这一条今天就是对的,
        //    它在这里是为了证明上面那条红不是「降序一律不合并」这句话本身造出来的。
        let desc2 = [
            DeadGroup { start: 3, end: 3, shift: -6 },
            DeadGroup { start: 1, end: 1, shift: -7 },
        ];
        assert_eq!(dead_group_windows(&nn, &fr, &desc2).len(), 2);

        // ⭐⭐ 阴性对照 ②(**下一把刀的地基,别拆**):乐句**内部**同位移拆开的两组,
        // 必须被这里重新合并成**与不拆时逐位相同**的那一个窗。
        // 机理:两组之间没有休止 ⇒ 前一组的 `post` 与后一组的 `pre` 从**同一条边界**
        // 向两侧各伸 `GUARD_FRAMES` ⇒ 两个窗**由构造重叠** ⇒ 任何「窗不许重叠」式的
        // 守卫都会在这里把它们劈开,凭空造出一条本来不存在的缝。
        // ⇒ 拆句那条线因此有一条**按构造的阴性对照**:同位移拆分的臂必须与今天逐样本相同。
        let nn_in = [0, 85, 85, 0];
        let fr_in = [10i64, 20, 20, 10];
        let whole = [DeadGroup { start: 1, end: 2, shift: -6 }];
        let split = [
            DeadGroup { start: 1, end: 1, shift: -6 },
            DeadGroup { start: 2, end: 2, shift: -6 },
        ];
        assert_eq!(
            dead_group_windows(&nn_in, &fr_in, &split),
            dead_group_windows(&nn_in, &fr_in, &whole),
            "同位移拆分必须重新合并成与不拆时逐位相同的窗"
        );
        assert_eq!(
            dead_group_windows(&nn_in, &fr_in, &whole),
            vec![DeadJob { shift: -6, start: 6, end: 52 }],
            "⛔ 期望值写字面量:拿被测函数自己算期望值 = 恒真"
        );
    }

    #[test]
    fn dead_group_windows_extend_into_rests_without_overlap() {
        // cum=[0,5,9,19,32,37,45];前间隙 9 帧→pre=4,后间隙 5 帧→post=2(半间隙封顶)。
        let nn = [0, 0, 73, 85, 0, 73];
        let fr = [5i64, 4, 10, 13, 5, 8];
        let plan = [DeadGroup { start: 2, end: 3, shift: -6 }];
        assert_eq!(dead_group_windows(&nn, &fr, &plan), vec![DeadJob { shift: -6, start: 5, end: 34 }]);
    }

    /// S151 护栏:窗边落在**唱音**上时,窗要多伸进那个乘客,好让 10 ms 淡化压在它身上。
    /// ⛔ 有休止可用时一个字不许变 —— 那条期望值就是上面那个测试,原样。
    #[test]
    fn a_window_edge_on_a_sung_note_keeps_its_cross_fade_off_the_rescued_note() {
        // 一整句 [1..=5](无休止),被裁成只救中间那个音 [3..=3]。
        // cum = [0,5,15,25,35,45,55];前邻 note2 = 10 帧 ⇒ pre = min(2, 5) = 2;后邻同理 post = 2。
        let nn = [0, 73, 73, 85, 73, 73, 0];
        let fr = [5i64, 10, 10, 10, 10, 10, 10];
        let plan = [DeadGroup { start: 3, end: 3, shift: -6 }];
        assert_eq!(
            dead_group_windows(&nn, &fr, &plan),
            vec![DeadJob { shift: -6, start: 25 - 2, end: 35 + 2 }],
            "窗必须伸进两侧的乘客,否则淡化压在被救的死音上"
        );
        // 封顶:邻居只有 2 帧 ⇒ 只能伸 1 帧(半个音),窗永远不许吃掉整个邻居。
        let fr = [5i64, 10, 2, 10, 2, 10, 10];
        assert_eq!(
            dead_group_windows(&nn, &fr, &plan),
            vec![DeadJob { shift: -6, start: 17 - 1, end: 27 + 1 }],
            "护栏以半个邻居封顶"
        );
    }

    /// S151:窗边贴着缓冲区边界时**不许淡化** —— 另一侧没有东西可以淡回去,淡了就等于把
    /// 最后 10 ms 还给那条坏渲染。这一格今天每首「最后一句被救」的歌都在发生。
    #[test]
    fn a_window_that_touches_the_end_of_the_take_does_not_fade_back_into_it() {
        let mut base = vec![0.0f32; 48000];
        let donor = vec![1.0f32; 48000];
        // 窗一直到最后一帧(end == total_frames)⇒ 尾部不淡;起点在 0 ⇒ 头部也不淡。
        apply_dead_only_windows(&mut base, 48000, 50, &[DeadJob { shift: -6, start: 0, end: 50 }], false, |_s, _o| Ok(donor.clone()))
            .unwrap();
        assert_eq!(base[0], 1.0, "缓冲区起点:没有「之前」可以淡回去");
        assert_eq!(base[47999], 1.0, "缓冲区终点:被救的最后 10 ms 不许还给 base");
        // 阴性对照:同样的窗放在中间,两头照旧淡化(否则上面两条什么也没验到)。
        let mut base = vec![0.0f32; 48000];
        apply_dead_only_windows(&mut base, 48000, 50, &[DeadJob { shift: -6, start: 10, end: 20 }], false, |_s, _o| Ok(donor.clone()))
            .unwrap();
        let (a, b) = ((10.0 / 50.0 * 48000.0) as usize, (20.0 / 50.0 * 48000.0) as usize);
        assert!(base[a] < 0.01, "窗心之外的边界仍然从 0 起淡");
        assert!((base[a + 240] - 0.5).abs() < 0.01);
        // ⭐ 这两行是**变异逼出来的**:把 `fade_out` 写死成 false,上面全部照绿 ——
        // 仓里**从来没有一条判据说过淡出必须存在**(连 S85 那条 `..._blends_only_the_windows`
        // 都只验了淡入)。「没有淡出」正是拼接器最容易出的那种缺陷。
        assert!(base[b - 1] < 0.01, "窗尾必须淡回 base,否则拼接处是硬切");
        assert!((base[b - 240] - 0.5).abs() < 0.01, "淡出也是 10 ms 余弦半程");
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
    fn the_phase_lock_is_on_by_default_and_an_explicit_zero_still_turns_it_off() {
        // ⛔ The gate for the FLIP itself (S150, user 2026-08-17). Without it, "we turned it on"
        // and "we forgot to turn it on" look identical from every other test in this repo — the
        // library entry points still default to 0.0 on purpose, so nothing else here can see it.
        assert!(
            parse_phase_lock(None) > 0.0,
            "production must run WITH the phase lock now that the blind test passed"
        );
        assert_eq!(parse_phase_lock(None), PHASE_LOCK_DEFAULT);
        // …and the old arm must stay renderable from the same binary, or a user complaint about
        // it cannot be reproduced.
        assert_eq!(parse_phase_lock(Some("0")), 0.0);
        assert_eq!(parse_phase_lock(Some("0.45")), 0.45);
        // Garbage must fall back to the default rather than silently disabling the arm.
        assert_eq!(parse_phase_lock(Some("")), PHASE_LOCK_DEFAULT);
        assert_eq!(parse_phase_lock(Some("nonsense")), PHASE_LOCK_DEFAULT);
        assert_eq!(parse_phase_lock(Some("-1")), PHASE_LOCK_DEFAULT);
        assert_eq!(parse_phase_lock(Some("NaN")), PHASE_LOCK_DEFAULT);
    }

    #[test]
    fn the_infrasonic_removal_is_on_by_default_and_only_an_explicit_no_turns_it_off() {
        // ⛔ 与 `parse_phase_lock` 同一条规矩:**默认值本身要有判据**。这一条在 S155 翻默认
        // 之后尤其重要,因为这个臂现在是**开着**的:没有它,「我们翻了」与「有人把它翻回去了」
        // 在别的每一条测试上长得一模一样(库入口 `Infrasonic::Off` 本来就是合法值)。
        assert!(
            parse_infrasonic_hp(None),
            "生产必须是开着的(S155 翻的)—— 再改它要成对 bump RANGE_ALGO_VERSION 与 audition_cache_tag"
        );
        assert_eq!(parse_infrasonic_hp(None), INFRASONIC_HP_DEFAULT);
        for on in ["1", "true", "on", "yes", " 1 "] {
            assert!(parse_infrasonic_hp(Some(on)), "{on:?} 应当打开");
        }
        // 显式的 0 必须能关掉它 —— 用户报「新版不对」时要渲得出旧臂(S150 那条)。
        for off in ["0", "false", "off", "no", " 0 "] {
            assert!(!parse_infrasonic_hp(Some(off)), "{off:?} 应当关掉");
        }
        // ⛔ 垃圾一律退回默认,**不许静默关掉**。默认关的时候这条是反的(不许静默打开),
        //    翻默认时这个方向必须跟着翻 —— 第一版没跟着翻,是这条判据抓出来的。
        for junk in ["", "nonsense", "2", "-1", "ON!"] {
            assert_eq!(parse_infrasonic_hp(Some(junk)), INFRASONIC_HP_DEFAULT, "{junk:?}");
        }
    }

    /// S155 —— 宽度旋钮:0 = 自适应(出厂),显式的 ms 覆盖它,垃圾退回默认。
    #[test]
    fn the_infrasonic_width_defaults_to_adaptive_and_an_explicit_ms_pins_it() {
        use utai_dsp::psola::Infrasonic;
        assert_eq!(parse_infrasonic_ms(None), 0.0, "出厂必须是自适应");
        assert_eq!(parse_infrasonic_ms(None), INFRASONIC_MS_DEFAULT);
        assert_eq!(parse_infrasonic_ms(Some("8")), 8.0);
        assert_eq!(parse_infrasonic_ms(Some(" 2.5 ")), 2.5);
        for bad in ["", "nonsense", "-1", "NaN", "201"] {
            assert_eq!(parse_infrasonic_ms(Some(bad)), INFRASONIC_MS_DEFAULT, "{bad:?}");
        }
        // ⛔ 三个取值必须真的能被组合出来,否则 `Infrasonic` 那三支里有一支是死代码。
        //    (env 是进程状态,所以这里只走纯函数那一层的组合。)
        let build = |on: bool, ms: f64| -> Infrasonic {
            if !on {
                Infrasonic::Off
            } else if ms > 0.0 {
                Infrasonic::FixedMs(ms)
            } else {
                Infrasonic::PerPeriod
            }
        };
        assert_eq!(build(false, 0.0), Infrasonic::Off);
        assert_eq!(build(true, 0.0), Infrasonic::PerPeriod);
        assert_eq!(build(true, 8.0), Infrasonic::FixedMs(8.0));
        assert_eq!(build(parse_infrasonic_hp(None), parse_infrasonic_ms(None)), Infrasonic::PerPeriod,
                   "出厂口径必须是 PerPeriod");
    }
    #[test]
    fn the_island_dilation_defaults_to_thirty_ms_and_garbage_never_silently_disables_it() {
        // ⛔ 与 `parse_phase_lock` / `parse_infrasonic_hp` 同一条规矩:**默认值本身要有判据**。
        // 这一条尤其重要,因为这个臂是**开着**的:没有它,「我们翻了」与「有人把它翻回去了」
        // 在别的每一条测试上长得一模一样。
        assert_eq!(
            parse_bridge_unvoiced_ms(None),
            30.0,
            "生产默认必须是 30 ms —— 改它要成对 bump RANGE_ALGO_VERSION 与 audition_cache_tag"
        );
        assert_eq!(parse_bridge_unvoiced_ms(None), BRIDGE_UNVOICED_MS_DEFAULT);
        // 显式的 0 必须能关掉它 —— 用户报「新版不对」时要渲得出旧臂(S150 那条)。
        assert_eq!(parse_bridge_unvoiced_ms(Some("0")), 0.0);
        assert_eq!(parse_bridge_unvoiced_ms(Some("60")), 60.0);
        assert_eq!(parse_bridge_unvoiced_ms(Some(" 45 ")), 45.0);
        // 垃圾与越界一律退回默认,不许静默改变行为。
        for bad in ["", "nonsense", "-1", "NaN", "501"] {
            assert_eq!(parse_bridge_unvoiced_ms(Some(bad)), BRIDGE_UNVOICED_MS_DEFAULT, "{bad:?}");
        }
    }

    /// S155 —— **探针的默认口径 = 生产的默认口径**,由这一条绑住。
    ///
    /// ⛔⛔ 为什么需要它:`psola_probe` 以前对每个臂都硬编码 `0.0` 作为回落值。那在「所有生产
    /// 默认都是 0」的时代是对的,而 S150 翻了 `phase_lock`、S154 翻了 `bridge` 之后就**静默地**
    /// 不对了 —— 照旧脚本跑一遍探针,拿到的是**改动之前的臂**,而没有任何一行输出会说破。
    /// 这条线上每一场的结论都建立在探针上,所以那是一个能污染整条线的静默失败。
    /// ⇒ 表在 `utai_dsp::psola::PROBE_ARM_DEFAULTS`,这一条让「改了默认却没改表」变成红。
    #[test]
    fn the_probe_defaults_are_the_production_defaults() {
        let want: [(&str, f64); 10] = [
            ("UTAI_PSOLA_FRAC", f64::from(u8::from(parse_frac_transport(None)))),
            ("UTAI_PSOLA_WSOLA", parse_wsola_frac(None)),
            ("UTAI_PSOLA_LOCK", parse_phase_lock(None)),
            ("UTAI_PSOLA_HP", f64::from(u8::from(parse_infrasonic_hp(None)))),
            ("UTAI_PSOLA_HP_MS", parse_infrasonic_ms(None)),
            ("UTAI_PSOLA_ENVFIX", parse_env_restore_ms(None)),
            ("UTAI_PSOLA_BRIDGE", parse_bridge_unvoiced_ms(None)),
            ("UTAI_PSOLA_WIN", parse_win_periods(None)),
            ("UTAI_PSOLA_XGRAIN", parse_xgrain(None)),
            ("UTAI_PSOLA_LPC", parse_lpc_order(None) as f64),
        ];
        let table = utai_dsp::psola::PROBE_ARM_DEFAULTS;
        assert_eq!(
            table.len(),
            want.len(),
            "加了一个臂旋钮却没加进探针的默认表 —— 探针会对它硬编码地跑另一条臂"
        );
        for (key, production) in want {
            let (_, probe) = table
                .iter()
                .find(|(k, _)| *k == key)
                .unwrap_or_else(|| panic!("{key} 不在 PROBE_ARM_DEFAULTS 里"));
            assert_eq!(
                *probe, production,
                "{key}:探针默认 {probe} ≠ 生产默认 {production} —— \
                 照脚本跑出来的「今天」其实是另一条臂"
            );
        }
    }

    /// S156 —— 读窗旋钮:**默认 1.0 = 教科书宽度**,显式 `0` 仍能渲出旧臂,垃圾与越界退回默认。
    ///
    /// ⛔ 后半句是 S155 血训 #2 的直接落点:**翻一个默认时,所有「垃圾值往哪边倒」的分支都要
    /// 跟着翻一遍。**默认关的时代,垃圾读成 0 = 「不许静默打开一个没被耳判过的臂」是对的;
    /// 默认开之后同一行变成「垃圾静默**关掉**一个已经上线的修法」。这里的写法让垃圾退回
    /// **默认**(= 已上线的那条),而**显式的 0** 仍然能把它关掉 —— 两边都有牙。
    #[test]
    fn the_read_window_defaults_to_the_wide_window_and_only_an_explicit_zero_narrows_it() {
        assert_eq!(
            parse_win_periods(None),
            1.0,
            "生产默认必须是教科书宽度 —— 翻它要成对 bump RANGE_ALGO_VERSION 与 audition_cache_tag"
        );
        assert_eq!(parse_win_periods(None), WIN_PERIODS_DEFAULT);
        assert_eq!(parse_win_periods(Some("0")), 0.0, "显式 0 必须仍能渲出旧臂");
        assert_eq!(parse_win_periods(Some(" 1.5 ")), 1.5);
        for bad in ["", "nonsense", "-1", "NaN", "5"] {
            assert_eq!(
                parse_win_periods(Some(bad)),
                WIN_PERIODS_DEFAULT,
                "{bad:?} 退回的必须是【已上线的默认】,不是 0"
            );
            assert_ne!(parse_win_periods(Some(bad)), 0.0, "{bad:?} 竟然静默关掉了上线的修法");
        }
    }

    /// S156 —— xgrain 旋钮:**默认 1.0**,显式 `0` 仍能渲出最近邻那条臂,垃圾与越界退回默认。
    /// ⛔ 垃圾值的方向与读窗那条同理,见它的 doc。
    #[test]
    fn the_grain_interpolation_is_on_by_default_and_an_explicit_zero_turns_it_off() {
        assert_eq!(
            parse_xgrain(None),
            1.0,
            "生产默认必须是相邻源脉冲插值 —— 翻它要成对 bump RANGE_ALGO_VERSION 与 audition_cache_tag"
        );
        assert_eq!(parse_xgrain(None), XGRAIN_DEFAULT);
        assert_eq!(parse_xgrain(Some("0")), 0.0, "显式 0 必须仍能渲出最近邻那条臂");
        assert_eq!(parse_xgrain(Some(" 0.5 ")), 0.5);
        for bad in ["", "nonsense", "-1", "NaN", "1.5"] {
            assert_eq!(
                parse_xgrain(Some(bad)),
                XGRAIN_DEFAULT,
                "{bad:?} 退回的必须是【已上线的默认】,不是 0"
            );
            assert_ne!(parse_xgrain(Some(bad)), 0.0, "{bad:?} 竟然静默关掉了上线的修法");
        }
    }

    fn inverse_probe_tone(sr: u32, secs: f32) -> Vec<f32> {
        let n = (sr as f32 * secs) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                0.4 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                    + 0.2 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            })
            .collect()
    }

    /// S159 —— ⭐⭐ **窗真的穿过了生产那道门,而且「窗切不到岛」不许被报成一条红。**
    ///
    /// 三件,每件都对应一个会静默出错的地方:
    /// ⑴ **接线**:`apply_inverse_windowed` 给了窄窗,输出必须与整条臂**不同** ——
    ///    否则 `keep` 在某一层被丢掉了,而结果是「音频完全正确、只是慢一倍」,
    ///    上面每一条判据都照绿(S142 那族:第一条判据必须是存在性/接线闸);
    /// ⑵ **窗内逐位**:同一批样本必须与整条臂逐位相同(引擎那侧已经钉过,这里钉的是**这条路**);
    /// ⑶ **归因**:窗落在没有岛的地方 ⇒ 必须 `Ok(原样)` 而**不是**
    ///    `Err(RANGE_INVERSE_NO_PITCH)`。⛔ 那条错误是给「模型没给 f0」留的;两者报成同一条红,
    ///    第二次出现一定被耸肩带过(S129,而这条线上已经出过两次)。
    ///    ⚠ 这条新分支是**真的被触发过一次**的,不是写完就放着(S129 同一条:一条从没被执行过的
    ///    错误分支就是一条空判据)。
    #[test]
    fn the_window_reaches_the_engine_and_an_empty_window_is_not_an_error() {
        let sr = 44_100u32;
        let hop = sr as usize / 100;
        // 两段浊音,中间 0.4 s 真休止(f0 = 0)——「窗切不到任何岛」才有地方落脚。
        let seg = inverse_probe_tone(sr, 0.30);
        let gap = vec![0.0f32; (sr as f64 * 0.40) as usize];
        let mut x = seg.clone();
        x.extend_from_slice(&gap);
        let isl2 = x.len();
        x.extend_from_slice(&seg);
        let mut f0 = vec![0.0f32; x.len() / hop + 2];
        for (i, v) in f0.iter_mut().enumerate() {
            let s = i * hop;
            if s < seg.len().saturating_sub(hop) || (s >= isl2 + hop && s + hop < x.len()) {
                *v = 220.0;
            }
        }
        let fed = Some((f0.as_slice(), hop));
        let full = apply_inverse_with(InverseEngine::Psola, x.clone(), sr, -6, 0.0, fed)
            .expect("整条臂必须成功");
        // ⑴ + ⑵ —— 窗只盖第一段。
        let keep = [(0usize, seg.len())];
        let win = apply_inverse_windowed_with(
            InverseEngine::Psola, x.clone(), sr, -6, 0.0, fed, &keep,
        )
        .expect("窗臂必须成功");
        assert_ne!(win, full, "窄窗与整条臂逐位相同 —— `keep` 在某一层被丢掉了,提速是假的");
        assert_eq!(&win[..seg.len()], &full[..seg.len()], "窗内不是逐位相同");
        // ⑶ —— 窗落在休止正中(离两段浊音都够远),必须 Ok 且**原样返回**。
        let mid = seg.len() + gap.len() / 2;
        let empty = apply_inverse_windowed_with(
            InverseEngine::Psola, x.clone(), sr, -6, 0.0, fed,
            &[(mid, mid + hop)],
        );
        assert_eq!(empty.as_deref(), Ok(x.as_slice()), "窗切不到岛时必须原样返回,而不是报错");
        // ⛔ 阴性对照:同一条路上「真的没有音高」仍然必须响亮失败 —— 否则 ⑶ 只是把那条闸拆了。
        let silent = vec![0.0f32; f0.len()];
        assert_eq!(
            apply_inverse_windowed_with(
                InverseEngine::Psola, x, sr, -6, 0.0, Some((&silent, hop)), &keep,
            )
            .err()
            .as_deref(),
            Some("RANGE_INVERSE_NO_PITCH"),
            "没有 f0 的时候必须还是那条红 —— 两种失败不许合并"
        );
    }

    #[test]
    fn the_inverse_honours_the_exact_length_contract() {
        // Every caller's trim/pad arithmetic depends on len(out) == len(in); shift 0 must be
        // the untouched passthrough (tier 1/2 bit-parity).
        // ⚠ This assertion is weak ON PURPOSE-ADJACENT grounds: apply_inverse resizes to the
        // contract itself, so nothing here could ever fail on length. The load-bearing length
        // gate is `utai_dsp::psola`'s own test, which runs with no such net.
        let sr = 44100u32;
        let x = inverse_probe_tone(sr, 1.0);
        let untouched =
            apply_inverse(x.clone(), sr, 0, DEFAULT_FORMANT_KAPPA, None).expect("shift 0");
        assert_eq!(untouched, x);
        let fed: Vec<f32> = vec![220.0; 101];
        let hop = sr as usize / 100;
        for engine in [InverseEngine::Psola, InverseEngine::Signalsmith] {
            for (shift, kappa) in [(-3i64, 0.0f32), (5, 0.0), (-7, 1.0), (12, 0.5)] {
                let y = apply_inverse_with(
                    engine,
                    x.clone(),
                    sr,
                    shift,
                    kappa,
                    Some((fed.as_slice(), hop)),
                )
                .unwrap_or_else(|e| panic!("{engine:?} shift={shift} kappa={kappa}: {e}"));
                assert_eq!(y.len(), x.len(), "{engine:?} shift={shift} kappa={kappa}");
                assert!(y.iter().all(|v| v.is_finite()));
            }
        }
    }

    #[test]
    fn the_inverse_refuses_loudly_when_it_has_no_pitch_to_work_from() {
        // TD-PSOLA cannot place a single grain without knowing where the periods are. The failure
        // mode this guards is NOT a crash — it is `Ok(un-inverted audio)`: a render at the wrong
        // pitch that every downstream length/finiteness check passes and no cache tag can see.
        // Production always supplies the fed f0 (score 50 fps, cover 100 fps); anything that does
        // not must be told, not quietly served the model's un-shifted take.
        let sr = 44100u32;
        let x = inverse_probe_tone(sr, 0.5);
        let hop = sr as usize / 100;
        for (name, fed) in [
            ("no track at all", None),
            ("empty track", Some(Vec::new())),
            ("all-unvoiced track", Some(vec![0.0f32; 51])),
        ] {
            let arg = fed.as_ref().map(|v| (v.as_slice(), hop));
            let got = apply_inverse_with(InverseEngine::Psola, x.clone(), sr, -6, 0.0, arg);
            assert_eq!(
                got.err().as_deref(),
                Some("RANGE_INVERSE_NO_PITCH"),
                "{name} must fail loudly, got Ok(..) or the wrong CODE"
            );
        }
        // …and a zero hop is the same class of "I cannot locate the periods".
        let fed = vec![220.0f32; 51];
        assert_eq!(
            apply_inverse_with(InverseEngine::Psola, x.clone(), sr, -6, 0.0, Some((&fed, 0)))
                .err()
                .as_deref(),
            Some("RANGE_INVERSE_NO_PITCH")
        );
        // Negative control: the same call WITH pitch must succeed — otherwise the assertions
        // above would pass on a function that always fails.
        assert!(
            apply_inverse_with(InverseEngine::Psola, x, sr, -6, 0.0, Some((&fed, hop))).is_ok(),
            "with a voiced fed f0 the inverse must succeed"
        );
    }

    /// `apply_inverse` is documented as THE single execution point so engine policy can never
    /// drift between the score and cover paths (S85 paid a night for that rule). Nothing in the
    /// tree enforced it — a future call site could reach an engine directly and inherit none of
    /// the loud-failure, κ or diagnostics policy, and every existing test would stay green.
    /// This is a wiring gate, not a behaviour test: it reads the sibling sources.
    /// Fixture shaped like a real record: slot 80 passes the 「あ」 probe at LANDING grade
    /// (err 2, voiced 0.97) while the onset probe says it does not (voiced 0.33) — the exact
    /// pair measured on akiko. 78 passes both. Everything else is healthy filler so the record
    /// has ≥2 tested slots (below that `speaker_range` returns a bounds-only record).
    fn onset_probe_record(with_onset: bool) -> ModelConfig {
        let mut semis = serde_json::Map::new();
        let mut onset = serde_json::Map::new();
        for midi in 60..=80 {
            // 80 mirrors akiko's real sidecar: err 2 / voiced 0.69 ⇒ SINGABLE but NOT landing.
            // That is why the plan lands 85 on 79 today — and 79 is the slot that came back
            // voiced 0.17 on the real render.
            let a_voiced = if midi == 80 { 0.69 } else { 0.97 };
            semis.insert(midi.to_string(), serde_json::json!([2, a_voiced, -3.0, 0.4]));
            // the hard probe agrees everywhere except 79/80 (akiko: 79 → 0.78, 80 → 0.33)
            let voiced = match midi {
                80 => 0.33,
                79 => 0.78,
                _ => 1.0,
            };
            onset.insert(midi.to_string(), serde_json::json!([2, voiced]));
        }
        let mut sp = serde_json::json!({
            "usable": [60, 80], "comfort": [60, 80], "semitones": semis
        });
        if with_onset {
            sp["semitones_onset"] = serde_json::Value::Object(onset);
        }
        config_with(sp)
    }

    /// akiko's real record, the shape that matters: damage 0.000 flat up to 78, 0.592 at 79
    /// (low_ratio 0.629 — the last slot before the comfort cliff), saturated at 80+.
    /// ⭐ THE fixture. akiko's real scan — damage 0.000 flat to 78, 0.592 at 79 (the last slot
    /// before the cliff), saturated at 80+ — with every knob a test might need to move.
    ///
    /// ⛔ One builder on purpose. Five near-identical ones is how the `usable_auto` trap got in:
    /// a fixture that omits it makes `reach == usable`, which turns every S146f split assertion
    /// into a tautology. Here the column is always written, so that cannot happen silently.
    struct Rec {
        usable: (i64, i64),
        comfort: (i64, i64),
        comfort_auto: (i64, i64),
        scan: (i64, i64),
        /// midi → the onset pass's tuple. Absent map ⇒ no `semitones_onset` key at all.
        onset: Option<Vec<(i64, serde_json::Value)>>,
    }

    impl Default for Rec {
        fn default() -> Self {
            Self {
                usable: (36, 80),
                comfort: (36, 79),
                comfort_auto: (36, 79),
                scan: (36, 80),
                onset: None,
            }
        }
    }

    impl Rec {
        fn build(self) -> SpeakerRange {
            let mut semis = serde_json::Map::new();
            for midi in 60..=84 {
                // [err, voiced, rms_db, low_ratio] — the four columns the damage curve integrates.
                semis.insert(
                    midi.to_string(),
                    match midi {
                        80 => serde_json::json!([3, 0.69, -6.7, 0.93]),
                        81 => serde_json::json!([9, 0.22, -13.3, 0.449]),
                        82..=84 => serde_json::json!([9999, 0, -22.0, 0.5]),
                        79 => serde_json::json!([1, 1, -1.3, 0.629]),
                        _ => serde_json::json!([1, 1, -1.0, 0.25]),
                    },
                );
            }
            let mut sp = serde_json::json!({
                "usable": [self.usable.0, self.usable.1],
                "usable_auto": [self.scan.0, self.scan.1],
                "comfort": [self.comfort.0, self.comfort.1],
                "comfort_auto": [self.comfort_auto.0, self.comfort_auto.1],
                "semitones": semis,
            });
            if let Some(rows) = self.onset {
                let mut m = serde_json::Map::new();
                for (midi, v) in rows {
                    m.insert(midi.to_string(), v);
                }
                sp["semitones_onset"] = serde_json::Value::Object(m);
            }
            speaker_range(&config_with(sp), 0).expect("record")
        }
    }

    /// The pre-S146e shape: no `usable_auto` column, so `reach == usable`.
    /// ⚠ Only for tests that are ABOUT that legacy shape — the split is invisible on it.
    fn legacy_rec(usable: (i64, i64), comfort: (i64, i64)) -> SpeakerRange {
        let mut semis = serde_json::Map::new();
        for midi in 60..=84 {
            semis.insert(
                midi.to_string(),
                match midi {
                    80 => serde_json::json!([3, 0.69, -6.7, 0.93]),
                    81 => serde_json::json!([9, 0.22, -13.3, 0.449]),
                    82..=84 => serde_json::json!([9999, 0, -22.0, 0.5]),
                    79 => serde_json::json!([1, 1, -1.3, 0.629]),
                    _ => serde_json::json!([1, 1, -1.0, 0.25]),
                },
            );
        }
        speaker_range(
            &config_with(serde_json::json!({
                "usable": [usable.0, usable.1],
                "comfort": [comfort.0, comfort.1],
                "semitones": semis
            })),
            0,
        )
        .expect("record")
    }

    /// An onset pass that is healthy everywhere except one slot, whose columns are given.
    fn onset_rows(midi: i64, err: f64, voiced: f64, rms_db: f64) -> Vec<(i64, serde_json::Value)> {
        (60..=84)
            .map(|m| {
                (
                    m,
                    if m == midi {
                        serde_json::json!([err, voiced, rms_db, 0.25])
                    } else {
                        serde_json::json!([1, 1, -0.5, 0.25])
                    },
                )
            })
            .collect()
    }

    #[test]
    fn a_deliberately_lowered_comfort_ceiling_pushes_the_landing_down() {
        // ⭐ S146f, the escape hatch the user asked for: 「如果我们的算法误判了 比如它移调到 78
        // 附近效果并不好 用户还是想往下压 那岂不是彻底没办法了」. After the usable split, this
        // is the only control that expresses it — so it has to actually work.
        let nn: Vec<i64> = vec![75, 76, 85, 83, 81, 80];
        let dead: Vec<i64> = vec![85, 83, 81];

        let auto = Rec { usable: (36, 80), comfort: (36, 79), comfort_auto: (36, 79), ..Default::default() }.build();
        let base = minimal_rescue_shift(&dead, &nn, &auto, None).expect("a landing exists");

        let pushed = Rec { usable: (36, 80), comfort: (36, 74), comfort_auto: (36, 79), ..Default::default() }.build();
        let deeper = minimal_rescue_shift(&dead, &nn, &pushed, None).expect("a landing exists");
        assert!(
            deeper < base,
            "dragging the landing ceiling 79→74 must go deeper than {base}, got {deeper}"
        );
        // …and land where the user asked, not merely "somewhere lower".
        for p in &dead {
            assert!(
                (p + deeper) as f32 <= pushed.comfort.1,
                "note {p} landed on {}, above the ceiling the user set",
                p + deeper
            );
        }
    }

    #[test]
    fn a_stored_target_range_is_believed_without_any_intent_heuristic() {
        // ⛔ THE migration trap, measured on the user's own disk: akiko reads comfort [36,74] /
        // comfort_auto [36,79] / usable [36,74] purely because the pre-S146f editor clamped
        // comfort into usable. Reading that as "the user wants landings ≤74" moves their real
        // rescues from −2/−5/−7 to −4/−6/−9/−11 — a version they have already rejected by ear.
        // The clamp that produced the artefact is gone (笔 3: the target range is bounded by the
        // scan, not by `usable`), so a stored 74 now means the user asked for 74 — and it is
        // obeyed. What this pins is that the two records are NOT the same setting: a target of
        // 74 must land deeper than a target of 79, because that is the whole point of the knob.
        let artefact = Rec { usable: (36, 74), comfort: (36, 74), comfort_auto: (36, 79), ..Default::default() }.build();
        let nn: Vec<i64> = vec![75, 76, 85, 83, 81, 80];
        let dead: Vec<i64> = vec![85, 83, 81];
        let untouched = Rec { usable: (36, 74), comfort: (36, 79), comfort_auto: (36, 79), ..Default::default() }.build();
        assert!(
            minimal_rescue_shift(&dead, &nn, &artefact, None).unwrap()
                < minimal_rescue_shift(&dead, &nn, &untouched, None).unwrap(),
            "a stored target is believed as-is — no intent heuristic decides for the user"
        );
    }

    #[test]
    fn an_unreachable_explicit_comfort_still_degrades_instead_of_refusing() {
        // 东雪莲's shape again, now as an explicit edit: no depth within ±MAX_RANGE_SHIFT can put
        // a 75-85 phrase inside [36,52] while keeping every note voiceable. It must fall back to
        // the normal budget, not stop rescuing.
        let nope = Rec { usable: (36, 80), comfort: (36, 52), comfort_auto: (36, 79), ..Default::default() }.build();
        let nn: Vec<i64> = vec![75, 76, 85, 83, 81, 80];
        let dead: Vec<i64> = vec![85, 83, 81];
        assert!(
            minimal_rescue_shift(&dead, &nn, &nope, None).is_some(),
            "an out-of-reach explicit comfort must degrade, never turn the rescue off"
        );
    }

    #[test]
    fn comfort_may_sit_above_the_users_rescue_line() {
        // "rescue everything above 74, but never land higher than 79" is a legal sentence now
        // that the two knobs are orthogonal. Before S146f the read side healed it away.
        let r = Rec { usable: (36, 74), comfort: (36, 79), comfort_auto: (36, 76), ..Default::default() }.build();
        assert_eq!(r.comfort, (36.0, 79.0), "comfort above usable must survive the read side");
        assert_eq!(r.usable, (36.0, 74.0));
    }

    #[test]
    fn the_write_gate_accepts_comfort_above_usable_but_never_outside_the_scan() {
        let ok = serde_json::json!({"speakers": {"0": {
            "usable": [36, 74], "usable_auto": [36, 80], "comfort": [36, 79]
        }}});
        assert!(validate_range_record(&ok).is_ok(), "comfort above the rescue line is legal");
        let outside = serde_json::json!({"speakers": {"0": {
            "usable": [36, 74], "usable_auto": [36, 80], "comfort": [36, 88]
        }}});
        assert!(validate_range_record(&outside).is_err(), "…but not above the scan");
        // A pre-S146e record (no usable_auto) keeps the old, stricter check verbatim.
        let legacy = serde_json::json!({"speakers": {"0": {
            "usable": [36, 74], "comfort": [36, 79]
        }}});
        assert!(validate_range_record(&legacy).is_err(), "no scan column ⇒ old bound");
    }

    #[test]
    fn the_onset_probe_vetoes_a_landing_the_model_can_pitch_but_cannot_voice() {
        // ⛔ THE counterexample, measured off the user's own disk: akiko's 「か」 probe renders at
        // **rms −12.27 dB** relative to that scale's own peak — all but mute — while its f0
        // columns read perfect, and the stored tuple was `[3, 1]` with LANDING left set.
        //
        // This is S81's lesson repeating: that session established the f0 pair CANNOT see a
        // level/timbre collapse, and S146b then built the entire second probe on that pair.
        //
        // ⚠ The slot under test is **78**, not the 80 the reading came from: on akiko's real scan
        // 80 already fails the main pass (voiced 0.69 < 0.9), so LANDING is never set there and
        // the onset veto could not be the thing acting. 78 is where the 「あ」 pass says
        // "comfortable" and only this veto can disagree — i.e. where the test can actually fail.
        let mute = Rec { onset: Some(onset_rows(78, 3.0, 1.0, -12.27)), ..Default::default() }.build();
        assert!(
            !mute.slot_landing_ok(78),
            "a slot the onset probe measured at −12.27 dB must not be a landing"
        );

        // …and the f0 columns alone must NOT be what rejected it — otherwise this test would
        // stay green on the build that throws the level away.
        let f0_only = Rec { onset: Some(onset_rows(78, 3.0, 1.0, -0.5)), ..Default::default() }.build();
        assert!(f0_only.slot_landing_ok(78), "err 3 / voiced 1.00 is a passing f0 reading");

        assert!(mute.slot_landing_ok(77) && mute.slot_landing_ok(76), "the veto is per slot");
        assert!(mute.slot_singable(78), "the onset probe must not change which notes get rescued");
    }

    #[test]
    fn the_onset_level_veto_fires_exactly_at_the_documented_cut() {
        // ⛔ LITERALS, and specifically the ones measured off the user's own record — NOT
        // `RMS_FREE_DB`. An assertion that references the constant under test moves with it and
        // can never catch a drift; that shape has already produced two empty criteria this
        // session. These numbers are akiko's real 「あ」 pass across its usable band, where the
        // only slot that is actually collapsing is 80.
        // The record's own values leave a gap between −3.0 and −6.7, so the literals below can
        // only catch a LOOSENING. Pin the number itself for the other direction — this is a
        // "changing it must be deliberate" guard, not an expected value computed FROM it.
        assert_eq!(
            RMS_FREE_DB, -6.0,
            "calibrated against measured records (akiko's healthy band bottoms out at −3.0, its \
             collapsing slot at −6.7); moving it needs new measurements, not a nudge"
        );
        for (rms, want_landing) in [
            (-0.3, true),  // slot 78
            (-1.3, true),  // slot 79
            (-1.4, true),  // slot 77
            (-3.0, true),  // slot 76 — the quietest healthy slot on this model
            (-6.7, false), // slot 80 — the one that measurably dies
        ] {
            assert_eq!(
                Rec { onset: Some(onset_rows(75, 1.0, 1.0, rms)), ..Default::default() }
                    .build()
                    .slot_landing_ok(75),
                want_landing,
                "onset rms {rms} dB should {} land",
                if want_landing { "" } else { "NOT" }
            );
        }
    }

    #[test]
    fn a_two_column_onset_record_decides_exactly_as_it_did_before() {
        // Every record on disk today is scan_version 3 (f0 columns only). Reading a missing
        // level column as "collapsed" would silently un-land half of everyone's models.
        let mut semis = serde_json::Map::new();
        let mut onset = serde_json::Map::new();
        for m in 60..=84 {
            semis.insert(m.to_string(), serde_json::json!([1, 1, -1.0, 0.25]));
            onset.insert(m.to_string(), serde_json::json!([1, 1]));
        }
        let r = speaker_range(
            &config_with(serde_json::json!({
                "usable": [36, 84], "comfort": [36, 84],
                "semitones": semis, "semitones_onset": onset
            })),
            0,
        )
        .expect("record");
        for m in 60..=84 {
            assert!(r.slot_landing_ok(m), "slot {m} must keep its pre-S146f landing verdict");
        }
    }

    #[test]
    fn match_levels_off_splices_the_donor_exactly_as_rendered() {
        // S147 笔 1:donor 与 base 现在共用一个归一标量 ⇒ 拼接层**不许**再自己调增益。
        // 这条钉的是 `match_levels=false` 那一臂真的什么都不做 —— 它是「共用标量」成立的前提。
        let base = vec![0.5f32; 4410];
        let jobs = [DeadJob { start: 10, end: 40, shift: -2 }];

        let mut off = base.clone();
        apply_dead_only_windows(&mut off, 44100, 100, &jobs, false, |_, _| Ok(vec![0.25f32; 4410]))
            .expect("splice");
        let mut on = base.clone();
        apply_dead_only_windows(&mut on, 44100, 100, &jobs, true, |_, _| Ok(vec![0.25f32; 4410]))
            .expect("splice");

        // 窗心(淡化之外)= 纯 donor。false 臂应当原样是 0.25;true 臂会被 RMS 比值拉回 ~0.5。
        let mid = ((10 + 40) / 2) as f64 / 100.0 * 4410.0;
        let mid = mid as usize;
        assert!(
            (off[mid] - 0.25).abs() < 1e-4,
            "match_levels=false 必须原样拼入,got {}",
            off[mid]
        );
        assert!(
            (on[mid] - 0.5).abs() < 0.05,
            "…而 true 臂必须仍然在调增益,否则这条判据两边都恒真,got {}",
            on[mid]
        );
    }

    #[test]
    fn the_ceiling_knob_adds_rescues_without_deepening_the_ones_already_there() {
        // ⭐⭐ S146f, the regression the user reported by ear: "把音域上限往下调 反而是负效果…
        // 之前报的那几处高音直接就炸了". Measured on their own model and song (akiko, 炉心融解):
        // dropping the ceiling 79→77 kept the group count and the rescued seconds IDENTICAL but
        // moved every landing one semitone deeper (−2/−5/−7 → −3/−6/−8), and 79→74 took it to
        // −6/−9/−11 — dose 251 → 523 semitone·seconds. The cause was one predicate doing two
        // jobs; see `SpeakerRange::reach`.
        //
        // This test is the whole point of the split, so it asserts BOTH halves: the knob must
        // still bite (more dead), and it must stop making existing rescues worse (same shift).
        // Both arms carry the SAME scan (`usable_auto` = [36,80], akiko's real value); the only
        // difference is where the user put the line.
        // ⛔ 目标范围两臂必须**相同**(都是 79):只动 usable 才测得到「天花板不加深落点」。
        // 旧版把 tight 的目标也设成 76,那是旧编辑器夹取的形状 —— 在两个旋钮正交之后,
        // 那样测的是两个旋钮的合力,而 76 那个目标本来就【应该】让落点变深。
        let wide = Rec { usable: (36, 80), comfort: (36, 79), comfort_auto: (36, 79), scan: (36, 80), ..Default::default() }.build();
        let tight = Rec { usable: (36, 76), comfort: (36, 79), comfort_auto: (36, 79), scan: (36, 80), ..Default::default() }.build();
        let nn: Vec<i64> = vec![70, 78, 85];

        let (pw, _) = dead_only_plan_with(&nn, &secs(nn.len()), 0, &wide, RescueTuning::today());
        let (pt, _) = dead_only_plan_with(&nn, &secs(nn.len()), 0, &tight, RescueTuning::today());
        // ⚠ S159zi —— tight 臂现在**被二级拆开**([`SPLIT_MIN_COST_DEFAULT`]):把上限压到 76
        // 之后 78 也成了死音,而它只要 −2、85 要 −9 ⇒ 1 s × 7 st = 14000 ms·st ⇒ 拆。
        // ⛔ 这条判据钉的不变量**一个字没变**(「天花板不许把【已经在救的那个音】的落点带深」),
        // 变的只是它要去哪一组里读那个落点 ⇒ 按**救 85 的那一组**比,而不是按 `[0]` 比。
        // ⛔ 不许改成「比最深的一组」——那样 tight 拆出来的浅组会被自动跳过,判据会退化成恒真。
        assert_eq!(pw.len(), 1, "wide 臂只有 85 是死音 ⇒ 二级无处可拆(读到 {pw:?})");
        let g85 = |p: &[DeadGroup], who: &str| -> DeadGroup {
            *p.iter().find(|d| (d.start..=d.end).contains(&2)).unwrap_or_else(|| panic!("{who} 必须救 85:{p:?}"))
        };
        let (w, t) = (g85(&pw, "wide"), g85(&pt, "tight"));
        assert_eq!(
            t.shift, w.shift,
            "the ceiling moved 80→76 and the landing must NOT follow it down (got {} vs {})",
            t.shift, w.shift
        );
        // ⭐ 而且拆出来的那一组必须**更浅** —— 否则「拆了」等于没拆,这条判据也就白改了。
        assert_eq!(pt.len(), 2, "tight 臂必须被二级拆成两组(读到 {pt:?})");
        let other = pt.iter().find(|d| !(d.start..=d.end).contains(&2)).expect("另一组");
        assert!(
            other.shift.abs() < t.shift.abs(),
            "只需要浅救的那一组必须真的更浅,读到 {} vs {}",
            other.shift, t.shift
        );

        // …and the knob is not merely inert: at the tight ceiling 78 IS dead, at the wide one it
        // is not. Without this half the test would pass on a build that ignored the knob entirely.
        assert!(wide.slot_singable(78) && !tight.slot_singable(78), "the knob must still bite");

        // The landing itself may sit above the user's line — that is exactly the semantic the
        // user chose (2026-08-15): the ceiling says WHICH NOTES to rescue, not what the model
        // may be asked to sing. Pin it, because it is the surprising half.
        // ⚠ S159zi —— `t` 而不是 `pt[0]`:二级拆之后 `pt[0]` 是**浅的那一组**(70/78 走 −2),
        //    照旧读会算出 85−2 = 83,而 83 在 scan(36,80)之外 ⇒ 下面那条当场红。
        //    ⭐ 它红得响,正是因为它盯的是「落点必须在 scan 之内」这件真事。
        let land = 85 + t.shift;
        assert!(
            land as f32 > tight.usable.1,
            "expected the rescue to land above the user's line (got {land})"
        );
        assert!(tight.slot_reachable(land), "…but never outside what the scan says is voiceable");
    }

    #[test]
    fn the_reach_bounds_fall_back_to_usable_when_the_record_has_no_usable_auto() {
        // Every pre-S146e record on disk lacks the column, and for those `usable` IS the scan's
        // answer (nothing could edit it then) ⇒ the split must be a no-op there, not a widening.
        let r = legacy_rec((36, 78), (36, 78));
        assert_eq!(r.reach, r.usable);
        for midi in 36..=96 {
            assert_eq!(
                r.slot_singable(midi),
                r.slot_reachable(midi),
                "slot {midi} must behave identically when the record predates usable_auto"
            );
        }
    }

    #[test]
    fn a_poisoned_usable_auto_can_only_widen_the_reach_never_narrow_it() {
        // Defensive: a hand-edited sidecar could hold a `usable_auto` NARROWER than `usable`.
        // The union in `speaker_range` must keep the drag check at least as permissive as the
        // build without the split — otherwise a bad file silently removes rescues.
        let mut semis = serde_json::Map::new();
        for midi in 60..=84 {
            semis.insert(midi.to_string(), serde_json::json!([1, 1, -1.0, 0.25]));
        }
        let r = speaker_range(
            &config_with(serde_json::json!({
                "usable": [36, 80], "usable_auto": [50, 70], "comfort": [36, 79],
                "semitones": semis
            })),
            0,
        )
        .expect("record");
        assert_eq!(r.reach, (36.0, 80.0), "reach must be the UNION, got {:?}", r.reach);
    }

    #[test]
    fn narrowing_the_usable_ceiling_actually_marks_more_notes_dead() {
        // S146e, the user's report: "调节那个可用范围没用…我想让高音用舒适的办法去唱也做不到".
        // The raw-scan arm of `slot_singable` read `slot_flags` only, so dragging the ceiling
        // changed the stored record, correctly invalidated the render — and produced the same
        // audio. 78 is a slot the scan calls fine (damage 0.000); the knob must still veto it.
        let wide = legacy_rec((36, 80), (36, 79));
        let tight = legacy_rec((36, 76), (36, 76));
        assert!(wide.slot_singable(78), "78 is scan-clean — the fixture must start here");
        assert!(!tight.slot_singable(78), "the ceiling at 76 must veto 78");
        assert!(tight.slot_singable(76), "the ceiling is inclusive");

        // …and the veto has to reach the thing the user hears: the phrase plan.
        let nn: Vec<i64> = vec![70, 78];
        let (plan_wide, _) = dead_only_plan_with(&nn, &secs(nn.len()), 0, &wide, RescueTuning::today());
        let (plan_tight, _) = dead_only_plan_with(&nn, &secs(nn.len()), 0, &tight, RescueTuning::today());
        assert!(plan_wide.is_empty(), "nothing is dead at the wide ceiling");
        assert_eq!(plan_tight.len(), 1, "the tight ceiling must hand this phrase to the rescue");
        assert!(plan_tight[0].shift < 0, "and pull it down, got {}", plan_tight[0].shift);
    }

    #[test]
    fn the_usable_knob_can_only_take_slots_away_never_add_them() {
        // Direction guard: a knob that could ADD singable slots would let the user talk the model
        // into singing something the scan measured as dead — the exact "自证" shape.
        let wide = legacy_rec((36, 96), (36, 96));
        for midi in 30..=100 {
            for &(lo, hi) in &[(36i64, 80i64), (60, 76), (36, 60), (70, 90)] {
                let narrow = legacy_rec((lo, hi), (lo, hi));
                if narrow.slot_singable(midi) {
                    assert!(
                        wide.slot_singable(midi),
                        "usable [{lo},{hi}] made {midi} singable when the open record does not"
                    );
                }
            }
        }
    }

    #[test]
    fn the_comfort_knob_prefers_a_comfortable_landing_inside_the_depth_budget() {
        // Ceiling 78 ⇒ 80 is dead and −1 is not available (79 stops being singable), so the
        // shallowest landing is −2 and the budget covers {−2, −3}. Damage is flat at 0.000 across
        // both, so today's rule breaks the tie by depth and answers −2. Moving comfort to 77 makes
        // −2's landing (78) uncomfortable and −3's (77) comfortable — the pick must follow.
        let nn: Vec<i64> = vec![70, 80];
        let dead: Vec<i64> = vec![80];
        let open = legacy_rec((36, 78), (36, 78));
        let tight = legacy_rec((36, 78), (36, 77));
        assert_eq!(minimal_rescue_shift(&dead, &nn, &open, None), Some(-2), "baseline");
        assert_eq!(
            minimal_rescue_shift(&dead, &nn, &tight, None),
            Some(-3),
            "comfort ending at 77 must pull the landing off 78 and onto 77"
        );
    }

    #[test]
    fn an_unreachable_comfort_band_degrades_instead_of_killing_the_rescue() {
        // 东雪莲's real shape: comfort [36,52] against a phrase at 75-85. A hard AND would take
        // it from "10 groups rescued" to "0 rescued / 10 unsolvable" — the knob as kill switch.
        let r = legacy_rec((36, 80), (36, 52));
        let nn: Vec<i64> = vec![75, 76, 85, 83, 81, 80];
        let dead: Vec<i64> = vec![85, 83, 81];
        assert_eq!(
            minimal_rescue_shift(&dead, &nn, &r, None),
            Some(-7),
            "an out-of-reach comfort band must fall back to today's answer, not refuse"
        );
    }

    #[test]
    fn an_unreachable_target_range_does_not_make_the_rule_dive() {
        // S146f: the target range is now authoritative — asking for a deep landing GETS a deep
        // landing, deliberately (用户拍板:「识别出来的范围就是【还原】的目标,手动调过了就
        // 听手动的」). What survives of the old "never dive" invariant is the degradation path:
        // when no qualifying shift can reach the band at all, the rule must fall back to the
        // normal budget rather than spending depth chasing something it will never reach.
        //
        // ⛔ Here the band [36,66] is out of reach: 70 would fall out of the scanned region long
        // before 80 reaches 66. The answer must stay at the ungoverned −2.
        let nn: Vec<i64> = vec![70, 80];
        let dead: Vec<i64> = vec![80];
        let r = legacy_rec((36, 78), (36, 66));

        let reachable: Vec<i64> = (-24..=-1)
            .filter(|&s| {
                dead.iter().all(|&p| r.slot_landing_preferred(p + s))
                    && nn.iter().all(|&p| r.slot_reachable(p + s))
            })
            .collect();
        assert!(
            reachable.is_empty(),
            "the band must genuinely be unreachable or this test proves nothing (got {reachable:?})"
        );

        let s = minimal_rescue_shift(&dead, &nn, &r, None).expect("a landing exists");
        assert_eq!(s, -2, "an unreachable target must degrade, not dive");
    }

    #[test]
    fn the_rescue_lands_where_the_record_says_the_model_is_fine_not_at_the_gate() {
        // The user's own phrase and model. Before: −6 puts the top dead note on 79, the last slot
        // that passes the binary LANDING gate — and the real render measured voiced 0.17 there.
        let r = Rec::default().build();
        let nn: Vec<i64> = vec![75, 76, 85, 83, 81, 80];
        let dead: Vec<i64> = vec![85, 83, 81];
        let s = minimal_rescue_shift(&dead, &nn, &r, None).expect("a landing exists");
        assert_eq!(s, -7, "must clear the cliff, not stop on it");

        // …and the reason, stated as a measurement rather than a belief: every landing is at the
        // damage floor, while the shift it used to pick was not.
        for p in dead.iter().chain(nn.iter()) {
            assert!(
                r.damage_at((p + s) as f32).unwrap() <= LANDING_DAMAGE_EPS,
                "note {p} lands on {} with damage {:?}",
                p + s,
                r.damage_at((p + s) as f32)
            );
        }
        assert!(
            r.damage_at((85 - 6) as f32).unwrap() > LANDING_DAMAGE_EPS,
            "the old landing (79) must be the thing this test is rejecting"
        );
    }

    #[test]
    fn the_landing_rule_stays_as_shallow_as_the_record_allows() {
        // ⛔ NOT "always deeper": once the damage floor is reached, extra depth buys nothing and
        // costs colouring, so the shallowest floor-level shift wins. Without this the rule would
        // happily walk to −24.
        let r = Rec::default().build();
        let nn: Vec<i64> = vec![75, 76, 85, 83, 81, 80];
        let s = minimal_rescue_shift(&[85, 83, 81], &nn, &r, None).expect("a landing exists");
        assert_eq!(s, -7);
        assert!(
            r.damage_at((85 - 8) as f32).unwrap() <= LANDING_DAMAGE_EPS,
            "−8 is also at the floor — the rule must have rejected it for depth, not quality"
        );
    }

    #[test]
    fn the_landing_rule_will_not_walk_into_the_basement_for_a_better_score() {
        // ⛔ The regression this pins, measured on Sovits4.1东雪莲主模型: its damage curve is poor
        // through the MIDDLE of its range (15 slots at low_ratio > 0.7 scattered over 53-79), so an
        // UNBOUNDED "land where damage is lowest" walks the rescue from −6 down to −24 — and S85
        // recorded a −24 whole-song recolour as a catastrophe the user identified by ear.
        // Fixture: a basement that is genuinely better AND genuinely reachable within ±24.
        // ⚠ The first version of this test was VACUOUS — it kept the dragged notes in the group,
        // and they pinned the worst-damage at the bad-band value for every candidate, so the rule
        // returned the shallowest shift no matter how large the cap was. Mutating the constant to
        // 99 left it green. A single dead note with no passengers is what exercises the bound.
        let mut semis = serde_json::Map::new();
        for midi in 40..=70 {
            let v = match midi {
                40..=49 => serde_json::json!([1, 1, -1.0, 0.20]), // basement: damage 0
                50..=64 => serde_json::json!([1, 1, -1.0, 0.85]), // landable but damaged
                _ => serde_json::json!([9999, 0, -22.0, 0.5]),    // dead
            };
            semis.insert(midi.to_string(), v);
        }
        let r = speaker_range(
            &config_with(serde_json::json!({
                "usable": [40, 64], "comfort": [40, 64], "semitones": semis
            })),
            0,
        )
        .expect("record");
        assert!(
            r.damage_at(49.0).unwrap() + 1.0 < r.damage_at(64.0).unwrap(),
            "fixture must offer a MUCH better landing far below, or this test proves nothing"
        );
        assert!(!r.slot_singable(65) && r.slot_landing_ok(64), "65 dead, 64 landable");
        let s = minimal_rescue_shift(&[65], &[65], &r, None).expect("a landing exists");
        // ⛔ The bound is a LITERAL, deliberately. Writing it as `-1 - LANDING_MAX_EXTRA_DEPTH`
        // makes the assertion move with the constant it is guarding, and the test can then never
        // fail — the second vacuous version of this test did exactly that and stayed green with
        // the constant mutated to 99 (which returns −16 here).
        assert!(
            s >= -3,
            "the rule must stay near the shallowest qualifying shift (−1), got {s} — \
             unbounded it dives to −16 to reach the basement"
        );
    }

    #[test]
    fn a_record_with_no_scan_keeps_the_shallowest_qualifying_shift() {
        // Bounds-only records have no damage curve to rank by; their behaviour must not move.
        let r = SpeakerRange::bounds((36.0, 80.0), (36.0, 79.0));
        let nn: Vec<i64> = vec![75, 76, 85, 83, 81, 80];
        assert_eq!(minimal_rescue_shift(&[85, 83, 81], &nn, &r, None), Some(-6));
    }

    #[test]
    fn the_landing_rule_never_changes_WHICH_notes_get_rescued() {
        // The safety property: ranking happens INSIDE the qualifying set, so the dead set and the
        // set of rescued groups are untouched. Asserted against the same plan the decision layer
        // builds, not against the ranking function alone.
        let r = Rec::default().build();
        let nn: Vec<i64> = vec![75, 76, 85, 83, 81, 80, 0, 70, 71];
        let fr: Vec<i64> = vec![9; nn.len()];
        // ⛔ S158: trim 写死成 `None` —— 这条钉的是**落点规则**,而 `RescueTuning::today()`
        //    从 S158 起自带「只裁尾」,会把乐句尾巴上的乘客切掉、把 span 从 (0,5) 变成 (0,4)。
        //    那不是落点规则改了,是这条判据被换了臂。
        let land = RescueTuning::today().landing;
        // ⚠ S159zi —— 显式关掉二级拆:这条钉的是**落点规则**,而二级会把这一句按深度需求断成
        //    两组(85 要 −7、80/81 只要 −1/−2)⇒ `plan.len() == 1` 不再是「落点规则没改
        //    哪些音被救」的探针。⛔ 隔离,不改期望值。
        let iso = |t: RescueTuning| t.with_split_cost(f32::INFINITY);
        let (plan, unfixable) =
            dead_only_plan_with(&nn, &secs(nn.len()), 0, &r, iso(RescueTuning::new(None, land)));
        assert_eq!(unfixable.len(), 0);
        assert_eq!(plan.len(), 1, "exactly one dead phrase, as before");
        assert_eq!((plan[0].start, plan[0].end), (0, 5), "the same note span as before");
        assert_eq!(plan[0].shift, -7, "…only the depth moved");
        // ⭐ 而这条判据真正的安全性质(「哪些音被救」不许变)在**旋钮开着**时也必须成立:
        //    裁剪只放掉乘客,死音集合与落点一个字不动。(出厂默认今天是关的。)
        let (shipped, unfix_s) = dead_only_plan_with(
            &nn, &secs(nn.len()), 0, &r, iso(trim_arms(Some((f32::INFINITY, 500.0))).1));
        let dead_of = |p: &[DeadGroup]| -> Vec<i64> {
            let mut v: Vec<i64> =
                p.iter().flat_map(|g| g.start..=g.end).filter(|&k| nn[k] > 0 && !r.slot_singable(nn[k]))
                    .map(|k| nn[k]).collect();
            v.sort_unstable();
            v
        };
        assert_eq!(unfix_s, unfixable, "裁剪不许改无解集合");
        assert_eq!(dead_of(&shipped), dead_of(&plan), "裁剪不许改被救的死音");
        assert_eq!(shipped[0].shift, -7, "裁剪不许改落点");
        assert_eq!((shipped[0].start, shipped[0].end), (0, 4), "…它改的只有『谁陪着走』");
        let _ = fr;
    }

    /// S159g —— ⛔⛔⛔ **反向守卫:接点音程小,不是不裁的理由。**
    ///
    /// S159f 真的按「小音程不裁」发过一版,用户实机判负(能听出缝的音 1 → 3-4 个)。
    /// 机理与四条读数写在 `dead_only_plan_with` 里 trim 那一段的注释上。这里钉两件:
    /// ⑴ **一度接点照裁** —— 这是被推翻的那条规矩留下的坑,写成判据免得有人凭「听起来有道理」
    ///    再加一次;
    /// ⑵ **大跳接点也照裁** —— 光有 ⑴ 会被「干脆全都不裁」满足,那会把 S148 r3 盲测背书过的
    ///    收益整个扔掉。两条一起才把「裁剪与接点音程无关」钉死。
    /// ⑶ 头尾两侧各钉一次(只钉一侧的话,另一侧再被加上规矩不会有判据变红)。
    #[test]
    fn a_small_melodic_joint_does_not_block_the_trim() {
        let r = Rec::default().build();
        let secs9 = |n: usize| vec![50i64; n]; // 每音 1 s ⇒ 回收量远超 500 ms 的门限
        // ⚠ S159zi —— 显式关掉二级拆:这条钉的是**裁剪与接点音程无关**,
        //    而它的夹具 `[74, 80, 81, 83]` 正好也是二级那一刀的形状(83 要 −3、81 只要 −1)。
        let arm = RescueTuning::new(Some((500.0, 500.0)), RescueTuning::today().landing)
            .with_split_cost(f32::INFINITY);
        let span = |nn: &Vec<i64>| {
            let p = dead_only_plan_with(nn, &secs9(nn.len()), 0, &r, arm).0;
            assert_eq!(p.len(), 1, "夹具必须只有一条死乐句:{p:?}");
            (p[0].start, p[0].end)
        };
        // ⚠ `Rec::default()` 的 usable 是 (36, 80) ⇒ 死音必须 **≥81**,乘客必须 ≤80。
        // ⑴ 乘客 80、第一个死音 81 ⇒ 接点 **1 度**,照裁。
        assert_eq!(span(&vec![74, 80, 81, 83]), (2, 3), "一度接点不许挡住裁剪");
        // ⑵ 大跳(69→81 = 12 度)⇒ 同样照裁 —— 两条一起才说明「裁剪与音程无关」。
        assert_eq!(span(&vec![74, 69, 81, 83]), (2, 3), "大跳接点照裁");
        // ⑶ 尾侧同样:83→80 是 3 度,照裁。
        let tail = |nn: &Vec<i64>| {
            let p = dead_only_plan_with(nn, &secs9(nn.len()), 0, &r, arm).0;
            (p[0].start, p[0].end)
        };
        assert_eq!(tail(&vec![85, 83, 80, 0, 70]), (0, 1), "尾侧三度不许挡住裁剪");
        assert_eq!(tail(&vec![85, 83, 74, 0, 70]), (0, 1), "尾侧大跳照裁");
    }

    #[test]
    fn the_onset_probe_narrows_the_landing_band_and_leaves_the_dead_set_alone() {
        // ⭐ THE discriminating shape. Anything weaker is satisfiable by a one-line change that
        // has nothing to do with probes: lowering `usable`'s top from 80 to 79 covers 20/20 of
        // the notes measured dead on the user's own render — the same coverage as an elaborate
        // per-probe table. So "slot 80 is now judged dead" proves nothing. What only a
        // second-probe implementation can produce is: **same slot, same record, LANDING flips
        // while SINGABLE does not**.
        let before = speaker_range(&onset_probe_record(false), 0).expect("record");
        let after = speaker_range(&onset_probe_record(true), 0).expect("record");

        assert!(before.slot_landing_ok(79), "the 「あ」 probe calls 79 a good landing");
        assert!(!after.slot_landing_ok(79), "the onset probe must veto 79 as a landing (0.78)");
        assert!(after.slot_landing_ok(78), "78 passes both probes — the veto must be per-slot");
        // 80 was already not landing-grade on 「あ」 (voiced 0.69) — the veto must not disturb
        // slots it agrees with, in either direction.
        assert!(!before.slot_landing_ok(80) && !after.slot_landing_ok(80));

        // …and the dead set is untouched: this change must not rescue one extra note.
        for midi in 60..=80 {
            assert_eq!(
                before.slot_singable(midi),
                after.slot_singable(midi),
                "slot {midi}: the onset probe must not move SLOT_SINGABLE"
            );
        }
    }

    #[test]
    fn the_onset_probe_can_only_narrow_never_widen() {
        // The invariant that makes this change safe to ship: an extra probe may say "that
        // landing is not safe after all", never "…is safe after all". Without it, a bad second
        // scan could widen the landing band and land rescues somewhere nothing has ever tested.
        let mut semis = serde_json::Map::new();
        let mut onset = serde_json::Map::new();
        for midi in 60..=80 {
            // 「あ」 says NOT landing-grade anywhere (voiced 0.6 < 0.9) but still singable
            semis.insert(midi.to_string(), serde_json::json!([2, 0.6]));
            // the onset probe says everything is perfect — it must NOT be believed
            onset.insert(midi.to_string(), serde_json::json!([0, 1.0]));
        }
        let r = speaker_range(
            &config_with(serde_json::json!({
                "usable": [60, 80], "comfort": [60, 80],
                "semitones": semis, "semitones_onset": onset
            })),
            0,
        )
        .expect("record");
        for midi in 60..=80 {
            assert!(r.slot_singable(midi), "slot {midi} stays singable");
            assert!(!r.slot_landing_ok(midi), "slot {midi}: a second probe must never ADD landing");
        }
    }

    #[test]
    fn a_record_without_the_onset_probe_is_byte_identical_to_before() {
        // Every existing sidecar on disk lacks the new key; they must decide exactly as they did.
        let r = speaker_range(&onset_probe_record(false), 0).expect("record");
        let flags = r.slot_flags.expect("scan present");
        for midi in 60..=80 {
            let slot = (midi - DAMAGE_LO_MIDI as i64) as usize;
            let want = if midi == 80 { SLOT_SINGABLE } else { SLOT_SINGABLE | SLOT_LANDING };
            assert_eq!(flags[slot], want, "slot {midi} must keep the pre-S146b verdict");
        }
        // …and a malformed onset map must degrade to that same verdict rather than to nothing.
        let junk = speaker_range(
            &config_with(serde_json::json!({
                "usable": [60, 80], "comfort": [60, 80],
                "semitones": { "60": [2, 0.97], "61": [2, 0.97] },
                "semitones_onset": { "60": "not-an-array", "61": [], "999": [0, 0.0] }
            })),
            0,
        )
        .expect("record");
        assert!(junk.slot_landing_ok(60) && junk.slot_landing_ok(61));
    }

    #[test]
    fn the_narrowed_landing_pushes_the_rescue_deeper() {
        // The behavioural payoff, on the user's own phrase: notes[186..=191] of 炉心融解 are
        // [75, 76, 85, 83, 81, 80]. With the 「あ」-only record the plan lands the dead notes on
        // 79 (shift −6) — and on the real render う@85→79 came back voiced 0.17. With 79/80
        // vetoed as landings the same group must go deeper, and the dead set must not change.
        let nn: Vec<i64> = vec![75, 76, 85, 83, 81, 80];
        let before = speaker_range(&onset_probe_record(false), 0).expect("record");
        let after = speaker_range(&onset_probe_record(true), 0).expect("record");
        let s_before = minimal_rescue_shift(&[85, 83, 81], &nn, &before, None).expect("a landing exists");
        let s_after = minimal_rescue_shift(&[85, 83, 81], &nn, &after, None).expect("a landing exists");
        assert_eq!(s_before, -6, "before: 85 lands on 79, the shallowest landing-grade slot");
        assert!(
            s_after < s_before,
            "the veto must push the rescue deeper, got {s_after} (was {s_before})"
        );
        assert_eq!(s_after, -7, "78 is the deepest slot both probes still call landable");
    }

    #[test]
    fn the_inverse_keeps_exactly_one_execution_point() {
        const CONSUMERS: [(&str, &str); 3] = [
            ("score2svc.rs", include_str!("score2svc.rs")),
            ("sovits.rs", include_str!("sovits.rs")),
            ("rvc.rs", include_str!("rvc.rs")),
        ];
        for (name, src) in CONSUMERS {
            assert!(
                src.contains("vocal_range::apply_inverse"),
                "{name} no longer routes through the single execution point"
            );
            for direct in ["psola_shift", "stretch_interleaved"] {
                assert!(
                    !src.contains(direct),
                    "{name} calls the engine ({direct}) directly, bypassing apply_inverse's \
                     loud-failure / kappa / diagnostics policy"
                );
            }
        }
        // …and this module really is the only place that names an engine.
        //
        // ⛔⛔ S159 —— **这两行以前有一行是恒真的。**`include_str!("vocal_range.rs")` 把断言
        // **自己**读了进来,所以只要断言里写着那个字面量,`contains` 就一定成立。当时写的是
        // `psola_shift_formant`,而全文件里那个名字**只出现在断言自己那一行**(真正的调用是
        // `psola_shift_win`)—— 也就是说:把引擎调用整个删掉,这条判据照绿。
        // ⇒ 期望的符号用 `concat!` 拼出来,源码里就不存在那个字面量,自证的路被堵死。
        // ⚠ 同族提醒:凡是 `include_str!(自己)` 的判据,断言里出现的每一个字面量都要这样处理。
        let me = include_str!("vocal_range.rs");
        for want in [
            // ⚠ S159zj —— 入口从 `psola_shift_win` 换成了 `psola_shift_edge`(多一个 `edge_fill`)。
            // 这条闸盯的是「vocal_range 里到底还有没有那一处引擎调用」,所以它必须跟着改名走;
            // 忘了改它会红,而那正是它该做的事。
            concat!("utai_dsp::psola::psola_shift", "_edge("),
            concat!("utai_stretch::stretch", "_interleaved("),
        ] {
            assert!(me.contains(want), "vocal_range 里找不到引擎调用 {want} —— 执行点被搬走了?");
        }
    }

    /// Apply ONLY the inverse to a wav on disk, through the production entry point.
    ///
    /// Why this exists rather than "render each arm and compare": the SVC render is **not**
    /// bit-reproducible (measured S146 — two identical `mg_render_sovits` runs differ by 1.47
    /// peak, the same order as the engine effect itself). Rendering one arm per engine therefore
    /// hands a listener two different takes and asks them to attribute the difference to the
    /// engine. Feeding both engines the SAME rendered wav removes that confound entirely: the
    /// only difference left in the pair IS the engine.
    ///
    /// ```powershell
    /// $env:UTAI_INV_IN="…\arm_raw.wav"; $env:UTAI_INV_F0="…\f0.f32"; $env:UTAI_INV_HOP="882"
    /// $env:UTAI_INV_SHIFT="-6"; $env:UTAI_INV_KAPPA="0"; $env:UTAI_INV_ENGINE="psola|signalsmith"
    /// $env:UTAI_INV_OUT="…\arm.wav"
    /// cargo test --lib inference::vocal_range::tests::inverse_probe -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "probe: needs a wav + f0 track on disk (set UTAI_INV_*)"]
    fn inverse_probe() {
        let inp = std::env::var("UTAI_INV_IN").expect("UTAI_INV_IN");
        let out = std::env::var("UTAI_INV_OUT").expect("UTAI_INV_OUT");
        let f0p = std::env::var("UTAI_INV_F0").expect("UTAI_INV_F0");
        let hop: usize = std::env::var("UTAI_INV_HOP").expect("UTAI_INV_HOP").parse().unwrap();
        let shift: i64 = std::env::var("UTAI_INV_SHIFT").expect("UTAI_INV_SHIFT").parse().unwrap();
        let kappa: f32 =
            std::env::var("UTAI_INV_KAPPA").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0);
        let engine = match std::env::var("UTAI_INV_ENGINE").as_deref() {
            Ok("signalsmith") => InverseEngine::Signalsmith,
            _ => InverseEngine::Psola,
        };

        let mut rd = hound::WavReader::open(&inp).expect("open");
        let spec = rd.spec();
        let x: Vec<f32> = rd
            .samples::<i32>()
            .map(|s| s.unwrap() as f32 / (1i32 << (spec.bits_per_sample - 1)) as f32)
            .collect();
        let x: Vec<f32> = if spec.channels > 1 {
            x.chunks(spec.channels as usize).map(|c| c.iter().sum::<f32>() / c.len() as f32).collect()
        } else {
            x
        };
        let f0: Vec<f32> = std::fs::read(&f0p)
            .expect("f0")
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let n = x.len();
        let y = apply_inverse_with(engine, x, spec.sample_rate, shift, kappa, Some((&f0, hop)))
            .expect("inverse");
        assert_eq!(y.len(), n, "exact-length contract");
        let peak = y.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-9);
        let g = 0.92 / peak;
        let mut w = hound::WavWriter::create(
            &out,
            hound::WavSpec {
                channels: 1,
                sample_rate: spec.sample_rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .expect("create");
        for v in &y {
            w.write_sample((((v * g).clamp(-1.0, 1.0)) * 32767.0).round() as i16).unwrap();
        }
        w.finalize().unwrap();
        println!("inverse_probe: {engine:?} {shift:+} st kappa {kappa} -> {out}");
    }

    #[test]
    fn the_two_engines_are_both_reachable_and_actually_different() {
        // The A/B arm has to stay alive: if `UTAI_RANGE_ENGINE=signalsmith` silently ran PSOLA,
        // every future comparison would compare a thing with itself and report "no difference" —
        // the exact shape of false negative RANGE_ALGO_VERSION exists to prevent.
        let sr = 44100u32;
        let x = inverse_probe_tone(sr, 0.5);
        let fed = vec![220.0f32; 51];
        let hop = sr as usize / 100;
        let a = apply_inverse_with(InverseEngine::Psola, x.clone(), sr, -6, 0.0, Some((&fed, hop)))
            .expect("psola");
        let b =
            apply_inverse_with(InverseEngine::Signalsmith, x.clone(), sr, -6, 0.0, Some((&fed, hop)))
                .expect("signalsmith");
        assert_eq!(a.len(), b.len());
        let diff = a.iter().zip(b.iter()).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
        assert!(diff > 1e-3, "the two engines produced the same audio (max |Δ| {diff})");
        // and the default really is PSOLA
        assert_eq!(inverse_engine(), InverseEngine::Psola);
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
