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
}

impl RescueTuning {
    /// S159 —— 只换深度上限的那一条臂(计划台专用)。
    pub fn with_cap(self, cap: i64) -> Self {
        Self { cap, ..self }
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
        Self { trim: TRIM_DEFAULT, landing: LANDING_DEFAULT, cap: LANDING_RATIO_TWO_ST }
    }

    pub fn new(trim: Option<(f32, f32)>, landing: Option<i64>) -> Self {
        Self { trim, landing, cap: LANDING_RATIO_TWO_ST }
    }

    pub fn from_env() -> Self {
        Self {
            trim: parse_trim(std::env::var("UTAI_RANGE_TRIM").ok().as_deref()),
            landing: parse_landing(std::env::var("UTAI_RANGE_LANDING").ok().as_deref()),
            // ⛔ 深度上限**没有** env 缝:生产只有 `LANDING_RATIO_TWO_ST` 一条路。
            cap: LANDING_RATIO_TWO_ST,
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
            let (first_dead, last_dead) = (dead_at[0], *dead_at.last().unwrap());
            let (mut a, mut b) = (i, j);
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
                let guard_ok = |k: Option<usize>, s: i64| -> bool {
                    match k.and_then(|k| note_nums.get(k).copied()) {
                        Some(n) if n > 0 => range.slot_reachable(eff(n) + s),
                        _ => true, // 没有邻音 / 那边是休止 ⇒ 护栏伸不进任何唱音
                    }
                };
                let (freed_head, freed_tail) = (ms(i, first_dead), ms(last_dead + 1, j + 1));
                if freed_head >= head_ms && guard_ok(first_dead.checked_sub(1), whole_shift) {
                    a = first_dead;
                }
                if freed_tail >= tail_ms && guard_ok(Some(last_dead + 1), whole_shift) {
                    b = last_dead;
                }
                if (a, b) != (i, j) {
                    tracing::info!(
                        "range: phrase notes[{i}..={j}] rescued as [{a}..={b}] — dropped {:.2}s of \
                         passengers at the head, {:.2}s at the tail",
                        if a > i { freed_head } else { 0.0 } / 1000.0,
                        if b < j { freed_tail } else { 0.0 } / 1000.0,
                    );
                }
            }
            let shift = if (a, b) == (i, j) {
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
            out.push(DeadGroup { start: a, end: b, shift });
        }
        i = j + 1;
    }
    (out, unfixable)
}

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
    let ok = |x: f32| x.is_finite() && x >= 0.0;
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
    let mut out = Vec::new();
    let mut unfixable = Vec::new();
    for &(a, b, _) in &groups {
        let pitches: Vec<i64> = idx
            .iter()
            .zip(midi.iter())
            .filter(|(&i, _)| i >= a && i <= b)
            .map(|(_, &m)| m.round() as i64)
            .collect();
        let dead: Vec<i64> =
            pitches.iter().copied().filter(|&p| !range.slot_singable(p)).collect();
        match minimal_rescue_shift(&dead, &pitches, range, None) {
            Some(s) => out.push(DeadJob { shift: s, start: a as i64, end: (b + 1) as i64 }),
            None => unfixable.push((a as i64, (b + 1) as i64)),
        }
    }
    (out, unfixable)
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

/// ⚙ 出厂默认 = 0.0 —— 0 = 关
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
            let lpc = lpc_order();
            let (out, diag) = utai_dsp::psola::psola_shift_win(
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
    /// ⭐ 顺带写清楚一件容易读反的事:**cover 轨从来就只覆盖死音区间本身**
    /// (按死帧成区 + `GAP_TOL_MS` 桥接),它**根本不拖乘客** —— 也就是说它一直是
    /// 「头尾都裁」的。S158d 这一刀是让**谱面轨往它那边走了半步**(只裁尾)。
    ///
    /// ⛔ 这条判据**不主张「应该接上」** —— 那是另一笔要单独定价的账(cover 的素材、
    /// 判据、以及「一整段音频而不是一个乐句」这个前提都不一样)。它只要求这件事
    /// **是有意的、而且被写下来了**:先用阳性对照证明旋钮在谱面轨上真的咬人,
    /// 再钉住 cover 轨给的是**旋钮之前**那个答案。哪天有人把它接上,这条会红,
    /// 而那时候红的是「你改了一条用户听得见的规则」,不是「测试碍事」。
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
        let f0 = vec![hz(80.0); 200]; // 2 s @ 100 fps,远超 MIN_VIOLATION_MS
        let (jobs, unfix) = cover_dead_plan(&f0, 100.0, &r);
        assert!(unfix.is_empty());
        assert_eq!(jobs.len(), 1, "一整段 80 应当成一个区域");
        assert_eq!(
            jobs[0].shift, -2,
            "cover 轨拿不到 LANDING_DEFAULT —— 若这里变成 -4,说明有人把谱面轨的旋钮接到了              音频轨上,那是用户听得见的改动,必须是有意的并且要单独定价"
        );
        // ⓒ 而「裁剪」在 cover 轨上**结构性地不存在**:它覆盖的就是死音区间本身。
        //    这里用一段「前后都是唱得动的材料、中间一段死音」来钉住这个形状。
        let mut f0b = vec![hz(60.0); 200];
        f0b.extend(vec![hz(80.0); 200]);
        f0b.extend(vec![hz(60.0); 200]);
        let (jobs2, _) = cover_dead_plan(&f0b, 100.0, &r);
        assert_eq!(jobs2.len(), 1);
        assert!(
            jobs2[0].start >= 195 && jobs2[0].end <= 405,
            "cover 轨只覆盖死音那一段(拿到的是 {:?})—— 它没有乘客可卸,             所以 `TRIM_DEFAULT` 对它是空操作",
            (jobs2[0].start, jobs2[0].end)
        );
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
    #[test]
    fn a_passenger_may_not_veto_the_landing_the_dead_note_needs() {
        let phrase = [0, 90, 80, 78, 75, 68, 71, 70, 71, 80, 0];
        let fr = secs(phrase.len());
        let r = pya_like();

        // ⑴ 默认臂 = 生产实测的那一档(用户听到的就是它):落点 78。
        // ⛔ S157c:`today()` 已翻成 `Some(3)` ⇒ 「S151 之前那条臂」必须显式写。
        let (today, _) = dead_only_plan_with(&phrase, &fr, 0, &r, RescueTuning::new(None, None));
        assert_eq!(
            today,
            vec![DeadGroup { start: 1, end: 9, shift: -12 }],
            "默认臂必须逐字是今天:顶音 90 停在落点 78"
        );

        // ⭐ 旋钮开着:同样一批合格落点里,死音自己的 `low_ratio` 说话 ⇒ 顶音落到 76。
        let (on, _) = dead_only_plan_with(&phrase, &fr, 0, &r, RescueTuning::new(None, Some(3)));
        assert_eq!(
            on,
            vec![DeadGroup { start: 1, end: 9, shift: -14 }],
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
        assert_eq!(dead_of(&on), vec![1, 2, 3, 9], "而且就是那四个(90 / 80 / 78 / 80)");

        // ⑶ ⛔⛔ **夹具有效性** —— 少了这一段,上面那个 −14 可能只是「−12 从来没被谁挡过」。
        //    这里直接把被测机理量出来:同一个 `damage_at`,同一批音高。
        let worst = |set: &[i64], s: i64| -> f32 {
            set.iter().map(|&p| r.damage_at((p + s) as f32).unwrap_or(DAMAGE_MAX)).fold(0.0, f32::max)
        };
        let pax = [75i64, 68, 71, 70, 71];
        let dead = [90i64, 80, 78, 80];
        assert_eq!(worst(&pax, -12), 0.0, "−12 上五个乘客必须全都干净");
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
        const TAG: &str = "s158b";
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
            ("JOIN_RESTS_DEFAULT", "pub fn join_rests_enabled("),
            ("FRAC_TRANSPORT_DEFAULT", "pub fn frac_transport("),
            ("PHASE_LOCK_DEFAULT", "pub fn phase_lock("),
            ("INFRASONIC_HP_DEFAULT", "pub fn infrasonic("),
            ("INFRASONIC_MS_DEFAULT", "pub fn infrasonic_fixed_ms("),
            ("XGRAIN_DEFAULT", "pub fn xgrain("),
            ("LPC_ORDER_DEFAULT", "pub fn lpc_order("),
            ("WIN_PERIODS_DEFAULT", "pub fn win_periods("),
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
        for junk in ["x", "800", "800:", "800:x", "-1:0", "1:2:3", "nan:1"] {
            assert_eq!(parse_trim(Some(junk)), parse_trim(Some("1")),
                       "垃圾 {junk} 必须回落到**默认**,不是静默关掉");
        }
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
        assert_eq!(pw.len(), 1);
        assert_eq!(pt.len(), 1);
        assert_eq!(
            pt[0].shift, pw[0].shift,
            "the ceiling moved 80→76 and the landing must NOT follow it down (got {} vs {})",
            pt[0].shift, pw[0].shift
        );

        // …and the knob is not merely inert: at the tight ceiling 78 IS dead, at the wide one it
        // is not. Without this half the test would pass on a build that ignored the knob entirely.
        assert!(wide.slot_singable(78) && !tight.slot_singable(78), "the knob must still bite");

        // The landing itself may sit above the user's line — that is exactly the semantic the
        // user chose (2026-08-15): the ceiling says WHICH NOTES to rescue, not what the model
        // may be asked to sing. Pin it, because it is the surprising half.
        let land = 85 + pt[0].shift;
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
        let (plan, unfixable) =
            dead_only_plan_with(&nn, &secs(nn.len()), 0, &r, RescueTuning::new(None, land));
        assert_eq!(unfixable.len(), 0);
        assert_eq!(plan.len(), 1, "exactly one dead phrase, as before");
        assert_eq!((plan[0].start, plan[0].end), (0, 5), "the same note span as before");
        assert_eq!(plan[0].shift, -7, "…only the depth moved");
        // ⭐ 而这条判据真正的安全性质(「哪些音被救」不许变)在**旋钮开着**时也必须成立:
        //    裁剪只放掉乘客,死音集合与落点一个字不动。(出厂默认今天是关的。)
        let (shipped, unfix_s) = dead_only_plan_with(
            &nn, &secs(nn.len()), 0, &r, trim_arms(Some((f32::INFINITY, 500.0))).1);
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
            concat!("utai_dsp::psola::psola_shift", "_win("),
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
