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
    fn slot_singable(&self, midi: i64) -> bool {
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
    fn thinness(&self, midi: i64) -> Option<f32> {
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
/// `dead_group_windows` reads) — it exists only for [`trim_freed_ms`], which needs to know how
/// much sung material a cut would actually free.
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
}

impl RescueTuning {
    /// Exactly what ships today.
    pub fn today() -> Self {
        Self { trim: TRIM_DEFAULT, landing: None }
    }

    pub fn new(trim: Option<(f32, f32)>, landing: Option<i64>) -> Self {
        Self { trim, landing }
    }

    pub fn from_env() -> Self {
        Self {
            trim: parse_trim(std::env::var("UTAI_RANGE_TRIM").ok().as_deref()),
            landing: parse_landing(std::env::var("UTAI_RANGE_LANDING").ok().as_deref()),
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
            let Some(whole_shift) = minimal_rescue_shift(&dead, &whole, range, tune.landing) else {
                unfixable.push((i, j));
                i = j + 1;
                continue;
            };
            let (first_dead, last_dead) = (dead_at[0], *dead_at.last().unwrap());
            let (mut a, mut b) = (i, j);
            if let Some((head_ms, tail_ms)) = trim {
                let (freed_head, freed_tail) = (ms(i, first_dead), ms(last_dead + 1, j + 1));
                if freed_head >= head_ms {
                    a = first_dead;
                }
                if freed_tail >= tail_ms {
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
                let s = minimal_rescue_shift(&dead, &kept, range, tune.landing).unwrap_or(whole_shift);
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

/// S151 卸乘客 —— 一刀要**回收多少毫秒**的活音才值得它造出来的那条缝,`(裁头, 裁尾)`。
/// **`None` = 关 = 今天的整句组,逐位不变**;`UTAI_RANGE_TRIM=0` 关 · `=1` 用下面两个常量 ·
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
/// p90 1.452 dB against a 0.060 floor, at a cut **tail** 0.033 / 0.407 — an **8×** asymmetry,
/// because a head cut lands the 10 ms fade-in on a note's ONSET.
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
/// ⛔ Deliberately NOT promoted by these numbers: the instrument cannot tell whether a seam is
/// audible, and this line has been wrong in exactly that way twice (S148 WSOLA, S150 v1/v2).
/// Blind test first, flip after — the S146 protocol.
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
const LANDING_DEFAULT: Option<i64> = None;

/// ⛔ Off until a blind test says otherwise — flipping this changes what every rescued phrase
/// sounds like, so it also needs `RANGE_ALGO_VERSION` ↔ `audition_cache_tag` bumped IN PAIR
/// (S150: missing one of those is not an error, it is the user hearing a stale cache).
const TRIM_DEFAULT: Option<(f32, f32)> = None;
/// ⛔⛔ **头裁已被盲测判负 —— 这个值是「关」,不是一个门限。**
///
/// S151 r1(5 组 × 2 文件,`level_match: none`,两个对照都答对):
/// * `[685..=693]` **尾裁**,放掉 2.56 s ⇒ 用户选**改动版** ✅
/// * `[796..=802]` **头裁**,放掉 4.36 s(全曲最大的一刀)⇒ 用户选**今天** ❌
/// * `[612..=624]` **头裁**,放掉 3.44 s ⇒ 「区别不是很大」⚪
///
/// 三组按**裁哪一侧**干净地分开,而这正好压在 S148 独立量到的缝代价上:裁头的缝 Δripple
/// p50 **0.258** / p90 **1.452** dB,裁尾 **0.033** / **0.407**(同处地板 0.060)—— **8 倍**,
/// 因为 10 ms 淡入压在**起音**上。⚠ 每侧只有 1 个数据点 ⇒ 这是「机理 + 一个同向数据点」,
/// 不是判决;要复活头裁,举证责任是**一次通过的盲测**,不是把这个数调小。
/// (想扫参数仍然可以:`UTAI_RANGE_TRIM=<head_ms>:<tail_ms>`。)
const TRIM_HEAD_MS: f32 = f32::INFINITY;
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
            let room = (LANDING_RATIO_TWO_ST - shallowest.abs()).max(0);
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

    let best = pool.iter().map(|&s| worst(s)).fold(f32::INFINITY, f32::min);
    let tied: Vec<i64> = pool.into_iter().filter(|&s| worst(s) <= best + LANDING_DAMAGE_EPS).collect();

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
    if landing.is_none() {
        // 默认臂:到此为止,与 S151 之前逐字相同。
        return tied.into_iter().min_by_key(|s| s.abs());
    }
    let best_thin = tied.iter().copied().map(thinner).fold(f32::INFINITY, f32::min);
    tied.into_iter()
        .filter(|&s| thinner(s) <= best_thin + LANDING_THIN_EPS)
        .min_by_key(|s| s.abs())
}

/// `ratio = 2` — the shift at which the synthesis stops reading part of every pitch period
/// (see `utai_dsp::psola::PsolaDiagnostics::src_uncovered_frac`: +12 → 0.00 %, +14 → 10.2 %,
/// +16 → 20.0 %, measured on the real mark train). Optional depth is never spent past it.
const LANDING_RATIO_TWO_ST: i64 = 12;

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
        let margin = (MERGE_BRIDGE_FRAMES as f64 * spf) as usize;
        for (ji, j) in jobs.iter().enumerate().filter(|(_, j)| j.shift == s) {
            let a = ((j.start.max(0) as f64 * spf) as usize).min(n);
            let b = ((j.end.max(0) as f64 * spf) as usize).min(n);
            let (lo, hi) = (a.saturating_sub(margin), (b + margin).min(n));
            if hi > lo {
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

/// S146g — carry the sub-sample transport residual instead of dropping it. **Default off.**
///
/// The measurement is unambiguous (whole-sample transport discards a residual whose RMS is
/// exactly 0 at ratio 1.0 and a flat ≈0.41 samples everywhere else — the shape of the fixed toll
/// we could not explain), and carrying it recovers ~80-85% of that toll on the production
/// caliber. What is NOT settled is whether it sounds better: on the registered fixture ΔHNR
/// ranks the praat gold standard BELOW two arms the user already rejected by ear, and the
/// carrying arm reads ABOVE gold — ΔHNR > 0 means "more periodic than the input", which is what
/// WORLD bought by collapsing unvoiced plosives. ⇒ blind test first, flip after (S146 protocol).
pub fn frac_transport() -> bool {
    parse_frac_transport(std::env::var("UTAI_PSOLA_FRAC").ok().as_deref())
}

/// The env parse, as a pure function so it can be asserted without touching process state
/// (and so `the_probe_defaults_are_the_production_defaults` can read the default off it).
fn parse_frac_transport(v: Option<&str>) -> bool {
    matches!(v, Some("1"))
}

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
/// ⛔ **Why it is still off by default.** The rulers cannot promote it — that is the whole lesson
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
/// ## ⛔ 为什么它默认关
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
const XGRAIN_DEFAULT: f64 = 0.0;

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
const WIN_PERIODS_DEFAULT: f64 = 0.0;

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
            let (out, diag) = utai_dsp::psola::psola_shift_env(
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
            );
            if diag.islands == 0 {
                return Err("RANGE_INVERSE_NO_PITCH".into());
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
            tracing::info!(
                "range-extend: inverse {semis:+.0} st, formant kappa {k:.2}, psola {} islands / \
                 {} marks, cola gap {:.2}% (w p01/median/p99 {:.3}/{:.3}/{:.3}, over 1.05 {:.2}%), \
                 src uncovered {:.2}%, infrasonic {:.2}%{}, env dev p50 {:.3} dB{}, \
                 transport residual {:.4}{}",
                diag.islands,
                diag.marks,
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
                    format!(" (envfix {envfix} ms — 拉回到 {:.3} dB)", diag.env_dev_after_db)
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
                }
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
        let nn = [0, 73, 73, 0, 85, 73, 0];
        let (plan, unfix) = dead_only_plan_with(&nn, &secs(nn.len()), 0, &dxl_like(), RescueTuning::today());
        assert!(unfix.is_empty());
        assert_eq!(plan, vec![DeadGroup { start: 4, end: 5, shift: -6 }]);
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

    /// 乐句 = 3 个健康音 · 1 个死音 · 2 个健康音。头 1.5 s、尾 0.8 s 都够本 ⇒ 两边都裁掉。
    #[test]
    fn a_rescue_group_sheds_the_passengers_that_pay_for_their_own_seam() {
        let nn = [0, 73, 73, 73, 85, 73, 73, 0];
        let fr = [10, 25, 25, 25, 40, 20, 20, 10]; // 头 75 帧 = 1.50 s;尾 40 帧 = 0.80 s
        let (whole, _) = dead_only_plan_with(&nn, &fr, 0, &dxl_like(), RescueTuning::today());
        assert_eq!(
            whole,
            vec![DeadGroup { start: 1, end: 6, shift: -6 }],
            "关掉时必须是 S150 之前那条整句臂"
        );
        let (trimmed, unfix) = dead_only_plan_with(&nn, &fr, 0, &dxl_like(), RescueTuning::new(Some((1000.0, 500.0)), None));
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
        let (plan, _) = dead_only_plan_with(&nn, &fr, 0, &dxl_like(), RescueTuning::new(Some((1000.0, 500.0)), None));
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
        let (plan, _) = dead_only_plan_with(&nn, &fr, 0, &dxl_like(), RescueTuning::new(Some((1000.0, 500.0)), None));
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
        let (off, unfix_off) = dead_only_plan_with(&nn, &fr, 0, &r, RescueTuning::today());
        let (on, unfix_on) = dead_only_plan_with(&nn, &fr, 0, &r, RescueTuning::new(Some((1000.0, 500.0)), None));
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
        let (off, unfix_off) = dead_only_plan_with(&nn, &fr, 0, &dxl_like(), RescueTuning::today());
        let (on, unfix_on) = dead_only_plan_with(&nn, &fr, 0, &dxl_like(), RescueTuning::new(Some((1000.0, 500.0)), None));
        assert!(off.is_empty() && on.is_empty(), "两条臂都必须放弃这一句");
        assert_eq!(unfix_off, vec![(1, 2)]);
        assert_eq!(unfix_on, vec![(1, 2)], "卸乘客不许把无解句变成被救句");
        // 阳性对照:同一句,乘客换成一个不挡路的音 ⇒ 两条臂都救,且开着的那条真的裁了。
        let nn = [0, 85, 73, 0];
        let (off, _) = dead_only_plan_with(&nn, &fr, 0, &dxl_like(), RescueTuning::today());
        let (on, _) = dead_only_plan_with(&nn, &fr, 0, &dxl_like(), RescueTuning::new(Some((1000.0, 500.0)), None));
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
        let (today, _) = dead_only_plan_with(&nn, &fr, 0, &r, RescueTuning::today());
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
        let (near_today, _) = dead_only_plan_with(&[0, 79, 0], &fr, 0, &r, RescueTuning::today());
        assert_eq!(near_today[0].shift, -1, "默认臂:damage 打平就取最浅 ⇒ 停在 78");
        let (near_on, _) = dead_only_plan_with(&[0, 79, 0], &fr, 0, &r, RescueTuning::new(None, Some(1)));
        assert_eq!(near_on[0].shift, -2, "开着:同一个预算内也要挑 low_ratio 更低的 77");
        // 阴性对照:一个 low_ratio 平坦的记录上,开与关必须给出**同一个**落点。
        let flat = dxl_like();
        let (a, _) = dead_only_plan_with(&[0, 85, 0], &fr, 0, &flat, RescueTuning::today());
        let (b, _) = dead_only_plan_with(&[0, 85, 0], &fr, 0, &flat, RescueTuning::new(None, Some(3)));
        assert_eq!(a, b, "没有 low_ratio 可排时,这一刀必须什么也不做");
    }

    /// 与卸乘客同一条定义域:它只决定**落在哪**,不许改**哪些音被救**。
    #[test]
    fn the_landing_arm_never_changes_which_notes_get_rescued() {
        let nn = [0, 73, 80, 73, 0, 60, 0, 85, 40, 0];
        let fr = secs(nn.len());
        let r = akiko_like();
        let (a, ua) = dead_only_plan_with(&nn, &fr, 0, &r, RescueTuning::today());
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

    /// ⛔ 可选的深度不许把合成推过 `ratio = 2` —— 那之后每个基音周期都有一段永远不被读到
    /// (笔 3 的 `src_uncovered_frac`:+12 → 0.00%,+14 → 10.2%,+16 → 20.0%)。
    /// ⚠ 但**必须**走那么深才够得着的组照走不误:那是「救不救得了」,不是「落得干不干净」。
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
        let (deep_today, _) = dead_only_plan_with(&[0, 92, 0], &secs(3), 0, &r, RescueTuning::today());
        let (deep_on, _) = dead_only_plan_with(&[0, 92, 0], &secs(3), 0, &r, RescueTuning::new(None, Some(3)));
        assert_eq!(deep_today[0].shift, -13, "这个音本来就要 −13 才够得着");
        assert_eq!(
            deep_on[0].shift, -13,
            "已经越过 ratio=2 的组:救援照做,但**不许为了更干净的落点再往下花**"
        );
    }

    #[test]
    fn the_landing_knob_is_off_by_default_and_parses() {
        assert!(parse_landing(None).is_none(), "默认必须是今天那条臂");
        assert!(parse_landing(Some("")).is_none());
        assert!(parse_landing(Some("0")).is_none(), "显式关得掉 —— 抱怨时要能用同一个二进制渲旧臂");
        assert_eq!(parse_landing(Some("3")), Some(3));
        assert_eq!(parse_landing(Some(" 4 ")), Some(4));
        for junk in ["x", "-1", "999", "1.5", ""] {
            assert!(parse_landing(Some(junk)).is_none(), "垃圾 {junk} 必须回落到默认");
        }
    }

    /// ⛔ 盲测把**头裁**判负(见 `TRIM_HEAD_MS`)⇒ 出厂那一档必须**只裁尾**。
    /// 这条判据钉的是「`=1` 到底给出什么」,不是常量等于常量:它构造一个头尾都够本的乐句,
    /// 断言出厂档**只切了尾巴**。
    #[test]
    fn the_shipped_trim_only_ever_cuts_the_tail() {
        let on = parse_trim(Some("1")).expect("`1` = 出厂那一档");
        let nn = [0, 73, 73, 73, 85, 73, 73, 0];
        let fr = [10, 100, 100, 100, 40, 100, 100, 10]; // 头 6.0 s、尾 4.0 s,都远超任何门限
        let (plan, _) = dead_only_plan_with(&nn, &fr, 0, &dxl_like(), RescueTuning::new(Some(on), None));
        assert_eq!(
            plan,
            vec![DeadGroup { start: 1, end: 4, shift: -6 }],
            "出厂档:尾巴放掉、组头一个音不许动(头裁的缝贵 8 倍,而且盲测输了)"
        );
    }

    /// 旋钮本身:**默认关**,而且解析是纯函数(测试里读进程环境既会被并行污染,
    /// 又会在别人 export 了变量时**静默通过** —— S150 在 `parse_phase_lock` 上付过这笔学费)。
    #[test]
    fn the_passenger_trim_is_off_by_default_and_the_knob_parses() {
        assert!(parse_trim(None).is_none(), "默认必须是今天那条臂(逐位不变)");
        assert!(parse_trim(Some("")).is_none());
        assert!(parse_trim(Some("0")).is_none(), "显式关得掉 —— 用户抱怨时要能用同一个二进制渲旧臂");
        let on = parse_trim(Some("1")).expect("`1` = 用登记的两个常量");
        assert_eq!(on, (TRIM_HEAD_MS, TRIM_TAIL_MS));
        assert!(on.0.is_infinite(), "头裁被盲测判负 ⇒ 出厂档必须够不到它");
        assert!(on.1.is_finite() && on.1 > 0.0, "…而尾裁是那一轮唯一赢了的一刀");
        assert_eq!(parse_trim(Some("800:300")), Some((800.0, 300.0)), "扫参数用");
        assert_eq!(parse_trim(Some(" 800 : 300 ")), Some((800.0, 300.0)));
        for junk in ["x", "800", "800:", "800:x", "-1:0", "1:2:3", "nan:1"] {
            assert!(parse_trim(Some(junk)).is_none(), "垃圾 {junk} 必须回落到默认,不是静默乱开");
        }
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
        let want: [(&str, f64); 9] = [
            ("UTAI_PSOLA_FRAC", f64::from(u8::from(parse_frac_transport(None)))),
            ("UTAI_PSOLA_WSOLA", parse_wsola_frac(None)),
            ("UTAI_PSOLA_LOCK", parse_phase_lock(None)),
            ("UTAI_PSOLA_HP", f64::from(u8::from(parse_infrasonic_hp(None)))),
            ("UTAI_PSOLA_HP_MS", parse_infrasonic_ms(None)),
            ("UTAI_PSOLA_ENVFIX", parse_env_restore_ms(None)),
            ("UTAI_PSOLA_BRIDGE", parse_bridge_unvoiced_ms(None)),
            ("UTAI_PSOLA_WIN", parse_win_periods(None)),
            ("UTAI_PSOLA_XGRAIN", parse_xgrain(None)),
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

    /// S155 —— 读窗旋钮:默认 0 = 今天,显式值覆盖,垃圾与越界退回默认。
    #[test]
    fn the_read_window_defaults_to_todays_and_only_an_explicit_value_widens_it() {
        assert_eq!(
            parse_win_periods(None),
            0.0,
            "生产必须仍然是今天那个窗 —— 翻它要成对 bump RANGE_ALGO_VERSION 与 audition_cache_tag"
        );
        assert_eq!(parse_win_periods(None), WIN_PERIODS_DEFAULT);
        assert_eq!(parse_win_periods(Some("1")), 1.0);
        assert_eq!(parse_win_periods(Some(" 1.5 ")), 1.5);
        for bad in ["", "nonsense", "-1", "NaN", "5"] {
            assert_eq!(parse_win_periods(Some(bad)), WIN_PERIODS_DEFAULT, "{bad:?}");
        }
    }

    /// S156 —— xgrain 旋钮:默认 0 = 今天,显式值覆盖,垃圾与越界退回默认。
    #[test]
    fn the_grain_interpolation_defaults_to_todays_nearest_pulse() {
        assert_eq!(
            parse_xgrain(None),
            0.0,
            "生产必须仍然是最近邻 —— 翻它要成对 bump RANGE_ALGO_VERSION 与 audition_cache_tag"
        );
        assert_eq!(parse_xgrain(None), XGRAIN_DEFAULT);
        assert_eq!(parse_xgrain(Some("1")), 1.0);
        assert_eq!(parse_xgrain(Some(" 0.5 ")), 0.5);
        for bad in ["", "nonsense", "-1", "NaN", "1.5"] {
            assert_eq!(parse_xgrain(Some(bad)), XGRAIN_DEFAULT, "{bad:?}");
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
        let (plan, unfixable) = dead_only_plan_with(&nn, &secs(nn.len()), 0, &r, RescueTuning::today());
        assert_eq!(unfixable.len(), 0);
        assert_eq!(plan.len(), 1, "exactly one dead phrase, as before");
        assert_eq!((plan[0].start, plan[0].end), (0, 5), "the same note span as before");
        assert_eq!(plan[0].shift, -7, "…only the depth moved");
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
        let me = include_str!("vocal_range.rs");
        assert!(me.contains("utai_dsp::psola::psola_shift_formant"));
        assert!(me.contains("utai_stretch::stretch_interleaved"));
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
