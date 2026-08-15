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
    /// S146f: the user deliberately moved `comfort` (it is neither `comfort_auto` nor
    /// `comfort_auto` clamped into `usable`). Only then may the landing search spend depth to
    /// honour it — the escape hatch for "the record says 78 is fine and it is not".
    /// ⛔ Gating on this is not politeness: without it, yuyuko's untouched comfort would move
    /// its real rescues from −3/−1 to −7/−5/−4, a four-semitone recolour nobody has heard.
    pub comfort_explicit: bool,
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
        Self { usable, comfort, reach: usable, comfort_explicit: false, damage: None, slot_flags: None }
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
        .find(|c| c.1 - c.0 >= MIN_COMFORT_SPAN && c.0 >= reach.0 && c.1 <= reach.1)?;
    // S146f: did the user actually MOVE comfort, or is this just the auto value (possibly
    // dragged down by the old clamp)? Only a deliberate edit may spend extra depth to be
    // honoured — see `minimal_rescue_shift`.
    //
    // ⛔ The comparison is against comfort_auto **clamped into `usable`**, which is precisely the
    // inverse of the operation that produced the artefact: akiko on disk reads comfort [36,74]
    // / comfort_auto [36,79] / usable [36,74] purely because the editor clamped it, and treating
    // that as intent would move the user's own rescues from −2/−5/−7 to −4/−6/−9/−11.
    // Measured on the eight installed records: with this gate, 8/8 decide exactly as today;
    // without it, akiko AND yuyuko both move (yuyuko −3/−1 → −7/−5/−4, unheard by anyone).
    //
    // ⚠ No `comfort_auto` column ⇒ NOT explicit. Intent is undecidable there (the record predates
    // S60d), and the conservative reading is the one that cannot spend depth nobody asked for.
    let comfort_explicit = match (pair("comfort"), pair("comfort_auto")) {
        (Some(c), Some(auto)) => {
            let dragged = (auto.0.max(usable.0), auto.1.min(usable.1));
            c != auto && c != dragged
        }
        _ => false,
    };
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
    Some(SpeakerRange { usable, comfort, reach, comfort_explicit, damage, slot_flags })
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
        if !(u_lo <= u_hi && c_lo <= c_hi && c_lo >= r_lo && c_hi <= r_hi) {
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
            match minimal_rescue_shift(&dead, &sung, range) {
                Some(shift) => out.push(DeadGroup { start: i, end: j, shift }),
                None => unfixable.push((i, j)),
            }
        }
        i = j + 1;
    }
    (out, unfixable)
}

/// THE single landing search for both tracks (score phrases / cover regions): the minimal |s|
/// that lands every DEAD pitch on a landing-grade slot while every dragged pitch stays
/// singable. Candidate order: single-sided dead searches its own direction by growing |s|;
/// INTERIOR dead (a bridged-weak slot inside usable — a legal record form the write side
/// produces on purpose, rangeTest.ts longestRun) has no inherent direction and tries both,
/// down first at each magnitude. Dead on both sides is untranslatable ⇒ None.
fn minimal_rescue_shift(dead: &[i64], all: &[i64], range: &SpeakerRange) -> Option<i64> {
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
    // S146f, the escape hatch. The user asked: 「如果我们的算法误判了 比如它移调到 78 附近效果
    // 并不好 用户还是想往下压 那岂不是彻底没办法了」— and after the S146f split they were right,
    // nothing expressed that any more. A DELIBERATELY moved comfort is that control: the search
    // may spend whatever depth reaching it costs, instead of only ±LANDING_MAX_EXTRA_DEPTH.
    //
    // Two things keep this from becoming the catastrophe S146c-hotfix removed:
    //   * it needs an explicit edit (`comfort_explicit`) — an untouched record behaves as today;
    //   * it still DEGRADES rather than vetoes: an unreachable band falls through to the normal
    //     budget below, audited, instead of refusing to rescue (东雪莲's [36,52] against a
    //     phrase at 75-85 would otherwise take it from 10 groups rescued to 0).
    // Sweep on the user's own song (akiko, usable 74): comfort 79/78 → −1/−2/−5/−7 (today's
    // answer), 77 → −1/−3/−6/−8, 76 → −1/−2/−4/−7/−9, 74 → −1/−4/−6/−9/−11. Monotone and
    // legible — which is the whole point of putting it on a slider the user can see.
    let anchor: Option<i64> = if range.comfort_explicit {
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
    } else {
        None
    };
    let shallowest = anchor.or_else(|| qualifying.iter().copied().min_by_key(|s| s.abs()))?;
    let mut pool: Vec<i64> = qualifying
        .into_iter()
        .filter(|s| s.abs() <= shallowest.abs() + LANDING_MAX_EXTRA_DEPTH)
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
             (shallowest {shallowest}, +{LANDING_MAX_EXTRA_DEPTH}) — using landing-grade slots {:?}",
            dead,
            range.comfort.0,
            range.comfort.1,
            pool
        );
    } else {
        pool = in_comfort;
    }

    let best = pool.iter().map(|&s| worst(s)).fold(f32::INFINITY, f32::min);
    pool.into_iter()
        .filter(|&s| worst(s) <= best + LANDING_DAMAGE_EPS)
        .min_by_key(|s| s.abs())
}

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
        match minimal_rescue_shift(&dead, &pitches, range) {
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
            DeadJob {
                shift: g.shift,
                start: cum[g.start] - 4.min(gap_prev / 2),
                end: cum[g.end + 1] + 2.min(gap_next / 2),
            }
        })
        .collect()
}

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
pub fn apply_dead_only_windows(
    base: &mut [f32],
    sample_rate: u32,
    total_frames: i64,
    jobs: &[DeadJob],
    match_levels: bool,
    mut donor_render: impl FnMut(i64) -> crate::Result<Vec<f32>>,
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
    for s in shifts {
        let mut donor = donor_render(s)?;
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
        for j in jobs.iter().filter(|j| j.shift == s) {
            let (fa, fb) = (j.start, j.end);
            let a = ((fa.max(0) as f64 * spf) as usize).min(n);
            let b = ((fb.max(0) as f64 * spf) as usize).min(n);
            // 窗短于双淡化 → 收缩淡化宽度;完全空窗才放弃,响亮。
            let xfw = xf.min((b.saturating_sub(a)) / 2);
            if b <= a || xfw == 0 {
                tracing::warn!(
                    "range-extend(dead-only): window frames {fa}..{fb} degenerate after clamp ({a}..{b} samples) — NOT rescued"
                );
                continue;
            }
            for k in a..b {
                let w = if k < a + xfw {
                    0.5 - 0.5 * (std::f32::consts::PI * (k - a) as f32 / xfw as f32).cos()
                } else if k >= b - xfw {
                    0.5 - 0.5 * (std::f32::consts::PI * (b - k) as f32 / xfw as f32).cos()
                } else {
                    1.0
                };
                base[k] = base[k] * (1.0 - w) + donor[k] * w;
            }
        }
    }
    Ok(())
}

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
            let (out, diag) = utai_dsp::psola::psola_shift_formant(
                &audio,
                sample_rate,
                semis,
                f64::from(k) * semis,
                f0,
                hop,
            );
            if diag.islands == 0 {
                return Err("RANGE_INVERSE_NO_PITCH".into());
            }
            tracing::debug!(
                "range-extend: inverse {semis:+.0} st, formant kappa {k:.2}, psola {} islands / \
                 {} marks, cola gap {:.1}% (w median {:.3})",
                diag.islands,
                diag.marks,
                diag.cola_gap_frac * 100.0,
                diag.cola_w_median
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
            tracing::debug!(
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
        apply_dead_only_windows(&mut base, 48000, 2100, &jobs, false, |s| {
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
        apply_dead_only_windows(&mut base, 48000, 50, &[DeadJob { shift: -6, start: 10, end: 20 }], true, |s| {
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

    #[test]
    fn dead_only_splice_matches_donor_level_to_base() {
        // 审查 S85 major 的钉子:base/donor 独立归一 → 拼接窗响度台阶;donor 按全曲
        // active-RMS 对齐 base。base 恒 0.5、donor 恒 0.25 ⇒ g=2 ⇒ 窗心 ≈0.5。
        let mut base = vec![0.5f32; 48000];
        let donor = vec![0.25f32; 48000];
        apply_dead_only_windows(&mut base, 48000, 50, &[DeadJob { shift: -6, start: 10, end: 20 }], true, |_| Ok(donor.clone()))
            .unwrap();
        assert!((base[(0.3 * 48000.0) as usize] - 0.5).abs() < 1e-3, "donor 缩放到 base 电平");
        assert!((base[0] - 0.5).abs() < 1e-6, "窗外不动");
    }

    #[test]
    fn dead_only_splice_rescues_a_single_frame_window() {
        // 1 帧窗收缩淡化宽度仍拿到 donor 内容,绝不静默丢弃(曾静默 continue+审计谎报)。
        let mut base = vec![0.0f32; 48000];
        let donor = vec![1.0f32; 48000];
        apply_dead_only_windows(&mut base, 48000, 50, &[DeadJob { shift: -6, start: 10, end: 11 }], true, |_| Ok(donor.clone()))
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
        apply_dead_only_windows(&mut base, 48000, 50, &[DeadJob { shift: -6, start: 10, end: 20 }], false, |_| Ok(donor.clone()))
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
        assert_eq!(dead_group_windows(&nn, &fr, &plan), vec![DeadJob { shift: -6, start: 5, end: 34 }]);
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
    fn akiko_like() -> SpeakerRange {
        let mut semis = serde_json::Map::new();
        for midi in 60..=84 {
            // [err, voiced, rms_db, low_ratio] — the four columns the damage curve integrates.
            let v = match midi {
                80 => serde_json::json!([3, 0.69, -6.7, 0.93]),
                81 => serde_json::json!([9, 0.22, -13.3, 0.449]),
                82..=84 => serde_json::json!([9999, 0, -22.0, 0.5]),
                79 => serde_json::json!([1, 1, -1.3, 0.629]),
                _ => serde_json::json!([1, 1, -1.0, 0.25]),
            };
            semis.insert(midi.to_string(), v);
        }
        speaker_range(
            &config_with(serde_json::json!({
                "usable": [36, 80], "comfort": [36, 79], "semitones": semis
            })),
            0,
        )
        .expect("record")
    }

    /// `akiko_like()` with the two knobs moved AND the scan's own bounds recorded — i.e. the
    /// shape every record written since S146e has, and the only shape in which the S146f split
    /// is observable at all.
    /// ⛔ A fixture without `usable_auto` makes `reach == usable`, which silently turns every
    /// split assertion into a tautology. That is why this exists separately.
    fn akiko_like_edited(usable: (i64, i64), comfort: (i64, i64), auto: (i64, i64)) -> SpeakerRange {
        let mut semis = serde_json::Map::new();
        for midi in 60..=84 {
            let v = match midi {
                80 => serde_json::json!([3, 0.69, -6.7, 0.93]),
                81 => serde_json::json!([9, 0.22, -13.3, 0.449]),
                82..=84 => serde_json::json!([9999, 0, -22.0, 0.5]),
                79 => serde_json::json!([1, 1, -1.3, 0.629]),
                _ => serde_json::json!([1, 1, -1.0, 0.25]),
            };
            semis.insert(midi.to_string(), v);
        }
        speaker_range(
            &config_with(serde_json::json!({
                "usable": [usable.0, usable.1],
                "usable_auto": [auto.0, auto.1],
                "comfort": [comfort.0, comfort.1],
                "semitones": semis
            })),
            0,
        )
        .expect("record")
    }

    /// `akiko_like()` with the two knobs moved — the only thing the user can actually change.
    /// ⚠ No `usable_auto` ⇒ this is the PRE-S146e shape (reach == usable). Use
    /// `akiko_like_edited` for anything that exercises the split.
    fn akiko_like_with(usable: (i64, i64), comfort: (i64, i64)) -> SpeakerRange {
        let mut semis = serde_json::Map::new();
        for midi in 60..=84 {
            let v = match midi {
                80 => serde_json::json!([3, 0.69, -6.7, 0.93]),
                81 => serde_json::json!([9, 0.22, -13.3, 0.449]),
                82..=84 => serde_json::json!([9999, 0, -22.0, 0.5]),
                79 => serde_json::json!([1, 1, -1.3, 0.629]),
                _ => serde_json::json!([1, 1, -1.0, 0.25]),
            };
            semis.insert(midi.to_string(), v);
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

    /// A record whose onset pass carries the S146f level column.
    fn with_onset_level(midi: i64, err: f64, voiced: f64, rms_db: f64) -> SpeakerRange {
        let mut semis = serde_json::Map::new();
        let mut onset = serde_json::Map::new();
        for m in 60..=84 {
            semis.insert(m.to_string(), serde_json::json!([1, 1, -1.0, 0.25]));
            onset.insert(m.to_string(), serde_json::json!([1, 1, -0.5, 0.25]));
        }
        onset.insert(midi.to_string(), serde_json::json!([err, voiced, rms_db, 0.25]));
        speaker_range(
            &config_with(serde_json::json!({
                "usable": [36, 84], "comfort": [36, 84],
                "semitones": semis, "semitones_onset": onset
            })),
            0,
        )
        .expect("record")
    }

    /// akiko's record with an explicitly chosen comfort band (comfort ≠ comfort_auto and ≠
    /// comfort_auto clamped into usable) — i.e. the user actually dragged the landing ceiling.
    fn akiko_comfort_target(usable: (i64, i64), comfort: (i64, i64), comfort_auto: (i64, i64)) -> SpeakerRange {
        let mut semis = serde_json::Map::new();
        for midi in 60..=84 {
            let v = match midi {
                80 => serde_json::json!([3, 0.69, -6.7, 0.93]),
                81 => serde_json::json!([9, 0.22, -13.3, 0.449]),
                82..=84 => serde_json::json!([9999, 0, -22.0, 0.5]),
                79 => serde_json::json!([1, 1, -1.3, 0.629]),
                _ => serde_json::json!([1, 1, -1.0, 0.25]),
            };
            semis.insert(midi.to_string(), v);
        }
        speaker_range(
            &config_with(serde_json::json!({
                "usable": [usable.0, usable.1],
                "usable_auto": [36, 80],
                "comfort": [comfort.0, comfort.1],
                "comfort_auto": [comfort_auto.0, comfort_auto.1],
                "semitones": semis
            })),
            0,
        )
        .expect("record")
    }

    #[test]
    fn a_deliberately_lowered_comfort_ceiling_pushes_the_landing_down() {
        // ⭐ S146f, the escape hatch the user asked for: 「如果我们的算法误判了 比如它移调到 78
        // 附近效果并不好 用户还是想往下压 那岂不是彻底没办法了」. After the usable split, this
        // is the only control that expresses it — so it has to actually work.
        let nn: Vec<i64> = vec![75, 76, 85, 83, 81, 80];
        let dead: Vec<i64> = vec![85, 83, 81];

        let auto = akiko_comfort_target((36, 80), (36, 79), (36, 79));
        let base = minimal_rescue_shift(&dead, &nn, &auto).expect("a landing exists");
        assert!(!auto.comfort_explicit, "an untouched comfort must not read as intent");

        let pushed = akiko_comfort_target((36, 80), (36, 74), (36, 79));
        assert!(pushed.comfort_explicit, "a dragged comfort IS intent");
        let deeper = minimal_rescue_shift(&dead, &nn, &pushed).expect("a landing exists");
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
    fn a_comfort_that_only_the_old_clamp_moved_is_not_treated_as_intent() {
        // ⛔ THE migration trap, measured on the user's own disk: akiko reads comfort [36,74] /
        // comfort_auto [36,79] / usable [36,74] purely because the pre-S146f editor clamped
        // comfort into usable. Reading that as "the user wants landings ≤74" moves their real
        // rescues from −2/−5/−7 to −4/−6/−9/−11 — a version they have already rejected by ear.
        let artefact = akiko_comfort_target((36, 74), (36, 74), (36, 79));
        assert!(
            !artefact.comfort_explicit,
            "comfort == comfort_auto clamped into usable is the clamp's doing, not the user's"
        );
        let nn: Vec<i64> = vec![75, 76, 85, 83, 81, 80];
        let dead: Vec<i64> = vec![85, 83, 81];
        let untouched = akiko_comfort_target((36, 74), (36, 79), (36, 79));
        assert_eq!(
            minimal_rescue_shift(&dead, &nn, &artefact),
            minimal_rescue_shift(&dead, &nn, &untouched),
            "the clamp artefact must decide exactly as an untouched record does"
        );
    }

    #[test]
    fn an_unreachable_explicit_comfort_still_degrades_instead_of_refusing() {
        // 东雪莲's shape again, now as an explicit edit: no depth within ±MAX_RANGE_SHIFT can put
        // a 75-85 phrase inside [36,52] while keeping every note voiceable. It must fall back to
        // the normal budget, not stop rescuing.
        let nope = akiko_comfort_target((36, 80), (36, 52), (36, 79));
        assert!(nope.comfort_explicit);
        let nn: Vec<i64> = vec![75, 76, 85, 83, 81, 80];
        let dead: Vec<i64> = vec![85, 83, 81];
        assert!(
            minimal_rescue_shift(&dead, &nn, &nope).is_some(),
            "an out-of-reach explicit comfort must degrade, never turn the rescue off"
        );
    }

    #[test]
    fn comfort_may_sit_above_the_users_rescue_line() {
        // "rescue everything above 74, but never land higher than 79" is a legal sentence now
        // that the two knobs are orthogonal. Before S146f the read side healed it away.
        let r = akiko_comfort_target((36, 74), (36, 79), (36, 76));
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
        // ⛔ THE counterexample, measured off the user's own disk: akiko's 「か」 probe at MIDI 80
        // renders at rms −12.27 dB relative to that scale's own peak — all but mute — while its
        // f0 columns read perfect. The stored tuple was `[3, 1]` and LANDING stayed set.
        //
        // This is S81's lesson repeating: that session established the f0 pair cannot see a
        // level/timbre collapse, and S146b then built the entire second probe on that pair.
        let mute = with_onset_level(80, 3.0, 1.0, -12.27);
        assert!(
            !mute.slot_landing_ok(80),
            "a slot the onset probe measured at −12.27 dB must not be a landing"
        );
        // …and the f0 columns alone must NOT have been what rejected it — otherwise this test
        // would pass on the build that throws the level away.
        let f0_only = with_onset_level(80, 3.0, 1.0, -0.5);
        assert!(f0_only.slot_landing_ok(80), "err 3 / voiced 1.00 is a passing f0 reading");
        // Neighbours are untouched: the veto is per slot, not a band.
        assert!(mute.slot_landing_ok(79) && mute.slot_landing_ok(81));
        // And it never touches the DEAD set — the onset pass may only ever narrow landings.
        assert!(mute.slot_singable(80), "the onset probe must not change which notes get rescued");
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
                with_onset_level(75, 1.0, 1.0, rms).slot_landing_ok(75),
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
        let wide = akiko_like_edited((36, 80), (36, 79), (36, 80));
        let tight = akiko_like_edited((36, 76), (36, 76), (36, 80));
        let nn: Vec<i64> = vec![70, 78, 85];

        let (pw, _) = dead_only_plan(&nn, 0, &wide);
        let (pt, _) = dead_only_plan(&nn, 0, &tight);
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
        let r = akiko_like_with((36, 78), (36, 78));
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
        let wide = akiko_like_with((36, 80), (36, 79));
        let tight = akiko_like_with((36, 76), (36, 76));
        assert!(wide.slot_singable(78), "78 is scan-clean — the fixture must start here");
        assert!(!tight.slot_singable(78), "the ceiling at 76 must veto 78");
        assert!(tight.slot_singable(76), "the ceiling is inclusive");

        // …and the veto has to reach the thing the user hears: the phrase plan.
        let nn: Vec<i64> = vec![70, 78];
        let (plan_wide, _) = dead_only_plan(&nn, 0, &wide);
        let (plan_tight, _) = dead_only_plan(&nn, 0, &tight);
        assert!(plan_wide.is_empty(), "nothing is dead at the wide ceiling");
        assert_eq!(plan_tight.len(), 1, "the tight ceiling must hand this phrase to the rescue");
        assert!(plan_tight[0].shift < 0, "and pull it down, got {}", plan_tight[0].shift);
    }

    #[test]
    fn the_usable_knob_can_only_take_slots_away_never_add_them() {
        // Direction guard: a knob that could ADD singable slots would let the user talk the model
        // into singing something the scan measured as dead — the exact "自证" shape.
        let wide = akiko_like_with((36, 96), (36, 96));
        for midi in 30..=100 {
            for &(lo, hi) in &[(36i64, 80i64), (60, 76), (36, 60), (70, 90)] {
                let narrow = akiko_like_with((lo, hi), (lo, hi.min(hi)));
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
        let open = akiko_like_with((36, 78), (36, 78));
        let tight = akiko_like_with((36, 78), (36, 77));
        assert_eq!(minimal_rescue_shift(&dead, &nn, &open), Some(-2), "baseline");
        assert_eq!(
            minimal_rescue_shift(&dead, &nn, &tight),
            Some(-3),
            "comfort ending at 77 must pull the landing off 78 and onto 77"
        );
    }

    #[test]
    fn an_unreachable_comfort_band_degrades_instead_of_killing_the_rescue() {
        // 东雪莲's real shape: comfort [36,52] against a phrase at 75-85. A hard AND would take
        // it from "10 groups rescued" to "0 rescued / 10 unsolvable" — the knob as kill switch.
        let r = akiko_like_with((36, 80), (36, 52));
        let nn: Vec<i64> = vec![75, 76, 85, 83, 81, 80];
        let dead: Vec<i64> = vec![85, 83, 81];
        assert_eq!(
            minimal_rescue_shift(&dead, &nn, &r),
            Some(-7),
            "an out-of-reach comfort band must fall back to today's answer, not refuse"
        );
    }

    #[test]
    fn the_comfort_preference_never_spends_more_depth_than_the_budget_allows() {
        // ⛔ The regression this pins is one I measured and rejected: restricting the qualifying
        // set to comfort BEFORE taking the shallowest re-anchors the depth cap, and on yuyuko
        // (usable [36,82], comfort [37,79]) that walks the real rescues from −3 to −7.
        // Here: comfort tops out at 72 while the budget covers only {−2, −3}. A comfortable
        // landing IS reachable (−8, −9, −10 all qualify and all land 80 inside comfort) — the
        // rule must decline to spend that depth and stay at −2.
        let nn: Vec<i64> = vec![70, 80];
        let dead: Vec<i64> = vec![80];
        let r = akiko_like_with((36, 78), (36, 72));

        // ⚠ S146f: this fixture carries no `comfort_auto`, so the record reads as NOT explicit —
        // which is the path this test is about (the automatic preference, bounded by the depth
        // budget). The deliberate-edit path is `a_deliberately_lowered_comfort_ceiling_...`.
        // ⛔ The positive control the first version of this test lacked, which made it vacuous:
        // with comfort at 66 no QUALIFYING shift could reach comfort at all (70 would fall out of
        // the scanned band), so the rejected implementation and this one agreed and the mutation
        // came back green. Assert the temptation exists before asserting it was resisted.
        let reachable: Vec<i64> = (-24..=-1)
            .filter(|&s| {
                dead.iter().all(|&p| r.slot_landing_preferred(p + s))
                    && nn.iter().all(|&p| r.slot_singable(p + s))
            })
            .collect();
        assert!(
            !reachable.is_empty(),
            "no comfortable landing is reachable ⇒ this test cannot distinguish the two rules"
        );

        let s = minimal_rescue_shift(&dead, &nn, &r).expect("a landing exists");
        assert_eq!(s, -2, "must not dive to reach comfort, got {s} (reachable: {reachable:?})");
        assert!(
            !r.slot_landing_preferred(80 + s),
            "the test is only meaningful if the chosen landing is genuinely outside comfort"
        );
    }

    #[test]
    fn the_rescue_lands_where_the_record_says_the_model_is_fine_not_at_the_gate() {
        // The user's own phrase and model. Before: −6 puts the top dead note on 79, the last slot
        // that passes the binary LANDING gate — and the real render measured voiced 0.17 there.
        let r = akiko_like();
        let nn: Vec<i64> = vec![75, 76, 85, 83, 81, 80];
        let dead: Vec<i64> = vec![85, 83, 81];
        let s = minimal_rescue_shift(&dead, &nn, &r).expect("a landing exists");
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
        let r = akiko_like();
        let nn: Vec<i64> = vec![75, 76, 85, 83, 81, 80];
        let s = minimal_rescue_shift(&[85, 83, 81], &nn, &r).expect("a landing exists");
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
        let s = minimal_rescue_shift(&[65], &[65], &r).expect("a landing exists");
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
        assert_eq!(minimal_rescue_shift(&[85, 83, 81], &nn, &r), Some(-6));
    }

    #[test]
    fn the_landing_rule_never_changes_WHICH_notes_get_rescued() {
        // The safety property: ranking happens INSIDE the qualifying set, so the dead set and the
        // set of rescued groups are untouched. Asserted against the same plan the decision layer
        // builds, not against the ranking function alone.
        let r = akiko_like();
        let nn: Vec<i64> = vec![75, 76, 85, 83, 81, 80, 0, 70, 71];
        let fr: Vec<i64> = vec![9; nn.len()];
        let (plan, unfixable) = dead_only_plan(&nn, 0, &r);
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
        let s_before = minimal_rescue_shift(&[85, 83, 81], &nn, &before).expect("a landing exists");
        let s_after = minimal_rescue_shift(&[85, 83, 81], &nn, &after).expect("a landing exists");
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
