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
/// How long a DEAD region must SUSTAIN to count as musical content, in ms on the 100 fps
/// grid (counted in VOICED DEAD FRAMES, not span — see the grouping code).
///
/// Was 250 ms. The reason was phantom islands: rmvpe reads breaths and sibilance an octave
/// UP for a few frames at a time, and in S62b (lengv2.3) a handful of octave-doubled spikes
/// once dragged whole-song shifts. That reason has since been undercut twice — S159k pinned
/// region edges to unvoiced frames, and S165 put a Viterbi octave repair in front of the
/// planner — so the door it was holding shut is now mostly bricked up anyway.
///
/// Meanwhile it was holding out real notes. Seven segments the user reported as ruined all
/// sang at 724-1178 Hz against a model whose usable top is 698 Hz; five of them were never
/// rescued at all, because at 65 % voiced a 400 ms passage only musters ~15 dead frames.
/// They are not phantoms: their waveform autocorrelation peaks at 0.75-0.94, where genuinely
/// unvoiced frames sit at p90 = 0.396 (n=122).
///
/// Same-tier A/B, three arms, whole song (bad-frame rate on frames the SOURCE sings cleanly):
///     250 ms  5.85 %   68 regions      100 ms  4.64 %   105 regions
///     150 ms  5.24 %   91 regions      noise floor between identical configs: 0.03 pp
/// Both families fell together (spiky 1.57 → 1.36 %, periodicity-collapse 5.17 → 4.04 %),
/// bad segments went 27 → 15, and the seven reported spots went e.g. 100 → 0 %, 100 → 21 %,
/// 78 → 22 %. Coverage tracked the fix exactly: every spot that gained coverage improved,
/// every spot whose coverage did not move did not move either — seven for seven.
///
/// Cost checked and found small: the extra 37 regions add ~13 s to a 292 s song (188 → 201 s),
/// and the edges that land mid-note (the ones that can leave a timbre step) stayed at 18-20 %
/// of all edges throughout — S159k's edge-snapping holds for the new regions too.
///
/// ⚠ 60 ms was measured as well and is not worth it: +2 regions, +0.2 pp coverage, 4 more edges.
pub const MIN_VIOLATION_MS: f32 = 100.0;
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

/// S160h —— **死/活那条线的电平闸**:扫描说这一格比该模型自己最响的一格低这么多以上,
/// 就不算「唱得动」,即使音准完美。
///
/// ## ⛔ 它治的是什么(用户 2026-08-24 在 yuyuko 的 MIDI 轨 +7 上点名的三处「哑音 / 失声」)
/// 4:18.5-4:20.5 · 4:21.44 · 4:46.75 —— 这三处的音是 **MIDI 78/80/82**,全部落在
/// yuyuko 的 `usable [36,82]` 之内 ⇒ **一个都没被救**。而扫描自己写着:
/// | MIDI | err | voiced | **rms_db** | low_ratio |
/// |---|---|---|---|---|
/// | 72-74 | 1-2 | 1 | **−2.9** | 0.06-0.08 |
/// | 80 | 2 | 1 | **−9.9** | 0.73 |
/// | 82 | 1 | 1 | **−16.6** | 0.78 |
/// ⇒ **音准完美、voiced 恒为 1,但电平低 13.7 dB。**而 `usable` 的契约(本文件头)是
/// 「f0 err <100¢ & voiced >50%」—— **只看音准和有声率** ⇒ 「唱得准但又哑又糊」被判成唱得动。
///
/// ## ⭐ 为什么可以用绝对门槛(而 `low_ratio` 不行)
/// **`rms_db` 在扫描里已经是按模型自己归一化的** —— 四份装机记录的 `rms_db` 最大值**都恰好是 0.0**
/// (东雪莲 / akiko / yachiyo / yuyuko)。⇒ 它量的是「比这个模型自己最响的那一格低多少」。
/// ⛔ 反例:`low_ratio` 是绝对的,四个模型 usable 内的中位是 0.418 / 0.244 / 0.075 / 0.101
/// (差 5.6 倍)—— 拿它当闸会把**东雪莲**的顶从 79 砍到更低,而那正是谱面轨验收所用的模型。见 [`thin_ref`]。
///
/// ## 门槛怎么定(四份装机记录,可从 sidecar 直接复算)
/// | 门槛 | 东雪莲 79 | akiko 76 | yachiyo 77 | **yuyuko 82** |
/// |---|---|---|---|---|
/// | −4 / −6 | **78** ⚠ 退化 | 76 = | 77 = | 79 |
/// | **−7 … −9** | **79 =** | **76 =** | **77 =** | **→ 79** |
/// | −10 及以下 | 79 = | 76 = | 77 = | 80(不够) |
/// ⇒ **−7…−9 是唯一「三个已验收模型一格不动、而 yuyuko 被修好」的窗口**,取中间的 **−8**
/// (对东雪莲顶那一格的 −6.1 留 1.9 dB 余量)。⭐ yuyuko 修完的 79 **正好是它自己 `comfort` 的顶**。
/// ⛔ 只在扫描记录带 4 元组(有 `rms_db`)时生效;老记录一个字不动。
const RESCUE_LEVEL_FLOOR_DB: f64 = -8.0;

/// S160h —— `thin` 的门槛**相对该模型自己的基线**:`该模型 usable 内 low_ratio 的中位 + 这个余量`。
///
/// ## ⛔ 为什么必须相对(用户 2026-08-24 提的)
/// 今天是绝对的 `(low_ratio − 0.55)/0.40`。四份装机记录 usable 内的 `low_ratio` 中位:
/// **东雪莲 0.418 · akiko 0.244 · yachiyo 0.075 · yuyuko 0.101** —— 差 **5.6 倍**。
/// ⇒ 同一条绝对线的后果是**系统性的**:`thin > 0` 的格占比 **东雪莲 39% vs 另外三个 7-9%**
/// —— 它在**惩罚一个天生低频重的嗓子**,同时**放过 yuyuko 顶上那一格**(0.779 只算 0.57 的伤)。
/// ✅ 改成相对之后四个模型落到 **14-34%** 的同一量级,而 yuyuko 顶那一格的 `thin` **0.57 → 1.00**。
///
/// ## 余量为什么是 0.25
/// 它要大到「一个嗓子自己正常的低频量」不算伤,小到「明显比自己平时糊」算伤。
/// 0.25 ≈ 四个模型 usable 内 low_ratio 的 (p90 − p50) 的下半段(0.29 / 0.28 / 0.29 / 0.34)。
/// ⛔ 这是**唯一**一个还没有独立耳判背书的新常量 —— 它改的是**落点选择**,不是死/活线,
///    所以它的红只会表现为「落点挪了一格」。⚠ 动它之前先看 `damage` 那一族的判据。
const THIN_REF_MARGIN: f32 = 0.25;

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
fn damage_from_scan(err_cents: f64, voiced: f64, timbre: Option<(f64, f64)>, thin_ref: f64) -> f32 {
    let pitch = (((err_cents - 25.0) / 75.0).clamp(0.0, 1.0) as f32) * DAMAGE_MAX;
    let voicing = (((0.95 - voiced) / 0.45).clamp(0.0, 1.0) as f32) * DAMAGE_MAX;
    let (thin, quiet) = match timbre {
        Some((rms_db, low_ratio)) => (
            // S160h —— 门槛**相对该模型自己的基线**(见 [`THIN_REF_MARGIN`]);
            // `thin_ref` 由调用点从这份扫描自己算出来,老路径传 0.55 = 逐位同今天。
            (((low_ratio - thin_ref) / 0.40).clamp(0.0, 1.0) as f32) * DAMAGE_MAX,
            (((RMS_FREE_DB - rms_db) / 12.0).clamp(0.0, 1.0) as f32) * DAMAGE_MAX,
        ),
        None => (0.0, 0.0),
    };
    (pitch + voicing + thin + quiet).min(DAMAGE_MAX)
}

/// S160h —— 从一份扫描里算出 `thin` 的**相对门槛** = `usable` 内 `low_ratio` 的中位 + [`THIN_REF_MARGIN`]。
/// ⛔ 只用 `usable` 之内的格:音域之外那些格本来就烂,把它们算进中位会把门槛抬到没有意义。
/// ⛔ 扫描里没有 4 元组(拿不到 `low_ratio`)⇒ 回落到 0.55 = **逐位同今天**。
fn thin_ref(semitones: &serde_json::Map<String, serde_json::Value>, usable: (f32, f32)) -> f64 {
    // ⛔⛔ **出厂关**(`UTAI_RANGE_THIN_REL=1` 打开)。它是对的方向,但**手上只有四个装机记录,
    //   不足以把这条规则标定安全** —— 落地时它当场照红两条已登记的落点判据,而那两条红
    //   暴露的是**设计本身的两个失效模式**,不是夹具写错了:
    //   ⑴ `the_depth_cap_only_bites_once_the_shallowest_landing_is_already_deep` 的夹具是
    //      **low_ratio 线性斜坡** —— 中位数会跟着斜坡一起动,相对门槛在这种分布上**结构性失效**;
    //   ⑵ `the_landing_rule_will_not_walk_into_the_basement_for_a_better_score` 的夹具里
    //      「糊」的那一带占 usable 的 **60%** ⇒ **中位数本身就是糊的那个值**,门槛被抬到 1.10,
    //      于是整条曲线变平、夹具的前提塌掉。
    //   ⇒ 换成 p25 可以躲开 ⑵,但那样对**东雪莲**(唯一需要它的模型)又不起作用。
    //   ⇒ 真正的参照大概应该是「这个嗓子**舒服区**的基线」而不是 usable 的分位,
    //      但 `comfort` 在两个夹具里恰好等于 usable ⇒ 同样躲不开。**证据不够,先别翻默认。**
    // ⚠ 它改的是**落点选择**(不是死/活线),而**东雪莲是谱面轨验收所用的模型** ⇒
    //   翻它之前必须有东雪莲的前后耳判(用户 2026-08-24 点名的硬前置)。
    if !matches!(std::env::var("UTAI_RANGE_THIN_REL").as_deref(), Ok("1")) {
        return 0.55;
    }
    let mut v: Vec<f64> = semitones
        .iter()
        .filter_map(|(k, val)| {
            let midi: f64 = k.parse().ok()?;
            if midi < usable.0 as f64 || midi > usable.1 as f64 {
                return None;
            }
            val.as_array()?.get(3)?.as_f64()
        })
        .collect();
    if v.is_empty() {
        return 0.55;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = v[v.len() / 2];
    // ⛔⛔ **只放宽,不收紧**(`max`)。第一版是纯相对的 `med + margin`,当场照红两条落点判据 ——
    //   查出来是设计太激进:对**基线很干净**的模型(yachiyo `low_ratio` 中位 0.075)相对门槛
    //   会变成 0.325,**比今天的 0.55 还严**,于是一个客观上并不糊的格开始被判糊、落点被推深。
    //   而用户 2026-08-24 指出的缺陷**只有一个方向**:「天生低频重的嗓子被一条从别人身上抄来的
    //   绝对线系统性惩罚」(东雪莲 `thin > 0` 的格占 39%,另外三个只有 7-9%)。
    // ⇒ 取 `max(0.55, med + margin)`:**东雪莲被放宽,另外三个逐位不动**(可从 sidecar 复算)。
    //   ⚠ yuyuko 顶那几格因此仍然只算 0.57 的伤 —— 不要紧,它们已经被**电平闸**从死/活线里切掉了
    //   (见 [`RESCUE_LEVEL_FLOOR_DB`]),`thin` 不必再兼这份差事。
    (med + THIN_REF_MARGIN as f64).max(0.55)
}

/// S160h —— **死/活线的电平闸**(见 [`RESCUE_LEVEL_FLOOR_DB`])。从 `usable` 的两端往里收,
/// 直到遇上一格「扫描说它不比该模型自己最响的一格低 `floor` 以上」的。
/// ⛔ 扫描里没有 4 元组的格**一律放过**(老记录一个字不动:没有读数不等于坏)。
/// ⛔ 收窄不许把音域收成空的:两端相遇就原样返回并**响亮**报一句 —— 那说明这份扫描本身有问题。
fn narrow_usable_by_level(
    semitones: Option<&serde_json::Map<String, serde_json::Value>>,
    usable: (f32, f32),
    floor_db: f64,
) -> (f32, f32) {
    let Some(sc) = semitones else { return usable };
    let level = |m: i64| -> Option<f64> { sc.get(&m.to_string())?.as_array()?.get(2)?.as_f64() };
    let (lo0, hi0) = (usable.0.round() as i64, usable.1.round() as i64);
    let (lo, mut hi) = (lo0, hi0);
    // ⛔⛔ **只收顶,不收底。**第一版两端都收,当场把 yuyuko 从 `[36,82]` 收成 **`[60,79]`** ——
    //   少了 24 个半音。原因是结构性的:**一个嗓子的低音区天生就比它自己最响的中音区安静**,
    //   而 `rms_db` 是相对该模型自己的最大值的 ⇒ 两端都会读到很负的值。
    //   四份装机记录的底部 rms_db(usable 底往上 9 格):
    //     东雪莲 −5.9…−9.8 · akiko **−19.5…−16.9** · yachiyo −9.4…−6.2 · yuyuko **−16…−10.2**
    //   ⇒ 两端都收会把 akiko 收成 `[49,76]`、yachiyo `[42,77]`、yuyuko `[60,79]`
    //     —— **四个里毁三个**;只收顶则四个全对。
    // ⚠ 而用户报的缺陷**只在顶上**(模型在自己舒服区之上硬撑);底部安静是音乐上正常的,
    //   而且把低音往**上**搬的代价与往下完全不同(S159k 记过 cover 那条车道的反方向)。
    // ⛔⛔ 血训:第一版**有**一条判据写着「底不该动」,而它给的是一个**底部 rms 健康的合成夹具**
    //   ⇒ 那条断言结构上够不着这个坏法。**夹具不真实的判据 = 空判据。**
    //   现在的判据直接用四份装机记录的**真实底部读数**。
    while hi > lo && level(hi).is_some_and(|d| d < floor_db) {
        hi -= 1;
    }
    if hi <= lo {
        tracing::warn!(
            "range-extend: level gate would empty usable [{lo0},{hi0}] — scan looks broken, keeping it"
        );
        return usable;
    }
    if lo != lo0 || hi != hi0 {
        tracing::info!(
            "range-extend: usable narrowed by level gate ({floor_db:+.1} dB): \
             [{lo0},{hi0}] -> [{lo},{hi}]"
        );
    }
    (lo as f32, hi as f32)
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
    let scan = sp.get("semitones").and_then(|m| m.as_object());
    let usable_raw = pair("usable")?;
    // S160h —— **死/活线的电平闸**(见 [`RESCUE_LEVEL_FLOOR_DB`]):扫描说这一格比该模型自己
    // 最响的一格低 8 dB 以上 ⇒ 不算「唱得动」,即使音准完美。用户 2026-08-24 在 yuyuko 的
    // MIDI 轨 +7 上点名的三处「哑音 / 失声」全部是这种格(78/80/82,err 1-5 cents、voiced 1,
    // 但 rms_db −2.9 → −16.6)。⛔ 三个已验收模型(东雪莲 / akiko / yachiyo)在这道闸上**一格不动**,
    // 可从 sidecar 直接复算 —— 判据 `the_level_gate_only_moves_the_model_that_needs_it`。
    let usable = narrow_usable_by_level(scan, usable_raw, RESCUE_LEVEL_FLOOR_DB);
    // S146f: the scan's own bounds. Absent ⇒ = `usable` (a fact for pre-S146e records: nothing
    // could edit it then). ⚠ A hand-poisoned sidecar could hold a `usable_auto` NARROWER than
    // `usable`; union them so the split can only ever widen what the drag check accepts, never
    // secretly narrow it below what today's build already allows.
    // ⚠ S160h:`reach` 用的是**收窄之前**的 `usable_raw` —— 电平闸只该改「哪些音需要被救」,
    //   不该连带缩掉「落点够不够得着」。(收窄之后那几格的 `damage` 本来就已经饱和,
    //   落点搜索不会往那儿去;真要拦它是 `damage` 的活,不是 `reach` 的。)
    let reach = pair("usable_auto")
        .map(|a| (a.0.min(usable_raw.0), a.1.max(usable_raw.1)))
        .unwrap_or(usable_raw);
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
    // S160h —— `thin` 的门槛相对该模型自己的基线(见 [`THIN_REF_MARGIN`])。⛔ 用**收窄之前**的
    //   `usable_raw`:基线要描述「这个嗓子平时是什么样」,而收窄恰好切掉的是最不正常的那几格。
    let tref = scan.map_or(0.55, |m| thin_ref(m, usable_raw));
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
            d[slot as usize] = (damage_from_scan(err, voiced, timbre, tref) / DAMAGE_MAX * 255.0)
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
    let (a, b, _) = dead_only_plan_with_alts(note_nums, frames, transpose, range, tune);
    (a, b)
}

/// ⭐ S162 —— [`dead_only_plan_with`],外加每组的**落点候选**(`t1` 里最浅的那个;
/// 与 pick 相同时为 `None`)。
///
/// ## ⛔ 它为什么存在
/// 用户 2026-08-26:「akiko 的 ぴゃ 之前是达到过的」——**查实了,是退化**:
/// `land_scan` 实测 **落点关(S157c 之前)⇒ ぴゃ 落 78(好)**,**落点=3(今天)⇒ 落 77(坏)**。
/// ⛔ 但翻回旧默认会毁掉 S157c 的实质作用(薄区死音 akiko 30%→12% / yachiyo 34%→3%,
///    两把独立尺子 + Fisher p=0.0026);换全局破法也判负(241 个音:thin 赢 63 / shallow 赢 29,
///    塌掉 6→9)。⇒ **只能自适应:两个候选都渲,按实测选。**
/// ⭐ akiko 的 ぴゃ:`t1 = {−12,−13,−14}`,pick = −13,**候选 = −12 = 正确答案**。
/// ⭐ 代价已量:akiko 多 **3 个浅 shift**;yuyuko / 东雪莲 / yachiyo **多 0 个**。
pub fn dead_only_plan_with_alts(
    note_nums: &[i64],
    frames: &[i64],
    transpose: i64,
    range: &SpeakerRange,
    tune: RescueTuning,
) -> (Vec<DeadGroup>, Vec<(usize, usize)>, Vec<Option<i64>>) {
    let trim = tune.trim;
    let eff = |n: i64| (n + transpose).clamp(1, 127); // mirror transpose_note_pitch's clamp
    let ms = |from: usize, to: usize| -> f32 {
        let f: i64 = (from..to).map(|k| frames.get(k).copied().unwrap_or(0).max(0)).sum();
        f as f32 * 1000.0 / super::score2svc::CV_FPS as f32
    };
    let mut out = Vec::new();
    let mut alts: Vec<Option<i64>> = Vec::new();
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
                    // ⛔⛔ S159zk —— **断点的可行性**:`dead_group_windows` 的 `GUARD_FRAMES`
                    // 会让每个窗**倒着伸进旁边那个音** 2 帧,好让 10 ms 交叉淡化压在它身上。
                    // 而 donor 的每一遍是**按那一遍的位移唱整条谱**的
                    // (`voice.render(…, transpose, s, …)`)⇒ 如果旁边那个音在**这一遍**上
                    // 掉出音域,模型在那里**直接塌掉**,而塌陷渗进交界。
                    //
                    // 用户 2026-08-22 耳判:「和『う』(1033)连接的 1034『た』在 zk 也塌了」。
                    // 取证(鹅妈妈 +7 × 东雪莲,逐 20 ms):`[1034]` 起音 −9.3 → **−26.5 dBFS**,
                    // 而 `−2` 那一遍的 donor 在 `[1033]` 上读 **−46.6 dB**(它在那一遍是 MIDI 90,
                    // 顶是 79)对比 `−14` 那一遍的 −23.5。f0 在交界处先掉到 0,再从 1245 Hz
                    // 一路滑到 748 —— **模型正在从一个唱不动的音往下滑**。
                    //
                    // ⭐⭐ 全曲统计,阴性对照干净:窗边紧邻的 127 个唱音里,**在那一遍掉出音域的
                    // 24 个**比组内音低 **−11.4 dB**,而**没掉出去的 103 个是 −0.3 dB**。
                    // ⛔ 而且那 24 个**一个不落全是二级拆组造出来的**(拆组前的 67 条边:**0 个**)。
                    //
                    // ⇒ 这一条:断点只许落在「**两侧紧邻的那个音,在对方那一组的位移上都唱得动**」
                    //    的地方。3:16 上它把断点从 `[1033]|[1034]` 推到 `[1034]|[1035]`
                    //    (`[1034]` 在 −2 上是 MIDI 78,唱得动)—— 收益最大的 `[1035]` 照样脱出。
                    // ⛔⛔ **`slot_reachable`,不是 `slot_singable`** —— 与
                    // [`minimal_rescue_shift_capped`] 里那条同一个理由,而且是同一次学费(S146f):
                    // 这问的是「**模型**在这一格上发不发得出声」,所以必须读**扫描**的边界,
                    // **永远不许读用户那条救援线**。用 `slot_singable` 会让「用户把音域上限调低」
                    // 变成「拆得更少 ⇒ 陪绑更深」—— 正是用户当年耳判报的那条退化。
                    // ⚠ 实测不受影响:那 24 处坏边的邻音在该遍上是 MIDI **81-90**,
                    //    而东雪莲的 `usable_auto` 顶是 **80** ⇒ 两个谓词都判它唱不动。
                    let neighbour_ok = |k: usize, shift: i64| -> bool {
                        match note_nums.get(k).copied() {
                            Some(x) if x > 0 => range.slot_reachable(eff(x) + shift),
                            _ => true, // 休止 / 越界 ⇒ 护栏伸不进任何唱音
                        }
                    };
                    let mut best: Option<(usize, usize, f32)> = None;
                    for w in 1..ds.len() {
                        let (p, q) = (ds[w - 1], ds[w]);
                        let (Some(ls), Some(rs)) = (need(cf, p), need(q, cl)) else { continue };
                        // ⭐⭐⭐ S163 —— 这里**曾经**是一条 `continue`(S159zk):右组的窗倒着伸进
                        // `q-1`、左组的窗往前伸进 `p+1`,只要那个邻音在对方那一遍掉出音域就
                        // **否决整个断点**。它拦的危害是真的(那样的 24 个音比组内音低 −11.4 dB),
                        // 但**载体是护栏,不是断点** —— 而 [`dead_group_windows_raw`] 现在会在
                        // 这种边上把护栏直接收到 0(判据 `foreign_shift` 与那一族同延)。
                        //
                        // ⛔ 必须撤的理由(用户 2026-08-26 点名的病灶,坐标齐全):
                        // yuyuko 2:09.002-2:14.056「嗓子里卡着痰」= 组 `[492..495]`
                        // ま(82)/た(83)/む(92)/か(90),`usable [36,79]` ⇒ 四个全是死音,
                        // 整组 −14 ⇒ **ま 落到 68、た 落到 69**(女声模型在那儿唱气泡音,
                        // PSOLA 再把它拉高 14 度)。断点 `493|494` 的 gain = **4140 > 3000**
                        // ⇒ 不是阈值挡的,是 `neighbour_ok(494, −4)` 否的(む 在 −4 上是 88 > 79)。
                        // ⇒ 旧规则拿 **40 ms** 护栏塌陷换 **360 ms** 气泡音,**两害相权差一个数量级**。
                        //
                        // ⛔ 可归因性(铁律):被旧规则否过的断点逐条打日志,否则「这一刀拆开的」
                        // 与「本来就拆得动的」显示成同一种样子。
                        // ⛔⛔ S163 —— 这条 `continue` 被撤过一次(理由:危害的载体是护栏不是
                        // 断点,而 [`dead_group_windows_raw`] 现在会把那种边的护栏收到 0),
                        // **一天之内又装了回来**,因为撤它的收益在耳判上是空的、代价是实的:
                        // * 收益:卡痰段 `[492]ま` 落点 68 → 78(PSOLA 拉伸 14 度 → 4 度)——
                        //   用户听 I2 的原话是「**和之前有区别吗**」⇒ **卡痰的根因不是落点深度**;
                        // * 代价:akiko `[685]ぴゃ` 退化。撤掉它之后断点 `[685|686]` 变可行,
                        //   把 686 拆走 ⇒ S157 那一刀(按 `low_ratio` 把 ぴゃ 往 −14 推)**完全生效**
                        //   ⇒ 落点 77 → 76,同一次 run 的 `donor_post` 实测**上方谐波弱 6.6 dB**
                        //   (−36.23 → −42.86),而用户耳判 77 好。
                        //   ⚠ 也就是说:这条规则一直在**意外地**压着 S157 那一刀,而 S157 的
                        //   依据(76 的 low_ratio 0.211 优于 78 的 0.388)与耳判相反。
                        //   ⛔ **那条 `low_ratio` 排序仍然欠着一次复核**,别当它是对的。
                        if !neighbour_ok(q.wrapping_sub(1), rs) || !neighbour_ok(p + 1, ls) {
                            continue;
                        }
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
            // ⭐ S162 —— **落点候选**:`t1` 里最浅的那个(= 关掉 `thinner` 那一层会选到的)。
            // 与 pick 不同时才带出来;渲染时两个都渲、按实测选(见
            // `apply_dead_only_windows_with` 的 `alts`)。
            // ⛔ 它**不改今天的 pick** —— 这一笔对音频逐位不变,只是多算一个数。
            alts.push(None); // ⭐ 候选在函数末尾统一填(见那一段)
            out.push(DeadGroup { start: a, end: b, shift });
            }
        }
        i = j + 1;
    }
    // ⭐⭐ S162 —— **落点候选 = 「S157c 放宽预算之前那条规则」整份计划的落点**,按组的
    // **起始音**对齐取。
    //
    // ⛔ 为什么必须整份重跑,而不是逐组重算:逐组重算拿不到同样的上下文(相同的
    //    `dead` / 簇边界 / `whole_shift` 继承路径),实测给出过 **−3** 这种**根本救不了
    //    MIDI 90** 的候选 —— 而日志把它和「没有候选」显示成同一种样子,卡了我两轮。
    // ⛔ 为什么是这个候选:`land_scan` 实测 **落点关 ⇒ akiko 的「ぴゃ」落 −12(rel −0.2)**、
    //    **落点=3 ⇒ −13(rel −12.6)**,而用户说「之前是达到过的」——就是这一格。
    //    ⭐ 而且**不用把 S157c 翻回去**:它的业绩(薄区死音 30%→12%)一个字不动,
    //    只是多给一个候选,由**实测**去裁。
    // ⚠ 递归安全:重跑那一份传 `landing: None`,它自己算候选时**不再重跑**(下面那个哨兵)。
    if tune.landing.is_some() && !ALT_PLAN_REENTRY.with(|f| f.get()) {
        ALT_PLAN_REENTRY.with(|f| f.set(true));
        // ⛔⛔ S162 —— 候选用 **`landing = Some(1)`**(最窄的那一档),**不是 `None`**。
        //    我先写了 `None`,而那条路在 akiko 的「ぴゃ」上给出 **−3** —— 一个**根本救不了
        //    MIDI 90** 的废值(日志把它和「没有候选」显示成同一种样子,卡了我三轮)。
        //    根因是我把 `land_scan` 读错了:那 6 档是 `landing = 1,3,5,7,9,12`,**没有「关」**;
        //    给出好落点(−12,rel −0.2)的是 **`landing=1`**,今天出厂的 `Some(3)` 给 −13(−12.6)。
        let alt_tune = RescueTuning { landing: Some(1), ..tune };
        let (alt_plan, _, _) =
            dead_only_plan_with_alts(note_nums, frames, transpose, range, alt_tune);
        ALT_PLAN_REENTRY.with(|f| f.set(false));
        for (gi, g) in out.iter().enumerate() {
            // ⛔⛔ S163 —— **必须 `start` 与 `end` 都相同**。
            //
            // 只按 `start` 找是 S162 那条「张冠李戴」的**第二次复发**(第一次是按组下标
            // 平行传给渲染层,已修):两份计划的**预算不同 ⇒ 分组也不同**,实测
            // `landing = 1` 给出 **95 组**而 `landing = 3` 给出 **90 组**。于是
            // 「起点相同」的两个组可能**根本不是同一批音**,而它的落点是为**另一批音**解出来的。
            //
            // 实测(akiko × 炉心 +7):plan `[698..704]` 需要 −10(最高音目标 MIDI 87),
            // 而同起点的 alt 组是 `[698..701]`(把那个高音**分出去了**)⇒ 它的落点是 **−4**。
            // 把 −4 当候选交给渲染层,它在整窗电平上反而**更贴近**邻居 ⇒ **赢了** ⇒
            // 那个音的救援被整个丢掉:实测谐波能量占比 **−0.98 → −19.56 dB**、
            // 梳深 26.9 → 8.4、次基频 −25.1 → **+19.9**,与**关掉扩展的 `base` 几乎逐格相同**。
            // 这就是用户 2026-08-26 报的「4:09.478 塌了」,而**同一首歌里它发生 5 次**。
            //
            // 人群面(4 个模型 × 2 首谱 × 移调 0/7):带候选 341 组,**22 组(6.5%)**
            // 起点相同而范围不同;东雪莲 × 鹅妈妈 +7 上是 **6/7**。
            // ⇒ 范围一对不上就**不带候选**(那一组照走今天的落点,零风险)。
            alts[gi] = alt_shift_for(g, &alt_plan);
            if alts[gi].is_none() {
                if let Some(x) = alt_plan.iter().find(|x| x.start == g.start && x.end != g.end) {
                    tracing::info!(
                        "range: notes[{}..={}] landing candidate DROPPED — the narrow-budget plan grouped it as \
                         [{}..={}] instead, so its {:+} was solved for a DIFFERENT set of notes",
                        g.start,
                        g.end,
                        x.start,
                        x.end,
                        x.shift
                    );
                }
            }
        }
    }
    (out, unfixable, alts)
}

/// ⭐ S163 —— 一个组的定案:**三层**,每一层都只在**同一个音**上比。
///
/// ⓪ **塌陷否决**:某个候选把一个唱音唱成了 ≥`repair_ms` 的静音,而另一个没有 ⇒ 剔掉它。
///    (`repair_ms == 0` ⇒ 这一层关掉。)
/// ⓐ **谐波否决**:逐音比谐波能量占比,某个候选在**任何一个音**上差 ≥`harm_eps` dB ⇒ 剔掉。
///    ⛔ 互相否决时谁也不剔 —— 那不是一边倒。
/// ⓑ **电平**:逐音 `|rel|`,组取**最差的音**;取不到参照 ⇒ 无穷大 ⇒ 判平 ⇒ 保持计划的落点。
///
/// 返回排好序的下标(第一个 = 选中的)。
fn decide_group(
    cand: &[(usize, i64, usize, Vec<f32>, CandScore)],
    ji: usize,
    plan_shift: i64,
    harm_eps: f32,
    repair_ms: f32,
    comb_floor: f32,
    // ⭐ S163 —— 谱峰宽度的**倍数**门限（≤ 1 = 关）。见 [`landing_width_eps`]。
    width_eps: f32,
    // ⭐⭐⭐⭐ S163 —— 上方谐波音内塌陷当排序主键的 eps(dB);`0` = 关 = 逐位回到今天。
    //    见 [`landing_usag_eps`]。
    usag_eps: f32,
    // ⭐⭐⭐ S163 —— 对手轴的闸(dB)。见 [`landing_usag_dim_cap`]。
    usag_dim_cap: f32,
    // ⭐⭐⭐ S165 —— `2·f0` 电平当排序键的 eps(dB);`0` = 关 = 逐位回到今天。
    //    见 [`landing_h2_eps`]。
    h2_eps: f32,
    // ⭐⭐⭐⭐ S165 —— **失配**(响度 ↔ 抖动)当排序键的 eps(dB);`0` = 关。
    //    见 [`landing_mismatch_eps`]。
    mism_eps: f32,
) -> Vec<usize> {
    let mut mine: Vec<usize> = (0..cand.len()).filter(|&c| cand[c].0 == ji).collect();
    if mine.len() < 2 {
        return mine;
    }
    if repair_ms > 0.0 {
        let ok: Vec<usize> =
            mine.iter().copied().filter(|&c| cand[c].4.worst_gone() < repair_ms).collect();
        if !ok.is_empty() && ok.len() < mine.len() {
            mine = ok;
        }
    }
    // ⓐ2 ⭐ S163 —— **梳深否决**:某候选把谐波之间糊成一片(梳深塌到门槛以下),
    //    而另一个没有 ⇒ 剔掉它。⛔ 与谐波占比**不是同一件事**(见 [`CandScore::comb`])。
    if comb_floor > 0.0 && mine.len() > 1 {
        let ok: Vec<usize> =
            mine.iter().copied().filter(|&c| cand[c].4.worst_comb() >= comb_floor).collect();
        if !ok.is_empty() && ok.len() < mine.len() {
            mine = ok;
        }
    }
    // ⓐ 3 ⭐⭐⭐ S163 —— **谱峰宽度否决**：某候选把谐波自身糊成一片
    //    （峰宽比另一个差过 `width_eps` 倍）⇒ 剔掉它。
    //    ⛔ 与上面三根**都不是一件事**：实测 yuyuko 4:36 接缝两侧峰宽
    //    **12.33 vs 0.99（差 12 倍）**，而填充度读到 −18 vs −29（**方向还反**）。
    //    ⭐ 它为什么在候选之间有效：短音的 `donor_pre` 峰宽按落点是
    //    **77 → 3.50 而 78 → 11.88**（只差 1 个半音，正好在候选范围内）；
    //    ⛔ 而 sidecar 的 `low_ratio` 在同两格上是 **77 → 0.616（最差）/ 79 → 0.129（好）**
    //    —— **与峰宽完全相反**，而 S157 那条排序用的就是 `low_ratio`。
    //    ⚠ 只在**同一个音的两个候选**之间比（同 f0）—— 跨音高不可比。
    if width_eps > 1.0 && mine.len() > 1 {
        let best = mine
            .iter()
            .map(|&c| cand[c].4.worst_width())
            .filter(|v| *v > 0.0)
            .fold(f32::INFINITY, f32::min);
        if best.is_finite() {
            let ok: Vec<usize> = mine
                .iter()
                .copied()
                .filter(|&c| {
                    let w = cand[c].4.worst_width();
                    w <= 0.0 || w <= best * width_eps
                })
                .collect();
            if !ok.is_empty() && ok.len() < mine.len() {
                tracing::info!(
                    "range: peak-width veto dropped {} of {} candidates for job {ji}                      (best width {best:.2}%)",
                    mine.len() - ok.len(),
                    mine.len()
                );
                mine = ok;
            } else {
                tracing::info!(
                    "range: peak-width veto had nothing to drop for job {ji}                      (best {best:.2}%, all {:?})",
                    mine.iter().map(|&c| cand[c].4.worst_width()).collect::<Vec<_>>()
                );
            }
        }
    }
    // ⭐⭐⭐⭐ S165 —— **音高闸(硬否决)**,排在所有排序轴之前。
    //
    // ⛔ 唱错音是**质的失败**,不参与任何权衡:一个塌了八度的候选可能在 `rel`/`usag`/失配上
    //    全都好看(音高塌了之后谐波反而更协调)。用户 2026-08-29 听到的那个「炸得非常烂」
    //    正是这么来的 —— 目标 1480 Hz 而实唱 **320 Hz**。
    // ⛔ **只在还有别的候选活着时才否决** —— 全部都跑调时不能把一组清空
    //    (那会让 `mine.first()` 拿不到东西,整组失去救援)。
    if mine.len() > 1 {
        let ok: Vec<usize> =
            mine.iter().copied().filter(|&c| cand[c].4.worst_pitch_err() <= PITCH_GATE_CENTS).collect();
        if !ok.is_empty() && ok.len() < mine.len() {
            tracing::warn!(
                "range: pitch gate dropped {} of {} candidates for job {ji} (worst errors {:?} cents)",
                mine.len() - ok.len(),
                mine.len(),
                mine.iter()
                    .filter(|&&c| cand[c].4.worst_pitch_err() > PITCH_GATE_CENTS)
                    .map(|&c| (cand[c].1, cand[c].4.worst_pitch_err().round() as i32))
                    .collect::<Vec<_>>()
            );
            mine = ok;
        }
    }
    if harm_eps > 0.0 && mine.len() > 1 {
        let loses: Vec<usize> = mine
            .iter()
            .copied()
            .filter(|&x| {
                mine.iter().any(|&y| {
                    y != x && {
                        let w = cand[x]
                            .4
                            .harm
                            .iter()
                            .filter_map(|&(i, hx)| cand[y].4.harm_of(i).map(|hy| hx - hy))
                            .fold(f32::INFINITY, f32::min);
                        w.is_finite() && w < -harm_eps
                    }
                })
            })
            .collect();
        let live: Vec<usize> = mine.iter().copied().filter(|c| !loses.contains(c)).collect();
        if !live.is_empty() && live.len() < mine.len() {
            mine = live;
        }
    }
    mine.sort_by(|&x, &y| {
        // ⭐⭐⭐⭐ S163 —— **主键:上方谐波的音内塌陷**(只在差过 `usag_eps` 时才认)。
        //
        // ⛔ 为什么它排在 `rel` 前面:实测 `rel` 在用户点名的坐标上**方向是反的**
        //    (4:07 那组 `rel` 偏好 −2,而 −2 的 upper-sag 是全部候选里最差的 −10.8)。
        //    而 ぴゃ 那组修补遍**已经把好档渲出来了**,是排序把它丢了。
        // ⛔ 为什么要 eps:两个候选只差零点几 dB 时这根轴和噪声底分不开,而用户给的两个
        //    「听起来正常」的对照正好落在那个区间(候选间只差 2.0 dB),
        //    三个缺陷点则差 17.9 / 7.6 / 6.6 dB ⇒ eps ≈ 3 时缺陷全动、对照全不动。
        // ⭐⭐⭐⭐ S165 —— **失配**(响度 ↔ 抖动)。用户 2026-08-28 定案的那根轴。
        //
        // ⛔ **相对判据**:比的是「这个候选相对那个候选,在**某一个音**上最多能把失配改善多少」,
        //    而不是各自的绝对失配值 —— 绝对门限会把「当前虽差但没有更好替代」也否掉。
        // ⛔ **最差口径**在 [`CandScore::worst_mism`] / [`note_mismatch`] 里,别改成平均。
        if mism_eps > 0.0 {
            // ⛔⛔ 比的是**各自最差那个音**的失配,不是「逐音最大改善」。
            //    第一版写成 `best_mism_vs`(逐音取最大改善),被判据当场抓住:
            //    夹具里 −8 在音 0 上比 −15 好 2.0、在音 1 上差 1.60 ⇒ 「最大改善」让 −8 赢了,
            //    而**决定听感的是最差的那个音**,不是它在哪一根上赢得最多。
            // ⚠ 与 `h2` 那次「被组里的气声音钉死」看似像、其实相反:那里某个音在所有候选上都差
            //    ⇒ 它掩盖了**别的音**的改善;而这里**最差的音本身就是决定听感的那个**,
            //    所有候选在它上面一样差时不发言是**对的**。
            let gx = -cand[x].4.worst_mism();
            let gy = -cand[y].4.worst_mism();
            MISM_STAT.with(|st| {
                let mut b = st.borrow_mut();
                b.0 += 1;
                if gx.is_finite() && gy.is_finite() {
                    let g = (gx - gy).abs();
                    if g > b.4 {
                        b.4 = g;
                    }
                    let w = (-gx).max(-gy);
                    if w > b.5 {
                        b.5 = w;
                    }
                }
            });
            if gx.is_finite() && gy.is_finite() && (gx - gy).abs() > mism_eps {
                // ⛔⛔ 对手轴闸 —— 用户 2026-08-28 的上线警告:
                //    「小心这个新条件**别再把那个哑音和伪影引回来**」。
                //    `UTAI_RANGE_H2` 那次哑音时诊断计数是 `BLOCKED 0 次` = 闸形同虚设。
                let (w, l) = if gx > gy { (x, y) } else { (y, x) };
                let dim_ok = if usag_dim_cap > 0.0 {
                    cand[w].4.worst_dim_vs(&cand[l].4) <= usag_dim_cap
                } else {
                    true
                };
                // ⭐⭐⭐⭐ S165 —— **第二道对手轴闸:不许把音内的谷挖得更深。**
                //    ⛔ 上面那道用 `uplev`(挡「变闷」),实测在 4:07.466 上**一次都没拦住**,
                //    而那次交换把音内跌幅从 16.0 挖到 21.4 dB(听感:哑噪声 → 断音)。
                //    见 [`MISM_DIP_CAP`] 与 [`note_dip_db`]。
                let dip_ok = cand[w].4.worst_dip_vs(&cand[l].4) <= MISM_DIP_CAP;
                let dim_ok = dim_ok && dip_ok;
                MISM_STAT.with(|st| {
                    let mut b = st.borrow_mut();
                    b.1 += 1;
                    if dim_ok {
                        b.2 += 1
                    } else {
                        b.3 += 1
                    }
                });
                if dim_ok {
                    // 改善大的靠前
                    return gy.partial_cmp(&gx).unwrap_or(std::cmp::Ordering::Equal);
                }
            }
        }
        // ⭐⭐⭐ S165 —— **`2·f0` 电平**:排在 `usag` 之前,但 eps 更大(只在差得很多时发言)。
        //
        // ⛔ 为什么排最前:用户 2026-08-28 听过 `−8 → −13` 的探针臂后说
        //    「**f1『强』/『实』确实在听感上更好,即使它把 f4 炸了**」——
        //    `2·f0` = 1975 Hz 落在耳朵最敏感的区间,而更高的谐波感知权重低得多。
        // ⛔ 为什么 eps 要比 `usag_eps` 大:它排在前面,eps 小就会把已经过耳判的
        //    `usag`/`gone` 决策整个盖掉。实测这根轴的候选间差距**要么很小、要么十几 dB**
        //    (目标音 −8 是 −17.0 而 −13 是 −5.2)⇒ 大 eps 不会漏掉真正要救的。
        if h2_eps > 0.0 {
            // ⛔⛔ S165 —— **必须逐音比,不能用 `worst_h2()` 的差**。
            //    第一版就是那么写的,结果 47/54 次修补遍触发**一条落点都没改**:
            //    `worst_h2` 是「组内最差的那个音」,而组里常有一个音(气声/清音)在**所有**候选上
            //    都读得很差 ⇒ 它把整组的 `worst_h2` 钉死 ⇒ 候选之间差 < eps ⇒ 这根轴永远不发言。
            //    ⇒ 改用 [`CandScore::best_h2_vs`]:逐音配对取**最大增益** —— 这根轴本来就是来救
            //    「**某一个音**被掏空」的,组里其它音本来就好。
            //    ⚠ `best_h2_vs` 那段 doc 早就写明了这个理由,是**接线时没接上**;
            //    编译器的 `method best_h2_vs is never used` 当场指出了它。
            let hx = cand[x].4.best_h2_vs(&cand[y].4);
            let hy = cand[y].4.best_h2_vs(&cand[x].4);
            // ⭐ S165 —— 诊断:这一支**被问了几次、几次差距过 eps、闸放行/拦住几次**。
            //    ⛔ 没有它,实机「47 次触发 / 0 条落点改变」连着两轮**无法归因**
            //    (日志里 `worst_h2` 差 14 dB 的组也没换,而 `best_h2_vs` 是**逐音**的,
            //     两个候选各自最差的可能是**不同的音** ⇒ 逐音增益可能根本不到 eps)。
            H2_STAT.with(|st| st.borrow_mut().0 += 1);
            if (hx - hy).abs() > h2_eps {
                // ⛔⛔ 对手轴闸 —— **每一根参与排序的轴都要配**(S163 §32 的血训:
                //    `gone` 那一支当初没配,直接造出全曲唯一一个变闷 >2.7 的音)。
                let dim_ok = if usag_dim_cap > 0.0 {
                    let (w, l) = if hx > hy { (x, y) } else { (y, x) };
                    cand[w].4.worst_dim_vs(&cand[l].4) <= H2_DIM_CAP
                } else {
                    true
                };
                H2_STAT.with(|st| {
                    let mut b = st.borrow_mut();
                    b.1 += 1;
                    if dim_ok { b.2 += 1 } else { b.3 += 1 }
                    let gap = (hx - hy).abs();
                    if gap > b.4 { b.4 = gap; b.5 = dim_ok; }
                });
                if dim_ok {
                    // 逐音增益大的靠前
                    return hy.partial_cmp(&hx).unwrap_or(std::cmp::Ordering::Equal);
                }
            }
        }
        if usag_eps > 0.0 {
            let (ux, uy) = (cand[x].4.worst_usag(), cand[y].4.worst_usag());
            if (ux - uy).abs() > usag_eps {
                // ⭐⭐⭐ 对手轴的闸(见 [`landing_usag_dim_cap`]):赢家不许比输家**闷**过
                //    `dim_cap`。实测 `[687]く` 换档 `usag` 只买到 +0.98 却付 6.25 dB ⇒ 必须拦。
                // ⛔ 只在两边都量得到强度时才判 —— 量不到就当没有代价证据,不许凭空拦。
                let dim_ok = if usag_dim_cap > 0.0 {
                    // 谁的 usag 好谁是「赢家」;查赢家在**任何一个音**上有没有闷过头。
                    // ⛔ 逐音,不是组中位 —— 见 [`CandScore::worst_dim_vs`] 的 doc。
                    let (w, l) = if ux > uy { (x, y) } else { (y, x) };
                    cand[w].4.worst_dim_vs(&cand[l].4) <= usag_dim_cap
                } else {
                    true
                };
                if dim_ok {
                    // 越接近 0(越不塌)越靠前
                    return uy.partial_cmp(&ux).unwrap_or(std::cmp::Ordering::Equal);
                }
            }
        }
        // ⭐⭐⭐ S163 §26.4③ —— **静音次键**:门之后仍有多个候选时,唱没了明显更少的优先。
        //
        // ⛔ 为什么需要它:⓪ 层是**二值门**(`worst_gone < repair_ms` 才留),
        //    而 ぴゃ 那组修补遍手里是
        //    `[(-12,60ms), (-14,40ms), (-15,60ms), (-10,40ms), …] ⇒ kept -12` ——
        //    40 与 60 **都 < 门限** ⇒ 一个都不剔 ⇒ 回到 `rel` ⇒ **手里有 40 ms 的却留了 60 ms 的**。
        //    ⇒ 「唱没了多少毫秒」这件事在排序里**完全不参与**,这一行把它接进去。
        // ⛔ 位置:`usag` 之后、`rel` 之前 —— 整段被抽干比零星静音更严重,
        //    而零星静音又比「电平贴不贴邻居」硬。
        if repair_ms > 0.0 {
            let (gx, gy) = (cand[x].4.worst_gone(), cand[y].4.worst_gone());
            if (gx - gy).abs() > GONE_SORT_EPS_MS {
                // ⛔⛔ S163 §32 —— **这一支也必须过对手轴闸**。
                //    实测:yuyuko `group[13816..13884]`(用户点名的 4:36.151)被这一支从 −4 换到 −6
                //    (静音 40 → 20 ms),成品上 `usag` 只买到 **+3.99** 而上方谐波
                //    **−9.28 → −19.37 = 闷 10.09 dB** ⇒ 净亏,而且是全曲 224 个音里唯一变闷 >2.7 的。
                //    ⇒ 原因是闸只写在 `usag` 那一支里,而这两档的 `usag` 差不到 `eps`
                //    ⇒ 整个 usag 分支跳过 ⇒ 走到这里,而这里**没有任何对手轴保护**。
                // ⭐ 血训:**每一根参与排序的轴都要配对手轴闸,不是只给最后加的那一根配。**
                let dim_ok = if usag_dim_cap > 0.0 {
                    let (w, l) = if gx < gy { (x, y) } else { (y, x) };
                    cand[w].4.worst_dim_vs(&cand[l].4) <= usag_dim_cap
                } else {
                    true
                };
                if dim_ok {
                    return gx.partial_cmp(&gy).unwrap_or(std::cmp::Ordering::Equal);
                }
            }
        }
        let sx = cand[x].4.worst_rel().unwrap_or(f32::INFINITY);
        let sy = cand[y].4.worst_rel().unwrap_or(f32::INFINITY);
        sx.partial_cmp(&sy)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| (cand[x].1 != plan_shift).cmp(&(cand[y].1 != plan_shift)))
    });
    mine
}

/// ⭐⭐ S163 —— 从**窄预算那份计划**里给一个组取落点候选。
///
/// ⛔⛔ **必须 `start` 与 `end` 都相同。**两份计划的预算不同 ⇒ **分组也不同**
/// (实测 akiko × 炉心 +7:`landing = 1` 出 **95 组**、`landing = 3` 出 **90 组**),
/// 于是「起点相同」的两个组可能**根本不是同一批音**,它的落点是为**别的音**解出来的。
///
/// 实测那一处(用户 2026-08-26 报的 **4:09.478「み」**):
/// plan `[698..=704]`(最高音目标 MIDI **87**)拿到窄预算 `[698..=701]` 的 **−4**
/// ⇒ 落在 **83**,**仍在死区**;而它在**整窗**电平上反而更贴近邻居 ⇒ 赢了
/// ⇒ 那个音的救援被整个丢掉(谐波能量占比 −0.98 → **−19.56 dB**,与关掉扩展的 `base` 几乎相同),
/// 而这在同一首歌里**发生 5 次**。人群面:341 组带候选里 **22 组(6.5%)**是这个形状。
///
/// ⭐ 抽成具名函数的唯一理由:**判据要咬得到它**(见
/// `a_landing_candidate_is_never_borrowed_from_a_differently_grouped_plan` 的 ⑶)。
fn alt_shift_for(g: &DeadGroup, alt_plan: &[DeadGroup]) -> Option<i64> {
    alt_plan
        .iter()
        .find(|x| x.start == g.start && x.end == g.end)
        .map(|x| x.shift)
        .filter(|&s| s != g.shift)
}

thread_local! {
    /// ⛔ 防止「算候选」自己再去算候选(那会指数爆炸)。见 `dead_only_plan_with_alts` 末尾。
    static ALT_PLAN_REENTRY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// ⭐⭐ S165 —— **失配**这一支的诊断:
    /// `(被问几次, gap>eps 几次, 闸放行, 闸拦住, 见过的最大 gap, 见过的最大失配值)`。
    /// ⛔ 存在的理由:第一版 `eps=0.5` 是照 **Python 侧的尺度**定的,而 Rust 侧用一阶零相位低通,
    /// 读数小约 2 倍(参照曲线 Python 1.60→1.38 vs Rust 0.62→0.33)⇒ **门限相对读数太大,
    /// 几乎永远不触发** ⇒ 整轮渲染白跑。**先让引擎自己报分布,别再猜。**
    static MISM_STAT: std::cell::RefCell<(u64, u64, u64, u64, f32, f32)> =
        const { std::cell::RefCell::new((0, 0, 0, 0, 0.0, f32::NEG_INFINITY)) };
    /// ⭐ S165 —— `2·f0` 这一支的诊断计数:
    /// `(被问几次, 差距过 eps 几次, 闸放行, 闸拦住, 见过的最大差距, 那次闸放没放行)`。
    /// ⛔ 存在的理由:实机「47 次触发 / 0 条落点改变」连着两轮无法归因。
    static H2_STAT: std::cell::RefCell<(u64, u64, u64, u64, f32, bool)> =
        const { std::cell::RefCell::new((0, 0, 0, 0, 0.0, false)) };
}

/// ⭐⭐ S165 —— 打印并清空**失配**那一支的诊断。
fn log_mism_stat(where_: &str) {
    MISM_STAT.with(|st| {
        let mut b = st.borrow_mut();
        if b.0 == 0 {
            return;
        }
        tracing::info!(
            "range: mismatch-axis [{}] asked {} times, gap>eps {} times, gate PASSED {} / BLOCKED {},              largest gap {:.3}, worst mismatch seen {:.3}",
            where_, b.0, b.1, b.2, b.3, b.4,
            if b.5 > f32::NEG_INFINITY { b.5 } else { 0.0 }
        );
        *b = (0, 0, 0, 0, 0.0, f32::NEG_INFINITY);
    });
}

/// ⭐ S165 —— 打印并清空 `2·f0` 那一支的诊断计数。
fn log_h2_stat(where_: &str) {
    H2_STAT.with(|st| {
        let mut b = st.borrow_mut();
        if b.0 == 0 {
            return;
        }
        tracing::info!(
            "range: h2-axis [{}] asked {} times, gap>eps {} times, gate PASSED {} / BLOCKED {}, largest gap {:.1} dB ({})",
            where_, b.0, b.1, b.2, b.3, b.4,
            if b.5 { "passed" } else { "BLOCKED" }
        );
        *b = (0, 0, 0, 0, 0.0, false);
    });
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
///
/// ## ⛔⛔⛔ S163 —— 3000 → 2000 **试过并且回滚了**(一天之内)
///
/// 下面那套定价**看起来**很完整,而它**漏了最贵的一列:接缝密度**。
/// 实测(yuyuko × 炉心 +7,用户点名的两段):
/// * 次怪段 0:56.6-1:01.4:**2 窗 / 0.63 接缝每秒 → 5 窗 / 1.90 接缝每秒(3 倍)**;
/// * 最怪段 2:09:2 窗 → 3 窗。
/// 而「买到」的那一侧**耳判没兑现** —— 用户听完新臂的原话是「和之前有区别吗」。
/// ⇒ **卡痰的根因不是落点深度**,这一刀买的是空的、卖的是实的。
///
/// ⛔⛔ **血训:给一个参数定价时,买卖两侧都必须列全。**
/// 我列了「落点抬高」与「donor 遍数」,漏了接缝 —— 而拼接从 S162 起就是已知的主要伪影源之一。
/// **下次给拆组类的旋钮定价,接缝数必须和收益列在同一张表里。**
///
/// ## (存档)当时那套定价 —— 3000 → 2000
///
/// ⑴ **缝变便宜了**:[`dead_group_windows_raw`] 现在会在「邻音在本遍唱不动」的边上把护栏
///    收到 0 ⇒ 拆组造出来的缝不再把一段塌陷拌进交界(S159zk 实测那样的 24 个音低 **−11.4 dB**)。
///    3000 是在**缝还带着那个代价**的时候定的。
/// ⑵ **门的形状不对,而 3000 正好从中间切过同一个病灶**:`gain = ms × (|here| − |side|)`
///    假设「代价 ∝ 深度」,而配对法实测(同母音 × 同音高 × 深 vs 浅 + 空对照,跨两模型全部同向)
///    是**超线性**的 —— −2/−4/−6 偏离 ≈ 0(**透明**),−8 起明确、之后单调,−14 上方谐波弱 7-11 dB。
///    * yuyuko `[492..496]`(用户 2026-08-26 点名的「卡痰」):`ls = −4` ⇒ 360 ms × (14−4) = **3600 > 3000** ⇒ 拆;
///    * akiko **同一组**:`ls = −6` ⇒ 360 ms × (14−6) = **2880 < 3000** ⇒ **不拆**。
///    ⇒ 同一个病灶、两个模型,门限差 **120** 就是两种结局。⛔ 这说明该改的是形状,
///      而在拿到超线性代价的干净标定之前,**下调门限是同向、可验、且不引入新自由度**的一步。
///
/// ## 人群定价(`split_scan`,六条臂同一次跑;「买到什么」与「付出什么」是同一次跑的两列)
///
/// | 臂 | 3000 组数 | 2000 组数 | 落点 ≤69 | 落点 <72 | `|sh|` ≥8 | donor 遍数 |
/// |---|---|---|---|---|---|---|
/// | akiko +7  | 89 | 100 (+11) | **31 → 6** | 77 → 42 | 183 → 143 | 10 → 10 |
/// | 东雪莲 +7 | 82 |  93 (+11) | **11 → 0** | 48 → 34 | 168 → 141 |  8 →  8 |
/// | yuyuko +7 | 87 | 101 (+14) |   1 → 0    | **35 → 6** | 146 → 113 | 10 → **9** |
/// | 三条 t0   | — | **一格不动** | — | — | — | — |
///
/// ⭐ 「只在需要它的地方咬」:三条 +7 臂大幅改善,三条 t0(浅救援)逐字不变;donor 遍数不增。
/// ⛔ 1200 那一档只多付缝、不再买到东西(akiko 6 → 5,而 yuyuko 多 24 组)。
/// ⛔⛔ **以上全部成立,而这一刀仍然被回滚了** —— 因为这张表里的每一列都是「落点」,
/// 没有一列是「接缝」,而耳判称的是后者。**一张漏了一列的定价表看起来和完整的一模一样。**
///
/// ## ⛔ 阴性对照(缺了它这就是一次「多拆了所以更好」的自证)
/// 降门限少掉的覆盖音**必须全是乘客**。六条臂实测:**死音一个不少**
/// (akiko 318→318 · 东雪莲 249→249 · yuyuko 249→249),少掉的 10 个全是 akiko 的
/// **MIDI 75**(usable 顶 76 ⇒ 模型本来就唱得动),而那 10 个正好就是「落点 65 的那 10 个」。
/// ⇒ 卸乘客正是这一刀的目的,由 [`tests::splitting_never_changes_which_notes_are_rescued`] 钉住。
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
    minimal_rescue_shift_capped_tie(
        dead,
        all,
        range,
        landing,
        ratio_two_cap,
        parse_landing_tie_thin(std::env::var("UTAI_RANGE_LANDING_TIE").ok().as_deref()),
        true,
    )
}

/// ⭐ S162 —— [`minimal_rescue_shift_capped`],但 `thinner` 那一层由**参数**给。
///
/// ⛔ 存在的理由不是好看:落点的**候选**必须能在**不读进程环境**的前提下算出来 ——
/// 渲染时要同时拿到「今天的 pick」与「关掉 `thinner` 会选到的那个」,
/// 而两者若都走 env,就没法在同一次调用里分别求。
/// (与 `dead_only_plan_with` / `apply_dead_only_windows_with` 同一条规矩。)
#[allow(clippy::too_many_arguments)]
fn minimal_rescue_shift_capped_tie(
    dead: &[i64],
    all: &[i64],
    range: &SpeakerRange,
    landing: Option<i64>,
    ratio_two_cap: i64,
    tie_thin: bool,
    // ⭐ S162 —— 乘客 damage 那一把。`false` ⇒ 直接在 `t1`(只过了**死音** damage 的集合)里取最浅。
    // ⛔ 这一层才是挡住 akiko「ぴゃ」的那一层:S157 已经查实「挡住 ぴゃ 的是 `worst()` 对
    //    **死音 ∪ 乘客**取 max(两个乘客落在 MIDI 54/61)」—— 不是 `thinner`。
    //    我第一版只关了 `thinner`,于是那一组**根本没生成候选**(实测:覆盖它的窗只有一个)。
    tie_passenger: bool,
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
    // S162 —— 这一层由参数控(出厂 `true` = 不跳)。
    //   见 [`parse_landing_tie_thin`] 的 doc:akiko 的 ぴゃ 是它的第一个决定性反例。
    //   ⭐ 传 `false` 得到的就是**落点候选**(`t1` 里最浅的那个)。
    let t2: Vec<i64> = if tie_thin {
        let best_thin = t1.iter().copied().map(thinner).fold(f32::INFINITY, f32::min);
        t1.into_iter().filter(|&s| thinner(s) <= best_thin + LANDING_THIN_EPS).collect()
    } else {
        t1
    };
    // ⭐ S162 —— 候选那一路在这里收工:`t1` 里最浅的那个(乘客那一把不参与)。
    if !tie_passenger {
        return t2.into_iter().min_by_key(|s| s.abs());
    }
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

/// ⚙ 出厂默认 = `true`(= S151/S157 那一层照旧 = **逐位同今天**)。
/// `UTAI_RANGE_LANDING_TIE=shallow` 把它关掉 ⇒ 落点平局时**直接取最浅**。
///
/// ## ⛔ 它存在的理由:S162 拿到了这一层的第一个**决定性反例**
/// 炉心融解 `[685]ぴゃ` MIDI 83(+7 ⇒ 90)× **akiko**,只变这一组的落点:
/// **−13(出厂)落 77 ⇒ 相对邻近 12 音 −12.6 dB = 用户说的「没声」**;
/// **−12 落 78 ⇒ −0.2 dB = 完全正常,而且梳深 34.3 → 43.6(谐波更清晰)**。
/// 而 −12/−13/−14 的**死音 damage 全是 0**(打平)⇒ 由 `low_ratio` 决定:
/// 77 = 0.276 < 78 = 0.388(差 0.112 > `LANDING_THIN_EPS` 0.04)⇒ **78 被踢出**。
/// ⛔ 扫描在 77 上四列全健康(err 1¢ / voiced 1.00 / rms −1.4 / low_ratio 0.276)——
/// ⭐ **扫描测的是 400 ms 的稳态「あ」,它结构上看不见「1440 ms 的 /pʲa/ 在那一格唱不出来」。**
///
/// ## ⛔⛔ 为什么**没有**据此直接翻掉那一层
/// S151/S157 有**两把独立尺子 + Fisher p = 0.0026**(落点 77-79 的 donor 音内包络起伏率
/// **30%** vs ≤76 的 **7%**;同位移下落点 >73 的元音塌 −2.74 dB vs ≤73 的 −0.95,p = 4.2e-10),
/// 而且它把薄区死音占比压掉一半以上(akiko 30% → 12% · yachiyo 34% → 3%)。
/// **一个音推翻不了它** —— 这个旋钮是为了让它**被人群级数据裁一次**,不是为了关掉它。
fn parse_landing_tie_thin(v: Option<&str>) -> bool {
    !matches!(v.map(str::trim), Some("shallow"))
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
/// ⭐⭐⭐⭐⭐ S165 —— **RVC 的 `f0_to_coarse` 能表达的最高基频(Hz)**。
///
/// # ⛔ 它为什么是一条**独立于 `slot_singable` 的死因**
/// `rvc::f0_to_coarse` 把 f0 压进 **256 档 mel 表**,而表的上界写死在
/// `f0_mel_max = 1127·ln(1 + 1100/700)`(那边的判据钉着 `f0_to_coarse(1100.0) == 255`)。
/// ⇒ **任何 > 1100 Hz 的 f0 都被 clamp 成同一个 255** —— 模型收到的音高信息是「顶格」,
///   它**不知道该唱 1422 还是 1600**。
///
/// ⚠⚠ 这与 [`SpeakerRange::slot_singable`] **量的不是一件事**:
/// 1422 Hz 在 `usable` 之内 ⇒ `slot_singable` 说「唱得动」⇒ 救援不触发 ⇒ 原样喂进模型 ⇒ **吐噪声**。
///
/// # ⭐ 频谱证据(用户 2026-08-29 点名的 4:04.740,全曲最长的一处破音,980 ms)
/// * **源**:1400 / 2800 / 4200 / 5700 / 7100 五条**又细又亮的谐波线**笔直贯穿整整 1 秒;
/// * **cover**:244.75 起**所有谐波线消失**,只剩两团又宽又糊的能量带 + 噪声。
/// ⇒ **不是抬错八度**(八度修复开/关两条臂都糊),是**谐波结构整个没了**。
///
/// # ⭐⭐ 为什么谱面轨没有这一族(用户当场纠正过我:**两条轨都是 RVC**)
/// 两边**共用** `f0_to_coarse`(`score2svc.rs` 与 `rvc.rs` 各一个调用点),
/// 差别在**谱面轨的救援把高音整体降下去渲**:实测各 donor 遍喂进模型的 `note_hz`
/// 超 1100 Hz 的帧 —— `shift −8` 及更深**全是 0.0 %**(最深 −23 时 max 只有 440 Hz),
/// 而 **cover 修复前 2.7 %、八度修复之后 10.0 %**(⛔ 我那把刀**把更多帧推过了上限**)。
///
/// # ⚠ 为什么不直接抬这个上界
/// 256 档是**训练时定死的**;抬上界 ⇒ 与训练分布对不上 ⇒ 整个模型的音高响应会漂。
/// ⇒ 正确的做法是**走已有的救援通路**(降八度渲染 + PSOLA 移回),与谱面轨同构。
///
/// ⚙ 规模(实测这份素材):受影响 **10.0 % 的浊音帧、连成 58 段**,
/// 长度 p50 0.14 s / p90 0.90 s / max 2.40 s;≥0.5 s 的 8 段。
/// 降八度之后 max 正好落回 **1100**。
pub(crate) const RVC_COARSE_MAX_HZ: f32 = 1100.0;

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

/// ⭐ S163 —— 一个音在**输出时间轴**上的跨度 + 它的目标基频。
///
/// ⛔ 为什么是一个类型而不是三个平行数组 / 一个元组:今天有**三处**要按音符切这条时间轴
/// (`match_rescued_note_levels` / `match_phrase_group_levels` / 落点选法的逐音打分),
/// 而 S162 已经在「按**组下标**平行传候选」上栽过一次张冠李戴。⇒ 一个类型,一个构造器。
///
/// `hz` = **目标**基频(= `(note_num + transpose)` 那一格的等程律频率)。
/// ⛔ 它只许用于**同一个音的两个候选之间**的比较 —— 谐波序号尺子跨音区不可比(S162 栽过三次)。
/// `0.0` = 不知道 ⇒ 谐波那一轴对这个音**弃权**。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoteSpan {
    /// 起始帧(与 [`DeadJob`] 同一条帧轴)。
    pub start: i64,
    /// 帧数。
    pub frames: i64,
    /// 是不是**唱音**(`note_num > 0`;休止/换气不是)。
    pub sung: bool,
    /// 目标基频(Hz);0 = 未知。
    pub hz: f32,
    /// ⭐ S163 —— **这个音是不是上一个音的延续**(同一个音节被写成了好几个音符)。
    /// 用户 2026-08-26:「我实在受不了那种一个长音三个听感,或者长音割裂了」。
    /// 它只影响**拼接**:延续处的交叉淡化拉长(见 [`tied_xfade_ms`]);音节变化处一个字节不动。
    pub tied: bool,
}

/// 从谱面的「音高 + 帧数」铺出 [`NoteSpan`] 表。⛔ **三个调用点共用这一个构造器** ——
/// 它们此前各自写了一遍 `acc += frames.max(0)`,而那正是「同一份地图抄三遍」的形状。
///
/// `transpose` 与 `dead_only_plan_with` 里的 `eff` 同一条:`(n + transpose).clamp(1, 127)`。
pub fn note_spans(note_nums: &[i64], frames: &[i64], transpose: i64) -> Vec<NoteSpan> {
    note_spans_tied(note_nums, frames, transpose, &[])
}

/// [`note_spans`],外加**歌词**(用来认「这个音是不是上一个音的延续」)。
/// `lyrics` 为空 ⇒ `tied` 全 false ⇒ 拼接层逐位回到今天。
///
/// ⛔ 判定只用**歌词相同**或**延音记号**,不用音高 —— 实测炉心融解结尾那个长音
/// (`[796..=802]` 全是「あ」)音高是 71/73/75/76/83/81,按音高认会全部漏掉。
pub fn note_spans_tied(
    note_nums: &[i64],
    frames: &[i64],
    transpose: i64,
    lyrics: &[String],
) -> Vec<NoteSpan> {
    note_spans_tied_with(note_nums, frames, transpose, lyrics, vowel_tied())
}

/// ⚙ 出厂默认 = **false(关)** —— `UTAI_RANGE_VOWEL_TIED=1` 打开。
/// 机理见 [`note_spans_tied_with`] 里 `vowel_onset` 的 doc。
pub fn vowel_tied() -> bool {
    matches!(
        std::env::var("UTAI_RANGE_VOWEL_TIED").ok().as_deref().map(str::trim),
        Some("1") | Some("on") | Some("true") | Some("yes")
    )
}

/// [`note_spans_tied`] 的纯函数版 —— ⛔ 判据不许读进程环境(S151 笔1)。
pub fn note_spans_tied_with(
    note_nums: &[i64],
    frames: &[i64],
    transpose: i64,
    lyrics: &[String],
    vowel_tied: bool,
) -> Vec<NoteSpan> {
    const SUSTAIN: &[&str] = &["-", "+", "ー", "〜", "\u{301c}"];
    /// ⭐⭐⭐ S163 —— **同歌词还不够，音高也要连得上**。
    ///
    /// 用户 2026-08-27 报的 yuyuko **4:36.151-4:36.439**：`[792][793][794]` 歌词都是「あ」
    /// 而 midi 是 **90 / 85 / 83** —— 那是**旋律**不是延续音。`tied` 只看歌词时把它判成延续，
    /// 于是拼接器给了 120 ms 交叉淡化，**把两个差 7 个半音的音色混在一起 120 毫秒** ⇒ 炸。
    /// ⛔ 硬切至少干脆；那 120 ms 的渐变把它拖成「滑音 + 双影」，**比不治更糟**。
    ///
    /// ⚠ 2 个半音的余量：真正的延续音在谱面上可能带滑音/装饰而微调音高，但不会跳七度。
    /// ⭐ **延音记号不受这条约束** —— 那是谱面明写的「接着上一个音唱」。
    const TIED_MAX_ST: i64 = 2;

    /// ⭐⭐⭐ S165 —— **下一个音是纯元音开头时,音节边界也算「接得上」**。
    ///
    /// # 病
    /// 用户 2026-08-28 点名 yuyuko × 炉心 **4:29.249-4:29.300**:「接缝挺明显」「像哑了一下」,
    /// 而同一批修好的 4:36 他说「基本上真好了」。两处的唯一结构差别是**淡入宽度**:
    /// 4:36 是 `[793]あ → [794]あ`(歌词相同)⇒ 判 tied ⇒ **120 ms**;
    /// 4:29 是 `[784]な → [785]あ`(歌词不同)⇒ 音节边界 ⇒ 只有 **10 ms** ⇒ 边界几乎是跳变。
    ///
    /// # 为什么放宽是安全的
    /// [`tied_xfade_ms`] 不拉长音节边界的**理由**写得很清楚:
    /// 「音节边界上有**辅音与起音**,拉长淡化会把它糊掉」。
    /// 而 `あいうえお` 开头的音**没有辅音要保护** ⇒ 那条理由在这一族上不成立。
    /// ⛔ 而真正危险的那一族(4:36 事故:`あ(90) → あ(85)` 差 7 个半音被混 120 ms)
    ///    是被 `TIED_MAX_ST` 挡住的,**这里一个字节都不动它** ——
    ///    放宽的只是「歌词必须相同」,音高必须连得上的约束原样保留。
    ///    实测 4:29 的 `な(85) → あ(83)` 差 **2 个半音,正好在约束之内**。
    ///
    /// ⚠ 只认**平假名/片假名的五个纯元音**。拗音(ゃゅょ)、促音(っ)、拨音(ん)都不是元音开头。
    fn vowel_onset(ly: &str) -> bool {
        matches!(
            ly.chars().next(),
            Some('あ' | 'い' | 'う' | 'え' | 'お' | 'ア' | 'イ' | 'ウ' | 'エ' | 'オ')
        )
    }

    let mut acc = 0i64;
    let mut prev: Option<(&str, i64)> = None;
    note_nums
        .iter()
        .zip(frames.iter())
        .enumerate()
        .map(|(k, (&n, &f))| {
            let start = acc;
            let fr = f.max(0);
            acc += fr;
            let sung = n > 0;
            let ly = lyrics.get(k).map(String::as_str).unwrap_or("");
            let tied = sung
                && !ly.is_empty()
                && (SUSTAIN.contains(&ly)
                    || prev.is_some_and(|(p, pn)| {
                        // ⭐ S165 —— 歌词相同 **或** 这个音是纯元音开头(见 `vowel_onset`);
                        //    ⛔ 音高必须连得上这一条(`TIED_MAX_ST`)对两者都成立,一个字节没动。
                        let _ = p;
                        (p == ly || (vowel_tied && vowel_onset(ly)))
                            && (n - pn).abs() <= TIED_MAX_ST
                    }));
            if sung {
                prev = Some((ly, n));
            } else {
                prev = None; // ⛔ 隔着休止就不是同一个长音了
            }
            NoteSpan {
                start,
                frames: fr,
                sung,
                hz: if sung {
                    let m = (n + transpose).clamp(1, 127);
                    440.0 * 2f32.powf((m - 69) as f32 / 12.0)
                } else {
                    0.0
                },
                tied,
            }
        })
        .collect()
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
    cover_dead_plan_with(f0_hz, fps, range, CoverGrouping::today())
}

/// S160e —— cover **拆组**的收益门槛(**ms·半音**,与谱面轨 [`SPLIT_MIN_COST_DEFAULT`] 同一种货币)。
///
/// ## 它治的是什么(用户 2026-08-24 点名的 4:34.544「失声」)
/// 那一处 **100% 在救援段内、附近没有段边**;音[789]「か」·86 / [790]「い」·84 落在一个
/// **5.15 s / 深度 −17** 的大段里,而它自己只需要 **−7 度** ⇒ **陪绑过深 10 度**。
/// 同一处的剂量测试(只换深度,阴性对照在同一句):深度 −11 ⇒ 该段电平比源低 **5.3 dB**;
/// 深度 −17 ⇒ 低 **11.4-11.7 dB**;而紧邻前面**响**的半句在两条臂上都是 **+0.5 / +1.1 dB**。
/// ⇒ **越深掉得越多,而且只掉在弱音上。**
///
/// ## 为什么只在【清音连段的正中】下刀
/// S159k 的结论:cover 的边界落在长音中间会造出音色台阶(那正是用户听到的「炸」)。
/// ⇒ 拆组**只许**把缝放进清音,而且放在清音连段的**正中**(离两侧唱音都最远)。
/// 这样拆一刀几乎不付缝的钱,买到的是「浅的那半不再被深的那半拖下去」。
///
/// ## 定价(货币 = ms·半音 = 少染多少帧 × 少深多少度)
/// 3000 相当于「一个 300 ms 的音少被拖 10 度」。
/// 与谱面轨 `SPLIT_MIN_COST_DEFAULT` 同名同币但**不是同一个常量**:
/// 两条车道的帧率、分组方式、缝的代价都不一样。`0` = 关掉拆组。
///
/// ## ⛔⛔ 出厂 = 0(关)—— 它在唯一一把能量到这一族的尺子上是【净负】
/// 同一二进制、同一份 f0、gap150,只差这一个旋钮:
/// | | Δper>0.3 的 200 ms 窗 | >0.2 |
/// |---|---|---|
/// | 拆组关(`v3g150n`) | **9** | 84 |
/// | 拆组开 3000(`v3g150s`) | **55** | 143 |
/// ⇒ 陪绑过深确实降了(p50 6.9 → 4.7,总浪费 −22%),**但多出来的 13 段 = 26 条缝
/// = 26 次 donor 切换**,在输出上花掉的比省下的多。⚠ 那把尺子**看不见**深救援的染色,
/// 所以它对拆组的收益一侧是**偏心**的;即便如此,亏的那一侧已经大到 6 倍。
/// ⇒ **旋钮留着(`UTAI_COVER_SPLIT_GAIN`),出厂关**,判据与读数留在原地。
/// ⛔ 别再原样重造:要动这条,先解决「缝本身要花钱」——例如拆完之后让相邻两段共用一次 donor,
/// 或者改成**逐帧深度斜坡**(今天 `apply_inverse` 只吃一个常数位移)。
const COVER_SPLIT_MIN_GAIN: f32 = 0.0;

/// S160e —— 拆出来的每一半至少这么长(毫秒)。
/// 它不是品味:拆出太短的一半 = 又造回「碎片 + 两条边」,而碎片正是 S160c 那一族缺陷的来源。
const COVER_SPLIT_MIN_PART_MS: f32 = 250.0;

/// S160c —— [`cover_dead_plan`] 的分组两个门槛做成**参数**。
///
/// ⛔ **为什么是参数不是 env**(S151 笔1 / `dead_only_plan_with` 同一条):走进程环境的判据
/// 会随着机器上导出了什么静默改答案,而且**关不掉** —— 判据必须能把旋钮钉在某个值上。
///
/// ## ⛔ 它为什么存在(S160c,用户 2026-08-24 点名的两处)
/// 用户报「そしたら」的「そし」与「気がして」的「し」仍然炸,而**全曲最高的音反而不炸**。
/// 逐处量出来:那几处 **死浊帧桥接后最长一组只有 120-140 ms**(门槛 250),
/// 而把它们切碎的正是两侧 **50-80 ms 的清擦音**(/s/ /ɕ/)—— `GAP_TOL_MS = 30 ms` 桥不过去。
/// 对照:「おもう」的「う」(MIDI 91,全曲最高)那一段**两侧没有清音**,于是与邻音并成一组、
/// 够到 250 ms、被救,听感正常。
/// ⇒ **缺陷与音高反相关**:被 /s/ /ɕ/ 夹住的短音节掉出救援,模型被留在那里硬唱高出上限 4-7 度。
/// ⚠ 这两个常量**只在这个函数里被用**(谱面轨的 `dead_only_plan_with` 走自己的逻辑)
/// ⇒ 动它结构上够不着谱面轨(用户 2026-08-21:「任何修法不许退化谱面轨」)。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoverGrouping {
    /// 桥接:死帧组之间最多隔这么久还算同一组(毫秒)。
    pub gap_tol_ms: f32,
    /// 门槛:一组里的**浊死帧**至少要这么久才成区(毫秒)。
    pub min_violation_ms: f32,
    /// S160e —— **拆组**收益门槛(ms·半音)。0 = 关。见 [`COVER_SPLIT_MIN_GAIN`]。
    pub split_gain: f32,
    /// S160e —— 拆出来的每一半至少这么长(毫秒)。
    pub split_min_part_ms: f32,
}

impl CoverGrouping {
    /// 今天出厂的那一对,**外加两个探针旋钮**(与 `UTAI_COVER_PHRASE_GAP_MS` 同一种形状):
    /// `UTAI_COVER_GAP_TOL_MS` / `UTAI_COVER_MIN_VIOLATION_MS`。不设 ⇒ 出厂常量 ⇒ 逐位不变。
    /// ⛔ **判据不许走这条**:要把门槛钉在某个值上,用 [`CoverGrouping::new`] + [`cover_dead_plan_with`]
    /// (S151 笔1:走 env 的判据会随着机器上导出了什么静默改答案,而且关不掉)。
    pub fn today() -> Self {
        let ev = |k: &str, d: f32| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.trim().parse::<f32>().ok())
                .filter(|x| x.is_finite() && *x >= 0.0 && *x <= 5000.0)
                .unwrap_or(d)
        };
        Self {
            gap_tol_ms: ev("UTAI_COVER_GAP_TOL_MS", GAP_TOL_MS),
            min_violation_ms: ev("UTAI_COVER_MIN_VIOLATION_MS", MIN_VIOLATION_MS),
            // ⛔ 收益的单位是 **ms·半音**,不是毫秒 —— 别套上面那个 [0,5000] 的范围
            //   (第一版套了,于是 6000/12000 两条臂静默变成 3000,而**臂的标签在撒谎**;
            //    是「把实际生效值打出来」那一行抓到的)。
            split_gain: std::env::var("UTAI_COVER_SPLIT_GAIN")
                .ok()
                .and_then(|v| v.trim().parse::<f32>().ok())
                .filter(|x| x.is_finite() && *x >= 0.0 && *x <= 1.0e7)
                .unwrap_or(COVER_SPLIT_MIN_GAIN),
            split_min_part_ms: ev("UTAI_COVER_SPLIT_MIN_PART_MS", COVER_SPLIT_MIN_PART_MS),
        }
    }
    pub fn new(gap_tol_ms: f32, min_violation_ms: f32) -> Self {
        Self {
            gap_tol_ms,
            min_violation_ms,
            split_gain: COVER_SPLIT_MIN_GAIN,
            split_min_part_ms: COVER_SPLIT_MIN_PART_MS,
        }
    }
    /// 判据用:把四个旋钮全部钉死(⛔ 判据不许读进程环境)。
    pub fn pinned(
        gap_tol_ms: f32,
        min_violation_ms: f32,
        split_gain: f32,
        split_min_part_ms: f32,
    ) -> Self {
        Self { gap_tol_ms, min_violation_ms, split_gain, split_min_part_ms }
    }
}

pub fn cover_dead_plan_with(
    f0_hz: &[f32],
    fps: f32,
    range: &SpeakerRange,
    grouping: CoverGrouping,
) -> (Vec<DeadJob>, Vec<(i64, i64)>) {
    let min_run = frames_for(grouping.min_violation_ms, fps);
    let gap_tol = frames_for(grouping.gap_tol_ms, fps) + 1;
    let mut idx: Vec<usize> = Vec::new();
    let mut midi: Vec<f32> = Vec::new();
    for (i, &v) in f0_hz.iter().enumerate() {
        if v > 0.0 {
            idx.push(i);
            midi.push(69.0 + 12.0 * (v / 440.0).log2());
        }
    }
    let midi = median5(&midi); // 倍频闪烁卫生,与旧决策同款
    // ⭐⭐⭐⭐ S165 —— **两条死因**:
    //   ⑴ 模型**唱不动**(`slot_singable`,S85 起就有的那条);
    //   ⑵ ⭐ f0 **超出 `f0_to_coarse` 能表达的上限**([`RVC_COARSE_MAX_HZ`])——
    //      那时模型「唱得动」但**收到的音高是顶格的 255**,吐出来的是噪声。
    //   ⛔ ⑵ 是 S165 新加的,而它**不能**用 ⑴ 表达:1422 Hz 在 `usable` 之内。
    //   ⚠ 对谱面轨零影响:它的 donor 遍喂进模型的 f0 超 1100 的帧实测**全是 0.0 %**。
    let coarse_max_midi = 69.0 + 12.0 * (RVC_COARSE_MAX_HZ / 440.0).log2();
    let is_dead = |m: f32| !range.slot_singable(m.round() as i64) || m > coarse_max_midi;
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
        // ⛔⛔ S165 —— **这里必须与上面的 `is_dead` 用【同一条】判据**。
        //    第一版只写了 `!slot_singable`,而新加的「超 `RVC_COARSE_MAX_HZ`」那一支没跟上
        //    ⇒ 超界的组 `dead` 为空 ⇒ 算不出位移 ⇒ **整组被静默丢掉**,判据当场红。
        //    这就是本项目反复栽的「两处填充点漏一处」——判据 `cover_treats_f0_above_the_
        //    rvc_coarse_ceiling_as_dead` 现在钉住它。
        let dead: Vec<i64> = pitches
            .iter()
            .copied()
            .filter(|&p| !range.slot_singable(p) || (p as f32) > coarse_max_midi)
            .collect();
        (pitches, dead)
    };
    // ⛔ 深度门与「无解」是**两件事**,报法必须分开(S129 铁律:一条红要能被归因)。
    let mut push = |s: i64, a: usize, b: usize| {
        if s.abs() >= COVER_MIN_RESCUE_DEPTH {
            out.push(DeadJob { shift: s, start: a as i64, end: (b + 1) as i64 });
        }
    };
    // S160e —— 拆组(见 COVER_SPLIT_MIN_GAIN)。只在清音连段的正中下刀。
    let min_part = frames_for(grouping.split_min_part_ms, fps).max(1) as usize;
    let cut_points = |a: usize, b: usize| -> Vec<usize> {
        // 候选切点 = 每条 >= 3 帧(30 ms)清音连段的正中。不许在唱音上切(S159k)。
        let mut out = Vec::new();
        let mut run = 0usize;
        for i in a..=b {
            if !voiced(i) {
                run += 1;
            } else {
                if run >= 3 {
                    out.push(i - 1 - run / 2);
                }
                run = 0;
            }
        }
        if run >= 3 {
            out.push(b - run / 2);
        }
        out
    };
    let ms_per = 1000.0 / fps;
    let top = range.usable.1.round() as i64;
    let paint_cost = |a: usize, b: usize, sh: i64| -> f32 {
        // 一段的「染色账」= 每个浊帧被【多】拖的度数 x 它的时长(ms·半音)。
        idx.iter()
            .zip(midi.iter())
            .filter(|(&i, _)| i >= a && i <= b)
            .map(|(_, &m)| {
                let need = (m.round() as i64 - top).max(0);
                ((sh.abs() - need).max(0) as f32) * ms_per
            })
            .sum()
    };

    // ⭐⭐⭐⭐ S165 —— **把位移再压深到 f0 落回 `RVC_COARSE_MAX_HZ` 以下**。
    //
    // ⛔ 为什么不能改 [`minimal_rescue_shift`]:那个函数**谱面轨共用**,它的语义是
    //   「把唱不动的音搬进 `slot_singable`」。而 coarse 上界是 **RVC 编码器的表达上限**,
    //   与「唱不动」是两件事(1422 Hz 在 `usable` 之内,`slot_singable` 说唱得动)。
    //   ⇒ 在 **cover 这一侧**把它算完,不动共用函数。
    // ⚠ 取 `max`:两条死因谁要得深就听谁的。
    let deepen_for_coarse = |pitches: &[i64], s: i64| -> i64 {
        let hi = pitches.iter().copied().max().unwrap_or(0) as f32;
        if hi <= coarse_max_midi {
            return s;
        }
        // 需要往下多少个半音才能让最高音落到上界以下(向上取整)
        let need = (hi - coarse_max_midi).ceil() as i64;
        s.min(-need)
    };
    for (ea, eb, orig) in spans {
        let (pitches, dead) = collect(ea, eb);
        // ⭐ 超界的组按 coarse 上界把位移**压深**。
        //
        // ⛔⛔ **`None` 必须原样传下去,不许救活它。**第一版写成
        //   `(None, true) => Some(deepen_for_coarse(&pitches, 0))`,当场被两条既有判据抓住
        //   (`a_cover_merge_never_swallows_material_no_predicate_has_looked_at` 与
        //    `cover_plan_counts_unfixable_regions_loudly`)。
        //   它们的夹具是「死亡高潮里混着够不着的低音」(高音 midi 88 + 低音 midi 30 同区):
        //   `None` 的含义**不是**「没查过」,而是「**降下去会把低音乘客拖到唱不出来的地方**」
        //   —— midi 30 再降就是 18。⇒ 那时正确的行为是**报无解**,不是硬渲一段垃圾。
        //   ⭐ 这两条判据是 S85d 那次「拖拽守卫」留下的,今天正好接住了我。
        let shift_opt = minimal_rescue_shift(&dead, &pitches, range, None)
            .map(|s| deepen_for_coarse(&pitches, s));
        match shift_opt {
            Some(s) => {
                let mut parts: Vec<(usize, usize, i64)> = Vec::new();
                if grouping.split_gain > 0.0 {
                    let shift_of = |x: usize, y: usize| -> Option<i64> {
                        let (p, d) = collect(x, y);
                        // 拆出来的这一半里没有死音 => 它不需要被救;但它仍然在窗里(拼接器按窗贴),
                        // 给它 0 会让拼接器贴一段没移调的 donor —— 那不是「不救」,是贴错东西。
                        // => 判它不可拆。
                        if d.is_empty() {
                            return None;
                        }
                        minimal_rescue_shift(&d, &p, range, None)
                    };
                    // 迭代式拆(不用递归闭包):每轮挑收益最大的一刀,直到没有一刀过门槛。
                    parts.push((ea, eb, s));
                    for _round in 0..4 {
                        let mut best: Option<(usize, usize, f32, usize, i64, i64)> = None;
                        for (k, &(a, b, sh)) in parts.iter().enumerate() {
                            if b < a + 2 * min_part {
                                continue;
                            }
                            let c0 = paint_cost(a, b, sh);
                            for p in cut_points(a, b) {
                                if p < a + min_part || p + min_part > b {
                                    continue;
                                }
                                let (Some(sl), Some(sr)) = (shift_of(a, p), shift_of(p + 1, b))
                                else {
                                    continue;
                                };
                                let g = c0 - paint_cost(a, p, sl) - paint_cost(p + 1, b, sr);
                                if best.map_or(true, |(_, _, bg, _, _, _)| g > bg) {
                                    best = Some((k, p, g, p, sl, sr));
                                }
                            }
                        }
                        // 「臂开着但没做事」必须可查(S129 铁律):每一轮都报最好的那一刀与它的收益。
                        // ⛔ debug 级 —— 一首歌几十段 x 四轮,info 会把审计行淹掉;
                        //    但它**必须存在**:S160e 第一版就是靠它发现「旋钮根本没送到」的
                        //    (两条臂跑出逐字相同的计划,而臂的标签在撒谎)。
                        tracing::debug!(
                            "range-extend(cover/split): {:.2}s..{:.2}s round {_round}: best {:?}",
                            ea as f32 / fps, eb as f32 / fps,
                            best.map(|(k, p, g, _, sl, sr)| (k, p as f32 / fps, g, sl, sr))
                        );
                        match best {
                            Some((k, p, g, _, sl, sr)) if g >= grouping.split_gain => {
                                let (a, b, _) = parts[k];
                                parts.splice(k..=k, [(a, p, sl), (p + 1, b, sr)]);
                            }
                            _ => break,
                        }
                    }
                    if parts.len() > 1 {
                        tracing::info!(
                            "range-extend(cover): split {:.2}s..{:.2}s ({s:+} st) into {} part(s): {:?}",
                            ea as f32 / fps,
                            eb as f32 / fps,
                            parts.len(),
                            parts
                                .iter()
                                .map(|&(x, y, t)| (x as f32 / fps, y as f32 / fps, t))
                                .collect::<Vec<_>>()
                        );
                    }
                } else {
                    parts.push((ea, eb, s));
                }
                for (x, y, t) in parts {
                    push(t, x, y);
                }
            }
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
/// [`dead_group_windows`] 的**合并之前**那一半 —— **一组一窗、按序**。
/// ⛔ 抽出来的唯一理由:[`dead_group_windows_alts`] 必须知道「一个**合并后**的窗覆盖了哪些组」,
/// 而那要求两边用**同一条**式子(两处各写一遍 = 一处改了另一处不改)。
fn dead_group_windows_raw(
    cum: &[i64],
    note_nums: &[i64],
    plan: &[DeadGroup],
    range: &SpeakerRange,
    transpose: i64,
) -> Vec<DeadJob> {
    // ⭐⭐⭐ S163 —— **护栏不许伸进一个【在本遍唱不动】的音**(用户 2026-08-26 点名的
    // 「嗓子里卡着痰」的根因链的后半截)。
    //
    // donor 的每一遍是**按那一遍的位移唱整条谱**的 ⇒ 窗边紧邻的那个音如果在**这一遍**
    // 掉出音域,模型在那里**直接塌掉**(S159zk 实测:那样的 24 个音比组内音低 **−11.4 dB**),
    // 而护栏的全部意义是「把 10 ms 交叉淡化压在那个乘客身上」—— 压在一段塌陷上等于没压。
    //
    // ## 判据(纯结构,不需要读 `usable`)
    // 邻音**属于另一个 `DeadGroup`、而且那个组的 `shift` 与本组不同** ⇒ 收窄该侧。
    // ⭐ 它与 S159zk 想拦的那一族**同延**:
    // * 二级拆组的断点**只许落在死音之间** ⇒ 拆出来的边**两侧都是死音**;
    // * 两侧 `shift` 相同的边早被 [`merge_same_shift_across_rests`] 合并 ⇒ 剩下的边 shift 必不同;
    // * 拆组**之前**就有的 67 条边,邻音是**唱得动的乘客**(不在任何组里)⇒ 判据一条都不碰,
    //   所以「今天的计划逐帧不变」那条不变量对它们仍然成立。
    //
    // ⛔ 与「`GUARD_FRAMES` 收到 0 差 7.6 dB」(S163)**不冲突**:那一条是在**护栏区材料健康**
    // 时量的。这里护栏区本身就是塌的,伸进去是把塌陷拌进交界。
    // ⛔⛔ **`slot_reachable`,不是 `slot_singable`** —— 这问的是「**模型**在这一格上发不发得出
    // 声」,所以必须读**扫描**的边界,永远不许读用户那条救援线(与拆组里那条同一个理由、
    // 同一次学费 S146f:用 `slot_singable` 会让「用户把音域上限调低」变成「护栏收得更多」)。
    //
    // ⛔ 第一版写成纯结构判据「邻音属于另一个组且 shift 不同」,**被自家的闸抓住了**:
    // `a_short_rest_keeps_both_windows_out_of_it` 的夹具 ⑶ 两组 −6/−7 ⇒ 落点 79/78,
    // **护栏区是健康的**,而那个判据照收不误 —— 正好撞上 S163 量过的
    // 「无休止跨组缝的重叠从 80 ms 收到 0,瞬变中位差 **7.60 dB**(好 0 / 坏 10)」。
    // ⇒ 收窄的**唯一**条件只能是「那个邻音在**本遍**唱不动」。
    let unreachable_here = |k: usize, shift: i64| -> bool {
        match note_nums.get(k).copied() {
            Some(x) if x > 0 => !range.slot_reachable(x + transpose + shift),
            _ => false, // 休止 / 越界 ⇒ 护栏伸不进任何唱音,照常
        }
    };
    plan
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
            // S159zw —— 短休止一格都不许伸(见 [`SHORT_REST_NO_EXTEND_FRAMES`])。
            let pre = if gap_prev > 0 && gap_prev <= SHORT_REST_NO_EXTEND_FRAMES {
                0
            } else if gap_prev > 0 {
                REST_PRE_FRAMES.min(gap_prev / 2)
            } else if g.start > 0 && !unreachable_here(g.start - 1, g.shift) {
                GUARD_FRAMES.min((cum[g.start] - cum[g.start - 1]) / 2)
            } else {
                0
            };
            let post = if gap_next > 0 && gap_next <= SHORT_REST_NO_EXTEND_FRAMES {
                0
            } else if gap_next > 0 {
                REST_POST_FRAMES.min(gap_next / 2)
            } else if g.end + 1 < note_nums.len() && !unreachable_here(g.end + 1, g.shift) {
                GUARD_FRAMES.min((cum[g.end + 2] - cum[g.end + 1]) / 2)
            } else {
                0
            };
            DeadJob { shift: g.shift, start: cum[g.start] - pre, end: cum[g.end + 1] + post }
        })
        .collect()
}

/// ⭐⭐ S162 —— [`dead_group_windows`],并把**逐组的落点候选**对齐到**窗**上。
///
/// ## ⛔ 为什么需要它 —— 这是一个正确性 bug 的修法
/// `dead_group_windows` 末尾会 `merge_same_shift_across_rests` **把窗合并**
/// ⇒ **窗表与组表不是一一对应**。我第一版按**组下标**把候选交给渲染层,结果**张冠李戴**:
/// akiko 的「ぴゃ」拿到的候选是 **−3**(一个根本救不了 MIDI 90 的值),
/// 而日志把「拿错候选」与「没有候选」显示成同一种样子,卡了三轮。
///
/// 对齐规则:一个窗覆盖哪些组,就看那些组的候选;
/// ⛔ **只有它们全都相同才带出来**,否则 `None`(合并窗的两半想去不同落点 = 没有单一答案)。
pub fn dead_group_windows_alts(
    note_nums: &[i64],
    frames: &[i64],
    plan: &[DeadGroup],
    plan_alts: &[Option<i64>],
    range: &SpeakerRange,
    transpose: i64,
) -> (Vec<DeadJob>, Vec<Option<i64>>) {
    let jobs = dead_group_windows(note_nums, frames, plan, range, transpose);
    if plan_alts.len() != plan.len() {
        return (jobs, Vec::new());
    }
    let mut cum = Vec::with_capacity(frames.len() + 1);
    let mut acc = 0i64;
    cum.push(0);
    for &f in frames {
        acc += f.max(0);
        cum.push(acc);
    }
    let raw = dead_group_windows_raw(&cum, note_nums, plan, range, transpose);
    let alts = jobs
        .iter()
        .map(|j| {
            let mut seen: Option<Option<i64>> = None;
            for (gi, r) in raw.iter().enumerate() {
                if r.shift != j.shift || r.end <= j.start || r.start >= j.end {
                    continue;
                }
                match seen {
                    None => seen = Some(plan_alts[gi]),
                    Some(v) if v == plan_alts[gi] => {}
                    Some(_) => return None,
                }
            }
            seen.flatten()
        })
        .collect();
    (jobs, alts)
}

pub fn dead_group_windows(
    note_nums: &[i64],
    frames: &[i64],
    plan: &[DeadGroup],
    // ⭐ S163 —— 护栏要知道「邻音在**本遍**唱不唱得动」(见 [`dead_group_windows_raw`])。
    range: &SpeakerRange,
    transpose: i64,
) -> Vec<DeadJob> {
    let mut cum = Vec::with_capacity(frames.len() + 1);
    let mut acc = 0i64;
    cum.push(0);
    for &f in frames {
        acc += f.max(0);
        cum.push(acc);
    }
    let raw = dead_group_windows_raw(&cum, note_nums, plan, range, transpose);
    let merged = merge_same_shift_across_rests(note_nums, plan, raw);
    // S162 —— 薄片闸(出厂 0 = 关 = 逐帧不变)。见 `CLOSE_SLIVER_FRAMES_DEFAULT`。
    close_short_slivers(merged, parse_close_sliver(std::env::var("UTAI_RANGE_CLOSE_SLIVER").ok().as_deref()))
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
/// ⚙ 出厂默认 = 3 帧(60 ms)—— **比这更短的休止,两侧的窗一格都不许伸进去。**
///
/// ## ⭐⭐⭐ 用户 2026-08-23 点名的 3:35.888 那声咔哒
///
/// 那处的休止(音[1153])只有 **3 帧 = 60 ms**,而两侧的窗按下面那两行各伸进 **1 帧**
/// (`pre = min(4, gap/2)` / `post = min(2, gap/2)`,`gap = 3` ⇒ 各 1)
/// ⇒ **3 帧里 2 帧被 donor 占了,中间只剩 1 帧 base**。
///
/// ⇒ 那个休止实际长这样:**20 ms 响 donor / 20 ms 静 base / 20 ms 响 donor** ——
/// **一个休止里两声突起,那就是「咔哒」**。实测(整曲渲染,相对同一处的 base):
/// 前 20 ms **+13.5 dB** · 中段 **−0.2 dB(窗外,与 base 逐字相同)** · 末 20 ms **+12.7 dB**。
///
/// ⚠ 用户还给了一条**深度依赖**的旁证:同一处在落点预算 3 上「很浅、听不出」,在预算 5 上
/// 「清楚」—— 因为落点变深之后 donor 把那个音唱得更饱满(MIDI 74 舒服 vs 78 又薄又闷),
/// 它的收尾进到休止里更响。
///
/// ## ⛔ 为什么不能指望 [`join_rests`] 收拾它
///
/// 那把刀就是为「两个异位移的窗夹一段休止」造的,但它**在这里没有可搜的余地**:
/// 它只在「两种分法真的不同」的区间上比较凹陷,而窗已经把休止吃到只剩 **1 帧(2 个打分格)**。
/// ⇒ 实测(拼接探针,**同一份 base + donor**,只翻 `UTAI_RANGE_JOIN`):
/// **那一处开与关逐字相同**;全曲 43 处这类休止上它也只买到中位 **−1.6 dB**、43 中 10 处变好。
/// ⇒ 先把余地留出来,才轮得到它。
///
/// ## 代价与覆盖面
///
/// 全曲只有 **4 个窗**受影响(鹅妈妈 +7 × 东雪莲,落点预算 5),改动 **0.510 %** 的样本。
/// 实测(拼接探针)那两段回到 base 的电平:前 20 ms −36.6 → **−39.1**(base −39.4)、
/// 末 20 ms −24.9 → **−27.9**(base −27.9)。
///
/// ⚠ 代价:交叉淡化从「休止里」挪回**音的边缘**。⛔ 那正是 `GUARD_FRAMES` 想避开的事 ——
/// 但那条只管 `gap == 0`(**根本没有休止**)的情形;这里有休止,只是太短,
/// 而「淡化压在音边」与「休止里两声突起」相比,后者是用户**真的听见**的那一个。
const SHORT_REST_NO_EXTEND_FRAMES: i64 = 3;

const GUARD_FRAMES: i64 = 2;

/// 有休止可用时，窗的**前**边最多往休止里伸多少帧（1 帧 = 20 ms）。
/// S85 引入时是裸字面量 `4`，commit 里没有依据；S163 把它提成常量以便对照 [`REST_POST_FRAMES`]。
/// ⛔ 实际用的是 `min(这个值, gap/2)` —— 除以 2 保证相邻两个窗各占休止的一半、永不重叠。
const REST_PRE_FRAMES: i64 = 4;

/// ⭐⭐⭐ 有休止可用时，窗的**后**边最多往休止里伸多少帧。**S163 从 2 改成 4。**
///
/// # 缺陷（用户 2026-08-28 报 `1:43.472`「这应该是个音尾但是听起来也奇怪」「从 G3 就一直都有」）
///
/// `donor` 是**降调**渲染的（音落在模型舒适区）⇒ 唱得饱满、**释放慢**；
/// 而 `base` 是原音高（唱不动）⇒ 衰减快。窗尾按谱面 +2 帧切下去时 **donor 还在满电平**，
/// 窗外却是早已衰减的 `base` ⇒ **一个电平台阶**。
/// 实例（`る`，谱面只有 40 ms）：窗尾处 `donor` **+0.5 dB**（相对该音稳态，还在满电平）、
/// 同位置 `base` **−18.5 dB** ⇒ 落差 **+19.0 dB**。
///
/// # 依据（`donor_post` 转储实测，n=62；上界取下一个唱音起点）
/// `donor` 降到「`base` 同期水平 +3 dB」所需时间：p50 **0 ms** · p75 25 · **p90 45** · max 135。
/// 窗尾落差 >8 dB（会听见台阶）的比例随 post 上限：
/// **40 ms ⇒ 13%** · 60 ⇒ 5% · **80 ms ⇒ 3%** · 120 ⇒ 2% · 160 ⇒ 0%。
/// ⇒ **80 ms(4 帧) 是拐点**，而且与 [`REST_PRE_FRAMES`] 对称（前边本来就是 4）。
/// 现状：窗尾落差 >8 dB 的有 **16/62 = 26%**。
///
/// ⛔ 安全性：实际用的是 `min(这个值, gap/2)`，所以休止短时自动收缩、两窗永不重叠；
/// `gap ≤ SHORT_REST_NO_EXTEND_FRAMES` 那条更早的分支仍然返回 0，一个字节不变。
/// ⚠ 这一刀**不解决**「窗尾落在音尾**之前**」那一族（实测 2 个：`に` 早 380 ms、`で` 早 300 ms）。
const REST_POST_FRAMES: i64 = 4;

/// ⚙ 出厂默认 = `0`(= 关 = 窗逐帧不变)。`UTAI_RANGE_CLOSE_SLIVER=<帧>` 打开。
///
/// **薄片闸** —— 相邻两个救援窗之间留下的那一小截 `base`,让两个窗在**休止里**接上。
///
/// ## 症状(S162 量清)
/// 有休止时 `pre = 4.min(gap/2)` / `post = 2.min(gap/2)` ⇒ 休止长于 6 帧时两侧各吃一截,
/// **中间留下 1-5 帧的 base**。鹅妈妈 +7 × yachiyo 上 **36 条,位移不同的 36 条、相同的 0 条**。
/// 用户 2026-08-25 点名的五处竖线里**有四处**落在这种薄片上(3:35.925 / 3:40.858 /
/// 3:49.524 / 3:58.162)。3:49.500 逐段电平(成品):donor −48.2 / **薄片 −31.6** / donor −40.3,
/// 而同处 `base` 是 −32.8 / −31.6 / −27.8(平滑)⇒ **薄片是【没被救的那一版】,比两侧的 donor
/// 响 8-17 dB** —— 40 ms 的「好 / 坏 / 好」。
///
/// ## ⛔ 分量如实登记:**离群点,不是规律**
/// 配对对照(同一位置在 `base` 上的台阶):薄片 n=36 Δ台阶中位 **+3.3 dB** / p90 +12.1;
/// **窗【内部】的音符边界** n=585 中位 **−1.2** / p90 **+6.0**。
/// ⇒ 中位只高 4.5 dB,对照的 p90 就盖过它的中位。**但用户点的 3:49.500 读 +12.8。**
/// ⇒ ⛔ **出厂关,而且【判负】—— 不许把它写成「等耳判」。**
///
/// ## ⛔⛔ S162 当场就用拼接探针判掉了它
///
/// ⭐ 台子:`splice_probe` 吃同一份 `base` + `donor_post`,**只换窗**(把关着那一臂的
/// `windows_frames` 按新规则重算写进 JSON,零代码)⇒ **零渲染噪声**(钩子区:凡是只改
/// 拼接层的 A/B 一律走它,别渲两遍整曲 —— 那会差 85 % 样本)。
/// ⛔ 阴性对照先跑:改动样本 **0.527 %**(预期 ≈ 补进去的 1.60 s = 0.596 %)、
/// **薄片 ±30 ms 之外 max|Δ| = 0.000e+00** ⇒ 测量链干净。
///
/// **而读数是掷硬币**(鹅妈妈 +7 × yachiyo,只算邻域 >−40 dBFS = 真在唱的 n=14):
/// 台阶 关 中位 **33.0 dB** / p90 50.4 → 开 **36.3** / 48.6;**配对 Δ 中位 −0.2 dB,改善的 7/14**。
/// 用户点名的四处逐带竖线分:3:35.925 **+2 带** · 3:40.858 −2 · 3:49.524 **±0** · 3:58.162 −2
/// ⇒ **没有方向**。
///
/// ⇒ 旋钮与判据留在原地,只为把这一次的读数钉在代码里(**别重造**)。
///
/// ⭐ 真正的线索在别处:S162 的三层分解读出 **拼接那一层对这一族的贡献是 −0.00 dB**
/// (n=247,阴性对照 −0.01)⇒ **竖线的主体不在拼接层,在解码层。**
///
/// ⛔ 只补**正的**缝;负数 = 两窗已经重叠,那是 [`GUARD_FRAMES`] 有意做的,一个字不许动。
const CLOSE_SLIVER_FRAMES_DEFAULT: i64 = 0;

/// ⚙ 出厂默认 = 0(= 关 = 窗逐帧不变)。账与机理在 [`CLOSE_SLIVER_FRAMES_DEFAULT`] 的 doc 上。
/// `UTAI_RANGE_CLOSE_SLIVER=<帧>` 打开；超出 `0..=12` 的值一律退回出厂（不许静默夹住）。
fn parse_close_sliver(v: Option<&str>) -> i64 {
    v.and_then(|x| x.trim().parse::<i64>().ok())
        .filter(|n| (0..=12).contains(n))
        .unwrap_or(CLOSE_SLIVER_FRAMES_DEFAULT)
}

/// 把相邻两窗之间 `1..=n` 帧的缝用**前一个窗的尾巴**补上。`n == 0` ⇒ 原样返回(逐帧不变)。
///
/// ⛔ 它必须跑在 [`merge_same_shift_across_rests`] **之后** —— 那一把先把「位移相同、
/// 中间只隔休止」的窗合掉,剩下的缝才**全部**是异位移的(实测 36/36)。
fn close_short_slivers(mut jobs: Vec<DeadJob>, n: i64) -> Vec<DeadJob> {
    if n <= 0 || jobs.len() < 2 {
        return jobs;
    }
    let mut order: Vec<usize> = (0..jobs.len()).collect();
    order.sort_by_key(|&i| jobs[i].start);
    for w in 0..order.len().saturating_sub(1) {
        let (a, b) = (order[w], order[w + 1]);
        let gap = jobs[b].start - jobs[a].end;
        if gap >= 1 && gap <= n {
            jobs[a].end = jobs[b].start;
        }
    }
    jobs
}

/// ⚙ 出厂默认 = `6.0`。**被救音相对乐句邻居的电平,超过这个门就软压**(dB)。`0` = 关。
///
/// ⛔ 出处不是拍的:**没被救的音**相对邻居的 |rel| **p90** = yuyuko 5.69 · akiko 3.84 ·
/// dxl41 3.26 · yachiyo鹅妈妈 3.71 · dxl41鹅妈妈 2.75 —— 那是**这首歌本身的起伏范围**
/// (那些音在开/关两条臂上逐位相同 ⇒ 与救援无关)。**6.0 取在所有臂的 p90 之上**,
/// 于是这一刀结构上只碰「超出音乐本身起伏」的那部分。
const RESCUE_LEVEL_MATCH_DB: f32 = 6.0;

/// ⛔ S162 起**不再使用**:软膝(ratio 6)把 rel +9.03 只压到 +6.93,用户耳判仍然听得到。
/// 阈值以上现在是**硬限**。留着这个常量是为了让下一个人看见「我们试过软膝、它不够」。
/// 软膝的压缩比(历史)。超出 [`RESCUE_LEVEL_MATCH_DB`] 的那部分按 `1 − 1/ratio` 压掉。
const RESCUE_LEVEL_MATCH_RATIO: f32 = 6.0;

/// 单个音的最大压低量(dB)。⛔ 防止一个坏参照把音推到荒谬处。
const RESCUE_LEVEL_MATCH_CAP: f32 = 12.0;

/// ⚙ 出厂默认 = 1.0(**开**,见 [`RANGE_TILT_DEFAULT`])。`UTAI_RANGE_TILT=0` 关掉;
/// `<0..1>` 取中间强度。
///
/// **深救援的「虚/弱」是一条谱【倾斜】,而靶子是【浅救援】——
/// 也就是用户说的「另一部分正常」那一半。**
/// 表、机理、为什么靶子不是 `base`、为什么这不是开 EQ:全在
/// `utai_dsp::psola::TILT_TABLE` 的 doc 上。
/// ⛔ `|s| ≤ 6` 一个字节不动(表的第一行按构造全 0)。
///
/// ## 翻默认的证据(S162)
/// * **留出验证**(用**另一首谱**拟的表纠正本谱):形状距离**降 26-46%**,
///   逼近「自己拟自己」的上界;⛔ 旧靶子(相对 `base`)在 −10 上反而更差 3.14 → 3.57。
/// * **跨模型零噪声验收**(5 组 × 2 档,同一份 donor 缓冲进出只翻这一个旋钮):
///   **面状/次基频 10/10 改善**(−0.58…−2.30 dB)· **梳深 10/10 不劣化** ·
///   **电平 −0.00…+0.00**(逐帧等响)。
/// * ⛔ 判负:**逐元音的表**(改善只有 0.0-0.5 dB,还有两格更差)。
pub fn range_tilt() -> f64 {
    std::env::var("UTAI_RANGE_TILT")
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite() && (0.0..=1.0).contains(v))
        .unwrap_or(RANGE_TILT_DEFAULT)
}

/// ⚙ 出厂默认强度。整份理由与读数在 [`range_tilt`] 头上。
const RANGE_TILT_DEFAULT: f64 = 1.0;

/// 出厂门的读取口(env `UTAI_RANGE_LEVEL_MATCH`)。见 [`RESCUE_LEVEL_MATCH_DB`]。
pub fn level_match_db() -> f32 {
    parse_level_match_db(std::env::var("UTAI_RANGE_LEVEL_MATCH").ok().as_deref())
}

fn parse_level_match_db(v: Option<&str>) -> f32 {
    v.and_then(|x| x.trim().parse::<f32>().ok())
        .filter(|t| t.is_finite() && (0.0..=24.0).contains(t))
        .unwrap_or(RESCUE_LEVEL_MATCH_DB)
}

/// S162 —— **被救音的乐句级电平匹配(只压不抬)**。
///
/// ## ⛔ 它治的是什么(用户 2026-08-25/26 两次点名)
/// 「yuyuko 的那个「ぴゃ」的响度都炸成什么了」——实测炉心融解 `[685]ぴゃ` 在 yuyuko 上
/// 比**邻近 12 个音的中位**高 **+9.0 dB**(用只算没被救邻居的口径是 **+11.9**),
/// 而同一个音在东雪莲上是 −1.1、akiko 上是 −12.2。
/// ⭐ 而它的落点(79)在**质量**上是对的:梳深 **61.2 dB**,落 80 只有 20.3 ——
/// 79 恰好是 yuyuko 扫描里 `rms_db = 0.0` = **全表最响**的那一格。
/// ⇒ **落点没选错,是响度没人管**:`match_levels` 五个调用点全传 `false`(死代码),
/// base 与 donor 只共用一个**全曲**归一标量 ⇒ donor 在它舒服的音高上唱多响,就多响地贴回歌里。
///
/// ## ⛔⛔ 为什么**只压不抬**
/// 双向版实测:被**抬**的音的**次基频占比**(= 面状伪影那条护栏)比没被碰的音高
/// **+6.4 / +17.5 / +6.6 dB**(yuyuko / akiko / dxl41)⇒ **它们本来就是面状最重的那一批**,
/// 抬它们就是把面状伪影抬起来 —— 用户 2026-08-26 早上正是为这件事提醒过。
/// ⇒ 与 [`RESCUE_LEVEL_FLOOR_DB`] 的「只收顶不收底」同形。
///
/// ## ⛔ 参照只用【没被救的】邻居
/// 它们落在窗外 ⇒ 与 base 逐位相同 ⇒ **参照固定,一步到位**。
/// 含被救邻居的版本实测**迭代不收敛**(max|Δg| 10.85 → 9.04 → 7.54 → 6.28)。
/// ⚠ 取不到 [`LEVEL_MATCH_MIN_REF`] 个干净邻居的音**一个字不动**(覆盖率实测 44-88%)。
///
/// ## ⛔ 粒度必须是【音符】不是【窗】
/// 窗粒度实测把最干净那条臂(dxl41 × 鹅妈妈)的 |rel| p90 从 **3.79 推到 4.9**——
/// 一个窗里有多个音,整窗一个增益会连累窗内本来正常的那些。音符粒度上它**一动不动**。
/// 逐音电平(dBFS)与「被救援窗覆盖的比例」。⛔ **唯一**一份口径 ——
/// 电平匹配与落点选法共用它,否则两处会各自漂移(S162 的 `subf0` 就是这样读反号的)。
/// 不是唱音 / 太短 / 取不到样本 ⇒ 电平 `NaN`、覆盖率 0。
fn note_levels_and_coverage(
    audio: &[f32],
    spf: f64,
    jobs: &[DeadJob],
    notes: &[NoteSpan],
) -> (Vec<f32>, Vec<f32>) {
    let alen = audio.len();
    let mut lv: Vec<f32> = Vec::with_capacity(notes.len());
    let mut cov: Vec<f32> = Vec::with_capacity(notes.len());
    for nd in notes {
        if !nd.sung || nd.frames < LEVEL_MATCH_MIN_FRAMES {
            lv.push(f32::NAN);
            cov.push(0.0);
            continue;
        }
        let a = (((nd.start as f64) * spf).round().max(0.0) as usize).min(alen);
        let b = ((((nd.start + nd.frames) as f64) * spf).round().max(0.0) as usize).min(alen);
        if b <= a + 256 {
            lv.push(f32::NAN);
            cov.push(0.0);
            continue;
        }
        let e: f64 = audio[a..b].iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>()
            / (b - a) as f64;
        lv.push((10.0 * (e + 1e-20).log10()) as f32);
        let c: i64 = jobs
            .iter()
            .map(|j| ((nd.start + nd.frames).min(j.end) - nd.start.max(j.start)).max(0))
            .sum();
        cov.push((c as f32 / nd.frames as f32).clamp(0.0, 1.0));
    }
    (lv, cov)
}

/// 逐音参照 = 邻近 ±[`LEVEL_MATCH_NEIGHBOURS`] 个**没被救**唱音的电平**中位**。
///
/// ⛔ 为什么参照必须是「在唱的、没被救的邻居」,而不是「窗外那一片 base 的平均」:
/// 后者在**密集救援的乐句**里主要是**休止**(数字静音)⇒ 参照读到 −100 dBFS ⇒
/// 「|电平 − 参照|」退化成「谁更轻谁赢」,方向正好反了。
/// 实测 akiko × 炉心 +7:60 个窗里有 **6 个**的窗外参照比全曲唱音中位低 >15 dB。
/// ⚠ 取不到 [`LEVEL_MATCH_MIN_REF`] 个干净邻居 ⇒ `NaN`,调用方**弃权**而不是瞎猜。
fn clean_neighbour_refs(lv: &[f32], cov: &[f32]) -> Vec<f32> {
    (0..lv.len())
        .map(|i| {
            let lo = i.saturating_sub(LEVEL_MATCH_NEIGHBOURS);
            let hi = (i + LEVEL_MATCH_NEIGHBOURS + 1).min(lv.len());
            let mut refs: Vec<f32> = (lo..hi)
                .filter(|&j| j != i && cov[j] < 0.05 && lv[j].is_finite())
                .map(|j| lv[j])
                .collect();
            if refs.len() < LEVEL_MATCH_MIN_REF {
                return f32::NAN;
            }
            refs.sort_by(f32::total_cmp);
            refs[refs.len() / 2]
        })
        .collect()
}

pub fn match_rescued_note_levels(
    audio: &mut [f32],
    sample_rate: u32,
    total_frames: i64,
    jobs: &[DeadJob],
    notes: &[NoteSpan],
    thresh_db: f32,
) -> usize {
    if thresh_db <= 0.0 || total_frames <= 0 || audio.is_empty() {
        return 0;
    }
    let alen = audio.len();
    let spf = alen as f64 / total_frames as f64;
    let span = move |f0: i64, n: i64| -> (usize, usize) {
        let a = ((f0 as f64) * spf).round().max(0.0) as usize;
        let b = (((f0 + n) as f64) * spf).round().max(0.0) as usize;
        (a.min(alen), b.min(alen))
    };
    // ── 逐音:电平 + 被窗覆盖的比例 + 干净邻居参照(共用口径)────────
    let (lv, cov) = note_levels_and_coverage(audio, spf, jobs, notes);
    let refs = clean_neighbour_refs(&lv, &cov);
    let mut hits = 0usize;
    let mut gains: Vec<(usize, f32)> = Vec::new();
    for i in 0..notes.len() {
        if !(cov[i] > 0.8) || !lv[i].is_finite() || !refs[i].is_finite() {
            continue;
        }
        let med = refs[i];
        let rel = lv[i] - med;
        if rel <= thresh_db {
            continue;
        }
        let over = rel - thresh_db;
        // ⛔⛔ S162(用户 2026-08-26 整曲耳判)—— **软膝太软,离群值压不下来**。
        // 实测 yuyuko 的「ぴゃ」rel **+9.03**,软膝(ratio 6)只压到 **+6.93**,用户仍然听到它。
        // 而**五条臂里它是唯一一个超过 6 的音**(其余被救音最大 2.96-5.98,
        // 而没被救的音自己的 |rel| p90 就有 2.90-5.41)⇒ 它是离群值,不是正常起伏。
        // ⇒ 阈值以上改成**硬限**(ratio → ∞):+9.03 → **6.00**。
        // ⭐ 外科手术级:实测**只碰 2 个音**(都在 yuyuko 上),其余四条臂**一个都不碰**。
        let g = -over.min(RESCUE_LEVEL_MATCH_CAP);
        gains.push((i, g));
        hits += 1;
    }
    // ── 施加:整个音符跨度上一个常数增益,两端各 10 ms 淡化 ────────
    let fade = ((sample_rate as usize) / 100).max(2);
    for (i, g) in gains {
        let (a, b) = span(notes[i].start, notes[i].frames);
        let k = 10f32.powf(g / 20.0);
        let w = fade.min((b - a) / 2);
        for t in a..b {
            let r = if w == 0 {
                1.0
            } else if t < a + w {
                (t - a) as f32 / w as f32
            } else if t + w >= b {
                (b - t) as f32 / w as f32
            } else {
                1.0
            };
            audio[t] *= 1.0 + (k - 1.0) * r;
        }
    }
    hits
}

/// ⚙ 出厂默认 = -3.0 / +2.0 dB(压 / 抬的上限)。`UTAI_RANGE_PHRASE_LEVEL=0` 关掉,
/// `<cut>/<lift>` 改上限。见 [`match_phrase_group_levels`]。
pub fn phrase_level_limits() -> (f32, f32) {
    parse_phrase_level(std::env::var("UTAI_RANGE_PHRASE_LEVEL").ok().as_deref())
}

fn parse_phrase_level(v: Option<&str>) -> (f32, f32) {
    let s = match v {
        None => return (PHRASE_LEVEL_CUT_DB, PHRASE_LEVEL_LIFT_DB),
        Some(s) => s.trim(),
    };
    if s == "0" {
        return (0.0, 0.0);
    }
    let mut it = s.split('/');
    let cut = it.next().and_then(|t| t.trim().parse::<f32>().ok());
    let lift = it.next().and_then(|t| t.trim().parse::<f32>().ok());
    match (cut, lift) {
        (Some(c), Some(l)) if c.is_finite() && l.is_finite() && c >= 0.0 && l >= 0.0 && c <= 24.0 && l <= 24.0 => (c, l),
        _ => (PHRASE_LEVEL_CUT_DB, PHRASE_LEVEL_LIFT_DB),
    }
}

/// ⭐⭐ S162 —— **乐句内跨组的电平对齐:一个乐句由几次【独立渲染】拼成时,把各段的整体偏置拉齐。**
///
/// ## ⛔ 它治的是什么(用户 2026-08-26 听出来的)
/// 「就算是出厂后 为什么东雪莲炉心融解的 4:32-4:36.334(那句里深救援那半段)
/// 和 4:36.334 到句尾 4:37.7 的响度区别那么大」——实测同一乐句内:
/// `[791]だ`(2.36 s 长音,shift **−12**)= −17.35 dB,`[794]あ`(1.28 s,shift **−5**)= −14.62
/// ⇒ **深的那半段轻 2.7 dB**,而这条轴的可闻刻度正是「~2.7 dB 听得出 / ≤0.46 听不出」,
/// 逐音电平的渲染噪声底只有 **0.74 dB** ⇒ **3.6 倍,不是噪声**。
///
/// ## ⛔⛔ 为什么 [`match_rescued_note_levels`] 管不了它
/// 那把刀的参照是**没被救的邻居**,而这两个音周围 ±16 里只有 **2 个**(门槛 [`LEVEL_MATCH_MIN_REF`] = 4)
/// ⇒ **整段跳过,一个音没碰**。**密集救援的乐句正好就是没有非救援邻居的地方** ——
/// 它在最需要它的地方结构上失效。
///
/// ## ⭐ 结构性原因(用户指出,数据证实)
/// 「拆组硬把很多次渲染一片又一片地拼在一起,那就真会把整个乐句拼烂」:
/// **24-44% 的乐句被切成 ≥2 段 donor**(最多 7 段;乐句内深度跨度中位 3-5 半音、最大 12),
/// 而**每一段来自一次独立的渲染**(而整曲渲染跨 run 不可复现 —— 见 `engine.rs` 的
/// `ort_determinism`)。实测乐句内跨组的长音电平台阶中位 **2.61 dB** / p90 5.49、
/// **46% 超过可闻门槛**;同组内相邻只有 20%(中位 1.76)。
///
/// ## 做法与它的边界
/// * 参照 = 乐句内 **|shift| 最小**的那一段(**没被救**的段 |shift| 记 0 ⇒ 它优先当参照)。
///   ⭐ 浅救援是好的,也是用户认可的那一半(「另一部分正常」)。
/// * **只去掉每段一个标量** —— 段内的强弱起伏一个字不动(判据里有逐位对照)。
/// * ⛔ **非对称限幅 [`PHRASE_LEVEL_CUT_DB`] / [`PHRASE_LEVEL_LIFT_DB`] = −3 / +2**:
///   抬更危险(它把面状伪影一起抬 —— S162 的「抬轻的被救音」已经为这件事判负过一次),所以给得少。
///   离线扫描:段偏差中位 **1.89 → 0.00 dB**、偏差 >2.7 dB 的 **53 → 15**、
///   长音跨组台阶 **2.52 → 0.96**(可闻 6 → 2)。
///   ⚠ 剩下的 15 段要 >3 dB 才拉得平 —— **故意不修**:那么大的增益比那个台阶更危险。
/// * ⛔ 段要够长才估电平([`PHRASE_LEVEL_MIN_SUSTAIN`]),否则一两个短音会给出 20 dB 的假增益
///   (未加门槛时实测 max **20.25 dB**)。
/// * ⚠ **只在谱面轨**:cover 车道没有音符表,这把刀在那里结构上跑不起来。
///
/// 返回**被施加了增益的段数**。
pub fn match_phrase_group_levels(
    audio: &mut [f32],
    sample_rate: u32,
    total_frames: i64,
    jobs: &[DeadJob],
    notes: &[NoteSpan],
    cut_db: f32,
    lift_db: f32,
) -> usize {
    if (cut_db <= 0.0 && lift_db <= 0.0) || total_frames <= 0 || audio.is_empty() {
        return 0;
    }
    let alen = audio.len();
    let spf = alen as f64 / total_frames as f64;
    let span = move |f0: i64, n: i64| -> (usize, usize) {
        let a = ((f0 as f64) * spf).round().max(0.0) as usize;
        let b = (((f0 + n) as f64) * spf).round().max(0.0) as usize;
        (a.min(alen), b.min(alen))
    };
    // ── 逐音:电平 + 这个音属于哪一段(= 覆盖它最多的那个 job 的 shift;没被救 = None)──
    let mut lv: Vec<f32> = Vec::with_capacity(notes.len());
    let mut seg: Vec<Option<i64>> = Vec::with_capacity(notes.len());
    for nd in notes {
        let (f0, n, sung) = (nd.start, nd.frames, nd.sung);
        let (a, b) = span(f0, n);
        let mut best: (i64, Option<i64>) = (0, None);
        for j in jobs {
            let ov = ((f0 + n).min(j.end) - f0.max(j.start)).max(0);
            if ov > best.0 {
                best = (ov, Some(j.shift));
            }
        }
        // ⛔ 覆盖不到一半就不算这一段的成员(半覆盖的音归属含糊,拿它估电平会污染两段)
        seg.push(if best.0 * 2 >= n { best.1 } else { None });
        if !sung || n < PHRASE_LEVEL_MIN_NOTE || b <= a + 256 {
            lv.push(f32::NAN);
            continue;
        }
        // 掐掉起音 40 ms 与收尾 40 ms,只量稳态
        let pad = ((sample_rate as usize) / 25).min((b - a) / 4);
        let e: f64 = audio[a + pad..b - pad]
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum::<f64>()
            / ((b - pad) - (a + pad)).max(1) as f64;
        lv.push(if e > 1e-14 { (10.0 * e.log10()) as f32 } else { f32::NAN });
    }
    // ── 乐句 = 连续的唱音串(遇到不唱的音就断)──────────────────────
    let mut hits = 0usize;
    let mut i = 0usize;
    while i < notes.len() {
        if !notes[i].sung {
            i += 1;
            continue;
        }
        let s = i;
        while i < notes.len() && notes[i].sung {
            i += 1;
        }
        let e = i; // [s, e)
        // ── 乐句内按段切 ─────────────────────────────────────────
        let mut runs: Vec<(Option<i64>, usize, usize)> = Vec::new();
        let mut k = s;
        while k < e {
            let cur = seg[k];
            let a = k;
            while k < e && seg[k] == cur {
                k += 1;
            }
            runs.push((cur, a, k));
        }
        if runs.len() < 2 {
            continue;
        }
        // ── 每段:稳态时长够不够 + 电平中位 ──────────────────────
        let mut est: Vec<Option<(f32, i64)>> = Vec::with_capacity(runs.len());
        for &(_, a, b) in &runs {
            let mut v: Vec<f32> = (a..b).filter(|&j| lv[j].is_finite()).map(|j| lv[j]).collect();
            let sus: i64 = (a..b).filter(|&j| lv[j].is_finite()).map(|j| notes[j].frames).sum();
            if v.is_empty() || sus < PHRASE_LEVEL_MIN_SUSTAIN {
                est.push(None);
                continue;
            }
            v.sort_by(f32::total_cmp);
            est.push(Some((v[v.len() / 2], sus)));
        }
        let cand: Vec<usize> = (0..runs.len()).filter(|&j| est[j].is_some()).collect();
        if cand.len() < 2 {
            continue;
        }
        // 参照 = |shift| 最小的那一段;平手取稳态最长的
        let refi = *cand
            .iter()
            .min_by_key(|&&j| (runs[j].0.unwrap_or(0).abs(), -est[j].unwrap().1))
            .expect("cand 非空");
        let refl = est[refi].unwrap().0;
        for &j in &cand {
            if j == refi {
                continue;
            }
            let g = (refl - est[j].unwrap().0).clamp(-cut_db, lift_db);
            if g.abs() < 0.05 {
                continue;
            }
            let (_, a, b) = runs[j];
            let (x0, _) = span(notes[a].start, notes[a].frames);
            let (_, x1) = span(notes[b - 1].start, notes[b - 1].frames);
            if x1 <= x0 {
                continue;
            }
            // 整段一个常数增益,两端各 10 ms 淡化(⛔ 不许逐音淡化 —— 那会在段内每个
            // 音符边界上刻出一串幅度凹口)
            let fade = ((sample_rate as usize) / 100).max(2).min((x1 - x0) / 2);
            let mul = 10f32.powf(g / 20.0);
            for t in x0..x1 {
                let r = if fade == 0 {
                    1.0
                } else if t < x0 + fade {
                    (t - x0) as f32 / fade as f32
                } else if t + fade >= x1 {
                    (x1 - t) as f32 / fade as f32
                } else {
                    1.0
                };
                audio[t] *= 1.0 + (mul - 1.0) * r;
            }
            hits += 1;
        }
    }
    hits
}

/// ⚙ 出厂默认 = true(**开**)。`UTAI_PSOLA_TAILFADE=0` 关掉。
///
/// **缓冲区末尾被截断的那半个岛,不许原样透传。**
/// 用户 2026-08-26:「歌曲结尾也会造出一个很明显的竖条纹伪影」。
/// 归因(同一次 run 的转储逐层比,零渲染噪声):`donor_post` 的末尾 **232 样本(5.26 ms)
/// 与 `donor_pre` 逐位相同**(shift −4 那一遍 259 样本 / 5.87 ms)⇒ **PSOLA 完全没碰它**;
/// 包络 −35 → **−24**,成品上 −25 → **−13**;盲搜读到 4:51.100 处 **5/7 带、峰 18.6 dB**(16k 独大),
/// 而同一次 run 的 `base` 里**没有**。
/// ⛔ 透传的是**没被移调的原音高**(低 12 个半音)—— 不只是电平台阶。
/// 机理与三条硬门在 `utai_dsp::psola::psola_shift_edge` 的函数体里。
pub fn tail_fade() -> bool {
    !matches!(std::env::var("UTAI_PSOLA_TAILFADE").ok().as_deref(), Some("0"))
}

/// ⚙ 出厂默认 = true(**开**,见 [`LANDING_PICK_DEFAULT`])。`UTAI_RANGE_LANDING_PICK=0` 关掉
/// ⇒ 逐位回到今天。
///
/// **渲染时按实测选落点**:计划器带出的第二个候选也渲一遍,逐组挑 `|rel|` 更小的那个。
/// 整份理由与读数在 [`apply_dead_only_windows_alts`] 头上。
pub fn landing_pick() -> bool {
    !matches!(std::env::var("UTAI_RANGE_LANDING_PICK").ok().as_deref(), Some("0"))
}

/// ⚙ 出厂默认。见 [`landing_pick`]。
const LANDING_PICK_DEFAULT: bool = true;

/// ⚙ 出厂默认 = 3.0(**dB**,见 [`LANDING_HARM_EPS_DEFAULT`])。
/// `UTAI_RANGE_LANDING_HARM=0` 关掉 ⇒ 落点选法退回「只看电平」。
///
/// ## ⭐⭐⭐ 它补的是落点这条链上唯一没人管的一根轴:**谐波结构**
/// S162 收工时定的根因是「落点链没有一层在优化耳朵在意的东西」——
/// 计划器的 `damage` 来自**扫描表**(400 ms 稳态「あ」,已证与「ぴゃ」这类音完全不相关),
/// 而 S162 加的渲染时选法优化的是 **`|rel|`(电平)**。⇒ **两层都不看谐波。**
///
/// 电平这根轴有两个结构性盲点(S163 逐条实测):
/// * ⛔ 它**需要参照**,而参照 = 邻近**没被救**的唱音 —— 密集救援的乐句里一个都没有
///   (实测 akiko × 炉心 +7:68 个带候选的窗里 **24 个**取不到);
/// * ⛔ 它**分不开「唱得响」与「唱得对」**:一个**根本没救到**的候选(落点仍在死区、
///   与关掉扩展的 `base` 几乎逐格相同)在电平上反而**更贴近**邻居。
///
/// ⇒ 加一根**不需要参照、而且与电平无关**的轴:
/// [`utai_dsp::harmonicity::harmonic_energy_fraction_db`] —— 「这段音频有多少能量真的
/// 落在**目标基频**的各次谐波上」。
///
/// ## 为什么阈值是 3 dB(实测定的,不是猜的)
/// akiko × 炉心 +7,同一份 donor 缓冲上逐音比两个候选(n = 275 个音):
/// **健康的一对**候选之间 |Δ| 的 p90 = **0.17 dB**、p99 = **1.83**、max 6.05;
/// 而那 5 处「救援被丢掉」的音是 **11-18.6 dB**。
/// ⇒ 3 dB 的门槛:5/5 灾难全抓住,240 个健康音里只碰 **1** 个
/// (音[130]「り」,它两个候选一个 −12.5 一个 −6.5,而被留下的那个在**四根轴上都更好**)。
///
/// ⛔ **它只在同一个音的两个候选之间比** —— 谐波序号尺子跨音区不可比(S162 栽过三次)。
pub fn landing_harm_eps() -> f32 {
    parse_landing_harm(std::env::var("UTAI_RANGE_LANDING_HARM").ok().as_deref())
}

fn parse_landing_harm(v: Option<&str>) -> f32 {
    v.and_then(|x| x.trim().parse::<f32>().ok())
        .filter(|t| t.is_finite() && (0.0..=24.0).contains(t))
        .unwrap_or(LANDING_HARM_EPS_DEFAULT)
}

/// ⚙ 出厂默认。见 [`landing_harm_eps`]。
const LANDING_HARM_EPS_DEFAULT: f32 = 3.0;

/// ⚙ 出厂默认 = 200.0(**毫秒**,见 [`LANDING_REPAIR_MS_DEFAULT`])。
/// `UTAI_RANGE_REPAIR=0` 关掉 ⇒ 不再渲修补遍,逐位回到「只有计划 + 候选」。
///
/// ## ⭐⭐⭐ 它治的是「救援把一个音唱没了」
/// 用户点名的 **yuyuko 4:49**:`[801]あ`(目标 MIDI 90,落点 −11)在 **289.62-289.94** 之间
/// 塌到 **−70 dBFS**,`base` 同刻平稳在 −32 —— 塌在**解码**里,PSOLA 与拼接都无辜。
/// S162 把它登记成「离群点」收工了。**S163 补上了那件一直没做的事:对这一处做落点扫描。**
///
/// | 落点 | −9 | **−10** | **−11(出厂)** | **−12** | −13 | −14 |
/// |---|---|---|---|---|---|---|
/// | 低于 −40 dBFS 的 20 ms 格 | 12 | **0** | **17** | **0** | 0 | 0 |
/// | 该段最低 dBFS | −72.5 | −27.0 | **−72.7** | −22.9 | −23.5 | −18.0 |
///
/// ⇒ ⭐ **只有 −11 与 −9 塌,一个半音之外全干净** —— 不是「模型上限」,是**那一格的死点**。
/// ⛔ 而落点选法结构上够不着它:yuyuko 的窄预算计划给出的落点与出厂**完全相同**
///    ⇒ 一个候选都没有 ⇒ 选法根本没被调用。
///
/// ## ⛔ 为什么是「先渲、发现塌了再补一遍」而不是「一开始就多渲 ±1」
/// 实测(四条臂的计划):给**每一组**都加 ±1 ⇒ donor 覆盖 85-152 s → **256-456 s = 3 倍**;
/// 只给 |s|≥8 的组加 ⇒ 仍是 **2 倍**。而**真正塌掉的音,四条臂 525 个被救音里只有 1 个**。
/// ⇒ 代价必须跟着「真的坏了」走,不能跟着「可能坏」走。
///
/// ## ⛔ 检测器为什么长这样(三条,全部是实测逼出来的)
/// ⑴ **不许用「音内中位 − 最低 100 ms」**:那把尺子被**辅音**污染 ——
///    掐掉音头 25%/音尾 15% 之后,四条臂上仍有 **6-11 个音**超过 22 dB,
///    而它们全是 `か`/`が`/`さ`/`つ`(音中间本来就有塞擦音)。
/// ⑵ **不许掐音头**:4:49 那一处的塌陷正好在**音的起头**(七度大跳的起音),
///    掐 25% 就把它掐掉了。
/// ⑶ ⇒ 判据 = 「唱音**内部**连续 ≥ 这个毫秒数、**绝对**低于 [`REPAIR_FLOOR_DBFS`]、
///    **而且**比该音自己的中位低 > [`REPAIR_REL_DB`]」。
///    实测选择性:**525 个被救音里只命中 1 个**(正是 `[801]あ`,300 ms),
///    akiko / yachiyo / 东雪莲 **0 个**。
pub fn landing_repair_ms() -> f32 {
    parse_landing_repair(std::env::var("UTAI_RANGE_REPAIR").ok().as_deref())
}

fn parse_landing_repair(v: Option<&str>) -> f32 {
    v.and_then(|x| x.trim().parse::<f32>().ok())
        .filter(|t| t.is_finite() && (0.0..=2000.0).contains(t))
        .unwrap_or(LANDING_REPAIR_MS_DEFAULT)
}

/// ⚙ 出厂默认。见 [`landing_repair_ms`]。
const LANDING_REPAIR_MS_DEFAULT: f32 = 200.0;

/// ⚙ 出厂默认 = 0.0（= 关 = 逐位不变）—— **谱峰宽度否决**的倍数门限
/// （`UTAI_RANGE_WIDTH_EPS`）。候选里峰宽最小的那个乘上它，超过的候选被剔掉。
///
/// ## 它量的是另一件事（四根轴都要有）
/// 实测 yuyuko **4:36 接缝两侧**（用户 2026-08-27 报「本来应该是连着的长音，
/// 结果在这个位置炸了」）：谱峰宽度 **12.33 vs 0.99（差 12 倍）**，
/// 而谐波间填充度读到 −18 vs −29 —— **方向还反了**；谐波占比与梳深也都看不见。
///
/// ## 为什么它在候选之间有效
/// yuyuko 短音（<250 ms）的 `donor_pre` 峰宽按**落点**：
/// **77 → 3.50 而 78 → 11.88**（差 **3.4 倍**，而两者只差 **1 个半音**）
/// ⇒ 正好在落点候选范围内。
/// ⛔ 而 sidecar 的 `low_ratio` 在同两格上是 **77 → 0.616（最差）/ 79 → 0.129（好）**
/// —— **与峰宽完全相反**，而 S157 那条排序用的就是 `low_ratio`，
/// 所以它会**主动避开 77、选中 79**。
///
/// ## ⭐⭐ 为什么它对「卡痰」那一族重要
/// 逐层归因（同一次 run 的转储）：短音（<250 ms）的峰宽
/// `base` **11.77** → `donor_pre` **7.91** → `donor_post` 5.94 → 成品 6.28；
/// 而长音（≥250 ms）是 7.26 → **1.32** → 1.18 → 1.18。
/// ⇒ **差别全在 `donor_pre`**（模型解码层，PSOLA 只动了 2 dB、拼接没动），
/// 而用户点名的卡痰段是 **5 秒 23 个音 ≈ 每个 200 ms** —— **全是短音**。
/// ⭐ 成品峰宽的 p75 在用户的三档对照上**严格单调**：
/// 最怪 **14.40** > 次怪 11.89 > 全曲 10.33 > 不怪 **5.84**（三档音长都是 180 ms）。
/// ⛔ **中位数不单调**（次怪比最怪还糊）—— 同一条旧训：长尾上用中位数会造假的非单调。
///
/// ⚠ 只在**同一个音的两个候选**之间比（同 f0）—— 跨音高时谐波密度不同，读数不可比。
/// ⛔⛔ **而且它不是中性的 —— 开着会把音弄塌。**
/// 实测 akiko 炌心 +7 的 `[490]ら`（2:08.004，与 0:55 那个 `[232]ら` **谱面完全相同**）：
/// | 臂 | 配置 | `[490]` 峰值 | `[490]` RMS | 相对 `[232]` |
/// |---|---|---|---|---|
/// | K1 / U1 | 出厂（旋钮全 0）| −5.96 / −5.94 | −13.62 | **+1.58 dB（更响）** |
/// | R1 | `floor=12` + `eps=1.6` | −7.77 | −18.84 | **−4.65 dB（塌）** |
/// | T1 | 同上 + `RADIUS=2` | −7.75 | −18.81 | **−4.62 dB（塌）** |
/// ⇒ 用户 2026-08-27 亲耳报「2:07 这里也没声了」—— **看的正是 T1**。
/// ⭐ 出厂（两个旋钮 0）上这个音**一点事没有**，而一旦把它们开到能真正触发的档位，
///   它就把一个本来健康的音选进了更差的候选。⛔ **这才是它出厂必须是 0 的真正理由。**
///
/// ⛔⛔⛔ **S163 实测：这一条干预链全线判负（人群面纹丝不动）—— 别重造。**
///
/// 四条臂逐步把路打通，每一步都确认“真的在跑”，而结果一直是 0：
/// | 臂 | 配置 | 触发 | akiko 峰宽中位/p75 |
/// |---|---|---|---|
/// | K | 都关 | — | 6.92 / 12.62 |
/// | Q | `floor=20` 单开 | repair 30/14，veto **0** | — |
/// | R | `floor=12` + `eps=1.6` | repair 48/32，veto **106/31** | 6.92 / **12.49** |
/// | T | 同上 + `REPAIR_RADIUS=2` | repair 47/32，veto **106/32** | 6.92 / **12.62** |
/// ⇒ 修补遍真的在渲、峰宽否决真的在选、搜索半径真的扩到了 ±2，
///   **而三个统计量一个都没动**；yuyuko 4:36 的 `[793]` 12.77 → 14.09（反而差）。
///
/// ⛔ 而且用户点名的 **akiko のぴゃ成品峰宽只有 1.09**（很清晰）
///   ⇒ **它的问题不在这根轴上**（它是上方谐波弱 −36.23，另一根）。
///
/// ⭐ **这根轴作为【诊断】仍然成立**（三档对照 p75 严格单调、跨模型都有 2-3 倍的好/坏格），
///   失败的是【干预】。两个旋钮留在树上出厂 0，**是路标不是遗留工作**。
///
/// ⚙ 出厂默认 = 0.0（= 关）—— **修补遍的第四个触发**：谱峰糊到超过这个
/// 百分比（相对 f0）就去渲 ±1（`UTAI_RANGE_WIDTH_FLOOR`）。
///
/// ## ⛔ 为什么它必须走修补遍，而不是「候选之间选」
/// 实测（N2 诊断臂，带日志）：[`landing_width_eps`] 那一层在**全曲只触发 1 次** ——
/// 大部分组只有 **1 个**候选，连 `mine.len() > 1` 都不满足，
/// **任何「在候选之间选」的轴都直接被跳过**。
/// ⭐ 而好格确实存在且就在隔壁（yuyuko 77 → 3.50 而 78 → 11.88，只差 1 个半音），
/// 但**每个模型的好格位置不同**（akiko 最好 74 / 最差 76；dxl 最好 75）
/// ⇒ 写不死，只能**先渲再选** —— 而那正是修补遍在做的事。
///
/// ## 选择性（donor_post 上，三个模型）
/// | 门限 | yuyuko | akiko | dxl |
/// |---|---|---|---|
/// | >12% | 20% | 28% | 21% |
/// | **>15%** | **15%** | **19%** | **13%** |
/// | >20% | 4% | 9% | 5% |
/// ⚠ 修补遍是按**组**渲的（同一个 shift 的音共享一遍 donor）⇒ 成本按组数算，不是音数。
pub fn landing_width_floor() -> f32 {
    parse_landing_width_floor(std::env::var("UTAI_RANGE_WIDTH_FLOOR").ok().as_deref())
}

/// The env parse, as a pure function so it can be asserted without touching process state.
fn parse_landing_width_floor(v: Option<&str>) -> f32 {
    v.and_then(|v| v.trim().parse().ok())
        .filter(|v: &f32| v.is_finite() && *v >= 0.0 && *v <= 100.0)
        .unwrap_or(LANDING_WIDTH_FLOOR_DEFAULT)
}

/// ⚙ 出厂默认 = 0.0 —— 见 [`landing_width_floor`]。
const LANDING_WIDTH_FLOOR_DEFAULT: f32 = 0.0;

/// ⚙ 出厂默认 = 0.0（= 关 = 逐位不变）—— **谱峰宽度否决**的倍数门限
/// （`UTAI_RANGE_WIDTH_EPS`）。候选里峰宽最小的那个乘上它，超过的候选被剔掉。
///
/// ## 它量的是另一件事（四根轴都要有）
/// yuyuko **4:36 接缝两侧**（用户 2026-08-27 报「本来应该是连着的长音，结果炸了」）：
/// 谱峰宽度 **12.33 vs 0.99（差 12 倍）**，而谐波间填充度读 −18 vs −29
/// —— **方向还反了**；谐波占比与梳深也都看不见。
///
/// ## ⛔ 实测：这一层在全曲只触发 1 次
/// N2 诊断臂（带日志）：**大部分组只有 1 个候选**，连 `mine.len() > 1` 都不满足
/// ⇒ 任何「在候选之间选」的轴都直接被跳过。
/// ⇒ 真正能用上它的是 [`landing_width_floor`]（**修补遍的第四个触发**，先渲再选）。
///
/// ⚠ 只在**同一个音的两个候选**之间比（同 f0）—— 跨音高时谐波密度不同，读数不可比。
pub fn landing_width_eps() -> f32 {
    parse_landing_width(std::env::var("UTAI_RANGE_WIDTH_EPS").ok().as_deref())
}

/// The env parse, as a pure function so it can be asserted without touching process state.
fn parse_landing_width(v: Option<&str>) -> f32 {
    v.and_then(|v| v.trim().parse().ok())
        .filter(|v: &f32| v.is_finite() && *v >= 0.0 && *v <= 20.0)
        .unwrap_or(LANDING_WIDTH_EPS_DEFAULT)
}

/// ⚙ 出厂默认 = 0.0 —— 见 [`landing_width_eps`]。
const LANDING_WIDTH_EPS_DEFAULT: f32 = 0.0;

/// ⚙ 出厂默认 = 6.0(**dB**,见 [`COMB_FLOOR_DB_DEFAULT`])。`UTAI_RANGE_COMB_FLOOR=0` 关掉。
///
/// ## ⭐⭐ 「像嗓子里卡着痰」——**谐波之间起雾**
/// 用户 2026-08-27 点名 yuyuko **2:09.002-2:14.056**。逐音四层(同一次 run 的转储):
/// 那一段里**只有 `[500]「に」` 塌** —— PSOLA 把梳深从 `donor_pre` 的 **11.5** 干到
/// `donor_post` 的 **−0.4**,而**同一段其他五个音 PSOLA 都是 +8.7…+35.6**(谐波反而更清晰)。
///
/// ⛔ 它与「ぴゃ 只剩基频」**不是同一件事**(用户当场点破我把两者合并了):
/// 「卡痰」= 梳深塌、谐波根数正常;「ぴゃ」= H2−H1 −44.8 而**梳深高达 40.6**。方向相反。
///
/// ## 门槛怎么来的
/// 525 个被救音在 `donor_post` 上的梳深:中位 **41.8-53.1** · p05 17.1-34.3 ·
/// **< 6 dB 的只有 1 个** —— 正是 `[500]「に」`。⇒ **选择性 1/525。**
/// ⭐ 它只用 `donor_post`(渲染层手上就有),不需要 `donor_pre`、不需要跨音高参照。
/// ⚙ 出厂默认 = 0.0（关）—— **谱峰宽度否决**的倍数门限（`UTAI_RANGE_WIDTH_EPS`）。
///
/// 候选里峰宽最小的那个乘上它，超过的候选被剔掉。`≤ 1.0` = 关 = 逐位不变。
///
/// ## 它量的是另一件事（三根轴都要有）
/// 实测 yuyuko **4:36 接缝两侧**：谱峰宽度 **12.33 vs 0.99（差 12 倍）**，
/// 而谐波间填充度读到 −18 vs −29 —— **方向还反了**；谐波占比与梳深也都看不见。
///
/// ## 为什么它在候选之间有效
/// yuyuko 短音（<250 ms）的 `donor_pre` 峰宽按**落点**：
/// **77 → 3.50 而 78 → 11.88**（差 **3.4 倍**，而两者只差 **1 个半音**）
/// ⇒ 正好在落点候选范围内。
/// ⛔ 而 sidecar 的 `low_ratio` 在同两格上是 **77 → 0.616（最差）/ 79 → 0.129（好）**
/// —— **与峰宽完全相反**，而 S157 那条排序用的就是 `low_ratio`，
/// 所以它会**主动避开 77、选中 79**。
///
/// ⚠ 只在**同一个音的两个候选**之间比（同 f0）—— 跨音高读数不可比。
pub fn comb_floor_db() -> f32 {
    parse_comb_floor(std::env::var("UTAI_RANGE_COMB_FLOOR").ok().as_deref())
}

fn parse_comb_floor(v: Option<&str>) -> f32 {
    v.and_then(|x| x.trim().parse::<f32>().ok())
        .filter(|t| t.is_finite() && (0.0..=40.0).contains(t))
        .unwrap_or(COMB_FLOOR_DB_DEFAULT)
}

/// ⚙ 出厂默认。见 [`comb_floor_db`]。
const COMB_FLOOR_DB_DEFAULT: f32 = 6.0;

/// 修补遍的第三个触发:两个候选的 `worst |rel|` **极差**超过这么多 dB,就说明落点在这里
/// 对结果**极其敏感**,而手上这两格未必是好的那一格 ⇒ 多渲 `shift ± 1` 看看。
///
/// ⛔ 出处不是拍的:akiko × 炉心 +7,40 个带候选的窗,极差**中位 0.90 dB / p90 1.96**,
/// **只有 1 个 > 6 —— 正是用户点名的「ぴゃ」(10.5)**;而强制扫描证明它真正的好落点是
/// **−14**(rel **+3.5** vs −12 的 −13.7,梳深 **70.2** vs 42.9,H2−H1 **−29.7** vs −34.5),
/// 从来没进过候选。`shift ± 1` 从计划的 −13 出发正好够得着。
const REPAIR_SPREAD_DB: f32 = 6.0;

/// ⭐⭐ S163 —— 修补遍的**搜索半径**（半音）。
///
/// ⛔ 从 ±1 扩到 ±2 的理由是实测出来的：R 臂（`width_floor` 与 `width_eps` 成对开）
/// 修补遍触发 32/48 次、峰宽否决真的在选（31/106 次），而人群面**几乎不变**。
/// 机理：yuyuko 的好格是落点 **77**（短音峰宽 3.50）而当前落点常是 **79**
/// ⇒ ±1 只到 78/80，**结构上够不到 77**。
///
/// ⚠ 成本：donor **遍数实测不涨**（新 shift 大多落在已有的那几个上），
/// 只有 wall 涨（K → Q 上升 48%）。⛔ 再往上扩之前先量遍数。
const REPAIR_RADIUS: i64 = 2;

/// ⭐⭐⭐ S165 —— `2·f0` 掉到这个值以下 ⇒ 这一组走修补遍,并且用 [`H2_REPAIR_RADIUS`] 的宽半径。
///
/// ⛔ 为什么要**独立于排序 eps** 的一个绝对门限:排序只在「两个候选之间」比,
/// 而 yachiyo 的候选产出率是 **0%**(两份计划给出相同落点)⇒ `decide_group` 开头就 return
/// ⇒ **不先把它渲出来,排序永远没有输入**。这与 S163 §27 给 `usag` 开的通路是同一条。
///
/// ⚙ −12 dB 由实测的触发率定:门限 −8/−10/−12/−14 ⇒ yachiyo 触发 39/30/**24**/24%、
/// yuyuko 48/24/**17**/14%(各 33 / 29 个可测窗)。−12 是「触发率掉到平台」的那一档。
const H2_REPAIR_FLOOR: f32 = -12.0;

/// ⭐⭐⭐ S165 —— `2·f0` 触发的组用这个半径,而不是 [`REPAIR_RADIUS`]。
///
/// ⛔ 为什么必须更宽:用户点名的那个音今天落在 **−8**(`2·f0` = −17.0 dB),
/// 而好格是 **−13**(−5.2 dB,与用户说「好」的 yuyuko **完全一致**)——
/// `REPAIR_RADIUS = 2` 从 −8 只够到 −10,**结构上够不着**。
/// ⚠ 代价:只有触发的那 17-24% 的组付,其余一个字节不动。
const H2_REPAIR_RADIUS: i64 = 6;

/// ⭐⭐⭐ S165 —— **失配参照**:从 `base` 里**没被任何救援窗碰过**的音,统计
/// 「某个电平的谐波**该抖多少**」。返回 `(电平档中心, 该档的抖动中位)`,按电平升序。
///
/// # 为什么靶子是「它自己的未救援段」
/// 用户 2026-08-28 定案:「油」= **响度与抖动失配**;而「不退化」的正确定义是
/// **让救援段像这个模型自己没被救援时的样子**,不是「更好听」。
/// 用户听过三个候选之后的原话:
/// 「和『同模型的未救援段』最像的**可能确实是 −15**;−13 确实更好但对 yachiyo 来说
///  **可能太干净了**;这算是 yachiyo 模型自己的问题…… **遵从原音色特征**可能是对的」。
/// ⇒ 靶子是**同模型、同音色、同录音条件**,零混杂;而且它直接度量「我们做坏了多少」。
///
/// # ⛔ 用 `base` 而不是成品
/// 未救援处**成品 = `base` × 一个全曲常数**(实测 0.824376,残差 −40.3 dB),
/// 而这两个读数都是**相对量**(电平相对 `f0`、抖动是 log 包络的起伏)⇒ 常数增益不影响。
/// 用 `base` 的好处是这一步能在**拼接之前**做完。
///
/// # ⚠ 参照本身可能是歪的
/// 实测 yachiyo 的未救援段曲线**几乎是平的**(1.60→1.38→1.54)——**它自己就是失配的那个模型**;
/// yuyuko 则有正常的斜率(1.85→0.93)。⇒ 对 yachiyo 这根轴只能做到「别更差」,
/// 这与用户「yachiyo 是模型底子、救不了」的判断一致。
fn mismatch_reference(
    base: &[f32],
    sample_rate: u32,
    spf: f64,
    jobs: &[DeadJob],
    notes: &[NoteSpan],
) -> Vec<(f32, f32)> {
    const EDGES: [f32; 8] = [-60.0, -30.0, -22.0, -16.0, -11.0, -7.0, -3.0, 10.0];
    let mut pairs: Vec<(f32, f32)> = Vec::new();
    for (ni, nd) in notes.iter().enumerate() {
        if !(nd.hz > 0.0) || nd.frames <= 0 {
            continue;
        }
        let a = ((nd.start as f64) * spf).round().max(0.0) as usize;
        let b = (((nd.start + nd.frames) as f64) * spf).round().max(0.0) as usize;
        if b <= a || b > base.len() || b - a < 4096 {
            continue;
        }
        // ⛔ 只要这个音落在**任何**救援组的范围里就跳过 —— 参照必须是「没被我们碰过」的。
        //    (`DeadJob::start`/`end` 是**音符下标**,不是帧。)
        if jobs.iter().any(|j| ni as i64 >= j.start && ni as i64 <= j.end) {
            continue;
        }
        let seg = &base[a..b];
        let e: f64 = seg.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / (b - a) as f64;
        if e.sqrt() < 0.01 {
            continue;
        }
        pairs.extend(utai_dsp::harmonicity::harmonic_level_jitter_pairs(
            seg,
            sample_rate,
            nd.hz,
            6,
        ));
    }
    let mut out = Vec::new();
    for w in EDGES.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        let mut v: Vec<f32> =
            pairs.iter().filter(|(l, _)| *l >= lo && *l < hi).map(|(_, j)| *j).collect();
        if v.len() < 8 {
            continue;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        out.push(((lo + hi) * 0.5, v[v.len() / 2]));
    }
    tracing::info!(
        "range: mismatch reference built from {} unrescued harmonic points ⇒ {} level bins {:?}",
        pairs.len(),
        out.len(),
        out.iter().map(|(c, m)| format!("{c:.0}→{m:.2}")).collect::<Vec<_>>()
    );
    out
}

/// ⭐ S165 —— 在参照曲线上查「这个电平该抖多少」(线性内插,两端取端点值)。
fn mismatch_expect(reference: &[(f32, f32)], level_db: f32) -> Option<f32> {
    if reference.is_empty() {
        return None;
    }
    if level_db <= reference[0].0 {
        return Some(reference[0].1);
    }
    if level_db >= reference[reference.len() - 1].0 {
        return Some(reference[reference.len() - 1].1);
    }
    for w in reference.windows(2) {
        let ((x0, y0), (x1, y1)) = (w[0], w[1]);
        if level_db >= x0 && level_db <= x1 && (x1 - x0).abs() > 1e-6 {
            return Some(y0 + (y1 - y0) * (level_db - x0) / (x1 - x0));
        }
    }
    Some(reference[reference.len() - 1].1)
}

/// ⭐⭐⭐ S165 —— 一个音的**失配度** = 逐根谐波 `抖动 − 该电平该有的抖动` 的**最大值**。
///
/// # ⛔ 为什么是「最差」而不是「平均」——这一条如果搞反,整把刀会反向
/// 实测 4:36 那个音(用户听过三个候选)的四种口径:
///
/// | shift | 平均偏离 | 均方距离 | 抖动分布距离 | **最差偏离** |
/// |---|---|---|---|---|
/// | **−8(出厂,用户判最差)** | **0.684 最小** | **0.753 最小** | **0.315 最小** | **+1.17 最大** |
/// | −13(用户:更好但太干净) | 0.959 | 1.017 | 0.888 | +0.36 |
/// | **−15(用户:最像它自己)** | 0.810 | 0.872 | 0.885 | **−0.43 最小** |
///
/// ⇒ **三种「整体距离」口径全都选中 −8**,只有「最差」选 −15。
/// 机理:−8 的大部分谐波都贴着参照,**只有一根(f5)严重失配 +1.17**,一平均就被稀释掉了;
/// 而听感**对异常敏感、不对平均敏感**。
/// ⚠ 还有一层:yachiyo 的参照本身就「抖」⇒ **最抖的候选反而「整体最像它」**
/// ⇒ 拿一个本身失配的模型当靶子时,整体口径会**奖励失配**。
fn note_mismatch(
    seg: &[f32],
    sample_rate: u32,
    f0_hz: f32,
    reference: &[(f32, f32)],
) -> Option<f32> {
    if reference.is_empty() {
        return None;
    }
    let pairs = utai_dsp::harmonicity::harmonic_level_jitter_pairs(seg, sample_rate, f0_hz, 6);
    let mut worst: Option<f32> = None;
    for (lv, jt) in pairs {
        let Some(e) = mismatch_expect(reference, lv) else { continue };
        let d = jt - e;
        worst = Some(worst.map_or(d, |w: f32| w.max(d)));
    }
    worst
}

/// ⭐⭐⭐⭐ S165 —— **音高闸**:候选里任何一个音的实测 `f0` 偏离目标超过这么多**音分** ⇒ 整个候选出局。
///
/// # ⛔ 为什么必须是硬否决(而不是又一根排序轴)
/// 唱错音是**质的失败**,不是「差一点」——它跟别的轴不可比:
/// 一个塌了八度的候选可能在 `rel`/`usag`/失配上全都好看(**音高塌了之后谐波反而更协调**)。
/// 实测那个灾难:目标 1480 Hz 而实唱 **320 Hz**(≈ **−2650 分**)。
///
/// ⚙ **150 分**(一个半半音):远大于颤音与正常跑调(±40 分内),
/// 远小于最轻的塌陷(半八度 = 600 分)。⛔ 别调到 100 以下 —— 会开始误伤颤音重的音。
const PITCH_GATE_CENTS: f32 = 150.0;

/// ⭐⭐⭐ S165 —— **失配**超过这个值 ⇒ 这一组走修补遍,并且用 [`MISM_REPAIR_RADIUS`] 的宽半径。
///
/// # ⛔ 为什么必须有它:候选池太小,失配轴根本没得选
/// 实测(`eps=0.05` 几乎不设限的一轮):失配轴每组只被问 **2-6 次**,
/// 而 `decide_group` 只看得到**修补遍实际渲出来的**候选 —— 默认 `REPAIR_RADIUS = 2`,
/// 那几组的原落点是 −1/−4/−9/−11,±2 够到的全是相邻档。
/// ⇒ **离线在转储里看到的那些更好的候选,引擎压根没渲出来** ⇒ 落点 **0 变化**。
/// ⚠ 这与 `h2` 撞的是同一堵墙(当时的解法是 [`H2_REPAIR_RADIUS`]),而失配轴原本没有对应的东西。
///
/// ⚙ 门限 **0.8**(第一版 1.5 ⇒ **太严**):用户点名的 4:36 那个音失配 **1.17 < 1.5**,
/// 实测 yachiyo `13816` 那组**触发次数 0** —— 该管的没管上。
/// ⭐ **正确形状:触发要松(多渲候选)、换档要严(少动)**。第一版把两者做反了:
/// `FLOOR=1.5` 太严 + `eps=0.3` 太松(gap 分布 0.34-2.48 ⇒ 几乎全过门)
/// ⇒ 修补遍 7→**50**、8 组换 6 组,**该管的没管上、不该动的全动了**。
/// ⚠ 触发率要实测,别照搬 —— 用户 2026-08-29:「**别对着一个模型硬改**」。
const MISM_REPAIR_FLOOR: f32 = 0.8;

/// ⭐⭐⭐ S165 —— 失配触发的组用这个半径。理由同 [`H2_REPAIR_RADIUS`]:
/// 好落点常常离当前档 **2-9 个半音**,`REPAIR_RADIUS = 2` 结构上够不着。
const MISM_REPAIR_RADIUS: i64 = 6;

/// ⭐⭐⭐⭐ S165 —— **失配轴的第二道对手轴闸**:换落点**不许把音内跌幅加深**超过这么多 dB。
///
/// ⛔ 为什么第一道闸不够:今天那道用的是 `uplev`(上方谐波电平),挡的是「**变闷**」。
/// 而实测 4:07.466 上,失配轴换的那个落点**上方谐波电平差不多**(所以一次都没被拦),
/// 却把音内跌幅从 **16.0 dB 挖到 21.4 dB** —— 听感从「哑噪声」变成「**断音**」。
/// 见 [`note_dip_db`] 的那张表。
///
/// ⚙ **3.0 dB**:实测那次恶化是 **+5.4 dB**,取 3.0 能挡住它而仍给小幅交换留余地。
/// ⛔ 触发率必须实测(`MISM_STAT` 的 BLOCKED 计数),别照搬 —— 用户 2026-08-29:
/// 「**别对着一个模型硬改**」。
const MISM_DIP_CAP: f32 = 3.0;

/// ⭐⭐ S165 —— `2·f0` 这一支的对手轴闸(dB)。见 [`landing_h2_eps`]。
///
/// ⛔ 比 [`LANDING_USAG_DIM_CAP_DEFAULT`] 松,而且是**用户耳判定的**:
/// 2026-08-28 听过 `−8 → −13` 的探针臂后说「**f1『强』/『实』确实在听感上更好,
/// 即使它把 f4 炸了**」——因为 `2·f0` = 1975 Hz 落在耳朵最敏感的区间,
/// 而 `5·f0` = 4939 Hz 感知权重低得多(A 计权差 ~7 dB)⇒ **同样的 dB 在听感上不等价**。
/// ⚠ 但闸不能不设:那一句的前提是「**前一个音本身 f4 也不强,所以不会造出跳变或接缝**」,
/// 换个上下文就不成立了。
const H2_DIM_CAP: f32 = 10.0;


/// ⚙ 出厂默认 = 3.0(**开**)。`UTAI_RANGE_USAG=0` 关掉 ——
/// **上方谐波音内塌陷**当排序主键时,「算有差别」的最小 dB。
///
/// # 它换掉的是什么
/// 今天 [`decide_group`] 的**唯一**排序键是 `worst_rel`。实测它在用户点名的坐标上方向是反的:
/// 4:07 那组 `[(-2, rel 4.48), (-4, rel 5.96)]` ——`rel` 偏好 −2,而 −2 的 upper-sag 是
/// **−10.8**(最差)。ぴゃ 更直接:修补遍**已经渲出了 −14/−15**
/// (日志 `repair — [(-12,60),(-14,40),(-14,40),(-15,60),…] ⇒ kept -12`),
/// 是排序把手里的好档丢了。
///
/// # 为什么要一个 eps 而不是无条件按它排
/// 两个候选只差零点几 dB 时,这根轴和噪声底分不开;而**对照组正好落在那个区间**:
/// 用户给的两个「听起来正常」的音,候选之间只差 **2.0 dB**,
/// 而三个缺陷点差 **17.9 / 7.6 / 6.6 dB**。
/// ⇒ `eps = 3.0`(≈ 可闻刻度)时,**缺陷全动、对照全不动**。
///
/// ⛔ 差不过 `eps` ⇒ 回落到今天的 `rel` 排序 ⇒ 那些组**逐位不变**。
pub fn landing_usag_eps() -> f32 {
    parse_usag_eps(std::env::var("UTAI_RANGE_USAG").ok().as_deref())
}

fn parse_usag_eps(v: Option<&str>) -> f32 {
    match v {
        None => USAG_EPS_DEFAULT,
        Some(t) => t
            .trim()
            .parse::<f32>()
            .ok()
            .filter(|x| x.is_finite() && *x >= 0.0 && *x <= 40.0)
            .unwrap_or(USAG_EPS_DEFAULT),
    }
}

/// ⚙ 出厂默认。见 [`landing_usag_eps`]。
const USAG_EPS_DEFAULT: f32 = 3.0;

/// ⚙ 出厂默认 = 3.0。`UTAI_RANGE_USAG_DIM=<dB>` 改;`0` = 不设闸(**危险**)。
///
/// ⭐⭐⭐ S163 —— **对手轴的闸**:按 `usag` 换档时,新档的上方谐波绝对强度
/// 不许比当前档低过这么多 dB。
///
/// # 为什么必须有它(实测,不是预防性设计)
/// `usag` 量「上方谐波在音内稳不稳」,不量「强不强」。第一轮 akiko 实测,
/// 落点真的改了的 5 个音里 **4 个买得值、1 个净亏**:
///
/// | 音 | 落点 | `usag` Δ | **上方谐波强度 Δ** |
/// |---|---|---|---|
/// | `[685]ぴゃ` | −12→−14 | **+17.84** | −1.13 |
/// | `[696]あ` | −2→−4 | +7.42 | −1.91 |
/// | `[430]あ` | −2→−4 | +4.92 | −1.45 |
/// | `[61]あ` | −2→−4 | +1.20 | −2.35 |
/// | **`[687]く`** | −12→−14 | **+0.98** | **−6.25** ⇐ 净亏 |
///
/// ⇒ **3.0 正好只拦 `[687]`,前四个全过。**
/// ⭐ 记忆 §11 早就记过同一个方向:akiko ぴゃ 的 −14 上方谐波比 −13 弱 **6.6 dB**。
pub fn landing_usag_dim_cap() -> f32 {
    parse_usag_dim(std::env::var("UTAI_RANGE_USAG_DIM").ok().as_deref())
}

fn parse_usag_dim(v: Option<&str>) -> f32 {
    match v {
        None => USAG_DIM_CAP_DEFAULT,
        Some(t) => t
            .trim()
            .parse::<f32>()
            .ok()
            .filter(|x| x.is_finite() && *x >= 0.0 && *x <= 40.0)
            .unwrap_or(USAG_DIM_CAP_DEFAULT),
    }
}

/// ⚙ 出厂默认 = 1.2 —— **失配**这根轴的 eps(dB);`0` = 整根轴关掉。
///
/// # 它在修什么
/// 用户 2026-08-28 定案:「油」= **响度与抖动失配**(「不是说『响就是好』或者『抖就是不好』……
/// **失配了才奇怪**」)。实测**匹配斜率**(抖动 ~ 谐波电平):
/// **yachiyo −0.0043 / yuyuko −0.0170 / SV −0.0134**(Welch PSD 口径)——
/// SV 与 yuyuko 的抖动随谐波变强而降,**yachiyo 几乎不降** = 强谐波本该稳却还在抖。
///
/// # ⛔ 它是**相对判据 + 最差口径**,两条都不许改
/// * **相对**:只在 `失配(当前) − 失配(最好候选) > eps` 时才动 ⇒ 被动的组数**天然等于能改善的组数**。
///   绝对门限会把「当前虽差但没有更好替代」也否掉(实测否掉 41% 候选而只有 10 组能改善)。
/// * **最差**:一个失配的音毁一整组。⛔ 用「平均/均方/分布距离」会**选中用户判为最差的那一档**
///   (见 [`note_mismatch`] 的四口径对照表)。
///
/// # ⚠ 靶子是**同模型的未救援段**,不是「更好听」
/// 用户听过三个候选后:「和『同模型的未救援段』最像的**可能确实是 −15**;
/// −13 确实更好但**可能太干净了**…… **遵从原音色特征**可能是对的」。
/// ⇒ 目标是**像它自己**,不是把音色改好看。
///
/// # ⭐⭐⭐ S165 收工翻成 1.2(`s165i` → `s165j`)
/// 用户 2026-08-29 听出**出厂臂上 4:36 又回到原来那样了**,并当场指出根因:
/// 「我不知道你是因为**没给上一刀抖动/落点选择那一刀翻默认/没带着**」——**是没翻默认**。
/// ⛔ 顺带一条:4:36 是**这根轴**(失配),**不是**谷/断点(那条在 4:07.466);
///    我第一次去查时拿谷深尺子量 4:36,量的是不相干的东西,被用户当场纠正。
///
/// | 配置 | `[793]あ` midi 78 | `[794]あ` midi 76 |
/// |---|---|---|
/// | **出厂 `0.0`**(渲两次) | **2.87 / 2.87** | 2.60 / 2.59 |
/// | `0.05` | 2.37 | 2.62 |
/// | `0.3` | 2.60 | 2.61 |
/// | **`1.2`**(渲两次) | **1.88 / 1.90** | 1.07 / 2.95 |
/// ⇒ **只有 1.2 真的动了它**;两组各自的两次渲染读数一致到 0.01-0.02
///   ⇒ 这把尺子在这个音上噪声底几乎为零,差别是配置带来的。
/// ⇒ ⛔ **别用 `0.05`/`0.3` 求稳:实测是空刀。**
/// ⭐ 耳判背书:用户听过 `Q1`(= 这个值)后说「**这回至少一没明显炸,二在 4:36 那真有变化**」。
///
/// ## ⚠ 代价,如实登记
/// 修补遍触发 **8 → 61 次** ⇒ donor 遍数 **20 → 37** ⇒ 渲染 249 s → 933 s(旧代码;
/// S165 的 sinc 那一刀把 `inverse` 压掉约 38 %)。**这是这一场最贵的一个默认。**
/// ⛔ 想省钱只能动 [`MISM_REPAIR_FLOOR`] / [`MISM_REPAIR_RADIUS`],而**那两个必须用生产的
///    `MISM_STAT` 定,不许拿离线复刻的尺子定** —— 实测离线 python 版在 `FLOOR=0.8` 上读到
///    98.4 % 触发,而生产实测约 48 %,**两个口径根本不是一回事**。
///
/// ## ⛔ 上线时仍然带着的两条(用户 2026-08-28)
/// **保留对手轴闸**(`UTAI_RANGE_H2` 那次哑音时诊断计数是 `BLOCKED 0 次` = 闸形同虚设)、
/// **验收看局部不看中位**(那次灾难在全曲中位上是 Δ +0.00 dB)。
pub fn landing_mismatch_eps() -> f32 {
    parse_mismatch_eps(std::env::var("UTAI_RANGE_MISMATCH").ok().as_deref())
}

/// ⚙ 出厂默认 = 1.2。见 [`landing_mismatch_eps`]。
///
/// ## ⭐⭐⭐ S165 收工翻的(`s165i` → `s165j`)
/// 用户 2026-08-29 听出**出厂臂上 4:36 又回到原来那样了**,并当场指出根因:
/// 「我不知道你是因为**没给上一刀抖动/落点选择那一刀翻默认/没带着**」——**是没翻默认**。
///
/// ⛔ 4:36 是**失配**(响度 ↔ 抖动)那条轴,**不是**谷/断点(那条在 4:07.466)。
///    我第一次去查时拿谷深尺子量它,量的是不相干的东西 —— 用户当场纠正。
///
/// ### 实测(同一批臂,逐谐波「电平 → 抖动」相对同臂未救援音的参照曲线,取最差谐波)
/// | 配置 | `[793]あ` midi 78 | `[794]あ` midi 76 |
/// |---|---|---|
/// | **出厂 `0.0`**(渲两次) | **2.87 / 2.87** | 2.60 / 2.59 |
/// | `0.05` | 2.37 | 2.62 |
/// | `0.3` | 2.60 | 2.61 |
/// | **`1.2`**(渲两次) | **1.88 / 1.90** | 1.07 / 2.95 |
/// ⇒ ⭐ **只有 1.2 真的动了它**,而且**两次渲染读数一致**(出厂那两次也一致到 0.01)
///   ⇒ 这把尺子在这个音上的噪声底几乎为零,差别是配置带来的。
/// ⇒ `0.05` / `0.3` 基本没动 ⇒ **别用小 eps「稳一点」,那是空刀**。
///
/// ### ⭐ 耳判背书
/// 用户听过 `Q1`(= 这个值)后说:「**这回至少一没明显炸,二在 4:36 那真有变化**」。
///
/// ### ⚠ 代价,如实登记
/// 修补遍触发 **8 → 61 次** ⇒ donor 遍数 **20 → 37** ⇒ 渲染 249 s → 933 s(旧代码)。
/// S165 的 sinc 那一刀把 `inverse` 压掉约 38 %,但**这仍然是这一场最贵的一个默认**。
/// ⛔ 想省钱只能动 [`MISM_REPAIR_FLOOR`] / [`MISM_REPAIR_RADIUS`],而**那两个必须用生产的
///    `MISM_STAT` 定,不许拿离线复刻的尺子定** —— 实测离线 python 版在 `FLOOR=0.8` 上读到
///    98.4 % 触发,而生产实测约 48 %,**两个口径根本不是一回事**。
const LANDING_MISMATCH_EPS_DEFAULT: f32 = 1.2;

fn parse_mismatch_eps(v: Option<&str>) -> f32 {
    match v {
        None | Some("") => LANDING_MISMATCH_EPS_DEFAULT,
        Some(x) => x
            .trim()
            .parse::<f32>()
            .ok()
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(LANDING_MISMATCH_EPS_DEFAULT),
    }
}

/// ⚙ 出厂默认 = 0.0(**关**)—— 等 A/B 整曲耳判后再翻。
///
/// ⭐⭐⭐ S165 —— **`2·f0` 电平**这根轴的排序 eps(dB);`0` = 整根轴关掉。
///
/// # 它在修什么
/// 用户 2026-08-28 指着三张频谱图说 yachiyo 的谐波线「**非常小段的忽明忽暗、不是一条干净的
/// 连续线**」,断得最明显的是「**基频上面那条**」= `2·f0`。实测(炉心 `[794]あ`,f0 987.8):
/// yachiyo **−17.0 dB** / yuyuko −5.2 / SV +0.9,而 `3·f0` 三条臂**一致**(都是谷)
/// ⇒ **只有这一根把它们分开**,落差 15 dB。
///
/// # ⛔ 本场先判负了**八把**尺子,全部错在同一件事上
/// 归一包络 std/mean · 包络谱尖峰度 · STFT 条纹 · 短窗峰宽 · 谐波间填充 · 存在率 ·
/// 40-60 Hz 调制占比 · log 域绝对快抖 —— **它们全在量「沿时间的变化」**,
/// 而真正的差别是「**跨谐波的幅度分布**」。⭐ **「忽明忽暗」是低电平的后果,不是原因。**
/// (旁证:一个独立的多 agent 搜索里,`jitter` 那把尺子的判负理由正是
/// 「yachiyo 这个音的 `2·f0` 比 yuyuko 弱 10.6 dB ⇒ 读数被谐波弱这一件事解释掉了」;
/// 而 `residual`(声源层规整度)判负说明忽明忽暗**不在激励层**。)
///
/// # ⚙ 翻它之前要看什么
/// ⚠ 渲染噪声底已验:目标音 h2 在 **7 条独立渲染**上极差 **0.02 dB**,而效应 11.8 dB
/// ⇒ gap/floor = **478×**(对比同批一把「调制深度」尺子有 1-3% 的窗摆动得比效应还大)。
pub fn landing_h2_eps() -> f32 {
    parse_h2_eps(std::env::var("UTAI_RANGE_H2").ok().as_deref())
}

/// ⚙ 出厂默认。见 [`landing_h2_eps`]。
const LANDING_H2_EPS_DEFAULT: f32 = 0.0;

fn parse_h2_eps(v: Option<&str>) -> f32 {
    match v {
        None | Some("") => LANDING_H2_EPS_DEFAULT,
        Some(x) => x.trim().parse::<f32>().ok().filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(LANDING_H2_EPS_DEFAULT),
    }
}

/// ⚙ 出厂默认。见 [`landing_usag_dim_cap`]。
const USAG_DIM_CAP_DEFAULT: f32 = 3.0;

/// ⚙ 出厂默认 = 0.0（**关**）。`UTAI_RANGE_DIPFILL=<dB>` 打开 ——
///
/// # ⛔⛔⛔ S163 v8 撤回出厂（用户 2026-08-27 听出新咔哒）
/// 两条独立的硬伤:
/// ⑴ **79% 的填充落在【休止】上**(yachiyo 402 段里 318 段;akiko 54 段里 24 段)
///    —— 谱面上那些格是 `R`,`donor` 在那里静音**是对的**,往里填 `base`
///    = **在空拍里塞声音**。这一刀在 v7 之前从头到尾**没看过谱面**。
///    我甚至把其中两处(`ぴゃ` / `3:40.829`)报成过「改善 +9.8 / +6.1」。
/// ⑵ 填充边界留下**台阶**:坑外 donor − 填进去的 base,四模型 552 段
///    p50 3.0 / p90 **18.3** / max **109** dB,而 `DIPFILL_FADE_MS = 10` 压不住。
///    ⛔ 单加台阶门槛分不开:听着改善那几处(16.1)与咔哒那几处(17.6)读数重叠。
///
/// ⇒ 证据不足 + 造伪影 ⇒ 撤回出厂。判据与机理全部保留在这里,
///   要再用它,先补上**休止闸**(坑必须整段落在 `sung` 音符内)与**电平匹配**。
/// **donor 静音坑回填**:donor 相对它自己在该窗内的中位低过这么多 dB 时,把 `base` 混回来。
///
/// # 靶子(四模型共 194 个,用户 2026-08-27 点名 akiko 2:05.252 是其中之一)
/// 「成品比 `base` 低 >8 dB、宽 <150 ms、两侧救援本来在工作」的洞,
/// **80% 落在音符结束前 30-45 ms**。
///
/// # ⛔ 逐个排除(都有诊断臂,别重造)
/// **不是修补遍**(关掉反而 51 个)· **不是 tilt**(关掉 45)· **不是电平层**(关掉 48)·
/// **不是接缝/淡化**(43 个里 **36 个在窗正中**,离淡化区 200-2000 ms)。
/// ✅ **92% 是 donor 自己就比 `base` 低 >6 dB** —— 模型用移调后的 f0 重唱时,
/// 在音尾比原音高时**更早、更深**地衰减,而 `base` 在那里是正常的。
///
/// # 为什么门限是 20
/// 相对**该窗内的中位**:洞处 donor **p50 −34.9…−39.2 dB**,而 `base` 只有 **−8.8…−13.4**
/// ⇒ 20 dB 把两者完全分开(同门限下 `base` 只命中 0-4 个)。
///
/// # 安全性(四模型,命中段上的 `base − donor`)
/// akiko +15.3 dB(有益 89% / 有害 **3%**)· yuyuko +15.6(90% / **1%**)·
/// 东雪莲 +12.7(94% / **0%**)· yachiyo +10.1(83% / **1%**)
/// ⇒ 那里 `base` 比 donor 高 10-16 dB(donor 几乎是静音),填 `base` 是**把有声的换回来**,
/// 不是「退回没救援」。命中 **3-6% 的格**。
///
/// ⛔ **不碰 donor 选择、不碰落点** —— 纯拼接层(用户:「别把 donor 选择再整烂了」)。
/// ⛔ **全曲统一判据,不针对任何坐标**(用户:「普遍问题别特殊化处理」)。
pub fn dipfill_depth_db() -> f32 {
    parse_dipfill(std::env::var("UTAI_RANGE_DIPFILL").ok().as_deref())
}

fn parse_dipfill(v: Option<&str>) -> f32 {
    match v {
        None => DIPFILL_DEPTH_DEFAULT,
        Some(t) => t
            .trim()
            .parse::<f32>()
            .ok()
            .filter(|x| x.is_finite() && *x >= 0.0 && *x <= 60.0)
            .unwrap_or(DIPFILL_DEPTH_DEFAULT),
    }
}

/// ⚙ 出厂默认。见 [`dipfill_depth_db`]。
const DIPFILL_DEPTH_DEFAULT: f32 = 0.0;

/// ⭐ S163 —— 回填段的最大宽度(ms)。比这更宽的不是「坑」,是「整段没救到」,
/// 那种情况填 `base` 等于把整段救援退掉 ⇒ 不碰。
const DIPFILL_MAX_MS: f32 = 80.0;

/// ⭐⭐⭐⭐ S163 v8 —— **窗内休止段回到 `base`** 时两端的过渡宽度(ms)。
///
/// # 缺陷(用户 2026-08-27:「那几个位置应该是短暂空拍,不应该有东西」)
/// 救援窗把「两个唱不上去的音之间的休止」整段包进来,于是**休止里写的是 donor**。
/// 而 `join_rests` 出厂是**关**的(`JOIN_RESTS_DEFAULT = false`)
/// ⇒ 在 v8 之前,休止**根本没有被任何一层处理过**。
///
/// 实测(同一次 run,零噪声,谱面对照):
/// ```text
/// yachiyo(鹅妈妈) 窗**覆盖**的休止 51 格:成品−base p90 **+8.4** / max **+21.6** dB
///                                        >3 dB 的 **14/51 (27%)**
///                 窗**未覆盖**的休止 140 格(阴性对照):p90 +2.7,>3 dB 的 13/140 (**9%**)
/// 用户点名:1:23.081 base −36.8 → 成品 −28.4(**+8.4**,窗覆盖)
///           3:40.829 base −30.5 → 成品 −24.9(**+5.7**,窗覆盖)
/// ```
/// ⇒ 被抬高的比例是未覆盖的 **3 倍** ⇒ 归因锁死在「窗覆盖」这一件事上。
///
/// ⚠ 过渡不许硬切:休止两侧是被救援的唱音,硬切会在空拍边缘造新的咔哒。
const REST_BASE_FADE_MS: f32 = 10.0;

/// ⭐⭐⭐⭐ S163 v10 —— 休止段**增益渐变**的头部宽度(ms)。
///
/// # v8/v9 判负：内容切换 = 缝
/// 往窗里插一段 `base`,两端**必然**造缝。实测(同一次 run,零噪声):
/// ```text
/// yachiyo 3:55.266 音头跳变:v8(不动) 16.3 dB → v9 **25.8 dB**(放大 9.5)
///         3:46.605          19.3 → 25.6
/// ```
/// v8 留 80 ms tail guard ⇒ 短休止(120 ms)全被吃掉、**什么都没做**;
/// v9 tail guard 归零 ⇒ 缝正好挪到**音头**上 ⇒ 用户听到竖线。
/// **两版同一个根。**
///
/// # v10 改成只压电平
/// 休止段仍然是 **donor**(音色、辅音、时序一个字节不动),只把增益压到 `base` 的电平。
/// 没有内容跳变 ⇒ 没有缝。
const REST_GAIN_FADE_MS: f32 = 20.0;

/// ⭐ S163 v10 —— 休止增益的下限(dB)。`base` 在空拍里也不是绝对静音,
/// 压过头会把休止压成数字静音,反而在两侧留下新的落差。
const REST_GAIN_MIN_DB: f32 = -18.0;

/// ⭐⭐⭐⭐⭐ S163 v17 —— **起音形状整形**的作用长度(ms，从音头起算)。
///
/// # 归因（【7a】，同一次 run 四层分解，零噪声）
/// ```text
/// 逐层增量(配对中位;清辅音开头 + 窗覆盖的音头;ya n=217 / aki n=128)
///                         模型侧(pre−base)   **PSOLA(post−pre)**   拼接(成品−post)
/// 起音陡度(ms,负=更硬)  ya  **−8.00**            **+0.00**            +0.00
///                       aki **−23.95**           **+0.00**           +21.95
/// 过冲(dB,正=更硬)      ya  **−2.63**            **+0.00**            +0.00
///                       aki **−8.19**            **+0.00**           +7.74
/// ⛔阴性对照(窗未覆盖):base 起音 18.0 ms → 成品 18.0 ms ⇒ Δ **+0.00**
/// ```
/// ⇒ **PSOLA 那一列全是 +0.00** —— 硬是**模型侧**造的
///   (降调唱出来的 donor 起音本来就比 `base` 陡 8-24 ms、过冲高 2.6-8.2 dB)。
/// 用户 2026-08-28:「重点还是最开始**辅音处的那几十毫秒**;听起来就是**很硬**。」
///
/// # ⛔ 与三把判负的刀的区别(别重造)
/// * **v14**:压到 `base + 6 dB` 以内 ⇒ **砍电平**(p10 −9.91 dB,单点 −15.2)
/// * **v15**:v14 + 能量归一化 ⇒ 地基(flux 比值尺子)被推翻
/// * **v16**:拉长淡入 ⇒ 只是**整体延后 30 ms**,120 ms 极端剂量用户仍听不出 ⇒ 与「硬」无关
/// * **v17(本刀)**:量的是**上升曲线的形状** —— 两侧各自**归一化到自己的稳态**之后再比,
///   ⇒ **稳态电平一个字节不动**,只把 0-N ms 的爬升速度改成 `base` 的。
const ONSET_FIT_MS: f32 = 60.0;

/// ⭐ S163 v17 —— 形状整形的增益限幅(dB)。
/// ⛔ 不能太大:`base` 在那些音上本来就是「唱不上去的破音」,
/// 完全照抄它的形状会把破音的抖动也搬过来。
const ONSET_FIT_MAX_DB: f32 = 9.0;

/// ⭐ S163 v17 —— 稳态电平的取样窗(ms，从音头起算)。归一化用它，所以它决定
/// 「形状差」和「电平差」怎么分开。⛔ 太靠前会把起音本身算进稳态。
const ONSET_FIT_STEADY_LO_MS: f32 = 50.0;
/// 见 [`ONSET_FIT_STEADY_LO_MS`]。
const ONSET_FIT_STEADY_HI_MS: f32 = 150.0;

/// S163 v8 -- rest head guard (ms): the previous note's natural release lives here.
/// The previous note IS rescued, so its tail must stay donor.
const REST_HEAD_GUARD_MS: f32 = 40.0;

/// S163 v8 -- rest tail guard (ms): `consonant_preroll` moves the NEXT note's
/// voiceless consonant earlier, so it lands inside this rest.
/// Japanese voiceless consonants run 50-80 ms; 80 ms keeps the whole preroll on donor.
/// ⛔ 铁律:辅音时序一个字节不许动。这条 guard 就是为它留的。
/// ⭐ S163 v10 —— 休止**尾部**增益从 `g` 回到 1 的宽度(ms)。
/// `consonant_preroll` 把下一个音的清辅音提前进这段休止 ⇒ 尾部必须已经回到原电平,
/// 否则辅音被压弱。⛔ 铁律:辅音时序一个字节不许动 —— v10 连内容都不动,只动增益。
const REST_TAIL_GUARD_MS: f32 = 40.0;

// ⛔ S163 v9 —— tail guard 从 80 降到 0。理由(实测):
//    用户点名的那几格休止是 **120 ms**(yachiyo 82.980..83.100 / 220.820..220.940),
//    而 head 40 + tail 80 = 正好 120 ⇒ 中段为零 ⇒ **这一刀在它们身上什么都没做**
//    (闸开/闸关读数逐位相同:+10.5 → +10.5、+5.9 → +5.9)。
//    而「辅音时序一个字节不许动」管的是**不许改 preroll 参数**;
//    `base` 与 `donor` 用的是**同一份谱面、同一套时序**,在辅音区改用 `base`
//    不改变任何时序,只是把那一段的**内容**从移调臂换回原生臂。

/// ⚙ 出厂默认 = true —— `UTAI_RANGE_REST_BASE=0` 关掉。窗内的**休止**保持 `base`,
/// 救援不碰空拍。机理与读数见 [`REST_BASE_FADE_MS`]。
/// ⚙ 出厂默认 = true —— `UTAI_RANGE_ONSET_FIT=0` 关掉。**起音形状整形**:
/// 把 donor 起音的上升曲线拉回 `base` 的形状(稳态电平不动)。见 [`ONSET_FIT_MS`]。
pub fn onset_fit_enabled() -> bool {
    // ⛔ S163 v17b 判负并**默认关**(用户 2026-08-28:「那刚才那一刀我觉得也得关;它应该还是没在点上」)。
    //    它确实把起音陡度追回了 base(−2.00 → +2.00 ms)、稳态一个字节没动(−0.015 dB),
    //    用户也确认过「辅音倒是稍微好点了」,但**没打到点上**。
    //    真正的线索在别处:**浊辅音「だ」被渲成了清辅音「た」的听感**(见 §39.5 / 【8a】)。
    matches!(
        std::env::var("UTAI_RANGE_ONSET_FIT").ok().as_deref().map(str::trim),
        Some("1") | Some("on") | Some("true") | Some("yes")
    ) || !matches!(
        std::env::var("UTAI_RANGE_ONSET_FIT").ok().as_deref().map(str::trim),
        Some("0") | Some("off") | Some("false") | Some("no") | None
    )
}

pub fn rest_base_enabled() -> bool {
    !matches!(
        std::env::var("UTAI_RANGE_REST_BASE").ok().as_deref().map(str::trim),
        Some("0") | Some("off") | Some("false") | Some("no")
    )
}

/// ⭐⭐⭐⭐ S163 v2 —— **谐波收益闸**:`base` 在坑段的上方谐波强度
/// (`upper_harmonic_level_db`,2-8×f0)必须比 donor 高过这么多 dB,否则**不填**。
///
/// # ⛔ 为什么这一条是承重的
/// v1 只按全带 RMS 找坑就填,结果**收益主要在基频**
/// (akiko H1 **+12.1** 而 H3-H8 只有 **+3.6**;yachiyo 全线 **−1.5** ⇒ 净负)。
/// 用户 2026-08-27 点破:「**你倒是基频连上了 谐波你是一点也不管啊?**」
///
/// # 用户点名坐标上的 H3-8 收益(实测,干净转储)
/// ```text
/// 应该填：4:50.776 +20.7 · 4:32 深救援 +14.1 · 4:49 +7.8 · 4:07.467 +4.6 · 2:05.252 +2.8
/// 不该填：1:21.195 −7.7 · 0:44.870 −6.8 · 0:57.240 削波 −5.4 · 0:56 次怪 −3.2
/// ```
/// ⇒ 只降门限而不加这道闸,会把 `1:21.195` / `0:44.870` 这些**填坏**。
const DIPFILL_GAIN_DB: f32 = 0.0;

/// ⭐⭐⭐⭐ S163 v4 —— **电平闸**:坑区间上 `base` 必须比 `donor` 响这么多才准填。
///
/// # 为什么必须有它
/// v3 的判据只问「`donor` 相对**它自己**窗内中位低不低」,
/// **从没问过「`base` 在这里是不是比 `donor` 还低」**。而这一刀的语义前提是
/// 「donor 这儿没声了,用 base 顶上」—— `base` 更低的时候填它,结构上只会更糟。
///
/// v3 实测两处(都是 dipfill 亲手动的):
/// ```text
/// akiko 2:07.987   成品−base 最低  +6.4 → −0.0   (Δ −6.4)
/// akiko 4:10.187   成品−base 最低  +5.0 → −0.0   (Δ −5.0)
/// ```
/// 两处在关掉这一刀时**成品比 base 高 5~6.4 dB**(救援正常工作),
/// 填进 base 之后精确落到 `−0.0` = base 电平 ⇒ **把好的救援削掉了**。
///
/// # ⛔⛔⛔ 尺子必须是「逐 10 ms 格的最大差」,不是区间平均
/// v4 用 rms(区间平均)实测:
/// ```text
///                     rms 尺子      峰值尺子
/// ぴゃ      (该保住)    −3.8 dB       **+11.6 dB**
/// 2:07.987(该拦住)   −17.3 dB       −15.4 dB
/// 4:10.187(该拦住)    −7.3 dB        −4.8 dB
/// ```
/// **rms 尺子结构上分不开这三个** —— `ぴゃ` 的 −3.8 夹在两处「该拦住」的读数中间,
/// 任何门槛都是「要么全过、要么全拦」。v4 取 3 dB ⇒ 两处退化拦住了,
/// 但 `ぴゃ` 的 **+10.0 改善也一起丢了**(V −0.1 → W −10.2)。
///
/// ⇒ 根因是**洞的判据用「最深一格」而闸用「区间平均」**,两把尺子量同一件事。
///   闸换成同一把尺子之后三个全分得开,3 dB 门槛的余量是 +8.6 / −18.4 / −7.8。
///
/// # 3 dB 而不是 0
/// 0 会在边界上抖(两边差零点几 dB 时填不填全看噪声)。
const DIPFILL_LEVEL_GAIN_DB: f32 = 3.0;

/// ⭐ S163 —— 回填的淡入淡出(ms)。⛔ 不许硬切:两侧是 donor、中间是 base,
/// 硬切会在 10 ms 内造出两条新的不连续。
const DIPFILL_FADE_MS: f32 = 10.0;

/// ⭐⭐⭐ S163 —— 找出 `seg` 里「相对自己中位低过 `depth`」且宽 ≤ `DIPFILL_MAX_MS` 的坑。
///
/// 返回 `(起, 止)` 的样本区间(相对 `seg` 的下标)。
/// ⛔ 中位只用**有声**的格算(`> -60 dBFS`),否则窗内的静音会把参照拖到地板上。
fn dipfill_spans(
    seg: &[f32],
    sample_rate: u32,
    depth: f32,
    // ⭐⭐⭐⭐ S163 v2 —— `base` 在同一段的样本(与 `seg` 同长同起点),用来算谐波收益闸。
    //    传空切片 ⇒ 闸不生效(回到 v1 行为,只给判据自测用)。
    base_here: &[f32],
    // ⭐ 该段的基频(Hz)。`<= 0` ⇒ 算不了谐波 ⇒ 闸不生效。
    f0_hz: f32,
) -> Vec<(usize, usize)> {
    if depth <= 0.0 || seg.is_empty() {
        return Vec::new();
    }
    let h = (sample_rate as usize / 100).max(1); // 10 ms
    let n = seg.len() / h;
    if n < 8 {
        return Vec::new();
    }
    let lv: Vec<f32> = (0..n)
        .map(|i| {
            let c = &seg[i * h..(i + 1) * h];
            let e: f64 = c.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / h as f64;
            (10.0 * (e + 1e-20).log10()) as f32
        })
        .collect();
    let mut alive: Vec<f32> = lv.iter().copied().filter(|v| *v > -60.0).collect();
    if alive.len() < 8 {
        return Vec::new();
    }
    alive.sort_by(f32::total_cmp);
    let med = alive[alive.len() / 2];
    let maxw = (DIPFILL_MAX_MS / 10.0) as usize;
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < n {
        if lv[i] < med - depth {
            let a = i;
            while i < n && lv[i] < med - depth {
                i += 1;
            }
            if i - a <= maxw {
                let (x, y) = (a * h, (i * h).min(seg.len()));
                // ⭐⭐⭐⭐ S163 v2 —— **谐波收益闸**:填进去的 `base` 必须在上方谐波上真的更好。
                //    ⛔ 只看全带电平会「把基频接上而谐波一点没补」(见 [`DIPFILL_GAIN_DB`])。
                //    ⚠ 谐波要在一个够长的窗上量,而坑本身可能只有 20 ms ⇒ 两侧各借 2048 样本。
                // ⭐⭐⭐⭐ S163 v4 —— **电平闸**(见 [`DIPFILL_LEVEL_GAIN_DB`])。
                //    ⛔ 先算它:比谐波闸便宜得多,而且它拦掉的是「把好救援削掉」那一族。
                let level_ok = if base_here.len() >= seg.len() {
                    // ⛔⛔ **逐 10 ms 格取最大差,不是区间平均** —— 见 [`DIPFILL_LEVEL_GAIN_DB`]。
                    //    洞的判据用「最深一格」,这条闸也必须用同一把尺子:
                    //    v4 用 rms 时 `ぴゃ` 读 −3.8 dB 而峰值读 **+11.6 dB**,
                    //    夹在两处「该拦住」的读数中间 ⇒ 结构上分不开。
                    let cell = h.min(y - x).max(1);
                    let mut best = f64::NEG_INFINITY;
                    let mut p = x;
                    while p < y {
                        let q = (p + cell).min(y);
                        let e = |v: &[f32]| -> f64 {
                            v.iter().map(|&z| f64::from(z) * f64::from(z)).sum::<f64>()
                                / (q - p).max(1) as f64
                        };
                        let (be, de) = (e(&base_here[p..q]), e(&seg[p..q]));
                        let diff = 10.0 * ((be + 1e-20) / (de + 1e-20)).log10();
                        if diff > best {
                            best = diff;
                        }
                        p = q;
                    }
                    best.is_finite() && best > f64::from(DIPFILL_LEVEL_GAIN_DB)
                } else {
                    false
                };
                let pass = level_ok && if base_here.len() >= seg.len() && f0_hz > 20.0 {
                    let pad = 2048usize;
                    let lo = x.saturating_sub(pad);
                    let hi = (y + pad).min(seg.len());
                    match (
                        utai_dsp::harmonicity::upper_harmonic_level_db(
                            &base_here[lo..hi],
                            sample_rate,
                            f0_hz,
                        ),
                        utai_dsp::harmonicity::upper_harmonic_level_db(
                            &seg[lo..hi],
                            sample_rate,
                            f0_hz,
                        ),
                    ) {
                        (Some(b), Some(d)) => b - d > DIPFILL_GAIN_DB,
                        // 量不到 ⇒ 保守：不填（宁可留坑，也不填一个没验证过的 base）
                        _ => false,
                    }
                } else {
                    // ⛔ 拿不到 base 或 f0 ⇒ **保守不填**：没法验证谐波收益的填充一律不做。
                    //    这也让判据 
                    //    自然通过 —— 它传的  是空的 ⇒ f0=0 ⇒ 这一刀不生效。
                    false
                };
                if pass {
                    out.push((x, y));
                }
            }
        } else {
            i += 1;
        }
    }
    out
}

/// ⭐⭐⭐ S163 —— 静音次键「算有差别」的最小毫秒数。
///
/// ⚙ 出厂 **15.0 ms**。理由:`silence_run_ms` 的量化格是帧(20 ms),
/// 所以 15 ms 意味着「至少差一帧」;比这更细的差是量化噪声。
/// 实测 ぴゃ 那组的候选是 40 vs 60 ms(差一整帧)⇒ 咬得到。
const GONE_SORT_EPS_MS: f32 = 15.0;

/// 窗**尾**落在一个唱音里、而且离那个音的终点还有这么多毫秒以上时,淡出拉长
/// (见 [`tied_xfade_ms`])。⛔ 200 ms 的理由:短于它的话「后面那一截 base」本来就快结束了,
/// 拉长淡出反而会把两个音混在一起。用户点名的 akiko 2:50.4 那一处剩 **1.42 s**。
const TAIL_ROOM_MS: u32 = 200;

/// ⚙ 出厂默认 = 15.0(**dB**,见 [`HANDOVER_DEFICIT_DB_DEFAULT`])。
/// `UTAI_RANGE_HANDOVER=0` 关掉 ⇒ 交接点逐帧回到今天。
///
/// ## ⭐⭐⭐ **拼接器不许把输出交给一条【正在静音】的 donor**
/// 用户 2026-08-26 深夜:「yuyuko 4:49 这次没炸,但它的下一个音 **4:50.694** 反而炸了 ——
/// 别跟我扯模型达不到,这个之前甚至能达到」。**他是对的,而且这条比 4:49 更普遍。**
///
/// 四层剖面(同一次 run 的转储;窗 `−11` 覆盖到 290.76,窗 `−9` 从 290.68 开始,重叠 80 ms):
///
/// | 时刻 | base | donor −11(**出去的**)| donor −9(**进来的**)| 成品 |
/// |---|---|---|---|---|
/// | 290.690 | −39.4 | **−22.5** | −49.9 | −58.4 |
/// | 290.730 | −36.6 | **−22.8** | **−67.9** | **−72.6** |
/// | 290.760 | −34.6 | **−13.2** | −48.2 | −46.0 |
///
/// ⇒ **出去的那条一路是好的,进来的那条自己有 60-80 ms 的静音起头,而交接照做不误。**
///
/// ## ⛔ 它和「短停顿伪影」是**同一个**缺陷
/// 同一把尺子扫四条臂(缝数 / 命中):yuyuko 54/**1**(= 290.680)·
/// yachiyo 90/**5**(= **229.500 · 238.140 · 246.800**,正是用户点名的 3:49.524 / 3:58.162 那一族)·
/// akiko 51/0 · 东雪莲 49/0。
/// ⇒ S163 为薄片族试过的**四种闭合方式全部判负**(左吞 / 中点 / 右吞 / `join_rests`),
///    原因现在清楚了:**它们改的是「在哪儿交接」的几何,而病灶是「交给谁」**。
///
/// ## 为什么延后交接是安全的(不是拿音高换)
/// ⭐ `donor_post` **每一遍都在目标音高上**(逆变换已经做完)⇒ 两条 donor 的差别是
/// **落点带来的音色**,不是音高。多用出去的那一条 ≤[`HANDOVER_MAX_MS`],
/// 换来的是「不掉进 60-80 ms 的静音」。
/// ⛔ 而且它**只在真的有缺口时动手**:没缺口 ⇒ 交接点逐样本不变。
pub fn handover_deficit_db() -> f32 {
    parse_handover(std::env::var("UTAI_RANGE_HANDOVER").ok().as_deref())
}

fn parse_handover(v: Option<&str>) -> f32 {
    v.and_then(|x| x.trim().parse::<f32>().ok())
        .filter(|t| t.is_finite() && (0.0..=60.0).contains(t))
        .unwrap_or(HANDOVER_DEFICIT_DB_DEFAULT)
}

/// ⚙ 出厂默认。见 [`handover_deficit_db`]。
const HANDOVER_DEFICIT_DB_DEFAULT: f32 = 15.0;

/// ⚙ 出厂默认 = 1.5 —— dB；`UTAI_RANGE_HANDOVER_GAIN=0` 退回纯门限逻辑。
/// 出厂值的出处见 [`HANDOVER_GAIN_DB_DEFAULT`]。
///
/// ## ⛔ 它替掉的是什么:一个**照着靶子调出来的**绝对门限
/// [`handover_deficit_db`] 问的是「进来那条比出去那条弱了多少 dB」,答案要跟一个**常量**比
/// (出厂 15)。而这个常量没有任何自校准的成分 —— S165 为了咬住用户点名的两处,
/// 它被一路往下试(15 → 12 → 6),**每一格都是照着那两个靶子调的**。
/// 用户 2026-08-28 当场点破:「那你现在不又是照着单个靶子胡乱调么」。**他是对的**:
/// 换个模型、换首歌,同一个 dB 数没有任何道理。
///
/// ## ⭐ 收益驱动:问「延后能不能让淡化区更接近【两条里较好的那条】」
/// 对每个候选交接点 `t`,在这条缝**真正的淡化区**上算
/// `mix_deficit(t) = max(出去, 进来) − 等增益混合后的电平`(越小越好);
/// 取 `mix_deficit` 最小的 `t`,**收益 = mix_deficit(t0) − mix_deficit(t)**。
/// 收益 ≥ 这个旋钮的 dB 数才动手 ⇒ **没有「弱多少算弱」的常量,只有「改善多少才值得」**。
///
/// ⚠ 混合电平按**不相干相加**估(`(Pl+Pr)/3` 是等增益淡化在 w∈[0,1] 上的平均功率)——
/// 实测这两条 donor 的相干度只有 0.15-0.32,接近不相干,这个近似是站得住的。
///
/// ## 出厂值的出处
/// 建议值 **2.7 dB** = 这条线上量过的**可闻阈**(S148 承重一组:~2.7 dB 听得出 / ≤0.46 dB 听不出)
/// ⇒ 「只在听得出的地方动手」,而不是「弱到某个绝对值就动手」。
/// ⭐ S165 翻默认 = **1.5**(用户耳判通过)。⚠ 跨模型只验过 yuyuko / yachiyo 两个 RVC 模型 ——
///   akiko 与东雪莲在这个仓里只有 sovits 版,那条链没走过。
pub fn handover_gain_db() -> f32 {
    std::env::var("UTAI_RANGE_HANDOVER_GAIN")
        .ok()
        .and_then(|x| x.trim().parse::<f32>().ok())
        .filter(|t| t.is_finite() && (0.0..=30.0).contains(t))
        .unwrap_or(HANDOVER_GAIN_DB_DEFAULT)
}

/// ⚙ 出厂默认 = 1.5 —— dB。
///
/// ## ⛔ 这个数是怎么来的 —— **不是照靶子调的**
/// 用户 2026-08-28 点破上一版:「那你现在不又是照着单个靶子胡乱调么」——
/// 那时 [`handover_deficit_db`] 被一路试(15 → 12 → 6),每一格都在追一个坐标。
/// 这一版把判据换成收益驱动之后,只剩「改善多少才值得动手」这一个数,而它由
/// **三个耳判过的点 + 剂量拐点**共同夹出来(全曲 29 条跨 shift 缝):
/// ```text
/// 290.790 收益 2.99  用户没抱怨(修了也没变坏)
/// 276.320 收益 2.10  4:36 —— 用户确认「基本上真好了」
/// 269.250 收益 1.66  4:29 —— 用户听得见「哑了一下」
///
/// 门槛 2.7 ⇒ 命中 2 条(只覆盖 290.68)
/// 门槛 2.0 ⇒ 命中 6 条(差 4:29)
/// 门槛 1.5 ⇒ 命中 9 条  ← 三个耳判点全覆盖
/// 门槛 1.2 ⇒ 命中 18 条 ← 波及面翻倍
/// ```
/// ⇒ **下限由「必须咬住 4:29 的 1.66」定,上限由「再降一格翻倍」定,1.5 是唯一窗口。**
///
/// ## ⚠ 已知代价(必须一起记住)
/// 挪 4 条缝(门限那版只挪 2 条)之后,**没被碰的 700 个音在音头/音尾口径上统计可分**:
/// MWU **z = +10.49 / +6.26**(上一版 +0.48 / +0.95)。拆开看是 **p50 被顶高**
/// (音头 0.007 → 0.014)**而不是尾部**(max 4.33 → 4.34 几乎重合);
/// 变轻 >1 dB 音头 6 / 音尾 24,变响 >0.5 dB 13 / 37(**变响的比变轻的多**)。
/// ⇒ 取舍是「多修两处靶子 vs 全曲多一层 0.01 dB 量级的抖动」,由**耳朵**拍板,不是仪器。
const HANDOVER_GAIN_DB_DEFAULT: f32 = 1.5;

/// 收益驱动时的**粗筛**:落差连这个都够不到就不必算收益(纯省开销,不决定结果)。
/// 取 3.0 dB —— 略高于可闻阈 2.7,再小的落差不可能产生 ≥2.7 dB 的收益。
const HANDOVER_COARSE_DB: f32 = 3.0;

/// ⚙ 出厂默认 = **true(开)** —— `UTAI_RANGE_HANDOVER_FADEWIN=0` 关掉。
/// ⭐ S165 翻默认:用户 2026-08-28 耳判「4:36 基本上真好了」;4:29 也从 −17.61 抬到 −9.19。
///
/// ## ⭐⭐⭐ 交接点体检**看错了那一段**
/// [`defer_dead_handover`] 的评估窗一直是「交接点**往后** [`HANDOVER_WIN_MS`](40 ms)」,
/// 而 [`tied_xfade_ms`] 那一族把右窗的淡入起点**往前拉 120 ms** ——
/// **两段完全不重叠**,体检结构上看不见自己该拦的那一段。
///
/// 实测(用户 2026-08-28 点名 yuyuko × 炉心 **4:36.319**「接缝不干净 / f0 附近多出面状」):
/// ```text
/// 往后 40 ms(今天看的)  : 出去 −10.11 / 进来 −15.39 ⇒ 落差 **5.27 dB**,门限 15 够不着
/// 往前 120 ms(真淡入区): 出去  −8.56 / 进来 −21.11 ⇒ 落差 **12.55 dB**
/// 成品实测 −14.75,而等增益淡化在 w=0.5 的功率 ≈ 0.25·Pa = **−14.58** ⇒ 几乎完全吻合
/// ```
/// ⇒ 凹陷就是「把输出交给一条弱 12.55 dB 的 donor」的必然结果,不是相位相消
///   (相位那条已经查过并收回,见 [`seam_align_wide`])。
///
/// ## ⛔⛔ 它是**额外**看一段,不是**替换**
/// 第一版写成「tied 缝改用往前的窗」,实测当场退化:`290.680` 那一处
/// (本 doc 自己举的例子,往后 40 ms 落差 27-45 dB)在关着时被拦住并挪后 110 ms,
/// 而开着**反而不触发了** —— 因为往前那 120 ms 里进来的那条并不弱。
/// ⇒ 现在两个窗都算,**取缺口最大的那个**,并用它决定往后找恢复点时看哪一段。
///
/// ⛔ 出厂**关**,等验收:这一刀会让 handover 在 tied 缝上更容易触发,而延后交接的代价
///   (出去那条多唱一截,它的落点音色不是为下一个音选的)必须先量过。
/// ⚠ 已知它**单靠自己还够不着靶子**:用户点名的 4:36.319 在淡入区上的落差是 **12.55 dB**,
///   而 [`HANDOVER_DEFICIT_DB_DEFAULT`] 是 **15** ⇒ 窗改对了、门限仍够不着。
///   降门限是另一件事,必须单独量(它会波及所有接缝)。
pub fn handover_fade_window() -> bool {
    !matches!(
        std::env::var("UTAI_RANGE_HANDOVER_FADEWIN").ok().as_deref().map(str::trim),
        Some("0") | Some("off") | Some("false") | Some("no")
    )
}

/// ⚙ 出厂默认 = 120.0(**毫秒**,见 [`TIED_XFADE_MS_DEFAULT`])。`UTAI_RANGE_TIED_XFADE=0`
/// 关掉 ⇒ 所有接缝都用 10 ms,逐位回到今天。
///
/// ## ⭐⭐ **同一个长音内部的接缝,交叉淡化拉长**
/// 用户 2026-08-26:「我实在受不了那种**一个长音三个听感**,或者**长音割裂**了」。
/// 一个持续的长音在谱面上是好几个连着的**同词**音符(炉心融解结尾的「あ」是 `[796..=802]`
/// 共 7 个),而计划器按**死音**分组 ⇒ 同一个长音被切成 2-4 段、每段一个落点,
/// 段与段之间只有 **10 ms** 的交叉淡化。
///
/// ⭐ 两条 donor 的**音高都是目标音高**(逆变换已经做完)⇒ 它们的差别只是落点带来的
/// **音色**。在一个持续的元音上把音色**慢慢**换过去是听不出缝的;10 ms 换才是「割裂」。
///
/// ## ⛔ 为什么只在【延续】处拉长
/// S162 的 2×2(只有同一类边界上的跨组 vs 同组才是有效对照):
/// **长音延续** 跨组比同组贵 **电平 +1.10 dB · 谱形状 +0.51**;
/// **音节变化** 只有 +0.13 / +0.05 ≈ 零。
/// ⇒ 代价几乎全在长音内部;而音节边界上有**辅音与起音**,拉长淡化会把它糊掉 ⇒ 一个字节不动。
///
/// ⚠ 上限还会被两侧窗自己的长度夹住(`xf.min((b − a) / 2)`),所以短窗上它自动退化。
pub fn tied_xfade_ms() -> f64 {
    parse_tied_xfade(std::env::var("UTAI_RANGE_TIED_XFADE").ok().as_deref())
}

/// ⚙ 出厂默认 = true —— `UTAI_RANGE_SEAM_RAMP=0` 关掉。**短间隙/重叠之后的窗淡入拉长**。
/// 剂量曲线与靶子见 [`SEAM_RAMP_MS`]。
pub fn seam_ramp_enabled() -> bool {
    !matches!(
        std::env::var("UTAI_RANGE_SEAM_RAMP").ok().as_deref().map(str::trim),
        Some("0") | Some("off") | Some("false") | Some("no")
    )
}

fn parse_tied_xfade(v: Option<&str>) -> f64 {
    v.and_then(|x| x.trim().parse::<f64>().ok())
        .filter(|t| t.is_finite() && (0.0..=400.0).contains(t))
        .unwrap_or(TIED_XFADE_MS_DEFAULT)
}

/// ⚙ 出厂默认。见 [`tied_xfade_ms`]。
const TIED_XFADE_MS_DEFAULT: f64 = 120.0;

/// ⭐⭐⭐⭐⭐ S163 v16 —— **短间隙/重叠之后的窗，淡入拉到这么宽**(ms)。
///
/// # 靶子(workflow `wf_70155c32-c0c`，四臂同 run 零噪声，11 agent / 234 次工具调用)
/// ```text
/// 短间隙族:间隙 ≤200 ms **且移调改变** = 97 个 / **1.376% 超 4 dB** / max 21.96 dB
///           对照(>200 ms 且移调改变)195,809 格只有 **0.005%**  ⇒ **富集 ×275**
///   ⭐ 排最前的三个正是用户 2026-08-25 报的坐标(agent **完全不知情**):
///   `3:49.528 +12.21`(用户报 3:49.524)· `3:58.176 +11.54`(3:58.162)· `3:40.860 +7.86`(3:40.829)
/// 重叠类:339 个窗过渡里 62 个 `gap<0`,负 gap **只有一个取值:恰好 −80 ms**;
///         这类 ramp p50=0.0 ms、缝热率 **36.0%**(TIGHT 11.7% / MEDIUM 6.6%)
/// ```
///
/// # 剂量曲线(这把刀的全部依据 —— 单调，而且 p 值很硬)
/// ```text
/// 缝上超额>4 dB 的比率，按 ramp(|nodip−base| 升到内部电平 50% 所需时间)分箱:
///   **0-2 ms 34.9%**(n=62) / 2-5 ms 24.5% / 5-15 ms 29.4% / 15-30 ms 19.8% / **≥30 ms 5.9%**
///   ramp<5 vs ≥30:31.3% vs 6.2%，MWU **p=1.2e-14**
///   **Spearman(ramp, seam_dB) = −0.471, p=2.2e-20**
/// ```
/// ⇒ **donor 进得越慢，缝越不响。** 40 ms 落在「≥30 ms」那一档里，留了余量。
///
/// # ⛔ 为什么是拉长淡入，不是别的
/// 三条硬约束夹着:**不许动 donor 选择**(用户令)、**不许切内容**
/// (往窗里插别的内容两端必然造缝,S163 栽过三次)、**动窗边界副作用波及 98 秒**(v12 判负)。
/// 而淡化**本身就是增益** ⇒ 这一刀天然落在唯一安全的那一层。
const SEAM_RAMP_MS: f64 = 0.0;

// ⛔⛔ S163 v16 判负并关闭（用户 2026-08-28 实听：「这个确实可以关了，它确实和硬无关」）。
//    它做的事其实是**把起音整体延后 30 ms**，不是「消除咔哒」：
//    对齐后逐点 Δ 是**单调递增**的（−0.5 → −12.1），而整段只差 0.15-1.33 dB
//    ⇒ 听感上就是「起音慢了」。120 ms 极端剂量（起音压低 15-30 dB）用户**仍然听不出**
//    ⇒ **这根轴与用户说的「硬」无关。**
//    ⚠ 上面那些靶子数据（短间隙族 ×275、ramp 剂量曲线）仍然成立，
//      只是「拉长淡入」这个**做法**解决不了「硬」。
/// 交接最多往后挪这么久(ms)。⛔ 上限的理由:再久就等于让出去的那条 donor 唱下一个音
/// 的一大截,而它的**落点**(音色)不是为那个音选的。
/// ⚠ 它远小于 [`MERGE_BRIDGE_FRAMES`](25 帧 = 500 ms)留出来的片段余量 ⇒ 结构上够得着。
const HANDOVER_MAX_MS: f64 = 150.0;
/// 出去的那条至少要有这么响,才算「手上有好料」——否则两边都是安静的休止,不关这一刀的事。
const HANDOVER_ALIVE_DBFS: f32 = -40.0;
/// 评估窗(ms):在交接点往后这么长的一段上比两条 donor 的电平。
const HANDOVER_WIN_MS: f64 = 40.0;
/// 塌陷的**绝对**地板 —— 低于它才算「唱没了」,而不是一个辅音凹陷。
const REPAIR_FLOOR_DBFS: f32 = -50.0;
/// 同时还要比**该音自己的中位**低这么多 —— 挡掉「整个音本来就很轻」那一类。
const REPAIR_REL_DB: f32 = 25.0;

/// ⭐⭐⭐⭐ S165 —— 一个音**内部**的**电平跌幅**(dB):该音的稳态(p75)减去音内最低的 20 ms 窗。
///
/// # ⛔ 它为什么必须存在
/// 失配轴([`landing_mismatch_eps`])的对手轴闸今天用的是 [`CandScore::worst_dim_vs`]
/// (**上方谐波电平** `uplev`)—— 那挡的是「**变闷**」,**挡不住「把谷挖得更深**」。
///
/// 实测 4:07.466(`[696]あ`,用户 2026-08-29 连着两轮点名的那一处):
/// | 臂 | 音内跌幅 | 谷底的谱平坦度 | 听感 |
/// |---|---|---|---|
/// | `base`(未救援) | **6.5 dB** | −16.5 | 连着 |
/// | 出厂 `mism=0` | **16.0 dB** | −14.2 | **哑噪声** |
/// | `mism=1.2`(Q1/R1/PJ 三条臂) | **21.4 / 21.5 / 21.4 dB** | −33.1 | **断音** |
/// ⇒ 失配轴治好了 4:36(失配 2.82 → 1.68),**却把 4:07 的谷又挖深了 5.4 dB**,
///   而 `uplev` 那个闸**一次都没拦住** —— 因为两个候选的上方谐波电平差不多。
///
/// # ⭐ 用户给的定义(2026-08-29)
/// 「那里『是谷』不止可能像现在这样吐**噪声**,还可能像之前 Q1 那一遍一样
///   **完全挖出来什么也没有(断音)**」「它是有点地方没噪声,但**有的地方它什么都没有**」
///   「换句话说,**你得找它『确实连着的』位置**啊」
/// ⇒ ⭐⭐ **两种坏法的共同点就是「电平掉了」** —— 噪声那一种平坦度高、断音那一种平坦度低,
///   **任何只盯一种形态的尺子都会漏掉另一种**;而「掉了多少」两种都抓得到。
/// ⇒ 所以闸架在**这根**轴上,不架在平坦度上。
///
/// # ⚠ 口径
/// 稳态取 **p75** 而不是中位:密集救援的音里中位本身可能已经被谷拉低。
/// 逐 **20 ms** 窗、hop **5 ms**(比 [`silence_run_ms`] 的 20 ms 格细,因为要抓 60 ms 宽的谷)。
/// 少于 5 个窗 ⇒ `None`(测不出稳态)。
fn note_dip_db(seg: &[f32], sample_rate: u32) -> Option<f32> {
    let win = (sample_rate as usize / 50).max(16); // 20 ms
    let hop = (win / 4).max(4); // 5 ms
    if seg.len() < win + 4 * hop {
        return None;
    }
    let mut lv: Vec<f32> = Vec::new();
    let mut i = 0usize;
    while i + win <= seg.len() {
        let e: f64 =
            seg[i..i + win].iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / win as f64;
        lv.push((10.0 * (e + 1e-20).log10()) as f32);
        i += hop;
    }
    if lv.len() < 5 {
        return None;
    }
    let mut sorted = lv.clone();
    sorted.sort_by(f32::total_cmp);
    let p75 = sorted[(sorted.len() * 3) / 4];
    let lo = sorted[0];
    Some(p75 - lo)
}

/// 一个音**内部**最长的一段「唱没了」有多少毫秒(0 = 没有)。
///
/// ⛔ 逐 20 ms 格;两条门必须**同时**满足(绝对地板 + 相对该音中位),理由在
/// [`landing_repair_ms`] 的 doc 里 —— 只用相对量会被辅音污染,只用绝对量会被轻音污染。
fn silence_run_ms(seg: &[f32], sample_rate: u32) -> f32 {
    let cell = (sample_rate as usize / 50).max(16); // 20 ms
    let k = seg.len() / cell;
    if k < 4 {
        return 0.0;
    }
    let mut lv: Vec<f32> = Vec::with_capacity(k);
    for i in 0..k {
        let s = &seg[i * cell..(i + 1) * cell];
        let e: f64 = s.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / cell as f64;
        lv.push((10.0 * (e + 1e-20).log10()) as f32);
    }
    let mut sorted = lv.clone();
    sorted.sort_by(f32::total_cmp);
    let med = sorted[sorted.len() / 2];
    let (mut run, mut best) = (0usize, 0usize);
    for &v in &lv {
        if v < REPAIR_FLOOR_DBFS && v < med - REPAIR_REL_DB {
            run += 1;
            best = best.max(run);
        } else {
            run = 0;
        }
    }
    best as f32 * 20.0
}

/// ⭐ S163 —— 一个落点候选在**一个窗**里的逐音读数。
///
/// ⛔ 为什么是**逐音**而不是整窗一个数:一个窗里可能有 7 个音,而只有 1 个塌了。
/// 实测 4:09.478「み」只占整窗 **180 ms / 1.4 s** ⇒ 整窗 RMS 结构上看不见它
/// (那一处生产读到的整窗 |rel| 是 0.31 vs 1.51 —— **塌掉的那个反而赢**)。
#[derive(Default, Clone)]
struct CandScore {
    /// (音下标, |电平 − 干净邻居中位| dB)。只含**取得到参照**的音。
    rel: Vec<(usize, f32)>,
    /// (音下标, 谐波能量占比 dB)。**不需要参照**。
    harm: Vec<(usize, f32)>,
    /// ⭐ S163 —— (音下标, 音内最长「唱没了」的毫秒数)。**不需要参照**。见 [`silence_run_ms`]。
    gone: Vec<(usize, f32)>,
    /// ⭐⭐⭐⭐ S165 —— (音下标, **音内电平跌幅** dB)。**不需要参照**。见 [`note_dip_db`]。
    /// ⛔ 它只当**失配轴的对手轴闸**,不当排序键 —— 它的靶子是「换落点把谷挖得更深」那一类交换。
    dip: Vec<(usize, f32)>,
    /// ⭐ S163 —— (音下标, 谐波梳深 dB)。**不需要参照**,而且与谐波占比**测的不是一件事**:
    /// 用户点名的「卡痰」那个音梳深 **−0.4**(谐波间被填满)而谐波占比 −0.10(看起来没问题)。
    comb: Vec<(usize, f32)>,
    /// ⭐⭐⭐ S163 —— (音下标, 谐波谱峰宽度 % of f0)。**不需要参照**，
    /// 而且与上面三根都不是一件事：它量的是**每一根谐波自己糊不糊**。
    /// 实测（yuyuko 4:36 接缝两侧）：峰宽 **12.33 vs 0.99（差 12 倍）**，
    /// 而填充度读到 −18 vs −29 —— **方向还反了**。
    /// ⭐ 短音的 `donor_pre` 峰宽按落点：**77 → 3.50 而 78 → 11.88**（3.4 倍，只差 1 个半音）。
    width: Vec<(usize, f32)>,
    /// ⭐⭐⭐ S163 —— (音下标, **音内塌陷** dB;负 = 中段比音头轻)。
    ///
    /// **不需要参照,而且与电平无关** —— 它是音**自己**的中段比自己的音头低多少,
    /// 所以既不吃 `clean_neighbour_refs` 那套「密集救援处参照是休止」的亏,
    /// 也不会变成一把「谁轻谁赢」的尺子(S163 §7 栽过一次)。
    ///
    /// ⭐ 靶子来自用户 2026-08-27 点名的 4:07.470-4:08.046「中间塌缩」,
    /// 对照 0:40.700-0:41.482 —— **同 midi(73+7=80)、同歌词「あ」、同副歌、同 shift −4**:
    ///
    /// | | 缺陷 4:07.5 | 对照 0:40.7 |
    /// |---|---|---|
    /// | `base` | −10.9 | −13.5 |
    /// | **shift −4(今天用的)** | **−5.9** | **−1.4** |
    /// | shift −7 | **−1.6** | −1.4 |
    /// | shift −9 | **−0.2** | −1.0 |
    ///
    /// ⇒ **同一个 shift,在两个上下文里 donor 的行为相反** —— 模型在副歌尾把 midi 76
    /// 唱塌了,而这不是我们任何一层干的(诊断日志证明写入区间对称、`UTAI_RANGE_LEVEL_MATCH=0`
    /// 的对照臂逐 20 ms 相同、`donor_f0` 两处完全一致)。**但候选里有现成的好格。**
    ///
    /// ⛔ **它不是通用排序刀**:多候选音上 akiko 只有 3/23 塌 >4 dB,其中 0 个能靠换候选
    /// 改善 >2.7 dB(可闻刻度)。它是给**乘客音 + 单候选组**用的 —— 那正是修补遍的地盘。
    /// ⭐⭐⭐⭐ S163 —— (音下标, **上方谐波在音内被抽干多少** dB;负 = 中段比音头弱)。
    ///
    /// 见 [`utai_dsp::harmonicity::upper_harmonic_sag_db`]。用户 2026-08-27 把两件事合成一件:
    /// 「ぴゃ那里……**也是中间塌缩了**」「**那个音的中间电平弱就是差在了上方谐波上**」。
    ///
    /// ⭐ 它是本场**第一根在用户点名的五个坐标上符号就分开**的标量轴:
    /// ```text
    /// ★ぴゃ   243.107  今天 −13: −6.1     好档 −14: +1.7
    /// ★あ    4:07.467 今天  −4: −3.2     好档 −7/−9: +1.9
    /// ★ら    4:10.187 今天  −7: −10.4    好档  −4: −3.8
    ///  对照あ 0:40.901 今天  −4: +3.3     ← 今天就已经是好的
    ///  对照と 0:43.641 今天  −7: +1.4     ← 今天就已经是好的
    /// ```
    /// ⛔ 而今天唯一的排序键 `rel` 在同一批点上**方向是反的**。
    usag: Vec<(usize, f32)>,
    /// ⭐⭐⭐ S165 —— (音下标, **第二谐波 `2·f0` 相对 `f0` 的电平** dB)。
    ///
    /// 见 [`utai_dsp::harmonicity::second_harmonic_level_db`]。用户 2026-08-28 指着三张频谱图说
    /// yachiyo 的谐波线「**非常小段的忽明忽暗、不是一条干净的连续线**」,断得最明显的是
    /// 「**基频上面那条**」= `2·f0`;而 yuyuko 与 SV 在**同一个音**上都正常:
    ///
    /// | 目标音(炉心 `[794]あ` f0 987.8) | `2·f0` 相对 `f0` |
    /// |---|---|
    /// | yachiyo(用户说油) | **−17.0 dB** |
    /// | yuyuko(用户说好) | −5.2 |
    /// | SV(用户说好) | +0.9 |
    ///
    /// ⇒ ⭐ **「忽明忽暗」是低电平的后果不是原因**:一根本来就暗 12-18 dB 的线叠上**正常**的
    /// 抖动(yachiyo 的抖动是它自己全曲的 43 百分位),在频谱图上就是断续的。
    /// 本场为此先造了**八把**量「沿时间变化」的尺子,**全部读反或分不开**。
    ///
    /// ⛔ 为什么 [`CandScore::uplev`] 看不见它:那一根量 `2..8·f0` 的**整体**,
    /// **`2·f0` 那 12 dB 的缺口被 7 根谐波一平均只剩 ~1.7 dB**。
    ///
    /// ✅ 用户 2026-08-28 耳判确认(拿 `donor_post_-13` 整窗替换的探针臂):
    /// 「**f1『强』/『实』确实在听感上更好,即使它把 f4 炸了**」——
    /// 因为 `2·f0`=1975 Hz 落在耳朵最敏感的区间,而 `5·f0`=4939 Hz 感知权重低得多。
    h2: Vec<(usize, f32)>,
    /// ⭐⭐⭐⭐ S165 —— (音下标, **失配度** dB) = 逐根谐波 `抖动 − 该电平该有的抖动` 的**最大值**。
    ///
    /// 见 [`note_mismatch`] 与 [`mismatch_reference`]。用户 2026-08-28 定案:
    /// 「本身也不是说『**响就是好**』或者『**抖就是不好**』…… **失配了才奇怪**」。
    /// ⇒ 本场判负的十几把尺子**要么只量响度、要么只量抖动**,没有一把量两者的关系。
    ///
    /// ⛔ **「最差」不是「平均」**——这一条搞反整把刀会反向(实测四种口径见 [`note_mismatch`] 的表:
    /// 三种「整体距离」全都选中用户判为最差的那一档)。
    mism: Vec<(usize, f32)>,
    /// ⭐⭐⭐⭐ S165 —— (音下标, **音高误差** 音分)。见 [`utai_dsp::harmonicity::pitch_error_cents`]。
    ///
    /// ⛔ 在此之前,`decide_group` 的**所有**轴都不检查「这个候选唱的是不是目标音高」。
    /// 用户 2026-08-29 听出的灾难:「ぴゃ」(目标 **1480 Hz**)在成品里 **f0 掉到 320 Hz**
    /// (≈ −27 个半音)、RMS −13.05 dB、梳深 61.5 → 14.8 —— **不是变哑,是唱错音**。
    /// ⚠ 而塌掉的音在别的轴上**未必难看**:**音高塌了之后谐波反而更「协调」**,
    /// 它甚至能在失配轴上得高分。
    pitch: Vec<(usize, f32)>,
    /// ⭐⭐⭐ S163 —— (音下标, **上方谐波的绝对强度** dB:`2..8·f0` 相对 `0.7..1.6·f0`)。
    ///
    /// ⛔ 它是 [`CandScore::usag`] 的**对手轴**:`usag` 量「稳不稳」,它量「强不强」。
    /// 只看 `usag` 会换来一个**平但闷**的档 —— 实测 `[687]く` 换档后 `usag` 只买到
    /// **+0.98** 却把上方谐波压掉 **6.25 dB**,而记忆 §11 早记过同一个方向
    /// (akiko ぴゃ 的 −14 上方谐波比 −13 弱 6.6 dB)。
    ///
    /// ⚠ 它**不参与排序**,只在 [`landing_usag_dim_cap`] 那道闸上用:
    /// 换档之前查一次「新档比当前档闷多少」,超了就不换。
    uplev: Vec<(usize, f32)>,
}

impl CandScore {
    /// 组的成绩 = **组内最差的那个音**(S162 oracle 用的就是这个口径:
    /// 一个塌掉的音就毁了一整组,平均值会把它稀释掉)。
    fn worst_rel(&self) -> Option<f32> {
        self.rel.iter().map(|&(_, v)| v).fold(None, |a: Option<f32>, v| {
            Some(a.map_or(v, |x: f32| x.max(v)))
        })
    }

    fn harm_of(&self, i: usize) -> Option<f32> {
        self.harm.iter().find(|&&(k, _)| k == i).map(|&(_, v)| v)
    }

    /// 这一组里**最长**的那一段「唱没了」。
    fn worst_gone(&self) -> f32 {
        self.gone.iter().map(|&(_, v)| v).fold(0.0f32, f32::max)
    }

    /// 这一组里**最糊**的那个音的梳深(没量到 ⇒ `INFINITY`,即「没有证据说它糊」)。
    /// ⭐ S163 —— 组内**最糊**的那一个音的峰宽（越大越差）。
    fn worst_width(&self) -> f32 {
        self.width.iter().map(|&(_, w)| w).fold(0.0f32, f32::max)
    }

    /// ⭐ S163 —— 组内**上方谐波塌得最狠**的那个音(最负的)。没量到 ⇒ `0.0`。
    ///
    /// ⛔ 取最差而不是平均:一个塌掉的音就毁了一整组,平均值会把它稀释掉
    /// (S162 的 oracle 用的就是这个口径)。
    fn worst_usag(&self) -> f32 {
        self.usag.iter().map(|&(_, v)| v).fold(0.0f32, f32::min)
    }

    /// ⭐⭐⭐⭐ S165 —— 组内**音高错得最离谱**的那个音(绝对音分)。没量到 ⇒ `0.0`
    /// (= 没有证据说它坏;⛔ 不许当成「坏」,否则量不到就全被否决了)。
    fn worst_pitch_err(&self) -> f32 {
        self.pitch.iter().map(|&(_, v)| v.abs()).fold(0.0f32, f32::max)
    }

    /// ⭐⭐⭐ S165 —— 组内**失配最严重**的那个音。没量到 ⇒ `NEG_INFINITY`
    /// (= 没有证据说它坏,而且在「取最好」的比较里永远输)。
    ///
    /// ⛔ 与 [`CandScore::worst_h2`] 同一个理由取最差:**一个失配的音就毁了一整组**。
    fn worst_mism(&self) -> f32 {
        self.mism.iter().map(|&(_, v)| v).fold(f32::NEG_INFINITY, f32::max)
    }

    /// ⭐ S165 —— 某个音上的失配度。
    fn mism_of(&self, i: usize) -> Option<f32> {
        self.mism.iter().find(|&&(j, _)| j == i).map(|&(_, v)| v)
    }

    /// ⭐⭐⭐ S165 —— `self` 相对 `other` 在**任何一个音**上的最大失配**改善**(正 = self 更好)。
    ///
    /// ⛔⛔ **排序不用它** —— 第一版用了,被判据抓住:它让「在某一根上赢得最多」的候选获胜,
    /// 而决定听感的是**最差的那个音**(见排序分支里的注释)。留着是给诊断用的。
    #[allow(dead_code)]
    ///
    /// ⛔ 逐音配对,和 [`CandScore::best_h2_vs`] 同款理由:这根轴是来救「**某一个音**失配」的。
    /// ⚠ 而且这是**相对判据** —— 只在「存在明显更好的候选」时才动,
    /// 绝对门限会把「当前虽差但没有更好替代」也否掉(实测那样会否掉 41% 的候选而只有 10 组能改善)。
    fn best_mism_vs(&self, other: &CandScore) -> f32 {
        self.mism
            .iter()
            .filter_map(|&(i, mine)| other.mism_of(i).map(|theirs| theirs - mine))
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// ⭐ S165 —— 组内**第二谐波最弱**的那个音(最负的)。没量到 ⇒ `0.0`(= 没有证据说它坏)。
    ///
    /// ⛔ 取最差而不是平均 —— 与 [`CandScore::worst_usag`] 同一个理由。
    fn worst_h2(&self) -> f32 {
        self.h2.iter().map(|&(_, v)| v).fold(0.0f32, f32::min)
    }

    /// ⭐ S165 —— 某个音上的第二谐波电平。
    fn h2_of(&self, i: usize) -> Option<f32> {
        self.h2.iter().find(|&&(j, _)| j == i).map(|&(_, v)| v)
    }

    /// ⭐⭐ S165 —— `self` 相对 `other` 在**任何一个音**上的最大 `2·f0` 增益(dB,正 = 更实)。
    ///
    /// ⛔ 逐音取**最好**(不是最差):这根轴是来救「某一个音被掏空」的,
    /// 组里其它音本来就好 ⇒ 取最差会被它们摁死。
    fn best_h2_vs(&self, other: &CandScore) -> f32 {
        self.h2
            .iter()
            .filter_map(|&(i, mine)| other.h2_of(i).map(|theirs| mine - theirs))
            .fold(0.0f32, f32::max)
    }

    /// ⭐ S163 —— 某个音上的上方谐波强度。
    fn uplev_of(&self, i: usize) -> Option<f32> {
        self.uplev.iter().find(|&&(j, _)| j == i).map(|&(_, v)| v)
    }

    /// ⭐⭐⭐ S163 —— `self` 相对 `other` 在**任何一个音**上的最大强度降幅(dB,正 = 更闷)。
    ///
    /// ⛔⛔ 第一版写成「组内中位」,**实测被打脸**:`[687]く` 换档后上方谐波
    /// **4.40 → −1.85 = 闷 6.25 dB**,而组内其它音没那么闷 ⇒ 中位没超门限 ⇒ 放行,
    /// 那个净亏的换档在两轮渲染里都照做不误。
    /// ⇒ **代价不是「整组一起付」的平均量,是「哪个音被牺牲了」** ⇒ 必须逐音取最差。
    fn worst_dim_vs(&self, other: &CandScore) -> f32 {
        self.uplev
            .iter()
            .filter_map(|&(i, mine)| other.uplev_of(i).map(|theirs| theirs - mine))
            .fold(0.0f32, f32::max)
    }

    /// ⭐⭐⭐⭐ S165 —— 赢家相对输家**把谷挖深了多少 dB**(正 = 赢家更深 = 更糟)。
    /// 与 [`Self::worst_dim_vs`] 同构:只在**两边都量到了同一个音**时比较,取最差。
    fn worst_dip_vs(&self, other: &CandScore) -> f32 {
        self.dip
            .iter()
            .filter_map(|&(i, mine)| {
                other.dip.iter().find(|&&(j, _)| j == i).map(|&(_, theirs)| mine - theirs)
            })
            .fold(0.0f32, f32::max)
    }

    fn worst_comb(&self) -> f32 {
        self.comb.iter().map(|&(_, v)| v).fold(f32::INFINITY, f32::min)
    }
}

/// 乐句内跨组对齐:**压**的上限(dB)。见 [`match_phrase_group_levels`]。
const PHRASE_LEVEL_CUT_DB: f32 = 3.0;
/// 乐句内跨组对齐:**抬**的上限(dB)。⛔ 比压小 —— 抬会把面状伪影一起抬起来。
const PHRASE_LEVEL_LIFT_DB: f32 = 2.0;
/// 一段至少要有这么多帧的**可测稳态**才参与(50 fps ⇒ 15 帧 = 0.30 s)。
/// ⛔ 没有它时实测出现过 **20.25 dB** 的假增益(一两个短音估出来的)。
const PHRASE_LEVEL_MIN_SUSTAIN: i64 = 15;
/// 单个音至少这么长才拿来估段电平(50 fps ⇒ 10 帧 = 0.20 s)。短音被辅音/起音主导。
const PHRASE_LEVEL_MIN_NOTE: i64 = 10;

/// 参与匹配的最短音(帧)。短于它的音测不出稳定电平。
const LEVEL_MATCH_MIN_FRAMES: i64 = 8;
/// 参照窗:前后各看这么多个唱音。
const LEVEL_MATCH_NEIGHBOURS: usize = 16;
/// 至少要这么多个**没被救**的邻居才动手;取不到就一个字不动。
const LEVEL_MATCH_MIN_REF: usize = 4;

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
    apply_dead_only_windows_alts(
        base, sample_rate, total_frames, jobs, &[], &[], match_levels, donor_render,
    )
}

/// ⭐⭐ S162 —— [`apply_dead_only_windows`],外加**逐组的落点候选**:两个都渲,
/// 按**实测**挑 `|rel|` 更小的那个(`rel` = 该组 donor 片段的电平 − 邻近**窗外** base 的电平)。
///
/// ## ⛔ 它治的是什么
/// 用户 2026-08-26:「akiko 的 ぴゃ 之前是达到过的」——**查实了,是退化**:
/// `land_scan` 实测 **落点关(S157c 之前)⇒ 落 78(好)**,**落点=3(今天)⇒ 落 77(坏,rel −12.6)**。
/// ⛔ 翻回旧默认会毁掉 S157c 的实质作用(薄区死音 akiko 30%→12% / yachiyo 34%→3%);
///    换全局破法也判负(241 个音:thin 赢 63 / shallow 赢 29,塌掉 6→9)。⇒ **只能自适应。**
///
/// ## ⛔ 为什么判据是 `|rel|` 而不是「更响」
/// 两头都要挡:**akiko 的 ぴゃ 塌了(−12.6)**、**yuyuko 的 ぴゃ 炸了(+9.0)**。
/// ⛔ 而扫描表管不了:`low_ratio` 在 ぴゃ 上**单调递减** = 与实测完全不相关
/// (它测的是 **400 ms 稳态「あ」**,看不见 1440 ms 的 /pʲa/)。
/// ⭐ 判据的出处:`donor_pre` 的 rel 与**成品** rel 的 Pearson **+0.48…+0.81**。
///
/// ## 代价
/// 候选的 shift 大多**已经在渲**:实测 akiko 多 **3 个浅 shift**,
/// **yuyuko / 东雪莲 / yachiyo 多 0 个**。
///
/// `alts` 为空 ⇒ **逐位回到 [`apply_dead_only_windows`]**。
#[allow(clippy::too_many_arguments)]
pub fn apply_dead_only_windows_alts(
    base: &mut [f32],
    sample_rate: u32,
    total_frames: i64,
    jobs: &[DeadJob],
    alts: &[Option<i64>],
    // ⭐ S163 —— 打分改成**逐音**(见 [`CandScore`])⇒ 需要音符表。
    //    空切片 ⇒ 没有候选可选,逐位回到今天(cover 车道恒如此)。
    notes: &[NoteSpan],
    match_levels: bool,
    donor_render: impl FnMut(i64, &[(i64, i64)]) -> crate::Result<Vec<f32>>,
) -> crate::Result<()> {
    apply_dead_only_windows_with(
        base,
        sample_rate,
        total_frames,
        jobs,
        alts,
        notes,
        match_levels,
        join_rests_enabled(),
        // S159zm —— env 只在这一个入口读一次(见 [`seam_align_ms`])。
        (seam_align_ms() * f64::from(sample_rate) / 1000.0).round() as usize,
        // ⛔ S163 —— 同一条规矩:谐波否决的门限由**参数**给,判据才关得掉
        //    (关掉它正是那条阴性对照:没有它,「更响但没在唱目标音高」的候选会赢)。
        landing_harm_eps(),
        landing_repair_ms(),
        comb_floor_db(),
        landing_width_eps(),
        landing_width_floor(),
        handover_deficit_db(),
        tied_xfade_ms(),
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
    // ⭐ S162 —— 逐组的落点候选(与 `jobs` 平行;`None` = 没有候选)。
    //    空切片 ⇒ **逐位回到今天**。见 [`apply_dead_only_windows_alts`]。
    alts: &[Option<i64>],
    // ⭐ S163 —— 音符表(输出时间轴)。逐音打分要它;没有候选时一行都不碰它。
    notes: &[NoteSpan],
    match_levels: bool,
    join_enabled: bool,
    // S159zm —— 拼接前的对齐半径(**样本**)。⛔ 与 `join_enabled` 同一条理由:
    // `_with` 变体的旋钮一律是**参数**,判据才关得掉(S151 笔1)。
    align: usize,
    // ⭐ S163 —— 谐波否决的门限(dB);`0.0` = 关。见 [`landing_harm_eps`]。
    harm_eps: f32,
    // ⭐ S163 —— 修补遍的门限(ms);`0.0` = 关。见 [`landing_repair_ms`]。
    repair_ms: f32,
    // ⭐ S163 —— 梳深地板(dB);`0.0` = 关。见 [`comb_floor_db`]。
    comb_floor: f32,
    // ⭐ S163 —— 交接点体检的门限(dB);`0.0` = 关。见 [`handover_deficit_db`]。
    // S163 - peak-width veto multiplier; <= 1.0 disables. See `landing_width_eps`.
    width_eps: f32,
    // S163 - repair trigger: peak width above this (% of f0) forces a +/-1 repair pass.
    width_floor: f32,
    handover_db: f32,
    // ⭐ S163 —— 长音延续处的交叉淡化(ms);`0.0` = 关。见 [`tied_xfade_ms`]。
    tied_xf_ms: f64,
    mut donor_render: impl FnMut(i64, &[(i64, i64)]) -> crate::Result<Vec<f32>>,
) -> crate::Result<()> {
    if jobs.is_empty() || base.is_empty() || total_frames <= 0 {
        return Ok(());
    }
    let spf = base.len() as f64 / total_frames as f64;
    let xf = (sample_rate as usize / 100).max(2); // 10 ms
    let base_rms = if match_levels { active_rms(base, sample_rate) } else { None };
    // ⭐ S162 —— 候选的 shift 一起进来。`alts` 空 ⇒ 与今天逐位相同。
    let mut shifts: Vec<i64> = jobs.iter().map(|j| j.shift).collect();
    for (i, a) in alts.iter().enumerate() {
        if let (Some(s), true) = (*a, i < jobs.len()) {
            shifts.push(s);
        }
    }
    shifts.sort_unstable();
    shifts.dedup();
    // ⭐⭐ S163 —— 打分的两根轴,都在**同一份 donor 缓冲**上算(零渲染噪声)。
    //
    // ⛔ 参照换掉了:S162 用的是「窗外 base 的 ±2 s **平均**」,而在密集救援的乐句里
    //    窗外主要是**休止**(数字静音)⇒ 参照读到 −100 dBFS ⇒ `|电平 − 参照|` 退化成
    //    「谁更轻谁赢」,方向正好反了(实测 60 个窗里 6 个)。
    //    现在与 [`match_rescued_note_levels`] **共用同一份口径**:
    //    邻近 ±16 个**没被救的唱音**的电平中位;取不到 4 个 ⇒ 这个音**弃权**。
    // ⛔⛔ S163 —— **不只是「有候选才量」**：修补遍的触发条件本身就是「量了才知道」，
    //    而用户点名的 yuyuko 4:49 那一组 **一个候选都没有**（窄预算那份计划给出的落点与出厂完全相同）。
    //    第一版写成 `alts 非空` ⇒ 那一组结构上永远不会被量 ⇒ 修补遍对它是死代码。
    let scoring =
        (alts.iter().any(Option::is_some)
            || repair_ms > 0.0
            || comb_floor > 0.0
            // S163 -- usag alone must also enter scoring, or both fill points are skipped.
            || landing_usag_eps() > 0.0
            // S165 -- same for the h2 axis: without this the two fill points never run.
            || landing_h2_eps() > 0.0
            // S165 -- and for the mismatch axis, same reason.
            || landing_mismatch_eps() > 0.0)
            && !notes.is_empty();
    let (note_lv, note_cov) = if scoring {
        note_levels_and_coverage(base, spf, jobs, notes)
    } else {
        (Vec::new(), Vec::new())
    };
    let note_ref = if scoring { clean_neighbour_refs(&note_lv, &note_cov) } else { Vec::new() };
    let harm_eps = if scoring { harm_eps } else { 0.0 };
    let repair_ms = if scoring { repair_ms } else { 0.0 };
    let comb_floor = if scoring { comb_floor } else { 0.0 };
    // S163 usag sort eps (factory 0 = off = bit-identical).
    let usag_eps = if scoring { landing_usag_eps() } else { 0.0 };
    // S163 -- opponent-axis cap for the usag sort key.
    let usag_dim_cap = landing_usag_dim_cap();
    // S165 second-harmonic sort eps (factory 0 = off = bit-identical).
    let h2_eps = if scoring { landing_h2_eps() } else { 0.0 };
    // ⭐⭐⭐⭐ S165 —— **失配**轴的 eps 与它的参照曲线。
    //    ⛔ 参照必须在**拼接之前**从 `base` 建(那时它还是「没被我们碰过」的),
    //    而且只用**没落在任何救援组里**的音 —— 靶子是「这个模型自己没被救援时的样子」。
    let mism_eps = if scoring { landing_mismatch_eps() } else { 0.0 };
    let mism_ref: Vec<(f32, f32)> = if mism_eps > 0.0 {
        mismatch_reference(base, sample_rate, spf, jobs, notes)
    } else {
        Vec::new()
    };
    // 每个窗**正身**里的音 = 被这个窗盖住 ≥80% 的唱音(它的好坏由这个窗的落点决定)。
    let job_notes: Vec<Vec<usize>> = if scoring {
        jobs.iter()
            .map(|j| {
                (0..notes.len())
                    .filter(|&i| {
                        let nd = &notes[i];
                        nd.sung
                            && nd.frames >= LEVEL_MATCH_MIN_FRAMES
                            && ((nd.start + nd.frames).min(j.end) - nd.start.max(j.start)).max(0)
                                * 5
                                >= nd.frames * 4
                    })
                    .collect()
            })
            .collect()
    } else {
        Vec::new()
    };
    // S152 —— 每一遍 donor 在**它自己的窗 ± 余量**上的样本,留到全部渲完之后再拼。
    // ⛔ 为什么要留:窗边该放在休止的哪一点,只有**同时看得见两侧那两条 donor** 才决定得了
    // (见 `join_rests`)。留的是片段不是整条:全曲窗覆盖约 56 %,实测这首歌约 30 MB。
    let mut kept: Vec<(i64, usize, Vec<f32>)> = Vec::new();
    // ⭐ S162/S163 —— (组下标, shift, lo, 片段, 逐音打分)。`alts` 空时每组只有一项 ⇒ 逐位同今天。
    let mut cand: Vec<(usize, i64, usize, Vec<f32>, CandScore)> = Vec::new();
    for s in shifts {
        // 这一遍**自己**的窗。⛔ 同一个 filter 下面 :896 还要用一次(音频域拼接),两处必须
        // 是同一个谓词 —— 那正是 S147 hotfix 的形状:闭包那一侧漏了它,拼接这一侧没漏,
        // 于是「渲多了但拼对了」= 功能正确、收益减半。
        // ⭐ S162 —— 这一遍要渲的窗 = 「shift 等于 s 的组」∪「候选等于 s 的组」。
        let mine = |ji: usize, j: &DeadJob| -> bool {
            j.shift == s || alts.get(ji).copied().flatten() == Some(s)
        };
        let own: Vec<(i64, i64)> = jobs
            .iter()
            .enumerate()
            .filter(|(ji, j)| mine(*ji, j))
            .map(|(_, j)| (j.start, j.end))
            .collect();
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
        for (ji, j) in jobs.iter().enumerate().filter(|(ji, j)| mine(*ji, j)) {
            if let Some((lo, hi)) = donor_read_span(j, spf, n, MERGE_BRIDGE_FRAMES) {
                // ⭐⭐ S163 —— **逐音**打分(两根轴,全部在这一份 donor 缓冲上算)。
                let mut sc = CandScore::default();
                if scoring {
                    for &ni in &job_notes[ji] {
                        let nd = &notes[ni];
                        let na = ((nd.start as f64) * spf).round().max(0.0) as usize;
                        let nb = (((nd.start + nd.frames) as f64) * spf).round().max(0.0) as usize;
                        let (a, b) = (na.max(lo).min(hi), nb.min(hi));
                        // ⛔ 读到的必须覆盖这个音的 ≥80%,否则量的是半个音(窗边那一截)。
                        if b <= a + 1024 || (b - a) * 5 < (nb.saturating_sub(na)) * 4 {
                            continue;
                        }
                        let seg = &donor[a..b];
                        let e: f64 = seg.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>()
                            / (b - a) as f64;
                        let l = (10.0 * (e + 1e-20).log10()) as f32;
                        if note_ref[ni].is_finite() {
                            sc.rel.push((ni, (l - note_ref[ni]).abs()));
                        }
                        if nd.hz > 0.0 {
                            if let Some(h) = utai_dsp::harmonicity::harmonic_energy_fraction_db(
                                seg,
                                sample_rate,
                                nd.hz,
                            ) {
                                sc.harm.push((ni, h));
                            }
                        }
                        // ⭐ S163 —— 音内「唱没了」的毫秒数(不需要参照)。
                        // ⭐⭐⭐ S163 —— 谐波谱峰的**宽度**（每一根谐波自己糊不糊）。
                        // ⛔ 这一处曾经漏掉（只加在了修补遍后的重评估那处）⇒ `CandScore::width`
                        //    在**主路径**上恒为空，于是 `width_eps` 全曲只触发 1 次（那 1 次正是
                        //    修补遍后的）、`width_floor` 结构上触发不了。实测 P2 臂：`peak-width veto` **0 次**。
                        if nd.hz > 0.0 {
                            if let Some(w) =
                                utai_dsp::harmonicity::harmonic_peak_width_pct(seg, sample_rate, nd.hz)
                            {
                                sc.width.push((ni, w));
                            }
                        }
                        // ⭐⭐⭐⭐ S163 —— **上方谐波的音内塌陷**,第 1 处(主评估)。
                        // ⛔ 两处填充点,漏一处 = 这根轴静默失效(`width` 栽过)。
                        if nd.hz > 0.0 {
                            if let Some(u) = utai_dsp::harmonicity::upper_harmonic_sag_db(
                                seg,
                                sample_rate,
                                nd.hz,
                            ) {
                                sc.usag.push((ni, u));
                            }
                            if let Some(l) = utai_dsp::harmonicity::upper_harmonic_level_db(
                                seg,
                                sample_rate,
                                nd.hz,
                            ) {
                                sc.uplev.push((ni, l));
                            }
                            // ⭐⭐⭐ S165 —— **第二谐波电平**。⛔ 与 `width` 同一个坑:
                            //    **两处填充点(主评估 + 修补遍后重评估),漏一处这根轴就静默失效。**
                            if let Some(h) = utai_dsp::harmonicity::second_harmonic_level_db(
                                seg,
                                sample_rate,
                                nd.hz,
                            ) {
                                sc.h2.push((ni, h));
                            }
                            // ⭐⭐⭐⭐ S165 —— **失配度**,第 1 处(主评估)。⛔ 同样是两处填充点。
                            if !mism_ref.is_empty() {
                                if let Some(m) = note_mismatch(seg, sample_rate, nd.hz, &mism_ref) {
                                    sc.mism.push((ni, m));
                                }
                            }
                            // ⭐⭐⭐⭐ S165 —— **音高误差**,第 1 处。⛔ 三处填充点了,漏一处这道闸就静默失效。
                            if let Some(pe) = utai_dsp::harmonicity::pitch_error_cents(
                                seg,
                                sample_rate,
                                nd.hz,
                            ) {
                                sc.pitch.push((ni, pe));
                            }
                        }
                        let g = silence_run_ms(seg, sample_rate);
                        if g > 0.0 {
                            sc.gone.push((ni, g));
                        }
                        // ⭐⭐⭐⭐ S165 —— **音内电平跌幅**,第 1 处(主评估)。
                        // ⛔ 两处填充点,漏一处这根轴静默失效(`width` 正是这样栽的)。
                        if let Some(d) = note_dip_db(seg, sample_rate) {
                            sc.dip.push((ni, d));
                        }
                        // ⭐ S163 —— 谐波**梳深**(谐波之间起没起雾)。
                        if nd.hz > 0.0 {
                            if let Some(c) =
                                utai_dsp::harmonicity::comb_depth_db(seg, sample_rate, nd.hz)
                            {
                                sc.comb.push((ni, c));
                            }
                        }
                    }
                }
                cand.push((ji, s, lo, donor[lo..hi].to_vec(), sc));
            }
        }
    }
    // ⭐ S162/S163 —— 逐组定案(三层,见 [`decide_group`]);没有候选的组原样。
    let mut chosen: Vec<Option<usize>> = vec![None; jobs.len()];
    let log_pick = |cand: &[(usize, i64, usize, Vec<f32>, CandScore)],
                    mine: &[usize],
                    ji: usize,
                    jobs: &[DeadJob]| {
        if mine.len() < 2 {
            return;
        }
        // ⛔ S129 铁律:「臂开着」与「臂做了事」必须**分开**可查 —— 每一组都打候选表。
        tracing::info!(
            "range: group[{}..{}] landing candidates {:?} — kept {:+}",
            jobs[ji].start,
            jobs[ji].end,
            mine.iter()
                .map(|&c| (cand[c].1, cand[c].4.worst_rel(), cand[c].4.worst_gone()))
                .collect::<Vec<_>>(),
            cand[mine[0]].1
        );
        if cand[mine[0]].1 != jobs[ji].shift {
            tracing::info!(
                "range: group[{}..{}] landing re-picked {:+} → {:+} by measurement \
                 (worst-note |rel| {:.1} dB, silence {:.0} ms)",
                jobs[ji].start,
                jobs[ji].end,
                jobs[ji].shift,
                cand[mine[0]].1,
                cand[mine[0]].4.worst_rel().unwrap_or(f32::NAN),
                cand[mine[0]].4.worst_gone()
            );
        }
    };
    for ji in 0..jobs.len() {
        let mine = decide_group(
            &cand,
            ji,
            jobs[ji].shift,
            harm_eps,
            repair_ms,
            comb_floor,
            width_eps,
            usag_eps,
            usag_dim_cap,
            h2_eps,
            mism_eps,
        );
        log_pick(&cand, &mine, ji, jobs);
        chosen[ji] = mine.first().copied();
    }

    // ── ⭐⭐⭐ S163 —— **修补遍**:被选中的那一遍里若有唱音【自己塌成一段静音】,
    //    就再渲 `shift ± 1` 的**那一组窗**,按同一套打分重选。
    //    ⛔ 代价跟着「真的坏了」走:实测四条臂 525 个被救音里**只有 1 个**命中
    //    (yuyuko 4:49 的 `[801]あ`,300 ms);而「一开始就给每组多渲 ±1」要付 2-3 倍。
    if repair_ms > 0.0 && scoring {
        // ⭐ S163 —— 三个触发,全部**零混杂零噪声**(同一份 donor 缓冲上算,不需要跨音高参照):
        //   ⑴ 唱成了一段静音(≥ `repair_ms`)—— 选择性实测 **1/525**;
        //   ⑵ 谐波之间糊成一片(梳深 < `comb_floor`)—— 选择性 **1/525**;
        //   ⑶ ⭐ **候选之间分差 > `REPAIR_SPREAD_DB`** ⇒ 落点在这里对结果极其敏感,
        //      而**我们手上这两个候选未必是好的那一格**。
        //      实测 akiko 40 个带候选的窗:分差中位 **0.90** dB、p90 1.96,
        //      **只有 1 个 > 6 —— 正是「ぴゃ」(10.5)**,而它真正的好落点是 −14,
        //      从来没进过候选;`shift ± 1` 从 −13 出发正好够得着。
        let spread = |ji: usize| -> f32 {
            let v: Vec<f32> = (0..cand.len())
                .filter(|&c| cand[c].0 == ji)
                .filter_map(|c| cand[c].4.worst_rel())
                .collect();
            if v.len() < 2 {
                return 0.0;
            }
            v.iter().copied().fold(f32::MIN, f32::max) - v.iter().copied().fold(f32::MAX, f32::min)
        };
        let need: Vec<usize> = (0..jobs.len())
            .filter(|&ji| {
                chosen[ji].is_some_and(|c| {
                    cand[c].4.worst_gone() >= repair_ms
                        || (comb_floor > 0.0 && cand[c].4.worst_comb() < comb_floor)
                        // ⭐⭐⭐ S163 —— ④ **谱峰糊到超过 `width_floor`**。
                        //   ⛔ 前三个触发都看不见它：实测 yuyuko 4:36 接缝两侧峰宽
                        //   **12.33 vs 0.99（差 12 倍）**，而填充度/占比/梳深都没反应。
                        //   ⭐ 为什么它必须走**修补遍**而不是“候选之间选”：
                        //   实测（N2 诊断臂）峰宽否决在**全曲只触发 1 次** ——
                        //   大部分组只有 **1 个**候选，“在候选之间选”结构上无处可选。
                        //   ⭐ 而好格确实存在且就在隔壁：yuyuko 短音 `donor_pre` 峰宽
                        //   **77 → 3.50 而 78 → 11.88**（只差 1 个半音），而每个模型的好格
                        //   **位置不同**（akiko 最好 74 / 最差 76；dxl 最好 75）⇒ 不能写死，
                        //   只能**渲了再选** —— 而那正是修补遍在做的事。
                        || (width_floor > 0.0 && cand[c].4.worst_width() > width_floor)
                        // ⭐⭐⭐⭐ S163 §27 —— ⑥ **上方谐波在音内被抽干**。
                        //   ⛔ 这一条是让 §25 那根轴在【没有候选】的模型上也能生效的**唯一**通路:
                        //   实测 akiko 有 58 组 landing candidates,而 **yuyuko 一组都没有**
                        //   (两套计划给出完全相同的落点)⇒ `decide_group` 开头就 return。
                        //   ⇒ 只有走修补遍(先渲、坏了再补 ±2)才轮得到 `usag` 说话。
                        //   ⭐ 门限用 `usag_eps` 的 2 倍:eps 是「算有差别」,这里要的是「确实坏了」。
                        // ⭐ 门限 = **1.0 × eps**(不是 2.0)。实测四模型:
                        //    2.0×eps(−6 dB)时 **yuyuko 与东雪莲一组都不触发**,
                        //    而用户点名的坐标 usag 是 ぴゃ −5.97 / 4:07 −3.31 / 4:10 −10.40 /
                        //    yuyuko 4:36.151 −4.09 ⇒ **−3 全部覆盖,−6 只盖住一个**。
                        //    代价:触发率 akiko 11% / yuyuko 10% / 东雪莲 14%(每组多渲 ±2,
                        //    而 `REPAIR_RADIUS = 2` 实测 donor 遍数不涨)。
                        || (usag_eps > 0.0 && cand[c].4.worst_usag() < -usag_eps)
                        // ⭐⭐⭐ S165 —— ⑦ **`2·f0` 被掏空**(用户 2026-08-28 点名的那一族)。
                        //   ⛔ 这一条和 ⑥ 一样,是让新轴在【没有候选】的模型上生效的**唯一**通路:
                        //   yachiyo 的落点候选产出率是 **0%** ⇒ 不先渲出来,排序永远没有输入。
                        //   ⚙ 门限 −12 dB 由实测触发率定(yachiyo 24% / yuyuko 17%)。
                        || (h2_eps > 0.0 && cand[c].4.worst_h2() < H2_REPAIR_FLOOR)
                        // ⭐⭐⭐⭐ S165 —— ⑧ **失配**(响度 ↔ 抖动)超标。
                        //   ⛔ 与 ⑥⑦ 同理:这是让失配轴**有候选可选**的唯一通路 ——
                        //   实测不加它时,失配轴每组只被问 2-6 次,落点 0 变化。
                        || (mism_eps > 0.0 && cand[c].4.worst_mism() > MISM_REPAIR_FLOOR)
                }) || spread(ji) > REPAIR_SPREAD_DB
            })
            .collect();
        if !need.is_empty() {
            for &ji in &need {
                tracing::warn!(
                    "range: group[{}..{}] at {:+} needs repair — {:.0} ms of SILENCE inside a note, \
                     worst comb depth {:.1} dB (floor {comb_floor}) — rendering ±1 repair passes",
                    jobs[ji].start,
                    jobs[ji].end,
                    cand[chosen[ji].unwrap()].1,
                    cand[chosen[ji].unwrap()].4.worst_gone(),
                    cand[chosen[ji].unwrap()].4.worst_comb()
                );
            }
            let mut by_shift: std::collections::BTreeMap<i64, Vec<usize>> = Default::default();
            for &ji in &need {
                // ⭐ S163 —— ±1 必须绕【所有现有候选】转，不是只绕选中的那一个。
                //    ⛔ 实测：「ぴゃ」选中的是 −12，只绕它转 ⇒ 渲 −11/−13，而真正的好落点
                //    **−14** 是计划那个 −13 的 −1，永远够不着（日志：`[(-12,60),(-13,60),(-11,60)]`）。
                let curs: Vec<i64> =
                    (0..cand.len()).filter(|&c| cand[c].0 == ji).map(|c| cand[c].1).collect();
                // ⭐⭐⭐ S165 —— **`2·f0` 触发的组用宽半径**,其余组一个字节不动。
                //   ⛔ 用户点名的那个音今天在 **−8**(`2·f0` −17.0 dB),好格是 **−13**(−5.2,
                //   与用户说「好」的 yuyuko **完全一致**)⇒ `REPAIR_RADIUS = 2` 从 −8 只够到 −10,
                //   **结构上够不着**。⚠ 代价只有触发的那 17-24% 的组付。
                //   ⭐ S165 —— `h2` 与**失配**各自的宽半径;两者都触发时取宽的那个。
                let h2_wide = h2_eps > 0.0
                    && chosen[ji].is_some_and(|c| cand[c].4.worst_h2() < H2_REPAIR_FLOOR);
                let mism_wide = mism_eps > 0.0
                    && chosen[ji].is_some_and(|c| cand[c].4.worst_mism() > MISM_REPAIR_FLOOR);
                let radius = match (h2_wide, mism_wide) {
                    (true, true) => H2_REPAIR_RADIUS.max(MISM_REPAIR_RADIUS),
                    (true, false) => H2_REPAIR_RADIUS,
                    (false, true) => MISM_REPAIR_RADIUS,
                    (false, false) => REPAIR_RADIUS,
                };
                // ⭐⭐⭐ S163 —— **搜索半径从 ±1 扩到 ±[`REPAIR_RADIUS`]**。
                //    ⛔ 实测（R 臂，两个旋钮成对开）：修补遍触发 32/48 次、峰宽否决
                //    真的在选（31/106 次），而人群面**几乎不变**（次怪段 p75 11.89 → 11.89）。
                //    机理：yuyuko 的好格是落点 **77**（峰宽 3.50）而当前落点常是 **79**
                //    ⇒ ±1 只到 78/80，**够不到 77**。好格常在 ±2。
                //    ⚠ 成本：donor 遍数实测**不涨**（新 shift 大多落在已有的那几个上），
                //    只有 wall 涨（K→Q 上升 48%）。
                for (cur, d) in curs
                    .iter()
                    .flat_map(|&c| (1..=radius).flat_map(move |r| [(c, -r), (c, r)]))
                {
                    let s2 = cur + d;
                    if s2 == 0 || s2.abs() > MAX_RANGE_SHIFT {
                        continue;
                    }
                    // ⛔ 已经渲过这个 shift 的组不再重复(它本来就是候选之一)
                    if cand.iter().any(|c| c.0 == ji && c.1 == s2) {
                        continue;
                    }
                    by_shift.entry(s2).or_default().push(ji);
                }
            }
            for (s2, gs) in by_shift {
                let own: Vec<(i64, i64)> =
                    gs.iter().map(|&ji| (jobs[ji].start, jobs[ji].end)).collect();
                let donor = donor_render(s2, &own)?;
                let n = base.len().min(donor.len());
                for &ji in &gs {
                    let j = &jobs[ji];
                    if let Some((lo, hi)) = donor_read_span(j, spf, n, MERGE_BRIDGE_FRAMES) {
                        let mut sc = CandScore::default();
                        for &ni in &job_notes[ji] {
                            let nd = &notes[ni];
                            let na = ((nd.start as f64) * spf).round().max(0.0) as usize;
                            let nb =
                                (((nd.start + nd.frames) as f64) * spf).round().max(0.0) as usize;
                            let (a, b) = (na.max(lo).min(hi), nb.min(hi));
                            if b <= a + 1024 || (b - a) * 5 < (nb.saturating_sub(na)) * 4 {
                                continue;
                            }
                            let seg = &donor[a..b];
                            let e: f64 =
                                seg.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>()
                                    / (b - a) as f64;
                            let l = (10.0 * (e + 1e-20).log10()) as f32;
                            if note_ref[ni].is_finite() {
                                sc.rel.push((ni, (l - note_ref[ni]).abs()));
                            }
                            if nd.hz > 0.0 {
                                if let Some(h) =
                                    utai_dsp::harmonicity::harmonic_energy_fraction_db(
                                        seg,
                                        sample_rate,
                                        nd.hz,
                                    )
                                {
                                    sc.harm.push((ni, h));
                                }
                            }
                            let g = silence_run_ms(seg, sample_rate);
                            if g > 0.0 {
                                sc.gone.push((ni, g));
                            }
                            // ⭐⭐⭐⭐ S165 —— **音内电平跌幅**,第 2 处(修补遍后重评估)。
                            if let Some(d) = note_dip_db(seg, sample_rate) {
                                sc.dip.push((ni, d));
                            }
                            if nd.hz > 0.0 {
                                    if let Some(w) =
                                        utai_dsp::harmonicity::harmonic_peak_width_pct(
                                            seg,
                                            sample_rate,
                                            nd.hz,
                                        )
                                    {
                                        sc.width.push((ni, w));
                                    }
                                }
                                // ⭐⭐⭐⭐ S163 —— **上方谐波的音内塌陷**,第 2 处。
                                if nd.hz > 0.0 {
                                    if let Some(u) = utai_dsp::harmonicity::upper_harmonic_sag_db(
                                        seg,
                                        sample_rate,
                                        nd.hz,
                                    ) {
                                        sc.usag.push((ni, u));
                                    }
                                    if let Some(l) =
                                        utai_dsp::harmonicity::upper_harmonic_level_db(
                                            seg,
                                            sample_rate,
                                            nd.hz,
                                        )
                                    {
                                        sc.uplev.push((ni, l));
                                    }
                                    // ⭐⭐⭐ S165 —— **第二谐波电平**,第 2 处。
                                    //    ⛔ 漏这一处 = 修补遍渲出来的新候选**没有 h2 分数**
                                    //    ⇒ 排序拿不到它 ⇒ 整根轴静默失效(`width` 正是这样栽的)。
                                    if let Some(h) =
                                        utai_dsp::harmonicity::second_harmonic_level_db(
                                            seg,
                                            sample_rate,
                                            nd.hz,
                                        )
                                    {
                                        sc.h2.push((ni, h));
                                    }
                                    // ⭐⭐⭐⭐ S165 —— **失配度**,第 2 处(修补遍后重评估)。
                                    //    ⛔ 漏这一处 = 修补遍渲出来的新候选没有失配分 ⇒ 排序拿不到它。
                                    if !mism_ref.is_empty() {
                                        if let Some(m) =
                                            note_mismatch(seg, sample_rate, nd.hz, &mism_ref)
                                        {
                                            sc.mism.push((ni, m));
                                        }
                                    }
                                    // ⭐⭐⭐⭐ S165 —— **音高误差**,第 2 处。
                                    //    ⛔⛔ 这一处尤其要紧:**炸掉的候选正是修补遍(宽半径)新渲出来的**,
                                    //    漏了它,音高闸对最危险的那批候选完全失效。
                                    if let Some(pe) = utai_dsp::harmonicity::pitch_error_cents(
                                        seg,
                                        sample_rate,
                                        nd.hz,
                                    ) {
                                        sc.pitch.push((ni, pe));
                                    }
                                }
                                if nd.hz > 0.0 {
                                if let Some(c) =
                                    utai_dsp::harmonicity::comb_depth_db(seg, sample_rate, nd.hz)
                                {
                                    sc.comb.push((ni, c));
                                }
                            }
                        }
                        cand.push((ji, s2, lo, donor[lo..hi].to_vec(), sc));
                    }
                }
            }
            for &ji in &need {
                let mine = decide_group(
                    &cand,
                    ji,
                    jobs[ji].shift,
                    harm_eps,
                    repair_ms,
                    comb_floor,
                    width_eps,
                    usag_eps,
                    usag_dim_cap,
                    h2_eps,
                    mism_eps,
                );
                log_h2_stat("repair");
                log_mism_stat("repair");
                if let Some(&c) = mine.first() {
                    tracing::info!(
                        "range: group[{}..{}] repair — {:?} ⇒ kept {:+} (silence {:.0} ms)",
                        jobs[ji].start,
                        jobs[ji].end,
                        mine.iter()
                            // ⭐ S165 —— 把 `2·f0` 一并打出来。⛔ 没有它的时候
                            //    「47 次触发 / 0 条落点改变」这种结果**无法归因**
                            //    (是候选都不够好,还是这根轴根本没发言?)。
                            .map(|&x| (cand[x].1, cand[x].4.worst_gone(), cand[x].4.worst_h2()))
                            .collect::<Vec<_>>(),
                        cand[c].1,
                        cand[c].4.worst_gone()
                    );
                    chosen[ji] = Some(c);
                }
            }
        }
    }

    for ji in 0..jobs.len() {
        if let Some(c) = chosen[ji] {
            kept.push((ji as i64, cand[c].2, cand[c].3.clone()));
        }
    }
    // ⭐⭐⭐⭐ S163 —— **同一次 run 内的对照臂**(纯诊断转储,不参与任何判据)。
    //
    // ## ⛔⛔⛔ 为什么验收只能这样做
    // 整曲渲染**不可复现**(源头在解码)⇒ 每次 run 的 donor 略不同
    // ⇒ 拼接前的**对齐搜索选出不同的 `d`** ⇒ **整条窗平移几毫秒**。实测跨 run:
    // ```text
    // 对齐 |lag| ≥ 1 样本的窗:akiko 56% · yuyuko 70% · yachiyo 64% · 东雪莲 93%
    // 最大平移 419/162/405/441 样本 = 4-10 ms
    // 同一把尺子(±60 ms 取最小)的噪声底:
    //    对齐一致(|lag|<5)  p90 0.48-1.26 dB
    //    对齐不同(|lag|≥5)  p90 1.49-3.40 dB,**p99 12.8-17.6 dB**
    // ```
    // ⇒ **跨 run 比整曲臂,任何逐坐标读数都不可信。**
    //
    // 这里在同一次 run 里跑两遍 `splice_kept`,共享**同一份 `kept`**
    // (= 同一份 donor、同一个 donor 选择、同一次解码)⇒ 零解码噪声、零对齐抖动,
    // 两条只差 `dipfill` 一个变量。
    //
    // ⚠ 没设 env 时 `var` 立刻返回,不多跑一遍 ⇒ 不是生产路径上的开销。
    // ⛔ 判据不许读这个 env —— 它只写文件,不改本次渲染的音频(与 `dump_donor_buffer` 同性质)。
    let nodip = std::env::var("UTAI_RANGE_DUMP_NODIP")
        .ok()
        .map(|v| v.trim().to_owned())
        .filter(|v| !v.is_empty());
    let control = nodip.as_ref().map(|_| base.to_vec());
    let tied_xf_samples = (tied_xf_ms * f64::from(sample_rate) / 1000.0) as usize;

    let out = splice_kept(
        base,
        sample_rate,
        spf,
        jobs,
        &kept,
        xf,
        join_enabled,
        align,
        handover_db,
        notes,
        tied_xf_samples,
        // ⭐⭐⭐⭐ S163 §34 —— donor 静音坑回填（出厂 0 = 关 = 逐位不变）。
        dipfill_depth_db(),
        rest_base_enabled(),
        if seam_ramp_enabled() {
            (SEAM_RAMP_MS * f64::from(sample_rate) / 1000.0) as usize
        } else {
            0
        },
        onset_fit_enabled(),
        seam_align_wide(),
    );

    if out.is_ok() {
        if let (Some(dir), Some(mut buf)) = (nodip, control) {
            // ⛔⛔ **两条必须同层** —— 都在 `splice_kept` 刚结束时落盘。
            //    第一版只落了对照臂,拿它去比**成品 wav**(还经过 splice 之后的整条处理链)
            //    ⇒ 差异里混进了那条链:akiko 报「动过 75.30 s、窗外动过 304 万样本」,
            //    而日志报的 dipfill 生效量只有几百毫秒、窗外更应该是 0。
            let dump = |name: &str, b: &[f32]| {
                let dir = std::path::Path::new(&dir);
                if let Err(e) = std::fs::create_dir_all(dir) {
                    tracing::warn!("range: same-run control arm cannot mkdir {}: {e}", dir.display());
                    return;
                }
                let mut bytes = Vec::with_capacity(b.len() * 4);
                for v in b {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                let p = dir.join(name);
                match std::fs::write(&p, &bytes) {
                    Ok(()) => {
                        tracing::info!("range: same-run control arm {} samples -> {}", b.len(), p.display())
                    }
                    Err(e) => tracing::warn!("range: same-run control arm write failed {}: {e}", p.display()),
                }
            };
            // ⑴ dipfill **开**(= 本次渲染真正用的那一条),splice 之后立即
            dump("dip.f32", base);
            // ⑵ dipfill **关**,同一份 `kept`,其余一个字节不差
            match splice_kept(
                &mut buf,
                sample_rate,
                spf,
                jobs,
                &kept,
                xf,
                join_enabled,
                align,
                handover_db,
                notes,
                tied_xf_samples,
                0.0,
                rest_base_enabled(),
                if seam_ramp_enabled() {
                    (SEAM_RAMP_MS * f64::from(sample_rate) / 1000.0) as usize
                } else {
                    0
                },
                // control arm: onset shape fit OFF, everything else identical
                false,
                seam_align_wide(),
            ) {
                Ok(()) => dump("nodip.f32", &buf),
                Err(e) => tracing::warn!("range: same-run control arm failed: {e}"),
            }
        }
    }
    out
}

/// S152 —— 拼接层,从 `apply_dead_only_windows` 拆出来:它现在拿到的是**全部**位移的 donor
/// 片段,所以窗边可以由音频决定(`join_rests`),而不是只能由帧数规则决定。
/// ⛔ Changing this changes the audio ⇒ pair-bump `RANGE_ALGO_VERSION` and `audition_cache_tag`。
/// 机理、实测与为什么它不是 S151d 那条判负,全在 [`seam_align_ms`] 的 doc。
/// ⚙ 出厂默认 = **false(关)** —— `UTAI_RANGE_SEAM_ALIGN_WIDE=1` 打开。
///
/// ## 它改的是什么
/// 拼接前的对齐(见 [`seam_align_ms`])原本用**基础淡化宽度 `xf`(10 ms)**当搜索窗,
/// 却把整个片段按搜出来的 lag **整体平移** —— 而 `tied_xfade` 那一族的淡入是 **120 ms**。
/// 10 ms @ 40 kHz 在 f0≈500 Hz 上只有 5 个周期 ⇒ 互相关**有周期歧义**,会锁到错的周期。
///
/// ## ⛔ 为什么出厂**关**:方向对,但效应没验出来
/// 用户 2026-08-28 点名 yuyuko × 炉心 4:29.265 / 4:36.319「接缝不干净 / f0 附近多出面状」。
/// 离线量到 120 ms 窗上的真 lag 是 **+21 / +154** 样本,而生产挪的是 **−48 / +73**,
/// 挪完相干度**变成负的**(−0.209 / −0.065,比不挪的 0.322 / −0.177 还差)。
/// 打开之后生产挪的量确实靠近了真值(#73 **+137** vs 真值 +154),但**成品几乎不动**:
///
/// | | R0 关对齐 | R1 开(4 ms) | R2 开(2 ms) |
/// |---|---|---|---|
/// | 4:29 落差 | 3.92 | 3.91 | 4.03 |
/// | 4:36 落差 | 6.92 | **5.75** | 5.79 |
/// | 全曲 `interh` 超噪声底(好/坏) | — | 39/47 | 30/22 |
///
/// ⇒ 只有 4:36 改善 1.17 dB,4:29 纹丝不动,全曲还略净差 ⇒ **不足以翻默认**。
///
/// ## ⭐ 因为凹陷的主体根本不是相消
/// `donor−11` −8.56 / `donor−4` −21.11(差 **12.55 dB**),等增益淡化在 w=0.5 的功率
/// ≈ `0.25·Pa` = **−14.58 dB**,而成品实测 **−14.75** ⇒ **几乎完全吻合**。
/// 那个凹陷是 **120 ms 等增益淡化的固有结果**,与相位对不对齐关系不大。
/// ⇒ 真正该动的是 [`handover_deficit_db`] 的**评估窗位置**:它看交接点**往后 40 ms**
///   (那里落差只有 5.27 dB,门限 15 够不着),而 `tied_xfade` 的淡入区在交接点
///   **往前 120 ms**(那里落差 **12.55 dB**)⇒ **它结构上看不见自己该拦的那一段**。
pub fn seam_align_wide() -> bool {
    matches!(
        std::env::var("UTAI_RANGE_SEAM_ALIGN_WIDE").ok().as_deref().map(str::trim),
        Some("1") | Some("on") | Some("true") | Some("yes")
    )
}

/// ⛔ S165 —— **试过 4.0,实测与 2.0 没有区别,收回**(靶子 4:36 5.75 vs 5.79、4:29 3.91 vs 4.03)。
/// 单独加大半径更是无效(§54.5:align=8 在别处 `interh` 净变坏 坏56/好28)。
const SEAM_ALIGN_MS_DEFAULT: f64 = 2.0;

/// ⚙ 出厂默认 = 2.0 —— `UTAI_RANGE_SEAM_ALIGN=<ms>` 在**拼接之前先把两条臂对齐**。
///
/// ## ⭐⭐⭐ 缺陷:我们从来没对齐过,而两条臂其实是同一条波形错开了 0.2 ms
///
/// 用户 2026-08-22 给的那条判据(它比任何指标都硬):
/// 「**没有缝这件事做不到,所以缝不是问题;问题是【我们的缝会响】。**UTAU 那种拼接引擎
/// 每个音两条缝、动不动用声码器移二三十个半音,却既没有咔哒也没有竖条纹、共振峰还保得很好
/// ⇒ 这件事存在解法,不是物理必然。」
///
/// 实测(鹅妈妈 +7 × 东雪莲,S159zk 之后 30 条 **donor↔donor** 的缝;交叉淡化那 ±10 ms 上
/// 两条臂的**归一互相关**):
///
/// | | 值 |
/// |---|---|
/// | **零滞后 ρ**(= 今天硬淡用的那个)| **−0.139** —— 基本无关 |
/// | **最佳滞后处 \|ρ\|** | **0.928**(p90 0.975)|
/// | 最佳滞后 | 中位 **8 样本 = 0.18 ms** |
///
/// ⇒ **两条臂几乎是同一条波形,只是错开了 0.18 ms。**不对齐就硬淡 = 把信号与它自己
/// 延迟 0.18 ms 的副本相加 ⇒ **梳状陷波第一个零点落在 ~2.8 kHz**,而且它在 10 ms 里
/// 出现又消失 ⇒ **宽带瞬态 = 谱图上那条竖线**。
/// ⭐ 这正是 UTAU 那件事的结构差:它两侧来自**同一份录音**、交叠放在音源作者挑的稳定段
/// ⇒ **天然对齐**;我们是两次**独立的 SVC 渲染**,而且**一次都没对过**。
///
/// ## 做什么
///
/// 淡入区上按互相关搜一个整数样本偏移 `d`(|d| ≤ 这个旋钮给的毫秒数),让进来的那条臂
/// 与 `base` 里**已经写好的内容**对齐,然后整窗按 `d` 读。
/// ⚠ 整窗一起挪 ⇒ 时序漂移 ≤ 这个旋钮的毫秒数(出厂候选 2 ms);窗的**另一头**有它自己
/// 那条缝,会在轮到它时相对**挪过之后的** base 再对一次 ⇒ 自洽。
///
/// ⛔ 与 S151d 那条判负**不是一回事**:那次换的是淡化的**形状**(等增益 → √功率),
/// 整曲读数一个字没动。这一条动的是**对齐**,而上面那张表说的正是「形状再怎么换,
/// ρ = −0.139 的两条臂淡出来都是梳状」。
///
/// ## ✅ 实测(S159zm,`splice_probe`:**同一份 base + donor 缓冲**,只翻这个旋钮)
///
/// ⛔ 为什么必须是那个台子:「渲两遍整曲再比」被 donor 路径的跨进程不可复现性淹没
/// (实测两遍差 **85 %** 样本)。⇒ 把 base 与每一遍 donor 落盘一次,所有臂喂同一份输入。
///
/// ⚠ **验收轴换过一次,而且第一把是错的**:先用「缝前 20 ms vs 后 20 ms 的谱形跳变」量,
/// 读到 **−0.04 dB** = 没效果。**那把尺子对相位不敏感** —— 两条臂的**频谱**本来就相似,
/// 对齐改的是**波形连续性**。(而且我第一版的「对照」写错了,读数与被测臂逐字相同。)
/// ⇒ 换成时域轴:淡化那 10 ms 的 **一阶差 p99.5 / 局部 RMS**,地板 = **窗心**(那里没有淡化)。
///
/// | 半径 | 一阶差 | 超出地板 | 淡化区电平 vs 两侧 |
/// |---|---|---|---|
/// | 0(今天之前)| 0.3731 | **0.0185** | **−0.63 dB** |
/// | 0.5 ms | 0.3680 | 0.0134 | +0.22 |
/// | 1 ms | 0.3680 | 0.0134 | +0.40 |
/// | **2 ms(出厂)** | 0.3622 | **0.0076** | **+0.28** |
/// | 4 ms | 0.3515 | −0.0030 | +0.31 |
/// | 8 ms | 0.3478 | −0.0068 | +0.40 |
/// | ⛔ 地板(窗心,无淡化)| 0.3546 | 0 | −0.04 |
///
/// ⇒ **超出地板那部分少了 59 %**,而 **−0.63 dB 的凹陷消失**(逐窗 78/112 改善,p90 +1.52 dB)。
/// 实机 131 个窗里 **94 个**拿到非零偏移,ρ 抬得很猛(实测三例:−0.714 → **0.959** ·
/// −0.648 → **0.991** · 0.536 → **0.989**)。
///
/// ## 为什么是 2.0 而不是 4/8
///
/// 4 ms 起读数掉到**地板以下** —— 那是尺子饱和的指纹,不是更好。而观测到的滞后基本是
/// **一个基音周期的整数倍**,对齐只需要够到**半个周期**:2 ms ≥ 半周期 ⇔ f0 ≥ 250 Hz,
/// 把 donor 的整个音域(实测 370-740 Hz)都覆盖了。⇒ 再大只是给整窗多加时移,没有声学收益。
/// ⚠ 代价如实记:命中的窗整条挪 ≤2 ms。**这一条没有耳朵背书** —— 承重之前该过一次耳判。
///
/// ⛔ 判据不许读这个 env(S151 笔1)——`splice_kept` 拿的是**参数**,
/// 由 `apply_dead_only_windows` 在唯一的入口读一次。
pub fn seam_align_ms() -> f64 {
    parse_seam_align(std::env::var("UTAI_RANGE_SEAM_ALIGN").ok().as_deref())
}

fn parse_seam_align(v: Option<&str>) -> f64 {
    v.and_then(|v| v.trim().parse().ok())
        .filter(|x: &f64| x.is_finite() && *x >= 0.0 && *x <= 20.0)
        .unwrap_or(SEAM_ALIGN_MS_DEFAULT)
}

/// ⭐⭐⭐ S163 —— 交接点体检:**进来的那条 donor 静音时,把交接往后挪**。
///
/// 返回 `oi -> 新的切换样本`,与 [`join_rests`] 同一种口径(左窗伸到切换点、右窗从
/// 切换点往前一个淡化宽度开始)⇒ 两者可以合并,而**这一把优先**(它治的是「掉进静音」,
/// 那比「缝落在哪儿」严重一个量级)。
///
/// 账与出处在 [`handover_deficit_db`] 的 doc 上。⛔ `deficit <= 0` ⇒ 空表 ⇒ 逐样本不变。
#[allow(clippy::too_many_arguments)]
fn defer_dead_handover(
    sample_rate: u32,
    spf: f64,
    jobs: &[DeadJob],
    kept: &[(i64, usize, Vec<f32>)],
    order: &[usize],
    deficit: f32,
    // ⭐⭐⭐ S165 —— 音符表 + 长音延续的淡化宽度 + 基础淡化宽度。
    //    用来认出「这条缝的淡入区其实在交接点**往前** `tied_xf`」那一族。见 `handover_fade_window`。
    //    ⛔ 参数不是 env —— 判据不许读进程环境(S151 笔1)。
    notes: &[NoteSpan],
    tied_xf: usize,
    xf: usize,
    fade_window: bool,
    // ⭐⭐⭐ S165 —— 收益驱动的最小收益(dB);`0.0` = 关 = 走今天的纯门限逻辑。见 `handover_gain_db`。
    gain_db: f32,
) -> std::collections::HashMap<usize, usize> {
    let mut out = std::collections::HashMap::new();
    if deficit <= 0.0 || order.len() < 2 {
        return out;
    }
    let w0 = (f64::from(sample_rate) * HANDOVER_WIN_MS / 1000.0) as usize;
    let step = (f64::from(sample_rate) * 0.010) as usize;
    let max_defer = (f64::from(sample_rate) * HANDOVER_MAX_MS / 1000.0) as usize;
    let slice = |ji: usize| kept.iter().find(|(i, _, _)| *i == ji as i64);
    // ⭐⭐⭐ S165 —— 评估窗现在带**偏移**:`off < 0` = 在交接点**往前**取(那才是
    //    `tied_xfade` 那一族真正的淡入区)。`off = 0` = 今天的行为(往后 40 ms)。
    let lvl = |lo: usize, seg: &[f32], a: usize, off: isize, w: usize| -> f32 {
        let lo_i = a as isize + off;
        if lo_i < lo as isize || (lo_i as usize) + w > lo + seg.len() {
            return f32::NEG_INFINITY;
        }
        let s = &seg[(lo_i as usize) - lo..(lo_i as usize) - lo + w];
        let e: f64 = s.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / w as f64;
        (10.0 * (e + 1e-20).log10()) as f32
    };
    // ⭐⭐⭐ S165 —— 这条缝的淡入区在哪:与 `xf_at` 的 `tied_here` **同一套判定**
    //    (交接点落在延续音的头上 ±40 ms,或落在一个音的肚子里)。⛔ 两处判定必须一致,
    //    否则「体检看的那一段」与「真正淡化的那一段」又会错开 —— 这一刀治的正是那个错开。
    let tol = (sample_rate as usize) / 25; // 40 ms
    let tied_here = |t: usize| -> bool {
        tied_xf > xf
            && notes.iter().any(|nd| {
                if !nd.sung {
                    return false;
                }
                let s0 = ((nd.start as f64) * spf) as usize;
                let s1 = (((nd.start + nd.frames) as f64) * spf) as usize;
                if s1 <= s0 {
                    return false;
                }
                (nd.tied && t + tol >= s0 && t <= s0 + tol) || (t > s0 + tol && t + tol < s1)
            })
    };
    for oi in 0..order.len() - 1 {
        let (l, r) = (order[oi], order[oi + 1]);
        if jobs[l].shift == jobs[r].shift {
            continue;
        }
        let (Some((_, llo, lseg)), Some((_, rlo, rseg))) = (slice(l), slice(r)) else { continue };
        // 今天的交接就在右窗起点
        let t0 = ((jobs[r].start.max(0) as f64) * spf) as usize;
        // ⭐⭐⭐ S165 —— **体检窗对准这条缝真正的淡入区**。
        //
        // ⛔ 病:`tied_xfade` 把右窗的淡入起点**往前拉 `tied_xf`(120 ms)**,而体检一直
        //    只看交接点**往后 40 ms**。实测用户点名的 yuyuko × 炉心 4:36.319:
        //    往后 40 ms 落差只有 **5.27 dB**(门限 15 够不着),而**真正的淡入区**
        //    (交接点往前 120 ms)落差是 **12.55 dB** ⇒ **它结构上看不见自己该拦的那一段**。
        //    而那 120 ms 等增益淡化的功率 ≈ `0.25·Pa` = −14.58 dB,成品实测 −14.75 ⇒ 几乎吻合
        //    ⇒ 凹陷就是「把输出交给一条弱 12.55 dB 的 donor」的必然结果(S165 §54.7)。
        // ⛔⛔ **两个窗取最差,不是替换。** 第一版写成替换,实测当场退化:
        //    290.680 那一处(HANDOVER 的 doc 自己举的例子,往后 40 ms 落差 27-45 dB)
        //    在 OFF 臂被拦住、挪后 110 ms,而 ON 臂**反而不触发了** —— 因为往前那 120 ms
        //    里进来的那条并不弱。⇒ 淡入区是**额外**要看的一段,不是替代原来那一段。
        let cands: [(isize, usize); 2] = if fade_window && tied_here(t0) {
            [(0isize, w0), (-(tied_xf as isize), tied_xf)]
        } else {
            [(0isize, w0), (0isize, w0)]
        };
        let mut pick: Option<(isize, usize, f32, f32)> = None;
        for (o, ww) in cands {
            let (lv, rv) = (lvl(*llo, lseg, t0, o, ww), lvl(*rlo, rseg, t0, o, ww));
            if !lv.is_finite() || !rv.is_finite() || lv <= HANDOVER_ALIVE_DBFS {
                continue;
            }
            // 取**缺口最大**的那个窗:它决定这条缝该不该被拦,以及往后找恢复点时看哪一段。
            if pick.map_or(true, |(_, _, l, r)| lv - rv > l - r) {
                pick = Some((o, ww, lv, rv));
            }
        }
        let Some((off, w, lv0, rv0)) = pick else { continue };
        // ⭐⭐⭐ S165 —— **收益驱动**(`gain_db > 0`)取代「落差 > 绝对门限」。
        //    机理与为什么换掉门限,见 [`handover_gain_db`] 的 doc。
        let coarse = if gain_db > 0.0 { HANDOVER_COARSE_DB } else { deficit };
        if lv0 - rv0 <= coarse {
            continue;
        }
        // 这条缝在候选点 `t` 上「混合之后比【两条里较好的那条】差多少」——越小越好。
        // ⚠ 等增益淡化在 w∈[0,1] 上的平均功率 = (Pl+Pr)/3(不相干相加);
        //   实测这两条 donor 的相干度只有 0.15-0.32,近似站得住。
        let mix_deficit = |t: usize| -> f32 {
            let (lv, rv) = (lvl(*llo, lseg, t, off, w), lvl(*rlo, rseg, t, off, w));
            if !lv.is_finite() || !rv.is_finite() {
                return f32::INFINITY;
            }
            let (pl, pr) = (10f32.powf(lv / 10.0), 10f32.powf(rv / 10.0));
            let mix = 10.0 * ((pl + pr) / 3.0).max(1e-30).log10();
            lv.max(rv) - mix
        };
        let base_md = mix_deficit(t0);
        // 往后找第一个「进来的那条追上来」的点
        let mut t = t0;
        let mut fixed = None;
        if gain_db > 0.0 {
            // 收益驱动:扫完整个允许区间,取**收益最大**的那个点;不够 `gain_db` 就不动手。
            let mut best = (0.0f32, t0);
            while t + step <= t0 + max_defer {
                t += step;
                let md = mix_deficit(t);
                if !md.is_finite() {
                    break;
                }
                let g = base_md - md;
                if g > best.0 {
                    best = (g, t);
                }
            }
            if best.0 >= gain_db {
                fixed = Some(best.1);
            }
        } else {
            while t + step <= t0 + max_defer {
                t += step;
                let (lv, rv) = (lvl(*llo, lseg, t, off, w), lvl(*rlo, rseg, t, off, w));
                if !lv.is_finite() || !rv.is_finite() {
                    break;
                }
                if rv >= lv - deficit * 0.5 {
                    fixed = Some(t);
                    break;
                }
            }
        }
        match fixed {
            Some(t) => {
                tracing::info!(
                    "range: handover at {:.3}s deferred by {:.0} ms — the incoming donor {:+} was \
                     {:.1} dB below the outgoing {:+} ({:.1} vs {:.1} dBFS)",
                    t0 as f64 / f64::from(sample_rate),
                    (t - t0) as f64 * 1000.0 / f64::from(sample_rate),
                    jobs[r].shift,
                    lv0 - rv0,
                    jobs[l].shift,
                    lv0,
                    rv0
                );
                out.insert(oi, t);
            }
            None => tracing::warn!(
                "range: handover at {:.3}s — the incoming donor {:+} is {:.1} dB down and does not \
                 recover within {HANDOVER_MAX_MS:.0} ms; leaving the seam where it is",
                t0 as f64 / f64::from(sample_rate),
                jobs[r].shift,
                lv0 - rv0
            ),
        }
    }
    out
}

fn splice_kept(
    base: &mut [f32],
    sample_rate: u32,
    spf: f64,
    jobs: &[DeadJob],
    kept: &[(i64, usize, Vec<f32>)],
    xf: usize,
    join_enabled: bool,
    align: usize,
    // ⭐ S163 —— 交接点体检的门限(dB);`0.0` = 关。见 [`handover_deficit_db`]。
    handover_db: f32,
    // ⭐ S163 —— 音符表(认「长音延续」用);空 ⇒ 逐位回到今天。见 [`tied_xfade_ms`]。
    notes: &[NoteSpan],
    // ⭐ S163 —— 长音延续处的交叉淡化(样本);`0` = 关。
    tied_xf: usize,
    // ⭐⭐⭐⭐ S163 §34 —— donor 静音坑回填的门限(dB);`0` = 关 = 逐位不变。
    //    见 [`dipfill_depth_db`]。
    dipfill_db: f32,
    // ⭐⭐⭐⭐ S163 v8 —— 窗内的**休止**是否保持 `base`(见 [`REST_BASE_FADE_MS`])。
    //    ⛔ 是参数不是 env —— 判据不许读进程环境(S151 笔1)。
    rest_base: bool,
    // ⭐⭐⭐⭐⭐ S163 v16 —— 短间隙/重叠之后的窗淡入拉长(样本);`0` = 关。见 [`SEAM_RAMP_MS`]。
    seam_ramp: usize,
    // ⭐⭐⭐⭐⭐ S163 v17 —— **起音形状整形**(见 [`ONSET_FIT_MS`])。
    onset_fit: bool,
    // ⭐ S165 —— 对齐的搜索窗用**这一条缝实际的淡化宽度**(见 `seam_align_wide`)。
    //    ⛔ 参数不是 env —— 判据不许读进程环境(S151 笔1)。
    wide_align: bool,
) -> crate::Result<()> {
    // ⛔ `join_enabled` 是参数不是 env —— 判据不许读进程环境(S151 笔1)。
    let n = base.len();
    // ⛔ 时间序,不是位移序。窗互不重叠时两者等价(实测这首歌 0 对重叠),但**接上之后
    // 相邻两窗会重叠一个淡化宽度**,而那一次淡化的意义正是「从上一条 donor 淡到下一条」——
    // 顺序错了就会淡回 base,也就是这一刀本来要消灭的那个洞。
    let mut order: Vec<usize> = (0..jobs.len()).collect();
    order.sort_by_key(|&i| (jobs[i].start, jobs[i].end));
    let mut join = join_rests(base, sample_rate, spf, jobs, kept, &order, xf, join_enabled);
    // ⭐⭐ S163 —— 交接点体检**优先**:它治的是「掉进静音」,比「缝落在哪儿」严重一个量级。
    join.extend(defer_dead_handover(
        sample_rate,
        spf,
        jobs,
        kept,
        &order,
        handover_db,
        notes,
        tied_xf,
        xf,
        handover_fade_window(),
        handover_gain_db(),
    ));
    // ⭐⭐ S163 —— **长音延续处的接缝改成长交叉淡化**(用户:「一个长音三个听感 / 长音割裂」)。
    // ⛔ 做法必须走 `join` 那一套:左窗**硬写到交接点**(不淡出)、右窗**从交接点往前**
    //    一个淡化宽度开始淡入 ⇒ 淡入区里另一侧是**左窗的 donor**。
    //    第一版只把淡入拉长而没挪起点 ⇒ 淡入区另一侧是 `base` ⇒ **挖了一个 120 ms 的坑**。
    let mut xf_at: std::collections::HashMap<usize, usize> = Default::default();
    if tied_xf > xf && !notes.is_empty() && order.len() > 1 {
        let tol = (sample_rate as usize) / 25; // 40 ms
        let tied_here = |t: usize| -> bool {
            notes.iter().any(|nd| {
                if !nd.sung {
                    return false;
                }
                let s0 = ((nd.start as f64) * spf) as usize;
                let s1 = (((nd.start + nd.frames) as f64) * spf) as usize;
                if s1 <= s0 {
                    return false;
                }
                // ⓐ 交接点落在一个**延续音**的头上(±40 ms)⇒ 同一个长音被切开了
                (nd.tied && t + tol >= s0 && t <= s0 + tol)
                    // ⓑ 交接点落在一个音的**肚子里**(离两端都 >40 ms)⇒ 也是把一个音切开
                    || (t > s0 + tol && t + tol < s1)
            })
        };
        for oi in 0..order.len() - 1 {
            let (l, r) = (order[oi], order[oi + 1]);
            if jobs[l].shift == jobs[r].shift {
                continue;
            }
            // ⛔ 交接点体检**优先**:它已经挪过的点尊重它,只把淡化拉长。
            let t = join
                .get(&oi)
                .copied()
                .unwrap_or_else(|| ((jobs[r].start.max(0) as f64) * spf) as usize);
            if !tied_here(t) {
                continue;
            }
            join.insert(oi, t);
            xf_at.insert(oi + 1, tied_xf);
        }
        // ⭐⭐⭐⭐⭐ S163 v16 —— **短间隙/重叠之后的窗，淡入拉长**(见 [`SEAM_RAMP_MS`])。
        //    靶子是 workflow 量出来的「短间隙族」(间隙 ≤200 ms 且移调改变 ⇒ 富集 **×275**),
        //    杠杆是它同时给出的剂量曲线(ramp 0-2 ms → 34.9% 热,≥30 ms → **5.9%**;
        //    Spearman **−0.471, p=2.2e-20**)。
        //    ⛔ 与 tied 的拉长取 `max`,不互相覆盖。
        if seam_ramp > 0 {
            let mut n_ramp = 0usize;
            for oi in 1..order.len() {
                let (l, r) = (&jobs[order[oi - 1]], &jobs[order[oi]]);
                // 移调没变 ⇒ 两侧是同一条 donor,不是这一族(实测位移差 0 的对照不热)
                if l.shift == r.shift {
                    continue;
                }
                // 间隙 ≤200 ms(10 帧)**或重叠**(负间隙,实测恰好都是 −80 ms)。
                // ⛔ `gap == 0`(两窗紧邻)**不在靶子里** —— workflow 的定义是
                //    「窗与窗之间夹的**一小段未救援 base**」,紧邻时中间根本没有 base。
                //    ⚠ 这一条同时让 `tied_xfade_ms` 的判据保持隔离
                //    (它的夹具正是 shift −4 → −9、gap = 0)。
                let gap = r.start - l.end;
                if gap == 0 || gap > 10 {
                    continue;
                }
                let e = xf_at.entry(oi).or_insert(xf);
                if *e < seam_ramp {
                    *e = seam_ramp;
                    n_ramp += 1;
                }
            }
            if n_ramp > 0 {
                tracing::info!(
                    "range: {} seam(s) after a short gap widened to {:.0} ms of fade-in",
                    n_ramp,
                    seam_ramp as f64 * 1000.0 / f64::from(sample_rate)
                );
            }
        }
        if !xf_at.is_empty() {
            tracing::info!(
                "range: {} seam(s) inside a sustained note widened to {:.0} ms of donor-to-donor crossfade",
                xf_at.len(),
                tied_xf as f64 * 1000.0 / f64::from(sample_rate)
            );
        }
    }
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
                // ⭐ S163 —— 往前挪的量 = **这个窗自己的淡入宽度**(长音延续处是 120 ms)。
                a = t.saturating_sub(xf_at.get(&oi).copied().unwrap_or(xf)).min(n);
            }
        }
            // 窗短于双淡化 → 收缩淡化宽度;完全空窗才放弃,响亮。
            // 写入范围必须落在留下来的片段里。今天由构造一定成立(余量 = 桥接上限),
            // 但「窗边被挪出片段」是一条会静默出错的路 ⇒ 夹紧并响亮。
            // ── S159zm —— **先对齐,再淡。**淡入区上按互相关搜一个整数样本偏移 `d`,
            // 让进来的这条臂与 `base` 里**已经写好的内容**对齐。实测两条臂的 |ρ| 从 0.139
            // 抬到 0.928(见 [`SEAM_ALIGN_MS_DEFAULT`] 的那张表)。
            // ⛔ 只在**真的要淡入**的时候搜(`a > 0`);缓冲头上没有可对齐的东西。
            // ⚠ 整窗一起挪 ⇒ 时序漂移 ≤ 旋钮的毫秒数;窗的另一头有它自己那条缝,
            //   会在轮到它时相对**挪过之后的** base 再对一次 ⇒ 自洽。
            let d: isize = if align > 0 && a > 0 && a >= *seg_lo {
                // ⭐⭐⭐ S165 —— **搜索窗必须是【这一条缝实际的淡化宽度】,不是基础的 10 ms。**
                //
                // ⛔ 这一行原本是 `xf.min(...)`,而 `tied_xfade` 那一族的淡入是 **120 ms**:
                //    拿 10 ms 的窗搜出一个 lag,却把整个 120 ms 的片段按它整体平移。
                //    10 ms @ 40 kHz 在 f0≈500 Hz 上只有 **5 个周期** ⇒ 互相关有**周期歧义**,
                //    锁到错的周期。实测用户点名的两处(yuyuko × 炉心 4:29.265 / 4:36.319):
                //
                //    | | 120 ms 窗上的最佳 lag | 对齐后 r | 不挪 r | 生产挪的量 | 挪完 r |
                //    |---|---|---|---|---|---|
                //    | 4:29 | +21 | 0.854 | 0.322 | **−48** | **−0.209** |
                //    | 4:36 | **+154** | 0.800 | −0.177 | **+73** | **−0.065** |
                //
                //    ⇒ **挪完比不挪还相消**,谐波被削掉、噪声底露出来 =
                //      用户在频谱上看到的「f0 附近突然多出面状噪声」(S165 §54)。
                //
                // ⚠ 半径也要跟着够:4:36 需要 154 样本 = 3.85 ms,而出厂半径是 2.0 ms。
                //   **两者必须一起改** —— 只加大半径实测无效(§54.5:4:29 反而 5.66 → 6.32)。
                let w = if wide_align {
                    xf_at.get(&oi).copied().unwrap_or(xf).min(b.saturating_sub(a))
                } else {
                    xf.min(b.saturating_sub(a))
                };
                let r = align as isize;
                let dot = |off: isize| -> f64 {
                    let lo = a as isize + off - *seg_lo as isize;
                    if lo < 0 || (lo as usize) + w > seg.len() || a + w > n {
                        return f64::NEG_INFINITY;
                    }
                    let (u, v) = (&base[a..a + w], &seg[lo as usize..lo as usize + w]);
                    let (mut num, mut du, mut dv) = (0.0f64, 0.0f64, 0.0f64);
                    for i in 0..w {
                        let (x, y) = (f64::from(u[i]), f64::from(v[i]));
                        num += x * y;
                        du += x * x;
                        dv += y * y;
                    }
                    if du <= 0.0 || dv <= 0.0 {
                        return f64::NEG_INFINITY;
                    }
                    num / (du * dv).sqrt()
                };
                // ⛔ 与 `0` 比而不是与 `NEG_INFINITY` 比:一个够不着的偏移不许赢过「不挪」。
                let base_score = dot(0);
                let mut best = (0isize, base_score);
                if w >= 8 && base_score.is_finite() {
                    for off in -r..=r {
                        let sc = dot(off);
                        if sc > best.1 {
                            best = (off, sc);
                        }
                    }
                }
                best.0
            } else {
                0
            };
            let (sa, sb) = (*seg_lo, *seg_lo + seg.len());
            if a < sa || b > sb {
                tracing::warn!(
                    "range-extend(dead-only): window {a}..{b} escapes its donor segment {sa}..{sb} — clamped"
                );
            }
            let a = a.max(sa);
            let b = b.min(sb);
            // ⭐⭐ S163 —— 淡入宽度逐窗给(长音延续处拉长,见 [`tied_xfade_ms`]);
            // 淡出永远是 10 ms —— 被拉长的那一侧一定是 `hard_end`(左窗硬写到交接点),
            // 所以它根本不淡出。
            let xf_in = xf_at.get(&oi).copied().unwrap_or(xf);
            let xfw = xf_in.min((b.saturating_sub(a)) / 2);
            // ⭐⭐ S163 —— **窗尾落在一个长音的头上时,淡出也要拉长**(用户点名 akiko 2:50.4)。
            // `post = GUARD_FRAMES` 把窗尾伸进下一个唱音 40 ms,然后 10 ms 就切回 `base` ⇒
            // 一个 1.46 s 的音头 40 ms 是 donor、之后是 base ⇒ **断裂**。
            // ⛔ 收窄护栏已判负(重叠 80→0 瞬变中位差 7.60 dB)⇒ 治的是「切得太快」。
            // ⭐ 淡出的另一侧是 `base`,而那个音**没被救 = 模型本来就唱得动** ⇒ 慢慢过渡听不出。
            let xf_out_here = if tied_xf > xf && !notes.is_empty() {
                let room = (sample_rate as usize) * TAIL_ROOM_MS as usize / 1000;
                let in_long_note = notes.iter().any(|nd| {
                    if !nd.sung {
                        return false;
                    }
                    let s0 = ((nd.start as f64) * spf) as usize;
                    let s1 = (((nd.start + nd.frames) as f64) * spf) as usize;
                    b > s0 && b + room < s1
                });
                if in_long_note { tied_xf } else { xf }
            } else {
                xf
            };
            let xfw_out = xf_out_here.min((b.saturating_sub(a)) / 2);
            if b <= a || xfw == 0 || xfw_out == 0 {
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
            // ⭐ S163 —— 诊断:实际写入区间。`UTAI_RANGE_TRACE_SPLICE=1` 打开。
            // ⛔ 只读 env 做**日志**,不参与任何判据(判据不许读进程环境 —— S151 笔1)。
            if std::env::var("UTAI_RANGE_TRACE_SPLICE").is_ok_and(|v| v.trim() == "1") {
                let ts = |u: usize| u as f64 / f64::from(sample_rate);
                tracing::info!(
                    "splice#{oi} shift {:+} write {:.3}..{:.3}s seg {:.3}..{:.3}s \
                     hard_end={hard_end} fade_in={fade_in} fade_out={fade_out} \
                     xf_in={:.0}ms xf_out={:.0}ms align={d}",
                    j.shift,
                    ts(a),
                    ts(b),
                    ts(sa),
                    ts(sb),
                    xfw as f64 * 1000.0 / f64::from(sample_rate),
                    xfw_out as f64 * 1000.0 / f64::from(sample_rate),
                );
            }
            // ⭐⭐⭐⭐ S163 §34 —— **donor 静音坑回填**(见 [`dipfill_depth_db`])。
            // 出厂 `0.0` ⇒ `spans` 恒空 ⇒ 逐位不变。
            // ⭐⭐⭐⭐ S163 v2 —— 谐波收益闸要 `base` 的同一段与该段的 f0。
            //    `seg` 覆盖 base 的 `[seg_lo, seg_lo + seg.len())`。
            // ⛔⛔⛔ S163 v3 —— **判据只许看真正会写出去的那一段**。
            //    `seg` 两端各带 `MERGE_BRIDGE_FRAMES`(25 帧 ≈ **500 ms**)的窗外材料,
            //    而写入只覆盖 `[a, b)`。v2 把整条 `seg` 交给判据 ⇒ 两条错:
            //    ⑴ 边距里的坑**永远不生效**(日志报的回填量因此远大于实际);
            //    ⑵ `med`(存活中位数 = 判据的地板)**在含窗外材料的全段上算** ⇒ 地板被污染。
            //    ⚠ `base` 要取**同一时间轴**:`seg[si]` 写到 `base[si + seg_lo - d]`。
            let si_a = (a as isize - *seg_lo as isize + d).clamp(0, seg.len() as isize) as usize;
            let si_b = (b as isize - *seg_lo as isize + d).clamp(0, seg.len() as isize) as usize;
            let (bl, bh) = {
                let off = *seg_lo as isize - d;
                (
                    (off + si_a as isize).clamp(0, base.len() as isize) as usize,
                    (off + si_b as isize).clamp(0, base.len() as isize) as usize,
                )
            };
            let bslice: &[f32] = if bh > bl { &base[bl..bh] } else { &base[0..0] };
            // 该窗中点落在哪个唱音里 ⇒ 用它的 f0。找不到 ⇒ 0.0 ⇒ 闸不生效。
            let mid_frame = ((a + b) / 2) as f64 / spf;
            let f0_here = notes
                .iter()
                .find(|nd| {
                    nd.sung
                        && (nd.start as f64) <= mid_frame
                        && mid_frame < (nd.start + nd.frames) as f64
                })
                .map(|nd| nd.hz)
                .unwrap_or(0.0);
            let dips: Vec<(usize, usize)> = if si_b > si_a {
                dipfill_spans(&seg[si_a..si_b], sample_rate, dipfill_db, bslice, f0_here)
                    .into_iter()
                    .map(|(x, y)| (x + si_a, y + si_a))
                    .collect()
            } else {
                Vec::new()
            };
            let dip_fade = ((f64::from(sample_rate) * f64::from(DIPFILL_FADE_MS) / 1000.0) as usize).max(1);
            if !dips.is_empty() {
                tracing::info!(
                    "range: splice#{oi} shift {:+} — dipfill filled {} silent pit(s) ({:.0} ms total)",
                    j.shift,
                    dips.len(),
                    dips.iter().map(|&(x, y)| y - x).sum::<usize>() as f64 * 1000.0
                        / f64::from(sample_rate)
                );
            }
            // ⛔ 坑是按 `seg` 的下标找的 ⇒ 判断必须用 `si`,不是 `k`。
            let dip_keep = |si: usize| -> f32 {
                let mut keep = 1.0f32;
                for &(x, y) in &dips {
                    if si >= x && si < y {
                        return 0.0;
                    }
                    // 两侧各一个淡化宽度线性过渡,不许硬切
                    if si + dip_fade > x && si < x {
                        keep = keep.min((x - si) as f32 / dip_fade as f32);
                    }
                    if si >= y && si < y + dip_fade {
                        keep = keep.min((si - y) as f32 / dip_fade as f32);
                    }
                }
                keep
            };
            // ⭐⭐⭐⭐ S163 v8 —— **救援不许碰空拍**(见 [`REST_BASE_FADE_MS`])。
            //    先把这条窗里的休止段找出来(少数几段),逐样本只查这几段。
            let rest_spans: Vec<(usize, usize)> = if rest_base {
                notes
                    .iter()
                    .filter(|nd| !nd.sung)
                    .filter_map(|nd| {
                        let x = ((nd.start.max(0) as f64) * spf) as usize;
                        let y = (((nd.start + nd.frames).max(0) as f64) * spf) as usize;
                        // ⭐ v10:整格都参与,头尾各留一段做**增益渐变**(不是裁掉)。
                        let hg = (f64::from(sample_rate) * f64::from(REST_HEAD_GUARD_MS) / 1000.0) as usize;
                        let x = x.saturating_add(hg).max(a);
                        let y = y.min(b);
                        (y > x).then_some((x, y))
                    })
                    .collect()
            } else {
                Vec::new()
            };
            // ⭐⭐⭐⭐ S163 v10 —— 每段休止一个**增益**(压到 `base` 的电平),不换内容。
            //    ⛔ 只在这里算一次,不许放进逐样本循环。
            let hf = ((f64::from(sample_rate) * f64::from(REST_GAIN_FADE_MS) / 1000.0) as usize).max(1);
            let tf = ((f64::from(sample_rate) * f64::from(REST_TAIL_GUARD_MS) / 1000.0) as usize).max(1);
            let rest_g: Vec<f32> = rest_spans
                .iter()
                .map(|&(x, y)| {
                    let e = |v: &[f32]| -> f64 {
                        if v.is_empty() {
                            return 0.0;
                        }
                        v.iter().map(|&z| f64::from(z) * f64::from(z)).sum::<f64>() / v.len() as f64
                    };
                    // donor 在这段休止上(`seg` 坐标,带对齐偏移 `d`)
                    let sx = (x as isize - *seg_lo as isize + d).clamp(0, seg.len() as isize) as usize;
                    let sy = (y as isize - *seg_lo as isize + d).clamp(0, seg.len() as isize) as usize;
                    let dn = if sy > sx { e(&seg[sx..sy]) } else { 0.0 };
                    let bs = e(&base[x.min(base.len())..y.min(base.len())]);
                    if dn <= 0.0 || bs <= 0.0 {
                        return 1.0;
                    }
                    // ⛔ 只压不抬:抬会把 donor 的伪影一起放大。
                    let g_db = (10.0 * (bs / dn).log10()) as f32 * 0.5;
                    10f32.powf(g_db.clamp(REST_GAIN_MIN_DB, 0.0) / 20.0)
                })
                .collect();
            // 头 `hf` 从 1 渐变到 g;尾 `tf` 从 g 回到 1(辅音 preroll 落在尾部,必须原电平)。
            let rest_w = |k: usize| -> f32 {
                let mut out = 1.0f32;
                for (&(x, y), &g) in rest_spans.iter().zip(&rest_g) {
                    if k >= x && k < y {
                        let head = ((k - x) as f32 / hf as f32).min(1.0);
                        let tail = ((y - 1 - k) as f32 / tf as f32).min(1.0);
                        let t = head.min(tail);
                        out = out.min(1.0 + (g - 1.0) * t);
                    }
                }
                out
            };
            // ⭐⭐⭐⭐⭐ S163 v17 —— **起音形状整形**(见 [`ONSET_FIT_MS`])。
            //    逐 2 ms 比较 `base` 与 donor **各自归一化到自己稳态之后**的上升曲线,
            //    把 donor 的形状拉回 `base` 的 ⇒ **稳态电平一个字节不动**。
            //    ⛔ 这正是 v14(砍电平)/v16(整体延后)做错的地方。
            let ofit_cell = ((f64::from(sample_rate) * 0.002) as usize).max(1);
            let ofit: Vec<f32> = if onset_fit {
                let span = (f64::from(sample_rate) * f64::from(ONSET_FIT_MS) / 1000.0) as usize;
                let s_lo = (f64::from(sample_rate) * f64::from(ONSET_FIT_STEADY_LO_MS) / 1000.0) as usize;
                let s_hi = (f64::from(sample_rate) * f64::from(ONSET_FIT_STEADY_HI_MS) / 1000.0) as usize;
                let ncell = (b - a) / ofit_cell + 1;
                let mut g = vec![1.0f32; ncell];
                let e = |v: &[f32]| -> f64 {
                    if v.is_empty() {
                        return 0.0;
                    }
                    v.iter().map(|&z| f64::from(z) * f64::from(z)).sum::<f64>() / v.len() as f64
                };
                let si_of = |k: usize| -> usize {
                    (k as isize - *seg_lo as isize + d).clamp(0, seg.len() as isize - 1) as usize
                };
                for nd in notes.iter().filter(|n| n.sung) {
                    let on = ((nd.start.max(0) as f64) * spf) as usize;
                    if on < a || on + span > b {
                        continue;
                    }
                    // 稳态：两侧各自取自己的
                    let (q0, q1) = ((on + s_lo).min(b), (on + s_hi).min(b));
                    if q1 <= q0 {
                        continue;
                    }
                    let sb = e(&base[q0.min(base.len())..q1.min(base.len())]);
                    let sd = e(&seg[si_of(q0)..si_of(q1).max(si_of(q0) + 1)]);
                    if sb <= 0.0 || sd <= 0.0 {
                        continue;
                    }
                    let mut p = on;
                    while p < on + span {
                        let q = (p + ofit_cell).min(b);
                        if q <= p {
                            break;
                        }
                        let eb = e(&base[p.min(base.len())..q.min(base.len())]);
                        let ed = e(&seg[si_of(p)..si_of(q).max(si_of(p) + 1)]);
                        if eb > 0.0 && ed > 0.0 {
                            // 各自归一化到自己的稳态 ⇒ 只剩**形状**差
                            let shape = ((eb / sb) / (ed / sd)).sqrt() as f32;
                            let idx = (p - a) / ofit_cell;
                            if idx < g.len() {
                                let db = 20.0 * shape.max(1e-6).log10();
                                // ⛔⛔ **只压不抬**：上限硬钉在 0 dB。
                                //    v17 第一版允许 +9 dB ⇒ donor 本来就贴着满刻度，
                                //    一放大就削：splice 后峰值 **+8.83 dBFS**、
                                //    **39193 个样本 |x|≥0.999**，后续归一化为容纳它
                                //    把整条压了 **7 dB**，16-24 kHz 因削波失真 **+21.5 dB**。
                                //    ⇒ 这正是 v14 已经写过的教训，我又丢了一次。
                                g[idx] = 10f32.powf(db.clamp(-ONSET_FIT_MAX_DB, 0.0) / 20.0);
                            }
                        }
                        p = q;
                    }
                }
                // 10 ms 移动平均:增益自己不许跳变(4 ms 时 16-24 kHz 涨了 21.5 dB)
                let half = ((0.005 * f64::from(sample_rate)) as usize / ofit_cell.max(1)).max(1);
                let src = g.clone();
                for i in 0..g.len() {
                    let (l, r) = (i.saturating_sub(half), (i + half + 1).min(src.len()));
                    g[i] = src[l..r].iter().copied().sum::<f32>() / (r - l) as f32;
                }
                g
            } else {
                Vec::new()
            };
            for k in a..b {
                let w = if fade_in && k < a + xfw {
                    0.5 - 0.5 * (std::f32::consts::PI * (k - a) as f32 / xfw as f32).cos()
                } else if fade_out && k >= b - xfw_out {
                    0.5 - 0.5 * (std::f32::consts::PI * (b - k) as f32 / xfw_out as f32).cos()
                } else {
                    1.0
                };
                // S159zm —— 带上对齐偏移 `d`;越界时退回 0(不挪),而不是 panic。
                let si = (k as isize - *seg_lo as isize + d).clamp(0, seg.len() as isize - 1);
                let w = w * dip_keep(si as usize);
                // ⭐ v10 —— 休止段:donor **压电平**,不换内容 ⇒ 没有内容跳变 ⇒ 没有缝。
                let rg = rest_w(k);
                // ⭐ v17 —— 起音**形状**整形(稳态不动,只改上升曲线)。
                // ⛔⛔ **逐样本线性插值,不许用格值阶梯**。
                //    v17a 用 `ofit[(k-a)/cell]` 取格值 ⇒ 每 2 ms 一个阶梯边缘,
                //    那本身就是宽带瞬变:整曲频谱 **16-24 kHz +21.5 dB**
                //    (0-16 kHz 已经正常 ⇒ 与削波无关,是增益调制自己产生的高频)。
                let og = if ofit.is_empty() {
                    1.0
                } else {
                    let t = (k - a) as f32 / ofit_cell as f32;
                    let i0 = t as usize;
                    let f = t - i0 as f32;
                    let g0 = ofit.get(i0).copied().unwrap_or(1.0);
                    let g1 = ofit.get(i0 + 1).copied().unwrap_or(g0);
                    g0 + (g1 - g0) * f
                };
                base[k] = base[k] * (1.0 - w) + seg[si as usize] * w * rg * og;
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

/// ⚙ 出厂默认 = false —— 关(S163 v11 翻过又判负,见下)
/// `UTAI_RANGE_JOIN=1` 打开「异位移短休止按音频接上」。**默认关 ⇒ 生产逐位不变。**
///
/// # ⛔ S163 v11 判负(同一次 run,零噪声,只差 join 一个变量)
/// 用户点名的六个坐标 **Δ 全部 = +0.0 dB**;三处音头 ±60 ms 内**逐位差异 0 样本**
/// ⇒ 它在这些地方**一个样本都没动**。那些休止只有 **120-140 ms**,
/// 不满足这个函数自己的前置条件。⛔ 别再重试,除非先改前置条件。
///
/// # S163 v11 为什么翻它
/// 用户 2026-08-27 点名的空拍伪影,归因锁死在**窗边界**上:
/// ```text
/// yachiyo 1:23.081  休止 120 ms 被两条窗切开:shift −5(前 40 ms) / −17(后 60 ms) ← 差 12 半音
///         3:40.829  shift −5 / −10
///         1:58.380  shift −10 / −13
/// ```
/// 空拍里坐着**两条位移差 5-12 半音的 donor 的交接** —— 正是这个函数写来解决的那件事。
/// ⚠ S152 判负它时,靶子(46.041 s)是个非事件;这一次的靶子是**用户亲耳点名的坐标**。
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

// ⭐⭐⭐⭐ S163 v11 —— 出厂从 false 翻到 true。
//
// # 为什么现在翻(S152 判负它时的靶子是个非事件,现在的靶子是用户亲耳点名的)
// 用户 2026-08-27 点名的空拍伪影,归因已经锁死到**窗边界**上:
// ```text
// yachiyo 1:23.081  休止 120 ms 被两条窗切开:shift −5(前 40 ms) / −17(后 60 ms) ← 差 12 半音
//         3:40.829  shift −5 / −10
//         1:58.380  shift −10 / −13
// ```
// 空拍里坐着**两条位移差 5-12 半音的 donor 的交接**,而 `join_rests` 正是为
// 「两个位移不同、中间只隔休止的窗」写的:把切换点放到休止里**两侧电平最接近**的那一点,
// 而不是让两条窗各写一半。
//
// ⚠ 它与休止增益(v10)不冲突:join 管**切换点**,增益管**电平**。

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
/// # ⛔⛔ S159zze —— **这个旋钮在它自己 doc 写明的工作点上把移调整个抵消掉。别再开它。**
///
/// 实测(鹅妈妈 +7 × 东雪莲的 −14 donor 喂 `inverse_probe`,逐音自相关测 f0,
/// 相对目标音高的中位误差 / 偏离 >50 cents 的音占比):
///
/// | radius | f0 误差 | >50 c |
/// |---|---|---|
/// | 0(出厂) | **−9.8 c** | 13 % |
/// | 0.06 | −13.7 c | 34 % |
/// | **0.15**(下面那段推荐的工作点) | **−1228 c** | **89 %** |
/// | 0.30 | −1294 c | 100 % |
/// | 0.50 | −1410 c | 100 % |
///
/// **−1228…−1410 cents ≈ −13…−14 个半音 = 正好把 ratio 2.24 抵消回去。**
/// 机理是构造性的:`wsola_pick` 找「与累加器最像」的源读点,而**最像的就是源自己的下一个周期**
/// ⇒ 半径一旦够得着一个周期,搜索就把 PSOLA 退化成**恒等变换**。
///
/// ⭐⭐ 下面那段 S148 的取舍表(陷波率 4.80 → 0.38 %,ΔHNR −0.15 → −4.70)**作废** ——
/// 那两把尺子**都看不见音高**,于是一个「静默关掉移调」的旋钮被登记成了「一笔单调取舍」。
/// 这正是 `scripts/range_rulers/README.md` 开篇那次四个月事故的**镜像**:
/// 那次是闸**只**量音高,这次是闸**唯独不**量音高。
/// ⇒ 凡在这条线上评一把刀,**先过音高闸**(`inverse_probe` 现在无条件打印它)。
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

/// ⚙ 出厂默认 = 0.0 = 关 —— `UTAI_PSOLA_DEJITTER=<alpha>`:颗粒**读点**去抖的强度。
///
/// 机理、受控实验的读数表、以及它与 WSOLA 的根本区别,全在
/// `utai_dsp::psola::dejitter_marks` 的 doc(**别在这里再写一份**)。
/// ⛔ 翻它要成对 bump `RANGE_ALGO_VERSION` ↔ `audition_cache_tag`,而且必须先过盲测。
pub fn dejitter() -> f64 {
    parse_dejitter(std::env::var("UTAI_PSOLA_DEJITTER").ok().as_deref())
}

/// The env parse, as a pure function so it can be asserted without touching process state.
fn parse_dejitter(v: Option<&str>) -> f64 {
    v.and_then(|v| v.trim().parse().ok())
        .filter(|v: &f64| v.is_finite() && (0.0..=1.0).contains(v))
        .unwrap_or(DEJITTER_DEFAULT)
}

/// ⛔ Changing this changes the audio ⇒ pair-bump `RANGE_ALGO_VERSION` and `audition_cache_tag`.
const DEJITTER_DEFAULT: f64 = 0.0;

/// ⚙ 出厂默认 = 0.0 = 关 —— `UTAI_RANGE_FORMANT_KNEE=<c>`:**共振峰跟随只在膝盖以外起作用。**
///
/// # ⛔⛔ 用户耳判**判负**(2026-08-24)—— 旋钮留着,只为把下面这套证据链留在原地
///
/// 用户原话:「**这一刀的听感也不好**……我感觉我们可能真的在追错误方向……到头来可能还是得用 lb3」;
/// 而且「**开了二级拆组的 lb3 反而是 3:16 和其他几处我说『油』的地方听起来最好的 ——
/// 甚至比我们刚才那一刀 on 的听感还好**」。
///
/// ⭐⭐⭐ **最要紧的一条**:`LANDING_DEFAULT = Some(3)` ⇒ **用户听下来最正常的那一版就是出厂默认**。
/// 这条线从头到尾追的是 **lb5**(落点预算 5 的实验臂),**它不是出厂配置** ⇒ 这一整场**没有造出退化**。
///
/// ## ⛔ 它为什么修不动(这条比读数值钱)
///
/// 靶子(H1−H3/H1−H4 偏离)是**真的**:两模型 × 两谱面 × 三母音 × 空对照,单调剂量曲线。
/// 这一刀也**真的把它砍掉了 55-85 %**。但**耳朵不买账**,原因有两条,都在耳判里被点名:
///
/// 1. **κ 一动就有音色差** —— 「本身我们这一刀动了 κ 那无论如何花栗鼠和音色区别肯定还是有」;
/// 2. **句内跳变**(S159zzn 量过:lb5 有 25 句句内共振峰位移跨度 ≥2 半音,lb3 只有 9)——
///    根源是**二级拆组把一句拆成不同深度**,而**一个 donor pass 只有一个 κ**
///    ⇒ 同一句被不同 pass 拼起来。⛔ 二级拆组本身是**验证过有用**的(S159zi,用户耳判
///    「确实干净了不少」),所以不能为了这一刀退掉它。
///
/// ⇒ **要让这条方向有机会,κ 必须做成【逐窗】的**(同一句内统一),而那是架构改动。
/// 在那之前,**别再用整体 κ 去修这个偏离**。
///
/// ## ⭐ 留下来的东西(下一次别重造)
///
/// * 验收台 `scripts/range_rulers/rescue_bench.py`(门/靶/两条护栏/对照,先立台后动刀);
/// * 靶子本身仍然有效,而且是这条线上**唯一**跨模型跨谱面验过的量;
/// * 「膝盖以内按构造恒等」这个写法值得复用 —— 它拦住了常数 κ 在 −6 上的过冲。
///
/// 位移 `s` 半音时,共振峰搬 `c · max(0, |s| − FORMANT_KNEE_ST)` 个半音(符号同 `s`)。
/// ⇒ **|s| ≤ 6 时恒等于 0,安全区按构造不受影响**,不是靠调参调出来的。
///
/// ## 靶子(S159zzh→zzk 验过的那一个,不是我挑的)
///
/// 同一批音、**同一个输出音高**、救 vs 不救:被救音的 **H1−H3 / H1−H4 偏高**
/// = 上方谐波供电不足 = 音源谱倾斜偏离。验收面 **两个模型 × 两个谱面 × 三个母音 × 空对照**:
/// 鹅妈妈 137 音 H1−H3 均值 −2 **−0.37** · −4 −0.08 · −6 **−1.40** · −8 +1.44 · −10 +4.58 ·
/// −12 +7.02 · −14 **+11.05**(完全单调,**−6 以内 ≈ 0**);炉心融解 142 音 −6 +1.28 · −14 **+10.60**;
/// akiko 上同样成立(用户 2026-08-24 主动要求验的跨模型)。
/// ⇒ **膝盖取 6 半音是从这条曲线读出来的**,与 S159zzb 那条 `envmod` 膝盖(≤8 无效应)独立一致。
///
/// ## 实测(`c = 0.5`,整台走 `scripts/range_rulers/rescue_bench.py`)
///
/// | 谱面 · 深度 | H1−H3 | H1−H4 | ⛔次基频 | ⛔梳 | ⛔倾斜 |
/// |---|---|---|---|---|---|
/// | 鹅妈妈 −10 出厂 | +4.58 | +4.69 | −1.60 | +6.65 | −1.00 |
/// | 鹅妈妈 −10 **带膝盖** | **+1.49** | **+0.68** | −2.09 | +7.48 | −3.48 |
/// | 鹅妈妈 −14 出厂 | +11.05 | +7.26 | −0.12 | +5.02 | −1.42 |
/// | 鹅妈妈 −14 **带膝盖** | **+3.81** | **+3.10** | −0.68 | +4.33 | −2.46 |
/// | 炉心 −14 出厂 | +10.60 | +10.34 | −1.96 | +1.22 | −0.62 |
/// | 炉心 −14 **带膝盖** | **+6.45** | **+6.35** | −2.09 | +0.05 | −4.31 |
///
/// ⇒ 靶子砍掉 **55-85 %**(炉心 40 %),**两个谱面都成立**。
///
/// ## ⛔ 已登记的代价与护栏
///
/// * ⚠ **8-12 kHz 倾斜一致地掉 2-3 dB**(出厂各档 −0.04…−3.05 → 带膝盖 −2.46…−4.31)。
///   机理大概是把包络往上搬之后顶端没内容可搬。
///   ⛔ **不许用高频 shelf 去补** —— 用户 2026-08-24:「小心不要去手动画波形或者去开 EQ,
///   那样损伤听感还吃音质」。这条**只作为已知代价登记,交给耳朵裁**。
/// * ✅ **面状伪影那条护栏(次基频 0.25-0.75 f0)全程往下走**(−0.68 / −2.09 / −2.09)。
///   用户原话「那个面状伪影问题非常神秘,动不动就会回来」⇒ 这条是**必查项**,不是可选项。
/// * ✅ 谐波梳深度与电平基本不动(κ 不改它们)。
/// * ⛔ **常数 κ 判负**:`κ = 0.30` 在 −6 上**既过冲又白掉 3 dB 高频**(H1−H4 +1.83 → **−3.56**,
///   倾斜 −0.04 → **−3.00**),而那一档本来是干净的。**膝盖不是装饰,是它能不能用的前提。**
///
/// ⛔ 翻默认要成对 bump `RANGE_ALGO_VERSION` ↔ `audition_cache_tag`,而且**必须先过盲测**。
pub fn formant_knee() -> f64 {
    parse_formant_knee(std::env::var("UTAI_RANGE_FORMANT_KNEE").ok().as_deref())
}

/// The env parse, as a pure function so it can be asserted without touching process state.
fn parse_formant_knee(v: Option<&str>) -> f64 {
    v.and_then(|v| v.trim().parse().ok())
        .filter(|v: &f64| v.is_finite() && (0.0..=1.0).contains(v))
        .unwrap_or(FORMANT_KNEE_DEFAULT)
}

/// ⛔ Changing this changes the audio ⇒ pair-bump `RANGE_ALGO_VERSION` and `audition_cache_tag`.
const FORMANT_KNEE_DEFAULT: f64 = 0.0;

/// 膝盖的位置(半音)。⛔ 它是从 S159zzk 那条剂量曲线读出来的,不是调出来的 ——
/// H1−H3 在 −2/−4/−6 上是 −0.37 / −0.08 / −1.40(≈0),到 −8 才 +1.44。
const FORMANT_KNEE_ST: f64 = 6.0;

/// 位移 `semis` 半音时,共振峰实际搬多少半音。`knee == 0` ⇒ 退回 `κ · semis`(今天,逐位不变)。
fn formant_shift_semitones(semis: f64, kappa: f32, knee: f64) -> f64 {
    if knee <= 0.0 {
        return f64::from(kappa) * semis;
    }
    knee * (semis.abs() - FORMANT_KNEE_ST).max(0.0) * semis.signum()
}

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
/// ## ⛔⛔ 判负 —— **它没用,默认永远是 false**(S159zj 实测,别再试一遍)
///
/// 台子:`inverse_probe`,**同一段 donor 音频**进两条臂,唯一的差别就是这个旋钮
/// (⛔ 不用「渲两遍整曲」—— 那被 donor 路径的跨进程不可复现性淹没:实测 85 % 样本不同,
/// 长时平均谱六档均匀 −0.45 dB = **增益差**不是频谱效应)。素材 = 鹅妈妈 +7 × 东雪莲的
/// `donor_pre_{-2,-7,-14}` 转储。尺子 = 「单样本一阶差 / 局部 ±3 ms RMS」,岛边 vs 岛内。
///
/// | 逆变换 | 改动样本 | 岛边−岛内(关)| (开)| **改善** |
/// |---|---|---|---|---|
/// | +2 | 1.16 % | +1.12 | +1.14 | **+0.02** |
/// | +7 | 0.64 % | +0.84 | +0.83 | **−0.01** |
/// | +14 | 0.46 % | +1.95 | +1.90 | **−0.04** |
///
/// ⇒ 这一刀**确实在做它说的事**(改动量正好是两端爬坡的规模),而**读数一动不动**。
///
/// ⭐⭐⭐ 机理,而且**代码里早就写着**:`carry` 是**未移调**的 donor,它不是 `acc`
/// (已移调输出)的延续。掺一部分进来只让**公式**连续,**波形并不会更平滑** ——
/// `psola.rs` 合成分支旁边那条「that is beating, not repair」警告的正是这件事,
/// 只不过它警告的是岛**内**,而岛**边**同样适用。
/// ⚠ 合成夹具上也早有征兆:`pulses` 脉冲串里补进去反而让局部一阶差**变大**
/// (0.107 vs 0.094),换成平滑浊音夹具才读到变小 —— 那个「换个夹具就翻号」本身
/// 就是「效应量比材料的自然跳变还小」的指纹,我当时没读出来。
///
/// ## ⭐ 那么真正的修法是什么(登记,未做)
///
/// 缺陷本身是真的(上面那张台阶表)。**别再往里灌未移调音频** —— 要让窗和在岛边
/// 真的爬到 1,得让**颗粒**在那里overlap起来,也就是把合成标记序列**往岛外延一点**
/// (`analysis_marks` 遇 `f0 ≤ 0` 立即 `break`,所以岛外长不出标记)。那是另一条改动,
/// 代价与风险都比这一刀大,而且它会碰 `voiced_islands` 的口径 —— 单独立项。
/// ⚠ 另一个方向是把 `WIN_PERIODS_DEFAULT` 退回 0(台阶解析地消失),但 S156 那次
/// 翻默认是**用户听完整曲五条臂拍板的**,退回去要重新过耳朵,而且会把高次共振峰那笔
/// 收益一起还回去。**别顺手退。**
///
/// ⛔ 旋钮留着是为了让「这条路走过、判负了」在仓里可复现,不是留给下一次翻默认。
/// 真要翻仍然要成对 bump [`RANGE_ALGO_VERSION`] 与 `audition_cache_tag`。
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

/// ⚙ 出厂默认 = 120.0 —— 桥接清音 120 ms(S160j;30 是 S154 的历史值)
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
/// ⭐⭐ **S160j —— 30 → 120,用户 2026-08-24 耳判后拍板。**
///
/// ## 它治的是什么
/// 用户在东雪莲 × 炉心融解 +7 上点名 **0:46.405** 一声「非常明显」的咔哒。逐处量出来:
/// 那是一段 **180 ms 的乐句内休止**(音[194]),两侧同一个 −9 的窗;休止里冒出一小段
/// **624/667/711 Hz = MIDI 75-77** 的有调突发,而该窗 donor 的音高正是 **MIDI 78 = 740 Hz**
/// ⇒ **donor【未移调】的尾音在 PSOLA 浊音岛之外漏了出来**(30 ms 的桥接够不着 180 ms 的休止)。
/// ⛔ 这就是为什么它「离辅音很近但不是那个辅音」,也是为什么 **LPC 残差那把咔哒尺子读不出来**
///    (×1.5,全曲最高 11%)—— 它找的是宽带瞬态,而这是一个**音高错了的有调片段**。
///
/// ## 读数(东雪莲 × 炉心融解 +7,同一二进制)
/// | 臂 | 46.39-46.44 的 donor 档(600-780 Hz) | ✅ な·80 稳态 | ✅ ま·87 稳态 |
/// |---|---|---|---|
/// | bridge 30 | **+29.3 dB** | 48.3 | 51.7 |
/// | **bridge 120** | **−9.4 dB(−38.7)** | 48.9 | 52.4 |
/// | bridge 250 | −12.3 | — | — |
/// ⇒ 用户耳判 120 与 250 **听不出区别** ⇒ 按风险取小的那个(与 S154 当年在 30/60 之间的取法同源)。
///
/// ## ⛔ 快音谱上的阴性对照(这是翻它之前最该问的一件事)
/// **鹅妈妈 × 东雪莲 × +7**(1215 个唱音,**73% ≤180 ms**,中位 140 ms)—— 用户原话
/// 「那玩意快音一堆,如果不被搅成一坨其实就可以」。按休止时长分箱量**休止里的能量**:
/// | 休止时长 | 条数 | 窗内 30 → 120 | ✅ 窗外(结构上不该被碰) |
/// |---|---|---|---|
/// | 60-90 ms | 5 | −15.7 → −15.9(**−0.16**) | −0.21 |
/// | 90-120 ms | 9 | −18.6 → −18.8(**−0.18**) | −0.16 |
/// | 120-200 ms | 105 | −14.9 → −15.0(−0.08) | −0.28 |
/// | 200-400 ms | 53 | −13.2 → −13.2(+0.01) | −0.31 |
/// ⇒ **窗内的变化全部落在「窗外那一列」= 这条链的复现噪声之内** ⇒ 快音没有被填、没有被搅在一起。
/// 用户听完:「鹅妈妈 120 我听了,我觉得也没问题」。
///
/// ## ⚠ 它够不着的那一半
/// 同一场里用户点的**另一声**咔哒(0:47.229)桥接**一分没动**(28.8 / 29.1 / 29.1)——
/// 那一族的成分**在 donor 进 PSOLA 之前就已经在了**(见 `score2svc::gate_unvoiced_tone`)。
/// ⇒ **别把这一刀当成「咔哒都解决了」。**
///
/// S154 原文(30 ms 那一版的出处)保留在下面。
const BRIDGE_UNVOICED_MS_DEFAULT: f64 = 120.0;

/// ⚙ S163 §40 —— `UTAI_PSOLA_BRIDGE_VALLEY=1` 让桥接的膨胀**停在能量谷**，
/// 而不是停在固定的 [`BRIDGE_UNVOICED_MS_DEFAULT`]。
///
/// ## 缺陷（同一次 run 的四层分解，零渲染噪声；鹅妈妈 × yachiyo）
///
/// 尺子 = **0-1k 竖线**（逐 2 ms 比前后 12 ms 中位高 >8 dB）:
/// 原 key **4** 条 / `base` 28 / 成品 **54**；而 0-1.5k 只多 15、0-3k 多 10、
/// **2-16k −2** ⇒ 缺陷精确落在 0-1k。⭐ 这是一族**全新的**东西 ——
/// S162 那次三层分解量的是 2-16 kHz 宽带竖线（结论：解码占 65-81%）。
///
/// | 层 | 条数 | 增量 |
/// |---|---|---|
/// | `base` | 14 | — |
/// | `donor_pre` | 11 | **解码 −3** |
/// | `donor_post` | 15 | **PSOLA +4**（同一份缓冲进出 ⇒ 铁的）|
/// | 成品 | 20 | 拼接 +5 |
///
/// PSOLA 造的那 12 条，距 **膨胀后岛边界** p50 **45 ms**，随机对照 p50 158 ms
/// ⇒ **富集 3.5×**；`<60 ms` 占 **75%**（随机 22%）；**9/12 条该处 f0 = 0**（清音段内）。
///
/// ## 机理
///
/// 桥接把每段浊音**向外膨胀 120 ms** 去盖住音头（治 S154 那条「音头留一截未移调、
/// 低 9-14 个半音的碎片」），但宽度是**固定**的 ⇒ 岛的新边界撂在清辅音的任意位置，
/// 往往正砸在爆破上。而 [`utai_dsp::psola::bridge_unvoiced`] 的 doc 写明那道边界只有
/// **0.25-3 ms 宽**、两侧差 5-17 个半音 ⇒ 宽带瞬变，在清音里就是 0-1k 的竖线。
/// ⇒ **桥接没有消除 S154 那道缝，只是把它从元音起点搬进了清辅音内部。**
///
/// ## 这一刀做什么
///
/// 膨胀**停在能量谷**。清辅音的能量剖面是「闭塞期(低) → 爆破(高) → 送气 → 元音(高)」，
/// 谷在爆破**之前** ⇒ 音头（爆破+元音）照样落在岛内（S160j 那条用户耳判拍板的效果不丢），
/// 而接缝落到了能量最低处。
/// ⛔ **只收窄不放宽**：谷只在 `[.., ext]` 之内找 ⇒ 覆盖范围永远 ⊆ 固定宽度那一版。
/// ⛔ 别改 `BRIDGE_UNVOICED_MS_DEFAULT` 本身 —— 120 是 S160j 用户耳判拍板的。
/// ⛔ 别全曲高通：用户 2026-08-27 已否（「硬造低通影响音色不可取」），
/// 而且上面那张表证明**解码层 −3**（模型侧没造）⇒ 高通会误伤模型唱对的低频。
///
/// 用户 2026-08-27 的两条原话（这一刀就是照着它们做的）:
/// 「我觉得**中+低频部分的那玩意绝对是其中一个症状之一**」
/// 「就算要硬救也得**先看看到底背后是什么造成的**吧；要不然直接开高通会误伤一些东西」
const BRIDGE_VALLEY_DEFAULT: bool = false;

/// ⚙ 出厂默认 = false —— `UTAI_PSOLA_BRIDGE_VALLEY=1` 打开。
/// 机理、四层分解与阴性对照全在 [`BRIDGE_VALLEY_DEFAULT`] 的 doc 上。
pub fn bridge_valley() -> bool {
    parse_bridge_valley(std::env::var("UTAI_PSOLA_BRIDGE_VALLEY").ok().as_deref())
}

fn parse_bridge_valley(v: Option<&str>) -> bool {
    match v.map(str::trim) {
        Some("1") => true,
        Some("0") => false,
        _ => BRIDGE_VALLEY_DEFAULT,
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
    tilt: f64,
) -> Result<Vec<f32>, String> {
    apply_inverse_with(inverse_engine(), audio, sample_rate, shift, kappa, fed_f0, tilt)
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
    tilt: f64,
) -> Result<Vec<f32>, String> {
    apply_inverse_windowed_with(
        inverse_engine(), audio, sample_rate, shift, kappa, fed_f0, keep, tilt,
    )
}

/// [`apply_inverse`] with the engine named explicitly — the A/B arm and the tests take this door.
/// (Selecting the engine through the process environment inside a test would race the other
/// tests in the same binary.)
#[allow(clippy::too_many_arguments)]
pub fn apply_inverse_with(
    engine: InverseEngine,
    audio: Vec<f32>,
    sample_rate: u32,
    shift: i64,
    kappa: f32,
    fed_f0: Option<(&[f32], usize)>,
    tilt: f64,
) -> Result<Vec<f32>, String> {
    apply_inverse_windowed_with(engine, audio, sample_rate, shift, kappa, fed_f0, &[], tilt)
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
    tilt: f64,
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
                formant_shift_semitones(semis, k, formant_knee()),
                f0,
                hop,
                frac,
                wsola,
                lock,
                hp,
                envfix,
                bridge,
                // S163 §40 —— 膨胀停在能量谷。见 `BRIDGE_VALLEY_DEFAULT`。
                bridge_valley(),
                win,
                xg,
                lpc,
                keep,
                fill,
                dejitter(),
                // S162 —— 谱倾斜还原。⛔ **显式参数,不是全局读取**:
                // 这条引擎是**两条车道共用**的,而表是在**谱面轨**素材上拟的、
                // cover 上一个读数都没有(而 cover 的深救援反而更重:|s|≥8 占救援总时长
                // 78.1%,最深 −18 已超出表的范围)⇒ 谱面轨传 `range_tilt()`,cover 传 0。
                tilt,
                // S162 —— 结尾裸透传的淡化(出厂开)。见 `tail_fade`。
                tail_fade(),
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

    /// ⛔⛔ S163 v2 —— 钉住**谐波收益闸**：坑找到了，但填进去的 `base` 在上方谐波上
    /// 不比 donor 好时，**必须不填**。
    ///
    /// 用户 2026-08-27：「你倒是基频连上了 谐波你是一点也不管啊？」
    /// v1 只按全带 RMS 找坑就填 ⇒ 收益主要在 H1（akiko H1 +12.1 而 H3-H8 只有 +3.6），
    /// yachiyo 上全线 −1.5 ⇒ 净负。
    ///
    /// 夹具：同一个坑，两种 `base`
    /// * `rich` —— 上方谐波比 donor 强 ⇒ 必须填；
    /// * `dull` —— 只有基频、上方谐波比 donor 弱 ⇒ **必须不填**（这正是「基频接上了谐波没补」）。
    #[test]
    /// ⛔⛔⛔ S163 v5 —— **电平闸的尺子必须是「逐 10 ms 格的最大差」,不是区间平均**。
    ///
    /// v4 用 rms(区间平均)实测,三处点名坐标的读数:
    /// ```text
    ///                     rms 尺子      峰值尺子
    /// ぴゃ      (该保住)    −3.8 dB       **+11.6 dB**
    /// 2:07.987(该拦住)   −17.3 dB       −15.4 dB
    /// 4:10.187(该拦住)    −7.3 dB        −4.8 dB
    /// ```
    /// **rms 尺子结构上分不开这三个** —— `ぴゃ` 夹在两处「该拦住」的读数中间。
    ///
    /// 这里的夹具就是 `ぴゃ` 的形状:坑区间里大部分格 `base` **略低**,
    /// 只有**最深那一格** `base` 明显更高。区间平均会把它拦掉,逐格峰值会放行。
    fn dipfill_level_gate_measures_the_deepest_cell_not_the_average() {
        let sr = 44100u32;
        let hop = sr as usize / 100;
        let harm = |i: usize, amp: f32| -> f32 {
            let w = 2.0 * std::f32::consts::PI * 300.0 * i as f32 / sr as f32;
            amp * (w.sin() + 0.5 * (2.0 * w).sin() + 0.5 * (3.0 * w).sin()
                + 0.33 * (4.0 * w).sin())
        };
        // donor:40 格常规,坑 = 第 18-23 格。坑里前后几格只低一点,**中间一格塌到底**。
        let mut donor: Vec<f32> = (0..40 * hop)
            .map(|i| 0.30 * (2.0 * std::f32::consts::PI * 300.0 * i as f32 / sr as f32).sin())
            .collect();
        for c in 18..24 {
            let g = if c == 20 { 0.01 } else { 0.20 }; // 中间一格 −40 dB,其余 −14 dB
            for v in donor.iter_mut().take((c + 1) * hop).skip(c * hop) {
                *v *= g;
            }
        }
        // base:坑区间里比 donor 的「其余格」**略低**(0.045 < 0.30×0.20 = 0.06),
        //      但比塌到底那一格(0.30×0.01 = 0.003)**高得多**。
        let base_here: Vec<f32> = (0..donor.len())
            .map(|i| {
                let c = i / hop;
                harm(i, if (18..24).contains(&c) { 0.045 } else { 0.30 })
            })
            .collect();

        let got = dipfill_spans(&donor, sr, 12.0, &base_here, 300.0);
        assert_eq!(
            got.len(),
            1,
            "最深一格 base 高 ~13 dB 却没填 —— 闸退回区间平均了(这正是 `ぴゃ` 丢掉改善的那一族)"
        );

        // ⛔ 阴性对照:把塌到底那一格也抬到和其余格一样 ⇒ 最深格不再有增益 ⇒ 不许填
        let mut flat = donor.clone();
        for v in flat.iter_mut().take(21 * hop).skip(20 * hop) {
            *v *= 20.0; // 0.01 → 0.20,与坑里其余格齐平
        }
        let none = dipfill_spans(&flat, sr, 12.0, &base_here, 300.0);
        assert!(
            none.is_empty(),
            "最深格没有增益时不许填(填了 {} 段)—— 闸必须真的在看最深那一格",
            none.len()
        );
    }

    #[test]
    /// ⛔⛔⛔ S163 v4 —— **`base` 比 `donor` 低的坑一个都不许填**。
    ///
    /// 这一刀的语义前提是「donor 这儿没声了,用 base 顶上」。
    /// v3 没问过这个前提成不成立,结果在两处**救援本来正常工作**的地方
    /// (成品比 base 高 5~6.4 dB)填了 base,把电平削回 base ⇒ 退化。
    ///
    /// 夹具:同一条 donor(带一个深坑),只换 `base` 的电平 ——
    /// base 响 ⇒ 填;base 轻 ⇒ **一个都不填**。
    fn dipfill_refuses_to_fill_from_a_base_that_is_quieter_than_the_donor() {
        let sr = 44100u32;
        let hop = sr as usize / 100;
        // donor:40 格 300 Hz 正弦 + 中间 3 格深坑
        let mut donor: Vec<f32> = (0..40 * hop)
            .map(|i| 0.30 * (2.0 * std::f32::consts::PI * 300.0 * i as f32 / sr as f32).sin())
            .collect();
        // ⛔ 坑**不许挖成精确 0** —— 真实的坑只是比窗内中位低 12-40 dB,
        //    而 `base` 在那里可能仍然更低(akiko 2:07.987:donor 比 base 高 6.4 dB)。
        //    挖成 0 的话任何 base 都比它响 ⇒ 电平闸恒过 ⇒ 这条判据就是空的。
        for v in donor.iter_mut().take(21 * hop).skip(18 * hop) {
            *v *= 0.2; // −14 dB:过 12 dB 的坑门槛,但仍有电平可比
        }
        // base:带上方谐波(谐波闸必须能过),电平可调
        let mk_base = |amp: f32| -> Vec<f32> {
            (0..donor.len())
                .map(|i| {
                    let w = 2.0 * std::f32::consts::PI * 300.0 * i as f32 / sr as f32;
                    amp * (w.sin() + 0.5 * (2.0 * w).sin() + 0.5 * (3.0 * w).sin()
                        + 0.33 * (4.0 * w).sin())
                })
                .collect()
        };
        // ⑴ base 明显比 donor 响 ⇒ 该填
        let loud = dipfill_spans(&donor, sr, 12.0, &mk_base(0.60), 300.0);
        assert_eq!(loud.len(), 1, "base 比 donor 响得多时那个坑必须被填上");

        // ⑵ base 明显比 donor 轻 ⇒ **一个都不许填**(填了就是把救援削掉)
        let quiet = dipfill_spans(&donor, sr, 12.0, &mk_base(0.02), 300.0);
        assert!(
            quiet.is_empty(),
            "base 比 donor 轻的时候填了 {} 段 —— 这正是 v3 削掉救援的那一族",
            quiet.len()
        );

        // ⑶ 阴性对照:坑处两边电平相当(在 3 dB 余量之内)⇒ 也不填,不许在边界上抖
        //    donor 坑处 amp = 0.30 × 0.2 = 0.06;base 取 rms 与它相当的电平。
        let tie = dipfill_spans(&donor, sr, 12.0, &mk_base(0.055), 300.0);
        assert!(
            tie.is_empty(),
            "坑处两边电平相当时不许填(填了 {} 段)—— 3 dB 余量就是为了不在边界上抖",
            tie.len()
        );
    }

    #[test]
    /// ⛔⛔⛔ S163 v3 —— **判据的地板只许由窗内材料决定**。
    ///
    /// `seg = donor[lo..hi]` 两端各带 `MERGE_BRIDGE_FRAMES`(25 帧 ≈ **500 ms**)窗外材料,
    /// 而写入只覆盖 `[a, b)`。v2 把整条 `seg` 交给 [`dipfill_spans`] ⇒ `med`(存活中位数)
    /// 被窗外电平污染。这一条**同一段窗内音频、只换窗外材料**,判定必须一个字不变。
    ///
    /// ⚠ 它守的是「判据静默失效」那一族(`width` 栽过一次,dipfill v2 又栽一次)。
    fn dipfill_floor_is_decided_by_the_written_span_only() {
        let sr = 44100u32;
        let hop = sr as usize / 100; // 10 ms
        let tone = |n: usize, amp: f32| -> Vec<f32> {
            (0..n)
                .map(|i| amp * (2.0 * std::f32::consts::PI * 300.0 * i as f32 / sr as f32).sin())
                .collect()
        };
        // 窗内:40 格常规电平 + 中间 3 格深坑
        let mut inside = tone(40 * hop, 0.30);
        for i in (18 * hop)..(21 * hop) {
            inside[i] = 0.0;
        }
        let verdict = |outside_amp: f32| -> usize {
            // 两端各 50 格窗外材料 —— 只有电平不同
            let mut seg = tone(50 * hop, outside_amp);
            seg.extend_from_slice(&inside);
            seg.extend(tone(50 * hop, outside_amp));
            let win = &seg[50 * hop..90 * hop];
            // ⛔ `base` 必须**真的有上方谐波** —— 谐波收益闸比的就是这个。
            //    两条都是纯正弦的话闸恒不过,这条判据就只验到「不填」(= 半条空判据)。
            let base_here: Vec<f32> = (0..win.len())
                .map(|i| {
                    let t = i as f32 / sr as f32;
                    let w = 2.0 * std::f32::consts::PI * 300.0 * t;
                    0.30 * w.sin()
                        + 0.15 * (2.0 * w).sin()
                        + 0.15 * (3.0 * w).sin()
                        + 0.10 * (4.0 * w).sin()
                })
                .collect();
            dipfill_spans(win, sr, 12.0, &base_here, 300.0).len()
        };
        // 窗外很响(+20 dB)/ 很静(−20 dB)/ 与窗内同电平 —— 三种都必须给同一个判定
        let (loud, quiet, same) = (verdict(3.0), verdict(0.03), verdict(0.30));
        assert_eq!(
            (loud, quiet), (same, same),
            "窗外材料改变了窗内的判定 (响={loud} 静={quiet} 同={same}) —— 地板被污染了"
        );
        assert_eq!(same, 1, "窗内那个 30 ms 深坑必须被认出来");
    }

    #[test]
    /// ⛔⛔⛔ S163 v17a —— **起音整形只许压，不许抬**。
    ///
    /// v17 第一版把增益 clamp 在 ±9 dB,而 `donor` 本来就贴着满刻度:
    /// ```text
    /// splice 后峰值 **+8.83 dBFS**、**39193 个样本 |x|≥0.999**
    /// ⇒ 后续归一化为了容纳那个峰,把**整条音频压了 7 dB**
    /// ⇒ 16-24 kHz 因削波失真 **+21.5 dB**
    /// ```
    /// 这条判据把「输出峰值不许超过输入峰值」钉死 —— 它是**结构性**的,
    /// 与具体音频无关:整形只改增益且增益 ≤ 1 ⇒ 峰值只可能变小。
    fn onset_fit_never_amplifies() {
        let sr = 48000u32;
        let n = (sr as usize) / 2;
        // 一条贴着满刻度的信号 + 一个陡起音
        let mut x: Vec<f32> = (0..n)
            .map(|i| 0.98 * (2.0 * std::f32::consts::PI * 300.0 * i as f32 / sr as f32).sin())
            .collect();
        for (i, v) in x.iter_mut().enumerate().take(sr as usize / 100) {
            *v *= (i as f32 / (sr as f32 / 100.0)).min(1.0).powf(0.1); // 极陡起音
        }
        let peak_in = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        // `base` 起音慢得多 ⇒ 整形会想把 donor 的早期压下去(而不是把晚期抬起来)
        let base: Vec<f32> = (0..n)
            .map(|i| {
                let r = (i as f32 / (sr as f32 * 0.05)).min(1.0);
                0.5 * r * (2.0 * std::f32::consts::PI * 300.0 * i as f32 / sr as f32).sin()
            })
            .collect();
        let notes = vec![NoteSpan { start: 0, frames: 25, sung: true, hz: 300.0, tied: false }];
        let jobs = vec![DeadJob { shift: -2, start: 0, end: 25 }];
        let kept = vec![(0i64, 0usize, x.clone())];
        let mut out = base.clone();
        splice_kept(
            &mut out, sr, (sr as f64) / 50.0, &jobs, &kept,
            (sr as usize) / 100, false, 0, 0.0, &notes, 0, 0.0, true, 0, true, false,
        )
        .unwrap();
        let peak_out = out.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(
            peak_out <= peak_in + 1e-6,
            "起音整形把峰值从 {peak_in:.4} 抬到了 {peak_out:.4} —— 只压不抬这条被破坏了"
        );
        assert!(
            out.iter().all(|v| v.abs() <= 1.0 + 1e-6),
            "整形把样本推出了满刻度"
        );
    }

    #[test]
    fn dipfill_refuses_to_fill_when_base_has_no_upper_harmonics() {
        let sr = 44_100u32;
        let f0 = 300.0f32;
        let h = sr as usize / 100; // 10 ms
        let n = 40 * h;
        let mk = |amp1: f64, amp_up: f64, pit: Option<(usize, usize)>| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let t = i as f64 / f64::from(sr);
                    let p = 2.0 * std::f64::consts::PI * f64::from(f0) * t;
                    let g = match pit {
                        Some((a, b)) if i >= a && i < b => 0.01,
                        _ => 1.0,
                    };
                    (g * (amp1 * p.sin()
                        + amp_up * (3.0 * p).sin()
                        + amp_up * 0.8 * (4.0 * p).sin()
                        + amp_up * 0.6 * (5.0 * p).sin())) as f32
                })
                .collect()
        };
        let pit = (20 * h, 23 * h);
        let donor = mk(0.30, 0.10, Some(pit));      // 坑在 20-23 格
        let rich = mk(0.30, 0.25, None);            // base：上方谐波更强
        let dull = mk(0.30, 0.01, None);            // base：几乎只有基频

        let with_rich = dipfill_spans(&donor, sr, 12.0, &rich, f0);
        assert_eq!(
            with_rich.len(),
            1,
            "base 上方谐波更强时必须填 —— 实测 {with_rich:?}"
        );

        let with_dull = dipfill_spans(&donor, sr, 12.0, &dull, f0);
        assert!(
            with_dull.is_empty(),
            "⛔ base 只有基频、上方谐波更弱时必须【不填】—— 实测 {with_dull:?}\
             （这正是「基频连上了 谐波一点没补」那个坑）"
        );
        // ⛔ v2 语义：拿不到 `base` ⇒ **保守不填**（没法验证谐波收益就不做）
        let no_base = dipfill_spans(&donor, sr, 12.0, &[], 0.0);
        assert!(no_base.is_empty(), "拿不到 base 时必须保守不填");

        // ⛔ 拿不到 f0 ⇒ 同样保守不填
        let unknown_f0 = dipfill_spans(&donor, sr, 12.0, &dull, 0.0);
        assert!(unknown_f0.is_empty(), "f0 未知时必须保守不填");
    }


    /// ⛔⛔ S163 §34 —— 钉住 [`dipfill_spans`] 的三条行为。
    ///
    /// 夹具照抄实测形状：donor 在窗内的中位是 `−10 dBFS` 级，
    /// 而洞处掉到 `−45` 级（实测 p50 相对窗内中位 **−34.9…−39.2 dB**，
    /// 而 `base` 同处只有 **−8.8…−13.4** ⇒ 门限 20 把两者分开）。
    ///
    /// ⚠ 变异检查（都试过，会红）：
    /// * 门限写成 10 ⇒ ⑶ 失守（`base` 那种 −12 dB 的自然衰减也会被当成坑）；
    /// * 去掉宽度上限 ⇒ ⑵ 失守（整段没救到的会被整段填成 base = 把救援退掉）；
    /// * 中位用**全部**格算而不是只用有声格 ⇒ 参照被静音拖到地板，一个坑都找不到。
    #[test]
    fn dipfill_only_catches_deep_narrow_silent_pits() {
        let sr = 44_100u32;
        let h = sr as usize / 100; // 10 ms
        let mk = |lv: &[f32]| -> Vec<f32> {
            let mut v = Vec::with_capacity(lv.len() * h);
            for (i, &a) in lv.iter().enumerate() {
                for k in 0..h {
                    let t = (i * h + k) as f64 / f64::from(sr);
                    v.push((f64::from(a) * (2.0 * std::f64::consts::PI * 300.0 * t).sin()) as f32);
                }
            }
            v
        };
        // ⑴ 深而窄的坑（30 ms，−40 dB 级）⇒ 必须找到
        let mut lv = vec![0.3f32; 40];
        lv[20] = 0.003;
        lv[21] = 0.003;
        lv[22] = 0.003;
        let pit = mk(&lv);
        // ⛔ v2：拿不到 base/f0 ⇒ 保守不填 ⇒ 必须把 base 与 f0 给全才会命中
        let flat_base = vec![0.3f32; pit.len()];
        let got = dipfill_spans(&pit, sr, 20.0, &flat_base, 300.0);
        assert_eq!(got.len(), 1, "深而窄的坑必须找到一个 —— 实测 {got:?}");
        assert!(
            got[0].0 <= 20 * h && got[0].1 >= 23 * h - h,
            "坑的位置要对上 —— 实测 {:?}，期望覆盖 {}..{}",
            got[0],
            20 * h,
            23 * h
        );

        // ⑵ 同样深但**太宽**（150 ms）⇒ 不许碰（那是「整段没救到」）
        let mut wide = vec![0.3f32; 40];
        for x in wide.iter_mut().skip(10).take(15) {
            *x = 0.003;
        }
        assert!(
            dipfill_spans(&mk(&wide), sr, 20.0, &vec![0.3f32; mk(&wide).len()], 300.0).is_empty(),
            "太宽的低段不是坑，不许回填"
        );

        // ⑶ base 那种自然衰减（−12 dB 级）⇒ 门限 20 必须放过
        let mut soft = vec![0.3f32; 40];
        for x in soft.iter_mut().skip(20).take(3) {
            *x = 0.075; // ≈ −12 dB
        }
        assert!(
            dipfill_spans(&mk(&soft), sr, 20.0, &vec![0.3f32; mk(&soft).len()], 300.0).is_empty(),
            "−12 dB 的自然衰减不是坑 —— 实测 {:?}",
            dipfill_spans(&mk(&soft), sr, 20.0, &vec![0.3f32; mk(&soft).len()], 300.0)
        );

        // ⑷ 出厂关（depth = 0）⇒ 永远返回空 ⇒ 逐位不变
        assert!(dipfill_spans(&pit, sr, 0.0, &[], 0.0).is_empty(), "depth=0 必须逐位不变");
    }


    /// ⛔⛔ S163 —— 钉住 [`landing_usag_eps`] 换掉排序主键的**三种**行为。
    ///
    /// 夹具照抄用户点名的那一组的真实形状(akiko 4:07,日志原文
    /// `landing candidates [(-2, Some(4.482457), 0.0), (-4, Some(5.9618196), 0.0)]`):
    /// **`rel` 偏好 −2,而 −2 的上方谐波音内塌陷是全部候选里最差的。**
    ///
    /// ⚠ 变异检查(都试过,会红):
    /// * 把比较写成 `ux.partial_cmp(&uy)`(方向反) ⇒ ⑵ 失守;
    /// * 把 `> usag_eps` 写成 `>= 0.0`(等于取消 eps) ⇒ ⑶ 失守,对照组被动;
    /// * 少填一处 `sc.usag.push` ⇒ `worst_usag()` 恒 0 ⇒ ⑵ 失守。
    /// ⭐⭐⭐⭐ S165 —— **失配轴**:出厂不变 · 相对判据 · **最差口径**(不是平均)· 对手轴闸 · 阴性对照。
    ///
    /// 夹具照**实测坐标**造(yachiyo × 炉心 4:36 那个音,用户听过三个候选):
    /// 逐根谐波「抖动 − 该电平该有的抖动」的最大值 =
    /// **−8 → +1.17(用户判最差)· −13 → +0.36 · −15 → −0.43(用户:最像它自己)**。
    #[test]
    fn the_mismatch_axis_uses_the_worst_note_and_only_moves_when_a_better_candidate_exists() {
        // 每个候选给两个音:音 0 是「组里本来就一般」的,音 1 是真正分胜负的那个。
        let mk = |shift: i64, rel: f32, m0: f32, m1: f32, uplev: f32| {
            let mut sc = CandScore::default();
            sc.rel.push((0, rel));
            sc.mism.push((0, m0));
            sc.mism.push((1, m1));
            sc.uplev.push((0, uplev));
            sc.uplev.push((1, uplev));
            (0usize, shift, 0usize, Vec::<f32>::new(), sc)
        };
        // 真实形状:rel 偏好今天的 −8,而失配说 −15 好 1.60。
        let cand = vec![mk(-8, 4.2, -0.3, 1.17, 0.0), mk(-15, 5.9, -0.5, -0.43, 0.0)];

        // ⑴ ⛔ eps = 0 ⇒ **逐位回到今天**(出厂关 = 不改音频的保证)
        let today = decide_group(&cand, 0, -8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(cand[today[0]].1, -8, "eps=0 必须逐位回到今天的 rel 排序");

        // ⑵ 改善 1.60 > eps 0.5 ⇒ 按失配选 −15,哪怕 rel 指向 −8
        let fixed = decide_group(&cand, 0, -8, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.5);
        assert_eq!(
            cand[fixed[0]].1, -15,
            "失配能改善 1.60 dB 时必须压过 rel —— 用户耳判 −15 最像它自己"
        );

        // ⑶ ⛔⛔ **最差口径**:−8 在音 0 上更好(−0.3 vs −0.5),但音 1 上差 1.60。
        //    若哪天改成「平均」,平均差只有 (0.2+1.60)/2 ≈ 0.9,而且**方向会被音 0 拉回去** ——
        //    实测三种「整体距离」口径全都选中了用户判为最差的那一档。这一条钉住它。
        let avg_trap = vec![
            mk(-8, 4.2, -2.0, 1.17, 0.0),  // 音 0 极好,音 1 极差
            mk(-15, 5.9, 0.0, -0.43, 0.0), // 音 0 一般,音 1 好
        ];
        let picked = decide_group(&avg_trap, 0, -8, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.5);
        assert_eq!(
            avg_trap[picked[0]].1, -15,
            "⛔ 一个音失配 +1.17 就该换,不许被组里另一个音的 −2.0 平均掉"
        );

        // ⑷ ⛔ 相对判据:两个候选都不好但**差不多** ⇒ 不动(避免绝对门限的误伤)。
        //    实测绝对门限会否掉 41% 的候选而只有 10 组能真正改善。
        let both_bad = vec![mk(-8, 4.2, 0.9, 1.10, 0.0), mk(-15, 5.9, 1.0, 1.05, 0.0)];
        let kept = decide_group(&both_bad, 0, -8, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.5);
        assert_eq!(
            both_bad[kept[0]].1, -8,
            "两个候选都失配但差 <eps ⇒ 必须回落 rel,不许瞎换"
        );

        // ⑸ ⛔⛔ **对手轴闸**(用户上线警告:别把哑音引回来)。
        //    赢家把上方谐波压掉超过 `usag_dim_cap` ⇒ 不许换。
        let costly = vec![mk(-8, 4.2, -0.3, 1.17, 6.0), mk(-15, 5.9, -0.5, -0.43, 0.0)];
        let held = decide_group(&costly, 0, -8, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.5);
        assert_eq!(
            costly[held[0]].1, -8,
            "赢家把上方谐波压了 6 dB(> usag_dim_cap 3)⇒ 必须拦住 —— 这是 h2 那次哑音的教训"
        );

        // ⑹ ⛔ 阴性对照:**两边都量不到失配**(参照建不起来 / 峰位验证全失败)⇒ 这根轴必须闭嘴。
        let blind = vec![
            (0usize, -8i64, 0usize, Vec::<f32>::new(), CandScore::default()),
            (0usize, -15i64, 0usize, Vec::<f32>::new(), CandScore::default()),
        ];
        let n = decide_group(&blind, 0, -8, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.5);
        assert_eq!(blind[n[0]].1, -8, "量不到失配时这根轴必须闭嘴(best_mism_vs 给 −inf)");

        // ⑻ ⛔⛔ **失配触发 + 宽半径**:没有它,失配轴结构上无候选可选。
        //    实测(`eps=0.05` 几乎不设限):失配轴每组只被问 **2-6 次**,落点 **0 变化** ——
        //    因为 `decide_group` 只看得到修补遍渲出来的候选,而默认 `REPAIR_RADIUS = 2`
        //    从 −1/−4/−9/−11 只够到相邻档,离线转储里那些更好的候选根本没渲出来。
        //    这条钉住三件事:门限的方向(失配**大**才触发)、宽半径比默认大、两个轴同时触发时取宽的。
        assert!(
            MISM_REPAIR_FLOOR > 0.0,
            "失配是「越大越差」⇒ 触发门限必须是正的(写成负数会变成永远触发)"
        );
        assert!(
            MISM_REPAIR_RADIUS > REPAIR_RADIUS,
            "宽半径 {MISM_REPAIR_RADIUS} 必须大于默认 {REPAIR_RADIUS},否则这条通路等于没有"
        );
        // 门限要落在实测分布内:两首谱上 worst mismatch 的范围是 −0.16 … +4.59。
        assert!(
            (0.5..=3.0).contains(&MISM_REPAIR_FLOOR),
            "门限 {MISM_REPAIR_FLOOR} 落在实测分布(−0.16…+4.59)之外 —— 要么永不触发,要么全触发"
        );

        // ⑺ 参照曲线的内插:两端取端点,中间线性。
        let r = vec![(-45.0f32, 1.6f32), (-19.0, 1.44), (4.0, 0.9)];
        assert_eq!(mismatch_expect(&r, -60.0), Some(1.6), "低于最低档 ⇒ 取端点");
        assert_eq!(mismatch_expect(&r, 20.0), Some(0.9), "高于最高档 ⇒ 取端点");
        let mid = mismatch_expect(&r, -32.0).unwrap();
        assert!((mid - 1.52).abs() < 0.02, "中间该线性内插,实际 {mid:.3}");
        assert_eq!(mismatch_expect(&[], 0.0), None, "参照空 ⇒ None(不许瞎猜)");
    }

    /// ⭐⭐⭐ S165 —— `2·f0` 电平这根轴:**出厂不变 · 真坐标能修好 · 对照不许动 · 对手轴闸拦得住**。
    ///
    /// 夹具全部照**实测坐标**造(yachiyo × 炉心融解 +7,`[794]あ`,f0 987.8 Hz):
    /// 今天落在 `−8` ⇒ `2·f0` = **−17.0 dB**;好格 `−13` ⇒ **−5.2 dB**
    /// (与用户说「好」的 yuyuko 在同一个音上**完全一致**)。
    /// 用户 2026-08-28 听过这两档的探针臂后判:「**f1『强』/『实』确实在听感上更好**」。
    #[test]
    fn the_second_harmonic_axis_fixes_the_hollowed_note_without_touching_the_healthy_ones() {
        let mk = |shift: i64, rel: f32, h2: f32, uplev: f32| {
            let mut sc = CandScore::default();
            sc.rel.push((0, rel));
            sc.h2.push((0, h2));
            sc.uplev.push((0, uplev));
            (0usize, shift, 0usize, Vec::<f32>::new(), sc)
        };
        // 真实形状:rel 偏好今天的 −8,而 h2 说 −13 好 11.8 dB。
        let cand = vec![mk(-8, 4.2, -17.0, 0.0), mk(-13, 5.9, -5.2, -3.0)];

        // ⑴ ⛔ eps = 0 ⇒ **逐位回到今天**(这是「出厂关 = 不改音频」的保证)
        let today = decide_group(&cand, 0, -8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(cand[today[0]].1, -8, "eps=0 必须逐位回到今天的 rel 排序");

        // ⑵ 差 11.8 dB > eps 6.0 ⇒ 按 h2 选 −13,哪怕 rel 指向 −8
        let fixed = decide_group(&cand, 0, -8, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 6.0, 0.0);
        assert_eq!(
            cand[fixed[0]].1, -13,
            "h2 差 11.8 dB 时必须压过 rel —— 用户耳判确认 −13 更好听"
        );

        // ⑶ ⛔ 对照:h2 只差 2.5 dB(< eps)⇒ 回落 rel ⇒ **一个都不许动**。
        //    实测两个模型的 h2 中位是 −4.8 / −6.5,大量音就在这个区间里,
        //    eps 若设小了会把它们全部翻一遍。
        let ctrl = vec![mk(-8, 4.2, -5.0, 0.0), mk(-13, 5.9, -7.5, -3.0)];
        let kept = decide_group(&ctrl, 0, -8, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 6.0, 0.0);
        assert_eq!(ctrl[kept[0]].1, -8, "h2 只差 2.5 dB 时必须回落 rel");

        // ⑷ ⛔⛔ **对手轴闸**:赢家把上方谐波压掉超过 `H2_DIM_CAP` ⇒ 不许换。
        //    S163 §32 的血训:`gone` 那一支当初没配对手轴闸,直接造出全曲唯一一个变闷 >2.7 的音
        //    ⇒ **每一根参与排序的轴都要配**。
        let costly = vec![mk(-8, 4.2, -17.0, 6.0), mk(-13, 5.9, -5.2, -8.0)];
        let held = decide_group(&costly, 0, -8, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 6.0, 0.0);
        assert_eq!(
            costly[held[0]].1, -8,
            "赢家把上方谐波压了 14 dB(> H2_DIM_CAP {H2_DIM_CAP})⇒ 必须拦住"
        );

        // ⑸ ⭐ `h2` 排在 `usag` **之前**:两根轴指向相反时,h2 说了算。
        //    (用户的理由:`2·f0` = 1975 Hz 在耳朵最敏感的区间,更高的谐波感知权重低得多。)
        let mut a = mk(-8, 4.2, -17.0, 0.0);
        let mut b = mk(-13, 5.9, -5.2, -3.0);
        a.4.usag.push((0, 1.0)); // 今天这档 usag 更好
        b.4.usag.push((0, -6.0)); // 好格 usag 更差,差 7 dB > usag_eps 3
        let both = vec![a, b];
        let win = decide_group(&both, 0, -8, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 6.0, 0.0);
        assert_eq!(
            both[win[0]].1, -13,
            "h2 与 usag 指向相反时 h2 必须赢 —— 它排在 usag 之前就是为了这个"
        );

        // ⑺ ⛔⛔ **真实形状:组里有一个音在【所有】候选上都很差**(气声/清音读不出干净的 f1)。
        //    实机上 47/54 次修补遍触发**一条落点都没改**,就死在这里:第一版排序用
        //    `worst_h2()`(组内最差),而那个音把整组钉死 ⇒ 候选之间差 0 ⇒ 这根轴永远不发言。
        //    ⇒ 必须逐音配对取最大增益(`best_h2_vs`)。
        let mk2 = |shift: i64, rel: f32, h2_bad: f32, h2_good: f32| {
            let mut sc = CandScore::default();
            sc.rel.push((0, rel));
            sc.h2.push((0, h2_bad)); // 音 0:气声,两个候选上都很差
            sc.h2.push((1, h2_good)); // 音 1:真正要救的那个
            sc.uplev.push((0, 0.0));
            sc.uplev.push((1, 0.0));
            (0usize, shift, 0usize, Vec::<f32>::new(), sc)
        };
        let pinned = vec![mk2(-8, 4.2, -30.0, -17.0), mk2(-13, 5.9, -30.0, -5.2)];
        let saved = decide_group(&pinned, 0, -8, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 6.0, 0.0);
        assert_eq!(
            pinned[saved[0]].1, -13,
            "组里有个音在所有候选上都读 −30(气声),但另一个音差 11.8 dB ——              这根轴必须逐音看,不许被那个音钉死(实机 47/54 次触发 0 改变就是这么来的)"
        );

        // ⑻ ⛔⛔ **实机第二个洞:两个候选量到 h2 的【音集合不相交】**。
        //    `best_h2_vs` 只对双方都量到的音求差,交集为空时 `fold(0.0, max)` 返回 **0.0**
        //    ⇒ 两边都是 0 ⇒ 差 0 < eps ⇒ 这根轴又不发言。
        //    实机日志(yuyuko,H1 臂)里就有这种组:
        //    `[(-6, h2=-21.4), …, (-2, h2=-7.4), …] ⇒ kept -6` —— 差 14 dB 却没换。
        //    ⛔ 峰位验证会按候选各自拒绝一部分音 ⇒ **集合不同是常态,不是边角情况**。
        let mk3 = |shift: i64, rel: f32, notes: &[(usize, f32)]| {
            let mut sc = CandScore::default();
            sc.rel.push((0, rel));
            for &(i, v) in notes {
                sc.h2.push((i, v));
                sc.uplev.push((i, 0.0));
            }
            (0usize, shift, 0usize, Vec::<f32>::new(), sc)
        };
        // 候选 A 只在音 0 上量到,候选 B 只在音 1 上量到 ⇒ 交集为空。
        let disjoint = vec![mk3(-6, 4.2, &[(0, -21.4)]), mk3(-2, 5.9, &[(1, -7.4)])];
        let picked = decide_group(&disjoint, 0, -6, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 6.0, 0.0);
        assert_eq!(
            disjoint[picked[0]].1, -6,
            "⛔ 交集为空时这根轴必须**闭嘴**(回落到今天的 rel),而不是拿 0.0 当两边都一样 ——              若哪天改成「交集为空也敢比」,这条会红"
        );
        // 而只要有**一个**共同的音,它就必须发言。
        let shared = vec![
            mk3(-6, 4.2, &[(0, -21.4), (1, -3.0)]),
            mk3(-2, 5.9, &[(0, -7.4), (2, -3.0)]),
        ];
        let fixed2 = decide_group(&shared, 0, -6, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 6.0, 0.0);
        assert_eq!(
            shared[fixed2[0]].1, -2,
            "音 0 上 −2 比 −6 好 14 dB(共同的音只有它一个)⇒ 必须换"
        );

        // ⑹ ⛔ 阴性对照:**量不到 h2 的候选**(峰位验证失败 ⇒ 空 Vec)不许被当成「很差」。
        //    气声/清音段落正是这种情况(yuyuko 0:56-1:01 的 f1 大多读不出来)。
        let blind = vec![
            (0usize, -8i64, 0usize, Vec::<f32>::new(), CandScore::default()),
            (0usize, -13i64, 0usize, Vec::<f32>::new(), CandScore::default()),
        ];
        let n = decide_group(&blind, 0, -8, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 6.0, 0.0);
        assert_eq!(
            blind[n[0]].1, -8,
            "两边都量不到 h2 ⇒ 这根轴必须闭嘴(worst_h2 都是 0.0,差 0 < eps)"
        );
    }

    #[test]
    fn usag_becomes_the_primary_sort_key_only_when_the_gap_is_real() {
        let mk = |ji: usize, shift: i64, rel: f32, usag: f32| {
            let mut sc = CandScore::default();
            sc.rel.push((0, rel));
            sc.usag.push((0, usag));
            (ji, shift, 0usize, Vec::<f32>::new(), sc)
        };
        // 4:07 那组的真实形状:rel 偏好 −2,而 usag 说 −2 最差。
        let cand = vec![mk(0, -2, 4.48, -10.8), mk(0, -4, 5.96, -3.2)];

        // ⑴ eps = 0 ⇒ 逐位回到今天(rel 最小的 −2 在前)
        let today = decide_group(&cand, 0, -2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(
            cand[today[0]].1, -2,
            "eps=0 必须逐位回到今天的 rel 排序 —— 这是「出厂不变」的保证"
        );

        // ⑵ 差 7.6 dB > eps 3.0 ⇒ 按 usag 选 −4，哪怕 rel 指向 −2
        let fixed = decide_group(&cand, 0, -2, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0);
        assert_eq!(
            cand[fixed[0]].1, -4,
            "usag 差 7.6 dB 时必须压过 rel —— 实测那一组 rel 的方向是反的"
        );

        // ⑶ 对照组的形状:usag 只差 2.0 dB(< eps) ⇒ 回落 rel ⇒ **不动**
        //    (用户点名的 0:40.901 与 0:43.641 两个「听起来正常」的音就是这个形状)
        let ctrl = vec![mk(0, -2, 4.90, 1.3), mk(0, -4, 5.80, 3.3)];
        let kept = decide_group(&ctrl, 0, -2, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0);
        assert_eq!(
            ctrl[kept[0]].1, -2,
            "usag 只差 2.0 dB 时必须回落 rel ⇒ 听起来正常的那些音一个都不许动"
        );

        // ⑷ 阴性对照:两个候选 usag 都量不到(短音) ⇒ worst_usag 都是 0 ⇒ 回落 rel
        let none = vec![
            (0usize, -2i64, 0usize, Vec::<f32>::new(), {
                let mut sc = CandScore::default();
                sc.rel.push((0, 4.90));
                sc
            }),
            (0usize, -4i64, 0usize, Vec::<f32>::new(), {
                let mut sc = CandScore::default();
                sc.rel.push((0, 5.80));
                sc
            }),
        ];
        let n = decide_group(&none, 0, -2, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0);
        assert_eq!(
            none[n[0]].1, -2,
            "量不到 usag 的组必须原样走 rel —— 短音上这根轴是噪声,不许拿它决策"
        );
    }

    /// ⭐⭐⭐⭐ S165 —— **失配轴的第二道对手轴闸:不许把音内的谷挖得更深。**
    ///
    /// # ⛔ 它为什么存在
    /// 用户 2026-08-29 听出:开了失配轴之后 4:07.466 从「哑噪声」变成了「**断音**」。
    /// 实测(同一个音 `[696]あ`,三条独立渲染的臂读数一致):
    /// **出厂 `mism=0` 音内跌幅 16.0 dB → `mism=1.2` 变成 21.4 / 21.5 / 21.4 dB**,
    /// 而当时**唯一的对手轴闸**用的是 `uplev`(上方谐波电平)⇒ **一次都没拦住**
    /// (两个候选的上方谐波电平差不多)。
    ///
    /// # ⭐ 用户给的关键定义
    /// 「那里『是谷』不止可能像现在这样吐**噪声**,还可能**完全挖出来什么也没有(断音)**」
    /// ⇒ ⭐⭐ **两种坏法的共同点是「电平掉了」** —— 噪声那种谱平坦度**高**、断音那种**低**,
    ///   **任何只盯一种形态的尺子都会漏掉另一种**。所以闸架在**跌幅**上。
    ///
    /// 钉五件:
    /// ⑴ 闸关着(`cap` 极大)时,失配轴照常换 —— 否则 ⑵ 可能只是「失配轴根本没生效」;
    /// ⑵ 闸开着时,**把谷挖深的那个交换被拦住**;
    /// ⑶ ⛔ **阴性对照**:谷**没有**变深的交换**不许**被拦(否则这个闸等于关掉整根轴);
    /// ⑷ ⛔ **阴性对照**:两个候选都量不到 `dip` 时,闸不许拦(`worst_dip_vs` 恒 0);
    /// ⑸ ⭐ **两处填充点都在** —— 少填一处 `worst_dip_vs` 恒 0,⑵ 会静默失守。
    ///     这里直接对 [`note_dip_db`] 造两种坏法的信号,证明**一把尺子两种都抓得到**。
    #[test]
    fn the_mismatch_axis_may_not_deepen_the_dip_inside_a_note() {
        let sr = 48_000u32;
        // ── ⑸ 先证明尺子对【两种坏法】都有反应,而对「连着」的信号没有 ──
        let tone = |n: usize, gap: Option<(usize, usize, f32)>| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let t = i as f64 / f64::from(sr);
                    let mut v = (2.0 * std::f64::consts::PI * 440.0 * t).sin() as f32 * 0.5;
                    if let Some((a, b, g)) = gap {
                        if i >= a && i < b {
                            v *= g;
                        }
                    }
                    v
                })
                .collect()
        };
        let n = sr as usize / 2; // 0.5 s
        let clean = tone(n, None);
        // 坏法 A:挖空(断音)—— 60 ms 降到 −26 dB
        let mut cut = tone(n, Some((n / 3, n / 3 + sr as usize * 6 / 100, 0.05)));
        // 坏法 B:同一段换成噪声(电平也掉,但谱是平的)
        let mut noisy = clean.clone();
        let (a, b) = (n / 3, n / 3 + sr as usize * 6 / 100);
        let mut seed = 12345u64;
        for v in noisy[a..b].iter_mut() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = ((seed >> 33) as f32 / 2_147_483_648.0 - 0.5) * 0.05;
        }
        let d_clean = note_dip_db(&clean, sr).expect("连着的信号量得到");
        let d_cut = note_dip_db(&cut, sr).expect("断音量得到");
        let d_noisy = note_dip_db(&noisy, sr).expect("噪声量得到");
        assert!(d_clean < 3.0, "「确实连着」的信号跌幅应当很小,实际 {d_clean}");
        assert!(d_cut > 15.0, "断音那种坏法没被抓到(跌幅只有 {d_cut})");
        assert!(d_noisy > 15.0, "噪声那种坏法没被抓到(跌幅只有 {d_noisy})");
        cut.truncate(0);

        // ── ⑴⑵⑶⑷ 闸本身 ──
        let mk = |rel: f32, mism: f32, dip: f32| {
            let mut sc = CandScore::default();
            sc.rel.push((0, rel));
            sc.mism.push((0, mism));
            sc.dip.push((0, dip));
            sc
        };
        // 候选 A(今天的落点):失配差、谷浅   候选 B:失配好、**谷深 5.4 dB**(实测那次的形状)
        let deepen = vec![
            (0usize, -5i64, 0usize, Vec::<f32>::new(), mk(4.0, 2.9, 16.0)),
            (0usize, -8i64, 0usize, Vec::<f32>::new(), mk(5.0, 1.5, 21.4)),
        ];
        // ⑴ 闸关着(cap 由常量给,这里用 `mism_eps` 大 + `usag_dim_cap`=0 走到闸)
        //    ⇒ 先确认失配轴在没有 dip 差时**确实会换**
        let same_dip = vec![
            (0usize, -5i64, 0usize, Vec::<f32>::new(), mk(4.0, 2.9, 16.0)),
            (0usize, -8i64, 0usize, Vec::<f32>::new(), mk(5.0, 1.5, 16.2)),
        ];
        let moved = decide_group(&same_dip, 0, -5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.2);
        assert_eq!(
            same_dip[moved[0]].1, -8,
            "谷没变深时失配轴必须照常换 —— 否则下面那条「被拦住」只是因为这根轴没生效"
        );
        // ⑵ 谷挖深 5.4 dB > MISM_DIP_CAP ⇒ 拦住,留在今天的落点
        let held = decide_group(&deepen, 0, -5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.2);
        assert_eq!(
            deepen[held[0]].1, -5,
            "把音内跌幅从 16.0 挖到 21.4(+5.4 dB > {MISM_DIP_CAP})必须被对手轴闸拦住"
        );
        // ⑶ 阴性对照:谷反而变浅 ⇒ 不许拦
        let shallower = vec![
            (0usize, -5i64, 0usize, Vec::<f32>::new(), mk(4.0, 2.9, 16.0)),
            (0usize, -8i64, 0usize, Vec::<f32>::new(), mk(5.0, 1.5, 11.0)),
        ];
        let ok = decide_group(&shallower, 0, -5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.2);
        assert_eq!(
            shallower[ok[0]].1, -8,
            "谷变浅的交换被拦住了 —— 这个闸把整根失配轴关掉了"
        );
        // ⑷ 阴性对照:量不到 dip ⇒ `worst_dip_vs` 恒 0 ⇒ 不许拦
        let nodip = vec![
            (0usize, -5i64, 0usize, Vec::<f32>::new(), {
                let mut sc = CandScore::default();
                sc.rel.push((0, 4.0));
                sc.mism.push((0, 2.9));
                sc
            }),
            (0usize, -8i64, 0usize, Vec::<f32>::new(), {
                let mut sc = CandScore::default();
                sc.rel.push((0, 5.0));
                sc.mism.push((0, 1.5));
                sc
            }),
        ];
        let nd = decide_group(&nodip, 0, -5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.2);
        assert_eq!(nodip[nd[0]].1, -8, "两边都量不到 dip 时不许拦 —— 那会让短音上整根轴失效");
    }

    /// ⭐⭐⭐⭐⭐ S165 —— **cover 的第二条死因:f0 超出 `f0_to_coarse` 的表达上限。**
    ///
    /// # ⛔ 它治什么
    /// 用户 2026-08-29 点名翻唱轨破音,频谱图上是**谐波结构整个消失、变成宽带噪声**
    /// (4:04.740 那 980 ms:源五条细亮谐波线,cover 只剩两团糊带)。
    /// 根因:`rvc::f0_to_coarse` 的 256 档 mel 表上界是 **1100 Hz**,
    /// 而那个音真 f0 **1422.6 Hz** ⇒ clamp 成 255 ⇒ 模型收到「顶格」⇒ 吐噪声。
    /// ⚠⚠ 它**不能**用 `slot_singable` 表达 —— 1422 Hz 在 `usable` 之内,「唱得动」。
    ///
    /// 钉四件:
    /// ⑴ **超上界的音被判死**(哪怕 `slot_singable` 说唱得动);
    /// ⑵ ⛔ **阴性对照**:同样的音高、但把上界调高到它之上 ⇒ **不判死**
    ///    (否则 ⑴ 可能只是因为 `slot_singable` 自己就判死了);
    /// ⑶ ⛔ **阴性对照**:上界以下的正常高音**一格都不许被新判死**;
    /// ⑷ ⭐ 判死之后**救援真的把它降到上界以下** —— 否则只是换了个地方吐噪声。
    #[test]
    fn cover_treats_f0_above_the_rvc_coarse_ceiling_as_dead() {
        // 这条记录的 usable 顶到 midi 96(≈ 2093 Hz)⇒ 1422 Hz(midi ≈ 89.7)「唱得动」
        let r = SpeakerRange::bounds((40.0, 96.0), (40.0, 96.0));
        let hz = |m: f32| 440.0f32 * 2f32.powf((m - 69.0) / 12.0);
        // ⑶ 先建对照:一段稳稳在上界【以下】的高音(midi 84 ≈ 1046 Hz)
        let below: Vec<f32> = vec![hz(84.0); 200];
        let (j_below, _) = cover_dead_plan_with(&below, 100.0, &r, CoverGrouping::today());
        assert!(
            j_below.is_empty(),
            "上界以下的正常高音(1046 Hz)被判死了 {} 段 —— 这一刀在误伤",
            j_below.len()
        );
        // ⑴ 同样长度、但在上界【以上】(midi 89.7 ≈ 1422 Hz)
        let above: Vec<f32> = vec![hz(89.7); 200];
        let (j_above, _) = cover_dead_plan_with(&above, 100.0, &r, CoverGrouping::today());
        assert!(
            !j_above.is_empty(),
            "1422 Hz 超过 f0_to_coarse 的上界 {RVC_COARSE_MAX_HZ} Hz 却没被判死 —— 
             模型会收到 clamp 成 255 的音高然后吐噪声(见本判据的 doc)"
        );
        // ⑵ 阴性对照:把 usable 顶降到 88(< 89.7)⇒ `slot_singable` 自己就会判死
        //    ⇒ 用它证明 ⑴ 不是靠 `slot_singable` 达成的:这里反过来把上界【抬高】,
        //    如果 ⑴ 真是新判据起的作用,那么在一个 coarse 上界更高的世界里它就不该判死。
        //    (`RVC_COARSE_MAX_HZ` 是常量,不能在测试里改 ⇒ 改用一个**低于**上界的音高,
        //     并确认它在同一条记录下**不**被判死 —— 那就是 ⑶,已经断言过。)
        //    这里再补一条:上界【正上方一点点】也要被判死,证明边界位置就在 1100 Hz。
        // ⚠⚠ 边界位置**不能**用「刚过上界 5%」来钉:那只需要降 1 个半音,
        //    而 [`COVER_MIN_RESCUE_DEPTH`] = 3 会把它整组丢掉(既有设计,不是 bug)。
        //    ⇒ 用**刚过 3 个半音**的那一档来钉边界:它是这条死因**实际能生效的最浅处**。
        let just_deep: Vec<f32> = vec![RVC_COARSE_MAX_HZ * 2f32.powf(3.2 / 12.0); 200];
        let (j_ja, _) = cover_dead_plan_with(&just_deep, 100.0, &r, CoverGrouping::today());
        assert!(
            !j_ja.is_empty(),
            "超上界 3.2 个半音的音没被判死 —— 这条死因在它实际能生效的最浅处就失守了"
        );
        let just_below: Vec<f32> = vec![RVC_COARSE_MAX_HZ * 0.95; 200];
        let (j_jb, _) = cover_dead_plan_with(&just_below, 100.0, &r, CoverGrouping::today());
        assert!(j_jb.is_empty(), "刚不到上界 5% 的音被判死了 —— 边界位置不对");
        // ⚠ 如实登记:**超界 0-3 个半音的音这条刀够不着**(被 COVER_MIN_RESCUE_DEPTH 挡下)。
        //   实测这份素材里超界帧的分布是「要么不超、要么超很多」,所以这个盲区代价可接受;
        //   但它是**真盲区**,别读成「全覆盖」。
        // ⑷ 救援真的把它降到上界以下
        let sh = j_above[0].shift;
        assert!(sh < 0, "救援位移应当是往下的,实际 {sh:+}");
        let after = hz(89.7) * 2f32.powf(sh as f32 / 12.0);
        assert!(
            after <= RVC_COARSE_MAX_HZ,
            "救援之后 f0 还是 {after:.0} Hz(> {RVC_COARSE_MAX_HZ})—— 只是换了个地方吐噪声"
        );
    }

    use super::*;

    fn range() -> SpeakerRange {
        SpeakerRange::bounds((48.0, 84.0), (52.0, 79.0))
    }

    /// 窗测试用的「**处处可达**」记录 —— 语义 = 「护栏区材料是健康的」。
    /// ⛔ 它不是「把新判据关掉」的开关:关掉判据的写法(传 `None` 走老路)会让
    /// S163 那条护栏收窄**没有任何闸守着**。这里传的是真实谓词,只是喂了一个处处成立的
    /// 音域 ⇒ 老断言仍然逐条走生产代码,而且语义比以前更明确。
    /// 新行为由 [`tests::the_guard_never_reaches_into_a_note_that_collapses_on_this_pass`] 钉住。
    fn all_reachable() -> SpeakerRange {
        SpeakerRange::bounds((0.0, 127.0), (0.0, 127.0))
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

    /// S160e —— **拆组**:一段里深浅需求差得多时,在**清音连段的正中**拆开,
    /// 免得只需要浅位移的那一半被深的那一半拖下去(用户 2026-08-24 点名的 4:34.544「失声」)。
    #[test]
    fn cover_split_puts_the_seam_in_the_unvoiced_run_and_only_when_it_pays() {
        // usable 顶 77 = 用户那份 yachiyo sidecar 今天的值。
        // ⛔ 第一个元组是 **usable**(第二个是 comfort)—— 第一版写反了,于是 MIDI 80
        //   变成「唱得动」、根本不进死音组,判据当场红。usable 顶 77 = 用户那份
        //   yachiyo sidecar 今天的值。
        let r = SpeakerRange::bounds((36.0, 77.0), (36.0, 75.0));
        // 左边 0.6 s 的 MIDI 80(只需 −3),右边 0.6 s 的 MIDI 91(需 −14),
        // 中间夹 0.08 s 清音(8 帧 ⇒ 过得了「≥3 帧」这一条,也过得了 GAP_TOL 150 的桥接)。
        let mut f0 = Vec::new();
        f0.extend(std::iter::repeat(0.0).take(20));
        f0.extend(std::iter::repeat(hz(80.0)).take(60));
        f0.extend(std::iter::repeat(0.0).take(8));
        f0.extend(std::iter::repeat(hz(91.0)).take(60));
        f0.extend(std::iter::repeat(0.0).take(20));

        // ⛔ 先钉「不拆」的样子:整段一个位移,由最深的那一头决定。
        let g_off = CoverGrouping::pinned(150.0, 250.0, 0.0, 250.0);
        let (jobs_off, _) = cover_dead_plan_with(&f0, 100.0, &r, g_off);
        assert_eq!(jobs_off.len(), 1, "拆组关 ⇒ 一段:{jobs_off:?}");
        let deep = jobs_off[0].shift;
        assert!(deep <= -14, "整段该被最深的那一头拖到 ≤ −14,实际 {deep}");

        // 拆组开 ⇒ 两段,而且浅的那一半确实浅了。
        let g_on = CoverGrouping::pinned(150.0, 250.0, 1000.0, 250.0);
        let (jobs_on, _) = cover_dead_plan_with(&f0, 100.0, &r, g_on);
        assert_eq!(jobs_on.len(), 2, "拆组开 ⇒ 两段:{jobs_on:?}");
        assert!(
            jobs_on[0].shift > deep,
            "浅的那一半应当比整段浅:{:?} vs {deep}",
            jobs_on[0].shift
        );
        assert!(jobs_on[1].shift <= -14, "深的那一半仍然要够深:{:?}", jobs_on[1].shift);

        // ⭐ 缝必须落在**清音**里:两段的接缝处 f0 == 0。
        let seam = jobs_on[0].end as usize;
        assert_eq!(f0[seam - 1], 0.0, "缝的左侧必须是清音(实际 {})", f0[seam - 1]);

        // ⛔ 收益门槛真的挡得住:把门槛抬到天上 ⇒ 回到一段。
        let g_hi = CoverGrouping::pinned(150.0, 250.0, 1.0e6, 250.0);
        let (jobs_hi, _) = cover_dead_plan_with(&f0, 100.0, &r, g_hi);
        assert_eq!(jobs_hi.len(), 1, "门槛够高就不该拆:{jobs_hi:?}");

        // ⛔ 每一半的最小长度也真的挡得住:要求每半 ≥ 1 s ⇒ 拆不动(每半只有 0.6 s)。
        let g_long = CoverGrouping::pinned(150.0, 250.0, 1000.0, 1000.0);
        let (jobs_long, _) = cover_dead_plan_with(&f0, 100.0, &r, g_long);
        assert_eq!(jobs_long.len(), 1, "每半 ≥1 s 的要求该挡住这一刀:{jobs_long:?}");
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

        /// S165 §105 —— 钉住 [`MIN_VIOLATION_MS`] 的出厂值,以及它两头各自还守着什么。
    ///
    /// 它从 250 ms 降到 100 ms,是因为 250 挡掉的不只是幻影:用户报的七处里有五处
    /// 根本没被救援过,而它们唱在 724-1178 Hz、模型可用顶只有 698 Hz。同 tier 三臂实测
    /// 全曲坏帧率 5.85 %(250)→ 5.24 %(150)→ 4.64 %(100),两族齐降,坏段 27 → 15。
    ///
    /// 这条判据钉三件,缺一不可:
    /// ⑴ 出厂值就是 100 ms;
    /// ⑵ **够长的真死区必须成区** —— 10 个浊死帧(= 门槛本身)要救,9 个不救;
    /// ⑶ ⭐ **幻影岛仍然挡得住** —— 降门槛最该担心的就是这个。rmvpe 把气声读高八度是
    ///    几帧几帧地读,桥接容差只有 30 ms,所以那些爆点各自独立、每个都远够不着 10 帧。
    #[test]
    fn the_dead_region_threshold_ships_at_100ms_and_still_blocks_phantoms() {
        assert!(
            (super::MIN_VIOLATION_MS - 100.0).abs() < f32::EPSILON,
            "出厂门槛应当是 100 ms —— 改它之前先重跑 §105 那组同 tier 三臂 A/B"
        );
        let g = super::CoverGrouping::pinned(
            super::GAP_TOL_MS,
            super::MIN_VIOLATION_MS,
            super::COVER_SPLIT_MIN_GAIN,
            super::COVER_SPLIT_MIN_PART_MS,
        );
        // ⑵ 一段连续的高音:10 帧(= 100 ms)要救,9 帧不救。
        let run_of = |n: usize| {
            let mut f0 = vec![hz(60.0); 400];
            f0.extend(vec![hz(95.0); n]);
            f0.extend(vec![hz(60.0); 400]);
            f0
        };
        let (j10, _) = cover_dead_plan_with(&run_of(10), 100.0, &range(), g);
        assert!(!j10.is_empty(), "刚好够门槛的真死区必须成区(10 帧 = 100 ms)");
        let (j9, _) = cover_dead_plan_with(&run_of(9), 100.0, &range(), g);
        assert!(j9.is_empty(), "差一帧就不该成区,否则门槛形同虚设");
        // ⑶ 幻影岛:5 组各 3 帧,组间隔 4 帧活帧 > 桥接容差 ⇒ 各自独立,谁也够不着门槛。
        let mut ph = vec![hz(60.0); 400];
        for _ in 0..5 {
            ph.extend(vec![hz(95.0); 3]);
            ph.extend(vec![hz(60.0); 4]);
        }
        ph.extend(vec![hz(60.0); 400]);
        let (jp, up) = cover_dead_plan_with(&ph, 100.0, &range(), g);
        assert!(
            jp.is_empty() && up.is_empty(),
            "把门槛降到 100 ms 之后,几帧几帧的幻影爆点仍然不许触发染色"
        );
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

    /// ⭐ S165 —— `seam_align_wide` 的两条不变量。
    ///
    /// ⛔ 它存在的理由:对齐原本用**基础淡化宽度 `xf`(10 ms)**当搜索窗,却把整个片段
    /// 按搜出来的 lag **整体平移** —— 而 `tied_xfade` 那一族的淡入是 **120 ms**。
    /// 10 ms 在 f0≈500 Hz 上只有 5 个周期 ⇒ 互相关有**周期歧义**,会锁到错的周期
    /// (实测生产挪 −48 / +73,而 120 ms 窗上的真值是 +21 / +154,挪完相干度变成负的)。
    ///
    /// ⚠ 出厂**关**:方向对但成品几乎不动(§54.7)⇒ 这条判据锁的是
    /// 「**关着时逐位不变**」与「**没有长淡化时开关无意义**」,不是它有没有用。
    #[test]
    fn wide_seam_align_is_a_no_op_without_a_long_crossfade() {
        const SR: u32 = 48_000;
        const N: usize = 48_000;
        let wave = |i: usize| -> f32 {
            let t = i as f64 / f64::from(SR);
            ((2.0 * std::f64::consts::PI * 220.0 * t).sin()
                + 0.5 * (2.0 * std::f64::consts::PI * 517.0 * t).sin()) as f32
        };
        let base0: Vec<f32> = (0..N).map(wave).collect();
        let donor: Vec<f32> = (0..N).map(|i| if i < 13 { 0.0 } else { wave(i - 13) }).collect();
        let jobs = [DeadJob { shift: -6, start: 20, end: 60 }];
        let spf = base0.len() as f64 / 100.0;
        let xf = (SR as usize / 100).max(2);
        let run = |wide: bool| -> Vec<f32> {
            let mut b = base0.clone();
            let mut kept: Vec<(i64, usize, Vec<f32>)> = Vec::new();
            if let Some((lo, hi)) = donor_read_span(&jobs[0], spf, b.len(), MERGE_BRIDGE_FRAMES) {
                kept.push((0, lo, donor[lo..hi].to_vec()));
            }
            // ⚠ `tied_xf = 0` ⇒ `xf_at` 恒空 ⇒ 宽窗退化成 `xf` ⇒ 两档必须逐位相同。
            splice_kept(
                &mut b, SR, spf, &jobs, &kept, xf, false, 48, 0.0, &[], 0, 0.0, true, 0, false, wide,
            )
            .unwrap();
            b
        };
        assert_eq!(
            run(false),
            run(true),
            "没有长交叉淡化时 `xf_at` 是空的 ⇒ 宽窗退化成 `xf` ⇒ 两档必须逐位相同"
        );

        // ⛔ 阴性对照:这个夹具本身要能分辨「对齐有没有在做事」,否则上面那条是恒真的。
        let mut b0 = base0.clone();
        let mut kept: Vec<(i64, usize, Vec<f32>)> = Vec::new();
        if let Some((lo, hi)) = donor_read_span(&jobs[0], spf, b0.len(), MERGE_BRIDGE_FRAMES) {
            kept.push((0, lo, donor[lo..hi].to_vec()));
        }
        splice_kept(
            &mut b0, SR, spf, &jobs, &kept, xf, false, 0, 0.0, &[], 0, 0.0, true, 0, false, false,
        )
        .unwrap();
        assert_ne!(b0, run(false), "对齐半径 0 与 48 必须给出不同结果 —— 否则这个夹具测不到对齐");
    }

    /// ⭐⭐⭐ S165 —— 交接点体检**看的那一段**必须是这条缝真正的淡入区。
    ///
    /// ⛔ 病:`tied_xfade` 把右窗的淡入起点**往前拉 120 ms**,而体检一直只看交接点
    /// **往后 40 ms** —— 两段完全不重叠,体检结构上看不见自己该拦的那一段
    /// (实测 yuyuko × 炉心 4:36.319:往后 40 ms 落差 5.27 dB 门限够不着,
    ///  往前 120 ms 落差 **12.55 dB**)。
    ///
    /// 夹具就照着那个形状造:进来的那条 donor 在交接点**之前**是静音、之后立刻恢复
    /// ⇒ **今天的窗(往后)看不见它,往前的窗看得见**。
    #[test]
    fn the_handover_check_looks_at_the_actual_fade_in_window() {
        const SR: u32 = 48_000;
        const FR: i64 = 100; // 100 帧
        let spf = 480.0f64; // 每帧 480 样本 ⇒ 10 ms/帧
        let n = (FR as f64 * spf) as usize;
        // 两条窗:左 [0,50)、右 [50,100),shift 不同 ⇒ 是一条跨 shift 的缝。
        let jobs = [DeadJob { shift: -6, start: 0, end: 50 }, DeadJob { shift: -9, start: 50, end: 100 }];
        let order = [0usize, 1usize];
        let t0 = (50.0 * spf) as usize; // 交接点 = 右窗起点
        let tied_xf = (SR as usize) * 120 / 1000; // 120 ms
        let xf = (SR as usize) / 100; // 10 ms

        // 左 donor:全程响亮。
        let lseg: Vec<f32> = (0..n).map(|i| 0.3 * ((i % 97) as f32 / 97.0 - 0.5)).collect();
        // 右 donor:交接点**之前** tied_xf 那一段是静音,之后立刻恢复。
        let rseg: Vec<f32> = (0..n)
            .map(|i| {
                if i + tied_xf >= t0 && i < t0 {
                    0.0
                } else {
                    0.3 * ((i % 89) as f32 / 89.0 - 0.5)
                }
            })
            .collect();
        let kept = vec![(0i64, 0usize, lseg), (1i64, 0usize, rseg)];
        // 一个横跨交接点的**延续**长音 ⇒ `tied_here(t0)` 成立。
        let notes = [NoteSpan { start: 20, frames: 60, sung: true, hz: 440.0, tied: true }];

        let run = |fade_window: bool| {
            defer_dead_handover(SR, spf, &jobs, &kept, &order, 15.0, &notes, tied_xf, xf, fade_window, 0.0)
        };

        // ⑴ 今天的窗(往后 40 ms):那里右 donor 已经恢复 ⇒ **看不见** ⇒ 不该触发。
        assert!(
            run(false).is_empty(),
            "往后 40 ms 的窗里进来的那条已经恢复了 —— 今天的体检本就该看不见它(这是病,不是判据)"
        );
        // ⑵ 对准淡入区(往前 120 ms):那里右 donor 是静音 ⇒ **必须触发**。
        let fixed = run(true);
        assert!(
            !fixed.is_empty(),
            "对准淡入区之后,进来的那条在整段淡入里都是静音 —— 体检必须拦住它"
        );
        // ⑶ 挪的方向必须是**往后**(让出去的那条多唱一截),而且不许超过上限。
        let t = fixed[&0usize];
        assert!(t > t0, "交接点必须往后挪 —— 拿到 {t} vs 原来的 {t0}");
        let max_defer = (f64::from(SR) * HANDOVER_MAX_MS / 1000.0) as usize;
        assert!(t <= t0 + max_defer, "挪的量不许超过 HANDOVER_MAX_MS —— {t} vs {}", t0 + max_defer);

        // ⑷ ⛔ 阴性对照:同一个夹具,但那个音**不是延续音也不横跨交接点**
        //    ⇒ `tied_here` 不成立 ⇒ 开关开着也必须退回今天的窗 ⇒ 不触发。
        let notes_off = [NoteSpan { start: 0, frames: 10, sung: true, hz: 440.0, tied: false }];
        assert!(
            defer_dead_handover(SR, spf, &jobs, &kept, &order, 15.0, &notes_off, tied_xf, xf, true, 0.0)
                .is_empty(),
            "不是 tied 缝时,这一刀必须退回今天的窗 —— 否则它会波及所有接缝"
        );

        // ⑸ ⛔⛔ **回归判据**:这一刀是【额外看一段】,不许把原本拦得住的那一处放走。
        //    第一版写成「tied 缝改用往前的窗」,实测当场退化:`290.680` 那一处
        //    (往后 40 ms 落差 27-45 dB)关着时被拦住,开着**反而不触发了**。
        //    夹具就照那个形状造:进来的那条在交接点**之后**弱、**之前**不弱。
        let rseg_after: Vec<f32> = (0..n)
            .map(|i| {
                if i >= t0 && i < t0 + (SR as usize) * 60 / 1000 {
                    0.0
                } else {
                    0.3 * ((i % 89) as f32 / 89.0 - 0.5)
                }
            })
            .collect();
        let kept_after = vec![(0i64, 0usize, kept[0].2.clone()), (1i64, 0usize, rseg_after)];
        let hit_off =
            defer_dead_handover(SR, spf, &jobs, &kept_after, &order, 15.0, &notes, tied_xf, xf, false, 0.0);
        let hit_on =
            defer_dead_handover(SR, spf, &jobs, &kept_after, &order, 15.0, &notes, tied_xf, xf, true, 0.0);
        assert!(!hit_off.is_empty(), "往后那一段是静音 —— 今天的体检本来就拦得住它");
        assert!(
            !hit_on.is_empty(),
            "⛔ 开着之后反而放走了原本拦得住的缝 —— 这一刀是【额外看一段】,不是替换那一段"
        );
    }

    /// ⭐⭐⭐ S165 —— 元音开头的音节边界算「接得上」,而**音高约束一个字节不许动**。
    ///
    /// 夹具全部照**真实坐标**造(yuyuko × 炉心融解 × +7):
    /// * `[784]な(85) → [785]あ(83)` —— 用户 2026-08-28 报「4:29 接缝挺明显」的那一处:
    ///   歌词不同、但下一个是纯元音、音高差 2 ⇒ **该判 tied**(拿 120 ms 淡入);
    /// * `[792]あ(90) → [793]あ(85)` —— 用户 2026-08-27 报的 4:36 事故:
    ///   歌词相同但**差 5 个半音** ⇒ **绝不许判 tied**(否则把两个音色混 120 ms,比不治更糟);
    /// * `し → か` —— 下一个音有辅音 ⇒ **不许判 tied**(拉长会糊掉起音,这是那条规则的本意)。
    #[test]
    fn a_vowel_onset_bridges_a_syllable_boundary_but_never_a_pitch_jump() {
        let ly = |v: &[&str]| -> Vec<String> { v.iter().map(|s| (*s).to_string()).collect() };
        let fr = vec![9i64; 8];
        let run = |nums: &[i64], lyr: &[&str], on: bool| -> Vec<bool> {
            note_spans_tied_with(nums, &fr[..nums.len()], 0, &ly(lyr), on)
                .iter()
                .map(|s| s.tied)
                .collect()
        };

        // ⑴ 4:29 那一处:な(85) → あ(83)。关着 = 今天(不 tied);开着必须 tied。
        assert_eq!(run(&[85, 83], &["な", "あ"], false)[1], false, "关着必须与今天逐位相同");
        assert_eq!(
            run(&[85, 83], &["な", "あ"], true)[1],
            true,
            "な(85) → あ(83):下一个是纯元音、音高差 2 ⇒ 该判 tied"
        );

        // ⑵ ⛔ 4:36 那个事故:あ(90) → あ(85),差 5 个半音。开着也**绝不许** tied。
        assert_eq!(
            run(&[90, 85], &["あ", "あ"], true)[1],
            false,
            "⛔ 差 5 个半音是旋律不是延续音 —— 放宽歌词条件不许动 TIED_MAX_ST"
        );
        // 同一对歌词、音高连得上 ⇒ 本来就该 tied(两档都一样)。
        assert_eq!(run(&[85, 83], &["あ", "あ"], false)[1], true, "歌词相同 + 音高连得上 = 今天就 tied");

        // ⑶ ⛔ 下一个音有辅音 ⇒ 不许 tied(拉长会糊掉起音)。
        for nxt in ["か", "し", "ぱ", "ん", "っ"] {
            assert_eq!(
                run(&[85, 83], &["な", nxt], true)[1],
                false,
                "「{nxt}」不是纯元音开头 ⇒ 不许判 tied(那条规则本来就是为了保护起音)"
            );
        }

        // ⑷ 片假名同样算元音开头;延音记号不受音高约束(谱面明写「接着唱」)。
        assert_eq!(run(&[85, 83], &["ナ", "ア"], true)[1], true, "片假名的纯元音也算");
        assert_eq!(run(&[90, 60], &["あ", "ー"], true)[1], true, "延音记号不受 TIED_MAX_ST 约束");

        // ⑸ ⛔ 阴性对照:隔着休止不算同一个长音。
        assert_eq!(
            run(&[85, 0, 83], &["な", "R", "あ"], true)[2],
            false,
            "隔着休止就不是同一个长音了 —— 这条不许被放宽绕过"
        );
    }

    /// ⭐⭐⭐ S159zm —— **拼接前先对齐**:缝两侧其实是同一条波形,只是错开了零点几毫秒。
    ///
    /// 机理与实测在 [`SEAM_ALIGN_MS_DEFAULT`] 的 doc(30 条 donor↔donor 的缝:
    /// 零滞后 ρ **−0.139**、最佳滞后处 |ρ| **0.928**、最佳滞后中位 **8 样本 = 0.18 ms**)。
    ///
    /// 这条判据钉三件:
    /// ⑴ **旋钮关着时逐位不变**(出厂默认 0.0 ⇒ 今天的输出一个样本都不许动);
    /// ⑵ 开着时**真的找到那个偏移** —— donor 是 base **延迟 k 个样本**的副本,
    ///    对齐之后淡化区必须与 base 逐样本一致(误差 < 1e-6);
    /// ⑶ **半径不够时不许乱挪** —— 把半径设成小于真实偏移,输出不许比不挪更差。
    ///
    /// ⛔ 变异(写这条判据时逐个真跑过,读数记在各行后面)。
    #[test]
    fn the_splicer_aligns_the_two_arms_before_it_crossfades() {
        const SR: u32 = 48_000;
        const N: usize = 48_000;
        const LAG: usize = 13; // 0.27 ms —— 与实测中位(8 样本)同一个量级
        // 一段**逐样本都在变**的材料:等值常量会让「对齐」这件事恒真。
        let wave = |i: usize| -> f32 {
            let t = i as f64 / f64::from(SR);
            ((2.0 * std::f64::consts::PI * 220.0 * t).sin()
                + 0.5 * (2.0 * std::f64::consts::PI * 517.0 * t).sin()) as f32
        };
        let base0: Vec<f32> = (0..N).map(wave).collect();
        // donor = base 延迟 LAG 个样本 ⇒ 正确的对齐偏移就是 −LAG(往回读)。
        let donor: Vec<f32> = (0..N).map(|i| if i < LAG { 0.0 } else { wave(i - LAG) }).collect();
        let jobs = [DeadJob { shift: -6, start: 20, end: 60 }];

        let run = |align_ms: f64| -> Vec<f32> {
            let mut b = base0.clone();
            let align = (align_ms * f64::from(SR) / 1000.0).round() as usize;
            let spf = b.len() as f64 / 100.0;
            let mut kept: Vec<(i64, usize, Vec<f32>)> = Vec::new();
            if let Some((lo, hi)) = donor_read_span(&jobs[0], spf, b.len(), MERGE_BRIDGE_FRAMES) {
                kept.push((0, lo, donor[lo..hi].to_vec()));
            }
            let xf = (SR as usize / 100).max(2);
            splice_kept(&mut b, SR, spf, &jobs, &kept, xf, false, align, 0.0, &[], 0, 0.0, true, 0, false, false).unwrap();
            b
        };

        // ⑴ 关着 = 今天,逐位。
        let off = run(0.0);
        assert_eq!(off, run(0.0), "同一档两次跑必须逐位相同(纯函数)");

        // ⑵ 开着必须找到那个偏移 —— 对齐之后 donor 与 base 逐样本一致 ⇒ 淡化区就是 base。
        let on = run(1.0); // 半径 48 样本 > LAG
        assert_ne!(on, off, "对齐开着却逐位相同 ⇒ 它没生效");
        let a = (20.0 * (N as f64 / 100.0)) as usize; // 窗起点(样本)
        let xf = (SR as usize / 100).max(2);
        let err = |y: &[f32], lo: usize, hi: usize| -> f64 {
            (lo..hi).map(|i| (f64::from(y[i]) - f64::from(base0[i])).abs()).fold(0.0, f64::max)
        };
        assert!(
            err(&on, a, a + xf) < 1e-6,
            "对齐之后淡化区必须与 base 一致(最大逐样本差 {})",
            err(&on, a, a + xf)
        );
        assert!(
            err(&off, a, a + xf) > 0.05,
            "阴性对照:不对齐时淡化区必须**明显**偏离 base(读到 {}),否则 ⑵ 是恒真的",
            err(&off, a, a + xf)
        );

        // ⑶ ⭐ **旋钮真的在限半径**:0.1 ms = 5 样本 < LAG(13)⇒ 它**够不着**那个偏移,
        //    误差必须仍然在「不挪」的量级上(而不是被修好)。
        // ⛔ 写这条判据时我先写的是「不许比不挪更差」,而变异(把 `r` 从 `align` 换成 `xf`)
        //    **照绿** —— 半径变大之后它照样找得到,「不更差」当然成立。
        //    ⇒ 要盯的是**上界**,断言必须是「够不着」而不是「不更差」。
        let small = run(0.1);
        assert!(
            err(&small, a, a + xf) > 0.5 * err(&off, a, a + xf),
            "半径 5 样本竟然把 13 样本的偏移修好了 ⇒ 旋钮没在限半径({} vs 不挪 {})",
            err(&small, a, a + xf),
            err(&off, a, a + xf)
        );
        assert!(
            err(&small, a, a + xf) <= err(&off, a, a + xf) + 1e-9,
            "半径不够时反而比不挪更差({} vs {})",
            err(&small, a, a + xf),
            err(&off, a, a + xf)
        );
    }

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

        // ⚠⚠ S159zm —— **显式关掉拼接前的对齐**([`SEAM_ALIGN_MS_DEFAULT`],出厂 2.0 ms)。
        // 这条判据钉的是「两阶段拼接 == 一遍式参照实现」,而参照实现里没有对齐;
        // 两把刀一起开时它读到样本 9601 就分开(−0.18398008 vs −0.18397935)。
        // ⛔ 正确的隔离是**关掉另一把刀**,不是给参照实现也抄一份对齐 ——
        //    那样参照就不再是「独立写的」,这条判据的全部价值就没了。
        let mut got = mk(0, 220.0);
        apply_dead_only_windows_with(
            &mut got,
            sr,
            total,
            &jobs,
            &[],
            &[],
            false,
            join_rests_enabled(),
            0, // ⛔ 关掉对齐:见上面那段
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            |s, _own| Ok(donor_of(s)),
        )
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
        splice_kept(&mut off, sr, spf, &jobs, &kept, xf, false, 0, 0.0, &[], 0, 0.0, true, 0, false, false).unwrap();
        let mut on = base.clone();
        splice_kept(&mut on, sr, spf, &jobs, &kept, xf, true, 0, 0.0, &[], 0, 0.0, true, 0, false, false).unwrap();

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
        splice_kept(&mut out, sr, spf, &diff, &short, xf, true, 0, 0.0, &[], 0, 0.0, true, 0, false, false).unwrap();
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
            apply_dead_only_windows_with(&mut b, sr, total, &jobs, &[], &[], false, join, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, |s, _| {
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
            apply_dead_only_windows_with(&mut base, SR, TOTAL_FRAMES, &jobs, &[], &[], false, join, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, |s, own| {
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
                &mut tight, SR, TOTAL_FRAMES, &jobs, &[], &[], false, join, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
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
    /// * S159zk 的 `neighbour_ok` **右**侧那道门去掉 ⇒ ⑷/⑹ 读回 `[1..1] −13, [2..3] −2`,**红**
    ///   —— 那正是用户耳判报「1034 塌了」的那个形状;
    /// * **左**侧那道门去掉 ⇒ ⑺ 的 `[92, 40, 81]` 当场拆开,**红**。
    ///   ⚠ 写这条判据时我先只写了 ⑹,而「去掉左门」是**绿的** —— 那半条规则当时是**没人盯的**。
    ///   ⛔ 一个绿不代表任何事,除非它红过。
    /// * `slot_reachable` 改回 `slot_singable` ⇒ ⑻ 读 `[1..2] −9, [3..3] −5`(断点被推到 `[2|3]`),**红**。
    ///   ⛔⛔ 血训:⑻ 一开始只断言「拆成两组」+「位移是 −5」,而那条变异**照绿** ——
    ///   换谓词之后它把断点从 `[1|2]` 推到 `[2|3]`,**组数与位移一个字没变**。
    ///   **「拆了」和「断在该断的地方」是两件事**;凡是钉「断点」的判据,断言里必须出现下标。
    #[test]
    fn a_run_of_dead_notes_splits_where_the_depth_requirement_drops() {
        let r = dxl_like(); // usable [36,80];81 起是死音,落点只到 79 ⇒ 81 要 −2、92 要 −13
        let plan = |p: &[i64], f: &[i64]| dead_only_plan_with(p, f, 0, &r, RescueTuning::today()).0;

        // ⑷ ⭐ 主臂 —— 3:16 的形状:顶音要 −13,后面两个只高出 usable 顶 1 度、只要 −2。
        // ⚠⚠ S159zk 时断点被推到 `[2|3]`,理由是「`[1|2]` 会让浅组的窗倒着伸进顶音 92,
        //    而 92 在浅组的 −2 上是 MIDI 90,那一遍唱不动」。
        // ⭐⭐⭐ S163 —— **危害的载体是护栏,不是断点**:[`dead_group_windows_raw`] 现在会在
        //    这种边上把那一侧的护栏收到 0,所以 `[1|2]` 变得可行,而且**明显更好**:
        //    * 旧 `[1..2]@−13` ⇒ 那个 **81 陪着顶音走 −13,落到 68**;
        //    * 新 `[1..1]@−13` + `[2..3]@−2` ⇒ 81 落到 **79**。
        //    ⇒ 这个夹具正是用户 2026-08-26 点名的 yuyuko 2:09「嗓子里卡着痰」的缩影
        //      (ま 82 陪着 92 走 −14 落到 **68**,女声模型在那儿唱气泡音)。
        let deep = [0i64, 92, 81, 81, 0];
        let g = plan(&deep, &secs(deep.len()));
        assert_eq!(
            g,
            vec![
                DeadGroup { start: 1, end: 2, shift: -13 },
                DeadGroup { start: 3, end: 3, shift: -2 },
            ],
            "全是死音也要按深度拆开,而且陪绑的音一个都不许多(读到 {g:?})"
        );
        assert!(
            !r.slot_reachable(81 - 13) || r.slot_reachable(81 + g[1].shift),
            "夹具有效性:被解放的那个 81 在新落点上必须唱得动,否则这一刀没买到东西"
        );
        // ⭐⭐⭐ 这一刀买到的就是这 11 个半音:不拆的话那个音会跟着走 −13,
        // 按 S159z 的定价 ≈ 白丢 **高频 14.4 dB**。
        assert_eq!(g[0].shift.abs() - g[1].shift.abs(), 11, "省下来的正是这 11 个半音");

        // ⑹ ⭐⭐⭐ **断点的可行性** —— 用户 2026-08-22 耳判报的那条(「1034 在 zk 也塌了」)。
        //
        // `dead_group_windows` 的 `GUARD_FRAMES` 让每个窗倒着伸进旁边那个音 2 帧,而 donor
        // **按那一遍的位移唱整条谱** ⇒ 旁边那个音在这一遍掉出音域时模型直接塌掉。
        // 实测(鹅妈妈 +7 × 东雪莲):窗边紧邻的 127 个唱音里,那一遍掉出音域的 24 个比组内音
        // 低 **−11.4 dB**,没掉出去的 103 个是 **−0.3 dB**;而那 24 个**全部**是二级拆造出来的
        // (拆组前 67 条边:0 个)。
        //
        // ⭐⭐⭐ S163 —— 判据迁到**窗层**(危害的载体在那儿),而且**一条断言钉两个方向**:
        // * 浅组 `[2..3]@−2` 的 `pre` **不许**伸进 `[1] = 92`(92−2 = 90 > reach 顶 80);
        // * 深组 `[1..1]@−13` 的 `post` **必须照常**伸进 `[2] = 81`(81−13 = 68,唱得动)。
        // ⇒ 交叉淡化仍然有地方压(两窗仍然重叠),只是压在**健康**的那一侧。
        // ⛔ 断点层的老断言只能钉前者;少了后者,「把护栏全关掉」会照绿。
        assert_eq!(
            (g[0].start, g[0].end),
            (1, 2),
            "断点必须避开「浅组的窗会伸进一个它唱不动的音」那一处(读到 {g:?})"
        );
        assert!(
            r.slot_reachable(81 + g[1].shift) && !r.slot_reachable(92 + g[1].shift),
            "夹具有效性:81 在浅组位移上必须唱得出、92 必须唱不出,否则 ⑹ 是恒真的"
        );

        // ⑺ ⭐ **左侧那道门**(左/深组的窗往前伸进 `p+1`)。
        //
        // ⛔⛔ 它**只有在有夹心时才够得着**,而且构造起来相当窄 —— 写这条判据时我先随手写了
        // `[92, 40, 81]`,真跑读到**零组**:整句的 `minimal_rescue_shift_capped` 本来就把
        // 夹心音算进 `all` 并要求它 `slot_reachable`,所以**整句层面永远拖不出边界**,
        // 那个夹具只是变成了「这一句无解」。
        // ⇒ 唯一的活路是:**拆出来的深组自己重解位移时,夹心音不在它的 `all` 里**,
        //    而落点旋钮(出厂 `Some(3)`)允许它比整句再深最多 3 度。那 3 度就是这道门的战场。
        //
        // 夹具(`pya_like`:`reach` = `usable_auto` = [36, 80]):`[90, 49, 81]`
        // * 整句:`s = −14` 会让夹心 49 → 35 掉出下边界 ⇒ 整句只能停在 −11…−13;
        // * 拆出来的深组只有 90 ⇒ 它自己能走到 **−14**(76 的 `low_ratio` 0.21 比 79 的 0.63 好);
        // * 而 `49 − 14 = 35` **唱不出** ⇒ 这个断点不可行 ⇒ **不许拆**。
        let pr = pya_like();
        let sandwich = [0i64, 90, 49, 81, 0];
        let gw = dead_only_plan_with(&sandwich, &secs(sandwich.len()), 0, &pr, RescueTuning::today()).0;
        assert!(
            pr.slot_reachable(49) && !pr.slot_reachable(49 - 14),
            "夹具有效性:夹心 49 本来唱得出、被 −14 一拖必须掉出下边界,否则 ⑺ 是恒真的"
        );
        assert!(
            !gw.is_empty(),
            "夹具有效性:这一句必须**有解**(否则读到零组,测的是「无解」不是「不许拆」)"
        );
        // ⭐⭐⭐ S163 —— 同样迁到窗层:断点**照拆**(81 不必陪着走 −14),
        // 而深组的窗**不许**伸进夹心 49(49−14 = 35 掉出下边界),浅组的**可以**(49−5 = 44)。
        assert_eq!(
            gw.len(),
            1,
            "断点会让深组的窗伸进一个它唱不出的夹心音 ⇒ 不许拆(读到 {gw:?})"
        );
        // ⑻ ⭐⭐⭐ **谓词必须是 `slot_reachable`,不许是 `slot_singable`**(S146f 的学费)。
        //
        // 差别只在「用户那条救援线」`usable` 与「扫描到的边界」`reach` 之间那一段。
        // `pya_like`:`usable` 顶 **76**、`usable_auto`(= `reach`)顶 **80** ⇒ 77-80 这四格
        // **模型唱得出、但在用户线之外**。
        //
        // 夹具 `[85, 81, 81]`:断点 `[1|2]` 会让浅组的窗倒着伸进 85,而 85 在浅组的位移上
        // 落到 **80** —— `slot_reachable` 判**能**,`slot_singable` 判**不能**。
        // ⇒ 用 `slot_singable` 的话这一处就白白不拆了,而且**用户每把上限调低一格,
        //    就多一处不拆 ⇒ 陪绑更深** —— 那正是 S146f 用户耳判报的那条退化的形状。
        let bar = [0i64, 85, 81, 81, 0];
        let gb = dead_only_plan_with(&bar, &secs(bar.len()), 0, &pr, RescueTuning::today()).0;
        assert!(
            pr.slot_reachable(80) && !pr.slot_singable(80),
            "夹具有效性:80 必须是「唱得出但在用户线之外」那一格,否则 ⑻ 两个谓词读一样"
        );
        // ⛔⛔ 断言必须钉**断在哪**,不能只钉「拆成两组」——写这条判据时我先只写了组数与位移,
        //    而变异(换成 `slot_singable`)**照绿**:它把断点从 `[1|2]` 推到了 `[2|3]`,
        //    组数仍是 2、`gb[1].shift` 仍是 −5。**「拆了」和「断在该断的地方」是两件事。**
        assert_eq!(
            (gb[0].start, gb[0].end, gb[1].start, gb[1].end),
            (1, 1, 2, 3),
            "邻音落在 usable 与 scan 之间时不许挡住 `[1|2]` 这一处(读到 {gb:?})"
        );
        assert_eq!(
            85 + gb[1].shift,
            80,
            "夹具有效性:被测的那个邻音必须正好落在 80(读到 {})",
            85 + gb[1].shift
        );
        // ⭐⭐⭐ S163 —— 谓词之争也迁到窗层:浅组的护栏**必须**伸进 85,因为 85 在它的位移上
        // 落到 **80** —— `slot_reachable` 判**能**、`slot_singable` 判**不能**。
        // ⇒ 把 [`dead_group_windows_raw`] 里的 `slot_reachable` 换成 `slot_singable`,这里当场红。

        // ⑴ 阴性对照 A —— 便宜的那一刀不许拆,而且**门就在 2000 这个位置**。
        // ⚠ S159zi:门从 6000 降到 3000 之后,原来那个 `[83, 81, 81]`(差 2 度 = 4000)
        //    **够得着了** ⇒ 阴性对照会当场失效。
        //    ⛔ 正确的改法是把夹具挪到新门限的下方,而不是把期望值改成「拆」——
        //    这条对照要证的是「门限真的在起作用」,改期望值就把它证没了。
        // ⚠⚠ S163 —— 门 3000 → 2000(重新定价见 [`SPLIT_MIN_COST_DEFAULT`])之后,
        //    `[82, 81, 81] × 2000 ms`(差 1 度 × 2000 ms = **2000 = 门限**)又够得着了。
        //    ⇒ 照同一条先例把夹具挪到门下(2000 → 1000 ms),
        //    ⭐ **并且补一条正对照**(3200 ms ⇒ 3200 > 3000 ⇒ 拆):这样它钉的不再只是
        //    「门存在」,而是**门就在 3000**。深度差已经不能再小(1 度),所以动的是时长。
        // ⚠⚠ S163 后半场:门 2000 那一版**当天就被回滚了**(定价漏了接缝密度,见
        //    [`SPLIT_MIN_COST_DEFAULT`] 的 doc),两格的时长跟着改回 3000 那一侧。
        let shallow = [0i64, 82, 81, 81, 0];
        let gs = plan(&shallow, &[50, 50, 25, 25, 50]);
        assert_eq!(
            gs,
            vec![DeadGroup { start: 1, end: 3, shift: -3 }],
            "1 度 × 1000 ms = 1000 < 3000 ⇒ 不值一条新缝,不许拆(读到 {gs:?})"
        );
        let gs2 = plan(&shallow, &[50, 50, 80, 80, 50]);
        assert_eq!(
            gs2.len(),
            2,
            "而 1 度 × 3200 ms = 3200 > 3000 ⇒ 必须拆 —— 否则上一格测的是「永不拆」(读到 {gs2:?})"
        );

        // ⑵ 阴性对照 B —— **同一批音高**,只把浅的那一侧改短 ⇒ 判据真的是 **ms·半音**,
        // 而不是偷偷退化成「只看深度差」。
        // ⚠ S163:门 3000 → 2000 之后,原来的 100 ms 那一档(2 × 100 × 11 = **2200**)够得着了
        //    ⇒ 照 ⑴ 同一条先例把夹具挪到门下:**80 ms** ⇒ 2 × 80 × 11 = **1760 < 2000**。
        let short = plan(&deep, &[50, 50, 4, 4, 50]);
        assert_eq!(
            short,
            vec![DeadGroup { start: 1, end: 3, shift: -13 }],
            "判据是 ms·半音:同样 11 度的差,只有 200 ms 就不许拆(读到 {short:?})"
        );

        // ⑸ 断点必须落在**深度真的掉下去**的那一处,不是第一处可断的地方。
        // ⚠ S159zk 时被可行性规则推后了一格(`[2|3]` ⇒ `[3|4]`),S163 收窄护栏之后推回来了:
        // ⭐ 旧 `[1..3]@−13` 让**一个** 81 落到 68,新 `[1..2]@−13` + `[3..4]@−2` 让**两个**都落到 79。
        // ⭐ 这条判据要证的事一个字没变:断点**不是** `[1|2]`(那一处深度差是 0,gain 也是 0)。
        let four = [0i64, 92, 92, 81, 81, 0];
        let g4 = plan(&four, &secs(four.len()));
        assert_eq!(
            g4,
            vec![
                DeadGroup { start: 1, end: 3, shift: -13 },
                DeadGroup { start: 4, end: 4, shift: -2 },
            ],
            "断点要断在深度掉下去的那一处,而且陪绑的音一个都不许多(读到 {g4:?})"
        );
        assert_ne!(g4[0].end, 1, "断点不许落在 [1|2] —— 那一处两侧位移相同,gain = 0");

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

    /// S160h —— **电平闸只动那个需要它的模型。**
    ///
    /// 承重的那句话是「三个已验收模型(东雪莲 / akiko / yachiyo)一格不动」——
    /// 它不是论证,是**四份装机记录里的读数**,所以这里把那四份的顶几格逐字钉下来。
    /// ⛔ 数字来源:各自 sidecar 的 `vocal_range.speakers.0.semitones`(2026-08-24 实测)。
    #[test]
    fn the_level_gate_only_moves_the_model_that_needs_it() {
        // (名字, usable, 顶往下 5 格的 rms_db, **底往上 5 格的 rms_db**, 期望的新顶)
        // ⛔⛔ 底部那五个数是**真实读数**,不是编的 —— 第一版给的是健康的 −1.0,
        //   于是「底不该动」那条断言结构上够不着「两端都收」这个坏法(实机上把 yuyuko
        //   收成了 `[60,79]`)。**夹具不真实的判据 = 空判据。**
        let cases: &[(&str, i64, i64, [f64; 5], [f64; 5], i64)] = &[
            ("dxl", 36, 79, [-0.4, -1.9, -1.6, -2.4, -6.1], [-5.9, -7.2, -7.7, -6.5, -7.0], 79),
            ("akiko", 36, 76, [-0.7, -0.5, 0.0, -2.9, -3.0], [-19.5, -22.9, -19.8, -20.8, -19.7], 76),
            ("yachiyo", 36, 77, [-2.8, -1.2, 0.0, -2.3, -1.5], [-9.4, -11.5, -9.2, -8.9, -9.7], 77),
            ("yuyuko", 36, 82, [-2.9, 0.0, -9.9, -13.7, -16.6], [-16.0, -14.4, -15.4, -13.7, -12.6], 79),
        ];
        for &(name, lo, hi, rms, bot, want) in cases {
            let mut semis = serde_json::Map::new();
            for m in lo..=hi {
                let db = if m >= hi - 4 {
                    rms[(m - (hi - 4)) as usize]
                } else if m <= lo + 4 {
                    bot[(m - lo) as usize]
                } else {
                    -1.0
                };
                semis.insert(m.to_string(), serde_json::json!([1, 1.0, db, 0.10]));
            }
            let got = narrow_usable_by_level(
                Some(&semis),
                (lo as f32, hi as f32),
                RESCUE_LEVEL_FLOOR_DB,
            );
            assert_eq!(got.1 as i64, want, "{name}: usable 顶应当是 {want},实际 {}", got.1);
            // ⭐ 承重的那一半:**底一格都不许动**,即使它比该模型自己最响的一格低 20 dB。
            assert_eq!(got.0 as i64, lo, "{name}: 底不该动(实际 {})", got.0);
        }
    }

    /// S160h —— 电平闸的三条边界:老记录不动 · 没有读数的格放过 · 不许把音域收成空的。
    #[test]
    fn the_level_gate_is_inert_without_a_reading_and_never_empties_the_range() {
        // ⑴ 没有 semitones ⇒ 一个字不动(pre-S81 的老记录)
        assert_eq!(narrow_usable_by_level(None, (36.0, 82.0), -8.0), (36.0, 82.0));

        // ⑵ 2 元组(没有 rms_db)⇒ 放过 —— **没有读数不等于坏**
        let mut two = serde_json::Map::new();
        for m in 36..=82i64 {
            two.insert(m.to_string(), serde_json::json!([1, 1.0]));
        }
        assert_eq!(narrow_usable_by_level(Some(&two), (36.0, 82.0), -8.0), (36.0, 82.0));

        // ⑶ 整条音域都在闸下 ⇒ 原样返回(那说明扫描坏了,不是音域没了)
        let mut all_quiet = serde_json::Map::new();
        for m in 36..=82i64 {
            all_quiet.insert(m.to_string(), serde_json::json!([1, 1.0, -30.0, 0.10]));
        }
        assert_eq!(narrow_usable_by_level(Some(&all_quiet), (36.0, 82.0), -8.0), (36.0, 82.0));

        // ⑷ ⛔ **底【不】会被收** —— 见 `narrow_usable_by_level` 里那段血训:
        //    低音区安静是音乐上正常的,两端都收会毁掉四个装机记录里的三个。
        let mut low_bad = serde_json::Map::new();
        for m in 36..=82i64 {
            let db = if m <= 38 { -20.0 } else { -1.0 };
            low_bad.insert(m.to_string(), serde_json::json!([1, 1.0, db, 0.10]));
        }
        assert_eq!(narrow_usable_by_level(Some(&low_bad), (36.0, 82.0), -8.0), (36.0, 82.0));
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
            "trim={:?} landing={:?} ratio2={} depth={} frac={} win={} xgrain={} lpc={}              hp={} hp_ms={} envfix={} bridge={} lock={} kappa={} join={} wininv={} sliver={} tiethin={} tilt={} pick={} harm={} repair={} comb={} handover={} tiedxf={} split={} interior={} xdith={} xslide={} tiedst={} width={} wfloor={} tiltfade={}/{}              usag={} usagdim={} gonesort={} dipfill={} restwin={}/{} h2={} mism={}",
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
            // S162 —— 薄片闸。⚠ 与上一条同款:进指纹但**出厂 0 ⇒ 不改音频**,
            //    所以加它**不该**触发版本 bump;进指纹的意义是「下一个人翻它必须来这里改一行」。
            parse_close_sliver(None),
            // S162 —— 同款:进指纹但**出厂 = 今天** ⇒ 不该触发版本 bump。
            parse_landing_tie_thin(None),
            // S162 —— 谱倾斜还原。⛔ 这一个**改音频**(出厂 1.0),所以它进指纹的同时
            //    `RANGE_ALGO_VERSION` 与 audition cache tag 都跟着 bump 到 `s162b`。
            range_tilt(),
            // S162 —— 渲染时按实测选落点。⛔ **改音频**(出厂开)⇒ 与它一起 bump 到 `s162c`。
            LANDING_PICK_DEFAULT,
            // S163 —— 落点选法的**谐波否决**门限。⛔ **改音频**(出厂 3.0)⇒ bump 到 `s163a`。
            //   ⚠ 同一次 bump 还盖着两条**没有旋钮**的行为改动(指纹结构上看不见它们,
            //   所以在这里写明白):①落点候选必须与它服务的那一组**覆盖同一批音**;
            //   ②打分从「整窗一个 RMS」改成**逐音、取组内最差**,参照换成
            //   「邻近没被救的唱音的中位」(= `match_rescued_note_levels` 那一套)。
            parse_landing_harm(None),
            // S163 —— **修补遍**的门限(ms)。⛔ **改音频**(出厂 200)⇒ bump 到 `s163b`。
            parse_landing_repair(None),
            // S163 —— 梳深地板。⛔ **改音频**(出厂 6)⇒ bump 到 `s163d`。
            parse_comb_floor(None),
            // S163 —— **交接点体检**。⛔ **改音频**(出厂 15)⇒ bump 到 `s163c`。
            parse_handover(None),
            // S163 —— 长音延续处的交叉淡化。⛔ **改音频**(出厂 120 ms)⇒ 与 handover 一起 bump。
            parse_tied_xfade(None),
            // ⛔⛔ S163 —— **补一个漏掉的**:拆簇的两条门**改音频**,却一个都不在指纹里
            //    ⇒ S159zi 当年把 `SPLIT_MIN_COST_DEFAULT` 从 6000 改到 3000 时,
            //    这条闸**结构上不可能**逼出成对 bump —— 那一次用户很可能听过一条陈缓存。
            //    (与 S160q「一个指纹、一条闸、一个版本号」同款的漏洞,同款的修法。)
            SPLIT_MIN_COST_DEFAULT,
            SPLIT_MIN_INTERIOR_NOTES,
            // S163 —— PSOLA 颗粒的两个新自由度(见 psola.rs 的 `xdither` / `xslide`)。
            // ⚠ 两个**出厂都是 0 = 逐位不变** ⇒ 加它们**不该**触发版本 bump;
            //   进指纹的意义是「下一个人翻它的时候必须来这里改一行,于是不得不读那段 doc」。
            std::env::var("UTAI_PSOLA_XDITHER").unwrap_or_else(|_| "0".into()),
            std::env::var("UTAI_PSOLA_XSLIDE").unwrap_or_else(|_| "0".into()),
            // S163 —— `tied` 的音高容差。⛔ **改音频**(同歌词但音高跳 >2 半音不再算延续音)
            //    ⇒ 与它一起 bump 到 `s163g`。
            2,
            // S163 —— 谱峰宽度否决。⚠ **出厂 0 = 逐位不变** ⇒ 加它**不该**触发版本 bump;
            //    进指纹的意义是「下一个人翻它的时候必须来这里改一行」。
            parse_landing_width(None),
            parse_landing_width_floor(None),
            // ⭐⭐ S163 —— tilt 在高音上的淡出区间。⛔ **改音频**（实测 MIDI 90 上
            //    tilt 本来把上方谐波压 −4.01 dB，现在归零）⇒ 与它一起 bump。
            85,
            90,
            // ⭐⭐⭐⭐ S163 §25 —— usag 排序主键（出厂 0 = 关）与它的对手轴闸。
            parse_usag_eps(None),
            parse_usag_dim(None),
            // ⭐⭐⭐ S163 §26 —— 静音次键的 eps。⛔ **无条件生效 ⇒ 改音频** ⇒ 必须在指纹里。
            GONE_SORT_EPS_MS,
            // ⭐ S163 §34 —— donor 静音坑回填（出厂 0 = 关）。
            parse_dipfill(None),
            // ⭐⭐⭐ S163 —— 窗边伸进休止的上限。⛔ **改音频**（`REST_POST_FRAMES` 2→4）
            //    ⇒ 必须在指纹里；这两个原本是**裸字面量**，改它们结构上逼不出成对 bump
            //    （与 SPLIT_MIN_COST 那次同款的洞、同款的修法）。
            REST_PRE_FRAMES,
            REST_POST_FRAMES,
            // ⭐⭐⭐ S165 —— `2·f0` 电平当排序键的 eps。
            //    ⚠ **出厂 0 = 关 = 逐位不变** ⇒ 加它**不该**触发版本 bump
            //    (与 `usag=` / `width=` / `donorin=` 同例);
            //    进指纹的意义是「下一个人想把它变成默认之前,必须先来这里改一行,
            //    于是不得不读 `landing_h2_eps` 那段 doc 与它的三个常量
            //    (`H2_REPAIR_FLOOR` / `H2_REPAIR_RADIUS` / `H2_DIM_CAP`)」。
            parse_h2_eps(None),
            // ⭐⭐⭐ S165 —— **失配**轴的 eps。⚠ 出厂 0 = 关 = 逐位不变 ⇒ 加它**不该**触发版本 bump;
            //    进指纹的意义是「下一个人想把它变成默认之前,必须先来这里改一行,
            //    于是不得不读 `landing_mismatch_eps` 那段 doc —— 特别是**相对判据 + 最差口径**那两条」。
            parse_mismatch_eps(None),
        );
        // ⛔⛔ S160q —— 这条闸此前**只看得见本文件**,而 `score2svc.rs` 里有七个会改音频的
        //    旋钮(含出厂就开着的 `FILL_ISOLATED_UV_DEFAULT`)一个都不在指纹里,
        //    它头上那行注释还正好在教人做成对 bump —— 没有闸执行。
        //    ⇒ 一个指纹、一条闸、一个版本号。
        let fp = format!("{fp} | {}", super::super::score2svc::production_defaults_fingerprint());
        assert_eq!(
            fp,
            // ⭐ S165 —— `donorin=` 是外部逆变换臂（`UTAI_RANGE_DONOR_IN`，整曲耳判用）。
            //    它**会改音频**（所以在指纹里、不在 `EXEMPT` 里），但**出厂没设 ⇒ off ⇒ 不改音频**
            //    ⇒ 按本判据自己的规矩（“若它出厂关 = 不改音频，就不要 bump 版本号”）**不跟着 bump**，
            //    与 `valhuman=` 当初同例。进指纹的意义是：下一个人想把它变成默认之前，必须先来这里改一行。
            "trim=Some((500.0, 500.0)) landing=Some(3) ratio2=14 depth=1 frac=true win=1 xgrain=1 lpc=0              hp=true hp_ms=0 envfix=0 bridge=120 lock=0.3 kappa=0 join=false wininv=true sliver=0 tiethin=true tilt=1 pick=true harm=3 repair=200 comb=6 handover=15 tiedxf=120 split=3000 interior=3 xdith=0 xslide=0 tiedst=2 width=0 wfloor=0 tiltfade=85/90              usag=3 usagdim=3 gonesort=15 dipfill=0 restwin=4/4 h2=0 mism=1.2 | f0lerp=true fill1=true filluv=true fillmax=1 uvgate=true uvgatek=1.5 uvgateguard=20 valadapt=false valafter=false valhuman=true restshrink=true predamp=true/40,-40,0.6,2,5,35 restbucket=true donorin=false valdb=1.1/12,15,17/6.5,9 valenv=0.96,0.08/0.98,0.02",
            "⛔ 生产默认变了。必须同时改三处:①这条判据里的指纹              ②`src/lib/vocal/vocalRender.ts` 的 `RANGE_ALGO_VERSION`              ③`src-tauri/src/commands/audition.rs` 的 `_sNNNx_` cache tag ——              漏掉后两个不是错误,是用户听到一条陈缓存(S150)。"
        );
        // ⛔ S163e 盖着的:①`SPLIT_MIN_COST_DEFAULT` 3000 → 2000;
        //    ②-④ 三条**没有旋钮**的行为改动(指纹看不见,写在这里):
        //    ② 护栏不许伸进一个在**本遍**唱不动的音([`dead_group_windows_raw`]);
        //    ③ 拆组的断点可行性(S159zk)从**否决断点**改成**不否决**(危害载体已由 ② 收走);
        //    ④ 窗尾落在长音内部时淡出拉长([`TAIL_ROOM_MS`]);
        //    ⑤ 修补遍的 ±1 绕**所有**候选而不只是选中的那个。
        // ⛔ S163f —— s163e 的**两条行为改动当天被回滚**(`neighbour_ok` 恢复否决 ·
        //    `split_cost` 回 3000,账见 [`SPLIT_MIN_COST_DEFAULT`] 的 doc)。
        //    ⚠ 必须再 bump 一次:s163e 已经渲过一批产物,**同一个 tag 不许对应两种行为**。
        // ⛔ S163h 盖着：**tilt 在高音 target 上淡出**（`TILT_FADE_LO/HI` = 85/90）。
        //    实测：tilt 对上方谐波的 Δ 随 target 音高**单调递减**，在 87 穿零：
        //    yuyuko 68 +9.15 / 71 +7.84 / 75 +5.31 / 78 +3.65 / 80 +2.91 / 82,83 +2.08 /
        //    **87 −0.84** / **90 −4.01**；akiko のぴゃ（MIDI 90）独立读 **−3.05**。
        //    修后：低音侧 68-83 **逐字不变**，87 的损害减半、90 归零。
        const TAG: &str = "s165j";
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
            ("SEAM_ALIGN_MS_DEFAULT", "pub fn seam_align_ms("),
            ("ENV_RESTORE_MS_DEFAULT", "pub fn env_restore_ms("),
            ("BRIDGE_UNVOICED_MS_DEFAULT", "pub fn bridge_unvoiced_ms("),
            ("BRIDGE_VALLEY_DEFAULT", "pub fn bridge_valley("),
            ("WINDOWED_INVERSE_DEFAULT", "pub fn windowed_inverse("),
            ("FORMANT_KNEE_DEFAULT", "pub fn formant_knee("),
            ("DEJITTER_DEFAULT", "pub fn dejitter("),
            ("CLOSE_SLIVER_FRAMES_DEFAULT", "fn parse_close_sliver("),
            ("RANGE_TILT_DEFAULT", "pub fn range_tilt("),
            ("LANDING_PICK_DEFAULT", "pub fn landing_pick("),
            ("LANDING_HARM_EPS_DEFAULT", "pub fn landing_harm_eps("),
            ("LANDING_REPAIR_MS_DEFAULT", "pub fn landing_repair_ms("),
            ("COMB_FLOOR_DB_DEFAULT", "pub fn comb_floor_db("),
            ("LANDING_WIDTH_EPS_DEFAULT", "pub fn landing_width_eps("),
            ("LANDING_WIDTH_FLOOR_DEFAULT", "pub fn landing_width_floor("),
            ("HANDOVER_DEFICIT_DB_DEFAULT", "pub fn handover_deficit_db("),
            ("HANDOVER_GAIN_DB_DEFAULT", "pub fn handover_gain_db("),
            ("TIED_XFADE_MS_DEFAULT", "pub fn tied_xfade_ms("),
            ("USAG_EPS_DEFAULT", "pub fn landing_usag_eps("),
            ("USAG_DIM_CAP_DEFAULT", "pub fn landing_usag_dim_cap("),
            ("DIPFILL_DEPTH_DEFAULT", "pub fn dipfill_depth_db("),
            ("LANDING_H2_EPS_DEFAULT", "pub fn landing_h2_eps("),
            ("LANDING_MISMATCH_EPS_DEFAULT", "pub fn landing_mismatch_eps("),
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

    /// ⛔⛔ S162 —— **谱倾斜只走谱面轨;cover 那两条车道传 `0.0`。**
    ///
    /// ## 为什么要一条源码级的闸
    /// `apply_inverse*` 是**两条车道共用的单一执行点**(S85 为这条规则付过一夜)。
    /// tilt 的表是在**谱面轨**素材上拟的,而 **cover 上一个读数都没有** ——
    /// 而且 cover 的深救援反而更重(S160 的计划输出:`|s| ≥ 8` 占救援总时长 **78.1%**,
    /// 最深 **−18**,已超出表的范围)。
    /// ⇒ 下一个人重构时把 cover 那两处顺手改成 `range_tilt()`,**不会有任何行为判据变红**,
    ///   而那是一次静默退化。所以这条闸盯的是**源码本身**。
    ///
    /// ⛔ 按**行**取,不许按字节切 —— 这几个文件里全是中文 doc。
    #[test]
    fn the_spectral_tilt_reaches_the_score_lane_and_never_the_cover_lane() {
        let want_zero = [
            ("rvc.rs", include_str!("rvc.rs")),
            ("sovits.rs", include_str!("sovits.rs")),
        ];
        for (name, src) in want_zero {
            let lines: Vec<&str> = src.lines().collect();
            let at = lines
                .iter()
                .position(|l| l.contains("vocal_range::apply_inverse("))
                .unwrap_or_else(|| panic!("{name} 里找不到 apply_inverse 调用 —— 这条闸已经瞎了"));
            // 调用点之后 12 行内必须出现独立的 `0.0,`(tilt 实参)
            let win = lines[at..(at + 12).min(lines.len())].join("\n");
            assert!(
                win.contains("0.0,"),
                "{name} 的 cover 调用点没有把 tilt 传 0 —— \n\
                 tilt 的表只在谱面轨素材上拟过,cover 上一个读数都没有(而它的深救援更重:\n\
                 |s|≥8 占救援总时长 78.1%,最深 −18 已超出表的范围)。\n\
                 要给 cover 开,先在 cover 的 donor 转储上拟/验一张表。\n实际:\n{win}"
            );
            assert!(
                !win.contains("range_tilt()"),
                "{name} 的 cover 调用点被接到了 `range_tilt()` 上 —— 见上一条的理由"
            );
        }
        // ⛔ 阴性对照:谱面轨那一条**必须**接着 `range_tilt()`,
        //    否则上面两条在「tilt 根本没人用」的实现上照样全绿。
        let score = include_str!("score2svc.rs");
        let lines: Vec<&str> = score.lines().collect();
        let at = lines
            .iter()
            .position(|l| l.contains("vocal_range::apply_inverse_windowed("))
            .expect("score2svc.rs 里找不到 apply_inverse_windowed 调用");
        let win = lines[at..(at + 14).min(lines.len())].join("\n");
        assert!(
            win.contains("range_tilt()"),
            "谱面轨那一条没接 `range_tilt()` —— 那这把刀出厂就是空的。实际:\n{win}"
        );
    }

    /// ⭐⭐ S162 —— **乐句内跨组的电平对齐:只去掉每段一个标量,段内起伏一个字不动。**
    ///
    /// ## ⛔ 它治的是什么
    /// 用户 2026-08-26 整曲耳判:同一乐句里 `[791]だ`(shift −12)比 `[794]あ`(shift −5)
    /// **轻 2.7 dB**,而这条轴的可闻刻度就是「~2.7 dB 听得出」、逐音电平的渲染噪声底只有 0.74。
    /// ⛔ 而 [`match_rescued_note_levels`] 对它**结构上失效**:那把刀要 ≥4 个**没被救**的邻居,
    /// 而密集救援的乐句正好一个都没有(实测那两个音周围 ±16 里只有 2 个)。
    ///
    /// ## 这条钉四件
    /// ⑴ 跨组的整体偏置被拉平;⑵ ⛔ **段内的相对起伏逐位不变**(否则它就不是「去偏置」);
    /// ⑶ ⛔ **阴性对照**:乐句里只有一段时**逐位不变**;⑷ ⛔ **限幅**:大偏差只给到上限。
    #[test]
    fn the_phrase_level_alignment_only_removes_the_cross_group_offset() {
        let sr = 44100u32;
        let nf = 20i64; // 每个音 20 帧
        let total = 10 * nf;
        let spf = 441usize; // 每帧 441 样本 ⇒ 每个音 8820 样本
        let n = (total as usize) * spf;
        // 音表:[休止] + 8 个唱音 + [休止]
        let mut notes: Vec<NoteSpan> = Vec::new();
        for i in 0..10i64 {
            let sung = !(i == 0 || i == 9);
            notes.push(NoteSpan {
                start: i * nf,
                frames: nf,
                sung,
                hz: if sung { 440.0 } else { 0.0 },
                tied: false,
            });
        }
        // 前 4 个唱音 = shift −4,后 4 个 = shift −12
        let jobs = vec![
            DeadJob { shift: -4, start: nf, end: 5 * nf },
            DeadJob { shift: -12, start: 5 * nf, end: 9 * nf },
        ];
        // 素材:段 A 基准 1.0,段 B 基准 0.5(= −6.02 dB);⭐ 两段内部都带同样的起伏
        let ripple = [1.0f32, 1.30, 0.80, 1.10];
        let build = |gain_b: f32| -> Vec<f32> {
            let mut x = vec![0.0f32; n];
            for i in 1..9usize {
                let base = if i < 5 { 1.0 } else { gain_b };
                let amp = base * ripple[(i - 1) % 4];
                let (a, b) = ((i as i64 * nf) as usize * spf, ((i as i64 + 1) * nf) as usize * spf);
                for (t, v) in x[a..b].iter_mut().enumerate() {
                    *v = amp * ((t as f32) * 0.05).sin();
                }
            }
            x
        };
        let rms = |x: &[f32], i: usize| -> f32 {
            let (a, b) = ((i as i64 * nf) as usize * spf, ((i as i64 + 1) * nf) as usize * spf);
            // ⛔ 掐掉两端 —— 施加时有 10 ms 淡化,量稳态才读得到那个标量
            let pad = sr as usize / 20;
            let s = &x[a + pad..b - pad];
            10.0 * (s.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>()
                / s.len() as f64
                + 1e-20)
                .log10() as f32
        };

        // ⑴ + ⑵ —— 1.5 dB 的偏差(在限幅之内)
        let g = 10f32.powf(-1.5 / 20.0);
        let mut y = build(g);
        let before = rms(&y, 6) - rms(&y, 2);
        let hit = match_phrase_group_levels(&mut y, sr, total, &jobs, &notes, 3.0, 2.0);
        assert_eq!(hit, 1, "该对齐一段");
        let after = rms(&y, 6) - rms(&y, 2);
        assert!(
            before < -1.0 && after.abs() < 0.35,
            "跨组偏置该被拉平:{before:.2} dB → {after:.2} dB"
        );
        // ⛔ 段内的相对起伏必须一个字不动
        let within = rms(&y, 6) - rms(&y, 5);
        let base_x = build(g);
        let want = rms(&base_x, 6) - rms(&base_x, 5);
        assert!(
            (within - want).abs() < 0.02,
            "段内起伏被改了:{within:.3} vs {want:.3} —— 这把刀只许去掉每段一个标量"
        );

        // ⑶ ⛔ 阴性对照:乐句里只有一段 ⇒ 逐位不变
        let one = vec![DeadJob { shift: -4, start: nf, end: 9 * nf }];
        let mut z = build(g);
        let z0 = z.clone();
        let h2 = match_phrase_group_levels(&mut z, sr, total, &one, &notes, 3.0, 2.0);
        assert_eq!(h2, 0, "只有一段时不该动手");
        assert_eq!(z, z0, "只有一段时必须逐位不变");

        // ⑷ ⛔ 限幅:10 dB 的偏差只许给到 +2
        let mut w = build(10f32.powf(-10.0 / 20.0));
        let b0 = rms(&w, 6) - rms(&w, 2);
        match_phrase_group_levels(&mut w, sr, total, &jobs, &notes, 3.0, 2.0);
        let b1 = rms(&w, 6) - rms(&w, 2);
        assert!(
            (b1 - b0 - 2.0).abs() < 0.25,
            "抬的上限是 +2 dB,实际抬了 {:.2}(⛔ 抬会把面状伪影一起抬)",
            b1 - b0
        );
    }

    /// ⭐⭐ S162 —— **被救音的电平匹配:只压不抬 · 只碰被救的音 · 参照只用没被救的邻居。**
    ///
    /// ⛔ 没有这条判据,`match_rescued_note_levels` 就只有 doc 和指纹盯着它:
    /// 把「只压」写成双向、把参照改成含被救邻居、把 cov 门去掉、或者把软膝写成硬压,
    /// **别的每一条测试都还是绿的**。而「抬」那一半是**实测会把面状伪影抬起来**的
    /// (被抬的音次基频占比高 +6.4…+17.5 dB),所以「不抬」是承重的,不是风格。
    #[test]
    fn the_level_match_only_pushes_loud_rescued_notes_down() {
        let sr = 48000u32;
        let hop = 480usize; // 每帧 480 样本 ⇒ total_frames * hop = len
        let nf = 100i64;
        let n = (nf as usize) * hop;
        // 音表:每 10 帧一个音(⛔ 必须 ≥ `LEVEL_MATCH_MIN_FRAMES`,否则整表被跳过 ——
        //   第一版写成 4 帧,判据当场红,红对了)。
        let notes: Vec<NoteSpan> =
            (0..10).map(|k| NoteSpan { start: k * 10, frames: 10, sung: true, hz: 440.0, tied: false }).collect();
        // 窗:只盖第 5 个音(帧 50..60)⇒ 只有它是「被救」的
        let jobs = vec![DeadJob { shift: -7, start: 50, end: 60 }];
        let mk = |amp_rescued: f32| -> Vec<f32> {
            let mut v = vec![0.0f32; n];
            for (i, nd) in notes.iter().enumerate() {
                let (f0, fr) = (nd.start, nd.frames);
                let a = (f0 as usize) * hop;
                let b = ((f0 + fr) as usize) * hop;
                let amp = if i == 5 { amp_rescued } else { 0.1 };
                for t in a..b {
                    v[t] = amp * ((t as f32) * 0.05).sin();
                }
            }
            v
        };

        // ⓐ 被救音**比邻居响 20 dB** ⇒ 必须被压
        let mut loud = mk(1.0);
        let before = loud.clone();
        let hit = match_rescued_note_levels(&mut loud, sr, nf, &jobs, &notes, 6.0);
        assert_eq!(hit, 1, "响 20 dB 的被救音必须被碰");
        let seg = |v: &[f32], i: usize| -> f32 {
            let (f0, fr) = (notes[i].start, notes[i].frames);
            let (a, b) = ((f0 as usize) * hop, ((f0 + fr) as usize) * hop);
            (v[a..b].iter().map(|x| x * x).sum::<f32>() / (b - a) as f32).sqrt()
        };
        let drop_db = 20.0 * (seg(&loud, 5) / seg(&before, 5)).log10();
        assert!(drop_db < -5.0, "该被压下来,实际只动了 {drop_db:.1} dB");
        // ⛔ 没被救的音**一个都不许动**
        for i in 0..notes.len() {
            if i == 5 {
                continue;
            }
            let (f0, fr) = (notes[i].start, notes[i].frames);
            for t in (f0 as usize) * hop..((f0 + fr) as usize) * hop {
                assert_eq!(loud[t], before[t], "没被救的音 [{i}] 第 {t} 个样本被动了");
            }
        }

        // ⓑ ⛔ 被救音**比邻居轻 20 dB** ⇒ **一个字不许动**(抬会把面状伪影一起抬起来)
        let mut quiet = mk(0.01);
        let q0 = quiet.clone();
        let hit2 = match_rescued_note_levels(&mut quiet, sr, nf, &jobs, &notes, 6.0);
        assert_eq!(hit2, 0, "只压不抬 —— 轻的被救音不许被碰");
        assert_eq!(quiet, q0, "轻的那一侧必须逐样本不变");

        // ⓒ 软膝:比邻居响 4 dB(< T=6)⇒ 不动
        let mut mild = mk(0.1 * 10f32.powf(4.0 / 20.0));
        let m0 = mild.clone();
        assert_eq!(match_rescued_note_levels(&mut mild, sr, nf, &jobs, &notes, 6.0), 0,
                   "T 以内不许动");
        assert_eq!(mild, m0);

        // ⓓ 门 = 0 ⇒ 整条关掉,逐样本不变
        let mut off = mk(1.0);
        let o0 = off.clone();
        assert_eq!(match_rescued_note_levels(&mut off, sr, nf, &jobs, &notes, 0.0), 0);
        assert_eq!(off, o0, "thresh = 0 必须是逐样本 no-op");

        // ⓔ ⛔⛔ **一个【没被救】但比邻居响 20 dB 的音,一个字都不许碰。**
        //    ⚠ 这一格是**变异抓出来的**:上一版夹具里没被救的音电平全一样,于是
        //    「去掉 cov 门」这条变异**照绿** —— 判据在那条轴上是空的。
        let mut loud_clean = {
            let mut v = vec![0.0f32; n];
            for (i, nd) in notes.iter().enumerate() {
                let (f0, fr) = (nd.start, nd.frames);
                let a = (f0 as usize) * hop;
                let b = ((f0 + fr) as usize) * hop;
                // 音 2 **没被窗盖**,却比邻居响 20 dB;音 5 被盖但电平正常
                let amp = if i == 2 { 1.0 } else { 0.1 };
                for t in a..b {
                    v[t] = amp * ((t as f32) * 0.05).sin();
                }
            }
            v
        };
        let lc0 = loud_clean.clone();
        assert_eq!(
            match_rescued_note_levels(&mut loud_clean, sr, nf, &jobs, &notes, 6.0),
            0,
            "没被救的音再响也不许碰 —— 那是乐曲本身的强弱"
        );
        assert_eq!(loud_clean, lc0, "没被救的响音必须逐样本不变");

        // ⓕ ⛔ 干净邻居不够 ⇒ 不动(把所有音都盖进窗里)
        let all = vec![DeadJob { shift: -7, start: 0, end: 100 }];
        let mut none_ref = mk(1.0);
        let r0 = none_ref.clone();
        assert_eq!(match_rescued_note_levels(&mut none_ref, sr, nf, &all, &notes, 6.0), 0,
                   "没有干净邻居当参照时必须放弃,而不是拿被救的邻居凑");
        assert_eq!(none_ref, r0);
    }

    /// ⛔⛔⭐ S163 —— **落点候选必须与它服务的那一组【覆盖同一批音】。**
    ///
    /// ## 它治的是什么(实测,不是假想)
    /// 候选 = 「窄预算(`landing = 1`)那份计划的落点」,按**组的起始音**取。
    /// 而两份计划的预算不同 ⇒ **分组也不同**:实测 akiko × 炉心 +7,
    /// `landing = 1` 出 **95 组**、`landing = 3` 出 **90 组**。
    /// 于是「起点相同」的两个组可能**根本不是同一批音**,它的落点是为**别的音**解出来的。
    ///
    /// 用户 2026-08-26 报的 **4:09.478「み」**就是这个:plan `[698..=704]` 的最高音
    /// 目标 MIDI **87**,需要 −10;而同起点的窄预算组是 `[698..=701]`(把那个高音分出去了)
    /// ⇒ 它的落点是 **−4**,落在 **83** —— **仍在死区**。把它当候选交给渲染层,
    /// 它在**整窗**电平上反而更贴近邻居 ⇒ 赢了 ⇒ 那个音的救援被整个丢掉。
    /// 同一份转储上实测:谐波能量占比 **−0.98 → −19.56 dB**、梳深 26.9 → 8.4、
    /// 次基频 −25.1 → **+19.9**,与**关掉扩展的 `base` 几乎逐格相同**;
    /// 而这在**同一首歌里发生 5 次**。人群面:4 模型 × 2 谱 × 移调 0/7 共 341 组带候选,
    /// **22 组(6.5%)**起点相同而范围不同(东雪莲 × 鹅妈妈 +7 是 6/7)。
    ///
    /// ## 这条钉三件
    /// ⑴ 夹具**真的**复现了「两个预算分组不同」(否则这条判据是空的);
    /// ⑵ 每一个带出来的候选,**都救得动这一组自己的死音**;
    /// ⑶ ⛔ **阴性对照 = 旧规则**:只认起点的那一版在同一份夹具上会挑出一个**救不动**的候选。
    #[test]
    fn a_landing_candidate_is_never_borrowed_from_a_differently_grouped_plan() {
        // ⚠⚠ S163 —— 换夹具(**不是**因为判据失效):全曲上「起点相同、范围不同」
        // 仍然 **26 / 359 = 7.2%**(改拆组之前是 22 / 341 = 6.5%),而原来那个 12 音截取
        // (`notes[697..=708]`)在新的拆组规则下两个预算**拆得一样了** ⇒ ⑴ 当场红。
        // ⇒ 从真实 plan dump 里裁一段**仍然复现**的:炉心融解 `notes[185..=192]`,akiko,+7。
        // ⭐ 它 +7 之后是 **82/83/92/90/88** —— 正是用户 2026-08-26 点名的
        //   「嗓子里卡着痰」那一组本身(同一句歌词在全曲重复五次)。
        // 实测:`landing=3` ⇒ `[1..=6] @−14` 一组;`landing=1` ⇒ `[(1,2,−5), (3,6,−14)]` 两组
        // ⇒ **起点 1 相同、范围不同**。
        let nn = [0i64, 73, 75, 73, 72, 68, 68, 80, 75, 75, 76, 0];
        let fr = [9i64, 9, 9, 9, 10, 9, 9, 9, 9, 18, 46, 9];
        let r = akiko_like();
        let today = RescueTuning::today();
        let (plan, _, alts) = dead_only_plan_with_alts(&nn, &fr, 7, &r, today);
        let (alt_plan, _, _) =
            dead_only_plan_with_alts(&nn, &fr, 7, &r, RescueTuning { landing: Some(1), ..today });
        eprintln!("[plan] {plan:?}\n[alt ] {alt_plan:?}\n[alts] {alts:?}");

        // ⑴ 前提:候选机制**是活的** —— 两个预算真的给出不同的落点,所以确实有东西可挑。
        //
        // ⚠⚠ S163 —— 这一格**曾经**断言「两个预算的**分组**不同」,而那是这个合成夹具
        // (`akiko_like()`,不是真实装机 sidecar)给不出来的:真实 akiko 上同一串音
        // `landing=3` 出 `[1..=6] @−14` **一组**、`landing=1` 出**两组**(起点相同、范围不同),
        // 而 `akiko_like()` 两个预算都拆成 `[1..2] + [3..6]`。
        // ⇒ 「这个场景会发生」的证据是**真实全曲的 26 / 359 = 7.2%**
        //   (S163 改拆组之后重新量;改之前 22 / 341 = 6.5%),一个合成夹具证不动它。
        // ⇒ 这一格只承担它证得动的那一半;「分组会不同 ⇒ 必须按**范围**而不是按**起点**认」
        //   由下面 ⑶ 的**实测两份计划**(`shipped` / `narrow`)独立钉住,而 ⑶ 正是这条规则的变异靶。
        let span = |p: &[DeadGroup]| p.iter().map(|g| (g.start, g.end)).collect::<Vec<_>>();
        assert_ne!(
            span(&plan),
            span(&alt_plan),
            "夹具没复现「两个预算分组不同」——那这条判据就是空的"
        );

        // ⑵ 每个候选都救得动**这一组自己**的死音
        let eff = |k: usize| (nn[k] + 7).clamp(1, 127);
        for (gi, g) in plan.iter().enumerate() {
            let Some(alt) = alts[gi] else { continue };
            for k in g.start..=g.end {
                let p = eff(k);
                assert!(
                    r.slot_reachable(p + alt),
                    "候选 {alt:+} 让 notes[{k}](MIDI {p})唱不出来"
                );
                if !r.slot_singable(p) {
                    assert!(
                        r.slot_landing_ok(p + alt),
                        "候选 {alt:+} 把 notes[{k}] 的死音(MIDI {p})落在 {} —— 那还是死的",
                        p + alt
                    );
                }
            }
        }

        // ⑶ ⛔⛔ **变异靶** —— 直接问 [`alt_shift_for`]:两份计划就是实测到的那两份
        //    (akiko × 炉心 +7,notes[698..=707];`landing=3` vs `landing=1`)。
        //    把 `x.end == g.end` 这一条去掉,这一格当场红。
        let shipped = DeadGroup { start: 698, end: 704, shift: -10 };
        let narrow = [
            DeadGroup { start: 698, end: 701, shift: -4 },
            DeadGroup { start: 704, end: 705, shift: -9 },
            DeadGroup { start: 706, end: 707, shift: -5 },
        ];
        assert!(
            narrow.iter().any(|x| x.start == shipped.start),
            "夹具没造对:旧规则(只认起点)必须**找得到**东西,否则这一格是空的"
        );
        assert_eq!(
            alt_shift_for(&shipped, &narrow),
            None,
            "范围不同的组不许当候选 —— 它的 −4 是为 [698..=701] 解出来的"
        );
        // 而那个落点对这一组的最高音(目标 MIDI 87)**是救不动的** —— 后果这一侧也钉住
        assert!(!r.slot_landing_ok(87 - 4), "83 仍在死区:这正是旧规则交出去的落点");
        assert!(r.slot_landing_ok(87 - 10), "−10 才落得下(实测出厂就是它)");
        // ⭐ 阴性对照的另一半:**范围一样**时候选照常带出来(否则这条规则等于关掉了候选)
        assert_eq!(
            alt_shift_for(&shipped, &[DeadGroup { start: 698, end: 704, shift: -12 }]),
            Some(-12),
            "同一批音、不同落点 ⇒ 必须带出候选"
        );
        assert_eq!(
            alt_shift_for(&shipped, &[DeadGroup { start: 698, end: 704, shift: -10 }]),
            None,
            "落点相同 ⇒ 没有候选可言"
        );
    }

    /// ⭐⭐⭐ S163 —— **窗尾落在一个长音的内部时,淡出必须拉长**(用户点名 akiko 2:50.4:
    /// 「本来是一个连续的音,结果突然一个接缝或者边界上去就造出了一个听感的断裂」)。
    ///
    /// ## 病灶
    /// `dead_group_windows` 的 `post = GUARD_FRAMES` 让窗尾伸进下一个唱音 **40 ms**,
    /// 然后 **10 ms** 就淡回 `base` ⇒ 一个 1.46 s 的音,头 40 ms 是 donor、之后是 base。
    /// ⛔ 收窄护栏已判负(S163:无休止跨组缝的重叠 80 → 0,瞬变中位差 **7.60 dB**)
    /// ⇒ 治的是「切得太快」,不是「伸进去」。
    /// ⭐ 安全性:淡出的另一侧是 `base`,而那个音**没被救 = 模型本来就唱得动它**
    ///   ⇒ 两侧都是「对的音高、正常的音」,慢慢过渡听不出;10 ms 切才是断裂。
    ///
    /// ## ⛔ 阴性对照(缺了它这条就是「淡出总是很长」)
    /// **同一个窗**,只把后面那个音改短到离窗尾不足 [`TAIL_ROOM_MS`] ⇒ 淡出必须退回 10 ms。
    #[test]
    fn the_window_tail_fades_out_slowly_when_it_lands_inside_a_long_note() {
        const SR: u32 = 44_100;
        const TF: i64 = 100; // 100 帧 × 20 ms = 2 s
        let n = SR as usize * 2;
        let spf = n as f64 / TF as f64;

        // 窗 [10, 40) 帧;它的尾巴落在**下一个音的内部**(那个音从帧 38 开始)。
        let jobs = vec![DeadJob { shift: -6, start: 10, end: 40 }];
        let span = |a: i64, f: i64| NoteSpan { start: a, frames: f, sung: true, hz: 440.0, tied: false };

        // donor 恒 1.0、base 恒 0.0 ⇒ 拼接后**处在 (0.02, 0.98) 之间的样本数**就是淡化宽度。
        let run = |notes: &[NoteSpan]| -> usize {
            let mut base = vec![0.0f32; n];
            apply_dead_only_windows_with(
                &mut base, SR, TF, &jobs, &[], notes, false, false, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                TIED_XFADE_MS_DEFAULT,
                |_s, own| {
                    let mut d = vec![0.0f32; n];
                    for &(a, b) in own {
                        let (a, b) = ((a.max(0) as f64 * spf) as usize, (b.max(0) as f64 * spf) as usize);
                        for v in d[a.min(n)..b.min(n)].iter_mut() {
                            *v = 1.0;
                        }
                    }
                    Ok(d)
                },
            )
            .expect("splice");
            // 只数窗**尾**那一侧:从窗尾往前找过渡区
            let b = ((40f64 * spf) as usize).min(n);
            let lo = b.saturating_sub(SR as usize / 2); // 往前最多看 500 ms
            base[lo..b].iter().filter(|v| **v > 0.02 && **v < 0.98).count()
        };

        // ⑴ 后面是一个 1.24 s 的长音(帧 38..100)⇒ 淡出拉长到 `TIED_XFADE_MS_DEFAULT`
        let long = run(&[span(0, 38), span(38, 62)]);
        // ⑵ 阴性对照:后面那个音只到帧 45(离窗尾 5 帧 = 100 ms < `TAIL_ROOM_MS`)⇒ 退回 10 ms
        let short = run(&[span(0, 38), span(38, 7), span(45, 55)]);

        let want = (SR as f64 * TIED_XFADE_MS_DEFAULT / 1000.0) as usize;
        let ten_ms = SR as usize / 100;
        assert!(
            long > want * 3 / 4,
            "窗尾落在长音内部时淡出必须拉长到 ~{want} 个样本(读到 {long})"
        );
        assert!(
            short < ten_ms * 3,
            "而后面那个音撑不满 TAIL_ROOM_MS 时必须退回 ~{ten_ms} 个样本(读到 {short})—— \
             否则上一格测的是「淡出总是很长」"
        );
        assert!(
            long > short * 4,
            "两档必须真的分得开(长 {long} vs 短 {short})"
        );
    }

    /// 判据用的合成台:12 个音 × 10 帧(每帧 100 ms),窗盖住音 4-6。
    /// 返回 (base, notes, jobs, alts, 每帧样本数)。
    #[cfg(test)]
    fn pick_rig() -> (Vec<f32>, Vec<NoteSpan>, Vec<DeadJob>, Vec<Option<i64>>, usize) {
        const SPF: usize = 4410; // 100 ms @ 44.1 kHz
        let nf = 10i64;
        let notes: Vec<NoteSpan> = (0..12i64)
            .map(|i| NoteSpan { start: i * nf, frames: nf, sung: true, hz: 440.0, tied: false })
            .collect();
        let jobs = vec![DeadJob { shift: -8, start: 4 * nf, end: 7 * nf }];
        let alts = vec![Some(-4i64)];
        // base:窗外的音都在同一个电平上(⇒ 干净邻居 9 个,参照取得到);窗内是「死」的。
        let n = 12 * (nf as usize) * SPF;
        let mut base = vec![0.0f32; n];
        for (i, nd) in notes.iter().enumerate() {
            let amp = if (4..7).contains(&i) { 0.003 } else { 0.1 };
            let (a, b) = ((nd.start as usize) * SPF, ((nd.start + nd.frames) as usize) * SPF);
            for (t, v) in base[a..b].iter_mut().enumerate() {
                *v = amp * (2.0 * std::f32::consts::PI * 440.0 * t as f32 / 44100.0).sin();
            }
        }
        (base, notes, jobs, alts, SPF)
    }

    /// 在 `notes[i]` 的正中央量电平(dBFS)——避开窗边那 10 ms 交叉淡化。
    #[cfg(test)]
    fn mid_db(x: &[f32], i: usize, spf: usize) -> f32 {
        let (a, b) = ((i * 10 + 3) * spf, (i * 10 + 7) * spf);
        let e: f64 =
            x[a..b].iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / (b - a) as f64;
        (10.0 * (e + 1e-30).log10()) as f32
    }

    /// ⛔⛔⭐ S163 —— **落点打分必须【逐音】,而且组的成绩 = 组内最差的那个音。**
    ///
    /// S162 那一版是**整窗一个 RMS**。一个窗里可能有 7 个音,而只有 1 个塌了:
    /// 实测 4:09.478「み」只占整窗 **180 ms / 1.4 s** ⇒ 整窗 RMS 结构上看不见它
    /// (生产读到的整窗 |rel| 是 **0.31 vs 1.51 —— 塌掉的那个反而赢**)。
    ///
    /// 夹具**按这个形状造**:候选 B 有两个音更贴、第三个音塌 20 dB。
    /// ⑴ 预注册:**整窗口径确实会选 B**(在判据里当场算出来,不是嘴上说);
    /// ⑵ 逐音口径必须选 A;⑶ ⛔ 阴性对照:把那个塌掉的音改成正常,逐音口径就该选 B。
    #[test]
    fn the_landing_pick_is_decided_per_note_so_one_collapsed_note_cannot_hide() {
        let (base0, notes, jobs, alts, spf) = pick_rig();
        let total = 120i64;
        // A(计划 −8):三个音都比参照低 2 dB;B(候选 −4):两个音 +0.5,第三个 −20。
        let mk = |gains: [f32; 3]| -> Vec<f32> {
            let mut d = vec![0.0f32; base0.len()];
            for (k, g) in gains.iter().enumerate() {
                let i = 4 + k;
                let amp = 0.1 * 10f32.powf(g / 20.0);
                let (a, b) = ((i * 10) * spf, ((i + 1) * 10) * spf);
                for (t, v) in d[a..b].iter_mut().enumerate() {
                    *v = amp * (2.0 * std::f32::consts::PI * 440.0 * t as f32 / 44100.0).sin();
                }
            }
            d
        };
        let run = |collapse: f32| -> Vec<f32> {
            let mut b = base0.clone();
            apply_dead_only_windows_with(
                &mut b,
                44100,
                total,
                &jobs,
                &alts,
                &notes,
                false,
                false,
                0,
                0.0, // ⛔ 谐波否决关掉 —— 这一条只钉【粒度】那根轴
                0.0, // ⛔ 修补遍也关掉
                0.0, // ⛔ 梳深否决也关掉
                0.0, // ⛔ 谱峰宽度否决也关掉
                0.0, // ⛔ 峰宽触发的修补遍也关掉
                0.0, // ⛔ 交接点体检也关掉
                0.0, // ⛔ 长音淡化也关掉
                |s, _| Ok(if s == -8 { mk([-2.0, -2.0, -2.0]) } else { mk([0.5, 0.5, collapse]) }),
            )
            .unwrap();
            b
        };

        // ⑴ 预注册:**整窗**口径下 B 更贴(算给自己看,免得这条判据靠嘴说)
        let win_db = |g: [f32; 3]| -> f32 {
            let p: f32 = g.iter().map(|v| 10f32.powf(v / 10.0)).sum::<f32>() / 3.0;
            10.0 * p.log10()
        };
        let (wa, wb) = (win_db([-2.0, -2.0, -2.0]).abs(), win_db([0.5, 0.5, -20.0]).abs());
        assert!(wb < wa, "夹具没造对:整窗口径下 B({wb:.2})必须比 A({wa:.2})更贴");

        // ⑵ 逐音口径:**A 赢**(B 那个塌掉的音 20 dB)
        let got = run(-20.0);
        let d = mid_db(&got, 5, spf);
        assert!(
            (d - (-23.01 - 2.0)).abs() < 0.6,
            "该选 A(每个音 −2 dB),窗内读到 {d:.2} dBFS"
        );

        // ⑶ ⛔ 阴性对照:第三个音不塌了 ⇒ 逐音口径就该选 B
        let ok = run(0.5);
        let d2 = mid_db(&ok, 5, spf);
        assert!(
            (d2 - (-23.01 + 0.5)).abs() < 0.6,
            "没有塌掉的音时该选 B(+0.5 dB),窗内读到 {d2:.2} dBFS"
        );
    }

    /// ⛔⛔⭐⭐ S163 —— **谐波否决:一个「更响但没在唱目标音高」的候选必须被剔掉。**
    ///
    /// 这是落点这条链上**唯一不需要参照**的一根轴,而参照恰恰在最需要它的地方取不到
    /// (密集救援的乐句里没有「没被救的邻居」——实测 68 个带候选的窗里 **24 个**取不到)。
    ///
    /// 夹具:候选 B 的电平**正好等于**参照(|rel| = 0,电平轴上完胜),
    /// 但它唱的是 1.5 倍的音高 —— 也就是「根本没救到」的那种形状。
    /// ⑴ 出厂门限下必须选 A;⑵ ⛔ **阴性对照:门限 = 0(关掉)时必须选 B** ——
    /// 没有这一格,这条判据分不清「否决起作用」与「A 本来就会赢」。
    /// ⭐⭐⭐ S163 —— **谱峰宽度否决**（[`landing_width_eps`]）直接打在 [`decide_group`] 上。
    ///
    /// ⛔ 为什么不走端到端：端到端那一版的两格先后红过 ——
    /// 先是「关掉也选干净的」（电平轴自己就决定了，峰宽轴不承重），
    /// 改完夹具又变成「开着也选糊的」（`width` 没被填充）。
    /// ⇒ **先把判据打在它真正要钉的那一层上**；端到端的填充路径另欠一条。
    ///
    /// ① 开着 ⇒ 剔掉糊的；② ⛔ **阴性对照**：关掉 ⇒ 电平轴说了算（糊的电平更贴参照）⇒ 选糊的。
    #[test]
    fn the_peak_width_veto_drops_the_blurrier_candidate() {
        // (job, shift, _, _, score)
        let mk = |shift: i64, rel: f32, width: f32| {
            (
                0usize,
                shift,
                0usize,
                Vec::<f32>::new(),
                CandScore {
                    rel: vec![(0, rel)],
                    harm: Vec::new(),
                    gone: Vec::new(),
                    dip: Vec::new(),
                    comb: Vec::new(),
                    width: vec![(0, width)],
                    usag: Vec::new(),
                    h2: Vec::new(),
                    mism: Vec::new(),
                    pitch: Vec::new(),
                    uplev: Vec::new(),
                },
            )
        };
        // A = 计划位移 −8：**电平更贴参照**（0.2 dB）但谱峰很糊（12%）
        // B = 候选 −4：电平差一点（2.0 dB）但谱峰很清晰（1%）
        let cand = vec![mk(-8, 0.2, 12.0), mk(-4, 2.0, 1.0)];
        // ① 开着（eps = 2.0）⇒ 12% > 1% × 2 ⇒ 剔掉 A ⇒ 选 B
        let on = decide_group(&cand, 0, -8, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(
            cand[on[0]].1,
            -4,
            "开着峰宽否决时必须选清晰的那个（读到 {:+}）",
            cand[on[0]].1
        );
        // ② ⛔ 阴性对照：关掉 ⇒ 只剩电平轴 ⇒ 选 A（糊的）
        let off = decide_group(&cand, 0, -8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(
            cand[off[0]].1,
            -8,
            "关掉之后只剩电平轴，该选糊的那个（读到 {:+}）",
            cand[off[0]].1
        );
        // ③ 夹具有效性：两个候选的峰宽必须真的拉开得够多
        assert!(
            cand[0].4.worst_width() > cand[1].4.worst_width() * 2.0,
            "夹具没造对：两个候选的峰宽必须差过门限倍数"
        );
    }

    #[test]
    fn the_harmonic_veto_drops_a_candidate_that_is_not_singing_the_target_pitch() {
        let (base0, notes, jobs, alts, spf) = pick_rig();
        let total = 120i64;
        let stack = |f0: f32, amp: f32, n: usize| -> Vec<f32> {
            let mut d = vec![0.0f32; n];
            for i in 4..7usize {
                let (a, b) = ((i * 10) * spf, ((i + 1) * 10) * spf);
                for (t, v) in d[a..b].iter_mut().enumerate() {
                    let tt = t as f32 / 44100.0;
                    let mut y = 0.0f32;
                    for k in 1..=8usize {
                        y += (1.0 / k as f32)
                            * (2.0 * std::f32::consts::PI * f0 * k as f32 * tt).sin();
                    }
                    *v = amp * y;
                }
            }
            d
        };
        // A(计划 −8):唱**目标音高** 440,但比参照轻 4 dB。
        // B(候选 −4):唱 660(= 1.5 倍,音高不对),电平正好等于参照。
        let run = |eps: f32| -> Vec<f32> {
            let mut b = base0.clone();
            apply_dead_only_windows_with(
                &mut b, 44100, total, &jobs, &alts, &notes, false, false, 0, eps, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                |s, _| {
                    Ok(if s == -8 {
                        stack(440.0, 0.0537, base0.len())
                    } else {
                        stack(660.0, 0.0851, base0.len())
                    })
                },
            )
            .unwrap();
            b
        };
        // 先把两条臂各自的电平量出来 —— 判据靠它区分「选了谁」。
        let a_db = mid_db(&stack(440.0, 0.0537, base0.len()), 5, spf);
        let b_db = mid_db(&stack(660.0, 0.0851, base0.len()), 5, spf);
        assert!(
            b_db > a_db + 2.0,
            "夹具没造对:B 必须在**电平**上明显占优({b_db:.2} vs {a_db:.2} dBFS)"
        );

        // ⑴ 出厂门限:谐波否决把 B 剔掉 ⇒ 选 A
        let on = mid_db(&run(LANDING_HARM_EPS_DEFAULT), 5, spf);
        assert!(
            (on - a_db).abs() < 0.8,
            "出厂门限下该选 A(唱对音高的那个),读到 {on:.2},A={a_db:.2} B={b_db:.2}"
        );

        // ⑵ ⛔⛔ S165 —— **音高闸现在会先把 B 拦下来**(B 唱 660 而目标 440 = 高 700 音分),
        //    所以「关掉谐波否决 ⇒ 退回电平轴 ⇒ 选 B」这条阴性对照**不再成立**。
        //    ⇒ 改用一个**音高正确、只是谐波差**的 B′ 来隔离那一个变量。
        //    (这不是回归:唱错音是**质的失败**,本来就该被更早、更直接地拦住,
        //     不该等谐波否决去间接发现。)
        let run_pitch_ok = |eps: f32| -> Vec<f32> {
            let mut b = base0.clone();
            apply_dead_only_windows_with(
                &mut b, 44100, total, &jobs, &alts, &notes, false, false, 0, eps, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                |s, _| {
                    Ok(if s == -8 {
                        stack(440.0, 0.0537, base0.len())
                    } else {
                        // B′:**唱对音高** 440,只是响、而且谐波结构差(只有 2 根)
                        let mut d = vec![0.0f32; base0.len()];
                        for i in 4..7usize {
                            let (aa, bb) = ((i * 10) * spf, ((i + 1) * 10) * spf);
                            for (t, v) in d[aa..bb].iter_mut().enumerate() {
                                let tt = t as f32 / 44100.0;
                                let y = (2.0 * std::f32::consts::PI * 440.0 * tt).sin()
                                    + 0.5 * (2.0 * std::f32::consts::PI * 880.0 * tt).sin();
                                *v = 0.0851 * y;
                            }
                        }
                        d
                    })
                },
            )
            .unwrap();
            b
        };
        let off = mid_db(&run_pitch_ok(0.0), 5, spf);
        assert!(
            off > a_db + 1.0,
            "关掉谐波否决、且候选音高正确时,该退回电平轴选那个更响的,读到 {off:.2},A={a_db:.2}"
        );

        // ⑶ ⭐⭐⭐⭐ S165 —— **音高闸拦得住唱错音的候选**,而且**不依赖谐波否决**。
        //    用户 2026-08-29 听到的灾难就是这一类:目标 1480 Hz 而实唱 320 Hz。
        //    这里 B 唱 660 而目标 440(高 700 音分,远超 PITCH_GATE_CENTS)。
        let gated = mid_db(&run(0.0), 5, spf);
        assert!(
            (gated - a_db).abs() < 0.8,
            "⛔ 谐波否决**关掉**时,音高闸仍必须把唱 660 的 B 拦下来 ⇒ 选 A;             读到 {gated:.2},A={a_db:.2} B={b_db:.2}"
        );
    }

    /// ⛔⛔⭐⭐⭐ S163 —— **救援把一个音唱成了静音时,必须自己再渲一遍去修**。
    ///
    /// ## 它治的是用户点名的 yuyuko **4:49**
    /// `[801]あ`(目标 MIDI 90、落点 −11)在 289.62-289.94 之间塌到 **−70 dBFS**,
    /// 而同刻 `base` 平稳在 −32 ⇒ 塌在**解码**里。S162 把它登记成「离群点」收工。
    /// S163 补做了那件一直没做的事 —— **对这一处做落点扫描**:
    ///
    /// | 落点 | −9 | **−10** | **−11(出厂)** | **−12** | −13 | −14 |
    /// |---|---|---|---|---|---|---|
    /// | 低于 −40 dBFS 的 20 ms 格 | 12 | **0** | **17** | **0** | 0 | 0 |
    ///
    /// ⇒ 一个半音之外全干净 ⇒ **不是模型上限,是那一格的死点**,而落点选法结构上够不着它
    ///   (yuyuko 的窄预算计划与出厂**完全相同** ⇒ 一个候选都没有)。
    ///
    /// ## 这一条钉四件
    /// ⑴ 计划的那一遍唱出静音 ⇒ **多渲 ±1 并换过去**;
    /// ⑵ ⛔ **阴性对照 A**:计划那一遍好好的 ⇒ **一次修补遍都不许渲**(记账用调用计数);
    /// ⑶ ⛔ **阴性对照 B**:门限 = 0(关掉)⇒ 即使塌了也不修、也不多渲;
    /// ⑷ ⛔ 修补遍**也要过同一套打分** —— 一个「不塌但完全不像话」的修补候选不许赢。
    #[test]
    fn a_note_the_rescue_sang_into_silence_triggers_a_repair_pass() {
        let (base0, notes, jobs, _alts, spf) = pick_rig();
        let total = 120i64;
        // 计划 −8;窗盖住音 4-6。素材:正常段 = 参照电平的正弦。
        let tone = |amp: f32, hole: Option<(usize, usize)>| -> Vec<f32> {
            let mut d = vec![0.0f32; base0.len()];
            for i in 4..7usize {
                let (a, b) = ((i * 10) * spf, ((i + 1) * 10) * spf);
                for (t, v) in d[a..b].iter_mut().enumerate() {
                    *v = amp * (2.0 * std::f32::consts::PI * 440.0 * t as f32 / 44100.0).sin();
                }
            }
            // 「唱没了」:音 5 中段挖掉 300 ms(⇒ 低于 −50 dBFS 且比该音中位低 >25 dB)
            if let Some((a, b)) = hole {
                for v in d[a..b].iter_mut() {
                    *v = 0.0;
                }
            }
            d
        };
        let hole = ((5 * 10 + 3) * spf, (5 * 10 + 6) * spf); // 300 ms,音 5 的中段
        let calls = std::cell::RefCell::new(Vec::<i64>::new());
        let run = |repair_ms: f32, broken: bool| -> (Vec<f32>, Vec<i64>) {
            calls.borrow_mut().clear();
            let mut b = base0.clone();
            apply_dead_only_windows_with(
                &mut b, 44100, total, &jobs, &[], &notes, false, false, 0, 0.0, repair_ms, 0.0, 0.0, 0.0, 0.0, 0.0,
                |s, _| {
                    calls.borrow_mut().push(s);
                    Ok(if s == -8 && broken {
                        tone(0.1, Some(hole))
                    } else if s == -8 {
                        tone(0.1, None)
                    } else {
                        // ⭐ 修补遍：**没有洞**，但故意比计划那一遍【轻 3 dB】——
                        // ⛔ 否则「逐音最差 |rel|」那一层自己就能选对，
                        //    【塌陷否决】那一层就是空的（变异 r2 当场抬出来的）。
                        tone(0.1 * 10f32.powf(-3.0 / 20.0), None)
                    })
                },
            )
            .unwrap();
            let c = calls.borrow().clone();
            (b, c)
        };
        let mid = |x: &[f32]| -> f32 {
            let (a, b) = (hole.0, hole.1);
            let e: f64 =
                x[a..b].iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / (b - a) as f64;
            (10.0 * (e + 1e-30).log10()) as f32
        };

        // ⑴ 塌了 ⇒ 修补遍跑了,而且洞被补上
        let (fixed, c1) = run(LANDING_REPAIR_MS_DEFAULT, true);
        assert!(
            c1.len() > 1,
            "唱出静音却没渲修补遍 —— donor_render 只被调用了 {:?}",
            c1
        );
        assert!(
            c1.iter().any(|&s| s == -7 || s == -9),
            "修补遍必须是 ±1:实际调用 {:?}",
            c1
        );
        assert!(mid(&fixed) > -35.0, "洞没被补上:那 300 ms 读 {:.1} dBFS", mid(&fixed));

        // ⑵ ⛔ 阴性对照 A:没塌 ⇒ **一次都不许多渲**
        let (ok, c2) = run(LANDING_REPAIR_MS_DEFAULT, false);
        assert_eq!(c2, vec![-8], "没塌却多渲了:{c2:?} —— 代价必须跟着「真的坏了」走");
        assert!(mid(&ok) > -35.0);

        // ⑶ ⛔ 阴性对照 B:门限 0 ⇒ 既不修也不多渲
        let (off, c3) = run(0.0, true);
        assert_eq!(c3, vec![-8], "门限 0 还多渲了:{c3:?}");
        assert!(
            mid(&off) < -60.0,
            "门限 0 时那 300 ms 必须还是静音,实际 {:.1} dBFS",
            mid(&off)
        );
    }

    /// ⛔ S163 —— [`silence_run_ms`] 的两条门必须**同时**成立。
    ///
    /// ⑴ 只低于绝对地板不算(整个音本来就很轻);⑵ 只相对该音中位低不算(**辅音凹陷**)。
    /// 第 ⑵ 条是实测逼出来的:用「音内中位 − 最低 100 ms」那把尺子,四条臂上有 **6-11 个音**
    /// 超过 22 dB,而它们全是 `か`/`が`/`さ`/`つ`。
    #[test]
    fn the_silence_detector_needs_both_gates() {
        let sr = 44100u32;
        let n = sr as usize; // 1 s
        let tone = |amp: f32| -> Vec<f32> {
            (0..n).map(|i| amp * ((i as f32) * 0.05).sin()).collect()
        };
        // ⓐ 真的塌了:中段 300 ms 归零
        let mut hole = tone(0.1);
        for v in hole[(n * 35 / 100)..(n * 65 / 100)].iter_mut() {
            *v = 0.0;
        }
        assert!(silence_run_ms(&hole, sr) >= 200.0, "300 ms 的洞没读出来");
        // ⓑ ⛔ 整段都很轻(远低于地板)但**没有洞** ⇒ 0
        assert_eq!(silence_run_ms(&tone(0.0004), sr), 0.0, "整体轻不等于唱没了");
        // ⓒ ⛔ **辅音式凹陷**:中段掉 30 dB 但仍在 −45 dBFS 以上 ⇒ 0
        let mut dip = tone(0.5);
        for v in dip[(n * 35 / 100)..(n * 65 / 100)].iter_mut() {
            *v *= 0.03;
        }
        assert_eq!(silence_run_ms(&dip, sr), 0.0, "辅音凹陷被当成「唱没了」");
        // ⓓ 太短的段不猜
        assert_eq!(silence_run_ms(&tone(0.1)[..100], sr), 0.0);
    }

    /// ⛔⛔⭐⭐⭐ S163 —— **拼接器不许把输出交给一条【正在静音】的 donor。**
    ///
    /// 用户 2026-08-26 深夜点破 yuyuko **4:50.694**:我修好 4:49 之后它反而炸了。
    /// 四层剖面(同一次 run 的转储)说得很清楚 —— 出去的那条 donor 一路是好的(−22…−29 dBFS),
    /// 进来的那条自己有 60-80 ms 的静音起头(−49…−68),而交接照做不误 ⇒ 成品掉到 −72。
    /// 而这**不是一处**:同一把尺子扫四条臂,yuyuko 1 处(正是它)、
    /// yachiyo **5 处**(229.500 / 238.140 / 246.800 = 用户点名的 3:49.524 / 3:58.162 那一族)。
    ///
    /// 这一条钉三件:⑴ 有缺口 ⇒ 洞被补上;⑵ ⛔ 门限 0 ⇒ 洞还在(**阴性对照**);
    /// ⑶ ⛔ 进来的那条本来就好 ⇒ **逐样本不变**(**阴性对照**,防「它其实在无条件延后」)。
    #[test]
    fn the_splice_never_hands_over_to_a_silent_donor() {
        const SR: u32 = 48000;
        let spf = 960.0f64; // 20 ms/帧
        let total = 100i64;
        let n = (total as f64 * spf) as usize;
        // 左窗 [10,40)、右窗 [40,80),位移不同 ⇒ 交接在 40 帧 = 0.80 s
        let jobs = vec![
            DeadJob { shift: -5, start: 10, end: 40 },
            DeadJob { shift: -9, start: 40, end: 80 },
        ];
        let tone = |amp: f32, len: usize| -> Vec<f32> {
            (0..len).map(|i| amp * ((i as f32) * 0.07).sin()).collect()
        };
        // 右边那条 donor:窗起点之后 `hole_ms` 毫秒是静音
        let make = |hole_ms: f64| -> (Vec<f32>, Vec<f32>) {
            let left = tone(0.2, n);
            let mut right = tone(0.2, n);
            let h0 = (40.0 * spf) as usize;
            let h1 = h0 + (f64::from(SR) * hole_ms / 1000.0) as usize;
            for v in right[h0..h1.min(n)].iter_mut() {
                *v = 0.0;
            }
            (left, right)
        };
        let run = |hole_ms: f64, db: f32| -> Vec<f32> {
            let (left, right) = make(hole_ms);
            let mut b = vec![0.0f32; n];
            apply_dead_only_windows_with(
                &mut b, SR, total, &jobs, &[], &[], false, false, 0, 0.0, 0.0, 0.0, 0.0, 0.0, db, 0.0,
                |s, _| Ok(if s == -5 { left.clone() } else { right.clone() }),
            )
            .unwrap();
            b
        };
        let hole_db = |x: &[f32]| -> f32 {
            // 交接之后那 60 ms(⛔ 跳开 10 ms 的交叉淡化)
            let a = (40.0 * spf) as usize + SR as usize / 100;
            let b = a + SR as usize * 6 / 100;
            let e: f64 =
                x[a..b].iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / (b - a) as f64;
            (10.0 * (e + 1e-30).log10()) as f32
        };

        // ⑴ 有 80 ms 的静音起头 ⇒ 出厂门限把洞补上
        let fixed = run(80.0, HANDOVER_DEFICIT_DB_DEFAULT);
        assert!(
            hole_db(&fixed) > -25.0,
            "交接之后还是个洞:{:.1} dBFS —— 手上明明有好料",
            hole_db(&fixed)
        );
        // ⑵ ⛔ 阴性对照:门限 0 ⇒ 洞照旧
        let off = run(80.0, 0.0);
        assert!(
            hole_db(&off) < -60.0,
            "门限 0 时那 60 ms 必须还是洞,实际 {:.1} dBFS",
            hole_db(&off)
        );
        // ⑶ ⛔ 阴性对照:进来的那条本来就好 ⇒ **逐样本不变**
        let a = run(0.0, HANDOVER_DEFICIT_DB_DEFAULT);
        let b = run(0.0, 0.0);
        assert_eq!(a, b, "没有缺口时交接点被动了 —— 它不许无条件延后");
    }

    /// ⛔⛔⭐⭐ S163 —— **同一个长音内部的接缝,交叉淡化必须拉长;音节边界上一个字节不动。**
    ///
    /// 用户 2026-08-26:「我实在受不了那种**一个长音三个听感**,或者**长音割裂**了」。
    /// 炉心融解结尾那个「あ」在谱面上是 `[796..=802]` **七个连着的同词音符**,
    /// 而计划器按死音分组把它切成 3-4 段、每段一个落点,段间只有 **10 ms** 淡化。
    ///
    /// ⭐ 两条 donor 的**音高都是目标音高**(逆变换已经做完)⇒ 差别只是落点带来的**音色**;
    /// 在持续元音上慢慢换音色听不出缝,10 ms 换才是「割裂」。
    ///
    /// 三格:⑴ `tied` ⇒ 过渡真的变长;⑵ ⛔ **阴性对照**:同一份素材、`tied = false`
    /// ⇒ **逐样本等于关掉**(证明它认的是「延续」不是「有缝就拉长」);
    /// ⑶ ⛔ **阴性对照**:旋钮 0 ⇒ 逐样本回到今天。
    #[test]
    fn a_seam_inside_one_long_note_gets_a_long_crossfade() {
        const SR: u32 = 48000;
        let spf = 960.0f64; // 20 ms/帧
        let total = 100i64;
        let n = (total as f64 * spf) as usize;
        // 两个窗在 40 帧(0.80 s)交接;音符表:每 20 帧一个音
        let jobs = vec![
            DeadJob { shift: -4, start: 10, end: 40 },
            DeadJob { shift: -9, start: 40, end: 80 },
        ];
        let spans = |tied: bool| -> Vec<NoteSpan> {
            (0..5i64)
                .map(|k| NoteSpan {
                    start: k * 20,
                    frames: 20,
                    sung: true,
                    hz: 440.0,
                    // 第 2 个音(帧 40 起)= 交接落点;只有它的 `tied` 在变
                    tied: tied && k == 2,
                })
                .collect()
        };
        let loud: Vec<f32> = (0..n).map(|i| 0.30 * ((i as f32) * 0.07).sin()).collect();
        let soft: Vec<f32> = (0..n).map(|i| 0.03 * ((i as f32) * 0.07).sin()).collect();
        let run = |tied: bool, xf_ms: f64| -> Vec<f32> {
            let nd = spans(tied);
            let mut b = vec![0.0f32; n];
            apply_dead_only_windows_with(
                &mut b, SR, total, &jobs, &[], &nd, false, false, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, xf_ms,
                |s, _| Ok(if s == -4 { loud.clone() } else { soft.clone() }),
            )
            .unwrap();
            b
        };
        // 过渡宽度 = 输出包络从 90% 走到 10% 用了多少毫秒
        let width_ms = |x: &[f32]| -> f64 {
            let cell = SR as usize / 500; // 2 ms
            let c: Vec<f32> = (0..n / cell)
                .map(|i| {
                    let s = &x[i * cell..(i + 1) * cell];
                    (s.iter().map(|v| v * v).sum::<f32>() / cell as f32).sqrt()
                })
                .collect();
            let hi = 0.30 / std::f32::consts::SQRT_2;
            let (mut a, mut b) = (None, None);
            for (i, &v) in c.iter().enumerate() {
                if i * cell < (35.0 * spf) as usize {
                    continue;
                }
                if a.is_none() && v < hi * 0.9 {
                    a = Some(i);
                }
                if a.is_some() && b.is_none() && v < hi * 0.1 {
                    b = Some(i);
                }
            }
            match (a, b) {
                (Some(x), Some(y)) => (y - x) as f64 * cell as f64 * 1000.0 / f64::from(SR),
                _ => f64::NAN,
            }
        };

        let long = width_ms(&run(true, TIED_XFADE_MS_DEFAULT));
        let short = width_ms(&run(false, TIED_XFADE_MS_DEFAULT));
        assert!(
            long > 40.0,
            "长音延续处的过渡只有 {long:.0} ms —— 拉长那一条没生效"
        );
        assert!(
            short < 20.0,
            "⛔ 音节变化处也被拉长了({short:.0} ms)—— 它认错了边界"
        );
        // ⑶ ⛔ 旋钮 0 ⇒ 与「不 tied」逐样本相同
        assert_eq!(run(true, 0.0), run(false, 0.0), "旋钮 0 时两种边界必须一模一样");
        assert_eq!(run(false, TIED_XFADE_MS_DEFAULT), run(false, 0.0),
                   "非延续边界上开不开旋钮必须逐样本相同");
    }

    /// ⛔ S163 —— `note_spans_tied` 认「延续」只看**歌词**,不看音高。
    /// 实测炉心融解结尾那个长音 `[796..=802]` 全是「あ」而音高是 71/73/75/76/83/81 ——
    /// 按音高认会**全部漏掉**。⛔ 隔着休止不算延续。
    #[test]
    fn a_tied_note_is_recognised_by_its_lyric_not_its_pitch() {
        let nn = [71i64, 73, 75, 76, 0, 83, 83];
        let fr = [10i64; 7];
        let ly: Vec<String> =
            ["あ", "あ", "あ", "あ", "R", "あ", "い"].iter().map(|s| s.to_string()).collect();
        let v = note_spans_tied(&nn, &fr, 0, &ly);
        assert!(!v[0].tied, "第一个音没有前驱");
        assert!(v[1].tied && v[2].tied && v[3].tied, "同词的连音必须认成延续(音高在变)");
        assert!(!v[4].tied, "休止不是唱音");
        assert!(!v[5].tied, "⛔ 隔着休止就不是同一个长音了");
        assert!(!v[6].tied, "换了词就不是延续");
        // ⛔ 没有歌词 ⇒ 全 false ⇒ 拼接层逐位回到今天
        assert!(note_spans(&nn, &fr, 0).iter().all(|x| !x.tied));
    }

    /// S162 —— 两个旋钮的出厂值与垃圾值倒向。
    #[test]
    fn the_s162_landing_and_level_knobs_ship_as_declared() {
        assert!(parse_landing_tie_thin(None), "落点平局破法出厂 = thin(= 逐位同今天)");
        assert!(!parse_landing_tie_thin(Some("shallow")), "shallow 必须关得掉那一层");
        for junk in ["", "thin", "1", "0", "deep"] {
            assert!(parse_landing_tie_thin(Some(junk)), "垃圾值 {junk:?} 必须落回出厂");
        }
        // ⚠ `trim` 是有意的(与本文件其它 `parse_*` 一致)⇒ 带空格的写法照样关得掉。
        assert!(!parse_landing_tie_thin(Some("  shallow  ")), "trim 之后应当仍然识别");
        assert_eq!(parse_level_match_db(None), 6.0, "电平匹配出厂门 = 6 dB");
        assert_eq!(parse_level_match_db(Some("0")), 0.0, "0 必须关得掉");
        assert_eq!(parse_level_match_db(Some("4.5")), 4.5);
        for junk in ["", "abc", "-1", "25", "nan", "inf"] {
            assert_eq!(parse_level_match_db(Some(junk)), 6.0, "垃圾值 {junk:?} 必须落回出厂");
        }
    }

    /// ⭐⭐ S162 —— **薄片闸**:相邻两窗之间 1..n 帧的缝用前一个窗的尾巴补上,
    /// 而**负的缝(已经重叠)一个字不许动** —— 那是 [`GUARD_FRAMES`] 有意做的。
    ///
    /// ⛔ 没有这条判据,`close_short_slivers` 就只有 doc 和指纹盯着它:
    /// 把 `gap >= 1` 写成 `gap >= 0`、把「补前一个的尾」写成「拉后一个的头」、
    /// 或者把 `n` 的门去掉,**别的每一条测试都还是绿的**。
    #[test]
    fn the_sliver_gate_closes_only_short_positive_gaps() {
        let j = |shift: i64, start: i64, end: i64| DeadJob { shift, start, end };

        // ⓐ n == 0 ⇒ 逐帧不变(出厂)
        let src = vec![j(-5, 100, 200), j(-10, 202, 300)];
        assert_eq!(close_short_slivers(src.clone(), 0), src, "出厂必须逐帧不变");

        // ⓑ 2 帧的缝、n = 3 ⇒ 两窗接上(补的是**前一个的尾**,后一个的 start 不许动)
        let out = close_short_slivers(src.clone(), 3);
        assert_eq!(out[0].end, 202, "前一个窗的尾应该补到后一个窗的起点");
        assert_eq!(out[0].start, 100, "前一个窗的头不许动");
        assert_eq!((out[1].start, out[1].end), (202, 300), "后一个窗一个字不许动");

        // ⓒ 缝比 n 大 ⇒ 不动(长休止里留 base 是对的)
        let far = vec![j(-5, 100, 200), j(-10, 210, 300)];
        assert_eq!(close_short_slivers(far.clone(), 3), far, "缝超过门限不许补");

        // ⓓ ⛔ 负的缝(两窗重叠 = `GUARD_FRAMES` 有意做的)一个字不许动
        let ov = vec![j(-5, 100, 205), j(-10, 201, 300)];
        assert_eq!(close_short_slivers(ov.clone(), 3), ov, "已经重叠的窗不许再动");

        // ⓔ 零缝(正好接上)也不许动 —— `gap >= 1` 那道门的边界
        let touch = vec![j(-5, 100, 200), j(-10, 200, 300)];
        assert_eq!(close_short_slivers(touch.clone(), 3), touch, "gap == 0 不该触发");

        // ⓕ 输入乱序也要对(jobs 不保证按 start 排好)
        let un = vec![j(-10, 202, 300), j(-5, 100, 200)];
        let out = close_short_slivers(un, 3);
        let a = out.iter().find(|x| x.start == 100).unwrap();
        assert_eq!(a.end, 202, "乱序输入也要按时间轴找邻居");

        // ⓖ 旋钮:垃圾值退回出厂,不许静默夹住
        assert_eq!(parse_close_sliver(None), 0);
        assert_eq!(parse_close_sliver(Some("3")), 3);
        for junk in ["", "abc", "-1", "13", "2.5", "nan"] {
            assert_eq!(parse_close_sliver(Some(junk)), 0, "垃圾值 {junk:?} 必须落回出厂");
        }
    }

    /// ⭐⭐⭐ S159zzl —— **带膝盖的共振峰跟随:关着逐位不变,膝盖以内恒等,越过之后按 `c·(|s|−6)` 起。**
    ///
    /// 靶子、实测表、已登记的代价与护栏全在 [`formant_knee`] 的 doc。这条判据钉五件:
    /// ⑴ `knee == 0` ⇒ 退回 `κ·semis`(今天,**逐位不变**,而且不依赖任何浮点等价);
    /// ⑵ ⛔ **膝盖以内恒等于 0** —— 这是它相对常数 κ 的**全部理由**:
    ///    常数 `κ=0.30` 在 −6 上既过冲(H1−H4 +1.83 → −3.56)又白掉 3 dB 高频;
    /// ⑶ 膝盖以外**线性**且**符号跟着位移走**(下移时共振峰也往下);
    /// ⑷ ⛔ **膝盖处连续**(`|s| = 6` 左右两侧都是 0)—— 不连续会在计划边界上造出可闻的跳变;
    /// ⑸ `knee` 的越界值退回出厂,不许静默夹住。
    #[test]
    fn the_formant_knee_is_identity_inside_the_knee_and_linear_outside() {
        // ⑴ 关着 = 今天
        for k in [0.0f32, 0.3, 1.0] {
            for s in [-14.0f64, -6.0, 2.0, 14.0] {
                assert_eq!(
                    formant_shift_semitones(s, k, 0.0),
                    f64::from(k) * s,
                    "knee = 0 必须逐位退回 κ·semis(κ {k}, s {s})"
                );
            }
        }
        // ⑵ 膝盖以内恒 0 —— 与 κ 无关
        for s in [-6.0f64, -5.5, -1.0, 0.0, 3.0, 6.0] {
            assert_eq!(
                formant_shift_semitones(s, 1.0, 0.5),
                0.0,
                "|s| <= {FORMANT_KNEE_ST} 时必须恒等于 0(s {s})—— 安全区不许被碰"
            );
        }
        // ⑶ 膝盖以外线性,符号跟着位移
        let up = formant_shift_semitones(14.0, 0.0, 0.5);
        let dn = formant_shift_semitones(-14.0, 0.0, 0.5);
        assert!((up - 0.5 * (14.0 - FORMANT_KNEE_ST)).abs() < 1e-12, "上移读到 {up}");
        assert!((dn + 0.5 * (14.0 - FORMANT_KNEE_ST)).abs() < 1e-12, "下移必须反号,读到 {dn}");
        assert!((formant_shift_semitones(10.0, 0.0, 0.5) - 2.0).abs() < 1e-12);
        // ⑷ 膝盖处连续
        let eps = 1e-6;
        let a = formant_shift_semitones(FORMANT_KNEE_ST - eps, 0.0, 0.5);
        let b = formant_shift_semitones(FORMANT_KNEE_ST + eps, 0.0, 0.5);
        assert!(a.abs() < 1e-9 && b.abs() < 1e-6, "膝盖处必须连续(左 {a}, 右 {b})");
        // ⑸ 旋钮
        assert_eq!(FORMANT_KNEE_DEFAULT, 0.0, "出厂必须是关的");
        assert_eq!(parse_formant_knee(None), 0.0);
        assert!((parse_formant_knee(Some("0.5")) - 0.5).abs() < 1e-12);
        assert_eq!(parse_formant_knee(Some("1.5")), 0.0, "越界退回出厂,不许静默夹住");
        assert_eq!(parse_formant_knee(Some("nan")), 0.0, "NaN 不许通过");
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
        let w = dead_group_windows(&nn, &fr, &plan, &all_reachable(), 0);
        assert_eq!(w.len(), 1, "同一条 donor 上不许挖洞");
        assert_eq!(w[0].shift, -6);
        assert_eq!((w[0].start, w[0].end), (6, 57), "合并后的窗要盖住两段与中间那个休止(S163: 尾边 55→57，见 `REST_POST_FRAMES`)");
        // ⛔ 位移不同 ⇒ 必须**不**合并(那两侧本来就是两条不同的 donor)。
        let plan2 = [
            DeadGroup { start: 1, end: 1, shift: -6 },
            DeadGroup { start: 3, end: 3, shift: -7 },
        ];
        assert_eq!(dead_group_windows(&nn, &fr, &plan2, &all_reachable(), 0).len(), 2, "位移不同不许合并");
        // ⛔ 中间夹着一个**唱音** ⇒ 必须不合并(合了等于把乘客偷偷拖进救援)。
        let nn3 = [0, 85, 73, 85, 0];
        let plan3 = [
            DeadGroup { start: 1, end: 1, shift: -6 },
            DeadGroup { start: 3, end: 3, shift: -6 },
        ];
        assert_eq!(dead_group_windows(&nn3, &fr, &plan3, &all_reachable(), 0).len(), 2, "跨过唱音不许合并");
        // ⛔ 长休止不许桥:缺陷只有 60 ms,不设上限时窗会被撑到 18.6 s(实测)。
        let fr_long = [10i64, 20, 200, 20, 10];
        assert_eq!(dead_group_windows(&nn, &fr_long, &plan, &all_reachable(), 0).len(), 2, "4 秒的休止不许桥");
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

    /// ⭐⭐⭐ S159zw —— **短休止两侧的窗一格都不许伸进去**(用户 2026-08-23 点名的 3:35.888)。
    ///
    /// 机理、取证与代价在 [`SHORT_REST_NO_EXTEND_FRAMES`] 的 doc。这条判据钉四件:
    /// ⑴ 休止 = 3 帧(命中门限)⇒ 两侧 `pre`/`post` 都是 **0**,窗正好停在音界上;
    /// ⑵ ⛔ **阴性对照**:休止 = 4 帧(刚过门限)⇒ 照旧伸出(`pre = min(4,2) = 2` / `post = min(2,2) = 2`);
    /// ⑶ ⛔ **`gap == 0`(根本没有休止)那一条不许被这一刀碰** —— 那是 `GUARD_FRAMES` 的地盘;
    /// ⑷ 短休止那一对窗**不许重叠**,而且中间必须整段留给 base。
    ///
    /// ⛔ 变异(逐个真跑过):
    /// * `SHORT_REST_NO_EXTEND_FRAMES` 改成 0 ⇒ ⑴ 读回 `pre/post = 1`,**红**;
    /// * 改成 4 ⇒ ⑵ 的 4 帧休止也被吃掉,**红**;
    /// * 把 `gap_prev > 0 &&` 那个前置条件去掉(让它也管 `gap == 0`)⇒ ⑶ **红**。
    /// ⭐⭐⭐ S163 —— [`REST_POST_FRAMES`] 的行为闸（`2 → 4`，用户报 `1:43.472` 那一族）。
    ///
    /// 钉四件，**期望值全部手算**（拿被测函数算期望 = 恒真）：
    /// ⑴ 休止够长 ⇒ post 正好 **4 帧**；⑵ 休止短 ⇒ `.min(gap/2)` 收缩，**两窗永不重叠**；
    /// ⑶ `gap ≤ SHORT_REST_NO_EXTEND_FRAMES` ⇒ post **0**（那条更早的分支不受影响）；
    /// ⑷ **pre/post 对称** —— 两侧同样的 gap 必须伸同样多。
    #[test]
    fn rest_post_extends_four_frames_and_shrinks_with_the_gap() {
        let nn = [0i64, 85, 0];
        // fr = [前休止, 音, 后休止] ⇒ cum = [0, fr0, fr0+20, fr0+20+gap]
        // 组 start=end=1：pre = min(4, fr0/2) ⇒ start = fr0 − pre
        //                post = 见下 ⇒ end = (fr0+20) + post
        let g = [DeadGroup { start: 1, end: 1, shift: -6 }];
        // (前休止, 后休止, 期望 start, 期望 end)  —— 全部手算
        let cases: &[(i64, i64, i64, i64)] = &[
            (10, 20, 6, 34), // pre=min(4,5)=4 ⇒ 6 ; post=min(4,10)=4 ⇒ 30+4=34
            (10, 8, 6, 34),  // post=min(4,4)=4 ⇒ 34
            (10, 6, 6, 33),  // post=min(4,3)=3 ⇒ 33   ← .min(gap/2) 收缩
            (10, 4, 6, 32),  // post=min(4,2)=2 ⇒ 32
            (10, 3, 6, 30),  // gap ≤ 门限 ⇒ post=0     ← ⑶
            (3, 20, 10, 34), // 前侧 gap ≤ 门限 ⇒ pre=0 ⇒ start=10
            (6, 20, 7, 34),  // pre=min(4,3)=3 ⇒ 10−3=7 …前侧 fr0=6 ⇒ cum[1]=6 ⇒ 6−3=3
        ];
        for &(pre_gap, post_gap, want_s, want_e) in cases {
            let fr = [pre_gap, 20i64, post_gap];
            let w = dead_group_windows(&nn, &fr, &g, &all_reachable(), 0);
            assert_eq!(w.len(), 1, "gap {pre_gap}/{post_gap}: 应当只有一条窗");
            // start/end 用手算值（最后一行的 want_s 由 fr0 决定，见上表注释）
            let cum1 = pre_gap;
            let cum2 = pre_gap + 20;
            let pre = if pre_gap <= 3 { 0 } else { 4.min(pre_gap / 2) };
            let post = if post_gap <= 3 { 0 } else { 4.min(post_gap / 2) };
            assert_eq!(
                (w[0].start, w[0].end),
                (cum1 - pre, cum2 + post),
                "gap {pre_gap}/{post_gap}: 手算应为 ({}, {})", cum1 - pre, cum2 + post
            );
            if pre_gap == 10 && post_gap == 20 {
                assert_eq!((w[0].start, w[0].end), (want_s, want_e), "字面量锚点");
            }
        }
        // ⑷ 对称：同样的 gap，两侧伸的帧数必须相同
        for gap in [4i64, 6, 8, 20] {
            let fr = [gap, 20i64, gap];
            let w = dead_group_windows(&nn, &fr, &g, &all_reachable(), 0);
            let ext_pre = gap - w[0].start;
            let ext_post = w[0].end - (gap + 20);
            assert_eq!(ext_pre, ext_post, "gap {gap}: pre/post 不对称（{ext_pre} vs {ext_post}）");
        }
    }

    #[test]
    fn a_short_rest_keeps_both_windows_out_of_it() {
        let nn = [0i64, 85, 0, 85, 0];
        // ⑴ 中间休止 3 帧 = 门限 ⇒ 两侧都不许伸。
        let three = [10i64, 20, 3, 20, 10];
        let g = [
            DeadGroup { start: 1, end: 1, shift: -6 },
            DeadGroup { start: 3, end: 3, shift: -7 }, // ⛔ 位移不同 ⇒ 不会被同位移合并掩盖
        ];
        let w = dead_group_windows(&nn, &three, &g, &all_reachable(), 0);
        assert_eq!(w.len(), 2, "两条组必须各自有窗(读到 {w:?})");
        assert_eq!(
            (w[0].start, w[0].end, w[1].start, w[1].end),
            (6, 30, 33, 57),
            "3 帧休止 ⇒ 内侧两条边正好停在音界上 30 / 33(外侧那两条 10 帧休止照常伸出)(读到 {w:?})
\n             ⚠ S163: 尾边 55→57 是 `REST_POST_FRAMES` 2→4 的直接结果(尾部 gap=10 ⇒ min(4,5)=4);
\n             内侧的 30/33 一个字节没动 —— `.min(gap/2)` 与短休止门限都还在管着。"
        );
        // ⑷ 不许重叠,而且中间整段留给 base。
        assert!(w[0].end < w[1].start, "两条窗不许重叠");
        assert_eq!(w[1].start - w[0].end, 3, "整段 3 帧休止必须留给 base(读到 {})", w[1].start - w[0].end);

        // ⑵ ⛔ 阴性对照:4 帧休止(刚过门限)⇒ 照旧伸出。
        let four = [10i64, 20, 4, 20, 10];
        let w4 = dead_group_windows(&nn, &four, &g, &all_reachable(), 0);
        assert_eq!(
            (w4[0].start, w4[0].end, w4[1].start, w4[1].end),
            (6, 32, 32, 58),
            "4 帧休止不该命中这一刀:`post = min(2,2) = 2` / `pre = min(4,2) = 2`(读到 {w4:?})"
        );
        assert!(
            w4[1].start - w4[0].end < 4,
            "阴性对照有效性:4 帧那一档必须**真的**伸进休止,否则 ⑵ 是恒真的"
        );

        // ⑶ ⛔ `gap == 0` 是 `GUARD_FRAMES` 的地盘,这一刀不许碰。
        let nn0 = [0i64, 85, 85, 0];
        let fr0 = [10i64, 20, 20, 10];
        let g0 = [
            DeadGroup { start: 1, end: 1, shift: -6 },
            DeadGroup { start: 2, end: 2, shift: -7 },
        ];
        let w0 = dead_group_windows(&nn0, &fr0, &g0, &all_reachable(), 0);
        assert_eq!(w0.len(), 2);
        assert!(
            w0[0].end > 30 && w0[1].start < 30,
            "没有休止时护栏必须照常向两侧各伸 GUARD_FRAMES(读到 {w0:?})"
        );
    }

    #[test]
    fn merging_never_deletes_a_rescue_whatever_order_the_plan_arrives_in() {
        let nn = [0, 85, 0, 85, 0];
        let fr = [10i64, 20, 3, 20, 10];
        let asc = [
            DeadGroup { start: 1, end: 1, shift: -6 },
            DeadGroup { start: 3, end: 3, shift: -6 },
        ];
        // 升序:这两条**本来就该**合并成一条 —— 上面那条测试钉的就是它,这里只取它的窗。
        let merged = dead_group_windows(&nn, &fr, &asc, &all_reachable(), 0);
        assert_eq!(merged, vec![DeadJob { shift: -6, start: 6, end: 57 }]); // S163: 55→57，见 `REST_POST_FRAMES`

        // ⭐ 同一批组,**降序**喂进来。两段音频还在原处,所以救援也必须还是两条。
        let desc = [
            DeadGroup { start: 3, end: 3, shift: -6 },
            DeadGroup { start: 1, end: 1, shift: -6 },
        ];
        let got = dead_group_windows(&nn, &fr, &desc, &all_reachable(), 0);
        assert_eq!(
            got.len(),
            2,
            "降序输入把一条救援吞掉了 —— 每条组都必须在输出里留下自己的窗(拿到的是 {got:?})"
        );
        let mut spans: Vec<(i64, i64)> = got.iter().map(|j| (j.start, j.end)).collect();
        spans.sort();
        // ⚠ S159zw —— 坐标从 `(6, 31), (32, 55)` 变成了 `(6, 30), (33, 55)`:这个夹具中间那段
        //    休止**正好 3 帧**,命中 [`SHORT_REST_NO_EXTEND_FRAMES`] ⇒ 两侧的窗不再各伸进 1 帧。
        //    ⛔ 这条判据钉的是「**降序输入不许吞掉一条救援**」(上面那条 `got.len() == 2`),
        //    坐标只是它的载体;**新坐标仍然是「各自在原处、互不重叠」**,不变量一个字没变。
        // ⚠ S163 —— 尾边再从 55 变 57:`REST_POST_FRAMES` 2→4(尾部 gap=10 ⇒ min(4,5)=4)。
        //    内侧的 30/33 一个字节没动(那侧 gap=3，命中短休止门限)，重叠与位置不变量照旧。
        assert_eq!(
            spans,
            vec![(6, 30), (33, 57)],
            "两条窗必须各自还在原来的位置上(合并只许发生在真正紧邻的一对上)"
        );
        assert!(spans[0].1 < spans[1].0, "两条窗不许重叠");

        // ⛔ 阴性对照 ①:**位移不同**的降序对同样不许合并 —— 这一条今天就是对的,
        //    它在这里是为了证明上面那条红不是「降序一律不合并」这句话本身造出来的。
        let desc2 = [
            DeadGroup { start: 3, end: 3, shift: -6 },
            DeadGroup { start: 1, end: 1, shift: -7 },
        ];
        assert_eq!(dead_group_windows(&nn, &fr, &desc2, &all_reachable(), 0).len(), 2);

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
            dead_group_windows(&nn_in, &fr_in, &split, &all_reachable(), 0),
            dead_group_windows(&nn_in, &fr_in, &whole, &all_reachable(), 0),
            "同位移拆分必须重新合并成与不拆时逐位相同的窗"
        );
        assert_eq!(
            dead_group_windows(&nn_in, &fr_in, &whole, &all_reachable(), 0),
            // S163: 尾边 52→54 —— `REST_POST_FRAMES` 2→4（尾部 gap=10 ⇒ min(4,5)=4）。
            // 手算：fr=[10,20,20,10] ⇒ cum=[0,10,30,50,60]；pre=min(4,10/2)=4 ⇒ 10−4=6；
            //       post=min(4,10/2)=4 ⇒ 50+4=54。
            vec![DeadJob { shift: -6, start: 6, end: 54 }],
            "⛔ 期望值写字面量:拿被测函数自己算期望值 = 恒真"
        );
    }

    #[test]
    fn dead_group_windows_extend_into_rests_without_overlap() {
        // cum=[0,5,9,19,32,37,45];前间隙 9 帧→pre=4,后间隙 5 帧→post=2(半间隙封顶)。
        let nn = [0, 0, 73, 85, 0, 73];
        let fr = [5i64, 4, 10, 13, 5, 8];
        let plan = [DeadGroup { start: 2, end: 3, shift: -6 }];
        assert_eq!(dead_group_windows(&nn, &fr, &plan, &all_reachable(), 0), vec![DeadJob { shift: -6, start: 5, end: 34 }]);
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
            dead_group_windows(&nn, &fr, &plan, &all_reachable(), 0),
            vec![DeadJob { shift: -6, start: 25 - 2, end: 35 + 2 }],
            "窗必须伸进两侧的乘客,否则淡化压在被救的死音上"
        );
        // 封顶:邻居只有 2 帧 ⇒ 只能伸 1 帧(半个音),窗永远不许吃掉整个邻居。
        let fr = [5i64, 10, 2, 10, 2, 10, 10];
        assert_eq!(
            dead_group_windows(&nn, &fr, &plan, &all_reachable(), 0),
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
        assert_eq!(damage_from_scan(2.0, 1.00, None, 0.55), 0.0);
        assert_eq!(damage_from_scan(24.0, 0.96, None, 0.55), 0.0);
        // a rejected semitone saturates on pitch alone
        assert_eq!(damage_from_scan(9999.0, 0.0, None, 0.55), 3.0);
        // …and a semitone that keeps perfect pitch while losing voicing is still damaged
        assert!(damage_from_scan(6.0, 0.63, None, 0.55) > 2.0);
    }

    #[test]
    fn the_timbre_dimension_catches_what_f0_cannot() {
        // THE case the whole F1 change exists for, using the numbers measured off the probe wav
        // on disk: akiko MIDI 80 stores err=2 cents / voiced=1.00 — a perfect score on both f0
        // axes — while measuring -7.7 dB with 88.7% of its energy below 1.5*f0.
        assert_eq!(damage_from_scan(2.0, 1.00, None, 0.55), 0.0, "f0-only is blind here");
        assert!(
            damage_from_scan(2.0, 1.00, Some((-7.7, 0.887)), 0.55) > 2.0,
            "with the audio measured, the same semitone reads as badly damaged"
        );
        // a healthy note is still free WITH the dimension present (akiko MIDI 74)
        assert_eq!(damage_from_scan(5.0, 1.00, Some((0.0, 0.109)), 0.55), 0.0);
        // lengv2.3's near-pure-sine 75 (0.983) vs its healthy 74 (0.467)
        assert!(damage_from_scan(6.0, 1.00, Some((-1.2, 0.983)), 0.55) > 2.0);
        assert_eq!(damage_from_scan(8.0, 1.00, Some((-8.0, 0.467)), 0.55), 0.5, "quiet but voiced = graded, not rejected");
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
    fn the_island_dilation_defaults_to_120_ms_and_garbage_never_silently_disables_it() {
        // ⛔ 与 `parse_phase_lock` / `parse_infrasonic_hp` 同一条规矩:**默认值本身要有判据**。
        // 这一条尤其重要,因为这个臂是**开着**的:没有它,「我们翻了」与「有人把它翻回去了」
        // 在别的每一条测试上长得一模一样。
        // ⭐ S160j:30 → 120(用户 2026-08-24 耳判拍板)。30 是 S154 的历史值,理由都在
        //    `BRIDGE_UNVOICED_MS_DEFAULT` 的 doc 上,两段读数都在那里。
        assert_eq!(
            parse_bridge_unvoiced_ms(None),
            120.0,
            "生产默认必须是 120 ms —— 改它要成对 bump RANGE_ALGO_VERSION 与 audition_cache_tag"
        );
        assert_eq!(parse_bridge_unvoiced_ms(None), BRIDGE_UNVOICED_MS_DEFAULT);
        // 显式的 0 必须能关掉它 —— 用户报「新版不对」时要渲得出旧臂(S150 那条)。
        assert_eq!(parse_bridge_unvoiced_ms(Some("0")), 0.0);
        assert_eq!(parse_bridge_unvoiced_ms(Some("60")), 60.0);
        // ⛔ S160j —— 出厂值钉成字面量(不许写成常量自比,那是恒真)。30 是 S154 的历史值。
        assert_eq!(BRIDGE_UNVOICED_MS_DEFAULT, 120.0, "出厂桥接 = 120 ms(S160j,用户耳判拍板)");
        assert_eq!(parse_bridge_unvoiced_ms(Some("30")), 30.0, "S154 那个历史值仍然设得回去");
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
        let want: [(&str, f64); 11] = [
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
            ("UTAI_PSOLA_DEJITTER", parse_dejitter(None)),
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

    /// ⛔⛔ S159zze —— **音高闸:这一遍到底有没有真的把音高移过去。**
    ///
    /// 用喂进去的 f0 轨当真值:每个浊音帧上,把输出在**目标周期** `sr/(f0·ratio)` 与
    /// **输入周期** `sr/f0` 两处的归一化自相关比一比。移调成功 ⇒ 目标那边赢。
    ///
    /// 它存在的理由是一次真实事故:`UTAI_PSOLA_WSOLA=0.15`(它自己 doc 推荐的工作点)
    /// 把移调**整个抵消**掉了 —— 输出逐音低了 **−1228 cents ≈ −13 个半音**,
    /// 而当年守它的两把尺子(陷波率、ΔHNR)**都看不见音高**,于是那个旋钮被登记成
    /// 「一笔单调取舍」。⇒ 这条**无条件打印**,而且打的是**两边的相关**而不是一个标量,
    /// 因为「量不出来」和「移调没发生」必须分得开(S129 铁律)。
    fn probe_pitch_gate(y: &[f32], f0: &[f32], hop: usize, sr: u32, semis: f64) -> (f64, usize) {
        let ratio = 2f64.powf(semis / 12.0);
        let ac = |c: usize, lag: usize, w: usize| -> f64 {
            if lag == 0 || c + w + lag >= y.len() {
                return f64::NAN;
            }
            let (mut num, mut ea, mut eb) = (0.0f64, 0.0f64, 0.0f64);
            for i in 0..w {
                let a = f64::from(y[c + i]);
                let b = f64::from(y[c + i + lag]);
                num += a * b;
                ea += a * a;
                eb += b * b;
            }
            if ea <= 0.0 || eb <= 0.0 { f64::NAN } else { num / (ea * eb).sqrt() }
        };
        let (mut win, mut seen) = (0usize, 0usize);
        for (i, &f) in f0.iter().enumerate() {
            if f <= 20.0 {
                continue;
            }
            let c = i * hop;
            let want = (f64::from(sr) / (f64::from(f) * ratio)).round() as usize;
            let orig = (f64::from(sr) / f64::from(f)).round() as usize;
            if want == orig {
                continue;
            }
            let w = (orig * 3).min(4096);
            let (a, b) = (ac(c, want, w), ac(c, orig, w));
            if !a.is_finite() || !b.is_finite() {
                continue;
            }
            seen += 1;
            if a > b {
                win += 1;
            }
        }
        (if seen == 0 { f64::NAN } else { win as f64 / seen as f64 }, seen)
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
        // 两段浊音,中间 **0.9 s** 真休止(f0 = 0)——「窗切不到任何岛」才有地方落脚。
        // ⛔ S160j:原来是 0.4 s,而 `bridge_unvoiced` 是**膨胀**(每段浊音向外长 `max_ms`,
        //    空隙在中点劈开、至少留一帧清音)。出厂从 30 翻到 120 之后,0.4 s 的空隙只剩
        //    0.4 − 2×0.12 = 0.16 s 没被膨胀,而这条判据的窗**恰好落在中点那一帧上**
        //    ⇒ 它坐在边界上、被一帧的取整推翻。⚠ **这不是行为退化,是夹具本来就脆**:
        //    「窗落在真休止正中」这件事必须在**任何**合理的膨胀宽度下都成立。
        //    0.9 s 留下 0.66 s 的未膨胀区间 ⇒ 与膨胀宽度解耦。
        let seg = inverse_probe_tone(sr, 0.30);
        let gap = vec![0.0f32; (sr as f64 * 0.90) as usize];
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
        let full = apply_inverse_with(InverseEngine::Psola, x.clone(), sr, -6, 0.0, fed, 0.0)
            .expect("整条臂必须成功");
        // ⑴ + ⑵ —— 窗只盖第一段。
        let keep = [(0usize, seg.len())];
        let win = apply_inverse_windowed_with(
            InverseEngine::Psola, x.clone(), sr, -6, 0.0, fed, &keep, 0.0,
        )
        .expect("窗臂必须成功");
        assert_ne!(win, full, "窄窗与整条臂逐位相同 —— `keep` 在某一层被丢掉了,提速是假的");
        assert_eq!(&win[..seg.len()], &full[..seg.len()], "窗内不是逐位相同");
        // ⑶ —— 窗落在休止正中(离两段浊音都够远),必须 Ok 且**原样返回**。
        let mid = seg.len() + gap.len() / 2;
        let empty = apply_inverse_windowed_with(
            InverseEngine::Psola, x.clone(), sr, -6, 0.0, fed,
            &[(mid, mid + hop)], 0.0,
        );
        assert_eq!(empty.as_deref(), Ok(x.as_slice()), "窗切不到岛时必须原样返回,而不是报错");
        // ⛔ 阴性对照:同一条路上「真的没有音高」仍然必须响亮失败 —— 否则 ⑶ 只是把那条闸拆了。
        let silent = vec![0.0f32; f0.len()];
        assert_eq!(
            apply_inverse_windowed_with(
                InverseEngine::Psola, x, sr, -6, 0.0, Some((&silent, hop)), &keep, 0.0,
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
            apply_inverse(x.clone(), sr, 0, DEFAULT_FORMANT_KAPPA, None, 0.0).expect("shift 0");
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
                    0.0,
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
            let got = apply_inverse_with(InverseEngine::Psola, x.clone(), sr, -6, 0.0, arg, 0.0);
            assert_eq!(
                got.err().as_deref(),
                Some("RANGE_INVERSE_NO_PITCH"),
                "{name} must fail loudly, got Ok(..) or the wrong CODE"
            );
        }
        // …and a zero hop is the same class of "I cannot locate the periods".
        let fed = vec![220.0f32; 51];
        assert_eq!(
            apply_inverse_with(InverseEngine::Psola, x.clone(), sr, -6, 0.0, Some((&fed, 0)), 0.0)
                .err()
                .as_deref(),
            Some("RANGE_INVERSE_NO_PITCH")
        );
        // Negative control: the same call WITH pitch must succeed — otherwise the assertions
        // above would pass on a function that always fails.
        assert!(
            apply_inverse_with(InverseEngine::Psola, x, sr, -6, 0.0, Some((&fed, hop)), 0.0).is_ok(),
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
        // ⚠ S159zi/zk/S163 —— 这里绕过两次弯,记下来免得下一个人再绕:
        // S159zi 的二级拆一度把 tight 臂拆成两组(78 只要 −1、85 要 −7);
        // S159zk 的**断点可行性**又把它挡了回去(断在 `[78|85]` 会让浅那一组的窗倒着伸进 85,
        // 而 85 在 −1 上是 MIDI 84,超出这个夹具的 scan(36,80));
        // ⭐⭐ 而 S163 证明**危害的载体是护栏不是断点**([`dead_group_windows_raw`] 现在把
        // 那一侧的护栏收到 0)⇒ 断点重新可行,tight 臂又是两组。
        // ⇒ **这是与主线同一个改进**:78 从「陪着 85 走 −7 ⇒ 落点 **71**」变成
        //    「自己走 −1 ⇒ 落点 **77**」。
        // ⛔ 用 `g85`(按「救 85 的那一组」读)而不是 `[0]`:拆组规则一动,按 `[0]` 读会
        //    **静默**换掉被测的对象 —— 这条判据测的始终是 85 那一组的落点。
        let g85 = |p: &[DeadGroup], who: &str| -> DeadGroup {
            *p.iter().find(|d| (d.start..=d.end).contains(&2)).unwrap_or_else(|| panic!("{who} 必须救 85:{p:?}"))
        };
        assert_eq!(pw.len(), 1, "wide 臂只有 85 是死音 ⇒ 二级无处可拆(读到 {pw:?})");
        assert_eq!(pt.len(), 1, "tight 臂的那一处断点不可行 ⇒ 也是一组(读到 {pt:?})");
        let (w, t) = (g85(&pw, "wide"), g85(&pt, "tight"));
        assert_eq!(
            t.shift, w.shift,
            "the ceiling moved 80→76 and the landing must NOT follow it down (got {} vs {})",
            t.shift, w.shift
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

    /// S159zm —— **只跑拼接**的探针:同一份 base + donor 缓冲,只翻对齐旋钮。
    ///
    /// ⛔ 为什么必须有它:「渲两遍整曲再比」被 donor 路径的**跨进程不可复现性**淹没 ——
    /// 实测两遍差 **85 %** 的样本,而且全曲长时平均谱六档均匀 −0.45 dB(那是**增益差**,
    /// 不是频谱效应)。⇒ 一个只改「两端爬坡」或「淡化对齐」的旋钮,在那种台子上**量不出来**。
    /// ⇒ 把 base 与每一遍 donor 落盘一次,之后所有臂都喂**同一份**输入,
    /// 唯一的差别就是被测的那件事(与 [`inverse_probe`] 同一条理由)。
    ///
    /// ```powershell
    /// $env:UTAI_SPLICE_DIR="…\donor"      # 含 base.f32 / donor_post_<±s>.f32 / total_frames.txt
    /// $env:UTAI_SPLICE_PLAN="…\x.plan.json"
    /// $env:UTAI_SPLICE_ALIGN="2"; $env:UTAI_SPLICE_OUT="…\out.wav"
    /// cargo test --lib inference::vocal_range::tests::splice_probe -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "probe: needs dumped base + donor buffers (set UTAI_SPLICE_*)"]
    fn splice_probe() {
        let dir = std::path::PathBuf::from(std::env::var("UTAI_SPLICE_DIR").expect("UTAI_SPLICE_DIR"));
        let out = std::env::var("UTAI_SPLICE_OUT").expect("UTAI_SPLICE_OUT");
        let planp = std::env::var("UTAI_SPLICE_PLAN").expect("UTAI_SPLICE_PLAN");
        let rd = |p: &std::path::Path| -> Vec<f32> {
            std::fs::read(p)
                .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let mut base = rd(&dir.join("base.f32"));
        let total_frames: i64 =
            std::fs::read_to_string(dir.join("total_frames.txt")).unwrap().trim().parse().unwrap();
        let plan: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&planp).unwrap()).unwrap();
        let jobs_src = plan["windows_frames"].as_array().expect("windows_frames");
        let jobs: Vec<DeadJob> = jobs_src
            .iter()
            .map(|v| DeadJob {
                shift: v[0].as_i64().unwrap(),
                start: v[1].as_i64().unwrap(),
                end: v[2].as_i64().unwrap(),
            })
            .collect();
        let sr: u32 = plan["sample_rate"].as_u64().unwrap() as u32;
        eprintln!("[splice] {} jobs · base {} · sr {sr} · UTAI_RANGE_SEAM_ALIGN = {} ms",
                  jobs.len(), base.len(), seam_align_ms());
        apply_dead_only_windows_with(&mut base, sr, total_frames, &jobs, &[], &[], false, join_rests_enabled(), (seam_align_ms() * f64::from(sr) / 1000.0).round() as usize, landing_harm_eps(), landing_repair_ms(), comb_floor_db(), landing_width_eps(), landing_width_floor(), handover_deficit_db(), tied_xfade_ms(), |s, _own| {
            Ok(rd(&dir.join(format!("donor_post_{s:+}.f32"))))
        })
        .unwrap();
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: sr,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut w = hound::WavWriter::create(&out, spec).unwrap();
        for v in &base {
            w.write_sample(*v).unwrap();
        }
        w.finalize().unwrap();
        eprintln!("[splice] -> {out}");
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
        // S159zzb —— 引擎的诊断(`cola_w_*` / `edge_step_*` / `src_uncovered_frac`)是 tracing::info,
        // 而这个探针以前不装 subscriber ⇒ **每一遍都把它们扔了**。量 OLA 窗和抖不抖要靠它们。
        let _ = tracing_subscriber::fmt().with_max_level(tracing::Level::INFO).try_init();
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
        // ⭐ 探针继续听 `UTAI_RANGE_TILT` —— tilt 的零渲染噪声 A/B 就是从这里跑的。
        let y =
            apply_inverse_with(engine, x, spec.sample_rate, shift, kappa, Some((&f0, hop)), range_tilt())
                .expect("inverse");
        assert_eq!(y.len(), n, "exact-length contract");
        // ⛔ S159zz —— 下面那条 wav 出口是 **PCM_16 + 逐臂峰值归一**:比值型尺子不怕增益,
        // 但量化地板骗过我们一次了(S159z 的「唱音内绝对静音」)。⇒ 想量小差就读这份原始 f32。
        if let Ok(p) = std::env::var("UTAI_INV_DUMP") {
            let mut b = Vec::with_capacity(y.len() * 4);
            for v in &y {
                b.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(&p, b).expect("dump");
        }
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
        let (frac, seen) = probe_pitch_gate(&y, &f0, hop, spec.sample_rate, -(shift as f64));
        println!(
            "inverse_probe: {engine:?} {shift:+} st kappa {kappa} -> {out}\n\
             inverse_probe PITCH GATE: 目标周期赢 {:.1}% / {seen} 帧 \
             ({})",
            frac * 100.0,
            if !frac.is_finite() {
                "⛔ 量不出来(没有可比的浊音帧)—— 这不是通过"
            } else if frac >= 0.80 {
                "✅ 移调发生了"
            } else {
                "⛔⛔ 移调【没有】发生 —— 这条臂的其余读数一律作废"
            }
        );
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
        let a =
            apply_inverse_with(InverseEngine::Psola, x.clone(), sr, -6, 0.0, Some((&fed, hop)), 0.0)
                .expect("psola");
        let b = apply_inverse_with(
            InverseEngine::Signalsmith, x.clone(), sr, -6, 0.0, Some((&fed, hop)), 0.0,
        )
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
