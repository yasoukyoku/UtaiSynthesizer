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
    candidates.into_iter().find(|&s| {
        dead.iter().all(|&p| range.slot_landing_ok(p + s))
            && all.iter().all(|&p| range.slot_singable(p + s))
    })
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
