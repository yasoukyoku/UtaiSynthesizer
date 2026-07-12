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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeakerRange {
    /// MIDI bounds, inclusive.
    pub usable: (f32, f32),
    pub comfort: (f32, f32),
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
    Some(SpeakerRange { usable, comfort })
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

/// v1 three-tier decision. `bounds` = the part's effective pitch range in MIDI (transpose
/// already folded in; from the f0 curve when present, else the note pitches). Returns the
/// integer semitone translation to render at (0 = tiers 1/2, no inverse pass), clamped to
/// ±MAX_RANGE_SHIFT (a clamp firing means the record is stale — retest).
pub fn compute_range_shift(bounds: (f32, f32), range: &SpeakerRange) -> i64 {
    let raw = compute_range_shift_raw(bounds, range);
    if raw.abs() > MAX_RANGE_SHIFT {
        tracing::warn!(
            "range-extend: computed shift {raw:+} st exceeds ±{MAX_RANGE_SHIFT} — clamped (stale record? retest the model)"
        );
    }
    raw.clamp(-MAX_RANGE_SHIFT, MAX_RANGE_SHIFT)
}

fn compute_range_shift_raw(bounds: (f32, f32), range: &SpeakerRange) -> i64 {
    let (min_p, max_p) = bounds;
    let (u_lo, u_hi) = range.usable;
    let (c_lo, c_hi) = range.comfort;
    if c_hi - c_lo <= 0.0 {
        return 0; // degenerate comfort never a target (speaker_range heals; defensive here)
    }
    if min_p >= u_lo && max_p <= u_hi {
        return 0; // tiers 1/2 — inside usable renders untouched
    }
    if max_p - min_p > c_hi - c_lo {
        // wider than the comfort zone: best-effort centering (compression not ported)
        let shift = (((c_lo + c_hi) - (min_p + max_p)) / 2.0).round() as i64;
        tracing::warn!(
            "range-extend: part spans {:.1} st > comfort span {:.1} st — centering by {} st (edges stay outside)",
            max_p - min_p,
            c_hi - c_lo,
            shift
        );
        return shift;
    }
    if max_p > c_hi {
        let need = -((max_p - c_hi).ceil() as i64);
        let cushioned = -((max_p - (c_hi - CEILING_MARGIN)).ceil() as i64);
        // take the ceiling margin only while the bottom still fits inside comfort
        if min_p + cushioned as f32 >= c_lo {
            cushioned
        } else {
            need
        }
    } else if min_p < c_lo {
        (c_lo - min_p).ceil() as i64
    } else {
        0 // inside comfort but outside usable is impossible (comfort ⊆ usable); defensive
    }
}

/// Effective pitch bounds of a score render in MIDI: voiced f0-curve extremes when the DAW
/// supplies Option-A f0 (cents/100), else the note pitches (`note_nums`, rests ≤ 0);
/// `transpose` folded in. None when nothing voiced (all-rest part → no shift).
pub fn score_pitch_bounds(
    note_nums: &[i64],
    f0_cents: &[f32],
    f0_voiced: &[u8],
    transpose: i64,
) -> Option<(f32, f32)> {
    let mut lo = f32::MAX;
    let mut hi = f32::MIN;
    if !f0_cents.is_empty() {
        for (i, &c) in f0_cents.iter().enumerate() {
            if f0_voiced.get(i).copied().unwrap_or(0) == 0 {
                continue;
            }
            let m = c / 100.0;
            lo = lo.min(m);
            hi = hi.max(m);
        }
    } else {
        for &n in note_nums {
            if n > 0 {
                lo = lo.min(n as f32);
                hi = hi.max(n as f32);
            }
        }
    }
    (lo <= hi).then(|| (lo + transpose as f32, hi + transpose as f32))
}

/// Whole-signal shift decision for the COVER/audition path (S60d2 — frame-mass optimizer).
///
/// The previous p02/p98-bounds version had two blind spots the user could HEAR: the top 2%
/// of frames (= seconds of the climax on a full song) stayed above the ceiling by
/// construction, and the minimal translation parked the material top exactly ON c_hi with
/// zero headroom. Instead: brute-force the integer translation in ±MAX_RANGE_SHIFT that
/// minimizes the sung-outside-the-zone frame mass, where
///   - frames above (c_hi - CEILING_MARGIN) weigh 3× frames below c_lo (top overflow =
///     saturation/mute, the audible disaster; bottom overflow = fry, degrades softer),
///   - a small |shift| penalty (0.003/st) breaks near-ties toward the least PSOLA coloration.
/// Isolated rmvpe spikes are inherently harmless (tiny mass never outweighs the shift
/// penalty). A piece entirely inside USABLE renders untouched (tiers 1/2, byte-identical).
pub fn piece_range_shift(f0_hz: &[f32], range: &SpeakerRange) -> i64 {
    let voiced: Vec<f32> = f0_hz
        .iter()
        .filter(|&&v| v > 0.0)
        .map(|&hz| 69.0 + 12.0 * (hz / 440.0).log2())
        .collect();
    if voiced.len() < 10 {
        return 0; // too little voiced material to judge — render untouched
    }
    let (u_lo, u_hi) = range.usable;
    let (c_lo, c_hi) = range.comfort;
    if c_hi - c_lo <= 0.0 {
        return 0; // degenerate comfort never a target (speaker_range heals; defensive)
    }
    if voiced.iter().all(|&m| m >= u_lo && m <= u_hi) {
        return 0; // tiers 1/2 — the whole piece sits in the proven zone
    }
    let top = c_hi - CEILING_MARGIN;
    let n = voiced.len() as f32;
    let mut best_cost = f32::MAX;
    let mut best_shift = 0i64;
    for s in -MAX_RANGE_SHIFT..=MAX_RANGE_SHIFT {
        let sf = s as f32;
        let mut above = 0usize;
        let mut below = 0usize;
        for &m in &voiced {
            if m + sf > top {
                above += 1;
            } else if m + sf < c_lo {
                below += 1;
            }
        }
        let cost = (3.0 * above as f32 + below as f32) / n + 0.003 * sf.abs();
        if cost < best_cost {
            best_cost = cost;
            best_shift = s;
        }
    }
    best_shift
}

/// Undo a chunk's range shift in the audio domain: TD-PSOLA at constant ratio, guided by
/// the FED f0 (`f0_hz`, one frame per `hop` output samples — the SoVITS hop grid or RVC's
/// sr/100). shift 0 ⇒ untouched.
pub fn psola_inverse_hop(
    audio: Vec<f32>,
    f0_hz: &[f32],
    hop: usize,
    sample_rate: u32,
    shift: i64,
) -> Vec<f32> {
    if shift == 0 || audio.is_empty() || f0_hz.is_empty() {
        return audio;
    }
    let ratio = vec![2f32.powf(-(shift as f32) / 12.0); f0_hz.len()];
    utai_dsp::psola_shift(&audio, sample_rate, f0_hz, &ratio, utai_dsp::PsolaParams { hop })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn range() -> SpeakerRange {
        SpeakerRange { usable: (48.0, 84.0), comfort: (52.0, 79.0) }
    }

    #[test]
    fn tier12_no_shift() {
        // inside comfort
        assert_eq!(compute_range_shift((55.0, 70.0), &range()), 0);
        // outside comfort but inside usable — render as-is, skip the inverse (v1 tier 2)
        assert_eq!(compute_range_shift((49.0, 82.0), &range()), 0);
        assert_eq!(compute_range_shift((48.0, 84.0), &range()), 0);
    }

    #[test]
    fn tier3_minimal_translation() {
        // above usable, bottom too tight for the ceiling margin (60-9=51 < c_lo 52) →
        // plain minimal translation to c_hi
        assert_eq!(compute_range_shift((60.0, 86.0), &range()), -7);
        // below usable → shift UP into comfort
        assert_eq!(compute_range_shift((40.0, 60.0), &range()), 12);
        // bottom has room (60-8=52 ≥ c_lo) → the CEILING_MARGIN cushion is taken:
        // top lands at 77 (= c_hi 79 - 2), not hugging the boundary
        assert_eq!(compute_range_shift((60.0, 84.5), &range()), -8);
    }

    #[test]
    fn wider_than_comfort_centers() {
        let s = compute_range_shift((30.0, 90.0), &range());
        // centered: midpoint 60 → comfort midpoint 65.5 → +5/+6
        assert!((5..=6).contains(&s), "got {s}");
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
    fn shift_is_clamped_to_max() {
        // material 31 st above the comfort ceiling → raw -31, brake at -24
        assert_eq!(compute_range_shift((100.0, 110.0), &range()), -24);
        // degenerate comfort (defensive; speaker_range normally heals first) → 0
        let degenerate = SpeakerRange { usable: (48.0, 84.0), comfort: (52.0, 52.0) };
        assert_eq!(compute_range_shift((90.0, 100.0), &degenerate), 0);
    }

    fn hz(midi: f32) -> f32 {
        440.0 * 2f32.powf((midi - 69.0) / 12.0)
    }

    #[test]
    fn piece_optimizer_counts_the_climax_mass() {
        // the S60d2 field case shape: 96% of frames at 67, 4% climax at 75 — a p98-bounds
        // decision ignored the climax entirely; the optimizer must land it under the
        // margined ceiling (70-2=68): 75-7=68 ⇒ exactly -7, no more (|shift| penalty)
        let r = SpeakerRange { usable: (42.0, 70.0), comfort: (42.0, 70.0) };
        let mut f0 = vec![hz(67.0); 960];
        f0.extend(vec![hz(75.0); 40]);
        assert_eq!(piece_range_shift(&f0, &r), -7);
    }

    #[test]
    fn piece_optimizer_ignores_isolated_spikes_and_proven_zone() {
        let r = SpeakerRange { usable: (42.0, 70.0), comfort: (42.0, 70.0) };
        // 3 spike frames at 90 can't justify dragging 1000 frames below the floor → 0
        let mut f0 = vec![hz(60.0); 1000];
        f0.extend(vec![hz(90.0); 3]);
        assert_eq!(piece_range_shift(&f0, &r), 0);
        // entirely inside usable → untouched (tiers 1/2 preserved, byte-identical path)
        assert_eq!(piece_range_shift(&vec![hz(65.0); 500], &r), 0);
    }

    #[test]
    fn piece_optimizer_shifts_low_material_up() {
        let r = SpeakerRange { usable: (48.0, 84.0), comfort: (52.0, 79.0) };
        let f0: Vec<f32> = (0..500).map(|i| hz(30.0 + (i % 11) as f32)).collect();
        // lowest frames at 30 must clear c_lo 52 → +22; the |shift| penalty stops there
        assert_eq!(piece_range_shift(&f0, &r), 22);
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
