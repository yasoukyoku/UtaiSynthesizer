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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeakerRange {
    /// MIDI bounds, inclusive.
    pub usable: (f32, f32),
    pub comfort: (f32, f32),
}

/// Parse the sidecar `vocal_range` record for one speaker id (falls back to speaker "0"
/// when the exact id has no record — single-speaker models always test as 0).
pub fn speaker_range(config: &ModelConfig, speaker_id: u32) -> Option<SpeakerRange> {
    let rec = config.extra.get("vocal_range")?;
    let speakers = rec.get("speakers")?;
    let sp = speakers
        .get(speaker_id.to_string())
        .or_else(|| speakers.get("0"))?;
    let pair = |key: &str| -> Option<(f32, f32)> {
        let v = sp.get(key)?.as_array()?;
        let lo = v.first()?.as_f64()? as f32;
        let hi = v.get(1)?.as_f64()? as f32;
        (lo <= hi).then_some((lo, hi))
    };
    let usable = pair("usable")?;
    let comfort = pair("comfort")?;
    Some(SpeakerRange { usable, comfort })
}

// NOTE: "which speaker governs a blend" = the existing ①c `crate::inference::dominant_speaker`
// (max-weight entry, else speaker_id) — reused, NOT re-implemented here (NO-dup).

/// v1 three-tier decision. `bounds` = the part's effective pitch range in MIDI (transpose
/// already folded in; from the f0 curve when present, else the note pitches). Returns the
/// integer semitone translation to render at (0 = tiers 1/2, no inverse pass).
pub fn compute_range_shift(bounds: (f32, f32), range: &SpeakerRange) -> i64 {
    let (min_p, max_p) = bounds;
    let (u_lo, u_hi) = range.usable;
    let (c_lo, c_hi) = range.comfort;
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
        -((max_p - c_hi).ceil() as i64)
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

/// Chunk-level tier decision for the COVER path: bounds from the voiced f0's p02/p98
/// percentiles (rmvpe outlier frames must not trigger a whole-piece shift), in MIDI.
/// The silence-sliced piece is the natural phrase unit — a constant-ratio inverse's
/// seams land in silence between pieces.
pub fn piece_range_shift(f0_hz: &[f32], range: &SpeakerRange) -> i64 {
    let mut voiced: Vec<f32> = f0_hz.iter().copied().filter(|&v| v > 0.0).collect();
    if voiced.len() < 10 {
        return 0; // too little voiced material to judge — render untouched
    }
    voiced.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pick = |q: f64| voiced[((voiced.len() - 1) as f64 * q).round() as usize];
    let midi = |hz: f32| 69.0 + 12.0 * (hz / 440.0).log2();
    compute_range_shift((midi(pick(0.02)), midi(pick(0.98))), range)
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
        // one note above usable → shift DOWN just enough to reach comfort's ceiling
        assert_eq!(compute_range_shift((60.0, 86.0), &range()), -7);
        // below usable → shift UP into comfort
        assert_eq!(compute_range_shift((40.0, 60.0), &range()), 12);
        // fractional excess rounds AWAY from the edge (ceil)
        assert_eq!(compute_range_shift((60.0, 84.5), &range()), -6);
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
        // unknown speaker falls back to speaker 0's record
        assert_eq!(speaker_range(&config, 3), Some(r));
    }

}
