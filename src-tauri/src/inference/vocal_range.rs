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
pub const MIN_VIOLATION_RUN: usize = 25;
/// Weight of a frame inside PROVEN usable but above the margined comfort ceiling. Tier-2
/// semantics ("inside usable renders untouched") mean these must never trigger a shift on
/// their own — but when a shift happens anyway they still nudge the optimizer to land the
/// material under the margin. Mirror band below comfort weighs less (fry degrades softer).
const BOUNDARY_WEIGHT_TOP: f32 = 0.3;
const BOUNDARY_WEIGHT_BOTTOM: f32 = 0.1;

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

/// Phantom-island mask: a run of consecutive out-of-USABLE frames shorter than
/// MIN_VIOLATION_RUN is detector noise (octave-read breaths, fry blips), not singing — it must
/// neither defeat the byte-identical short-circuit nor add rescue mass to the optimizer.
/// In-usable frames are always kept; sustained violations (a real climax) are always kept.
fn phantom_kept_mask(seq: &[f32], u_lo: f32, u_hi: f32) -> Vec<bool> {
    let mut kept = vec![true; seq.len()];
    let mut i = 0;
    while i < seq.len() {
        let out = seq[i] < u_lo || seq[i] > u_hi;
        let mut j = i;
        while j < seq.len() && ((seq[j] < u_lo || seq[j] > u_hi) == out) {
            j += 1;
        }
        if out && j - i < MIN_VIOLATION_RUN {
            kept[i..j].fill(false);
        }
        i = j;
    }
    kept
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
///   - a small |shift| penalty (0.003/st) breaks near-ties toward the least PSOLA coloration.
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
pub fn piece_range_shift(f0_hz: &[f32], energy: Option<&[f32]>, range: &SpeakerRange) -> i64 {
    let mut midis: Vec<f32> = Vec::with_capacity(f0_hz.len());
    let mut weights: Vec<f32> = Vec::with_capacity(f0_hz.len());
    for (i, &v) in f0_hz.iter().enumerate() {
        if v <= 0.0 {
            continue;
        }
        midis.push(69.0 + 12.0 * (v / 440.0).log2());
        weights.push(energy.and_then(|e| e.get(i)).copied().unwrap_or(1.0));
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
    let kept = phantom_kept_mask(&filtered, u_lo, u_hi);
    let dropped = kept.iter().filter(|&&k| !k).count();
    let voiced: Vec<(f32, f32)> = filtered
        .iter()
        .zip(&weights)
        .zip(&kept)
        .filter(|(_, &k)| k)
        .map(|((&m, &w), _)| (m, w))
        .collect();
    if voiced.len() < 10 {
        return 0;
    }
    if voiced.iter().all(|&(m, _)| m >= u_lo && m <= u_hi) {
        if dropped > 0 {
            tracing::info!(
                "range-extend: {dropped} phantom out-of-range frame(s) ignored (detector noise) — piece is in-range, rendering untouched"
            );
        }
        return 0; // tiers 1/2 — the whole piece sits in the proven zone
    }
    let top = c_hi - CEILING_MARGIN;
    let n: f32 = voiced.iter().map(|&(_, w)| w).sum();
    let frame_mass = |sf: f32| -> f32 {
        let mut mass = 0f32;
        for &(m, w) in &voiced {
            let p = m + sf;
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
    let mut best_cost = f32::MAX;
    let mut best_shift = 0i64;
    for s in -MAX_RANGE_SHIFT..=MAX_RANGE_SHIFT {
        let sf = s as f32;
        let cost = frame_mass(sf) + 0.003 * sf.abs();
        if cost < best_cost {
            best_cost = cost;
            best_shift = s;
        }
    }
    if best_shift != 0 {
        // One-line decision audit: enough to reconstruct WHY from a user log (S62b lesson —
        // the -6/-9 mystery took a session; with this it is one grep).
        let above0: f32 = voiced.iter().filter(|&&(m, _)| m > u_hi).map(|&(_, w)| w).sum::<f32>() / n * 100.0;
        let below0: f32 = voiced.iter().filter(|&&(m, _)| m < u_lo).map(|&(_, w)| w).sum::<f32>() / n * 100.0;
        tracing::info!(
            "range-extend optimizer: shift {best_shift:+} st (frames={}, phantom-dropped={dropped}, loudness-weighted at 0: {above0:.1}% above-usable / {below0:.1}% below; cost {:.4} -> {best_cost:.4})",
            voiced.len(),
            frame_mass(0.0)
        );
    }
    best_shift
}

/// Voiced-aware median-5 over a PSOLA guide track (zeros = unvoiced, preserved verbatim; a
/// window with fewer than 3 voiced samples keeps the original value). An octave-misread
/// breath/sibilant frame halves the analysis-mark spacing for exactly that frame and POPS in
/// the inverse (S62c 破音 — model conditioning sees the same spike with extension OFF, so the
/// pop was an extension-ON-only artifact). Legitimate sustained octave jumps (>2 frames) pass
/// through; smooth parametric guides (vocal score path) are a near-identity under a median.
fn sanitize_guide(f0: &[f32]) -> Vec<f32> {
    let n = f0.len();
    let mut out = f0.to_vec();
    for i in 0..n {
        if f0[i] <= 0.0 {
            continue;
        }
        let lo = i.saturating_sub(2);
        let hi = (i + 3).min(n);
        let mut w: Vec<f32> = f0[lo..hi].iter().copied().filter(|&v| v > 0.0).collect();
        if w.len() < 3 {
            continue;
        }
        w.sort_by(|a, b| a.total_cmp(b));
        out[i] = w[w.len() / 2];
    }
    out
}

/// Undo a chunk's range shift in the audio domain: TD-PSOLA at constant ratio, guided by
/// the FED f0 (`f0_hz`, one frame per `hop` output samples — the SoVITS hop grid or RVC's
/// sr/100), spike-sanitized first (see sanitize_guide). shift 0 ⇒ untouched.
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
    let guide = sanitize_guide(f0_hz);
    let ratio = vec![2f32.powf(-(shift as f32) / 12.0); guide.len()];
    utai_dsp::psola_shift(&audio, sample_rate, &guide, &ratio, utai_dsp::PsolaParams { hop })
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
        assert_eq!(piece_range_shift(&f0, None, &r), -7);
    }

    #[test]
    fn piece_optimizer_ignores_isolated_spikes_and_proven_zone() {
        let r = SpeakerRange { usable: (42.0, 70.0), comfort: (42.0, 70.0) };
        // 3 spike frames at 90 can't justify dragging 1000 frames below the floor → 0
        let mut f0 = vec![hz(60.0); 1000];
        f0.extend(vec![hz(90.0); 3]);
        assert_eq!(piece_range_shift(&f0, None, &r), 0);
        // entirely inside usable → untouched (tiers 1/2 preserved, byte-identical path)
        assert_eq!(piece_range_shift(&vec![hz(65.0); 500], None, &r), 0);
    }

    #[test]
    fn piece_optimizer_ignores_phantom_octave_bursts() {
        // S62b field case shape (lengv2.3, HEALTHY record [36,77]): melody well in range,
        // sustained boundary chorus at 76.5 (PROVEN usable), plus scattered short bursts of
        // octave-doubled breath frames at 81. The whole song must render UNTOUCHED — the old
        // code let one phantom frame defeat the tier-2 short-circuit and then chased the
        // burst mass down -6 st, recoloring the entire render.
        let r = SpeakerRange { usable: (36.0, 77.0), comfort: (36.0, 77.0) };
        let mut f0: Vec<f32> = Vec::new();
        for i in 0..20000 {
            f0.push(hz(55.0 + (i % 17) as f32)); // 55..71 melody mass
        }
        f0.extend(vec![hz(76.5); 800]); // boundary chorus — inside usable
        for _ in 0..5 {
            f0.extend(vec![hz(81.0); 10]); // phantom octave burst (~100 ms)
            f0.extend(vec![hz(65.0); 200]);
        }
        assert_eq!(piece_range_shift(&f0, None, &r), 0);
        // the minimal version: a SINGLE spike frame must not defeat the short-circuit either
        let mut f1 = vec![hz(60.0); 5000];
        f1.push(hz(84.0));
        assert_eq!(piece_range_shift(&f1, None, &r), 0);
    }

    #[test]
    fn piece_optimizer_still_rescues_sustained_true_high() {
        // A genuinely out-of-range SUSTAINED section (4 s continuous at 80 > u_hi 77) still
        // triggers the rescue — spike hygiene must not neuter the feature. 80 + s ≤ top (75)
        // ⇒ exactly -5 (the |shift| penalty stops there).
        let r = SpeakerRange { usable: (36.0, 77.0), comfort: (36.0, 77.0) };
        let mut f0 = vec![hz(60.0); 1000];
        f0.extend(vec![hz(80.0); 400]);
        assert_eq!(piece_range_shift(&f0, None, &r), -5);
    }

    #[test]
    fn piece_optimizer_shifts_low_material_up() {
        let r = SpeakerRange { usable: (48.0, 84.0), comfort: (52.0, 79.0) };
        // 50-frame plateaus stepping 30..40 — real f0 is smooth at the 100 fps grid, so the
        // material must be median-filter-stable (a per-frame sawtooth is not a voice).
        let f0: Vec<f32> = (0..550).map(|i| hz(30.0 + ((i / 50) % 11) as f32)).collect();
        // lowest frames at 30 must clear c_lo 52 → +22; the |shift| penalty stops there
        assert_eq!(piece_range_shift(&f0, None, &r), 22);
    }

    #[test]
    fn piece_optimizer_discounts_quiet_phantom_sustains() {
        // S62c field case: a separated stem's reverb tail / harmony bleed reads high for
        // SECONDS (defeats the run filter) but sits far below the lead's level — loudness
        // weighting must keep the piece untouched…
        let r = SpeakerRange { usable: (36.0, 77.0), comfort: (36.0, 77.0) };
        let mut f0 = vec![hz(60.0); 5000];
        let mut en = vec![0.3f32; 5000];
        f0.extend(vec![hz(81.0); 200]); // 2 s sustained phantom above usable
        en.extend(vec![0.01f32; 200]); // …at tail energy (~-30 dB vs lead)
        assert_eq!(piece_range_shift(&f0, Some(&en), &r), 0);
        // …while a LOUD sustained true-high still rescues (same shape as the None test).
        let mut f1 = vec![hz(60.0); 1000];
        let mut e1 = vec![0.25f32; 1000];
        f1.extend(vec![hz(80.0); 400]);
        e1.extend(vec![0.35f32; 400]);
        assert_eq!(piece_range_shift(&f1, Some(&e1), &r), -5);
    }

    #[test]
    fn guide_sanitizer_snaps_isolated_octave_spikes() {
        // 220 Hz steady voice with a 2-frame octave misread and interleaved unvoiced gaps:
        // the spike snaps to the local median, zeros stay zero (unvoiced passes through dry).
        let mut g = vec![220.0f32; 20];
        g[7] = 440.0;
        g[8] = 440.0;
        g[3] = 0.0;
        let s = sanitize_guide(&g);
        assert_eq!(s[7], 220.0);
        assert_eq!(s[8], 220.0);
        assert_eq!(s[3], 0.0);
        // a SUSTAINED legitimate octave jump (>2 frames) must survive
        let mut j = vec![220.0f32; 10];
        j.extend(vec![440.0f32; 10]);
        let s = sanitize_guide(&j);
        assert_eq!(s[15], 440.0);
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
