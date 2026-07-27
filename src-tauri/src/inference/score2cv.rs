//! ScoreToCV — "自己唱" (score → ContentVec) preprocessing + inference (S48 Phase 1).
//!
//! A faithful Rust port of the deterministic frontend in
//! `D:\MyDev\Much-Better-S2H\scripts\render_ust.py` (`lyric_to_phones` / `split_dur` / `build_arrays`
//! + SP-boundary chunking + per-chunk rebase). The big lookup tables (210-token IPA vocab + the JA
//! kana/romaji→IPA G2P) live in the GENERATED `score2cv_tables.rs` (dumped from the model repo), and the
//! `#[test] build_arrays_matches_python` proves every array is bit-for-bit identical to the Python on a
//! fixed score — the Phase 1c gate. Replaces the dead double-head `s2h` stub (wrong contract, pre-S35).
//!
//! The model produces ONLY content (cv[T,D], D=768 SoVITS4.1/RVCv2 or 256 SoVITS4.0). f0 is a separate
//! DAW-side stream; pitch/loudness/timbre are NOT in cv. Deploy is always B=1, never padded.

use std::collections::HashMap;
use std::sync::OnceLock;

use ndarray::Array2;

use super::engine::{InputTensor, OnnxEngine};
use super::g2p;
use super::score2cv_tables as tbl;
use crate::{Result, UtaiError};

// ─── table accessors (built once from the generated const slices) ───────────────────────────────

fn phone_to_id_map() -> &'static HashMap<&'static str, i64> {
    static M: OnceLock<HashMap<&'static str, i64>> = OnceLock::new();
    M.get_or_init(|| tbl::PHONE_TO_ID.iter().copied().collect())
}
fn kana_map() -> &'static HashMap<&'static str, &'static str> {
    static M: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
    // base table + the S58 coverage additions (missing yōon rows + ゔ; generated, non-colliding).
    M.get_or_init(|| tbl::KANA.iter().chain(super::g2p_tables::KANA_EXTRA).copied().collect())
}
fn r2ipa_map() -> &'static HashMap<&'static str, &'static [&'static str]> {
    static M: OnceLock<HashMap<&'static str, &'static [&'static str]>> = OnceLock::new();
    M.get_or_init(|| tbl::R2IPA.iter().chain(super::g2p_tables::R2IPA_EXTRA).copied().collect())
}

// ─── G2P: one lyric token → IPA phones / rest / sustain ─────────────────────────────────────────

/// Outcome of `lyric_to_phones` — mirrors the Python `(phones, is_rest, is_sustain)` triple, with
/// `Unknown` for an OOV lyric (the DAW must LOUD-error, never the reference's silent SP fallback).
enum Lyric {
    Rest,
    /// A breath token (`AP`/`ap`) — an audible intake (`AP` id), NOT silence (M3, §11.3). Unvoiced.
    Breath,
    Sustain,
    Phones(Vec<&'static str>),
    Unknown,
}

/// Port of `render_ust.lyric_to_phones`. NB: kana are multi-byte, so the `s[:2]` / `s[0]` lookups are
/// CHAR-based (`.chars()`), never byte slicing.
fn lyric_to_phones(lyr: &str) -> Lyric {
    let s0 = lyr.trim();
    if matches!(s0, "R" | "r" | "" | "rest" | "sil" | "pau") {
        return Lyric::Rest;
    }
    // M3 breath: the CANONICAL inhale token → the `AP` phone. Only `AP`/`ap` (the vocab's own breath
    // token, never a sung phoneme) are hard-wired; the DAW lets the user pick a convenient trigger that the
    // frontend maps to `AP` (VocalTrackParams.breathToken), so a common glyph is never stolen for breath.
    if matches!(s0, "AP" | "ap") {
        return Lyric::Breath;
    }
    if matches!(s0, "-" | "ー" | "+") {
        return Lyric::Sustain;
    }
    if matches!(s0, "っ" | "cl" | "q") {
        return Lyric::Phones(vec!["ʔ"]);
    }
    // ── UTAI EXTENSION (S69, beyond render_ust.py): UTAU-convention foreign-sound kana (外来拗音).
    // Community report: すぃ/すぇ fell through the lossy first-char fallback below and sang as す.
    // Checked BEFORE the kana chain; fires ONLY for 「base+small-vowel」/the explicit vowel-onset
    // rows — strings no generated table contains, so parity inputs (3279 golden vectors) never
    // reach it and the upstream mapping stays byte-identical.
    if let Some(v) = foreign_kana_phones(s0) {
        return Lyric::Phones(v);
    }
    // kana → romaji: whole string, else first 2 chars, else first char (if/elif chain — one branch).
    let kana = kana_map();
    let s: String = if let Some(&r) = kana.get(s0) {
        r.to_string()
    } else {
        let chars: Vec<char> = s0.chars().collect();
        let two: String = chars.iter().take(2).collect();
        let one: String = chars.iter().take(1).collect();
        if chars.len() >= 2 && kana.contains_key(two.as_str()) {
            kana[two.as_str()].to_string()
        } else if kana.contains_key(one.as_str()) {
            kana[one.as_str()].to_string()
        } else {
            s0.to_string()
        }
    };
    let s = s.to_lowercase();
    let r2ipa = r2ipa_map();
    if let Some(&seq) = r2ipa.get(s.as_str()) {
        return Lyric::Phones(seq.to_vec());
    }
    let sc: Vec<char> = s.chars().collect();
    // geminate: doubled leading consonant (tta/kke/ssa) = っ(ʔ) + the mora.
    if sc.len() >= 3 && sc[0] == sc[1] {
        let rest: String = sc.iter().skip(1).collect();
        if let Some(&seq) = r2ipa.get(rest.as_str()) {
            let mut v = vec!["ʔ"];
            v.extend_from_slice(seq);
            return Lyric::Phones(v);
        }
    }
    // tchi → っ ち
    if s.starts_with("tch") {
        let rest: String = format!("ch{}", sc.iter().skip(3).collect::<String>());
        if let Some(&seq) = r2ipa.get(rest.as_str()) {
            let mut v = vec!["ʔ"];
            v.extend_from_slice(seq);
            return Lyric::Phones(v);
        }
    }
    Lyric::Unknown
}

/// Small-vowel kana → the vowel IPA it substitutes (外来拗音 second element).
const SMALL_VOWEL_IPA: &[(char, &'static str)] = &[('ぁ', "a"), ('ぃ', "i"), ('ぅ', "ɯ"), ('ぇ', "e"), ('ぉ', "o")];

/// Vowel-onset foreign rows the generic vowel-swap can't derive (う/い have no consonant onset):
/// the UTAU convention reads them as w-/y-glide syllables. ゔ行 is NOT here — base ゔ ([v ɯ], S58
/// KANA_EXTRA) goes through the generic rule like any consonant kana.
const FOREIGN_KANA_EXPLICIT: &[(&str, &[&'static str])] =
    &[("うぃ", &["w", "i"]), ("うぇ", &["w", "e"]), ("うぉ", &["w", "o"]), ("いぇ", &["j", "e"])];

/// UTAI EXTENSION (S69): resolve a UTAU-convention foreign-sound kana (外来拗音) lyric, or None to
/// fall through to the legacy chain. Generic rule: 「base kana + small vowel ぁぃぅぇぉ」 = the
/// base's onset + the small vowel — the base resolves through the UNTOUCHED generated tables
/// (kana→romaji→IPA), then the final vowel is swapped (all-IPA level, so palatalized onsets come
/// out right for free: しぇ→[ɕ e], ちぇ→[tɕ e], てぃ→[t i], ふぁ→[ɸ a], つぁ→[ts a], すぃ→[s i],
/// ゔぇ→[v e]…). Bases whose IPA doesn't end in a plain vowel (ん…) return None. NB the romaji
/// spelling "si" stays Kunrei-shiki ɕi as upstream defined it — kana すぃ is the true /si/, that
/// distinction is exactly the UTAU convention. Small-ya combos (てゅ) and katakana forms are out of
/// scope here (katakana is folded to hiragana upstream in g2p::fold_katakana, so スィ arrives as すぃ).
fn foreign_kana_phones(s0: &str) -> Option<Vec<&'static str>> {
    if let Some(&(_, seq)) = FOREIGN_KANA_EXPLICIT.iter().find(|&&(k, _)| k == s0) {
        return Some(seq.to_vec());
    }
    let chars: Vec<char> = s0.chars().collect();
    if chars.len() < 2 {
        return None;
    }
    let last = *chars.last().unwrap();
    let &(_, small_ipa) = SMALL_VOWEL_IPA.iter().find(|&&(c, _)| c == last)?;
    let base: String = chars[..chars.len() - 1].iter().collect();
    let romaji = kana_map().get(base.as_str())?;
    let seq = r2ipa_map().get(*romaji)?;
    let (&tail, head) = seq.split_last()?;
    if !tbl::VOWEL_SET.contains(&tail) {
        return None; // no plain-vowel tail to swap (ん etc.) — legacy chain decides
    }
    let mut v: Vec<&'static str> = head.to_vec();
    v.push(small_ipa);
    Some(v)
}

/// Public classification of ONE lyric token for the frontend (§9.5 single Rust classifier: the editor's
/// rest/sustain/OOV verdict MUST equal the render's — no JS dictionary copy that drifts from
/// `lyric_to_phones`). Serialized as `{kind:"rest"|"sustain"|"phones"|"unknown", phones?:[…]}`.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum LyricClass {
    /// A rest token (`R`/`r`/``/`rest`/`sil`/`pau`) — silence, no phones.
    Rest,
    /// A breath token (`AP`/`ap`) — an audible inhale (`AP`), unvoiced (M3). The editor may show it
    /// distinctly from a rest; the render emits the `AP` phone (not silence). The DAW maps a user-chosen
    /// trigger (VocalTrackParams.breathToken) to `AP` before classifying, so a common glyph isn't stolen.
    Breath,
    /// A sustain token (`-`/`ー`/`+`) — continues the previous vowel (承前元音 legato).
    Sustain,
    /// A pronounceable lyric → its IPA phones (all in the 210-token vocab).
    Phones { phones: Vec<&'static str> },
    /// OOV — no G2P mapping. The editor must LOUD-mark it (never silent SP), the render LOUD-errors.
    Unknown,
}

/// Classify one lyric token via the SAME `lyric_to_phones` the render uses (owned result for the wire).
pub fn classify_lyric(lyr: &str) -> LyricClass {
    match lyric_to_phones(lyr) {
        Lyric::Rest => LyricClass::Rest,
        Lyric::Breath => LyricClass::Breath,
        Lyric::Sustain => LyricClass::Sustain,
        Lyric::Phones(ph) => LyricClass::Phones { phones: ph },
        Lyric::Unknown => LyricClass::Unknown,
    }
}

/// Port of `render_ust.split_dur`: distribute a note's frames across its phones — each leading consonant
/// gets ≤4 frames, the (final) vowel gets the remainder. `n` = phone count.
fn split_dur(fr: i64, n: usize) -> Vec<i64> {
    if n <= 1 {
        return vec![fr.max(1)];
    }
    let c = (fr / (n as i64 + 1)).max(1).min(4);
    let mut v = vec![c; n - 1];
    v.push((fr - c * (n as i64 - 1)).max(1));
    v
}

// ─── build_arrays: score (lyric, note, frames) → the model's per-phone arrays ────────────────────

/// The per-phone arrays a chunk feeds to ScoreToCV. `phon` (the IPA strings) is kept so chunking can
/// split on the "SP" token exactly like the Python (never on the id, which an OOV fallback could alias).
/// `lang` (S58) is the per-phone RUN language id (uniform within a note; sustains inherit the carrier,
/// rests attach to the previous run) — chunking cuts at every lang change so each ScoreToCV call gets a
/// single-language chunk (the model's lang_id is a per-call scalar).
pub struct ScoreArrays {
    pub phonemes: Vec<i64>,
    pub phone_dur: Vec<i64>,
    pub note_pitch: Vec<i64>,
    pub note_dur: Vec<i64>,
    pub note_to_phone: Vec<i64>,
    pub phon: Vec<&'static str>,
    pub lang: Vec<i64>,
    /// S83: source EVENT index (into the input `score` slice) each phone was emitted from — the phoneme
    /// lane maps phones back to notes/rests through it (a zh same-pitch hold that merely EXTENDS the
    /// previous entry emits no phone of its own, so its frames show as the carrier stretching through).
    pub evt: Vec<usize>,
}

/// S69 R0b①: whether a vocab phone token is VOICELESS (no vocal-fold vibration) — voiceless
/// obstruents (incl. aspirated/tense/long/unreleased/labialized/palatalized variants and
/// affricates) plus the JA devoiced vowels (i̥/ɨ̥/ɯ̥, ring-below U+0325). Used by the score render
/// to zero f0 on these frames so the SVC feed matches the COVER path, where RMVPE emits exact 0
/// there (SoVITS: uv=0 + gap-interpolated f0; RVC: pitchf=0 → the protect blend finally fires).
/// Rule-based on the leading IPA base symbol: across this vocab every voiceless segment starts
/// with a voiceless stop/fricative letter (affricates start with their stop half), and no voiced
/// token starts with one — the exhaustiveness test below walks ALL 210 tokens so a future vocab
/// regen can't silently slip an unclassified/misclassified token through.
pub fn is_voiceless_phone(p: &str) -> bool {
    if p.contains('\u{0325}') {
        return true; // devoiced vowels i̥ ɨ̥ ɯ̥
    }
    matches!(
        p.chars().next(),
        Some('p' | 't' | 'k' | 'c' | 'q' | 'ʈ' | 'ʔ' | 'f' | 's' | 'ʃ' | 'ɕ' | 'ç' | 'x' | 'h' | 'ʂ' | 'ɸ' | 'θ')
    )
}

/// S83: whether a vocab phone token can carry a syllable NUCLEUS (a vowel — incl. long/nasal/devoiced
/// variants, diphthongs, the zh atomic finals and the r-colored ɝ — or a syllabic consonant m̩/n̩/l̩/ɹ̩/ɻ̩,
/// U+0329). The DAW frame allocator uses it to split a note's phones into onset|nucleus|coda: the note's
/// beat-anchored remainder goes to the NUCLEUS (never to a trailing coda consonant — the pre-S83
/// `split_dur` "last phone takes the rest" rule sang mine→[m aɪ n] as an 84% [n] hum). Rule-based on the
/// leading IPA base symbol: across this vocab every nucleus-capable token starts with a vowel letter, no
/// consonant token does, and the syllabic mark is decisive — the exhaustiveness test below walks ALL 210
/// tokens so a vocab regen can't silently misroute one.
pub fn is_nucleus_phone(p: &str) -> bool {
    if matches!(p, "SP" | "AP" | "PAD" | "BOS" | "EOS") {
        return false;
    }
    if p.contains('\u{0329}') {
        return true; // syllabic consonants: m̩ n̩ l̩ (de/en) + the zh apical "vowels" ɹ̩ ɻ̩
    }
    matches!(
        p.chars().next(),
        Some(
            'a' | 'e' | 'i' | 'o' | 'u' | 'y' | 'ɯ' | 'ɛ' | 'ɔ' | 'ɪ' | 'ʊ' | 'ɐ' | 'ə' | 'ɤ' | 'æ'
                | 'ʌ' | 'ɑ' | 'ø' | 'œ' | 'ʉ' | 'ɵ' | 'ɨ' | 'ʏ' | 'ɝ'
        )
    )
}

// ─── S83 DAW note allocation (syllable-structure-aware; replaces split_dur on the ② render path) ──
//
// WHY (S83 triage, user-verified symptoms): `split_dur` is the training-demo rule (JA CV mora shape:
// consonants ≤4 frames AT THE NOTE START, last phone takes the remainder). Two structural faults on a
// real DAW score: (1) EN closed syllables end in a CODA consonant → the coda takes the note remainder
// and the vowel is crushed to ≤80 ms (mine→"me", fined→760 ms of voiced [d]); (2) onset consonants sit
// INSIDE the note window, so every CV syllable's vowel lands 20-80 ms AFTER the beat (あ/た/し triplet
// unevenness, fast-passage smear, systematic misalignment vs an OpenUtau reference — which pre-rolls
// consonants via preutterance). The training data itself (92.2% of frames: m4singer + gtsinger) anchors
// note onsets at the CONSONANT start — i.e. real singers place the consonant BEFORE the musical beat and
// the vowel ON it. The model only ever sees duration arrays (no absolute beats), so pre-rolling the onset
// is not just safe but strictly CLOSER to the training distribution than the old shape.
//
// The allocator: nucleus = LAST nucleus-capable phone (fallback: the last phone — the legacy CV shape for
// all-consonant notes like っ). Onset = the consonant PREFIX before the FIRST nucleus-capable phone (a
// medial glide vowel, "più"'s i, stays in-note). In-note frames: medial small shares → coda (≤4 frames
// each, ≥2 or dropped first-first, total ≤2/5 of the note — the training vowel-share prior) → nucleus
// takes the remainder. Onset frames are BORROWED from the tail of the previously emitted phone (rest
// keeps ≥1 frame, sung phone keeps ≥2; target 4 frames/consonant ≈ the training median 4-5), so the
// nucleus still starts ON the beat; with no lender (score start) the onset falls back in-note (≤2 frames
// each from the nucleus — the legacy shape, degraded gracefully). INVARIANT: Σ emitted durations ==
// Σ event frames (borrowing is zero-sum, in-note allocation never exceeds fr — the old `max(1)` inflation
// that pushed every later note off the timeline is gone; a coda/onset that can't get its minimum is
// DROPPED, never inflated).
const CODA_MIN_FRAMES: i64 = 2; // a 1-frame phone is categorically OOD (0 occurrences in training npz)
const REST_KEEP_MIN: i64 = 1; // a lent-from rest keeps ≥1 frame (chunk_at_sp still cuts on it)
const SUNG_KEEP_MIN: i64 = 2; // a lent-from sung phone keeps ≥2 frames
/// Fallback targets for a consonant missing from the measured priors (defensive — the generator
/// covers every non-nucleus token; ≈ the global consonant medians).
const ONSET_TARGET_FALLBACK: i64 = 4;
const CODA_TARGET_FALLBACK: i64 = 4;

/// S83 knives 2+3: per-consonant duration TARGETS from the training distribution (see the
/// generated score2cv_dur_priors.rs header), BUCKETED by note length. One flat 4-frame target for
/// every consonant parked each token OFF its own distribution center (stops t/d sit at 3 frames,
/// fricatives s/ɕ at 7 on LONG notes) — and one flat per-token value re-flattens fast passages,
/// where the training data compresses consonants (t 4→2, fricatives 8→3-5: the UTAU preutterance
/// auto-scaling, measured instead of invented). Bucket by the note's own frame count — in a
/// continuous run the beat interval IS the sung group length the training annotation measures.
fn dur_prior(p: &str) -> Option<([i64; 3], [i64; 3], [i64; 3])> {
    static M: OnceLock<HashMap<&'static str, ([i64; 3], [i64; 3], [i64; 3])>> = OnceLock::new();
    M.get_or_init(|| {
        super::score2cv_dur_priors::PHONE_DUR_PRIORS
            .iter()
            .map(|&(t, o, c, z)| (t, (o, c, z)))
            .collect()
    })
    .get(p)
    .copied()
}
fn dur_bucket(fr: i64) -> usize {
    if fr <= 7 {
        0 // short (≤140 ms — e.g. a tempo-222 240t note)
    } else if fr <= 15 {
        1
    } else {
        2
    }
}
fn onset_target_frames(p: &str, fr: i64) -> i64 {
    dur_prior(p).map(|(o, _, _)| o[dur_bucket(fr)]).unwrap_or(ONSET_TARGET_FALLBACK)
}
fn coda_target_frames(p: &str, fr: i64) -> i64 {
    dur_prior(p).map(|(_, c, _)| c[dur_bucket(fr)]).unwrap_or(CODA_TARGET_FALLBACK)
}

/// S83 knife 5: measured f0==0 fraction (permille) inside a voiceless phone's window, bucketed by
/// its note GROUP length. Real singing zeroes only 17-48% of a SHORT-note voiceless window (the
/// RMVPE track drags in from the previous vowel and pre-voices into the next), while the render
/// zeroed 100% (the S69 R0b① over-correction) — on fast runs that collapsed the voiced duty cycle
/// into the audible "briefly mute" さ/こ/け the user pinpointed. Fallback 1000 = full-window zero:
/// exactly right for the devoiced vowels i̥/ɨ̥/ɯ̥ (true whispers, not in the consonant table) and
/// the conservative legacy behavior for anything else unmapped.
pub fn voiceless_zero_permille(p: &str, group_frames: i64) -> i64 {
    dur_prior(p).map(|(_, _, z)| z[dur_bucket(group_frames)]).unwrap_or(1000)
}

/// In-note allocation for one note's phones (`fr` ≥ 1 frames inside the note window): medial + coda get
/// bounded shares, the nucleus takes the remainder, onset positions are left 0 (the caller fills them by
/// borrowing / supplement). Returns per-phone durations aligned to `ph`; entries may be 0 (dropped
/// medial/coda / unfilled onset) — the caller skips 0-duration phones at emission. Σ(durs) ≤ fr always.
fn allocate_in_note(ph: &[&'static str], fr: i64, onset_end: usize, nuc: usize) -> Vec<i64> {
    let n = ph.len();
    let mut durs = vec![0i64; n];
    let n_coda = n - nuc - 1;
    let n_medial = nuc - onset_end;
    let nuc_floor = fr.min(2).max(1); // the nucleus never drops below min(fr,2)
    let mut used = 0i64;
    // medial (between the first and last nucleus): a medial CONSONANT is really the NEXT
    // syllable's ONSET — a multi-syllable word on ONE note flattens its syllable boundaries
    // (refined = [ɹ ə f aɪ n d]: the f leads the second syllable) — so it gets its own measured
    // onset target (the old flat 2..4 share made the f inaudible; S83 user-verified). A medial
    // VOWEL (più's i) keeps the small share. Same ≥2-or-DROP policy as codas (1-frame = OOD);
    // the break drops later medials first.
    for i in onset_end..nuc {
        let c = if is_nucleus_phone(ph[i]) {
            (fr / ((n_medial + n_coda) as i64 + 2)).clamp(CODA_MIN_FRAMES, 4)
        } else {
            onset_target_frames(ph[i], fr)
        }
        .min(fr - nuc_floor - used);
        if c < CODA_MIN_FRAMES {
            break;
        }
        durs[i] = c;
        used += c;
    }
    // coda: per-token measured target each (S83 second knife: t/d≈3, n≈4, s/ɕ≈6-7 — one flat cap
    // flattened the 程度), ≥2 each (else DROPPED — never a 1-frame phone), total ≤ 2/5 of the note
    // (training: vowel share median 44-47%). LAST-first: the word-final release is the perceptually
    // load-bearing cue — when the budget starves, inner codas drop before it.
    if n_coda > 0 {
        let want: i64 = ph[nuc + 1..].iter().map(|p| coda_target_frames(p, fr)).sum();
        let mut budget = want.min(fr - nuc_floor - used).min(fr * 2 / 5);
        for i in (nuc + 1..n).rev() {
            let give = coda_target_frames(ph[i], fr).min(budget);
            if give < CODA_MIN_FRAMES {
                continue; // dropped; the remaining budget stays for the codas before it
            }
            durs[i] = give;
            budget -= give;
            used += give;
        }
    }
    durs[nuc] = fr - used; // nucleus takes the whole remainder (≥ nuc_floor by construction)
    durs
}

#[cfg(test)]
mod foreign_kana_tests {
    use super::*;

    fn phones(lyr: &str) -> Vec<&'static str> {
        match classify_lyric(lyr) {
            LyricClass::Phones { phones } => phones,
            other => panic!("{lyr} should sing, got {other:?}"),
        }
    }

    #[test]
    fn foreign_kana_generic_swap_and_explicit_rows() {
        // the community-reported pair first (used to sing as す via the lossy first-char fallback):
        assert_eq!(phones("すぃ"), vec!["s", "i"]);
        assert_eq!(phones("すぇ"), vec!["s", "e"]);
        for (k, want) in [
            ("てぃ", vec!["t", "i"]), ("とぅ", vec!["t", "ɯ"]),
            ("でぃ", vec!["d", "i"]), ("どぅ", vec!["d", "ɯ"]),
            ("ふぁ", vec!["ɸ", "a"]), ("ふぃ", vec!["ɸ", "i"]), ("ふぇ", vec!["ɸ", "e"]), ("ふぉ", vec!["ɸ", "o"]),
            ("つぁ", vec!["ts", "a"]), ("つぉ", vec!["ts", "o"]),
            ("しぇ", vec!["ɕ", "e"]), ("ちぇ", vec!["tɕ", "e"]), ("じぇ", vec!["dʑ", "e"]),
            ("ずぃ", vec!["z", "i"]),
            ("ゔぁ", vec!["v", "a"]), ("ゔぃ", vec!["v", "i"]), ("ゔぇ", vec!["v", "e"]), ("ゔぉ", vec!["v", "o"]),
            ("うぃ", vec!["w", "i"]), ("うぇ", vec!["w", "e"]), ("うぉ", vec!["w", "o"]), ("いぇ", vec!["j", "e"]),
        ] {
            assert_eq!(phones(k), want, "{k}");
        }
        // katakana arrives folded (g2p::fold_katakana upstream): ティ → てぃ.
        assert_eq!(phones(&super::super::g2p::fold_katakana("ティ")), vec!["t", "i"]);
    }

    #[test]
    fn foreign_kana_never_emits_out_of_vocab_and_legacy_unchanged() {
        // vocabulary-safety sweep: EVERY base×small combo the generic rule accepts must emit only
        // 210-vocab tokens — an out-of-vocab phone would LOUD-error at build_arrays, so this pins
        // the failure to compile-time-adjacent instead of a user's render.
        let ids = phone_to_id_map();
        let smalls = ['ぁ', 'ぃ', 'ぅ', 'ぇ', 'ぉ'];
        let mut combos = 0usize;
        for (base, _) in tbl::KANA.iter().chain(super::super::g2p_tables::KANA_EXTRA) {
            for sv in smalls {
                let s = format!("{base}{sv}");
                if let Some(v) = foreign_kana_phones(&s) {
                    combos += 1;
                    for p in v {
                        assert!(ids.contains_key(p), "{s} emitted out-of-vocab phone {p}");
                    }
                }
            }
        }
        assert!(combos > 300, "sweep actually exercised the rule (got {combos})");
        // legacy behavior untouched (parity anchors):
        assert_eq!(phones("す"), vec!["s", "ɯ"]);
        assert_eq!(phones("し"), vec!["ɕ", "i"]);
        assert_eq!(phones("きゃ"), vec!["c", "a"]);
        assert_eq!(phones("ぃ"), vec!["i"], "a lone small vowel still sings as its plain vowel");
        assert!(matches!(classify_lyric("ー"), LyricClass::Sustain));
    }

    #[test]
    fn foreign_kana_sustain_carrier_integration() {
        // すぃ + ー: the sustain must carry the SWAPPED vowel (i), not the base's ɯ.
        let arr = build_arrays(&[("すぃ", 60, 80), ("ー", 60, 80)]).unwrap();
        assert_eq!(arr.phon, vec!["s", "i", "i"], "sustain re-emits the foreign vowel");
    }
}

#[cfg(test)]
mod voiceless_tests {
    use super::is_voiceless_phone;
    use super::super::score2cv_tables::PHONE_TO_ID;

    #[test]
    fn voiceless_classification_is_exhaustive_and_stable() {
        // Spot anchors (both polarities, every rule family).
        for p in ["k", "s", "t", "ʔ", "tɕ", "tʃː", "pʰ", "t͈", "k̚", "ɸʷ", "sʲ", "t̪s̪", "i̥", "ɯ̥", "h"] {
            assert!(is_voiceless_phone(p), "{p} must be voiceless");
        }
        for p in ["b", "d", "ɡ", "z", "ʒ", "dʑ", "dʒː", "m", "ɴ", "ɾ", "w", "j", "a", "ɯ", "əɻ", "ɦ", "ʁ", "β", "n̪", "m̩"] {
            assert!(!is_voiceless_phone(p), "{p} must NOT be voiceless");
        }
        // Specials never classify voiceless (their frames are already rest/breath-zeroed upstream).
        for p in ["SP", "AP", "PAD", "BOS", "EOS"] {
            assert!(!is_voiceless_phone(p), "{p} special");
        }
        // Exhaustive walk: the count is pinned so a vocab regen that adds/renames tokens forces a
        // REVIEW of this classifier instead of silently misrouting frames (72 = hand-audited count
        // over the S69 vocab: 27 base obstruents + 3 devoiced vowels + 5 palatalized + 9 long +
        // 6 dental + 2 aspirated-fricative + 7 tense + 3 unreleased + 7 labialized + t͈ʲ + tɕː/tʲː).
        let n = PHONE_TO_ID.iter().filter(|(p, _)| is_voiceless_phone(p)).count();
        assert_eq!(n, 72, "voiceless token count drifted — vocab regen? re-audit the classifier");
    }
}

#[cfg(test)]
mod nucleus_tests {
    use super::is_nucleus_phone;
    use super::super::score2cv_tables::PHONE_TO_ID;

    #[test]
    fn duration_priors_cover_every_consonant_and_stay_in_window() {
        use super::is_voiceless_phone;
        use super::super::score2cv_dur_priors::PHONE_DUR_PRIORS;
        let prior: std::collections::HashMap<&str, ([i64; 3], [i64; 3], [i64; 3])> =
            PHONE_DUR_PRIORS.iter().map(|&(t, o, c, z)| (t, (o, c, z))).collect();
        assert_eq!(prior.len(), PHONE_DUR_PRIORS.len(), "no duplicate tokens in the priors table");
        let mut consonants = 0usize;
        for &(p, _) in PHONE_TO_ID {
            let special = matches!(p, "SP" | "AP" | "PAD" | "BOS" | "EOS");
            if !special && !is_nucleus_phone(p) {
                consonants += 1;
                let (o, c, z) =
                    *prior.get(p).unwrap_or_else(|| panic!("consonant {p} missing a duration prior"));
                for v in o.iter().chain(c.iter()) {
                    assert!((2..=7).contains(v), "{p} prior out of window: {o:?}/{c:?}");
                }
                if is_voiceless_phone(p) {
                    assert!(z.iter().all(|v| (1..=1000).contains(v)), "{p} zero-permille out of range: {z:?}");
                } else {
                    assert_eq!(z, [0, 0, 0], "{p} is voiced — its zero column must be 0 (never consulted)");
                }
            } else {
                assert!(!prior.contains_key(p), "{p} (nucleus/special) must not carry a prior");
            }
        }
        assert_eq!(consonants, PHONE_DUR_PRIORS.len(), "table covers exactly the consonant set");
    }

    #[test]
    fn nucleus_classification_is_exhaustive_and_stable() {
        // Spot anchors (both polarities, every rule family).
        for p in ["a", "ɯ", "aɪ", "oʊ", "ɑŋ", "uən", "yɛn", "əɻ", "ɝ", "i̥", "ɛ̃", "aː", "m̩", "n̩", "l̩", "ɹ̩", "ɻ̩"] {
            assert!(is_nucleus_phone(p), "{p} must be nucleus-capable");
        }
        for p in ["m", "n", "ŋ", "t", "d", "ɹ", "ɻ", "w", "j", "ɥ", "ʔ", "ts", "ʈʂ", "nʲ", "rː", "n̪"] {
            assert!(!is_nucleus_phone(p), "{p} must NOT be nucleus-capable");
        }
        for p in ["SP", "AP", "PAD", "BOS", "EOS"] {
            assert!(!is_nucleus_phone(p), "{p} special");
        }
        // Exhaustive walk, count pinned (76 = hand-audited over the S69 vocab: 23 base vowels +
        // 13 long/nasal + 3 devoiced + 6 diphthongs + 25 zh atomic finals + ɝ + 5 syllabic
        // consonants m̩ n̩ l̩ ɹ̩ ɻ̩). A vocab regen that shifts this count must re-audit the classifier
        // (a misrouted token would silently starve/inflate its syllable's nucleus).
        let n = PHONE_TO_ID.iter().filter(|(p, _)| is_nucleus_phone(p)).count();
        assert_eq!(n, 76, "nucleus token count drifted — vocab regen? re-audit the classifier");
    }
}

/// Port of `render_ust.build_arrays` (+ its `lyric_to_phones` front-end). `score` = (lyric, note_num,
/// frames) per note. Rest frames are capped (first/last note → `CAP_LEAD`, mid → `CAP_MID`); a sustain
/// (`-`/`ー`/`+`) continues the previous vowel (default `a`); notes are grouped into `note_to_phone` by
/// consecutive equal pitch. An OOV lyric is a LOUD error (never the reference's silent SP fallback — the
/// v1 "啊啊啊" regression). This is the Phase-1c PARITY entry (rest-capped, JA, == render_ust); since S58
/// it routes through the SAME g2p resolve + assembly core as the multi-language DAW path — the 1c
/// bit-parity tests below prove the shared core reproduces the legacy arrays exactly. The ② vocal DAW
/// render uses `build_arrays_daw` (rests uncapped, per-note language) so the stem aligns to the timeline.
pub fn build_arrays(score: &[(&str, i64, i64)]) -> Result<ScoreArrays> {
    let evts: Vec<g2p::ScoreEvt> = score.iter().map(g2p::ScoreEvt::ja).collect();
    let resolved = g2p::resolve_score(&evts, &NoDicts)?;
    assemble_arrays(&evts, &resolved, true, false) // Phase-1c PARITY: rest-capped, split_dur, no pre-roll
}

/// A `DictSource` for pure-JA paths (parity + tests): JA needs no dictionary files; any zh/word-dict
/// request is a loud missing-dictionary error.
pub struct NoDicts;
impl g2p::DictSource for NoDicts {
    fn zh(&self) -> Result<&g2p::ZhDict> {
        Err(UtaiError::Inference("VOCAL_DICT_MISSING: zh".into()))
    }
    fn words(&self, lang: g2p::Lang) -> Result<&g2p::WordDict> {
        Err(UtaiError::Inference(format!("VOCAL_DICT_MISSING: {}", lang.code())))
    }
}

/// M3 minimum cv frames a SUNG vowel gets (via borrow-time in the DAW render) so net_g renders an audible
/// vowel rather than a 1-frame smear. ~100 ms @50fps; kept small so a normal note is never inflated.
const VOWEL_MIN_FRAMES: i64 = 5;

/// The phone a zh MELISMA hold re-emits (a pitch-changed sustain needs its own per-phone entry —
/// pitch is per-phone). Re-emitting a glide-carrying final re-articulates the glide ("xiang -" →
/// iɑŋ iɑŋ ≈ singing "yang" again), so strip a leading glide vowel (i/u/y) when the remainder is
/// itself a 210-vocab token with a vocalic head (uɑŋ→ɑŋ, iaʊ→aʊ, ua→a); anything else (glide-free
/// finals, or remainders like the bare nasals of in/iŋ/yn) holds the full carrier unchanged.
fn zh_hold_phone(carrier: &'static str) -> &'static str {
    let mut chars = carrier.chars();
    if let Some(first) = chars.next() {
        if matches!(first, 'i' | 'u' | 'y') {
            let rest: &'static str = &carrier[first.len_utf8()..];
            let vocalic_head = rest.chars().next().is_some_and(|c| !matches!(c, 'n' | 'ŋ' | 'ɻ'));
            if !rest.is_empty() && vocalic_head && phone_to_id_map().contains_key(rest) {
                return rest;
            }
        }
    }
    carrier
}

/// ② 自己唱 DAW render entry: identical to `build_arrays` EXCEPT rest frames are NOT capped, so the cv
/// frame count == the DAW frame count (a rest holds its full duration) and the rendered stem stays
/// aligned to the segment's tick timeline; and the score is MULTI-LANGUAGE (S58): per-note effective
/// language + optional traditional-phoneme override, resolved by `g2p::resolve_score` (dictionaries,
/// zh phrase context, western syllable spans/归韵). A deliberate fork from the Python parity (which caps
/// rests for a standalone song render); the Phase-1c gate still tests the capped `build_arrays`.
/// ⚠ `chunk_at_sp` CANNOT subdivide a single rest (it only splits AT an SP after the running count
/// exceeds max_frames), so one very long rest becomes one big chunk — the TOTAL frame count is bounded
/// upstream by `render_vocal_segment`'s `MAX_TOTAL_FRAMES`, not here.
pub fn build_arrays_daw(score: &[g2p::ScoreEvt], dicts: &dyn g2p::DictSource) -> Result<ScoreArrays> {
    let resolved = g2p::resolve_score(score, dicts)?;
    // ② DAW: rests uncapped + S83 syllable-aware allocation (onset pre-roll / nucleus-remainder /
    // bounded coda) + short-note borrow-time (M3). Frame-CONSERVING: Σ phone_dur == Σ evt frames.
    assemble_arrays(score, &resolved, false, true)
}

/// THE single array-assembly core (S58): resolved per-note phones → the model's per-phone arrays.
/// Frame policy, id mapping, and note grouping — grouping keys on (pitch, RUN LANGUAGE) so a group never
/// spans a language cut (single-language scores group exactly as before). Both `build_arrays` (parity,
/// capped, legacy `split_dur`) and `build_arrays_daw` (DAW, uncapped, S83 syllable-aware allocation +
/// onset pre-roll) feed through here — one implementation, proven by the 1c bit-parity gate.
fn assemble_arrays(
    score: &[g2p::ScoreEvt],
    resolved: &[g2p::ResolvedNote],
    cap_rests: bool,
    daw: bool,
) -> Result<ScoreArrays> {
    let m = score.len();
    let mut phon: Vec<&'static str> = Vec::new();
    let mut pdur: Vec<i64> = Vec::new();
    let mut npitch: Vec<i64> = Vec::new();
    let mut plang: Vec<i64> = Vec::new();
    let mut pevt: Vec<usize> = Vec::new();

    for (k, (evt, res)) in score.iter().zip(resolved.iter()).enumerate() {
        let (nn, fr) = (evt.note_num, evt.frames);
        let lang_id = res.run_lang.id();
        match &res.kind {
            g2p::ResolvedKind::Rest | g2p::ResolvedKind::Breath => {
                let cap = if !cap_rests {
                    i64::MAX // DAW render: keep the full rest so the stem aligns to the timeline
                } else if k == 0 || k == m - 1 {
                    tbl::CAP_LEAD
                } else {
                    tbl::CAP_MID
                };
                phon.push(if matches!(res.kind, g2p::ResolvedKind::Rest) { "SP" } else { "AP" });
                pdur.push(fr.min(cap).max(1));
                npitch.push(0);
                plang.push(lang_id);
                pevt.push(k);
            }
            g2p::ResolvedKind::Phones(ph) => {
                // S66 zh sustain fix (user bug: [wang][-] sang "wang wang"): the zh carrier is the
                // ATOMIC final token ("uɑŋ") — re-emitting it as a fresh phone entry makes ScoreToCV
                // re-articulate the glide+coda, i.e. sing the syllable again (the model's trained
                // hold convention is ja-style repeated bare VOWELS, and opencpop-style zh data holds
                // a note as ONE long final, never a repeated one). So:
                //   same pitch  → EXTEND the previous entry's duration (a true hold, no new phone;
                //                 also covers chained holds — the previous entry may already be a
                //                 hold nucleus);
                //   pitch change (melisma) → a new entry MUST exist (pitch is per-phone), so emit
                //                 the carrier final's vocalic tail (glide stripped when the残り is
                //                 itself a vocab token: uɑŋ→ɑŋ, iaʊ→aʊ), else the full final.
                // ja sustains keep the legacy repeated-vowel path bit-for-bit (Phase-1c parity gate).
                if res.is_sustain && res.run_lang == g2p::Lang::Zh {
                    let prev_sung = phon.last().is_some_and(|&p| !matches!(p, "SP" | "AP"));
                    if prev_sung && npitch.last() == Some(&nn) {
                        *pdur.last_mut().unwrap() += fr;
                        continue;
                    }
                    if prev_sung {
                        if let Some(&carrier) = ph.last() {
                            let hold = zh_hold_phone(carrier);
                            phon.push(hold);
                            pdur.push(fr.max(1));
                            npitch.push(nn);
                            plang.push(lang_id);
                            pevt.push(k);
                            continue;
                        }
                    }
                    // sustain after silence (orphan): fall through to the normal emit ("a" default)
                }
                if daw {
                    // ── S83 syllable-aware allocation + onset pre-roll (see the block comment above
                    //    allocate_in_note for the full rationale) ──
                    let n = ph.len();
                    let fr = fr.max(1);
                    let nuc = ph.iter().rposition(|p| is_nucleus_phone(p)).unwrap_or(n - 1);
                    let onset_end = ph.iter().position(|p| is_nucleus_phone(p)).unwrap_or(n - 1).min(nuc);
                    let mut durs = allocate_in_note(ph, fr, onset_end, nuc);
                    if onset_end > 0 {
                        // borrow the onset consonants' frames from the tail of the previous phone so
                        // the NUCLEUS starts on the beat (zero-sum: the timeline never moves).
                        let avail = match phon.last() {
                            Some(&"SP") | Some(&"AP") => (pdur.last().copied().unwrap_or(0) - REST_KEEP_MIN).max(0),
                            Some(_) => {
                                // UTAU-style auto-scale, structural half: a SUNG lender never loses
                                // more than half its frames (ceil) — in a fast run the previous
                                // vowel must stay audible. ⚠this clamp alone CANNOT keep vowels at
                                // the training ~3 frames when every target exceeds it (S84 review:
                                // fr=4/5 pinned whole passages at exactly 2) — the fr≤5 target cap
                                // below is what restores the measured C/V split there.
                                let last = pdur.last().copied().unwrap_or(0);
                                (last - SUNG_KEEP_MIN).max(0).min((last + 1) / 2)
                            }
                            None => 0, // score start: no lender — the onset falls back in-note below
                        };
                        // S84 A 刀: at fr ≤ 5 (the tempo-222 160t fast-run regime) the short-bucket
                        // MEDIAN targets (t3/k4/s4 — a ≤7 bucket dominated by 6-7-frame groups)
                        // always exceed the ceil-half clamp, so the LENDER vowel was pinned at
                        // exactly 2 frames for whole passages — contradicting this file's own
                        // "targets handle the scaling" intent (S84 review) AND the measured 4-5
                        // frame population: the DOMINANT total=5 CV shape is C2/V3 (119/261) and
                        // total=4 sits at median C2 (C1V3 37 / C2V2 23 / C3V1 30). Capping each
                        // onset target at 2 when the borrowing note is this short lands both:
                        // a 5-frame lender keeps a 3-frame vowel, fr=4 behavior is unchanged
                        // (avail already clamped to 2), and fr ≥ 6 (the ear-anchored 240t triplet)
                        // never enters this branch — bit-identical by construction.
                        let target = |p: &'static str| {
                            let t = onset_target_frames(p, fr);
                            if fr <= 5 { t.min(2) } else { t }
                        };
                        let want: i64 = ph[..onset_end].iter().map(|&p| target(p)).sum();
                        let mut left = want.min(avail);
                        if left > 0 {
                            *pdur.last_mut().unwrap() -= left;
                        }
                        // distribute LAST-first (the consonant adjacent to the vowel carries the
                        // syllable identity — a starved cluster sheds its outermost member first),
                        // each capped at its own measured, note-length-bucketed target.
                        for i in (0..onset_end).rev() {
                            let give = left.min(target(ph[i]));
                            durs[i] = give;
                            left -= give;
                        }
                        // in-note supplement for underfed onsets (score start / drained lender):
                        // ALL-OR-NOTHING top-up to the 2-frame minimum out of the nucleus's spare above
                        // its own min(fr,2) floor — a partial top-up would blend lender/nucleus frames
                        // and the drop pass below could no longer return them to their sources, and a
                        // 1-frame phone is categorically OOD anyway (S83 review #0).
                        let nuc_floor = fr.min(2);
                        for i in (0..onset_end).rev() {
                            if durs[i] >= 2 {
                                continue;
                            }
                            let need = 2 - durs[i];
                            if (durs[nuc] - nuc_floor).max(0) >= need {
                                durs[nuc] -= need;
                                durs[i] += need;
                            }
                        }
                        // sub-minimum onsets DROP (same policy as codas/medials); at this point their
                        // frames are pure LENDER frames (the supplement is all-or-nothing), so hand them
                        // back — the borrow must stay zero-sum even when it fails (conservation).
                        let mut returned = 0i64;
                        for d in durs.iter_mut().take(onset_end) {
                            if *d > 0 && *d < 2 {
                                returned += *d;
                                *d = 0;
                            }
                        }
                        if returned > 0 {
                            *pdur.last_mut().unwrap() += returned;
                        }
                    }
                    for (&p, &d) in ph.iter().zip(durs.iter()) {
                        if d <= 0 {
                            continue; // dropped medial/coda / sub-minimum onset — never emit a 0-frame phone
                        }
                        phon.push(p);
                        pdur.push(d);
                        npitch.push(nn);
                        plang.push(lang_id);
                        pevt.push(k);
                    }
                } else {
                    let durs = split_dur(fr, ph.len());
                    for (&p, &d) in ph.iter().zip(durs.iter()) {
                        phon.push(p);
                        pdur.push(d);
                        npitch.push(nn);
                        plang.push(lang_id);
                        pevt.push(k);
                    }
                }
            }
            g2p::ResolvedKind::Unknown => {
                // unreachable via resolve_score (strict errors first) — defensive LOUD error.
                return Err(UtaiError::Inference(format!("VOCAL_OOV: {}", evt.lyric)));
            }
        }
    }

    // M3 short-note borrow-time (DAW render only): give each SUNG vowel at least VOWEL_MIN_FRAMES cv frames
    // by borrowing from an IMMEDIATELY-FOLLOWING rest/breath (SP/AP), keeping that ≥1 frame. The borrow only
    // shifts the vowel↔rest boundary LATER — the total frame count and every note ONSET are preserved — so
    // the rendered stem stays aligned to the DAW tick timeline (节奏不变). A short note with no trailing rest
    // is left as-is (extending it would eat the next note's onset; the decode pad-and-trim covers the hard
    // sub-min-frames floor). Deliberate fork from Phase-1c parity (build_arrays keeps borrow_time=false).
    if daw {
        // S83: the vowel test widened from tbl::VOWEL_SET (the 5 JA vowels) to the full nucleus
        // classifier — an EN aɪ / zh final / syllabic n̩ deserves the same floor as a JA vowel.
        for i in 0..phon.len() {
            if npitch[i] > 0 && is_nucleus_phone(phon[i]) && pdur[i] < VOWEL_MIN_FRAMES {
                let deficit = VOWEL_MIN_FRAMES - pdur[i];
                if i + 1 < phon.len() && matches!(phon[i + 1], "SP" | "AP") {
                    let take = deficit.min((pdur[i + 1] - 1).max(0));
                    pdur[i] += take;
                    pdur[i + 1] -= take;
                }
            }
        }
    }

    // phone → id (LOUD error on any phone outside the 210-token vocab; the reference SP-falls-back).
    let map = phone_to_id_map();
    let mut phonemes = Vec::with_capacity(phon.len());
    for &p in &phon {
        let id = *map.get(p).ok_or_else(|| {
            // CODE + phone detail (i18n'd frontend-side) — a mapped phone outside the 210-token vocab.
            UtaiError::Inference(format!("VOCAL_PHONE_MISSING: {}", p))
        })?;
        phonemes.push(id);
    }

    // note grouping: consecutive equal (pitch, run-language) → one note group; note_dur = Σ phone_dur
    // within a group. The language term (S58) keeps a group from spanning a language cut (per-chunk
    // lang_id must be uniform); single-language scores group exactly as the legacy pitch-only rule.
    let mut note_to_phone = Vec::with_capacity(npitch.len());
    let mut nidx: i64 = -1;
    let mut prev: Option<(i64, i64)> = None;
    for (i, &p) in npitch.iter().enumerate() {
        if prev != Some((p, plang[i])) {
            nidx += 1;
            prev = Some((p, plang[i]));
        }
        note_to_phone.push(nidx);
    }
    let group_count = (nidx + 1).max(0) as usize;
    let mut group_frames = vec![0i64; group_count];
    for (i, &g) in note_to_phone.iter().enumerate() {
        group_frames[g as usize] += pdur[i];
    }
    let note_dur: Vec<i64> = note_to_phone.iter().map(|&g| group_frames[g as usize]).collect();

    Ok(ScoreArrays { phonemes, phone_dur: pdur, note_pitch: npitch, note_dur, note_to_phone, phon, lang: plang, evt: pevt })
}

// ─── SP-boundary chunking (≤ max_frames) + per-chunk rebase ──────────────────────────────────────

/// One inference chunk: the per-phone arrays sliced to `[start, end)`, with `note_to_phone` rebased to 0.
pub struct Chunk {
    pub start: usize,
    pub end: usize,
    pub phonemes: Vec<i64>,
    pub note_pitch: Vec<i64>,
    pub phone_dur: Vec<i64>,
    pub note_dur: Vec<i64>,
    pub note_to_phone: Vec<i64>,
    /// Output frame count = Σ phone_dur in this chunk (= cv rows).
    pub t: usize,
    /// The chunk's (uniform) ScoreToCV language id — chunks are cut at every language change (S58).
    pub lang_id: i64,
    /// True when the seam BEFORE this chunk is a mid-voiced language cut (no SP at the boundary) —
    /// the decode concat applies a micro-fade there to mask the splice (an SP seam is silence).
    pub hard_seam: bool,
}

fn make_chunk(a: &ScoreArrays, s: usize, e: usize, hard_seam: bool) -> Chunk {
    let base = a.note_to_phone[s];
    Chunk {
        start: s,
        end: e,
        phonemes: a.phonemes[s..e].to_vec(),
        note_pitch: a.note_pitch[s..e].to_vec(),
        phone_dur: a.phone_dur[s..e].to_vec(),
        note_dur: a.note_dur[s..e].to_vec(),
        note_to_phone: a.note_to_phone[s..e].iter().map(|x| x - base).collect(),
        t: a.phone_dur[s..e].iter().sum::<i64>() as usize,
        lang_id: a.lang.get(s).copied().unwrap_or(2),
        hard_seam,
    }
}

/// Cut the score into chunks at SP (rest) boundaries once the running frame count exceeds `max_frames`
/// (deploy default 400), bounding SVC memory + O(N²) — verbatim from `render_ust.render_song`: split on
/// the phone STRING "SP" (never the id), the SP is included in the closing chunk, and each chunk's
/// `note_to_phone` is rebased to start at 0. S58: ALSO cut at every LANGUAGE change (the model's lang_id
/// is a per-call scalar, so a chunk must be single-language); a language cut not adjacent to an SP marks
/// the following chunk `hard_seam` for the decode-concat micro-fade. Single-language scores cut exactly
/// as before (the 1c chunking parity test locks it).
pub fn chunk_at_sp(a: &ScoreArrays, max_frames: i64) -> Vec<Chunk> {
    let n = a.phonemes.len();
    let mut chunks = Vec::new();
    let (mut start, mut cf) = (0usize, 0i64);
    let mut next_hard = false; // seam flag for the chunk that `start` begins
    for i in 0..n {
        cf += a.phone_dur[i];
        let lang_cut = i + 1 < n && a.lang[i + 1] != a.lang[i];
        if lang_cut || (cf > max_frames && a.phon[i] == "SP") {
            chunks.push(make_chunk(a, start, i + 1, next_hard));
            start = i + 1;
            cf = 0;
            // the NEXT chunk's leading seam: hard iff this was a language cut not landing in silence.
            next_hard = lang_cut && a.phon[i] != "SP";
        }
    }
    if start < n {
        chunks.push(make_chunk(a, start, n, next_hard));
    }
    chunks
}

// ─── inference: one chunk → cv[T, dim] ───────────────────────────────────────────────────────────

/// Run ScoreToCV on one chunk → de-normalized content features `[T, dim]` (T = Σ phone_dur; dim = 768 or
/// 256). Feeds the 9 graph inputs (phone_mask all-true at B=1; technique all-zero — the dead channel).
pub fn run_score2cv(
    engine: &OnnxEngine,
    session_id: &str,
    chunk: &Chunk,
    dim: usize,
    speaker_id: i64,
    lang_id: i64,
) -> Result<Array2<f32>> {
    let n = chunk.phonemes.len();
    let ni = n as i64;
    let inputs = vec![
        ("phonemes", InputTensor::I64 { data: chunk.phonemes.clone(), shape: vec![1, ni] }),
        ("note_pitch", InputTensor::I64 { data: chunk.note_pitch.clone(), shape: vec![1, ni] }),
        ("phone_dur", InputTensor::I64 { data: chunk.phone_dur.clone(), shape: vec![1, ni] }),
        ("note_dur", InputTensor::I64 { data: chunk.note_dur.clone(), shape: vec![1, ni] }),
        ("note_to_phone", InputTensor::I64 { data: chunk.note_to_phone.clone(), shape: vec![1, ni] }),
        ("speaker_id", InputTensor::I64 { data: vec![speaker_id], shape: vec![1] }),
        ("lang_id", InputTensor::I64 { data: vec![lang_id], shape: vec![1] }),
        ("phone_mask", InputTensor::Bool { data: vec![true; n], shape: vec![1, ni] }),
        ("technique", InputTensor::F32 { data: vec![0.0; n * 7], shape: vec![1, ni, 7] }),
    ];
    let outputs = engine.run(session_id, inputs)?;
    let cv = outputs
        .into_iter()
        .next()
        .ok_or_else(|| UtaiError::Inference("SCORE2CV_NO_OUTPUT".into()))?;
    let t = chunk.t;
    if cv.len() != t * dim {
        return Err(UtaiError::Inference(format!(
            "SCORE2CV_SHAPE: expected {}x{}={}, got {}",
            t,
            dim,
            t * dim,
            cv.len()
        )));
    }
    Array2::from_shape_vec((t, dim), cv).map_err(|e| UtaiError::Inference(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::tbl::parity_ref as pr;

    // The Phase 1c GATE: the Rust port must reproduce render_ust.build_arrays bit-for-bit on the fixed
    // score (reference dumped by scratchpad/dump_g2p.py + gen_rust_tables.py).
    #[test]
    fn build_arrays_matches_python() {
        let a = build_arrays(pr::SCORE).unwrap();
        assert_eq!(a.phon.as_slice(), pr::PHON_STR, "phon strings (G2P)");
        assert_eq!(a.phonemes.as_slice(), pr::PHONEMES, "phonemes (ids)");
        assert_eq!(a.phone_dur.as_slice(), pr::PHONE_DUR, "phone_dur (split_dur + rest caps)");
        assert_eq!(a.note_pitch.as_slice(), pr::NOTE_PITCH, "note_pitch");
        assert_eq!(a.note_dur.as_slice(), pr::NOTE_DUR, "note_dur (group sums)");
        assert_eq!(a.note_to_phone.as_slice(), pr::NOTE_TO_PHONE, "note_to_phone (grouping)");
        assert_eq!(a.phonemes.len(), pr::N);
    }

    #[test]
    fn chunking_and_rebase_matches_python() {
        let a = build_arrays(pr::SCORE).unwrap();
        let chunks = chunk_at_sp(&a, 400);
        assert_eq!(chunks.len(), pr::CHUNKS.len(), "chunk count");
        for (i, (c, r)) in chunks.iter().zip(pr::PER_CHUNK).enumerate() {
            assert_eq!((c.start, c.end), pr::CHUNKS[i], "chunk {} range", i);
            assert_eq!(c.phonemes.as_slice(), r.phonemes, "chunk {} phonemes", i);
            assert_eq!(c.note_pitch.as_slice(), r.note_pitch, "chunk {} note_pitch", i);
            assert_eq!(c.phone_dur.as_slice(), r.phone_dur, "chunk {} phone_dur", i);
            assert_eq!(c.note_dur.as_slice(), r.note_dur, "chunk {} note_dur", i);
            assert_eq!(c.note_to_phone.as_slice(), r.note_to_phone, "chunk {} note_to_phone (rebased)", i);
            assert_eq!(c.t, r.t, "chunk {} T", i);
        }
    }

    #[test]
    fn split_dur_matches_python() {
        assert_eq!(split_dur(100, 2), vec![4, 96]); // か: c=min(4,33)=4
        assert_eq!(split_dur(100, 3), vec![4, 4, 92]); // tta: 3 phones
        assert_eq!(split_dur(20, 1), vec![20]); // single phone
        assert_eq!(split_dur(2, 5), vec![1, 1, 1, 1, 1]); // n>fr: c=max(1,0)=1, last=max(1,-2)=1
        assert_eq!(split_dur(0, 1), vec![1]); // zero-frame guard → max(1,·)
    }

    #[test]
    fn oov_lyric_errors_loudly() {
        // Unknown lyric must ERROR — never the silent SP fallback (v1 "啊啊啊" regression).
        assert!(build_arrays(&[("か", 60, 100), ("zzzz", 62, 80)]).is_err());
        // …but a clean score of the same shape must succeed.
        assert!(build_arrays(&[("か", 60, 100), ("き", 62, 80)]).is_ok());
    }

    /// JA-defaulted DAW build over legacy triples (the pre-S58 test fixtures).
    fn daw_ja(score: &[(&str, i64, i64)]) -> Result<ScoreArrays> {
        let evts: Vec<g2p::ScoreEvt> = score.iter().map(g2p::ScoreEvt::ja).collect();
        build_arrays_daw(&evts, &NoDicts)
    }

    // ② vocal DAW render (S53): `build_arrays_daw` keeps the FULL rest so the stem aligns to the
    // timeline; `build_arrays` (parity) caps it (CAP_MID=70 mid / CAP_LEAD=25 first/last).
    #[test]
    fn build_arrays_daw_uncaps_rests() {
        let score = [("か", 60, 80), ("R", 0, 300), ("お", 67, 80)];
        let capped = build_arrays(&score).unwrap();
        let daw = daw_ja(&score).unwrap();
        // find the SP phone's frame count in each.
        let sp_capped = capped.phon.iter().zip(&capped.phone_dur).find(|(p, _)| **p == "SP").map(|(_, d)| *d);
        let sp_daw = daw.phon.iter().zip(&daw.phone_dur).find(|(p, _)| **p == "SP").map(|(_, d)| *d);
        assert_eq!(sp_capped, Some(tbl::CAP_MID), "parity build caps a mid rest to CAP_MID");
        assert_eq!(sp_daw, Some(300), "DAW build keeps the full 300-frame rest");
    }

    // M3 breath (fork from Phase-1c parity — pr::SCORE has no breath): a breath token emits the AP phone
    // (id AP_ID), npitch 0 (unvoiced), classified distinctly from a rest.
    #[test]
    fn breath_emits_ap() {
        let arr = daw_ja(&[("か", 60, 80), ("AP", 0, 60), ("お", 67, 80)]).unwrap();
        let ap = arr.phon.iter().position(|&p| p == "AP").expect("breath emits an AP phone");
        assert_eq!(arr.note_pitch[ap], 0, "breath is unvoiced (npitch 0)");
        assert_eq!(arr.phonemes[ap], tbl::AP_ID, "AP phone maps to AP_ID");
    }

    // M3 borrow-time (fork from parity): a short sung vowel followed by a rest borrows frames from the rest
    // up to VOWEL_MIN_FRAMES, keeping the rest ≥1 and the TOTAL frame count (timeline) unchanged. The parity
    // build (build_arrays) does NOT borrow.
    #[test]
    fn borrow_time_extends_short_vowel() {
        let score = [("お", 60, 3), ("R", 0, 40)]; // a 3-frame vowel then a rest
        let daw = daw_ja(&score).unwrap();
        assert_eq!(daw.phon[0], "o");
        assert_eq!(daw.phone_dur[0], VOWEL_MIN_FRAMES, "short vowel borrowed up to the floor");
        assert_eq!(daw.phone_dur[1], 40 - (VOWEL_MIN_FRAMES - 3), "the rest shrank by the borrowed amount");
        assert_eq!(daw.phone_dur[0] + daw.phone_dur[1], 3 + 40, "total frames (timeline) preserved");
        // parity build: no borrow (the vowel keeps its 3 frames).
        assert_eq!(build_arrays(&score).unwrap().phone_dur[0], 3, "parity build does not borrow-time");
    }

    // ── S83 syllable-aware DAW allocation + onset pre-roll ──

    // The user-verified S83 triplet case (tempo-222 UST, 240t≈7fr notes): every CV syllable's vowel
    // must start ON its beat — the onset consonant borrows frames from the PREVIOUS phone's tail.
    #[test]
    fn preroll_puts_every_vowel_on_the_beat() {
        let daw = daw_ja(&[("あ", 69, 7), ("た", 71, 7), ("し", 73, 6)]).unwrap();
        assert_eq!(daw.phon, vec!["a", "t", "a", "ɕ", "i"]);
        // Bucketed priors (knife 3) + voiceless-onset p75 (knife 4): on a SHORT note (≤7fr) the
        // stop t sits at its measured p75 = 3 frames (clarity lift — the p50 was 2) and the
        // fricative ɕ is held to the lender-half cap 4 — the previous vowels keep 4/3 frames
        // (the training short-bucket vowel median is 3): fast runs stay intelligible AND crisp.
        assert_eq!(daw.phone_dur, vec![4, 3, 3, 4, 6], "short-bucket p75 targets + lender-half cap");
        assert_eq!(daw.phone_dur.iter().sum::<i64>(), 7 + 7 + 6, "frame-conserving (timeline unmoved)");
        // vowel onsets (cumulative) land exactly on the beats 0 / 7 / 14:
        assert_eq!(4 + 3, 7, "た's vowel starts on beat 7");
        assert_eq!(4 + 3 + 3 + 4, 14, "し's vowel starts on beat 14");
        // evt mapping (the phoneme lane's note attribution):
        assert_eq!(daw.evt, vec![0, 1, 1, 2, 2]);
        // parity build untouched: split_dur shape, consonant INSIDE the note window.
        let parity = build_arrays(&[("あ", 69, 7), ("た", 71, 7), ("し", 73, 6)]).unwrap();
        assert_eq!(parity.phone_dur, vec![7, 2, 5, 2, 4], "parity keeps the legacy split_dur shape");
    }

    // S84 A 刀: fr≤5 (tempo-222 160t fast runs) caps every onset target at 2 — the measured 4-5
    // frame population's C/V split (dominant total=5 shape = C2/V3 at 119/261; total=4 median C2).
    // A full-size 5-frame lender now keeps a 3-frame vowel instead of being drained to the clamp
    // floor 2 (the S84 "whole passage pinned at 2 frames" equilibrium); fr=4 lending is unchanged
    // (avail was already 2); fr≥6 never enters the cap — the 240t triplet above stays bit-identical.
    #[test]
    fn fast_run_fr5_vowels_keep_three_frames() {
        let daw = daw_ja(&[("R", 0, 10), ("こ", 73, 5), ("こ", 73, 4), ("こ", 73, 5), ("こ", 73, 4)]).unwrap();
        assert_eq!(daw.phon, vec!["SP", "k", "o", "k", "o", "k", "o", "k", "o"]);
        // k always takes exactly 2 (capped); a 5-frame note's vowel keeps 3 after lending, a
        // 4-frame note's keeps 2, the terminal vowel keeps its full note.
        assert_eq!(daw.phone_dur, vec![8, 2, 3, 2, 2, 2, 3, 2, 4]);
        assert_eq!(daw.phone_dur.iter().sum::<i64>(), 10 + 5 + 4 + 5 + 4, "frame-conserving");
    }

    // Score-start fallback: no lender → the onset falls back IN-note (≤2 frames from the nucleus).
    #[test]
    fn preroll_score_start_falls_back_in_note() {
        let daw = daw_ja(&[("た", 60, 7)]).unwrap();
        assert_eq!(daw.phon, vec!["t", "a"]);
        assert_eq!(daw.phone_dur, vec![2, 5], "no previous phone: 2 in-note frames, nucleus keeps the rest");
    }

    // Borrowing from a REST keeps ≥1 frame of it (chunk_at_sp still cuts there).
    #[test]
    fn preroll_borrows_from_rest_keeping_one_frame() {
        let daw = daw_ja(&[("あ", 60, 5), ("R", 0, 3), ("た", 62, 7)]).unwrap();
        assert_eq!(daw.phon, vec!["a", "SP", "t", "a"]);
        // SP had 3 and keeps ≥1 → lends only 2 of the 4 wanted; t's 2 frames clear the supplement
        // floor, so the nucleus keeps the full note.
        assert_eq!(daw.phone_dur, vec![5, 1, 2, 7]);
        assert_eq!(daw.phone_dur.iter().sum::<i64>(), 5 + 3 + 7);
    }

    // EN closed syllable (the S83 "mine→me" crown case): the CODA is bounded, the NUCLEUS takes the
    // note's remainder, and the onset pre-rolls — the exact inversion of the old split_dur shape.
    struct EnOnly(g2p::WordDict);
    impl g2p::DictSource for EnOnly {
        fn zh(&self) -> Result<&g2p::ZhDict> {
            Err(UtaiError::Inference("VOCAL_DICT_MISSING: fixture".into()))
        }
        fn words(&self, lang: g2p::Lang) -> Result<&g2p::WordDict> {
            if lang == g2p::Lang::En { Ok(&self.0) } else { Err(UtaiError::Inference("VOCAL_DICT_MISSING: fixture".into())) }
        }
    }
    fn en_dicts() -> EnOnly {
        EnOnly(g2p::WordDict::from_tsv(g2p::Lang::En, "mine\tM AY1 N\nfined\tF AY1 N D\n"))
    }
    fn en_evt(lyric: &'static str, note_num: i64, frames: i64) -> g2p::ScoreEvt<'static> {
        g2p::ScoreEvt { lyric, note_num, frames, lang: g2p::Lang::En, phoneme_input: None }
    }

    #[test]
    fn en_closed_syllable_nucleus_takes_the_remainder() {
        let d = en_dicts();
        // R(10) mine(50) R(10) — a 1-second note at 50fps.
        let score = [en_evt("R", 0, 10), en_evt("mine", 69, 50), en_evt("R", 0, 10)];
        let arr = build_arrays_daw(&score, &d).unwrap();
        assert_eq!(arr.phon, vec!["SP", "m", "aɪ", "n", "SP"]);
        // m pre-rolls at its measured 5-frame target from the leading rest; coda n is bounded at
        // its 4-frame target; the vowel gets the remainder (46 frames = 92% — the old split_dur
        // gave it 4 and the [n] hum 42).
        assert_eq!(arr.phone_dur, vec![5, 5, 46, 4, 10]);
        assert_eq!(arr.phone_dur.iter().sum::<i64>(), 70, "frame-conserving");
    }

    #[test]
    fn en_double_coda_bounded_and_dropped_when_starved() {
        let d = en_dicts();
        let arr = build_arrays_daw(&[en_evt("R", 0, 10), en_evt("fined", 69, 50)], &d).unwrap();
        assert_eq!(arr.phon, vec!["SP", "f", "aɪ", "n", "d"]);
        // f (voiceless fricative, long-bucket p75) targets 7, coda n targets 4, the stop d 3 —
        // the 760ms flat [d] is gone AND the consonants are no longer one flat size.
        assert_eq!(arr.phone_dur, vec![3, 7, 43, 4, 3], "codas at their own measured targets");
        // a 3-frame note can't fit any coda at the 2-frame minimum → both drop, the nucleus survives.
        let tiny = build_arrays_daw(&[en_evt("R", 0, 10), en_evt("fined", 69, 3)], &d).unwrap();
        assert_eq!(tiny.phon, vec!["SP", "f", "aɪ"], "starved codas DROP (never a 1-frame OOD phone)");
        assert_eq!(tiny.phone_dur.iter().sum::<i64>(), 13, "still frame-conserving");
    }

    // Conservation invariant across a mixed score (the old split_dur could INFLATE Σ beyond the
    // timeline on short multi-phone notes, pushing every later note off the grid).
    #[test]
    fn daw_allocation_is_frame_conserving() {
        let d = en_dicts();
        let score = [
            en_evt("R", 0, 4), en_evt("mine", 69, 3), en_evt("fined", 71, 2), en_evt("R", 0, 6),
            en_evt("mine", 60, 13), en_evt("fined", 62, 25),
        ];
        let arr = build_arrays_daw(&score, &d).unwrap();
        let total: i64 = score.iter().map(|e| e.frames).sum();
        assert_eq!(arr.phone_dur.iter().sum::<i64>(), total, "Σ phone_dur == Σ event frames, always");
        assert!(arr.phone_dur.iter().all(|&d| d >= 1), "no 0-frame phone is ever emitted");
    }

    // S83 review #0: a borrowed onset that can't reach the 2-frame minimum (drained lender + no
    // nucleus spare) DROPS and hands its frames BACK to the lender — conservation survives the
    // failed pre-roll (a kept 1-frame phone would be the OOD class; a kept borrow with a dropped
    // phone would silently shift the timeline).
    #[test]
    fn starved_onset_drops_and_returns_its_borrowed_frames() {
        // geminate-shaped [ʔ,t,a] (raw-IPA override) on a 2-frame note after a 3-frame vowel: the
        // lender spares only 1 frame, the nucleus (min(fr,2)=2) has no spare → the 1-frame [t]
        // drops and its borrowed frame RETURNS to the lender (ʔ got nothing and drops too).
        let tta = g2p::ScoreEvt {
            lyric: "x", note_num: 62, frames: 2, lang: g2p::Lang::Ja, phoneme_input: Some("ʔ t a"),
        };
        let daw = build_arrays_daw(&[g2p::ScoreEvt::ja(&("あ", 60, 3)), tta], &NoDicts).unwrap();
        assert_eq!(daw.phon, vec!["a", "a"]);
        assert_eq!(daw.phone_dur, vec![3, 2], "borrowed frame returned to the lender");
    }

    // S83 review #1: medial phones follow the same ≥2-or-DROP policy as codas (never a 1-frame
    // phone); raw-IPA override [p i u] exercises the medial branch without a western dictionary.
    #[test]
    fn medial_vowels_get_two_frames_or_drop() {
        let evt = |fr| g2p::ScoreEvt {
            lyric: "x", note_num: 60, frames: fr, lang: g2p::Lang::Ja, phoneme_input: Some("p i u"),
        };
        let a10 = build_arrays_daw(&[evt(10)], &NoDicts).unwrap();
        assert_eq!(a10.phon, vec!["p", "i", "u"]);
        assert_eq!(a10.phone_dur, vec![2, 3, 5], "medial ≥2; onset supplemented in-note at score start");
        let a3 = build_arrays_daw(&[evt(3)], &NoDicts).unwrap();
        assert_eq!(a3.phon, vec!["u"], "sub-minimum medial AND onset drop — never a 1-frame phone");
        assert_eq!(a3.phone_dur, vec![3]);
    }

    // S83 refined-fix: a medial CONSONANT (a later syllable's onset flattened onto one note) gets
    // its own measured onset target — the old flat 2..4 share left refined's f inaudible.
    #[test]
    fn medial_syllable_onset_gets_its_measured_target() {
        let evt = g2p::ScoreEvt {
            lyric: "x", note_num: 60, frames: 50, lang: g2p::Lang::Ja,
            phoneme_input: Some("ɹ ə f aɪ n d"), // refined's shape as a raw-IPA override
        };
        let arr = build_arrays_daw(&[g2p::ScoreEvt::ja(&("R", 0, 10)), evt], &NoDicts).unwrap();
        assert_eq!(arr.phon, vec!["SP", "ɹ", "ə", "f", "aɪ", "n", "d"]);
        // onset ɹ pre-rolls its long-bucket 7 from the rest; medial vowel ə keeps the small share;
        // medial f takes its own onset target 7 (was ≤4); codas n/d at 4/3; aɪ gets the remainder.
        assert_eq!(arr.phone_dur, vec![3, 7, 4, 7, 32, 4, 3]);
        assert_eq!(arr.phone_dur.iter().sum::<i64>(), 60, "frame-conserving");
    }

    // M3 widened: an EN nucleus (aɪ) below the vowel floor borrows from a following rest exactly
    // like a JA vowel (the old tbl::VOWEL_SET check covered only the 5 JA vowels).
    #[test]
    fn m3_borrow_covers_en_nuclei() {
        let d = en_dicts();
        let arr = build_arrays_daw(&[en_evt("mine", 69, 3), en_evt("R", 0, 40)], &d).unwrap();
        let ai = arr.phon.iter().position(|&p| p == "aɪ").unwrap();
        assert_eq!(arr.phone_dur[ai], VOWEL_MIN_FRAMES, "short EN nucleus borrowed up to the floor");
        assert_eq!(arr.phone_dur.iter().sum::<i64>(), 43, "borrow is zero-sum");
    }

    // ── S66 zh sustain fix (user bug: [wang][-] sang "wang wang") ──
    // Every per-phone entry is a fresh articulation to ScoreToCV, so a zh hold re-emitting the
    // carrier final re-onsets it (ɑŋ after wang ≈ singing the syllable again). Same-pitch hold
    // → EXTEND the carrier entry; pitch-change (melisma) hold → glide-stripped vocalic tail.
    struct ZhOnly(g2p::ZhDict);
    impl g2p::DictSource for ZhOnly {
        fn zh(&self) -> Result<&g2p::ZhDict> {
            Ok(&self.0)
        }
        fn words(&self, _lang: g2p::Lang) -> Result<&g2p::WordDict> {
            Err(UtaiError::Inference("VOCAL_DICT_MISSING: fixture".into()))
        }
    }
    fn zh_dicts() -> ZhOnly {
        ZhOnly(g2p::ZhDict::from_tsv("wang\tw ang\nxiang\tx iang\n", "", ""))
    }
    fn zh_evt(lyric: &'static str, note_num: i64, frames: i64) -> g2p::ScoreEvt<'static> {
        g2p::ScoreEvt { lyric, note_num, frames, lang: g2p::Lang::Zh, phoneme_input: None }
    }

    #[test]
    fn zh_same_pitch_sustain_extends_the_final() {
        let d = zh_dicts();
        let a = build_arrays_daw(&[zh_evt("wang", 60, 20), zh_evt("-", 60, 30)], &d).unwrap();
        assert_eq!(a.phon, vec!["w", "ɑŋ"], "the hold adds NO phone entry (no re-articulation)");
        assert_eq!(a.phone_dur.iter().sum::<i64>(), 50, "total frames (timeline) preserved");
        assert!(*a.phone_dur.last().unwrap() >= 30, "the hold's frames extended the final");
        // chained same-pitch holds keep extending
        let b = build_arrays_daw(
            &[zh_evt("wang", 60, 20), zh_evt("-", 60, 30), zh_evt("-", 60, 10)],
            &d,
        )
        .unwrap();
        assert_eq!(b.phon, vec!["w", "ɑŋ"]);
        assert_eq!(b.phone_dur.iter().sum::<i64>(), 60);
    }

    #[test]
    fn zh_melisma_sustain_emits_glide_stripped_tail() {
        let d = zh_dicts();
        // xiang = [ɕ, iɑŋ]; the pitch-changed hold re-emits the glide-stripped tail ɑŋ (NOT iɑŋ,
        // which would re-articulate the glide ≈ "yang" again), at the NEW pitch.
        let a = build_arrays_daw(&[zh_evt("xiang", 60, 20), zh_evt("-", 62, 30)], &d).unwrap();
        assert_eq!(a.phon, vec!["ɕ", "iɑŋ", "ɑŋ"]);
        assert_eq!(a.note_pitch, vec![60, 60, 62]);
        // …and a FURTHER same-pitch hold extends that melisma entry instead of re-emitting.
        let b = build_arrays_daw(
            &[zh_evt("xiang", 60, 20), zh_evt("-", 62, 30), zh_evt("-", 62, 10)],
            &d,
        )
        .unwrap();
        assert_eq!(b.phon, vec!["ɕ", "iɑŋ", "ɑŋ"]);
        assert_eq!(b.phone_dur.iter().sum::<i64>(), 60);
    }

    #[test]
    fn zh_hold_phone_glide_strip_table() {
        assert_eq!(zh_hold_phone("uɑŋ"), "ɑŋ");
        assert_eq!(zh_hold_phone("iaʊ"), "aʊ");
        assert_eq!(zh_hold_phone("ua"), "a");
        assert_eq!(zh_hold_phone("in"), "in", "bare-nasal remainder → keep the full final");
        assert_eq!(zh_hold_phone("iŋ"), "iŋ");
        assert_eq!(zh_hold_phone("ɑŋ"), "ɑŋ", "glide-free finals hold unchanged");
        assert_eq!(zh_hold_phone("i"), "i");
    }

    #[test]
    fn ja_sustain_keeps_legacy_repeated_vowel() {
        // ja is the model's TRAINED hold convention (repeated bare vowel) and the Phase-1c parity
        // anchor — the zh fix must not leak into it.
        let a = daw_ja(&[("か", 60, 80), ("ー", 60, 40)]).unwrap();
        assert_eq!(a.phon, vec!["k", "a", "a"], "ja hold still re-emits the carrier vowel entry");
    }

    // §9.5 single Rust classifier: `classify_lyric` (exposed via the validate_lyrics command) MUST agree
    // with `lyric_to_phones` (which build_arrays uses) so the editor's verdict == the render's.
    #[test]
    fn classify_lyric_matches_render() {
        assert!(matches!(classify_lyric("R"), LyricClass::Rest));
        assert!(matches!(classify_lyric(""), LyricClass::Rest));
        assert!(matches!(classify_lyric("ー"), LyricClass::Sustain));
        assert!(matches!(classify_lyric("AP"), LyricClass::Breath));
        assert!(matches!(classify_lyric("ap"), LyricClass::Breath));
        assert!(matches!(classify_lyric("zzzz"), LyricClass::Unknown));
        match classify_lyric("か") {
            LyricClass::Phones { phones } => assert_eq!(phones, vec!["k", "a"]),
            other => panic!("か should classify as phones [k,a], got {:?}", other),
        }
    }

    #[test]
    fn tables_are_well_formed() {
        assert_eq!(tbl::PHONE_TO_ID.len(), 210, "vocab size");
        assert_eq!(phone_to_id_map()["SP"], tbl::SP_ID);
        assert_eq!(phone_to_id_map()["AP"], tbl::AP_ID);
        // no duplicate keys collapsed the maps (base tables + the generated S58 EXTRA rows)
        assert_eq!(phone_to_id_map().len(), tbl::PHONE_TO_ID.len());
        assert_eq!(kana_map().len(), tbl::KANA.len() + super::super::g2p_tables::KANA_EXTRA.len());
        assert_eq!(r2ipa_map().len(), tbl::R2IPA.len() + super::super::g2p_tables::R2IPA_EXTRA.len());
    }

    // ── Phase 1d GATE: end-to-end Rust → ORT → cv, matched ≤1e-3 to Python-ORT (score2cv_cv_ref.rs).
    // Needs the 181MB models (data/models/aux) + the dev ORT dll (runtime/ort) — hence #[ignore]; run:
    //   cargo test --lib inference::score2cv::tests::onnx_cv_parity_cpu -- --ignored --nocapture
    // Forces the CPU EP so numerics equal the Python CPUExecutionProvider reference exactly. ──
    #[test]
    #[ignore]
    fn onnx_cv_parity_cpu() {
        use super::super::engine::DeviceConfig;
        use super::super::score2cv_cv_ref as cvref;
        use std::path::{Path, PathBuf};

        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let dll = root.join("../runtime/ort/onnxruntime.dll");
        assert!(dll.exists(), "ORT dll missing at {} (dev runtime required)", dll.display());
        match ort::init_from(&dll) {
            Ok(b) => {
                let _ = b.commit();
            }
            Err(e) => panic!("ort init_from failed: {e}"),
        }

        let engine = OnnxEngine::new();
        engine.set_device(DeviceConfig::Cpu); // deterministic; matches the Python CPU reference

        let arr = build_arrays(pr::SCORE).unwrap();
        let chunks = chunk_at_sp(&arr, 400);

        for (dim, model, refs) in [
            (768usize, "score2cv_768.onnx", cvref::REF_768),
            (256usize, "score2cv_256.onnx", cvref::REF_256),
        ] {
            let path: PathBuf = root.join("../data/models").join(crate::models::AUX_DIR_NAME).join(model);
            assert!(path.exists(), "model missing: {}", path.display());
            let sid = engine.load_model_with(&path, false).unwrap();
            assert_eq!(chunks.len(), refs.len(), "chunk count vs reference");
            for (ci, chunk) in chunks.iter().enumerate() {
                let cv = run_score2cv(&engine, &sid, chunk, dim, 49, 2).unwrap();
                let r = &refs[ci];
                assert_eq!(cv.nrows(), r.t, "{} c{} T", model, ci);
                assert_eq!(cv.ncols(), r.dim, "{} c{} dim", model, ci);
                let flat = cv.as_slice().expect("cv is contiguous");
                let mut worst = 0.0f32;
                for (&i, &v) in r.idx.iter().zip(r.val) {
                    worst = worst.max((flat[i] - v).abs());
                }
                assert!(worst <= 1e-3, "{} c{}: worst sampled cv diff {:.3e} > 1e-3", model, ci, worst);
                let sum: f64 = flat.iter().map(|&x| x as f64).sum();
                let sumsq: f64 = flat.iter().map(|&x| (x as f64) * (x as f64)).sum();
                assert!((sum - r.sum).abs() <= 0.1 + r.sum.abs() * 1e-4, "{} c{}: sum {} vs {}", model, ci, sum, r.sum);
                assert!((sumsq - r.sumsq).abs() <= 0.1 + r.sumsq * 1e-4, "{} c{}: sumsq {} vs {}", model, ci, sumsq, r.sumsq);
                eprintln!("[1d] {} chunk{}: T={} dim={} sampled-worst={:.2e} sum={:.2} PASS", model, ci, r.t, r.dim, worst, sum);
            }
        }
    }
}
