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
/// S86 TRAINING-ALIGNMENT DIVERGENCE — the ONLY place Utai deliberately departs from the upstream
/// `render_ust.py` JA romaji table. Kept here (hand-written) and NOT in the generated tables, so a
/// regeneration can never silently drop it and a reader can never mistake it for a mirrored row.
///
/// `に` — upstream inference emits the palatal nasal `ɲ i`, but the model was TRAINED on `n i`:
///   * the training-side map (`MBS2H/src/preprocessing/phoneme_vocab.py::JA_ROMAJI_TO_IPA`) only has
///     `"n"→n` and `"ny"→ɲ`, and the HTS/sinsy `.lab` files write に as the two phones `n` `i`;
///   * raw lab counts over the 6 JA datasets: `n`=3234 vs `ny`=37 — `ny` is にゃ/にゅ/にょ only;
///   * final-training-set frames: `n`=4338 vs `ɲ`=**92**, i.e. every に was asking the model for an
///     embedding it had seen ~47× less often, for one of the most frequent morae in the language;
///   * `dict_fixes.py` has a JA allophone rule (D) for ひ (`h`→`ç` before i) and deliberately NONE
///     for に, so there is no training-side counterpart that would justify ɲ.
/// にゃ/にゅ/にょ keep `ɲ` — those really are `ny` in the labs. This override changes JA render output
/// and therefore intentionally breaks bit-parity with upstream; the parity fixtures do not contain に.
const R2IPA_TRAINING_OVERRIDE: &[(&str, &[&str])] = &[("ni", &["n", "i"])];

fn r2ipa_map() -> &'static HashMap<&'static str, &'static [&'static str]> {
    static M: OnceLock<HashMap<&'static str, &'static [&'static str]>> = OnceLock::new();
    M.get_or_init(|| {
        tbl::R2IPA
            .iter()
            .chain(super::g2p_tables::R2IPA_EXTRA)
            .chain(R2IPA_TRAINING_OVERRIDE)
            .copied()
            .collect()
    })
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
    // ⚠ Keep `rest`/`sil`/`pau` HERE: this function is the bit-parity port of upstream
    // `render_ust.lyric_to_phones` and the golden/parity gates compare against it. The DAW's own
    // reserved-token set (`g2p::token_class`) is deliberately narrower (S86) — do NOT sync the two.
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
        // ── UTAI EXTENSION (S86, beyond render_ust.py): PARSE the whole kana string ──
        // Upstream fell back to the first TWO (then ONE) kana and silently DROPPED the rest, so a
        // multi-mora lyric on one note sang only its head mora — ずっと→[z ɯ], きっと→[k i],
        // がっこう→[ɡ a] — with no OOV and no red mark. But writing 「ずっと」 or 「っと」 on one note is a
        // LEGITIMATE way to enter lyrics (real UST files do it), so the answer is to sing all of it,
        // not to reject it: `kana_tokenize` consumes the string mora by mora (longest-match, sokuon,
        // moraic n, 外来拗音, ー) and returns every phone. Only a string it cannot fully consume falls
        // through — to the romaji chain below, and ultimately to a LOUD `Unknown`.
        // Single morae never reach here (`kana.get(s0)` above already matched), and non-kana strings
        // fail at the first char, so romaji (`tta`, `ka`, `tchi`) and every Phase-1c parity input are
        // untouched.
        if let Some(v) = kana_tokenize(s0) {
            return Lyric::Phones(v);
        }
        s0.to_string()
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

/// UTAI EXTENSION (S86): consume a WHOLE kana string mora by mora → the concatenated phones, so
/// 「ずっと」/「っと」/「ずーっと」 on one note sing in full instead of being truncated to their head mora.
///
/// LONGEST-MATCH-FIRST is load-bearing: 「ぁぃぅぇぉ」 are themselves KANA keys, so a shortest-first scan
/// would SILENTLY parse ふぁい as ふ+ぁ+い ([ɸ ɯ a i]) instead of ふぁ+い ([ɸ a i]) — a wrong parse that
/// still "succeeds". Plain greedy needs no backtracking here: every multi-char unit ends in a small
/// kana (ゃゅょ / ぁぃぅぇぉ), and a small kana can never START a mora, so a longer match never steals a
/// character a shorter parse would have needed.
///
/// ALL-OR-NOTHING: any position that cannot be consumed returns None (→ the caller's romaji chain,
/// and ultimately a LOUD `Unknown`). A partial parse is precisely the silent truncation this replaces.
fn kana_tokenize(s: &str) -> Option<Vec<&'static str>> {
    let chars: Vec<char> = s.chars().collect();
    let (kana, r2ipa) = (kana_map(), r2ipa_map());
    let mut out: Vec<&'static str> = Vec::new();
    let mut i = 0usize;
    while i < chars.len() {
        let mut took = 0usize;
        // 3 covers a 2-char base + small vowel (しゃぁ); KANA keys themselves are at most 2 chars.
        for w in (1..=3.min(chars.len() - i)).rev() {
            let slice: String = chars[i..i + w].iter().collect();
            if w >= 2 {
                if let Some(v) = foreign_kana_phones(&slice) {
                    out.extend(v); // S69 外来拗音 keeps the precedence it has in `lyric_to_phones`
                    took = w;
                    break;
                }
            }
            if w <= 2 {
                if let Some(&romaji) = kana.get(slice.as_str()) {
                    if let Some(&seq) = r2ipa.get(romaji.to_lowercase().as_str()) {
                        out.extend_from_slice(seq);
                        took = w;
                        break;
                    }
                }
            }
        }
        if took == 0 {
            match chars[i] {
                // sokuon: the geminate closure, same phone the 「っ」-on-its-own-note branch emits
                'っ' => out.push("ʔ"),
                // 長音符 contributes NO phone. Every training label lengthens a vowel by DURATION on
                // ONE phone — 「あー」 is [a] held longer, never [a a] — so emitting a second copy would
                // be out of distribution. The note's own frame count already carries the length.
                'ー' if !out.is_empty() => {}
                // Anything else ends the kana run. A non-kana TAIL is the UTAU appended-voicebank /
                // CVVC alias convention — 「あ弱」「か強」「か_G3」「あ t」 mean "mora あ, voicebank flavour X"
                // and must sing the mora (the repo's own .ust corpus has 26 such notes). So we keep
                // what we consumed instead of failing; a lyric where NOTHING parsed still returns
                // None → the romaji chain → a LOUD `Unknown`.
                _ => break,
            }
            took = 1;
        }
        i += took;
    }
    (!out.is_empty()).then_some(out)
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
//
// S89: the pre-roll is a per-track SWITCH (`ArticulationTiming`). Turning it off does NOT go back to
// `split_dur` — see that enum's doc for why conservation forbids it. It only redirects where the onset's
// frames come from: the note's own nucleus instead of the previous phone. Everything else on this page
// (targets, buckets, ≥2-or-drop, LAST-first, nucleus-remainder, coda bounds) is shared by both arms.
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

/// How many frames this pass may hand out, and which note-length bucket the MEASURED duration priors
/// should be read at. On the Auto arm they are the same number. On the InNote arm they diverge: the
/// onset is reserved out of the note BEFORE this pass runs, so `spendable` shrinks while the bucket key
/// must keep describing the note the singer actually sees.
///
/// ⚠ Two bare `i64`s in a row is the shape S85 shipped a bug through — hence the named fields.
#[derive(Debug, Clone, Copy)]
struct NoteBudget {
    /// The note's own frame count. Bucket key for `onset_target_frames`/`coda_target_frames` ONLY.
    note_frames: i64,
    /// Frames this pass may actually distribute (nucleus remainder included).
    spendable: i64,
}

/// In-note allocation for one note's phones: medial + coda get bounded shares, the nucleus takes the
/// remainder, onset positions are left 0 (the caller funds them — by borrowing from the previous phone
/// on the Auto arm, or by reserving out of `b.spendable` BEFORE this call on the InNote arm). Returns
/// per-phone durations aligned to `ph`; entries may be 0 (dropped medial/coda / unfunded onset) — the
/// caller skips 0-duration phones at emission. Σ(durs) ≤ b.spendable always.
fn allocate_in_note(ph: &[&'static str], b: NoteBudget, onset_end: usize, nuc: usize) -> Vec<i64> {
    let NoteBudget { note_frames, spendable: fr } = b;
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
            onset_target_frames(ph[i], note_frames)
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
        let want: i64 = ph[nuc + 1..].iter().map(|p| coda_target_frames(p, note_frames)).sum();
        let mut budget = want.min(fr - nuc_floor - used).min(fr * 2 / 5);
        for i in (nuc + 1..n).rev() {
            let give = coda_target_frames(ph[i], note_frames).min(budget);
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
    assemble_arrays(&evts, &resolved, Assembly::Parity)
}

/// S89: WHERE a note's onset consonants get their frames — the ② render's per-track
/// 「自动音素时序」 switch (`VocalTrackParams.consonantPreroll`).
///
/// ⚠ This is NOT "S83 allocator vs the legacy `split_dur`". `split_dur` is the Phase-1c parity port
/// and it is **not frame-conserving** (`split_dur(2, 5) == [1,1,1,1,1]`, Σ 5 > 2 frames). Σ phone_dur
/// == Σ event frames is load-bearing far downstream: `build_note_hz`/`build_note_param` take a
/// conserving FAST PATH (`t_total == f0.cents.len()`) and fall into the group remap otherwise —
/// which, per its own comment, samples the vowel ~2 frames LATE, i.e. it would silently re-apply the
/// very head start this switch exists to remove. The phoneme lane's x-coordinate contract and
/// "stem length == segment timeline" rest on the same invariant. So BOTH arms below conserve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArticulationTiming {
    /// S83 crown knife: onset consonants are PRE-ROLLED before the beat by borrowing frames from the
    /// previously emitted phone, so the NUCLEUS lands on the beat. 92.2% of training frames annotate
    /// note onsets at the consonant start (singers place the consonant ahead of the beat), and an
    /// OpenUtau voicebank does the same thing via its oto preutterance.
    Auto,
    /// Every phone stays INSIDE its own note; nothing is borrowed from the previous phone. For a
    /// score whose author already placed the consonants ahead of the beat BY HAND — UTAU CVVC/VCCV
    /// alias scores are transition units carrying their own preutterance compensation — pre-rolling
    /// again applies the same head start TWICE.
    /// ⚠ Known cost, by construction: with no lender, a note too short to fund a 2-frame onset out
    /// of its own budget drops it (the same ≥2-or-drop policy codas already use). That is the honest
    /// reading of "no borrowing"; the phoneme lane shows it. The onset is funded FIRST and clamped to
    /// half the note, so this only bites on genuinely tiny notes and never on a long one.
    InNote,
}

/// Which assembly the shared core is running. Replaces the old `(cap_rests, daw)` bool pair — only
/// two of its four combinations were ever legal, and a same-typed bool pair across a call boundary
/// is precisely the shape S85 shipped a bug through ([[project_v2_session85]]).
enum Assembly {
    /// Phase-1c bit-parity port of `render_ust.build_arrays`: rests capped, `split_dur`, no pre-roll.
    Parity,
    /// ② DAW render: rests uncapped (stem aligns to the tick timeline), S83 syllable-aware
    /// allocation, M3 short-vowel rest-borrow.
    Daw(ArticulationTiming),
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
pub fn build_arrays_daw(
    score: &[g2p::ScoreEvt],
    dicts: &dyn g2p::DictSource,
    timing: ArticulationTiming,
) -> Result<ScoreArrays> {
    let resolved = g2p::resolve_score(score, dicts)?;
    // ② DAW: rests uncapped + S83 syllable-aware allocation (onset pre-roll / nucleus-remainder /
    // bounded coda) + short-note borrow-time (M3). Frame-CONSERVING: Σ phone_dur == Σ evt frames.
    assemble_arrays(score, &resolved, Assembly::Daw(timing))
}

/// THE single array-assembly core (S58): resolved per-note phones → the model's per-phone arrays.
/// Frame policy, id mapping, and note grouping — grouping keys on (pitch, RUN LANGUAGE) so a group never
/// spans a language cut (single-language scores group exactly as before). Both `build_arrays` (parity,
/// capped, legacy `split_dur`) and `build_arrays_daw` (DAW, uncapped, S83 syllable-aware allocation +
/// onset pre-roll) feed through here — one implementation, proven by the 1c bit-parity gate.
fn assemble_arrays(
    score: &[g2p::ScoreEvt],
    resolved: &[g2p::ResolvedNote],
    mode: Assembly,
) -> Result<ScoreArrays> {
    let cap_rests = matches!(mode, Assembly::Parity);
    let daw_timing = match mode {
        Assembly::Parity => None,
        Assembly::Daw(t) => Some(t),
    };
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
                if let Some(timing) = daw_timing {
                    // ── S83 syllable-aware allocation + onset pre-roll (see the block comment above
                    //    allocate_in_note for the full rationale) ──
                    let n = ph.len();
                    let fr = fr.max(1);
                    let nuc = ph.iter().rposition(|p| is_nucleus_phone(p)).unwrap_or(n - 1);
                    let onset_end = ph.iter().position(|p| is_nucleus_phone(p)).unwrap_or(n - 1).min(nuc);
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
                    // ★The cap keys on the NOTE's own frame count, not on who lends, so it holds
                    // identically on the InNote arm (same measured C/V split, same note length).
                    let target = |p: &'static str| {
                        let t = onset_target_frames(p, fr);
                        if fr <= 5 {
                            t.min(2)
                        } else {
                            t
                        }
                    };
                    // ── preroll OFF: the onset is funded by THIS note, and it is funded FIRST ──
                    //
                    // ★ORDER IS THE WHOLE DESIGN. The medial/coda pass below is bounded only by the
                    // nucleus's 2-frame floor, so serving the onset *after* it (the first cut of this
                    // switch) let a multi-syllable word — or a multi-mora kana on one note, which is
                    // ordinary UST practice — starve its WORD-INITIAL consonant to exactly zero at
                    // perfectly normal note lengths, while word-INTERNAL consonants kept their full
                    // measured target: `refined`@400ms sang "efined", `ずっと`@500ms lost its z.
                    // Two adversarial-review dimensions found that independently. Funding the onset
                    // first also makes the result MONOTONE in note length (lengthening a note can no
                    // longer delete a consonant).
                    //
                    // ★The clamp is the Auto arm's own UTAU-style structural half, re-aimed at the
                    // note instead of at a neighbour: the onset cluster may take at most half the
                    // note and must leave ≥ SUNG_KEEP_MIN. My first cut invented a `fr*2/5` floor
                    // instead — which integer-truncates to exactly 2 across the WHOLE short bucket
                    // (fr ≤ 7), i.e. no protection at all where it was needed most: か@6 came out
                    // k4/a2 and し@7 came out ɕ5/i2, the 2-frame vowel S84 measured as the collapse
                    // region. Reusing the shipped, ear-validated half-clamp gives k3/a3 and ɕ4/i3.
                    let mut reserved = 0i64;
                    let mut onset_durs = vec![0i64; n];
                    if onset_end > 0 && timing == ArticulationTiming::InNote {
                        let avail = (fr - SUNG_KEEP_MIN).max(0).min((fr + 1) / 2);
                        let want: i64 = ph[..onset_end].iter().map(|&p| target(p)).sum();
                        let mut left = want.min(avail);
                        reserved = left;
                        // LAST-first, exactly as the pre-roll arm: the consonant touching the vowel
                        // carries the syllable's identity, so a starved cluster sheds its OUTERMOST
                        // member first.
                        for i in (0..onset_end).rev() {
                            let give = left.min(target(ph[i]));
                            onset_durs[i] = give;
                            left -= give;
                        }
                        // all-or-nothing top-up to the 2-frame minimum out of what is still
                        // unallocated (i.e. the nucleus's future remainder), never below the
                        // nucleus's own floor — same policy as the pre-roll arm's supplement.
                        let nuc_floor = fr.min(2);
                        for i in (0..onset_end).rev() {
                            if onset_durs[i] >= CODA_MIN_FRAMES {
                                continue;
                            }
                            let need = CODA_MIN_FRAMES - onset_durs[i];
                            if fr - reserved - need >= nuc_floor {
                                onset_durs[i] += need;
                                reserved += need;
                            }
                        }
                        // sub-minimum onsets DROP (a 1-frame phone is categorically OOD); their
                        // frames simply stay in the budget the pass below distributes.
                        for d in onset_durs.iter_mut().take(onset_end) {
                            if *d > 0 && *d < CODA_MIN_FRAMES {
                                reserved -= *d;
                                *d = 0;
                            }
                        }
                    }
                    // Auto: reserved == 0 ⇒ `spendable == note_frames` and every onset slot is 0,
                    // which is exactly what this call produced before the switch existed.
                    let mut durs =
                        allocate_in_note(ph, NoteBudget { note_frames: fr, spendable: fr - reserved }, onset_end, nuc);
                    durs[..onset_end].copy_from_slice(&onset_durs[..onset_end]);
                    if onset_end > 0 && timing == ArticulationTiming::Auto {
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
                        // (the fr≤5 target cap referenced above is hoisted into `target`, shared with
                        // the InNote arm — same measured justification, same note-length key.)
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
    // ★Runs on BOTH articulation-timing arms (S89): this is not the onset pre-roll. It never moves a
    // consonant ahead of a beat — it only lets a starved vowel eat into the SILENCE that follows it,
    // which is orthogonal to "the author already placed the consonants" and is what keeps a short
    // CVVC note audible at all.
    if daw_timing.is_some() {
        // S83: the vowel test widened from tbl::VOWEL_SET (the 5 JA vowels) to the full nucleus
        // classifier — an EN aɪ / zh final / syllabic n̩ deserves the same floor as a JA vowel.
        for i in 0..phon.len() {
            // S84 D 刀: a nucleus-less note's SOLE VOICED phone (ん = [ɴ], the moraic nasal — it
            // carries the note like a vowel would) gets the same floor: the S84 audit found 17
            // lone-ɴ notes with no M3 eligibility, drained to 2-3 frames. Voiceless sole phones
            // (っ = [ʔ]) stay excluded — stretching a glottal closure into a rest lengthens
            // silence, not song. (A mid-run drained ɴ with no following rest still gets nothing —
            // same as any drained vowel; only the rest-borrow path exists here.)
            let sole_voiced_of_event = (i == 0 || pevt[i - 1] != pevt[i])
                && (i + 1 >= pevt.len() || pevt[i + 1] != pevt[i])
                && !is_voiceless_phone(phon[i]);
            if npitch[i] > 0
                && (is_nucleus_phone(phon[i]) || sole_voiced_of_event)
                && pdur[i] < VOWEL_MIN_FRAMES
            {
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
// ─── S84 E 刀: vowel-clarity articulation oversampling (「渲染长音素再缩短」, cv 域) ───
//
// Fast-run vowels (≤4 frames) undershoot their articulation target — S2CV's det trajectory never
// REACHES /a/ in 40-80 ms (measured: ま's /a/ F1 646 vs cover 1070; the S84 closed-vowel
// deviations). The UTAU-analogy fix, in the ONLY domain where it is artifact-free (cv — audio-domain
// time-compression would ring phase/stretch artifacts): render the chunk with short nuclei INFLATED
// to VOWEL_CLARITY_INFLATE frames (the model, seeing a longer duration input, computes a fully
// articulated vowel), then resample the cv rows back to the true durations (uniform center-aligned
// nearest). Measured: ま F1 646→836 (E on/off, robust), user-ear-confirmed "开口部分出来了".
// Sampling-window refinements (tail-trim / steady-core) were measured PERCEPTUALLY INERT (band
// metric flat, ear-confirmed) — hence deliberately absent: the benefit comes from the inflated
// duration INPUT changing the vowel's cv target, not from which rows get picked. The residual
// "mai" coloring (band gap vs cover, invariant to inflate amount) is the det contextual
// undershoot — out of this knife's reach (menu ④ manual/auto timing).
const VOWEL_CLARITY_INFLATE: i64 = 6;
const VOWEL_CLARITY_MAX_DUR: i64 = 4;
/// The exported ScoreToCV graph carries a FIXED 8000-row frame-axis positional encoding
/// (RelativePositionalEncoding max_len; S84 review) — a twin that would exceed it falls back to
/// the plain call so "renders with clarity OFF ⇒ renders with clarity ON" holds even on
/// pathological rest-less mega-chunks (the plain path itself errors loudly past 8000).
const VOWEL_CLARITY_MAX_T: usize = 8000;

/// The inflation plan for one chunk: `None` = nothing qualifies (caller uses the plain path —
/// bit-exact off-switch by construction); `Some(durs)` = per-phone durations with every
/// qualifying nucleus raised to the inflate target. Scope = the S84-validated regime ONLY: the
/// FINAL nucleus of its source event at 1..=4 frames (fast-run CV notes / EN closed syllables).
/// MEDIAL vowels (refined's ə, più's i) are allocation-short at ANY tempo — no undershoot
/// premise, never ear-validated — and stay untouched (S84 review; possible future extension).
fn clarity_inflated_durs(
    phon: &[&'static str],
    note_pitch: &[i64],
    phone_dur: &[i64],
    evt: &[usize],
) -> Option<Vec<i64>> {
    let mut durs = phone_dur.to_vec();
    let mut any = false;
    for i in 0..phon.len() {
        let final_nucleus_of_event = is_nucleus_phone(phon[i])
            && !(i + 1..phon.len()).any(|j| evt[j] == evt[i] && is_nucleus_phone(phon[j]));
        if note_pitch[i] > 0
            && final_nucleus_of_event
            && (1..=VOWEL_CLARITY_MAX_DUR).contains(&phone_dur[i])
        {
            durs[i] = VOWEL_CLARITY_INFLATE.max(phone_dur[i]);
            any = true;
        }
    }
    any.then_some(durs)
}

/// Uniform center-aligned nearest resample of the inflated cv rows back onto the true per-phone
/// durations (row counts: Σpd_inf → Σpd_true).
fn clarity_resample(cv_inf: &Array2<f32>, pd_true: &[i64], pd_inf: &[i64]) -> Array2<f32> {
    let t_true: usize = pd_true.iter().map(|&d| d.max(0) as usize).sum();
    let mut cv = Array2::<f32>::zeros((t_true, cv_inf.ncols()));
    let (mut c_true, mut c_inf) = (0usize, 0usize);
    for k in 0..pd_true.len() {
        let d_true = pd_true[k].max(0) as usize;
        let d_inf = pd_inf[k].max(0) as usize;
        for j in 0..d_true {
            let src = c_inf + ((j as f64 + 0.5) * d_inf as f64 / d_true as f64) as usize;
            let src = src.min(c_inf + d_inf.saturating_sub(1)).min(cv_inf.nrows().saturating_sub(1));
            cv.row_mut(c_true + j).assign(&cv_inf.row(src));
        }
        c_true += d_true;
        c_inf += d_inf;
    }
    cv
}

/// ScoreToCV with the S84 vowel-clarity oversampling: build the inflated TWIN chunk (same phone
/// range/pitches/grouping), run the model once on it, resample the rows back. `phon`/`evt` = the
/// arr's slices for this chunk (Chunk carries only ids/groups). No qualifying nucleus, or a twin
/// past the model's positional cap ⇒ the PLAIN `run_score2cv` call.
/// note_dur is re-summed ONLY for groups that contain an inflated member — every other group
/// keeps the sliced full-array value verbatim. That matters for pitch-0 groups: SP|AP neighbours
/// group together and CAN span a chunk cut (S84 review — sung groups never do, (pitch,lang)
/// keying cuts at every SP/lang change), so a blanket in-chunk re-sum would silently change
/// breath-frame conditioning far from any inflated vowel.
pub fn run_score2cv_vowel_clarity(
    engine: &OnnxEngine,
    session_id: &str,
    chunk: &Chunk,
    phon: &[&'static str],
    evt: &[usize],
    dim: usize,
    speaker_id: i64,
    lang_id: i64,
) -> Result<Array2<f32>> {
    let Some(pd_inf) = clarity_inflated_durs(phon, &chunk.note_pitch, &chunk.phone_dur, evt) else {
        return run_score2cv(engine, session_id, chunk, dim, speaker_id, lang_id);
    };
    let t_inf: usize = pd_inf.iter().map(|&d| d.max(0) as usize).sum();
    if t_inf > VOWEL_CLARITY_MAX_T {
        return run_score2cv(engine, session_id, chunk, dim, speaker_id, lang_id);
    }
    let touched: std::collections::HashSet<i64> = (0..pd_inf.len())
        .filter(|&k| pd_inf[k] != chunk.phone_dur[k])
        .map(|k| chunk.note_to_phone[k])
        .collect();
    let mut nd_inf = chunk.note_dur.clone();
    for k in 0..pd_inf.len() {
        let g = chunk.note_to_phone[k];
        if touched.contains(&g) {
            nd_inf[k] = (0..pd_inf.len())
                .filter(|&j| chunk.note_to_phone[j] == g)
                .map(|j| pd_inf[j])
                .sum();
        }
    }
    let twin = Chunk {
        start: chunk.start,
        end: chunk.end,
        phonemes: chunk.phonemes.clone(),
        note_pitch: chunk.note_pitch.clone(),
        phone_dur: pd_inf.clone(),
        note_dur: nd_inf,
        note_to_phone: chunk.note_to_phone.clone(),
        t: t_inf,
        lang_id: chunk.lang_id,
        hard_seam: chunk.hard_seam,
    };
    let cv_inf = run_score2cv(engine, session_id, &twin, dim, speaker_id, lang_id)?;
    Ok(clarity_resample(&cv_inf, &chunk.phone_dur, &pd_inf))
}

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
    use crate::inference::g2p_alias::PhonemeSet;

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
        build_arrays_daw(&evts, &NoDicts, ArticulationTiming::Auto)
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

    // S84 E 刀: the vowel-clarity plan inflates only the FINAL nucleus of a sung event at ≤4
    // frames (the validated fast-run regime; medial vowels are allocation-short at any tempo and
    // stay untouched); the resample maps rows center-aligned; nothing-qualifies → None (the
    // off-path and no-op path are the same code).
    #[test]
    fn vowel_clarity_plan_and_resample() {
        let phon: Vec<&'static str> = vec!["SP", "k", "a", "a"];
        let pitch = vec![0i64, 60, 60, 60];
        let evt = vec![0usize, 1, 1, 2];
        // evt1's final nucleus (3fr) inflates; evt2's long one (12) doesn't; consonants never do.
        let durs = vec![10i64, 2, 3, 12];
        let plan = clarity_inflated_durs(&phon, &pitch, &durs, &evt).expect("one short nucleus qualifies");
        assert_eq!(plan, vec![10, 2, 6, 12]);
        // nothing qualifies → None (rest-only / long-only scores take the plain path).
        assert!(clarity_inflated_durs(&phon, &pitch, &[10, 2, 12, 12], &evt).is_none());
        // MEDIAL vowel exclusion (refined-shape on ONE event): the ə (short, medial) must NOT
        // inflate — only the final nucleus aɪ qualifies (here long → whole plan is None).
        let phon2: Vec<&'static str> = vec!["ɹ", "ə", "f", "aɪ"];
        let evt2 = vec![0usize, 0, 0, 0];
        assert!(
            clarity_inflated_durs(&phon2, &[60; 4], &[2, 3, 2, 12], &evt2).is_none(),
            "medial ə never inflates (unvalidated slow-note scope, S84 review)"
        );
        // …and when the final nucleus IS short, it inflates while the medial still doesn't.
        let plan2 = clarity_inflated_durs(&phon2, &[60; 4], &[2, 3, 2, 4], &evt2).unwrap();
        assert_eq!(plan2, vec![2, 3, 2, 6]);
        // resample: a 6-row phone down to 2 rows samples centers {1.5→1, 4.5→4}; identity spans
        // (pd_true == pd_inf) map 1:1.
        let mut cv = Array2::<f32>::zeros((8, 1));
        for r in 0..8 {
            cv[[r, 0]] = r as f32;
        }
        let out = clarity_resample(&cv, &[2, 2], &[6, 2]);
        assert_eq!(out.column(0).to_vec(), vec![1.0, 4.0, 6.0, 7.0]);
    }

    // S84 D 刀: a lone moraic-ん note ([ɴ], nucleus-less, VOICED) gets the M3 rest-borrow floor
    // like a vowel; a lone っ ([ʔ], voiceless) stays excluded (stretching a glottal closure into
    // a rest = longer silence, not song).
    #[test]
    fn m3_floor_covers_lone_hatsuon_but_not_sokuon() {
        let daw = daw_ja(&[("あ", 60, 10), ("ん", 60, 3), ("R", 0, 10)]).unwrap();
        assert_eq!(daw.phon, vec!["a", "ɴ", "SP"]);
        assert_eq!(daw.phone_dur, vec![10, VOWEL_MIN_FRAMES, 10 - (VOWEL_MIN_FRAMES - 3)]);
        assert_eq!(daw.phone_dur.iter().sum::<i64>(), 23, "frame-conserving");
        let daw2 = daw_ja(&[("あ", 60, 10), ("っ", 60, 3), ("R", 0, 10)]).unwrap();
        assert_eq!(daw2.phon, vec!["a", "ʔ", "SP"]);
        assert_eq!(daw2.phone_dur, vec![10, 3, 10], "voiceless sokuon never borrows the rest");
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
        g2p::ScoreEvt {
            lyric,
            note_num,
            frames,
            lang: g2p::Lang::En,
            phoneme_input: None,
            phoneme_set: PhonemeSet::Words,
        }
    }

    #[test]
    fn en_closed_syllable_nucleus_takes_the_remainder() {
        let d = en_dicts();
        // R(10) mine(50) R(10) — a 1-second note at 50fps.
        let score = [en_evt("R", 0, 10), en_evt("mine", 69, 50), en_evt("R", 0, 10)];
        let arr = build_arrays_daw(&score, &d, ArticulationTiming::Auto).unwrap();
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
        let arr = build_arrays_daw(&[en_evt("R", 0, 10), en_evt("fined", 69, 50)], &d, ArticulationTiming::Auto).unwrap();
        assert_eq!(arr.phon, vec!["SP", "f", "aɪ", "n", "d"]);
        // f (voiceless fricative, long-bucket p75) targets 7, coda n targets 4, the stop d 3 —
        // the 760ms flat [d] is gone AND the consonants are no longer one flat size.
        assert_eq!(arr.phone_dur, vec![3, 7, 43, 4, 3], "codas at their own measured targets");
        // a 3-frame note can't fit any coda at the 2-frame minimum → both drop, the nucleus survives.
        let tiny = build_arrays_daw(&[en_evt("R", 0, 10), en_evt("fined", 69, 3)], &d, ArticulationTiming::Auto).unwrap();
        assert_eq!(tiny.phon, vec!["SP", "f", "aɪ"], "starved codas DROP (never a 1-frame OOD phone)");
        assert_eq!(tiny.phone_dur.iter().sum::<i64>(), 13, "still frame-conserving");
    }

    // ─── S89 「自动音素时序」 OFF (ArticulationTiming::InNote) ───
    //
    // Every expected number below is hand-derived from the priors table + the allocator's stated
    // rules, NOT copied from a run (S87: two of my expectations were wrong and the TEST caught the
    // arithmetic — that only works if the expectation is independent).
    //
    // mine = [m, aɪ, n] on a 50-frame note (long bucket): coda n targets 4, so allocate_in_note
    // leaves [_, 46, 4]. InNote then funds the onset out of the NUCLEUS: m targets 5 (long bucket),
    // the nucleus floor is max(2, 50*2/5) = 20 and 46 is well above it ⇒ m takes 5, the vowel 41.
    // ★The leading rest keeps ALL 10 of its frames — under Auto it lends 5 of them away. That one
    // number IS the feature: the author's note positions are left exactly where they were put.
    #[test]
    fn in_note_timing_funds_the_onset_from_its_own_nucleus() {
        let d = en_dicts();
        let score = [en_evt("R", 0, 10), en_evt("mine", 69, 50)];
        let off = build_arrays_daw(&score, &d, ArticulationTiming::InNote).unwrap();
        assert_eq!(off.phon, vec!["SP", "m", "aɪ", "n"]);
        assert_eq!(off.phone_dur, vec![10, 5, 41, 4]);
        // the note's own phones account for exactly its own frames — nothing crossed the boundary
        assert_eq!(off.phone_dur[1..].iter().sum::<i64>(), 50);
        // …and the Auto arm demonstrably DOES cross it (guards against a vacuous test: if the two
        // arms ever produced the same arrays, the assertion above would prove nothing).
        let on = build_arrays_daw(&score, &d, ArticulationTiming::Auto).unwrap();
        assert_eq!(on.phone_dur, vec![5, 5, 46, 4], "Auto pre-rolls m out of the rest");
        assert_ne!(on.phone_dur, off.phone_dur);
        for arr in [&on, &off] {
            assert_eq!(arr.phone_dur.iter().sum::<i64>(), 60, "both arms conserve frames");
        }
    }

    // The documented COST of "no borrowing": a note with no spare above its nucleus floor drops the
    // onset entirely, where Auto would have rescued it from the neighbour.
    // mine on a 3-frame note: coda n can't reach 2 frames (budget = min(3, 1, 1) = 1) so it drops
    // and the nucleus holds 3. InNote's floor is max(min(3,2), 3*2/5=1) = 2 ⇒ spare = 1, and m's
    // target (short bucket 3, capped to 2 by the fr≤5 rule) can only be funded to 1 < 2 ⇒ DROPPED.
    // Auto instead borrows 2 frames from the 10-frame rest and keeps m.
    #[test]
    fn in_note_timing_drops_an_onset_it_cannot_fund() {
        let d = en_dicts();
        let score = [en_evt("R", 0, 10), en_evt("mine", 69, 3)];
        let off = build_arrays_daw(&score, &d, ArticulationTiming::InNote).unwrap();
        assert_eq!(off.phon, vec!["SP", "aɪ"], "no room for m INSIDE a 3-frame note");
        assert_eq!(off.phone_dur, vec![10, 3]);
        let on = build_arrays_daw(&score, &d, ArticulationTiming::Auto).unwrap();
        assert_eq!(on.phon, vec!["SP", "m", "aɪ"], "Auto rescues it from the rest");
        assert_eq!(on.phone_dur, vec![8, 2, 3]);
        assert_eq!(off.phone_dur.iter().sum::<i64>(), 13);
        assert_eq!(on.phone_dur.iter().sum::<i64>(), 13);
    }

    // Onset CLUSTER: LAST-first (the consonant touching the vowel wins the scarce frames) and the
    // 2/5 nucleus floor keeps the syllable a syllable.
    // [s t a] on a 10-frame note (mid bucket): allocate_in_note leaves [_, _, 10] (no coda).
    // floor = max(2, 10*2/5) = 4. t first: target 4, spare 6 ⇒ t=4, nucleus 6. then s: target 7 but
    // spare is only 6-4 = 2 ⇒ s=2 (still ≥ the 2-frame OOD minimum), nucleus 4.
    // Without that floor the same input would give s4/t4/a2 — the vowel below its own minimum share.
    #[test]
    fn in_note_timing_cluster_is_last_first_and_keeps_the_vowel_share() {
        let sta = g2p::ScoreEvt {
            lyric: "x", note_num: 60, frames: 10, lang: g2p::Lang::Ja, phoneme_input: Some("s t a"),
            phoneme_set: PhonemeSet::Words,
        };
        let off = build_arrays_daw(&[sta.clone()], &NoDicts, ArticulationTiming::InNote).unwrap();
        assert_eq!(off.phon, vec!["s", "t", "a"]);
        assert_eq!(off.phone_dur, vec![2, 4, 4]);
        assert_eq!(off.phone_dur.iter().sum::<i64>(), 10);
        // Auto with NO lender (score start) is the degraded 2-frame rescue — proof that InNote is
        // NOT simply "the no-lender path" (an early design hypothesis this test exists to refute).
        let on = build_arrays_daw(&[sta], &NoDicts, ArticulationTiming::Auto).unwrap();
        assert_eq!(on.phone_dur, vec![2, 2, 6], "no-lender Auto tops up to the bare minimum only");
    }

    // ★★ THE regression the adversarial review caught (two dimensions, independently).
    //
    // First cut funded the onset LAST, out of whatever the medial/coda pass left on the nucleus. The
    // medial pass is bounded only by the nucleus's 2-frame floor, so on any note carrying a medial —
    // a multi-syllable word, or a multi-mora kana, both ordinary UST practice — the nucleus was
    // already at/below the floor by the time the onset asked, and the WORD-INITIAL consonant was
    // silently deleted while word-INTERNAL ones kept their full measured target.
    // refined = [ɹ ə f aɪ n d] @ 20 frames (400 ms) sang "efined"; ずっと @ 25 frames lost its z.
    // Funding the onset FIRST fixes it. Hand-derived expectation for refined@20:
    //   onset budget = min(fr - SUNG_KEEP_MIN, ceil(fr/2)) = min(18, 10) = 10; ɹ targets 7 ⇒ ɹ = 7.
    //   the rest (13) then runs the shared pass at the note's own bucket (fr=20 ⇒ long):
    //   medial ə (a VOWEL — small share) = clamp(13/6, 2, 4) = 2; medial f = 7; coda budget =
    //   min(4+3, 13-2-9, 13*2/5) = 2 ⇒ d takes 2 LAST-first, n starves and drops; nucleus = 2.
    #[test]
    fn in_note_timing_never_starves_the_word_initial_onset() {
        let refined = g2p::ScoreEvt {
            lyric: "x", note_num: 60, frames: 20, lang: g2p::Lang::Ja,
            phoneme_input: Some("ɹ ə f aɪ n d"),
            phoneme_set: PhonemeSet::Words,
        };
        let score = [g2p::ScoreEvt::ja(&("R", 0, 10)), refined];
        let off = build_arrays_daw(&score, &NoDicts, ArticulationTiming::InNote).unwrap();
        assert_eq!(off.phon, vec!["SP", "ɹ", "ə", "f", "aɪ", "d"], "the WORD-INITIAL ɹ must survive");
        assert_eq!(off.phone_dur, vec![10, 7, 2, 7, 2, 2]);
        assert_eq!(off.phone_dur.iter().sum::<i64>(), 30, "frame-conserving");
        // the rest is left untouched — that is still the point of the switch
        assert_eq!(off.phone_dur[0], 10);
    }

    // Same root cause seen from the other side: a plain CV note in the SHORT bucket. The first cut's
    // invented `fr*2/5` nucleus floor integer-truncates to exactly 2 for every fr ≤ 7, i.e. it was no
    // protection at all precisely where the S84-measured 2-frame-vowel collapse lives. The Auto arm's
    // own structural-half clamp (which shipped and was ear-validated) does the job.
    #[test]
    fn in_note_timing_does_not_crush_the_vowel_on_short_notes() {
        let lead = g2p::ScoreEvt::ja(&("R", 0, 20));
        // か = [k, a] @ 6 frames: onset budget = min(6-2, ceil(6/2)) = 3; k targets 4 ⇒ k=3, a=3.
        let ka = g2p::ScoreEvt::ja(&("か", 60, 6));
        let off = build_arrays_daw(&[lead.clone(), ka], &NoDicts, ArticulationTiming::InNote).unwrap();
        assert_eq!(off.phone_dur, vec![20, 3, 3], "the vowel keeps half the note, not 2 frames");
        // し = [ɕ, i] @ 7: budget = min(5, 4) = 4; ɕ targets 7 (its short-bucket prior) ⇒ ɕ=4, i=3.
        let si = g2p::ScoreEvt::ja(&("し", 60, 7));
        let off2 = build_arrays_daw(&[lead, si], &NoDicts, ArticulationTiming::InNote).unwrap();
        assert_eq!(off2.phone_dur, vec![20, 4, 3], "a 7-frame prior cannot eat the whole note");
    }

    // MONOTONICITY: lengthening a note must never delete one of its phones. The first cut violated
    // this (a longer note gave the medial pass more room, which starved the onset to zero), which is
    // the kind of behaviour no user could ever form a mental model of.
    #[test]
    fn in_note_timing_is_monotone_in_note_length() {
        let mut kept_at: Vec<(i64, usize)> = Vec::new();
        for fr in 2..=60 {
            let w = g2p::ScoreEvt {
                lyric: "x", note_num: 60, frames: fr, lang: g2p::Lang::Ja,
                phoneme_input: Some("ɹ ə f aɪ n d"),
                phoneme_set: PhonemeSet::Words,
            };
            let arr = build_arrays_daw(&[w], &NoDicts, ArticulationTiming::InNote).unwrap();
            assert_eq!(arr.phone_dur.iter().sum::<i64>(), fr, "conservation at fr={fr}");
            kept_at.push((fr, arr.phon.len()));
        }
        for w in kept_at.windows(2) {
            let ((f0, n0), (f1, n1)) = (w[0], w[1]);
            assert!(n1 >= n0, "fr {f0}→{f1} DROPPED a phone ({n0}→{n1}) — non-monotone in length");
        }
        // and the sweep really did cross the interesting region (it must not be all-or-nothing)
        assert!(kept_at.first().unwrap().1 < kept_at.last().unwrap().1, "sweep never gained a phone");
    }

    // ★ SWEEP — the DEFINING invariant, stated so it cannot hold vacuously.
    // With rests only at the START of a score, the M3 rest-borrow (which needs a FOLLOWING rest)
    // can never fire, so under InNote every event's emitted phones must sum to EXACTLY that event's
    // own frames: no frame ever crosses a note boundary. The same scores under Auto must violate it
    // often — otherwise the sweep is not exercising the pre-roll at all and proves nothing.
    #[test]
    fn in_note_timing_never_crosses_a_note_boundary_sweep() {
        let d = en_dicts();
        let mut seed: u64 = 20260729;
        let mut rnd = |n: u64| {
            seed = (seed.wrapping_mul(1103515245).wrapping_add(12345)) & 0x7fff_ffff;
            (seed % n) as i64
        };
        let words = ["mine", "fined"];
        let (mut auto_crossings, mut cases) = (0usize, 0usize);
        for _ in 0..400 {
            let mut score: Vec<g2p::ScoreEvt> = vec![en_evt("R", 0, 1 + rnd(30))];
            for _ in 0..(2 + rnd(5)) {
                let w = words[rnd(2) as usize];
                score.push(en_evt(w, 55 + rnd(20), 1 + rnd(40)));
            }
            let total: i64 = score.iter().map(|e| e.frames).sum();
            for timing in [ArticulationTiming::Auto, ArticulationTiming::InNote] {
                let arr = build_arrays_daw(&score, &d, timing).unwrap();
                assert_eq!(arr.phone_dur.iter().sum::<i64>(), total, "conservation, both arms");
                assert!(arr.phone_dur.iter().all(|&x| x >= 1), "no 0-frame phone is ever emitted");
                // per-event totals
                let mut per_evt = vec![0i64; score.len()];
                for (i, &e) in arr.evt.iter().enumerate() {
                    per_evt[e] += arr.phone_dur[i];
                }
                let crossings =
                    (0..score.len()).filter(|&k| per_evt[k] != score[k].frames).count();
                match timing {
                    ArticulationTiming::InNote => assert_eq!(
                        crossings, 0,
                        "InNote must never move a frame across a note boundary (score {score:?})"
                    ),
                    ArticulationTiming::Auto => auto_crossings += crossings,
                }
            }
            cases += 1;
        }
        assert_eq!(cases, 400);
        // self-check: the sweep really did exercise pre-rolling (a sweep that never triggers the
        // mechanism under test is green by accident — S87 血训).
        assert!(auto_crossings > 200, "sweep never exercised the pre-roll ({auto_crossings})");
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
        let arr = build_arrays_daw(&score, &d, ArticulationTiming::Auto).unwrap();
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
            phoneme_set: PhonemeSet::Words,
        };
        let daw = build_arrays_daw(&[g2p::ScoreEvt::ja(&("あ", 60, 3)), tta], &NoDicts, ArticulationTiming::Auto).unwrap();
        assert_eq!(daw.phon, vec!["a", "a"]);
        assert_eq!(daw.phone_dur, vec![3, 2], "borrowed frame returned to the lender");
    }

    // S83 review #1: medial phones follow the same ≥2-or-DROP policy as codas (never a 1-frame
    // phone); raw-IPA override [p i u] exercises the medial branch without a western dictionary.
    #[test]
    fn medial_vowels_get_two_frames_or_drop() {
        let evt = |fr| g2p::ScoreEvt {
            lyric: "x", note_num: 60, frames: fr, lang: g2p::Lang::Ja, phoneme_input: Some("p i u"),
            phoneme_set: PhonemeSet::Words,
        };
        let a10 = build_arrays_daw(&[evt(10)], &NoDicts, ArticulationTiming::Auto).unwrap();
        assert_eq!(a10.phon, vec!["p", "i", "u"]);
        assert_eq!(a10.phone_dur, vec![2, 3, 5], "medial ≥2; onset supplemented in-note at score start");
        let a3 = build_arrays_daw(&[evt(3)], &NoDicts, ArticulationTiming::Auto).unwrap();
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
            phoneme_set: PhonemeSet::Words,
        };
        let arr = build_arrays_daw(&[g2p::ScoreEvt::ja(&("R", 0, 10)), evt], &NoDicts, ArticulationTiming::Auto).unwrap();
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
        let arr = build_arrays_daw(&[en_evt("mine", 69, 3), en_evt("R", 0, 40)], &d, ArticulationTiming::Auto).unwrap();
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
        g2p::ScoreEvt {
            lyric,
            note_num,
            frames,
            lang: g2p::Lang::Zh,
            phoneme_input: None,
            phoneme_set: PhonemeSet::Words,
        }
    }

    #[test]
    fn zh_same_pitch_sustain_extends_the_final() {
        let d = zh_dicts();
        let a = build_arrays_daw(&[zh_evt("wang", 60, 20), zh_evt("-", 60, 30)], &d, ArticulationTiming::Auto).unwrap();
        assert_eq!(a.phon, vec!["w", "ɑŋ"], "the hold adds NO phone entry (no re-articulation)");
        assert_eq!(a.phone_dur.iter().sum::<i64>(), 50, "total frames (timeline) preserved");
        assert!(*a.phone_dur.last().unwrap() >= 30, "the hold's frames extended the final");
        // chained same-pitch holds keep extending
        let b = build_arrays_daw(
            &[zh_evt("wang", 60, 20), zh_evt("-", 60, 30), zh_evt("-", 60, 10)],
            &d,
            ArticulationTiming::Auto,
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
        let a = build_arrays_daw(&[zh_evt("xiang", 60, 20), zh_evt("-", 62, 30)], &d, ArticulationTiming::Auto).unwrap();
        assert_eq!(a.phon, vec!["ɕ", "iɑŋ", "ɑŋ"]);
        assert_eq!(a.note_pitch, vec![60, 60, 62]);
        // …and a FURTHER same-pitch hold extends that melisma entry instead of re-emitting.
        let b = build_arrays_daw(
            &[zh_evt("xiang", 60, 20), zh_evt("-", 62, 30), zh_evt("-", 62, 10)],
            &d,
            ArticulationTiming::Auto,
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
        // R2IPA = base + EXTRA (additive) + OVERRIDE (replaces, never adds) — so the length is
        // unchanged by the override, and this assert is what stops an override row from silently
        // becoming a NEW romaji key.
        assert_eq!(r2ipa_map().len(), tbl::R2IPA.len() + super::super::g2p_tables::R2IPA_EXTRA.len());
    }

    // ── S86: the deliberate training-alignment divergence must (a) only ever OVERRIDE an existing
    //    upstream row and (b) actually be the value the engine resolves. ──
    #[test]
    fn r2ipa_training_override_replaces_and_takes_effect() {
        let base: HashMap<&str, &[&str]> = tbl::R2IPA.iter().copied().collect();
        for (romaji, phones) in R2IPA_TRAINING_OVERRIDE {
            let upstream = base
                .get(romaji)
                .unwrap_or_else(|| panic!("override {romaji:?} adds a NEW key — it must only replace"));
            assert_ne!(upstream, phones, "override {romaji:?} is identical to upstream — delete it");
            assert_eq!(r2ipa_map()[romaji], *phones, "override {romaji:?} did not win the chain");
        }
        // に sings the well-trained alveolar n (4338 ja frames), NOT the palatal ɲ (92)
        assert_eq!(lyric_phones_for_test("に"), vec!["n", "i"]);
        // …while にゃ/にゅ/にょ keep ɲ — those genuinely are `ny` in the training labels
        assert_eq!(lyric_phones_for_test("にゃ"), vec!["ɲ", "a"]);
        assert_eq!(lyric_phones_for_test("にゅ"), vec!["ɲ", "ɯ"]);
        assert_eq!(lyric_phones_for_test("にょ"), vec!["ɲ", "o"]);
        // and the untouched neighbours of に in the な-row are unaffected
        assert_eq!(lyric_phones_for_test("な"), vec!["n", "a"]);
        assert_eq!(lyric_phones_for_test("ぬ"), vec!["n", "ɯ"]);
    }

    fn lyric_phones_for_test(lyric: &str) -> Vec<&'static str> {
        match classify_lyric(lyric) {
            LyricClass::Phones { phones } => phones,
            other => panic!("{lyric:?} did not resolve to phones: {other:?}"),
        }
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
