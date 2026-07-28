//! Multi-language G2P (S58 dictionary work-line, §3.7): lyric tokens → IPA phones for all 7 languages.
//!
//! Two-stage design (§3.7): **stage1** = word → *traditional* phonemes via the shipped dictionaries
//! (`data/dictionaries/*.tsv`: zh pinyin/opencpop, en CMUdict ARPABET-with-stress, de/fr/es/it the exact
//! MFA dictionaries the model was TRAINED with); **stage2** = traditional → the 210-token IPA vocab,
//! mirrored bit-for-bit from the model repo's `phoneme_vocab.py` + `dict_fixes.py` via the GENERATED
//! `g2p_tables.rs` (IPA is never hand-typed; `g2p_golden_ref.rs` proves the port on golden vectors).
//! JA stays on the existing mora tables in `score2cv.rs`/`score2cv_tables.rs` (Phase-1c parity), extended
//! here only by katakana folding + the generated `KANA_EXTRA` coverage rows.
//!
//! The resolve pass turns a whole score (per-note lyric + effective language + optional traditional-layer
//! `phoneme_input` override) into per-note IPA phones + a per-note **run language** for chunking:
//!  * zh — notes are single hanzi (or pinyin); consecutive hanzi notes form a phrase window and polyphones
//!    are resolved by GREEDY LONGEST MATCH against `zh_phrases.tsv` (render verdict == editor verdict —
//!    context lives in the note sequence, nothing is stamped on the notes). Sustains re-emit the FINAL.
//!  * en/de/fr/es/it — a note carries a whole word; following `+` notes take its next syllables
//!    (SynthV semantics; syllabified by DATA-DRIVEN maximal onset — legal onsets are the word-initial
//!    clusters observed in that language's own dictionary); `-`/`ー` notes hold the current nucleus; the
//!    word-final coda is DEFERRED to the end of the span's last note (归韵 — "light --" sings l-aɪ|aɪ|aɪ-t).
//!  * ja — byte-identical to the legacy path (`+`≡hold, carrier vowel via VOWEL_SET, geminates, っ…).
//! Rest/breath notes are language-neutral and attach to the PREVIOUS run (a language cut then lands in
//! silence); sustains inherit the carrier's language. The universal reserved tokens (`R`/`r`/empty =
//! rest, `AP` = breath, `-`/`ー` = hold, `+` = next) stay reserved in EVERY language. S86 narrowed
//! that set: `rest`/`sil`/`pau` are NO LONGER reserved here (they are real dictionary words and were
//! silently swallowing sung notes) — `score2cv::lyric_to_phones` still accepts them because it is the
//! upstream parity port, and the two are deliberately out of step. `g2p::is_silent_token` is the one
//! predicate every DAW-side consumer must use.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::OnceLock;

use super::g2p_tables as gt;
use super::score2cv::{classify_lyric as classify_lyric_ja, LyricClass};
use super::score2cv_tables as tbl;
use crate::{Result, UtaiError};

// ─── languages ───────────────────────────────────────────────────────────────────────────────────

/// ScoreToCV language conditioning (LANG_TO_ID; ko/ru exist in the embedding but have no training data).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Lang {
    Zh,
    En,
    Ja,
    De,
    Fr,
    Es,
    It,
}

impl Lang {
    pub fn from_id(id: i64) -> Option<Lang> {
        match id {
            0 => Some(Lang::Zh),
            1 => Some(Lang::En),
            2 => Some(Lang::Ja),
            3 => Some(Lang::De),
            4 => Some(Lang::Fr),
            5 => Some(Lang::Es),
            6 => Some(Lang::It),
            _ => None,
        }
    }
    pub fn id(self) -> i64 {
        match self {
            Lang::Zh => 0,
            Lang::En => 1,
            Lang::Ja => 2,
            Lang::De => 3,
            Lang::Fr => 4,
            Lang::Es => 5,
            Lang::It => 6,
        }
    }
    pub fn code(self) -> &'static str {
        match self {
            Lang::Zh => "zh",
            Lang::En => "en",
            Lang::Ja => "ja",
            Lang::De => "de",
            Lang::Fr => "fr",
            Lang::Es => "es",
            Lang::It => "it",
        }
    }
}

/// One score note as the render/validate front-end sees it (the wire `ScoreNote` resolved).
#[derive(Debug, Clone)]
pub struct ScoreEvt<'a> {
    pub lyric: &'a str,
    pub note_num: i64,
    pub frames: i64,
    /// Effective language (per-note override ?? track default — resolved by the frontend).
    pub lang: Lang,
    /// Traditional-phoneme override (§3.7 user layer): with whitespace = raw traditional phones;
    /// without = a syllable/mora (zh pinyin, ja kana/romaji; a single bare phone for en/de/fr/es/it).
    pub phoneme_input: Option<&'a str>,
}

impl<'a> ScoreEvt<'a> {
    /// JA-defaulted event from a legacy `(lyric, note_num, frames)` triple (parity paths + tests).
    pub fn ja(t: &(&'a str, i64, i64)) -> ScoreEvt<'a> {
        ScoreEvt { lyric: t.0, note_num: t.1, frames: t.2, lang: Lang::Ja, phoneme_input: None }
    }
}

// ─── vocab interning (IPA string → the 'static vocab key; membership check in one step) ──────────

fn vocab_intern_map() -> &'static HashMap<&'static str, &'static str> {
    static M: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
    M.get_or_init(|| tbl::PHONE_TO_ID.iter().map(|&(k, _)| (k, k)).collect())
}

fn intern(ipa: &str) -> Option<&'static str> {
    vocab_intern_map().get(ipa).copied()
}

// ─── stage2: traditional phones → vocab IPA (port of phoneme_vocab.convert_* + dict_fixes) ──────

fn map_of(pairs: &'static [(&'static str, &'static str)]) -> HashMap<&'static str, &'static str> {
    pairs.iter().copied().collect()
}
fn zh_initials() -> &'static HashMap<&'static str, &'static str> {
    static M: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
    M.get_or_init(|| map_of(gt::OPENCPOP_INITIALS_IPA))
}
fn zh_finals() -> &'static HashMap<&'static str, &'static str> {
    static M: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
    M.get_or_init(|| map_of(gt::OPENCPOP_FINALS_IPA))
}
fn arpabet_map() -> &'static HashMap<&'static str, &'static str> {
    static M: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
    M.get_or_init(|| map_of(gt::ARPABET_IPA))
}
fn mfa_normalize_map() -> &'static HashMap<&'static str, &'static str> {
    static M: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
    M.get_or_init(|| map_of(gt::MFA_NORMALIZE))
}
fn c2_map() -> &'static HashMap<&'static str, &'static str> {
    static M: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
    M.get_or_init(|| map_of(gt::FIX_C2))
}
fn c3_map() -> &'static HashMap<&'static str, &'static str> {
    static M: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
    M.get_or_init(|| map_of(gt::FIX_C3_GLOBAL))
}

/// Port of `convert_opencpop` (initials first, then finals, else passthrough; SP/AP verbatim).
fn convert_opencpop(p: &str) -> String {
    if p == "SP" || p == "AP" {
        return p.to_string();
    }
    if let Some(&ipa) = zh_initials().get(p) {
        return ipa.to_string();
    }
    if let Some(&ipa) = zh_finals().get(p) {
        return ipa.to_string();
    }
    p.to_string()
}

/// Port of `convert_arpabet`: sil/sp/spn → SP; `AH*` resolved BEFORE stress-strip (AH0=ə else ʌ —
/// the A2 fix: stress encodes a phonemic split); else strip a trailing 0/1/2 and map.
fn convert_arpabet(p: &str) -> String {
    let lower = p.to_ascii_lowercase();
    if matches!(lower.as_str(), "sil" | "sp" | "spn") {
        return "SP".to_string();
    }
    let up = p.to_ascii_uppercase();
    if let Some(stress) = up.strip_prefix("AH") {
        return if stress == "0" { "ə".to_string() } else { "ʌ".to_string() };
    }
    let base = up.strip_suffix(['0', '1', '2']).unwrap_or(&up);
    if let Some(&ipa) = arpabet_map().get(base) {
        return ipa.to_string();
    }
    p.to_string()
}

/// Port of `convert_mfa`: empty/spn → SP; NFC-normalize; MFA_NORMALIZE map; else passthrough.
/// (The canonical dictionaries are NFC already — build_dictionaries normalizes — so no NFC pass here;
/// phoneme_input overrides are NFC-normalized by the frontend sanitizer before they reach Rust.)
fn convert_mfa(p: &str) -> String {
    if p.is_empty() || p == "spn" {
        return "SP".to_string();
    }
    if let Some(&ipa) = mfa_normalize_map().get(p) {
        return ipa.to_string();
    }
    p.to_string()
}

/// Port of `apply_dict_fixes` for the 7 shipped languages (A1 zh apical-i by ORIGINAL prev phone;
/// C2 non-ja palatal-stop de-narrow; C3 global dead tokens; ja handled upstream by its own tables).
fn apply_fixes(ipa: Vec<String>, lang: Lang) -> Vec<String> {
    let n = ipa.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let ph = ipa[i].as_str();
        if lang == Lang::Zh && ph == "i" {
            let prev = if i > 0 { Some(ipa[i - 1].as_str()) } else { None };
            if prev.is_some_and(|p| gt::FIX_A1_DENTAL.contains(&p)) {
                out.push("ɹ̩".to_string());
                continue;
            }
            if prev.is_some_and(|p| gt::FIX_A1_RETRO.contains(&p)) {
                out.push("ɻ̩".to_string());
                continue;
            }
            out.push("i".to_string());
            continue;
        }
        if lang != Lang::Ja {
            if let Some(&r) = c2_map().get(ph) {
                out.push(r.to_string());
                continue;
            }
        }
        if let Some(&r) = c3_map().get(ph) {
            out.push(r.to_string());
            continue;
        }
        out.push(ipa[i].clone());
    }
    out
}

/// stage2: traditional phones → interned vocab IPA. Err = the offending phone (caller wraps the CODE).
pub fn stage2(lang: Lang, phones: &[String]) -> std::result::Result<Vec<&'static str>, String> {
    let converted: Vec<String> = phones
        .iter()
        .map(|p| match lang {
            Lang::Zh => convert_opencpop(p),
            Lang::En => convert_arpabet(p),
            Lang::Ja => p.clone(), // ja never routes here (legacy tables); identity for safety
            _ => convert_mfa(p),
        })
        .collect();
    let fixed = apply_fixes(converted, lang);
    fixed
        .iter()
        .map(|p| intern(p).ok_or_else(|| p.clone()))
        .collect()
}

// ─── dictionaries (lazy, per-language, from data/dictionaries) ───────────────────────────────────

/// Word dictionary for en/de/fr/es/it: word → primary traditional phones, + the language's observed
/// word-initial consonant clusters (LEGAL ONSETS for maximal-onset syllabification) + its vowel test.
pub struct WordDict {
    lang: Lang,
    map: HashMap<String, String>,
    onsets: HashSet<String>,
    vowels: HashSet<&'static str>,
}

impl WordDict {
    /// A traditional-layer phone is a syllable NUCLEUS: en = ARPABET vowels carry a stress digit;
    /// MFA languages use the generated per-dictionary vowel inventory.
    pub fn is_vowel(&self, ph: &str) -> bool {
        match self.lang {
            Lang::En => ph.ends_with(['0', '1', '2']),
            _ => self.vowels.contains(ph),
        }
    }

    pub fn lookup(&self, word: &str) -> Option<Vec<String>> {
        lookup_candidates(word)
            .iter()
            .find_map(|k| self.map.get(k))
            .map(|p| p.split_whitespace().map(str::to_string).collect())
    }

    /// Parse the canonical `word<TAB>phones` TSV. First-seen pronunciation wins (the build emits the
    /// primary first); every word's initial consonant cluster feeds the legal-onset set.
    pub fn from_tsv(lang: Lang, tsv: &str) -> WordDict {
        let vowels: HashSet<&'static str> = match lang {
            Lang::De => gt::MFA_VOWELS_DE.iter().copied().collect(),
            Lang::Fr => gt::MFA_VOWELS_FR.iter().copied().collect(),
            Lang::Es => gt::MFA_VOWELS_ES.iter().copied().collect(),
            Lang::It => gt::MFA_VOWELS_IT.iter().copied().collect(),
            _ => HashSet::new(),
        };
        let mut dict = WordDict { lang, map: HashMap::new(), onsets: HashSet::new(), vowels };
        dict.onsets.insert(String::new()); // the empty onset is always legal (V-initial syllables)
        for line in tsv.lines() {
            let Some((word, phones)) = line.split_once('\t') else { continue };
            let phones = phones.trim();
            if word.is_empty() || phones.is_empty() {
                continue;
            }
            let key = word.to_lowercase();
            dict.map.entry(key).or_insert_with(|| phones.to_string());
            // word-initial cluster = phones before the first vowel (words with no vowel don't vote)
            let toks: Vec<&str> = phones.split_whitespace().collect();
            if let Some(vi) = toks.iter().position(|t| dict_is_vowel(lang, &dict.vowels, t)) {
                dict.onsets.insert(toks[..vi].join(" "));
            }
        }
        dict
    }
}

/// Candidate dictionary keys for one raw lyric, MOST FAITHFUL FIRST (S86 input-tolerance ladder).
///
/// The score stores lyrics exactly as typed — `sanitizeText` only NFC-normalizes and strips control
/// chars — so the tolerance lives HERE, at lookup, and never rewrites what the user wrote:
///  1. lowercase only — the spelling the user typed always gets first refusal.
///  2. + typographic apostrophes folded to ASCII `'` (phone keyboards and lyric sites emit U+2019).
///     ⚠ this is a RUNG, not a rewrite of the base: it.tsv ships **7 keys spelled with U+2018**, and
///     folding the base would make them unreachable (that mistake shipped in the first S86 draft).
///  3. + surrounding punctuation trimmed (`Love,` `(oh)` — pasted lyric sheets carry it). `'` and `-`
///     are word-INTERNAL in these dictionaries (`'bout`, `l'`, 8164 en keys) and are never trimmed.
///  4. + ß→ss LAST: upstream `german_mfa` uses Swiss orthography and ships ZERO ß spellings, so
///     `weiß`/`groß`/`Straße`/`heißt` would otherwise every one be OOV — and OOV aborts the render.
///     ⚠ Swiss orthography merges ß/ss, so this rung can land on a homograph (`Maße`→*Masse*). It is
///     the last rung precisely so it only ever fires where the faithful spelling found nothing.
///
/// Faithful-first ordering means a word that genuinely needs its raw form still wins; a candidate is
/// only consulted when every more-faithful one missed.
fn lookup_candidates(raw: &str) -> Vec<String> {
    let fold_quotes = |s: &str| -> String {
        s.chars().map(|c| if matches!(c, '\u{2019}' | '\u{2018}' | '\u{02BC}' | '\u{FF07}') { '\'' } else { c }).collect()
    };
    let trim_punct =
        |s: &str| -> String { s.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'' && c != '-').to_string() };

    let base: String = raw.trim().chars().flat_map(char::to_lowercase).collect();
    let mut out: Vec<String> = Vec::with_capacity(8);
    // Every transform is its OWN rung, applied to a COPY — never folded into the base. it.tsv really
    // does ship 7 keys spelled with U+2018, so rewriting the base would make them unreachable.
    for quoted in [base.clone(), fold_quotes(&base)] {
        for trimmed in [quoted.clone(), trim_punct(&quoted)] {
            // ß→ss last: the spelling the user actually wrote always gets first refusal
            for key in [trimmed.clone(), trimmed.replace('ß', "ss")] {
                if !key.is_empty() && !out.contains(&key) {
                    out.push(key);
                }
            }
        }
    }
    out
}

fn dict_is_vowel(lang: Lang, vowels: &HashSet<&'static str>, ph: &str) -> bool {
    match lang {
        Lang::En => ph.ends_with(['0', '1', '2']),
        _ => vowels.contains(ph),
    }
}

/// zh dictionary: pinyin syllable → opencpop phones (M4Singer convention), hanzi → readings
/// (primary first), phrase → per-char readings (polyphone context).
pub struct ZhDict {
    syllables: HashMap<String, String>,
    chars: HashMap<char, Vec<String>>,
    phrases: HashMap<String, Vec<String>>,
    max_phrase: usize,
}

impl ZhDict {
    pub fn from_tsv(syllables: &str, chars: &str, phrases: &str) -> ZhDict {
        let mut d = ZhDict {
            syllables: HashMap::new(),
            chars: HashMap::new(),
            phrases: HashMap::new(),
            max_phrase: 0,
        };
        for line in syllables.lines() {
            if let Some((s, ph)) = line.split_once('\t') {
                // empty-phones guard (mirrors WordDict): a "syl\t" row would resolve to ZERO phones and
                // silently desync the cv↔DAW group alignment (audit) — skipping it makes the syllable a
                // LOUD OOV instead.
                if !s.is_empty() && !ph.trim().is_empty() {
                    d.syllables.insert(s.to_string(), ph.trim().to_string());
                }
            }
        }
        for line in chars.lines() {
            if let Some((c, readings)) = line.split_once('\t') {
                let mut it = c.chars();
                if let (Some(ch), None) = (it.next(), it.next()) {
                    d.chars.insert(ch, readings.trim().split(',').map(str::to_string).collect());
                }
            }
        }
        for line in phrases.lines() {
            if let Some((p, syls)) = line.split_once('\t') {
                let n = p.chars().count();
                let parsed: Vec<String> = syls.trim().split_whitespace().map(str::to_string).collect();
                // a phrase row must carry exactly one syllable per char, else the greedy assign would
                // mislabel neighbouring notes — drop malformed rows (chars fall back to defaults).
                if n >= 2 && parsed.len() == n {
                    d.max_phrase = d.max_phrase.max(n);
                    d.phrases.insert(p.to_string(), parsed);
                }
            }
        }
        d
    }

    pub fn syllable_phones(&self, syl: &str) -> Option<Vec<String>> {
        self.syllables.get(syl).map(|p| p.split_whitespace().map(str::to_string).collect())
    }
    pub fn is_hanzi(&self, c: char) -> bool {
        self.chars.contains_key(&c)
    }
    pub fn char_default(&self, c: char) -> Option<&str> {
        self.chars.get(&c).and_then(|r| r.first()).map(String::as_str)
    }
    pub fn char_readings(&self, c: char) -> Option<&[String]> {
        self.chars.get(&c).map(Vec::as_slice)
    }
}

/// Dictionary provider — the resolve pass asks for a language's dictionary only when the score uses it
/// (a pure-JA score never touches disk). The global impl lazy-loads from `data/dictionaries`; tests
/// inject in-memory fixtures.
pub trait DictSource {
    fn zh(&self) -> Result<&ZhDict>;
    fn words(&self, lang: Lang) -> Result<&WordDict>;
}

static DICT_DIR: OnceLock<PathBuf> = OnceLock::new();

/// Set the dictionaries directory (`<data>/dictionaries`) — called by the command layer; first call wins
/// (the data dir is fixed for the process lifetime).
pub fn set_dict_dir(dir: PathBuf) {
    let _ = DICT_DIR.set(dir);
}

fn read_dict_file(name: &str) -> Result<String> {
    let dir = DICT_DIR
        .get()
        .ok_or_else(|| UtaiError::Inference("VOCAL_DICT_MISSING: dictionaries dir not set".into()))?;
    std::fs::read_to_string(dir.join(name))
        .map_err(|_| UtaiError::Inference(format!("VOCAL_DICT_MISSING: {}", name)))
}

/// The process-wide lazy dictionary store. A SUCCESSFUL load is cached for the process lifetime
/// (Box::leak — bounded: ≤6 dictionaries); a FAILED load is NOT cached, so restoring a missing TSV
/// recovers on the next render/validation without an app restart (audit: the old OnceLock cached the
/// failure forever).
pub struct GlobalDicts;

static ZH_DICT: parking_lot::Mutex<Option<&'static ZhDict>> = parking_lot::Mutex::new(None);
static WORD_DICTS: parking_lot::Mutex<[Option<&'static WordDict>; 5]> = parking_lot::Mutex::new([None; 5]);

fn word_slot(lang: Lang) -> usize {
    match lang {
        Lang::En => 0,
        Lang::De => 1,
        Lang::Fr => 2,
        Lang::Es => 3,
        Lang::It => 4,
        _ => unreachable!("word_slot: {lang:?} is not a word-dictionary language"),
    }
}

impl DictSource for GlobalDicts {
    fn zh(&self) -> Result<&ZhDict> {
        let mut cell = ZH_DICT.lock();
        if let Some(d) = *cell {
            return Ok(d);
        }
        let d = ZhDict::from_tsv(
            &read_dict_file("zh_syllables.tsv")?,
            &read_dict_file("zh_chars.tsv")?,
            &read_dict_file("zh_phrases.tsv")?,
        );
        let leaked: &'static ZhDict = Box::leak(Box::new(d));
        *cell = Some(leaked);
        Ok(leaked)
    }

    fn words(&self, lang: Lang) -> Result<&WordDict> {
        let slot = word_slot(lang);
        let mut cells = WORD_DICTS.lock();
        if let Some(d) = cells[slot] {
            return Ok(d);
        }
        let tsv = read_dict_file(&format!("{}.tsv", lang.code()))?;
        let leaked: &'static WordDict = Box::leak(Box::new(WordDict::from_tsv(lang, &tsv)));
        cells[slot] = Some(leaked);
        Ok(leaked)
    }
}

// ─── syllabification (data-driven maximal onset) ─────────────────────────────────────────────────

/// Split a word's traditional phones into syllables by MAXIMAL ONSET: between two nuclei, the largest
/// suffix of the consonant cluster that is an OBSERVED word-initial cluster of this language starts the
/// next syllable; the rest closes the previous one. A word with no nucleus is one "syllable".
pub fn syllabify(dict: &WordDict, phones: &[String]) -> Vec<Vec<String>> {
    let nuclei: Vec<usize> = (0..phones.len()).filter(|&i| dict.is_vowel(&phones[i])).collect();
    if nuclei.is_empty() {
        return vec![phones.to_vec()];
    }
    let mut bounds = vec![0usize]; // syllable start indices
    for w in nuclei.windows(2) {
        let (a, b) = (w[0], w[1]);
        let cluster = &phones[a + 1..b];
        // longest legal onset = the SMALLEST cut whose suffix is an observed word-initial cluster
        // (the empty suffix is always legal, so `cut` always resolves).
        let mut cut = cluster.len();
        for s in 0..=cluster.len() {
            if dict.onsets.contains(&cluster[s..].join(" ")) {
                cut = s;
                break;
            }
        }
        bounds.push(a + 1 + cut);
    }
    bounds.push(phones.len());
    bounds.windows(2).map(|w| phones[w[0]..w[1]].to_vec()).collect()
}

/// A syllable's nucleus index (first vowel; falls back to the last phone for vowel-less words).
fn nucleus_idx(dict: &WordDict, syl: &[String]) -> usize {
    syl.iter().position(|p| dict.is_vowel(p)).unwrap_or(syl.len().saturating_sub(1))
}

// ─── the resolve pass ────────────────────────────────────────────────────────────────────────────

/// Per-note resolution outcome (the render consumes `Phones`; the editor consumes the class verbatim).
#[derive(Debug, Clone)]
pub enum ResolvedKind {
    Rest,
    Breath,
    /// Sung phones (words, sustains resolved to their carrier nucleus, zh finals, ja morae…).
    Phones(Vec<&'static str>),
    /// OOV (lenient mode only — strict render errors instead).
    Unknown,
}

#[derive(Debug, Clone)]
pub struct ResolvedNote {
    pub kind: ResolvedKind,
    /// The chunk-run language (sustains inherit the carrier; rests attach to the previous run).
    pub run_lang: Lang,
    /// True when the note was a sustain/next token (editor classification).
    pub is_sustain: bool,
}

/// Universal reserved-token classes (identical in every language — they are checked BEFORE any
/// language dispatch, so e.g. the English word "rest" is a reserved rest token by design).
#[derive(Clone, Copy, PartialEq)]
enum Tok {
    Rest,
    Breath,
    Hold,
    Next,
    Word,
}

/// ⚠ DELIBERATELY NARROWER than `score2cv::lyric_to_phones`' rest set — do NOT "unify" them. That
/// one is a bit-parity port of upstream `render_ust.lyric_to_phones`; THIS one is the DAW's
/// user-facing convention, which is ours to choose.
///
/// S86 frees `rest`, `sil` AND `pau`. All three are real lexical material, not just jargon:
/// `rest` is common English lyric vocabulary ("give it a rest"), and once word-splitting across
/// notes exists `sil` and `pau` appear as FRAGMENTS of ordinary words (sil|ver, pau|se) — a reserved
/// token that can be a piece of a word will collide by construction. They were also swallowing notes
/// silently, and only in lowercase, since this check is case-sensitive while the dictionary lookup is
/// not (so `Rest` sang and `rest` did not).
///
/// What remains hard-wired is only the UTAU/OpenUtau convention `R`/`r`/empty — the sole rest token
/// this app ever WRITES (vocalRender.ts, rangeTest.ts, export_score.rs, import.rs). Users who want a
/// different trigger get it the way breath already works: a per-track token the frontend maps onto
/// the canonical one before Rust ever sees it (`VocalTrackParams.breathToken` → `AP`; `restToken` →
/// `R`), so a convenient glyph is never stolen from real lyrics.
/// Anything changed here must also hold for `is_silent_token`'s consumers.
fn token_class(lyric: &str) -> Tok {
    match lyric.trim() {
        "R" | "r" | "" => Tok::Rest,
        "AP" | "ap" => Tok::Breath,
        "-" | "ー" => Tok::Hold,
        "+" => Tok::Next,
        _ => Tok::Word,
    }
}

/// Does this lyric produce NO sung phones (rest or breath)? THE single source for that question.
///
/// `score2svc::compute_note_groups` builds the DAW-side note grouping from this, so it can never key
/// differently from the cv-side grouping `assemble_arrays` builds out of `resolve_core`. They MUST
/// agree: a one-note disagreement shifts every later group index, and `build_vol_env` (no
/// frame-conservation fast path, live on the vol_embedding models) then applies each note's dynamics
/// to the wrong note for the rest of the segment. Before S86 the DAW side called
/// `score2cv::classify_lyric`, whose rest set is deliberately WIDER (it is the upstream parity port)
/// — narrowing `token_class` without moving this predicate would have silently reopened that desync.
pub fn is_silent_token(lyric: &str) -> bool {
    matches!(token_class(lyric), Tok::Rest | Tok::Breath)
}

/// Per-note run language for chunking — shared by `build_arrays`' assembly AND `compute_note_groups`
/// (score2svc) so grouping can never drift between the cv side and the DAW side. Sustains inherit the
/// previous note's run; rests/breaths attach to the previous run; leading rests take the first run.
pub fn note_run_langs(score: &[ScoreEvt]) -> Vec<Lang> {
    let n = score.len();
    let mut out: Vec<Option<Lang>> = vec![None; n];
    let mut cur: Option<Lang> = None;
    for i in 0..n {
        match token_class(score[i].lyric) {
            Tok::Word => {
                cur = Some(score[i].lang);
                out[i] = cur;
            }
            Tok::Hold | Tok::Next => {
                out[i] = Some(cur.unwrap_or(score[i].lang));
                if cur.is_none() {
                    cur = out[i];
                }
            }
            Tok::Rest | Tok::Breath => out[i] = cur, // None for leading rests → backfilled below
        }
    }
    // leading rests (and an all-rest score) take the first resolved run / ja default
    let first = out.iter().flatten().next().copied().unwrap_or(Lang::Ja);
    let mut fill = first;
    for slot in out.iter_mut() {
        match slot {
            Some(l) => fill = *l,
            None => *slot = Some(fill),
        }
    }
    out.into_iter().map(|l| l.unwrap_or(Lang::Ja)).collect()
}

/// Resolve a whole score to per-note phones (strict mode: LOUD `VOCAL_OOV` on the first unresolvable
/// note; lenient mode: per-note `Unknown` for the editor's marking pass).
fn resolve_core(score: &[ScoreEvt], dicts: &dyn DictSource, strict: bool) -> Result<Vec<ResolvedNote>> {
    let n = score.len();
    let run_langs = note_run_langs(score);
    let toks: Vec<Tok> = score.iter().map(|e| token_class(e.lyric)).collect();

    // zh phrase pass: phrase context flows over a window of PLAIN single-hanzi word notes (no override,
    // no pinyin) where sustains/rests/breaths are TRANSPARENT — a hold belongs to the previous char and
    // a breath doesn't end a word (audit: [了][-][解] must still read 了解=liǎo, not the 了 default).
    // Any other note (pinyin, override, another language's word) breaks the window. Greedy longest
    // phrase match assigns each participating note its resolved pinyin syllable.
    // Dictionary availability is checked ONCE per zh word note (dicts.zh()? propagates a missing
    // dictionary as VOCAL_DICT_MISSING — never masked as per-word OOV; audit MAJOR).
    let mut zh_syl: Vec<Option<String>> = vec![None; n];
    let has_zh_word = (0..n).any(|k| toks[k] == Tok::Word && score[k].lang == Lang::Zh);
    if has_zh_word {
        let zh = dicts.zh()?;
        let is_plain_hanzi = |k: usize| -> bool {
            toks[k] == Tok::Word
                && score[k].lang == Lang::Zh
                && score[k].phoneme_input.is_none()
                && {
                    let mut cs = score[k].lyric.trim().chars();
                    matches!((cs.next(), cs.next()), (Some(c), None) if zh.is_hanzi(c))
                }
        };
        let transparent = |k: usize| matches!(toks[k], Tok::Hold | Tok::Next | Tok::Rest | Tok::Breath);
        let mut i = 0;
        while i < n {
            if !is_plain_hanzi(i) {
                i += 1;
                continue;
            }
            // collect the hanzi-note indices of this window (transparent notes skipped, not breaking)
            let mut idx: Vec<usize> = Vec::new();
            let mut j = i;
            while j < n {
                if is_plain_hanzi(j) {
                    idx.push(j);
                    j += 1;
                } else if transparent(j) {
                    j += 1;
                } else {
                    break;
                }
            }
            let chars: Vec<char> = idx.iter().map(|&k| score[k].lyric.trim().chars().next().unwrap()).collect();
            let mut pos = 0usize;
            while pos < chars.len() {
                let maxw = zh.max_phrase.min(chars.len() - pos);
                let mut matched = 0usize;
                for w in (2..=maxw).rev() {
                    let phrase: String = chars[pos..pos + w].iter().collect();
                    if let Some(syls) = zh.phrases.get(&phrase) {
                        for (k, s) in syls.iter().enumerate() {
                            zh_syl[idx[pos + k]] = Some(s.clone());
                        }
                        matched = w;
                        break;
                    }
                }
                if matched == 0 {
                    zh_syl[idx[pos]] = zh.char_default(chars[pos]).map(str::to_string);
                    matched = 1;
                }
                pos += matched;
            }
            i = j;
        }
    }

    // main pass: per-note phones + carrier state for sustains; western spans handled look-ahead.
    let mut out: Vec<Option<ResolvedNote>> = vec![None; n];
    let oov = |lyr: &str| UtaiError::Inference(format!("VOCAL_OOV: {}", lyr));
    // carrier nucleus for holds outside western spans (ja legacy prev_vowel / zh final).
    let mut carrier: Option<&'static str> = None;

    let mut i = 0;
    while i < n {
        let evt = &score[i];
        let run_lang = run_langs[i];
        match toks[i] {
            Tok::Rest => {
                carrier = None;
                out[i] = Some(ResolvedNote { kind: ResolvedKind::Rest, run_lang, is_sustain: false });
                i += 1;
            }
            Tok::Breath => {
                carrier = None;
                out[i] = Some(ResolvedNote { kind: ResolvedKind::Breath, run_lang, is_sustain: false });
                i += 1;
            }
            Tok::Hold | Tok::Next => {
                // an orphan sustain (span-attached ones were consumed below): legacy ja semantics —
                // re-emit the carrier nucleus, default "a".
                let ph = vec![carrier.unwrap_or("a")];
                out[i] = Some(ResolvedNote { kind: ResolvedKind::Phones(ph), run_lang, is_sustain: true });
                i += 1;
            }
            Tok::Word => {
                match evt.lang {
                    Lang::Ja | Lang::Zh => {
                        match resolve_east_word(evt, zh_syl[i].as_deref(), dicts)? {
                            Some(ph) => {
                                // carrier update: ja = last phone if in VOWEL_SET (legacy prev_vowel
                                // rule — persists across a non-vowel-final note like ん); zh = the
                                // final (always the last phone of [initial?, final]).
                                match evt.lang {
                                    Lang::Ja => {
                                        if let Some(&last) = ph.last() {
                                            if tbl::VOWEL_SET.contains(&last) {
                                                carrier = Some(last);
                                            }
                                        }
                                    }
                                    _ => carrier = ph.last().copied(),
                                }
                                out[i] =
                                    Some(ResolvedNote { kind: ResolvedKind::Phones(ph), run_lang, is_sustain: false });
                            }
                            None => {
                                if strict {
                                    return Err(oov(evt.lyric));
                                }
                                out[i] = Some(ResolvedNote { kind: ResolvedKind::Unknown, run_lang, is_sustain: false });
                            }
                        }
                        i += 1;
                    }
                    _ => {
                        // western span: this word + following hold/next notes (any language change in a
                        // sustain is ignored — sustains inherit the carrier by construction).
                        let mut span_end = i + 1;
                        while span_end < n && matches!(toks[span_end], Tok::Hold | Tok::Next) {
                            span_end += 1;
                        }
                        match resolve_west_span(evt, &score[i..span_end], &toks[i..span_end], dicts)? {
                            Some(assignments) => {
                                for (j, ph) in assignments.into_iter().enumerate() {
                                    carrier = ph.last().copied().or(carrier);
                                    out[i + j] = Some(ResolvedNote {
                                        kind: ResolvedKind::Phones(ph),
                                        run_lang: run_langs[i + j],
                                        is_sustain: j > 0,
                                    });
                                }
                            }
                            None => {
                                if strict {
                                    return Err(oov(evt.lyric));
                                }
                                out[i] = Some(ResolvedNote { kind: ResolvedKind::Unknown, run_lang, is_sustain: false });
                                for j in i + 1..span_end {
                                    // the sustains still resolve (hold "a") so ONLY the word marks OOV
                                    out[j] = Some(ResolvedNote {
                                        kind: ResolvedKind::Phones(vec!["a"]),
                                        run_lang: run_langs[j],
                                        is_sustain: true,
                                    });
                                }
                            }
                        }
                        carrier = None; // western carrier state is span-internal; a NEW word resets it
                        i = span_end;
                    }
                }
            }
        }
    }

    Ok(out.into_iter().map(|o| o.expect("every note resolved")).collect())
}

/// Resolve a JA/ZH sung word note to IPA phones. `Ok(None)` = real OOV (unknown word/mora/pinyin);
/// `Err` = INFRASTRUCTURE failure (missing dictionary — propagated as VOCAL_DICT_MISSING, never
/// masked as OOV; audit MAJOR). §3.7 override precedence: whitespace phoneme_input = raw traditional
/// phones; no-space = a mora (ja) / pinyin syllable (zh); otherwise ja = legacy mora path, zh =
/// phrase-resolved reading (or the lyric as bare pinyin).
fn resolve_east_word(
    evt: &ScoreEvt,
    zh_phrase_syl: Option<&str>,
    dicts: &dyn DictSource,
) -> Result<Option<Vec<&'static str>>> {
    if let Some(pi) = evt.phoneme_input {
        let pi = pi.trim();
        if pi.contains(char::is_whitespace) {
            let phones: Vec<String> = pi.split_whitespace().map(str::to_string).collect();
            return Ok(match evt.lang {
                Lang::Ja => ja_phones_from_tokens(&phones),
                _ => stage2(evt.lang, &phones).ok(),
            });
        }
    }
    match evt.lang {
        Lang::Ja => {
            let token = evt.phoneme_input.map(str::trim).unwrap_or(evt.lyric);
            Ok(ja_word_phones(token))
        }
        _ => {
            let zh = dicts.zh()?;
            let syl: String = match (evt.phoneme_input, zh_phrase_syl) {
                (Some(pi), _) => pi.trim().to_lowercase(),
                (None, Some(s)) => s.to_string(),
                // not a plain hanzi: try the lyric itself as a bare pinyin syllable
                (None, None) => evt.lyric.trim().to_lowercase(),
            };
            let Some(trad) = zh.syllable_phones(&syl) else { return Ok(None) };
            Ok(stage2(Lang::Zh, &trad).ok())
        }
    }
}

/// JA word → IPA phones via the legacy mora path (`score2cv::lyric_to_phones` incl. geminates/っ), with
/// katakana folded to hiragana first (S58 coverage fix — katakana lyrics used to OOV).
/// S86: the same faithful-first tolerance ladder the word dictionaries use (`lookup_candidates`) —
/// ONE source, so ja can never drift from en/de/fr/es/it on what counts as "the same lyric". Required
/// in the same round as `kana_tokenize`: `か、` used to "work" only because the old truncating fallback
/// silently ate the 、, and the tokenizer (correctly) refuses to consume it.
fn ja_word_phones(token: &str) -> Option<Vec<&'static str>> {
    lookup_candidates(token).iter().find_map(|cand| match classify_lyric_ja(&fold_katakana(cand)) {
        LyricClass::Phones { phones } => Some(phones),
        _ => None,
    })
}

/// JA raw-phoneme override: each token must be a vocab IPA phone already (advanced escape hatch).
fn ja_phones_from_tokens(phones: &[String]) -> Option<Vec<&'static str>> {
    phones.iter().map(|p| intern(p)).collect()
}

/// Fold katakana (ァ..ヶ + ヴ) to hiragana by the standard −0x60 codepoint shift; everything else
/// (incl. the ー sustain mark, handled upstream) passes through.
pub fn fold_katakana(s: &str) -> String {
    s.chars()
        .map(|c| {
            let cp = c as u32;
            if (0x30A1..=0x30F6).contains(&cp) {
                char::from_u32(cp - 0x60).unwrap_or(c)
            } else {
                c
            }
        })
        .collect()
}

/// Resolve one western-word span: syllabify the word, distribute syllables over the carrier + `+`
/// notes (last consumer takes the remainder), holds re-emit the current nucleus, and DEFER the
/// word-final coda to the span's last note (归韵). Returns per-span-note IPA phone lists;
/// `Ok(None)` = real OOV, `Err` = missing dictionary (VOCAL_DICT_MISSING — never masked as OOV).
fn resolve_west_span(
    evt: &ScoreEvt,
    span: &[ScoreEvt],
    toks: &[Tok],
    dicts: &dyn DictSource,
) -> Result<Option<Vec<Vec<&'static str>>>> {
    let dict = dicts.words(evt.lang)?;
    // stage1: the word's traditional phones (override with spaces already handled by the caller — a
    // no-space override here is a single traditional phone).
    let trad: Vec<String> = if let Some(pi) = evt.phoneme_input {
        pi.split_whitespace().map(str::to_string).collect()
    } else {
        match dict.lookup(evt.lyric.trim()) {
            Some(t) => t,
            None => return Ok(None),
        }
    };
    if trad.is_empty() {
        return Ok(None);
    }
    let sylls = syllabify(dict, &trad);

    // distribute: consumers = the carrier note + every `+` note, in order. Consumer k takes ONE
    // syllable; the FINAL consumer (the min(consumers, syllables)-th) absorbs every remaining
    // syllable (SynthV squeeze). Holds — and `+` notes arriving after the syllables ran out — re-emit
    // the CURRENT syllable's nucleus.
    let n_consumers = (1 + toks.iter().skip(1).filter(|&&t| t == Tok::Next).count()).min(sylls.len());
    let mut assign_trad: Vec<Vec<String>> = vec![Vec::new(); span.len()];
    let mut cur_syl = 0usize; // syllable in effect (for holds)
    let mut next_syl = 0usize; // next unconsumed syllable
    let mut taken = 0usize; // consumers that took so far
    let mut last_holder = 0usize; // the note holding the word's LAST syllable
    for j in 0..span.len() {
        let takes = (j == 0 || toks[j] == Tok::Next) && next_syl < sylls.len();
        if takes {
            taken += 1;
            let until = if taken == n_consumers { sylls.len() } else { next_syl + 1 };
            for syl in &sylls[next_syl..until] {
                assign_trad[j].extend(syl.iter().cloned());
            }
            cur_syl = until - 1;
            next_syl = until;
            last_holder = j;
        } else {
            let syl = &sylls[cur_syl];
            assign_trad[j].push(syl[nucleus_idx(dict, syl)].clone());
        }
    }

    // 归韵 (coda deferral): the WORD-FINAL coda (phones after the last syllable's nucleus) moves to
    // the END of the span's LAST note, so "light --" sings l-aɪ | aɪ | aɪ-t (never li-t-aaa). The
    // holder's assignment ends with that syllable, so the truncate is always in-bounds.
    let last_note = span.len() - 1;
    let coda: Vec<String> = {
        let syl = &sylls[sylls.len() - 1];
        syl[nucleus_idx(dict, syl) + 1..].to_vec()
    };
    if !coda.is_empty() && last_holder != last_note {
        let a = &mut assign_trad[last_holder];
        a.truncate(a.len() - coda.len());
        assign_trad[last_note].extend(coda);
    }

    // stage2 each note's traditional phones → interned IPA (a bad phone = real OOV, not an error)
    let mut out: Vec<Vec<&'static str>> = Vec::with_capacity(assign_trad.len());
    for tr in assign_trad {
        match stage2(evt.lang, &tr) {
            Ok(ph) => out.push(ph),
            Err(_) => return Ok(None),
        }
    }
    Ok(Some(out))
}

/// STRICT resolve for the render: every note must resolve (LOUD `VOCAL_OOV` otherwise).
pub fn resolve_score(score: &[ScoreEvt], dicts: &dyn DictSource) -> Result<Vec<ResolvedNote>> {
    resolve_core(score, dicts, true)
}

/// LENIENT resolve for the editor (§9.5 single classifier): per-note verdicts, OOV as `Unknown` —
/// same code path as the render, so the marking can never drift from what actually renders. `Err` =
/// infrastructure failure (missing dictionary): the caller must NOT mark notes (red marks would point
/// the user at their lyrics when the problem is a missing file — the render reports it precisely).
pub fn classify_score(score: &[ScoreEvt], dicts: &dyn DictSource) -> Result<Vec<LyricClass>> {
    Ok(resolve_core(score, dicts, false)?
        .into_iter()
        .map(|nt| match nt.kind {
            ResolvedKind::Rest => LyricClass::Rest,
            ResolvedKind::Breath => LyricClass::Breath,
            ResolvedKind::Phones(ph) => {
                if nt.is_sustain {
                    LyricClass::Sustain
                } else {
                    LyricClass::Phones { phones: ph }
                }
            }
            ResolvedKind::Unknown => LyricClass::Unknown,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::super::g2p_golden_ref::G2P_GOLDEN;
    use super::*;

    fn lang_of(code: &str) -> Lang {
        match code {
            "zh" => Lang::Zh,
            "en" => Lang::En,
            "de" => Lang::De,
            "fr" => Lang::Fr,
            "es" => Lang::Es,
            "it" => Lang::It,
            "ja" => Lang::Ja, // never in the golden vectors; the probe below uses it
            other => panic!("unexpected golden lang {other}"),
        }
    }

    fn id_map() -> HashMap<&'static str, i64> {
        tbl::PHONE_TO_ID.iter().copied().collect()
    }

    // ── THE stage2 GATE: every golden row (dumped by phoneme_vocab.py + dict_fixes.py over the shipped
    // dictionaries, coverage-guaranteed) must convert bit-exact. Hermetic (no dictionary files needed).
    #[test]
    fn stage2_matches_python_golden() {
        let ids = id_map();
        let mut n = 0usize;
        for &(lang, _word, src, expect) in G2P_GOLDEN {
            let phones: Vec<String> = src.split_whitespace().map(str::to_string).collect();
            let got = stage2(lang_of(lang), &phones)
                .unwrap_or_else(|p| panic!("stage2({lang}, {src:?}) rejected phone {p:?}"));
            let got_ids: Vec<i64> = got.iter().map(|p| ids[p]).collect();
            let want: Vec<i64> = expect.split_whitespace().map(|s| s.parse().unwrap()).collect();
            assert_eq!(got_ids, want, "stage2 mismatch: lang={lang} src={src}");
            n += 1;
        }
        assert!(n > 3000, "golden vector count sanity ({n})");
    }

    // ── fixture dictionaries (hermetic stage1/span/phrase tests — tiny, inline) ──
    fn en_fixture() -> WordDict {
        // two/fun make bare T/F legal onsets (as in the real dictionary), so beautiful syllabifies
        // B Y UW1 | T AH0 | F AH0 L; NG stays coda-only (no fixture word starts with it).
        WordDict::from_tsv(
            Lang::En,
            "light\tL AY1 T\nbeautiful\tB Y UW1 T AH0 F AH0 L\ntree\tT R IY1\nsinger\tS IH1 NG ER0\nextra\tEH1 K S T R AH0\ntwo\tT UW1\nfun\tF AH1 N\nlove\tL AH1 V\ndon't\tD OW1 N T\n",
        )
    }
    fn zh_fixture() -> ZhDict {
        ZhDict::from_tsv(
            "zhang\tzh ang\nchang\tch ang\nda\td a\nle\tl e\nliao\tl iao\njie\tj ie\nzhi\tzh i\n",
            "长\tzhang,chang\n大\tda\n了\tle,liao\n解\tjie\n之\tzhi\n",
            "长大\tzhang da\n了解\tliao jie\n",
        )
    }
    fn de_fixture() -> WordDict {
        // haben's n̩ is a SYLLABIC consonant (nucleus) — baum makes bare B a legal onset. NB the de
        // TRADITIONAL layer spells the diphthong "aw" (MFA notation; stage2 normalizes it to aʊ).
        // `weiss` mirrors the SHIPPED spelling: upstream german_mfa is Swiss-orthography and has no
        // ß rows at all, which is exactly what the S86 lookup ladder has to bridge.
        WordDict::from_tsv(Lang::De, "haben\th aː b n̩\nbaum\tb aw m\nweiss\tv aj s\n")
    }
    struct Fixtures {
        zh: ZhDict,
        en: WordDict,
        de: WordDict,
    }
    impl DictSource for Fixtures {
        fn zh(&self) -> Result<&ZhDict> {
            Ok(&self.zh)
        }
        fn words(&self, lang: Lang) -> Result<&WordDict> {
            match lang {
                Lang::En => Ok(&self.en),
                Lang::De => Ok(&self.de),
                _ => Err(UtaiError::Inference("VOCAL_DICT_MISSING: fixture".into())),
            }
        }
    }
    fn fixtures() -> Fixtures {
        Fixtures { zh: zh_fixture(), en: en_fixture(), de: de_fixture() }
    }
    fn evt(lyric: &str, lang: Lang) -> ScoreEvt<'_> {
        ScoreEvt { lyric, note_num: 60, frames: 20, lang, phoneme_input: None }
    }
    fn phones_of(nt: &ResolvedNote) -> Vec<&'static str> {
        match &nt.kind {
            ResolvedKind::Phones(p) => p.clone(),
            other => panic!("expected phones, got {other:?}"),
        }
    }

    // ── syllabification: data-driven maximal onset ──
    #[test]
    fn syllabify_maximal_onset() {
        let d = en_fixture();
        let s = |w: &str| -> Vec<Vec<String>> {
            syllabify(&d, &d.lookup(w).unwrap())
        };
        // singer: NG is never word-initial in the fixture → it closes the first syllable (si-ng.er → "sing-er")
        assert_eq!(s("singer"), vec![vec!["S", "IH1", "NG"], vec!["ER0"]]);
        // extra: "T R" is a legal onset (tree) but "S T R" / "K S T R" are not observed → EH1 K S | T R AH0
        assert_eq!(s("extra"), vec![vec!["EH1", "K", "S"], vec!["T", "R", "AH0"]]);
        // beautiful: B Y UW1 | T AH0 | F AH0 L
        assert_eq!(s("beautiful"), vec![vec!["B", "Y", "UW1"], vec!["T", "AH0"], vec!["F", "AH0", "L"]]);
        // single-syllable word stays whole
        assert_eq!(s("light"), vec![vec!["L", "AY1", "T"]]);
    }

    // ── S86 input tolerance: the score keeps the lyric the user typed; the LOOKUP is what bends ──
    #[test]
    fn lookup_ladder_is_faithful_first() {
        // most-faithful first, and only the forms that actually differ are added
        assert_eq!(lookup_candidates("Love,"), vec!["love,", "love"]);
        assert_eq!(lookup_candidates("wei\u{00df}"), vec!["wei\u{00df}", "weiss"]);
        assert_eq!(lookup_candidates("'bout"), vec!["'bout"], "leading ' is word-internal, never trimmed");
        assert_eq!(lookup_candidates("re-do"), vec!["re-do"], "internal hyphen kept");
        assert!(lookup_candidates("...").iter().all(|k| k == "..."), "punctuation-only stays as-is (OOV)");
        // ★ the quote fold is a RUNG, never a rewrite of the base: it.tsv ships 7 keys spelled with
        //   U+2018, so folding candidate #0 would make them permanently unreachable (review R4/SS-4).
        let curly = lookup_candidates("don\u{2019}t");
        assert_eq!(curly[0], "don\u{2019}t", "the typed spelling gets first refusal");
        assert!(curly.contains(&"don't".to_string()), "…and the ASCII fold is still reachable");
        let u2018 = lookup_candidates("\u{2018}ndrangheta");
        assert_eq!(u2018[0], "\u{2018}ndrangheta", "a real it.tsv U+2018 key stays reachable");
    }

    #[test]
    fn lookup_tolerates_typography_and_esszett() {
        let f = fixtures();
        let sing = |lyric: &str, lang: Lang| -> Vec<&'static str> {
            phones_of(&resolve_score(&[evt(lyric, lang)], &f).unwrap()[0])
        };
        // ß: the shipped de dictionary has NO ß rows, so this is the only way `weiß` ever sings
        assert_eq!(sing("wei\u{00df}", Lang::De), vec!["v", "aɪ", "s"]);
        assert_eq!(sing("Wei\u{00df}", Lang::De), vec!["v", "aɪ", "s"], "capitalized German noun");
        // typographic apostrophe (what every phone keyboard emits) == ASCII apostrophe
        assert_eq!(sing("don\u{2019}t", Lang::En), sing("don't", Lang::En));
        // punctuation glued on by a pasted lyric sheet
        assert_eq!(sing("Love,", Lang::En), vec!["l", "ʌ", "v"]);
        assert_eq!(sing("(love)", Lang::En), vec!["l", "ʌ", "v"]);
        // …and a genuinely unknown word is still a LOUD OOV — tolerance must not invent hits
        assert!(resolve_score(&[evt("zzzzq,", Lang::En)], &f).is_err());
    }

    // ── S86: `rest`/`sil`/`pau` are no longer stolen from the lyric vocabulary (UTAU convention = R) ──
    #[test]
    fn only_the_utau_r_convention_stays_reserved() {
        let f = fixtures();
        for r in ["R", "r", "", "  "] {
            let got = resolve_score(&[evt(r, Lang::En)], &f).unwrap();
            assert!(matches!(got[0].kind, ResolvedKind::Rest), "{r:?} must stay a rest token");
        }
        // `sil`/`pau` are freed too: once a word can be split across notes they show up as fragments
        // (sil|ver, pau|se), so a reserved token that can be part of a word collides by construction.
        for w in ["rest", "sil", "pau"] {
            assert!(!is_silent_token(w), "{w:?} must be available as lyric material");
        }
        // ★ the DAW-side note grouping must key off the SAME predicate the cv side resolves with —
        //   `score2cv::classify_lyric` keeps a WIDER rest set (upstream parity port) and using it in
        //   `compute_note_groups` would desync every later group index (review R2/SS-1/GATE-3).
        for lyric in ["R", "r", "", "AP", "ap", "  "] {
            assert!(is_silent_token(lyric), "{lyric:?} produces no sung phones");
        }
        for lyric in ["rest", "sil", "pau", "light", "か"] {
            assert!(!is_silent_token(lyric), "{lyric:?} is a WORD for the DAW side too");
        }
        // the real English word now sings, and case no longer decides whether it does
        let f2 = fixtures();
        let lower = resolve_score(&[evt("rest", Lang::En)], &f2);
        assert!(lower.is_err(), "not in the tiny fixture dict → LOUD OOV, never silence");
        assert!(lower.unwrap_err().to_string().contains("VOCAL_OOV: rest"));
        // and a ja score treats it as a word too (→ OOV), instead of silently eating the note
        assert!(resolve_score(&[evt("rest", Lang::Ja)], &f2).is_err());
    }

    // ── S86: JA multi-mora on ONE note used to be SILENTLY truncated to its head mora (no OOV, no
    //    mark). It must fail LOUDLY instead — and punctuation must still be tolerated, or a pasted
    //    Japanese lyric sheet would go wall-to-wall OOV the moment the truncation fallback died. ──
    #[test]
    fn ja_multi_mora_on_one_note_sings_in_full() {
        let f = fixtures();
        let ok = |lyric: &str| -> Vec<&'static str> {
            phones_of(&resolve_score(&[evt(lyric, Lang::Ja)], &f).unwrap()[0])
        };
        // these ALL used to sing just their HEAD mora, silently (ずっと → [z ɯ])
        assert_eq!(ok("ずっと"), vec!["z", "ɯ", "ʔ", "t", "o"]);
        assert_eq!(ok("きっと"), vec!["k", "i", "ʔ", "t", "o"]);
        assert_eq!(ok("まって"), vec!["m", "a", "ʔ", "t", "e"]);
        assert_eq!(ok("がっこう"), vec!["ɡ", "a", "ʔ", "k", "o", "ɯ"]);
        assert_eq!(ok("ちょっと"), vec!["tɕ", "o", "ʔ", "t", "o"]);
        // the split the user actually writes: 「ず」「っと」 — the sokuon leads the second note
        assert_eq!(ok("っと"), vec!["ʔ", "t", "o"]);
        assert_eq!(ok("ずっ"), vec!["z", "ɯ", "ʔ"], "trailing sokuon is legal too");
        // ー contributes NO phone: training lengthens a vowel by DURATION on one phone, never by a
        // second copy of it (review JA-3). The note's own frames already carry the length.
        assert_eq!(ok("ずーっと"), vec!["z", "ɯ", "ʔ", "t", "o"]);
        assert_eq!(ok("あー"), vec!["a"]);
        assert_eq!(ok("カー"), vec!["k", "a"]);
        assert_eq!(ok("んー"), vec!["ɴ"], "holding the moraic nasal is ordinary Japanese");
        // ★ UTAU appended-voicebank / CVVC alias suffixes: a NON-KANA tail names a voicebank flavour,
        //   it is not part of the word. The repo's own .ust corpus has 26 such notes (review JA-1/R1).
        assert_eq!(ok("あ弱"), vec!["a"]);
        assert_eq!(ok("か強"), vec!["k", "a"]);
        assert_eq!(ok("あ t"), vec!["a"], "CVVC alias tail");
        assert_eq!(ok("か_G3"), vec!["k", "a"], "appended pitch suffix");
        assert_eq!(ok("きゃ強"), vec!["c", "a"], "yōon + suffix");
        // ★ longest-match-first: ぁぃぅぇぉ are standalone KANA keys, so a shortest-first scan would
        //   silently mis-parse these as base+small-vowel-as-its-own-mora
        assert_eq!(ok("ふぁい"), vec!["ɸ", "a", "i"], "NOT ɸ ɯ a i");
        assert_eq!(ok("うぉん"), vec!["w", "o", "ɴ"]);
        // Known cost of the suffix rule (unchanged from pre-S86, where the truncating fallback did the
        // same): a kana head with a junk tail sings the head rather than erroring. Only a lyric where
        // NOTHING parses as kana is a LOUD OOV.
        assert_eq!(ok("かtta"), vec!["k", "a"]);
        assert!(resolve_score(&[evt("恋", Lang::Ja)], &f).is_err(), "kanji still OOV");
        assert!(resolve_score(&[evt("zzz", Lang::Ja)], &f).is_err(), "no kana at all → OOV");
        assert_eq!(ok("か"), vec!["k", "a"]);
        assert_eq!(ok("きゃ"), vec!["c", "a"]);
        assert_eq!(ok("しょ"), vec!["ɕ", "o"]);
        assert_eq!(ok("うぉ"), vec!["w", "o"], "S69 foreign kana still bypasses the chain");
        assert_eq!(ok("っ"), vec!["ʔ"]);
        assert_eq!(ok("ん"), vec!["ɴ"]);
        assert_eq!(ok("ゔ"), vec!["v", "ɯ"], "S58 KANA_EXTRA row");
        assert_eq!(ok("カ"), vec!["k", "a"], "katakana folding");
        assert_eq!(ok("tta"), vec!["ʔ", "t", "a"], "romaji geminate — never touched the kana chain");
        // punctuation tolerance must arrive in the SAME round as the truncation fix
        assert_eq!(ok("か、"), vec!["k", "a"]);
        assert_eq!(ok("か。"), vec!["k", "a"]);
    }

    // ── western span: coda deferral (归韵) on pure holds ──
    #[test]
    fn west_span_coda_deferral() {
        let f = fixtures();
        let score = [evt("light", Lang::En), evt("-", Lang::En), evt("-", Lang::En)];
        let r = resolve_score(&score, &f).unwrap();
        // L AY1 T syllable: note0 = [l aɪ], holds re-emit aɪ, the coda t closes the LAST note.
        assert_eq!(phones_of(&r[0]), vec!["l", "aɪ"]);
        assert_eq!(phones_of(&r[1]), vec!["aɪ"]);
        assert_eq!(phones_of(&r[2]), vec!["aɪ", "t"]);
        assert!(r[1].is_sustain && r[2].is_sustain);
    }

    // ── western span: `+` advances syllables (SynthV), remainder squeezes into the last consumer ──
    #[test]
    fn west_span_plus_advances_syllables() {
        let f = fixtures();
        // beau-ti-ful over [word, +, -]: note0=beau, note1=ti+ful (last consumer takes the rest),
        // note2 holds ʊ, and the word-final coda l defers to note2 (AH0→ə).
        let score = [evt("beautiful", Lang::En), evt("+", Lang::En), evt("-", Lang::En)];
        let r = resolve_score(&score, &f).unwrap();
        assert_eq!(phones_of(&r[0]), vec!["b", "j", "u"]);
        assert_eq!(phones_of(&r[1]), vec!["t", "ə", "f", "ə"]);
        assert_eq!(phones_of(&r[2]), vec!["ə", "l"]);
        // word alone on ONE note: everything (incl. the coda) on that note.
        let solo = [evt("beautiful", Lang::En)];
        let r1 = resolve_score(&solo, &f).unwrap();
        assert_eq!(phones_of(&r1[0]), vec!["b", "j", "u", "t", "ə", "f", "ə", "l"]);
    }

    // ── zh: greedy phrase disambiguation over the NOTE SEQUENCE (长大 → zhǎng, not cháng) ──
    #[test]
    fn zh_phrase_greedy_polyphones() {
        let f = fixtures();
        let score = [evt("长", Lang::Zh), evt("大", Lang::Zh)];
        let r = resolve_score(&score, &f).unwrap();
        assert_eq!(phones_of(&r[0]), vec!["ʈʂ", "ɑŋ"], "长大 phrase → zhang (ʈʂ ɑŋ)");
        // isolated 长 → char default reading (first = zhang per fixture kMandarin order)
        let solo = [evt("长", Lang::Zh)];
        let r1 = resolve_score(&solo, &f).unwrap();
        assert_eq!(phones_of(&r1[0]), vec!["ʈʂ", "ɑŋ"]);
        // 了解 phrase → 了 reads liǎo (not the default le)
        let score2 = [evt("了", Lang::Zh), evt("解", Lang::Zh)];
        let r2 = resolve_score(&score2, &f).unwrap();
        assert_eq!(phones_of(&r2[0]), vec!["l", "iaʊ"], "了解 → liao");
        // pinyin lyric bypasses the hanzi path; A1 apical-i fires (zhi → ʈʂ ɻ̩)
        let score3 = [evt("zhi", Lang::Zh)];
        let r3 = resolve_score(&score3, &f).unwrap();
        assert_eq!(phones_of(&r3[0]), vec!["ʈʂ", "ɻ̩"]);
    }

    // ── zh sustain re-emits the FINAL (whole final token, coda included — it is atomic in the vocab) ──
    #[test]
    fn zh_sustain_reemits_final() {
        let f = fixtures();
        let score = [evt("长", Lang::Zh), evt("-", Lang::Zh)];
        let r = resolve_score(&score, &f).unwrap();
        assert_eq!(phones_of(&r[1]), vec!["ɑŋ"], "hold re-emits the syllable's final");
    }

    // ── phoneme_input overrides (§3.7): no-space = syllable; with-space = raw traditional phones ──
    #[test]
    fn phoneme_input_overrides() {
        let f = fixtures();
        let mut e1 = evt("长", Lang::Zh);
        e1.phoneme_input = Some("chang");
        let r1 = resolve_score(&[e1], &f).unwrap();
        assert_eq!(phones_of(&r1[0]), vec!["ʈʂʰ", "ɑŋ"], "pinyin override wins over the phrase/default");
        let mut e2 = evt("xxxx", Lang::En);
        e2.phoneme_input = Some("L AY1 T");
        let r2 = resolve_score(&[e2], &f).unwrap();
        assert_eq!(phones_of(&r2[0]), vec!["l", "aɪ", "t"], "raw ARPABET override bypasses the dict");
    }

    // ── run languages: sustains inherit the carrier, rests attach to the previous run ──
    #[test]
    fn run_langs_inherit_and_attach() {
        let score = [
            evt("R", Lang::En),  // leading rest → first run (ja)
            evt("か", Lang::Ja),
            evt("-", Lang::En),  // sustain: inherits ja (its own lang field is IGNORED)
            evt("R", Lang::En),  // rest attaches to the PREVIOUS run (ja) → the cut lands in silence
            evt("light", Lang::En),
        ];
        let langs = note_run_langs(&score);
        assert_eq!(langs, vec![Lang::Ja, Lang::Ja, Lang::Ja, Lang::Ja, Lang::En]);
    }

    // ── katakana folding (S58 coverage fix) ──
    #[test]
    fn katakana_folds_to_hiragana() {
        assert_eq!(fold_katakana("カ"), "か");
        assert_eq!(fold_katakana("ギュ"), "ぎゅ");
        assert_eq!(fold_katakana("ー"), "ー", "the prolonged-sound mark is NOT folded (sustain token)");
        let f = fixtures();
        let r = resolve_score(&[evt("カ", Lang::Ja)], &f).unwrap();
        assert_eq!(phones_of(&r[0]), vec!["k", "a"], "katakana lyric now sings (used to OOV)");
        // the new KANA_EXTRA rows: ぎゅ (missing pre-S58) resolves via gyu → [ɟ, ɯ]
        let r2 = resolve_score(&[evt("ぎゅ", Lang::Ja)], &f).unwrap();
        assert_eq!(phones_of(&r2[0]), vec!["ɟ", "ɯ"]);
    }

    // ── language-run chunking: a zh|en switch cuts the chunk; direct voiced contact = hard seam,
    //    via a rest = soft (the rest attaches to the previous run, so the cut lands in silence) ──
    #[test]
    fn chunking_cuts_at_language_change() {
        use super::super::score2cv::{build_arrays_daw, chunk_at_sp};
        let f = fixtures();
        let score = [evt("长", Lang::Zh), evt("light", Lang::En)];
        let arr = build_arrays_daw(&score, &f).unwrap();
        let chunks = chunk_at_sp(&arr, 400);
        assert_eq!(chunks.len(), 2, "language change forces a cut even under max_frames");
        assert_eq!(chunks[0].lang_id, Lang::Zh.id());
        assert_eq!(chunks[1].lang_id, Lang::En.id());
        assert!(!chunks[0].hard_seam, "first chunk has no leading seam");
        assert!(chunks[1].hard_seam, "mid-voiced language cut → hard seam (micro-fade)");
        let score2 = [evt("长", Lang::Zh), evt("R", Lang::Zh), evt("light", Lang::En)];
        let arr2 = build_arrays_daw(&score2, &f).unwrap();
        let chunks2 = chunk_at_sp(&arr2, 400);
        assert_eq!(chunks2.len(), 2);
        assert!(!chunks2[1].hard_seam, "cut adjacent to SP is a soft seam (silence)");
    }

    // ── grouping: same-pitch notes in DIFFERENT languages must be separate note groups (a group
    //    spanning a language cut would desync note_dur inside the rebased chunks) ──
    #[test]
    fn groups_never_span_languages() {
        let f = fixtures();
        let score = [evt("长", Lang::Zh), evt("light", Lang::En)]; // same pitch 60
        let arr = super::super::score2cv::build_arrays_daw(&score, &f).unwrap();
        assert_ne!(
            arr.note_to_phone[0],
            *arr.note_to_phone.last().unwrap(),
            "same-pitch cross-language notes form separate groups"
        );
    }

    // ── #[ignore] E2E: the SHIPPED dictionaries load + look up every golden word (needs the 18MB
    //    data/dictionaries TSVs). Run:
    //      cargo test --lib inference::g2p::tests::dictionaries_end_to_end -- --ignored --nocapture
    #[test]
    #[ignore]
    fn dictionaries_end_to_end() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        set_dict_dir(root.join("../data/dictionaries"));
        let g = GlobalDicts;
        let zh = g.zh().expect("zh dictionaries load");
        let (mut primary, mut total) = (0usize, 0usize);
        for &(lang, word, src, _) in G2P_GOLDEN {
            let src_ph: Vec<String> = src.split_whitespace().map(str::to_string).collect();
            if lang == "zh" {
                assert_eq!(zh.syllable_phones(word).unwrap(), src_ph, "zh syllable {word}");
                continue;
            }
            let d = g.words(lang_of(lang)).expect("dict loads");
            let got = d.lookup(word).unwrap_or_else(|| panic!("{lang} lookup missing: {word}"));
            total += 1;
            if got == src_ph {
                primary += 1;
            }
        }
        // golden rows sample raw TSV lines, so some hit NON-primary pronunciations; the loader keeps
        // the primary (first) — equality holds for the vast majority, membership for all.
        assert!(primary * 10 >= total * 8, "primary-pron match rate too low: {primary}/{total}");
        eprintln!("[g2p-e2e] zh syllables all exact; word lookups {total}, primary matches {primary}");
    }

    // ── #[ignore] DIAGNOSTIC PROBE (S86 dictionary work-line): run the REAL engine over the REAL
    //    shipped dictionaries so every audit finding is grounded in behaviour, not in reading code.
    //      UTAI_G2P_PROBE=<file>  each non-empty, non-`#` line is  <lang> TAB <note>|<note>|...
    //    Run: UTAI_G2P_PROBE=probe.txt cargo test --lib inference::g2p::tests::g2p_probe -- --ignored --nocapture
    #[test]
    #[ignore]
    fn g2p_probe() {
        // SKIP (never fail) without the env var: `--include-ignored` is a legitimate gate mode, and a
        // diagnostic probe must not be able to redden it.
        let Ok(spec) = std::env::var("UTAI_G2P_PROBE") else {
            println!("[g2p-probe] skipped — set UTAI_G2P_PROBE=<file> to run it");
            return;
        };
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        set_dict_dir(root.join("../data/dictionaries"));
        let g = GlobalDicts;
        for line in std::fs::read_to_string(&spec).expect("probe file").lines() {
            let line = line.trim_end();
            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((code, notes)) = line.split_once('\t') else {
                println!("!! malformed probe line (need a TAB): {line}");
                continue;
            };
            let lang = lang_of(code.trim());
            let lyrics: Vec<&str> = notes.split('|').collect();
            let score: Vec<ScoreEvt> = lyrics.iter().map(|l| evt(l, lang)).collect();
            println!("\n=== [{code}] {}", lyrics.join(" | "));
            // stage1 + syllabification for the western languages (the whole-word head note)
            if !matches!(lang, Lang::Ja | Lang::Zh) {
                match g.words(lang) {
                    Ok(d) => {
                        let head = lyrics[0].trim();
                        match d.lookup(head) {
                            Some(trad) => {
                                let sy = syllabify(d, &trad);
                                println!("    trad: {}", trad.join(" "));
                                println!(
                                    "    syl : {}",
                                    sy.iter().map(|s| s.join(" ")).collect::<Vec<_>>().join("  /  ")
                                );
                            }
                            None => println!("    trad: <OOV — not in {code}.tsv>"),
                        }
                    }
                    Err(e) => println!("    !! dictionary error: {e}"),
                }
            }
            // RENDER path (authoritative — the editor's classify collapses `+` notes to "Sustain",
            // which would hide the phones those notes actually sing).
            match resolve_score(&score, &g) {
                Ok(notes) => {
                    for (i, nt) in notes.iter().enumerate() {
                        let what = match &nt.kind {
                            ResolvedKind::Rest => "REST".to_string(),
                            ResolvedKind::Breath => "BREATH".to_string(),
                            ResolvedKind::Unknown => "**OOV**".to_string(),
                            ResolvedKind::Phones(p) => p.join(" "),
                        };
                        let tag = if nt.is_sustain { " [sustain]" } else { "" };
                        println!("    note[{i}] {:>12} -> {what}{tag}", format!("{:?}", lyrics[i]));
                    }
                }
                Err(e) => println!("    render(strict): {e}"),
            }
            // editor verdicts must agree with the render on WHICH notes are OOV
            match classify_score(&score, &g) {
                Ok(cl) => {
                    let bad: Vec<usize> =
                        (0..cl.len()).filter(|&i| matches!(cl[i], LyricClass::Unknown)).collect();
                    if !bad.is_empty() {
                        println!("    editor marks OOV at {bad:?}");
                    }
                }
                Err(e) => println!("    !! classify error: {e}"),
            }
        }
    }

    // ── OOV verdicts: strict errors with the CODE; lenient marks ONLY the bad note ──
    #[test]
    fn oov_strict_and_lenient() {
        let f = fixtures();
        let score = [evt("light", Lang::En), evt("zzzzq", Lang::En), evt("か", Lang::Ja)];
        let err = resolve_score(&score, &f).unwrap_err().to_string();
        assert!(err.contains("VOCAL_OOV: zzzzq"), "strict render error carries the CODE + lyric: {err}");
        let classes = classify_score(&score, &f).unwrap();
        assert!(matches!(classes[0], LyricClass::Phones { .. }));
        assert!(matches!(classes[1], LyricClass::Unknown));
        assert!(matches!(classes[2], LyricClass::Phones { .. }), "notes after the OOV still classify");
    }

    // ── audit MAJOR: a MISSING DICTIONARY must surface as VOCAL_DICT_MISSING — never masked as
    //    per-word VOCAL_OOV (which points the user at their lyrics instead of the broken install) ──
    #[test]
    fn dict_missing_is_not_oov() {
        use super::super::score2cv::NoDicts;
        for score in [vec![evt("长", Lang::Zh)], vec![evt("light", Lang::En)]] {
            let err = resolve_score(&score, &NoDicts).unwrap_err().to_string();
            assert!(err.contains("VOCAL_DICT_MISSING"), "infrastructure error surfaces its own CODE: {err}");
            assert!(!err.contains("VOCAL_OOV"), "never masked as OOV: {err}");
            assert!(classify_score(&score, &NoDicts).is_err(), "lenient classify propagates too (no wrong red marks)");
        }
        // pure-JA scores never touch the dictionaries → still fine with none present
        assert!(resolve_score(&[evt("か", Lang::Ja)], &NoDicts).is_ok());
    }

    // ── audit MAJOR: sustains/rests are TRANSPARENT to the zh phrase window — [了][-][解] (melisma on
    //    了) must still read 了解 = liǎo, and a breath gap must not break 长大 ──
    #[test]
    fn zh_phrase_window_transparent_over_sustains_and_rests() {
        let f = fixtures();
        let score = [evt("了", Lang::Zh), evt("-", Lang::Zh), evt("解", Lang::Zh)];
        let r = resolve_score(&score, &f).unwrap();
        assert_eq!(phones_of(&r[0]), vec!["l", "iaʊ"], "了解 across a sustain → liao (not the 了 default le)");
        assert_eq!(phones_of(&r[1]), vec!["iaʊ"], "the hold re-emits the resolved final");
        let score2 = [evt("长", Lang::Zh), evt("R", Lang::Zh), evt("大", Lang::Zh)];
        let r2 = resolve_score(&score2, &f).unwrap();
        assert_eq!(phones_of(&r2[0]), vec!["ʈʂ", "ɑŋ"], "长大 across a rest → zhang");
    }

    // ── audit MINOR: de syllabic consonants (l̩/m̩/n̩) count as nuclei — haben = ha|bn̩ (2 syllables),
    //    so a + note takes the second syllable instead of a bare vowel hold ──
    #[test]
    fn de_syllabic_consonant_is_nucleus() {
        let f = fixtures();
        let d = de_fixture();
        let sylls = syllabify(&d, &d.lookup("haben").unwrap());
        assert_eq!(sylls, vec![vec!["h", "aː"], vec!["b", "n̩"]], "n̩ carries the 2nd syllable");
        let score = [evt("haben", Lang::De), evt("+", Lang::De)];
        let r = resolve_score(&score, &f).unwrap();
        assert_eq!(phones_of(&r[0]), vec!["h", "aː"]);
        assert_eq!(phones_of(&r[1]), vec!["b", "n̩"], "+ advances to the syllabic-consonant syllable");
    }

    // ── audit MINOR: malformed zh TSV rows are dropped at load (empty phones / phrase-length mismatch)
    //    so a hand-edited dictionary degrades to LOUD OOV instead of silent frame desync ──
    #[test]
    fn zh_dict_drops_malformed_rows() {
        let d = ZhDict::from_tsv("ok\to k\nbad\t\n", "长\tzhang\n", "长大\tzhang\n好了\thao le\n");
        assert!(d.syllable_phones("ok").is_some());
        assert!(d.syllable_phones("bad").is_none(), "empty-phones row dropped → OOV, not zero phones");
        assert!(!d.phrases.contains_key("长大"), "syllable-count mismatch row dropped");
        assert!(d.phrases.contains_key("好了"));
    }
}
