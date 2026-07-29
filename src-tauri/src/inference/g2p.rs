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

use super::g2p_alias::{alias_phones, PhonemeSet};
use super::g2p_tables as gt;
use super::score2cv::{classify_lyric as classify_lyric_ja, is_nucleus_phone, LyricClass};
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
    /// S91: which UTAU alias convention this note's ENGLISH lyric is written in (`Words` = ordinary
    /// spelling, the default and the pre-S91 behaviour). Per-note like `lang`, but the command layer
    /// fans ONE per-track setting out over every note — a score never mixes conventions.
    pub phoneme_set: PhonemeSet,
}

impl<'a> ScoreEvt<'a> {
    /// JA-defaulted event from a legacy `(lyric, note_num, frames)` triple (parity paths + tests).
    pub fn ja(t: &(&'a str, i64, i64)) -> ScoreEvt<'a> {
        ScoreEvt {
            lyric: t.0,
            note_num: t.1,
            frames: t.2,
            lang: Lang::Ja,
            phoneme_input: None,
            phoneme_set: PhonemeSet::Words,
        }
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
///
/// S90 — **"no stress digit ≡ unstressed"**: a BARE `ah`/`AH` resolves to ə (it used to give ʌ).
/// Stress digits are a CMUdict property; the ARPABET people TYPE has none — OpenUtau phonetic hints
/// and ARPAsing voicebank reclists are written without them — so a bare `ah` is an unstressed vowel,
/// not a stressed one. Three reasons this is the right default: the shipped en.tsv has AH0 63181 vs
/// AH1+AH2 8022 (**7.9×**); the errors are asymmetric (ʌ read as ə is a mild centralization, ə read as
/// ʌ moves the PERCEIVED stress of the word); and `[w ah dh]`-style hints in real UST files are exactly
/// the reduced-vowel case. Write `ah1`/`AH1` (or `ah2`) to get ʌ.
/// AH is today the only ARPABET symbol whose IPA splits on stress, so the general rule and this one
/// branch coincide — but the rule is what the docs promise, so a future split must follow it too.
/// ⚠ ZERO effect on the dictionary path: en.tsv contains 0 bare-AH tokens and the stage2 golden
/// vectors 0 — walked exhaustively by `arpabet_stressless_nucleus_is_zero_regression` (golden, in this
/// file) and by `dictionaries_end_to_end` (the whole shipped en.tsv).
fn convert_arpabet(p: &str) -> String {
    let lower = p.to_ascii_lowercase();
    if matches!(lower.as_str(), "sil" | "sp" | "spn") {
        return "SP".to_string();
    }
    let up = p.to_ascii_uppercase();
    if let Some(stress) = up.strip_prefix("AH") {
        match stress {
            "1" | "2" => return "ʌ".to_string(),
            // (spelled out rather than left to fall through: the generated table's `("AH","ə")` row —
            //  dead code until now — would agree, but a rule this load-bearing should not depend on a
            //  row of a file we regenerate from another repo. Mutation-checked both ways.)
            "" | "0" => return "ə".to_string(),
            // not an AH symbol at all (a typo like `AHX`) — fall through to the generic path, which
            // leaves it unmapped so `stage2` reports it LOUDLY instead of guessing a vowel.
            _ => {}
        }
    }
    let base = up.strip_suffix(['0', '1', '2']).unwrap_or(&up);
    if let Some(&ipa) = arpabet_map().get(base) {
        return ipa.to_string();
    }
    p.to_string()
}

/// Is this token a REAL ARPABET symbol (case-insensitive, optional CMUdict stress digit)?
///
/// S91: the alias conventions hand raw tokens to `stage2`, and `convert_arpabet` passes an unknown
/// symbol through UNMAPPED so interning rejects it — correct, but the resulting error names the whole
/// lyric. Asking here instead lets the alias layer name the offending SYMBOL. Deliberately does NOT
/// accept `sil`/`sp`/`spn`: those mean silence to `convert_arpabet`, and a silence token hiding inside
/// a sung alias is exactly the kind of quiet surprise S90 spent a round removing.
pub(crate) fn arpabet_is_known(tok: &str) -> bool {
    let up = tok.to_ascii_uppercase();
    let base = up.strip_suffix(['0', '1', '2']).unwrap_or(&up);
    arpabet_map().contains_key(base)
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
    /// A traditional-layer phone is a syllable NUCLEUS. Delegates to `dict_is_vowel` — ONE
    /// implementation, so the dictionary BUILD pass (legal onsets) and the syllabifier can never
    /// drift apart (they were two copies of the same rule until S90).
    pub fn is_vowel(&self, ph: &str) -> bool {
        dict_is_vowel(self.lang, &self.vowels, ph)
    }

    /// Normalize an onset cluster for the `onsets` lookup. EN only, and ONLY here: the dictionary
    /// ships uppercase ARPABET, so that is what `onsets` is keyed by, while phonetic hints are
    /// conventionally lowercase (OpenUtau's symbol table is). Without this an un-normalized hint
    /// misses every non-empty onset and each consonant cluster falls into the PREVIOUS syllable's
    /// coda — `[k ae n d ah l ih t]` cuts as k-ae-n-d | ah-l | ih-t instead of can | dle | lit.
    ///
    /// ⚠ It folds the SYLLABIFIER'S LOOKUP KEY and nothing else. The first S90 draft normalized the
    /// phones themselves, which also fed `stage2` — and there `to_ascii_uppercase` is not a no-op:
    /// an ASCII IPA token like `ts` stops matching the vocab (`TS` interns to nothing = a brand-new
    /// silent OOV), so an override written in IPA on an EN note broke. Case belongs to the onset
    /// index, not to the user's phones. (Review S90 major #1.)
    fn onset_key(&self, cluster: String) -> String {
        match self.lang {
            Lang::En => cluster.to_ascii_uppercase(),
            _ => cluster,
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
    // ⚠ SQUARE BRACKETS ARE NOT PUNCTUATION HERE (S90 review major #2). They delimit a phonetic hint,
    // so a bracket-shaped lyric that `phoneme_hint` refused must NOT be rescued into a real word by
    // trimming them away: `[dr` (a missing `]`) trimmed to `dr` sings **"drive"**, `[k].` sings "kay",
    // `[[k]]` sings "kay" — a completely different word, no red mark, no error, straight past the
    // "loud failure" promise this feature is documented with. Zero cost: all 8 shipped dictionaries
    // contain 0 keys with a bracket in them, so nothing reachable is lost.
    let trim_punct = |s: &str| -> String {
        s.trim_matches(|c: char| {
            !c.is_alphanumeric() && c != '\'' && c != '-' && !matches!(c, '[' | ']' | '［' | '］')
        })
        .to_string()
    };

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

/// THE nucleus test for a TRADITIONAL-layer phone (en = ARPABET, the MFA languages = their own
/// generated vowel inventory). `WordDict::is_vowel` delegates here; `WordDict::from_tsv` calls it
/// directly while the dictionary is still half-built.
fn dict_is_vowel(lang: Lang, vowels: &HashSet<&'static str>, ph: &str) -> bool {
    match lang {
        Lang::En => en_is_nucleus(ph),
        _ => vowels.contains(ph),
    }
}

/// EN nucleus test — DERIVED, not a second hand-typed vowel list: strip the optional stress digit and
/// ask whether that ARPABET symbol's IPA (the generated `ARPABET_IPA` table) is nucleus-capable by
/// `score2cv::is_nucleus_phone`, the one classifier this repo already walks over all 210 vocab tokens.
///
/// ⚠ S90 replaced `ph.ends_with(['0','1','2'])` (= "carries a CMUdict stress digit") with this. Over the
/// SHIPPED en.tsv the two verdicts agree on every one of the 69 distinct tokens (863018 instances), so
/// the whole dictionary path — legal onsets, syllable cuts, coda deferral — is unchanged to the byte.
/// What it ADDS is the spelling users actually type: OpenUtau phonetic hints and ARPAsing reclists carry
/// NO stress digits, so `[dh ae dh]` had no nucleus at all and the entire word collapsed onto the first
/// note of its span (S86 audit `syl-en-override-without-stress-collapses`).
/// Case-insensitive because hints are conventionally lowercase while CMUdict is upper — but the
/// UPPERCASE membership test is tried first without allocating, which is the only path the dictionary
/// build (863018 tokens, all uppercase) ever takes.
pub(crate) fn en_is_nucleus(ph: &str) -> bool {
    let base = ph.strip_suffix(['0', '1', '2']).unwrap_or(ph);
    let syms = arpabet_nucleus_syms();
    syms.contains(base)
        || (!base.bytes().all(|b| b.is_ascii_uppercase()) && syms.contains(base.to_ascii_uppercase().as_str()))
}

/// The ARPABET symbols whose IPA can carry a nucleus — DERIVED once from the generated table, never a
/// second hand-typed vowel list (and `is_nucleus_phone` must be asked about the IPA, not the symbol:
/// bare `Y`/`y` would look like a nucleus to it, since IPA /y/ is a vowel, while ARPABET Y is /j/).
fn arpabet_nucleus_syms() -> &'static HashSet<&'static str> {
    static M: OnceLock<HashSet<&'static str>> = OnceLock::new();
    M.get_or_init(|| {
        gt::ARPABET_IPA.iter().filter(|&&(_, ipa)| is_nucleus_phone(ipa)).map(|&(sym, _)| sym).collect()
    })
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
            if dict.onsets.contains(&dict.onset_key(cluster[s..].join(" "))) {
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
/// S91: `set` because ONE reserved spelling genuinely collides with the alias conventions —
/// lowercase **`ap`** is an ordinary VC alias (`a`+`p`, the coda of *stop* / *top* / *drop*) in both
/// X-SAMPA and VCCV, and swallowing it as a breath renders an audible INHALE mid-word with no error
/// and no red mark (review S91, MAJOR). On an alias track it is therefore a WORD. Nothing is lost:
/// the canonical breath is `AP` — which the frontend always writes (`mapLyric` maps the track's
/// `breathToken` onto it) and which is not a legal alias in either table — and a track can point its
/// breath trigger anywhere it likes.
///
/// ⚠ `r` is deliberately NOT freed the same way, and the asymmetry is evidence-based rather than
/// tidy: a bare `r` really is used as a REST by real banks (in `duvet - vocal.ust` the two `r` notes
/// carry only Length/Lyric/NoteNum — byte-for-byte the shape of that file's 46 `R` rests, while every
/// sung note there also carries Envelope/PBW/PBS), and a genuine standalone /ɹ/ still has `-r`, `r-`
/// and `_r`. We have no such evidence for `ap` in any bank.
fn token_class(lyric: &str, set: PhonemeSet) -> Tok {
    match lyric.trim() {
        "R" | "r" | "" => Tok::Rest,
        "AP" => Tok::Breath,
        "ap" if set == PhonemeSet::Words => Tok::Breath,
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
/// ⚠ S91 added the `set` parameter for exactly the reason this doc comment gives: lowercase `ap` is
/// silent on a words track and SUNG on an alias track, so the two sides of the grouping must be asked
/// the same question. The compiler now forces every caller to say which track it is talking about.
pub fn is_silent_token(lyric: &str, set: PhonemeSet) -> bool {
    matches!(token_class(lyric, set), Tok::Rest | Tok::Breath)
}

/// An OpenUtau **phonetic hint**: square brackets CLOSING the lyric pin that note's phones —
/// `read[r iy d]` (the word stays readable, the phones come from the hint) or the bare `[dh ae dh]`
/// that UST files written against an ARPAsing bank use throughout. Returns the inner text.
///
/// It is the SAME user layer as `phoneme_input` (§3.7), only spelled inside the lyric, so `resolve_core`
/// folds it into that field once and every language branch downstream sees ONE override notion (en/de/
/// fr/es/it = traditional phones, ja = raw vocab IPA, zh = a pinyin syllable — unchanged semantics).
/// Three things it deliberately does NOT do:
///  * it never rewrites the lyric — the score keeps exactly what the user typed (import.rs stores the
///    raw UST line verbatim) and `token_class` keeps classifying that RAW text, so an all-hint lyric
///    stays a WORD. Stripping it to an empty word part would classify as a REST and silence the note;
///  * it never guesses at a malformed hint: `[k aa}` (a real typo in the wild, source file included)
///    has no closing bracket, so the lyric goes to the dictionary and fails LOUDLY as OOV on that note;
///  * an EMPTY hint (`[]`, `word[  ]`) is no hint at all — again the lyric goes to the dictionary as
///    written, rather than resolving to zero phones (which the span builder would report as OOV anyway).
///
/// ⚠ DELIBERATELY STRICTER THAN UPSTREAM in exactly one place. OpenUtau's `UNote.cs` uses a GREEDY
/// `\[(.*)\]` and cuts the match out of the lyric wherever it sits, so `a[x]b[y]c` yields the hint
/// `x]b[y` — symbols that its own `IsValidSymbol` then silently drops. We require ONE bracket pair
/// CLOSING the lyric; anything else is not a hint and the lyric goes to the dictionary, i.e. fails
/// LOUDLY. Real material agrees this costs nothing: across the five reference scores on disk there are
/// 34 bare `[hint]` lyrics, 1 typo, and zero multi-bracket or trailing-text forms.
/// ⚠ FULL-WIDTH ［］ (U+FF3B/U+FF3D) count as brackets too — the same tolerance rung S86 added for
/// typographic apostrophes, and for the same reason: a CJK IME emits them by default, and the
/// ASCII-only rule failed SILENTLY. `［k］` fell through to the dictionary, where the punctuation-trim
/// rung reduced it to `k` — a real en.tsv entry — and the note sang the LETTER NAME "kay". Tolerating
/// the glyph turns a silent mis-pronunciation into the phoneme the user meant; the lyric is still
/// stored exactly as typed.
fn phoneme_hint(lyric: &str) -> Option<&str> {
    const OPEN: [char; 2] = ['[', '［'];
    const CLOSE: [char; 2] = [']', '］'];
    let l = lyric.trim_end();
    let inner = l.strip_suffix(CLOSE)?;
    let open = inner.rfind(OPEN)?;
    if inner[..open].contains(OPEN) {
        return None; // more than one bracket pair — ambiguous, so it is not a hint (see above)
    }
    // step over the bracket by its own width — ［ is three bytes, and slicing mid-character panics
    let after = open + inner[open..].chars().next().map_or(1, char::len_utf8);
    let hint = inner[after..].trim();
    (!hint.is_empty()).then_some(hint)
}

/// Per-note run language for chunking — shared by `build_arrays`' assembly AND `compute_note_groups`
/// (score2svc) so grouping can never drift between the cv side and the DAW side. Sustains inherit the
/// previous note's run; rests/breaths attach to the previous run; leading rests take the first run.
pub fn note_run_langs(score: &[ScoreEvt]) -> Vec<Lang> {
    let n = score.len();
    let mut out: Vec<Option<Lang>> = vec![None; n];
    let mut cur: Option<Lang> = None;
    for i in 0..n {
        match token_class(score[i].lyric, score[i].phoneme_set) {
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
    // S90: fold an OpenUtau phonetic hint written in the LYRIC into `phoneme_input` exactly once, here,
    // so every branch below sees a single override notion (and the editor's classify pass, which shares
    // this function, can never disagree with the render about it). An explicit `phoneme_input` still
    // wins — it is the more specific, later user action. The lyric is left untouched; only lookup moves.
    //
    // S91: an ALIAS convention folds in at the same point and for the same reason — one override
    // notion, one code path for the render and the editor. Order of precedence, most specific first:
    //   explicit `phoneme_input` → bracket hint `[...]` → the track's alias convention → dictionary.
    // Only ENGLISH word notes are converted: the conventions ARE English reclists, and a mixed-language
    // track must keep its ja/zh/de/… notes on the dictionary path.
    // A FAILED alias is recorded, never fallen back on: 31-38 % of these aliases are also real en.tsv
    // keys (`ju`, `to`, `E`, `O`), so "try the alias, else look it up" would sing a different WORD.
    let mut alias_failed: Vec<bool> = vec![false; score.len()];
    let alias_owned: Vec<Option<String>> = score
        .iter()
        .enumerate()
        .map(|(i, e)| {
            if e.lang != Lang::En
                || e.phoneme_input.is_some()
                || token_class(e.lyric, e.phoneme_set) != Tok::Word
                || phoneme_hint(e.lyric).is_some()
            {
                return None;
            }
            match alias_phones(e.phoneme_set, e.lyric) {
                Some(Ok(ph)) => Some(ph),
                Some(Err(_sym)) => {
                    alias_failed[i] = true;
                    None
                }
                None => None, // PhonemeSet::Words — the dictionary path, unchanged
            }
        })
        .collect();
    let hinted: Vec<ScoreEvt> = score
        .iter()
        .enumerate()
        .map(|(i, e)| ScoreEvt {
            phoneme_input: e
                .phoneme_input
                .or_else(|| phoneme_hint(e.lyric))
                .or_else(|| alias_owned[i].as_deref()),
            ..e.clone()
        })
        .collect();
    let score = &hinted[..];

    let n = score.len();
    let run_langs = note_run_langs(score);
    let toks: Vec<Tok> = score.iter().map(|e| token_class(e.lyric, e.phoneme_set)).collect();

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
                // S91: an alias that the track's convention cannot read fails LOUDLY with its own
                // CODE, and is NEVER handed to the dictionary (see the fold above — a third of these
                // aliases are also real English words, so a fallback sings a different word silently).
                if alias_failed[i] {
                    if strict {
                        return Err(UtaiError::Inference(format!(
                            "VOCAL_ALIAS: {} {}",
                            evt.phoneme_set.as_str(),
                            evt.lyric
                        )));
                    }
                    carrier = None;
                    out[i] = Some(ResolvedNote { kind: ResolvedKind::Unknown, run_lang, is_sustain: false });
                    i += 1;
                    continue;
                }
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

    // ── S90 ZERO-REGRESSION GATE for the stressless-nucleus switch. The whole argument that changing
    // `WordDict::is_vowel` cannot move a single dictionary syllable boundary is this equality: on the
    // ARPABET the dictionaries actually ship, "carries a CMUdict stress digit" and "this symbol's IPA
    // is nucleus-capable" are the SAME verdict. Hermetic (golden vectors are compiled in); the file-wide
    // walk over all 863018 en.tsv tokens lives in `dictionaries_end_to_end`.
    // It also pins the premise of the bare-`ah`→ə rule: the gate material never spells AH without a digit.
    #[test]
    fn arpabet_stressless_nucleus_is_zero_regression() {
        let mut toks = 0usize;
        for &(lang, word, src, _) in G2P_GOLDEN {
            if lang != "en" {
                continue;
            }
            for t in src.split_whitespace() {
                assert_eq!(
                    t.ends_with(['0', '1', '2']),
                    en_is_nucleus(t),
                    "nucleus verdict drifted on {t:?} (golden word {word:?})"
                );
                assert!(
                    !t.eq_ignore_ascii_case("AH"),
                    "golden carries a BARE AH ({word:?}) — the s90 'no digit ≡ unstressed' rule would move the gate"
                );
                toks += 1;
            }
        }
        assert!(toks > 1500, "golden en token count sanity ({toks})");
    }

    // ── S90 the FEATURE half (this one goes red if the switch is reverted): stressless ARPABET — what
    // OpenUtau phonetic hints and ARPAsing reclists are written in — must resolve a nucleus, and the
    // bare-`ah` reading must be the unstressed one.
    #[test]
    fn stressless_arpabet_resolves_a_nucleus_and_bare_ah_is_schwa() {
        for t in ["ae", "ih", "ah", "er", "ow", "AA", "Uw", "ay"] {
            assert!(en_is_nucleus(t), "{t} must be nucleus-capable without a stress digit");
        }
        for t in ["dh", "hh", "ng", "T", "zh", "sp", "sil"] {
            assert!(!en_is_nucleus(t), "{t} must NOT be nucleus-capable");
        }
        // "no stress digit ≡ unstressed" — the ONE symbol whose IPA splits on stress today
        assert_eq!(convert_arpabet("ah"), "ə");
        assert_eq!(convert_arpabet("AH"), "ə");
        assert_eq!(convert_arpabet("AH0"), "ə");
        assert_eq!(convert_arpabet("ah1"), "ʌ");
        assert_eq!(convert_arpabet("AH2"), "ʌ");
        // a typo near AH is left unmapped so stage2 rejects it LOUDLY instead of guessing a vowel
        assert_eq!(convert_arpabet("AHX"), "AHX");
        assert!(stage2(Lang::En, &["AHX".to_string()]).is_err());
        // and the whole point: a stressless word now SPLITS instead of collapsing into one syllable
        let d = en_fixture();
        let phones: Vec<String> = "k ae n d ah l ih t".split(' ').map(str::to_string).collect();
        let syl = syllabify(&d, &phones);
        assert_eq!(syl.len(), 3, "candlelit must syllabify into 3, got {syl:?}");
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
        ScoreEvt { lyric, note_num: 60, frames: 20, lang, phoneme_input: None, phoneme_set: PhonemeSet::Words }
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
            assert!(!is_silent_token(w, PhonemeSet::Words), "{w:?} must be available as lyric material");
        }
        // ★ the DAW-side note grouping must key off the SAME predicate the cv side resolves with —
        //   `score2cv::classify_lyric` keeps a WIDER rest set (upstream parity port) and using it in
        //   `compute_note_groups` would desync every later group index (review R2/SS-1/GATE-3).
        for lyric in ["R", "r", "", "AP", "ap", "  "] {
            assert!(is_silent_token(lyric, PhonemeSet::Words), "{lyric:?} produces no sung phones");
        }
        for lyric in ["rest", "sil", "pau", "light", "か"] {
            assert!(!is_silent_token(lyric, PhonemeSet::Words), "{lyric:?} is a WORD for the DAW side too");
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

    // ── S90 OpenUtau phonetic hints in the LYRIC (`[p h]` / `word[p h]`) ──
    #[test]
    fn phoneme_hint_parsing_forms() {
        assert_eq!(phoneme_hint("[dh ae dh]"), Some("dh ae dh"), "the bare form real UST files use");
        assert_eq!(phoneme_hint("read[r iy d]"), Some("r iy d"), "OpenUtau's word+hint form");
        assert_eq!(phoneme_hint("  [ ae n ]  "), Some("ae n"), "surrounding + inner space tolerated");
        assert_eq!(phoneme_hint("a[b][c d]"), None, "two bracket pairs are ambiguous → not a hint (upstream would emit garbage here)");
        // …but a stray CLOSING bracket in the WORD part is not a second pair: there is exactly one `[`,
        // so every reading (ours, upstream's greedy, upstream's lazy) yields the same hint. The word
        // part is free text by design — `xyzzy[k ae]` sings `k æ` too. (Review: guard, then un-guard.)
        assert_eq!(phoneme_hint("a]b[c d]"), Some("c d"));
        assert_eq!(phoneme_hint("[DH AE DH]"), Some("DH AE DH"), "case is tolerated (convert_arpabet folds it)");
        // full-width brackets: a CJK IME emits these, and without the rung `［k］` silently sang the
        // LETTER NAME (dictionary `k` = K EY1) instead of the phone. Mixed pairs count too.
        assert_eq!(phoneme_hint("［k ae］"), Some("k ae"));
        assert_eq!(phoneme_hint("［k ae]"), Some("k ae"));
        assert_eq!(phoneme_hint("word［p h］"), Some("p h"));
        assert_eq!(phoneme_hint("［］"), None, "an empty full-width hint is still no hint");
        // multi-byte content must not be sliced mid-character (Rust slices are BYTE ranges)
        assert_eq!(phoneme_hint("こ［k a］"), Some("k a"));
        assert_eq!(phoneme_hint("あ[k a]"), Some("k a"));
        assert_eq!(phoneme_hint("[k aa}"), None, "no closing bracket = not a hint (a real typo in the wild)");
        assert_eq!(phoneme_hint("[]"), None, "empty hint = no hint");
        assert_eq!(phoneme_hint("word[  ]"), None, "whitespace-only hint = no hint");
        assert_eq!(phoneme_hint("[ae n] tail"), None, "the hint must CLOSE the lyric");
        assert_eq!(phoneme_hint("light"), None);
        assert_eq!(phoneme_hint("R"), None);
    }

    #[test]
    fn phoneme_hint_pins_the_phones_and_survives_the_span() {
        let f = fixtures();
        // 1. bare, STRESSLESS hint (what an ARPAsing-bank UST writes) — needs the S90 nucleus rule too
        let r = resolve_score(&[evt("[l ay t]", Lang::En)], &f).unwrap();
        assert_eq!(phones_of(&r[0]), vec!["l", "aɪ", "t"]);
        // 2. the hint WINS over a lyric that is itself a perfectly good dictionary word
        let r = resolve_score(&[evt("light[t r iy]", Lang::En)], &f).unwrap();
        assert_eq!(phones_of(&r[0]), vec!["t", "ɹ", "i"], "the hint, not the word, decides the phones");
        // 3. an explicit phoneme_input still outranks the hint (later, more specific user action)
        let mut e = evt("light[t r iy]", Lang::En);
        e.phoneme_input = Some("F AH1 N");
        let r = resolve_score(&[e], &f).unwrap();
        assert_eq!(phones_of(&r[0]), vec!["f", "ʌ", "n"]);
        // 4. ★ an all-hint lyric must stay a WORD: its "word part" is empty, and an empty lyric is the
        //    REST token — stripping the hint out of the lyric would silence the note (and desync the
        //    DAW-side grouping, which classifies the RAW lyric through `is_silent_token`).
        assert!(!is_silent_token("[ae n]", PhonemeSet::Words));
        assert_eq!(token_class("[ae n]", PhonemeSet::Words) == Tok::Word, true);
        // 5. a multi-syllable hint spreads over its `+` notes exactly like a dictionary word does —
        //    three nuclei that only the S90 stressless rule can see. ⚠ the CUT points come from the
        //    fixture's onset set: on the SHIPPED dictionary the same word cuts k æ | n d ə | l ɪ t,
        //    because `N D` is a legal onset there (imported from foreign proper names — the S86 audit's
        //    `en-onset-pollution`, still open). The syllable COUNT is the invariant this test owns.
        let span = [evt("[k ae n d ah l ih t]", Lang::En), evt("+", Lang::En), evt("+", Lang::En)];
        let r = resolve_score(&span, &f).unwrap();
        assert_eq!(phones_of(&r[0]), vec!["k", "æ", "n"]);
        assert_eq!(phones_of(&r[1]), vec!["d", "ə"], "bare `ah` reads as the unstressed vowel");
        assert_eq!(phones_of(&r[2]), vec!["l", "ɪ", "t"]);
        // 6. malformed hint → the lyric goes to the dictionary and fails LOUDLY (never a silent guess)
        assert!(resolve_score(&[evt("[k aa}", Lang::En)], &f).is_err());
        assert!(matches!(classify_score(&[evt("[k aa}", Lang::En)], &f).unwrap()[0], LyricClass::Unknown));
        // 6b. an EN override written in RAW IPA is off-contract (§3.7 = traditional phones) but used to
        //     resolve by passthrough — the S90 case normalization must not turn it into a silent OOV
        //     (`aɪ`.to_ascii_uppercase() would be `Aɪ`, which interns to nothing).
        let mut e = evt("xxxx", Lang::En);
        e.phoneme_input = Some("ɹ ə f aɪ n d");
        assert_eq!(phones_of(&resolve_score(&[e], &f).unwrap()[0]), vec!["ɹ", "ə", "f", "aɪ", "n", "d"]);
        // 7. other languages get the same one notion: ja hints are raw vocab IPA (§3.7), zh a syllable
        let r = resolve_score(&[evt("[k a]", Lang::Ja)], &f).unwrap();
        assert_eq!(phones_of(&r[0]), vec!["k", "a"]);
        let r = resolve_score(&[evt("长[chang]", Lang::Zh)], &f).unwrap();
        assert_eq!(phones_of(&r[0]), vec!["ʈʂʰ", "ɑŋ"]);
    }

    // ── S90 review major #2: NOTHING bracket-shaped may be quietly rewritten into a different word ──
    // The lookup ladder trims punctuation off a lyric before consulting the dictionary, and the shipped
    // en.tsv has an entry for every single letter (`k` = "kay") and for plenty of two-letter forms
    // (`dr` = "drive", `mr` = "mister"). Once brackets MEAN something, that rescue turns a typo into a
    // silent mis-pronunciation: `[dr` (missing `]`) sang "drive", `[k].` sang "kay", `[[k]]` sang "kay"
    // — no red mark, no error, straight past the "either your phones or a red note" promise the feature
    // is documented with. The parser-level test above cannot see this; only a resolve-level one can.
    #[test]
    fn a_bracket_lyric_never_silently_sings_a_different_word() {
        let f = Fixtures {
            zh: zh_fixture(),
            de: de_fixture(),
            // mirrors the real dictionary's letter-name and abbreviation entries
            en: WordDict::from_tsv(Lang::En, "k\tK EY1\ndr\tD R AY1 V\nchorus\tK AO1 R AH0 S\n"),
        };
        for lyric in [
            "[dr",     // the closing bracket never got typed
            "[k",      //
            "[k].",    // well-formed but with trailing punctuation
            "[[k]]",   // doubled
            "\"[k]\"", // quoted
            "[k]!",    //
            "[]",      // empty
            "[chorus]xyz",
        ] {
            let r = classify_score(&[evt(lyric, Lang::En)], &f).unwrap();
            assert!(
                matches!(r[0], LyricClass::Unknown),
                "{lyric:?} must be a LOUD OOV, not a quietly substituted word (got {:?})",
                r[0]
            );
            assert!(resolve_score(&[evt(lyric, Lang::En)], &f).is_err(), "{lyric:?} must fail the render");
        }
        // …while the well-formed forms still resolve, and a plain word is still trimmed as before
        assert_eq!(phones_of(&resolve_score(&[evt("[k]", Lang::En)], &f).unwrap()[0]), vec!["k"]);
        assert_eq!(
            phones_of(&resolve_score(&[evt("chorus,", Lang::En)], &f).unwrap()[0]),
            vec!["k", "ɔ", "ɹ", "ə", "s"],
            "ordinary punctuation trimming is untouched"
        );
        // A bracket that holds a WORD instead of phones is a loud OOV too — `[Chorus]`/`[Verse 1]`
        // section markers pasted from a lyric sheet are not lyrics, and singing them (which the
        // pre-S90 trim rung did) is worse than showing the user a red note. Documented in §5.4.
        assert!(resolve_score(&[evt("[chorus]", Lang::En)], &f).is_err());
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
        use super::super::score2cv::{build_arrays_daw, chunk_at_sp, ArticulationTiming};
        let f = fixtures();
        let score = [evt("长", Lang::Zh), evt("light", Lang::En)];
        let arr = build_arrays_daw(&score, &f, ArticulationTiming::Auto).unwrap();
        let chunks = chunk_at_sp(&arr, 400);
        assert_eq!(chunks.len(), 2, "language change forces a cut even under max_frames");
        assert_eq!(chunks[0].lang_id, Lang::Zh.id());
        assert_eq!(chunks[1].lang_id, Lang::En.id());
        assert!(!chunks[0].hard_seam, "first chunk has no leading seam");
        assert!(chunks[1].hard_seam, "mid-voiced language cut → hard seam (micro-fade)");
        let score2 = [evt("长", Lang::Zh), evt("R", Lang::Zh), evt("light", Lang::En)];
        let arr2 = build_arrays_daw(&score2, &f, ArticulationTiming::Auto).unwrap();
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
        use super::super::score2cv::ArticulationTiming;
        let arr = super::super::score2cv::build_arrays_daw(&score, &f, ArticulationTiming::Auto).unwrap();
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

        // ── S90: the SHIPPED-dictionary half of the stressless-nucleus zero-regression argument. The
        // golden vectors sample words; this walks every token of every line of en.tsv, which is what
        // `WordDict::from_tsv` actually feeds the legal-onset pass. If the two verdicts ever diverge
        // on a real token, the blast radius is not "stressless input" — it is every English syllable
        // boundary in the app.
        let en_tsv = std::fs::read_to_string(root.join("../data/dictionaries/en.tsv")).expect("en.tsv");
        let (mut toks, mut bare_ah) = (0usize, 0usize);
        let mut distinct: HashSet<&str> = HashSet::new();
        for line in en_tsv.lines() {
            let Some((_w, phones)) = line.split_once('\t') else { continue };
            for t in phones.split_whitespace() {
                assert_eq!(
                    t.ends_with(['0', '1', '2']),
                    en_is_nucleus(t),
                    "en.tsv token {t:?}: the stress-digit and nucleus verdicts disagree"
                );
                if t.eq_ignore_ascii_case("AH") {
                    bare_ah += 1;
                }
                distinct.insert(t);
                toks += 1;
            }
        }
        assert_eq!(bare_ah, 0, "en.tsv now spells a BARE AH — the s90 'no digit ≡ unstressed' rule would move those words");
        assert!(toks > 500_000 && distinct.len() > 50, "en.tsv walk looks truncated ({toks} tokens)");
        eprintln!("[g2p-e2e] en.tsv nucleus equivalence: {toks} tokens / {} distinct, 0 disagreements", distinct.len());
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
                        // S90: show the HINT when the lyric carries one — otherwise this line reports a
                        // dictionary miss for a note that resolves perfectly well, which reads as a bug.
                        if let Some(h) = phoneme_hint(head) {
                            println!("    hint: [{h}]");
                        }
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

    // ─── S91 UTAU alias conventions (queue 5c) — the resolve_core integration ────────────────────
    fn alias_evt(lyric: &str, set: PhonemeSet) -> ScoreEvt<'_> {
        ScoreEvt { lyric, note_num: 60, frames: 20, lang: Lang::En, phoneme_input: None, phoneme_set: set }
    }

    /// ★★ THE regression this feature could most easily ship: an alias that is ALSO a real dictionary
    /// word must sing the ALIAS, never the word. Measured on the reference scores, 31 % of the X-SAMPA
    /// aliases and 38 % of the VCCV ones are real `en.tsv` keys, so a "try the alias, else look it up"
    /// design would sing a DIFFERENT WORD on a third of the score, silently — the S90 `[dr]`→"drive"
    /// pathology at scale. Every lyric below is in the fixture dictionary AND a legal alias.
    #[test]
    fn alias_never_falls_back_to_the_dictionary() {
        let f = fixtures();
        let sing = |lyric: &str, set: PhonemeSet| -> Vec<&'static str> {
            phones_of(&resolve_score(&[alias_evt(lyric, set)], &f).unwrap()[0])
        };
        // the word readings, for contrast (these are what the dictionary path gives)
        assert_eq!(sing("two", PhonemeSet::Words), vec!["t", "u"]);
        assert_eq!(sing("love", PhonemeSet::Words), vec!["l", "ʌ", "v"]);
        assert_eq!(sing("fun", PhonemeSet::Words), vec!["f", "ʌ", "n"]);
        // …and the alias readings of the SAME strings
        assert_eq!(sing("two", PhonemeSet::Xsampa), vec!["t", "w", "oʊ"]);
        assert_eq!(sing("love", PhonemeSet::Xsampa), vec!["l", "oʊ", "v", "ɛ"]);
        assert_eq!(sing("fun", PhonemeSet::Vccv), vec!["f", "ə", "n"]);
    }

    /// An alias the convention cannot read fails LOUDLY with its OWN code (never the dictionary's
    /// `VOCAL_OOV`, whose message sends the user to "check the lyric or the language" — the wrong
    /// advice here), and the editor's lenient pass marks exactly that note.
    #[test]
    fn alias_failure_is_loud_and_marks_only_that_note() {
        let f = fixtures();
        let bad = [alias_evt("light", PhonemeSet::Xsampa), alias_evt("zq", PhonemeSet::Xsampa)];
        let err = resolve_score(&bad, &f).unwrap_err().to_string();
        assert!(err.contains("VOCAL_ALIAS: xsampa zq"), "own CODE + convention + lyric: {err}");
        let lenient = classify_score(&bad, &f).unwrap();
        assert!(matches!(lenient[0], LyricClass::Phones { .. }), "the readable alias still resolves");
        assert!(matches!(lenient[1], LyricClass::Unknown), "…and only the bad one is marked");
    }

    /// The precedence ladder, and the things an alias convention must NOT capture.
    #[test]
    fn alias_precedence_and_what_it_leaves_alone() {
        let f = fixtures();
        let one = |e: ScoreEvt| -> ResolvedNote { resolve_score(&[e], &f).unwrap().remove(0) };
        // an explicit phoneme_input still wins (the more specific, later user action)
        let mut e = alias_evt("ju", PhonemeSet::Xsampa);
        e.phoneme_input = Some("K AE1 T");
        assert_eq!(phones_of(&one(e)), vec!["k", "æ", "t"]);
        // a bracket hint wins over the convention (it names the phones outright)
        assert_eq!(phones_of(&one(alias_evt("[k ae t]", PhonemeSet::Vccv))), vec!["k", "æ", "t"]);
        // reserved tokens are checked BEFORE any of this and are unaffected
        assert!(matches!(one(alias_evt("R", PhonemeSet::Vccv)).kind, ResolvedKind::Rest));
        assert!(matches!(one(alias_evt("AP", PhonemeSet::Xsampa)).kind, ResolvedKind::Breath));
        // a NON-English note on an alias track keeps the dictionary path — the conventions are
        // English reclists, and a mixed-language track must not have its ja/zh notes reinterpreted
        let mut ja = alias_evt("か", PhonemeSet::Vccv);
        ja.lang = Lang::Ja;
        assert_eq!(phones_of(&one(ja)), vec!["k", "a"]);
        // and `Words` is the pre-S91 behaviour, byte for byte
        assert_eq!(
            phones_of(&one(alias_evt("light", PhonemeSet::Words))),
            phones_of(&one(evt("light", Lang::En)))
        );
    }

    /// ★ S91 review MAJOR: lowercase `ap` is an ordinary VC alias (`a`+`p` — the coda of *stop* /
    /// *top*), and swallowing it as a breath rendered an audible INHALE mid-word with no error and no
    /// red mark. On an alias track it must be a WORD — and, critically, the DAW-side grouping
    /// predicate has to agree, or the cv side sings a phone the note grouping thinks is silence and
    /// every later group index shifts (the S86 desync this predicate was extracted to prevent).
    #[test]
    fn alias_frees_the_lowercase_breath_token_and_both_sides_agree() {
        let f = fixtures();
        // words track: unchanged, `ap` is still a breath
        assert!(is_silent_token("ap", PhonemeSet::Words));
        assert!(matches!(
            resolve_score(&[alias_evt("ap", PhonemeSet::Words)], &f).unwrap()[0].kind,
            ResolvedKind::Breath
        ));
        // alias track: it is the alias, and the two sides say the same thing
        for set in [PhonemeSet::Xsampa, PhonemeSet::Vccv] {
            assert!(!is_silent_token("ap", set), "{set:?}");
            assert_eq!(phones_of(&resolve_score(&[alias_evt("ap", set)], &f).unwrap()[0]), vec!["p"]);
        }
        // …and the canonical spellings are untouched everywhere, in BOTH predicates
        for set in [PhonemeSet::Words, PhonemeSet::Arpasing, PhonemeSet::Xsampa, PhonemeSet::Vccv] {
            for (lyric, silent) in [("R", true), ("r", true), ("", true), ("AP", true), ("light", false)] {
                assert_eq!(is_silent_token(lyric, set), silent, "{lyric:?} @ {set:?}");
                let kind = &resolve_score(&[alias_evt(lyric, PhonemeSet::Words)], &f).unwrap()[0].kind;
                assert_eq!(matches!(kind, ResolvedKind::Rest | ResolvedKind::Breath), silent);
            }
        }
        // `r` stays a REST on purpose: real banks use it that way (the two `r` notes in
        // `duvet - vocal.ust` carry only Length/Lyric/NoteNum, the shape of that file's `R` rests),
        // and a genuine standalone /ɹ/ still has three spellings.
        for spelling in ["-r", "r-", "_r"] {
            assert_eq!(phones_of(&resolve_score(&[alias_evt(spelling, PhonemeSet::Vccv)], &f).unwrap()[0]), vec!["ɹ"]);
        }
    }

    /// A sustain after an alias note re-emits that note's own carrier — the same rule a word note
    /// gets, so an alias score can use `-`/`+` for held notes.
    #[test]
    fn alias_note_carries_its_sustain() {
        let f = fixtures();
        let score = [alias_evt("ju", PhonemeSet::Xsampa), alias_evt("-", PhonemeSet::Xsampa)];
        let r = resolve_score(&score, &f).unwrap();
        assert_eq!(phones_of(&r[0]), vec!["j", "u"]);
        assert_eq!(phones_of(&r[1]), vec!["u"], "the hold re-emits the alias's own nucleus");
        assert!(r[1].is_sustain);
    }
}
