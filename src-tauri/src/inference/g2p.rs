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
/// AH1+AH2 8022 (**7.9×**; S90 counts — AH0 is 63260 after the S94 -en schwa regeneration, same
/// ratio same argument); the errors are asymmetric (ʌ read as ə is a mild centralization, ə read as
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

    /// The S86 tolerance ladder ONLY (case/quotes/punct/ß) — no plural reconstruction.
    /// The fragment-merge TRIGGER must ask THIS question (review S95R-2): with the plural rung
    /// in the trigger, any s-final fragment that parses as base+s stops counting as OOV —
    /// "mes" reads as me+Z, the merge never fires, and mes|sage sings "meez sage" with no red
    /// mark instead of joining to "message". A fragment that can still become a whole word by
    /// merging must get that chance BEFORE the last-resort plural reading.
    fn lookup_faithful(&self, word: &str) -> Option<Vec<String>> {
        lookup_candidates(word)
            .iter()
            .find_map(|k| self.map.get(k))
            .map(|p| p.split_whitespace().map(str::to_string).collect())
    }

    pub fn lookup(&self, word: &str) -> Option<Vec<String>> {
        if let Some(p) = self.lookup_faithful(word) {
            return Some(p);
        }
        // S104: FR/IT elision rung — `l'amour` = `l'` + `amour`. Before it, every productive
        // elision was VOCAL_OOV, i.e. it ABORTED THE WHOLE SEGMENT (the strict resolve returns
        // Err), and elision is not an edge case in these languages — French orthography REQUIRES
        // it before a vowel, so a lyric line carrying l'/j'/t'/m'/s'/n'/d'/c'/qu' + a vowel-initial
        // word simply could not be sung unless that exact form happened to be one of the 209
        // lexicalised `clitic+X` rows fr.tsv inherited from MFA.
        if matches!(self.lang, Lang::Fr | Lang::It) {
            if let Some(p) = lookup_candidates(word).iter().find_map(|k| self.elision(k)) {
                return Some(p);
            }
        }
        // S95: EN regular plural/possessive rung, strictly LAST — cmudict omits many regular
        // plurals ("dears" while "dear" ships), and before this rung such a lyric hard-aborted
        // the render as VOCAL_OOV. EN only: fr plural -s is SILENT, de/es/it inflect on their
        // own rules — appending /s z ɪz/ there would sing another language's morphology.
        // (Defense in depth, verified by mutation: even WITHOUT this gate, stage2 rejects the
        // uppercase ARPABET suffix against every non-EN vocab, so those words stay OOV either
        // way — the gate states the intent instead of leaning on that coincidence.)
        if self.lang == Lang::En {
            lookup_candidates(word).iter().find_map(|k| self.en_plural(k))
        } else {
            None
        }
    }

    /// S104 — FR/IT elision: split a lookup key at its FIRST internal apostrophe and concatenate
    /// `<proclitic'>` + `<rest>`. Three conditions, and all three are the GRAMMAR of elision rather
    /// than heuristics tuned to a sample:
    ///   1. the key has an apostrophe that is neither first nor last;
    ///   2. the part up to and including it is itself a dictionary key — i.e. an attested
    ///      proclitic. fr.tsv carries `l' j' t' m' s' n' d' c' ç' p' z' qu'` (the 12 vowel-less rows
    ///      the S86 audit flagged as a hazard — they are the ENABLING data here) plus `jusqu'`,
    ///      `lorsqu'`, `aujourd'`…; it.tsv carries `l' c' d' all' dell' un'`…;
    ///   3. ★the rest begins with a VOWEL LETTER or mute `h`. Elision happens BECAUSE the next word
    ///      is vowel-initial; a consonant-initial right half means this is not an elision at all.
    ///
    /// Why the SPELLING test and not "the right half's first PHONE is a vowel" — measured, all three
    /// variants, on the shipped dictionaries (`TESTING/s104_dict_recheck/guard_compare.py`):
    ///   phone-vowel        fr 208/222 = 93.7% · **misses l'oiseau / l'ouest / l'huile / l'hiérarchie**
    ///                      (⟨oi ou hui hi⟩ surface as the glides /w ɥ j/, which are not nuclei)
    ///   phone-vowel+glide  fr 208/223 = 93.3% · admits `l'yacht` / `l'watt` / `l'week-end`, where
    ///                      French does NOT elide — and it needs a hand-written glide table
    ///   **spelling**       **fr 204/216 = 94.4% · it 1446/1446 = 100% · zero non-elision leaks in it**
    /// The letter test is also the rule as it is actually taught (elide before a vowel letter or
    /// mute h; ⟨y⟩ and foreign ⟨w⟩ block it), so ⟨y⟩ is deliberately NOT elidable here.
    ///
    /// The percentages above are a VALIDATION, not a prediction: they replay the rung against the
    /// `clitic+X` rows the dictionary already ships, where the shipped reading is the known-correct
    /// answer. In production those rows are hit by `lookup_faithful` first and the rung never runs.
    /// The residual disagreements are (a) English junk rows (`a's`, `edward's`, `rock'n`), (b) MFA's
    /// cross-boundary narrowing (`l'initiative` = `ʎ i…`, where the GTSinger French word-level
    /// annotation writes the composed `l i` — the clitic is a word boundary), and (c) the e-caduc
    /// (`l'oeuvre` keeps a final `ə`) — see the queue's fr e-caduc item.
    ///
    /// ⚠ KNOWN IMPRECISION, stated so nobody reads this as more than it is: French *h aspiré*
    /// (`le héros`, `le hasard`) blocks elision lexically, and a lookup table cannot know which
    /// ⟨h⟩ is which, so `l'héros` composes. It sings what the author typed rather than aborting the
    /// segment, which is the trade this rung exists to make.
    ///
    /// ⚠ SCOPE: fr/it only. es.tsv has **0** apostrophe keys and de.tsv has 1, so the rung is a
    /// structural no-op there; EN's 7478 apostrophe keys are possessives and contractions
    /// (`don't`, `john's`) where splitting is simply wrong.
    ///
    /// This never changes an answer the dictionary already had: it runs only on the branch where
    /// `lookup_faithful` returned None, so it can only turn a hard `VOCAL_OOV` abort into a reading
    /// (`s104_elision_never_overrides_a_faithful_hit` pins that rather than leaving it to construction).
    fn elision(&self, key: &str) -> Option<Vec<String>> {
        let i = key.find('\'')?;
        let (left, right) = (key.get(..=i)?, key.get(i + 1..)?);
        if left.len() == 1 || right.is_empty() {
            return None; // leading apostrophe, or nothing to the right
        }
        if !right.chars().next().is_some_and(elidable_head) {
            return None;
        }
        let lp = self.map.get(left)?;
        let rp = self.map.get(right)?;
        Some(lp.split_whitespace().chain(rp.split_whitespace()).map(str::to_string).collect())
    }

    /// One `-'s` / `-s` / `-es` strip of an EN lookup key whose base IS in the dictionary →
    /// base phones + the suffix cmudict itself uses: sibilant-final → `IH0 Z` (roses), voiceless-
    /// final → `S` (lights), else → `Z` (dears). Tried per tolerance-ladder candidate so case/
    /// quote/punct rungs keep working. A base under 3 chars is refused (review S95R-3): en.tsv's
    /// 1-2 char rows are largely letter names and cmudict ABBREVIATION EXPANSIONS (dr=drive,
    /// st=street, me/to/be…), so short bases turn mis-typed fragments into silent whole words —
    /// "mes" would sing "meez", "dres" would sing "drives". The refused recall (djs, tvs) is
    /// tiny and stays loud instead of guessing.
    fn en_plural(&self, key: &str) -> Option<Vec<String>> {
        for base in [key.strip_suffix("'s"), key.strip_suffix('s'), key.strip_suffix("es")] {
            let Some(b) = base else { continue };
            if b.chars().count() < 3 {
                continue;
            }
            let Some(p) = self.map.get(b) else { continue };
            let mut phones: Vec<String> = p.split_whitespace().map(str::to_string).collect();
            let last = phones.last().map(|s| s.trim_end_matches(['0', '1', '2']).to_string())?;
            let suffix: &[&str] = match last.as_str() {
                "S" | "Z" | "SH" | "ZH" | "CH" | "JH" => &["IH0", "Z"],
                "P" | "T" | "K" | "F" | "TH" => &["S"],
                _ => &["Z"],
            };
            phones.extend(suffix.iter().map(|s| s.to_string()));
            return Some(phones);
        }
        None
    }

    /// Parse the canonical `word<TAB>phones` TSV. First-seen pronunciation wins (the build emits the
    /// primary first); every word's initial consonant cluster VOTES for the legal-onset set, and for
    /// EN a multi-consonant cluster needs `EN_ONSET_MIN_VOTES` votes to be admitted (see the const).
    pub fn from_tsv(lang: Lang, tsv: &str) -> WordDict {
        let min_votes = match lang {
            Lang::En => EN_ONSET_MIN_VOTES,
            // de/fr/es/it are NOT gated by a vote count — see `WEST_ONSET_KEEP` below. The count is
            // left at 1 so the vote table stays a pure observation and the keep list is the only
            // filter; `s105_west_onset_gate` pins that reading.
            _ => 1,
        };
        Self::from_tsv_min_votes(lang, tsv, min_votes)
    }

    fn from_tsv_min_votes(lang: Lang, tsv: &str, min_votes: u32) -> WordDict {
        let vowels: HashSet<&'static str> = match lang {
            Lang::De => gt::MFA_VOWELS_DE.iter().copied().collect(),
            Lang::Fr => gt::MFA_VOWELS_FR.iter().copied().collect(),
            Lang::Es => gt::MFA_VOWELS_ES.iter().copied().collect(),
            Lang::It => gt::MFA_VOWELS_IT.iter().copied().collect(),
            _ => HashSet::new(),
        };
        let mut dict = WordDict { lang, map: HashMap::new(), onsets: HashSet::new(), vowels };
        dict.onsets.insert(String::new()); // the empty onset is always legal (V-initial syllables)
        let mut votes: HashMap<String, u32> = HashMap::new();
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
                *votes.entry(toks[..vi].join(" ")).or_default() += 1;
            }
        }
        for (cluster, n) in votes {
            // The vote threshold disciplines CLUSTERS only. A lone consonant is never gated: all 23
            // observed EN singles are ordinary English onsets (weakest = DH with 71 votes — the/this
            // closed class; review S94R-3 corrected an earlier "ZH 94" claim here), and dropping one
            // would glue every intervocalic instance of that consonant to the previous syllable — a
            // category of damage no attestation count justifies.
            if cluster.contains(' ') && !onset_admitted(lang, &cluster, n, min_votes) {
                continue;
            }
            dict.onsets.insert(cluster);
        }
        dict
    }
}

/// S94 (dictionary re-audit, tier-1): an EN onset CLUSTER must be attested by at least this many
/// word-initial dictionary entries before it may drive intervocalic cuts in `syllabify`.
///
/// Why: `from_tsv` admits the prefix of EVERY line with zero filtering, so one loanword/proper-noun
/// entry rewrites how thousands of ordinary words split across notes. Census over the shipped
/// en.tsv (S94, blind-verified twice): 149 distinct onsets, 126 multi-consonant; `N D` is supported
/// by ONE entry (n'dour) yet recuts 2811 words (candle → K AE1 | N D AH0 L, kidney → K IH1 | D N IY0,
/// sadness, midnight, abandon, window…); `Z B` (4 votes, zbigniew) breaks husband, `M L` (5, mladic)
/// breaks aimless, `L W` (5, luo/lwin) breaks always — the S86 `always|+ → ɔ | l w eɪ z` finding.
///
/// Why SIX (honest version — review S94R-2 broke the first draft's "clean gap" claim): 6 keeps the
/// weakest NATIVE clusters (S P Y spew and TH W thwart sit exactly at 6, S K Y skew at 9) while
/// removing the worst 60. It is NOT a clean native/foreign separator: S K L (5 votes) counts real
/// words (sclerosis/scleroderma) among its voters — dropping it still nets out (disclaim/exclusive
/// improve) — and several foreign-only families clear 6 on name volume alone; those go through
/// `EN_ONSET_DROP` below with per-cluster verdicts instead of a blunter threshold.
///
/// ★KNOWN COST, accepted deliberately (review S94-DATA-1, then re-judged word-by-word): dropping
/// the sub-6 C+glide clusters (V W, M W, F W, ZH W, TH Y, S Y…) miscuts ~90 rare French/loan words
/// (reservoir → Z AH0 V | W AA2 R, bourgeois, armoire, bivouac, Matthew-as-M AE1 TH|Y UW0 is fine).
/// Restoring any of them was measured to break far commoner words the drop fixed: V W would re-break
/// driveway, M W teamwork/dreamworks, F W halfway/safeway, L W always, ZH W visualize, TH Y matthew.
/// Net direction favors the drop everywhere; do NOT add a keep-list without re-running that table.
///
/// ⛔ Deliberately KEPT (S92: their cuts are correct — do not "clean" them): ZH (94, asia/azure),
/// K N (14, technique), the schm-/schn-/schl-/schw- loan families.
/// (`T S` sat on this list until S102 re-judged it on evidence S92 did not have — see EN_ONSET_DROP.)
///
/// A cluster losing its vote NEVER hurts the words that voted it in: word-initial phones always stay
/// in the first syllable; the onset set only decides INTERVOCALIC cuts.
const EN_ONSET_MIN_VOTES: u32 = 6;

/// S94 review follow-up (S94R-2/S94-DATA-5): foreign-supported clusters that clear the vote gate on
/// proper-noun volume alone, judged cluster-by-cluster (every English word their removal re-cuts was
/// eyeballed; the improvement lists below are exhaustive samples, not picks):
///   S R  (12, sri/srebrenica)     → classroom/crossroads/disrespect/disregard now cut at the seam
///   M R  (11, mraz/mroczek)       → armrest, comrade, camry, amritsar
///   Z L  (11, zlata/zloty)        → beasley/beardsley name family, ceaselessly (cease|less|ly)
///   V R  (14, vrabel/vranitzky)   → average(d/s), beverages, chevrolet, avril
///   K V  (9, kvam/kvetch)         → the mc-V name family (mcveigh/mcvay/macvicar), bankverein
///   SH T (6, shtick/schtick)      → ashton, hashtag, washtub, rushton, ishtar (German -stadt names
///                                    get a worse cut — rare, accepted)
///   HH R (13, hraw-class)         → blast = 1 word (warhol's); hygiene
///   HH L (6, hlad-class)          → blast = 0 words; hygiene
///   T S  (27, tsar/tsunami/matsu-) → S102; the one entry here that OVERTURNS a verdict, so its
///                                    evidence is written out in full below.
/// Word-initial voters themselves are untouched (kvetch/shtick still sing whole), same invariant as
/// the threshold. de/fr/es/it deliberately not curated here — S86#10 owns their verdicts.
///
/// ★S102 — why `T S` moved from the KEPT list to this one (queue (a3-2); it was carried as
/// 【未验证】 because two criteria disagreed, and they disagreed because they were looking at
/// DIFFERENT WORDS. Dropping it moves 464 word types, in two opposed families).
/// S92's case for keeping it was one eyeballed word: "ZH and T S cut correctly — versions /
/// massagers / pizzeria". `ZH` is a LONE consonant and is structurally never gated (see the
/// paragraph above this list), so `pizzeria` WAS the entire case. What S102 measured against it:
///  · all 27 word-initial voters are loans (tsar, tsunami, tsetse, tsu-/matsu- names, csaszar,
///    zeitgeist) — NO native English word begins with /ts/. That is the definition of this list.
///  · 77 of the 464 carry mechanical evidence FOR the drop, out of en.tsv itself: the word splits
///    into two dictionary words whose phones concatenate to exactly its own, at exactly the dropped
///    cut — albert+son, out+side, best+seller, white+side, pant+suit, sight+see, it+self,
///    august+son, short+sighted, west+side, sweat+suit… Only FOUR carry evidence the other way
///    (ma+tsuda, ma+tsui, mi+tsui, tse+tse).
///  · EXPOSURE, and it is weak — recorded at its true strength because an adversarial pass demoted
///    it from "what anyone actually sings settles it", which is what the first draft said. Across
///    GTSinger English exactly ONE of the 464 is ever sung — `outside`, 25 times — and it is on the
///    drop side. But that corpus is 65319 tokens over only 4827 items / 1156 DISTINCT lyric lines,
///    vocabulary 1602 types = 1.24% of en.tsv, and those 25 tokens are 6 lyric lines. A random
///    464-word draw would be expected to hit ~5.7. So this says "neither family is sung often enough
///    for the knife to be loud either way", NOT "the drop side wins".
///    ⛔ Do NOT re-add "and the keep side has zero occurrences there" as corroboration: with ONE hit
///    across 464 words, zero on the keep side is the MODAL outcome under no asymmetry whatsoever
///    (P = 0.52-0.90). The instrument itself is alive — the same measurement over `V R`'s 92 capture
///    words returns 7 lyric types / 94 tokens — which is precisely why its near-silence here is a
///    statement about coverage, not about the verdict.
///    ⇒ THE VERDICT RESTS ON THE TWO en.tsv-INTERNAL LINES ABOVE (27/27 loan voters; 77 vs 4), not
///      on this one.
///  · `T S` is TWO IPA phones on the English path — the real affricate token t͡s lives in it.tsv
///    (1316 lines) and en.tsv never emits it — so keeping it preserved no affricate. It only moved
///    a /t/ onto the next note.
/// ⚠ RECORDED NEGATIVE so nobody re-runs it: the upstream note-boundary surface (GTSinger's own
///    note annotation, the S97/S98 truth surface) CANNOT judge this cluster. Say it PRECISELY,
///    because the first draft of this note did not and an adversarial pass caught it: "English has
///    ZERO word-internal `T S`" is FALSE — there are 27 word-internal intervocalic ones (outside
///    x25, outskirts x2). What is true, and is what empties the surface, is that ALL 27 sit entirely
///    inside ONE note: not a single one is CROSSED by a note boundary, so none of them can say which
///    side of the cut the /t/ belongs on. The 201 npz instances are all CROSS-WORD, where a
///    word-final /t/ is structurally unable to open a note; counting those would be S98's "a thing
///    that could not have moved is not a control".
/// ★KNOWN COST, accepted deliberately (same shape as the SH T / German -stadt cost above): the
///    Japanese <tsu> family (matsumoto, mitsubishi, fujitsu, atsushi — ~45 types), the Slavic/Greek
///    <ts> names (yeltsin, tutsi, vorontsov, mitsotakis), and `tsetse` — the one ordinary English
///    word on that side — now put /t/ in the coda, which their source languages do not want. A
///    word-level compound-seam exception would serve both families and is NOT built here for a
///    measured reason: the general seam rule is wrong. Over the whole dictionary 10191 of 27743
///    live seams disagree with maximal onset, and the disagreements include SUFFIX seams English
///    genuinely does resyllabify (aachen+er, mast+er → mas-ter). Restricting it to true compounds
///    needs its own round with its own truth surface.
const EN_ONSET_DROP: &[&str] =
    &["S R", "M R", "Z L", "V R", "K V", "SH T", "HH R", "HH L", "T S"];

/// S101 blast containment for the fr D6 mirror — NOT a general curation of the French onset set.
///
/// The D6 mirror (MBS2H `build_dictionaries.py`) rewrites `ɲ`→`n` before `i`, which renames three
/// word-initial clusters rather than inventing them: the SAME three entries vote before and after —
/// `d ɲ`→`d n` (dniepr / dnipro / dnipropetrovsk, 3 votes), `s f ɲ`→`s f n` (spheniscidae, 1),
/// `z v ɲ`→`z v n` (zvenigorod, 1). With `min_votes = 1` for fr, each is instantly legal.
///
/// `d n` then captures a family it has no business in. Measured over the shipped fr.tsv, admitting it
/// re-cuts **45** word types whose own phones never changed — sidney/sydney → `s i | d n ɛ`,
/// kidnappe/kidnapping, wednesday, midnight, madness, ordnance, gardner, dreadnought, cadena,
/// pasadena, grodno… — i.e. it moves /d/ out of the coda where every one of those loanwords wants it.
/// **This is literally the same river S94 already judged on the English side**: `EN_ONSET_DROP`'s
/// gate pins `kidney → K IH1 D | N IY0` with the comment "was K IH1 | D N IY0 (dniester's D N)".
/// Same cluster, same source language, same damage family — so the verdict is not a new opinion.
///
/// Dropping costs the voters nothing, by the invariant stated above `EN_ONSET_MIN_VOTES`:
/// word-initial phones always stay in the first syllable, the onset set only decides INTERVOCALIC
/// cuts. `dniepr` still sings `dn-` whole. `s f n` / `z v n` capture 0 words today and are listed
/// only so the knife's invariant is exact and testable: **the D6 mirror changes phone identity and
/// creates no new legal onset.**
///
/// ⛔ The three clusters D6 REMOVES (`s ɲ` 5 votes sneek/snider/sniffer/snyder, `ɡ ɲ` gnifetti,
/// `ʃ ɲ` schnitzler) are deliberately allowed to go: they were legal only because of the upstream
/// `ni`→`ɲi` bug, and losing them moves 16 words the other way — `bosnien` `b ɔ s | ɲ ɛ̃` instead of
/// `b ɔ | s ɲ ɛ̃`, `baguenier` `b a ɡ | ɲ e` — which is where French phonotactics wants the cut.
///
/// ⚠ fr `min_votes` is still 1 and this list does NOT change that. The general four-language vote
/// gate remains untested and unowned by this round (see `EN_ONSET_MIN_VOTES`, S86#10).
const FR_ONSET_DROP: &[&str] = &["d n", "s f n", "z v n"];

/// S105 (queue §C2) — the FOUR-LANGUAGE onset gate. Until now de/fr/es/it had NONE: `from_tsv`
/// admitted the word-initial cluster of every line with zero filtering, so ONE loanword or proper
/// noun made a cluster legal and rewrote how thousands of ordinary words split across notes.
/// Measured on the shipped dictionaries: it `n t` is attested by a single entry (`N-terminali`) and
/// captures **5825** word types (`fronte` → `f r o | n t e`); es `n d̪` by one (`ndocciata`) → 3820
/// (`abandera` → `a | β a | n d̪ e | ɾ a`); de `t l` by two exonyms (tlingit / tlaxcalteken) → 2182
/// (`deutlich` → `d ɔʏ | t l ɪ ç`, `atlas` → `a | t l a s`).
///
/// ★ THE CRITERION IS NOT THE VOTE COUNT — and it is not "can this cluster begin a word" either. It
///   is: **when the cluster sits between two vowels INSIDE a word, does the language put the WHOLE
///   of it into the second syllable?** French /sp st sk/ do begin words (sport, station, ski) and
///   French still syllabifies es-pace, es-ca-lier with /s/ in the CODA. Italian `v r` is attested
///   word-initially by ONE Dutch surname (`Vries`) and must be KEPT (a-vran-no, muta cum liquida);
///   Italian `k k` is likewise attested once (`K-Chart`) and must be GATED (ac-qua). Same vote
///   count, opposite verdicts — that pair is the ruler, and it is why no threshold can work here.
///
/// ⇒ SHAPE: per-language curated KEEP lists, and the onset set becomes `observed ∩ KEEP`, never
///   `observed − DROP`. Fail-CLOSED on purpose: a dictionary regeneration that introduces a NEW
///   word-initial cluster gets it gated and turns `s105_west_onset_gate` red for judgement instead
///   of silently making it legal — the river S94 (`N D` / kidney) and S101 (`d n` / sidney) each had
///   to dam by hand after the fact. The cost of that choice is stated under RESIDUE below.
///
/// ⚠ BLAST RADIUS, measured not argued: over all 406182 keys of the four dictionaries this moves
///   consonants BETWEEN syllables and never changes the SYLLABLE COUNT or the phone sequence
///   (`syllabify` emits exactly one syllable per nucleus, whatever the onset set). So
///   `resolve_west_span`'s `n_consumers` is untouched, no word becomes OOV or stops being OOV, and
///   the only audible difference is which note a consonant lands on in a multi-note word.
///   Deduplicated word types re-cut: de 27119 (18.9%) · fr 11347 (10.7%) · es 21285 (23.6%) ·
///   it 12939 (19.4%) = 72690 in total.
///
/// ⛔ TWO NEGATIVE RESULTS, recorded so nobody re-runs them:
///  1. The upstream GTSinger note-boundary surface — which queue (b)8 proposed for exactly this
///     question — CANNOT judge onset legality. Control: a SINGLE intervocalic consonant,
///     unambiguously the onset of the next syllable in all four languages, is assigned to the
///     PREVIOUS note 43% / 47% / 55% / 46% of the time (de/fr/es/it, n = 2221/2323/2817/3531, and
///     identical under both note-segmentation conventions). On the two Spanish clusters whose answer
///     is certain it returns the OPPOSITE of the truth (`t̪ ɾ` whole-in-next 24%, `s t̪` 66%). That
///     annotation encodes articulatory timing, not syllable affiliation.
///  2. The morphological-seam oracle (S102's `T S` instrument) DOES work here — de Fugen-s scores
///     `s f` 666:2, `s b` 596:0, `t l` 426:0 for GATE while obstruent+liquid scores `ɡ ʁ` 0:538,
///     `ʃ p` 0:425 for KEEP — but only after two fixes: a word with a seam BOTH at the cluster start
///     and inside it (`abtragen` = ab+tragen vs abt+ragen) must count as ambiguous, and suffix seams
///     must be excluded (the German superlative `-ste` gave `s t` 324 false KEEP hits, the same
///     mast+er shape S102 already warned about).
///
/// ★ FIVE VERDICTS THE PHONOTACTIC TEMPLATE GOT WRONG ON ITS OWN — all caught by counting the words
///   each cluster actually captures, none by reasoning:
///  · de `s j` / `ts j` — the template said GATE; 1104 of `s j`'s 1292 captures are the ⟨-tion⟩ /
///    ⟨-sion⟩ family (`abduktion` = `a p | d ʊ k t | s j oː n`, `dimension`, `depression`) where the
///    present cut is right or one phone from it and gating moves a SECOND phone into the coda. KEEP.
///    The ~40 Fugen-s compounds on the other side (`abschlussjahr`) keep their existing wrong cut:
///    a named cost, not a regression. ⚠ The clause is RECURSIVE — `X j` is kept only when `X` itself
///    is admitted — so `k s j` and `s t j` are gated with their bases (`Kriegs-jahr`, `Dienst-jah-re`,
///    and `Reflexion` is [ʁeflɛk.si̯oːn] exactly as `Hexe` is Hek-se).
///  · de `t l` — the blanket obstruent+liquid template has to carve out the *tl/*dl gap. 2182 word
///    types, every one of them German morphology (deut-lich, Sport-ler, Abend-land) or a name from a
///    language that forbids /tl/ too. en.tsv was ALREADY gated for this exact cluster citing this
///    exact exonym (`EN_ONSET_DROP`'s gate pins `atlas` = `AE1 T | L AH0 S`, "was … tlingit's T L").
///    French carries the identical gap (183 types: ath-lète, at-las, mat(e)-lot, out-law).
///  · fr obstruent+`ʎ` and obstruent+`ɟ`/`c` — fr.tsv narrows /l/ to `ʎ` before /i/ (S104 §4a proved
///    the training side does the same) and writes /k ɡ/ as `c ɟ` before front vowels (`guerre` =
///    `ɟ ɛ ʁ`, `basket` = `b a s c ɛ t`). So `p ʎ` IS muta cum liquida (`accompli`) and `ɟ ʁ` IS
///    /ɡʁ/. Missing either half of an allophone pair is how a template silently regresses.
///  · it `s`-vs-`z` — it.tsv writes the same environment both ways (`disgustato` = `d i s ɡ …` but
///    `sbagliando` = `z b …`), so the s-impura clause must treat them identically or a transcription
///    detail decides a linguistic verdict (di-sgu-sto, di-sgra-zia).
///  · it `ʃ r` / `ʃ l` — upstream transcribes ⟨scr⟩/⟨scl⟩ that way (`Scritta`, `Sclerosi`,
///    `javascript` = java+script): s-impura + liquid, not /ʃ/ + liquid.
///
/// ★ KNOWN COSTS, each a decision rather than an oversight:
///  · es `t̪ l` / `d̪ l` gated = the PENINSULAR norm (at-le-ta, At-lan-ta). 79 word types; the ~30
///    Nahuatl toponyms in that set (`amatitlán`, `tuxtla`) want the Mexican a-tla and now lose.
///  · it `t l` KEPT while `d l` and `v l` are gated — deliberately NOT symmetric, and the asymmetry
///    is the double-entry, not a rule: `t l`'s 102 captures include ~28 everyday Italian words
///    (atleta / atletica / atlantico / -athlon) that DOP hyphenates a-tlè-ta, whereas `d l` (33) and
///    `v l` (10) contain **no Italian word at all** — only adler/bundle/chandler and
///    pavlova/yakovlev/Vlad, which want the coda. The ~70 foreign names inside `t l` keep the wrong
///    cut; that is the compound-seam residue below, not a new defect.
///  · it `p s` / `p n` / `k t` / `m n` gated = printed hyphenation (au-top-sì-a, cap-su-la,
///    am-ni-stì-a) rather than the school rule "clusters that can begin a word never divide".
///    Genuinely contested in Italian; 192 word types.
///  · es keeps NO obstruent+trill onset. A handful of rows spell the tap with the trill glyph
///    (`labranza` = `l a β r a n s a`, `enjambre` = `e ŋ x a m b r e`) and lose their cut — ~9 word
///    types. Adding `β r` / `b r` back was measured and REJECTED: every genuine Spanish complex
///    onset is written `ɾ` with hundreds of votes, every obstruent+trill entry is a 1-3 vote junk
///    row, and 16 of `β r`'s 23 captures are the sub- prefix family where sub-ra-yar is correct.
///    That is an es.tsv transcription defect and must be fixed there.
///  · Likewise es `d̪ ʃ` (1 word, `neozelandés` — a corrupt two-phone spelling of ⟨z⟩) and fr `ɡ n`
///    (39 of 104 captures are ⟨gn⟩=/ɲ/ that upstream failed to write as `ɲ`; the other 65 are
///    genuine /ɡn/ and want the gate). Both are dictionary rows, not onset-set questions.
///
/// ★ RESIDUE this round does NOT fix, named so it is not mistaken for done:
///  · `observed ∩ KEEP` can only REMOVE. Clusters that are legal but never attested word-initially
///    stay absent, so e.g. es `k l w` (39 types: `excluida` → `e k s k | l w i ð | a`) and `ɡ l j`
///    (4) still mis-cut. Making the keep list authoritative would fix them but requires the list to
///    be exhaustive, which is a separate round with its own evidence.
///  · Compound seams INSIDE kept clusters are untouched (German `Bank|rott`, `Blick|richtung`,
///    `alternativ|los`; the ~70 foreign names in it `t l`). Same missing instrument as queue §C3 —
///    a true-compound seam exception — now owed in four more languages.
const DE_ONSET_KEEP: &[&str] = &[
    // obstruent + liquid (muta cum liquida) — 15 clusters, 16051 captured word types
    "t ʁ", "ɡ ʁ", "k l", "b ʁ", "k ʁ", "p ʁ", "p l", "f l", "f ʁ", "b l", "d ʁ", "ɡ l", "pf l",
    "v ʁ", "pf ʁ",
    // ⟨sch⟩ family, ʃ + C — 7 clusters, 8582 types (ver-ste-hen, ge-spielt)
    "ʃ t", "ʃ p", "ʃ l", "ʃ ʁ", "ʃ v", "ʃ n", "ʃ m",
    // ʃ + stop + liquid — 3 clusters, 1400 types (Straße, sprechen, Splitter)
    "ʃ t ʁ", "ʃ p ʁ", "ʃ p l",
    // the glide clause, single consonant + j (⟨-ion⟩ / ⟨-ie⟩) — 10 clusters, 2105 types
    "s j", "ts j", "p j", "t j", "m j", "ʁ j", "f j", "v j", "d j", "k j",
    // …and an ADMITTED cluster + j (recursive; `k s j` / `s t j` / `m b j` are gated with their
    // bases) — 3 clusters, 2 types
    "b ʁ j", "k n j", "p l j",
    // German-specific onsets: kn-, gn-, ⟨qu⟩ = k v, ⟨zw⟩ = ts v.
    // ⛔ `ɡ v` was proposed here as "⟨qu⟩'s voiced twin" and MEASURED DOWN: ⟨qu⟩+V really is
    //    `k v` (843/909 = 93%), but ⟨gu⟩+V is `ɡ ʊ` — a VOWEL, its own nucleus — in 173 words
    //    against 11 with `ɡ v`, and `guadalquivir` = `ɡ ʊ a d aː l k v ɪ v iː ɐ` carries both
    //    graphemes in one entry. Of the 15 words `ɡ v` captures, the four where the /v/ is real
    //    (Rogg+wil, Edg+ware, Trygg-va-son, mogwai) are exactly the ones German splits; the other
    //    11 are the 6% error tail of ⟨gu⟩ (baguette `ɡ ʊ` vs baguettes `ɡ v`; paraguay `ɡ ʊ` vs
    //    paraguayischen `ɡ v`). Fix belongs in de.tsv, not in the onset set.
    "k v", "ɡ n", "ts v", "k n",
];

/// French. `ʎ` is /l/ before /i/ and `c`/`ɟ` are /k/ /ɡ/ before front vowels — both are allophone
/// spellings of members already in the list, and both cost real words if only half the pair is kept.
const FR_ONSET_KEEP: &[&str] = &[
    // C + glide — 52 clusters, 10828 captured word types (pied, bien, soi, nuit)
    "s j", "ʁ j", "d j", "z j", "t j", "f j", "v j", "t ɥ", "n j", "t w", "ʒ j", "n w", "v w",
    "b j", "d ɥ", "p j", "ʁ w", "k w", "d w", "l w", "n ɥ", "s w", "m w", "ʒ w", "s ɥ", "ɡ w",
    "b w", "l ɥ", "p w", "p ɥ", "b ɥ", "ɡ j", "ʃ w", "ʃ j", "k ɥ", "ɡ ɥ", "z w", "m ɥ", "f w",
    "ʁ ɥ", "f ɥ", "ts w", "ʒ ɥ", "ts j", "c w", "ts ɥ", "ɟ w", "ʃ ɥ", "tʃ j", "ŋ ɥ", "ʎ w", "ɟ j",
    // obstruent + liquid, MINUS the *tl/*dl gap — 21 clusters, 9990 types
    "t ʁ", "p ʁ", "ɡ ʁ", "d ʁ", "b ʁ", "k ʁ", "p l", "k l", "b l", "v ʁ", "f ʁ", "f l", "p ʎ",
    "ɡ l", "b ʎ", "k ʎ", "ɡ ʎ", "v l", "f ʎ", "v ʎ", "ɟ ʁ",
    // obstruent + liquid + glide — 28 clusters, 254 types (croire, trois, fruit)
    "p l w", "t ʁ ɥ", "k ʁ w", "f ʁ w", "t ʁ w", "d ʁ w", "p ʁ j", "b l w", "t ʁ j", "p l ɥ",
    "ɡ l w", "ɡ ʁ j", "ɡ ʁ w", "b ʁ j", "b ʁ w", "b ʁ ɥ", "d ʁ ɥ", "f l w", "f l ɥ", "f ʁ ɥ",
    "k l w", "b l ɥ", "k l ɥ", "k ʁ j", "k ʁ ɥ", "p ʁ w", "p ʁ ɥ", "ɡ ʁ ɥ",
];

/// Spanish. RAE's inventory of complex onsets is exactly obstruent+liquid, plus the rising diphthong
/// (C + glide) which is one syllable with its onset. Everything else closes the previous syllable —
/// es-tar, cam-po, bol-sa, abs-trac-ción — which is why this list is the shortest relative to what
/// the dictionary observes (192 clusters seen, 63 kept).
const ES_ONSET_KEEP: &[&str] = &[
    // C + glide — 34 clusters, 10684 captured word types (cie-lo, puer-ta, pre-mio)
    "θ j", "s j", "m j", "ɲ j", "t̪ j", "l j", "t̪ w", "k w", "p j", "x j", "ɣ w", "f j", "d̪ j",
    "s w", "k j", "p w", "n w", "b j", "l w", "r j", "ɡ w", "x w", "m w", "f w", "r w", "b w",
    "θ w", "d̪ w", "tʃ w", "ʃ w", "ɡ j", "ʎ w", "ɟʝ w", "tʃ j",
    // obstruent + liquid, tap only, minus *t̪l / *d̪l — 12 clusters, 6316 types
    "t̪ ɾ", "p l", "p ɾ", "k ɾ", "k l", "f ɾ", "b ɾ", "f l", "d̪ ɾ", "ɡ ɾ", "b l", "ɡ l",
    // obstruent + liquid + glide — 17 clusters, 314 types (prie-to, true-no)
    "t̪ ɾ j", "t̪ ɾ w", "f l w", "p l j", "f ɾ j", "b ɾ j", "p ɾ j", "k ɾ j", "p ɾ w", "ɡ ɾ j",
    "ɡ ɾ w", "b ɾ w", "d̪ ɾ w", "f ɾ w", "k l j", "k ɾ w", "ɡ l w",
];

/// Italian. Two families only: muta cum liquida, and s impura — the one syllabification rule Italian
/// grammars are unanimous about ("la s seguita da consonante non si divide mai": pa-sta, que-sto,
/// mo-stro, di-sgu-sto). `j`/`w` get no clause at all: it.tsv writes native rising diphthongs as
/// VOWELS (`piano` = `p i a n o`, `acqua` = `a k k u a`), so every `C j` / `C w` cluster it observes
/// is a loanword (Twain, Brentwood) and belongs in the coda.
const IT_ONSET_KEEP: &[&str] = &[
    // obstruent + liquid — 14 clusters, 5034 captured word types (a-vran-no; `t l` see KNOWN COSTS)
    "t r", "p r", "ɡ r", "b r", "k r", "k l", "p l", "d r", "f r", "b l", "v r", "ɡ l", "f l",
    "t l",
    // s impura, s and z treated identically — 14 clusters, 5919 types
    "s t", "s t͡ʃ", "s p", "z m", "s k", "z l", "s f", "z b", "z n", "z d", "z v", "z r", "s ɡ",
    "s m",
    // s impura + obstruent + liquid — 10 clusters, 966 types (mo-stro, di-sgra-zia)
    "s t r", "s p l", "s p r", "s ɡ r", "s k r", "z b r", "z d r", "s k l", "s f r", "z b l",
    // ⟨scr⟩ / ⟨scl⟩ as upstream spells them — 2 clusters, 141 types
    "ʃ r", "ʃ l",
];

/// May this OBSERVED word-initial multi-consonant cluster drive INTERVOCALIC cuts in `syllabify`?
/// EN keeps the S94 vote threshold plus its curated drop list; de/fr/es/it are governed by the
/// per-language keep lists above (`FR_ONSET_DROP` is redundant under them and kept as the S101
/// D6 invariant's own tripwire). zh/ja never reach here with a space in the cluster — their phones
/// are single consonants by construction — so the fallback preserves the pre-S105 behaviour.
fn onset_admitted(lang: Lang, cluster: &str, votes: u32, min_votes: u32) -> bool {
    match lang {
        Lang::En => votes >= min_votes && !EN_ONSET_DROP.contains(&cluster),
        Lang::De => DE_ONSET_KEEP.contains(&cluster),
        Lang::Fr => FR_ONSET_KEEP.contains(&cluster) && !FR_ONSET_DROP.contains(&cluster),
        Lang::Es => ES_ONSET_KEEP.contains(&cluster),
        Lang::It => IT_ONSET_KEEP.contains(&cluster),
        _ => votes >= min_votes,
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

/// S104 — may a FR/IT elision attach to a word that starts with this letter? (`WordDict::elision`.)
///
/// This is an ORTHOGRAPHIC test on purpose and it is not interchangeable with `dict_is_vowel`: the
/// question elision asks is the one French spelling asks — "does the next word begin with a vowel
/// letter or a mute h?" — not "is its first PHONE a nucleus". The two differ exactly on ⟨oi ou hui
/// hi⟩ (`oiseau` = `w a z o`, `huile` = `ɥ i l`: elidable, but the first phone is a glide) and on
/// ⟨y w⟩ (`yacht`, `week-end`: first phone is likewise a glide, and French does NOT elide). A phone
/// test has to get one of those two families wrong; the letter test gets both right. ⟨y⟩ is
/// therefore deliberately absent from the list below.
fn elidable_head(c: char) -> bool {
    matches!(
        c.to_lowercase().next().unwrap_or(c),
        'a' | 'e' | 'i' | 'o' | 'u'
            | 'à' | 'á' | 'â' | 'ä' | 'ã' | 'å'
            | 'è' | 'é' | 'ê' | 'ë'
            | 'ì' | 'í' | 'î' | 'ï'
            | 'ò' | 'ó' | 'ô' | 'ö' | 'õ'
            | 'ù' | 'ú' | 'û' | 'ü'
            | 'æ' | 'œ'
            | 'h'
    )
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
/// SHIPPED en.tsv the two verdicts agree on every one of the 69 distinct tokens (863018 instances at
/// the S90 measurement; 862976 after the S94 -en regeneration — the E2E walk re-proves it live), so
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

// ─── S95 fragment merge (拆词) ───────────────────────────────────────────────────────────────────

/// Most word-fragment spellings need at most three notes (to|get|ther); four is headroom, and the
/// cap keeps the join search bounded on pathological scores (eighty one-letter notes must not
/// explode into windows).
const MERGE_MAX_WORDS: usize = 4;

/// S95 fragment merge: try every deterministic join of `frags` against the dictionary, fewest
/// seam-dedupes first (the faithful concatenation is candidate #0). A seam may fold ONE doubled
/// consonant LETTER: the UTAU re-attack convention writes "nev|ver" / "giv|ving" / "look|king" —
/// the second note re-attacks with the consonant it shares with the first, but the WORD carries it
/// once. Vowel letters never fold (no attested case, and "a|and" must not become "and"… minus a).
/// Every candidate goes through `WordDict::lookup`, i.e. the full S86 tolerance ladder plus the
/// S95 plural rung — ONE lookup notion, nothing new to drift.
/// Returns the best (fewest-dedupe = longest) hit for THIS window as
/// `(joined char length, seam dedupes, trad phones)` — the caller compares windows by length
/// (see the selection comment in the merge pass; review S95R-1).
fn join_lookup(dict: &WordDict, frags: &[&str]) -> Option<(usize, u32, Vec<String>)> {
    let seams = frags.len() - 1;
    let foldable: Vec<bool> = (0..seams)
        .map(|s| {
            match (frags[s].chars().last(), frags[s + 1].chars().next()) {
                (Some(a), Some(b)) => {
                    a.eq_ignore_ascii_case(&b)
                        && a.is_ascii_alphabetic()
                        && !matches!(a.to_ascii_lowercase(), 'a' | 'e' | 'i' | 'o' | 'u')
                }
                _ => false,
            }
        })
        .collect();
    let mut masks: Vec<u32> = (0u32..1 << seams)
        .filter(|m| (0..seams).all(|s| m & (1 << s) == 0 || foldable[s]))
        .collect();
    masks.sort_by_key(|m| (m.count_ones(), *m));
    for mask in masks {
        let mut joined = frags[0].to_string();
        for (s, f) in frags[1..].iter().enumerate() {
            if mask & (1 << s) != 0 {
                joined.pop(); // drop the LEFT copy of the doubled letter (char-wise, UTF-8 safe)
            }
            joined.push_str(f);
        }
        if let Some(t) = dict.lookup(&joined) {
            return Some((joined.chars().count(), mask.count_ones(), t));
        }
    }
    None
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
    /// ★S96 knife ① — per-NUCLEUS ARPABET stress for this note's phones, in nucleus order
    /// (0 = unstressed, 1 = primary, 2 = secondary). `Some` ONLY for dictionary/hint words whose
    /// traditional phones carry stress digits (en; the MFA sets have none ⇒ de/fr/es/it stay `None`
    /// and allocate exactly as before). The S90 zero-regression gate proved "carries a digit" ≡
    /// "IPA nucleus-capable" over all 863018 en.tsv tokens, so the count matches the resolved
    /// nuclei BY CONSTRUCTION — the allocator still re-verifies and falls back to None on mismatch.
    pub nucleus_stress: Option<Vec<u8>>,
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
                && zh_lyric_hanzi(zh, score[k].lyric).is_some()
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
            // same resolver as the window scan above — a second, subtly different reading of "which
            // char is this note" is exactly how a phrase window would silently mis-assign syllables
            let chars: Vec<char> = idx
                .iter()
                .map(|&k| {
                    zh_lyric_hanzi(zh, score[k].lyric).expect("is_plain_hanzi already proved this resolves")
                })
                .collect();
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

    // ── S95 fragment-merge pass (拆词): join adjacent OOV word fragments into ONE dictionary word.
    // UTAU authors write one word as per-note fragments — "e|ven", "nev|ver", "to|get|ther" — with
    // the seam consonant usually doubled (CV re-attack). Trigger = a western Word note whose lyric
    // the FAITHFUL tolerance ladder misses (`lookup_faithful`; the plural rung is deliberately NOT
    // part of the trigger — review S95R-2: with it, "mes" reads as me+Z, stops counting as OOV,
    // and mes|sage sings "meez sage" instead of merging to "message". Plural is a last-resort
    // reading; a fragment that can still become a whole word by merging gets that chance first).
    // Version note (the precise invariant — review S95R-1 corrected an overclaimed draft): every
    // affected note (the trigger AND the neighbours a merge re-syllabifies) sits in a segment
    // whose strict render always ERRORED before this pass, so any bake such a segment still holds
    // is necessarily sig-DIRTY (a stale stem kept alive by the render-failure fallback is real;
    // a sig-CLEAN one is impossible pre-S95). First play re-renders through the ordinary dirty
    // path — no `G2P_ALGO_VERSION` bump is needed for S95 itself, and bumping would force a
    // pointless full re-render of every project (S90: "hardening" passes the same bar).
    // ⚠ FORWARD CAVEAT — this argument is NOT reusable for future changes to what a merge SINGS:
    // post-S95, splitting a cleanly-baked merged span can leave an OOV-only half holding a
    // window-sig-clean bake, and the pass sees exactly the render wire (buildScoreTriples output;
    // that shared wire is what keeps editor/render/merge domains equal). Any later change to
    // merge output MUST bump `G2P_ALGO_VERSION`.
    // Non-OOV fragment pairs (look|king, pro|miss — both real words) are DELIBERATELY not merged:
    // they render today, mostly AS the author's intended double-consonant re-attack; rewriting
    // them would alter already-renderable scores and needs its own bump + ear evidence (user
    // verdict, S95 scope).
    // Determinism: triggers scan left→right (a successful merge CLAIMS its words; claimed words
    // never merge again); window selection is greedy-longest (see the comment at the search).
    // Holds between fragments are TRANSPARENT — they join the span, where the distribution logic
    // re-emits the current nucleus ("e|-|ven" holds the /i/). `+`, rests, breaths, hints, aliases,
    // other languages, `phoneme_input` and alias-failures all BREAK the window: those notes are
    // not on the plain-dictionary path this pass rescues.
    let mut merge_trad: Vec<Option<Vec<String>>> = vec![None; n];
    let mut merge_last: Vec<Option<usize>> = vec![None; n];
    {
        let mut claimed: Vec<bool> = vec![false; n];
        let frag_ok = |k: usize, lang: Lang, claimed: &Vec<bool>| -> bool {
            !claimed[k]
                && toks[k] == Tok::Word
                && !matches!(score[k].lang, Lang::Ja | Lang::Zh)
                && score[k].lang == lang
                && score[k].phoneme_input.is_none()
                && !alias_failed[k]
        };
        for i in 0..n {
            let lang = score[i].lang;
            if !frag_ok(i, lang, &claimed) {
                continue;
            }
            let dict = dicts.words(lang)?;
            if dict.lookup_faithful(score[i].lyric.trim()).is_some() {
                continue; // not faithful-OOV — never a trigger (see the scope note above)
            }
            // fragment words reachable from the trigger, nearest-first, holds transparent
            let mut left: Vec<usize> = Vec::new();
            let mut k = i;
            'l: while left.len() < MERGE_MAX_WORDS - 1 {
                let mut p = k;
                loop {
                    if p == 0 {
                        break 'l;
                    }
                    p -= 1;
                    if toks[p] != Tok::Hold {
                        break;
                    }
                }
                if frag_ok(p, lang, &claimed) {
                    left.push(p);
                    k = p;
                } else {
                    break;
                }
            }
            let mut right: Vec<usize> = Vec::new();
            let mut k = i;
            'r: while right.len() < MERGE_MAX_WORDS - 1 {
                let mut p = k + 1;
                while p < n && toks[p] == Tok::Hold {
                    p += 1;
                }
                if p >= n {
                    break 'r;
                }
                if frag_ok(p, lang, &claimed) {
                    right.push(p);
                    k = p;
                } else {
                    break;
                }
            }
            // Window selection is GREEDY-LONGEST over every window containing the trigger — the
            // same rule the zh phrase window and the alias tokenizer already live by. The shipped
            // first draft took the FIRST hit in fewest-words-first order and review S95R-1 broke
            // it on the real dictionary: the 2-word forward join ful|fil→"fulfil" stole the tail
            // of won|der|ful before the 3-word "wonderful" was ever tried (won|der|fulfil|ling,
            // zero red marks), and the 2-word backward join a|mes→"ames" beats mes|sage→"message"
            // the same way. Longest joined string wins both. Ties: fewer seam-dedupes (more
            // faithful), then more backward context (OOV fragments are usually word TAILS), then
            // the leftmost window — a total order, so the choice is deterministic.
            let mut best_key: Option<(usize, std::cmp::Reverse<u32>, usize, std::cmp::Reverse<usize>)> = None;
            let mut best_hit: Option<(Vec<usize>, Vec<String>)> = None;
            for m in 2..=MERGE_MAX_WORDS {
                for back in (0..m).rev() {
                    let fwd = m - 1 - back;
                    if back > left.len() || fwd > right.len() {
                        continue;
                    }
                    let mut words: Vec<usize> = left[..back].iter().rev().copied().collect();
                    words.push(i);
                    words.extend_from_slice(&right[..fwd]);
                    let frags: Vec<&str> = words.iter().map(|&w| score[w].lyric.trim()).collect();
                    if let Some((len, dedupes, trad)) = join_lookup(dict, &frags) {
                        let key = (len, std::cmp::Reverse(dedupes), back, std::cmp::Reverse(words[0]));
                        if best_key.as_ref().map_or(true, |k| key > *k) {
                            best_key = Some(key);
                            best_hit = Some((words, trad));
                        }
                    }
                }
            }
            if let Some((words, trad)) = best_hit {
                let head = words[0];
                merge_trad[head] = Some(trad);
                merge_last[head] = Some(*words.last().expect("merge window has >= 2 words"));
                // Claiming is the determinism invariant ("a word belongs to at most one merge"),
                // not a behaviour observable at fixture scale: the main pass only consults heads
                // it visits, so an overlapping later merge is naturally shadowed unless a 4-word
                // double-overlap re-heads an earlier span — a shape no attested material produces
                // (mutation M2 survives the test suite; documented rather than pinned, S92p rule
                // against pinning coincidences).
                for &w in &words {
                    claimed[w] = true;
                }
            }
        }
    }

    // main pass: per-note phones + carrier state for sustains; western spans handled look-ahead.
    let mut out: Vec<Option<ResolvedNote>> = vec![None; n];
    // (S99: the local `oov` closure that used to live here is gone — `NoteFail::into_error` is now the
    // ONE place that formats a per-note failure, so the OOV wording cannot drift from the new
    // unknown-phone wording. The alias arm below still formats its own CODE: it fails before a note
    // resolver is ever called, and its payload is the CONVENTION + lyric, not a phone.)
    // carrier nucleus for holds outside western spans (ja legacy prev_vowel / zh final).
    let mut carrier: Option<&'static str> = None;

    let mut i = 0;
    while i < n {
        let evt = &score[i];
        let run_lang = run_langs[i];
        match toks[i] {
            Tok::Rest => {
                carrier = None;
                out[i] = Some(ResolvedNote { kind: ResolvedKind::Rest, run_lang, is_sustain: false, nucleus_stress: None });
                i += 1;
            }
            Tok::Breath => {
                carrier = None;
                out[i] = Some(ResolvedNote { kind: ResolvedKind::Breath, run_lang, is_sustain: false, nucleus_stress: None });
                i += 1;
            }
            Tok::Hold | Tok::Next => {
                // an orphan sustain (span-attached ones were consumed below): legacy ja semantics —
                // re-emit the carrier nucleus, default "a".
                let ph = vec![carrier.unwrap_or("a")];
                out[i] = Some(ResolvedNote { kind: ResolvedKind::Phones(ph), run_lang, is_sustain: true, nucleus_stress: None });
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
                    out[i] = Some(ResolvedNote { kind: ResolvedKind::Unknown, run_lang, is_sustain: false, nucleus_stress: None });
                    i += 1;
                    continue;
                }
                match evt.lang {
                    Lang::Ja | Lang::Zh => {
                        match resolve_east_word(evt, zh_syl[i].as_deref(), dicts)? {
                            Ok(ph) => {
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
                                out[i] = Some(ResolvedNote {
                                    kind: ResolvedKind::Phones(ph),
                                    run_lang,
                                    is_sustain: false,
                                    nucleus_stress: None,
                                });
                            }
                            Err(fail) => {
                                if strict {
                                    return Err(fail.into_error(evt.lyric));
                                }
                                out[i] = Some(ResolvedNote { kind: ResolvedKind::Unknown, run_lang, is_sustain: false, nucleus_stress: None });
                            }
                        }
                        i += 1;
                    }
                    _ => {
                        // western span: this word + following hold/next notes (any language change in a
                        // sustain is ignored — sustains inherit the carrier by construction).
                        // S95: a fragment-merged head extends the span over its member words (and any
                        // holds between them); members consume syllables exactly like `+` notes, so the
                        // Word→Next rewrite below is the WHOLE integration — distribution, hold re-emit
                        // and coda deferral are the machinery `+` spans already run.
                        let mut span_end = merge_last[i].map_or(i + 1, |l| l + 1);
                        while span_end < n && matches!(toks[span_end], Tok::Hold | Tok::Next) {
                            span_end += 1;
                        }
                        let merged_toks: Vec<Tok>;
                        let span_toks: &[Tok] = if merge_last[i].is_some() {
                            merged_toks = toks[i..span_end]
                                .iter()
                                .map(|&t| if t == Tok::Word { Tok::Next } else { t })
                                .collect();
                            &merged_toks
                        } else {
                            &toks[i..span_end]
                        };
                        match resolve_west_span(
                            evt,
                            &score[i..span_end],
                            span_toks,
                            dicts,
                            merge_trad[i].as_deref(),
                        )? {
                            Ok(assignments) => {
                                for (j, (ph, stress)) in assignments.into_iter().enumerate() {
                                    carrier = ph.last().copied().or(carrier);
                                    out[i + j] = Some(ResolvedNote {
                                        kind: ResolvedKind::Phones(ph),
                                        run_lang: run_langs[i + j],
                                        is_sustain: j > 0,
                                        nucleus_stress: stress,
                                    });
                                }
                            }
                            Err(fail) => {
                                if strict {
                                    return Err(fail.into_error(evt.lyric));
                                }
                                out[i] = Some(ResolvedNote { kind: ResolvedKind::Unknown, run_lang, is_sustain: false, nucleus_stress: None });
                                for j in i + 1..span_end {
                                    // the sustains still resolve (hold "a") so ONLY the word marks OOV
                                    out[j] = Some(ResolvedNote {
                                        kind: ResolvedKind::Phones(vec!["a"]),
                                        run_lang: run_langs[j],
                                        is_sustain: true,
                                        nucleus_stress: None,
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

/// Why ONE sung word note did not resolve — the distinction the USER sees in the message.
///
/// `stage2` and `intern` already NAME the phone they reject — `stage2`'s own doc says so: "Err = the
/// offending phone (caller wraps the CODE)". Both callers used to drop that name on the floor
/// (`Err(_) => Ok(None)` and `.ok()`), so a mistyped bracket hint surfaced as `VOCAL_OOV:
/// [dh ae zzz]` and the frontend told the user to *check the lyrics or the language* — for a note
/// whose lyric is fine and whose THIRD PHONEME is the typo. S99 pays off that S90 debt by carrying
/// the name out. (Nothing else changes: an unknown phone is still a per-note failure, so the editor
/// still marks exactly that note red and the render still stops.)
enum NoteFail {
    /// Ordinary OOV: this lyric is not a word / mora / syllable of its language.
    Oov,
    /// A phone is not in the 210-token inventory. In practice always user-supplied — a bracket hint
    /// or a `phoneme_input` override — because every dictionary phone is proven to intern by
    /// `dictionaries_end_to_end`; naming it is right either way.
    UnknownPhone(String),
}

impl NoteFail {
    /// The strict-render error this becomes. `lyric` is consulted only by the OOV arm: an unknown
    /// phone names the PHONE, which is the entire point of this type.
    fn into_error(self, lyric: &str) -> UtaiError {
        match self {
            NoteFail::Oov => UtaiError::Inference(format!("VOCAL_OOV: {lyric}")),
            NoteFail::UnknownPhone(p) => UtaiError::Inference(format!("VOCAL_UNKNOWN_PHONE: {p}")),
        }
    }
}

/// Outer `Err` = INFRASTRUCTURE failure (a missing dictionary — surfaced as `VOCAL_DICT_MISSING` and
/// never masked as a per-note verdict, audit MAJOR). Inner `Err` = this NOTE failed, and why.
type NoteResolve<T> = Result<std::result::Result<T, NoteFail>>;

/// Resolve a JA/ZH sung word note to IPA phones. Inner `Err` = the note failed (see `NoteFail`);
/// `Err` = INFRASTRUCTURE failure (missing dictionary — propagated as VOCAL_DICT_MISSING, never
/// masked as OOV; audit MAJOR). §3.7 override precedence: whitespace phoneme_input = raw traditional
/// phones; no-space = a mora (ja) / pinyin syllable (zh); otherwise ja = legacy mora path, zh =
/// phrase-resolved reading (or the lyric as bare pinyin).
fn resolve_east_word(
    evt: &ScoreEvt,
    zh_phrase_syl: Option<&str>,
    dicts: &dyn DictSource,
) -> NoteResolve<Vec<&'static str>> {
    if let Some(pi) = evt.phoneme_input {
        let pi = pi.trim();
        if pi.contains(char::is_whitespace) {
            // THE raw-phones escape hatch, and the landing site of a bracket hint — i.e. the one place
            // where the phones are the USER's, so a rejected one must be named back to them.
            let phones: Vec<String> = pi.split_whitespace().map(str::to_string).collect();
            let r = match evt.lang {
                Lang::Ja => ja_phones_from_tokens(&phones),
                _ => stage2(evt.lang, &phones),
            };
            return Ok(r.map_err(NoteFail::UnknownPhone));
        }
    }
    match evt.lang {
        Lang::Ja => {
            let token = evt.phoneme_input.map(str::trim).unwrap_or(evt.lyric);
            Ok(ja_word_phones(token).ok_or(NoteFail::Oov))
        }
        _ => {
            let zh = dicts.zh()?;
            let trad = match (evt.phoneme_input, zh_phrase_syl) {
                // an explicit override is taken at face value — the user typed the syllable itself
                (Some(pi), _) => zh.syllable_phones(&pi.trim().to_lowercase()),
                (None, Some(s)) => zh.syllable_phones(s),
                // not a plain hanzi: try the lyric as a bare pinyin syllable, through the SAME ladder
                // (S99 — 「hao,」 was OOV where 「Love,」 was not)
                (None, None) => lookup_candidates(evt.lyric).iter().find_map(|c| zh.syllable_phones(c)),
            };
            let Some(trad) = trad else { return Ok(Err(NoteFail::Oov)) };
            Ok(stage2(Lang::Zh, &trad).map_err(NoteFail::UnknownPhone))
        }
    }
}

/// ZH word lyric → the single hanzi it stands for, through the SAME faithful-first tolerance ladder
/// (`lookup_candidates`) that `ja_word_phones` and `WordDict::lookup_faithful` already consume.
///
/// S99 (S86#8-2): zh was the ONE language left out of that ladder, so a pasted 「我，」 was a hard OOV
/// and aborted the whole render, while 「Love,」 and 「か、」 resolved fine — six languages tolerated
/// trailing punctuation and Chinese did not. Returning the RESOLVED char (not a bool) is what lets the
/// phrase-context pass use it too: the window scan and the char extraction must agree on what a note
/// "is", and before this they were two separate `lyric.trim().chars()` expressions.
///
/// Faithful-first is preserved by construction: the raw spelling is candidate #0, so a lyric that
/// really is a bare hanzi never consults a trimmed rung.
fn zh_lyric_hanzi(zh: &ZhDict, lyric: &str) -> Option<char> {
    lookup_candidates(lyric).iter().find_map(|cand| {
        let mut cs = cand.chars();
        match (cs.next(), cs.next()) {
            (Some(c), None) if zh.is_hanzi(c) => Some(c),
            _ => None,
        }
    })
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
fn ja_phones_from_tokens(phones: &[String]) -> std::result::Result<Vec<&'static str>, String> {
    phones.iter().map(|p| intern(p).ok_or_else(|| p.clone())).collect()
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
    // S95 fragment merge: the pre-pass already joined this span's fragment lyrics into one
    // dictionary word — these are its traditional phones, overriding the per-note lookup
    // (which by construction MISSED on the trigger fragment). `phoneme_input` still wins:
    // a merge head never has one (that is a mergeability condition), so the order is moot
    // today, but the precedence mirrors the documented user-layer chain.
    merged_trad: Option<&[String]>,
) -> NoteResolve<Vec<(Vec<&'static str>, Option<Vec<u8>>)>> {
    let dict = dicts.words(evt.lang)?;
    // stage1: the word's traditional phones (override with spaces already handled by the caller — a
    // no-space override here is a single traditional phone).
    let trad: Vec<String> = if let Some(pi) = evt.phoneme_input {
        pi.split_whitespace().map(str::to_string).collect()
    } else if let Some(mt) = merged_trad {
        mt.to_vec()
    } else {
        match dict.lookup(evt.lyric.trim()) {
            Some(t) => t,
            None => return Ok(Err(NoteFail::Oov)),
        }
    };
    if trad.is_empty() {
        return Ok(Err(NoteFail::Oov));
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

    // stage2 each note's traditional phones → interned IPA (a bad phone = real OOV, not an error).
    // ★S96 knife ① — extract the per-NUCLEUS stress digits BEFORE stage2 strips them (the ONLY
    // point in the pipeline where both the note assignment and the digits exist). "Carries a digit"
    // ≡ "IPA nucleus-capable" on the shipped dictionaries (the S90 zero-regression gate, all 863018
    // en.tsv tokens), so collecting digits in token order IS the note's nucleus-stress vector.
    // A set with no digits at all (the MFA languages) yields empty vecs ⇒ `None` — those languages
    // allocate exactly as before, no new behaviour without evidence.
    let mut out: Vec<(Vec<&'static str>, Option<Vec<u8>>)> = Vec::with_capacity(assign_trad.len());
    for tr in assign_trad {
        let stress: Vec<u8> = tr
            .iter()
            .filter_map(|t| match t.chars().last() {
                Some('0') => Some(0),
                Some('1') => Some(1),
                Some('2') => Some(2),
                _ => None,
            })
            .collect();
        match stage2(evt.lang, &tr) {
            Ok(ph) => out.push((ph, (!stress.is_empty()).then_some(stress))),
            // stage2 NAMES the phone it rejected — carry it out instead of collapsing the whole note
            // to "OOV lyric" (S99 / the S90 debt). Reachable only through user-supplied phones: every
            // dictionary phone is proven to intern by `dictionaries_end_to_end`.
            Err(bad) => return Ok(Err(NoteFail::UnknownPhone(bad))),
        }
    }
    Ok(Ok(out))
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
        // S94: built with min_votes = 1 — a nine-word fixture cannot attest a cluster
        // EN_ONSET_MIN_VOTES times, and these tests exercise the maximal-onset MECHANISM, not the
        // production threshold. The threshold's behavior over the REAL en.tsv is pinned by
        // `s94_en_onset_vote_gate` below.
        WordDict::from_tsv_min_votes(
            Lang::En,
            // S95 rows (fragment merge / plural rung): all single-consonant or vowel-initial
            // onsets, so none of them can move an existing maximal-onset cut. Deliberately
            // ABSENT: nev/ven/ther/giv/ving/leeve/ful/mes + gether/nevver/givving/amessage —
            // the merge tests need the fragments to genuinely miss and the joins to be
            // unambiguous. `ames`/`fulfil` are PRESENT on purpose: they are the shorter-window
            // thieves the greedy-longest selection must beat (review S95R-1).
            "light\tL AY1 T\nbeautiful\tB Y UW1 T AH0 F AH0 L\ntree\tT R IY1\nsinger\tS IH1 NG ER0\nextra\tEH1 K S T R AH0\ntwo\tT UW1\nfun\tF AH1 N\nlove\tL AH1 V\ndon't\tD OW1 N T\ne\tIY1\neven\tIY1 V AH0 N\nver\tV ER1\nnever\tN EH1 V ER0\nto\tT UW1\nget\tG EH1 T\nthe\tDH AH0\ntogether\tT AH0 G EH1 DH ER0\ngiving\tG IH1 V IH0 NG\nlook\tL UH1 K\nking\tK IH1 NG\nlooking\tL UH1 K IH0 NG\ndear\tD IH1 R\nrose\tR OW1 Z\nbe\tB IY1\na\tAH0\nwon\tW AH1 N\nder\tD ER1\nfil\tF IH1 L\nling\tL IH1 NG\nwonderful\tW AH1 N D ER0 F AH0 L\nfulfil\tF UH0 L F IH1 L\nme\tM IY1\nsage\tS EY1 JH\nmessage\tM EH1 S AH0 JH\names\tEY1 M Z\ngue\tG Y UW1\nsing\tS IH1 NG\nguessing\tG EH1 S IH0 NG\n",
            1,
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
    /// S104 — the REAL fr.tsv as a `DictSource`, so a French test can drive `resolve_score` /
    /// `classify_score` down the same chain the app uses instead of only calling `lookup`.
    struct FrOnly(WordDict);
    impl DictSource for FrOnly {
        fn zh(&self) -> Result<&ZhDict> {
            Err(UtaiError::Inference("VOCAL_DICT_MISSING: test".into()))
        }
        fn words(&self, lang: Lang) -> Result<&WordDict> {
            if lang == Lang::Fr {
                Ok(&self.0)
            } else {
                Err(UtaiError::Inference("VOCAL_DICT_MISSING: test".into()))
            }
        }
    }
    /// Read the shipped fr.tsv, or `None` when this is a bare checkout (data/ is gitignored).
    fn fr_tsv() -> Option<String> {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/dictionaries/fr.tsv");
        std::fs::read_to_string(&p).ok()
    }
    fn evt(lyric: &str, lang: Lang) -> ScoreEvt<'_> {
        evt_set(lyric, lang, PhonemeSet::Words)
    }
    /// Same note, on a chosen phoneme CONVENTION. S99: `evt` used to hard-code `Words`, which meant
    /// the `g2p_probe` diagnostic could not reach the alias path AT ALL — the one code path whose
    /// tokenizer we most need to observe on real symbols (S91's "unknown multi-letter symbol splits
    /// silently" debt). A probe that structurally cannot exercise an arm cannot report on it.
    fn evt_set(lyric: &str, lang: Lang, phoneme_set: PhonemeSet) -> ScoreEvt<'_> {
        ScoreEvt { lyric, note_num: 60, frames: 20, lang, phoneme_input: None, phoneme_set }
    }
    fn phones_of(nt: &ResolvedNote) -> Vec<&'static str> {
        match &nt.kind {
            ResolvedKind::Phones(p) => p.clone(),
            other => panic!("expected phones, got {other:?}"),
        }
    }

    // ── S95 fragment merge (拆词): adjacent OOV fragments join into one dictionary word ──

    #[test]
    fn fragment_merge_joins_backward() {
        let f = fixtures();
        // "e|ven": the trigger is the SECOND note (ven is OOV, e is a real word) — the window
        // reaches BACKWARD and the pair resolves as "even", the member consuming a syllable
        // exactly like a `+` note.
        let score = [evt("e", Lang::En), evt("ven", Lang::En)];
        let r = resolve_score(&score, &f).unwrap();
        assert_eq!(phones_of(&r[0]), vec!["i"]);
        assert_eq!(phones_of(&r[1]), vec!["v", "ə", "n"]);
        assert!(!r[0].is_sustain && r[1].is_sustain);
        // discriminator: the fragment alone is genuinely OOV — the merge is what rescued it
        let e = resolve_score(&[evt("ven", Lang::En)], &f).unwrap_err();
        assert!(e.to_string().contains("VOCAL_OOV: ven"), "{e}");
    }

    #[test]
    fn fragment_merge_folds_the_doubled_seam_consonant() {
        let f = fixtures();
        // "nev|ver": the faithful join "nevver" misses; folding the doubled v finds "never".
        // NB "ver" IS a real dictionary word — only the TRIGGER has to be OOV, not every member.
        let score = [evt("nev", Lang::En), evt("ver", Lang::En)];
        let r = resolve_score(&score, &f).unwrap();
        assert_eq!(phones_of(&r[0]), vec!["n", "ɛ"]);
        assert_eq!(phones_of(&r[1]), vec!["v", "ɝ"]);
        assert!(r[1].is_sustain);
    }

    #[test]
    fn fragment_merge_three_word_window_and_disjoint_merges() {
        let f = fixtures();
        // TYFD shape "to|get|ther giv|ving": a 3-word window (seam t folded: togetther→together)
        // and a 2-word window resolve side by side. NB (review S95R-4): the claiming machinery
        // and the scan direction are NOT observable at this fixture's scale — "thergiv" misses
        // the dictionary either way — so this test pins the two disjoint merges, nothing more;
        // claiming stays documented as a determinism invariant at the implementation site.
        let score = [
            evt("to", Lang::En),
            evt("get", Lang::En),
            evt("ther", Lang::En),
            evt("giv", Lang::En),
            evt("ving", Lang::En),
        ];
        let r = resolve_score(&score, &f).unwrap();
        assert_eq!(phones_of(&r[0]), vec!["t", "ə"]);
        assert_eq!(phones_of(&r[1]), vec!["ɡ", "ɛ"]);
        assert_eq!(phones_of(&r[2]), vec!["ð", "ɝ"]);
        assert_eq!(phones_of(&r[3]), vec!["ɡ", "ɪ"]);
        assert_eq!(phones_of(&r[4]), vec!["v", "ɪ", "ŋ"]);
        assert!(r[1].is_sustain && r[2].is_sustain && !r[3].is_sustain && r[4].is_sustain);
    }

    #[test]
    fn fragment_merge_hold_is_transparent() {
        let f = fixtures();
        // "e | - | ven": the author held the first fragment's vowel. The hold joins the merged
        // span, where the distribution logic re-emits the current nucleus — same as a hold
        // inside an ordinary `+` span.
        let score = [evt("e", Lang::En), evt("-", Lang::En), evt("ven", Lang::En)];
        let r = resolve_score(&score, &f).unwrap();
        assert_eq!(phones_of(&r[0]), vec!["i"]);
        assert_eq!(phones_of(&r[1]), vec!["i"]);
        assert_eq!(phones_of(&r[2]), vec!["v", "ə", "n"]);
    }

    #[test]
    fn fragment_merge_never_rewrites_working_notes() {
        let f = fixtures();
        // "look|king": BOTH are real words → no OOV → no trigger. They keep singing as two
        // words (the author's double-consonant re-attack), byte-identical to pre-S95 output.
        // This IS the scope boundary: merging non-OOV pairs would alter already-renderable
        // scores and needs its own version bump + ear evidence (user verdict, S95).
        let score = [evt("look", Lang::En), evt("king", Lang::En)];
        let r = resolve_score(&score, &f).unwrap();
        assert_eq!(phones_of(&r[0]), vec!["l", "ʊ", "k"]);
        assert_eq!(phones_of(&r[1]), vec!["k", "ɪ", "ŋ"]);
        assert!(!r[1].is_sustain);
    }

    #[test]
    fn fragment_merge_window_breaks_where_the_dictionary_path_ends() {
        let f = fixtures();
        let oov = |score: &[ScoreEvt]| {
            let e = resolve_score(score, &f).unwrap_err().to_string();
            assert!(e.contains("VOCAL_OOV"), "{e}");
        };
        // a rest between fragments = the author separated words → stays loud OOV
        oov(&[evt("e", Lang::En), evt("R", Lang::En), evt("ven", Lang::En)]);
        // language mismatch breaks the window (both notes end up individually OOV)
        oov(&[evt("e", Lang::De), evt("ven", Lang::En)]);
        // …and in the direction stage2 can NOT mask (review S95R-5): an EN trigger must not
        // pull a De member in — without the lang gate this would merge to "never" with the De
        // note silently singing an English syllable, and no later stage would object.
        oov(&[evt("nev", Lang::En), evt("ver", Lang::De)]);
        // a bracket hint takes the neighbour off the dictionary path — no merge across it
        oov(&[evt("e[iy1]", Lang::En), evt("ven", Lang::En)]);
        // an explicit phoneme_input does the same
        let mut with_pi = [evt("e", Lang::En), evt("ven", Lang::En)];
        with_pi[0].phoneme_input = Some("IY1");
        oov(&with_pi);
        // no dictionary join exists (respelling, not fragmentation) → loud on the OOV note
        let e = resolve_score(&[evt("be", Lang::En), evt("leeve", Lang::En)], &f).unwrap_err();
        assert!(e.to_string().contains("VOCAL_OOV: leeve"), "{e}");
    }

    #[test]
    fn fragment_merge_editor_and_render_agree() {
        let f = fixtures();
        // §9.5 single classifier: merged members classify Sustain (their red marks clear with
        // ZERO frontend changes), a failed merge stays Unknown on exactly the OOV note.
        let score =
            [evt("nev", Lang::En), evt("ver", Lang::En), evt("be", Lang::En), evt("leeve", Lang::En)];
        let c = classify_score(&score, &f).unwrap();
        assert!(matches!(c[0], LyricClass::Phones { .. }), "{:?}", c[0]);
        assert!(matches!(c[1], LyricClass::Sustain), "{:?}", c[1]);
        assert!(matches!(c[2], LyricClass::Phones { .. }), "{:?}", c[2]);
        assert!(matches!(c[3], LyricClass::Unknown), "{:?}", c[3]);
    }

    #[test]
    fn fragment_merge_stays_off_alias_tracks() {
        let f = fixtures();
        // On an alias track EN words resolve through the convention, never the dictionary —
        // the pass must not merge them even though "ven" would be dictionary-OOV.
        let mut score = [evt("e", Lang::En), evt("ven", Lang::En)];
        for e in &mut score {
            e.phoneme_set = PhonemeSet::Xsampa;
        }
        let r = resolve_score(&score, &f).unwrap();
        assert!(!r[1].is_sustain, "alias note must stay its own word");
        assert_eq!(phones_of(&r[1]).len(), 3, "v-e-n as alias symbols, not the merged tail");
    }

    #[test]
    fn fragment_merge_longest_join_beats_prefix_theft() {
        let f = fixtures();
        // Review S95R-1's real-dictionary counterexample, pinned in fixture form: trigger "ful"
        // sees BOTH the 2-word forward join ful|fil→"fulfil" (a real word) and the 3-word
        // backward join won|der|ful→"wonderful". Greedy-longest must pick "wonderful", leaving
        // fil|ling as two ordinary words (both in-dictionary → outside merge scope).
        let score = [
            evt("won", Lang::En),
            evt("der", Lang::En),
            evt("ful", Lang::En),
            evt("fil", Lang::En),
            evt("ling", Lang::En),
        ];
        let r = resolve_score(&score, &f).unwrap();
        assert_eq!(phones_of(&r[0]), vec!["w", "ʌ", "n"]);
        assert_eq!(phones_of(&r[1]), vec!["d", "ɝ"]);
        assert_eq!(phones_of(&r[2]), vec!["f", "ə", "l"]);
        assert_eq!(phones_of(&r[3]), vec!["f", "ɪ", "l"], "fil must stay its own word");
        assert_eq!(phones_of(&r[4]), vec!["l", "ɪ", "ŋ"], "ling must stay its own word");
        assert!(r[1].is_sustain && r[2].is_sustain && !r[3].is_sustain && !r[4].is_sustain);
    }

    #[test]
    fn fragment_merge_longest_join_beats_backward_theft() {
        let f = fixtures();
        // Same defect, backward direction: a|mes→"ames" (a real dictionary word) must lose to
        // mes|sage→"message". "a" stays its own word.
        let score = [evt("a", Lang::En), evt("mes", Lang::En), evt("sage", Lang::En)];
        let r = resolve_score(&score, &f).unwrap();
        assert_eq!(phones_of(&r[0]), vec!["ə"], "a must stay its own word");
        assert_eq!(phones_of(&r[1]), vec!["m", "ɛ"]);
        assert_eq!(phones_of(&r[2]), vec!["s", "ə", "dʒ"]);
        assert!(!r[1].is_sustain && r[2].is_sustain);
    }

    #[test]
    fn fragment_merge_trigger_is_faithful_only() {
        let f = fixtures();
        // Review S95R-2: the plural rung must NOT be part of the merge trigger. "mes" parses as
        // me+Z under the rung — if that killed the trigger, mes|sage would sing "meez sage" with
        // no red mark. The trigger asks the FAITHFUL ladder, so the merge still fires…
        let r = resolve_score(&[evt("mes", Lang::En), evt("sage", Lang::En)], &f).unwrap();
        assert_eq!(phones_of(&r[0]), vec!["m", "ɛ"]);
        assert_eq!(phones_of(&r[1]), vec!["s", "ə", "dʒ"]);
        // …and a lone "mes" (no merge partner) stays LOUD: the 3-char base floor refuses me+Z.
        let e = resolve_score(&[evt("mes", Lang::En)], &f).unwrap_err();
        assert!(e.to_string().contains("VOCAL_OOV: mes"), "{e}");
        // The discriminator the floor CANNOT mask (mutation R2): "gues" parses as gue+Z — gue is
        // a 3-char dictionary word, so only the faithful trigger lets gues|sing merge to
        // "guessing" instead of singing "gue-z sing".
        let r = resolve_score(&[evt("gues", Lang::En), evt("sing", Lang::En)], &f).unwrap();
        assert_eq!(phones_of(&r[0]), vec!["ɡ", "ɛ"]);
        assert_eq!(phones_of(&r[1]), vec!["s", "ɪ", "ŋ"]);
    }

    #[test]
    fn fragment_merge_ignores_frames_and_pitch() {
        let f = fixtures();
        // §9.5 load-bearing premise (review S95R-6): `validate_lyrics` classifies with dummy
        // frames=1 / note_num=60, so the merge must reach identical verdicts regardless of
        // frames/pitch — otherwise the editor clears a red mark the render then trips over.
        let real = [
            ScoreEvt { lyric: "nev", note_num: 72, frames: 37, lang: Lang::En, phoneme_input: None, phoneme_set: PhonemeSet::Words },
            ScoreEvt { lyric: "ver", note_num: 48, frames: 3, lang: Lang::En, phoneme_input: None, phoneme_set: PhonemeSet::Words },
            ScoreEvt { lyric: "be", note_num: 60, frames: 9, lang: Lang::En, phoneme_input: None, phoneme_set: PhonemeSet::Words },
            ScoreEvt { lyric: "leeve", note_num: 84, frames: 100, lang: Lang::En, phoneme_input: None, phoneme_set: PhonemeSet::Words },
        ];
        let dummy: Vec<ScoreEvt> = real
            .iter()
            .map(|e| ScoreEvt { note_num: 60, frames: 1, ..e.clone() })
            .collect();
        let r = resolve_core(&real, &f, false).unwrap();
        let c = classify_score(&dummy, &f).unwrap();
        assert!(matches!((&r[0].kind, &c[0]), (ResolvedKind::Phones(_), LyricClass::Phones { .. })));
        assert!(r[1].is_sustain && matches!(c[1], LyricClass::Sustain));
        assert!(matches!((&r[2].kind, &c[2]), (ResolvedKind::Phones(_), LyricClass::Phones { .. })));
        assert!(matches!((&r[3].kind, &c[3]), (ResolvedKind::Unknown, LyricClass::Unknown)));
    }

    // ── S95 EN plural rung: -'s/-s/-es of an in-dictionary base, voice follows the base ──
    #[test]
    fn en_plural_rung_is_last_and_voice_follows_the_base() {
        let f = fixtures();
        let one = |lyric: &str| {
            let r = resolve_score(&[evt(lyric, Lang::En)], &f).unwrap();
            phones_of(&r[0])
        };
        assert_eq!(one("dears"), vec!["d", "ɪ", "ɹ", "z"]); // voiced final → Z
        assert_eq!(one("lights"), vec!["l", "aɪ", "t", "s"]); // voiceless final → S
        assert_eq!(one("roses"), vec!["ɹ", "oʊ", "z", "ɪ", "z"]); // sibilant final → IH0 Z (-es via the e-final base)
        // EN only: de/fr/es/it morphology is not "-s plus voicing" — the rung must not fire there
        let e = resolve_score(&[evt("baums", Lang::De)], &f).unwrap_err();
        assert!(e.to_string().contains("VOCAL_OOV: baums"), "{e}");
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

    /// S99 (S86#8-2): zh was the ONE language left out of the `lookup_candidates` tolerance ladder —
    /// 「Love,」 and 「か、」 resolved, 「长,」 was a hard OOV that aborted the whole render.
    #[test]
    fn zh_tolerates_trailing_punctuation_like_every_other_language() {
        let f = fixtures();
        // the thing that must not come back (non-vacuity: prove the two arms really differ pre-fix)
        assert!(
            !matches!(classify_score(&[evt("长，", Lang::Zh)], &f).unwrap()[0], LyricClass::Unknown),
            "a punctuated hanzi is still being marked OOV"
        );
        for (punctuated, bare) in [("长，", "长"), ("大。", "大"), ("解！", "解"), ("之…", "之"), ("「长」", "长")] {
            let a = resolve_score(&[evt(punctuated, Lang::Zh)], &f).unwrap();
            let b = resolve_score(&[evt(bare, Lang::Zh)], &f).unwrap();
            assert_eq!(phones_of(&a[0]), phones_of(&b[0]), "{punctuated} ≠ {bare}");
        }
        // bare pinyin gets the same courtesy
        assert_eq!(
            phones_of(&resolve_score(&[evt("zhi,", Lang::Zh)], &f).unwrap()[0]),
            vec!["ʈʂ", "ɻ̩"]
        );
        // ★ and the punctuated note now PARTICIPATES in phrase context instead of breaking the window:
        // 了 must read liǎo in 「了，|解」 exactly as it does in 「了|解」. (Before the fix the first note
        // was OOV, so any bake such a segment holds is necessarily already dirty — no extra bump owed.)
        let r = resolve_score(&[evt("了，", Lang::Zh), evt("解", Lang::Zh)], &f).unwrap();
        assert_eq!(phones_of(&r[0]), vec!["l", "iaʊ"], "phrase context must cross the punctuation");
        // …and what must STAY loud: punctuation alone, and more than one hanzi on one note
        for bad in ["，", "。", "长大", "长，大"] {
            assert!(
                matches!(classify_score(&[evt(bad, Lang::Zh)], &f).unwrap()[0], LyricClass::Unknown),
                "{bad} must stay OOV — the ladder trims EDGES, it never invents a reading"
            );
        }
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
            // mirrors the real dictionary's letter-name and abbreviation entries — for LOOKUP only.
            // S94: its 1-vote "D R" onset is vote-gated away here (unlike the real dict's 482-vote
            // D R), so never grow an onset/syllabify assertion onto this fixture (review S94R-4).
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

    /// S99 (S86#8-4) — the four German ß spellings whose ss-homograph is a DIFFERENT word. Runs on the
    /// SHIPPED de.tsv because that is where the fix lives: upstream `german_mfa` is Swiss orthography
    /// with 0 ß keys, so the generator (MBS2H `build_dictionaries.py::DE_SHARP_S_ENTRIES`) adds them as
    /// ordinary faithful keys, and the ß→ss rung of `lookup_candidates` is then never consulted for
    /// them. The Rust side is unchanged — which is exactly what this test has to keep true.
    /// Same loud-SKIP contract as `s94_en_onset_vote_gate` below (de.tsv is a gitignored generated asset).
    #[test]
    fn s99_de_sharp_s_homographs() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/dictionaries/de.tsv");
        let Ok(tsv) = std::fs::read_to_string(&p) else {
            eprintln!("[s99-de-ß] SKIPPED — {} not present (gitignored generated asset; run MBS2H build_dictionaries.py)", p.display());
            return;
        };
        let d = WordDict::from_tsv(Lang::De, &tsv);
        let ph = |w: &str| d.lookup(w).unwrap_or_else(|| panic!("{w} is OOV in de.tsv")).join(" ");
        for (esz, ss, want_esz) in [
            ("maße", "masse", "m aː s ə"),
            ("buße", "busse", "b uː s ə"),
            ("floß", "floss", "f l oː s"),
            ("saß", "sass", "z aː s"),
        ] {
            assert_eq!(ph(esz), want_esz, "{esz}");
            // the whole point: the two spellings must NOT resolve alike any more
            assert_ne!(ph(esz), ph(ss), "{esz} and {ss} collapsed back onto one reading");
        }
        // ⚠ the 5875 ss keys whose primary ALREADY carries a long vowel/diphthong were correct through
        // the ß rung before this and must be untouched by it — no new keys, same answer both ways.
        for (esz, ss) in [("weiß", "weiss"), ("groß", "gross"), ("straße", "strasse"), ("heißt", "heisst"), ("spaß", "spass")] {
            assert_eq!(ph(esz), ph(ss), "{esz}/{ss} must still fold together");
        }
        // ⚠ pre-1996 orthography wrote ß after SHORT vowels too; those fold to the modern ss spelling
        // of the same short word, which was already right — the curated table must not have moved them.
        for (old, modern) in [("daß", "dass"), ("muß", "muss"), ("fluß", "fluss"), ("paßt", "passt"), ("kuß", "kuss")] {
            assert_eq!(ph(old), ph(modern), "{old}/{modern} must still fold together");
        }
    }

    /// S94 onset vote gate — the syllabification half `dictionaries_end_to_end` is structurally
    /// blind to (S92 measured: deleting onset clusters changes ZERO of its assertions). Loads the
    /// REAL en.tsv through the production `from_tsv`, then pins (a) the vote-gated onset set and
    /// (b) full syllable splits for both directions: words the S94 threshold FIXES and words whose
    /// (deliberately kept) loanword clusters must keep cutting exactly as before.
    /// NOT #[ignore] — but with a loud SKIP when the dictionary is absent: en.tsv is a GITIGNORED
    /// generated asset (data/ is ignored; provenance = the MBS2H generator), so a fresh checkout
    /// does not have it and must not turn the whole suite red (review S94R-1 — the first draft
    /// claimed "in-repo bundle resource", which was simply false). On the dev machine the file is
    /// always present and this bites on every `cargo test`, which is the point: a dictionary
    /// regeneration is exactly when it has to.
    #[test]
    fn s94_en_onset_vote_gate() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/dictionaries/en.tsv");
        let Ok(tsv) = std::fs::read_to_string(&p) else {
            eprintln!("[s94-onset-gate] SKIPPED — {} not present (gitignored generated asset; run MBS2H build_dictionaries.py)", p.display());
            return;
        };
        let d = WordDict::from_tsv(Lang::En, &tsv);
        // (a) gated OUT: the sub-6-vote proper-noun clusters AND the curated EN_ONSET_DROP families.
        for gone in [
            "N D", "D N", "M B", "T L", "S B", "K M", "Z M", "D M", "P SH", "L W", "Z B", "M L", "N W",
            "S R", "M R", "Z L", "V R", "K V", "SH T", "HH R", "HH L",
            "T S", // S102 — see the evidence block above EN_ONSET_DROP; this one overturns S92.
        ] {
            assert!(!d.onsets.contains(gone), "{gone} must be vote-gated out of the EN onset set");
        }
        // (a') kept: the S92-verified correct cutters and the weakest NATIVE clusters (6/9 votes).
        for kept in ["ZH", "K N", "TH W", "S P Y", "S K Y", "S T R", "S K W", "SH L"] {
            assert!(d.onsets.contains(kept), "{kept} must stay a legal EN onset");
        }
        // (b) splits the threshold FIXES (before S94 the bracketed consonant opened the NEXT syllable):
        let s = |w: &str| -> Vec<Vec<String>> { syllabify(&d, &d.lookup(w).unwrap()) };
        let want = |spec: &str| -> Vec<Vec<String>> {
            spec.split('|').map(|syl| syl.split_whitespace().map(str::to_string).collect()).collect()
        };
        for (w, spec) in [
            ("candle", "K AE1 N | D AH0 L"),        // was K AE1 | N D AH0 L (n'dour's N D)
            ("window", "W IH1 N | D OW0"),
            ("abandon", "AH0 | B AE1 N | D AH0 N"),
            ("kidney", "K IH1 D | N IY0"),          // was K IH1 | D N IY0 (dniester's D N)
            ("midnight", "M IH1 D | N AY2 T"),
            ("sadness", "S AE1 D | N AH0 S"),
            ("husband", "HH AH1 Z | B AH0 N D"),    // was HH AH1 | Z B AH0 N D (zbigniew's Z B)
            ("always", "AO1 L | W EY2 Z"),          // the S86 `always|+ → ɔ | l w eɪ z` finding (luo's L W)
            ("atlas", "AE1 T | L AH0 S"),           // was AE1 | T L AH0 S (tlingit's T L)
            ("admit", "AH0 D | M IH1 T"),           // was AH0 | D M IH1 T (dmitri's D M)
            ("aimless", "EY1 M | L AH0 S"),         // was EY1 | M L AH0 S (mladic's M L)
            ("understand", "AH2 N | D ER0 | S T AE1 N D"),
            // …and the curated EN_ONSET_DROP families (review S94R-2 follow-up):
            ("classroom", "K L AE1 S | R UW2 M"),   // was K L AE1 | S R UW2 M (sri's S R)
            ("comrade", "K AA1 M | R AE2 D"),       // was K AA1 | M R AE2 D (mraz's M R)
            ("ceaselessly", "S IY1 Z | L AH0 | S L IY0"), // was S IY1 | Z L AH0 | S L IY0 (zloty's Z L)
            ("averaged", "AE1 V | R AH0 JH D"),     // was AE1 | V R AH0 JH D (vrabel's V R)
            ("hashtag", "HH AE1 SH | T AE2 G"),     // was HH AE1 | SH T AE2 G (shtick's SH T)
            ("armrest", "AA1 R M | R EH2 S T"),
            // …and S102's `T S` drop. The first three are the ONLY member of the 464-word capture
            // set anyone was ever measured singing (`outside`, 25× in GTSinger) plus two of the 77
            // that carry en.tsv-internal decomposition evidence for exactly this cut.
            ("outside", "AW1 T | S AY1 D"),         // was AW1 | T S AY1 D; out+side
            ("albertson", "AE1 L | B ER0 T | S AH0 N"), // was …| B ER0 | T S AH0 N; albert+son
            ("bestseller", "B EH1 S T | S EH1 | L ER0"), // was B EH1 S | T S EH1 |…; best+seller
            ("itself", "IH2 T | S EH1 L F"),        // was IH2 | T S EH1 L F; it+self
            ("antsy", "AE1 N T | S IY0"),           // no decomposition, but /ts/ is not an EN onset
        ] {
            assert_eq!(s(w), want(spec), "S94-fixed split for {w}");
        }
        // (b'') the documented KNOWN COST, pinned so it stays a decision instead of decaying into a
        // surprise: reservoir's French vw- onset is miscut by the V W drop, and it STAYS miscut
        // because restoring V W was measured to re-break driveway/lovewell (see EN_ONSET_MIN_VOTES).
        assert_eq!(
            s("reservoir"),
            want("R EH1 | Z AH0 V | W AA2 R"),
            "the accepted V W cost moved — re-run the S94 keep/drop table before shipping this"
        );
        // (b''') S102's own accepted cost, pinned for the same reason: the source languages of these
        // want /ts/ to OPEN the syllable (ja つ, ru ц, and `tsetse` is the one ordinary English word
        // on that side), and after the drop they do not get it. Zero of them occur in any corpus or
        // score on this machine — that is the whole reason the trade was taken. If someone reports
        // one of these singing wrong, the fix is the compound-seam exception named above
        // EN_ONSET_DROP, NOT putting `T S` back (that would re-break outside/itself/bestseller).
        for (w, spec) in [
            ("matsumoto", "M AA0 T | S UW0 | M OW1 | T OW0"),
            ("tsetse", "T S IY1 T | S IY0"),        // word-INITIAL T S is untouched, as always
            ("yeltsin", "Y EH1 L T | S AH0 N"),
        ] {
            assert_eq!(s(w), want(spec), "the accepted S102 T S cost moved for {w}");
        }
        // (b') splits that must NOT move — the kept clusters keep cutting exactly as shipped:
        for (w, spec) in [
            ("asia", "EY1 | ZH AH0"),
            ("technique", "T EH0 | K N IY1 K"),     // K N kept (14 votes, knish-family) — documented
            ("southwest", "S AW2 | TH W EH1 S T"),  // TH W native (thwart)
            ("extra", "EH1 K | S T R AH0"),         // maximal onset itself is alive (S T R, 461 votes)
        ] {
            assert_eq!(s(w), want(spec), "S94-unchanged split for {w}");
        }
        // (c) the S94 en.tsv REGENERATION knives, pinned by explicit lookup. The sampled golden
        // walks every ~41st line, and none of these five landed on a sampled row — so without
        // these pins a regeneration that silently loses the generator knives (an MBS2H revert, a
        // cmudict re-import) would ship the old readings with every Utai test green.
        for (w, spec) in [
            ("even", "IY1 V AH0 N"),   // word-final -en schwa consistency (was IH0 N — sang "i-i")
            ("tears", "T IH1 R Z"),    // curated primary flips: crying, not ripping
            ("wind", "W IH1 N D"),     // the weather noun (winds already led with W IH1 N D Z)
            ("live", "L IH1 V"),       // the verb (lives/lived already led with L IH1 V-)
            ("ba", "B AA1"),           // vocalise syllable, not the letter name "bee-ay"
        ] {
            let got = d.lookup(w).unwrap_or_else(|| panic!("{w} missing from en.tsv"));
            let want: Vec<String> = spec.split_whitespace().map(str::to_string).collect();
            assert_eq!(got, want, "S94 regeneration-knife primary for {w}");
        }
    }

    /// S102 — the onset set's BEHAVIOURAL blast radius, proved by mutation rather than asserted.
    ///
    /// Two things nobody had ever demonstrated, and both were needed before dropping `T S`:
    ///  (1) the legal-onset set is invisible on a note that holds a whole word. `resolve_west_span`
    ///      gives the FINAL consumer every remaining syllable, so a one-note `outside` emits the
    ///      word's phones in dictionary order no matter where the syllables were cut. That is the
    ///      reason this whole question is smaller than it looks — but "by construction" is exactly
    ///      the kind of claim S88 caught being silently false, so it is measured here: the SAME
    ///      score is resolved against two dictionaries that differ ONLY in whether `T S` is a legal
    ///      onset, and the one-note arm must come out identical.
    ///  (2) on a `+` span it is fully visible, and the second arm shows exactly where: `out|side`
    ///      instead of `ou|tside`. If a future refactor made the cut stop reaching the render, arm
    ///      (2) would go green-by-vacuum — the `assert_ne!` is what stops that (S92p: a pinning test
    ///      whose two arms agree is testing nothing).
    #[test]
    fn s102_onset_set_only_moves_multi_note_spans() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/dictionaries/en.tsv");
        let Ok(tsv) = std::fs::read_to_string(&p) else {
            eprintln!("[s102-onset-scope] SKIPPED — {} not present (gitignored generated asset)", p.display());
            return;
        };
        let shipped = WordDict::from_tsv(Lang::En, &tsv);
        assert!(!shipped.onsets.contains("T S"), "precondition: the shipped set has T S dropped");
        let mut pre_s102 = WordDict::from_tsv(Lang::En, &tsv);
        pre_s102.onsets.insert("T S".to_string()); // the ONLY difference between the two dicts
        let source = |w: WordDict| Fixtures { zh: zh_fixture(), en: w, de: de_fixture() };
        let (fx_new, fx_old) = (source(shipped), source(pre_s102));

        // (1) one note holding the whole word — the two onset sets must agree, byte for byte.
        let one = [evt("outside", Lang::En)];
        let a = phones_of(&resolve_score(&one, &fx_new).unwrap()[0]);
        let b = phones_of(&resolve_score(&one, &fx_old).unwrap()[0]);
        assert_eq!(a, b, "a single note must not see the onset set at all");
        assert_eq!(a, vec!["aʊ", "t", "s", "aɪ", "d"], "…and it emits the word in dictionary order");

        // (2) the same word over a `+` span — now the cut is audible, and it moved.
        let span = [evt("outside", Lang::En), evt("+", Lang::En)];
        let rn = resolve_score(&span, &fx_new).unwrap();
        let ro = resolve_score(&span, &fx_old).unwrap();
        let (n0, n1) = (phones_of(&rn[0]), phones_of(&rn[1]));
        let (o0, o1) = (phones_of(&ro[0]), phones_of(&ro[1]));
        assert_ne!((&n0, &n1), (&o0, &o1), "the knife must actually reach the render wire");
        assert_eq!((o0, o1), (vec!["aʊ"], vec!["t", "s", "aɪ", "d"]), "pre-S102: ou|tside");
        assert_eq!((n0, n1), (vec!["aʊ", "t"], vec!["s", "aɪ", "d"]), "S102: out|side");
    }

    /// S101 fr D6 gate over the REAL fr.tsv (same SKIP contract as the S94 gate above — the
    /// dictionary is a gitignored generated asset, a bare checkout must not go red).
    ///
    /// WHY IT EXISTS: before this, French had no dictionary tripwire that runs in `cargo test` at
    /// all. `stage2_matches_python_golden` is hermetic (it reads the COMPILED golden and would stay
    /// green if fr.tsv changed and the golden were never regenerated), and the only test that
    /// cross-checks the two — `dictionaries_end_to_end` — is `#[ignore]`, while `release.ps1` runs a
    /// bare `cargo test`. So an fr.tsv regeneration that silently loses the MBS2H D6 mirror would
    /// have shipped completely green. That is exactly the hole `s94_en_onset_vote_gate` was created
    /// to close on the English side.
    ///
    /// Pins the two invariants the D6 mirror is defined by:
    ///  (a) POSTCONDITION — no French primary emits `ɲ` immediately before `i`. The training npz the
    ///      shipped model was built from contains that bigram ZERO times (fr has 55 `ɲ` total;
    ///      it 30 / es 3 / de 0 on the same measurement), because the training side's
    ///      `repair_d_relabel.py` D6 relabelled it. The dictionary now agrees with the weights.
    ///  (b) NO NEW LEGAL ONSET — see `FR_ONSET_DROP`. `d n` is the one that bites (45 loanwords).
    /// Both are pinned by explicit lookup as well, so a regeneration that loses the knife goes red
    /// with a message naming the cause rather than a bare count.
    #[test]
    fn s101_fr_d6_gate() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/dictionaries/fr.tsv");
        let Ok(tsv) = std::fs::read_to_string(&p) else {
            eprintln!("[s101-fr-d6-gate] SKIPPED — {} not present (gitignored generated asset; run MBS2H build_dictionaries.py)", p.display());
            return;
        };
        let d = WordDict::from_tsv(Lang::Fr, &tsv);
        // (a) the postcondition, over every LINE of the file (variants included, not just primaries).
        let mut offenders = Vec::new();
        for line in tsv.lines() {
            let Some((w, phones)) = line.split_once('\t') else { continue };
            let toks: Vec<&str> = phones.split_whitespace().collect();
            if toks.windows(2).any(|p| p[0] == "ɲ" && p[1] == "i") {
                offenders.push(w);
            }
        }
        assert!(
            offenders.is_empty(),
            "fr.tsv emits ɲ before i in {} entries (e.g. {:?}) — the MBS2H D6 mirror is missing from \
             this build of the dictionary. The model was never trained on that bigram.",
            offenders.len(),
            &offenders[..offenders.len().min(5)]
        );
        // (a') spot-pins, including the four words the training-side D6 was originally verified on.
        for (w, spec) in [
            ("souvenir", "s u v n i ʁ"),
            ("fini", "f i n i"),
            ("venir", "v ə n i ʁ"),
            ("harmonie", "a ʁ m ɔ n i"),
            // magnassini is the internal control: TWO ɲ, only the one before `i` moves.
            ("magnassini", "m a ɲ a s i n i"),
            // and the ɲ that must NOT move — before a non-`i` vowel, and word-final.
            ("agneau", "a ɲ o"),
            ("montagne", "m ɔ̃ t a ɲ"),
        ] {
            let got = d.lookup(w).unwrap_or_else(|| panic!("{w} missing from fr.tsv"));
            let want: Vec<String> = spec.split_whitespace().map(str::to_string).collect();
            assert_eq!(got, want, "fr D6 primary for {w}");
        }
        // (b) the knife creates no new legal onset.
        for gone in FR_ONSET_DROP {
            assert!(!d.onsets.contains(*gone), "{gone} must stay out of the FR onset set");
        }
        // (b') and the cut those drops protect — /d/ belongs to the coda in this loanword family.
        let s = |w: &str| -> Vec<Vec<String>> { syllabify(&d, &d.lookup(w).unwrap()) };
        let want = |spec: &str| -> Vec<Vec<String>> {
            spec.split('|').map(|syl| syl.split_whitespace().map(str::to_string).collect()).collect()
        };
        for (w, spec) in [
            ("sidney", "s i d | n ɛ"),
            ("kidnappe", "c i d | n a p"),
            ("midnight", "m i d | n a j t"),
        ] {
            assert_eq!(s(w), want(spec), "the FR_ONSET_DROP cut for {w}");
        }
        // (b'') the OTHER direction, pinned so it stays a decision: the clusters D6 removes are
        // allowed to go, and these cuts move BECAUSE of that. French wants /s/ and /ɡ/ in the coda.
        for (w, spec) in [("bosnien", "b ɔ s | ɲ ɛ̃"), ("baguenier", "b a ɡ | ɲ e")] {
            assert_eq!(s(w), want(spec), "the post-D6 cut for {w}");
        }
    }

    /// S105 (queue §C2) — the four-language onset gate over the REAL dictionaries. The criterion,
    /// the measurements, the two negative results and every accepted cost live above
    /// `DE_ONSET_KEEP`; this pins the behaviour they justify. Same loud-SKIP contract as the gates
    /// around it (data/dictionaries is a gitignored generated asset).
    ///
    /// Four groups, each able to fail for its OWN reason (S101: one mutation must not satisfy them
    /// all):
    ///  (1) INVENTORY — every keep-list entry is still ATTESTED word-initially, and the number of
    ///      multi-consonant clusters each dictionary observes is unchanged. This is the fail-loud
    ///      half of `observed ∩ KEEP`: a regeneration that invents a cluster moves the count and
    ///      names itself HERE instead of silently becoming a legal onset (the S94 `N D` / S101
    ///      `d n` river). ⚠ If one of these numbers fires, judge the newcomer — do not update it.
    ///  (2) GATED — the high-impact clusters this round removed are out of the onset set.
    ///  (3) KEPT — the clusters that must survive DESPITE one-vote loanword attestation are in it.
    ///      `v r` (1 vote, `Vries`) vs it `k k` (1 vote, `K-Chart`) is the pair that proves this
    ///      cannot be a threshold.
    ///  (4) CUTS — full splits in both directions, including the verdicts the phonotactic template
    ///      got wrong on its own: de `s j` (⟨-tion⟩), fr `p ʎ` (/l/ narrowed before /i/) and it
    ///      `s ɡ` (s impura the dictionary forgot to voice) — plus `uruguay`, where the SAME kind of
    ///      argument was made for `ɡ v` and lost (see the ⛔ note in `DE_ONSET_KEEP`).
    #[test]
    fn s105_west_onset_gate() {
        let read = |name: &str| {
            let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../data/dictionaries")
                .join(name);
            std::fs::read_to_string(&p)
        };
        let (Ok(de_tsv), Ok(fr_tsv), Ok(es_tsv), Ok(it_tsv)) =
            (read("de.tsv"), read("fr.tsv"), read("es.tsv"), read("it.tsv"))
        else {
            eprintln!("[s105-west-onset-gate] SKIPPED — data/dictionaries/{{de,fr,es,it}}.tsv not \
                       present (gitignored generated assets; run MBS2H build_dictionaries.py)");
            return;
        };
        // (name, dict, tsv, keep list, clusters the dictionary OBSERVES, entries the list KEEPS)
        let dicts = [
            ("de", WordDict::from_tsv(Lang::De, &de_tsv), &de_tsv, DE_ONSET_KEEP, 145usize, 42usize),
            ("fr", WordDict::from_tsv(Lang::Fr, &fr_tsv), &fr_tsv, FR_ONSET_KEEP, 295, 101),
            ("es", WordDict::from_tsv(Lang::Es, &es_tsv), &es_tsv, ES_ONSET_KEEP, 192, 63),
            ("it", WordDict::from_tsv(Lang::It, &it_tsv), &it_tsv, IT_ONSET_KEEP, 160, 40),
        ];

        // (1) INVENTORY
        for (name, d, tsv, keep, observed, kept) in &dicts {
            // ★ this one exists because a mutation probe found the hole: without it, ADDING a bogus
            //   cluster to a keep list passed every other assertion in this test (the live-set
            //   comparison below is against the list itself, so it cannot see the list moving).
            //   Editing a curated list is a linguistic decision — it must be announced here.
            assert_eq!(
                keep.len(),
                *kept,
                "{name}: the curated keep list now has {} entries, not {kept}. Adding or removing \
                 one is a per-cluster VERDICT — write its evidence next to the entry (which words \
                 it captures, and which cut each of them wants) before moving this number.",
                keep.len()
            );
            let mut seen: HashSet<String> = HashSet::new();
            for line in tsv.lines() {
                let Some((_, phones)) = line.split_once('\t') else { continue };
                let toks: Vec<&str> = phones.split_whitespace().collect();
                if let Some(vi) = toks.iter().position(|t| d.is_vowel(t)) {
                    if vi >= 2 {
                        seen.insert(toks[..vi].join(" "));
                    }
                }
            }
            assert_eq!(
                seen.len(),
                *observed,
                "{name}: the dictionary now observes {} multi-consonant word-initial clusters, not \
                 {observed} — a regeneration introduced or lost one. JUDGE the newcomer against the \
                 criterion above DE_ONSET_KEEP; do not update this number.",
                seen.len()
            );
            for c in *keep {
                assert!(
                    seen.contains(*c),
                    "{name}: keep-list entry {c:?} is no longer attested word-initially — the \
                     curated list has drifted from the dictionary it was derived on"
                );
            }
            // and the keep list is exactly what survived into the onset set
            let live: HashSet<&str> =
                d.onsets.iter().filter(|c| c.contains(' ')).map(|c| c.as_str()).collect();
            let want: HashSet<&str> = keep.iter().copied().collect();
            assert_eq!(live, want, "{name}: admitted multi-consonant onsets ≠ the keep list");
        }

        // (2) GATED — the knives, by language
        for (name, gone) in [
            ("de", &["s t", "t l", "s f", "s b", "s m", "k s", "t v", "s l", "s p", "s v",
                     "s t ʁ", "s k", "k s j", "s t j", "p s", "m b", "p tʰ", "t kʰ", "ɡ v"][..]),
            ("fr", &["s t", "k t", "s p", "k s", "s k", "s t ʁ", "s m", "k s j", "p t", "ɡ z",
                     "t l", "p s", "s c", "ɡ n"][..]),
            ("es", &["s t̪", "n d̪", "s k", "s p", "m p", "m b", "s t̪ ɾ", "r s", "n t̪ ɾ",
                     "s m", "r l", "l d̪", "t̪ l", "s ʝ", "β r", "b r"][..]),
            ("it", &["n t", "n d", "r s", "m p", "n ɡ", "m b", "k k", "r p", "k s", "n d r",
                     "k t", "d l", "v l", "p s", "m n", "t w"][..]),
        ] {
            let d = &dicts.iter().find(|x| x.0 == name).unwrap().1;
            for c in gone {
                assert!(!d.onsets.contains(*c), "{name}: {c:?} must be GATED out of the onset set");
            }
        }
        // (3) KEPT — the ones a vote threshold would have killed, plus the allophone pairs
        for (name, alive) in [
            ("de", &["ʃ t", "ʃ p ʁ", "s j", "ts j", "pf l", "k v", "ɡ n", "k n"][..]),
            ("fr", &["p ʎ", "ɟ ʁ", "s j", "v ʁ", "t ʁ w", "k ʎ", "v l"][..]),
            ("es", &["θ j", "t̪ ɾ", "m j", "ɡ w", "p l j", "ʃ w"][..]),
            ("it", &["v r", "s t", "s ɡ", "s ɡ r", "ʃ r", "t l", "z b", "s t r"][..]),
        ] {
            let d = &dicts.iter().find(|x| x.0 == name).unwrap().1;
            for c in alive {
                assert!(d.onsets.contains(*c), "{name}: {c:?} must STAY in the onset set");
            }
        }

        // (4) CUTS
        let want = |spec: &str| -> Vec<Vec<String>> {
            spec.split('|').map(|s| s.split_whitespace().map(str::to_string).collect()).collect()
        };
        for (name, cases) in [
            ("de", &[
                // gated: the Fugen-s / compound seam and the *tl gap
                ("fenster", "f ɛ n s | t ɐ"),
                ("geburtstag", "ɡ ə | b ʊ ʁ t s | t aː k"),
                ("deutlich", "d ɔʏ t | l ɪ ç"),
                ("atlas", "a t | l a s"),
                ("abendland", "aː | b n̩ t | l a n t"),
                // …and the one where the "it is just ⟨qu⟩'s voiced twin" argument LOST
                ("uruguay", "uː | ʁ ʊ ɡ | v aj"),
                // kept: ⟨sch⟩ at a morpheme start, and the ⟨-tion⟩ glide
                ("verstehen", "f ɛ | ɐ | ʃ t eː | ə n"),
                ("abduktion", "a p | d ʊ k t | s j oː n"),
            ][..]),
            ("fr", &[
                ("espace", "ɛ s | p a s"),
                ("escalier", "ɛ s | k a | ʎ e"),
                ("athlète", "a t | l ɛ t"),
                ("atlas", "a t | l a s"),
                ("accompli", "a | k ɔ̃ | p ʎ i"),
                ("abbatial", "a | b a | s j a l"),
            ][..]),
            ("es", &[
                ("especifica", "e s | p e | s i | f i | k a"),
                ("fundamentos", "f u n | d̪ a | m e n | t̪ o s"),
                ("abstracción", "a β s | t̪ ɾ a ɣ | θ j o n"),
                ("premio", "p ɾ e | m j o"),
            ][..]),
            ("it", &[
                ("fronte", "f r o n | t e"),
                ("acqua", "a k | k u | a"),
                ("avranno", "a | v r a nː | o"),
                ("questo", "k u | e | s t o"),
                ("mostro", "m o | s t r o"),
                ("disgusto", "d i | s ɡ u | s t o"),
                ("atleta", "a | t l e | t a"),
            ][..]),
        ] {
            let d = &dicts.iter().find(|x| x.0 == name).unwrap().1;
            for (w, spec) in cases {
                let phones = d.lookup(w).unwrap_or_else(|| panic!("{name}: {w} is OOV"));
                assert_eq!(syllabify(d, &phones), want(spec), "{name}: the S105 cut for {w}");
            }
        }

        // (5) ★END TO END — the only group that runs the PRODUCTION path this knife actually
        // changes. Everything above tests `syllabify` in isolation; what the user hears is
        // `resolve_score` distributing those syllables over a multi-note word (`word` + `+`),
        // which is the ONE consumer of the onset set (`resolve_west_span`, g2p.rs). S85 rule 4 —
        // "the user must never be the first to execute a path" — is why this group exists: a
        // cluster verdict that is right in the table and wrong on the wire is exactly the shape
        // that has bitten this repo before.
        struct OneLang(Lang, WordDict);
        impl DictSource for OneLang {
            fn zh(&self) -> Result<&ZhDict> {
                Err(UtaiError::Inference("VOCAL_DICT_MISSING: test".into()))
            }
            fn words(&self, lang: Lang) -> Result<&WordDict> {
                if lang == self.0 {
                    Ok(&self.1)
                } else {
                    Err(UtaiError::Inference("VOCAL_DICT_MISSING: test".into()))
                }
            }
        }
        // Each case is an A/B against a dictionary that differs ONLY in the gate, exactly like
        // `s102_onset_set_only_moves_multi_note_spans` does for English. The `assert_ne!` is the
        // point (S92p: a pinning test whose two arms agree is testing nothing) — it is what would
        // catch a future refactor that stops the onset set from reaching the render.
        for (name, lang, word, before, after) in [
            ("de", Lang::De, "fenster",
             &[&["f", "ɛ", "n"][..], &["s", "t", "ɐ"][..]][..],
             &[&["f", "ɛ", "n", "s"][..], &["t", "ɐ"][..]][..]),
            ("fr", Lang::Fr, "espace",
             &[&["ɛ"][..], &["s", "p", "a", "s"][..]][..],
             &[&["ɛ", "s"][..], &["p", "a", "s"][..]][..]),
            ("it", Lang::It, "fronte",
             &[&["f", "r", "o"][..], &["n", "t", "e"][..]][..],
             &[&["f", "r", "o", "n"][..], &["t", "e"][..]][..]),
            // ⚠ only the FIRST cut moves here: `n t̪` was never in the onset set even pre-S105
            // (nothing attests it word-initially), so `men|tos` was already right — a live example
            // of the `observed ∩ KEEP` limitation noted above (`n t̪ ɾ` is attested, `n t̪` is not).
            ("es", Lang::Es, "fundamentos",
             &[&["f", "u"][..], &["n", "d̪", "a"][..], &["m", "e", "n"][..], &["t̪", "o", "s"][..]][..],
             &[&["f", "u", "n"][..], &["d̪", "a"][..], &["m", "e", "n"][..], &["t̪", "o", "s"][..]][..]),
        ] {
            let tsv = dicts.iter().find(|x| x.0 == name).unwrap().2;
            // pre-S105 = the same dictionary with EVERY observed cluster admitted (`min_votes = 1`)
            let mut pre = WordDict::from_tsv(lang, tsv);
            for line in tsv.lines() {
                let Some((_, phones)) = line.split_once('\t') else { continue };
                let toks: Vec<&str> = phones.split_whitespace().collect();
                if let Some(vi) = toks.iter().position(|t| pre.is_vowel(t)) {
                    if vi >= 2 {
                        pre.onsets.insert(toks[..vi].join(" "));
                    }
                }
            }
            let now = OneLang(lang, WordDict::from_tsv(lang, tsv));
            let old = OneLang(lang, pre);
            let mut score = vec![evt(word, lang)];
            score.extend(std::iter::repeat_with(|| evt("+", lang)).take(after.len() - 1));
            let run = |src: &OneLang| -> Vec<Vec<&'static str>> {
                resolve_score(&score, src)
                    .unwrap_or_else(|e| panic!("{name}: strict resolve of {word} failed: {e:?}"))
                    .iter()
                    .map(phones_of)
                    .collect()
            };
            let (a, b) = (run(&old), run(&now));
            assert_ne!(a, b, "{name}: the S105 gate must actually reach the render wire for {word}");
            assert_eq!(a, *before, "{name}: pre-S105 wire shape for {word}");
            assert_eq!(
                b, *after,
                "{name}: {word} over {} notes came out wrong ON THE WIRE (the table may still be \
                 right — this is the consumer, `resolve_west_span`)",
                after.len()
            );
            // …and on ONE note the onset set is invisible (S102 arm (1)): same phones either way.
            let one = [evt(word, lang)];
            assert_eq!(
                phones_of(&resolve_score(&one, &old).unwrap()[0]),
                phones_of(&resolve_score(&one, &now).unwrap()[0]),
                "{name}: a single note must not see the onset set at all"
            );
        }
    }

    /// S104 — the FR/IT elision rung over the REAL dictionaries. Same loud-SKIP contract.
    ///
    /// WHAT IT PROTECTS: before this rung, `l'amour` / `j'aime` / `t'aime` were `VOCAL_OOV`, and OOV
    /// aborts the WHOLE SEGMENT (`resolve_score` is the strict resolve). Elision is obligatory
    /// French orthography, so that is not an edge case — measured on the shipped fr.tsv, the
    /// lexicalised `clitic+X` rows inherited from MFA cover 209 of the ~220k possible combinations
    /// (0.095%). Every one of the 14 elisions the S86 audit named was still absent today.
    ///
    /// The four groups below fail for four DIFFERENT reasons, on purpose (S101: one mutation must
    /// not be able to satisfy them all):
    ///  (1) RESCUE — the rung fires and the phones are right;
    ///  (2) GUARD — a non-elision apostrophe string stays LOUDLY OOV. Deleting the `elidable_head`
    ///      check turns `o'clock` / `c'mon` / `n'djamena` into silently-sung nonsense;
    ///  (3) ⟨y⟩ vs ⟨oi⟩ — the reason the guard is a LETTER test. Both `oiseau` and `yacht` begin
    ///      with a GLIDE phone, so any phone-based guard must get one of them wrong;
    ///  (4) INERTNESS — the rung never overrides a reading the dictionary already had. This is what
    ///      makes "no successful render can change" a measured fact instead of an argument from
    ///      construction (S88: constructions fail silently when the domain gains a new member).
    #[test]
    fn s104_fr_it_elision_gate() {
        let read = |name: &str| {
            let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../data/dictionaries")
                .join(name);
            std::fs::read_to_string(&p).map_err(|_| p)
        };
        let (Ok(fr_tsv), Ok(it_tsv)) = (read("fr.tsv"), read("it.tsv")) else {
            eprintln!("[s104-elision-gate] SKIPPED — data/dictionaries/{{fr,it}}.tsv not present \
                       (gitignored generated asset; run MBS2H build_dictionaries.py)");
            return;
        };
        let fr = WordDict::from_tsv(Lang::Fr, &fr_tsv);
        let it = WordDict::from_tsv(Lang::It, &it_tsv);

        // (1) RESCUE — each of these hard-aborted the segment before S104.
        for (w, spec) in [
            ("l'amour", "l a m u ʁ"),
            ("l'homme", "l ɔ m"),      // mute h
            ("l'eau", "l o"),
            ("j'aime", "ʒ ɛ m"),
            ("t'aime", "t ɛ m"),
            ("m'aime", "m ɛ m"),
            ("c'était", "s e t ɛ"),
            ("d'amour", "d a m u ʁ"),
            ("l'enfant", "l ɑ̃ f ɑ̃"),
            ("j'entends", "ʒ ɑ̃ t ɑ̃"),
            ("l'hiver", "l i v ɛ ʁ"),  // mute h again, and the ⟨i⟩ the MFA narrowing would palatalise
        ] {
            let got = fr.lookup(w).unwrap_or_else(|| panic!("fr elision {w} still OOV"));
            let want: Vec<String> = spec.split_whitespace().map(str::to_string).collect();
            assert_eq!(got, want, "fr elision reading for {w}");
        }
        // …and the composed word syllabifies like French, not like a vowel-less clitic note.
        assert_eq!(
            syllabify(&fr, &fr.lookup("l'amour").unwrap()),
            vec![vec!["l", "a"], vec!["m", "u", "ʁ"]],
            "l'amour must cut la|mour"
        );

        // (2) GUARD — apostrophe strings that are NOT elisions must stay loudly OOV rather than
        //     silently compose. (`d'artagnan` is deliberately NOT in this list: it composes, and
        //     `d a ʁ t a ɲ ɑ̃` is the correct French reading of it.)
        for w in ["o'clock", "c'mon", "s'more", "n'djamena", "m'bappe", "l'oreal", "y'all"] {
            assert!(fr.lookup(w).is_none(), "fr must stay OOV on the non-elision {w}");
        }
        for w in ["o'brien", "o'connor", "o'neill", "o'clock", "c'mon", "s'more", "rock'n'roll"] {
            assert!(it.lookup(w).is_none(), "it must stay OOV on the non-elision {w}");
        }

        // (3) The exact pair that rules out a phone-based guard: `oiseau` = `w a z o` and `yacht`
        //     also opens on a glide, yet French elides before only one of them. A guard reading the
        //     first PHONE cannot separate these two; the letter test does.
        assert_eq!(fr.map.get("oiseau").map(String::as_str), Some("w a z o"), "fixture assumption");
        assert_eq!(
            fr.lookup("l'oiseau"),
            Some(vec!["l".into(), "w".into(), "a".into(), "z".into(), "o".into()]),
            "⟨oi⟩ elides even though its first phone is a glide"
        );
        assert!(fr.lookup("l'yacht").is_none(), "⟨y⟩ blocks elision");

        // (4) INERTNESS over EVERY apostrophe-bearing key the dictionaries already ship: wherever
        //     the faithful ladder answers, `lookup` must return THAT answer, not a composed one.
        for (tag, d, tsv) in [("fr", &fr, &fr_tsv), ("it", &it, &it_tsv)] {
            let mut checked = 0usize;
            for line in tsv.lines() {
                let Some((w, _)) = line.split_once('\t') else { continue };
                if !w.contains('\'') {
                    continue;
                }
                let key = w.to_lowercase();
                let Some(faithful) = d.lookup_faithful(&key) else { continue };
                assert_eq!(d.lookup(&key), Some(faithful), "{tag}: elision overrode {w}");
                checked += 1;
            }
            // …and the loop really ran (an empty sweep would pass vacuously — S102's "which set is
            // this zero over" rule applied to a test).
            assert!(checked > 300, "{tag}: only {checked} apostrophe keys swept");
        }

        // Italian works the same way and is where the rung is most accurate (1446/1446 of the
        // shipped `clitic+X` rows reproduce exactly when composed — see `WordDict::elision`).
        // (`uomo` is `u o m o` in it.tsv, not `u ɔ m o` — the upstream italian_cv transcription does
        //  not mark the open-mid vowel here. Pinned as it is, not as the IPA chart would have it.)
        for (w, spec) in [("l'uomo", "l u o m o"), ("un'ora", "u n o r a")] {
            let got = it.lookup(w).unwrap_or_else(|| panic!("it elision {w} still OOV"));
            let want: Vec<String> = spec.split_whitespace().map(str::to_string).collect();
            assert_eq!(got, want, "it elision reading for {w}");
        }
    }

    /// S104 — the editor and the render must agree about an elision. `classify_score` is the lenient
    /// resolve behind the red OOV marks in the piano roll and `resolve_score` is the strict one the
    /// render uses; they share `WordDict::lookup`, which is WHY the rung was put there instead of in
    /// `resolve_west_span`. Pinning it makes that a test rather than a code-reading.
    #[test]
    fn s104_elision_agrees_between_editor_and_render() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/dictionaries/fr.tsv");
        let Ok(tsv) = std::fs::read_to_string(&p) else {
            eprintln!("[s104-elision-editor] SKIPPED — {} not present", p.display());
            return;
        };
        let src = FrOnly(WordDict::from_tsv(Lang::Fr, &tsv));
        let score = [evt("l'amour", Lang::Fr)];
        let rendered = resolve_score(&score, &src).expect("the render must not abort on an elision");
        assert_eq!(phones_of(&rendered[0]), vec!["l", "a", "m", "u", "ʁ"]);
        let classed = classify_score(&score, &src).unwrap();
        assert!(
            matches!(classed[0], LyricClass::Phones { .. }),
            "the editor still marks l'amour unknown while the render sings it — editor/render split \
             (got {:?})",
            classed[0]
        );
        // …and the same call really did mark an OOV neighbour, so the assertion above is not
        // passing because `classify_score` stopped reporting anything (S102: which set is the zero over).
        let control = classify_score(&[evt("zzqqxx'zzqq", Lang::Fr)], &src).unwrap();
        assert!(matches!(control[0], LyricClass::Unknown), "control: {:?}", control[0]);
    }

    /// S104 (b)-G3 — `yeux` must sing the WORD, not the phrase *les yeux*.
    ///
    /// Upstream ships two rows and `build_mfa` sorts by MFA's probability column, which put the
    /// liaison reading `l e z j ø` first; the Rust loader keeps the first row, so `les | yeux` on
    /// two notes sang `l e | l e z j ø` — "le-lezyeu". MBS2H `FR_PRIMARY_OVERRIDES` now flips it,
    /// REORDER ONLY (the liaison row is still there for a future per-note reading selector).
    ///
    /// This gate exists because the flip lives in the generator: a regeneration that loses it, or
    /// an upstream that respells the wanted variant, must go red here rather than ship green — the
    /// exact hole `s94_en_onset_vote_gate` / `s101_fr_d6_gate` were created to close for their own
    /// knives. Same loud-SKIP contract.
    #[test]
    fn s104_fr_yeux_primary_is_the_word_not_the_phrase() {
        let Some(tsv) = fr_tsv() else {
            eprintln!("[s104-yeux] SKIPPED — data/dictionaries/fr.tsv not present (gitignored)");
            return;
        };
        let src = FrOnly(WordDict::from_tsv(Lang::Fr, &tsv));

        // (a) the reading itself.
        assert_eq!(
            src.0.lookup("yeux"),
            Some(vec!["j".to_string(), "ø".to_string()]),
            "yeux must be /jø/ — if this is `l e z j ø` the MBS2H FR_PRIMARY_OVERRIDES flip is gone"
        );

        // (b) REORDER ONLY: the liaison reading is demoted, never deleted. (A future per-note
        //     reading selector needs it, and `EN_PRIMARY_OVERRIDES` makes the same promise.)
        let rows: Vec<&str> = tsv
            .lines()
            .filter_map(|l| l.split_once('\t'))
            .filter(|(w, _)| *w == "yeux")
            .map(|(_, p)| p)
            .collect();
        assert_eq!(rows, vec!["j ø", "l e z j ø"], "yeux rows: flipped, not pruned");

        // (c) the behaviour that was actually wrong — two ordinary notes of a French lyric line.
        let score = [evt("les", Lang::Fr), evt("yeux", Lang::Fr)];
        let r = resolve_score(&score, &src).unwrap();
        assert_eq!(phones_of(&r[0]), vec!["l", "e"]);
        assert_eq!(
            phones_of(&r[1]),
            vec!["j", "ø"],
            "`les | yeux` must not re-sing the article on the second note"
        );
    }

    // ─── S102: the three dictionaries that had NO tripwire at all ────────────────────────────────
    //
    // THE HOLE (found in S101, and S101 only closed the French half of it): the golden test
    // `stage2_matches_python_golden` is hermetic — it reads the COMPILED golden, so changing a .tsv
    // leaves it green; the ONE test that cross-checks golden against the shipped .tsv
    // (`dictionaries_end_to_end`) is `#[ignore]`; and `release.ps1` runs a bare `cargo test`.
    // ⇒ a regeneration that silently loses a generator knife, or picks up a drifted upstream, SHIPS
    // ALL GREEN. en has had a gate since S94, de since S99, fr since S101. es / it / zh had none.
    //
    // Shape copied from `s101_fr_d6_gate`: (a) the generator's POSTCONDITION over every line,
    // (b) spot-pinned primaries, (c) an upstream-drift fingerprint whose message names the cause.
    // Same loud-SKIP contract — data/dictionaries is a gitignored generated asset (MBS2H
    // `build_dictionaries.py`), so a bare checkout must not turn the suite red.
    //
    // ⚠ The exact counts in (c) are DELIBERATELY brittle. They exist to go red when the upstream
    // file changes underneath us — which is the failure S98 proved is otherwise invisible (a
    // different `german_mfa` release silently rewrote 3051 primary readings while the row count,
    // the word set and the file size all matched). If one of them fires, the answer is to diff the
    // regeneration against a snapshot (`verify_dictionaries.py --against`), not to update the number.

    /// The generator's only Italian knife is `MFA_SUBS` (it is the one language where it actually
    /// fires: ɡː×130, vː×324, ã×9, ẽ×1, ħ×1 at the S102 measurement). Everything else Italian comes
    /// from upstream `italian_cv.dict` verbatim, so the fingerprint carries the weight here.
    #[test]
    fn s102_it_dictionary_gate() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/dictionaries/it.tsv");
        let Ok(tsv) = std::fs::read_to_string(&p) else {
            eprintln!("[s102-it-gate] SKIPPED — {} not present (gitignored generated asset; run MBS2H build_dictionaries.py)", p.display());
            return;
        };
        let d = WordDict::from_tsv(Lang::It, &tsv);
        // (a) MFA_SUBS postcondition: none of the substituted tokens may survive into the product.
        for dead in ["vː", "ɡː", "ã", "ẽ", "ħ"] {
            let n = tsv.lines().filter(|l| l.split_whitespace().any(|t| t == dead)).count();
            assert_eq!(n, 0, "it.tsv still emits {dead} in {n} lines — MBS2H MFA_SUBS did not run");
        }
        // (b) spot pins. `canzone`/`grazie`/`zucchero` are pinned on purpose even though their `s`
        //     for single ⟨z⟩ is the OPEN question (queue (b)2, "it z→ts"): pinning current behaviour
        //     is what makes a change to it VISIBLE instead of silent.
        for (w, spec) in [
            ("mezzo", "m e t͡s o"),        // ⟨zz⟩ = the real affricate token
            ("pizza", "p i t͡s a"),
            ("canzone", "k a n s o n e"), // single ⟨z⟩ → s  (OPEN: queue (b)2)
            ("zucchero", "s u kː e r o"),
        ] {
            let got = d.lookup(w).unwrap_or_else(|| panic!("{w} missing from it.tsv"));
            let want: Vec<String> = spec.split_whitespace().map(str::to_string).collect();
            assert_eq!(got, want, "it primary for {w}");
        }
        // (c) upstream-drift fingerprint + the case-collapse fact.
        //     it.tsv is the ONE dictionary whose keys are capitalised, and `from_tsv` lowercases
        //     them — so 76963 rows become 66881 keys, first row winning. That is where the queue's
        //     "it 66,881 条是五语最少" comes from; if the case convention upstream ever changes,
        //     this number moves and a lot of readings silently change which variant is primary.
        let rows = tsv.lines().filter(|l| l.contains('\t')).count();
        assert_eq!(rows, 76963, "it.tsv row count moved — upstream italian_cv.dict drifted");
        assert_eq!(d.map.len(), 66881, "it.tsv case-collapsed key count moved (rows {rows})");
        let affr = tsv.lines().filter(|l| l.split_whitespace().any(|t| t == "t͡s")).count();
        assert_eq!(affr, 1316, "it.tsv rows carrying the t͡s affricate moved — upstream drifted");
    }

    /// Spanish has NO generator knife of its own (`MFA_SUBS` fires zero times on it), so this gate
    /// is entirely an upstream-drift detector — plus the θ pins, which guard a DECISION: S100
    /// measured that [θ] is real in the corpus and that our pipeline renders θ/s apart, and
    /// explicitly REFUSED a θ→s remap. If a regeneration ever flattens θ, that refusal must go red
    /// here rather than ship silently.
    #[test]
    fn s102_es_dictionary_gate() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/dictionaries/es.tsv");
        let Ok(tsv) = std::fs::read_to_string(&p) else {
            eprintln!("[s102-es-gate] SKIPPED — {} not present (gitignored generated asset; run MBS2H build_dictionaries.py)", p.display());
            return;
        };
        let d = WordDict::from_tsv(Lang::Es, &tsv);
        for dead in ["vː", "ɡː", "ã", "ẽ", "ħ"] {
            let n = tsv.lines().filter(|l| l.split_whitespace().any(|t| t == dead)).count();
            assert_eq!(n, 0, "es.tsv emits {dead} in {n} lines — MBS2H MFA_SUBS did not run");
        }
        for (w, spec) in [
            ("cielo", "θ j e l o"),          // distinción — the S100 decision lives on these
            ("corazón", "k o ɾ a θ o n"),
            ("zapato", "θ a p a t̪ o"),
            ("llorar", "ʎ o ɾ a ɾ"),         // and the yeísmo axis (S86#3)
            ("caballo", "k a β a ʝ o"),
        ] {
            let got = d.lookup(w).unwrap_or_else(|| panic!("{w} missing from es.tsv"));
            let want: Vec<String> = spec.split_whitespace().map(str::to_string).collect();
            assert_eq!(got, want, "es primary for {w}");
        }
        let rows = tsv.lines().filter(|l| l.contains('\t')).count();
        assert_eq!(rows, 95998, "es.tsv row count moved — upstream spanish_mfa drifted");
        assert_eq!(d.map.len(), 90319, "es.tsv key count moved (rows {rows})");
        for (tok, want) in [("θ", 14064), ("ʎ", 2147), ("ɟʝ", 598), ("ʝ", 2241)] {
            let n = tsv.lines().filter(|l| l.split_whitespace().any(|t| t == tok)).count();
            assert_eq!(n, want, "es.tsv rows carrying {tok} moved — upstream drifted, or a remap \
                                 that S100 explicitly refused has been introduced");
        }
    }

    /// Chinese is the one built from THREE files, and its knives are all in the syllable table:
    /// the bare-final folding (yi→i, ya→ia …), the apical-i split, and the two DECIDED
    /// substitutions for finals M4Singer never trained (`weng`→ong, `yo`→o). None of it had a test.
    #[test]
    fn s102_zh_dictionary_gate() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/dictionaries");
        let read = |n: &str| std::fs::read_to_string(dir.join(n));
        let (Ok(syl), Ok(chars), Ok(phrases)) =
            (read("zh_syllables.tsv"), read("zh_chars.tsv"), read("zh_phrases.tsv"))
        else {
            eprintln!("[s102-zh-gate] SKIPPED — zh dictionaries not present in {} (gitignored generated assets)", dir.display());
            return;
        };
        let d = ZhDict::from_tsv(&syl, &chars, &phrases);
        let phones = |s: &str| d.syllable_phones(s).unwrap_or_else(|| panic!("{s} missing from zh_syllables.tsv")).join(" ");
        // (a) the two DECIDED substitutions — finals absent from the M4Singer inventory, mapped to
        //     the nearest TRAINED final. Losing them means asking the model for a final it never saw.
        assert_eq!(phones("weng"), "ong", "the zh weng→ong substitution is missing");
        assert_eq!(phones("yo"), "o", "the zh yo→o substitution is missing");
        // (b) the bare-final folding: y-/w- initials are spelling, not phonology, in this convention.
        for (syl, want) in [("yi", "i"), ("ya", "ia"), ("you", "iu"), ("yan", "ian"),
                            ("yong", "iong"), ("wo", "uo")] {
            assert_eq!(phones(syl), want, "the zh bare-final folding for {syl}");
        }
        // (c) the apical-i split stays a two-phone syllable (initial + final), not a fused token.
        assert_eq!(phones("zhi"), "zh i");
        assert_eq!(phones("si"), "s i");
        // (d) the four syllables the generator reports as unmapped must stay ABSENT — if one of them
        //     silently gains a reading, some hanzi starts singing a final nobody chose.
        for gone in ["hm", "hng", "m", "wong"] {
            assert!(d.syllable_phones(gone).is_none(), "{gone} must stay unmapped in zh_syllables.tsv");
        }
        // (e) upstream-drift fingerprint across all three files.
        let n = |t: &str| t.lines().filter(|l| l.contains('\t')).count();
        assert_eq!(n(&syl), 422, "zh_syllables.tsv row count moved");
        assert_eq!(n(&chars), 44434, "zh_chars.tsv row count moved — kMandarin/pinyin drifted");
        assert_eq!(n(&phrases), 47111, "zh_phrases.tsv row count moved — phrase_pinyin drifted");
        // …and one real lookup through each of the other two tables, so the loader path is exercised
        // rather than just the file contents.
        assert!(d.is_hanzi('中'), "zh_chars.tsv did not load");
        assert_eq!(d.char_default('中'), Some("zhong"));
    }

    /// S95 fragment-merge / plural-rung gate over the REAL en.tsv (same SKIP contract as the S94
    /// gate above: the dictionary is a gitignored generated asset — a bare checkout must not go
    /// red). Pins the TYFD material's verdicts end-to-end: which fragments merge, into what,
    /// which stay loud — AND the join strings whose ABSENCE the window order silently depends on
    /// (a regeneration that gains "gether" would re-route to|get|ther through get+ther).
    #[test]
    fn s95_fragment_merge_e2e_gate() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/dictionaries/en.tsv");
        let Ok(tsv) = std::fs::read_to_string(&p) else {
            eprintln!(
                "[s95-merge-gate] SKIPPED — {} not present (gitignored generated asset; run MBS2H build_dictionaries.py)",
                p.display()
            );
            return;
        };
        struct EnOnly(WordDict);
        impl DictSource for EnOnly {
            fn zh(&self) -> Result<&ZhDict> {
                Err(UtaiError::Inference("VOCAL_DICT_MISSING: test".into()))
            }
            fn words(&self, lang: Lang) -> Result<&WordDict> {
                if lang == Lang::En {
                    Ok(&self.0)
                } else {
                    Err(UtaiError::Inference("VOCAL_DICT_MISSING: test".into()))
                }
            }
        }
        let src = EnOnly(WordDict::from_tsv(Lang::En, &tsv));
        // (a) order-load-bearing ABSENCES: shorter windows probe these joins BEFORE the winning
        // one. If a dictionary regeneration ever ships one, the merge re-routes — re-judge.
        for absent in
            ["gether", "getther", "nevver", "givving", "willnev", "mydears", "beleeve", "togetther", "amessage"]
        {
            assert!(
                src.0.lookup(absent).is_none(),
                "en.tsv now resolves {absent:?} — the S95 window order routes through it, re-judge the merge pins"
            );
        }
        // (b) the TYFD fragments end-to-end through the production resolver:
        let ph = |lyrics: &[&str]| -> Vec<Vec<&'static str>> {
            let score: Vec<ScoreEvt> = lyrics.iter().map(|l| evt(l, Lang::En)).collect();
            resolve_score(&score, &src).unwrap().iter().map(phones_of).collect()
        };
        assert_eq!(ph(&["e", "ven"]), [vec!["i"], vec!["v", "ə", "n"]]); // S94 even-primary
        assert_eq!(ph(&["nev", "ver"]), [vec!["n", "ɛ"], vec!["v", "ɝ"]]);
        assert_eq!(ph(&["to", "get", "ther"]), [vec!["t", "ə"], vec!["ɡ", "ɛ"], vec!["ð", "ɝ"]]);
        assert_eq!(ph(&["giv", "ving"]), [vec!["ɡ", "ɪ"], vec!["v", "ɪ", "ŋ"]]);
        assert_eq!(ph(&["dears"]), [vec!["d", "ɪ", "ɹ", "z"]]); // plural rung, S94 tears-family vowel
        // (c) a respelling no join can rescue stays LOUD on the OOV note:
        let score: Vec<ScoreEvt> = ["be", "leeve"].iter().map(|l| evt(l, Lang::En)).collect();
        let e = resolve_score(&score, &src).unwrap_err().to_string();
        assert!(e.contains("VOCAL_OOV: leeve"), "{e}");
        // (d) scope boundary: real-word double-consonant pairs stay two words:
        let r = resolve_score(&[evt("look", Lang::En), evt("king", Lang::En)], &src).unwrap();
        assert!(!r[1].is_sustain, "look|king must stay two independent words");
        // (e) review S95R-1's theft shapes on the LIVE dictionary. Backward theft: a|mes→"ames"
        // (a real cmudict name) must lose to mes|sage→"message" under greedy-longest:
        assert_eq!(ph(&["a", "mes", "sage"]), [vec!["ə"], vec!["m", "ɛ"], vec!["s", "ə", "dʒ"]]);
        // Forward case, pinned HONESTLY: on the live dictionary the longest join for
        // won|der|ful|fil|ling is "fulfilling" (10 chars, beats "wonderful"'s 9), so won|der
        // stay their own words. Both parses are dictionary-valid segmentations of the author's
        // "wonderful filling" and land within a schwa of it phonetically; dictionary-only
        // merging cannot rank them further (per-note phoneme control is the manual override).
        // What this pin guards is the MECHANISM: the 2-word thief "fulfil" (in-dictionary!)
        // must never win — the fixture test pins the wonderful-wins shape where no longer
        // join exists.
        assert_eq!(
            ph(&["won", "der", "ful", "fil", "ling"]),
            [
                vec!["w", "ʌ", "n"],
                vec!["d", "ɝ"],
                vec!["f", "ʊ", "l"],
                vec!["f", "ɪ"],
                vec!["l", "ɪ", "ŋ"]
            ]
        );
        // (f) reviews S95R-2/R-3: a lone plural-parsable fragment stays LOUD — the merge trigger
        // ignores the plural rung, and the 3-char base floor refuses me+Z:
        let score: Vec<ScoreEvt> = ["mes"].iter().map(|l| evt(l, Lang::En)).collect();
        let e = resolve_score(&score, &src).unwrap_err().to_string();
        assert!(e.contains("VOCAL_OOV: mes"), "{e}");
        // …and the live-dictionary R-2 discriminator the floor cannot mask: "gues" parses as
        // gue+Z (gue = G Y UW1, 3 chars), so only the faithful trigger merges gues|sing:
        assert_eq!(ph(&["gues", "sing"]), [vec!["ɡ", "ɛ"], vec!["s", "ɪ", "ŋ"]]);
    }

    // ── #[ignore] DIAGNOSTIC PROBE (S86 dictionary work-line): run the REAL engine over the REAL
    //    shipped dictionaries so every audit finding is grounded in behaviour, not in reading code.
    //      UTAI_G2P_PROBE=<file>  each non-empty, non-`#` line is  <lang>[:<set>] TAB <note>|<note>|...
    //      <set> ∈ words|arpasing|xsampa|vccv (absent = words). S99 added it: without a convention
    //      selector the probe could only ever see the dictionary arm, so every question about the
    //      ALIAS tokenizer had to be answered by reading code — exactly the habit this probe exists
    //      to replace. An unrecognised <set> is LOUD here (production's `from_wire` folds unknown to
    //      `words` on purpose — for a diagnostic that same fold would silently answer the wrong question).
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
            let (lang_code, set_code) = match code.trim().split_once(':') {
                Some((l, s)) => (l, s),
                None => (code.trim(), "words"),
            };
            let set = match set_code {
                "words" => PhonemeSet::Words,
                "arpasing" => PhonemeSet::Arpasing,
                "xsampa" => PhonemeSet::Xsampa,
                "vccv" => PhonemeSet::Vccv,
                other => {
                    println!("!! unknown phoneme set {other:?} (want words|arpasing|xsampa|vccv): {line}");
                    continue;
                }
            };
            let lang = lang_of(lang_code);
            let lyrics: Vec<&str> = notes.split('|').collect();
            let score: Vec<ScoreEvt> = lyrics.iter().map(|l| evt_set(l, lang, set)).collect();
            println!("\n=== [{code}] {}", lyrics.join(" | "));
            // stage1 + syllabification for the western languages (the whole-word head note).
            // Only meaningful on the WORDS arm: on an alias track the lyric is a symbol string, so a
            // dictionary miss here is expected and printing "OOV — not in en.tsv" would read as a bug.
            if !matches!(lang, Lang::Ja | Lang::Zh) && set == PhonemeSet::Words {
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

    /// S99 (S90 debt): a mistyped PHONEME in a bracket hint / raw override must name THE PHONEME.
    /// Before this, `stage2`'s `Err(phone)` was discarded (`Err(_) => Ok(None)` / `.ok()`) and the
    /// note came out as `VOCAL_OOV: [dh ae zzz]` — a message whose advice ("check the lyric or the
    /// language") points at the two things that are fine.
    #[test]
    fn a_mistyped_phoneme_names_the_phoneme_not_the_lyric() {
        let f = fixtures();
        let bad = [evt("[dh ae zzz]", Lang::En)];
        let e = resolve_score(&bad, &f).unwrap_err().to_string();
        assert!(e.contains("VOCAL_UNKNOWN_PHONE: zzz"), "must name the offending phone: {e}");
        assert!(!e.contains("VOCAL_OOV"), "must NOT be reported as a bad lyric: {e}");
        // the editor still marks exactly that note (a per-note verdict, not an infrastructure error)
        assert!(matches!(classify_score(&bad, &f).unwrap()[0], LyricClass::Unknown));
        // a well-formed hint is untouched
        assert_eq!(phones_of(&resolve_score(&[evt("[dh ae dh]", Lang::En)], &f).unwrap()[0]), vec!["ð", "æ", "ð"]);
        // ★ and an ORDINARY OOV still says VOCAL_OOV with the LYRIC — the two arms must stay distinct
        let e2 = resolve_score(&[evt("zzzzq", Lang::En)], &f).unwrap_err().to_string();
        assert!(e2.contains("VOCAL_OOV: zzzzq"), "{e2}");
        assert!(!e2.contains("VOCAL_UNKNOWN_PHONE"), "{e2}");
        // ja/zh raw-phoneme overrides go through the same naming (they used to `.ok()` it away too)
        let mut ja = evt("か", Lang::Ja);
        ja.phoneme_input = Some("k zzz a");
        let e3 = resolve_score(&[ja], &f).unwrap_err().to_string();
        assert!(e3.contains("VOCAL_UNKNOWN_PHONE: zzz"), "{e3}");
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
        evt_set(lyric, Lang::En, set) // alias conventions are EN-only; one struct literal (`evt_set`)
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
        // ⚠ the readable neighbour used to be `light`; S99 rejects it, and correctly — under X-SAMPA
        // it reads l+i+g+h+t = [l iy g hh t], whose `g hh t` is a 3-consonant cluster no alias of this
        // convention can produce (see `impossible_cluster`). `two` = [t w oʊ] stays legal.
        let bad = [alias_evt("two", PhonemeSet::Xsampa), alias_evt("zq", PhonemeSet::Xsampa)];
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
