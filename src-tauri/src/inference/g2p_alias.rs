//! S91 — UTAU **alias conventions** for English scores (queue 5c).
//!
//! A UST written against an English UTAU voicebank does not carry words: every note's "lyric" is a
//! **sample alias** from that bank's reclist — `-aI`, `e@n`, `y uw`, `&m`, `1ng-`. Three published
//! conventions cover the material we have, and this module turns one alias into the ARPABET phones
//! the note articulates. Nothing here invents IPA: the output is fed straight to
//! [`super::g2p::stage2`], which converts ARPABET through the generated `ARPABET_IPA` table.
//!
//! ## The three conventions
//! | key | what it is | how a lyric looks |
//! |---|---|---|
//! | `arpasing` | OpenUtau ARPAsing — whitespace-separated ARPABET, plus a bare `-` for silence | `- ay`, `ae n`, `y uw`, `ow -` |
//! | `xsampa` | X-SAMPA / **GrayGlish** (GraySlate's CVVC English reclist; OpenUtau's `EN X-SAMPA` symbol set) | `-aI`, `e@n`, `ju`, `N-` |
//! | `vccv` | **VCCV English** (CZ's reclist) | `-&`, `&m`, `yo`, `1ng-` |
//!
//! ## Where the tables come from (they are NOT hand-copied from a standard table)
//! The reference material is one song — *Duvet* (Bôa) — that GraySlate published as **three parallel
//! USTs**, one per convention, on the SAME timeline. Aligning them by (cumulative tick, NoteNum)
//! gives **527 rows** in which the ARPAsing track is literal ARPABET, i.e. ground truth for the other
//! two. Every symbol below carries the support count that alignment gave it. Where the corpus and the
//! published chart disagree, **the corpus wins** and the row says so (S89: the standard tables are
//! about the *notation*, the bank decides the *sound* — copying `e@` = /eə/ out of the X-SAMPA chart
//! would have been wrong 77 times over).
//! Symbols the song never uses are filled from the published charts and marked `[chart]`:
//!  * `xsampa` ← OpenUtau `EnXSampaPhonemizer` (`dictionaryReplacements`: `aa=A ae={ ah=V ao=O ax=@
//!    ay=aI ch=tS dh=D dx=4 eh=E er=3 ey=eI ih=I iy=i jh=dZ ng=N ow=oU oy=OI sh=S th=T uh=U uw=u
//!    y=j zh=Z`) — GrayGlish is built on exactly that phonemizer.
//!  * `vccv` ← CZ's published VCCV English phoneme chart.
//! A `[chart]` row has NOT been heard in a render. They are listed as a group in
//! `project_v2_pending_cleanups` so a future bug report has somewhere to land.
//!
//! ## THE structural rule: an alias is a TRANSITION, so its leading vowel is already sung
//! These banks are diphone systems. A note's sample covers ONE transition and the note SUSTAINS the
//! last symbol; UTAU's `PreUtterance` pulls everything before it ahead of the note start. So in a
//! VC/VV alias (`ae n`, `e@n`, `&m`, `aI e@`) the leading vowel is the vowel the PREVIOUS note is
//! already holding — writing it again would re-articulate it. We therefore **drop a leading symbol
//! whose ARPABET expansion is a single vowel**, unless the alias carries the phrase-onset marker `-`
//! (= "from silence": nothing is carried in).
//!
//! That one rule is what makes the three conventions agree: `ih ng` (ARPAsing), `N` (GrayGlish) and
//! `1ng` (VCCV) all become `[ng]`. Measured over the 527 aligned rows, **460 of 481 sung rows produce
//! byte-identical phones in all three conventions**; every one of the 21 exceptions is a divergence
//! between the three human authors (listed in `alias_cross_convention_equivalence`), not a table
//! disagreement.
//!
//! It is also what keeps the frame allocator honest without touching it. `[ae, n]` on a 3-frame coda
//! note (the modal shape here — 91 % of VC notes are ≤ 5 frames) allocates as nucleus=`ae`,
//! coda=`n`, and the coda budget `min(target, fr-2, 2fr/5)` lands under `CODA_MIN_FRAMES` ⇒ **the `n`
//! is dropped and the note sings only `ae`**. Emitting `[n]` instead makes the note's own phone the
//! nucleus, which is what the bank meant.
//!
//! ## Failure is LOUD (S90)
//! An unknown symbol aborts the note — it never falls back to the word dictionary. That fallback
//! would be the S90 `[dr]`→"drive" pathology at scale: **31 % of the GrayGlish score's lyrics and
//! 38 % of the VCCV score's are also real `en.tsv` keys** (`ju`→"JH UW1", `to`→"T UW1", `E`→"IY1"),
//! so a "try the alias, else look it up" design would sing a different WORD, silently, on a third of
//! the score.

use std::collections::HashMap;
use std::sync::OnceLock;

/// Which alias convention a track's ENGLISH lyrics are written in. `Words` = ordinary spelling
/// through the dictionary — the default, and byte-for-byte the pre-S91 behaviour.
///
/// Per-note (it rides `ScoreEvt` beside `lang`) but sourced per-TRACK: the command layer fans the
/// track's setting out over every note, so there is exactly one place a user sets it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PhonemeSet {
    #[default]
    Words,
    Arpasing,
    Xsampa,
    Vccv,
}

impl PhonemeSet {
    /// The wire spelling, kept next to the enum so the TS union and this can only drift together.
    pub fn as_str(self) -> &'static str {
        match self {
            PhonemeSet::Words => "words",
            PhonemeSet::Arpasing => "arpasing",
            PhonemeSet::Xsampa => "xsampa",
            PhonemeSet::Vccv => "vccv",
        }
    }
    /// Unknown/absent → `Words`: a newer project opened by an older build, or a caller that simply
    /// does not know, must land on the production default rather than on a silently different arm.
    pub fn from_wire(s: Option<&str>) -> PhonemeSet {
        match s.unwrap_or("words") {
            "arpasing" => PhonemeSet::Arpasing,
            "xsampa" => PhonemeSet::Xsampa,
            "vccv" => PhonemeSet::Vccv,
            _ => PhonemeSet::Words,
        }
    }
}

/// X-SAMPA / GrayGlish symbol → ARPABET. Support counts are aligned corpus rows; `[chart]` = filled
/// from OpenUtau's `EnXSampaPhonemizer`, never observed in the reference material.
///
/// ⚠ CASE IS MEANING here (`i`/`I`, `u`/`U`, `s`/`S`, `t`/`T`, `a`/`A`, `e`/`E`, `o`/`O`, `z`/`Z`) —
/// never fold an alias before tokenising it (S90 review major #1: fold the *lookup key*, never the
/// user's phonemes).
const XSAMPA: &[(&str, &str)] = &[
    // ── attested ────────────────────────────────────────────────────────────────────────────────
    ("e@", "ae"),  // 77 — the PRE-NASAL tensed æ. The published chart reads /eə/ (SQUARE); this bank
    //                    uses it for æ before a nasal and `{` elsewhere (76/77 vs 0/25 — a clean
    //                    complementary split), ARPABET has no SQUARE vowel, and the VCCV track writes
    //                    its own pre-nasal symbol `&` on all 77. CORPUS WINS.
    // ⚠ The phonemizer's vowel list ALSO carries `e@n`/`e@m` as single units, and adding them looked
    //   free (identical phones to e@ + n). It is not: an ATOM has ONE symbol, so the carried-vowel
    //   rule stops firing and `e@n` sings "ae n" instead of "n" — the very re-articulation this module
    //   exists to avoid. `alias_cross_convention_equivalence` caught it. Compositional parse only.
    ("{", "ae"),   // 25
    ("aI", "ay"),  // 57   ⚠ must beat `a` + `I` — see MULTI ordering
    ("oU", "ow"),  // 43
    ("eI", "ey"),  // 25
    ("aU", "aw"),  // 6    ⚠ must beat `a` + `U`
    ("tS", "ch"),  // 2    ⚠ must beat `t` + `S`
    ("i", "iy"),   // 58
    ("n", "n"),    // 47
    ("l", "l"),    // 43
    ("m", "m"),    // 37
    ("u", "uw"),   // 33
    ("I", "ih"),   // 32
    ("t", "t"),    // 31
    ("r", "r"),    // 29 — the phonemizer also uses plain `r` for ARPABET R (not the trill), agrees
    ("h", "hh"),   // 28
    ("a", "aa"),   // 27 — this bank writes lowercase `a` for AA; the chart spells that `A` (both below)
    ("V", "ah"),   // 22 — see the ʌ/ə note under `@`
    ("s", "s"),    // 20
    ("d", "d"),    // 21
    ("E", "eh"),   // 19
    ("N", "ng"),   // 19
    ("T", "th"),   // 17 — this bank has NO `D`, so it writes /ð/ as `T` too (19 of its 23 `T` rows are
    //                     really /ð/: the, that, they, there, breathe). We keep the convention's own
    //                     meaning; a score that wants /ð/ should write `D`. Documented in the guide.
    ("j", "y"),    // 16
    ("3", "er"),   // 11
    ("f", "f"),    // 11
    ("b", "b"),    // 8
    ("w", "w"),    // 8
    ("4", "d"),    // 7 — the alveolar flap. ARPABET has none and the ground-truth track writes `d`.
    ("p", "p"),    // 7
    ("v", "v"),    // 6
    ("k", "k"),    // 4
    ("S", "sh"),   // 3
    ("U", "uh"),   // 2
    ("z", "z"),    // 1
    // ── [chart] fills — published EnXSampaPhonemizer, never heard in a render ────────────────────
    ("dZ", "jh"),  // [chart] ⚠ must beat `d` + `Z`
    ("OI", "oy"),  // [chart] ⚠ must beat `O` + `I`
    ("r\\", "r"),  // [chart] the strict X-SAMPA spelling of /ɹ/
    ("3`", "er"),  // [chart] rhotic NURSE
    ("@`", "er"),  // [chart] rhotic schwa
    ("@r", "er"),  // [chart]
    ("A", "aa"),   // [chart] the phonemizer's spelling of AA
    ("O", "ao"),   // [chart]
    ("Q", "aa"),   // [chart] LOT /ɒ/ — General American merges it with AA
    ("@", "ah"),   // [chart] the schwa. Note BOTH `@` and `V` land on ARPABET `ah`, which with no
    //                       stress digit is ə (the S90 rule). The phonemizer maps every CMUdict AH —
    //                       stressed or not — to `V`, so `V` in a real GrayGlish UST means "some AH",
    //                       and S90's asymmetry argument applies unchanged: reading ʌ as ə is a mild
    //                       centralisation, reading ə as ʌ moves the word's perceived stress.
    ("D", "dh"),   // [chart]
    ("Z", "zh"),   // [chart]
    ("g", "g"),    // [chart]
    ("e", "eh"),   // [chart] bare `e`
    ("o", "ow"),   // [chart] bare `o`
    ("1", "ih"),   // [chart] barred i
];

/// VCCV English symbol → ARPABET. Same rules as [`XSAMPA`]; `[chart]` = CZ's published chart.
///
/// ⚠ CASE IS MEANING and the pairs are *different vowels*, with heavy support on both sides:
/// `A`=EY/`a`=AA (28/27), `E`=IY/`e`=EH (58/19), `I`=AY/`i`=IH (55/16), `O`=OW/`o`=UW (45/34).
const VCCV: &[(&str, &str)] = &[
    // ── vowels ──────────────────────────────────────────────────────────────────────────────────
    ("&", "ae"),   // 79 — pre-nasal tensed æ (30/30 followed by a nasal); chart `&`, "band"
    ("@", "ae"),   // 25 — plain æ (11/11 never before a nasal); chart `@`={, "bat"
    ("E", "iy"),   // 58 — chart i:, "bead"
    ("I", "ay"),   // 55 — chart aI, "bike"
    ("O", "ow"),   // 45 — chart @U, "boat"
    ("1", "ih"),   // 45 — chart I before a velar nasal, "blink"
    ("o", "uw"),   // 34 — chart u:, "boot"
    ("A", "ey"),   // 28 — chart eI, "bake"
    ("a", "aa"),   // 27 — chart a, "bot/bar"
    ("u", "ah"),   // 22 — chart V, "but" (see the ʌ/ə note in XSAMPA's `@`)
    ("e", "eh"),   // 19 — chart e, "bet"
    ("i", "ih"),   // 16 — chart I, "bit"
    ("3", "er"),   // 11 — chart @r/3, "bird"
    ("8", "aw"),   // 6  — chart aU, "bout"
    ("6", "uh"),   // 2  — chart U, "book"
    ("9", "ao"),   // [chart] O:, "bought/soft" — the AO this song never uses
    ("Q", "oy"),   // [chart] OI, "boy"
    ("x", "ah"),   // [chart] @, "about" — VCCV's explicit schwa
    ("0", "ao"),   // [chart] "bore/bowl"
    // NO r/l-coloured or nasal ATOMS. `ar`=aa r, `IR`=ay r, `0r`=ao r, `9l`=ao l, `8n`=aw n,
    // `1ng`=ih ng, `1nk`=ih ng k all equal their compositional parse, so they need no row — and the
    // two that do NOT are both deliberately left out:
    //  * `Ar` ("air" = eh r): the corpus's one `Ar-` row is the VC of "share" and wants A+r, which
    //    is also what the ARPAsing and GrayGlish tracks write there.
    //  * `0l` ("bowl" = ow l): S91 review MAJOR. As a one-symbol ATOM it has nsym==1, so the
    //    carried-vowel rule cannot fire and `hO`|`0l`|`ld` re-articulates the [oʊ] the previous note
    //    is still holding — the exact pathology this module exists to prevent, and inconsistent with
    //    the ATTESTED corpus row `("ow l", "oU l", "O l", "l")` for that same word. Compositionally
    //    `0`+`l` reduces to `l`, which IS that row's answer.
    // ── consonants ──────────────────────────────────────────────────────────────────────────────
    ("nng", "ng"), // [chart] "sing"   ⚠ longest first: must beat `n`+`ng` and `nn`+`g`
    ("ng", "ng"),  // 19
    ("nk", "ng k"), // [chart] "thank" — NOT n+k
    ("sh", "sh"),  // 3
    ("ch", "ch"),  // 2
    ("th", "th"),  // 20 — like GrayGlish `T`, this score writes /θ/ AND /ð/ with it; `dh` is the
    //                     chart's voiced member and is honoured below.
    ("dh", "dh"),  // [chart] "this"
    ("zh", "zh"),  // [chart] "genre"
    ("dd", "d"),   // 7  — the flap (X-SAMPA `4` on all 7 rows); ARPABET has none
    ("hh", "hh"),  // 2  — corpus only (`hhE` ↔ ARPAsing `hh iy`); not on the chart
    ("ll", "l"),   // [chart] "sill"
    ("mm", "m"),   // [chart] "sim"
    ("nn", "n"),   // [chart] "sin"
    ("b", "b"),    // 8
    ("d", "d"),    // 23
    ("f", "f"),    // 11
    ("g", "g"),    // [chart] never bare in this score (all 20 `g` are inside `ng`)
    ("h", "hh"),   // 26
    ("j", "jh"),   // [chart] "jet" — VCCV spells the glide `y`, so there is no collision
    ("k", "k"),    // 4
    ("l", "l"),    // 43
    ("m", "m"),    // 37
    ("n", "n"),    // 48
    ("p", "p"),    // 7
    ("r", "r"),    // 31
    ("s", "s"),    // 21
    ("t", "t"),    // 31
    ("v", "v"),    // 6
    ("w", "w"),    // 8
    ("y", "y"),    // 17
    ("z", "z"),    // 1
];

/// Symbols of one table, longest first — THE tokenizer order. A shorter symbol that is a prefix of a
/// longer one must never win: `aI`→`a`+`I` (=`aa ih`), `tS`→`t`+`S`, `1ng`→`1`+`n`+`g` all produce a
/// perfectly well-formed but WRONG phone list, with no error anywhere. 68 of the GrayGlish score's
/// 535 notes take that path if the order is lost, which is why `alias_longest_match_is_load_bearing`
/// pins it.
fn ordered(table: &'static [(&'static str, &'static str)]) -> Vec<&'static str> {
    let mut keys: Vec<&'static str> = table.iter().map(|&(k, _)| k).collect();
    keys.sort_by(|a, b| b.len().cmp(&a.len()).then(a.cmp(b)));
    keys
}

struct Conv {
    map: HashMap<&'static str, &'static str>,
    keys: Vec<&'static str>,
}

fn conv(set: PhonemeSet) -> Option<&'static Conv> {
    static XS: OnceLock<Conv> = OnceLock::new();
    static VC: OnceLock<Conv> = OnceLock::new();
    let build = |t: &'static [(&'static str, &'static str)]| Conv {
        map: t.iter().copied().collect(),
        keys: ordered(t),
    };
    match set {
        PhonemeSet::Xsampa => Some(XS.get_or_init(|| build(XSAMPA))),
        PhonemeSet::Vccv => Some(VC.get_or_init(|| build(VCCV))),
        _ => None,
    }
}

/// Strip the bank's non-phonemic boundary markers. Returns `(core, from_silence)`.
///
/// * leading `-` — phrase onset ("from silence"). GrayGlish: 32 notes, preceded by a rest 32/32.
///   VCCV: 45 notes, phrase-initial 45/45. It is the ONE thing that suppresses the carried-vowel drop.
/// * leading `_` — a recording variant, never a phone. GrayGlish: 73 notes, all a bare sustained
///   vowel; VCCV: 9 notes, all `_rE`/`_r8` where the `r` IS a phoneme (strip only the underscore —
///   eating `_r` loses the /r/).
/// * trailing `-` — release into silence. GrayGlish 31/31 and VCCV 30/30 are phrase-final.
fn strip_markers(alias: &str) -> (&str, bool) {
    let mut s = alias.trim();
    let mut from_silence = false;
    if let Some(rest) = s.strip_prefix('-') {
        from_silence = true;
        s = rest;
    } else if let Some(rest) = s.strip_prefix('_') {
        s = rest;
    }
    if let Some(rest) = s.strip_suffix('-') {
        s = rest;
    }
    (s, from_silence)
}

/// One alias → the ARPABET phones this note articulates, plus how many SYMBOLS the alias had (the
/// carried-vowel rule keys on symbols, not phones: a one-symbol rime like VCCV `0l` is not a VC).
/// `Err` = the offending symbol, for a LOUD failure.
fn symbols_to_phones(set: PhonemeSet, core: &str) -> Result<(Vec<String>, usize), String> {
    let mut phones: Vec<String> = Vec::new();
    let mut nsym = 0usize;
    if set == PhonemeSet::Arpasing {
        // ARPAsing writes ARPABET directly, space separated; a bare `-` token is the silence marker.
        // A stress digit is legal and kept (`ah1` = ʌ, the S90 escape hatch); `sil`/`sp`/`spn` are
        // NOT accepted — this convention spells silence `-`, and letting a phone token mean silence
        // mid-alias would be a silent surprise.
        for tok in core.split_whitespace() {
            if tok == "-" {
                continue;
            }
            if !super::g2p::arpabet_is_known(tok) {
                return Err(tok.to_string());
            }
            phones.push(tok.to_string());
            nsym += 1;
        }
        return Ok((phones, nsym));
    }
    let c = conv(set).expect("conv() covers every non-Words, non-Arpasing set");
    // a space inside an alias is a HARD symbol boundary — never let a multi-char symbol span it
    for run in core.split_whitespace() {
        let mut i = 0usize;
        'next: while i < run.len() {
            for &k in &c.keys {
                if run[i..].starts_with(k) {
                    for p in c.map[k].split(' ') {
                        phones.push(p.to_string());
                    }
                    i += k.len();
                    nsym += 1;
                    continue 'next;
                }
            }
            // unknown: report ONE character (every table key is ASCII, so `i` is always a boundary)
            let bad: String = run[i..].chars().next().into_iter().collect();
            return Err(bad);
        }
    }
    Ok((phones, nsym))
}

/// The longest run of non-nucleus phones a LEGAL alias of these conventions can produce is TWO.
///
/// MEASURED, not assumed. (a) Over the 440 distinct aliases of the three parallel reference scores
/// (arpasing 118 / vccv 153 / xsampa 149, resolved through the production path) the run histogram is
/// 0→55, 1→334, 2→31 and **nothing reaches 3**. (b) The tables corroborate it structurally: exactly
/// ONE row maps a symbol to two phones (`nk` = "ng k"), and these are diphone banks whose alias is at
/// most a two-unit transition — three consonants in a row cannot be assembled out of that.
const MAX_CONSONANT_RUN: usize = 2;

/// S99 (S91 debt): the longest run of consonants in the RESULT, when it exceeds what the convention
/// can produce — `None` when the alias is fine.
///
/// The tokenizer only ever failed on an unknown CHARACTER, and X-SAMPA has 39 single-character keys /
/// VCCV 37 — nearly the whole ASCII alphabet — so "unknown symbol" is effectively unreachable for any
/// all-letter alias (S98 measured 39304 of 140608 three-letter strings passing silently under xsampa).
/// The RESULT got no check at all: no phone-count bound, no phonotactic test, not even a requirement
/// to contain a nucleus. `tth` (the fourth bank's ð) came out as [t t h], `ptk` as [p t k], `kkkkkk`
/// as six /k/ — all sung, silently.
///
/// ⚠ The predicate is on the RESULT, never on the input symbols, and that is deliberate (the user's
/// own criterion, 2026-08-02): splitting an unknown multi-letter symbol into single letters is fine
/// **as long as the split is something this convention could have produced** — only a cluster it
/// CANNOT produce is singing nonsense. So a 2-consonant result stays legal even when it came from a
/// symbol we do not know (`tth` under vccv is t+th = [t θ], an ordinary CC transition, and VCCV really
/// does have geminate rows `ll`/`mm`/`nn`). That residue is hint-level, deliberately NOT an error.
/// ⚠ Nucleus-free aliases are legal and common (168 of the 440 above are VC/CC/C transition units) —
/// requiring a nucleus would reject a third of every real score.
fn impossible_cluster(phones: &[String]) -> Option<String> {
    let mut run: Vec<&str> = Vec::new();
    for p in phones {
        if super::g2p::en_is_nucleus(p) {
            run.clear();
        } else {
            run.push(p);
            if run.len() > MAX_CONSONANT_RUN {
                return Some(run.join(" "));
            }
        }
    }
    None
}

/// Resolve one alias to the space-joined ARPABET phones the note should sing, ready to be handed to
/// the ordinary `phoneme_input` path (→ `stage2` → vocab IPA). `Err(symbol)` = LOUD failure.
///
/// Returns `None` only for `PhonemeSet::Words`, i.e. "not my job — use the dictionary".
pub fn alias_phones(set: PhonemeSet, lyric: &str) -> Option<Result<String, String>> {
    if set == PhonemeSet::Words {
        return None;
    }
    let (core, from_silence) = strip_markers(lyric);
    Some(symbols_to_phones(set, core).and_then(|(mut phones, nsym)| {
        if phones.is_empty() {
            // `-`, `_`, `--`, `  ` … : markers with nothing left. A bare `-`/`` never reaches here
            // (token_class already made it a sustain/rest), so this really is a malformed alias.
            return Err(lyric.trim().to_string());
        }
        // THE carried-vowel rule (see the module header). `from_silence` suppresses it.
        if !from_silence && nsym >= 2 && phones.len() >= 2 && super::g2p::en_is_nucleus(&phones[0]) {
            phones.remove(0);
        }
        // …and only now, on what the note will ACTUALLY sing, ask whether this convention could have
        // produced it. Checking before the carried-vowel rule would measure a shape nobody hears.
        if let Some(cluster) = impossible_cluster(&phones) {
            return Err(cluster);
        }
        Ok(phones.join(" "))
    }))
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::g2p::{stage2, Lang};

    fn ph(set: PhonemeSet, alias: &str) -> String {
        alias_phones(set, alias).expect("not Words").unwrap_or_else(|s| panic!("{alias:?} -> Err({s})"))
    }

    /// ★ THE gate for the tables. The three reference USTs are the SAME song on the SAME timeline in
    /// three conventions, so a row must produce the SAME phones through all three — a symbol mapped
    /// wrong in one table cannot possibly agree with the other two. 142 distinct (ARPAsing, X-SAMPA,
    /// VCCV) triples, covering 460 of the 481 sung aligned notes and 50 distinct alias symbols.
    ///
    /// The 21 notes NOT here are the ones where the three human authors genuinely wrote different
    /// sounds; they are enumerated in `alias_known_author_divergences`, so the exclusion list is
    /// closed and short rather than "whatever did not pass".
    const CROSS: &[(&str, &str, &str, &str)] = &[
    ("- ay",      "-aI",       "-I",        "ay"),
    ("ae",        "_e@",       "&",         "ae"),
    ("ae m",      "e@m",       "&m",        "m"),
    ("ay ae",     "aI e@",     "I&",        "ae"),
    ("ae n",      "e@n",       "&n",        "n"),
    ("iy",        "_i",        "E",         "iy"),
    ("ow",        "_oU",       "O",         "ow"),
    ("y uw",      "ju",        "yo",        "y uw"),
    ("ih ng",     "N",         "1ng",       "ng"),
    ("- ae",      "-e@",       "-&",        "ae"),
    ("ih ng",     "N-",        "1ng-",      "ng"),
    ("m iy",      "mi",        "mE",        "m iy"),
    ("t uw",      "tu",        "to",        "t uw"),
    ("ey",        "_eI",       "A",         "ey"),
    ("ae",        "_{",        "@",         "ae"),
    ("l ih",      "lI",        "l1",        "l ih"),
    ("b r",       "br",        "br",        "b r"),
    ("eh l",      "E l",       "e l",       "l"),
    ("hh eh",     "hE",        "-he",       "hh eh"),
    ("l p",       "lp",        "lp",        "l p"),
    ("r iy",      "ri",        "_rE",       "r iy"),
    ("d ow",      "doU",       "dO",        "d ow"),
    ("hh er",     "h3",        "h3",        "hh er"),
    ("th ae",     "T{",        "th@",       "th ae"),
    ("aa s",      "as",        "as",        "s"),
    ("ae v",      "{v",        "@v",        "v"),
    ("ay hh",     "aI h",      "I h",       "hh"),
    ("d aa",      "4a",        "dda",       "d aa"),
    ("f aa",      "fa",        "fa",        "f aa"),
    ("f ey",      "feI",       "fA",        "f ey"),
    ("hh ae",     "h{",        "h@",        "hh ae"),
    ("l aa",      "la",        "la",        "l aa"),
    ("n ow",      "noU",       "nO",        "n ow"),
    ("ow -",      "oU-",       "O-",        "ow"),
    ("t ih",      "tI",        "ti",        "t ih"),
    ("- ah",      "-V",        "-u",        "ah"),
    ("aa",        "_a",        "a",         "aa"),
    ("aa l",      "a l",       "a l",       "l"),
    ("aw n",      "aU n",      "8 n",       "n"),
    ("d r",       "dr",        "dr",        "d r"),
    ("ey l",      "eI l",      "A l",       "l"),
    ("ey m",      "eIm",       "Am",        "m"),
    ("f iy",      "fi",        "fE",        "f iy"),
    ("hh ow",     "hoU",       "-hO",       "hh ow"),
    ("ih t",      "It",        "it",        "t"),
    ("iy m",      "im",        "Em",        "m"),
    ("iy th",     "iT",        "Eth",       "th"),
    ("iy th",     "iT-",       "Eth-",      "th"),
    ("l uw",      "lu",        "lo",        "l uw"),
    ("n aa",      "na",        "na",        "n aa"),
    ("n d",       "nd-",       "nd-",       "n d"),
    ("n ih",      "nI",        "n1",        "n ih"),
    ("ow n",      "oUn",       "On",        "n"),
    ("r aw",      "raU",       "_r8",       "r aw"),
    ("r iy",      "ri",        "rE",        "r iy"),
    ("s ih",      "sI",        "s1",        "s ih"),
    ("s iy",      "si",        "sE",        "s iy"),
    ("s ow",      "soU",       "sO",        "s ow"),
    ("sh ey",     "SeI",       "shA",       "sh ey"),
    ("t ih",      "tI",        "t1",        "t ih"),
    ("th ah",     "TV",        "thu",       "th ah"),
    ("uw",        "_u",        "o",         "uw"),
    ("uw -",      "u-",        "o-",        "uw"),
    ("ae -",      "{-",        "@-",        "ae"),
    ("ah ch",     "VtS-",      "uch-",      "ch"),
    ("ah t",      "Vt",        "ut",        "t"),
    ("ay",        "_aI",       "I",         "ay"),
    ("d ih",      "4I",        "ddi",       "d ih"),
    ("eh n",      "En",        "en",        "n"),
    ("er t",      "3t",        "3t",        "t"),
    ("hh iy",     "hi",        "hhE",       "hh iy"),
    ("iy n",      "in",        "En",        "n"),
    ("iy r",      "ir-",       "Er-",       "r"),
    ("l ah",      "lV",        "lu",        "l ah"),
    ("m ah",      "mV",        "mu",        "m ah"),
    ("m ay",      "maI",       "mI",        "m ay"),
    ("n iy",      "ni",        "nE",        "n iy"),
    ("n s",       "ns",        "ns",        "n s"),
    ("n t",       "nt",        "nt",        "n t"),
    ("s t",       "st",        "st",        "s t"),
    ("uw",        "oU u",      "o",         "uw"),
    ("v eh",      "vE",        "ve",        "v eh"),
    ("w ah",      "wV",        "wu",        "w ah"),
    ("w ow",      "woU",       "wO",        "w ow"),
    ("y ae",      "j{",        "y@",        "y ae"),
    ("y ow",      "joU",       "yO",        "y ow"),
    ("- aa",      "-a",        "-a",        "aa"),
    ("- uw",      "-u",        "-o",        "uw"),
    ("aa l",      "al",        "al",        "l"),
    ("ae n",      "e@n-",      "&n-",       "n"),
    ("ae t",      "{t",        "@t",        "t"),
    ("ah hh",     "V h",       "u h",       "hh"),
    ("ah l",      "V l",       "u l",       "l"),
    ("ah n",      "Vn",        "un",        "n"),
    ("ay d",      "aI d",      "I d",       "d"),
    ("ay k",      "aI k",      "I k",       "k"),
    ("ay l",      "aI l",      "I l",       "l"),
    ("ay n",      "aIn",       "In",        "n"),
    ("ay r",      "aI r",      "I r",       "r"),
    ("ay t",      "aIt",       "It",        "t"),
    ("b ah",      "bV",        "-bu",       "b ah"),
    ("b ow",      "boU",       "bO",        "b ow"),
    ("d ae",      "de@",       "d&",        "d ae"),
    ("d ah",      "dV",        "du",        "d ah"),
    ("d ay",      "4aI",       "ddI",       "d ay"),
    ("d ay",      "daI",       "dI",        "d ay"),
    ("d er",      "d3",        "d3",        "d er"),
    ("d ih",      "dI",        "d1",        "d ih"),
    ("d iy",      "di",        "dE",        "d iy"),
    ("eh r",      "Er-",       "e r",       "r"),
    ("eh r",      "eIr-",      "Ar-",       "r"),
    ("er n",      "3n",        "3n",        "n"),
    ("ey s",      "eIs",       "As",        "s"),
    ("hh ah",     "hV",        "hu",        "hh ah"),
    ("hh ey",     "heI",       "-hA",       "hh ey"),
    ("hh ow",     "hoU",       "hO",        "hh ow"),
    ("ih l",      "Il",        "il",        "l"),
    ("ih n",      "In",        "in",        "n"),
    ("iy",        "aI i",      "IE",        "iy"),
    ("iy d",      "id",        "Ed",        "d"),
    ("iy r",      "i r",       "E r",       "r"),
    ("k ae",      "ke@",       "-k&",       "k ae"),
    ("k ay",      "kaI",       "kI",        "k ay"),
    ("k uh",      "kU",        "k6",        "k uh"),
    ("l ay",      "laI",       "lI",        "l ay"),
    ("l d",       "ld",        "ld",        "l d"),
    ("l ih",      "lI",        "li",        "l ih"),
    ("l ow",      "loU",       "lO",        "l ow"),
    ("m ae",      "me@",       "m&",        "m ae"),
    ("n eh",      "nE",        "ne",        "n eh"),
    ("ow l",      "oU l",      "O l",       "l"),
    ("p er",      "p3",        "p3",        "p er"),
    ("r ih",      "rI",        "ri",        "r ih"),
    ("r s",       "rs",        "rs",        "r s"),
    ("s ey",      "seI",       "sA",        "s ey"),
    ("t ae",      "te@",       "t&",        "t ae"),
    ("t er",      "t3",        "t3",        "t er"),
    ("uh d",      "Ud",        "6d",        "d"),
    ("w ay",      "waI",       "wI",        "w ay"),
    ("w ih",      "wI",        "-wi",       "w ih"),
    ("w ih",      "wI",        "wi",        "w ih"),
    ("w iy",      "wi",        "wE",        "w iy"),
    ];

    #[test]
    fn alias_cross_convention_equivalence() {
        for &(a, x, v, want) in CROSS {
            assert_eq!(ph(PhonemeSet::Arpasing, a), want, "arpasing {a:?}");
            assert_eq!(ph(PhonemeSet::Xsampa, x), want, "xsampa {x:?}");
            assert_eq!(ph(PhonemeSet::Vccv, v), want, "vccv {v:?}");
        }
    }

    /// The COMPLETE list of aligned notes where the three tracks do not agree — every one traced to a
    /// human choice, none to a table. Pinned so a future table edit that "fixes" one of these has to
    /// come here and say why.
    #[test]
    fn alias_known_author_divergences() {
        let cases: &[(&str, &str, &str, &str, &str, &str)] = &[
            // The X-SAMPA author wrote FLEECE where the word ("thing/things") has KIT; 2 of 3 say ih.
            ("ih", "_i", "1", "ih", "iy", "ih"),
            ("th ih", "Ti", "th1", "th ih", "th iy", "th ih"),
            // Both alias banks spell /ð/ with their /θ/ symbol (neither uses its voiced member in THIS
            // score); ARPAsing can and does write `dh`. See the `T` / `th` table rows.
            ("dh ey", "TeI", "thA", "dh ey", "th ey", "th ey"),
            ("dh eh", "TE", "the", "dh eh", "th eh", "th eh"),
            ("ow dh", "oUT", "Oth", "dh", "th", "th"),
            // The two CVVC tracks carry the /θ/ of "breathe" into the next word instead of restarting.
            ("- ay", "TaI", "thI", "ay", "th ay", "th ay"),
            // The X-SAMPA author's one slip, on the sustain of "man" (`_E` where `_e@` was meant).
            ("ae", "_E", "&", "ae", "eh", "ae"),
            // An ARPAsing-side anomaly on "your ear" (`er hh` where both others read ow + r).
            ("er hh", "oU r", "O r", "hh", "r", "r"),
            // An ARPAsing typo: "things" written `n z` where the word is /ŋz/.
            ("n z", "Nz", "1ngz", "n z", "ng z", "ng z"),
        ];
        for &(a, x, v, pa, px, pv) in cases {
            assert_eq!(ph(PhonemeSet::Arpasing, a), pa, "{a:?}");
            assert_eq!(ph(PhonemeSet::Xsampa, x), px, "{x:?}");
            assert_eq!(ph(PhonemeSet::Vccv, v), pv, "{v:?}");
        }
        // …and the tenth: an ARPAsing typo that is not ARPABET at all, so it fails LOUDLY there while
        // the other two conventions read it as a + l. That asymmetry is correct, not a bug.
        assert_eq!(alias_phones(PhonemeSet::Arpasing, "al"), Some(Err("al".to_string())));
        assert_eq!(ph(PhonemeSet::Xsampa, "al"), "l");
        assert_eq!(ph(PhonemeSet::Vccv, "al"), "l");
    }

    /// ★ Longest-match is LOAD-BEARING and its failure mode is SILENT. Every alternative parse below
    /// is made of symbols that are themselves in the table, so a wrong tokenizer order produces a
    /// well-formed but WRONG phone list with no error anywhere — 68 of the X-SAMPA score's 535 notes.
    #[test]
    fn alias_longest_match_is_load_bearing() {
        // X-SAMPA: aI / aU / tS / e@ / dZ / OI all contain attested single symbols
        assert_eq!(ph(PhonemeSet::Xsampa, "maI"), "m ay"); // NOT m + aa + ih
        assert_eq!(ph(PhonemeSet::Xsampa, "-aI"), "ay"); // NOT aa + ih
        assert_eq!(ph(PhonemeSet::Xsampa, "raU"), "r aw"); // NOT r + aa + uh
        assert_eq!(ph(PhonemeSet::Xsampa, "VtS-"), "ch"); // NOT ah + t + sh (the leading V drops)
        assert_eq!(ph(PhonemeSet::Xsampa, "dZi"), "jh iy"); // NOT d + zh + iy
        assert_eq!(ph(PhonemeSet::Xsampa, "-OI"), "oy"); // NOT ao + ih
        assert_eq!(ph(PhonemeSet::Xsampa, "e@n"), "n"); // e@ then n, NOT eh + ah + n
        // VCCV: every digraph is spelled out of letters that are singles on their own
        assert_eq!(ph(PhonemeSet::Vccv, "1ng"), "ng"); // NOT ih + n + g
        assert_eq!(ph(PhonemeSet::Vccv, "-1ng"), "ih ng"); // …and the onset marker keeps the vowel
        assert_eq!(ph(PhonemeSet::Vccv, "-nng"), "ng"); // nng, NOT n + ng and NOT nn + g
        assert_eq!(ph(PhonemeSet::Vccv, "-shA"), "sh ey"); // NOT s + hh + ey
        assert_eq!(ph(PhonemeSet::Vccv, "-th@"), "th ae"); // NOT t + hh + ae
        assert_eq!(ph(PhonemeSet::Vccv, "-dh@"), "dh ae"); // NOT d + hh + ae
        assert_eq!(ph(PhonemeSet::Vccv, "-dda"), "d aa"); // the flap, NOT d + d + aa
        assert_eq!(ph(PhonemeSet::Vccv, "-hhE"), "hh iy"); // NOT hh + hh + iy
        assert_eq!(ph(PhonemeSet::Vccv, "@nk"), "ng k"); // nk = ŋk, NOT n + k (the leading @ drops)
        // ★ a SPACE inside an alias is a HARD boundary — no symbol may span it. Nothing in the
        // reference corpus distinguishes this (review S91 found the invariant had zero coverage: a
        // "normalise the whitespace away" refactor ships green and then silently reads `n g` as ŋ).
        // The `-` keeps the leading symbol so the carried-vowel rule cannot mask the difference.
        assert_eq!(ph(PhonemeSet::Vccv, "-n g"), "n g"); // NOT the `ng` digraph
        assert_eq!(ph(PhonemeSet::Xsampa, "-t S"), "t sh"); // NOT the `tS` digraph
    }

    /// The carried-vowel rule, in all four shapes it has to tell apart.
    #[test]
    fn alias_drops_only_the_carried_leading_vowel() {
        // VC / VV: the leading vowel belongs to the PREVIOUS note (which is already holding it)
        assert_eq!(ph(PhonemeSet::Arpasing, "ae n"), "n");
        assert_eq!(ph(PhonemeSet::Arpasing, "ay ae"), "ae");
        assert_eq!(ph(PhonemeSet::Vccv, "&m"), "m");
        // CV / CC: the leading symbol is a real onset — never dropped
        assert_eq!(ph(PhonemeSet::Arpasing, "y uw"), "y uw");
        assert_eq!(ph(PhonemeSet::Arpasing, "s t"), "s t");
        assert_eq!(ph(PhonemeSet::Xsampa, "br"), "b r");
        // a ONE-symbol alias is never a transition, even when it expands to several phones
        assert_eq!(ph(PhonemeSet::Arpasing, "ae"), "ae");
        assert_eq!(ph(PhonemeSet::Vccv, "nk"), "ng k");
        // …and an r/l-coloured chart "atom" must NOT be one, or the rule stops firing on the very
        // shape it exists for: `hO`|`0l`|`ld` ("hold") would re-articulate the [oʊ] note 1 still
        // holds. Compositional `0`+`l` gives the same answer as the ATTESTED `O l` row. (Review S91.)
        assert_eq!(ph(PhonemeSet::Vccv, "0l"), "l");
        assert_eq!(ph(PhonemeSet::Vccv, "-0l"), "ao l", "…while from silence it keeps its vowel");
        assert_eq!(ph(PhonemeSet::Vccv, "Ar-"), "r", "same for `Ar`: the corpus row wants A + r");
        // the phrase-onset marker means "from silence": nothing is carried in, so nothing drops
        assert_eq!(ph(PhonemeSet::Arpasing, "- ay"), "ay");
        assert_eq!(ph(PhonemeSet::Xsampa, "-e@n"), "ae n");
        assert_eq!(ph(PhonemeSet::Vccv, "-&n"), "ae n");
        // …while `_` (a recording variant) and a TRAILING `-` (release) do NOT suppress it
        assert_eq!(ph(PhonemeSet::Xsampa, "e@n-"), "n");
        assert_eq!(ph(PhonemeSet::Vccv, "_rE"), "r iy"); // strip ONLY the underscore — the r is a phone
        // ⚠ `_rE` alone cannot tell `_` from `-`: its first phone is a consonant, so the drop branch
        // is never reached either way. THIS is the input that distinguishes them (review S91: the two
        // prefixes look alike and a "unify them" refactor was passing every test).
        assert_eq!(ph(PhonemeSet::Xsampa, "_e@n"), "n");
        assert_eq!(ph(PhonemeSet::Xsampa, "-e@n"), "ae n");
    }

    /// Case is meaning in both alias tables (S90: fold the LOOKUP KEY, never the user's phonemes).
    #[test]
    fn alias_case_is_meaning() {
        // the X-SAMPA pairs ARPABET can actually tell apart (`a`/`A` and `e`/`E` both collapse — the
        // vocab has one AA and one EH — which is a fact about ARPABET, not a table slip)
        for (lower, upper) in [("i", "I"), ("u", "U"), ("s", "S"), ("t", "T"), ("o", "O"), ("z", "Z")] {
            assert_ne!(
                ph(PhonemeSet::Xsampa, lower),
                ph(PhonemeSet::Xsampa, upper),
                "x-sampa {lower} vs {upper}"
            );
        }
        for (alias, want) in [
            ("A", "ey"),
            ("a", "aa"),
            ("E", "iy"),
            ("e", "eh"),
            ("I", "ay"),
            ("i", "ih"),
            ("O", "ow"),
            ("o", "uw"),
        ] {
            assert_eq!(ph(PhonemeSet::Vccv, alias), want, "vccv {alias}");
        }
    }

    /// Every phone either table can emit must survive the REAL stage2 (ARPABET → the 210-token vocab).
    /// A row that mistypes `"ng"` as `"nng"` compiles and tokenises and then dies at RENDER time; here
    /// it dies at `cargo test` instead.
    #[test]
    fn alias_tables_emit_only_real_arpabet() {
        for (name, table) in [("xsampa", XSAMPA), ("vccv", VCCV)] {
            for &(sym, arp) in table {
                let phones: Vec<String> = arp.split(' ').map(str::to_string).collect();
                assert!(!phones.is_empty() && !arp.is_empty(), "{name} {sym:?} maps to nothing");
                stage2(Lang::En, &phones)
                    .unwrap_or_else(|bad| panic!("{name} {sym:?} -> {arp:?}: {bad:?} is not vocab"));
            }
        }
        // ARPAsing accepts anything `arpabet_is_known` does, including an explicit stress digit —
        // which is the documented escape hatch for ʌ (S90's bare-`ah` rule makes the plain token ə).
        assert_eq!(ph(PhonemeSet::Arpasing, "ah1"), "ah1");
        assert_eq!(stage2(Lang::En, &["ah1".to_string()]).unwrap(), vec!["ʌ"]);
        assert_eq!(stage2(Lang::En, &["ah".to_string()]).unwrap(), vec!["ə"]);
    }

    /// An unreadable alias is a LOUD failure, never a guess (S90). The offending SYMBOL is reported.
    #[test]
    fn alias_unknown_symbol_is_loud() {
        assert_eq!(alias_phones(PhonemeSet::Xsampa, "mØ"), Some(Err("Ø".to_string())));
        assert_eq!(alias_phones(PhonemeSet::Vccv, "q"), Some(Err("q".to_string())));
        assert_eq!(alias_phones(PhonemeSet::Arpasing, "ae zz"), Some(Err("zz".to_string())));
        // `sil`/`sp`/`spn` mean SILENCE to convert_arpabet — refused here rather than sung as a gap
        assert_eq!(alias_phones(PhonemeSet::Arpasing, "ae sil"), Some(Err("sil".to_string())));
        // markers with nothing left is malformed, not silence
        assert_eq!(alias_phones(PhonemeSet::Vccv, "_"), Some(Err("_".to_string())));
        assert_eq!(alias_phones(PhonemeSet::Xsampa, "-"), Some(Err("-".to_string())));
        // …and `Words` never claims a lyric at all
        assert_eq!(alias_phones(PhonemeSet::Words, "hello"), None);
    }

    /// ★ S99 (S91 debt) — the predicate is on the RESULT, not on the input symbols.
    ///
    /// Before it, failure required an unknown CHARACTER, which for an all-letter alias is effectively
    /// unreachable (39 single-char keys in X-SAMPA, 37 in VCCV): `ptk`, `sfth`, `kkkkkk` and the
    /// fourth bank's `tth` were all split into single letters and SUNG. The user's criterion is not
    /// "was the symbol known" but "could this convention have produced this" — so the bound is on the
    /// consonant run, measured at 2 over 440 real aliases.
    #[test]
    fn alias_rejects_clusters_the_convention_cannot_produce() {
        let bad = |set, a: &str| match alias_phones(set, a) {
            Some(Err(e)) => e,
            other => panic!("{a:?} should have been rejected, got {other:?}"),
        };
        // 3+ consonants in a row — the shape zero of the 440 reference aliases has
        assert_eq!(bad(PhonemeSet::Xsampa, "ptk"), "p t k");
        assert_eq!(bad(PhonemeSet::Xsampa, "sfth"), "s f t"); // reported at the first offending run
        assert_eq!(bad(PhonemeSet::Xsampa, "tth"), "t t hh"); // the 4th bank's ð, split into letters
        assert_eq!(bad(PhonemeSet::Xsampa, "zzz"), "z z z");
        assert_eq!(bad(PhonemeSet::Vccv, "kkkkkk"), "k k k");
        // …and it reaches through the carried-vowel rule: the leading vowel is dropped FIRST, so a
        // shape that only becomes impossible after the drop is still caught.
        assert_eq!(bad(PhonemeSet::Xsampa, "Emst"), "m s t");

        // ⚠ NON-VACUITY + the anti-over-reach half. These must all still resolve:
        let ok = |set, a: &str| alias_phones(set, a).expect("not Words").unwrap_or_else(|e| panic!("{a:?} → Err({e})"));
        for (set, a) in [
            (PhonemeSet::Xsampa, "e@m"), (PhonemeSet::Xsampa, "N-"), (PhonemeSet::Xsampa, "-aI"),
            (PhonemeSet::Vccv, "1ng"), (PhonemeSet::Vccv, "hO"), (PhonemeSet::Vccv, "ld"),
            (PhonemeSet::Arpasing, "n t"), (PhonemeSet::Arpasing, "s t"), (PhonemeSet::Arpasing, "ih ng"),
            // a 2-consonant result stays legal even when it came from a symbol we do not know:
            // `tth` under VCCV is t + th = [T TH], an ordinary CC transition (VCCV really does have
            // geminate rows ll/mm/nn). Deliberately hint-level, NOT an error — the user's calibration.
            (PhonemeSet::Vccv, "tth"), (PhonemeSet::Xsampa, "TT"),
        ] {
            let n = ok(set, a).split_whitespace().count();
            assert!(n >= 1, "{a:?} resolved to nothing");
        }
        // nucleus-free aliases are legal and common (168 of the 440 reference aliases are VC/CC/C
        // transition units) — requiring a nucleus would reject a third of every real score
        assert_eq!(ok(PhonemeSet::Arpasing, "n d"), "n d");
        assert_eq!(MAX_CONSONANT_RUN, 2, "raising this needs new corpus evidence, not a hunch");
    }

    /// The tokenizer order this module depends on, as a PROPERTY rather than an assertion about a few
    /// rows: no symbol may be reached before a longer one that also matches.
    #[test]
    fn alias_symbol_order_is_longest_first() {
        for (name, table) in [("xsampa", XSAMPA), ("vccv", VCCV)] {
            let keys = ordered(table);
            for w in keys.windows(2) {
                assert!(w[0].len() >= w[1].len(), "{name}: {:?} before {:?}", w[0], w[1]);
            }
            let uniq: std::collections::HashSet<&str> = keys.iter().copied().collect();
            assert_eq!(uniq.len(), keys.len(), "{name} has a duplicate symbol");
        }
    }

    #[test]
    fn phoneme_set_wire_is_tolerant() {
        for (s, want) in [
            (Some("arpasing"), PhonemeSet::Arpasing),
            (Some("xsampa"), PhonemeSet::Xsampa),
            (Some("vccv"), PhonemeSet::Vccv),
            (Some("words"), PhonemeSet::Words),
            (Some("VCCV"), PhonemeSet::Words), // unknown → the production default, never an error
            (Some(""), PhonemeSet::Words),
            (None, PhonemeSet::Words),
        ] {
            assert_eq!(PhonemeSet::from_wire(s), want, "{s:?}");
            assert_eq!(PhonemeSet::from_wire(Some(want.as_str())), want, "round trip {s:?}");
        }
    }
}
