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
            // S99 THIRD branch — strictly AFTER the table lookup above, never beside the 外来拗音 one:
            // 「base + small ゃゅょ」 rows the generated chart has no romaji for (てゅ/でゅ/ふゅ/ゔゅ…).
            // Running it earlier would resolve きゃ here instead of from the table (see the ORDER TRAP
            // note on `small_ya_kana_phones`, which also refuses table-owned strings on its own).
            if w == 2 {
                if let Some(v) = small_ya_kana_phones(&slice) {
                    out.extend(v);
                    took = w;
                    break;
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

/// Small palatal-glide kana → the romaji key of the standalone ya-row mora it stands for.
const SMALL_YA_ROMAJI: &[(char, &str)] = &[('ゃ', "ya"), ('ゅ', "yu"), ('ょ', "yo")];

/// UTAI EXTENSION (S99, S86#8-3): 「base kana + small ゃゅょ」 that the generated kana table does NOT
/// contain — てゅ/でゅ/ふゅ/ゔゅ and the ゃ/ょ members of those rows. Before this, TWO gaps stacked into a
/// silent truncation: `foreign_kana_phones` only knows the small VOWELS 「ぁぃぅぇぉ」, and the kana table
/// only carries the 拗音 rows that exist in the upstream romaji chart (k/g/n/h/b/p/m/r + the ɕ/dʑ/tɕ
/// rows) — so 「てゅ」 consumed 「て」, hit the small ゅ, and ENDED THE RUN, singing [t e] with no OOV and
/// no red mark. Audited real UST corpus: 11 files / 101 notes use this family.
///
/// The phones are DERIVED, never invented: base onset (base kana → romaji → IPA, minus its vowel) +
/// the small kana's OWN R2IPA row (`ya`/`yu`/`yo` → `[j a]`/`[j ɯ]`/`[j o]`). So 「てゅ」 = [t] + [j ɯ].
///
/// ⚠ WHY NOT the single palatalized token `tʲ`, which the 210-token vocab does contain: **its training
/// exposure is zero.** `score2cv_dur_priors.rs` measures `tʲ`/`dʲ`/`vʲ`/`fʲ`/`sʲ`/`zʲ`/`rʲ`/`tsʲ` at
/// `onset n=0, coda n=0` (they are in the vocab for Russian/Korean, neither of which is in the final
/// training split) — every value on those rows is a fallback, not a measurement. Emitting one is the
/// "dictionary can spell it, model cannot sing it" dead end S94 ruled out. The pieces used here are
/// instead heavily attested: `j` onset n=4868, `t` n=16602, `d` n=7759, `ɸ` n=80, `v` n=3475.
/// The real 拗音 rows keep their measured single-token spelling (きゃ→[c a]) — the generated table
/// still owns those, see the guard below.
///
/// ⚠ ORDER TRAP (S86/S98 both flagged it): this MUST come after the kana table lookup. Placed beside
/// `foreign_kana_phones` — which runs BEFORE it — 「きゃ」 would resolve here to [k j a] instead of the
/// table's measured [c a], destroying every ordinary 拗音. Rather than leave that as a call-site
/// convention that a future edit can quietly break, the invariant is enforced IN the function: a
/// string the generated table already owns returns None no matter who calls it.
fn small_ya_kana_phones(s0: &str) -> Option<Vec<&'static str>> {
    if kana_map().contains_key(s0) {
        return None; // the generated table owns the real 拗音 rows — see ORDER TRAP above
    }
    let chars: Vec<char> = s0.chars().collect();
    let last = *chars.last()?;
    let &(_, ya) = SMALL_YA_ROMAJI.iter().find(|&&(c, _)| c == last)?;
    let base: String = chars[..chars.len() - 1].iter().collect();
    let romaji = kana_map().get(base.as_str())?;
    let seq = r2ipa_map().get(*romaji)?;
    let (&tail, head) = seq.split_last()?;
    if !tbl::VOWEL_SET.contains(&tail) {
        return None; // no plain-vowel tail to replace (ん etc.) — legacy chain decides
    }
    let mut v: Vec<&'static str> = head.to_vec();
    v.extend_from_slice(r2ipa_map().get(ya)?);
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
    /// S92k, **测试构建专有**(生产二进制里这个字段不存在,零运行时代价):Auto 臂的前借账本
    /// `(出借音素下标, 借走的帧数)`,已扣掉「onset 被丢弃后还回去」的那部分 = **净额**。
    ///
    /// 为什么必须由生产记而不能由审计件从最终数组反推:一个音符**同时**向前借进(喂自己的词首
    /// 辅音)又向后借出(喂下一个词的词首辅音),最终数组里只剩净额,两者不可分离 —— 而
    /// 「另一个词伸过来把这个词的元音剪短」正是用户耳判的那个artifact,它必须能被单独读出来。
    /// (S92k 对抗审查的 major:核原先连目标都没有,这条轴整个是瞎的。)
    #[cfg(test)]
    pub borrow_ledger: Vec<(usize, i64)>,
    /// S92k, **测试构建专有**:每个走分配器的事件在**借帧发生之前**拿到的音符内分配
    /// `(事件下标, durs)`。审计件用它当每个音素的「分配器发了多少」基准。
    ///
    /// 为什么记而不是让审计件自己再调一次 `allocate_in_note`:两条臂的 `NoteBudget.spendable`
    /// 不同(InNote 臂先把 onset 预留出去),审计件要么得复制那段预留算术(= 第二份实现,
    /// S92k 审查抓到的三条 major 就是这么来的),要么就只能在一条臂上正确。记下来即可两条臂全对。
    #[cfg(test)]
    pub in_note_alloc: Vec<(usize, Vec<i64>)>,
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
// ⚠ S93 corrected the parenthetical: "0 occurrences" is true for the REALIGNED chaining-language
// consonants (realign_mindur DCONS=3 — 0/5608 English consonants under 3 frames), NOT for the whole
// corpus: the ja hand labs never went through that realignment and DO attest 1-frame phones
// (score2cv_audit_ref.rs: ja ɾ onset p05=1 n=219, ja nucleus short-bucket p05=1). 2 stays the shipped
// EMISSION floor for every phone we place; the one measured exception is the S93 rescue LENDER
// (`RESCUE_LENDER_KEEP` below), which may keep 1 because real zh/ja singing does exactly that.
const CODA_MIN_FRAMES: i64 = 2; // the shipped emission floor: we never PLACE a phone under 2 frames
/// The shortest a consonant ever is in the TRAINING corpus. This is a fact about the data the
/// model was fitted on, NOT about singers: `Much-Better-S2H/scripts/realign_mindur.py` sets
/// `DVOW, DCONS, DSP = 3, 3, 1` and `processed/realign_mindur3_apply.log` records
/// `=== APPLY (in-place) min-dur vow>=3 cons>=3 ===  gtsinger_en re=7423 keep=62`, i.e. the DP
/// PUT it there. It is still the right axis for the audit's `BelowTrainingFloor` finding — a phone
/// shorter than anything the model ever saw is out of the model's distribution — but it must never
/// again be quoted as evidence of what a real singer does. For that, see `chaining_coda_floor`.
/// (test-only since S97: its one consumer is the audit件's `BelowTrainingFloor` axis. Production
/// no longer has a flat consonant floor at all — that was the whole point of the change.)
#[cfg(test)]
pub(super) const TRAINING_MIN_FRAMES: i64 = 3;

/// ★S97 — the frames a CHAINING-language coda may be topped up to, **per phone**.
///
/// ⛔ WHY THIS IS NO LONGER ONE NUMBER. The S96f constant this replaces was justified by two
/// sources presented as independent: (a) `score2cv_audit_ref.rs` "en coda p05 = 3, real singers
/// essentially never go under 3" and (b) `realign_mindur.py DCONS = 3`. **They are the same
/// source.** (b) was applied in place to gtsinger_en and (a) is measured on the result — every one
/// of the 165 en cells in that table has p05 = 3 because the min-duration DP put it there, so the
/// table structurally cannot test the floor that produced it. The knife shipped on a circular
/// argument, and it delivered zero frames to the /l/ it was written for (measured on all eight
/// lane dumps: post_e → post_f raised n f ŋ t tʃ d k z, no /l/ anywhere).
///
/// The numbers below come from the one surface that never touched our code: the GTSinger release
/// annotation (`datasets/gtsinger/processed/English/metadata.json`, 4827 utterances, 38471
/// word-final consonant tokens, `ph_durs` × 50 fps). ⚠ The un-floored backup of OUR OWN alignment
/// (`alignment_backup_pre_mindur`) is NOT a substitute: S97 showed our aligner is systematically
/// early on rhotics (its `ɹ` boundary sits 231-270 ms before both upstream annotators) and
/// late/short on l/n/m/z/t/d, so it disagrees with the upstream annotation in both directions.
///
/// RULE: keep a 3-frame floor only where the upstream p25 is ≥ 3 — i.e. where lifting a 2-frame
/// coda to 3 moves it INTO the human distribution instead of past its lower quartile.
///   floor 3 (upstream p25/p50/p75, n): `l` 3/6/10 (3300) · `n` 3/5/8 (5591) · `z` 4/6/8 (4189) ·
///     `ŋ` 4/6/9 (2542) · `s` 4/6/8 (2093) · `m` 4/6/8 (1942) · `k` 3/5/7 (1386) · `f` 4/5/7 (561) ·
///     `θ` 4/6/9 (315) · `tʃ` 6/8/14 (247) · `ɡ` 4/6/9 (132) · `dʒ` 4/6/8 (104) · `ʃ` 4/6/8 (63)
///   floor 2 (upstream p25 = 2): `t` 2/3/5 (5458) · `d` 2/2/4 (4697) · `ɹ` 2/4/8 (3371) ·
///     `v` 2/4/6 (1907) · `p` 2/4/6 (494) · `ð` 2/3/6 (58)
/// `d` is the one the user reported by ear (2026-08-02, "and 听起来像是尾辅音过重"): S96f lifted
/// `and`'s /d/ from 2 to 3 while the upstream MEDIAN for a word-final /d/ is 2 — the lift pushed it
/// past the 75th percentile of real singing, i.e. exactly backwards.
/// Anything not listed has < 50 upstream coda observations ⇒ **no floor**: "not measured" must not
/// read as "measured and fine" (`score2cv_audit_ref.rs`'s own doctrine, applied to ourselves).
///
/// ⚠ SCOPE, stated rather than hidden: these are ENGLISH numbers applied to every chaining
/// language, because the constant they replace already was. de/fr/es/it have no ear-judgeable
/// material at all; measuring their own upstream annotations is its own round.
/// ⚠ zh/ja never reach here (`consonant_chaining_language`), and structurally have no coda at all
/// (measured: 0 of 1215 ja sung notes, 0 of 489 on each of the three UTAU alias tracks).
pub(super) fn chaining_coda_floor(p: &str) -> i64 {
    match p {
        "l" | "n" | "z" | "ŋ" | "s" | "m" | "k" | "f" | "θ" | "tʃ" | "ɡ" | "dʒ" | "ʃ" => 3,
        _ => CODA_MIN_FRAMES,
    }
}
const REST_KEEP_MIN: i64 = 1; // a lent-from rest keeps ≥1 frame (chunk_at_sp still cuts on it)
const SUNG_KEEP_MIN: i64 = 2; // a lent-from sung phone keeps ≥2 frames
/// S93 — DROP-RESCUE ONLY: the floor an ADJACENT sung-VOWEL lender may fall to when the alternative
/// is deleting a word-initial consonant outright (the note then sings the WRONG syllable). A single
/// frame is in-distribution for real zh/ja singing on short notes — the reference distribution
/// (score2cv_audit_ref.rs, real aligned singing) puts the short-bucket nucleus p05 at 1 for ja
/// o/ɯ/i (o/ɯ MEDIAN 2) and zh a/i/o/u (i median 1); JALAB kept its 1-frame phones through S57.
/// Everywhere else `SUNG_KEEP_MIN` stands.
const RESCUE_LENDER_KEEP: i64 = 1;
/// Fallback targets for a consonant missing from the measured priors (defensive — the generator
/// covers every non-nucleus token; ≈ the global consonant medians).
const ONSET_TARGET_FALLBACK: i64 = 4;
const CODA_TARGET_FALLBACK: i64 = 4;
/// S92c: may an underfed word-initial consonant keep taking from its OWN nucleus until it reaches its
/// measured target, instead of stopping at the bare 2-frame rescue?
///
/// Only for the languages whose syllables CHAIN consonants. zh/ja are CV: the phone before an onset is
/// almost always a long vowel, so the pre-roll borrow reaches the target and this pass has nothing to do
/// — except in fast runs, where it WOULD move behaviour the user has already accepted by ear (S84/S89:
/// あたし's three vowels land exactly on beats 0/7/14, and that test is the Auto arm's whole contract).
/// English is the opposite: 46 of 121 word-initial consonants on the user's own track had a lender with
/// NOTHING to give, so the rescue's 40 ms consonant is the norm rather than the exception there.
///
/// Split per language on the user's explicit instruction ("按语言分别使用每个语言的规则"), and because
/// the alternative was measured: applied unconditionally it turns 6 ja-material tests red, one of them
/// by pushing し's vowel off the beat. zh/ja do not move until they get their own ear test.
fn onset_may_reach_target(lang: g2p::Lang) -> bool {
    consonant_chaining_language(lang)
}

/// The languages whose syllables CHAIN consonants — a word can end in one, and an onset can be preceded
/// by one. zh finals are atomic vocab tokens (`wang` = [w, ɑŋ], no coda at all) and ja is CV with at most
/// a moraic nasal, so for those two the pre-roll borrow reaches its target and a "coda" barely exists;
/// their fast-run allocation and their word endings are ear-verified (S84/S89) and do not move without
/// their own ear test. ONE predicate, now FOUR consumers (the onset supplement, the coda clarity pass,
/// the S97 coda floor top-up, and the S97 phrase-final coda restore in `score2svc`) — the language
/// split must not drift between them, which is why this stays a single function and never a copy.
pub(super) fn consonant_chaining_language(lang: g2p::Lang) -> bool {
    !matches!(lang, g2p::Lang::Zh | g2p::Lang::Ja)
}

/// S92: the frames a coda CLUSTER's raised ceiling may never take from the nucleus. A 2-frame vowel is
/// the cv/decoder collapse region S84 measured (the audible "briefly mute" fast run the user reported);
/// `fast_run_fr5_vowels_keep_three_frames` is where the ear-validated onset clamp already lands, so the
/// cluster ceiling stops at the same place instead of inventing its own number. Without it, [i n z]@80ms
/// pinned the vowel at 2 frames to make room for a coda — a new instance of a bug we already paid for.
const NUCLEUS_KEEP_MIN: i64 = 3;

/// S92j: the divisor of the pre-roll borrow's clamp once the lender is NOT the phone next door — a
/// non-adjacent vowel may give away at most `d / DEEP_LENDER_SHARE` of itself (the adjacent lender
/// keeps the shipped, ear-validated HALF; see the borrow loop for why the two differ perceptually).
/// Measured on the user's 283-note English track, not derived: /3 still leaves 30 vowels shortened by
/// more than a quarter, /4 leaves 14, /5 leaves 13 but costs an onset frame ⇒ 4 is the knee.
const DEEP_LENDER_SHARE: i64 = 4;

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

/// ★S96d — the RATIO weight a coda cluster member claims (S96 knife ③'s proportional split), read
/// at the LONG bucket for every note length.
///
/// A cluster's internal proportions are a property of the phones' IDENTITY (how much of "-ears" is
/// r-colour); the note's compression is already expressed by the shared `budget`. Reading the
/// weights at the note's own bucket instead made the RATIO jump at a bucket boundary — the review
/// CONFIRMED it by measurement: `ɹ`'s coda prior is [3, 3, 16], so stretching a `dears`-class note
/// from 15 to 16 frames (300 → 320 ms) flipped the split from `ɹ2 z4` to `ɹ4 z2` — the word-final
/// /z/ HALVED because the note got LONGER, the exact S89 non-monotonicity this file pins by sweep
/// (and the shipped sweep's `[i n z]`, whose two members scale near-proportionally, is blind to it).
/// The long bucket is the least-compressed measurement, so it is the honest ratio to carry.
/// Absolute lengths are untouched: every member is still bounded by the budget and by its floor.
fn coda_share_weight(p: &str) -> i64 {
    coda_target_frames(p, LONG_BUCKET_FRAMES)
}
/// Any note length that lands in `dur_bucket`'s last bucket (≥16 frames = 320 ms).
const LONG_BUCKET_FRAMES: i64 = 16;

/// S83 knife 5: measured f0==0 fraction (permille) inside a voiceless phone's window, bucketed by
/// its note GROUP length. Real singing zeroes only 17-48% of a SHORT-note voiceless window (the
/// RMVPE track drags in from the previous vowel and pre-voices into the next), while the render
/// zeroed 100% (the S69 R0b① over-correction) — on fast runs that collapsed the voiced duty cycle
/// into the audible "briefly mute" さ/こ/け the user pinpointed. Fallback 1000 = full-window zero:
/// exactly right for the devoiced vowels i̥/ɨ̥/ɯ̥ (true whispers, not in the consonant table) and
/// the conservative legacy behavior for anything else unmapped.
pub fn voiceless_zero_permille(p: &str, group_frames: i64, lang: g2p::Lang) -> i64 {
    let bi = dur_bucket(group_frames);
    // ★S92h: the pooled fourth column is dragged DOWN by Chinese — it supplies ~48% of every voiceless
    // window in the training set and zeroes far less of each one (zh `t` = 43/132/166 permille against
    // a pooled 195/248/370). Every other language is therefore told to stay VOICED through more of its
    // own stops and fricatives than its own singers do, which is the "the consonant is there but it
    // buzzes" end of the user's report. English's own numbers: t 480/344/434, s _/674/701, ʃ _/688/751,
    // k _/522/525 — up to 2.5x the pooled value.
    // ⚠ zh/ja are NOT switched over even though their rows exist: their voiceless rendering is the one
    // the user has verified by ear (S84 knife 5 was tuned on exactly this material), so they keep the
    // pooled column until they get their own listening round. Same predicate as every other language
    // split this round, so the four cannot drift apart.
    if consonant_chaining_language(lang) {
        let code = lang.code();
        if let Some(&(_, _, z)) = super::score2cv_dur_priors::PHONE_ZERO_PERMILLE_LANG
            .iter()
            .find(|&&(lg, tok, _)| lg == code && tok == p)
        {
            if z[bi] != 0 {
                return z[bi]; // 0 = this language has no own data for the bucket → pooled below
            }
        }
    }
    dur_prior(p).map(|(_, _, z)| z[bi]).unwrap_or(1000)
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
    /// S92b: this note STARTS with its nucleus and that nucleus is the very phone the previous note
    /// ended on — i.e. the vowel is not being attacked here, it is being HELD (the western span's
    /// `+` hold re-emitting its syllable's nucleus, with the word-final coda deferred onto it).
    /// It is the one case where leaving the nucleus at `CODA_MIN_FRAMES` is safe: those 2 frames
    /// continue a vowel the model has already been singing, so they are not the 2-frame ATTACK S84
    /// measured as the collapse region. A fresh syllable (ja かん, any onset-bearing note) can never
    /// satisfy it, which is what keeps zh/ja out of this branch by construction.
    nucleus_continues: bool,
    /// S94 G1 (enabler, S92 rollout list): does this note's language CHAIN consonants — the same
    /// `consonant_chaining_language` verdict the borrow loop and the coda clarity pass already key
    /// on, threaded here so the per-language allocation policies queued behind it (G2 k=1 coda
    /// ceiling / G3 medial reserve / G5 word-final release) get the language WITHOUT growing a
    /// second evaluation point. Deliberately unread by the allocation itself today: adding the
    /// field must not move a single frame (the ten probe lanes are byte-identical across it).
    chaining: bool,
}

/// In-note allocation for one note's phones: medial + coda get bounded shares, the nucleus takes the
/// remainder, onset positions are left 0 (the caller funds them — by borrowing from the previous phone
/// on the Auto arm, or by reserving out of `b.spendable` BEFORE this call on the InNote arm). Returns
/// per-phone durations aligned to `ph`; entries may be 0 (dropped medial/coda / unfunded onset) — the
/// caller skips 0-duration phones at emission. Σ(durs) ≤ b.spendable always.
/// The syllable split every downstream pass keys on: `[0, onset_end)` = onset, `[onset_end, nuc)` =
/// medial, `nuc` = nucleus, `(nuc, n)` = coda. Extracted from `build_arrays_daw` verbatim so the
/// AUDIT harness (`score2cv_audit.rs`) reads each phone's target through the SAME split the allocator
/// used — a second copy of these two lines is exactly the kind of drift that makes a checker lie.
/// S92b 的「核不是起音,是被**延长**」判据,抽成单一真源(同 `syllable_split` 的理由:审计件必须
/// 用生产的这一条,而不是自己再写一遍 —— 它决定了一个 2 帧的核算不算 S84 的塌陷区)。
/// `nuc == 0` 是关键:核之前有任何辅音(か / `v ɪ`)都打断元音的连续性。
///
/// ★S92p — `is_sustain` 是**必要**的一项,不是装饰。只比音素字符串时,任何**元音起头的音节**接在
/// 以同一个元音收尾的音节之后都会判「延长」,而那是**全新起音**:日语的「な」→「あ」、英语的
/// `my eyes`。实测:日文探针歌 1215 个有声音符里有 **169 个**命中这条(元音起头的假名极常见),
/// 三条别名轨 82 个。生产侧后果为 0(那些音符 `n_coda == 0`,`nucleus_continues` 只在 coda 段被读),
/// 但**审计件用同一个谓词豁免核塌陷报告** ⇒ 那 169 个全新起音被错误豁免,**仪器在替生产隐藏东西**。
/// 前向孪生 `nucleus_held_by_next` 一直就检查 `r.is_sustain`;这里把两边对齐。
fn nucleus_is_held(
    prev_phone: Option<&&'static str>,
    ph: &[&'static str],
    nuc: usize,
    is_sustain: bool,
) -> bool {
    is_sustain && nuc == 0 && prev_phone == Some(&ph[0])
}

/// 同 `syllable_split`,给同级探针模块(`score2svc_mg`)用 —— 它不是 score2cv 的子模块,看不到私有件,
/// 而反投影对拍必须用**生产的这一条**切分,不许自己再写一份(S92k 审查的三条 major 就是这么来的)。
#[cfg(test)]
pub(crate) fn syllable_split_for_audit(ph: &[&'static str]) -> (usize, usize) {
    syllable_split(ph)
}

fn syllable_split(ph: &[&'static str]) -> (usize, usize) {
    let n = ph.len();
    let nuc = ph.iter().rposition(|p| is_nucleus_phone(p)).unwrap_or(n - 1);
    let onset_end = ph.iter().position(|p| is_nucleus_phone(p)).unwrap_or(n - 1).min(nuc);
    (onset_end, nuc)
}

fn allocate_in_note(
    ph: &[&'static str],
    b: NoteBudget,
    onset_end: usize,
    nuc: usize,
    nuc_stress: Option<&[u8]>,
    // ★S96 knife ① — what the stress POOL must set aside for the Auto arm's post-allocate onset
    // funding (floor pass + target chase), which draws ONLY from the FINAL nucleus's remainder.
    // Historically that remainder was always fat (last-takes-all); the stress pool moves the fat to
    // a MEDIAL nucleus, and without this reserve a standalone `flowers` [f l aʊ ɝ z]@22 DELETED its
    // /f/ outright (sang "lowers") — caught by the probe, now pinned by a test. The caller computes
    // it (it knows the arm and the phrase position); 0 whenever the pool is off or InNote already
    // reserved the onsets upfront.
    onset_allowance: i64,
) -> Vec<i64> {
    let NoteBudget { note_frames, spendable: fr, nucleus_continues, chaining } = b;
    let n = ph.len();
    let mut durs = vec![0i64; n];
    let n_coda = n - nuc - 1;
    let n_medial = nuc - onset_end;
    let nuc_floor = fr.min(2).max(1); // the nucleus never drops below min(fr,2)
    // ★S92g: what the MEDIAL pass must leave. `nuc_floor` (2) is the collapse region S84 measured —
    // a review walked `refined` = [ɹ ə f aɪ n d] @ 20 frames (400 ms!) through this loop and the medial
    // consonants took it down to a 2-frame aɪ. The coda pass and the S92 borrow both already keep
    // NUCLEUS_KEEP_MIN; the medial pass was the one branch still on the old floor, and it is the branch
    // that fires on exactly the shape English produces most (a multi-syllable word on one note).
    let nuc_keep = fr.min(NUCLEUS_KEEP_MIN).max(1);
    let mut used = 0i64;
    // medial (between the first and last nucleus): a medial CONSONANT is really the NEXT
    // syllable's ONSET — a multi-syllable word on ONE note flattens its syllable boundaries
    // (refined = [ɹ ə f aɪ n d]: the f leads the second syllable) — so it gets its own measured
    // onset target (the old flat 2..4 share made the f inaudible; S83 user-verified). A medial
    // VOWEL (più's i) keeps the small share. Same ≥2-or-DROP policy as codas (1-frame = OOD);
    // the break drops later medials first.
    //
    // ★S96 knife ① — when the word's ARPABET stress digits made it here (en dictionary/hint words
    // only; `ResolvedNote::nucleus_stress`), the nuclei split their POOL by stress weight instead of
    // "every non-final nucleus is clamped at 4 and the LAST one takes all the remainder". The user's
    // `every`@18fr sang ɛ:3 v:3 ɝ:3 i:9 — the STRESSED (EH1) vowel got 60 ms while the word-final
    // unstressed IY0 swallowed the remainder, purely because of its position; the SV reference sings
    // two clearly attacked segments (EV-ry) and the reference distribution puts a real long-note
    // non-final nucleus at p50 = 10 frames (gen_vowel_placement, en b2 max-nonfinal), not 4.
    // Weights 3/2/1 (primary/secondary/unstressed); the pool = spendable minus the medial
    // consonants' targets minus the coda pass's own budget formula (an ESTIMATE — the per-step
    // min-guards below still bound every give, so a bad estimate can squeeze but never break
    // conservation or the nucleus floor). The FINAL nucleus still takes the remainder line — its
    // weight participates only through what the medials are allowed to claim.
    // zh/ja/alias and the MFA languages carry no digits ⇒ `nuc_stress` is None ⇒ byte-identical.
    let n_nuclei = ph.iter().filter(|p| is_nucleus_phone(p)).count();
    // ★S96d (review CONFIRMED): a count mismatch is a LEGAL, ORDINARY input — S90 made the
    // stressless ARPABET nucleus a first-class spelling, so a PARTIALLY-stressed hint like
    // `[EH1 V ER IY]` extracts fewer digits than it has nuclei. The first cut debug_assert!'d
    // here ("equivalence broke") and a user hint could panic the whole render in a dev build
    // (tauri dev = debug_assertions on — the user's actual bench). The digit≡nucleus equivalence
    // is a DICTIONARY property (S90 gate, 863018 tokens), never a hint property. Mismatch ⇒ the
    // stress channel simply doesn't exist for this note: stress-blind allocation, no noise.
    let stress_w: Option<Vec<i64>> = nuc_stress.and_then(|s| {
        if !chaining || n_nuclei < 2 || s.len() != n_nuclei {
            return None;
        }
        Some(s.iter().map(|&d| match d { 1 => 3, 2 => 2, _ => 1 }).collect())
    });
    for i in onset_end..nuc {
        let c = if is_nucleus_phone(ph[i]) {
            (fr / ((n_medial + n_coda) as i64 + 2)).clamp(CODA_MIN_FRAMES, 4)
        } else {
            onset_target_frames(ph[i], note_frames)
        }
        .min(fr - nuc_keep - used);
        if c < CODA_MIN_FRAMES {
            break;
        }
        durs[i] = c;
        used += c;
    }
    // coda: per-token measured target each (S83 second knife: t/d≈3, n≈4, s/ɕ≈6-7 — one flat cap
    // flattened the 程度), ≥2 each (else DROPPED — never a 1-frame phone), total ≤ 2/5 of the note
    // (training: vowel share median 44-47%). LAST-first is the DROP rule: the word-final release is
    // the perceptually load-bearing cue — when the budget starves, inner codas drop before it.
    //
    // ★S92 (history): "LAST-first, each takes its full measured target" let the outermost consonant
    // eat the whole budget and SILENTLY DELETED the inner one (`means`@320ms sang "meez",
    // `don't`@160ms "dote", `find`@160ms "fide" — 6 notes on the user's real track). S92 patched
    // that with a reserve (outermost may no longer starve the inner ones below their minimum) and
    // with the `cluster_floor` ceiling raise below. S96 replaces the reserve+serve-in-full loop with
    // the proportional split (see the block at the budget line) — the reserve's guarantee (every
    // live member ≥ CODA_MIN_FRAMES) is subsumed by the floors, and the cluster_floor raise is kept.
    // ⚠ Deliberately NOT fixed here: a SINGLE coda that cannot reach 2 frames on a very short note
    // (`ɪ n`@80ms, 2 of the 8 real drops) — raising the k=1 ceiling would change ja (かん@80ms) and
    // needs its own ear test. See project_v2_pending_cleanups.
    if n_coda > 0 {
        let want: i64 = ph[nuc + 1..].iter().map(|p| coda_target_frames(p, note_frames)).sum();
        // S92b: a HELD nucleus (see `NoteBudget::nucleus_continues`) may fall to CODA_MIN_FRAMES,
        // because those frames continue a vowel already in flight rather than attacking a new one —
        // and it earns the floor at n_coda == 1 too. That single case is the whole reason the user's
        // `even` sang without its /n/: the word's deferred word-final coda lands on a 4-frame hold,
        // where 2/5 of the note is ONE frame, so no coda could exist there at any target.
        let keep = if nucleus_continues { CODA_MIN_FRAMES } else { NUCLEUS_KEEP_MIN };
        let cluster_floor = if n_coda >= 2 || nucleus_continues {
            (n_coda as i64 * CODA_MIN_FRAMES).min((fr - keep).max(0))
        } else {
            0
        };
        let budget = want.min(fr - nuc_floor - used).min((fr * 2 / 5).max(cluster_floor));
        // ★S96 — the budget is split IN PROPORTION TO THE MEASURED TARGETS, not LAST-first-take-all.
        // LAST-first survives as the DROP rule (inner codas die first — the word-final release is
        // still the perceptually load-bearing cue) but no longer as the SERVING rule: serving the
        // outermost member its FULL target first inverted the cluster's internal ratio whenever the
        // budget was tight. The user's own `dears` [d ɪ ɹ z]@21fr: budget = 21*2/5 = 8, z took its
        // full 6 and ɹ was left at the 2-frame floor — while the language-specific reference
        // distribution (score2cv_audit_ref.rs) puts a long-note en ɹ coda at p50=16 vs z p50=6:
        // the ratio was UPSIDE-DOWN (the r-colour is most of what the listener hears in "-ears").
        //
        // ★★S98 — WHICH reference said 16 matters, and the answer is now on the record.
        // That p50=16 is the `t_*` (TRAINING) column: our own aligner + realign_mindur's DP floor.
        // The dataset's OWN annotation (`h_*`, added S98) says en ɹ coda long-bucket p50 = **5**
        // (n=2603) against z p50 = 6 — i.e. on the human surface the ratio is NOT upside-down, it is
        // roughly even. The same signature is now visible in French (fr ʁ coda long: train p50=15 vs
        // human p50=7), which is exactly the cross-language rhotic-boundary artefact S97 traced to
        // our aligner sitting 231-270 ms early on rhotics.
        // ⚠ This is NOT a licence to flip the split. Two facts point the other way and neither is
        // settled: (a) the content model was TRAINED on the `t_*` label convention, so feeding it our
        // own convention may well be what makes it render correctly — S97 declined to revert S92n for
        // this reason; (b) the user's ear on this very word asked for MORE r ("dear 的 /r/ 出不来"),
        // which the current 6:2 delivers. The weight itself comes from `coda_share_weight` ->
        // PHONE_DUR_PRIORS, which is still wholly on the training surface.
        // ⇒ Deliberately UNCHANGED this round. Re-deciding it needs its own round: regenerate the
        // priors on both surfaces, A/B render, and an ear pass. Booked in pending_cleanups.
        // Proportional split gives ɹ6/z2 out of the SAME 8 — zero-sum inside the budget; the nucleus
        // keeps exactly the frames it kept before, and the S92 cluster minima are unchanged (every
        // member that lives still gets ≥ CODA_MIN_FRAMES; `means` n2/z4 and `don't` n2/t2 come out
        // byte-identical — verified by the pinned tests below).
        // zh/ja/alias tracks are no-ops BY CONSTRUCTION, same argument as S92: n_coda ≤ 1 there, and
        // a single member's proportional share IS the whole budget (give = min(budget, target),
        // exactly the old arithmetic). ⚠ NOT touched this round: the 2/5 ceiling itself (raising it
        // is the deferred "r-colour is part of the vowel" decision — S92n said ear first).
        // How many members can live at ≥2 — inner members drop first, exactly as before.
        let k_live = ((budget / CODA_MIN_FRAMES).max(0) as usize).min(n_coda);
        if k_live > 0 {
            let live0 = n - k_live;
            // ★A DROPPED member still holds back its floor (capped so the survivors keep theirs) —
            // the exact semantics of the S92 reserve it replaces. Without this, the lone survivor
            // of a starved cluster swallows the whole budget, and one frame of extra note length
            // (k_live 1 → 2) then SHRINKS it: the sweep test caught `[.. 0 3] -> [.. 2 2]` at
            // fr 6 → 7, a violation of "a longer note never shortens a consonant" (S89).
            let held = (CODA_MIN_FRAMES * (n_coda - k_live) as i64)
                .min((budget - CODA_MIN_FRAMES * k_live as i64).max(0));
            let budget = budget - held;
            // Iterative proportional fill: members whose proportional share rounds below the 2-frame
            // floor get pinned AT the floor and removed from the pool, the rest re-shares the
            // remaining budget (straight water-filling; ≤ n_coda rounds). Leftover frames from
            // integer truncation go to the largest remainders, OUTER member first on ties.
            let mut give = vec![0i64; k_live];
            let mut pool: Vec<usize> = (0..k_live).collect(); // indices into give/live members
            let mut b = budget;
            while !pool.is_empty() {
                let w_pool: i64 =
                    pool.iter().map(|&j| coda_share_weight(ph[live0 + j])).sum();
                if w_pool <= 0 {
                    break;
                }
                let mut under: Vec<usize> = Vec::new();
                let mut shares: Vec<(usize, i64, i64)> = Vec::new(); // (j, floor, remainder)
                for &j in &pool {
                    let t = coda_share_weight(ph[live0 + j]);
                    let q = b * t / w_pool;
                    if q < CODA_MIN_FRAMES {
                        under.push(j);
                    } else {
                        shares.push((j, q, b * t % w_pool));
                    }
                }
                if under.is_empty() {
                    let assigned: i64 = shares.iter().map(|s| s.1).sum();
                    for &(j, q, _) in &shares {
                        give[j] = q;
                    }
                    // Leftover from integer truncation (< pool size by construction, so each member
                    // gains at most 1 and can never exceed its target — q ≤ t−1 whenever b < w_pool,
                    // and b == w_pool leaves zero remainder): largest remainder first, ties → the
                    // OUTER (later) member (the word-final release keeps its priority).
                    shares.sort_by(|x, y| y.2.cmp(&x.2).then(y.0.cmp(&x.0)));
                    let mut left = b - assigned;
                    for &(j, _, _) in &shares {
                        if left == 0 {
                            break;
                        }
                        give[j] += 1;
                        left -= 1;
                    }
                    break;
                }
                for &j in &under {
                    give[j] = CODA_MIN_FRAMES;
                    b -= CODA_MIN_FRAMES;
                    pool.retain(|&x| x != j);
                }
            }
            for (j, &g) in give.iter().enumerate() {
                debug_assert!(g >= CODA_MIN_FRAMES, "a live coda member below the emission floor");
                durs[live0 + j] = g;
                used += g;
            }
            debug_assert!(give.iter().sum::<i64>() <= budget, "coda pass overspent its budget");
        }
    }
    durs[nuc] = fr - used; // nucleus takes the whole remainder (≥ nuc_floor by construction)
    // ★S96d knife ① (redesigned after the review + the fr∈[6,13] deletion sweep): stress is a
    // REDISTRIBUTION over the finished baseline allocation, never a different targeting formula.
    // The first cut gave stressed medials their own pool targets inside the loop above — and the
    // fattened early nucleus pushed later medials into the `break`: `every`@12 silently deleted
    // its ɝ, `very`@7 its /ɹ/ ("ve-y"), while a reserve strict enough to prevent that deleted
    // EVERYTHING at fr 6..8. Running the shipped arithmetic first makes the survival set equal to
    // the baseline's BY CONSTRUCTION (the sweep test pins it); stress then only moves SPARE frames
    // between the surviving nuclei:
    //   • a medial nucleus keeps ≥ CODA_MIN_FRAMES (its shipped floor);
    //   • the FINAL nucleus keeps ≥ nuc_keep + onset_allowance — the onset-funding passes draw
    //     from ITS remainder only, and without the reserve a standalone `flowers` lost its /f/
    //     (the probe-caught "lowers" bug; the allowance mirrors exactly what those passes may take);
    //   • the freed frames re-split proportional to weights 3/2/1 (primary/secondary/unstressed),
    //     largest remainder, ties → higher weight then earlier position. `every`@18fr: the
    //     stressed ɛ rises 3→7 and the word-final unstressed IY0 falls 9→4 — position privilege
    //     gone, and the user's "重读 EH1 只拿 60ms" symptom with it. Conservation is structural
    //     (zero-sum transfer); consonants never move here.
    if let Some(w) = &stress_w {
        let mut idx: Vec<(usize, i64)> = Vec::new(); // (ph position, weight) of SURVIVING nuclei
        let mut ord = 0usize;
        for (i, p) in ph.iter().enumerate() {
            if is_nucleus_phone(p) {
                if durs[i] > 0 {
                    idx.push((i, w[ord]));
                }
                ord += 1;
            }
        }
        if idx.len() >= 2 {
            // floors captured BEFORE any write (the final nucleus's floor reads its own current
            // remainder — a live read inside the assignment loop would be order-sensitive)
            let floors: Vec<i64> = idx
                .iter()
                .map(|&(i, _)| {
                    if i == nuc {
                        durs[nuc].min(nuc_keep + onset_allowance)
                    } else {
                        CODA_MIN_FRAMES
                    }
                })
                .collect();
            let free: i64 =
                idx.iter().zip(&floors).map(|(&(i, _), &fl)| (durs[i] - fl).max(0)).sum();
            let w_sum: i64 = idx.iter().map(|&(_, wi)| wi).sum();
            if free > 0 && w_sum > 0 {
                let mut shares: Vec<(usize, i64, i64, i64, i64)> = idx
                    .iter()
                    .zip(&floors)
                    .map(|(&(i, wi), &fl)| (i, free * wi / w_sum, free * wi % w_sum, wi, fl))
                    .collect();
                let assigned: i64 = shares.iter().map(|s| s.1).sum();
                // leftover by largest remainder; ties → higher weight, then earlier position
                shares.sort_by(|a, b| b.2.cmp(&a.2).then(b.3.cmp(&a.3)).then(a.0.cmp(&b.0)));
                let mut left = free - assigned;
                for s in shares.iter_mut() {
                    if left == 0 {
                        break;
                    }
                    s.1 += 1;
                    left -= 1;
                }
                for &(i, share, _, _, fl) in &shares {
                    durs[i] = fl + share;
                }
            }
        }
    }
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

    /// S99 (S86#8-3): 「base + small ゃゅょ」 rows the generated chart has no romaji for. Before this
    /// they were SILENTLY TRUNCATED to the base mora — てゅ sang [t e] with no OOV and no red mark.
    #[test]
    fn small_ya_kana_rows_sing_in_full() {
        for (k, want) in [
            ("てゅ", vec!["t", "j", "ɯ"]), ("てゃ", vec!["t", "j", "a"]), ("てょ", vec!["t", "j", "o"]),
            ("でゅ", vec!["d", "j", "ɯ"]),
            ("ふゅ", vec!["ɸ", "j", "ɯ"]), ("ふゃ", vec!["ɸ", "j", "a"]), ("ふょ", vec!["ɸ", "j", "o"]),
            ("ゔゅ", vec!["v", "j", "ɯ"]),
        ] {
            assert_eq!(phones(k), want, "{k}");
        }
        // katakana arrives folded upstream, same as the small-vowel family
        assert_eq!(phones(&super::super::g2p::fold_katakana("テュ")), vec!["t", "j", "ɯ"]);
        // inside a multi-mora string, and with a 長音符 that must add no phone
        assert_eq!(phones("てゅーん"), vec!["t", "j", "ɯ", "ɴ"]);
        // a sustain after it must carry the SWAPPED vowel (ɯ), not the base's e
        let arr = build_arrays(&[("てゅ", 60, 80), ("ー", 60, 80)]).unwrap();
        assert_eq!(arr.phon, vec!["t", "j", "ɯ", "ɯ"], "sustain re-emits the small-ya vowel");
    }

    /// ★ THE order trap this rule could most easily cause (S86 and S98 both flagged it in advance):
    /// run it before the kana table and every ordinary 拗音 resolves compositionally instead of from
    /// its MEASURED single token — きゃ would become [k j a] rather than [c a]. Two independent guards
    /// are pinned here: the rule itself refuses table-owned strings, and the resolved answer is still
    /// the table's. The `assert_ne!` is what keeps this test from being vacuous (S92p): it proves the
    /// two arms really do disagree, so a regression cannot pass by making them coincide.
    #[test]
    fn small_ya_never_steals_a_real_yoon_row() {
        assert_ne!(phones("きゃ"), vec!["k", "j", "a"], "compositional answer must NOT win here");
        let mut yoon = 0usize;
        for (base, _) in tbl::KANA.iter().chain(super::super::g2p_tables::KANA_EXTRA) {
            for sy in ['ゃ', 'ゅ', 'ょ'] {
                let s = format!("{base}{sy}");
                if let Some(&romaji) = kana_map().get(s.as_str()) {
                    yoon += 1;
                    assert!(small_ya_kana_phones(&s).is_none(), "{s} is a table row — the rule must refuse it");
                    // and the engine still answers with the table's measured token
                    assert_eq!(phones(&s), r2ipa_map()[romaji].to_vec(), "{s} drifted off the table");
                }
            }
        }
        assert!(yoon >= 33, "sweep actually covered the 拗音 rows (got {yoon})");
    }

    /// Vocabulary + training-exposure safety for the new rule. The vocab CONTAINS single palatalized
    /// tokens (`tʲ` id 142, `dʲ` 141, …) that would spell てゅ in two phones instead of three — and
    /// they are exactly the wrong answer: `score2cv_dur_priors.rs` measures every one of them at
    /// `onset n=0, coda n=0` (they are in the 210-token vocab for Russian/Korean, neither of which is
    /// in the final training split). Pinning it here means a later "tidy-up" toward the shorter
    /// spelling has to delete this assert and read why.
    #[test]
    fn small_ya_emits_only_well_trained_vocab() {
        let ids = phone_to_id_map();
        const ZERO_EXPOSURE: &[&str] = &["tʲ", "dʲ", "vʲ", "fʲ", "sʲ", "zʲ", "rʲ", "tsʲ"];
        let mut combos = 0usize;
        for (base, _) in tbl::KANA.iter().chain(super::super::g2p_tables::KANA_EXTRA) {
            for sy in ['ゃ', 'ゅ', 'ょ'] {
                let s = format!("{base}{sy}");
                let Some(v) = small_ya_kana_phones(&s) else { continue };
                combos += 1;
                assert_eq!(v.last(), Some(&["a", "ɯ", "o"][['ゃ', 'ゅ', 'ょ'].iter().position(|&c| c == sy).unwrap()]));
                for p in &v {
                    assert!(ids.contains_key(p), "{s} emitted out-of-vocab phone {p}");
                    assert!(!ZERO_EXPOSURE.contains(p), "{s} emitted zero-training-exposure token {p}");
                }
            }
        }
        assert!(combos > 30, "sweep actually exercised the rule (got {combos})");
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
                // ★S92n: onset 与 coda 的上限**不同**。onset 靠向邻居借帧,抬它等于加剧 S92j 刚修好的
                // 「掏空邻居元音」,所以它仍钉在 7;coda 由音符自己的预算出(`fr*2/5` + 余量两道界),
                // 抬它只会让词尾辅音回到真人时长。反投影实测:含 `ɹ` coda 的 11 个音符里,我们给的帧数
                // **每一个都恰好是 7** = 这个钳位本身,而真人给 10-54 帧。
                for v in o.iter() {
                    assert!((2..=7).contains(v), "{p} ONSET prior out of window: {o:?}");
                }
                for v in c.iter() {
                    assert!((2..=20).contains(v), "{p} CODA prior out of window: {c:?}");
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
    // S92k: test-build-only borrow ledger (see `ScoreArrays::borrow_ledger`).
    #[cfg(test)]
    let mut ledger: Vec<(usize, i64)> = Vec::new();
    // ★S92o: indices in `pdur` of codas that the CODA PRE-ROLL just fed. The next note's onset borrow
    // must not drain them again — measured on the user's track before this guard existed: the pre-roll
    // moved 2 frames into `even`'s /n/ and `feel`'s /f/ took them straight back out at depth 1, so the
    // frames ended up in the NEXT word's onset and the release was no better off. Same self-defeating
    // shape S92e already had to bound once (a deep borrow re-draining a consonant it had just fed).
    let mut coda_preroll_fed: Vec<usize> = Vec::new();
    #[cfg(test)]
    let mut alloc_snap: Vec<(usize, Vec<i64>)> = Vec::new();

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
                    let (onset_end, nuc) = syllable_split(ph);
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
                    // ★S92i: …but a short note whose VOWEL IS HELD BY THE NEXT NOTE is not a fast run at
                    // all — it is the head of a melisma, the shape this author writes constantly
                    // (`thing`@4fr followed by 20+8+29+4+20+12 frames of the same ɪ). The cap then cut
                    // /θ/ from its measured 7 frames to 2 (40 ms) to protect a vowel that is in no
                    // danger: the note's own frames all stay with the vowel, and the onset is funded by
                    // BORROWING from the previous note. Measured on the user's track: 7 of 7 short
                    // onset-bearing notes are of exactly this shape (`man` `your` `thing`×2 `share` …),
                    // and the user named three of them by ear.
                    // The predicate is the forward twin of `nucleus_continues` (S92b): the NEXT event is
                    // a sustain re-emitting this note's nucleus. A genuine fast run — か し た, separate
                    // syllables — never satisfies it, so S84's ear-validated regime is untouched.
                    let nucleus_held_by_next = nuc < ph.len()
                        && resolved.get(k + 1).is_some_and(|r| {
                            r.is_sustain
                                && matches!(&r.kind, g2p::ResolvedKind::Phones(np) if np.first() == Some(&ph[nuc]))
                        });
                    let target = |p: &'static str| {
                        let t = onset_target_frames(p, fr);
                        if fr <= 5 && !nucleus_held_by_next {
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
                    // S92b: "the vowel is being HELD, not attacked" — this note starts with its
                    // nucleus AND that nucleus is the phone the previous note ended on. `nuc == 0`
                    // is what makes it exact: any onset before the nucleus (か, `v ɪ`) breaks the
                    // vowel's continuity, so those notes keep the full nucleus protection.
                    let nucleus_continues = nucleus_is_held(phon.last(), ph, nuc, res.is_sustain);
                    // ONE evaluation point per note for the language split (the borrow loop binds
                    // its own `chaining` from the same predicate; they must never disagree).
                    let chaining_lang = consonant_chaining_language(res.run_lang);
                    // ★S96 knife ① — the stress pool's onset reserve (see `allocate_in_note`'s
                    // parameter doc). Auto arm only: the InNote arm reserved its onsets out of
                    // `spendable` BEFORE this call, so its allowance is structurally 0. It mirrors
                    // exactly what the funding passes below may take from the final nucleus (the
                    // measured targets under the shipped half-note clamp), so a fat MEDIAL can
                    // never starve a word-initial consonant.
                    let onset_allowance = if timing == ArticulationTiming::Auto && onset_end > 0 {
                        let floors = CODA_MIN_FRAMES * onset_end as i64;
                        let want: i64 = ph[..onset_end].iter().map(|&p| target(p)).sum();
                        want.max(floors).min((fr - SUNG_KEEP_MIN).max(0).min((fr + 1) / 2))
                    } else {
                        0
                    };
                    let mut durs = allocate_in_note(
                        ph,
                        NoteBudget {
                            note_frames: fr,
                            spendable: fr - reserved,
                            nucleus_continues,
                            chaining: chaining_lang,
                        },
                        onset_end,
                        nuc,
                        res.nucleus_stress.as_deref(),
                        onset_allowance,
                    );
                    durs[..onset_end].copy_from_slice(&onset_durs[..onset_end]);
                    // S92k: snapshot the IN-NOTE allocation before any borrow moves frames.
                    #[cfg(test)]
                    alloc_snap.push((k, durs.clone()));
                    if onset_end > 0 && timing == ArticulationTiming::Auto {
                        // borrow the onset consonants' frames from the tail of the previous phone so
                        // the NUCLEUS starts on the beat (zero-sum: the timeline never moves).
                        // (the fr≤5 target cap is hoisted into `target`, shared with the InNote arm —
                        // same measured justification, same note-length key.)
                        let want: i64 = ph[..onset_end].iter().map(|&p| target(p)).sum();
                        // ★S92d — the borrow walks BACK over the preceding phones instead of inspecting
                        // only the immediately previous one. S92c fed a starved onset from its own nucleus,
                        // which works but delays the vowel — and "the nucleus starts on the beat" is this
                        // arm's whole contract; the user heard exactly that ("时序有点怪"). Measured on the
                        // user's track: all 60 starved onsets (179 frames) could be fed from further back
                        // instead, because the phone one or two steps back is a long vowel with 5-7 frames
                        // to spare (`t:2(+0) <- n:3(+1) <- oʊ:10(+5)`). Taking from THERE shifts only the
                        // intervening short consonants earlier — what English singers do with a word-final
                        // consonant — and the nucleus keeps every frame it had.
                        //
                        // Each lender keeps its own floor and its own shipped half-clamp; a rest may lend
                        // (REST_KEEP_MIN) but the walk STOPS there (never dig through silence into the
                        // phrase before). Depth 4 is what the measurement needed for 100% coverage.
                        // ⚠ zh/ja walk depth 1 = EXACTLY the previous single-phone rule, same clamp, so
                        // their allocation is bit-identical by construction (their fast-run timing is
                        // ear-verified — S84 あたし's vowels land on beats 0/7/14).
                        const BORROW_MAX_DEPTH: usize = 4;
                        // ONE evaluation point for "does this language chain consonants" — the depth
                        // limit, the vowel-lender floor and the in-note supplement below must never be
                        // able to disagree about which arm a track is on.
                        let chaining = onset_may_reach_target(res.run_lang);
                        let depth_limit = if chaining { BORROW_MAX_DEPTH } else { 1 };
                        // ★S92j — a VOWEL lender's floor is NUCLEUS_KEEP_MIN at EVERY depth on the
                        // chaining arm. It used to be NUCLEUS_KEEP_MIN from depth 2 and SUNG_KEEP_MIN
                        // (= 2 = the S84 collapse region) at depth 1, and that inconsistency is what
                        // tightening the deep clamp exposed: the demand simply moved to the adjacent
                        // lender, which had no vowel protection. Measured on the user's track, `so`@8fr
                        // came out `s:7 oʊ:2` — a 40 ms diphthong. The in-note half already keeps
                        // NUCLEUS_KEEP_MIN (S92g); this is the same invariant on the borrow half, so a
                        // BORROWED-FROM vowel never lands in the collapse region.
                        // ⚠ S92p — the earlier wording here ("…now holds everywhere") was an
                        // over-claim: a note's OWN nucleus can still end at 2 via the all-or-nothing
                        // floor pass (`nuc_floor = fr.min(2)`) and via the InNote arm's `avail`. The
                        // honest scope is written out in `s92g_medial_and_supplement_no_longer_eat_the_vowel`.
                        // zh/ja keep SUNG_KEEP_MIN = the shipped, ear-validated rule, byte-for-byte.
                        let vowel_keep = if chaining { NUCLEUS_KEEP_MIN } else { SUNG_KEEP_MIN };
                        // ledger of (index in pdur, frames) so a DROPPED onset can hand its frames back to
                        // the exact phones they came from — the borrow stays zero-sum even when it fails.
                        let mut borrowed: Vec<(usize, i64)> = Vec::new();
                        let mut left = 0i64;
                        let mut j = pdur.len();
                        let mut depth = 0usize;
                        let prev_evt = pevt.last().copied();
                        while left < want && j > 0 && depth < depth_limit {
                            j -= 1;
                            depth += 1;
                            let is_rest = matches!(phon[j], "SP" | "AP");
                            // ★S92o: a coda the pre-roll just fed is off limits — see `coda_preroll_fed`.
                            if coda_preroll_fed.contains(&j) {
                                continue;
                            }
                            // ★Stay inside the note we are ALREADY restructuring (a rest is silence and may
                            // always lend). Reaching two notes back would shift a whole intervening note
                            // earlier — the very timing displacement this round exists to remove. Measured:
                            // without this bound, note 45's `s p` cluster reached past `whisper` into the
                            // note before it and moved 5 frames across two boundaries.
                            if depth > 1 && !is_rest && pevt.get(j).copied() != prev_evt {
                                break;
                            }
                            let d = pdur[j];
                            // UTAU-style auto-scale, structural half: a SUNG lender never loses more than
                            // half its frames (ceil) — in a fast run the previous vowel must stay audible.
                            // ★At depth ≥ 2 only a VOWEL lends. Draining a preceding CONSONANT would undo
                            // the very fix this round makes for it — measured: note 45's `s p` cluster ate
                            // `whisper`'s /w/ back down from 4 frames to 2, i.e. the cascade re-appearing
                            // through the other end of the borrow.
                            //
                            // ★S92j — the half-clamp becomes a QUARTER-clamp once the lender is no longer
                            // adjacent. Half was ear-validated for the phone RIGHT BEFORE the onset: that
                            // phone belongs to the syllable the listener already groups with this
                            // consonant, so shortening it reads as normal legato. From depth 2 the lender
                            // is a DIFFERENT syllable, and halving it is what the user heard as splicing:
                            // `hurt`'s onset walked back into `might` and took its /aɪ/ from 5 frames to 3,
                            // and `shame`'s /eɪ/ lost 3 of 11 — a diphthong that loses a quarter of itself
                            // stops completing its glide, which is why an open vowel came out sounding
                            // closed. Quartering does NOT reduce how many frames the onsets get (measured:
                            // Σ onset frames identical, 595, across the whole track) — it forces the walk
                            // to spread the same demand over several lenders instead of gutting the first
                            // one. Measured on the user's track: vowels shortened by >25% fall 42 → 14,
                            // total timeline displacement 540 → 484 frames, and the count of vowels driven
                            // into the S84 collapse region falls 3 → 2 (the three alias tracks: 3 → 0).
                            // ⚠ The coefficient is measured, not derived: /3 leaves 30 vowels over 25% and
                            // /5 buys only one more (13) while costing an onset frame, i.e. 4 is the knee.
                            let cap_j = if is_rest {
                                // ★S96 knife ② (S96d revision — the absolute cut-off was CONFIRMED
                                // to starve short post-rest notes): a REST lends a CHAINING-language
                                // onset only the SHORTFALL — what the note structurally cannot fund
                                // from itself. On roomy notes the shortfall is 0 and the attack
                                // starts AT the boundary (consonant on the beat, vowel after it =
                                // the SV reference the user's ear ratified: `look` +4). On short
                                // notes (`can`@7: the nucleus guard leaves only 2 in-note;
                                // `thing`@4 + sustain: the S92i melisma head wants θ=7 a 4-frame
                                // note can never hold) the rest funds the difference — the first
                                // cut set this cap to a flat 0 and can/thing/king/smell's onsets
                                // collapsed to 40 ms floors, exactly the "乱发音" the user reported.
                                // The in-note deliverable mirrors the funding passes below: floors
                                // down to min(fr,2), the chase down to NUCLEUS_KEEP_MIN, both under
                                // this note's own budget (post-rest = the half-note cap; a DEEP
                                // rest reached through sung material = the in-phrase attack bound).
                                // zh/ja keep the shipped full pre-roll into silence — real ja
                                // singing does exactly that (gen_vowel_placement: 2478 consonants
                                // inside rest groups) and their timing is ear-anchored (S84/S89).
                                if chaining {
                                    let budget_here = (fr - SUNG_KEEP_MIN).max(0).min((fr + 1) / 2);
                                    // self-funding stops at NUCLEUS_KEEP_MIN, not at the hard
                                    // min(fr,2) floor: with a rest sitting right there, squeezing
                                    // the vowel into the 2-frame collapse region to save silence
                                    // frames is the wrong trade (the floor pass CAN go to 2, but
                                    // the rest should fund first).
                                    let deliverable =
                                        budget_here.min((durs[nuc] - NUCLEUS_KEEP_MIN).max(0));
                                    (want - left - deliverable).max(0).min((d - REST_KEEP_MIN).max(0))
                                } else {
                                    (d - REST_KEEP_MIN).max(0)
                                }
                            } else if !is_nucleus_phone(phon[j]) {
                                // consonant lender: the shipped rule next door, nothing further back.
                                if depth == 1 {
                                    (d - SUNG_KEEP_MIN).max(0).min((d + 1) / 2)
                                } else {
                                    0
                                }
                            } else if depth == 1 {
                                (d - vowel_keep).max(0).min((d + 1) / 2)
                            } else {
                                (d - NUCLEUS_KEEP_MIN).max(0).min(d / DEEP_LENDER_SHARE)
                            };
                            let take = (want - left).min(cap_j);
                            if take > 0 {
                                pdur[j] -= take;
                                borrowed.push((j, take));
                                left += take;
                            }
                            if is_rest {
                                break;
                            }
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
                        let nuc_before_supplement = durs[nuc];
                        for i in (0..onset_end).rev() {
                            // ★S92p: `CODA_MIN_FRAMES`, not a bare `2`. The InNote twin twenty lines up
                            // already uses the constant; leaving a literal here means a future round that
                            // raises the floor (the training data says 0/5608 English consonants are
                            // shorter than 3 frames) would silently move ONE arm and split the two.
                            if durs[i] >= CODA_MIN_FRAMES {
                                continue;
                            }
                            let need = CODA_MIN_FRAMES - durs[i];
                            if (durs[nuc] - nuc_floor).max(0) >= need {
                                durs[nuc] -= need;
                                durs[i] += need;
                            }
                        }
                        // ★S92c — and then keep going to the MEASURED TARGET instead of stopping at the
                        // 2-frame floor. The floor was written as a rescue for the rare no-lender case
                        // (score start); on an English line it is the NORM, because the previous phone is
                        // itself a short consonant with nothing to lend, and the rescue then hard-codes a
                        // 40 ms consonant no matter what the training data says. Measured on the user's own
                        // 283-note English track: 48 of 121 word-initial consonants sat on exactly that
                        // floor, 46 of them because `avail == 0`, and 38 of THOSE had a nucleus with 6+
                        // frames to spare — 188 frames (3.76 s) of articulation missing, and self-
                        // reinforcing, since a 2-frame consonant cannot lend to the next onset either.
                        // `s` in "seem"/"smell" wants 7 frames and got 2, with a 10-frame vowel next to it.
                        //
                        // The clamp is `(have - SUNG_KEEP_MIN).max(0).min((have+1)/2)` — the SAME shipped,
                        // ear-validated half-clamp the sung-lender borrow above uses, re-aimed at the
                        // nucleus (S89's rule: reuse a verified rule instead of inventing a constant).
                        // ⚠ Runs AFTER the all-or-nothing floor pass and skips `durs[i] == 0`, so an onset
                        // that will DROP below never holds nucleus frames — the drop pass hands its frames
                        // back to the LENDER, and that stays exact (the conservation invariant this
                        // supplement was originally written all-or-nothing to protect).
                        if chaining {
                            // The nucleus's TOTAL contribution is bounded once for the whole cluster by the
                            // InNote arm's own clamp — never below SUNG_KEEP_MIN, never more than half the
                            // note. ⚠ A per-onset clamp is NOT enough: [s t a]@10 then had each of the two
                            // onsets take "half of what is left" in turn and the vowel ended at 2 frames,
                            // i.e. the S84 collapse region. Measured, not imagined — a test caught it.
                            let cap = (fr - SUNG_KEEP_MIN).max(0).min((fr + 1) / 2);
                            // ★S96 knife ② — IN-PHRASE, the chase is additionally bounded by
                            // the funding passes' own reach: the floors always fit
                            // (never drop a consonant), but the target-chase beyond them may no
                            // longer push the vowel arbitrarily far off the beat (`ties` t took 5
                            // in-note, `flowers` 7 — while the SV reference pins these vowels to
                            // the grid). Attack-after-silence (incl. score start) is exempt: with
                            // knife ②a the rest no longer lends there, the whole onset is in-note
                            // by design, and the delayed attack IS the reference behaviour.
                            // ★S96e — knife ②b (an in-phrase cap of IN_NOTE_ATTACK_MAX on this
                            // chase) is REVERTED. It did bound the worst vowel lateness (+7 → +3),
                            // but the user's ear judged the trade backwards: with dry lenders — the
                            // norm on a dense line — every word-initial consonant fell back onto a
                            // 40-60 ms stub (`smell` s 7→2, `drowning` ɹ 7→3, `ties` t 5→3) and the
                            // report was "发音反而乱了 / 辅音和元音割裂". That is S92c's whole
                            // finding re-created, and S92c/S92e/S92j are ear-VALIDATED shipped work.
                            // The beat half of knife ② lives on in the rest arm above (②a), which
                            // buys the on-beat attack WITHOUT taking frames from articulation.
                            let floor_takes = nuc_before_supplement - durs[nuc];
                            let mut allow = (cap - floor_takes).max(0);
                            for i in (0..onset_end).rev() {
                                let t = target(ph[i]);
                                if allow == 0 || durs[i] == 0 || durs[i] >= t {
                                    continue;
                                }
                                // ★S92g: NUCLEUS_KEEP_MIN, not SUNG_KEEP_MIN. The S92c supplement was
                                // written with the lender constant and could therefore leave the note's
                                // OWN vowel at 2 frames — the collapse region — on ordinary English
                                // notes (a review's worked case: [f i l] @ 200 ms with an empty lender).
                                // The walk-back above already keeps 3 from depth 2; this is the same
                                // invariant on the in-note half, so the two halves can no longer disagree.
                                let give = (t - durs[i])
                                    .min(allow)
                                    .min((durs[nuc] - NUCLEUS_KEEP_MIN).max(0));
                                durs[nuc] -= give;
                                durs[i] += give;
                                allow -= give;
                            }
                        }
                        // ★S93 — LAST-RESORT drop rescue, NON-chaining arm only (zh/ja). Every pass
                        // above can still leave a word-initial consonant under the 2-frame minimum
                        // when the note is very short AND the adjacent lender is itself squeezed: a ja
                        // fast run pins every vowel at SUNG_KEEP_MIN = 2, so the depth-1 borrow caps
                        // at 0, and a 3-frame note's nucleus spare above its fr.min(2) floor is 1 < 2.
                        // The drop pass below then deletes the consonant and the note sings the WRONG
                        // syllable — S92k's audit found exactly two on the ja probe song (し@3fr sang
                        // "i", の@3fr sang "o") and the user confirmed both audible. That outcome is
                        // strictly worse than any in-distribution squeeze, and the real data funds
                        // one: short-note vowels at a single frame are what real zh/ja singers do
                        // (see RESCUE_LENDER_KEEP). So, ONLY when the phone would otherwise drop:
                        //   • the ADJACENT sung-vowel lender's floor relaxes SUNG_KEEP_MIN → 1, still
                        //     under the shipped ceil-half clamp on its ORIGINAL length (a rest already
                        //     lends to REST_KEEP_MIN = 1; a consonant lender stays untouchable — NOT
                        //     because 1-frame consonants are unattested (ja ɾ onset p05 = 1 exists),
                        //     but because draining a consonant re-creates the S92e cascade this file
                        //     already paid for — the walk itself gives consonants nothing at depth ≥ 2
                        //     for the same reason);
                        //   • the remainder comes from the nucleus's spare above fr.min(2), exactly
                        //     like the all-or-nothing floor pass above;
                        //   • all-or-nothing: if the two together cannot reach CODA_MIN_FRAMES, touch
                        //     NOTHING and drop exactly as before (a 1-frame consonant stays forbidden,
                        //     and a note that does not drop stays byte-identical by construction).
                        // ⚠ NOT extended to the chaining arm this round: its S92c supplement already
                        // funds onsets on ordinary notes, its tracks are ear-validated as shipped
                        // (S92j/S92o), and the en-words + three alias lanes must not move here.
                        if !chaining {
                            for i in (0..onset_end).rev() {
                                if durs[i] >= CODA_MIN_FRAMES {
                                    continue;
                                }
                                let need = CODA_MIN_FRAMES - durs[i];
                                // The depth-1 lender, mirrored from the walk above (same skip rule for
                                // a coda the S92o pre-roll just fed — never immediately re-drain it).
                                let mut extra = 0i64;
                                let j = pdur.len().wrapping_sub(1);
                                if j < pdur.len()
                                    && !coda_preroll_fed.contains(&j)
                                    && is_nucleus_phone(phon[j])
                                {
                                    let already: i64 = borrowed
                                        .iter()
                                        .filter(|(idx, _)| *idx == j)
                                        .map(|(_, t)| *t)
                                        .sum();
                                    let orig = pdur[j] + already;
                                    let cap = (pdur[j] - RESCUE_LENDER_KEEP)
                                        .max(0)
                                        .min((orig + 1) / 2 - already)
                                        .max(0);
                                    extra = need.min(cap);
                                }
                                let from_nuc = (need - extra).min((durs[nuc] - nuc_floor).max(0));
                                if extra + from_nuc < need {
                                    continue; // cannot reach the minimum — drop as before, zero state touched
                                }
                                if extra > 0 {
                                    pdur[j] -= extra;
                                    borrowed.push((j, extra));
                                }
                                durs[nuc] -= from_nuc;
                                durs[i] += extra + from_nuc;
                            }
                        }
                        // ★S96 knife ② would-drop rescue, CHAINING arm. Knife ②a stopped the rest
                        // from funding the attack — but on a TINY post-rest note the in-note passes
                        // can fail the 2-frame floor too (the nucleus's spare above min(fr,2) is
                        // under 2). Deleting the consonant sings the WRONG word, which outranks any
                        // timing consideration (the user's own "唱对最重要" rule, S93). So, ONLY
                        // when the phone would otherwise drop, the adjacent REST lends back exactly
                        // the missing frames (down to REST_KEEP_MIN, which it always could) —
                        // all-or-nothing, so a note that does not drop keeps knife ②a's on-beat
                        // attack byte-for-byte, and a non-rest neighbour changes nothing (the
                        // in-phrase drop rules are exactly the shipped ones).
                        if chaining {
                            for i in (0..onset_end).rev() {
                                if durs[i] >= CODA_MIN_FRAMES {
                                    continue;
                                }
                                let need = CODA_MIN_FRAMES - durs[i];
                                // ★S96d (review CONFIRMED): the two funding sources COMBINE, exactly
                                // like the S93 rescue next door — the first cut was single-source
                                // all-or-nothing, and [R@2fr][fined@3fr] deleted the /f/ that one
                                // rest frame plus one nucleus frame together could have saved.
                                let j = pdur.len().wrapping_sub(1);
                                let mut extra = 0i64;
                                if j < pdur.len() && matches!(phon[j], "SP" | "AP") {
                                    extra = need.min((pdur[j] - REST_KEEP_MIN).max(0));
                                }
                                let from_nuc =
                                    (need - extra).min((durs[nuc] - fr.min(2)).max(0));
                                if extra + from_nuc < need {
                                    continue; // cannot reach the floor — drop exactly as before
                                }
                                if extra > 0 {
                                    pdur[j] -= extra;
                                    borrowed.push((j, extra));
                                }
                                durs[nuc] -= from_nuc;
                                durs[i] += extra + from_nuc;
                            }
                        }
                        // sub-minimum onsets DROP (same policy as codas/medials); at this point their
                        // frames are pure LENDER frames (the supplement is all-or-nothing), so hand them
                        // back — the borrow must stay zero-sum even when it fails (conservation).
                        let mut returned = 0i64;
                        for d in durs.iter_mut().take(onset_end) {
                            if *d > 0 && *d < CODA_MIN_FRAMES {
                                returned += *d;
                                *d = 0;
                            }
                        }
                        // S92d: give them back to the exact phones they came from, newest lender first
                        // (with depth 1 — zh/ja — this ledger holds a single entry and the arithmetic is
                        // identical to the old `*pdur.last_mut() += returned`).
                        #[cfg(test)]
                        let mut returns: Vec<(usize, i64)> = Vec::new();
                        for (idx, given) in borrowed.iter().rev() {
                            if returned == 0 {
                                break;
                            }
                            let back = returned.min(*given);
                            pdur[*idx] += back;
                            returned -= back;
                            #[cfg(test)]
                            returns.push((*idx, back));
                        }
                        debug_assert_eq!(returned, 0, "a dropped onset held frames no lender gave it");
                        // S92k: record the NET take per lender (after the drop-return above), so the
                        // audit harness can say exactly how many frames left a NEIGHBOUR's vowel.
                        #[cfg(test)]
                        {
                            let mut net: Vec<(usize, i64)> = Vec::new();
                            let mut add = |idx: usize, d: i64| match net.iter_mut().find(|(i, _)| *i == idx) {
                                Some(e) => e.1 += d,
                                None => net.push((idx, d)),
                            };
                            for &(idx, given) in &borrowed {
                                add(idx, given);
                            }
                            for &(idx, back) in &returns {
                                add(idx, -back); // a hand-back is a negative take
                            }
                            ledger.extend(net.into_iter().filter(|(_, d)| *d > 0));
                        }
                    }
                    // ★S92o — CODA pre-roll: the word-final consonant of a MELISMA borrows backwards
                    //   from the held vowel, exactly as the onset pre-roll borrows from the phone before
                    //   it. User's ear, on his own track: `dear`'s /ɹ/ is audible and `ear`'s is not,
                    //   "and the second half of those two words should be the same".
                    //
                    //   He is right, and the cause is neither the phones nor the render:
                    //     • both words are V + /ɹ/ with the coda 归韵-deferred onto the LAST note of the
                    //       span; `dear` ends on a 32-frame note ⇒ /ɹ/ 12 frames, `ear` on an 8-frame one
                    //       ⇒ /ɹ/ 3 frames (60 ms). The author wrote the releases that differently.
                    //     • MEASURED in the rendered audio: `ear`'s /ɹ/ sits at −17.1 dBFS and `dear`'s at
                    //       −16.7, i.e. **the same level, ~3 dB under their own vowel** — the 3 frames are
                    //       not missing energy, 60 ms is simply too short to be heard as an /r/.
                    //     • and per-note we are IN distribution: real English gives an 8-15 frame note's
                    //       coda a median of 4-5 frames (33%); we give 3 (38%).
                    //   So the only honest fix is to make the release LONGER, and the only frames that may
                    //   pay for it belong to the same vowel that is already being held — which is what
                    //   English does anyway: r-colouring starts in the tail of the hold, not in its last
                    //   60 ms (training data: a long note's /ɹ/ coda runs 16 frames at p50, 90 at p95).
                    //
                    //   ⚠ Sized by the SPAN, not by this short release note: `coda_target_frames` keyed on
                    //   8 frames answers 3 — the same number we already have — so aiming at the note's own
                    //   bucket could never move anything. The span is the vowel's accumulated length,
                    //   walked back over the identical held phone.
                    //   ⚠ Auto arm only: "自动咬字时序 = 关" (InNote, S89) means the author placed the
                    //   consonants and nothing may cross a note boundary.
                    let mut preroll_fed = vec![false; n];
                    if timing == ArticulationTiming::Auto && nucleus_continues && n > nuc + 1 {
                        let held = ph[nuc];
                        let mut span = fr;
                        let mut j = pdur.len();
                        while j > 0 && phon[j - 1] == held {
                            j -= 1;
                            span += pdur[j];
                        }
                        let want_span: i64 =
                            ph[nuc + 1..].iter().map(|p| coda_target_frames(p, span)).sum();
                        let have: i64 = durs[nuc + 1..].iter().sum();
                        let mut need = (want_span - have).max(0);
                        // Lend from the held vowel's tail, newest first. It is the most benign lender
                        // there is — the SAME vowel, and it continues into this note, so shortening it
                        // changes no syllable's identity. Still bounded by the shipped, ear-validated
                        // half-clamp + NUCLEUS_KEEP_MIN, so it can never reach the S84 collapse region.
                        let mut j = pdur.len();
                        while need > 0 && j > 0 && phon[j - 1] == held {
                            j -= 1;
                            let d = pdur[j];
                            let cap = (d - NUCLEUS_KEEP_MIN).max(0).min((d + 1) / 2);
                            let take = need.min(cap);
                            if take > 0 {
                                pdur[j] -= take;
                                need -= take;
                                // LAST-first: the word-final release carries the cue (same policy as the
                                // in-note coda pass), each still capped at its own span-sized target.
                                // `left` MUST decrement — otherwise every coda would be handed the full
                                // `take` and conservation would break (Σ durs > Σ frames).
                                let mut left = take;
                                for i in (nuc + 1..n).rev() {
                                    if left == 0 {
                                        break;
                                    }
                                    let room = (coda_target_frames(ph[i], span) - durs[i]).max(0);
                                    let give = room.min(left);
                                    durs[i] += give;
                                    left -= give;
                                    if give > 0 {
                                        preroll_fed[i] = true;
                                    }
                                }
                                debug_assert_eq!(left, 0, "coda pre-roll took frames it could not place");
                                #[cfg(test)]
                                ledger.push((j, take));
                            }
                        }
                    }
                    // (S97: the coda floor top-up used to sit HERE, per note. It now runs once,
                    // globally, after the whole loop — see `coda_floor_top_up` below for why.)
                    for (i, (&p, &d)) in ph.iter().zip(durs.iter()).enumerate() {
                        if d <= 0 {
                            continue; // dropped medial/coda / sub-minimum onset — never emit a 0-frame phone
                        }
                        if preroll_fed[i] {
                            coda_preroll_fed.push(pdur.len());
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

    // ★S97 — CODA FLOOR TOP-UP, run ONCE over the finished arrays instead of per note.
    //
    // What it does: a chaining-language coda that ended below its own measured floor
    // (`chaining_coda_floor`) takes the shortfall from ITS OWN event's last nucleus, never below
    // NUCLEUS_KEEP_MIN. Nothing else is touched — no onset is read or written, and no phone's
    // START moves except the codas' own, so the project's beat ruler (nucleus.frame0 −
    // note.frame0) is unchanged by construction.
    //
    // ★WHY IT MOVED OUT OF THE NOTE LOOP (S96f ran it there). Inside the loop it fires BEFORE the
    // next note's onset pre-roll walks back into this note — so it could spend a nucleus frame
    // that a word-INITIAL consonant was about to borrow. Measured on the user's own track:
    // `and`'s /n/ took the frame and `mor`'s /m/ fell 3 → 2 frames, the exact S92c/S92e/S92j
    // shape whose reversal the user's ear condemned in S96e. "Fund the onset first" is this
    // file's own stated doctrine (see `allocate_in_note`'s ORDER note); the word-final release
    // gets what is genuinely left over after every borrow has settled.
    // Second, smaller gain: run last, a frame it spends can no longer be taken back — S96f spent
    // 13 nucleus frames on the user's track to keep 8 coda lifts, the rest leaking to later notes.
    //
    // zh/ja never qualify (`consonant_chaining_language`) and structurally have no coda at all
    // (measured: 0 of 1215 ja sung notes, 0 of 489 on each UTAU alias track) — the ja, ja_innote
    // and three alias lane dumps are byte-identical across this change, which is how that claim
    // is checked, not by argument.
    {
        let mut i = 0usize;
        while i < phon.len() {
            let mut j = i;
            while j + 1 < phon.len() && pevt[j + 1] == pevt[i] {
                j += 1;
            }
            let seg = i..=j;
            let chaining = g2p::Lang::from_id(plang[i]).is_some_and(consonant_chaining_language);
            if chaining && npitch[i] > 0 {
                if let Some(nuc) = seg.clone().filter(|&x| is_nucleus_phone(phon[x])).next_back() {
                    // ★S97b — a PROPORTIONAL nucleus floor on top of the absolute one.
                    //
                    // `NUCLEUS_KEEP_MIN` is an absolute 3 frames, so on a LONG note whose onset
                    // cluster already ate most of it, this pass could still take the vowel's last
                    // spare frame. Measured on the user's own material: `smell`@16fr ships
                    // [s7 m5 ɛ4 l2] — 78% of its sung frames are consonant BEFORE this pass — and
                    // the pass moved it to [s7 m5 ɛ3 l3] = 83%. The user located exactly that
                    // phrase by ear ("Can only smell that we both share 还有点割裂", 2026-08-02).
                    //
                    // The bound is measured, not invented: over 53696 real word spans in the
                    // upstream GTSinger annotation the consonant share is p50 0.37 / p75 0.53 /
                    // p95 0.76, so a note already past ~3/4 consonant is out of distribution and
                    // must not be pushed further. Keeping a quarter of the event's sung frames for
                    // the nucleus is that p95 expressed as a floor.
                    // ⚠ This does NOT fix the 78% — that is the S92c/S92e onset supplement serving
                    // BOTH cluster members their full measured target, which is ear-validated
                    // ground and needs its own round. It only stops THIS pass making it worse.
                    let sung: i64 = seg.clone().filter(|&x| !matches!(phon[x], "SP" | "AP")).map(|x| pdur[x]).sum();
                    let nuc_keep = NUCLEUS_KEEP_MIN.max(sung / 4);
                    // LAST-first: the word-final release carries the cue (this file's doctrine).
                    // A member already at ITS OWN floor is skipped, so a cluster's spare frame
                    // goes to the member that is actually short — `and`'s /n/ (upstream p25 3)
                    // rather than its /d/ (upstream p25 2), which is the S96f complaint.
                    for x in (nuc + 1..=j).rev() {
                        if matches!(phon[x], "SP" | "AP") {
                            continue;
                        }
                        let floor = chaining_coda_floor(phon[x]);
                        if pdur[x] <= 0 || pdur[x] >= floor {
                            continue;
                        }
                        let take = (floor - pdur[x]).min((pdur[nuc] - nuc_keep).max(0));
                        if take > 0 {
                            pdur[nuc] -= take;
                            pdur[x] += take;
                        }
                    }
                }
            }
            i = j + 1;
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

    Ok(ScoreArrays {
        phonemes,
        phone_dur: pdur,
        note_pitch: npitch,
        note_dur,
        note_to_phone,
        phon,
        lang: plang,
        evt: pevt,
        #[cfg(test)]
        borrow_ledger: ledger,
        #[cfg(test)]
        in_note_alloc: alloc_snap,
    })
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
    coda_too: bool,
) -> Option<Vec<i64>> {
    let mut durs = phone_dur.to_vec();
    let mut any = false;
    for i in 0..phon.len() {
        let last_nuc = (0..phon.len())
            .filter(|&j| evt[j] == evt[i] && is_nucleus_phone(phon[j]))
            .next_back();
        let final_nucleus_of_event = is_nucleus_phone(phon[i]) && last_nuc == Some(i);
        // ★S92f: a word-final CONSONANT gets the same cv-domain treatment. The user's report was
        // "I can hear the th trying to close but it is faint" — and measurement agreed: the phone is
        // allocated, but 16 of 117 codas render 12-20 dB below their own vowel (`ð` −19.7, `ɹ` −19.0,
        // `l` −17.5). A 2-4 frame coda is exactly the OOD duration this knife was invented for: the
        // model sees an in-distribution phone, produces well-formed content, and it is resampled back.
        // ⚠ The other two knives (voiceless emphasis, closure valley) cannot help these at all: they
        // skip codas by design (词尾顿挫 guard) AND they only ever fire on VOICELESS phones, while the
        // three the user named — ɹ, l, ð — are all voiced. This is the only one of the three that can.
        let short_coda = coda_too
            && !is_nucleus_phone(phon[i])
            && matches!(last_nuc, Some(n) if i > n)
            && !matches!(phon[i], "SP" | "AP");
        if note_pitch[i] > 0
            && (final_nucleus_of_event || short_coda)
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
    // S92f: the coda arm only applies where a word-final consonant exists as a phonological category —
    // the chunk is single-language by construction (`chunk_at_sp` cuts at every language change), so one
    // scalar answers it for the whole call.
    let coda_too = g2p::Lang::from_id(chunk.lang_id).is_some_and(consonant_chaining_language);
    let Some(pd_inf) =
        clarity_inflated_durs(phon, &chunk.note_pitch, &chunk.phone_dur, evt, coda_too)
    else {
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

// S92k 分配器审计件(记账核对,不是启发式)—— 同 e1/mg 姿势挂子模块,以便复用本文件的私有件
// (syllable_split / onset_target_frames / coda_target_frames / zh_hold_phone):审计件读的是
// 生产的同一份切分与同一张目标表,所以它与分配器之间**没有第二份实现可漂移**。详见该文件头注。
#[cfg(test)]
#[path = "score2cv_audit.rs"]
pub(crate) mod audit;

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
        let plan = clarity_inflated_durs(&phon, &pitch, &durs, &evt, false).expect("one short nucleus qualifies");
        assert_eq!(plan, vec![10, 2, 6, 12]);
        // nothing qualifies → None (rest-only / long-only scores take the plain path).
        assert!(clarity_inflated_durs(&phon, &pitch, &[10, 2, 12, 12], &evt, false).is_none());
        // MEDIAL vowel exclusion (refined-shape on ONE event): the ə (short, medial) must NOT
        // inflate — only the final nucleus aɪ qualifies (here long → whole plan is None).
        let phon2: Vec<&'static str> = vec!["ɹ", "ə", "f", "aɪ"];
        let evt2 = vec![0usize, 0, 0, 0];
        assert!(
            clarity_inflated_durs(&phon2, &[60; 4], &[2, 3, 2, 12], &evt2, false).is_none(),
            "medial ə never inflates (unvalidated slow-note scope, S84 review)"
        );
        // …and when the final nucleus IS short, it inflates while the medial still doesn't.
        let plan2 = clarity_inflated_durs(&phon2, &[60; 4], &[2, 3, 2, 4], &evt2, false).unwrap();
        assert_eq!(plan2, vec![2, 3, 2, 6]);
        // ★S92f: the CODA arm. `light` = [l aɪ t] on one event with a 3-frame /t/: with `coda_too` the
        // t inflates like a short nucleus would, so ScoreToCV sees an in-distribution consonant instead
        // of a 60 ms smear (the user: "I can hear the th trying to close but it is faint"; measured
        // −12..−20 dB vs the vowel on 16 of 117 codas). The vowel arm is unchanged — aɪ is long here.
        let phon3: Vec<&'static str> = vec!["l", "aɪ", "t"];
        let evt3 = vec![0usize, 0, 0];
        assert!(
            clarity_inflated_durs(&phon3, &[60; 3], &[3, 12, 3], &evt3, false).is_none(),
            "zh/ja arm: a coda never inflates (词尾顿挫 guard stays for CV languages)"
        );
        let coda = clarity_inflated_durs(&phon3, &[60; 3], &[3, 12, 3], &evt3, true).unwrap();
        assert_eq!(coda, vec![3, 12, 6], "the word-final /t/ inflates; the ONSET /l/ never does");
        // a coda that is already long stays put, and a rest is not a coda
        assert!(clarity_inflated_durs(&phon3, &[60; 3], &[3, 12, 9], &evt3, true).is_none());
        let with_rest: Vec<&'static str> = vec!["l", "aɪ", "t", "SP"];
        let plan_r = clarity_inflated_durs(&with_rest, &[60, 60, 60, 0], &[3, 12, 3, 4], &[0, 0, 0, 1], true).unwrap();
        assert_eq!(plan_r, vec![3, 12, 6, 4], "the SP is untouched (pitch 0 and not a coda)");
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
    // `pub(super)`: the S92k audit harness is a SIBLING test module and reuses these fixtures rather
    // than growing a second copy of them (HARD RULE: one source of truth, tests included).
    pub(super) struct EnOnly(g2p::WordDict);
    impl g2p::DictSource for EnOnly {
        fn zh(&self) -> Result<&g2p::ZhDict> {
            Err(UtaiError::Inference("VOCAL_DICT_MISSING: fixture".into()))
        }
        fn words(&self, lang: g2p::Lang) -> Result<&g2p::WordDict> {
            if lang == g2p::Lang::En { Ok(&self.0) } else { Err(UtaiError::Inference("VOCAL_DICT_MISSING: fixture".into())) }
        }
    }
    /// 自定义词表的英语 `DictSource` —— 需要 `resolve_west_span`(跨音符的词 + 归韵)时用它,
    /// 因为**只有走词典的多音符词才会产生真正的 `is_sustain` 音符**;`phoneme_input` 夹具全是
    /// Word,`is_sustain` 恒 false(S92p:我原先的 S92b/S92o 夹具就是这么绕过延音判据的)。
    pub(super) fn en_dicts_from(tsv: &str) -> EnOnly {
        EnOnly(g2p::WordDict::from_tsv(g2p::Lang::En, tsv))
    }

    pub(super) fn en_dicts() -> EnOnly {
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
        // ★S96 knife ②a: the attack after silence starts AT the boundary — the rest keeps all 10
        // of its frames and m takes its measured 5 from the note itself (the SV-reference post-rest
        // behaviour; pre-S96 the rest lent 5 and the vowel sat on the boundary, which the user's SV
        // comparison read as "every post-rest note comes in early"). Coda n still bounded at its
        // 4-frame target; the vowel takes the remainder.
        assert_eq!(arr.phone_dur, vec![10, 5, 41, 4, 10]);
        assert_eq!(arr.phone_dur.iter().sum::<i64>(), 70, "frame-conserving");
    }

    #[test]
    fn en_double_coda_bounded_and_dropped_when_starved() {
        let d = en_dicts();
        let arr = build_arrays_daw(&[en_evt("R", 0, 10), en_evt("fined", 69, 50)], &d, ArticulationTiming::Auto).unwrap();
        assert_eq!(arr.phon, vec!["SP", "f", "aɪ", "n", "d"]);
        // f (voiceless fricative, long-bucket p75) targets 7, coda n targets 4, the stop d 3 —
        // the 760ms flat [d] is gone AND the consonants are no longer one flat size.
        // S96 ②a: f now takes its 7 from the note (attack after silence), the rest keeps its 10.
        assert_eq!(arr.phone_dur, vec![10, 7, 36, 4, 3], "codas at their own measured targets");
        // a 3-frame note can't fit any coda at the 2-frame minimum → both drop, the nucleus survives.
        let tiny = build_arrays_daw(&[en_evt("R", 0, 10), en_evt("fined", 69, 3)], &d, ArticulationTiming::Auto).unwrap();
        assert_eq!(tiny.phon, vec!["SP", "f", "aɪ"], "starved codas DROP (never a 1-frame OOD phone)");
        // ★S96 ② would-drop rescue: a 3-frame note cannot fund f's floor from itself (nucleus spare
        // above min(3,2) is 1), so the REST hands back exactly the 2 missing frames — the one case
        // where silence still lends on the chaining arm. Without it, f is deleted: wrong word.
        assert_eq!(tiny.phone_dur, vec![8, 2, 3], "the rest funds exactly the would-drop floor");
        assert_eq!(tiny.phone_dur.iter().sum::<i64>(), 13, "still frame-conserving");
    }

    // ─── S92: the coda CLUSTER (n_coda ≥ 2) — English's shape, which zh/ja cannot produce ───
    //
    // Every expectation below is hand-derived from score2cv_dur_priors.rs + the allocator's rules,
    // never copied from a run (S87: the test only catches arithmetic if the expectation is independent).
    // Relevant coda targets, [short, mid, long] buckets:
    //   n [3,3,4]  z [5,5,6]  t [4,3,3]  s [3,5,6]  ŋ [6,6,7]  θ [7,7,7]
    // A bare vowel-initial phone list is used on purpose: with no onset there is nothing to borrow or
    // reserve, so each number below isolates the coda pass.
    pub(super) fn raw(phones: &'static str, frames: i64) -> g2p::ScoreEvt<'static> {
        g2p::ScoreEvt {
            lyric: "x", note_num: 60, frames, lang: g2p::Lang::Ja,
            phoneme_input: Some(phones), phoneme_set: PhonemeSet::Words,
        }
    }

    /// `means`@320ms — the real note from the user's English track that sang "meez".
    /// [i n z] @ 16 fr (long bucket): want = 4+6 = 10, cluster_floor = 2*2 = 4,
    /// cap = max(16*2/5, 4) = 6 ⇒ budget 6. LAST-first: z holds back 2 for the unserved n ⇒ z = min(6, 6-2) = 4,
    /// then n = min(4, 2) = 2. Nucleus = 16-6 = 10 — **exactly what it was before the fix**, because the
    /// TOTAL coda budget did not change; only its split did.
    #[test]
    fn s92_coda_cluster_shares_the_budget_instead_of_dropping_the_inner_one() {
        let arr = build_arrays_daw(&[raw("i n z", 16)], &NoDicts, ArticulationTiming::Auto).unwrap();
        assert_eq!(arr.phon, vec!["i", "n", "z"], "the inner coda must survive (pre-S92 it vanished)");
        assert_eq!(arr.phone_dur, vec![10, 2, 4]);
        assert_eq!(arr.phone_dur.iter().sum::<i64>(), 16, "frame-conserving");
    }

    /// `don't`@160ms — sang "dote". This one needs the SECOND half of the fix: at 8 frames the 2/5
    /// ceiling is 3, which cannot fund two 2-frame minimums at all, so it is raised to n_coda*2 = 4.
    /// [oʊ n t] @ 8 fr (mid bucket): want = 3+3 = 6, budget = min(6, 8-2, max(3,4)) = 4 ⇒
    /// t = min(3, 4-2) = 2, n = min(3, 2) = 2, nucleus = 4. The nucleus pays exactly ONE frame for the
    /// /n/ it used to lose entirely (pre-S92: nucleus 5, t 3, n DROPPED).
    #[test]
    fn s92_cluster_ceiling_is_raised_only_when_two_fifths_cannot_fund_the_minimums() {
        let arr = build_arrays_daw(&[raw("oʊ n t", 8)], &NoDicts, ArticulationTiming::Auto).unwrap();
        assert_eq!(arr.phon, vec!["oʊ", "n", "t"]);
        assert_eq!(arr.phone_dur, vec![4, 2, 2]);
        assert_eq!(arr.phone_dur.iter().sum::<i64>(), 8, "frame-conserving");
    }

    /// A 3-consonant coda (`strengths`) — all three survive (the S92 guarantee, unchanged), and the
    /// budget now follows the MEASURED target ratio (S96 proportional split): post-S92n targets are
    /// ŋ 10 / θ 7 / s 6 (W = 23), budget 8 ⇒ shares 3/2/2 + the leftover frame to the largest
    /// remainder (ŋ) ⇒ [4, 2, 2]. The pre-S96 [2, 2, 4] gave the word-final release the most by
    /// SERVING ORDER, not by measurement — the sonorant hugging the vowel is what carries the
    /// cluster's colour (same evidence family as `dears`' ɹ vs z).
    #[test]
    fn s92_three_consonant_coda_all_survive() {
        let arr = build_arrays_daw(&[raw("i ŋ θ s", 20)], &NoDicts, ArticulationTiming::Auto).unwrap();
        assert_eq!(arr.phon, vec!["i", "ŋ", "θ", "s"]);
        assert_eq!(arr.phone_dur, vec![12, 4, 2, 2]);
    }

    /// ★The ja/zh-neutrality guard. A SINGLE coda takes exactly the pre-S92 path: `unserved` is 0 for
    /// the only coda, and `cluster_floor` is 0, so neither half of the fix can bind. That is the whole
    /// reason no language flag is needed — the ja probe song has 0 notes with n_coda ≥ 2 out of 1215
    /// sung notes, and so do all three UTAU-alias tracks. (The behavioural proof is the byte-identical
    /// lane dump; this test exists so a future edit to the single-coda path turns something red.)
    /// [i n] @ 16 fr: budget = min(4, 14, 6) = 4 ⇒ n = 4, nucleus 12.
    /// ★S96 knife ① onset-allowance regression pin. The stress pool moves the note's fat from the
    /// FINAL nucleus to a stressed MEDIAL — but every onset-funding pass draws only from the final
    /// nucleus's remainder, and the first cut of this knife therefore DELETED standalone `flowers`'
    /// /f/ outright (sang "lowers"; the investigation probe caught it before it shipped). The
    /// allowance reserves, out of the pool, exactly what the funding passes may take — so:
    ///  • standalone (attack after silence, full-target chase): f SURVIVES at a healthy size and
    ///    the pool shrinks back toward the old split (the attack has priority there);
    ///  • in-phrase: the stressed aʊ keeps its win (9 frames vs the old clamp-4) AND f/l stay
    ///    funded — the win and the funding coexist because the reserve mirrors knife ②b's bound.
    #[test]
    fn s96_stress_pool_reserves_the_onset_allowance() {
        let d = en_dicts();
        let en = |p: &'static str, fr: i64| g2p::ScoreEvt {
            lyric: "x", note_num: 60, frames: fr, lang: g2p::Lang::En,
            phoneme_input: Some(p), phoneme_set: PhonemeSet::Words,
        };
        let a = build_arrays_daw(&[en("F L AW1 ER0 Z", 22)], &d, ArticulationTiming::Auto).unwrap();
        assert_eq!(a.phon, vec!["f", "l", "aʊ", "ɝ", "z"], "the /f/ must never be deleted");
        assert_eq!(a.phone_dur, vec![5, 4, 4, 3, 6]);
        let b = build_arrays_daw(
            &[en("N IH1 NG", 14), en("F L AW1 ER0 Z", 22), en("N AY1 T", 7)],
            &d, ArticulationTiming::Auto,
        ).unwrap();
        assert_eq!(
            b.phon,
            vec!["n", "ɪ", "ŋ", "f", "l", "aʊ", "ɝ", "z", "n", "aɪ", "t"],
            "no phone deleted in context either"
        );
        // ★S96e priority order, stated as the invariant: ARTICULATION outranks the stress balance.
        // In-phrase the onsets take their measured targets first (f reaches its full 7 — this is
        // the S92c work the user's ear validated and knife ②b had been eroding), and only what is
        // left of the note re-splits by stress; the stressed medial is merely guaranteed never to
        // fall BELOW its stress-blind share.
        let fl = &b.phone_dur[3..8];
        assert_eq!(fl[0], 7, "in-phrase, the word-initial /f/ reaches its measured target: {fl:?}");
        assert!(fl[1] >= CODA_MIN_FRAMES, "…and /l/ stays funded: {fl:?}");
        assert!(fl[2] >= 4, "…while the stressed aʊ never drops below its stress-blind share: {fl:?}");
    }



    /// ★S96d 属性 sweep(审查 CONFIRMED 的零覆盖带):带重音数字的多核词扫过整个短-中桶,
    /// 任何基线保得住音素的帧数上,重音臂也**一个都不许丢**——第一版核池在 fr∈[10,13] 把 every
    /// 的 ɝ 静默吞掉(very@7 吞 ɹ),而全部既有 sweep 都是无数字输入,对此完全失明。
    /// 同时断言与无数字基线的存活集合逐帧一致(不是时长一致——时长本来就该不同)。
    #[test]
    fn s96d_stressed_multicore_never_deletes_where_baseline_survives() {
        let d = en_dicts();
        let mk = |p: &'static str, fr: i64| g2p::ScoreEvt {
            lyric: "x", note_num: 60, frames: fr, lang: g2p::Lang::En,
            phoneme_input: Some(p), phoneme_set: PhonemeSet::Words,
        };
        for (stressed, plain) in
            [("EH1 V ER0 IY0", "EH V ER IY"), ("V EH1 R IY0", "V EH R IY"), ("AE1 F T ER0", "AE F T ER")]
        {
            for fr in 6..=24 {
                let a = build_arrays_daw(&[mk(stressed, fr)], &d, ArticulationTiming::Auto).unwrap();
                let b = build_arrays_daw(&[mk(plain, fr)], &d, ArticulationTiming::Auto).unwrap();
                assert_eq!(
                    a.phon, b.phon,
                    "{stressed}@{fr}: stress arm dropped a phone the baseline keeps (or vice versa)"
                );
                assert_eq!(
                    a.phone_dur.iter().sum::<i64>(),
                    b.phone_dur.iter().sum::<i64>(),
                    "{stressed}@{fr}: conservation"
                );
            }
        }
    }

    /// ★S96 knife ① discriminator — the user's `every`@18fr shape: [ɛ v ɝ i] with ARPABET stress
    /// EH1/ER0/IY0. Stress-blind allocation sang ɛ:3 v:3 ɝ:3 i:9 — the PRIMARY-stressed vowel got
    /// 60 ms and the word-final unstressed IY0 swallowed the remainder purely by position (the SV
    /// reference sings EV-ry, two clear segments; real long-note non-final nuclei sit at p50 = 10).
    /// With stress (S96d redistribution semantics): baseline [3,3,3,9] → nuclei keep floors
    /// (ɛ2/ɝ2/i3), the freed 8 frames re-split 3/1/1 ⇒ [7,3,4,4] — ɛ 3→7 (140 ms), the word-final
    /// unstressed i 9→4. The ja arm (raw IPA carries no digits) keeps the OLD allocation
    /// byte-for-byte — the stress channel simply never exists there, which is the whole gate.
    #[test]
    fn s96_stressed_medial_nucleus_gets_its_share() {
        let en = |p: &'static str, fr: i64| g2p::ScoreEvt {
            lyric: "x", note_num: 60, frames: fr, lang: g2p::Lang::En,
            phoneme_input: Some(p), phoneme_set: PhonemeSet::Words,
        };
        let a = build_arrays_daw(&[en("EH1 V ER0 IY0", 18)], &en_dicts(), ArticulationTiming::Auto).unwrap();
        assert_eq!(a.phon, vec!["ɛ", "v", "ɝ", "i"]);
        assert_eq!(a.phone_dur, vec![7, 3, 4, 4], "primary-stressed ɛ outweighs the unstressed tail");
        assert_eq!(a.phone_dur.iter().sum::<i64>(), 18, "frame-conserving");
        // ja raw-IPA (no digits ⇒ no stress channel): the shipped stress-blind allocation, untouched.
        let j = build_arrays_daw(&[raw("ɛ v ɝ i", 18)], &NoDicts, ArticulationTiming::Auto).unwrap();
        assert_eq!(j.phone_dur, vec![3, 3, 3, 9], "no digits ⇒ the old medial clamp + last-takes-all");
        assert_ne!(a.phone_dur, j.phone_dur, "the two regimes must actually differ on this shape");
    }

    /// ★S96 knife ②a language discriminator — the SAME "rest, then a consonant-initial note" shape
    /// under the two regimes: the CHAINING arm (en) attacks AT the boundary (the rest keeps all its
    /// frames, the onset is funded in-note = the SV-reference post-rest behaviour the user's ear
    /// ratified), while ja keeps the shipped pre-roll INTO the rest (先行発声 — what real ja singing
    /// does: 2478 real consonants live inside rest groups in the aligned corpus, and ja timing is
    /// ear-anchored S84/S89). Same shape, different language, different rule — BY DESIGN, and this
    /// test is the only place that states it side by side.
    #[test]
    fn s96_post_rest_attack_is_per_language() {
        // en: raw-IPA override so no dictionary is needed; [m i] after a 10-frame rest.
        let en = |p: &'static str, fr: i64| g2p::ScoreEvt {
            lyric: "x", note_num: 60, frames: fr, lang: g2p::Lang::En,
            phoneme_input: Some(p), phoneme_set: PhonemeSet::Words,
        };
        let rest = |fr: i64| g2p::ScoreEvt {
            lyric: "R", note_num: 0, frames: fr, lang: g2p::Lang::En,
            phoneme_input: None, phoneme_set: PhonemeSet::Words,
        };
        let a = build_arrays_daw(&[rest(10), en("M IY1", 10)], &en_dicts(), ArticulationTiming::Auto).unwrap();
        assert_eq!(a.phon, vec!["SP", "m", "i"]);
        assert_eq!(a.phone_dur[0], 10, "en: the rest keeps ALL its frames (attack on the boundary)");
        assert!(a.phone_dur[1] >= CODA_MIN_FRAMES, "…and the onset is funded in-note");
        // ja, same shape (raw IPA): the rest lends and the vowel sits on the boundary, as shipped.
        let ja_rest = g2p::ScoreEvt {
            lyric: "R", note_num: 0, frames: 10, lang: g2p::Lang::Ja,
            phoneme_input: None, phoneme_set: PhonemeSet::Words,
        };
        let j = build_arrays_daw(&[ja_rest, raw("m i", 10)], &NoDicts, ArticulationTiming::Auto).unwrap();
        assert_eq!(j.phon, vec!["SP", "m", "i"]);
        assert!(j.phone_dur[0] < 10, "ja: the rest still lends (先行発声): {:?}", j.phone_dur);
        assert_ne!(a.phone_dur, j.phone_dur, "the two regimes must actually differ on this shape");
    }

    /// ★S96 discriminator — the user's `dears` shape ([d] is borrow-funded on the Auto arm; the
    /// note itself feeds ɪ ɹ z @ 21 fr): budget = 21*2/5 = 8, targets ɹ 16 / z 6 (the en reference
    /// distribution: real long-note ɹ codas sit at p50 = 16, z at 6 — the r-colour IS the "-ears")
    /// ⇒ proportional 5(+1 largest-remainder)/2 ⇒ ɹ 6, z 2, nucleus keeps 13. Pre-S96 the
    /// LAST-first serving order handed z its full 6 and left ɹ on the 2-frame floor — the audible
    /// "dears 的 r 被吞" the user pinpointed, with the cluster ratio UPSIDE-DOWN vs measurement.
    /// (Budget itself unchanged — the 2/5 ceiling stays; raising it is the deferred r-colour call.)
    #[test]
    fn s96_dears_cluster_ratio_follows_targets() {
        let arr = build_arrays_daw(&[raw("ɪ ɹ z", 21)], &NoDicts, ArticulationTiming::Auto).unwrap();
        assert_eq!(arr.phon, vec!["ɪ", "ɹ", "z"]);
        assert_eq!(arr.phone_dur, vec![13, 6, 2], "ɹ must outweigh z (targets 16:6), budget/nucleus unmoved");
    }

    #[test]
    fn s92_single_coda_is_untouched() {
        let arr = build_arrays_daw(&[raw("i n", 16)], &NoDicts, ArticulationTiming::Auto).unwrap();
        assert_eq!(arr.phone_dur, vec![12, 4]);
    }

    /// ★The `fr * 2 / 5` ceiling had ZERO test coverage (an adversarial review measured that changing
    /// it to 1/2 or 3/5 turned no test red). Pin it on a single coda, where S92 leaves it alone:
    /// [i s] @ 10 fr (mid bucket): s targets 5 but 10*2/5 = 4 caps it ⇒ s = 4, nucleus 6.
    /// A 1/2 ceiling would give [5, 5]; 3/5 would give [5, 5]. Either turns this red.
    #[test]
    fn coda_ceiling_is_two_fifths_of_the_note() {
        let arr = build_arrays_daw(&[raw("i s", 10)], &NoDicts, ArticulationTiming::Auto).unwrap();
        assert_eq!(arr.phone_dur, vec![6, 4], "2/5 of a 10-frame note = 4");
    }

    /// The fix does NOT pretend a note has room it does not have: 5 frames cannot hold a nucleus floor
    /// plus two 2-frame codas, so the inner one still drops (loudly documented, not silently assumed).
    /// [i n z] @ 5 fr: cluster_floor = min(2*2, 5-NUCLEUS_KEEP_MIN) = 2, so the ceiling stays at
    /// max(5*2/5, 2) = 2 ⇒ budget 2 ⇒ z takes 2, n = min(3, 0) = 0 < 2 ⇒ dropped, nucleus keeps 3.
    #[test]
    fn s92_cluster_still_drops_when_the_note_physically_cannot_hold_it() {
        let arr = build_arrays_daw(&[raw("i n z", 5)], &NoDicts, ArticulationTiming::Auto).unwrap();
        assert_eq!(arr.phon, vec!["i", "z"]);
        assert_eq!(arr.phone_dur, vec![3, 2]);
        assert_eq!(arr.phone_dur.iter().sum::<i64>(), 5, "frame-conserving even when a phone drops");
    }

    /// ★S92b — the user's `even`. The word is spread over three notes, so `resolve_west_span` defers the
    /// word-final /n/ onto the LAST one, which the author wrote as a 4-frame (80 ms) hold: the note's
    /// whole content is [re-emitted ɪ, deferred n]. 2/5 of 4 frames is ONE, so no coda could exist there
    /// at any target and the word sang "even" without its n — in both choruses.
    /// Held nucleus ⇒ keep = CODA_MIN_FRAMES and the floor applies at n_coda == 1:
    /// ceiling = max(4*2/5, min(2, 4-2)) = 2 ⇒ n takes 2, the held ɪ keeps 2.
    /// The SECOND half of this test is the discriminator: the identical 4-frame [ɪ n] note preceded by a
    /// DIFFERENT vowel is not a continuation, so it keeps the pre-S92b behaviour (n dropped). Same note,
    /// same length, different history — that is what proves the predicate is doing the work, and it is
    /// also why ja/zh cannot enter this branch (a fresh syllable never continues the previous phone).
    #[test]
    fn s92b_held_nucleus_lets_the_deferred_coda_exist() {
        // ★S92p: this fixture now goes through the REAL path — a dictionary word spread over two
        // notes, so `resolve_west_span` defers the word-final /n/ AND marks the second note
        // `is_sustain`. The old fixture used two `phoneme_input` WORD notes whose phones merely
        // happened to touch; those are `is_sustain: false` and, once the predicate stopped ignoring
        // that flag, they stopped being "held" — i.e. the old fixture was exercising the mechanism
        // through the very hole S92p closes, and never covered the shape the user actually hears.
        let d = en_dicts_from("mine\tM AY1 N\n");
        let held = build_arrays_daw(
            &[en_evt("mine", 60, 12), en_evt("+", 60, 4)], &d, ArticulationTiming::Auto).unwrap();
        assert_eq!(held.phon, vec!["m", "aɪ", "aɪ", "n"], "the deferred word-final n must survive");
        // ★S92o moved this by ONE frame: the coda pre-roll sizes /n/ by the SPAN (12+4 = 16 frames ⇒
        // its measured target is 3, not the 4-frame note's 2) and takes the missing frame from the held
        // ɪ before it. Same word, same total, the release just starts one frame earlier.
        assert_eq!(held.phone_dur, vec![4, 7, 2, 3]);
        assert_eq!(held.phone_dur.iter().sum::<i64>(), 16, "frame-conserving");
        assert_eq!(held.phone_dur[2], CODA_MIN_FRAMES, "★the HELD nucleus may fall to 2 — S92b's point");
        assert!(held.phone_dur[3] >= CODA_MIN_FRAMES, "★and the deferred coda therefore fits");

        // ★判别器 1(原有):同样 4 帧的 [ɪ n],前面是**别的**元音 ⇒ 不是延续 ⇒ 恢复 pre-S92b 行为。
        let fresh = build_arrays_daw(
            &[raw("a", 12), raw("ɪ n", 4)], &NoDicts, ArticulationTiming::Auto).unwrap();
        assert_eq!(fresh.phon, vec!["a", "ɪ"], "a FRESH vowel keeps the full nucleus protection");
        assert_eq!(fresh.phone_dur, vec![12, 4]);

        // ★判别器 2(S92p 新增,补的正是老夹具漏掉的那一维):音素**恰好相同**但不是延音 ⇒
        //   仍然不算延续。这是日语「な」→「あ」和英语 `my eyes` 的形状 —— 实测日文探针歌里有
        //   169 个音符命中「首音素 == 上一个发出的音素」,它们全都是全新起音。
        let same_phone_fresh_word = build_arrays_daw(
            &[raw("v ɪ", 12), raw("ɪ n", 4)], &NoDicts, ArticulationTiming::Auto).unwrap();
        assert_eq!(
            same_phone_fresh_word.phon, vec!["v", "ɪ", "ɪ"],
            "音素相同但不是延音 ⇒ 不许当延续(否则就是 S92p 补掉的那个洞)"
        );
    }

    /// ★S92c — the onset-starvation cascade. On the Auto arm a word-initial consonant is funded by
    /// borrowing from the PREVIOUS phone; when that phone is itself a 2-frame consonant (46 of 121 cases
    /// on the user's own English track) the borrow yields nothing and the onset used to stop at the bare
    /// 2-frame rescue — a 40 ms /s/ next to a 14-frame vowel, whatever the measured target says.
    /// Hand-derived: note A `D OW1 N T`@8 ends on t:2, so note B's lender can give
    /// `(2-SUNG_KEEP_MIN).max(0)` = 0. B = `S IY1`@16 ⇒ allocate leaves [_, 16]; the floor pass gives s 2
    /// (i:14); then the target pass may take up to `cap = (16-2).min(8) = 8` from the nucleus in total, 2
    /// of which the floor pass already used ⇒ allow 6, and s's measured onset target at the long bucket
    /// is 7 ⇒ s takes 5 more. s:7 i:9.
    /// ★The second arm is the language discriminator: the SAME phones and the SAME frames under ja keep
    /// the old 2-frame rescue, because zh/ja are CV — their borrow normally works, and their fast-run
    /// behaviour is ear-verified (S84). Same input, different language, different rule — by design.
    #[test]
    fn s92c_starved_onset_reaches_its_target_in_english_only() {
        let d = en_dicts();
        let en = |p: &'static str, fr: i64| g2p::ScoreEvt {
            lyric: "x", note_num: 60, frames: fr, lang: g2p::Lang::En,
            phoneme_input: Some(p), phoneme_set: PhonemeSet::Words,
        };
        let a = build_arrays_daw(&[en("D OW1 N T", 8), en("S IY1", 16)], &d, ArticulationTiming::Auto).unwrap();
        assert_eq!(a.phon, vec!["d", "oʊ", "n", "t", "s", "i"]);
        // ★S96e: knife ②b's in-phrase cap is REVERTED (the user heard the articulation loss), so
        // this is back to the shipped S92c number — the starved /s/ reaches its measured 7 frames.
        assert_eq!(a.phone_dur, vec![2, 2, 2, 2, 7, 9], "the starved /s/ reaches its measured 7-frame target");
        assert_eq!(a.phone_dur.iter().sum::<i64>(), 24, "frame-conserving");

        // ja, identical phones and frames (raw IPA override): the lender is still empty, but the rule
        // does not apply — 2-frame rescue, exactly as before S92c.
        let j = build_arrays_daw(
            &[raw("d oʊ n t", 8), raw("s i", 16)], &NoDicts, ArticulationTiming::Auto).unwrap();
        assert_eq!(j.phon, vec!["d", "oʊ", "n", "t", "s", "i"]);
        assert_eq!(j.phone_dur, vec![2, 2, 2, 2, 2, 14], "zh/ja keep the 2-frame rescue (ear-verified)");
    }

    /// ★S92d — the walk-back borrow, and the property the user actually complained about: with S92c the
    /// starved onset was fed from its OWN nucleus, so the vowel started late ("时序有点怪"). Feeding it
    /// from FURTHER BACK restores the Auto arm's contract — **the nucleus keeps every frame it had**.
    /// Hand-derived. Note A `M AY1 N D`@20 (long bucket): coda want = n4 + d3 = 7, ceiling
    /// max(20*2/5, min(4, 20-3)) = 8 ⇒ budget 7 ⇒ d 3 (holding 2 back for n), n 4, nucleus 13. Onset m at
    /// score start: floor pass 2 (nucleus 11), then the target pass — cap (20-2).min(10) = 10, 2 already
    /// used ⇒ allow 8, m's long-bucket target 5 ⇒ m 5, nucleus 8. A = [m5, aɪ8, n2, d2]… wait: n and d
    /// then LEND to B below, which is the whole point.
    /// Note B `S IY1`@16: s's target is 7, and the two mechanisms STACK (the user asked for exactly that).
    /// Walk-back first: the immediate lender d:3 gives (3-2).min(2) = 1 (depth 1 = the shipped rule); n:4
    /// is a CONSONANT at depth 2 so it gives NOTHING (draining it would undo its own fix); aɪ:8 is a vowel
    /// at depth 3 ⇒ **S92j** (8-NUCLEUS_KEEP_MIN).min(8/DEEP_LENDER_SHARE) = 2, where it used to be
    /// .min(ceil(8/2)) = 4. That is 3 of the 7 from BEFORE the note; the in-note supplement covers the
    /// last 4 out of B's OWN nucleus ⇒ s 7, i 12.
    /// So `mind`'s vowel keeps 6 of the 8 frames it has when the note stands alone (it kept 4 before) —
    /// the "another word reached in and cut this one short" artifact the user named on `might`/`shame` —
    /// while the onset still reaches its full measured target and `n` keeps its 4 frames.
    #[test]
    fn s92d_walk_back_borrow_keeps_the_vowel_on_the_beat() {
        let d = en_dicts();
        let en = |p: &'static str, fr: i64| g2p::ScoreEvt {
            lyric: "x", note_num: 60, frames: fr, lang: g2p::Lang::En,
            phoneme_input: Some(p), phoneme_set: PhonemeSet::Words,
        };
        let a = build_arrays_daw(&[en("M AY1 N D", 20), en("S IY1", 16)], &d, ArticulationTiming::Auto).unwrap();
        assert_eq!(a.phon, vec!["m", "aɪ", "n", "d", "s", "i"]);
        // ★S96e: back to the shipped S92d/S92j numbers after knife ②b was reverted.
        assert_eq!(a.phone_dur, vec![5, 6, 4, 2, 7, 12]);
        assert_eq!(a.phone_dur[4], 7, "★the onset still reaches its measured target");
        assert_eq!(a.phone_dur[2], 4, "★a preceding CONSONANT is not drained back down (its own fix holds)");
        // ★S92j, stated as the invariant instead of as a number: the lender's UNDISTURBED duration is
        // derived from the same note standing alone, so this stays honest if a target or bucket moves.
        let alone = build_arrays_daw(&[en("M AY1 N D", 20)], &d, ArticulationTiming::Auto).unwrap();
        assert_eq!(alone.phone_dur[1], 8, "note A's own allocation (sanity anchor for the ratio below)");
        assert!(
            a.phone_dur[1] >= alone.phone_dur[1] - alone.phone_dur[1] / DEEP_LENDER_SHARE,
            "★a NON-ADJACENT vowel lender keeps at least (1 - 1/{DEEP_LENDER_SHARE}) of itself: {} of {}",
            a.phone_dur[1], alone.phone_dur[1]
        );
        assert_eq!(a.phone_dur.iter().sum::<i64>(), 36, "frame-conserving across the walk-back");

        // ja walks depth 1 = the pre-S92d single-phone rule, bit-identical: only d:3 can lend (1 frame),
        // so s falls back to the 2-frame rescue out of its own nucleus.
        let j = build_arrays_daw(
            &[raw("m aɪ n d", 20), raw("s i", 16)], &NoDicts, ArticulationTiming::Auto).unwrap();
        assert_eq!(j.phone_dur, vec![2, 11, 4, 2, 2, 15], "zh/ja: depth 1, unchanged");
        assert_eq!(j.phone_dur.iter().sum::<i64>(), 36);
    }

    /// ★S92p — **钉死现行行为,不是改它**:S92i 的「旋律头不算快段」豁免(`fr <= 5 &&
    /// !nucleus_held_by_next`)**没有语言门**,而日语的「こ」@4帧 +「ー」完全满足它(`Tok::Hold` 把
    /// 延音解析成 `Phones(vec![carrier])` + `is_sustain`,carrier 就是该 mora 的元音)。
    /// S92i 当初「日文轨逐字节相同 ⇒ 不需要语言门」的结论**是探针素材的属性,不是语言的属性**。
    ///
    /// ⚠ 这里**故意不加语言门**:豁免的道理(「短音符 + 后面接同元音延音 = 旋律头,元音不缺帧」)
    /// 本身是语言中立的;而 ja 的借帧规则更弱(`depth_limit == 1`、`vowel_keep == SUNG_KEEP_MIN`),
    /// 所以**代价**才因语言而异。在没有日语耳测的情况下改日语行为,违反本仓自己的规矩。
    /// ⇒ 把当前行为钉成 golden,让它从「没人知道」变成「已知且有据可查」。
    /// ★S93 耳测已裁(2026-08-01,用户,A/B=probe 的 s93_h2demo_{current,gated}):**维持现行豁免**
    /// ——「current 可以,gated 反而听起来有点奇怪」。该债销账;要再改这条得有新的耳测理由。
    #[test]
    fn s92p_ja_melisma_head_exemption_is_pinned_not_gated() {
        let hold = |fr: i64| g2p::ScoreEvt {
            lyric: "ー", note_num: 60, frames: fr, lang: g2p::Lang::Ja,
            phoneme_input: None, phoneme_set: PhonemeSet::Words,
        };
        // ⚠ 必须给「こ」一个**出借者**,否则两臂都会落进 2 帧兜底、测不出任何差别 ——
        //   第一版就是这么写的,是一条空测试(ja 不走 S92c 补齐,谱首又无人可借)。
        // 「こ」@4 帧,后接同元音延音 ⇒ `nucleus_held_by_next` 为真 ⇒ fr≤5 的快段封顶**不生效**。
        let melisma = build_arrays_daw(
            &[raw("a", 20), raw("k o", 4), hold(40)], &NoDicts, ArticulationTiming::Auto).unwrap();
        assert_eq!(melisma.phon, vec!["a", "k", "o", "o"]);
        assert_eq!(melisma.phone_dur, vec![16, 4, 4, 40], "现行行为(S93 耳测通过,维持豁免)");
        assert_eq!(melisma.phone_dur.iter().sum::<i64>(), 64, "conserving");

        // ★判别器:同样 4 帧的「こ」、同样的出借者,后面**不是**延音 ⇒ 豁免不适用 ⇒ 快段封顶生效,
        //   k 被钳回 2,出借者少掉一帧的支出。两者必须不同,否则这条钉死测试是空的。
        let fast = build_arrays_daw(
            &[raw("a", 20), raw("k o", 4), raw("s a", 12)], &NoDicts, ArticulationTiming::Auto).unwrap();
        assert_ne!(
            fast.phone_dur[1], melisma.phone_dur[1],
            "豁免对 ja 没有任何行为差别 ⇒ 这条测试是空的,别拿它当证据"
        );
        assert_eq!(fast.phone_dur[1], 2, "快段封顶把 k 钳在 2");
    }

    /// ★S93 — the LAST-RESORT drop rescue (non-chaining arm). The real shape from the ja probe song:
    /// a fast run pins the previous vowel at SUNG_KEEP_MIN = 2, so a 3-frame CV note's onset can
    /// neither borrow (cap 0) nor self-fund (nucleus spare 1 < 2) nor take the S92c supplement
    /// (ja is not a chaining language) — し@3fr sang "i" and の@3fr sang "o", both confirmed by the
    /// user's ear (S92k audit, evts 755/731).
    /// Hand-derived: note A `k ɯ`@4 (score start) ⇒ floor pass k:2, ɯ:2. Note B `n o`@3: borrow cap
    /// (2-2) = 0; floor needs 2, spare = 3-2 = 1 ⇒ would DROP. Rescue: the adjacent vowel relaxes to
    /// RESCUE_LENDER_KEEP = 1 under the ceil-half clamp on its ORIGINAL length (min(2-1, (2+1)/2) = 1),
    /// the nucleus pays its 1 spare frame ⇒ n:2, o:2, ɯ:1 — every value in-distribution (short-bucket
    /// nucleus p05 = 1 for ja o/ɯ/i, o/ɯ median 2; see RESCUE_LENDER_KEEP).
    #[test]
    fn s93_would_drop_onset_is_rescued_by_the_adjacent_vowel_falling_to_one_frame() {
        let arr = build_arrays_daw(
            &[raw("k ɯ", 4), raw("n o", 3)], &NoDicts, ArticulationTiming::Auto).unwrap();
        assert_eq!(arr.phon, vec!["k", "ɯ", "n", "o"], "★the word-initial consonant must survive");
        assert_eq!(arr.phone_dur, vec![2, 1, 2, 2]);
        assert_eq!(arr.phone_dur.iter().sum::<i64>(), 7, "frame-conserving across the rescue");
        // ★审查 confirmed(S93):rescue 的取帧必须进借帧账本 —— 它是 NUCLEUS_LENT_AWAY /
        //   「元音总损失帧数」轴的全新生产者,而这条接线此前零变异覆盖(删掉 `borrowed.push`
        //   全套测试照绿、审计从此对 rescue 静默失明 = S89「零覆盖接线点」的标准形状)。
        //   ɯ 在 pdur 里的下标是 1,rescue 恰取 1 帧。
        assert!(
            arr.borrow_ledger.iter().any(|&(idx, n)| idx == 1 && n == 1),
            "the rescue's take from ɯ must be ON the borrow ledger (audit visibility): {:?}",
            arr.borrow_ledger
        );

        // ★判别器 1:出借元音**富裕**(≥3 帧)时,正常借帧已经喂饱 onset ⇒ rescue 一帧不动,
        //   出借者保持 SUNG_KEEP_MIN 以上 —— rescue 只在「否则整个音素消失」时才存在。
        let rich = build_arrays_daw(
            &[raw("k a", 8), raw("n o", 3)], &NoDicts, ArticulationTiming::Auto).unwrap();
        assert_eq!(rich.phon, vec!["k", "a", "n", "o"]);
        assert_eq!(rich.phone_dur, vec![2, 4, 2, 3], "normal borrow path — rescue idle");
        assert!(rich.phone_dur[1] >= SUNG_KEEP_MIN, "a healthy lender never falls below the shipped floor");
        // (两臂的出借者余量 4 vs 1 —— 判别器与主臂确实分流,这条测试不是空的。)
        assert_ne!(rich.phone_dur[1], arr.phone_dur[1]);
    }

    /// ★S93 — the CASCADE steady state, PINNED not blessed(与 S92p 的 H2 钉法同款:钉死现行行为,
    /// 耳测待裁)。rescue 的触发前提(邻居元音被压在 2 帧)恰是它自己的产出(被救音符付完 from_nuc
    /// 后核也停在 2)⇒ 在连续 3 帧 CV 快跑里逐音符点火,稳态 = 内部元音全部 1 帧、辅音全部 2 帧。
    /// 手推([a@10] + [k a]@3 ×4):n2 从富裕出借者正常借 2;n3 借 1 + 核补 1(邻居 a 3→2);
    /// n4/n5 走 rescue(邻居 a 2→1 + 核补 1)。**pre-S93 同一份输入是隔一个音符删一个辅音**
    /// (n4 的 k 整个消失 = S92k 定罪的「唱错音节」)。
    /// ★S93 耳测已裁(2026-08-01,用户,A/B=probe 的 s93fastrun_{pre,post}):**稳态放行**——
    /// 「保辅音取舍元音是对的:极端快音硬保元音听起来唱得很拖沓;『唱对』最重要,隔个丢辅音
    /// 那是『没唱对』」。⇒ 反级联界不做(已裁,别再排期)。⚠取样面=合成快跑+teto/RVC 一条链,
    /// 真快歌真素材仍无取样面(管中窥豹,S92 全局盲区)。
    /// ⚠ 审计对该稳态的可见度有限:1 帧核 == p05 不触发 OOD(判据严格 <),2→1 不改
    /// NUCLEUS_COLLAPSE 计数(判据 ≤2)—— 位移记在 NUCLEUS_LENT_AWAY 的 deficit 里。
    #[test]
    fn s93_cascade_on_a_three_frame_run_is_pinned_awaiting_the_ear() {
        let arr = build_arrays_daw(
            &[raw("a", 10), raw("k a", 3), raw("k a", 3), raw("k a", 3), raw("k a", 3)],
            &NoDicts, ArticulationTiming::Auto).unwrap();
        assert_eq!(arr.phon, vec!["a", "k", "a", "k", "a", "k", "a", "k", "a"],
            "★every consonant in the run survives (pre-S93: every other one vanished)");
        assert_eq!(arr.phone_dur, vec![8, 2, 2, 2, 1, 2, 1, 2, 2], "级联稳态(S93 耳测放行)");
        assert_eq!(arr.phone_dur.iter().sum::<i64>(), 22, "frame-conserving across the whole run");
    }

    /// ★S93 all-or-nothing: with NO lender at all (score start) a 3-frame note still cannot reach the
    /// 2-frame minimum (nucleus spare is 1), so the onset drops exactly as before and nothing is
    /// touched — the rescue never invents frames and a 1-frame consonant stays forbidden.
    #[test]
    fn s93_rescue_stays_all_or_nothing_when_the_minimum_is_unreachable() {
        let arr = build_arrays_daw(&[raw("n o", 3)], &NoDicts, ArticulationTiming::Auto).unwrap();
        assert_eq!(arr.phon, vec!["o"], "no lender + no spare ⇒ the drop is unchanged");
        assert_eq!(arr.phone_dur, vec![3], "…and the nucleus keeps every frame (zero state touched)");
    }

    /// ★S93 — the ceil-half clamp binds on the lender's ORIGINAL length across both passes: at
    /// `n o`@2 the walk first takes 1 (cap (3-2).min(2) = 1), the floor pass has zero spare
    /// (fr = nuc_floor = 2), and the rescue may take exactly ONE more — (3+1)/2 minus the 1 already
    /// taken — landing the lender at 1. Total taken = 2 = ceil(3/2), never more.
    #[test]
    fn s93_rescue_completes_a_partial_borrow_under_the_original_half_clamp() {
        let arr = build_arrays_daw(
            &[raw("k a", 5), raw("n o", 2)], &NoDicts, ArticulationTiming::Auto).unwrap();
        assert_eq!(arr.phon, vec!["k", "a", "n", "o"]);
        assert_eq!(arr.phone_dur, vec![2, 1, 2, 2]);
        assert_eq!(arr.phone_dur.iter().sum::<i64>(), 7, "frame-conserving");
    }

    /// ★S93 语言门,两个方向都钉死:
    /// ① zh 与 ja 走同一条 `!consonant_chaining_language` 谓词(单一求值点)⇒ 同形状同获救 ——
    ///    真人数据对 zh 同样背书(短桶核 p05:zh a/i/o/u 全是 1,i 的中位数就是 1)。
    /// ② chaining 臂(en/de/fr/es/it)**故意不进 rescue**:S92c 补齐在普通音符上已喂饱 onset,
    ///    其轨道按出货状态耳测验收过(S92j/S92o),本轮 en-words + 三条别名泳道必须逐字节不变。
    ///    en 的同形状今天仍然丢弃 —— 这是**已知边界不是疏漏**,要动它得走推广清单 + en 耳测
    ///    (翻转这条断言 = 那个决定,别顺手)。
    #[test]
    fn s93_rescue_is_gated_by_the_chaining_predicate_not_by_ja() {
        let zh = |p: &'static str, fr: i64| g2p::ScoreEvt {
            lyric: "x", note_num: 60, frames: fr, lang: g2p::Lang::Zh,
            phoneme_input: Some(p), phoneme_set: PhonemeSet::Words,
        };
        let z = build_arrays_daw(&[zh("k ɯ", 4), zh("n o", 3)], &zh_dicts(), ArticulationTiming::Auto).unwrap();
        // (zh 的记号归一把裸 k 写成送气 kʰ —— 与本测试无关,rescue 才是被测物。)
        assert_eq!(z.phon, vec!["kʰ", "ɯ", "n", "o"], "zh rides the same non-chaining rescue");
        assert_eq!(z.phone_dur, vec![2, 1, 2, 2]);

        let d = en_dicts();
        let en = |p: &'static str, fr: i64| g2p::ScoreEvt {
            lyric: "x", note_num: 60, frames: fr, lang: g2p::Lang::En,
            phoneme_input: Some(p), phoneme_set: PhonemeSet::Words,
        };
        let e = build_arrays_daw(&[en("K UH1", 4), en("N OW1", 3)], &d, ArticulationTiming::Auto).unwrap();
        assert_eq!(e.phon, vec!["k", "ʊ", "oʊ"], "chaining arm: the boundary is intentional (see doc)");
        assert_eq!(e.phone_dur, vec![2, 2, 3]);
    }

    /// ★S92o — the CODA pre-roll: a melisma's deferred word-final consonant borrows backwards from the
    /// held vowel, mirroring the onset pre-roll.
    ///
    /// User's ear on his own track: `dear`'s /ɹ/ is audible, `ear`'s is not, "and the second half of
    /// those two words should be the same". He is right; the cause is the AUTHOR's note lengths —
    /// both defer /ɹ/ onto the span's last note, `dear`'s is 32 frames (⇒ /ɹ/ 12) and `ear`'s is 8
    /// (⇒ /ɹ/ 3 = 60 ms). Ruled out by measurement, in order: not the S92n clamp (`ear` never touched
    /// it), not the `fr*2/5` ceiling (dropping it for held nuclei moved 17 notes and not this one),
    /// and NOT the render — in the rendered audio `ear`'s /ɹ/ sits at −17.1 dBFS against `dear`'s
    /// −16.7, the same ~3 dB under their own vowel. 60 ms simply is not heard as an /r/.
    ///
    /// ⚠ The target must be sized by the SPAN: `coda_target_frames` keyed on the 8-frame release note
    /// answers 3 — the number we already have — so nothing could ever move.
    /// ⚠ **No distribution evidence is available for this shape.** The reverse-projected corpus has
    /// 0 of 403 notes with a held nucleus AND a coda: the training aligner never emits a repeated vowel
    /// across notes, so the melisma-release shape is OURS (the `+` sustain), not the corpus's. The
    /// evidence here is the user's ear plus English r-colouring, not `mg_truth_cmp`.
    #[test]
    fn s92o_coda_preroll_borrows_from_the_held_vowel() {
        let d = en_dicts();
        let en = |p: &'static str, fr: i64| g2p::ScoreEvt {
            lyric: "x", note_num: 60, frames: fr, lang: g2p::Lang::En,
            phoneme_input: Some(p), phoneme_set: PhonemeSet::Words,
        };
        // ★S92p: the REAL `ear` shape — the dictionary word spread over two notes, so
        // `resolve_west_span` defers /ɹ/ onto the second note AND marks it `is_sustain`.
        // (The first version of this fixture used two `phoneme_input` WORD notes; those are
        // `is_sustain: false`, i.e. it was driving the mechanism through the hole S92p closes.)
        let ed = en_dicts_from("ear\tIY1 R\n");
        let a = build_arrays_daw(
            &[en_evt("ear", 60, 40), en_evt("+", 60, 8)], &ed, ArticulationTiming::Auto).unwrap();
        assert_eq!(a.phon, vec!["i", "i", "ɹ"]);
        assert_eq!(a.phone_dur, vec![27, 5, 16], "the release is sized by the SPAN (48 frames), not by 8");
        assert_eq!(a.phone_dur.iter().sum::<i64>(), 48, "zero-sum: the timeline does not move");

        // ★判别器 1:同样的音素、同样的长度,但第二个音符是**独立的词**而不是延音 ⇒ 不许前借。
        let fresh = build_arrays_daw(&[en("IY1", 40), en("IY1 R", 8)], &d, ArticulationTiming::Auto).unwrap();
        assert_eq!(fresh.phone_dur, vec![40, 5, 3], "非延音:释放保持原来的 3 帧");

        // ★判别器 2:InNote 臂(「自动咬字时序 = 关」)的契约是不跨音符移动 ⇒ 也不许发生。
        let innote = build_arrays_daw(
            &[en_evt("ear", 60, 40), en_evt("+", 60, 8)], &ed, ArticulationTiming::InNote).unwrap();
        assert_eq!(innote.phone_dur[0], 40, "InNote must not touch the previous note");

        // ★判别器 3:出借的被延长元音**很短**时,那道半数钳位 + `NUCLEUS_KEEP_MIN` 必须咬住 ——
        //   变异实测:补这条之前,把出借上限改成「随便拿」整套自检**照样全绿**(上面几个夹具的
        //   `need` 都小于上限,钳位从没生效过 = 零覆盖)。不设限的话这里会把出借者掏成 0 帧,
        //   连音素都被丢掉(8 帧的 held 元音被拿走 8 帧)。
        let short_lender = build_arrays_daw(
            &[en_evt("ear", 60, 8), en_evt("+", 60, 8)], &ed, ArticulationTiming::Auto).unwrap();
        assert_eq!(short_lender.phon, vec!["i", "i", "ɹ"], "出借者不许被掏到 0 帧而丢音");
        assert_eq!(short_lender.phone_dur, vec![4, 5, 7], "半数钳位 + NUCLEUS_KEEP_MIN 咬住了");
        assert_eq!(short_lender.phone_dur.iter().sum::<i64>(), 16, "conserving");

        // ★★抽干保护:下一个词的**词首**辅音在深度 1 上本可以从这个 16 帧的 /ɹ/ 拿走 8 帧 ——
        //   实测(用户那首歌,加保护之前):前借喂给 `even` 的 /n/ 的 2 帧被 `feel` 的 /f/ 当场取走,
        //   帧最后落到了**下一个词**的词首,收尾一点没改善。同一形状 S92e 已经栽过一次。
        let cd = en_dicts_from("ear\tIY1 R\nfee\tF IY1\n");
        let chain = build_arrays_daw(
            &[en_evt("ear", 60, 40), en_evt("+", 60, 8), en_evt("fee", 60, 16)],
            &cd,
            ArticulationTiming::Auto,
        )
        .unwrap();
        let ri = chain.phon.iter().position(|&p| p == "ɹ").unwrap();
        assert_eq!(chain.phon[ri], "ɹ");
        assert_eq!(chain.phone_dur[ri], 16, "刚被前借喂饱的 coda 不许再被下一个 onset 抽干");
        assert_eq!(chain.phone_dur.iter().sum::<i64>(), 64, "conserving");
    }

    /// ★S92n — the coda duration target is no longer clamped at 7 frames.
    ///
    /// 反投影对拍(36 个真人英语乐句 / 403 音符,`mg_truth_cmp`)给的是字面证据:含 `ɹ` coda 的
    /// **11 个音符里,我们给 `ɹ` 的帧数每一个都恰好是 7** —— 那就是钳位本身,不是分配器的算术;
    /// 真人给 10-54 帧(`d:3 ɔ:6 ɹ:54` vs 我们 `d:3 ɔ:55 ɹ:7`)。放开后 `ɹ` coda 长音桶 7→16。
    ///
    /// ⚠ **onset 的 7 没动**:onset 靠向邻居借帧,抬它等于加剧 S92j 刚修好的「掏空邻居元音」。
    /// ⚠ zh/ja 之所以逐字节不变,**不是语言门,是结构**:zh 韵母是原子 token(n_coda=0),ja 的
    /// coda 只有 `ɴ`/`ʔ` —— 变了的那些格子(`ɹ`/`ʁ`/`ŋ`/`tʃ`…)在它们的材料里从不出现在 coda 位置。
    /// 这条由**逐字节泳道**证明,不由这段话证明。
    #[test]
    fn s92n_coda_target_is_no_longer_clamped_at_seven() {
        let d = en_dicts();
        let en = |p: &'static str, fr: i64| g2p::ScoreEvt {
            lyric: "x", note_num: 60, frames: fr, lang: g2p::Lang::En,
            phoneme_input: Some(p), phoneme_set: PhonemeSet::Words,
        };
        // 先钉表本身:长音桶的 coda 目标必须真的超过旧上限,否则下面测的是空气。
        assert!(coda_target_frames("ɹ", 28) > 7, "ɹ 的 coda 目标还被钳在 7 —— 表没重生成?");
        assert_eq!(onset_target_frames("ɹ", 28), 7, "onset 的上限**不该**动");

        // [i ɹ] @28:budget = min(want 16, 28-2-0, 28*2/5=11) = 11 ⇒ ɹ 11(旧值 7),核拿余量。
        let a = build_arrays_daw(&[en("IY1 R", 28)], &d, ArticulationTiming::Auto).unwrap();
        assert_eq!(a.phon, vec!["i", "ɹ"]);
        assert_eq!(a.phone_dur, vec![17, 11]);
        assert_eq!(a.phone_dur.iter().sum::<i64>(), 28, "守恒");
        // ★判别器:`fr*2/5` 那道预算仍然在管事 —— 目标 16 拿不满,拿到的是 11。
        assert!(a.phone_dur[1] < coda_target_frames("ɹ", 28), "预算上限没生效?那是另一个 bug");

        // ★不是所有辅音都跟着涨:`t` 的 coda 目标本来就低于旧钳位,一帧不该变。
        let t = build_arrays_daw(&[en("AA1 T", 28)], &d, ArticulationTiming::Auto).unwrap();
        assert_eq!(t.phone_dur, vec![25, 3], "t 的 coda 目标(3)与钳位无关,不该被这一刀带动");
    }

    /// ★S92j — the OTHER half of the same round, and the one that only showed up once the deep clamp was
    /// tightened: an ADJACENT vowel lender had no collapse-region protection at all. Depth 1 kept only
    /// `SUNG_KEEP_MIN` = 2, which is exactly the 2-frame cv/decoder collapse S84 measured, while the
    /// in-note supplement (S92g) and the deep steps both already kept `NUCLEUS_KEEP_MIN` = 3. So the
    /// demand the quarter-clamp pushed off the deep lenders landed on the neighbour instead and put it
    /// where no other path was allowed to: measured on the user's track, `so`@8fr came out `s:7 oʊ:2`.
    ///
    /// Fixture: `S OW1`@8 leaves the vowel at 4 frames on its own; the next note's /m/ then borrows from
    /// it at depth 1. English keeps 3, the shipped zh/ja rule keeps 2 — the gate is what makes the ja
    /// lane byte-identical (verified on the full 4838-frame track, SHA256 equal).
    #[test]
    fn s92j_adjacent_vowel_lender_stays_out_of_the_collapse_region() {
        let d = en_dicts();
        let en = |p: &'static str, fr: i64| g2p::ScoreEvt {
            lyric: "x", note_num: 60, frames: fr, lang: g2p::Lang::En,
            phoneme_input: Some(p), phoneme_set: PhonemeSet::Words,
        };
        let alone = build_arrays_daw(&[en("S OW1", 8)], &d, ArticulationTiming::Auto).unwrap();
        assert_eq!(alone.phon, vec!["s", "oʊ"]);
        assert_eq!(alone.phone_dur, vec![4, 4], "the lender's undisturbed allocation");

        let a = build_arrays_daw(&[en("S OW1", 8), en("M IY1", 10)], &d, ArticulationTiming::Auto).unwrap();
        assert_eq!(a.phon, vec!["s", "oʊ", "m", "i"]);
        assert_eq!(a.phone_dur, vec![4, 3, 4, 7]);
        assert!(
            a.phone_dur[1] >= NUCLEUS_KEEP_MIN,
            "★an adjacent vowel lender never lands in the 2-frame collapse region: {}",
            a.phone_dur[1]
        );
        assert_eq!(a.phone_dur[2], 4, "the onset still gets fed (this is not a rollback of S92c/d)");
        assert_eq!(a.phone_dur.iter().sum::<i64>(), 18, "conserving");

        // zh/ja: the shipped clamp, unchanged — the vowel may still go to SUNG_KEEP_MIN. Their fast-run
        // timing is the ear-validated contract (S84 あたし), so it does not move without its own ear test.
        let j0 = build_arrays_daw(&[raw("s oʊ", 6)], &NoDicts, ArticulationTiming::Auto).unwrap();
        assert_eq!(j0.phone_dur, vec![2, 4], "ja lender's undisturbed allocation");
        let j = build_arrays_daw(
            &[raw("s oʊ", 6), raw("m i", 10)], &NoDicts, ArticulationTiming::Auto).unwrap();
        assert_eq!(j.phone_dur, vec![2, 2, 2, 10], "zh/ja keep SUNG_KEEP_MIN (gate must hold)");
        assert_eq!(j.phone_dur.iter().sum::<i64>(), 16, "conserving");
    }

    /// ★S92h — the per-language voiceless zero-permille table: well-formed, and actually consulted for
    /// the languages it is meant for while zh/ja stay on the pooled column byte-for-byte.
    #[test]
    fn s92h_per_language_zero_permille_table_is_wellformed_and_gated() {
        use super::super::score2cv_dur_priors::PHONE_ZERO_PERMILLE_LANG as L;
        use super::is_voiceless_phone;
        let mut seen = std::collections::HashSet::new();
        for &(lg, tok, z) in L {
            assert!(seen.insert((lg, tok)), "duplicate row {lg}/{tok}");
            assert!(
                (0..7).filter_map(g2p::Lang::from_id).any(|l| l.code() == lg),
                "unknown language code {lg}"
            );
            assert!(is_voiceless_phone(tok), "{tok} is voiced — it can never be consulted");
            assert!(dur_prior(tok).is_some(), "{tok} has no pooled row to fall back to");
            for v in z {
                assert!((0..=1000).contains(&v), "{lg}/{tok} permille out of range: {z:?}");
            }
            assert!(z.iter().any(|&v| v != 0), "{lg}/{tok} is all-zero — the row means nothing");
        }
        // BEHAVIOUR, not just shape. English /t/ on a short note: its own data says 480 permille where
        // the pooled column (dragged down by Chinese) says 195 — and ja/zh must NOT move.
        let short = 5; // bucket 0
        let pooled = dur_prior("t").unwrap().2[0];
        assert_eq!(pooled, 195, "pooled /t/ short-bucket permille (regenerate changed it?)");
        assert_eq!(voiceless_zero_permille("t", short, g2p::Lang::En), 480, "en uses its own");
        assert_eq!(voiceless_zero_permille("t", short, g2p::Lang::Ja), pooled, "ja stays pooled");
        assert_eq!(voiceless_zero_permille("t", short, g2p::Lang::Zh), pooled, "zh stays pooled");
        // a bucket the language has no data for falls back rather than emitting the 0 sentinel
        let s_pooled = dur_prior("s").unwrap().2[0];
        assert_eq!(voiceless_zero_permille("s", short, g2p::Lang::En), s_pooled, "0 cell → pooled");
        // an unmapped token keeps the historical full-window default
        assert_eq!(voiceless_zero_permille("zzz", short, g2p::Lang::En), 1000);
    }

    /// ★S92g — the two branches that could still leave a note's vowel in the 2-frame collapse region
    /// S84 measured, each pinned at the exact shape a mutation run reproduced it on:
    ///   • the MEDIAL pass bounded itself by `nuc_floor` (2) instead of `NUCLEUS_KEEP_MIN`, so a
    ///     multi-syllable word crammed onto one note ate its own vowel — `ə n ə n ə`@6 gave [2,2,2].
    ///   • the S92c onset supplement used the LENDER constant `SUNG_KEEP_MIN` (2) on the note's OWN
    ///     vowel — `f i l`@10 with no lender left the vowel at 2.
    /// Both now keep 3. Reverting either constant turns exactly one of these red (verified by mutation).
    ///
    /// ⚠ HONEST SCOPE: "the nucleus is never below 3 when the note can afford it" is NOT a property of
    /// this allocator today. The S83 all-or-nothing floor pass and the S89 InNote onset reservation can
    /// both still land on 2 (`ɹ ə f aɪ n d`@10 InNote = [4,2,2,2]) — a property-sweep version of this
    /// test found them one after another. They are ear-validated shipped clamps, so they are RECORDED in
    /// pending_cleanups for their own round with its own listening test, not widened here at the end of
    /// an unrelated one.
    #[test]
    fn s92g_medial_and_supplement_no_longer_eat_the_vowel() {
        // case 1 (medial pass) is language-neutral, so a raw ja fixture exercises it;
        // case 2 (the S92c supplement) only runs for consonant-chaining languages, so it MUST be an
        // English event — a ja fixture would silently test nothing (this test caught exactly that).
        let d = en_dicts();
        let en_note = |p: &'static str, fr: i64| g2p::ScoreEvt {
            lyric: "x", note_num: 60, frames: fr, lang: g2p::Lang::En,
            phoneme_input: Some(p), phoneme_set: PhonemeSet::Words,
        };
        let cases: [(Vec<g2p::ScoreEvt<'static>>, &'static str, i64); 2] = [
            (vec![raw("ə n ə n ə", 6)], "ə n ə n ə", 6),
            (vec![en_note("S IY1 N Z", 10)], "S IY1 N Z", 10),
        ];
        for (score, shape, fr) in cases {
            let arr = if shape.starts_with('S') {
                build_arrays_daw(&score, &d, ArticulationTiming::Auto).unwrap()
            } else {
                build_arrays_daw(&score, &NoDicts, ArticulationTiming::Auto).unwrap()
            };
            assert_eq!(arr.phone_dur.iter().sum::<i64>(), fr, "conservation {shape}@{fr}");
            let nuc = arr
                .phon
                .iter()
                .enumerate()
                .filter(|(_, p)| is_nucleus_phone(p))
                .map(|(i, _)| arr.phone_dur[i])
                .next_back()
                .expect("the shape has a nucleus");
            assert!(nuc >= NUCLEUS_KEEP_MIN, "{shape}@{fr}: nucleus {nuc} ({:?})", arr.phone_dur);
        }
    }

    /// Property sweep over note length for a cluster — three invariants, and the FIRST version of this
    /// test got the third one wrong in a way worth recording: it demanded every position be monotone,
    /// including the NUCLEUS. The nucleus takes the remainder, so the moment a coda becomes affordable
    /// the vowel must give frames back — non-monotone BY CONSERVATION, not by bug. What the test found
    /// while stating it wrongly was real, though: the cluster ceiling had pinned the vowel at 2 frames
    /// at fr=4 (the S84 collapse region) — hence `NUCLEUS_KEEP_MIN`.
    ///   1. conservation: Σ durs == fr
    ///   2. CONSONANTS are monotone in fr (a longer note may never shorten or delete one — the S89 property)
    ///   3. the nucleus never lands in the ≤2-frame collapse region once the note can afford 3
    #[test]
    fn s92_cluster_allocation_sweep_over_note_length() {
        let mut prev = vec![0i64; 3];
        for fr in 3..=60 {
            let arr = build_arrays_daw(&[raw("i n z", fr)], &NoDicts, ArticulationTiming::Auto).unwrap();
            assert_eq!(arr.phone_dur.iter().sum::<i64>(), fr, "conservation at fr={fr}");
            let mut by_pos = vec![0i64; 3]; // 0 = a dropped phone, so presence is comparable across fr
            for (p, d) in arr.phon.iter().zip(arr.phone_dur.iter()) {
                by_pos[["i", "n", "z"].iter().position(|x| x == p).expect("known phone")] = *d;
            }
            for k in [1usize, 2] {
                assert!(by_pos[k] >= prev[k], "fr={fr} shortened/deleted a consonant: {prev:?} -> {by_pos:?}");
            }
            assert!(
                by_pos[0] >= NUCLEUS_KEEP_MIN.min(fr),
                "fr={fr} left the nucleus in the collapse region: {by_pos:?}"
            );
            prev = by_pos;
        }
    }

    /// ★S96f — the user's 2026-08-01 report ("其他地方的 l 处理的也有点割裂 / and 听起来像错的"):
    /// `fr*2/5` on a 7-8 frame note funds exactly 2 frames of coda, and the rendered audio showed
    /// those consonants at NORMAL energy (−0.7…+1.6 dB vs their own vowel) — not weak, just too
    /// short to read as themselves, while real English singing puts en coda p05 at 3 frames.
    /// The top-up runs AFTER every onset-funding pass (an earlier cut raised the floor inside the
    /// allocator instead and deleted `don't`'s /d/), and stops at NUCLEUS_KEEP_MIN.
    /// Three arms in one test: it fires (call/and), it does NOT fire when the nucleus cannot pay,
    /// and ja is untouched.
    #[test]
    fn s97_coda_reaches_its_own_upstream_floor_when_the_nucleus_can_pay() {
        let d = en_dicts();
        let en = |p: &'static str, fr: i64| g2p::ScoreEvt {
            lyric: "x", note_num: 60, frames: fr, lang: g2p::Lang::En,
            phoneme_input: Some(p), phoneme_set: PhonemeSet::Words,
        };
        // single coda on a dense-line note: /l/ 2 → 3, paid by the vowel's spare above its floor
        // a note with a nucleus that HAS spare pays for the release out of it
        let a = build_arrays_daw(&[en("M AY1", 8), en("K AO1 L", 11)], &d, ArticulationTiming::Auto).unwrap();
        assert_eq!(a.phon, vec!["m", "aɪ", "k", "ɔ", "l"]);
        assert_eq!(a.phone_dur[4], chaining_coda_floor("l"), "the /l/ reaches the real-singer floor: {:?}", a.phone_dur);
        assert_eq!(chaining_coda_floor("l"), 3, "upstream en word-final /l/ p25=3 p50=6 (n=3300)");
        assert!(a.phone_dur[3] >= NUCLEUS_KEEP_MIN, "…and the vowel keeps its floor: {:?}", a.phone_dur);
        assert_eq!(a.phone_dur.iter().sum::<i64>(), 19, "conservation");
        // ★and the honest limit, pinned so nobody "fixes" it by squeezing the vowel: at 7 frames a
        // CVC note cannot pay — onset 3 (ear-validated S92c) + nucleus 3 (S84 collapse guard) + the
        // release leaves exactly 2. The user's `call`/`will`/`tell` are this shape, and the upstream
        // annotation puts a word-final /l/ at p50 = 6, so we really are short here — but the frames
        // do not exist on this note. ⚠ S97 measured the obvious escape and it is WRONG: letting the
        // release ride into the next attack is the opposite of what real singers do (after a
        // coda-final short group the NEXT group's nucleus arrives EARLIER, p50 4 frames vs 6 after
        // an open one). Whatever buys this, it is not delaying the next note.
        let tight = build_arrays_daw(&[en("M AY1", 8), en("K AO1 L", 7)], &d, ArticulationTiming::Auto).unwrap();
        assert_eq!(tight.phone_dur[4], CODA_MIN_FRAMES, "cannot pay ⇒ untouched: {:?}", tight.phone_dur);
        assert!(tight.phone_dur[3] >= NUCLEUS_KEEP_MIN, "…and the vowel is NOT squeezed: {:?}", tight.phone_dur);
        // ★S97 cluster: LAST-first still SERVES first, but a member already at ITS OWN floor is
        // skipped — so `and`'s spare frame goes to the /n/ (upstream p25 3 / p50 5) instead of the
        // /d/ (upstream p25 2 / p50 2). S96f gave it to the /d/ and the user heard it as
        // "尾辅音过重". This is the whole point of making the floor per-phone.
        let b = build_arrays_daw(&[en("M AY1", 8), en("AH0 N D", 8)], &d, ArticulationTiming::Auto).unwrap();
        assert_eq!(b.phon, vec!["m", "aɪ", "ə", "n", "d"], "no phone deleted");
        assert_eq!(b.phone_dur[4], CODA_MIN_FRAMES, "/d/ stays at its own floor: {:?}", b.phone_dur);
        assert_eq!(b.phone_dur[3], 3, "/n/ gets the spare frame instead: {:?}", b.phone_dur);
        assert!(b.phone_dur[3] + b.phone_dur[4] >= 5, "the cluster gains: {:?}", b.phone_dur);
        // ja: same CVC shape, byte-identical to the shipped allocation (no top-up there)
        let j = build_arrays_daw(&[raw("k a ɴ", 7)], &NoDicts, ArticulationTiming::Auto).unwrap();
        assert_eq!(j.phone_dur[2], CODA_MIN_FRAMES, "ja coda keeps the shipped floor: {:?}", j.phone_dur);
    }

    /// ★S97b — the coda top-up must not push an already consonant-heavy note further out of
    /// distribution. `NUCLEUS_KEEP_MIN` is absolute (3 frames), so on a LONG note whose onset
    /// cluster already ate most of it the pass could still take the vowel's last spare frame:
    /// `smell`@16fr ships [s7 m5 ɛ4 l2], 78% consonant BEFORE the pass, and the pass moved it to
    /// [s7 m5 ɛ3 l3] = 83% — the user located exactly that phrase by ear. Real singing puts the
    /// per-word consonant share at p50 0.37 / p75 0.53 / p95 0.76 (53696 upstream word spans), so
    /// the nucleus keeping a quarter of the event's sung frames IS that p95 expressed as a floor.
    #[test]
    fn s97b_coda_top_up_respects_a_proportional_nucleus_floor() {
        let d = en_dicts_from("smell\tS M EH1 L\nmine\tM AY1\nthat\tDH AE1 T\nand\tAH0 N D\n");
        let ev = |lyric: &'static str, fr: i64| g2p::ScoreEvt {
            lyric, note_num: 60, frames: fr, lang: g2p::Lang::En,
            phoneme_input: None, phoneme_set: PhonemeSet::Words,
        };
        // The user's own shape, reproduced end to end: `smell` allocates its /l/ at 3, the NEXT
        // word's onset borrows that third frame straight back, and this pass would then re-fill it
        // out of a vowel that is already only 4 of the note's 18 sung frames (78% consonant).
        let a = build_arrays_daw(
            &[ev("mine", 12), ev("smell", 16), ev("that", 17)], &d, ArticulationTiming::Auto,
        ).unwrap();
        assert_eq!(&a.phon[2..6], &["s", "m", "ɛ", "l"], "{:?}", a.phon);
        let nuc = a.phone_dur[4];
        let sung: i64 = a.phone_dur[2..6].iter().sum();
        assert!(
            nuc >= sung / 4,
            "the nucleus keeps its proportional share: nuc={nuc} of {sung} sung frames, {:?}",
            a.phone_dur
        );
        // Pinned literally: WITH the guard `smell` keeps [s7 m5 ɛ4 l2]; WITHOUT it the pass takes
        // the vowel's last frame and it becomes [.. ɛ3 l3] = 83% consonant, past the real p95 of
        // 0.76 (53696 upstream word spans). The mutation test is what proves that, not this line.
        assert_eq!(&a.phone_dur[2..6], &[7, 5, 4, 2], "the guard held: {:?}", a.phone_dur);
        // DISCRIMINATOR: a note whose vowel is NOT starved still gets its coda lifted — otherwise
        // this guard would have silently disabled the whole pass.
        let b = build_arrays_daw(&[ev("mine", 8), ev("and", 8)], &d, ArticulationTiming::Auto).unwrap();
        let n_i = b.phon.iter().position(|&p| p == "n").unwrap();
        assert_eq!(b.phone_dur[n_i], chaining_coda_floor("n"), "the pass still fires normally: {:?}", b.phone_dur);
        assert_ne!(
            a.phone_dur[5], b.phone_dur[n_i],
            "the two arms must really differ (the guard binds in one and not the other)"
        );
    }

    /// ★S97 — ORDER: the coda floor top-up must run AFTER every borrow, not inside the note loop.
    ///
    /// The shape that decides it is the user's own (`tyfd` evt28→29, `and` then `mor`): a note
    /// whose coda is under its floor, immediately followed by a note whose WORD-INITIAL consonant
    /// funds itself by borrowing backwards. Run per-note (S96f), the top-up spends the nucleus
    /// frame first and the onset falls to 2 — the S92c/S92e/S92j reversal the user's ear
    /// condemned. Run globally, the onset is already funded and the release takes the remainder.
    ///
    /// The two arms really are different: with the pass in the note loop this fixture gives
    /// `m` = 2, with it after the loop `m` = 3, so deleting or moving the pass turns this red.
    #[test]
    fn s97_coda_top_up_never_outbids_the_next_words_onset() {
        let d = en_dicts();
        let en = |p: &'static str, fr: i64| g2p::ScoreEvt {
            lyric: "x", note_num: 60, frames: fr, lang: g2p::Lang::En,
            phoneme_input: Some(p), phoneme_set: PhonemeSet::Words,
        };
        let a = build_arrays_daw(
            &[en("M AY1", 8), en("AH0 N D", 8), en("M AO1 R", 7)],
            &d,
            ArticulationTiming::Auto,
        )
        .unwrap();
        assert_eq!(a.phon, vec!["m", "aɪ", "ə", "n", "d", "m", "ɔ", "ɹ"], "{:?}", a.phon);
        let m_onset = a.phone_dur[5];
        assert!(
            m_onset >= 3,
            "the next word's onset must be funded FIRST — got m={m_onset}, all={:?}",
            a.phone_dur
        );
        // ★The honest half, pinned so nobody later "improves" it into the S96f behaviour: once the
        // onset has borrowed, `and`'s nucleus IS at NUCLEUS_KEEP_MIN, so the /n/ genuinely cannot
        // be topped up and stays at the emission floor. That is the trade this ordering makes —
        // a word-INITIAL consonant outranks a word-final one when only one frame exists.
        assert_eq!(a.phone_dur[2], NUCLEUS_KEEP_MIN, "the nucleus really is out of spare: {:?}", a.phone_dur);
        assert_eq!(a.phone_dur[3], CODA_MIN_FRAMES, "…so the /n/ is NOT topped up: {:?}", a.phone_dur);
        // DISCRIMINATOR — the same word with NO following onset to outbid it: the nucleus keeps its
        // spare and the /n/ does reach its floor. Two arms, genuinely different (S92p rule).
        let b = build_arrays_daw(&[en("M AY1", 8), en("AH0 N D", 8)], &d, ArticulationTiming::Auto).unwrap();
        assert_eq!(b.phone_dur[3], chaining_coda_floor("n"), "no rival ⇒ the /n/ gets its floor: {:?}", b.phone_dur);
        assert_ne!(a.phone_dur[3], b.phone_dur[3], "the two arms must really differ");
        assert_eq!(a.phone_dur.iter().sum::<i64>(), 23, "conservation");
        assert_eq!(b.phone_dur.iter().sum::<i64>(), 16, "conservation");
    }

    /// ★S96d — the sweep the shipped one was BLIND to (review CONFIRMED). `[i n z]` has two members
    /// whose priors scale near-proportionally across the buckets, so it can never expose a RATIO
    /// flip; `ɹ`-family clusters can (ɹ coda = [3, 3, 16]) and did: at the 15→16 frame bucket
    /// boundary the word-final consonant used to halve while the note got LONGER. Sweeps every real
    /// English cluster family that contains a member with a steep bucket step.
    #[test]
    fn s96d_cluster_monotonicity_over_bucket_boundaries() {
        for phones in [
            "ɪ ɹ z",   // dears / years / tears — the user's own shape
            "ɪ ɹ s",   // pierce
            "ɑ ɹ t",   // art
            "ɪ ŋ z",   // things
            "aɪ n d",  // find
            "ɛ l p",   // help
            "i ŋ θ s", // strengths (three-member)
        ] {
            let toks: Vec<&str> = phones.split(' ').collect();
            let mut prev = vec![0i64; toks.len()];
            for fr in 3..=60 {
                let arr =
                    build_arrays_daw(&[raw(phones, fr)], &NoDicts, ArticulationTiming::Auto).unwrap();
                assert_eq!(arr.phone_dur.iter().sum::<i64>(), fr, "{phones}@{fr}: conservation");
                // 0 = dropped, so presence stays comparable across note lengths
                let mut by_pos = vec![0i64; toks.len()];
                let mut seen = 0usize;
                for (p, d) in arr.phon.iter().zip(arr.phone_dur.iter()) {
                    // walk forward so a repeated phone maps to its own slot
                    while seen < toks.len() && toks[seen] != *p {
                        seen += 1;
                    }
                    assert!(seen < toks.len(), "{phones}@{fr}: unexpected phone {p}");
                    by_pos[seen] = *d;
                    seen += 1;
                }
                for k in 1..toks.len() {
                    assert!(
                        by_pos[k] >= prev[k],
                        "{phones}: fr={fr} SHORTENED a consonant vs fr={}: {prev:?} -> {by_pos:?}",
                        fr - 1
                    );
                }
                prev = by_pos;
            }
        }
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
        // …and the Auto arm demonstrably DOES cross a boundary — ★S96 ②a made the two arms
        // COINCIDE on the post-rest shape (both attack at the boundary now), so the anti-vacuous
        // discriminator moves to an IN-PHRASE shape: after a sung word, Auto pre-rolls m out of
        // the neighbour's material while InNote takes it from its own nucleus.
        let on = build_arrays_daw(&score, &d, ArticulationTiming::Auto).unwrap();
        assert_eq!(on.phone_dur, vec![10, 5, 41, 4], "post-rest: Auto now attacks on the boundary too");
        let phrase = [en_evt("mine", 69, 16), en_evt("mine", 69, 50)];
        let on2 = build_arrays_daw(&phrase, &d, ArticulationTiming::Auto).unwrap();
        let off2 = build_arrays_daw(&phrase, &d, ArticulationTiming::InNote).unwrap();
        assert_ne!(on2.phone_dur, off2.phone_dur, "in-phrase, the two arms must still differ");
        assert!(
            on2.phone_dur[..3].iter().sum::<i64>() < 16,
            "Auto borrowed from the first word's material: {:?}",
            on2.phone_dur
        );
        assert_eq!(
            off2.phone_dur[..3].iter().sum::<i64>(),
            16,
            "InNote: the first word's frames stay its own: {:?}",
            off2.phone_dur
        );
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
    // ★S93 changed this fixture: the original lender was a VOWEL (あ@3), and the drop rescue now
    // legitimately saves /t/ there (see the second half). The return path needs a lender the rescue
    // is structurally barred from — a CONSONANT (no data says a 1-frame consonant exists).
    #[test]
    fn starved_onset_drops_and_returns_its_borrowed_frames() {
        // [ʔ,t,a] (raw-IPA override) on a 2-frame note after [a t] (coda t:3): the consonant lender
        // spares (3-2).min(2) = 1 frame, the nucleus (min(fr,2)=2) has no spare, and the rescue may
        // not touch a consonant lender → the 1-frame inner [t] drops and its borrowed frame RETURNS
        // to the lender (ʔ got nothing and drops too).
        let tta = g2p::ScoreEvt {
            lyric: "x", note_num: 62, frames: 2, lang: g2p::Lang::Ja, phoneme_input: Some("ʔ t a"),
            phoneme_set: PhonemeSet::Words,
        };
        let at = g2p::ScoreEvt {
            lyric: "x", note_num: 60, frames: 10, lang: g2p::Lang::Ja, phoneme_input: Some("a t"),
            phoneme_set: PhonemeSet::Words,
        };
        let daw = build_arrays_daw(&[at, tta.clone()], &NoDicts, ArticulationTiming::Auto).unwrap();
        assert_eq!(daw.phon, vec!["a", "t", "a"]);
        assert_eq!(daw.phone_dur, vec![7, 3, 2], "borrowed frame returned to the CONSONANT lender");

        // ★S93 判别器:同一个 [ʔ t a]@2,出借者换成**元音**(原夹具形状)⇒ rescue 生效,
        //   内侧的 t 活下来(LAST-first:贴着元音的辅音承载音节身份),ʔ 仍丢弃 ——
        //   唱「た」而不是唱「あ」。出借元音落到 RESCUE_LENDER_KEEP = 1。
        let rescued = build_arrays_daw(
            &[g2p::ScoreEvt::ja(&("あ", 60, 3)), tta], &NoDicts, ArticulationTiming::Auto).unwrap();
        assert_eq!(rescued.phon, vec!["a", "t", "a"], "the vowel-lender arm is RESCUED since S93");
        assert_eq!(rescued.phone_dur, vec![1, 2, 2]);
        assert_eq!(rescued.phone_dur.iter().sum::<i64>(), 5, "frame-conserving");
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
