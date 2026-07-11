//! GENERATED — do not hand-edit. Regenerate with Much-Better-S2H/_onnx_derisk/gen_rust_g2p_tables.py.
//! stage2 (traditional phoneme -> IPA) tables + dict_fixes rules for zh/en/de/fr/es/it, mirrored
//! bit-for-bit from the model repo's phoneme_vocab.py + dict_fixes.py (the single source — IPA is
//! NEVER hand-typed); + JA kana-coverage additions validated against the existing R2IPA convention.
//! The golden test (g2p_golden_ref.rs) proves the Rust port over the shipped dictionaries.

/// zh opencpop initials -> IPA (lookup BEFORE finals, mirroring convert_opencpop).
pub const OPENCPOP_INITIALS_IPA: &[(&str, &str)] = &[
    ("b", "p"),
    ("p", "pʰ"),
    ("m", "m"),
    ("f", "f"),
    ("d", "t"),
    ("t", "tʰ"),
    ("n", "n"),
    ("l", "l"),
    ("g", "k"),
    ("k", "kʰ"),
    ("h", "x"),
    ("j", "tɕ"),
    ("q", "tɕʰ"),
    ("x", "ɕ"),
    ("zh", "ʈʂ"),
    ("ch", "ʈʂʰ"),
    ("sh", "ʂ"),
    ("r", "ɻ"),
    ("z", "ts"),
    ("c", "tsʰ"),
    ("s", "s"),
    ("w", "w"),
    ("y", "j"),
];

/// zh opencpop finals -> IPA.
pub const OPENCPOP_FINALS_IPA: &[(&str, &str)] = &[
    ("a", "a"),
    ("o", "o"),
    ("e", "ɤ"),
    ("i", "i"),
    ("u", "u"),
    ("v", "y"),
    ("ai", "aɪ"),
    ("ei", "eɪ"),
    ("ao", "aʊ"),
    ("ou", "oʊ"),
    ("an", "an"),
    ("en", "ən"),
    ("ang", "ɑŋ"),
    ("eng", "əŋ"),
    ("ong", "ʊŋ"),
    ("ia", "ia"),
    ("ie", "iɛ"),
    ("iao", "iaʊ"),
    ("iou", "ioʊ"),
    ("ian", "iɛn"),
    ("in", "in"),
    ("iang", "iɑŋ"),
    ("ing", "iŋ"),
    ("iong", "iʊŋ"),
    ("ua", "ua"),
    ("uo", "uo"),
    ("uai", "uaɪ"),
    ("uei", "ueɪ"),
    ("uan", "uan"),
    ("uen", "uən"),
    ("uang", "uɑŋ"),
    ("van", "yɛn"),
    ("ve", "yɛ"),
    ("vn", "yn"),
    ("er", "əɻ"),
    ("iu", "ioʊ"),
    ("un", "uən"),
    ("ui", "ueɪ"),
];

/// en ARPABET (stress-stripped) -> IPA. AH is resolved BEFORE stripping (AH0=ə else ʌ) — in code.
pub const ARPABET_IPA: &[(&str, &str)] = &[
    ("AA", "ɑ"),
    ("AE", "æ"),
    ("AH", "ə"),
    ("AO", "ɔ"),
    ("AX", "ə"),
    ("AW", "aʊ"),
    ("AY", "aɪ"),
    ("B", "b"),
    ("CH", "tʃ"),
    ("D", "d"),
    ("DH", "ð"),
    ("EH", "ɛ"),
    ("ER", "ɝ"),
    ("EY", "eɪ"),
    ("F", "f"),
    ("G", "ɡ"),
    ("HH", "h"),
    ("IH", "ɪ"),
    ("IY", "i"),
    ("JH", "dʒ"),
    ("K", "k"),
    ("L", "l"),
    ("M", "m"),
    ("N", "n"),
    ("NG", "ŋ"),
    ("OW", "oʊ"),
    ("OY", "ɔɪ"),
    ("P", "p"),
    ("R", "ɹ"),
    ("S", "s"),
    ("SH", "ʃ"),
    ("T", "t"),
    ("TH", "θ"),
    ("UH", "ʊ"),
    ("UW", "u"),
    ("V", "v"),
    ("W", "w"),
    ("Y", "j"),
    ("Z", "z"),
    ("ZH", "ʒ"),
];

/// MFA phone normalization (tie-bars, DE diphthongs, ASCII g, tʂː) — applied after NFC.
pub const MFA_NORMALIZE: &[(&str, &str)] = &[
    ("t͡ʃ", "tʃ"),
    ("t͡ʃː", "tʃː"),
    ("d͡ʒ", "dʒ"),
    ("d͡ʒː", "dʒː"),
    ("t͡s", "ts"),
    ("aj", "aɪ"),
    ("aw", "aʊ"),
    ("g", "ɡ"),
    ("tʂː", "ʂː"),
];

/// dict_fixes A1 (zh apical-i by onset), C2 (non-ja palatal-stop de-narrow), C3 (dead tokens).
pub const FIX_A1_DENTAL: &[&str] = &["s", "ts", "tsʰ"];
pub const FIX_A1_RETRO: &[&str] = &["ɻ", "ʂ", "ʈʂ", "ʈʂʰ"];
pub const FIX_C2: &[(&str, &str)] = &[
    ("c", "k"),
    ("ɟ", "ɡ"),
    ("cʰ", "kʰ"),
];
pub const FIX_C3_GLOBAL: &[(&str, &str)] = &[
    ("l̩", "l"),
    ("bː", "b"),
    ("dː", "d"),
    ("t͈ʲ", "t͈"),
    ("c͈", "k͈"),
];

/// Per-language MFA vowel phones (traditional layer) — nucleus detection for
/// syllabification / sustain / coda-deferral. Derived from each shipped dictionary's inventory.
pub const MFA_VOWELS_DE: &[&str] = &["a", "aj", "aw", "aː", "eː", "iː", "oː", "uː", "yː", "øː", "œ", "ɐ", "ɔ", "ɔʏ", "ə", "ɛ", "ɪ", "ʊ", "ʏ"];
pub const MFA_VOWELS_FR: &[&str] = &["a", "e", "i", "o", "u", "y", "ø", "œ", "ɑ", "ɑ̃", "ɔ", "ɔ̃", "ə", "ɛ", "ɛ̃"];
pub const MFA_VOWELS_ES: &[&str] = &["a", "e", "i", "o", "u"];
pub const MFA_VOWELS_IT: &[&str] = &["a", "aː", "e", "eː", "i", "o", "oː", "u", "y", "æ", "ø", "œ", "ɔ", "ə", "ɛ"];

/// JA kana coverage additions (missing yōon rows + ゔ) — chained AFTER the base KANA table.
pub const KANA_EXTRA: &[(&str, &str)] = &[
    ("ぎゅ", "gyu"),
    ("ぎょ", "gyo"),
    ("ぢゃ", "ja"),
    ("ぢゅ", "ju"),
    ("ぢょ", "jo"),
    ("ひゅ", "hyu"),
    ("びゃ", "bya"),
    ("びゅ", "byu"),
    ("びょ", "byo"),
    ("ぴゃ", "pya"),
    ("ぴゅ", "pyu"),
    ("ぴょ", "pyo"),
    ("みゅ", "myu"),
    ("みょ", "myo"),
    ("ゔ", "vu"),
];

/// Romaji the extra kana need that the base R2IPA lacks (composed via the validated composer).
pub const R2IPA_EXTRA: &[(&str, &[&str])] = &[
    ("vu", &["v", "ɯ"]),
];
