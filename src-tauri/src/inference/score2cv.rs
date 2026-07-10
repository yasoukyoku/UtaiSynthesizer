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
use super::score2cv_tables as tbl;
use crate::{Result, UtaiError};

// ─── table accessors (built once from the generated const slices) ───────────────────────────────

fn phone_to_id_map() -> &'static HashMap<&'static str, i64> {
    static M: OnceLock<HashMap<&'static str, i64>> = OnceLock::new();
    M.get_or_init(|| tbl::PHONE_TO_ID.iter().copied().collect())
}
fn kana_map() -> &'static HashMap<&'static str, &'static str> {
    static M: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
    M.get_or_init(|| tbl::KANA.iter().copied().collect())
}
fn r2ipa_map() -> &'static HashMap<&'static str, &'static [&'static str]> {
    static M: OnceLock<HashMap<&'static str, &'static [&'static str]>> = OnceLock::new();
    M.get_or_init(|| tbl::R2IPA.iter().copied().collect())
}

// ─── G2P: one lyric token → IPA phones / rest / sustain ─────────────────────────────────────────

/// Outcome of `lyric_to_phones` — mirrors the Python `(phones, is_rest, is_sustain)` triple, with
/// `Unknown` for an OOV lyric (the DAW must LOUD-error, never the reference's silent SP fallback).
enum Lyric {
    Rest,
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
    if matches!(s0, "-" | "ー" | "+") {
        return Lyric::Sustain;
    }
    if matches!(s0, "っ" | "cl" | "q") {
        return Lyric::Phones(vec!["ʔ"]);
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

/// Public classification of ONE lyric token for the frontend (§9.5 single Rust classifier: the editor's
/// rest/sustain/OOV verdict MUST equal the render's — no JS dictionary copy that drifts from
/// `lyric_to_phones`). Serialized as `{kind:"rest"|"sustain"|"phones"|"unknown", phones?:[…]}`.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum LyricClass {
    /// A rest token (`R`/`r`/``/`rest`/`sil`/`pau`) — silence, no phones.
    Rest,
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
pub struct ScoreArrays {
    pub phonemes: Vec<i64>,
    pub phone_dur: Vec<i64>,
    pub note_pitch: Vec<i64>,
    pub note_dur: Vec<i64>,
    pub note_to_phone: Vec<i64>,
    pub phon: Vec<&'static str>,
}

/// Port of `render_ust.build_arrays` (+ its `lyric_to_phones` front-end). `score` = (lyric, note_num,
/// frames) per note. Rest frames are capped (first/last note → `CAP_LEAD`, mid → `CAP_MID`); a sustain
/// (`-`/`ー`/`+`) continues the previous vowel (default `a`); notes are grouped into `note_to_phone` by
/// consecutive equal pitch. An OOV lyric is a LOUD error (never the reference's silent SP fallback — the
/// v1 "啊啊啊" regression). This is the Phase-1c PARITY entry (rest-capped, == render_ust); the ② vocal
/// DAW render uses `build_arrays_daw` (rests uncapped) so the rendered stem aligns to the segment timeline.
pub fn build_arrays(score: &[(&str, i64, i64)]) -> Result<ScoreArrays> {
    build_arrays_impl(score, true)
}

/// ② 自己唱 DAW render entry: identical to `build_arrays` EXCEPT rest frames are NOT capped, so the cv
/// frame count == the DAW frame count (a rest holds its full duration) and the rendered stem stays
/// aligned to the segment's tick timeline. A deliberate fork from the Python parity (which caps rests for
/// a standalone song render, where absolute timing doesn't matter); the Phase-1c gate still tests the
/// capped `build_arrays`. ⚠ `chunk_at_sp` CANNOT subdivide a single rest (it only splits AT an SP after
/// the running count exceeds max_frames), so one very long rest becomes one big chunk — the TOTAL frame
/// count is bounded upstream by `render_vocal_segment`'s `MAX_TOTAL_FRAMES`, not here.
pub fn build_arrays_daw(score: &[(&str, i64, i64)]) -> Result<ScoreArrays> {
    build_arrays_impl(score, false)
}

fn build_arrays_impl(score: &[(&str, i64, i64)], cap_rests: bool) -> Result<ScoreArrays> {
    let m = score.len();
    let mut phon: Vec<&'static str> = Vec::new();
    let mut pdur: Vec<i64> = Vec::new();
    let mut npitch: Vec<i64> = Vec::new();
    let mut prev_vowel: Option<&'static str> = None;

    for (k, &(lyr, nn, fr)) in score.iter().enumerate() {
        match lyric_to_phones(lyr) {
            Lyric::Rest => {
                let cap = if !cap_rests {
                    i64::MAX // DAW render: keep the full rest so the stem aligns to the timeline
                } else if k == 0 || k == m - 1 {
                    tbl::CAP_LEAD
                } else {
                    tbl::CAP_MID
                };
                phon.push("SP");
                pdur.push(fr.min(cap).max(1));
                npitch.push(0);
                prev_vowel = None;
            }
            Lyric::Sustain => {
                phon.push(prev_vowel.unwrap_or("a"));
                pdur.push(fr.max(1));
                npitch.push(nn);
            }
            Lyric::Phones(ph) => {
                let durs = split_dur(fr, ph.len());
                for (&p, &d) in ph.iter().zip(durs.iter()) {
                    phon.push(p);
                    pdur.push(d);
                    npitch.push(nn);
                }
                if let Some(&last) = ph.last() {
                    if tbl::VOWEL_SET.contains(&last) {
                        prev_vowel = Some(last);
                    }
                }
            }
            Lyric::Unknown => {
                return Err(UtaiError::Inference(format!(
                    "歌词 “{}” 无法映射到音素（OOV）——请检查歌词或语言设置（绝不静默兜底为静音）",
                    lyr
                )));
            }
        }
    }

    // phone → id (LOUD error on any phone outside the 210-token vocab; the reference SP-falls-back).
    let map = phone_to_id_map();
    let mut phonemes = Vec::with_capacity(phon.len());
    for &p in &phon {
        let id = *map.get(p).ok_or_else(|| {
            UtaiError::Inference(format!("音素 “{}” 不在 ScoreToCV 词表（210 token）中", p))
        })?;
        phonemes.push(id);
    }

    // note grouping: consecutive equal pitch → one note group; note_dur = Σ phone_dur within a group.
    let mut note_to_phone = Vec::with_capacity(npitch.len());
    let mut nidx: i64 = -1;
    let mut prev: Option<i64> = None;
    for &p in &npitch {
        if prev != Some(p) {
            nidx += 1;
            prev = Some(p);
        }
        note_to_phone.push(nidx);
    }
    let group_count = (nidx + 1).max(0) as usize;
    let mut group_frames = vec![0i64; group_count];
    for (i, &g) in note_to_phone.iter().enumerate() {
        group_frames[g as usize] += pdur[i];
    }
    let note_dur: Vec<i64> = note_to_phone.iter().map(|&g| group_frames[g as usize]).collect();

    Ok(ScoreArrays { phonemes, phone_dur: pdur, note_pitch: npitch, note_dur, note_to_phone, phon })
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
}

fn make_chunk(a: &ScoreArrays, s: usize, e: usize) -> Chunk {
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
    }
}

/// Cut the score into chunks at SP (rest) boundaries once the running frame count exceeds `max_frames`
/// (deploy default 400), bounding SVC memory + O(N²). Verbatim from `render_ust.render_song`: split on
/// the phone STRING "SP" (never the id), the SP is included in the closing chunk, and each chunk's
/// `note_to_phone` is rebased to start at 0.
pub fn chunk_at_sp(a: &ScoreArrays, max_frames: i64) -> Vec<Chunk> {
    let n = a.phonemes.len();
    let mut chunks = Vec::new();
    let (mut start, mut cf) = (0usize, 0i64);
    for i in 0..n {
        cf += a.phone_dur[i];
        if cf > max_frames && a.phon[i] == "SP" {
            chunks.push(make_chunk(a, start, i + 1));
            start = i + 1;
            cf = 0;
        }
    }
    if start < n {
        chunks.push(make_chunk(a, start, n));
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
        .ok_or_else(|| UtaiError::Inference("ScoreToCV 模型没有返回输出".into()))?;
    let t = chunk.t;
    if cv.len() != t * dim {
        return Err(UtaiError::Inference(format!(
            "ScoreToCV 输出尺寸异常：期望 {}x{}={}，得到 {}",
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

    // ② vocal DAW render (S53): `build_arrays_daw` keeps the FULL rest so the stem aligns to the
    // timeline; `build_arrays` (parity) caps it (CAP_MID=70 mid / CAP_LEAD=25 first/last).
    #[test]
    fn build_arrays_daw_uncaps_rests() {
        let score = [("か", 60, 80), ("R", 0, 300), ("お", 67, 80)];
        let capped = build_arrays(&score).unwrap();
        let daw = build_arrays_daw(&score).unwrap();
        // find the SP phone's frame count in each.
        let sp_capped = capped.phon.iter().zip(&capped.phone_dur).find(|(p, _)| **p == "SP").map(|(_, d)| *d);
        let sp_daw = daw.phon.iter().zip(&daw.phone_dur).find(|(p, _)| **p == "SP").map(|(_, d)| *d);
        assert_eq!(sp_capped, Some(tbl::CAP_MID), "parity build caps a mid rest to CAP_MID");
        assert_eq!(sp_daw, Some(300), "DAW build keeps the full 300-frame rest");
    }

    // §9.5 single Rust classifier: `classify_lyric` (exposed via the validate_lyrics command) MUST agree
    // with `lyric_to_phones` (which build_arrays uses) so the editor's verdict == the render's.
    #[test]
    fn classify_lyric_matches_render() {
        assert!(matches!(classify_lyric("R"), LyricClass::Rest));
        assert!(matches!(classify_lyric(""), LyricClass::Rest));
        assert!(matches!(classify_lyric("ー"), LyricClass::Sustain));
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
        // no duplicate keys collapsed the maps
        assert_eq!(phone_to_id_map().len(), tbl::PHONE_TO_ID.len());
        assert_eq!(kana_map().len(), tbl::KANA.len());
        assert_eq!(r2ipa_map().len(), tbl::R2IPA.len());
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
            let path: PathBuf = root.join("../data/models/aux").join(model);
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
