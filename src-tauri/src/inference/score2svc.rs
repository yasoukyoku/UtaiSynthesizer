//! ScoreToCV → SVC decode glue — "自己唱" (score → singing) render tail (S48 Phase 2).
//!
//! Phase 1 produced the content features (`score2cv.rs`: score → cv[T,dim] @ 50 fps, bit-exact vs
//! Python-ORT). Phase 2 is the deterministic glue that turns a score into audible singing by swapping
//! TWO producers into the existing SVC decode tail and reusing everything else:
//!   * cv  ← `run_score2cv` (replaces the audio ContentVec extractor `features::contentvec_extract`)
//!   * f0  ← a DAW-parameterized stream (replaces RMVPE `f0::rmvpe_detect`)
//!
//! The f0 here is the BARE "noteonly" step — `note_hz[t] = 440·2^((midi[t]−69)/12)` on voiced frames,
//! 0 at rests — reproduced bit-for-bit from the Python reference `render_ust.render_song`'s noteonly
//! path (§Ground truth). It has NO portamento/vibrato: those are pitch-EXPRESSION and land in Phase 5
//! (音高编辑) with the pitch-editing UI, per the design's §6 phase plan. `midi[t]` is the length-
//! regulated `note_pitch` (each phone's MIDI repeated `phone_dur[i]` frames) — a pure function of the
//! score, identical to `ScoreToF0.encode`'s `midi_frame`, so NO f0 model is needed.
//!
//! Then cv/f0/uv are resampled 50 fps → the SVC grid (SoVITS 44100/512 ≈ 86 fps `nearest`; RVC 2×
//! nearest → 100 fps) and fed to the exported net_g (SoVITS `c/f0/uv/noise/sid[,vol]`; RVC
//! `phone/phone_lengths/pitch/pitchf/sid/rnd`). uv = (f0 < 30) is the only voiced switch.
//!
//! GROUND TRUTH / GATE: the net_g wav is NOT bit-exact vs Python (ONNX-vs-PyTorch + the S35 export moved
//! net_g's randn to an explicit seeded `noise` input), so the parity gate is on the deterministic net_g
//! INPUT tensors (`score2svc_ref.rs`, dumped by `_onnx_derisk/dump_score2svc_ref.py`): midi_frame /
//! note_hz @50fps + f0_rs / uv_rs / cv_rs @86fps, all reproduced bit-for-bit. Audible wav + the §3.4
//! legato-vs-SP behavior are confirmed by ear (Tier-2 render tests).

use ndarray::Array2;

use super::engine::{InputTensor, OnnxEngine};
use super::features::{repeat_expand_2d, torch_interp_nearest, upsample_2x_nearest};
use super::score2cv::{
    build_arrays_daw, chunk_at_sp, classify_lyric, run_score2cv, LyricClass, ScoreArrays,
};
use super::SynthesisResult;
use crate::{Result, UtaiError};

/// ScoreToCV frame rate (score2cv sidecar `fps`). The per-phone `phone_dur` frames are 20 ms.
const CV_FPS: f64 = 50.0;
/// so-vits-svc 4.x default output geometry (== the Python reference `synth_sovits` CV_FPS/SOVITS_HOP).
/// A future non-44100/512 SoVITS export would carry these in its sidecar; 4.x is always this.
pub const SOVITS_SR: u32 = 44100;
pub const SOVITS_HOP: usize = 512;

// ─── bare noteonly f0 (model-free; the DAW's activity, decoupled from cv) ─────────────────────────

/// Length-regulate `note_pitch` to the frame grid: each phone's MIDI repeated `phone_dur[i]` frames.
/// This is exactly `ScoreToF0.encode`'s `midi_frame` (`note_pitch.gather(1, phone_idx)`), where
/// `phone_idx = searchsorted(cumsum(phone_dur), arange(T), right=True)` — a repeat is bit-identical to
/// that searchsorted expansion. Length = T = Σ phone_dur (matches the cv's T exactly).
pub fn midi_frame_50(note_pitch: &[i64], phone_dur: &[i64]) -> Vec<i64> {
    let mut out = Vec::with_capacity(phone_dur.iter().map(|&d| d.max(0) as usize).sum());
    for (&p, &d) in note_pitch.iter().zip(phone_dur.iter()) {
        for _ in 0..d.max(0) {
            out.push(p);
        }
    }
    out
}

/// Bare noteonly f0 @ 50 fps from the per-frame MIDI: `440·2^((midi−69)/12)` where `midi > 0`, else 0.
/// Computed in f64 then cast to f32 (matches `render_ust.render_song:177`'s float64→float32).
pub fn note_hz_50(midi_frame: &[i64]) -> Vec<f32> {
    midi_frame
        .iter()
        .map(|&m| {
            if m > 0 {
                (440.0_f64 * 2.0_f64.powf((m as f64 - 69.0) / 12.0)) as f32
            } else {
                0.0
            }
        })
        .collect()
}

// ─── Option-A DAW f0 (§10.1): a TS-computed layered pitch curve fed per-frame ─────────────────────

/// The DAW's WHOLE-segment pitch curve @50fps (Option A, §10.1): the SINGLE `evalF0Cents` output that
/// also drives the overlay + preview → what-you-see == what-you-hear == what-renders. `cents` is
/// WRITTEN-pitch cents (MIDI·100; transpose is applied HERE in Rust, never on the TS side); `voiced`
/// is a 1/0 mask (a 0¢ cents value is a VALID pitch, so voicing MUST come from this mask, never from the
/// Hz magnitude — that was the Phase-2 shortcut `note_hz<30` and it breaks under a layered f0). Both are
/// segment-relative, index = the DAW 50fps frame from segment start.
pub struct VocalF0<'a> {
    pub cents: &'a [f32],
    pub voiced: &'a [u8],
}

/// cents (MIDI·100, A4=6900) → Hz: `440·2^((cents−6900)/1200)`.
fn cents_to_hz(cents: f64) -> f32 {
    (440.0_f64 * 2.0_f64.powf((cents - 6900.0) / 1200.0)) as f32
}

/// Add `transpose` semitones to the per-phone content `note_pitch` (>0 frames only — rests stay 0),
/// clamped to a valid MIDI note. Note grouping (note_to_phone / note_dur) is UNCHANGED by an equal shift,
/// so it stays valid without recompute. §9.3: transpose is applied ONLY in the render, never on-canvas.
fn transpose_note_pitch(note_pitch: &mut [i64], transpose: i64) {
    if transpose == 0 {
        return;
    }
    for p in note_pitch.iter_mut() {
        if *p > 0 {
            *p = (*p + transpose).clamp(1, 127);
        }
    }
}

/// Build the per-frame f0 (Hz) @50fps for the WHOLE score (length = Σ arr.phone_dur = T50 total).
/// `arr.note_pitch` is RAW here (transpose is applied to the OUTPUT Hz, and to the content pitch
/// separately by the caller). Two modes:
///  * `f0 = None` → bare noteonly (Phase-2): `note_hz_50(midi_frame)` with `transpose` folded into the
///    MIDI. transpose=0 ⇒ byte-identical to Phase 2 (the parity anchor).
///  * `f0 = Some` → Option A: each cv frame maps to a DAW 50fps index through its note GROUP's cv↔DAW
///    frame ranges (which differ when a short note inflated in `split_dur`, or — with capped rests — a
///    rest compressed), samples `f0.cents` there (+ transpose·100¢) → Hz. Voiced iff the group has
///    pitch>0 AND `f0.voiced[idx]`; unvoiced → 0 Hz. The GROUPS come from `arr.note_to_phone`; the DAW
///    frame spans come from the score triples' native `frames` (grouped identically by note_num changes).
pub fn build_note_hz(
    arr: &ScoreArrays,
    score: &[(&str, i64, i64)],
    transpose: i64,
    f0: Option<&VocalF0>,
) -> Vec<f32> {
    let t_total: usize = arr.phone_dur.iter().map(|&d| d.max(0) as usize).sum();
    let f0 = match f0 {
        None => {
            // bare noteonly: per-frame MIDI (transposed, clamped) → the SAME note_hz_50 Hz formula.
            // transpose=0 ⇒ midi unchanged (valid notes are 1..127) ⇒ byte-identical to Phase 2.
            let midi: Vec<i64> = midi_frame_50(&arr.note_pitch, &arr.phone_dur)
                .into_iter()
                .map(|m| if m > 0 { (m + transpose).clamp(1, 127) } else { 0 })
                .collect();
            return note_hz_50(&midi);
        }
        Some(f0) => f0,
    };

    let ng = arr.note_to_phone.last().map(|&g| g as usize + 1).unwrap_or(0);
    if ng == 0 || f0.cents.is_empty() {
        return vec![0.0; t_total];
    }

    // Per-group cv frame range (from arr): the group's start cv frame + total cv frames.
    let mut cv_start = vec![0usize; ng];
    let mut cv_count = vec![0usize; ng];
    {
        let mut cursor = 0usize;
        let mut seen = vec![false; ng];
        for (i, &g) in arr.note_to_phone.iter().enumerate() {
            let g = g as usize;
            if !seen[g] {
                cv_start[g] = cursor;
                seen[g] = true;
            }
            let d = arr.phone_dur[i].max(0) as usize;
            cv_count[g] += d;
            cursor += d;
        }
    }

    // Per-group DAW frame range (from the score triples). Grouped by the SAME `npitch` rule build_arrays
    // uses — rest → 0, else → note_num (classified via the single `classify_lyric`, so a rest triple with
    // a stray non-zero note_num still groups as a rest, matching the cv side; §9.5 editor==render).
    let mut daw_start = vec![0usize; ng];
    let mut daw_count = vec![0usize; ng];
    let mut group_pitch = vec![0i64; ng];
    {
        let mut g: i64 = -1;
        let mut prev: Option<i64> = None;
        let mut cursor = 0usize;
        let mut seen = vec![false; ng];
        for &(lyr, nn, fr) in score {
            let npitch = if matches!(classify_lyric(lyr), LyricClass::Rest) { 0 } else { nn };
            if prev != Some(npitch) {
                g += 1;
                prev = Some(npitch);
            }
            let gi = (g as usize).min(ng - 1);
            if !seen[gi] {
                daw_start[gi] = cursor;
                group_pitch[gi] = npitch;
                seen[gi] = true;
            }
            let d = fr.max(0) as usize;
            daw_count[gi] += d;
            cursor += d;
        }
    }

    let flen = f0.cents.len();
    let mut out = vec![0.0f32; t_total];
    for g in 0..ng {
        if group_pitch[g] <= 0 || cv_count[g] == 0 {
            continue; // rest group → 0 Hz (unvoiced); nothing to sample
        }
        for k in 0..cv_count[g] {
            let cv_i = cv_start[g] + k;
            if cv_i >= t_total {
                break;
            }
            // map this cv frame (center) to a DAW 50fps index within the group's DAW span.
            let frac = (k as f64 + 0.5) / cv_count[g] as f64;
            let daw_f = daw_start[g] as f64 + frac * daw_count[g] as f64;
            let idx = (daw_f.floor() as usize).min(flen - 1);
            if f0.voiced.get(idx).copied().unwrap_or(0) != 0 {
                out[cv_i] = cents_to_hz(f0.cents[idx] as f64 + (transpose as f64) * 100.0);
            }
        }
    }
    out
}

// ─── resample 50 fps → SVC grid ──────────────────────────────────────────────────────────────────

/// `int(round(x))` with banker's rounding (round-half-to-even) — matches Python's built-in `round`
/// used by `synth_sovits.resample_2d`'s `int(round(...))`. Half-cases are measure-zero for real
/// scores, but matching exactly keeps the grid length bit-identical to the reference.
fn round_half_even(x: f64) -> usize {
    let f = x.floor();
    let diff = x - f;
    let r = if (diff - 0.5).abs() < 1e-9 {
        // exactly .5 → round to the even integer
        if (f as i64) % 2 == 0 {
            f
        } else {
            f + 1.0
        }
    } else {
        x.round()
    };
    r.max(0.0) as usize
}

/// Target frame count for the SVC grid: `round(T50 · (sr/hop) / 50)` (== the Python reference's
/// `resample_*d` `T_tgt`).
pub fn sovits_grid_len(t50: usize, sr: u32, hop: usize) -> usize {
    round_half_even(t50 as f64 * (sr as f64 / hop as f64) / CV_FPS)
}

/// The SVC net_g input feed for one chunk on the SoVITS hop grid.
pub struct SovitsFeed {
    /// cv resampled to the hop grid, `[t_tgt, dim]`.
    pub cv: Array2<f32>,
    /// f0 (Hz) per hop frame, clipped to `[0, 1100]`, length `t_tgt`.
    pub f0: Vec<f32>,
    /// voiced mask (0.0/1.0) per hop frame, length `t_tgt`.
    pub uv: Vec<f32>,
    pub t_tgt: usize,
}

/// Resample a chunk's `(cv @50fps, note_hz @50fps)` to the SoVITS hop grid, mirroring
/// `render_derisk.render_cv` EXACTLY: uv=(note_hz<30) is taken on the 50 fps f0 FIRST, then cv/f0/uv
/// are `F.interpolate('nearest')`-resampled to `t_tgt = round(T·sr/hop/50)`, f0 clipped to [0,1100],
/// uv re-thresholded `>0.5`. cv nearest == `repeat_expand_2d(_, _, "nearest")` (torch-parity tested).
pub fn resample_to_sovits_grid(cv: &Array2<f32>, note_hz: &[f32], sr: u32, hop: usize) -> Result<SovitsFeed> {
    debug_assert_eq!(cv.nrows(), note_hz.len(), "cv rows must equal note_hz length (both T50)");
    let t_tgt = sovits_grid_len(cv.nrows(), sr, hop);
    if t_tgt == 0 {
        return Err(UtaiError::Inference("score2svc: 目标帧数为 0（谱太短）".into()));
    }
    // uv on the 50 fps f0 (render_cv order), then nearest-resample the float mask. Under Option-A the real
    // voiced mask already lives in `build_note_hz` (which writes 0 Hz for every unvoiced frame), so deriving
    // uv=(note_hz<30) here round-trips it EXACTLY via the 0-Hz sentinel — no real sung pitch is <30 Hz
    // (MIDI 24 ≈ 32.7 Hz). ⚠ An extreme downward transpose past MIDI 24 would sit below 30 Hz and be marked
    // unvoiced; threading the mask through instead of re-deriving is the clean fix if that ever bites.
    let uv50: Vec<f32> = note_hz.iter().map(|&f| if f < 30.0 { 1.0 } else { 0.0 }).collect();
    let cv_rs = repeat_expand_2d(cv, t_tgt, "nearest")?;
    let f0_rs: Vec<f32> = torch_interp_nearest(note_hz, t_tgt)
        .into_iter()
        .map(|f| f.clamp(0.0, 1100.0))
        .collect();
    let uv_rs: Vec<f32> = torch_interp_nearest(&uv50, t_tgt)
        .into_iter()
        .map(|v| if v > 0.5 { 1.0 } else { 0.0 })
        .collect();
    Ok(SovitsFeed { cv: cv_rs, f0: f0_rs, uv: uv_rs, t_tgt })
}

// ─── SoVITS net_g decode (mirrors sovits::infer_segment's VITS input build, score inputs only) ───

/// The facts a Phase-2 SoVITS decode needs from the voice sidecar (a subset of `SovitsModel`; the score
/// path skips ContentVec/RMVPE/cluster/auto-f0 entirely).
pub struct SovitsVoice {
    pub features_dim: usize,
    /// inter_channels of the explicit `noise` input (192 for 4.0/4.1).
    pub noise_channels: usize,
    /// Whether the exported graph HAS a `vol` input (sidecar vol_embedding).
    pub vol_embedding: bool,
    pub sample_rate: u32,
    pub hop_size: usize,
    /// Minimum frame count the exported net_g accepts (sidecar `min_frames`, 6 for SoVITS). A chunk
    /// shorter than this cannot be decoded — the cover path guards it (sovits.rs:255); the score path
    /// LOUD-errors (a too-short note) rather than feeding net_g a shape it rejects.
    pub min_frames: usize,
}

/// Decode ONE chunk's feed through the SoVITS net_g → wav. Feeds `c/f0/uv/noise/sid[,vol]` (the same
/// contract `sovits::infer_segment` uses at 426-481); `noise` is the shared `sovits::seg_noise` draw so
/// it is byte-identical to the cover path. `vol` is required iff `voice.vol_embedding` (the graph has
/// the input); Phase 2 has no loudness UI yet, so the caller passes a flat placeholder (real vol lane =
/// Phase 5). Non-deterministic across ONNX-vs-PyTorch/seeded-noise — validated by ear, not bit-parity.
#[allow(clippy::too_many_arguments)]
pub fn sovits_decode_chunk(
    engine: &OnnxEngine,
    voice_session: &str,
    feed: &SovitsFeed,
    sid: i64,
    voice: &SovitsVoice,
    vol: Option<&[f32]>,
    seed: u64,
    seg_idx: u64,
    noise_scale: f32,
) -> Result<Vec<f32>> {
    let t = feed.t_tgt as i64;
    if feed.cv.nrows() != feed.t_tgt || feed.f0.len() != feed.t_tgt || feed.uv.len() != feed.t_tgt {
        return Err(UtaiError::Inference("score2svc: cv/f0/uv 帧数不一致".into()));
    }
    // min_frames guard (net_g rejects a sub-min chunk; the cover path guards this at sovits.rs:255).
    if feed.t_tgt < voice.min_frames {
        return Err(UtaiError::Inference(format!(
            "score2svc: 片段过短，帧数 {} 小于模型最小帧数 {}（音符太短，请延长）",
            feed.t_tgt, voice.min_frames
        )));
    }
    let noise = super::sovits::seg_noise(voice.noise_channels, feed.t_tgt, seed, seg_idx, noise_scale);
    let mut inputs = vec![
        ("c", InputTensor::F32 { data: feed.cv.iter().copied().collect(), shape: vec![1, t, voice.features_dim as i64] }),
        ("f0", InputTensor::F32 { data: feed.f0.clone(), shape: vec![1, t] }),
        ("uv", InputTensor::F32 { data: feed.uv.clone(), shape: vec![1, t] }),
        ("noise", InputTensor::F32 { data: noise, shape: vec![1, voice.noise_channels as i64, t] }),
        ("sid", InputTensor::I64 { data: vec![sid], shape: vec![1] }),
    ];
    if voice.vol_embedding {
        let v = vol.ok_or_else(|| {
            UtaiError::Inference("该 SoVITS 模型需要 vol 输入（vol_embedding=true），但未提供".into())
        })?;
        if v.len() != feed.t_tgt {
            return Err(UtaiError::Inference(format!("vol 帧数异常：{} != {}", v.len(), feed.t_tgt)));
        }
        inputs.push(("vol", InputTensor::F32 { data: v.to_vec(), shape: vec![1, t] }));
    }
    let outputs = engine.run(voice_session, inputs)?;
    outputs
        .into_iter()
        .next()
        .ok_or_else(|| UtaiError::Inference("SoVITS 模型没有返回输出".into()))
}

/// Full score → SoVITS wav (自己唱). build_arrays_daw (rests uncapped → stem aligns to the timeline) →
/// SP-chunk (≤400) → per chunk: run_score2cv → cv; f0 from `build_note_hz` (bare noteonly when `f0` is
/// None, else the DAW's layered Option-A pitch); resample to the hop grid; net_g decode. Chunk wavs are
/// concatenated then peak-normalized to 0.92. `transpose` (semitones) shifts both the content note_pitch
/// and the f0 (§9.3, Rust-only). `flat_vol` seeds the placeholder vol tensor for vol_embedding models.
/// `cancel`/`progress` are polled per chunk (a multi-chunk render aborts at the next boundary).
#[allow(clippy::too_many_arguments)]
pub fn render_score_sovits(
    engine: &OnnxEngine,
    score2cv_session: &str,
    voice_session: &str,
    score: &[(&str, i64, i64)],
    dim: usize,
    cv_speaker_id: i64,
    lang_id: i64,
    voice: &SovitsVoice,
    sid: i64,
    flat_vol: f32,
    seed: u64,
    noise_scale: f32,
    transpose: i64,
    f0: Option<&VocalF0>,
    cancel: &dyn Fn() -> bool,
    progress: &dyn Fn(f32),
) -> Result<SynthesisResult> {
    let mut arr = build_arrays_daw(score)?;
    // f0 uses RAW note_pitch (grouping/voicing); transpose folds into the OUTPUT Hz.
    let note_hz_full = build_note_hz(&arr, score, transpose, f0);
    // content note_pitch is transposed separately (grouping is shift-invariant).
    transpose_note_pitch(&mut arr.note_pitch, transpose);
    let chunks = chunk_at_sp(&arr, 400);
    let n_chunks = chunks.len().max(1);
    let mut audio: Vec<f32> = Vec::new();
    let mut cv_cursor = 0usize;
    for (ci, chunk) in chunks.iter().enumerate() {
        if cancel() {
            return Err(UtaiError::Inference("已取消".into()));
        }
        let cv = run_score2cv(engine, score2cv_session, chunk, dim, cv_speaker_id, lang_id)?;
        let note_hz = &note_hz_full[cv_cursor..(cv_cursor + chunk.t).min(note_hz_full.len())];
        let feed = resample_to_sovits_grid(&cv, note_hz, voice.sample_rate, voice.hop_size)?;
        let vol = if voice.vol_embedding { Some(vec![flat_vol; feed.t_tgt]) } else { None };
        let wav = sovits_decode_chunk(
            engine, voice_session, &feed, sid, voice, vol.as_deref(), seed, ci as u64, noise_scale,
        )?;
        audio.extend_from_slice(&wav);
        cv_cursor += chunk.t;
        progress((ci + 1) as f32 / n_chunks as f32);
    }
    peak_normalize(&mut audio, 0.92);
    Ok(SynthesisResult { audio, sample_rate: voice.sample_rate })
}

// ─── RVC net_g decode (mirrors rvc::vc_chunk's input build, score inputs only) ───────────────────

/// The facts a Phase-2 RVC decode needs from the voice sidecar (subset of `RvcModel`).
pub struct RvcVoice {
    pub features_dim: usize,
    /// inter_channels of the explicit `rnd` input (192 for v1/v2).
    pub noise_channels: usize,
    pub sample_rate: u32,
    /// Minimum frame count the exported net_g accepts (sidecar `min_frames`, 12 for RVC). The cover path
    /// hard-errors below it (rvc.rs:291); the score path does the same (a too-short note).
    pub min_frames: usize,
}

/// The RVC net_g feed on the 100 fps grid (cv 50→100 via 2× nearest; f0 likewise).
pub struct RvcFeed {
    pub phone: Array2<f32>, // [t, dim] @100fps
    pub pitchf: Vec<f32>,   // Hz per 100fps frame
    pub pitch: Vec<i64>,    // coarse bin per frame
    pub t: usize,
}

/// Resample a chunk's `(cv @50fps, note_hz @50fps)` to the RVC 100 fps grid: cv → `upsample_2x_nearest`;
/// f0 → repeat each frame twice (the SAME 2× nearest, keeping cv/f0 on one grid); pitch = coarse bins.
/// RVC has NO uv input — unvoiced is implicit (f0==0 → coarse bin 1). No Python A/B reference exists for
/// RVC; the glue is validated by the (torch-tested) `upsample_2x_nearest`/`f0_to_coarse` + audible wav.
pub fn resample_to_rvc_grid(cv: &Array2<f32>, note_hz: &[f32]) -> RvcFeed {
    let phone = upsample_2x_nearest(cv); // [2·T50, dim]
    let mut pitchf: Vec<f32> = Vec::with_capacity(note_hz.len() * 2);
    for &f in note_hz {
        pitchf.push(f);
        pitchf.push(f);
    }
    let t = phone.nrows().min(pitchf.len());
    pitchf.truncate(t);
    let pitch: Vec<i64> = pitchf.iter().map(|&f| super::rvc::f0_to_coarse(f)).collect();
    RvcFeed { phone, pitchf, pitch, t }
}

/// Decode ONE chunk's RVC feed → wav. Feeds `phone/phone_lengths/pitch/pitchf/sid/rnd` (the contract
/// `rvc::vc_chunk` uses at 325-380); `rnd` is the shared `rvc::chunk_noise` draw. Skips the audio-only
/// retrieval/protect (meaningless for generated content).
#[allow(clippy::too_many_arguments)]
pub fn rvc_decode_chunk(
    engine: &OnnxEngine,
    voice_session: &str,
    feed: &RvcFeed,
    sid: i64,
    voice: &RvcVoice,
    seed: u64,
    chunk_idx: u64,
    noise_scale: f32,
) -> Result<Vec<f32>> {
    let t = feed.t as i64;
    if feed.t < voice.min_frames {
        return Err(UtaiError::Inference(format!(
            "score2svc: 片段过短，帧数 {} 小于模型最小帧数 {}（音符太短，请延长）",
            feed.t, voice.min_frames
        )));
    }
    let rnd = super::rvc::chunk_noise(voice.noise_channels, feed.t, seed, chunk_idx, noise_scale);
    let phone_flat: Vec<f32> = feed.phone.slice(ndarray::s![..feed.t, ..]).iter().copied().collect();
    let inputs = vec![
        ("phone", InputTensor::F32 { data: phone_flat, shape: vec![1, t, voice.features_dim as i64] }),
        ("phone_lengths", InputTensor::I64 { data: vec![t], shape: vec![1] }),
        ("pitch", InputTensor::I64 { data: feed.pitch.clone(), shape: vec![1, t] }),
        ("pitchf", InputTensor::F32 { data: feed.pitchf.clone(), shape: vec![1, t] }),
        ("sid", InputTensor::I64 { data: vec![sid], shape: vec![1] }),
        ("rnd", InputTensor::F32 { data: rnd, shape: vec![1, voice.noise_channels as i64, t] }),
    ];
    let outputs = engine.run(voice_session, inputs)?;
    outputs
        .into_iter()
        .next()
        .ok_or_else(|| UtaiError::Inference("RVC 模型没有返回输出".into()))
}

/// Full score → RVC wav (自己唱). Same shape as `render_score_sovits` but on the 100 fps grid and with
/// no uv/vol. RVC v2 uses cv768 (dim=768), same as SoVITS 4.1.
#[allow(clippy::too_many_arguments)]
pub fn render_score_rvc(
    engine: &OnnxEngine,
    score2cv_session: &str,
    voice_session: &str,
    score: &[(&str, i64, i64)],
    dim: usize,
    cv_speaker_id: i64,
    lang_id: i64,
    voice: &RvcVoice,
    sid: i64,
    seed: u64,
    noise_scale: f32,
    transpose: i64,
    f0: Option<&VocalF0>,
    cancel: &dyn Fn() -> bool,
    progress: &dyn Fn(f32),
) -> Result<SynthesisResult> {
    let mut arr = build_arrays_daw(score)?;
    let note_hz_full = build_note_hz(&arr, score, transpose, f0);
    transpose_note_pitch(&mut arr.note_pitch, transpose);
    let chunks = chunk_at_sp(&arr, 400);
    let n_chunks = chunks.len().max(1);
    let mut audio: Vec<f32> = Vec::new();
    let mut cv_cursor = 0usize;
    for (ci, chunk) in chunks.iter().enumerate() {
        if cancel() {
            return Err(UtaiError::Inference("已取消".into()));
        }
        let cv = run_score2cv(engine, score2cv_session, chunk, dim, cv_speaker_id, lang_id)?;
        let note_hz = &note_hz_full[cv_cursor..(cv_cursor + chunk.t).min(note_hz_full.len())];
        let feed = resample_to_rvc_grid(&cv, note_hz);
        let wav = rvc_decode_chunk(engine, voice_session, &feed, sid, voice, seed, ci as u64, noise_scale)?;
        audio.extend_from_slice(&wav);
        cv_cursor += chunk.t;
        progress((ci + 1) as f32 / n_chunks as f32);
    }
    peak_normalize(&mut audio, 0.92);
    Ok(SynthesisResult { audio, sample_rate: voice.sample_rate })
}

/// `w *= peak / (max|w| + 1e-9)` — render_ust.render_song's final output normalization.
fn peak_normalize(w: &mut [f32], peak: f32) {
    let m = w.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let g = peak / (m + 1e-9);
    for v in w.iter_mut() {
        *v *= g;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::score2cv::build_arrays; // Phase-1c parity entry (rest-capped)
    use super::super::score2cv_tables::parity_ref as pr;

    #[test]
    fn midi_frame_is_length_regulation() {
        // 3 phones, durs [2,1,3] → repeat each note_pitch by its dur.
        let midi = midi_frame_50(&[60, 0, 62], &[2, 1, 3]);
        assert_eq!(midi, vec![60, 60, 0, 62, 62, 62]);
    }

    #[test]
    fn note_hz_a4_and_rest() {
        let hz = note_hz_50(&[69, 0, 60]);
        assert!((hz[0] - 440.0).abs() < 1e-3, "A4 = 440 Hz, got {}", hz[0]);
        assert_eq!(hz[1], 0.0, "rest (midi 0) → 0 Hz");
        assert!((hz[2] - 261.6256).abs() < 1e-2, "C4 ≈ 261.63 Hz, got {}", hz[2]);
    }

    // ── Phase 6 (S53): Option-A f0 + transpose (build_note_hz) ──

    #[test]
    fn build_note_hz_bare_transpose() {
        // A4 (69) for 4 frames, 1 vowel phone → 4 cv frames all A4. transpose+12 → A5 (880 Hz).
        let score = [("あ", 69, 4)];
        let arr = build_arrays_daw(&score).unwrap();
        let hz0 = build_note_hz(&arr, &score, 0, None);
        assert_eq!(hz0.len(), 4);
        assert!(hz0.iter().all(|&h| (h - 440.0).abs() < 0.5), "A4 → 440, got {:?}", hz0);
        let hz12 = build_note_hz(&arr, &score, 12, None);
        assert!(hz12.iter().all(|&h| (h - 880.0).abs() < 1.0), "A4+12st → 880, got {:?}", hz12);
    }

    #[test]
    fn build_note_hz_option_a_samples_cents() {
        // f0 comes from the DAW cents curve, NOT the note's own pitch (note is 60, curve is A4=6900¢).
        let score = [("あ", 60, 6)];
        let arr = build_arrays_daw(&score).unwrap();
        let cents = vec![6900.0f32; 6];
        let voiced = vec![1u8; 6];
        let f0 = VocalF0 { cents: &cents, voiced: &voiced };
        let hz = build_note_hz(&arr, &score, 0, Some(&f0));
        assert_eq!(hz.len(), 6);
        assert!(hz.iter().all(|&h| (h - 440.0).abs() < 0.5), "6900¢ → 440Hz (ignores note 60), got {:?}", hz);
        // transpose +12st adds 1200¢ → A5 = 880 Hz.
        let hz12 = build_note_hz(&arr, &score, 12, Some(&f0));
        assert!(hz12.iter().all(|&h| (h - 880.0).abs() < 1.0), "6900¢+12st → 880, got {:?}", hz12);
    }

    #[test]
    fn build_note_hz_option_a_rest_and_unvoiced() {
        // note / rest / note — the rest group is silent, and cv↔DAW frames align (uncapped rest).
        let score = [("あ", 60, 4), ("R", 0, 4), ("い", 62, 4)];
        let arr = build_arrays_daw(&score).unwrap();
        assert_eq!(arr.phone_dur.iter().sum::<i64>(), 12, "1+1+1 phones, 4+4(uncapped SP)+4 frames");
        let mut cents = vec![0.0f32; 12];
        let mut voiced = vec![0u8; 12];
        for c in cents.iter_mut().take(4) {
            *c = 6000.0;
        }
        for v in voiced.iter_mut().take(4) {
            *v = 1;
        }
        for c in cents.iter_mut().skip(8) {
            *c = 6200.0;
        }
        for v in voiced.iter_mut().skip(8) {
            *v = 1;
        }
        let f0 = VocalF0 { cents: &cents, voiced: &voiced };
        let hz = build_note_hz(&arr, &score, 0, Some(&f0));
        assert_eq!(hz.len(), 12);
        assert!(hz[0..4].iter().all(|&h| h > 200.0), "note0 voiced, got {:?}", &hz[0..4]);
        assert!(hz[4..8].iter().all(|&h| h == 0.0), "rest group → silent, got {:?}", &hz[4..8]);
        assert!(hz[8..12].iter().all(|&h| h > 200.0), "note2 voiced, got {:?}", &hz[8..12]);
    }

    // Note-model contrast (user Q, 2026-07-09): `ー` sustain (prolongation) vs a repeated `か` token
    // (re-articulation). The audible difference is a SECOND consonant, and it is decided here at the
    // phone level. `か`+`ー` same pitch = ONE held "ka" (phones k,a,a — one note group, no re-attack);
    // `か`+`か` = "ka-ka" (phones k,a,k,a — the 2nd 'k' is the re-attack). NOT a Phase-2 bug: this is the
    // ported build_arrays (1c, bit-exact vs render_ust), just made explicit.
    #[test]
    fn sustain_vs_rearticulation_phones() {
        let sustain = build_arrays(&[("か", 60, 80), ("ー", 60, 80)]).unwrap();
        assert_eq!(sustain.phon, vec!["k", "a", "a"], "ー extends the previous vowel — no 2nd consonant");
        assert_eq!(sustain.note_to_phone, vec![0, 0, 0], "same pitch → one held note group");

        let reartic = build_arrays(&[("か", 60, 80), ("か", 60, 80)]).unwrap();
        assert_eq!(reartic.phon, vec!["k", "a", "k", "a"], "a second か token re-attacks the consonant");
    }

    #[test]
    fn banker_rounding_matches_python() {
        assert_eq!(round_half_even(990.5273), 991);
        assert_eq!(round_half_even(422.05), 422);
        assert_eq!(round_half_even(0.5), 0); // half → even
        assert_eq!(round_half_even(1.5), 2); // half → even
        assert_eq!(round_half_even(2.5), 2); // half → even
    }

    // A sub-min_frames chunk must LOUD-error at the guard, BEFORE net_g is ever run (no model needed —
    // the guard returns first). Mirrors the cover paths' short-piece guards (sovits.rs:255 / rvc.rs:291).
    #[test]
    fn sovits_short_chunk_errors_before_decode() {
        let engine = OnnxEngine::new();
        let feed = SovitsFeed { cv: Array2::zeros((3, 768)), f0: vec![0.0; 3], uv: vec![1.0; 3], t_tgt: 3 };
        let voice = SovitsVoice { features_dim: 768, noise_channels: 192, vol_embedding: false, sample_rate: 44100, hop_size: 512, min_frames: 6 };
        let r = sovits_decode_chunk(&engine, "no-such-session", &feed, 0, &voice, None, 0, 0, 0.4);
        let msg = format!("{}", r.expect_err("t_tgt=3 < min_frames=6 must error"));
        assert!(msg.contains("片段过短"), "expected the min_frames guard, got: {msg}");
    }

    #[test]
    fn rvc_short_chunk_errors_before_decode() {
        let engine = OnnxEngine::new();
        let feed = RvcFeed { phone: Array2::zeros((8, 768)), pitchf: vec![0.0; 8], pitch: vec![1; 8], t: 8 };
        let voice = RvcVoice { features_dim: 768, noise_channels: 192, sample_rate: 48000, min_frames: 12 };
        let r = rvc_decode_chunk(&engine, "no-such-session", &feed, 0, &voice, 0, 0, 0.66666);
        let msg = format!("{}", r.expect_err("t=8 < min_frames=12 must error"));
        assert!(msg.contains("片段过短"), "expected the min_frames guard, got: {msg}");
    }

    // ── Phase 2 GATE (Tier-1): the deterministic net_g INPUT tensors reproduce the Python reference
    //    (score2svc_ref.rs, dumped by dump_score2svc_ref.py) bit-for-bit on the fixed score. Needs the
    //    181MB score2cv models (data/models/aux) + the dev ORT dll — hence #[ignore]; run:
    //      cargo test --lib inference::score2svc::tests::score2svc_glue_parity_cpu -- --ignored --nocapture
    //    Forces CPU EP so numerics equal the Python CPUExecutionProvider reference exactly. ──
    #[test]
    #[ignore]
    fn score2svc_glue_parity_cpu() {
        use super::super::engine::DeviceConfig;
        use super::super::score2svc_ref::SVC_REFS;
        use std::path::{Path, PathBuf};

        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let dll = root.join("../runtime/ort/onnxruntime.dll");
        assert!(dll.exists(), "ORT dll missing at {} (dev runtime required)", dll.display());
        if let Ok(b) = ort::init_from(&dll) {
            let _ = b.commit();
        }
        let engine = OnnxEngine::new();
        engine.set_device(DeviceConfig::Cpu);

        let arr = build_arrays(pr::SCORE).unwrap();
        let chunks = chunk_at_sp(&arr, 400);
        assert_eq!(chunks.len(), SVC_REFS.len(), "chunk count vs reference");

        // f0 glue (midi/note_hz/f0_rs/uv_rs/t_tgt) is dim-independent — re-checked on each dim pass
        // (cheap); cv_rs is the dim-specific reference.
        for (dim, model) in [(768usize, "score2cv_768.onnx"), (256usize, "score2cv_256.onnx")] {
            let path: PathBuf = root.join("../data/models/aux").join(model);
            assert!(path.exists(), "model missing: {}", path.display());
            let sid = engine.load_model_with(&path, false).unwrap();

            for (ci, chunk) in chunks.iter().enumerate() {
                let r = &SVC_REFS[ci];
                let cvr = if dim == 768 { &r.cv768_rs } else { &r.cv256_rs };
                let cv = run_score2cv(&engine, &sid, chunk, dim, 49, 2).unwrap();

                let midi = midi_frame_50(&chunk.note_pitch, &chunk.phone_dur);
                let note_hz = note_hz_50(&midi);
                let feed = resample_to_sovits_grid(&cv, &note_hz, SOVITS_SR, SOVITS_HOP).unwrap();

                // midi_frame: bit-exact (i64 length-regulation)
                assert_eq!(midi.as_slice(), r.midi_frame, "{} c{}: midi_frame", model, ci);
                // t_tgt: exact
                assert_eq!(feed.t_tgt, r.t_tgt, "{} c{}: t_tgt", model, ci);
                // note_hz @50fps: tight tolerance (transcendental pow, f64→f32; a real bug is Hz-scale)
                let nh_worst = worst_abs(&note_hz, r.note_hz);
                assert!(nh_worst <= 1e-2, "{} c{}: note_hz worst {:.3e} Hz > 1e-2", model, ci, nh_worst);
                // f0_rs @86fps: same tolerance (resample is exact index pick of note_hz)
                let f0_worst = worst_abs(&feed.f0, r.f0_rs);
                assert!(f0_worst <= 1e-2, "{} c{}: f0_rs worst {:.3e} Hz > 1e-2", model, ci, f0_worst);
                // uv_rs @86fps: bit-exact (0.0/1.0)
                assert_eq!(feed.uv.len(), r.uv_rs.len(), "{} c{}: uv len", model, ci);
                assert!(
                    feed.uv.iter().zip(r.uv_rs).all(|(a, b)| a == b),
                    "{} c{}: uv_rs mismatch", model, ci
                );
                // cv_rs @86fps: sampled ≤1e-3 + global stats (mirrors the 1d cv gate)
                assert_eq!(cv.nrows(), r.t, "{} c{}: cv T50", model, ci);
                assert_eq!(feed.cv.nrows(), r.t_tgt, "{} c{}: cv_rs rows", model, ci);
                assert_eq!(feed.cv.ncols(), dim, "{} c{}: cv_rs dim", model, ci);
                let flat = feed.cv.as_slice().expect("cv_rs contiguous");
                let mut worst = 0.0f32;
                for (&i, &v) in cvr.idx.iter().zip(cvr.val) {
                    worst = worst.max((flat[i] - v).abs());
                }
                assert!(worst <= 1e-3, "{} c{}: cv_rs sampled worst {:.3e} > 1e-3", model, ci, worst);
                let sum: f64 = flat.iter().map(|&x| x as f64).sum();
                let sumsq: f64 = flat.iter().map(|&x| (x as f64) * (x as f64)).sum();
                assert!((sum - cvr.sum).abs() <= 0.1 + cvr.sum.abs() * 1e-4, "{} c{}: cv_rs sum", model, ci);
                assert!((sumsq - cvr.sumsq).abs() <= 0.1 + cvr.sumsq * 1e-4, "{} c{}: cv_rs sumsq", model, ci);

                let voiced = feed.uv.iter().filter(|&&u| u < 0.5).count();
                eprintln!(
                    "[P2/Tier1] {} c{}: T={} t_tgt={} voiced={}/{} note_hz≤{:.1e} f0≤{:.1e} cv≤{:.1e} PASS",
                    model, ci, r.t, r.t_tgt, voiced, r.t_tgt, nh_worst, f0_worst, worst
                );
            }
        }
    }

    fn worst_abs(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "length mismatch {} != {}", a.len(), b.len());
        a.iter().zip(b).fold(0.0f32, |w, (&x, &y)| w.max((x - y).abs()))
    }

    // ── Phase 2 AUDITION (Tier-2): render the fixed score end-to-end through the REAL voice net_g
    //    (东雪莲 4.1/768, akiko 4.0/256, lengv2 RVC v2/768) and write wavs for the EAR — plus a
    //    legato-vs-SP A/B demo (§3.4: `ー` sustain continues voiced vs `R` rest = silence). Non-
    //    deterministic (net_g), so no assert beyond non-silence; the numeric gate is Tier-1. Needs the
    //    voice models — hence #[ignore]. Run:
    //      cargo test --lib inference::score2svc::tests::render_audition_wavs -- --ignored --nocapture
    #[test]
    #[ignore]
    fn render_audition_wavs() {
        use super::super::engine::DeviceConfig;
        use std::path::{Path, PathBuf};

        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let dll = root.join("../runtime/ort/onnxruntime.dll");
        assert!(dll.exists(), "ORT dll missing at {}", dll.display());
        if let Ok(b) = ort::init_from(&dll) {
            let _ = b.commit();
        }
        let engine = OnnxEngine::new();
        engine.set_device(DeviceConfig::Cpu); // deterministic-ish + no GPU setup in a test

        let aux = root.join("../data/models/aux");
        let sov = root.join("../data/models/sovits");
        let rvcd = root.join("../data/models/rvc");
        let out = PathBuf::from(
            r"C:\Users\admin\AppData\Local\Temp\claude\D--MyDev-Utai-v2-dev\c0c6255b-6ea4-4b70-b88d-d3f6203bc23a\scratchpad\phase2_audition",
        );
        std::fs::create_dir_all(&out).unwrap();

        let s2cv768 = engine.load_model_with(&aux.join("score2cv_768.onnx"), false).unwrap();
        let s2cv256 = engine.load_model_with(&aux.join("score2cv_256.onnx"), false).unwrap();

        // legato (承前元音, voiced-continuous pitch jump) vs SP rest (silence gap) — §3.4, same 2 notes.
        const LEGATO: &[(&str, i64, i64)] = &[("か", 60, 80), ("ー", 67, 80), ("お", 67, 80)];
        const REST: &[(&str, i64, i64)] = &[("か", 60, 80), ("R", 0, 80), ("お", 67, 80)];
        // note-model contrast (user Q, 2026-07-09): SUSTAIN `ー` at the SAME pitch = ONE held "ka"
        // (phones [k,a,a], no 2nd consonant) vs REARTIC = two `か` tokens = "ka-ka" (phones [k,a,k,a],
        // a 2nd 'k' re-attack). Proves the note model distinguishes prolongation from re-articulation.
        const SUSTAIN: &[(&str, i64, i64)] = &[("か", 60, 80), ("ー", 60, 80)];
        const REARTIC: &[(&str, i64, i64)] = &[("か", 60, 80), ("か", 60, 80)];

        // Phase-6 signature: bare noteonly f0 (None) + no transpose + no-op cancel/progress = the
        // Phase-2 render path (this ear test predates the DAW f0/transpose/cancel wiring).
        let no_cancel = || false;
        let no_prog = |_: f32| {};

        let mut wrote: Vec<(String, usize, f32, u32)> = Vec::new();
        let mut save = |name: &str, r: &SynthesisResult| {
            let peak = r.audio.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            write_wav16(&out.join(format!("{name}.wav")), &r.audio, r.sample_rate);
            wrote.push((name.to_string(), r.audio.len(), peak, r.sample_rate));
        };

        // akiko 4.0 / 256 (vol-free — cleanest audible on the 256 path)
        let akiko = engine.load_model_with(&sov.join("akiko_320000.onnx"), false).unwrap();
        let av = SovitsVoice { features_dim: 256, noise_channels: 192, vol_embedding: false, sample_rate: 44100, hop_size: 512, min_frames: 6 };
        save("p2_akiko256_main", &render_score_sovits(&engine, &s2cv256, &akiko, pr::SCORE, 256, 49, 2, &av, 0, 0.0, 0, 0.4, 0, None, &no_cancel, &no_prog).unwrap());
        save("p2_akiko256_demo_legato", &render_score_sovits(&engine, &s2cv256, &akiko, LEGATO, 256, 49, 2, &av, 0, 0.0, 0, 0.4, 0, None, &no_cancel, &no_prog).unwrap());
        save("p2_akiko256_demo_rest", &render_score_sovits(&engine, &s2cv256, &akiko, REST, 256, 49, 2, &av, 0, 0.0, 0, 0.4, 0, None, &no_cancel, &no_prog).unwrap());
        save("p2_akiko256_demo_sustain_same", &render_score_sovits(&engine, &s2cv256, &akiko, SUSTAIN, 256, 49, 2, &av, 0, 0.0, 0, 0.4, 0, None, &no_cancel, &no_prog).unwrap());
        save("p2_akiko256_demo_reartic_same", &render_score_sovits(&engine, &s2cv256, &akiko, REARTIC, 256, 49, 2, &av, 0, 0.0, 0, 0.4, 0, None, &no_cancel, &no_prog).unwrap());

        // 东雪莲 4.1 / 768 (SAME voice as the Python reference; vol_embedding → flat placeholder vol)
        let dx = engine.load_model_with(&sov.join("Sovits4.1东雪莲主模型.onnx"), false).unwrap();
        let dv = SovitsVoice { features_dim: 768, noise_channels: 192, vol_embedding: true, sample_rate: 44100, hop_size: 512, min_frames: 6 };
        save("p2_dongxuelian768_main", &render_score_sovits(&engine, &s2cv768, &dx, pr::SCORE, 768, 49, 2, &dv, 0, 0.1, 0, 0.4, 0, None, &no_cancel, &no_prog).unwrap());

        // RVC v2 lengv2 / 768 (100 fps grid; no Python A/B reference — audible + glue self-consistency)
        let leng = engine.load_model_with(&rvcd.join("lengv2.3.onnx"), false).unwrap();
        let rv = RvcVoice { features_dim: 768, noise_channels: 192, sample_rate: 48000, min_frames: 12 };
        save("p2_rvc_lengv2_main", &render_score_rvc(&engine, &s2cv768, &leng, pr::SCORE, 768, 49, 2, &rv, 0, 0, 0.66666, 0, None, &no_cancel, &no_prog).unwrap());

        drop(save); // release the &mut wrote borrow before reading it back
        eprintln!("\n[P2/Tier2] wrote {} wavs to {}", wrote.len(), out.display());
        for (name, n, peak, sr) in &wrote {
            eprintln!("  {name}.wav  {:.2}s  peak={:.3}  ({} samples @ {} Hz)", *n as f32 / *sr as f32, peak, n, sr);
            assert!(*peak > 1e-3, "{name}: rendered audio is silent (peak {})", peak);
        }
    }

    fn write_wav16(path: &std::path::Path, samples: &[f32], sr: u32) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: sr,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(path, spec).unwrap();
        for &s in samples {
            w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16).unwrap();
        }
        w.finalize().unwrap();
    }
}
