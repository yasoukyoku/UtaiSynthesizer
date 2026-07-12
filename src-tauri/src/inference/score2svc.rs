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

#[cfg(test)]
use super::engine::OnnxEngine; // only the #[ignore] gates name the engine type; the render fns use m.engine
use super::features::{repeat_expand_2d, torch_interp_nearest};
use super::g2p::{self, Lang, ScoreEvt};
use super::rvc::{f0_to_coarse, vc_decode, RvcModel};
use super::score2cv::{
    build_arrays_daw, chunk_at_sp, classify_lyric, run_score2cv, LyricClass, ScoreArrays,
};
use super::sovits::{apply_cluster_blend, decode_features, SovitsModel};
use super::{build_spk_mix_dense, RvcOptions, SovitsOptions, SynthesisResult};
use crate::{Result, UtaiError};
use utai_dsp::formant_warp;

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

/// Per-note-group cv↔DAW 50fps frame ranges — the SHARED remap core of `build_note_hz`/`build_note_param`.
/// The cv side comes from `arr.note_to_phone`/`arr.phone_dur`; the DAW side from the score triples grouped
/// by the SAME `npitch` rule `build_arrays` uses (rest/breath → 0, else note_num, via `classify_lyric`), so
/// f0 / loudness / formant all share ONE alignment source (NO-duplication). Pure.
struct NoteGroups {
    ng: usize,
    cv_start: Vec<usize>,
    cv_count: Vec<usize>,
    daw_start: Vec<usize>,
    daw_count: Vec<usize>,
    group_pitch: Vec<i64>,
}

fn compute_note_groups(arr: &ScoreArrays, score: &[ScoreEvt]) -> NoteGroups {
    let ng = arr.note_to_phone.last().map(|&g| g as usize + 1).unwrap_or(0);
    // Per-group cv frame range (from arr): the group's start cv frame + total cv frames.
    let mut cv_start = vec![0usize; ng];
    let mut cv_count = vec![0usize; ng];
    if ng > 0 {
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
    // Per-group DAW frame range (from the score notes). Grouped by the SAME (npitch, RUN LANGUAGE) rule
    // the assembly uses — rest → 0 (classified via the single `classify_lyric`'s universal tokens, so a
    // rest with a stray non-zero note_num still groups as a rest) + `g2p::note_run_langs` (S58: the ONE
    // shared run-language source, so cv-side and DAW-side grouping can never drift; §9.5 editor==render).
    let mut daw_start = vec![0usize; ng];
    let mut daw_count = vec![0usize; ng];
    let mut group_pitch = vec![0i64; ng];
    if ng > 0 {
        let run_langs = g2p::note_run_langs(score);
        let mut g: i64 = -1;
        let mut prev: Option<(i64, Lang)> = None;
        let mut cursor = 0usize;
        let mut seen = vec![false; ng];
        for (k, evt) in score.iter().enumerate() {
            let npitch = if matches!(classify_lyric(evt.lyric), LyricClass::Rest | LyricClass::Breath) {
                0
            } else {
                evt.note_num
            };
            if prev != Some((npitch, run_langs[k])) {
                g += 1;
                prev = Some((npitch, run_langs[k]));
            }
            let gi = (g as usize).min(ng - 1);
            if !seen[gi] {
                daw_start[gi] = cursor;
                group_pitch[gi] = npitch;
                seen[gi] = true;
            }
            let d = evt.frames.max(0) as usize;
            daw_count[gi] += d;
            cursor += d;
        }
    }
    NoteGroups { ng, cv_start, cv_count, daw_start, daw_count, group_pitch }
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
    score: &[ScoreEvt],
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

    let g = compute_note_groups(arr, score);
    if g.ng == 0 || f0.cents.is_empty() {
        return vec![0.0; t_total];
    }
    let flen = f0.cents.len();
    let mut out = vec![0.0f32; t_total];
    for gi in 0..g.ng {
        if g.group_pitch[gi] <= 0 || g.cv_count[gi] == 0 {
            continue; // rest group → 0 Hz (unvoiced); nothing to sample
        }
        for k in 0..g.cv_count[gi] {
            let cv_i = g.cv_start[gi] + k;
            if cv_i >= t_total {
                break;
            }
            // map this cv frame (center) to a DAW 50fps index within the group's DAW span.
            let frac = (k as f64 + 0.5) / g.cv_count[gi] as f64;
            let daw_f = g.daw_start[gi] as f64 + frac * g.daw_count[gi] as f64;
            let idx = (daw_f.floor() as usize).min(flen - 1);
            if f0.voiced.get(idx).copied().unwrap_or(0) != 0 {
                out[cv_i] = cents_to_hz(f0.cents[idx] as f64 + (transpose as f64) * 100.0);
            }
        }
    }
    out
}

/// Map a whole-score @50fps DAW envelope (`env`, length = Σ triple frames) to a per-cv-frame @50fps array
/// (length = t_total = Σ phone_dur) via the SAME note-group cv↔DAW remap as `build_note_hz` — so a loudness
/// or formant lane aligns with cv/f0 exactly even where a short note inflated its cv frames. EVERY group is
/// sampled (rests too, for a continuous envelope — unlike f0 which zeroes rest groups). Empty `env` → all
/// `default` (the flat / no-lane path = a no-op at the render). Pure.
fn build_note_param(arr: &ScoreArrays, score: &[ScoreEvt], env: &[f32], default: f32) -> Vec<f32> {
    let t_total: usize = arr.phone_dur.iter().map(|&d| d.max(0) as usize).sum();
    if env.is_empty() {
        return vec![default; t_total];
    }
    let g = compute_note_groups(arr, score);
    let flen = env.len();
    let mut out = vec![default; t_total];
    for gi in 0..g.ng {
        if g.cv_count[gi] == 0 {
            continue;
        }
        for k in 0..g.cv_count[gi] {
            let cv_i = g.cv_start[gi] + k;
            if cv_i >= t_total {
                break;
            }
            let frac = (k as f64 + 0.5) / g.cv_count[gi] as f64;
            let daw_f = g.daw_start[gi] as f64 + frac * g.daw_count[gi] as f64;
            let idx = (daw_f.floor() as usize).min(flen - 1);
            out[cv_i] = env[idx];
        }
    }
    out
}

/// Apply a per-cv-frame absolute-gain envelope (loudness multiplier, length = t_total) to the concatenated
/// mono audio IN PLACE: sample `s` maps to cv frame `floor(s/len · t_total)`, scaled by `gain_cv[cv]`.
/// Uniform map (each cv frame ≈ equal audio samples). Applied BEFORE `peak_normalize` so it shapes RELATIVE
/// dynamics — the volume fader owns the absolute level (§M-defer). Empty/flat env ⇒ untouched.
fn apply_gain_env(audio: &mut [f32], gain_cv: &[f32]) {
    if gain_cv.is_empty() || audio.is_empty() {
        return;
    }
    let n = audio.len() as f64;
    let tt = gain_cv.len();
    for (s, v) in audio.iter_mut().enumerate() {
        let cv = ((s as f64 / n) * tt as f64).floor() as usize;
        *v *= gain_cv[cv.min(tt - 1)];
    }
}

/// Warp the formant envelope of the concatenated mono audio by a per-cv-frame SEMITONE envelope
/// (`formant_cv`, length = t_total), ratio = 2^(semi/12); sample→cv via the same uniform map as
/// `apply_gain_env`. `formant_warp` passes ratio≈1 frames through verbatim, so an all-zero envelope is
/// (near) lossless. Empty env ⇒ returns the audio unchanged. Applied AFTER the loudness gain, BEFORE
/// `peak_normalize` (§M-defer order: 响度增益 → 共振腔 → 归一化).
fn apply_formant_env(audio: Vec<f32>, formant_cv: &[f32]) -> Vec<f32> {
    if formant_cv.is_empty() || audio.is_empty() {
        return audio;
    }
    let n = audio.len() as f64;
    let tt = formant_cv.len();
    formant_warp(&audio, |s| {
        let cv = ((s as f64 / n) * tt as f64).floor() as usize;
        2.0_f32.powf(formant_cv[cv.min(tt - 1)] / 12.0)
    })
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

// ─── score → SoVITS render (Item-1: reuses the SHARED cover-path decode tail) ─────────────────────

/// Pad a sub-`min` SoVITS feed (a short trailing chunk) up to `min` frames by REPEATING the last frame
/// of cv/f0/uv so net_g accepts the shape; returns the ORIGINAL t_tgt so the caller trims the pad off the
/// decoded wav. A no-op (returns t_tgt) when already ≥ min or empty. M3 short-note handling.
fn pad_sovits_feed(feed: &mut SovitsFeed, min: usize) -> usize {
    let orig = feed.t_tgt;
    if orig >= min || orig == 0 {
        return orig;
    }
    let dim = feed.cv.ncols();
    let mut cv = Array2::<f32>::zeros((min, dim));
    for i in 0..min {
        cv.row_mut(i).assign(&feed.cv.row(i.min(orig - 1)));
    }
    feed.cv = cv;
    let last_f0 = *feed.f0.last().unwrap_or(&0.0);
    let last_uv = *feed.uv.last().unwrap_or(&0.0);
    feed.f0.resize(min, last_f0);
    feed.uv.resize(min, last_uv);
    feed.t_tgt = min;
    orig
}

/// Full score → SoVITS wav (自己唱). build_arrays_daw (rests uncapped → stem aligns to the timeline) →
/// SP-chunk (≤400) → per chunk: run_score2cv → cv; f0 from `build_note_hz` (bare noteonly when `f0` is
/// None, else the DAW's layered Option-A pitch); resample to the hop grid; cluster/retrieval blend; then
/// the SHARED `decode_features` (spk_mix + shallow/only diffusion + NSF enhancer — the SAME quality path
/// the 翻唱 render uses, no longer a net_g-only copy). Chunk wavs are concatenated then peak-normalized to
/// 0.92. `transpose` (semitones) shifts both content note_pitch and f0 (§9.3, Rust-only). `flat_vol` seeds
/// the placeholder vol tensor for vol_embedding models. A sub-min chunk is padded then trimmed (M3).
/// `cancel`/`progress` are polled per chunk. ⚠ `options.auto_f0`/`f0_shift`/`loudness_envelope` MUST be
/// neutralized by the caller (the command layer) — auto-f0 would overwrite the DAW f0 (Option-A).
#[allow(clippy::too_many_arguments)]
pub fn render_score_sovits(
    m: &SovitsModel,
    score2cv_session: &str,
    score: &[ScoreEvt],
    dim: usize,
    cv_speaker_id: i64,
    dicts: &dyn g2p::DictSource,
    options: &SovitsOptions,
    flat_vol: f32,
    transpose: i64,
    range_shift: i64,
    f0: Option<&VocalF0>,
    loudness: Option<&[f32]>,
    formant: Option<&[f32]>,
    cancel: &(dyn Fn() -> bool + Sync),
    progress: &dyn Fn(f32),
) -> Result<SynthesisResult> {
    let mut arr = build_arrays_daw(score, dicts)?;
    // S60-2 音域扩展: the model RENDERS at transpose+range_shift (inside its comfort zone);
    // apply_range_inverse below shifts the audio back. range_shift=0 ⇒ byte-identical to before.
    let transpose_eff = transpose + range_shift;
    // f0 uses RAW note_pitch (grouping/voicing); transpose folds into the OUTPUT Hz.
    let note_hz_full = build_note_hz(&arr, score, transpose_eff, f0);
    // ② per-cv-frame loudness (multiplier, unity default) + formant (semitones, 0 default) envelopes,
    // aligned to cv via the SAME group remap as f0 (short-note-inflation safe). None/empty ⇒ flat = no-op.
    let loud_cv = loudness.map(|l| build_note_param(&arr, score, l, 1.0));
    let formant_cv = formant.map(|f| build_note_param(&arr, score, f, 0.0));
    // content note_pitch is transposed separately (grouping is shift-invariant).
    transpose_note_pitch(&mut arr.note_pitch, transpose_eff);
    let chunks = chunk_at_sp(&arr, 400);
    let n_chunks = chunks.len().max(1);
    let has_diff = m.diffusion.is_some();
    let p_vits = if has_diff { 0.5 } else { 0.95 };
    let noop = |_: f32| {}; // decode_features' internal sub-progress is ignored (per-chunk coarse below)
    let mut audio: Vec<f32> = Vec::new();
    let mut cv_cursor = 0usize;
    for (ci, chunk) in chunks.iter().enumerate() {
        if cancel() {
            return Err(UtaiError::Inference("已取消".into()));
        }
        let cv = run_score2cv(m.engine, score2cv_session, chunk, dim, cv_speaker_id, chunk.lang_id)?;
        let note_hz = &note_hz_full[cv_cursor..(cv_cursor + chunk.t).min(note_hz_full.len())];
        let mut feed = resample_to_sovits_grid(&cv, note_hz, m.sample_rate, m.hop_size)?;
        let real_t = pad_sovits_feed(&mut feed, m.min_frames); // M3: short trailing chunk
        apply_cluster_blend(&mut feed.cv, m.cluster, options.cluster_ratio); // SHARED retrieval blend
        // vol_embedding (SoVITS 4.1): feed net_g a per-frame loudness = flat_vol · lane multiplier (this
        // chunk's cv slice resampled to the hop grid). No lane ⇒ the original flat placeholder (parity).
        let vol = if m.vol_embedding {
            Some(match &loud_cv {
                Some(lc) => {
                    let seg = &lc[cv_cursor..(cv_cursor + chunk.t).min(lc.len())];
                    torch_interp_nearest(seg, feed.t_tgt).into_iter().map(|mult| flat_vol * mult).collect()
                }
                None => vec![flat_vol; feed.t_tgt],
            })
        } else {
            None
        };
        let padded_t = feed.t_tgt;
        // score has no source wav → wav_m is `&[]` (only read by only_diffusion + no-vol, which the
        // command layer disallows for the score path).
        let mut wav = decode_features(
            m, feed.cv, feed.f0, feed.uv, vol, &[], ci as u64, padded_t, has_diff, p_vits, options,
            &noop, cancel,
        )?;
        if padded_t > real_t {
            wav.truncate((real_t * m.hop_size).min(wav.len())); // drop the pad samples
        }
        if chunk.hard_seam {
            seam_fade(&mut audio, &mut wav, m.sample_rate); // S58: mid-voiced language cut → micro-fade
        }
        audio.extend_from_slice(&wav);
        cv_cursor += chunk.t;
        progress((ci + 1) as f32 / n_chunks as f32);
    }
    // Post-decode (§M-defer order 响度增益 → 共振腔 → 归一化): a vol_embedding model already got loudness in
    // net_g above, so gain it here ONLY when there's no vol port (4.0); formant warps both. Then normalize.
    if !m.vol_embedding {
        if let Some(lc) = &loud_cv {
            apply_gain_env(&mut audio, lc);
        }
    }
    if let Some(fc) = &formant_cv {
        audio = apply_formant_env(audio, fc);
    }
    audio = apply_range_inverse(audio, &note_hz_full, m.sample_rate, range_shift);
    peak_normalize(&mut audio, 0.92);
    Ok(SynthesisResult { audio, sample_rate: m.sample_rate })
}

// ─── score → RVC render (Item-1: reuses the SHARED cover-path `vc_decode` tail) ───────────────────

/// Build the RVC 100 fps pitch grid from a chunk's `(cv @50fps, note_hz @50fps)`: each note_hz frame is
/// repeated twice (pitchf), coarse-binned (pitch). cv stays 50 fps — `vc_decode` upsamples it 2× itself
/// (so the retrieval/protect blend runs on the same 50 fps features the cover path uses). M3: if
/// `2·T50 < min` (a short trailing note), cv is padded (repeat last frame) to `ceil(min/2)` and the pitch
/// grid to `2·ceil(min/2)`. Returns `(cv, pitch, pitchf, real_100fps_frames)` — the last is the pre-pad
/// 100 fps length so the caller trims the pad off the decoded wav.
fn rvc_feed_100(mut cv: Array2<f32>, note_hz: &[f32], min: usize) -> (Array2<f32>, Vec<i64>, Vec<f32>, usize) {
    let t50 = cv.nrows();
    // Defensive: a 0-row cv (unreachable — chunk_at_sp emits only s<e ranges with pdur≥1, and run_score2cv
    // errors before yielding 0 rows) would panic the pad loop's `cv.row(0)`. Symmetric with the SoVITS
    // path's t_tgt==0 guard (resample_to_sovits_grid); vc_decode then LOUD-errors on the sub-min frame count.
    if t50 == 0 {
        return (cv, Vec::new(), Vec::new(), 0);
    }
    let real_100 = t50 * 2;
    let pad50 = if real_100 < min { min.div_ceil(2) } else { t50 };
    if pad50 > t50 {
        let dim = cv.ncols();
        let mut padded = Array2::<f32>::zeros((pad50, dim));
        for i in 0..pad50 {
            padded.row_mut(i).assign(&cv.row(i.min(t50.saturating_sub(1))));
        }
        cv = padded;
    }
    let mut pitchf: Vec<f32> = Vec::with_capacity(pad50 * 2);
    for i in 0..pad50 {
        let f = note_hz.get(i.min(note_hz.len().saturating_sub(1))).copied().unwrap_or(0.0);
        pitchf.push(f);
        pitchf.push(f);
    }
    let pitch: Vec<i64> = pitchf.iter().map(|&f| f0_to_coarse(f)).collect();
    (cv, pitch, pitchf, real_100)
}

/// Full score → RVC wav (自己唱). Same shape as `render_score_sovits` but on the 100 fps grid, no uv/vol,
/// and the SHARED `vc_decode` tail (index retrieval + protect + net_g — the SAME the 翻唱 render uses).
/// RVC v2 uses cv768 (dim=768), same as SoVITS 4.1. A sub-min chunk is padded then trimmed (M3). ⚠ the
/// command layer neutralizes `options.f0_shift`/`rms_mix_rate` (redundant with transpose / no source wav).
#[allow(clippy::too_many_arguments)]
pub fn render_score_rvc(
    m: &RvcModel,
    score2cv_session: &str,
    score: &[ScoreEvt],
    dim: usize,
    cv_speaker_id: i64,
    dicts: &dyn g2p::DictSource,
    options: &RvcOptions,
    transpose: i64,
    range_shift: i64,
    f0: Option<&VocalF0>,
    loudness: Option<&[f32]>,
    formant: Option<&[f32]>,
    cancel: &(dyn Fn() -> bool + Sync),
    progress: &dyn Fn(f32),
) -> Result<SynthesisResult> {
    let mut arr = build_arrays_daw(score, dicts)?;
    // S60-2 音域扩展 — same recipe as the SoVITS render: sing at transpose+range_shift, shift back below.
    let transpose_eff = transpose + range_shift;
    let note_hz_full = build_note_hz(&arr, score, transpose_eff, f0);
    // ② loudness/formant per-cv-frame envelopes (RVC has no vol port → both are post-decode). Empty ⇒ no-op.
    let loud_cv = loudness.map(|l| build_note_param(&arr, score, l, 1.0));
    let formant_cv = formant.map(|f| build_note_param(&arr, score, f, 0.0));
    transpose_note_pitch(&mut arr.note_pitch, transpose_eff);
    let chunks = chunk_at_sp(&arr, 400);
    let n_chunks = chunks.len().max(1);
    let sid = options.speaker_id.unwrap_or(0) as i64;
    // ①c: a genuine multi-speaker RVC export takes a dense spk_mix blend in place of scalar sid.
    let spk_mix_dense = m
        .spk_mix
        .map(|n| build_spk_mix_dense(&options.spk_mix, options.speaker_id, n));
    let mut audio: Vec<f32> = Vec::new();
    let mut cv_cursor = 0usize;
    for (ci, chunk) in chunks.iter().enumerate() {
        if cancel() {
            return Err(UtaiError::Inference("已取消".into()));
        }
        let cv = run_score2cv(m.engine, score2cv_session, chunk, dim, cv_speaker_id, chunk.lang_id)?;
        let note_hz = &note_hz_full[cv_cursor..(cv_cursor + chunk.t).min(note_hz_full.len())];
        let (cv_p, pitch, pitchf, real_t) = rvc_feed_100(cv, note_hz, m.min_frames);
        let mut wav = vc_decode(
            m, cv_p, &pitch, &pitchf, sid, spk_mix_dense.as_deref(), options, ci as u64, usize::MAX,
        )?;
        if pitchf.len() > real_t {
            // RVC net_g emits ~ p_len·(sr/100) samples; keep only the pre-pad span.
            wav.truncate((real_t * (m.sample_rate as usize / 100)).min(wav.len()));
        }
        if chunk.hard_seam {
            seam_fade(&mut audio, &mut wav, m.sample_rate); // S58: mid-voiced language cut → micro-fade
        }
        audio.extend_from_slice(&wav);
        cv_cursor += chunk.t;
        progress((ci + 1) as f32 / n_chunks as f32);
    }
    // Post-decode (§M-defer order 响度增益 → 共振腔 → 归一化): RVC has no net_g vol port, so loudness is an
    // absolute gain envelope; formant warps the timbre. Both no-op when their env is None/flat.
    if let Some(lc) = &loud_cv {
        apply_gain_env(&mut audio, lc);
    }
    if let Some(fc) = &formant_cv {
        audio = apply_formant_env(audio, fc);
    }
    audio = apply_range_inverse(audio, &note_hz_full, m.sample_rate, range_shift);
    peak_normalize(&mut audio, 0.92);
    Ok(SynthesisResult { audio, sample_rate: m.sample_rate })
}

/// Micro-fade a HARD chunk seam (a mid-voiced LANGUAGE cut, S58): linearly fade the tail of the
/// accumulated audio and the head of the incoming chunk over ~5 ms each. Sample counts are untouched
/// (never an overlap-shift — the stem must stay tick-aligned to the DAW timeline); the fades only mask
/// the waveform discontinuity of two independently decoded chunks. SP seams are silence and skip this.
fn seam_fade(audio: &mut [f32], wav: &mut [f32], sample_rate: u32) {
    let k = (sample_rate as usize / 200).max(1); // ≈5 ms
    let n = audio.len();
    let ka = k.min(n);
    for j in 0..ka {
        audio[n - 1 - j] *= (j + 1) as f32 / (ka + 1) as f32; // 1 → ~0 toward the seam
    }
    let kw = k.min(wav.len());
    for j in 0..kw {
        wav[j] *= (j + 1) as f32 / (kw + 1) as f32; // ~0 → 1 away from the seam
    }
}

/// S60-2 音域扩展: undo the range-extension shift in the AUDIO domain. The render was fed
/// `transpose + range_shift` (content + f0 together, so the model sings inside its comfort
/// zone); this shifts the decoded audio back by `-range_shift` semitones with TD-PSOLA,
/// guided by the EXACT fed f0 (`note_hz_cv`, @50fps cv frames — the same uniform sample→cv
/// map as `apply_gain_env`, resampled onto a 100 fps hop grid). Formants are preserved by
/// PSOLA itself (the v1 "raw F0 shift only" rule). shift 0 / empty ⇒ untouched (tier 1/2:
/// in-comfort renders NEVER pass through here — bit-parity by construction).
fn apply_range_inverse(audio: Vec<f32>, note_hz_cv: &[f32], sample_rate: u32, range_shift: i64) -> Vec<f32> {
    if range_shift == 0 || audio.is_empty() || note_hz_cv.is_empty() {
        return audio;
    }
    let hop = (sample_rate as usize / 100).max(1);
    let nfr = audio.len() / hop + 1;
    let n = audio.len() as f64;
    let tt = note_hz_cv.len();
    let mut f0 = Vec::with_capacity(nfr);
    for i in 0..nfr {
        let s = (i * hop).min(audio.len() - 1);
        let cv = ((s as f64 / n) * tt as f64).floor() as usize;
        f0.push(note_hz_cv[cv.min(tt - 1)]);
    }
    let ratio = vec![2f32.powf(-(range_shift as f32) / 12.0); f0.len()];
    utai_dsp::psola_shift(&audio, sample_rate, &f0, &ratio, utai_dsp::PsolaParams { hop })
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
    use super::super::score2cv::{build_arrays, NoDicts}; // Phase-1c parity entry (rest-capped)
    use super::super::score2cv_tables::parity_ref as pr;

    /// JA-defaulted events from legacy triples (the pre-S58 test fixtures).
    fn ja_evts<'a>(score: &'a [(&'a str, i64, i64)]) -> Vec<ScoreEvt<'a>> {
        score.iter().map(ScoreEvt::ja).collect()
    }
    /// DAW build over a JA triple fixture (rests uncapped + borrow-time).
    fn daw_ja(score: &[(&str, i64, i64)]) -> ScoreArrays {
        build_arrays_daw(&ja_evts(score), &NoDicts).unwrap()
    }

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
        let evts = ja_evts(&score);
        let arr = daw_ja(&score);
        let hz0 = build_note_hz(&arr, &evts, 0, None);
        assert_eq!(hz0.len(), 4);
        assert!(hz0.iter().all(|&h| (h - 440.0).abs() < 0.5), "A4 → 440, got {:?}", hz0);
        let hz12 = build_note_hz(&arr, &evts, 12, None);
        assert!(hz12.iter().all(|&h| (h - 880.0).abs() < 1.0), "A4+12st → 880, got {:?}", hz12);
    }

    #[test]
    fn build_note_hz_option_a_samples_cents() {
        // f0 comes from the DAW cents curve, NOT the note's own pitch (note is 60, curve is A4=6900¢).
        let score = [("あ", 60, 6)];
        let evts = ja_evts(&score);
        let arr = daw_ja(&score);
        let cents = vec![6900.0f32; 6];
        let voiced = vec![1u8; 6];
        let f0 = VocalF0 { cents: &cents, voiced: &voiced };
        let hz = build_note_hz(&arr, &evts, 0, Some(&f0));
        assert_eq!(hz.len(), 6);
        assert!(hz.iter().all(|&h| (h - 440.0).abs() < 0.5), "6900¢ → 440Hz (ignores note 60), got {:?}", hz);
        // transpose +12st adds 1200¢ → A5 = 880 Hz.
        let hz12 = build_note_hz(&arr, &evts, 12, Some(&f0));
        assert!(hz12.iter().all(|&h| (h - 880.0).abs() < 1.0), "6900¢+12st → 880, got {:?}", hz12);
    }

    #[test]
    fn build_note_hz_option_a_rest_and_unvoiced() {
        // note / rest / note — the rest group is silent, and cv↔DAW frames align (uncapped rest). Each note
        // is ≥ VOWEL_MIN_FRAMES so the M3 borrow-time leaves the boundaries clean (a separate test covers
        // the short-note borrow); [0..5]=note0, [5..10]=rest, [10..15]=note2.
        let score = [("あ", 60, 5), ("R", 0, 5), ("い", 62, 5)];
        let evts = ja_evts(&score);
        let arr = daw_ja(&score);
        assert_eq!(arr.phone_dur.iter().sum::<i64>(), 15, "1+1+1 phones, 5+5(uncapped SP)+5 frames");
        let mut cents = vec![0.0f32; 15];
        let mut voiced = vec![0u8; 15];
        for c in cents.iter_mut().take(5) {
            *c = 6000.0;
        }
        for v in voiced.iter_mut().take(5) {
            *v = 1;
        }
        for c in cents.iter_mut().skip(10) {
            *c = 6200.0;
        }
        for v in voiced.iter_mut().skip(10) {
            *v = 1;
        }
        let f0 = VocalF0 { cents: &cents, voiced: &voiced };
        let hz = build_note_hz(&arr, &evts, 0, Some(&f0));
        assert_eq!(hz.len(), 15);
        assert!(hz[0..5].iter().all(|&h| h > 200.0), "note0 voiced, got {:?}", &hz[0..5]);
        assert!(hz[5..10].iter().all(|&h| h == 0.0), "rest group → silent, got {:?}", &hz[5..10]);
        assert!(hz[10..15].iter().all(|&h| h > 200.0), "note2 voiced, got {:?}", &hz[10..15]);
    }

    // ── ② M-defer: loudness/formant envelope alignment (build_note_param) + gain/formant application ──
    #[test]
    fn build_note_param_aligns_via_group_remap_and_defaults() {
        // note / rest / note, uncapped rest → 15 cv == 15 DAW frames (1:1 here). A @50fps env samples through
        // the SAME group remap as f0; EVERY group is sampled (rests too, unlike f0); empty env → default.
        let score = [("あ", 60, 5), ("R", 0, 5), ("い", 62, 5)];
        let evts = ja_evts(&score);
        let arr = daw_ja(&score);
        let env: Vec<f32> = (0..15).map(|i| i as f32).collect(); // one value per DAW frame
        let out = build_note_param(&arr, &evts, &env, 1.0);
        assert_eq!(out.len(), 15);
        assert!((out[0] - 0.0).abs() < 1.0, "frame 0 ≈ env[0], got {}", out[0]);
        assert!((out[14] - 14.0).abs() < 1.0, "last frame ≈ env[14], got {}", out[14]);
        assert!(out[7] > 4.0 && out[7] < 10.0, "rest group IS sampled (continuity), got {}", out[7]);
        let flat = build_note_param(&arr, &evts, &[], 0.5);
        assert!(flat.iter().all(|&v| v == 0.5), "empty env → default everywhere (the flat/no-lane path)");
    }

    #[test]
    fn apply_gain_env_scales_by_cv_frame() {
        // 4 samples, 2-frame gain [1,3] → first half ×1, second half ×3 (uniform sample→cv map).
        let mut audio = vec![1.0f32, 1.0, 1.0, 1.0];
        apply_gain_env(&mut audio, &[1.0, 3.0]);
        assert_eq!(audio, vec![1.0, 1.0, 3.0, 3.0]);
        let mut a2 = vec![0.5f32, 0.5];
        apply_gain_env(&mut a2, &[]); // empty → untouched
        assert_eq!(a2, vec![0.5, 0.5]);
    }

    #[test]
    fn apply_formant_env_empty_is_identity() {
        let audio = vec![0.1f32; 5000];
        assert_eq!(apply_formant_env(audio.clone(), &[]), audio, "empty formant env → unchanged (no warp)");
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

    // M3 short-note handling: a sub-min_frames chunk is PADDED (repeat the last frame) so net_g accepts
    // the shape, then the pad is trimmed off the decoded wav — NOT errored (the old hard-error is gone).
    #[test]
    fn pad_sovits_feed_short_chunk() {
        let mut feed = SovitsFeed {
            cv: Array2::from_shape_fn((3, 4), |(i, _)| i as f32),
            f0: vec![100.0, 200.0, 300.0],
            uv: vec![0.0, 0.0, 0.0],
            t_tgt: 3,
        };
        let orig = pad_sovits_feed(&mut feed, 6);
        assert_eq!(orig, 3, "returns the pre-pad frame count (for the trim)");
        assert_eq!((feed.t_tgt, feed.cv.nrows(), feed.f0.len(), feed.uv.len()), (6, 6, 6, 6));
        assert_eq!(feed.f0[5], 300.0, "padded frames repeat the last real f0");
        assert_eq!(feed.cv[[5, 0]], 2.0, "padded rows repeat the last real cv row");
        // already ≥ min → untouched
        let mut ok = SovitsFeed { cv: Array2::zeros((8, 4)), f0: vec![0.0; 8], uv: vec![0.0; 8], t_tgt: 8 };
        assert_eq!(pad_sovits_feed(&mut ok, 6), 8);
        assert_eq!(ok.t_tgt, 8);
    }

    #[test]
    fn rvc_feed_100_pads_short_chunk() {
        // T50=3 → real_100=6 < min=12 → pad50=6 → cv 6 rows, 12 pitch frames.
        let cv = Array2::from_shape_fn((3, 4), |(i, _)| i as f32);
        let (cv_p, pitch, pitchf, real_t) = rvc_feed_100(cv, &[110.0, 220.0, 330.0], 12);
        assert_eq!(real_t, 6, "pre-pad 100fps length = 2·T50");
        assert_eq!((cv_p.nrows(), pitch.len(), pitchf.len()), (6, 12, 12));
        assert_eq!((pitchf[0], pitchf[1]), (110.0, 110.0), "each note_hz frame repeated 2×");
        assert_eq!(pitchf[11], 330.0, "padded frames repeat the last note_hz");
        assert_eq!(cv_p[[5, 0]], 2.0, "padded cv rows repeat the last real row");
        // already ≥ min → not padded
        let (cv2, _, pf, rt) = rvc_feed_100(Array2::zeros((10, 4)), &vec![100.0; 10], 12);
        assert_eq!((rt, cv2.nrows(), pf.len()), (20, 10, 20));
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

        // Item-1: the score render now drives the SHARED quality path (decode_features / vc_decode), so
        // the audition builds REAL SovitsModel/RvcModel (contentvec/rmvpe/mel loaded; diffusion/cluster off
        // for a clean plain-path demo). ContentVec/RMVPE are unused by the score decode tail but the model
        // struct requires them (auto-f0 off → f0.onnx never loaded → the DAW/noteonly f0 is preserved).
        let cv256 = engine.load_model_with(&aux.join("contentvec_256l9.onnx"), false).unwrap();
        let cv768 = engine.load_model_with(&aux.join("contentvec_768l12.onnx"), false).unwrap();
        let rmvpe = engine.load_model_with(&aux.join("rmvpe_e2e.onnx"), false).unwrap();
        let rmvpe_mel: Array2<f32> = ndarray_npy::read_npy(&aux.join("rmvpe_mel_filters.npy")).unwrap();
        let sopts = SovitsOptions { seed: 0, noise_scale: 0.4, ..Default::default() };
        let ropts = RvcOptions { seed: 0, index_ratio: 0.0, protect: 0.5, ..Default::default() };
        fn sov_model<'a>(
            engine: &'a OnnxEngine, voice: &'a str, cv: &'a str, rmvpe: &'a str, mel: &'a Array2<f32>,
            dim: usize, vol: bool,
        ) -> SovitsModel<'a> {
            SovitsModel {
                engine, voice_session: voice, contentvec_session: cv, rmvpe_session: rmvpe,
                mel_filters: mel, cluster: None, diffusion: None, vocoder: None,
                f0_predictor_session: None, sample_rate: 44100, hop_size: 512, features_dim: dim,
                vol_embedding: vol, spk_mix: None, unit_interpolate_mode: "left".into(),
                noise_channels: 192, min_frames: 6,
            }
        }

        // akiko 4.0 / 256 (vol-free — cleanest audible on the 256 path)
        let akiko = engine.load_model_with(&sov.join("akiko_320000.onnx"), false).unwrap();
        let am = sov_model(&engine, &akiko, &cv256, &rmvpe, &rmvpe_mel, 256, false);
        save("p2_akiko256_main", &render_score_sovits(&am, &s2cv256, &ja_evts(pr::SCORE), 256, 49, &NoDicts, &sopts, 0.0, 0, 0, None, None, None, &no_cancel, &no_prog).unwrap());
        save("p2_akiko256_demo_legato", &render_score_sovits(&am, &s2cv256, &ja_evts(LEGATO), 256, 49, &NoDicts, &sopts, 0.0, 0, 0, None, None, None, &no_cancel, &no_prog).unwrap());
        save("p2_akiko256_demo_rest", &render_score_sovits(&am, &s2cv256, &ja_evts(REST), 256, 49, &NoDicts, &sopts, 0.0, 0, 0, None, None, None, &no_cancel, &no_prog).unwrap());
        save("p2_akiko256_demo_sustain_same", &render_score_sovits(&am, &s2cv256, &ja_evts(SUSTAIN), 256, 49, &NoDicts, &sopts, 0.0, 0, 0, None, None, None, &no_cancel, &no_prog).unwrap());
        save("p2_akiko256_demo_reartic_same", &render_score_sovits(&am, &s2cv256, &ja_evts(REARTIC), 256, 49, &NoDicts, &sopts, 0.0, 0, 0, None, None, None, &no_cancel, &no_prog).unwrap());

        // 东雪莲 4.1 / 768 (SAME voice as the Python reference; vol_embedding → flat placeholder vol)
        let dx = engine.load_model_with(&sov.join("Sovits4.1东雪莲主模型.onnx"), false).unwrap();
        let dm = sov_model(&engine, &dx, &cv768, &rmvpe, &rmvpe_mel, 768, true);
        save("p2_dongxuelian768_main", &render_score_sovits(&dm, &s2cv768, &ja_evts(pr::SCORE), 768, 49, &NoDicts, &sopts, 0.1, 0, 0, None, None, None, &no_cancel, &no_prog).unwrap());

        // RVC v2 lengv2 / 768 (100 fps grid; no Python A/B reference — audible + glue self-consistency)
        let leng = engine.load_model_with(&rvcd.join("lengv2.3.onnx"), false).unwrap();
        let rm = RvcModel {
            engine: &engine, voice_session: &leng, contentvec_session: &cv768, rmvpe_session: &rmvpe,
            mel_filters: &rmvpe_mel, index: None, sample_rate: 48000, features_dim: 768, spk_mix: None,
            noise_channels: 192, min_frames: 12,
        };
        save("p2_rvc_lengv2_main", &render_score_rvc(&rm, &s2cv768, &ja_evts(pr::SCORE), 768, 49, &NoDicts, &ropts, 0, 0, None, None, None, &no_cancel, &no_prog).unwrap());

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

    // ── STEP 0 (Item-1) COVER DECODE SMOKE (all branches) ────────────────────────────────────────
    //    Runs the REAL cover pipeline (sovits/rvc run_pipeline) end-to-end on a fixed synthetic input,
    //    exercising EVERY branch of the extracted decode tail: plain VITS · vol_embedding · auto-f0 ·
    //    cluster blend (apply_cluster_blend) · NSF enhancer · shallow diffusion (±second_encoding) ·
    //    RVC plain · RVC index+protect. Asserts each branch still runs and yields non-silent audio of
    //    the (deterministic) frame-derived length — a smoke gate that the decode_features / vc_decode
    //    extraction did not break the pipeline shape.
    //    ⚠ The AUDIO ITSELF is NOT bit-reproducible run-to-run: the net_g ONNX graphs carry
    //    RandomNormalLike/RandomUniform nodes (VITS flow z-sampling) with NO seed attribute, so ORT
    //    draws fresh randomness each run (score2svc.rs's own note: "validated by ear, not bit-parity";
    //    empirically confirmed here — two identical runs differ). The extraction's byte fidelity is
    //    therefore proven at the SOURCE level (the moved fn bodies are character-identical to the
    //    originals — see scratchpad/verbatim_check.py) + the deterministic feed-builder gate
    //    (score2svc_glue_parity_cpu) + ear. Needs the voice models + dev ORT dll — hence #[ignore]:
    //      cargo test --lib inference::score2svc::tests::cover_decode_smoke -- --ignored --nocapture
    #[test]
    #[ignore]
    fn cover_decode_smoke() {
        use super::super::engine::DeviceConfig;
        use super::super::{diffusion, rvc, sovits, RvcOptions, SovitsOptions};
        use crate::audio::AudioBuffer;
        use ndarray::Array2;
        use std::path::Path;
        use std::sync::Arc;

        let read_npy2 = |p: &Path| -> Array2<f32> { ndarray_npy::read_npy(p).unwrap() };

        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let dll = root.join("../runtime/ort/onnxruntime.dll");
        assert!(dll.exists(), "ORT dll missing at {}", dll.display());
        if let Ok(b) = ort::init_from(&dll) {
            let _ = b.commit();
        }
        let engine = OnnxEngine::new();
        engine.set_device(DeviceConfig::Cpu); // deterministic + matches the parity gate

        let aux = root.join("../data/models/aux");
        let sov = root.join("../data/models/sovits");
        let rvcd = root.join("../data/models/rvc");

        // shared aux sessions + filterbanks
        let cv768 = engine.load_model_with(&aux.join("contentvec_768l12.onnx"), false).unwrap();
        let cv256 = engine.load_model_with(&aux.join("contentvec_256l9.onnx"), false).unwrap();
        let rmvpe = engine.load_model_with(&aux.join("rmvpe_e2e.onnx"), false).unwrap();
        let rmvpe_mel = read_npy2(&aux.join("rmvpe_mel_filters.npy"));
        let nsf_mel = Arc::new(read_npy2(&aux.join("nsf_hifigan_mel.npy")));
        let nsf_sid = engine.load_model_with(&aux.join("nsf_hifigan.onnx"), false).unwrap();

        // voice sessions
        let akiko = engine.load_model_with(&sov.join("akiko_320000.onnx"), false).unwrap();
        let akiko_f0 = engine.load_model_with(&sov.join("akiko_320000.f0.onnx"), false).unwrap();
        let dx = engine.load_model_with(&sov.join("Sovits4.1东雪莲主模型.onnx"), false).unwrap();
        let dxdiff = sov.join("Sovits4.1东雪莲主模型.diffusion");
        let dx_enc = engine.load_model_with(&dxdiff.join("encoder.onnx"), false).unwrap();
        let dx_den = engine.load_model_with(&dxdiff.join("denoiser.onnx"), false).unwrap();
        let leng = engine.load_model_with(&rvcd.join("lengv2.3.onnx"), false).unwrap();
        // かざね 4.0/256 carries a feature-retrieval index → the cluster branch (apply_cluster_blend).
        // 东雪莲 has no .cluster dir, so a separate 256 voice hosts the cluster case.
        let kazane = engine.load_model_with(&sov.join("かざねsayo（测试）_best.onnx"), false).unwrap();
        let kazane_cluster = sovits::ClusterAsset::FeatureIndex(super::super::features::KnnIndex::new(
            read_npy2(&sov.join("かざねsayo（测试）_best.cluster").join("0.index_vectors.npy")),
        ));
        // RVC leng retrieval index
        let leng_index = rvc::RvcIndex::load(&rvcd.join("lengv2.3.npy")).unwrap();

        // vocoder / diffusion runtime builders (fresh per case; cloned session ids share the graph)
        let mk_voc = || sovits::VocoderRuntime {
            session: nsf_sid.clone(),
            mel_filters: nsf_mel.clone(),
            cfg: super::super::nsf_hifigan::VocoderConfig { sample_rate: 44100, hop_size: 512, num_mels: 128 },
        };
        let mk_diff = || sovits::DiffusionRuntime {
            encoder_session: dx_enc.clone(),
            denoiser_session: dx_den.clone(),
            schedule: diffusion::DiffusionSchedule::linear(1000, 0.02, &[-12.0], &[2.0], 1000),
            method: diffusion::SamplerMethod::parse("dpm-solver++").unwrap(),
            n_hidden: 256,
            n_spk: 1,
            unit_interpolate_mode: "nearest".into(),
        };

        // fixed synthetic input: 1.2 s of a 220 Hz sine (clear pitch → RMVPE detects f0, voiced path runs).
        let sr_in = 44100u32;
        let ns = (sr_in as f64 * 1.2) as usize;
        let input: Vec<f32> = (0..ns)
            .map(|i| 0.3 * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / sr_in as f32).sin())
            .collect();
        let buf = AudioBuffer::new_mono(input, sr_in);
        let noop = |_: f32| {};
        let live = || false;

        // base SoVITS model (nested fn = explicit lifetimes, no transmute). A case tweaks fields after.
        fn sov_base<'a>(
            engine: &'a OnnxEngine,
            voice: &'a str,
            cv: &'a str,
            rmvpe: &'a str,
            mel: &'a Array2<f32>,
            dim: usize,
            vol: bool,
        ) -> sovits::SovitsModel<'a> {
            sovits::SovitsModel {
                engine,
                voice_session: voice,
                contentvec_session: cv,
                rmvpe_session: rmvpe,
                mel_filters: mel,
                cluster: None,
                diffusion: None,
                vocoder: None,
                f0_predictor_session: None,
                sample_rate: 44100,
                hop_size: 512,
                features_dim: dim,
                vol_embedding: vol,
                spk_mix: None,
                unit_interpolate_mode: "left".into(),
                noise_channels: 192,
                min_frames: 6,
            }
        }
        let dopts = |noise: f32| SovitsOptions { seed: 0, noise_scale: noise, ..Default::default() };
        // each case must run end-to-end and produce non-silent audio (peak > 1e-3); returns the sample
        // length (deterministic — set by the frame count, unaffected by the graph's internal randomness).
        let run_sov = |m: &sovits::SovitsModel, o: &SovitsOptions| -> u64 {
            let a = sovits::run_pipeline(m, &buf, o, None, &noop, &live).unwrap().audio;
            let peak = a.iter().fold(0.0f32, |p, &v| p.max(v.abs()));
            assert!(peak > 1e-3, "cover render is silent (peak {peak})");
            a.len() as u64
        };
        let akiko256 = || sov_base(&engine, &akiko, &cv256, &rmvpe, &rmvpe_mel, 256, false);
        let dx768 = || sov_base(&engine, &dx, &cv768, &rmvpe, &rmvpe_mel, 768, true);

        let mut results: Vec<(&str, u64)> = Vec::new();

        // 1) akiko 4.0/256 plain VITS (sid path, vits_out passthrough)
        results.push(("sov_akiko256_plain", run_sov(&akiko256(), &dopts(0.4))));
        // 2) 东雪莲 4.1/768 plain (vol_embedding tensor fed)
        results.push(("sov_dx768_plain", run_sov(&dx768(), &dopts(0.4))));
        // 3) akiko + auto-f0 (f0_predictor block REPLACES f0)
        {
            let mut m = akiko256();
            m.f0_predictor_session = Some(akiko_f0.clone());
            results.push(("sov_akiko256_autof0", run_sov(&m, &dopts(0.4))));
        }
        // 4) かざね 4.0/256 + cluster/retrieval blend (apply_cluster_blend, FeatureIndex)
        {
            let mut m = sov_base(&engine, &kazane, &cv256, &rmvpe, &rmvpe_mel, 256, false);
            m.cluster = Some(&kazane_cluster);
            let o = SovitsOptions { seed: 0, noise_scale: 0.4, cluster_ratio: 0.5, ..Default::default() };
            results.push(("sov_kazane256_cluster", run_sov(&m, &o)));
        }
        // 5) akiko + NSF enhancer (plain path enhancer branch)
        {
            let mut m = akiko256();
            m.vocoder = Some(mk_voc());
            let o = SovitsOptions { seed: 0, noise_scale: 0.4, nsf_enhance: true, enhancer_adaptive_key: 0, ..Default::default() };
            results.push(("sov_akiko256_enhancer", run_sov(&m, &o)));
        }
        // 6) 东雪莲 + shallow diffusion (dpm-solver++), no second encoding
        {
            let mut m = dx768();
            m.diffusion = Some(mk_diff());
            m.vocoder = Some(mk_voc());
            let o = SovitsOptions { seed: 0, noise_scale: 0.4, shallow_diffusion: true, k_step: 100, diffusion_method: "dpm-solver++".into(), diffusion_speedup: 10, ..Default::default() };
            results.push(("sov_dx768_diffusion", run_sov(&m, &o)));
        }
        // 6b) 东雪莲 + shallow diffusion + second_encoding (re-extract ContentVec from VITS out)
        {
            let mut m = dx768();
            m.diffusion = Some(mk_diff());
            m.vocoder = Some(mk_voc());
            let o = SovitsOptions { seed: 0, noise_scale: 0.4, shallow_diffusion: true, second_encoding: true, k_step: 100, diffusion_method: "dpm-solver++".into(), diffusion_speedup: 10, ..Default::default() };
            results.push(("sov_dx768_diffusion_2enc", run_sov(&m, &o)));
        }

        // 7/8) RVC (100 fps grid). Plain (no retrieval, protect≥0.5) then index+protect (<0.5 → feats0 blend).
        fn rvc_base<'a>(
            engine: &'a OnnxEngine,
            voice: &'a str,
            cv: &'a str,
            rmvpe: &'a str,
            mel: &'a Array2<f32>,
            index: Option<&'a rvc::RvcIndex>,
        ) -> rvc::RvcModel<'a> {
            rvc::RvcModel {
                engine,
                voice_session: voice,
                contentvec_session: cv,
                rmvpe_session: rmvpe,
                mel_filters: mel,
                index,
                sample_rate: 48000,
                features_dim: 768,
                spk_mix: None,
                noise_channels: 192,
                min_frames: 12,
            }
        }
        let run_rvc = |m: &rvc::RvcModel, o: &RvcOptions| -> u64 {
            let a = rvc::run_pipeline(m, &buf, o, None, &noop, &live).unwrap().audio;
            let peak = a.iter().fold(0.0f32, |p, &v| p.max(v.abs()));
            assert!(peak > 1e-3, "cover render is silent (peak {peak})");
            a.len() as u64
        };
        {
            let o = RvcOptions { seed: 0, index_ratio: 0.0, protect: 0.5, ..Default::default() };
            results.push(("rvc_leng_plain", run_rvc(&rvc_base(&engine, &leng, &cv768, &rmvpe, &rmvpe_mel, None), &o)));
        }
        {
            let o = RvcOptions { seed: 0, index_ratio: 0.5, protect: 0.33, ..Default::default() };
            results.push(("rvc_leng_index_protect", run_rvc(&rvc_base(&engine, &leng, &cv768, &rmvpe, &rmvpe_mel, Some(&leng_index)), &o)));
        }

        eprintln!("\n[smoke] cover decode — all branches ran + non-silent (length in samples):");
        for (name, len) in &results {
            eprintln!("    {name:<28} len={len}");
            assert!(*len > 0, "{name}: empty render");
        }
        eprintln!("[smoke] ✓ all {} decode branches ran non-silent", results.len());
    }
}
