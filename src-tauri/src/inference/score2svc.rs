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
//! Then cv/f0/uv are resampled 50 fps → the SVC grid and fed to the exported net_g (SoVITS
//! `c/f0/uv/noise/sid[,vol]`; RVC `phone/phone_lengths/pitch/pitchf/sid/rnd`). S69 R0a: the SoVITS
//! feed is shaped by the SAME cover-path code the model was trained against — cv via the model's
//! `unit_interpolate_mode`, f0/uv via `sovits_f0_postprocess` (uv = (f0 > 0), **1 = voiced**, rests
//! gap-interpolated so f0 is never 0). RVC stays 2× nearest → 100 fps with raw 0-Hz rests (that IS
//! the cover convention there — RMVPE emits exact 0 on unvoiced and RVC takes no uv). S69 R0b adds
//! voiceless-phone frames → 0 Hz for BOTH backends (`zero_voiceless_frames`) and a phrase-ADSR vol
//! for vol_embedding models. (An R0b procedural f0 micro-texture was ear-vetoed and removed.)
//!
//! GROUND TRUTH / GATE: the net_g wav is NOT bit-exact vs Python (ONNX-vs-PyTorch + the S35 export moved
//! net_g's randn to an explicit seeded `noise` input), so the parity gate is on the deterministic net_g
//! INPUT tensors (`score2svc_ref.rs`, dumped by `_onnx_derisk/dump_score2svc_ref.py`): midi_frame /
//! note_hz @50fps + cv_rs @86fps bit-for-bit. ⚠ S69: f0_rs/uv_rs are NO LONGER pinned to that dump —
//! the dump reproduced `render_derisk`'s conventions (uv=(f0<30) i.e. INVERTED, raw 0-Hz rests), which
//! were a semantic CONTRACT BUG vs the trained net_g. The gate now anchors f0/uv to the official
//! cover-path shaping (`sovits_f0_postprocess`, itself pinned to original so-vits by gen_refs.py
//! vectors) + cross-checks the old dump on frames where the conventions overlap (voiced frames).
//! Audible wav + the §3.4 legato-vs-SP behavior are confirmed by ear (Tier-2 render tests).

use ndarray::Array2;

#[cfg(test)]
use super::engine::OnnxEngine; // only the #[ignore] gates name the engine type; the render fns use m.engine
use super::features::{repeat_expand_2d, torch_interp_nearest};
use super::g2p::{self, Lang, ScoreEvt};
use super::rvc::{f0_to_coarse, vc_decode, RvcModel};
use super::score2cv::{
    build_arrays_daw, chunk_at_sp, classify_lyric, is_voiceless_phone, run_score2cv, Chunk,
    LyricClass, ScoreArrays,
};
use super::score2cv_tables as tbl;
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

// ─── S69 R0b: cover-parity f0 shaping + phrase dynamics (自己唱 texture layer) ────────────────────

/// R0b①: zero f0 on frames whose phone is voiceless (obstruents + JA devoiced vowels, see
/// `is_voiceless_phone`) — in the cover path RMVPE emits exact 0 there, so the trained contract is
/// "voiceless frames carry no pitch". Downstream this makes SoVITS mark them uv=0 (+ gap-interp f0)
/// and RVC's protect blend fire (pitchf=0). Until S69 the score path sang straight through /s/ /k/
/// with full voicing — the prime suspect for the community "清浊不分" (k→g-ish) reports.
fn zero_voiceless_frames(note_hz: &mut [f32], arr: &ScoreArrays) {
    let n = note_hz.len();
    let mut cursor = 0usize;
    for (i, &d) in arr.phone_dur.iter().enumerate() {
        let d = d.max(0) as usize;
        if is_voiceless_phone(arr.phon[i]) {
            for f in &mut note_hz[cursor.min(n)..(cursor + d).min(n)] {
                *f = 0.0;
            }
        }
        cursor += d;
    }
}

// R0b② (procedural f0 micro-texture) LIVED HERE and was REMOVED same-day by user ear verdict
// (commit c0c4f0d → reverted): two seeded LPF-noise layers on voiced frames sounded like
// MECHANICAL wobble (buzz stayed, plus faint glide artifacts around silences via the textured
// gap-interp endpoints). Lesson locked into the S69 research memory: real micro-motion is
// STRUCTURED (breath/effort/vibrato-coupled), not filtered white noise — it must come from
// real-audio-derived condition curves (variance/f0) or a learned predictor, never from
// procedural randomness. Do NOT re-add a "cheap jitter" here.

/// R0b③ phrase-dynamics constants (v1 ear-tuning). Attack/release live only at PHRASE edges
/// (after/before a rest or breath) — consecutive notes inside a phrase stay legato-flat: real
/// phrasing dips at breaths, not at every note boundary, and か+ー sustains are ONE note group so
/// they can never re-swell mid-hold.
const VOL_ATTACK_FRAMES: usize = 4; // 80 ms @50fps
const VOL_RELEASE_FRAMES: usize = 5; // 100 ms
const VOL_EDGE_LEVEL: f32 = 0.55;
const VOL_REST_LEVEL: f32 = 0.35;

/// R0b③: per-cv-frame vol multiplier (unity = nominal) for vol_embedding models — replaces the
/// "perfectly flat dynamics" story the constant placeholder told net_g (no real Volume_Extractor
/// stream is constant). Built on the note groups' CV spans (short-note-inflation safe, same remap
/// family as f0/lanes); the caller multiplies with `flat_vol` and the loudness lane. POST-DECODE
/// gain (the no-vol-port 4.0/RVC path) is deliberately NOT shaped by this — the decoder's output
/// already carries its own dynamics; shaping there would double-apply.
fn build_vol_env(arr: &ScoreArrays, score: &[ScoreEvt]) -> Vec<f32> {
    let t_total: usize = arr.phone_dur.iter().map(|&d| d.max(0) as usize).sum();
    let g = compute_note_groups(arr, score);
    let mut env = vec![1.0f32; t_total];
    for gi in 0..g.ng {
        let (s, c) = (g.cv_start[gi], g.cv_count[gi]);
        let e = (s + c).min(t_total);
        if s >= e {
            continue;
        }
        if g.group_pitch[gi] <= 0 {
            for v in &mut env[s..e] {
                *v = VOL_REST_LEVEL;
            }
            continue;
        }
        let phrase_start = gi == 0 || g.group_pitch[gi - 1] <= 0;
        let phrase_end = gi + 1 == g.ng || g.group_pitch[gi + 1] <= 0;
        let len = e - s;
        if phrase_start {
            let a = VOL_ATTACK_FRAMES.min(len / 2).max(1);
            for k in 0..a {
                let t = (k + 1) as f32 / (a + 1) as f32;
                env[s + k] = VOL_EDGE_LEVEL + (1.0 - VOL_EDGE_LEVEL) * t;
            }
        }
        if phrase_end {
            let r = VOL_RELEASE_FRAMES.min(len / 2).max(1);
            for k in 0..r {
                let t = (k + 1) as f32 / (r + 1) as f32;
                let idx = e - 1 - k;
                let v = VOL_EDGE_LEVEL + (1.0 - VOL_EDGE_LEVEL) * t;
                env[idx] = env[idx].min(v);
            }
        }
    }
    env
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
    /// f0 (Hz) per hop frame, length `t_tgt` — gap-interpolated like the cover path (never 0 unless
    /// the whole chunk is rests), NOT clamped (the cover path has no clamp either).
    pub f0: Vec<f32>,
    /// voiced mask (0.0/1.0) per hop frame, length `t_tgt`. **1.0 = voiced** — the official so-vits
    /// convention (`sovits_f0_postprocess`), same as the cover path feeds net_g's `uv` input.
    pub uv: Vec<f32>,
    pub t_tgt: usize,
}

/// Resample a chunk's `(cv @50fps, note_hz @50fps)` to the SoVITS hop grid the way the COVER path
/// shapes net_g inputs (S69 R0a — 自己唱 must feed the contract the model was TRAINED on):
///   * cv: `repeat_expand_2d` with the MODEL's `unit_interpolate_mode` (`expand_mode`; mirrors
///     sovits.rs's cover-path choice — only_diffusion is command-layer-disallowed for the score
///     path, so the main model's mode is always the right one here).
///   * f0/uv: `sovits_f0_postprocess` — the bit-exact `RMVPEF0Predictor.post_process` port the
///     cover path uses: nearest-resize to `t_tgt`, uv = (f0 > 0) (**1 = voiced**), then np.interp
///     across unvoiced gaps so the f0 stream is never 0 (training preprocessing does the same —
///     net_g never saw f0=0). Voicing still rides `build_note_hz`'s 0-Hz sentinel (Option-A mask →
///     0 Hz), and any nonzero pitch now counts as voiced (the old `<30 Hz` threshold is gone, so an
///     extreme down-transpose below MIDI 24 stays voiced). An all-rest chunk degenerates to
///     all-zero f0 + all-zero uv — identical to the cover path's all-zero short-circuit.
/// ⚠ HISTORY: until S69 this mirrored the research repo's `render_derisk.render_cv` — uv=(f0<30)
/// (INVERTED, 1-was-unvoiced), raw 0-Hz rests, a [0,1100] clamp, and hardcoded 'nearest' cv. Every
/// sung frame wore the net_g's "unvoiced" embedding and rests fed an out-of-distribution f0=0 —
/// a contract bug the Tier-1 gate couldn't see (it pinned tensors against the same-convention
/// Python dump: bit-exact, semantically wrong).
pub fn resample_to_sovits_grid(
    cv: &Array2<f32>,
    note_hz: &[f32],
    sr: u32,
    hop: usize,
    expand_mode: &str,
) -> Result<SovitsFeed> {
    debug_assert_eq!(cv.nrows(), note_hz.len(), "cv rows must equal note_hz length (both T50)");
    let t_tgt = sovits_grid_len(cv.nrows(), sr, hop);
    if t_tgt == 0 {
        return Err(UtaiError::Inference("SCORE2SVC_ZERO_FRAMES".into()));
    }
    let cv_rs = repeat_expand_2d(cv, t_tgt, expand_mode)?;
    let (f0_rs, uv_rs) = super::f0::sovits_f0_postprocess(note_hz, t_tgt, hop, sr);
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
    let mut note_hz_full = build_note_hz(&arr, score, transpose_eff, f0);
    // S69 R0b①: voiceless frames → 0 Hz (cover parity: RMVPE emits 0 there; SoVITS then gets
    // uv=0 + gap-interp). Also improves the apply_range_inverse PSOLA guide (unvoiced stays dry).
    // (R0b②'s procedural micro-texture was ear-vetoed and removed — see the note above
    // build_vol_env; micro-motion waits for the conditioned-S2CV line.)
    zero_voiceless_frames(&mut note_hz_full, &arr);
    // ② per-cv-frame loudness (multiplier, unity default) + formant (semitones, 0 default) envelopes,
    // aligned to cv via the SAME group remap as f0 (short-note-inflation safe). None/empty ⇒ flat = no-op.
    let loud_cv = loudness.map(|l| build_note_param(&arr, score, l, 1.0));
    let formant_cv = formant.map(|f| build_note_param(&arr, score, f, 0.0));
    // R0b③: phrase ADSR for vol_embedding models, once at the 50fps cv grid (transpose-independent:
    // note groups read note_to_phone/score, never arr.note_pitch).
    let vol_env_cv = if m.vol_embedding { Some(build_vol_env(&arr, score)) } else { None };
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
            return Err(UtaiError::Inference("CANCELLED".into()));
        }
        let cv = run_score2cv(m.engine, score2cv_session, chunk, dim, cv_speaker_id, chunk.lang_id)?;
        let note_hz = &note_hz_full[cv_cursor..(cv_cursor + chunk.t).min(note_hz_full.len())];
        // S69: cv expand uses the model's sidecar mode, same as the cover path (only_diffusion is
        // disallowed for the score path, so the diffusion-yaml override branch never applies).
        let mut feed =
            resample_to_sovits_grid(&cv, note_hz, m.sample_rate, m.hop_size, &m.unit_interpolate_mode)?;
        let real_t = pad_sovits_feed(&mut feed, m.min_frames); // M3: short trailing chunk
        apply_cluster_blend(&mut feed.cv, m.cluster, options.cluster_ratio); // SHARED retrieval blend
        // vol_embedding (SoVITS 4.1): per-frame loudness = flat_vol · phrase ADSR (R0b③) · loudness
        // lane, combined on the 50fps cv grid then one nearest resample to the hop grid (the same
        // resample the lane-only path always used). The pre-S69 constant placeholder told net_g
        // "perfectly flat dynamics" — a stream no real Volume_Extractor ever produces.
        let vol = vol_env_cv.as_ref().map(|env| {
            let end = (cv_cursor + chunk.t).min(env.len());
            let combined: Vec<f32> = (cv_cursor..end)
                .map(|i| {
                    let lane = loud_cv.as_ref().and_then(|lc| lc.get(i).copied()).unwrap_or(1.0);
                    env[i] * lane
                })
                .collect();
            torch_interp_nearest(&combined, feed.t_tgt).into_iter().map(|v| flat_vol * v).collect()
        });
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
        // S73e: 真休止(SP)窗零化——cover slicer 铺零的输出域等价物(AP 呼吸不动;详见 gate 头注)
        let sp_wins = chunk_sp_windows(chunk, wav.len());
        apply_rest_gate(&mut wav, &sp_wins, rest_gate_fade_samples(m.sample_rate));
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
    let mut note_hz_full = build_note_hz(&arr, score, transpose_eff, f0);
    // S69 R0b① (same as the SoVITS render): voiceless frames → pitchf 0 — RVC's cover convention
    // (RMVPE zeros) — which finally lets the protect blend fire on consonants. (② removed, ear-veto.)
    zero_voiceless_frames(&mut note_hz_full, &arr);
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
            return Err(UtaiError::Inference("CANCELLED".into()));
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
        // S73e: 真休止(SP)窗零化(RVC 症状最轻但同样受益;AP 呼吸不动)
        let sp_wins = chunk_sp_windows(chunk, wav.len());
        apply_rest_gate(&mut wav, &sp_wins, rest_gate_fade_samples(m.sample_rate));
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

// ─── S73e 休止零化(rest gate)────────────────────────────────────────────────────────────────────
//
// cover slicer「静音段不进模型/输出直接铺零」(sovits.rs:256-261)的 score 域等价物,输出域最小刀。
// 根因链(S73e 三路审计):cover 官方口径下 net_g 从不渲染长静音;score 路径整段休止喂入且无
// gate;v2 主图无 uv/vol(无声只剩 cv 一道闸门),SP-token cv(=训练语料休止帧的房间底噪 cv,
// 非数字零)压不住插值 f0 驱动的谐波源 → 长空拍定音高电流(chunk 尾=f0 边缘外推常数,渐强=
// 超长 SP 串 OOD 生成漂移)/短空拍滑音底噪(chunk 中部 np.interp 滑坡)。4.0/4.1 有 uv=0 兜底
// 故轻,RVC 裸 0+coarse=1+protect 最轻——实测排序吻合。
// 本 gate 按【谱面真值】对 SP(真休止)窗口内的输出乘衰减包络:窗缘与音符样本连续(keep=1),
// REST_GATE_FADE_MS 内渐落至 0(保住 net_g 的自然 release 尾),窗心全零;短窗成浅谷不至 0。
// ★AP(呼吸)绝不 gate(audible intake);清辅音帧(f0=0 但在音符内)不在 SP 窗,不受影响。
// feed 级 slicer 对齐(长休止内部不渲染+边距+chunk 整改=根治 OOD 与算力)=后续结构优化轮。
const REST_GATE_FADE_MS: f32 = 40.0;

fn rest_gate_fade_samples(sample_rate: u32) -> usize {
    ((REST_GATE_FADE_MS / 1000.0) * sample_rate as f32).round().max(1.0) as usize
}

/// 该 chunk 内 SP 音素的输出样本窗(50fps 帧域 → 按比例映射到 chunk 输出;chunk 内网格均匀,
/// 比例映射即精确,逐 chunk 应用避免全局取整漂移)。
fn chunk_sp_windows(chunk: &Chunk, out_len: usize) -> Vec<(usize, usize)> {
    let t = chunk.t.max(1);
    let mut wins = Vec::new();
    let mut cursor: i64 = 0;
    for (i, &pid) in chunk.phonemes.iter().enumerate() {
        let d = chunk.phone_dur.get(i).copied().unwrap_or(0).max(0);
        if pid == tbl::SP_ID && d > 0 {
            let s = (cursor as f64 / t as f64 * out_len as f64).round() as usize;
            let e = (((cursor + d) as f64) / t as f64 * out_len as f64).round() as usize;
            let e = e.min(out_len);
            if e > s {
                wins.push((s, e));
            }
        }
        cursor += d;
    }
    wins
}

/// 窗内逐样本乘 keep = max(0, 1 − 距窗缘样本数/fade):窗缘 keep=1(与音符样本连续),
/// fade 内渐落,深处全零;窗宽 < 2·fade 时为浅谷(不到 0,自然)。
fn apply_rest_gate(audio: &mut [f32], windows: &[(usize, usize)], fade: usize) {
    let fade = fade.max(1) as f32;
    for &(s, e) in windows {
        let e = e.min(audio.len());
        for i in s..e {
            let edge = (i - s).min(e - 1 - i) as f32;
            let keep = (1.0 - edge / fade).max(0.0);
            audio[i] *= keep;
        }
    }
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

// E1 交叉判别实验 harness(S70,diagnostic #[ignore])— 挂为子模块以复用本文件私有整形函数
// (zero_voiceless_frames / build_vol_env / pad_sovits_feed / seam_fade / peak_normalize),
// 生产代码零触碰。详见 score2svc_e1.rs 头注。
#[cfg(test)]
#[path = "score2svc_e1.rs"]
mod e1_tests;

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
    fn rest_gate_envelope_shape() {
        // S73e:长窗——窗缘 keep=1 与音符样本连续、fade 内线性渐落、窗心全零;窗外不动
        let mut a = vec![1.0f32; 1000];
        apply_rest_gate(&mut a, &[(100, 900)], 100);
        assert_eq!(a[99], 1.0); // 窗外
        assert_eq!(a[100], 1.0); // 窗缘连续
        assert!((a[150] - 0.5).abs() < 1e-6); // fade 半程
        assert_eq!(a[500], 0.0); // 窗心全零
        assert_eq!(a[899], 1.0); // 右缘连续
        assert_eq!(a[900], 1.0); // 窗外
        // 短窗(< 2·fade)= 浅谷,不至 0(自然过渡)
        let mut b = vec![1.0f32; 300];
        apply_rest_gate(&mut b, &[(100, 160)], 100);
        let mid = b[130];
        assert!(mid > 0.0 && mid < 1.0, "短窗应为浅谷,得 {mid}");
    }

    #[test]
    fn sp_windows_gate_rests_only_never_breath() {
        // SP(真休止)按 50fps 帧比例映射到输出样本;AP(呼吸)绝不 gate
        let chunk = Chunk {
            start: 0,
            end: 4,
            phonemes: vec![10, tbl::SP_ID, tbl::AP_ID, 11],
            note_pitch: vec![60, 0, 0, 62],
            phone_dur: vec![50, 100, 30, 20],
            note_dur: vec![50, 100, 30, 20],
            note_to_phone: vec![0, 1, 2, 3],
            t: 200,
            lang_id: 2,
            hard_seam: false,
        };
        let wins = chunk_sp_windows(&chunk, 2000); // 10 samples/50fps-frame
        assert_eq!(wins, vec![(500, 1500)]);
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

    // ── S69 R0b: f0 shaping + phrase vol ──

    #[test]
    fn zero_voiceless_frames_zeroes_k_keeps_g() {
        // か = [k(4fr), a(36fr)]: the voiceless k frames go 0 Hz, the vowel keeps pitch.
        let score = [("か", 69, 40)];
        let arr = daw_ja(&score);
        let mut hz = build_note_hz(&arr, &ja_evts(&score), 0, None);
        assert!(hz.iter().all(|&h| h > 0.0), "pre: whole note voiced");
        zero_voiceless_frames(&mut hz, &arr);
        let k = arr.phone_dur[0] as usize;
        assert!(hz[..k].iter().all(|&h| h == 0.0), "k frames zeroed");
        assert!(hz[k..].iter().all(|&h| h > 0.0), "vowel frames keep pitch");
        // が = [ɡ, a] — voiced consonant stays pitched (清浊 distinction is the whole point).
        let score_g = [("が", 69, 40)];
        let arr_g = daw_ja(&score_g);
        let mut hz_g = build_note_hz(&arr_g, &ja_evts(&score_g), 0, None);
        zero_voiceless_frames(&mut hz_g, &arr_g);
        assert!(hz_g.iter().all(|&h| h > 0.0), "ɡ (voiced) untouched");
    }

    #[test]
    fn vol_env_phrase_shape_and_sustain_no_reswell() {
        // note(20) / rest(10) / note(20): attack+release at phrase edges, rest floor between.
        let score = [("あ", 60, 20), ("R", 0, 10), ("い", 62, 20)];
        let arr = daw_ja(&score);
        let env = build_vol_env(&arr, &ja_evts(&score));
        assert_eq!(env.len(), 50);
        assert!(env[0] < env[3] && env[3] < 1.0, "phrase attack rises: {} {}", env[0], env[3]);
        assert_eq!(env[10], 1.0, "mid-note sustain flat");
        assert!(env[19] < env[15], "phrase release falls into the rest");
        assert!(env[20..30].iter().all(|&v| v == VOL_REST_LEVEL), "rest floor");
        assert!(env[30] < 1.0 && env[38] == 1.0, "second phrase re-attacks then sustains");
        // か+ー same pitch = ONE note group → no re-swell at the sustain join.
        let sus = [("か", 60, 30), ("ー", 60, 30)];
        let arr_s = daw_ja(&sus);
        let env_s = build_vol_env(&arr_s, &ja_evts(&sus));
        assert_eq!(env_s.len(), 60);
        assert!(env_s[25..35].iter().all(|&v| v == 1.0), "sustain join stays flat (no ADSR re-trigger)");
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
            let path: PathBuf = root.join("../data/models").join(crate::models::AUX_DIR_NAME).join(model);
            assert!(path.exists(), "model missing: {}", path.display());
            let sid = engine.load_model_with(&path, false).unwrap();

            for (ci, chunk) in chunks.iter().enumerate() {
                let r = &SVC_REFS[ci];
                let cvr = if dim == 768 { &r.cv768_rs } else { &r.cv256_rs };
                let cv = run_score2cv(&engine, &sid, chunk, dim, 49, 2).unwrap();

                let midi = midi_frame_50(&chunk.note_pitch, &chunk.phone_dur);
                let note_hz = note_hz_50(&midi);
                // "nearest": the Python dump was rendered in nearest mode — keeping the cv_rs
                // reference anchored. Production passes the model sidecar's unit_interpolate_mode;
                // the 'left' variant is pinned by features.rs's own upstream (ast-exec) vectors.
                let feed = resample_to_sovits_grid(&cv, &note_hz, SOVITS_SR, SOVITS_HOP, "nearest").unwrap();

                // midi_frame: bit-exact (i64 length-regulation)
                assert_eq!(midi.as_slice(), r.midi_frame, "{} c{}: midi_frame", model, ci);
                // t_tgt: exact
                assert_eq!(feed.t_tgt, r.t_tgt, "{} c{}: t_tgt", model, ci);
                // note_hz @50fps: tight tolerance (transcendental pow, f64→f32; a real bug is Hz-scale)
                let nh_worst = worst_abs(&note_hz, r.note_hz);
                assert!(nh_worst <= 1e-2, "{} c{}: note_hz worst {:.3e} Hz > 1e-2", model, ci, nh_worst);
                // f0_rs/uv_rs @86fps (S69 R0a): the dump predates the cover-parity fix — its uv was
                // INVERTED (render_derisk uv=(f0<30)) and its rests raw 0 Hz, i.e. the dumped f0_rs/
                // uv_rs encode the CONTRACT BUG. Two-way anchor instead:
                //  1. expected tensors from `sovits_f0_postprocess` computed ON THE REFERENCE note_hz
                //     (never through the production path) — that port is pinned to ORIGINAL so-vits
                //     python by its own gen_refs.py vectors (f0.rs tests);
                //  2. cross-anchor vs the old dump where the conventions overlap: every frame the dump
                //     had f0>0 (voiced) must keep the SAME f0 (resize unchanged; this score sits far
                //     below the old 1100 Hz clamp) and read uv=1; every dumped-0 frame must read uv=0.
                let (f0_exp, uv_exp) =
                    super::super::f0::sovits_f0_postprocess(r.note_hz, r.t_tgt, SOVITS_HOP, SOVITS_SR);
                let f0_worst = worst_abs(&feed.f0, &f0_exp);
                assert!(f0_worst <= 1e-2, "{} c{}: f0_rs worst {:.3e} Hz > 1e-2", model, ci, f0_worst);
                assert_eq!(feed.uv.len(), r.uv_rs.len(), "{} c{}: uv len", model, ci);
                assert!(
                    feed.uv.iter().zip(&uv_exp).all(|(a, b)| a == b),
                    "{} c{}: uv_rs vs postprocess expectation", model, ci
                );
                for i in 0..r.t_tgt {
                    let dump_voiced = r.f0_rs[i] > 0.0;
                    assert_eq!(
                        feed.uv[i],
                        if dump_voiced { 1.0 } else { 0.0 },
                        "{} c{}: uv frame {} vs old-dump voicing", model, ci, i
                    );
                    if dump_voiced {
                        assert!(
                            (feed.f0[i] - r.f0_rs[i]).abs() <= 1e-2,
                            "{} c{}: voiced f0 frame {} drifted: {} vs dump {}",
                            model, ci, i, feed.f0[i], r.f0_rs[i]
                        );
                    } else {
                        // a rest frame must now be gap-interpolated non-zero — unless the whole
                        // chunk is rests (then postprocess degenerates to zeros, like cover).
                        assert!(
                            feed.f0[i] > 0.0 || f0_exp.iter().all(|&v| v == 0.0),
                            "{} c{}: rest frame {} still 0 Hz (gap interpolation missing)", model, ci, i
                        );
                    }
                }
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

                let voiced = feed.uv.iter().filter(|&&u| u > 0.5).count(); // S69: 1 = voiced
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

        let aux = root.join("../data/models").join(crate::models::AUX_DIR_NAME);
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
                vol_embedding: vol, phase_bins: None, f0d_cond_channels: None,
                feed_uv: true, spk_mix: None,
                // "nearest" = the pre-S69 hardcode → the ear A/B against archived baselines stays
                // single-variable (f0/uv semantics only). Production reads the voice sidecar's mode.
                unit_interpolate_mode: "nearest".into(),
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

        let aux = root.join("../data/models").join(crate::models::AUX_DIR_NAME);
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
                phase_bins: None,
                f0d_cond_channels: None,
                feed_uv: true,
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
