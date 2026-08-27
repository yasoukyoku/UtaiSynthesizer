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
    build_arrays_daw, chunk_at_sp, classify_lyric, is_voiceless_phone, run_score2cv,
    run_score2cv_vowel_clarity, Chunk, LyricClass, ScoreArrays,
};
use super::score2cv_tables as tbl;
use super::sovits::{apply_cluster_blend, decode_features, SovitsModel};
use super::{build_spk_mix_dense, RvcOptions, SovitsOptions, SynthesisResult};
use crate::{Result, UtaiError};
use utai_dsp::formant_warp;

/// ScoreToCV frame rate (score2cv sidecar `fps`). The per-phone `phone_dur` frames are 20 ms.
/// ⚠ `pub(crate)` since S151: `vocal_range::dead_only_plan` turns a triple's `frames` into
/// milliseconds and must not carry a second copy of this number (feedback_no_duplication_drift).
pub(crate) const CV_FPS: f64 = 50.0;

/// S147 B2 — margin (in 50 fps frames) around a rescue window when deciding which chunks a donor
/// must render.
///
/// ⛔ Not caution — measurement. The splice layer maps frames to samples with a **linear** ratio
/// (`spf` in `vocal_range::apply_dead_only_windows`), while real chunk boundaries land wherever
/// `sovits_grid_len`'s rounding puts them: measured divergence up to **458.8 samples (10.4 ms)**,
/// against a tightest real margin of **262 samples (5.9 ms)** in the current plan. Intersecting
/// with no margin therefore splices the window's own 10 ms cross-fade tail into digital zero.
/// Two frames = 40 ms covers both the divergence and the fade itself.
///
/// ⛔⛔ **S152 把它从 2 提到 29,而那是一条正确性修复,不是余量加保险。**
/// `vocal_range::join_rests` 会在拼接阶段(donor 早就渲完之后)把一条窗边**挪到休止的另一头**
/// —— 最远 `MERGE_BRIDGE_FRAMES(25) + 4 = 29` 帧。若那一段落在一个**没被渲的 chunk** 里,
/// 拼进去的就是铺零 = **一个新的洞**,而且形状与它要修的那个一模一样。
/// ⇒ 渲染侧的余量必须**覆盖拼接侧够得到的最远处**;这两个数字从此是一对,改一个必须改另一个。
/// ⚠ 代价可忽略:chunk 长约 11.6 s(整曲 291 s / 25 个),580 ms 的余量只在窗正好压在
/// chunk 边界上时才多渲一个。
pub(crate) const DONOR_WINDOW_MARGIN_FRAMES: i64 = 29;
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
            // S86: `g2p::is_silent_token`, NOT `classify_lyric` — the latter keeps the upstream
            // parity port's WIDER rest set (rest/sil/pau), which would key this grouping differently
            // from the cv-side grouping and shift every later group index.
            let npitch = if g2p::is_silent_token(evt.lyric, evt.phoneme_set) {
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

    // S83 conservation fast-path: the DAW assembler is frame-CONSERVING (Σ phone_dur == Σ triple frames
    // == the f0 grid), so cv frame i IS DAW frame i — sample identity. This is not just an optimization:
    // with onset pre-roll a note GROUP's cv span starts earlier than its triple span (borrowed frames),
    // and the group remap below would compress the vowel's samples ~2 frames late — re-delaying the very
    // pitch alignment the pre-roll fixes. The remap stays for non-conserving arrays only (capped rests /
    // legacy split_dur inflation in tests / E1 harness).
    if t_total == f0.cents.len() {
        // The slow path's "rest group → 0 Hz" guard is preserved by masking frames OWNED by SP/AP
        // (pitch-0) phones: the frontend's frame rounding can leave a rest triple's first frame
        // sampling voiced=1 just inside the note it follows (S83 review #2) — identity alignment
        // makes the phone-ownership walk exact, so gate on it, not on the voiced mask alone.
        let mut out = vec![0.0f32; t_total];
        let mut cursor = 0usize;
        for (i, &d) in arr.phone_dur.iter().enumerate() {
            let d = d.max(0) as usize;
            if !matches!(arr.phon[i], "SP" | "AP") && arr.note_pitch[i] > 0 {
                for f in cursor..(cursor + d).min(t_total) {
                    if f0.voiced.get(f).copied().unwrap_or(0) != 0 {
                        out[f] = cents_to_hz(f0.cents[f] as f64 + (transpose as f64) * 100.0);
                    }
                }
            }
            cursor += d;
        }
        return out;
    }
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
    // S83: same conservation fast-path as build_note_hz (identity when the grids already agree) —
    // the group remap would misplace lane samples around pre-rolled onsets.
    if t_total == env.len() {
        return env.to_vec();
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
///
/// S83 knife 5 refinement: zero only the MEASURED fraction of the window, CENTERED — full-window
/// zeroing was the R0b① over-correction. RMVPE actually zeroes 17-48% of a short-note voiceless
/// window (the track drags in from the previous vowel and pre-voices into the next); zeroing 100%
/// collapsed fast runs' voiced duty cycle into audible "briefly mute" syllables (さ/こ/け). The
/// per-token per-bucket fraction comes from the same generated priors table (training-measured);
/// the bucket keys on the phone's note-GROUP length (arr.note_dur — the training stat's exact
/// frame of reference). Devoiced vowels / unmapped tokens keep the full-window zero (fallback).
/// ⚙ 出厂默认 = true —— 把夹在两个浊帧之间的**孤立单帧 0** 填回去。
///
/// ## ⭐⭐⭐ 用户 2026-08-23 点名的那一族(「基频附近的竖线,几毫秒,像敲了一下」)
///
/// 用户给了东雪莲 +7 上的五个坐标(1:05.683 / 1:05.867 / 1:06.400 / 1:07.121 / 3:21.998),
/// **每一个都精确落在一个孤立单帧 `0` 上**,而那一段是 17 个连续的 80-100 ms 音
/// (密度 7.7 音/s,全曲均值 4.5)。
///
/// 机理链:`consonant_preroll` 把**下一个音**的清音辅音提前 ⇒ 它的音素窗落进**当前音的元音正中**;
/// 快音上那个窗只有 **2 帧**,于是 [`zero_voiceless_frames`] 的 `z = round(d × permille/1000)`
/// 取到 **1** ⇒ 一记**单帧硬置零**,谐波栈断 20 ms。
///
/// | 洞的位置 | 落在 | 前置的是 |
/// |---|---|---|
/// | 65.600 | 音[313]「り」(**浊音**)| 音[314]「さ」的 /s/ |
/// | **65.680** | 音[314]「さ」 | 音[315]「け」的 /k/ |
/// | **66.400** | 音[322]「な」(**浊音**)| 音[323]「こ」的 /k/ |
///
/// ## ✅ 归因(实测,`donor_pre` / `donor_post` / 成品三段对拍)
///
/// 那一帧在**基频附近**相对前后两帧的能量:
/// **donor 逆变换【前】−8.69 dB** · 逆变换后 −9.22(逆变换只加 **−0.54**)· 成品 −9.04(拼接 +0.18);
/// ⛔ 阴性对照(音内**没有** 0 帧的随机帧)**+0.16 dB**。
/// ⇒ **这个洞 94 % 是这条链挖的,PSOLA 与拼接几乎没份** —— 我在缝与岛边上找了一整晚,
/// 而那把 LPC 尺子的归因早就说了「**74.8 % 的检出不在我们的任何结构边界上**」。
///
/// ## ⛔ 为什么填回去比留着更接近「cover 车道对齐」
///
/// [`zero_voiceless_frames`] 的初衷(S69 R0b①)是与 cover 车道对齐:RMVPE 在真实音频上
/// 对清音帧输出 0。**但真实音频里的 /s/ 会给出一串 0(60-120 ms),不是一个孤立单帧** ——
/// 单帧洞是我们**帧量化 + 辅音前置**在快音上造出来的假象,它离 RMVPE 的样子**更远**。
/// ⇒ 只填**孤立单帧**,两帧及以上一个不动(那是真辅音的长度)。
///
/// ## 覆盖面(鹅妈妈 +7,唱音内部的 0 段)
///
/// **1 帧 × 141** · 2 帧 × 97 · 3 帧 × 14 · 4 帧 × 1 · 5 帧 × 1 ⇒ 这一刀只碰最左边那一列(**56 %**)。
/// ⭐ 密度相关性(用户观察的直接复现):1:05-1:08(7.7 音/s)**3.0 个/s** · 3:44-3:47(3.3 音/s)**0.0 个/s**。
///
/// ## ✅ 实测(鹅妈妈 +7 × 东雪莲,整曲 A/B,`UTAI_MG_FILL1` 两档)
///
/// f0 层面:唱音内孤立单帧 0 **153 → 4**(剩下的 4 个是 `anchor_voiced_phone_f0` 之后又冒出来的边界情形)。
/// 音频层面,那 **153 个洞位**在基频附近相对前后两帧的能量:
///
/// | | 洞位(153)| ⛔ 阴性对照(音内**无洞**的随机帧,610)|
/// |---|---|---|
/// | 关 | **−8.37 dB** | +0.15 |
/// | 开 | **−3.59 dB** | +0.16 |
/// | 改善 | **+4.78 dB**(131/153 变好)| **+0.01** |
///
/// ⚠ 如实记三件:⑴ **洞只是浅了一半,没消失** —— 音素内容本身仍是清音,模型照样把那一帧渲弱;
/// ⑵ 用户点名的五处改善偏小(+0.21…+1.63),而 **3:21.998 反而 −1.71**
///(单点读数受两次独立渲染的噪声影响,承重的是那 153 个的中位数与那条干净的对照);
/// ⑶ 全曲长时平均谱各档 **−0.19…−0.46 dB**,在两次渲染的台面噪声内。
/// ⛔ **这一刀没有耳朵背书**,承重之前该过一次耳判。
///
/// ⛔ 关掉:`UTAI_MG_FILL1=0`。改它要成对 bump `RANGE_ALGO_VERSION` 与 `audition_cache_tag`。
pub fn fill_isolated_unvoiced() -> bool {
    parse_fill1(std::env::var("UTAI_MG_FILL1").ok().as_deref())
}

fn parse_fill1(v: Option<&str>) -> bool {
    match v.map(str::trim) {
        Some("1" | "true" | "on" | "yes") => true,
        Some("0" | "false" | "off" | "no") => false,
        _ => FILL_ISOLATED_UV_DEFAULT,
    }
}

/// ⛔ Changing this changes the audio ⇒ pair-bump `RANGE_ALGO_VERSION` and `audition_cache_tag`.
const FILL_ISOLATED_UV_DEFAULT: bool = true;

/// ⚙ 出厂默认 = 1 —— 夹在浊音之间的无声洞,**最多填多长**(帧)。`UTAI_MG_FILL_MAX=<n>`。
///
/// # ⛔⛔ 判负(S159zzr 实测)—— 旋钮留着只为把下面那张洞长分布表留在原地
///
/// 放宽到 2 / 4 帧,用户点名的四处 lb3 咔哒**谷深几乎不动**
/// (1:07.053 −52.1 → −52.7 → −52.1 · 1:36.448 −17.6 → −16.3 → −15.4 ·
/// 3:33.193 −13.8 → −13.3 → **−15.5** · 3:33.749 −15.8 → −18.5 → −18.6)。
///
/// ⭐ **为什么没用(这条比读数值钱)**:[`zero_voiceless_frames`] 只把 **f0** 置零,
/// 而**音素序列里那个清音辅音还在** —— 模型照样把 /s/ 渲出来。
/// S159zp 那一刀之所以有效,是因为**单帧** f0 = 0 会让 NSF 激励打嗝;
/// 而 2-4 帧的洞里,**真正在响的是辅音本身**。
/// ⇒ 那四处的「咔哒」是 **`consonant_preroll`(用户可见选项,出厂 `true`)把下一个音的辅音
/// 提前放进上一个音正中间**的结果 —— 在 120-140 ms 的快音上它吃掉一大截,听起来就是个断口。
/// **那是设计意图(让元音落在拍点上),不是 DSP 伪影** ⇒ 要动只能动 preroll 的时序策略,
/// 而那有音乐上的后果(元音会偏离拍点)。⛔ **别再从 f0 那一侧修它。**
///
/// ## ⛔ 为什么它需要是个旋钮而不是一个常数
///
/// S159zp 定 1 的理由是「**两帧及以上是真辅音的长度**」。但 S159zzq 里用户点名的四处 lb3 咔哒
/// (**1:07.053 / 1:36.448 / 3:33.193 / 3:33.749**)转储出来是 **4 / 2 / 3 / 2 帧**的洞 ——
/// 全部**夹在浊音之间**、全部由 [`zero_voiceless_frames`] 打出、全部落在 120-140 ms 的快音上,
/// 而且三处的**下一个音都以清音辅音开头**(す / し / す)⇒ 正是 `consonant_preroll` 那条链。
///
/// ⭐ 洞长是 `z = round(d × permille/1000)` —— **音素自己时长的一个比例**,
/// 所以「2 帧 = 真辅音」并不自动成立:快音上它就是量化落点。
/// 实测(本曲 8 遍 donor 的 f0 轨,**夹在浊音之间**的洞按长度):
/// **1 帧 32 · 2 帧 832 · 3 帧 248 · 4 帧 184 · 5 帧 80 · 6 帧 64 · 7 帧 376 · ≥8 帧 616**。
/// ⇒ **2 帧那一桶是 1 帧的 26 倍** —— 这个悬殊本身就说明它是量化落点而不是辅音长度。
///
/// ⚠ 但「填到几帧算过头」**只有耳朵能裁**:填多了会把真辅音吃掉。⇒ 出厂保持 **1**(逐位不变)。
const FILL_MAX_FRAMES_DEFAULT: usize = 1;

fn fill_max_frames() -> usize {
    std::env::var("UTAI_MG_FILL_MAX")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|v| (1..=6).contains(v))
        .unwrap_or(FILL_MAX_FRAMES_DEFAULT)
}

/// 把 `note_hz` 里**长度 ≤ `max_len`**、两侧都是浊音的 0 段填成两侧的线性插值。
///
/// ⛔ `max_len == 1` 时与 [`fill_isolated_uv`] **逐位相同**(单帧的线性插值就是两侧均值)——
/// 那条判据仍然钉在 `fill_isolated_uv` 上,这里不许绕过它。
/// ⛔ 先收集再写:边扫边写会让连续的 0 被逐个「补」成浊音,越过 `max_len` 这条界。
fn fill_isolated_uv_max(note_hz: &mut [f32], max_len: usize) -> usize {
    let n = note_hz.len();
    if n < 3 || max_len == 0 {
        return 0;
    }
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 1usize;
    while i < n - 1 {
        if note_hz[i] == 0.0 {
            let mut j = i;
            while j + 1 < n - 1 && note_hz[j + 1] == 0.0 {
                j += 1;
            }
            if j - i + 1 <= max_len && note_hz[i - 1] > 0.0 && note_hz[j + 1] > 0.0 {
                runs.push((i, j));
            }
            i = j + 1;
        } else {
            i += 1;
        }
    }
    let filled: usize = runs.iter().map(|(a, b)| b - a + 1).sum();
    for (a, b) in runs {
        let (lo, hi) = (note_hz[a - 1], note_hz[b + 1]);
        let steps = (b - a + 2) as f32;
        for (t, f) in note_hz[a..=b].iter_mut().enumerate() {
            *f = lo + (hi - lo) * ((t + 1) as f32 / steps);
        }
    }
    filled
}


/// 把 `note_hz` 里**只有一帧**、而且两侧都是浊音的 0 填成两侧的均值。
///
/// ⛔ 只填**长度恰好 1** 的段。见 [`fill_isolated_unvoiced`] 的 doc:两帧及以上是真辅音的长度,
/// 动它就不再是「拿掉帧量化的假象」,而是「替模型决定这里该不该有辅音」。
fn fill_isolated_uv(note_hz: &mut [f32]) -> usize {
    let n = note_hz.len();
    if n < 3 {
        return 0;
    }
    // ⚠⚠ 先收集再写是**防御性**的,**不是**承重的 —— 写这段时我以为「边扫边写会让连续两个 0
    // 被逐个补掉」,而那条变异**真跑是绿的**:谓词要求**两侧都浊**,而两连零的第一个 0
    // 右邻还是 0 ⇒ 条件不成立 ⇒ 根本不会级联(三连、四连同理)。
    // ⇒ 留着它只是让「只填单帧」这个不变量在**将来改谓词时**仍然显然,别把它当判据背书过的东西。
    let mut hits: Vec<usize> = Vec::new();
    for i in 1..n - 1 {
        if note_hz[i] == 0.0 && note_hz[i - 1] > 0.0 && note_hz[i + 1] > 0.0 {
            hits.push(i);
        }
    }
    for &i in &hits {
        note_hz[i] = 0.5 * (note_hz[i - 1] + note_hz[i + 1]);
    }
    hits.len()
}

fn zero_voiceless_frames(note_hz: &mut [f32], arr: &ScoreArrays) {
    let n = note_hz.len();
    let mut cursor = 0usize;
    for (i, &d) in arr.phone_dur.iter().enumerate() {
        let d = d.max(0) as usize;
        if is_voiceless_phone(arr.phon[i]) {
            // S92h: the per-phone RUN language (ScoreArrays already carries it) selects the measured
            // zero-fraction — no call site changes, because every caller already passes `arr`.
            let lang = super::g2p::Lang::from_id(arr.lang.get(i).copied().unwrap_or(2))
                .unwrap_or(super::g2p::Lang::Ja);
            let permille = super::score2cv::voiceless_zero_permille(
                arr.phon[i],
                arr.note_dur.get(i).copied().unwrap_or(d as i64),
                lang,
            );
            let z = ((d as f64 * permille as f64 / 1000.0).round() as usize).min(d);
            let start = cursor + (d - z) / 2;
            for f in &mut note_hz[start.min(n)..(start + z).min(n)] {
                *f = 0.0;
            }
        }
        cursor += d;
    }
}

/// S83: with onset pre-roll, a phrase-INITIAL voiced consonant's frames sit BEFORE the beat — in the DAW
/// f0 curve that region is the preceding rest (voiced=0 → 0 Hz). A voiced phone must never carry 0 Hz
/// (cover convention: RMVPE reads a real pitch on m/n/ɡ onsets); 0 there means SoVITS uv=0 and RVC
/// pitchf=0 → NSF noise excitation + protect = an audibly mute onset (the exact trap the S83 triage
/// verifier flagged). Fill every zero frame of a SUNG, non-voiceless phone from the nearest nonzero f0
/// within its contiguous SUNG RUN (forward first — the vowel's opening pitch — then backward at tails).
/// Rests/breaths keep their zeros (rest-gate / protect semantics untouched); voiceless phones were just
/// zeroed on purpose and are skipped. Runs after `zero_voiceless_frames` on both backends.
fn anchor_voiced_phone_f0(note_hz: &mut [f32], arr: &ScoreArrays) {
    let total = note_hz.len();
    // run id per frame (usize::MAX = rest/breath frames) — fills never cross an SP/AP boundary.
    let mut run_of = vec![usize::MAX; total];
    let mut cursor = 0usize;
    let mut run_id = 0usize;
    let mut in_run = false;
    for (i, &d) in arr.phone_dur.iter().enumerate() {
        let d = d.max(0) as usize;
        if matches!(arr.phon[i], "SP" | "AP") {
            if in_run {
                run_id += 1;
                in_run = false;
            }
        } else {
            in_run = true;
            for r in run_of.iter_mut().skip(cursor.min(total)).take(d.min(total.saturating_sub(cursor))) {
                *r = run_id;
            }
        }
        cursor += d;
    }
    // two sweeps: nearest nonzero f0 before/after each frame, resetting at run boundaries.
    let sweep = |indices: &mut dyn Iterator<Item = usize>, out: &mut [f32]| {
        let (mut carry, mut carry_run) = (0.0f32, usize::MAX);
        for i in indices {
            if run_of[i] != carry_run {
                carry = 0.0;
                carry_run = run_of[i];
            }
            if note_hz[i] > 0.0 {
                carry = note_hz[i];
            }
            out[i] = carry;
        }
    };
    let mut prev_nz = vec![0.0f32; total];
    let mut next_nz = vec![0.0f32; total];
    sweep(&mut (0..total), &mut prev_nz);
    sweep(&mut (0..total).rev(), &mut next_nz);
    // fill pass: zeros on sung, voiced phones only.
    let mut cursor = 0usize;
    for (i, &d) in arr.phone_dur.iter().enumerate() {
        let d = d.max(0) as usize;
        let voiced_sung =
            !matches!(arr.phon[i], "SP" | "AP") && arr.note_pitch[i] > 0 && !is_voiceless_phone(arr.phon[i]);
        if voiced_sung {
            for f in cursor.min(total)..(cursor + d).min(total) {
                if note_hz[f] <= 0.0 {
                    note_hz[f] = if next_nz[f] > 0.0 { next_nz[f] } else { prev_nz[f] };
                }
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

/// S147 — what makes a render a DONOR pass rather than a base pass.
///
/// Both fields exist only because a donor is spliced into a base, and both would be wrong to set
/// on a base render, so they travel together instead of as two independent `Option`s that could
/// disagree.
#[derive(Clone, Copy)]
pub struct DonorCtx<'a> {
    /// Normalize against the BASE render's pre-norm peak so the two share one scalar
    /// (S147 笔 1 — replaces the whole-song `active_rms` guess).
    pub norm_peak_target: f32,
    /// The rescue windows this donor will actually be spliced into, as (start, end) frames on the
    /// score grid. Chunks that do not intersect any window are **not rendered** — they are the
    /// 76% of every donor pass that gets thrown away today (measured: 285.2s → 130.0s wall).
    pub windows: &'a [(i64, i64)],
    /// S159 —— 同一批窗的**样本**坐标(含渲染侧余量,已合并),交给逆变换。空 = 整条 = 今天。
    ///
    /// ⛔ 为什么不在这里从 `windows` 现算:帧→样本的比例 `spf = base.len() / total_frames`
    /// 里的 `total_frames` 是**谱面帧**的和(`commands/inference.rs` 算的),而这个函数内部
    /// 只有 `note_hz_full.len()`(= Σ `phone_dur`,cv 网格)。两者只在「帧守恒」时相等,
    /// 而本文件 :245-250 自己就列了三类不守恒的例外。⇒ 由**拥有那份地图的人**算好传下来,
    /// 与 `windows` 同一条理由(见 S148 在 `apply_dead_only_windows` 上写的那段)。
    pub keep_samples: &'a [(usize, usize)],
}

/// Which chunks a donor must actually render.
///
/// ⛔ Every rule here is a measurement, not a design instinct:
/// * **Margin.** The window→sample map (`spf`, used by the splice layer) is a LINEAR frame→sample
///   ratio, while real chunk boundaries land wherever `sovits_grid_len` rounding puts them —
///   measured divergence up to **458.8 samples (10.4 ms)**, against a tightest real margin of
///   **262 samples (5.9 ms)**. Intersecting in the frame domain with no margin therefore splices
///   the window's 10 ms cross-fade tail into digital zero.
/// * **`hard_seam` neighbours.** A hard seam is by definition a language cut that does NOT land in
///   silence (`score2cv.rs`), i.e. mid-voiced. Putting a hole edge there re-phases the whole PSOLA
///   island downstream (measured worst +2.9 dB, correlation gone). Single-language scores carry
///   `hard_seam == false` throughout, so this rule costs them exactly nothing.
fn donor_keep_mask(chunks: &[Chunk], windows: &[(i64, i64)], margin_frames: i64) -> Vec<bool> {
    let mut keep = vec![false; chunks.len()];
    let mut fstart = 0i64;
    for (i, c) in chunks.iter().enumerate() {
        let fend = fstart + c.t as i64;
        keep[i] = windows
            .iter()
            .any(|&(a, b)| fstart < b + margin_frames && a - margin_frames < fend);
        fstart = fend;
    }
    // hard_seam 落在浊音正中 ⇒ 它两侧都不许是洞边。
    for i in 0..chunks.len() {
        if chunks[i].hard_seam && (keep[i] || i > 0 && keep[i - 1]) {
            keep[i] = true;
            if i > 0 {
                keep[i - 1] = true;
            }
        }
    }
    keep
}

/// S159b —— 一个 chunk 在 **RVC 臂**上产出多少样本。
///
/// ⛔ 它必须与那一行 `wav.truncate((real_t * (m.sample_rate / 100)).min(wav.len()))` 用**同一个
/// 信念**:100 fps 的每一帧对应 `sr/100` 个输出样本,而 `real_t = 2·chunk.t`(`rvc_feed_100` 把
/// 50 fps 的 cv 复制成 100 fps 的音高栅格)。`min_frames` 的 pad 只影响喂进去的长度,**输出**那一侧
/// 已经被上面那行截回 `real_t`,所以这里不看它。
///
/// ⛔⛔ 为什么这个数是承重的:donor 跳过的 chunk 要**铺同样长的零**,而拼接层按**绝对样本**索引
/// `audio`。差一个样本,这一遍之后的每一条救援窗都跟着滑走 —— 而组数 / 乘客数 / donor 遍数 /
/// 位移集 / 音频秒数**全部正常**。SoVITS 臂上正是在这里踩过:用 `chunk.t * sr / 50` 在 akiko 的
/// hop 上每个 chunk 差 **124** 个样本(见 `sovits_grid_len` 那一行的注释)。
/// ⇒ 这条公式**不许推**,它由 `the_rvc_hole_is_exactly_what_a_chunk_produces` 钉住,
/// 而且真渲出来的每个 chunk 都会与它对一次(对不上就进 `[perf]` 的 `len!=` 计数,见那一行)。
pub(crate) fn rvc_out_len(t50: usize, sample_rate: u32) -> usize {
    t50 * 2 * (sample_rate as usize / 100)
}

/// S160q —— 把 50 fps 的 `note_hz` **线性**升到 `factor ×` 帧率(出厂关,见 [`score_f0_lerp`])。
///
/// ## 它治的是什么
/// 用户 2026-08-25 在频谱图上看出「我们的谐波线是一格一格的台阶,SV 是光滑曲线」,并且给了
/// 两条把嫌疑收死的信息:⑴ **3:55.791 那个音根本没进任何救援组** ⇒ PSOLA 不是唯一来源;
/// ⑵ **同一个模型在翻唱轨上做得到那种平滑** ⇒ 不是模型上限。
///
/// 根因:**两条车道用同一个 `sovits_f0_postprocess`,但方向相反。**
/// | 车道 | 进去的 f0 | 目标 | `torch_interp_nearest` 的后果 |
/// |---|---|---|---|
/// | 翻唱 | RMVPE **100 fps**(hop 160 @16k) | 86 fps | **降**采样 ⇒ 不保持 ⇒ 光滑 |
/// | 谱面 | 我们的 **50 fps** | 86 fps | **升**采样 ⇒ 每值保持 1-2 帧 = **11.6 / 23.2 ms 的台阶** |
/// RVC 那条更直白:`rvc_feed_100` 里字面上 `pitchf.push(f); pitchf.push(f);` = 20 ms 零阶保持。
/// ⇒ 先把谱面轨升到 100 fps,它进共用函数时就和翻唱轨**站在同一个起点**上。
///
/// ⛔⛔ **只在浊音游程【内部】插值。** 相邻两个源帧只要有一个是 0(清音 / 休止 /
/// [`zero_voiceless_frames`] 归零的帧),这一段就退回零阶保持 —— 否则会把音高抹进休止,
/// 破坏「`pitchf == 0` ⇒ NSF 噪声激励 + protect」这条契约(S83 那次哑起音正是它)。
/// 零帧的**位置与个数**因此逐位可预测:源里每个 0 变成 `factor` 个 0。
///
/// ⛔ **别去改 [`super::f0::sovits_f0_postprocess`]** —— 它是与上游 `RMVPEF0Predictor.post_process`
/// 逐位对齐的孪生,翻唱轨也吃它;改它等于毁掉翻唱轨的对齐。这里改的是**喂给它的东西**。
pub(crate) fn upsample_note_hz_linear(note_hz: &[f32], factor: usize) -> Vec<f32> {
    let n = note_hz.len();
    if factor <= 1 || n == 0 {
        return note_hz.to_vec();
    }
    let mut out = Vec::with_capacity(n * factor);
    for i in 0..n {
        let a = note_hz[i];
        let b = if i + 1 < n { note_hz[i + 1] } else { a };
        // 两端都浊才插值;任一端为 0 ⇒ 零阶保持(零帧原样复制 factor 份)。
        let lerp_ok = a > 0.0 && b > 0.0;
        for k in 0..factor {
            out.push(if k == 0 || !lerp_ok {
                a
            } else {
                a + (b - a) * (k as f32) / (factor as f32)
            });
        }
    }
    out
}

/// ⚙ 出厂默认 = **开**(S160q)。`UTAI_SCORE_F0_LERP=0` 关回零阶保持。
///
/// ## 翻它的账(东雪莲/akiko SoVITS + yachiyo/yuyuko RVC,各 hold/lerp,同一二进制 `2f68e47670f5`)
/// | 模型 | 后端 | 波形 \|Δ\| | >6 dB 的格 | 段/时长 | 最深局部 | 长时谱最大差 |
/// |---|---|---|---|---|---|---|
/// | 东雪莲 | SoVITS | +3.15 dB | 0.13% | 20 / 0.24 s | −13.7(音头晚 10 ms)| 0.29 dB |
/// | akiko | SoVITS | +2.67 | 0.15% | 16 / 0.27 s | **+17.7(救回一个失声的音)** | **0.04 dB** |
/// | yachiyo | RVC | +2.79 | 0.14% | 23 / 0.26 s | **+13.0(救回一个虚音头)** | 0.24 dB |
/// | yuyuko | RVC 40k | +3.19 | 0.18% | 27 / 0.32 s | −17.0(音头晚 10-20 ms)| 0.53 dB |
/// (两跑复现地板 = **−6.29 dB** ⇒ +2.7…+3.2 是真的动了。)
/// ✅ **计划逐组相同**(80/80 · 90/90 · 95/95 · 82/82)· ✅ **`uv` 掩码逐位不变**
/// (25073 帧 0 处不同,190 个浊/清边界位移 **0.0 ms** ⇒ **辅音时序一个字节没动**)
/// · ✅ **零帧位置逐位不变** · ✅ **音头无系统代价**(698 音 × 4 模型,音头 30 ms/稳态 的配对 Δ
/// 中位 +0.05 / +0.02 / −0.02 / +0.00 dB,变弱 3.0-9.3% vs 变强 4.2-6.5%,**对称**)。
///
/// ⛔ **这是这条线上爆炸半径最大的一刀:全曲每一个音都变,不只救援窗**;而且效应**均匀分布**
///    (窗外 +3.46 / 浅 +3.02 / 中 +3.42 / 深 ≥10 +2.17 dB)⇒ **它不是深救援的解药**,别指望。
pub(crate) fn score_f0_lerp() -> bool {
    parse_score_f0_lerp(std::env::var("UTAI_SCORE_F0_LERP").ok().as_deref())
}

/// ⛔ 纯函数,好让「出厂默认」本身有一条不依赖进程环境的判据(与 `parse_phase_lock` 同规矩)。
/// 出厂开 ⇒ **只有字面量 `"0"` 关得掉**(垃圾值一律落回出厂,与 `parse_fill1` 同极性)。
fn parse_score_f0_lerp(v: Option<&str>) -> bool {
    !matches!(v, Some("0"))
}

/// ⚙ **出厂默认 = 开**(S162 翻的;S160q 起是 [`gate_unvoiced_tone`] 的纯函数入口,
/// 做成纯函数只为一件事:**让它进得了指纹**)。**只有字面量 `"0"` 关得掉**
/// —— 与 `parse_fill1` / `parse_score_f0_lerp` 同极性,垃圾值一律落回出厂。
///
/// ⛔ 为什么 S162 才敢翻(S160k 那一版**没有护栏**):用户耳判确认它除掉了 0:47.229 那声咔哒,
/// 但全曲对拍发现它在 **46 条段里削掉真起音**(模型常把浊音渲得比乐谱音头早)。
/// S162 给它加了 [`parse_uvgate_guard_ms`](出厂 20 ms),把**真起音区**(距清音 run 尾 ≤1 帧)
/// 被压 >6 dB 的格从 **28 → 8(−71%)**,而治愈只从 −13.6 掉到 **−13.0 dB**;
/// 护栏 40 那一档是**预注册的阴性对照**,治愈当场归零(+0.5 dB)⇒ 20 是唯一工作点。
fn parse_uvgate(v: Option<&str>) -> bool {
    !matches!(v, Some("0"))
}
fn parse_uvgate_k(v: Option<&str>) -> f32 {
    v.and_then(|x| x.trim().parse::<f32>().ok())
        .filter(|k| k.is_finite() && (1.0..=6.0).contains(k))
        .unwrap_or(1.5)
}
/// ⚙ **出厂默认 = 20.0**(S162 翻的)。起音护栏(毫秒)。
///
/// 门只认「喂进去的 `note_hz == 0`」,而**模型常把浊音渲得比乐谱音头早**几十毫秒
/// ⇒ 清音 run 的**后**端(紧挨着后继浊音那一侧)躺着的往往是**真起音**,
/// 门会把它的基频整个滤掉(S160k 全曲对拍:46 条段 / 0.76 s,最深 −37.5 dB)。
/// 护栏 = 那一侧留 `guard_ms` 不碰。⛔ 只在**真有后继浊音**时收 —— 没有后继就没有起音可保。
fn parse_uvgate_guard_ms(v: Option<&str>) -> f32 {
    v.and_then(|x| x.trim().parse::<f32>().ok())
        .filter(|g| g.is_finite() && (0.0..=200.0).contains(g))
        .unwrap_or(UVGATE_GUARD_MS_DEFAULT)
}

/// ⚙ 出厂默认 = `20.0`。**宽度是数据给的,不是拍的**:门会删掉的那部分能量(截止以下)
/// 沿清音段位置的剖面 —— 距段尾 0-20 ms **+19.5 dB** · 20-40 ms **−1.8** · 40-60 ms **−130.8**
/// (鹅妈妈 +7 × yachiyo,141 个 ≥100 ms 的清音段,中位)⇒ **真起音就在最后那 20 ms 里**。
/// ⛔ 再宽就开始吃掉门本身:护栏 40 上 0:47.229 那个 3 帧的 run 只剩 1 帧,治愈从 −13.0 变成 **+0.5 dB**。
const UVGATE_GUARD_MS_DEFAULT: f32 = 20.0;
fn parse_valley_adaptive(v: Option<&str>) -> bool {
    matches!(v.map(str::trim), Some("1" | "true" | "on" | "yes"))
}
/// S161 —— ⚙ **出厂默认 = 开**(用户 2026-08-25 耳判:「这一刀确实让它听起来变好了 ——
/// 感觉它把中间有一串连续闭塞音部分的那个怪异的顿挫感解决了」)。`UTAI_VALLEY_HUMAN=0` 关回 S84 的表。
///
/// 辅音谷的**类深度**改用【真人差额】,而不是 S84 的「OpenUtau 参照 − 我们的渲染」差额。
///
/// ## ⛔ 为什么(S161 逐格量出来的,两根轴)
///
/// S84 那把刀的 doc 写着 "scale 1.0 statistically lands the render on the reference",
/// 而它的 reference 是 `未命名_MixDown.wav`(OpenUtau/UTAU 拼接渲染)。S161 拿
/// **GTSinger 日语 Control_Group**(832 clip,上游标注 + 数据集自带源音频,
/// **从没经过我们的对齐器**;英语那半边逐格复现了 `coda_ref_upstream.json`,8/8 PASS)
/// 重新量了两根轴,**两边窗口定义逐字相同**:
///
/// * `dip`   = 辅音窗外两侧各 60 ms 的 4 ms 包络中位 − 窗内最低  ← S84 那把刀**瞄准**的量
/// * `level` = 辅音整段 RMS − 紧随核音整段 RMS                   ← 这把刀**实际改变**的量
///
/// | 60-80 ms 的辅音 | 真人 dip / level | 我们 `UTAI_MG_VALLEY=0` | 我们出厂 |
/// |---|---|---|---|
/// | 浊塞音 | 22.3 / −5.4 | 7.9 / −2.5 | 19.8 / **−10.4** |
/// | **鼻音** | **4.4 / −1.5** | **3.3 / −0.9** | **14.3 / −7.6** |
/// | 闪音 | 9.6 / −1.8 | 2.8 / −0.1 | 12.9 / −7.4 |
/// | 清塞音(80-120 ms) | 37.1 / −15.2 | 34.2 / −4.7 | 46.0 / −15.3 |
///
/// ⇒ ⭐ **浊塞音/闪音上这把刀是对的**(把 dip 从 7.9 抬到 19.8,真人 22.3);
/// ⇒ ⛔ **鼻音/边音上它整整多了一个量级**:**关掉它**我们就已经在真人身上(3.3 vs 4.4),
///   开着它把 dip 打到 14.3(真人的 3.2 倍)、把电平压到 −7.6(真人 −1.5)。
///   跨语言复现:真人英语 L 2.85 / M 3.06 / N 3.53 / NG 2.21 dB(n=848…5162)。
/// ⇒ ⚠ 而**每一个浊类的 level 都过冲 5-8 dB** —— 那是「拿平坦增益去买凹陷」的结构性代价,
///   不是深度调错。要同时命中两根轴得把谷做成**窄槽**,那是另一件事(S161 只记账不做)。
///
/// 本旋钮只改**两个桶**,取「真人 dip − 我们自己的 dip」:
/// 鼻/边 11.4 → [`VALLEY_NASAL_HUMAN_DB`],闪 10.4 → [`VALLEY_TAP_HUMAN_DB`]。
/// 其余桶一个字节不动 —— 塞音那一格两根轴给出**相反方向**,没有任何一个平坦增益能同时对。
/// ⛔ 纯函数,好让「出厂默认」本身有一条不依赖进程环境的判据(与 `parse_score_f0_lerp` 同规矩)。
/// 出厂开 ⇒ **只有字面量 `"0"` 关得掉**(垃圾值一律落回出厂)。
fn parse_valley_human(v: Option<&str>) -> bool {
    !matches!(v, Some("0"))
}

fn valley_human() -> bool {
    parse_valley_human(std::env::var("UTAI_VALLEY_HUMAN").ok().as_deref())
}

/// S161d —— ⭐⭐⭐ **谷的形状直接从真人剖面取**,不再是任何参数化的槽。
///
/// ## ⛔ 为什么(S161c 的窄槽是错的,用户当场听出来)
///
/// 用户 2026-08-25:「这玩意是直接在频谱上**画竖条纹**啊」「顿挫感怎么听怎么不自然」。
/// 前三根轴(`level` 整段均值 / `dip` 最低点 / `rise` 上升时间)**对凹口的位置完全是瞎的**
/// —— 「窗中间挖一个洞、两侧照常响」与「前半闭塞、后半释放」给出**一模一样**的读数。
/// ⇒ 加第四根轴(`pos` 最低点在窗内的相对位置 · `前−后` 窗前 1/3 减窗后 1/3),当场现形:
///
/// | 3 帧浊塞音,窗内归一剖面(dB re 紧随元音) | 5% | 25% | 45% | 65% | 85% | pos | 前−后 |
/// |---|---|---|---|---|---|---|---|
/// | **真人** | −12.5 | −15.1 | −16.7 | −9.2 | −2.6 | 0.42 | **−10.0** |
/// | 谷全关(模型自己) | −4.8 | −6.0 | −5.7 | −2.6 | −0.3 | 0.45 | −5.3 |
/// | S161c 窄槽 | −4.7 | −6.1 | −5.6 | **−15.7** | −15.6 | **0.68** | **+9.4** |
///
/// ⇒ 真人是**前深后浅、单调回升**(先闭后放);窄槽是**前面照常响 → 中后段一个洞 → 末尾又回来**,
///   **洞的两侧都有全电平的肩** —— 那就是「竖条纹」。⭐ 而**模型自己的形状是对的**(pos 0.45,
///   单调回升),只是浅了约 10 dB ⇒ **该做的是把它加深,不是在中间挖洞。**
///
/// ## 形状 = (真人剖面 − 我们自己的剖面),峰值归一
///
/// 把窗归一成 10 格,逐格取「真人中位 − 谷全关中位」,负的截零,再除以峰值。
/// 两个时长桶算出来的模板几乎同形 ⇒ **每类只留一条**,深度另外按帧数给(见 `valley_*_human_db`)。
const VALLEY_ENV_N: usize = 10;
/// 浊塞音/浊塞擦的谷包络(真人 n=86/136)。**前重后轻,尾端归零 = 释放不被推迟。**
const VALLEY_ENV_VSTOP: [f32; VALLEY_ENV_N] =
    [0.82, 0.84, 0.86, 0.96, 0.84, 0.62, 0.46, 0.39, 0.21, 0.08];
/// 闪音的谷包络(真人 n=148/185)。峰更靠前,**65% 之后基本归零**。
const VALLEY_ENV_TAP: [f32; VALLEY_ENV_N] =
    [0.65, 0.87, 0.98, 0.94, 0.79, 0.59, 0.28, 0.10, 0.06, 0.02];

/// 每个音素的谷**包络**:`None` = 今天的整窗矩形(鼻/边、清塞音、擦音、滑音都走这条)。
/// ⛔ 只有**浊**塞音/塞擦(b d ɡ ɟ 起头,含 bʲ dʑ dz)与闪音取真人包络;清塞音(p t k c q ʈ ʔ)
/// 首字符与浊的不重叠,所以这条分支天然只吃浊的。
fn valley_shape_human(p: &str) -> Option<&'static [f32; VALLEY_ENV_N]> {
    match p.chars().next() {
        Some('b' | 'd' | 'ɡ' | 'ɟ' | 'ɢ') => Some(&VALLEY_ENV_VSTOP),
        Some('ɾ' | 'ɽ' | 'r') => Some(&VALLEY_ENV_TAP),
        _ => None,
    }
}

/// 真人差额(GTSinger ja Control_Group):鼻/边 dip 4.4 − 我们自己的 3.3 ≈ **1.1 dB**。
pub const VALLEY_NASAL_HUMAN_DB: f32 = 1.1;
/// ⛔⛔ S161c —— **槽深必须随辅音【时长】变,常数是错的。**
///
/// S161b 给了浊塞音一个常数 20 dB —— 而 S161 自己刚记过这条血训(「S84 标定用的正是快段,
/// 真人在那儿凹得最浅,而刀是常数」),**然后我又造了一把常数刀**。用户 2026-08-25 当场点名
/// 1:19.444 / 1:20.706 / 1:21.080 / 1:21.264 四处「可能是硬挖出来的」——**四处全是 3 帧的浊塞音**。
///
/// 真人日语 Control_Group 的 **10 ms 细分**(浊塞音 dip):
/// 20-40 ms **2.1** · 40-50 **13.5** · 50-60 13.4 · 60-70 **21.1** · 70-80 28.6 · 80-100 32.2 · 100+ 36-39。
/// 而我们**自己的** dip(谷全关)是 2 帧 2.9 / 3 帧 7.5 / 4 帧 17.8 ⇒ **差额本身就随时长变**。
///
/// 逐格扫参(两份素材一起:炉心融解 3/4 帧多 + 鹅妈妈 2 帧多,已排除救援窗):
///
/// | 类 | 帧 | n | 定出的深度 | 得到 level / dip | 真人 |
/// |---|---|---|---|---|---|
/// | 浊塞音 | 2 | 21 | **13.0** | −3.18 / 13.35 | −4.02 / 13.50 |
/// | 浊塞音 | 3 | 107 | **16.0** | −4.52 / 20.75 | −5.04 / 21.07 |
/// | 浊塞音 | ≥4 | 8 | **18.0** | −4.74 / 32.14 | −6.51 / 32.19 |
/// | 闪音 | 2 | 152 | **7.0** | −1.46 / 7.35 | −1.48 / 7.74 |
/// | 闪音 | ≥3 | 19 | **10.0** | −1.50 / 10.33 | −1.84 / 10.45 |
///
/// ⚠ ≥4 帧那一格 n=8,**小样本**;更长的辅音真人还在继续加深(100-130 ms → 36.4),
///   这里**故意封顶在 18**,不外推到没量过的区间。
fn valley_vstop_human_db(frames: i64) -> f32 {
    match frames {
        i64::MIN..=2 => 12.0,
        3 => 15.0,
        _ => 17.0,
    }
}

fn valley_tap_human_db(frames: i64) -> f32 {
    if frames <= 2 {
        6.5
    } else {
        9.0
    }
}

fn parse_valley_after(v: Option<&str>) -> bool {
    matches!(v.map(str::trim), Some("1" | "true" | "on" | "yes"))
}

/// ⛔⛔ S160q —— **这个文件里的生产默认,在这之前【完全不在任何指纹里】。**
///
/// S157c 立指纹闸的理由是「改了生产默认却没 bump 版本 = 零红,而那不是一个错误、是用户听到
/// 一条陈缓存」。但那条闸的指纹串**只由 `vocal_range.rs` 的默认拼出来** ⇒ 本文件里
/// **七个会改音频的旋钮**(含 **出厂就开着的** [`FILL_ISOLATED_UV_DEFAULT`])一个都看不见。
/// [`FILL_ISOLATED_UV_DEFAULT`] 头上那行注释甚至就在教人做成对 bump —— **而没有闸执行它**。
/// ⇒ 同一个形状,在隔壁文件里原封不动地又活了一遍。
///
/// ⚠ **一个指纹、一条闸、一个版本号**:这里只出串,核对与 bump 仍然由
/// `vocal_range` 那条唯一的判据做 —— 两条会互相不同意的闸比没有闸更糟。
pub(crate) fn production_defaults_fingerprint() -> String {
    format!(
        "f0lerp={} fill1={} filluv={} fillmax={} uvgate={} uvgatek={} uvgateguard={} valadapt={} valafter={} valhuman={} restshrink={} predamp={}/{},{},{},{},{},{} valdb={}/{},{},{}/{},{} valenv={:.2},{:.2}/{:.2},{:.2}",
        parse_score_f0_lerp(None),
        parse_fill1(None),
        FILL_ISOLATED_UV_DEFAULT,
        FILL_MAX_FRAMES_DEFAULT,
        parse_uvgate(None),
        parse_uvgate_k(None),
        parse_uvgate_guard_ms(None),
        parse_valley_adaptive(None),
        parse_valley_after(None),
        // S161 —— ⚠ 与 `parse_windowed_inverse` 同款:它进指纹但**出厂关 ⇒ 不改音频**,
        //    所以加它**不该**触发版本 bump;进指纹的意义是「下一个人翻它之前必须来这里改一行」。
        parse_valley_human(None),
        // ⭐ S163 —— 休止门的 fade 随窗长收缩。**出厂开 ⇒ 改音频 ⇒ 必须配版本 bump**
        //    (与 `valley_human` 那条相反，它出厂关所以不 bump)。
        parse_rest_gate_shrink(None),
        // ⭐ S163 —— preroll 辅音前部衰减。**出厂开 ⇒ 改音频 ⇒ 必须配版本 bump**。
        //    三个常量都进指纹：改任何一个都会动听感。
        parse_preroll_damp(None),
        PREROLL_KEEP_MS,
        PREROLL_DAMP_THRESH_DB,
        PREROLL_DAMP_MAX_SLOPE_DB_PER_MS,
        PREROLL_DAMP_LEFT_SLOPE_DB_PER_MS,
        PREROLL_DAMP_LEFT_BORROW_MS,
        PREROLL_PEAK_WIN_MS,
        // ⛔⛔ S161b —— **类深度与槽形也进指纹**。S161 只登记了旋钮,而这一场改的是**常量**
        //    (浊塞音 11.7 → 20 窄槽、闪音 6.8 → 8):旋钮没动、指纹没动、音频却变了 ⇒ 又是零红。
        //    ⇒ 凡是这几个常量被改,这条闸当场红,红的措辞会指到那三处版本字面量。
        VALLEY_NASAL_HUMAN_DB,
        valley_vstop_human_db(2),
        valley_vstop_human_db(3),
        valley_vstop_human_db(4),
        valley_tap_human_db(2),
        valley_tap_human_db(3),
        // S161d —— 包络的**形状**也必须进指纹(它决定听感,比深度更甚)。
        //    取每条模板的「峰值格位置 + 尾格值」两个数:改任何一格都会动其中之一。
        VALLEY_ENV_VSTOP.iter().copied().fold(0.0f32, f32::max),
        VALLEY_ENV_VSTOP[VALLEY_ENV_N - 1],
        VALLEY_ENV_TAP.iter().copied().fold(0.0f32, f32::max),
        VALLEY_ENV_TAP[VALLEY_ENV_N - 1],
    )
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
    // S160q:先线性升到 100 fps,让这一步变成【降】采样(= 翻唱轨的方向)。⚙ 出厂 **开**(见 `score_f0_lerp` 的 doc);`UTAI_SCORE_F0_LERP=0` 关回零阶保持。
    let hz_up: Vec<f32>;
    let hz_in: &[f32] = if score_f0_lerp() {
        hz_up = upsample_note_hz_linear(note_hz, 2);
        &hz_up
    } else {
        note_hz
    };
    let (f0_rs, uv_rs) = super::f0::sovits_f0_postprocess(hz_in, t_tgt, hop, sr);
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

/// The ② render's SCORE-SHAPING knobs, as ONE named argument.
///
/// ⚠ These used to ride as positional parameters on `render_score_{sovits,rvc}` — already two
/// adjacent `f32`s, and the next knob queued behind them is a `bool` that would have landed right
/// next to `vowel_clarity`. S85 paid for that shape once already — a same-typed tuple silently
/// permuted across a function boundary, was
/// invisible to both human review and a 3-finder audit, and shipped as "+7512 st". The rule that
/// came out of it (`[[project_v2_session85]]`): **≥2 same-typed semantic fields must be a named
/// struct, so the contract is enforced by the compiler.** Add knobs HERE, never as new positionals.
///
/// Every field's DEFAULT is the production default, so `ScoreShaping::default()` is exactly what a
/// no-track-context path (model audition) wants — and `..Default::default()` keeps a new knob from
/// silently landing as `false`/`0.0` at the paths nobody remembered to update.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoreShaping {
    /// S83 knife 6b: voiceless-onset emphasis, dB (0 = exact no-op).
    pub consonant_emphasis_db: f32,
    /// S84 C 刀: chain-internal consonant-valley scale ×measured depth (0 = exact no-op).
    pub consonant_valley_scale: f32,
    /// S84 E 刀: vowel-clarity articulation oversampling (cv-domain).
    pub vowel_clarity: bool,
    /// S89 「自动音素时序」: `true` = S83 onset pre-roll (the nucleus lands on the beat);
    /// `false` = every phone stays inside its own note. See `score2cv::ArticulationTiming`.
    pub consonant_preroll: bool,
}

impl ScoreShaping {
    /// The allocator's view of `consonant_preroll`. ONE conversion point, so the bool can never be
    /// read as an `ArticulationTiming` with the polarity flipped at one call site out of three.
    pub fn articulation_timing(&self) -> super::score2cv::ArticulationTiming {
        if self.consonant_preroll {
            super::score2cv::ArticulationTiming::Auto
        } else {
            super::score2cv::ArticulationTiming::InNote
        }
    }
}

impl Default for ScoreShaping {
    fn default() -> Self {
        ScoreShaping {
            consonant_emphasis_db: DEFAULT_VOICELESS_ONSET_EMPHASIS_DB,
            consonant_valley_scale: DEFAULT_CONSONANT_VALLEY_SCALE,
            vowel_clarity: true,
            consonant_preroll: true,
        }
    }
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
    shaping: ScoreShaping,
    transpose: i64,
    range_shift: i64,
    f0: Option<&VocalF0>,
    loudness: Option<&[f32]>,
    formant: Option<&[f32]>,
    cancel: &(dyn Fn() -> bool + Sync),
    progress: &dyn Fn(f32),
    // S147: donor-pass context. `None` = a normal (base) render, byte-identical to before —
    // which is what all 19 existing call sites pass. See `DonorCtx`.
    donor: Option<DonorCtx<'_>>,
) -> Result<SynthesisResult> {
    let mut arr = build_arrays_daw(score, dicts, shaping.articulation_timing())?;
    // S60-2 音域扩展: the model RENDERS at transpose+range_shift (inside its comfort zone);
    // apply_range_inverse below shifts the audio back. range_shift=0 ⇒ byte-identical to before.
    let transpose_eff = transpose + range_shift;
    // f0 uses RAW note_pitch (grouping/voicing); transpose folds into the OUTPUT Hz.
    let mut note_hz_full = build_note_hz(&arr, score, transpose_eff, f0);
    // S69 R0b①: voiceless frames → 0 Hz (cover parity: RMVPE emits 0 there; SoVITS then gets
    // uv=0 + gap-interp).
    // (R0b②'s procedural micro-texture was ear-vetoed and removed — see the note above
    // build_vol_env; micro-motion waits for the conditioned-S2CV line.)
    zero_voiceless_frames(&mut note_hz_full, &arr);
    // S159zp —— 见 [`fill_isolated_unvoiced`]:帧量化 + 辅音前置在快音上造出来的孤立单帧 0。
    if fill_isolated_unvoiced() {
        let k = fill_isolated_uv_max(&mut note_hz_full, fill_max_frames());
        if k > 0 {
            tracing::info!("score2svc: filled {k} isolated single-frame unvoiced holes");
        }
    }
    // S83: pre-rolled voiced onsets (m/n/ɡ… before the beat) must carry the note's opening pitch,
    // never the rest's 0 Hz (uv=0 would mute them) — fill from the run's nearest voiced frame.
    anchor_voiced_phone_f0(&mut note_hz_full, &arr);
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
    let vl_onset = voiceless_onset_flags(&arr);
    // S163 —— 被 `consonant_preroll` 提前到休止里的辅音串。见 `PREROLL_DAMP_DEFAULT`。
    let pre_cons = preroll_consonant_flags(&arr);
    let mut preroll_damped = 0usize;
    let emphasis_gain = if shaping.consonant_emphasis_db.is_finite() && shaping.consonant_emphasis_db > 0.0 {
        10f32.powf(shaping.consonant_emphasis_db.min(12.0) / 20.0)
    } else {
        1.0 // 0/invalid = exact no-op (×1.0 is bit-transparent)
    };
    // S84 C 刀: chain-internal consonant valley (scale 0/invalid = stage skipped, bit-exact no-op)
    let valley_depths = boundary_valley_depths(&arr);
    let valley_shapes = boundary_valley_shapes(&arr);
    let coda_lifts = phrase_final_coda_lifts(&arr);
    let valley_scale = if shaping.consonant_valley_scale.is_finite() && shaping.consonant_valley_scale > 0.0 {
        shaping.consonant_valley_scale.min(2.0)
    } else {
        0.0
    };
    // S159zb —— **donor 那一遍**把辅音谷推迟到逆变换之后(见 `valley_after_inverse`)。
    // ⛔ `range_shift == 0` 时不推迟 ⇒ 未开扩展的渲染逐位不变。
    let defer_valley = range_shift != 0 && valley_after_inverse();
    let mut deferred_valley: Vec<ValleyCluster> = Vec::new();
    let n_chunks = chunks.len().max(1);
    let has_diff = m.diffusion.is_some();
    let p_vits = if has_diff { 0.5 } else { 0.95 };
    let noop = |_: f32| {}; // decode_features' internal sub-progress is ignored (per-chunk coarse below)
    let mut audio: Vec<f32> = Vec::new();
    let mut cv_cursor = 0usize;
    // S147 秒表(纯 tracing,零行为改动)。⛔ 存在的理由:在此之前 score 渲染这条路上
    // `Instant::now` **一处都没有**,追速度时只能靠外部采样,而每换一个口径读数就不可比了
    // (实测:开/关增强器两个口径下同一笔改动的收益差 1.81 倍)。累加器只在**段边界**取时间,
    // 不进任何逐样本循环 ⇒ 开销可忽略,而且**不打 per-chunk 日志**(25 chunk × 5 段 = 125 行噪声)。
    let t_render = std::time::Instant::now();
    let (mut t_s2cv, mut t_decode) = (0f64, 0f64);
    // S147 B2: a donor only has to render the chunks it will be spliced FROM. `None` (base pass)
    // ⇒ every chunk kept ⇒ byte-identical to before.
    let keep: Option<Vec<bool>> =
        donor.map(|d| donor_keep_mask(&chunks, d.windows, DONOR_WINDOW_MARGIN_FRAMES));
    let mut skipped = 0usize;
    for (ci, chunk) in chunks.iter().enumerate() {
        if keep.as_ref().is_some_and(|k| !k[ci]) {
            // ⛔ Equal-length ZEROS, sized by `sovits_grid_len` — the splice layer indexes `audio`
            // by absolute sample, so the buffer's length contract is load-bearing. Using
            // `chunk.t * sr / 50` instead is off by 124 samples per chunk on akiko's hop.
            audio.resize(audio.len() + sovits_grid_len(chunk.t, m.sample_rate, m.hop_size) * m.hop_size, 0.0);
            cv_cursor += chunk.t; // ⚠ must advance even when skipped: note_hz slices are absolute
            skipped += 1;
            progress(p_vits * (ci + 1) as f32 / n_chunks as f32);
            continue;
        }
        if cancel() {
            return Err(UtaiError::Inference("CANCELLED".into()));
        }
        // S84 E 刀: vowel-clarity oversampling (off / no qualifying nucleus = the plain call).
        let t0 = std::time::Instant::now();
        let cv = if shaping.vowel_clarity {
            run_score2cv_vowel_clarity(
                m.engine, score2cv_session, chunk, &arr.phon[chunk.start..chunk.end],
                &arr.evt[chunk.start..chunk.end], dim, cv_speaker_id, chunk.lang_id,
            )?
        } else {
            run_score2cv(m.engine, score2cv_session, chunk, dim, cv_speaker_id, chunk.lang_id)?
        };
        t_s2cv += t0.elapsed().as_secs_f64();
        // ⛔ S147 B2 的铺零公式(跳过分支)用 `chunk.t` 算样本数,而真正决定长度的是
        // `sovits_grid_len(cv.nrows(), …)`。两者相等是**读出来的**(clarity_resample 把过采样
        // 的 cv 重采样回 chunk.phone_dur),不是被守卫的 —— 一旦谁改了那个重采样契约,
        // 跳过的 chunk 就会铺错长度,而拼接层用绝对样本下标 ⇒ 之后每个窗静默滑位。
        debug_assert_eq!(cv.nrows(), chunk.t, "cv rows must equal chunk.t — the zero-fill depends on it");
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
        let t0 = std::time::Instant::now();
        let mut wav = decode_features(
            m, feed.cv, feed.f0, feed.uv, vol, &[], ci as u64, padded_t, has_diff, p_vits, options,
            &noop, cancel,
        )?;
        t_decode += t0.elapsed().as_secs_f64();
        if padded_t > real_t {
            wav.truncate((real_t * m.hop_size).min(wav.len())); // drop the pad samples
        }
        // S73e: 真休止(SP)窗零化——cover slicer 铺零的输出域等价物(AP 呼吸不动;详见 gate 头注)
        let sp_wins = chunk_sp_windows(chunk, wav.len());
        apply_rest_gate(&mut wav, &sp_wins, rest_gate_fade_samples(m.sample_rate));
        // S83 knife 6: crisp up voiceless onsets (+2.5 dB trapezoid on their windows)
        let emph_wins = chunk_flag_windows(chunk, wav.len(), &vl_onset[chunk.start..chunk.end]);
        apply_emphasis(&mut wav, &emph_wins, emphasis_gain, emphasis_fade_samples(m.sample_rate));
        // ⭐ S163 —— 压 preroll 辅音窗的**前部**（紧贴音头的真辅音不动）。见 `PREROLL_DAMP_DEFAULT`。
        // ⛔ 放在 `apply_emphasis` **之后**：那一刀给清辅音 onset +2.5 dB，先压后抬会互相抵消。
        if preroll_damp_enabled() {
            let pc_wins = chunk_flag_windows(chunk, wav.len(), &pre_cons[chunk.start..chunk.end]);
            let keep = ((PREROLL_KEEP_MS / 1000.0) * m.sample_rate as f32).round().max(1.0) as usize;
            preroll_damped += apply_preroll_damp(
                &mut wav,
                &pc_wins,
                keep,
                PREROLL_DAMP_THRESH_DB,
                PREROLL_DAMP_MAX_DB,
                m.sample_rate,
            );
        }
        // S97 ②a 刀: restore phrase-final sonorant codas swallowed by the release ramp
        apply_coda_lift(&mut wav, chunk, &coda_lifts, emphasis_fade_samples(m.sample_rate));
        // S84 C 刀: carve the chain-internal syllable-boundary valleys (measured class depths × scale)
        if valley_scale > 0.0 {
            let val_cls = chunk_valley_clusters(chunk, wav.len(), &valley_depths[chunk.start..chunk.end], &valley_shapes[chunk.start..chunk.end]);
            if defer_valley {
                // S159zb —— 攒起来,等 `apply_range_inverse` 之后再刻(见 `valley_after_inverse`)。
                // ⛔ 偏移取**扩展之前**的 `audio.len()`:`seam_fade` 只做淡化、不改两侧长度,
                //    而 TD-PSOLA 不改时长 ⇒ 绝对下标在逆变换之后仍然有效。
                let base = audio.len();
                deferred_valley.extend(val_cls.into_iter().map(|c| ValleyCluster {
                    win: c.win.into_iter().map(|(a, b, d)| (base + a, base + b, d)).collect(),
                    env: c.env,
                }));
            } else {
                apply_valley(&mut wav, &val_cls, valley_scale, emphasis_fade_samples(m.sample_rate), m.sample_rate);
            }
        }
        // S161f —— **每一条接缝都淡化**(以前只 hard_seam;见 `seam_fade` 的 doc)。
        seam_fade(&mut audio, &mut wav, m.sample_rate);
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
    let t0 = std::time::Instant::now();
    audio = apply_range_inverse(
        audio, m.sample_rate, range_shift, options.range_formant_follow, &note_hz_full,
        donor.map_or(&[][..], |d| d.keep_samples),
    )?;
    let t_inverse = t0.elapsed().as_secs_f64();
    // S159zb —— 攒下来的辅音谷在这里刻(见 `valley_after_inverse`)。
    if !deferred_valley.is_empty() {
        let cls = if valley_adaptive() {
            // S159zb ⑵ —— **量了再补差额**(与 `apply_coda_lift` 同一条;见 `valley_adaptive`)。
            deferred_valley
                .iter()
                .map(|c| {
                    let (s0, e1) = (c.win[0].0, c.win[c.win.len() - 1].1);
                    let have = measured_valley_db(&audio, s0, e1);
                    ValleyCluster {
                        win: c
                            .win
                            .iter()
                            .map(|&(a, b, d)| (a, b, (d - have / valley_scale.max(1e-6)).clamp(0.0, d)))
                            .collect(),
                        env: c.env,
                    }
                })
                .collect::<Vec<_>>()
        } else {
            deferred_valley.clone()
        };
        apply_valley(&mut audio, &cls, valley_scale, emphasis_fade_samples(m.sample_rate), m.sample_rate);
    }
    let pre_norm_peak = audio.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    peak_normalize_to(&mut audio, 0.92, donor.map(|d| d.norm_peak_target));

    let wall = t_render.elapsed().as_secs_f64();
    let secs = audio.len() as f64 / f64::from(m.sample_rate);
    tracing::info!(
        "[perf] score/sovits {secs:.1}s audio in {wall:.2}s (RTF {:.3}) · {} chunks ·          s2cv {t_s2cv:.2}s ({:.0}%) · decode(net_g+voc) {t_decode:.2}s ({:.0}%) ·          inverse {t_inverse:.2}s ({:.0}%) · other {:.2}s · range_shift {range_shift:+} \
         · skipped {skipped}/{}",
        if secs > 0.0 { wall / secs } else { 0.0 },
        chunks.len(),
        100.0 * t_s2cv / wall.max(1e-9),
        100.0 * t_decode / wall.max(1e-9),
        100.0 * t_inverse / wall.max(1e-9),
        (wall - t_s2cv - t_decode - t_inverse).max(0.0),
        chunks.len(),
    );
    Ok(SynthesisResult { audio, sample_rate: m.sample_rate, pre_norm_peak: Some(pre_norm_peak) })
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
    // S160q:⚙ 出厂 **开** = 浊音游程内的线性插值;`UTAI_SCORE_F0_LERP=0` 关回零阶保持(与 S160q 之前逐位相同)。
    let at = |k: usize| note_hz.get(k.min(note_hz.len().saturating_sub(1))).copied().unwrap_or(0.0);
    let lerp = score_f0_lerp();
    let mut pitchf: Vec<f32> = Vec::with_capacity(pad50 * 2);
    for i in 0..pad50 {
        let f = at(i);
        let nxt = at(i + 1);
        let mid = if lerp && f > 0.0 && nxt > 0.0 { 0.5 * (f + nxt) } else { f };
        pitchf.push(f);
        pitchf.push(mid);
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
    shaping: ScoreShaping,
    transpose: i64,
    range_shift: i64,
    f0: Option<&VocalF0>,
    loudness: Option<&[f32]>,
    formant: Option<&[f32]>,
    cancel: &(dyn Fn() -> bool + Sync),
    progress: &dyn Fn(f32),
    // S147: donor-pass context. `None` = a normal (base) render, byte-identical to before —
    // which is what all 19 existing call sites pass. See `DonorCtx`.
    donor: Option<DonorCtx<'_>>,
) -> Result<SynthesisResult> {
    let mut arr = build_arrays_daw(score, dicts, shaping.articulation_timing())?;
    // S60-2 音域扩展 — same recipe as the SoVITS render: sing at transpose+range_shift, shift back below.
    let transpose_eff = transpose + range_shift;
    let mut note_hz_full = build_note_hz(&arr, score, transpose_eff, f0);
    // S69 R0b① (same as the SoVITS render): voiceless frames → pitchf 0 — RVC's cover convention
    // (RMVPE zeros) — which finally lets the protect blend fire on consonants. (② removed, ear-veto.)
    zero_voiceless_frames(&mut note_hz_full, &arr);
    // S159zp —— 见 [`fill_isolated_unvoiced`]:帧量化 + 辅音前置在快音上造出来的孤立单帧 0。
    if fill_isolated_unvoiced() {
        let k = fill_isolated_uv_max(&mut note_hz_full, fill_max_frames());
        if k > 0 {
            tracing::info!("score2svc: filled {k} isolated single-frame unvoiced holes");
        }
    }
    // S83: pre-rolled voiced onsets must carry pitch (pitchf=0 = NSF noise excitation + protect —
    // an audibly mute onset); fill from the run's nearest voiced frame.
    anchor_voiced_phone_f0(&mut note_hz_full, &arr);
    // ② loudness/formant per-cv-frame envelopes (RVC has no vol port → both are post-decode). Empty ⇒ no-op.
    let loud_cv = loudness.map(|l| build_note_param(&arr, score, l, 1.0));
    let formant_cv = formant.map(|f| build_note_param(&arr, score, f, 0.0));
    transpose_note_pitch(&mut arr.note_pitch, transpose_eff);
    let chunks = chunk_at_sp(&arr, 400);
    let vl_onset = voiceless_onset_flags(&arr);
    // S163 —— 被 `consonant_preroll` 提前到休止里的辅音串。见 `PREROLL_DAMP_DEFAULT`。
    let pre_cons = preroll_consonant_flags(&arr);
    let mut preroll_damped = 0usize;
    let emphasis_gain = if shaping.consonant_emphasis_db.is_finite() && shaping.consonant_emphasis_db > 0.0 {
        10f32.powf(shaping.consonant_emphasis_db.min(12.0) / 20.0)
    } else {
        1.0 // 0/invalid = exact no-op (×1.0 is bit-transparent)
    };
    // S84 C 刀: chain-internal consonant valley (scale 0/invalid = stage skipped, bit-exact no-op)
    let valley_depths = boundary_valley_depths(&arr);
    let valley_shapes = boundary_valley_shapes(&arr);
    let coda_lifts = phrase_final_coda_lifts(&arr);
    let valley_scale = if shaping.consonant_valley_scale.is_finite() && shaping.consonant_valley_scale > 0.0 {
        shaping.consonant_valley_scale.min(2.0)
    } else {
        0.0
    };
    // S159zb —— **donor 那一遍**把辅音谷推迟到逆变换之后(见 `valley_after_inverse`)。
    // ⛔ `range_shift == 0` 时不推迟 ⇒ 未开扩展的渲染逐位不变。
    let defer_valley = range_shift != 0 && valley_after_inverse();
    let mut deferred_valley: Vec<ValleyCluster> = Vec::new();
    let n_chunks = chunks.len().max(1);
    let sid = options.speaker_id.unwrap_or(0) as i64;
    // ①c: a genuine multi-speaker RVC export takes a dense spk_mix blend in place of scalar sid.
    let spk_mix_dense = m
        .spk_mix
        .map(|n| build_spk_mix_dense(&options.spk_mix, options.speaker_id, n));
    // S84 B 刀: per-50fps-frame retrieval weights (fast notes retrieve wrong neighbours).
    let idx_weights = fast_index_weights(&arr);
    let mut audio: Vec<f32> = Vec::new();
    let mut cv_cursor = 0usize;
    // S159b —— **秒表**。RVC 谱面臂到今天为止**一行都没有**,所以「这条臂慢在哪」在日志里
    // 读不出来(S159 那次是按日志时间戳手算出 182 s vs SoVITS 78.6 s 的)。
    // ⛔ 与 SoVITS 臂同款:只在段边界取时间,不进任何逐样本循环,且不打 per-chunk 日志。
    let t_render = std::time::Instant::now();
    let (mut t_s2cv, mut t_decode) = (0f64, 0f64);
    // S159b —— **B2 移植**:一遍 donor 只需要渲它会被拼回去的那些 chunk。
    // ⛔ 这一刀 S147 只做在 SoVITS 臂上,RVC 谱面臂一直每遍渲整曲(实测同曲 182 s vs 78.6 s)。
    // 谓词、余量、`hard_seam` 两侧必保这三条全部复用 `donor_keep_mask` —— ⛔ 不许在这里重写一份:
    // S147 那次「渲多了但拼对了 = 功能正确、收益静默减半」正是因为同一个谓词写了两遍。
    let keep: Option<Vec<bool>> =
        donor.map(|d| donor_keep_mask(&chunks, d.windows, DONOR_WINDOW_MARGIN_FRAMES));
    let (mut skipped, mut len_mismatch) = (0usize, 0usize);
    for (ci, chunk) in chunks.iter().enumerate() {
        // ⛔ 铺零的长度必须与真渲出来的一样(见 `rvc_out_len` 的 doc:差一个样本,这一遍之后
        //    的每一条救援窗都静默滑走)。
        let want = rvc_out_len(chunk.t, m.sample_rate);
        if keep.as_ref().is_some_and(|k| !k[ci]) {
            audio.resize(audio.len() + want, 0.0);
            cv_cursor += chunk.t; // ⚠ 必须照样前进:note_hz / idx_weights 的切片是绝对下标
            skipped += 1;
            progress((ci + 1) as f32 / n_chunks as f32);
            continue;
        }
        if cancel() {
            return Err(UtaiError::Inference("CANCELLED".into()));
        }
        // S84 E 刀: vowel-clarity oversampling (off / no qualifying nucleus = the plain call).
        let t0 = std::time::Instant::now();
        let cv = if shaping.vowel_clarity {
            run_score2cv_vowel_clarity(
                m.engine, score2cv_session, chunk, &arr.phon[chunk.start..chunk.end],
                &arr.evt[chunk.start..chunk.end], dim, cv_speaker_id, chunk.lang_id,
            )?
        } else {
            run_score2cv(m.engine, score2cv_session, chunk, dim, cv_speaker_id, chunk.lang_id)?
        };
        t_s2cv += t0.elapsed().as_secs_f64();
        let note_hz = &note_hz_full[cv_cursor..(cv_cursor + chunk.t).min(note_hz_full.len())];
        let (cv_p, pitch, pitchf, real_t) = rvc_feed_100(cv, note_hz, m.min_frames);
        // chunk 权重切片(min_frames pad 出的行由 vc_decode 的 unwrap_or(1.0) 兜=不加权)。
        let w_chunk = &idx_weights[cv_cursor..(cv_cursor + chunk.t).min(idx_weights.len())];
        let t1 = std::time::Instant::now();
        let mut wav = vc_decode(
            m, cv_p, &pitch, &pitchf, sid, spk_mix_dense.as_deref(), options, ci as u64, usize::MAX,
            Some(w_chunk),
        )?;
        t_decode += t1.elapsed().as_secs_f64();
        if pitchf.len() > real_t {
            // RVC net_g emits ~ p_len·(sr/100) samples; keep only the pre-pad span.
            wav.truncate((real_t * (m.sample_rate as usize / 100)).min(wav.len()));
        }
        // S159b —— ⛔ 那个「~」必须变成一条会响的检查:`rvc_out_len` 是**跳过的 chunk 铺多少零**
        // 的唯一依据,而它与上面那行截断共用同一个信念(每 100 fps 帧 = `sr/100` 个样本)。
        // 对不上就意味着「洞」的长度是错的 ⇒ 这一遍之后每条救援窗都滑走,而所有计数器全绿。
        // ⇒ 计数进 `[perf]`(与 `skipped` 同一行)—— 那正是 S147 那次静默减半唯一被抓到的方式。
        if wav.len() != want {
            len_mismatch += 1;
        }
        // S73e: 真休止(SP)窗零化(RVC 症状最轻但同样受益;AP 呼吸不动)
        let sp_wins = chunk_sp_windows(chunk, wav.len());
        apply_rest_gate(&mut wav, &sp_wins, rest_gate_fade_samples(m.sample_rate));
        // S83 knife 6: crisp up voiceless onsets (+2.5 dB trapezoid on their windows)
        let emph_wins = chunk_flag_windows(chunk, wav.len(), &vl_onset[chunk.start..chunk.end]);
        apply_emphasis(&mut wav, &emph_wins, emphasis_gain, emphasis_fade_samples(m.sample_rate));
        // ⭐ S163 —— 压 preroll 辅音窗的**前部**（紧贴音头的真辅音不动）。见 `PREROLL_DAMP_DEFAULT`。
        // ⛔ 放在 `apply_emphasis` **之后**：那一刀给清辅音 onset +2.5 dB，先压后抬会互相抵消。
        if preroll_damp_enabled() {
            let pc_wins = chunk_flag_windows(chunk, wav.len(), &pre_cons[chunk.start..chunk.end]);
            let keep = ((PREROLL_KEEP_MS / 1000.0) * m.sample_rate as f32).round().max(1.0) as usize;
            preroll_damped += apply_preroll_damp(
                &mut wav,
                &pc_wins,
                keep,
                PREROLL_DAMP_THRESH_DB,
                PREROLL_DAMP_MAX_DB,
                m.sample_rate,
            );
        }
        // S97 ②a 刀: restore phrase-final sonorant codas swallowed by the release ramp
        apply_coda_lift(&mut wav, chunk, &coda_lifts, emphasis_fade_samples(m.sample_rate));
        // S84 C 刀: carve the chain-internal syllable-boundary valleys (measured class depths × scale)
        if valley_scale > 0.0 {
            let val_cls = chunk_valley_clusters(chunk, wav.len(), &valley_depths[chunk.start..chunk.end], &valley_shapes[chunk.start..chunk.end]);
            if defer_valley {
                // S159zb —— 攒起来,等 `apply_range_inverse` 之后再刻(见 `valley_after_inverse`)。
                // ⛔ 偏移取**扩展之前**的 `audio.len()`:`seam_fade` 只做淡化、不改两侧长度,
                //    而 TD-PSOLA 不改时长 ⇒ 绝对下标在逆变换之后仍然有效。
                let base = audio.len();
                deferred_valley.extend(val_cls.into_iter().map(|c| ValleyCluster {
                    win: c.win.into_iter().map(|(a, b, d)| (base + a, base + b, d)).collect(),
                    env: c.env,
                }));
            } else {
                apply_valley(&mut wav, &val_cls, valley_scale, emphasis_fade_samples(m.sample_rate), m.sample_rate);
            }
        }
        // S161f —— **每一条接缝都淡化**(以前只 hard_seam;见 `seam_fade` 的 doc)。
        seam_fade(&mut audio, &mut wav, m.sample_rate);
        audio.extend_from_slice(&wav);
        cv_cursor += chunk.t;
        progress((ci + 1) as f32 / n_chunks as f32);
    }
    // ⛔⛔ S159b —— **整条缓冲的长度必须与「每个 chunk 都渲」时一模一样**,而且要在**循环刚结束**
    // 处量(`apply_formant_env` 之后再量会把它自己的长度变化算进来 = 假警报)。
    // `len!=` 只看**真渲出来的** chunk,铺零那一侧它一个字都看不见 —— 而铺零正是这一刀新加的那条路。
    // 少铺 / 多铺一个样本,之后每条救援窗都滑走;而 `apply_dead_only_windows` 的
    // 「donor N vs base M — clamped」那条 warn 只在差超过一整帧(≈960 样本)时才响
    // ⇒ 20 个洞各差 1 个样本**够不着它**。
    let want_total: usize = chunks.iter().map(|c| rvc_out_len(c.t, m.sample_rate)).sum();
    let total_delta = audio.len() as i64 - want_total as i64;
    if total_delta != 0 {
        tracing::warn!(
            "score/rvc: the decoded buffer is {total_delta:+} samples off what the chunk grid says              ({} vs {want_total}) — every rescue window after the first hole would slide",
            audio.len()
        );
    }
    // Post-decode (§M-defer order 响度增益 → 共振腔 → 归一化): RVC has no net_g vol port, so loudness is an
    // absolute gain envelope; formant warps the timbre. Both no-op when their env is None/flat.
    if let Some(lc) = &loud_cv {
        apply_gain_env(&mut audio, lc);
    }
    if let Some(fc) = &formant_cv {
        audio = apply_formant_env(audio, fc);
    }
    let t0 = std::time::Instant::now();
    audio = apply_range_inverse(
        audio, m.sample_rate, range_shift, options.range_formant_follow, &note_hz_full,
        donor.map_or(&[][..], |d| d.keep_samples),
    )?;
    let t_inverse = t0.elapsed().as_secs_f64();
    // S159zb —— 攒下来的辅音谷在这里刻(见 `valley_after_inverse`)。
    if !deferred_valley.is_empty() {
        let cls = if valley_adaptive() {
            // S159zb ⑵ —— **量了再补差额**(与 `apply_coda_lift` 同一条;见 `valley_adaptive`)。
            deferred_valley
                .iter()
                .map(|c| {
                    let (s0, e1) = (c.win[0].0, c.win[c.win.len() - 1].1);
                    let have = measured_valley_db(&audio, s0, e1);
                    ValleyCluster {
                        win: c
                            .win
                            .iter()
                            .map(|&(a, b, d)| (a, b, (d - have / valley_scale.max(1e-6)).clamp(0.0, d)))
                            .collect(),
                        env: c.env,
                    }
                })
                .collect::<Vec<_>>()
        } else {
            deferred_valley.clone()
        };
        apply_valley(&mut audio, &cls, valley_scale, emphasis_fade_samples(m.sample_rate), m.sample_rate);
    }
    let pre_norm_peak = audio.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    peak_normalize_to(&mut audio, 0.92, donor.map(|d| d.norm_peak_target));

    // S159b —— RVC 谱面臂到今天为止**一行秒表都没有**,所以「这条臂慢在哪」在日志里读不出来
    // (S159 那次只能按日志时间戳手算)。⛔ `skipped` 与 `len!=` 与 `skipped` 同行:前者是
    // S147 那次「收益静默减半」唯一被抓到的指纹,后者是「洞的长度算错了」唯一会露头的地方。
    let wall = t_render.elapsed().as_secs_f64();
    let secs = audio.len() as f64 / f64::from(m.sample_rate);
    tracing::info!(
        "[perf] score/rvc {secs:.1}s audio in {wall:.2}s (RTF {:.3}) · {} chunks · \
         s2cv {t_s2cv:.2}s ({:.0}%) · decode(net_g) {t_decode:.2}s ({:.0}%) · \
         inverse {t_inverse:.2}s ({:.0}%) · other {:.2}s · range_shift {range_shift:+} \
         · skipped {skipped}/{} · len!= {len_mismatch} · total{total_delta:+} \n         · predamp {preroll_damped}",
        if secs > 0.0 { wall / secs } else { 0.0 },
        chunks.len(),
        100.0 * t_s2cv / wall.max(1e-9),
        100.0 * t_decode / wall.max(1e-9),
        100.0 * t_inverse / wall.max(1e-9),
        (wall - t_s2cv - t_decode - t_inverse).max(0.0),
        chunks.len(),
    );
    Ok(SynthesisResult { audio, sample_rate: m.sample_rate, pre_norm_peak: Some(pre_norm_peak) })
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
// ─── S83 knife 6: voiceless ONSET emphasis (user: fast-run fricatives still not crisp enough) ───
//
// The duration (p75) and zero-fraction knives put the voiceless windows AT the training
// distribution — the remaining "not crisp" is an ENERGY ask, and duration can't buy it (fast-run
// vowels have no more frames to give). This is the deliberate SynthV-style engine emphasis: a
// small trapezoid gain on voiceless ONSET phone windows only (codas excluded — boosting word-final
// t/k would re-awaken the "词尾顿挫" the crown knife fixed). An explicit aesthetic lift, NOT a
// distribution fact — a per-track user knob since S83 knife 6b (VocalTrackParams.consonantEmphasis,
// the SynthV "consonant strength" analogue); this is the DEFAULT (mirrored by the frontend's
// DEFAULT_CONSONANT_EMPHASIS_DB) and what the model-audition path uses (no per-track context).
pub const DEFAULT_VOICELESS_ONSET_EMPHASIS_DB: f32 = 2.5;
/// Short edge ramp for the emphasis trapezoid (~5 ms) — windows are only 40-140 ms wide.
fn emphasis_fade_samples(sample_rate: u32) -> usize {
    (sample_rate as usize / 200).max(1)
}

/// Per-phone flag: a VOICELESS consonant that still LEADS TOWARD a nucleus within its SOURCE NOTE
/// EVENT — i.e. every syllable-onset position, pre-rolled or medial (a multi-syllable word on one
/// note flattens its boundaries: refined's medial f leads the second syllable and needs the same
/// bite). Codas (voiceless AFTER the event's last nucleus) never flag — boosting word-final t/k
/// would re-awaken the 词尾顿挫. Events without a nucleus (ʔ-only sokuon notes) flag nothing.
/// ★The run key is `arr.evt` (the source triple), NOT `note_to_phone`: consecutive same-pitch
/// notes merge into ONE pitch group, and anchoring on the merged group's nuclei either boosted
/// the previous word's codas (last-nucleus anchor) or missed every later onset in a repeated-
/// pitch run (first-nucleus anchor) — S83 review, both defects one root.
fn voiceless_onset_flags(arr: &ScoreArrays) -> Vec<bool> {
    let n = arr.phon.len();
    let mut flags = vec![false; n];
    let mut i = 0usize;
    while i < n {
        let mut j = i;
        while j + 1 < n && arr.evt[j + 1] == arr.evt[i] {
            j += 1;
        }
        if arr.note_pitch[i] > 0 {
            if let Some(last_nuc) = (i..=j).rev().find(|&x| super::score2cv::is_nucleus_phone(arr.phon[x])) {
                for x in i..last_nuc {
                    flags[x] = is_voiceless_phone(arr.phon[x])
                        && !super::score2cv::is_nucleus_phone(arr.phon[x]); // devoiced vowels are nuclei, not onsets
                }
            }
        }
        i = j + 1;
    }
    flags
}

/// Chunk-relative output-sample windows of FLAGGED phones (same proportional frame→sample map as
/// `chunk_sp_windows`; `flags` is the arr-level slice for this chunk's phone range).
fn chunk_flag_windows(chunk: &Chunk, out_len: usize, flags: &[bool]) -> Vec<(usize, usize)> {
    let t = chunk.t.max(1);
    let mut wins: Vec<(usize, usize)> = Vec::new();
    let mut cursor: i64 = 0;
    for (i, &d) in chunk.phone_dur.iter().enumerate() {
        let d = d.max(0);
        if d > 0 && flags.get(i).copied().unwrap_or(false) {
            let s = (cursor as f64 / t as f64 * out_len as f64).round() as usize;
            let e = ((((cursor + d) as f64) / t as f64) * out_len as f64).round() as usize;
            let e = e.min(out_len);
            if e > s {
                // coalesce contiguous flagged phones (an onset cluster like [s t] shares its
                // boundary sample by construction): one trapezoid per run — per-phone windows
                // would each ramp back to unity at the junction, carving a ~10 ms gain valley
                // inside one articulation gesture (S83 review).
                match wins.last_mut() {
                    Some(last) if last.1 == s => last.1 = e,
                    _ => wins.push((s, e)),
                }
            }
        }
        cursor += d;
    }
    wins
}

/// ⚙ 出厂默认 = true —— `UTAI_PREROLL_DAMP=0` 关。
///
/// ## 缺陷（用户 2026-08-28：「**很短的休止中间会漏出伪影**」）
///
/// `consonant_preroll` 把下一个音的辅音提前到休止里，而它分配的长度**远超**那个辅音真正需要的：
/// 实测（`diag_sp_phone_dur_after_preroll`，真实 `build_arrays_daw`）休止 140 ms 时
/// `k` 占 **120 ms**、`w` 占 **80 ms**，而真实的 k/w 只需 30-60 ms。
///
/// 音头前的能量剖面（鹅妈妈 × yachiyo × +7，`base`，相对该音稳态 dB）：
/// ```text
///            −120   −100    −80    −60    −40    −20      0
/// 元音        −50   −279   −279   −279   −270    −35     −0   ← 基线：音头前 40 ms 才起音
/// 清辅音     −274   −270    −45    −40    −25     −1     +8   ← 从 −80 ms 就开始出声
/// 近音(w/j) −274   −273    −42    −31    −26     −7     −0
/// ```
/// ⇒ 多出来的那 40 ms 落在**谱面的休止**里。用户报的坐标（全是「うぉ」= `w`+`o`）平台电平
/// **−15…−37 dB**，而他自己排除的 1:51.451（「这个不是空拍」）只有 **−45…−64** ——
/// **阴性对照是用户给的**，阈值就落在 −40 附近。
///
/// ## ⛔ 不是「割裂」
/// 用户同时怀疑「preroll 过长导致辅音提前发声、和元音割裂」。**实测方向相反**：
/// 辅音类的「辅音包→谷→元音起音」谷深 p50 **−0.7 dB**（基本单调爬升，没有谷），
/// 而用户报的三个谷深只有 **3.2 / 4.2 / 5.6**，他排除的那个反而 **28.2**（谷最深）。
/// ⇒ 问题是**平台太响**，不是断开。⇒ 这一刀压平台，**绝不挖谷**。
///
/// ## 做什么
/// 对**跟在 SP 之后的辅音音素窗**，只处理它的**前部**（`PREROLL_KEEP_MS` 之外的那段，
/// 紧贴音头的真辅音一个字节不动），量它相对该音稳态的电平，超过 `PREROLL_DAMP_THRESH_DB`
/// 才压，压到阈值为止，最多 `PREROLL_DAMP_MAX_DB`。梯形淡入淡出，只改增益、**不改时序**
/// （用户：辅音时序一个字节不许动）。
const PREROLL_DAMP_DEFAULT: bool = true;

/// 紧贴音头保留的真辅音长度（ms）—— 元音基线证明音头前 40 ms 本来就该有起音。
const PREROLL_KEEP_MS: f32 = 40.0;
/// 平台电平（相对该音稳态）超过这个才压。用户给的阴性对照落在 −45…−64，报的落在 −15…−37。
const PREROLL_DAMP_THRESH_DB: f32 = -40.0;
/// 最多压多少 dB（护栏，不是参数）。
const PREROLL_DAMP_MAX_DB: f32 = 18.0;

/// ⭐⭐ 增益的**斜率上限**（dB/ms）。用户 2026-08-28 实听第一版：「**3:40.861 应该是压出竖条纹了**」。
///
/// 第一版直接复用了 `emphasis_fade_samples`（**5 ms**）——而 `apply_emphasis` 只抬 2.5 dB，
/// 这一刀却压 10-18 dB ⇒ 窗边界上 **2.5 dB/ms** 的增益台阶 = 宽带瞬变 = 竖条纹。
/// 实测该处：平台压了 **17 dB**（顶到上限），Δ 在 **6 ms 内从 −15 跳回 0**。
///
/// ⇒ 淡化长度**随压幅走**（`fade = cut_db / 这个斜率`），
///   而且**压幅反过来受窗长约束**（`cut ≤ 斜率 × 半窗`）——窗短就少压，绝不缩短淡化。
/// ⚠ 这一条与 S163 的另一条同源血训并列：**任何逐格/逐窗增益都必须给斜率封顶**
///   （v17a 的 2 ms 阶梯把 16-24 kHz 抬了 21.5 dB，也是这个形状）。
const PREROLL_DAMP_MAX_SLOPE_DB_PER_MS: f32 = 0.6;

/// **左侧**（淡入那一头）的斜率上限。左扩只发生在「已经很安静」的地方
/// （`rest_gate` 常常已把那里归成数字静音），在 −60…−300 dBFS 上改增益不会变成可闻瞬变，
/// 所以这一头可以比右侧陡得多 —— 右侧挨着真辅音、信号在爬升，才是需要平缓的那一头。
/// ⛔ 第一版把两头平均（`(左+右)/2`）⇒ **安全的一头去限制了压幅**，
/// 实测压幅卡在 4-6 dB，用户实听「还是能听见」。
const PREROLL_DAMP_LEFT_SLOPE_DB_PER_MS: f32 = 2.0;

/// 左扩没拿到空间时（前一个音紧邻），允许在**窗内**借这么多 ms 做淡入 ——
/// 否则 `f_left` 会退化成几个样本 = 极陡 = 又一条竖条纹。
const PREROLL_DAMP_LEFT_BORROW_MS: f32 = 5.0;

/// 判定平台「有多响」时取的窗宽（ms）——**取压制区内最响的这么一窗**，不是全区平均。
/// ⛔ 与验收/耳判的口径对齐（那边量的是音头前 [−80,−45] 的平台）。
/// 全区平均会被压制区里安静的部分拉低：实测 1:44.097 平均 −36 dB 而峰段 −16 dB
/// ⇒ 闸只压 4 dB，而耳朵听到的是 −16 那一段。
const PREROLL_PEAK_WIN_MS: f32 = 35.0;

fn preroll_damp_enabled() -> bool {
    parse_preroll_damp(std::env::var("UTAI_PREROLL_DAMP").ok().as_deref())
}

fn parse_preroll_damp(v: Option<&str>) -> bool {
    match v.map(str::trim) {
        Some("0") => false,
        Some("1") => true,
        _ => PREROLL_DAMP_DEFAULT,
    }
}

/// **跟在 SP 之后的辅音音素**（= 被 `consonant_preroll` 提前到休止里的那些）。
///
/// 与 [`voiceless_onset_flags`] 同一个模式：在 `ScoreArrays` 层算（那里有音素字符串），
/// 调用点按 `chunk.start..chunk.end` 切片。
/// ⛔ 判「辅音」用的是 **不在 `tbl::VOWEL_SET` 里**，而不是「清辅音」——
/// 用户报的坐标全是「うぉ」，它分解成 **`w` + `o`**，`w` 是**浊近音**，
/// 按清浊分类会把整条链漏掉（S163 §45.3 踩过一次）。
fn preroll_consonant_flags(arr: &ScoreArrays) -> Vec<bool> {
    let n = arr.phon.len();
    let mut f = vec![false; n];
    let mut after_sp = false;
    for i in 0..n {
        let p = arr.phon[i];
        if p == "SP" {
            after_sp = true;
            continue;
        }
        if p == "AP" {
            after_sp = false; // 呼吸不是休止，别把它后面的辅音也算进来
            continue;
        }
        if !after_sp {
            continue;
        }
        if super::score2cv_tables::VOWEL_SET.contains(&p) {
            after_sp = false; // 到元音为止
        } else {
            f[i] = true;
        }
    }
    f
}

/// 压 preroll 辅音窗的**前部**。见 [`PREROLL_DAMP_DEFAULT`]。
///
/// ⛔ 只改增益、不改时序；紧贴音头的 `keep` 样本一个字节不动。
/// ⛔ 阈值以下的窗**逐位不变**（`gain == 1.0` 时直接跳过，不做恒等乘法）。
fn apply_preroll_damp(
    audio: &mut [f32],
    windows: &[(usize, usize)],
    keep: usize,
    thresh_db: f32,
    max_cut_db: f32,
    sample_rate: u32,
) -> usize {
    let mut hit = 0usize;
    for &(s, e0) in windows {
        let e0 = e0.min(audio.len());
        if e0 <= s {
            continue;
        }
        // 只处理「远离音头」的那段
        let e = e0.saturating_sub(keep);
        if e <= s {
            continue; // 辅音本来就短于 keep ⇒ 全是真辅音，不碰
        }
        let ms = |v: f32| ((v / 1000.0) * sample_rate as f32).round().max(1.0) as usize;
        // 参照：窗之后 50-150 ms 的稳态（= 那个音的元音）
        let rs = (e0 + ms(50.0)).min(audio.len());
        let re = (rs + ms(100.0)).min(audio.len());
        if re <= rs {
            continue;
        }
        let energy = |a: usize, b: usize| -> f64 {
            if b <= a {
                return 0.0;
            }
            audio[a..b].iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / (b - a) as f64
        };
        // ⛔⛔ **闸和它服务的判据必须同尺子**（S163 血训，这里踩过一次）：
        //    第一版 `pe` 取压制区 `[s,e]` 的**平均**能量，而压制区里大半是很安静的段
        //    ⇒ 平均被拉低到 −36 dB、判定「不太响」⇒ 只压 4 dB；
        //    而耳朵（和验收指标）听的是那段平台**最响**的部分（−16 dB）。
        //    ⇒ 改成取压制区内**最响的 `PREROLL_PEAK_WIN_MS` 窗**，与验收口径一致。
        let pw = ms(PREROLL_PEAK_WIN_MS).min(e - s);
        let pe = if pw >= 1 && e > s {
            let mut best = 0.0f64;
            let mut i = s;
            while i + pw <= e {
                let v = energy(i, i + pw);
                if v > best {
                    best = v;
                }
                i += (pw / 4).max(1);
            }
            best
        } else {
            energy(s, e)
        };
        let se = energy(rs, re);
        if !(se > 0.0) || !(pe > 0.0) {
            continue;
        }
        let rel_db = 10.0 * (pe / se).log10();
        if rel_db <= f64::from(thresh_db) {
            continue; // 平台本来就安静 ⇒ 逐位不变
        }
        // ⭐⭐ **淡入段伸进休止里** —— 那里本来就该静音，压什么都无所谓，
        //    所以不必占用辅音区的长度。这让压幅从「0.6 × 半窗」放宽到「0.6 × 窗长」。
        //    ⛔ 但**绝不能碰前一个音的释放**：向左只扩到「仍然很安静」的地方为止
        //    （阈值取平台电平再低 12 dB；一遇到更响的样本就停）。
        let quiet = (pe * 10f64.powf(-12.0 / 10.0)).sqrt();
        let mut s2 = s;
        let left_cap = ms(60.0);
        while s2 > 0 && s - s2 < left_cap {
            let v = f64::from(audio[s2 - 1]).abs();
            if v > quiet {
                break;
            }
            s2 -= 1;
        }
        // ⭐⭐⭐ **淡入淡出不对称** —— 两头的约束根本不同：
        // * **右侧**（挨着真辅音、信号在爬升）必须平缓 ⇒ 受 `..MAX_SLOPE..` 约束；
        // * **左侧**落在 `s2..s` 这段**已经很安静**（左扩的条件就是"安静"，
        //   而 `rest_gate` 常常已经把那里归成数字静音）⇒ 陡一点是安全的：
        //   在 −60…−300 dBFS 的地方改增益，改多少都不会变成可闻的瞬变。
        //
        // 第一版把两头平均（`(左+右)/2`）是错的：它让**安全的那一头去限制压幅**，
        // 实测压幅因此卡在 4-6 dB，而用户实听「还是能听见」。
        let right_ms = (e - s) as f64 / f64::from(sample_rate) * 1000.0;
        let left_ms = (s - s2) as f64 / f64::from(sample_rate) * 1000.0;
        // ⛔ 左侧也要有下界：左扩没拿到空间时（前一个音紧邻），淡入会退化成几个样本 = 极陡。
        //    窗内允许借 `LEFT_BORROW_MS` 做淡入 ⇒ 左侧可用 = 左扩到的 + 借的。
        let left_eff_ms = left_ms + f64::from(PREROLL_DAMP_LEFT_BORROW_MS);
        let by_slope = (f64::from(PREROLL_DAMP_MAX_SLOPE_DB_PER_MS) * right_ms)
            .min(f64::from(PREROLL_DAMP_LEFT_SLOPE_DB_PER_MS) * left_eff_ms);
        let cut_db = (rel_db - f64::from(thresh_db))
            .min(f64::from(max_cut_db))
            .min(by_slope);
        if cut_db <= 0.1 {
            continue; // 窗太短，压不了几个 dB ⇒ 不如不动（也就不会造边界台阶）
        }
        hit += 1;
        // 淡化长度**随压幅走**，两头各用各的斜率。
        let smp = |ms: f64| ((ms / 1000.0) * f64::from(sample_rate)).round().max(1.0);
        let f_right = smp(cut_db / f64::from(PREROLL_DAMP_MAX_SLOPE_DB_PER_MS));
        // 左侧：能用多少用多少，但不必比 `LEFT_SLOPE` 更缓（用不完就留着）。
        let f_left = smp(cut_db / f64::from(PREROLL_DAMP_LEFT_SLOPE_DB_PER_MS));
        for i in s2..e {
            // 左侧的上升沿从 `s2` 起（落在已经很安静的地方），右侧的下降沿在 `e` 收住。
            let (d_l, d_r) = ((i - s2) as f64, (e - 1 - i) as f64);
            let ramp = ((d_l / f_left).min(1.0)).min((d_r / f_right).min(1.0));
            // ⛔⛔ 在 **dB 域**插值，不是幅度域。
            //    幅度域线性（`1 + (gain−1)·ramp`）会让 **dB 斜率不均匀、末端更陡**：
            //    幅度 1→0.25 线性时前半只掉 4 dB、后半掉 8 dB ⇒ 实测 1.13 dB/ms，
            //    是斜率上限 0.6 的近两倍（判据 `preroll_damp_never_exceeds_the_slope_cap` 当场抓到）。
            audio[i] = (f64::from(audio[i]) * 10f64.powf(-cut_db * ramp / 20.0)) as f32;
        }
    }
    hit
}

/// Trapezoid gain over each window: edges ramp 1→gain over `fade` samples (no clicks), plateau at
/// `gain` in between; a window shorter than 2·fade peaks proportionally (mirror of apply_rest_gate).
fn apply_emphasis(audio: &mut [f32], windows: &[(usize, usize)], gain: f32, fade: usize) {
    let fade = fade.max(1) as f32;
    for &(s, e) in windows {
        let e = e.min(audio.len());
        for i in s..e {
            let edge = (i - s).min(e - 1 - i) as f32;
            let g = 1.0 + (gain - 1.0) * (edge / fade).min(1.0);
            audio[i] *= g;
        }
    }
}

// ─── S84 C 刀: chain-internal consonant VALLEY (fast-run 粘连 root-cause #4) ───
//
// Measured (S84, 116 fast-run boundaries, OpenUtau reference vs our render): real singing carves an
// energy valley at EVERY syllable boundary — stops ~36 dB deep, nasals ~17, taps ~14 — while our
// render leaves voiced-consonant boundaries nearly flat (3.3 dB group median vs 15.5) and 2-frame
// voiceless windows barely notch. f0 stays CONTINUOUS through these closures in real singing (the
// zero-fraction knife is honest about voicing), so the missing valley is an ENERGY fact the f0/uv
// stream cannot express — an output-domain gain valley is the honest treatment (rest-gate 近亲).
// Depths below are the MEASURED mix−render gap per consonant class (not the full real-singing
// depth — the render already notches partially), so scale 1.0 statistically lands the render on the
// reference; the offline twin of this stage (probe shapedA) was ear-validated by the user (S84).
// Codas are excluded (same 顿挫 guard as the emphasis knife); post-rest onsets are excluded (the
// rest itself is the valley). User knob = VocalTrackParams.consonantValley (scale ×depth, 0 = off
// bit-exact); this DEFAULT is what no-context paths (audition) use.
pub const DEFAULT_CONSONANT_VALLEY_SCALE: f32 = 1.0;
/// S84 anchors: per-class (mix − render) valley-depth gap in dB. Unmeasured classes get judgment
/// values in-family (laterals ride the nasal anchor: same "voiced continuous with mild dip" family).
fn valley_depth_db(p: &str) -> f32 {
    match p.chars().next() {
        // stops + affricates (ts/tɕ/dʑ/ɟʝ start with their stop half) + glottal ʔ + palatal c/ɟ
        // (JA きゃ/ぎゃ rows emit bare c/ɟ — S84 review caught ɟ falling to the fricative default)
        Some('p' | 't' | 'k' | 'b' | 'd' | 'ɡ' | 'c' | 'ɟ' | 'q' | 'ʈ' | 'ʔ') => 11.7,
        // nasals + laterals (incl. dark/retroflex ɫ/ɭ): the "voiced continuous with mild dip" family
        Some('m' | 'n' | 'ɲ' | 'ŋ' | 'ɴ' | 'l' | 'ʎ' | 'ɫ' | 'ɭ') => 11.4,
        Some('ɾ' | 'ɽ' | 'r') => 10.4,
        Some('s' | 'ʃ' | 'ɕ' | 'ʂ' | 'ç' | 'x' | 'h' | 'ɸ' | 'θ' | 'f' | 'z' | 'ʒ' | 'β' | 'ð'
            | 'v' | 'ʁ' | 'ɣ' | 'ɦ' | 'ʑ' | 'ʐ' | 'ʝ') => 5.1,
        // approximants/glides: barely a dip in real singing (measured 1.4 for w/j; ɹ/ɻ/ɰ ride along)
        Some('w' | 'j' | 'ɥ' | 'ɹ' | 'ɻ' | 'ɰ') => 1.4,
        _ => 5.1, // unclassified consonant: conservative fricative-level dip (exhaustiveness-pinned: 0 today)
    }
}

/// S161 —— [`valley_depth_db`] 的**真人标定**版本(见 [`parse_valley_human`] 的 doc)。
/// ⛔ 只有鼻/边与闪两个桶不同;其余每一个分支都必须**逐字**等于 [`valley_depth_db`],
/// 由 `the_human_table_differs_only_in_the_two_measured_buckets` 对全部 210 个音素逐个盯住。
fn valley_depth_db_human(p: &str, frames: i64) -> f32 {
    match p.chars().next() {
        Some('m' | 'n' | 'ɲ' | 'ŋ' | 'ɴ' | 'l' | 'ʎ' | 'ɫ' | 'ɭ') => VALLEY_NASAL_HUMAN_DB,
        Some('ɾ' | 'ɽ' | 'r') => valley_tap_human_db(frames),
        Some('b' | 'd' | 'ɡ' | 'ɟ' | 'ɢ') => valley_vstop_human_db(frames),
        _ => valley_depth_db(p),
    }
}

/// Per-phone valley depth (dB; 0.0 = untouched): consonant phones BEFORE the source event's last
/// nucleus (onsets + medials — voiced AND voiceless, unlike the emphasis flags), but only when the
/// boundary is CHAIN-INTERNAL — the emitted phone immediately before the consonant cluster belongs
/// to a SUNG event. After SP/AP (or at score start) the rest itself is the valley and the natural
/// release/attack must stay untouched (the offline twin skipped those too). Codas never flag
/// (词尾顿挫 guard, same as `voiceless_onset_flags`); run key = `arr.evt` (S83 review: pitch-group
/// anchoring misflags repeated-pitch runs).
fn boundary_valley_depths(arr: &ScoreArrays) -> Vec<f32> {
    let human = valley_human();
    let n = arr.phon.len();
    let mut depths = vec![0.0f32; n];
    let mut i = 0usize;
    while i < n {
        let mut j = i;
        while j + 1 < n && arr.evt[j + 1] == arr.evt[i] {
            j += 1;
        }
        if arr.note_pitch[i] > 0 {
            if let Some(last_nuc) = (i..=j).rev().find(|&x| super::score2cv::is_nucleus_phone(arr.phon[x])) {
                for x in i..last_nuc {
                    if super::score2cv::is_nucleus_phone(arr.phon[x]) {
                        continue; // medial vowels (più's i) and devoiced vowels are nuclei-like, not closures
                    }
                    // chain-internal test on the phone right before THIS consonant's CLUSTER: walk
                    // back over same-event non-nucleus mates (NOT over flag state — a post-rest
                    // cluster's skipped first member must not launder its later members into
                    // "chain-internal"). c0 > i ⇒ a medial cluster after a nucleus in the SAME
                    // event = chain-internal by construction; c0 == i ⇒ the event-leading cluster:
                    // look at the previous event's last emitted phone (SP/AP ⇒ the rest IS the
                    // valley; a sung phone — vowel or a hummed ɴ — ⇒ chain-internal).
                    let mut c0 = x;
                    while c0 > i && !super::score2cv::is_nucleus_phone(arr.phon[c0 - 1]) {
                        c0 -= 1;
                    }
                    let chain_internal = if c0 > i {
                        true
                    } else {
                        c0 > 0 && !matches!(arr.phon[c0 - 1], "SP" | "AP")
                    };
                    if chain_internal {
                        depths[x] = if human {
                            valley_depth_db_human(arr.phon[x], arr.phone_dur[x].max(0))
                        } else {
                            valley_depth_db(arr.phon[x])
                        };
                    }
                }
            }
        }
        i = j + 1;
    }
    depths
}

// ─── S97 ②a 刀: PHRASE-FINAL CODA RESTORE ──────────────────────────────────────────────────────
//
// 症状(用户 2026-08-02 点名 `bloom` 的 /m/、`all` 的 /l/):音素发出来了、帧数也对,但**没有能量**。
// 实测:凡「下一个音符是休止」的音符,末尾都有一段单调衰减(0-10 帧,中位 4-6),深到平台以下
// 20-35 dB,**且与音素身份无关**(以元音收尾的句尾音符同样衰)。coda 按构造就是最后一个音素,
// 所以比这段斜坡短的 coda 被整个吞掉。它**不是**任何后处理造成的(rest_gate 只动 SP 窗内、
// 强调只吃清音 onset、谷刀按构造跳过 coda、vol ADSR 只在 vol_embedding 模型上),是 cv/解码器产出的。
//
// ★目标值不是 Synthesizer V。S97 实测 SV 把词尾辅音压得**远比真人狠**(SV 给 `and` 的 n/d 压
// 13-20 dB,真人只压 1.9 / 4.5)。目标取自 `Much-Better-S2H/_onnx_derisk/coda_ref_upstream.json`
// —— GTSinger **上游标注 + 数据集源音频**(全量 4827 clip,从没经过我们的对齐器),词尾辅音相对
// **自己那个词的元音** 的电平。取 **p25 而不是 p50**:这是个**单向**(只抬不压)修正,瞄准中位会
// 把一半样本推到真人之上。
//
// 只对**响音**开火:同一张表说真人**确实**把词尾阻塞音压下去(K −15.3 / F −15.3 / T −9.3 / Z −9.9),
// 那不是缺陷,是对的;而 `dears` 的 /z/ −15.5 dB 正落在真人区间里,抬它才是新缺陷。
//
// ⚠ zh/ja 由构造 no-op:实测 ja 1215 个有声音符、三条 UTAU 别名轨 489 个音符里,**位于末核之后的
// 音素 = 0 个**,所以这一刀在那五条泳道上一次都不会开火(用逐字节泳道证明,不靠论证)。
fn coda_sonorant_target_db(p: &str) -> Option<f32> {
    // upstream p25 (dB re its own word's vowel), n = 3293 / 5584 / 1940 / 2538 / 3366
    match p {
        "l" | "ɫ" | "ɭ" | "ʎ" => Some(-3.1),
        "n" | "ɲ" | "n̪" => Some(-4.4),
        "m" => Some(-5.0),
        "ŋ" => Some(-3.5),
        "ɹ" | "ɻ" | "r" => Some(-3.9),
        _ => None,
    }
}

/// The most this stage may add, in dB. A policy cap, not a measurement: the worst real gap on the
/// user's own material is `all`'s /l/ at 10.4 dB, but that phone's last frame sits at −48 dBFS
/// absolute, where more gain is amplified decoder noise rather than a consonant. 9 dB covers
/// `bloom` (3.1) and `gain` (4.3) with room to spare and stops short of the noise floor case.
const CODA_LIFT_MAX_DB: f32 = 9.0;

/// Per-phone `(target_db, nucleus_phone_index)` for every PHRASE-FINAL sonorant coda; `None` = this
/// stage does not touch the phone. Phrase-final = the next emitted phone is a rest/breath (or the
/// array ends) — mid-phrase codas are already right (measured −1.1 dB median vs the −1.9 target),
/// it is only the ones sitting inside the release ramp that collapse.
fn phrase_final_coda_lifts(arr: &ScoreArrays) -> Vec<Option<(f32, usize)>> {
    let n = arr.phon.len();
    let mut out = vec![None; n];
    let mut i = 0usize;
    while i < n {
        let mut j = i;
        while j + 1 < n && arr.evt[j + 1] == arr.evt[i] {
            j += 1;
        }
        let chaining = super::g2p::Lang::from_id(arr.lang[i])
            .is_some_and(super::score2cv::consonant_chaining_language);
        // phrase-final: nothing after this event, or the next emitted phone is a rest/breath
        let phrase_final = j + 1 >= n || matches!(arr.phon[j + 1], "SP" | "AP");
        if chaining && phrase_final && arr.note_pitch[i] > 0 {
            if let Some(nuc) = (i..=j).rev().find(|&x| super::score2cv::is_nucleus_phone(arr.phon[x])) {
                for x in nuc + 1..=j {
                    if let Some(t) = coda_sonorant_target_db(arr.phon[x]) {
                        out[x] = Some((t, nuc));
                    }
                }
            }
        }
        i = j + 1;
    }
    out
}

/// Measure each eligible coda against its own nucleus in the DECODED audio and lift it toward the
/// upstream target, bounded by `CODA_LIFT_MAX_DB` and never attenuating. Measuring rather than
/// applying a fixed curve is deliberate: the release ramp's length varies 0-10 frames, so a fixed
/// compensation curve would over- or under-correct depending on where the ramp happened to land.
fn apply_coda_lift(
    audio: &mut [f32],
    chunk: &Chunk,
    lifts: &[Option<(f32, usize)>],
    fade: usize,
) {
    let t = chunk.t.max(1);
    let out_len = audio.len();
    let span = |from: i64, dur: i64| -> (usize, usize) {
        let s = (from as f64 / t as f64 * out_len as f64).round() as usize;
        let e = (((from + dur) as f64) / t as f64 * out_len as f64).round() as usize;
        (s.min(out_len), e.min(out_len))
    };
    let rms = |a: &[f32], (s, e): (usize, usize)| -> f32 {
        if e <= s {
            return 0.0;
        }
        (a[s..e].iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / (e - s) as f64).sqrt() as f32
    };
    // chunk-local phone start offsets
    let mut starts = Vec::with_capacity(chunk.phone_dur.len());
    let mut cur = 0i64;
    for &d in &chunk.phone_dur {
        starts.push(cur);
        cur += d.max(0);
    }
    let mut wins: Vec<(usize, usize, f32)> = Vec::new();
    for (k, &d) in chunk.phone_dur.iter().enumerate() {
        let Some(Some((target_db, nuc_global))) = lifts.get(chunk.start + k).copied() else {
            continue;
        };
        if d <= 0 {
            continue;
        }
        // the nucleus must live in THIS chunk (it always does — chunks cut at SP, and an event
        // never spans a rest — but a defensive skip beats a wrong reference window).
        if nuc_global < chunk.start || nuc_global >= chunk.end {
            continue;
        }
        let nk = nuc_global - chunk.start;
        let cw = span(starts[k], d);
        let nw = span(starts[nk], chunk.phone_dur[nk].max(0));
        let (rc, rn) = (rms(audio, cw), rms(audio, nw));
        if rc <= 1e-7 || rn <= 1e-7 {
            continue; // no signal to lift (or no reference) — leave it alone
        }
        let measured = 20.0 * (rc / rn).log10();
        let want = (target_db - measured).clamp(0.0, CODA_LIFT_MAX_DB);
        if want > 0.05 {
            wins.push((cw.0, cw.1, 10f32.powf(want / 20.0)));
        }
    }
    for (s, e, gain) in wins {
        apply_emphasis(audio, &[(s, e)], gain, fade);
    }
}

/// S159zb —— `UTAI_MG_VALLEY_AFTER=0/1`:**donor 那一遍**的辅音谷改在
/// [`apply_range_inverse`] **之后**施加。**默认 0 = 与今天逐位相同。**
///
/// ## ⛔ 为什么(实测的三级串联,S159za/zb)
///
/// 用户 2026-08-22 点名的 12 处「咔哒 + 面状」,根因是**三级串联**,每一级都在自己的上界之内:
/// ⑴ [`apply_valley`] 在 **donor 上**刻一个 11.4(鼻/边)/ 11.7(塞)dB 的辅音谷;
/// ⑵ **TD-PSOLA 是一台凹陷放大器**,而且在 `ratio 2.0`(= |位移| 12 半音)处上台阶
///    —— 实测把「已经存在」的凹陷再加深:浅窗 −3/−5/−7 只有 **−0.94 / −1.00 / −1.47 dB**,
///    深窗 −12/−17 是 **−4.34 / −4.16**(最坏 −17.3 / −20.3);
/// ⑶ 导出 PCM_16 把最深的那个削成精确零。
///
/// 单变量消融(六条臂 `.plan.json` 逐字段相同,独立重渲的噪声地板 ≤0.5 dB):
/// `UTAI_MG_VALLEY=0` 让那 6 处凹陷变浅 **+13.2 / +9.1 / +9.4 / +25.3 / +200 / +11.8 dB(6/6 同号)**,
/// 而 9-13 dB **正好等于 `valley_depth_db` 的 11.4 / 11.7**。
///
/// ## 为什么是「挪到之后」而不是「关掉」或「调浅」
///
/// ⛔ 用户明确要求**不许直接关掉**:这把刀是为咬字做的(S84,mix−render 的实测差)。
/// ⭐ 而 [`apply_valley`] 自己的 doc 写着它是 **an output-domain gain valley**
///    —— 输出域的乘性包络。**今天它却被施加在 PSOLA 之前**,于是 PSOLA 把它当成信号去放大。
/// ⇒ 挪到逆变换之后:**咬字收益一分不少**(同样的窗、同样的深度、同样的时间位置,
///    因为 TD-PSOLA 不改时长),而 PSOLA **再也看不到那个谷**。
///
/// ⚠ 只在 **donor 那一遍**(`range_shift != 0`)改;base 那一遍的逆变换是恒等,
///    保持在循环里施加 ⇒ 未开扩展的渲染**逐位不变**。
/// S159zb —— `UTAI_MG_VALLEY_ADAPT=0/1`:辅音谷改成**刻到目标深度**,而不是固定衰减。
/// **默认 0 = 与今天逐位相同。**⚠ 只在 [`valley_after_inverse`] 也开着时才有意义(要量成品)。
///
/// ## ⛔ 为什么:深窗里我们在【重复计数】
///
/// [`apply_valley`] 是**加法式**的固定衰减(`valley_depth_db`:鼻/边 11.4、塞 11.7 dB),
/// 而它当年(S84)是在**没有移调**的渲染上标定的 —— 那时「模型自己的辅音凹陷 + 这一刀」
/// 正好落在实测的 mix−render 参照上。
///
/// ⭐ 但在深位移的 donor 上,**模型自己的凹陷已经更深了**。实测(炉心融解 +7 × yachiyo,
/// 用户点名的 6 处,2 ms 包络的局部中位 − 谷底,中位):
///
/// | | 谷深 |
/// |---|---|
/// | 出厂(有 valley) | **27.2 dB** |
/// | `UTAI_MG_VALLEY=0` | **16.6 dB** |
/// | 差 = 这一刀贡献的 | **10.6 dB** ≈ 设计值 11.4/11.7 |
///
/// ⇒ ⭐ **辅音谷是线性穿过 PSOLA 的(没被放大)**,但**总深度 27.2 dB 远超 S84 的参照(11-15 dB)**
///   —— 因为那 16.6 dB 里已经有模型自己的凹陷 + PSOLA 对它的放大(S159zb 的 M2:深窗 4-20 dB)。
///   **同一个谷被算了两次。**
///
/// ## 形状:与 [`apply_coda_lift`] 同一条(量了再补差额)
///
/// `want = clamp(target − measured, 0, target)`。
/// * `measured` = 该簇内 2 ms 包络的**局部中位 − 谷底**(与上面那张表同一把尺子);
/// * `target` = 该簇成员的类深度(`valley_depth_db`,不变);
/// * 已经够深的地方 `want → 0` ⇒ **一个字节都不动**。
///
/// ⛔ 这不是「把刀调浅」:在没移调的渲染上 `measured` 很小 ⇒ `want ≈ target` ⇒ 行为与今天一样。
///   它只在**已经过深**的地方收手 —— 也就是深救援窗里的鼻音/浊塞音,正是用户报缺陷的地方。
fn valley_adaptive() -> bool {
    parse_valley_adaptive(std::env::var("UTAI_MG_VALLEY_ADAPT").ok().as_deref())
}

/// S159zb —— 一个簇里**已经有多深的谷**(2 ms RMS 包络:局部中位 − 谷底,dB)。
/// ⚠ 与 `TESTING\s159za_*` 那批读数用的是同一把尺子,所以 doc 里的表可以直接对上。
fn measured_valley_db(audio: &[f32], s: usize, e: usize) -> f32 {
    let (s, e) = (s.min(audio.len()), e.min(audio.len()));
    if e <= s {
        return 0.0;
    }
    // 局部参照取簇两侧各 60 ms(够长到跨过整个辅音,够短到跟得上乐句的强弱)。
    // 参照窗：簇两侧各 ~60 ms（48k 下 2880 样本）。⚠ 写成常数而不是按 sr 算，是因为它只决定参照窗多宽。
    const PAD: usize = 2880;
    let pad = PAD.min(audio.len());
    let (a0, b0) = (s.saturating_sub(pad), (e + pad).min(audio.len()));
    // 2 ms @48k（44.1k 下 2.18 ms）—— 与 TESTING\s159za_* 那批读数同一个窗宽。
    const H: usize = 96;
    let h = H;
    let db = |from: usize, to: usize| -> Vec<f32> {
        let mut v = Vec::new();
        let mut i = from;
        while i + h <= to {
            let r = (audio[i..i + h].iter().map(|x| x * x).sum::<f32>() / h as f32).sqrt();
            v.push(20.0 * (r + 1e-12).log10());
            i += h;
        }
        v
    };
    let mut ref_v = db(a0, b0);
    if ref_v.len() < 3 {
        return 0.0;
    }
    ref_v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = ref_v[ref_v.len() / 2];
    let floor = db(s, e).into_iter().fold(f32::INFINITY, f32::min);
    if floor.is_finite() {
        (med - floor).max(0.0)
    } else {
        0.0
    }
}

fn valley_after_inverse() -> bool {
    parse_valley_after(std::env::var("UTAI_MG_VALLEY_AFTER").ok().as_deref())
}

/// S161b —— 一个簇 = 一个闭塞动作 + **它的形状**。`frac == 1.0` ⇒ 今天的整窗矩形。
/// ⚠ 一个簇只有一套形状:取**深度最大的那个成员**的形状(单成员簇 = 精确;
/// 混合簇如 EN `[k w]` 由最深的那个说了算,与「一个簇一个动作」的既有契约一致)。
#[derive(Clone, Debug)]
pub(crate) struct ValleyCluster {
    pub win: Vec<(usize, usize, f32)>,
    /// `None` = 整窗矩形(今天的形状);`Some` = 真人包络,见 [`valley_shape_human`]。
    pub env: Option<&'static [f32; VALLEY_ENV_N]>,
}

/// 逐音素的谷形状,与 [`boundary_valley_depths`] 并列。出厂开时走 [`valley_shape_human`]。
fn boundary_valley_shapes(arr: &ScoreArrays) -> Vec<Option<&'static [f32; VALLEY_ENV_N]>> {
    let human = valley_human();
    arr.phon.iter().map(|p| if human { valley_shape_human(p) } else { None }).collect()
}

/// Chunk-relative CLUSTERS of contiguous depth>0 phones, each member keeping its OWN class depth.
/// One cluster = one closure gesture: apply_valley ramps only at the cluster's outer edges and
/// blends depth ACROSS internal junctions — per-phone windows would ramp a gain BUMP back to unity
/// at the junction (the valley-polarity twin of the S83 emphasis review bug), while a single
/// max-depth window would over-carve the shallower member (an EN [k w] cluster must not sink the
/// 1.4 dB glide to the stop's 11.7 — S84 review, per-class calibration is the stage's contract).
fn chunk_valley_clusters(
    chunk: &Chunk,
    out_len: usize,
    depths: &[f32],
    shapes: &[Option<&'static [f32; VALLEY_ENV_N]>],
) -> Vec<ValleyCluster> {
    let t = chunk.t.max(1);
    let mut clusters: Vec<ValleyCluster> = Vec::new();
    let mut deepest: Vec<f32> = Vec::new();
    let mut cursor: i64 = 0;
    for (i, &d) in chunk.phone_dur.iter().enumerate() {
        let d = d.max(0);
        let depth = depths.get(i).copied().unwrap_or(0.0);
        let env = shapes.get(i).copied().flatten();
        if d > 0 && depth > 0.0 {
            let s = (cursor as f64 / t as f64 * out_len as f64).round() as usize;
            let e = ((((cursor + d) as f64) / t as f64) * out_len as f64).round() as usize;
            let e = e.min(out_len);
            if e > s {
                let joins = clusters.last().map(|cl| cl.win.last().map(|w| w.1) == Some(s)).unwrap_or(false);
                if joins {
                    let last = clusters.last_mut().expect("joins implies non-empty");
                    last.win.push((s, e, depth));
                    // 形状由最深的成员说了算(见 `ValleyCluster` 的 doc)。
                    if depth > *deepest.last().expect("parallel to clusters") {
                        last.env = env;
                        *deepest.last_mut().expect("parallel") = depth;
                    }
                } else {
                    clusters.push(ValleyCluster { win: vec![(s, e, depth)], env });
                    deepest.push(depth);
                }
            }
        }
        cursor += d;
    }
    clusters
}

/// Trapezoid ATTENUATION per CLUSTER: each sample's target depth = its owning member's class depth,
/// linearly blended (dB domain) across internal junctions over `fade` samples centered on the
/// junction (no step-click, no unity bump), then scaled by the cluster-outer edge ramp (0→plateau
/// over `fade`). `scale` multiplies depth in the dB domain.
fn apply_valley(audio: &mut [f32], clusters: &[ValleyCluster], scale: f32, fade: usize, _sample_rate: u32) {
    let fade_f = fade.max(1) as f32;
    let half = (fade.max(1) / 2).max(1);
    for c in clusters {
        let cl = &c.win;
        let Some(&(s0, ..)) = cl.first() else { continue };
        let Some(&(.., e1, _)) = cl.last() else { continue };
        let e1 = e1.min(audio.len());
        if e1 <= s0 {
            continue;
        }
        // S161d —— 真人包络:把簇窗归一到 10 格,逐样本线性插值。`None` 退回整窗矩形(逐位相同)。
        let n = e1 - s0;
        for (wi, &(s, e, depth)) in cl.iter().enumerate() {
            let e = e.min(audio.len());
            for i in s..e {
                let mut d = depth;
                if wi > 0 && (i - s) < half {
                    let w = 0.5 + (i - s) as f32 / fade_f;
                    d = cl[wi - 1].2 * (1.0 - w) + depth * w;
                } else if wi + 1 < cl.len() && (e - 1 - i) < half {
                    let w = 0.5 + (e - 1 - i) as f32 / fade_f;
                    d = cl[wi + 1].2 * (1.0 - w) + depth * w;
                }
                let ramp = match c.env {
                    None => {
                        let edg = (i - s0).min(e1.saturating_sub(1 + i)) as f32;
                        (edg / fade_f).min(1.0)
                    }
                    Some(env) => {
                        // 10 格的格心在 (k+0.5)/10;两端各自夹住,不外推。
                        let u = (i - s0) as f32 / n as f32 * VALLEY_ENV_N as f32 - 0.5;
                        let kf = u.floor();
                        let t = u - kf;
                        let k = kf as isize;
                        let a = env[k.clamp(0, VALLEY_ENV_N as isize - 1) as usize];
                        let b = env[(k + 1).clamp(0, VALLEY_ENV_N as isize - 1) as usize];
                        let w = (a + (b - a) * t).clamp(0.0, 1.0);
                        // ⛔⛔ S161e —— **外缘淡化不许省**。模板首格是 0.82,直接铺 ⇒ 簇窗起点是一个
                        //    **一样本的 ~10 dB 台阶** = 宽带咔哒。实测(鹅妈妈原 key,211 个簇):
                        //    >16 kHz 在窗**起点**的尖峰 **+27.15 dB**(终点 +4.87),而矩形/窄槽/谷全关
                        //    都是 −2 dB(没有尖峰)—— 用户在频谱图 16 k 以上一眼看到「一堆细竖线」。
                        //    ⇒ 与矩形路径同一条外缘斜坡,乘在权重上:窗缘 w=0(增益 1.0,连续)。
                        let edg = (i - s0).min(e1.saturating_sub(1 + i)) as f32;
                        w * (edg / fade_f).min(1.0)
                    }
                };
                audio[i] *= 10f32.powf(-(d * scale * ramp) / 20.0);
            }
        }
    }
}

// ─── S84 B 刀: short-phone retrieval weights (② score path only) ───
//
// The S84 measurement indicted RVC index retrieval on SHORT phones: transitional cv gets pulled
// toward WRONG neighbours — ま's 4-frame /a/ measured F1 646 with retrieval vs 938 without
// (closed-vowel直拉, cover 1070), and retrieval deepened the pre-A-knife ko1 syllable dropout
// ~20 dB. The criterion is per-PHONE duration (a first evt-total design missed ま entirely —
// its un-robbed note totals 6 frames while the harmed vowel itself is 4): frames of sung phones
// with dur ≤ FAST_INDEX_MAX_PHONE_FRAMES weigh 0 (no retrieval — their cv never reaches a stable
// target the index could match), longer phones and rests weigh 1 = unchanged (5+ frame ≈ 100 ms+
// stable vowels are the regime retrieval demonstrably helps; the user's global A/B found
// retrieval-off barely audible, so the cost of 0-weighting short phones is bounded while the
// measured articulation win is real).
const FAST_INDEX_MAX_PHONE_FRAMES: i64 = 4;
fn fast_index_weights(arr: &ScoreArrays) -> Vec<f32> {
    let mut w = Vec::new();
    for i in 0..arr.phon.len() {
        let d = arr.phone_dur[i].max(0);
        let fast = arr.note_pitch[i] > 0 && d <= FAST_INDEX_MAX_PHONE_FRAMES;
        w.extend(std::iter::repeat(if fast { 0.0f32 } else { 1.0 }).take(d as usize));
    }
    w
}

const REST_GATE_FADE_MS: f32 = 40.0;

/// ⚙ 出厂默认 = true —— `UTAI_REST_GATE_SHRINK=0` 关回固定 40 ms 那一版。
///
/// 见 [`apply_rest_gate`] 里那一段：固定 fade 让 **SP 窗 < 2·fade(80 ms)** 的休止
/// **中心结构上归不了零**，而 `consonant_preroll` 会把 SP 音素本身压短
/// ⇒ 谱面 140 ms 的休止常常只剩 60-80 ms 的 SP 窗 ⇒ 门关不严。
/// ⛔ 长窗(n ≥ 2·fade+1)**逐位不变**，判据 `rest_gate_is_bit_identical_on_windows_long_enough_for_the_full_fade`。
const REST_GATE_SHRINK_DEFAULT: bool = true;

/// See [`REST_GATE_SHRINK_DEFAULT`].
fn rest_gate_shrink() -> bool {
    parse_rest_gate_shrink(std::env::var("UTAI_REST_GATE_SHRINK").ok().as_deref())
}

fn parse_rest_gate_shrink(v: Option<&str>) -> bool {
    match v.map(str::trim) {
        Some("0") => false,
        Some("1") => true,
        _ => REST_GATE_SHRINK_DEFAULT,
    }
}

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
    apply_rest_gate_with(audio, windows, fade, rest_gate_shrink())
}

fn apply_rest_gate_with(audio: &mut [f32], windows: &[(usize, usize)], fade: usize, shrink: bool) {
    let fade0 = fade.max(1);
    for &(s, e) in windows {
        let e = e.min(audio.len());
        if e <= s {
            continue;
        }
        // ⭐⭐⭐ S163 —— **fade 按窗长收缩**,否则短休止的中心**结构上归不了零**。
        //
        // `keep = 1 − edge/fade` 要到 0 需要 `edge ≥ fade`,而 `edge` 是到最近边缘的距离
        // ⇒ 窗长必须 ≥ `2·fade`(出厂 `REST_GATE_FADE_MS = 40` ⇒ **80 ms**)。
        // 而 `chunk_sp_windows` 取的是 **SP 音素自己的 `phone_dur`**,`consonant_preroll`
        // 把下一个音的辅音提前 ⇒ **SP 音素本身被压短** ⇒ 谱面 140 ms 的休止,SP 窗常常只剩
        // 60-80 ms ⇒ 门永远关不严,休止中间**漏出模型渲的东西**。
        //
        // 实测(鹅妈妈 × yachiyo × +7,`base` 层,休止中段 0.35-0.65 相对相邻音稳态):
        //
        // | 谱面休止时长 | 中段最安静点 | 中段 p50 | >−40 dB 的比例 |
        // |---|---|---|---|
        // | 0-150 ms | **−48 dB** | **−43.8** | **38%** |
        // | 150-250 ms | −282 dB | −274.9 | 10% |
        // | 250-400 ms | −281 dB | −279.3 | 6% |
        // | 400-1000 ms | −283 dB | −283.4 | 0% |
        //
        // ⇒ 长休止能到 **−280 dB(数字静音)**,短休止只到 −48 —— 分界线正是 150 ms 附近。
        // 用户 2026-08-28:「**很短的休止中间会漏出伪影**」;他报的三个坐标
        // (1:44.098 / 1:48.431 / 1:57.094)在这把尺子上是 **96% / 91% / 89% 分位**,
        // 而他自己排除的 1:51.451(「这个不是空拍」)只有 **53%** —— 阴性对照是他给的。
        //
        // ⛔ 为什么这样改是安全的:
        // * **长窗逐位不变** —— `(e−s)/2 ≥ fade` 时 `min` 不起作用;
        // * **辅音时序一个字节不动** —— SP 窗里没有 preroll 提前进来的辅音,
        //   那是**下一个音的音素**,不在 `chunk_sp_windows` 里;
        // * 淡化仍是**线性连续**的(不是阶跃),窗再短也不会自己造出咔哒。
        // ⛔ `edge` 的最大值是 `(n−1)/2`（`min(i−s, e−1−i)`），不是 `n/2` ——
        //    收缩到 `n/2` 会差一个样本，中心停在 `keep = 1/fade` 而不是 0（判据抓到过）。
        // ⛔ `edge` 的最大值是 `(n−1)/2`（`min(i−s, e−1−i)`），不是 `n/2`。
        // ⛔⛔ 而且**收缩到 `edge_max` 只会让【中心一两个样本】到 0，不是一段平底** ——
        //    那样连「≥2 ms 的零区」都测不出来（S163 实测：两臂零区间逐段完全相同）。
        //    ⇒ 收缩到 `edge_max/2`，让中间**一半窗长**是真正的零。
        let edge_max = ((e - s).saturating_sub(1)) / 2;
        let fade = if shrink { fade0.min((edge_max / 2).max(1)) } else { fade0 } as f32;
        for i in s..e {
            let edge = (i - s).min(e - 1 - i) as f32;
            let keep = (1.0 - edge / fade).max(0.0);
            audio[i] *= keep;
        }
    }
}

/// Micro-fade a chunk seam: linearly fade the tail of the accumulated audio and the head of the
/// incoming chunk over ~5 ms each. Sample counts are untouched (never an overlap-shift — the stem
/// must stay tick-aligned to the DAW timeline); the fades only mask the waveform discontinuity of
/// two independently decoded chunks.
///
/// ⛔⛔ S161f —— **以前只在 `hard_seam` 上跑,而那条注释里的「SP seams are silence and skip this」
/// 是错的**。`chunk_at_sp` 切在一个完整 SP **之后**,两侧是**两次独立解码**的缓冲,直接
/// `extend_from_slice` 硬拼 —— 休止里不是数字静音,是两段**不同的**极低电平信号,拼接处
/// 就是一个波形台阶 = 宽带咔哒。
///
/// 实测(鹅妈妈原 key,>18 kHz 盲搜全曲,突出度 >20 dB 的瞬变 32 个):**84-87% 落在 SP 边
/// 20 ms 以内**,而且**把辅音谷整个关掉读数一模一样** ⇒ 与谷无关,是这条接缝。
/// 用户在频谱图 18 kHz 以上看到的「一堆细竖线」的主体就是它。
///
/// ⇒ 现在**每一条 chunk 接缝都跑**。代价:接缝落在休止里 ⇒ 淡化的是两段 −60…−80 dBFS 的信号;
/// `hard_seam`(语言切,落在浊音正中)行为与以前**逐位相同**(同一个函数、同一个 5 ms)。
fn seam_fade(audio: &mut [f32], wav: &mut [f32], sample_rate: u32) {
    // ⛔ S161g —— **第一块没有左侧 ⇒ 那不是接缝**。S161f 把这个函数改成无条件调用之后,
    //    ci==0 上 `audio` 还是空的,而下面 `wav` 那半段照跑 ⇒ **整首歌的头 5 ms 被淡入**。
    //    以前只在 `hard_seam` 上跑,结构上碰不到这一格;判据 `the_first_chunk_is_never_faded_in`。
    if audio.is_empty() {
        return;
    }
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
/// zone); this shifts the decoded audio back by `-range_shift` semitones through the single
/// execution point vocal_range::apply_inverse (Signalsmith — no f0 guide needed). shift 0 /
/// empty ⇒ untouched (tier 1/2: in-comfort renders NEVER pass through here — bit-parity by
/// construction).
///
/// S159 —— `keep` = 这一遍的 donor 里**会被拼回歌里**的那几段样本(空 = 整条 = 今天)。
/// 逆变换只在这些段上跑,其余的岛原样透传 —— 因为那些样本在拼接层一个也读不到。
/// ⛔ 它不在这里从 `DonorCtx.windows` 现算,理由写在 `DonorCtx::keep_samples` 的 doc 里。
/// ⚙ 出厂默认 = **0 = 关 = 逐位不变**(S160k 差点翻,被自己的全曲对拍拦下,见下面「为什么没翻」)。
///
/// **donor 那一遍**里,把我们已经
/// 写成 `note_hz == 0` 的那些帧上、模型仍然吐出来的**有调成分**滤掉。`UTAI_MG_UVGATE_K` = 截止
/// 频率相对参照基频的倍数(默认 1.5)。
///
/// ## ⛔ 它治的是什么(用户 2026-08-24 点名的 0:47.229 那声咔哒)
/// 逐阶段定位(窗 47.200-47.252,喂给 PSOLA 的 `note_hz` 在这 60 ms 上**是 0** ——
/// ひ 的 /h/ 被 `consonant_preroll` 前置、被 `zero_voiceless_frames` 置零):
/// | 阶段 | 480-540 Hz(donor 档) |
/// |---|---|
/// | `base`(不救) | **0.6 dB** |
/// | **`donor_pre`(逆变换【前】)** | **23.3 dB @528 Hz** |
/// | `donor_post`(逆变换后) | 18.6 |
/// | 成品 | 28.8 |
/// ⇒ **那个音在 donor 还没进 PSOLA 之前就已经在了** —— 不是拼接、不是 PSOLA、也不是辅音本身。
/// 它是 S159zzs 结案的那族「**模型把清音辅音渲成半浊化**」。
///
/// ## ⭐ 但 S159zzs 的「不可闻」在这里**不成立**,而这条是本场最该记住的
/// 那次的结论是「不可闻 / 模型侧 / 渲染层修不动」。**在救援窗里它变得可闻了**:
/// 那个半浊化的音落在 **donor 的音高**上,比目标**低 9-15 个半音**;
/// 而 `note_hz == 0` 恰恰意味着 PSOLA **结构上不会去抬它**(那些帧在浊音岛之外)。
/// ⇒ **半浊化 × 深救援 = 一声听得见的咔哒。**
/// ⛔ 实测两把现成的刀都够不着它:`UTAI_PSOLA_BRIDGE` 30→120→250(28.8 / 29.1 / 29.1)与
/// `UTAI_MG_VALLEY_ADAPT=1`(28.5)—— 因为它们改的是**怎么搬**,而这个成分**本来就不该在**。
///
/// ## 做法与两条硬约束
/// 对每一段 `note_hz == 0` 的连续帧:截止频率 = `k ×`(该段**最近的浊音帧**的 `note_hz`),
/// 夹在 `[200, 2500]` Hz;在**该段的样本范围**上做零相位高通(双二阶前向+反向),
/// 段外一个样本都不碰。
/// ⛔ **两端必须交叉淡化**(默认 4 ms):硬切换会**造出一声新的咔哒** —— 这一族的解药不许
///    自己成为同一族的病因。
///
/// ## ⛔⛔ 为什么 S160k 没有把它翻成出厂(用户已耳判确认它除掉了咔哒,我仍然没翻)
/// 用户确认「gate 和 gate2 里面那个位置的咔哒现在确实没了」之后,我在**全曲**上对拍
/// `br120` vs `br120+门k1.5`(东雪莲 × 炉心融解 +7,同一二进制 `2f68e47670f5`,10 ms 格,
/// 只统计 >−40 dB 的有声格 18868 个):
/// | 门相对 br120 | 格数 | 占比 |
/// |---|---|---|
/// | 低 >3 dB | 142 | 0.75% |
/// | 低 >6 dB | 76 | 0.40% |
/// | 低 >10 dB | 30 | 0.16% |
/// **>6 dB 的连续段 46 条 / 共 0.76 s**,最长的三条:**1:19.950(40 ms,−11.6 dB)**、
/// 2:02.320(40 ms,−16.1)、1:50.660(30 ms,**−37.5**)。
///
/// ⭐ 机理(查清楚了,不是猜):**模型经常把浊音渲得比乐谱的音头早**几十毫秒,而这道门只认
/// 「**喂进去的** f0 == 0」⇒ 在音头前那几帧,它把真起音的**基频**(例:392 Hz,截止 588 Hz)
/// 整个滤掉,音头塌 11 dB。1:19.950 正是用户当场点名「有个跳变」的那个「し」。
/// ⛔ **这就是「一把刀在它被点名的那一处有效」与「它在全曲上净正」之间的差**;用户的耳判
/// 覆盖的是前者,后者要我自己去量 —— 我差一步就把它当出厂发出去了。
///
/// ⭐ 下次要救它,判别式已经现成:**咔哒那一族的音高是【错的】**(donor 未移调的尾音,
/// 624-711 Hz 对该窗 donor 的 740 Hz),而早起音的音高**与紧邻的浊音一致**。
/// ⇒ 门的条件应该从「f0==0」收窄成「f0==0 **且** 该段的实测音高与邻接浊音不符」,
/// 或者简单地在浊音边缘留 30-40 ms 余量。**两条都还没做,也都还没量。**
///
/// ## k 为什么取 1.5 而不是 2.0
/// 两条臂用户都确认「那个位置的咔哒现在确实没了」⇒ **平局按风险破**(与 S154 在 30/60 之间、
/// S160j 在 120/250 之间同一条规矩):k=2.0 把截止推到 `2 ×` 基频,更靠近辅音噪声自己的低频边缘。
/// 读数(47.20-47.252):有调档 28.8 → **15.5(k1.5)/ 5.5(k2.0)**;
/// ✅ 同一段的辅音噪声(2-8 kHz)31.0 → **32.2 / 31.7** —— 两档都没伤到它,所以多压的 10 dB
/// 买不到额外的确定性,却多了一分吃掉辅音的风险。
/// ⛔ 只在 **donor 那一遍**(`range_shift != 0`)走这条路;base 一个字节不动。
fn gate_unvoiced_tone(
    audio: &mut [f32],
    sample_rate: u32,
    note_hz: &[f32],
    k: f32,
    fade_ms: f32,
    guard_ms: f32,
) -> usize {
    let hop = (sample_rate as usize / 50).max(1);
    let n = audio.len();
    let fade = ((fade_ms / 1000.0) * sample_rate as f32) as usize;
    let guard = ((guard_ms / 1000.0) * sample_rate as f32) as usize;
    let mut hit = 0usize;
    let mut i = 0usize;
    while i < note_hz.len() {
        if note_hz[i] != 0.0 {
            i += 1;
            continue;
        }
        let mut j = i;
        while j + 1 < note_hz.len() && note_hz[j + 1] == 0.0 {
            j += 1;
        }
        // 参照基频:优先取**前**一个浊音帧(那才是正在延续的那个嗓音),没有就取后一个。
        let prev = (0..i).rev().find(|&t| note_hz[t] > 0.0).map(|t| note_hz[t]);
        let next = (j + 1..note_hz.len()).find(|&t| note_hz[t] > 0.0).map(|t| note_hz[t]);
        let Some(f_ref) = prev.or(next) else {
            i = j + 1;
            continue;
        };
        let cut = (k * f_ref).clamp(200.0, 2500.0);
        let a = (i * hop).min(n);
        let mut b = ((j + 1) * hop).min(n);
        // S162 —— 起音护栏:后继浊音那一侧留 `guard` 个样本不碰(见 `parse_uvgate_guard_ms`)。
        // ⛔ 只在**真有后继浊音**时收:run 落在曲尾(没有 `next`)时没有起音可保,收了纯亏。
        if next.is_some() {
            b = b.saturating_sub(guard).max(a);
        }
        if b > a + 2 * fade + 8 {
            highpass_span(audio, sample_rate, a, b, cut, fade);
            hit += 1;
        }
        i = j + 1;
    }
    hit
}

/// 零相位高通(RBJ 双二阶,前向 + 反向),只作用在 `[a, b)`,两端各 `fade` 个样本交叉淡化。
/// ⛔ 为了不在段边引入瞬态,滤波在 `[a-ctx, b+ctx]` 上跑(`ctx` = 4 个截止周期),
///    但**只有 `[a, b)` 的样本被写回**。
fn highpass_span(audio: &mut [f32], sample_rate: u32, a: usize, b: usize, cut: f32, fade: usize) {
    let sr = sample_rate as f32;
    let ctx = ((4.0 * sr / cut) as usize).min(a).min(audio.len() - b);
    let lo = a - ctx;
    let hi = b + ctx;
    let src: Vec<f32> = audio[lo..hi].to_vec();
    // RBJ high-pass, Q = 1/sqrt(2)
    let w0 = 2.0 * std::f32::consts::PI * cut / sr;
    let (sn, cs) = (w0.sin(), w0.cos());
    // Butterworth: Q = 1/sqrt(2) ⇒ alpha = sin(w0) / (2Q)。
    // ⛔ 第一版写成 `.recip()`(= Q 取了 sqrt(2))⇒ 截止处鼓一个 +3 dB 的包 —— 自查抓到的。
    let alpha = sn / (2.0 * std::f32::consts::FRAC_1_SQRT_2);
    let b0 = (1.0 + cs) / 2.0;
    let b1 = -(1.0 + cs);
    let b2 = (1.0 + cs) / 2.0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cs;
    let a2 = 1.0 - alpha;
    let (b0, b1, b2, a1, a2) = (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0);
    let run = |x: &[f32]| -> Vec<f32> {
        let (mut x1, mut x2, mut y1, mut y2) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        x.iter()
            .map(|&v| {
                let y = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
                x2 = x1;
                x1 = v;
                y2 = y1;
                y1 = y;
                y
            })
            .collect()
    };
    let f1 = run(&src);
    let mut rev: Vec<f32> = f1.into_iter().rev().collect();
    rev = run(&rev);
    let filt: Vec<f32> = rev.into_iter().rev().collect();
    for t in a..b {
        let u = t - lo;
        // 两端交叉淡化:段内 `fade` 个样本从原始渐变到滤波后的
        let w = if fade == 0 {
            1.0
        } else if t < a + fade {
            (t - a) as f32 / fade as f32
        } else if t + fade >= b {
            (b - t) as f32 / fade as f32
        } else {
            1.0
        };
        audio[t] = (1.0 - w) * audio[t] + w * filt[u];
    }
}

fn apply_range_inverse(
    audio: Vec<f32>,
    sample_rate: u32,
    range_shift: i64,
    kappa: f32,
    note_hz: &[f32],
    keep: &[(usize, usize)],
) -> crate::Result<Vec<f32>> {
    if range_shift == 0 || audio.is_empty() {
        return Ok(audio);
    }
    // note_hz is the FED (already range-shifted) parametric pitch on the 50 fps cv grid — it
    // drives the inverse's streaming formant base (S82b anti-pop; vocal_range folds it into
    // a sticky schedule).
    // S160k —— 清音帧去调门(见 [`gate_unvoiced_tone`])。出厂关 ⇒ 一个分支都不走。
    // ⚠ 放在 dump **之前**:那份 dump 的语义是「**PSOLA 实际吃到的东西**」。
    let mut audio = audio;
    if parse_uvgate(std::env::var("UTAI_MG_UVGATE").ok().as_deref()) {
        // ⛔ S162 —— 调用点**必须走那两个纯函数**,否则「进指纹的默认」与「实际跑的默认」
        //    会各走各的(钩子区:纯函数不接到调用点上就是空判据)。以前这里自己抄了一份
        //    `parse::<f32>() + 范围过滤`,范围与 `parse_uvgate_k` 并不相同。
        let k = parse_uvgate_k(std::env::var("UTAI_MG_UVGATE_K").ok().as_deref());
        let guard_ms = parse_uvgate_guard_ms(std::env::var("UTAI_MG_UVGATE_GUARD_MS").ok().as_deref());
        let hit = gate_unvoiced_tone(&mut audio, sample_rate, note_hz, k, 4.0, guard_ms);
        // 「臂开着」与「臂做了事」必须分开可查(S129 铁律);⭐ 并且把**实际生效值**打出来
        // (钩子区:凡加一个旋钮,同时加一行打印实际生效值 —— S160e/f 两个静默失效的旋钮
        //  都是被这一行抓住的)。
        tracing::info!(
            "range-extend(donor {range_shift:+}): uv-gate k={k} guard={guard_ms}ms — {hit} unvoiced span(s) high-passed"
        );
    }
    dump_donor_buffer("pre", range_shift, &audio, note_hz);
    let out = super::vocal_range::apply_inverse_windowed(
        audio,
        sample_rate,
        range_shift,
        kappa,
        Some((note_hz, (sample_rate as usize / 50).max(1))),
        keep,
        // ⭐ S162 —— **谱面轨吃谱倾斜**(出厂 1.0)。表就是在这条车道的素材上拟的:
        // 靶子是**浅救援 −6**(用户说的「另一部分正常」那一半),留出验证下形状距离降 26-46%,
        // 跨模型零噪声护栏 10/10 改善。⛔ cover 那两个调用点传 0 —— 见它们头上的注释。
        super::vocal_range::range_tilt(),
    )
    .map_err(UtaiError::Inference);
    if let Ok(y) = &out {
        dump_donor_buffer("post", range_shift, y, note_hz);
    }
    out
}

/// S159g —— `UTAI_RANGE_DUMP_DONOR=<dir>`:把逆变换**前后**的缓冲各落一份裸 f32(小端),
/// 外加那一遍喂进去的 `note_hz`。文件名 `donor_<pre|post>_<shift>.f32`。
///
/// ⛔ 为什么需要一个新出口:S159g 已经把 donor 那一路在**音符交界处**的塌陷
/// (~40 ms 宽 · 电平 −2…−4 dB · 谱心 −20…−30%)量清楚了,并且逐条排除了
/// **PSOLA 本身**(同一段 base 音频过生产口径 +8 ⇒ 1.11-1.43 dB ≈ 不过 PSOLA 的原始)、
/// **喂进去的阶梯基频轨**(换成实测滑音轨读数几乎不动)、以及 **decode 之后那几把逐 chunk 的刀**
/// (它们对 base 与 donor 施加逐样本相同的乘性包络)。
/// 剩下的嫌疑只在**逆变换的输入**上,而在这条转储之前,**没有任何出口能看到它** ——
/// 「看不见的地方」正是上一轮我把归因搞错的地方。
///
/// ⚠ 只在 env 存在时写盘;写不动只 `warn!`,不许让渲染失败。
/// ⚠ 它**不是**生产路径上的开销:`var_os` 每次都读,但没设时立刻返回。
fn dump_donor_buffer(tag: &str, shift: i64, buf: &[f32], note_hz: &[f32]) {
    let Some(dir) = std::env::var_os("UTAI_RANGE_DUMP_DONOR") else { return };
    let dir = std::path::PathBuf::from(dir);
    if let Err(e) = std::fs::create_dir_all(&dir) {
        tracing::warn!("range dump: cannot create {}: {e}", dir.display());
        return;
    }
    let mut bytes = Vec::with_capacity(buf.len() * 4);
    for v in buf {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let p = dir.join(format!("donor_{tag}_{shift:+}.f32"));
    match std::fs::write(&p, &bytes) {
        Ok(()) => tracing::info!("range dump: {} samples -> {}", buf.len(), p.display()),
        Err(e) => tracing::warn!("range dump: {} failed: {e}", p.display()),
    }
    if tag == "pre" {
        let mut hz = Vec::with_capacity(note_hz.len() * 4);
        for v in note_hz {
            hz.extend_from_slice(&v.to_le_bytes());
        }
        let p = dir.join(format!("donor_f0_{shift:+}.f32"));
        if let Err(e) = std::fs::write(&p, &hz) {
            tracing::warn!("range dump: {} failed: {e}", p.display());
        }
    }
}


/// `w *= peak / (max|w| + 1e-9)` — render_ust.render_song's final output normalization.
/// Normalize to `peak`; returns the peak the buffer had **before** the scaling.
fn peak_normalize(w: &mut [f32], peak: f32) -> f32 {
    let m = w.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    peak_normalize_to(w, peak, None);
    m
}

/// S147: normalize to `peak`, but measure against `target` (the BASE render's pre-norm peak)
/// instead of this buffer's own.
///
/// ⛔ `target = Some(p)` is what makes base and donor share one scalar. Using each buffer's own
/// peak was fine while every donor rendered the whole song; it stops being fine the moment a
/// donor renders a subset, because "the loudest sample" then depends on **which chunks happened
/// to be kept** — measured pre-norm peaks moved 1.303 → 0.922 (+3.00 dB) on one shift and
/// 1.287 → 1.289 (nothing) on another, purely by where the maximum lived. There is no bound on
/// that error, which is why this is a shared scalar and not a corrective ratio.
fn peak_normalize_to(w: &mut [f32], peak: f32, target: Option<f32>) {
    let m = target.unwrap_or_else(|| w.iter().fold(0.0f32, |a, &v| a.max(v.abs())));
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

// S84 鹅妈妈快段探针(diagnostic #[ignore])— 同 e1 姿势挂子模块复用本文件私有件
// (build_note_hz / zero_voiceless_frames / anchor_voiced_phone_f0 / voiceless_onset_flags)。
#[cfg(test)]
#[path = "score2svc_mg.rs"]
mod mg_tests;

#[cfg(test)]
mod tests {
    use super::*;
    // ── S147 B2:donor 打洞 ─────────────────────────────────────────────────────────
    fn kmask_chunks(lens: &[usize], hard: &[usize]) -> Vec<Chunk> {
        lens.iter()
            .enumerate()
            .map(|(i, &t)| Chunk {
                start: 0,
                end: 0,
                phonemes: vec![],
                note_pitch: vec![],
                phone_dur: vec![],
                note_dur: vec![],
                note_to_phone: vec![],
                t,
                lang_id: 2,
                hard_seam: hard.contains(&i),
            })
            .collect()
    }

    #[test]
    fn a_donor_keeps_only_the_chunks_its_windows_touch() {
        // 帧轴:chunk 0 = 0..100 · 1 = 100..200 · 2 = 200..300 · 3 = 300..400
        let chunks = kmask_chunks(&[100, 100, 100, 100], &[]);
        let keep = donor_keep_mask(&chunks, &[(210, 240)], 0);
        assert_eq!(keep, vec![false, false, true, false]);
        // ⛔ 存在性闸:必须真的跳过了东西,否则下面每一条断言都可能在恒真格上
        assert!(keep.iter().any(|&k| !k), "no chunk skipped ⇒ this test proves nothing");
    }

    #[test]
    fn the_margin_pulls_in_the_neighbour_a_window_only_just_misses() {
        // ⛔ 这条钉的是那 458.8 样本(10.4 ms)的映射偏差:窗**帧域上**刚好不碰 chunk 1,
        // 但拼接层的线性 spf 会把它的 10 ms 淡出算到那边去 ⇒ 没有边距就会拼进数字零。
        let chunks = kmask_chunks(&[100, 100, 100], &[]);
        assert_eq!(
            donor_keep_mask(&chunks, &[(201, 250)], 0),
            vec![false, false, true],
            "零边距时只保留 chunk 2 —— 这是被修的那个行为"
        );
        assert_eq!(
            donor_keep_mask(&chunks, &[(201, 250)], DONOR_WINDOW_MARGIN_FRAMES),
            vec![false, true, true],
            "有边距时必须把左邻也拉进来"
        );
    }

    #[test]
    fn a_hard_seam_is_never_left_on_the_edge_of_a_hole() {
        // hard_seam 按定义落在**浊音正中**(score2cv:语言切点不落在静音上)⇒ 洞边落那儿会让
        // PSOLA 的整条岛重新定相(实测最坏 +2.9 dB、完全去相关)。
        let chunks = kmask_chunks(&[100, 100, 100, 100], &[2]);
        let keep = donor_keep_mask(&chunks, &[(210, 240)], 0);
        assert_eq!(keep, vec![false, true, true, false], "hard_seam 的左邻必须一起保留");

        // ⛔ 阴性对照必须能抓住「无条件保留 hard_seam」这个错法:chunk 0 是 hard_seam 但
        // **离任何窗都很远**,它不该被保留。(上一版的对照用的是「没有 hard_seam」的谱,
        // 那种谱上错法与正法给出同一个答案 ⇒ 恒真。)
        let far = kmask_chunks(&[100, 100, 100, 100], &[0, 2]);
        assert_eq!(
            donor_keep_mask(&far, &[(210, 240)], 0),
            vec![false, true, true, false],
            "远离窗的 hard_seam 不许被保留;只有洞边上的那些才要"
        );
        // 单语谱 hard_seam 恒 false ⇒ 规则零代价
        let plain = kmask_chunks(&[100, 100, 100, 100], &[]);
        assert_eq!(
            donor_keep_mask(&plain, &[(210, 240)], 0),
            vec![false, false, true, false],
            "没有 hard_seam 时规则不许多留任何 chunk"
        );
    }

    #[test]
    fn two_shifts_with_different_windows_must_keep_different_chunks() {
        // ⛔ 真机抓到的 bug 的判据。第一版把**全部位移的窗的并集**传给每一遍 ⇒ 四个 donor
        // 渲同一批 chunk,`skipped` 全是 12/25。功能是对的(渲多了不会错),收益少了一大半,
        // 而**唯一暴露它的就是「不同位移的跳过数完全相同」这个指纹**。
        // ⇒ 这条判据直接钉那个指纹:窗不同 ⇒ 保留集必须不同。
        let chunks = kmask_chunks(&[100, 100, 100, 100], &[]);
        let early = donor_keep_mask(&chunks, &[(10, 40)], 0);
        let late = donor_keep_mask(&chunks, &[(310, 340)], 0);
        assert_eq!(early, vec![true, false, false, false]);
        assert_eq!(late, vec![false, false, false, true]);
        assert_ne!(early, late, "不同的窗必须给出不同的保留集");

        // 阴性对照:两遍的窗**相同**时保留集当然该相同 —— 否则上面那条可能是别的原因红的
        assert_eq!(
            donor_keep_mask(&chunks, &[(10, 40)], 0),
            early,
            "同样的窗必须给出同样的答案(不然这把尺子自己不稳)"
        );
    }

    #[test]
    fn a_base_pass_keeps_every_chunk_and_a_full_window_skips_nothing() {
        // 两条退化格,都必须是 no-op —— 它们是「这一笔对 base 逐位不变」的前提。
        let chunks = kmask_chunks(&[100, 100, 100], &[]);
        assert_eq!(donor_keep_mask(&chunks, &[(0, 300)], 0), vec![true; 3]);
        // 空窗集:一个 chunk 都不留(调用方保证 windows 非空;这里钉行为不钉期望)
        assert_eq!(donor_keep_mask(&chunks, &[], 0), vec![false; 3]);
    }

    #[test]
    fn the_zero_fill_uses_the_hop_grid_not_the_cv_grid() {
        // ⛔ 拼接层用**绝对样本下标**,所以跳过的 chunk 铺多少零是承重的。
        // 用 `chunk.t * sr / 50` 代替 `sovits_grid_len` 在 akiko 的口径上单 chunk 差 124 样本
        // ⇒ 之后每个窗都会滑位,而日志只会说 "clamped"。
        let (sr, hop) = (44100u32, 512usize);
        for t in [1usize, 7, 100, 400, 401] {
            let grid = sovits_grid_len(t, sr, hop) * hop;
            let naive = t * sr as usize / 50;
            assert!(
                grid.abs_diff(naive) <= 512,
                "t={t}: grid {grid} vs naive {naive} —— 差得离谱说明我算错了口径"
            );
        }
        // 而在真实 chunk 长度上,两者**确实不同** —— 否则这条判据没有意义
        let t = 400usize;
        assert_ne!(
            sovits_grid_len(t, sr, hop) * hop,
            t * sr as usize / 50,
            "两个口径必须真的不同,否则这条判据是恒真的"
        );
    }

    /// S159b —— ⛔⛔ **RVC 臂上「洞」的长度**。同一条理由:拼接层按绝对样本索引,
    /// 差一个样本这一遍之后每条救援窗都滑走,而所有计数器全绿。
    ///
    /// ⚠ SoVITS 那一侧的坑是 hop 栅格取整;RVC 这一侧的坑是**那句「~」** ——
    /// 截断行的注释写着「RVC net_g emits **~** p_len·(sr/100) samples」,而跳过的 chunk 要铺的零
    /// 用的就是这个信念。⇒ 这条判据钉住公式本身(**期望值写字面量**),
    /// 而**真渲出来的每个 chunk 都会与它对一次** —— 对不上就进 `[perf]` 的 `len!=` 计数。
    /// ⭐ S159b 拿 yachiyo(48 kHz)在 `mg_render_rvc` 上实跑,`len!= 0` ⇒ 这个信念**被证伪过一次机会**。
    #[test]
    fn the_rvc_hole_is_exactly_what_a_chunk_produces() {
        // 48 kHz:100 fps 的每一帧 = 480 个样本,而 `rvc_feed_100` 把 50 fps 的 cv 复制成 100 fps
        // ⇒ 一个 50 fps 帧 = 960 个样本。⛔ 期望值写死,不许拿被测的公式算。
        assert_eq!(rvc_out_len(1, 48_000), 960);
        assert_eq!(rvc_out_len(400, 48_000), 384_000);
        assert_eq!(rvc_out_len(0, 48_000), 0);
        // 40 kHz / 32 kHz 的 RVC 导出(仓里支持的另外两档)
        assert_eq!(rvc_out_len(1, 40_000), 800);
        assert_eq!(rvc_out_len(1, 32_000), 640);
        // ⭐ 它必须与**截断那一行**是同一个信念:`real_t * (sr/100)`,而 `real_t = 2·t50`。
        for (t, sr) in [(1usize, 48_000u32), (7, 40_000), (400, 32_000), (401, 48_000)] {
            let real_t = t * 2;
            assert_eq!(rvc_out_len(t, sr), real_t * (sr as usize / 100), "t={t} sr={sr}");
        }
        // ⛔ 阴性对照:它与 SoVITS 那一侧的公式**不是**一回事(照抄过去会差出量级)。
        assert_ne!(rvc_out_len(400, 44_100), sovits_grid_len(400, 44_100, 512) * 512);
    }

    /// S159b —— **接线闸:两条谱面臂都必须问 `donor_keep_mask` 该渲哪些 chunk。**
    ///
    /// ⛔ 为什么需要它:S147 的 B2 当年**只做在 SoVITS 臂上**,RVC 谱面臂就这样每遍渲整曲
    /// 渲了十二场 —— 而它「功能完全正确」,只是慢一倍(实测同曲 182 s vs 78.6 s)。
    /// 没有任何判据看得见「一条臂没接这一刀」,因为**不接也是对的**。
    ///
    /// ⚠ 这是文本闸,只证明那一行在;它证明不了参数对(那是上面那几条 `donor_keep_mask` 的行为
    /// 判据的事)。⛔ 期望的符号用 `concat!` 拼 —— `include_str!(自己)` 会把断言自己读进来,
    /// 断言里出现的字面量一定命中(S159 在 `vocal_range.rs` 上刚踩过这条)。
    #[test]
    fn both_score_arms_ask_donor_keep_mask_which_chunks_to_render() {
        let me = include_str!("score2svc.rs");
        let call = concat!("donor_keep", "_mask(&chunks, d.windows, DONOR_WINDOW_MARGIN_FRAMES)");
        assert_eq!(
            me.matches(call).count(),
            2,
            "谱面轨的两条臂(SoVITS / RVC)必须各问一次 —— 少一条 = 那条臂每遍 donor 都渲整曲,\
             而功能完全正确、没有任何计数器会红"
        );
        // 阴性对照:两条臂各自的洞也必须各用各的长度公式,而它们不同。
        assert_eq!(me.matches(concat!("rvc_out", "_len(chunk.t, m.sample_rate)")).count(), 1);
        assert_eq!(me.matches(concat!("sovits_grid", "_len(chunk.t, m.sample_rate, m.hop_size)")).count(), 1);
    }


    // ── S147 归一口径 ───────────────────────────────────────────────────────────────
    #[test]
    fn a_shared_normalization_target_preserves_the_relative_level_of_two_renders() {
        // ⭐ THE property this change exists for. base and each donor used to normalize to their
        // own peak, so they arrived on different absolute scales and the splice layer had to
        // guess the difference back with a whole-song `active_rms` ratio — a guess that breaks
        // the moment a donor renders less than the whole song (measured: +0.618 dB on the
        // rescued phrases, 8.7× the per-chunk level floor).
        let mut base = vec![0.5f32, -0.25, 0.10];
        let mut donor = vec![0.25f32, -0.125, 0.05]; // 与 base 逐样本差 6 dB
        let base_peak = peak_normalize(&mut base, 0.92);
        assert!((base_peak - 0.5).abs() < 1e-6, "must report the PRE-norm peak, got {base_peak}");

        peak_normalize_to(&mut donor, 0.92, Some(base_peak));
        // 共用标量 ⇒ 6 dB 的相对关系原样保留。自归一会把它抹平成 0 dB。
        let rel = 20.0 * (donor[0] / base[0]).log10();
        assert!((rel + 6.0206).abs() < 0.01, "relative level must survive, got {rel:+.4} dB");

        // 阴性对照:同一个 donor 自归一时,那 6 dB **消失** —— 这正是旧口径要用 RMS 猜回来的东西。
        let mut solo = vec![0.25f32, -0.125, 0.05];
        peak_normalize_to(&mut solo, 0.92, None);
        assert!(
            (20.0 * (solo[0] / base[0]).log10()).abs() < 0.01,
            "self-normalization must flatten it — otherwise this test proves nothing"
        );
    }

    #[test]
    fn passing_no_target_is_byte_identical_to_self_normalization() {
        // 全部老调用点都传 `None` ⇒ 它们必须逐位不变(additive-then-flip 的前提)。
        let src: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.37).sin() * 0.8).collect();
        let (mut a, mut b) = (src.clone(), src.clone());
        peak_normalize(&mut a, 0.92);
        peak_normalize_to(&mut b, 0.92, None);
        assert_eq!(a, b, "None must be exactly the old behaviour");
        assert!((a.iter().fold(0.0f32, |m, &v| m.max(v.abs())) - 0.92).abs() < 1e-6);
    }

    use crate::inference::g2p_alias::PhonemeSet;
    use super::super::score2cv::{build_arrays, ArticulationTiming, NoDicts}; // Phase-1c parity entry (rest-capped)
    use super::super::score2cv_tables::parity_ref as pr;

    /// JA-defaulted events from legacy triples (the pre-S58 test fixtures).
    fn ja_evts<'a>(score: &'a [(&'a str, i64, i64)]) -> Vec<ScoreEvt<'a>> {
        score.iter().map(ScoreEvt::ja).collect()
    }
    /// DAW build over a JA triple fixture (rests uncapped + borrow-time).
    fn daw_ja(score: &[(&str, i64, i64)]) -> ScoreArrays {
        build_arrays_daw(&ja_evts(score), &NoDicts, ArticulationTiming::Auto).unwrap()
    }

    /// 诊断（S163）：短休止的 SP 音素在 `build_arrays_daw` 之后还剩多少帧。
    /// 用户报「很短的休止中间会漏出伪影」，而实测 0-150 ms 的休止 **80/102 完全没有零区**
    /// ⇒ 怀疑 `consonant_preroll` 把短休止的 SP 时长吃光了 ⇒ `chunk_sp_windows` 收不到窗。
    #[test]
    #[ignore = "诊断用，打印 SP 时长分配"]
    fn diag_sp_phone_dur_after_preroll() {
        // 音 / 休止(R) / 以清辅音开头的音 —— 正是 preroll 会伸手的形状
        for rest_frames in [3i64, 5, 7, 10, 12, 20] {
            for nxt in ["か", "た", "さ", "あ", "うぉ", "わ", "を", "お"] {
                let score = [("あ", 69, 20), ("R", 0, rest_frames), (nxt, 71, 20)];
                let arr = daw_ja(&score);
                let sp: Vec<(usize, &str, i64)> = arr
                    .phon
                    .iter()
                    .enumerate()
                    .map(|(i, p)| (i, *p, arr.phone_dur[i]))
                    .collect();
                let sp_total: i64 = sp.iter().filter(|(_, p, _)| *p == "SP").map(|(_, _, d)| *d).sum();
                println!(
                    "休止 {:>2} 帧({:>3} ms) 下一个音 {} ⇒ SP 剩 **{} 帧({} ms)**   全部: {:?}",
                    rest_frames,
                    rest_frames * 20,
                    nxt,
                    sp_total,
                    sp_total * 20,
                    sp
                );
            }
        }
    }

    /// ⛔⛔ **承重**：平台电平在阈值以下的窗**逐位不变**。
    /// 这一刀只该碰「本该静音却响着」的那些；用户自己给的阴性对照（1:51.451，平台 −45…−64 dB）
    /// 必须一个字节都不动。
    #[test]
    fn preroll_damp_leaves_quiet_platforms_bit_identical() {
        let sr = 44_100usize;
        let fade = sr / 200;
        let keep = (PREROLL_KEEP_MS / 1000.0 * sr as f32) as usize;
        // 窗 200 ms，平台很轻（相对稳态 −50 dB），后面接一段稳态
        let n = sr / 5;
        let mut x = vec![0.0f32; n + sr];
        for v in x[..n].iter_mut() {
            *v = 0.003;                     // 平台
        }
        for v in x[n + fade..n + fade + fade * 20].iter_mut() {
            *v = 1.0;                       // 稳态参照
        }
        let before = x.clone();
        let hit = apply_preroll_damp(&mut x, &[(0, n)], keep, PREROLL_DAMP_THRESH_DB, PREROLL_DAMP_MAX_DB, sr as u32);
        assert_eq!(hit, 0, "安静平台不该被判为要压");
        assert_eq!(x, before, "阈值以下必须逐位不变");
    }

    /// ⭐ 平台响到超阈值时**真的被压**，而且**紧贴音头的 `keep` 一个字节不动**。
    #[test]
    fn preroll_damp_cuts_the_loud_platform_but_never_the_real_consonant() {
        let sr = 44_100usize;
        let fade = sr / 200;
        let keep = (PREROLL_KEEP_MS / 1000.0 * sr as f32) as usize;
        let n = sr / 5; // 200 ms 窗
        let mut x = vec![0.0f32; n + sr];
        for v in x[..n].iter_mut() {
            *v = 0.5;                       // 平台：与稳态同量级 ⇒ 远超 −40 dB
        }
        for v in x[n + fade..n + fade + fade * 20].iter_mut() {
            *v = 1.0;
        }
        let tail_before: Vec<f32> = x[n - keep..n].to_vec();
        let hit = apply_preroll_damp(&mut x, &[(0, n)], keep, PREROLL_DAMP_THRESH_DB, PREROLL_DAMP_MAX_DB, sr as u32);
        assert_eq!(hit, 1, "响平台必须被判为要压");
        // 前部中心被压下去了
        let mid = (n - keep) / 2;
        assert!(x[mid].abs() < 0.5 * 0.5, "平台中心没被压：{}", x[mid]);
        // ⛔ 紧贴音头的 keep 一个字节不动（辅音时序/真辅音不许碰）
        assert_eq!(&x[n - keep..n], &tail_before[..], "紧贴音头的真辅音被动了");
        // ⛔ 压幅有护栏
        let cut_db = 20.0 * (0.5f32 / x[mid].abs().max(1e-9)).log10();
        assert!(cut_db <= PREROLL_DAMP_MAX_DB + 0.5, "压过头了：{cut_db} dB");
    }

    /// ⛔⛔ **承重**：**右侧**（挨着真辅音那一头）的斜率必须封顶。
    /// 用户 2026-08-28 实听第一版：「**3:40.861 应该是压出竖条纹了**」——
    /// 那一版复用了 `emphasis` 的 5 ms 淡化，压幅 10-18 dB ⇒ **2.5 dB/ms** 的边界台阶。
    /// ⚠ 左侧不在这条判据里：它落在已经很安静的地方（见 `..LEFT_SLOPE..`），另有一条判据管它。
    #[test]
    fn preroll_damp_never_exceeds_the_slope_cap_on_the_right() {
        let sr = 44_100usize;
        let keep = (PREROLL_KEEP_MS / 1000.0 * sr as f32) as usize;
        for win_ms in [50usize, 80, 120, 200, 400] {
            let n = win_ms * sr / 1000;
            if n <= keep + 4 {
                continue;
            }
            let pre = 60 * sr / 1000;
            let mut x = vec![0.0f32; pre + n + sr];
            for v in x[pre..pre + n].iter_mut() {
                *v = 0.5;
            }
            for v in x[pre + n + sr / 20..pre + n + sr / 20 + sr / 10].iter_mut() {
                *v = 1.0;
            }
            let before = x.clone();
            apply_preroll_damp(&mut x, &[(pre, pre + n)], keep, PREROLL_DAMP_THRESH_DB, PREROLL_DAMP_MAX_DB, sr as u32);
            let cell = sr / 1000;
            let gain_db = |i: usize| -> f64 {
                let (a, b) = (i * cell, (i + 1) * cell);
                let e0: f64 = before[a..b].iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
                let e1: f64 = x[a..b].iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
                10.0 * ((e1 + 1e-30) / (e0 + 1e-30)).log10()
            };
            // 只看**右半**（从压制区中点到窗尾）—— 那一头挨着真辅音
            let mid = (pre + (pre + n - keep)) / 2 / cell;
            let end = (pre + n - keep) / cell;
            let step = (mid..end.saturating_sub(1))
                .map(|i| (gain_db(i + 1) - gain_db(i)).abs())
                .fold(0.0f64, f64::max);
            assert!(
                step <= f64::from(PREROLL_DAMP_MAX_SLOPE_DB_PER_MS) * 1.6,
                "{win_ms} ms 窗：右侧斜率 {step:.2} dB/ms 超过上限 {} —— 那就是竖条纹",
                PREROLL_DAMP_MAX_SLOPE_DB_PER_MS
            );
        }
    }

    /// ⭐ 压幅由**两头各自的**可用长度决定（右 × 0.6，左 × 2.0），取小者；
    /// 而且**左侧没空间时不许退化成极陡的淡入**（窗内借 `LEFT_BORROW_MS`）。
    #[test]
    fn preroll_damp_cut_is_bounded_by_both_sides() {
        let sr = 44_100usize;
        let keep = (PREROLL_KEEP_MS / 1000.0 * sr as f32) as usize;
        let cut_with = |win_ms: usize, left_ms: usize| -> f64 {
            let n = win_ms * sr / 1000;
            let pre = left_ms * sr / 1000;
            let mut x = vec![0.0f32; pre + n + sr];
            for v in x[pre..pre + n].iter_mut() {
                *v = 0.05;
            }
            for v in x[pre + n + sr / 20..pre + n + sr / 20 + sr / 10].iter_mut() {
                *v = 1.0;
            }
            apply_preroll_damp(&mut x, &[(pre, pre + n)], keep, PREROLL_DAMP_THRESH_DB, PREROLL_DAMP_MAX_DB, sr as u32);
            (pre..pre + n)
                .map(|i| -20.0 * f64::from(x[i].abs() / 0.05).log10())
                .fold(0.0f64, f64::max)
        };
        // 右侧越长压得越多（左侧充裕）
        let (a, b) = (cut_with(60, 60), cut_with(100, 60));
        assert!(b > a + 1.0, "右侧更长该压得更多：{a:.1} → {b:.1} dB");
        // ⛔ 左侧没空间时仍然受控（不为 0、也不许无限压）
        let c = cut_with(100, 0);
        assert!(c > 0.0, "左侧没空间就完全不压了？{c:.1}");
        assert!(
            c <= f64::from(PREROLL_DAMP_LEFT_SLOPE_DB_PER_MS) * f64::from(PREROLL_DAMP_LEFT_BORROW_MS) + 1.0,
            "左侧没空间却压了 {c:.1} dB —— 借的长度撑不住这个斜率"
        );
    }

    /// ⛔⛔ **承重**：判定「有多响」必须看**峰段**，不是全区平均。
    /// S163 实测 1:44.097：压制区平均 −36 dB 而平台峰段 −16 dB
    /// ⇒ 用平均时闸只压 4 dB，而耳朵听到的正是 −16 那一段（用户：「1:44 那个还挺明显的」）。
    /// 夹具：压制区里**只有一小段是响的**，其余很安静 —— 平均口径会漏判，峰段口径不会。
    #[test]
    fn preroll_damp_judges_by_the_peak_window_not_the_average() {
        let sr = 44_100usize;
        let keep = (PREROLL_KEEP_MS / 1000.0 * sr as f32) as usize;
        let n = 160 * sr / 1000;
        let pre = 60 * sr / 1000;
        let mut x = vec![0.0f32; pre + n + sr];
        // 压制区 = [pre, pre+n-keep] = 120 ms：前 40 ms 很响，其余很安静
        let loud_end = pre + 40 * sr / 1000;
        for v in x[pre..loud_end].iter_mut() {
            *v = 0.2;                      // 响段
        }
        for v in x[loud_end..pre + n].iter_mut() {
            *v = 0.0008;                   // 安静段（把平均拉到远低于阈值）
        }
        for v in x[pre + n + sr / 20..pre + n + sr / 20 + sr / 10].iter_mut() {
            *v = 1.0;                      // 稳态参照
        }
        let hit = apply_preroll_damp(&mut x, &[(pre, pre + n)], keep, PREROLL_DAMP_THRESH_DB, PREROLL_DAMP_MAX_DB, sr as u32);
        assert_eq!(hit, 1, "峰段 −14 dB 远超阈值，必须判为要压（全区平均会漏判）");
        let cut = (pre..loud_end)
            .map(|i| -20.0 * f64::from(x[i].abs() / 0.2).log10())
            .fold(0.0f64, f64::max);
        assert!(cut > 6.0, "响段只压了 {cut:.1} dB —— 闸还是在看平均");
    }

    /// ⛔ flags：只标 **SP 之后的辅音串**，到元音为止；**AP（呼吸）不算休止**。
    /// ⛔ 判「辅音」用的是不在 VOWEL_SET 里，**不是**「清辅音」——
    /// 用户报的坐标全是「うぉ」= `w`+`o`，`w` 是浊近音，按清浊分类会整条链漏掉（S163 §45.3）。
    /// ⚠ 用真实的 `build_arrays_daw` 造夹具，不手搓 `ScoreArrays`。
    #[test]
    fn preroll_flags_take_the_consonant_run_after_a_rest_including_voiced_glides() {
        let check = |score: &[(&str, i64, i64)], want_marked: &[&str]| {
            let arr = daw_ja(score);
            let f = preroll_consonant_flags(&arr);
            let got: Vec<&str> = arr
                .phon
                .iter()
                .zip(&f)
                .filter(|(_, &m)| m)
                .map(|(p, _)| *p)
                .collect();
            assert_eq!(got, want_marked, "谱 {score:?} ⇒ 音素 {:?}", arr.phon);
        };
        // ⭐ 用户报的形状：休止之后是「うぉ」= w + o ⇒ 只标 w
        check(&[("あ", 69, 20), ("R", 0, 7), ("うぉ", 71, 20)], &["w"]);
        // 清辅音同样标
        check(&[("あ", 69, 20), ("R", 0, 7), ("か", 71, 20)], &["k"]);
        // 纯元音开头 ⇒ 什么都不标
        check(&[("あ", 69, 20), ("R", 0, 7), ("お", 71, 20)], &[]);
        // 没有休止 ⇒ 什么都不标（这一刀只管被 preroll 推进休止的那些）
        check(&[("あ", 69, 20), ("か", 71, 20)], &[]);
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
        // ⛔⛔ S163 —— 这里原本钉的是「短窗(< 2·fade)= **浅谷,不至 0**(自然过渡)」。
        //    那句「自然过渡」是**推理，从来没有测量支持**，而用户 2026-08-28 的耳朵推翻了它：
        //    「**很短的休止中间会漏出伪影**」「我们可能一直在这地方差点东西」。
        //    实测(鹅妈妈×yachiyo×+7，base 层，休止中段相对相邻音稳态)：
        //      0-150 ms 中段 p50 **−43.8 dB**、**38%** 高于 −40；≥150 ms 能到 **−280 dB**。
        //    用户报的 1:44.098 / 1:48.431 / 1:57.094 在这把尺子上是 **96% / 91% / 89% 分位**，
        //    而他自己排除的 1:51.451(「这个不是空拍」)只有 53% —— 阴性对照是他给的。
        //    ⇒ 「浅谷」不是自然过渡，是**门关不严**。
        //
        //    ⚠ 「自然过渡」背后**真实**的担心是「突然归零会造咔哒」——
        //    那个担心由 `rest_gate_stays_a_continuous_ramp_after_shrinking` 承担(斜坡不是阶跃)，
        //    而不是靠「不许归零」来回避。长窗本来就归零(上面 a[500] == 0.0)。
        //    ⚠ 「归零 = 绝对静音」也不是问题：S159z 已结案(PCM_16 量化地板，不可闻)。
        let mut b = vec![1.0f32; 300];
        apply_rest_gate_with(&mut b, &[(100, 160)], 100, true);
        assert_eq!(b[130], 0.0, "短窗的中心也必须真的归零(门要关严)");
        assert_eq!(b[100], 1.0, "左缘仍与音符连续");
        assert_eq!(b[159], 1.0, "右缘仍与音符连续");
        assert_eq!(b[99], 1.0, "窗外不动");
        assert_eq!(b[160], 1.0, "窗外不动");
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

    // S83 knife 5: voiceless windows zero only their MEASURED (bucketed) fraction, centered —
    // edges keep the dragged-in/pre-voicing f0 exactly like RMVPE on real singing.
    #[test]
    fn zero_voiceless_is_partial_and_centered() {
        let arr = |phon: Vec<&'static str>, dur: Vec<i64>, nd: i64| ScoreArrays {
            phonemes: vec![0; phon.len()],
            phone_dur: dur.clone(),
            note_pitch: vec![60; phon.len()],
            note_dur: vec![nd; phon.len()],
            note_to_phone: vec![0; phon.len()],
            phon,
            lang: vec![2; dur.len()],
            evt: vec![0; dur.len()],
            // S92k: test-build-only audit fields, irrelevant to this fixture
            borrow_ledger: Vec::new(),
            in_note_alloc: Vec::new(),
        };
        // s in a mid-length group (10 fr): zero‰=575 → 4-frame window zeroes round(2.3)=2, centered.
        let a = arr(vec!["s", "a"], vec![4, 6], 10);
        let mut hz = vec![300.0f32; 10];
        zero_voiceless_frames(&mut hz, &a);
        assert_eq!(hz[0], 300.0, "leading edge keeps the dragged-in f0");
        assert_eq!((hz[1], hz[2]), (0.0, 0.0), "window center zeroed");
        assert_eq!(hz[3], 300.0, "vowel-adjacent edge keeps the pre-voicing f0");
        assert!(hz[4..].iter().all(|&v| v == 300.0), "the vowel is untouched");
        // devoiced vowel (not in the consonant table) → fallback 1000 = full-window zero.
        let b = arr(vec!["i̥"], vec![4], 4);
        let mut hz2 = vec![300.0f32; 4];
        zero_voiceless_frames(&mut hz2, &b);
        assert!(hz2.iter().all(|&v| v == 0.0), "a true devoiced vowel still zeroes fully");
        // voiced phone: never touched.
        let c = arr(vec!["m", "a"], vec![3, 3], 6);
        let mut hz3 = vec![300.0f32; 6];
        zero_voiceless_frames(&mut hz3, &c);
        assert!(hz3.iter().all(|&v| v == 300.0));
    }

    // S83 knife 6: only voiceless ONSET phones get the emphasis window — codas and voiced
    // onsets stay untouched; the trapezoid plateaus at the gain with clean edge ramps.
    #[test]
    fn voiceless_onset_emphasis_flags_and_shape() {
        // か(k a) + が(ɡ a) + "light"-shaped coda: only the k flags.
        let arr = daw_ja(&[("か", 60, 20), ("が", 62, 20)]);
        let flags = voiceless_onset_flags(&arr);
        let k = arr.phon.iter().position(|&p| p == "k").unwrap();
        let g = arr.phon.iter().position(|&p| p == "ɡ").unwrap();
        assert!(flags[k], "voiceless onset k flags");
        assert!(!flags[g], "voiced onset ɡ never flags");
        assert!(arr.phon.iter().zip(&flags).all(|(&p, &f)| !f || is_voiceless_phone(p)));
        // S83 refined-fix: a MEDIAL voiceless consonant (later syllable's onset on one note) also
        // flags — it leads toward a nucleus; the coda d (after the last nucleus) never does.
        let refined = g2p::ScoreEvt {
            lyric: "x", note_num: 60, frames: 50, lang: g2p::Lang::Ja,
            phoneme_input: Some("ɹ ə f aɪ n d"),
            phoneme_set: PhonemeSet::Words,
        };
        let arr2 = build_arrays_daw(&[refined], &NoDicts, ArticulationTiming::Auto).unwrap();
        let flags2 = voiceless_onset_flags(&arr2);
        let f_i = arr2.phon.iter().position(|&p| p == "f").unwrap();
        let d_i = arr2.phon.iter().position(|&p| p == "d").unwrap();
        assert!(flags2[f_i], "medial f (2nd-syllable onset) flags for emphasis");
        assert!(!flags2[d_i], "coda d never flags (word-final thud guard)");
        // trapezoid: edges ramp, plateau boosts, outside untouched.
        let mut a = vec![1.0f32; 100];
        apply_emphasis(&mut a, &[(20, 80)], 2.0, 10);
        assert_eq!(a[19], 1.0, "outside untouched");
        assert!((a[20] - 1.0).abs() < 0.11, "edge starts near unity");
        assert_eq!(a[40], 2.0, "plateau at gain");
        assert!((a[79] - 1.0).abs() < 0.11, "far edge back near unity");
    }

    // ── S84 C 刀: chain-internal consonant valley ──

    #[test]
    fn boundary_valley_chain_classes_and_exclusions() {
        // R か が ろ: post-rest k excluded (the rest IS the valley); chain-internal ɡ = stop
        // depth, ɾ = tap depth — VOICED consonants valley too (that's the whole point: the
        // measured render hole is 3.3 vs 15.5 dB precisely on voiced boundaries).
        let arr = daw_ja(&[("R", 0, 10), ("か", 60, 10), ("が", 62, 10), ("ろ", 62, 10)]);
        let d = boundary_valley_depths(&arr);
        let k = arr.phon.iter().position(|&p| p == "k").unwrap();
        let g = arr.phon.iter().position(|&p| p == "ɡ").unwrap();
        let r = arr.phon.iter().position(|&p| p == "ɾ").unwrap();
        assert_eq!(d[k], 0.0, "post-rest onset never valleys");
        // ⭐ S161b:出厂表的浊塞音换成**窄槽** 20 dB(只在 20% 的窗上,整段电平反而更接近真人)。
        // ⛔ 用**该 token 自己的帧数**去取靶,别把帧数写死 —— 写死就成了「夹具决定答案」。
        assert!((d[g] - valley_vstop_human_db(arr.phone_dur[g])).abs() < 1e-6,
                "出厂(真人表)浊塞音按时长取深度(该 token {} 帧)", arr.phone_dur[g]);
        assert!((valley_depth_db("ɡ") - 11.7).abs() < 1e-6, "S84 那张表本身不许被改动(回退臂用)");
        // ⭐ S161:出厂表已换成**真人差额**(闪音 10.4 → 6.8;鼻/边 11.4 → 1.1;塞音**没动**)。
        //    这条判据当场红过一次,红在 `tap valleys at tap depth` —— 正是它该红的地方。
        //    ⇒ 两张表都钉住:出厂那张 + `UTAI_VALLEY_HUMAN=0` 回退的那张(S84 的)。
        assert!((d[r] - valley_tap_human_db(arr.phone_dur[r])).abs() < 1e-6,
                "出厂(真人表)闪音按时长取深度(该 token {} 帧)", arr.phone_dur[r]);
        assert!((valley_depth_db("ɾ") - 10.4).abs() < 1e-6, "S84 那张表本身不许被改动(回退臂用)");
        assert!((valley_depth_db("n") - 11.4).abs() < 1e-6, "S84 那张表本身不许被改动(回退臂用)");
        for fr in [1i64, 2, 3, 5] {
            assert!((valley_depth_db_human("n", fr) - VALLEY_NASAL_HUMAN_DB).abs() < 1e-6, "真人表鼻音 = 1.1,且不随时长变");
        }
        // ⭐ S161b:**清**塞音两张表必须一样(它的 level 已经正好);**浊**塞音必须不一样。
        for p in ["k", "t", "p", "c", "ts", "tɕ", "ʔ"] {
            for fr in [1i64, 2, 3, 5, 20] {
                assert!((valley_depth_db_human(p, fr) - valley_depth_db(p)).abs() < 1e-6,
                        "{p} 清塞音两张表必须一样(且不随时长变),帧={fr}");
            }
            assert!(valley_shape_human(p).is_none(), "{p} 清塞音必须是矩形");
        }
        for p in ["ɡ", "b", "d", "dʑ", "dz", "bʲ"] {
            assert!(valley_depth_db_human(p, 3) > valley_depth_db(p), "{p} 浊塞音人表必须更深(窄槽)");
            assert!(valley_shape_human(p).is_some(), "{p} 浊塞音必须取真人包络");
            // ⛔⛔ S161c —— **槽深必须随时长单调不减**;常数刀就是这一条的反例。
            let (d2, d3, d4) = (valley_depth_db_human(p, 2), valley_depth_db_human(p, 3), valley_depth_db_human(p, 5));
            assert!(d2 < d3 && d3 < d4, "{p} 槽深必须随时长加深:{d2}/{d3}/{d4}");
        }
        for p in ["ɾ", "r"] {
            let (d2, d3) = (valley_depth_db_human(p, 2), valley_depth_db_human(p, 4));
            assert!(d2 < d3, "{p} 闪音槽深也必须随时长加深:{d2}/{d3}");
        }
        assert!(
            arr.phon.iter().zip(&d).all(|(&p, &v)| v == 0.0 || !super::super::score2cv::is_nucleus_phone(p)),
            "nuclei never valley"
        );
        // refined on one note after a sung あ: leading ɹ = approximant (tiny), medial f =
        // fricative depth (2nd-syllable boundary), coda n/d = NEVER (词尾顿挫 guard).
        let refined = g2p::ScoreEvt {
            lyric: "x", note_num: 60, frames: 50, lang: g2p::Lang::Ja,
            phoneme_input: Some("ɹ ə f aɪ n d"),
            phoneme_set: PhonemeSet::Words,
        };
        let arr2 = build_arrays_daw(&[g2p::ScoreEvt::ja(&("あ", 60, 10)), refined], &NoDicts, ArticulationTiming::Auto).unwrap();
        let d2 = boundary_valley_depths(&arr2);
        let ri = arr2.phon.iter().position(|&p| p == "ɹ").unwrap();
        let fi = arr2.phon.iter().position(|&p| p == "f").unwrap();
        let ni = arr2.phon.iter().position(|&p| p == "n").unwrap();
        let di = arr2.phon.iter().position(|&p| p == "d").unwrap();
        assert!((d2[ri] - 1.4).abs() < 1e-6, "chain-internal ɹ = approximant depth");
        assert!((d2[fi] - 5.1).abs() < 1e-6, "medial f = fricative depth");
        assert_eq!(d2[ni], 0.0, "coda n never valleys");
        assert_eq!(d2[di], 0.0, "coda d never valleys");
        // lone ん ([ɴ], no nucleus): nothing flags — ducking a whole musical note would be wrong.
        let arr3 = daw_ja(&[("あ", 60, 10), ("ん", 60, 10)]);
        let d3 = boundary_valley_depths(&arr3);
        assert!(d3.iter().all(|&v| v == 0.0), "nucleus-less ん note stays untouched: {d3:?}");
        // post-rest CLUSTER: neither member valleys — the skipped first member must not launder
        // the later ones into "chain-internal" (the walk-back is over same-event non-nucleus
        // mates, not over flag state).
        let sta = g2p::ScoreEvt {
            lyric: "x", note_num: 60, frames: 30, lang: g2p::Lang::Ja,
            phoneme_input: Some("s t a"),
            phoneme_set: PhonemeSet::Words,
        };
        let arr4 = build_arrays_daw(&[g2p::ScoreEvt::ja(&("R", 0, 10)), sta], &NoDicts, ArticulationTiming::Auto).unwrap();
        let d4 = boundary_valley_depths(&arr4);
        let si = arr4.phon.iter().position(|&p| p == "s").unwrap();
        let ti = arr4.phon.iter().position(|&p| p == "t").unwrap();
        assert_eq!(d4[si], 0.0, "post-rest cluster head never valleys");
        assert_eq!(d4[ti], 0.0, "post-rest cluster TAIL never valleys either (no laundering)");
        // …and the same cluster chain-internally (after a sung あ) valleys both members.
        let sta2 = g2p::ScoreEvt {
            lyric: "x", note_num: 60, frames: 30, lang: g2p::Lang::Ja,
            phoneme_input: Some("s t a"),
            phoneme_set: PhonemeSet::Words,
        };
        let arr5 = build_arrays_daw(&[g2p::ScoreEvt::ja(&("あ", 60, 10)), sta2], &NoDicts, ArticulationTiming::Auto).unwrap();
        let d5 = boundary_valley_depths(&arr5);
        let si5 = arr5.phon.iter().position(|&p| p == "s").unwrap();
        let ti5 = arr5.phon.iter().position(|&p| p == "t").unwrap();
        assert!((d5[si5] - 5.1).abs() < 1e-6, "chain-internal cluster head valleys (fric)");
        assert!((d5[ti5] - 11.7).abs() < 1e-6, "chain-internal cluster tail valleys (stop)");
    }

    // S89: a VOICED onset must carry the note's pitch on BOTH articulation arms.
    //
    // Under Auto the S83 pre-roll parks that onset inside the PREVIOUS rest's frame span, where
    // `build_note_hz` reads 0 Hz — SoVITS uv=0 / RVC pitchf=0 = an audibly mute onset, which is the
    // whole reason `anchor_voiced_phone_f0` exists. Under InNote the onset never leaves its own
    // note, so the repair should have nothing left to do. The assertion is on the OUTCOME ("no
    // voiced-onset frame is ever 0 Hz"), NOT on "the repair is a no-op": the recon flagged that
    // no-op-ness as something to PROVE rather than assume, and an outcome assertion stays true (and
    // stays meaningful) whichever way that turns out.
    ///
    /// ⚠ This MUST run on the layered DAW f0 (`Some(&VocalF0)`), not the bare note-only mode. With
    /// `None`, `build_note_hz` derives each phone's Hz from its own `note_pitch`, so a pre-rolled
    /// onset is never 0 Hz and the test passes with the repair DELETED — I wrote it that way first
    /// and the mutation probe caught it (S87 血训:「测试可能是瞎的」). The zero only exists because
    /// the DAW f0 is a per-FRAME array whose rest region is voiced=0.
    #[test]
    fn voiced_onset_carries_pitch_on_both_articulation_arms() {
        for timing in [ArticulationTiming::Auto, ArticulationTiming::InNote] {
            let ma = g2p::ScoreEvt {
                lyric: "x", note_num: 69, frames: 20, lang: g2p::Lang::Ja,
                phoneme_input: Some("m a"),
                phoneme_set: PhonemeSet::Words,
            };
            let score = [g2p::ScoreEvt::ja(&("R", 0, 10)), ma];
            // the DAW's layered pitch: 10 silent frames, then 20 frames of A4 (6900 cents)
            let cents: Vec<f32> = (0..30).map(|i| if i < 10 { 0.0 } else { 6900.0 }).collect();
            let voiced: Vec<u8> = (0..30).map(|i| u8::from(i >= 10)).collect();
            let vf0 = VocalF0 { cents: &cents, voiced: &voiced };
            let arr = build_arrays_daw(&score, &NoDicts, timing).unwrap();
            let mut hz = build_note_hz(&arr, &score, 0, Some(&vf0));
            zero_voiceless_frames(&mut hz, &arr);
            anchor_voiced_phone_f0(&mut hz, &arr);
            let mi = arr.phon.iter().position(|&p| p == "m").expect("the voiced onset survives");
            let start: i64 = arr.phone_dur[..mi].iter().sum();
            for f in start..(start + arr.phone_dur[mi]) {
                assert!(
                    hz[f as usize] > 0.0,
                    "{timing:?}: voiced onset frame {f} is 0 Hz — that is a muted consonant"
                );
            }
            // and the arms really are different here (else the loop proves one thing twice)
            if timing == ArticulationTiming::InNote {
                assert_eq!(arr.phone_dur[0], 10, "InNote leaves the rest whole");
            } else {
                assert!(arr.phone_dur[0] < 10, "Auto pre-rolls out of the rest");
            }
        }
    }

    // S84 B 刀: sung phones with dur ≤4 weigh 0 (no retrieval); longer phones and rests weigh 1.
    #[test]
    fn fast_index_weights_by_phone_duration() {
        // R(10) + fast こ chain (k:2/o:3, k:2/o:2 …) + a long note (its vowel > 4 frames).
        let arr = daw_ja(&[("R", 0, 10), ("こ", 73, 5), ("こ", 73, 4), ("こ", 73, 5), ("あ", 73, 20)]);
        let w = fast_index_weights(&arr);
        assert_eq!(w.len() as i64, arr.phone_dur.iter().sum::<i64>());
        let mut cursor = 0usize;
        for i in 0..arr.phon.len() {
            let d = arr.phone_dur[i].max(0) as usize;
            let expect = if arr.phon[i] == "SP" || arr.phone_dur[i] > 4 { 1.0 } else { 0.0 };
            assert!(
                w[cursor..cursor + d].iter().all(|&x| x == expect),
                "phone {} (dur {}) expected weight {expect}",
                arr.phon[i],
                arr.phone_dur[i]
            );
            cursor += d;
        }
        // the harmed regime is present in the fixture: at least one ≤4-frame sung vowel weighs 0.
        assert!(
            arr.phon.iter().enumerate().any(|(i, &p)| p == "o" && arr.phone_dur[i] <= 4),
            "fixture must contain a short vowel"
        );
    }

    // S84 review: the sibling classifiers (is_voiceless/is_nucleus) are exhaustiveness-pinned;
    // valley_depth_db was not, and ɟ/ɫ/ɭ/ɰ silently fell to the wildcard arm. Walk ALL 210 vocab
    // tokens: every consonant must land in a known class value, per-class counts are pinned so a
    // vocab regen (or a new IPA base char) forces a re-audit instead of a silent misroute.
    #[test]
    fn valley_depth_classification_is_exhaustive_and_stable() {
        use super::super::score2cv_tables::PHONE_TO_ID;
        use std::collections::HashMap;
        // spot anchors, both polarities of the S84 review fix:
        for (p, want) in [
            ("ɟ", 11.7f32), ("c", 11.7), ("ts", 11.7), ("dʑ", 11.7), ("ʔ", 11.7),
            ("ɫ", 11.4), ("ɭ", 11.4), ("l", 11.4), ("ɴ", 11.4),
            ("ɾ", 10.4), ("r", 10.4),
            ("s", 5.1), ("ç", 5.1), ("ʁ", 5.1),
            ("ɰ", 1.4), ("w", 1.4), ("ɹ", 1.4),
        ] {
            assert!((valley_depth_db(p) - want).abs() < 1e-6, "{p} must be {want}");
        }
        let mut counts: HashMap<i32, usize> = HashMap::new();
        let mut total = 0usize;
        for (p, _) in PHONE_TO_ID.iter() {
            if matches!(*p, "SP" | "AP" | "PAD" | "BOS" | "EOS")
                || super::super::score2cv::is_nucleus_phone(p)
            {
                continue;
            }
            total += 1;
            let d = valley_depth_db(p);
            *counts.entry((d * 10.0).round() as i32).or_default() += 1;
        }
        assert_eq!(total, 129, "consonant token count drifted — vocab regen? re-audit the classes");
        let n = |k: i32| counts.get(&k).copied().unwrap_or(0);
        assert_eq!(
            (n(117), n(114), n(104), n(51), n(14)),
            (58, 19, 5, 40, 7),
            "per-class counts drifted (stop/nasal-lateral/tap/fric/glide) — re-audit valley_depth_db: {counts:?}"
        );
    }

    #[test]
    fn valley_clusters_keep_member_depths() {
        // two contiguous flagged phones (an onset cluster) → ONE cluster, each member keeping its
        // OWN class depth (S84 review: a single max-depth window over-carved mixed clusters — an
        // EN [k w] must not sink the 1.4 dB glide to the stop's 11.7).
        let chunk = Chunk {
            start: 0,
            end: 4,
            phonemes: vec![10, 11, 12, 13],
            note_pitch: vec![60; 4],
            phone_dur: vec![50, 25, 25, 100],
            note_dur: vec![50, 25, 25, 100],
            note_to_phone: vec![0, 1, 2, 3],
            t: 200,
            lang_id: 2,
            hard_seam: false,
        };
        let flat: [Option<&[f32; VALLEY_ENV_N]>; 4] = [None; 4];
        let cls = chunk_valley_clusters(&chunk, 2000, &[0.0, 11.7, 10.4, 0.0], &flat);
        assert_eq!(cls.len(), 1);
        assert_eq!(cls[0].win, vec![(500, 750, 11.7), (750, 1000, 10.4)]);
        // a gap (unflagged phone) splits clusters:
        let cls2 = chunk_valley_clusters(&chunk, 2000, &[11.7, 0.0, 10.4, 0.0], &flat);
        assert_eq!(cls2.len(), 2);
        assert_eq!(cls2[0].win, vec![(0, 500, 11.7)]);
        assert_eq!(cls2[1].win, vec![(750, 1000, 10.4)]);
        // ⛔ S161b —— 一个簇只有一套形状,由**最深的成员**说了算(见 `ValleyCluster` 的 doc)。
        //    这里第二个成员更浅却带着窄槽形状 ⇒ 簇必须仍然是矩形。
        let shapes = [None, None, Some(&VALLEY_ENV_VSTOP), None];
        let cls3 = chunk_valley_clusters(&chunk, 2000, &[0.0, 11.7, 10.4, 0.0], &shapes);
        assert_eq!(cls3.len(), 1);
        assert!(cls3[0].env.is_none(), "最深的成员是矩形 ⇒ 整簇矩形");
        // 反过来:最深的成员带槽 ⇒ 整簇带槽。
        let shapes2 = [None, Some(&VALLEY_ENV_VSTOP), None, None];
        let cls4 = chunk_valley_clusters(&chunk, 2000, &[0.0, 11.7, 10.4, 0.0], &shapes2);
        assert!(cls4[0].env.is_some(), "最深的成员带包络 ⇒ 整簇带包络");
    }

    /// ★S97 ②a — WHO the phrase-final coda restore fires on. Four arms, deliberately different:
    /// phrase-final sonorant coda (fires) / the SAME word mid-phrase (does not) / a phrase-final
    /// OBSTRUENT coda (does not — real singers genuinely devoice those: upstream K −15.3, Z −9.9)
    /// / ja (structurally cannot — no phone ever sits after its last nucleus).
    #[test]
    fn s97_phrase_final_coda_lift_eligibility() {
        // Built directly so the arms differ ONLY in what this predicate reads (event grouping,
        // language, phone class, what follows) — a dictionary fixture would drag `resolve_west_span`
        // into a test about a render-side predicate.
        let mk = |phon: Vec<&'static str>, dur: Vec<i64>, evt: Vec<usize>, pitch: Vec<i64>, lang_id: i64| ScoreArrays {
            phonemes: vec![0; phon.len()],
            phone_dur: dur.clone(),
            note_pitch: pitch,
            note_dur: dur.clone(),
            note_to_phone: evt.iter().map(|&e| e as i64).collect(),
            phon,
            lang: vec![lang_id; dur.len()],
            evt,
            #[cfg(test)]
            borrow_ledger: Vec::new(),
            #[cfg(test)]
            in_note_alloc: Vec::new(),
        };
        let en_id = g2p::Lang::En.id();
        // bloom + rest → the /m/ is phrase-final and sonorant ⇒ eligible at the upstream p25
        let a = mk(vec!["b", "l", "u", "m", "SP"], vec![3, 4, 20, 5, 10], vec![0, 0, 0, 0, 1], vec![60, 60, 60, 60, 0], en_id);
        let la = phrase_final_coda_lifts(&a);
        assert_eq!(la[3].map(|x| x.0), Some(-5.0), "phrase-final /m/ targets upstream p25");
        assert_eq!(la[3].map(|x| a.phon[x.1]), Some("u"), "…measured against its own nucleus");
        assert!(la.iter().enumerate().all(|(i, v)| v.is_none() || i == 3), "nothing else is touched: {la:?}");
        // DISCRIMINATOR: the same word followed by a SUNG note must NOT fire (mid-phrase codas
        // measure −1.1 dB = already right). If this ever matches the arm above, the test is empty.
        let b = mk(vec!["b", "l", "u", "m", "s", "i"], vec![3, 4, 20, 5, 4, 20], vec![0, 0, 0, 0, 1, 1], vec![60; 6], en_id);
        let lb = phrase_final_coda_lifts(&b);
        assert_eq!(lb[3], None, "mid-phrase coda is left alone");
        assert_ne!(la[3].is_some(), lb[3].is_some(), "the two arms must really differ");
        // phrase-final OBSTRUENT: `dears` → the /ɹ/ lifts, the /z/ does not.
        let c = mk(vec!["d", "ɪ", "ɹ", "z", "SP"], vec![3, 12, 6, 3, 10], vec![0, 0, 0, 0, 1], vec![60, 60, 60, 60, 0], en_id);
        let lc = phrase_final_coda_lifts(&c);
        assert_eq!(lc[2].map(|x| x.0), Some(-3.9), "sonorant coda lifts");
        assert_eq!(lc[3], None, "a word-final /z/ is SUPPOSED to be quiet (upstream p50 −9.9)");
        // ja: gated out by language even if a coda-shaped phone existed (and in real ja material
        // none does — 0 of 1215 sung notes have a phone after their last nucleus).
        let j = mk(vec!["k", "a", "ɴ", "SP"], vec![3, 12, 4, 10], vec![0, 0, 0, 1], vec![60, 60, 60, 0], g2p::Lang::Ja.id());
        assert!(phrase_final_coda_lifts(&j).iter().all(|v| v.is_none()), "ja is untouched");
    }

    /// ★S97 ②a — HOW MUCH. Measures against the note's own nucleus, lifts toward the target, never
    /// attenuates, and stops at `CODA_LIFT_MAX_DB`.
    #[test]
    fn s97_coda_lift_is_bounded_and_one_sided() {
        let chunk = Chunk {
            start: 0, end: 2, phonemes: vec![10, 11],
            note_pitch: vec![60; 2], phone_dur: vec![100, 100], note_dur: vec![200; 2],
            note_to_phone: vec![0, 0], t: 200, lang_id: 1, hard_seam: false,
        };
        // nucleus at 1.0, coda 20 dB below it, target −5 dB ⇒ wants +15 dB, capped at 9.
        let mut a = vec![1.0f32; 200];
        for v in a[100..].iter_mut() { *v = 0.1; }
        apply_coda_lift(&mut a, &chunk, &[None, Some((-5.0, 0))], 4);
        let lifted = (a[150] / 0.1).max(1.0);
        assert!((20.0 * lifted.log10() - CODA_LIFT_MAX_DB).abs() < 0.2, "capped at the policy max: {}", 20.0 * lifted.log10());
        // already ABOVE the target ⇒ bit-exact untouched (this stage never attenuates).
        let mut b = vec![1.0f32; 200];
        for v in b[100..].iter_mut() { *v = 0.9; }
        let before = b.clone();
        apply_coda_lift(&mut b, &chunk, &[None, Some((-5.0, 0))], 4);
        assert_eq!(b, before, "a coda already at/above target is bit-exact untouched");
        // silent coda (nothing to lift) ⇒ untouched, never amplified noise.
        let mut c = vec![1.0f32; 200];
        for v in c[100..].iter_mut() { *v = 0.0; }
        let cbefore = c.clone();
        apply_coda_lift(&mut c, &chunk, &[None, Some((-5.0, 0))], 4);
        assert_eq!(c, cbefore, "a coda with no signal is not amplified");
    }

    #[test]
    fn apply_valley_shape_scale_junction_and_outside() {
        fn rect(win: Vec<(usize, usize, f32)>) -> ValleyCluster {
            ValleyCluster { win, env: None }
        }
        let mut a = vec![1.0f32; 300];
        apply_valley(&mut a, &[rect(vec![(20, 120, 20.0)])], 1.0, 10, 44100);
        assert_eq!(a[19], 1.0, "outside untouched");
        assert!((a[20] - 1.0).abs() < 0.11, "edge starts near unity");
        assert!((a[60] - 0.1).abs() < 1e-3, "plateau at −20 dB");
        assert!((a[119] - 1.0).abs() < 0.11, "far edge back near unity");
        // scale halves the dB depth (not the linear gain): −10 dB plateau.
        let mut b = vec![1.0f32; 300];
        apply_valley(&mut b, &[rect(vec![(20, 120, 20.0)])], 0.5, 10, 44100);
        assert!((b[60] - 10f32.powf(-0.5)).abs() < 1e-3, "scale is in dB domain");
        // mixed-depth cluster: each member plateaus at its OWN depth; the junction crossfades in
        // the dB domain (≈ −15 dB at the boundary) with NO unity bump anywhere inside the cluster.
        let mut c = vec![1.0f32; 300];
        apply_valley(&mut c, &[rect(vec![(20, 120, 20.0), (120, 220, 10.0)])], 1.0, 10, 44100);
        assert!((c[60] - 0.1).abs() < 1e-3, "member 1 plateaus at its own −20 dB");
        assert!((c[170] - 10f32.powf(-0.5)).abs() < 1e-3, "member 2 plateaus at its own −10 dB");
        let j = c[120];
        assert!(j > 0.1 && j < 10f32.powf(-0.5), "junction blends between the two depths, got {j}");
        assert!(
            c[31..210].iter().all(|&g| g < 0.75),
            "no unity bump inside the cluster (min plateau −10 dB, outer fade zones excluded)"
        );
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

    /// `range_shift != 0` had ZERO coverage in the whole Rust tree (S145). That matters because
    /// S145's headline finding rests on `transpose_eff = transpose + range_shift`: the pitch the
    /// model is fed — and therefore the f0 the inverse is later handed — is the SHIFTED one. If a
    /// refactor quietly dropped `range_shift` from that sum, the inverse would be seeded with the
    /// written pitch instead of the sung one, the mark search would hunt in the wrong band, and
    /// 601 cargo tests would all stay green. This is the cheap existence gate for that identity.
    /// Pure arithmetic — no model, no render.
    #[test]
    fn the_range_shift_really_reaches_the_pitch_the_model_is_fed() {
        let score = [("あ", 69, 4), ("い", 76, 4)];
        let evts = ja_evts(&score);
        let arr = daw_ja(&score);
        let base = build_note_hz(&arr, &evts, 0, None);
        for shift in [-6i64, -1, 3, 12] {
            // transpose_eff = transpose + range_shift, and transpose is 0 here, so a lone
            // range_shift must land as a pure ratio on every voiced frame.
            let moved = build_note_hz(&arr, &evts, shift, None);
            assert_eq!(moved.len(), base.len());
            let want = 2f32.powf(shift as f32 / 12.0);
            let mut voiced = 0;
            for (b, m) in base.iter().zip(moved.iter()) {
                if *b <= 0.0 {
                    assert_eq!(*m, 0.0, "an unvoiced frame must stay unvoiced under a shift");
                    continue;
                }
                voiced += 1;
                let got = m / b;
                assert!(
                    (got - want).abs() < 1e-3,
                    "range_shift {shift}: frame ratio {got} != {want} ({b} Hz → {m} Hz)"
                );
            }
            assert!(voiced >= 4, "the fixture must carry voiced frames, got {voiced}");
        }
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
        // S83 review #2: the identity fast-path must keep the rest → 0 Hz contract even when the
        // frontend's frame rounding leaves a voiced=1 frame just inside the SP window.
        let mut voiced2 = voiced.clone();
        let mut cents2 = cents.clone();
        voiced2[5] = 1;
        cents2[5] = 6000.0;
        let f0b = VocalF0 { cents: &cents2, voiced: &voiced2 };
        let hz2 = build_note_hz(&arr, &evts, 0, Some(&f0b));
        assert_eq!(hz2[5], 0.0, "an SP-owned frame stays 0 Hz regardless of the voiced mask");
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


    /// ⭐⭐⭐ S159zp —— **只填孤立单帧的无声洞**(用户 2026-08-23 点名的那一族)。
    ///
    /// 机理、坐标与归因在 [`fill_isolated_unvoiced`] 的 doc。这条判据钉四件:
    /// ⑴ 夹在两个浊帧之间的**单帧** 0 被填成两侧均值;
    /// ⑵ ⛔ **连续两帧的 0 一个都不许动** —— 那是真辅音的长度。
    ///    ⚠⚠ 我原以为这一条是主刀(「边扫边写会把两帧洞逐个补掉」),**真跑那条变异是绿的**:
    ///    谓词要求两侧都浊,而两连零的第一个 0 右邻还是 0 ⇒ 不成立 ⇒ 不会级联。
    ///    ⇒ `fill_isolated_uv` 里那句「先收集再写」是**防御性**的,没有判据背书。
    /// ⑶ 首尾的 0(没有一侧的锚)不许动;
    /// ⑷ 出厂默认(S159zp 起 = **开**);⛔ 旋钮两个方向都要认,垃圾值退回默认不许静默开启。
    ///
    /// ⛔ 变异(逐个真跑过):
    /// * `1..n - 1` 改成 `0..n`(也填首尾)⇒ ⑶ **红**;
    /// * [`FILL_ISOLATED_UV_DEFAULT`] 翻回 `false` ⇒ ⑷ **红**;
    /// * ⚠ **「先收集再写」改成边扫边写 ⇒ 绿** —— 见 ⑵ 那段:它挡的那件事**结构上不会发生**。
    #[test]
    fn only_a_single_frame_unvoiced_hole_between_voiced_frames_gets_filled() {
        // ⑶ 首尾 · ⑴ 单帧 · ⑵ 连续两帧 —— 一条轨里同时摆齐。
        let mut hz = vec![0.0f32, 0.0, 200.0, 0.0, 240.0, 300.0, 0.0, 0.0, 400.0, 0.0];
        let before = hz.clone();
        let k = fill_isolated_uv(&mut hz);
        assert_eq!(k, 1, "只有一个孤立单帧洞(读到 {k})");
        assert_eq!(hz[3], 220.0, "⑴ 单帧洞必须填成两侧均值 (200+240)/2");
        assert_eq!(
            (hz[6], hz[7]),
            (0.0, 0.0),
            "⑵ 连续两帧的 0 一个都不许动(读到 {:?})—— 边扫边写会把它们逐个补掉",
            (hz[6], hz[7])
        );
        assert_eq!((hz[0], hz[1], hz[9]), (0.0, 0.0, 0.0), "⑶ 首尾的 0 没有锚,不许动");
        assert_eq!(
            hz.iter().zip(&before).filter(|(a, b)| a != b).count(),
            1,
            "除了那一帧,其余必须逐位不变"
        );

        // ⑷ 出厂默认关 ⇒ 生产路径今天逐位不变。
        assert!(FILL_ISOLATED_UV_DEFAULT, "S159zp 已翻默认为开(实测洞浅 4.78 dB,对照 +0.01)");
        assert_eq!(parse_fill1(None), FILL_ISOLATED_UV_DEFAULT, "没设 env 时跟随出厂默认");
        assert!(parse_fill1(Some("1")) && !parse_fill1(Some("0")), "旋钮两个方向都要认");
        assert_eq!(
            parse_fill1(Some("垃圾")),
            FILL_ISOLATED_UV_DEFAULT,
            "垃圾值退回出厂默认,不许静默翻向任何一边"
        );

        // ⑸ ⛔ 阴性对照:一条**全是浊音**的轨,这把刀必须一个字节都不动。
        let mut voiced = vec![100.0f32, 200.0, 300.0, 400.0];
        let c = voiced.clone();
        assert_eq!(fill_isolated_uv(&mut voiced), 0);
        assert_eq!(voiced, c, "全浊音的轨不许被碰");
    }

    /// ⛔⛔ **承重**：S163 把 `apply_rest_gate` 的 fade 改成随窗长收缩之后，
    /// **长窗必须逐位不变** —— 40 ms 那个值是当年为**长休止**定的（用户 2026-08-28：
    /// 「当时长休止里面露出东西更恶劣，能直接拿大电噪把电平铺满」），
    /// 这一刀只补**从来没被覆盖过**的短窗，绝不许把当年治好的东西弄回来。
    #[test]
    fn rest_gate_is_bit_identical_on_windows_long_enough_for_the_full_fade() {
        let sr = 44_100u32;
        let fade = rest_gate_fade_samples(sr); // 40 ms
        // ⛔ 收缩到 `edge_max/2` 之后，`min` 不起作用的严格条件是 `((n−1)/2)/2 ≥ fade`，
        //    即 **n ≥ 4·fade+1**（比 2·fade+1 严，因为要留出中间一半窗长的平底）。
        for extra in [1usize, 2, 100] {
            let n = fade * 4 + extra;
            let src: Vec<f32> = (0..n + 200)
                .map(|i| ((i * 7919 % 1000) as f32 / 500.0) - 1.0)
                .collect();
            let mut a = src.clone();
            let mut b = src.clone();
            apply_rest_gate_with(&mut a, &[(100, 100 + n)], fade, true);
            apply_rest_gate_with(&mut b, &[(100, 100 + n)], fade, false);
            assert_eq!(a, b, "窗长 4·fade+{extra}：收缩版与固定版必须逐位相同");
        }
    }

    /// ⭐ 短窗（< 2·fade）的**中心必须真的归零** —— 这正是缺陷本身：
    /// 固定 40 ms 时 `keep = 1 − edge/fade` 恒 > 0，休止中间永远漏着模型渲的东西。
    /// 实测（鹅妈妈×yachiyo×+7，base 层，休止中段相对相邻音稳态）：
    /// 0-150 ms 的休止中段 p50 **−43.8 dB**、38% 高于 −40；而 ≥150 ms 的能到 **−280 dB**。
    #[test]
    fn rest_gate_closes_fully_even_on_windows_shorter_than_twice_the_fade() {
        let sr = 44_100u32;
        let fade = rest_gate_fade_samples(sr);
        for ms in [10usize, 20, 40, 60, 79] {
            let n = (ms * sr as usize) / 1000;
            if n < 4 {
                continue;
            }
            let mut x = vec![1.0f32; n + 200];
            apply_rest_gate_with(&mut x, &[(100, 100 + n)], fade, true);
            let mid = 100 + n / 2;
            assert!(
                x[mid].abs() < 1e-6,
                "{ms} ms 窗（{n} 样本）中心没归零：{} —— 门还是关不严",
                x[mid]
            );
            // 阴性对照：窗外一个字节都不许动
            assert_eq!(x[99], 1.0, "{ms} ms：窗前被动了");
            assert_eq!(x[100 + n], 1.0, "{ms} ms：窗后被动了");
        }
    }

    /// ⛔⛔ 短窗归零必须是**一段平底**，不是一两个样本 ——
    /// 第一版收缩到 `edge_max`，中心只有一两个样本真的到 0，
    /// 实机上两臂的零区间**逐段完全相同**（105 段 / 37.96 s），等于什么都没做。
    #[test]
    fn rest_gate_short_window_gets_a_real_flat_bottom_not_a_single_zero_sample() {
        let sr = 44_100u32;
        let fade = rest_gate_fade_samples(sr);
        for ms in [10usize, 20, 40, 60] {
            let n = (ms * sr as usize) / 1000;
            let mut x = vec![1.0f32; n + 200];
            apply_rest_gate_with(&mut x, &[(100, 100 + n)], fade, true);
            let zeros = (100..100 + n).filter(|&i| x[i] == 0.0).count();
            assert!(
                zeros as f32 >= n as f32 * 0.25,
                "{ms} ms 窗：只有 {zeros}/{n} 个样本归零 —— 那是一个点不是平底"
            );
        }
    }

    /// ⛔ 收缩之后**淡化仍然是连续的**（不是阶跃）——
    /// 窗再短也不许自己造出一个咔哒。判据：相邻样本的增益差有界。
    #[test]
    fn rest_gate_stays_a_continuous_ramp_after_shrinking() {
        let sr = 44_100u32;
        let fade = rest_gate_fade_samples(sr);
        for ms in [6usize, 12, 25, 50] {
            let n = (ms * sr as usize) / 1000;
            if n < 6 {
                continue;
            }
            let mut x = vec![1.0f32; n + 200];
            apply_rest_gate_with(&mut x, &[(100, 100 + n)], fade, true);
            let step = (100..100 + n - 1)
                .map(|i| (x[i] - x[i + 1]).abs())
                .fold(0.0f32, f32::max);
            // 半窗上从 1 走到 0 ⇒ 逐样本步长 ≈ 2/n；给 3× 余量
            assert!(
                step <= 12.0 / n as f32,
                "{ms} ms：逐样本增益跳变 {step} 太大（上界 {}）—— 那是阶跃不是斜坡",
                12.0 / n as f32
            );
        }
    }

    #[test]
    fn zero_voiceless_frames_zeroes_k_keeps_g() {
        // か = [k, a]: the voiceless k window zeroes its MEASURED central fraction (S83 knife 5 —
        // no longer the full window: RMVPE keeps edge f0 on real singing), the vowel keeps pitch.
        let score = [("か", 69, 40)];
        let arr = daw_ja(&score);
        let mut hz = build_note_hz(&arr, &ja_evts(&score), 0, None);
        assert!(hz.iter().all(|&h| h > 0.0), "pre: whole note voiced");
        zero_voiceless_frames(&mut hz, &arr);
        let k = arr.phone_dur[0] as usize;
        let permille =
            super::super::score2cv::voiceless_zero_permille("k", arr.note_dur[0], super::g2p::Lang::Ja);
        let z = (k as f64 * permille as f64 / 1000.0).round() as usize;
        assert!(z >= 1 && z < k, "k zeroes a nonzero PARTIAL core (got z={z} of {k})");
        let zeroed = hz[..k].iter().filter(|&&h| h == 0.0).count();
        assert_eq!(zeroed, z, "k zeroes exactly the measured fraction");
        // centering rounds LEFT, so with z<k the vowel-adjacent edge is ALWAYS preserved (the
        // pre-voicing frame — the perceptually load-bearing end; a 2-frame window zeroes its head).
        assert!(hz[k - 1] > 0.0, "vowel-adjacent edge keeps the pre-voicing f0");
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
        // ⭐ S160q:出厂已从「复制两遍」翻成**浊音游程内的中点插值**。
        //    这条断言此前钉的是旧行为,翻默认时它按设计红了一次 —— 那正是它的作用。
        assert_eq!(
            (pitchf[0], pitchf[1]),
            (110.0, 165.0),
            "出厂 = 中点插值(110→220 的中点 165);要看旧的零阶保持臂,见              `parse_score_f0_lerp` 与 `upsample_note_hz_linear` 的判据"
        );
        assert_eq!((pitchf[2], pitchf[3]), (220.0, 275.0), "偶数格 = 原采样点,奇数格 = 中点");
        assert_eq!(pitchf[11], 330.0, "padded frames repeat the last note_hz");
        // ⛔ 末尾那一格没有「下一个采样点」⇒ 必须退回保持,不许外推。
        assert_eq!((pitchf[4], pitchf[5]), (330.0, 330.0), "最后一个真实帧不外推");
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

        // These wavs are the ARCHIVED ear-A/B baselines: they predate the S84 valley and clarity
        // knives, so both stay OFF here and only the emphasis knob rides at its production default.
        // Keeping the deviation in ONE named value (instead of re-typing the same two literals +
        // comment on every call) is what makes it obvious these are NOT production settings.
        let baseline_shaping = ScoreShaping {
            consonant_emphasis_db: DEFAULT_VOICELESS_ONSET_EMPHASIS_DB,
            consonant_valley_scale: 0.0,
            vowel_clarity: false,
            ..Default::default() // preroll: the baselines were rendered with it ON
        };

        // akiko 4.0 / 256 (vol-free — cleanest audible on the 256 path)
        let akiko = engine.load_model_with(&sov.join("akiko_320000.onnx"), false).unwrap();
        let am = sov_model(&engine, &akiko, &cv256, &rmvpe, &rmvpe_mel, 256, false);
        save("p2_akiko256_main", &render_score_sovits(&am, &s2cv256, &ja_evts(pr::SCORE), 256, 49, &NoDicts, &sopts, 0.0, baseline_shaping, 0, 0, None, None, None, &no_cancel, &no_prog, None).unwrap());
        save("p2_akiko256_demo_legato", &render_score_sovits(&am, &s2cv256, &ja_evts(LEGATO), 256, 49, &NoDicts, &sopts, 0.0, baseline_shaping, 0, 0, None, None, None, &no_cancel, &no_prog, None).unwrap());
        save("p2_akiko256_demo_rest", &render_score_sovits(&am, &s2cv256, &ja_evts(REST), 256, 49, &NoDicts, &sopts, 0.0, baseline_shaping, 0, 0, None, None, None, &no_cancel, &no_prog, None).unwrap());
        save("p2_akiko256_demo_sustain_same", &render_score_sovits(&am, &s2cv256, &ja_evts(SUSTAIN), 256, 49, &NoDicts, &sopts, 0.0, baseline_shaping, 0, 0, None, None, None, &no_cancel, &no_prog, None).unwrap());
        save("p2_akiko256_demo_reartic_same", &render_score_sovits(&am, &s2cv256, &ja_evts(REARTIC), 256, 49, &NoDicts, &sopts, 0.0, baseline_shaping, 0, 0, None, None, None, &no_cancel, &no_prog, None).unwrap());

        // 东雪莲 4.1 / 768 (SAME voice as the Python reference; vol_embedding → flat placeholder vol)
        let dx = engine.load_model_with(&sov.join("Sovits4.1东雪莲主模型.onnx"), false).unwrap();
        let dm = sov_model(&engine, &dx, &cv768, &rmvpe, &rmvpe_mel, 768, true);
        save("p2_dongxuelian768_main", &render_score_sovits(&dm, &s2cv768, &ja_evts(pr::SCORE), 768, 49, &NoDicts, &sopts, 0.1, baseline_shaping, 0, 0, None, None, None, &no_cancel, &no_prog, None).unwrap());

        // RVC v2 lengv2 / 768 (100 fps grid; no Python A/B reference — audible + glue self-consistency)
        let leng = engine.load_model_with(&rvcd.join("lengv2.3.onnx"), false).unwrap();
        let rm = RvcModel {
            engine: &engine, voice_session: &leng, contentvec_session: &cv768, rmvpe_session: &rmvpe,
            mel_filters: &rmvpe_mel, index: None, sample_rate: 48000, features_dim: 768, spk_mix: None,
            noise_channels: 192, min_frames: 12,
        };
        save("p2_rvc_lengv2_main", &render_score_rvc(&rm, &s2cv768, &ja_evts(pr::SCORE), 768, 49, &NoDicts, &ropts, baseline_shaping, 0, 0, None, None, None, &no_cancel, &no_prog, None).unwrap());

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

#[cfg(test)]
mod s160k_uvgate_tests {
    use super::{gate_unvoiced_tone, highpass_span};

    fn tone(n: usize, sr: f32, f: f32) -> Vec<f32> {
        (0..n).map(|i| (2.0 * std::f32::consts::PI * f * i as f32 / sr).sin()).collect()
    }
    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / x.len() as f64).sqrt() as f32
    }

    /// S160k —— 高通只作用在 `[a, b)`,而且**段外一个样本都不许动**。
    #[test]
    fn the_highpass_touches_only_its_own_span() {
        let sr = 44100u32;
        let mut x = tone(sr as usize / 2, sr as f32, 500.0);
        let before = x.clone();
        let (a, b) = (8000usize, 12000usize);
        highpass_span(&mut x, sr, a, b, 1500.0, 200);
        for t in 0..a {
            assert_eq!(x[t], before[t], "段前第 {t} 个样本被动了");
        }
        for t in b..x.len() {
            assert_eq!(x[t], before[t], "段后第 {t} 个样本被动了");
        }
        // 段【正中】(离交叉淡化远)必须被明显衰减:500 Hz 远在 1500 Hz 截止之下。
        let mid = &x[a + 1000..b - 1000];
        let ref_ = &before[a + 1000..b - 1000];
        let att = 20.0 * (rms(mid) / rms(ref_)).log10();
        assert!(att < -18.0, "500 Hz 在 1500 Hz 高通下该被压掉 ≥18 dB,实际 {att:.1} dB");
    }

    /// S160k —— 截止**之上**的成分要基本原样过去(不然它连辅音噪声一起吃掉)。
    #[test]
    fn the_highpass_keeps_what_is_above_the_cutoff() {
        let sr = 44100u32;
        let mut x = tone(sr as usize / 2, sr as f32, 4000.0);
        let before = x.clone();
        let (a, b) = (8000usize, 12000usize);
        highpass_span(&mut x, sr, a, b, 1500.0, 200);
        let att = 20.0
            * (rms(&x[a + 1000..b - 1000]) / rms(&before[a + 1000..b - 1000])).log10();
        assert!(att > -1.5, "4 kHz 该基本过得去,实际 {att:.1} dB");
    }

    /// S160k —— 门只碰 `note_hz == 0` 的连续帧,而且**必须有一个浊音参照**才动手。
    #[test]
    fn the_gate_only_fires_on_unvoiced_runs_that_have_a_voiced_neighbour() {
        let sr = 44100u32;
        let hop = sr as usize / 50;
        // 20 帧:0-7 浊音 494 Hz,8-11 清音(0),12-19 浊音
        let mut hz = vec![494.0f32; 20];
        for t in 8..12 {
            hz[t] = 0.0;
        }
        let mut x = tone(20 * hop, sr as f32, 494.0);
        let before = x.clone();
        let hit = gate_unvoiced_tone(&mut x, sr, &hz, 1.5, 4.0, 0.0);
        assert_eq!(hit, 1, "只该命中一段");
        // 清音段正中被压掉(494 Hz 在 1.5×494 = 741 Hz 截止之下)
        let (a, b) = (8 * hop, 12 * hop);
        let att = 20.0 * (rms(&x[a + 400..b - 400]) / rms(&before[a + 400..b - 400])).log10();
        assert!(att < -10.0, "清音段里的 494 Hz 该被压掉,实际 {att:.1} dB");
        // ⛔ 浊音段一个样本都不许动
        for t in 0..a {
            assert_eq!(x[t], before[t], "浊音段(前)第 {t} 个样本被动了");
        }
        for t in b..x.len() {
            assert_eq!(x[t], before[t], "浊音段(后)第 {t} 个样本被动了");
        }

        // ⛔ 全清音(没有任何浊音参照)⇒ 一个字不动
        let hz0 = vec![0.0f32; 20];
        let mut y = before.clone();
        assert_eq!(gate_unvoiced_tone(&mut y, sr, &hz0, 1.5, 4.0, 0.0), 0);
        assert_eq!(y, before);

        // ⛔ 全浊音 ⇒ 一个字不动
        let hz1 = vec![494.0f32; 20];
        let mut z = before.clone();
        assert_eq!(gate_unvoiced_tone(&mut z, sr, &hz1, 1.5, 4.0, 0.0), 0);
        assert_eq!(z, before);
    }

    /// S162 —— **起音护栏**:清音 run 的【后】端留 `guard_ms` 不碰,而【前】端照旧。
    ///
    /// ⛔ 为什么要这条:S160k 那把刀**用户耳判确认有效**却一直没翻默认,原因是全曲对拍
    /// 发现它在 **46 条段里削掉真起音**(0.76 s,最深 −37.5 dB)—— 模型常把浊音渲得比
    /// 乐谱音头早几十毫秒。护栏就是这条尾巴;没有一条判据钉住它「保的是后端不是前端」,
    /// 下一个人把 `b` 写成 `a` 或者两端都收都不会红。
    #[test]
    fn the_onset_guard_spares_the_tail_of_the_run_and_only_the_tail() {
        let sr = 44100u32;
        let hop = sr as usize / 50; // 882
        // 20 帧:0-7 浊音,8-15 清音(8 帧 = 160 ms),16-19 浊音
        let mut hz = vec![494.0f32; 20];
        for t in 8..16 {
            hz[t] = 0.0;
        }
        let x0 = tone(20 * hop, sr as f32, 494.0);
        let (a, b) = (8 * hop, 16 * hop);

        // 护栏 0 = 老行为:整段被压
        let mut none = x0.clone();
        assert_eq!(gate_unvoiced_tone(&mut none, sr, &hz, 1.5, 4.0, 0.0), 1);

        // 护栏 40 ms = 1764 样本:后 40 ms 必须**逐位不变**
        let mut g40 = x0.clone();
        assert_eq!(gate_unvoiced_tone(&mut g40, sr, &hz, 1.5, 4.0, 40.0), 1, "护栏不该让它不开火");
        let guard = (0.040 * sr as f32) as usize;
        for t in (b - guard)..b {
            assert_eq!(g40[t], x0[t], "护栏内第 {t} 个样本被门碰了");
        }
        // ⛔ 阴性对照:同样长度的**前**端必须仍然被压(护栏只保后端)
        let att_head = 20.0
            * (rms(&g40[a + 400..a + guard]) / rms(&x0[a + 400..a + guard])).log10();
        assert!(att_head < -10.0, "run 的前端不该被护栏保住,实际只衰减 {att_head:.1} dB");
        // ⛔ 而 guard=0 的那条臂在同一段后端上**必须**被压 —— 否则这条判据是空的
        let att_tail0 = 20.0
            * (rms(&none[b - guard..b - 400]) / rms(&x0[b - guard..b - 400])).log10();
        assert!(att_tail0 < -10.0, "guard=0 时后端本该被压,实际 {att_tail0:.1} dB ⇒ 判据是空的");

        // ⛔ 没有后继浊音(run 一直到结尾)⇒ 护栏不许生效(没有起音可保)
        let mut hz_end = vec![494.0f32; 20];
        for t in 8..20 {
            hz_end[t] = 0.0;
        }
        let mut tail = x0.clone();
        assert_eq!(gate_unvoiced_tone(&mut tail, sr, &hz_end, 1.5, 4.0, 40.0), 1);
        let e = 20 * hop;
        let att_eof = 20.0 * (rms(&tail[e - guard..e - 400]) / rms(&x0[e - guard..e - 400])).log10();
        assert!(att_eof < -10.0, "曲尾那段没有后继浊音,护栏不该收,实际 {att_eof:.1} dB");
    }

    /// S162 —— 出厂默认:**门开着 + 护栏 20 ms**,而且垃圾值不许静默改变出厂臂。
    ///
    /// ⛔ 没有这条判据,「我们翻了」与「有人翻回去了」在别的每一条测试上长得一模一样。
    /// 账在 `parse_uvgate` 的 doc 上(治愈 −13.0 dB / 真起音区代价 28 → 8 格)。
    #[test]
    fn the_gate_ships_on_with_a_twenty_millisecond_onset_guard() {
        use super::{parse_uvgate, parse_uvgate_guard_ms};
        assert!(parse_uvgate(None), "出厂必须【开】(S162 翻的;UTAI_MG_UVGATE=0 才关)");
        assert!(!parse_uvgate(Some("0")), "字面量 0 必须关得掉");
        for junk in ["", "true", "yes", "2", "on", " 0", "false"] {
            assert!(parse_uvgate(Some(junk)), "垃圾值 {junk:?} 不许静默关掉出厂臂");
        }
        assert_eq!(parse_uvgate_guard_ms(None), 20.0, "出厂护栏必须是 20 ms");
        assert_eq!(parse_uvgate_guard_ms(Some("0")), 0.0, "旋钮要能退回 S160k 那一版");
        assert_eq!(parse_uvgate_guard_ms(Some("40")), 40.0);
        assert_eq!(parse_uvgate_guard_ms(Some(" 25.5 ")), 25.5);
        for junk in ["", "abc", "-1", "nan", "inf", "1e9", "201"] {
            assert_eq!(parse_uvgate_guard_ms(Some(junk)), 20.0, "垃圾值 {junk:?} 必须落回出厂");
        }
    }
}

// ─── S160q:50→100 fps 的 f0 线性升采样 ────────────────────────────────────────────────
#[cfg(test)]
mod s160q_f0_lerp_tests {
    use super::*;

    #[test]
    fn the_default_is_on_and_only_the_literal_zero_turns_it_off() {
        // ⛔ S160q 翻成出厂开。没有这条判据,「我们翻了」和「有人翻回去了」在别的每一条
        //    测试上长得一模一样。四个模型的净账写在 `score_f0_lerp` 的 doc 上。
        assert!(parse_score_f0_lerp(None), "出厂必须开(S160q;UTAI_SCORE_F0_LERP=0 才关)");
        assert!(!parse_score_f0_lerp(Some("0")), "字面量 0 必须关得掉");
        for junk in ["", "true", "yes", "2", "on", " 0", "false"] {
            assert!(parse_score_f0_lerp(Some(junk)), "垃圾值 {junk:?} 不许静默关掉出厂臂");
        }
    }

    #[test]
    fn the_fingerprint_covers_every_audio_changing_knob_in_this_file() {
        // ⛔⛔ 这条判据存在的理由:S160q 之前,本文件的生产默认【完全不在任何指纹里】,
        //     而 `FILL_ISOLATED_UV_DEFAULT` 出厂就是开着的。
        let fp = production_defaults_fingerprint();
        for key in ["f0lerp=", "fill1=", "filluv=", "fillmax=", "uvgate=", "uvgatek=", "uvgateguard=", "valadapt=", "valafter=", "valhuman=", "valdb=", "valenv="] {
            assert!(fp.contains(key), "指纹串缺 {key} —— 少一个默认就少一道成对 bump 的闸:{fp}");
        }
    }

    /// ⛔⛔ S161f/g —— **接缝淡化必须无条件跑**,而且**只在真有接缝时**跑。
    ///
    /// 这是一条**没有旋钮的行为改动** ⇒ `production_defaults_fingerprint()` 结构上盖不住它,
    /// 谁把 `if chunk.hard_seam` 加回去,别的每一条测试都还是绿的(S161 的血训:
    /// 「只登记旋钮 ⇒ 改类常量零红」的同一族)。所以这里同时钉住**源码调用点**与**行为**。
    ///
    /// 读数(鹅妈妈原 key,>18 kHz 盲搜全曲,突出度 >20 dB 的瞬变):
    /// 谷全关 30 · S161e 硬拼接缝 32 · **S161f 接缝淡化 5**;用户点的四处 43.9/38.9/40.1/35.4 dB
    /// 全部消失;残留的 5 个 **100% 落在救援窗内**(窗只覆盖全曲 6% ⇒ 不是空判据)。
    #[test]
    fn the_seam_fade_runs_on_every_chunk_seam() {
        let src = include_str!("score2svc.rs");
        let call = concat!("seam_", "fade(&mut audio, &mut wav, m.sample_rate);");
        let n = src.matches(call).count();
        assert_eq!(n, 2, "两条渲染链(SoVITS/RVC)各要有一个接缝淡化调用点,现在有 {n} 个");
        for (i, _) in src.match_indices(call) {
            // ⛔ 剥掉行注释再查 —— 否则调用点上那句「以前只 hard_seam」的注释自己就把判据点红了
            //    (第一次跑就是这么红的:红对了位置、红错了原因)。
            let before: String = src[i.saturating_sub(400)..i]
                .lines()
                .map(|l| l.split("//").next().unwrap_or(""))
                .collect::<Vec<_>>()
                .join("
");
            assert!(
                !before.contains("hard_seam"),
                "接缝淡化又被 `hard_seam` 圈起来了 —— S161f 的结论是 **SP 接缝才是主体**
                 (SP 两侧是两次独立解码的缓冲,硬拼 = 波形台阶 = 宽带咔哒;
                  实测 >20 dB 的 >18 kHz 瞬变 84-87% 落在 SP 边 20 ms 内,且把辅音谷全关读数一模一样)"
            );
        }
    }

    /// ⛔ S161g —— 上面那一刀的**边界**:第一块没有左侧,那不是接缝。
    /// 阴性对照就在同一条判据里:有左侧时**必须**淡。
    #[test]
    fn the_first_chunk_is_never_faded_in() {
        let mut empty: Vec<f32> = Vec::new();
        let mut head = vec![1.0f32; 4410];
        seam_fade(&mut empty, &mut head, 44100);
        assert!(head.iter().all(|&v| (v - 1.0).abs() < 1e-9), "整首歌的头 5 ms 不许被淡入");
        // 阴性对照:真有左侧 ⇒ 两侧都必须动,否则这条判据是空的。
        let mut tail = vec![1.0f32; 4410];
        let mut head2 = vec![1.0f32; 4410];
        seam_fade(&mut tail, &mut head2, 44100);
        assert!(tail[4409] < 0.01 && head2[0] < 0.01, "真接缝上两侧都要淡到近零");
        assert!((tail[0] - 1.0).abs() < 1e-9 && (head2[4409] - 1.0).abs() < 1e-9, "只许动接缝那 5 ms");
    }

    /// ⛔⛔ S161e —— **谷的增益曲线在簇窗边缘必须连续**。
    ///
    /// S161d 的包络路径直接铺模板值(首格 0.82)⇒ 窗起点是**一样本的 ~10 dB 台阶**,
    /// 而台阶 = 宽带咔哒。实测(鹅妈妈原 key,211 个簇)>16 kHz 在窗起点的尖峰 **+27.15 dB**,
    /// 矩形 / 窄槽 / 谷全关都是 −2 dB。**用户在频谱图 16 k 以上一眼看到一堆细竖线。**
    /// ⇒ 这条判据把「逐样本增益的最大跳变」钉住:任何形状、任何深度都不许在边缘跳。
    #[test]
    fn the_valley_gain_never_steps_at_a_cluster_edge() {
        for env in [None, Some(&VALLEY_ENV_VSTOP), Some(&VALLEY_ENV_TAP)] {
            for depth in [6.0f32, 12.0, 20.0] {
                let n = 4410usize; // 100 ms @44.1k
                let mut a = vec![1.0f32; n];
                let (s, e) = (1000usize, 3000usize);
                apply_valley(
                    &mut a,
                    &[ValleyCluster { win: vec![(s, e, depth)], env }],
                    1.0,
                    emphasis_fade_samples(44100),
                    44100,
                );
                assert!((a[s - 1] - 1.0).abs() < 1e-6, "窗外必须原样");
                assert!((a[e] - 1.0).abs() < 1e-6, "窗外必须原样");
                // 逐样本增益跳变(dB)。5 ms 斜坡上 20 dB ⇒ 每样本 ~0.09 dB,给 10 倍余量。
                let mut worst = 0.0f32;
                for i in (s - 2)..(e + 2).min(n - 1) {
                    let d = (20.0 * (a[i + 1].max(1e-9) / a[i].max(1e-9)).log10()).abs();
                    worst = worst.max(d);
                }
                assert!(
                    worst < 1.0,
                    "env={:?} depth={depth}: 增益逐样本最大跳变 {worst:.2} dB —— 边缘有台阶 = 宽带咔哒",
                    env.map(|_| "human")
                );
            }
        }
    }

    /// ⛔⛔ S161 —— **把上面那条判据的洞堵上**:它只检查「已知的那几个 key 还在」,
    /// 于是**加一个新旋钮却不登记进指纹 = 零红**(S161 的 recon 当场指出这个洞)。
    /// 这条改成**反向**判据:从本文件源码里数出所有被读的 `UTAI_*` 环境变量,
    /// 除掉登记在案的豁免项,剩下的**个数必须等于**指纹里的字段数。
    /// ⇒ 加旋钮不加指纹 ⇒ 当场红,而且红的措辞直接告诉他去哪儿加。
    #[test]
    fn every_env_knob_in_this_file_is_registered_in_the_fingerprint() {
        // (env 变量名, 它在指纹里的 key)。⛔ 加旋钮不加这一行 ⇒ 当场红。
        const MAP: &[(&str, &str)] = &[
            ("UTAI_SCORE_F0_LERP", "f0lerp="),
            ("UTAI_MG_FILL1", "fill1="),
            ("UTAI_MG_FILL_MAX", "fillmax="),
            ("UTAI_MG_UVGATE", "uvgate="),
            ("UTAI_MG_UVGATE_K", "uvgatek="),
            ("UTAI_MG_UVGATE_GUARD_MS", "uvgateguard="),
            ("UTAI_MG_VALLEY_ADAPT", "valadapt="),
            ("UTAI_MG_VALLEY_AFTER", "valafter="),
            ("UTAI_VALLEY_HUMAN", "valhuman="),
            ("UTAI_REST_GATE_SHRINK", "restshrink="),
            ("UTAI_PREROLL_DAMP", "predamp="),
        ];
        // 只写文件、不改音频的诊断路径(见 `dump_donor_buffer` 的 doc)。
        const EXEMPT: &[&str] = &["UTAI_RANGE_DUMP_DONOR"];
        // 指纹里**不由 env 驱动**的格:纯常量默认,没有对应的环境变量。
        const CONST_ONLY: &[&str] = &["filluv=", "valdb=", "valenv="];

        let src = include_str!("score2svc.rs");
        let mut found: Vec<&str> = Vec::new();
        for (i, _) in src.match_indices("env::var(\"UTAI_") {
            let rest = &src[i + "env::var(\"".len()..];
            if let Some(end) = rest.find('"') {
                let name = &rest[..end];
                if !found.contains(&name) && !EXEMPT.contains(&name) {
                    found.push(name);
                }
            }
        }
        for name in &found {
            assert!(
                MAP.iter().any(|(n, _)| n == name),
                "本文件读了 {name},但它既不在指纹映射表里也不在豁免表里。
                 ⇒ 新旋钮必须同时:①做一个 `parse_*(Option<&str>)` 纯函数                  ②在 `production_defaults_fingerprint()` 里加一格                  ③在本判据的 MAP 里加一行                  ④去 `vocal_range.rs` 的 `changing_a_production_default_forces_a_paired_version_bump`                  补期望串(**若它出厂关 = 不改音频,就不要 bump 版本号**,理由见那里的 S159 注释)。"
            );
        }
        for (name, _) in MAP {
            assert!(found.contains(name), "MAP 里的 {name} 在本文件里已经没人读了 —— 删旋钮也要删这一行");
        }
        let fp = production_defaults_fingerprint();
        for (_, key) in MAP {
            assert!(fp.contains(key), "指纹串缺 {key}:{fp}");
        }
        let fields = fp.split(' ').filter(|f| f.contains('=')).count();
        assert_eq!(
            fields,
            MAP.len() + CONST_ONLY.len(),
            "指纹有 {fields} 格,但映射表 {} + 纯常量格 {} 只解释得了 {}:{fp}",
            MAP.len(),
            CONST_ONLY.len(),
            MAP.len() + CONST_ONLY.len()
        );
    }

    /// ⛔ S161/S161b —— 人表**只许**在量过的那三个桶上与出厂表不同(鼻/边 · 闪 · 浊塞)。
    /// 走的是与 `valley_depth_db` 那条穷举判据同一份 210 音素词表。
    #[test]
    fn the_human_table_differs_only_in_the_measured_buckets() {
        // ⛔ S161 翻成出厂开(用户耳判)。没有这条判据,「我们翻了」和「有人翻回去了」
        //    在别的每一条测试上长得一模一样。全曲净账写在 `parse_valley_human` 的 doc 上。
        assert!(parse_valley_human(None), "出厂必须开(S161;UTAI_VALLEY_HUMAN=0 才关)");
        assert!(!parse_valley_human(Some("0")), "字面量 0 必须关得掉");
        for junk in ["", "1", "true", "yes", "on", "2", " 0", "false"] {
            assert!(parse_valley_human(Some(junk)), "垃圾值 {junk:?} 不许静默关掉出厂臂");
        }
        use super::super::score2cv_tables::PHONE_TO_ID;
        let mut moved = 0usize;
        for (p, _) in PHONE_TO_ID.iter() {
            let p: &str = p;
            if matches!(p, "SP" | "AP" | "PAD" | "BOS" | "EOS") || super::super::score2cv::is_nucleus_phone(p) {
                continue;
            }
            let a = valley_depth_db(p);
            let b = valley_depth_db_human(p, 3);
            let nasal = matches!(p.chars().next(), Some('m' | 'n' | 'ɲ' | 'ŋ' | 'ɴ' | 'l' | 'ʎ' | 'ɫ' | 'ɭ'));
            let tap = matches!(p.chars().next(), Some('ɾ' | 'ɽ' | 'r'));
            let vstop = matches!(p.chars().next(), Some('b' | 'd' | 'ɡ' | 'ɟ' | 'ɢ'));
            // ⛔ 形状表必须与深度表**同步**:开槽的正好是「深度被改大了」的那一族。
            assert_eq!(valley_shape_human(p).is_some(), tap || vstop, "{p} 的形状与深度表对不上");
            if nasal {
                assert!((b - VALLEY_NASAL_HUMAN_DB).abs() < 1e-6, "{p} 应是鼻/边人表值");
                moved += 1;
            } else if tap {
                assert!((b - valley_tap_human_db(3)).abs() < 1e-6, "{p} 应是闪音人表值");
                moved += 1;
            } else if vstop {
                assert!((b - valley_vstop_human_db(3)).abs() < 1e-6, "{p} 应是浊塞音人表值");
                // ⛔ 清塞音一个都不许进来(首字符不重叠 —— 这条就是钉那句话的)
                assert!(!super::super::score2cv::is_voiceless_phone(p), "{p} 是清音,不该走浊塞分支");
                moved += 1;
            } else {
                assert!((a - b).abs() < 1e-6, "{p} 不该被人表动到:{a} vs {b}");
            }
        }
        assert!(moved >= 12, "只动到 {moved} 个音素,词表或分支坏了");
    }

    #[test]
    fn the_upsample_never_moves_a_voiced_boundary() {
        // ⛔⛔ 硬规矩:辅音时序一个字节不许动。`uv` 是 resize【之后】按 `f0 > 0` 推的,
        //     所以只要零帧的【图案】被原样放大,浊/清边界就一帧都不会移。
        //     实测(整曲 25073 帧 @86 fps):0 处不同、190 个边界位移 0.0 ms。
        for factor in [2usize, 3, 4] {
            let src = [0.0f32, 0.0, 220.0, 230.0, 0.0, 240.0, 250.0, 0.0, 0.0, 260.0];
            let up = upsample_note_hz_linear(&src, factor);
            assert_eq!(up.len(), src.len() * factor);
            for (i, &v) in src.iter().enumerate() {
                for k in 0..factor {
                    let z = up[i * factor + k] == 0.0;
                    assert_eq!(z, v == 0.0, "factor={factor} 源帧 {i} 的第 {k} 个子帧零性变了");
                }
            }
        }
    }

    #[test]
    fn a_voiced_ramp_gets_real_midpoints_and_the_length_is_exact() {
        let src = [100.0f32, 200.0, 400.0];
        let got = upsample_note_hz_linear(&src, 2);
        assert_eq!(got.len(), 6, "长度必须是 n×factor");
        // 偶数格 = 原采样点;奇数格 = 与下一点的中点;最后一点没有下一点 ⇒ 保持自己。
        assert_eq!(got, vec![100.0, 150.0, 200.0, 300.0, 400.0, 400.0]);
        let g4 = upsample_note_hz_linear(&[0.0f32.max(100.0), 200.0], 4);
        assert_eq!(g4, vec![100.0, 125.0, 150.0, 175.0, 200.0, 200.0, 200.0, 200.0]);
    }

    #[test]
    fn a_zero_on_either_side_falls_back_to_hold_and_zeros_stay_exactly_zero() {
        // ⛔⛔ 这是这把刀唯一会造成灾难的地方:插进休止 = 把音高抹进不该发声的帧,
        //     破坏「pitchf == 0 ⇒ NSF 噪声激励 + protect」的契约(S83 的哑起音)。
        let src = [200.0f32, 0.0, 0.0, 300.0, 400.0];
        let got = upsample_note_hz_linear(&src, 2);
        assert_eq!(got, vec![200.0, 200.0, 0.0, 0.0, 0.0, 0.0, 300.0, 350.0, 400.0, 400.0]);
        // 零帧的个数与位置逐位可预测:源里每个 0 变成 factor 个 0。
        let z_src = src.iter().filter(|v| **v == 0.0).count();
        let z_out = got.iter().filter(|v| **v == 0.0).count();
        assert_eq!(z_out, z_src * 2, "零帧个数必须正好 ×factor");
        assert!(got.iter().all(|v| *v == 0.0 || *v >= 200.0), "不许在零两侧造出中间值");
    }

    #[test]
    fn factor_one_and_empty_are_the_identity() {
        let src = [100.0f32, 0.0, 300.0];
        assert_eq!(upsample_note_hz_linear(&src, 1), src.to_vec());
        assert_eq!(upsample_note_hz_linear(&src, 0), src.to_vec());
        assert!(upsample_note_hz_linear(&[], 2).is_empty());
    }

    #[test]
    fn the_upsampled_track_is_what_actually_removes_the_20ms_tread() {
        // 变异判据:零阶保持时,相邻 100 fps 帧【有一半】完全相等;线性插值之后,
        // 浊音游程内部不应再有相等的相邻对(单调斜坡上)。
        let src: Vec<f32> = (0..8).map(|i| 200.0 + 20.0 * i as f32).collect();
        let hold: Vec<f32> = src.iter().flat_map(|&f| [f, f]).collect();
        let lerp = upsample_note_hz_linear(&src, 2);
        let eq = |v: &[f32]| v.windows(2).filter(|w| w[0] == w[1]).count();
        assert_eq!(eq(&hold), 8, "零阶保持:8 对相等(每个源帧一对)");
        assert_eq!(eq(&lerp), 1, "线性:只剩末尾那一对(最后一点没有下一点)");
    }
}
