//! GAME (openvpi) — singing-voice audio → MIDI note extraction (S60-1).
//!
//! Runs the OFFICIAL 1.0.3 ONNX deployment pipeline (5 graphs + config.json under
//! models/auxiliary/game/): encoder → D3PM segmenter loop → bd2dur → estimator, then the
//! official cross-chunk note assembly. References, in authority order:
//!   - GAME ONNX.md (tensor contract) — verified against the real graphs;
//!   - OpenUtau Game.cs (the openvpi-blessed ONNX consumer: t = i/K ascending,
//!     known/prev boundaries start all-false — mathematically equal to torch's
//!     format_boundaries([full_duration]) whose last cumsum index is excluded);
//!   - GAME inference/slicer2.py (silence slicer, params = infer.py extract defaults)
//!     and inference/callbacks.py (cumsum-clamp assembly + global monotonic pass).
//! Parity gate: `cargo test --test game_parity -- --ignored` vs the python-ORT oracle
//! (TESTING/utai-v2-testing/game_oracle/, oracle_notes.json includes chunk boundaries).
//!
//! Device: sessions follow the GLOBAL device preference (S60c) and fall back to CPU once,
//! loudly, on a run-time failure — see extract_notes. (The original "CPU only, the GPU is
//! never touched" note here was stale from S60-1 and misled the S74 bug hunt; CPU is only
//! the batch harness's default, and ~7.5× realtime is the CPU figure.)
//! GAME's encoder convs are the app's only user of cuDNN's FRONTEND graph API, which lazily
//! loads engine libraries the ort crate does not preload — see lib.rs CUDNN_FRONTEND_EXTRA_DLLS.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::inference::engine::{InputTensor, OnnxEngine, OutputTensor};

pub const GAME_FILES: [&str; 5] = [
    "config.json",
    "encoder.onnx",
    "segmenter.onnx",
    "bd2dur.onnx",
    "estimator.onnx",
];

/// Official defaults (infer.py extract == OpenUtau dialog defaults). Deliberately not
/// user-tunable — openvpi's recommended values are the shipped defaults everywhere.
const SEG_THRESHOLD: f32 = 0.2;
const SEG_RADIUS: i64 = 2;
const EST_THRESHOLD: f32 = 0.2;
const NSTEPS: usize = 8;

#[derive(serde::Deserialize, Clone)]
pub struct GameConfig {
    pub samplerate: u32,
    #[allow(dead_code)]
    pub timestep: f32,
    #[serde(default)]
    pub languages: Option<BTreeMap<String, i64>>,
    #[serde(rename = "loop", default = "default_true")]
    pub d3pm_loop: bool,
}

fn default_true() -> bool {
    true
}

pub fn game_dir(models_dir: &Path) -> PathBuf {
    models_dir.join(crate::models::AUX_DIR_NAME).join("game")
}

pub fn game_installed(models_dir: &Path) -> bool {
    let dir = game_dir(models_dir);
    GAME_FILES.iter().all(|f| dir.join(f).is_file())
}

pub fn load_game_config(models_dir: &Path) -> Result<GameConfig, String> {
    let path = game_dir(models_dir).join("config.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("MIDI_EXTRACT_NOT_INSTALLED: {e}"))?;
    serde_json::from_str(&text).map_err(|e| format!("MIDI_EXTRACT_NOT_INSTALLED: bad config.json: {e}"))
}

// ─── slicer2.py port (silence-based slicing, sample-exact) ───────────────────

pub struct SliceChunk {
    /// Chunk start in samples of the input waveform.
    pub offset_samples: usize,
    pub samples: Vec<f32>,
}

/// librosa-style frame RMS (get_rms in slicer2.py): constant zero padding of
/// frame_length/2 on both sides, frames every `hop`, sqrt(mean(x²)).
fn rms_frames(samples: &[f32], frame_length: usize, hop: usize) -> Vec<f32> {
    let pad = frame_length / 2;
    let padded_len = samples.len() + 2 * pad;
    if padded_len < frame_length {
        return Vec::new();
    }
    let n = (padded_len - frame_length) / hop + 1;
    let len = samples.len() as isize;
    let mut out = Vec::with_capacity(n);
    for f in 0..n {
        let start = (f * hop) as isize - pad as isize;
        let mut acc = 0.0f64;
        for k in 0..frame_length as isize {
            let i = start + k;
            if i >= 0 && i < len {
                let v = samples[i as usize] as f64;
                acc += v * v;
            }
        }
        out.push((acc / frame_length as f64).sqrt() as f32);
    }
    out
}

/// numpy argmin over a slice: index of the FIRST strict minimum.
fn argmin(xs: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v < best_v {
            best = i;
            best_v = v;
        }
    }
    best
}

/// Faithful port of GAME inference/slicer2.py `Slicer.slice` with the infer.py extract
/// params: threshold −40 dB, min_length 1000 ms, min_interval 200 ms, hop 20 ms,
/// max_sil_kept 100 ms. Frame arithmetic mirrors python (round-ties-even; integer
/// divisions on the same quantities), argmin windows are inclusive like the numpy slices.
pub fn slice_silence(waveform: &[f32], sr: u32) -> Vec<SliceChunk> {
    let sr_f = sr as f64;
    let threshold = 10f32.powf(-40.0 / 20.0);
    let min_interval_samples = sr_f * 200.0 / 1000.0;
    let hop = (sr_f * 20.0 / 1000.0).round_ties_even() as usize;
    let win_size = (min_interval_samples.round_ties_even() as usize).min(4 * hop);
    let min_length = (sr_f * 1000.0 / 1000.0 / hop as f64).round_ties_even() as usize;
    let min_interval = (min_interval_samples / hop as f64).round_ties_even() as usize;
    let max_sil_kept = (sr_f * 100.0 / 1000.0 / hop as f64).round_ties_even() as usize;

    let whole = |wf: &[f32]| {
        vec![SliceChunk { offset_samples: 0, samples: wf.to_vec() }]
    };
    if (waveform.len() + hop - 1) / hop <= min_length {
        return whole(waveform);
    }
    let rms_list = rms_frames(waveform, win_size, hop);
    let total_frames = rms_list.len();

    // inclusive-window argmin helper: python's rms_list[a : b+1].argmin() + a (slices clip)
    let win_min = |a: usize, b_incl: usize| -> usize {
        let end = (b_incl + 1).min(total_frames);
        argmin(&rms_list[a..end]) + a
    };

    let mut sil_tags: Vec<(usize, usize)> = Vec::new();
    let mut silence_start: Option<usize> = None;
    let mut clip_start = 0usize;
    for (i, &rms) in rms_list.iter().enumerate() {
        if rms < threshold {
            if silence_start.is_none() {
                silence_start = Some(i);
            }
            continue;
        }
        let Some(sil_start) = silence_start else { continue };
        let is_leading_silence = sil_start == 0 && i > max_sil_kept;
        let need_slice_middle = i - sil_start >= min_interval && i - clip_start >= min_length;
        if !is_leading_silence && !need_slice_middle {
            silence_start = None;
            continue;
        }
        if i - sil_start <= max_sil_kept {
            let pos = win_min(sil_start, i);
            if sil_start == 0 {
                sil_tags.push((0, pos));
            } else {
                sil_tags.push((pos, pos));
            }
            clip_start = pos;
        } else if i - sil_start <= max_sil_kept * 2 {
            let pos = win_min(i - max_sil_kept, sil_start + max_sil_kept);
            let pos_l = win_min(sil_start, sil_start + max_sil_kept);
            let pos_r = win_min(i - max_sil_kept, i);
            if sil_start == 0 {
                sil_tags.push((0, pos_r));
                clip_start = pos_r;
            } else {
                sil_tags.push((pos_l.min(pos), pos_r.max(pos)));
                clip_start = pos_r.max(pos);
            }
        } else {
            let pos_l = win_min(sil_start, sil_start + max_sil_kept);
            let pos_r = win_min(i - max_sil_kept, i);
            if sil_start == 0 {
                sil_tags.push((0, pos_r));
            } else {
                sil_tags.push((pos_l, pos_r));
            }
            clip_start = pos_r;
        }
        silence_start = None;
    }
    if let Some(sil_start) = silence_start {
        if total_frames - sil_start >= min_interval {
            let silence_end = total_frames.min(sil_start + max_sil_kept);
            let pos = win_min(sil_start, silence_end);
            sil_tags.push((pos, total_frames + 1));
        }
    }
    if sil_tags.is_empty() {
        return whole(waveform);
    }
    let apply = |begin: usize, end: usize| -> Option<SliceChunk> {
        let a = (begin * hop).min(waveform.len());
        let b = (end * hop).min(waveform.len());
        if b <= a {
            return None; // unreachable for real tags; guards a zero-length slice
        }
        Some(SliceChunk { offset_samples: a, samples: waveform[a..b].to_vec() })
    };
    let mut chunks = Vec::new();
    if sil_tags[0].0 > 0 {
        chunks.extend(apply(0, sil_tags[0].0));
    }
    for i in 0..sil_tags.len() - 1 {
        chunks.extend(apply(sil_tags[i].1, sil_tags[i + 1].0));
    }
    if sil_tags[sil_tags.len() - 1].1 < total_frames {
        chunks.extend(apply(sil_tags[sil_tags.len() - 1].1, total_frames));
    }
    chunks
}

// ─── the 5-graph pipeline ────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize)]
pub struct GameNote {
    /// Global onset/offset in seconds of the INPUT waveform (chunk offsets folded in).
    pub onset_sec: f64,
    pub offset_sec: f64,
    /// Float MIDI pitch (A4 = 69, carries cents).
    pub pitch: f32,
}

fn expect_f32(out: OutputTensor, what: &str) -> Result<(Vec<i64>, Vec<f32>), String> {
    match out {
        OutputTensor::F32 { shape, data } => Ok((shape, data)),
        _ => Err(format!("MIDI_EXTRACT_FAILED: {what}: unexpected dtype")),
    }
}

fn expect_bool(out: OutputTensor, what: &str) -> Result<(Vec<i64>, Vec<bool>), String> {
    match out {
        OutputTensor::Bool { shape, data } => Ok((shape, data)),
        _ => Err(format!("MIDI_EXTRACT_FAILED: {what}: unexpected dtype")),
    }
}

struct GameSessions {
    encoder: String,
    segmenter: String,
    bd2dur: String,
    estimator: String,
    has_language: bool,
    d3pm_loop: bool,
    samplerate: u32,
}

fn load_sessions(engine: &OnnxEngine, models_dir: &Path, force_cpu: bool) -> Result<GameSessions, String> {
    if !game_installed(models_dir) {
        return Err("MIDI_EXTRACT_NOT_INSTALLED".to_string());
    }
    let config = load_game_config(models_dir)?;
    let dir = game_dir(models_dir);
    // S60c (§user): follow the GLOBAL device preference (Auto probes CUDA→DirectML→CPU with
    // CPU fallback) — forced CPU was noticeably slow on long stems. Dynamic shapes → mem
    // pattern off. Sessions are unloaded when the last extraction finishes (refcount), so
    // the GPU-class session cache pressure is transient. `force_cpu` = the run-time fallback
    // path (a GPU stack can pass session BUILD yet fail at the first conv — e.g. a cudnn
    // frontend engine DLL that won't resolve); extract_notes retries once on CPU.
    let load = |name: &str| {
        let path = dir.join(name);
        if force_cpu {
            engine.load_model_on(&path, false, super::engine::DeviceConfig::Cpu)
        } else {
            engine.load_model_with(&path, false)
        }
        .map_err(|e| format!("MIDI_EXTRACT_LOAD_FAILED: {name}: {e}"))
    };
    Ok(GameSessions {
        encoder: load("encoder.onnx")?,
        segmenter: load("segmenter.onnx")?,
        bd2dur: load("bd2dur.onnx")?,
        estimator: load("estimator.onnx")?,
        has_language: config.languages.is_some(),
        d3pm_loop: config.d3pm_loop,
        samplerate: config.samplerate,
    })
}

/// One chunk through the pipeline (B = 1, like OpenUtau). Returns per-note
/// (durations_sec, presence, scores) truncated to the maskN-valid prefix.
fn transcribe_chunk(
    engine: &OnnxEngine,
    s: &GameSessions,
    chunk: &[f32],
    language_id: i64,
    cancelled: &dyn Fn() -> bool,
    mut step_progress: impl FnMut(f32),
) -> Result<(Vec<f32>, Vec<bool>, Vec<f32>), String> {
    let l = chunk.len();
    let duration = l as f32 / s.samplerate as f32;
    let enc_out = engine
        .run_typed(
            &s.encoder,
            vec![
                ("waveform", InputTensor::F32 { data: chunk.to_vec(), shape: vec![1, l as i64] }),
                ("duration", InputTensor::F32 { data: vec![duration], shape: vec![1] }),
            ],
        )
        .map_err(|e| format!("MIDI_EXTRACT_FAILED: encoder: {e}"))?;
    let mut enc_it = enc_out.into_iter();
    let (xseg_shape, x_seg) = expect_f32(enc_it.next().ok_or("MIDI_EXTRACT_FAILED: encoder outputs")?, "x_seg")?;
    let (_, x_est) = expect_f32(enc_it.next().ok_or("MIDI_EXTRACT_FAILED: encoder outputs")?, "x_est")?;
    let (_, mask_t) = expect_bool(enc_it.next().ok_or("MIDI_EXTRACT_FAILED: encoder outputs")?, "maskT")?;
    let t_frames = xseg_shape.get(1).copied().unwrap_or(0) as usize;

    let known = vec![false; t_frames];
    let mut boundaries = vec![false; t_frames];
    let steps = if s.d3pm_loop { NSTEPS } else { 1 };
    for i in 0..steps {
        if cancelled() {
            return Err("MIDI_EXTRACT_CANCELLED".to_string());
        }
        let mut inputs: Vec<(&str, InputTensor)> = vec![
            ("x_seg", InputTensor::F32 { data: x_seg.clone(), shape: xseg_shape.clone() }),
            ("known_boundaries", InputTensor::Bool { data: known.clone(), shape: vec![1, t_frames as i64] }),
            ("maskT", InputTensor::Bool { data: mask_t.clone(), shape: vec![1, t_frames as i64] }),
            ("threshold", InputTensor::F32 { data: vec![SEG_THRESHOLD], shape: vec![] }),
            ("radius", InputTensor::I64 { data: vec![SEG_RADIUS], shape: vec![] }),
        ];
        if s.d3pm_loop {
            inputs.push(("prev_boundaries", InputTensor::Bool {
                data: boundaries.clone(),
                shape: vec![1, t_frames as i64],
            }));
            inputs.push(("t", InputTensor::F32 { data: vec![i as f32 / steps as f32], shape: vec![1] }));
        }
        if s.has_language {
            inputs.push(("language", InputTensor::I64 { data: vec![language_id], shape: vec![1] }));
        }
        let seg_out = engine
            .run_typed(&s.segmenter, inputs)
            .map_err(|e| format!("MIDI_EXTRACT_FAILED: segmenter: {e}"))?;
        let (_, b) = expect_bool(
            seg_out.into_iter().next().ok_or("MIDI_EXTRACT_FAILED: segmenter outputs")?,
            "boundaries",
        )?;
        boundaries = b;
        step_progress((i + 1) as f32 / steps as f32);
    }

    let b2d_out = engine
        .run_typed(
            &s.bd2dur,
            vec![
                ("boundaries", InputTensor::Bool { data: boundaries.clone(), shape: vec![1, t_frames as i64] }),
                ("maskT", InputTensor::Bool { data: mask_t.clone(), shape: vec![1, t_frames as i64] }),
            ],
        )
        .map_err(|e| format!("MIDI_EXTRACT_FAILED: bd2dur: {e}"))?;
    let mut b2d_it = b2d_out.into_iter();
    let (_, durations) = expect_f32(b2d_it.next().ok_or("MIDI_EXTRACT_FAILED: bd2dur outputs")?, "durations")?;
    let (maskn_shape, mask_n) = expect_bool(b2d_it.next().ok_or("MIDI_EXTRACT_FAILED: bd2dur outputs")?, "maskN")?;
    let n_notes = maskn_shape.get(1).copied().unwrap_or(0) as usize;

    let est_out = engine
        .run_typed(
            &s.estimator,
            vec![
                ("x_est", InputTensor::F32 { data: x_est, shape: xseg_shape.clone() }),
                ("boundaries", InputTensor::Bool { data: boundaries, shape: vec![1, t_frames as i64] }),
                ("maskT", InputTensor::Bool { data: mask_t, shape: vec![1, t_frames as i64] }),
                ("maskN", InputTensor::Bool { data: mask_n.clone(), shape: vec![1, n_notes as i64] }),
                ("threshold", InputTensor::F32 { data: vec![EST_THRESHOLD], shape: vec![] }),
            ],
        )
        .map_err(|e| format!("MIDI_EXTRACT_FAILED: estimator: {e}"))?;
    let mut est_it = est_out.into_iter();
    let (_, presence) = expect_bool(est_it.next().ok_or("MIDI_EXTRACT_FAILED: estimator outputs")?, "presence")?;
    let (_, scores) = expect_f32(est_it.next().ok_or("MIDI_EXTRACT_FAILED: estimator outputs")?, "scores")?;

    // paddings only appear at the end (ONNX.md) — truncate to the valid prefix
    let n_valid = mask_n.iter().take_while(|&&m| m).count();
    Ok((
        durations[..n_valid.min(durations.len())].to_vec(),
        presence[..n_valid.min(presence.len())].to_vec(),
        scores[..n_valid.min(scores.len())].to_vec(),
    ))
}

/// Official callbacks.py assembly: per-chunk f32 cumsum onsets clamped to the chunk
/// length, +offset (all f32, matching torch), voiced filter; then the global
/// sort + monotonic clamp pass. Offsets/lengths in seconds.
fn assemble_notes(
    chunks: Vec<(Vec<f32>, Vec<bool>, Vec<f32>)>,
    offsets_sec: &[f64],
    lengths_sec: &[f64],
) -> Vec<GameNote> {
    let mut notes: Vec<GameNote> = Vec::new();
    for (ci, (durs, pres, scores)) in chunks.into_iter().enumerate() {
        let off = offsets_sec[ci] as f32;
        let len = lengths_sec[ci] as f32;
        let mut cum = 0f32;
        let mut prev = 0f32;
        for i in 0..durs.len() {
            cum += durs[i];
            let onset = prev.min(len) + off;
            let offset = cum.min(len) + off;
            prev = cum;
            if offset - onset <= 0.0 || !pres[i] {
                continue;
            }
            notes.push(GameNote {
                onset_sec: onset as f64,
                offset_sec: offset as f64,
                pitch: scores[i],
            });
        }
    }
    notes.sort_by(|a, b| {
        (a.onset_sec, a.offset_sec, a.pitch)
            .partial_cmp(&(b.onset_sec, b.offset_sec, b.pitch))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut out: Vec<GameNote> = Vec::with_capacity(notes.len());
    let mut last = 0f64;
    for mut n in notes {
        n.onset_sec = n.onset_sec.max(last);
        n.offset_sec = n.offset_sec.max(n.onset_sec);
        if n.offset_sec <= n.onset_sec {
            continue;
        }
        last = n.offset_sec;
        out.push(n);
    }
    out
}

/// Extract notes from a mono waveform at the model's samplerate (44100). `language_id`
/// follows config.json's map (0 = universal — our only caller; the field exists for parity
/// with the official CLI). Progress is 0..1 over input samples.
///
/// Device policy (S60c): try the GLOBAL device first; on a RUN-TIME inference failure
/// (session build can succeed while the first conv still fails — brittle GPU stacks) the
/// whole extraction retries ONCE on forced CPU. Cancellation is never retried.
pub fn extract_notes(
    engine: &OnnxEngine,
    models_dir: &Path,
    samples: &[f32],
    language_id: i64,
    cancelled: &dyn Fn() -> bool,
    progress: &mut dyn FnMut(f32),
) -> Result<Vec<GameNote>, String> {
    match extract_notes_on(engine, models_dir, samples, language_id, cancelled, progress, false) {
        Err(e)
            if !e.contains("MIDI_EXTRACT_CANCELLED")
                && !e.contains("MIDI_EXTRACT_NOT_INSTALLED") =>
        {
            tracing::warn!("GAME on the global device failed ({e}) — retrying once on CPU");
            unload_sessions(engine, models_dir);
            extract_notes_on(engine, models_dir, samples, language_id, cancelled, progress, true)
        }
        r => r,
    }
}

fn extract_notes_on(
    engine: &OnnxEngine,
    models_dir: &Path,
    samples: &[f32],
    language_id: i64,
    cancelled: &dyn Fn() -> bool,
    progress: &mut dyn FnMut(f32),
    force_cpu: bool,
) -> Result<Vec<GameNote>, String> {
    let s = load_sessions(engine, models_dir, force_cpu)?;
    let sr = s.samplerate;
    let chunks = slice_silence(samples, sr);
    tracing::info!("GAME: {} chunk(s) from {:.1}s audio", chunks.len(), samples.len() as f64 / sr as f64);
    let total: usize = chunks.iter().map(|c| c.samples.len()).sum();
    let mut done = 0usize;
    let mut results = Vec::with_capacity(chunks.len());
    let mut offsets = Vec::with_capacity(chunks.len());
    let mut lengths = Vec::with_capacity(chunks.len());
    for ch in &chunks {
        if cancelled() {
            return Err("MIDI_EXTRACT_CANCELLED".to_string());
        }
        let chunk_len = ch.samples.len();
        let res = transcribe_chunk(engine, &s, &ch.samples, language_id, cancelled, |frac| {
            progress(((done as f64 + frac as f64 * chunk_len as f64) / total.max(1) as f64) as f32);
        })?;
        results.push(res);
        offsets.push(ch.offset_samples as f64 / sr as f64);
        lengths.push(chunk_len as f64 / sr as f64);
        done += chunk_len;
        progress((done as f64 / total.max(1) as f64) as f32);
    }
    // final check narrows the cancel-vs-complete race (the frontend closes it fully with
    // its own cancelled-keys mark — a cancel can still land after this command returns)
    if cancelled() {
        return Err("MIDI_EXTRACT_CANCELLED".to_string());
    }
    Ok(assemble_notes(results, &offsets, &lengths))
}

/// Free the GAME sessions (they live on CPU RAM; call after a batch of extractions).
pub fn unload_sessions(engine: &OnnxEngine, models_dir: &Path) {
    engine.unload_paths_with_prefix(&game_dir(models_dir));
}
