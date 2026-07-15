//! S41 audition rendering (试听多选保留) — render a training-run candidate
//! checkpoint over the bundled 试听 clip (training/assets/audition_10s.wav)
//! so the user can pick which snapshots to keep.
//!
//! Three commands, one per candidate class:
//!   render_audition_voice     rvc / sovits weights snapshots (plain S35 path:
//!                             no index/cluster/diffusion/enhancer/auto-f0)
//!   render_audition_vocoder   vocoder ckpts via the mel->vocode self-loop
//!                             (+ the aux default vocoder as the A/B reference)
//!   render_audition_diffusion diff model_<step>.pt through an INSTALLED host
//!                             SoVITS model with the candidate's freshly
//!                             converted assets overriding the attachment
//!
//! Design contract (s41_two_features_design.md A2/A3, red-team rulings):
//!   - candidates are NOT registry entries: facts come from the converted
//!     sidecar json in the audition dir; sessions load by path
//!   - conversions + renders run under spawn_blocking (V15/F20) and cache in
//!     <workspace>/audition/<ckpt-stem>/ — audition.wav present = instant
//!     return (atomic writes, A19); the dir is wiped by start_training /
//!     清空结果 (commands/training.rs owns the ordering)
//!   - one flight at a time (AUDITION_IN_FLIGHT — start_training and
//!     reset_training_display refuse while it is set, R4/A2)
//!   - session hygiene per S36: begin_voice_run FIRST, evict-then-load with an
//!     explicit keep set, resolve AFTER load_voice, unload the audition dir's
//!     sessions when done (Windows file locks + VRAM)
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use tauri::{Emitter, State};

use crate::commands::inference::{
    aux_path, contentvec_for_dim, min_frames, noise_channels, require_input,
    resolve_sovits_quality, AUX_NSF_HIFIGAN, AUX_NSF_HIFIGAN_JSON, AUX_NSF_HIFIGAN_MEL, AUX_RMVPE,
    AUX_RMVPE_MEL,
};
use crate::inference::{rvc, sovits, RvcOptions, SovitsOptions, VoiceBackendType};
use crate::models::{ModelConfig, ModelType};
use crate::AppState;

/// Process-wide "an audition render is in flight" flag. start_training and
/// reset_training_display consult it: both delete/wipe the audition dir (or
/// the whole workspace) and must not race a conversion subprocess writing
/// into it (red-team R4/A2).
pub static AUDITION_IN_FLIGHT: AtomicBool = AtomicBool::new(false);

/// RAII holder of AUDITION_IN_FLIGHT. Also acquired (with their own message)
/// by start_training / reset_training_display — holding the SAME flag for the
/// whole critical section closes the check-then-act window between the two
/// sides (审查修复 S41-INT-4; a plain load() check was a Dekker anti-pattern).
pub struct FlightGuard;
impl FlightGuard {
    pub fn acquire(busy_msg: &str) -> Result<Self, String> {
        if AUDITION_IN_FLIGHT.swap(true, Ordering::SeqCst) {
            return Err(busy_msg.to_string());
        }
        Ok(FlightGuard)
    }
}
impl Drop for FlightGuard {
    fn drop(&mut self) {
        AUDITION_IN_FLIGHT.store(false, Ordering::SeqCst);
    }
}

/// Frontend state reconciliation on remount (审查修复 FE-1/AUD-DONE-DROPPED):
/// transient converting/rendering phases live in component state and die with
/// the page — the remounted page asks whether anything is actually in flight.
#[tauri::command]
pub fn audition_active() -> bool {
    AUDITION_IN_FLIGHT.load(Ordering::SeqCst)
}

#[derive(Clone, serde::Serialize)]
struct AuditionProgress {
    candidate_id: String,
    phase: String, // "converting" | "rendering" | "done" | "error"
    progress: Option<f32>,
    /// done only: the rendered wav path (the frontend refills its cache map —
    /// it cannot guess the diff render's host-suffixed name; 审查修复 FE-1)
    wav: Option<String>,
}

fn emit_phase(app: &tauri::AppHandle, candidate_id: &str, phase: &str) {
    let _ = app.emit(
        "audition-progress",
        AuditionProgress {
            candidate_id: candidate_id.to_string(),
            phase: phase.to_string(),
            progress: None,
            wav: None,
        },
    );
}

/// Terminal-event wrapper for the render bodies (审查修复 FE-1): every exit —
/// Ok or Err — emits a terminal phase so the frontend can never be stranded
/// in a busy state by a dropped resolution (page remounts, stale closures).
/// `wav_of` extracts the done-payload wav; a render whose product is NOT an
/// audition clip (the S60c range-test scale) passes `|_| None` so the frontend
/// never caches a probe wav as a playable audition.
fn with_terminal_events_as<T>(
    app: &tauri::AppHandle,
    candidate_id: &str,
    wav_of: impl Fn(&T) -> Option<String>,
    body: impl FnOnce() -> Result<T, String>,
) -> Result<T, String> {
    let r = body();
    let _ = app.emit(
        "audition-progress",
        AuditionProgress {
            candidate_id: candidate_id.to_string(),
            phase: if r.is_ok() { "done" } else { "error" }.to_string(),
            progress: None,
            wav: r.as_ref().ok().and_then(&wav_of),
        },
    );
    r
}

fn with_terminal_events(
    app: &tauri::AppHandle,
    candidate_id: &str,
    body: impl FnOnce() -> Result<String, String>,
) -> Result<String, String> {
    with_terminal_events_as(app, candidate_id, |wav: &String| Some(wav.clone()), body)
}

/// Failed/interrupted conversions must not leave a complete-looking cache
/// (审查修复 S41-RUST-1/2, AUD-CACHE-MARKER): the sidecar json is the LAST
/// artifact every exporter writes (completion marker), so on a convert error
/// the whole candidate dir is swept — a half-written onnx with no json would
/// otherwise stall retries forever, and a selfcheck-rejected graph could get
/// batch-imported as a permanent resource.
fn sweep_candidate_dir(dir: &Path) {
    if let Err(e) = std::fs::remove_dir_all(dir) {
        tracing::warn!("audition candidate sweep failed (non-fatal): {}", e);
    }
}

/// Convert a candidate ckpt into its audition dir if not already done (the sidecar json is
/// the exporter's LAST artifact = the completion marker; 审查修复 S41-RUST-2: a bare onnx
/// means an interrupted conversion — sweep and redo, never trust it). Shared by the audition
/// render and the S60c range-scale render (single conversion source).
fn ensure_candidate_converted(
    app: &AppState,
    apph: &tauri::AppHandle,
    candidate_id: &str,
    dir: &Path,
    onnx: &Path,
    backend: &str,
    ckpt_path: &str,
) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("AUDITION_DIR_CREATE_FAILED: {}", e))?;
    if dir.join("model.json").is_file() {
        return Ok(());
    }
    if onnx.exists() {
        sweep_candidate_dir(dir);
        std::fs::create_dir_all(dir).map_err(|e| format!("AUDITION_DIR_CREATE_FAILED: {}", e))?;
    }
    // S66 (review): this forks its own torch-export python — never in parallel with the
    // app-wide convert slot. We already hold AUDITION_IN_FLIGHT and the slot holder checks
    // that flag symmetrically, so the two flags close the check-then-act window both ways.
    // Cache hits above stay unblocked (no conversion happens).
    if app.task_active("convert") {
        return Err("CONVERT_BUSY".to_string());
    }
    emit_phase(apph, candidate_id, "converting");
    let mtype = match backend {
        "rvc" => ModelType::Rvc,
        "sovits" => ModelType::SoVits,
        other => return Err(format!("AUDITION_BACKEND_UNSUPPORTED: {}", other)),
    };
    // sovits release snapshots auto-detect weights/config.json next to the ckpt;
    // rvc savee models embed their config
    crate::models::convert::convert_pth_to_onnx(Path::new(ckpt_path), onnx, &mtype, &app.app_dir)
        .map_err(|e| {
            sweep_candidate_dir(dir);
            e.to_string()
        })
}

/// Throttled per-candidate progress for the pipeline section (≥1% steps).
fn progress_emitter(
    app: tauri::AppHandle,
    state: Arc<AppState>,
    run_epoch: u64,
    candidate_id: String,
) -> impl Fn(f32) {
    let last = AtomicU32::new(0);
    move |p: f32| {
        if state.inference.voice_cancelled(run_epoch) {
            return;
        }
        let pct = (p * 100.0).round() as u32;
        if p >= 1.0 || pct > last.load(Ordering::Relaxed) {
            last.store(pct, Ordering::Relaxed);
            let _ = app.emit(
                "audition-progress",
                AuditionProgress {
                    candidate_id: candidate_id.clone(),
                    phase: "rendering".into(),
                    progress: Some(p),
                    wav: None,
                },
            );
        }
    }
}

fn ensure_trainable_idle(state: &AppState) -> Result<(), String> {
    if state.training.is_active() {
        return Err("AUDITION_TRAINING_ACTIVE".into());
    }
    Ok(())
}

/// 审查修复 S41-INT-3: auditions and DAW voice renders both start by evicting
/// every foreign GPU session — running them concurrently is a VRAM tug-of-war
/// and the global cancel_voice would cross-kill. Mutual friendly rejection.
fn ensure_no_voice_render() -> Result<(), String> {
    if crate::commands::inference::voice_render_active() {
        return Err("AUDITION_RENDER_BUSY".into());
    }
    Ok(())
}

/// THE generic busy-retry CODE for every FlightGuard/interlock rejection (user S61: the flag's
/// holder can be an audition, a voice render, OR a storage cleanup — naming a specific reason was
/// factually wrong half the time, and threading the real holder through the guard isn't worth it).
/// Stable English CODE per the i18n hard rule — the frontend maps it via backendErrorMessage
/// (src/lib/backendError.ts) → t("common.busyRetry"); never put user-facing prose here.
pub const BUSY_RETRY_MSG: &str = "APP_BUSY";

const AUDITION_BUSY_MSG: &str = BUSY_RETRY_MSG;

fn audition_dir(workspace: &str, stem: &str) -> PathBuf {
    Path::new(workspace).join("audition").join(stem)
}

fn ckpt_stem(ckpt_path: &str) -> Result<String, String> {
    Path::new(ckpt_path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| "AUDITION_BAD_CANDIDATE_PATH".into())
}

// ─── S60c: whole-clip range extension for auditions ──────────────────────────
// The bundled clip skews HIGH — a low-range singer (e.g. F#2–A#4) sounds mute/broken in
// audition and reads as "training failed" (§user实测). When the model/candidate carries a
// vocal_range record, ONE tier decision runs over the ENTIRE clip (uniform shift = uniform
// formant character — the clip is deliberately a single chunk, §user), the shift folds into
// options.f0_shift for the render, and the WHOLE output is TD-PSOLA'd back.

/// Cache-name suffix carrying the range record (a re-test/adjusted comfort must MISS the old
/// cache). No record keeps the pre-S60c names byte-identical (existing caches stay valid).
fn audition_cache_tag(range: &Option<crate::inference::vocal_range::SpeakerRange>) -> String {
    match range {
        None => String::new(),
        Some(r) => format!(
            "_ru{:.0}-{:.0}c{:.0}-{:.0}",
            r.usable.0, r.usable.1, r.comfort.0, r.comfort.1
        ),
    }
}

/// Compute the whole-clip shift + the (shifted, zero-preserving) rmvpe guide for the inverse.
/// None = no record / already in range → render untouched.
fn audition_range_prep(
    app: &AppState,
    rmvpe_sid: &str,
    mel: &ndarray::Array2<f32>,
    src: &crate::audio::AudioBuffer,
    range: Option<crate::inference::vocal_range::SpeakerRange>,
) -> Result<Option<(i64, Vec<f32>)>, String> {
    let Some(r) = range else { return Ok(None) };
    let mono = crate::audio::resample::to_mono(src);
    let wav16k = crate::inference::features::resample(
        &mono.samples,
        mono.sample_rate,
        crate::inference::f0::RMVPE_SR,
    );
    let f0 = crate::inference::f0::rmvpe_detect(
        &app.inference.engine,
        rmvpe_sid,
        mel,
        &wav16k,
        crate::inference::f0::SOVITS_RMVPE_THRESHOLD,
    )
    .map_err(|e| e.to_string())?;
    // loudness weighting on the rmvpe 100 fps grid (16 kHz, hop 160) — see piece_range_shift
    let rms = crate::inference::vocal_range::frame_rms(
        &wav16k,
        crate::inference::f0::RMVPE_SR as usize / 100,
        f0.len(),
    );
    let shift = crate::inference::vocal_range::piece_range_shift(&f0, Some(&rms), &r);
    if shift == 0 {
        return Ok(None);
    }
    tracing::info!("audition range-extend: clip rendered {shift:+} st into comfort");
    let k = 2.0f32.powf(shift as f32 / 12.0);
    // raw rmvpe zeros preserved → unvoiced passes through the inverse dry
    let guide: Vec<f32> = f0.iter().map(|v| v * k).collect();
    Ok(Some((shift, guide)))
}

/// Apply the inverse of `audition_range_prep`'s shift to a finished render.
fn audition_range_invert(
    result: crate::inference::SynthesisResult,
    prep: &Option<(i64, Vec<f32>)>,
) -> crate::inference::SynthesisResult {
    match prep {
        Some((shift, guide)) if result.sample_rate % 100 == 0 => crate::inference::SynthesisResult {
            audio: crate::inference::vocal_range::psola_inverse_hop(
                result.audio,
                guide,
                (result.sample_rate / 100) as usize,
                result.sample_rate,
                *shift,
            ),
            sample_rate: result.sample_rate,
        },
        _ => result,
    }
}

fn audition_source(state: &AppState) -> Result<PathBuf, String> {
    let p = state
        .app_dir
        .join("training")
        .join("assets")
        .join("audition_10s.wav");
    if !p.is_file() {
        return Err(format!("AUDITION_CLIP_MISSING: {}", p.display()));
    }
    Ok(p)
}

/// 16-bit PCM like every other app-written wav; tmp+rename so a killed render
/// can never leave a truncated file the cache short-circuit would trust (A19).
/// Thin wrapper over the shared `audio::save_wav_atomic` (S66 single source),
/// keeping the audition-scoped error CODE.
fn write_wav_atomic(path: &Path, samples: &[f32], sample_rate: u32) -> Result<(), String> {
    let buf = crate::audio::AudioBuffer {
        samples: samples.to_vec(),
        sample_rate,
        channels: 1,
    };
    crate::audio::save_wav_atomic(path, &buf)
        .map_err(|e| format!("AUDITION_WAV_WRITE_FAILED: {}", e))
}

/// Candidate sidecar → ModelConfig, with sample_rate REQUIRED (A15: the serde
/// default would silently be 40000 — a corrupt sidecar must be loud, not
/// detuned).
fn read_candidate_config(json_path: &Path) -> Result<(ModelConfig, u32), String> {
    let text = std::fs::read_to_string(json_path).map_err(|e| {
        format!(
            "AUDITION_CONFIG_READ_FAILED: {} ({})",
            json_path.display(),
            e
        )
    })?;
    let raw: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
        format!(
            "AUDITION_CONFIG_PARSE_FAILED: {} ({})",
            json_path.display(),
            e
        )
    })?;
    let sr = raw
        .get("sample_rate")
        .and_then(|v| v.as_u64())
        .filter(|&v| v > 0)
        .ok_or_else(|| {
            format!(
                "AUDITION_CONFIG_NO_SAMPLE_RATE: {}",
                json_path.display()
            )
        })? as u32;
    let config: ModelConfig = serde_json::from_value(raw)
        .map_err(|e| format!("AUDITION_CONFIG_PARSE_FAILED: {} ({})", json_path.display(), e))?;
    Ok((config, sr))
}

// ─── command 1: rvc / sovits candidates ─────────────────────────────────────

#[tauri::command]
pub async fn render_audition_voice(
    app_handle: tauri::AppHandle,
    state: State<'_, Arc<AppState>>,
    backend: String,
    ckpt_path: String,
    workspace: String,
    candidate_id: String,
    // ①c: which speaker to audition for a multi-speaker candidate. None = single-speaker (or
    // pre-①c) → speaker 0 via the sid/one-hot fallback. Some(id) one-hots that emb_g row.
    speaker_id: Option<u32>,
) -> Result<String, String> {
    let app = state.inner().clone();
    // guard FIRST, state checks after — holding the flag through the checks
    // closes the audition↔start_training check-then-act window (S41-INT-4)
    let guard = FlightGuard::acquire(AUDITION_BUSY_MSG)?;
    ensure_trainable_idle(&app)?;
    ensure_no_voice_render()?;
    let stem = ckpt_stem(&ckpt_path)?;
    let dir = audition_dir(&workspace, &stem);
    // ①c: cache the rendered clip PER speaker (the onnx is shared across speakers — only the fed
    // speaker differs). A single-speaker candidate (speaker_id None) keeps "audition.wav" =
    // byte-identical caching to pre-①c. S60c: the name also carries the candidate's vocal_range
    // record (whole-clip pre-shift is a render input) — computable only once the sidecar exists,
    // so the early probe is conditional; the authoritative name is re-derived in the closure.
    let cache_name = move |range: &Option<crate::inference::vocal_range::SpeakerRange>| match speaker_id {
        Some(s) => format!("audition_spk{}{}.wav", s, audition_cache_tag(range)),
        None => format!("audition{}.wav", audition_cache_tag(range)),
    };
    if dir.join("model.json").is_file() {
        if let Ok((cfg, _)) = read_candidate_config(&dir.join("model.json")) {
            let range = crate::inference::vocal_range::speaker_range(&cfg, speaker_id.unwrap_or(0));
            let out = dir.join(cache_name(&range));
            if out.is_file() {
                drop(guard);
                return Ok(out.to_string_lossy().into_owned());
            }
        }
    }
    let src = audition_source(&app)?;

    let apph = app_handle.clone();
    tauri::async_runtime::spawn_blocking(move || -> Result<String, String> {
        let _guard = guard; // released when this render finishes (any path)
        let cid = candidate_id.clone();
        with_terminal_events(&apph.clone(), &cid, || {
        let onnx = dir.join("model.onnx");
        ensure_candidate_converted(&app, &apph, &candidate_id, &dir, &onnx, &backend, &ckpt_path)?;
        let (config, sample_rate) = read_candidate_config(&dir.join("model.json"))?;
        // S60c: the candidate's tested range (written by the post-training auto-test into this
        // sidecar) drives the whole-clip pre-shift; the cache name carries it.
        let range = crate::inference::vocal_range::speaker_range(&config, speaker_id.unwrap_or(0));
        let out = dir.join(cache_name(&range));
        if out.is_file() {
            return Ok(out.to_string_lossy().into_owned());
        }

        emit_phase(&apph, &candidate_id, "rendering");
        // arm the cancel epoch before the load phase (S36 discipline)
        let run_epoch = app.inference.begin_voice_run();
        app.inference
            .engine
            .release_gpu_sessions_except(&[onnx.clone()]);
        let sid = app
            .inference
            .engine
            .load_model_with(&onnx, false)
            .map_err(|e| e.to_string())?;

        let audio_buf = crate::audio::load_audio(&src).map_err(|e| e.to_string())?;
        let rmvpe_sid = app
            .inference
            .ensure_aux_loaded_on(&aux_path(&app, AUX_RMVPE, "RMVPE model")?, false)
            .map_err(|e| e.to_string())?;
        let mel = app
            .inference
            .load_npy(&aux_path(&app, AUX_RMVPE_MEL, "RMVPE mel filterbank")?)
            .map_err(|e| e.to_string())?;
        // S60c: whole-clip range pre-shift (single chunk = uniform formant character)
        let range_prep = audition_range_prep(&app, &rmvpe_sid, mel.as_ref(), &audio_buf, range)?;
        let range_f0_shift = range_prep.as_ref().map(|(s, _)| *s as f32).unwrap_or(0.0);
        let progress = progress_emitter(apph.clone(), app.clone(), run_epoch, candidate_id.clone());
        let cancel = || app.inference.voice_cancelled(run_epoch);

        // fixed audition recipe: transpose 0(+S60c range shift), rmvpe, no retrieval/cluster,
        // no quality path, CPU extractors — the candidate model itself, bare
        let result = match backend.as_str() {
            "rvc" => {
                let dim = config.features_dim as usize;
                if dim == 0 {
                    return Err("AUDITION_CONFIG_NO_FEATURES_DIM".into());
                }
                let cv_sid = app
                    .inference
                    .ensure_aux_loaded_on(&contentvec_for_dim(&app, dim)?, false)
                    .map_err(|e| e.to_string())?;
                // ①c: a genuine multi-speaker RVC candidate exports "spk_mix" (no "sid") — the
                // audition recipe has no blend UI so this falls to a one-hot on speaker 0.
                let rvc_spk_mix = if config
                    .inputs
                    .as_ref()
                    .and_then(|v| v.as_array())
                    .map(|l| l.iter().any(|v| v.as_str() == Some("spk_mix")))
                    == Some(true)
                    && config.n_speakers > 0
                {
                    Some(config.n_speakers as usize)
                } else {
                    None
                };
                let model = rvc::RvcModel {
                    engine: &app.inference.engine,
                    voice_session: &sid,
                    contentvec_session: &cv_sid,
                    rmvpe_session: &rmvpe_sid,
                    mel_filters: mel.as_ref(),
                    index: None,
                    sample_rate,
                    features_dim: dim,
                    spk_mix: rvc_spk_mix,
                    noise_channels: noise_channels(&config),
                    min_frames: min_frames(&config, 12),
                };
                let options = RvcOptions {
                    index_ratio: 0.0,
                    gpu_extract: false,
                    speaker_id,
                    f0_shift: range_f0_shift,
                    ..Default::default()
                };
                rvc::run_pipeline(&model, &audio_buf, &options, None, &progress, &cancel)
                    .map_err(|e| e.to_string())?
            }
            _ => {
                let dim = config.resolved_features_dim()?;
                let hop_size = config.hop_size.unwrap_or(512) as usize;
                if hop_size == 0 {
                    return Err("AUDITION_CONFIG_HOP_ZERO".into());
                }
                let cv_sid = app
                    .inference
                    .ensure_aux_loaded_on(&contentvec_for_dim(&app, dim)?, false)
                    .map_err(|e| e.to_string())?;
                let has_input = |name: &str| {
                    config
                        .inputs
                        .as_ref()
                        .and_then(|v| v.as_array())
                        .map(|l| l.iter().any(|v| v.as_str() == Some(name)))
                };
                let vol_embedding = has_input("vol")
                    .unwrap_or_else(|| config.vol_embedding.unwrap_or(false));
                // ①c: a genuine multi-speaker candidate exports "spk_mix" (no "sid") — the audition
                // recipe has no blend UI so this falls to a one-hot on speaker 0 (default options).
                let spk_mix = if has_input("spk_mix") == Some(true) && config.n_speakers > 0 {
                    Some(config.n_speakers as usize)
                } else {
                    None
                };
                let model = sovits::SovitsModel {
                    engine: &app.inference.engine,
                    voice_session: &sid,
                    contentvec_session: &cv_sid,
                    rmvpe_session: &rmvpe_sid,
                    mel_filters: mel.as_ref(),
                    cluster: None,
                    diffusion: None,
                    vocoder: None,
                    f0_predictor_session: None,
                    sample_rate,
                    hop_size,
                    features_dim: dim,
                    vol_embedding,
                    spk_mix,
                    unit_interpolate_mode: config
                        .unit_interpolate_mode
                        .clone()
                        .unwrap_or_else(|| "left".to_string()),
                    noise_channels: noise_channels(&config),
                    min_frames: min_frames(&config, 6),
                };
                let options = SovitsOptions {
                    cluster_ratio: 0.0,
                    gpu_extract: false,
                    speaker_id,
                    f0_shift: range_f0_shift,
                    ..Default::default()
                };
                sovits::run_pipeline(&model, &audio_buf, &options, None, &progress, &cancel)
                    .map_err(|e| e.to_string())?
            }
        };
        let result = audition_range_invert(result, &range_prep);
        write_wav_atomic(&out, &result.audio, result.sample_rate)?;
        // free the candidate's VRAM + Windows file locks (the dir must stay
        // deletable by 清空结果 / the next training run)
        app.inference.engine.unload_paths_with_prefix(&dir);
        Ok(out.to_string_lossy().into_owned())
        })
    })
    .await
    .map_err(|e| format!("AUDITION_TASK_PANICKED: {}", e))?
}

// ─── command 2: vocoder candidates (mel→vocode self-loop) ───────────────────

#[tauri::command]
pub async fn render_audition_vocoder(
    app_handle: tauri::AppHandle,
    state: State<'_, Arc<AppState>>,
    ckpt_path: Option<String>, // None = the aux default vocoder (A/B reference row)
    workspace: String,
    candidate_id: String,
) -> Result<String, String> {
    let app = state.inner().clone();
    let guard = FlightGuard::acquire(AUDITION_BUSY_MSG)?;
    ensure_trainable_idle(&app)?;
    ensure_no_voice_render()?;
    let stem = match &ckpt_path {
        Some(p) => ckpt_stem(p)?,
        None => "_default".to_string(),
    };
    let dir = audition_dir(&workspace, &stem);
    let out = dir.join("audition.wav");
    if out.is_file() {
        drop(guard);
        return Ok(out.to_string_lossy().into_owned());
    }
    let src = audition_source(&app)?;

    let apph = app_handle.clone();
    tauri::async_runtime::spawn_blocking(move || -> Result<String, String> {
        let _guard = guard;
        let cid = candidate_id.clone();
        with_terminal_events(&apph.clone(), &cid, || {
        std::fs::create_dir_all(&dir).map_err(|e| format!("AUDITION_DIR_CREATE_FAILED: {}", e))?;

        // resolve the vocoder triple: candidate (converted into the audition
        // dir) or the aux default
        let (voc_onnx, voc_json, voc_base) = match &ckpt_path {
            Some(ckpt) => {
                let onnx = dir.join("vocoder.onnx");
                // vocoder.json = completion marker: the exporter selfchecks
                // BEFORE writing it (审查修复 S41-RUST-1 — a selfcheck-rejected
                // graph must never look like a complete cache)
                if !dir.join("vocoder.json").is_file() {
                    if onnx.exists() {
                        sweep_candidate_dir(&dir);
                        std::fs::create_dir_all(&dir)
                            .map_err(|e| format!("AUDITION_DIR_CREATE_FAILED: {}", e))?;
                    }
                    // S66 (review): same convert-slot exclusion as ensure_candidate_converted.
                    if app.task_active("convert") {
                        return Err("CONVERT_BUSY".to_string());
                    }
                    emit_phase(&apph, &candidate_id, "converting");
                    // config.json sits next to the ckpt in weights/ — the
                    // exporter auto-detects it
                    crate::models::convert::convert_vocoder_to_onnx(
                        Path::new(ckpt),
                        None,
                        &dir,
                        "vocoder",
                        &app.app_dir,
                    )
                    .map_err(|e| {
                        sweep_candidate_dir(&dir);
                        e.to_string()
                    })?;
                }
                (onnx, dir.join("vocoder.json"), dir.clone())
            }
            None => (
                aux_path(&app, AUX_NSF_HIFIGAN, "NSF-HiFiGAN vocoder")?,
                aux_path(&app, AUX_NSF_HIFIGAN_JSON, "NSF-HiFiGAN vocoder config")?,
                app.models.aux_dir(),
            ),
        };
        let sidecar: serde_json::Value = crate::commands::inference::read_json(&voc_json, "vocoder config")?;
        let sr = sidecar.get("sample_rate").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        let hop = sidecar.get("hop_size").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let num_mels = sidecar.get("num_mels").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        // mel::nsf_mel geometry is pinned to the OpenVPI standard — anything
        // else cannot be auditioned through this loop (and cannot be trained
        // by the vocoder backend either, so this is belt-and-suspenders)
        if sr != 44100 || hop != 512 || num_mels != 128 {
            return Err(format!(
                "AUDITION_VOCODER_NONSTANDARD: {}Hz/hop {}/{} mel (need 44100/512/128)",
                sr, hop, num_mels
            ));
        }
        let mel_name = sidecar
            .get("mel_filters")
            .and_then(|v| v.as_str())
            .unwrap_or(AUX_NSF_HIFIGAN_MEL);
        let filters = app
            .inference
            .load_npy(&voc_base.join(mel_name))
            .map_err(|e| e.to_string())?;
        if filters.nrows() != num_mels {
            return Err(format!("VOCODER_FILTER_SHAPE_MISMATCH: {} rows vs num_mels={}", filters.nrows(), num_mels));
        }

        emit_phase(&apph, &candidate_id, "rendering");
        let run_epoch = app.inference.begin_voice_run();
        app.inference
            .engine
            .release_gpu_sessions_except(&[voc_onnx.clone()]);
        let voc_sid = app
            .inference
            .engine
            .load_model_with(&voc_onnx, false)
            .map_err(|e| e.to_string())?;
        let rmvpe_sid = app
            .inference
            .ensure_aux_loaded_on(&aux_path(&app, AUX_RMVPE, "RMVPE model")?, false)
            .map_err(|e| e.to_string())?;
        let rmvpe_mel = app
            .inference
            .load_npy(&aux_path(&app, AUX_RMVPE_MEL, "RMVPE mel filterbank")?)
            .map_err(|e| e.to_string())?;

        // self-loop: source → (16k → rmvpe f0) + (44.1k → nsf mel) → vocode.
        // TWO resample branches — rmvpe's contract is 16 kHz mono (feeding it
        // 44.1k would mistrack both pitch and the frame clock; red-team A8)
        let audio_buf = crate::audio::load_audio(&src).map_err(|e| e.to_string())?;
        let mono = crate::audio::resample::to_mono(&audio_buf);
        let x44 = crate::inference::features::resample(&mono.samples, mono.sample_rate, 44100);
        let wav16k = crate::inference::features::resample(
            &mono.samples,
            mono.sample_rate,
            crate::inference::f0::RMVPE_SR,
        );
        if app.inference.voice_cancelled(run_epoch) {
            return Err("CANCELLED".into());
        }
        let f0_raw = crate::inference::f0::rmvpe_detect(
            &app.inference.engine,
            &rmvpe_sid,
            &rmvpe_mel,
            &wav16k,
            crate::inference::f0::SOVITS_RMVPE_THRESHOLD,
        )
        .map_err(|e| e.to_string())?;
        let mel = crate::inference::mel::nsf_mel(&x44, filters.as_ref());
        let (f0, _uv) =
            crate::inference::f0::sovits_f0_postprocess(&f0_raw, mel.ncols(), 512, 44100);
        if app.inference.voice_cancelled(run_epoch) {
            return Err("CANCELLED".into());
        }
        let samples = crate::inference::nsf_hifigan::vocode(
            &app.inference.engine,
            &voc_sid,
            &mel,
            &f0,
        )
        .map_err(|e| e.to_string())?;

        write_wav_atomic(&out, &samples, 44100)?;
        // only the audition dir's sessions — the aux default vocoder session
        // (None branch) lives outside and stays warm like any aux
        app.inference.engine.unload_paths_with_prefix(&dir);
        Ok(out.to_string_lossy().into_owned())
        })
    })
    .await
    .map_err(|e| format!("AUDITION_TASK_PANICKED: {}", e))?
}

// ─── command 3: diffusion candidates (host model + override) ────────────────

#[tauri::command]
pub async fn render_audition_diffusion(
    app_handle: tauri::AppHandle,
    state: State<'_, Arc<AppState>>,
    host_name: String,
    ckpt_path: String,
    workspace: String,
    candidate_id: String,
) -> Result<String, String> {
    let app = state.inner().clone();
    let guard = FlightGuard::acquire(AUDITION_BUSY_MSG)?;
    ensure_trainable_idle(&app)?;
    ensure_no_voice_render()?;
    let stem = ckpt_stem(&ckpt_path)?;
    let dir = audition_dir(&workspace, &stem);
    // cache is host-specific: switching the host must re-render
    let out = dir.join(format!(
        "audition_{}.wav",
        crate::models::sanitize_file_stem(&host_name)
    ));
    if out.is_file() {
        drop(guard);
        return Ok(out.to_string_lossy().into_owned());
    }
    let src = audition_source(&app)?;

    let apph = app_handle.clone();
    tauri::async_runtime::spawn_blocking(move || -> Result<String, String> {
        let _guard = guard;
        let cid = candidate_id.clone();
        with_terminal_events(&apph.clone(), &cid, || {
        std::fs::create_dir_all(&dir).map_err(|e| format!("AUDITION_DIR_CREATE_FAILED: {}", e))?;

        // host = an INSTALLED SoVITS model (type-scoped — same-name rvc pairs
        // are the app's standard usage, red-team A5 precedent)
        let entry = app
            .models
            .get_by_type(&host_name, &ModelType::SoVits)
            .ok_or_else(|| format!("AUDITION_HOST_MODEL_MISSING: {}", host_name))?;
        require_input(&entry, "noise")?;
        let dim = entry.config.resolved_features_dim()?;
        let hop_size = entry.config.hop_size.unwrap_or(512) as usize;

        // candidate assets → audition dir (encoder/denoiser/diffusion.json);
        // config.yaml auto-detected next to model_<step>.pt in diffusion/.
        // diffusion.json is written LAST by the exporter (after both graphs
        // selfcheck) — it is the completion marker
        let diff_dir = dir.join("diffusion");
        if !diff_dir.join("diffusion.json").is_file() {
            // S66 (review): same convert-slot exclusion as ensure_candidate_converted.
            if app.task_active("convert") {
                return Err("CONVERT_BUSY".to_string());
            }
            emit_phase(&apph, &candidate_id, "converting");
            crate::models::convert::convert_diffusion_assets(
                Path::new(&ckpt_path),
                None,
                &diff_dir,
                &app.app_dir,
            )
            .map_err(|e| {
                sweep_candidate_dir(&dir);
                e.to_string()
            })?;
        }
        // audition depth: k_step = min(100, candidate k_step_max) and a
        // speedup that keeps dpm++ solver_steps ≥ 2 (red-team F14: a
        // k_step_max<20 candidate would otherwise refuse to render)
        let dj: serde_json::Value =
            crate::commands::inference::read_json(&diff_dir.join("diffusion.json"), "diffusion config")?;
        let timesteps = dj.get("timesteps").and_then(|v| v.as_u64()).unwrap_or(1000) as u32;
        let k_max = {
            let k = dj.get("k_step_max").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            if k > 0 && k < timesteps {
                k
            } else {
                timesteps
            }
        };
        let k_step = k_max.min(100).max(1);
        let speedup = (k_step / 2).clamp(1, 10);

        emit_phase(&apph, &candidate_id, "rendering");
        let run_epoch = app.inference.begin_voice_run();
        // keep set (red-team R8): the host graph + its auto-f0 companion + the
        // CANDIDATE's diffusion dir. Deliberately NOT the host's attached
        // `.diffusion` — this run overrides it; a later real render reloads it
        // on miss.
        let host_path = entry.path.clone();
        {
            let mut keep = vec![host_path.clone(), diff_dir.clone()];
            if let (Some(parent), Some(hstem)) = (host_path.parent(), host_path.file_stem()) {
                keep.push(parent.join(format!("{}.f0.onnx", hstem.to_string_lossy())));
            }
            app.inference.engine.release_gpu_sessions_except(&keep);
        }
        app.inference
            .load_voice(
                &host_name,
                &host_path,
                VoiceBackendType::SoVits,
                entry.sample_rate,
                None,
            )
            .map_err(|e| e.to_string())?;

        let mut options = SovitsOptions {
            shallow_diffusion: true,
            k_step,
            diffusion_speedup: speedup,
            cluster_ratio: 0.0,
            gpu_extract: false,
            ..Default::default()
        };
        // resolve AFTER load_voice (S36 rule — see run_sovits); the override
        // points the diffusion branch at the candidate's converted assets
        let (diffusion, vocoder, f0_predictor) = resolve_sovits_quality(
            &app,
            &entry,
            dim,
            hop_size,
            &mut options,
            Some(diff_dir.as_path()),
        )?;

        let cv_sid = app
            .inference
            .ensure_aux_loaded_on(&contentvec_for_dim(&app, dim)?, false)
            .map_err(|e| e.to_string())?;
        let rmvpe_sid = app
            .inference
            .ensure_aux_loaded_on(&aux_path(&app, AUX_RMVPE, "RMVPE model")?, false)
            .map_err(|e| e.to_string())?;
        let mel = app
            .inference
            .load_npy(&aux_path(&app, AUX_RMVPE_MEL, "RMVPE mel filterbank")?)
            .map_err(|e| e.to_string())?;
        let handle = app
            .inference
            .voice_handle(&host_name)
            .map_err(|e| e.to_string())?;
        let host_has_input = |name: &str| {
            entry
                .config
                .inputs
                .as_ref()
                .and_then(|v| v.as_array())
                .map(|l| l.iter().any(|v| v.as_str() == Some(name)))
        };
        let vol_embedding = host_has_input("vol")
            .unwrap_or_else(|| entry.config.vol_embedding.unwrap_or(false));
        // ①c: multi-speaker host graph → feed the dense spk_mix (one-hot on speaker 0 here, no
        // blend UI on the audition path). α refuses multi-speaker diffusion so this is rarely hit,
        // but a multi-speaker host must not be fed the absent scalar `sid`.
        let spk_mix = if host_has_input("spk_mix") == Some(true) && entry.config.n_speakers > 0 {
            Some(entry.config.n_speakers as usize)
        } else {
            None
        };

        let audio_buf = crate::audio::load_audio(&src).map_err(|e| e.to_string())?;
        let progress = progress_emitter(apph.clone(), app.clone(), run_epoch, candidate_id.clone());
        let cancel = || app.inference.voice_cancelled(run_epoch);
        let model = sovits::SovitsModel {
            engine: &app.inference.engine,
            voice_session: &handle.session_id,
            contentvec_session: &cv_sid,
            rmvpe_session: &rmvpe_sid,
            mel_filters: mel.as_ref(),
            cluster: None,
            diffusion,
            vocoder,
            f0_predictor_session: f0_predictor,
            sample_rate: handle.sample_rate,
            hop_size,
            features_dim: dim,
            vol_embedding,
            spk_mix,
            unit_interpolate_mode: entry
                .config
                .unit_interpolate_mode
                .clone()
                .unwrap_or_else(|| "left".to_string()),
            noise_channels: noise_channels(&entry.config),
            min_frames: min_frames(&entry.config, 6),
        };
        let result = sovits::run_pipeline(&model, &audio_buf, &options, None, &progress, &cancel)
            .map_err(|e| e.to_string())?;
        write_wav_atomic(&out, &result.audio, result.sample_rate)?;
        // candidate diffusion sessions go; the host stays warm (normal model)
        app.inference.engine.unload_paths_with_prefix(&dir);
        Ok(out.to_string_lossy().into_owned())
        })
    })
    .await
    .map_err(|e| format!("AUDITION_TASK_PANICKED: {}", e))?
}

/// S60-4: audition an INSTALLED voice model from the resource manager — the SAME bare recipe
/// as the training audition (transpose 0 / rmvpe / no index/cluster/diffusion/enhancer / CPU
/// extract) over the SAME bundled clip, so the two surfaces are directly comparable. No ckpt
/// conversion (the model is already ONNX + a registry entry). Per-speaker cache lives in the
/// model's STEM FAMILY (`<stem>.audition_spk{N}.wav`) — delete AND same-name re-import both
/// invalidate it via remove_entry_files. Events: "audition-progress" with candidate_id
/// `model:<name>` (the shared protocol; the manager UI listens the same way the training page
/// does). Errors are stable CODEs: AUDITION_BAD_TYPE / AUDITION_MODEL_MISSING.
#[tauri::command]
pub async fn render_model_audition(
    app_handle: tauri::AppHandle,
    state: State<'_, Arc<AppState>>,
    name: String,
    model_type: String,
    speaker_id: Option<u32>,
) -> Result<String, String> {
    let app = state.inner().clone();
    let guard = FlightGuard::acquire(AUDITION_BUSY_MSG)?;
    ensure_trainable_idle(&app)?;
    ensure_no_voice_render()?;
    let backend = match model_type.as_str() {
        "rvc" => VoiceBackendType::Rvc,
        "sovits" => VoiceBackendType::SoVits,
        _ => return Err("AUDITION_BAD_TYPE".to_string()),
    };
    let mt = match backend {
        VoiceBackendType::Rvc => ModelType::Rvc,
        VoiceBackendType::SoVits => ModelType::SoVits,
    };
    let entry = app
        .models
        .get_by_type(&name, &mt)
        .ok_or_else(|| "AUDITION_MODEL_MISSING".to_string())?;
    let spk = speaker_id.unwrap_or(0);
    let stem = entry
        .path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "model".to_string());
    let dir = entry
        .path
        .parent()
        .ok_or_else(|| "AUDITION_MODEL_MISSING".to_string())?
        .to_path_buf();
    // S60c: the range record is a render input here (whole-clip pre-shift) — the cache name
    // carries it so a re-test / adjusted comfort misses the stale wav.
    let range = crate::inference::vocal_range::speaker_range(&entry.config, spk);
    let out = dir.join(format!(
        "{}.audition_spk{}{}.wav",
        stem,
        spk,
        audition_cache_tag(&range)
    ));
    if out.is_file() {
        drop(guard);
        return Ok(out.to_string_lossy().into_owned());
    }
    let src = audition_source(&app)?;
    let candidate_id = format!("model:{}", name);

    let apph = app_handle.clone();
    tauri::async_runtime::spawn_blocking(move || -> Result<String, String> {
        let _guard = guard;
        let cid = candidate_id.clone();
        with_terminal_events(&apph.clone(), &cid, || {
            emit_phase(&apph, &candidate_id, "rendering");
            let run_epoch = app.inference.begin_voice_run();
            // session hygiene per S36: evict foreign GPU sessions, keep this model's family
            {
                let mut keep = vec![entry.path.clone()];
                keep.push(dir.join(format!("{}.f0.onnx", stem)));
                keep.push(dir.join(format!("{}.diffusion", stem)));
                app.inference.engine.release_gpu_sessions_except(&keep);
            }
            // bare recipe → no index asset loaded (mirrors render_audition_voice)
            app.inference
                .load_voice(&name, &entry.path, backend.clone(), entry.sample_rate, None)
                .map_err(|e| e.to_string())?;
            let dim = entry.config.resolved_features_dim()?;
            let cv_sid = app
                .inference
                .ensure_aux_loaded_on(&contentvec_for_dim(&app, dim)?, false)
                .map_err(|e| e.to_string())?;
            let rmvpe_sid = app
                .inference
                .ensure_aux_loaded_on(&aux_path(&app, AUX_RMVPE, "RMVPE model")?, false)
                .map_err(|e| e.to_string())?;
            let mel = app
                .inference
                .load_npy(&aux_path(&app, AUX_RMVPE_MEL, "RMVPE mel filterbank")?)
                .map_err(|e| e.to_string())?;
            let handle = app.inference.voice_handle(&name).map_err(|e| e.to_string())?;
            let has_input = |input: &str| {
                entry
                    .config
                    .inputs
                    .as_ref()
                    .and_then(|v| v.as_array())
                    .map(|l| l.iter().any(|v| v.as_str() == Some(input)))
            };
            let spk_mix = if has_input("spk_mix") == Some(true) && entry.config.n_speakers > 0 {
                Some(entry.config.n_speakers as usize)
            } else {
                None
            };
            let audio_buf = crate::audio::load_audio(&src).map_err(|e| e.to_string())?;
            // S60c: whole-clip range pre-shift (single chunk = uniform formant character)
            let range_prep = audition_range_prep(&app, &rmvpe_sid, mel.as_ref(), &audio_buf, range)?;
            let range_f0_shift = range_prep.as_ref().map(|(s, _)| *s as f32).unwrap_or(0.0);
            let progress = progress_emitter(apph.clone(), app.clone(), run_epoch, candidate_id.clone());
            let cancel = || app.inference.voice_cancelled(run_epoch);
            let nch = noise_channels(&entry.config);

            let result = match backend {
                VoiceBackendType::Rvc => {
                    require_input(&entry, "rnd")?;
                    let options = RvcOptions {
                        index_ratio: 0.0,
                        speaker_id: Some(spk),
                        gpu_extract: false,
                        f0_shift: range_f0_shift,
                        ..Default::default()
                    };
                    let model = rvc::RvcModel {
                        engine: &app.inference.engine,
                        voice_session: &handle.session_id,
                        contentvec_session: &cv_sid,
                        rmvpe_session: &rmvpe_sid,
                        mel_filters: mel.as_ref(),
                        index: None, // bare recipe — retrieval off even when an index exists
                        sample_rate: handle.sample_rate,
                        features_dim: dim,
                        spk_mix,
                        noise_channels: nch,
                        min_frames: min_frames(&entry.config, 12),
                    };
                    rvc::run_pipeline(&model, &audio_buf, &options, None, &progress, &cancel)
                        .map_err(|e| e.to_string())?
                }
                VoiceBackendType::SoVits => {
                    require_input(&entry, "noise")?;
                    let hop_size = entry.config.hop_size.unwrap_or(512) as usize;
                    let vol_embedding = has_input("vol")
                        .unwrap_or_else(|| entry.config.vol_embedding.unwrap_or(false));
                    let options = SovitsOptions {
                        cluster_ratio: 0.0,
                        speaker_id: Some(spk),
                        gpu_extract: false,
                        f0_shift: range_f0_shift,
                        ..Default::default()
                    };
                    let model = sovits::SovitsModel {
                        engine: &app.inference.engine,
                        voice_session: &handle.session_id,
                        contentvec_session: &cv_sid,
                        rmvpe_session: &rmvpe_sid,
                        mel_filters: mel.as_ref(),
                        cluster: None,
                        diffusion: None, // bare recipe — the attachment stays out of the A/B
                        vocoder: None,
                        f0_predictor_session: None,
                        sample_rate: handle.sample_rate,
                        hop_size,
                        features_dim: dim,
                        vol_embedding,
                        spk_mix,
                        unit_interpolate_mode: entry
                            .config
                            .unit_interpolate_mode
                            .clone()
                            .unwrap_or_else(|| "left".to_string()),
                        noise_channels: nch,
                        min_frames: min_frames(&entry.config, 6),
                    };
                    sovits::run_pipeline(&model, &audio_buf, &options, None, &progress, &cancel)
                        .map_err(|e| e.to_string())?
                }
            };
            let result = audition_range_invert(result, &range_prep);
            write_wav_atomic(&out, &result.audio, result.sample_rate)?;
            Ok(out.to_string_lossy().into_owned())
        })
    })
    .await
    .map_err(|e| format!("AUDITION_TASK_PANICKED: {e}"))?
}

// ─── S60c: candidate range test (post-training auto-test, §user) ─────────────

/// Render an ARBITRARY score (the range-test scale) through a training CANDIDATE — converts
/// on demand (shared conversion source) and drives the SAME score render the vocal track uses
/// (score2svc). The frontend orchestrates the measurement (detect_f0 + the TS classification,
/// its single source) and persists the record via `set_candidate_vocal_range`; the audition
/// render then reads it for the whole-clip pre-shift. Speaker 0 only (the audition default).
#[tauri::command]
pub async fn render_candidate_scale(
    app_handle: tauri::AppHandle,
    state: State<'_, Arc<AppState>>,
    backend: String,
    ckpt_path: String,
    workspace: String,
    candidate_id: String,
    score: Vec<crate::commands::inference::ScoreNote>,
    output_path: String,
) -> Result<crate::inference::RenderedAudio, String> {
    let app = state.inner().clone();
    let guard = FlightGuard::acquire(AUDITION_BUSY_MSG)?;
    ensure_trainable_idle(&app)?;
    ensure_no_voice_render()?;
    if score.is_empty() {
        return Err("RANGE_EMPTY_SCORE".to_string());
    }
    let stem = ckpt_stem(&ckpt_path)?;
    let dir = audition_dir(&workspace, &stem);

    let apph = app_handle.clone();
    tauri::async_runtime::spawn_blocking(move || -> Result<crate::inference::RenderedAudio, String> {
        let _guard = guard;
        let cid = candidate_id.clone();
        // S67: this was the ONLY audition command without the terminal wrapper — its
        // converting/rendering phases stranded the row busy forever once the scale
        // finished (the post-training auto range test = the stuck 渲染中 + dead buttons).
        // wav_of = None: the scale probe must never enter the audition wav cache.
        with_terminal_events_as(&apph.clone(), &cid, |_| None, || {
        let onnx = dir.join("model.onnx");
        ensure_candidate_converted(&app, &apph, &candidate_id, &dir, &onnx, &backend, &ckpt_path)?;
        let (config, sample_rate) = read_candidate_config(&dir.join("model.json"))?;

        emit_phase(&apph, &candidate_id, "rendering");
        let run_epoch = app.inference.begin_voice_run();
        app.inference.engine.release_gpu_sessions_except(&[onnx.clone()]);
        let sid = app
            .inference
            .engine
            .load_model_with(&onnx, false)
            .map_err(|e| e.to_string())?;

        let dim = match backend.as_str() {
            "rvc" => config.features_dim as usize,
            _ => config.resolved_features_dim()?,
        };
        if dim == 0 {
            return Err("AUDITION_CONFIG_NO_FEATURES_DIM".to_string());
        }
        let s2cv_sid = app
            .inference
            .ensure_aux_loaded_on(&crate::commands::inference::score2cv_for_dim(&app, dim)?, true)
            .map_err(|e| e.to_string())?;
        let cv_sid = app
            .inference
            .ensure_aux_loaded_on(&contentvec_for_dim(&app, dim)?, false)
            .map_err(|e| e.to_string())?;
        let rmvpe_sid = app
            .inference
            .ensure_aux_loaded_on(&aux_path(&app, AUX_RMVPE, "RMVPE model")?, false)
            .map_err(|e| e.to_string())?;
        let mel = app
            .inference
            .load_npy(&aux_path(&app, AUX_RMVPE_MEL, "RMVPE mel filterbank")?)
            .map_err(|e| e.to_string())?;

        let has_input = |name: &str| {
            config
                .inputs
                .as_ref()
                .and_then(|v| v.as_array())
                .map(|l| l.iter().any(|v| v.as_str() == Some(name)))
        };
        let spk_mix = if has_input("spk_mix") == Some(true) && config.n_speakers > 0 {
            Some(config.n_speakers as usize)
        } else {
            None
        };
        let progress = progress_emitter(apph.clone(), app.clone(), run_epoch, candidate_id.clone());
        let cancel = || app.inference.voice_cancelled(run_epoch);
        use crate::inference::g2p;
        let score_ref: Vec<g2p::ScoreEvt> = score
            .iter()
            .map(|n| g2p::ScoreEvt {
                lyric: n.lyric.as_str(),
                note_num: n.note_num,
                frames: n.frames,
                lang: n.lang.and_then(g2p::Lang::from_id).unwrap_or(g2p::Lang::Ja),
                phoneme_input: n.phoneme_input.as_deref(),
            })
            .collect();

        let result = match backend.as_str() {
            "rvc" => {
                let options = RvcOptions { index_ratio: 0.0, gpu_extract: false, ..Default::default() };
                let model = crate::inference::rvc::RvcModel {
                    engine: &app.inference.engine,
                    voice_session: &sid,
                    contentvec_session: &cv_sid,
                    rmvpe_session: &rmvpe_sid,
                    mel_filters: mel.as_ref(),
                    index: None,
                    sample_rate,
                    features_dim: dim,
                    spk_mix,
                    noise_channels: noise_channels(&config),
                    min_frames: min_frames(&config, 12),
                };
                crate::inference::score2svc::render_score_rvc(
                    &model, &s2cv_sid, &score_ref, dim, 49, &g2p::GlobalDicts, &options,
                    0, 0, None, None, None, &cancel, &progress,
                )
                .map_err(|e| e.to_string())?
            }
            _ => {
                let hop_size = config.hop_size.unwrap_or(512) as usize;
                if hop_size == 0 {
                    return Err("AUDITION_CONFIG_HOP_ZERO".to_string());
                }
                let vol_embedding =
                    has_input("vol").unwrap_or_else(|| config.vol_embedding.unwrap_or(false));
                let options = SovitsOptions { cluster_ratio: 0.0, gpu_extract: false, ..Default::default() };
                let model = crate::inference::sovits::SovitsModel {
                    engine: &app.inference.engine,
                    voice_session: &sid,
                    contentvec_session: &cv_sid,
                    rmvpe_session: &rmvpe_sid,
                    mel_filters: mel.as_ref(),
                    cluster: None,
                    diffusion: None,
                    vocoder: None,
                    f0_predictor_session: None,
                    sample_rate,
                    hop_size,
                    features_dim: dim,
                    vol_embedding,
                    spk_mix,
                    unit_interpolate_mode: config
                        .unit_interpolate_mode
                        .clone()
                        .unwrap_or_else(|| "left".to_string()),
                    noise_channels: noise_channels(&config),
                    min_frames: min_frames(&config, 6),
                };
                crate::inference::score2svc::render_score_sovits(
                    &model, &s2cv_sid, &score_ref, dim, 49, &g2p::GlobalDicts, &options,
                    crate::commands::inference::VOCAL_FLAT_VOL, 0, 0, None, None, None,
                    &cancel, &progress,
                )
                .map_err(|e| e.to_string())?
            }
        };
        app.inference.engine.unload_paths_with_prefix(&dir);
        crate::commands::inference::commit_rendered_audio(result, output_path)
        })
    })
    .await
    .map_err(|e| format!("AUDITION_TASK_PANICKED: {e}"))?
}

/// Persist a candidate's tested vocal range into its AUDITION sidecar (model.json) — the
/// audition render reads it back for the whole-clip pre-shift. Strict read (a present-but-
/// corrupt sidecar must abort, never be rewritten from empty — the set_config_extra_key rule).
#[tauri::command]
pub async fn set_candidate_vocal_range(
    workspace: String,
    ckpt_path: String,
    record: serde_json::Value,
) -> Result<(), String> {
    if !record.is_object() {
        return Err("RANGE_BAD_RECORD".to_string());
    }
    crate::inference::vocal_range::validate_range_record(&record)?;
    let stem = ckpt_stem(&ckpt_path)?;
    let json_path = audition_dir(&workspace, &stem).join("model.json");
    let text = std::fs::read_to_string(&json_path)
        .map_err(|e| format!("RANGE_CANDIDATE_MISSING: {e}"))?;
    let mut map: serde_json::Map<String, serde_json::Value> = serde_json::from_str(&text)
        .map_err(|e| format!("RANGE_CANDIDATE_MISSING: unparseable sidecar: {e}"))?;
    map.insert("vocal_range".to_string(), record);
    // tmp+rename (S67 review): the row un-busies before this write lands, so a user
    // audition's strict read_candidate_config could otherwise catch a half-written json
    let tmp = json_path.with_extension("json.tmp");
    std::fs::write(
        &tmp,
        serde_json::to_string_pretty(&serde_json::Value::Object(map))
            .map_err(|e| format!("RANGE_CANDIDATE_MISSING: serialize: {e}"))?,
    )
    .map_err(|e| format!("RANGE_CANDIDATE_MISSING: write: {e}"))?;
    std::fs::rename(&tmp, &json_path).map_err(|e| format!("RANGE_CANDIDATE_MISSING: write: {e}"))?;
    Ok(())
}

/// Read back a candidate's tested vocal range from its audition sidecar. `None` =
/// no sidecar yet / no record stored — never an error (S67: the post-training auto
/// battery short-circuits on an existing record instead of re-measuring every
/// candidate on every page mount, and the row label restores from disk).
#[tauri::command]
pub async fn get_candidate_vocal_range(
    workspace: String,
    ckpt_path: String,
) -> Result<Option<serde_json::Value>, String> {
    let stem = ckpt_stem(&ckpt_path)?;
    let json_path = audition_dir(&workspace, &stem).join("model.json");
    let Ok(text) = std::fs::read_to_string(&json_path) else {
        return Ok(None);
    };
    let Ok(map) = serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(&text) else {
        return Ok(None);
    };
    Ok(map.get("vocal_range").cloned())
}
