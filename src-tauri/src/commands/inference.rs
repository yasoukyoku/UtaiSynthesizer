use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;
use tauri::{Emitter, State};

use crate::inference::{
    g2p, rvc, score2cv, score2svc, sovits, RenderedAudio, RvcOptions, SovitsOptions, SynthesisResult,
    VoiceBackendType,
};
use crate::models::{ModelConfig, ModelEntry};
use crate::AppState;

/// S41 audition interlock (审查修复 S41-INT-3): formal voice renders and
/// audition renders both open by evicting every foreign GPU session and share
/// the global cancel epoch — running them concurrently is a VRAM tug-of-war
/// with cross-kill cancels. Both sides reject the other with a friendly error.
static VOICE_RENDER_ACTIVE: AtomicUsize = AtomicUsize::new(0);

pub(crate) fn voice_render_active() -> bool {
    VOICE_RENDER_ACTIVE.load(Ordering::SeqCst) > 0
}

struct VoiceRunGuard;
impl VoiceRunGuard {
    fn acquire() -> Result<Self, String> {
        if crate::commands::audition::AUDITION_IN_FLIGHT.load(Ordering::SeqCst) {
            // Generic on purpose: the flag's holder may be an audition OR a storage cleanup (S61).
            return Err(crate::commands::audition::BUSY_RETRY_MSG.into());
        }
        VOICE_RENDER_ACTIVE.fetch_add(1, Ordering::SeqCst);
        Ok(VoiceRunGuard)
    }
}
impl Drop for VoiceRunGuard {
    fn drop(&mut self) {
        VOICE_RENDER_ACTIVE.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Write a finished pipeline result to `output_path` (16-bit like the old save_temp_audio round
/// trip — byte-identical quantization — but atomic) and return the path-bearing IPC payload
/// (S66 / O5: only the path crosses IPC; the ~100 MB `SynthesisResult` JSON double-trip is gone).
/// Call from inside the render's spawn_blocking task — file IO must never ride the async workers.
pub(crate) fn commit_rendered_audio(
    result: SynthesisResult,
    output_path: String,
) -> Result<RenderedAudio, String> {
    let out = PathBuf::from(&output_path);
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("RENDER_WRITE_FAILED: {}", e))?;
    }
    let sample_rate = result.sample_rate;
    let buf = crate::audio::AudioBuffer {
        samples: result.audio,
        sample_rate,
        channels: 1,
    };
    crate::audio::save_wav_atomic(&out, &buf).map_err(|e| format!("RENDER_WRITE_FAILED: {}", e))?;
    Ok(RenderedAudio { path: output_path, sample_rate })
}

/// Per-node inference progress, emitted as the `voice-progress` event. The frontend workflow
/// engine listens during the run_rvc/run_sovits invoke and drives the node's progress bar,
/// filtering by node_id.
#[derive(Clone, serde::Serialize)]
struct VoiceProgress {
    node_id: String,
    progress: f32,
}

/// Build a progress callback that emits throttled `voice-progress` events (only on a ≥1% step,
/// plus the terminal 1.0) so a many-chunk RVC run doesn't spam the event bus. A CANCELLED run
/// goes silent immediately: its pipeline may keep draining until the next cancel poll (a
/// multi-second ONNX Run), and late emissions would fight a freshly started run's bar for the
/// same node.
fn progress_emitter(
    app: tauri::AppHandle,
    state: Arc<AppState>,
    run_epoch: u64,
    node_id: String,
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
                "voice-progress",
                VoiceProgress {
                    node_id: node_id.clone(),
                    progress: p,
                },
            );
        }
    }
}

// ─── aux model resolution (models_dir/auxiliary/..., models::AUX_DIR_NAME) ───

// pub(crate): the S41 audition commands (commands/audition.rs) resolve the
// same aux fleet without going through the registry
pub(crate) const AUX_CONTENTVEC_768: &str = "contentvec_768l12.onnx";
pub(crate) const AUX_CONTENTVEC_256: &str = "contentvec_256l9.onnx";
pub(crate) const AUX_RMVPE: &str = "rmvpe_e2e.onnx";
pub(crate) const AUX_RMVPE_MEL: &str = "rmvpe_mel_filters.npy";
// S36 quality path: the NSF-HiFiGAN vocoder (shared by shallow diffusion + the enhancer),
// exported once by converter/export_nsf_hifigan.py alongside its sidecar json + filterbank.
pub(crate) const AUX_NSF_HIFIGAN: &str = "nsf_hifigan.onnx";
pub(crate) const AUX_NSF_HIFIGAN_JSON: &str = "nsf_hifigan.json";
pub(crate) const AUX_NSF_HIFIGAN_MEL: &str = "nsf_hifigan_mel.npy";
// ② 自己唱 (S48 Phase 6): the ScoreToCV content models (score → cv[T,dim] @50fps), aux infra like
// ContentVec — resolved by direct path, NOT the registry (models/mod.rs scan() must not surface them
// as phantom user voices). 768 = SoVITS4.1/RVCv2, 256 = SoVITS4.0 (picked by the VOICE's features_dim).
pub(crate) const AUX_SCORE2CV_768: &str = "score2cv_768.onnx";
pub(crate) const AUX_SCORE2CV_256: &str = "score2cv_256.onnx";

/// models_dir/auxiliary/<filename>, with a stable CODE naming the missing file + the
/// exact directory it must be placed in (the frontend maps the code to localized text;
/// `label` is a short English token interpolated into the detail payload).
pub(crate) fn aux_path(state: &AppState, filename: &str, label: &str) -> Result<PathBuf, String> {
    let dir = state.models.aux_dir();
    let path = dir.join(filename);
    if !path.exists() {
        return Err(format!(
            "AUX_FILE_MISSING: {} {} (place into {})",
            label,
            filename,
            dir.display()
        ));
    }
    Ok(path)
}

/// ContentVec variant routing: vec768l12 → RVC v2 / SoVITS 4.1, vec256l9 → RVC v1 / SoVITS 4.0.
pub(crate) fn contentvec_for_dim(state: &AppState, dim: usize) -> Result<PathBuf, String> {
    match dim {
        768 => aux_path(state, AUX_CONTENTVEC_768, "ContentVec model"),
        256 => aux_path(state, AUX_CONTENTVEC_256, "ContentVec model"),
        other => Err(format!(
            "FEATURES_DIM_UNSUPPORTED: {} (only 256 / 768; check features_dim / speech_encoder)",
            other
        )),
    }
}

/// ② 自己唱: the ScoreToCV model for a voice's feature dim (768 → SoVITS4.1/RVCv2, 256 → SoVITS4.0).
/// Same aux-by-path resolution as `contentvec_for_dim` (the score render swaps ScoreToCV in for the
/// audio ContentVec extractor). A missing model names the file + the aux dir it must go in.
pub(crate) fn score2cv_for_dim(state: &AppState, dim: usize) -> Result<PathBuf, String> {
    match dim {
        768 => aux_path(state, AUX_SCORE2CV_768, "ScoreToCV model"),
        256 => aux_path(state, AUX_SCORE2CV_256, "ScoreToCV model"),
        other => Err(format!(
            "SCORE2CV_DIM_UNSUPPORTED: {} (only 256 / 768; check the voice's features_dim)",
            other
        )),
    }
}

/// Effective feature dim — delegates to the single source on ModelConfig (shared with the
/// import-time diffusion-attachment cross-check).
fn features_dim(config: &ModelConfig) -> Result<usize, String> {
    config.resolved_features_dim()
}

/// inter_channels of the model's noise input, from the sidecar "noise" block when present
/// (converter writes {"rnd_input"/"noise_input": [1, C, "T"]}); 192 for every standard
/// RVC v1/v2 and SoVITS 4.x config.
pub(crate) fn noise_channels(config: &ModelConfig) -> usize {
    config
        .noise
        .as_ref()
        .and_then(|v| v.get("rnd_input").or_else(|| v.get("noise_input")))
        .and_then(|v| v.as_array())
        .and_then(|a| a.get(1))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(192)
}

/// 4.0-v2 (VISinger2): bin count of the model's explicit `phase` input, from the sidecar
/// "phase" block (converter writes {"phase_input": [1, n_fft/2+1, "T"]}). None for every
/// 4.0/4.1 export (no such block) → the phase tensor is not fed.
pub(crate) fn phase_bins(config: &ModelConfig) -> Option<usize> {
    config
        .extra
        .get("phase")
        .and_then(|v| v.get("phase_input"))
        .and_then(|v| v.as_array())
        .and_then(|a| a.get(1))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
}

/// 4.0-v2 (VISinger2): channel count of the model's `f0d_cond` input (export
/// deviation 7 — the upstream auto-f0 detach-alias side effect made explicit),
/// from the sidecar "f0d_cond" block {"input": [1, prior_hidden, "T"]}. None for
/// every 4.0/4.1 export.
pub(crate) fn f0d_cond_channels(config: &ModelConfig) -> Option<usize> {
    config
        .extra
        .get("f0d_cond")
        .and_then(|v| v.get("input"))
        .and_then(|v| v.as_array())
        .and_then(|a| a.get(1))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
}

/// Sidecar "min_frames": the minimum T the exported graph accepts (final contract:
/// RVC 12 / SoVITS 6). Tolerant field — lives in ModelConfig.extra.
pub(crate) fn min_frames(config: &ModelConfig, default: usize) -> usize {
    config
        .extra
        .get("min_frames")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(default)
        .max(1)
}

/// Whether the sidecar "inputs" array contains `input` (None when the sidecar predates
/// the converter rework and has no such array).
fn sidecar_has_input(entry: &ModelEntry, input: &str) -> Option<bool> {
    entry
        .config
        .inputs
        .as_ref()
        .and_then(|v| v.as_array())
        .map(|list| list.iter().any(|v| v.as_str() == Some(input)))
}

/// New-signature guard: the S35 converter ALWAYS writes an `inputs` array listing the graph
/// inputs. Proceed ONLY when that array is present AND contains the required new input. Both a
/// missing input (Some(false), old export WITH an inputs list) and a missing inputs array
/// (None, pre-rework sidecar that never wrote one) mean the ONNX predates the rework — fail with
/// an actionable message instead of a cryptic raw ORT "Invalid Feed Input Name" crash.
pub(crate) fn require_input(entry: &ModelEntry, input: &str) -> Result<(), String> {
    if sidecar_has_input(entry, input) != Some(true) {
        return Err(format!(
            "MODEL_LEGACY_EXPORT: {} (missing '{}' input signature)",
            entry.name, input
        ));
    }
    Ok(())
}

pub(crate) fn get_entry(state: &AppState, voice_name: &str) -> Result<ModelEntry, String> {
    state.models.get(voice_name).ok_or_else(|| {
        format!("MODEL_NOT_FOUND: {}", voice_name)
    })
}

// ─── cancel_voice ────────────────────────────────────────────────────────────

/// Abort the in-flight voice run(s). Global like cancel_separation — the pipelines poll the
/// flag per piece / per diffusion step; each run_rvc/run_sovits re-arms it at start.
#[tauri::command]
pub async fn cancel_voice(state: State<'_, Arc<AppState>>) -> Result<(), String> {
    state.inference.cancel_voice();
    Ok(())
}

// ─── S36 quality-path sidecars ───────────────────────────────────────────────

/// `<stem>.diffusion/diffusion.json` — written by converter/export_diffusion.py. Strict on
/// the fields the runtime cannot guess (schedule facts); everything is validated against the
/// MAIN model in resolve_sovits_quality.
#[derive(serde::Deserialize)]
struct DiffusionSidecar {
    #[serde(default)]
    encoder_out_channels: u32,
    #[serde(default)]
    sample_rate: u32,
    #[serde(default)]
    block_size: u32,
    // Schedule/net facts have NO silent fallbacks: the converter always writes them, so a
    // missing one means a corrupt/foreign diffusion.json — hard-error (re-attach) instead
    // of quietly running with guessed constants that produce garbage audio.
    n_hidden: Option<u32>,
    #[serde(default)]
    timesteps: u32,
    #[serde(default)]
    k_step_max: u32,
    #[serde(default)]
    schedule: String,
    max_beta: Option<f64>,
    spec_min: Option<Vec<f32>>,
    spec_max: Option<Vec<f32>>,
    #[serde(default = "one")]
    n_spk: u32,
    #[serde(default)]
    unit_interpolate_mode: Option<String>,
    #[serde(default)]
    files: DiffusionFiles,
}
fn one() -> u32 {
    1
}

#[derive(serde::Deserialize, Default)]
struct DiffusionFiles {
    #[serde(default)]
    encoder: String,
    #[serde(default)]
    denoiser: String,
}

/// nsf_hifigan sidecar json — export_nsf_hifigan.py schema (the aux default
/// AND the S40 vocoder resources under models/nsf_hifigan/ share it).
#[derive(serde::Deserialize)]
struct VocoderSidecar {
    #[serde(default)]
    sample_rate: u32,
    #[serde(default)]
    hop_size: u32,
    #[serde(default)]
    num_mels: u32,
    #[serde(default)]
    mel_filters: Option<String>,
    // full mel recipe — resource vocoders are checked field-by-field against
    // the standard format (设计红队 A9: same geometry with a different recipe,
    // e.g. fmax 8000, would silently mismatch the diffusion training domain)
    #[serde(default)]
    n_fft: Option<f64>,
    #[serde(default)]
    win_size: Option<f64>,
    #[serde(default)]
    fmin: Option<f64>,
    #[serde(default)]
    fmax: Option<f64>,
}

/// 一期唯一声码器格式类 = the OpenVPI standard (the aux default vocoder's
/// recipe — the domain every SoVITS diffusion attachment and the enhancer mel
/// are anchored to). Mirrored by VOCODER_STD_FORMAT in src/store/voice-models.ts.
const VOCODER_STD_N_FFT: f64 = 2048.0;
const VOCODER_STD_WIN_SIZE: f64 = 2048.0;
const VOCODER_STD_FMIN: f64 = 40.0;
const VOCODER_STD_FMAX: f64 = 16000.0;
/// 审查修复: A9 says FULL-field equality incl. 128 — an 80-mel vocoder passes
/// every geometry check (its own filterbank is self-consistently 80 rows) and
/// only dies two layers away inside the denoiser graph (or silently degrades
/// the enhancer path, which is self-consistent at ANY bin count).
const VOCODER_STD_NUM_MELS: f64 = 128.0;

/// S40: facts about the BUILT-IN default vocoder (aux/nsf_hifigan.* — app
/// infrastructure, not a registry entry) for the resource manager's pinned
/// read-only row: zero-knowledge users must be able to see what the node
/// dropdown's「默认声码器」refers to, its format class, and — when the aux
/// files are missing — learn it HERE instead of at render time.
#[derive(serde::Serialize)]
pub struct DefaultVocoderInfo {
    /// all three files (onnx + sidecar json + mel filterbank npy) present
    pub present: bool,
    /// file names missing from models/aux (diagnostics for the warning chip)
    pub missing: Vec<String>,
    pub sample_rate: Option<u32>,
    pub hop_size: Option<u32>,
    pub num_mels: Option<u32>,
}

#[tauri::command]
pub fn get_default_vocoder_info(state: State<'_, Arc<AppState>>) -> DefaultVocoderInfo {
    let aux = state.models.aux_dir();
    let json_path = aux.join(AUX_NSF_HIFIGAN_JSON);
    let sidecar: Option<VocoderSidecar> = std::fs::read_to_string(&json_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok());
    let mel_name = sidecar
        .as_ref()
        .and_then(|s| s.mel_filters.clone())
        .unwrap_or_else(|| AUX_NSF_HIFIGAN_MEL.to_string());
    let mut missing = Vec::new();
    for name in [AUX_NSF_HIFIGAN, AUX_NSF_HIFIGAN_JSON, mel_name.as_str()] {
        if !aux.join(name).is_file() {
            missing.push(name.to_string());
        }
    }
    DefaultVocoderInfo {
        present: missing.is_empty(),
        missing,
        sample_rate: sidecar.as_ref().map(|s| s.sample_rate),
        hop_size: sidecar.as_ref().map(|s| s.hop_size),
        num_mels: sidecar.as_ref().map(|s| s.num_mels),
    }
}

pub(crate) fn read_json<T: serde::de::DeserializeOwned>(path: &PathBuf, what: &str) -> Result<T, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("FILE_READ_FAILED: {} ({}): {}", what, path.display(), e))?;
    serde_json::from_str(&content)
        .map_err(|e| format!("JSON_PARSE_FAILED: {} ({}): {}", what, path.display(), e))
}

/// ①c guard — the SoVITS shallow/only-diffusion CONDITION encoder (sovits.rs `run_diffusion`) does NOT
/// honor a spk_mix BLEND: it one-hots a SINGLE speaker, so a genuine multi-speaker interpolation together
/// with a diffusion companion would silently pull the timbre back toward one speaker (only_diffusion drops
/// the blend entirely; shallow drops it partially). The core blend on the VITS net_g is bit-exact (verified
/// S56), and training refuses multi-speaker diffusion, so a properly-produced blend model has NO companion
/// and this never fires — it catches the pathological "diffusion attached to a real multi-speaker blend"
/// combo with a clear error instead of wrong audio. A single-speaker selection via spk_mix is fine (≤1
/// distinct id → the diffusion's single speaker matches).
fn guard_blend_vs_diffusion(
    entry: &ModelEntry,
    spk_mix: &[crate::inference::SpkMixEntry],
    diffusion_present: bool,
) -> std::result::Result<(), String> {
    if !diffusion_present || sidecar_has_input(entry, "spk_mix") != Some(true) {
        return Ok(());
    }
    let distinct: std::collections::HashSet<u32> =
        spk_mix.iter().filter(|e| e.weight > 0.0).map(|e| e.id).collect();
    if distinct.len() >= 2 {
        // Stable CODE, not a hardcoded Chinese message — the frontend maps it to t(...) (i18n rule, S56).
        return Err("SPK_MIX_DIFFUSION".to_string());
    }
    Ok(())
}

/// Everything the SoVITS quality path needs beyond the plain S35 pipeline: the vocoder
/// runtime (diffusion OR enhancer), the diffusion runtime (validated against the main
/// model + the run options), and the auto-f0 predictor session. Also enforces the original
/// mutual exclusions by MUTATING options (enhancer forced off under diffusion — original
/// infer_tool.py:183-184 behavior, surfaced as a warn instead of silence).
/// `diffusion_dir_override`: S41 audition ONLY — render through a candidate's
/// freshly converted `.diffusion` assets in the workspace audition dir instead
/// of the entry's attached ones. `None` = the S36/S40 behavior, line-for-line
/// (every production caller passes None; the override changes nothing but
/// WHERE the diffusion dir is looked up).
pub(crate) fn resolve_sovits_quality(
    app: &Arc<AppState>,
    entry: &ModelEntry,
    dim: usize,
    hop_size: usize,
    options: &mut SovitsOptions,
    diffusion_dir_override: Option<&std::path::Path>,
) -> Result<
    (
        Option<sovits::DiffusionRuntime>,
        Option<sovits::VocoderRuntime>,
        Option<String>,
    ),
    String,
> {
    let diffusion_on = options.shallow_diffusion || options.only_diffusion;

    // Original mutual exclusion: any diffusion mode disables the enhancer.
    if diffusion_on && options.nsf_enhance {
        tracing::warn!("shallow/only-diffusion active — NSF enhancer disabled (upstream mutual exclusion)");
        options.nsf_enhance = false;
    }

    // ── vocoder (needed by diffusion AND the enhancer) ──
    let vocoder = if diffusion_on || options.nsf_enhance {
        // S40: an installed vocoder RESOURCE by registry name, else the aux
        // default (byte-identical S36 path). "" normalizes to None (设计红队
        // A21 — the frontend sentinel for「默认声码器」).
        let picked = options
            .vocoder_name
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string);
        let (voc_path, voc_json, voc_base_dir, voc_what) = match picked.as_deref() {
            None => (
                aux_path(app, AUX_NSF_HIFIGAN, "NSF-HiFiGAN vocoder")?,
                aux_path(app, AUX_NSF_HIFIGAN_JSON, "NSF-HiFiGAN vocoder config")?,
                app.models.aux_dir(),
                "NSF-HiFiGAN vocoder".to_string(),
            ),
            Some(name) => {
                // type-scoped lookup (设计红队 A5): singers commonly own a
                // same-name rvc/sovits pair AND a same-name vocoder
                let ventry = app
                    .models
                    .get_by_type(name, &crate::models::ModelType::NsfHifigan)
                    .ok_or_else(|| format!("VOCODER_NOT_FOUND: {}", name))?;
                let json = ventry.path.with_extension("json");
                if !json.is_file() {
                    return Err(format!(
                        "VOCODER_CONFIG_MISSING: {} ({})",
                        name,
                        json.display()
                    ));
                }
                let base = ventry
                    .path
                    .parent()
                    .map(|p| p.to_path_buf())
                    .unwrap_or_default();
                (ventry.path.clone(), json, base, format!("vocoder '{}'", name))
            }
        };
        let sidecar: VocoderSidecar = read_json(&voc_json, "vocoder config")?;
        let mel_name = sidecar
            .mel_filters
            .clone()
            .unwrap_or_else(|| AUX_NSF_HIFIGAN_MEL.to_string());
        // mel_filters resolves against ITS OWN sidecar's directory — a resource
        // vocoder's filterbank must NEVER fall back to the aux file of the same
        // name (设计红队 A6: silent wrong-filterbank path)
        let voc_mel_path = voc_base_dir.join(&mel_name);
        if !voc_mel_path.is_file() {
            // 审查修复: the aux default vocoder is NOT importable — telling the
            // user to "re-import" it is a dead end; restore the S36-style
            // place-the-file guidance for the None branch
            return Err(if picked.is_some() {
                format!(
                    "VOCODER_MEL_MISSING: {} ({})",
                    voc_what,
                    voc_mel_path.display()
                )
            } else {
                format!(
                    "AUX_VOCODER_MEL_MISSING: {} (place into {})",
                    mel_name,
                    voc_base_dir.display()
                )
            });
        }
        if picked.is_some() {
            // resource vocoders: FULL recipe equality against the standard
            // format class; a missing field = an unverifiable format = refuse
            for (key, got, want) in [
                ("n_fft", sidecar.n_fft, VOCODER_STD_N_FFT),
                ("win_size", sidecar.win_size, VOCODER_STD_WIN_SIZE),
                ("fmin", sidecar.fmin, VOCODER_STD_FMIN),
                ("fmax", sidecar.fmax, VOCODER_STD_FMAX),
                ("num_mels", Some(sidecar.num_mels as f64), VOCODER_STD_NUM_MELS),
            ] {
                match got {
                    None => {
                        return Err(format!(
                            "VOCODER_CONFIG_FIELD_MISSING: {} '{}'",
                            voc_what, key
                        ))
                    }
                    Some(v) if v != want => {
                        return Err(format!(
                            "VOCODER_MEL_FORMAT_MISMATCH: {} {} = {} (standard requires {})",
                            voc_what, key, v, want
                        ))
                    }
                    _ => {}
                }
            }
        }
        if sidecar.sample_rate != entry.sample_rate || sidecar.hop_size as usize != hop_size {
            return Err(format!(
                "VOCODER_GEOMETRY_MISMATCH: {} ({}Hz/hop {}) vs model ({}Hz/hop {})",
                voc_what, sidecar.sample_rate, sidecar.hop_size, entry.sample_rate, hop_size
            ));
        }
        let filters = app
            .inference
            .load_npy(&voc_mel_path)
            .map_err(|e| e.to_string())?;
        if filters.nrows() != sidecar.num_mels as usize {
            return Err(format!(
                "VOCODER_FILTER_SHAPE_MISMATCH: {}x{} vs num_mels={}",
                filters.nrows(),
                filters.ncols(),
                sidecar.num_mels
            ));
        }
        // The vocoder is a per-piece hot loop → global device (GPU when available),
        // mem_pattern off (dynamic T).
        let sid = app
            .inference
            .engine
            .load_model_with(&voc_path, false)
            .map_err(|e| e.to_string())?;
        Some(sovits::VocoderRuntime {
            session: sid,
            mel_filters: filters,
            cfg: crate::inference::nsf_hifigan::VocoderConfig {
                sample_rate: sidecar.sample_rate,
                hop_size: sidecar.hop_size as usize,
                num_mels: sidecar.num_mels as usize,
            },
        })
    } else {
        None
    };

    // ── diffusion runtime ──
    let diffusion = if diffusion_on {
        let diff_dir = match diffusion_dir_override {
            Some(dir) => dir.to_path_buf(),
            None => entry.diffusion_path.clone().ok_or_else(|| {
                format!("DIFFUSION_NOT_ATTACHED: {}", entry.name)
            })?,
        };
        let sidecar: DiffusionSidecar =
            read_json(&diff_dir.join("diffusion.json"), "diffusion config")?;

        if sidecar.schedule != "linear" || sidecar.timesteps == 0 {
            return Err(format!(
                "DIFFUSION_SCHEDULE_UNSUPPORTED: {} (timesteps={})",
                sidecar.schedule, sidecar.timesteps
            ));
        }
        if sidecar.encoder_out_channels as usize != dim {
            return Err(format!(
                "DIFFUSION_DIM_MISMATCH: {} vs {}",
                sidecar.encoder_out_channels, dim
            ));
        }
        if sidecar.sample_rate != entry.sample_rate || sidecar.block_size as usize != hop_size {
            return Err(format!(
                "DIFFUSION_GEOMETRY_MISMATCH: {}Hz/block {} vs model {}Hz/hop {}",
                sidecar.sample_rate, sidecar.block_size, entry.sample_rate, hop_size
            ));
        }
        let method = crate::inference::diffusion::SamplerMethod::parse(&options.diffusion_method)
            .ok_or_else(|| {
                format!(
                    "DIFFUSION_SAMPLER_UNKNOWN: {} (naive/ddim/pndm/dpm-solver/dpm-solver++/unipc)",
                    options.diffusion_method
                )
            })?;
        let timesteps = sidecar.timesteps as usize;
        // Same resolution rule as unit2mel.py:87 / DiffusionSchedule::linear: 0 or
        // ≥timesteps → timesteps (full-diffusion-capable) — NOT a floor of 1.
        let k_step_max = {
            let k = sidecar.k_step_max as usize;
            if k > 0 && k < timesteps { k } else { timesteps }
        };
        if options.only_diffusion {
            if k_step_max < timesteps {
                return Err(
                    "DIFFUSION_SHALLOW_ONLY: k_step_max < timesteps".to_string(),
                );
            }
        } else {
            if options.k_step == 0 {
                return Err("DIFFUSION_KSTEP_ZERO".to_string());
            }
            if options.k_step as usize > k_step_max {
                return Err(format!(
                    "DIFFUSION_KSTEP_EXCEEDS_MAX: {} > k_step_max={}",
                    options.k_step, k_step_max
                ));
            }
        }
        // dpm/unipc need ≥2 solver steps (original asserts steps >= order); the ≤1-speedup
        // case legitimately falls back to the plain DDPM loop (original semantics).
        if options.diffusion_speedup > 1
            && matches!(
                method,
                crate::inference::diffusion::SamplerMethod::DpmSolver
                    | crate::inference::diffusion::SamplerMethod::DpmSolverPp
                    | crate::inference::diffusion::SamplerMethod::UniPc
            )
        {
            let t_total = if options.only_diffusion {
                k_step_max
            } else {
                options.k_step as usize
            };
            let solver_steps = t_total / options.diffusion_speedup.max(1) as usize;
            if solver_steps < 2 {
                return Err(format!(
                    "DIFFUSION_SPEEDUP_TOO_FEW_STEPS: {} (dpm/unipc need >= 2)",
                    solver_steps
                ));
            }
        }
        if sidecar.n_spk > 1 {
            let spk = options.speaker_id.unwrap_or(0);
            if spk >= sidecar.n_spk {
                return Err(format!(
                    "DIFFUSION_SPEAKER_OUT_OF_RANGE: {} >= n_spk={}",
                    spk, sidecar.n_spk
                ));
            }
        }

        let enc_name = if sidecar.files.encoder.is_empty() {
            "encoder.onnx".to_string()
        } else {
            sidecar.files.encoder.clone()
        };
        let den_name = if sidecar.files.denoiser.is_empty() {
            "denoiser.onnx".to_string()
        } else {
            sidecar.files.denoiser.clone()
        };
        let enc_path = diff_dir.join(&enc_name);
        let den_path = diff_dir.join(&den_name);
        for p in [&enc_path, &den_path] {
            if !p.exists() {
                return Err(format!("DIFFUSION_FILE_MISSING: {}", p.display()));
            }
        }
        let enc_sid = app
            .inference
            .engine
            .load_model_with(&enc_path, false)
            .map_err(|e| e.to_string())?;
        let den_sid = app
            .inference
            .engine
            .load_model_with(&den_path, false)
            .map_err(|e| e.to_string())?;

        // No silent fallbacks (converter always writes these — absent = corrupt sidecar).
        let corrupt = |what: &str| {
            format!("DIFFUSION_SIDECAR_FIELD_MISSING: {}", what)
        };
        let max_beta = sidecar.max_beta.ok_or_else(|| corrupt("max_beta"))?;
        let spec_min = sidecar.spec_min.clone().filter(|v| !v.is_empty()).ok_or_else(|| corrupt("spec_min"))?;
        let spec_max = sidecar.spec_max.clone().filter(|v| !v.is_empty()).ok_or_else(|| corrupt("spec_max"))?;
        let n_hidden = sidecar.n_hidden.filter(|&v| v > 0).ok_or_else(|| corrupt("n_hidden"))? as usize;

        let schedule = crate::inference::diffusion::DiffusionSchedule::linear(
            timesteps,
            max_beta,
            &spec_min,
            &spec_max,
            k_step_max,
        );

        Some(sovits::DiffusionRuntime {
            encoder_session: enc_sid,
            denoiser_session: den_sid,
            schedule,
            method,
            n_hidden,
            n_spk: sidecar.n_spk as usize,
            // only_diffusion expands ContentVec with the DIFFUSION yaml's mode (original
            // infer_tool.py:156); shallow keeps the main model's (line 142). Default 'left'
            // mirrors the original's None-fallback.
            unit_interpolate_mode: sidecar
                .unit_interpolate_mode
                .clone()
                .unwrap_or_else(|| "left".to_string()),
        })
    } else {
        None
    };

    // ── auto-f0 predictor ──
    let f0_predictor = if options.auto_f0 {
        let auto = entry.config.extra.get("auto_f0");
        let available = auto
            .and_then(|v| v.get("available"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if !available {
            return Err(format!("AUTO_F0_NOT_EXPORTED: {}", entry.name));
        }
        let file = auto
            .and_then(|v| v.get("file"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                format!(
                    "{}.f0.onnx",
                    entry.path.file_stem().unwrap_or_default().to_string_lossy()
                )
            });
        let f0_path = entry
            .path
            .parent()
            .map(|p| p.join(&file))
            .filter(|p| p.exists())
            .ok_or_else(|| format!("AUTO_F0_FILE_MISSING: {}", file))?;
        let sid = app
            .inference
            .engine
            .load_model_with(&f0_path, false)
            .map_err(|e| e.to_string())?;
        Some(sid)
    } else {
        None
    };

    Ok((diffusion, vocoder, f0_predictor))
}

// ─── run_rvc ─────────────────────────────────────────────────────────────────

#[tauri::command]
pub async fn run_rvc(
    app_handle: tauri::AppHandle,
    state: State<'_, Arc<AppState>>,
    voice_name: String,
    model_path: String,
    audio_path: String,
    node_id: String,
    output_path: String,
    options: RvcOptions,
) -> Result<RenderedAudio, String> {
    let app = state.inner().clone();
    let _voice_guard = VoiceRunGuard::acquire()?; // held to the end of the render
    // Arm the cancel epoch BEFORE the multi-second load phase (a cancel during loading
    // must be honored at the first pipeline poll).
    let run_epoch = app.inference.begin_voice_run();
    let entry = get_entry(&app, &voice_name)?;
    require_input(&entry, "rnd")?;

    let dim = entry.config.features_dim as usize; // RVC sidecars carry features_dim directly
    let nch = noise_channels(&entry.config);
    let min_t = min_frames(&entry.config, 12);
    // ①c (α′): a genuine multi-speaker RVC export renames scalar `sid` to a dense `spk_mix`
    // [1, n_spk] blend (n_spk = emb_g table width = config.n_speakers). Feed the blend IFF the
    // graph actually carries that input; None → the sid path (single-speaker / pre-①c, unchanged).
    let rvc_spk_mix = if sidecar_has_input(&entry, "spk_mix") == Some(true) && entry.config.n_speakers > 0
    {
        Some(entry.config.n_speakers as usize)
    } else {
        None
    };
    let cv_path = contentvec_for_dim(&app, dim)?;
    let rmvpe_path = aux_path(&app, AUX_RMVPE, "RMVPE model")?;
    let mel_path = aux_path(&app, AUX_RMVPE_MEL, "RMVPE mel filterbank")?;

    let path = PathBuf::from(&model_path);
    // Evict every GPU session this run doesn't own (leftover MSST arena / the SoVITS
    // fleet / GPU aux extractors with their previous-run arena high-water) BEFORE
    // loading. Keep = the model itself only; a re-run reloads aux in a couple seconds
    // and the run's VRAM equals its own footprint (see release_gpu_sessions_except).
    app.inference
        .engine
        .release_gpu_sessions_except(&[path.clone()]);
    app.inference
        .load_voice(
            &voice_name,
            &path,
            VoiceBackendType::Rvc,
            entry.sample_rate,
            entry.index_path.as_ref(),
        )
        .map_err(|e| e.to_string())?;

    // gpu_extract: the per-node aux-device toggle (ContentVec + RMVPE only; the voice
    // synthesizer is on the global device regardless).
    let cv_sid = app
        .inference
        .ensure_aux_loaded_on(&cv_path, options.gpu_extract)
        .map_err(|e| e.to_string())?;
    let rmvpe_sid = app
        .inference
        .ensure_aux_loaded_on(&rmvpe_path, options.gpu_extract)
        .map_err(|e| e.to_string())?;
    let mel = app.inference.load_npy(&mel_path).map_err(|e| e.to_string())?;
    let handle = app.inference.voice_handle(&voice_name).map_err(|e| e.to_string())?;

    let audio_buf =
        crate::audio::load_audio(&PathBuf::from(&audio_path)).map_err(|e| e.to_string())?;

    // S60-2 音域扩展: the governing speaker's tested range (None = off / no sidecar record
    // ⇒ the pipeline is byte-identical to before).
    let vocal_range = if options.range_extend {
        crate::inference::vocal_range::speaker_range(
            &entry.config,
            crate::inference::dominant_speaker(&options.spk_mix, options.speaker_id),
        )
    } else {
        None
    };

    // The pipeline is minutes of CPU+GPU work — keep it off the async runtime workers.
    let progress = progress_emitter(app_handle, app.clone(), run_epoch, node_id);
    tauri::async_runtime::spawn_blocking(move || {
        let cancel = || app.inference.voice_cancelled(run_epoch);
        let model = rvc::RvcModel {
            engine: &app.inference.engine,
            voice_session: &handle.session_id,
            contentvec_session: &cv_sid,
            rmvpe_session: &rmvpe_sid,
            mel_filters: mel.as_ref(),
            index: handle.index.as_deref(),
            sample_rate: handle.sample_rate,
            features_dim: dim,
            spk_mix: rvc_spk_mix,
            noise_channels: nch,
            min_frames: min_t,
        };
        let result = rvc::run_pipeline(&model, &audio_buf, &options, vocal_range, &progress, &cancel)
            .map_err(|e| e.to_string())?;
        commit_rendered_audio(result, output_path)
    })
    .await
    .map_err(|e| format!("INFER_TASK_PANICKED: {}", e))?
}

/// Resolve the SoVITS cluster / feature-retrieval asset (`<stem>.cluster/` sibling npy) for the dominant
/// speaker — SHARED by 翻唱 (run_sovits) AND 自己唱 (render_vocal_segment). Returns None when cluster_ratio
/// ≤ 0, no asset exists, or a present file is unreadable (the blend is optional — a bad file must not abort
/// the render, matching the original's missing-file skip). ①c: under a blend the asset follows the
/// max-weight speaker; without a blend it's just `speaker_id` (fallback 0).
fn resolve_cluster_asset(
    app: &Arc<AppState>,
    entry: &crate::models::ModelEntry,
    spk_mix: &[crate::inference::SpkMixEntry],
    speaker_id: Option<u32>,
    cluster_ratio: f32,
) -> Option<sovits::ClusterAsset> {
    if cluster_ratio <= 0.0 {
        return None;
    }
    let spk = crate::inference::dominant_speaker(spk_mix, speaker_id);
    let parent = entry.path.parent().map(|p| p.to_path_buf()).unwrap_or_default();
    let stem = entry.path.file_stem().unwrap_or_default().to_string_lossy().to_string();
    let model_dir = parent.join(format!("{}.cluster", stem)); // primary probe; falls back to `parent`
    let safe = |name: &str| -> String {
        name.chars()
            .map(|c| if matches!(c, '\\' | '/' | ':' | '*' | '?' | '"' | '<' | '>' | '|') { '_' } else { c })
            .collect()
    };

    let mut found = None;
    'dirs: for dir in [&model_dir, &parent] {
        let index_path = dir.join(format!("{}.index_vectors.npy", spk));
        if index_path.exists() {
            match app.inference.load_npy(&index_path) {
                Ok(arr) => {
                    found = Some(sovits::ClusterAsset::FeatureIndex(
                        crate::inference::features::KnnIndex::new((*arr).clone()),
                    ));
                    break;
                }
                Err(e) => tracing::warn!("retrieval asset {} failed to load — skipping cluster blend: {}", index_path.display(), e),
            }
        }
        // kmeans 文件名用 speaker 名（config.speakers 反查 id）
        for (name, _) in entry.config.speakers.iter().filter(|(_, &id)| id == spk) {
            let kmeans_path = dir.join(format!("{}.centers.npy", safe(name)));
            if kmeans_path.exists() {
                match app.inference.load_npy(&kmeans_path) {
                    Ok(arr) => {
                        found = Some(sovits::ClusterAsset::KmeansCenters(
                            crate::inference::features::KnnIndex::new((*arr).clone()),
                        ));
                        break 'dirs;
                    }
                    Err(e) => tracing::warn!("cluster asset {} failed to load — skipping cluster blend: {}", kmeans_path.display(), e),
                }
            }
        }
    }
    found
}

// ─── run_sovits ──────────────────────────────────────────────────────────────

#[tauri::command]
pub async fn run_sovits(
    app_handle: tauri::AppHandle,
    state: State<'_, Arc<AppState>>,
    voice_name: String,
    model_path: String,
    audio_path: String,
    node_id: String,
    output_path: String,
    options: SovitsOptions,
) -> Result<RenderedAudio, String> {
    let mut options = options;
    let app = state.inner().clone();
    let _voice_guard = VoiceRunGuard::acquire()?; // held to the end of the render
    // See run_rvc: arm the cancel epoch before the load phase.
    let run_epoch = app.inference.begin_voice_run();
    let entry = get_entry(&app, &voice_name)?;
    require_input(&entry, "noise")?;

    let dim = features_dim(&entry.config)?;
    let nch = noise_channels(&entry.config);
    let hop_size = entry.config.hop_size.unwrap_or(512) as usize;
    if hop_size == 0 {
        return Err(format!("MODEL_HOP_SIZE_ZERO: {}", voice_name));
    }
    let min_t = min_frames(&entry.config, 6);
    // Feed vol IFF the exported graph HAS the input — the sidecar "inputs" array is the
    // authority (final contract); vol_embedding bool is the fallback for older sidecars.
    let vol_embedding = sidecar_has_input(&entry, "vol")
        .unwrap_or_else(|| entry.config.vol_embedding.unwrap_or(false));
    // 4.0-v2 (VISinger2): explicit `phase` input + NO `uv` input on the main graph —
    // both facts come from the sidecar (phase block / inputs array), never a version string.
    let v2_phase_bins = phase_bins(&entry.config);
    let v2_f0d_channels = f0d_cond_channels(&entry.config);
    let feed_uv = sidecar_has_input(&entry, "uv").unwrap_or(true);
    // ①c: a genuine multi-speaker export renames the scalar `sid` input to a dense `spk_mix`
    // [1, n_spk] blend (n_spk = emb_g table width = config.n_speakers). Feed the blend IFF the
    // graph actually carries that input; None → the sid path (single-speaker / pre-①c export).
    let spk_mix = if sidecar_has_input(&entry, "spk_mix") == Some(true) && entry.config.n_speakers > 0
    {
        Some(entry.config.n_speakers as usize)
    } else {
        None
    };
    let unit_interpolate_mode = entry
        .config
        .unit_interpolate_mode
        .clone()
        .unwrap_or_else(|| "left".to_string());

    let cv_path = contentvec_for_dim(&app, dim)?;
    let rmvpe_path = aux_path(&app, AUX_RMVPE, "RMVPE model")?;
    let mel_path = aux_path(&app, AUX_RMVPE_MEL, "RMVPE mel filterbank")?;

    let path = PathBuf::from(&model_path);
    // Evict every GPU session this run doesn't own (leftover MSST arena / the RVC
    // family / another voice / GPU aux extractors carrying the previous run's arena
    // high-water — see release_gpu_sessions_except) BEFORE the quality-path fleet
    // loads. Keep = this model's own family: consecutive re-renders of the same node
    // skip the big reloads (main graph + 220 MB denoiser). Path::starts_with is
    // COMPONENT-wise, so each companion is listed explicitly (the .diffusion dir
    // covers its contents; a bare `<stem>` would NOT cover `<stem>.f0.onnx`).
    {
        let mut keep = vec![path.clone()];
        if let (Some(dir), Some(stem)) = (path.parent(), path.file_stem()) {
            let stem = stem.to_string_lossy();
            keep.push(dir.join(format!("{}.f0.onnx", stem)));
            keep.push(dir.join(format!("{}.diffusion", stem)));
        }
        app.inference.engine.release_gpu_sessions_except(&keep);
    }
    app.inference
        .load_voice(
            &voice_name,
            &path,
            VoiceBackendType::SoVits,
            entry.sample_rate,
            None,
        )
        .map_err(|e| e.to_string())?;

    // S36 quality path: vocoder / diffusion / auto-f0 resolution + validation (also
    // enforces the original diffusion↔enhancer mutual exclusion by mutating options).
    // MUST come AFTER load_voice: a cold-start (or idle-swept) voice triggers
    // unload_voice inside load_voice, which evicts the model's companion sessions
    // (`<stem>.f0.onnx` / `.diffusion/*`) INCLUDING their reload specs — resolving the
    // companions first would hand the pipeline session ids that no longer exist
    // ("Session ... not found" on the first piece).
    let (diffusion, vocoder, f0_predictor) =
        resolve_sovits_quality(&app, &entry, dim, hop_size, &mut options, None)?;
    guard_blend_vs_diffusion(&entry, &options.spk_mix, diffusion.is_some())?;

    let cv_sid = app
        .inference
        .ensure_aux_loaded_on(&cv_path, options.gpu_extract)
        .map_err(|e| e.to_string())?;
    let rmvpe_sid = app
        .inference
        .ensure_aux_loaded_on(&rmvpe_path, options.gpu_extract)
        .map_err(|e| e.to_string())?;
    let mel = app.inference.load_npy(&mel_path).map_err(|e| e.to_string())?;
    let handle = app.inference.voice_handle(&voice_name).map_err(|e| e.to_string())?;

    // cluster 资产（converter\export_cluster.py 最终合约）：导入时落进 <stem>.cluster\ 子目录
    // （resolve_cluster_assets；多个 SoVITS 模型共用 sovits\ 目录，平铺 spk-id 会撞名）。
    //   特征检索：<speaker_id>.index_vectors.npy（spk2id 整数键，[N, dim]，优先，
    //             与原版 feature_retrieval 一致）
    //   kmeans： <speaker_name>.centers.npy（speaker 名字键，可能是中文；
    //             路径非法字符按 export_cluster 的 _safe_name 规则 →'_'，[K, dim]）
    // 兼容手动平铺在模型旁的旧摆法。
    // cluster/retrieval asset — shared with 自己唱 (render_vocal_segment) via resolve_cluster_asset.
    let cluster = resolve_cluster_asset(&app, &entry, &options.spk_mix, options.speaker_id, options.cluster_ratio);

    let audio_buf =
        crate::audio::load_audio(&PathBuf::from(&audio_path)).map_err(|e| e.to_string())?;

    let progress = progress_emitter(app_handle, app.clone(), run_epoch, node_id);
    // S60-2 音域扩展: the governing speaker's tested range (None = off / no sidecar record
    // ⇒ the pipeline is byte-identical to before).
    let vocal_range = if options.range_extend {
        crate::inference::vocal_range::speaker_range(
            &entry.config,
            crate::inference::dominant_speaker(&options.spk_mix, options.speaker_id),
        )
    } else {
        None
    };
    tauri::async_runtime::spawn_blocking(move || {
        let cancel = || app.inference.voice_cancelled(run_epoch);
        let model = sovits::SovitsModel {
            engine: &app.inference.engine,
            voice_session: &handle.session_id,
            contentvec_session: &cv_sid,
            rmvpe_session: &rmvpe_sid,
            mel_filters: mel.as_ref(),
            cluster: cluster.as_ref(),
            diffusion,
            vocoder,
            f0_predictor_session: f0_predictor,
            sample_rate: handle.sample_rate,
            hop_size,
            features_dim: dim,
            vol_embedding,
            phase_bins: v2_phase_bins,
            f0d_cond_channels: v2_f0d_channels,
            feed_uv,
            spk_mix,
            unit_interpolate_mode,
            noise_channels: nch,
            min_frames: min_t,
        };
        let result = sovits::run_pipeline(&model, &audio_buf, &options, vocal_range, &progress, &cancel)
            .map_err(|e| e.to_string())?;
        commit_rendered_audio(result, output_path)
    })
    .await
    .map_err(|e| format!("INFER_TASK_PANICKED: {}", e))?
}

// ─── detect_f0 (kept signature: audio path → f0 Hz @ 100 fps) ────────────────

#[tauri::command]
pub async fn detect_f0(
    state: State<'_, Arc<AppState>>,
    audio_path: String,
) -> Result<Vec<f32>, String> {
    let app = state.inner().clone();
    let rmvpe_path = aux_path(&app, AUX_RMVPE, "RMVPE model")?;
    let mel_path = aux_path(&app, AUX_RMVPE_MEL, "RMVPE mel filterbank")?;
    let rmvpe_sid = app.inference.ensure_aux_loaded(&rmvpe_path).map_err(|e| e.to_string())?;
    let mel = app.inference.load_npy(&mel_path).map_err(|e| e.to_string())?;

    let audio_buf =
        crate::audio::load_audio(&PathBuf::from(&audio_path)).map_err(|e| e.to_string())?;

    tauri::async_runtime::spawn_blocking(move || {
        let mono = crate::audio::resample::to_mono(&audio_buf);
        let wav16k = crate::inference::features::resample(
            &mono.samples,
            mono.sample_rate,
            crate::inference::f0::RMVPE_SR,
        );
        crate::inference::f0::rmvpe_detect(
            &app.inference.engine,
            &rmvpe_sid,
            &mel,
            &wav16k,
            crate::inference::f0::RVC_RMVPE_THRESHOLD,
        )
        .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("F0_TASK_PANICKED: {}", e))?
}

// ─── ② 自己唱 vocal render (S48 Phase 6) ─────────────────────────────────────
//
// (run_s2h + the s2h double-head module were removed in S48 Phase 1c — that pre-S35 contract was wrong
//  for ScoreToCV. The real score→cv path is inference::score2cv (build_arrays + ONNX); Phase 6 wires it
//  into a render command here.)

/// One note for `validate_lyrics` — mirrors the render's `ScoreNote` language/override semantics so the
/// editor's verdict can never drift from what actually renders (§9.5).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct LyricNote {
    pub lyric: String,
    /// Effective per-note language id (note override ?? track default; absent → `default_lang`).
    #[serde(default)]
    pub lang: Option<i64>,
    /// Traditional-phoneme override (§3.7).
    #[serde(default)]
    pub phoneme_input: Option<String>,
}

/// §9.5 single Rust classifier: classify each note's lyric (rest / breath / sustain / valid phones /
/// OOV) via the SAME `g2p::resolve` pass the render uses — language-aware (S58: per-note language,
/// zh phrase context from the NOTE SEQUENCE, western word lookup, phoneme_input overrides) with NO JS
/// dictionary copy. Double-side capped (DoS): note count + per-token char length (an over-long lyric
/// classifies Unknown without ever being looked up).
#[tauri::command]
pub async fn validate_lyrics(
    state: State<'_, Arc<AppState>>,
    notes: Vec<LyricNote>,
    default_lang: i64,
) -> Result<Vec<score2cv::LyricClass>, String> {
    // ≥ 2×MAX_SCORE_NOTES: the validation payload mirrors the render triples (notes + gap rests can
    // reach twice the note count), so the cap must never reject a segment the render itself accepts
    // (audit: a legal huge segment would otherwise silently lose its OOV marking forever).
    const MAX_TOKENS: usize = 2 * MAX_SCORE_NOTES + 1;
    const MAX_LEN: usize = 256;
    if notes.len() > MAX_TOKENS {
        return Err(format!("VOCAL_TOO_MANY_NOTES: {} > {}", notes.len(), MAX_TOKENS));
    }
    if let Some(data_dir) = state.models.models_dir().parent() {
        g2p::set_dict_dir(data_dir.join("dictionaries"));
    }
    let fallback = g2p::Lang::from_id(default_lang).unwrap_or(g2p::Lang::Ja);
    // spawn_blocking: the FIRST validation of a language lazily parses its dictionary (the en TSV is
    // ~3.7MB / 135k lines) — that one-time load must never block the IPC/main thread.
    tauri::async_runtime::spawn_blocking(move || {
        // over-long lyrics are replaced by a token that is OOV in EVERY language (never truncate — a
        // truncation could accidentally form a valid word). U+FFFD is not hanzi/kana/ascii/dict material.
        const TOO_LONG: &str = "\u{FFFD}";
        let evts: Vec<g2p::ScoreEvt> = notes
            .iter()
            .map(|n| g2p::ScoreEvt {
                lyric: if n.lyric.chars().count() > MAX_LEN { TOO_LONG } else { n.lyric.as_str() },
                note_num: 60,
                frames: 1,
                lang: n.lang.and_then(g2p::Lang::from_id).unwrap_or(fallback),
                phoneme_input: n
                    .phoneme_input
                    .as_deref()
                    .filter(|p| p.chars().count() <= MAX_LEN),
            })
            .collect();
        // Err = infrastructure (VOCAL_DICT_MISSING) — the watcher must NOT paint OOV marks for it.
        g2p::classify_score(&evts, &g2p::GlobalDicts).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("validate_lyrics task failed: {e}"))?
}

/// One note of the score from the frontend. `lyric` = the note's lyric (JA kana / rest `R` / sustain
/// `ー`); `note_num` = the RAW note MIDI (transpose is applied Rust-side, §9.3); `frames` = the note's
/// duration in 50fps frames (TimeAxis.tick_to_frame absolute diff — NEVER per-note round). Gap
/// rests/sustains are inserted by the frontend (§3.4, never inferred from note_num==0).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ScoreNote {
    pub lyric: String,
    pub note_num: i64,
    pub frames: i64,
    /// Effective per-note language id (note override ?? track default, resolved frontend-side).
    /// Absent (old callers) → the request-level `options.lang_id`. S58 §3.7.
    #[serde(default)]
    pub lang: Option<i64>,
    /// Traditional-phoneme override (§3.7 user layer: pinyin/kana/ARPABET/MFA — never raw IPA).
    #[serde(default)]
    pub phoneme_input: Option<String>,
}

/// Wire-contract options for render_vocal_segment — mirrored by src\lib\vocal\vocalRender.ts (VocalRenderOptions).
/// Item-1: the score render now drives the SAME quality path as 翻唱, so the backend-specific knobs REUSE the
/// existing `SovitsOptions`/`RvcOptions` contracts (no third source of truth). The command layer force-
/// neutralizes the params that would break the ② render (see `render_vocal_segment`).
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
pub struct VocalRenderOptions {
    /// "sovits" | "rvc".
    pub backend: String,
    /// ScoreToCV conditioning speaker (0–76; near speaker-invariant, default 49). NOT the SVC voice.
    pub cv_speaker_id: i64,
    /// ScoreToCV language id (zh0 en1 ja2 de3 fr4 es5 it6).
    pub lang_id: i64,
    /// Track-level transpose in semitones (applied to content note_pitch AND f0, Rust-side).
    pub transpose: i64,
    /// S60-2 音域扩展: when true AND the model carries a vocal_range record for the resolved
    /// speaker, out-of-comfort parts render at a minimal semitone translation into the comfort
    /// zone and are TD-PSOLA'd back (v1 affine recipe). In-range parts render EXACTLY as before
    /// (tier 1/2 = shift 0 = byte-identical), so enabling this never degrades in-range material.
    pub range_extend: bool,
    /// Reused SoVITS quality contract (backend=="sovits"): noise_scale/seed/cluster_ratio/spk_mix/speaker_id
    /// + the shallow/only-diffusion group + NSF enhancer + vocoder + gpu_extract. auto_f0/f0_shift/
    /// loudness_envelope/only_diffusion are force-neutralized by the command (they'd break Option-A / need
    /// a source wav the score doesn't have).
    pub sovits: crate::inference::SovitsOptions,
    /// Reused RVC quality contract (backend=="rvc"): index_ratio/protect/l2_normalize/noise_scale/seed/
    /// speaker_id/spk_mix/gpu_extract. f0_shift/rms_mix_rate are force-neutralized (redundant / no source wav).
    pub rvc: crate::inference::RvcOptions,
}

impl Default for VocalRenderOptions {
    fn default() -> Self {
        Self {
            backend: "sovits".into(),
            cv_speaker_id: 49,
            lang_id: 2,
            transpose: 0,
            range_extend: false,
            sovits: Default::default(),
            rvc: Default::default(),
        }
    }
}

/// Flat placeholder loudness for vol_embedding (SoVITS 4.1) models — Phase-2 validated (东雪莲 audition,
/// 用户耳审 OK). A real per-frame 响度泳道 (SegmentContent.paramCurves["loudness"]) is deferred (§10.5).
pub(crate) const VOCAL_FLAT_VOL: f32 = 0.1;
/// DoS cap on the note count of one render request.
const MAX_SCORE_NOTES: usize = 500_000;
/// DoS cap on the TOTAL 50fps frames of one render request (~200 min @50fps). Rests are UNCAPPED in the
/// DAW build (for timeline alignment) and `chunk_at_sp` can't subdivide a single rest, so a pathological
/// `frames` value would otherwise attempt a multi-TB alloc → process abort (uncatchable). Split long parts.
const MAX_TOTAL_FRAMES: i64 = 600_000;

/// Render a vocal-track notes segment → singing wav (自己唱). Mirrors run_sovits' load/guard/evict flow
/// but swaps ScoreToCV in for the audio ContentVec extractor and takes a DAW score + Option-A f0 (§10.1)
/// instead of an input wav. `score` = the per-note triples (built frontend-side incl. gap rests); when
/// `f0_cents`/`f0_voiced` are non-empty they are the whole-segment 50fps layered pitch (else bare
/// noteonly). Writes the wav to `output_path` Rust-side and returns the path (S66/O5 — the frontend
/// deposits it as a processedOutputs overlay). `node_id` = the segment id (progress routing).
#[tauri::command]
pub async fn render_vocal_segment(
    app_handle: tauri::AppHandle,
    state: State<'_, Arc<AppState>>,
    voice_name: String,
    model_path: String,
    node_id: String,
    score: Vec<ScoreNote>,
    f0_cents: Vec<f32>,
    f0_voiced: Vec<u8>,
    loudness_env: Vec<f32>,
    formant_env: Vec<f32>,
    output_path: String,
    options: VocalRenderOptions,
) -> Result<RenderedAudio, String> {
    let app = state.inner().clone();
    let _voice_guard = VoiceRunGuard::acquire()?; // held to the end of the render
    let run_epoch = app.inference.begin_voice_run();

    // ── validate the request (敌意输入边界) ──
    if score.is_empty() {
        // Reuses the frontend's existing VOCAL_EMPTY code ("no renderable notes" — same state,
        // vocalRender.ts throws it pre-flight and maps it to vocalEditor.render.empty).
        return Err("VOCAL_EMPTY".into());
    }
    if score.len() > MAX_SCORE_NOTES {
        return Err(format!("VOCAL_TOO_MANY_NOTES: {} > {}", score.len(), MAX_SCORE_NOTES));
    }
    if !f0_cents.is_empty() && f0_cents.len() != f0_voiced.len() {
        return Err(format!(
            "VOCAL_F0_LEN_MISMATCH: cents {} != voiced {}",
            f0_cents.len(),
            f0_voiced.len()
        ));
    }
    let total_frames: i64 = score.iter().map(|n| n.frames.max(0)).sum();
    if total_frames > MAX_TOTAL_FRAMES {
        return Err(format!(
            "VOCAL_SEGMENT_TOO_LONG: {} frames > {} (~{} min)",
            total_frames,
            MAX_TOTAL_FRAMES,
            MAX_TOTAL_FRAMES / 3000
        ));
    }
    // Option-A f0 is indexed by DAW frame, so its length MUST equal Σframes — a disagreement would
    // SILENTLY drift the pitch (build_note_hz clamps the index rather than crash). Reject the mismatch.
    if !f0_cents.is_empty() && f0_cents.len() as i64 != total_frames {
        return Err(format!(
            "VOCAL_F0_FRAMES_MISMATCH: {} != {}",
            f0_cents.len(),
            total_frames
        ));
    }
    // ② loudness/formant lanes are @50fps DAW-frame envelopes (like f0). A non-empty one MUST match Σframes,
    // else build_note_param would silently misalign it. Empty = no lane (flat = no-op). This is a DEFENSIVE
    // backstop, practically unreachable — buildVocalScore samples both envelopes on the SAME frameCount as f0,
    // and f0 is already length-checked above. It returns a stable English CODE (honors the S56 no-hardcoded-
    // Chinese rule); the frontend has no special toast for it (unreachable), so it falls through to the generic
    // render-failed message — acceptable for an internal invariant that can't fire on real input.
    if (!loudness_env.is_empty() && loudness_env.len() as i64 != total_frames)
        || (!formant_env.is_empty() && formant_env.len() as i64 != total_frames)
    {
        return Err("VOCAL_ENV_LEN".into());
    }
    let backend_type = match options.backend.as_str() {
        "rvc" => VoiceBackendType::Rvc,
        "sovits" => VoiceBackendType::SoVits,
        other => return Err(format!("VOCAL_BACKEND_UNKNOWN: {} (sovits / rvc)", other)),
    };

    // ── resolve the voice + ScoreToCV facts ── (Item-1: builds a REAL SovitsModel/RvcModel and drives the
    // SHARED quality path — decode_features/vc_decode — mirroring run_sovits/run_rvc's load flow.)
    let entry = get_entry(&app, &voice_name)?;
    let dim = features_dim(&entry.config)?; // 768 → SoVITS4.1/RVCv2, 256 → SoVITS4.0
    let nch = noise_channels(&entry.config);
    let sample_rate = entry.sample_rate;
    let (min_default, noise_input) = match backend_type {
        VoiceBackendType::Rvc => (12usize, "rnd"),
        VoiceBackendType::SoVits => (6usize, "noise"),
    };
    require_input(&entry, noise_input)?;
    let min_t = min_frames(&entry.config, min_default);
    // ①c: a genuine multi-speaker export renames scalar `sid` to a dense `spk_mix` [1,n_spk] blend. The
    // shared decode tail already branches spk_mix, so 自己唱 now SUPPORTS multi-speaker singers (the M1
    // hard-block is gone) — feed the blend iff the graph carries the input.
    let spk_mix = if sidecar_has_input(&entry, "spk_mix") == Some(true) && entry.config.n_speakers > 0 {
        Some(entry.config.n_speakers as usize)
    } else {
        None
    };

    let s2cv_path = score2cv_for_dim(&app, dim)?;
    let cv_path = contentvec_for_dim(&app, dim)?; // second_encoding needs it; struct requires it regardless
    let rmvpe_path = aux_path(&app, AUX_RMVPE, "RMVPE model")?; // unused by the decode tail; struct field
    let mel_path = aux_path(&app, AUX_RMVPE_MEL, "RMVPE mel filterbank")?;
    let path = PathBuf::from(&model_path);

    // S58: resolve each note's effective language up-front (LOUD on an out-of-enum id — never index
    // with a raw value) and point the g2p dictionary loader at <data>/dictionaries (lazy per language;
    // a pure-JA score never touches disk).
    let fallback_lang = g2p::Lang::from_id(options.lang_id).unwrap_or(g2p::Lang::Ja);
    let mut note_langs: Vec<g2p::Lang> = Vec::with_capacity(score.len());
    for n in &score {
        match n.lang {
            None => note_langs.push(fallback_lang),
            Some(id) => match g2p::Lang::from_id(id) {
                Some(l) => note_langs.push(l),
                None => return Err(format!("VOCAL_BAD_LANG: {}", id)),
            },
        }
    }
    if let Some(data_dir) = app.models.models_dir().parent() {
        g2p::set_dict_dir(data_dir.join("dictionaries"));
    }
    // S60-2 音域扩展: resolve the governing speaker's tested range and the v1 three-tier
    // shift. Disabled / no sidecar record / in-range ⇒ 0 ⇒ byte-identical render + no
    // inverse pass. The renders sing at transpose+shift and TD-PSOLA the audio back.
    let range_shift = if options.range_extend {
        let speaker = match backend_type {
            VoiceBackendType::SoVits => {
                crate::inference::dominant_speaker(&options.sovits.spk_mix, options.sovits.speaker_id)
            }
            VoiceBackendType::Rvc => {
                crate::inference::dominant_speaker(&options.rvc.spk_mix, options.rvc.speaker_id)
            }
        };
        match crate::inference::vocal_range::speaker_range(&entry.config, speaker) {
            Some(r) => {
                let nn: Vec<i64> = score.iter().map(|n| n.note_num).collect();
                let shift = crate::inference::vocal_range::score_pitch_bounds(
                    &nn,
                    &f0_cents,
                    &f0_voiced,
                    options.transpose,
                )
                .map(|b| crate::inference::vocal_range::compute_range_shift(b, &r))
                .unwrap_or(0);
                if shift != 0 {
                    tracing::info!(
                        "range-extend: rendering '{}' at {:+} st into comfort [{:.0},{:.0}] (speaker {})",
                        voice_name, shift, r.comfort.0, r.comfort.1, speaker
                    );
                }
                shift
            }
            None => 0,
        }
    } else {
        0
    };

    // own the score notes so the render can borrow them as &[ScoreEvt] inside spawn_blocking.
    let score_owned: Vec<ScoreNote> = score;
    let cv_speaker_id = options.cv_speaker_id;
    let transpose = options.transpose;
    let progress = progress_emitter(app_handle, app.clone(), run_epoch, node_id);

    match backend_type {
        VoiceBackendType::SoVits => {
            let hop_size = entry.config.hop_size.unwrap_or(512) as usize;
            if hop_size == 0 {
                return Err(format!("MODEL_HOP_SIZE_ZERO: {}", voice_name));
            }
            let vol_embedding = sidecar_has_input(&entry, "vol")
                .unwrap_or_else(|| entry.config.vol_embedding.unwrap_or(false));
            let v2_phase_bins = phase_bins(&entry.config);
            let v2_f0d_channels = f0d_cond_channels(&entry.config);
            let feed_uv = sidecar_has_input(&entry, "uv").unwrap_or(true);
            let unit_interpolate_mode = entry
                .config
                .unit_interpolate_mode
                .clone()
                .unwrap_or_else(|| "left".to_string());

            // §P5/P6 force-neutralize the params that would break the ② render (NOT just hidden in the UI).
            let mut sv = options.sovits.clone();
            sv.auto_f0 = false; // an f0 predictor would OVERWRITE the DAW f0 (Option-A head trap)
            sv.f0_shift = 0.0; // pitch shift is the Rust-side `transpose` (double-apply otherwise)
            sv.loudness_envelope = 1.0; // change_rms needs a source wav — the score has none
            sv.only_diffusion = false; // self-sing keeps the VITS synthesis of its own content
            sv.formant = 0.0; // the audio-NODE formant scalar is separate — the vocal editor owns its formant lane/scalar (formant_env)

            // mirror run_sovits: evict foreign GPU sessions, keep this model's own family.
            {
                let mut keep = vec![path.clone()];
                if let (Some(dir), Some(stem)) = (path.parent(), path.file_stem()) {
                    let stem = stem.to_string_lossy();
                    keep.push(dir.join(format!("{}.f0.onnx", stem)));
                    keep.push(dir.join(format!("{}.diffusion", stem)));
                }
                app.inference.engine.release_gpu_sessions_except(&keep);
            }
            app.inference
                .load_voice(&voice_name, &path, VoiceBackendType::SoVits, sample_rate, None)
                .map_err(|e| e.to_string())?;
            // resolve diffusion/vocoder AFTER load_voice; f0_predictor resolves to None (auto_f0 forced off).
            let (diffusion, vocoder, f0_predictor) =
                resolve_sovits_quality(&app, &entry, dim, hop_size, &mut sv, None)?;
            guard_blend_vs_diffusion(&entry, &sv.spk_mix, diffusion.is_some())?;
            let cv_sid = app.inference.ensure_aux_loaded_on(&cv_path, sv.gpu_extract).map_err(|e| e.to_string())?;
            let rmvpe_sid = app.inference.ensure_aux_loaded_on(&rmvpe_path, sv.gpu_extract).map_err(|e| e.to_string())?;
            let mel = app.inference.load_npy(&mel_path).map_err(|e| e.to_string())?;
            // ScoreToCV is the self-sing CONTENT workhorse (net_g already runs on the global device) and the
            // #1 render-time cost. Unlike ContentVec — whose whole-song activations peaked ~9 GB (S35) so it
            // stays pinned to CPU — ScoreToCV is chunked (sidecar chunk_max_frames ≤400), activations small.
            // So load it on the GLOBAL device (on_gpu=true = FOLLOW the device preference, NOT force GPU)
            // instead of forced-CPU: with the default Auto that probes CUDA→DirectML→CPU and FALLS BACK to
            // CPU on a GPU-less / incompatible machine (exactly like net_g), so it's fast where a GPU exists
            // and still runs CPU-only. TF32 blur is fine on this ear-validated path; cv/rmvpe keep the toggle.
            let s2cv_sid = app.inference.ensure_aux_loaded_on(&s2cv_path, true).map_err(|e| e.to_string())?;
            let handle = app.inference.voice_handle(&voice_name).map_err(|e| e.to_string())?;
            let cluster = resolve_cluster_asset(&app, &entry, &sv.spk_mix, sv.speaker_id, sv.cluster_ratio);

            tauri::async_runtime::spawn_blocking(move || {
                let cancel = || app.inference.voice_cancelled(run_epoch);
                let score_ref: Vec<g2p::ScoreEvt> = score_owned
                    .iter()
                    .zip(note_langs.iter())
                    .map(|(n, &lang)| g2p::ScoreEvt {
                        lyric: n.lyric.as_str(),
                        note_num: n.note_num,
                        frames: n.frames,
                        lang,
                        phoneme_input: n.phoneme_input.as_deref(),
                    })
                    .collect();
                let f0 = if f0_cents.is_empty() {
                    None
                } else {
                    Some(score2svc::VocalF0 { cents: f0_cents.as_slice(), voiced: f0_voiced.as_slice() })
                };
                let model = sovits::SovitsModel {
                    engine: &app.inference.engine,
                    voice_session: &handle.session_id,
                    contentvec_session: &cv_sid,
                    rmvpe_session: &rmvpe_sid,
                    mel_filters: mel.as_ref(),
                    cluster: cluster.as_ref(),
                    diffusion,
                    vocoder,
                    f0_predictor_session: f0_predictor,
                    sample_rate: handle.sample_rate,
                    hop_size,
                    features_dim: dim,
                    vol_embedding,
                    phase_bins: v2_phase_bins,
                    f0d_cond_channels: v2_f0d_channels,
                    feed_uv,
                    spk_mix,
                    unit_interpolate_mode,
                    noise_channels: nch,
                    min_frames: min_t,
                };
                let loud = if loudness_env.is_empty() { None } else { Some(loudness_env.as_slice()) };
                let formant = if formant_env.is_empty() { None } else { Some(formant_env.as_slice()) };
                let result = score2svc::render_score_sovits(
                    &model, &s2cv_sid, &score_ref, dim, cv_speaker_id, &g2p::GlobalDicts, &sv,
                    VOCAL_FLAT_VOL, transpose, range_shift, f0.as_ref(), loud, formant, &cancel, &progress,
                )
                .map_err(|e| e.to_string())?;
                commit_rendered_audio(result, output_path)
            })
            .await
            .map_err(|e| format!("VOCAL_TASK_PANICKED: {}", e))?
        }
        VoiceBackendType::Rvc => {
            // §P5 force-neutralize (redundant with transpose / no source wav — no-ops on the score path).
            let mut rv = options.rvc.clone();
            rv.f0_shift = 0.0;
            rv.rms_mix_rate = 1.0;
            rv.formant = 0.0; // audio-node formant is separate from the vocal editor's formant lane/scalar (formant_env)

            app.inference.engine.release_gpu_sessions_except(&[path.clone()]);
            app.inference
                .load_voice(&voice_name, &path, VoiceBackendType::Rvc, sample_rate, entry.index_path.as_ref())
                .map_err(|e| e.to_string())?;
            let cv_sid = app.inference.ensure_aux_loaded_on(&cv_path, rv.gpu_extract).map_err(|e| e.to_string())?;
            let rmvpe_sid = app.inference.ensure_aux_loaded_on(&rmvpe_path, rv.gpu_extract).map_err(|e| e.to_string())?;
            let mel = app.inference.load_npy(&mel_path).map_err(|e| e.to_string())?;
            // ScoreToCV on the GLOBAL device (on_gpu=true = FOLLOW the device preference; the default Auto
            // falls back CUDA→DirectML→CPU, so no GPU-less crash) instead of forced-CPU — it's the self-sing
            // content workhorse + #1 render cost, chunked so VRAM-bounded. See the SoVits arm for full rationale.
            let s2cv_sid = app.inference.ensure_aux_loaded_on(&s2cv_path, true).map_err(|e| e.to_string())?;
            let handle = app.inference.voice_handle(&voice_name).map_err(|e| e.to_string())?;

            tauri::async_runtime::spawn_blocking(move || {
                let cancel = || app.inference.voice_cancelled(run_epoch);
                let score_ref: Vec<g2p::ScoreEvt> = score_owned
                    .iter()
                    .zip(note_langs.iter())
                    .map(|(n, &lang)| g2p::ScoreEvt {
                        lyric: n.lyric.as_str(),
                        note_num: n.note_num,
                        frames: n.frames,
                        lang,
                        phoneme_input: n.phoneme_input.as_deref(),
                    })
                    .collect();
                let f0 = if f0_cents.is_empty() {
                    None
                } else {
                    Some(score2svc::VocalF0 { cents: f0_cents.as_slice(), voiced: f0_voiced.as_slice() })
                };
                let model = rvc::RvcModel {
                    engine: &app.inference.engine,
                    voice_session: &handle.session_id,
                    contentvec_session: &cv_sid,
                    rmvpe_session: &rmvpe_sid,
                    mel_filters: mel.as_ref(),
                    index: handle.index.as_deref(),
                    sample_rate: handle.sample_rate,
                    features_dim: dim,
                    spk_mix,
                    noise_channels: nch,
                    min_frames: min_t,
                };
                let loud = if loudness_env.is_empty() { None } else { Some(loudness_env.as_slice()) };
                let formant = if formant_env.is_empty() { None } else { Some(formant_env.as_slice()) };
                let result = score2svc::render_score_rvc(
                    &model, &s2cv_sid, &score_ref, dim, cv_speaker_id, &g2p::GlobalDicts, &rv,
                    transpose, range_shift, f0.as_ref(), loud, formant, &cancel, &progress,
                )
                .map_err(|e| e.to_string())?;
                commit_rendered_audio(result, output_path)
            })
            .await
            .map_err(|e| format!("VOCAL_TASK_PANICKED: {}", e))?
        }
    }
}
