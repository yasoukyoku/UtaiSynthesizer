use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;
use tauri::{Emitter, State};

use crate::inference::{
    autotune, g2p, rvc, score2cv, score2svc, sovits, RenderedAudio, RvcOptions, SovitsOptions,
    SynthesisResult, VoiceBackendType,
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
// ② 自动音高调教(旋钮线 Phase A,S73):note 特征 → per-note θ(transition/vibrato)。
// 几 MB 小 Transformer,同 aux-by-path 纪律;CPU 即可(inference/autotune.rs)。
pub(crate) const AUX_AUTOTUNE: &str = "autotune_a1.onnx";

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

/// Resolve a voice model BY NAME AND TYPE. An rvc+sovits pair sharing one singer's name is a
/// standard workflow (ModelRegistry::get_by_type's own contract: "any consumer that knows the
/// type must use this instead of the first-match `get`"), and every caller here does know it —
/// run_rvc/run_sovits statically, the score render from its `backend` option. The old untyped
/// lookup returned whichever entry the scan happened to list first, so for such a pair one of
/// the two backends would load the other's graph and die on the very next signature check with
/// `MODEL_LEGACY_EXPORT ... missing 'noise'/'rnd' input` — an error blaming the user's export
/// for a lookup bug (S81 drift audit).
pub(crate) fn get_entry(
    state: &AppState,
    voice_name: &str,
    model_type: &crate::models::ModelType,
) -> Result<ModelEntry, String> {
    state
        .models
        .get_by_type(voice_name, model_type)
        .ok_or_else(|| format!("MODEL_NOT_FOUND: {}", voice_name))
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
/// distinct id → the diffusion's single speaker matches) — and that is only TRUE because the diffusion
/// stage resolves its speaker through `dominant_speaker` like everything else. It used to read
/// `speaker_id` alone, which made this very sentence false for the one state the UI can produce
/// (blend stack set, speaker_id null): net_g sang the blend, diffusion rendered speaker 0 (S81).
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
            // Must resolve exactly like the tensor build in sovits.rs does, or the bounds check
            // validates a speaker the render never uses (S81 drift audit).
            let spk = crate::inference::dominant_speaker(&options.spk_mix, options.speaker_id);
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
    let entry = get_entry(&app, &voice_name, &crate::models::ModelType::Rvc)?;
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
    let entry = get_entry(&app, &voice_name, &crate::models::ModelType::SoVits)?;
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

/// Per-semitone VOICE-QUALITY measurement for the range test (S81 F1).
///
/// The range test's two criteria (f0 error, voiced ratio) are both derived from f0 — and f0 is
/// an explicit conditioning INPUT to net_g, so as long as that path works the model reports
/// perfect pitch and full voicing even where its TIMBRE has collapsed. Forensics on the probe
/// wavs already on disk: akiko's stored comfort ceiling (MIDI 80) reads err=2 cents /
/// voiced=1.00 while measuring 7.7 dB below the scale peak with 88.7% of its energy below
/// 1.5*f0 — a bare fundamental, no harmonics. The criteria could not have caught it; nothing in
/// the chain looked at the audio. This does.
///
/// Returns one `[rms_db, low_ratio]` per span:
///   rms_db    — level RELATIVE to the loudest span in the scale (≤ 0; the scale is
///               peak-normalised once as a whole, so cross-span comparison is meaningful);
///   low_ratio — energy below 1.5*f0 over total energy up to 8 kHz. A healthy sung vowel spreads
///               energy across harmonics (~0.1-0.5); a collapsed one is a near-pure sine (~0.9+).
/// Spans are 100 fps frame windows (the same grid classifySemitones uses); only the STEADY
/// second half of each is measured, so onsets and the phrase-envelope ramp cannot flatter it.
#[tauri::command]
pub async fn analyze_scale_quality(
    audio_path: String,
    spans: Vec<(usize, usize)>,
    expected_hz: Vec<f32>,
) -> Result<Vec<(f32, f32)>, String> {
    if spans.len() != expected_hz.len() {
        return Err("SCALE_QUALITY_SHAPE".to_string());
    }
    let buf = crate::audio::load_audio(&PathBuf::from(&audio_path)).map_err(|e| e.to_string())?;
    tauri::async_runtime::spawn_blocking(move || {
        let mono = crate::audio::resample::to_mono(&buf);
        let sr = mono.sample_rate.max(1) as f32;
        let x = &mono.samples;
        const NFFT: usize = 2048;
        let mut planner = rustfft::FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(NFFT);
        let win: Vec<f32> = (0..NFFT)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / NFFT as f32).cos())
            .collect();

        let mut out: Vec<(f32, f32)> = Vec::with_capacity(spans.len());
        for (i, &(a, b)) in spans.iter().enumerate() {
            // steady state only: the back half of the note, past the attack and the ADSR ramp
            let mid = a + (b.saturating_sub(a)) / 2;
            let s0 = ((mid as f32 / 100.0) * sr) as usize;
            let s1 = (((b as f32) / 100.0) * sr) as usize;
            let (s0, s1) = (s0.min(x.len()), s1.min(x.len()));
            if s1 <= s0 {
                out.push((-120.0, 1.0)); // nothing there = treat as fully collapsed
                continue;
            }
            let seg = &x[s0..s1];
            let rms = (seg.iter().map(|v| v * v).sum::<f32>() / seg.len() as f32).sqrt();

            // one centred FFT frame is enough — the note is a steady sustained vowel here
            let mut frame = vec![rustfft::num_complex::Complex::<f32>::new(0.0, 0.0); NFFT];
            let c0 = s0 + seg.len() / 2;
            let start = c0.saturating_sub(NFFT / 2);
            for k in 0..NFFT {
                let v = x.get(start + k).copied().unwrap_or(0.0);
                frame[k] = rustfft::num_complex::Complex::new(v * win[k], 0.0);
            }
            fft.process(&mut frame);
            let bin_hz = sr / NFFT as f32;
            let cut = (1.5 * expected_hz[i] / bin_hz).round() as usize;
            let top = ((8000.0 / bin_hz) as usize).min(NFFT / 2);
            let mut low = 0f32;
            let mut total = 0f32;
            for (k, c) in frame.iter().enumerate().take(top).skip(1) {
                let e = c.norm_sqr();
                total += e;
                if k <= cut {
                    low += e;
                }
            }
            out.push((rms, if total > 0.0 { low / total } else { 1.0 }));
        }
        // level is only meaningful RELATIVE to this scale's own loudest note
        let peak = out.iter().map(|&(r, _)| r).fold(0.0f32, f32::max).max(1e-9);
        Ok(out
            .into_iter()
            .map(|(r, lr)| (20.0 * (r / peak).max(1e-6).log10(), lr))
            .collect())
    })
    .await
    .map_err(|e| e.to_string())?
}

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

/// S109 (§G13-M2) — the ONE text-length bound the editor-side commands apply, hoisted out of the two
/// local `const MAX_LEN: usize = 256` that used to declare it twice with the same literal.
///
/// ⚠ IT IS APPLIED ASYMMETRICALLY, AND THAT ASYMMETRY IS ONLY SAFE BECAUSE OF A CONSTANT IN ANOTHER
/// LANGUAGE. Read this before touching either side:
///   · an over-long LYRIC is a loud error here (`VOCAL_LYRIC_TOO_LONG`, `preview_vocal_phonemes`);
///   · an over-long `phoneme_input` is SILENTLY dropped (`Option::filter` → `None`, both editor
///     commands), so the editor would preview a note WITHOUT the override;
///   · `render_vocal_segment` applies no text bound at all, so it would sing WITH the override.
/// ⇒ On the same note the editor and the render would disagree — the exact editor-vs-render fork
/// this file's `phoneme_set` parameter exists to prevent (see `validate_lyrics` below).
///
/// It is unreachable today for a reason that lives in `src/lib/vocalNotes.ts`: `MAX_LYRIC_LEN = 64`,
/// and `sanitizeText` slices `phonemeInput` to it on EVERY ingress (store edits, `.usp` load, UST/
/// USTX import), so no `phoneme_input` over 64 chars can exist in a project. 64 < 256, therefore the
/// filter has never fired. `phoneme_input_bound_is_unreachable_from_the_editor` (tests below) pins
/// that inequality by READING the TypeScript file, so raising `MAX_LYRIC_LEN` — or shipping the
/// backlogged SV-style phoneme editor with its own larger cap — goes red HERE rather than shipping a
/// note that previews one thing and sings another.
/// (S88's rule: when A is used as a proxy for B, write down why A ⇒ B, and re-read it before adding
/// a new member to the domain. The proxy here is "the frontend caps shorter than we do".)
pub(crate) const MAX_LYRIC_CHARS: usize = 256;

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
    // S91: the editor's red marks and the render MUST judge a lyric under the same convention —
    // that is the whole reason both go through `resolve_core`. Without this the marking pass would
    // read `&m` as a word (OOV) while the render sings it, or vice versa.
    phoneme_set: Option<String>,
) -> Result<Vec<score2cv::LyricClass>, String> {
    // ≥ 2×MAX_SCORE_NOTES: the validation payload mirrors the render triples (notes + gap rests can
    // reach twice the note count), so the cap must never reject a segment the render itself accepts
    // (audit: a legal huge segment would otherwise silently lose its OOV marking forever).
    const MAX_TOKENS: usize = 2 * MAX_SCORE_NOTES + 1;
    if notes.len() > MAX_TOKENS {
        return Err(format!("VOCAL_TOO_MANY_NOTES: {} > {}", notes.len(), MAX_TOKENS));
    }
    if let Some(data_dir) = state.models.models_dir().parent() {
        g2p::set_dict_dir(data_dir.join("dictionaries"));
    }
    let fallback = g2p::Lang::from_id(default_lang).unwrap_or(g2p::Lang::Ja);
    let phoneme_set = crate::inference::g2p_alias::PhonemeSet::from_wire(phoneme_set.as_deref());
    // spawn_blocking: the FIRST validation of a language lazily parses its dictionary (the en TSV is
    // ~3.7MB / 135k lines) — that one-time load must never block the IPC/main thread.
    tauri::async_runtime::spawn_blocking(move || {
        // over-long lyrics are replaced by a token that is OOV in EVERY language (never truncate — a
        // truncation could accidentally form a valid word). U+FFFD is not hanzi/kana/ascii/dict material.
        const TOO_LONG: &str = "\u{FFFD}";
        let evts: Vec<g2p::ScoreEvt> = notes
            .iter()
            .map(|n| g2p::ScoreEvt {
                lyric: if n.lyric.chars().count() > MAX_LYRIC_CHARS { TOO_LONG } else { n.lyric.as_str() },
                note_num: 60,
                frames: 1,
                lang: n.lang.and_then(g2p::Lang::from_id).unwrap_or(fallback),
                phoneme_input: n
                    .phoneme_input
                    .as_deref()
                    .filter(|p| p.chars().count() <= MAX_LYRIC_CHARS),
                phoneme_set,
            })
            .collect();
        // Err = infrastructure (VOCAL_DICT_MISSING) — the watcher must NOT paint OOV marks for it.
        g2p::classify_score(&evts, &g2p::GlobalDicts).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("validate_lyrics task failed: {e}"))?
}

/// S83 phoneme-lane preview: one emitted phone of the DAW assembly, wire-shaped for the editor's
/// read-only phoneme lane. `evt` = the index of the input score triple this phone came from (the
/// frontend maps it back to a note id / gap rest through its parallel `tripleNoteIds`); frames are
/// consumed cumulatively (Σ frames == Σ triple frames — the assembler is frame-conserving, on BOTH
/// articulation-timing arms), so the lane's x-positions are exactly the render's, borrowed pre-beat
/// onsets included — and with the pre-roll switched off, exactly the in-note layout instead.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PhonemeSpan {
    pub phone: String,
    pub frames: i64,
    pub evt: usize,
    pub voiceless: bool,
    pub nucleus: bool,
}

/// S83: dry-run the ② DAW assembly (`build_arrays_daw` — THE render's allocator, single source) and
/// return the emitted phone spans. No model, no audio — just G2P + frame allocation, so the editor's
/// phoneme lane always shows exactly what a render would sing (incl. onset pre-roll and dropped codas).
/// Errors (OOV / missing dictionary) surface as the same stable CODEs the render uses; the lane simply
/// clears (the note area already paints OOV red via oovWatch).
#[tauri::command]
pub async fn preview_vocal_phonemes(
    state: State<'_, Arc<AppState>>,
    score: Vec<ScoreNote>,
    default_lang: i64,
    // S89: the lane is the ONLY place the articulation-timing switch is visible before a render, so
    // it must receive the track's setting. `Option` (not a bare bool) keeps an older frontend — or any
    // caller that simply doesn't know — on the production default instead of silently previewing the
    // OTHER arm; a missing field on a bool would have deserialized to `false` = the wrong picture.
    consonant_preroll: Option<bool>,
    // S91: same rationale — the lane's contract is "exactly what a render would sing", so it must
    // read the track's alias convention or it would show the dictionary's phones for an alias score.
    phoneme_set: Option<String>,
) -> Result<Vec<PhonemeSpan>, String> {
    // Same 1× cap as render_vocal_segment: the payload here is byte-for-byte the SAME triples array the
    // render caps at MAX_SCORE_NOTES — an over-cap score can never render, so previewing it has no value
    // (the lane's contract is "show what a render would sing"). S83 review #11.
    if score.len() > MAX_SCORE_NOTES {
        return Err(format!("VOCAL_TOO_MANY_NOTES: {} > {}", score.len(), MAX_SCORE_NOTES));
    }
    let total_frames: i64 = score.iter().map(|n| n.frames.max(0)).sum();
    if total_frames > MAX_TOTAL_FRAMES {
        return Err(format!("VOCAL_SEGMENT_TOO_LONG: {} frames > {}", total_frames, MAX_TOTAL_FRAMES));
    }
    // DoS bound on the watcher-driven surface (same bound as the editor commands — S109 hoisted it to
    // `MAX_LYRIC_CHARS`, read its doc before touching either side). LOUD error instead of the U+FFFD
    // swap: build_arrays_daw is strict (whole-score error), so a swap would just rename the failure —
    // and a >256-char lyric is pathological (the editor caps at 64). Review #10.
    if score.iter().any(|n| n.lyric.chars().count() > MAX_LYRIC_CHARS) {
        return Err("VOCAL_LYRIC_TOO_LONG".into());
    }
    if let Some(data_dir) = state.models.models_dir().parent() {
        g2p::set_dict_dir(data_dir.join("dictionaries"));
    }
    let fallback = g2p::Lang::from_id(default_lang).unwrap_or(g2p::Lang::Ja);
    let timing = crate::inference::score2svc::ScoreShaping {
        consonant_preroll: consonant_preroll.unwrap_or(true),
        ..Default::default()
    }
    .articulation_timing(); // ONE bool→enum conversion, shared with the render
    let phoneme_set = crate::inference::g2p_alias::PhonemeSet::from_wire(phoneme_set.as_deref());
    tauri::async_runtime::spawn_blocking(move || {
        let mut evts: Vec<g2p::ScoreEvt> = Vec::with_capacity(score.len());
        for n in &score {
            evts.push(g2p::ScoreEvt {
                lyric: n.lyric.as_str(),
                note_num: n.note_num,
                frames: n.frames,
                lang: n.lang.and_then(g2p::Lang::from_id).unwrap_or(fallback),
                phoneme_input: n.phoneme_input.as_deref().filter(|p| p.chars().count() <= MAX_LYRIC_CHARS),
                phoneme_set,
            });
        }
        let arr =
            score2cv::build_arrays_daw(&evts, &g2p::GlobalDicts, timing).map_err(|e| e.to_string())?;
        Ok(arr
            .phon
            .iter()
            .enumerate()
            .map(|(i, &p)| PhonemeSpan {
                phone: p.to_string(),
                frames: arr.phone_dur[i],
                evt: arr.evt[i],
                voiceless: score2cv::is_voiceless_phone(p),
                nucleus: score2cv::is_nucleus_phone(p),
            })
            .collect())
    })
    .await
    .map_err(|e| format!("preview_vocal_phonemes task failed: {e}"))?
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

/// The note-pitch list the RANGE-EXTEND dead-zone planner reads, with SILENT tokens zeroed — exactly the
/// `npitch` rule `score2svc::compute_note_groups` uses, from the same single predicate.
///
/// ★ Why this cannot be `|n| n.note_num`: `vocal_range::dead_only_plan` and `dead_group_windows` detect a
/// rest by `note_num <= 0` (that is how a phrase — the unit a rescue shift is decided over — is delimited).
/// A gap rest always arrives as 0, but a rest or breath written as a NOTE carries whatever pitch it was
/// drawn on, so a silence read as a sung note: it welds two phrases into ONE scan window (the shift is then
/// chosen against notes that belong to a different phrase) and can itself be judged "dead" and drag a
/// transposition onto material that makes no sound. Silent-token notes are the only inputs this changes;
/// every score without one is bit-identical, which is why S88's timing-version bump is enough to invalidate
/// the affected bakes. (S86 extracted `is_silent_token` for precisely this class of consumer drift — this
/// call site was the last one still bypassing it.)
/// S91: takes the track's convention because `is_silent_token` now depends on it (lowercase `ap` is
/// a breath on a words track and a sung VC alias on an alias track). Passing the wrong one here would
/// hand the range planner a pitch for a silent note, or zero for a sung one — the S88 bug this
/// function was written to fix, in the other direction.
fn plan_note_nums(score: &[ScoreNote], set: crate::inference::g2p_alias::PhonemeSet) -> Vec<i64> {
    score
        .iter()
        .map(|n| {
            if crate::inference::g2p::is_silent_token(&n.lyric, set) {
                0
            } else {
                n.note_num
            }
        })
        .collect()
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
    /// zone and are shifted back (Signalsmith inverse). In-range parts render EXACTLY as before
    /// (tier 1/2 = shift 0 = byte-identical), so enabling this never degrades in-range material.
    pub range_extend: bool,
    /// S83 knife 6b: voiceless-ONSET emphasis in dB (the SynthV "consonant strength" analogue).
    /// 0 = off (exact no-op); clamped to [0, 12] render-side. Absent (old frontends) → the default.
    pub consonant_emphasis_db: f32,
    /// S84 C 刀: chain-internal consonant-valley scale (×measured per-class depth; the fast-run
    /// 粘连 treatment). 0 = off (exact no-op); clamped to [0, 2]. Absent (old frontends) → default.
    pub consonant_valley: f32,
    /// S84 E 刀: vowel-clarity articulation oversampling — short nuclei (≤4 frames) render at an
    /// inflated S2CV duration and resample back, so fast-run vowels reach their articulation
    /// target (「渲染长音素再缩短」, cv-domain). Absent → true (the production default).
    pub vowel_clarity: bool,
    /// S89 「自动音素时序」: onset consonants are pre-rolled before the beat (S83 crown knife) so the
    /// nucleus lands ON the beat. `false` keeps every phone inside its own note — for UTAU CVVC/VCCV
    /// alias scores, whose author already moved the consonants ahead by hand, pre-rolling would apply
    /// that head start twice. Absent (old frontends) → true (the production default).
    pub consonant_preroll: bool,
    /// S91 「音素约定」: which UTAU alias convention this track's ENGLISH lyrics are written in —
    /// `"arpasing"` | `"xsampa"` | `"vccv"`. Absent/unknown → words through the dictionary (the
    /// default, byte-for-byte the pre-S91 behaviour). A `String` rather than the enum on purpose: an
    /// unknown value from a newer frontend must land on the default, not fail the whole render.
    pub phoneme_set: Option<String>,
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
            consonant_emphasis_db: crate::inference::score2svc::DEFAULT_VOICELESS_ONSET_EMPHASIS_DB,
            consonant_valley: crate::inference::score2svc::DEFAULT_CONSONANT_VALLEY_SCALE,
            vowel_clarity: true,
            consonant_preroll: true,
            phoneme_set: None,
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
// ─── ② 自动音高调教(旋钮线 Phase A,S73) ───────────────────────────────────────────────────

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AutotuneNoteIn {
    pub start_ms: f64,
    pub dur_ms: f64,
    /// float MIDI = 书写音高 + detune/100(训练侧 GAME 含-cents 口径同构)。
    pub pitch: f64,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AutotuneTransitionOut {
    pub offset_ms: f64,
    pub dur_left_ms: f64,
    pub dur_right_ms: f64,
    pub depth_left_cents: f64,
    pub depth_right_cents: f64,
    pub open_edge_cents: f64,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AutotuneVibratoOut {
    pub depth_cents: f64,
    pub freq_hz: f64,
    pub phase: f64,
    pub start_ms: f64,
    pub ease_in_ms: f64,
    pub ease_out_ms: f64,
}

#[derive(serde::Serialize)]
pub struct AutotuneThetaOut {
    pub transition: AutotuneTransitionOut,
    pub vibrato: AutotuneVibratoOut,
}

pub(crate) const MAX_AUTOTUNE_NOTES: usize = 100_000;

/// ② 自动音高调教:音符序列(ms 域,升序)→ per-note θ。整段音符都要传(模型吃乐句
/// 上下文);θ 的应用范围/所有权(用户调教 vs 机器调教)、Expressiveness 缩放、retake
/// 随机 phase 都是 TS 侧的事(src/lib/vocal/autoTune.ts)。
#[tauri::command]
pub async fn run_autotune(
    state: State<'_, Arc<AppState>>,
    notes: Vec<AutotuneNoteIn>,
) -> Result<Vec<AutotuneThetaOut>, String> {
    if notes.is_empty() {
        return Err("AUTOTUNE_EMPTY".into());
    }
    if notes.len() > MAX_AUTOTUNE_NOTES {
        return Err(format!(
            "AUTOTUNE_TOO_MANY_NOTES: {} > {}",
            notes.len(),
            MAX_AUTOTUNE_NOTES
        ));
    }
    let mut prev_start = f64::NEG_INFINITY;
    for n in &notes {
        if !(n.start_ms.is_finite() && n.dur_ms.is_finite() && n.pitch.is_finite())
            || n.dur_ms <= 0.0
        {
            return Err("AUTOTUNE_BAD_NOTE".into());
        }
        if n.start_ms < prev_start {
            return Err("AUTOTUNE_UNSORTED".into());
        }
        prev_start = n.start_ms;
    }
    let path = aux_path(&state, AUX_AUTOTUNE, "Autotune model")?;
    let app = state.inner().clone();
    let sid = app
        .inference
        .ensure_aux_loaded(&path)
        .map_err(|e| e.to_string())?;
    let ins: Vec<autotune::NoteIn> = notes
        .iter()
        .map(|n| autotune::NoteIn { start_ms: n.start_ms, dur_ms: n.dur_ms, pitch: n.pitch })
        .collect();
    let thetas = tauri::async_runtime::spawn_blocking(move || {
        autotune::run_autotune_model(&app.inference.engine, &sid, &ins)
    })
    .await
    .map_err(|e| format!("AUTOTUNE_JOIN: {e}"))?
    .map_err(|e| e.to_string())?;
    Ok(thetas
        .into_iter()
        .map(|t| AutotuneThetaOut {
            transition: AutotuneTransitionOut {
                offset_ms: t.transition[0],
                dur_left_ms: t.transition[1],
                dur_right_ms: t.transition[2],
                depth_left_cents: t.transition[3],
                depth_right_cents: t.transition[4],
                open_edge_cents: t.transition[5],
            },
            vibrato: AutotuneVibratoOut {
                depth_cents: t.vibrato[0],
                freq_hz: t.vibrato[1],
                phase: t.vibrato[2],
                start_ms: t.vibrato[3],
                ease_in_ms: t.vibrato[4],
                ease_out_ms: t.vibrato[5],
            },
        })
        .collect())
}

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
    let entry = get_entry(
        &app,
        &voice_name,
        match backend_type {
            VoiceBackendType::Rvc => &crate::models::ModelType::Rvc,
            VoiceBackendType::SoVits => &crate::models::ModelType::SoVits,
        },
    )?;
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
    // S91: ONE place converts the wire string to the enum, and the track's single setting is fanned
    // out over every note below (a score never mixes conventions — see `ScoreEvt::phoneme_set`).
    let phoneme_set = crate::inference::g2p_alias::PhonemeSet::from_wire(options.phoneme_set.as_deref());
    // S60-2 → S85 音域扩展(score):整曲平移废除(三轮耳判「开了不如不开」——救 1.7% 极端音
    // 却让其余音符各自去赌 per-(音素×落点) 渲染死区 + 随深度增长的往返税;memory S85)。
    // dead-only:仅含「真死音」(记录 f0 判据连音高都发不出)的休止分界短语,以最小深度渲染到
    // 最近可唱槽再逆变换回写谱位;其余音符与关扩展逐位一致(输出音高恒=写谱音高,乐谱内容是
    // 底线)。cover/audition 维持整段优化器不变(无音符结构;S82 耳判过的工况)。
    let range_windows: Vec<crate::inference::vocal_range::DeadJob> = if options.range_extend {
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
                let nn = plan_note_nums(&score, phoneme_set);
                let fr: Vec<i64> = score.iter().map(|n| n.frames).collect();
                let (plan, unfixable) =
                    crate::inference::vocal_range::dead_only_plan(&nn, options.transpose, &r);
                // 审计恒打印(S83 承诺):无死音也是一个判决;无解组必须响亮(warn+位置,
                // 事后取证要「在哪」不只是「几个」——审查 S85)。
                if plan.is_empty() && unfixable.is_empty() {
                    tracing::info!(
                        "range-extend(score/dead-only): no dead notes for '{}' (usable [{:.0},{:.0}], speaker {}) — rendering untouched",
                        voice_name, r.usable.0, r.usable.1, speaker
                    );
                } else {
                    tracing::info!(
                        "range-extend(score/dead-only): '{}' — {} dead group(s), {} unfixable (usable [{:.0},{:.0}], speaker {})",
                        voice_name, plan.len(), unfixable.len(), r.usable.0, r.usable.1, speaker
                    );
                    // The verdict is taken on `eff = written + transpose`, so printing the WRITTEN
                    // MIDI alone put two coordinate systems in one line and made the log
                    // unreconstructable after the fact whenever transpose != 0 (S145). Print both.
                    let tp = options.transpose;
                    for g in &plan {
                        let hi = (g.start..=g.end).map(|k| nn[k]).max().unwrap_or(0);
                        tracing::info!(
                            "range-extend(score/dead-only):   group notes[{}..={}] (top MIDI {hi} written, {} effective @ transpose {:+}) renders at {:+} st",
                            g.start, g.end, hi + tp, tp, g.shift
                        );
                    }
                    for &(a, b) in &unfixable {
                        let hi = (a..=b).map(|k| nn[k]).max().unwrap_or(0);
                        tracing::warn!(
                            "range-extend(score/dead-only):   notes[{a}..={b}] (top MIDI {hi} written, {} effective @ transpose {:+}) has NO landing within ±24 st — rendered broken as written",
                            hi + tp, tp
                        );
                    }
                }
                crate::inference::vocal_range::dead_group_windows(&nn, &fr, &plan)
            }
            None => {
                // 「没记录所以没做」必须与「做了且无死音」可区分(审查 S85)。
                tracing::info!(
                    "range-extend(score/dead-only): '{}' has no usable vocal-range record for speaker {} — extension inactive",
                    voice_name, speaker
                );
                Vec::new()
            }
        }
    } else {
        Vec::new()
    };
    // dead-only donor 补渲的进度量纲:base 占 1/(1+K),每个 distinct-shift donor 各占一份
    // (审查 S85:进度条曾在 100% 停住继续渲 K 遍)。无 donor 时 =1,基线量纲逐位不变。
    let range_passes = 1 + {
        let mut s: Vec<i64> = range_windows.iter().map(|w| w.shift).collect();
        s.sort_unstable();
        s.dedup();
        s.len()
    };

    // own the score notes so the render can borrow them as &[ScoreEvt] inside spawn_blocking.
    let score_owned: Vec<ScoreNote> = score;
    let cv_speaker_id = options.cv_speaker_id;
    let transpose = options.transpose;
    // knob hygiene, all in ONE place: non-finite → default; the render treats ≤0 as an exact no-op.
    let shaping = crate::inference::score2svc::ScoreShaping {
        consonant_emphasis_db: if options.consonant_emphasis_db.is_finite() {
            options.consonant_emphasis_db.clamp(0.0, 12.0) // cap 12 dB
        } else {
            crate::inference::score2svc::DEFAULT_VOICELESS_ONSET_EMPHASIS_DB
        },
        // S84 C 刀 knob hygiene: same policy, scale capped [0, 2] (render skips the stage at ≤0).
        consonant_valley_scale: if options.consonant_valley.is_finite() {
            options.consonant_valley.clamp(0.0, 2.0)
        } else {
            crate::inference::score2svc::DEFAULT_CONSONANT_VALLEY_SCALE
        },
        vowel_clarity: options.vowel_clarity, // S84 E 刀 toggle (bool — nothing to sanitize)
        consonant_preroll: options.consonant_preroll, // S89 toggle (bool — nothing to sanitize)
    };
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
            // ScoreToCV is the self-sing CONTENT workhorse (net_g already runs on the global device).
            // ⛔ S147 corrects this comment: it used to claim ScoreToCV was "the #1 render-time cost".
            // **Measured, production caliber (enhancer on): s2cv 4.1%, net_g 54.7%, vocoder 41.3%,
            // ALL CPU-side work 1.1%.** net_g alone is 13-20× ScoreToCV (24-28× on the CPU EP), and the
            // `[perf]` line in score2svc.rs now prints the split on every render so this cannot go stale
            // again. ⚠ The DECISION below is still right — the number behind it was not.
            // ⚠ And do NOT read this as "s2cv could move back to CPU": measured CUDA 2.0-2.5s vs CPU
            // 4.0-7.2s, i.e. +2~5s per pass (×(1+K)), and it would change the output (TF32 vs fp32 on an
            // ear-validated path). Unlike ContentVec — whose whole-song activations peaked ~9 GB (S35) so it
            // stays pinned to CPU — ScoreToCV is chunked (sidecar chunk_max_frames ≤400 as a SOFT cut at
            // SPs; rest-less passages exceed it, and the S84 vowel-clarity twin inflates fast-run chunks
            // ~1.3-1.5× — still small next to ContentVec, hard-bounded by the twin's 8000-frame fallback).
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
                        phoneme_set,
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
                let base_progress = |p: f32| progress(p / range_passes as f32);
                let mut result = score2svc::render_score_sovits(
                    &model, &s2cv_sid, &score_ref, dim, cv_speaker_id, &g2p::GlobalDicts, &sv,
                    VOCAL_FLAT_VOL, shaping, transpose, 0, f0.as_ref(), loud, formant, &cancel, &base_progress, None,
                )
                .map_err(|e| e.to_string())?;
                // S85 dead-only: donor 全曲渲染在 range_shift=s(函数内部逆变换回写谱位 +
                // peak-norm,与 base 同构 = 三轮耳判验证过的拼接口径),短语窗交叉淡化贴回;
                // 每个 donor 占 1/(1+K) 进度区间(审查 S85)。
                if !range_windows.is_empty() {
                    let sr = result.sample_rate;
                    // `.max(0)` matches the other two call sites (:1391, :1720). A negative
                    // `frames` would otherwise shorten the total and slide every splice window
                    // left of where the decision layer put it — silently, and only on the
                    // rescued phrases (S145 spotted the divergence).
                    let total_frames: i64 = score_ref.iter().map(|n| n.frames.max(0)).sum();
                    let pass = std::cell::Cell::new(0usize);
                    // S147: donor 与 base **共用 base 的归一前峰** ⇒ 两边乘同一个标量,
                    // 于是 `match_levels` 那个「用全曲 active-RMS 比值把台阶猜回来」的启发式
                    // 整个不需要了 ⇒ 传 **false**。
                    // ⛔ 为什么不是「把 RMS 的统计区域对齐」:两种对齐法都实测判死 ——
                    // 窗内对齐时 base 在救援窗里**正是那段坏渲染**(shift −7 读 −4.7 dB);
                    // 保留区对齐 pooled mean −0.829 dB,而且**对完整 donor 一样坏**。
                    // ⛔ 也不是「donor 干脆不归一」:那会把 `clamp(0.25,4.0)` 这个 ±12 dB 安全笼
                    // 变成**承重件**(实测 dxl41 g=2.69,离上限只剩 3.5 dB),而响度泳道
                    // (`apply_gain_env`,合法量程 ±12 dB)是第二个绝对电平搬运工。
                    // ⚠ 这一笔**会改今天的输出**:逐 shift 一个常数(实测 −0.114/+0.064/−0.280/
                    // −0.056 dB),低于 ~1 dB 的电平 JND 但高于逐 chunk 电平地板 0.004 dB 二十倍。
                    let base_peak = result.pre_norm_peak;
                    crate::inference::vocal_range::apply_dead_only_windows(&mut result.audio, sr, total_frames, &range_windows, false, |s| {
                        pass.set(pass.get() + 1);
                        let off = pass.get() as f32;
                        let donor_progress = |p: f32| progress((off + p) / range_passes as f32);
                        score2svc::render_score_sovits(
                            &model, &s2cv_sid, &score_ref, dim, cv_speaker_id, &g2p::GlobalDicts, &sv,
                            VOCAL_FLAT_VOL, shaping, transpose, s, f0.as_ref(), loud, formant, &cancel,
                            &donor_progress, base_peak,
                        )
                        .map(|r| r.audio)
                    })
                    .map_err(|e| e.to_string())?;
                }
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
            // content workhorse, chunked so VRAM-bounded. See the SoVits arm for full rationale.
            // ⛔ S147: the "#1 render cost" claim that used to be here is removed. It was never measured on
            // THIS arm — the RVC render path has no per-stage timing at all (the `[perf]` line lives in the
            // sovits arm of score2svc.rs), and on the sovits arm the same claim measured 4.1%. Do not
            // reinstate it, on either arm, without a reading.
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
                        phoneme_set,
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
                let base_progress = |p: f32| progress(p / range_passes as f32);
                let mut result = score2svc::render_score_rvc(
                    &model, &s2cv_sid, &score_ref, dim, cv_speaker_id, &g2p::GlobalDicts, &rv,
                    shaping, transpose, 0, f0.as_ref(), loud, formant, &cancel, &base_progress, None,
                )
                .map_err(|e| e.to_string())?;
                // S85 dead-only(镜像 SoVits 臂,机理注释见彼处)。
                if !range_windows.is_empty() {
                    let sr = result.sample_rate;
                    // `.max(0)` matches the other two call sites (:1391, :1720). A negative
                    // `frames` would otherwise shorten the total and slide every splice window
                    // left of where the decision layer put it — silently, and only on the
                    // rescued phrases (S145 spotted the divergence).
                    let total_frames: i64 = score_ref.iter().map(|n| n.frames.max(0)).sum();
                    let pass = std::cell::Cell::new(0usize);
                    // S147:与 SoVits 臂同一口径(机理注释见彼处)—— donor 共用 base 的归一前峰,
                    // `match_levels` 因此不再需要。
                    let base_peak = result.pre_norm_peak;
                    crate::inference::vocal_range::apply_dead_only_windows(&mut result.audio, sr, total_frames, &range_windows, false, |s| {
                        pass.set(pass.get() + 1);
                        let off = pass.get() as f32;
                        let donor_progress = |p: f32| progress((off + p) / range_passes as f32);
                        score2svc::render_score_rvc(
                            &model, &s2cv_sid, &score_ref, dim, cv_speaker_id, &g2p::GlobalDicts, &rv,
                            shaping, transpose, s, f0.as_ref(), loud, formant, &cancel,
                            &donor_progress, base_peak,
                        )
                        .map(|r| r.audio)
                    })
                    .map_err(|e| e.to_string())?;
                }
                commit_rendered_audio(result, output_path)
            })
            .await
            .map_err(|e| format!("VOCAL_TASK_PANICKED: {}", e))?
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn note(lyric: &str, note_num: i64) -> ScoreNote {
        ScoreNote { lyric: lyric.to_string(), note_num, frames: 10, lang: None, phoneme_input: None }
    }

    /// A rest/breath written as a NOTE keeps whatever pitch it was drawn on. The dead-zone planner
    /// delimits phrases by `note_num <= 0`, so that pitch must not reach it (see `plan_note_nums`).
    #[test]
    fn plan_note_nums_zeroes_silent_tokens() {
        let score = [
            note("R", 71),   // a rest drawn high on the staff
            note("か", 60),
            note("AP", 84),  // a breath drawn even higher
            note("き", 62),
            note("r", 0),
            note("", 55),    // an empty lyric is a rest too (g2p token_class)
        ];
        assert_eq!(plan_note_nums(&score, crate::inference::g2p_alias::PhonemeSet::Words), vec![0, 60, 0, 62, 0, 0]);
    }

    /// The words S86 deliberately freed are ORDINARY lyrics — zeroing one here would silently drop a sung
    /// note out of its own phrase (and out of the range decision made for it).
    #[test]
    fn plan_note_nums_keeps_freed_words_and_sustains() {
        let score = [note("rest", 60), note("sil", 61), note("pau", 62), note("-", 63), note("+", 64)];
        assert_eq!(plan_note_nums(&score, crate::inference::g2p_alias::PhonemeSet::Words), vec![60, 61, 62, 63, 64]);
    }

    /// The phrase-delimiting consequence, stated as BEHAVIOUR — the planner is actually run, because
    /// comparing `plan_note_nums` with itself would be equal by construction and green under any mutation
    /// that changes both sides (S88 review caught exactly that).
    ///
    /// usable 48..79 ⇒ か(60) sings and き(85) is dead. The rest between them is DRAWN at 71, i.e. INSIDE
    /// the usable band, so if its pitch reaches the planner it reads as a healthy sung note and welds the
    /// two phrases into one scan window — and the rescue shift then drags the healthy か down with it.
    #[test]
    fn a_written_rest_delimits_phrases_like_a_gap() {
        use crate::inference::vocal_range::{dead_only_plan, SpeakerRange};
        let range = SpeakerRange::bounds((48.0, 79.0), (55.0, 74.0));
        let score = [note("か", 60), note("R", 71), note("き", 85)];

        let (plan, unfixable) = dead_only_plan(&plan_note_nums(&score, crate::inference::g2p_alias::PhonemeSet::Words), 0, &range);
        assert!(unfixable.is_empty(), "the dead phrase has a landing — nothing should be unfixable");
        assert_eq!(plan.len(), 1, "one dead phrase ⇒ one group");
        assert_eq!(
            (plan[0].start, plan[0].end),
            (2, 2),
            "only the phrase that is actually dead may be transposed"
        );

        // The pre-fix input, for contrast: the same score with the drawn pitch left in place.
        let raw: Vec<i64> = score.iter().map(|n| n.note_num).collect();
        let (welded, _) = dead_only_plan(&raw, 0, &range);
        assert_ne!(plan, welded, "reading a silence's drawn pitch changes the range decision");
        assert!(
            welded.first().is_none_or(|g| g.start == 0),
            "…and when it does decide, it drags the healthy phrase along (start 0)"
        );
    }

    /// S109 (§G13-M2) — the ONE inequality that keeps the editor/render `phoneme_input` fork
    /// unreachable, pinned across the language boundary.
    ///
    /// The editor commands silently DROP a `phoneme_input` longer than `MAX_LYRIC_CHARS`, while
    /// `render_vocal_segment` applies no text bound at all — so on such a note the preview would show
    /// one thing and the render sing another. That never happens today only because the FRONTEND caps
    /// `phonemeInput` at `MAX_LYRIC_LEN = 64` on every ingress (`sanitizeText`, `src/lib/vocalNotes.ts`),
    /// and 64 < 256. That inequality is the whole guarantee, it lives in another language, and until
    /// now nothing checked it — raise `MAX_LYRIC_LEN` past 256 (or ship the backlogged SV-style
    /// phoneme editor with its own cap) and the fork goes live silently.
    ///
    /// Read out of the TypeScript source with `include_str!` — the same zero-drift trick
    /// `bundled_dictionary_targets` uses on tauri.conf.json — so this cannot rot into a hand-copied
    /// number. If the declaration is ever reshaped, this test fails at the PARSE with a message
    /// saying so, rather than silently finding nothing and passing (S108: a gate that can't see its
    /// own subject is worse than no gate).
    #[test]
    fn phoneme_input_bound_is_unreachable_from_the_editor() {
        static VOCAL_NOTES_TS: &str = include_str!("../../../src/lib/vocalNotes.ts");
        let front_cap: usize = VOCAL_NOTES_TS
            .lines()
            .find_map(|l| l.trim().strip_prefix("export const MAX_LYRIC_LEN"))
            .and_then(|rest| rest.split('=').nth(1))
            .and_then(|v| v.trim().trim_end_matches(';').parse().ok())
            .expect(
                "could not parse `export const MAX_LYRIC_LEN = <n>;` out of src/lib/vocalNotes.ts — \
                 the declaration was reshaped, so this guard is now blind. Re-point it, do NOT delete it: \
                 it is the only thing keeping the editor/render phoneme_input fork unreachable.",
            );
        assert!(
            front_cap < MAX_LYRIC_CHARS,
            "MAX_LYRIC_LEN ({front_cap}) is no longer below the backend bound MAX_LYRIC_CHARS ({}) — \
             a phoneme_input can now reach the editor commands over that bound, where it is SILENTLY \
             dropped, while render_vocal_segment keeps it. Fix the fork (make both sides agree and be \
             loud) before raising the frontend cap; see MAX_LYRIC_CHARS' doc comment.",
            MAX_LYRIC_CHARS
        );
    }

    /// S113 (§C14) — the alias-hint wire spelling exists in TWO languages, and this is what keeps
    /// them from drifting. `validate_lyrics` serializes `AliasHint` into the `hint` field of a
    /// `phones` verdict; `oovWatch.ts` types that field as `AliasHintId`. Add a variant on one side
    /// only and the frontend silently stops recognising it — the note would keep sounding (this
    /// channel is advisory) and NOTHING would report the gap, which is the worst shape a warning
    /// channel can fail in.
    ///
    /// Same `include_str!` trick as the bound test above: read the union out of the TypeScript
    /// source rather than hand-copying it, and fail at the PARSE if the declaration is reshaped.
    /// ⚠ `PhonemeSetId`, three lines above it in that file, carries the same "must change together"
    /// comment with NO guard — recorded in `project_v2_pending_cleanups`, deliberately not widened
    /// into this round.
    #[test]
    fn s113_alias_hint_wire_matches_the_ts_union() {
        use crate::inference::g2p_alias::AliasHint;
        static PROJECT_TS: &str = include_str!("../../../src/types/project.ts");
        let decl = PROJECT_TS
            .lines()
            .find_map(|l| l.trim().strip_prefix("export type AliasHintId ="))
            .expect(
                "could not find `export type AliasHintId = …;` in src/types/project.ts — the \
                 declaration was reshaped, so this guard is now blind. Re-point it, do NOT delete it.",
            );
        let ts: Vec<&str> = decl
            .split('|')
            .map(|s| s.trim().trim_end_matches(';').trim().trim_matches('"'))
            .filter(|s| !s.is_empty())
            .collect();
        let mut rust: Vec<&str> = AliasHint::ALL.iter().map(|h| h.wire()).collect();
        rust.sort_unstable();
        let mut ts_sorted = ts.clone();
        ts_sorted.sort_unstable();
        assert_eq!(
            rust, ts_sorted,
            "AliasHint::ALL and the TS `AliasHintId` union disagree — a hint the frontend cannot \
             name is a hint nobody sees"
        );
    }
}
