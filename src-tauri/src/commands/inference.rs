use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;
use tauri::{Emitter, State};

use crate::inference::{rvc, sovits, RvcOptions, SovitsOptions, SynthesisResult, VoiceBackendType};
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
            return Err("试听渲染进行中，请等待完成后再渲染".into());
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

// ─── aux model resolution (models_dir/aux/...) ───────────────────────────────

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

/// models_dir/aux/<filename>, with a clear Chinese error naming the missing file + the
/// exact directory it must be placed in.
pub(crate) fn aux_path(state: &AppState, filename: &str, label: &str) -> Result<PathBuf, String> {
    let dir = state.models.models_dir().join("aux");
    let path = dir.join(filename);
    if !path.exists() {
        return Err(format!(
            "缺少{} {}，请将其放入 {}",
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
        768 => aux_path(state, AUX_CONTENTVEC_768, "内容特征模型"),
        256 => aux_path(state, AUX_CONTENTVEC_256, "内容特征模型"),
        other => Err(format!(
            "不支持的内容特征维度 {}（仅支持 256 / 768）——请检查模型配置 features_dim / speech_encoder",
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
            "模型 '{}' 是旧版导出格式（缺少 {} 输入签名），请删除后重新导入以完成升级",
            entry.name, input
        ));
    }
    Ok(())
}

pub(crate) fn get_entry(state: &AppState, voice_name: &str) -> Result<ModelEntry, String> {
    state.models.get(voice_name).ok_or_else(|| {
        format!(
            "未找到模型 '{}'，请先在资源管理器中导入",
            voice_name
        )
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
    let aux = state.models.models_dir().join("aux");
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
        .map_err(|e| format!("无法读取{}（{}）：{}", what, path.display(), e))?;
    serde_json::from_str(&content)
        .map_err(|e| format!("{}解析失败（{}）：{}", what, path.display(), e))
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
                aux_path(app, AUX_NSF_HIFIGAN, "NSF-HiFiGAN声码器")?,
                aux_path(app, AUX_NSF_HIFIGAN_JSON, "NSF-HiFiGAN声码器配置")?,
                app.models.models_dir().join("aux"),
                "NSF-HiFiGAN 声码器".to_string(),
            ),
            Some(name) => {
                // type-scoped lookup (设计红队 A5): singers commonly own a
                // same-name rvc/sovits pair AND a same-name vocoder
                let ventry = app
                    .models
                    .get_by_type(name, &crate::models::ModelType::NsfHifigan)
                    .ok_or_else(|| {
                        format!(
                            "声码器「{}」不存在或已被删除——请在节点里重新选择声码器（或选回默认声码器）",
                            name
                        )
                    })?;
                let json = ventry.path.with_extension("json");
                if !json.is_file() {
                    return Err(format!(
                        "声码器「{}」缺少配置文件 {}——请在资源管理中重新导入",
                        name,
                        json.display()
                    ));
                }
                let base = ventry
                    .path
                    .parent()
                    .map(|p| p.to_path_buf())
                    .unwrap_or_default();
                (ventry.path.clone(), json, base, format!("声码器「{}」", name))
            }
        };
        let sidecar: VocoderSidecar = read_json(&voc_json, "声码器配置")?;
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
                    "缺少{}的滤波器文件 {}——请在资源管理中重新导入该声码器",
                    voc_what,
                    voc_mel_path.display()
                )
            } else {
                format!(
                    "缺少NSF-HiFiGAN滤波器 {}，请将其放入 {}",
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
                            "{}的配置缺少字段「{}」——无法确认梅尔频谱格式，请重新导入",
                            voc_what, key
                        ))
                    }
                    Some(v) if v != want => {
                        return Err(format!(
                            "{}的梅尔频谱格式与模型不一致（{} = {}，标准格式需要 {}）——声码器只能用于频谱格式一致的模型",
                            voc_what, key, v, want
                        ))
                    }
                    _ => {}
                }
            }
        }
        if sidecar.sample_rate != entry.sample_rate || sidecar.hop_size as usize != hop_size {
            return Err(format!(
                "{}（{}Hz/hop {}）与模型（{}Hz/hop {}）几何不一致——浅扩散/增强器仅支持 44.1kHz/512 的 SoVITS 模型",
                voc_what, sidecar.sample_rate, sidecar.hop_size, entry.sample_rate, hop_size
            ));
        }
        let filters = app
            .inference
            .load_npy(&voc_mel_path)
            .map_err(|e| e.to_string())?;
        if filters.nrows() != sidecar.num_mels as usize {
            return Err(format!(
                "声码器滤波器形状（{}×{}）与配置 num_mels={} 不一致",
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
                format!(
                    "模型 '{}' 未附带扩散模型——导入时附加扩散 .pt+.yaml 可启用浅扩散",
                    entry.name
                )
            })?,
        };
        let sidecar: DiffusionSidecar =
            read_json(&diff_dir.join("diffusion.json"), "扩散模型配置")?;

        if sidecar.schedule != "linear" || sidecar.timesteps == 0 {
            return Err(format!(
                "扩散模型的 schedule（{}，timesteps={}）不受支持——请用当前版本的转换器重新导入",
                sidecar.schedule, sidecar.timesteps
            ));
        }
        if sidecar.encoder_out_channels as usize != dim {
            return Err(format!(
                "扩散模型的特征维度（{}）与主模型（{}）不一致，无法配合使用",
                sidecar.encoder_out_channels, dim
            ));
        }
        if sidecar.sample_rate != entry.sample_rate || sidecar.block_size as usize != hop_size {
            return Err(format!(
                "扩散模型（{}Hz/block {}）与主模型（{}Hz/hop {}）几何不一致",
                sidecar.sample_rate, sidecar.block_size, entry.sample_rate, hop_size
            ));
        }
        let method = crate::inference::diffusion::SamplerMethod::parse(&options.diffusion_method)
            .ok_or_else(|| {
                format!(
                    "未知的扩散采样器：{}（支持 naive/ddim/pndm/dpm-solver/dpm-solver++/unipc）",
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
                    "该扩散模型仅支持浅扩散，无法单独推理（k_step_max < timesteps）".to_string(),
                );
            }
        } else {
            if options.k_step == 0 {
                return Err("扩散步数 k_step 不能为 0".to_string());
            }
            if options.k_step as usize > k_step_max {
                return Err(format!(
                    "浅扩散 k_step（{}）超过该扩散模型的上限 k_step_max={}",
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
                    "扩散步数 ÷ 加速倍数 = {} 步，dpm/unipc 采样器至少需要 2 步——请降低加速倍数",
                    solver_steps
                ));
            }
        }
        if sidecar.n_spk > 1 {
            let spk = options.speaker_id.unwrap_or(0);
            if spk >= sidecar.n_spk {
                return Err(format!(
                    "说话人 id {} 超出扩散模型的 n_spk={}",
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
                return Err(format!("扩散模型文件缺失：{}——请重新附加扩散模型", p.display()));
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
            format!(
                "扩散附件配置缺少 {}（diffusion.json 损坏或版本过旧）——请重新附加扩散模型",
                what
            )
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
            return Err(format!(
                "模型 '{}' 导出时未包含自动音高预测器（该模型可能没有 f0_decoder 权重，或导出于旧版转换器——重新导入 .pth 可生成）",
                entry.name
            ));
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
            .ok_or_else(|| {
                format!(
                    "自动音高预测器文件缺失：{}——请重新导入模型",
                    file
                )
            })?;
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
    options: RvcOptions,
) -> Result<SynthesisResult, String> {
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
    let rmvpe_path = aux_path(&app, AUX_RMVPE, "音高检测模型")?;
    let mel_path = aux_path(&app, AUX_RMVPE_MEL, "音高检测滤波器")?;

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
        rvc::run_pipeline(&model, &audio_buf, &options, &progress, &cancel)
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("推理任务失败: {}", e))?
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
    options: SovitsOptions,
) -> Result<SynthesisResult, String> {
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
        return Err(format!("模型 '{}' 配置的 hop_size 为 0，无法推理", voice_name));
    }
    let min_t = min_frames(&entry.config, 6);
    // Feed vol IFF the exported graph HAS the input — the sidecar "inputs" array is the
    // authority (final contract); vol_embedding bool is the fallback for older sidecars.
    let vol_embedding = sidecar_has_input(&entry, "vol")
        .unwrap_or_else(|| entry.config.vol_embedding.unwrap_or(false));
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
    let rmvpe_path = aux_path(&app, AUX_RMVPE, "音高检测模型")?;
    let mel_path = aux_path(&app, AUX_RMVPE_MEL, "音高检测滤波器")?;

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
    let cluster = if options.cluster_ratio > 0.0 {
        // ①c: under a blend, the retrieval/cluster asset follows the DOMINANT (max-weight)
        // speaker; without a blend this is just speaker_id (fallback 0) — unchanged behavior.
        let spk = crate::inference::dominant_speaker(&options.spk_mix, options.speaker_id);
        let parent = entry.path.parent().map(|p| p.to_path_buf()).unwrap_or_default();
        let stem = entry.path.file_stem().unwrap_or_default().to_string_lossy().to_string();
        let cluster_dir = parent.join(format!("{}.cluster", stem));
        let model_dir = cluster_dir; // primary probe location; falls back to `parent` below
        let safe = |name: &str| -> String {
            name.chars()
                .map(|c| if matches!(c, '\\' | '/' | ':' | '*' | '?' | '"' | '<' | '>' | '|') { '_' } else { c })
                .collect()
        };

        let mut found = None;
        // A present-but-unreadable cluster asset (wrong dtype/rank) is treated as ABSENT — the
        // cluster blend is optional, so a bad file must not abort the whole 翻唱 (matches the
        // original's missing-file skip). Warn and fall through to None.
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
        found // None → pipeline logs the skip (mirrors the original's missing-file behavior)
    } else {
        None
    };

    let audio_buf =
        crate::audio::load_audio(&PathBuf::from(&audio_path)).map_err(|e| e.to_string())?;

    let progress = progress_emitter(app_handle, app.clone(), run_epoch, node_id);
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
            spk_mix,
            unit_interpolate_mode,
            noise_channels: nch,
            min_frames: min_t,
        };
        sovits::run_pipeline(&model, &audio_buf, &options, &progress, &cancel)
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("推理任务失败: {}", e))?
}

// ─── detect_f0 (kept signature: audio path → f0 Hz @ 100 fps) ────────────────

#[tauri::command]
pub async fn detect_f0(
    state: State<'_, Arc<AppState>>,
    audio_path: String,
) -> Result<Vec<f32>, String> {
    let app = state.inner().clone();
    let rmvpe_path = aux_path(&app, AUX_RMVPE, "音高检测模型")?;
    let mel_path = aux_path(&app, AUX_RMVPE_MEL, "音高检测滤波器")?;
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
    .map_err(|e| format!("音高检测任务失败: {}", e))?
}

// run_s2h + the s2h double-head module were removed in S48 Phase 1c — that pre-S35 contract
// (phonemes/durations/pitches → hubert+contentvec) was wrong for ScoreToCV. The real score→cv path is
// inference::score2cv (build_arrays + ONNX), wired into the vocal render pipeline in a later phase.
