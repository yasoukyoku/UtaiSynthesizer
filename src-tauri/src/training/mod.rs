//! Training run lifecycle: spawns `python -m utai_train.runner` (training/ package,
//! its own venv), relays the stdout JSONL protocol v2 as tauri events, keeps the
//! loss history for the training page, and owns the graceful-stop flag file.
//!
//! Everything is app_dir/data_dir absolute (the opus4.6-era module was cwd-relative
//! — that debt is gone with this rewrite). stdout belongs to the protocol; stderr
//! goes to a ring buffer surfaced LOUDLY on abnormal exit (antivirus kills, OOM).
//! Post-processing (pth→onnx conversion, registry import, audition rendering) is
//! driven by the frontend through the EXISTING model-import command chain — this
//! module ends at the protocol `done`.

use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tauri::Emitter;

use crate::{Result, UtaiError};

const STDERR_RING_CAP: usize = 200;
const HISTORY_CAP: usize = 40_000;
const SEED: u32 = 1234;

fn d_true() -> bool {
    true
}
fn d_save_every() -> u32 {
    5
}
fn d_save_steps() -> u32 {
    800
}
fn d_keep_ckpts() -> u32 {
    3
}
fn d_total_steps() -> u32 {
    100_000
}
fn d_force_save() -> u32 {
    10_000
}
fn d_crop_mel() -> u32 {
    32
}

/// Workspace lineage for the cross-backend collision guard: sovits_diff shares
/// the sovits workspace (that is the whole point — the diffusion companion
/// reuses the main model's preprocessing caches), rvc stays its own family.
/// The manifest stores THIS value under its historical "backend" key, so
/// pre-S39 manifests need zero migration.
pub(crate) fn backend_family(backend: &str) -> &str {
    match backend {
        "sovits_diff" => "sovits",
        other => other,
    }
}

/// ①c one co-trained speaker: a display name + its own audio files. The id
/// (emb_g row / config.spk value) is the group's index in the request order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerGroup {
    pub name: String,
    pub files: Vec<String>,
}

/// Every disk asset one training run needs, resolved from (backend, version, sample_rate,
/// aug_copies) — THE single source shared by try_start's up-front verification and the S66
/// `training_required_assets` pre-flight command (the frontend's "missing base model"
/// dialog), so the two can never drift. Pure path math except the 4.0 diffusion-base
/// probe, whose OPTIONALITY is existence-defined (present = used, absent = from-scratch).
pub struct ResolvedTrainingAssets {
    /// (label, path) in the exact order try_start verifies them; labels are the stable
    /// English tokens that ride in the TRAINING_ASSET_MISSING detail payload.
    pub required: Vec<(String, PathBuf)>,
    pub contentvec: PathBuf,
    pub rmvpe_pt: PathBuf,
    pub pretrain_g: PathBuf,
    pub pretrain_d: PathBuf,
    pub nsf_hifigan_model: PathBuf,
    pub diffusion_pretrain: PathBuf,
    pub vocoder_pretrain: PathBuf,
}

pub fn resolve_training_assets(
    data_dir: &Path,
    backend: &str,
    version: &str,
    sample_rate: &str,
    aug_copies: u32,
) -> Result<ResolvedTrainingAssets> {
    let aux_dir = data_dir.join("models").join(crate::models::AUX_DIR_NAME);
    let sovits_train_dir = data_dir.join("models").join("training").join("sovits");
    // one-ContentVec-space principle: the training extractor must be the same
    // aux graph inference uses — rvc v1 / sovits(_diff) 4.0 / sovits_v2 = 256l9,
    // rvc v2 / sovits(_diff) 4.1 = 768l12
    let use_256 = version == "v1" || version == "4.0" || version == "4.0-v2";
    let contentvec = aux_dir.join(if use_256 {
        "contentvec_256l9.onnx"
    } else {
        "contentvec_768l12.onnx"
    });
    // rmvpe is TWO different lineages: aux/rmvpe.pt = RVC's raw-state-dict E2E;
    // so-vits vendors the yxlllc/RMVPE fork (E2E0, +unet.tf.* layers, wrapped
    // as {'model': sd}) — the files are NOT interchangeable.
    // vocoder also gets the SOVITS lineage: its own f0 products are
    // parselmouth-blooded and measurably blind to PSOLA glitches, so the
    // S41 aug quality gate re-analyzes the audio with the sovits RMVPE
    // (gate_aug_semantic part 4 keeps the blind spot on record)
    // sovits_v2 is its own workspace family but the same yxlllc rmvpe lineage
    let rmvpe_pt = if matches!(backend_family(backend), "sovits" | "sovits_v2") || backend == "vocoder" {
        sovits_train_dir.join("rmvpe.pt")
    } else {
        aux_dir.join("rmvpe.pt")
    };
    // per-backend required files beyond contentvec+rmvpe
    let mut required: Vec<(String, PathBuf)> = Vec::new();
    let mut pretrain_g = PathBuf::new();
    let mut pretrain_d = PathBuf::new();
    let mut nsf_hifigan_model = PathBuf::new();
    let mut diffusion_pretrain = PathBuf::new();
    let mut vocoder_pretrain = PathBuf::new();
    match backend {
        "rvc" => {
            let pretrain_dir = data_dir.join("models").join("training").join("rvc").join(
                if version == "v1" {
                    "pretrained"
                } else {
                    "pretrained_v2"
                },
            );
            pretrain_g = pretrain_dir.join(format!("f0G{}.pth", sample_rate));
            pretrain_d = pretrain_dir.join(format!("f0D{}.pth", sample_rate));
            required.push(("pretrained base G".into(), pretrain_g.clone()));
            required.push(("pretrained base D".into(), pretrain_d.clone()));
        }
        "sovits" => {
            let pretrain_dir =
                sovits_train_dir.join(if version == "4.0" { "vec256" } else { "vec768" });
            pretrain_g = pretrain_dir.join("G_0.pth");
            pretrain_d = pretrain_dir.join("D_0.pth");
            required.push(("pretrained base G".into(), pretrain_g.clone()));
            required.push(("pretrained base D".into(), pretrain_d.clone()));
        }
        "sovits_v2" => {
            // 4.0-v2 (VISinger2): the official base pair from the 4.0-v2
            // branch, its own asset dir (the ckpt layout is NOT interchangeable
            // with the 4.x vec256/vec768 bases)
            let pretrain_dir = data_dir.join("models").join("training").join("sovits_v2");
            pretrain_g = pretrain_dir.join("G_0.pth");
            pretrain_d = pretrain_dir.join("D_0.pth");
            required.push(("pretrained base G".into(), pretrain_g.clone()));
            required.push(("pretrained base D".into(), pretrain_d.clone()));
        }
        "vocoder" => {
            // NSF-HiFiGAN finetune (S40): the ONLY asset is the classic
            // 2024.02 community base checkpoint (lightning format, G+D).
            // CC BY-NC-SA weights — never bundled, the user downloads it;
            // the label doubles as the download guidance in the missing-
            // file error. ContentVec/RMVPE/configs/mute are NOT used by
            // this pipeline (设计红队 A17: required 收敛进各臂).
            vocoder_pretrain = data_dir
                .join("models")
                .join("training")
                .join("vocoder")
                .join("nsf_hifigan_44.1k_hop512_128bin_2024.02.ckpt");
            required.push((
                "vocoder finetune base ckpt (download \
                 nsf_hifigan_44.1k_hop512_128bin_2024.02.zip from \
                 github.com/openvpi/SingingVocoders releases v0.0.2 and extract the \
                 .ckpt; weights are CC BY-NC-SA 4.0 — finetuned outputs inherit the \
                 license)"
                    .into(),
                vocoder_pretrain.clone(),
            ));
        }
        "sovits_diff" => {
            // sovits_diff: the mel recipe IS the vocoder's (torch ckpt, not
            // the aux onnx) + the diffusion base model. The vec256 ecosystem
            // has NO public diffusion base (the one community HF repo went
            // private, 2026-07) — 4.0 trains from scratch, loudly surfaced
            // in the params UI; the vec768 base ships as a dev asset and is
            // hard-required so its absence can never silently degrade.
            nsf_hifigan_model = sovits_train_dir.join("nsf_hifigan").join("model");
            required.push(("NSF-HiFiGAN vocoder (model)".into(), nsf_hifigan_model.clone()));
            required.push((
                "NSF-HiFiGAN config (config.json)".into(),
                sovits_train_dir.join("nsf_hifigan").join("config.json"),
            ));
            let base = sovits_train_dir
                .join("diffusion")
                .join(if version == "4.0" { "vec256" } else { "vec768" })
                .join("model_0.pt");
            if version == "4.0" {
                if base.is_file() {
                    diffusion_pretrain = base;
                } else {
                    tracing::warn!("no vec256 diffusion base model — training from scratch");
                }
            } else {
                diffusion_pretrain = base.clone();
                required.push(("diffusion base model (model_0.pt)".into(), base));
            }
        }
        // the whitelist match above already rejected unknown backends —
        // this arm exists so a future backend CANNOT silently inherit
        // another backend's asset resolution (设计红队 A17)
        other => {
            return Err(UtaiError::Training(format!(
                "TRAINING_INTERNAL_ASSET_BRANCH: {}",
                other
            )));
        }
    }
    if backend != "vocoder" {
        // the vocoder pipeline extracts neither features nor f0-by-model
        // (parselmouth is in-process) — requiring these would be a lie
        required.push(("ContentVec feature extractor".into(), contentvec.clone()));
        required.push(("RMVPE pitch model (rmvpe.pt)".into(), rmvpe_pt.clone()));
    } else if aug_copies > 0 {
        // ...except the S41 aug quality gate, which is rmvpe-blooded by
        // design (see the lineage comment above) — only when augmenting
        required.push((
            "RMVPE pitch model (rmvpe.pt, augmentation quality gate)".into(),
            rmvpe_pt.clone(),
        ));
    }
    Ok(ResolvedTrainingAssets {
        required,
        contentvec,
        rmvpe_pt,
        pretrain_g,
        pretrain_d,
        nsf_hifigan_model,
        diffusion_pretrain,
        vocoder_pretrain,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartTrainingRequest {
    pub model_name: String,
    pub backend: String, // "rvc" | "sovits" | "sovits_v2" | "sovits_diff" | "vocoder"
    /// rvc: "v1" | "v2" — sovits/sovits_diff: "4.1" | "4.0" — sovits_v2: fixed
    /// "4.0-v2" — vocoder: fixed "nsf_hifigan" (manifest markers, 一期单格式类)
    pub version: String,
    /// rvc: "32k" | "40k" | "48k" — sovits/vocoder: fixed "44k"
    pub sample_rate: String,
    pub dataset_files: Vec<String>,
    /// ①c multi-speaker co-training (SoVITS = α, RVC = α′). Empty or 1 group =
    /// single-speaker = the byte-identical legacy path (uses dataset_files).
    /// >1 groups = per-speaker subdir import + run.json "speakers"; the emb_g
    /// speaker id is the group's index (list order, frozen in the manifest).
    #[serde(default)]
    pub speakers: Vec<SpeakerGroup>,
    pub total_epoch: u32,
    pub batch_size: u32,
    #[serde(default = "d_save_every")]
    pub save_every_epoch: u32,
    #[serde(default = "d_true")]
    pub save_every_weights: bool,
    #[serde(default = "d_true")]
    pub keep_only_latest: bool,
    #[serde(default)]
    pub cache_gpu: bool,
    #[serde(default = "d_true")]
    pub fp16: bool,
    /// Device identity in the ACCELERATOR'S own namespace, straight from
    /// get_hardware_info.training_gpus: an NVIDIA UUID ("GPU-…", what
    /// CUDA_VISIBLE_DEVICES actually accepts) or a vendor-relative index.
    /// "" = auto (leave visibility unset → torch's own default device). S67:
    /// this was a raw WMI adapter index — on an iGPU+NVIDIA box, SELECTING the
    /// NVIDIA card masked every GPU and training fell back to CPU silently.
    #[serde(default)]
    pub gpu: String,
    #[serde(default)]
    pub force_cpu: bool,
    #[serde(default)]
    pub spk_id: u32,
    /// true = 重训 (wipe the workspace), false = 续训 (resume from latest ckpt)
    #[serde(default)]
    pub fresh: bool,
    /// S41 PSOLA data augmentation: pitch-shifted copies per slice (0-3, 0 =
    /// off). Applies to rvc / sovits / vocoder; sovits_diff IGNORES the
    /// request value and inherits the workspace manifest's (shared dataset_44k
    /// — same posture as vol_embedding/loudnorm).
    #[serde(default)]
    pub aug_copies: u32,
    // ---- SoVITS-only knobs (ignored by the rvc backend) ----
    /// 响度嵌入 (couples train.vol_aug + model.vol_embedding, like upstream --vol_aug)
    #[serde(default)]
    pub vol_embedding: bool,
    /// resample 响度归一 (upstream default ON; ours OFF — lossy per upstream README)
    #[serde(default)]
    pub loudnorm: bool,
    /// 聚类中心 (kmeans) instead of the default retrieval matrix
    #[serde(default)]
    pub kmeans: bool,
    /// ckpt/eval cadence in global steps (upstream eval_interval)
    #[serde(default = "d_save_steps")]
    pub save_every_steps: u32,
    /// how many G_/D_ checkpoints to keep (upstream keep_ckpts; *_0.pth exempt)
    #[serde(default = "d_keep_ckpts")]
    pub keep_ckpts: u32,
    /// cache the whole dataset in RAM (upstream all_in_mem)
    #[serde(default)]
    pub all_in_mem: bool,
    // ---- sovits_diff-only knobs (ignored by the other backends) ----
    /// completion target in global steps (diffusion epochs are tiny sentinel
    /// units — upstream itself thinks in steps; total_epoch is sent as 0)
    #[serde(default = "d_total_steps")]
    pub total_steps: u32,
    /// 0 = full diffusion (train all 1000 t), else shallow k_step_max
    #[serde(default)]
    pub k_step_max: u32,
    /// milestone keep cadence in steps — normalized to a multiple of
    /// save_every_steps (upstream's delete-previous rule only ever keeps
    /// checkpoints on the save grid, so a non-multiple would silently shift
    /// the real milestone grid to the lcm)
    #[serde(default = "d_force_save")]
    pub interval_force_save: u32,
    /// cache the whole dataset in RAM during diffusion training
    #[serde(default = "d_true")]
    pub cache_all_data: bool,
    // ---- vocoder-only knobs (ignored by the other backends) ----
    /// dataset crop window in mel frames (upstream crop_mel_frames; 32 = the
    /// ft_hifigan 16G preset, 48 = 24G)
    #[serde(default = "d_crop_mel")]
    pub crop_mel_frames: u32,
    /// freeze the MPD discriminator (upstream README: small-step finetunes
    /// may benefit; couples freezing_enabled + frozen_params python-side)
    #[serde(default)]
    pub freeze_mpd: bool,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct StageInfo {
    pub stage: String,
    pub done: Option<u64>,
    pub total: Option<u64>,
    pub progress: Option<f32>,
    pub message: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StepInfo {
    pub step: u64,
    pub total_steps: u64,
    pub epoch: u32,
    pub total_epochs: u32,
    pub lr: f64,
    pub losses: HashMap<String, f64>,
    pub eta_secs: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StepPoint {
    pub step: u64,
    pub lr: f64,
    pub losses: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CkptInfo {
    pub kind: String, // periodic | best | final | stop
    pub path: String,
    pub step: u64,
    pub epoch: u32,
    pub metric: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct TrainingSnapshot {
    /// idle | starting | running | completed | stopped | error
    pub state: String,
    pub error: Option<String>,
    pub backend: String,
    pub model_name: String,
    pub model_slug: String,
    pub workspace: String,
    pub total_epochs: u32,
    pub stage: Option<StageInfo>,
    pub step: Option<StepInfo>,
    pub ckpts: Vec<CkptInfo>,
    pub summary: Option<serde_json::Value>,
    pub stop_requested: bool,
    pub elapsed_secs: u64,
    /// last stderr lines — populated when state == error (loud failures)
    pub stderr_tail: Vec<String>,
    /// ①c: ordered speaker DISPLAY names for a multi-speaker run (index = emb_g id), so the
    /// audition speaker picker can label by name without depending on the editable DataStep
    /// state. Empty for single-speaker runs. Reflects the RUN (frozen at start), not the form.
    #[serde(default)]
    pub speakers: Vec<String>,
}

struct Inner {
    snapshot: Mutex<TrainingSnapshot>,
    history: Mutex<Vec<StepPoint>>,
    stderr_ring: Mutex<VecDeque<String>>,
    child: Mutex<Option<std::process::Child>>,
    stop_file: Mutex<Option<PathBuf>>,
    running: AtomicBool,
    /// Hard-abort request covering the PRE-SPAWN window (dataset import → spawn →
    /// child slotting): force_stop/quit can otherwise only kill an already-slotted
    /// child, silently no-oping during a minutes-long import.
    abort: AtomicBool,
    started_at: Mutex<Option<Instant>>,
}

pub struct TrainingManager {
    app_dir: PathBuf,
    inner: Arc<Inner>,
}

/// Workspace directory for a model name (the frontend's "does a resumable
/// workspace exist?" check needs the same mapping start() uses).
pub fn workspace_path(data_dir: &Path, model_name: &str) -> PathBuf {
    data_dir.join("training").join(slugify(model_name))
}

/// Structured workspace facts for the frontend confirm dialogs: the main
/// retrain dialog warns when it would also wipe diffusion progress; the
/// diffusion card phrases its dialog by resume-vs-cache-reuse. Read-only.
#[derive(Debug, Clone, Serialize)]
pub struct WorkspaceInfo {
    pub exists: bool,
    /// manifest family ("rvc"/"sovits"); "" when absent/unreadable
    pub family: String,
    /// manifest version ("v1"/"v2"/"4.1"/"4.0"); "" when absent — the frontend
    /// must not offer「续训」across a version mismatch (the Rust resume guard
    /// would refuse it anyway, but only AFTER the dialog promised it)
    pub version: String,
    /// manifest sample rate ("32k"/"40k"/"48k"/"44k"); "" when absent
    pub sample_rate: String,
    /// any main-model checkpoint (G_*.pth) at the workspace root
    pub has_main_progress: bool,
    /// max numbered diffusion checkpoint step (model_<n>.pt); 0 = none/base only
    pub diff_steps: u64,
    /// manifest aug_copies (S41 数据增强份数) — diff runs inherit it from the
    /// main training; surfaced so the diff params page shows the real value
    pub aug_copies: u64,
    /// a reusable shared slice pool exists (prior completed import): diff runs
    /// may start WITHOUT re-importing data when this is true (S41 共享池模式)
    pub has_dataset: bool,
    /// ①c resume config-diff: manifest vol_embedding (SoVITS main model) — None when absent /
    /// not sovits. Surfaced so the resume dialog can show a mismatch BEFORE start (the Rust guard
    /// rejects it otherwise, but only after the dialog already promised 续训).
    pub vol_embedding: Option<bool>,
    /// ①c: manifest n_speakers (multi-speaker co-train); 1 when absent (single-speaker).
    pub n_speakers: u64,
    /// ①c: ordered speaker DISPLAY names (from the prior run.json, index = emb_g id = DataStep
    /// order); empty for single-speaker. The manifest stores only slugs, so names come from run.json.
    pub speakers: Vec<String>,
    /// ①c: manifest diff_k_step_max (sovits_diff); 0 when absent.
    pub diff_k_step_max: u64,
}

/// A reusable shared slice pool: raw dataset files from a prior import plus
/// the fingerprint marker (written when a run ENTERS preprocessing — partial
/// caches are fine, the diff pipeline re-runs the shared preprocess chain and
/// fills whatever is missing; what matters is that dataset/ holds a real
/// prior import, not that it finished).
pub fn has_dataset_pool(ws: &Path) -> bool {
    ws.join("dataset.fingerprint").is_file()
        && std::fs::read_dir(ws.join("dataset"))
            .map(|mut d| d.next().is_some())
            .unwrap_or(false)
}

pub fn workspace_info(data_dir: &Path, name: &str) -> WorkspaceInfo {
    let ws = workspace_path(data_dir, name);
    let manifest = std::fs::read_to_string(ws.join("run_manifest.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .unwrap_or_default();
    let field = |k: &str| manifest[k].as_str().unwrap_or("").to_string();
    // ①c: display speaker names (ordered = emb_g id) from the manifest's `speaker_names` — the
    // manifest is merge-preserved across a sovits_diff run (unlike run.json, which the diff run
    // overwrites without a speakers key). Empty for single-speaker. (Falls back to run.json for a
    // pre-fix multi-speaker workspace that predates speaker_names — cheap, keeps old runs correct.)
    let speakers: Vec<String> = manifest["speaker_names"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect::<Vec<String>>()
        })
        .filter(|v| !v.is_empty())
        .or_else(|| {
            std::fs::read_to_string(ws.join("run.json"))
                .ok()
                .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                .and_then(|v| {
                    v.get("speakers").and_then(|s| s.as_array()).map(|arr| {
                        arr.iter()
                            .filter_map(|e| e.get("name").and_then(|n| n.as_str()).map(String::from))
                            .collect()
                    })
                })
                .filter(|v: &Vec<String>| !v.is_empty())
        })
        .unwrap_or_default();
    WorkspaceInfo {
        exists: ws.exists(),
        family: field("backend"),
        version: field("version"),
        sample_rate: field("sample_rate"),
        has_main_progress: has_main_progress(&ws),
        diff_steps: max_diffusion_step(&ws).unwrap_or(0),
        aug_copies: manifest["aug_copies"].as_u64().unwrap_or(0),
        has_dataset: has_dataset_pool(&ws),
        vol_embedding: manifest["vol_embedding"].as_bool(),
        n_speakers: manifest["n_speakers"].as_u64().unwrap_or(1),
        speakers,
        diff_k_step_max: manifest["diff_k_step_max"].as_u64().unwrap_or(0),
    }
}

fn has_main_progress(workspace: &Path) -> bool {
    std::fs::read_dir(workspace)
        .map(|rd| {
            rd.filter_map(|e| e.ok()).any(|e| {
                let n = e.file_name().to_string_lossy().into_owned();
                n.starts_with("G_") && n.ends_with(".pth")
            })
        })
        .unwrap_or(false)
}

/// Max numbered model_ckpt_steps_<N>.ckpt at the workspace root — the vocoder
/// backend's lightning checkpoints (mirrors get_latest_checkpoint_path in the
/// sidecar). ⚠️ N is in lightning GLOBAL units: the manual-opt GAN counts the
/// D and G optimizer steps separately, so N = 2 × 实际步 — every comparison
/// against total_steps must divide by 2 first (设计红队 A8).
fn max_vocoder_ckpt_step(workspace: &Path) -> Option<u64> {
    let rd = std::fs::read_dir(workspace).ok()?;
    let mut max: Option<u64> = None;
    for e in rd.filter_map(|e| e.ok()) {
        let n = e.file_name().to_string_lossy().into_owned();
        if let Some(num) = n
            .strip_prefix("model_ckpt_steps_")
            .and_then(|s| s.strip_suffix(".ckpt"))
        {
            if let Ok(v) = num.parse::<u64>() {
                max = Some(max.map_or(v, |m| m.max(v)));
            }
        }
    }
    max
}

/// Max numbered model_<n>.pt in workspace/diffusion — mirrors the sidecar's
/// load_model resume scan (model_0.pt = the seeded base counts as 0).
fn max_diffusion_step(workspace: &Path) -> Option<u64> {
    let rd = std::fs::read_dir(workspace.join("diffusion")).ok()?;
    let mut max: Option<u64> = None;
    for e in rd.filter_map(|e| e.ok()) {
        let n = e.file_name().to_string_lossy().into_owned();
        if let Some(num) = n.strip_prefix("model_").and_then(|s| s.strip_suffix(".pt")) {
            if let Ok(v) = num.parse::<u64>() {
                max = Some(max.map_or(v, |m| m.max(v)));
            }
        }
    }
    max
}

/// ASCII-safe workspace slug for a (possibly CJK) display name: the original
/// RVC/SoVITS toolchains choke on non-ANSI experiment paths, so the workspace
/// stays ASCII and the unicode name lives only in our registry / final artifacts.
fn slugify(name: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut base: String = name
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .take(24)
        .collect();
    if base.is_empty() {
        base = "model".to_string();
    }
    let mut h = std::collections::hash_map::DefaultHasher::new();
    name.hash(&mut h);
    format!("{}_{:08x}", base, h.finish() as u32)
}

/// ①c deterministic ASCII slug per co-trained speaker — the slug is the
/// dataset_44k subdir name AND the config.spk key AND (by list order) the
/// emb_g id, so it MUST be stable across resume (frozen in the manifest) and
/// unique (two speakers sharing a subdir would clobber each other's slices).
/// slugify already hash-suffixes so distinct names rarely collide; dedupe
/// defensively (identical slugs -> _2, _3 …). Returns (display_name, slug) in
/// request order — do NOT sort (id order is authoritative).
fn assign_speaker_slugs(speakers: &[SpeakerGroup]) -> Vec<(String, String)> {
    let mut used = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(speakers.len());
    for sp in speakers {
        let base = slugify(&sp.name);
        let mut slug = base.clone();
        let mut n = 2;
        while used.contains(&slug) {
            slug = format!("{}_{}", base, n);
            n += 1;
        }
        used.insert(slug.clone());
        out.push((sp.name.clone(), slug));
    }
    out
}

impl TrainingManager {
    pub fn new(app_dir: PathBuf) -> Self {
        Self {
            app_dir,
            inner: Arc::new(Inner {
                snapshot: Mutex::new(TrainingSnapshot {
                    state: "idle".into(),
                    ..Default::default()
                }),
                history: Mutex::new(Vec::new()),
                stderr_ring: Mutex::new(VecDeque::new()),
                child: Mutex::new(None),
                stop_file: Mutex::new(None),
                running: AtomicBool::new(false),
                abort: AtomicBool::new(false),
                started_at: Mutex::new(None),
            }),
        }
    }

    pub fn is_active(&self) -> bool {
        self.inner.running.load(Ordering::SeqCst)
    }

    pub fn status(&self) -> TrainingSnapshot {
        let mut s = self.inner.snapshot.lock().clone();
        // started_at is Some only while the run is live; afterwards the final
        // elapsed is frozen into the snapshot (finalize_elapsed)
        if let Some(t) = *self.inner.started_at.lock() {
            s.elapsed_secs = t.elapsed().as_secs();
        }
        s
    }

    pub fn history(&self) -> Vec<StepPoint> {
        self.inner.history.lock().clone()
    }

    /// Reset the DISPLAY state of a finished run back to idle (snapshot, loss
    /// history, stderr ring). Purely cosmetic — workspace files/checkpoints are
    /// untouched and the run stays resumable. Refused while a run is live.
    pub fn reset_display(&self) -> Result<()> {
        if self.is_active() {
            return Err(UtaiError::Training("TRAINING_ACTIVE".into()));
        }
        *self.inner.snapshot.lock() = TrainingSnapshot {
            state: "idle".into(),
            ..Default::default()
        };
        self.inner.history.lock().clear();
        self.inner.stderr_ring.lock().clear();
        *self.inner.started_at.lock() = None;
        Ok(())
    }

    /// Graceful stop: create the flag file; the sidecar saves + finalizes at the
    /// next safe boundary and reports `done(stopped)` through the protocol. If the
    /// run hasn't reached its workspace yet (validation window), fall back to abort.
    pub fn stop(&self) -> Result<()> {
        if !self.is_active() {
            return Ok(());
        }
        self.inner.snapshot.lock().stop_requested = true;
        match self.inner.stop_file.lock().clone() {
            Some(stop_file) => {
                std::fs::write(&stop_file, "stop")?;
                tracing::info!("training stop requested via {}", stop_file.display());
            }
            None => {
                self.inner.abort.store(true, Ordering::SeqCst);
            }
        }
        Ok(())
    }

    /// Hard kill — quit flow / user-confirmed force stop. No finalization. The
    /// abort flag closes the pre-spawn window: the worker checks it during dataset
    /// import and inside the child-slotting critical section, so either the worker
    /// self-terminates or the child is here to be killed.
    pub fn force_stop(&self) -> Result<()> {
        self.inner.abort.store(true, Ordering::SeqCst);
        if let Some(mut child) = self.inner.child.lock().take() {
            child
                .kill()
                .map_err(|e| UtaiError::Training(format!("TRAINING_KILL_FAILED: {}", e)))?;
            tracing::warn!("training force-killed");
        }
        Ok(())
    }

    pub fn start(
        &self,
        app: tauri::AppHandle,
        data_dir: PathBuf,
        req: StartTrainingRequest,
    ) -> Result<()> {
        if self
            .inner
            .running
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(UtaiError::Training("TRAINING_ALREADY_RUNNING".into()));
        }
        let launched = self.try_start(app, data_dir, req);
        if launched.is_err() {
            self.inner.running.store(false, Ordering::SeqCst);
        }
        launched
    }

    fn try_start(
        &self,
        app: tauri::AppHandle,
        data_dir: PathBuf,
        req: StartTrainingRequest,
    ) -> Result<()> {
        // reset the per-run controls FIRST: a stale stop_file path would let stop()
        // write into the previous workspace; a stale abort flag would kill this run
        self.inner.abort.store(false, Ordering::SeqCst);
        *self.inner.stop_file.lock() = None;

        match req.backend.as_str() {
            "rvc" => {
                if !matches!(req.version.as_str(), "v1" | "v2") {
                    return Err(UtaiError::Training(format!(
                        "TRAINING_BAD_RVC_VERSION: {}",
                        req.version
                    )));
                }
                if !matches!(req.sample_rate.as_str(), "32k" | "40k" | "48k") {
                    return Err(UtaiError::Training(format!(
                        "TRAINING_BAD_SAMPLE_RATE: {}",
                        req.sample_rate
                    )));
                }
            }
            "sovits" | "sovits_diff" => {
                if !matches!(req.version.as_str(), "4.1" | "4.0") {
                    return Err(UtaiError::Training(format!(
                        "TRAINING_BAD_SOVITS_VERSION: {}",
                        req.version
                    )));
                }
                if req.sample_rate != "44k" {
                    return Err(UtaiError::Training(format!(
                        "TRAINING_SR_FIXED_44K: {}",
                        req.sample_rate
                    )));
                }
                if req.save_every_steps == 0 {
                    return Err(UtaiError::Training("TRAINING_SAVE_INTERVAL_ZERO".into()));
                }
                if req.backend == "sovits_diff" && req.total_steps == 0 {
                    return Err(UtaiError::Training("TRAINING_TOTAL_STEPS_ZERO".into()));
                }
            }
            "sovits_v2" => {
                // S68: VISinger2 backend — its own family/workspace, one fixed
                // version string ("4.0-v2" is what the exported sidecar carries)
                if req.version != "4.0-v2" {
                    return Err(UtaiError::Training(format!(
                        "TRAINING_BAD_SOVITS_VERSION: {}",
                        req.version
                    )));
                }
                if req.sample_rate != "44k" {
                    return Err(UtaiError::Training(format!(
                        "TRAINING_SR_FIXED_44K: {}",
                        req.sample_rate
                    )));
                }
                if req.save_every_steps == 0 {
                    return Err(UtaiError::Training("TRAINING_SAVE_INTERVAL_ZERO".into()));
                }
            }
            "vocoder" => {
                // version is a manifest marker (一期单格式类), not a user choice
                if req.version != "nsf_hifigan" {
                    return Err(UtaiError::Training(format!(
                        "TRAINING_BAD_VOCODER_FORMAT: {}",
                        req.version
                    )));
                }
                if req.sample_rate != "44k" {
                    return Err(UtaiError::Training(format!(
                        "TRAINING_SR_FIXED_44K: {}",
                        req.sample_rate
                    )));
                }
                if req.save_every_steps == 0 {
                    return Err(UtaiError::Training("TRAINING_SAVE_INTERVAL_ZERO".into()));
                }
                if req.total_steps == 0 {
                    return Err(UtaiError::Training("TRAINING_TOTAL_STEPS_ZERO".into()));
                }
                if req.crop_mel_frames == 0 {
                    return Err(UtaiError::Training("TRAINING_CROP_FRAMES_ZERO".into()));
                }
            }
            other => {
                return Err(UtaiError::Training(format!(
                    "TRAINING_BACKEND_UNSUPPORTED: {}",
                    other
                )));
            }
        }
        if req.aug_copies > 3 {
            return Err(UtaiError::Training(format!(
                "TRAINING_AUG_COPIES_MAX: {}",
                req.aug_copies
            )));
        }
        if req.model_name.trim().is_empty() {
            return Err(UtaiError::Training("TRAINING_NAME_EMPTY".into()));
        }

        // ①c multi-speaker co-training (>1 group): the dataset lives in the
        // per-speaker `speakers` files, NOT dataset_files, so validate those and
        // skip the single-speaker empty-dataset gate below. Single-speaker (0 or
        // 1 group) falls through to the byte-identical legacy path.
        let is_multi = req.speakers.len() > 1;
        if is_multi {
            // ①c: multi-speaker co-train = SoVITS (α) + RVC (α′) + SoVITS 4.0-v2
            // (S68, natively multi-speaker upstream). Shallow-diffusion / vocoder
            // stay single-speaker (their loaders assume one speaker).
            if !matches!(req.backend.as_str(), "sovits" | "rvc" | "sovits_v2") {
                return Err(UtaiError::Training("TRAINING_MULTI_BACKEND".into()));
            }
            // RVC emb_g is a FIXED 109-row table (spk_embed_dim in the config templates) — cap
            // the co-train count so a huge set fails loud here, not as an out-of-range train id.
            if req.backend == "rvc" && req.speakers.len() > 109 {
                return Err(UtaiError::Training(format!(
                    "TRAINING_SPEAKER_LIMIT: {}",
                    req.speakers.len()
                )));
            }
            // sovits_v2 keeps the base model's 200-row emb_spk table (n_speakers
            // stays the template 200, upstream v2 semantics) — same loud cap.
            if req.backend == "sovits_v2" && req.speakers.len() > 200 {
                return Err(UtaiError::Training(format!(
                    "TRAINING_SPEAKER_LIMIT: {}",
                    req.speakers.len()
                )));
            }
            let mut seen = std::collections::HashSet::new();
            for sp in &req.speakers {
                let name = sp.name.trim();
                if name.is_empty() {
                    return Err(UtaiError::Training("TRAINING_SPEAKER_NAME_EMPTY".into()));
                }
                if !seen.insert(name.to_string()) {
                    // duplicate display names would collapse the release config's
                    // spk dict (train.py) -> a missing sidecar speaker
                    return Err(UtaiError::Training(format!(
                        "TRAINING_SPEAKER_NAME_DUP: {}",
                        name
                    )));
                }
                if sp.files.is_empty() {
                    return Err(UtaiError::Training(format!(
                        "TRAINING_SPEAKER_NO_DATA: {}",
                        name
                    )));
                }
                for f in &sp.files {
                    if !Path::new(f).is_file() {
                        return Err(UtaiError::Training(format!(
                            "TRAINING_DATA_FILE_MISSING: {}",
                            f
                        )));
                    }
                }
            }
        } else if req.dataset_files.is_empty() {
            // 共享池模式（S41 用户裁定）：浅扩散与主训练共享 dataset/dataset_44k
            // 切片池——宿主工作区已有完整导入时无须重新导入数据。其余后端一律
            // 拒绝（这是防「空数据逃课」的权威闸门，前端禁用只是第一道线）。
            let pool_ok = req.backend == "sovits_diff" && {
                let ws = workspace_path(&data_dir, &req.model_name);
                let family = std::fs::read_to_string(ws.join("run_manifest.json"))
                    .ok()
                    .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                    .map(|m| m["backend"].as_str().unwrap_or("").to_string())
                    .unwrap_or_default();
                family == "sovits" && has_dataset_pool(&ws)
            };
            if !pool_ok {
                return Err(UtaiError::Training(if req.backend == "sovits_diff" {
                    "TRAINING_NO_SHARED_POOL".into()
                } else {
                    "TRAINING_NO_DATA".to_string()
                }));
            }
        }
        for f in &req.dataset_files {
            if !Path::new(f).is_file() {
                return Err(UtaiError::Training(format!(
                    "TRAINING_DATA_FILE_MISSING: {}",
                    f
                )));
            }
        }

        // ---- resolve + verify every asset up front (loud, specific errors) ----
        // Resolution lives in resolve_training_assets — the SINGLE SOURCE shared with the
        // S66 training_required_assets pre-flight command, so the "missing base model"
        // dialog can never drift from what start() actually demands.
        let assets = resolve_training_assets(
            &data_dir,
            &req.backend,
            &req.version,
            &req.sample_rate,
            req.aug_copies,
        )?;
        let ffmpeg = crate::audio::find_ffmpeg()
            .ok_or_else(|| UtaiError::Training("FFMPEG_MISSING".into()))?;
        for (label, p) in &assets.required {
            if !p.is_file() {
                return Err(UtaiError::Training(format!(
                    "TRAINING_ASSET_MISSING: {} -> {}",
                    label,
                    p.display()
                )));
            }
        }
        let ResolvedTrainingAssets {
            contentvec,
            rmvpe_pt,
            pretrain_g,
            pretrain_d,
            nsf_hifigan_model,
            diffusion_pretrain,
            vocoder_pretrain,
            ..
        } = assets;

        let slug = slugify(&req.model_name);
        let workspace = data_dir.join("training").join(&slug);
        let manifest_path = workspace.join("run_manifest.json");
        let family = backend_family(&req.backend).to_string();

        // READ the manifest BEFORE any deletion: the family guard must hold on
        // the fresh path too — a diffusion「重训」must never partial-wipe a
        // same-named RVC workspace (RVC roots also contain G_*.pth, so file
        // heuristics alone cannot tell the families apart).
        let mut old_manifest: Option<serde_json::Value> =
            std::fs::read_to_string(&manifest_path)
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok());
        let old_family = old_manifest
            .as_ref()
            .and_then(|m| m["backend"].as_str())
            .unwrap_or("")
            .to_string();
        if !old_family.is_empty() && old_family != family {
            if req.backend == "sovits_diff" {
                // refuse even on retrain: the diff card's「重训」semantics are
                // "clear diffusion progress", never "sacrifice a foreign
                // workspace" — the user meant a different model name
                return Err(UtaiError::Training(format!(
                    "WORKSPACE_BACKEND_MISMATCH: {}",
                    old_family
                )));
            }
            if !req.fresh {
                return Err(UtaiError::Training(format!(
                    "WORKSPACE_BACKEND_MISMATCH: {} -> {}",
                    old_family, family
                )));
            }
            // main backends keep the S37 behavior: retrain wipes with user consent
        }
        // a diff resume must never silently colonize a manifest-less workspace
        // — its family is unknowable, and the diff pipeline would then slice /
        // flist / extract INTO whatever那是 (红队 A2)
        if req.backend == "sovits_diff"
            && !req.fresh
            && workspace.exists()
            && old_manifest.is_none()
        {
            return Err(UtaiError::Training("WORKSPACE_MANIFEST_MISSING".into()));
        }

        let has_main = has_main_progress(&workspace);
        // a manifest-less workspace that still holds checkpoints is an anomaly
        // (every run since S37 writes the manifest before spawning): resuming
        // into it would let e.g. 4.1 weights stream into a 4.0 graph through
        // the tolerant checkpoint loader — silently degrading to near-scratch
        // while claiming「续训」. Refuse loudly; retrain wipes it.
        if !req.fresh && workspace.exists() && old_manifest.is_none() && has_main {
            return Err(UtaiError::Training("WORKSPACE_MANIFEST_MISSING".into()));
        }
        // vocoder twin: a manifest-less workspace holding lightning checkpoints
        // would let get_latest_checkpoint_path resume into it AND silently skip
        // the finetune base seeding (setup() only loads the base when no ckpt
        // exists) — the S39 尾修 4 lineage of "quiet fake resume"
        if req.backend == "vocoder"
            && !req.fresh
            && workspace.exists()
            && old_manifest.is_none()
            && max_vocoder_ckpt_step(&workspace).is_some()
        {
            return Err(UtaiError::Training("WORKSPACE_MANIFEST_MISSING".into()));
        }
        // the diff「重训」only clears diffusion/ when a live main model shares
        // the workspace — everything else is a full wipe
        let diff_partial_wipe =
            req.fresh && req.backend == "sovits_diff" && workspace.exists()
                && old_manifest.is_some() && has_main;

        // resume-parameter guard: version/sample_rate (and for the sovits main
        // model the vol_embedding switch — it changes the model architecture AND
        // the wire inputs) are baked into the workspace artifacts; for a diff
        // run the version additionally pins the ContentVec space of the cached
        // .soft.pt files and of the main model the attachment will pair with.
        // It runs for resumes AND for the diff partial wipe — the partial wipe
        // keeps the manifest, so a mismatched version could never train
        // afterwards anyway; deleting first would destroy hours of diffusion
        // progress and THEN refuse (review F0).
        if !req.fresh || diff_partial_wipe {
            if let Some(old) = old_manifest.as_ref() {
                let old_ver = old["version"].as_str().unwrap_or("");
                let old_sr = old["sample_rate"].as_str().unwrap_or("");
                if (!old_ver.is_empty() && old_ver != req.version)
                    || (!old_sr.is_empty() && old_sr != req.sample_rate)
                {
                    return Err(UtaiError::Training(
                        if req.backend == "sovits_diff" && has_main {
                            // 重训(仅扩散) cannot unlock the version here — it is
                            // pinned by the main model; don't suggest it
                            format!(
                                "DIFF_VERSION_MISMATCH: {}/{} -> {}/{}",
                                old_ver, old_sr, req.version, req.sample_rate
                            )
                        } else {
                            format!(
                                "RESUME_PARAMS_MISMATCH: {}/{} -> {}/{}",
                                old_ver, old_sr, req.version, req.sample_rate
                            )
                        },
                    ));
                }
                if req.backend == "sovits" {
                    if let Some(old_vol) = old["vol_embedding"].as_bool() {
                        if old_vol != req.vol_embedding {
                            return Err(UtaiError::Training(format!(
                                "RESUME_VOL_EMBEDDING_MISMATCH: {} -> {}",
                                if old_vol { "on" } else { "off" },
                                if req.vol_embedding { "on" } else { "off" }
                            )));
                        }
                    }
                }
                // ①c: n_speakers + the ordered speaker set are baked into the emb_g
                // rows — resuming with a different count / order / set would silently
                // mis-assign every speaker's timbre. Immutable for SoVITS (α), RVC
                // (α′) and SoVITS 4.0-v2 (S68 emb_spk). (Old single-speaker manifests
                // have no n_speakers key -> 1, which matches a single-speaker resume
                // = no false rejection.)
                if matches!(req.backend.as_str(), "sovits" | "rvc" | "sovits_v2") {
                    let old_n = old["n_speakers"].as_u64().unwrap_or(1);
                    let cur_n = if req.speakers.len() > 1 {
                        req.speakers.len() as u64
                    } else {
                        1
                    };
                    if old_n != cur_n {
                        return Err(UtaiError::Training(format!(
                            "RESUME_SPEAKER_COUNT_MISMATCH: {} -> {}",
                            old_n, cur_n
                        )));
                    }
                    if cur_n > 1 {
                        let old_slugs: Vec<String> = old["speakers"]
                            .as_array()
                            .map(|a| {
                                a.iter()
                                    .filter_map(|v| v.as_str().map(String::from))
                                    .collect()
                            })
                            .unwrap_or_default();
                        let cur_slugs: Vec<String> = assign_speaker_slugs(&req.speakers)
                            .into_iter()
                            .map(|(_, s)| s)
                            .collect();
                        if old_slugs != cur_slugs {
                            return Err(UtaiError::Training(
                                "RESUME_SPEAKER_SET_MISMATCH".into(),
                            ));
                        }
                    }
                }
                // k_step_max pins the diffusion TRAINING distribution (t ~
                // [0,k)) and the exported sidecar contract — resuming across a
                // flip would silently blend two objectives (review F6). The
                // fresh partial-wipe path resets the progress, so it may change.
                if req.backend == "sovits_diff" && !req.fresh {
                    if let (Some(old_k), Some(max_step)) = (
                        old["diff_k_step_max"].as_u64(),
                        max_diffusion_step(&workspace),
                    ) {
                        if max_step > 0 && old_k != req.k_step_max as u64 {
                            let show = |k: u64| {
                                if k == 0 { "full-diffusion".to_string() } else { k.to_string() }
                            };
                            return Err(UtaiError::Training(format!(
                                "RESUME_KSTEP_MISMATCH: {} -> {}",
                                show(old_k),
                                show(req.k_step_max as u64)
                            )));
                        }
                    }
                }
            }
        }

        if req.fresh && workspace.exists() {
            if diff_partial_wipe {
                // diffusion retrain inside a live main-model workspace: clear
                // ONLY the diffusion progress — the main checkpoints and the
                // shared preprocessing caches survive
                let diff_dir = workspace.join("diffusion");
                if diff_dir.exists() {
                    std::fs::remove_dir_all(&diff_dir).map_err(|e| {
                        UtaiError::Training(format!("DIFF_WIPE_FAILED: {}", e))
                    })?;
                }
            } else {
                // main retrain / diff-only workspace (a full wipe here is what
                // unlocks a version change) / manifest-less anomaly
                std::fs::remove_dir_all(&workspace)
                    .map_err(|e| UtaiError::Training(format!("WORKSPACE_WIPE_FAILED: {}", e)))?;
                old_manifest = None;
            }
        }
        std::fs::create_dir_all(&workspace)?;

        // resume dead-end guard: a resume whose target步数 is already reached
        // would "complete" instantly without training a step (S37 的续训 config
        // 校验同族坑) — refuse loudly so the user fixes 总步数 first
        if req.backend == "sovits_diff" && !req.fresh {
            if let Some(max_step) = max_diffusion_step(&workspace) {
                if max_step > 0 && max_step >= req.total_steps as u64 {
                    return Err(UtaiError::Training(format!(
                        "RESUME_TARGET_REACHED_DIFF: {} >= {}",
                        max_step, req.total_steps
                    )));
                }
            }
        }
        // vocoder twin of the guard — ckpt numbers are GLOBAL (2× real), the
        // //2 here is exactly the ×2-class bug the design flagged (红队 A8)
        if req.backend == "vocoder" && !req.fresh {
            if let Some(max_global) = max_vocoder_ckpt_step(&workspace) {
                let real = max_global / 2;
                if real > 0 && real >= req.total_steps as u64 {
                    return Err(UtaiError::Training(format!(
                        "RESUME_TARGET_REACHED_VOCODER: {} >= {}",
                        real, req.total_steps
                    )));
                }
            }
        }

        // merge-write: a diff run must not drop the main run's fields (the
        // vol_embedding guard above dies silently if its key vanishes) and
        // vice versa — read-modify-write, never rebuild from scratch
        let mut manifest = match old_manifest {
            Some(m @ serde_json::Value::Object(_)) => m,
            _ => serde_json::json!({}),
        };
        manifest["backend"] = serde_json::json!(family);
        manifest["version"] = serde_json::json!(req.version);
        manifest["sample_rate"] = serde_json::json!(req.sample_rate);
        if req.backend == "sovits" {
            manifest["vol_embedding"] = serde_json::json!(req.vol_embedding);
            // recorded so a later diff run inherits it (a loudnorm flip would
            // wipe the shared caches AND desync the diffusion training domain
            // from the main model's)
            manifest["loudnorm"] = serde_json::json!(req.loudnorm);
        }
        // ①c: freeze the speaker count + ordered slug set (resume-immutable, guarded above)
        // for SoVITS (α), RVC (α′) and SoVITS 4.0-v2 (S68). Only written for a genuine
        // co-training (>1) so a single-speaker manifest stays byte-identical to pre-①c.
        if matches!(req.backend.as_str(), "sovits" | "rvc" | "sovits_v2") && req.speakers.len() > 1 {
            let assigned = assign_speaker_slugs(&req.speakers);
            let slugs: Vec<String> = assigned.iter().map(|(_, s)| s.clone()).collect();
            let names: Vec<String> = assigned.iter().map(|(n, _)| n.clone()).collect();
            manifest["n_speakers"] = serde_json::json!(slugs.len());
            manifest["speakers"] = serde_json::json!(slugs);
            // ①c: display NAMES too. The manifest is merge-preserved across a later sovits_diff run
            // (which reuses this workspace and OVERWRITES run.json WITHOUT a speakers key) — so the
            // resume config-diff must read names from HERE, not run.json, or it would falsely report
            // a speaker mismatch after any diffusion run and block a valid multi-speaker resume.
            manifest["speaker_names"] = serde_json::json!(names);
        }
        if req.backend != "sovits_diff" {
            // S41: recorded for every non-diff backend; the sovits value is
            // the diff inheritance source (shared dataset_44k slice pool),
            // rvc/vocoder entries are informational
            manifest["aug_copies"] = serde_json::json!(req.aug_copies);
        }
        if req.backend == "sovits_diff" {
            manifest["diff_k_step_max"] = serde_json::json!(req.k_step_max);
        }

        // diff runs inherit the dataset-affecting switches from the manifest —
        // their own request never carries them
        let eff_vol_embedding = if req.backend == "sovits_diff" {
            manifest["vol_embedding"].as_bool().unwrap_or(false)
        } else {
            req.vol_embedding
        };
        let eff_loudnorm = if req.backend == "sovits_diff" {
            match manifest["loudnorm"].as_bool() {
                Some(v) => v,
                None => {
                    // S38-era manifests predate the loudnorm field. Recover the
                    // value the caches were actually built with from the stored
                    // fingerprint text ("<hash>|enc=..|loudnorm=N") — guessing
                    // false would wipe the shared caches AND train the companion
                    // on a different loudness domain than the main model
                    // (review F1); backfilled into the manifest so the next
                    // main resume doesn't re-wipe either.
                    let v = std::fs::read_to_string(workspace.join("dataset.fingerprint"))
                        .map(|s| s.contains("|loudnorm=1"))
                        .unwrap_or(false);
                    manifest["loudnorm"] = serde_json::json!(v);
                    v
                }
            }
        } else {
            req.loudnorm
        };
        // pure inheritance, NO rejection branch (loudnorm posture; a missing
        // key = pre-S41 or diff-first workspace = 0). The diff pipeline runs
        // the same augment stage with this value so a cache-wipe rebuild
        // regenerates the aug slices the manifest promises.
        let eff_aug_copies = if req.backend == "sovits_diff" {
            manifest["aug_copies"].as_u64().unwrap_or(0) as u32
        } else {
            req.aug_copies
        };
        std::fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
        // milestone cadence normalized onto the save grid (see field docs)
        let interval_val = req.save_every_steps.max(1);
        let interval_force_save =
            ((req.interval_force_save.max(1) + interval_val - 1) / interval_val) * interval_val;

        let stop_file = workspace.join("stop.flag");
        let _ = std::fs::remove_file(&stop_file); // stale flag would insta-stop the run

        // ---- reset run state ----
        {
            let mut s = self.inner.snapshot.lock();
            *s = TrainingSnapshot {
                state: "starting".into(),
                backend: req.backend.clone(),
                model_name: req.model_name.clone(),
                model_slug: slug.clone(),
                workspace: workspace.to_string_lossy().into_owned(),
                total_epochs: req.total_epoch,
                // ①c: freeze the run's speaker names (id order) for the audition picker; empty
                // for a single-speaker run (len ≤ 1) so nothing changes there.
                speakers: if req.speakers.len() > 1 {
                    req.speakers.iter().map(|sp| sp.name.clone()).collect()
                } else {
                    Vec::new()
                },
                ..Default::default()
            };
        }
        self.inner.history.lock().clear();
        self.inner.stderr_ring.lock().clear();
        *self.inner.stop_file.lock() = Some(stop_file.clone());
        *self.inner.started_at.lock() = Some(Instant::now());

        let ctx = RunCtx {
            ffmpeg,
            contentvec,
            rmvpe_pt,
            pretrain_g,
            pretrain_d,
            nsf_hifigan_model,
            diffusion_pretrain,
            vocoder_pretrain,
            vol_embedding: eff_vol_embedding,
            loudnorm: eff_loudnorm,
            interval_force_save,
            aug_copies: eff_aug_copies,
        };
        let inner = Arc::clone(&self.inner);
        let app_dir = self.app_dir.clone();
        std::thread::Builder::new()
            .name("training-run".into())
            .spawn(move || {
                let outcome = run_worker(
                    &inner, &app, &app_dir, &data_dir, &workspace, &stop_file, &req, &ctx, &slug,
                );
                if let Err(e) = outcome {
                    finalize_elapsed(&inner);
                    let tail = stderr_tail(&inner);
                    let mut s = inner.snapshot.lock();
                    s.state = "error".into();
                    s.error = Some(e.to_string());
                    s.stderr_tail = tail;
                    drop(s);
                    tracing::error!("training run failed: {}", e);
                    emit_done(&inner, &app);
                }
                finalize_elapsed(&inner); // idempotent — freezes elapsed on every exit path
                let _ = std::fs::remove_file(&stop_file);
                *inner.child.lock() = None;
                inner.running.store(false, Ordering::SeqCst);
            })
            .map_err(|e| UtaiError::Training(format!("TRAINING_THREAD_SPAWN_FAILED: {}", e)))?;
        Ok(())
    }
}

/// Freeze the final elapsed time into the snapshot and stop the live clock.
/// Idempotent (take()) — safe to call from every exit path.
fn finalize_elapsed(inner: &Inner) {
    if let Some(t) = inner.started_at.lock().take() {
        inner.snapshot.lock().elapsed_secs = t.elapsed().as_secs();
    }
}

/// Pre-spawn abort exit: the run never (or barely) reached python; report a clean
/// "stopped" so the frontend leaves the running state.
fn abort_finish(inner: &Arc<Inner>, app: &tauri::AppHandle) -> Result<()> {
    finalize_elapsed(inner);
    inner.snapshot.lock().state = "stopped".into();
    emit_done(inner, app);
    tracing::warn!("training aborted before/at sidecar spawn");
    Ok(())
}

fn stderr_tail(inner: &Inner) -> Vec<String> {
    inner
        .stderr_ring
        .lock()
        .iter()
        .rev()
        .take(30)
        .rev()
        .cloned()
        .collect()
}

fn emit_done(inner: &Inner, app: &tauri::AppHandle) {
    let snap = inner.snapshot.lock().clone();
    let _ = app.emit("training-done", &snap);
}

/// Everything try_start resolves for the sidecar run: asset paths plus the
/// values a diff run inherits from the workspace manifest.
struct RunCtx {
    ffmpeg: PathBuf,
    contentvec: PathBuf,
    rmvpe_pt: PathBuf,
    /// empty for sovits_diff (no G/D pair — the diffusion base seeds instead)
    pretrain_g: PathBuf,
    pretrain_d: PathBuf,
    /// sovits_diff only: the torch NSF-HiFiGAN ckpt (the diffusion mel recipe)
    nsf_hifigan_model: PathBuf,
    /// sovits_diff only; empty = train from scratch (no vec256 base exists)
    diffusion_pretrain: PathBuf,
    /// vocoder only: the classic NSF-HiFiGAN finetune base (lightning ckpt, G+D)
    vocoder_pretrain: PathBuf,
    /// effective values (manifest-inherited for sovits_diff)
    vol_embedding: bool,
    loudnorm: bool,
    /// normalized to a multiple of save_every_steps
    interval_force_save: u32,
    /// S41 effective augmentation copies (manifest-inherited for sovits_diff)
    aug_copies: u32,
}

#[allow(clippy::too_many_arguments)]
fn run_worker(
    inner: &Arc<Inner>,
    app: &tauri::AppHandle,
    app_dir: &Path,
    _data_dir: &Path,
    workspace: &Path,
    stop_file: &Path,
    req: &StartTrainingRequest,
    ctx: &RunCtx,
    slug: &str,
) -> Result<()> {
    // ---- stage: import the dataset into the ASCII workspace ----
    let dataset_dir = workspace.join("dataset");
    // ①c: >1 speaker group = per-speaker subdir import; else the pre-①c flat
    // (or shared-pool) path, verbatim. run_speakers is filled only for multi
    // and becomes run.json "speakers" so the sovits/rvc pipeline co-trains them.
    let is_multi = req.speakers.len() > 1;
    let mut run_speakers: Vec<serde_json::Value> = Vec::new();
    if is_multi {
        // import EACH speaker's files into dataset/<slug>/ (000..N per speaker,
        // sorted — same deterministic-order + fingerprint rationale as the flat
        // path). The pipeline slices each subdir into dataset_44k/<slug> and the
        // loader derives the emb_g id from the dir name, so these slugs MUST
        // match the manifest — assign_speaker_slugs is deterministic on the
        // same request.
        if dataset_dir.exists() {
            std::fs::remove_dir_all(&dataset_dir)?;
        }
        std::fs::create_dir_all(&dataset_dir)?;
        let assigned = assign_speaker_slugs(&req.speakers);
        let total: usize = req.speakers.iter().map(|s| s.files.len()).sum();
        let mut done = 0usize;
        for (gi, (name, slug)) in assigned.iter().enumerate() {
            let sub = dataset_dir.join(slug);
            std::fs::create_dir_all(&sub)?;
            let mut files = req.speakers[gi].files.clone();
            files.sort();
            for (i, f) in files.iter().enumerate() {
                if inner.abort.load(Ordering::SeqCst) {
                    return abort_finish(inner, app);
                }
                let src = Path::new(f);
                let ext = src
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("wav")
                    .to_ascii_lowercase();
                let dst = sub.join(format!("{:03}.{}", i, ext));
                std::fs::copy(src, &dst).map_err(|e| {
                    UtaiError::Training(format!(
                        "TRAINING_IMPORT_COPY_FAILED: {}: {}",
                        src.display(),
                        e
                    ))
                })?;
                done += 1;
                let stage = StageInfo {
                    stage: "import".into(),
                    done: Some(done as u64),
                    total: Some(total as u64),
                    progress: Some(done as f32 / total.max(1) as f32),
                    message: src.file_name().map(|n| n.to_string_lossy().into_owned()),
                };
                inner.snapshot.lock().stage = Some(stage.clone());
                let _ = app.emit("training-stage", &stage);
            }
            run_speakers.push(serde_json::json!({
                "name": name,
                "slug": slug,
                "dataset_dir": sub,
            }));
        }
    } else {
        if req.dataset_files.is_empty() {
            // shared-pool reuse (only sovits_diff reaches here — start() validated
            // the pool): dataset/ and dataset.fingerprint stay UNTOUCHED, so the
            // python side reads an unchanged dataset and takes the cache-reuse
            // path — wiping here would destroy the very pool being shared
            let stage = StageInfo {
                stage: "import".into(),
                done: Some(1),
                total: Some(1),
                progress: Some(1.0),
                message: Some("SHARED_POOL_REUSED".into()),
            };
            inner.snapshot.lock().stage = Some(stage.clone());
            let _ = app.emit("training-stage", &stage);
        } else {
            if dataset_dir.exists() {
                std::fs::remove_dir_all(&dataset_dir)?;
            }
            std::fs::create_dir_all(&dataset_dir)?;
        }
        // deterministic import order: the workspace copies are named 000..N in
        // list order and the extraction-cache fingerprint hashes name+content, so
        // the same SELECTION re-picked in a different dialog order must not read
        // as "dataset changed" (which would silently re-extract everything —
        // exactly the cache-reuse promise the diffusion card is built on)
        let mut dataset_files = req.dataset_files.clone();
        dataset_files.sort();
        let total = dataset_files.len();
        for (i, f) in dataset_files.iter().enumerate() {
            if inner.abort.load(Ordering::SeqCst) {
                return abort_finish(inner, app);
            }
            let src = Path::new(f);
            let ext = src
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("wav")
                .to_ascii_lowercase();
            let dst = dataset_dir.join(format!("{:03}.{}", i, ext));
            std::fs::copy(src, &dst).map_err(|e| {
                UtaiError::Training(format!(
                    "TRAINING_IMPORT_COPY_FAILED: {}: {}",
                    src.display(),
                    e
                ))
            })?;
            let stage = StageInfo {
                stage: "import".into(),
                done: Some((i + 1) as u64),
                total: Some(total as u64),
                progress: Some((i + 1) as f32 / total as f32),
                message: src.file_name().map(|n| n.to_string_lossy().into_owned()),
            };
            inner.snapshot.lock().stage = Some(stage.clone());
            let _ = app.emit("training-stage", &stage);
        }
    }

    // resolve the training interpreter + its device backend up front: dev venv →
    // installed runtime pack (the pack VARIANT fixes the backend) → manual slot →
    // bare python. device_backend feeds device.py's shim via run.json below.
    let (python, device_backend) = crate::pyenv::training_interpreter(app_dir, req.force_cpu);
    // Silent GPU→CPU downgrade guard (S43 review): the user didn't force CPU (so they
    // asked for their GPU) yet the resolved backend is "cpu" — the ONLY way that happens
    // is a CPU-only runtime pack installed on a GPU box (the dev venv and every GPU pack
    // yield "cuda"). Silent slow-training is the user's #1 pain, so say it loudly (the
    // fuller "gate the GPU option / prompt to install the GPU pack" is the release
    // gating dialog — see pending_cleanups).
    if !req.force_cpu && device_backend == "cpu" {
        tracing::warn!(
            "Training runtime is the CPU variant: this run will train on CPU (slow). For GPU training install the runtime pack matching your GPU in Settings → Training Environment."
        );
    }

    // ---- run config for the sidecar ----
    let mut run_config = serde_json::json!({
        "backend": req.backend,
        "workspace": workspace,
        "dataset_dir": dataset_dir,
        "model_slug": slug,
        "model_name": req.model_name,
        "sample_rate": req.sample_rate,
        "version": req.version,
        "total_epoch": req.total_epoch,
        "batch_size": req.batch_size,
        "save_every_epoch": req.save_every_epoch,
        "save_every_weights": req.save_every_weights,
        "keep_only_latest": req.keep_only_latest,
        "cache_gpu": req.cache_gpu,
        // sovits_v2 is pure fp32 (upstream VISinger2 has no amp) — the switch
        // is hidden in the UI and normalized off here, belt and suspenders
        "fp16": if req.backend == "sovits_v2" { false } else { req.fp16 },
        "spk_id": req.spk_id,
        // sovits-only knobs (the rvc pipeline ignores them); vol_embedding /
        // loudnorm are the EFFECTIVE values (manifest-inherited for diff runs)
        "vol_embedding": ctx.vol_embedding,
        "loudnorm": ctx.loudnorm,
        // S41 augmentation copies — the EFFECTIVE value (manifest-inherited
        // for diff runs); every pipeline reads it uniformly
        "aug_copies": ctx.aug_copies,
        "kmeans": req.kmeans,
        "save_every_steps": req.save_every_steps,
        "keep_ckpts": req.keep_ckpts,
        "all_in_mem": req.all_in_mem,
        // sovits_diff-only knobs (ignored by the other pipelines)
        "total_steps": req.total_steps,
        "k_step_max": req.k_step_max,
        "interval_force_save": ctx.interval_force_save,
        "cache_all_data": req.cache_all_data,
        // vocoder-only knobs (ignored by the other pipelines)
        "crop_mel_frames": req.crop_mel_frames,
        "freeze_mpd": req.freeze_mpd,
        "seed": SEED,
        // Windows cannot hold an EMPTY env var (empty = deleted = all GPUs
        // visible) — CPU mode must be the explicit sentinel "-1". Otherwise the
        // value is the accelerator-native identity (NVIDIA UUID / vendor index)
        // from training_gpus; "" = auto (setup_visibility leaves it unset).
        "gpu": if req.force_cpu { "-1".to_string() } else { req.gpu.clone() },
        // device.py's shim reads this BEFORE torch import (visibility) and to pick
        // autocast/scaler. Sourced from the resolved interpreter: dev venv → the box's
        // GPU (cuda) or force_cpu; installed pack → its variant (nv-cu130/amd->cuda,
        // xpu->xpu, cpu->cpu). Absent field => shim defaults to cuda-with-availability-
        // fallback, so a pre-Phase-B run.json stays valid.
        "device_backend": device_backend,
        "stop_file": stop_file,
        "pretrain_g": ctx.pretrain_g,
        "pretrain_d": ctx.pretrain_d,
        "assets": {
            "ffmpeg": ctx.ffmpeg,
            "rmvpe_pt": ctx.rmvpe_pt,
            "contentvec_onnx": ctx.contentvec,
            // family, not backend: sovits_diff shares the sovits templates
            "configs_dir": app_dir.join("training").join("assets").join("configs").join(backend_family(&req.backend)),
            "mute_dir": app_dir.join("training").join("assets").join("mute"),
            "nsf_hifigan_model": ctx.nsf_hifigan_model,
            "diffusion_pretrain": ctx.diffusion_pretrain,
            "vocoder_pretrain": ctx.vocoder_pretrain,
        },
    });
    // ①c: the sovits pipeline's resolve_speakers reads this array for co-training;
    // single-speaker omits the key entirely -> pipeline falls back to
    // dataset_dir/model_slug = byte-identical run.json / behavior.
    if is_multi {
        run_config["speakers"] = serde_json::json!(run_speakers);
    }
    let run_json = workspace.join("run.json");
    std::fs::write(&run_json, serde_json::to_vec_pretty(&run_config)?)?;

    // ---- spawn the sidecar ----
    if inner.abort.load(Ordering::SeqCst) {
        return abort_finish(inner, app);
    }
    let training_dir = app_dir.join("training");
    // `python` was resolved above by training_interpreter (dev venv / installed pack)
    tracing::info!(
        "spawning training sidecar: {} -m utai_train.runner --config {}",
        python.display(),
        run_json.display()
    );
    let mut child = crate::util::python_command(&python)
        .current_dir(&training_dir)
        .arg("-m")
        .arg("utai_train.runner")
        .arg("--config")
        .arg(&run_json)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| {
            UtaiError::Training(format!(
                "TRAINING_PYTHON_SPAWN_FAILED: {}: {}",
                python.display(),
                e
            ))
        })?;

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    {
        // slot-or-die: force_stop sets abort THEN drains the slot, so under any
        // interleaving either we see abort here and kill the fresh child, or the
        // slotted child is visible to force_stop's kill
        let mut slot = inner.child.lock();
        if inner.abort.load(Ordering::SeqCst) {
            drop(slot);
            let _ = child.kill();
            let _ = child.wait();
            return abort_finish(inner, app);
        }
        *slot = Some(child);
    }
    {
        let mut s = inner.snapshot.lock();
        s.state = "running".into();
    }
    let _ = app.emit("training-state", "running");

    // stderr → ring buffer (surfaced on abnormal exit) + debug tracing
    if let Some(stderr) = stderr {
        let ring_inner = Arc::clone(inner);
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(|l| l.ok()) {
                tracing::debug!(target: "utai", "[train-py] {}", line);
                let mut ring = ring_inner.stderr_ring.lock();
                if ring.len() >= STDERR_RING_CAP {
                    ring.pop_front();
                }
                ring.push_back(line);
            }
        });
    }

    // stdout protocol loop (this thread)
    let mut got_done = false;
    let mut got_error: Option<String> = None;
    if let Some(stdout) = stdout {
        for line in BufReader::new(stdout).lines().map_while(|l| l.ok()) {
            let Ok(msg) = serde_json::from_str::<serde_json::Value>(&line) else {
                tracing::debug!(target: "utai", "[train-proto?] {}", line);
                continue;
            };
            match msg.get("type").and_then(|t| t.as_str()) {
                Some("stage") => {
                    let stage = StageInfo {
                        stage: msg["stage"].as_str().unwrap_or("").to_string(),
                        done: msg["done"].as_u64(),
                        total: msg["total"].as_u64(),
                        progress: msg["progress"].as_f64().map(|p| p as f32),
                        message: msg["message"].as_str().map(str::to_string),
                    };
                    inner.snapshot.lock().stage = Some(stage.clone());
                    let _ = app.emit("training-stage", &stage);
                }
                Some("step") => {
                    let losses: HashMap<String, f64> = msg["losses"]
                        .as_object()
                        .map(|o| {
                            o.iter()
                                .filter_map(|(k, v)| v.as_f64().map(|f| (k.clone(), f)))
                                .collect()
                        })
                        .unwrap_or_default();
                    let step = StepInfo {
                        step: msg["step"].as_u64().unwrap_or(0),
                        total_steps: msg["total_steps"].as_u64().unwrap_or(0),
                        epoch: msg["epoch"].as_u64().unwrap_or(0) as u32,
                        total_epochs: msg["total_epochs"].as_u64().unwrap_or(0) as u32,
                        lr: msg["lr"].as_f64().unwrap_or(0.0),
                        losses: losses.clone(),
                        eta_secs: msg["eta_secs"].as_u64(),
                    };
                    {
                        let mut hist = inner.history.lock();
                        if hist.len() >= HISTORY_CAP {
                            // thin to half; the curve keeps its shape, memory stays bounded
                            let thinned: Vec<StepPoint> =
                                hist.iter().step_by(2).cloned().collect();
                            *hist = thinned;
                        }
                        hist.push(StepPoint {
                            step: step.step,
                            lr: step.lr,
                            losses,
                        });
                    }
                    inner.snapshot.lock().step = Some(step.clone());
                    let _ = app.emit("training-step", &step);
                }
                Some("ckpt") => {
                    let ckpt = CkptInfo {
                        kind: msg["kind"].as_str().unwrap_or("").to_string(),
                        path: msg["path"].as_str().unwrap_or("").to_string(),
                        step: msg["step"].as_u64().unwrap_or(0),
                        epoch: msg["epoch"].as_u64().unwrap_or(0) as u32,
                        metric: msg["metric"].as_f64(),
                    };
                    {
                        let mut s = inner.snapshot.lock();
                        // best/final overwrite their previous entry; periodics accumulate
                        if ckpt.kind == "best" || ckpt.kind == "final" {
                            s.ckpts.retain(|c| c.kind != ckpt.kind);
                        }
                        s.ckpts.push(ckpt.clone());
                    }
                    let _ = app.emit("training-ckpt", &ckpt);
                }
                Some("done") => {
                    got_done = true;
                    let reason = msg["reason"].as_str().unwrap_or("completed");
                    let mut s = inner.snapshot.lock();
                    s.state = if reason == "stopped" { "stopped" } else { "completed" }.into();
                    s.summary = Some(msg["summary"].clone());
                }
                Some("error") => {
                    got_error = Some(
                        msg["message"]
                            .as_str()
                            .unwrap_or("TRAINING_UNKNOWN_ERROR")
                            .to_string(),
                    );
                }
                _ => tracing::debug!(target: "utai", "[train-proto?] {}", line),
            }
        }
    }

    // ---- child exit ----
    // take the child OUT before waiting — wait() must not hold the lock (force_stop
    // and the quit flow would otherwise block on it during the exit window)
    let mut child_opt = inner.child.lock().take();
    let status = match child_opt.as_mut() {
        Some(child) => child.wait().ok(),
        None => None, // force-killed (slot drained by force_stop)
    };
    let code = status.and_then(|s| s.code());

    if got_done {
        finalize_elapsed(inner);
        emit_done(inner, app);
        tracing::info!("training run finished ({:?})", inner.snapshot.lock().state);
        return Ok(());
    }
    if let Some(err) = got_error {
        return Err(UtaiError::Training(err));
    }
    // no protocol verdict at all — crashed / killed externally. BE LOUD.
    if status.is_none() {
        finalize_elapsed(inner);
        let mut s = inner.snapshot.lock();
        s.state = "stopped".into();
        drop(s);
        emit_done(inner, app);
        tracing::warn!("training force-stopped by user");
        return Ok(());
    }
    Err(UtaiError::Training(format!(
        "TRAINING_PROCESS_CRASHED: exit code {:?}",
        code
    )))
}
