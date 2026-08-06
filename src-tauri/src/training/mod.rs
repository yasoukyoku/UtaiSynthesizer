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

pub mod dsmanifest;
pub mod resume_lock;
pub mod tproject;

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

/// S68b loud-degradation preflight (community RTX 3080 report: GPU box + CPU-only
/// runtime pack silently trained on CPU; the only warn was log-file-only AND gated
/// behind !force_cpu, so nobody ever saw it). Refuses with TRAINING_RUNTIME_CPU_ONLY
/// — trilingual message names both ways out (install the matching GPU pack / check
/// 强制 CPU 训练). Fires ONLY when a GPU runtime pack is actually offerable on this
/// box (variant_supported): Pascal/cc<7.5 NVIDIA, TheRock-unsupported AMD and non-Arc
/// Intel machines have no in-app pack to install, so they keep training on CPU exactly
/// as before (review round 1: the unconditional refusal sent those users chasing a
/// download the Settings UI deliberately hides). The nvidia-smi/DXGI probes only run
/// on the rare cpu-pack path — zero cost for every GPU-pack install.
fn refuse_cpu_only_runtime(app_dir: &Path, force_cpu: bool) -> Result<()> {
    if force_cpu {
        return Ok(());
    }
    let (_python, device_backend) = crate::pyenv::training_interpreter(app_dir, false);
    if device_backend != "cpu" {
        return Ok(());
    }
    let gpus = crate::commands::settings::query_gpu_adapters();
    let nv_cc10 = crate::commands::settings::nvidia_compute_caps_cc10();
    let offerable = ["nv-cu130", "amd", "xpu"]
        .iter()
        .any(|v| crate::commands::settings::variant_supported(v, &gpus, &nv_cc10));
    if offerable {
        return Err(UtaiError::Training("TRAINING_RUNTIME_CPU_ONLY".into()));
    }
    Ok(())
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
            // CC BY-NC-SA weights — never bundled, but S75 made them
            // pack-distributed (`training-vocoder`, mirrored + license-badged;
            // the label no longer has to carry download instructions).
            // ⚠️ NOT interchangeable with the aux default vocoder onnx: that
            // one is generator-only and a whole release older (2022.12).
            // ContentVec/RMVPE/configs/mute are NOT used by this pipeline
            // (设计红队 A17: required 收敛进各臂).
            vocoder_pretrain = data_dir
                .join("models")
                .join("training")
                .join("vocoder")
                .join("nsf_hifigan_44.1k_hop512_128bin_2024.02.ckpt");
            required.push((
                "vocoder finetune base ckpt (NSF-HiFiGAN 2024.02, CC BY-NC-SA 4.0)".into(),
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
    /// S76 batch 4: WHICH PROJECT this run trains into. Empty = resolve by `model_name`, the
    /// pre-batch-4 behaviour.
    ///
    /// It has to be explicit. `model_name` became「本次训练名」— editable, defaulting to the
    /// project name but free to differ — and it is also the artifact identity
    /// (`slugify(model_name)` = `hps.name` / `weights/<slug>*`). Resolving the DIRECTORY from
    /// it too would mean that renaming a run forks a second project: the old checkpoints keep
    /// existing with nothing able to reach them, and `find_by_name` then picks between two
    /// same-named projects by directory order.
    #[serde(default)]
    pub project_id: String,
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
    /// The UI id of the picked device (`TrainingGpu.id`, e.g. `nvidia:GPU-8a2c…` / `amd:0`);
    /// "" = auto (leave visibility unset → torch's own default device).
    /// ⚠ S75: this is an IDENTITY, not the device mask. try_start resolves it against a freshly
    /// built device list and takes the mask from THAT entry — the mask alone is only unique
    /// within a vendor, so a mask-keyed payload silently resolved to the wrong card on a
    /// multi-vendor box. S67 (the ancestor of that bug): it was once a raw WMI adapter index, and
    /// on an iGPU+NVIDIA box SELECTING the NVIDIA card masked every GPU and training fell back to
    /// CPU silently.
    #[serde(default)]
    pub gpu: String,
    #[serde(default)]
    pub force_cpu: bool,
    #[serde(default)]
    pub spk_id: u32,
    /// true = 重训 (wipe the workspace), false = 续训 (resume from latest ckpt)
    #[serde(default)]
    pub fresh: bool,
    /// The user answered a destructive-wipe dialog with「重训」for THIS run. Fail-closed
    /// (`#[serde(default)]` = false): a caller that forgets the field gets a loud refusal,
    /// never a silent wipe.
    ///
    /// Why this exists: `fresh` alone could not tell「用户按了重训」from「前端探测挂了,于是
    /// 默认当作全新」. onStart seeds `let fresh = true` and only narrows it inside the three
    /// dialog branches, each gated on a probe whose failure is swallowed by `catch` — so a
    /// broken/renamed probe command made every start a no-dialog `remove_dir_all` of a
    /// workspace holding hours of training. The frontend now refuses to start on a probe
    /// failure; this is the backstop that makes the same mistake impossible to reintroduce.
    #[serde(default)]
    pub wipe_confirmed: bool,
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
    /// S76: which training PROJECT this run belongs to. Filled at start; empty while idle.
    /// Batch 2's export ledger and batch 4's explicit routing both key on it, so it is
    /// carried from the very first batch rather than bolted on later.
    #[serde(default)]
    pub project_id: String,
    /// The FAMILY SLOT directory of this run (`<data>/training/<project>/<family>`) — the
    /// exact equivalent of the pre-S76 workspace root, which is why every consumer that
    /// joins `audition/` or reads `weights/` off it keeps working unchanged.
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

/// The family SLOT directory for a (model name, backend) pair — the frontend's "does a
/// resumable workspace exist?" probes need the same mapping `try_start` uses.
///
/// S76: identity moved from「模型名 → 目录」to「模型名 → 项目 → 架构槽」, so the backend is
/// now part of the question. When no project exists yet the returned path is deliberately one
/// that cannot exist, so every `.exists()` probe answers false instead of erroring.
pub fn slot_path(data_dir: &Path, model_name: &str, backend: &str) -> PathBuf {
    let family = backend_family(backend);
    match tproject::find_by_name(data_dir, model_name) {
        Some(p) => tproject::family_dir(data_dir, &p.id, family),
        None => tproject::family_dir(data_dir, &slugify(model_name), family),
    }
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
    /// ①c: ordered speaker DISPLAY names, index = emb_g row id = the order the data page listed
    /// them in; empty for single-speaker. Read from the MANIFEST's `speaker_names`, which is
    /// merge-preserved — `run.json` is only the pre-fix fallback, and a later `sovits_diff` run
    /// rewrites that file WITHOUT a speakers key. (The manifest's `speakers` array is the
    /// matching slug list, same order.)
    pub speakers: Vec<String>,
    /// ①c: manifest diff_k_step_max (sovits_diff); 0 when absent.
    pub diff_k_step_max: u64,
}

/// LEGACY-SHAPE predicate: a pre-S76 workspace where `dataset/` and `dataset.fingerprint`
/// were siblings of the checkpoints. Still meaningful for exactly two callers — the
/// wipe-consent guard and the migration's empty-shell test, both of which look at directories
/// that may still have the old shape. The live "is there a reusable pool?" question is now
/// [`tproject::has_dataset`], asked of the PROJECT.
pub(crate) fn has_dataset_pool(ws: &Path) -> bool {
    ws.join("dataset.fingerprint").is_file()
        && std::fs::read_dir(ws.join("dataset"))
            .map(|mut d| d.next().is_some())
            .unwrap_or(false)
}

/// Does this workspace hold anything a wipe would destroy? = any family's checkpoints, any
/// diffusion progress, or an imported dataset pool (which cost a multi-minute import). An
/// empty leftover directory — try_start's `create_dir_all` runs before the run can fail —
/// holds nothing and stays freely wipeable.
///
/// Single source for the fail-closed wipe-consent guard; keep it a superset of every artifact
/// class the resume paths can read back (add a family ⇒ add its ckpt shape here).
pub(crate) fn workspace_holds_work(ws: &Path) -> bool {
    has_main_progress(ws)
        || max_vocoder_ckpt_step(ws).is_some()
        || max_diffusion_step(ws).unwrap_or(0) > 0
        // Preprocessing counts as work: slicing + f0 + feature extraction is the multi-HOUR
        // part of a training run, and a slot that has it but no checkpoint yet is the normal
        // state of「刚开始练」. `dataset.fingerprint` is the one artifact every family writes
        // (python does it on ENTERING preprocessing — `utai_train/cache.py`), which makes it
        // the single portable judge. Without this the wipe-consent guard would let a
        // half-trained slot be erased with no dialog, and the shared-dataset guard would not
        // recognise a sibling slot as "using this data".
        || ws.join("dataset.fingerprint").is_file()
        // pre-S76 shape only (dataset/ used to be a sibling of the checkpoints); still true
        // for a directory the migration has not folded yet.
        || has_dataset_pool(ws)
}

// `workspace_info(name, backend)` lived here until S76 batch 4. Every consumer now knows WHICH
// PROJECT it means and calls `slot_info` — display names became user-editable in that batch, so
// resolving a workspace from one could only ever go stale. Its one extra behaviour (probing the
// legacy slug path so an UNMIGRATED pre-S76 workspace still reported `exists`) moved to where it
// belongs: `list_project_summaries` lists such directories as「待迁移」rows, which is visible
// instead of merely non-empty.

/// Structured facts about ONE architecture slot of ONE project — the id-keyed form, which is
/// the only one that stays correct across a rename.
pub fn slot_info(data_dir: &Path, project_id: &str, backend: &str) -> WorkspaceInfo {
    let ws = tproject::family_dir(data_dir, project_id, backend_family(backend));
    let manifest = std::fs::read_to_string(ws.join("run_manifest.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .unwrap_or_default();
    let field = |k: &str| manifest[k].as_str().unwrap_or("").to_string();
    // ①c: display speaker names, ordered = emb_g id. The carriers and their precedence live in
    // ONE place (`frozen_speakers`) so this and the project's dataset view can never disagree
    // about who row i is. Empty for single-speaker — and also when NO carrier holds a name, so
    // the pre-existing "nothing to compare" semantics of the resume dialog are preserved
    // (a vec of blanks would read as a speaker mismatch).
    let speakers: Vec<String> = {
        let mut v: Vec<String> = frozen_speakers(data_dir, project_id, backend)
            .into_iter()
            .map(|s| s.name)
            .collect();
        if v.iter().all(|n| n.is_empty()) {
            v.clear();
        }
        v
    };
    WorkspaceInfo {
        exists: ws.exists(),
        family: field("backend"),
        version: field("version"),
        sample_rate: field("sample_rate"),
        has_main_progress: has_main_progress(&ws),
        diff_steps: max_diffusion_step(&ws).unwrap_or(0),
        aug_copies: manifest["aug_copies"].as_u64().unwrap_or(0),
        // S76: the reusable pool is the PROJECT's dataset, shared by every slot — not a
        // sibling of this slot's checkpoints any more.
        has_dataset: tproject::has_dataset(data_dir, project_id),
        vol_embedding: manifest["vol_embedding"].as_bool(),
        n_speakers: manifest["n_speakers"].as_u64().unwrap_or(1),
        speakers,
        diff_k_step_max: manifest["diff_k_step_max"].as_u64().unwrap_or(0),
    }
}

/// The `(slug, display name)` pairs ONE slot froze, in emb_g row order. Empty when this slot
/// never co-trained speakers.
///
/// SINGLE SOURCE for「这个槽的第 i 号歌手是谁」 — `slot_info` reports the names half of it.
/// `slugify` is one-way, so without this a `dataset/<slug>/` directory can never be shown as
/// the singer it holds, and the order is exactly what a manual rebuild must reproduce.
///
/// Two carriers, in this order:
/// * `run_manifest.json` — `speakers` (slugs) + `speaker_names`. Durable: merge-preserved
///   across a later `sovits_diff` run.
/// * `run.json` — `speakers[]` with `slug` AND `name` per entry. Older workspaces predate
///   `speaker_names` and this is the only place their names survive (verified against this
///   machine's real 2-singer projects). Matched BY SLUG, never by position: a `sovits_diff`
///   run rewrites `run.json` without the key at all, and a mismatched pair would print one
///   singer's name against another's emb_g row — the exact confusion this is meant to end.
pub fn frozen_speakers(data_dir: &Path, project_id: &str, family: &str) -> Vec<dsmanifest::DsSpeaker> {
    let ws = tproject::family_dir(data_dir, project_id, backend_family(family));
    let read_json = |p: PathBuf| -> Option<serde_json::Value> {
        std::fs::read_to_string(p)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
    };
    // `run.json` pairs, whether or not the manifest needs them.
    let run_pairs: Vec<(String, String)> = read_json(ws.join("run.json"))
        .and_then(|v| {
            v.get("speakers").and_then(|s| s.as_array()).map(|arr| {
                arr.iter()
                    .filter_map(|e| {
                        Some((
                            e.get("slug")?.as_str()?.to_string(),
                            e.get("name")?.as_str()?.to_string(),
                        ))
                    })
                    .collect()
            })
        })
        .unwrap_or_default();
    let Some(manifest) = read_json(ws.join("run_manifest.json")) else {
        return run_pairs
            .into_iter()
            .map(|(slug, name)| dsmanifest::DsSpeaker { slug, name })
            .collect();
    };
    let str_array = |k: &str| -> Vec<String> {
        manifest[k]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    };
    let slugs = str_array("speakers");
    if slugs.is_empty() {
        // single-speaker (no key) — or a manifest that lost it; `run.json` only ever carries
        // the array for a genuine co-training, so this stays empty for single-speaker runs.
        return run_pairs
            .into_iter()
            .map(|(slug, name)| dsmanifest::DsSpeaker { slug, name })
            .collect();
    }
    let names = str_array("speaker_names");
    slugs
        .into_iter()
        .enumerate()
        .map(|(i, slug)| {
            let name = names
                .get(i)
                .filter(|n| !n.is_empty())
                .cloned()
                .or_else(|| {
                    run_pairs
                        .iter()
                        .find(|(s, _)| *s == slug)
                        .map(|(_, n)| n.clone())
                })
                .unwrap_or_default();
            dsmanifest::DsSpeaker { slug, name }
        })
        .collect()
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
pub(crate) fn slugify(name: &str) -> String {
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

/// The `(display name, slug)` this run must use for each co-trained speaker.
///
/// Fresh run ⇒ derive from the names. RESUME of a slot that already froze a speaker set ⇒ REUSE
/// the frozen slugs, matched to the current names BY POSITION (the emb_g row order is what the
/// resume guard demands be identical anyway).
///
/// Why reuse rather than re-derive: `slugify` hash-suffixes with `DefaultHasher` (SipHash-1-3),
/// which std explicitly does not promise to keep stable across Rust releases — and the slug is
/// the `dataset/<slug>/` directory name, the `dataset_44k/<slug>/` slice directory and the
/// `config.spk` key. Re-deriving on every start means one toolchain bump turns every existing
/// multi-speaker project unresumable AND orphans its data directories. (Frozen values are only
/// adopted when the COUNT matches; anything else is a genuine structure change and falls through
/// to the resume guard, which refuses it with a specific CODE.)
fn effective_speaker_slugs(
    data_dir: &Path,
    project_id: &str,
    family: &str,
    req: &StartTrainingRequest,
) -> Vec<(String, String)> {
    if req.speakers.len() <= 1 {
        return Vec::new();
    }
    let fresh = assign_speaker_slugs(&req.speakers);
    if req.fresh {
        return fresh;
    }
    let frozen = frozen_speakers(data_dir, project_id, family);
    if frozen.len() != req.speakers.len() {
        return fresh;
    }
    req.speakers
        .iter()
        .zip(frozen.iter())
        .map(|(sp, fz)| (sp.name.clone(), fz.slug.clone()))
        .collect()
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
        // S68b loud-degradation guard, at PREFLIGHT: refuse before the workspace wipe /
        // dataset import (review: the run_worker placement cost a wiped workspace plus a
        // multi-minute import before erroring on a fully-decidable condition).
        refuse_cpu_only_runtime(&self.app_dir, req.force_cpu)?;

        // ★S75 device resolution — HERE, for the same reason as the guard above. The chosen GPU
        // decides which runtime drives the run, so a choice that no longer resolves is decidable
        // the moment the button is pressed; deciding it in run_worker would burn the workspace
        // and the whole dataset copy first. `req.gpu` is the UI id, re-derived against a freshly
        // built list (never trusted): id → entry → (variant, mask).
        let (gpu_mask, want_variant) = if req.force_cpu {
            ("-1".to_string(), None)
        } else if req.gpu.is_empty() {
            (String::new(), None)
        } else {
            let g = crate::commands::settings::training_gpu_by_id(&self.app_dir, &req.gpu)
                .ok_or_else(|| {
                    UtaiError::Training(format!("TRAINING_GPU_UNKNOWN: {}", req.gpu))
                })?;
            if !g.selectable {
                return Err(UtaiError::Training(
                    g.reason.unwrap_or_else(|| "TRAINING_GPU_UNSUPPORTED".to_string()),
                ));
            }
            (g.value, g.variant)
        };
        let (python, device_backend) = crate::pyenv::training_interpreter_for(
            &self.app_dir,
            req.force_cpu,
            want_variant.as_deref(),
        )
        .ok_or_else(|| {
            // Only reachable if the pack vanished between the UI listing it and this call — fail
            // CLOSED rather than fall back to whatever else happens to be installed.
            UtaiError::Training(format!(
                "TRAINING_RUNTIME_VARIANT_MISSING: {}",
                want_variant.clone().unwrap_or_default()
            ))
        })?;
        if !req.force_cpu && device_backend == "cpu" {
            tracing::warn!(
                "Training runtime is the CPU variant: this run will train on CPU (slow). For GPU training install the runtime pack matching your GPU in Settings → Training Environment."
            );
        }

        // READ-ONLY view of the target project, for the pre-flight checks below. Deliberately
        // not `resolve_or_create`: creation stays after every check that can still refuse, so a
        // rejected start never leaves an empty project behind. Keyed by id when the caller gave
        // one — otherwise「复用项目数据集」would look up the editable 本次训练名 and answer
        // TRAINING_NO_DATA for a project whose `dataset/` is right there.
        let existing_project: Option<tproject::ProjectMeta> = if req.project_id.trim().is_empty() {
            tproject::find_by_name(&data_dir, &req.model_name)
        } else {
            tproject::read_meta(&data_dir, &req.project_id)
        };

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
                for f in &sp.files {
                    if !Path::new(f).is_file() {
                        return Err(UtaiError::Training(format!(
                            "TRAINING_DATA_FILE_MISSING: {}",
                            f
                        )));
                    }
                }
            }
            // ── S78: 结构声明式复用 ──────────────────────────────────────────
            // Every group empty = 「就用这个项目盘上已有的这套歌手结构」, the multi-speaker twin
            // of the flat reuse path. Expressing it as「把磁盘上那些文件的路径原样传回来」would
            // also work today (an exactly-matching plan is a no-op) but only by coincidence: one
            // removed file renumbers the rest, one renamed singer changes a slug, and the request
            // silently becomes a full REPLACE of the shared dataset instead.
            //
            // The declaration is checked against the disk here, loudly, because nothing later
            // will: the import loop has nothing to copy and python would just train on whatever
            // subdirectories happen to exist.
            let declared_only = req.speakers.iter().all(|s| s.files.is_empty());
            let partial = !declared_only && req.speakers.iter().any(|s| s.files.is_empty());
            if partial {
                // half a declaration is not a declaration — the empty ones would silently get an
                // emb_g row with no audio
                let who = req
                    .speakers
                    .iter()
                    .find(|s| s.files.is_empty())
                    .map(|s| s.name.clone())
                    .unwrap_or_default();
                return Err(UtaiError::Training(format!("TRAINING_SPEAKER_NO_DATA: {who}")));
            }
            if declared_only {
                let existing = existing_project.as_ref();
                let ds = match existing {
                    Some(p) => tproject::dataset_dir(&data_dir, &p.id),
                    None => return Err(UtaiError::Training("TRAINING_NO_DATA".into())),
                };
                let on_disk: std::collections::BTreeSet<String> = std::fs::read_dir(&ds)
                    .map(|rd| {
                        rd.flatten()
                            .filter(|e| e.path().is_dir())
                            .map(|e| e.file_name().to_string_lossy().into_owned())
                            .collect()
                    })
                    .unwrap_or_default();
                let declared: std::collections::BTreeSet<String> = effective_speaker_slugs(
                    &data_dir,
                    &existing.unwrap().id,
                    backend_family(&req.backend),
                    &req,
                )
                .into_iter()
                .map(|(_, s)| s)
                .collect();
                if on_disk.is_empty() || on_disk != declared {
                    return Err(UtaiError::Training("PROJECT_DATASET_SHAPE".into()));
                }
                // a speaker directory with no audio would train an emb_g row on nothing
                for slug in &declared {
                    let empty = std::fs::read_dir(ds.join(slug))
                        .map(|mut d| d.next().is_none())
                        .unwrap_or(true);
                    if empty {
                        return Err(UtaiError::Training(format!(
                            "TRAINING_SPEAKER_NO_DATA: {slug}"
                        )));
                    }
                }
            }
        } else if req.dataset_files.is_empty() {
            // 复用项目数据集(S76 拓宽)。dataset/ 现在住在项目层、由全部架构槽共享,所以
            // 「不带数据启动」的正当性判据从「浅扩散 + 宿主是 sovits」拓宽成「这个项目已经
            // 有导入好的数据」——这正是工作区化最核心的那件事:一份数据喂多个架构。
            // 防「空数据逃课」的权威闸门仍在:项目里一个音频都没有就一律拒绝(前端禁用只是
            // 第一道线)。CODE 按后端分流保持不变,浅扩散那条对话框链的文案依赖它。
            let existing = existing_project.as_ref();
            let pool_ok = existing
                .map(|p| tproject::has_dataset(&data_dir, &p.id))
                .unwrap_or(false);
            if !pool_ok {
                return Err(UtaiError::Training(if req.backend == "sovits_diff" {
                    "TRAINING_NO_SHARED_POOL".into()
                } else {
                    "TRAINING_NO_DATA".to_string()
                }));
            }
            // Reuse carries no speaker groups, so it can only consume a FLAT dataset. Handing
            // a per-speaker (multi-singer) dataset to a run that believes it is single-speaker
            // would either crash the slicer or — worse, if it silently skipped the
            // subdirectories — fingerprint the empty set and freeze the caches forever.
            // Refuse here; re-importing with the speaker groups is the way through until the
            // data page learns to reuse a multi-speaker project (batch 5).
            // `pool_ok` above already proved `existing` is Some.
            let ds = tproject::dataset_dir(&data_dir, &existing.unwrap().id);
            let has_subdirs = std::fs::read_dir(&ds)
                .map(|rd| rd.flatten().any(|e| e.path().is_dir()))
                .unwrap_or(false);
            if has_subdirs {
                return Err(UtaiError::Training("PROJECT_DATASET_SHAPE".into()));
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

        // Artifact identity — `hps.name`, `weights/<slug>*.pth`, the `config.spk` key. It has
        // NOTHING to do with the directory layout any more (S76) and must keep deriving from
        // the run's model name, or every existing checkpoint would change its file name and
        // `best_state.json`'s carried-over metric would suppress the next best write.
        let slug = slugify(&req.model_name);
        // Directory identity — separate from the artifact identity above, and NOT derived from
        // the model name whenever the caller knows better (see `StartTrainingRequest.project_id`).
        // An id that names nothing is a hard refusal: silently falling back to name resolution
        // would create a SECOND project under the display name and train into it, which is the
        // exact fork this field exists to prevent.
        let project = if req.project_id.trim().is_empty() {
            // Pre-batch-4 path: resolve by name, create on first use. Reproduces the pre-S76
            // mapping exactly, including for a migrated workspace whose display name could not
            // be recovered (find_by_name falls back to the legacy slug).
            tproject::resolve_or_create(&data_dir, &req.model_name)?
        } else {
            // Same lookup the pre-flight above already did — reused rather than repeated so the
            // two can never disagree about which project this run is for.
            let m = existing_project
                .ok_or_else(|| UtaiError::Training("PROJECT_META_UNREADABLE".into()))?;
            // Same refusal `resolve_or_create` makes: an undecidable directory still holds its
            // content wherever migration found it, possibly at the project root where our
            // `dataset/` would land on top of it.
            if let Some(reason) = m.needs_attention.clone() {
                return Err(UtaiError::Training(format!("PROJECT_NEEDS_ATTENTION: {reason}")));
            }
            m
        };
        let family = backend_family(&req.backend).to_string();
        let workspace = tproject::family_dir(&data_dir, &project.id, &family);
        let manifest_path = workspace.join("run_manifest.json");

        // ---- shared-dataset guard (PREFLIGHT, never in run_worker) ----
        // `dataset/` belongs to the project now. Replacing it re-fingerprints every sibling
        // slot, so their next「续训」would rmtree hours of preprocessing AND continue on data
        // the user never meant to switch to — silently, since the fingerprint mismatch reads
        // as a legitimate change. Refuse while any OTHER slot holds work; the run's own slot
        // may still swap its data (that is the pre-S76 behaviour, unchanged).
        // The placement matters: `training/mod.rs`'s own history says a refusal on a fully
        // decidable condition must never cost a wiped slot plus a multi-minute import.
        let dataset_dir = tproject::dataset_dir(&data_dir, &project.id);
        // ①c/S78: the run's EFFECTIVE speaker slugs, decided once here and carried in `RunCtx`.
        // A RESUME reuses what the slot froze instead of re-deriving from the names — `slugify`
        // hash-suffixes with `DefaultHasher`, which std does not promise to keep stable across
        // Rust releases, and every one of those slugs is a directory name on disk plus a
        // `config.spk` key. Re-deriving would mean a toolchain bump silently renames every
        // co-trained speaker's data directory out from under a half-trained model.
        let eff_speakers = effective_speaker_slugs(&data_dir, &project.id, &family, &req);
        let planned = dataset_plan(&req, &eff_speakers);
        if !planned.is_empty() {
            let replacing = !current_dataset_listing(&dataset_dir).is_empty()
                && !dataset_matches(&dataset_dir, &planned);
            if replacing {
                if let Some(other) = tproject::FAMILIES.iter().find(|f| {
                    **f != family
                        && workspace_holds_work(&tproject::family_dir(&data_dir, &project.id, f))
                }) {
                    return Err(UtaiError::Training(format!(
                        "PROJECT_DATASET_IN_USE: {}",
                        other
                    )));
                }
                // A source that lives INSIDE the dataset we are about to replace would be
                // deleted by the swap and then fail to copy. Unreachable from today's UI
                // (the data page only offers外部文件), which is exactly why it is worth
                // nailing shut before batch 5 makes the project's own files selectable.
                let inside = req
                    .dataset_files
                    .iter()
                    .chain(req.speakers.iter().flat_map(|s| s.files.iter()))
                    .any(|f| Path::new(f).starts_with(&dataset_dir));
                if inside {
                    return Err(UtaiError::Training("TRAINING_DATASET_SELF_SOURCE".into()));
                }
            }
        }

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

        // ---- resume-parameter guard ----
        // The rule itself lives in `resume_lock` — ONE table plus ONE enforcement, driven
        // against each other by a unit test, because three other places (the run step's
        // pre-start dialog, the project page's form restore, the parameters page's read-only
        // rendering) have to agree with it and used to do so from memory.
        if let Some(code) = resume_lock::check_resume_locks(
            &req,
            &resume_lock::ResumeState {
                manifest: old_manifest.as_ref(),
                has_main,
                max_diffusion_step: max_diffusion_step(&workspace),
                frozen_speakers: &frozen_speakers(&data_dir, &project.id, &family),
            },
            !req.fresh || diff_partial_wipe,
        ) {
            return Err(UtaiError::Training(code));
        }

        // fail-closed wipe consent: a 重训 that would destroy real work (checkpoints of any
        // family, diffusion progress, or an imported dataset pool that cost the user a
        // multi-minute import) may only proceed when the frontend states the user actually
        // answered the destructive dialog. An empty leftover directory (a prior start that
        // died after create_dir_all) holds nothing and stays freely wipeable.
        if req.fresh && workspace.exists() && !req.wipe_confirmed {
            if workspace_holds_work(&workspace) {
                tracing::error!(
                    "refusing unconfirmed wipe of {} — the caller sent fresh=true without \
                     wipe_confirmed; a UI probe most likely failed silently",
                    workspace.display()
                );
                return Err(UtaiError::Training("TRAINING_WIPE_NOT_CONFIRMED".into()));
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
                // main retrain / diff-only slot (a full wipe here is what unlocks a version
                // change) / manifest-less anomaly.
                //
                // S76: `workspace` is now the FAMILY slot, so the project's shared `dataset/`
                // is outside it and survives structurally — a retrain clears this
                // architecture's progress and keeps the data, exactly as 拍板 1 says. Robust
                // removal because a READONLY attribute (backup/网盘 restores carry them) used
                // to fail the whole start with WORKSPACE_WIPE_FAILED.
                crate::util::remove_dir_all_robust(&workspace)
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
            // the EFFECTIVE list — a resume re-freezes exactly the slugs it inherited
            let slugs: Vec<String> = eff_speakers.iter().map(|(_, s)| s.clone()).collect();
            let names: Vec<String> = eff_speakers.iter().map(|(n, _)| n.clone()).collect();
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
        // A diffusion run trains on the SoVITS slot's own slice pool (`dataset_44k` under this
        // very workspace — that shared cache is the entire reason shallow diffusion lives in the
        // sovits family), and it re-fingerprints that pool. So choosing its own augmentation
        // count would rebuild the shared slices to a different recipe and silently change the
        // data the MAIN model resumes on ⇒ it inherits instead.
        //
        // S78: unless there is no main model in the slot (diff-first). Then nothing is sharing
        // the pool and the run's own value stands — otherwise a diff-first project could only
        // ever train at aug=0, for the sake of a main model that does not exist.
        let eff_aug_copies = if req.backend == "sovits_diff" && has_main {
            manifest["aug_copies"].as_u64().unwrap_or(0) as u32
        } else {
            req.aug_copies
        };
        // …and record it, so the NEXT diff run inherits what this one actually preprocessed with
        // rather than re-fingerprinting the pool back to 0. (A later main run overwrites it with
        // its own value, which is correct: from then on the main model owns the pool.)
        if req.backend == "sovits_diff" && !has_main {
            manifest["aug_copies"] = serde_json::json!(eff_aug_copies);
        }
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
                project_id: project.id.clone(),
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
            python,
            device_backend,
            gpu_mask,
            project_id: project.id.clone(),
            speakers: eff_speakers,
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
    /// S75: device resolution, decided at PREFLIGHT and carried here. It is NOT re-derived in
    /// run_worker — that placement is exactly what the S68b review moved out (a refusal on a
    /// fully-decidable condition must not cost a wiped workspace plus a multi-minute import).
    python: PathBuf,
    device_backend: String,
    /// run.json "gpu": the accelerator-native mask (UUID / vendor index), "-1" for forced CPU,
    /// "" when no device was chosen. Resolved from the picked entry's `value` — NOT from the
    /// request, whose `gpu` field carries the UI id.
    gpu_mask: String,
    /// S76: the resolved project. The dataset lives at the PROJECT level now, so the worker
    /// needs an identity the family slot path cannot give it. Decided in try_start (one
    /// resolution per run) — never re-derived here.
    project_id: String,
    /// ①c/S78: `(display name, slug)` per co-trained speaker, in emb_g row order. Empty for a
    /// single-speaker run.
    ///
    /// Resolved ONCE in try_start and carried, for the same reason the device is: the worker used
    /// to call `assign_speaker_slugs` again and the two agreed only because the function was a
    /// pure function of the request. It no longer is — a RESUME reuses the slugs frozen in the
    /// manifest instead of re-deriving them from the names — so a second derivation here would be
    /// a second answer.
    speakers: Vec<(String, String)>,
}

/// Content identity of one dataset file: byte size plus a digest of its first and last 64 KiB.
///
/// Deliberately the SAME shape as `utai_train/cache.py`'s `dataset_fingerprint` — the python
/// side decides whether the extraction caches are still valid by exactly this measure, so
/// judging by anything weaker here would let Rust say「没变」about a change python would (or
/// would not) notice. Reading 128 KiB per file is nothing next to the copy it may skip.
fn file_probe(path: &Path) -> (u64, String) {
    use sha2::{Digest, Sha256};
    use std::io::{Read, Seek, SeekFrom};
    let Ok(mut f) = std::fs::File::open(path) else {
        return (0, String::new());
    };
    let size = f.metadata().map(|m| m.len()).unwrap_or(0);
    let mut h = Sha256::new();
    h.update(size.to_le_bytes());
    let mut head = vec![0u8; 65536.min(size as usize)];
    if f.read_exact(&mut head).is_ok() {
        h.update(&head);
    }
    if size > 131072 {
        let mut tail = vec![0u8; 65536];
        if f.seek(SeekFrom::End(-65536)).is_ok() && f.read_exact(&mut tail).is_ok() {
            h.update(&tail);
        }
    }
    (size, format!("{:x}", h.finalize()))
}

/// One file of a dataset, keyed by where it lands and what it contains.
#[derive(Debug, Clone, PartialEq, Eq)]
struct DatasetItem {
    /// Where it lands under `dataset/`: `{:03}.<lowercased ext>`, prefixed by the speaker slug
    /// for a co-trained run.
    rel: String,
    size: u64,
    digest: String,
}

/// THE naming rule for a dataset copy: `<slug>/`-prefixed when co-training, then the file's
/// position in the sorted selection and the source's lowercased extension.
///
/// Single source on purpose — `dataset_plan` PREDICTS these names, the import loops WRITE them,
/// `dataset_matches` compares the two, and the annotation keys on them. Four readings of one
/// rule; if any of them ever spelled it differently the reuse path would silently turn into a
/// full replace (and the original file names would attach to nothing).
pub(crate) fn dataset_rel(slug: Option<&str>, i: usize, src: &str) -> String {
    let ext = Path::new(src)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("wav")
        .to_ascii_lowercase();
    match slug {
        Some(s) => format!("{}/{:03}.{}", s, i, ext),
        None => format!("{:03}.{}", i, ext),
    }
}

/// Exactly what this request will import, in the order the import loops write it.
///
/// `slugs` must be the run's EFFECTIVE `(name, slug)` list (see `RunCtx::speakers`) — passing a
/// freshly derived one would predict directory names a resume is not going to use.
fn dataset_plan(req: &StartTrainingRequest, slugs: &[(String, String)]) -> Vec<DatasetItem> {
    let mut out = Vec::new();
    let mut push = |src: &str, rel: String| {
        let (size, digest) = file_probe(Path::new(src));
        out.push(DatasetItem { rel, size, digest });
    };
    if req.speakers.len() > 1 {
        for (gi, (_name, slug)) in slugs.iter().enumerate() {
            let mut files = req.speakers[gi].files.clone();
            files.sort();
            for (i, f) in files.iter().enumerate() {
                push(f, dataset_rel(Some(slug), i, f));
            }
        }
    } else {
        let mut files = req.dataset_files.clone();
        files.sort();
        for (i, f) in files.iter().enumerate() {
            push(f, dataset_rel(None, i, f));
        }
    }
    out.sort_by(|a, b| a.rel.cmp(&b.rel));
    out
}

/// Is the project's shared `dataset/` ALREADY, byte for byte, what this plan would produce?
///
/// Judged by CONTENT, not by `(名字, 字节数)`: loudness normalisation and denoising rewrite a
/// wav in place without changing its length, and the first version of this compared only name
/// and size — so an edited dataset compared equal, the import was skipped, and python's
/// fingerprint (reading those same untouched copies) matched too, reusing every stale feature
/// cache. The run trained on the pre-edit audio and reported success.
///
/// Being exact here is what lets ONE judgement answer both questions safely:
/// * false ⇒ this start really is replacing the project's shared dataset, so the sibling-slot
///   guard must fire;
/// * true ⇒ nothing to copy and nothing to protect.
///
/// It must therefore work with no bookkeeping of any kind — a migrated project carries no
/// record of which sources produced its dataset, and a ledger-based judgement would have told
/// every existing user that their data was "changing" and blocked the entire point of the
/// refactor (一份数据喂多个架构) forever.
fn dataset_matches(dataset_dir: &Path, plan: &[DatasetItem]) -> bool {
    !plan.is_empty() && current_dataset_listing(dataset_dir) == plan
}

/// The same shape, read off disk (flat files plus one level of speaker subdirectories — the
/// only two shapes the import ever writes).
fn current_dataset_listing(dataset_dir: &Path) -> Vec<DatasetItem> {
    let mut out = Vec::new();
    let mut probe = |rel: String, p: &Path| {
        let (size, digest) = file_probe(p);
        out.push(DatasetItem { rel, size, digest });
    };
    let Ok(rd) = std::fs::read_dir(dataset_dir) else {
        return out;
    };
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        // a `.part` is append_files' stage-then-rename crash remnant, not a dataset file — skip it
        // so a leftover never makes dataset_matches judge a ready dataset "changed" (审查 S78).
        if name.ends_with(".part") {
            continue;
        }
        match e.metadata() {
            Ok(md) if md.is_file() => probe(name, &e.path()),
            Ok(md) if md.is_dir() => {
                if let Ok(sub) = std::fs::read_dir(e.path()) {
                    for se in sub.flatten() {
                        let sname = se.file_name().to_string_lossy().into_owned();
                        if !sname.ends_with(".part")
                            && se.metadata().map(|m| m.is_file()).unwrap_or(false)
                        {
                            probe(format!("{}/{}", name, sname), &se.path());
                        }
                    }
                }
            }
            _ => {}
        }
    }
    out.sort_by(|a, b| a.rel.cmp(&b.rel));
    out
}

/// Replaces the project's shared dataset without ever losing it.
///
/// The old dataset is moved aside by a same-volume rename (atomic), the caller fills a fresh
/// one, and `commit()` drops the aside copy. Anything else — an early `return` on 强制停止
/// mid-import, a `?` on a failed copy, a panic — runs `Drop`, which puts the old dataset
/// back. Without that, force-stopping during the import of a REPLACEMENT dataset would leave
/// the project with an empty `dataset/` and an orphaned `.dataset.old_<pid>` that nothing
/// reclaims (it is inside the project dir, so it would also be counted in the project's size
/// forever). The pre-S76 code simply `remove_dir_all`'d first, which had no recovery at all.
struct DatasetSwap {
    dataset: PathBuf,
    aside: Option<PathBuf>,
    /// There was nothing to replace — this import is creating `dataset/` for the first time.
    /// An abandoned FIRST import must leave no dataset at all, or the half-copied prefix would
    /// pass `tproject::has_dataset` and a later run could quietly train on it.
    created_fresh: bool,
    committed: bool,
}

impl DatasetSwap {
    /// No-op when there is nothing to replace (first import into a fresh project).
    fn begin(dataset_dir: &Path) -> Result<Self> {
        let mut swap = DatasetSwap {
            dataset: dataset_dir.to_path_buf(),
            aside: None,
            created_fresh: !dataset_dir.exists(),
            committed: false,
        };
        if !dataset_dir.exists() {
            return Ok(swap);
        }
        let aside = dataset_dir.with_file_name(format!(".dataset.old_{}", std::process::id()));
        let _ = crate::util::remove_dir_all_robust(&aside);
        crate::util::rename_with_retry(dataset_dir, &aside, "TRAINING_DATASET_SWAP")
            .map_err(UtaiError::Training)?;
        swap.aside = Some(aside);
        Ok(swap)
    }

    fn commit(&mut self) {
        self.committed = true;
    }
}

impl Drop for DatasetSwap {
    fn drop(&mut self) {
        let Some(aside) = self.aside.take() else {
            if !self.committed && self.created_fresh {
                tracing::warn!(
                    "first dataset import did not complete — removing the partial {}",
                    self.dataset.display()
                );
                let _ = crate::util::remove_dir_all_robust(&self.dataset);
            }
            return;
        };
        if self.committed {
            let _ = crate::util::remove_dir_all_robust(&aside);
            return;
        }
        tracing::warn!(
            "dataset import did not complete — restoring the previous dataset from {}",
            aside.display()
        );
        let _ = crate::util::remove_dir_all_robust(&self.dataset);
        if let Err(e) = crate::util::rename_with_retry(&aside, &self.dataset, "TRAINING_DATASET_RESTORE") {
            // Loud: the data is still on disk under `.dataset.old_*`, but the project now
            // looks empty and only a human can tell which is which.
            tracing::error!("could not restore the previous dataset ({e}) — it is kept at {}", aside.display());
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_worker(
    inner: &Arc<Inner>,
    app: &tauri::AppHandle,
    app_dir: &Path,
    data_dir: &Path,
    workspace: &Path,
    stop_file: &Path,
    req: &StartTrainingRequest,
    ctx: &RunCtx,
    slug: &str,
) -> Result<()> {
    // ---- stage: import the dataset into the PROJECT (shared by every family slot) ----
    let dataset_dir = tproject::dataset_dir(data_dir, &ctx.project_id);
    // The import used to be `remove_dir_all(dataset) + copy everything`, which was safe only
    // while a dataset belonged to exactly one workspace. It is now the project's shared layer,
    // so that shape would mean「训 RVC 顺手删掉 SoVITS 赖以续训的数据」 — and worse, once the
    // data page can list the project's own files as sources, it would delete its own copy
    // sources and then fail the copy. So: predict the exact resulting listing, and if the
    // dataset on disk already IS that listing, touch nothing at all.
    let plan = dataset_plan(req, &ctx.speakers);
    let dataset_unchanged = dataset_matches(&dataset_dir, &plan);
    // ★ An EMPTY plan means this run imports nothing — either the flat reuse path, or (S78) a
    // multi-speaker run that declares its structure and consumes the data already on disk.
    // `dataset_matches` returns false for an empty plan (by design: "nothing" must never compare
    // equal to a real dataset), so keying the swap on `dataset_unchanged` alone would move the
    // whole dataset aside, copy nothing into the fresh one, and commit — deleting the very data
    // the run was going to train on.
    let importing = !plan.is_empty();
    if dataset_unchanged {
        tracing::info!(
            "dataset import skipped: {} already holds exactly this selection",
            dataset_dir.display()
        );
    }
    // ①c: >1 speaker group = per-speaker subdir import; else the pre-①c flat
    // (or shared-pool) path, verbatim. run_speakers is filled only for multi
    // and becomes run.json "speakers" so the sovits/rvc pipeline co-trains them.
    let is_multi = req.speakers.len() > 1;
    let mut run_speakers: Vec<serde_json::Value> = Vec::new();
    // What this import puts on disk, for `<project>/dataset.json` — the only carrier of the
    // ORIGINAL file names and of the speaker display names once the copies are renamed to
    // `000.wav`. Collected in the copy loops so it describes what actually landed, and written
    // (best effort) after the swap commits; see `dsmanifest`.
    let mut ds_files: Vec<dsmanifest::DsFile> = Vec::new();
    let mut ds_speakers: Vec<dsmanifest::DsSpeaker> = Vec::new();
    if is_multi {
        // import EACH speaker's files into dataset/<slug>/ (000..N per speaker,
        // sorted — same deterministic-order + fingerprint rationale as the flat
        // path). The pipeline slices each subdir into dataset_44k/<slug> and the
        // loader derives the emb_g id from the dir name, so these slugs MUST
        // match the manifest — assign_speaker_slugs is deterministic on the
        // same request.
        let mut swap = if dataset_unchanged || !importing {
            None
        } else {
            Some(DatasetSwap::begin(&dataset_dir)?)
        };
        std::fs::create_dir_all(&dataset_dir)?;
        let assigned = ctx.speakers.clone();
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
                let rel = dataset_rel(Some(slug), i, f);
                // `rel` already carries the slug — join on the DATASET root, not on `sub`
                let dst = dataset_dir.join(&rel);
                // dataset_unchanged ⇒ dst already holds this exact file; the loop still runs
                // because run_speakers (→ run.json) is built from it.
                if !dataset_unchanged {
                    std::fs::copy(src, &dst).map_err(|e| {
                        UtaiError::Training(format!(
                            "TRAINING_IMPORT_COPY_FAILED: {}: {}",
                            src.display(),
                            e
                        ))
                    })?;
                }
                ds_files.push(dsmanifest::DsFile {
                    rel: rel.clone(),
                    name: src
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                    bytes: std::fs::metadata(&dst).map(|m| m.len()).unwrap_or(0),
                    duration_ms: None,
                });
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
            // list order = emb_g row id, same as `assign_speaker_slugs` promises
            ds_speakers.push(dsmanifest::DsSpeaker {
                slug: slug.clone(),
                name: name.clone(),
            });
        }
        if let Some(s) = swap.as_mut() {
            s.commit();
        }
    } else {
        let mut swap: Option<DatasetSwap> = None;
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
        } else if !dataset_unchanged {
            swap = Some(DatasetSwap::begin(&dataset_dir)?);
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
            let rel = dataset_rel(None, i, f);
            let dst = dataset_dir.join(&rel);
            if !dataset_unchanged {
                std::fs::copy(src, &dst).map_err(|e| {
                    UtaiError::Training(format!(
                        "TRAINING_IMPORT_COPY_FAILED: {}: {}",
                        src.display(),
                        e
                    ))
                })?;
            }
            ds_files.push(dsmanifest::DsFile {
                rel,
                name: src
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                bytes: std::fs::metadata(&dst).map(|m| m.len()).unwrap_or(0),
                duration_ms: None,
            });
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
        if let Some(s) = swap.as_mut() {
            s.commit();
        }
    }

    // ---- annotate what was just imported (original names + speaker order) ----
    // Guarded by「这次 run 自带选择」: on the reuse path the plan is empty and we know nothing
    // about the source names, so writing here would REPLACE a good annotation with unknowns.
    // An aborted import never reaches this line — `DatasetSwap`'s Drop has put the previous
    // dataset back by then, and its previous annotation still describes it exactly.
    //
    // Also skip when dataset_unchanged (审查 S78): the disk is byte-identical to what the
    // annotation already describes, and ds_files here carry duration_ms:None (run_worker never
    // probes), so rewriting would only wipe the durations a data-page import recorded. `unchanged`
    // implies a prior import already wrote the annotation, so nothing is lost by leaving it.
    if !plan.is_empty() && !dataset_unchanged {
        dsmanifest::record_import(data_dir, &ctx.project_id, ds_speakers, ds_files);
    }

    // Device resolution happened at PREFLIGHT (S75) — interpreter, backend and the run.json mask
    // all ride in on ctx. Nothing device-related is decided here, on purpose: this point is past
    // the workspace wipe and the dataset import.
    let (python, device_backend, gpu_mask) =
        (&ctx.python, ctx.device_backend.as_str(), ctx.gpu_mask.as_str());

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
        // The accelerator-native MASK (NVIDIA UUID / vendor-relative index), "-1" = forced CPU,
        // "" = auto (setup_visibility leaves it unset). S75: resolved at preflight from the
        // picked entry's `value` — NEVER `req.gpu`, which now carries the UI id (`vendor:n`).
        // Feeding an id to CUDA_VISIBLE_DEVICES would mask every device.
        "gpu": gpu_mask,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// S114 §F5-3: the numerical-divergence guard is python, so `cargo test` cannot
    /// exercise its behaviour — that is `converter/verify/training/gate_numerics_guard.py`
    /// (38 checks, 10 mutation probes). What CAN rot without anyone noticing is the
    /// cross-language contract, and that is what this pins, in the same
    /// `include_str!` style as `s113_alias_hint_wire_matches_the_ts_union` and
    /// `shipped_dictionaries_match_the_committed_manifest`.
    ///
    /// The failure this prevents is concrete: the trainer raises
    /// `RuntimeError("TRAINING_NUMERICS_DIVERGED: ...")`, `runner.py` turns it into a
    /// protocol error, and the frontend maps the CODE to i18n. Rename the CODE on the
    /// python side, or ship a language whose json never got the key, and the user sees
    /// a raw English CODE at the exact moment their training just died — the S67
    /// `TRAINING_GPU_UNAVAILABLE` chain has the identical shape.
    ///
    /// ⚠ It pins TEXT, not behaviour. If you rename the constant, this test tells you
    /// the four other places that have to move with it; that IS the point.
    #[test]
    fn s114_divergence_code_is_wired_across_python_rust_and_all_three_locales() {
        static NUMERICS_PY: &str = include_str!("../../../training/utai_train/numerics.py");
        static BACKEND_ERR_TS: &str = include_str!("../../../src/lib/backendError.ts");

        // The CODE's single source is the python constant — parse it, never retype it.
        let code = NUMERICS_PY
            .lines()
            .find_map(|l| l.trim().strip_prefix("CODE_DIVERGED = "))
            .map(|v| v.trim().trim_matches('"').to_string())
            .expect(
                "numerics.py must keep `CODE_DIVERGED = \"...\"` as a plain top-level literal — \
                 this gate parses it as the single source for the i18n key",
            );
        assert!(
            code.starts_with("TRAINING_") && code.chars().all(|c| c.is_ascii_uppercase() || c == '_'),
            "the CODE crosses a process boundary and lands in json keys: keep it SCREAMING_SNAKE ascii, got {code:?}"
        );

        assert!(
            BACKEND_ERR_TS.contains(&format!("{code}: {{ key: \"backend.{code}\" }}")),
            "src/lib/backendError.ts has no mapping for {code} — the user would see the raw CODE"
        );

        for (lang, raw) in [
            ("zh", include_str!("../../../src/i18n/zh.json")),
            ("en", include_str!("../../../src/i18n/en.json")),
            ("ja", include_str!("../../../src/i18n/ja.json")),
        ] {
            let v: serde_json::Value = serde_json::from_str(raw).unwrap();
            let msg = v.pointer(&format!("/backend/{code}")).and_then(|m| m.as_str());
            let msg = msg.unwrap_or_else(|| {
                panic!("src/i18n/{lang}.json is missing backend.{code} (docs are trilingual and so is this)")
            });
            // A stub like "TODO" would satisfy a mere presence check; this text is what a
            // user reads while their run is dying, so require it to actually say something.
            assert!(
                msg.chars().count() >= 30,
                "backend.{code} in {lang}.json is {} chars — too short to be the real message: {msg:?}",
                msg.chars().count()
            );
        }

        // And the guard must still be CALLED. A guard nobody calls passes review and
        // protects nothing (S109 §G14: `sync_bundled_dictionaries` had exactly one
        // production call site and nothing pinned it).
        for (name, src) in [
            ("rvc", include_str!("../../../training/utai_train/rvc/train.py")),
            ("sovits", include_str!("../../../training/utai_train/sovits/train.py")),
            ("sovits_v2", include_str!("../../../training/utai_train/sovits_v2/train.py")),
        ] {
            assert!(
                src.contains("divergence.observe("),
                "{name}/train.py no longer calls divergence.observe() — a run that goes nan would \
                 again burn hours in silence"
            );
            assert!(
                src.contains("numerics.best_save_is_safe("),
                "{name}/train.py no longer consults numerics.best_save_is_safe() — save_best would \
                 again be free to overwrite a good checkpoint with nan weights"
            );
        }
    }

    fn tmp_ws(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("utai_ws_test_{}_{}", tag, uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// The naming rule has FOUR readers (plan / import / match / annotation). This closes the
    /// loop end to end without a training process: plan a selection, write it exactly where
    /// `run_worker` writes it, annotate it exactly as `run_worker` annotates it, and require
    /// that the dataset view then names every single file.
    ///
    /// A drift between any two of those readers is invisible at compile time and shows up as
    /// either「导入完还说数据变了」(a silent full re-import) or a file list of bare `000.wav`.
    #[test]
    fn planned_names_imported_names_and_annotated_names_are_one_rule() {
        let src = tmp_ws("rel_src");
        let data = tmp_ws("rel_data");
        let id = "proj_rel";
        let b = src_file(&src, "b.WAV", 10);
        let a = src_file(&src, "a.flac", 20);
        assert_eq!(dataset_rel(None, 0, &a), "000.flac");
        assert_eq!(dataset_rel(None, 7, "x/y.MP3"), "007.mp3");
        assert_eq!(dataset_rel(Some("spk_1"), 2, "no_extension"), "spk_1/002.wav");

        let req = req_from(serde_json::json!({
            "model_name": "t", "backend": "sovits", "version": "4.1", "sample_rate": "44k",
            "dataset_files": [],
            "speakers": [
                {"name": "歌姫", "files": [b.clone()]},
                {"name": "second", "files": [a.clone(), b.clone()]},
            ],
            "total_epoch": 1, "batch_size": 1,
        }));
        let ds = tproject::dataset_dir(&data, id);
        let plan = dataset_plan(&req, &assign_speaker_slugs(&req.speakers));
        let assigned = assign_speaker_slugs(&req.speakers);

        // ---- import, byte for byte as run_worker does ----
        let mut annotated: Vec<dsmanifest::DsFile> = Vec::new();
        for (gi, (_n, slug)) in assigned.iter().enumerate() {
            std::fs::create_dir_all(ds.join(slug)).unwrap();
            let mut files = req.speakers[gi].files.clone();
            files.sort();
            for (i, f) in files.iter().enumerate() {
                let rel = dataset_rel(Some(slug), i, f);
                let dst = ds.join(&rel);
                std::fs::copy(f, &dst).unwrap();
                annotated.push(dsmanifest::DsFile {
                    rel,
                    name: Path::new(f).file_name().unwrap().to_string_lossy().into_owned(),
                    bytes: std::fs::metadata(&dst).map(|m| m.len()).unwrap_or(0),
                    duration_ms: None,
                });
            }
        }
        // what was written IS what was planned — the reuse path depends on this exact equality
        assert!(
            dataset_matches(&ds, &dataset_plan(&req, &assign_speaker_slugs(&req.speakers))),
            "the import must land on the names the plan predicted"
        );

        dsmanifest::record_import(
            &data,
            id,
            assigned
                .iter()
                .map(|(n, s)| dsmanifest::DsSpeaker { slug: s.clone(), name: n.clone() })
                .collect(),
            annotated,
        );

        // ---- and the view names every file, in emb_g order ----
        let frozen: Vec<Vec<dsmanifest::DsSpeaker>> = Vec::new();
        let facts = dsmanifest::read_facts(&data, id, &frozen);
        assert_eq!(facts.files, 3);
        assert!(
            facts.entries.iter().all(|e| !e.name.is_empty()),
            "every imported file must carry its original name: {:?}",
            facts.entries
        );
        assert!(facts.order_known);
        assert_eq!(
            facts.groups.iter().map(|g| g.speaker.name.as_str()).collect::<Vec<_>>(),
            vec!["歌姫", "second"],
            "emb_g order is the REQUEST order, not the alphabetical slug order"
        );
        assert_eq!(facts.groups[0].files, 1);
        assert_eq!(facts.groups[1].files, 2);
        let _ = std::fs::remove_dir_all(&src);
        let _ = std::fs::remove_dir_all(&data);
    }

    /// ★ The slug debt, closed: a RESUME must reuse the slugs the slot froze, never re-derive
    /// them from the names.
    ///
    /// `slugify` hash-suffixes with `DefaultHasher`, which std does not promise to keep stable
    /// across Rust releases — and that slug is `dataset/<slug>/`, `dataset_44k/<slug>/` and the
    /// `config.spk` key. Re-deriving on every start means one toolchain bump renames every
    /// co-trained speaker's data directory out from under a half-trained model. The test proves
    /// reuse by freezing slugs that `slugify` would NEVER produce.
    #[test]
    fn a_resume_reuses_the_frozen_speaker_slugs_instead_of_re_deriving_them() {
        let data = tmp_ws("effslug");
        let id = "proj_s";
        let ws = tproject::family_dir(&data, id, "rvc");
        std::fs::create_dir_all(&ws).unwrap();
        let req = |fresh: bool| {
            req_from(serde_json::json!({
                "model_name": "t", "backend": "rvc", "version": "v2", "sample_rate": "40k",
                "dataset_files": [],
                "speakers": [{"name": "sayo", "files": []}, {"name": "teto", "files": []}],
                "fresh": fresh, "total_epoch": 1, "batch_size": 1,
            }))
        };

        // no manifest yet ⇒ derive
        let fresh_slugs = effective_speaker_slugs(&data, id, "rvc", &req(false));
        assert_eq!(fresh_slugs, assign_speaker_slugs(&req(false).speakers));

        // slugs a toolchain change (or an older build) could have produced — nothing `slugify`
        // would output for these names today
        std::fs::write(
            ws.join("run_manifest.json"),
            serde_json::to_string(&serde_json::json!({
                "backend": "rvc",
                "n_speakers": 2,
                "speakers": ["sayo_deadbeef", "teto_cafebabe"],
                "speaker_names": ["sayo", "teto"],
            }))
            .unwrap(),
        )
        .unwrap();
        let resumed = effective_speaker_slugs(&data, id, "rvc", &req(false));
        assert_eq!(
            resumed,
            vec![
                ("sayo".to_string(), "sayo_deadbeef".to_string()),
                ("teto".to_string(), "teto_cafebabe".to_string()),
            ],
            "a resume must keep training into the directories that already exist"
        );
        assert_ne!(resumed, fresh_slugs, "…which are NOT what slugify derives today");

        // 重训 wipes the slot, so it is free to mint new ones
        assert_eq!(
            effective_speaker_slugs(&data, id, "rvc", &req(true)),
            fresh_slugs
        );

        // a genuine structure change (count differs) falls through to freshly derived slugs —
        // the resume guard is what refuses it, with a specific CODE
        let three = req_from(serde_json::json!({
            "model_name": "t", "backend": "rvc", "version": "v2", "sample_rate": "40k",
            "dataset_files": [],
            "speakers": [{"name": "a", "files": []}, {"name": "b", "files": []}, {"name": "c", "files": []}],
            "total_epoch": 1, "batch_size": 1,
        }));
        assert_eq!(
            effective_speaker_slugs(&data, id, "rvc", &three),
            assign_speaker_slugs(&three.speakers)
        );
        let _ = std::fs::remove_dir_all(&data);
    }

    /// ★ Why `run_worker` needs an `importing` flag separate from `dataset_unchanged`.
    ///
    /// A structure declaration (every group's files empty) plans NOTHING, and `dataset_matches`
    /// answers false for an empty plan by design — "nothing" must never compare equal to a real
    /// dataset. Keying the dataset swap on `dataset_unchanged` alone would therefore move the
    /// whole dataset aside, copy nothing in, and commit: the data the run was about to train on,
    /// deleted. This test states that shape so the flag cannot be "simplified" away.
    #[test]
    fn a_structure_declaration_plans_no_import_and_must_not_look_like_a_replacement() {
        let proj = tmp_ws("decl");
        let ds = proj.join("dataset");
        std::fs::create_dir_all(ds.join("sayo_x")).unwrap();
        std::fs::write(ds.join("sayo_x").join("000.wav"), b"x").unwrap();
        let req = req_from(serde_json::json!({
            "model_name": "t", "backend": "rvc", "version": "v2", "sample_rate": "40k",
            "dataset_files": [],
            "speakers": [{"name": "sayo", "files": []}, {"name": "teto", "files": []}],
            "total_epoch": 1, "batch_size": 1,
        }));
        let plan = dataset_plan(&req, &assign_speaker_slugs(&req.speakers));
        assert!(plan.is_empty(), "a declaration imports nothing");
        assert!(
            !dataset_matches(&ds, &plan),
            "…and an empty plan never 'matches' — hence the separate importing flag"
        );
        let _ = std::fs::remove_dir_all(&proj);
    }

    /// Both carriers of「第 i 号歌手是谁」, and the two semantics `slot_info` must keep.
    ///
    /// REGRESSION GUARD: `slot_info().speakers` used to fall back from the manifest to
    /// `run.json` on its own. That logic now lives in `frozen_speakers` (shared with the
    /// dataset view), and the one behaviour that must survive the move is the LAST case —
    /// no name anywhere yields an EMPTY vec, not a vec of blanks. The resume dialog compares
    /// that vec positionally against the form, so blanks would read as a speaker mismatch and
    /// refuse a perfectly valid 续训.
    #[test]
    fn frozen_speakers_reads_both_carriers_and_stays_empty_when_no_name_survives() {
        let data = tmp_ws("frozen");
        let id = "proj_1";
        let ws = tproject::family_dir(&data, id, "rvc");
        std::fs::create_dir_all(&ws).unwrap();
        let write = |name: &str, v: serde_json::Value| {
            std::fs::write(ws.join(name), serde_json::to_string(&v).unwrap()).unwrap()
        };

        // 1. the durable pair
        write(
            "run_manifest.json",
            serde_json::json!({
                "backend": "rvc",
                "n_speakers": 2,
                "speakers": ["sayo_a", "teto_b"],
                "speaker_names": ["sayo", "teto"],
            }),
        );
        let f = frozen_speakers(&data, id, "rvc");
        assert_eq!(f.len(), 2);
        assert_eq!((f[0].slug.as_str(), f[0].name.as_str()), ("sayo_a", "sayo"));
        assert_eq!((f[1].slug.as_str(), f[1].name.as_str()), ("teto_b", "teto"));
        assert_eq!(slot_info(&data, id, "rvc").speakers, vec!["sayo", "teto"]);

        // 2. pre-`speaker_names` workspace: names live only in run.json, matched BY SLUG —
        //    and note run.json lists them in the OTHER order, which must not reorder anything
        write(
            "run_manifest.json",
            serde_json::json!({
                "backend": "rvc",
                "n_speakers": 2,
                "speakers": ["sayo_a", "teto_b"],
            }),
        );
        write(
            "run.json",
            serde_json::json!({
                "speakers": [
                    {"slug": "teto_b", "name": "teto"},
                    {"slug": "sayo_a", "name": "sayo"},
                ]
            }),
        );
        let f = frozen_speakers(&data, id, "rvc");
        assert_eq!(
            f.iter().map(|s| s.slug.as_str()).collect::<Vec<_>>(),
            vec!["sayo_a", "teto_b"],
            "the MANIFEST owns the emb_g order; run.json only supplies names"
        );
        assert_eq!(f[0].name, "sayo");
        assert_eq!(f[1].name, "teto");
        assert_eq!(slot_info(&data, id, "rvc").speakers, vec!["sayo", "teto"]);

        // 3. a later sovits_diff run rewrote run.json without the key: order survives, names do
        //    not — and `slot_info` must then report NOTHING rather than two blanks
        write("run.json", serde_json::json!({"backend": "sovits_diff"}));
        let f = frozen_speakers(&data, id, "rvc");
        assert_eq!(f.len(), 2, "the order is still recoverable");
        assert!(f.iter().all(|s| s.name.is_empty()));
        assert!(
            slot_info(&data, id, "rvc").speakers.is_empty(),
            "all-blank must collapse to empty — a blank vec of the right length would read as \
             a speaker mismatch in the resume dialog"
        );

        // 4. single-speaker: neither carrier has a speakers array
        let ws2 = tproject::family_dir(&data, id, "sovits");
        std::fs::create_dir_all(&ws2).unwrap();
        std::fs::write(
            ws2.join("run_manifest.json"),
            serde_json::to_string(&serde_json::json!({"backend": "sovits", "version": "4.1"}))
                .unwrap(),
        )
        .unwrap();
        assert!(frozen_speakers(&data, id, "sovits").is_empty());
        assert!(slot_info(&data, id, "sovits").speakers.is_empty());
        let _ = std::fs::remove_dir_all(&data);
    }

    /// The wipe-consent guard's judgement, artifact class by artifact class. Each `true` case is
    /// hours of work an unconfirmed `fresh` start would have deleted; the `false` case is the
    /// leftover directory try_start itself creates, which must stay freely wipeable (else a
    /// crashed first run would lock the user out of ever retrying that model name).
    #[test]
    fn workspace_holds_work_covers_every_artifact_class() {
        let empty = tmp_ws("empty");
        assert!(!workspace_holds_work(&empty), "empty leftover dir holds nothing");

        let main = tmp_ws("main");
        std::fs::write(main.join("G_2333333.pth"), b"x").unwrap();
        assert!(workspace_holds_work(&main), "rvc/sovits main checkpoint");

        let voc = tmp_ws("voc");
        std::fs::write(voc.join("model_ckpt_steps_4000.ckpt"), b"x").unwrap();
        assert!(workspace_holds_work(&voc), "vocoder lightning checkpoint");

        let diff = tmp_ws("diff");
        std::fs::create_dir_all(diff.join("diffusion")).unwrap();
        std::fs::write(diff.join("diffusion").join("model_5000.pt"), b"x").unwrap();
        assert!(workspace_holds_work(&diff), "diffusion progress");

        // model_0.pt = the seeded base only (step 0) — no user progress yet.
        let base_only = tmp_ws("base");
        std::fs::create_dir_all(base_only.join("diffusion")).unwrap();
        std::fs::write(base_only.join("diffusion").join("model_0.pt"), b"x").unwrap();
        assert!(!workspace_holds_work(&base_only), "seeded diffusion base is not progress");

        // preprocessing alone is HOURS of work — a slot with a fingerprint but no checkpoint
        // yet is just「刚开始练」, and it is also what makes a sibling slot count as "using"
        // the shared dataset.
        let pre = tmp_ws("pre");
        std::fs::write(pre.join("dataset.fingerprint"), b"abc").unwrap();
        assert!(workspace_holds_work(&pre), "preprocessing counts as work");
        let _ = std::fs::remove_dir_all(pre);

        // an imported dataset pool alone is worth protecting: re-importing costs minutes.
        let pool = tmp_ws("pool");
        std::fs::create_dir_all(pool.join("dataset")).unwrap();
        std::fs::write(pool.join("dataset").join("000.wav"), b"x").unwrap();
        std::fs::write(pool.join("dataset.fingerprint"), b"abc").unwrap();
        assert!(workspace_holds_work(&pool), "imported dataset pool");

        for d in [empty, main, voc, diff, base_only, pool] {
            let _ = std::fs::remove_dir_all(d);
        }
    }

    fn src_file(dir: &Path, name: &str, bytes: usize) -> String {
        let p = dir.join(name);
        std::fs::write(&p, vec![b'x'; bytes]).unwrap();
        p.to_string_lossy().into_owned()
    }

    fn req_from(v: serde_json::Value) -> StartTrainingRequest {
        serde_json::from_value(v).unwrap()
    }

    /// THE load-bearing property of the shared project dataset: "已经是这份数据就一个字节都不动"
    /// must be judged by CONTENT, and must need no bookkeeping.
    ///
    /// Two failures this pins down, both found by review:
    /// * comparing `(产物名, 字节数)` cannot see an in-place edit — loudness normalisation and
    ///   denoising rewrite a wav without changing its length. The import was skipped, python's
    ///   fingerprint read the same untouched copies and matched too, every stale feature cache
    ///   was reused, and the run trained on the pre-edit audio while reporting success.
    /// * judging by a written-at-import ledger instead would have been exact but useless: a
    ///   MIGRATED project has no such record, so every existing user would have been told
    ///   their data was "changing" and blocked from the entire point of this refactor
    ///   (一份数据喂多个架构).
    #[test]
    fn dataset_match_is_judged_by_content_and_needs_no_bookkeeping() {
        let src = tmp_ws("plan_src");
        let proj = tmp_ws("plan_proj");
        let ds = proj.join("dataset");
        // deliberately out of order and mixed-case extensions — the import sorts paths and
        // lowercases extensions, and the plan must do the identical thing
        let b = src_file(&src, "b.WAV", 10);
        let a = src_file(&src, "a.flac", 20);
        let mk = |files: Vec<String>| {
            req_from(serde_json::json!({
                "model_name": "t", "backend": "rvc", "version": "v2", "sample_rate": "40k",
                "dataset_files": files, "total_epoch": 1, "batch_size": 1,
            }))
        };
        let req = mk(vec![b.clone(), a.clone()]);
        let plan = dataset_plan(&req, &assign_speaker_slugs(&req.speakers));
        assert_eq!(
            plan.iter().map(|i| i.rel.as_str()).collect::<Vec<_>>(),
            vec!["000.flac", "001.wav"],
            "names are positional in SORTED source order, extensions lowercased"
        );
        assert!(!dataset_matches(&ds, &plan), "nothing imported yet");

        // import exactly as run_worker does — no ledger is written anywhere
        std::fs::create_dir_all(&ds).unwrap();
        for item in &plan {
            let srcp = if item.rel.ends_with(".flac") { &a } else { &b };
            std::fs::copy(srcp, ds.join(&item.rel)).unwrap();
        }
        assert!(
            dataset_matches(&ds, &dataset_plan(&req, &assign_speaker_slugs(&req.speakers))),
            "a migrated project has no bookkeeping either — content alone must answer this"
        );

        // ★ in-place edit, byte length unchanged → MUST read as changed
        std::fs::write(&a, vec![b'y'; 20]).unwrap();
        assert!(
            !dataset_matches(&ds, &dataset_plan(&req, &assign_speaker_slugs(&req.speakers))),
            "an edited source with the same size must never be judged unchanged"
        );
        std::fs::write(&a, vec![b'x'; 20]).unwrap();
        assert!(dataset_matches(&ds, &dataset_plan(&req, &assign_speaker_slugs(&req.speakers))), "restoring content restores the match");

        // a different selection is a mismatch, and so is a dataset someone deleted files from
        assert!(!dataset_matches(&ds, &dataset_plan(&mk(vec![a.clone()]), &[])));
        std::fs::remove_file(ds.join("000.flac")).unwrap();
        assert!(!dataset_matches(&ds, &dataset_plan(&req, &assign_speaker_slugs(&req.speakers))));

        // multi-speaker: per-speaker subdirectory keyed by the frozen slug
        let m = req_from(serde_json::json!({
            "model_name": "t", "backend": "sovits", "version": "4.1", "sample_rate": "44k",
            "dataset_files": [],
            "speakers": [
                {"name": "歌姫", "files": [a.clone()]},
                {"name": "second", "files": [b.clone(), a.clone()]},
            ],
            "total_epoch": 1, "batch_size": 1,
        }));
        let mplan = dataset_plan(&m, &assign_speaker_slugs(&m.speakers));
        let slugs: Vec<String> =
            assign_speaker_slugs(&m.speakers).into_iter().map(|(_, s)| s).collect();
        assert!(mplan.iter().all(|i| slugs.iter().any(|s| i.rel.starts_with(&format!("{s}/")))));
        let mds = tmp_ws("plan_multi").join("dataset");
        for item in &mplan {
            let dst = mds.join(&item.rel);
            std::fs::create_dir_all(dst.parent().unwrap()).unwrap();
            let srcp = if item.rel.ends_with(".flac") { &a } else { &b };
            std::fs::copy(srcp, &dst).unwrap();
        }
        assert!(dataset_matches(&mds, &mplan));

        for d in [src, proj, mds] {
            let _ = std::fs::remove_dir_all(d);
        }
    }

    /// 强制停止 during the import of a REPLACEMENT dataset must not cost the user the dataset
    /// they already had. (The pre-S76 code `remove_dir_all`'d first, so an abort there was
    /// unrecoverable; the swap exists precisely to make the failure path restore.)
    #[test]
    fn dataset_swap_restores_on_failure_and_reclaims_on_commit() {
        let proj = tmp_ws("swap");
        let ds = proj.join("dataset");
        std::fs::create_dir_all(&ds).unwrap();
        std::fs::write(ds.join("000.wav"), b"original").unwrap();

        // abandoned (abort / error / panic): the old dataset comes back untouched
        {
            let _swap = DatasetSwap::begin(&ds).unwrap();
            std::fs::create_dir_all(&ds).unwrap();
            std::fs::write(ds.join("000.wav"), b"half-written replacement").unwrap();
        }
        assert_eq!(std::fs::read(ds.join("000.wav")).unwrap(), b"original");
        // and nothing is left lying around inside the project
        assert!(
            std::fs::read_dir(&proj).unwrap().flatten().all(|e| {
                !e.file_name().to_string_lossy().starts_with(".dataset.old")
            }),
            "an orphaned aside copy would be counted in the project size forever"
        );

        // committed: the replacement stands and the old copy is reclaimed
        {
            let mut swap = DatasetSwap::begin(&ds).unwrap();
            std::fs::create_dir_all(&ds).unwrap();
            std::fs::write(ds.join("000.wav"), b"new").unwrap();
            swap.commit();
        }
        assert_eq!(std::fs::read(ds.join("000.wav")).unwrap(), b"new");
        assert!(std::fs::read_dir(&proj).unwrap().flatten().all(|e| {
            !e.file_name().to_string_lossy().starts_with(".dataset.old")
        }));

        // a first import (nothing to replace) is a no-op that still commits cleanly
        let fresh = tmp_ws("swap_fresh");
        let mut s = DatasetSwap::begin(&fresh.join("dataset")).unwrap();
        s.commit();

        for d in [proj, fresh] {
            let _ = std::fs::remove_dir_all(d);
        }
    }

    /// Identity is now「模型名 → 项目 → 架构槽」, and sovits_diff deliberately resolves to the
    /// sovits slot — shallow diffusion shares the main model's preprocessing caches, which is
    /// the entire reason it exists.
    #[test]
    fn slot_path_maps_backend_to_family_and_never_escapes_the_project() {
        let data = std::env::temp_dir().join(format!("utai_slot_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(data.join("training")).unwrap();
        let p = tproject::resolve_or_create(&data, "歌姫テスト").unwrap();
        for (backend, family) in [
            ("rvc", "rvc"),
            ("sovits", "sovits"),
            ("sovits_diff", "sovits"),
            ("sovits_v2", "sovits_v2"),
            ("vocoder", "vocoder"),
        ] {
            let got = slot_path(&data, "歌姫テスト", backend);
            assert_eq!(got, tproject::project_dir(&data, &p.id).join(family));
        }
        // an unknown name resolves to a path that cannot exist, so `.exists()` probes answer
        // false instead of erroring — and it must still land under the training root
        assert!(slot_path(&data, "nobody", "rvc").starts_with(data.join("training")));
        let _ = std::fs::remove_dir_all(data);
    }
}
