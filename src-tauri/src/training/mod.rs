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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartTrainingRequest {
    pub model_name: String,
    pub backend: String, // "rvc" (sovits / diffusion / vocoder land in later phases)
    pub version: String, // "v1" | "v2"
    pub sample_rate: String, // "32k" | "40k" | "48k"
    pub dataset_files: Vec<String>,
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
    #[serde(default)]
    pub gpu: u32,
    #[serde(default)]
    pub force_cpu: bool,
    #[serde(default)]
    pub spk_id: u32,
    /// true = 重训 (wipe the workspace), false = 续训 (resume from latest ckpt)
    #[serde(default)]
    pub fresh: bool,
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
                .map_err(|e| UtaiError::Training(format!("终止训练进程失败: {}", e)))?;
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
            return Err(UtaiError::Training("已有训练在进行中".into()));
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

        if req.backend != "rvc" {
            return Err(UtaiError::Training(format!(
                "训练后端「{}」尚未实现（当前仅 RVC）",
                req.backend
            )));
        }
        if req.model_name.trim().is_empty() {
            return Err(UtaiError::Training("模型名不能为空".into()));
        }
        if req.dataset_files.is_empty() {
            return Err(UtaiError::Training("请先导入训练数据".into()));
        }
        if !matches!(req.sample_rate.as_str(), "32k" | "40k" | "48k") {
            return Err(UtaiError::Training(format!("非法采样率 {}", req.sample_rate)));
        }
        for f in &req.dataset_files {
            if !Path::new(f).is_file() {
                return Err(UtaiError::Training(format!("数据文件不存在: {}", f)));
            }
        }

        // ---- resolve + verify every asset up front (loud, specific errors) ----
        let aux_dir = data_dir.join("models").join("aux");
        let contentvec = aux_dir.join(if req.version == "v1" {
            "contentvec_256l9.onnx"
        } else {
            "contentvec_768l12.onnx"
        });
        let rmvpe_pt = aux_dir.join("rmvpe.pt");
        let pretrain_dir = data_dir.join("models").join("training").join("rvc").join(
            if req.version == "v1" {
                "pretrained"
            } else {
                "pretrained_v2"
            },
        );
        let pretrain_g = pretrain_dir.join(format!("f0G{}.pth", req.sample_rate));
        let pretrain_d = pretrain_dir.join(format!("f0D{}.pth", req.sample_rate));
        let ffmpeg = crate::audio::find_ffmpeg()
            .ok_or_else(|| UtaiError::Training("找不到 ffmpeg.exe（训练预处理需要）".into()))?;
        for (label, p) in [
            ("ContentVec 特征提取器", &contentvec),
            ("RMVPE 音高模型 (rmvpe.pt)", &rmvpe_pt),
            ("预训练底模 G", &pretrain_g),
            ("预训练底模 D", &pretrain_d),
        ] {
            if !p.is_file() {
                return Err(UtaiError::Training(format!(
                    "缺少{}: {}（请将文件放置到该路径）",
                    label,
                    p.display()
                )));
            }
        }

        let slug = slugify(&req.model_name);
        let workspace = data_dir.join("training").join(&slug);
        if req.fresh && workspace.exists() {
            std::fs::remove_dir_all(&workspace)
                .map_err(|e| UtaiError::Training(format!("清空旧训练工作区失败: {}", e)))?;
        }
        std::fs::create_dir_all(&workspace)?;

        // resume-parameter guard: version/sample_rate are baked into the workspace
        // artifacts (slices, features, config.json) — changing them on a resume
        // would mismatch (48k wavs vs 40k hps) or silently degrade to retrain
        let manifest_path = workspace.join("run_manifest.json");
        if !req.fresh && manifest_path.exists() {
            let old: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(&manifest_path).unwrap_or_default(),
            )
            .unwrap_or_default();
            let old_ver = old["version"].as_str().unwrap_or("");
            let old_sr = old["sample_rate"].as_str().unwrap_or("");
            if (!old_ver.is_empty() && old_ver != req.version)
                || (!old_sr.is_empty() && old_sr != req.sample_rate)
            {
                return Err(UtaiError::Training(format!(
                    "续训参数与原工作区不一致（原 {}/{}，现 {}/{}）：续训必须沿用原版本与采样率，或选择重训",
                    old_ver, old_sr, req.version, req.sample_rate
                )));
            }
        }
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&serde_json::json!({
                "version": req.version,
                "sample_rate": req.sample_rate,
            }))?,
        )?;

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
                ..Default::default()
            };
        }
        self.inner.history.lock().clear();
        self.inner.stderr_ring.lock().clear();
        *self.inner.stop_file.lock() = Some(stop_file.clone());
        *self.inner.started_at.lock() = Some(Instant::now());

        let inner = Arc::clone(&self.inner);
        let app_dir = self.app_dir.clone();
        std::thread::Builder::new()
            .name("training-run".into())
            .spawn(move || {
                let outcome = run_worker(
                    &inner, &app, &app_dir, &data_dir, &workspace, &stop_file, &req, &ffmpeg,
                    &contentvec, &rmvpe_pt, &pretrain_g, &pretrain_d, &slug,
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
            .map_err(|e| UtaiError::Training(format!("启动训练线程失败: {}", e)))?;
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

#[allow(clippy::too_many_arguments)]
fn run_worker(
    inner: &Arc<Inner>,
    app: &tauri::AppHandle,
    app_dir: &Path,
    _data_dir: &Path,
    workspace: &Path,
    stop_file: &Path,
    req: &StartTrainingRequest,
    ffmpeg: &Path,
    contentvec: &Path,
    rmvpe_pt: &Path,
    pretrain_g: &Path,
    pretrain_d: &Path,
    slug: &str,
) -> Result<()> {
    // ---- stage: import the dataset into the ASCII workspace ----
    let dataset_dir = workspace.join("dataset");
    if dataset_dir.exists() {
        std::fs::remove_dir_all(&dataset_dir)?;
    }
    std::fs::create_dir_all(&dataset_dir)?;
    let total = req.dataset_files.len();
    for (i, f) in req.dataset_files.iter().enumerate() {
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
        std::fs::copy(src, &dst)
            .map_err(|e| UtaiError::Training(format!("导入数据 {} 失败: {}", src.display(), e)))?;
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

    // ---- run config for the sidecar ----
    let run_config = serde_json::json!({
        "backend": req.backend,
        "workspace": workspace,
        "dataset_dir": dataset_dir,
        "model_slug": slug,
        "sample_rate": req.sample_rate,
        "version": req.version,
        "total_epoch": req.total_epoch,
        "batch_size": req.batch_size,
        "save_every_epoch": req.save_every_epoch,
        "save_every_weights": req.save_every_weights,
        "keep_only_latest": req.keep_only_latest,
        "cache_gpu": req.cache_gpu,
        "fp16": req.fp16,
        "spk_id": req.spk_id,
        "seed": SEED,
        // Windows cannot hold an EMPTY env var (empty = deleted = all GPUs
        // visible) — CPU mode must be the explicit sentinel "-1"
        "gpu": if req.force_cpu { "-1".to_string() } else { req.gpu.to_string() },
        "stop_file": stop_file,
        "pretrain_g": pretrain_g,
        "pretrain_d": pretrain_d,
        "assets": {
            "ffmpeg": ffmpeg,
            "rmvpe_pt": rmvpe_pt,
            "contentvec_onnx": contentvec,
            "configs_dir": app_dir.join("training").join("assets").join("configs").join("rvc"),
            "mute_dir": app_dir.join("training").join("assets").join("mute"),
        },
    });
    let run_json = workspace.join("run.json");
    std::fs::write(&run_json, serde_json::to_vec_pretty(&run_config)?)?;

    // ---- spawn the sidecar ----
    if inner.abort.load(Ordering::SeqCst) {
        return abort_finish(inner, app);
    }
    let training_dir = app_dir.join("training");
    let python = crate::util::find_python(&training_dir, app_dir);
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
                "启动训练 Python 失败 ({}): {}（训练环境未配置？）",
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
                        msg["message"].as_str().unwrap_or("未知训练错误").to_string(),
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
        "训练进程异常退出 (exit code {:?})。常见原因：显存/内存不足(OOM)、被杀毒软件终止、训练环境损坏。\
         详细日志见训练工作区 train.log；最近输出已附在状态面板。",
        code
    )))
}
