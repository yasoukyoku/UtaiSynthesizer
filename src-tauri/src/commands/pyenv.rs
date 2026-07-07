//! Tauri commands for the embedded Python runtime packs (S42 Phase A).
//! Thin orchestration over crate::pyenv + crate::download; all UI progress flows
//! through ONE event channel:
//!   `pyenv-progress`  { id, phase, progress, message }   phase: manifest|download|
//!                                                        verify|extract|envtest|done|error
//!   `pyenv-envtest`   { id, event }                      raw envtest JSONL objects

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tauri::{Emitter, State};

use crate::pyenv;
use crate::AppState;

#[derive(serde::Serialize, Clone)]
struct PyenvProgress {
    id: String,
    phase: String,
    progress: f32,
    message: String,
}

fn emit_progress(app: &tauri::AppHandle, id: &str, phase: &str, progress: f32, message: impl Into<String>) {
    let _ = app.emit(
        "pyenv-progress",
        PyenvProgress {
            id: id.to_string(),
            phase: phase.to_string(),
            progress,
            message: message.into(),
        },
    );
}

#[derive(serde::Serialize)]
pub struct RuntimeEnvInfo {
    pub root: String,
    pub root_ascii_ok: bool,
    pub packs: Vec<pyenv::PackStatus>,
    pub catalog: Vec<CatalogItem>,
    /// Busy flags so a REOPENED settings panel can rebuild its state — component
    /// state alone loses "installing" on unmount and strands the cancel button.
    pub installing: bool,
    pub envtest_running: bool,
}

#[derive(serde::Serialize)]
pub struct CatalogItem {
    pub id: String,
    pub variant: String,
    pub label: String,
    pub download_bytes: u64,
    pub disk_bytes: u64,
    pub experimental: bool,
    /// Any manifest source known (published URLs or the dev UTAI_PACK_BASE_URL
    /// override) — drives whether the 下载 button shows at all.
    pub downloadable: bool,
    pub installed: bool,
}

#[tauri::command]
pub fn get_runtime_env_info() -> Result<RuntimeEnvInfo, String> {
    let root = pyenv::runtime_root()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();
    let root_ascii_ok = root.is_ascii() && !root.is_empty();
    let packs = pyenv::list_packs();
    let catalog = pyenv::CATALOG
        .iter()
        .map(|e| CatalogItem {
            id: e.id.to_string(),
            variant: e.variant.to_string(),
            label: e.label.to_string(),
            download_bytes: e.download_bytes,
            disk_bytes: e.disk_bytes,
            experimental: e.experimental,
            downloadable: !pyenv::manifest_url_candidates(e).is_empty(),
            // By ID, not variant: during a v1→v2 upgrade both coexist and the v2
            // catalog row must stay visible/downloadable while v1 is installed.
            installed: packs.iter().any(|p| p.meta.id == e.id),
        })
        .collect();
    Ok(RuntimeEnvInfo {
        root,
        root_ascii_ok,
        packs,
        catalog,
        installing: pyenv::install_active(),
        envtest_running: pyenv::envtest_active(),
    })
}

#[tauri::command]
pub async fn download_runtime_pack(
    app: tauri::AppHandle,
    state: State<'_, Arc<AppState>>,
    id: String,
) -> Result<(), String> {
    let entry = pyenv::CATALOG
        .iter()
        .find(|e| e.id == id)
        .ok_or_else(|| format!("未知运行时包: {id}"))?;
    let (guard, cancel) = pyenv::InstallGuard::acquire().map_err(|e| e.to_string())?;
    let _task = state.begin_task("pyenv_install"); // close-flow in-progress listing
    let result = do_download_and_install(&app, &state, entry, &cancel).await;
    // Terminal events fire AFTER the guard drops: the frontend's done-handler
    // refreshes get_runtime_env_info, and a still-held guard would report
    // installing=true and wedge the rebuilt busy state (audit S42-r2).
    drop(guard);
    match &result {
        Ok(msg) => {
            tracing::info!("runtime pack installed: {id}");
            emit_progress(&app, &id, "done", 1.0, msg.clone());
        }
        Err(e) => {
            tracing::error!("runtime pack install failed ({id}): {e}");
            emit_progress(&app, &id, "error", 0.0, e.clone());
        }
    }
    result.map(|_| ())
}

async fn do_download_and_install(
    app: &tauri::AppHandle,
    state: &Arc<AppState>,
    entry: &pyenv::CatalogEntry,
    cancel: &Arc<AtomicBool>,
) -> Result<String, String> {
    let id = entry.id;
    let root = pyenv::install_root().map_err(|e| e.to_string())?;
    let client = crate::download::client().map_err(|e| e.to_string())?;

    emit_progress(app, id, "manifest", 0.0, "获取包清单...");
    let candidates = pyenv::manifest_url_candidates(entry);
    let (manifest, bases) = pyenv::fetch_manifest(&client, &candidates)
        .await
        .map_err(|e| e.to_string())?;
    if manifest.id != id {
        return Err(format!("清单 id 不匹配（{} ≠ {id}）", manifest.id));
    }
    // Remote json feeds filesystem paths below — validate before ANY join.
    pyenv::validate_manifest(&manifest).map_err(|e| e.to_string())?;

    // Parts land in a resumable staging dir keyed on the pack id — a later retry
    // finds the .part files (and completed parts) exactly where they were.
    let dl_dir = root.join(".staging").join(format!("dl-{id}"));
    std::fs::create_dir_all(&dl_dir).map_err(|e| e.to_string())?;

    let total_bytes: u64 = manifest.parts.iter().map(|p| p.size).sum();
    let mut done_before: u64 = 0;
    for part in &manifest.parts {
        let urls: Vec<String> = bases.iter().map(|b| format!("{b}/{}", part.name)).collect();
        let req = crate::download::DownloadRequest {
            urls,
            dest: dl_dir.join(&part.name),
            sha256: Some(part.sha256.clone()),
            expected_size: Some(part.size),
        };
        let app2 = app.clone();
        let name = part.name.clone();
        // Throttle: raw per-chunk emits are hundreds of IPC+setState per second for
        // the whole download (the old msst downloader throttles at 1 MB for exactly
        // this reason; S35's ORT log flood is the same failure shape).
        let mut last_emitted: u64 = 0;
        crate::download::download(&client, &req, cancel, |done, total| {
            let abs = done_before + done;
            let complete = total.map(|t| done >= t).unwrap_or(false);
            if abs.saturating_sub(last_emitted) < 2_000_000 && !complete && abs != 0 {
                return;
            }
            last_emitted = abs;
            let overall = abs as f32 / total_bytes.max(1) as f32;
            emit_progress(
                &app2,
                id,
                "download",
                overall * 0.85,
                format!(
                    "下载 {name}  {:.1} / {:.1} MB",
                    abs as f64 / 1.0e6,
                    total_bytes as f64 / 1.0e6
                ),
            );
        })
        .await
        .map_err(|e| e.to_string())?;
        done_before += part.size;
    }

    let parts: Vec<PathBuf> = manifest.parts.iter().map(|p| dl_dir.join(&p.name)).collect();
    let msg = install_parts(app, state, id, parts, cancel).await?;
    // Reclaim the downloaded archives only after a successful commit.
    let _ = std::fs::remove_dir_all(&dl_dir);
    Ok(msg)
}

/// Shared tail of both install flows: extract+commit (blocking task), then the
/// automatic post-install self-test. Returns the terminal "done" MESSAGE — the
/// caller emits it after dropping the InstallGuard (never before: the frontend's
/// done-handler refresh must observe installing=false).
async fn install_parts(
    app: &tauri::AppHandle,
    state: &Arc<AppState>,
    id: &str,
    parts: Vec<PathBuf>,
    cancel: &Arc<AtomicBool>,
) -> Result<String, String> {
    emit_progress(app, id, "extract", 0.86, "解压运行时包...");
    let app2 = app.clone();
    let id2 = id.to_string();
    let cancel2 = Arc::clone(cancel);
    let meta = tokio::task::spawn_blocking(move || {
        pyenv::extract_and_commit(&parts, &cancel2, |entries| {
            emit_progress(
                &app2,
                &id2,
                "extract",
                0.86,
                format!("解压运行时包... {entries} 个文件"),
            );
        })
    })
    .await
    .map_err(|e| format!("解压任务失败: {e}"))?
    .map_err(|e| e.to_string())?;

    // Post-install self-test: pack correctness is only claimed once envtest says so —
    // an envtest failure does NOT roll back the install (the report shows what broke,
    // and 重新自检 is a button), but the UI badge stays red until it passes.
    if cancel.load(Ordering::SeqCst) {
        // Cancel landed after the commit: the pack IS installed — say so honestly
        // instead of running a self-test the user asked to stop waiting for.
        return Ok("已安装（取消跳过了自检——可在列表中手动自检）。".to_string());
    }
    emit_progress(app, id, "envtest", 0.95, "运行环境自检...");
    if let Err(e) = run_envtest_inner(app, state, &meta.id, Some(cancel)).await {
        tracing::warn!("post-install envtest failed for {}: {e}", meta.id);
        return Ok(format!("已安装，但自检未通过：{e}"));
    }
    Ok("安装完成，自检通过。".to_string())
}

#[tauri::command]
pub async fn install_runtime_pack_local(
    app: tauri::AppHandle,
    state: State<'_, Arc<AppState>>,
    path: String,
) -> Result<(), String> {
    let picked = PathBuf::from(&path);
    let (guard, cancel) = pyenv::InstallGuard::acquire().map_err(|e| e.to_string())?;
    let _task = state.begin_task("pyenv_install");
    let result = async {
        let (parts, manifest) = pyenv::resolve_local_parts(&picked).map_err(|e| e.to_string())?;
        let display_id = manifest
            .as_ref()
            .map(|m| m.id.clone())
            .unwrap_or_else(|| "local".to_string());
        if let Some(man) = &manifest {
            emit_progress(&app, &display_id, "verify", 0.3, "校验分卷 sha256...");
            let man2 = man.clone();
            let dir = picked.parent().map(|p| p.to_path_buf()).unwrap_or_default();
            tokio::task::spawn_blocking(move || pyenv::verify_parts(&man2, &dir))
                .await
                .map_err(|e| format!("校验任务失败: {e}"))?
                .map_err(|e| e.to_string())?;
        } else {
            emit_progress(
                &app,
                &display_id,
                "verify",
                0.3,
                "未找到 manifest——跳过校验（仅建议用于本地构建的包）",
            );
        }
        let msg = install_parts(&app, &state, &display_id, parts, &cancel).await?;
        Ok::<(String, String), String>((display_id, msg))
    }
    .await;
    // Terminal events strictly AFTER the guard drop — see download_runtime_pack.
    drop(guard);
    match &result {
        Ok((display_id, msg)) => {
            tracing::info!("runtime pack installed from local archive: {path}");
            emit_progress(&app, display_id, "done", 1.0, msg.clone());
        }
        Err(e) => {
            // Parity with the download flow — a local-install failure must reach the
            // log file (S42: the first field failure left NO trace in utai.log).
            tracing::error!("local runtime pack install failed ({path}): {e}");
            emit_progress(&app, "local", "error", 0.0, e.clone());
        }
    }
    result.map(|_| ())
}

#[tauri::command]
pub fn cancel_runtime_install() -> Result<bool, String> {
    Ok(pyenv::cancel_active_install())
}

#[tauri::command]
pub async fn delete_runtime_pack(id: String) -> Result<(), String> {
    // remove_dir_all over a ~1 GB / 10k-file tree takes seconds (more under AV) —
    // run it off the IPC pool; the frontend shows a 删除中… state meanwhile.
    tokio::task::spawn_blocking(move || pyenv::delete_pack(&id))
        .await
        .map_err(|e| format!("删除任务失败: {e}"))?
        .map_err(|e| e.to_string())
}

// ─── envtest ────────────────────────────────────────────────────────────────

/// Hard ceiling on a self-test run — a wedged interpreter (e.g. antivirus holding
/// python.exe) must not leave a zombie + a forever-spinning badge.
const ENVTEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15 * 60);

/// The envtest tier a pack must pass, from its variant. AMD deliberately maps to
/// "cuda": torch-hip exposes the `torch.cuda.*` namespace (design §4.2), so the
/// cuda-tier checks ARE the ROCm checks. Forgetting this mapping when GPU packs
/// land would hand GPU packs a green badge from a cpu-tier run — the exact silent
/// false-green §2.6 exists to prevent.
fn envtest_device_for_variant(variant: &str) -> &'static str {
    match variant {
        v if v.starts_with("nv") => "cuda",
        "amd" => "cuda",
        "xpu" => "xpu",
        _ => "cpu",
    }
}

#[tauri::command]
pub async fn run_pack_envtest(
    app: tauri::AppHandle,
    state: State<'_, Arc<AppState>>,
    id: String,
) -> Result<serde_json::Value, String> {
    run_envtest_inner(&app, &state, &id, None).await
}

async fn run_envtest_inner(
    app: &tauri::AppHandle,
    state: &Arc<AppState>,
    id: &str,
    cancel: Option<&Arc<AtomicBool>>,
) -> Result<serde_json::Value, String> {
    use tokio::io::AsyncBufReadExt;

    let _busy = pyenv::EnvtestGuard::acquire().map_err(|e| e.to_string())?;
    let pack = pyenv::find_pack(id).ok_or_else(|| format!("运行时包不存在: {id}"))?;
    let pack_dir = PathBuf::from(&pack.path);
    let python = pyenv::pack_python(&pack_dir);
    if !python.exists() {
        return Err(format!("包内缺少 python.exe: {}", python.display()));
    }
    let training_dir = state.app_dir.join("training");
    if !training_dir.join("utai_train").join("envtest.py").exists() {
        return Err(format!(
            "找不到自检脚本（{}）——应用目录不完整？",
            training_dir.join("utai_train").join("envtest.py").display()
        ));
    }
    let report_path = pack_dir.join("envtest.json");
    // A STALE report is a lie waiting to happen: if this run's python dies before
    // main() writes --out (AV kill, broken import, native crash), reading the old
    // file would return the previous verdict as this run's. Delete first so the
    // no-report path stays loud.
    match std::fs::remove_file(&report_path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(format!("无法清除旧自检报告（{}）: {e}", report_path.display())),
    }
    let device = envtest_device_for_variant(&pack.meta.variant);

    let mut cmd = tokio::process::Command::from(crate::util::python_command(&python));
    cmd.current_dir(&training_dir)
        .arg("-m")
        .arg("utai_train.envtest")
        .arg("--out")
        .arg(&report_path)
        .arg("--device")
        .arg(device)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    if device == "xpu" {
        // Design §4.1: must be set before torch import — missing operators become
        // loud-but-diagnosable CPU fallbacks (reported by envtest) instead of aborts.
        cmd.env("PYTORCH_ENABLE_XPU_FALLBACK", "1");
        cmd.env("PYTORCH_DEBUG_XPU_FALLBACK", "1");
    }
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("启动自检失败 ({}): {e}", python.display()))?;

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let app2 = app.clone();
    let id2 = id.to_string();
    let stdout_task = tokio::spawn(async move {
        if let Some(out) = stdout {
            let mut lines = tokio::io::BufReader::new(out).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
                    let _ = app2.emit("pyenv-envtest", serde_json::json!({ "id": id2, "event": v }));
                }
            }
        }
    });
    // Drain stderr into a bounded tail (surfaced on failure — same posture as the
    // training manager's ring buffer).
    let stderr_task = tokio::spawn(async move {
        let mut tail: std::collections::VecDeque<String> = std::collections::VecDeque::new();
        if let Some(errs) = stderr {
            let mut lines = tokio::io::BufReader::new(errs).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                tracing::debug!(target: "utai", "[envtest] {line}");
                if tail.len() >= 40 {
                    tail.pop_front();
                }
                tail.push_back(line);
            }
        }
        tail
    });

    // Wait with BOTH a hard timeout and (install flow) cancel polling — the install
    // cancel button must reach this phase too, not just download/extract.
    let deadline = tokio::time::Instant::now() + ENVTEST_TIMEOUT;
    let status = loop {
        if let Some(c) = cancel {
            if c.load(Ordering::SeqCst) {
                let _ = child.kill().await;
                return Err("自检已取消（包已安装，可稍后在列表中手动自检）".into());
            }
        }
        // Child::wait is documented cancel-safe — timing out the future and
        // re-awaiting it later loses nothing.
        match tokio::time::timeout(std::time::Duration::from_millis(500), child.wait()).await {
            Ok(res) => break res.map_err(|e| format!("自检进程等待失败: {e}"))?,
            Err(_) => {
                if tokio::time::Instant::now() >= deadline {
                    let _ = child.kill().await;
                    return Err(format!(
                        "自检超时（>{} 分钟）已终止——检查杀毒软件是否拦截了内嵌 python.exe",
                        ENVTEST_TIMEOUT.as_secs() / 60
                    ));
                }
            }
        }
    };
    let _ = stdout_task.await;
    let stderr_tail = stderr_task.await.unwrap_or_default();

    let report: Option<serde_json::Value> = std::fs::read_to_string(&report_path)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok());
    match report {
        Some(rep) => {
            let overall = rep.get("overall").and_then(|v| v.as_str()).unwrap_or("unknown");
            if overall == "pass" && !status.success() {
                // Belt-and-suspenders with the stale-report deletion above: a pass
                // report must AGREE with the exit code (envtest exits 0 iff no fails).
                return Err(format!(
                    "自检报告与进程退出码矛盾（overall=pass, code {:?}）——按失败处理。stderr 尾部：\n{}",
                    status.code(),
                    stderr_tail.iter().cloned().collect::<Vec<_>>().join("\n")
                ));
            }
            if overall == "pass" {
                Ok(rep)
            } else {
                Err(format!(
                    "自检未通过（详情见包内 envtest.json / 设置面板）：{}",
                    rep.get("failed_names")
                        .and_then(|v| v.as_array())
                        .map(|a| a
                            .iter()
                            .filter_map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .join(", "))
                        .unwrap_or_else(|| "unknown".into())
                ))
            }
        }
        None => Err(format!(
            "自检异常退出（code {:?}）且未产出报告。stderr 尾部：\n{}",
            status.code(),
            stderr_tail.iter().cloned().collect::<Vec<_>>().join("\n")
        )),
    }
}
