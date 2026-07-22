//! Tauri commands for the embedded Python runtime packs (S42 Phase A).
//! Thin orchestration over crate::pyenv + crate::download; all UI progress flows
//! through ONE event channel:
//!   `pyenv-progress`  { id, phase, progress, message, code, params }
//!                     phase: manifest|download|verify|extract|envtest|done|error;
//!                     `code` is the stable CODE the Settings panel localizes from
//!                     (params = positional payload); `message` is the English
//!                     log/fallback text.
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
    /// English log/fallback text — the frontend renders from `code`+`params` when present.
    message: String,
    /// Stable stage/outcome/error CODE for frontend localization (None on legacy emits).
    code: Option<String>,
    /// Positional payload for the localized template (e.g. [name, doneMB, totalMB]).
    params: Vec<String>,
}

fn emit_progress(
    app: &tauri::AppHandle,
    id: &str,
    phase: &str,
    progress: f32,
    code: Option<&str>,
    params: Vec<String>,
    message: impl Into<String>,
) {
    let _ = app.emit(
        "pyenv-progress",
        PyenvProgress {
            id: id.to_string(),
            phase: phase.to_string(),
            progress,
            message: message.into(),
            code: code.map(str::to_string),
            params,
        },
    );
}

/// First SCREAMING_SNAKE token in an error message ("Runtime pack error: CODE: detail"
/// → CODE) — error emits carry it so the frontend localizes without string-matching.
fn leading_code(msg: &str) -> Option<&str> {
    msg.split(|c: char| c == ':' || c == '(' || c == ')' || c.is_whitespace()).find(|t| {
        t.len() >= 4
            && t.contains('_')
            && t.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
    })
}

/// Terminal "done" outcome of an install: `code`+`params` feed the frontend i18n
/// template; `message` stays the English log/fallback text.
struct DoneMsg {
    code: &'static str,
    params: Vec<String>,
    message: String,
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
    /// Whether THIS machine's hardware can actually run this variant (settings.rs
    /// `variant_supported`: CPU always; nv needs an sm_75+ NVIDIA card; amd/intel need
    /// the matching-vendor GPU). The UI hides download entries for unsupported variants
    /// so a box is only offered packs it can use. Local-file install is NOT gated by this.
    pub supported: bool,
}

#[tauri::command]
pub fn get_runtime_env_info() -> Result<RuntimeEnvInfo, String> {
    let root = pyenv::runtime_root()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();
    let root_ascii_ok = root.is_ascii() && !root.is_empty();
    let mut packs = pyenv::list_packs();
    // Hardware facts for the per-variant support gate — queried ONCE per refresh (each
    // is a subprocess: WMI for vendors, nvidia-smi for the NVIDIA compute cap).
    let gpus = crate::commands::settings::query_gpu_adapters();
    let nv_cc10 = crate::commands::settings::nvidia_compute_caps_cc10();
    // S74b: the SAME predicate decides both "offer this download" and "is the installed one
    // usable here" — one sentence, two consumers, no second rule to drift.
    let sig = crate::commands::settings::machine_sig();
    for p in &mut packs {
        p.supported = crate::commands::settings::variant_supported(&p.meta.variant, &gpus, &nv_cc10);
        // Only an EXPLICIT disagreement is staleness — a report written before the stamp existed
        // carries no claim about its machine, so it keeps its verdict (see PackStatus).
        p.envtest_stale = p
            .envtest
            .as_ref()
            .and_then(|e| e.get("machine"))
            .and_then(|m| m.as_str())
            .is_some_and(|m| m != sig);
    }
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
            supported: crate::commands::settings::variant_supported(&e.variant, &gpus, &nv_cc10),
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

/// S64 release gating (the S43 decision, frontend half in TrainingPage): TRUE when a REAL training
/// interpreter resolves (dev venv / installed runtime pack / manual python slot). FALSE = the spawn
/// would fall back to bare PATH `python` — doomed on end-user machines, so the start button shows
/// the "download a training runtime first" dialog instead. Mirrors the actual resolution by CALLING
/// pyenv::training_interpreter (never a forked re-implementation).
#[tauri::command]
pub fn training_env_ready(state: State<'_, Arc<AppState>>) -> bool {
    let (py, _) = pyenv::training_interpreter(&state.app_dir, false);
    py != std::path::Path::new("python")
}

/// S66 startup component check, converter twin of training_env_ready: TRUE when a REAL
/// converter interpreter resolves (dev venv / installed pack / manual slot) — FALSE = any
/// model import/conversion would die with RUNTIME_PACK_REQUIRED, so the startup dialog
/// offers the runtime download up front. Mirrors resolution by CALLING pyenv::converter_python.
#[tauri::command]
pub fn converter_env_ready(state: State<'_, Arc<AppState>>) -> bool {
    pyenv::converter_python(&state.app_dir) != std::path::Path::new("python")
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
        .ok_or_else(|| format!("PACK_UNKNOWN: {id}"))?;
    // S68f pre-download gate: cu130 = CUDA 13 runtime = NVIDIA driver r580+. The
    // variant gate only checks vendor + compute cap, so a perfectly capable card on a
    // pre-580 driver (community RTX 4070 Laptop: CUDA-12 inference fine, torch-cu130
    // zero devices) installed 1.8 GB it could not use. Loud actionable CODE instead of
    // hiding the pack (S64c CUDA_TOOLKIT_REQUIRED posture); unknown version fails open.
    if entry.variant == "nv-cu130" {
        if let Some(major) = crate::commands::settings::nvidia_driver_major() {
            if major < 580 {
                return Err(format!("RUNTIME_DRIVER_TOO_OLD: NVIDIA driver {major} < 580 (CUDA 13)"));
            }
        }
    }
    let (guard, cancel) = pyenv::InstallGuard::acquire().map_err(|e| e.to_string())?;
    let _task = state.begin_task("pyenv_install"); // close-flow in-progress listing
    let result = do_download_and_install(&app, &state, entry, &cancel).await;
    // Terminal events fire AFTER the guard drops: the frontend's done-handler
    // refreshes get_runtime_env_info, and a still-held guard would report
    // installing=true and wedge the rebuilt busy state (audit S42-r2).
    drop(guard);
    match &result {
        Ok(done) => {
            tracing::info!("runtime pack installed: {id}");
            emit_progress(&app, &id, "done", 1.0, Some(done.code), done.params.clone(), done.message.clone());
        }
        Err(e) => {
            tracing::error!("runtime pack install failed ({id}): {e}");
            emit_progress(&app, &id, "error", 0.0, leading_code(e), vec![], e.clone());
        }
    }
    result.map(|_| ())
}

async fn do_download_and_install(
    app: &tauri::AppHandle,
    state: &Arc<AppState>,
    entry: &pyenv::CatalogEntry,
    cancel: &Arc<AtomicBool>,
) -> Result<DoneMsg, String> {
    let id = entry.id;
    let root = pyenv::install_root().map_err(|e| e.to_string())?;
    let client = crate::download::client().map_err(|e| e.to_string())?;

    emit_progress(app, id, "manifest", 0.0, Some("STAGE_FETCH_MANIFEST"), vec![], "Fetching pack manifest...");
    let candidates = pyenv::manifest_url_candidates(entry);
    let (manifest, bases) = pyenv::fetch_manifest(&client, &candidates)
        .await
        .map_err(|e| e.to_string())?;
    if manifest.id != id {
        return Err(format!("MANIFEST_ID_MISMATCH: {} != {id}", manifest.id));
    }
    // Remote json feeds filesystem paths below — validate before ANY join.
    pyenv::validate_manifest(&manifest).map_err(|e| e.to_string())?;

    // Parts land in a resumable staging dir keyed on the pack id — a later retry
    // finds the .part files (and completed parts) exactly where they were.
    let dl_dir = root.join(".staging").join(format!("dl-{id}"));
    std::fs::create_dir_all(&dl_dir).map_err(|e| e.to_string())?;

    let total_bytes: u64 = manifest.parts.iter().map(|p| p.size).sum();

    // S68d disk preflight: the parts staging dir and the final tree share the runtimes
    // volume — peak usage = remaining download + extracted tree (the archives are only
    // reclaimed after commit). Bytes already staged (resume: whole parts or in-flight
    // .part files) are credited so a retry that mostly needs the extract step is never
    // refused. Fail open when the probe or the catalog size is unavailable; the extract
    // step re-checks its own (smaller) requirement against pack.json disk_bytes.
    // Size source: the freshly fetched manifest when it carries disk_bytes (真源 —
    // pack rebuilds change footprint without an app update, review S68d), the catalog
    // constant as fallback for older manifests.
    let disk_bytes = if manifest.disk_bytes > 0 { manifest.disk_bytes } else { entry.disk_bytes };
    if disk_bytes > 0 {
        if let Some(free) = crate::util::free_bytes_at(&root) {
            let staged: u64 = manifest
                .parts
                .iter()
                .map(|p| {
                    let done = std::fs::metadata(dl_dir.join(&p.name)).map(|m| m.len()).unwrap_or(0);
                    let inflight = std::fs::metadata(dl_dir.join(format!("{}.part", p.name)))
                        .map(|m| m.len())
                        .unwrap_or(0);
                    done.max(inflight).min(p.size)
                })
                .sum();
            let needed = total_bytes.saturating_sub(staged).saturating_add(disk_bytes);
            if free < needed {
                return Err(format!(
                    "INSTALL_DISK_FULL: {} MB needed, {} MB free at {}",
                    needed / 1_000_000,
                    free / 1_000_000,
                    root.display()
                ));
            }
        }
    }

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
            let mb_done = format!("{:.1}", abs as f64 / 1.0e6);
            let mb_total = format!("{:.1}", total_bytes as f64 / 1.0e6);
            emit_progress(
                &app2,
                id,
                "download",
                overall * 0.85,
                Some("STAGE_DOWNLOADING"),
                vec![name.clone(), mb_done.clone(), mb_total.clone()],
                format!("Downloading {name}  {mb_done} / {mb_total} MB"),
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
/// automatic post-install self-test. Returns the terminal "done" outcome — the
/// caller emits it after dropping the InstallGuard (never before: the frontend's
/// done-handler refresh must observe installing=false).
async fn install_parts(
    app: &tauri::AppHandle,
    state: &Arc<AppState>,
    id: &str,
    parts: Vec<PathBuf>,
    cancel: &Arc<AtomicBool>,
) -> Result<DoneMsg, String> {
    emit_progress(app, id, "extract", 0.86, Some("STAGE_EXTRACTING"), vec![], "Extracting runtime pack...");
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
                Some("STAGE_EXTRACTING"),
                vec![entries.to_string()],
                format!("Extracting runtime pack... {entries} files"),
            );
        })
    })
    .await
    .map_err(|e| format!("EXTRACT_TASK_FAILED: {e}"))?
    .map_err(|e| e.to_string())?;

    // Post-install self-test: pack correctness is only claimed once envtest says so —
    // an envtest failure does NOT roll back the install (the report shows what broke,
    // and 重新自检 is a button), but the UI badge stays red until it passes.
    if cancel.load(Ordering::SeqCst) {
        // Cancel landed after the commit: the pack IS installed — say so honestly
        // instead of running a self-test the user asked to stop waiting for.
        return Ok(DoneMsg {
            code: "INSTALLED_ENVTEST_SKIPPED",
            params: vec![],
            message: "Installed (cancel skipped the self-test — run it manually from the pack list).".to_string(),
        });
    }
    emit_progress(app, id, "envtest", 0.95, Some("STAGE_ENVTEST"), vec![], "Running environment self-test...");
    if let Err(e) = run_envtest_inner(app, state, &meta.id, Some(cancel)).await {
        tracing::warn!("post-install envtest failed for {}: {e}", meta.id);
        return Ok(DoneMsg {
            code: "INSTALLED_ENVTEST_FAILED",
            params: vec![e.clone()],
            message: format!("Installed, but the self-test failed: {e}"),
        });
    }
    Ok(DoneMsg {
        code: "INSTALL_DONE",
        params: vec![],
        message: "Install complete; self-test passed.".to_string(),
    })
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
            emit_progress(&app, &display_id, "verify", 0.3, Some("STAGE_VERIFY"), vec![], "Verifying part sha256...");
            let man2 = man.clone();
            let dir = picked.parent().map(|p| p.to_path_buf()).unwrap_or_default();
            tokio::task::spawn_blocking(move || pyenv::verify_parts(&man2, &dir))
                .await
                .map_err(|e| format!("VERIFY_TASK_FAILED: {e}"))?
                .map_err(|e| e.to_string())?;
        } else {
            emit_progress(
                &app,
                &display_id,
                "verify",
                0.3,
                Some("STAGE_VERIFY_SKIPPED"),
                vec![],
                "No manifest found next to the archive — skipping verification (only recommended for locally built packs)",
            );
        }
        let msg = install_parts(&app, &state, &display_id, parts, &cancel).await?;
        Ok::<(String, DoneMsg), String>((display_id, msg))
    }
    .await;
    // Terminal events strictly AFTER the guard drop — see download_runtime_pack.
    drop(guard);
    match &result {
        Ok((display_id, done)) => {
            tracing::info!("runtime pack installed from local archive: {path}");
            emit_progress(&app, display_id, "done", 1.0, Some(done.code), done.params.clone(), done.message.clone());
        }
        Err(e) => {
            // Parity with the download flow — a local-install failure must reach the
            // log file (S42: the first field failure left NO trace in utai.log).
            tracing::error!("local runtime pack install failed ({path}): {e}");
            emit_progress(&app, "local", "error", 0.0, leading_code(e), vec![], e.clone());
        }
    }
    result.map(|_| ())
}

#[tauri::command]
pub fn cancel_runtime_install() -> Result<bool, String> {
    Ok(pyenv::cancel_active_install())
}

#[tauri::command]
pub async fn delete_runtime_pack(
    state: State<'_, Arc<AppState>>,
    id: String,
) -> Result<(), String> {
    // S74b: a runtime pack IS the interpreter that training and model conversion run on — pulling
    // it out mid-job breaks the child process in a way that surfaces nowhere near the cause.
    // Fail-closed pre-flight (see ensure_idle_for_package_delete); the frontend refuses earlier
    // with a nicer message, this is the TOCTOU backstop.
    crate::commands::window::ensure_idle_for_package_delete(&state)?;
    // remove_dir_all over a ~1 GB / 10k-file tree takes seconds (more under AV) —
    // run it off the IPC pool; the frontend shows a 删除中… state meanwhile.
    tokio::task::spawn_blocking(move || pyenv::delete_pack(&id))
        .await
        .map_err(|e| format!("DELETE_TASK_FAILED: {e}"))?
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

/// Per-failed-check detail cap — a check's detail carries a source location, and an error
/// toast must not become a wall of text (the Settings panel renders the full breakdown).
const ENVTEST_DETAIL_MAX: usize = 160;

/// How many failed checks carry their detail into the error string. Failures CASCADE (a dead
/// torch backend fails every on-device check after it), and the FIRST one is the root cause —
/// the rest are named by count so the headline stays readable.
const ENVTEST_DETAILED_ITEMS: usize = 3;

/// "check: root cause | check2: root cause2" over a failed report's items (S74).
/// The message used to carry the failed check NAMES only ("torch_backend") — the community
/// Iris Xe reporter had no way to tell WHY, which is exactly the guessing game the error-UX
/// rules forbid. Details are the checks' own raw/technical strings (stable CODEs where the
/// check emits one, e.g. XPU_NO_DEVICE), truncated. Falls back to `failed_names` for a report
/// that has no items array (defensive — every schema-1 report has one).
fn failed_items_summary(rep: &serde_json::Value) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut extra = 0usize;
    if let Some(items) = rep.get("items").and_then(|v| v.as_array()) {
        for it in items {
            if it.get("status").and_then(|v| v.as_str()) != Some("fail") {
                continue;
            }
            let name = it.get("name").and_then(|v| v.as_str()).unwrap_or("?");
            if parts.len() >= ENVTEST_DETAILED_ITEMS {
                extra += 1;
                continue;
            }
            let detail = it.get("detail").and_then(|v| v.as_str()).unwrap_or("").trim();
            if detail.is_empty() {
                parts.push(name.to_string());
            } else {
                // char-wise (details are UTF-8 and may be non-ASCII) — never split a code point.
                let short: String = detail.chars().take(ENVTEST_DETAIL_MAX).collect();
                let ellipsis = if detail.chars().count() > ENVTEST_DETAIL_MAX { "…" } else { "" };
                parts.push(format!("{name}: {short}{ellipsis}"));
            }
        }
    }
    if extra > 0 {
        parts.push(format!("+{extra}"));
    }
    if parts.is_empty() {
        return rep
            .get("failed_names")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|s| s.as_str()).collect::<Vec<_>>().join(", "))
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "unknown".into());
    }
    parts.join(" | ")
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
    let pack = pyenv::find_pack(id).ok_or_else(|| format!("PACK_NOT_FOUND: {id}"))?;
    let pack_dir = PathBuf::from(&pack.path);
    let python = pyenv::pack_python(&pack_dir);
    if !python.exists() {
        return Err(format!("PACK_NO_PYTHON: {}", python.display()));
    }
    let training_dir = state.app_dir.join("training");
    if !training_dir.join("utai_train").join("envtest.py").exists() {
        return Err(format!(
            "ENVTEST_SCRIPT_MISSING: {}",
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
        Err(e) => return Err(format!("ENVTEST_REPORT_CLEAR_FAILED: {}: {e}", report_path.display())),
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
        .map_err(|e| format!("ENVTEST_SPAWN_FAILED: {}: {e}", python.display()))?;

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
                return Err("ENVTEST_CANCELLED".into());
            }
        }
        // Child::wait is documented cancel-safe — timing out the future and
        // re-awaiting it later loses nothing.
        match tokio::time::timeout(std::time::Duration::from_millis(500), child.wait()).await {
            Ok(res) => break res.map_err(|e| format!("ENVTEST_WAIT_FAILED: {e}"))?,
            Err(_) => {
                if tokio::time::Instant::now() >= deadline {
                    let _ = child.kill().await;
                    return Err(format!(
                        "ENVTEST_TIMEOUT: killed after {} min",
                        ENVTEST_TIMEOUT.as_secs() / 60
                    ));
                }
            }
        }
    };
    let _ = stdout_task.await;
    let stderr_tail = stderr_task.await.unwrap_or_default();

    let mut report: Option<serde_json::Value> = std::fs::read_to_string(&report_path)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok());
    // S74b: stamp WHICH MACHINE produced this verdict, so a report that outlives the hardware it
    // describes can be shown as "re-run me" instead of masquerading as a current pass/fail. Written
    // here rather than in envtest.py because the hardware inventory lives on this side (and the cpu
    // tier has no GPU context at all inside python).
    if let Some(rep) = report.as_mut().and_then(|r| r.as_object_mut()) {
        rep.insert(
            "machine".to_string(),
            serde_json::Value::String(crate::commands::settings::machine_sig()),
        );
        if let Ok(text) = serde_json::to_string_pretty(&report) {
            if let Err(e) = std::fs::write(&report_path, text) {
                // Non-fatal: the verdict below is unaffected; only staleness detection degrades.
                tracing::warn!("could not stamp the machine signature into {}: {e}", report_path.display());
            }
        }
    }
    match report {
        Some(rep) => {
            let overall = rep.get("overall").and_then(|v| v.as_str()).unwrap_or("unknown");
            if overall == "pass" && !status.success() {
                // Belt-and-suspenders with the stale-report deletion above: a pass
                // report must AGREE with the exit code (envtest exits 0 iff no fails).
                return Err(format!(
                    "ENVTEST_REPORT_CONTRADICTION: overall=pass but exit code {:?}. stderr tail:\n{}",
                    status.code(),
                    stderr_tail.iter().cloned().collect::<Vec<_>>().join("\n")
                ));
            }
            if overall == "pass" {
                Ok(rep)
            } else {
                // S74b: log the RAW verdict, uncapped, one line per failed check — in LOG format,
                // not the localized text the panel shows (a trilingual sentence is worthless in a
                // bug report). Both entry points reach here (manual self-test and the post-install
                // run), so this is the single place that guarantees it lands on disk.
                //
                // It matters more than it looks: with the S74b gates, a pack that is offered,
                // installed AND still fails its self-test usually means something let an
                // unsupported pack through (residue, a hand-dropped folder, a machine swap) — the
                // log is the only thing that can tell us WHICH, and the per-check detail carries
                // the stable CODE (XPU_NO_DEVICE, RUNTIME_DRIVER_TOO_OLD, …) that names it.
                for it in rep.get("items").and_then(|v| v.as_array()).into_iter().flatten() {
                    if it.get("status").and_then(|v| v.as_str()) == Some("fail") {
                        tracing::warn!(
                            "envtest FAILED [{}] check={} detail={}",
                            id,
                            it.get("name").and_then(|v| v.as_str()).unwrap_or("?"),
                            it.get("detail").and_then(|v| v.as_str()).unwrap_or("")
                        );
                    }
                }
                Err(format!("ENVTEST_FAILED: {}", failed_items_summary(&rep)))
            }
        }
        None => {
            let mut msg = format!(
                "ENVTEST_CRASHED: exit code {:?}, no report. stderr tail:\n{}",
                status.code(),
                stderr_tail.iter().cloned().collect::<Vec<_>>().join("\n")
            );
            // S68f diagnosis: the classic no-report crash on nv-cu130 is a pre-580
            // driver ACCESS-VIOLATing inside the CUDA 13 runtime probe — say so
            // instead of leaving the user a bare stderr tail (packs installed before
            // the download gate existed keep hitting this path).
            if pack.meta.variant == "nv-cu130" {
                if let Some(major) = crate::commands::settings::nvidia_driver_major() {
                    if major < 580 {
                        msg.push_str(&format!(
                            "\n[driver check: NVIDIA {major} < 580 — CUDA 13 requires an r580+ driver; update it at nvidia.com]"
                        ));
                    }
                }
            }
            Err(msg)
        }
    }
}

/// Download-source connection test (S43) — a REAL few-MB transfer, not a ping/HEAD
/// (which false-positives under GFW small-packet passthrough: it lets the handshake +
/// small responses through, then throttles/resets sustained transfers). `url` is the
/// mirror-resolved URL of a real published asset (the frontend applies the mirror
/// transform). Never errors — the ProbeResult carries the reason so the UI shows
/// "通畅 / 偏慢 / 疑似被限速 / 不通" instead of a toast.
#[tauri::command]
pub async fn test_download_source(url: String) -> crate::download::ProbeResult {
    crate::download::probe(&url).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The xpu-tier report shape captured from a REAL headless run
    /// (`python -m utai_train.envtest --device xpu` on a non-XPU box, S74): a dead torch
    /// backend plus the three on-device checks that cascade off it.
    fn xpu_report() -> serde_json::Value {
        json!({
            "overall": "fail",
            "items": [
                {"name": "python_info", "status": "pass", "detail": "CPython 3.10.20"},
                {"name": "torch_backend", "status": "fail",
                 "detail": "RuntimeError: XPU_NO_DEVICE: torch.xpu.is_available()=False (torch 2.11.0+cpu) @ File \"envtest.py\", line 228, in check_torch_backend"},
                {"name": "tiny_gan", "status": "fail",
                 "detail": "AssertionError: Torch not compiled with XPU enabled"},
                {"name": "gpu_stft_vs_cpu", "status": "fail",
                 "detail": "AssertionError: Torch not compiled with XPU enabled"},
                {"name": "gpu_amp_step", "status": "fail",
                 "detail": "AssertionError: Torch not compiled with XPU enabled"},
            ],
            "failed_names": ["torch_backend", "tiny_gan", "gpu_stft_vs_cpu", "gpu_amp_step"],
        })
    }

    #[test]
    fn summary_carries_root_cause_code_not_just_check_names() {
        let s = failed_items_summary(&xpu_report());
        // THE regression this guards: the message used to be "torch_backend, tiny_gan, …".
        assert!(s.contains("XPU_NO_DEVICE"), "{s}");
        assert!(s.starts_with("torch_backend: "), "{s}");
        // Cascaded failures are capped by count, not spelled out.
        assert!(s.ends_with("+1"), "{s}");
        // 3 detailed items + the "+1" tail = 3 joins; a detail's own location separator is
        // "@", never " | ", so the join stays unambiguous (guards that contract too).
        assert_eq!(s.matches(" | ").count(), 3, "{s}");
    }

    #[test]
    fn summary_truncates_on_char_boundaries() {
        let long: String = "错".repeat(400); // multibyte: a byte-wise cut would panic
        let rep = json!({"items": [{"name": "x", "status": "fail", "detail": long}]});
        let s = failed_items_summary(&rep);
        assert!(s.starts_with("x: 错"), "{s}");
        assert!(s.ends_with('…'), "{s}");
        assert_eq!(s.chars().filter(|c| *c == '错').count(), ENVTEST_DETAIL_MAX);
    }

    #[test]
    fn summary_falls_back_to_names_then_unknown() {
        // No items array (report older than schema 1's items, or hand-edited).
        let rep = json!({"failed_names": ["imports", "faiss_search"]});
        assert_eq!(failed_items_summary(&rep), "imports, faiss_search");
        // Nothing usable at all — never emit an empty message.
        assert_eq!(failed_items_summary(&json!({})), "unknown");
        assert_eq!(failed_items_summary(&json!({"failed_names": []})), "unknown");
        // Detail-less items still name their check.
        let rep = json!({"items": [{"name": "faiss_search", "status": "fail", "detail": "  "}]});
        assert_eq!(failed_items_summary(&rep), "faiss_search");
    }

    #[test]
    fn summary_ignores_pass_warn_skip_items() {
        let rep = json!({"items": [
            {"name": "a", "status": "pass", "detail": "ok"},
            {"name": "b", "status": "warn", "detail": "slow"},
            {"name": "c", "status": "skip", "detail": "n/a"},
        ]});
        // No failures in the items → falls through to the names path → "unknown"
        // (this function is only ever called on an overall=fail report).
        assert_eq!(failed_items_summary(&rep), "unknown");
    }
}
