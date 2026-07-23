use std::path::PathBuf;
use std::sync::Arc;
use tauri::State;

use crate::training::{StartTrainingRequest, StepPoint, TrainingSnapshot};
use crate::AppState;

fn data_root(state: &AppState) -> PathBuf {
    // data root = parent of the models dir (data/models -> data/)
    state
        .models
        .models_dir()
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| state.app_dir.join("data"))
}

#[tauri::command]
pub async fn start_training(
    app: tauri::AppHandle,
    state: State<'_, Arc<AppState>>,
    request: StartTrainingRequest,
) -> Result<(), String> {
    // S41 audition interlock (red-team R4/A2; 审查修复 S41-INT-4): HOLD the
    // audition flag for the whole start sequence — a conversion subprocess may
    // be writing into <workspace>/audition and its ONNX sessions hold Windows
    // file locks; a mere load() check would leave a check-then-act window for
    // an audition to slip in mid-start. The frontend disables the button too;
    // this guard is the authoritative gate.
    let _audition_lock =
        crate::commands::audition::FlightGuard::acquire(crate::commands::audition::BUSY_RETRY_MSG)?;
    // S66: training ↔ conversion are excluded BOTH ways (each forks a multi-GB torch python;
    // the convert side checks training.is_active() in acquire_convert_slot).
    if state.task_active("convert") {
        return Err("CONVERT_BUSY".into());
    }
    let data_dir = data_root(&state);
    let audition_dir =
        crate::training::slot_path(&data_dir, &request.model_name, &request.backend).join("audition");
    // BEFORE manager.start(): drop every audition session (file locks) so the
    // fresh-wipe path inside try_start cannot trip over them. Non-destructive —
    // an evicted session reloads on miss.
    state.inference.engine.unload_paths_with_prefix(&audition_dir);
    state
        .training
        .start(app, data_dir, request)
        .map_err(|e| e.to_string())?;
    // AFTER a successful launch (never on guard-rejected starts, red-team R10 —
    // a rejected start must not cost the user their audition cache): the new
    // run's candidate list supersedes the old one.
    if audition_dir.exists() {
        if let Err(e) = std::fs::remove_dir_all(&audition_dir) {
            tracing::warn!("audition dir cleanup failed (non-fatal): {}", e);
        }
    }
    // torch needs the VRAM — every ORT GPU session goes (CPU aux stays warm;
    // reload-on-miss restores them later). Doing this on the failure path
    // would evict the whole fleet for nothing.
    state.inference.engine.release_gpu_sessions_except(&[]);
    Ok(())
}

/// One row of the S66 pre-start asset check (mirrors resolve_training_assets — the single
/// source try_start verifies against, so this pre-flight can never drift from the real gate).
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RequiredAssetStatus {
    pub label: String,
    pub path: String,
    pub exists: bool,
    /// Asset-pack id covering this file (drives the one-click download button); None = not
    /// pack-distributed.
    pub pack: Option<String>,
    /// S75: license id when this file's pack carries its own terms (CC BY-NC-SA for the vocoder
    /// base). Present ⇒ the download dialog MUST say so before fetching — we mirror those
    /// weights, we don't own them.
    pub license: Option<String>,
    /// Upstream release page. Was "you must download this yourself" (pre-S75); now it is the
    /// attribution link + the offline escape hatch when no HF host answers.
    pub self_url: Option<String>,
}

#[tauri::command]
pub fn training_required_assets(
    state: State<'_, Arc<AppState>>,
    backend: String,
    version: String,
    sample_rate: String,
    aug_copies: u32,
) -> Result<Vec<RequiredAssetStatus>, String> {
    let data_dir = data_root(&state);
    let assets =
        crate::training::resolve_training_assets(&data_dir, &backend, &version, &sample_rate, aug_copies)
            .map_err(|e| e.to_string())?;
    let models_dir = data_dir.join("models");
    Ok(assets
        .required
        .into_iter()
        .map(|(label, p)| {
            let rel = p
                .strip_prefix(&models_dir)
                .ok()
                .map(|r| r.to_string_lossy().replace('\\', "/"));
            let pack = rel
                .as_deref()
                .and_then(crate::commands::assets::pack_for_rel)
                .map(|s| s.to_string());
            // S75: license + upstream come from the SAME catalog entry as `pack` (assets.rs), so a
            // license-bound file can never be offered for one-click download without its terms.
            // Pre-S75 this was a hardcoded "training/vocoder/ ⇒ self-download URL" special case
            // living here, one table away from the catalog it described.
            let (license, upstream) = rel
                .as_deref()
                .map(crate::commands::assets::pack_terms_for_rel)
                .unwrap_or((None, None));
            RequiredAssetStatus {
                label,
                path: p.to_string_lossy().to_string(),
                exists: p.is_file(),
                pack,
                license: license.map(str::to_string),
                self_url: upstream.map(str::to_string),
            }
        })
        .collect())
}

#[tauri::command]
pub async fn stop_training(state: State<'_, Arc<AppState>>) -> Result<(), String> {
    state.training.stop().map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn force_stop_training(state: State<'_, Arc<AppState>>) -> Result<(), String> {
    state.training.force_stop().map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_training_status(
    state: State<'_, Arc<AppState>>,
) -> Result<TrainingSnapshot, String> {
    Ok(state.training.status())
}

/// Clear the finished run's DISPLAY state (snapshot + loss history) back to idle.
/// Files are untouched — the workspace/checkpoints stay resumable. S41: the
/// audition cache dir IS removed (清空结果 = giving up this run's archive entry
/// points, user decision 52588f8) — and the workspace path must be read from
/// the snapshot BEFORE reset clears it (red-team F19/R10).
#[tauri::command]
pub async fn reset_training_display(state: State<'_, Arc<AppState>>) -> Result<(), String> {
    // held for the whole clear (S41-INT-4 — same rationale as start_training)
    let _audition_lock =
        crate::commands::audition::FlightGuard::acquire(crate::commands::audition::BUSY_RETRY_MSG)?;
    let workspace = state.training.status().workspace;
    if !workspace.is_empty() {
        let audition_dir = std::path::Path::new(&workspace).join("audition");
        if audition_dir.exists() {
            state.inference.engine.unload_paths_with_prefix(&audition_dir);
            if let Err(e) = std::fs::remove_dir_all(&audition_dir) {
                tracing::warn!("audition dir cleanup failed (non-fatal): {}", e);
            }
        }
    }
    state.training.reset_display().map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_training_history(
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<StepPoint>, String> {
    Ok(state.training.history())
}

/// Whether a training SLOT for this (name, backend) exists (checkpoints the registry doesn't
/// know about yet) — the retrain-wipes-everything confirm must fire for these too, not only
/// for imported models.
///
/// S76: takes the backend because identity is now「项目 → 架构槽」; one project can hold four
/// slots, and answering for the wrong one would offer 续训 where there is nothing to resume.
#[tauri::command]
pub async fn check_training_workspace(
    state: State<'_, Arc<AppState>>,
    name: String,
    backend: String,
) -> Result<bool, String> {
    let ws = crate::training::slot_path(&data_root(&state), &name, &backend);
    Ok(ws.join("config.json").exists() || ws.join("weights").exists())
}

/// Every destructive training-archive action passes through here first.
///
/// One gate, three consumers — the S74b discipline. `ensure_idle_for_package_delete` already
/// enumerates every in-process holder (convert / training / separation / voice render /
/// audition); copying `TRAINING_ACTIVE + FlightGuard` from the old workspace delete would have
/// missed `convert`, and an `import_model` converting a multi-GB `.pth` out of the very slot
/// being deleted holds only that slot.
///
/// The other two are cross-process and cannot be expressed as locks at all: a sibling instance
/// (double-launch is supported here) may be training out of this tree, and the data-dir reclaim
/// thread copies files back by relpath — it would resurrect exactly what was just deleted.
fn ensure_safe_to_delete(state: &AppState) -> Result<(), String> {
    crate::commands::window::ensure_idle_for_package_delete(state)?;
    if crate::crashlog::other_instance_alive() {
        return Err("DELETE_OTHER_INSTANCE".into());
    }
    if crate::training::tproject::RECLAIM_TOUCHING_TRAINING.load(std::sync::atomic::Ordering::SeqCst) {
        return Err("DELETE_RECLAIM_IN_PROGRESS".into());
    }
    Ok(())
}

/// Drop every ORT session and reload-spec rooted under a path we are about to delete. The
/// prefix is derived HERE from the data root — never taken from the frontend — because the
/// match is a raw path prefix and a `\?\`-prefixed or differently-cased string would miss,
/// leaving a Windows handle to fail the delete and a stale spec to reload afterwards.
fn unload_under(state: &AppState, p: &std::path::Path) {
    state.inference.engine.unload_paths_with_prefix(p);
}

/// Remove the periodic snapshots under `weights/` that nothing needs any more.
///
/// `family` mirrors list_project_ckpts so the button acts on exactly the list the user is
/// looking at. Returns a full account — what went, what stayed and why — because「已释放 0 B」
/// is the CORRECT outcome for a migrated project and would otherwise read as a broken button.
#[tauri::command]
pub async fn training_cleanup_snapshots(
    state: State<'_, Arc<AppState>>,
    project_id: String,
    family: Option<String>,
) -> Result<crate::training::tproject::DeleteReport, String> {
    ensure_safe_to_delete(&state)?;
    let data_dir = data_root(&state);
    // A snapshot whose stem is an installed model must survive even when the ledger has no row
    // for it (imports predating S76, a torn ledger write) — enumerate every holder, S61.
    let installed: std::collections::HashSet<String> =
        state.models.list().into_iter().map(|m| m.name).collect();
    unload_under(&state, &crate::training::tproject::project_dir(&data_dir, &project_id));
    tauri::async_runtime::spawn_blocking(move || {
        let stems = installed;
        crate::training::tproject::cleanup_snapshots(
            &data_dir,
            &project_id,
            family.as_deref(),
            &|stem: &str| stems.iter().any(|n| n == stem || stem.starts_with(n.as_str())),
        )
        .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("TRAINING_DELETE_JOIN: {e}"))?
}

/// Delete ONE architecture slot. The project's shared dataset and its sibling slots stay.
#[tauri::command]
pub async fn training_delete_slot(
    state: State<'_, Arc<AppState>>,
    project_id: String,
    family: String,
) -> Result<crate::training::tproject::DeleteReport, String> {
    ensure_safe_to_delete(&state)?;
    let data_dir = data_root(&state);
    unload_under(&state, &crate::training::tproject::family_dir(&data_dir, &project_id, &family));
    tauri::async_runtime::spawn_blocking(move || {
        crate::training::tproject::delete_slot(&data_dir, &project_id, &family)
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("TRAINING_DELETE_JOIN: {e}"))?
}

/// Delete a whole training project, including its shared dataset. Models already exported into
/// the registry are independent copies and are NOT affected.
#[tauri::command]
pub async fn training_delete_project(
    state: State<'_, Arc<AppState>>,
    project_id: String,
) -> Result<crate::training::tproject::DeleteReport, String> {
    ensure_safe_to_delete(&state)?;
    let data_dir = data_root(&state);
    unload_under(&state, &crate::training::tproject::project_dir(&data_dir, &project_id));
    tauri::async_runtime::spawn_blocking(move || {
        crate::training::tproject::delete_project(&data_dir, &project_id).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("TRAINING_DELETE_JOIN: {e}"))?
}

/// Which training project a model name resolves to, WITHOUT creating one. The archive list
/// must work while idle — that is its whole point (an app restart or「清空结果」leaves the
/// snapshot empty while the files are very much still on disk) — and `snapshot.project_id`
/// describes THIS RUN, so it is empty exactly then.
#[tauri::command]
pub async fn find_training_project(
    state: State<'_, Arc<AppState>>,
    name: String,
) -> Result<Option<String>, String> {
    Ok(crate::training::tproject::find_by_name(&data_root(&state), &name).map(|m| m.id))
}

/// Every checkpoint this project holds on DISK, newest first — the answer to「关掉 app 或点过
/// 『清空结果』之后还剩什么」. Until S76 the candidate list was emitted by the sidecar into
/// memory and nothing ever scanned the disk, so those files kept existing with no way left to
/// reach them. Also the data source for batch 3's snapshot cleanup and batch 5's resume point.
/// Only stats files (never opens them) and runs off-thread — a `weights/` with dozens of
/// multi-GB snapshots must not stall the UI.
#[tauri::command]
pub async fn list_project_ckpts(
    state: State<'_, Arc<AppState>>,
    project_id: String,
    family: Option<String>,
) -> Result<Vec<crate::training::tproject::CkptRecord>, String> {
    let data_dir = data_root(&state);
    tauri::async_runtime::spawn_blocking(move || {
        crate::training::tproject::scan_project_ckpts(&data_dir, &project_id, family.as_deref())
    })
    .await
    .map_err(|e| format!("TRAINING_SCAN_JOIN: {e}"))
}

/// Note that a checkpoint became an installed model. Feeds the protection set behind
/// 「清理未导入的快照」— without it every snapshot a user kept on purpose would read as
/// unimported and be deleted.
#[tauri::command]
pub async fn record_project_export(
    state: State<'_, Arc<AppState>>,
    project_id: String,
    name: String,
    model_type: String,
    from_ckpt: String,
) -> Result<(), String> {
    crate::training::tproject::record_export(
        &data_root(&state),
        &project_id,
        &name,
        &model_type,
        &from_ckpt,
    )
    .map_err(|e| e.to_string())
}

/// Structured workspace facts (S39): the main-model retrain dialog must warn
/// when the wipe would also destroy diffusion training progress, and the
/// 浅扩散 card phrases its own dialog by resume-vs-cache-reuse.
#[tauri::command]
pub async fn get_training_workspace_info(
    state: State<'_, Arc<AppState>>,
    name: String,
    backend: String,
) -> Result<crate::training::WorkspaceInfo, String> {
    Ok(crate::training::workspace_info(&data_root(&state), &name, &backend))
}
