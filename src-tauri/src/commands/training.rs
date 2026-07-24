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
/// THE trust boundary for a frontend-supplied project id.
///
/// Every id this app mints — `new_project_id` (sha2) and the legacy `slugify` — is
/// `[A-Za-z0-9_-]+`, so refusing anything else costs nothing legitimate. What it buys is that
/// `training_root.join(id)` can never leave the training root: `training_delete_project("../..")`
/// would otherwise `dir_size` and RENAME the data directory's parent into a tombstone.
///
/// Not reachable from today's UI (ids come from the backend's own listings), which is exactly
/// why it must be asserted rather than assumed. The pre-S76 `delete_training_workspace` had
/// this check (`storage.rs`); its S76 replacements were written without it.
fn checked_project_id(id: &str) -> Result<(), String> {
    if id.is_empty() || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
        return Err("PROJECT_ID_INVALID".into());
    }
    Ok(())
}

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
    checked_project_id(&project_id)?;
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
    checked_project_id(&project_id)?;
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
    checked_project_id(&project_id)?;
    ensure_safe_to_delete(&state)?;
    let data_dir = data_root(&state);
    let app_dir = state.app_dir.clone();
    unload_under(&state, &crate::training::tproject::project_dir(&data_dir, &project_id));
    tauri::async_runtime::spawn_blocking(move || {
        let report = crate::training::tproject::delete_project(&data_dir, &project_id)
            .map_err(|e| e.to_string())?;
        // Drop the listing cache's row too, or a project the user just deleted on purpose comes
        // straight back as a MISSING ghost. Only after the delete SUCCEEDED — a refused delete
        // must leave every trace of the project exactly as it was.
        crate::training::tproject::forget_project(&app_dir, &data_dir, &project_id);
        Ok(report)
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
    checked_project_id(&project_id)?;
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
    checked_project_id(&project_id)?;
    crate::training::tproject::record_export(
        &data_root(&state),
        &project_id,
        &name,
        &model_type,
        &from_ckpt,
    )
    .map_err(|e| e.to_string())
}

// ───────────────────────── project pages (S76 batch 4) ─────────────────────────

/// Every training project, for the landing page.
///
/// `refresh` = walk the disk for per-project sizes (seconds over tens of GB on a real machine)
/// and update the cache; `false` answers from the cache instantly. The page paints from the
/// cache and asks for one refresh afterwards, so opening it is never a stall.
#[tauri::command]
pub async fn list_training_projects(
    state: State<'_, Arc<AppState>>,
    refresh: bool,
) -> Result<Vec<crate::training::tproject::ProjectSummary>, String> {
    let data_dir = data_root(&state);
    let app_dir = state.app_dir.clone();
    tauri::async_runtime::spawn_blocking(move || {
        crate::training::tproject::list_project_summaries(&app_dir, &data_dir, refresh)
    })
    .await
    .map_err(|e| format!("TRAINING_SCAN_JOIN: {e}"))
}

/// Create a project explicitly. Returns its id — the identity every later call uses, and the
/// one thing a display name is NOT (names are editable and must stay unique only for the
/// legacy name-keyed callers).
#[tauri::command]
pub async fn create_training_project(
    state: State<'_, Arc<AppState>>,
    name: String,
    note: String,
) -> Result<String, String> {
    crate::training::tproject::create_project(&data_root(&state), &name, &note)
        .map(|m| m.id)
        .map_err(|e| e.to_string())
}

/// Rename / re-annotate. The directory never moves and no artifact is renamed — see
/// `tproject::update_project`.
#[tauri::command]
pub async fn update_training_project(
    state: State<'_, Arc<AppState>>,
    project_id: String,
    name: String,
    note: String,
) -> Result<(), String> {
    checked_project_id(&project_id)?;
    crate::training::tproject::update_project(&data_root(&state), &project_id, &name, &note)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

/// Drop a MISSING project's cache row (「移除记录」). Touches nothing on disk — by definition
/// there is nothing there — so it needs none of the delete guards.
#[tauri::command]
pub async fn forget_training_project(
    state: State<'_, Arc<AppState>>,
    project_id: String,
) -> Result<(), String> {
    checked_project_id(&project_id)?;
    crate::training::tproject::forget_project(&state.app_dir, &data_root(&state), &project_id);
    Ok(())
}

/// One architecture slot of a project, as the detail page shows it.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SlotDetail {
    pub family: String,
    /// The「本次训练名」this slot's artifacts were built under (`weights/<slug>*`, `hps.name`).
    /// Empty = this slot never completed a run, so the next one may choose freely.
    pub model_name: String,
    pub info: crate::training::WorkspaceInfo,
    pub bytes: u64,
    /// Newest checkpoint training can CONTINUE from. `None` with `has_resume_point` still true
    /// is RVC's「只保留最新」sentinel, whose file name carries no step.
    pub resume_step: Option<u64>,
    pub has_resume_point: bool,
    /// Everything under this slot that the archive list would show, and what it weighs.
    pub ckpt_count: u32,
    pub ckpt_bytes: u64,
}

/// A ledger row plus the answer to「它现在还在不在」.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportedModelStatus {
    pub name: String,
    pub model_type: String,
    pub from_ckpt_rel: String,
    pub at_ms: u64,
    /// LIVE registry check. False = the user deleted it in the resource manager since; the row
    /// stays visible and greyed rather than vanishing, because「导出过」is history, not state.
    pub installed: bool,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DatasetSummary {
    pub files: u32,
    pub bytes: u64,
    /// Per-speaker subdirectory names (multi-singer projects). Empty = flat, single speaker.
    pub speakers: Vec<String>,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectDetail {
    pub id: String,
    pub name: String,
    pub note: String,
    pub created_ms: u64,
    pub updated_ms: u64,
    pub needs_attention: Option<String>,
    pub dataset: DatasetSummary,
    pub slots: Vec<SlotDetail>,
    pub exported: Vec<ExportedModelStatus>,
}

/// The ledger's `model_type` vocabulary. It is a SUPERSET of what `parse_voice_type` accepts,
/// because until batch 4 the export was recorded twice — once by Rust (`import_model` /
/// `attach_diffusion`, which write a REGISTRY type) and once by the training page (which wrote
/// `snapshot.backend`, so a shallow-diffusion attach landed as `sovits_diff`). Those rows are
/// on users' disks already. Everything else defers to the single source in `commands::models`.
fn ledger_model_type(s: &str) -> Option<crate::models::ModelType> {
    if s == "sovits_diff" {
        return Some(crate::models::ModelType::SoVits);
    }
    crate::commands::models::parse_voice_type(s)
}

#[tauri::command]
pub async fn get_training_project(
    state: State<'_, Arc<AppState>>,
    project_id: String,
) -> Result<ProjectDetail, String> {
    checked_project_id(&project_id)?;
    let data_dir = data_root(&state);
    let Some(meta) = crate::training::tproject::read_meta(&data_dir, &project_id) else {
        return Err("PROJECT_META_UNREADABLE".into());
    };
    // LIVE cross-check: `list`/`exists` read an in-memory cache that is only refreshed
    // explicitly, so without this a model deleted through the resource manager would still
    // report「已安装」here. Typed lookup, never `get` — one singer legitimately owns an rvc, a
    // sovits AND a same-named vocoder, and an untyped first-match would answer for whichever
    // the scan order happened to reach first.
    state.models.scan().map_err(|e| e.to_string())?;
    let exported = meta
        .exported
        .iter()
        .map(|e| ExportedModelStatus {
            name: e.name.clone(),
            model_type: e.model_type.clone(),
            from_ckpt_rel: e.from_ckpt_rel.clone(),
            at_ms: e.at_ms,
            installed: ledger_model_type(&e.model_type)
                .map(|mt| state.models.exists(&e.name, &mt))
                .unwrap_or(false),
        })
        .collect();

    let dataset_dir = crate::training::tproject::dataset_dir(&data_dir, &project_id);
    let mut files = 0u32;
    let mut speakers: Vec<String> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&dataset_dir) {
        for e in rd.flatten() {
            if e.path().is_dir() {
                speakers.push(e.file_name().to_string_lossy().into_owned());
            } else {
                files += 1;
            }
        }
    }
    // A multi-singer dataset keeps its audio one level down, so the flat count would read 0.
    for s in &speakers {
        files += std::fs::read_dir(dataset_dir.join(s))
            .map(|rd| rd.flatten().filter(|e| e.path().is_file()).count() as u32)
            .unwrap_or(0);
    }
    speakers.sort();

    let slots = crate::training::tproject::FAMILIES
        .iter()
        .filter(|f| crate::training::tproject::family_dir(&data_dir, &project_id, f).is_dir())
        .map(|f| {
            let recs = crate::training::tproject::scan_project_ckpts(&data_dir, &project_id, Some(f));
            // `scan_project_ckpts` returns newest-first by mtime — the same ordering upstream
            // itself resumes by (the RVC sentinel makes step numbers unorderable).
            let newest_resumable = recs
                .iter()
                .find(|r| matches!(r.kind, crate::training::tproject::CkptKind::Resumable));
            SlotDetail {
                family: f.to_string(),
                model_name: crate::training::tproject::slot_model_name(&data_dir, &project_id, f)
                    .unwrap_or_default(),
                info: crate::training::slot_info(&data_dir, &project_id, f),
                bytes: crate::commands::storage::dir_size(&crate::training::tproject::family_dir(
                    &data_dir,
                    &project_id,
                    f,
                )),
                resume_step: newest_resumable.and_then(|r| r.step),
                has_resume_point: newest_resumable.is_some(),
                ckpt_count: recs.len() as u32,
                ckpt_bytes: recs.iter().map(|r| r.bytes).sum(),
            }
        })
        .collect();

    Ok(ProjectDetail {
        id: meta.id,
        name: meta.name,
        note: meta.note,
        created_ms: meta.created_ms,
        updated_ms: meta.updated_ms,
        needs_attention: meta.needs_attention,
        dataset: DatasetSummary {
            files,
            bytes: crate::commands::storage::dir_size(&dataset_dir),
            speakers,
        },
        slots,
        exported,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_ids_that_could_escape_the_training_root_are_refused() {
        // every id this app mints passes
        assert!(checked_project_id("sayo_84dba34a").is_ok());
        assert!(checked_project_id(&crate::training::tproject::new_project_id("歌姫テスト")).is_ok());
        assert!(checked_project_id(&crate::training::slugify("歌姫テスト")).is_ok());
        assert!(checked_project_id("a-b_C9").is_ok());
        // anything path-like does not — `training_root.join(id)` must stay inside the root
        for bad in ["", "..", "../..", "a/b", "a\\b", ".del_x", "C:", "a:b", "a b", "项目"] {
            assert!(checked_project_id(bad).is_err(), "must refuse {bad:?}");
        }
    }

    #[test]
    fn the_ledgers_model_types_all_resolve() {
        use crate::models::ModelType;
        // what `import_model` writes …
        assert!(matches!(ledger_model_type("rvc"), Some(ModelType::Rvc)));
        assert!(matches!(ledger_model_type("sovits"), Some(ModelType::SoVits)));
        assert!(matches!(ledger_model_type("sovits_v2"), Some(ModelType::SoVits)));
        assert!(matches!(ledger_model_type("vocoder"), Some(ModelType::NsfHifigan)));
        // … plus the one value only PRE-batch-4 ledgers hold (the training page used to record
        // the export a second time, with `snapshot.backend`). Without it a shallow-diffusion row
        // would resolve to nothing and report「已删除」about a model that is installed.
        assert!(matches!(ledger_model_type("sovits_diff"), Some(ModelType::SoVits)));
        assert!(ledger_model_type("nonsense").is_none());
    }
}

/// Slot facts keyed by PROJECT ID — the rename-proof twin of `get_training_workspace_info`.
#[tauri::command]
pub async fn get_training_slot_info(
    state: State<'_, Arc<AppState>>,
    project_id: String,
    backend: String,
) -> Result<crate::training::WorkspaceInfo, String> {
    checked_project_id(&project_id)?;
    Ok(crate::training::slot_info(&data_root(&state), &project_id, &backend))
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
