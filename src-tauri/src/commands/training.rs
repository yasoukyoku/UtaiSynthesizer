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
    // ★ The SAME directory `try_start` will train into — resolved from `project_id`, exactly as
    // it resolves it. This used to go through `slot_path(model_name, …)`, which was right only
    // while the model name WAS the directory identity. Batch 4 made it「本次训练名」: editable,
    // free to differ from the project name, and frozen per slot. A miss here is silent and
    // expensive: the unload below would release no session (so a `fresh` wipe meets live
    // Windows file handles), the cleanup below would remove nothing (so the PREVIOUS run's
    // audition renders survive under identical `weights/<slug>_best` names and the next
    // 「试听」plays the old voice), and a run name that happens to match ANOTHER project's name
    // would resolve there and delete that project's audition cache instead.
    //
    // An empty `project_id` is still the documented legacy shape (resolve by name), so it keeps
    // the old derivation — which is correct for exactly that case.
    let audition_dir = if request.project_id.trim().is_empty() {
        crate::training::slot_path(&data_dir, &request.model_name, &request.backend)
    } else {
        checked_project_id(&request.project_id)?;
        crate::training::tproject::family_dir(
            &data_dir,
            &request.project_id,
            crate::training::backend_family(&request.backend),
        )
    }
    .join("audition");
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

// `check_training_workspace` lived here until S76 batch 4. It existed as the CRUDE half of a
// deliberate pair: `onStart` asked `get_training_workspace_info` first and fell back here when
// that answer never arrived — the caller seeds `fresh = true` (= wipe) and only narrows it
// inside dialogs that hang off the probe, so a probe that fails with nothing behind it means
// 「没弹任何对话框就删了」. That pairing stopped meaning anything once both commands became
// `checked_project_id` + a path join off the same project id: they now fail and succeed
// together, so the fallback was answering exactly when it could not. The primary probe is
// fail-CLOSED instead, which is what the whole guard chain was after.

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

// `find_training_project(name)` lived here from batch 2 until batch 4. It answered「这个模型名
// 属于哪个项目」for the archive list, which had no other way to identify a project while the
// snapshot was idle. Batch 4 gave the page an explicit `route.projectId`, and its last caller —
// the shallow-diffusion card's cross-project host picker — was removed with that picker (it
// rewrote the host slot's frozen run name). Name→project resolution still exists Rust-side
// (`tproject::find_by_name`) for `slot_path` and `resolve_or_create`; what is gone is the
// ability to ask it from the UI, which is the right direction: names are user-editable now.

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

// `record_project_export` lived here until S76 batch 4. Batch 3 pushed the bookkeeping down
// into `import_model` / `attach_diffusion` (`commands::models::record_training_export`) so that
// the resource manager's file picker — which can browse straight into a training slot's
// `weights/` — would be covered too. This command then survived as a SECOND writer that the
// training page still called, and the two disagreed: Rust recorded the registry type, the page
// recorded `snapshot.backend`, and the later write won, so a shallow-diffusion attach ended up
// filed under `sovits_diff` — a value the registry's own type table does not contain. The page
// now only re-reads the list; `import_model` takes a `source_ckpt` for the one case that made
// the frontend write look necessary (importing the audition cache's converted copy).

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

/// One file of the project's shared dataset, as the UI lists it.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DatasetFileRow {
    /// Path under `dataset/`, forward-slashed (`000.wav` / `<slug>/000.wav`).
    pub rel: String,
    /// The name the file had when it was imported. Empty = unrecorded (imported before batch 5,
    /// or the annotation was lost) — the UI shows `rel` instead of inventing one.
    pub name: String,
    pub bytes: u64,
    pub duration_ms: Option<f64>,
}

/// One co-trained speaker of the project's dataset. The POSITION is the emb_g row id whenever
/// [`DatasetSummary::order_known`] is true.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DatasetGroupRow {
    pub slug: String,
    /// Display name, recovered from `dataset.json` or a slot manifest. Empty = unrecoverable
    /// (`slugify` is one-way), and the UI must then show the slug rather than guess.
    pub name: String,
    pub files: u32,
    pub bytes: u64,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DatasetSummary {
    pub files: u32,
    pub bytes: u64,
    /// Absolute path of `<project>/dataset`. The UI joins it with a row's `rel` to preview the
    /// audio — the frontend must never rebuild this from the data root, which it does not know
    /// (and which the user can move).
    pub dataset_dir: String,
    /// Per-speaker subdirectory names (multi-singer projects). Empty = flat, single speaker.
    /// SORTED — kept as-is because `poolFlat` keys on its emptiness; the ordered, named view is
    /// `groups`.
    pub speakers: Vec<String>,
    /// Every file on disk, sorted by `rel`. This is what makes「时间长了忘了当初导入的是什么」
    /// answerable at all: the copies are named positionally, so without the annotation layer the
    /// list would read `000.wav, 001.wav, …`.
    pub entries: Vec<DatasetFileRow>,
    /// Speakers in emb_g order when it is knowable, alphabetical otherwise.
    pub groups: Vec<DatasetGroupRow>,
    /// Is `groups`' order the real emb_g order? False for a multi-speaker dataset that has never
    /// been trained and predates the annotation — the UI must NOT print row numbers then, since
    /// reproducing a wrong order is what mis-assigns every singer's timbre on a rebuild.
    pub order_known: bool,
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

/// Guard for every command that WRITES `<project>/dataset/`.
///
/// Same three conditions as `ensure_safe_to_delete` — a run slicing the dataset while files
/// appear or vanish under it, a second instance doing the same, or the reclaim thread copying
/// into the tree — with dataset-shaped CODEs so the text can say what was actually refused.
fn ensure_safe_dataset_write(state: &AppState) -> Result<(), String> {
    crate::commands::window::ensure_idle_for_dataset_write(state)?;
    if crate::crashlog::other_instance_alive() {
        return Err("DATASET_OTHER_INSTANCE".into());
    }
    if crate::training::tproject::RECLAIM_TOUCHING_TRAINING.load(std::sync::atomic::Ordering::SeqCst)
    {
        return Err("DATASET_RECLAIM_IN_PROGRESS".into());
    }
    Ok(())
}

/// Every architecture slot that froze a speaker order. ONE source for the three consumers
/// (`get_training_project`, and both dataset writers): they must see the SAME view of who the
/// project's singers are, or a name the UI got from one will not resolve in the other — which
/// is exactly how「给已有歌手加文件」came out as「新增歌手」and hit the frozen-structure guard.
fn frozen_lists(
    data_dir: &std::path::Path,
    project_id: &str,
) -> Vec<Vec<crate::training::dsmanifest::DsSpeaker>> {
    crate::training::tproject::FAMILIES
        .iter()
        .map(|f| crate::training::frozen_speakers(data_dir, project_id, f))
        .filter(|v| !v.is_empty())
        .collect()
}

/// The first architecture slot that has FROZEN a speaker set, if any.
///
/// While one exists the speaker structure is immutable: `n_speakers` and the ordered slug list
/// are baked into that slot's emb_g rows, and changing either makes it unresumable
/// (`RESUME_SPEAKER_COUNT_MISMATCH` / `RESUME_SPEAKER_SET_MISMATCH`). Adding or removing FILES
/// stays allowed — it only costs a re-extraction, which the UI says out loud.
fn frozen_structure_family(data_dir: &std::path::Path, project_id: &str) -> Option<String> {
    crate::training::tproject::FAMILIES
        .iter()
        .find(|f| !crate::training::frozen_speakers(data_dir, project_id, f).is_empty())
        .map(|f| f.to_string())
}

/// Everything an EXPORT needs to know about a slot, resolved from disk instead of from a live
/// run's snapshot.
///
/// The run summary derives all three from `TrainingSnapshot` — which exists only while (or just
/// after) a run is displayed. That is why a finished shallow-diffusion checkpoint became
/// unreachable the moment anything else was trained: the summary that carried its attach button
/// was replaced, and nothing else could name the artifacts. Reading them off the slot makes the
/// project's archive actionable at any time, including after a restart.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SlotExportContext {
    /// The「本次训练名」this slot's artifacts carry — the default name an import suggests.
    /// Empty when the slot never completed a run (no `run.json`), in which case the caller
    /// falls back to the project name.
    pub model_name: String,
    /// Absolute path of the family slot (the old「workspace」).
    pub workspace: String,
    /// The retrieval/cluster companion an import should carry, if one exists on disk. Same
    /// probe order the run summary uses as its no-summary fallback: RVC keeps its historical
    /// `total_fea.npy`, SoVITS looks for the cluster assets (built BEFORE training, so they
    /// exist even for an early stop). Vocoders have none — probing would only find another
    /// backend's leftovers.
    pub index_path: Option<String>,
}

#[tauri::command]
pub async fn get_slot_export_context(
    state: State<'_, Arc<AppState>>,
    project_id: String,
    backend: String,
) -> Result<SlotExportContext, String> {
    checked_project_id(&project_id)?;
    let data_dir = data_root(&state);
    let family = crate::training::backend_family(&backend);
    let ws = crate::training::tproject::family_dir(&data_dir, &project_id, family);
    let index_path = if backend == "vocoder" {
        None
    } else if backend == "rvc" {
        let p = ws.join("total_fea.npy");
        p.is_file().then(|| p.to_string_lossy().into_owned())
    } else {
        ["cluster/kmeans_10000.pt", "cluster/0.index_vectors.npy"]
            .iter()
            .map(|rel| ws.join(rel))
            .find(|p| p.is_file())
            .map(|p| p.to_string_lossy().into_owned())
    };
    Ok(SlotExportContext {
        model_name: crate::training::tproject::slot_model_name(&data_dir, &project_id, family)
            .unwrap_or_default(),
        workspace: ws.to_string_lossy().into_owned(),
        index_path,
    })
}

/// Import audio INTO the project's shared dataset, independent of any training run.
///
/// Appends — the run-time import replaces wholesale, this one adds. `speaker` is a display name
/// (a co-trained singer); `None` targets the flat dataset. Mixing the two shapes is refused:
/// python's fingerprint hard-fails on a subdirectory for a flat backend, and a stray flat file in
/// a multi-singer dataset belongs to no emb_g row.
#[tauri::command]
pub async fn import_project_dataset(
    state: State<'_, Arc<AppState>>,
    project_id: String,
    files: Vec<String>,
    speaker: Option<String>,
) -> Result<(), String> {
    checked_project_id(&project_id)?;
    ensure_safe_dataset_write(&state)?;
    let data_dir = data_root(&state);
    if crate::training::tproject::read_meta(&data_dir, &project_id).is_none() {
        return Err("PROJECT_META_UNREADABLE".into());
    }
    if files.is_empty() {
        return Err("TRAINING_NO_DATA".into());
    }
    for f in &files {
        if !std::path::Path::new(f).is_file() {
            return Err(format!("TRAINING_DATA_FILE_MISSING: {f}"));
        }
    }
    let facts = crate::training::dsmanifest::read_facts(
        &data_dir,
        &project_id,
        &frozen_lists(&data_dir, &project_id),
    );
    let has_flat = facts.entries.iter().any(|e| !e.rel.contains('/'));
    let name = speaker.as_deref().map(str::trim).filter(|s| !s.is_empty());
    let slug = match name {
        Some(n) => {
            if has_flat {
                return Err("PROJECT_DATASET_SHAPE".into());
            }
            match crate::training::dsmanifest::find_group(&facts, n) {
                // an existing singer: the slug is already frozen on disk, never re-derive it
                Some(g) => Some(g.speaker.slug.clone()),
                None => {
                    // a NEW singer changes the speaker set — refuse while any slot's emb_g rows
                    // depend on it
                    if let Some(fam) = frozen_structure_family(&data_dir, &project_id) {
                        // The family id goes to the LOG, not into the message: the text already
                        // explains what to do, and a bare「(rvc)」tacked onto the end of a
                        // paragraph reads as noise (the error funnel appends any payload
                        // verbatim in parentheses — it is for names and paths, not internal ids).
                        tracing::info!("refusing new speaker in {project_id}: {fam} froze the set");
                        return Err("DATASET_SPEAKERS_FROZEN".into());
                    }
                    let base = crate::training::slugify(n);
                    let mut s = base.clone();
                    let mut k = 2;
                    while facts.speaker_slugs.iter().any(|e| *e == s) {
                        s = format!("{base}_{k}");
                        k += 1;
                    }
                    Some(s)
                }
            }
        }
        None => {
            if !facts.speaker_slugs.is_empty() {
                return Err("PROJECT_DATASET_SHAPE".into());
            }
            None
        }
    };
    crate::training::dsmanifest::append_files(
        &data_dir,
        &project_id,
        slug.as_deref(),
        name,
        &files,
        &|p| crate::audio::probe_duration_ms(p).ok(),
    )
    .map_err(|e| e.to_string())
}

/// Remove files from the project's shared dataset. `rels` are `DatasetFileRow.rel` values.
///
/// Emptying a singer entirely changes the speaker SET, so it is refused while any slot has frozen
/// one; removing files from a singer that keeps at least one is fine. Numbering is NOT compacted
/// afterwards — see the mutation rules in `dsmanifest`.
#[tauri::command]
pub async fn delete_project_dataset_files(
    state: State<'_, Arc<AppState>>,
    project_id: String,
    rels: Vec<String>,
) -> Result<(), String> {
    checked_project_id(&project_id)?;
    ensure_safe_dataset_write(&state)?;
    let data_dir = data_root(&state);
    if rels.is_empty() {
        return Ok(());
    }
    let frozen = frozen_structure_family(&data_dir, &project_id);
    let facts = crate::training::dsmanifest::read_facts(
        &data_dir,
        &project_id,
        &frozen_lists(&data_dir, &project_id),
    );
    let plan = crate::training::dsmanifest::plan_delete(&facts, &rels);
    if !plan.emptied_speakers.is_empty() {
        if let Some(fam) = frozen.as_deref() {
            tracing::info!("refusing to empty a speaker in {project_id}: {fam} froze the set");
            return Err("DATASET_SPEAKERS_FROZEN".into());
        }
    }
    crate::training::dsmanifest::delete_files(&data_dir, &project_id, &rels, frozen.is_none())
        .map_err(|e| e.to_string())
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
    // Every architecture slot that froze a speaker order — each is a per-SLOT truth, so they are
    // all handed over and `resolve_speakers` decides what the project-level answer may claim.
    let facts = crate::training::dsmanifest::read_facts(
        &data_dir,
        &project_id,
        &frozen_lists(&data_dir, &project_id),
    );
    let entries: Vec<DatasetFileRow> = facts
        .entries
        .iter()
        .map(|e| DatasetFileRow {
            rel: e.rel.clone(),
            name: e.name.clone(),
            bytes: e.bytes,
            duration_ms: e.duration_ms,
        })
        .collect();
    let groups: Vec<DatasetGroupRow> = facts
        .groups
        .iter()
        .map(|g| DatasetGroupRow {
            slug: g.speaker.slug.clone(),
            name: g.speaker.name.clone(),
            files: g.files,
            bytes: g.bytes,
        })
        .collect();

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
            files: facts.files,
            bytes: crate::commands::storage::dir_size(&dataset_dir),
            dataset_dir: dataset_dir.to_string_lossy().into_owned(),
            speakers: facts.speaker_slugs,
            entries,
            groups,
            order_known: facts.order_known,
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

// `get_training_workspace_info(name, backend)` lived here until S76 batch 4 — see the note in
// `training::mod` where its implementation was. `get_training_slot_info` is the same answer
// keyed by the project id, which is the only identity that survives a rename.
