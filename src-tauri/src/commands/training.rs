use std::path::{Path, PathBuf};
use std::sync::Arc;
use tauri::State;

use crate::training::{StartTrainingRequest, StepPoint, TrainingSnapshot};
use crate::AppState;

/// The run a start request addresses, in the shape `trun` takes.
///
/// ★§F2⒝ batch 2 step ④ — `""` and `None` mean the SAME thing here ("the slot holds at most one
/// run"), and they have to: the field is `#[serde(default)]`, so an older frontend and every
/// non-run-aware caller send the empty string, and mapping that to `Some("")` would ask
/// `resolve_run_dir` for a run literally named `""` and get `RUN_ID_INVALID` on every start.
fn run_id_of(req: &StartTrainingRequest) -> Option<&str> {
    opt_run_id(&req.run_id)
}

/// Same normalization for the commands that take `run_id: Option<String>` over IPC.
///
/// ⛔ `Some("")` must never reach `trun::resolve_run_dir`: it would look for a run literally named
/// `""`, fail `run_id_is_usable`, and answer `RUN_ID_INVALID` — turning "this slot has one run"
/// (which every caller means by an empty string) into a hard error on every call.
fn opt_run_id(s: &str) -> Option<&str> {
    Some(s.trim()).filter(|s| !s.is_empty())
}

pub(crate) fn data_root(state: &AppState) -> PathBuf {
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
    let slot = if request.project_id.trim().is_empty() {
        crate::training::slot_path(&data_dir, &request.model_name, &request.backend)
    } else {
        checked_project_id(&request.project_id)?;
        crate::training::tproject::family_dir(
            &data_dir,
            &request.project_id,
            crate::training::backend_family(&request.backend),
        )
    };
    // ★§F2⒝ batch 2 — the cache is a RUN product, so the cleanup below has to name the run.
    // ⛔ step ④: the run the REQUEST names. Resolving `None` here once a slot holds several runs
    // is not a wrong number, it is `remove_dir_all` of another run's converted .onnx — and of the
    // measured vocal range stored beside it in `model.json`, which nothing re-measures.
    let audition_dir = crate::training::trun::resolve_run_dir(&slot, run_id_of(&request))
        .map_err(|e| e.to_string())?
        .join("audition");
    // read before `request` is moved into `start`; see the cleanup below for why it is `fresh`
    let request_was_fresh = request.fresh;
    // BEFORE manager.start(): drop every audition session (file locks) so the
    // fresh-wipe path inside try_start cannot trip over them. Non-destructive —
    // an evicted session reloads on miss.
    // ⚠ Prefixed on the SLOT, not on the run: what `fresh` erases is the whole slot, so a
    // session held open under ANY of its runs is a live Windows file handle in the way. Scoping
    // this to one run would leave the others locked and turn the wipe into a hard failure.
    state.inference.engine.unload_paths_with_prefix(&slot);
    state
        .training
        .start(app, data_dir, request)
        .map_err(|e| e.to_string())?;
    // AFTER a successful launch (never on guard-rejected starts, red-team R10 —
    // a rejected start must not cost the user their audition cache): the new
    // run's candidate list supersedes the old one.
    //
    // ⛔★★§F2⒝ ④e — …but ONLY for a start that trains into this same run. `audition_dir` is
    // resolved from `request.run_id`, and 「再训一个」 sends the id of the run the user pressed it
    // on (correctly — every guard in `try_start` has to judge THAT run). Before ④e that run was
    // about to be erased anyway, so deleting its cache was free. Now the training lands in a
    // newly minted run and this line would delete the cache of the run ④e exists to KEEP —
    // and what is in there is a converted .onnx plus the measured vocal range in `model.json`,
    // which nothing re-measures.
    //
    // ⚠ The condition is `!fresh`, not「不是铸新」: the command cannot compute `mints_fresh_run`
    // (it does not know `diff_partial_wipe`) and re-deriving it here would be a second copy of a
    // rule that already exists in one place. The two answers differ only for 「重训(仅扩散)」.
    //
    // ⛔★S133 — that difference was NOT free, and the original wording ("costs a stale candidate
    // list") under-read it by an order of magnitude: the diffusion products are FIXED-NAME
    // (`model_best.pt`), the cache key is the checkpoint's stem, and a hit is decided by
    // `model.json` existing and nothing else ⇒ after a diffusion retrain 「试听」 replayed the
    // PREVIOUS run's converted graph and wav with nothing on screen to tell them apart.
    // It is now handled where it belongs — `training::evict_audition_of`, called right next to the
    // deletion whose staleness it mirrors, per checkpoint rather than per directory (the sibling
    // entries are the MAIN model's, and they hold the only copy of a measured vocal range).
    if !request_was_fresh && audition_dir.exists() {
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
    let app_dir = state.app_dir.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let rel = format!("training/{project_id}/{family}");
        let report = crate::training::tproject::delete_slot(&data_dir, &project_id, &family)
            .map_err(|e| e.to_string())?;
        crate::commands::settings::record_deliberate_delete(&app_dir, &rel);
        Ok(report)
    })
    .await
    .map_err(|e| format!("TRAINING_DELETE_JOIN: {e}"))?
}

/// Delete ONE run of one architecture slot. The slot's preprocessing pools, its sibling runs and
/// the project's shared dataset all stay.
///
/// ★★§F2⒝ 批 2 ④e —— 「重训 = 铸新 run」的另一条腿。三道前置与删槽/删项目**完全一致**,
/// 而它们不是可选的:
/// * `ensure_safe_to_delete` —— ⛔ 别照抄 `rename_training_run` 的闸链,那一条**没有**
///   `RECLAIM_TOUCHING_TRAINING`(改名不搬字节,删除搬);数据根回收线程按相对路径往回拷,
///   它会把刚删掉的东西原样拷回来;
/// * `unload_under` —— ⚠ 前缀取到 **run**,不是槽。`start_training` 那处取槽是因为当年
///   `fresh` 擦整槽(理由已随 flip 死掉);这里只删一个 run,收窄到 run 语义更对而且安全:
///   匹配是 `Path::starts_with`(按**路径分量**比),兄弟 run 的 id 定长同构,谁也不是谁的前缀;
///   漏卸只会让 rename 响亮失败,不会静默毁数据。
/// * ⚠ 粒度:进程级,而且**比「有没有训练在跑」还宽** —— `ensure_idle_for_package_delete` 走
///   `running_tasks_of`,它把 training / separation / render / audition / 全部 `active_tasks`
///   一起算(那个函数的 doc 明写 `Deliberately FAIL-CLOSED and coarse`)。⛔ 这道保险不许放宽。
///
///   ⛔⛔ **这里原本写着「前端可以更精确地禁用按钮(`resolveRowIdentity` 已经能判这一行是不是
///   正在跑的那个 run)」—— 那句话是假的,而 §E2E-M25 差点照着它做**:
///   * `resolveRowIdentity` 的 `LiveRunIdentity` **没有 `state`**,它只比工作区路径是否相等 ⇒
///     一个 `completed` / `stopped` 的 run **照样**答 `source: "live"`(快照只有用户点「清空结果」
///     才清)。拿它当「这一行正在训练」用,做出来的是**跑完之后按钮永久禁死**的界面。
///   * 它还需要 `get_slot_export_context` 的返回值 —— 那是**每一行一次 async invoke**,
///     项目页那面卡片墙付不起。
///
///   ⇒ 前端今天的正确接法(S143 落地):**两个正交谓词**,都在 `lib/training/liveRun.ts` 里。
///   「删除此 run」跟**这道闸同粒度**的全局谓词走(镜像它,而不是比它更细 —— 更细就等于
///   在最常见的那一格〔同槽兄弟 run / 试听在飞〕上仍然让用户白点一次);per-run 身份
///   (`TrainingSnapshot.run_id`,§E2E-M25 笔 0 加的)只用来**解释为什么**。
#[tauri::command]
pub async fn training_delete_run(
    state: State<'_, Arc<AppState>>,
    project_id: String,
    family: String,
    run_id: String,
) -> Result<crate::training::tproject::DeleteReport, String> {
    checked_project_id(&project_id)?;
    ensure_safe_to_delete(&state)?;
    let data_dir = data_root(&state);
    let slot = crate::training::tproject::family_dir(&data_dir, &project_id, &family);
    // Resolve for the UNLOAD only, and tolerate a failure: `delete_run` re-resolves and is the
    // authority on whether this id names anything. A refusal here would turn 「this run is gone
    // already」 into an untranslated error before the real guard ever spoke.
    if let Ok(run) = crate::training::trun::resolve_run_dir(&slot, opt_run_id(run_id.trim())) {
        unload_under(&state, run.path());
    }
    let app_dir = state.app_dir.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let rel = format!(
            "training/{project_id}/{}/{}/{}",
            crate::training::backend_family(&family),
            crate::training::trun::RUNS_DIR,
            run_id.trim()
        );
        let report = crate::training::tproject::delete_run(&data_dir, &project_id, &family, &run_id)
            .map_err(|e| e.to_string())?;
        // ★S133 — only AFTER it succeeded, and only while a data-root reclaim is queued (the
        // helper is a no-op otherwise): the reclaim copies back by relpath, so without this the
        // run the user just deleted comes back at the next boot and the log calls it 「freed」.
        crate::commands::settings::record_deliberate_delete(&app_dir, &rel);
        Ok(report)
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
        crate::commands::settings::record_deliberate_delete(
            &app_dir,
            &format!("training/{project_id}"),
        );
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

/// ONE RUN of one architecture slot, as the detail page's run list shows it.
///
/// ★§F2⒝ batch 2 step ④. Every field here is a RUN fact and was previously read off the slot
/// through a resolver that refuses to answer once there are two runs — which is why splitting the
/// shape had to come BEFORE anything mints a second run, not after: the failure of the old shape
/// is `slot_info` returning `Err`, and `get_training_project` collects the four slots through one
/// `Result`, so a single two-run slot took the whole project page down.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RunDetail {
    /// `trun` run id. Empty string = an UNMIGRATED slot whose root is the run (layout ≤ 2) —
    /// a positive fact, not a missing value, and the same one `trun::resolve_run_dir` encodes by
    /// answering with the slot root. Commands take `run_id: Option<String>` and `None` means
    /// exactly this.
    pub id: String,
    /// The「本次训练名」THIS run's artifacts were built under (`weights/<slug>*`, `hps.name`),
    /// read from its own `run.json`. Empty = it never completed a run.
    pub model_name: String,
    pub info: crate::training::WorkspaceInfo,
    /// Newest checkpoint training can CONTINUE from, within THIS run. `None` with
    /// `has_resume_point` still true is RVC's「只保留最新」sentinel, whose file name carries no step.
    pub resume_step: Option<u64>,
    pub has_resume_point: bool,
    /// What the archive list would show FOR THIS RUN, and what it weighs.
    pub ckpt_count: u32,
    pub ckpt_bytes: u64,
}

/// One architecture slot of a project, as the detail page shows it.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SlotDetail {
    pub family: String,
    /// ★§F2⒝ batch 2 step ④ — every run this slot holds, sorted by id (see `trun::list_runs`).
    /// Exactly one entry for every slot that exists today; the shape is plural first so that the
    /// batch which mints a second one changes no consumer.
    pub runs: Vec<RunDetail>,
    pub bytes: u64,
    /// Everything under this slot that the archive list would show, and what it weighs —
    /// the SLOT total, summed over runs. Kept alongside the per-run numbers because the
    /// destructive/disk questions ("how big is this architecture") are slot questions.
    pub ckpt_count: u32,
    pub ckpt_bytes: u64,
    /// ★§F2⒝ — how many PREPROCESSING pools this slot holds, and what they weigh.
    ///
    /// This is the visible half of the batch's one real cost: a preprocessing identity change
    /// used to `shutil.rmtree` the old products and now keeps them as a sibling. Disk therefore
    /// grows where it used to shrink, and the user has to be able to SEE that rather than
    /// discover it as an unexplained slot size. (Reclaiming them is batch 2's job — a pool has
    /// to know which run references it before anything may delete it.)
    ///
    /// ⚠ `prep_` prefix ON PURPOSE: "pool" already means the project's imported DATASET
    /// elsewhere in this app (`training.ts`'s `poolCount` is that one's file count, and
    /// `has_dataset_pool` is about that too). Two different things called `poolCount` on the same
    /// screen is how a reader ends up reasoning about the wrong one.
    pub prep_pool_count: u32,
    pub prep_pool_bytes: u64,
}

/// A ledger row plus the answer to「它现在还在不在」.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportedModelStatus {
    pub name: String,
    pub model_type: String,
    pub from_ckpt_rel: String,
    pub at_ms: u64,
    /// ★§F2⒝ ④e — the RUN (or slot) that produced this checkpoint was deleted on purpose. The row
    /// still lists, for the same reason `installed: false` does: 「导出过」is history. What it
    /// changes is only what the row can still be USED for — it no longer protects anything from
    /// the snapshot cleanup, and it no longer counts as evidence for the stale-ledger tripwire.
    pub source_deleted: bool,
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
///
/// ⛔ S132 — propagates instead of skipping a slot it could not read: EMPTY is the permissive
/// answer for every consumer, so an unreadable slot must not be able to look like an unfrozen one.
fn frozen_lists(
    data_dir: &std::path::Path,
    project_id: &str,
) -> Result<Vec<Vec<crate::training::dsmanifest::DsSpeaker>>, String> {
    let mut out = Vec::new();
    for f in crate::training::tproject::FAMILIES {
        let v = crate::training::frozen_speakers(data_dir, project_id, f)
            .map_err(|e| e.to_string())?;
        if !v.is_empty() {
            out.push(v);
        }
    }
    Ok(out)
}

/// The first architecture slot that has FROZEN a speaker set, if any.
///
/// While one exists the speaker structure is immutable: `n_speakers` and the ordered slug list
/// are baked into that slot's emb_g rows, and changing either makes it unresumable
/// (`RESUME_SPEAKER_COUNT_MISMATCH` / `RESUME_SPEAKER_SET_MISMATCH`). Adding or removing FILES
/// stays allowed — it only costs a re-extraction, which the UI says out loud.
fn frozen_structure_family(
    data_dir: &std::path::Path,
    project_id: &str,
) -> Result<Option<String>, String> {
    for f in crate::training::tproject::FAMILIES {
        if !crate::training::frozen_speakers(data_dir, project_id, f)
            .map_err(|e| e.to_string())?
            .is_empty()
        {
            return Ok(Some(f.to_string()));
        }
    }
    Ok(None)
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
    /// Absolute path of the RUN whose artifacts this describes (the old「workspace」).
    ///
    /// ★§F2⒝ batch 2: `weights/` and the audition cache are run products, and the frontend joins
    /// both off this string. It resolves to the slot root for as long as there is no `runs/`
    /// container, so nothing about today's behaviour changes.
    pub workspace: String,
    /// The retrieval/cluster companion an import should carry, if one exists on disk. Same
    /// probe order the run summary uses as its no-summary fallback: RVC keeps its historical
    /// `total_fea.npy`, SoVITS looks for the cluster assets (built BEFORE training, so they
    /// exist even for an early stop). Vocoders have none — probing would only find another
    /// backend's leftovers.
    pub index_path: Option<String>,
}

/// The retrieval companion an import of THIS run should carry, or `None`.
///
/// ★§F2⒝ batch 2 — `total_fea.npy` and `cluster/` are RUN products. Layout 2 deliberately left
/// them at the slot root and `tpool::POOL_ENTRIES`' reason 2 names this very probe; that reasoning
/// was about the pool and does not survive per-run, because both are rebuilt wholesale by every
/// run. A fixed slot-relative name would hand one model the OTHER run's retrieval matrix, and this
/// probe fails OPEN — the wrong index arrives as a warning at most.
///
/// Split out of the command so it can be driven against a real layout-3 tree: the command itself
/// takes tauri `State` and nothing can call it from a test.
fn run_index_path(run: &crate::training::trun::RunDir, backend: &str) -> Option<String> {
    if backend == "vocoder" {
        // vocoders have none — probing would only find another backend's leftovers
        return None;
    }
    if backend == "rvc" {
        let p = run.join("total_fea.npy");
        return p.is_file().then(|| p.to_string_lossy().into_owned());
    }
    ["cluster/kmeans_10000.pt", "cluster/0.index_vectors.npy"]
        .iter()
        .map(|rel| run.join(rel))
        .find(|p| p.is_file())
        .map(|p| p.to_string_lossy().into_owned())
}

#[tauri::command]
pub async fn get_slot_export_context(
    state: State<'_, Arc<AppState>>,
    project_id: String,
    backend: String,
    run_id: Option<String>,
) -> Result<SlotExportContext, String> {
    checked_project_id(&project_id)?;
    let data_dir = data_root(&state);
    let family = crate::training::backend_family(&backend);
    let slot = crate::training::tproject::family_dir(&data_dir, &project_id, family);
    // ★§F2⒝ batch 2 step ④ — the SILENT twin of the project-detail coupling, and the reason this
    // command had to take a run id in the same batch as the run list. Its Err is CAUGHT by the
    // frontend (`TrainingPage`'s archive context probe), which then falls back to an empty
    // workspace string — and an empty string is not a visible failure, it is a WRONG ANSWER that
    // keeps going: `resolveIndexPath` probes a path under `""`, misses, and passes `index_file:
    // None` to `import_model`; that skips the WARN_INDEX_MISSING branch entirely and runs the
    // auto-detect BESIDE THE CHECKPOINT (`<run>/weights/`), where RVC's `total_fea.npy` has never
    // lived. Result: the model installs with no retrieval matrix, no warning, no CODE, no toast —
    // it only sounds wrong.
    let ws = crate::training::trun::resolve_run_dir(&slot, opt_run_id(run_id.as_deref().unwrap_or("")))
        .map_err(|e| e.to_string())?;
    let index_path = run_index_path(&ws, &backend);
    Ok(SlotExportContext {
        model_name: crate::training::tproject::run_model_name(&ws).unwrap_or_default(),
        workspace: ws.to_string_lossy().into_owned(),
        index_path,
    })
}

/// Rename ONE run's「本次训练名」. ★§F2⒝ 批 2 ④b —— **它只改标签。**
///
/// The capability only became safe once the artifact identity was frozen in the run
/// ([`crate::training::tproject::run_artifact_slug`]). Before that, a rename would have
/// re-pointed `hps.name`, `weights/<slug>*`, `audition/<slug>_*` and — through the pool the runs
/// SHARE — the `dataset_44k/<slug>/` slice directory, on the run's next start: every existing
/// product orphaned, a second full preprocessing tree grown, and nothing anywhere reporting it.
/// ⚠ §F2⒝ ④d took the last of those off the table for a sole speaker on an identity-v2 slot (the
/// slice directory is a constant there, not a name); the other three are still name-derived, so
/// the freeze is still what makes this command safe.
///
/// `run_id` empty ⇒「这个槽至多一个 run」, the same convention as every other run-taking command;
/// a slot with several runs refuses rather than picks (`RUN_AMBIGUOUS`).
#[tauri::command]
pub async fn rename_training_run(
    state: State<'_, Arc<AppState>>,
    project_id: String,
    backend: String,
    run_id: Option<String>,
    name: String,
) -> Result<(), String> {
    checked_project_id(&project_id)?;
    let name = name.trim();
    if name.is_empty() {
        return Err("TRAINING_NAME_EMPTY".into());
    }
    crate::commands::window::ensure_idle_for_run_rename(&state)?;
    if crate::crashlog::other_instance_alive() {
        return Err("RENAME_OTHER_INSTANCE".into());
    }
    let data_dir = data_root(&state);
    let family = crate::training::backend_family(&backend);
    let slot = crate::training::tproject::family_dir(&data_dir, &project_id, family);
    let run =
        crate::training::trun::resolve_run_dir(&slot, opt_run_id(run_id.as_deref().unwrap_or("")))
            .map_err(|e| e.to_string())?;
    crate::training::tproject::rename_run(&slot, &run, name).map_err(|e| e.to_string())
}

/// S167 (§F2⒟): export ONE archive checkpoint as the COMMUNITY-standard file set into `dest_dir`.
///
///   rvc         → `<name>.pth` — our `weights/*.pth` release snapshots ARE upstream savee()'s
///                 community format — plus `added_IVF{n}_Flat_nprobe_1_<name>_<v>.index` built
///                 from the run's `total_fea.npy` (upstream train_index()'s faiss half, run under
///                 the CONVERTER role; ASCII temp + rename, faiss cannot open CJK paths — S68f2).
///   sovits(_v2) → `<name>.pth` (the generator release snapshot) + `config.json` — the pair every
///                 so-vits-svc 4.x tool expects.
///
/// Reads training assets only; writes only into the user-picked `dest_dir`. Holds the convert
/// slot across the faiss build (the same interlock every converter user takes).
#[tauri::command]
pub async fn export_community_ckpt(
    state: State<'_, Arc<AppState>>,
    project_id: String,
    backend: String,
    ckpt_path: String,
    name: String,
    dest_dir: String,
) -> Result<Vec<String>, String> {
    checked_project_id(&project_id)?;
    let name = crate::models::sanitize_file_stem(name.trim());
    if name.is_empty() {
        return Err("TRAINING_NAME_EMPTY".into());
    }
    let data_dir = data_root(&state);
    let family = crate::training::backend_family(&backend).to_string();
    let proj = crate::training::tproject::project_dir(&data_dir, &project_id);
    // the row must live inside THIS project's tree — the command reads whatever path it is handed
    let proj_canon =
        proj.canonicalize().map_err(|e| format!("EXPORT_COMMUNITY_BAD_PROJECT: {e}"))?;
    let ckpt_canon = std::path::PathBuf::from(&ckpt_path)
        .canonicalize()
        .map_err(|e| format!("EXPORT_COMMUNITY_CKPT_MISSING: {e}"))?;
    if !ckpt_canon.starts_with(&proj_canon) {
        return Err("EXPORT_COMMUNITY_OUTSIDE_PROJECT".into());
    }
    let dest = std::path::PathBuf::from(&dest_dir);
    if !dest.is_dir() {
        return Err("EXPORT_COMMUNITY_DEST_MISSING".into());
    }
    let app_dir = state.app_dir.clone();
    let cache_dir = state.cache_dir.clone();
    // S167: the family dispatch is shared with the resource manager's community export
    // (`models::export_model_community`) — ONE source of truth for the file-set contract.
    let _convert = state.acquire_convert_slot()?;
    community_export_files(app_dir, cache_dir, &family, ckpt_canon, &name, dest).await
}

/// S167: community file-set builder shared by the training page (`export_community_ckpt`) and
/// the resource manager (`models::export_model_community`). The CALLER holds the convert slot —
/// the faiss index build runs under the converter role, same interlock as every converter user.
pub(crate) async fn community_export_files(
    app_dir: std::path::PathBuf,
    cache_dir: std::path::PathBuf,
    family: &str,
    ckpt_canon: std::path::PathBuf,
    name: &str,
    dest: std::path::PathBuf,
) -> Result<Vec<String>, String> {
    let name = name.to_string();
    match family {
        "rvc" => {
            // weights/<slug>*.pth → the run root (total_fea.npy's home) is two levels up
            let run_dir = ckpt_canon
                .parent()
                .filter(|p| p.file_name().is_some_and(|n| n == "weights"))
                .and_then(|p| p.parent())
                .ok_or("EXPORT_COMMUNITY_NOT_A_RELEASE")?
                .to_path_buf();
            let features = run_dir.join("total_fea.npy");
            if !features.exists() {
                return Err("EXPORT_COMMUNITY_NO_FEATURES".into());
            }
            // v1/v2 from the feature dimension itself — the one witness that cannot drift from
            // the matrix we are about to index (256 = ContentVec-256 = v1, 768 = v2).
            let version = match npy_second_dim(&features) {
                Some(256) => "v1",
                Some(768) => "v2",
                other => return Err(format!("EXPORT_COMMUNITY_BAD_FEATURES: dim {other:?}")),
            };
            let out_pth = dest.join(format!("{name}.pth"));
            tauri::async_runtime::spawn_blocking(move || -> Result<Vec<String>, String> {
                std::fs::copy(&ckpt_canon, &out_pth)
                    .map_err(|e| format!("EXPORT_COMMUNITY_COPY: {e}"))?;
                let tmp = cache_dir.join(format!("community_{}.index", std::process::id()));
                let _ = std::fs::remove_file(&tmp);
                let nlist =
                    crate::models::convert::build_community_index(&features, &tmp, &app_dir)
                        .map_err(|e| e.to_string())?;
                let out_index =
                    dest.join(format!("added_IVF{nlist}_Flat_nprobe_1_{name}_{version}.index"));
                let _ = std::fs::remove_file(&out_index);
                if std::fs::rename(&tmp, &out_index).is_err() {
                    // cache and dest can sit on different volumes — rename cannot cross them
                    std::fs::copy(&tmp, &out_index)
                        .map_err(|e| format!("EXPORT_COMMUNITY_INDEX_COPY: {e}"))?;
                    let _ = std::fs::remove_file(&tmp);
                }
                Ok(vec![out_pth.display().to_string(), out_index.display().to_string()])
            })
            .await
            .map_err(|e| format!("EXPORT_COMMUNITY_TASK: {e}"))?
        }
        "sovits" | "sovits_v2" => {
            // release snapshots live under <run>/weights/; the community pair needs the run's
            // own config.json (one level up)
            let run_dir = ckpt_canon
                .parent()
                .map(|p| {
                    if p.file_name().is_some_and(|n| n == "weights") {
                        p.parent().unwrap_or(p)
                    } else {
                        p
                    }
                })
                .ok_or("EXPORT_COMMUNITY_NOT_A_RELEASE")?
                .to_path_buf();
            let config = run_dir.join("config.json");
            if !config.exists() {
                return Err("EXPORT_COMMUNITY_NO_CONFIG".into());
            }
            let out_pth = dest.join(format!("{name}.pth"));
            let out_cfg = dest.join("config.json");
            tauri::async_runtime::spawn_blocking(move || -> Result<Vec<String>, String> {
                std::fs::copy(&ckpt_canon, &out_pth)
                    .map_err(|e| format!("EXPORT_COMMUNITY_COPY: {e}"))?;
                std::fs::copy(&config, &out_cfg)
                    .map_err(|e| format!("EXPORT_COMMUNITY_COPY: {e}"))?;
                Ok(vec![out_pth.display().to_string(), out_cfg.display().to_string()])
            })
            .await
            .map_err(|e| format!("EXPORT_COMMUNITY_TASK: {e}"))?
        }
        _ => Err("EXPORT_COMMUNITY_UNSUPPORTED".into()),
    }
}

/// Minimal .npy header reader: the SECOND dimension of a 2-D array, None on anything else —
/// enough to tell a 256-dim (v1) retrieval matrix from a 768-dim (v2) one without numpy.
fn npy_second_dim(path: &std::path::Path) -> Option<u64> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut head = [0u8; 10];
    f.read_exact(&mut head).ok()?;
    if &head[..6] != b"\x93NUMPY" {
        return None;
    }
    let hlen = if head[6] >= 2 {
        // format 2.0+: u32 header length at offset 8 (we already consumed 2 of its bytes)
        let mut rest = [0u8; 2];
        f.read_exact(&mut rest).ok()?;
        u32::from_le_bytes([head[8], head[9], rest[0], rest[1]]) as usize
    } else {
        u16::from_le_bytes([head[8], head[9]]) as usize
    };
    let mut hdr = vec![0u8; hlen.min(65536)];
    f.read_exact(&mut hdr).ok()?;
    let text = String::from_utf8_lossy(&hdr);
    let inner = text.split("'shape':").nth(1)?.split('(').nth(1)?.split(')').next()?;
    let dims: Vec<u64> = inner.split(',').filter_map(|s| s.trim().parse().ok()).collect();
    if dims.len() == 2 {
        Some(dims[1])
    } else {
        None
    }
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
        &frozen_lists(&data_dir, &project_id)?,
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
                    if let Some(fam) = frozen_structure_family(&data_dir, &project_id)? {
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
    let frozen = frozen_structure_family(&data_dir, &project_id)?;
    let facts = crate::training::dsmanifest::read_facts(
        &data_dir,
        &project_id,
        &frozen_lists(&data_dir, &project_id)?,
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

/// The two preprocessing facts a slot card carries: how many pools it holds, and what they cost.
///
/// ⛔ S141 §E2E-M5 — extracted so this seam is drivable. `get_training_project` takes
/// `State<'_, Arc<AppState>>` and calls `state.models.scan()` on the way in, so nothing can reach
/// these lines through the command; as two inline expressions they had no judgement of any kind,
/// and "the number on the card comes from the pool listing" was prose. It is the number the user
/// reads to decide what to delete.
///
/// ⛔ S132 — an unreadable `pools/` is not「零个池」. Reporting 0 B of preprocessing for a slot
/// holding gigabytes of it, on that exact screen, is worse than refusing to draw the page: the
/// error propagates, it does not get rounded down to zero.
fn slot_pool_facts(slot: &Path) -> Result<(u32, u64), String> {
    let pools = crate::training::tpool::list_pools(slot).map_err(|e| e.to_string())?;
    Ok((
        pools.len() as u32,
        pools.iter().map(|p| crate::commands::storage::dir_size(&p.dir)).sum(),
    ))
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
            source_deleted: !e.source_live(),
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
        &frozen_lists(&data_dir, &project_id)?,
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

    // ★§F2⒝ batch 2 step ④ — ONE ROW PER RUN, and the slot only carries what is genuinely a slot
    // fact (its bytes, its preprocessing pools, the totals).
    //
    // ⛔ What this replaces mattered more than it looks: `slot_info` REFUSES to answer「这个 run
    // 练到哪了」for a slot holding several runs, and the four slots were collected through a single
    // `Result` — so one ambiguous slot took the whole project page down. Asking per RUN removes the
    // ambiguity at the source instead of papering over it: every question here now names its run.
    let slots: Vec<SlotDetail> = crate::training::tproject::FAMILIES
        .iter()
        .filter(|f| crate::training::tproject::family_dir(&data_dir, &project_id, f).is_dir())
        .map(|f| -> Result<SlotDetail, String> {
            let slot = crate::training::tproject::family_dir(&data_dir, &project_id, f);
            let recs = crate::training::tproject::scan_project_ckpts(&data_dir, &project_id, Some(f));
            // The count/bytes pair and its S132 refusal live in `slot_pool_facts` (S141 §E2E-M5),
            // which is where they are drivable — this command is not.
            let (prep_pool_count, prep_pool_bytes) = slot_pool_facts(&slot)?;
            // `""` for an unmigrated slot, matching `CkptRecord::run_id` and the `None` that every
            // command takes — one vocabulary for「槽根就是那个 run」across Rust, IPC and the UI.
            let ids: Vec<String> = {
                // ⛔ S132 — an unreadable `runs/` is NOT「this slot has one unnamed run」: that
                // would show the user a slot whose real runs are hidden, and every action on the
                // fabricated row would address the slot root.
                let listed = crate::training::trun::list_runs(&slot).map_err(|e| e.to_string())?;
                if listed.is_empty() {
                    vec![String::new()]
                } else {
                    listed.into_iter().map(|r| r.id).collect()
                }
            };
            let runs = ids
                .into_iter()
                .map(|id| -> Result<RunDetail, String> {
                    let opt = if id.is_empty() { None } else { Some(id.as_str()) };
                    let dir = crate::training::trun::resolve_run_dir(&slot, opt)
                        .map_err(|e| e.to_string())?;
                    let mine: Vec<&crate::training::tproject::CkptRecord> =
                        recs.iter().filter(|r| r.run_id == id).collect();
                    // `scan_project_ckpts` returns newest-first by mtime — the same ordering
                    // upstream itself resumes by (the RVC sentinel makes step numbers unorderable).
                    // ★S118 §F8-res⒈: the CHOICE lives in `default_resume_record`, because "the
                    // mtime-newest Resumable" stopped meaning "what a 续训 continues from" the
                    // moment S117 started writing a `resume_best/` pair after the rolling one.
                    // ⚠ Fed THIS run's records only: across runs the mtime order says which run was
                    // trained last, which is a different question from「这个 run 从哪继续」.
                    let newest = crate::training::tproject::default_resume_record_of(&mine);
                    Ok(RunDetail {
                        model_name: crate::training::tproject::run_model_name(&dir)
                            .unwrap_or_default(),
                        info: crate::training::slot_info(&data_dir, &project_id, f, opt)
                            .map_err(|e| e.to_string())?,
                        resume_step: newest.and_then(|r| r.step),
                        has_resume_point: newest.is_some(),
                        ckpt_count: mine.len() as u32,
                        ckpt_bytes: mine.iter().map(|r| r.bytes).sum(),
                        id,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(SlotDetail {
                family: f.to_string(),
                runs,
                bytes: crate::commands::storage::dir_size(&slot),
                ckpt_count: recs.len() as u32,
                ckpt_bytes: recs.iter().map(|r| r.bytes).sum(),
                prep_pool_count,
                prep_pool_bytes,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;

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

    /// ★S141 §E2E-M5 —— 槽卡片上「预处理 N 份 · X」的那两个数,第一次有判据。
    ///
    /// 它们此前是 `get_training_project` 体内的两句内联表达式,而那个命令吃
    /// `State<'_, Arc<AppState>>` 并在半路 `state.models.scan()` ⇒ 结构上驱不动 ⇒
    /// 「这个数来自池的列举」只是一句散文。
    ///
    /// ⛔ 字节数用**字面值**断言,不用 `dir_size(p1) + dir_size(p2)`:后者拿被测代码用的
    /// 同一个求和器去算期望值,对「求和的是哪几个目录」有分辨力,对「它自己算错了」没有。
    ///
    /// ⚠ 夹具里那四样多余的字节各有各的用途,而且是**归因**用的,不是判死活用的:
    /// 每一种写错的求和法都会返回一个**一眼认得出是谁**的数(3000 = 对 / 15000 = 求了
    /// 整个 `pools/` / 20000 = 求了整个槽 / 4000 或 8000 混进来 = 那两条过滤掉了一半)。
    /// 少了它们,四种坏法会挤在同一个「不等于 3000」里,红是红了,却说不出红在哪一种 ——
    /// 而「一条闸的红必须能被归因」是 S129 立的铁律。
    #[test]
    fn the_preprocessing_line_counts_pools_and_charges_only_their_bytes() {
        let base = std::env::temp_dir().join(format!("utai_poolfacts_{}", uuid::Uuid::new_v4()));
        let slot = base.join("rvc");
        let pools = crate::training::tpool::pools_root(&slot);
        std::fs::create_dir_all(pools.join("paaa")).unwrap();
        std::fs::create_dir_all(pools.join("pbbb").join("dataset_44k")).unwrap();
        std::fs::write(pools.join("paaa").join("a.wav"), vec![0u8; 1000]).unwrap();
        std::fs::write(pools.join("pbbb").join("dataset_44k").join("b.wav"), vec![0u8; 2000])
            .unwrap();
        // 池之外的槽内字节 —— 没有它,「把 dir_size(slot) 当答案」那条变异不可见
        std::fs::create_dir_all(slot.join("weights")).unwrap();
        std::fs::write(slot.join("weights").join("G.pth"), vec![0u8; 5000]).unwrap();
        // pools/ 里不是池的两样东西,各自带着**很大**的字节数:如果它们混进来,3000 这个
        // 数字会变成一个一眼认得出是谁的数
        std::fs::create_dir_all(pools.join(".staging_x")).unwrap();
        std::fs::write(pools.join(".staging_x").join("half.wav"), vec![0u8; 4000]).unwrap();
        std::fs::write(pools.join("stray.txt"), vec![0u8; 8000]).unwrap();

        assert_eq!(
            slot_pool_facts(&slot).unwrap(),
            (2, 3000),
            "两个池、合计 3000 B。4000 混进来 = `.staging` 被当成池;8000 混进来 = pools/ 里的\
             普通文件被当成池;5000 混进来 = 求的是整个槽而不是那几个池;15000/20000 = 两者都有"
        );

        // 从没预处理过的槽是**正常状态**,不是错误
        let virgin = base.join("sovits");
        std::fs::create_dir_all(&virgin).unwrap();
        assert_eq!(slot_pool_facts(&virgin).unwrap(), (0, 0), "没有 pools/ ⇒ 零份,不是报错");

        // ⛔ 但「读不动」必须响亮:退化成 (0,0) 会在用户用来决定删什么的那一屏上,
        // 把一个压着几 GB 预处理的槽画成 0 B(S132)
        let broken = base.join("vocoder");
        std::fs::create_dir_all(&broken).unwrap();
        std::fs::write(crate::training::tpool::pools_root(&broken), b"not a directory").unwrap();
        // ⛔ 用 match 而不是 `unwrap_err()`:被吞掉时 `unwrap_err` 抛的是
        // 「called Result::unwrap_err() on an Ok value」,读起来像**测试写坏了**,
        // 而它其实是产品缺陷 —— 一条红必须能被归因(S129 铁律),两种坏法各说各的话。
        match slot_pool_facts(&broken) {
            Ok(v) => panic!(
                "读不动的 pools/ 被吞成了 {v:?} —— 必须让整个槽报错。退化成 (0, 0) 会在用户\
                 用来决定删什么的那一屏上,把一个压着几 GB 预处理的槽画成 0 B(S132)"
            ),
            Err(e) => assert!(
                e.contains("POOLS_DIR_UNREADABLE"),
                "报错了,但没说出是 pools/ 读不动 ⇒ 下一个人分不清「闸坏了」和「盘坏了」:{e}"
            ),
        }

        let _ = std::fs::remove_dir_all(&base);
    }

    /// ★§F2⒝ 批 2 ④b —— 改名命令的四道闸,钉在源码上。
    ///
    /// `rename_training_run` takes `State`, so no unit test can drive it;「`rename_run` 是对的」and
    /// 「命令在调它之前该拒绝的都拒绝了」are two separate claims and only the first is drivable.
    /// Dropping any one of these is silent in a different way: no idle guard ⇒ `run.json` is
    /// rewritten under a live run while the running snapshot keeps the old label; no
    /// other-instance guard ⇒ two processes write the same file; no empty-name guard ⇒ every
    /// reader that treats `""` as「这个 run 还没起过名」starts asking for a name on every 继续训练.
    ///
    /// ⛔ Full-line comments are blanked before the search: a raw substring scan reads comments
    /// too, so one comment naming a guard would satisfy the assertion the guard is supposed to
    /// carry (the sibling ratchets in `training::mod` were hardened for the same reason).
    #[test]
    fn renaming_a_run_is_guarded_before_it_touches_the_file() {
        static THIS_RS: &str = include_str!("training.rs");
        let code: String = THIS_RS
            .lines()
            .map(|l| {
                if l.trim_start().starts_with("//") {
                    " ".repeat(l.len())
                } else {
                    l.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        let open = code
            .find("pub async fn rename_training_run(")
            .expect("the rename command is gone");
        let body = &code[open..];
        let end = body.find("\n}\n").expect("unterminated fn");
        let body = &body[..end];
        for needle in [
            "TRAINING_NAME_EMPTY",
            "ensure_idle_for_run_rename(",
            "other_instance_alive()",
        ] {
            let at = body.find(needle).unwrap_or_else(|| {
                panic!("rename_training_run no longer contains {needle:?} — a guard was dropped")
            });
            let write = body
                .find("tproject::rename_run(")
                .expect("the rename command no longer writes");
            assert!(at < write, "{needle} is checked AFTER the file is rewritten");
        }
    }

    /// ⛔ §F2⒝ batch 2 — the export's retrieval companion, found in the RUN.
    ///
    /// It fails OPEN by design: a missing index is a warning, never an error. That is exactly why
    /// it needs a test of its own — pointed at the slot after the migration it would find nothing,
    /// every RVC import would install without its retrieval matrix, and the only symptom is that
    /// the voice sounds wrong. The command around it takes tauri `State`, so this is the level a
    /// test can reach.
    #[test]
    fn the_export_index_is_found_inside_the_run() {
        use crate::training::trun::RunDir;
        let base = std::env::temp_dir().join(format!("utai_idx_{}", uuid::Uuid::new_v4()));
        let slot = base.join("rvc");
        let run = RunDir::for_test(slot.join("runs").join("rfeedfacefeed"));
        std::fs::create_dir_all(run.join("cluster")).unwrap();
        std::fs::write(run.join("total_fea.npy"), b"x").unwrap();
        std::fs::write(run.join("cluster").join("0.index_vectors.npy"), b"x").unwrap();

        assert_eq!(run_index_path(&run, "rvc"), Some(run.join("total_fea.npy").to_string_lossy().into_owned()));
        // ⚠ the literal carries its own `/`, so the answer is a mixed-separator path — that is
        // pre-existing and Windows accepts it. Asserting a `join`-built expectation here was MY
        // invention and it is what turned this test red the first time it ran.
        assert_eq!(
            run_index_path(&run, "sovits"),
            Some(run.join("cluster/0.index_vectors.npy").to_string_lossy().into_owned()),
            "kmeans is preferred but absent here, so the vectors file answers"
        );
        assert_eq!(run_index_path(&run, "vocoder"), None, "probing would find another backend's");

        // …and the slot itself holds neither, which is the whole point
        assert_eq!(run_index_path(&RunDir::for_test(slot.clone()), "rvc"), None);
        assert_eq!(run_index_path(&RunDir::for_test(slot), "sovits"), None);
        let _ = std::fs::remove_dir_all(&base);
    }

    /// ⛔ §F2⒝ batch 2 step ④ — `""` and「没传」are the SAME question.
    ///
    /// A mutation probe found this line had nothing driving it: drop the emptiness filter and every
    /// start / slot probe asks `trun::resolve_run_dir` for a run literally NAMED `""`, which fails
    /// `run_id_is_usable` and answers `RUN_ID_INVALID` — i.e. the app stops working entirely, from
    /// a one-token change, with the whole suite still green. The frontend sends `""` from a
    /// `#[serde(default)]` field on every non-run-aware path, so this is the normal case, not an
    /// edge one.
    #[test]
    fn an_empty_run_id_means_the_slot_holds_one_run() {
        assert_eq!(opt_run_id(""), None);
        assert_eq!(opt_run_id("   "), None, "a whitespace payload is not a run name");
        assert_eq!(opt_run_id("rfeedfacefeed"), Some("rfeedfacefeed"));
        assert_eq!(opt_run_id(" rfeedfacefeed "), Some("rfeedfacefeed"));
        // …and the request wrapper answers identically, because the two must never diverge
        let req = |v: serde_json::Value| -> StartTrainingRequest { serde_json::from_value(v).unwrap() };
        let base = serde_json::json!({
            "model_name": "m", "backend": "rvc", "version": "v2", "sample_rate": "40k",
            "dataset_files": [], "total_epoch": 1, "batch_size": 1,
        });
        assert_eq!(run_id_of(&req(base.clone())), None, "an absent field is「一个 run」");
        let mut with = base.clone();
        with["run_id"] = serde_json::json!("");
        assert_eq!(run_id_of(&req(with)), None, "…and so is an empty one");
        let mut named = base;
        named["run_id"] = serde_json::json!("rfeedfacefeed");
        assert_eq!(run_id_of(&req(named)), Some("rfeedfacefeed"));
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
    run_id: Option<String>,
) -> Result<crate::training::WorkspaceInfo, String> {
    checked_project_id(&project_id)?;
    crate::training::slot_info(
        &data_root(&state),
        &project_id,
        &backend,
        opt_run_id(run_id.as_deref().unwrap_or("")),
    )
    .map_err(|e| e.to_string())
}

// `get_training_workspace_info(name, backend)` lived here until S76 batch 4 — see the note in
// `training::mod` where its implementation was. `get_training_slot_info` is the same answer
// keyed by the project id, which is the only identity that survives a rename.
