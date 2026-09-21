//! S61 — storage usage report + cleanup commands (Settings「存储占用与清理」).
//!
//! What accumulates with normal use (and what these commands touch):
//! - `<data>/cache/**` — decode dedup + stretch products + workflow/vocal render run dirs +
//!   range-test scratch. Regenerable (re-decode / re-render); `cleanup_render_cache` deletes
//!   everything EXCEPT `usp_work` (the OPEN project's extracted media — deleting it destroys the
//!   session) and a frontend-supplied protected set (paths the open project still references).
//! - `<data>/training/<slug>` — training workspaces: raw dataset copies + preprocessed features +
//!   checkpoints. NOT regenerable (retraining costs hours); deleted per-workspace, whole-dir only
//!   (a partial delete leaves the manifest-less-checkpoint anomaly the resume guards refuse).
//! - audition caches — `<ws>/audition/*` + `<models>/**/<stem>.audition_spk*.wav`. Regenerable.
//! - logs — the install dir's `logs/` (get_log_dir; pre-S68e legacy home was
//!   `%LOCALAPPDATA%/com.utaisynthesizer.app/logs`) daily files (never pruned elsewhere).
//! Models / runtime packs are USER ASSETS managed by their own UIs (resource manager, MSST
//! manager, Settings runtime panel) — the report shows their size, cleanup never touches them.
//!
//! Errors are stable CODEs per the i18n rule (CLEANUP_BUSY / TRAINING_ACTIVE / WORKSPACE_MISSING).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use tauri::State;

use crate::AppState;

fn data_root(state: &AppState) -> PathBuf {
    state
        .cache_dir
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| state.app_dir.join("data"))
}

/// Recursive directory size; unreadable entries count as 0 (never fails the whole report).
/// pub(crate): also used by settings' data-dir reclaim to report freed bytes.
pub(crate) fn dir_size(path: &Path) -> u64 {
    let mut total = 0u64;
    let Ok(rd) = std::fs::read_dir(path) else { return 0 };
    for entry in rd.flatten() {
        let p = entry.path();
        if let Ok(md) = entry.metadata() {
            if md.is_dir() {
                total += dir_size(&p);
            } else {
                total += md.len();
            }
        }
    }
    total
}

/// Path normalization for the protected-set compare: Windows paths are case-insensitive and the
/// frontend mixes `/` and `\` — compare lowercase forward-slash forms.
fn norm_key(p: &Path) -> String {
    p.to_string_lossy().replace('\\', "/").to_lowercase()
}

/// One architecture slot's share of a project. Everything the confirm dialogs need to say
/// what a given button is about to destroy — a body that under-states its blast radius is the
/// exact failure「绝不静默」exists to prevent.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SlotUsage {
    pub family: String,
    pub bytes: u64,
    /// Periodic snapshots under `weights/` (the cleanup's candidate pool, before protections).
    pub snapshots: u32,
    /// How much「清理未导入的快照」would actually free right now.
    pub cleanable_bytes: u64,
    /// Shallow-diffusion progress living inside the sovits slot — deleting that slot takes it
    /// too, and the word "sovits" says nothing about diffusion.
    pub diff_steps: u64,
}

#[derive(serde::Serialize)]
pub struct WorkspaceUsage {
    /// Project id = the directory name under `<data>/training`.
    pub slug: String,
    /// Display name from project.json — the id is ASCII-lossy for CJK names.
    pub name: String,
    /// Which architecture slots this project actually holds, joined with `+`
    /// ("rvc" / "sovits+vocoder" / ""). One project can now hold several.
    pub family: String,
    pub bytes: u64,
    /// A reusable shared dataset pool exists (every slot of this project trains off it).
    pub has_pool: bool,
    /// The project's shared `dataset/` — deleting the PROJECT takes it, deleting a slot never does.
    pub dataset_bytes: u64,
    pub slots: Vec<SlotUsage>,
    /// Set when the layout migration could not classify this directory — nothing was moved
    /// and nothing was deleted, but the user has to decide what it is. Never hide such a
    /// project: "盘上还在、app 里没了" is the one outcome this refactor must not produce.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub needs_attention: Option<String>,
}

#[derive(serde::Serialize)]
pub struct StorageReport {
    pub data_dir: String,
    pub cache_bytes: u64,
    pub models_bytes: u64,
    pub msst_bytes: u64,
    pub amt_bytes: u64,
    pub runtimes_bytes: u64,
    /// S74b: the CUDA inference runtime (~1.6 GB across `<app>/runtime/ort/cuda` + `<app>/runtime/
    /// cuda`). It lives next to the PROGRAM, not under the data root, so every earlier version of
    /// this report omitted the single biggest optional download in the app — the storage page
    /// could not account for it and a user reclaiming space had no idea it existed.
    pub cuda_runtime_bytes: u64,
    pub dictionaries_bytes: u64,
    pub logs_bytes: u64,
    /// Audition caches (workspace audition dirs + model-side audition wavs) — a subset of
    /// models_bytes/training totals, reported separately because it is cleanable on its own.
    pub audition_bytes: u64,
    pub training_bytes: u64,
    pub workspaces: Vec<WorkspaceUsage>,
}

/// Model-side audition wav? (`<stem>.audition_spk*.wav`, written next to the model by
/// render_model_audition; invalidated wholesale here.)
fn is_audition_wav(name: &str) -> bool {
    name.contains(".audition_spk") && name.ends_with(".wav")
}

/// Every audition-cache directory one training PROJECT can hold.
///
/// ONE source, because the two consumers — the storage report's byte total and 「清理试听缓存」—
/// have to agree about what that button empties, and the note beside the cleanup records that they
/// already drifted once: S76 moved the cache one level down and only one of them followed, which
/// made the button silently free zero bytes forever.
///
/// Three levels, all real and all live at once:
/// * `<project>/<family>/runs/<run>/audition` — ★§F2⒝ batch 2, one cache per RUN;
/// * `<project>/<family>/audition` — every slot that has not been through the run migration
///   (`trun::run_dirs` answers with the slot itself there, so this arm is the same expression);
/// * `<project>/audition` — the pre-S76 shape, still on disk for anything flagged or postponed.
pub(crate) fn audition_dirs_of_project(project: &Path) -> Vec<std::path::PathBuf> {
    crate::training::tproject::FAMILIES
        .iter()
        // ⛔ S132 — a slot whose `runs/` cannot be listed contributes NO directories rather than
        // the slot root. Both consumers (the size report and the cache purge) then under-count /
        // under-delete, which is the harmless direction; pretending the slot root is the run would
        // point a purge at a directory that is not a cache.
        .flat_map(|f| {
            let slot = project.join(f);
            crate::training::trun::run_dirs(&slot).unwrap_or_else(|e| {
                tracing::warn!("cannot enumerate the runs of {} ({e}) — its audition caches are not counted", slot.display());
                Vec::new()
            })
        })
        .map(|r| r.join("audition"))
        .chain(std::iter::once(project.join("audition")))
        .collect()
}

/// Sum of model-side audition wavs under the models tree (recursive).
fn model_audition_bytes(dir: &Path) -> u64 {
    let mut total = 0u64;
    let Ok(rd) = std::fs::read_dir(dir) else { return 0 };
    for entry in rd.flatten() {
        let p = entry.path();
        if let Ok(md) = entry.metadata() {
            if md.is_dir() {
                total += model_audition_bytes(&p);
            } else if is_audition_wav(&entry.file_name().to_string_lossy()) {
                total += md.len();
            }
        }
    }
    total
}

#[tauri::command]
pub async fn get_storage_report(state: State<'_, Arc<AppState>>) -> Result<StorageReport, String> {
    let root = data_root(&state);
    // The CUDA runtime sits under the APP dir (program-adjacent), not the data root.
    let cuda_dirs = [
        state.app_dir.join("runtime").join("ort").join("cuda"),
        state.app_dir.join("runtime").join("cuda"),
    ];
    // Snapshot the registry names here: `cleanable_bytes` must answer「点这个按钮会释放多少」
    // with the SAME protection set the command itself will apply, and one of its rules is
    // "the stem is still an installed model" (a ledger row can be missing — S61: enumerate
    // every holder). Taken on this thread; the blocking closure below owns AppState-free data.
    let installed: Vec<String> = state.models.list().into_iter().map(|m| m.name).collect();
    tauri::async_runtime::spawn_blocking(move || {
        let models_dir = root.join("models");
        let training_dir = root.join("training");
        let mut workspaces = Vec::new();
        let mut training_bytes = 0u64;
        let mut ws_audition = 0u64;
        if let Ok(rd) = std::fs::read_dir(&training_dir) {
            for entry in rd.flatten() {
                let p = entry.path();
                if !p.is_dir() {
                    continue;
                }
                let id = entry.file_name().to_string_lossy().to_string();
                // `.del_*` = a torn delete (rename-then-remove, see below) — invisible to the
                // layout, finish removing it here instead of listing it. The prefix is checked
                // EXACTLY: the migration stages a legacy tree under `.mig_<id>` in this very
                // directory, and the old「任何点开头目录」判据 would have eaten it mid-migration.
                if id.starts_with(".del_") {
                    let _ = crate::util::remove_dir_all_robust(&p);
                    continue;
                }
                if id.starts_with('.') {
                    // `.mig_<id>` = a torn layout migration's staging tree. It is REAL user
                    // data (the whole legacy workspace lives in it until the next boot folds
                    // it back or forward), so it must not vanish from the totals — but it has
                    // no project row of its own to sit on.
                    training_bytes += dir_size(&p);
                    continue;
                }
                // S168: the bundled code dirs (utai_train/assets, present when the data root
                // is the install root) are app code, not workspaces — no row, no delete
                // button, and their few MB are not training data. This enumerator never keyed
                // on project.json, which is exactly how the phantom row got its delete button.
                if crate::training::tproject::is_reserved_training_dir(&id) {
                    continue;
                }
                let bytes = dir_size(&p);
                training_bytes += bytes;
                // S76: one row per PROJECT. Display name comes from project.json (the only
                // place it survives a workspace whose run.json was never written), families
                // are whichever slots exist, and the reusable pool is the project's dataset.
                let meta = crate::training::tproject::read_meta(&root, &id);
                let name = meta
                    .as_ref()
                    .map(|m| m.name.clone())
                    .filter(|n| !n.is_empty())
                    .unwrap_or_else(|| id.clone());
                let mut fams: Vec<String> = Vec::new();
                let mut slots: Vec<SlotUsage> = Vec::new();
                for f in crate::training::tproject::FAMILIES {
                    let fd = p.join(f);
                    if !fd.is_dir() {
                        continue;
                    }
                    fams.push(f.to_string());
                    let recs = crate::training::tproject::scan_project_ckpts(&root, &id, Some(f));
                    let plan = crate::training::tproject::plan_cleanup(
                        &recs,
                        meta.as_ref().map(|m| m.export_ledger_since_ms).unwrap_or(0),
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_millis() as u64)
                            .unwrap_or(0),
                        &|stem: &str| installed.iter().any(|n| n == stem || stem.starts_with(n.as_str())),
                    );
                    slots.push(SlotUsage {
                        family: f.to_string(),
                        bytes: dir_size(&fd),
                        snapshots: recs
                            .iter()
                            .filter(|r| {
                                matches!(r.kind, crate::training::tproject::CkptKind::Release)
                            })
                            .count() as u32,
                        cleanable_bytes: plan.freeable_bytes,
                        diff_steps: recs
                            .iter()
                            .filter(|r| r.rel.contains("/diffusion/"))
                            .filter_map(|r| r.step)
                            .max()
                            .unwrap_or(0),
                    });
                }
                // every family slot's runs, plus the two legacy shapes — ONE source, shared with
                // 「清理试听缓存」so the number and the button can never describe different sets
                ws_audition += audition_dirs_of_project(&p).iter().map(|d| dir_size(d)).sum::<u64>();
                let has_pool = crate::training::tproject::has_dataset(&root, &id);
                workspaces.push(WorkspaceUsage {
                    slug: id,
                    name,
                    family: fams.join("+"),
                    bytes,
                    has_pool,
                    dataset_bytes: dir_size(&p.join(crate::training::tproject::DATASET_DIR)),
                    slots,
                    needs_attention: meta.as_ref().and_then(|m| m.needs_attention.clone()),
                });
            }
        }
        workspaces.sort_by(|a, b| b.bytes.cmp(&a.bytes));
        let mut amt_bytes = dir_size(&models_dir.join("amt"));
        if let Ok(home) = std::env::var("USERPROFILE") {
            let home_path = PathBuf::from(home);
            amt_bytes += dir_size(&home_path.join(".cache").join("music_ai_models"));
            amt_bytes += dir_size(&home_path.join(".music-to-midi").join("models"));
        }

        Ok(StorageReport {
            data_dir: root.to_string_lossy().to_string(),
            cache_bytes: dir_size(&root.join("cache")),
            models_bytes: dir_size(&models_dir),
            msst_bytes: dir_size(&models_dir.join("msst")),
            amt_bytes,
            runtimes_bytes: dir_size(&root.join("runtimes")),
            cuda_runtime_bytes: cuda_dirs.iter().map(|d| dir_size(d)).sum(),
            dictionaries_bytes: dir_size(&root.join("dictionaries")),
            logs_bytes: dir_size(&crate::logging::get_log_dir()),
            audition_bytes: ws_audition + model_audition_bytes(&models_dir),
            training_bytes,
            workspaces,
        })
    })
    .await
    .map_err(|e| format!("STORAGE_JOIN: {e}"))?
}

/// Delete regenerable render/decode caches under `<data>/cache`, EXCEPT: `usp_work` (the open
/// project's extracted media), the frontend-supplied `protected` paths (everything the open
/// project still references: clip sources, deposited lane audio, the runtime node-output cache)
/// and their sidecar jsons. Returns bytes freed. The frontend additionally gates on
/// playing/rendering; the Rust guards below are the authoritative backstop for backend jobs.
#[tauri::command]
pub async fn cleanup_render_cache(
    state: State<'_, Arc<AppState>>,
    protected: Vec<String>,
) -> Result<u64, String> {
    if crate::commands::inference::voice_render_active() {
        return Err("CLEANUP_BUSY".into());
    }
    // MSST separation writes stems straight into the cache tree and does NOT check the audition
    // flight flag — a live worker mid-write would race the sweep (S61 leftover, closed here). The
    // frontend additionally gates on running executions, but that state desyncs when a run errors
    // out while the backend worker keeps going — this is the authoritative check.
    if matches!(
        state.separation.status().state,
        crate::separation::SeparationState::LoadingModel | crate::separation::SeparationState::Separating
    ) {
        return Err("CLEANUP_BUSY".into());
    }
    // HOLD the audition flight flag for the whole sweep (not a mere load() check): VoiceRunGuard +
    // audition both refuse to START while it is held, so no render can begin writing fresh run-dir
    // files mid-sweep and get them deleted from under it (audit S61 — check-then-act window).
    let _flight = crate::commands::audition::FlightGuard::acquire("CLEANUP_BUSY")?;
    let cache_dir = state.cache_dir.clone();
    tauri::async_runtime::spawn_blocking(move || Ok(sweep_cache_tree(&cache_dir, &protected)))
        .await
        .map_err(|e| format!("STORAGE_JOIN: {e}"))?
}

/// The cache sweep core (testable): delete every file under `cache_dir` except (a) anything inside
/// the top-level `usp_work` subtree (the open .usp project's extracted media — its own lifecycle
/// prunes it), (b) the `protected` paths, and (c) a protected file's `<key>.json` completion-marker
/// sidecar (audio_cache/stretch pattern). Empty dirs are pruned afterwards (writers create_dir_all
/// on demand). Returns bytes freed; locked/undeletable files are skipped and not counted.
fn sweep_cache_tree(cache_dir: &Path, protected: &[String]) -> u64 {
    let prot: std::collections::HashSet<String> =
        protected.iter().map(|p| norm_key(Path::new(p))).collect();
    let is_protected = |p: &Path| -> bool {
        if prot.contains(&norm_key(p)) {
            return true;
        }
        if p.extension().map(|e| e == "json").unwrap_or(false) {
            return prot.contains(&norm_key(&p.with_extension("wav")));
        }
        false
    };
    fn sweep(dir: &Path, is_protected: &dyn Fn(&Path) -> bool, freed: &mut u64) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for entry in rd.flatten() {
            let p = entry.path();
            let Ok(md) = entry.metadata() else { continue };
            if md.is_dir() {
                sweep(&p, is_protected, freed);
                let _ = std::fs::remove_dir(&p); // removes only if now empty
            } else if !is_protected(&p) {
                let len = md.len();
                if std::fs::remove_file(&p).is_ok() {
                    *freed += len;
                }
            }
        }
    }
    let mut freed = 0u64;
    let Ok(rd) = std::fs::read_dir(cache_dir) else { return 0 };
    for entry in rd.flatten() {
        let p = entry.path();
        if entry.file_name().to_string_lossy() == "usp_work" {
            continue;
        }
        if p.is_dir() {
            sweep(&p, &is_protected, &mut freed);
            let _ = std::fs::remove_dir(&p);
        } else if !is_protected(&p) {
            let len = entry.metadata().map(|m| m.len()).unwrap_or(0);
            if std::fs::remove_file(&p).is_ok() {
                freed += len;
            }
        }
    }
    freed
}

// `delete_training_workspace` lived here until S76 batch 4. Batch 3b replaced it with
// `training_delete_project` / `training_delete_slot`, which know about the project layout
// (slot-granular, shared `dataset/`, GAN pairs kept together, structured report) and are
// guarded a full tier harder: `ensure_idle_for_package_delete` enumerates every in-process
// holder (convert included — this one missed it), plus the two cross-process cases a lock
// cannot express (`other_instance_alive`, `RECLAIM_TOUCHING_TRAINING`). It survived batch 3
// registered but with ZERO callers: a destructive command reachable by name over IPC and by
// nothing else. Its one good idea — refuse a path-like id — is now `checked_project_id` in
// `commands::training`, applied to every id-taking command instead of just this one.

// ─────────────────────────────────────────────────────────────────────────────
// #21 音频缓存管理 — 逐条占用明细 + 一键清理超大文件。
// 语义与 cleanup_render_cache 同一保护集:usp_work(打开的工程媒体)与 protected(当前工程仍引用的
// 路径)永不触碰;只有「清得安全」的文件才进统计/被删。CLEANUP_BUSY 护栏与渲染缓存清扫完全一致。
// ─────────────────────────────────────────────────────────────────────────────

/// One top-level cache entry's usage, for the per-entry audit (folder or loose file).
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CacheEntryDetail {
    pub name: String,
    /// "dir" | "file"
    pub kind: &'static str,
    pub bytes: u64,
    pub file_count: u64,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CacheAudit {
    pub cache_dir: String,
    pub total_bytes: u64,
    pub entries: Vec<CacheEntryDetail>,
}

fn count_files(dir: &Path) -> u64 {
    let mut n = 0u64;
    let Ok(rd) = std::fs::read_dir(dir) else { return 0 };
    for entry in rd.flatten() {
        let p = entry.path();
        if p.is_dir() {
            n += count_files(&p);
        } else {
            n += 1;
        }
    }
    n
}

/// Walk a top-level cache entry and build its usage detail. Never fails the whole audit.
fn describe_entry(entry: &std::fs::DirEntry) -> Option<CacheEntryDetail> {
    let name = entry.file_name().to_string_lossy().to_string();
    let md = entry.metadata().ok()?;
    let (bytes, file_count, kind) = if md.is_dir() {
        (dir_size(&entry.path()), count_files(&entry.path()), "dir")
    } else {
        (md.len(), 1, "file")
    };
    Some(CacheEntryDetail { name, kind, bytes, file_count })
}

/// Per-entry cache usage report (#21). Sync walk; runs on the blocking pool because a multi-GB
/// cache tree with thousands of files takes a moment — never blocks the async UI thread.
#[tauri::command]
pub async fn cache_audit(state: State<'_, Arc<AppState>>) -> Result<CacheAudit, String> {
    let cache_dir = state.cache_dir.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let mut entries: Vec<CacheEntryDetail> = Vec::new();
        let mut total = 0u64;
        if let Ok(rd) = std::fs::read_dir(&cache_dir) {
            for entry in rd.flatten() {
                if let Some(d) = describe_entry(&entry) {
                    total += d.bytes;
                    entries.push(d);
                }
            }
        }
        entries.sort_by(|a, b| b.bytes.cmp(&a.bytes)); // biggest first
        Ok(CacheAudit { cache_dir: cache_dir.to_string_lossy().to_string(), total_bytes: total, entries })
    })
    .await
    .map_err(|e| format!("STORAGE_JOIN: {e}"))?
}

/// Shared recursion: delete files matching `should_delete`, honouring the protected set. When
/// `prune_empty` the now-empty dirs are removed (writers create_dir_all on demand).
fn delete_cache_files(
    dir: &Path,
    is_protected: &dyn Fn(&Path) -> bool,
    should_delete: &dyn Fn(u64) -> bool,
    prune_empty: bool,
    freed: &mut u64,
    count: &mut u64,
) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for entry in rd.flatten() {
        let p = entry.path();
        let Ok(md) = entry.metadata() else { continue };
        if md.is_dir() {
            delete_cache_files(&p, is_protected, should_delete, prune_empty, freed, count);
            if prune_empty {
                let _ = std::fs::remove_dir(&p);
            }
        } else if !is_protected(&p) {
            let len = md.len();
            if should_delete(len) && std::fs::remove_file(&p).is_ok() {
                *freed += len;
                *count += 1;
            }
        }
    }
}

/// Build the protected predicate for a given protected-set (mirrors sweep_cache_tree): a protected
/// path AND a protected file's `<key>.json` completion-marker sidecar both survive.
fn make_is_protected(protected: &[String]) -> impl Fn(&Path) -> bool {
    let prot: std::collections::HashSet<String> =
        protected.iter().map(|p| norm_key(Path::new(p))).collect();
    move |p: &Path| {
        if prot.contains(&norm_key(p)) {
            return true;
        }
        if p.extension().map(|e| e == "json").unwrap_or(false) {
            return prot.contains(&norm_key(&p.with_extension("wav")));
        }
        false
    }
}

/// The CLEANUP_BUSY flight gate for the #21 commands — mirrors cleanup_render_cache so a live
/// render/separation file is never deleted out from under its writer (audit S61). The returned
/// guard MUST be held across the whole delete (the `.await` on spawn_blocking), or a render could
/// start writing mid-sweep.
struct CacheCleanupGuard {
    _flight: crate::commands::audition::FlightGuard,
}

fn begin_cache_cleanup(state: &tauri::State<'_, Arc<AppState>>) -> Result<CacheCleanupGuard, String> {
    // ORDER MATTERS (audit M1): acquire the flight flag FIRST, then check the renderers. The
    // renderer's VoiceRunGuard::acquire refuses while AUDITION_IN_FLIGHT is set, so holding the
    // flag here makes "check then start" atomic — a render cannot slip in between the check and
    // the sweep. Checking renderers first would leave that window (cleanup sees nothing running,
    // a render starts, then the sweep deletes its output mid-write).
    let _flight = crate::commands::audition::FlightGuard::acquire("CLEANUP_BUSY")
        .map_err(|e| format!("{e}"))?;
    if crate::commands::inference::voice_render_active() {
        return Err("CLEANUP_BUSY".into());
    }
    if matches!(
        state.separation.status().state,
        crate::separation::SeparationState::LoadingModel | crate::separation::SeparationState::Separating
    ) {
        return Err("CLEANUP_BUSY".into());
    }
    Ok(CacheCleanupGuard { _flight })
}

/// Delete ONE top-level cache entry (a subfolder or a loose file) by name, honouring the protected
/// set. `name` must be a single path element — anything with a separator is refused (no traversal).
#[tauri::command]
pub async fn cleanup_cache_entry(
    state: State<'_, Arc<AppState>>,
    name: String,
    protected: Vec<String>,
) -> Result<u64, String> {
    // Path-traversal guard: refuse separators / relative escapes. A cache entry name is a bare
    // element ("audio_cache", "<segment_id>", "<hash>.wav"…). Anything else is a caller bug.
    if name.is_empty()
        || name.contains('/')
        || name.contains('\\')
        || name == ".."
        || name == "."
    {
        return Err("BAD_CACHE_NAME".into());
    }
    // usp_work is the OPEN project's extracted media — deleting it destroys the session. The
    // other cache cleanups sweep it only through the frontend-protected set; a per-entry delete
    // must not be able to remove it whole (audit H1).
    if name == "usp_work" {
        return Err("CLEANUP_USP_WORK".into());
    }
    let _guard = begin_cache_cleanup(&state)?;
    let cache_dir = state.cache_dir.clone();
    let prot = protected.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let target = cache_dir.join(&name);
        if !target.starts_with(&cache_dir) || !target.exists() {
            return Ok(0u64);
        }
        if target.file_name().map(|f| f.to_string_lossy().to_string()) != Some(name) {
            return Ok(0u64); // belt-and-suspenders: target is not exactly the named element
        }
        let is_protected = make_is_protected(&prot);
        let mut freed = 0u64;
        let mut count = 0u64;
        if target.is_dir() {
            delete_cache_files(&target, &is_protected, &|_| true, true, &mut freed, &mut count);
            let _ = std::fs::remove_dir(&target); // remove dir if now empty
            // remove_dir_all would ignore the protected set — never use it here.
        } else if !is_protected(&target) {
            if let Ok(len) = target.metadata().map(|m| m.len()) {
                if std::fs::remove_file(&target).is_ok() {
                    freed += len;
                }
            }
            let _ = std::fs::remove_file(target.with_extension("json")); // stray sidecar
        }
        Ok(freed)
    })
    .await
    .map_err(|e| format!("STORAGE_JOIN: {e}"))?
}

/// "一键清理超大文件": delete files exceeding `threshold_bytes` anywhere under the cache tree
/// (except usp_work + the protected set), returning bytes freed. Small files are left in place.
#[tauri::command]
pub async fn cleanup_oversized_cache(
    state: State<'_, Arc<AppState>>,
    threshold_bytes: u64,
    protected: Vec<String>,
) -> Result<u64, String> {
    let _guard = begin_cache_cleanup(&state)?;
    let cache_dir = state.cache_dir.clone();
    let prot = protected.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let is_protected = make_is_protected(&prot);
        let mut freed = 0u64;
        let mut count = 0u64;
        if let Ok(rd) = std::fs::read_dir(&cache_dir) {
            for entry in rd.flatten() {
                let p = entry.path();
                if p.file_name().map(|f| f.to_string_lossy().to_string()).as_deref() == Some("usp_work") {
                    continue; // the OPEN project's extracted media — its own lifecycle prunes it
                }
                if p.is_dir() {
                    delete_cache_files(&p, &is_protected, &|len| len > threshold_bytes, true, &mut freed, &mut count);
                    let _ = std::fs::remove_dir(&p);
                } else if !is_protected(&p) {
                    if let Ok(md) = p.metadata() {
                        let len = md.len();
                        if len > threshold_bytes && std::fs::remove_file(&p).is_ok() {
                            freed += len;
                        }
                    }
                }
            }
        }
        let _ = count;
        Ok(freed)
    })
    .await
    .map_err(|e| format!("STORAGE_JOIN: {e}"))?
}

/// Delete all audition caches: every workspace's `audition/` dir + every model-side
/// `<stem>.audition_spk*.wav`. Pure caches — re-auditioning regenerates them.
#[tauri::command]
pub async fn cleanup_audition_caches(state: State<'_, Arc<AppState>>) -> Result<u64, String> {
    if state.training.is_active() {
        return Err("TRAINING_ACTIVE".into());
    }
    let _audition_lock = crate::commands::audition::FlightGuard::acquire("CLEANUP_BUSY")?;
    let root = data_root(&state);
    // Candidate onnx under <ws>/audition may be loaded as ORT sessions — unload before deleting.
    state
        .inference
        .engine
        .unload_paths_with_prefix(&root.join("training"));
    tauri::async_runtime::spawn_blocking(move || {
        let mut freed = 0u64;
        if let Ok(rd) = std::fs::read_dir(root.join("training")) {
            for entry in rd.flatten() {
                // Where the caches are is decided ONCE, in `audition_dirs_of_project` — the
                // storage total above and this button must empty the same set, and S76 already
                // paid for them knowing it separately.
                for rel in audition_dirs_of_project(&entry.path()) {
                    if rel.is_dir() {
                        let len = dir_size(&rel);
                        if crate::util::remove_dir_all_robust(&rel).is_ok() {
                            freed += len;
                        }
                    }
                }
            }
        }
        fn sweep_wavs(dir: &Path, freed: &mut u64) {
            let Ok(rd) = std::fs::read_dir(dir) else { return };
            for entry in rd.flatten() {
                let p = entry.path();
                let Ok(md) = entry.metadata() else { continue };
                if md.is_dir() {
                    sweep_wavs(&p, freed);
                } else if is_audition_wav(&entry.file_name().to_string_lossy()) {
                    let len = md.len();
                    if std::fs::remove_file(&p).is_ok() {
                        *freed += len;
                    }
                }
            }
        }
        sweep_wavs(&root.join("models"), &mut freed);
        Ok(freed)
    })
    .await
    .map_err(|e| format!("STORAGE_JOIN: {e}"))?
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_root(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "utai_storage_test_{tag}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn write(p: &Path, bytes: usize) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, vec![0u8; bytes]).unwrap();
    }

    #[test]
    fn merge_presets_upserts_and_skips_invalid() {
        let mut arr = vec![
            serde_json::json!({ "name": "A", "workflow": { "nodes": [1] } }),
            serde_json::json!({ "name": "B", "workflow": { "nodes": [] } }),
        ];
        let incoming = vec![
            // 同名覆盖
            serde_json::json!({ "name": "A", "workflow": { "nodes": [9] } }),
            // 新名追加
            serde_json::json!({ "name": "C", "workflow": { "nodes": [3] } }),
            // 残缺条目逐类跳过:空白名 / 缺名 / workflow 非对象 / workflow 缺失
            serde_json::json!({ "name": "   ", "workflow": {} }),
            serde_json::json!({ "workflow": {} }),
            serde_json::json!({ "name": "D", "workflow": [1, 2] }),
            serde_json::json!({ "name": "E" }),
        ];
        let imported = merge_presets(&mut arr, incoming);
        assert_eq!(imported, 2);
        assert_eq!(arr.len(), 3);
        assert_eq!(
            arr[0].get("workflow").unwrap().get("nodes").unwrap().get(0).unwrap(),
            &serde_json::json!(9),
            "同名条目被覆盖"
        );
        assert_eq!(arr[2].get("name").unwrap(), "C");
    }

    #[test]
    fn sweep_respects_protected_uspwork_and_sidecars() {
        let root = tmp_root("sweep");
        // protected decode copy + its sidecar
        write(&root.join("audio_cache/aaaa.wav"), 100);
        write(&root.join("audio_cache/aaaa.json"), 10);
        // unprotected decode copy + sidecar
        write(&root.join("audio_cache/bbbb.wav"), 200);
        write(&root.join("audio_cache/bbbb.json"), 10);
        // protected stretch product + sidecar; unprotected sibling
        write(&root.join("audio_cache/stretch/cc_r1.100000.wav"), 300);
        write(&root.join("audio_cache/stretch/cc_r1.100000.json"), 10);
        write(&root.join("audio_cache/stretch/dd_r0.900000.wav"), 400);
        // run dir with a protected stem + an unprotected intermediate
        write(&root.join("seg1/r123/vocals.wav"), 500);
        write(&root.join("seg1/r123/node_tmp.wav"), 600);
        // usp_work must be untouched even though nothing in it is protected
        write(&root.join("usp_work/h1/media/song.wav"), 700);
        // range_test scratch — swept
        write(&root.join("range_test/scale_60.wav"), 800);

        // protected paths arrive frontend-style: forward slashes, mixed case
        let protected = vec![
            root.join("audio_cache/aaaa.wav").to_string_lossy().replace('\\', "/").to_uppercase(),
            root.join("audio_cache/stretch/cc_r1.100000.wav").to_string_lossy().to_string(),
            root.join("seg1/r123/vocals.wav").to_string_lossy().to_string(),
        ];
        let freed = sweep_cache_tree(&root, &protected);
        assert_eq!(freed, 200 + 10 + 400 + 600 + 800);
        assert!(root.join("audio_cache/aaaa.wav").exists());
        assert!(root.join("audio_cache/aaaa.json").exists(), "protected wav keeps its sidecar");
        assert!(!root.join("audio_cache/bbbb.wav").exists());
        assert!(!root.join("audio_cache/bbbb.json").exists());
        assert!(root.join("audio_cache/stretch/cc_r1.100000.wav").exists());
        assert!(root.join("audio_cache/stretch/cc_r1.100000.json").exists());
        assert!(!root.join("audio_cache/stretch/dd_r0.900000.wav").exists());
        assert!(root.join("seg1/r123/vocals.wav").exists());
        assert!(!root.join("seg1/r123/node_tmp.wav").exists());
        assert!(root.join("usp_work/h1/media/song.wav").exists(), "usp_work untouched");
        assert!(!root.join("range_test").exists(), "emptied dirs pruned");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn dir_size_and_audition_helpers() {
        let root = tmp_root("size");
        write(&root.join("a/b.bin"), 123);
        write(&root.join("a/c/d.bin"), 77);
        assert_eq!(dir_size(&root), 200);
        assert!(is_audition_wav("model.audition_spk0.wav"));
        assert!(is_audition_wav("model.audition_spk3_r2.wav"));
        assert!(!is_audition_wav("model.onnx"));
        assert!(!is_audition_wav("song.wav"));
        write(&root.join("m/voice.audition_spk0.wav"), 50);
        write(&root.join("m/voice.onnx"), 999);
        assert_eq!(model_audition_bytes(&root), 50);
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// ⛔ §F2⒝ batch 2 — 「清理试听缓存」and the number beside it must name the same set, and the
    /// set now has THREE shapes on disk at once. A per-run cache missing from this list is bytes
    /// the storage page counts (`dir_size` on the slot is recursive) and the button never frees:
    /// a cleanup that reports 「已释放 0 B」 while the disk stays full, which is exactly the
    /// regression the S76 note in this file records.
    #[test]
    fn audition_dirs_cover_every_layout_that_exists_on_disk() {
        let root = tmp_root("auddirs");
        let proj = root.join("p1_aaaabbbb");
        // pre-S76: the cache sat beside the project
        write(&proj.join("audition/x/model.json"), 1);
        // S76: one level down, in the family slot (a slot with no `runs/` container)
        write(&proj.join("sovits/audition/x/model.json"), 1);
        // §F2⒝ batch 2: one per run
        write(&proj.join("rvc/runs/rfeedfacefeed/audition/x/model.json"), 1);
        write(&proj.join("rvc/runs/rbeefbeefbeef/audition/y/model.json"), 1);

        let dirs = audition_dirs_of_project(&proj);
        for want in [
            proj.join("audition"),
            proj.join("sovits").join("audition"),
            proj.join("rvc").join("runs").join("rfeedfacefeed").join("audition"),
            proj.join("rvc").join("runs").join("rbeefbeefbeef").join("audition"),
        ] {
            assert!(dirs.contains(&want), "{} is a live cache and must be listed", want.display());
        }
        // …and nothing outside the project sneaks in
        assert!(dirs.iter().all(|d| d.starts_with(&proj)));
        std::fs::remove_dir_all(&root).unwrap();
    }
}

/// Delete rolled log files, keeping the newest (the active one is OS-locked on Windows anyway —
/// failures are skipped and simply not counted). Daily file names sort chronologically.
#[tauri::command]
pub async fn cleanup_logs() -> Result<u64, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let dir = crate::logging::get_log_dir();
        let mut files: Vec<(String, PathBuf, u64)> = Vec::new();
        if let Ok(rd) = std::fs::read_dir(&dir) {
            for entry in rd.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                // Only rolled DAILY LOG files are candidates. This directory also holds the crash
                // sentinels (`session.<pid>.alive`, crashlog.rs) for THIS session and any live
                // sibling instance — deleting our own sentinel would silently skip the next crash
                // autopsy after a hard kill, and deleting a sibling's would break
                // `other_instance_alive` (destructive-delete guard). Never touch non-log files.
                if !name.starts_with(crate::logging::LOG_PREFIX) {
                    continue;
                }
                let p = entry.path();
                if let Ok(md) = entry.metadata() {
                    if md.is_file() {
                        files.push((name, p, md.len()));
                    }
                }
            }
        }
        files.sort_by(|a, b| a.0.cmp(&b.0));
        files.pop(); // keep the newest (current) file
        let mut freed = 0u64;
        for (_, p, len) in files {
            if std::fs::remove_file(&p).is_ok() {
                freed += len;
            }
        }
        Ok(freed)
    })
    .await
    .map_err(|e| format!("STORAGE_JOIN: {e}"))?
}

// ─── 工作流预设 · 全局保存/加载 ─────────────────────────────────────────────────────────
// 保存到 <data_root>/workflow_presets.json(与模型/缓存同数据根,跨工程共享)。内容为
// 一个对象数组,每项 { name, workflow },workflow 是前端序列化好的整张工作流图。
fn preset_file(state: &AppState) -> PathBuf {
    data_root(state).join("workflow_presets.json")
}

/// 读预设文件为条目数组(文件缺失或损坏时返回 `[]`,与保存路径的容错口径一致)。
fn read_preset_array(path: &Path) -> Vec<serde_json::Value> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

/// 把一批导入条目合并进 `arr`(同名覆盖、新名追加),返回实际合并的条数。
/// 残缺条目(名字为空/缺失,或 workflow 不是对象)整条跳过,不中断整批 ——
/// 用户手改的导出文件不应把预设库写坏(规划 6-7)。
fn merge_presets(arr: &mut Vec<serde_json::Value>, incoming: Vec<serde_json::Value>) -> u32 {
    let mut imported = 0u32;
    for entry in incoming {
        let name = entry
            .get("name")
            .and_then(|n| n.as_str())
            .unwrap_or_default()
            .to_string();
        if name.trim().is_empty() || !entry.get("workflow").map_or(false, |w| w.is_object()) {
            continue;
        }
        if let Some(ix) = arr
            .iter()
            .position(|v| v.get("name").and_then(|n| n.as_str()) == Some(name.as_str()))
        {
            arr[ix] = entry;
        } else {
            arr.push(entry);
        }
        imported += 1;
    }
    imported
}

/// 返回所有已保存的工作流预设(JSON 数组字符串;首次使用尚无文件时返回 `[]`)。
#[tauri::command]
pub fn load_workflow_presets(state: State<'_, Arc<AppState>>) -> Result<String, String> {
    let path = preset_file(&state);
    match std::fs::read_to_string(&path) {
        Ok(s) => Ok(s),
        Err(_) => Ok("[]".into()),
    }
}

/// 按名字保存一个工作流预设(同名则覆盖)。workflow 为前端序列化好的工作流对象。
#[tauri::command]
pub fn save_workflow_preset(
    state: State<'_, Arc<AppState>>,
    name: String,
    workflow: serde_json::Value,
) -> Result<(), String> {
    let root = data_root(&state);
    std::fs::create_dir_all(&root).map_err(|e| e.to_string())?;
    let path = preset_file(&state);
    let mut arr = read_preset_array(&path);
    let entry = serde_json::json!({ "name": name, "workflow": workflow });
    if let Some(ix) = arr.iter().position(|v| v.get("name").and_then(|n| n.as_str()) == Some(name.as_str())) {
        arr[ix] = entry;
    } else {
        arr.push(entry);
    }
    let out = serde_json::to_string_pretty(&arr).map_err(|e| e.to_string())?;
    std::fs::write(&path, out).map_err(|e| e.to_string())
}

/// 导出全部工作流预设到用户选定的 JSON 文件,返回导出的预设条数(规划 6-7)。
#[tauri::command]
pub fn export_workflow_presets(
    state: State<'_, Arc<AppState>>,
    dest: String,
) -> Result<u32, String> {
    let arr = read_preset_array(&preset_file(&state));
    let out = serde_json::to_string_pretty(&arr).map_err(|e| e.to_string())?;
    std::fs::write(&dest, out).map_err(|e| e.to_string())?;
    Ok(arr.len() as u32)
}

/// 从用户选定的 JSON 文件导入工作流预设:同名覆盖、新名追加、残缺条目跳过,
/// 返回实际导入的条数(规划 6-7)。
#[tauri::command]
pub fn import_workflow_presets(
    state: State<'_, Arc<AppState>>,
    src: String,
) -> Result<u32, String> {
    let raw = std::fs::read_to_string(&src).map_err(|e| e.to_string())?;
    let incoming: Vec<serde_json::Value> =
        serde_json::from_str(&raw).map_err(|e| format!("PRESET_PARSE: {e}"))?;
    let root = data_root(&state);
    std::fs::create_dir_all(&root).map_err(|e| e.to_string())?;
    let path = preset_file(&state);
    let mut arr = read_preset_array(&path);
    let imported = merge_presets(&mut arr, incoming);
    let out = serde_json::to_string_pretty(&arr).map_err(|e| e.to_string())?;
    std::fs::write(&path, out).map_err(|e| e.to_string())?;
    Ok(imported)
}
