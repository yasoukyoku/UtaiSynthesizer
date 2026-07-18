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

#[derive(serde::Serialize)]
pub struct WorkspaceUsage {
    pub slug: String,
    /// Original model name (from run.json) — the slug is ASCII-lossy for CJK names.
    pub name: String,
    /// Training family from run_manifest.json ("rvc"/"sovits"/"vocoder"); "" when unreadable.
    pub family: String,
    pub bytes: u64,
    /// A reusable shared dataset pool exists (the diff 免导入直训 mode reads this workspace).
    pub has_pool: bool,
}

#[derive(serde::Serialize)]
pub struct StorageReport {
    pub data_dir: String,
    pub cache_bytes: u64,
    pub models_bytes: u64,
    pub msst_bytes: u64,
    pub runtimes_bytes: u64,
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
                let slug = entry.file_name().to_string_lossy().to_string();
                // `.del_*` = a torn workspace delete (rename-then-remove, see below) — invisible to
                // workspace_path, finish removing it here instead of listing it.
                if slug.starts_with('.') {
                    let _ = std::fs::remove_dir_all(&p);
                    continue;
                }
                let bytes = dir_size(&p);
                training_bytes += bytes;
                ws_audition += dir_size(&p.join("audition"));
                // Original name from run.json (the manifest stores no name; the slug is lossy).
                let name = std::fs::read_to_string(p.join("run.json"))
                    .ok()
                    .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                    .and_then(|v| v.get("model_name").and_then(|n| n.as_str()).map(String::from))
                    .unwrap_or_else(|| slug.clone());
                let family = std::fs::read_to_string(p.join("run_manifest.json"))
                    .ok()
                    .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                    .and_then(|v| v.get("backend").and_then(|b| b.as_str()).map(String::from))
                    .unwrap_or_default();
                let has_pool = crate::training::has_dataset_pool(&p);
                workspaces.push(WorkspaceUsage { slug, name, family, bytes, has_pool });
            }
        }
        workspaces.sort_by(|a, b| b.bytes.cmp(&a.bytes));
        Ok(StorageReport {
            data_dir: root.to_string_lossy().to_string(),
            cache_bytes: dir_size(&root.join("cache")),
            models_bytes: dir_size(&models_dir),
            msst_bytes: dir_size(&models_dir.join("msst")),
            runtimes_bytes: dir_size(&root.join("runtimes")),
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

/// Delete ONE training workspace, whole-dir (never leave manifest-less checkpoints — the resume
/// guards read that as a corrupt state — nor an empty dir, which fakes `exists` in the reuse
/// dialogs). Kills 续训 + the diff shared pool for that model, BY DESIGN: every dependent flow
/// re-probes workspace_info/has_dataset_pool live and degrades to "import data first".
#[tauri::command]
pub async fn delete_training_workspace(
    state: State<'_, Arc<AppState>>,
    slug: String,
) -> Result<u64, String> {
    if state.training.is_active() {
        return Err("TRAINING_ACTIVE".into());
    }
    // Block a concurrent audition (its conversion writes into <ws>/audition + holds ORT sessions)
    // for the whole delete — same interlock start_training takes.
    let _audition_lock = crate::commands::audition::FlightGuard::acquire("CLEANUP_BUSY")?;
    // The slug must be a plain directory name produced by slugify (ASCII + `_` + hex) — refuse
    // anything path-like so a hostile/buggy caller can't escape the training root.
    if slug.is_empty() || !slug.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
        return Err("WORKSPACE_MISSING".into());
    }
    let ws = data_root(&state).join("training").join(&slug);
    if !ws.is_dir() {
        return Err("WORKSPACE_MISSING".into());
    }
    // Drop any ORT sessions loaded from this workspace (audition candidate onnx) — Windows file
    // locks would otherwise fail the remove. Non-destructive: sessions reload on miss.
    state.inference.engine.unload_paths_with_prefix(&ws);
    let freed = dir_size(&ws);
    // RENAME-then-remove: remove_dir_all is not atomic — an interruption (crash / locked file)
    // could leave exactly the manifest-less-checkpoint partial state the resume guards treat as
    // corrupt. The same-volume rename IS atomic; a torn removal leaves only an invisible `.del_*`
    // dir (workspace_path never resolves to it) that get_storage_report finishes off later.
    let tomb = ws.with_file_name(format!(".del_{slug}_{}", std::process::id()));
    std::fs::rename(&ws, &tomb).map_err(|e| format!("WORKSPACE_DELETE_FAILED: {e}"))?;
    tauri::async_runtime::spawn_blocking(move || {
        std::fs::remove_dir_all(&tomb).map_err(|e| format!("WORKSPACE_DELETE_FAILED: {e}"))
    })
    .await
    .map_err(|e| format!("STORAGE_JOIN: {e}"))??;
    Ok(freed)
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
                let aud = entry.path().join("audition");
                if aud.is_dir() {
                    let len = dir_size(&aud);
                    if std::fs::remove_dir_all(&aud).is_ok() {
                        freed += len;
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
                let p = entry.path();
                if let Ok(md) = entry.metadata() {
                    if md.is_file() {
                        files.push((entry.file_name().to_string_lossy().to_string(), p, md.len()));
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
