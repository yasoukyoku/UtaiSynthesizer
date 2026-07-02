use std::path::PathBuf;
use std::sync::Arc;
use tauri::State;

use crate::AppState;

/// One media/render file to consolidate INTO the bundle: an absolute source path → a bundle-relative
/// destination (e.g. "media/foo.wav"). The frontend computes these (it knows which fields are paths).
#[derive(serde::Deserialize)]
pub struct FileCopy {
    pub from: String,
    pub to: String,
}

/// Returned by `open_project_archive`: the working dir the archive was extracted to + the project.json.
#[derive(serde::Serialize)]
pub struct OpenedProject {
    pub work_dir: String,
    pub project_json: String,
}

fn path_hash(s: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    format!("{:016x}", h.finish())
}

/// Save a `.usp` project as a SINGLE-FILE archive (a zip) holding `project.json` + the media/render
/// audio at bundle-relative paths. The TS store is the authoritative document model; this is
/// schema-agnostic, robust, unicode-safe file I/O. Written atomically (temp sibling + rename) so a
/// crash mid-write can't corrupt the prior good archive. Returns the list of copy sources that were
/// MISSING on disk (skipped) so the frontend can warn the user instead of a clean "saved".
#[tauri::command]
pub async fn save_project_archive(
    usp_path: String,
    project_json: String,
    copies: Vec<FileCopy>,
) -> Result<Vec<String>, String> {
    use std::io::Write;
    let final_path = PathBuf::from(&usp_path);
    let mut tmp = final_path.clone().into_os_string();
    tmp.push(".tmp");
    let tmp_path = PathBuf::from(tmp);
    let mut missing: Vec<String> = Vec::new();

    {
        let file = std::fs::File::create(&tmp_path).map_err(|e| format!("create {}: {e}", tmp_path.display()))?;
        let mut zip = zip::ZipWriter::new(file);
        let opts: zip::write::SimpleFileOptions =
            zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);

        zip.start_file("project.json", opts).map_err(|e| format!("zip project.json: {e}"))?;
        zip.write_all(project_json.as_bytes()).map_err(|e| format!("write project.json: {e}"))?;

        for c in &copies {
            let from = PathBuf::from(&c.from);
            if !from.exists() {
                // A referenced media/render file is gone (e.g. cache sweep). Skip it so the rest of the
                // project still saves, but REPORT it — silently writing an archive that's missing audio
                // over the previous good one reads as "saved" while baking in data loss.
                tracing::warn!("save_project_archive: source missing, skipped: {}", from.display());
                missing.push(c.from.clone());
                continue;
            }
            let bytes = std::fs::read(&from).map_err(|e| format!("read {}: {e}", from.display()))?;
            zip.start_file(c.to.replace('\\', "/"), opts).map_err(|e| format!("zip {}: {e}", c.to))?;
            zip.write_all(&bytes).map_err(|e| format!("write {}: {e}", c.to))?;
        }
        zip.finish().map_err(|e| format!("finalize archive: {e}"))?;
    }

    std::fs::rename(&tmp_path, &final_path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp_path); // don't strand a .usp.tmp if the rename fails (target locked)
        format!("finalize {}: {e}", final_path.display())
    })?;
    tracing::info!("Project archive saved: {}", final_path.display());
    Ok(missing)
}

/// Open a `.usp` archive: extract it to a per-archive working dir under the app cache and return that
/// dir + `project.json`. The frontend resolves the bundle-relative media/render paths against the work
/// dir (playback/decode need absolute paths) and hydrates its store. The work dir is regenerable
/// (re-extracted on reopen) and swept with the rest of the cache.
#[tauri::command]
pub async fn open_project_archive(
    state: State<'_, Arc<AppState>>,
    usp_path: String,
) -> Result<OpenedProject, String> {
    use std::io::Read;
    // Extract into a FRESH staging dir and swap it in only after the WHOLE archive extracted. The
    // still-open project's media/renders may live in usp_work (a .usp-opened project), and the frontend
    // keeps that project open on ANY open failure — so nothing here may destroy existing extractions
    // before this archive is proven good. Older extractions are pruned only once the frontend COMMITS
    // the load (`prune_usp_work`), or at startup when no autosave recovery is pending (lib.rs).
    let usp_work = state.cache_dir.join("usp_work");
    let work_dir = usp_work.join(path_hash(&usp_path));
    let staging = usp_work.join(format!("{}.extracting", path_hash(&usp_path)));
    // Crash-between-renames recovery: a prior swap that moved <hash> aside but died before moving the
    // staging in leaves <hash>.old with no <hash> — roll it back so the extraction (and a pending
    // autosave recovery pointing into it) isn't lost.
    let old = usp_work.join(format!("{}.old", path_hash(&usp_path)));
    if old.exists() && !work_dir.exists() {
        let _ = std::fs::rename(&old, &work_dir);
    }
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging).map_err(|e| format!("create work dir: {e}"))?;

    let extract = || -> Result<String, String> {
        let file = std::fs::File::open(&usp_path).map_err(|e| format!("open {usp_path}: {e}"))?;
        let mut archive = zip::ZipArchive::new(file).map_err(|e| format!("read archive: {e}"))?;

        let mut project_json = String::new();
        for i in 0..archive.len() {
            let mut entry = archive.by_index(i).map_err(|e| format!("archive entry {i}: {e}"))?;
            let name = entry.name().to_string();
            if name == "project.json" {
                entry.read_to_string(&mut project_json).map_err(|e| format!("read project.json: {e}"))?;
                continue;
            }
            if name.ends_with('/') {
                continue; // directory marker
            }
            // enclosed_name() rejects zip-slip (`..` traversal) — only extract safe, contained paths.
            let rel = match entry.enclosed_name() {
                Some(p) => p.to_path_buf(),
                None => return Err(format!("unsafe archive entry: {name}")),
            };
            let outpath = staging.join(rel);
            if let Some(parent) = outpath.parent() {
                std::fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
            }
            let mut out = std::fs::File::create(&outpath).map_err(|e| format!("create {}: {e}", outpath.display()))?;
            std::io::copy(&mut entry, &mut out).map_err(|e| format!("extract {}: {e}", outpath.display()))?;
        }
        if project_json.is_empty() {
            return Err("archive has no project.json".to_string());
        }
        // Sanity-parse before the swap: a zip whose project.json is truncated/garbage would otherwise
        // extract "successfully", replace the current extraction on a same-path re-open, and only THEN
        // fail in the frontend parser — with the old project still open on now-replaced media.
        serde_json::from_str::<serde_json::Value>(&project_json)
            .map_err(|e| format!("project.json is not valid JSON: {e}"))?;
        Ok(project_json)
    };

    let project_json = match extract() {
        Ok(j) => j,
        Err(e) => {
            let _ = std::fs::remove_dir_all(&staging);
            return Err(e);
        }
    };

    // Swap the staging dir into place. Re-opening the SAME archive replaces its prior extraction with
    // content-identical files — move the old one aside first so a failed rename can never leave neither.
    let _ = std::fs::remove_dir_all(&old);
    if work_dir.exists() {
        if let Err(e) = std::fs::rename(&work_dir, &old) {
            let _ = std::fs::remove_dir_all(&staging); // don't leak a fully-extracted orphan
            return Err(format!("stage out prior extraction: {e}"));
        }
    }
    if let Err(e) = std::fs::rename(&staging, &work_dir) {
        // Roll the prior extraction back so the current session keeps its files.
        if old.exists() {
            let _ = std::fs::rename(&old, &work_dir);
        }
        let _ = std::fs::remove_dir_all(&staging);
        return Err(format!("finalize extraction: {e}"));
    }
    let _ = std::fs::remove_dir_all(&old);

    Ok(OpenedProject {
        work_dir: work_dir.to_string_lossy().to_string(),
        project_json,
    })
}

/// Remove every usp_work extraction EXCEPT `keep_dir`. Called by the frontend only AFTER a newly
/// opened archive has fully hydrated the store (the point of no return) — any earlier and a failed
/// open could delete the still-open previous project's extracted media. The no-project-yet case is
/// covered at startup (lib.rs removes usp_work when no autosave recovery is pending).
#[tauri::command]
pub async fn prune_usp_work(state: State<'_, Arc<AppState>>, keep_dir: String) -> Result<(), String> {
    let usp_work = state.cache_dir.join("usp_work");
    let keep = PathBuf::from(&keep_dir);
    let entries = match std::fs::read_dir(&usp_work) {
        Ok(e) => e,
        Err(_) => return Ok(()), // nothing extracted yet
    };
    for entry in entries.flatten() {
        let p = entry.path();
        // Never touch in-flight staging / swap-transition dirs — a slow prune racing the NEXT open's
        // extraction must not delete files out from under it.
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.ends_with(".extracting") || name.ends_with(".old") {
            continue;
        }
        if p != keep {
            let _ = std::fs::remove_dir_all(&p);
        }
    }
    Ok(())
}

/// Whether a path currently exists on disk. Used so a plain Save doesn't silently re-create the
/// project archive at a path the user has deleted/moved — it falls through to Save As instead.
#[tauri::command]
pub fn path_exists(path: String) -> bool {
    std::path::Path::new(&path).exists()
}

// ── Autosave (crash recovery) ───────────────────────────────────────────────────────────────────
// Schema-agnostic, exactly like save_project_archive: the FRONTEND owns the autosave JSON shape — it
// reuses the SAME serialization as a real save (`buildAutosaveJson`, sharing `serializeProject`) so
// autosave can never drift from save as the document grows. Rust just reads/writes/removes the single
// `<app_dir>/autosave.json`. The FILE'S EXISTENCE is the unclean-exit marker: it's written while there
// are unsaved changes and removed on a clean save / new / open / discard-on-close, so finding it at
// startup means the last session didn't shut down cleanly → offer to recover.

#[tauri::command]
pub async fn write_autosave(state: State<'_, Arc<AppState>>, json: String) -> Result<(), String> {
    let path = state.app_dir.join("autosave.json");
    let tmp = state.app_dir.join("autosave.json.tmp");
    std::fs::write(&tmp, json).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, &path).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn read_autosave(state: State<'_, Arc<AppState>>) -> Result<Option<String>, String> {
    let path = state.app_dir.join("autosave.json");
    match std::fs::read_to_string(&path) {
        Ok(s) => Ok(Some(s)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.to_string()),
    }
}

#[tauri::command]
pub async fn clear_autosave(state: State<'_, Arc<AppState>>) -> Result<(), String> {
    let path = state.app_dir.join("autosave.json");
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.to_string()),
    }
}
