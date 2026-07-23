//! Training PROJECT identity, on-disk layout, and the one-way migration off the legacy
//! per-model-name workspace layout.
//!
//! ## Why this module exists
//!
//! The pre-S76 layout had no project concept at all: a training run lived in
//! `<data>/training/<slugify(model_name)>/`, so "which runs belong together" was *inferred*
//! from the display name. One data set could not feed two architectures without importing it
//! twice, the cross-family collision had to be refused at runtime
//! (`WORKSPACE_BACKEND_MISMATCH`), and the directory identity came from `DefaultHasher`
//! (SipHash-1-3), which std explicitly does not promise to keep stable across Rust releases —
//! a toolchain bump would have orphaned every existing workspace.
//!
//! The layout is now:
//!
//! ```text
//! <data>/training/
//!   <project_id>/
//!     project.json          ← self-describing; the AUTHORITY on what this directory is
//!     dataset/              ← the ONE shared layer (raw imported audio)
//!     rvc/ sovits/ sovits_v2/ vocoder/   ← one slot per family = the old workspace root
//! ```
//!
//! `sovits_diff` is not a family: `backend_family` maps it onto `sovits`, because shallow
//! diffusion shares the main model's preprocessing caches — that has always been the point.
//!
//! ## Invariants (breaking any of these silently loses user work)
//!
//! 1. **`project.json` presence is the migration commit point.** Discovery keys on it, so it
//!    is written last-but-one and the in-progress marker is cleared after it. There is
//!    deliberately no index file in the way: the truth is the directory itself
//!    (`projects.json` is a *cache* and is introduced later, with the listing UI that needs
//!    per-project sizes).
//! 2. **Migration never merges and never deletes.** It is three whole-directory renames, so
//!    it costs the same whether the workspace holds 12 files or 120 000, and every torn state
//!    is reversible by the mirror-image renames.
//! 3. **A migrated project keeps the old slug as its id.** That slug already satisfies the
//!    path-escape guard, already carries a hash suffix (so it can never be a Windows reserved
//!    device name), and reusing it means migration never has to rename the project directory
//!    itself. Only NEW projects get an id from [`new_project_id`], which uses sha2 instead of
//!    `DefaultHasher`.
//! 4. **`model_slug` is NOT this module's business.** It is the artifact identity
//!    (`hps.name`, `weights/<slug>*.pth`, the `config.spk` key) and still derives from the
//!    run's model name exactly as before. Directory identity and artifact identity are
//!    separate on purpose; conflating them would rename every existing checkpoint.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};

use crate::{Result, UtaiError};

/// Set while the data-dir reclaim thread is copying into `<data>/training`. The startup
/// migration runs before that thread exists, but `resolve_or_create`'s on-demand retry can
/// fire at any moment — and renaming `<id>` out from under a copier that keeps
/// `create_dir_all`-ing it back produces a project root with a family slot AND loose legacy
/// checkpoints beside it: the "checkpoints without a manifest" shape the resume guards call
/// corruption, with the migration marker already gone.
pub static RECLAIM_TOUCHING_TRAINING: AtomicBool = AtomicBool::new(false);

/// Per-project descriptor, written to `<project>/project.json`. Self-describing: a project
/// directory copied to another machine (or recovered from a backup) is complete on its own.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProjectMeta {
    /// Directory name under `<data>/training`. Never derived at runtime — read from disk.
    pub id: String,
    /// User-visible name. For a migrated project this is the old `run.json` `model_name`
    /// (the only place the display name was ever persisted); when that file never existed —
    /// the manifest is written BEFORE the data import, `run.json` only after it — the id
    /// stands in, and the user renames it later.
    pub name: String,
    #[serde(default)]
    pub note: String,
    #[serde(default)]
    pub created_ms: u64,
    #[serde(default)]
    pub updated_ms: u64,
    /// Set when migration could not decide what this directory holds. The content is left
    /// exactly where it was — nothing is moved, nothing is deleted — and the reason travels
    /// to the UI so the user can act on it instead of watching a workspace vanish.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub needs_attention: Option<String>,
    /// Snapshots produced BEFORE this instant predate the export ledger and are therefore
    /// unconditionally protected from「清理未导入的快照」: for a migrated project we cannot
    /// know which checkpoints the user already imported, and guessing would delete the ones
    /// they kept on purpose. Stamped once, at migration/creation.
    #[serde(default)]
    pub export_ledger_since_ms: u64,
    /// Checkpoints this project exported into the model registry (filled from S76 batch 2 on;
    /// paths are RELATIVE to the project dir so a data-dir move cannot orphan the ledger).
    #[serde(default)]
    pub exported: Vec<ExportedModel>,
    /// Forward compatibility: a downgraded build must not silently drop fields a newer build
    /// wrote (that is how an export ledger disappears and a cleanup then deletes real work).
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedModel {
    pub name: String,
    pub model_type: String,
    /// Path of the source checkpoint RELATIVE to the project directory.
    pub from_ckpt_rel: String,
    pub at_ms: u64,
}

pub const PROJECT_META: &str = "project.json";
pub const DATASET_DIR: &str = "dataset";
/// Every family that owns a slot directory. `sovits_diff` is absent by design — it lives in
/// the `sovits` slot (see [`crate::training::backend_family`]).
pub const FAMILIES: [&str; 4] = ["rvc", "sovits", "sovits_v2", "vocoder"];
/// Marker for an in-flight migration: `<training>/.migrating_<id>.json`. A FILE at the
/// training root, so it survives every rename the migration performs.
const MARKER_PREFIX: &str = ".migrating_";
/// Staging name for the legacy tree while it is being folded into its family slot.
const STAGING_PREFIX: &str = ".mig_";

pub fn training_root(data_dir: &Path) -> PathBuf {
    data_dir.join("training")
}

pub fn project_dir(data_dir: &Path, id: &str) -> PathBuf {
    training_root(data_dir).join(id)
}

/// The slot directory for one architecture — the exact equivalent of the old workspace root,
/// which is why every path inside it (`weights/`, `dataset_44k/`, `G_*.pth`,
/// `run_manifest.json`, `dataset.fingerprint`, `audition/`) keeps its old relative shape.
pub fn family_dir(data_dir: &Path, id: &str, family: &str) -> PathBuf {
    project_dir(data_dir, id).join(family)
}

/// The one shared layer: raw imported audio, flat (single speaker) or one subdirectory per
/// co-trained speaker. Shared across every slot of the project — this is the whole point of
/// the refactor, and the reason no slot may ever delete it wholesale.
pub fn dataset_dir(data_dir: &Path, id: &str) -> PathBuf {
    project_dir(data_dir, id).join(DATASET_DIR)
}

/// Does the project hold imported audio? (Weakened on purpose from the pre-S76 predicate,
/// which also required `dataset.fingerprint` next to it: the fingerprint is per-family — the
/// three families write mutually incompatible formats into that one filename — so it now
/// lives one level down and can no longer corroborate the dataset. Non-empty `dataset/` is
/// sufficient: the import stage is the only writer.)
pub fn has_dataset(data_dir: &Path, id: &str) -> bool {
    std::fs::read_dir(dataset_dir(data_dir, id))
        .map(|mut d| d.next().is_some())
        .unwrap_or(false)
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// A fresh project id: `<≤24 ascii>_<8 hex>`.
///
/// Three constraints, all of them paid for in blood elsewhere in this repo:
/// * charset `[A-Za-z0-9_-]` — `storage.rs` refuses anything path-like as a delete target;
/// * a mandatory `_<hex>` suffix — it is what makes a Windows reserved device name
///   structurally impossible (`models/aux` cost the beta testers os error 267/1200, and the
///   dev machine was immune because Win11 Pro for Workstations allows it), and it is also
///   what keeps CJK names apart after the ASCII filter empties them;
/// * sha2, not `DefaultHasher` — std does not promise SipHash stability across releases, and
///   a directory name that changes with the toolchain is a data-loss bug on a timer.
pub fn new_project_id(name: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut base: String = name
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .take(24)
        .collect();
    if base.is_empty() {
        base = "project".to_string();
    }
    let digest = Sha256::digest(name.as_bytes());
    format!(
        "{}_{:02x}{:02x}{:02x}{:02x}",
        base, digest[0], digest[1], digest[2], digest[3]
    )
}

fn meta_path(data_dir: &Path, id: &str) -> PathBuf {
    project_dir(data_dir, id).join(PROJECT_META)
}

pub fn read_meta(data_dir: &Path, id: &str) -> Option<ProjectMeta> {
    let raw = std::fs::read_to_string(meta_path(data_dir, id)).ok()?;
    let mut m: ProjectMeta = serde_json::from_str(&raw).ok()?;
    // The directory the user can rename by hand wins over the field: `id` is a convenience
    // copy, the path is the identity.
    m.id = id.to_string();
    Some(m)
}

/// Atomic write (tmp + rename in the same directory) — a torn `project.json` would read as
/// "not migrated" and send the next boot into a second migration of an already-migrated tree.
pub fn write_meta(data_dir: &Path, meta: &ProjectMeta) -> Result<()> {
    let dir = project_dir(data_dir, &meta.id);
    // Every step gets the SAME code: its trilingual text tells the user to check write
    // permission on the data directory, and a read-only / full / ACL-restricted root fails at
    // create_dir_all or the write long before the rename. A bare `?` here would hand them
    // "IO error: Access is denied. (os error 5)" instead — unmapped, untranslated.
    let io_err = |e: std::io::Error| {
        UtaiError::Training(format!("PROJECT_META_WRITE_FAILED: {}: {e}", dir.display()))
    };
    std::fs::create_dir_all(&dir).map_err(io_err)?;
    let final_path = dir.join(PROJECT_META);
    let tmp = dir.join(format!("{PROJECT_META}.tmp"));
    let body = serde_json::to_string_pretty(meta)
        .map_err(|e| UtaiError::Training(format!("PROJECT_META_ENCODE_FAILED: {e}")))?;
    std::fs::write(&tmp, body).map_err(io_err)?;
    std::fs::rename(&tmp, &final_path)
        .map_err(|e| UtaiError::Training(format!("PROJECT_META_WRITE_FAILED: {e}")))?;
    Ok(())
}

/// Every project on disk, in no particular order. Scans — there is no index to go stale, and
/// a project directory restored from a backup shows up without any bookkeeping.
pub fn list_projects(data_dir: &Path) -> Vec<ProjectMeta> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(training_root(data_dir)) else {
        return out;
    };
    for e in rd.flatten() {
        if !e.path().is_dir() {
            continue;
        }
        let id = e.file_name().to_string_lossy().into_owned();
        if id.starts_with('.') {
            continue;
        }
        match read_meta(data_dir, &id) {
            Some(m) => out.push(m),
            // A project.json that EXISTS but will not parse (truncated by a crash, mangled by
            // a sync client, hand-edited) must not read as「没有这个项目」: that is fail-OPEN,
            // and the next start would mint a second id beside it and orphan everything here.
            // Surface it as a project that needs attention — visible, and gated by
            // resolve_or_create.
            None if project_dir(data_dir, &id).join(PROJECT_META).is_file() => {
                out.push(ProjectMeta {
                    id: id.clone(),
                    name: id,
                    needs_attention: Some("PROJECT_META_UNREADABLE".into()),
                    ..Default::default()
                });
            }
            None => {}
        }
    }
    out
}

/// Name → project, for the pre-batch-4 bridge (the wizard still asks for a model name).
///
/// The `id == slugify(name)` fallback is what keeps a migrated workspace reachable when its
/// display name could not be recovered: `run_manifest.json` is written before the data import
/// and `run.json` only after it, so a workspace whose first run was interrupted mid-import
/// has no persisted display name at all. Its id is still the old slug, so the old mapping
/// resolves it exactly as it always did.
pub fn find_by_name(data_dir: &Path, name: &str) -> Option<ProjectMeta> {
    let projects = list_projects(data_dir);
    if let Some(m) = projects.iter().find(|m| m.name == name) {
        return Some(m.clone());
    }
    let legacy = crate::training::slugify(name);
    projects.into_iter().find(|m| m.id == legacy)
}

/// A pre-S76 workspace directory for this name that has NOT been migrated yet: it exists, and
/// it has no `project.json`.
///
/// This state is reachable on every boot — the startup pass stands down entirely when a
/// sibling instance is alive (double-launch is a supported reality here), and an individual
/// directory can fail its renames because Explorer or a backup agent holds a handle. It must
/// never be confused with「这个名字没有工作区」, which is what a plain `find_by_name` miss
/// looks like.
pub fn unmigrated_legacy_dir(data_dir: &Path, name: &str) -> Option<PathBuf> {
    let d = project_dir(data_dir, &crate::training::slugify(name));
    (d.is_dir() && !d.join(PROJECT_META).is_file()).then_some(d)
}

/// Resolve a project by name, creating it when absent. Used by the bridge above; project
/// creation becomes an explicit user action in batch 4.
pub fn resolve_or_create(data_dir: &Path, name: &str) -> Result<ProjectMeta> {
    if let Some(m) = find_by_name(data_dir, name) {
        if let Some(reason) = m.needs_attention.clone() {
            // Its content was left wherever it was — including, possibly, at the project root
            // where a `dataset/` of ours would land on top of it. Training into such a project
            // is the one thing that could still destroy what migration deliberately preserved.
            return Err(UtaiError::Training(format!(
                "PROJECT_NEEDS_ATTENTION: {reason}"
            )));
        }
        return Ok(m);
    }
    // Minting a new id here while an unmigrated workspace for the same name sits on disk would
    // fork the user's work in two: the old checkpoints keep existing but nothing can ever
    // reach them again, and once the migration DOES succeed there are two projects with the
    // same display name and `find_by_name` picks between them by directory order.
    if unmigrated_legacy_dir(data_dir, name).is_some() {
        if RECLAIM_TOUCHING_TRAINING.load(Ordering::SeqCst) {
            return Err(UtaiError::Training(
                "TRAINING_LAYOUT_MIGRATION_PENDING: data-dir reclaim in progress".into(),
            ));
        }
        if crate::crashlog::other_instance_alive() {
            return Err(UtaiError::Training(
                "TRAINING_LAYOUT_MIGRATION_PENDING: another instance".into(),
            ));
        }
        // Retry it right now: the usual cause is a transient handle that is long gone.
        let id = crate::training::slugify(name);
        match migrate_one(data_dir, &id) {
            Ok(_) => {
                if let Some(m) = find_by_name(data_dir, name) {
                    return match m.needs_attention.clone() {
                        Some(reason) => Err(UtaiError::Training(format!(
                            "PROJECT_NEEDS_ATTENTION: {reason}"
                        ))),
                        None => Ok(m),
                    };
                }
            }
            Err(e) => tracing::error!("on-demand layout migration for {id} failed: {e}"),
        }
        return Err(UtaiError::Training(
            "TRAINING_LAYOUT_MIGRATION_PENDING: retry failed".into(),
        ));
    }
    let now = now_ms();
    let id = new_project_id(name);
    // `new_project_id` is a deterministic function of the name, so an existing directory here
    // is almost certainly THIS project with an unusable `project.json` (truncated by a crash,
    // deleted by a sync client, locked) — not a hash collision. Minting `<id>_2` beside it
    // would start a second project from scratch and leave the real one's checkpoints reachable
    // only from the storage page, with its display name gone for good. Refuse instead.
    if project_dir(data_dir, &id).exists() {
        return Err(UtaiError::Training("PROJECT_META_UNREADABLE".into()));
    }
    let meta = ProjectMeta {
        id,
        name: name.to_string(),
        created_ms: now,
        updated_ms: now,
        export_ledger_since_ms: now,
        ..Default::default()
    };
    write_meta(data_dir, &meta)?;
    Ok(meta)
}

// ─────────────────────────── legacy layout migration ───────────────────────────

/// Which family does a legacy workspace hold?
///
/// The manifest is authoritative and effectively always present (every run since S37 writes
/// it before spawning). The file-shape heuristic below only covers manifest-less anomalies,
/// and only when exactly ONE family's signature is present — an ambiguous directory is left
/// untouched and flagged, because moving the wrong subtree is unrecoverable while a flagged
/// directory is merely inconvenient.
fn detect_family(ws: &Path) -> std::result::Result<String, String> {
    if let Some(fam) = std::fs::read_to_string(ws.join("run_manifest.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| v["backend"].as_str().map(String::from))
        .filter(|s| !s.is_empty())
    {
        return Ok(crate::training::backend_family(&fam).to_string());
    }
    let has = |rel: &str| ws.join(rel).exists();
    let mut hits: Vec<&str> = Vec::new();
    if has("0_gt_wavs") || has("1_16k_wavs") {
        hits.push("rvc");
    }
    if ws.join("diffusion").is_dir() {
        // shallow diffusion only ever lives in a sovits workspace (4.0-v2 has none)
        hits.push("sovits");
    } else if has("dataset_44k") {
        // ⚠ `dataset_44k` is written by BOTH sovits and sovits_v2
        // (`utai_train/sovits/pipeline.py` and `sovits_v2/pipeline.py` build the identical
        // path), so it cannot decide between them. Registering both is what makes this
        // AMBIGUOUS instead of a confident wrong answer: guessing "sovits" would fold a v2
        // workspace into the sovits slot, where v2 can never find it again and a later
        // sovits 重训 would wipe it.
        hits.push("sovits");
        hits.push("sovits_v2");
    }
    if has("slices") || has("npz") {
        hits.push("vocoder");
    }
    match hits.len() {
        1 => Ok(hits[0].to_string()),
        0 => Err("MIGRATE_FAMILY_UNKNOWN".into()),
        _ => Err("MIGRATE_FAMILY_AMBIGUOUS".into()),
    }
}

/// Does this directory already have the new shape? Used INSTEAD of a phase field in the
/// marker, because the two shapes are structurally distinguishable: a legacy workspace can
/// never contain a subdirectory named exactly after a family.
///
/// That claim is verified, not assumed — the complete set of subdirectories anything has ever
/// created inside a workspace root is
/// `0_gt_wavs 1_16k_wavs 2a_f0 2b-f0nsf 3_feature{256,768} aug_meta cluster dataset_44k
/// diffusion filelists mute npz slices weights` (every `os.path.join(exp_dir, …)` in
/// `training/utai_train/`) plus `dataset audition` from the Rust side. None collides.
/// ⚠ Adding a workspace subdirectory named after a family would silently break migration
/// recovery — grep this comment before you do.
fn has_family_slot(dir: &Path) -> bool {
    FAMILIES.iter().any(|f| dir.join(f).is_dir())
}

fn marker_path(data_dir: &Path, id: &str) -> PathBuf {
    training_root(data_dir).join(format!("{MARKER_PREFIX}{id}.json"))
}

fn staging_path(data_dir: &Path, id: &str) -> PathBuf {
    training_root(data_dir).join(format!("{STAGING_PREFIX}{id}"))
}

#[derive(Debug, Default)]
pub struct MigrationReport {
    pub migrated: Vec<String>,
    pub flagged: Vec<String>,
    pub failed: Vec<String>,
}

/// Fold every legacy workspace under `<data>/training` into the project layout.
///
/// Called once at startup, BEFORE anything can hold a handle into these trees and before the
/// data-dir reclaim thread starts. Never fails the boot: an unmigratable directory is either
/// flagged (visible, actionable) or retried on the next launch.
///
/// Double-launch is a supported reality in this app (there is no single-instance guard — see
/// `crashlog::other_instance_alive`, which the data-dir reclaim already consults). Two
/// instances racing over the same renames would produce exactly the half-in-half-out tree the
/// resume guards treat as corrupt, so the whole pass stands down when a sibling is alive.
pub fn migrate_legacy_layout(data_dir: &Path) -> MigrationReport {
    let mut report = MigrationReport::default();
    let root = training_root(data_dir);
    if !root.is_dir() {
        return report;
    }
    if crate::crashlog::other_instance_alive() {
        tracing::warn!("training layout migration postponed: another live instance detected");
        return report;
    }

    // Torn runs first: a leftover staging directory must be reconciled before the same id is
    // considered again (its legacy path is currently empty or missing).
    let mut ids: Vec<String> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&root) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if let Some(id) = name.strip_prefix(STAGING_PREFIX) {
                ids.push(id.to_string());
            } else if let Some(rest) = name.strip_prefix(MARKER_PREFIX) {
                if let Some(id) = rest.strip_suffix(".json") {
                    ids.push(id.to_string());
                }
            } else if e.path().is_dir() && !name.starts_with('.') {
                ids.push(name);
            }
        }
    }
    ids.sort();
    ids.dedup();

    sweep_orphan_dataset_copies(data_dir);
    for id in ids {
        match migrate_one(data_dir, &id) {
            Ok(Outcome::AlreadyDone) => {}
            Ok(Outcome::Migrated) => report.migrated.push(id),
            Ok(Outcome::Flagged(reason)) => {
                tracing::warn!("training project {id} needs attention: {reason}");
                report.flagged.push(id);
            }
            Err(e) => {
                tracing::error!("training layout migration failed for {id}: {e}");
                report.failed.push(id);
            }
        }
    }
    if !report.migrated.is_empty() || !report.flagged.is_empty() || !report.failed.is_empty() {
        tracing::info!(
            "training layout migration: {} migrated, {} flagged, {} failed (will retry next boot)",
            report.migrated.len(),
            report.flagged.len(),
            report.failed.len()
        );
    }
    report
}

/// Reclaim `<project>/.dataset.old_<pid>` copies left by a HARD kill (task manager, power
/// loss) — `DatasetSwap`'s Drop covers every graceful path but never runs for those. The copy
/// is real user data, so it is only deleted once `dataset/` is present and non-empty;
/// otherwise it is put back, because it is then the only copy that exists.
fn sweep_orphan_dataset_copies(data_dir: &Path) {
    let Ok(rd) = std::fs::read_dir(training_root(data_dir)) else { return };
    for e in rd.flatten() {
        let proj = e.path();
        if !proj.is_dir() || e.file_name().to_string_lossy().starts_with('.') {
            continue;
        }
        let Ok(inner) = std::fs::read_dir(&proj) else { continue };
        for c in inner.flatten() {
            if !c.file_name().to_string_lossy().starts_with(".dataset.old_") {
                continue;
            }
            let live = proj.join(DATASET_DIR);
            let live_ok = std::fs::read_dir(&live).map(|mut d| d.next().is_some()).unwrap_or(false);
            if live_ok {
                tracing::warn!("reclaiming orphaned dataset copy {}", c.path().display());
                let _ = crate::util::remove_dir_all_robust(&c.path());
            } else {
                tracing::warn!(
                    "restoring dataset from an interrupted import: {}",
                    c.path().display()
                );
                let _ = std::fs::remove_dir(&live);
                let _ = crate::util::rename_with_retry(&c.path(), &live, "TRAINING_DATASET_RECOVER");
            }
        }
    }
}

pub enum Outcome {
    AlreadyDone,
    Migrated,
    Flagged(String),
}

fn migrate_one(data_dir: &Path, id: &str) -> Result<Outcome> {
    let dir = project_dir(data_dir, id);
    let staging = staging_path(data_dir, id);
    let marker = marker_path(data_dir, id);

    // Already committed: project.json is the commit point. Clear any marker left behind by a
    // crash between the commit and the cleanup.
    if meta_path(data_dir, id).is_file() {
        let _ = std::fs::remove_file(&marker);
        // …but a flag must be CLEARABLE, or「请到设置里处理」is advice with no action behind
        // it: the only button such a project has is Delete, which destroys exactly what the
        // flag was protecting. If the user has since arranged the content into family slots
        // themselves, re-evaluate and let the project back in.
        if let Some(mut m) = read_meta(data_dir, id) {
            if m.needs_attention.is_some() && has_family_slot(&dir) {
                tracing::info!("training project {id}: arranged by hand — clearing the flag");
                m.needs_attention = None;
                m.updated_ms = now_ms();
                write_meta(data_dir, &m)?;
                return Ok(Outcome::Migrated);
            }
        }
        return Ok(Outcome::AlreadyDone);
    }

    // Reconcile a torn attempt. The migration is three renames, so the observable states are
    // few and each maps to one mirror-image undo.
    if staging.exists() {
        if dir.exists() && has_family_slot(&dir) {
            // The final rename landed; only the commit is missing. Roll FORWARD — rolling
            // back here would have to un-nest a slot we can no longer tell apart from a
            // legitimately migrated one.
            tracing::warn!("training project {id}: completing an interrupted migration");
        } else {
            roll_back(data_dir, id)?;
        }
    }

    if dir.join(PROJECT_META).is_file() {
        let _ = std::fs::remove_file(&marker);
        return Ok(Outcome::AlreadyDone);
    }
    if !dir.exists() {
        // Nothing here (a bare marker from a rolled-back attempt).
        let _ = std::fs::remove_file(&marker);
        return Ok(Outcome::AlreadyDone);
    }

    // The display name only ever lived in run.json. Look at the workspace root (legacy shape)
    // AND inside each family slot (a tree whose final rename landed before the commit) —
    // falling back to the id would silently rename the user's project to a hash.
    let name = std::iter::once(dir.join("run.json"))
        .chain(FAMILIES.iter().map(|f| dir.join(f).join("run.json")))
        .find_map(|p| {
            std::fs::read_to_string(p)
                .ok()
                .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                .and_then(|v| v["model_name"].as_str().map(String::from))
                .filter(|s| !s.is_empty())
        })
        .unwrap_or_else(|| id.to_string());
    let now = now_ms();
    let created = std::fs::metadata(&dir)
        .and_then(|m| m.created().or_else(|_| m.modified()))
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as u64)
        .unwrap_or(now);
    let mut meta = ProjectMeta {
        id: id.to_string(),
        name,
        created_ms: created,
        updated_ms: now,
        // Everything that already exists predates the ledger and is protected forever.
        export_ledger_since_ms: now,
        ..Default::default()
    };

    // Already the new shape but uncommitted (the interrupted-final-rename case above, or a
    // tree copied in by the data-dir reclaim after a partial migration elsewhere).
    if has_family_slot(&dir) {
        write_meta(data_dir, &meta)?;
        let _ = std::fs::remove_file(&marker);
        return Ok(Outcome::Migrated);
    }

    // An empty leftover (`try_start` creates the directory before it can fail) needs no
    // moving at all — and must not be flagged, or every abandoned name becomes a chore.
    let empty_shell = !crate::training::workspace_holds_work(&dir)
        && !dir.join("run_manifest.json").is_file()
        && !dir.join("config.json").is_file();
    if empty_shell {
        write_meta(data_dir, &meta)?;
        let _ = std::fs::remove_file(&marker);
        return Ok(Outcome::Migrated);
    }

    let family = match detect_family(&dir) {
        Ok(f) => f,
        Err(reason) => {
            // Leave every byte where it is; record WHY, so the UI can offer the user a choice
            // instead of showing an empty project.
            meta.needs_attention = Some(reason.clone());
            write_meta(data_dir, &meta)?;
            let _ = std::fs::remove_file(&marker);
            return Ok(Outcome::Flagged(reason));
        }
    };

    std::fs::write(
        &marker,
        serde_json::json!({ "id": id, "family": family, "pid": std::process::id(), "at_ms": now })
            .to_string(),
    )?;

    // Three renames, whatever the workspace weighs:
    //   1. legacy tree out of the way        <id>            -> .mig_<id>
    //   2. the project directory takes its place
    //   3. the shared dataset moves up       .mig_<id>/dataset -> <id>/dataset
    //   4. what remains IS the family slot   .mig_<id>       -> <id>/<family>
    let step = || -> Result<()> {
        crate::util::rename_with_retry(&dir, &staging, "TRAINING_MIGRATE_STAGE")
            .map_err(UtaiError::Training)?;
        std::fs::create_dir_all(&dir)?;
        let staged_dataset = staging.join(DATASET_DIR);
        if staged_dataset.exists() {
            crate::util::rename_with_retry(
                &staged_dataset,
                &dir.join(DATASET_DIR),
                "TRAINING_MIGRATE_DATASET",
            )
            .map_err(UtaiError::Training)?;
        }
        crate::util::rename_with_retry(&staging, &dir.join(&family), "TRAINING_MIGRATE_SLOT")
            .map_err(UtaiError::Training)?;
        Ok(())
    };
    if let Err(e) = step() {
        // Best effort back to the exact pre-migration shape. If even THAT fails we keep the
        // marker: the next boot retries, and until then nothing has been deleted.
        match roll_back(data_dir, id) {
            Ok(()) => {
                let _ = std::fs::remove_file(&marker);
            }
            Err(re) => tracing::error!("training project {id}: rollback also failed: {re}"),
        }
        return Err(e);
    }

    write_meta(data_dir, &meta)?;
    let _ = std::fs::remove_file(&marker);
    Ok(Outcome::Migrated)
}

/// Undo whatever of the three renames happened, in reverse. Idempotent: every step checks the
/// shape it is about to undo, so calling it on an untouched tree is a no-op.
fn roll_back(data_dir: &Path, id: &str) -> Result<()> {
    let dir = project_dir(data_dir, id);
    let staging = staging_path(data_dir, id);
    if !staging.exists() {
        return Ok(());
    }
    tracing::warn!("training project {id}: rolling back a torn migration");
    let moved_dataset = dir.join(DATASET_DIR);
    if moved_dataset.exists() && !staging.join(DATASET_DIR).exists() {
        crate::util::rename_with_retry(
            &moved_dataset,
            &staging.join(DATASET_DIR),
            "TRAINING_MIGRATE_UNDO_DATASET",
        )
        .map_err(UtaiError::Training)?;
    }
    if dir.exists() {
        // Only ever the empty shell created by step 2; a non-empty one means someone else
        // wrote here and we must not touch it.
        std::fs::remove_dir(&dir).map_err(|e| {
            UtaiError::Training(format!("TRAINING_MIGRATE_UNDO_BLOCKED: {} ({e})", dir.display()))
        })?;
    }
    crate::util::rename_with_retry(&staging, &dir, "TRAINING_MIGRATE_UNDO_STAGE")
        .map_err(UtaiError::Training)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_root(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("utai_tproj_{}_{}", tag, uuid::Uuid::new_v4()));
        std::fs::create_dir_all(d.join("training")).unwrap();
        d
    }

    fn legacy_rvc(data: &Path, slug: &str) -> PathBuf {
        let ws = training_root(data).join(slug);
        std::fs::create_dir_all(ws.join("dataset")).unwrap();
        std::fs::create_dir_all(ws.join("weights")).unwrap();
        std::fs::create_dir_all(ws.join("0_gt_wavs")).unwrap();
        std::fs::write(ws.join("dataset").join("000.wav"), b"a").unwrap();
        std::fs::write(ws.join("G_2333333.pth"), b"g").unwrap();
        std::fs::write(ws.join("D_2333333.pth"), b"d").unwrap();
        std::fs::write(ws.join("weights").join("m.pth"), b"w").unwrap();
        std::fs::write(ws.join("dataset.fingerprint"), b"fp").unwrap();
        std::fs::write(ws.join("run_manifest.json"), br#"{"backend":"rvc"}"#).unwrap();
        std::fs::write(ws.join("run.json"), r#"{"model_name":"歌姫テスト"}"#).unwrap();
        ws
    }

    #[test]
    fn migrate_folds_into_family_slot_and_lifts_dataset() {
        let data = tmp_root("basic");
        legacy_rvc(&data, "test_1a2b3c4d");
        let rep = migrate_legacy_layout(&data);
        assert_eq!(rep.migrated, vec!["test_1a2b3c4d".to_string()]);

        let p = project_dir(&data, "test_1a2b3c4d");
        // dataset lifted to the project (the shared layer), everything else folded into rvc/
        assert!(p.join("dataset").join("000.wav").is_file());
        assert!(p.join("rvc").join("G_2333333.pth").is_file());
        assert!(p.join("rvc").join("weights").join("m.pth").is_file());
        assert!(p.join("rvc").join("run_manifest.json").is_file());
        assert!(p.join("rvc").join("dataset.fingerprint").is_file());
        assert!(!p.join("rvc").join("dataset").exists(), "dataset must not be duplicated");
        assert!(!p.join("G_2333333.pth").exists(), "nothing may stay at the project root");

        // display name recovered from run.json; ledger stamped so legacy snapshots stay safe
        let m = read_meta(&data, "test_1a2b3c4d").unwrap();
        assert_eq!(m.name, "歌姫テスト");
        assert!(m.export_ledger_since_ms > 0);
        assert!(m.needs_attention.is_none());

        // idempotent
        let rep2 = migrate_legacy_layout(&data);
        assert!(rep2.migrated.is_empty() && rep2.flagged.is_empty() && rep2.failed.is_empty());
        assert!(p.join("rvc").join("G_2333333.pth").is_file());
        assert!(!p.join("rvc").join("rvc").exists(), "must never nest a second slot");

        let _ = std::fs::remove_dir_all(data);
    }

    #[test]
    fn migrate_flags_undecidable_workspace_without_moving_anything() {
        let data = tmp_root("flag");
        let ws = training_root(&data).join("weird_00000000");
        std::fs::create_dir_all(&ws).unwrap();
        // holds work (so it is not an empty shell) but nothing says which family
        std::fs::write(ws.join("G_100.pth"), b"g").unwrap();
        let rep = migrate_legacy_layout(&data);
        assert_eq!(rep.flagged, vec!["weird_00000000".to_string()]);
        assert!(ws.join("G_100.pth").is_file(), "content must stay put");
        let m = read_meta(&data, "weird_00000000").unwrap();
        assert_eq!(m.needs_attention.as_deref(), Some("MIGRATE_FAMILY_UNKNOWN"));
        let _ = std::fs::remove_dir_all(data);
    }

    #[test]
    fn migrate_adopts_empty_shell_without_flagging() {
        let data = tmp_root("shell");
        std::fs::create_dir_all(training_root(&data).join("ghost_deadbeef")).unwrap();
        let rep = migrate_legacy_layout(&data);
        assert_eq!(rep.migrated, vec!["ghost_deadbeef".to_string()]);
        assert!(read_meta(&data, "ghost_deadbeef").unwrap().needs_attention.is_none());
        let _ = std::fs::remove_dir_all(data);
    }

    /// A torn attempt (staging left behind, no commit) must return the tree to EXACTLY the
    /// pre-migration shape, then migrate cleanly on the retry.
    #[test]
    fn rollback_restores_pre_migration_shape() {
        let data = tmp_root("torn");
        let ws = legacy_rvc(&data, "torn_11223344");
        let before: Vec<String> = {
            let mut v: Vec<String> = std::fs::read_dir(&ws)
                .unwrap()
                .flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect();
            v.sort();
            v
        };
        // simulate a crash right after step 1
        std::fs::rename(&ws, staging_path(&data, "torn_11223344")).unwrap();
        std::fs::write(marker_path(&data, "torn_11223344"), b"{}").unwrap();

        roll_back(&data, "torn_11223344").unwrap();
        let after: Vec<String> = {
            let mut v: Vec<String> = std::fs::read_dir(&ws)
                .unwrap()
                .flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect();
            v.sort();
            v
        };
        assert_eq!(before, after);
        assert!(!staging_path(&data, "torn_11223344").exists());

        // and the retry still works
        let rep = migrate_legacy_layout(&data);
        assert_eq!(rep.migrated, vec!["torn_11223344".to_string()]);
        let _ = std::fs::remove_dir_all(data);
    }

    /// Crash between the last rename and the commit: the tree already has the new shape, so
    /// the retry must roll FORWARD (write project.json), never re-migrate into a nested slot.
    #[test]
    fn interrupted_after_final_rename_rolls_forward() {
        let data = tmp_root("fwd");
        let ws = legacy_rvc(&data, "fwd_55667788");
        let staging = staging_path(&data, "fwd_55667788");
        std::fs::rename(&ws, &staging).unwrap();
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::rename(staging.join("dataset"), ws.join("dataset")).unwrap();
        std::fs::rename(&staging, ws.join("rvc")).unwrap();
        std::fs::write(marker_path(&data, "fwd_55667788"), b"{}").unwrap();

        let rep = migrate_legacy_layout(&data);
        assert_eq!(rep.migrated, vec!["fwd_55667788".to_string()]);
        assert!(ws.join("rvc").join("G_2333333.pth").is_file());
        assert!(!ws.join("rvc").join("rvc").exists());
        assert!(!marker_path(&data, "fwd_55667788").exists());
        let _ = std::fs::remove_dir_all(data);
    }

    #[test]
    fn new_project_id_is_stable_charset_safe_and_dodges_reserved_names() {
        for name in ["aux", "CON", "nul", "com1", "LPT9", "prn"] {
            let id = new_project_id(name);
            assert!(
                id.len() > name.len() + 1 && id.contains('_'),
                "reserved device name {name} must gain a suffix: {id}"
            );
            let stem = id.split('_').next().unwrap().to_ascii_lowercase();
            assert_ne!(stem, id.to_ascii_lowercase());
        }
        // CJK filters down to nothing → the constant base plus a distinguishing digest
        assert_ne!(new_project_id("歌姫"), new_project_id("初音"));
        assert!(new_project_id("歌姫").starts_with("project_"));
        // charset the storage delete guard demands
        for name in ["a b/c\\d", "..", "歌 姫 2024!"] {
            assert!(
                new_project_id(name)
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
                "{name}"
            );
        }
        // stable across runs (sha2, not DefaultHasher)
        assert_eq!(new_project_id("test"), new_project_id("test"));
    }

    /// sovits and sovits_v2 write the SAME `dataset_44k/`, so a manifest-less workspace with
    /// only that signature is undecidable. Guessing "sovits" would fold a v2 workspace into a
    /// slot v2 can never look in — and a later sovits 重训 would wipe it.
    #[test]
    fn manifest_less_dataset_44k_is_ambiguous_not_a_guess() {
        let data = tmp_root("ambig");
        let ws = training_root(&data).join("v2ish_00001111");
        std::fs::create_dir_all(ws.join("dataset_44k")).unwrap();
        std::fs::write(ws.join("G_800.pth"), b"g").unwrap();
        let rep = migrate_legacy_layout(&data);
        assert_eq!(rep.flagged, vec!["v2ish_00001111".to_string()]);
        assert_eq!(
            read_meta(&data, "v2ish_00001111").unwrap().needs_attention.as_deref(),
            Some("MIGRATE_FAMILY_AMBIGUOUS")
        );
        assert!(ws.join("G_800.pth").is_file(), "content must stay put");
        // …but a diffusion dir IS sovits-only, so that one still decides
        let ws2 = training_root(&data).join("diffish_22223333");
        std::fs::create_dir_all(ws2.join("diffusion")).unwrap();
        std::fs::create_dir_all(ws2.join("dataset_44k")).unwrap();
        std::fs::write(ws2.join("G_800.pth"), b"g").unwrap();
        assert!(migrate_legacy_layout(&data).migrated.contains(&"diffish_22223333".to_string()));
        assert!(ws2.join("sovits").join("G_800.pth").is_file());
        let _ = std::fs::remove_dir_all(data);
    }

    /// The startup pass stands down whenever a sibling instance is alive, and any single
    /// directory can fail its renames. Minting a fresh id in that state would fork the user's
    /// work: the old checkpoints survive but nothing can reach them, and once migration does
    /// succeed there are two projects with the same display name.
    #[test]
    fn an_unmigrated_workspace_blocks_a_new_project_of_the_same_name() {
        let data = tmp_root("pending");
        let ws = legacy_rvc(&data, &crate::training::slugify("歌姫テスト"));
        assert!(unmigrated_legacy_dir(&data, "歌姫テスト").is_some());
        // resolve_or_create migrates it on demand rather than forking
        let m = resolve_or_create(&data, "歌姫テスト").unwrap();
        assert_eq!(m.id, crate::training::slugify("歌姫テスト"));
        assert!(ws.join("rvc").join("G_2333333.pth").is_file());
        assert_eq!(list_projects(&data).len(), 1, "must never mint a second project");
        let _ = std::fs::remove_dir_all(data);
    }

    /// A project.json that exists but will not parse must read as「需人工处理」, never as
    /// 「没有这个项目」 — the latter is fail-open and mints a second id beside the real one.
    #[test]
    fn unreadable_meta_is_surfaced_and_blocks_training() {
        let data = tmp_root("badmeta");
        let dir = project_dir(&data, "broken_44445555");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(PROJECT_META), b"{ truncated").unwrap();
        let listed = list_projects(&data);
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].needs_attention.as_deref(), Some("PROJECT_META_UNREADABLE"));
        let err = resolve_or_create(&data, "broken_44445555").unwrap_err().to_string();
        assert!(err.contains("PROJECT_NEEDS_ATTENTION"), "{err}");
        assert_eq!(list_projects(&data).len(), 1, "must not create a sibling project");
        let _ = std::fs::remove_dir_all(data);
    }

    /// `new_project_id` is deterministic, so「目录已存在」means「就是这个项目」— never a
    /// collision. Minting `<id>_2` there starts a second project from scratch and leaves the
    /// real one reachable only from the storage page, its display name gone.
    #[test]
    fn a_broken_meta_never_forks_a_second_project() {
        let data = tmp_root("nofork");
        let m = resolve_or_create(&data, "歌姫").unwrap();
        std::fs::create_dir_all(project_dir(&data, &m.id).join("rvc")).unwrap();
        std::fs::write(project_dir(&data, &m.id).join("rvc").join("G_9.pth"), b"g").unwrap();
        // the metadata is lost (crash / sync client / restore)
        std::fs::write(meta_path(&data, &m.id), b"{ truncated").unwrap();
        let err = resolve_or_create(&data, "歌姫").unwrap_err().to_string();
        assert!(err.contains("PROJECT_META_UNREADABLE") || err.contains("PROJECT_NEEDS_ATTENTION"), "{err}");
        assert_eq!(
            std::fs::read_dir(training_root(&data)).unwrap().flatten().count(),
            1,
            "must not create a sibling project directory"
        );
        // …and once the file is gone entirely, the same refusal (not a fresh mint)
        std::fs::remove_file(meta_path(&data, &m.id)).unwrap();
        assert!(resolve_or_create(&data, "歌姫").is_err());
        let _ = std::fs::remove_dir_all(data);
    }

    /// A flag the user cannot clear is not a flag, it is a dead end: the only other button
    /// such a project has is Delete, which destroys what the flag was protecting.
    #[test]
    fn needs_attention_clears_once_the_user_arranges_the_slots() {
        let data = tmp_root("unflag");
        let ws = training_root(&data).join("weird_00000000");
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::write(ws.join("G_100.pth"), b"g").unwrap();
        assert_eq!(migrate_legacy_layout(&data).flagged, vec!["weird_00000000".to_string()]);
        assert!(resolve_or_create(&data, "weird_00000000").is_err());

        // the user moves the content into the slot themselves and restarts
        std::fs::create_dir_all(ws.join("rvc")).unwrap();
        std::fs::rename(ws.join("G_100.pth"), ws.join("rvc").join("G_100.pth")).unwrap();
        migrate_legacy_layout(&data);
        assert!(read_meta(&data, "weird_00000000").unwrap().needs_attention.is_none());
        assert!(resolve_or_create(&data, "weird_00000000").is_ok());
        let _ = std::fs::remove_dir_all(data);
    }

    /// A hard kill (task manager, power loss) skips every Drop. The aside copy is then the
    /// user's only dataset if the import had already emptied the live one.
    #[test]
    fn orphaned_dataset_copies_are_reclaimed_or_restored() {
        let data = tmp_root("orphan");
        // case 1: live dataset intact → the orphan is redundant
        let p1 = project_dir(&data, "keep_11110000");
        std::fs::create_dir_all(p1.join("dataset")).unwrap();
        std::fs::create_dir_all(p1.join(".dataset.old_999")).unwrap();
        std::fs::write(p1.join("dataset").join("000.wav"), b"live").unwrap();
        std::fs::write(p1.join(".dataset.old_999").join("000.wav"), b"old").unwrap();
        write_meta(&data, &ProjectMeta { id: "keep_11110000".into(), name: "k".into(), ..Default::default() }).unwrap();
        // case 2: import was killed after the swap → the orphan is the ONLY copy
        let p2 = project_dir(&data, "lost_22220000");
        std::fs::create_dir_all(p2.join(".dataset.old_999")).unwrap();
        std::fs::write(p2.join(".dataset.old_999").join("000.wav"), b"only").unwrap();
        write_meta(&data, &ProjectMeta { id: "lost_22220000".into(), name: "l".into(), ..Default::default() }).unwrap();

        migrate_legacy_layout(&data);

        assert!(!p1.join(".dataset.old_999").exists(), "redundant copy reclaimed");
        assert_eq!(std::fs::read(p1.join("dataset").join("000.wav")).unwrap(), b"live");
        assert!(!p2.join(".dataset.old_999").exists());
        assert_eq!(
            std::fs::read(p2.join("dataset").join("000.wav")).unwrap(),
            b"only",
            "the only copy must be put back, never deleted"
        );
        let _ = std::fs::remove_dir_all(data);
    }

    #[test]
    fn find_by_name_falls_back_to_the_legacy_slug() {
        let data = tmp_root("byname");
        // a migrated workspace whose run.json never existed: name == id
        let id = "lost_9f9f9f9f";
        std::fs::create_dir_all(project_dir(&data, id)).unwrap();
        write_meta(
            &data,
            &ProjectMeta { id: id.into(), name: id.into(), ..Default::default() },
        )
        .unwrap();
        assert!(find_by_name(&data, "anything else").is_none());
        assert_eq!(find_by_name(&data, id).unwrap().id, id);
        let _ = std::fs::remove_dir_all(data);
    }
}
