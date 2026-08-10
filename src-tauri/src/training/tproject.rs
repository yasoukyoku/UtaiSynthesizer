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
//! 4. **`model_slug` is a different identity from the directory id.** It is the artifact identity
//!    (`hps.name`, `weights/<slug>*.pth`, the `config.spk` key). Directory identity and artifact
//!    identity are separate on purpose; conflating them would rename every existing checkpoint.
//!    ★§F2⒝ batch 2 step ④b: it is no longer derived from the display name on every start —
//!    it is FROZEN per run and read back from that run's `run.json` by [`run_artifact_slug`],
//!    so renaming a run moves no bytes. The minting rule itself still lives in `training::mod`.

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

// ─────────────────────────── explicit CRUD (batch 4) ───────────────────────────

/// A display name is user TEXT, never a path — the directory is [`new_project_id`]'s
/// ASCII+hash slug — so the only limits are the ones that keep it usable: non-empty after
/// trimming, no control characters (they corrupt every list they are rendered into and can
/// forge line breaks inside a confirm dialog), and short enough for a card to show.
const MAX_PROJECT_NAME_CHARS: usize = 80;

fn clean_project_name(name: &str) -> Result<String> {
    let n = name.trim();
    if n.is_empty() {
        return Err(UtaiError::Training("PROJECT_NAME_EMPTY".into()));
    }
    if n.chars().count() > MAX_PROJECT_NAME_CHARS {
        return Err(UtaiError::Training("PROJECT_NAME_TOO_LONG".into()));
    }
    if n.chars().any(char::is_control) {
        return Err(UtaiError::Training("PROJECT_NAME_INVALID".into()));
    }
    Ok(n.to_string())
}

/// Same note treatment: free text, but bounded and control-free for the same reasons.
const MAX_PROJECT_NOTE_CHARS: usize = 500;

fn clean_project_note(note: &str) -> Result<String> {
    let n = note.trim();
    if n.chars().count() > MAX_PROJECT_NOTE_CHARS {
        return Err(UtaiError::Training("PROJECT_NOTE_TOO_LONG".into()));
    }
    // newlines are legitimate in a note; other control characters are not
    if n.chars().any(|c| c.is_control() && c != '\n' && c != '\r') {
        return Err(UtaiError::Training("PROJECT_NAME_INVALID".into()));
    }
    Ok(n.to_string())
}

/// Create a project because the USER asked for one — as opposed to [`resolve_or_create`],
/// which mints one implicitly from a training run's model name.
///
/// Names must stay unique: `find_by_name` resolves a duplicate by directory order, and every
/// consumer that still speaks names (the diffusion host picker, the legacy start path) would
/// then address a project at random.
pub fn create_project(data_dir: &Path, name: &str, note: &str) -> Result<ProjectMeta> {
    let name = clean_project_name(name)?;
    let note = clean_project_note(note)?;
    if find_by_name(data_dir, &name).is_some() {
        return Err(UtaiError::Training("PROJECT_NAME_EXISTS".into()));
    }
    // A pre-S76 workspace answering to this name is NOT a free name: creating beside it forks
    // the user's work in two exactly as `resolve_or_create` describes. Migration retries every
    // boot, so this is a wait, not a dead end.
    if unmigrated_legacy_dir(data_dir, &name).is_some() {
        return Err(UtaiError::Training(
            "TRAINING_LAYOUT_MIGRATION_PENDING: unmigrated workspace with this name".into(),
        ));
    }
    let now = now_ms();
    // `new_project_id` is a pure function of the name, so the id it proposes can already be
    // taken — most plausibly by THIS project after a rename (create "A", rename to "B", create
    // "A" again). That case is safe to place beside; an occupied id whose `project.json` we
    // cannot read is NOT (it may be the user's damaged project, and a second directory would
    // make the first unreachable from the UI forever).
    let mut id = new_project_id(&name);
    let mut attempt = 1u32;
    while project_dir(data_dir, &id).exists() {
        if read_meta(data_dir, &id).is_none() {
            return Err(UtaiError::Training("PROJECT_META_UNREADABLE".into()));
        }
        attempt += 1;
        if attempt > 64 {
            return Err(UtaiError::Training("PROJECT_ID_EXHAUSTED".into()));
        }
        // Vary the HASH INPUT, never the shape: the id must keep its `<ascii>_<8 hex>` form so
        // it can never become a Windows reserved device name.
        id = new_project_id(&format!("{name}#{attempt}"));
    }
    let meta = ProjectMeta {
        id,
        name,
        note,
        created_ms: now,
        updated_ms: now,
        // Nothing exists yet, so there is nothing to protect — but stamping it keeps every
        // project on the same rule (`cleanup_snapshots` refuses an unstamped ledger outright).
        export_ledger_since_ms: now,
        ..Default::default()
    };
    write_meta(data_dir, &meta)?;
    Ok(meta)
}

/// Rename / re-annotate. The DIRECTORY never moves: id and display name are separate
/// identities on purpose (module docs, invariant 3), so a rename cannot orphan a checkpoint,
/// break a resume, or invalidate the export ledger's relative paths.
///
/// It also cannot change any artifact's file name — those carry `slugify(本次训练名)`, which
/// lives in the slot's own `run.json` (see [`slot_model_name`]) and is frozen there.
pub fn update_project(data_dir: &Path, id: &str, name: &str, note: &str) -> Result<ProjectMeta> {
    let name = clean_project_name(name)?;
    let note = clean_project_note(note)?;
    let Some(mut meta) = read_meta(data_dir, id) else {
        return Err(UtaiError::Training("PROJECT_META_UNREADABLE".into()));
    };
    if let Some(other) = find_by_name(data_dir, &name) {
        if other.id != id {
            return Err(UtaiError::Training("PROJECT_NAME_EXISTS".into()));
        }
    }
    meta.name = name;
    meta.note = note;
    meta.updated_ms = now_ms();
    write_meta(data_dir, &meta)?;
    Ok(meta)
}

/// The「本次训练名」a slot's run carries — a LABEL. ★§F2⒝ batch 2 step ④b: the artifact slug no
/// longer derives from it (see [`run_artifact_slug`]), so this answers「显示什么」and nothing else.
///
/// It only ever lived in the run's own `run.json`, which is written AFTER a successful data
/// import — so `None` means this slot never completed a run.
///
/// ⚠ It answers for ONE run and there is no run selector yet, so it declines rather than picks
/// when a slot holds several. The caller then falls back to the project name, which is a worse
/// SUGGESTION and nothing more — but it is also the shape §F2⒝ batch 5 removes outright: the
/// training name stops being an identity there, and this becomes a label lookup.
pub fn slot_model_name(data_dir: &Path, id: &str, family: &str) -> Option<String> {
    let run = crate::training::trun::resolve_run_dir(&family_dir(data_dir, id, family), None)
        .inspect_err(|e| tracing::warn!("slot_model_name({id}/{family}): {e}"))
        .ok()?;
    run_model_name(&run)
}

/// The「本次训练名」ONE run's artifacts were built under.
///
/// ★§F2⒝ batch 2 step ④ — split out of [`slot_model_name`] because the slot-level question has
/// no answer once a slot holds two runs, and the way it FAILED was the dangerous part: the `.ok()?`
/// above turns `RUN_AMBIGUOUS` into `None`, i.e. into「这个槽还没起过名」. That is not a blank label
/// but a live behaviour change — `askRunName` returns the frozen name only `if (slot?.modelName)`,
/// so an empty one makes the app ask for a name on every 继续训练, and the name is what
/// `slugify` turns into `dataset_44k/<slug>/`, `config.spk` keys and `weights/<slug>*`.
pub fn run_model_name(run: &crate::training::trun::RunDir) -> Option<String> {
    run_json(run)
        .and_then(|v| v["model_name"].as_str().map(String::from))
        .filter(|s| !s.is_empty())
}

/// THE artifact identity ONE run's products were actually built under — `hps.name` and the
/// `weights/<slug>*` prefix.
///
/// ⚠ §F2⒝ ④d removed two entries from that list, and the removal is the point: the `config.spk`
/// key and the `<pool>/dataset_44k/<slug>/` slice directory are POOL products, and from identity
/// v2 ([`crate::training::tpool::identity_version`]) a sole speaker's are the constant
/// `tpool::SOLE_SPEAKER_DIR` instead of this run's name. A RUN-scoped label naming a POOL product
/// is what let a second run of one slot grow a second complete slice tree inside the shared pool.
/// Co-trained speakers keep their own slugs there — those are folded into the pool fingerprint.
///
/// ★§F2⒝ batch 2 step ④b — the reason this is a READ rather than a derivation. Until now the
/// slug was re-derived from the display name on every start, so the name WAS the identity: rename
/// the run and every one of those paths moves, orphaning what is already on disk and (for the
/// slice directory, which lives in the pool the runs SHARE) growing a second full preprocessing
/// tree that nothing ever reclaims — the pool is selected by `dataset.fingerprint` CONTENT, and
/// the slug is not in it.
///
/// `run.json` is written by [`crate::training::TrainingManager`] on every start, so its
/// `model_slug` is a POSITIVE fact: it is the value this run's existing artifacts carry. `None`
/// means the run has never started, and a run that never started holds nothing to orphan — which
/// is what makes the freeze a read instead of a migration.
pub fn run_artifact_slug(run: &crate::training::trun::RunDir) -> Option<String> {
    run_json(run)
        .and_then(|v| v["model_slug"].as_str().map(String::from))
        .filter(|s| !s.is_empty())
}

/// Rewrite ONE run's training name — **and nothing else**.
///
/// ★§F2⒝ batch 2 step ④b. This command can only exist because the artifact identity was frozen
/// first ([`run_artifact_slug`]): before that, "renaming" would have re-pointed `hps.name`, the
/// `weights/<slug>*` prefix, the `audition/<slug>_*` stems and the pool's slice directory on the
/// run's NEXT start, leaving every existing product an orphan and growing a second full
/// preprocessing tree. Here the file keeps its `model_slug` byte for byte; only the label moves.
///
/// The write is atomic (temp + rename) because `run.json` is the file `try_start`'s guards and
/// python's five chains both read: a kill mid-write would strand a truncated one, and the readers
/// treat "unparseable" as「这个 run 还没起过名」— an absence, not an error.
pub fn rename_run(run: &crate::training::trun::RunDir, name: &str) -> Result<()> {
    let fail = |e: String| UtaiError::Training(format!("RUN_RENAME_FAILED: {e}"));
    let path = run.join("run.json");
    // A run that never started has no `run.json` — and no artifacts to label either. It is not a
    // failure of the rename, it is a run whose name has not been asked for yet.
    let text = std::fs::read_to_string(&path)
        .map_err(|_| UtaiError::Training("RUN_NEVER_NAMED".into()))?;
    let mut v: serde_json::Value = serde_json::from_str(&text).map_err(|e| fail(e.to_string()))?;
    let obj = v
        .as_object_mut()
        .ok_or_else(|| fail("run.json is not an object".to_string()))?;
    obj.insert(
        "model_name".to_string(),
        serde_json::Value::String(name.to_string()),
    );
    let tmp = run.join("run.json.tmp");
    let bytes = serde_json::to_vec_pretty(&v).map_err(|e| fail(e.to_string()))?;
    std::fs::write(&tmp, bytes).map_err(|e| fail(e.to_string()))?;
    std::fs::rename(&tmp, &path).map_err(|e| fail(e.to_string()))?;
    Ok(())
}

/// The run's own `run.json`, parsed. One reader for both accessors above so they can never
/// disagree about which file answers「这个 run 的身份」.
fn run_json(run: &crate::training::trun::RunDir) -> Option<serde_json::Value> {
    std::fs::read_to_string(run.join("run.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
}

// ─────────────────────────── checkpoint inventory ───────────────────────────

/// What a checkpoint file IS — the four shapes the four families actually produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum CkptKind {
    /// The seeded pretrained base (step 0). Present in every workspace from the moment it is
    /// created, so it must never read as「用户练出来的东西」.
    Base,
    /// Training can continue from here.
    Resumable,
    /// A release snapshot under `weights/` — what you audition and import. **Generator only**
    /// (`sovits/train.py` writes just the generator), so it can never resume a GAN.
    Release,
    /// The best release snapshot by validation metric.
    Best,
    /// The naturally-finished export: a plain `weights/<slug>.pth` with no step in its name.
    /// One per slot, and the artifact a user is most likely to actually want — kept out of
    /// `Release` so the snapshot cleanup can never treat it as one of the periodic ones.
    Final,
    /// Half of a torn GAN save — a `G_<n>.pth` whose `D_<n>.pth` never landed, or the mirror.
    /// Hundreds of MB that cannot be resumed from: listed so it stays visible and reclaimable,
    /// never offered as a resume point.
    Orphan,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CkptRecord {
    /// Relative to the PROJECT directory — a data-dir move must not orphan the ledger that
    /// protects these files from the cleanup.
    pub rel: String,
    /// ★§F2⒝ batch 2 step ④ — WHICH run of the slot produced it (`trun::run_id_in_rel`).
    /// `""` = the slot root is the run (layout ≤ 2), a positive fact rather than "unknown".
    ///
    /// Carried on the record because the archive view lists a whole family in one table: without
    /// it, two runs' checkpoints interleave by mtime with nothing on the row saying which model
    /// they belong to, and「导入」would offer the same suggested name for both.
    pub run_id: String,
    /// Absolute path: what audition / import / attach already take.
    pub path: String,
    pub family: String,
    pub kind: CkptKind,
    /// Real training step. `None` for the RVC「只保留最新」sentinel — see below.
    pub step: Option<u64>,
    pub bytes: u64,
    pub mtime_ms: u64,
    /// Already exported into the model registry (from `project.json`'s ledger).
    pub imported: bool,
    /// Files belonging to the SAME archive that must be deleted with it — the `D_<n>.pth` of a
    /// GAN pair. `bytes` already includes them.
    pub companions: Vec<String>,
}

fn stat_of(p: &Path) -> (u64, u64) {
    let Ok(md) = std::fs::metadata(p) else { return (0, 0) };
    let mtime = md
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    (md.len(), mtime)
}

/// Every checkpoint a project holds, newest first.
///
/// This is the answer to「关掉 app 或点过『清空结果』之后,盘上到底还有什么」— until now the
/// candidate list was emitted by the python sidecar into memory and nothing ever scanned the
/// disk, so those files existed with no way left to reach them.
///
/// Two shape traps, both live on this dev machine and both pinned by tests:
/// * **RVC's default `keep_only_latest`** writes a fixed sentinel name `G_2333333.pth` /
///   `D_2333333.pth` (`rvc/train.py`) — 2333333 is a placeholder, NOT a step. Reporting it as
///   one would show「2333333 步」and, worse, make it sort above every real checkpoint. Upstream
///   resumes by **mtime** for exactly this reason (`rvc/train_utils.py`), which is why the
///   ordering here is mtime-first too.
/// * **the vocoder's `model_ckpt_steps_<N>.ckpt` counts in lightning GLOBAL steps**, and its
///   manual-optimisation GAN steps the D and G optimizers separately ⇒ N = 2 × 实际步. The
///   `weights/vocoder_<real>.ckpt` snapshots next to them carry the halved number already —
///   on this machine root {1000,2000,3000,3644} sits beside weights {500,1000,1500,1822}.
pub fn scan_project_ckpts(data_dir: &Path, id: &str, only: Option<&str>) -> Vec<CkptRecord> {
    let proj = project_dir(data_dir, id);
    let exported: Vec<String> = read_meta(data_dir, id)
        .map(|m| m.exported.into_iter().map(|e| e.from_ckpt_rel).collect())
        .unwrap_or_default();
    let mut out: Vec<CkptRecord> = Vec::new();

    for family in FAMILIES {
        if only.is_some_and(|f| f != family) {
            continue;
        }
        let slot = proj.join(family);
        if !slot.is_dir() {
            continue;
        }
        let relof = |abs: &Path| {
            abs.strip_prefix(&proj)
                .unwrap_or(abs)
                .to_string_lossy()
                .replace('\\', "/")
        };
        let mut push = |abs: PathBuf,
                        companion: Option<PathBuf>,
                        kind: CkptKind,
                        step: Option<u64>| {
            let rel = relof(&abs);
            let (mut bytes, mtime_ms) = stat_of(&abs);
            let companions: Vec<String> = companion
                .into_iter()
                .map(|c| {
                    bytes += stat_of(&c).0;
                    relof(&c)
                })
                .collect();
            out.push(CkptRecord {
                imported: exported.contains(&rel),
                run_id: crate::training::trun::run_id_in_rel(&rel, family),
                rel,
                path: abs.to_string_lossy().into_owned(),
                family: family.to_string(),
                kind,
                step,
                bytes,
                mtime_ms,
                companions,
            });
        };

        // ★§F2⒝ batch 2 — EVERY run of the slot, not one of them. This inventory exists precisely
        // because「盘上几个 GB、UI 看不见」, so scanning a single run would recreate the failure it
        // was written to end: the other runs' checkpoints would be invisible in the archive view
        // AND unreachable by the cleanup, while still costing the disk. `trun::run_dirs` answers
        // `[slot]` for as long as there is no `runs/` container, so this is byte-identical today.
        // ⛔ S132 §F2⒝ ④e — 「there is no `runs/`」and「`runs/` could not be read」are different
        // answers now (`trun::list_runs`). This inventory cannot express「I could not look」in its
        // return type, so the honest thing it CAN do is refuse to pretend the slot was scanned:
        // skipping it loudly keeps the cleanup from treating an unreadable slot as an empty one.
        let runs = match crate::training::trun::run_dirs(&slot) {
            Ok(runs) => runs,
            Err(e) => {
                tracing::error!(
                    "cannot enumerate the runs of {} ({e}) — its checkpoints are missing from this \
                     inventory; the archive view will not list them and the cleanup will not \
                     consider them",
                    slot.display()
                );
                continue;
            }
        };
        for run in runs {
            // ── run root: the resumable pairs ─────────────────────────────────────────────
            // `.` entries are never archives — a delete stages files into `.del_*` and the layout
            // migration parks trees in `.mig_*`. Reading them back as checkpoints would put a
            // half-deleted file in the list AND feed it into the next cleanup round.
            let entries: Vec<String> = std::fs::read_dir(&run)
                .map(|rd| {
                    rd.flatten()
                        .filter(|e| e.path().is_file())
                        .map(|e| e.file_name().to_string_lossy().into_owned())
                        .filter(|n| !n.starts_with('.'))
                        .collect()
                })
                .unwrap_or_default();
            for n in &entries {
                if let Some(num) = n.strip_prefix("G_").and_then(|s| s.strip_suffix(".pth")) {
                    let step = if num == "2333333" { None } else { num.parse::<u64>().ok() };
                    // A GAN resumes only from a G+D PAIR — a lone half would silently restart the
                    // discriminator. But it is still hundreds of MB on disk, and the two halves are
                    // written by SEPARATE calls (a kill between them leaves one behind; upstream's
                    // `clean_checkpoints` also prunes the two sides independently), so dropping it
                    // from the inventory would recreate the exact problem this inventory exists to
                    // end: a file nothing in the UI can see or reclaim.
                    let paired = entries.iter().any(|d| d == &format!("D_{num}.pth"));
                    let kind = match (paired, step) {
                        (false, _) => CkptKind::Orphan,
                        (true, Some(0)) => CkptKind::Base,
                        (true, _) => CkptKind::Resumable,
                    };
                    // The pair is ONE archive. D is the same order of magnitude and can be LARGER
                    // than G (on this machine an RVC D is 857 MB against G's 452 MB), so counting
                    // only G would understate a project by nearly half — and leave batch 3 deleting
                    // one side of every pair.
                    let companion = paired.then(|| run.join(format!("D_{num}.pth")));
                    push(run.join(n), companion, kind, step);
                } else if let Some(num) = n.strip_prefix("D_").and_then(|s| s.strip_suffix(".pth")) {
                    // the mirror orphan: a D whose G is gone
                    if !entries.iter().any(|g| g == &format!("G_{num}.pth")) {
                        push(run.join(n), None, CkptKind::Orphan, num.parse::<u64>().ok());
                    }
                } else if let Some(num) = n
                    .strip_prefix("model_ckpt_steps_")
                    .and_then(|s| s.strip_suffix(".ckpt"))
                {
                    // GLOBAL lightning steps — halve for the real one (see the doc comment).
                    let step = num.parse::<u64>().ok().map(|v| v / 2);
                    push(run.join(n), None, CkptKind::Resumable, step);
                }
            }

            // ── ★S117 §F2⒜: the resumable BEST snapshot ──────────────────────────────────
            // `resume_best/{G,D}.pth` + `state.json`. It lives in a SUBDIRECTORY on purpose (five
            // separate consumers walk the slot root looking for `G_*`/`D_*` and every one of them
            // would mis-handle it there — see `utai_train/resume_state.BEST_DIR`), which is exactly
            // why it has to be listed HERE explicitly: a scan that only knows the slot root would
            // leave a gigabyte-scale pair that nothing in the UI can see or reclaim, the failure
            // this inventory exists to end.
            //
            // Reported as `Resumable` rather than `Best`: `Best` means the inference-only release
            // snapshot under `weights/`, and the snapshot-cleanup copy explicitly promises to keep
            // 「可续训存档」. Which of the two resumable rows is the best point is answered by the
            // path (and by the resume picker), not by inventing an eighth kind.
            {
                let bd = run.join("resume_best");
                let (g, d, st) = (bd.join("G.pth"), bd.join("D.pth"), bd.join("state.json"));
                // `state.json` is the completion marker python writes LAST — without it the pair
                // beside it may be half-written, and offering that as a resume point is worse than
                // not offering it.
                if st.is_file() && g.is_file() && d.is_file() {
                    let step = std::fs::read_to_string(&st)
                        .ok()
                        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                        .and_then(|v| v["global_step"].as_u64());
                    push(g, Some(d), CkptKind::Resumable, step);
                }
            }

            // ── ★S119 §F8⒝: the VOCODER's resumable best snapshot ─────────────────────────
            // Same directory as the GAN pair above and deliberately NOT gated on the family: the two
            // payload shapes are mutually exclusive on disk (a GAN slot has no `model.ckpt`, a
            // vocoder slot has no `G.pth`), so a family test here would be one more thing that can
            // drift out of step with python. `state.json` is again the completion marker written
            // LAST.
            //
            // ⚠ The step is HALVED like every other vocoder row (`model_ckpt_steps_3644.ckpt` is
            // step 1822 above): python records `trainer.global_step`, which this manual-optimization
            // GAN advances 2 per batch. Reporting it raw would put the one number in the archive
            // list that is twice everything beside it.
            {
                let bd = run.join("resume_best");
                let (m, st) = (bd.join("model.ckpt"), bd.join("state.json"));
                if st.is_file() && m.is_file() {
                    let step = std::fs::read_to_string(&st)
                        .ok()
                        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                        .and_then(|v| v["global_step"].as_u64())
                        .map(|g| g / 2);
                    push(m, Some(st), CkptKind::Resumable, step);
                }
            }

            // ── ★S118 §F8⒜: the diffusion resume snapshots ────────────────────────────────
            // `diffusion/resume_best/{model.pt,state.json}` and `diffusion/resume_latest/…`. Same
            // reason the GAN block above has to be explicit: the loop below only accepts names of the
            // form `model_<digits>.pt` / `model_best.pt`, so a DIRECTORY entry falls straight through
            // its `continue` and 600 MB apiece would never appear in the archive list.
            // ⚠ Their bytes DO already reach the user through the recursive `storage::dir_size`
            // totals, so this is about the ARCHIVE view: seeing that the resume point exists, what
            // step it is at, and being able to reason about it at all.
            //
            // Reported as `Resumable` for the same reason as the GAN pair: `Best` means the
            // inference-only export (here `diffusion/model_best.pt`, which is written with
            // `optimizer=None`), and the snapshot is the opposite of that — it is the ONLY diffusion
            // artifact that always carries the optimizer.
            {
                let dd = run.join("diffusion");
                for sub in ["resume_best", "resume_latest"] {
                    let sd = dd.join(sub);
                    let (m, st) = (sd.join("model.pt"), sd.join("state.json"));
                    // `state.json` is the completion marker python writes LAST — without it the
                    // payload beside it may be half-written, and offering that as a resume point is
                    // worse than not offering it.
                    if st.is_file() && m.is_file() {
                        let step = std::fs::read_to_string(&st)
                            .ok()
                            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                            .and_then(|v| v["global_step"].as_u64());
                        push(m, Some(st), CkptKind::Resumable, step);
                    }
                }
            }

            // ── diffusion progress (lives inside the sovits slot) ─────────────────────────
            if let Ok(rd) = std::fs::read_dir(run.join("diffusion")) {
                for e in rd.flatten() {
                    let n = e.file_name().to_string_lossy().into_owned();
                    if n.starts_with('.') {
                        continue;
                    }
                    let Some(num) = n.strip_prefix("model_").and_then(|s| s.strip_suffix(".pt"))
                    else {
                        continue;
                    };
                    if !e.path().is_file() {
                        continue;
                    }
                    if num == "best" {
                        // `model_best.pt` is a BEST SNAPSHOT, never a resume point: the solver
                        // writes it with `optimizer=None` so it carries no optimizer state, and
                        // upstream's resume scan reads a non-numeric name as step 0. Offering it
                        // would rewind thousands of steps AND zero the AdamW momentum.
                        push(e.path(), None, CkptKind::Best, None);
                        continue;
                    }
                    // anything that is not `model_<digits>.pt` is not ours — do not guess
                    let Some(step) = num.parse::<u64>().ok() else { continue };
                    let kind = if step == 0 { CkptKind::Base } else { CkptKind::Resumable };
                    push(e.path(), None, kind, Some(step));
                }
            }

            // ── release snapshots ─────────────────────────────────────────────────────────
            if let Ok(rd) = std::fs::read_dir(run.join("weights")) {
                for e in rd.flatten() {
                    let n = e.file_name().to_string_lossy().into_owned();
                    if n.starts_with('.')
                        || !(n.ends_with(".pth") || n.ends_with(".ckpt"))
                        || !e.path().is_file()
                    {
                        continue;
                    }
                    let stem = n.rsplit_once('.').map(|(s, _)| s).unwrap_or(&n);
                    // Exactly two real shapes: `<slug>_e<epoch>_s<step>` (rvc/sovits/sovits_v2
                    // periodic) and `vocoder_<real step>` (vocoder — already halved on this side).
                    //
                    // ⚠ There was a blind `rsplit_once('_')` fallback here. `<slug>` is
                    // `<≤24 ascii>_<8 hex>` and the naturally-finished export is a plain
                    // `weights/<slug>.pth`, so whenever that hash happened to be all decimal
                    // digits (~2% of names) the fallback reported the HASH as the training step.
                    // No step is the honest answer; the UI renders it as "—".
                    let step = stem
                        .rsplit_once("_s")
                        .and_then(|(_, s)| s.parse::<u64>().ok())
                        .or_else(|| stem.strip_prefix("vocoder_").and_then(|s| s.parse::<u64>().ok()));
                    let kind = if stem.ends_with("_best") {
                        CkptKind::Best
                    } else if step.is_none() {
                        // no step in the name and not `_best` ⇒ the plain `<slug>.pth` the run
                        // writes when it finishes naturally. Distinguishing it matters: as a
                        // `Release` it would sit in the cleanup's candidate set, and it is the one
                        // file in `weights/` a user is most likely to actually want.
                        CkptKind::Final
                    } else {
                        CkptKind::Release
                    };
                    push(e.path(), None, kind, step);
                }
            }
        } // ★ end of the per-RUN loop (§F2⒝ batch 2)
    }
    // Newest first — and mtime, not the step number, is the ordering upstream itself trusts
    // (the RVC sentinel makes step ordering meaningless).
    out.sort_by(|a, b| b.mtime_ms.cmp(&a.mtime_ms).then_with(|| a.rel.cmp(&b.rel)));
    out
}

/// Why a checkpoint was NOT deleted. Stable CODEs — the frontend localises them through the
/// shared mapper, and「为什么没删」must be as visible as「删了什么」.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum KeptReason {
    /// In the project's export ledger.
    Exported,
    /// A model with this stem is installed in the registry right now — the ledger may simply
    /// never have recorded it (imports predating S76, a torn ledger write).
    StillInstalled,
    /// Older than the project's ledger baseline. A MIGRATED project cannot know which
    /// snapshots the user already imported, so everything predating the ledger is untouchable.
    PreLedger,
    /// Written seconds ago — the run may not have finished announcing it yet.
    JustWritten,
    /// Best / final / base / resumable — never a cleanup target in the first place.
    NotASnapshot,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CleanupPlan {
    /// Files to remove, each with its companions (a GAN pair goes atomically or not at all).
    pub delete: Vec<CkptRecord>,
    pub kept: Vec<(String, KeptReason)>,
    pub freeable_bytes: u64,
}

/// Decide what「清理未导入的快照」may remove. Pure given the inputs, so the judgement can be
/// tested without a registry or a filesystem full of multi-GB files.
///
/// Only two kinds are ever candidates: periodic `Release` snapshots under `weights/` (which
/// NOTHING prunes automatically — that is the whole reason this exists) and `Orphan` halves of
/// a torn GAN save. Base / Resumable / Best / Final are structurally excluded: resumability and
/// the best/final artefacts are the things a user would be most upset to lose, and none of them
/// grows without bound.
///
/// ⚠ Two rules were in the first draft and are deliberately ABSENT:
/// * "protect everything in the current run's candidate list" — `snapshot.ckpts` accumulates
///   and is not cleared on completion, so that set is EXACTLY the set the user wants to delete
///   right after a run. It made the feature release 0 bytes in its main scenario. Not deleting
///   a file that is being written is already guaranteed by the idle gate.
/// * "protect anything from the last N hours" — same problem at a coarser grain. What remains
///   is a seconds-scale guard against a save landing between the scan and the delete.
/// The record a DEFAULT 续训 would actually continue from — the one the project card's
/// 「可从第 N 步继续」 must name. `recs` must be `scan_project_ckpts`' output (newest-first by mtime).
///
/// ⛔★S118 §F8-res⒈ — this exists because "the newest Resumable" stopped meaning that in S117.
/// `resume_best/` is written AFTER the rolling pair (`sovits/train.py` saves `save_gd` inside the
/// epoch and `save_best` at its end), so whenever the metric improved in the last epoch the best
/// snapshot IS the mtime-newest Resumable — and the card then printed the BEST step while a
/// default resume (`resume_from` = "latest", every previous release's behaviour) continues from
/// the latest one. A label naming a step the button will not continue from is the same class of
/// lie §F2⒜ was built to remove.
///
/// The mtime ordering itself is KEPT and is not incidental: it is the order upstream resumes by,
/// and RVC's rolling `G_2333333.pth` has no orderable step at all (it maps to `step: None`), so
/// "max step" is not available as a rule. The fix is therefore a filter, not a re-sort.
///
/// ⚠ A slot whose ONLY resumable record is the best snapshot still reports it: that really is the
/// only thing there to continue from, and answering "no resume point" would be its own lie.
pub fn default_resume_record(recs: &[CkptRecord]) -> Option<&CkptRecord> {
    pick_default_resume(recs.iter())
}

/// Same rule, asked of ONE RUN's records (which the project detail holds as borrowed rows).
///
/// ★§F2⒝ batch 2 step ④ — a slot's record list spans every run, and its mtime order then answers
/// 「哪个 run 最后练过」, not 「这个 run 从哪继续」. Filtering first and asking after is the only
/// arrangement in which the number on a run's row is that run's.
pub fn default_resume_record_of<'a>(recs: &[&'a CkptRecord]) -> Option<&'a CkptRecord> {
    pick_default_resume(recs.iter().copied())
}

/// THE rule, once. Two entry points above only differ in how the caller holds the records.
fn pick_default_resume<'a>(
    it: impl Iterator<Item = &'a CkptRecord> + Clone,
) -> Option<&'a CkptRecord> {
    let is_best_snapshot = |r: &&CkptRecord| r.rel.contains("/resume_best/");
    let mut primary = it.clone();
    primary
        .find(|r| matches!(r.kind, CkptKind::Resumable) && !is_best_snapshot(r))
        .or_else(|| {
            let mut fallback = it;
            fallback.find(|r| matches!(r.kind, CkptKind::Resumable))
        })
}

pub fn plan_cleanup(
    records: &[CkptRecord],
    ledger_since_ms: u64,
    now_ms_: u64,
    installed_stem: &dyn Fn(&str) -> bool,
) -> CleanupPlan {
    const JUST_WRITTEN_MS: u64 = 60_000;
    let mut plan = CleanupPlan { delete: Vec::new(), kept: Vec::new(), freeable_bytes: 0 };
    for r in records {
        let stem = Path::new(&r.rel)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let reason = if !matches!(r.kind, CkptKind::Release | CkptKind::Orphan) {
            Some(KeptReason::NotASnapshot)
        } else if r.imported {
            Some(KeptReason::Exported)
        } else if installed_stem(&stem) {
            Some(KeptReason::StillInstalled)
        } else if ledger_since_ms > 0 && r.mtime_ms < ledger_since_ms {
            Some(KeptReason::PreLedger)
        } else if now_ms_.saturating_sub(r.mtime_ms) < JUST_WRITTEN_MS {
            Some(KeptReason::JustWritten)
        } else {
            None
        };
        match reason {
            Some(why) => plan.kept.push((r.rel.clone(), why)),
            None => {
                plan.freeable_bytes += r.bytes;
                plan.delete.push(r.clone());
            }
        }
    }
    plan
}

/// What a delete actually did. A bare `freed: u64` cannot express this command's most common
/// correct outcome — on a MIGRATED project every snapshot predates the ledger, so「一个都不删」
/// is right, and rendering that as「已释放 0 B」reads as a broken button.
#[derive(Debug, Clone, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct DeleteReport {
    pub freed_bytes: u64,
    pub deleted: Vec<String>,
    pub kept: Vec<KeptEntry>,
    /// The rename succeeded but the background removal did not finish — the space comes back
    /// on the next launch. NOT a failure: the archives are already unreachable.
    pub deferred: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct KeptEntry {
    pub rel: String,
    pub reason: KeptReason,
}

/// Remove the periodic snapshots nothing needs any more, for one family or the whole project.
///
/// `family` mirrors `list_project_ckpts`: the training page shows ONE architecture's archives,
/// so its cleanup button must act on exactly that set — deleting slots the user was never
/// shown is the same class of surprise this refactor exists to remove.
pub fn cleanup_snapshots(
    data_dir: &Path,
    id: &str,
    family: Option<&str>,
    installed_stem: &dyn Fn(&str) -> bool,
) -> Result<DeleteReport> {
    // Fail-closed on the metadata: `scan_project_ckpts` degrades an unreadable project.json to
    // an EMPTY ledger, which would turn every protection off at once — and the "ledger says
    // things were exported but none of them are on disk" tripwire cannot fire either, because
    // an empty ledger has nothing to mismatch. Read it here and refuse instead.
    let Some(meta) = read_meta(data_dir, id) else {
        return Err(UtaiError::Training("PROJECT_META_UNREADABLE".into()));
    };
    if meta.export_ledger_since_ms == 0 {
        // Every project this app creates or migrates stamps it. A zero means the file was
        // written by something else, or by a version that did not have the concept — either
        // way the「早于记账基线的一律保护」rule silently evaporates.
        return Err(UtaiError::Training("PROJECT_LEDGER_UNSTAMPED".into()));
    }
    let records = scan_project_ckpts(data_dir, id, family);
    // Tripwire: the ledger names exports that are nowhere on disk. Something moved or deleted
    // files behind our back, so every judgement below is suspect — report, do not delete.
    //
    // ⚠ It has to compare LIKE WITH LIKE. `records` is family-filtered whenever the caller names
    // one — and the only caller always does (Settings' per-slot「清理未导入的快照」button passes
    // `slot.family`) — while `meta.exported` is PROJECT-wide. Comparing the two directly meant a
    // project that had exported from one family could never clean ANOTHER: every ledger row
    // starts with the exporting family's directory name, so it can never match a record from a
    // different slot, the tripwire fired on a perfectly healthy ledger, and the button became a
    // permanent hard modal for that slot. Multi-family projects are the whole point of the S76
    // layout, so this was reachable rather than theoretical.
    let in_scope = |rel: &str| match family {
        // `<family>/…`. The trailing separator is load-bearing: without it `sovits` would also
        // claim every `sovits_v2/…` row.
        Some(f) => rel.starts_with(f) && rel.as_bytes().get(f.len()) == Some(&b'/'),
        None => true,
    };
    let scoped: Vec<&ExportedModel> =
        meta.exported.iter().filter(|e| in_scope(&e.from_ckpt_rel)).collect();
    if !scoped.is_empty() && !scoped.iter().any(|e| records.iter().any(|r| r.rel == e.from_ckpt_rel))
    {
        return Err(UtaiError::Training("PROJECT_LEDGER_STALE".into()));
    }
    let plan = plan_cleanup(&records, meta.export_ledger_since_ms, now_ms(), installed_stem);
    let mut paths: Vec<PathBuf> = Vec::new();
    let proj = project_dir(data_dir, id);
    for r in &plan.delete {
        paths.push(PathBuf::from(&r.path));
        paths.extend(r.companions.iter().map(|c| proj.join(c)));
    }
    let tomb = tombstone(data_dir, id, &paths)?;
    let deferred = match &tomb {
        Some(t) => crate::util::remove_dir_all_robust(t).is_err(),
        None => false,
    };
    Ok(DeleteReport {
        freed_bytes: plan.freeable_bytes,
        deleted: plan.delete.iter().map(|r| r.rel.clone()).collect(),
        kept: plan.kept.into_iter().map(|(rel, reason)| KeptEntry { rel, reason }).collect(),
        deferred,
    })
}

/// Delete ONE architecture slot — its checkpoints, caches and audition renders. The project's
/// shared `dataset/` and every sibling slot are untouched.
pub fn delete_slot(data_dir: &Path, id: &str, family: &str) -> Result<DeleteReport> {
    if !FAMILIES.contains(&family) {
        return Err(UtaiError::Training(format!("TRAINING_BAD_FAMILY: {family}")));
    }
    let slot = family_dir(data_dir, id, family);
    if !slot.is_dir() {
        return Err(UtaiError::Training("WORKSPACE_MISSING".into()));
    }
    let freed = crate::commands::storage::dir_size(&slot);
    let tomb = tombstone(data_dir, id, &[slot])?;
    let deferred = match &tomb {
        Some(t) => crate::util::remove_dir_all_robust(t).is_err(),
        None => false,
    };
    Ok(DeleteReport { freed_bytes: freed, deleted: vec![family.to_string()], kept: Vec::new(), deferred })
}

/// Delete a whole project: every slot AND the shared dataset. Models already exported into the
/// registry are copies and are not affected.
pub fn delete_project(data_dir: &Path, id: &str) -> Result<DeleteReport> {
    let dir = project_dir(data_dir, id);
    if !dir.is_dir() {
        return Err(UtaiError::Training("WORKSPACE_MISSING".into()));
    }
    let freed = crate::commands::storage::dir_size(&dir);
    let tomb = tombstone(data_dir, id, &[dir])?;
    // A torn layout migration parks the rest of this project beside it; leaving those behind
    // would let the next boot fold them back into the id we just deleted.
    let root = training_root(data_dir);
    let _ = crate::util::remove_dir_all_robust(&root.join(format!("{STAGING_PREFIX}{id}")));
    let _ = std::fs::remove_file(root.join(format!("{MARKER_PREFIX}{id}.json")));
    // (The listing cache's row is dropped by the COMMAND — `forget_project` — because the cache
    // lives beside `config.json`, not in the data dir, and this function only knows the latter.
    // Leaving it would resurrect the project as a MISSING ghost right after a deliberate delete.)
    let deferred = match &tomb {
        Some(t) => crate::util::remove_dir_all_robust(t).is_err(),
        None => false,
    };
    Ok(DeleteReport { freed_bytes: freed, deleted: vec![id.to_string()], kept: Vec::new(), deferred })
}

/// Prefix for a delete-in-progress staging directory at the training root.
const TOMB_PREFIX: &str = ".del_";

/// Move `paths` into a fresh tombstone DIRECTORY at the training root and return it.
///
/// Three properties, each paid for by a specific failure:
/// * **a directory, at the ROOT** — the only reaper is `get_storage_report`'s depth-1 scan,
///   which skips non-directories and never recurses, so a tombstone FILE or one parked inside
///   a project would never be reclaimed and would keep counting toward the project's size;
/// * **rename, not delete** — same volume, so it is atomic, and a locked file fails the rename
///   instead of leaving a half-erased archive;
/// * **the whole unit at once** — a GAN pair must vanish together. `_seed_base_checkpoints`
///   checks `G_*` and `D_*` INDEPENDENTLY, so a crash between deleting one and the other makes
///   the next run seed a fresh 0-step counterpart for the surviving half: a silently corrupt
///   resume rather than a missing one.
fn tombstone(data_dir: &Path, id: &str, paths: &[PathBuf]) -> Result<Option<PathBuf>> {
    if paths.is_empty() {
        return Ok(None);
    }
    let tomb = training_root(data_dir).join(format!(
        "{TOMB_PREFIX}{id}_{}_{}",
        std::process::id(),
        now_ms()
    ));
    std::fs::create_dir_all(&tomb)
        .map_err(|e| UtaiError::Training(format!("TRAINING_DELETE_FAILED: {e}")))?;
    for (i, p) in paths.iter().enumerate() {
        if !p.exists() {
            continue;
        }
        let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        // prefix with an index: two families can hold identically-named files
        crate::util::rename_with_retry(p, &tomb.join(format!("{i:04}_{name}")), "TRAINING_DELETE")
            .map_err(UtaiError::Training)?;
    }
    Ok(Some(tomb))
}

/// Reclaim tombstones left by a previous session. Called at startup, next to the layout
/// migration — the same reasoning applies (nothing holds a handle under `<data>/training` yet).
///
/// Skips a tombstone whose pid is still ALIVE: with double-launch supported, the other instance
/// may be mid-delete, and racing it turns its successful delete into a reported failure.
pub fn sweep_tombstones(data_dir: &Path) {
    let Ok(rd) = std::fs::read_dir(training_root(data_dir)) else { return };
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let Some(rest) = name.strip_prefix(TOMB_PREFIX) else { continue };
        if !e.path().is_dir() {
            continue;
        }
        // `.del_<id>_<pid>_<ms>` — the pid is the second-to-last field.
        let alive = rest
            .rsplit('_')
            .nth(1)
            .and_then(|p| p.parse::<u32>().ok())
            .is_some_and(|pid| pid != std::process::id() && crate::crashlog::pid_alive(pid));
        if alive {
            tracing::info!("tombstone {name} belongs to a live sibling instance — leaving it");
            continue;
        }
        match crate::util::remove_dir_all_robust(&e.path()) {
            Ok(()) => tracing::info!("reclaimed training tombstone {name}"),
            // Loud, not silent: this is disk the user asked us to free.
            Err(err) => tracing::warn!("training tombstone {name} not reclaimed: {err}"),
        }
    }
}

/// Record that a checkpoint became an installed model. The ledger is what keeps「清理未导入的
/// 快照」(batch 3) from deleting the snapshots a user deliberately kept, so it stores paths
/// RELATIVE to the project — an absolute path would stop matching the moment the data
/// directory moves, and the cleanup would then see zero matches and delete everything.
pub fn record_export(
    data_dir: &Path,
    id: &str,
    name: &str,
    model_type: &str,
    from_ckpt: &str,
) -> Result<()> {
    let Some(mut meta) = read_meta(data_dir, id) else {
        return Err(UtaiError::Training("PROJECT_META_UNREADABLE".into()));
    };
    let proj = project_dir(data_dir, id);
    let rel = Path::new(from_ckpt)
        .strip_prefix(&proj)
        .map(|r| r.to_string_lossy().replace('\\', "/"))
        .unwrap_or_else(|_| from_ckpt.to_string());
    meta.exported.retain(|e| !(e.from_ckpt_rel == rel && e.name == name));
    meta.exported.push(ExportedModel {
        name: name.to_string(),
        model_type: model_type.to_string(),
        from_ckpt_rel: rel,
        at_ms: now_ms(),
    });
    meta.updated_ms = now_ms();
    write_meta(data_dir, &meta)
}

// ─────────────────────── the listing + its size cache (batch 4) ───────────────────────

/// `<app_dir>/training-projects.json` — a CACHE, never an authority.
///
/// The truth about a project is its own `project.json`, and the truth about which projects
/// exist is the directory listing (module docs, invariant 1). This file exists for exactly two
/// things the listing UI cannot get cheaply or at all:
///
/// * **sizes** — a per-project `dir_size` walk over tens of GB is not something to run on every
///   page open;
/// * **absence** — a project that DISAPPEARED (folder deleted outside the app, a data root
///   swapped underneath us) leaves nothing on disk to list, and「盘上还在、app 里没了」has an
///   equally bad mirror image:「app 里没了,用户也不知道为什么」. Remembering the id lets the row
///   survive as a visible, explainable, dismissible state.
///
/// ## Why it is NOT `<data>/training/projects.json` (the S75 design said it would be)
///
/// `training` is one of `MIGRATED_SUBTREES` (`commands::settings`), and `migrate_data_dir`
/// copies each subtree and then VERIFIES it file-by-file **by byte length** — with
/// `skips_dot_top("training") == false`, so no name is exempt. A cache rewritten between the
/// copy and the verify fails that comparison, and a `.tmp` that appears in that window reads as
/// "missing after copy": either way the user's entire data-directory migration aborts with
/// `MIGRATE_VERIFY_FAILED`. (The startup reclaim's delta-sync would also copy the OLD root's
/// stale cache over the new one — `layout_aware` only guards directories.) Nothing about a
/// derived, rebuildable cache is worth that, so it lives beside `config.json` instead, outside
/// every subtree the data-dir machinery touches.
///
/// It records WHICH data root it describes: switching data dirs invalidates every figure in it
/// at once, and silently showing another root's sizes would be worse than showing none.
/// Torn or hand-mangled content is rebuilt wholesale rather than reported — a broken cache must
/// never break a listing. Concurrent writers (double-launch is supported here) race harmlessly:
/// the write is atomic and a lost update costs one re-measure.
pub const PROJECTS_INDEX: &str = "training-projects.json";

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ProjectSizes {
    pub total_bytes: u64,
    pub dataset_bytes: u64,
    /// family → bytes, only for slots that exist. BTreeMap so the file stays diff-stable.
    #[serde(default)]
    pub family_bytes: std::collections::BTreeMap<String, u64>,
    /// 0 = never measured. The UI shows「—」rather than a confident, wrong「0 B」.
    pub computed_ms: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
struct IndexEntry {
    /// Last known display name — the only way a MISSING project can still be named.
    #[serde(default)]
    name: String,
    #[serde(default)]
    sizes: ProjectSizes,
    /// When this id was first seen in the index but not on disk. 0 = present.
    #[serde(default)]
    missing_since_ms: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
struct SizeIndex {
    #[serde(default)]
    version: u32,
    /// The data root these figures describe.
    #[serde(default)]
    data_dir: String,
    #[serde(default)]
    projects: std::collections::BTreeMap<String, IndexEntry>,
}

const INDEX_VERSION: u32 = 1;

fn index_path(app_dir: &Path) -> PathBuf {
    app_dir.join(PROJECTS_INDEX)
}

fn fresh_index(data_dir: &Path) -> SizeIndex {
    SizeIndex {
        version: INDEX_VERSION,
        data_dir: data_dir.to_string_lossy().into_owned(),
        ..Default::default()
    }
}

fn read_index(app_dir: &Path, data_dir: &Path) -> SizeIndex {
    let parsed: Option<SizeIndex> = std::fs::read_to_string(index_path(app_dir))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok());
    match parsed {
        // A future version's file is not ours to interpret, and another root's figures are not
        // ours to show — start over rather than read either with today's meaning.
        Some(ix)
            if ix.version == INDEX_VERSION && Path::new(&ix.data_dir) == data_dir =>
        {
            ix
        }
        _ => fresh_index(data_dir),
    }
}

/// Best effort by design: a cache that cannot be written must not fail the listing that
/// produced it (every write of real user data fails loudly on its own).
///
/// The temp name carries the pid because double-launch is supported: a shared fixed `.tmp`
/// would let two instances write the same file and rename each other's half-written bytes into
/// place. (`write_meta`'s fixed `project.json.tmp` is per-PROJECT and therefore far narrower;
/// this one is a single global file every listing touches.)
fn write_index(app_dir: &Path, ix: &SizeIndex) {
    let tmp = app_dir.join(format!("{PROJECTS_INDEX}.{}.tmp", std::process::id()));
    let Ok(body) = serde_json::to_vec_pretty(ix) else { return };
    if std::fs::write(&tmp, body).is_ok() && std::fs::rename(&tmp, index_path(app_dir)).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

/// Walk one project. `dir_size` is the single source for「一个目录有多大」in this app
/// (`commands::storage`), so the storage report and this listing can never disagree on the
/// arithmetic — only on when they last looked.
fn measure_project(data_dir: &Path, id: &str) -> ProjectSizes {
    use crate::commands::storage::dir_size;
    let proj = project_dir(data_dir, id);
    let mut family_bytes = std::collections::BTreeMap::new();
    for f in FAMILIES {
        let fd = proj.join(f);
        if fd.is_dir() {
            family_bytes.insert(f.to_string(), dir_size(&fd));
        }
    }
    ProjectSizes {
        total_bytes: dir_size(&proj),
        dataset_bytes: dir_size(&proj.join(DATASET_DIR)),
        family_bytes,
        computed_ms: now_ms(),
    }
}

/// One row of the project landing page.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectSummary {
    pub id: String,
    pub name: String,
    pub note: String,
    pub created_ms: u64,
    pub updated_ms: u64,
    /// Migration could not classify this directory: nothing was moved, nothing was deleted,
    /// and training into it is refused (`resolve_or_create`).
    pub needs_attention: Option<String>,
    /// Architecture slots that exist on disk, in [`FAMILIES`] order.
    pub families: Vec<String>,
    pub has_dataset: bool,
    #[serde(flatten)]
    pub sizes: ProjectSizes,
    /// Remembered by the index, absent from disk. Listed, never silently dropped, and barred
    /// from every training entry point.
    pub missing: bool,
}

/// Every project, for the landing page.
///
/// `measure` = walk the disk for sizes (seconds over tens of GB) and refresh the cache; `false`
/// answers instantly from the cache. ONE code path either way, so a cached row and a fresh row
/// can never be assembled differently.
pub fn list_project_summaries(app_dir: &Path, data_dir: &Path, measure: bool) -> Vec<ProjectSummary> {
    let on_disk = list_projects(data_dir);
    let mut ix = read_index(app_dir, data_dir);
    let before = ix.clone();
    let now = now_ms();
    let mut out: Vec<ProjectSummary> = Vec::new();

    for m in &on_disk {
        let entry = ix.projects.entry(m.id.clone()).or_default();
        entry.name = m.name.clone();
        entry.missing_since_ms = 0;
        if measure || entry.sizes.computed_ms == 0 {
            entry.sizes = measure_project(data_dir, &m.id);
        }
        let proj = project_dir(data_dir, &m.id);
        out.push(ProjectSummary {
            id: m.id.clone(),
            name: m.name.clone(),
            note: m.note.clone(),
            created_ms: m.created_ms,
            updated_ms: m.updated_ms,
            needs_attention: m.needs_attention.clone(),
            families: FAMILIES
                .iter()
                .filter(|f| proj.join(f).is_dir())
                .map(|f| f.to_string())
                .collect(),
            has_dataset: has_dataset(data_dir, &m.id),
            sizes: entry.sizes.clone(),
            missing: false,
        });
    }

    // A directory with NO `project.json` is a pre-S76 workspace the migration has not folded
    // yet — it stands down entirely while a sibling instance is alive, and an individual tree
    // can fail its renames because something holds a handle. It retries every boot, but until
    // then「盘上还在、app 里没了」is exactly the outcome this refactor must never produce, so it
    // gets a row: visible, explained, and not enterable (`needs_attention` bars every training
    // entry point, and `resolve_or_create` refuses it anyway).
    let mut pending: Vec<String> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(training_root(data_dir)) {
        for e in rd.flatten() {
            let id = e.file_name().to_string_lossy().into_owned();
            if id.starts_with('.') || !e.path().is_dir() {
                continue;
            }
            if !e.path().join(PROJECT_META).is_file() {
                pending.push(id);
            }
        }
    }
    pending.sort();
    let pending_ids: std::collections::HashSet<String> = pending.iter().cloned().collect();
    for id in pending {
        out.push(ProjectSummary {
            name: id.clone(),
            id,
            note: String::new(),
            created_ms: 0,
            updated_ms: 0,
            needs_attention: Some("TRAINING_LAYOUT_MIGRATION_PENDING".into()),
            families: Vec::new(),
            has_dataset: false,
            sizes: ProjectSizes::default(),
            missing: false,
        });
    }

    let present: std::collections::HashSet<&str> = on_disk.iter().map(|m| m.id.as_str()).collect();
    for (id, entry) in ix.projects.iter_mut() {
        // `pending_ids` counts as present: the directory IS there (it just has not been folded
        // into the new layout yet), and listing it twice — once as「待迁移」and once as
        // 「已不在磁盘上」— would be worse than either row alone.
        if present.contains(id.as_str()) || pending_ids.contains(id) {
            continue;
        }
        if entry.missing_since_ms == 0 {
            entry.missing_since_ms = now;
        }
        out.push(ProjectSummary {
            id: id.clone(),
            name: if entry.name.is_empty() { id.clone() } else { entry.name.clone() },
            note: String::new(),
            created_ms: 0,
            updated_ms: entry.missing_since_ms,
            needs_attention: None,
            families: Vec::new(),
            has_dataset: false,
            // Stale by definition — keep the last known figures so the row can still say how
            // much disk this project WAS using, with computed_ms telling the user how old that is.
            sizes: entry.sizes.clone(),
            missing: true,
        });
    }

    if ix != before {
        write_index(app_dir, &ix);
    }
    // Newest activity first; id breaks ties so the order never wobbles between calls.
    out.sort_by(|a, b| b.updated_ms.cmp(&a.updated_ms).then_with(|| a.id.cmp(&b.id)));
    out
}

/// Drop a project's cache row: what「移除记录」does to a MISSING project, and what a real
/// deletion must do so the row cannot come back as a ghost.
pub fn forget_project(app_dir: &Path, data_dir: &Path, id: &str) {
    let mut ix = read_index(app_dir, data_dir);
    if ix.projects.remove(id).is_some() {
        write_index(app_dir, &ix);
    }
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

/// Every subdirectory name this repo creates inside a workspace/slot root.
///
/// ⛔ It exists to be CHECKED, not read. `has_family_slot` decides "legacy or new layout" by
/// asking whether a subdirectory is named exactly after a family, and that decision is only sound
/// while no such name can occur — a claim that has to be enforced, because the thing that breaks
/// it (someone adding a directory) is precisely the thing nobody would think to re-verify.
///
/// The previous generation of this contract was the prose above the function claiming to list
/// "the complete set of subdirectories anything has ever created inside a workspace root". It was
/// ALREADY false when §F2⒝ read it: `resume_best/`, `resume_latest/`, `eval/` and
/// `lightning_logs/` had been added since, and nothing anywhere noticed. A claim of completeness
/// that nothing checks decays silently from the moment it is written (S120 §F9's blood lesson,
/// in the same shape).
///
/// Kept ALPHABETICAL, and asserted against the two Rust-side names plus this list's own length so
/// that adding an entry is a deliberate edit rather than a drive-by.
pub(crate) const WORKSPACE_SUBDIRS: &[&str] = &[
    "0_gt_wavs",     // rvc slices (gt sample rate)
    "1_16k_wavs",    // rvc slices (16k, the feature/f0 input)
    "2a_f0",         // rvc coarse f0
    "2b-f0nsf",      // rvc Hz f0
    "3_feature256",  // rvc ContentVec v1
    "3_feature768",  // rvc ContentVec v2
    "audition",      // Rust: per-checkpoint audition cache
    "aug_meta",      // S41 augmentation provenance
    "cluster",       // sovits retrieval / kmeans
    "dataset",       // Rust: pre-S76 imported dataset (a sibling of the checkpoints back then)
    "dataset_44k",   // sovits / sovits_v2 slices + extracted features
    "diffusion",     // sovits_diff run products (expdir)
    "eval",          // sovits TensorBoard eval subdir
    "filelists",     // train/val lists
    "lightning_logs",// vocoder (lightning)
    "mute",          // rvc mute-asset copy
    "npz",           // vocoder processed slices
    "pools",         // §F2⒝ preprocessing pools
    "resume_best",   // S117/S118/S119 resumable best snapshot
    "resume_latest", // S118 rolling resume point
    "runs",          // §F2⒝ batch 2 per-run container (`training::trun::RUNS_DIR`)
    "slices",        // vocoder slices
    "weights",       // published small checkpoints
];

/// Does this directory already have the new shape? Used INSTEAD of a phase field in the
/// marker, because the two shapes are structurally distinguishable: a legacy workspace can
/// never contain a subdirectory named exactly after a family.
///
/// The claim is enforced by [`WORKSPACE_SUBDIRS`] + `workspace_subdirs_never_collide_with_a_family`,
/// not by a comment.
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
    if !training_root(data_dir).is_dir() {
        return MigrationReport::default();
    }
    if crate::crashlog::other_instance_alive() {
        tracing::warn!("training layout migration postponed: another live instance detected");
        return MigrationReport::default();
    }
    migrate_all(data_dir)
}

/// The pass itself, without the sibling-instance stand-down.
///
/// Split out because `other_instance_alive` asks a MACHINE-wide question (it scans the log dir
/// for live `session.<pid>.alive` sentinels) while a test owns a private temp data dir no
/// sibling could possibly be touching. Fused, every migration test failed whenever a dev build
/// happened to be running — proof that the guard works, but not something a unit test may
/// depend on.
fn migrate_all(data_dir: &Path) -> MigrationReport {
    let mut report = MigrationReport::default();
    let root = training_root(data_dir);

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
    let empty_shell = !crate::training::slot_holds_work(&dir)
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

    /// ⛔ The enumeration that replaced a prose claim of completeness (see [`WORKSPACE_SUBDIRS`]).
    ///
    /// Three separate things are pinned, because each of them can rot on its own:
    /// (1) the load-bearing property — no workspace subdirectory is named after a family;
    /// (2) the LIST itself — otherwise "the live set equals my table" is the table compared with
    ///     itself, and adding a wrong entry passes (S105);
    /// (3) every pool product is in it — so adding one to `tpool::POOL_ENTRIES` without declaring
    ///     it here turns this red instead of quietly re-opening the collision question.
    #[test]
    fn workspace_subdirs_never_collide_with_a_family() {
        for name in WORKSPACE_SUBDIRS {
            assert!(
                !FAMILIES.contains(name),
                "{name:?} collides with a family name — `has_family_slot` would read a legacy \
                 workspace as already migrated and migration recovery would silently break"
            );
        }
        let mut sorted = WORKSPACE_SUBDIRS.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.as_slice(), WORKSPACE_SUBDIRS, "keep it sorted and duplicate-free");
        assert_eq!(
            WORKSPACE_SUBDIRS.len(),
            23,
            "adding a workspace subdirectory is a deliberate edit: add it above, then update this \
             count, then re-read why `has_family_slot` depends on the list"
        );
        for family in FAMILIES {
            for entry in crate::training::tpool::pool_entries_for(family) {
                // the fingerprint is a FILE, everything else in the table is a directory
                if entry == crate::training::tpool::FINGERPRINT {
                    continue;
                }
                assert!(
                    WORKSPACE_SUBDIRS.contains(&entry),
                    "pool entry {entry:?} ({family}) is not in WORKSPACE_SUBDIRS"
                );
            }
        }
        // and the container the pools live in, which is the newest member of the set
        assert!(WORKSPACE_SUBDIRS.contains(&crate::training::tpool::POOLS_DIR));
        // ★§F2⒝ batch 2 — and every RUN product that is a directory, for the same reason as the
        // pool half: the run table IS consumed in production (`trun::plan_slot_runs` classifies a
        // real slot with it), so anchoring the list to it means "someone added a directory kind"
        // cannot pass silently here. `Prefix` entries are checkpoint FILES; the exact names below
        // are the ones that are, or can be, directories.
        for name in ["weights", "resume_best", "resume_latest", "diffusion", "cluster", "eval",
                     "lightning_logs", "filelists", "audition"] {
            assert!(
                crate::training::trun::is_run_entry(name),
                "{name:?} is listed here as a run directory but the run table does not claim it"
            );
            assert!(WORKSPACE_SUBDIRS.contains(&name), "run directory {name:?} is not declared");
        }
        assert!(WORKSPACE_SUBDIRS.contains(&crate::training::trun::RUNS_DIR));
    }

    /// ⛔ §F2⒝ batch 2 — the claim [`WORKSPACE_SUBDIRS`] exists to protect, asserted against a
    /// REAL directory tree instead of against the list itself.
    ///
    /// The list-vs-list assertions above cannot see the failure that actually happens: someone
    /// CREATES a directory. This builds a slot that really contains every declared name plus both
    /// containers and then asks `has_family_slot` — the predicate the layout migration bets on —
    /// what it sees. If a future entry is ever named after a family, this goes red by running the
    /// real predicate over the real bytes rather than by comparing two copies of one table.
    ///
    /// ⚠ Its honest limit, stated so nothing reads it as more than it is: it can only build names
    /// that are DECLARED. A directory python invents and nobody lists is still invisible here —
    /// what catches that one is `trun::plan_slot_runs` reporting it as `unknown` (a loud warn at
    /// migration time, in production) and the python-side `gate_pool_table.py`.
    #[test]
    fn a_real_slot_tree_never_makes_has_family_slot_lie() {
        let data = tmp_root("subdirs");
        let proj = project_dir(&data, "p1_aaaabbbb");
        let slot = proj.join("rvc");
        for name in WORKSPACE_SUBDIRS {
            std::fs::create_dir_all(slot.join(name)).unwrap();
        }
        // …plus what really lives inside the two containers on a migrated slot
        std::fs::create_dir_all(slot.join(crate::training::tpool::POOLS_DIR).join("p0123456789ab"))
            .unwrap();
        std::fs::create_dir_all(
            crate::training::trun::runs_root(&slot).join(crate::training::trun::legacy_run_id("rvc")),
        )
        .unwrap();

        assert!(has_family_slot(&proj), "the project really does hold a family slot");
        assert!(
            !has_family_slot(&slot),
            "no entry inside a slot may be named after a family — `has_family_slot` decides \
             「legacy or migrated」 by exactly this question, and a hit here would make the \
             migration read an unmigrated tree as already done"
        );
        // the same must hold one level deeper, where a RUN id is a directory name
        assert!(!has_family_slot(&crate::training::trun::runs_root(&slot)));
        let _ = std::fs::remove_dir_all(data);
    }

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
        let rep = migrate_all(&data);
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
        let rep2 = migrate_all(&data);
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
        let rep = migrate_all(&data);
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
        let rep = migrate_all(&data);
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
        let rep = migrate_all(&data);
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

        let rep = migrate_all(&data);
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
        let rep = migrate_all(&data);
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
        assert!(migrate_all(&data).migrated.contains(&"diffish_22223333".to_string()));
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
        // Either outcome is correct — what must NEVER happen is a second project. With no
        // sibling instance the on-demand retry folds the workspace in; with one alive (a dev
        // build running while the suite does, say) it refuses loudly instead of racing it.
        // Asserting both keeps the invariant under test independent of machine state.
        match resolve_or_create(&data, "歌姫テスト") {
            Ok(m) => {
                assert_eq!(m.id, crate::training::slugify("歌姫テスト"));
                assert!(ws.join("rvc").join("G_2333333.pth").is_file());
            }
            Err(e) => assert!(
                e.to_string().contains("TRAINING_LAYOUT_MIGRATION_PENDING"),
                "the only acceptable refusal is the pending-migration one, got {e}"
            ),
        }
        assert_eq!(
            std::fs::read_dir(training_root(&data)).unwrap().flatten().count(),
            1,
            "must never mint a second project directory"
        );
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
        assert_eq!(migrate_all(&data).flagged, vec!["weird_00000000".to_string()]);
        assert!(resolve_or_create(&data, "weird_00000000").is_err());

        // the user moves the content into the slot themselves and restarts
        std::fs::create_dir_all(ws.join("rvc")).unwrap();
        std::fs::rename(ws.join("G_100.pth"), ws.join("rvc").join("G_100.pth")).unwrap();
        migrate_all(&data);
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

        migrate_all(&data);

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

    /// The two shape traps, built exactly as the four families really write them (the
    /// numbers are copied off this dev machine's own workspaces).
    #[test]
    fn scan_handles_the_rvc_sentinel_and_the_vocoder_double_count() {
        let data = tmp_root("scan");
        let id = "scan_12345678";
        let p = project_dir(&data, id);
        write_meta(&data, &ProjectMeta { id: id.into(), name: "n".into(), ..Default::default() }).unwrap();

        // rvc: the「只保留最新」sentinel pair + release snapshots
        let rvc = p.join("rvc");
        std::fs::create_dir_all(rvc.join("weights")).unwrap();
        std::fs::write(rvc.join("G_2333333.pth"), b"g").unwrap();
        std::fs::write(rvc.join("D_2333333.pth"), b"d").unwrap();
        std::fs::write(rvc.join("weights").join("m_e6_s1580.pth"), b"w").unwrap();
        std::fs::write(rvc.join("weights").join("m_best.pth"), b"w").unwrap();
        // a torn save: G without its D must NOT be offered as a resume point
        std::fs::write(rvc.join("G_777.pth"), b"g").unwrap();
        // the naturally-finished export: `<slug>.pth` where slug = <base>_<8 hex>. A hash of
        // all decimal digits must NOT be read as a training step.
        std::fs::write(rvc.join("weights").join("mymodel_12345678.pth"), b"w").unwrap();

        // sovits: seeded base pair + a real pair + diffusion
        let sov = p.join("sovits");
        std::fs::create_dir_all(sov.join("diffusion")).unwrap();
        for n in ["G_0.pth", "D_0.pth", "G_15473.pth", "D_15473.pth"] {
            std::fs::write(sov.join(n), b"x").unwrap();
        }
        std::fs::write(sov.join("diffusion").join("model_0.pt"), b"x").unwrap();
        std::fs::write(sov.join("diffusion").join("model_392.pt"), b"x").unwrap();
        std::fs::write(sov.join("diffusion").join("model_best.pt"), b"x").unwrap();

        // vocoder: root counts GLOBAL steps (2x), weights carry the real number
        let voc = p.join("vocoder");
        std::fs::create_dir_all(voc.join("weights")).unwrap();
        std::fs::write(voc.join("model_ckpt_steps_3644.ckpt"), b"x").unwrap();
        std::fs::write(voc.join("weights").join("vocoder_1822.ckpt"), b"x").unwrap();

        let all = scan_project_ckpts(&data, id, None);
        let by = |rel: &str| all.iter().find(|r| r.rel == rel).unwrap_or_else(|| panic!("missing {rel}"));

        // ★ the sentinel is a NAME, not a step — reporting 2333333 would also sort it above
        // every real checkpoint
        assert_eq!(by("rvc/G_2333333.pth").step, None);
        assert_eq!(by("rvc/G_2333333.pth").kind, CkptKind::Resumable);
        // ★ 3644 global = 1822 real, matching the weights snapshot beside it
        assert_eq!(by("vocoder/model_ckpt_steps_3644.ckpt").step, Some(1822));
        assert_eq!(by("vocoder/weights/vocoder_1822.ckpt").step, Some(1822));
        // ★ a lone G is a torn save: visible (hundreds of MB) but never a resume point
        assert_eq!(by("rvc/G_777.pth").kind, CkptKind::Orphan);
        // ★ the pair is ONE archive and its size counts BOTH halves (D can be larger than G)
        assert_eq!(by("rvc/G_2333333.pth").companions, vec!["rvc/D_2333333.pth".to_string()]);
        assert_eq!(by("rvc/G_2333333.pth").bytes, 2, "G(1) + D(1)");
        assert!(all.iter().all(|r| r.rel != "rvc/D_2333333.pth"), "the D rides on its G row");
        // ★ model_best.pt has no optimizer state — it is a best SNAPSHOT, not a resume point
        assert_eq!(by("sovits/diffusion/model_best.pt").kind, CkptKind::Best);
        assert_eq!(by("sovits/diffusion/model_best.pt").step, None);
        // ★ a plain `weights/<slug>.pth` carries no step — never mistake the slug hash for one
        assert_eq!(by("rvc/weights/mymodel_12345678.pth").step, None);
        // seeded bases are not the user's work
        assert_eq!(by("sovits/G_0.pth").kind, CkptKind::Base);
        assert_eq!(by("sovits/diffusion/model_0.pt").kind, CkptKind::Base);
        assert_eq!(by("sovits/G_15473.pth").step, Some(15473));
        assert_eq!(by("sovits/diffusion/model_392.pt").kind, CkptKind::Resumable);
        // release snapshots are never resumable, and best is called out
        assert_eq!(by("rvc/weights/m_e6_s1580.pth").kind, CkptKind::Release);
        assert_eq!(by("rvc/weights/m_e6_s1580.pth").step, Some(1580));
        assert_eq!(by("rvc/weights/m_best.pth").kind, CkptKind::Best);
        // family filter
        assert!(scan_project_ckpts(&data, id, Some("vocoder")).iter().all(|r| r.family == "vocoder"));

        // ledger: an absolute path is stored RELATIVE, so a data-dir move keeps matching
        assert!(!by("rvc/weights/m_best.pth").imported);
        record_export(&data, id, "MyModel", "rvc", &rvc.join("weights").join("m_best.pth").to_string_lossy()).unwrap();
        let all2 = scan_project_ckpts(&data, id, None);
        assert!(all2.iter().find(|r| r.rel == "rvc/weights/m_best.pth").unwrap().imported);
        assert_eq!(read_meta(&data, id).unwrap().exported[0].from_ckpt_rel, "rvc/weights/m_best.pth");

        let _ = std::fs::remove_dir_all(data);
    }

    /// Diagnostic against THIS machine's real training data — the synthetic fixtures above
    /// cannot cover naming the four toolchains actually produce (CJK model names, slugs that
    /// themselves contain `_s<digits>`, pre-S39 workspaces). Run it after touching the scanner:
    ///   cargo test --lib scan_this_machine -- --ignored --nocapture
    #[test]
    #[ignore]
    fn scan_this_machine() {
        let data = PathBuf::from("D:/MyDev/Utai_v2-dev/data");
        for m in list_projects(&data) {
            let recs = scan_project_ckpts(&data, &m.id, None);
            println!("
=== {} ({}) — {} ckpt(s)", m.id, m.name, recs.len());
            for r in recs.iter().take(60) {
                println!(
                    "  {:<10} {:>10} {:>9}MB  imported={}  {}",
                    format!("{:?}", r.kind),
                    r.step.map(|s| s.to_string()).unwrap_or_else(|| "latest".into()),
                    r.bytes / 1_000_000,
                    r.imported,
                    r.rel
                );
            }
            // every record must round-trip to a file that exists, or the list is lying
            for r in &recs {
                assert!(Path::new(&r.path).is_file(), "phantom record: {}", r.path);
                assert!(!r.rel.contains('\\'), "rel must be forward-slashed: {}", r.rel);
            }
        }
    }

    /// Diagnostic twin of `scan_this_machine` for the batch-4 LISTING, against this machine's
    /// real projects. The synthetic fixtures cannot exercise CJK display names, a 40 GB walk,
    /// or the cache surviving a second call. Run after touching the listing or the cache:
    ///   cargo test --lib list_this_machine -- --ignored --nocapture
    ///
    /// It writes `<app_dir>/training-projects.json` — deliberately, that is half of what it
    /// verifies (the file must NOT appear under `<data>/training`, where `migrate_data_dir`
    /// would verify it by byte length and abort on a concurrent rewrite).
    #[test]
    #[ignore]
    fn list_this_machine() {
        let app = PathBuf::from("D:/MyDev/Utai_v2-dev");
        let data = app.join("data");
        let t0 = std::time::Instant::now();
        let rows = list_project_summaries(&app, &data, true);
        println!("\n{} project(s), measured in {:?}", rows.len(), t0.elapsed());
        for r in &rows {
            println!(
                "  {:<26} {:>9}MB  data={:>7}MB  [{}]{}{}  {}",
                r.id,
                r.sizes.total_bytes / 1_000_000,
                r.sizes.dataset_bytes / 1_000_000,
                r.families.join("+"),
                if r.missing { " MISSING" } else { "" },
                r.needs_attention.as_deref().map(|s| format!(" ⚠{s}")).unwrap_or_default(),
                r.name
            );
        }
        // the cache belongs beside config.json, NEVER inside a migrated subtree
        assert!(app.join(PROJECTS_INDEX).is_file());
        assert!(!training_root(&data).join(PROJECTS_INDEX).exists());
        assert!(!training_root(&data).join("projects.json").exists());
        // every non-missing row must describe a directory that is really there, and the sizes
        // must be a cache HIT the second time (same figures, no re-walk)
        for r in &rows {
            if r.missing {
                continue;
            }
            assert!(project_dir(&data, &r.id).is_dir(), "phantom project: {}", r.id);
            assert!(r.sizes.computed_ms > 0, "unmeasured after refresh: {}", r.id);
        }
        let t1 = std::time::Instant::now();
        let cached = list_project_summaries(&app, &data, false);
        println!("cached listing in {:?}", t1.elapsed());
        assert_eq!(cached.len(), rows.len());
        for (a, b) in cached.iter().zip(rows.iter()) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.sizes.total_bytes, b.sizes.total_bytes, "cache disagrees for {}", a.id);
        }
    }

    fn ck(rel: &str, kind: CkptKind, mtime_ms: u64, imported: bool) -> CkptRecord {
        CkptRecord {
            run_id: crate::training::trun::run_id_in_rel(rel, "rvc"),
            rel: rel.into(),
            path: format!("D:/x/{rel}"),
            family: "rvc".into(),
            kind,
            step: None,
            bytes: 100,
            mtime_ms,
            imported,
            companions: Vec::new(),
        }
    }

    /// ★S118 §F8-res⒈ — the project card must name the step a DEFAULT 续训 continues from.
    ///
    /// The bug this pins was introduced by S117 and lived on the GAN side too: `resume_best/` is
    /// written after the rolling pair, so it becomes the mtime-newest Resumable and the card
    /// started printing the BEST step — a number the「续训」button will not continue from.
    #[test]
    fn s118_the_project_card_names_the_step_a_default_resume_uses() {
        let mut best = ck("sovits/resume_best/G.pth", CkptKind::Resumable, 9_000_500, false);
        best.step = Some(1400);
        let mut latest = ck("sovits/G_5000.pth", CkptKind::Resumable, 9_000_000, false);
        latest.step = Some(5000);
        // newest-first by mtime, exactly as `scan_project_ckpts` returns it: best is NEWER.
        let recs = vec![best.clone(), latest.clone()];
        assert_eq!(
            default_resume_record(&recs).and_then(|r| r.step),
            Some(5000),
            "the mtime-newest Resumable is the BEST snapshot — the card must still name the latest"
        );
        // ⚠ Companion arm, or the assertion above would also pass if the function just returned
        // the max step for unrelated reasons: with the best snapshot ABSENT nothing changes.
        assert_eq!(
            default_resume_record(&[latest.clone()]).and_then(|r| r.step),
            Some(5000)
        );
        // …and a slot whose ONLY resumable record IS the best snapshot must still report it:
        // that really is the only thing there to continue from.
        assert_eq!(default_resume_record(&[best]).and_then(|r| r.step), Some(1400));
        // Non-resumable kinds are never a resume point, however new they are.
        assert!(default_resume_record(&[ck("rvc/weights/m_best.pth", CkptKind::Best, 9_9, false)])
            .is_none());
    }

    /// The most expensive judgement in the whole refactor: getting it wrong deletes hours of a
    /// user's work. Every protection is pinned, and so are the two rules that were REMOVED.
    #[test]
    fn cleanup_protects_everything_that_anyone_could_still_want() {
        let none = |_: &str| false;
        let now = 10_000_000u64;
        let ledger = 5_000_000u64;
        let recs = vec![
            // structurally excluded — a cleanup must never touch these
            ck("rvc/G_500.pth", CkptKind::Resumable, 9_000_000, false),
            ck("rvc/G_0.pth", CkptKind::Base, 9_000_000, false),
            ck("rvc/weights/m_best.pth", CkptKind::Best, 9_000_000, false),
            ck("rvc/weights/m.pth", CkptKind::Final, 9_000_000, false),
            // the actual candidates
            ck("rvc/weights/m_e1_s100.pth", CkptKind::Release, 9_000_000, false),
            ck("rvc/weights/m_e2_s200.pth", CkptKind::Release, 9_000_000, true), // in the ledger
            ck("rvc/weights/old_e3_s300.pth", CkptKind::Release, 4_000_000, false), // pre-ledger
            ck("rvc/weights/m_e9_s900.pth", CkptKind::Release, now - 1_000, false), // seconds old
            ck("rvc/G_777.pth", CkptKind::Orphan, 9_000_000, false),
        ];
        let plan = plan_cleanup(&recs, ledger, now, &none);
        let deleted: Vec<&str> = plan.delete.iter().map(|r| r.rel.as_str()).collect();
        assert_eq!(deleted, vec!["rvc/weights/m_e1_s100.pth", "rvc/G_777.pth"]);
        let why = |rel: &str| plan.kept.iter().find(|(r, _)| r == rel).map(|(_, w)| *w);
        assert_eq!(why("rvc/G_500.pth"), Some(KeptReason::NotASnapshot), "resumability is sacred");
        assert_eq!(why("rvc/weights/m_best.pth"), Some(KeptReason::NotASnapshot));
        assert_eq!(
            why("rvc/weights/m.pth"),
            Some(KeptReason::NotASnapshot),
            "the naturally-finished export is not a periodic snapshot"
        );
        assert_eq!(why("rvc/weights/m_e2_s200.pth"), Some(KeptReason::Exported));
        assert_eq!(
            why("rvc/weights/old_e3_s300.pth"),
            Some(KeptReason::PreLedger),
            "a MIGRATED project cannot know what was imported — everything older is untouchable"
        );
        assert_eq!(why("rvc/weights/m_e9_s900.pth"), Some(KeptReason::JustWritten));
        assert_eq!(plan.freeable_bytes, 200);

        // ★ a snapshot whose stem is an installed model survives even with NO ledger row —
        // the ledger can be thin (imports predating S76, a torn write), the registry cannot lie
        let installed = |stem: &str| stem == "m_e1_s100";
        let plan2 = plan_cleanup(&recs, ledger, now, &installed);
        assert_eq!(
            plan2.kept.iter().find(|(r, _)| r == "rvc/weights/m_e1_s100.pth").map(|(_, w)| *w),
            Some(KeptReason::StillInstalled)
        );
    }

    /// ★ The two rules that were in the first draft and had to go. `snapshot.ckpts` accumulates
    /// and is not cleared on completion, so "protect the current run's candidates" protected
    /// EXACTLY the set a user wants to delete right after training — the feature would have
    /// released 0 bytes in its main scenario while looking like it worked.
    #[test]
    fn cleanup_actually_deletes_in_its_main_scenario() {
        let now = 10_000_000u64;
        // just finished a run: 19 periodic snapshots, all from this run, none imported
        let recs: Vec<CkptRecord> = (1..=19)
            .map(|i| ck(&format!("rvc/weights/m_e{i}_s{}.pth", i * 100), CkptKind::Release, now - 600_000, false))
            .collect();
        let plan = plan_cleanup(&recs, 1_000, now, &|_| false);
        assert_eq!(plan.delete.len(), 19, "this is the whole point of the feature");
        assert_eq!(plan.freeable_bytes, 1900);
    }

    /// ⛔ The stale-ledger tripwire compares a PROJECT-wide ledger against a FAMILY-filtered
    /// record set, and every caller names a family. A project that exported from one slot could
    /// therefore never clean another one — a hard modal, permanently, on a healthy ledger.
    ///
    /// Both directions are asserted here: the tripwire must stop firing on the innocent slot AND
    /// must still fire when the rows it is actually responsible for have vanished. Asserting only
    /// the first would pass just as well with the tripwire deleted.
    #[test]
    fn the_stale_ledger_tripwire_is_scoped_to_the_family_being_cleaned() {
        let data = tmp_root("ledger");
        let id = "led_11112222";
        let p = project_dir(&data, id);
        let rvc_rel = "rvc/weights/m_e1_s100.pth";
        for rel in [rvc_rel, "sovits/weights/s_e2_s200.pth"] {
            let f = p.join(rel);
            std::fs::create_dir_all(f.parent().unwrap()).unwrap();
            std::fs::write(&f, b"x").unwrap();
        }
        let meta = ProjectMeta {
            id: id.into(),
            name: "n".into(),
            // 1 ⇒ nothing is PreLedger-protected, so the plan is decided by the rules under test
            export_ledger_since_ms: 1,
            exported: vec![ExportedModel {
                name: "m".into(),
                model_type: "rvc".into(),
                from_ckpt_rel: rvc_rel.into(),
                at_ms: 1,
            }],
            ..Default::default()
        };
        write_meta(&data, &meta).unwrap();
        let none = |_: &str| false;

        // the innocent slot: no ledger row belongs to it, so there is nothing to be stale
        cleanup_snapshots(&data, id, Some("sovits"), &none)
            .expect("cleaning a family the ledger says nothing about must not read as stale");
        // …and the family that DOES own a row is fine while the file is there
        cleanup_snapshots(&data, id, Some("rvc"), &none).expect("its own file is on disk");
        // project-wide is fine too
        cleanup_snapshots(&data, id, None, &none).expect("at least one row matches");

        // now make that row's file vanish behind our back — the tripwire must fire, for the
        // family that owns it and project-wide, and only there.
        std::fs::remove_file(p.join(rvc_rel)).unwrap();
        for scope in [Some("rvc"), None] {
            let err = cleanup_snapshots(&data, id, scope, &none)
                .expect_err("a ledger row with no file on disk is exactly what this guards");
            assert!(err.to_string().contains("PROJECT_LEDGER_STALE"), "{err}");
        }
        cleanup_snapshots(&data, id, Some("sovits"), &none)
            .expect("still none of sovits' business");

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

    // ───────────────────── batch 4: explicit CRUD + the listing cache ─────────────────────

    fn app_root(data: &Path) -> PathBuf {
        let a = data.join("app");
        std::fs::create_dir_all(&a).unwrap();
        a
    }

    #[test]
    fn create_project_validates_and_refuses_duplicate_names() {
        let data = tmp_root("create");
        let m = create_project(&data, "  歌姫テスト  ", " a note ").unwrap();
        assert_eq!(m.name, "歌姫テスト", "trimmed");
        assert_eq!(m.note, "a note");
        assert!(m.export_ledger_since_ms > 0, "every project is stamped or cleanup refuses it");
        assert!(read_meta(&data, &m.id).is_some());

        for bad in ["", "   ", "a\nb", "a\u{0}b"] {
            assert!(create_project(&data, bad, "").is_err(), "must refuse {bad:?}");
        }
        assert!(create_project(&data, &"x".repeat(81), "").is_err(), "length cap");
        assert!(create_project(&data, "ok", &"n".repeat(501)).is_err(), "note cap");
        // A name that already resolves is refused — `find_by_name` picks duplicates by
        // directory order, so a second one would make BOTH unaddressable.
        assert!(create_project(&data, "歌姫テスト", "").is_err());
        assert!(create_project(&data, " 歌姫テスト", "").is_err(), "after trimming, too");
        let _ = std::fs::remove_dir_all(data);
    }

    /// create "A" → rename to "B" → create "A" again. The deterministic id for "A" is now taken
    /// by a project that answers to "B", so a naive `new_project_id` would collide with it.
    #[test]
    fn create_project_places_beside_a_renamed_project_that_owns_its_id() {
        let data = tmp_root("collide");
        let first = create_project(&data, "A", "").unwrap();
        update_project(&data, &first.id, "B", "").unwrap();
        assert_eq!(read_meta(&data, &first.id).unwrap().name, "B");
        assert_eq!(first.id, new_project_id("A"), "precondition: id derives from the name");

        let second = create_project(&data, "A", "").unwrap();
        assert_ne!(second.id, first.id, "must not reuse the occupied directory");
        // the shape must survive uniquifying — `<ascii>_<8 hex>` is what keeps a project id from
        // ever being a Windows reserved device name
        let (base, hex) = second.id.rsplit_once('_').unwrap();
        assert!(!base.is_empty() && hex.len() == 8 && hex.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(read_meta(&data, &first.id).unwrap().name, "B", "untouched");
        assert_eq!(read_meta(&data, &second.id).unwrap().name, "A");
        let _ = std::fs::remove_dir_all(data);
    }

    /// The opposite case: the directory is there but its `project.json` is unreadable. That may
    /// be the user's damaged project, and a second directory beside it would leave the real one
    /// reachable from nowhere — refuse instead (same posture as `resolve_or_create`).
    #[test]
    fn create_project_refuses_when_an_unreadable_project_holds_the_id() {
        let data = tmp_root("damaged");
        let id = new_project_id("A");
        std::fs::create_dir_all(project_dir(&data, &id)).unwrap();
        std::fs::write(project_dir(&data, &id).join(PROJECT_META), b"{ truncated").unwrap();
        let e = create_project(&data, "A", "").unwrap_err().to_string();
        assert!(e.contains("PROJECT_META_UNREADABLE"), "got {e}");
        let _ = std::fs::remove_dir_all(data);
    }

    /// ★§F2⒝ 批 2 ④b —— 改名**只改标签**。
    ///
    /// The assertion that carries the batch: `model_slug` must come out unchanged. It is
    /// `hps.name`, the `weights/<slug>*` prefix, the `audition/<slug>_*` stems and — through the
    /// pool the runs SHARE — the `dataset_44k/<slug>/` slice directory. A rename that moved it
    /// would orphan every existing product and grow a second full preprocessing tree on the run's
    /// next start, with nothing anywhere reporting it.
    ///
    /// ⚠ 「其余键不变」is asserted on the PARSED object, not on bytes: the file is rewritten
    /// pretty-printed, so a byte comparison would fail for a reason that has nothing to do with
    /// the property under test.
    #[test]
    fn renaming_a_run_moves_the_label_and_nothing_else() {
        let data = tmp_root("runrename");
        let ws = legacy_rvc(&data, "slugdir");
        // a realistic run.json: the label, the frozen artifact identity, and the absolute asset
        // paths a rewrite must not lose
        let before = serde_json::json!({
            "model_name": "初号机",
            "model_slug": "LEGACY-STEM",
            "backend": "rvc",
            "assets": { "rmvpe_pt": "C:\\x\\rmvpe.pt" },
            "total_epoch": 200,
            "fp16": true,
        });
        std::fs::write(ws.join("run.json"), serde_json::to_vec_pretty(&before).unwrap()).unwrap();
        let run = crate::training::trun::RunDir::for_test(ws.clone());

        rename_run(&run, "改了个名字").unwrap();

        assert_eq!(run_model_name(&run).as_deref(), Some("改了个名字"));
        assert_eq!(
            run_artifact_slug(&run).as_deref(),
            Some("LEGACY-STEM"),
            "★ the artifact identity must not move — that is the whole reason a rename is allowed"
        );
        let after: serde_json::Value =
            serde_json::from_slice(&std::fs::read(ws.join("run.json")).unwrap()).unwrap();
        for (k, v) in before.as_object().unwrap() {
            if k == "model_name" {
                continue;
            }
            assert_eq!(after.get(k), Some(v), "key {k} changed");
        }
        assert_eq!(
            after.as_object().unwrap().len(),
            before.as_object().unwrap().len(),
            "the rewrite added or dropped a key"
        );
        // the temp file must not survive: `run.json.tmp` sitting in a run directory is invisible
        // to every scanner but counts toward `dir_size`
        assert!(!ws.join("run.json.tmp").exists());

        // a run that never started has no label to change — and no artifacts to orphan either
        let fresh = tmp_root("runrename2");
        let empty = crate::training::trun::RunDir::for_test(training_root(&fresh).join("never"));
        std::fs::create_dir_all(empty.path()).unwrap();
        let e = rename_run(&empty, "x").unwrap_err().to_string();
        assert!(e.contains("RUN_NEVER_NAMED"), "got {e}");

        let _ = std::fs::remove_dir_all(data);
        let _ = std::fs::remove_dir_all(fresh);
    }

    #[test]
    fn rename_keeps_the_directory_and_every_artifact_path() {
        let data = tmp_root("rename");
        let m = create_project(&data, "before", "").unwrap();
        let ckpt = family_dir(&data, &m.id, "rvc").join("weights").join("x_e1_s10.pth");
        std::fs::create_dir_all(ckpt.parent().unwrap()).unwrap();
        std::fs::write(&ckpt, b"w").unwrap();

        update_project(&data, &m.id, "after", "note").unwrap();
        assert!(ckpt.is_file(), "a rename must never move a checkpoint");
        assert_eq!(read_meta(&data, &m.id).unwrap().name, "after");
        assert_eq!(find_by_name(&data, "after").unwrap().id, m.id);
        assert!(find_by_name(&data, "before").is_none());
        // renaming onto an existing name is refused; renaming to its OWN name is fine
        let other = create_project(&data, "taken", "").unwrap();
        assert!(update_project(&data, &m.id, "taken", "").is_err());
        assert!(update_project(&data, &m.id, "after", "still fine").is_ok());
        assert!(update_project(&data, "no_such_id", "x", "").is_err());
        assert_eq!(read_meta(&data, &other.id).unwrap().name, "taken");
        let _ = std::fs::remove_dir_all(data);
    }

    #[test]
    fn slot_model_name_comes_from_the_slots_own_run_json() {
        let data = tmp_root("slotname");
        let m = create_project(&data, "proj", "").unwrap();
        let rvc = family_dir(&data, &m.id, "rvc");
        std::fs::create_dir_all(&rvc).unwrap();
        // a slot that never completed an import has no run.json — and therefore no artifact
        // carrying a slug, which is why nothing needs back-filling
        assert_eq!(slot_model_name(&data, &m.id, "rvc"), None);
        std::fs::write(rvc.join("run.json"), r#"{"model_name":"旧的名字"}"#).unwrap();
        assert_eq!(slot_model_name(&data, &m.id, "rvc").as_deref(), Some("旧的名字"));
        // empty / missing / malformed all read as「没有」rather than as an empty name
        std::fs::write(rvc.join("run.json"), r#"{"model_name":""}"#).unwrap();
        assert_eq!(slot_model_name(&data, &m.id, "rvc"), None);
        std::fs::write(rvc.join("run.json"), b"{{{").unwrap();
        assert_eq!(slot_model_name(&data, &m.id, "rvc"), None);
        assert_eq!(slot_model_name(&data, &m.id, "sovits"), None);
        let _ = std::fs::remove_dir_all(data);
    }

    #[test]
    fn listing_caches_sizes_and_remembers_a_vanished_project() {
        let data = tmp_root("listing");
        let app = app_root(&data);
        let m = create_project(&data, "P", "").unwrap();
        std::fs::create_dir_all(family_dir(&data, &m.id, "rvc")).unwrap();
        std::fs::write(family_dir(&data, &m.id, "rvc").join("G_1.pth"), vec![7u8; 500]).unwrap();
        std::fs::create_dir_all(dataset_dir(&data, &m.id)).unwrap();
        std::fs::write(dataset_dir(&data, &m.id).join("000.wav"), vec![1u8; 300]).unwrap();

        let rows = list_project_summaries(&app, &data, true);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].families, vec!["rvc".to_string()]);
        assert!(rows[0].has_dataset);
        assert_eq!(rows[0].sizes.dataset_bytes, 300);
        assert_eq!(rows[0].sizes.family_bytes.get("rvc"), Some(&500));
        assert!(rows[0].sizes.total_bytes >= 800);
        assert!(rows[0].sizes.computed_ms > 0 && !rows[0].missing);

        // the cache lives OUTSIDE the data dir: `training` is a MIGRATED_SUBTREE and
        // `migrate_data_dir` verifies it file-by-file by byte length, so a file we rewrite on
        // every listing would abort the user's whole data-dir migration.
        assert!(app.join(PROJECTS_INDEX).is_file());
        assert!(!training_root(&data).join(PROJECTS_INDEX).exists());
        assert!(!training_root(&data).join("projects.json").exists());

        // a cached listing does not re-walk, but still answers with the stored figures
        std::fs::write(family_dir(&data, &m.id, "rvc").join("G_2.pth"), vec![7u8; 1000]).unwrap();
        let cached = list_project_summaries(&app, &data, false);
        assert_eq!(cached[0].sizes.family_bytes.get("rvc"), Some(&500), "cache, not a re-walk");
        let fresh = list_project_summaries(&app, &data, true);
        assert_eq!(fresh[0].sizes.family_bytes.get("rvc"), Some(&1500));

        // the directory disappears behind our back → the row survives, named, and barred
        crate::util::remove_dir_all_robust(&project_dir(&data, &m.id)).unwrap();
        let gone = list_project_summaries(&app, &data, false);
        assert_eq!(gone.len(), 1);
        assert!(gone[0].missing && gone[0].name == "P" && gone[0].families.is_empty());
        assert_eq!(
            gone[0].sizes.total_bytes, fresh[0].sizes.total_bytes,
            "last known size still tells the user how much disk this WAS using"
        );
        // …and「移除记录」makes it go away for good
        forget_project(&app, &data, &m.id);
        assert!(list_project_summaries(&app, &data, false).is_empty());
        let _ = std::fs::remove_dir_all(data);
    }

    #[test]
    fn listing_cache_is_discarded_when_the_data_root_changes() {
        let data_a = tmp_root("rootA");
        let data_b = tmp_root("rootB");
        let app = app_root(&data_a);
        let m = create_project(&data_a, "only-in-A", "").unwrap();
        assert_eq!(list_project_summaries(&app, &data_a, true).len(), 1);
        // Same app, different data root: A's projects must NOT show up as B's ghosts.
        assert!(list_project_summaries(&app, &data_b, true).is_empty());
        // and switching back rebuilds from disk rather than from the discarded rows
        let back = list_project_summaries(&app, &data_a, true);
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].id, m.id);
        let _ = std::fs::remove_dir_all(data_a);
        let _ = std::fs::remove_dir_all(data_b);
    }

    /// The wire shape the TS mirror is written against. Nothing on the frontend can catch a
    /// rename here — `invoke` is stringly typed — so pin it: the flattened `ProjectSizes` must
    /// land as top-level camelCase keys, not nested under `sizes`.
    #[test]
    fn project_summary_serializes_flat_and_camel_cased() {
        let data = tmp_root("wire");
        let app = app_root(&data);
        create_project(&data, "P", "n").unwrap();
        let rows = list_project_summaries(&app, &data, true);
        let v = serde_json::to_value(&rows[0]).unwrap();
        for k in [
            "id", "name", "note", "createdMs", "updatedMs", "needsAttention", "families",
            "hasDataset", "missing", "totalBytes", "datasetBytes", "familyBytes", "computedMs",
        ] {
            assert!(v.get(k).is_some(), "missing wire key {k}: {v}");
        }
        assert!(v.get("sizes").is_none(), "ProjectSizes must be FLATTENED, not nested");
        let _ = std::fs::remove_dir_all(data);
    }

    /// A pre-S76 workspace the migration has not folded yet has no `project.json`, so nothing
    /// else in the listing path would see it. It must still be a row:「盘上还在、app 里没了」is
    /// the one outcome this whole refactor exists to prevent.
    #[test]
    fn an_unmigrated_workspace_is_listed_as_pending_not_dropped() {
        let data = tmp_root("pendinglist");
        let app = app_root(&data);
        legacy_rvc(&data, "old_1a2b3c4d"); // no project.json
        create_project(&data, "P", "").unwrap();
        // decoys the listing must ignore: a tombstone, and the cache file itself in case someone
        // ever puts it back into the training root. (A `.mig_<id>` staging tree is deliberately
        // NOT one of them — that is a torn migration, which `migrate_one` reconciles; the
        // migration tests own that case.)
        std::fs::create_dir_all(training_root(&data).join(".del_x_1_2")).unwrap();
        std::fs::write(training_root(&data).join(PROJECTS_INDEX), b"{}").unwrap();

        let rows = list_project_summaries(&app, &data, true);
        assert_eq!(rows.len(), 2, "the legacy workspace + the real project, nothing else: {rows:?}");
        let old = rows.iter().find(|r| r.id == "old_1a2b3c4d").unwrap();
        assert_eq!(old.needs_attention.as_deref(), Some("TRAINING_LAYOUT_MIGRATION_PENDING"));
        assert!(!old.missing, "it IS on disk — just not folded yet");
        assert!(old.families.is_empty());

        // …and once it migrates it becomes an ordinary row, with no leftover ghost beside it
        migrate_all(&data);
        let after = list_project_summaries(&app, &data, true);
        assert_eq!(after.len(), 2);
        let now = after.iter().find(|r| r.id == "old_1a2b3c4d").unwrap();
        assert!(now.needs_attention.is_none() && !now.missing);
        assert_eq!(now.name, "歌姫テスト", "display name recovered by the migration");
        assert_eq!(now.families, vec!["rvc".to_string()]);
        let _ = std::fs::remove_dir_all(data);
    }

    /// A corrupt or future-version cache must degrade to「重新量一次」, never to a broken page.
    #[test]
    fn a_broken_cache_rebuilds_instead_of_failing_the_listing() {
        let data = tmp_root("badcache");
        let app = app_root(&data);
        create_project(&data, "P", "").unwrap();
        std::fs::write(app.join(PROJECTS_INDEX), b"{ not json").unwrap();
        assert_eq!(list_project_summaries(&app, &data, false).len(), 1);
        std::fs::write(app.join(PROJECTS_INDEX), br#"{"version":999,"projects":{"x":{}}}"#).unwrap();
        let rows = list_project_summaries(&app, &data, false);
        assert_eq!(rows.len(), 1, "a future version's rows are not ours to interpret");
        assert!(!rows[0].missing);
        let _ = std::fs::remove_dir_all(data);
    }
}
