//! Slot layout v2: the preprocessing POOL container.
//!
//! ## What changed and why
//!
//! Until now a family slot (`<data>/training/<project_id>/<family>/`) held two very different
//! kinds of thing side by side at its root:
//!
//! * **pool products** — slices, f0, ContentVec features, npz, `dataset_44k/`: determined ONLY
//!   by the dataset and the preprocessing parameters, and costing HOURS to rebuild;
//! * **run products** — checkpoints, `weights/`, the resume sidecars, logs: determined by the
//!   weights.
//!
//! Because they shared one directory, the only way python could react to a preprocessing
//! parameter change was `shutil.rmtree` (`utai_train/cache.py`): flip `loudnorm`, lose the
//! slices; flip it back, pay for them again. That is the whole of §F2⒝'s first half.
//!
//! The pool now lives one level down, in a directory named after its own identity:
//!
//! ```text
//! <slot>/
//!   slot.json              ← {"layout": 2} — the migration commit point
//!   pools/<pool_id>/
//!     dataset.fingerprint  ← THE identity. Same file python has always written.
//!     …pool products…
//!   …run products, exactly where they have always been…
//! ```
//!
//! **The pool boundary is byte-for-byte today's cache-invalidation boundary.** Nothing about
//! WHICH artifacts share an identity changed; the only change is that a non-matching pool is now
//! a SIBLING instead of a deletion. Keeping that statement true is what makes this batch
//! reviewable, and it is why no identity formula was touched (see [`POOL_ENTRIES`]).
//!
//! ## Why run products do not move in this batch
//!
//! Moving them requires re-pointing `scan_project_ckpts` (non-recursive, six hard-coded
//! sub-paths), the whole `has_main_progress` / `max_vocoder_ckpt_step` / `max_diffusion_step`
//! family, `project.json`'s project-relative export ledger, the audition cache, and two frontend
//! checkpoint lists — every one of which fails SILENTLY (an empty archive list, a wipe with no
//! dialog, a cleanup that deletes what the ledger existed to protect). That is §F2⒝'s second
//! half and it gets its own layout version.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{Result, UtaiError};

/// Container for every pool of one slot.
///
/// ⚠ The name is load-bearing in three ways, all of them silent if broken:
/// * it must not be a FAMILY name — `tproject::has_family_slot` decides "old or new project
///   layout" by looking for a subdirectory named exactly after a family;
/// * it must not contain `.wav` — `sovits/extract.py` and both `data_utils.py` derive the spec
///   path with `filename.replace(".wav", ".spec.pt")` on the WHOLE PATH;
/// * it must not start with `.` — every slot scan (`tproject.rs`) skips dot entries, which are
///   reserved for staging (`.mig_`, `.del_`, and [`STAGING_PREFIX`] below).
pub const POOLS_DIR: &str = "pools";

/// `<slot>/slot.json` — written LAST, so its presence is the commit point (same protocol as
/// `project.json` one level up).
pub const SLOT_META: &str = "slot.json";

/// The identity file. python has written this since S38 (`utai_train/cache.py`); layout 2 simply
/// moves it into the pool it describes, where it becomes that pool's name-independent identity.
pub const FINGERPRINT: &str = "dataset.fingerprint";

/// Current slot layout. Bumped when the meaning of the directory changes, never for content.
pub const SLOT_LAYOUT: u32 = 2;

/// Staging directory for a migration in flight. Dot-prefixed so every existing scan ignores it —
/// a half-filled pool must never be readable as a pool.
const STAGING_PREFIX: &str = ".mig_pool_";

/// The staging prefix, for the verifier that builds torn states from outside this module.
///
/// Exposed rather than duplicated: enumerating the crash points means creating exactly the
/// half-migrated shapes this module leaves behind, and a second copy of this string in the
/// verifier would let the two drift until the "crash recovery" leg silently tested nothing.
pub fn staging_prefix() -> &'static str {
    STAGING_PREFIX
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SlotMeta {
    pub layout: u32,
    /// Forward compatibility, same reasoning as `ProjectMeta::extra`: a downgraded build must not
    /// silently drop fields a newer build wrote.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

// ─────────────────────────── the decision table ───────────────────────────

/// ONE top-level slot entry that belongs to the POOL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolEntry {
    /// Family that produces it, or `"*"` for every family.
    pub family: &'static str,
    pub name: &'static str,
}

const fn e(family: &'static str, name: &'static str) -> PoolEntry {
    PoolEntry { family, name }
}

/// ⛔ **THE decision table.** Every top-level entry of a slot that moves into the pool.
///
/// This list is not documentation — `gate_pool_table.py` drives it against the python sources, so
/// adding a preprocessing product without adding it here turns that gate red. That is deliberate:
/// the previous generation of this contract was a prose comment in `tproject.rs` claiming to
/// enumerate "the complete set of subdirectories anything has ever created inside a workspace
/// root", and it was ALREADY missing `resume_best/`, `resume_latest/`, `eval/` and
/// `lightning_logs/` when this batch found it. A claim of completeness that nothing checks decays
/// the moment someone adds a directory (S120 §F9).
///
/// ## Why some pool-shaped artifacts are deliberately NOT here
///
/// `total_fea.npy`, `cluster/`, `filelist.txt`, `filelists/` and `aug_gate_report*.json` are all
/// derived from the pool, and one could argue they belong in it. They stay at the slot root, for
/// two independent reasons:
///
/// 1. **They are rebuilt unconditionally by every run anyway** (`rvc/index_npy.py` has no
///    skip-if-exists, `sovits/flist.py` reopens the lists with `"w"`), so pooling them saves no
///    work — only duplicate copies.
/// 2. **Two of them are read at fixed slot-relative names by the PUBLISH chain** —
///    `commands/training.rs`'s `get_slot_export_context` and its frontend twin in
///    `TrainingPage.tsx` probe `total_fea.npy` and `cluster/*`. Both fail OPEN (the model
///    installs with no retrieval asset, no error, no log), so moving them is a silent
///    inference-quality regression, and it buys nothing per (1).
///
/// `cluster/` additionally depends on the per-run `kmeans` request flag, so it is not a pure
/// function of the pool at all.
pub const POOL_ENTRIES: &[PoolEntry] = &[
    // ── rvc ──────────────────────────────────────────────────────────────────────────────
    e("rvc", "0_gt_wavs"),
    e("rvc", "1_16k_wavs"),
    e("rvc", "2a_f0"),
    e("rvc", "2b-f0nsf"),
    e("rvc", "3_feature256"),
    e("rvc", "3_feature768"),
    // an asset COPY, keyed by sample rate and feature dim, that the DataLoader also writes a
    // `.spec.pt` into (`rvc/filelist.py` explains why it is copied at all: the install dir may be
    // read-only). Pool-shaped, and its absolute path is written into `filelist.txt`.
    e("rvc", "mute"),
    e("rvc", "aug_meta"),
    // ── sovits / sovits_v2 (sovits_diff shares the sovits slot BY DESIGN) ────────────────
    e("sovits", "dataset_44k"),
    e("sovits", "aug_meta"),
    e("sovits_v2", "dataset_44k"),
    e("sovits_v2", "aug_meta"),
    // ── vocoder ─────────────────────────────────────────────────────────────────────────
    e("vocoder", "slices"),
    e("vocoder", "npz"),
    e("vocoder", "aug_meta"),
    // ── every family ────────────────────────────────────────────────────────────────────
    e("*", FINGERPRINT),
];

/// Does this top-level slot entry move into the pool?
pub fn is_pool_entry(family: &str, name: &str) -> bool {
    POOL_ENTRIES
        .iter()
        .any(|p| (p.family == "*" || p.family == family) && p.name == name)
}

/// Every pool entry name one family can produce, for gates and for the migrator's dry run.
pub fn pool_entries_for(family: &str) -> Vec<&'static str> {
    POOL_ENTRIES
        .iter()
        .filter(|p| p.family == "*" || p.family == family)
        .map(|p| p.name)
        .collect()
}

// ─────────────────────────── paths and identity ───────────────────────────

pub fn pools_root(slot: &Path) -> PathBuf {
    slot.join(POOLS_DIR)
}

fn slot_meta_path(slot: &Path) -> PathBuf {
    slot.join(SLOT_META)
}

/// `p<12 hex>` derived from the identity text.
///
/// Deriving the NAME from the identity is safe here even though the identity FORMULA may evolve,
/// because selection is by CONTENT (`dataset.fingerprint` inside the directory), never by name:
/// a pool whose name was derived by an older formula keeps working, it simply stops being the
/// name a new run would mint. What the derivation buys is that Rust (migration) and python (new
/// pools) agree on a name without having to coordinate, and that two distinct identities can
/// never propose the same directory.
///
/// ⚠ Charset is `[p0-9a-f]` on purpose — no `.`, no `wav`, no `spec` — so every substring
/// operation the training code performs on paths is structurally unable to see it. See
/// [`POOLS_DIR`] for the three rules and where each one is enforced.
pub fn pool_id_for(fp_text: &str) -> String {
    use sha2::{Digest, Sha256};
    let d = Sha256::digest(fp_text.as_bytes());
    let mut s = String::with_capacity(13);
    s.push('p');
    for b in d.iter().take(6) {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

#[derive(Debug, Clone)]
pub struct PoolInfo {
    pub id: String,
    pub dir: PathBuf,
    /// Content of `dataset.fingerprint`, trimmed. Empty when the file is missing or unreadable —
    /// such a pool is never MATCHED (it would be a guess), but it is still listed so its bytes
    /// stay visible and reclaimable.
    pub fp_text: String,
}

/// Every pool of one slot, sorted by id so listings never wobble.
pub fn list_pools(slot: &Path) -> Vec<PoolInfo> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(pools_root(slot)) else {
        return out;
    };
    for entry in rd.flatten() {
        let id = entry.file_name().to_string_lossy().into_owned();
        // `.` entries are staging, never a pool.
        if id.starts_with('.') || !entry.path().is_dir() {
            continue;
        }
        let fp_text = std::fs::read_to_string(entry.path().join(FINGERPRINT))
            .map(|s| s.trim().to_string())
            .unwrap_or_default();
        out.push(PoolInfo { id, dir: entry.path(), fp_text });
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

/// Has this slot ever been preprocessed?
///
/// Replaces the old `<slot>/dataset.fingerprint` existence test, which was the ONE artifact every
/// family wrote and therefore the single portable judge. It still is — it just lives inside the
/// pool now. The legacy arm is kept by the caller (`training::workspace_holds_work`) because a
/// pre-S76 directory that has not been through either migration still has it at the root.
pub fn slot_has_pool(slot: &Path) -> bool {
    list_pools(slot).iter().any(|p| !p.fp_text.is_empty())
}

/// The fingerprint text of this slot's pool, when exactly ONE pool can answer.
///
/// Deliberately refuses to guess with more than one: the single caller is the S38-era `loudnorm`
/// backfill, whose whole point is to recover a value that was never recorded, and picking the
/// wrong pool there makes a diffusion run inherit the wrong loudness domain — which then
/// re-fingerprints and rebuilds the slices the main model is training on. Every workspace that
/// can reach that backfill was migrated from a single flat slot and therefore has exactly one
/// pool; `None` keeps the pre-existing default (`false`) for anything else.
pub fn sole_pool_fingerprint(slot: &Path) -> Option<String> {
    let pools: Vec<PoolInfo> = list_pools(slot).into_iter().filter(|p| !p.fp_text.is_empty()).collect();
    match pools.len() {
        1 => Some(pools[0].fp_text.clone()),
        _ => None,
    }
}

// ─────────────────────────── slot meta ───────────────────────────

pub fn read_slot_meta(slot: &Path) -> Option<SlotMeta> {
    let raw = std::fs::read_to_string(slot_meta_path(slot)).ok()?;
    serde_json::from_str(&raw).ok()
}

/// Atomic write (tmp + rename in the same directory) — a torn `slot.json` reads as "not
/// migrated" and would send the next boot into a second migration of an already-migrated slot.
fn write_slot_meta(slot: &Path, meta: &SlotMeta) -> Result<()> {
    let io_err = |e: std::io::Error| {
        UtaiError::Training(format!("SLOT_META_WRITE_FAILED: {}: {e}", slot.display()))
    };
    std::fs::create_dir_all(slot).map_err(io_err)?;
    let final_path = slot_meta_path(slot);
    let tmp = slot.join(format!("{SLOT_META}.tmp"));
    let body = serde_json::to_string_pretty(meta)
        .map_err(|e| UtaiError::Training(format!("SLOT_META_ENCODE_FAILED: {e}")))?;
    std::fs::write(&tmp, body).map_err(io_err)?;
    std::fs::rename(&tmp, &final_path).map_err(io_err)?;
    Ok(())
}

// ─────────────────────────── migration (layout 1 → 2) ───────────────────────────

#[derive(Debug, PartialEq, Eq)]
pub enum SlotOutcome {
    /// `slot.json` was already there, or the slot does not exist.
    AlreadyDone,
    /// Nothing to move (a slot that has never been preprocessed) — committed anyway, so the next
    /// boot does not look again.
    Committed,
    /// Pool products were folded into `pools/<id>/`.
    Migrated(String),
}

/// What the migrator would do, computed BEFORE anything is touched.
///
/// Splitting the decision out from the action is what makes the fail-safe posture checkable: the
/// plan is a pure function of the directory listing, so a test can assert the classification of a
/// real workspace without moving a byte.
#[derive(Debug, Default)]
pub struct SlotPlan {
    /// Top-level entries that move into the pool.
    pub moving: Vec<String>,
    /// Top-level entries that stay where they are. Includes everything the table does not name.
    pub staying: Vec<String>,
    /// Entries that stay and that the table does not know about. NOT an error — see below.
    pub unknown: Vec<String>,
}

/// Classify a slot's top-level entries.
///
/// ## Why an unknown entry does NOT abort
///
/// The instinct (and this repo's usual posture) is fail-closed: refuse, flag, touch nothing. Here
/// the fail-SAFE choice is strictly better, because "do not move it" leaves the entry exactly
/// where it is today. A stray `Thumbs.db`, a user's own note, a crash-leftover `.tmp` — all of
/// them are correct to leave at the slot root, and aborting on them would strand a real migration
/// behind a file that does not matter.
///
/// The failure this guards against — a pool product we forgot to list — is therefore not
/// prevented by an abort either way: it would be *stranded* at the slot root, python would
/// rebuild that one product into the pool, and the cost is disk plus time, never data. The thing
/// that actually prevents it is `gate_pool_table.py`, which drives [`POOL_ENTRIES`] against the
/// python sources in both directions.
pub fn plan_slot(slot: &Path, family: &str) -> SlotPlan {
    let mut plan = SlotPlan::default();
    let Ok(rd) = std::fs::read_dir(slot) else {
        return plan;
    };
    for entry in rd.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        // dot entries are staging/tombstones and belong to whoever created them
        if name.starts_with('.') {
            continue;
        }
        if name == POOLS_DIR || name == SLOT_META {
            continue;
        }
        if is_pool_entry(family, &name) {
            plan.moving.push(name);
        } else {
            if !is_known_run_entry(family, &name) {
                plan.unknown.push(name.clone());
            }
            plan.staying.push(name);
        }
    }
    plan.moving.sort();
    plan.staying.sort();
    plan.unknown.sort();
    plan
}

/// Top-level slot entries this repo knowingly produces on the RUN side.
///
/// Only used to tell "we know this one stays" from "we have never seen this name", so the latter
/// can be logged once. Deliberately generous with prefixes: checkpoint and TensorBoard names
/// carry step numbers and host names.
fn is_known_run_entry(family: &str, name: &str) -> bool {
    const EXACT: &[&str] = &[
        "weights",
        "audition",
        "eval",
        "cluster",
        "diffusion",
        "diffusion.yaml",
        "lightning_logs",
        "filelists",
        "filelist.txt",
        "total_fea.npy",
        "config.json",
        "config.yaml",
        "run.json",
        "run_manifest.json",
        "stop.flag",
        "train.log",
        "best_state.json",
        "resume_state.json",
        "resume_best",
        "resume_latest",
    ];
    const PREFIX: &[&str] = &[
        "G_",
        "D_",
        "model_",
        "events.out.tfevents",
        "aug_gate_report",
    ];
    let _ = family;
    EXACT.contains(&name) || PREFIX.iter().any(|p| name.starts_with(p))
}

/// Fold one slot's pool products into `pools/<id>/`. Idempotent.
pub fn migrate_slot(slot: &Path, family: &str) -> Result<SlotOutcome> {
    if !slot.is_dir() {
        return Ok(SlotOutcome::AlreadyDone);
    }
    reconcile_staging(slot)?;
    if read_slot_meta(slot).is_some_and(|m| m.layout >= SLOT_LAYOUT) {
        return Ok(SlotOutcome::AlreadyDone);
    }
    let plan = plan_slot(slot, family);
    for u in &plan.unknown {
        // Loud, once per boot per entry: it is either harmless (a user file) or it is a pool
        // product this table forgot, and the second case must not be silent.
        tracing::warn!(
            "training slot {}: unrecognised entry {u:?} left at the slot root",
            slot.display()
        );
    }
    if plan.moving.is_empty() {
        // Never preprocessed (or already folded). Commit so the next boot does not re-scan.
        write_slot_meta(slot, &SlotMeta { layout: SLOT_LAYOUT, ..Default::default() })?;
        return Ok(SlotOutcome::Committed);
    }

    // The identity comes from the file python already wrote. Without it we cannot name the pool
    // after what it holds — but the products are real, so they still move, into a pool that will
    // simply never be MATCHED by a later run (it re-preprocesses instead of guessing). Its bytes
    // stay visible and reclaimable, which is the honest outcome.
    let fp_text = std::fs::read_to_string(slot.join(FINGERPRINT))
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let pool_id = if fp_text.is_empty() {
        "p000000000000".to_string()
    } else {
        pool_id_for(&fp_text)
    };

    let staging = slot.join(format!("{STAGING_PREFIX}{pool_id}"));
    std::fs::create_dir_all(&staging)
        .map_err(|e| UtaiError::Training(format!("SLOT_MIGRATE_FAILED: {e}")))?;

    // Move into a DOT-prefixed staging directory first: a half-filled `pools/<id>/` would be
    // readable as a pool by `list_pools`, and a run that matched it would train on half a cache.
    let step = || -> Result<()> {
        for name in &plan.moving {
            let from = slot.join(name);
            if !from.exists() {
                continue;
            }
            crate::util::rename_with_retry(&from, &staging.join(name), "SLOT_MIGRATE")
                .map_err(UtaiError::Training)?;
        }
        std::fs::create_dir_all(pools_root(slot))
            .map_err(|e| UtaiError::Training(format!("SLOT_MIGRATE_FAILED: {e}")))?;
        crate::util::rename_with_retry(&staging, &pools_root(slot).join(&pool_id), "SLOT_MIGRATE_COMMIT")
            .map_err(UtaiError::Training)?;
        Ok(())
    };
    if let Err(e) = step() {
        match roll_back(slot) {
            Ok(()) => {}
            Err(re) => tracing::error!("training slot {}: rollback also failed: {re}", slot.display()),
        }
        return Err(e);
    }
    write_slot_meta(slot, &SlotMeta { layout: SLOT_LAYOUT, ..Default::default() })?;
    Ok(SlotOutcome::Migrated(pool_id))
}

/// Fold every slot of every project into layout 2. Called at startup, immediately after
/// `tproject::migrate_legacy_layout` — the two are one migration seen from two levels.
///
/// ## Why here and not lazily on first use
///
/// python resolves its pool by identity and treats an unmigrated slot root as a legitimate pool
/// (`utai_train/pool.py`), so training is correct with or without this pass. What it buys is that
/// the DISK converges to one shape: the storage view can account for pools, the next batch's
/// per-run work has one layout to reason about, and a slot does not sit half in each world for as
/// long as nobody happens to train it.
///
/// Never fails the boot, per the same reasoning as the layout-1 migration: a slot that cannot be
/// migrated is logged and retried next launch, and a torn one rolls itself back first.
/// ⚠ Stands down entirely when another instance is alive — two processes racing these renames
/// would produce exactly the half-in-half-out tree the guards treat as corrupt.
pub fn migrate_all(data_dir: &Path) {
    let root = crate::training::tproject::training_root(data_dir);
    if !root.is_dir() {
        return;
    }
    if crate::crashlog::other_instance_alive() {
        tracing::warn!("training pool migration postponed: another live instance detected");
        return;
    }
    let Ok(rd) = std::fs::read_dir(&root) else { return };
    let (mut migrated, mut failed) = (0usize, 0usize);
    for entry in rd.flatten() {
        let proj = entry.path();
        // `project.json` is the authority for "this is a project" one level up, exactly as
        // `tproject` uses it; `.del_*` tombstones and `.migrating_*` markers are skipped by it.
        if !proj.join(crate::training::tproject::PROJECT_META).is_file() {
            continue;
        }
        for family in crate::training::tproject::FAMILIES {
            let slot = proj.join(family);
            if !slot.is_dir() {
                continue;
            }
            match migrate_slot(&slot, family) {
                Ok(SlotOutcome::Migrated(id)) => {
                    migrated += 1;
                    tracing::info!("pool layout: {} -> pools/{id}", slot.display());
                }
                Ok(_) => {}
                Err(e) => {
                    failed += 1;
                    tracing::error!("pool layout: {} could not be migrated: {e}", slot.display());
                }
            }
        }
    }
    if migrated > 0 || failed > 0 {
        tracing::info!("pool layout migration: {migrated} slot(s) folded, {failed} failed");
    }
}

/// Undo a torn migration: whatever is in staging goes back to the slot root.
///
/// Idempotent and mirror-image, exactly like `tproject::roll_back`. It is safe to run before the
/// commit check because staging only ever exists mid-migration: the commit renames it away.
pub fn reconcile_staging(slot: &Path) -> Result<()> {
    roll_back(slot)
}

fn roll_back(slot: &Path) -> Result<()> {
    let Ok(rd) = std::fs::read_dir(slot) else {
        return Ok(());
    };
    let staged: Vec<PathBuf> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_dir()
                && p.file_name()
                    .map(|n| n.to_string_lossy().starts_with(STAGING_PREFIX))
                    .unwrap_or(false)
        })
        .collect();
    for dir in staged {
        tracing::warn!("training slot {}: rolling back a torn pool migration", slot.display());
        let Ok(inner) = std::fs::read_dir(&dir) else { continue };
        for e in inner.flatten() {
            let name = e.file_name();
            let back = slot.join(&name);
            if back.exists() {
                // Something already re-created it at the root. Leave BOTH: deleting either one
                // could destroy hours of preprocessing, and a duplicated cache costs only disk.
                tracing::warn!(
                    "training slot {}: {:?} exists at the root already — leaving the staged copy in place",
                    slot.display(),
                    name
                );
                continue;
            }
            crate::util::rename_with_retry(&e.path(), &back, "SLOT_MIGRATE_UNDO")
                .map_err(UtaiError::Training)?;
        }
        // Only removes it when it came out empty — a leftover with content stays visible.
        let _ = std::fs::remove_dir(&dir);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_slot(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("utai_tpool_{}_{}", tag, uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn touch(p: &Path) {
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(p, b"x").unwrap();
    }

    /// A real RVC slot, in the shape the frozen pre-migration fixture actually has on disk
    /// (`TESTING/s120_f2_fixtures/pre_migration/111_efa35241/rvc`).
    fn legacy_rvc_slot(slot: &Path) {
        for d in ["0_gt_wavs", "1_16k_wavs", "2a_f0", "2b-f0nsf", "3_feature768", "mute", "weights", "audition"] {
            std::fs::create_dir_all(slot.join(d)).unwrap();
        }
        touch(&slot.join("0_gt_wavs").join("000_000.wav"));
        touch(&slot.join("mute").join("0_gt_wavs").join("mute48k.wav"));
        touch(&slot.join("weights").join("m_e14_s147.pth"));
        touch(&slot.join("audition").join("m_e14_s147").join("audition.wav"));
        for f in [
            "best_state.json",
            "config.json",
            "D_2333333.pth",
            "G_2333333.pth",
            "filelist.txt",
            "run.json",
            "run_manifest.json",
            "total_fea.npy",
            "train.log",
            "events.out.tfevents.1784135491.NucBox_k11.25768.0",
        ] {
            touch(&slot.join(f));
        }
        std::fs::write(slot.join(FINGERPRINT), b"abc123").unwrap();
    }

    #[test]
    fn pool_id_charset_can_never_collide_with_the_string_surgery() {
        // Three rules, each paid for elsewhere in this repo. A generated id that breaks any of
        // them fails SILENTLY, so they are asserted rather than documented.
        for probe in ["", "abc", "x|enc=vec768l12|loudnorm=1", "字|vocoder-v3"] {
            let id = pool_id_for(probe);
            assert_eq!(id.len(), 13, "id length must be stable: {id}");
            assert!(id.starts_with('p'), "must not start with a digit or a dot: {id}");
            assert!(
                id.chars().all(|c| c == 'p' || c.is_ascii_hexdigit()),
                "charset must stay [p0-9a-f]: {id}"
            );
            assert!(!id.contains("wav"), "would break rvc/extract_feature.py's replace: {id}");
            assert!(!id.contains("spec"), "would break rvc/extract_f0.py's skip: {id}");
            assert!(!id.contains('.'), "would break the .wav -> .spec.pt path surgery: {id}");
        }
        // distinct identities never propose the same directory
        assert_ne!(pool_id_for("a"), pool_id_for("b"));
        // and the same identity always does (this is what lets Rust and python agree)
        assert_eq!(pool_id_for("a|enc=x"), pool_id_for("a|enc=x"));
    }

    #[test]
    fn container_names_are_not_family_names() {
        // `tproject::has_family_slot` decides "old or new PROJECT layout" by looking for a
        // subdirectory named exactly after a family. A container that collided would make an
        // interrupted migration unrecoverable.
        for f in crate::training::tproject::FAMILIES {
            assert_ne!(POOLS_DIR, f);
            assert_ne!(SLOT_META, f);
        }
    }

    #[test]
    fn migrate_moves_exactly_the_pool_and_leaves_the_run_alone() {
        let slot = tmp_slot("rvc");
        legacy_rvc_slot(&slot);

        let plan = plan_slot(&slot, "rvc");
        assert_eq!(
            plan.moving,
            vec![
                "0_gt_wavs".to_string(),
                "1_16k_wavs".into(),
                "2a_f0".into(),
                "2b-f0nsf".into(),
                "3_feature768".into(),
                FINGERPRINT.into(),
                "mute".into(),
            ]
        );
        assert!(plan.unknown.is_empty(), "the real fixture must be fully classified: {:?}", plan.unknown);

        let out = migrate_slot(&slot, "rvc").unwrap();
        let pool_id = pool_id_for("abc123");
        assert_eq!(out, SlotOutcome::Migrated(pool_id.clone()));

        let pool = pools_root(&slot).join(&pool_id);
        assert!(pool.join("0_gt_wavs").join("000_000.wav").is_file());
        assert!(pool.join("mute").join("0_gt_wavs").join("mute48k.wav").is_file());
        assert!(pool.join(FINGERPRINT).is_file());

        // run products stay EXACTLY where they were — this is the whole claim of layout 2
        assert!(slot.join("G_2333333.pth").is_file());
        assert!(slot.join("weights").join("m_e14_s147.pth").is_file());
        assert!(slot.join("audition").join("m_e14_s147").join("audition.wav").is_file());
        assert!(slot.join("total_fea.npy").is_file());
        assert!(slot.join("filelist.txt").is_file());
        assert!(slot.join("run_manifest.json").is_file());
        assert!(!slot.join("0_gt_wavs").exists(), "the pool must not be left behind too");

        assert!(slot_has_pool(&slot));
        assert_eq!(sole_pool_fingerprint(&slot).as_deref(), Some("abc123"));

        // idempotent
        assert_eq!(migrate_slot(&slot, "rvc").unwrap(), SlotOutcome::AlreadyDone);
        assert!(pool.join("0_gt_wavs").join("000_000.wav").is_file());

        let _ = std::fs::remove_dir_all(slot);
    }

    #[test]
    fn a_slot_that_never_preprocessed_commits_without_moving_anything() {
        let slot = tmp_slot("empty");
        touch(&slot.join("run.json"));
        assert_eq!(migrate_slot(&slot, "sovits").unwrap(), SlotOutcome::Committed);
        assert!(read_slot_meta(&slot).unwrap().layout >= SLOT_LAYOUT);
        assert!(!slot_has_pool(&slot), "no fingerprint ⇒ no pool");
        assert!(slot.join("run.json").is_file());
        let _ = std::fs::remove_dir_all(slot);
    }

    /// An unknown entry must be LEFT ALONE, not moved and not fatal — and it must not stop the
    /// pool from migrating around it.
    #[test]
    fn unknown_entries_stay_put_and_do_not_block_the_migration() {
        let slot = tmp_slot("unknown");
        std::fs::create_dir_all(slot.join("dataset_44k")).unwrap();
        touch(&slot.join("dataset_44k").join("s").join("000.wav"));
        std::fs::write(slot.join(FINGERPRINT), b"fp|enc=vec768l12|loudnorm=0").unwrap();
        touch(&slot.join("Thumbs.db"));
        touch(&slot.join("my notes.txt"));

        let plan = plan_slot(&slot, "sovits");
        assert_eq!(plan.unknown, vec!["Thumbs.db".to_string(), "my notes.txt".into()]);

        assert!(matches!(migrate_slot(&slot, "sovits").unwrap(), SlotOutcome::Migrated(_)));
        assert!(slot.join("Thumbs.db").is_file(), "a stray file stays exactly where it was");
        assert!(slot.join("my notes.txt").is_file());
        assert!(slot_has_pool(&slot));
        let _ = std::fs::remove_dir_all(slot);
    }

    /// The sovits slot holds sovits AND sovits_diff products; `diffusion/` is a RUN product and
    /// must not follow the pool down.
    #[test]
    fn diffusion_progress_is_not_pool() {
        let slot = tmp_slot("diff");
        std::fs::create_dir_all(slot.join("dataset_44k")).unwrap();
        std::fs::create_dir_all(slot.join("diffusion").join("resume_best")).unwrap();
        touch(&slot.join("diffusion").join("model_5000.pt"));
        touch(&slot.join("diffusion").join("resume_best").join("model.pt"));
        touch(&slot.join("diffusion.yaml"));
        std::fs::write(slot.join(FINGERPRINT), b"z").unwrap();

        assert!(matches!(migrate_slot(&slot, "sovits").unwrap(), SlotOutcome::Migrated(_)));
        assert!(slot.join("diffusion").join("model_5000.pt").is_file());
        assert!(slot.join("diffusion").join("resume_best").join("model.pt").is_file());
        assert!(slot.join("diffusion.yaml").is_file());
        let _ = std::fs::remove_dir_all(slot);
    }

    /// A kill between "moved into staging" and "committed" must leave the slot exactly as it was.
    #[test]
    fn a_torn_migration_rolls_back_to_the_pre_migration_shape() {
        let slot = tmp_slot("torn");
        legacy_rvc_slot(&slot);
        let before = shape(&slot);

        // hand-build the torn state the migrator would leave behind
        let staging = slot.join(format!("{STAGING_PREFIX}{}", pool_id_for("abc123")));
        std::fs::create_dir_all(&staging).unwrap();
        for name in plan_slot(&slot, "rvc").moving {
            std::fs::rename(slot.join(&name), staging.join(&name)).unwrap();
        }
        assert!(!slot.join("0_gt_wavs").exists(), "the fixture really is torn");

        reconcile_staging(&slot).unwrap();
        assert_eq!(shape(&slot), before, "rollback must restore the exact pre-migration shape");
        assert!(!staging.exists());

        // …and the retry then migrates cleanly
        assert!(matches!(migrate_slot(&slot, "rvc").unwrap(), SlotOutcome::Migrated(_)));
        let _ = std::fs::remove_dir_all(slot);
    }

    /// (relative path, byte length) of every file under `dir`, sorted. Enough to catch a file that
    /// moved, vanished, or was truncated.
    fn shape(dir: &Path) -> Vec<(String, u64)> {
        fn walk(base: &Path, cur: &Path, out: &mut Vec<(String, u64)>) {
            let Ok(rd) = std::fs::read_dir(cur) else { return };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    walk(base, &p, out);
                } else if let Ok(md) = std::fs::metadata(&p) {
                    let rel = p.strip_prefix(base).unwrap_or(&p).to_string_lossy().replace('\\', "/");
                    out.push((rel, md.len()));
                }
            }
        }
        let mut v = Vec::new();
        walk(dir, dir, &mut v);
        v.sort();
        v
    }

    /// ⛔ The table must cover every family, or a whole chain silently keeps its pool at the slot
    /// root while python looks for it one level down.
    #[test]
    fn every_family_has_pool_entries_and_the_fingerprint() {
        for f in crate::training::tproject::FAMILIES {
            let names = pool_entries_for(f);
            assert!(names.contains(&FINGERPRINT), "{f} must carry the identity file");
            assert!(
                names.len() >= 2,
                "{f} has no preprocessing products in the table — that cannot be right"
            );
        }
        // `sovits_diff` is not a family: it shares the sovits slot, so it must resolve through it.
        assert_eq!(crate::training::backend_family("sovits_diff"), "sovits");
    }
}
