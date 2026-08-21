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
//! **In batch 1 the pool boundary was byte-for-byte the old cache-invalidation boundary.** Nothing
//! about WHICH artifacts share an identity changed; the only change was that a non-matching pool
//! became a SIBLING instead of a deletion. Keeping that statement true is what made that batch
//! reviewable, and it is why no identity formula was touched (see [`POOL_ENTRIES`]).
//!
//! §F2⒝ ④d is where the boundary itself moves: two knobs that decide what the products ARE (rvc's
//! sample rate, every chain's augmentation count) were never in it. Because the formula and the
//! text already on disk have to change together, the formula is VERSIONED and the version travels
//! to python in `run.json` — [`identity_version`], [`identity_suffix`], [`SLOT_LAYOUT_POOL_ID`].
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

/// Slot layout whose pools carry a **v2 identity text** ([`POOL_IDENTITY_VERSION`]).
///
/// ⛔ A THIRD constant, deliberately not a bump of either existing one — for the reason
/// [`trun::SLOT_LAYOUT_RUNS`](super::trun::SLOT_LAYOUT_RUNS) spells out: both migrations return
/// early on `layout >= <their own constant>` and otherwise compute a plan, so raising one of them
/// makes every already-folded slot compute an EMPTY plan, take the "nothing to move" branch and
/// stamp the new number — marked as migrated without a byte having moved. Each migration advances
/// the same file exactly one step, and the ordering between them is asserted in `trun`'s tests.
pub const SLOT_LAYOUT_POOL_ID: u32 = 4;

/// The pool-identity FORMULA version this build understands. Must equal
/// `utai_train.pool.POOL_IDENTITY_VERSION`.
pub const POOL_IDENTITY_VERSION: u32 = 2;

/// The `run.json` key that carries [`identity_version`] to python.
///
/// ⛔ A named constant rather than a literal in the `json!` block, because a rename on either
/// side is the one failure this whole mechanism cannot notice by itself: python's reader falls
/// back to 1 for an ABSENT key (an old `run.json` describes a v1 disk, so 1 is the truthful
/// answer there), which means a typo'd key does not error — it silently switches ④d off for
/// every slot, forever, with the old formula quietly computing against a re-stamped disk.
/// `tests/pool_identity_formula.rs` drives this constant into `utai_train/pool.py`'s reader.
pub const IDENTITY_VERSION_KEY: &str = "pool_identity_version";

/// The `dataset_44k` subdirectory a SINGLE-speaker sovits-family run slices into, from identity
/// v2 on. Must equal `utai_train.pool.SOLE_SPEAKER_DIR`, which states the three naming rules and
/// why a run's name had no business being a pool product's directory name.
pub const SOLE_SPEAKER_DIR: &str = "spk0";

/// WHICH identity formula this slot's pools are stamped with — the number handed to python in
/// `run.json`.
///
/// ★§F2⒝ ④d — this is the carrier that lets the formula and the disk change **at the same
/// instant** despite living in two languages. The new formula names a different directory, so a
/// python computing it against a disk still holding the old text finds no match and rebuilds
/// hours of preprocessing into a sibling pool; the reverse order costs the same. Tying the answer
/// to the slot's own layout marker means the answer flips exactly when the 3→4 migration commits
/// — i.e. only after every pool in this slot has been re-stamped.
///
/// ⛔ It is a pure function of the marker and nothing else. In particular it does NOT try to be
/// clever about a slot with no pools yet (a brand-new one, or one a full 重训 just erased): such a
/// slot is at layout 0 until the next boot folds it, so it is born with the v1 formula and gets
/// re-stamped like everything else. The alternative — "no pools ⇒ safe to use v2" — has to
/// exclude an UNMIGRATED slot whose pool sits at the slot root (`open_pool`'s legacy arm), and
/// getting that second condition wrong is a full re-preprocess for exactly the users whose data
/// root came back from a backup. One condition cannot be got wrong.
///
/// ⇒ A slot the migration skipped, failed on, or never reached (another instance was alive at
/// boot) answers 1 and keeps matching its own pools. Before ④d that refusal was invisible to
/// python, which would have gone on computing the new text against the old disk.
pub fn identity_version(slot: &Path) -> u32 {
    match read_slot_meta(slot) {
        Some(m) if m.layout >= SLOT_LAYOUT_POOL_ID => POOL_IDENTITY_VERSION,
        _ => 1,
    }
}

/// The trailing identity tokens, byte-for-byte `utai_train.pool.identity_suffix`.
///
/// ⛔ The two implementations produce ONE string or the user re-preprocesses. That is why both
/// append this suffix LAST (sovits_v2 has a conditional `|f0=` tail of its own, so a token folded
/// into the shared `extract_cache_fp_text` would sit before it there and after everything in the
/// other two sovits chains — one token set, two orders, and a migration that cannot reproduce
/// either without knowing which chain it is looking at).
///
/// `sample_rate_hz` is `Some` for **rvc only** — the one chain where the rate is a user choice.
/// It is Hz rather than the `"40k"` UI string because the migration's authority for an existing
/// pool is the header of a wav already in `0_gt_wavs`, and routing that through a display string
/// would be a second encoding of one fact.
pub fn identity_suffix(version: u32, aug_copies: u32, sample_rate_hz: Option<u32>) -> String {
    if version < 2 {
        return String::new();
    }
    let mut s = String::new();
    if let Some(hz) = sample_rate_hz {
        s.push_str(&format!("|sr={hz}"));
    }
    if aug_copies > 0 {
        s.push_str(&format!("|aug={aug_copies}"));
    }
    s
}

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
    /// Content of `dataset.fingerprint`, trimmed. Empty when the file is missing — such a pool is
    /// never MATCHED (it would be a guess), but it is still listed so its bytes stay visible and
    /// reclaimable.
    pub fp_text: String,
    /// ⛔★★S132 §F2⒝ — the file EXISTS but could not be read. Empty `fp_text` used to mean both
    /// this and「there is no identity here」, and the difference decides an irreversible action:
    /// `plan_slot_identity` skips an identity-less pool (correct — nothing matches it anyway) and
    /// then stamps the SLOT as identity-v2. Do that to a pool whose text is merely locked and the
    /// slot is told 「the disk is v2」 while that pool still holds v1 text — python then misses it
    /// by one byte, mints a sibling pool, and re-preprocesses for hours behind one `info` line.
    /// A slot at layout 4 never re-enters this chain, so it cannot heal.
    pub fp_unreadable: bool,
}

/// Every pool of one slot, sorted by id so listings never wobble.
///
/// ⛔ S132 — an unreadable `pools/` is an ERROR, not an empty slot: an empty list makes
/// `plan_slot_identity` return an empty plan, and an empty plan COMMITS layout 4 — stamping every
/// pool in the slot as v2 without having looked at a single one.
pub fn list_pools(slot: &Path) -> Result<Vec<PoolInfo>> {
    let mut out = Vec::new();
    let root = pools_root(slot);
    let rd = match std::fs::read_dir(&root) {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
        Err(e) => {
            return Err(UtaiError::Training(format!(
                "POOLS_DIR_UNREADABLE: {}: {e}",
                root.display()
            )))
        }
    };
    for entry in rd.flatten() {
        let id = entry.file_name().to_string_lossy().into_owned();
        // `.` entries are staging, never a pool.
        if id.starts_with('.') || !entry.path().is_dir() {
            continue;
        }
        let (fp_text, fp_unreadable) = match std::fs::read_to_string(entry.path().join(FINGERPRINT))
        {
            Ok(s) => (s.trim().to_string(), false),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => (String::new(), false),
            Err(_) => (String::new(), true),
        };
        out.push(PoolInfo { id, dir: entry.path(), fp_text, fp_unreadable });
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(out)
}

/// Has this slot ever been preprocessed?
///
/// Replaces the old `<slot>/dataset.fingerprint` existence test, which was the ONE artifact every
/// family wrote and therefore the single portable judge. It still is — it just lives inside the
/// pool now. The legacy arm is kept by the caller (`training::workspace_holds_work`) because a
/// pre-S76 directory that has not been through either migration still has it at the root.
pub fn slot_has_pool(slot: &Path) -> bool {
    // ⛔ S132 — an unreadable `pools/` answers YES. The consumer is `training::slot_holds_work`,
    // i.e. a refusal: 「no pool」 there is what lets an unconfirmed wipe through.
    // ⛔★S142 — ANY pool directory counts, **not** just one that still has a readable identity.
    // The filter used to be `!p.fp_text.is_empty()`, which asks 「is there a pool something could
    // MATCH」 — a different question from the one in this doc. A pool whose `dataset.fingerprint`
    // is missing still holds every slice and every derived feature (that is exactly why
    // `list_pools` keeps listing it, see `PoolInfo::fp_text`), and it is the case that costs the
    // MOST: nothing can match it, so the next run mints a sibling and re-preprocesses in full.
    // Both consumers want 「yes」 there — the wipe guard because those bytes are work, and the
    // params page because that cost is the one it exists to announce.
    match list_pools(slot) {
        Ok(pools) => !pools.is_empty(),
        Err(e) => {
            tracing::error!(
                "cannot list the pools of {} ({e}) — answering 「has a pool」 so the wipe-consent                  guard in front of hours of preprocessing stays closed",
                slot.display()
            );
            true
        }
    }
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
    // ⛔ S132 — 「could not list」 joins 「more than one」 in answering None (this function's whole
    // posture is「refuse to guess」), but it says so: silence here reads as「exactly zero pools」.
    let listed = match list_pools(slot) {
        Ok(l) => l,
        Err(e) => {
            tracing::warn!("cannot list the pools of {} ({e}) — refusing to name a sole pool", slot.display());
            return None;
        }
    };
    let pools: Vec<PoolInfo> = listed.into_iter().filter(|p| !p.fp_text.is_empty()).collect();
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
///
/// Shared with [`super::trun`], which advances the SAME file to layout 3. Deliberately one
/// writer: two atomic-write implementations for one commit point is how the two halves of a
/// staged migration start disagreeing about what "committed" means.
pub(crate) fn write_slot_meta(slot: &Path, meta: &SlotMeta) -> Result<()> {
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

// ─────────────────────────── migration (layout 3 → 4) ───────────────────────────
//
// ★§F2⒝ ④d — re-stamp every pool of a slot with the v2 identity text, and give a sole speaker's
// slice directory its constant name. THIS is what flips [`identity_version`] for the slot, so it
// is also the moment python starts computing the v2 formula: the two halves change together
// because the marker is written LAST and nothing else writes it.
//
// ## Why there is no staging directory here
//
// The other two migrations MOVE files, so they need a single rename to commit. This one edits a
// file's CONTENT and renames a directory INSIDE the pool — two actions no single rename can
// commit together. What makes that acceptable is the version gate rather than a protocol:
//
// * the marker is the commit point and it is written last, so any torn state still reads as
//   layout 3 ⇒ python still computes v1 ⇒ it still sees the world it knows;
// * every step is IDEMPOTENT (the new text is rebuilt from the stripped old one, the rename is
//   skipped when the target name is already there), so re-running converges;
// * this runs at boot, before a window exists — so "torn state, then the user trains" needs a
//   restart in between, and the restart re-runs this first.
//
// ⛔ The one hole that argument does NOT cover is a slot with SEVERAL pools where a later pool
// fails after an earlier one was already re-stamped: that state is persistent, the app keeps
// running, and python (still v1) would then miss the re-stamped pool and mint a sibling. So a
// failure mid-slot rolls back every step this slot already applied, mirror-image, exactly as the
// other two migrations do.

/// What re-stamping one slot did.
#[derive(Debug, PartialEq, Eq)]
pub enum IdentityOutcome {
    /// Already at layout 4, or the slot does not exist.
    AlreadyDone,
    /// Nothing needed changing (no pools, or every pool already carries the v2 text). Committed so
    /// the next boot does not look again.
    Committed,
    /// n pools re-stamped.
    Restamped(usize),
    /// The slot could not be decided, so NOTHING was changed and the marker was NOT advanced.
    ///
    /// ⚠ A real answer, not a failure: at layout 3 python is told identity v1, so every pool in
    /// this slot keeps matching and the user pays nothing. Looked at again on the next boot.
    /// ⛔ Refusing is always better than stamping a guessed value — a wrong `|sr=` is not a slow
    /// run, it is features computed at one sample rate silently reused at another.
    Refused(String),
}

/// One pool's work, computed before anything is touched.
#[derive(Debug, Clone)]
struct PoolStep {
    dir: PathBuf,
    old_fp: String,
    new_fp: String,
    /// The sole-speaker slice directory, `(from, to)`. `None` for every chain without one.
    rename: Option<(PathBuf, PathBuf)>,
}

/// The run-level facts every pool of one slot is stamped from.
struct SlotFacts {
    /// `run_manifest.json`'s `aug_copies`.
    ///
    /// ⛔ THE authority, and the reason is not that it is the most accurate record of what is on
    /// disk — it is that it is the value the NEXT request will send. `formForSlot` restores the
    /// params form from this same manifest (`WorkspaceInfo.aug_copies`), so a stamp that agrees
    /// with it is a stamp the next run will match. Counting `_aug<idx>` files instead would be a
    /// LOWER BOUND: `augment_slices` prunes before it generates, and the f0 gate deletes copies it
    /// rejects, so a pool built with 3 can hold a maximum index of 1.
    /// ⚠ It is also the EFFECTIVE value everywhere it matters: a non-diff start writes
    /// `req.aug_copies` here, and a diffusion start either inherits this very number or (diff-first)
    /// writes its own back into it.
    aug_copies: u32,
    /// `n_speakers` (absent = 1) — decides whether the slice directory is name-derived at all.
    n_speakers: usize,
}

/// Sample rate from a RIFF header.
///
/// Format-agnostic on purpose: rvc slices are IEEE-float wavs (`wavfile.write(..., float32)`), so
/// anything that assumed 16-bit PCM would read a garbage rate out of a valid file.
fn wav_sample_rate(path: &Path) -> Option<u32> {
    let b = std::fs::read(path).ok()?;
    if b.len() < 12 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return None;
    }
    let mut i = 12usize;
    while i + 8 <= b.len() {
        let size = u32::from_le_bytes([b[i + 4], b[i + 5], b[i + 6], b[i + 7]]) as usize;
        let body = i + 8;
        if &b[i..i + 4] == b"fmt " {
            if body + 8 > b.len() {
                return None;
            }
            return Some(u32::from_le_bytes([b[body + 4], b[body + 5], b[body + 6], b[body + 7]]));
        }
        i = body + size + (size & 1);
    }
    None
}

/// THE sample rate an rvc pool's products were built at, read off the pool itself.
/// `Ok(None)` = this pool holds no rvc slices at all.
///
/// ⛔ Deliberately NOT the run manifest. A wrong `|sr=` is a WRONG RESULT — the next run at that
/// rate would match this pool, re-slice, and reuse f0/features computed at the other rate (they
/// are cached by slice NAME and the names do not change) — so it has to be a positive fact about
/// these bytes. The manifest says what the last REQUEST asked for, and it is one record for a slot
/// that can hold several pools.
///
/// Two independent witnesses, and they must agree:
/// * the header of the gt slices (`<pool>/0_gt_wavs/*.wav`);
/// * the mute assets, whose FILENAME carries the rate (`<pool>/mute/0_gt_wavs/mute48k.wav`) and
///   which are copied skip-if-exists, so they accumulate one entry per rate the pool ever served.
///
/// Disagreement is what a pool merged from two data roots looks like (`pools/` merges file by
/// file and the pool id does not contain the rate), and there is no honest single answer for it.
fn rvc_pool_sample_rate(pool: &Path) -> std::result::Result<Option<u32>, String> {
    let mut seen: Option<u32> = None;
    let mut disagree: Vec<String> = Vec::new();
    let mut note = |what: String, hz: u32, seen: &mut Option<u32>| match *seen {
        Some(prev) if prev != hz => disagree.push(what),
        _ => *seen = Some(hz),
    };

    let gt = pool.join("0_gt_wavs");
    let mut wavs = 0usize;
    if let Ok(rd) = std::fs::read_dir(&gt) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            // ⚠ `.wav` only: a trained pool also holds `<stem>.spec.pt` in this very directory
            // (the loader writes it beside the wav), and it sorts BEFORE the wavs.
            if !name.ends_with(".wav") {
                continue;
            }
            wavs += 1;
            match wav_sample_rate(&e.path()) {
                Some(hz) => note(format!("{name}={hz}"), hz, &mut seen),
                None => continue,
            }
        }
    }
    if wavs > 0 && seen.is_none() {
        return Err(format!("{}: slices are present but none has a readable RIFF header", gt.display()));
    }
    if let Ok(rd) = std::fs::read_dir(pool.join("mute").join("0_gt_wavs")) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            let Some(k) = name
                .strip_prefix("mute")
                .and_then(|s| s.strip_suffix("k.wav"))
                .and_then(|s| s.parse::<u32>().ok())
            else {
                continue;
            };
            note(format!("mute/{name}"), k * 1000, &mut seen);
        }
    }
    if !disagree.is_empty() {
        return Err(format!(
            "{}: products of more than one sample rate in one pool ({}) — refusing to stamp a \
             single identity on it",
            pool.display(),
            disagree.join(", ")
        ));
    }
    Ok(seen)
}

/// The identity text with any `|sr=` / `|aug=` token removed — the v1 text a v2 one was built from.
///
/// This is what makes the whole migration idempotent (and therefore what makes a torn state
/// self-heal): the new text is always REBUILT from the stripped old one rather than appended to it,
/// so running twice cannot produce `…|aug=2|aug=2`.
///
/// ⚠ It strips those two keys ANYWHERE, not only at the end. That is safe because no chain emits
/// either key elsewhere — a fact `tests/pool_identity_formula.rs` pins per chain — and it is the
/// robust choice: a token in an unexpected position means the text was written by something we do
/// not understand, and dropping it puts us back on the one text we can rebuild.
/// ⚠ A multi-speaker rvc identity is a `|`-join of BARE hashes with no `=` at all; those have
/// neither prefix and are preserved.
fn strip_identity_suffix(fp: &str) -> String {
    let kept: Vec<&str> = fp
        .split('|')
        .enumerate()
        .filter(|(i, t)| *i == 0 || !(t.starts_with("sr=") || t.starts_with("aug=")))
        .map(|(_, t)| t)
        .collect();
    kept.join("|")
}

/// The run-level facts, or why this slot cannot be decided.
fn slot_facts(slot: &Path) -> std::result::Result<SlotFacts, String> {
    let mut found: Option<serde_json::Value> = None;
    // ⛔ S132 — the same distinction S131 笔 2 drew for the manifest FILE, one level up: an
    // unreadable `runs/` must not read as「this slot has no runs」, which here would silently
    // become「the slot root is the one run」and stamp every pool from a manifest that is not there.
    for run in super::trun::run_dirs(slot).map_err(|e| e.to_string())? {
        let p = run.path().join("run_manifest.json");
        // ⛔★§F2⒝ ④e 笔 2 — 「不在」与「读不动」必须分开,而它们此前是同一条 `continue`。
        //
        // 少一个 run 不是少一点信息,是**换一个答案**:这个函数的返回值会被打到这个槽的**每一个**
        // 池的身份串上。今天槽恒一个 run,所以「跳过」= `found` 仍是 None ⇒ 下面那条「没有 run
        // manifest」的拒绝,结果对但措辞错。④e 之后槽有 N 个 run,而那时同一条 `continue` 会让
        // 「两个 run 里有一个的 manifest 被杀软/网盘/ACL 占着」**静默降级成「只有一个 run」** ⇒
        // 拿另一个 run 的 `aug_copies` 给整槽打戳,而 E2 那条响亮拒绝**根本不会触发**。
        // ⇒ 这正是这道闸存在的理由反过来发生一遍,而且是**fail-open** 方向。
        let raw = match std::fs::read_to_string(&p) {
            Ok(raw) => raw,
            // 真的没有:一个 `try_start` 建了目录却没走到写 manifest 的 run(失败的启动留下的形状)。
            // 它确实什么都说不出来,跳过是对的。
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(format!("{}: unreadable run manifest: {e}", p.display())),
        };
        let v: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| format!("{}: unreadable run manifest: {e}", p.display()))?;
        if found.is_some() {
            // Layout 3 allows one run per slot; two manifests means ④e already happened and this
            // migration predates the rule for choosing between them.
            return Err(format!("{}: more than one run manifest", slot.display()));
        }
        found = Some(v);
    }
    let Some(m) = found else {
        return Err(format!("{}: no run manifest — nothing says what this pool was built with", slot.display()));
    };
    Ok(SlotFacts {
        aug_copies: m["aug_copies"].as_u64().unwrap_or(0) as u32,
        n_speakers: m["n_speakers"].as_u64().unwrap_or(1).max(1) as usize,
    })
}

/// Where this pool's sole-speaker slice directory has to move, if anywhere.
///
/// ⛔ `to.exists()` is not defensive. `std::fs::rename(dir, <an existing FILE>)` returns **Ok** on
/// Windows and destroys that file silently (measured), so a user note or a crash leftover named
/// `spk0` would disappear into a log line that says the migration succeeded.
fn plan_slice_rename(
    pool: &Path,
    n_speakers: usize,
) -> std::result::Result<Option<(PathBuf, PathBuf)>, String> {
    let root = pool.join("dataset_44k");
    // ⛔★★S132 — 「there is no slice tree」 and 「I could not look」 used to be the same `Ok(None)`.
    // `Ok(None)` means「no rename needed」, and at aug=0 the new fingerprint text equals the old
    // one, so the pool drops out of the plan entirely and the slot still gets stamped v2. python
    // then asks for `dataset_44k/spk0` (flist.py, identity ≥ 2), does not find the tree that is
    // still named `<model_slug>`, and slices a SECOND complete one — the exact shape ④d/S127 spent
    // a batch closing, resurrected from one transient read failure. And because the slot is now at
    // layout 4, the loud「two slice trees ⇒ Err」below can never fire for it again.
    let rd = match std::fs::read_dir(&root) {
        Ok(rd) => rd,
        // rvc / vocoder, or a sovits pool that never got as far as slicing — a positive fact.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("{}: slice tree unreadable: {e}", root.display())),
    };
    let mut subs: Vec<String> = rd
        .flatten()
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    subs.sort();
    if subs.is_empty() {
        return Ok(None);
    }
    if n_speakers > 1 {
        // Co-trained slugs are folded into the fingerprint itself, so renaming one would
        // re-identify the pool. They stay, and the count has to match or we do not understand
        // this pool.
        if subs.len() != n_speakers {
            return Err(format!(
                "{}: manifest says {n_speakers} speakers but {} slice tree(s) are present",
                root.display(),
                subs.len()
            ));
        }
        return Ok(None);
    }
    if subs.len() != 1 {
        // The shape S127 closed: a rename plus 「重训(仅扩散)」 used to grow a second complete
        // tree in one pool. Which one is live is not knowable from here, and picking wrong means
        // the next run slices a THIRD.
        return Err(format!(
            "{}: a sole-speaker pool with {} slice trees ({}) — cannot tell which one is live",
            root.display(),
            subs.len(),
            subs.join(", ")
        ));
    }
    if subs[0] == SOLE_SPEAKER_DIR {
        return Ok(None);
    }
    let to = root.join(SOLE_SPEAKER_DIR);
    if to.exists() {
        return Err(format!("{}: {} is already taken", root.display(), SOLE_SPEAKER_DIR));
    }
    Ok(Some((root.join(&subs[0]), to)))
}

/// Everything this slot needs, or why it cannot be decided. Pure — nothing is touched.
///
/// ⛔ The plan reads POOL CONTENTS, and that is the one thing it cannot inherit from the other two
/// migrations: their plans are computed from the slot's TOP-LEVEL listing, which for 3→4 is
/// identical before and after. A plan shaped like theirs would be EMPTY for every slot on earth,
/// take the "nothing to do" branch, stamp layout 4 — and hand python a v2 formula to run against a
/// v1 disk. That is not a rare edge; it is 100% of installations.
fn plan_slot_identity(slot: &Path, family: &str) -> std::result::Result<Vec<PoolStep>, String> {
    let pools = list_pools(slot).map_err(|e| e.to_string())?;
    if pools.is_empty() {
        return Ok(Vec::new());
    }
    let facts = slot_facts(slot)?;
    let mut out = Vec::new();
    for p in pools {
        // ⛔★★S132 — 「I could not READ this pool's identity」 must never take the same exit as
        // 「this pool HAS no identity」. Only the second one is safe to skip: nothing matches an
        // identity-less pool, so leaving it behind costs nothing. Skipping the first one and then
        // committing layout 4 tells python 「the disk is v2」 about a pool that still says v1.
        if p.fp_unreadable {
            return Err(format!("POOL_FINGERPRINT_UNREADABLE: {}", p.dir.display()));
        }
        if p.fp_text.is_empty() {
            // A pool with no identity is never MATCHED by anything (`open_pool` compares content),
            // so there is nothing to keep matching. Leave its bytes visible.
            continue;
        }
        let sr = if family == "rvc" {
            match rvc_pool_sample_rate(&p.dir)? {
                Some(hz) => Some(hz),
                None => {
                    // ⛔★★S132 — this used to `continue`, and the comment justified it with 「no
                    // slices ⇒ nothing worth protecting」. But we are PAST the `fp_text.is_empty()`
                    // arm, so this pool HAS an identity — i.e. it has been preprocessed, and
                    // `rvc/preprocess.py::_wipe_slice_dirs` empties `0_gt_wavs`/`1_16k_wavs` on
                    // every run while KEEPING the f0/feature trees. So the real shape here is
                    // 「hours of derived products, zero gt wavs」 (a run stopped mid-preprocess),
                    // and skipping it while stamping the slot v2 orphans exactly those products.
                    // Undecidable ⇒ refuse the SLOT, loudly, and look again next launch.
                    return Err(format!(
                        "POOL_SAMPLE_RATE_UNKNOWN: {} has an identity but no rvc slices to read \
                         the rate from",
                        p.dir.display()
                    ));
                }
            }
        } else {
            None
        };
        let new_fp = format!(
            "{}{}",
            strip_identity_suffix(&p.fp_text),
            identity_suffix(POOL_IDENTITY_VERSION, facts.aug_copies, sr)
        );
        let rename = plan_slice_rename(&p.dir, facts.n_speakers)?;
        if new_fp == p.fp_text && rename.is_none() {
            continue;
        }
        out.push(PoolStep { dir: p.dir, old_fp: p.fp_text, new_fp, rename });
    }
    Ok(out)
}

/// Atomic, because this file IS the pool's identity — the same protocol `utai_train/pool.py` uses
/// and for the same reason: a truncated identity matches nothing and rebuilds everything.
///
/// ⚠ Bare `rename`, not `rename_with_retry`: os error 5 (a READONLY target, which backup restores
/// really do produce) is in that helper's retry set, so it would burn ~12 s per pool before
/// failing anyway. Here failing immediately is the better answer — the caller rolls back.
fn write_fingerprint(pool: &Path, text: &str) -> Result<()> {
    let io_err =
        |e: std::io::Error| UtaiError::Training(format!("POOL_FINGERPRINT_WRITE_FAILED: {}: {e}", pool.display()));
    let final_path = pool.join(FINGERPRINT);
    let tmp = pool.join(format!("{FINGERPRINT}.tmp"));
    std::fs::write(&tmp, text.as_bytes()).map_err(io_err)?;
    std::fs::rename(&tmp, &final_path).map_err(io_err)?;
    Ok(())
}

fn apply_step(step: &PoolStep) -> Result<()> {
    if let Some((from, to)) = &step.rename {
        crate::util::rename_with_retry(from, to, "POOL_SLICE_RENAME").map_err(UtaiError::Training)?;
    }
    write_fingerprint(&step.dir, &step.new_fp)
}

/// Mirror image of [`apply_step`], best-effort: a rollback that itself fails is logged, never
/// hidden, and never turned into the caller's error (that would replace the real cause).
fn undo_step(step: &PoolStep) {
    if let Err(e) = write_fingerprint(&step.dir, &step.old_fp) {
        tracing::error!("pool identity: could not roll back {}: {e}", step.dir.display());
    }
    if let Some((from, to)) = &step.rename {
        if to.exists() && !from.exists() {
            if let Err(e) = crate::util::rename_with_retry(to, from, "POOL_SLICE_RENAME_UNDO") {
                tracing::error!("pool identity: could not roll back {}: {e}", to.display());
            }
        }
    }
}

/// The two sidecar file names that record a pool's identity text (`utai_train/resume_state.py`'s
/// `LATEST_NAME` and the `BEST_STATE` marker inside a snapshot directory).
const RESUME_SIDECARS: [&str; 2] = ["resume_state.json", "state.json"];

/// Re-point every resume sidecar that recorded one of these pools' OLD identity text.
///
/// Without this, the first 续训 of every existing run reports `TRAINING_RESUME_DATASET_CHANGED`
/// while the dataset has not moved a byte — and that CODE is the signal
/// `project_v2_resume_divergence_open` relies on to tell two different failures apart, so polluting
/// it has a real cost.
///
/// ⚠ Matched on the exact old TEXT, so it cannot touch anything that is not this pool's record;
/// and best-effort, because its failure mode is one spurious warning rather than a broken pool —
/// rolling a correct migration back over it would be the worse trade.
fn repoint_resume_sidecars(slot: &Path, steps: &[PoolStep]) -> usize {
    fn walk(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
        if depth == 0 {
            return;
        }
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            let name = e.file_name().to_string_lossy().into_owned();
            if p.is_dir() {
                walk(&p, depth - 1, out);
            } else if RESUME_SIDECARS.contains(&name.as_str()) {
                out.push(p);
            }
        }
    }
    let mut files = Vec::new();
    // ⛔ S132 — best-effort is this function's DOCUMENTED posture (its worst case is one spurious
    // warning), so an unreadable `runs/` still returns 0 rather than rolling a correct migration
    // back. What changes is that it stops being SILENT: 0 used to mean「没有 sidecar 要改」and
    // 「我根本没看成」at the same time, and only the second one needs a human.
    let runs = match super::trun::run_dirs(slot) {
        Ok(runs) => runs,
        Err(e) => {
            tracing::warn!(
                "cannot enumerate the runs of {} ({e}) — no resume sidecar was re-pointed, so the \
                 next resume of each run may report a spurious TRAINING_RESUME_DATASET_CHANGED",
                slot.display()
            );
            return 0;
        }
    };
    for run in runs {
        // `<run>/resume_state.json` · `<run>/resume_{best,latest}/state.json` ·
        // `<run>/diffusion/resume_{best,latest}/state.json` — three levels is all of them.
        walk(run.path(), 3, &mut files);
    }
    let mut changed = 0usize;
    for f in files {
        let Ok(raw) = std::fs::read_to_string(&f) else { continue };
        let Ok(mut v) = serde_json::from_str::<serde_json::Value>(&raw) else { continue };
        let Some(old) = v["dataset_fingerprint"].as_str().map(String::from) else { continue };
        let Some(step) = steps.iter().find(|s| s.old_fp == old) else { continue };
        v["dataset_fingerprint"] = serde_json::json!(step.new_fp);
        let Ok(body) = serde_json::to_vec_pretty(&v) else { continue };
        let tmp = f.with_extension("json.tmp");
        if std::fs::write(&tmp, body).is_ok() && std::fs::rename(&tmp, &f).is_ok() {
            changed += 1;
        } else {
            let _ = std::fs::remove_file(&tmp);
            tracing::warn!("pool identity: could not re-point {}", f.display());
        }
    }
    changed
}

/// Give every existing run its `pool.json`, while the answer is still cheap to know. Returns how
/// many were written.
///
/// ★§F2⒝ ④d — this migration is the LAST moment when the run↔pool edge is knowable from here: at
/// layout 3 a slot holds exactly one run, so a single pool leaves no ambiguity. ④e turns that into
/// several runs over several pools with nothing on disk to tell them apart, and re-deriving the
/// edge then means re-reading every imported audio file (the pool is named by its fingerprint).
///
/// ⛔ Written ONLY when the slot has exactly one pool. With two, this side genuinely does not know
/// — and a guessed edge is worse than an absent one, because ④e's reclamation would use it to
/// decide which bytes may go.
/// ⛔ Never overwrites: python records this from inside the run it belongs to and therefore always
/// knows better; a backfill that clobbered it would replace a fact with an inference.
fn backfill_pool_refs(slot: &Path) -> usize {
    // ⛔ S132 — best-effort like the sidecar re-pointer: an unreadable `pools/` writes no edge and
    // says so, rather than failing a migration whose real work is already done.
    let listed = match list_pools(slot) {
        Ok(l) => l,
        Err(e) => {
            tracing::warn!("cannot list the pools of {} ({e}) — no run↔pool edge was backfilled", slot.display());
            return 0;
        }
    };
    let named: Vec<PoolInfo> = listed.into_iter().filter(|p| !p.fp_text.is_empty()).collect();
    let [only] = &named[..] else { return 0 };
    let mut n = 0usize;
    // `list_runs`, not `run_dirs`: the latter answers「槽根就是那个 run」for a slot with no `runs/`
    // container, and writing this file at the SLOT root would put a run product one level up —
    // after the migration that would have moved it has already gone by.
    // ⛔ S132 — same posture as `repoint_resume_sidecars`: this edge is an ANNOTATION, never an
    // authority (`trun::pool_of_run`'s doc says a missing one must read as「不回收」), so a slot
    // whose `runs/` cannot be listed writes nothing and says so, rather than failing the migration.
    let listed = match super::trun::list_runs(slot) {
        Ok(l) => l,
        Err(e) => {
            tracing::warn!("cannot enumerate the runs of {} ({e}) — no run↔pool edge was backfilled", slot.display());
            return 0;
        }
    };
    let run_dirs: Vec<PathBuf> = listed.into_iter().map(|r| r.dir).collect();
    for dir in run_dirs {
        let final_path = dir.join(super::trun::POOL_REF);
        if final_path.exists() {
            continue;
        }
        let body = match serde_json::to_vec(&serde_json::json!({ "pool_id": only.id })) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let tmp = dir.join(format!("{}.tmp", super::trun::POOL_REF));
        if std::fs::write(&tmp, body).is_ok() && std::fs::rename(&tmp, &final_path).is_ok() {
            n += 1;
        } else {
            let _ = std::fs::remove_file(&tmp);
        }
    }
    n
}

/// Re-stamp one slot's pools with the v2 identity text. Idempotent.
pub fn migrate_slot_identity(slot: &Path, family: &str) -> Result<IdentityOutcome> {
    if !slot.is_dir() {
        return Ok(IdentityOutcome::AlreadyDone);
    }
    // Same rule as the other two: reconcile a torn PREDECESSOR before the early return, or a slot
    // that crashed mid-fold would be judged on a half-moved tree.
    reconcile_staging(slot)?;
    let layout = read_slot_meta(slot).map(|m| m.layout).unwrap_or(0);
    if layout >= SLOT_LAYOUT_POOL_ID {
        return Ok(IdentityOutcome::AlreadyDone);
    }
    // ⛔ …and NOT below layout 3 either. The chain (`training::migrate_layouts`) promotes every
    // healthy slot to 3 before this step runs, so a slot still under 3 means one of the earlier
    // folds FAILED for it — and such a slot still keeps its pool at the SLOT ROOT, where
    // `list_pools` cannot see it. Planning it would therefore produce an empty plan, stamp layout
    // 4, and tell python to compute the v2 formula against a root pool still holding v1 text: a
    // brand-new pool and hours of preprocessing. That is S123's "structurally empty plan" trap
    // arriving through a different door, and the admission range is the whole defence.
    if layout < super::trun::SLOT_LAYOUT_RUNS {
        return Ok(IdentityOutcome::Refused(format!(
            "layout {layout}: the earlier folds have not committed for this slot"
        )));
    }
    let commit = |slot: &Path| -> Result<()> {
        // The run↔pool edge, recorded while it is still knowable — see [`backfill_pool_refs`].
        // Inside `commit` so BOTH commit paths (work done / nothing to do) get it, and before the
        // marker for the same reason every non-idempotent side effect is: a kill in between leaves
        // the slot at layout 3 and the next boot does it again.
        let wrote = backfill_pool_refs(slot);
        if wrote > 0 {
            tracing::info!("pool identity: recorded the pool of {wrote} run(s) in {}", slot.display());
        }
        // Read-modify-write: `..Default::default()` would drop a newer build's unknown keys, which
        // is exactly what `SlotMeta::extra` exists to prevent.
        let mut meta = read_slot_meta(slot).unwrap_or_default();
        meta.layout = SLOT_LAYOUT_POOL_ID;
        write_slot_meta(slot, &meta)
    };
    let steps = match plan_slot_identity(slot, family) {
        Ok(s) => s,
        Err(why) => return Ok(IdentityOutcome::Refused(why)),
    };
    if steps.is_empty() {
        commit(slot)?;
        return Ok(IdentityOutcome::Committed);
    }
    let mut done: Vec<&PoolStep> = Vec::new();
    for s in &steps {
        if let Err(e) = apply_step(s) {
            // ⛔★ 失败的那一步**自己也是半应用的**,所以它必须第一个退回来。
            //
            // `apply_step` 先改名后写戳,于是「戳写失败」留下的是一个**改了名却还挂着旧戳**的池。
            // 只退 `done` 会把它留在盘上,而 layout 停在 3 ⇒ `identity_version` 答 1 ⇒ python 按
            // v1 取 slug(`flist.resolve_speakers` 的 `len(out)==1 && version>=2` 不成立)⇒ 它去
            // `dataset_44k/<model_slug>` 找不到东西,**重新切一棵树** ⇒ 这个池里从此有两棵,而
            // `plan_slice_rename` 见到两棵就再也决定不了哪棵是活的 ⇒ 这个槽**永久 Refused**。
            // ⇒ 一次写失败会把「一次全量重跑」升级成「再也迁不动」,而且不需要重启就能踩到。
            //
            // ⚠ `undo_step` 对「什么都没应用成」的步骤是安全的:写戳是把**旧串**写回去(盘上本来
            //    就是它),改名回滚有 `to.exists() && !from.exists()` 那道守卫挡着。
            undo_step(s);
            for d in done.iter().rev() {
                undo_step(d);
            }
            return Err(e);
        }
        done.push(s);
    }
    let repointed = repoint_resume_sidecars(slot, &steps);
    if repointed > 0 {
        tracing::info!(
            "pool identity: re-pointed {repointed} resume sidecar(s) of {}",
            slot.display()
        );
    }
    // ⛔★★§F2⒝ ④e 笔 2 —— 提交失败也必须退回去,而它此前**不在回滚里**。
    //
    // `commit` 的最后一步是 `write_slot_meta`,它会以 `SLOT_META_WRITE_FAILED` 失败(只读属性、
    // 网盘、杀软占住 `slot.json.tmp` —— 与上面那条 `apply_step` 会失败的理由一模一样)。
    // 那一刻盘上是:**每个池都已经是 v2 的串、单说话人切片树已经改名成 `spk0`、续训 sidecar 也已
    // 经跟上了新串**,而 marker 停在 3 ⇒ `identity_version` 答 1 ⇒ python 按 **v1** 拼串 ⇒ 和盘上
    // 的 v2 串**一个都对不上** ⇒ 当场铸一个兄弟池,把几小时预处理重跑一遍,唯一的痕迹是一行
    // `logger.info`。⇒ 正是 ④d「两半必须同一刻改变」那句话反过来发生一遍,只不过方向是**盘先走**。
    //
    // ⚠ sidecar 必须**先**退:它是按 `old_fp == 盘上的串` 匹配的,所以只要先把池的戳退回 v1,
    //    这份反向重写就再也匹配不上了。反向步骤只需要 old/new 对调 —— `repoint_resume_sidecars`
    //    不看 `rename`,所以这里的 `rename: None` 是如实陈述而不是偷懒。
    // ⚠ `backfill_pool_refs` 写的 `pool.json` **不需要退**:它记的是池的**目录名**,而这次迁移
    //    从头到尾没有重命名过任何池目录(身份是文件内容,不是目录名)。
    if let Err(e) = commit(slot) {
        let back: Vec<PoolStep> = done
            .iter()
            .map(|s| PoolStep {
                dir: s.dir.clone(),
                old_fp: s.new_fp.clone(),
                new_fp: s.old_fp.clone(),
                rename: None,
            })
            .collect();
        let un = repoint_resume_sidecars(slot, &back);
        for d in done.iter().rev() {
            undo_step(d);
        }
        tracing::error!(
            "pool identity: {} could not be committed ({e}) — rolled {} pool(s) and {un} resume \
             sidecar(s) back to v1 so python keeps matching what is actually on disk",
            slot.display(),
            done.len()
        );
        return Err(e);
    }
    Ok(IdentityOutcome::Restamped(steps.len()))
}

/// Re-stamp every slot of every project. Called from `training::migrate_layouts`, LAST.
///
/// Never fails the boot, for the same reason the other two do not: a slot that cannot be decided
/// keeps answering identity v1 and therefore keeps working, and it is looked at again next launch.
pub fn migrate_identity_all(data_dir: &Path) {
    let root = crate::training::tproject::training_root(data_dir);
    if !root.is_dir() {
        return;
    }
    if crate::crashlog::other_instance_alive() {
        tracing::warn!("pool identity migration postponed: another live instance detected");
        return;
    }
    let Ok(rd) = std::fs::read_dir(&root) else { return };
    let (mut done, mut refused, mut failed) = (0usize, 0usize, 0usize);
    for entry in rd.flatten() {
        let proj = entry.path();
        if !proj.join(crate::training::tproject::PROJECT_META).is_file() {
            continue;
        }
        for family in crate::training::tproject::FAMILIES {
            let slot = proj.join(family);
            if !slot.is_dir() {
                continue;
            }
            match migrate_slot_identity(&slot, family) {
                Ok(IdentityOutcome::Restamped(n)) => {
                    done += 1;
                    tracing::info!("pool identity: {} -> v2 ({n} pool(s))", slot.display());
                }
                Ok(IdentityOutcome::Refused(why)) => {
                    refused += 1;
                    // Loud, and once per boot: the slot keeps working on the v1 formula, but it
                    // will never advance until someone looks at this line.
                    tracing::warn!("pool identity: {} left at v1 — {why}", slot.display());
                }
                Ok(_) => {}
                Err(e) => {
                    failed += 1;
                    tracing::error!("pool identity: {} could not be re-stamped: {e}", slot.display());
                }
            }
        }
    }
    if done > 0 || refused > 0 || failed > 0 {
        tracing::info!(
            "pool identity migration: {done} slot(s) re-stamped, {refused} left at v1, {failed} failed"
        );
    }
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

    /// ★§F2⒝ ④d —— 池身份版本闸是这一批的**承重梁**:它是 python 唯一能看见的「这个槽迁没迁」。
    ///
    /// 判据必须驱动**真的 `slot.json`**,不能只比常量:一个把 `>=` 写成 `>` 或把常量记错的
    /// 实现,在「今天盘上还没有 layout 4」的世界里对任何比常量的断言都是绿的,而它的后果是
    /// ④d **永远不生效**(python 一直算旧公式)—— 一次静默的空转,没有任何报错。
    #[test]
    fn the_identity_version_flips_exactly_when_the_slot_marker_commits() {
        let slot = tmp_slot("identver");
        // 没有 slot.json = 从没迁过(全新槽 / 刚被整槽重训删掉的槽)
        assert_eq!(identity_version(&slot), 1, "没有 marker ⇒ 旧公式,它的池还是旧串");
        for layout in 0..SLOT_LAYOUT_POOL_ID {
            write_slot_meta(&slot, &SlotMeta { layout, ..Default::default() }).unwrap();
            assert_eq!(identity_version(&slot), 1, "layout {layout} 的槽还没被重新打戳过");
        }
        write_slot_meta(&slot, &SlotMeta { layout: SLOT_LAYOUT_POOL_ID, ..Default::default() })
            .unwrap();
        assert_eq!(identity_version(&slot), POOL_IDENTITY_VERSION, "3→4 提交那一刻才翻面");
        // 未来的 layout 仍然是 v2:版本闸问的是「有没有过 ④d 这一步」,不是「现在是第几层」。
        write_slot_meta(&slot, &SlotMeta { layout: SLOT_LAYOUT_POOL_ID + 7, ..Default::default() })
            .unwrap();
        assert_eq!(identity_version(&slot), POOL_IDENTITY_VERSION);
        // 一个读不动的 marker 必须答旧公式(fail-closed:猜错的代价是几小时重跑)
        std::fs::write(slot.join(SLOT_META), b"{not json").unwrap();
        assert_eq!(identity_version(&slot), 1, "读不动 marker ⇒ 不许猜新公式");
        let _ = std::fs::remove_dir_all(slot);
    }

    // ─────────────────── layout 3 → 4:重新打戳 ───────────────────

    /// ★§F2⒝ ④d(S130 · 待验清单 M14⑴)—— 身份文件是**先写 tmp 再 rename** 出来的。
    ///
    /// ## 为什么这条判据必须存在
    /// `write_fingerprint` 此前只有间接判据(「tmp 被占住就失败」)。而它写的是**命名了几小时
    /// 预处理产物的那个文件**:一个被截断的身份串与任何池都不匹配,于是全量重建到一个兄弟池里。
    /// 把它退回成一句 `fs::write(final)` 是一个**一个字节都不会报错**的改动。
    ///
    /// ## ⛔ 这条判据**不是**原子性判据
    /// 它证明的是【顺序】与【成功路径不留残渣】。真正的原子性要在 write 与 rename **之间**把进程
    /// 杀掉,`cargo test` 里做不到 —— 那是 M14⑵,归 §F7 第二遍(与 S122/S125 的撕裂态枚举同一族
    /// 工装)。别把这条读成「原子性已验」。
    ///
    /// ## 为什么是行为判据而不是源码断言
    /// 待验清单原本排的是「一条源码断言『先写 tmp 再 rename』(与 `write_slot_meta` 同形)」。
    /// 换成行为的理由有两条:⑴ 仓里的注释剥离器 `code_only` 是 `trun.rs` 测试模块的私有件,
    /// 照抄一份就是 drift;⑵ 行为判据**严格更强** —— 源码断言对「tmp 里写了错的内容再 rename」
    /// 完全瞎,而下面第二段连那个都能抓。
    #[test]
    fn the_pool_identity_lands_through_a_temp_file() {
        // ⑴ 成功路径:文件内容对,而且目录里**没有**任何残渣。
        let pool = tmp_slot("fp_write_ok");
        write_fingerprint(&pool, "abc123|sr=48000|aug=2").unwrap();
        assert_eq!(
            std::fs::read_to_string(pool.join(FINGERPRINT)).unwrap(),
            "abc123|sr=48000|aug=2"
        );
        let left: Vec<String> = std::fs::read_dir(&pool)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n != FINGERPRINT)
            .collect();
        assert!(left.is_empty(), "写完之后池里多了 {left:?}");

        // ⑵ ★ 顺序的**行为**判据:把最终路径变成一个**目录** ⇒ rename 必然失败。
        //    先 tmp 后 rename 的实现会在失败之前已经把内容落在 `<name>.tmp` 上;
        //    一个 `fs::write(final)` 的实现则连 tmp 都不会出现 —— 这就是两者分得开的地方。
        //    ⚠ 断言 rename **真的**失败,而不是假设:S129 实测过 `rename(目录, 已存在的文件)`
        //      在本机返回 Ok 并把那个文件无声销毁,所以这一族的方向性不许靠印象。
        let pool2 = tmp_slot("fp_write_torn");
        std::fs::create_dir_all(pool2.join(FINGERPRINT)).unwrap();
        let err = write_fingerprint(&pool2, "tmp-comes-first")
            .expect_err("最终路径是一个目录,rename 不该成功");
        assert!(
            format!("{err}").contains("POOL_FINGERPRINT_WRITE_FAILED"),
            "失败必须带得走归因:{err}"
        );
        let tmp = pool2.join(format!("{FINGERPRINT}.tmp"));
        assert!(tmp.is_file(), "没有 tmp ⇒ 这不是『先写 tmp 再 rename』");
        assert_eq!(
            std::fs::read_to_string(&tmp).unwrap(),
            "tmp-comes-first",
            "tmp 里必须已经是**要写的那个串**(源码断言看不见这一条)"
        );
        let _ = std::fs::remove_dir_all(pool);
        let _ = std::fs::remove_dir_all(pool2);
    }

    /// 最小 RIFF 头 + 一个静音样本。用真头是因为被测的正是「从头里读采样率」。
    fn wav_at(path: &Path, hz: u32) {
        let mut b = Vec::new();
        b.extend_from_slice(b"RIFF");
        b.extend_from_slice(&40u32.to_le_bytes());
        b.extend_from_slice(b"WAVEfmt ");
        b.extend_from_slice(&16u32.to_le_bytes());
        b.extend_from_slice(&1u16.to_le_bytes()); // PCM
        b.extend_from_slice(&1u16.to_le_bytes()); // mono
        b.extend_from_slice(&hz.to_le_bytes());
        b.extend_from_slice(&(hz * 2).to_le_bytes());
        b.extend_from_slice(&2u16.to_le_bytes());
        b.extend_from_slice(&16u16.to_le_bytes());
        b.extend_from_slice(b"data");
        b.extend_from_slice(&4u32.to_le_bytes());
        b.extend_from_slice(&[0u8; 4]);
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p).unwrap();
        }
        std::fs::write(path, &b).unwrap();
    }

    /// 一个 layout 3 的槽:`slot.json` + 一个 run(带 manifest)+ 若干池。
    fn layout3_slot(tag: &str, manifest: serde_json::Value) -> PathBuf {
        let slot = tmp_slot(tag);
        write_slot_meta(&slot, &SlotMeta { layout: 3, ..Default::default() }).unwrap();
        let run = slot.join("runs").join("r0123456789ab");
        std::fs::create_dir_all(&run).unwrap();
        std::fs::write(run.join("run_manifest.json"), serde_json::to_vec_pretty(&manifest).unwrap())
            .unwrap();
        slot
    }

    fn mk_pool(slot: &Path, id: &str, fp: &str) -> PathBuf {
        let d = pools_root(slot).join(id);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join(FINGERPRINT), fp).unwrap();
        d
    }

    fn fp_of(pool: &Path) -> String {
        std::fs::read_to_string(pool.join(FINGERPRINT)).unwrap()
    }

    /// `plan_slot_identity` 在看池**之前**先要 `slot_facts`(= 一份 run manifest),所以任何
    /// 「关于池的判据」的夹具都得先有它,否则红的是「没有 run manifest」而不是被测的那一条。
    fn mk_run_manifest(slot: &Path, aug: u32, n_speakers: usize) {
        let d = super::super::trun::runs_root(slot).join("r000000000000");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(
            d.join("run_manifest.json"),
            format!("{{\"aug_copies\":{aug},\"n_speakers\":{n_speakers}}}"),
        )
        .unwrap();
    }

    /// ★S141 §E2E-M5 —— `pools/` 里不是每一样东西都是池,而这条过滤此前没有任何判据。
    ///
    /// 它守的是屏幕上那句「预处理 N 份 · X」,而那正是用户用来决定**删什么**的读数。
    /// 数字虚高一份,他就会以为那里还压着一份预处理;数字里混进 `.staging` 的字节,
    /// 「删掉能省多少」就是错的。
    ///
    /// ⛔ 两条守卫写在**同一行**(`id.starts_with('.') || !entry.path().is_dir()`),所以
    /// 它们必须**分开断言**:合成一句 `assert_eq!(ids, [...])` 的话,删掉其中任一半红的都是
    /// 同一句,而「少了哪一半」看不出来 —— 一条红两种归因,正是 S129 铁律点名的形状。
    /// 兜底的全等断言排在最后(S108:具体的排前面,兜底的排最后)。
    #[test]
    fn the_pool_listing_counts_pools_and_not_whatever_else_sits_in_that_directory() {
        let slot = tmp_slot("pool_filter");
        // 故意乱序建,顺便钉住「按 id 排序,列表不会抖」这句 doc
        mk_pool(&slot, "pbbb", "fp-b");
        mk_pool(&slot, "paaa", "fp-a");
        // ⑴ staging:铸池铸到一半留下的形状
        std::fs::create_dir_all(pools_root(&slot).join(".staging_abcdef")).unwrap();
        // ⑵ 一个普通**文件**躺在 pools/ 里。今天没有任何代码会写出它 —— 而那恰恰是这半条
        //    守卫从来没被执行过的原因(Thumbs.db / desktop.ini / 用户手动放的东西都是这一形)
        std::fs::write(pools_root(&slot).join("stray.txt"), b"x").unwrap();

        let ids: Vec<String> = list_pools(&slot).unwrap().into_iter().map(|p| p.id).collect();

        assert!(
            !ids.iter().any(|i| i.starts_with('.')),
            "一个 `.staging` 目录被当成了池 —— 它是铸池的中间态,算进「预处理 N 份」\
             会让用户看见一份并不存在的预处理。实得:{ids:?}"
        );
        assert!(
            !ids.iter().any(|i| i == "stray.txt"),
            "`pools/` 里的一个普通文件被当成了池 —— 之后每一个把 `PoolInfo.dir` 当目录用的\
             读者都会在它身上得到「读不动」。实得:{ids:?}"
        );
        assert_eq!(ids, vec!["paaa".to_string(), "pbbb".to_string()], "兜底:恰好这两个,且有序");
    }

    /// ★★S132 §F2⒝ ④e —— 「我判不了这个池」**永远不许**和「这个池没什么可判的」走同一个出口。
    ///
    /// ## 为什么这条比它看起来重
    /// 3→4 的提交点是**整槽**的一个数字,而计划是**逐池**算的。任何一条让某个池悄悄退出计划的
    /// 路径,都会让这个槽被盖上「盘上是 v2」的章,而那个池还写着 v1 —— python 于是差一个字节
    /// 匹配不上,铸兄弟池、重跑几小时,痕迹只有一行 info。而 layout 4 的槽**再也不会回到这条链**,
    /// 所以它连自愈的机会都没有。⇒ 不可判定的池必须让**整槽** Refused(响亮、下次开机重来)。
    ///
    /// ⚠ 反方向同样要钉:一个**真的没有身份**的池(指纹文件不存在)跳过是对的 —— 它本来就
    /// 匹配不上任何东西。把它也变成 Refused 会让那种槽永远迁不动。所以每一格都有阴性对照。
    #[test]
    fn an_undecidable_pool_refuses_the_slot_instead_of_being_skipped() {
        // ⑴ 指纹文件**不存在** ⇒ 合法跳过,而且整槽照样能提交
        let slot = tmp_slot("skip_ok");
        std::fs::create_dir_all(pools_root(&slot).join("p_nameless")).unwrap();
        mk_run_manifest(&slot, 0, 1);
        let pools = list_pools(&slot).unwrap();
        assert_eq!(pools.len(), 1);
        assert!(!pools[0].fp_unreadable, "缺文件不是「读不动」");
        assert!(plan_slot_identity(&slot, "sovits").is_ok(), "没有身份的池跳过是对的");

        // ⑵ 指纹**读不动**(用目录冒充文件 —— 「存在但读不出来」唯一可移植的形状)
        let slot2 = tmp_slot("fp_unreadable");
        let p2 = pools_root(&slot2).join("p_locked");
        std::fs::create_dir_all(p2.join(FINGERPRINT)).unwrap();
        mk_run_manifest(&slot2, 0, 1);
        let pools2 = list_pools(&slot2).unwrap();
        assert!(pools2[0].fp_unreadable, "存在但读不出来必须与缺文件分开");
        let e = plan_slot_identity(&slot2, "sovits").unwrap_err();
        assert!(e.contains("POOL_FINGERPRINT_UNREADABLE"), "{e}");

        // ⑶ `pools/` 自己读不动 ⇒ 计划**不许**变成空的(空计划会直接提交 layout 4)
        let slot3 = tmp_slot("pools_unreadable");
        std::fs::create_dir_all(&slot3).unwrap();
        std::fs::write(pools_root(&slot3), b"not a directory").unwrap();
        assert!(list_pools(&slot3).unwrap_err().to_string().contains("POOLS_DIR_UNREADABLE"));
        let e3 = plan_slot_identity(&slot3, "sovits").unwrap_err();
        assert!(e3.contains("POOLS_DIR_UNREADABLE"), "{e3}");

        // ⑷ 切片树读不动 ⇒ 不许当成「还没切过片」
        let slot4 = tmp_slot("slices_unreadable");
        let p4 = mk_pool(&slot4, "p_sliced", "abc|enc=vec768l12|loudnorm=1");
        std::fs::write(p4.join("dataset_44k"), b"not a directory").unwrap();
        mk_run_manifest(&slot4, 0, 1);
        let e4 = plan_slot_identity(&slot4, "sovits").unwrap_err();
        assert!(e4.contains("slice tree unreadable"), "{e4}");
        // 阴性对照:同一个池**没有** dataset_44k ⇒ 正常(rvc/vocoder,或还没切片的 sovits)
        std::fs::remove_file(p4.join("dataset_44k")).unwrap();
        assert!(plan_slot_identity(&slot4, "sovits").is_ok(), "没有切片树是肯定事实,不是错误");

        for d in [slot, slot2, slot3, slot4] {
            let _ = std::fs::remove_dir_all(d);
        }
    }

    /// ⛔ **这条是这一批最重要的一条**:3→4 要干的活全在 `pools/<id>/` 里面,槽顶层前后**一模一样**。
    /// 照抄前两个迁移器的 plan(它们只 `read_dir(slot)` 顶层一层)会得到一个**结构上恒空**的
    /// 计划,于是每个槽都走「没什么可搬」那一支、盖上 layout 4 —— 而 layout 4 同时打开 python 的
    /// v2 公式。纸面迁移 + 真实全量重跑,命中 100% 的安装。
    ///
    /// 判据因此不是「plan 非空」,而是打戳之后盘上的那个串必须**恰好等于**python 会算出来的那个
    /// (用生产函数 `identity_suffix` 拼,不在测试里重打一遍)。
    #[test]
    fn the_identity_migration_reads_pool_contents_not_the_slot_top_level() {
        let slot = layout3_slot("id_contents", serde_json::json!({ "aug_copies": 2 }));
        let pool = mk_pool(&slot, "p000000000001", "d41d8c|enc=vec768l12|loudnorm=1");
        std::fs::create_dir_all(pool.join("dataset_44k").join("mymodel_deadbeef")).unwrap();
        let top = |p: &Path| -> Vec<String> {
            let mut v: Vec<String> = std::fs::read_dir(p)
                .unwrap()
                .flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect();
            v.sort();
            v
        };
        let before = top(&slot);

        assert_eq!(migrate_slot_identity(&slot, "sovits").unwrap(), IdentityOutcome::Restamped(1));
        assert_eq!(
            fp_of(&pool),
            format!(
                "d41d8c|enc=vec768l12|loudnorm=1{}",
                identity_suffix(POOL_IDENTITY_VERSION, 2, None)
            ),
            "盘上的串必须与 python 会算出来的逐字节相同"
        );
        assert!(pool.join("dataset_44k").join(SOLE_SPEAKER_DIR).is_dir(), "切片目录没改名");
        assert!(!pool.join("dataset_44k").join("mymodel_deadbeef").exists(), "旧名树还在 = 孤儿树");
        assert_eq!(read_slot_meta(&slot).unwrap().layout, SLOT_LAYOUT_POOL_ID);
        assert_eq!(identity_version(&slot), POOL_IDENTITY_VERSION, "打完戳才轮到 python 用 v2");

        // …而槽顶层在这次迁移前后**逐项一模一样**(`slot.json` 的内容变了,名字没变),
        // 这正是「照抄骨架的 plan 结构上恒空」的机械理由。
        assert_eq!(before, top(&slot), "槽顶层一项都没变 —— 只读顶层的 plan 会恒空");
        let _ = std::fs::remove_dir_all(slot);
    }

    /// 幂等,而且是靠**重建**而不是**追加**做到的 —— 这是撕裂态能自愈的全部理由。
    #[test]
    fn restamping_rebuilds_the_text_instead_of_appending_to_it() {
        assert_eq!(strip_identity_suffix("d41d8c|enc=x|loudnorm=1|aug=2"), "d41d8c|enc=x|loudnorm=1");
        assert_eq!(strip_identity_suffix("abc|sr=48000|aug=3"), "abc");
        assert_eq!(strip_identity_suffix("abc"), "abc", "v1 的串原样穿过");
        // 多说话人 rvc 的身份是一串**没有 `=`** 的裸 hash 用 `|` 拼 —— 一个都不许丢。
        assert_eq!(strip_identity_suffix("aaaa|bbbb|cccc|aug=1"), "aaaa|bbbb|cccc");

        let slot = layout3_slot("id_idem", serde_json::json!({ "aug_copies": 3 }));
        let pool = mk_pool(&slot, "p000000000001", "d41d8c|enc=vec768l12|loudnorm=0");
        assert_eq!(migrate_slot_identity(&slot, "sovits").unwrap(), IdentityOutcome::Restamped(1));
        let once = fp_of(&pool);
        // 把 marker 退回去(= 一次「盖章前被杀」的撕裂态),再跑一遍
        write_slot_meta(&slot, &SlotMeta { layout: 3, ..Default::default() }).unwrap();
        assert_eq!(migrate_slot_identity(&slot, "sovits").unwrap(), IdentityOutcome::Committed);
        assert_eq!(fp_of(&pool), once, "第二遍不许再追加一次 —— `|aug=3|aug=3` 就是这么来的");
        let _ = std::fs::remove_dir_all(slot);
    }

    /// rvc 的 `|sr=` 来自**这个池自己的盘**。判据是「同一个槽的两个池拿到两个不同的答案」——
    /// 那是 manifest(只有一份)结构上给不出的东西,也正是跨数据根逐文件合并会造出来的形状。
    ///
    /// ⚠ 那个 `.spec.pt` 是**健壮性**输入不是承重判据:训练跑过之后它就躺在 `0_gt_wavs` 里,而且
    /// 字典序排在 wav 之前 —— 一个「读第一个文件」的实现会把它当音频。这里的实现读全部并取第一个
    /// 解析得出的,所以它只证明「非音频邻居不会把答案带偏」。
    #[test]
    fn an_rvc_pool_is_stamped_with_the_rate_on_its_own_disk() {
        let slot = layout3_slot("id_sr", serde_json::json!({ "aug_copies": 0 }));
        let a = mk_pool(&slot, "p00000000000a", "aaaa");
        wav_at(&a.join("0_gt_wavs").join("0_0.wav"), 40_000);
        touch(&a.join("0_gt_wavs").join("0_0.spec.pt"));
        let b = mk_pool(&slot, "p00000000000b", "bbbb");
        wav_at(&b.join("0_gt_wavs").join("0_0.wav"), 48_000);

        assert_eq!(migrate_slot_identity(&slot, "rvc").unwrap(), IdentityOutcome::Restamped(2));
        assert_eq!(fp_of(&a), "aaaa|sr=40000");
        assert_eq!(fp_of(&b), "bbbb|sr=48000", "两个池两个答案 —— 一份 manifest 给不出这个");

        // mute 资产是第二个见证人(文件名带速率、skip-if-exists 所以每服务过一个速率就多一份)。
        // 它与切片的头不一致 = 这个池被两个数据根揉过 ⇒ 拒绝。
        let c = mk_pool(&slot, "p00000000000c", "cccc");
        wav_at(&c.join("0_gt_wavs").join("0_0.wav"), 40_000);
        touch(&c.join("mute").join("0_gt_wavs").join("mute48k.wav"));
        write_slot_meta(&slot, &SlotMeta { layout: 3, ..Default::default() }).unwrap();
        assert!(matches!(
            migrate_slot_identity(&slot, "rvc").unwrap(),
            IdentityOutcome::Refused(_)
        ));
        let _ = std::fs::remove_dir_all(slot);
    }

    /// 一个池里混着两种采样率 = 跨数据根合并出来的形状。没有诚实的单一答案 ⇒ **整槽拒绝**,
    /// 而拒绝之后这个槽仍然在 v1 上正常工作。
    #[test]
    fn a_pool_holding_two_sample_rates_is_refused_and_nothing_moves() {
        let slot = layout3_slot("id_mixed", serde_json::json!({ "aug_copies": 0 }));
        let a = mk_pool(&slot, "p00000000000a", "aaaa");
        wav_at(&a.join("0_gt_wavs").join("0_0.wav"), 40_000);
        wav_at(&a.join("0_gt_wavs").join("0_1.wav"), 48_000);

        let out = migrate_slot_identity(&slot, "rvc").unwrap();
        assert!(matches!(out, IdentityOutcome::Refused(_)), "混装池必须拒绝,不许挑一个");
        assert_eq!(fp_of(&a), "aaaa", "拒绝 = 一个字节都没动");
        assert_eq!(read_slot_meta(&slot).unwrap().layout, 3, "拒绝 = 不盖章");
        assert_eq!(identity_version(&slot), 1, "⇒ python 继续算 v1,这个槽照常工作");
        let _ = std::fs::remove_dir_all(slot);
    }

    /// 多说话人的 slug **折进了指纹**,改名 = 每个多说话人池当场失配。所以只有独唱才换名。
    #[test]
    fn only_a_sole_speaker_slice_tree_gets_the_constant_name() {
        let slot =
            layout3_slot("id_multi", serde_json::json!({ "aug_copies": 0, "n_speakers": 2 }));
        let pool = mk_pool(&slot, "p000000000001", "d41d8c|enc=vec768l12|loudnorm=0");
        for s in ["alice_11111111", "bob_22222222"] {
            std::fs::create_dir_all(pool.join("dataset_44k").join(s)).unwrap();
        }
        assert_eq!(migrate_slot_identity(&slot, "sovits").unwrap(), IdentityOutcome::Committed);
        for s in ["alice_11111111", "bob_22222222"] {
            assert!(pool.join("dataset_44k").join(s).is_dir(), "{s} 被改名了 —— 那个池当场失配");
        }
        assert!(!pool.join("dataset_44k").join(SOLE_SPEAKER_DIR).exists());
        assert_eq!(fp_of(&pool), "d41d8c|enc=vec768l12|loudnorm=0", "aug=0 ⇒ 串逐字节不变");
        let _ = std::fs::remove_dir_all(slot);
    }

    /// 独唱池里有两棵切片树 = S127 关掉的那个形状的存量。哪一棵是活的从这里判不出来,而挑错
    /// 一棵的后果是下一次运行切出**第三棵** ⇒ 拒绝。
    #[test]
    fn a_sole_speaker_pool_with_two_slice_trees_is_refused() {
        let slot = layout3_slot("id_twotrees", serde_json::json!({ "aug_copies": 0 }));
        let pool = mk_pool(&slot, "p000000000001", "d41d8c|enc=x|loudnorm=0");
        for s in ["old_11111111", "new_22222222"] {
            std::fs::create_dir_all(pool.join("dataset_44k").join(s)).unwrap();
        }
        assert!(matches!(
            migrate_slot_identity(&slot, "sovits").unwrap(),
            IdentityOutcome::Refused(_)
        ));
        assert_eq!(read_slot_meta(&slot).unwrap().layout, 3);
        let _ = std::fs::remove_dir_all(slot);
    }

    /// ⛔ `std::fs::rename(目录, 一个**已存在的文件**)` 在这台机器上返回 **Ok**,并把那个文件
    /// **无声销毁**(实测)。所以改名前的 `exists()` 不是防御性检查:少了它,一个恰好叫 `spk0`
    /// 的用户笔记 / 崩溃残留会消失在一行「迁移成功」的日志里。
    #[test]
    fn a_slice_rename_never_overwrites_something_that_is_already_there() {
        let slot = layout3_slot("id_occupied", serde_json::json!({ "aug_copies": 0 }));
        let pool = mk_pool(&slot, "p000000000001", "d41d8c|enc=x|loudnorm=0");
        std::fs::create_dir_all(pool.join("dataset_44k").join("mymodel_deadbeef")).unwrap();
        let squatter = pool.join("dataset_44k").join(SOLE_SPEAKER_DIR);
        std::fs::write(&squatter, b"a user's note").unwrap();

        assert!(matches!(
            migrate_slot_identity(&slot, "sovits").unwrap(),
            IdentityOutcome::Refused(_)
        ));
        assert_eq!(std::fs::read(&squatter).unwrap(), b"a user's note", "那个文件被顶掉了");
        assert!(pool.join("dataset_44k").join("mymodel_deadbeef").is_dir());
        assert_eq!(read_slot_meta(&slot).unwrap().layout, 3);
        let _ = std::fs::remove_dir_all(slot);
    }

    /// 存量 run 的续训 sidecar 记着**旧**的身份串;不一起改写,每个存量 run 的第一次续训都会报
    /// 一次假的 `TRAINING_RESUME_DATASET_CHANGED` —— 而那条 CODE 正是那卷未结案的续训崩溃赖以
    /// 区分问题的信号。四个落点(含扩散链自己那两份)都要跟上。
    #[test]
    fn the_resume_sidecars_follow_the_pool_they_describe() {
        let slot = layout3_slot("id_sidecar", serde_json::json!({ "aug_copies": 1 }));
        let old = "d41d8c|enc=vec768l12|loudnorm=1";
        let pool = mk_pool(&slot, "p000000000001", old);
        let run = slot.join("runs").join("r0123456789ab");
        let blob = serde_json::json!({ "schema": 1, "epoch": 7, "dataset_fingerprint": old });
        let spots = [
            run.join("resume_state.json"),
            run.join("resume_best").join("state.json"),
            run.join("resume_latest").join("state.json"),
            run.join("diffusion").join("resume_best").join("state.json"),
        ];
        for p in &spots {
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, serde_json::to_vec_pretty(&blob).unwrap()).unwrap();
        }
        // 阴性对照:一份记着**别的**串的 sidecar 不许被碰。
        let other = run.join("diffusion").join("resume_latest").join("state.json");
        std::fs::create_dir_all(other.parent().unwrap()).unwrap();
        let foreign = serde_json::json!({ "dataset_fingerprint": "somebody-elses-pool" });
        std::fs::write(&other, serde_json::to_vec_pretty(&foreign).unwrap()).unwrap();

        assert_eq!(migrate_slot_identity(&slot, "sovits").unwrap(), IdentityOutcome::Restamped(1));
        let now = fp_of(&pool);
        assert_ne!(now, old);
        for p in &spots {
            let v: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
            assert_eq!(v["dataset_fingerprint"], serde_json::json!(now), "{} 没跟上", p.display());
            assert_eq!(v["epoch"], serde_json::json!(7), "{} 的其余内容被动过", p.display());
        }
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&other).unwrap()).unwrap();
        assert_eq!(v["dataset_fingerprint"], serde_json::json!("somebody-elses-pool"));
        let _ = std::fs::remove_dir_all(slot);
    }

    /// ⛔ 这是「不做 staging」那个决定唯一没被版本闸兜住的洞:一个槽有几个池,第 2 个失败时第 1 个
    /// 已经带着新戳而 layout 仍是 3 ⇒ python 算 v1、池里是新串 ⇒ 铸兄弟池全量重跑,而且**不需要
    /// 重启就能踩到**(失败是持久的,app 照常在跑)。⇒ 中途失败必须把这一槽做过的每一步按镜像退回。
    #[test]
    fn a_failure_part_way_through_rolls_the_whole_slot_back() {
        let slot = layout3_slot("id_rollback", serde_json::json!({ "aug_copies": 2 }));
        let a = mk_pool(&slot, "p00000000000a", "aaaa|enc=x|loudnorm=0");
        let b = mk_pool(&slot, "p00000000000b", "bbbb|enc=x|loudnorm=0");
        std::fs::create_dir_all(a.join("dataset_44k").join("mymodel_deadbeef")).unwrap();
        // ★S130 —— B **也**要有一棵待改名的切片树。
        //
        // ⛔ 这一行修的是这条测试自己的一个洞:B 原本没有 `dataset_44k`,于是 `plan_slice_rename`
        //    对它早返回 None,B 的那一步**没有改名动作** ⇒ 「改了名却没打上戳」这个半应用状态
        //    在全仓一次都没有被执行过,而它恰恰是失败那一步唯一会留下的形状。
        //    (S129 的镜像回滚只退 `done`,而失败的那一步不在里面 —— 见 `migrate_slot_identity`。)
        std::fs::create_dir_all(b.join("dataset_44k").join("mymodel_deadbeef")).unwrap();
        // 让 B 的原子写**必然**失败:tmp 路径被一个目录占住 ⇒ `std::fs::write` 写不进去。
        std::fs::create_dir_all(b.join(format!("{FINGERPRINT}.tmp"))).unwrap();

        assert!(migrate_slot_identity(&slot, "sovits").is_err(), "B 写不下去必须是错误");
        assert_eq!(fp_of(&a), "aaaa|enc=x|loudnorm=0", "A 的戳必须退回去");
        assert!(
            a.join("dataset_44k").join("mymodel_deadbeef").is_dir(),
            "A 的切片目录名必须退回去"
        );
        assert!(!a.join("dataset_44k").join(SOLE_SPEAKER_DIR).exists());
        // ★ 失败的那一步自己:戳没写成(本来就该是旧串),但**改名已经发生了** ——
        //   它必须也被退回去,否则 python 按 v1 找不到 `<model_slug>` 那棵树,会重新切一棵,
        //   而一个池里两棵树会让这个槽从此**永久 Refused**。
        assert_eq!(fp_of(&b), "bbbb|enc=x|loudnorm=0", "B 的戳本来就没写成");
        assert!(
            b.join("dataset_44k").join("mymodel_deadbeef").is_dir(),
            "⛔ 失败那一步的切片改名也必须退回去 —— 留着它 = 下一次 python 重切一棵树,\
             而两棵树会让这个槽再也迁不动"
        );
        assert!(!b.join("dataset_44k").join(SOLE_SPEAKER_DIR).exists());
        assert_eq!(read_slot_meta(&slot).unwrap().layout, 3, "没盖章");
        assert_eq!(identity_version(&slot), 1, "⇒ python 仍然算 v1,而盘上确实还是 v1 的串");
        let _ = std::fs::remove_dir_all(slot);
    }

    /// ★★§F2⒝ ④e 笔 2 —— **提交那一步失败,盘也必须退回去**。
    ///
    /// ## 为什么这条判据必须存在
    ///
    /// 回滚此前只包住 `apply_step` 的循环,而 `commit` 排在循环**之后**。`commit` 的最后一步
    /// (`write_slot_meta`)会以 `SLOT_META_WRITE_FAILED` 失败 —— 理由和 `apply_step` 会失败的
    /// 一模一样(只读属性、网盘、杀软占住 tmp)。它失败的那一刻,盘上**每个池都已经是 v2 的串**、
    /// 切片树已经改名、sidecar 也已经跟上,而 marker 停在 3 ⇒ `identity_version` 答 1 ⇒
    /// python 按 v1 拼串、和盘上一个都对不上 ⇒ **铸兄弟池 + 几小时全量重跑**,唯一痕迹一行 info。
    ///
    /// ⇒ 这是 ④d 那句「两半必须同一刻改变」**反过来**发生一遍(盘先走了一步),
    /// 而 S130 修的是它的镜像(marker 先走)。两个方向都要有人守着。
    ///
    /// ⛔ 这条分支此前**一次都没有被执行过** —— 一条从没被执行过的错误分支就是一条空判据。
    #[test]
    fn a_failed_commit_rolls_the_disk_back_to_v1_too() {
        let slot = layout3_slot("id_commitfail", serde_json::json!({ "aug_copies": 2 }));
        let a = mk_pool(&slot, "p00000000000a", "aaaa|enc=x|loudnorm=0");
        std::fs::create_dir_all(a.join("dataset_44k").join("mymodel_deadbeef")).unwrap();
        // 一份续训 sidecar,记着**旧**串 —— 它是这条回滚里最容易被忘掉的一半。
        let run = super::super::trun::runs_root(&slot).join("r0123456789ab");
        std::fs::create_dir_all(&run).unwrap();
        let sidecar = run.join("resume_state.json");
        std::fs::write(
            &sidecar,
            serde_json::to_vec_pretty(&serde_json::json!({
                "schema": 1, "dataset_fingerprint": "aaaa|enc=x|loudnorm=0"
            }))
            .unwrap(),
        )
        .unwrap();
        std::fs::write(run.join("run_manifest.json"), br#"{"aug_copies":2}"#).unwrap();
        // ⇒ 让 `write_slot_meta` 的原子写**必然**失败:tmp 路径被一个目录占住。
        //   ⚠ 不能改 `slot.json` 本身 —— 那会让上面 `read_slot_meta` 判成 layout 0,
        //     于是在准入那一关就被挡掉,这条分支根本走不到。
        std::fs::create_dir_all(slot.join(format!("{SLOT_META}.tmp"))).unwrap();

        // 前置:这个池**确实**有活要干,否则下面全是对空气断言。
        let planned = plan_slot_identity(&slot, "sovits").unwrap();
        assert_eq!(planned.len(), 1, "前置:这一步本来就该改点什么");

        assert!(migrate_slot_identity(&slot, "sovits").is_err(), "盖章失败必须是错误");
        assert_eq!(fp_of(&a), "aaaa|enc=x|loudnorm=0", "池的戳必须退回 v1");
        assert!(
            a.join("dataset_44k").join("mymodel_deadbeef").is_dir(),
            "切片目录名必须退回去 —— 留着 `spk0` 而 marker 是 3,python 按 v1 会重切一棵树"
        );
        assert!(!a.join("dataset_44k").join(SOLE_SPEAKER_DIR).exists());
        let v: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&sidecar).unwrap()).unwrap();
        assert_eq!(
            v["dataset_fingerprint"].as_str().unwrap(),
            "aaaa|enc=x|loudnorm=0",
            "⛔ sidecar 也必须退 —— 否则每个存量 run 第一次续训报一次**假的** DATASET_CHANGED,\
             而那条信号正是「续训崩溃」那份未结案卷宗赖以区分问题的东西"
        );
        assert_eq!(read_slot_meta(&slot).unwrap().layout, 3, "没盖章");
        assert_eq!(identity_version(&slot), 1, "⇒ python 算 v1,而盘上确实还是 v1 —— 两边同步");
        let _ = std::fs::remove_dir_all(slot);
    }

    /// ★★§F2⒝ ④e 笔 2 —— 「读不动」不许被当成「不在」。
    ///
    /// 这个函数的返回值会被打到这个槽**每一个**池的身份串上,所以少看见一个 run 不是少一点信息,
    /// 是**换一个答案**。今天槽恒一个 run ⇒ 结果不变(照样 Refused),但措辞从「没有 run manifest」
    /// 变成「读不动」;④e 之后槽有 N 个 run,而那时旧写法会把「两个 run 里有一个的 manifest 被占着」
    /// **静默降级成「只有一个 run」** ⇒ 拿另一个 run 的份数给整槽打戳,而那条响亮的
    /// 「more than one run manifest」**根本不会触发**。⇒ fail-open,方向正好反了。
    #[test]
    fn a_run_manifest_that_cannot_be_read_is_refused_not_skipped() {
        let slot = tmp_slot("id_unreadable");
        write_slot_meta(&slot, &SlotMeta { layout: 3, ..Default::default() }).unwrap();
        mk_pool(&slot, "p000000000001", "aaaa|enc=x|loudnorm=0");
        let run = super::super::trun::runs_root(&slot).join("r0123456789ab");
        std::fs::create_dir_all(&run).unwrap();
        // 「在,但读不动」:用一个**目录**顶替那个文件 —— 跨平台都不是 NotFound。
        std::fs::create_dir_all(run.join("run_manifest.json")).unwrap();

        match migrate_slot_identity(&slot, "sovits") {
            Ok(IdentityOutcome::Refused(why)) => assert!(
                why.contains("unreadable run manifest"),
                "拒绝的理由必须说清是【读不动】,不是【不在】:{why}"
            ),
            other => panic!("读不动的 manifest 必须是响亮拒绝,得到 {other:?}"),
        }
        assert_eq!(read_slot_meta(&slot).unwrap().layout, 3, "拒绝不许盖章");
        let _ = std::fs::remove_dir_all(slot);
    }

    /// 取不到权威就不许打戳 —— 打错 `|aug=` 的代价是一次全量重跑,打错 `|sr=` 是错结果。
    #[test]
    fn a_slot_whose_facts_cannot_be_read_is_refused_rather_than_guessed() {
        let slot = tmp_slot("id_nofacts");
        write_slot_meta(&slot, &SlotMeta { layout: 3, ..Default::default() }).unwrap();
        let pool = mk_pool(&slot, "p000000000001", "aaaa|enc=x|loudnorm=0");
        // runs/ 里没有任何 run_manifest.json ⇒ 没有东西说得出这些切片是按几份增强建的。
        std::fs::create_dir_all(slot.join("runs").join("r0123456789ab")).unwrap();
        assert!(matches!(
            migrate_slot_identity(&slot, "sovits").unwrap(),
            IdentityOutcome::Refused(_)
        ));
        assert_eq!(fp_of(&pool), "aaaa|enc=x|loudnorm=0");
        assert_eq!(read_slot_meta(&slot).unwrap().layout, 3);

        // …而一个**从来没有池**的槽照常盖章:它没有任何要保住的东西。
        let empty = layout3_slot("id_nopools", serde_json::json!({}));
        assert_eq!(migrate_slot_identity(&empty, "sovits").unwrap(), IdentityOutcome::Committed);
        assert_eq!(read_slot_meta(&empty).unwrap().layout, SLOT_LAYOUT_POOL_ID);
        let _ = std::fs::remove_dir_all(slot);
        let _ = std::fs::remove_dir_all(empty);
    }

    /// ★§F2⒝ ④d —— run↔pool 这条边:盘上从来没有任何东西记过它,而 ④e 的「旧 run 可管理/删除」
    /// 离了它就回收不了池。迁移是**最后一次便宜地知道**它的时刻(layout 3 的槽恰好一个 run)。
    ///
    /// ⛔ 只在**恰好一个池**时写:两个池时这一侧是真的不知道,而一条猜出来的边比没有更糟 ——
    /// ④e 会拿它去决定哪些字节可以删。
    #[test]
    fn the_run_to_pool_edge_is_recorded_while_it_is_still_knowable() {
        let slot = layout3_slot("id_poolref", serde_json::json!({ "aug_copies": 0 }));
        let pool = mk_pool(&slot, "p00000000000a", "aaaa|enc=x|loudnorm=0");
        let run = slot.join("runs").join("r0123456789ab");
        assert_eq!(migrate_slot_identity(&slot, "sovits").unwrap(), IdentityOutcome::Committed);
        assert_eq!(
            super::super::trun::pool_of_run(&super::super::trun::run_dirs(&slot).unwrap()[0]),
            Some("p00000000000a".to_string())
        );
        assert!(!slot.join(super::super::trun::POOL_REF).exists(), "写到槽根去了");
        let _ = pool;

        // ⛔ 一个**没有 `runs/` 容器**的 layout-3 槽(迁移时无产物可搬)——`run_dirs` 对它答
        //    「槽根就是那个 run」,照它写就把一个 run 产物落在**槽根**,而会搬走它的那次迁移
        //    已经过去了。
        let noruns = tmp_slot("id_poolref_noruns");
        write_slot_meta(&noruns, &SlotMeta { layout: 3, ..Default::default() }).unwrap();
        mk_pool(&noruns, "p00000000000a", "aaaa|enc=x|loudnorm=0");
        std::fs::write(noruns.join("run_manifest.json"), br#"{"aug_copies":0}"#).unwrap();
        assert_eq!(migrate_slot_identity(&noruns, "sovits").unwrap(), IdentityOutcome::Committed);
        assert!(
            !noruns.join(super::super::trun::POOL_REF).exists(),
            "把一个 run 产物写到了槽根"
        );
        let _ = std::fs::remove_dir_all(noruns);

        // …而两个池时**不猜**。
        let two = layout3_slot("id_poolref2", serde_json::json!({ "aug_copies": 0 }));
        mk_pool(&two, "p00000000000a", "aaaa|enc=x|loudnorm=0");
        mk_pool(&two, "p00000000000b", "bbbb|enc=x|loudnorm=0");
        assert_eq!(migrate_slot_identity(&two, "sovits").unwrap(), IdentityOutcome::Committed);
        assert!(
            super::super::trun::pool_of_run(&super::super::trun::run_dirs(&two).unwrap()[0]).is_none(),
            "两个池时不许猜一条边出来"
        );

        // 读者对「记了但没答案」必须答 None:python 在**未迁移的槽**上写的正是 `null`,
        // 而一个空串会被当成一个真的池名传给 ④e 的回收。
        std::fs::write(run.join(super::super::trun::POOL_REF), br#"{"pool_id":null}"#).unwrap();
        assert!(super::super::trun::pool_of_run(&super::super::trun::run_dirs(&slot).unwrap()[0]).is_none());
        std::fs::write(run.join(super::super::trun::POOL_REF), br#"{"pool_id":""}"#).unwrap();
        assert!(
            super::super::trun::pool_of_run(&super::super::trun::run_dirs(&slot).unwrap()[0]).is_none(),
            "空串被当成了一个真的池名"
        );

        // …而 python 已经写下的那份不许被回填覆盖(它是从 run 内部看到的事实)。
        std::fs::write(run.join(super::super::trun::POOL_REF), br#"{"pool_id":"pFROMPYTHON"}"#)
            .unwrap();
        write_slot_meta(&slot, &SlotMeta { layout: 3, ..Default::default() }).unwrap();
        assert_eq!(migrate_slot_identity(&slot, "sovits").unwrap(), IdentityOutcome::Committed);
        assert_eq!(
            super::super::trun::pool_of_run(&super::super::trun::run_dirs(&slot).unwrap()[0]),
            Some("pFROMPYTHON".to_string()),
            "回填覆盖了 python 记下的事实"
        );
        let _ = std::fs::remove_dir_all(slot);
        let _ = std::fs::remove_dir_all(two);
    }

    /// ⛔ 准入的**下**界,而它守的是同一个「结构上恒空」的陷阱换了个入口:一个还没被前两步折过
    /// 的槽,它的池在**槽根**,而 `list_pools` 只看 `pools/` ⇒ plan 恒空 ⇒ 盖 layout 4 ⇒ 告诉
    /// python 用 v2 公式去对一个写着 v1 文本的槽根池 ⇒ 铸新池 + 全量重跑。
    ///
    /// 开机链保证健康的槽走到这一步已经是 3,所以「低于 3」只剩一种含义:前两步对它失败过。
    #[test]
    fn a_slot_the_earlier_folds_have_not_committed_is_never_stamped() {
        let slot = tmp_slot("id_tooearly");
        // layout 1 的形状:池产物与身份文件都在**槽根**,`pools/` 还不存在。
        std::fs::write(slot.join(FINGERPRINT), "aaaa|enc=x|loudnorm=0").unwrap();
        touch(&slot.join("0_gt_wavs").join("0_0.wav"));
        for layout in 0..super::super::trun::SLOT_LAYOUT_RUNS {
            write_slot_meta(&slot, &SlotMeta { layout, ..Default::default() }).unwrap();
            assert!(
                matches!(
                    migrate_slot_identity(&slot, "sovits").unwrap(),
                    IdentityOutcome::Refused(_)
                ),
                "layout {layout} 的槽被盖章了 —— 它的池还在槽根,plan 看不见它"
            );
            assert_eq!(read_slot_meta(&slot).unwrap().layout, layout, "layout {layout}: 盖章了");
            assert_eq!(identity_version(&slot), 1);
        }
        assert_eq!(
            std::fs::read_to_string(slot.join(FINGERPRINT)).unwrap(),
            "aaaa|enc=x|loudnorm=0",
            "槽根的身份一个字节都不许动"
        );
        let _ = std::fs::remove_dir_all(slot);
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
        // ⚠S142:措辞改了。这个槽**连 `pools/` 都没有**,所以它在新旧两种语义下都答 false ——
        // 原来那句「no fingerprint ⇒ no pool」描述的是旧实现的过滤条件,而不是这个夹具的形状。
        assert!(!slot_has_pool(&slot), "没有任何池目录 ⇒ 这个槽没预处理过");
        assert!(slot.join("run.json").is_file());
        let _ = std::fs::remove_dir_all(slot);
    }

    /// ⛔★S142 §E2E-M10-⒜ —— 一个**装满产物但指纹丢了**的池,照样算「这个槽预处理过」。
    ///
    /// 这是**唯一**分得开新旧两种语义的形状:旧实现用 `!fp_text.is_empty()` 过滤,于是对它
    /// 答 false —— 而它恰恰是代价**最大**的一格(没有指纹 ⇒ 谁也匹配不上 ⇒ 下一次运行必然
    /// 铸一个兄弟池、整份重跑)。
    /// ⚠ 另外三条既有用例在**两种**语义下都是绿的(空槽根本没有 `pools/`,而有指纹的池两边
    /// 都算数),所以少了这一条,那次改判据**没有任何判据看着**。
    #[test]
    fn a_pool_that_lost_its_fingerprint_still_counts_as_preprocessing() {
        let slot = tmp_slot("fp_lost");
        let pool = pools_root(&slot).join("p0000000000000");
        touch(&pool.join("0_gt_wavs").join("000_000.wav"));
        assert!(!pool.join(FINGERPRINT).exists(), "夹具前提:指纹确实不在盘上");

        assert!(slot_has_pool(&slot), "产物在盘上,那笔时间就得再付一遍 —— 指纹丢了不改变这一点");

        // 阴性对照:把那个池整个拿走 ⇒ 必须答 false,否则上面那句对任何输入都成立。
        std::fs::remove_dir_all(pools_root(&slot)).unwrap();
        assert!(!slot_has_pool(&slot));
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
