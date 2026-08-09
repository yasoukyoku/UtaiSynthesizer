//! Slot layout v3: the RUN container.
//!
//! ## What changes and why
//!
//! Layout 2 ([`super::tpool`]) split a family slot into "the preprocessing pool, keyed by its own
//! identity" and "everything else". Everything else is one training RUN, and until now a slot
//! could only ever hold one of them: 「重训」 was `remove_dir_all` of the whole slot
//! (`training::try_start`), so the second model in a workspace was bought by destroying the first.
//!
//! Layout 3 gives the run a directory of its own:
//!
//! ```text
//! <slot>/
//!   slot.json              ← {"layout": 3} — the migration commit point (same file as layout 2)
//!   pools/<pool_id>/       ← layout 2, untouched
//!   runs/<run_id>/
//!     run.json  run_manifest.json
//!     weights/  G_*.pth  D_*.pth | model_ckpt_steps_*.ckpt
//!     best_state.json  resume_state.json  resume_best/  resume_latest/
//!     config.json | config.yaml   filelist.txt | filelists/
//!     total_fea.npy  cluster/   eval/  lightning_logs/  events.out.tfevents.*  train.log
//!     aug_gate_report*.json  stop.flag  audition/  diffusion.yaml  diffusion/
//! ```
//!
//! ## Where this module currently stands (batch 2, step ③ landed — the bytes move now)
//!
//! * **The RESOLVERS are wired.** [`resolve_run_dir`] and [`run_dirs`] are what every reader of a
//!   run product goes through — the progress probes, `slot_holds_work`, `frozen_speakers_of_run`,
//!   `slot_info`, `scan_project_ckpts`, the export context, the audition paths, the storage
//!   totals. Nothing on disk moved to make that true: with no `runs/` container the answer IS the
//!   slot root, so the wiring batch was a byte-for-byte no-op that left every reader able to
//!   address a run once one exists.
//! * **The MIGRATION now runs**, from [`migrate_all`] at boot and again over the OLD root when a
//!   data-directory reclaim folds it. ⛔ Always AFTER `tpool::migrate_all` — see that function.
//! * **Where a START writes** is [`run_dir_for_start`], not [`resolve_run_dir`]: the guards resolve
//!   the run BEFORE the wipe (they judge the pre-wipe state), and the wipe takes the run directory
//!   with it.
//!
//! The ordering above was the whole point of the change. Moving the bytes before the readers were
//! run-aware would not merely break a listing; it silently destroys work, because a surprising
//! number of load-bearing predicates are spelled "is there a `G_*.pth` at the slot root". The
//! ones measured on this tree, each with what it costs:
//!
//! * `has_main_progress` goes false ⇒ `diff_partial_wipe` goes false ⇒ a shallow-diffusion
//!   「重训」 stops meaning "clear `diffusion/`" and becomes `remove_dir_all_robust(&workspace)`
//!   — taking the main model's runs AND layout 2's pools with it;
//! * the same flag drives `eff_aug_copies`: a diffusion run stops INHERITING the main model's
//!   augmentation count and starts using its own, which rebuilds the shared slices to a different
//!   recipe under a main model that is still resuming from them. The user changed nothing;
//! * `slot_holds_work` (named `workspace_holds_work` until step ②) goes false ⇒ the backend's
//!   refusal of an unconfirmed wipe stops
//!   firing (the dialog itself hangs off `WorkspaceInfo::exists`, so the prompt stays and only
//!   the guard disappears), and the sibling-slot `PROJECT_DATASET_IN_USE` pre-check fails open;
//! * `frozen_speakers` goes empty ⇒ the dataset page's REFUSAL goes with it: emptying a singer
//!   stops returning `DATASET_SPEAKERS_FROZEN`, the files go, and the speaker set changes under a
//!   model that can no longer resume from it. (The `drop_empty_speaker_dirs` flag beside it only
//!   removes an already-empty directory — the refusal is the part that was holding;)
//! * `project.json`'s `exported[].from_ckpt_rel` stops matching ⇒ every imported checkpoint loses
//!   `KeptReason::Exported` and becomes a cleanup candidate.
//!
//! So the order was: this data layer, then every reader routed through one resolver that accepts
//! BOTH shapes, and only then the migration. All three are done.
//!
//! ⚠ The list above is what breaks if the bytes move FIRST — it is not a list of things still
//! broken. Each of those readers now takes the run directory, and three of them can no longer be
//! handed a slot at all: [`RunDir`] is a type only this module can mint, which is what closes the
//! call sites no test can reach (`training::try_start` above all).
//!
//! ## The invariant step ③ establishes
//!
//! **layout ≥ 3 ⟹ every run product lives under `runs/`.** It is not automatic — [`run_dir_for_start`]
//! is what holds it up, and two shapes would break it without that:
//!
//! * a full 「重训」 deletes the WHOLE slot, `slot.json` included, so the slot falls back to layout 0
//!   and the slot root is legitimately the run again (the shape both migrations already handle);
//! * a slot that had nothing to move is committed at layout 3 with no `runs/` container at all, and
//!   its first training would otherwise write to the slot root under a marker that says otherwise.
//!
//! That second one reads fine today ([`run_dirs`] answers `[slot]` when the container is absent),
//! and goes silently wrong the moment a SECOND run appears beside it: the slot root stops being
//! scanned, so those checkpoints leave the archive listing while staying on disk — precisely the
//! failure `tproject::scan_project_ckpts` exists to end.

use std::path::{Path, PathBuf};

use super::tpool::{self, SlotMeta};
use super::tproject;
use crate::{Result, UtaiError};

/// Container for every run of one slot.
///
/// ⚠ Load-bearing in the same three ways [`tpool::POOLS_DIR`] is, and pinned by the same kind of
/// test: not a family name, no `.` prefix, and it must appear in `tproject::WORKSPACE_SUBDIRS`.
pub const RUNS_DIR: &str = "runs";

/// Slot layout that has a `runs/` container.
///
/// ⛔ Deliberately NOT a bump of [`tpool::SLOT_LAYOUT`]. That constant gates
/// `tpool::migrate_slot`, which returns early on `layout >= SLOT_LAYOUT` and otherwise computes a
/// plan; raising it to 3 would make every already-folded slot compute an EMPTY plan, take the
/// "nothing to move" branch and stamp `layout: 3` — marking the slot migrated without moving a
/// single run product. The two migrations advance the same file, one step each.
pub const SLOT_LAYOUT_RUNS: u32 = 3;

/// Staging directory for a run migration in flight. Dot-prefixed for the same reason layout 2's
/// is: a half-filled `runs/<id>/` would be readable as a run.
const STAGING_PREFIX: &str = ".mig_run_";

/// The staging prefix, for the verifier that builds torn states from outside this module.
///
/// Exposed rather than duplicated, for the reason [`tpool::staging_prefix`] gives: enumerating the
/// crash points means creating exactly the half-migrated shapes this module leaves behind, and a
/// second copy of this string in the verifier would let the two drift until the "crash recovery"
/// leg silently tested nothing. The wiring batch deleted this accessor because it had no consumer
/// then and an anti-drift mechanism with only one end is a claim nothing can falsify; it comes back
/// here WITH its caller, `tests/trun_migrate.rs`.
pub fn staging_prefix() -> &'static str {
    STAGING_PREFIX
}

// ─────────────────────────── the decision table ───────────────────────────

/// ONE top-level slot entry that belongs to a RUN.
///
/// `Exact` for fixed names, `Prefix` for the families whose names carry step numbers or host
/// names (`G_2333333.pth`, `events.out.tfevents.<epoch>.<host>.<pid>.<n>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunEntry {
    Exact(&'static str),
    Prefix(&'static str),
}

/// ⛔ **THE decision table.** Every top-level slot entry that moves into the run.
///
/// It is the complement of [`tpool::POOL_ENTRIES`] over the names this repo produces, and the two
/// are asserted disjoint below — a name in both tables would be moved twice, and the second move
/// would find nothing and silently succeed.
///
/// ## Three entries that look optional and are not
///
/// * `weights/` moves as a DIRECTORY. The sovits converter finds `weights/config.json` by
///   ADJACENCY to the `.pth` it is handed (`converter/architectures/sovits_v4.py`) and, when it is
///   not there, falls back to weight-shape inference with a warning — fail-OPEN. Separating the
///   two would cost the exported sidecar its speaker map with no error anywhere.
/// * `total_fea.npy` and `cluster/` are the two artifacts layout 2 deliberately LEFT at the slot
///   root (`tpool::POOL_ENTRIES`' doc comment says so, and reason 2 there is precisely that the
///   publish chain probes them at fixed slot-relative names and fails open). That reasoning was
///   about the POOL and does not survive per-run: both are rebuilt wholesale by every run, so two
///   runs sharing a slot root means the publish chain hands one model the OTHER one's retrieval
///   matrix. Under layout 3 they are run products, and the two probes
///   (`commands::training::get_slot_export_context` and its twin in `TrainingPage.tsx`) have to be
///   re-pointed in the same batch that turns the migration on.
/// * `diffusion/` moves whole, and it must stay ONE LEVEL DOWN from the run root rather than
///   becoming the run root itself: the diffusion snapshot scan slices checkpoint paths by a fixed
///   prefix length (`SNAPSHOT_DIR_MIN_LEN = 6` in `utai_train/resume_state.py`), and a run root
///   would put `eval/` (4) and `logs/` (4) inside its scope.
///   ⚠ The assertion this used to cite as its guard does NOT guard it: `training::tests`'
///   two-element loop checks that PYTHON's own snapshot directory names (`resume_best`,
///   `resume_latest`) clear the minimum, and knows nothing about run depth — `diffusion` (9) and
///   a minted `r`+12hex id (13) both pass it. What is asserted there now is the MECHANISM: that
///   `eval` and `logs` fall BELOW the minimum, which is precisely why the run root cannot be the
///   expdir. Turning that into a rule a test can fail is the batch that hands python its expdir.
pub const RUN_ENTRIES: &[RunEntry] = &[
    // ── run metadata ────────────────────────────────────────────────────────────────────
    RunEntry::Exact("run.json"),
    RunEntry::Exact("run_manifest.json"),
    // ── weights and the resume sidecars ─────────────────────────────────────────────────
    RunEntry::Exact("weights"),
    RunEntry::Exact("best_state.json"),
    RunEntry::Exact("resume_state.json"),
    RunEntry::Exact("resume_best"),
    RunEntry::Exact("resume_latest"),
    // `G_<step>.pth` / `D_<step>.pth` (rvc, sovits, sovits_v2) and
    // `model_ckpt_steps_<global>.ckpt` (vocoder)
    RunEntry::Prefix("G_"),
    RunEntry::Prefix("D_"),
    RunEntry::Prefix("model_"),
    // ── per-run configuration and lists ─────────────────────────────────────────────────
    RunEntry::Exact("config.json"),
    RunEntry::Exact("config.yaml"),
    RunEntry::Exact("filelist.txt"),
    RunEntry::Exact("filelists"),
    // ── retrieval assets: pool-shaped, run-owned (see the doc comment) ──────────────────
    RunEntry::Exact("total_fea.npy"),
    RunEntry::Exact("cluster"),
    // ── shallow diffusion (lives inside the sovits slot) ────────────────────────────────
    RunEntry::Exact("diffusion"),
    RunEntry::Exact("diffusion.yaml"),
    // ── logs, reports, control ──────────────────────────────────────────────────────────
    RunEntry::Exact("eval"),
    RunEntry::Exact("lightning_logs"),
    RunEntry::Exact("train.log"),
    RunEntry::Exact("stop.flag"),
    RunEntry::Prefix("events.out.tfevents"),
    RunEntry::Prefix("aug_gate_report"),
    // ── the audition cache ──────────────────────────────────────────────────────────────
    // Keyed by checkpoint file STEM, and the stem is a pure function of the training name
    // (`slugify(name)_e<n>_s<n>`), so two runs of the same slot produce colliding keys. A cache
    // hit is decided by the presence of `model.json` alone — no mtime, no content check — and the
    // archive's import path prefers an existing `.onnx` beside it, so a collision installs the
    // OTHER run's converted weights. It also stores the tested vocal RANGE, which suppresses
    // re-measurement. This is the entry that turns "two runs" into wrong data rather than a stale
    // preview, and today it is survivable only because 重训 deletes the whole slot.
    RunEntry::Exact("audition"),
];

/// Does this top-level slot entry move into the run?
pub fn is_run_entry(name: &str) -> bool {
    RUN_ENTRIES.iter().any(|e| match e {
        RunEntry::Exact(n) => *n == name,
        RunEntry::Prefix(p) => name.starts_with(p),
    })
}

// ─────────────────────────── run ids ───────────────────────────

/// Names a run directory may never take, because somewhere a path SUBSTRING decides what a
/// checkpoint row IS. Each one has two enforcement points, one per language boundary — which is
/// exactly why they are asserted here rather than written in a comment.
///
/// * `diffusion` — `rowIsDiffusion` in `TrainingPage.tsx` and `commands::storage`'s
///   `rel.contains("/diffusion/")` both classify by substring; a run named this would report a GAN
///   run's step count as shallow-diffusion progress and swap the archive row's action button.
/// * `resume_best` / `resume_latest` — `tproject::default_resume_record` filters on
///   `rel.contains("/resume_best/")` to answer 「可从第 N 步继续」, and the frontend has the
///   mirror regex. Every rolling checkpoint of such a run would read as a best snapshot.
const RESERVED_RUN_NAMES: &[&str] = &["diffusion", "resume_best", "resume_latest"];

/// `r` + 12 hex, derived with sha256 from `seed`.
///
/// ⛔ sha2, never `DefaultHasher`: `training::slugify` uses the latter and the standard library
/// does not promise it is stable across releases, which is the recorded reason `model_slug` cannot
/// be a run key in the first place. A run id that changes when the toolchain changes would orphan
/// every directory on disk.
///
/// ⚠ ASCII by construction, and that is load-bearing rather than tidy: the vendored
/// `latest_checkpoint_path` sorts by `int("".join(filter(str.isdigit, path)))`, and `str.isdigit`
/// accepts non-ASCII digits — `٣` in a path segment silently inflates the sort key, `²` makes the
/// call raise. The ordering itself is safe for a hex id (every candidate in one run directory
/// shares the same digit prefix, so the key stays monotone in the step), which was measured rather
/// than argued.
pub fn run_id_for(seed: &str) -> String {
    use sha2::{Digest, Sha256};
    let d = Sha256::digest(seed.as_bytes());
    let mut s = String::with_capacity(13);
    s.push('r');
    for b in d.iter().take(6) {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// The id the layout migration gives a slot's single pre-existing run.
///
/// Deterministic on purpose: the crash-point leg of the migration verification kills the process
/// at every intermediate state and restarts, and a random id would make each retry build a
/// different directory — the rollback would then be comparing two shapes that were never supposed
/// to be equal. One legacy run per slot, so uniqueness within the slot is structural.
pub fn legacy_run_id(family: &str) -> String {
    run_id_for(&format!("legacy-run/{family}"))
}

/// Is this a name a run directory may take?
///
/// Kept as a predicate (rather than a comment next to the minting code) because the wiring batch
/// mints ids from a different source and has to ask the same question.
pub fn run_id_is_usable(id: &str) -> bool {
    // ⚠ Every clause below is one a mutation can turn red. Three more were written first and
    // then DELETED, because probing them showed the charset clause already subsumes each one —
    // and a rule nothing can reach is a rule that rots into a false claim of protection. Their
    // reasons live here instead, because the reasons are real:
    //
    // * ASCII (`is_ascii()`): the vendored checkpoint sort is
    //   `int("".join(filter(str.isdigit, path)))`, and `str.isdigit` is NOT `isdecimal` — `٣` in
    //   a path segment silently inflates the key, `²` makes the call raise.
    // * no leading `.`: dot entries are staging and every slot scan skips them.
    // * no `_e<digits>_s<digits>.`: `TrainingPage.tsx` recovers a release snapshot's epoch with
    //   `/_e(\d+)_s\d+\./` over the WHOLE project-relative path and takes the FIRST match, so a
    //   run directory (which sorts ahead of the file name) could hijack it. It cannot: the
    //   pattern needs a `.` immediately after the step digits, and what follows a run id inside a
    //   rel is always `/`.
    //
    // All three reduce to "no character outside `[A-Za-z0-9_-]`", which is the clause that stays.
    !id.is_empty()
        && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        && !RESERVED_RUN_NAMES.contains(&id)
        && !tproject::FAMILIES.contains(&id)
}

// ─────────────────────────── paths and listing ───────────────────────────

pub fn runs_root(slot: &Path) -> PathBuf {
    slot.join(RUNS_DIR)
}

/// Which run a PROJECT-RELATIVE artifact path belongs to; `""` when the slot root is the run.
///
/// ★§F2⒝ batch 2 step ④. Derived rather than threaded through the scanners, and that is on
/// purpose: the export LEDGER already keys on exactly this prefix (`repoint_ledger` rewrites
/// `<family>/` rows into `<family>/runs/<id>/` ones), so a second derivation of "which run is this
/// path in" would be a second chance to disagree with the ledger about the same string. One
/// function, both readers.
///
/// `""` is a POSITIVE answer, not a failure: it is the layout ≤ 2 shape where the slot root holds
/// the run products, which is what [`resolve_run_dir`] answers with too.
pub fn run_id_in_rel(rel: &str, family: &str) -> String {
    let rel = rel.replace('\\', "/");
    let Some(tail) = rel.strip_prefix(&format!("{family}/{RUNS_DIR}/")) else {
        return String::new();
    };
    // A bare `<family>/runs/<id>` with nothing after it is not an artifact path; requiring the
    // separator keeps this from naming a run for the container listing itself.
    tail.split_once('/').map(|(id, _)| id.to_string()).unwrap_or_default()
}

/// A directory that has been RESOLVED to be one run's home.
///
/// ⛔ The point of the newtype is the thing a test cannot reach. Every per-run reader used to take
/// a `&Path`, and the family slot is also a `&Path` — so the whole per-run change came down to
/// remembering which of two identical-looking variables to pass at each of ~30 call sites, inside
/// functions (`try_start`) that no unit test drives. Getting one wrong is silent: the probe answers
/// "nothing here", `diff_partial_wipe` goes false, and a shallow-diffusion 「重训」 deletes the
/// whole slot. With this type that mistake does not compile.
///
/// It `Deref`s to `Path`, so a resolved run can still be handed to anything that takes a plain
/// path (`<run>/diffusion` is not itself a run, for instance) — the coercion only ever goes that
/// way, which is the direction that is safe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunDir(PathBuf);

impl RunDir {
    /// ⚠ Only [`resolve_run_dir`] and [`run_dirs`] may mint one in production — that is the entire
    /// guarantee. Tests build fixtures directly and are allowed to say so out loud.
    #[cfg(test)]
    pub fn for_test(p: PathBuf) -> Self {
        RunDir(p)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl std::ops::Deref for RunDir {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

impl AsRef<Path> for RunDir {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

#[derive(Debug, Clone)]
pub struct RunInfo {
    pub id: String,
    pub dir: PathBuf,
}

/// Every run of one slot, sorted by id so listings never wobble.
pub fn list_runs(slot: &Path) -> Vec<RunInfo> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(runs_root(slot)) else {
        return out;
    };
    for entry in rd.flatten() {
        let id = entry.file_name().to_string_lossy().into_owned();
        // `.` entries are staging, never a run — a half-migrated tree must not be selectable.
        if id.starts_with('.') || !entry.path().is_dir() {
            continue;
        }
        // ★§F2⒝ batch 2 step ④ — a name this app could never have minted is NOT a run.
        //
        // ⛔ This is a REACHABLE-TODAY fix, not preparation. Nothing else filtered here, so one
        // stray directory under `<slot>/runs/` — a backup tool's copy, a half-finished manual
        // rename, an antivirus quarantine folder — made the slot answer `RUN_AMBIGUOUS`, and that
        // answer is `?`-ed in front of BOTH `start_training` (`commands/training.rs`, before the
        // manager ever runs) and the project-detail command (which collects every slot through one
        // `Result`) ⇒ no training of ANY backend in that project, and a project page that will not
        // open, both reporting a raw untranslated Rust string with an absolute path in it.
        //
        // Skipping it is the lesser evil and it is not silent: the slot keeps working, the entry
        // still counts toward the slot's byte total (`dir_size` walks the directory, not this
        // list), and the reason is one `warn!` away. Widening `run_id_is_usable` instead would be
        // the mistake — it is the same predicate the minting paths assert on, so the two must
        // agree or a minted id could fail to list.
        if !run_id_is_usable(&id) {
            tracing::warn!(
                "ignoring {} — not a run id this app could have minted; \
                 it is invisible to every per-run reader (but still occupies disk)",
                entry.path().display()
            );
            continue;
        }
        out.push(RunInfo { id, dir: entry.path() });
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

/// Where a run's products live — THE single answer, for both layouts.
///
/// With no `runs/` container the slot root IS the run: it holds `run.json`, the checkpoints and
/// the resume sidecars, exactly as it has since S76. That arm fires on the POSITIVE fact that the
/// container is absent, and it is the reason every reader can be re-pointed through this function
/// in a batch that moves no bytes at all — the same shape `utai_train/pool.py` uses to let python
/// read pools before Rust started minting them.
///
/// ⚠ It is NOT "look in the new place, fall back to the old one on failure". A named run that
/// does not exist is an error, not a reason to answer with the slot root: that is how a wiring
/// mistake turns into a run silently training into another run's directory.
///
/// ## Singular or plural is not a style choice — see [`run_dirs`]
///
/// `None` here asserts a POSITIVE FACT: *this slot has at most one run*. It is checked, and when
/// it is false the answer is an error rather than a guess. That is what makes the wiring batches
/// safe to land one at a time: while the migration is off, and again after it has folded each
/// slot's single legacy run, `None` is exactly right everywhere. The batch that starts minting a
/// SECOND run per slot is therefore also the batch in which every one of these call sites has to
/// be handed a real id — and any that is forgotten fails loudly here instead of silently reading
/// another run's weights.
pub fn resolve_run_dir(slot: &Path, run_id: Option<&str>) -> Result<RunDir> {
    match run_id {
        Some(id) => {
            if !run_id_is_usable(id) {
                return Err(UtaiError::Training(format!("RUN_ID_INVALID: {id}")));
            }
            let dir = runs_root(slot).join(id);
            if !dir.is_dir() {
                return Err(UtaiError::Training(format!("RUN_NOT_FOUND: {id}")));
            }
            Ok(RunDir(dir))
        }
        None => {
            let runs = list_runs(slot);
            match runs.len() {
                // layout ≤ 2: the slot root is the one run
                0 => Ok(RunDir(slot.to_path_buf())),
                1 => Ok(RunDir(runs[0].dir.clone())),
                // Refuses to guess for the same reason `tpool::sole_pool_fingerprint` does: with
                // several runs present, picking one is a decision the caller has to have made.
                _ => Err(UtaiError::Training(format!(
                    "RUN_AMBIGUOUS: {} runs in {}",
                    runs.len(),
                    slot.display()
                ))),
            }
        }
    }
}

/// EVERY run directory of a slot — the answer for questions about the SLOT AS A WHOLE.
///
/// With no `runs/` container the answer is `[slot]`, for the same positive-fact reason
/// [`resolve_run_dir`] gives: the slot root IS the one run.
///
/// ## Which resolver a caller wants is decided by the QUESTION, not by convenience
///
/// * **Plural** — "what is on disk", "what would this wipe destroy", "what must the cleanup be
///   able to reach", "how many bytes is this slot". Answering any of these with one run is how
///   gigabytes become invisible AND unreclaimable, which is the exact failure
///   `tproject::scan_project_ckpts` was written to end. A guard that answers one of these with
///   `false` because it only looked at one run fails OPEN, and every one of those guards protects
///   against destroying work.
/// * **Singular** — "which weights is THIS run resuming from", "whose emb_g row is speaker 3",
///   "what version was THIS run frozen at". [`resolve_run_dir`] refuses to guess between runs.
///
/// ⚠ What this deliberately does NOT do: once a slot has a `runs/` container, the slot ROOT is no
/// longer scanned. A run product still sitting there means the migration classified it as
/// `unknown` and left it behind (`plan_slot_runs`), which is a loud `warn!` at migration time and
/// a gate against the python sources — not something to paper over here by scanning both, because
/// "both" is how one checkpoint gets listed twice and deleted once.
pub fn run_dirs(slot: &Path) -> Vec<RunDir> {
    let runs = list_runs(slot);
    if runs.is_empty() {
        // Also the shape of a MIGRATED slot whose runs have all been deleted: the root holds no
        // run products then either, so scanning it finds nothing and costs nothing.
        return vec![RunDir(slot.to_path_buf())];
    }
    runs.into_iter().map(|r| RunDir(r.dir)).collect()
}

/// Where a START writes — resolved AFTER the wipe, and it CREATES the directory.
///
/// ⛔ A different question from [`resolve_run_dir`], and the difference was a real hole:
/// `training::try_start` resolves the run BEFORE any deletion (its guards have to judge the
/// pre-wipe state), then a full 「重训」 removes the whole slot and re-creates only the SLOT root.
/// Everything written next — `run_manifest.json`, `run.json`, `stop.flag` — addresses a run
/// directory the wipe deleted. That failure is not loud-and-final either: the first press errors
/// out with an io NotFound *after* the slot has already been emptied, and the second press
/// succeeds (no `runs/` left ⇒ the slot root answers), so it reads as a one-off glitch while
/// having silently taken the slot back to layout 0.
///
/// The answers, and why each is a positive fact rather than a fallback:
/// * **one run** — that is the run, migrated or not;
/// * **no runs, layout < 3** — the slot root IS the run, exactly as [`resolve_run_dir`] says;
/// * **no runs, layout ≥ 3** — the marker asserts the container is where runs live, so MINT one
///   rather than writing beside it. This is what keeps "layout ≥ 3 ⟹ products under `runs/`" true
///   for a slot that had nothing to move at migration time, and for one whose runs were all
///   deleted;
/// * **several runs** — refuse. The batch that mints a second run hands this an id.
pub fn run_dir_for_start(slot: &Path, family: &str) -> Result<RunDir> {
    let runs = list_runs(slot);
    match runs.len() {
        1 => Ok(RunDir(runs.into_iter().next().unwrap().dir)),
        0 => {
            if !tpool::read_slot_meta(slot).is_some_and(|m| m.layout >= SLOT_LAYOUT_RUNS) {
                return Ok(RunDir(slot.to_path_buf()));
            }
            let id = legacy_run_id(family);
            debug_assert!(run_id_is_usable(&id));
            let dir = runs_root(slot).join(&id);
            std::fs::create_dir_all(&dir)
                .map_err(|e| UtaiError::Training(format!("RUN_CREATE_FAILED: {e}")))?;
            Ok(RunDir(dir))
        }
        _ => Err(UtaiError::Training(format!(
            "RUN_AMBIGUOUS: {} runs in {}",
            runs.len(),
            slot.display()
        ))),
    }
}

// ─────────────────────────── migration (layout 2 → 3) ───────────────────────────

#[derive(Debug, PartialEq, Eq)]
pub enum RunOutcome {
    /// `slot.json` already says layout ≥ 3, or the slot does not exist.
    AlreadyDone,
    /// Nothing to move (a slot that has never trained) — committed anyway, so the next boot does
    /// not look again.
    Committed,
    /// Run products were folded into `runs/<id>/`.
    Migrated(String),
}

/// What the run migrator would do, computed BEFORE anything is touched.
#[derive(Debug, Default)]
pub struct RunPlan {
    pub moving: Vec<String>,
    pub staying: Vec<String>,
    /// Stays AND the table does not know about it. Not an error — see [`plan_slot_runs`].
    pub unknown: Vec<String>,
}

/// Classify a slot's top-level entries for the run migration.
///
/// Same fail-SAFE posture as layout 2, and for the same reason: "do not move it" leaves an entry
/// exactly where it is today, which is always a defensible answer, whereas aborting would strand a
/// real migration behind a `Thumbs.db`. The failure this cannot prevent — a run product missing
/// from the table — is not prevented by aborting either; it would simply be left at the slot root,
/// where [`resolve_run_dir`]'s legacy arm still finds nothing and the run reads as fresh. What
/// actually prevents it is driving this table against the python sources from a gate.
pub fn plan_slot_runs(slot: &Path) -> RunPlan {
    let mut plan = RunPlan::default();
    let Ok(rd) = std::fs::read_dir(slot) else {
        return plan;
    };
    for entry in rd.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        // dot entries are staging/tombstones and belong to whoever created them
        if name.starts_with('.') {
            continue;
        }
        // the layout-2 container and both commit points stay where they are
        if name == tpool::POOLS_DIR || name == tpool::SLOT_META || name == RUNS_DIR {
            continue;
        }
        if is_run_entry(&name) {
            plan.moving.push(name);
        } else {
            // A pool product still at the slot root means layout 2 left it behind (an unknown
            // entry, or a slot that never ran the fold). Naming it here keeps the two tables from
            // both claiming it.
            if !tpool::POOL_ENTRIES.iter().any(|p| p.name == name) {
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

/// Fold EVERY slot of every project into layout 3.
///
/// ⛔ **MUST run after `tpool::migrate_all`, and the reason is byte-level rather than stylistic.**
/// This function commits `layout: 3` into the same `slot.json`, while `tpool::migrate_slot` returns
/// early on `layout >= tpool::SLOT_LAYOUT` (2). Folding the runs first would therefore make the
/// POOL migration answer `AlreadyDone` for that slot forever — and it would do so silently, because
/// [`plan_slot_runs`] classifies a pool product left at the slot root as `staying`, not `unknown`,
/// so not even a warn line would appear.
///
/// The concurrency guard sits in this public entry point rather than beside the walk, mirroring
/// `tpool::migrate_all`: every test in this module drives the per-slot [`migrate_slot_runs`], so
/// there is nothing here for a live dev build to break (`tproject::migrate_legacy_layout` splits
/// the two only because its own tests call the walk directly).
///
/// Unlike the pool half this needs the project ID as well as the path, because the export ledger
/// [`migrate_slot_runs`] re-points lives one level up. The DIRECTORY NAME is that identity —
/// `tproject::read_meta` overwrites the stored field with it precisely so a hand-renamed directory
/// stays addressable.
pub fn migrate_all(data_dir: &Path) {
    let root = tproject::training_root(data_dir);
    if !root.is_dir() {
        return;
    }
    if crate::crashlog::other_instance_alive() {
        tracing::warn!("training run migration postponed: another live instance detected");
        return;
    }
    let Ok(rd) = std::fs::read_dir(&root) else { return };
    let (mut migrated, mut failed) = (0usize, 0usize);
    for entry in rd.flatten() {
        let proj = entry.path();
        // `project.json` is the authority for "this is a project", exactly as `tpool::migrate_all`
        // has it; `.del_*` tombstones and `.migrating_*` markers do not have one.
        if !proj.join(tproject::PROJECT_META).is_file() {
            continue;
        }
        let Some(project_id) = proj.file_name().map(|n| n.to_string_lossy().into_owned()) else {
            continue;
        };
        for family in tproject::FAMILIES {
            if !proj.join(family).is_dir() {
                continue;
            }
            match migrate_slot_runs(data_dir, &project_id, family) {
                Ok(RunOutcome::Migrated(id)) => {
                    migrated += 1;
                    tracing::info!("run layout: {project_id}/{family} -> runs/{id}");
                }
                Ok(_) => {}
                Err(e) => {
                    failed += 1;
                    tracing::error!("run layout: {project_id}/{family} could not be migrated: {e}");
                }
            }
        }
    }
    if migrated > 0 || failed > 0 {
        tracing::info!("run layout migration: {migrated} slot(s) folded, {failed} failed");
    }
}

/// Fold one slot's run products into `runs/<id>/`, and re-point the project's export ledger at the
/// same time. Idempotent.
///
/// ⚠ Takes the project rather than the slot because the ledger lives one level up: the checkpoint
/// paths it stores are PROJECT-relative, so moving the files without rewriting them costs every
/// imported checkpoint its `KeptReason::Exported` protection. Doing it in a second pass would open
/// a window where the two disagree, and that window is exactly when a cleanup deletes work.
pub fn migrate_slot_runs(data_dir: &Path, project_id: &str, family: &str) -> Result<RunOutcome> {
    let slot = tproject::family_dir(data_dir, project_id, family);
    if !slot.is_dir() {
        return Ok(RunOutcome::AlreadyDone);
    }
    reconcile_staging(&slot)?;
    if tpool::read_slot_meta(&slot).is_some_and(|m| m.layout >= SLOT_LAYOUT_RUNS) {
        return Ok(RunOutcome::AlreadyDone);
    }
    let plan = plan_slot_runs(&slot);
    for u in &plan.unknown {
        tracing::warn!(
            "training slot {}: unrecognised entry {u:?} left at the slot root",
            slot.display()
        );
    }
    let commit = |slot: &Path| -> Result<()> {
        let mut meta = tpool::read_slot_meta(slot).unwrap_or(SlotMeta { layout: 0, extra: Default::default() });
        meta.layout = SLOT_LAYOUT_RUNS;
        tpool::write_slot_meta(slot, &meta)
    };
    if plan.moving.is_empty() {
        commit(&slot)?;
        return Ok(RunOutcome::Committed);
    }

    let run_id = legacy_run_id(family);
    debug_assert!(run_id_is_usable(&run_id));
    let staging = slot.join(format!("{STAGING_PREFIX}{run_id}"));
    std::fs::create_dir_all(&staging)
        .map_err(|e| UtaiError::Training(format!("RUN_MIGRATE_FAILED: {e}")))?;

    let step = || -> Result<()> {
        for name in &plan.moving {
            let from = slot.join(name);
            if !from.exists() {
                continue;
            }
            crate::util::rename_with_retry(&from, &staging.join(name), "RUN_MIGRATE")
                .map_err(UtaiError::Training)?;
        }
        std::fs::create_dir_all(runs_root(&slot))
            .map_err(|e| UtaiError::Training(format!("RUN_MIGRATE_FAILED: {e}")))?;
        crate::util::rename_with_retry(
            &staging,
            &runs_root(&slot).join(&run_id),
            "RUN_MIGRATE_COMMIT",
        )
        .map_err(UtaiError::Training)?;
        Ok(())
    };
    if let Err(e) = step() {
        match roll_back(&slot) {
            Ok(()) => {}
            Err(re) => tracing::error!("training slot {}: rollback also failed: {re}", slot.display()),
        }
        return Err(e);
    }

    // Ledger BEFORE the commit point: a kill between the two leaves `slot.json` at layout 2, so
    // the next boot re-enters this function, finds `moving` empty (the files are already down
    // there), and would stamp layout 3 without ever re-pointing the ledger. Rewriting first makes
    // the only reachable torn state "files moved, ledger correct, not yet committed".
    let repointed = repoint_ledger(data_dir, project_id, family, &run_id)?;
    if repointed > 0 {
        tracing::info!(
            "run layout: re-pointed {repointed} export ledger row(s) of {project_id}/{family}"
        );
    }
    commit(&slot)?;
    Ok(RunOutcome::Migrated(run_id))
}

/// Rewrite `<family>/…` export-ledger rows to `<family>/runs/<run_id>/…`.
///
/// Idempotent by construction (a row already under `runs/` is skipped), which is what lets it run
/// before an uncommitted migration and again after a restart. Returns how many rows changed.
fn repoint_ledger(data_dir: &Path, project_id: &str, family: &str, run_id: &str) -> Result<usize> {
    let Some(mut meta) = tproject::read_meta(data_dir, project_id) else {
        // A project without readable metadata has no ledger to protect; the migration itself is
        // still correct, and `cleanup_snapshots` refuses outright on an unreadable project.json.
        return Ok(0);
    };
    let old_prefix = format!("{family}/");
    let new_prefix = format!("{family}/{RUNS_DIR}/{run_id}/");
    let mut changed = 0usize;
    for e in meta.exported.iter_mut() {
        if !e.from_ckpt_rel.starts_with(&old_prefix) {
            continue;
        }
        if e.from_ckpt_rel.starts_with(&format!("{family}/{RUNS_DIR}/")) {
            continue;
        }
        let tail = &e.from_ckpt_rel[old_prefix.len()..];
        e.from_ckpt_rel = format!("{new_prefix}{tail}");
        changed += 1;
    }
    if changed > 0 {
        tproject::write_meta(data_dir, &meta)?;
    }
    Ok(changed)
}

/// Undo a torn run migration: whatever is in staging goes back to the slot root.
///
/// Idempotent and mirror-image, exactly like `tpool::reconcile_staging`.
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
        tracing::warn!("training slot {}: rolling back a torn run migration", slot.display());
        let Ok(inner) = std::fs::read_dir(&dir) else { continue };
        for e in inner.flatten() {
            let name = e.file_name();
            let back = slot.join(&name);
            if back.exists() {
                // Something re-created it at the root. Leave BOTH: one of them holds a trained
                // checkpoint and this code cannot tell which.
                tracing::warn!(
                    "training slot {}: {:?} exists at the root already — leaving the staged copy in place",
                    slot.display(),
                    name
                );
                continue;
            }
            crate::util::rename_with_retry(&e.path(), &back, "RUN_MIGRATE_UNDO")
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
    use crate::training::tproject::{project_dir, write_meta, ExportedModel, ProjectMeta};

    fn tmp_data(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("utai_trun_{}_{}", tag, uuid::Uuid::new_v4()));
        std::fs::create_dir_all(d.join("training")).unwrap();
        d
    }

    fn touch(p: &Path) {
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(p, b"x").unwrap();
    }

    /// The real layout-2 RVC slot, entry for entry, as the frozen post-migration workspace on this
    /// machine actually has it (`TESTING/s122_f2b1/LAYOUT2_migrated_and_trained_111_efa35241/rvc`).
    /// Using the measured shape rather than an invented one is the point: the entries that get
    /// forgotten are the ones nobody thinks to invent (here: two TensorBoard files, `audition/`).
    fn layout2_rvc_slot(slot: &Path) {
        for d in ["audition", "weights"] {
            std::fs::create_dir_all(slot.join(d)).unwrap();
        }
        touch(&slot.join("weights").join("m_e14_s147.pth"));
        touch(&slot.join("audition").join("m_e14_s147").join("audition.wav"));
        for f in [
            "best_state.json",
            "config.json",
            "D_2333333.pth",
            "G_2333333.pth",
            "events.out.tfevents.1784135491.NucBox_k11.25768.0",
            "events.out.tfevents.1784135788.NucBox_k11.9860.0",
            "filelist.txt",
            "run.json",
            "run_manifest.json",
            "total_fea.npy",
            "train.log",
        ] {
            touch(&slot.join(f));
        }
        // layout 2: the pool is already folded, and its commit point is present
        touch(&slot.join(tpool::POOLS_DIR).join("p55f7335bfa22").join("0_gt_wavs").join("0.wav"));
        std::fs::write(
            slot.join(tpool::POOLS_DIR).join("p55f7335bfa22").join("dataset.fingerprint"),
            b"abc123",
        )
        .unwrap();
        std::fs::write(slot.join(tpool::SLOT_META), br#"{"layout":2}"#).unwrap();
    }

    /// (relative path, byte length) of every file under `dir`, sorted.
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

    /// ⛔ REACHABLE TODAY, and it takes down more than the slot it is in.
    ///
    /// `list_runs` used to accept every non-dot directory under `runs/`, so ONE stray entry — a
    /// backup tool's copy, an antivirus quarantine folder, an abandoned manual rename — made the
    /// slot answer `RUN_AMBIGUOUS`. That answer is `?`-ed in front of `start_training` (so no
    /// backend in the project can train) and inside the project-detail command's per-slot
    /// `collect::<Result<_>>` (so the page will not open at all), both surfacing a raw
    /// untranslated Rust string. The stray directory is not a run and never was.
    ///
    /// ⚠ The assertion is deliberately two-sided: skipping it must NOT also make a legitimately
    /// minted id disappear, because `list_runs` and the minting paths share `run_id_is_usable` and
    /// a one-sided change there is how a real run becomes invisible (fail-OPEN for every byte).
    #[test]
    fn a_name_this_app_could_not_have_minted_is_not_a_run() {
        let data = tmp_data("stray");
        let slot = project_dir(&data, "p_11111111").join("rvc");
        let real = legacy_run_id("rvc");
        touch(&runs_root(&slot).join(&real).join("G_1.pth"));
        // what a human or a tool leaves behind
        for stray in ["G_2333333.pth backup", "runs.bak", "副本", "rvc"] {
            std::fs::create_dir_all(runs_root(&slot).join(stray)).unwrap();
        }
        let listed: Vec<String> = list_runs(&slot).into_iter().map(|r| r.id).collect();
        assert_eq!(listed, vec![real.clone()], "only the minted id is a run");
        // …and therefore the two commands that used to die still answer
        assert_eq!(
            resolve_run_dir(&slot, None).unwrap().path(),
            runs_root(&slot).join(&real),
            "one stray directory must not make the slot unanswerable"
        );
        assert_eq!(run_dirs(&slot).len(), 1);
        // the opposite direction: every id the app can mint must survive the filter
        for family in tproject::FAMILIES {
            assert!(
                run_id_is_usable(&legacy_run_id(family)),
                "the filter must never hide a minted run: {family}"
            );
        }
        let _ = std::fs::remove_dir_all(&data);
    }

    /// The record ↔ run attribution, in BOTH layouts, from the same function the ledger uses.
    #[test]
    fn a_relative_path_names_the_run_it_came_out_of() {
        let id = legacy_run_id("rvc");
        // layout ≥3 — the id is a path segment
        assert_eq!(run_id_in_rel(&format!("rvc/{RUNS_DIR}/{id}/G_1.pth"), "rvc"), id);
        assert_eq!(
            run_id_in_rel(&format!("rvc/{RUNS_DIR}/{id}/weights/m_e1_s1.pth"), "rvc"),
            id
        );
        // windows separators reach this from `path`-derived strings
        assert_eq!(run_id_in_rel(&format!("rvc\\{RUNS_DIR}\\{id}\\G_1.pth"), "rvc"), id);
        // layout ≤2 — the slot root IS the run, and "" is the POSITIVE answer for it
        assert_eq!(run_id_in_rel("rvc/G_1.pth", "rvc"), "");
        assert_eq!(run_id_in_rel("rvc/weights/m_e1_s1.pth", "rvc"), "");
        // a sibling family's rows are never attributed to this one
        assert_eq!(run_id_in_rel(&format!("sovits/{RUNS_DIR}/{id}/G_1.pth"), "rvc"), "");
        // the container listing itself names no run (no artifact after the id)
        assert_eq!(run_id_in_rel(&format!("rvc/{RUNS_DIR}/{id}"), "rvc"), "");
        // ⚠ the ledger rewrites the SAME prefix — if these two ever disagree, a re-pointed export
        // row stops matching the record it protects and the cleanup deletes it as unexported.
        let old = format!("rvc/weights/m.pth");
        let new = format!("rvc/{RUNS_DIR}/{id}/weights/m.pth");
        assert_eq!(run_id_in_rel(&old, "rvc"), "");
        assert_eq!(run_id_in_rel(&new, "rvc"), id);
    }

    #[test]
    fn a_run_id_can_never_be_read_as_something_else() {
        // Every rule below is paid for by a substring judgement somewhere else in the tree, and
        // every one of them fails silently, so they are asserted rather than documented.
        for probe in ["", "legacy-run/rvc", "字", "x|enc=vec768l12"] {
            let id = run_id_for(probe);
            assert_eq!(id.len(), 13, "{id}");
            assert!(id.starts_with('r'), "must not read as a number: {id}");
            assert!(id.chars().all(|c| c == 'r' || c.is_ascii_hexdigit()), "charset: {id}");
            assert!(run_id_is_usable(&id), "the minted id must satisfy its own predicate: {id}");
        }
        assert_ne!(run_id_for("a"), run_id_for("b"));
        assert_eq!(run_id_for("a"), run_id_for("a"), "Rust and the verifier must agree");

        // the reserved names, each with a live substring consumer on both sides of the IPC
        for bad in RESERVED_RUN_NAMES {
            assert!(!run_id_is_usable(bad), "{bad} is claimed by a path-substring judgement");
        }
        // a family name would make `has_family_slot` ambiguous one level up
        for f in tproject::FAMILIES {
            assert!(!run_id_is_usable(f), "{f}");
        }
        // non-ASCII: `str.isdigit()` accepts these and the vendored checkpoint sort would either
        // raise (superscripts) or silently inflate its key (other decimal scripts)
        for bad in ["r\u{00b2}", "r\u{0663}", "运行"] {
            assert!(!run_id_is_usable(bad), "{bad} must be refused: non-ASCII");
        }
        for bad in [".hidden", "a b", "a/b", "a.b", ""] {
            assert!(!run_id_is_usable(bad), "{bad:?}");
        }
        // ★ The frontend's epoch scanner `/_e(\d+)_s\d+\./` runs over the WHOLE relative path and
        // takes the FIRST match, so the question is whether an id can complete it. It cannot:
        // completing it needs a `.` right after the step digits, and the id may not contain one
        // (the character after an id inside a rel is always `/`). Both halves are asserted, so a
        // relaxed charset takes this down with it — which is the only way this rule can fail.
        assert!(!run_id_is_usable("r_e1_s100."), "the trailing `.` is what would complete it");
        for ok in ["r_e1_s100", "r_es100", "r_e_s1", "run_2"] {
            assert!(run_id_is_usable(ok), "{ok} cannot complete the pattern, so it stays usable");
        }
    }

    /// ⛔ The two resolvers answer DIFFERENT questions and the difference is the whole batch.
    ///
    /// Pinned together in one test on purpose: the failure mode is not "one of them is buggy", it
    /// is a caller reaching for the wrong one. The plural answer must cover every run (a slot-wide
    /// guard that saw one run would fail open), and the singular one must REFUSE rather than pick
    /// (a resume that guessed would train into another run's weights).
    #[test]
    fn the_plural_resolver_sees_every_run_and_the_singular_one_refuses_to_guess() {
        let data = tmp_data("resolvers");
        let slot = project_dir(&data, "p777_5555eeee").join("rvc");
        std::fs::create_dir_all(&slot).unwrap();

        // layout ≤ 2: the slot root is the one run, and BOTH resolvers say so
        assert_eq!(run_dirs(&slot), vec![RunDir::for_test(slot.clone())]);
        assert_eq!(resolve_run_dir(&slot, None).unwrap().path(), slot);

        // one run: still unambiguous
        let a = runs_root(&slot).join("ra11111111111");
        touch(&a.join("G_100.pth"));
        assert_eq!(run_dirs(&slot), vec![RunDir::for_test(a.clone())]);
        assert_eq!(resolve_run_dir(&slot, None).unwrap().path(), a);

        // two runs: the plural answer GROWS, the singular one becomes an error. Anything else —
        // "the newest", "the first" — is a guess, and the thing being guessed at is which
        // checkpoints a 续训 continues from.
        let b = runs_root(&slot).join("rb22222222222");
        touch(&b.join("G_200.pth"));
        assert_eq!(
            run_dirs(&slot),
            vec![RunDir::for_test(a.clone()), RunDir::for_test(b.clone())],
            "sorted by id, never wobbling"
        );
        let e = resolve_run_dir(&slot, None).unwrap_err().to_string();
        assert!(e.contains("RUN_AMBIGUOUS"), "{e}");
        // …and naming one still works, which is what the later batches hand it
        assert_eq!(resolve_run_dir(&slot, Some("rb22222222222")).unwrap().path(), b);

        // staging is never a run — a half-migrated tree must not be selectable by either resolver
        std::fs::create_dir_all(runs_root(&slot).join(format!("{STAGING_PREFIX}rc3"))).unwrap();
        assert_eq!(run_dirs(&slot), vec![RunDir::for_test(a), RunDir::for_test(b)]);

        let _ = std::fs::remove_dir_all(data);
    }

    /// The two tables must not both claim a name: a doubly-claimed entry is moved by whichever
    /// migration runs first and then silently "missing" for the other.
    #[test]
    fn the_run_table_and_the_pool_table_are_disjoint() {
        for p in tpool::POOL_ENTRIES {
            assert!(
                !is_run_entry(p.name),
                "{:?} is claimed by both the pool and the run table",
                p.name
            );
        }
        // and the container names are not claimed by either
        assert!(!is_run_entry(tpool::POOLS_DIR));
        assert!(!is_run_entry(tpool::SLOT_META));
        assert!(!is_run_entry(RUNS_DIR));
        // `runs` must be a name no other layer can mistake for its own
        for f in tproject::FAMILIES {
            assert_ne!(RUNS_DIR, f);
        }
        assert!(
            tproject::WORKSPACE_SUBDIRS.contains(&RUNS_DIR),
            "a new slot subdirectory has to be declared, or `has_family_slot`'s claim decays"
        );
    }

    #[test]
    fn migrate_moves_exactly_the_run_and_leaves_the_pool_alone() {
        let data = tmp_data("rvc");
        let id = "p111_efa35241";
        let slot = project_dir(&data, id).join("rvc");
        layout2_rvc_slot(&slot);
        write_meta(&data, &ProjectMeta { id: id.into(), name: "n".into(), ..Default::default() })
            .unwrap();

        let plan = plan_slot_runs(&slot);
        assert!(plan.unknown.is_empty(), "the real shape must classify fully: {:?}", plan.unknown);
        assert_eq!(plan.staying, Vec::<String>::new(), "layout 2 left nothing else at the root");
        assert_eq!(plan.moving.len(), 13, "{:?}", plan.moving);

        let out = migrate_slot_runs(&data, id, "rvc").unwrap();
        let run_id = legacy_run_id("rvc");
        assert_eq!(out, RunOutcome::Migrated(run_id.clone()));

        let run = runs_root(&slot).join(&run_id);
        assert!(run.join("G_2333333.pth").is_file());
        assert!(run.join("weights").join("m_e14_s147.pth").is_file());
        assert!(run.join("audition").join("m_e14_s147").join("audition.wav").is_file());
        assert!(run.join("total_fea.npy").is_file());
        assert!(run.join("events.out.tfevents.1784135788.NucBox_k11.9860.0").is_file());
        assert!(!slot.join("G_2333333.pth").exists(), "the run must not be left behind too");

        // the pool is layout 2's business and must not have moved a byte
        let pool = slot.join(tpool::POOLS_DIR).join("p55f7335bfa22");
        assert!(pool.join("0_gt_wavs").join("0.wav").is_file());
        assert!(pool.join("dataset.fingerprint").is_file());
        assert_eq!(tpool::sole_pool_fingerprint(&slot).as_deref(), Some("abc123"));

        assert_eq!(resolve_run_dir(&slot, None).unwrap().path(), run);
        assert_eq!(resolve_run_dir(&slot, Some(&run_id)).unwrap().path(), run);
        assert!(resolve_run_dir(&slot, Some("rdeadbeefdead")).is_err(), "a named miss is an error");

        // idempotent
        assert_eq!(migrate_slot_runs(&data, id, "rvc").unwrap(), RunOutcome::AlreadyDone);
        assert!(run.join("G_2333333.pth").is_file());

        let _ = std::fs::remove_dir_all(data);
    }

    /// ⛔ The ledger stores PROJECT-relative paths, so the file move alone silently strips every
    /// imported checkpoint of its cleanup protection.
    #[test]
    fn the_export_ledger_follows_the_files() {
        let data = tmp_data("ledger");
        let id = "p222_aaaabbbb";
        let slot = project_dir(&data, id).join("rvc");
        layout2_rvc_slot(&slot);
        // a sibling slot's row must not be touched by an rvc migration
        touch(&project_dir(&data, id).join("sovits").join("weights").join("s_e2_s200.pth"));
        write_meta(
            &data,
            &ProjectMeta {
                id: id.into(),
                name: "n".into(),
                export_ledger_since_ms: 1,
                exported: vec![
                    ExportedModel {
                        name: "m".into(),
                        model_type: "rvc".into(),
                        from_ckpt_rel: "rvc/weights/m_e14_s147.pth".into(),
                        at_ms: 1,
                    },
                    ExportedModel {
                        name: "s".into(),
                        model_type: "sovits".into(),
                        from_ckpt_rel: "sovits/weights/s_e2_s200.pth".into(),
                        at_ms: 1,
                    },
                ],
                ..Default::default()
            },
        )
        .unwrap();

        let run_id = legacy_run_id("rvc");
        migrate_slot_runs(&data, id, "rvc").unwrap();

        let meta = tproject::read_meta(&data, id).unwrap();
        assert_eq!(
            meta.exported[0].from_ckpt_rel,
            format!("rvc/runs/{run_id}/weights/m_e14_s147.pth"),
            "the row must name where the file now IS"
        );
        assert_eq!(
            meta.exported[1].from_ckpt_rel, "sovits/weights/s_e2_s200.pth",
            "another family's rows are none of this migration's business"
        );
        // and the rewritten row really addresses the file
        assert!(project_dir(&data, id).join(&meta.exported[0].from_ckpt_rel).is_file());

        // idempotent: a second pass (a restart between the rewrite and the commit) must not
        // prefix it twice
        assert_eq!(repoint_ledger(&data, id, "rvc", &run_id).unwrap(), 0);

        let _ = std::fs::remove_dir_all(data);
    }

    #[test]
    fn a_slot_that_never_trained_commits_without_moving_anything() {
        let data = tmp_data("empty");
        let id = "p333_ccccdddd";
        let slot = project_dir(&data, id).join("sovits");
        touch(&slot.join(tpool::POOLS_DIR).join("pdeadbeef0000").join("dataset.fingerprint"));
        std::fs::write(slot.join(tpool::SLOT_META), br#"{"layout":2}"#).unwrap();
        write_meta(&data, &ProjectMeta { id: id.into(), name: "n".into(), ..Default::default() })
            .unwrap();

        assert_eq!(migrate_slot_runs(&data, id, "sovits").unwrap(), RunOutcome::Committed);
        assert_eq!(tpool::read_slot_meta(&slot).unwrap().layout, SLOT_LAYOUT_RUNS);
        assert!(list_runs(&slot).is_empty());
        // …and with no runs the slot root still answers, because it IS the (empty) run
        assert_eq!(resolve_run_dir(&slot, None).unwrap().path(), slot);

        let _ = std::fs::remove_dir_all(data);
    }

    /// An unknown entry is left alone AND does not block the migration around it.
    #[test]
    fn unknown_entries_stay_put_and_do_not_block_the_migration() {
        let data = tmp_data("unknown");
        let id = "p444_eeeeffff";
        let slot = project_dir(&data, id).join("vocoder");
        touch(&slot.join("model_ckpt_steps_3644.ckpt"));
        touch(&slot.join("Thumbs.db"));
        touch(&slot.join("my notes.txt"));
        write_meta(&data, &ProjectMeta { id: id.into(), name: "n".into(), ..Default::default() })
            .unwrap();

        let plan = plan_slot_runs(&slot);
        assert_eq!(plan.unknown, vec!["Thumbs.db".to_string(), "my notes.txt".into()]);

        assert!(matches!(migrate_slot_runs(&data, id, "vocoder").unwrap(), RunOutcome::Migrated(_)));
        assert!(slot.join("Thumbs.db").is_file(), "a stray file stays exactly where it was");
        assert!(slot.join("my notes.txt").is_file());
        let run = runs_root(&slot).join(legacy_run_id("vocoder"));
        assert!(run.join("model_ckpt_steps_3644.ckpt").is_file());

        let _ = std::fs::remove_dir_all(data);
    }

    /// A kill between "moved into staging" and "committed" must leave the slot exactly as it was.
    #[test]
    fn a_torn_migration_rolls_back_to_the_pre_migration_shape() {
        let data = tmp_data("torn");
        let id = "p555_12341234";
        let slot = project_dir(&data, id).join("rvc");
        layout2_rvc_slot(&slot);
        write_meta(&data, &ProjectMeta { id: id.into(), name: "n".into(), ..Default::default() })
            .unwrap();
        let before = shape(&slot);

        // hand-build every torn state: kill after moving 1..n entries into staging
        let moving = plan_slot_runs(&slot).moving;
        for cut in 1..=moving.len() {
            let staging = slot.join(format!("{STAGING_PREFIX}{}", legacy_run_id("rvc")));
            std::fs::create_dir_all(&staging).unwrap();
            for name in moving.iter().take(cut) {
                std::fs::rename(slot.join(name), staging.join(name)).unwrap();
            }
            assert_ne!(shape(&slot), before, "cut {cut}: the fixture really is torn");

            reconcile_staging(&slot).unwrap();
            assert_eq!(shape(&slot), before, "cut {cut}: rollback must restore the exact shape");
            assert!(!staging.exists());
        }

        // …and the retry then migrates cleanly
        assert!(matches!(migrate_slot_runs(&data, id, "rvc").unwrap(), RunOutcome::Migrated(_)));
        let _ = std::fs::remove_dir_all(data);
    }

    /// ⛔ The staging reaper must run for a slot that is ALREADY migrated.
    ///
    /// This pins an ORDER, and the order is the only thing that recovers a killed migration.
    /// `.mig_run_*` is dot-prefixed, so nothing else on this tree can see it: `scan_project_ckpts`
    /// filters dot entries, `tproject::migrate_all` only strips `.mig_` at the TRAINING ROOT,
    /// `sweep_tombstones` only knows `.del_`, and the storage sweeper is depth-1 under the root.
    /// Meanwhile `dir_size` counts it — so a slot killed mid-migration would show its full size
    /// with zero snapshots and zero reclaimable bytes, forever. The one thing that gets those
    /// weights back is `reconcile_staging` running BEFORE the `layout >= 3` early return, on every
    /// boot, for every slot. Moving that call below the early return compiles, passes every other
    /// test, and silently strands the payload.
    #[test]
    fn a_torn_staging_directory_is_reclaimed_even_when_the_slot_is_already_migrated() {
        let data = tmp_data("reap");
        let id = "p888_7777cccc";
        let slot = project_dir(&data, id).join("rvc");
        write_meta(&data, &ProjectMeta { id: id.into(), name: "n".into(), ..Default::default() })
            .unwrap();
        // an already-migrated slot: one run folded, commit point written
        touch(&runs_root(&slot).join(legacy_run_id("rvc")).join("G_1400.pth"));
        std::fs::write(slot.join(tpool::SLOT_META), br#"{"layout":3}"#).unwrap();
        // …and a previous boot that was killed mid-migration, leaving a payload behind
        touch(&slot.join(format!("{STAGING_PREFIX}rdeadbeefdead")).join("D_1400.pth"));

        assert_eq!(migrate_slot_runs(&data, id, "rvc").unwrap(), RunOutcome::AlreadyDone);
        assert!(
            slot.join("D_1400.pth").is_file(),
            "the staged payload must come back to the slot root — nothing else on this tree can \
             see a dot directory, and `dir_size` keeps charging the user for it"
        );
        assert!(!slot.join(format!("{STAGING_PREFIX}rdeadbeefdead")).exists());
        assert!(runs_root(&slot).join(legacy_run_id("rvc")).join("G_1400.pth").is_file());

        let _ = std::fs::remove_dir_all(data);
    }

    /// ⛔ Where a START writes, across every shape `try_start` can hand it.
    ///
    /// The last arm is the one this batch exists to close, and it is a REGRESSION test rather than
    /// a design one: `try_start` resolves the run before its guards (they judge the pre-wipe state)
    /// and a full 「重训」 then deletes the whole slot and re-creates only the slot ROOT. Writing the
    /// manifest into the pre-wipe answer fails with a bare io NotFound *after* the slot is already
    /// empty — and the second press succeeds, so it reads as a glitch rather than as "that press
    /// deleted everything".
    #[test]
    fn a_start_always_gets_a_run_directory_that_exists() {
        let data = tmp_data("start");
        let id = "p999_5150abcd";
        let slot = project_dir(&data, id).join("rvc");
        std::fs::create_dir_all(&slot).unwrap();

        // layout ≤ 2 — the slot root IS the run, and it must NOT mint a container
        let r = run_dir_for_start(&slot, "rvc").unwrap();
        assert_eq!(r.path(), slot);
        assert!(!runs_root(&slot).exists(), "an unmigrated slot must stay unmigrated");

        // layout 3 with nothing folded (a slot that had nothing to move at migration time, or one
        // whose runs were all deleted): the marker says runs live in the container, so mint one.
        // Answering with the slot root here is readable TODAY and goes silently wrong the moment a
        // second run appears beside it — the products at the root leave `run_dirs`' view entirely.
        std::fs::write(slot.join(tpool::SLOT_META), br#"{"layout":3}"#).unwrap();
        let r = run_dir_for_start(&slot, "rvc").unwrap();
        assert_eq!(r.path(), runs_root(&slot).join(legacy_run_id("rvc")));
        assert!(r.is_dir(), "it must CREATE it — run.json is written before python is spawned");
        assert_eq!(run_dirs(&slot), vec![r.clone()], "and the plural reader must see it");

        // one run: the same answer, idempotently
        assert_eq!(run_dir_for_start(&slot, "rvc").unwrap(), r);

        // ★ the wipe: `remove_dir_all(slot)` + `create_dir_all(slot)`, exactly as try_start does.
        // `slot.json` goes with it, so the slot is honestly back at layout 0.
        std::fs::remove_dir_all(&slot).unwrap();
        std::fs::create_dir_all(&slot).unwrap();
        let r = run_dir_for_start(&slot, "rvc").unwrap();
        assert_eq!(r.path(), slot, "a wiped slot is a fresh slot");
        // …and the write that used to fail now lands
        std::fs::write(r.join("run_manifest.json"), b"{}").unwrap();

        // several runs: refuse rather than pick. The batch that mints a second one passes an id.
        touch(&runs_root(&slot).join("ra11111111111").join("G_1.pth"));
        touch(&runs_root(&slot).join("rb22222222222").join("G_2.pth"));
        let e = run_dir_for_start(&slot, "rvc").unwrap_err().to_string();
        assert!(e.contains("RUN_AMBIGUOUS"), "{e}");

        let _ = std::fs::remove_dir_all(data);
    }

    /// ⛔ THE ordering constraint of this batch, asserted on the two files that encode it.
    ///
    /// `trun` commits `layout: 3` into the same `slot.json` whose `layout >= 2` makes
    /// `tpool::migrate_slot` return early. Calling the run fold first therefore retires the POOL
    /// migration for that slot permanently, and silently: `plan_slot_runs` files a pool product
    /// left at the slot root under `staying`, not `unknown`, so there is not even a warn.
    ///
    /// A unit test cannot observe the boot sequence, so this reads the sequence itself. The
    /// mechanism is demonstrated below it, so the constant relationship is red-able too.
    #[test]
    fn the_boot_chain_folds_pools_before_runs() {
        const LIB_RS: &str = include_str!("../lib.rs");
        const SETTINGS_RS: &str = include_str!("../commands/settings.rs");
        for (name, src) in [("lib.rs", LIB_RS), ("commands/settings.rs", SETTINGS_RS)] {
            let pool = src.find("tpool::migrate_all(").unwrap_or_else(|| {
                panic!("{name} no longer folds the preprocessing pools at all")
            });
            let run = src
                .find("trun::migrate_all(")
                .unwrap_or_else(|| panic!("{name} no longer folds the runs at all"));
            assert!(
                pool < run,
                "{name} calls the RUN migration before the POOL one — that stamps layout 3 on a \
                 slot whose pool products are still at its root, and `tpool::migrate_slot` will \
                 answer AlreadyDone for it forever"
            );
        }
        // the mechanism itself: run-first really does retire the pool fold
        let data = tmp_data("order");
        let id = "paaa_0f0f0f0f";
        let slot = project_dir(&data, id).join("rvc");
        touch(&slot.join("0_gt_wavs").join("0.wav"));
        touch(&slot.join("G_2333333.pth"));
        std::fs::write(slot.join(tpool::FINGERPRINT), b"abc123").unwrap();
        write_meta(&data, &ProjectMeta { id: id.into(), name: "n".into(), ..Default::default() })
            .unwrap();

        assert!(matches!(migrate_slot_runs(&data, id, "rvc").unwrap(), RunOutcome::Migrated(_)));
        assert_eq!(tpool::migrate_slot(&slot, "rvc").unwrap(), tpool::SlotOutcome::AlreadyDone);
        assert!(
            slot.join("0_gt_wavs").join("0.wav").is_file() && !slot.join(tpool::POOLS_DIR).exists(),
            "…and the preprocessing products are stranded at the slot root, unfoldable"
        );

        let _ = std::fs::remove_dir_all(data);
    }

    /// The walk: every project, every family, and nothing that is not a project.
    #[test]
    fn migrate_all_folds_every_slot_of_every_project() {
        let data = tmp_data("all");
        for id in ["pone_11111111", "ptwo_22222222"] {
            write_meta(&data, &ProjectMeta { id: id.into(), name: "n".into(), ..Default::default() })
                .unwrap();
            for family in ["rvc", "sovits"] {
                let slot = project_dir(&data, id).join(family);
                layout2_rvc_slot(&slot);
            }
        }
        // a directory under the training root that is NOT a project (no project.json) — the pool
        // migration skips these by the same rule, and a tombstone must not be folded
        let bogus = tproject::training_root(&data).join(".del_pthree_33333333");
        touch(&bogus.join("rvc").join("G_1.pth"));

        migrate_all(&data);

        for id in ["pone_11111111", "ptwo_22222222"] {
            for family in ["rvc", "sovits"] {
                let slot = project_dir(&data, id).join(family);
                assert_eq!(tpool::read_slot_meta(&slot).unwrap().layout, SLOT_LAYOUT_RUNS);
                let run = runs_root(&slot).join(legacy_run_id(family));
                assert!(run.join("G_2333333.pth").is_file(), "{id}/{family}");
                assert!(!slot.join("G_2333333.pth").exists(), "{id}/{family}");
            }
        }
        assert!(bogus.join("rvc").join("G_1.pth").is_file(), "a tombstone is not a project");

        // idempotent: a second boot folds nothing and breaks nothing
        migrate_all(&data);
        let run = project_dir(&data, "pone_11111111").join("rvc").join(RUNS_DIR).join(legacy_run_id("rvc"));
        assert!(run.join("G_2333333.pth").is_file());

        let _ = std::fs::remove_dir_all(data);
    }

    /// ⛔ The two layouts advance the SAME `slot.json`, one step each. Bumping `tpool`'s constant
    /// instead of adding one would make an already-folded slot compute an empty pool plan, take
    /// the "nothing to move" branch and stamp the new layout — migrated on paper, untouched on
    /// disk.
    #[test]
    fn the_two_migrations_advance_the_same_commit_point_one_step_each() {
        assert_eq!(SLOT_LAYOUT_RUNS, tpool::SLOT_LAYOUT + 1);
        let data = tmp_data("layouts");
        let id = "p666_9999aaaa";
        let slot = project_dir(&data, id).join("rvc");
        write_meta(&data, &ProjectMeta { id: id.into(), name: "n".into(), ..Default::default() })
            .unwrap();
        // a layout-1 slot: pool products AND run products at the root
        touch(&slot.join("0_gt_wavs").join("0.wav"));
        touch(&slot.join("G_2333333.pth"));
        std::fs::write(slot.join("dataset.fingerprint"), b"abc123").unwrap();

        assert!(matches!(tpool::migrate_slot(&slot, "rvc").unwrap(), tpool::SlotOutcome::Migrated(_)));
        assert_eq!(tpool::read_slot_meta(&slot).unwrap().layout, tpool::SLOT_LAYOUT);
        assert!(slot.join("G_2333333.pth").is_file(), "layout 2 must not touch the run");

        assert!(matches!(migrate_slot_runs(&data, id, "rvc").unwrap(), RunOutcome::Migrated(_)));
        assert_eq!(tpool::read_slot_meta(&slot).unwrap().layout, SLOT_LAYOUT_RUNS);
        // …and layout 2 stays done rather than re-running on the new shape
        assert_eq!(tpool::migrate_slot(&slot, "rvc").unwrap(), tpool::SlotOutcome::AlreadyDone);
        assert!(runs_root(&slot).join(legacy_run_id("rvc")).join("G_2333333.pth").is_file());
        assert!(slot.join(tpool::POOLS_DIR).join(tpool::pool_id_for("abc123")).is_dir());

        let _ = std::fs::remove_dir_all(data);
    }
}
