//! Drive the §F2⒝ batch-2 RUN migration (slot layout 2 → 3) against a REAL project on disk.
//!
//! The twin of `tests/tpool_migrate.rs`, and it exists for the same reason: `trun`'s unit tests
//! build their own fixtures, so on their own they only prove the migrator correct on shapes I
//! invented — even the one modelled entry for entry on the surviving RVC slot is still a
//! transcription. This drives the production function against a real directory, which is what the
//! pool half was allowed to ship on (S122's five legs, run on the frozen pre-blast workspace).
//!
//! It also publishes the run-layout CONSTANTS as JSON. `TPOOL_JSON` carries not one run-side name,
//! and the run verifier needs more of them than the pool one did: the staging prefix (to build torn
//! states), the decision table (which has PREFIX entries — `G_2333333.pth` is not an exact name, so
//! a verifier that matched exactly would report every checkpoint as "not moved"), and the id
//! derivation (sha256; a second implementation in python is exactly the drift this avoids).
//!
//! ```text
//! # the layout contract, touching nothing:
//! cargo test --test trun_migrate -- --ignored --nocapture trun_layout_constants
//!
//! # fold every family slot of a project into runs/<id>/ (ON A COPY — this MOVES files and
//! # REWRITES project.json's export ledger):
//! set UTAI_TRUN_PROJECT=D:\...\arm_after\111_efa35241
//! cargo test --test trun_migrate -- --ignored --nocapture trun_migrate_project
//!
//! # roll back whatever a torn migration left behind (idempotent, mirror image):
//! set UTAI_TRUN_MODE=reconcile
//! cargo test --test trun_migrate -- --ignored --nocapture trun_migrate_project
//! ```
use std::path::Path;

use utai_lib::training::{tpool, tproject, trun};

/// `<data>/training/<project_id>` split back into the two arguments `migrate_slot_runs` takes.
///
/// ⚠ The run migrator needs the DATA ROOT and the project ID, not the slot path the pool migrator
/// takes, because the export ledger it re-points lives in `<project>/project.json` — one level
/// above the slot. Deriving both from the one path keeps the driver's interface identical to the
/// pool one instead of inventing a second environment variable that can disagree with the first.
fn split_project(project: &Path) -> (&Path, String) {
    let training = project.parent().expect("project directory has no parent");
    assert_eq!(
        training.file_name().and_then(|n| n.to_str()),
        Some("training"),
        "{} is not <data>/training/<project_id> — the ledger rewrite would address the wrong root",
        project.display()
    );
    let data = training.parent().expect("<data>/training has no parent");
    let id = project
        .file_name()
        .and_then(|n| n.to_str())
        .expect("project directory name is not utf-8")
        .to_string();
    (data, id)
}

/// Print the run-layout contract as JSON, so the verifier and the gates keep no second opinion.
#[test]
#[ignore]
fn trun_layout_constants() {
    // The decision table, with the two kinds kept APART. A verifier that flattened them would
    // silently stop recognising `G_*.pth` / `D_*.pth` / `model_ckpt_steps_*.ckpt` /
    // `events.out.tfevents.*` / `aug_gate_report*.json` as things that move.
    let mut exact: Vec<String> = Vec::new();
    let mut prefix: Vec<String> = Vec::new();
    for e in trun::RUN_ENTRIES {
        match e {
            trun::RunEntry::Exact(n) => exact.push(format!("\"{n}\"")),
            trun::RunEntry::Prefix(p) => prefix.push(format!("\"{p}\"")),
        }
    }
    // The id the migration gives each family's single legacy run — deterministic, so the verifier
    // can predict the destination path instead of globbing for whatever appeared.
    let legacy: Vec<String> = tproject::FAMILIES
        .iter()
        .map(|f| format!("\"{f}\":\"{}\"", trun::legacy_run_id(f)))
        .collect();
    // …and a couple of raw derivations, so a python side that ever needs one can assert it agrees
    // rather than reimplementing sha256 over a seed string.
    let probes = ["", "legacy-run/rvc", "字"];
    let ids: Vec<String> = probes
        .iter()
        .map(|p| format!("\"{p}\":\"{}\"", trun::run_id_for(p)))
        .collect();
    println!(
        "TRUN_JSON {{\"runs_dir\":\"{}\",\"slot_meta\":\"{}\",\"layout\":{},\
         \"pool_layout\":{},\"staging_prefix\":\"{}\",\"entries_exact\":[{}],\
         \"entries_prefix\":[{}],\"legacy_run_id\":{{{}}},\"run_id_for\":{{{}}}}}",
        trun::RUNS_DIR,
        tpool::SLOT_META,
        trun::SLOT_LAYOUT_RUNS,
        tpool::SLOT_LAYOUT,
        trun::staging_prefix(),
        exact.join(","),
        prefix.join(","),
        legacy.join(","),
        ids.join(",")
    );
}

/// Fold (or reconcile) every family slot of `UTAI_TRUN_PROJECT`.
///
/// ⛔ This MOVES FILES **and rewrites `project.json`**. The ledger rewrite is not incidental: the
/// export rows store PROJECT-relative checkpoint paths, so moving the files without them costs
/// every imported checkpoint its cleanup protection. A verifier comparing "no byte outside the
/// declared moves changed" has to account for that file specifically, or a correct behaviour reads
/// as a failure.
#[test]
#[ignore]
fn trun_migrate_project() {
    let Ok(project) = std::env::var("UTAI_TRUN_PROJECT") else {
        panic!("set UTAI_TRUN_PROJECT to the project directory to migrate (a COPY)");
    };
    let project = Path::new(&project);
    assert!(
        project.join("project.json").is_file(),
        "{} does not look like a training project (no project.json)",
        project.display()
    );
    let (data_dir, project_id) = split_project(project);
    let reconcile_only = std::env::var("UTAI_TRUN_MODE").as_deref() == Ok("reconcile");

    for family in tproject::FAMILIES {
        let slot = project.join(family);
        if !slot.is_dir() {
            continue;
        }
        if reconcile_only {
            trun::reconcile_staging(&slot).expect("reconcile");
            println!("TRUN_RESULT {family} reconciled");
            continue;
        }
        // pure function of the listing — printed BEFORE anything is touched, so a failed run still
        // says what it was about to do
        let plan = trun::plan_slot_runs(&slot);
        println!(
            "TRUN_PLAN {family} moving={:?} unknown={:?} staying={:?}",
            plan.moving, plan.unknown, plan.staying
        );
        let outcome = trun::migrate_slot_runs(data_dir, &project_id, family).expect("migrate");
        println!("TRUN_RESULT {family} {outcome:?}");
    }
}
