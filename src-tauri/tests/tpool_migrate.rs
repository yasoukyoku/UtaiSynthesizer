//! Drive the §F2⒝ pool migration against a REAL training project on disk.
//!
//! ## Why an `#[ignore]`d integration test and not a unit test
//!
//! `tpool`'s unit tests build their own fixtures, so they only ever prove the migrator correct on
//! shapes I invented. The one workspace on this machine whose shape I did NOT invent is the RVC
//! project the installed application produced before the S96 blast (frozen, sha256-verified, in
//! `TESTING/s120_f2_fixtures/pre_migration/`). Proving the migration on THAT requires driving the
//! production function against a real directory — the same reason `tests/pyenv_pack.rs` exists,
//! and the same shape (ignored by default, a path from the environment).
//!
//! It also publishes the layout CONSTANTS as JSON, so the python-side verifier and
//! `gate_pool_table.py` read them from Rust instead of keeping a second opinion that can drift.
//!
//! ```text
//! # what the migrator would do / what the layout names are, touching nothing:
//! cargo test --test tpool_migrate -- --ignored --nocapture tpool_layout_constants
//!
//! # migrate every family slot of a project (ON A COPY — this MOVES files):
//! set UTAI_TPOOL_PROJECT=D:\...\arm_after\111_efa35241
//! cargo test --test tpool_migrate -- --ignored --nocapture tpool_migrate_project
//!
//! # roll back whatever a torn migration left behind (idempotent, mirror image):
//! set UTAI_TPOOL_MODE=reconcile
//! cargo test --test tpool_migrate -- --ignored --nocapture tpool_migrate_project
//! ```
use std::path::Path;

use utai_lib::training::{tpool, tproject};

/// Print the layout contract as JSON. The verifier and the cross-language gate consume this so
/// there is exactly ONE definition of "where does a pool live and what goes in it".
#[test]
#[ignore]
fn tpool_layout_constants() {
    let mut families = String::new();
    for (i, f) in tproject::FAMILIES.iter().enumerate() {
        if i > 0 {
            families.push(',');
        }
        let names: Vec<String> = tpool::pool_entries_for(f)
            .into_iter()
            .map(|n| format!("\"{n}\""))
            .collect();
        families.push_str(&format!("\"{f}\":[{}]", names.join(",")));
    }
    // A couple of derivations too, so the python side can assert it computes the same ids.
    let probes = ["abc123", "x|enc=vec768l12|loudnorm=1", "字|vocoder-v3"];
    let ids: Vec<String> = probes
        .iter()
        .map(|p| format!("\"{p}\":\"{}\"", tpool::pool_id_for(p)))
        .collect();
    println!(
        "TPOOL_JSON {{\"pools_dir\":\"{}\",\"slot_meta\":\"{}\",\"fingerprint\":\"{}\",\
         \"layout\":{},\"staging_prefix\":\"{}\",\"families\":{{{}}},\"pool_id_for\":{{{}}}}}",
        tpool::POOLS_DIR,
        tpool::SLOT_META,
        tpool::FINGERPRINT,
        tpool::SLOT_LAYOUT,
        tpool::staging_prefix(),
        families,
        ids.join(",")
    );
}

/// Migrate (or reconcile) every family slot of `UTAI_TPOOL_PROJECT`.
///
/// ⛔ This MOVES FILES. It refuses to run against anything that is not clearly a copy: the caller
/// must point it at a directory holding `project.json`, and the surviving fixture and the live
/// data root are both read-only by policy — the verifier always works on a robocopy of them.
#[test]
#[ignore]
fn tpool_migrate_project() {
    let Ok(project) = std::env::var("UTAI_TPOOL_PROJECT") else {
        panic!("set UTAI_TPOOL_PROJECT to the project directory to migrate (a COPY)");
    };
    let project = Path::new(&project);
    assert!(
        project.join("project.json").is_file(),
        "{} does not look like a training project (no project.json)",
        project.display()
    );
    let reconcile_only = std::env::var("UTAI_TPOOL_MODE").as_deref() == Ok("reconcile");

    for family in tproject::FAMILIES {
        let slot = project.join(family);
        if !slot.is_dir() {
            continue;
        }
        if reconcile_only {
            tpool::reconcile_staging(&slot).expect("reconcile");
            println!("TPOOL_RESULT {family} reconciled");
            continue;
        }
        // the plan is a pure function of the listing — print it BEFORE touching anything, so a
        // failed run still says what it was about to do
        let plan = tpool::plan_slot(&slot, family);
        println!(
            "TPOOL_PLAN {family} moving={:?} unknown={:?} staying={}",
            plan.moving,
            plan.unknown,
            plan.staying.len()
        );
        let outcome = tpool::migrate_slot(&slot, family).expect("migrate_slot");
        println!("TPOOL_RESULT {family} {outcome:?}");
    }
}
