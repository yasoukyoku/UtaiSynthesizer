//! S168: runtime verify/restore of the bundled trainer package.
//!
//! `training/utai_train` next to the exe is written once by the installer and then has NO
//! guardian: the S168 community report lost it to a chain the app itself started (the
//! migration stamped it into a phantom project; deleting that row deleted the trainer), and
//! an AV quarantine or a stray hand-delete produces the same end state — training and the
//! pack self-test fail (`ENVTEST_SCRIPT_MISSING` / `No module named 'utai_train'`) with no
//! repair short of a reinstall. Dictionaries got a per-boot re-sync for the same class of
//! damage (`sync_bundled_dictionaries`); this is that pattern for the trainer code, with the
//! embedded copy as the source (the dictionaries could sync from the install dir — this tree
//! IS the install dir, so the source has to live in the binary).
//!
//! ⚠ Heals ONLY `*.py` under `utai_train` (what build.rs embeds). It never deletes extras
//! (`__pycache__`, the migration's leftovers — `unfold_reserved_dirs` owns those) and never
//! touches `training/assets` (binary wavs, installer-owned; `bundled_integrity_report`
//! at least detects their absence).

use std::path::Path;

include!(concat!(env!("OUT_DIR"), "/utai_train_embed.rs"));

/// Verify every embedded trainer file against the disk copy under `<app_dir>/training/
/// utai_train`, restoring any that are missing or differ. Logs one line per boot on every
/// return path (S110: "ran and did nothing" must be distinguishable from "never ran").
///
/// No-op on dev checkouts (`training/.venv` present — a disk fact no install ever has):
/// there the tree is the SOURCE being edited, and healing would revert any edit made since
/// the last build.
pub fn sync_bundled_training_code(app_dir: &Path) {
    let root = app_dir.join("training");
    if root.join(".venv").exists() {
        tracing::info!("trainer-code heal: dev checkout (training/.venv) — disabled");
        return;
    }
    let base = root.join("utai_train");
    let mut healed: Vec<&str> = Vec::new();
    let mut failed: Vec<&str> = Vec::new();
    let mut deferred = 0usize;
    // Aggregate budget (reviewed S168): this runs synchronously in setup, before any window
    // exists, and rename_with_retry sleeps up to ~10 s per AV-locked file — 124 files with
    // no bound could hold the boot for minutes. A partial heal is retried next boot; a
    // hung boot is a bug report.
    const HEAL_BUDGET: std::time::Duration = std::time::Duration::from_secs(10);
    let started = std::time::Instant::now();
    for (rel, bytes) in UTAI_TRAIN_FILES {
        let target = base.join(rel);
        if let Ok(cur) = std::fs::read(&target) {
            if cur == *bytes {
                continue;
            }
        }
        if started.elapsed() > HEAL_BUDGET {
            deferred += 1;
            continue;
        }
        // A read-only target defeats the replace-rename through every retry — strip the
        // attribute first (same posture as util::remove_dir_all_robust).
        if let Ok(meta) = std::fs::metadata(&target) {
            let mut perm = meta.permissions();
            if perm.readonly() {
                perm.set_readonly(false);
                let _ = std::fs::set_permissions(&target, perm);
            }
        }
        let ok = target
            .parent()
            .map(|p| std::fs::create_dir_all(p).is_ok())
            .unwrap_or(false)
            && {
                let tmp = target.with_extension("py.healing");
                std::fs::write(&tmp, bytes).is_ok()
                    && crate::util::rename_with_retry(&tmp, &target, "TRAINER_CODE_HEAL")
                        .map(|_| true)
                        .unwrap_or_else(|_| {
                            let _ = std::fs::remove_file(&tmp);
                            false
                        })
            };
        if ok {
            healed.push(rel);
        } else {
            failed.push(rel);
        }
    }
    if deferred > 0 {
        tracing::warn!(
            "trainer-code heal: budget exhausted after {:.1}s — {deferred} file(s) deferred to the next boot",
            started.elapsed().as_secs_f32()
        );
    }
    if healed.is_empty() && failed.is_empty() {
        tracing::info!(
            "trainer-code heal: {} bundled file(s) verified — intact",
            UTAI_TRAIN_FILES.len()
        );
    } else {
        let show = |v: &Vec<&str>| {
            let mut s = v.iter().take(8).cloned().collect::<Vec<_>>().join(", ");
            if v.len() > 8 {
                s.push_str(", …");
            }
            s
        };
        if !healed.is_empty() {
            tracing::warn!(
                "trainer-code heal: restored {} missing/altered file(s) under {} ({}) — \
                 something removed or modified the bundled trainer since the last boot",
                healed.len(),
                base.display(),
                show(&healed)
            );
        }
        if !failed.is_empty() {
            tracing::error!(
                "trainer-code heal: could NOT restore {} file(s) under {} ({}) — training and \
                 the pack self-test will fail until an app reinstall",
                failed.len(),
                base.display(),
                show(&failed)
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The embed is the heal's whole authority — a build.rs filter bug would silently shrink
    /// coverage, so the shape is pinned here (S105: assertions about "the table" need a
    /// second assertion that pins the table itself).
    #[test]
    fn embedded_tree_holds_the_load_bearing_files_and_no_bytecode() {
        assert!(UTAI_TRAIN_FILES.len() >= 100, "only {} files embedded", UTAI_TRAIN_FILES.len());
        for probe in
            // hipenum.py: S169 — every AMD run's device pick imports it; a healed tree
            // without it turns the whole AMD lane into TRAINING_AMD_ENUM_FAILED.
            ["envtest.py", "runner.py", "__init__.py", "sovits/diffusion/__init__.py", "rvc/train.py", "hipenum.py"]
        {
            assert!(
                UTAI_TRAIN_FILES.iter().any(|(r, _)| *r == probe),
                "load-bearing file missing from the embed: {probe}"
            );
        }
        for (rel, bytes) in UTAI_TRAIN_FILES {
            assert!(rel.ends_with(".py"), "non-python file embedded: {rel}");
            assert!(!rel.contains("__pycache__"), "bytecode embedded: {rel}");
            // `__init__.py` markers are legitimately 0 bytes (disk-verified); anything else
            // reading empty means the embed read a torn file.
            assert!(
                !bytes.is_empty() || rel.ends_with("__init__.py"),
                "empty embed: {rel}"
            );
        }
    }

    /// Drives the REAL function on a fake install tree (S101: the test rebuilds the
    /// program's own call, not a convenient shortcut): a missing file is written back, a
    /// tampered file is restored byte-for-byte, and the dev-checkout guard leaves a tampered
    /// file alone.
    #[test]
    fn heal_restores_missing_and_tampered_files_but_never_on_a_dev_checkout() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let tmp = std::env::temp_dir().join(format!("utai_heal_{}_{nanos}", std::process::id()));
        let base = tmp.join("training").join("utai_train");
        std::fs::create_dir_all(&base).unwrap();
        std::fs::write(base.join("runner.py"), b"tampered").unwrap();

        sync_bundled_training_code(&tmp);

        let runner = UTAI_TRAIN_FILES.iter().find(|(r, _)| *r == "runner.py").unwrap();
        assert_eq!(
            std::fs::read(base.join("runner.py")).unwrap(),
            runner.1,
            "a tampered file must be restored to the embedded bytes"
        );
        assert!(base.join("envtest.py").is_file(), "a missing file must be written back");
        assert!(
            base.join("sovits").join("diffusion").join("__init__.py").is_file(),
            "nested subpackages must be restored too"
        );

        // Dev-checkout guard: with training/.venv present the SAME damage stays untouched.
        std::fs::create_dir_all(tmp.join("training").join(".venv")).unwrap();
        std::fs::write(base.join("runner.py"), b"tampered-again").unwrap();
        sync_bundled_training_code(&tmp);
        assert_eq!(
            std::fs::read(base.join("runner.py")).unwrap(),
            b"tampered-again",
            "a dev checkout must never be healed — it would revert live edits"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
