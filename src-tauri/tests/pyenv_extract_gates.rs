//! S68d behavior gates for runtime-pack extraction — the E:\ EXTRACT_FAILED batch.
//!
//! Own test binary on purpose: pyenv's RUNTIME_ROOT is a process-wide OnceLock, and
//! the manual E2E (pyenv_pack.rs) roots it elsewhere — sharing a binary would make
//! whichever test runs first decide the root for both. All tests here share ONE
//! scratch root (set once) and use distinct pack ids, so they can run in parallel.
//!
//! ⚠️ Name deliberately avoids install/setup/update/patch — Windows Installer
//! Detection demands elevation for manifest-less exes named like installers
//! (os error 740; see pyenv_pack.rs).

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::OnceLock;
use std::time::{Duration, SystemTime};

fn test_root() -> &'static PathBuf {
    static ROOT: OnceLock<PathBuf> = OnceLock::new();
    ROOT.get_or_init(|| {
        // Sweep predecessors first — the pid suffix (parallel-run safety) means a
        // fresh run never matches an old root, so litter would otherwise accumulate.
        if let Ok(rd) = std::fs::read_dir(std::env::temp_dir()) {
            for e in rd.flatten() {
                if e.file_name().to_string_lossy().starts_with("utai_extract_gates_") {
                    let _ = std::fs::remove_dir_all(e.path());
                }
            }
        }
        let root = std::env::temp_dir().join(format!("utai_extract_gates_{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        utai_lib::pyenv::init_runtime_root(&root);
        root
    })
}

/// Build a minimal single-part pack archive: pack.json first (extract_and_commit's
/// format contract), then python/python.exe (the PACK_NO_PYTHON sentinel) and one
/// payload file. Every entry carries the given tar mtime — the shipped packs carry 0,
/// which is exactly the S68d poison.
fn build_pack_zst(dir: &Path, id: &str, disk_bytes: u64, mtime: u64) -> PathBuf {
    let mut tar_bytes: Vec<u8> = Vec::new();
    {
        let mut b = tar::Builder::new(&mut tar_bytes);
        let meta = format!(
            r#"{{"schema":1,"id":"{id}","variant":"cpu","version":1,"label":"gate","python":"3.10","torch":"none","disk_bytes":{disk_bytes},"built":"2026-07-18"}}"#
        );
        let mut add = |name: &str, data: &[u8]| {
            let mut h = tar::Header::new_gnu();
            h.set_size(data.len() as u64);
            h.set_mode(0o666);
            h.set_mtime(mtime);
            h.set_cksum();
            b.append_data(&mut h, name, data).unwrap();
        };
        add("pack.json", meta.as_bytes());
        add("python/python.exe", b"dummy-python");
        add("python/DLLs/_gate.pdb", b"payload");
        b.finish().unwrap();
    }
    let zst = zstd::stream::encode_all(&tar_bytes[..], 3).unwrap();
    let p = dir.join(format!("{id}.tar.zst"));
    std::fs::write(&p, zst).unwrap();
    p
}

fn extract(parts: &[PathBuf]) -> Result<utai_lib::pyenv::PackMeta, String> {
    let cancel = AtomicBool::new(false);
    utai_lib::pyenv::extract_and_commit(parts, &cancel, |_| {}).map_err(|e| e.to_string())
}

/// Root-cause regression gate: an archive whose entries say mtime=0 (all four shipped
/// packs do) must extract fine AND the extracted files must NOT carry the 1970 stamp —
/// preserve_mtime(false) means "extraction time", which FAT-epoch volumes accept.
#[test]
fn mtime_zero_archive_extracts_with_fresh_mtimes() {
    let root = test_root();
    let part = build_pack_zst(root, "gate-mtime-v1", 64, 0);
    let meta = extract(&[part]).expect("extract must succeed");
    assert_eq!(meta.id, "gate-mtime-v1");
    let exe = root.join("runtimes").join("gate-mtime-v1").join("python").join("python.exe");
    assert!(exe.exists(), "committed pack incomplete");
    let mtime = std::fs::metadata(&exe).unwrap().modified().unwrap();
    let age = SystemTime::now().duration_since(mtime).unwrap_or(Duration::ZERO);
    assert!(
        age < Duration::from_secs(3600),
        "extracted file still carries the archive's 1970 mtime — preserve_mtime regressed"
    );
    assert!(root.join("runtimes").join("gate-mtime-v1").join("pack.json").exists());
}

/// Disk preflight gate: an absurd disk_bytes must be refused with INSTALL_DISK_FULL
/// (with the needed/free numbers in the detail) BEFORE anything lands on disk.
#[test]
fn oversized_pack_refused_by_preflight() {
    let root = test_root();
    let part = build_pack_zst(root, "gate-huge-v1", u64::MAX / 4, 0);
    let e = extract(&[part]).expect_err("preflight must refuse");
    assert!(e.contains("INSTALL_DISK_FULL"), "wrong error: {e}");
    assert!(e.contains("MB needed"), "numbers missing from detail: {e}");
    assert!(
        !root.join("runtimes").join("gate-huge-v1").exists(),
        "refusal must not leave a torn dir"
    );
}

/// Diagnosability gate (the actual S68d field pain): when a torn remnant can't be
/// cleared AND the extract-over then fails on it, the error must carry (a) the real
/// os error from the source() chain and (b) the pre-clean failure — not just
/// "failed to unpack `path`".
#[cfg(windows)]
#[test]
fn locked_remnant_surfaces_os_error_and_preclean_cause() {
    use std::os::windows::fs::OpenOptionsExt;
    let root = test_root();
    let part = build_pack_zst(root, "gate-lock-v1", 64, 0);

    // Torn remnant (no pack.json marker) with the payload file held open, share-none —
    // remove_dir_all fails (pre-clean), then tar's overwrite (remove_file + create) fails.
    let torn = root.join("runtimes").join("gate-lock-v1").join("python").join("DLLs");
    std::fs::create_dir_all(&torn).unwrap();
    let locked_path = torn.join("_gate.pdb");
    std::fs::write(&locked_path, b"remnant").unwrap();
    let _hold = std::fs::OpenOptions::new()
        .read(true)
        .share_mode(0)
        .open(&locked_path)
        .unwrap();

    let e = extract(&[part]).expect_err("locked remnant must fail the extract");
    assert!(e.contains("EXTRACT_FAILED"), "wrong code: {e}");
    assert!(e.contains("os error"), "io cause missing — source() chain not surfaced: {e}");
    assert!(e.contains("pre-clean incomplete"), "pre-clean cause missing: {e}");
    drop(_hold);
    let _ = std::fs::remove_dir_all(root.join("runtimes").join("gate-lock-v1"));
}
