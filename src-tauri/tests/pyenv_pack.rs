//! Manual E2E for the S42 runtime-pack chain, WITHOUT the UI:
//! local archive → (manifest verify) → extract+commit → scan/resolve → envtest.
//!
//!   UTAI_PACK_FILE=D:\...\runtime-cpu-v1.tar.zst \
//!     cargo test --test pyenv_pack -- --ignored --nocapture
//!
//! Optional: UTAI_TEST_ROOT=<dir> (default %TEMP%\utai_pyenv_test — WIPED each run).
//! The envtest step runs against the repo's real training/ dir (utai_train.envtest).
//!
//! ⚠️ File deliberately NOT named `pyenv_install.rs`: Windows Installer Detection
//! demands elevation (os error 740, "requires elevation") for any manifest-less exe
//! whose NAME contains install/setup/update/patch — which is exactly what a cargo
//! test binary is. Keep those words out of test target names.

#[test]
#[ignore]
fn install_local_pack_and_envtest() {
    utai_lib::suppress_windows_dll_error_dialogs();
    let file = std::env::var("UTAI_PACK_FILE").expect("set UTAI_PACK_FILE to the built .tar.zst");
    let picked = std::path::PathBuf::from(&file);

    let root = std::env::var("UTAI_TEST_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("utai_pyenv_test"));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    utai_lib::pyenv::init_runtime_root(&root);

    // 1. local-archive resolution (+ hash verification when the manifest travels along)
    let (parts, manifest) = utai_lib::pyenv::resolve_local_parts(&picked).unwrap();
    println!("parts: {:?}", parts.iter().map(|p| p.file_name()).collect::<Vec<_>>());
    if let Some(man) = &manifest {
        utai_lib::pyenv::verify_parts(man, picked.parent().unwrap()).unwrap();
        println!("manifest sha256 verified ({} parts)", man.parts.len());
    } else {
        println!("no manifest next to archive — verification skipped");
    }

    // 2. extract + staging→final commit
    let cancel = std::sync::atomic::AtomicBool::new(false);
    let t0 = std::time::Instant::now();
    let meta = utai_lib::pyenv::extract_and_commit(&parts, &cancel, |n| {
        if n % 5000 == 0 {
            println!("  ... {n} entries");
        }
    })
    .unwrap();
    println!("installed {} ({}, torch {}) in {:?}", meta.id, meta.variant, meta.torch, t0.elapsed());

    // 3. scan-based discovery + converter-role resolution (fake app dir = no dev venv,
    //    so the pack MUST win over the PATH fallback)
    let packs = utai_lib::pyenv::list_packs();
    assert!(packs.iter().any(|p| p.meta.id == meta.id), "installed pack not discovered by scan");
    let fake_app = root.join("fake_app");
    std::fs::create_dir_all(&fake_app).unwrap();
    let py = utai_lib::pyenv::converter_python(&fake_app);
    assert!(
        py.exists() && py.extension().map(|e| e == "exe").unwrap_or(false),
        "converter_python did not resolve to the pack: {}",
        py.display()
    );
    println!("converter python -> {}", py.display());

    // 4. the pack's own numeric self-test against the repo's real utai_train
    let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let training = repo.join("training");
    assert!(training.join("utai_train").join("envtest.py").exists());
    let report_path = root.join("runtimes").join(&meta.id).join("envtest.json");
    let status = utai_lib::util::python_command(&py)
        .current_dir(&training)
        .args(["-m", "utai_train.envtest", "--out"])
        .arg(&report_path)
        .status()
        .unwrap();
    let text = std::fs::read_to_string(&report_path).expect("envtest.json written");
    let report: serde_json::Value = serde_json::from_str(&text).unwrap();
    for item in report["items"].as_array().unwrap() {
        println!("  {:<22} {}  {}", item["name"].as_str().unwrap(), item["status"].as_str().unwrap(), item["detail"].as_str().unwrap_or(""));
    }
    assert_eq!(report["overall"], "pass", "envtest failed: {}", report["failed_names"]);
    assert!(status.success(), "envtest exit code nonzero");
    println!("envtest overall = pass");
}
