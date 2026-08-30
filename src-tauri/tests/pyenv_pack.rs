//! Manual E2E for the S42 runtime-pack chain, WITHOUT the UI:
//! local archive → (manifest verify) → extract+commit → scan/resolve → envtest.
//!
//!   UTAI_PACK_FILE=D:\...\runtime-cpu-v1.tar.zst \
//!     cargo test --test pyenv_pack -- --ignored --nocapture
//!
//! Optional: UTAI_TEST_ROOT=<dir> (default %TEMP%\utai_pyenv_test — WIPED each run).
//!           ⛔ NEVER point it at the repo's data/ — step 0 is `remove_dir_all(root)`.
//! Optional: UTAI_PACK_DEVICE=cpu|cuda|xpu to override the envtest tier. By default the
//!           tier comes from the pack's variant via `pyenv::envtest_device_for_variant`,
//!           the same function the app uses (S115 — before that this harness passed no
//!           --device at all, and envtest defaults to `cpu`, so a GPU pack could pass
//!           here without a single GPU check having run). Use the override only to run a
//!           GPU pack's CPU half on a box that lacks the device, and say so when you do.
//! The envtest step runs against the repo's real training/ dir (utai_train.envtest).
//!
//! ⚠️ File deliberately NOT named `pyenv_install.rs`: Windows Installer Detection
//! demands elevation (os error 740, "requires elevation") for any manifest-less exe
//! whose NAME contains install/setup/update/patch — which is exactly what a cargo
//! test binary is. Keep those words out of test target names.

#[test]
#[ignore]
fn install_local_pack_and_envtest() {
    let root = std::env::var("UTAI_TEST_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("utai_pyenv_test"));
    let _ = std::fs::remove_dir_all(&root);
    run_chain(root);
}

/// S167: the same chain WITHOUT the wipe, for restoring a pack into a root that already
/// holds data (the dev repo's `data/` after the S96 blast, a user's data root, ...).
/// `UTAI_TEST_ROOT` is MANDATORY here — there is no temp default, because the only reason
/// to reach for this variant is that the root is precious. `extract_and_commit` itself
/// handles a pre-existing install of the same id (moved aside, restored on failure).
///
///   UTAI_TEST_ROOT=D:\MyDev\Utai_v2-dev\data UTAI_PACK_FILE=...\runtime-cpu-v1.tar.zst \
///     cargo test --test pyenv_pack install_local_pack_into_root_no_wipe -- --ignored --nocapture
#[test]
#[ignore]
fn install_local_pack_into_root_no_wipe() {
    let root = std::path::PathBuf::from(
        std::env::var("UTAI_TEST_ROOT").expect("set UTAI_TEST_ROOT explicitly (this variant never wipes)"),
    );
    assert!(root.is_dir(), "UTAI_TEST_ROOT must already exist: {}", root.display());
    run_chain(root);
}

fn run_chain(root: std::path::PathBuf) {
    utai_lib::suppress_windows_dll_error_dialogs();
    let file = std::env::var("UTAI_PACK_FILE").expect("set UTAI_PACK_FILE to the built .tar.zst");
    let picked = std::path::PathBuf::from(&file);

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

    // 2. extract DIRECTLY into <root>/<id>, then commit by writing pack.json LAST
    //    (same-dir tmp+rename). There is NO staging→final directory rename — see the
    //    WHY-NOT on extract_and_commit. (S115: this line used to say there was one.)
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
    // ★S115: the tier MUST be derived from the pack's variant, through the SAME function
    // the app uses. envtest's own `--device` defaults to `cpu`, and this harness used to
    // pass no --device at all — so re-verifying a GPU pack here produced a green badge
    // from a run that never touched the GPU (`cuda_driver`, `gpu_stft_vs_cpu`,
    // `gpu_amp_step` and `fallback_ops` all report "not applicable to this tier"). That is
    // precisely the false-green `envtest_device_for_variant`'s own doc warns about,
    // reached through a different door.
    // On a box WITHOUT that device the run is expected to fail loudly — measured S115: the
    // xpu pack reports `XPU_NO_DEVICE: torch.xpu.is_available()=False` and overall=fail.
    // That failure is not a nuisance, it is the control that proves the passing runs
    // actually reached the device.
    let device = std::env::var("UTAI_PACK_DEVICE")
        .unwrap_or_else(|_| utai_lib::pyenv::envtest_device_for_variant(&meta.variant).to_string());
    println!("envtest tier: variant {:?} -> --device {device}", meta.variant);
    let status = utai_lib::util::python_command(&py)
        .current_dir(&training)
        .args(["-m", "utai_train.envtest", "--device", &device, "--out"])
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
