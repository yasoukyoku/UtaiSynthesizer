//! S40 vocoder resource import-chain integration test (审查 HIGH 修复的机器验证).
//!
//! Drives the REAL ModelRegistry::import_file against a TEMP models dir with the
//! real converter venv (spawns export_nsf_hifigan.py — torch runs ~40s per
//! conversion), so it is #[ignore]d like the other env-heavy harnesses:
//!
//!   cargo test --test vocoder_import -- --ignored --nocapture
//!
//! Dev-machine paths (converter gate precedent — these tests document the S40
//! smoke artifacts they consume):
//!   good ckpt  = TESTING/smoke_vocoder/ws/weights/vocoder_best.ckpt (+config.json)
//!   bad  ckpt  = the mini_nsf PC vocoder (rejected by the exporter)

use std::path::{Path, PathBuf};

use utai_lib::models::{ModelRegistry, ModelType};

fn app_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

const GOOD_CKPT: &str = r"D:\MyDev\TESTING\smoke_vocoder\ws\weights\vocoder_best.ckpt";
const MINI_NSF_CKPT: &str =
    r"D:\MyDev\DiffSinger\checkpoints\pc_nsf_hifigan_44.1k_hop512_128bin_2025.02\model.ckpt";

fn temp_models_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("utai_vocimport_test_{}", tag));
    if d.exists() {
        std::fs::remove_dir_all(&d).ok();
    }
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn import_vocoder(
    reg: &ModelRegistry,
    name: &str,
    src: &Path,
) -> utai_lib::Result<utai_lib::models::ImportOutcome> {
    reg.import_file(
        name,
        src,
        ModelType::NsfHifigan,
        &app_root(),
        None,
        None,
        None,
        None,
        None,
    )
}

fn trio_paths(models_dir: &Path, stem: &str) -> [PathBuf; 3] {
    let sub = models_dir.join("nsf_hifigan");
    [
        sub.join(format!("{}.onnx", stem)),
        sub.join(format!("{}.json", stem)),
        sub.join(format!("{}_mel.npy", stem)),
    ]
}

#[test]
#[ignore]
fn vocoder_import_chain_end_to_end() {
    let models_dir = temp_models_dir("e2e");
    let reg = ModelRegistry::new(models_dir.clone());

    // ---- (1) good torch ckpt → temp-convert → three-piece set + entry ----
    let outcome = import_vocoder(&reg, "冒烟声码器", Path::new(GOOD_CKPT))
        .expect("good ckpt import must succeed");
    let stem = outcome
        .entry
        .path
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .to_string();
    for p in trio_paths(&models_dir, &stem) {
        assert!(p.is_file(), "trio piece missing: {}", p.display());
    }
    // sidecar mel_filters must follow OUR stem (A6)
    let sidecar: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(outcome.entry.path.with_extension("json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        sidecar["mel_filters"].as_str().unwrap(),
        format!("{}_mel.npy", stem)
    );
    assert_eq!(sidecar["num_mels"].as_i64(), Some(128));
    // no temp/ghost residue
    assert!(
        !models_dir.join("nsf_hifigan").join(".vocimport_tmp").exists(),
        "temp conversion dir must be cleaned"
    );

    // ---- (2) FAILING same-name re-import (mini_nsf rejected by the exporter):
    // the old resource must SURVIVE untouched and no residue may appear (HIGH) ----
    if Path::new(MINI_NSF_CKPT).is_file() {
        let before: Vec<Vec<u8>> = trio_paths(&models_dir, &stem)
            .iter()
            .map(|p| std::fs::read(p).unwrap())
            .collect();
        let err = import_vocoder(&reg, "冒烟声码器", Path::new(MINI_NSF_CKPT));
        assert!(err.is_err(), "mini_nsf import must be rejected");
        for (p, prev) in trio_paths(&models_dir, &stem).iter().zip(&before) {
            let now = std::fs::read(p).unwrap_or_default();
            assert_eq!(&now, prev, "old resource file disturbed: {}", p.display());
        }
        assert!(
            !models_dir.join("nsf_hifigan").join(".vocimport_tmp").exists(),
            "failed conversion must clean its temp dir"
        );
        // registry entry still resolvable after a fresh rescan (no ghosts, no dup)
        let reg2 = ModelRegistry::new(models_dir.clone());
        let entries = reg2.list_by_type(&ModelType::NsfHifigan);
        assert_eq!(entries.len(), 1, "exactly one vocoder entry, no ghosts");
        assert_eq!(entries[0].name, "冒烟声码器");
    } else {
        eprintln!("[skip] mini_nsf ckpt not present — rejection leg skipped");
    }

    // ---- (3) direct .onnx three-piece import (re-import under a new name) ----
    let outcome3 = import_vocoder(&reg, "直拷声码器", &outcome.entry.path)
        .expect("direct onnx three-piece import must succeed");
    let stem3 = outcome3
        .entry
        .path
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .to_string();
    for p in trio_paths(&models_dir, &stem3) {
        assert!(p.is_file(), "direct-import trio piece missing: {}", p.display());
    }

    std::fs::remove_dir_all(&models_dir).ok();
    eprintln!("[vocoder_import] all legs passed");
}
