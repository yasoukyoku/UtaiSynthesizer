//! Voice-model ONNX smoke tests against the NEW converter contracts:
//!   - ContentVec extractors: waveform [1,N] @16k raw → features [1,T,dim], T=(N-400)/320+1
//!   - RMVPE e2e: log-mel [1,128,T] + threshold [1] → f0 [1,T] Hz@100fps, unvoiced == 0.0
//!   - RVC voice: phone/phone_lengths/pitch/pitchf/sid/rnd → audio [1,1,T·(sr/100)]
//!   - SoVITS voice: c/f0/uv/noise/sid(+vol) → audio [1,1,T·hop]
//!
//! Every test is gated on the model file existing (and, for the voice models, on the
//! sidecar json declaring the NEW input signature — the pre-rework exports still use the
//! old noise_scale/no-noise signatures and must be skipped, not failed). Numerical parity
//! gates live python-side in converter\verify\voice — these assert LENGTH relationships
//! and no-NaN only.

use std::path::PathBuf;
use utai_lib::inference::engine::{InputTensor, OnnxEngine};

fn app_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

/// Same contract as tests/separation_pipeline.rs: these tests predate the load-dynamic
/// ORT switch (S12) — without this init any ORT touch hangs FOREVER at 0 CPU behind an
/// invisible modal DLL dialog (S32 found this exact landmine via a full `cargo test`).
fn init_ort() {
    // S74: without a subscriber every engine INFO/WARN (chosen EP, the Auto-CUDA fallback
    // warning UTAI_SIMULATE_CUDA_FAIL exists to exercise) goes nowhere — a "verification" run
    // that can't show the path it claims to test proves nothing. `RUST_LOG=utai_lib=info` +
    // `-- --nocapture` to see them.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("utai_lib=warn")),
        )
        .try_init();
    utai_lib::suppress_windows_dll_error_dialogs();
    // PATH must carry runtime/cuda BEFORE ORT builds a CUDA session: the cudnn 9
    // shim resolves its sub-DLLs via PATH at graph-build time — without this the
    // first Conv dies with CUDNN_BACKEND_API_FAILED (bare-harness-only failure
    // that reads like an environment drift; the app does this in run()).
    utai_lib::setup_cuda_dll_paths(&app_root());
    utai_lib::init_ort_runtime(&app_root());
}

fn test_output(name: &str) -> PathBuf {
    app_root().join("converter/test_output").join(name)
}

/// Sidecar-json gate: run only when the sidecar declares the new-signature input list
/// including `required_input` (converter rework writes "inputs": [...] into the sidecar).
fn sidecar_has_input(onnx_path: &PathBuf, required_input: &str) -> Option<serde_json::Value> {
    let json_path = onnx_path.with_extension("json");
    let sidecar: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(json_path).ok()?).ok()?;
    let has = sidecar
        .get("inputs")?
        .as_array()?
        .iter()
        .any(|v| v.as_str() == Some(required_input));
    if has {
        Some(sidecar)
    } else {
        None
    }
}

/// 220 Hz sine at 16 kHz — deterministic voiced-ish test signal.
fn sine_16k(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 16000.0).sin() * 0.5)
        .collect()
}

#[test]
fn test_contentvec_onnx_length_contract() {
    init_ort();
    let engine = OnnxEngine::new();
    for (file, dim) in [("contentvec_768l12.onnx", 768usize), ("contentvec_256l9.onnx", 256)] {
        let path = test_output(file);
        if !path.exists() {
            eprintln!("Skipping ContentVec test: {} not found", path.display());
            continue;
        }
        let session = engine.load_model(&path).expect("load contentvec");
        // two lengths → T = (N-400)/320 + 1 (odd + even frame counts)
        for n in [16000usize, 8000] {
            let t = (n - 400) / 320 + 1;
            let feats = utai_lib::inference::features::contentvec_extract(
                &engine,
                &session,
                &sine_16k(n),
                dim,
            )
            .expect("contentvec extract");
            assert_eq!(feats.nrows(), t, "{}: T for N={}", file, n);
            assert_eq!(feats.ncols(), dim, "{}: dim", file);
            assert!(feats.iter().all(|v| v.is_finite()), "{}: NaN in features", file);
        }
        engine.unload_model(&session);
        eprintln!("ContentVec OK: {}", file);
    }
}

#[test]
fn test_rmvpe_onnx_f0_contract() {
    let path = test_output("rmvpe_e2e.onnx");
    let mel_path = test_output("rmvpe_mel_filters.npy");
    if !path.exists() || !mel_path.exists() {
        eprintln!("Skipping RMVPE test: {} / {} not found", path.display(), mel_path.display());
        return;
    }
    init_ort();
    let engine = OnnxEngine::new();
    let session = engine.load_model(&path).expect("load rmvpe");
    let mel: ndarray::Array2<f32> = ndarray_npy::read_npy(&mel_path).expect("mel filters npy");

    // T = 1 + N/160, N deliberately NOT a multiple of 160 or 32-frame-aligned
    for n in [16000usize, 24135, 700] {
        let f0 = utai_lib::inference::f0::rmvpe_detect(&engine, &session, &mel, &sine_16k(n), 0.03)
            .expect("rmvpe detect");
        let expect_t = 1 + n.max(513) / 160;
        assert_eq!(f0.len(), expect_t, "T for N={}", n);
        // contract: unvoiced == exact 0.0; anything voiced is a plausible Hz value
        for (i, &v) in f0.iter().enumerate() {
            assert!(v.is_finite(), "NaN f0 at {}", i);
            assert!(
                v == 0.0 || (10.0..2500.0).contains(&v),
                "f0[{}] = {} outside contract (0.0 or ~Hz)",
                i,
                v
            );
        }
        let voiced: Vec<f32> = f0.iter().copied().filter(|&v| v > 0.0).collect();
        eprintln!(
            "RMVPE N={}: T={}, voiced {}/{}, voiced range [{:.1}, {:.1}] Hz",
            n,
            f0.len(),
            voiced.len(),
            f0.len(),
            voiced.iter().cloned().fold(f32::INFINITY, f32::min),
            voiced.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        );
        // 220 Hz sine, longest case: the mel chain feeding a correct model must lock onto
        // the fundamental for the bulk of frames (loose band — synthetic tone, not voice)
        if n >= 16000 {
            let near: usize = voiced.iter().filter(|v| (200.0..240.0).contains(*v)).count();
            assert!(
                near * 2 > f0.len(),
                "expected most frames near 220 Hz, got {}/{}",
                near,
                f0.len()
            );
        }
    }
    engine.unload_model(&session);
    eprintln!("RMVPE OK");
}

#[test]
fn test_rvc_onnx_inference() {
    // the final converter export (rnd signature); lengv2.3_integrated.onnx is the OLD one
    let path = test_output("lengv2.3.onnx");
    if !path.exists() {
        eprintln!("Skipping RVC test: {} not found", path.display());
        return;
    }
    let Some(sidecar) = sidecar_has_input(&path, "rnd") else {
        eprintln!("Skipping RVC test: sidecar has no 'rnd' input (old-format export)");
        return;
    };
    let sr = sidecar.get("sample_rate").and_then(|v| v.as_u64()).unwrap_or(48000) as usize;
    let dim = sidecar.get("features_dim").and_then(|v| v.as_u64()).unwrap_or(768) as usize;

    init_ort();
    let engine = OnnxEngine::new();
    let session_id = engine.load_model(&path).expect("Failed to load RVC model");

    let t: i64 = 30;
    let inputs = vec![
        ("phone", InputTensor::F32 { data: vec![0.1f32; (t as usize) * dim], shape: vec![1, t, dim as i64] }),
        ("phone_lengths", InputTensor::I64 { data: vec![t], shape: vec![1] }),
        ("pitch", InputTensor::I64 { data: vec![128i64; t as usize], shape: vec![1, t] }),
        ("pitchf", InputTensor::F32 { data: vec![220.0f32; t as usize], shape: vec![1, t] }),
        ("sid", InputTensor::I64 { data: vec![0], shape: vec![1] }),
        // rnd = N(0,1)·noise_scale, pre-scaled by the caller (zeros = deterministic center)
        ("rnd", InputTensor::F32 { data: vec![0.0f32; 192 * t as usize], shape: vec![1, 192, t] }),
    ];

    let outputs = engine.run(&session_id, inputs).expect("RVC inference failed");
    assert_eq!(outputs.len(), 1, "Expected 1 output");

    let audio = &outputs[0];
    // audio [1,1,L]: L = T · (sr/100) samples (100 fps frame grid)
    let expected_len = t as usize * (sr / 100);
    assert_eq!(audio.len(), expected_len, "Expected {} samples, got {}", expected_len, audio.len());
    assert!(audio.iter().all(|x| !x.is_nan()), "Output contains NaN");

    eprintln!(
        "RVC inference OK: {} samples, range [{:.3}, {:.3}]",
        audio.len(),
        audio.iter().cloned().fold(f32::INFINITY, f32::min),
        audio.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
    engine.unload_model(&session_id);
}

#[test]
fn test_sovits_onnx_inference() {
    init_ort();
    let engine = OnnxEngine::new();
    // 4.0 (no vol) + 4.1 (vol_embedding — exercises the conditional vol input)
    for file in ["akiko_320000.onnx", "Sovits4.1东雪莲主模型.onnx"] {
        let path = test_output(file);
        if !path.exists() {
            eprintln!("Skipping SoVITS test: {} not found", path.display());
            continue;
        }
        let Some(sidecar) = sidecar_has_input(&path, "noise") else {
            eprintln!("Skipping SoVITS test: {} sidecar has no 'noise' input (old-format export)", file);
            continue;
        };
        let hop = sidecar.get("hop_size").and_then(|v| v.as_u64()).unwrap_or(512) as usize;
        let dim = sidecar.get("features_dim").and_then(|v| v.as_u64()).unwrap_or(256) as usize;
        // final contract: the sidecar "inputs" array is the authority on whether vol exists
        let has_vol = sidecar
            .get("inputs")
            .and_then(|v| v.as_array())
            .map(|l| l.iter().any(|v| v.as_str() == Some("vol")))
            .unwrap_or(false);

        // The 东雪莲 file doubles as the non-ACP-path regression check: ORT's CreateSession
        // converts paths through the process ACP on Windows ("No mapping for the Unicode
        // character" on cp932 + 东) — engine.rs routes non-ASCII paths through
        // commit_from_memory, so this MUST load directly, no ASCII-copy workaround.
        let session_id = engine.load_model(&path).expect("Failed to load SoVITS model");

        let t: i64 = 30; // ≥ min_frames (6)
        let mut inputs = vec![
            ("c", InputTensor::F32 { data: vec![0.1f32; (t as usize) * dim], shape: vec![1, t, dim as i64] }),
            // NEW contract: f0 is RAW Hz (coarse quantization happens in-graph)
            ("f0", InputTensor::F32 { data: vec![220.0f32; t as usize], shape: vec![1, t] }),
            // NEW contract: uv is f32 0/1 (1 = voiced)
            ("uv", InputTensor::F32 { data: vec![1.0f32; t as usize], shape: vec![1, t] }),
            // noise = N(0,1)·noise_scale, pre-scaled by the caller
            ("noise", InputTensor::F32 { data: vec![0.0f32; 192 * t as usize], shape: vec![1, 192, t] }),
            ("sid", InputTensor::I64 { data: vec![0], shape: vec![1] }),
        ];
        if has_vol {
            inputs.push(("vol", InputTensor::F32 { data: vec![0.1f32; t as usize], shape: vec![1, t] }));
        }

        let outputs = engine.run(&session_id, inputs).expect("SoVITS inference failed");
        assert_eq!(outputs.len(), 1, "Expected 1 output");

        let audio = &outputs[0];
        let expected_len = t as usize * hop;
        assert_eq!(audio.len(), expected_len, "{}: expected {} samples, got {}", file, expected_len, audio.len());
        assert!(audio.iter().all(|x| !x.is_nan()), "{}: output contains NaN", file);

        eprintln!(
            "SoVITS inference OK ({}, vol={}): {} samples, range [{:.3}, {:.3}]",
            file,
            has_vol,
            audio.len(),
            audio.iter().cloned().fold(f32::INFINITY, f32::min),
            audio.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        );
        engine.unload_model(&session_id);
    }
}

/// The three REAL sidecars shipped by the converter workflow must round-trip through the
/// tolerant serde ModelConfig with every field the pipelines consume intact.
#[test]
fn test_real_sidecars_parse_into_model_config() {
    use utai_lib::models::ModelConfig;

    // RVC v2 (lengv2.3.json)
    let p = test_output("lengv2.3.json");
    if p.exists() {
        let cfg: ModelConfig =
            serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).expect("rvc sidecar parse");
        assert_eq!(cfg.r#type, "rvc");
        assert_eq!(cfg.version, "v2");
        assert_eq!(cfg.features_dim, 768);
        assert_eq!(cfg.sample_rate, 48000);
        // noise.rnd_input[1] == 192 (noise_channels source)
        let rnd = cfg.noise.as_ref().unwrap().get("rnd_input").unwrap().as_array().unwrap();
        assert_eq!(rnd[1].as_u64(), Some(192));
        // min_frames flows through the tolerant `extra` flatten
        assert_eq!(cfg.extra.get("min_frames").and_then(|v| v.as_u64()), Some(12));
        let inputs = cfg.inputs.as_ref().unwrap().as_array().unwrap();
        assert!(inputs.iter().any(|v| v.as_str() == Some("rnd")));
    } else {
        eprintln!("Skipping sidecar check: {} not found", p.display());
    }

    // SoVITS 4.0 (akiko_320000.json) — no vol
    let p = test_output("akiko_320000.json");
    if p.exists() {
        let cfg: ModelConfig =
            serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).expect("sovits 4.0 sidecar parse");
        assert_eq!(cfg.speech_encoder.as_deref(), Some("vec256l9"));
        assert_eq!(cfg.hop_size, Some(512));
        assert_eq!(cfg.vol_embedding, Some(false));
        assert_eq!(cfg.unit_interpolate_mode.as_deref(), Some("left"));
        assert_eq!(cfg.speakers.get("akiko4.0"), Some(&0));
        let noise = cfg.noise.as_ref().unwrap().get("noise_input").unwrap().as_array().unwrap();
        assert_eq!(noise[1].as_u64(), Some(192));
        assert_eq!(cfg.extra.get("min_frames").and_then(|v| v.as_u64()), Some(6));
        let inputs = cfg.inputs.as_ref().unwrap().as_array().unwrap();
        assert!(!inputs.iter().any(|v| v.as_str() == Some("vol")));
    } else {
        eprintln!("Skipping sidecar check: {} not found", p.display());
    }

    // SoVITS 4.1 (Sovits4.1东雪莲主模型.json) — vol_embedding, Chinese stem + speaker map
    let p = test_output("Sovits4.1东雪莲主模型.json");
    if p.exists() {
        let cfg: ModelConfig =
            serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).expect("sovits 4.1 sidecar parse");
        assert_eq!(cfg.speech_encoder.as_deref(), Some("vec768l12"));
        assert_eq!(cfg.vol_embedding, Some(true));
        assert_eq!(cfg.unit_interpolate_mode.as_deref(), Some("nearest"));
        assert_eq!(cfg.speakers.get("AzumaVocal"), Some(&0));
        let inputs = cfg.inputs.as_ref().unwrap().as_array().unwrap();
        assert!(inputs.iter().any(|v| v.as_str() == Some("vol")));
    } else {
        eprintln!("Skipping sidecar check: {} not found", p.display());
    }
}
