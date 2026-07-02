use std::path::PathBuf;
use utai_lib::inference::engine::{InputTensor, OnnxEngine};

fn rvc_onnx_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("converter/test_output/lengv2.3_integrated.onnx")
}

fn sovits_onnx_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("converter/test_output/akiko_cli.onnx")
}

#[test]
fn test_rvc_onnx_inference() {
    let path = rvc_onnx_path();
    if !path.exists() {
        eprintln!("Skipping RVC test: {} not found", path.display());
        return;
    }

    let engine = OnnxEngine::new();
    let session_id = engine.load_model(&path).expect("Failed to load RVC model");

    let t: i64 = 30;
    let phone_data = vec![0.0f32; (t * 768) as usize];
    let pitch_data = vec![128i64; t as usize];
    let pitchf_data = vec![220.0f32; t as usize];

    let inputs = vec![
        ("phone", InputTensor::F32 { data: phone_data, shape: vec![1, t, 768] }),
        ("phone_lengths", InputTensor::I64 { data: vec![t], shape: vec![1] }),
        ("pitch", InputTensor::I64 { data: pitch_data, shape: vec![1, t] }),
        ("pitchf", InputTensor::F32 { data: pitchf_data, shape: vec![1, t] }),
        ("sid", InputTensor::I64 { data: vec![0], shape: vec![1] }),
        ("noise_scale", InputTensor::F32 { data: vec![0.667], shape: vec![1] }),
    ];

    let outputs = engine.run(&session_id, inputs).expect("RVC inference failed");
    assert_eq!(outputs.len(), 1, "Expected 1 output");

    let audio = &outputs[0];
    let expected_len = (t * 480) as usize;
    assert_eq!(audio.len(), expected_len, "Expected {} samples, got {}", expected_len, audio.len());
    assert!(audio.iter().all(|x| !x.is_nan()), "Output contains NaN");

    eprintln!("RVC inference OK: {} samples, range [{:.3}, {:.3}]",
        audio.len(),
        audio.iter().cloned().fold(f32::INFINITY, f32::min),
        audio.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

    engine.unload_model(&session_id);
}

#[test]
fn test_sovits_onnx_inference() {
    let path = sovits_onnx_path();
    if !path.exists() {
        eprintln!("Skipping SoVITS test: {} not found", path.display());
        return;
    }

    let engine = OnnxEngine::new();
    let session_id = engine.load_model(&path).expect("Failed to load SoVITS model");

    let t: i64 = 30;
    let c_data = vec![0.0f32; (t * 256) as usize];
    let f0_data = vec![220.0f32; t as usize];
    let uv_data = vec![1i64; t as usize];

    let inputs = vec![
        ("c", InputTensor::F32 { data: c_data, shape: vec![1, t, 256] }),
        ("f0", InputTensor::F32 { data: f0_data, shape: vec![1, t] }),
        ("uv", InputTensor::I64 { data: uv_data, shape: vec![1, t] }),
        ("sid", InputTensor::I64 { data: vec![0], shape: vec![1] }),
    ];

    let outputs = engine.run(&session_id, inputs).expect("SoVITS inference failed");
    assert_eq!(outputs.len(), 1, "Expected 1 output");

    let audio = &outputs[0];
    let expected_len = (t * 512) as usize;
    assert_eq!(audio.len(), expected_len, "Expected {} samples, got {}", expected_len, audio.len());
    assert!(audio.iter().all(|x| !x.is_nan()), "Output contains NaN");

    eprintln!("SoVITS inference OK: {} samples, range [{:.3}, {:.3}]",
        audio.len(),
        audio.iter().cloned().fold(f32::INFINITY, f32::min),
        audio.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

    engine.unload_model(&session_id);
}
