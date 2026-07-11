//! S41 audition-render integration legs (design B5 试听侧; red-team V15/V16/V17).
//!
//! Drives the LIB-LEVEL core of the audition commands (candidate conversion →
//! sidecar facts → run_pipeline / vocoder self-loop) against the smoke-run
//! fixtures listed below. The command shells (flight flag, cleanup ordering,
//! events) are exercised by the live dev test.
//!
//! Assertion set (V17 — "nonzero sound" is NOT correctness, S31 lesson):
//!   header: expected sample rate, duration ≈ the 9.6 s clip (±0.5%), all
//!   finite, peak ≤ 1, RMS > -50 dBFS
//!   pitch identity: rmvpe(output) vs rmvpe(input) on both-voiced frames,
//!   median |cents| < 50 (catches wrong-model / garbled / wrong-clock output)
//!   vocoder A/B: the candidate's render must NOT be bit-identical to the
//!   default vocoder's (an identical pair means the override didn't take)
//!
//! Fixtures (skip-if-absent, vocoder_import.rs precedent):
//!   rvc    D:\MyDev\TESTING\utai-v2-testing\smoke_rvc\weights\smoke_rvc_best.pth
//!   sovits D:\MyDev\TESTING\utai-v2-testing\smoke_sovits41\weights\smoke_sovits41_best.pth
//!          (+ sibling config.json, written by the S38 trainer)
//!   vocoder D:\MyDev\TESTING\smoke_vocoder\ws\weights\vocoder_best.ckpt (+config.json)
//!   aux fleet under data\models\aux (same layout as the app)
//!
//! Run:  cargo test --test audition_render -- --ignored --nocapture
//! (GPU by default — these are perceptual-threshold legs, not bitwise gates;
//!  UTAI_AUD_CPU=1 forces the CPU EP.)

use std::path::PathBuf;

use utai_lib::inference::engine::{DeviceConfig, OnnxEngine};
use utai_lib::inference::{RvcOptions, SovitsOptions};

fn app_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

fn init_ort() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("utai_lib=info")),
        )
        .try_init();
    utai_lib::suppress_windows_dll_error_dialogs();
    utai_lib::setup_cuda_dll_paths(&app_root());
    utai_lib::init_ort_runtime(&app_root());
}

fn aux_dir() -> PathBuf {
    app_root().join("data").join("models").join("aux")
}

fn audition_wav() -> PathBuf {
    app_root().join("training").join("assets").join("audition_10s.wav")
}

fn work_dir() -> PathBuf {
    let d = std::env::temp_dir().join("s41_audition_test");
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn skip_unless(paths: &[&PathBuf]) -> bool {
    for p in paths {
        if !p.is_file() {
            eprintln!("[skip] fixture missing: {}", p.display());
            return false;
        }
    }
    true
}

fn set_device(engine: &OnnxEngine) {
    if std::env::var("UTAI_AUD_CPU").as_deref() == Ok("1") {
        engine.set_device(DeviceConfig::Cpu);
    }
}

struct PitchKit {
    rmvpe_sid: String,
    mel: ndarray::Array2<f32>,
}

fn pitch_kit(engine: &OnnxEngine) -> PitchKit {
    let rmvpe_sid = engine
        .load_model_on(&aux_dir().join("rmvpe_e2e.onnx"), false, DeviceConfig::Cpu)
        .expect("load rmvpe");
    let mel: ndarray::Array2<f32> =
        ndarray_npy::read_npy(aux_dir().join("rmvpe_mel_filters.npy")).expect("mel filters");
    PitchKit { rmvpe_sid, mel }
}

fn f0_of(engine: &OnnxEngine, kit: &PitchKit, samples: &[f32], sr: u32) -> Vec<f32> {
    let wav16k =
        utai_lib::inference::features::resample(samples, sr, utai_lib::inference::f0::RMVPE_SR);
    utai_lib::inference::f0::rmvpe_detect(
        engine,
        &kit.rmvpe_sid,
        &kit.mel,
        &wav16k,
        utai_lib::inference::f0::SOVITS_RMVPE_THRESHOLD,
    )
    .expect("rmvpe detect")
}

/// V17 assertion set on a rendered clip vs the source clip.
fn assert_render(
    label: &str,
    engine: &OnnxEngine,
    kit: &PitchKit,
    src: &(Vec<f32>, u32),
    out: &[f32],
    out_sr: u32,
    want_sr: u32,
) {
    assert_eq!(out_sr, want_sr, "{}: sample rate", label);
    let dur_in = src.0.len() as f64 / src.1 as f64;
    let dur_out = out.len() as f64 / out_sr as f64;
    assert!(
        (dur_out - dur_in).abs() / dur_in < 0.005,
        "{}: duration {:.3}s vs source {:.3}s",
        label,
        dur_out,
        dur_in
    );
    assert!(out.iter().all(|v| v.is_finite()), "{}: non-finite samples", label);
    let peak = out.iter().fold(0f32, |m, v| m.max(v.abs()));
    assert!(peak <= 1.0 + 1e-4, "{}: peak {}", label, peak);
    let rms = (out.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>()
        / out.len().max(1) as f64)
        .sqrt();
    assert!(
        20.0 * rms.log10() > -50.0,
        "{}: RMS {:.1} dBFS (near-silence)",
        label,
        20.0 * rms.log10()
    );

    // pitch identity — transpose 0, so output f0 must track input f0
    let f0_in = f0_of(engine, kit, &src.0, src.1);
    let f0_out = f0_of(engine, kit, out, out_sr);
    let n = f0_in.len().min(f0_out.len());
    let mut cents: Vec<f64> = Vec::new();
    for i in 0..n {
        if f0_in[i] > 0.0 && f0_out[i] > 0.0 {
            cents.push(1200.0 * (f0_out[i] as f64 / f0_in[i] as f64).log2());
        }
    }
    assert!(
        cents.len() >= 50,
        "{}: too few both-voiced frames ({})",
        label,
        cents.len()
    );
    cents.sort_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap());
    let median = cents[cents.len() / 2].abs();
    assert!(median < 50.0, "{}: pitch identity broken (median {:.1} cents)", label, median);
    eprintln!(
        "[{}] OK — dur {:.2}s peak {:.3} rms {:.1} dBFS pitch median {:.1} cents ({} frames)",
        label,
        dur_out,
        peak,
        20.0 * rms.log10(),
        median,
        cents.len()
    );
}

fn read_sidecar(model: &PathBuf) -> serde_json::Value {
    let json_path = model.with_extension("json");
    serde_json::from_str(&std::fs::read_to_string(&json_path).expect("read sidecar"))
        .expect("parse sidecar")
}

fn convert_candidate_cached(
    ckpt: &PathBuf,
    out_onnx: &PathBuf,
    mtype: &utai_lib::models::ModelType,
) {
    if out_onnx.is_file() {
        return; // conversion cache — mirrors the production short-circuit
    }
    utai_lib::models::convert::convert_pth_to_onnx(ckpt, out_onnx, mtype, &app_root())
        .expect("convert candidate");
}

fn load_source() -> (Vec<f32>, u32) {
    let buf = utai_lib::audio::load_audio(&audition_wav()).expect("load audition clip");
    let mono = utai_lib::audio::resample::to_mono(&buf);
    (mono.samples.clone(), mono.sample_rate)
}

#[test]
#[ignore]
fn audition_rvc_candidate() {
    let ckpt =
        PathBuf::from(r"D:\MyDev\TESTING\utai-v2-testing\smoke_rvc\weights\smoke_rvc_best.pth");
    let clip = audition_wav();
    if !skip_unless(&[&ckpt, &clip]) {
        return;
    }
    init_ort();
    let engine = OnnxEngine::new();
    set_device(&engine);

    let onnx = work_dir().join("rvc_best.onnx");
    convert_candidate_cached(&ckpt, &onnx, &utai_lib::models::ModelType::Rvc);
    let sc = read_sidecar(&onnx);
    let dim = sc["features_dim"].as_u64().expect("features_dim") as usize;
    let sample_rate = sc["sample_rate"].as_u64().expect("sample_rate") as u32;

    let cv = engine
        .load_model_on(
            &aux_dir().join(if dim == 256 { "contentvec_256l9.onnx" } else { "contentvec_768l12.onnx" }),
            false,
            DeviceConfig::Cpu,
        )
        .expect("contentvec");
    let kit = pitch_kit(&engine);
    let voice = engine.load_model_with(&onnx, false).expect("candidate session");
    let audio = utai_lib::audio::load_audio(&clip).expect("clip");

    let m = utai_lib::inference::rvc::RvcModel {
        engine: &engine,
        voice_session: &voice,
        contentvec_session: &cv,
        rmvpe_session: &kit.rmvpe_sid,
        mel_filters: &kit.mel,
        index: None,
        sample_rate,
        features_dim: dim,
        spk_mix: None, // single-speaker test model → the sid path (①c multi-speaker unused here)
        noise_channels: 192,
        min_frames: sc["min_frames"].as_u64().unwrap_or(12) as usize,
    };
    let options = RvcOptions { index_ratio: 0.0, ..Default::default() };
    let r = utai_lib::inference::rvc::run_pipeline(&m, &audio, &options, &|_| {}, &|| false)
        .expect("pipeline");
    let src = load_source();
    assert_render("rvc", &engine, &kit, &src, &r.audio, r.sample_rate, sample_rate);
}

#[test]
#[ignore]
fn audition_sovits_candidate() {
    let ckpt = PathBuf::from(
        r"D:\MyDev\TESTING\utai-v2-testing\smoke_sovits41\weights\smoke_sovits41_best.pth",
    );
    let clip = audition_wav();
    if !skip_unless(&[&ckpt, &clip]) {
        return;
    }
    init_ort();
    let engine = OnnxEngine::new();
    set_device(&engine);

    let onnx = work_dir().join("sovits_best.onnx");
    convert_candidate_cached(&ckpt, &onnx, &utai_lib::models::ModelType::SoVits);
    let sc = read_sidecar(&onnx);
    let dim = if sc["speech_encoder"].as_str() == Some("vec256l9") { 256 } else { 768 };
    let sample_rate = sc["sample_rate"].as_u64().expect("sample_rate") as u32;
    let hop = sc["hop_size"].as_u64().unwrap_or(512) as usize;

    let cv = engine
        .load_model_on(
            &aux_dir().join(if dim == 256 { "contentvec_256l9.onnx" } else { "contentvec_768l12.onnx" }),
            false,
            DeviceConfig::Cpu,
        )
        .expect("contentvec");
    let kit = pitch_kit(&engine);
    let voice = engine.load_model_with(&onnx, false).expect("candidate session");
    let audio = utai_lib::audio::load_audio(&clip).expect("clip");

    let vol_embedding = sc["inputs"]
        .as_array()
        .map(|l| l.iter().any(|v| v.as_str() == Some("vol")))
        .unwrap_or(false);
    let m = utai_lib::inference::sovits::SovitsModel {
        engine: &engine,
        voice_session: &voice,
        contentvec_session: &cv,
        rmvpe_session: &kit.rmvpe_sid,
        mel_filters: &kit.mel,
        cluster: None,
        diffusion: None,
        vocoder: None,
        f0_predictor_session: None,
        sample_rate,
        hop_size: hop,
        features_dim: dim,
        vol_embedding,
        spk_mix: None, // single-speaker test model → the sid path (①c multi-speaker unused here)
        unit_interpolate_mode: "left".into(),
        noise_channels: 192,
        min_frames: sc["min_frames"].as_u64().unwrap_or(6) as usize,
    };
    let options = SovitsOptions { cluster_ratio: 0.0, ..Default::default() };
    let r = utai_lib::inference::sovits::run_pipeline(&m, &audio, &options, &|_| {}, &|| false)
        .expect("pipeline");
    let src = load_source();
    assert_render("sovits", &engine, &kit, &src, &r.audio, r.sample_rate, sample_rate);
}

#[test]
#[ignore]
fn audition_vocoder_ab() {
    let ckpt = PathBuf::from(r"D:\MyDev\TESTING\smoke_vocoder\ws\weights\vocoder_best.ckpt");
    let clip = audition_wav();
    let aux_voc = aux_dir().join("nsf_hifigan.onnx");
    if !skip_unless(&[&ckpt, &clip, &aux_voc]) {
        return;
    }
    init_ort();
    let engine = OnnxEngine::new();
    set_device(&engine);
    let kit = pitch_kit(&engine);

    // candidate triple via the exporter (cached like the production dir)
    let vdir = work_dir().join("voc_best");
    if !vdir.join("vocoder.onnx").is_file() {
        std::fs::create_dir_all(&vdir).unwrap();
        utai_lib::models::convert::convert_vocoder_to_onnx(&ckpt, None, &vdir, "vocoder", &app_root())
            .expect("convert vocoder candidate");
    }

    // shared self-loop front half: 16k → f0, 44.1k → source samples
    let buf = utai_lib::audio::load_audio(&clip).expect("clip");
    let mono = utai_lib::audio::resample::to_mono(&buf);
    let x44 = utai_lib::inference::features::resample(&mono.samples, mono.sample_rate, 44100);
    let wav16k = utai_lib::inference::features::resample(
        &mono.samples,
        mono.sample_rate,
        utai_lib::inference::f0::RMVPE_SR,
    );
    let f0_raw = utai_lib::inference::f0::rmvpe_detect(
        &engine,
        &kit.rmvpe_sid,
        &kit.mel,
        &wav16k,
        utai_lib::inference::f0::SOVITS_RMVPE_THRESHOLD,
    )
    .expect("f0");

    let mut renders: Vec<Vec<f32>> = Vec::new();
    for (label, onnx, mel_npy) in [
        ("voc-candidate", vdir.join("vocoder.onnx"), vdir.join("vocoder_mel.npy")),
        ("voc-default", aux_voc.clone(), aux_dir().join("nsf_hifigan_mel.npy")),
    ] {
        let filters: ndarray::Array2<f32> = ndarray_npy::read_npy(&mel_npy).expect("filterbank");
        let sid = engine.load_model_with(&onnx, false).expect("vocoder session");
        let mel = utai_lib::inference::mel::nsf_mel(&x44, &filters);
        let (f0, _uv) =
            utai_lib::inference::f0::sovits_f0_postprocess(&f0_raw, mel.ncols(), 512, 44100);
        let out = utai_lib::inference::nsf_hifigan::vocode(&engine, &sid, &mel, &f0)
            .expect("vocode");
        let src = (x44.clone(), 44100u32);
        assert_render(label, &engine, &kit, &src, &out, 44100, 44100);
        renders.push(out);
    }
    // A/B: identical output would mean the candidate override never took
    // (S31 "output didn't switch" audit pattern)
    assert!(
        renders[0] != renders[1],
        "candidate render is bit-identical to the default vocoder — override not in effect"
    );
}
