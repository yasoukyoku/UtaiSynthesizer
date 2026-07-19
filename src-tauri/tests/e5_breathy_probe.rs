//! E5 裁决实验 harness — breathy/falsetto expressiveness through the REAL cover pipeline.
//!
//! Diagnostic only, NOT a gate (`--ignored`). Converts paired GTSinger recordings
//! (breathy vs control / falsetto vs control, same singer, same phrase) to target
//! singers through the exact production path (RMVPE → ContentVec → net_g via
//! utai_lib::inference::{rvc,sovits}::run_pipeline) so the user can EAR-judge whether
//! breathiness survives the ContentVec bottleneck. Mirrors tests\voice_pipeline.rs's
//! init_ort + model construction verbatim; no production code is touched.
//!
//! Inputs:  D:\MyDev\TESTING\e5_breathy_probe\*_src_*.wav  (ASCII-named copies)
//! Outputs: same dir, `{song}_{seg}_{cond}_{arm}.wav` (f32 wav at the model's rate)
//! Arms:    akiko40   = SoVITS 4.0 akiko_320000 (vec256l9, no vol) — cleanest isolation
//!          dxl41     = SoVITS 4.1 东雪莲 (vec768l12, vol_embedding) — production-ish
//!          rvc_idx0  = RVC lengv2.3, index_ratio = 0.0
//!          rvc_idx75 = RVC lengv2.3, index_ratio = 0.75 (shipped default)
//! All other options are the shipped defaults (f0_shift = 0, seed = 0, auto_f0 off).
//! Existing outputs are SKIPPED, so an interrupted run resumes where it stopped.
//!
//! Run (PowerShell, from src-tauri; CPU EP by default for determinism):
//!   cargo test --test e5_breathy_probe e5_breathy_probe -- --ignored --nocapture
//! Optional env:
//!   UTAI_E5_DEVICE=auto      voice net_g on the GPU chain (aux stays CPU, like the app)
//!   UTAI_E5_ONLY=yunyan_0002 only inputs whose file name contains this substring
//!   UTAI_E5_ARMS=akiko40,rvc_idx0   subset of arms

use std::path::PathBuf;
use std::time::Instant;

use utai_lib::inference::engine::{DeviceConfig, OnnxEngine};
use utai_lib::inference::{RvcOptions, SovitsOptions};

const WORK_DIR: &str = r"D:\MyDev\TESTING\e5_breathy_probe";

fn app_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

/// Same init as voice_pipeline.rs (bare harnesses must init ORT or they hang forever
/// at 0 CPU on the invisible modal DLL dialog).
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
    app_root().join("data").join("models").join(utai_lib::models::AUX_DIR_NAME)
}

fn read_sidecar(model: &PathBuf) -> serde_json::Value {
    let json_path = model.with_extension("json");
    let text = std::fs::read_to_string(&json_path)
        .unwrap_or_else(|e| panic!("read sidecar {}: {}", json_path.display(), e));
    serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("parse sidecar {}: {}", json_path.display(), e))
}

fn noise_channels(sc: &serde_json::Value) -> usize {
    sc.get("noise")
        .and_then(|v| v.get("rnd_input").or_else(|| v.get("noise_input")))
        .and_then(|v| v.as_array())
        .and_then(|a| a.get(1))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(192)
}

fn write_f32_wav(path: &PathBuf, samples: &[f32], sample_rate: u32) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut w = hound::WavWriter::create(path, spec).unwrap();
    for &s in samples {
        w.write_sample(s).unwrap();
    }
    w.finalize().unwrap();
}

/// `{song}_{seg}_src_{cond}.wav` → `{song}_{seg}_{cond}_{arm}.wav`
fn out_path(work: &PathBuf, src_name: &str, arm: &str) -> PathBuf {
    let stem = src_name.strip_suffix(".wav").expect("wav suffix");
    work.join(format!("{}_{}.wav", stem.replace("_src_", "_"), arm))
}

#[test]
#[ignore]
fn e5_breathy_probe() {
    init_ort();
    let engine = OnnxEngine::new();
    // CPU EP by default (deterministic, comparable arms); UTAI_E5_DEVICE=auto = GPU chain.
    if std::env::var("UTAI_E5_DEVICE").as_deref() != Ok("auto") {
        engine.set_device(DeviceConfig::Cpu);
    }

    let work = PathBuf::from(WORK_DIR);
    let only = std::env::var("UTAI_E5_ONLY").unwrap_or_default();
    let arms_filter = std::env::var("UTAI_E5_ARMS").unwrap_or_default();
    let arm_enabled = |name: &str| -> bool {
        arms_filter.is_empty() || arms_filter.split(',').any(|a| a.trim() == name)
    };

    let mut sources: Vec<String> = std::fs::read_dir(&work)
        .expect("read work dir")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".wav") && n.contains("_src_") && n.contains(&only))
        .collect();
    sources.sort();
    assert!(!sources.is_empty(), "no *_src_*.wav inputs in {}", work.display());
    eprintln!("[e5] {} source files", sources.len());

    // ── aux models: ALWAYS CPU, exactly like the app's default (gpu_extract=false) ──
    let cv256_sid = engine
        .load_model_on(&aux_dir().join("contentvec_256l9.onnx"), false, DeviceConfig::Cpu)
        .expect("load contentvec 256");
    let cv768_sid = engine
        .load_model_on(&aux_dir().join("contentvec_768l12.onnx"), false, DeviceConfig::Cpu)
        .expect("load contentvec 768");
    let rmvpe_sid = engine
        .load_model_on(&aux_dir().join("rmvpe_e2e.onnx"), false, DeviceConfig::Cpu)
        .expect("load rmvpe");
    let mel: ndarray::Array2<f32> =
        ndarray_npy::read_npy(aux_dir().join("rmvpe_mel_filters.npy")).expect("mel filters");

    let sovits_dir = app_root().join("data").join("models").join("sovits");
    let rvc_dir = app_root().join("data").join("models").join("rvc");

    // ── SoVITS arms (akiko40 / dxl41): sidecar-driven construction, verbatim from
    //    voice_pipeline.rs (no diffusion / vocoder / cluster / f0 predictor — defaults) ──
    let sovits_arms: [(&str, PathBuf); 2] = [
        ("akiko40", sovits_dir.join("akiko_320000.onnx")),
        ("dxl41", sovits_dir.join("Sovits4.1东雪莲主模型.onnx")),
    ];
    for (arm, model_path) in &sovits_arms {
        if !arm_enabled(arm) {
            continue;
        }
        assert!(model_path.exists(), "missing model {}", model_path.display());
        let sc = read_sidecar(model_path);
        let enc = sc["speech_encoder"].as_str().expect("speech_encoder");
        let dim = match enc {
            "vec768l12" => 768usize,
            "vec256l9" => 256usize,
            other => panic!("unsupported speech_encoder {}", other),
        };
        let sample_rate = sc["sample_rate"].as_u64().expect("sample_rate") as u32;
        let hop_size = sc["hop_size"].as_u64().unwrap_or(512) as usize;
        let min_frames = sc["min_frames"].as_u64().unwrap_or(6) as usize;
        let nch = noise_channels(&sc);
        let inputs_list = sc.get("inputs").and_then(|v| v.as_array());
        let vol_embedding = inputs_list
            .map(|l| l.iter().any(|v| v.as_str() == Some("vol")))
            .unwrap_or_else(|| sc.get("vol_embedding").and_then(|v| v.as_bool()).unwrap_or(false));
        let feed_uv = inputs_list
            .map(|l| l.iter().any(|v| v.as_str() == Some("uv")))
            .unwrap_or(true);
        let unit_interpolate_mode =
            sc.get("unit_interpolate_mode").and_then(|v| v.as_str()).unwrap_or("left").to_string();
        let phase_bins = sc
            .get("phase")
            .and_then(|v| v.get("phase_input"))
            .and_then(|v| v.as_array())
            .and_then(|a| a.get(1))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let f0d_cond_channels = sc
            .get("f0d_cond")
            .and_then(|v| v.get("input"))
            .and_then(|v| v.as_array())
            .and_then(|a| a.get(1))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        let voice_sid = engine.load_model_with(model_path, false).expect("load sovits net_g");
        let cv_sid = if dim == 768 { &cv768_sid } else { &cv256_sid };
        let options = SovitsOptions::default(); // f0_shift 0, seed 0, noise 0.4, auto_f0 off
        eprintln!("[e5] arm {}: {} (dim={} sr={} vol={} uv={})",
            arm, model_path.display(), dim, sample_rate, vol_embedding, feed_uv);

        let m = utai_lib::inference::sovits::SovitsModel {
            engine: &engine,
            voice_session: &voice_sid,
            contentvec_session: cv_sid,
            rmvpe_session: &rmvpe_sid,
            mel_filters: &mel,
            cluster: None,
            diffusion: None,
            vocoder: None,
            f0_predictor_session: None,
            sample_rate,
            hop_size,
            features_dim: dim,
            vol_embedding,
            phase_bins,
            f0d_cond_channels,
            feed_uv,
            spk_mix: None,
            unit_interpolate_mode,
            noise_channels: nch,
            min_frames,
        };
        for src in &sources {
            let out = out_path(&work, src, arm);
            if out.exists() {
                eprintln!("[e5] skip (exists): {}", out.display());
                continue;
            }
            let t0 = Instant::now();
            let audio = utai_lib::audio::load_audio(&work.join(src)).expect("load input wav");
            let result = utai_lib::inference::sovits::run_pipeline(
                &m, &audio, &options, None, &|_p| {}, &|| false,
            )
            .expect("sovits pipeline");
            assert!(!result.audio.iter().any(|x| x.is_nan()), "NaN in {} output for {}", arm, src);
            write_f32_wav(&out, &result.audio, result.sample_rate);
            eprintln!("[e5] {} <- {} ({:.1}s)", out.display(), src, t0.elapsed().as_secs_f64());
        }
        engine.unload_model(&voice_sid);
    }

    // ── RVC arms (rvc_idx0 / rvc_idx75): same model, only index_ratio differs ──
    let rvc_model = rvc_dir.join("lengv2.3.onnx");
    let rvc_arms: [(&str, f32); 2] = [("rvc_idx0", 0.0), ("rvc_idx75", 0.75)];
    let any_rvc = rvc_arms.iter().any(|(a, _)| arm_enabled(a));
    if any_rvc {
        assert!(rvc_model.exists(), "missing model {}", rvc_model.display());
        let sc = read_sidecar(&rvc_model);
        let dim = sc["features_dim"].as_u64().expect("features_dim") as usize;
        let sample_rate = sc["sample_rate"].as_u64().expect("sample_rate") as u32;
        let min_frames = sc["min_frames"].as_u64().unwrap_or(12) as usize;
        let nch = noise_channels(&sc);
        let index = utai_lib::inference::rvc::RvcIndex::load(&rvc_dir.join("lengv2.3.npy"))
            .expect("load index npy");
        let voice_sid = engine.load_model_with(&rvc_model, false).expect("load rvc net_g");
        let cv_sid = if dim == 768 { &cv768_sid } else { &cv256_sid };

        let m = utai_lib::inference::rvc::RvcModel {
            engine: &engine,
            voice_session: &voice_sid,
            contentvec_session: cv_sid,
            rmvpe_session: &rmvpe_sid,
            mel_filters: &mel,
            index: Some(&index),
            sample_rate,
            features_dim: dim,
            spk_mix: None,
            noise_channels: nch,
            min_frames,
        };
        for (arm, ratio) in &rvc_arms {
            if !arm_enabled(arm) {
                continue;
            }
            let options = RvcOptions { index_ratio: *ratio, ..RvcOptions::default() };
            eprintln!("[e5] arm {}: {} (dim={} sr={} index_ratio={})",
                arm, rvc_model.display(), dim, sample_rate, ratio);
            for src in &sources {
                let out = out_path(&work, src, arm);
                if out.exists() {
                    eprintln!("[e5] skip (exists): {}", out.display());
                    continue;
                }
                let t0 = Instant::now();
                let audio = utai_lib::audio::load_audio(&work.join(src)).expect("load input wav");
                let result = utai_lib::inference::rvc::run_pipeline(
                    &m, &audio, &options, None, &|_p| {}, &|| false,
                )
                .expect("rvc pipeline");
                assert!(!result.audio.iter().any(|x| x.is_nan()), "NaN in {} output for {}", arm, src);
                write_f32_wav(&out, &result.audio, result.sample_rate);
                eprintln!("[e5] {} <- {} ({:.1}s)", out.display(), src, t0.elapsed().as_secs_f64());
            }
        }
        engine.unload_model(&voice_sid);
    }

    eprintln!("[e5] all arms done");
}
