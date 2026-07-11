//! 关卡-2 E2E harness for the VOICE pipeline (RVC / SoVITS full chain).
//!
//! Mirrors tests\separation_pipeline.rs's init_ort + env-gated pattern. It constructs
//! RvcModel/SovitsModel directly (engine + aux sessions), reads the model's sidecar json
//! for the model facts, runs run_pipeline, and writes an f32 wav — so a Python reference
//! (converter\verify\voice\e2e_{rvc,sovits}_ref.py) can SNR-compare the two.
//!
//! Run (bash), CPU EP for numerical purity — CUDA needs runtime\cuda on PATH:
//!   UTAI_VOICE_KIND=rvc UTAI_VOICE_INPUT=<16k mono wav> \
//!   UTAI_VOICE_MODEL=<...\lengv2.3.det.onnx> UTAI_VOICE_INDEX=<...\lengv2.3.npy> \
//!   UTAI_VOICE_OPTS='{"noise_scale":0.0,"index_ratio":0.0,"protect":0.33,"rms_mix_rate":0.25}' \
//!   UTAI_VOICE_OUT=<out.wav> \
//!   cargo test --test voice_pipeline voice_env_wav -- --ignored --nocapture
//!
//! UTAI_VOICE_DEVICE=auto uses the GPU chain (default: CPU EP). Aux models (ContentVec
//! variants, RMVPE, mel filters) resolve from data\models\aux — same layout as the app.

use std::path::PathBuf;

use utai_lib::inference::engine::{DeviceConfig, OnnxEngine};
use utai_lib::inference::{RvcOptions, SovitsOptions};

fn app_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

/// S40 wire-compat pin: every pre-S40 payload (old .usp projects, older
/// frontends) carries NO vocoder_name key and must land on None — the
/// byte-identical aux-default vocoder path; unknown future keys must not
/// break deserialization either (no deny_unknown_fields).
#[test]
fn sovits_options_vocoder_name_serde_compat() {
    let legacy: SovitsOptions = serde_json::from_str("{}").unwrap();
    assert!(legacy.vocoder_name.is_none(), "legacy payload => default vocoder");
    let null: SovitsOptions = serde_json::from_str(r#"{"vocoder_name":null}"#).unwrap();
    assert!(null.vocoder_name.is_none());
    let named: SovitsOptions =
        serde_json::from_str(r#"{"vocoder_name":"かざね 声码器"}"#).unwrap();
    assert_eq!(named.vocoder_name.as_deref(), Some("かざね 声码器"));
    let forward: SovitsOptions =
        serde_json::from_str(r#"{"vocoder_name":"x","some_future_key":123}"#).unwrap();
    assert_eq!(forward.vocoder_name.as_deref(), Some("x"));
}

/// The app does this in run() before any ort use; without it, `cargo test` hangs forever
/// at 0 CPU on the first session build (invisible modal DLL dialog + uninitialized
/// load-dynamic ORT). Copied from separation_pipeline.rs.
fn init_ort() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("utai_lib=info")),
        )
        .try_init();
    utai_lib::suppress_windows_dll_error_dialogs();
    // cudnn 9's shim resolves its sub-DLLs via PATH at graph-build time — the
    // app sets this up in run(); a bare harness without it fails the first
    // CUDA Conv with CUDNN_BACKEND_API_FAILED (looks like environment drift).
    utai_lib::setup_cuda_dll_paths(&app_root());
    utai_lib::init_ort_runtime(&app_root());
}

fn aux_dir() -> PathBuf {
    app_root().join("data").join("models").join("aux")
}

/// ContentVec variant routing (mirrors commands\inference.rs): 768 → vec768l12, 256 → vec256l9.
fn contentvec_path(dim: usize) -> PathBuf {
    let name = match dim {
        768 => "contentvec_768l12.onnx",
        256 => "contentvec_256l9.onnx",
        other => panic!("unsupported features_dim {} (want 256/768)", other),
    };
    aux_dir().join(name)
}

fn read_sidecar(model: &PathBuf) -> serde_json::Value {
    let json_path = model.with_extension("json");
    let text = std::fs::read_to_string(&json_path)
        .unwrap_or_else(|e| panic!("read sidecar {}: {}", json_path.display(), e));
    serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("parse sidecar {}: {}", json_path.display(), e))
}

fn features_dim(sc: &serde_json::Value) -> usize {
    // SoVITS: speech_encoder wins; RVC carries features_dim directly.
    if let Some(enc) = sc.get("speech_encoder").and_then(|v| v.as_str()) {
        return match enc {
            "vec768l12" => 768,
            "vec256l9" => 256,
            other => panic!("unsupported speech_encoder {}", other),
        };
    }
    sc.get("features_dim").and_then(|v| v.as_u64()).expect("features_dim") as usize
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

fn set_device(engine: &OnnxEngine) {
    // CPU EP for numerical parity (CUDA TF32 blurs the comparison); UTAI_VOICE_DEVICE=auto
    // uses the GPU chain — mirrors UTAI_SEP_DEVICE in the separation harness.
    if std::env::var("UTAI_VOICE_DEVICE").as_deref() != Ok("auto") {
        engine.set_device(DeviceConfig::Cpu);
    }
}

#[test]
#[ignore]
fn voice_env_wav() {
    let kind = std::env::var("UTAI_VOICE_KIND").expect("set UTAI_VOICE_KIND=rvc|sovits");
    let input = PathBuf::from(std::env::var("UTAI_VOICE_INPUT").expect("set UTAI_VOICE_INPUT"));
    let model = PathBuf::from(std::env::var("UTAI_VOICE_MODEL").expect("set UTAI_VOICE_MODEL"));
    let out = PathBuf::from(std::env::var("UTAI_VOICE_OUT").expect("set UTAI_VOICE_OUT"));
    let opts_json = std::env::var("UTAI_VOICE_OPTS").unwrap_or_else(|_| "{}".to_string());

    init_ort();
    let engine = OnnxEngine::new();
    set_device(&engine);

    let sc = read_sidecar(&model);
    let dim = features_dim(&sc);
    let sample_rate = sc.get("sample_rate").and_then(|v| v.as_u64()).expect("sample_rate") as u32;
    let nch = noise_channels(&sc);

    let audio = utai_lib::audio::load_audio(&input).expect("load input wav");

    // Mirror the app (commands\inference.rs / InferenceManager): aux feature extractors on CPU
    // (they're the dominant VRAM consumer — full-signal fp32 activations), voice synth on the
    // global device. mem_pattern=false everywhere (dynamic shapes).
    let cv_sid = engine
        .load_model_on(&contentvec_path(dim), false, DeviceConfig::Cpu)
        .expect("load contentvec");
    let rmvpe_sid = engine
        .load_model_on(&aux_dir().join("rmvpe_e2e.onnx"), false, DeviceConfig::Cpu)
        .expect("load rmvpe");
    let voice_sid = engine.load_model_with(&model, false).expect("load voice model");
    let mel: ndarray::Array2<f32> =
        ndarray_npy::read_npy(aux_dir().join("rmvpe_mel_filters.npy")).expect("mel filters");

    eprintln!(
        "[harness] kind={} dim={} sr={} nch={} device={}",
        kind,
        dim,
        sample_rate,
        nch,
        if std::env::var("UTAI_VOICE_DEVICE").as_deref() == Ok("auto") { "auto" } else { "cpu" }
    );

    let result = match kind.as_str() {
        "rvc" => {
            let options: RvcOptions = serde_json::from_str(&opts_json).expect("parse RvcOptions");
            eprintln!("[harness] RvcOptions = {:?}", options);
            let min_frames = sc.get("min_frames").and_then(|v| v.as_u64()).unwrap_or(12) as usize;
            let index = std::env::var("UTAI_VOICE_INDEX").ok().map(|p| {
                utai_lib::inference::rvc::RvcIndex::load(&PathBuf::from(p)).expect("load index npy")
            });
            let m = utai_lib::inference::rvc::RvcModel {
                engine: &engine,
                voice_session: &voice_sid,
                contentvec_session: &cv_sid,
                rmvpe_session: &rmvpe_sid,
                mel_filters: &mel,
                index: index.as_ref(),
                sample_rate,
                features_dim: dim,
                spk_mix: None, // single-speaker E2E fixture → the sid path (①c multi-speaker unused here)
                noise_channels: nch,
                min_frames,
            };
            utai_lib::inference::rvc::run_pipeline(&m, &audio, &options, &|_p| {}, &|| false)
                .expect("rvc pipeline")
        }
        "sovits" => {
            let options: SovitsOptions =
                serde_json::from_str(&opts_json).expect("parse SovitsOptions");
            eprintln!("[harness] SovitsOptions = {:?}", options);
            let hop_size = sc.get("hop_size").and_then(|v| v.as_u64()).unwrap_or(512) as usize;
            let min_frames = sc.get("min_frames").and_then(|v| v.as_u64()).unwrap_or(6) as usize;
            let vol_embedding = sc
                .get("inputs")
                .and_then(|v| v.as_array())
                .map(|l| l.iter().any(|v| v.as_str() == Some("vol")))
                .unwrap_or_else(|| {
                    sc.get("vol_embedding").and_then(|v| v.as_bool()).unwrap_or(false)
                });
            let unit_interpolate_mode = sc
                .get("unit_interpolate_mode")
                .and_then(|v| v.as_str())
                .unwrap_or("left")
                .to_string();
            // S36 quality-path env plumbing (all optional; mirrors resolve_sovits_quality
            // WITHOUT its validations — the gate scripts control their own inputs):
            //   UTAI_VOICE_DIFF   = <stem>.diffusion dir (encoder/denoiser onnx + diffusion.json)
            //   UTAI_VOICE_VOCODER= path to nsf_hifigan.onnx (sidecar .json + mel .npy siblings
            //                       named nsf_hifigan.json / nsf_hifigan_mel.npy in the same dir)
            //   UTAI_VOICE_F0PRED = path to <stem>.f0.onnx
            let vocoder = std::env::var("UTAI_VOICE_VOCODER").ok().map(|p| {
                let voc_path = PathBuf::from(&p);
                let dir = voc_path.parent().expect("vocoder parent dir").to_path_buf();
                let vj: serde_json::Value = serde_json::from_str(
                    &std::fs::read_to_string(dir.join("nsf_hifigan.json")).expect("vocoder json"),
                )
                .expect("parse vocoder json");
                let filters: ndarray::Array2<f32> =
                    ndarray_npy::read_npy(dir.join("nsf_hifigan_mel.npy")).expect("vocoder mel npy");
                let sid = engine.load_model_with(&voc_path, false).expect("load vocoder");
                utai_lib::inference::sovits::VocoderRuntime {
                    session: sid,
                    mel_filters: std::sync::Arc::new(filters),
                    cfg: utai_lib::inference::nsf_hifigan::VocoderConfig {
                        sample_rate: vj["sample_rate"].as_u64().unwrap_or(44100) as u32,
                        hop_size: vj["hop_size"].as_u64().unwrap_or(512) as usize,
                        num_mels: vj["num_mels"].as_u64().unwrap_or(128) as usize,
                    },
                }
            });
            let diffusion = std::env::var("UTAI_VOICE_DIFF").ok().map(|p| {
                let dir = PathBuf::from(&p);
                let dj: serde_json::Value = serde_json::from_str(
                    &std::fs::read_to_string(dir.join("diffusion.json")).expect("diffusion json"),
                )
                .expect("parse diffusion json");
                let as_f32_vec = |v: &serde_json::Value, dflt: f32| -> Vec<f32> {
                    v.as_array()
                        .map(|a| a.iter().filter_map(|x| x.as_f64()).map(|x| x as f32).collect())
                        .filter(|v: &Vec<f32>| !v.is_empty())
                        .unwrap_or_else(|| vec![dflt])
                };
                let schedule = utai_lib::inference::diffusion::DiffusionSchedule::linear(
                    dj["timesteps"].as_u64().expect("timesteps") as usize,
                    dj["max_beta"].as_f64().unwrap_or(0.02),
                    &as_f32_vec(&dj["spec_min"], -12.0),
                    &as_f32_vec(&dj["spec_max"], 2.0),
                    dj["k_step_max"].as_u64().unwrap_or(0) as usize,
                );
                let enc = engine
                    .load_model_with(&dir.join("encoder.onnx"), false)
                    .expect("load diffusion encoder");
                let den = engine
                    .load_model_with(&dir.join("denoiser.onnx"), false)
                    .expect("load diffusion denoiser");
                utai_lib::inference::sovits::DiffusionRuntime {
                    encoder_session: enc,
                    denoiser_session: den,
                    schedule,
                    method: utai_lib::inference::diffusion::SamplerMethod::parse(
                        &options.diffusion_method,
                    )
                    .expect("diffusion_method"),
                    n_hidden: dj["n_hidden"].as_u64().unwrap_or(256) as usize,
                    n_spk: dj["n_spk"].as_u64().unwrap_or(1) as usize,
                    unit_interpolate_mode: dj["unit_interpolate_mode"]
                        .as_str()
                        .unwrap_or("left")
                        .to_string(),
                }
            });
            let f0_predictor = std::env::var("UTAI_VOICE_F0PRED").ok().map(|p| {
                engine
                    .load_model_with(&PathBuf::from(p), false)
                    .expect("load f0 predictor")
            });
            let m = utai_lib::inference::sovits::SovitsModel {
                engine: &engine,
                voice_session: &voice_sid,
                contentvec_session: &cv_sid,
                rmvpe_session: &rmvpe_sid,
                mel_filters: &mel,
                cluster: None,
                diffusion,
                vocoder,
                f0_predictor_session: f0_predictor,
                sample_rate,
                hop_size,
                features_dim: dim,
                vol_embedding,
                spk_mix: None, // single-speaker E2E fixture → the sid path (①c multi-speaker unused here)
                unit_interpolate_mode,
                noise_channels: nch,
                min_frames,
            };
            utai_lib::inference::sovits::run_pipeline(&m, &audio, &options, &|_p| {}, &|| false)
                .expect("sovits pipeline")
        }
        other => panic!("UTAI_VOICE_KIND must be rvc|sovits (got {})", other),
    };

    let peak = result.audio.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let rms = (result.audio.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>()
        / result.audio.len().max(1) as f64)
        .sqrt();
    let nan = result.audio.iter().any(|x| x.is_nan());
    eprintln!(
        "[harness] out: n={} sr={} peak={:.4} rms={:.4} nan={}",
        result.audio.len(),
        result.sample_rate,
        peak,
        rms,
        nan
    );
    assert!(!nan, "output contains NaN");
    write_f32_wav(&out, &result.audio, result.sample_rate);
    eprintln!("[harness] wrote {}", out.display());
}
