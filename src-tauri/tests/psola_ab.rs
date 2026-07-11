//! S60-2 TD-PSOLA quality gate: our Rust engine vs praat's PSOLA (the S41-blessed
//! reference) on the bundled dry vocal, measured with the SAME rmvpe stick.
//!
//! Metric = the S41 augmentation gate recipe (run_f0_gate): expected f0 = source
//! rmvpe track × 2^(st/12); compare against the shifted output's rmvpe track on
//! both-voiced frames with the voiced-boundary erosion (±2 frames — rmvpe legally
//! disagrees by octaves on note-edge frames, S41) and the >1100 Hz ceiling exemption.
//! PASS bar (each of ±3 st, the range-extension sweet band): median |cents| ≤ 30,
//! p90 ≤ 100 (the S41 gate numbers), voiced coverage ≥ 85% of the source's.
//! ±5 st (beyond the S41 knee) prints metrics and must stay within 1.6× praat's
//! numbers when the praat reference exists (不比官方差 — the relative bar).
//!
//! Fixtures: praat refs from TESTING\utai-v2-testing\psola_ab\praat_ref.py (training
//! venv); rmvpe under data\models\aux. Skip-if-absent.
//!
//! Run:  cargo test --test psola_ab -- --ignored --nocapture   (CPU, fast)

use std::path::PathBuf;

use utai_dsp::psola::{psola_shift, PsolaParams};
use utai_lib::inference::engine::{DeviceConfig, OnnxEngine};

fn app_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

fn aux_dir() -> PathBuf {
    app_root().join("data").join("models").join("aux")
}

fn ab_dir() -> PathBuf {
    PathBuf::from(r"D:\MyDev\TESTING\utai-v2-testing\psola_ab")
}

fn init_ort() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new("utai_lib=warn"))
        .try_init();
    utai_lib::suppress_windows_dll_error_dialogs();
    utai_lib::init_ort_runtime(&app_root());
}

struct Kit {
    engine: OnnxEngine,
    rmvpe_sid: String,
    mel: ndarray::Array2<f32>,
}

fn kit() -> Kit {
    let engine = OnnxEngine::new();
    let rmvpe_sid = engine
        .load_model_on(&aux_dir().join("rmvpe_e2e.onnx"), false, DeviceConfig::Cpu)
        .expect("load rmvpe");
    let mel: ndarray::Array2<f32> =
        ndarray_npy::read_npy(aux_dir().join("rmvpe_mel_filters.npy")).expect("mel filters");
    Kit { engine, rmvpe_sid, mel }
}

fn f0_of(k: &Kit, samples: &[f32], sr: u32) -> Vec<f32> {
    let wav16k =
        utai_lib::inference::features::resample(samples, sr, utai_lib::inference::f0::RMVPE_SR);
    utai_lib::inference::f0::rmvpe_detect(
        &k.engine,
        &k.rmvpe_sid,
        &k.mel,
        &wav16k,
        utai_lib::inference::f0::SOVITS_RMVPE_THRESHOLD,
    )
    .expect("rmvpe detect")
}

/// Voiced mask eroded by ±2 frames (S41 GATE_EDGE_ERODE).
fn eroded_voiced(f0: &[f32]) -> Vec<bool> {
    let v: Vec<bool> = f0.iter().map(|&f| f > 0.0).collect();
    (0..v.len())
        .map(|i| {
            (i.saturating_sub(2)..=(i + 2).min(v.len() - 1)).all(|j| v[j])
        })
        .collect()
}

struct Metrics {
    median: f32,
    p90: f32,
    coverage: f32,
}

/// S41 gate metric: |cents| stats of `out_f0` vs `src_f0 × 2^(st/12)` on eroded
/// both-voiced frames; expected > 1100 Hz frames are exempt from p90 (headroom).
fn gate_metrics(src_f0: &[f32], out_f0: &[f32], st: f32) -> Metrics {
    let n = src_f0.len().min(out_f0.len());
    let factor = 2f32.powf(st / 12.0);
    let sv = eroded_voiced(&src_f0[..n]);
    let ov = eroded_voiced(&out_f0[..n]);
    let mut all: Vec<f32> = Vec::new();
    let mut capped: Vec<f32> = Vec::new();
    let mut src_voiced = 0usize;
    let mut both = 0usize;
    for i in 0..n {
        if sv[i] {
            src_voiced += 1;
        }
        if !(sv[i] && ov[i]) {
            continue;
        }
        both += 1;
        let expected = src_f0[i] * factor;
        let c = (1200.0 * (out_f0[i] / expected).log2()).abs();
        all.push(c);
        if expected <= 1100.0 {
            capped.push(c);
        }
    }
    all.sort_by(|a, b| a.partial_cmp(b).unwrap());
    capped.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = all.get(all.len() / 2).copied().unwrap_or(f32::NAN);
    let p90 = capped.get((capped.len() as f32 * 0.9) as usize).copied().unwrap_or(f32::NAN);
    Metrics { median: med, p90, coverage: both as f32 / src_voiced.max(1) as f32 }
}

#[test]
#[ignore]
fn psola_vs_praat_on_real_vocal() {
    let src_path = app_root().join("training").join("assets").join("audition_10s.wav");
    if !src_path.is_file() || !aux_dir().join("rmvpe_e2e.onnx").is_file() {
        eprintln!("SKIP: fixtures missing");
        return;
    }
    init_ort();
    let k = kit();

    let buf = utai_lib::audio::load_audio(&src_path).unwrap();
    let mono = utai_lib::audio::resample::to_mono(&buf);
    let sr = mono.sample_rate;
    let hop = (sr / 100) as usize; // rmvpe = 100 fps
    assert_eq!(sr % 100, 0, "hop must divide sr");
    let src_f0 = f0_of(&k, &mono.samples, sr);

    let mut failures: Vec<String> = Vec::new();
    for st in [3.0f32, -3.0, 5.0, -5.0] {
        let factor = 2f32.powf(st / 12.0);
        let ratio = vec![factor; src_f0.len()];
        let ours = psola_shift(&mono.samples, sr, &src_f0, &ratio, PsolaParams { hop });
        assert_eq!(ours.len(), mono.samples.len(), "length must be preserved");
        let ours_f0 = f0_of(&k, &ours, sr);
        let m = gate_metrics(&src_f0, &ours_f0, st);

        // praat reference (if generated) measured with the SAME stick
        let tag = format!("{}{}", if st > 0.0 { "p" } else { "m" }, st.abs() as i32);
        let praat_path = ab_dir().join(format!("praat_{tag}.wav"));
        let praat = praat_path.is_file().then(|| {
            let pb = utai_lib::audio::load_audio(&praat_path).unwrap();
            let pm = utai_lib::audio::resample::to_mono(&pb);
            let pf0 = f0_of(&k, &pm.samples, pm.sample_rate);
            gate_metrics(&src_f0, &pf0, st)
        });

        match &praat {
            Some(p) => eprintln!(
                "shift {st:+.0}st  ours: med {:.1}¢ p90 {:.1}¢ cov {:.2}   praat: med {:.1}¢ p90 {:.1}¢ cov {:.2}",
                m.median, m.p90, m.coverage, p.median, p.p90, p.coverage
            ),
            None => eprintln!(
                "shift {st:+.0}st  ours: med {:.1}¢ p90 {:.1}¢ cov {:.2}   (no praat ref)",
                m.median, m.p90, m.coverage
            ),
        }

        if st.abs() <= 3.0 {
            if !(m.median <= 30.0 && m.p90 <= 100.0) {
                failures.push(format!("{st:+.0}st: med {:.1} p90 {:.1} breaks the S41 gate", m.median, m.p90));
            }
        }
        if m.coverage < 0.85 {
            failures.push(format!("{st:+.0}st: voiced coverage {:.2} < 0.85", m.coverage));
        }
        if let Some(p) = praat {
            if m.median > p.median * 1.6 + 5.0 || m.p90 > p.p90 * 1.6 + 10.0 {
                failures.push(format!(
                    "{st:+.0}st: ours(med {:.1}, p90 {:.1}) worse than 1.6×praat(med {:.1}, p90 {:.1})",
                    m.median, m.p90, p.median, p.p90
                ));
            }
        }

        // keep the artifacts for listening
        let out_buf = utai_lib::audio::AudioBuffer { samples: ours, sample_rate: sr, channels: 1 };
        let _ = utai_lib::audio::save_wav_f32(&ab_dir().join(format!("ours_{tag}.wav")), &out_buf);
    }
    assert!(failures.is_empty(), "PSOLA gate failures:\n{}", failures.join("\n"));
}
