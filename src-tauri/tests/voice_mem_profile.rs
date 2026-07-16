//! Memory-profile harness for the RVC cover pipeline (S67b: community report —
//! 16 GB machine, DirectML build, aux on CPU, 264.5 s song → silent crash at 20%
//! progress = right after the whole-song RMVPE pass, during the first chunk's
//! ContentVec/net_g). This test reproduces the workload shape and prints a
//! per-stage peak-commit/working-set profile, so the actual memory hog is
//! MEASURED, not theorized (feedback_silent_regressions: instrument, don't guess).
//!
//! Not a gate — diagnostic only, `--ignored`. Run (PowerShell, from src-tauri):
//!   $env:UTAI_MEM_INPUT="D:\MyDev\TESTING\ikanaiteyo\vocal.wav"   # tiled to UTAI_MEM_SECONDS
//!   $env:UTAI_MEM_SECONDS="264.5"
//!   $env:UTAI_MEM_DEVICE="directml"   # net_g EP: directml | auto | cpu (aux ALWAYS forced CPU)
//!   cargo test --test voice_mem_profile rvc_mem_profile -- --ignored --nocapture
//! Continuous 20 ms samples land in UTAI_MEM_CSV (default scratch csv next to the input).

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use utai_lib::inference::engine::{DeviceConfig, OnnxEngine};
use utai_lib::inference::RvcOptions;

fn app_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

/// Same init as voice_pipeline.rs (the S66 rule: bare harnesses must init ORT or they
/// hang forever at 0 CPU on the invisible modal DLL dialog).
fn init_ort() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("utai_lib=info")),
        )
        .try_init();
    utai_lib::suppress_windows_dll_error_dialogs();
    utai_lib::setup_cuda_dll_paths(&app_root());
    // UTAI_MEM_ORT_DLL: load a SPECIFIC ORT build (e.g. runtime\ort\onnxruntime.dll = the
    // DirectML build the release ships) — dev boxes with CUDA otherwise auto-pick the CUDA
    // build, which has no DirectML provider (same trick as f0.rs's ignored parity test).
    if let Ok(dll) = std::env::var("UTAI_MEM_ORT_DLL") {
        if let Ok(b) = ort::init_from(&dll) {
            let _ = b.commit();
        }
        eprintln!("[mem] ORT loaded from {dll}");
    } else {
        utai_lib::init_ort_runtime(&app_root());
    }
}

// ── process memory via K32GetProcessMemoryInfo (kernel32; no new crate) ──
#[repr(C)]
#[allow(non_snake_case)]
struct ProcessMemoryCountersEx {
    cb: u32,
    PageFaultCount: u32,
    PeakWorkingSetSize: usize,
    WorkingSetSize: usize,
    QuotaPeakPagedPoolUsage: usize,
    QuotaPagedPoolUsage: usize,
    QuotaPeakNonPagedPoolUsage: usize,
    QuotaNonPagedPoolUsage: usize,
    PagefileUsage: usize,
    PeakPagefileUsage: usize,
    PrivateUsage: usize,
}

#[link(name = "kernel32")]
extern "system" {
    fn K32GetProcessMemoryInfo(h: isize, p: *mut ProcessMemoryCountersEx, cb: u32) -> i32;
    fn GetCurrentProcess() -> isize;
}

/// (private/commit bytes, working set bytes) of this process, or (0,0) on failure.
fn mem_now() -> (usize, usize) {
    unsafe {
        let mut c: ProcessMemoryCountersEx = std::mem::zeroed();
        c.cb = std::mem::size_of::<ProcessMemoryCountersEx>() as u32;
        if K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) != 0 {
            (c.PrivateUsage, c.WorkingSetSize)
        } else {
            (0, 0)
        }
    }
}

const MB: f64 = 1024.0 * 1024.0;

/// Rolling-peak sampler: a 20 ms thread tracks the max commit/WS since the last `mark`,
/// and appends every sample to a CSV for post-hoc timeline inspection.
struct Sampler {
    peak_private: AtomicUsize,
    peak_ws: AtomicUsize,
    stop: AtomicBool,
}

impl Sampler {
    fn mark(&self, t0: Instant, label: &str) {
        let (p, w) = mem_now();
        let pk_p = self.peak_private.swap(p, Ordering::Relaxed);
        let pk_w = self.peak_ws.swap(w, Ordering::Relaxed);
        eprintln!(
            "[mem] t={:8.2}s {:<28} now: commit={:7.1}MB ws={:7.1}MB | peak since last mark: commit={:7.1}MB ws={:7.1}MB",
            t0.elapsed().as_secs_f64(),
            label,
            p as f64 / MB,
            w as f64 / MB,
            pk_p.max(p) as f64 / MB,
            pk_w.max(w) as f64 / MB,
        );
    }
}

/// Start the 20 ms peak-tracking thread behind a Sampler (optionally appending every sample
/// to a CSV buffer). Shared by all probes — without it the "peak since last mark" column
/// silently degrades to two point samples.
fn spawn_peak_thread(sampler: &Arc<Sampler>, t0: Instant, csv: Option<Arc<parking_lot::Mutex<String>>>) {
    let s = sampler.clone();
    std::thread::spawn(move || {
        while !s.stop.load(Ordering::Relaxed) {
            let (p, w) = mem_now();
            s.peak_private.fetch_max(p, Ordering::Relaxed);
            s.peak_ws.fetch_max(w, Ordering::Relaxed);
            if let Some(csv) = &csv {
                csv.lock().push_str(&format!(
                    "{:.3},{:.1},{:.1}\n",
                    t0.elapsed().as_secs_f64(),
                    p as f64 / MB,
                    w as f64 / MB
                ));
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    });
}

/// One net_g run at frame length `t` with deterministic pseudo-audio inputs (contents are
/// irrelevant to allocator behavior — this exercises shape-dependent DML pool allocation
/// only). Shared by dml_shape_growth_probe and msst_then_rvc_probe.
fn run_netg_shape(engine: &OnnxEngine, voice_sid: &str, t: usize) {
    use utai_lib::inference::engine::InputTensor;
    let phone: Vec<f32> = (0..t * 768).map(|i| ((i % 997) as f32) / 997.0 - 0.5).collect();
    let pitch: Vec<i64> = (0..t).map(|i| 60 + (i % 40) as i64).collect();
    let pitchf: Vec<f32> = (0..t).map(|i| 220.0 + (i % 40) as f32).collect();
    let rnd: Vec<f32> = (0..192 * t).map(|i| ((i % 613) as f32) / 613.0 - 0.5).collect();
    let inputs = vec![
        ("phone", InputTensor::F32 { data: phone, shape: vec![1, t as i64, 768] }),
        ("phone_lengths", InputTensor::I64 { data: vec![t as i64], shape: vec![1] }),
        ("pitch", InputTensor::I64 { data: pitch, shape: vec![1, t as i64] }),
        ("pitchf", InputTensor::F32 { data: pitchf, shape: vec![1, t as i64] }),
        ("sid", InputTensor::I64 { data: vec![0], shape: vec![1] }),
        ("rnd", InputTensor::F32 { data: rnd, shape: vec![1, 192, t as i64] }),
    ];
    let out = engine.run(voice_sid, inputs).expect("net_g run");
    assert!(!out.is_empty());
}

/// Mechanism probe for the DML commit growth seen in rvc_mem_profile: run net_g repeatedly
/// through ONE DirectML session — 3× the SAME shape, then a series of NEW shapes, then unload.
/// If commit only grows on new shapes, the growth is the DML EP's per-shape operator
/// re-initialization (dynamic shapes → one compiled-kernel/pool set per distinct T);
/// if it grows every run it's a genuine allocator leak. Run:
///   $env:UTAI_MEM_ORT_DLL="D:\MyDev\Utai_v2-dev\runtime\ort\onnxruntime.dll"
///   cargo test --test voice_mem_profile dml_shape_growth_probe -- --ignored --nocapture
#[test]
#[ignore]
fn dml_shape_growth_probe() {
    init_ort();
    let engine = OnnxEngine::new();
    engine.set_device(DeviceConfig::DirectMl { device_id: 0 });

    let rvc_dir = app_root().join("data").join("models").join("rvc");
    let model = rvc_dir.join("lengv2.3.onnx");
    let voice_sid = engine.load_model_with(&model, false).expect("load voice model");

    let t0 = Instant::now();
    let sampler = Arc::new(Sampler {
        peak_private: AtomicUsize::new(0),
        peak_ws: AtomicUsize::new(0),
        stop: AtomicBool::new(false),
    });
    spawn_peak_thread(&sampler, t0, None);
    sampler.mark(t0, "net_g loaded (DML)");

    let run_t = |t: usize, tag: &str| {
        run_netg_shape(&engine, &voice_sid, t);
        sampler.mark(t0, tag);
    };

    // Same shape 3× — growth here would mean a per-RUN leak.
    for i in 0..3 {
        run_t(4000, &format!("T=4000 run#{}", i + 1));
    }
    // New shapes — growth here = per-SHAPE kernel/pool accumulation.
    for t in [4100usize, 4200, 4300, 4400, 4500] {
        run_t(t, &format!("T={t} (new shape)"));
    }
    // Same first shape again — a shape CACHE would stay flat here.
    run_t(4000, "T=4000 again");

    engine.unload_model(&voice_sid);
    sampler.mark(t0, "session unloaded");
    std::thread::sleep(std::time::Duration::from_millis(300));
    sampler.mark(t0, "after settle");
    sampler.stop.store(true, Ordering::Relaxed);
}

/// S67c probe: first-shape TICKET SIZE as a function of chunk length T. Each T gets a
/// FRESH session (load → first run → unload; unload returns everything, so consecutive
/// measurements don't pollute each other) and the commit delta across the first run IS
/// that T's ticket. Answers "how much does the upstream ≤4GB tier (x_max 41→32) or an
/// even shorter chunk actually shave off?" — pool sizes jump in allocation buckets
/// (S67b: +32MB…+3.1GB steps), so the curve must be measured, not assumed linear.
///   $env:UTAI_MEM_ORT_DLL="D:\MyDev\Utai_v2-dev\runtime\ort\onnxruntime.dll"
///   cargo test --test voice_mem_profile ticket_size_by_t -- --ignored --nocapture
#[test]
#[ignore]
fn ticket_size_by_t() {
    init_ort();
    let engine = OnnxEngine::new();
    engine.set_device(DeviceConfig::DirectMl { device_id: 0 });
    let model = app_root().join("data").join("models").join("rvc").join("lengv2.3.onnx");

    let t0 = Instant::now();
    let sampler = Arc::new(Sampler {
        peak_private: AtomicUsize::new(0),
        peak_ws: AtomicUsize::new(0),
        stop: AtomicBool::new(false),
    });
    spawn_peak_thread(&sampler, t0, None);

    // T = net_g frame count at 100 fps. Chunk seconds ≈ x_max + 2*x_pad:
    //   upstream fp32/5G tier (ours): 41+2 = 43 s → T≈4300
    //   upstream ≤4GB tier:           32+2 = 34 s → T≈3400
    //   hypothetical shorter tiers:   ~19 s → T≈1900, ~10 s → T≈1000
    for t in [4300usize, 3400, 1900, 1000] {
        let sid = engine.load_model_with(&model, false).expect("load net_g");
        let before = {
            let (p, _) = mem_now();
            p
        };
        run_netg_shape(&engine, &sid, t);
        let (after, _) = mem_now();
        eprintln!(
            "[ticket] T={t:5} (~{:.0}s chunk): first-run ticket = {:.0} MB",
            t as f64 / 100.0,
            (after.saturating_sub(before)) as f64 / MB
        );
        engine.unload_model(&sid);
        std::thread::sleep(std::time::Duration::from_millis(400));
        sampler.mark(t0, &format!("T={t} unloaded"));
    }
    sampler.stop.store(true, Ordering::Relaxed);
}

/// S67c probe (community follow-up): the 6-node workflow — 3 MSST separations, each DML
/// session released before the next loads — STILL crashes a 16 GB DirectML machine at the
/// FIRST RVC chunk (growth accounting can't fire there: first-shape ticket is excluded),
/// while the same chain with only ONE MSST model before RVC survives. This mirrors that
/// exact engine-call sequence on the dev box and measures where the commit goes:
///   round r: release_others(model) → NativePipeline::new → separate(~60 s) → drop stems
///   then:    release_gpu_sessions_except([rvc]) → settle → net_g load → FIRST-chunk run
/// Compare UTAI_PROBE_MSST_ROUNDS=0 (control) vs 1 vs 3: if the ticket/commit profile at
/// the net_g stage differs, the MSST prehistory is the mechanism; if identical, the crash
/// is system-level on the reporter's machine (VRAM spill / total-commit pressure), not a
/// process-commit leak. Third round reuses the karaoke model (denoise isn't on the dev
/// box) — a released model reloads as a brand-new session incarnation, so the ticket
/// mechanics are equivalent.
/// Run (PowerShell, from src-tauri):
///   $env:UTAI_MEM_ORT_DLL="D:\MyDev\Utai_v2-dev\runtime\ort\onnxruntime.dll"
///   $env:UTAI_PROBE_MSST_ROUNDS="3"
///   cargo test --test voice_mem_profile msst_then_rvc_probe -- --ignored --nocapture
#[test]
#[ignore]
fn msst_then_rvc_probe() {
    init_ort();
    let engine = OnnxEngine::new();
    // The reporter's Auto resolves to DirectML (no CUDA DLLs on their box); pin DML directly.
    engine.set_device(DeviceConfig::DirectMl { device_id: 0 });

    let rounds: usize = std::env::var("UTAI_PROBE_MSST_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let input = PathBuf::from(
        std::env::var("UTAI_PROBE_INPUT")
            .unwrap_or_else(|_| "D:\\MyDev\\TESTING\\ikanaiteyo\\vocal.wav".to_string()),
    );
    let seconds: f64 = std::env::var("UTAI_PROBE_SECONDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60.0);
    let csv_path = std::env::var("UTAI_MEM_CSV")
        .map(PathBuf::from)
        .unwrap_or_else(|_| input.with_file_name(format!("msst_rvc_probe_r{rounds}.csv")));

    let t0 = Instant::now();
    let sampler = Arc::new(Sampler {
        peak_private: AtomicUsize::new(0),
        peak_ws: AtomicUsize::new(0),
        stop: AtomicBool::new(false),
    });
    let csv = Arc::new(parking_lot::Mutex::new(String::from("t_s,commit_mb,ws_mb\n")));
    spawn_peak_thread(&sampler, t0, Some(csv.clone()));
    sampler.mark(t0, "start");

    // ── separation input: load + tile to `seconds` (mono duplicates to stereo in separate()) ──
    use utai_lib::separation::pipeline::{load_wav, AudioData, NativePipeline};
    let src = load_wav(&input).expect("load separation input");
    let frames_target = (seconds * src.sample_rate as f64) as usize;
    let tile = |ch: &[f32]| -> Vec<f32> {
        let mut v = Vec::with_capacity(frames_target);
        while v.len() < frames_target {
            let take = (frames_target - v.len()).min(ch.len());
            v.extend_from_slice(&ch[..take]);
        }
        v
    };
    let audio = AudioData {
        left: tile(&src.left),
        right: tile(if src.right.is_empty() { &src.left } else { &src.right }),
        channels: 2,
        sample_rate: src.sample_rate,
    };
    sampler.mark(t0, "input tiled");

    // ── MSST rounds, mirroring the tester's 0.3.1 chain (karaoke → dereverb → denoise; the
    //    third slot reuses karaoke as a stand-in for the missing denoise model) ──
    let msst_dir = app_root().join("data").join("models").join("msst");
    let models = [
        msst_dir.join("mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.fp16.onnx"),
        msst_dir.join("dereverb_mel_band_roformer_anvuew_sdr_19.1729.fp16.onnx"),
        msst_dir.join("mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.fp16.onnx"),
    ];
    for r in 0..rounds {
        let path = &models[r % models.len()];
        // separation/mod.rs:179 — evict everything else BEFORE the big load, same as the app.
        engine.release_others(path);
        sampler.mark(t0, &format!("round {}: released others", r + 1));
        let pipeline = NativePipeline::new(&engine, path).expect("msst pipeline");
        sampler.mark(t0, &format!("round {}: session built", r + 1));
        let stems = pipeline.separate(&audio, &|_| true).expect("separate");
        sampler.mark(t0, &format!("round {}: separated ({} stems)", r + 1, stems.len()));
        drop(stems);
        sampler.mark(t0, &format!("round {}: stems dropped", r + 1));
    }

    // ── voice run entry, mirroring commands/inference.rs run_rvc ──
    let rvc_dir = app_root().join("data").join("models").join("rvc");
    let model = rvc_dir.join("lengv2.3.onnx");
    engine.release_gpu_sessions_except(&[model.clone()]);
    sampler.mark(t0, "voice: MSST sessions released");
    std::thread::sleep(std::time::Duration::from_millis(1000));
    sampler.mark(t0, "voice: after 1 s settle");

    let voice_sid = engine.load_model_with(&model, false).expect("load net_g");
    sampler.mark(t0, "net_g loaded (DML)");
    // First-chunk ticket at the tester's scale: 264.2 s / 7 chunks ≈ 37.7 s @16 k → T≈1885.
    run_netg_shape(&engine, &voice_sid, 1885);
    sampler.mark(t0, "net_g FIRST chunk (ticket)");
    run_netg_shape(&engine, &voice_sid, 1930);
    sampler.mark(t0, "net_g second shape");

    engine.unload_model(&voice_sid);
    std::thread::sleep(std::time::Duration::from_millis(500));
    sampler.mark(t0, "net_g unloaded + settle");
    sampler.stop.store(true, Ordering::Relaxed);
    std::thread::sleep(std::time::Duration::from_millis(60));
    std::fs::write(&csv_path, csv.lock().as_str()).expect("write csv");
    eprintln!("[mem] rounds={rounds} | csv: {}", csv_path.display());
}

#[test]
#[ignore]
fn rvc_mem_profile() {
    let input = PathBuf::from(
        std::env::var("UTAI_MEM_INPUT")
            .unwrap_or_else(|_| "D:\\MyDev\\TESTING\\ikanaiteyo\\vocal.wav".to_string()),
    );
    let seconds: f64 = std::env::var("UTAI_MEM_SECONDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(264.5);
    let device = std::env::var("UTAI_MEM_DEVICE").unwrap_or_else(|_| "directml".to_string());
    let csv_path = std::env::var("UTAI_MEM_CSV")
        .map(PathBuf::from)
        .unwrap_or_else(|_| input.with_file_name("rvc_mem_profile.csv"));

    init_ort();
    let engine = OnnxEngine::new();
    match device.as_str() {
        "cpu" => engine.set_device(DeviceConfig::Cpu),
        "auto" => engine.set_device(DeviceConfig::Auto),
        "directml" => engine.set_device(DeviceConfig::DirectMl { device_id: 0 }),
        other => panic!("UTAI_MEM_DEVICE must be directml|auto|cpu (got {other})"),
    }

    let t0 = Instant::now();
    let sampler = Arc::new(Sampler {
        peak_private: AtomicUsize::new(0),
        peak_ws: AtomicUsize::new(0),
        stop: AtomicBool::new(false),
    });
    let csv = Arc::new(parking_lot::Mutex::new(String::from("t_s,commit_mb,ws_mb\n")));
    spawn_peak_thread(&sampler, t0, Some(csv.clone()));
    sampler.mark(t0, "start");

    // ── input: tile the real vocal to the target duration (user: 264.5 s separated vocal) ──
    let src = utai_lib::audio::load_audio(&input).expect("load input wav");
    let frames_target = (seconds * src.sample_rate as f64) as usize;
    let ch = src.channels.max(1) as usize;
    let mut samples = Vec::with_capacity(frames_target * ch);
    while samples.len() < frames_target * ch {
        let take = (frames_target * ch - samples.len()).min(src.samples.len());
        samples.extend_from_slice(&src.samples[..take]);
    }
    let audio = utai_lib::audio::AudioBuffer {
        samples,
        sample_rate: src.sample_rate,
        channels: src.channels,
    };
    eprintln!(
        "[mem] input: {:.1}s x{}ch @{}Hz (tiled from {})",
        seconds,
        ch,
        audio.sample_rate,
        input.display()
    );
    sampler.mark(t0, "input tiled");

    // ── models: mirror run_rvc (aux forced CPU = gpu_extract off, like the reporter) ──
    let aux = app_root().join("data").join("models").join(utai_lib::models::AUX_DIR_NAME);
    let rvc_dir = app_root().join("data").join("models").join("rvc");
    let model = rvc_dir.join("lengv2.3.onnx");
    let sc: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(model.with_extension("json")).expect("sidecar"),
    )
    .expect("sidecar json");
    let dim = sc["features_dim"].as_u64().expect("features_dim") as usize;
    let sample_rate = sc["sample_rate"].as_u64().expect("sample_rate") as u32;
    let nch = sc["noise"]["rnd_input"][1].as_u64().unwrap_or(192) as usize;
    let min_frames = sc["min_frames"].as_u64().unwrap_or(12) as usize;

    let voice_sid = engine.load_model_with(&model, false).expect("load voice model");
    sampler.mark(t0, "net_g loaded");
    let cv_sid = engine
        .load_model_on(
            &aux.join(if dim == 768 { "contentvec_768l12.onnx" } else { "contentvec_256l9.onnx" }),
            false,
            DeviceConfig::Cpu,
        )
        .expect("load contentvec");
    sampler.mark(t0, "contentvec loaded (CPU)");
    let rmvpe_sid = engine
        .load_model_on(&aux.join("rmvpe_e2e.onnx"), false, DeviceConfig::Cpu)
        .expect("load rmvpe");
    sampler.mark(t0, "rmvpe loaded (CPU)");
    let mel: ndarray::Array2<f32> =
        ndarray_npy::read_npy(aux.join("rmvpe_mel_filters.npy")).expect("mel filters");
    let index =
        utai_lib::inference::rvc::RvcIndex::load(&rvc_dir.join("lengv2.3.npy")).expect("index");
    sampler.mark(t0, "index loaded");

    let m = utai_lib::inference::rvc::RvcModel {
        engine: &engine,
        voice_session: &voice_sid,
        contentvec_session: &cv_sid,
        rmvpe_session: &rmvpe_sid,
        mel_filters: &mel,
        index: Some(&index),
        sample_rate,
        features_dim: dim,
        spk_mix: None,
        noise_channels: nch,
        min_frames,
    };
    let options = RvcOptions::default(); // index_ratio 0.75 etc. — the shipped defaults

    // Stage marks ride the pipeline's own progress values: 0.03 = input prepped,
    // 0.2 = whole-song RMVPE done, then one call per completed chunk.
    let s2 = sampler.clone();
    let progress = move |p: f32| {
        let label = if p <= 0.031 {
            "p=0.03 resample+filtfilt".to_string()
        } else if (p - 0.2).abs() < 1e-4 {
            "p=0.20 f0 (RMVPE) done".to_string()
        } else if p >= 1.0 {
            "p=1.00 pipeline done".to_string()
        } else {
            format!("p={p:.3} chunk done")
        };
        s2.mark(t0, &label);
    };
    let result =
        utai_lib::inference::rvc::run_pipeline(&m, &audio, &options, None, &progress, &|| false)
            .expect("rvc pipeline");

    sampler.mark(t0, "returned");
    sampler.stop.store(true, Ordering::Relaxed);
    std::thread::sleep(std::time::Duration::from_millis(60));
    std::fs::write(&csv_path, csv.lock().as_str()).expect("write csv");
    eprintln!(
        "[mem] out: n={} sr={} | csv: {}",
        result.audio.len(),
        result.sample_rate,
        csv_path.display()
    );
}
