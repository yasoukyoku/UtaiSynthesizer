//! GAME 离线批量打谱 harness(SVC2SVS 旋钮线第二刀)— 诊断工具,不是 gate。
//!
//! 目的:对 MBS2H 最终训练集的 utterance wav 批量跑生产 GAME 管线(midi_extract::extract_notes,
//! 零复制铁律:绝不在 Python 重写 GAME),音符 JSON 落盘给 SVC2SVS 的 build_labels.py 消费。
//! 官方管线非确定(D3PM×ORT 线程归约)→ 标注**跑一次固定落盘**,已存在输出直接跳过(断点续跑,
//! 重打需先删旧 JSON)。CPU 钉死(S60 口径,~7.5× 实时,零显存竞争);权重 CC BY-NC-SA,
//! 产物只进本地训练数据,不进任何分发物。
//!
//! 任务单(SVC2SVS pitch/dataprep/build_game_tasks.py 生成):
//!   JSON 数组 [{"id": "...", "wav": "<绝对路径>", "out": "<绝对路径>.json"}, ...]
//! 运行(可多进程分片并行,shard 按 idx % n):
//!   $env:UTAI_GAME_TASKS='D:\MyDev\SVC2SVS\labels\game_tasks.json'
//!   $env:UTAI_GAME_SHARD='0/4'   # 可选,缺省 0/1=全量
//!   cargo test --release --test game_batch -- --ignored --nocapture
use std::io::Write as _;
use std::path::{Path, PathBuf};

use utai_lib::inference::midi_extract;

fn app_root() -> PathBuf {
    // tests run from src-tauri; the models dir is the dev-checkout data/models (game_parity 同款)
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

#[derive(serde::Deserialize)]
struct Task {
    id: String,
    wav: String,
    out: String,
}

#[test]
#[ignore]
fn game_batch_label() {
    let tasks_path = match std::env::var("UTAI_GAME_TASKS") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            eprintln!("skip: UTAI_GAME_TASKS not set");
            return;
        }
    };
    let shard = std::env::var("UTAI_GAME_SHARD").unwrap_or_else(|_| "0/1".into());
    let (si, sn) = shard
        .split_once('/')
        .map(|(a, b)| (a.parse::<usize>().unwrap(), b.parse::<usize>().unwrap()))
        .expect("UTAI_GAME_SHARD 形如 0/4");
    assert!(si < sn, "shard index 必须 < shard count");

    // ORT 初始化 = game_parity 同款(aux 走 CPU;测试进程无 GPU 全局设备设置)
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("utai_lib=warn")),
        )
        .try_init();
    utai_lib::suppress_windows_dll_error_dialogs();
    // S74: MUST precede init_ort_runtime and MUST be here even though the default tier is CPU —
    // UTAI_GAME_DEVICE=cuda:N makes this harness a GPU harness, and without the app's CUDA DLL
    // directories the cudnn 9 shim can't resolve its engine sub-DLLs (CUDNN_BACKEND_API_FAILED at
    // the first Conv). Diagnosing GAME-on-CUDA in a loader context the app never uses is the S39
    // trap; every other GPU-capable harness already calls this.
    // UTAI_GAME_DLL_MODE=nopath — S74 regression hook for the cuDNN-frontend fix. It SKIPS
    // setup_cuda_dll_paths (so runtime/cuda is NOT on PATH) and only registers the dir via
    // AddDllDirectory, which is inert unless the process is in user-dirs mode. That is the
    // loader context in which cuDNN could not find cudnn_engines_tensor_ir64_9.dll and GAME's
    // first Conv died with CUDNN_FE failure 11; with lib.rs's absolute-path preload of the
    // libraries the ort crate misses, `nopath` must now transcribe on CUDA exactly like `full`.
    // Delete this hook only together with that preload.
    if std::env::var("UTAI_GAME_DLL_MODE").as_deref() == Ok("nopath") {
        extern "system" {
            fn AddDllDirectory(new_directory: *const u16) -> *mut std::ffi::c_void;
        }
        let d = app_root().join("runtime").join("cuda");
        let wide: Vec<u16> = d.to_string_lossy().encode_utf16().chain(std::iter::once(0)).collect();
        unsafe { AddDllDirectory(wide.as_ptr()) };
        eprintln!("[game_batch] DLL mode = nopath (AddDllDirectory only)");
    } else {
        utai_lib::setup_cuda_dll_paths(&app_root());
    }
    utai_lib::init_ort_runtime(&app_root());

    let models_dir = app_root().join("data").join("models");
    assert!(
        midi_extract::game_installed(&models_dir),
        "GAME 未安装于 {models_dir:?}(应用内先下载)"
    );

    let tasks: Vec<Task> =
        serde_json::from_str(&std::fs::read_to_string(&tasks_path).unwrap()).unwrap();
    let mine: Vec<&Task> = tasks
        .iter()
        .enumerate()
        .filter(|(i, _)| i % sn == si)
        .map(|(_, t)| t)
        .collect();
    eprintln!("[game_batch] shard {shard}: {}/{} tasks", mine.len(), tasks.len());

    let engine = utai_lib::inference::engine::OnnxEngine::new();
    // 默认钉 CPU:S71 实测本机 CUDA 跑 GAME 在 cudnn frontend 构图即炸(encoder conv,
    // CUDNN_FE failure 11)→「先试全局设备」政策会让每任务白付 GPU 会话构建+失败+重建。
    // UTAI_GAME_DEVICE=cuda:<id> 供 GPU 试探(⚠extract_notes 对 GPU 运行期失败会静默退
    // CPU 重试——判断 GPU 是否真通要看 stderr 有无「retrying once on CPU」告警+速度)。
    let dev = std::env::var("UTAI_GAME_DEVICE").unwrap_or_else(|_| "cpu".into());
    let cfg = match dev.strip_prefix("cuda:") {
        Some(id) => utai_lib::inference::engine::DeviceConfig::Cuda { device_id: id.parse().expect("UTAI_GAME_DEVICE 形如 cuda:0") },
        None => utai_lib::inference::engine::DeviceConfig::Cpu,
    };
    eprintln!("[game_batch] device = {dev}");
    engine.set_device(cfg);
    let t0 = std::time::Instant::now();
    let (mut done, mut skipped, mut failed) = (0usize, 0usize, 0usize);
    for (k, t) in mine.iter().enumerate() {
        let out = Path::new(&t.out);
        if out.exists() {
            skipped += 1;
            continue;
        }
        let r = (|| -> Result<usize, String> {
            let buf = utai_lib::audio::load_audio_at_rate(Path::new(&t.wav), 44100)
                .map_err(|e| format!("load: {e}"))?;
            let mut mono = utai_lib::audio::resample::to_mono(&buf).samples;
            utai_lib::audio::sanitize_non_finite(&mut mono);
            let notes =
                midi_extract::extract_notes(&engine, &models_dir, &mono, 0, &|| false, &mut |_| {})?;
            let dump: Vec<serde_json::Value> = notes
                .iter()
                .map(|n| {
                    serde_json::json!({"onset": n.onset_sec, "offset": n.offset_sec, "pitch": n.pitch})
                })
                .collect();
            if let Some(dir) = out.parent() {
                std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
            }
            // 原子写:先 tmp 再 rename(断点续跑靠「文件存在=完整」,半截文件会毒化下游)
            let tmp = out.with_extension("json.tmp");
            let mut f = std::fs::File::create(&tmp).map_err(|e| e.to_string())?;
            f.write_all(
                serde_json::to_string(&serde_json::json!({
                    "id": t.id, "sr": 44100, "n_samples": mono.len(), "notes": dump
                }))
                .unwrap()
                .as_bytes(),
            )
            .map_err(|e| e.to_string())?;
            drop(f);
            std::fs::rename(&tmp, out).map_err(|e| e.to_string())?;
            Ok(notes.len())
        })();
        match r {
            Ok(n_notes) => {
                done += 1;
                if done % 50 == 0 {
                    let el = t0.elapsed().as_secs_f64();
                    eprintln!(
                        "[game_batch] {k}/{} done={done} skip={skipped} fail={failed} ({:.1}s, {:.2}s/任务) last={} notes={n_notes}",
                        mine.len(), el, el / done as f64, t.id
                    );
                }
            }
            Err(e) => {
                failed += 1;
                eprintln!("[game_batch] FAIL {}: {e}", t.id);
            }
        }
    }
    midi_extract::unload_sessions(&engine, &models_dir);
    eprintln!(
        "[game_batch] shard {shard} finished: done={done} skip={skipped} fail={failed} in {:.0}s",
        t0.elapsed().as_secs_f64()
    );
    assert_eq!(failed, 0, "有 {failed} 个任务失败(见上方 FAIL 行)");
}
