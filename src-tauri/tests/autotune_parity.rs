//! ② 自动音高调教 parity gate(S73)——Rust 特征构造 + ONNX θ 对拍 SVC2SVS Python 真值。
//!
//! 夹具 = tests/fixtures/autotune_parity.json,由 SVC2SVS `pitch/export_onnx.py` 生成
//! (E1 verse/chorus 实谱 + 合成 snap 边界序列;期望值 = python build_note_arrays/
//! note_features + ORT θ)。改 SVC2SVS pitch/dataset.py 特征口径必须重导夹具再跑本 gate。
//!
//! - `feature_parity`:纯数学(吸附/贴合旗/12 维特征),常规套件常跑。
//! - `theta_parity_onnx`:#[ignore],需本机 data/models/auxiliary/autotune_a1.onnx + ORT;
//!   跑法:cargo test --test autotune_parity -- --ignored --nocapture

use std::path::PathBuf;

use utai_lib::inference::autotune::{self, NoteIn};

fn app_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

#[derive(serde::Deserialize)]
struct Fixture {
    model: String,
    cases: Vec<Case>,
}

// ★输入与精确期望走 f64 位模式(hex):serde_json 默认浮点解析 best-effort 非正确舍入
// (1 ulp 抖动实测撞过),parity gate 不容解析器噪声;十进制字段仅供人读。
#[derive(serde::Deserialize)]
struct Case {
    name: String,
    notes: Vec<FxNote>,
    snapped_tick_bits: Vec<String>,
    snapped_dur_bits: Vec<String>,
    abut_prev: Vec<bool>,
    feats: Vec<Vec<f64>>,
    theta_t: Vec<Vec<f64>>,
    theta_v: Vec<Vec<f64>>,
}

#[derive(serde::Deserialize)]
struct FxNote {
    start_bits: String,
    dur_bits: String,
    pitch_bits: String,
}

fn from_bits_hex(s: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(s, 16).expect("bad bits hex"))
}

fn load_fixture() -> Fixture {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("autotune_parity.json");
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| {
        panic!("读不到夹具 {p:?}: {e}(由 SVC2SVS pitch/export_onnx.py 生成)")
    }))
    .expect("夹具 JSON 解析失败")
}

fn case_notes(c: &Case) -> Vec<NoteIn> {
    c.notes
        .iter()
        .map(|n| NoteIn {
            start_ms: from_bits_hex(&n.start_bits),
            dur_ms: from_bits_hex(&n.dur_bits),
            pitch: from_bits_hex(&n.pitch_bits),
        })
        .collect()
}

#[test]
fn feature_parity() {
    let fx = load_fixture();
    assert!(!fx.cases.is_empty());
    for c in &fx.cases {
        let a = autotune::build_note_arrays(&case_notes(c));
        let tick_bits: Vec<String> = a.tick.iter().map(|x| format!("{:016x}", x.to_bits())).collect();
        let dur_bits: Vec<String> = a.dur.iter().map(|x| format!("{:016x}", x.to_bits())).collect();
        assert_eq!(tick_bits, c.snapped_tick_bits, "{}: snapped tick 漂移", c.name);
        assert_eq!(dur_bits, c.snapped_dur_bits, "{}: snapped dur 漂移", c.name);
        assert_eq!(a.abut_prev, c.abut_prev, "{}: abut_prev 漂移", c.name);
        let feats = autotune::note_features(&a);
        let n = a.tick.len();
        assert_eq!(feats.len(), n * autotune::N_FEATS);
        for i in 0..n {
            for j in 0..autotune::N_FEATS {
                let got = feats[i * autotune::N_FEATS + j] as f64;
                let want = c.feats[i][j];
                assert!(
                    (got - want).abs() <= 1e-6,
                    "{}: feats[{i}][{j}] {got} != {want}",
                    c.name
                );
            }
        }
    }
}

#[test]
#[ignore]
fn theta_parity_onnx() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("utai_lib=warn")),
        )
        .try_init();
    utai_lib::suppress_windows_dll_error_dialogs();
    utai_lib::init_ort_runtime(&app_root());

    let fx = load_fixture();
    let model_path = app_root().join("data").join("models").join("auxiliary").join(&fx.model);
    assert!(model_path.exists(), "缺 {model_path:?}(pitch/export_onnx.py 产物拷入 auxiliary)");
    let engine = utai_lib::inference::engine::OnnxEngine::new();
    let sid = engine
        .load_model_on(&model_path, false, utai_lib::inference::engine::DeviceConfig::Cpu)
        .expect("load autotune onnx");

    let mut worst = 0.0f64;
    for c in &fx.cases {
        let notes = case_notes(c);
        // 夹具期望值是 python 单窗推理——乐句切分不得改变口径(E1 最大休止 612ms < 2s,合成
        // 序列 gap ≤ 250ms;若未来夹具加入含长休止 case,这里会响亮失败提醒分窗对拍)
        let arrays = autotune::build_note_arrays(&notes);
        assert_eq!(
            autotune::chunk_ranges(&arrays).len(),
            1,
            "{}: 夹具 case 应为单 chunk",
            c.name
        );
        let thetas = autotune::run_autotune_model(&engine, &sid, &notes).expect("run model");
        assert_eq!(thetas.len(), notes.len());
        for (i, th) in thetas.iter().enumerate() {
            for j in 0..6 {
                let dt = (th.transition[j] - c.theta_t[i][j]).abs();
                let dv = (th.vibrato[j] - c.theta_v[i][j]).abs();
                worst = worst.max(dt).max(dv);
                assert!(dt <= 1e-3, "{}: theta_t[{i}][{j}] |d|={dt}", c.name);
                assert!(dv <= 1e-3, "{}: theta_v[{i}][{j}] |d|={dv}", c.name);
            }
        }
    }
    eprintln!("[autotune parity] worst |d| = {worst:.2e} (tol 1e-3), {} cases", fx.cases.len());
}
