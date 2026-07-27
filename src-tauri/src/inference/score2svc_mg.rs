//! S84 鹅妈妈快音段探针(diagnostic,NOT a gate,全 `#[ignore]`)——63 小节快段丢音/粘连调查。
//!
//! 链(三段全生产代码,零复刻):
//!   1. `commands::import::mg_probe::dump_mg_notes`(生产 parse_ust → probe\mg_notes.json)
//!   2. vitest `src\lib\vocal\mgScoreDump.test.ts`(生产 buildVocalScore:triples+默认调教 f0
//!      → probe\mg_score.json;UST 导入后无 pitchDev/无 vibrato = 装机版实际状态,S83)
//!   3. 本文件:
//!      `mg_lane_dump`  = build_arrays_daw 干跑(泳道单一真源)+ 生产 f0 整形三连
//!                        (build_note_hz + zero_voiceless_frames + anchor_voiced_phone_f0)
//!                        + emphasis flag → probe\mg_lane.json(逐 phone 帧分配 + 逐帧 f0)
//!      `mg_render_rvc` = 生产 render_score_rvc 全链(素材同目录 tetoRVC_best,含 index 0.75
//!                        生产默认)按三元组下标切片渲染 → probe\mg_render_{a}_{b}_teto.wav
//! Run(src-tauri 下;CPU EP):
//!   cargo test --lib inference::score2svc::mg_tests::mg_lane_dump -- --ignored --nocapture
//!   $env:UTAI_MG_SLICE='a..b'; cargo test --lib inference::score2svc::mg_tests::mg_render_rvc -- --ignored --nocapture
//! 切片坐标 = mg_lane.json 里 notes[].k(三元组下标);f0 数组按累计帧同窗切,借帧语义与整曲
//! 渲染一致的前提 = 切片首元素是 R(SP lender),分析侧选窗时遵守。
//!
//! S85 音域扩展靶场(mg_render_rvc + mg_render_cover 共用,零生产改动):
//!   UTAI_MG_SHIFT=-7(半音)UTAI_MG_INVERSE=0/1 UTAI_MG_KAPPA=κ UTAI_MG_MODEL/UTAI_MG_INDEXFILE
//!   score 臂:raw=transpose 承载(渲染钉在移调位)/inv=range_shift 承载(生产 Signalsmith 逆变换);
//!   cover 臂:f0_shift 承载 + 探针侧整段 apply_inverse(偏差记档于函数内注释)。

use super::*;
use super::e1_tests::write_wav16;
use super::super::engine::{DeviceConfig, OnnxEngine};
use super::super::rvc;
use super::super::score2cv::{is_nucleus_phone, NoDicts};
use super::super::sovits;
use std::path::Path;
use std::time::Instant;

const WORK: &str = r"D:\MyDev\TESTING\不为人所知的鹅妈妈童谣";

#[derive(serde::Deserialize)]
struct ScoreJson {
    tempo: f64,
    triples: Vec<TripleJson>,
    #[serde(rename = "f0Cents")]
    f0_cents: Vec<f32>,
    #[serde(rename = "f0Voiced")]
    f0_voiced: Vec<u8>,
}
#[derive(serde::Deserialize)]
struct TripleJson {
    lyric: String,
    note_num: i64,
    frames: i64,
    #[allow(dead_code)]
    lang: i64,
}

fn load_score() -> ScoreJson {
    let p = Path::new(WORK).join("probe").join("mg_score.json");
    let s = std::fs::read_to_string(&p).unwrap_or_else(|e| {
        panic!("missing {} ({e}) — run dump_mg_notes then mgScoreDump.test.ts first", p.display())
    });
    serde_json::from_str(&s).unwrap()
}

fn to_evts(triples: &[TripleJson]) -> Vec<ScoreEvt<'_>> {
    triples
        .iter()
        .map(|t| ScoreEvt {
            lyric: &t.lyric,
            note_num: t.note_num,
            frames: t.frames,
            lang: Lang::Ja,
            phoneme_input: None,
        })
        .collect()
}

/// S85 靶场共用 env:UTAI_MG_SHIFT(半音)/ UTAI_MG_INVERSE(0=raw 臂)/ UTAI_MG_KAPPA
/// (默认生产 κ)——语义见 mg_render_rvc 内注释。
fn mg_shift_envs() -> (i64, bool, f32) {
    let shift = std::env::var("UTAI_MG_SHIFT").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let inverse = std::env::var("UTAI_MG_INVERSE").map(|s| s != "0").unwrap_or(true);
    let kappa = std::env::var("UTAI_MG_KAPPA")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(super::super::vocal_range::DEFAULT_FORMANT_KAPPA);
    (shift, inverse, kappa)
}

/// UTAI_MG_MODEL / UTAI_MG_INDEXFILE(默认 teto;索引默认=模型同名 .npy)→ (onnx, npy, 名标)。
/// 探针的 RvcModel 参数钉死 768/48k/v2——换模型时从 sidecar 响亮核对,不许静默错规格。
fn mg_model_envs() -> (std::path::PathBuf, std::path::PathBuf, String) {
    let model = std::env::var("UTAI_MG_MODEL")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| Path::new(WORK).join("tetoRVC_best.onnx"));
    let index = std::env::var("UTAI_MG_INDEXFILE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| model.with_extension("npy"));
    if let Ok(s) = std::fs::read_to_string(model.with_extension("json")) {
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["sample_rate"].as_i64(), Some(48000), "mg probe assumes 48k RVC");
        assert_eq!(v["features_dim"].as_i64(), Some(768), "mg probe assumes 768-dim RVC");
    }
    let stem = model.file_stem().unwrap().to_string_lossy().to_string();
    let mtag = if stem == "tetoRVC_best" { "teto".to_string() } else { stem };
    (model, index, mtag)
}

/// shift 臂的文件名后缀:`_s{shift}[_raw][_k{κ}]`(κ 只在偏离生产默认时标)。
fn mg_shift_tag(shift: i64, inverse: bool, kappa: f32) -> String {
    if shift == 0 {
        return String::new();
    }
    let ktag = if kappa != super::super::vocal_range::DEFAULT_FORMANT_KAPPA {
        format!("_k{kappa}")
    } else {
        String::new()
    };
    format!("_s{shift}{}{ktag}", if inverse { "" } else { "_raw" })
}

#[test]
#[ignore]
fn mg_lane_dump() {
    let sj = load_score();
    let evts = to_evts(&sj.triples);
    let total: i64 = sj.triples.iter().map(|t| t.frames).sum();
    assert_eq!(sj.f0_cents.len() as i64, total, "f0 length vs Σframes");
    let arr = build_arrays_daw(&evts, &NoDicts).unwrap();
    let vf0 = VocalF0 { cents: &sj.f0_cents, voiced: &sj.f0_voiced };
    // 生产 f0 整形三连(render_score_rvc 同款;transpose 0)——hz 即模型真实吃到的 f0,
    // 归零窗/借帧锚定的落点不重算、直接数。
    let mut hz = build_note_hz(&arr, &evts, 0, Some(&vf0));
    zero_voiceless_frames(&mut hz, &arr);
    anchor_voiced_phone_f0(&mut hz, &arr);
    let emph = voiceless_onset_flags(&arr);
    let mut cursor = 0usize;
    let mut phones = Vec::with_capacity(arr.phon.len());
    for i in 0..arr.phon.len() {
        let d = arr.phone_dur[i].max(0) as usize;
        let zeros = hz[cursor..cursor + d].iter().filter(|&&v| v == 0.0).count();
        phones.push(serde_json::json!({
            "i": i, "phone": arr.phon[i], "dur": arr.phone_dur[i], "evt": arr.evt[i],
            "frame0": cursor, "voiceless": is_voiceless_phone(arr.phon[i]),
            "nucleus": is_nucleus_phone(arr.phon[i]), "zero_frames": zeros,
            "emphasis": emph[i],
        }));
        cursor += d;
    }
    assert_eq!(cursor as i64, total, "Σ phone_dur == Σ frames(守恒铁律)");
    let mut cf = 0i64;
    let notes: Vec<_> = sj
        .triples
        .iter()
        .enumerate()
        .map(|(k, t)| {
            let j = serde_json::json!({
                "k": k, "lyric": t.lyric, "note_num": t.note_num, "frames": t.frames, "frame0": cf,
            });
            cf += t.frames;
            j
        })
        .collect();
    let out = Path::new(WORK).join("probe").join("mg_lane.json");
    std::fs::write(
        &out,
        serde_json::to_string(&serde_json::json!({
            "tempo": sj.tempo, "total_frames": total, "notes": notes, "phones": phones,
            "f0_hz": hz,
        }))
        .unwrap(),
    )
    .unwrap();
    eprintln!(
        "[mg] lane dumped: {} phones, {} notes, {} frames -> {}",
        arr.phon.len(),
        sj.triples.len(),
        total,
        out.display()
    );
}

#[test]
#[ignore]
fn mg_render_rvc() {
    let sj = load_score();
    let slice = std::env::var("UTAI_MG_SLICE").unwrap_or_default();
    let (a, b) = if slice.is_empty() {
        (0usize, sj.triples.len())
    } else {
        let (s, e) = slice.split_once("..").expect("UTAI_MG_SLICE=a..b (triple indices)");
        (s.parse().unwrap(), e.parse().unwrap())
    };
    assert!(a < b && b <= sj.triples.len(), "bad slice {a}..{b}");
    let f_start: i64 = sj.triples[..a].iter().map(|t| t.frames).sum();
    let f_end: i64 = sj.triples[..b].iter().map(|t| t.frames).sum();
    let triples = &sj.triples[a..b];
    let cents = &sj.f0_cents[f_start as usize..f_end as usize];
    let voiced = &sj.f0_voiced[f_start as usize..f_end as usize];
    let evts = to_evts(triples);
    let vf0 = VocalF0 { cents, voiced };

    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let dll = root.join("../runtime/ort/onnxruntime.dll");
    assert!(dll.exists(), "ORT dll missing at {}", dll.display());
    if let Ok(bld) = ort::init_from(&dll) {
        let _ = bld.commit();
    }
    let engine = OnnxEngine::new();
    engine.set_device(DeviceConfig::Cpu); // deterministic + no GPU setup in a test

    let aux = root.join("../data/models").join(crate::models::AUX_DIR_NAME);
    let s2cv768 = engine.load_model_with(&aux.join("score2cv_768.onnx"), false).unwrap();
    let cv768 = engine.load_model_with(&aux.join("contentvec_768l12.onnx"), false).unwrap();
    let rmvpe = engine.load_model_with(&aux.join("rmvpe_e2e.onnx"), false).unwrap();
    let rmvpe_mel: Array2<f32> = ndarray_npy::read_npy(&aux.join("rmvpe_mel_filters.npy")).unwrap();
    let (model_path, index_path, mtag) = mg_model_envs();
    let teto = engine.load_model_with(&model_path, false).unwrap();
    let index = rvc::RvcIndex::load(&index_path).unwrap();
    let m = rvc::RvcModel {
        engine: &engine,
        voice_session: &teto,
        contentvec_session: &cv768,
        rmvpe_session: &rmvpe,
        mel_filters: &rmvpe_mel,
        index: Some(&index),
        sample_rate: 48000,
        features_dim: 768,
        spk_mix: None,
        noise_channels: 192,
        min_frames: 12,
    };
    // 生产默认口径 = RvcOptions::default()(index 0.75/protect 0.33/rms_mix 0.25——score 路径
    // 命令层会中和 rms_mix,这里 render_score_rvc 内部同款)、emphasis 默认旋钮值、
    // transpose/range_shift 0(63 小节症状与扩展开关无关=S83 用户控制变量实验)。
    // ⚠首轮探针曾用 protect=0.5(=禁用 blend,偏离生产 0.33,skeptic3 抓的)——已改默认跟生产。
    // 消融 env(S84 验证轮:emphasis 与 voiceless/归零在快段完全共线,需消融分离):
    //   UTAI_MG_EMPH=0 关强调;UTAI_MG_INDEX=0 关检索;UTAI_MG_PROTECT=0.5 禁 protect blend。
    let emph: f32 = std::env::var("UTAI_MG_EMPH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_VOICELESS_ONSET_EMPHASIS_DB);
    let idx_ratio: f32 =
        std::env::var("UTAI_MG_INDEX").ok().and_then(|s| s.parse().ok()).unwrap_or(0.75);
    let protect: f32 = std::env::var("UTAI_MG_PROTECT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| RvcOptions::default().protect);
    let valley: f32 = std::env::var("UTAI_MG_VALLEY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_CONSONANT_VALLEY_SCALE);
    let clarity: bool = std::env::var("UTAI_MG_CLARITY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(true);
    // S85 音域扩展靶场(零生产改动,只换调用参数):
    //   UTAI_MG_SHIFT=半音(负=向下入舒适区;0=S84 原行为)。
    //   UTAI_MG_INVERSE=0 → raw 臂:shift 走 transpose 承载(渲染钉在移调位、无逆变换,
    //     听「模型在 shift 位唱得如何」);默认 1 → 生产等价臂:shift 走 range_shift
    //     (render_score_rvc 内部 Signalsmith 逆变换,inverse→peak-norm 顺序=生产)。
    //   UTAI_MG_KAPPA=κ(默认生产 DEFAULT_FORMANT_KAPPA=0;1=跳过 formant 机器)。
    //   UTAI_MG_MODEL=onnx 路径(默认 teto;yachiyo 同规格 768/48k/v2,sidecar 已核),
    //   UTAI_MG_INDEXFILE=npy(默认 = 模型同名 .npy)。
    let (shift, inverse, kappa) = mg_shift_envs();
    let (tp, rs) = if inverse { (0, shift) } else { (shift, 0) };
    let ropts = RvcOptions {
        seed: 0,
        index_ratio: idx_ratio,
        protect,
        range_formant_follow: kappa,
        ..Default::default()
    };
    let no_cancel = || false;
    let no_prog = |_: f32| {};
    let t0 = Instant::now();
    let r = render_score_rvc(
        &m, &s2cv768, &evts, 768, 49, &NoDicts, &ropts, emph, valley, clarity, tp, rs,
        Some(&vf0), None, None, &no_cancel, &no_prog,
    )
    .unwrap();
    let out_dir = Path::new(WORK).join("probe");
    std::fs::create_dir_all(&out_dir).unwrap();
    let tag = format!(
        "{}{}{}{}{}{}",
        if emph != DEFAULT_VOICELESS_ONSET_EMPHASIS_DB { format!("_e{emph}") } else { String::new() },
        if idx_ratio != 0.75 { format!("_i{idx_ratio}") } else { String::new() },
        if protect != RvcOptions::default().protect { format!("_p{protect}") } else { String::new() },
        if valley != DEFAULT_CONSONANT_VALLEY_SCALE { format!("_v{valley}") } else { String::new() },
        if !clarity { "_nc" } else { "" },
        mg_shift_tag(shift, inverse, kappa),
    );
    let name = format!("mg_render_{a}_{b}_{mtag}{tag}.wav");
    write_wav16(&out_dir.join(&name), &r.audio, r.sample_rate);
    eprintln!(
        "[mg] rendered triples[{a}..{b}] ({} frames): {:.2}s audio in {:.1}s wall -> probe\\{name}",
        f_end - f_start,
        r.audio.len() as f32 / r.sample_rate as f32,
        t0.elapsed().as_secs_f64()
    );
}

/// S85 SoVITS score 臂(sidecar 驱动配置,e1 姿势;东雪莲 4.1 / chika_v2 v2 实测目标):
/// UTAI_MG_MODEL=sovits onnx 必填;UTAI_MG_SLICE/SHIFT/INVERSE/KAPPA 语义同 RVC 臂;
/// cluster/diffusion/vocoder/auto-f0 全 None(score 生产默认口径;缺 cluster 资产时生产同样 skip)。
#[test]
#[ignore]
fn mg_render_sovits() {
    let sj = load_score();
    let slice = std::env::var("UTAI_MG_SLICE").unwrap_or_default();
    let (a, b) = if slice.is_empty() {
        (0usize, sj.triples.len())
    } else {
        let (s, e) = slice.split_once("..").expect("UTAI_MG_SLICE=a..b (triple indices)");
        (s.parse().unwrap(), e.parse().unwrap())
    };
    assert!(a < b && b <= sj.triples.len(), "bad slice {a}..{b}");
    let f_start: i64 = sj.triples[..a].iter().map(|t| t.frames).sum();
    let f_end: i64 = sj.triples[..b].iter().map(|t| t.frames).sum();
    let triples = &sj.triples[a..b];
    let cents = &sj.f0_cents[f_start as usize..f_end as usize];
    let voiced = &sj.f0_voiced[f_start as usize..f_end as usize];
    let evts = to_evts(triples);
    let vf0 = VocalF0 { cents, voiced };
    let (shift, inverse, kappa) = mg_shift_envs();
    let (tp, rs) = if inverse { (0, shift) } else { (shift, 0) };

    let model_path = std::path::PathBuf::from(
        std::env::var("UTAI_MG_MODEL").expect("UTAI_MG_MODEL (sovits onnx) required"),
    );
    let sc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(model_path.with_extension("json")).unwrap())
            .unwrap();
    assert_eq!(sc["type"].as_str(), Some("sovits"), "sovits arm wants a sovits sidecar");
    let dim = sc["features_dim"].as_u64().expect("features_dim") as usize;
    let sample_rate = sc["sample_rate"].as_u64().expect("sample_rate") as u32;
    let hop_size = sc["hop_size"].as_u64().unwrap_or(512) as usize;
    let min_frames = sc["min_frames"].as_u64().unwrap_or(6) as usize;
    let inputs: Vec<&str> = sc["inputs"]
        .as_array()
        .map(|l| l.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    let stem = model_path.file_stem().unwrap().to_string_lossy().to_string();
    let mtag = if stem.contains("东雪莲") { "dxl41".to_string() } else { stem };

    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let dll = root.join("../runtime/ort/onnxruntime.dll");
    assert!(dll.exists(), "ORT dll missing at {}", dll.display());
    if let Ok(bld) = ort::init_from(&dll) {
        let _ = bld.commit();
    }
    let engine = OnnxEngine::new();
    engine.set_device(DeviceConfig::Cpu);
    let aux = root.join("../data/models").join(crate::models::AUX_DIR_NAME);
    let s2cv = engine
        .load_model_with(
            &aux.join(if dim == 768 { "score2cv_768.onnx" } else { "score2cv_256.onnx" }),
            false,
        )
        .unwrap();
    let cv = engine
        .load_model_with(
            &aux.join(if dim == 768 { "contentvec_768l12.onnx" } else { "contentvec_256l9.onnx" }),
            false,
        )
        .unwrap();
    let rmvpe = engine.load_model_with(&aux.join("rmvpe_e2e.onnx"), false).unwrap();
    let rmvpe_mel: Array2<f32> = ndarray_npy::read_npy(&aux.join("rmvpe_mel_filters.npy")).unwrap();
    let voice = engine.load_model_with(&model_path, false).unwrap();
    let m = sovits::SovitsModel {
        engine: &engine,
        voice_session: &voice,
        contentvec_session: &cv,
        rmvpe_session: &rmvpe,
        mel_filters: &rmvpe_mel,
        cluster: None,
        diffusion: None,
        vocoder: None,
        f0_predictor_session: None,
        sample_rate,
        hop_size,
        features_dim: dim,
        vol_embedding: inputs.contains(&"vol"),
        phase_bins: sc["phase"]["phase_input"]
            .as_array()
            .and_then(|x| x.get(1))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize),
        f0d_cond_channels: sc["f0d_cond"]["input"]
            .as_array()
            .and_then(|x| x.get(1))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize),
        feed_uv: inputs.contains(&"uv"),
        spk_mix: None,
        unit_interpolate_mode: sc["unit_interpolate_mode"].as_str().unwrap_or("left").to_string(),
        noise_channels: sc["noise"]["noise_input"]
            .as_array()
            .and_then(|x| x.get(1))
            .and_then(|v| v.as_u64())
            .unwrap_or(192) as usize,
        min_frames,
    };
    let emph: f32 = std::env::var("UTAI_MG_EMPH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_VOICELESS_ONSET_EMPHASIS_DB);
    let valley: f32 = std::env::var("UTAI_MG_VALLEY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_CONSONANT_VALLEY_SCALE);
    let clarity: bool = std::env::var("UTAI_MG_CLARITY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(true);
    // e1 同款生产等价口径(noise_scale 0.4=这批模型 sidecar default_scale;seed 钉 0 保臂间对齐)。
    let sopts = SovitsOptions {
        seed: 0,
        noise_scale: 0.4,
        range_formant_follow: kappa,
        speaker_id: Some(0),
        ..Default::default()
    };
    let no_cancel = || false;
    let no_prog = |_: f32| {};
    let t0 = Instant::now();
    let r = render_score_sovits(
        &m, &s2cv, &evts, dim, 49, &NoDicts, &sopts,
        crate::commands::inference::VOCAL_FLAT_VOL, emph, valley, clarity, tp, rs,
        Some(&vf0), None, None, &no_cancel, &no_prog,
    )
    .unwrap();
    let out_dir = Path::new(WORK).join("probe");
    std::fs::create_dir_all(&out_dir).unwrap();
    let name = format!("mg_render_{a}_{b}_{mtag}{}.wav", mg_shift_tag(shift, inverse, kappa));
    write_wav16(&out_dir.join(&name), &r.audio, r.sample_rate);
    eprintln!(
        "[mg] sovits rendered triples[{a}..{b}] ({} frames): {:.2}s audio in {:.1}s wall -> probe\\{name}",
        f_end - f_start,
        r.audio.len() as f32 / r.sample_rate as f32,
        t0.elapsed().as_secs_f64()
    );
}

/// S84 E 刀原型(用户 UTAU 类比「渲染长音素再缩短」;diagnostic,零生产改动):快段元音在
/// S2CV 侧按放大时长渲染(articulation 轨迹完整到位),cv 行逐 phone 最近邻重采回真实时长
/// → f0/timing/net_g/输出级(rest-gate/emphasis/valley/B 刀权重)全部生产口径,**只换 cv 源
/// =单变量**。放大目标:唱段核 phone dur∈[1,4] → UTAI_MG_INFLATE(默认 6 帧=120ms 充分
/// articulation)。孪生 chunk 按真 chunk 的 phone 区间构造(同相位,组不跨 SP/lang 切=组内
/// note_dur 重算安全)。已知风险=压缩 cv 的超速过渡对 decoder 可能偏分布,实测裁决。
#[test]
#[ignore]
fn mg_render_rvc_oversampled() {
    let sj = load_score();
    let slice = std::env::var("UTAI_MG_SLICE").unwrap_or_default();
    let (a, b) = if slice.is_empty() {
        (0usize, sj.triples.len())
    } else {
        let (s, e) = slice.split_once("..").expect("UTAI_MG_SLICE=a..b");
        (s.parse().unwrap(), e.parse().unwrap())
    };
    let f_start: i64 = sj.triples[..a].iter().map(|t| t.frames).sum();
    let f_end: i64 = sj.triples[..b].iter().map(|t| t.frames).sum();
    let triples = &sj.triples[a..b];
    let cents = &sj.f0_cents[f_start as usize..f_end as usize];
    let voiced = &sj.f0_voiced[f_start as usize..f_end as usize];
    let evts = to_evts(triples);
    let vf0 = VocalF0 { cents, voiced };
    let inflate: i64 = std::env::var("UTAI_MG_INFLATE").ok().and_then(|s| s.parse().ok()).unwrap_or(6);
    // S84 E 刀二轮(用户耳测「ま→mai」):放大渲染的元音尾部是 S2CV 排的「向下一音预转」
    // (anticipation)——中心对齐采样会把它压进真音符里,i 向拐弯提前发生。TAIL=采样时从放大
    // 跨度末尾剪掉的帧数(core=起振+稳态;边界过渡交还给下一 phone 自己的 head 帧)。0=一轮行为。
    let tail: i64 = std::env::var("UTAI_MG_TAIL").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    // E 刀三轮:TAIL 去尾被耳测+band 双判无效(均匀采样保持「过渡:稳态」比例不变=听感不动)。
    // UTAI_MG_CORE="lo,hi"(如 0.30,0.70)= 只从放大跨度的稳态核窗采样(非均匀:过渡被挤出,
    // 相邻 phone 间变成 1 帧级 cv 快跳,decoder 平滑成快过渡=真唱边界干脆感)。设 CORE 时忽略 TAIL。
    let core: Option<(f64, f64)> = std::env::var("UTAI_MG_CORE").ok().and_then(|s| {
        let (lo, hi) = s.split_once(',')?;
        Some((lo.parse().ok()?, hi.parse().ok()?))
    });

    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let dll = root.join("../runtime/ort/onnxruntime.dll");
    assert!(dll.exists());
    if let Ok(bld) = ort::init_from(&dll) {
        let _ = bld.commit();
    }
    let engine = OnnxEngine::new();
    engine.set_device(DeviceConfig::Cpu);
    let aux = root.join("../data/models").join(crate::models::AUX_DIR_NAME);
    let s2cv768 = engine.load_model_with(&aux.join("score2cv_768.onnx"), false).unwrap();
    let cv768 = engine.load_model_with(&aux.join("contentvec_768l12.onnx"), false).unwrap();
    let rmvpe = engine.load_model_with(&aux.join("rmvpe_e2e.onnx"), false).unwrap();
    let rmvpe_mel: Array2<f32> = ndarray_npy::read_npy(&aux.join("rmvpe_mel_filters.npy")).unwrap();
    let teto = engine.load_model_with(&Path::new(WORK).join("tetoRVC_best.onnx"), false).unwrap();
    let index = rvc::RvcIndex::load(&Path::new(WORK).join("tetoRVC_best.npy")).unwrap();
    let m = rvc::RvcModel {
        engine: &engine,
        voice_session: &teto,
        contentvec_session: &cv768,
        rmvpe_session: &rmvpe,
        mel_filters: &rmvpe_mel,
        index: Some(&index),
        sample_rate: 48000,
        features_dim: 768,
        spk_mix: None,
        noise_channels: 192,
        min_frames: 12,
    };
    let ropts = RvcOptions { seed: 0, ..Default::default() };

    // ── 真时间轴(生产口径逐行)──
    let arr = build_arrays_daw(&evts, &NoDicts).unwrap();
    let mut note_hz_full = build_note_hz(&arr, &evts, 0, Some(&vf0));
    zero_voiceless_frames(&mut note_hz_full, &arr);
    anchor_voiced_phone_f0(&mut note_hz_full, &arr);
    let chunks = chunk_at_sp(&arr, 400);
    let vl_onset = voiceless_onset_flags(&arr);
    let emphasis_gain = 10f32.powf(DEFAULT_VOICELESS_ONSET_EMPHASIS_DB / 20.0);
    let valley_depths = boundary_valley_depths(&arr);
    let idx_weights = fast_index_weights(&arr);

    // ── 放大时长(核 phone ≤4 → inflate;组内重算 note_dur)──
    let mut dur_inf = arr.phone_dur.clone();
    let mut n_inflated = 0usize;
    for i in 0..arr.phon.len() {
        if arr.note_pitch[i] > 0
            && super::super::score2cv::is_nucleus_phone(arr.phon[i])
            && (1..=4).contains(&arr.phone_dur[i])
        {
            dur_inf[i] = inflate.max(arr.phone_dur[i]);
            n_inflated += 1;
        }
    }
    eprintln!("[mg] oversample: {n_inflated} nuclei inflated to {inflate} frames (S2CV side only)");

    let mut audio: Vec<f32> = Vec::new();
    let mut cv_cursor = 0usize;
    for (ci, chunk) in chunks.iter().enumerate() {
        // 孪生 chunk:同 phone 区间,放大时长;note_dur = 组内 dur_inf 和(组不跨 chunk 切)。
        let rng = chunk.start..chunk.end;
        let pd_inf: Vec<i64> = dur_inf[rng.clone()].to_vec();
        let mut nd_inf = vec![0i64; pd_inf.len()];
        let g = &chunk.note_to_phone;
        for k in 0..pd_inf.len() {
            nd_inf[k] = (0..pd_inf.len()).filter(|&j| g[j] == g[k]).map(|j| pd_inf[j]).sum();
        }
        let t_inf: usize = pd_inf.iter().map(|&d| d.max(0) as usize).sum();
        let chunk_inf = Chunk {
            start: chunk.start,
            end: chunk.end,
            phonemes: chunk.phonemes.clone(),
            note_pitch: chunk.note_pitch.clone(),
            phone_dur: pd_inf.clone(),
            note_dur: nd_inf,
            note_to_phone: chunk.note_to_phone.clone(),
            t: t_inf,
            lang_id: chunk.lang_id,
            hard_seam: chunk.hard_seam,
        };
        let cv_inf = run_score2cv(m.engine, &s2cv768, &chunk_inf, 768, 49, chunk.lang_id).unwrap();
        // E 刀三轮诊断(UTAI_MG_DUMPCV):放大元音的 cv 帧内变异——若帧间近乎相同(det 慢变),
        // 采样窗选择=空转钮,E 收益全来自「放大时长输入改变整个元音的 cv 目标」。
        if std::env::var("UTAI_MG_DUMPCV").is_ok() {
            let mut c0 = 0usize;
            for k in 0..pd_inf.len() {
                let d = pd_inf[k].max(0) as usize;
                let d_true0 = chunk.phone_dur[k].max(0) as usize;
                if d > d_true0 && d >= 2 {
                    // 帧间相邻余弦 + 首尾余弦(1.0=完全相同)
                    let cos = |x: usize, y: usize| {
                        let (a, b) = (cv_inf.row(c0 + x), cv_inf.row(c0 + y));
                        let dot: f32 = a.iter().zip(b.iter()).map(|(p, q)| p * q).sum();
                        let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
                        let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
                        dot / (na * nb + 1e-9)
                    };
                    let adj: Vec<f32> = (0..d - 1).map(|x| cos(x, x + 1)).collect();
                    let adj_min = adj.iter().cloned().fold(1.0f32, f32::min);
                    eprintln!(
                        "[mg] cv-var phone[{}]{} d_inf={} first-last cos={:.4} adj-min cos={:.4}",
                        chunk.start + k,
                        // phon not in Chunk — recover via arr
                        arr.phon[chunk.start + k],
                        d,
                        cos(0, d - 1),
                        adj_min
                    );
                }
                c0 += d;
            }
        }
        // 逐 phone 最近邻重采回真时长(中心对齐采样)。
        let mut cv = Array2::<f32>::zeros((chunk.t, 768));
        let (mut c_true, mut c_inf) = (0usize, 0usize);
        for k in 0..pd_inf.len() {
            let d_true = chunk.phone_dur[k].max(0) as usize;
            let d_inf = pd_inf[k].max(0) as usize;
            // 放大过的 phone:采样域=CORE 稳态窗(非均匀)或去尾窗;未放大:全域(=恒等)。
            let (w_lo, w_len) = if d_inf > d_true {
                if let Some((lo, hi)) = core {
                    let a0 = (d_inf as f64 * lo).min(d_inf as f64 - 1.0);
                    (a0, ((d_inf as f64 * hi) - a0).max(1.0))
                } else {
                    (0.0, (d_inf as i64 - tail).max(d_true as i64) as f64)
                }
            } else {
                (0.0, d_inf as f64)
            };
            for j in 0..d_true {
                let src = c_inf + (w_lo + (j as f64 + 0.5) * w_len / d_true as f64) as usize;
                let src = src.min(c_inf + d_inf.saturating_sub(1)).min(cv_inf.nrows().saturating_sub(1));
                cv.row_mut(c_true + j).assign(&cv_inf.row(src));
            }
            c_true += d_true;
            c_inf += d_inf;
        }
        // ── 以下 = render_score_rvc 主循环逐行(cv 已换源)──
        let note_hz = &note_hz_full[cv_cursor..(cv_cursor + chunk.t).min(note_hz_full.len())];
        let (cv_p, pitch, pitchf, real_t) = rvc_feed_100(cv, note_hz, m.min_frames);
        let w_chunk = &idx_weights[cv_cursor..(cv_cursor + chunk.t).min(idx_weights.len())];
        let mut wav = vc_decode(
            &m, cv_p, &pitch, &pitchf, 0, None, &ropts, ci as u64, usize::MAX, Some(w_chunk),
        )
        .unwrap();
        if pitchf.len() > real_t {
            wav.truncate((real_t * (m.sample_rate as usize / 100)).min(wav.len()));
        }
        let sp_wins = chunk_sp_windows(chunk, wav.len());
        apply_rest_gate(&mut wav, &sp_wins, rest_gate_fade_samples(m.sample_rate));
        let emph_wins = chunk_flag_windows(chunk, wav.len(), &vl_onset[chunk.start..chunk.end]);
        apply_emphasis(&mut wav, &emph_wins, emphasis_gain, emphasis_fade_samples(m.sample_rate));
        let val_cls = chunk_valley_clusters(chunk, wav.len(), &valley_depths[chunk.start..chunk.end]);
        apply_valley(&mut wav, &val_cls, DEFAULT_CONSONANT_VALLEY_SCALE, emphasis_fade_samples(m.sample_rate));
        if chunk.hard_seam {
            seam_fade(&mut audio, &mut wav, m.sample_rate);
        }
        audio.extend_from_slice(&wav);
        cv_cursor += chunk.t;
    }
    peak_normalize(&mut audio, 0.92);
    let out_dir = Path::new(WORK).join("probe");
    let ttag = if let Some((lo, hi)) = core {
        format!("_c{:.0}_{:.0}", lo * 100.0, hi * 100.0)
    } else if tail != 0 {
        format!("_t{tail}")
    } else {
        String::new()
    };
    let name = format!("mg_render_{a}_{b}_teto_oversampled{ttag}.wav");
    write_wav16(&out_dir.join(&name), &audio, m.sample_rate);
    eprintln!("[mg] oversampled render -> probe\\{name} ({:.2}s)", audio.len() as f32 / m.sample_rate as f32);
}

/// 目标组(用户点名):同一 UTAI_MG_SLICE 时窗的 MixDown(OpenUtau 渲染源)走**真翻唱管线**
/// `rvc::run_pipeline` 生产默认口径(RvcOptions::default():index 0.75/protect 0.33/rms_mix 0.25
/// =装机 cover 节点缺省;e1 A 臂同款调用面,不做 A 臂的归因偏离)——「无参换声」参照臂,
/// score 各变体与它对听才有裁决意义。时窗按三元组下标换算歌曲绝对时间
/// (frame/50 + start_tick 段偏移),MixDown t0=UST tick0 同轴(S83 取证)。
#[test]
#[ignore]
fn mg_render_cover() {
    let sj = load_score();
    let notes_meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(Path::new(WORK).join("probe").join("mg_notes.json")).unwrap(),
    )
    .unwrap();
    let start_tick = notes_meta["start_tick"].as_f64().expect("start_tick");
    let seg_off = start_tick / (480.0 * sj.tempo / 60.0);
    let slice = std::env::var("UTAI_MG_SLICE").unwrap_or_default();
    let (a, b) = if slice.is_empty() {
        (0usize, sj.triples.len())
    } else {
        let (s, e) = slice.split_once("..").expect("UTAI_MG_SLICE=a..b (triple indices)");
        (s.parse().unwrap(), e.parse().unwrap())
    };
    assert!(a < b && b <= sj.triples.len(), "bad slice {a}..{b}");
    let f_start: i64 = sj.triples[..a].iter().map(|t| t.frames).sum();
    let f_end: i64 = sj.triples[..b].iter().map(|t| t.frames).sum();
    let t_start = f_start as f64 / 50.0 + seg_off;
    let t_end = f_end as f64 / 50.0 + seg_off;

    let src_path = Path::new(WORK).join("未命名_MixDown.wav");
    let full = crate::audio::load_audio(&src_path).unwrap();
    // 时窗切片(按声道交错帧界)+ 立体声并单声道(翻唱管线口径=mono)。
    let ch = full.channels.max(1) as usize;
    let n_frames_total = full.samples.len() / ch;
    let s0 = ((t_start * full.sample_rate as f64) as usize).min(n_frames_total) * ch;
    let s1 = ((t_end * full.sample_rate as f64) as usize).min(n_frames_total) * ch;
    let mono: Vec<f32> = full.samples[s0..s1]
        .chunks_exact(ch)
        .map(|fr| fr.iter().sum::<f32>() / ch as f32)
        .collect();
    let src = crate::audio::AudioBuffer { samples: mono, sample_rate: full.sample_rate, channels: 1 };

    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let dll = root.join("../runtime/ort/onnxruntime.dll");
    assert!(dll.exists(), "ORT dll missing at {}", dll.display());
    if let Ok(bld) = ort::init_from(&dll) {
        let _ = bld.commit();
    }
    let engine = OnnxEngine::new();
    engine.set_device(DeviceConfig::Cpu);
    let aux = root.join("../data/models").join(crate::models::AUX_DIR_NAME);
    let cv768 = engine.load_model_with(&aux.join("contentvec_768l12.onnx"), false).unwrap();
    let rmvpe = engine.load_model_with(&aux.join("rmvpe_e2e.onnx"), false).unwrap();
    let rmvpe_mel: Array2<f32> = ndarray_npy::read_npy(&aux.join("rmvpe_mel_filters.npy")).unwrap();
    let (model_path, index_path, mtag) = mg_model_envs();
    let teto = engine.load_model_with(&model_path, false).unwrap();
    let index = rvc::RvcIndex::load(&index_path).unwrap();
    let m = rvc::RvcModel {
        engine: &engine,
        voice_session: &teto,
        contentvec_session: &cv768,
        rmvpe_session: &rmvpe,
        mel_filters: &rmvpe_mel,
        index: Some(&index),
        sample_rate: 48000,
        features_dim: 768,
        spk_mix: None,
        noise_channels: 192,
        min_frames: 12,
    };
    // S85 靶场:cover 的 shift 生产等价 = fed f0 ×2^(s/12)(输入音频/cv 不动)。f0_shift 在
    // run_pipeline 里先于 coarse 量化乘到整轨(rvc.rs:256)= 生产 ranged_chunk 逐 chunk 乘的
    // 同一数学;raw 臂(UTAI_MG_INVERSE=0)即到此为止=「模型在 shift 位唱 cover」。
    let (shift, inverse, kappa) = mg_shift_envs();
    let ropts = RvcOptions { f0_shift: shift as f32, ..Default::default() };
    let noop = |_: f32| {};
    let no_cancel = || false;
    let t0 = Instant::now();
    let r = rvc::run_pipeline(&m, &src, &ropts, None, &noop, &no_cancel).unwrap();
    let mut audio = r.audio;
    if inverse && shift != 0 {
        // 逆变换臂(记档偏差 vs 生产,三条均判无害:①生产=逐 chunk 对未裁 pad 输出 inverse
        //   (接缝色染留在被裁 pad 内),这里=整段单次(天然无缝,同或更干净);②fed base 网格
        //   取自未 pad 源(生产=padded 逐 chunk 切片)——sticky ~100ms 浊音中位对平移不敏感;
        //   ③省略 48Hz 高通(远低于人声 f0,不动中位数)。
        // 16k 降采样走生产同款 polyphase(features::resample=rvc 管线的 16k 路径);
        // audio::resample::resample 的 rubato FftFixedIn 在 debug 下对整曲长输入 pow 溢出(探针踩过)。
        let mono16 =
            super::super::features::resample(&src.samples, src.sample_rate, 16000);
        let mut pf = super::super::f0::rmvpe_detect_chunked(
            &engine,
            &rmvpe,
            &rmvpe_mel,
            &mono16,
            super::super::f0::RVC_RMVPE_THRESHOLD,
        )
        .unwrap();
        let kr = 2.0f32.powf(shift as f32 / 12.0);
        pf.iter_mut().for_each(|v| *v *= kr);
        audio = super::super::vocal_range::apply_inverse(
            audio,
            r.sample_rate,
            shift,
            kappa,
            Some((&pf, r.sample_rate as usize / 100)),
        )
        .unwrap();
    }
    let out_dir = Path::new(WORK).join("probe");
    std::fs::create_dir_all(&out_dir).unwrap();
    let name = format!("mg_cover_{a}_{b}_{mtag}{}.wav", mg_shift_tag(shift, inverse, kappa));
    write_wav16(&out_dir.join(&name), &audio, r.sample_rate);
    eprintln!(
        "[mg] cover {t_start:.3}s..{t_end:.3}s ({:.2}s in, {:.2}s out) in {:.1}s wall -> probe\\{name}",
        src.samples.len() as f32 / src.sample_rate as f32,
        audio.len() as f32 / r.sample_rate as f32,
        t0.elapsed().as_secs_f64()
    );
}
