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

use super::*;
use super::e1_tests::write_wav16;
use super::super::engine::{DeviceConfig, OnnxEngine};
use super::super::rvc;
use super::super::score2cv::{is_nucleus_phone, NoDicts};
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
    let ropts = RvcOptions { seed: 0, index_ratio: idx_ratio, protect, ..Default::default() };
    let no_cancel = || false;
    let no_prog = |_: f32| {};
    let t0 = Instant::now();
    let r = render_score_rvc(
        &m, &s2cv768, &evts, 768, 49, &NoDicts, &ropts, emph, 0, 0,
        Some(&vf0), None, None, &no_cancel, &no_prog,
    )
    .unwrap();
    let out_dir = Path::new(WORK).join("probe");
    std::fs::create_dir_all(&out_dir).unwrap();
    let tag = format!(
        "{}{}{}",
        if emph != DEFAULT_VOICELESS_ONSET_EMPHASIS_DB { format!("_e{emph}") } else { String::new() },
        if idx_ratio != 0.75 { format!("_i{idx_ratio}") } else { String::new() },
        if protect != RvcOptions::default().protect { format!("_p{protect}") } else { String::new() },
    );
    let name = format!("mg_render_{a}_{b}_teto{tag}.wav");
    write_wav16(&out_dir.join(&name), &r.audio, r.sample_rate);
    eprintln!(
        "[mg] rendered triples[{a}..{b}] ({} frames): {:.2}s audio in {:.1}s wall -> probe\\{name}",
        f_end - f_start,
        r.audio.len() as f32 / r.sample_rate as f32,
        t0.elapsed().as_secs_f64()
    );
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
    let ropts = RvcOptions::default();
    let noop = |_: f32| {};
    let no_cancel = || false;
    let t0 = Instant::now();
    let r = rvc::run_pipeline(&m, &src, &ropts, None, &noop, &no_cancel).unwrap();
    let out_dir = Path::new(WORK).join("probe");
    std::fs::create_dir_all(&out_dir).unwrap();
    let name = format!("mg_cover_{a}_{b}_teto.wav");
    write_wav16(&out_dir.join(&name), &r.audio, r.sample_rate);
    eprintln!(
        "[mg] cover {t_start:.3}s..{t_end:.3}s ({:.2}s in, {:.2}s out) in {:.1}s wall -> probe\\{name}",
        src.samples.len() as f32 / src.sample_rate as f32,
        r.audio.len() as f32 / r.sample_rate as f32,
        t0.elapsed().as_secs_f64()
    );
}
