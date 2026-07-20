//! E1 交叉判别实验 harness(S70)— diagnostic,NOT a gate(`#[ignore]`)。
//!
//! 目的:在 S69 干净基线上把「自己唱 vs SynthV 干声走翻唱」的残余差距按流归因——
//! 2×2 交叉 {cv: 天然 ContentVec(SynthV 干声) | S2CV} × {f0: 天然 RMVPE | 参数化 Option-A}:
//!   A_natCV_natF0   = 真翻唱管线跑 SynthV 干声(上限参照,run_pipeline 原样)
//!   B_natCV_paramF0 = 天然 cv × 生产默认参数化 f0(隔离:好 cv 遇上合成 f0)
//!   C_s2cv_natF0    = S2CV cv × 天然 f0(隔离:合成 cv 遇上真人 f0)
//!   D_s2cv_paramF0  = 生产自己唱(render_score_sovits/rvc 原入口,Option-A f0)
//! 判读:C≈A → 差距主要在 f0(f0 旋钮线优先);B≈A → 差距主要在 cv(v2 重训优先)。
//!
//! 素材:《虽然歌声无形》SynthV(Kasane Teto AI 2)无调教主旋律干声 + 同工程谱(svp 直出,
//! tick 0 == wav sample 0 == 段起点,同一时间轴)。参数化 f0 由 vitest 跑真前端 buildVocalScore
//! dump(src\lib\vocal\e1CrossDump.test.ts)——生产口径、零复制。
//!
//! 实现说明:挂为 score2svc.rs 的 #[cfg(test)] 子模块 → 可见父模块私有项
//! (zero_voiceless_frames / build_vol_env / pad_sovits_feed / seam_fade / peak_normalize),
//! B/C 交叉臂逐行镜像 render_score_sovits / render_score_rvc 的主循环(transpose=0、无泳道、
//! range_shift=0 的中性路径),只把 cv / f0 一条流换源;生产代码零触碰。
//! 天然臂提取不加翻唱的 0.5s pad(那是切片保护,加了反而错位)——切片两端本就落在休止里。
//! 天然 f0 按后端各自翻唱口径提两份(SoVITS: 裸 wav16k+阈值 0.05;RVC: 48Hz 高通+0.03,审查
//! confirmed 的单变量要求);已知不可消除的口径差(README「口径警告」全记):A 臂无 peak_normalize、
//! dxl41 的 A 臂独享真实 Volume_Extractor vol 流(B/C/D=flat×ADSR)→ dxl41 组只作旁证,
//! 流归因主判 = akiko256 + rvcleng(均无 vol 口)。
//!
//! Inputs:  D:\MyDev\TESTING\e1_cross_probe\{seg}_src48k.wav + {seg}_score.json
//! Outputs: 同目录 {seg}__{arm}__{model}.wav(已存在则跳过,可断点续跑)
//! Run(src-tauri 下;CPU EP 保证可比):
//!   cargo test --lib inference::score2svc::e1_tests::e1_cross_probe -- --ignored --nocapture
//! Env: UTAI_E1_ONLY=verse | UTAI_E1_ARMS=A,C | UTAI_E1_MODELS=akiko256,rvcleng | UTAI_E1_DEVICE=auto

use super::*;
use super::super::engine::{DeviceConfig, OnnxEngine};
use super::super::f0 as f0mod;
use super::super::features;
use super::super::score2cv::NoDicts;
use super::super::{rvc, sovits};
use std::path::{Path, PathBuf};
use std::time::Instant;

const WORK_DIR: &str = r"D:\MyDev\TESTING\e1_cross_probe";
const SEGMENTS: [&str; 2] = ["verse", "chorus"];
/// ScoreToCV 条件 speaker(生产默认 49 = kiritan,DEFAULT_VOCAL_PARAMS.speakerId)。
const CV_SPEAKER: i64 = 49;

#[derive(serde::Deserialize)]
struct ScoreJson {
    name: String,
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
    lang: i64,
}

/// cv 流来源:S2CV(生产)或 天然 ContentVec(50fps,与谱同一时间轴,按帧号直切)。
enum CvSrc<'a> {
    S2cv(&'a str),
    Natural(&'a Array2<f32>),
}
/// f0 流来源:参数化 Option-A(生产整形:build_note_hz + zero_voiceless_frames)或
/// 天然 RMVPE(100fps Hz,0=无声——真实 voicing,不再叠加谱面清音归零)。
enum F0Src<'a> {
    Param(&'a VocalF0<'a>),
    Natural(&'a [f32]),
}

/// 天然 cv 的帧号直切:[start, start+t) 行,越界 clamp 到最后一行(切片 wav 比谱长 0.4s 尾边距,
/// 只有谱尾 ±1 帧会触发 clamp)。时间轴对齐靠「tick 0 == sample 0」的素材约定,不做比例缩放
/// (缩放会把边距时长摊进内容,反而错位)。
fn slice_rows_clamped(src: &Array2<f32>, start: usize, t: usize) -> Array2<f32> {
    let n = src.nrows();
    assert!(n > 0, "empty natural cv");
    Array2::from_shape_fn((t, src.ncols()), |(i, j)| src[[(start + i).min(n - 1), j]])
}

/// B/C/D' 通用 SoVITS 渲染:render_score_sovits 的逐行镜像(中性参数路径),cv/f0 可换源。
/// 与生产的差异仅在两处注入点;其余(chunking/重采样/pad/vol/decode/seam/normalize)一致。
#[allow(clippy::too_many_arguments)]
fn e1_render_sovits(
    m: &sovits::SovitsModel,
    score: &[ScoreEvt],
    dim: usize,
    cv_src: &CvSrc,
    f0_src: &F0Src,
    options: &SovitsOptions,
    flat_vol: f32,
) -> crate::Result<SynthesisResult> {
    let arr = build_arrays_daw(score, &NoDicts)?;
    // 参数化 f0 = 生产整形(Option-A cents→Hz + 清音帧归零);天然 f0 不在此处理(真实 voicing)。
    let param_hz_full: Vec<f32> = match f0_src {
        F0Src::Param(f0) => {
            let mut h = build_note_hz(&arr, score, 0, Some(f0));
            zero_voiceless_frames(&mut h, &arr);
            h
        }
        F0Src::Natural(_) => Vec::new(),
    };
    let vol_env_cv = if m.vol_embedding { Some(build_vol_env(&arr, score)) } else { None };
    let chunks = chunk_at_sp(&arr, 400);
    let has_diff = false; // 诊断件不开扩散(与 r0a/r0b 基线同口径)
    let p_vits = 0.95;
    let noop = |_: f32| {};
    let no_cancel = || false;
    let mut audio: Vec<f32> = Vec::new();
    let mut cv_cursor = 0usize;
    for (ci, chunk) in chunks.iter().enumerate() {
        let cv = match cv_src {
            CvSrc::S2cv(session) => {
                run_score2cv(m.engine, session, chunk, dim, CV_SPEAKER, chunk.lang_id)?
            }
            CvSrc::Natural(nc) => slice_rows_clamped(nc, cv_cursor, chunk.t),
        };
        let mut feed = match f0_src {
            F0Src::Param(_) => {
                let hz = &param_hz_full[cv_cursor..(cv_cursor + chunk.t).min(param_hz_full.len())];
                resample_to_sovits_grid(&cv, hz, m.sample_rate, m.hop_size, &m.unit_interpolate_mode)?
            }
            F0Src::Natural(nf) => {
                // 天然 f0 以 100fps 原生分辨率直入 sovits_f0_postprocess(它 nearest 到 t_tgt),
                // 比先降到 50fps 再上采样少丢一半纹理;uv/gap 插值语义与翻唱路径完全一致。
                let t_tgt = sovits_grid_len(chunk.t, m.sample_rate, m.hop_size);
                if t_tgt == 0 {
                    return Err(crate::UtaiError::Inference("SCORE2SVC_ZERO_FRAMES".into()));
                }
                let cv_rs = repeat_expand_2d(&cv, t_tgt, &m.unit_interpolate_mode)?;
                let lo = (2 * cv_cursor).min(nf.len());
                let hi = (2 * (cv_cursor + chunk.t)).min(nf.len());
                let slice: &[f32] = if lo < hi { &nf[lo..hi] } else { &[0.0] };
                let (f0_rs, uv_rs) =
                    f0mod::sovits_f0_postprocess(slice, t_tgt, m.hop_size, m.sample_rate);
                SovitsFeed { cv: cv_rs, f0: f0_rs, uv: uv_rs, t_tgt }
            }
        };
        let real_t = pad_sovits_feed(&mut feed, m.min_frames);
        sovits::apply_cluster_blend(&mut feed.cv, m.cluster, options.cluster_ratio);
        let vol = vol_env_cv.as_ref().map(|env| {
            let end = (cv_cursor + chunk.t).min(env.len());
            let combined: Vec<f32> = env[cv_cursor..end].to_vec(); // 无响度泳道(lane=1.0)
            torch_interp_nearest(&combined, feed.t_tgt).into_iter().map(|v| flat_vol * v).collect()
        });
        let padded_t = feed.t_tgt;
        let mut wav = sovits::decode_features(
            m, feed.cv, feed.f0, feed.uv, vol, &[], ci as u64, padded_t, has_diff, p_vits, options,
            &noop, &no_cancel,
        )?;
        if padded_t > real_t {
            wav.truncate((real_t * m.hop_size).min(wav.len()));
        }
        if chunk.hard_seam {
            seam_fade(&mut audio, &mut wav, m.sample_rate);
        }
        audio.extend_from_slice(&wav);
        cv_cursor += chunk.t;
    }
    peak_normalize(&mut audio, 0.92);
    Ok(SynthesisResult { audio, sample_rate: m.sample_rate })
}

/// B/C 通用 RVC 渲染:render_score_rvc 的逐行镜像(中性参数路径),cv/f0 可换源。
/// 天然 f0 臂按 100fps 原生分辨率直建 pitch/pitchf(不经 50fps 中转)——RVC 翻唱口径本就是
/// 100fps 真 RMVPE(裸 0 休止,protect 靠它触发)。
fn e1_render_rvc(
    m: &rvc::RvcModel,
    score: &[ScoreEvt],
    dim: usize,
    cv_src: &CvSrc,
    f0_src: &F0Src,
    options: &RvcOptions,
) -> crate::Result<SynthesisResult> {
    let arr = build_arrays_daw(score, &NoDicts)?;
    let param_hz_full: Vec<f32> = match f0_src {
        F0Src::Param(f0) => {
            let mut h = build_note_hz(&arr, score, 0, Some(f0));
            zero_voiceless_frames(&mut h, &arr);
            h
        }
        F0Src::Natural(_) => Vec::new(),
    };
    let chunks = chunk_at_sp(&arr, 400);
    let sid = options.speaker_id.unwrap_or(0) as i64;
    let mut audio: Vec<f32> = Vec::new();
    let mut cv_cursor = 0usize;
    for (ci, chunk) in chunks.iter().enumerate() {
        let cv = match cv_src {
            CvSrc::S2cv(session) => {
                run_score2cv(m.engine, session, chunk, dim, CV_SPEAKER, chunk.lang_id)?
            }
            CvSrc::Natural(nc) => slice_rows_clamped(nc, cv_cursor, chunk.t),
        };
        let (cv_p, pitch, pitchf, real_t) = match f0_src {
            F0Src::Param(_) => {
                let hz = &param_hz_full[cv_cursor..(cv_cursor + chunk.t).min(param_hz_full.len())];
                rvc_feed_100(cv, hz, m.min_frames)
            }
            F0Src::Natural(nf) => {
                // rvc_feed_100 的 pad 语义逐行镜像,只把「50fps 重复 2×」换成真 100fps 采样。
                let t50 = cv.nrows();
                if t50 == 0 {
                    (cv, Vec::new(), Vec::new(), 0)
                } else {
                    let real_100 = t50 * 2;
                    let pad50 = if real_100 < m.min_frames { m.min_frames.div_ceil(2) } else { t50 };
                    let cv_p = if pad50 > t50 {
                        let mut padded = Array2::<f32>::zeros((pad50, cv.ncols()));
                        for i in 0..pad50 {
                            padded.row_mut(i).assign(&cv.row(i.min(t50 - 1)));
                        }
                        padded
                    } else {
                        cv
                    };
                    let base = 2 * cv_cursor;
                    let mut pitchf: Vec<f32> = Vec::with_capacity(pad50 * 2);
                    for i in 0..pad50 * 2 {
                        let idx = (base + i).min(nf.len().saturating_sub(1));
                        pitchf.push(nf.get(idx).copied().unwrap_or(0.0));
                    }
                    let pitch: Vec<i64> = pitchf.iter().map(|&f| f0_to_coarse(f)).collect();
                    (cv_p, pitch, pitchf, real_100)
                }
            }
        };
        let mut wav = vc_decode(m, cv_p, &pitch, &pitchf, sid, None, options, ci as u64, usize::MAX)?;
        if pitchf.len() > real_t {
            wav.truncate((real_t * (m.sample_rate as usize / 100)).min(wav.len()));
        }
        if chunk.hard_seam {
            seam_fade(&mut audio, &mut wav, m.sample_rate);
        }
        audio.extend_from_slice(&wav);
        cv_cursor += chunk.t;
    }
    peak_normalize(&mut audio, 0.92);
    Ok(SynthesisResult { audio, sample_rate: m.sample_rate })
}

fn read_sidecar(model: &Path) -> serde_json::Value {
    let p = model.with_extension("json");
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display())))
        .unwrap_or_else(|e| panic!("parse {}: {e}", p.display()))
}
fn sidecar_noise_channels(sc: &serde_json::Value) -> usize {
    sc.get("noise")
        .and_then(|v| v.get("rnd_input").or_else(|| v.get("noise_input")))
        .and_then(|v| v.as_array())
        .and_then(|a| a.get(1))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(192)
}

/// 16-bit wav 落盘(diagnostic 输出;与 tests 模块的 write_wav16 同构——那个是 tests 私有,
/// 子模块不可见,故本地复制这 10 行)。
fn write_wav16(path: &Path, samples: &[f32], sr: u32) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sr,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec).unwrap();
    for &s in samples {
        w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16).unwrap();
    }
    w.finalize().unwrap();
}

#[test]
#[ignore]
fn e1_cross_probe() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
    let dll = root.join("src-tauri/../runtime/ort/onnxruntime.dll");
    let dll = if dll.exists() { dll } else { root.join("runtime/ort/onnxruntime.dll") };
    assert!(dll.exists(), "ORT dll missing at {}", dll.display());
    if let Ok(b) = ort::init_from(&dll) {
        let _ = b.commit();
    }
    let engine = OnnxEngine::new();
    if std::env::var("UTAI_E1_DEVICE").as_deref() != Ok("auto") {
        engine.set_device(DeviceConfig::Cpu);
    }

    let work = PathBuf::from(WORK_DIR);
    let only = std::env::var("UTAI_E1_ONLY").unwrap_or_default();
    let arms_filter = std::env::var("UTAI_E1_ARMS").unwrap_or_default();
    let models_filter = std::env::var("UTAI_E1_MODELS").unwrap_or_default();
    let enabled = |filter: &str, name: &str| -> bool {
        filter.is_empty() || filter.split(',').any(|a| a.trim() == name)
    };

    let aux = root.join("data/models").join(crate::models::AUX_DIR_NAME);
    let sov_dir = root.join("data/models/sovits");
    let rvc_dir = root.join("data/models/rvc");

    let s2cv768 = engine.load_model_with(&aux.join("score2cv_768.onnx"), false).unwrap();
    let s2cv256 = engine.load_model_with(&aux.join("score2cv_256.onnx"), false).unwrap();
    let cv768 = engine.load_model_with(&aux.join("contentvec_768l12.onnx"), false).unwrap();
    let cv256 = engine.load_model_with(&aux.join("contentvec_256l9.onnx"), false).unwrap();
    let rmvpe = engine.load_model_with(&aux.join("rmvpe_e2e.onnx"), false).unwrap();
    let mel: Array2<f32> = ndarray_npy::read_npy(aux.join("rmvpe_mel_filters.npy")).unwrap();

    // 生产口径的质量参数(render_audition_wavs 同款),两处为归因纯度故意偏离翻唱默认(README 并记):
    //  * index_ratio 0.75→0 —— 检索会把 cv 差异洗掉一截(E5 已量化);
    //  * rms_mix_rate 0.25→1.0 —— 否则 rvc::run_pipeline(A 臂)会经 change_rms 把源响度包络
    //    以 0.75 权重转印到输出(审查 confirmed:B/C/D 结构上无此流,生产自己唱也强制 1.0),
    //    A 独享真人乐句动态 = 2×2 之外的第三条流,B/C→A 残差会被夸大。
    let sopts = SovitsOptions { seed: 0, noise_scale: 0.4, ..Default::default() };
    let ropts = RvcOptions { seed: 0, index_ratio: 0.0, rms_mix_rate: 1.0, ..Default::default() };
    let flat_vol = crate::commands::inference::VOCAL_FLAT_VOL;

    for seg in SEGMENTS {
        if !only.is_empty() && !seg.contains(&only) {
            continue;
        }
        let sj: ScoreJson = serde_json::from_str(
            &std::fs::read_to_string(work.join(format!("{seg}_score.json"))).unwrap(),
        )
        .unwrap();
        assert_eq!(sj.name, seg);
        let total: i64 = sj.triples.iter().map(|t| t.frames).sum();
        assert_eq!(sj.f0_cents.len() as i64, total, "{seg}: f0 length vs Σframes");
        assert!(sj.triples.iter().all(|t| t.lang == 2), "{seg}: JA-only song expected");
        // K 臂(S71 第四刀):Phase A 旋钮模型预测 θ 的 Option-A f0(e1KarmDump.test.ts 产)。
        // 与 D 臂唯一差异 = 调教参数(同 triples 同渲染入口);缺文件则跳过 K。
        // S71+1 第二轮多变体:UTAI_E1K_TAG(默认 "K")选源 {seg}_score_{TAG}.json 并命名
        // 输出臂 {TAG}_s2cv_knobF0(如 KA=run-A / KAB72=run-AB spk72);arms 过滤仍用 "K"。
        let ktag = std::env::var("UTAI_E1K_TAG").unwrap_or_else(|_| "K".into());
        let sk: Option<ScoreJson> = std::fs::read_to_string(work.join(format!("{seg}_score_{ktag}.json")))
            .ok()
            .map(|s| serde_json::from_str(&s).unwrap());
        if let Some(k) = &sk {
            let ktotal: i64 = k.triples.iter().map(|t| t.frames).sum();
            assert_eq!(ktotal, total, "{seg}: K 臂 Σframes 必须与 D 一致(θ 不动谱面)");
            assert_eq!(k.f0_cents.len() as i64, total, "{seg}: K f0 length vs Σframes");
        }
        let evts: Vec<ScoreEvt> = sj
            .triples
            .iter()
            .map(|t| ScoreEvt {
                lyric: &t.lyric,
                note_num: t.note_num,
                frames: t.frames,
                lang: Lang::Ja,
                phoneme_input: None,
            })
            .collect();
        let vf0 = VocalF0 { cents: &sj.f0_cents, voiced: &sj.f0_voiced };
        let vf0_k = sk.as_ref().map(|k| VocalF0 { cents: &k.f0_cents, voiced: &k.f0_voiced });

        // S72 K′ 物料:dump score2cv 模型输入数组(G2P 后 per-phone int64,python v2 推理用;
        // 单一源=生产 build_arrays_daw,python 侧零 G2P 复刻)。只 dump 不渲染。
        if std::env::var("UTAI_E1_DUMP_V2IN").is_ok() {
            let evts_dump: Vec<ScoreEvt> = sj
                .triples
                .iter()
                .map(|t| ScoreEvt {
                    lyric: &t.lyric,
                    note_num: t.note_num,
                    frames: t.frames,
                    lang: Lang::Ja,
                    phoneme_input: None,
                })
                .collect();
            let a = build_arrays_daw(&evts_dump, &NoDicts).unwrap();
            let j = serde_json::json!({
                "phonemes": a.phonemes, "phone_dur": a.phone_dur, "note_pitch": a.note_pitch,
                "note_dur": a.note_dur, "note_to_phone": a.note_to_phone,
            });
            std::fs::write(work.join(format!("{seg}_v2in.json")), serde_json::to_string(&j).unwrap())
                .unwrap();
            eprintln!("[e1] {seg}: v2in dumped ({} phones)", a.phonemes.len());
            continue;
        }
        // S72 K′(V2)臂:UTAI_E1_V2CV=cv npy 目录(python v2 推理产 {seg}_{TAG}_cv{dim}.npy)
        // + UTAI_E1_V2TAG(臂名)+ UTAI_E1_V2F0=knob|param|nat(net_g 的 f0 流;arms 过滤用 "V")
        let v2dir = std::env::var("UTAI_E1_V2CV").ok();
        let v2tag = std::env::var("UTAI_E1_V2TAG").unwrap_or_else(|_| "V2".into());
        let v2f0 = std::env::var("UTAI_E1_V2F0").unwrap_or_else(|_| "knob".into());

        let src = crate::audio::load_audio(&work.join(format!("{seg}_src48k.wav"))).unwrap();
        assert_eq!(src.channels, 1, "{seg}: src must be mono");
        // ── 天然流提取(每段一次;不加翻唱 0.5s pad——对齐优先,切口本就在休止里)──
        let wav16k = features::resample(&src.samples, src.sample_rate, f0mod::RMVPE_SR);
        let nat_f0 = f0mod::rmvpe_detect_chunked(
            &engine, &rmvpe, &mel, &wav16k, f0mod::SOVITS_RMVPE_THRESHOLD,
        )
        .unwrap();
        // RVC 翻唱口径的天然 f0 单独一份(与 SoVITS 差两点:48Hz filtfilt 高通 + 阈值 0.03。
        // 审查 confirmed:若沿用 0.05,salience 落在 [0.03,0.05) 的边缘帧(音头/清浊边界)会在
        // C 臂被多判无声而 A 臂有声——voicing 门限混成第二变量)。cv 不分家:高通对本素材
        // <48Hz 能量占比 ≈-62dB,对 ContentVec 影响可忽略(审查量化过),统一用未滤波 wav16k。
        let wav16k_hp = features::highpass_48hz_16k(&wav16k).unwrap();
        let nat_f0_rvc = f0mod::rmvpe_detect_chunked(
            &engine, &rmvpe, &mel, &wav16k_hp, f0mod::RVC_RMVPE_THRESHOLD,
        )
        .unwrap();
        let nat_cv768 = features::contentvec_extract(&engine, &cv768, &wav16k, 768).unwrap();
        let nat_cv256 = features::contentvec_extract(&engine, &cv256, &wav16k, 256).unwrap();
        eprintln!(
            "[e1] {seg}: score T50={} | natural cv768={} cv256={} rows, f0={} frames(100fps) | wav {:.2}s",
            total,
            nat_cv768.nrows(),
            nat_cv256.nrows(),
            nat_f0.len(),
            src.samples.len() as f32 / src.sample_rate as f32
        );

        let save = |name: String, r: &SynthesisResult| {
            assert!(!r.audio.iter().any(|x| x.is_nan()), "{name}: NaN in output");
            let peak = r.audio.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            assert!(peak > 1e-3, "{name}: silent output (peak {peak})");
            write_wav16(&work.join(format!("{name}.wav")), &r.audio, r.sample_rate);
            eprintln!("[e1]   {name}.wav  {:.2}s  peak={peak:.3}", r.audio.len() as f32 / r.sample_rate as f32);
        };
        let exists = |name: &String| work.join(format!("{name}.wav")).exists();

        // ── SoVITS 模型臂 ──
        let sovits_models: [(&str, PathBuf, &str, &Array2<f32>); 2] = [
            ("akiko256", sov_dir.join("akiko_320000.onnx"), &s2cv256, &nat_cv256),
            ("dxl41", sov_dir.join("Sovits4.1东雪莲主模型.onnx"), &s2cv768, &nat_cv768),
        ];
        for (model_name, model_path, s2cv_sid, nat_cv) in &sovits_models {
            if !enabled(&models_filter, model_name) {
                continue;
            }
            assert!(model_path.exists(), "missing {}", model_path.display());
            let sc = read_sidecar(model_path);
            let dim = match sc["speech_encoder"].as_str().expect("speech_encoder") {
                "vec768l12" => 768usize,
                "vec256l9" => 256usize,
                other => panic!("unsupported speech_encoder {other}"),
            };
            let sample_rate = sc["sample_rate"].as_u64().expect("sample_rate") as u32;
            let hop_size = sc["hop_size"].as_u64().unwrap_or(512) as usize;
            let min_frames = sc["min_frames"].as_u64().unwrap_or(6) as usize;
            let inputs_list = sc.get("inputs").and_then(|v| v.as_array());
            let vol_embedding = inputs_list
                .map(|l| l.iter().any(|v| v.as_str() == Some("vol")))
                .unwrap_or_else(|| sc.get("vol_embedding").and_then(|v| v.as_bool()).unwrap_or(false));
            let feed_uv = inputs_list
                .map(|l| l.iter().any(|v| v.as_str() == Some("uv")))
                .unwrap_or(true);
            let unit_interpolate_mode =
                sc.get("unit_interpolate_mode").and_then(|v| v.as_str()).unwrap_or("left").to_string();
            let voice_sid = engine.load_model_with(model_path, false).unwrap();
            let m = sovits::SovitsModel {
                engine: &engine,
                voice_session: &voice_sid,
                contentvec_session: if dim == 768 { &cv768 } else { &cv256 },
                rmvpe_session: &rmvpe,
                mel_filters: &mel,
                cluster: None,
                diffusion: None,
                vocoder: None,
                f0_predictor_session: None,
                sample_rate,
                hop_size,
                features_dim: dim,
                vol_embedding,
                phase_bins: None,
                f0d_cond_channels: None,
                feed_uv,
                spk_mix: None,
                unit_interpolate_mode,
                noise_channels: sidecar_noise_channels(&sc),
                min_frames,
            };
            eprintln!("[e1] {seg}/{model_name}: dim={dim} sr={sample_rate} vol={vol_embedding} mode={}", m.unit_interpolate_mode);
            let noop = |_: f32| {};
            let no_cancel = || false;
            let arm = |a: &str| format!("{seg}__{a}__{model_name}");

            if enabled(&arms_filter, "A") && !exists(&arm("A_natCV_natF0")) {
                let t0 = Instant::now();
                let r = sovits::run_pipeline(&m, &src, &sopts, None, &noop, &no_cancel).unwrap();
                save(arm("A_natCV_natF0"), &r);
                eprintln!("[e1]     ({:.1}s)", t0.elapsed().as_secs_f64());
            }
            if enabled(&arms_filter, "D") && !exists(&arm("D_s2cv_paramF0")) {
                let t0 = Instant::now();
                let r = render_score_sovits(
                    &m, s2cv_sid, &evts, dim, CV_SPEAKER, &NoDicts, &sopts, flat_vol, 0, 0,
                    Some(&vf0), None, None, &no_cancel, &noop,
                )
                .unwrap();
                save(arm("D_s2cv_paramF0"), &r);
                eprintln!("[e1]     ({:.1}s)", t0.elapsed().as_secs_f64());
            }
            if enabled(&arms_filter, "B") && !exists(&arm("B_natCV_paramF0")) {
                let t0 = Instant::now();
                let r = e1_render_sovits(
                    &m, &evts, dim, &CvSrc::Natural(nat_cv), &F0Src::Param(&vf0), &sopts, flat_vol,
                )
                .unwrap();
                save(arm("B_natCV_paramF0"), &r);
                eprintln!("[e1]     ({:.1}s)", t0.elapsed().as_secs_f64());
            }
            if let Some(vk) = &vf0_k {
                let karm = format!("{ktag}_s2cv_knobF0");
                if enabled(&arms_filter, "K") && !exists(&arm(&karm)) {
                    let t0 = Instant::now();
                    let r = render_score_sovits(
                        &m, s2cv_sid, &evts, dim, CV_SPEAKER, &NoDicts, &sopts, flat_vol, 0, 0,
                        Some(vk), None, None, &no_cancel, &noop,
                    )
                    .unwrap();
                    save(arm(&karm), &r);
                    eprintln!("[e1]     ({:.1}s)", t0.elapsed().as_secs_f64());
                }
            }
            if enabled(&arms_filter, "C") && !exists(&arm("C_s2cv_natF0")) {
                let t0 = Instant::now();
                let r = e1_render_sovits(
                    &m, &evts, dim, &CvSrc::S2cv(s2cv_sid), &F0Src::Natural(&nat_f0), &sopts, flat_vol,
                )
                .unwrap();
                save(arm("C_s2cv_natF0"), &r);
                eprintln!("[e1]     ({:.1}s)", t0.elapsed().as_secs_f64());
            }
            if let Some(dir) = &v2dir {
                let p = PathBuf::from(dir).join(format!("{seg}_{v2tag}_cv{dim}.npy"));
                let v2arm = format!("{v2tag}_v2cv");
                if p.exists() && enabled(&arms_filter, "V") && !exists(&arm(&v2arm)) {
                    let cv: Array2<f32> = ndarray_npy::read_npy(&p).unwrap();
                    let f0src = match v2f0.as_str() {
                        "nat" => F0Src::Natural(&nat_f0),
                        "param" => F0Src::Param(&vf0),
                        _ => F0Src::Param(vf0_k.as_ref().expect("V2F0=knob 需 {seg}_score_{TAG}.json 在场")),
                    };
                    let t0 = Instant::now();
                    let r = e1_render_sovits(&m, &evts, dim, &CvSrc::Natural(&cv), &f0src, &sopts, flat_vol)
                        .unwrap();
                    save(arm(&v2arm), &r);
                    eprintln!("[e1]     ({:.1}s)", t0.elapsed().as_secs_f64());
                }
            }
            engine.unload_model(&voice_sid);
        }

        // ── RVC 模型臂(lengv2.3,检索关)──
        if enabled(&models_filter, "rvcleng") {
            let model_path = rvc_dir.join("lengv2.3.onnx");
            assert!(model_path.exists(), "missing {}", model_path.display());
            let sc = read_sidecar(&model_path);
            let dim = sc["features_dim"].as_u64().expect("features_dim") as usize;
            let sample_rate = sc["sample_rate"].as_u64().expect("sample_rate") as u32;
            let min_frames = sc["min_frames"].as_u64().unwrap_or(12) as usize;
            let voice_sid = engine.load_model_with(&model_path, false).unwrap();
            let m = rvc::RvcModel {
                engine: &engine,
                voice_session: &voice_sid,
                contentvec_session: if dim == 768 { &cv768 } else { &cv256 },
                rmvpe_session: &rmvpe,
                mel_filters: &mel,
                index: None,
                sample_rate,
                features_dim: dim,
                spk_mix: None,
                noise_channels: sidecar_noise_channels(&sc),
                min_frames,
            };
            let s2cv_sid: &str = if dim == 768 { &s2cv768 } else { &s2cv256 };
            let nat_cv: &Array2<f32> = if dim == 768 { &nat_cv768 } else { &nat_cv256 };
            eprintln!("[e1] {seg}/rvcleng: dim={dim} sr={sample_rate} index=off");
            let noop = |_: f32| {};
            let no_cancel = || false;
            let arm = |a: &str| format!("{seg}__{a}__rvcleng");

            if enabled(&arms_filter, "A") && !exists(&arm("A_natCV_natF0")) {
                let t0 = Instant::now();
                let r = rvc::run_pipeline(&m, &src, &ropts, None, &noop, &no_cancel).unwrap();
                save(arm("A_natCV_natF0"), &r);
                eprintln!("[e1]     ({:.1}s)", t0.elapsed().as_secs_f64());
            }
            if enabled(&arms_filter, "D") && !exists(&arm("D_s2cv_paramF0")) {
                let t0 = Instant::now();
                let r = render_score_rvc(
                    &m, s2cv_sid, &evts, dim, CV_SPEAKER, &NoDicts, &ropts, 0, 0, Some(&vf0), None,
                    None, &no_cancel, &noop,
                )
                .unwrap();
                save(arm("D_s2cv_paramF0"), &r);
                eprintln!("[e1]     ({:.1}s)", t0.elapsed().as_secs_f64());
            }
            if enabled(&arms_filter, "B") && !exists(&arm("B_natCV_paramF0")) {
                let t0 = Instant::now();
                let r = e1_render_rvc(&m, &evts, dim, &CvSrc::Natural(nat_cv), &F0Src::Param(&vf0), &ropts)
                    .unwrap();
                save(arm("B_natCV_paramF0"), &r);
                eprintln!("[e1]     ({:.1}s)", t0.elapsed().as_secs_f64());
            }
            if let Some(vk) = &vf0_k {
                let karm = format!("{ktag}_s2cv_knobF0");
                if enabled(&arms_filter, "K") && !exists(&arm(&karm)) {
                    let t0 = Instant::now();
                    let r = render_score_rvc(
                        &m, s2cv_sid, &evts, dim, CV_SPEAKER, &NoDicts, &ropts, 0, 0, Some(vk),
                        None, None, &no_cancel, &noop,
                    )
                    .unwrap();
                    save(arm(&karm), &r);
                    eprintln!("[e1]     ({:.1}s)", t0.elapsed().as_secs_f64());
                }
            }
            if enabled(&arms_filter, "C") && !exists(&arm("C_s2cv_natF0")) {
                let t0 = Instant::now();
                let r = e1_render_rvc(
                    &m, &evts, dim, &CvSrc::S2cv(s2cv_sid), &F0Src::Natural(&nat_f0_rvc), &ropts,
                )
                .unwrap();
                save(arm("C_s2cv_natF0"), &r);
                eprintln!("[e1]     ({:.1}s)", t0.elapsed().as_secs_f64());
            }
            if let Some(dir) = &v2dir {
                let p = PathBuf::from(dir).join(format!("{seg}_{v2tag}_cv{dim}.npy"));
                let v2arm = format!("{v2tag}_v2cv");
                if p.exists() && enabled(&arms_filter, "V") && !exists(&arm(&v2arm)) {
                    let cv: Array2<f32> = ndarray_npy::read_npy(&p).unwrap();
                    let f0src = match v2f0.as_str() {
                        "nat" => F0Src::Natural(&nat_f0_rvc),
                        "param" => F0Src::Param(&vf0),
                        _ => F0Src::Param(vf0_k.as_ref().expect("V2F0=knob 需 {seg}_score_{TAG}.json 在场")),
                    };
                    let t0 = Instant::now();
                    let r = e1_render_rvc(&m, &evts, dim, &CvSrc::Natural(&cv), &f0src, &ropts).unwrap();
                    save(arm(&v2arm), &r);
                    eprintln!("[e1]     ({:.1}s)", t0.elapsed().as_secs_f64());
                }
            }
            engine.unload_model(&voice_sid);
        }
    }
    eprintln!("[e1] all requested arms done -> {}", work.display());
}
