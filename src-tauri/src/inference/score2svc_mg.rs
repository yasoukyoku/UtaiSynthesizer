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
// S90: the probe now resolves through the SHIPPED dictionaries (GlobalDicts), not the
// dictionary-free JA provider — an English/Chinese score needs stage1, and JA never
// consults a dictionary either way, so the JA arms are unchanged.
use super::super::g2p_alias;
use super::super::score2cv::{is_nucleus_phone, ArticulationTiming};
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
    /// S90: HONOURED now (it used to be dumped and ignored, with every probe render forced to JA —
    /// so an English score silently ran through the ja tables and the whole word/ARPABET layer was
    /// unreachable from this probe). Unknown ids fall back to JA, the historical default.
    lang: i64,
    /// S86: optional §3.7 traditional-phoneme override. With whitespace it is RAW phones, which lets
    /// one build render both arms of an A/B (e.g. 「に」 as `n i` vs `ɲ i`) from the same binary —
    /// a true controlled comparison instead of flipping a constant and rebuilding between takes.
    #[serde(default, rename = "phonemeInput")]
    phoneme_input: Option<String>,
}

/// Default = the 鹅妈妈 dump. `UTAI_MG_SCORE=<path>` points at any score JSON in the same shape
/// (S86: purpose-built A/B scores live beside it), so a probe score never overwrites the real dump.
fn load_score() -> ScoreJson {
    // S90: point the G2P at the shipped dictionaries (the command layer does this from the data dir).
    // Until this probe honoured `lang` every score ran as JA, which needs no dictionary file at all —
    // the first English score otherwise dies with VOCAL_DICT_MISSING inside the render.
    super::g2p::set_dict_dir(Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/dictionaries"));
    let p = std::env::var("UTAI_MG_SCORE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| Path::new(WORK).join("probe").join("mg_score.json"));
    let s = std::fs::read_to_string(&p).unwrap_or_else(|e| {
        panic!("missing {} ({e}) — run dump_mg_notes then mgScoreDump.test.ts first", p.display())
    });
    serde_json::from_str(&s).unwrap()
}

/// S91: `UTAI_MG_SET=arpasing|xsampa|vccv` renders the probe score as a UTAU ALIAS score — the only
/// way to hear a real CVVC/VCCV UST through the production pipeline before shipping it (S85 rule 4:
/// the user must never be the first to execute a new path). Absent/unknown → `words`, i.e. unchanged.
fn mg_phoneme_set() -> g2p_alias::PhonemeSet {
    g2p_alias::PhonemeSet::from_wire(std::env::var("UTAI_MG_SET").ok().as_deref())
}

fn to_evts(triples: &[TripleJson]) -> Vec<ScoreEvt<'_>> {
    let set = mg_phoneme_set();
    triples
        .iter()
        .map(|t| ScoreEvt {
            lyric: &t.lyric,
            note_num: t.note_num,
            frames: t.frames,
            lang: Lang::from_id(t.lang).unwrap_or(Lang::Ja),
            phoneme_input: t.phoneme_input.as_deref(),
            phoneme_set: set,
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

/// S85 cvfix 臂(用户提议「像 cover 一样只动 f0 不动 cv」):UTAI_MG_F0SHIFT=s →
/// cents 整体预移 s×100、渲染 transpose/range_shift 全 0 ⇒ f0 走移调而 cv/note_pitch
/// 钉在写谱位(build_note_hz 的 f0 用 RAW note_pitch 分组、cents 直进输出 Hz=单变量
/// 纯净,生产入口零镜像)。与 UTAI_MG_SHIFT 互斥。返回 (f0shift, 预移 cents 或 None)。
fn mg_f0shift_env(shift: i64, cents: &[f32]) -> (i64, Option<Vec<f32>>) {
    let f0shift: i64 =
        std::env::var("UTAI_MG_F0SHIFT").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    assert!(
        shift == 0 || f0shift == 0,
        "UTAI_MG_SHIFT and UTAI_MG_F0SHIFT are mutually exclusive"
    );
    if f0shift == 0 {
        return (0, None);
    }
    (f0shift, Some(cents.iter().map(|&c| c + (f0shift * 100) as f32).collect()))
}

/// S89 「自动音素时序」 probe switch: `UTAI_MG_PREROLL=0` renders the InNote arm, so the probe can
/// A/B the switch on real material with everything else held fixed. ONE reader — every place in this
/// file that needs the timing must call it, or a post-processing pass would shape one arm's audio
/// with the other arm's frame layout (review INFO).
fn mg_timing_env() -> ArticulationTiming {
    if std::env::var("UTAI_MG_PREROLL").map(|s| s != "0").unwrap_or(true) {
        ArticulationTiming::Auto
    } else {
        ArticulationTiming::InNote
    }
}

/// Filename marker for the non-default arm — WITHOUT it the two arms of an A/B silently overwrite
/// each other and you compare a file against itself (review INFO). Same posture as `mg_shift_tag`:
/// the production default leaves the name untouched.
fn mg_preroll_tag() -> &'static str {
    match mg_timing_env() {
        ArticulationTiming::Auto => "",
        ArticulationTiming::InNote => "_innote",
    }
}

/// The production ScoreToCV conditioning speaker. Mirrors `VocalRenderOptions::default().cv_speaker_id`
/// (commands/inference.rs) and the frontend `DEFAULT_VOCAL_PARAMS.speakerId`; the sidecar also carries it
/// as `default_speaker_id`. 49 = `kiritan` in the training speaker table — a JAPANESE singer.
const MG_CVSPK_PRODUCTION: i64 = 49;

/// S92 (5d 「非日语轨的日本味」) probe switch: `UTAI_MG_CVSPK=<0..76>` overrides the **ScoreToCV
/// conditioning speaker** — NOT the SVC voice's speaker (the voicebank is untouched, only the content
/// features change). The English singers in the training table are 32/33/34 = gt_EN-Alto-1/Alto-2/Tenor-1.
/// ONE reader (same posture as `mg_timing_env`), so the render arms and the filename tag can never
/// disagree about which speaker was actually fed.
fn mg_cvspk_env() -> i64 {
    std::env::var("UTAI_MG_CVSPK").ok().and_then(|s| s.parse().ok()).unwrap_or(MG_CVSPK_PRODUCTION)
}

/// Filename marker for a non-production cv speaker — without it two arms of the A/B overwrite each
/// other and you end up comparing a file with itself (S91: that exact trap cost a round).
fn mg_cvspk_tag(spk: i64) -> String {
    if spk == MG_CVSPK_PRODUCTION { String::new() } else { format!("_cvspk{spk}") }
}

/// cvfix 臂的探针侧逆变换(⚠与生产整段臂的唯一口径偏差:生产 inverse 在 peak-norm 之前,
/// 这里渲染已 norm 完才逆变换——电平语义微差,听感对比无碍,记档)。fed=移调后 note_hz
/// (生产整形三连同款)。
fn mg_cvfix_inverse(
    audio: Vec<f32>,
    sample_rate: u32,
    f0shift: i64,
    kappa: f32,
    evts: &[ScoreEvt<'_>],
    vf0: &VocalF0<'_>,
) -> Vec<f32> {
    // ⚠ must match the arm the audio was RENDERED with — the inverse is fed a per-frame f0 built
    // from this allocation, and the two arms lay the frames out differently.
    let arr = build_arrays_daw(evts, &super::g2p::GlobalDicts, mg_timing_env()).unwrap();
    let mut hz = build_note_hz(&arr, evts, 0, Some(vf0));
    zero_voiceless_frames(&mut hz, &arr);
    anchor_voiced_phone_f0(&mut hz, &arr);
    super::super::vocal_range::apply_inverse(
        audio,
        sample_rate,
        f0shift,
        kappa,
        Some((&hz, sample_rate as usize / 50)),
    )
    .unwrap()
}

#[test]
#[ignore]
fn mg_lane_dump() {
    let sj = load_score();
    let evts = to_evts(&sj.triples);
    let total: i64 = sj.triples.iter().map(|t| t.frames).sum();
    assert_eq!(sj.f0_cents.len() as i64, total, "f0 length vs Σframes");
    // S89: the lane dump is THE way to inspect an allocation without rendering, so it has to be
    // able to show BOTH articulation arms (`UTAI_MG_PREROLL=0` = the in-note arm).
    let timing = mg_timing_env();
    let arr = build_arrays_daw(&evts, &super::g2p::GlobalDicts, timing).unwrap();
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
    // per-event totals: under the in-note arm every event's phones must sum to that event's OWN
    // frames — the defining property of "nothing is borrowed across a note boundary". Dumped here so
    // the check can be made on REAL material without re-deriving anything outside production code.
    let mut per_evt = vec![0i64; sj.triples.len()];
    for (i, &e) in arr.evt.iter().enumerate() {
        per_evt[e] += arr.phone_dur[i];
    }
    let crossings: Vec<_> = (0..sj.triples.len())
        .filter(|&k| per_evt[k] != sj.triples[k].frames)
        .map(|k| serde_json::json!({ "k": k, "lyric": sj.triples[k].lyric, "own": sj.triples[k].frames, "got": per_evt[k] }))
        .collect();
    eprintln!(
        "[mg-lane] timing={timing:?}  phones={}  events crossing their note boundary = {}/{}",
        arr.phon.len(),
        crossings.len(),
        sj.triples.len()
    );
    let out = Path::new(WORK)
        .join("probe")
        .join(format!("mg_lane{}.json", mg_preroll_tag()));
    std::fs::write(
        &out,
        serde_json::to_string(&serde_json::json!({
            "tempo": sj.tempo, "total_frames": total, "notes": notes, "phones": phones,
            "f0_hz": hz, "timing": format!("{timing:?}"), "boundary_crossings": crossings,
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

/// S92k 分配器审计:把 `score2cv::audit` 跑在**真实谱面**上,按严重度排序打印 + 落盘 JSON。
/// 这是「不用耳朵一个词一个词找」的那台仪器的入口 —— 判据与自检见 `score2cv_audit.rs` 头注。
///
/// 两种模式:
///   ①默认 = 审计**当前代码**对这份谱的分配(`UTAI_MG_SCORE` / `UTAI_MG_SET` / `UTAI_MG_PREROLL` 同其它探针);
///   ②`UTAI_MG_AUDIT_LANE=<lane.json>` = 审计**一份存档泳道**(用今天的判据去审历史分配)。
///     ★这一路是仪器的验收凭证:拿 pre-S92 的泳道喂它,它必须**独立重新发现**当初那批静默丢音,
///     而我们对它一个字都没提示过。审计只读泳道的 phone/dur/evt 三列(其余列 audit 不看)。
///
/// Run:
///   $env:UTAI_MG_SCORE='<score.json>'; cargo test --lib inference::score2svc::mg_tests::mg_audit -- --ignored --nocapture
#[test]
#[ignore]
fn mg_audit() {
    use super::super::score2cv::audit;
    let sj = load_score();
    let evts = to_evts(&sj.triples);
    let timing = mg_timing_env();
    let resolved = super::super::g2p::resolve_score(&evts, &super::super::g2p::GlobalDicts).unwrap();

    let (arr, source, src_kind) = match std::env::var("UTAI_MG_AUDIT_LANE") {
        Ok(p) => {
            let v: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
            // 产物自述对断(S91 陈货规矩):存档泳道的臂/总帧必须与本次请求一致,否则响亮失败。
            assert_eq!(
                v["timing"].as_str().unwrap(),
                format!("{timing:?}"),
                "存档泳道的臂与本次请求不一致 —— 换 UTAI_MG_PREROLL 或换文件"
            );
            let total: i64 = sj.triples.iter().map(|t| t.frames).sum();
            assert_eq!(v["total_frames"].as_i64(), Some(total), "存档泳道与这份谱不是同一首");
            let mut a = build_arrays_daw(&evts, &super::super::g2p::GlobalDicts, timing).unwrap();
            let ph = v["phones"].as_array().unwrap();
            a.phon = ph
                .iter()
                .map(|p| audit::intern(p["phone"].as_str().unwrap()).expect("泳道里有词表外的音素"))
                .collect();
            a.phone_dur = ph.iter().map(|p| p["dur"].as_i64().unwrap()).collect();
            a.evt = ph.iter().map(|p| p["evt"].as_u64().unwrap() as usize).collect();
            // ★借帧账本是**当前代码**这次构建产生的,与这份存档泳道无关 —— 留着它会让「元音总
            //   损失帧数」拿今天的账去配历史的分配,是个静默的错数。存档泳道里重建不出账本
            //   (借进/借出在净额里不可分离),所以这条轴在这个模式下**不可用**,清空并声明。
            a.borrow_ledger.clear();
            a.in_note_alloc.clear();
            (a, format!("存档泳道 {p}"), audit::Source::ArchivedLane)
        }
        Err(_) => (
            build_arrays_daw(&evts, &super::super::g2p::GlobalDicts, timing).unwrap(),
            "当前代码".to_string(),
            audit::Source::Live,
        ),
    };

    let rep = audit::audit(&evts, &resolved, &arr, src_kind);
    eprintln!("[mg-audit] 来源 = {source}  臂 = {timing:?}");
    eprintln!("{}", rep.render(40));
    assert!(
        rep.unmodelled.is_empty(),
        "有未建模的发射路径 ⇒ 上面所有数字都不算数(仪器不许对没覆盖的东西宣称干净): {:?}",
        rep.unmodelled
    );
    assert_eq!(rep.conservation.0, rep.conservation.1, "守恒");

    let out = Path::new(WORK).join("probe").join(format!("mg_audit{}.json", mg_preroll_tag()));
    let rows: Vec<_> = rep
        .findings
        .iter()
        .map(|f| {
            serde_json::json!({
                "kind": f.kind.code(), "evt": f.evt, "lyric": f.lyric, "phone": f.phone,
                "position": f.position.code(), "lang": f.lang, "note_frames": f.note_frames,
                "actual": f.actual, "target_effective": f.target_effective,
                "target_measured": f.target_measured, "deficit": f.deficit(),
                "score_forced": f.score_forced, "ref_count": f.ref_count,
                "group_frames": f.group_frames,
            })
        })
        .collect();
    std::fs::write(
        &out,
        serde_json::to_string(&serde_json::json!({
            "source": source, "timing": format!("{timing:?}"),
            "events": rep.events, "phones_expected": rep.phones_expected,
            "phones_emitted": rep.phones_emitted, "displacement": rep.displacement,
            "findings": rows,
        }))
        .unwrap(),
    )
    .unwrap();
    eprintln!("[mg-audit] -> {}", out.display());
}

/// S92m 反投影对拍:**同一批音符、同一串音素 —— 参照给了几帧 / 我们给了几帧**,逐音素,零主观。
///
/// 用户手上没有真的快歌谱,而「我编一首」不能用来做听感判决。这条路绕开了那个问题:谱面是合成的,
/// 真值来自那位歌手这一句的标注;它不判听感,只做**覆盖** —— 补上「短音符/快段」这块用户唯一
/// 缺的取样面(反投影谱短音桶占 17-33%,用户那首歌只有 4.4%)。
///
/// ★★★S98 —— **真值面是哪一个,决定了这些数字能不能叫「真人」**,所以它现在必须随产物走。
/// 旧产物的 truth 列其实是 `npz.phone_dur` = 我们自己的对齐器(五个西语系语料还额外压过
/// `realign_mindur.py` 的 DVOW=DCONS=3 地板)⇒ 那时这台仪器是在拿我们跟我们自己比,而每一行
/// 都印着「真人」。`gen_reverse_score.py` 现在把 `_surface` 写进 truth.json:
///   `upstream` = 数据集自带标注,没经过我们任何一行对齐代码 ⇒ 可以叫「真人」;
///   `training` = 我们的对齐器 + 地板 ⇒ **本函数会把标签换成「训练面」**,别再误读。
/// 没有该字段的旧产物一律按 `training` 读(那就是它们的真实身份)。
/// ⚠ ja / zh 今天**没有**上游面,生成器会直接拒绝出 `upstream` 产物。
///
/// ⚠**诚实边界**:①英语训练语料本身就不快(最快乐句的音符中位也才 7 帧),所以这是「我们能拿到的
/// 最快的真人英语」,不是「真快歌」;②真人的音符分组里 20% 含多个音节,而我们的分配器把一个音符
/// 内除末核外的一切都当 medial —— 那部分差异是**我们的设计**,不是对齐误差,读数时要分开看。
///
/// ★S96 拍点轴(时长轴之外的第二半):**首核起点相对音符边界的偏移 —— 真人 vs 我们**,按
/// 「句首(前一事件是休止)/句中」分桶。审计件头注自认抓不到「时长对但落点错」,这条轴就是补它:
/// 真人侧 = 真值组内首核之前的时长和(前提 Σ真值帧 ≡ 音符帧,本函数里响亮断言);我们侧 = 首核
/// 音素的 wire 绝对起始帧(全音素游标累加,借帧落在前一事件地界时 onset 先行量为负)− 音符边界。
/// ⚠口径:「音符边界」= 反投影组头(对齐音素边界,gen_vowel_placement.py 头注★★),ja 六库的边界
/// 来自真 .mid = 拍点级;en/zh 系是标注惯例合成物,拍点真值另走层1 SV 对照 —— 这条轴回答的是
/// 「同一套边界下,真人怎么铺、我们怎么铺」,自洽且零主观。
///
/// Run:
///   $env:UTAI_MG_SCORE='<...>\gt_en_fast_score.json'; $env:UTAI_MG_TRUTH='<...>\gt_en_fast_truth.json'
///   cargo test --lib inference::score2svc::mg_tests::mg_truth_cmp -- --ignored --nocapture
#[test]
#[ignore]
fn mg_truth_cmp() {
    use super::super::score2cv::audit;
    let sj = load_score();
    let evts = to_evts(&sj.triples);
    let timing = mg_timing_env();
    let arr = build_arrays_daw(&evts, &super::super::g2p::GlobalDicts, timing).unwrap();
    let tp = std::env::var("UTAI_MG_TRUTH").expect("UTAI_MG_TRUTH=<truth.json>");
    let tv: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&tp).unwrap()).unwrap();
    let truth = tv["truth"].as_array().unwrap();
    assert_eq!(truth.len(), sj.triples.len(), "真值与谱面的事件数不一致 —— 不是同一份反投影");
    // ★S98:真值面身份。缺字段的旧产物 = training(那就是它们的真实身份,不是「未知」)。
    let surface = tv["_surface"].as_str().unwrap_or("training");
    let who = if surface == "upstream" { "真人" } else { "训练面" };
    eprintln!(
        "[truth] surface={surface} ⇒ 下面所有标着「{who}」的列都来自{}",
        if surface == "upstream" {
            "数据集自带标注(没经过我们的对齐器)"
        } else {
            "**我们自己的强制对齐 + min-dur 地板** —— 不是真人,别拿它论证任何地板"
        }
    );

    // 逐事件收集实发(与审计件同一条对齐规则:实发是期望的子序列)。S96:同步累加 wire 游标,
    // 每条实发多带自己的绝对起始帧 —— 拍点轴要的落点在生产里没有显式字段,只由顺序+时长隐式决定,
    // 这里的累加就是 mg_lane_dump 的同一条 cursor 规则(score2svc_mg.rs 泳道 frame0 的定义)。
    let mut ours: Vec<Vec<(&'static str, i64, i64)>> = vec![Vec::new(); sj.triples.len()];
    let mut cursor = 0i64;
    for i in 0..arr.phon.len() {
        if !matches!(arr.phon[i], "SP" | "AP") {
            ours[arr.evt[i]].push((arr.phon[i], arr.phone_dur[i], cursor));
        }
        cursor += arr.phone_dur[i];
    }
    // 音符边界 = 谱面 frames 顺序累加(音符边界的唯一表达,mg score JSON 无绝对起点字段)
    let mut note_frame0 = Vec::with_capacity(sj.triples.len());
    let mut cf = 0i64;
    for t in &sj.triples {
        note_frame0.push(cf);
        cf += t.frames;
    }

    // (position, 真人帧, 我们帧) —— position 走生产的 syllable_split,不另写一份
    let mut pairs: Vec<(&'static str, &'static str, i64, i64)> = Vec::new();
    let mut dropped: Vec<(usize, String, i64)> = Vec::new();
    // S96 拍点轴样本:(evt, 句首?, 真人首核偏移, 我们首核偏移, 我们首音素先行量)
    let mut beat: Vec<(usize, bool, i64, i64, i64)> = Vec::new();
    for (k, row) in truth.iter().enumerate() {
        let exp = row.as_array().unwrap();
        if exp.is_empty() {
            continue;
        }
        let toks: Vec<&'static str> = exp
            .iter()
            .map(|e| audit::intern(e[0].as_str().unwrap()).expect("真值里有词表外的音素"))
            .collect();
        let (onset_end, nuc) = super::super::score2cv::syllable_split_for_audit(&toks);
        // ★拍点轴前提,响亮断言:真值组内音素帧和 ≡ 音符帧(五语 2034 音符核对过 0 例外;
        //   这里钉死,防将来反投影生成器漂移后 cumsum 落点悄悄失义)
        let tsum: i64 = exp.iter().map(|e| e[1].as_i64().unwrap()).sum();
        assert_eq!(
            tsum, sj.triples[k].frames,
            "evt {k}: Σ真值帧 {tsum} != 音符帧 {} —— 拍点轴前提被打破,先查反投影生成器",
            sj.triples[k].frames
        );
        // ★S96d (review): a 0-frame triple would desync the two cursors silently (the note advances
        // the boundary walk but emits no phone) — the shipped generator skips those, so make the
        // assumption LOUD instead of leaving a ±1 drift that no assertion can see.
        assert!(sj.triples[k].frames > 0, "evt {k}: 0-frame sung triple breaks the beat-axis cursors");
        let truth_off: i64 = exp[..onset_end].iter().map(|e| e[1].as_i64().unwrap()).sum();
        let phrase_initial = k == 0 || truth[k - 1].as_array().is_none_or(|a| a.is_empty());
        let has_first_nucleus = super::super::score2cv::is_nucleus_phone(toks[onset_end]);
        let mut ours_nuc_off: Option<i64> = None;
        let mut ours_lead: Option<i64> = None;
        let mut gi = 0usize;
        for (i, &t) in toks.iter().enumerate() {
            let singer = exp[i][1].as_i64().unwrap();
            let pos = if i == nuc {
                "nucleus"
            } else if i < onset_end {
                "onset"
            } else if i > nuc {
                "coda"
            } else {
                "medial"
            };
            if gi < ours[k].len() && ours[k][gi].0 == t {
                if i == 0 {
                    ours_lead = Some(ours[k][gi].2 - note_frame0[k]);
                }
                if i == onset_end && has_first_nucleus {
                    ours_nuc_off = Some(ours[k][gi].2 - note_frame0[k]);
                }
                pairs.push((t, pos, singer, ours[k][gi].1));
                gi += 1;
            } else {
                dropped.push((k, t.to_string(), singer));
            }
        }
        if let (Some(no), Some(lead)) = (ours_nuc_off, ours_lead) {
            beat.push((k, phrase_initial, truth_off, no, lead));
        }
    }

    let stat = |label: &str, sel: &dyn Fn(&(&str, &str, i64, i64)) -> bool| {
        let s: Vec<_> = pairs.iter().filter(|p| sel(p)).collect();
        if s.is_empty() {
            return;
        }
        let mut d: Vec<i64> = s.iter().map(|p| p.3 - p.2).collect();
        d.sort_unstable();
        let short = s.iter().filter(|p| p.3 < p.2).count();
        let much = s.iter().filter(|p| p.3 * 5 < p.2 * 3).count();
        eprintln!(
            "  {label:<10} n={:<5} 差中位 {:>+3}  比{who}短 {:>3.0}%  短过 40% 的 {:>3.0}%",
            s.len(), d[d.len() / 2], 100.0 * short as f64 / s.len() as f64,
            100.0 * much as f64 / s.len() as f64
        );
    };
    eprintln!("[mg-truth] 对拍 {} 个音素,我们丢掉 {}", pairs.len(), dropped.len());
    for (k, t, sd) in dropped.iter().take(8) {
        eprintln!("   丢音 evt {k} {t} —— {who}给了 {sd} 帧");
    }
    stat("全部", &|_| true);
    for p in ["onset", "medial", "nucleus", "coda"] {
        stat(p, &|x| x.1 == p);
    }
    // 最系统性的偏差(样本 ≥8)
    let mut by: std::collections::HashMap<(&str, &str), Vec<i64>> = std::collections::HashMap::new();
    for p in &pairs {
        by.entry((p.0, p.1)).or_default().push(p.3 - p.2);
    }
    let mut rows: Vec<(f64, &str, &str, usize)> = by
        .iter()
        .filter(|(_, v)| v.len() >= 8)
        .map(|((t, pos), v)| (v.iter().sum::<i64>() as f64 / v.len() as f64, *t, *pos, v.len()))
        .collect();
    rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    eprintln!("  ── 最系统性的偏差 ──");
    for (m, t, pos, n) in rows.iter().take(6) {
        eprintln!("   {t:<4} {pos:<8} 平均比{who}短 {:>4.1} 帧 (n={n})", -m);
    }
    for (m, t, pos, n) in rows.iter().rev().take(3) {
        eprintln!("   {t:<4} {pos:<8} 平均比{who}长 {m:>4.1} 帧 (n={n})");
    }

    // ── S96 拍点轴:首核落点 真人 vs 我们(口径见头注;差 = 我们 − 真人,正 = 我们更晚)──
    let beat_stat = |label: &str, sel: &dyn Fn(&(usize, bool, i64, i64, i64)) -> bool| {
        let s: Vec<_> = beat.iter().filter(|b| sel(b)).collect();
        if s.is_empty() {
            return;
        }
        let mut d: Vec<i64> = s.iter().map(|b| b.3 - b.2).collect();
        d.sort_unstable();
        let within1 = d.iter().filter(|&&x| x.abs() <= 1).count();
        let mut troff: Vec<i64> = s.iter().map(|b| b.2).collect();
        troff.sort_unstable();
        let mut lead: Vec<i64> = s.iter().map(|b| b.4).collect();
        lead.sort_unstable();
        eprintln!(
            "  {label:<14} n={:<4} {who}首核偏移 p50={:>2}  差(我−参照) p25/p50/p75 = {:>+3}/{:>+3}/{:>+3}  |差|≤1 {:>3.0}%  首音素先行 p50={:>+3}",
            s.len(), troff[troff.len() / 2],
            d[d.len() / 4], d[d.len() / 2], d[3 * d.len() / 4],
            100.0 * within1 as f64 / s.len() as f64,
            lead[lead.len() / 2],
        );
    };
    eprintln!("  ── S96 拍点轴(首核起点 − 音符边界;{who}=组内 cumsum,我们=wire 游标)──");
    // ★S96d (review): the sample EXCLUDES notes whose first phone or first nucleus we dropped —
    // and those are exactly the most-compressed notes, i.e. the ones most likely to attack early.
    // Printing the exclusion inline stops the next reader from taking the medians as complete
    // ("没量过" must never read as "量过没问题" — the same rule the distribution table follows).
    let sung = truth.iter().filter(|r| !r.as_array().unwrap().is_empty()).count();
    eprintln!(
        "  样本 {}/{} 个有声音符(差 {} 个:首音素或首核被我们丢弃/非核起首 ⇒ 落点无定义;\
         这些恰是最被压缩的音符,读数对『早唱』一侧偏保守)",
        beat.len(),
        sung,
        sung - beat.len()
    );
    beat_stat("全部", &|_| true);
    beat_stat("句首(休止后)", &|b| b.1);
    beat_stat("句中", &|b| !b.1);
    beat_stat("句中·带onset", &|b| !b.1 && b.2 > 0);
    let mut worst: Vec<_> = beat.iter().collect();
    worst.sort_by_key(|b| -(b.3 - b.2).abs());
    eprintln!("  ── 拍点差最大的音符 ──");
    for b in worst.iter().take(6) {
        eprintln!(
            "   evt {:<4} {} {who} +{} 我们 {:+} (差 {:+}) 先行 {:+}",
            b.0,
            if b.1 { "句首" } else { "句中" },
            b.2, b.3, b.3 - b.2, b.4
        );
    }
    assert!(pairs.len() > 500, "对拍样本太少,是不是喂错了真值?");
    assert_eq!(
        arr.phone_dur.iter().sum::<i64>(),
        sj.triples.iter().map(|t| t.frames).sum::<i64>(),
        "守恒"
    );
}

/// S92 (5d 「非日语轨的日本味」) 的**数值前置件**:同一份乐谱数组喂 ScoreToCV,在
/// (speaker_id, lang_id) 网格上比较 cv 输出。它回答的只有「这条条件轴活着吗、有多大、差异落在
/// 元音还是辅音上」——**不回答「哪个更好」**(cv 域一切度量与耳朵解耦=度量坟场铁律,好坏只能耳测)。
///
/// ★测量底噪 = 恒 0:ScoreToCV 是确定性的 ⇒ 「基线组合再跑一遍」必须逐位相同,本测试把这条钉成
/// 断言(S86「同参双渲噪声底」的 cv 域版本)。没有这条,任何差值都可能只是噪声。
///
/// 生产口径 = speaker 恒 `MG_CVSPK_PRODUCTION`(49=kiritan,日语歌手)+ lang 逐 chunk 真喂;训练 manifest
/// 里 speaker→language 是个函数(每个 speaker 只唱一种语言)⇒ (49, en) 这个组合训练里一次都没出现过。
///
/// ★S92 实测结论(推翻了 S90 记的两条预期,别再照那个预期设计实验):①**speaker 空间没有「语言级」的
/// 大跳** —— 49→32 的位移(rel_rms .087)与「两个英语歌手之间」(32↔33 = .075)基本一样大;②**lang 不是
/// 死输入** —— 只换 lang(en→ja)也有 .081。这台仪器的产物是量级,不是好坏:cv 域一切度量与耳朵解耦。
///
/// Run(整曲,不切片;CPU EP):
///   $env:UTAI_MG_SCORE='<score.json>'; $env:UTAI_MG_OUTTAG='<tag>'
///   cargo test --lib inference::score2svc::mg_tests::mg_cv_cond_grid -- --ignored --nocapture
#[test]
#[ignore]
fn mg_cv_cond_grid() {
    let sj = load_score();
    let evts = to_evts(&sj.triples);
    let arr = build_arrays_daw(&evts, &super::g2p::GlobalDicts, mg_timing_env()).unwrap();
    let chunks = chunk_at_sp(&arr, 400);
    let total_t: usize = arr.phone_dur.iter().map(|&d| d.max(0) as usize).sum();
    let langs: Vec<i64> = {
        let mut v = arr.lang.clone();
        v.sort_unstable();
        v.dedup();
        v
    };

    // per-frame phone class, so a delta can be split by WHERE it lands (accent lives in the
    // consonants + vowel quality, not in the silence): 0 = SP/AP, 1 = nucleus (vowel), 2 = consonant.
    let mut cls: Vec<u8> = Vec::with_capacity(total_t);
    for i in 0..arr.phon.len() {
        let d = arr.phone_dur[i].max(0) as usize;
        let c = match arr.phon[i] {
            "SP" | "AP" => 0u8,
            p if is_nucleus_phone(p) => 1,
            _ => 2,
        };
        cls.extend(std::iter::repeat(c).take(d));
    }
    assert_eq!(cls.len(), total_t, "frame class map vs Σ phone_dur");

    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let dll = root.join("../runtime/ort/onnxruntime.dll");
    assert!(dll.exists(), "ORT dll missing at {}", dll.display());
    if let Ok(bld) = ort::init_from(&dll) {
        let _ = bld.commit();
    }
    let engine = OnnxEngine::new();
    engine.set_device(DeviceConfig::Cpu); // deterministic + no GPU setup in a test
    let aux = root.join("../data/models").join(crate::models::AUX_DIR_NAME);
    let s2cv = engine.load_model_with(&aux.join("score2cv_768.onnx"), false).unwrap();
    const DIM: usize = 768;

    // one arm = (speaker, optional lang override). `None` = every chunk keeps its OWN language = production.
    let run = |spk: i64, lang: Option<i64>| -> Vec<f32> {
        let mut out: Vec<f32> = Vec::with_capacity(total_t * DIM);
        for c in &chunks {
            let cv = run_score2cv(&engine, &s2cv, c, DIM, spk, lang.unwrap_or(c.lang_id)).unwrap();
            out.extend(cv.iter().copied());
        }
        assert_eq!(out.len(), total_t * DIM, "cv rows vs Σ phone_dur");
        out
    };

    // speaker ids ← the training table (Much-Better-S2H/processed/speakers.json).
    let arms: Vec<(&str, i64, Option<i64>)> = vec![
        ("PROD_49_kiritan_ja", 49, None),
        ("REPEAT_49_kiritan_ja", 49, None), // determinism floor — must be bit-identical to PROD
        ("EN_32_gt_EN-Alto-1", 32, None),
        ("EN_33_gt_EN-Alto-2", 33, None),
        ("EN_34_gt_EN-Tenor-1", 34, None),
        ("JA_42_gt_JA-Soprano-1", 42, None),
        ("JA_48_itako", 48, None),
        ("ZH_50_m4_Alto-1", 50, None),
        ("ZH_75_gt_ZH-Alto-1", 75, None),
        ("LANGONLY_49_ja", 49, Some(2)),
        ("LANGONLY_49_zh", 49, Some(0)),
        ("LANGONLY_49_en", 49, Some(1)),
    ];
    let t0 = Instant::now();
    let data: Vec<Vec<f32>> = arms
        .iter()
        .map(|&(n, s, l)| {
            eprintln!("[mg-cv] arm {n}: speaker_id={s} lang_id={l:?}");
            run(s, l)
        })
        .collect();
    let floor = data[0].iter().zip(&data[1]).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert_eq!(floor, 0.0, "ScoreToCV must be deterministic (measurement floor)");

    // relative RMS distance = RMS(Δ) / RMS(reference), per frame class, plus the mean per-frame cosine
    // (direction change) and max |Δ|. RMS-relative because cv is de-normalized (raw ContentVec scale).
    let stat = |a: &[f32], b: &[f32]| -> serde_json::Value {
        let mut sq = [0f64; 3];
        let mut ref_sq = [0f64; 3];
        let mut n = [0usize; 3];
        let (mut cos_sum, mut cos_n, mut maxabs) = (0f64, 0usize, 0f64);
        for t in 0..total_t {
            let c = cls[t] as usize;
            let (ra, rb) = (&a[t * DIM..(t + 1) * DIM], &b[t * DIM..(t + 1) * DIM]);
            let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
            for k in 0..DIM {
                let (x, y) = (ra[k] as f64, rb[k] as f64);
                let d = y - x;
                sq[c] += d * d;
                ref_sq[c] += x * x;
                maxabs = maxabs.max(d.abs());
                dot += x * y;
                na += x * x;
                nb += y * y;
            }
            n[c] += 1;
            if na > 0.0 && nb > 0.0 {
                cos_sum += dot / (na.sqrt() * nb.sqrt());
                cos_n += 1;
            }
        }
        let rel = |ks: &[usize]| -> f64 {
            let (d, r) = ks.iter().fold((0.0f64, 0.0f64), |acc, &c| (acc.0 + sq[c], acc.1 + ref_sq[c]));
            if r > 0.0 { (d / r).sqrt() } else { f64::NAN }
        };
        serde_json::json!({
            "rel_rms_all": rel(&[0, 1, 2]), "rel_rms_vowel": rel(&[1]),
            "rel_rms_consonant": rel(&[2]), "rel_rms_silence": rel(&[0]),
            "mean_frame_cosine": if cos_n > 0 { cos_sum / cos_n as f64 } else { f64::NAN },
            "max_abs_delta": maxabs,
            "frames": { "silence": n[0], "vowel": n[1], "consonant": n[2] },
        })
    };

    // every arm against the production baseline …
    let vs_prod: Vec<serde_json::Value> = (1..arms.len())
        .map(|i| {
            let s = stat(&data[0], &data[i]);
            eprintln!(
                "[mg-cv] {:<24} vs PROD: rel_rms all={:.4} vowel={:.4} cons={:.4}  cos={:.5}",
                arms[i].0,
                s["rel_rms_all"].as_f64().unwrap_or(f64::NAN),
                s["rel_rms_vowel"].as_f64().unwrap_or(f64::NAN),
                s["rel_rms_consonant"].as_f64().unwrap_or(f64::NAN),
                s["mean_frame_cosine"].as_f64().unwrap_or(f64::NAN),
            );
            serde_json::json!({ "arm": arms[i].0, "speaker_id": arms[i].1, "lang_override": arms[i].2, "stats": s })
        })
        .collect();
    // … plus the calibration pairs: how far apart are two speakers of the SAME language? That is the
    // yardstick the (49 vs 32) number has to be read against.
    let pairs: [(usize, usize); 6] = [(2, 3), (2, 4), (3, 4), (5, 6), (7, 8), (2, 5)];
    let calib: Vec<serde_json::Value> = pairs
        .iter()
        .map(|&(i, j)| {
            let s = stat(&data[i], &data[j]);
            eprintln!(
                "[mg-cv] CALIB {:<22} vs {:<22} rel_rms all={:.4} cos={:.5}",
                arms[i].0,
                arms[j].0,
                s["rel_rms_all"].as_f64().unwrap_or(f64::NAN),
                s["mean_frame_cosine"].as_f64().unwrap_or(f64::NAN),
            );
            serde_json::json!({ "a": arms[i].0, "b": arms[j].0, "stats": s })
        })
        .collect();

    let outtag = std::env::var("UTAI_MG_OUTTAG").map(|t| format!("_{t}")).unwrap_or_default();
    let out = Path::new(WORK).join("probe").join(format!("mg_cv_cond_grid{outtag}.json"));
    std::fs::create_dir_all(out.parent().unwrap()).unwrap();
    std::fs::write(
        &out,
        serde_json::to_string_pretty(&serde_json::json!({
            "score": std::env::var("UTAI_MG_SCORE").unwrap_or_else(|_| "<default mg_score.json>".into()),
            "phoneme_set": format!("{:?}", mg_phoneme_set()),
            "timing": format!("{:?}", mg_timing_env()),
            "notes": sj.triples.len(), "phones": arr.phon.len(), "frames": total_t,
            "chunks": chunks.len(), "langs_present": langs,
            "determinism_floor_max_abs": floor,
            "vs_production": vs_prod, "calibration_pairs": calib,
        }))
        .unwrap(),
    )
    .unwrap();
    eprintln!(
        "[mg-cv] {} arms x {} frames in {:.1}s -> {}",
        arms.len(),
        total_t,
        t0.elapsed().as_secs_f64(),
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
    let (f0shift, cents_shifted) = mg_f0shift_env(shift, cents);
    let vf0 = match &cents_shifted {
        Some(c) => VocalF0 { cents: c, voiced },
        None => vf0,
    };
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
    let cvspk = mg_cvspk_env();
    let r = render_score_rvc(
        &m, &s2cv768, &evts, 768, cvspk, &super::g2p::GlobalDicts, &ropts,
        ScoreShaping {
            consonant_emphasis_db: emph,
            consonant_valley_scale: valley,
            vowel_clarity: clarity,
            consonant_preroll: mg_timing_env() == ArticulationTiming::Auto,
        },
        tp, rs,
        Some(&vf0), None, None, &no_cancel, &no_prog,
    )
    .unwrap();
    let mut audio = r.audio;
    if f0shift != 0 && inverse {
        audio = mg_cvfix_inverse(audio, r.sample_rate, f0shift, kappa, &evts, &vf0);
    }
    let out_dir = Path::new(WORK).join("probe");
    std::fs::create_dir_all(&out_dir).unwrap();
    let ftag = if f0shift != 0 {
        format!("_f{f0shift}{}", if inverse { "" } else { "_raw" })
    } else {
        String::new()
    };
    let tag = format!(
        "{}{}{}{}{}{}{ftag}",
        if emph != DEFAULT_VOICELESS_ONSET_EMPHASIS_DB { format!("_e{emph}") } else { String::new() },
        if idx_ratio != 0.75 { format!("_i{idx_ratio}") } else { String::new() },
        if protect != RvcOptions::default().protect { format!("_p{protect}") } else { String::new() },
        if valley != DEFAULT_CONSONANT_VALLEY_SCALE { format!("_v{valley}") } else { String::new() },
        if !clarity { "_nc" } else { "" },
        mg_shift_tag(shift, inverse, kappa),
    ) + mg_preroll_tag()
        + &mg_cvspk_tag(cvspk);
    // S86: `UTAI_MG_OUTTAG` keeps two arms of the same score from overwriting each other.
    let outtag = std::env::var("UTAI_MG_OUTTAG").map(|t| format!("_{t}")).unwrap_or_default();
    let name = format!("mg_render_{a}_{b}_{mtag}{tag}{outtag}.wav");
    write_wav16(&out_dir.join(&name), &audio, r.sample_rate);
    eprintln!(
        "[mg] rendered triples[{a}..{b}] ({} frames): {:.2}s audio in {:.1}s wall -> probe\\{name}",
        f_end - f_start,
        audio.len() as f32 / r.sample_rate as f32,
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
    let (f0shift, cents_shifted) = mg_f0shift_env(shift, cents);
    let vf0 = match &cents_shifted {
        Some(c) => VocalF0 { cents: c, voiced },
        None => vf0,
    };

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
    let cvspk = mg_cvspk_env();
    let r = render_score_sovits(
        &m, &s2cv, &evts, dim, cvspk, &super::g2p::GlobalDicts, &sopts,
        crate::commands::inference::VOCAL_FLAT_VOL,
        ScoreShaping {
            consonant_emphasis_db: emph,
            consonant_valley_scale: valley,
            vowel_clarity: clarity,
            consonant_preroll: mg_timing_env() == ArticulationTiming::Auto,
        },
        tp, rs,
        Some(&vf0), None, None, &no_cancel, &no_prog,
    )
    .unwrap();
    let mut audio = r.audio;
    if f0shift != 0 && inverse {
        audio = mg_cvfix_inverse(audio, r.sample_rate, f0shift, kappa, &evts, &vf0);
    }
    let out_dir = Path::new(WORK).join("probe");
    std::fs::create_dir_all(&out_dir).unwrap();
    let ftag = if f0shift != 0 {
        format!("_f{f0shift}{}", if inverse { "" } else { "_raw" })
    } else {
        String::new()
    };
    let name = format!(
        "mg_render_{a}_{b}_{mtag}{}{ftag}{}{}.wav",
        mg_shift_tag(shift, inverse, kappa),
        mg_preroll_tag(),
        mg_cvspk_tag(cvspk)
    );
    write_wav16(&out_dir.join(&name), &audio, r.sample_rate);
    eprintln!(
        "[mg] sovits rendered triples[{a}..{b}] ({} frames): {:.2}s audio in {:.1}s wall -> probe\\{name}",
        f_end - f_start,
        audio.len() as f32 / r.sample_rate as f32,
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
    let arr = build_arrays_daw(&evts, &super::g2p::GlobalDicts, ArticulationTiming::Auto).unwrap();
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

/// S85 取证:cover dead-only 计划的 headless 复算——同 MixDown、同 sidecar 记录、生产同款
/// 探针链(16k+SOVITS 阈值 rmvpe@100fps),UTAI_MG_COVER_F0SHIFT 复现节点移调。打印每个
/// 死区(秒域)与其局部 shift,与 app 审计行逐指纹对拍。
#[test]
#[ignore]
fn mg_cover_range_replay() {
    let _ = tracing_subscriber::fmt().with_max_level(tracing::Level::DEBUG).try_init();
    // ⛔ The old default pointed inside the repo at a file that has never existed there (the
    // models live under D:\MyDev\TESTING\UtaiSynth2\models\), so running this probe without the
    // env var panicked on a missing file and read as "the probe is broken" (S145). Say what to
    // set instead.
    let model_json = std::env::var("UTAI_MG_RANGE_JSON").unwrap_or_else(|_| {
        r"D:\MyDev\TESTING\UtaiSynth2\models\sovits\Sovits4.1东雪莲主模型.json".into()
    });
    assert!(
        Path::new(&model_json).is_file(),
        "UTAI_MG_RANGE_JSON=<模型 .json 路径> — 缺省指向 {model_json},盘上没有。\
         设成一个带 vocal_range sidecar 的模型 json 再跑。"
    );
    let f0_shift: f32 = std::env::var("UTAI_MG_COVER_F0SHIFT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let cfg: crate::models::ModelConfig =
        serde_json::from_str(&std::fs::read_to_string(&model_json).unwrap()).unwrap();
    let r = super::super::vocal_range::speaker_range(&cfg, 0).expect("no vocal_range record");

    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let dll = root.join("../runtime/ort/onnxruntime.dll");
    assert!(dll.exists());
    if let Ok(bld) = ort::init_from(&dll) {
        let _ = bld.commit();
    }
    let engine = OnnxEngine::new();
    engine.set_device(DeviceConfig::Cpu);
    let aux = root.join("../data/models").join(crate::models::AUX_DIR_NAME);
    let rmvpe = engine.load_model_with(&aux.join("rmvpe_e2e.onnx"), false).unwrap();
    let rmvpe_mel: Array2<f32> = ndarray_npy::read_npy(&aux.join("rmvpe_mel_filters.npy")).unwrap();

    let src = crate::audio::load_audio(Path::new(WORK).join("未命名_MixDown.wav").as_path()).unwrap();
    let mono = crate::audio::resample::to_mono(&src);
    let wav16k =
        super::super::features::resample(&mono.samples, mono.sample_rate, super::super::f0::RMVPE_SR);
    let f0 = super::super::f0::rmvpe_detect_chunked(
        &engine,
        &rmvpe,
        &rmvpe_mel,
        &wav16k,
        super::super::f0::SOVITS_RMVPE_THRESHOLD,
    )
    .unwrap();
    let k = 2.0f32.powf(f0_shift / 12.0);
    let transposed: Vec<f32> = f0.iter().map(|v| v * k).collect();
    let (jobs, unfixable) =
        super::super::vocal_range::cover_dead_plan(&transposed, 100.0, &r);
    eprintln!(
        "[mg] cover dead-only replay: f0_shift={f0_shift} -> {} region(s), {} unfixable (usable [{:.0},{:.0}])",
        jobs.len(),
        unfixable.len(),
        r.usable.0,
        r.usable.1
    );
    for &(a, b) in &unfixable {
        eprintln!("[mg]   UNFIXABLE region {:.2}s..{:.2}s", a as f32 / 100.0, b as f32 / 100.0);
    }
    for j in &jobs {
        eprintln!(
            "[mg]   region {:.2}s..{:.2}s -> {:+} st",
            j.start as f32 / 100.0,
            j.end as f32 / 100.0,
            j.shift
        );
    }
}

/// S85e 冒烟:windowed-donor 全链行为验证(真 RVC 管线、CPU、秒域切片)。伪造「顶部截断」
/// bounds 记录让切片内高音区成死区,range 臂与 range=None 臂对拍三条硬不变量:
/// ①输出等长 ②差异样本占比 ∈ (0, 50%)=局部拼接而非整曲重着色 ③切片首尾各 1s 逐位不变
/// (死区居中时窗外必须 bit-identical)。UTAI_MG_SMOKE_SPAN="a..b"(秒,默认 70..82)。
#[test]
#[ignore]
fn mg_cover_deadonly_smoke() {
    let _ = tracing_subscriber::fmt().with_max_level(tracing::Level::INFO).try_init();
    let span = std::env::var("UTAI_MG_SMOKE_SPAN").unwrap_or_else(|_| "70..82".into());
    let (t0s, t1s) = span.split_once("..").expect("UTAI_MG_SMOKE_SPAN=a..b seconds");
    let (t0, t1): (f64, f64) = (t0s.parse().unwrap(), t1s.parse().unwrap());

    let full = crate::audio::load_audio(Path::new(WORK).join("未命名_MixDown.wav").as_path()).unwrap();
    let ch = full.channels.max(1) as usize;
    let nf = full.samples.len() / ch;
    let s0 = ((t0 * full.sample_rate as f64) as usize).min(nf) * ch;
    let s1 = ((t1 * full.sample_rate as f64) as usize).min(nf) * ch;
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
    let (model_path, index_path, _mtag) = mg_model_envs();
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
    let ropts = RvcOptions::default();
    let noop = |_: f32| {};
    let no_cancel = || false;

    // 对照:同参渲两次 base 自对拍——ORT CPU 归约序/线程分片可致运行间 fp 噪声;冒烟的
    // 窗外判据按此选「严格逐位」或「噪声底容差」(容差=对照差异 RMS ×8,兜阶段放大)。
    let t = Instant::now();
    let base = rvc::run_pipeline(&m, &src, &ropts, None, &noop, &no_cancel).unwrap();
    let t_base = t.elapsed().as_secs_f32();
    let base2 = rvc::run_pipeline(&m, &src, &ropts, None, &noop, &no_cancel).unwrap();
    let rms = |x: &[f32], y: &[f32]| {
        (x.iter().zip(y).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>()
            / x.len().max(1) as f64)
            .sqrt()
    };
    let noise_floor = rms(&base.audio, &base2.audio);
    // 顶部截断 bounds 记录(usable 顶 74/落点带 [40,70]):切片内 >D5 的持续高音成死区。
    let fake = super::super::vocal_range::SpeakerRange::bounds((36.0, 74.0), (40.0, 70.0));
    let t = Instant::now();
    let ext = rvc::run_pipeline(&m, &src, &ropts, Some(fake), &noop, &no_cancel).unwrap();
    let t_ext = t.elapsed().as_secs_f32();

    assert_eq!(ext.audio.len(), base.audio.len(), "①range 臂不得改输出长度");
    let n = base.audio.len();
    let sr = ext.sample_rate as usize;
    let head = rms(&base.audio[..sr], &ext.audio[..sr]);
    let tail = rms(&base.audio[n - sr..], &ext.audio[n - sr..]);
    let mid = rms(&base.audio, &ext.audio);
    eprintln!(
        "[mg] deadonly smoke: {:.1}s slice; base {t_base:.1}s vs ext {t_ext:.1}s (+{:.0}%); run-to-run noise rms {noise_floor:.2e}; head/tail/full diff rms {head:.2e}/{tail:.2e}/{mid:.2e}",
        (t1 - t0), (t_ext / t_base - 1.0) * 100.0
    );
    assert!(mid > noise_floor * 8.0 + 1e-9, "②拼接必须发生(死区计划空=探针或判据退化)");
    let tol = noise_floor * 8.0 + 1e-9;
    assert!(head <= tol, "③切片首 1s 窗外超噪声底({head:.2e} > {tol:.2e})——窗外被污染");
    assert!(tail <= tol, "③切片尾 1s 窗外超噪声底({tail:.2e} > {tol:.2e})——窗外被污染");
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
    // run_pipeline 里先于 coarse 量化乘到整轨(rvc.rs:256)= S85d 生产 donor 自递归
    // (f0_shift+=s)的同一数学;raw 臂(UTAI_MG_INVERSE=0)即到此为止=「模型在 shift 位唱」。
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
