// 乐谱导出(ust / ustx / midi)← 人声轨(vocal tracks)—— import.rs 的精确逆操作,round-trip 必须
// 闭合:这里写出的字节喂回 import(parse_ust / parse_ustx / parse_midi)要得到逐音符相等的
// tick/duration/pitch/lyric(ustx vibrato 各字段浮点近似相等)。测试用 import_score_file 读回验证。
//
// SCOPE:前端把每轨音符摊平到绝对 tick(工程时间线,480 ppq、tick 升序)后传入;这里只做共同预处理
// (防御性重叠截断)+ 格式序列化 + 写盘,不读任何工程状态。三种格式的天然缺口(格式语义所限,无法闭合):
//   - ust:格式没有拍号;歌词 "R"/"r"/空 会被任何 UTAU 方言(含我们的 parse_ust)读成休止。
//   - midi:bpm 量化到整数 µs/qn;拍号分母量化到最近的 2 的幂;空歌词重导入按 import 规则变占位「あ」。
//   - ustx:空/纯空白歌词重导入按 import 规则变占位「あ」(parse_ustx 的 lyric-less 兜底);
//     此外我们刻意不写 pitch 曲线(与导入对称,见 import.rs 头注释)。
//
// 错误约定(i18n 铁律):用户可见错误一律稳定 CODE —— EXPORT_SCORE_UNSUPPORTED / EXPORT_SCORE_EMPTY /
// EXPORT_SCORE_WRITE_FAIL: {detail},由前端统一映射成文案。

use serde::Deserialize;

/// 前端传入的一个音符。`tick` 是绝对工程 tick(480 ppq);进来时已按 tick 升序,这里仍防御性重排。
#[derive(Deserialize, Debug, Clone)]
pub struct ExportNote {
    pub tick: i64,
    pub duration: i64,
    pub pitch: i32, // MIDI note number
    pub lyric: String,
    pub velocity: i32, // 0-127,仅 midi NoteOn 使用;ust/ustx 忽略
    #[serde(default)]
    pub vibrato: Option<ExportVibrato>, // 仅 ustx 写出
}

/// 与 import.rs 的 ImportedVibrato 同形(camelCase 直连 TS 侧 VibratoSpec,无需再 shaping)。
#[derive(Deserialize, Debug, Clone, Copy)]
#[serde(rename_all = "camelCase")]
pub struct ExportVibrato {
    pub depth_cents: f32,
    pub freq_hz: f32,
    pub phase: f32,
    pub start_ms: f32,
    pub ease_in_ms: f32,
    pub ease_out_ms: f32,
}

#[derive(Deserialize, Debug)]
pub struct ExportTrack {
    pub name: String,
    pub notes: Vec<ExportNote>,
}

/// 我们的工程分辨率 —— 与 import.rs 的同名常量同值(那边是模块私有,这里镜像一份)。
const TICKS_PER_BEAT: i64 = 480;

#[tauri::command]
pub async fn export_score_files(
    format: String,
    path: String,
    tempo: f64,
    time_sig: [u32; 2],
    tracks: Vec<ExportTrack>,
) -> Result<(), String> {
    if !matches!(format.as_str(), "ust" | "ustx" | "midi") {
        return Err(format!("EXPORT_SCORE_UNSUPPORTED: {format}"));
    }
    // 共同预处理:排序 + 半开区间重叠截断(镜像 src/lib/vocalNotes.ts 的编辑期截断哲学)+ clamp,
    // 无音符轨剔除。前端编辑漏斗已保证单段内不重叠,但多段拼接可能重叠 —— 防御,不是热路径。
    let tracks: Vec<ExportTrack> = tracks
        .into_iter()
        .map(|t| ExportTrack { name: t.name, notes: sanitize_notes(t.notes) })
        .filter(|t| !t.notes.is_empty())
        .collect();
    if tracks.is_empty() {
        return Err("EXPORT_SCORE_EMPTY".to_string());
    }
    // bpm 防御性归一(镜像 import.rs map_vibrato 的兜底):非法值不该出现,但写盘侧绝不放 NaN 进文件。
    let bpm = if tempo.is_finite() && tempo > 0.0 { tempo } else { 120.0 };
    // 写盘/zip 是重活 → spawn_blocking(仓库惯例,见 storage.rs / midi_extract.rs)。
    tauri::async_runtime::spawn_blocking(move || -> Result<(), String> {
        match format.as_str() {
            "ust" => export_ust(&path, bpm, &tracks),
            "ustx" => export_ustx(&path, bpm, time_sig, &tracks),
            _ => export_midi(&path, bpm, time_sig, &tracks), // 开头已验证 format ∈ 三者
        }
    })
    .await
    .map_err(|e| format!("EXPORT_SCORE_WRITE_FAIL: join: {e}"))?
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────
// 共同预处理
// ─────────────────────────────────────────────────────────────────────────────────────────────────────

/// clamp(tick ≥ 0 / duration ≥ 1 / pitch 0-127)→ 按 tick 排序 → 半开区间重叠截断:
/// next.tick < cur.tick + cur.duration → cur 截到 next.tick;截完 ≤ 0(完全吞没)则丢弃 cur,
/// 并继续向前级联(一个长音符可能盖住多个后续音符)。空歌词音符照写,不发明占位。
fn sanitize_notes(mut notes: Vec<ExportNote>) -> Vec<ExportNote> {
    for n in &mut notes {
        n.tick = n.tick.max(0);
        n.duration = n.duration.max(1);
        n.pitch = n.pitch.clamp(0, 127);
    }
    notes.sort_by_key(|n| n.tick);
    let mut out: Vec<ExportNote> = Vec::with_capacity(notes.len());
    for n in notes {
        while let Some(prev) = out.last_mut() {
            if n.tick < prev.tick + prev.duration {
                prev.duration = n.tick - prev.tick;
                if prev.duration <= 0 {
                    out.pop(); // 前者被完全吞没 → 丢弃,再检查更前面的一个
                    continue;
                }
            }
            break;
        }
        out.push(n);
    }
    out
}

/// bpm 文本:f64 Display 本身就是最短 round-trip 表示(120.0 → "120",89.9 → "89.9"),
/// parse 回必然逐位相等 —— 整数值自然写成 "120",非整写最短小数,无需手工分支。
fn fmt_bpm(bpm: f64) -> String {
    format!("{bpm}")
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────
// ust(INI 式,Shift-JIS 优先;一轨一文件,多轨打 zip)
// ─────────────────────────────────────────────────────────────────────────────────────────────────────

/// 追加一个音符/休止节。节头从 [#0000] 连续 4 位编号 —— parse_ust 只认纯数字节头为音符,
/// {:04} 超过 9999 自动加宽,仍是纯数字。字段顺序与老 UTAU 输出一致(Length/Lyric/NoteNum/Intensity)。
fn push_ust_section(lines: &mut Vec<String>, idx: &mut usize, length: i64, lyric: &str, notenum: i32) {
    lines.push(format!("[#{:04}]", *idx));
    lines.push(format!("Length={length}"));
    lines.push(format!("Lyric={lyric}"));
    lines.push(format!("NoteNum={notenum}"));
    lines.push("Intensity=100".into());
    *idx += 1;
}

/// 组装一轨 ust 全文(行尾 \r\n —— UTAU 是 Windows 软件;parse_ust 对 \r\n 容忍)。
/// `charset` 只是写进 [#VERSION] 的声明行,本身纯 ASCII —— 两个分支下全文其余字节完全相同,
/// 所以编码可映射性判定可以在 "Shift-JIS 版全文" 上做(见 encode_ust)。
fn build_ust_text(charset: &str, track_name: &str, bpm: f64, notes: &[ExportNote]) -> String {
    let mut lines: Vec<String> = Vec::with_capacity(notes.len() * 6 + 10);
    lines.push("[#VERSION]".into());
    lines.push("UST Version1.2".into());
    lines.push(format!("Charset={charset}"));
    lines.push("[#SETTING]".into());
    lines.push(format!("Tempo={}", fmt_bpm(bpm)));
    lines.push("Tracks=1".into());
    lines.push(format!("ProjectName={track_name}"));
    lines.push("Mode2=True".into());
    // gap → Lyric=R 休止节(NoteNum=60 惯例 + Intensity=100):头部第一个音符前的空隙、音符间空隙
    // 都插 R —— 这正是 parse_ust cursor 累加(rest 只推进游标不出音符)的精确逆。
    let mut idx: usize = 0;
    let mut cursor: i64 = 0; // 绝对 tick
    for n in notes {
        if n.tick > cursor {
            push_ust_section(&mut lines, &mut idx, n.tick - cursor, "R", 60);
        }
        push_ust_section(&mut lines, &mut idx, n.duration, &n.lyric, n.pitch);
        cursor = n.tick + n.duration;
    }
    lines.push("[#TRACKEND]".into());
    lines.join("\r\n") + "\r\n"
}

/// 编码策略:完整最终文本(含轨名 ProjectName + 全部歌词,不只测歌词)先试 Shift-JIS ——
/// encode 无 unmappable(had_errors=false)→ 写 Shift-JIS + Charset=Shift-JIS(最大化老 UTAU 兼容);
/// 有任何映射不进的字符 → 落 UTF-8(无 BOM)+ Charset=UTF-8(decode_ust_bytes 两种都认)。
fn encode_ust(track_name: &str, bpm: f64, notes: &[ExportNote]) -> Vec<u8> {
    let sjis_text = build_ust_text("Shift-JIS", track_name, bpm, notes);
    let (bytes, _, had_errors) = encoding_rs::SHIFT_JIS.encode(&sjis_text);
    // had_errors 只报「映射不进」,不报「有损映射」:WHATWG Shift_JIS 把 ¥(U+00A5)→0x5C、
    // ‾(U+203E)→0x7E 当成功写出,读回就成了反斜杠/波浪号 —— 静默串改歌词(审计逮的)。
    // 所以 SJIS 分支必须再做一次 decode 往返比对,任何不等都落 UTF-8。
    if !had_errors && encoding_rs::SHIFT_JIS.decode(&bytes).0 == sjis_text {
        bytes.into_owned()
    } else {
        build_ust_text("UTF-8", track_name, bpm, notes).into_bytes()
    }
}

/// zip 成员名(不含扩展名):Windows 非法字符 <>:"/\|?* 与控制字符 → '_';trim 后空名 → "Track N"(1 起)。
fn sanitize_filename_component(raw: &str, index: usize) -> String {
    let cleaned: String = raw
        .chars()
        .map(|c| match c {
            '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*' => '_',
            c if c.is_control() => '_',
            c => c,
        })
        .collect();
    let trimmed = cleaned.trim();
    if trimmed.is_empty() { format!("Track {}", index + 1) } else { trimmed.to_string() }
}

/// 重名去重:大小写不敏感(zip 在 Windows 上解出来同目录会相撞),第二个起追加 " (2)" 式序号;
/// 追加后再撞(用户轨名本身就叫 "X (2)")继续 +1 直到唯一。
fn unique_member_name(raw: &str, index: usize, used: &mut std::collections::HashSet<String>) -> String {
    let base = sanitize_filename_component(raw, index);
    let mut name = base.clone();
    let mut n = 1u32;
    while !used.insert(name.to_lowercase()) {
        n += 1;
        name = format!("{base} ({n})");
    }
    name
}

/// 一轨 → path 就是最终 .ust;多轨 → path 是 zip,内部每轨一个 `<sanitized轨名>.ust`。
/// zip 写法与 project.rs 的 .usp 保存一致(ZipWriter + SimpleFileOptions + Deflated)。
fn export_ust(path: &str, bpm: f64, tracks: &[ExportTrack]) -> Result<(), String> {
    if tracks.len() == 1 {
        let bytes = encode_ust(&tracks[0].name, bpm, &tracks[0].notes);
        return std::fs::write(path, bytes).map_err(|e| format!("EXPORT_SCORE_WRITE_FAIL: {e}"));
    }
    use std::io::Write;
    let file = std::fs::File::create(path).map_err(|e| format!("EXPORT_SCORE_WRITE_FAIL: {e}"))?;
    let mut zip = zip::ZipWriter::new(file);
    let opts: zip::write::SimpleFileOptions =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
    let mut used: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (i, t) in tracks.iter().enumerate() {
        let member = format!("{}.ust", unique_member_name(&t.name, i, &mut used));
        zip.start_file(member.clone(), opts).map_err(|e| format!("EXPORT_SCORE_WRITE_FAIL: zip {member}: {e}"))?;
        // ProjectName 里仍写原始轨名(sanitize 只为文件名服务,不改内容)。
        let bytes = encode_ust(&t.name, bpm, &t.notes);
        zip.write_all(&bytes).map_err(|e| format!("EXPORT_SCORE_WRITE_FAIL: {e}"))?;
    }
    zip.finish().map_err(|e| format!("EXPORT_SCORE_WRITE_FAIL: {e}"))?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────
// ustx(YAML,OpenUTAU 0.6)
// ─────────────────────────────────────────────────────────────────────────────────────────────────────

// 写出字段刻意最小化:parse_ustx / OpenUTAU(YamlDotNet)对缺字段都按默认值处理,多写反而引入
// 漂移面。legacy 顶层 bpm/beat_per_bar/beat_unit 三件套照写,兼容旧 OpenUTAU(parse_ustx 也认)。
#[derive(serde::Serialize)]
struct UstxOutRoot {
    name: String,
    ustx_version: String, // "0.6"(字符串,OpenUTAU 按版本串解析)
    resolution: i64,      // 480
    bpm: f64,
    beat_per_bar: u32,
    beat_unit: u32,
    tempos: Vec<UstxOutTempo>,
    time_signatures: Vec<UstxOutTimeSig>,
    tracks: Vec<UstxOutTrack>,
    voice_parts: Vec<UstxOutVoicePart>,
}

#[derive(serde::Serialize)]
struct UstxOutTempo {
    position: i64,
    bpm: f64,
}

#[derive(serde::Serialize)]
struct UstxOutTimeSig {
    bar_position: i64,
    beat_per_bar: u32,
    beat_unit: u32,
}

#[derive(serde::Serialize)]
struct UstxOutTrack {
    track_name: String,
}

#[derive(serde::Serialize)]
struct UstxOutVoicePart {
    name: String,
    track_no: usize,
    position: i64, // part 在工程里的绝对 tick 偏移 = 首音符绝对 tick
    notes: Vec<UstxOutNote>,
}

#[derive(serde::Serialize)]
struct UstxOutNote {
    position: i64, // PART-RELATIVE tick(绝对 tick − part.position,与 import 的逆一致)
    duration: i64,
    tone: i32,
    lyric: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    vibrato: Option<UstxOutVibrato>,
}

/// OpenUTAU vibrato(字段语义见 import.rs UstxVibrato 注释)。`in` 是 Rust 关键字 → serde rename。
/// drift / vol_link 我们的模型没有对应物,恒写 0(OpenUTAU 读侧的合法默认)。
#[derive(serde::Serialize)]
struct UstxOutVibrato {
    length: f32,
    period: f32,
    depth: f32,
    #[serde(rename = "in")]
    fade_in: f32,
    out: f32,
    shift: f32,
    drift: f32,
    vol_link: f32,
}

/// map_vibrato(import.rs)的精确逆:note_ms = duration/480*60000/bpm;vib_ms = note_ms − start_ms;
/// vib_ms ≤ 0 或 freq_hz ≤ 0 → 不写 vibrato。length = vib_ms/note_ms*100 clamp 到 (0,100];
/// period = 1000/freq_hz;in/out = ease/vib_ms*100;shift = phase*100。在无 clamp 触发的正常域内,
/// 与 map_vibrato 往返各字段恒等(浮点近似,测试卡 <0.05)。
fn unmap_vibrato(v: &ExportVibrato, note_dur_ticks: i64, bpm: f64) -> Option<UstxOutVibrato> {
    if !(v.freq_hz > 0.0) {
        return None;
    }
    let note_ms = note_dur_ticks as f64 / TICKS_PER_BEAT as f64 * 60_000.0 / bpm;
    if !(note_ms > 0.0) {
        return None;
    }
    let vib_ms = note_ms - v.start_ms as f64;
    if !(vib_ms > 0.0) {
        return None;
    }
    let length = (vib_ms / note_ms * 100.0).min(100.0);
    Some(UstxOutVibrato {
        length: length as f32,
        period: (1000.0 / v.freq_hz as f64) as f32,
        depth: v.depth_cents,
        fade_in: ((v.ease_in_ms as f64 / vib_ms * 100.0).clamp(0.0, 100.0)) as f32,
        out: ((v.ease_out_ms as f64 / vib_ms * 100.0).clamp(0.0, 100.0)) as f32,
        // phase 在我们引擎里是循环量(f0eval sin(2π(f·t+phase)),−0.25 ≡ +0.75),而 OpenUTAU 的
        // shift 域是 [0,100] 且反序列化时硬钳 —— 负 phase 直接 ×100 会被 OpenUTAU 钳成 0(丢相位)。
        // rem_euclid(1.0) 先包卷到 [0,1) 再放大,无损等价(审计逮的)。
        shift: ((v.phase as f64).rem_euclid(1.0) * 100.0) as f32,
        drift: 0.0,
        vol_link: 0.0,
    })
}

fn export_ustx(path: &str, bpm: f64, time_sig: [u32; 2], tracks: &[ExportTrack]) -> Result<(), String> {
    let beat_per_bar = time_sig[0].max(1);
    let beat_unit = time_sig[1].max(1);
    // 工程名 = 第一轨名(非空白),否则 "UTAI Export"。
    let project_name = tracks
        .first()
        .map(|t| t.name.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "UTAI Export".to_string());
    let root = UstxOutRoot {
        name: project_name,
        ustx_version: "0.6".to_string(),
        resolution: TICKS_PER_BEAT,
        bpm,
        beat_per_bar,
        beat_unit,
        tempos: vec![UstxOutTempo { position: 0, bpm }],
        time_signatures: vec![UstxOutTimeSig { bar_position: 0, beat_per_bar, beat_unit }],
        tracks: tracks.iter().map(|t| UstxOutTrack { track_name: t.name.clone() }).collect(),
        voice_parts: tracks
            .iter()
            .enumerate()
            .map(|(i, t)| {
                // part.position = 首音符绝对 tick;note.position 相对 part 重定基 —— parse_ustx 里
                // "绝对 = position + note.position, rebase 到首音符" 的精确逆(首音符 position 0)。
                let part_pos = t.notes.first().map(|n| n.tick).unwrap_or(0);
                UstxOutVoicePart {
                    name: t.name.clone(),
                    track_no: i,
                    position: part_pos,
                    notes: t
                        .notes
                        .iter()
                        .map(|n| UstxOutNote {
                            position: n.tick - part_pos,
                            duration: n.duration,
                            tone: n.pitch,
                            lyric: n.lyric.clone(),
                            vibrato: n.vibrato.as_ref().and_then(|v| unmap_vibrato(v, n.duration, bpm)),
                        })
                        .collect(),
                }
            })
            .collect(),
    };
    let yaml = serde_yaml_ng::to_string(&root).map_err(|e| format!("EXPORT_SCORE_WRITE_FAIL: yaml: {e}"))?;
    // OpenUTAU 自己写 UTF-8 BOM(见 import.rs strip_bom)—— 我们也写,最大化其读侧兼容。
    let mut bytes = Vec::with_capacity(yaml.len() + 3);
    bytes.extend_from_slice(&[0xEF, 0xBB, 0xBF]);
    bytes.extend_from_slice(yaml.as_bytes());
    std::fs::write(path, bytes).map_err(|e| format!("EXPORT_SCORE_WRITE_FAIL: {e}"))
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────
// midi(SMF format 1,PPQ 480 → 与工程 tick 1:1)
// ─────────────────────────────────────────────────────────────────────────────────────────────────────

fn export_midi(path: &str, bpm: f64, time_sig: [u32; 2], tracks: &[ExportTrack]) -> Result<(), String> {
    use midly::num::{u15, u24, u28, u4, u7};
    use midly::{Format, Header, MetaMessage, MidiMessage, Smf, Timing, Track, TrackEvent, TrackEventKind};

    const U28_MAX: u32 = (1 << 28) - 1; // u28::new 越界会 panic → 上游 clamp

    let mut smf = Smf::new(Header::new(Format::Parallel, Timing::Metrical(u15::new(TICKS_PER_BEAT as u16))));

    // Track 0(conductor):Tempo + TimeSignature + EOT。us/qn 四舍五入 clamp 进 u24;拍号分母取
    // 最近的 2 的幂(log2 round,den=6 → 8)—— MIDI 只能存 2 的幂,直接 log2 非幂值会写错档,不 clamp 会 panic。
    let us_per_qn = (60_000_000.0 / bpm).round().clamp(1.0, 16_777_215.0) as u32;
    let num = time_sig[0].clamp(1, 255) as u8;
    let denom_pow = (time_sig[1].max(1) as f64).log2().round().clamp(0.0, 7.0) as u8;
    let mut master: Track = Vec::new();
    master.push(TrackEvent { delta: u28::new(0), kind: TrackEventKind::Meta(MetaMessage::Tempo(u24::new(us_per_qn))) });
    master.push(TrackEvent { delta: u28::new(0), kind: TrackEventKind::Meta(MetaMessage::TimeSignature(num, denom_pow, 24, 8)) });
    master.push(TrackEvent { delta: u28::new(0), kind: TrackEventKind::Meta(MetaMessage::EndOfTrack) });
    smf.tracks.push(master);

    // 同 tick 排序键:Lyric(0) → NoteOff(1) → NoteOn(2)。Lyric 紧贴其 note-on 之前(import 的
    // stash 规则);前一音符与后一音符同点相接时 NoteOff 先于 NoteOn,避免 FIFO 配对混淆。
    enum Ev<'a> {
        Lyric(&'a str),
        Off(i32),
        On(i32, u8),
    }
    for t in tracks {
        let mut evs: Vec<(i64, u8, Ev)> = Vec::with_capacity(t.notes.len() * 3);
        for n in &t.notes {
            let vel = n.velocity.clamp(1, 127) as u8; // vel 0 = note-off 语义,不可用 → 提到 1
            evs.push((n.tick, 0, Ev::Lyric(&n.lyric)));
            evs.push((n.tick, 2, Ev::On(n.pitch, vel)));
            evs.push((n.tick + n.duration, 1, Ev::Off(n.pitch)));
        }
        evs.sort_by_key(|(tick, order, _)| (*tick, *order)); // 稳定排序,同键保持推入顺序

        let mut track: Track = Vec::new();
        track.push(TrackEvent { delta: u28::new(0), kind: TrackEventKind::Meta(MetaMessage::TrackName(t.name.as_bytes())) });
        let mut last: i64 = 0; // 绝对 tick → delta 差分
        for (abs, _, ev) in &evs {
            let delta = u28::new(((*abs - last).max(0) as u32).min(U28_MAX));
            last = *abs;
            let kind = match ev {
                Ev::Lyric(s) => TrackEventKind::Meta(MetaMessage::Lyric(s.as_bytes())),
                Ev::Off(k) => TrackEventKind::Midi {
                    channel: u4::new(0),
                    message: MidiMessage::NoteOff { key: u7::new(*k as u8), vel: u7::new(0) },
                },
                Ev::On(k, v) => TrackEventKind::Midi {
                    channel: u4::new(0),
                    message: MidiMessage::NoteOn { key: u7::new(*k as u8), vel: u7::new(*v) },
                },
            };
            track.push(TrackEvent { delta, kind });
        }
        track.push(TrackEvent { delta: u28::new(0), kind: TrackEventKind::Meta(MetaMessage::EndOfTrack) });
        smf.tracks.push(track);
    }

    let mut buf: Vec<u8> = Vec::new();
    smf.write(&mut buf).map_err(|e| format!("EXPORT_SCORE_WRITE_FAIL: midi: {e}"))?;
    std::fs::write(path, buf).map_err(|e| format!("EXPORT_SCORE_WRITE_FAIL: {e}"))
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────
// Tests — round-trip 经真实 import 命令读回(import_score_file 按扩展名分发到 parse_*)
// ─────────────────────────────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::import::ImportedScore;

    fn n(tick: i64, duration: i64, pitch: i32, lyric: &str) -> ExportNote {
        ExportNote { tick, duration, pitch, lyric: lyric.to_string(), velocity: 100, vibrato: None }
    }

    fn track(name: &str, notes: Vec<ExportNote>) -> ExportTrack {
        ExportTrack { name: name.to_string(), notes }
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("utai_export_score_tests");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(format!("{}_{name}", std::process::id()))
    }

    fn export(format: &str, path: &std::path::Path, tempo: f64, ts: [u32; 2], tracks: Vec<ExportTrack>) -> Result<(), String> {
        tauri::async_runtime::block_on(export_score_files(
            format.to_string(),
            path.to_string_lossy().into_owned(),
            tempo,
            ts,
            tracks,
        ))
    }

    fn import_back(path: &std::path::Path) -> ImportedScore {
        tauri::async_runtime::block_on(crate::commands::import::import_score_file(path.to_string_lossy().into_owned()))
            .unwrap()
    }

    fn contains(hay: &[u8], needle: &[u8]) -> bool {
        hay.windows(needle.len()).any(|w| w == needle)
    }

    #[test]
    fn ust_roundtrip_offset_gap_japanese_sjis() {
        // 头部 offset 960 + 中间 240 gap + 日文歌词 → Shift-JIS 分支;parse_ust 读回全等。
        let notes = vec![n(960, 240, 60, "あ"), n(1200, 240, 62, "い"), n(1680, 480, 64, "う")];
        let path = tmp("rt1.ust");
        export("ust", &path, 120.0, [4, 4], vec![track("メロ", notes.clone())]).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert!(contains(&bytes, b"Charset=Shift-JIS"));
        assert!(contains(&bytes, b"Tempo=120\r\n")); // 整数 bpm 写成 "120"
        let s = import_back(&path);
        assert_eq!(s.bpm, Some(120.0));
        assert_eq!(s.tracks.len(), 1);
        let t = &s.tracks[0];
        assert_eq!(t.start_tick, 960); // 头部 gap → 前置 R 节 → 读回 offset
        assert_eq!(t.notes.len(), 3);
        for (got, orig) in t.notes.iter().zip(&notes) {
            assert_eq!(got.tick, orig.tick - 960); // import 重定基到首音符
            assert_eq!(got.duration, orig.duration);
            assert_eq!(got.pitch, orig.pitch);
            assert_eq!(got.lyric, orig.lyric);
        }
    }

    #[test]
    fn ust_chinese_lyric_takes_utf8_branch() {
        // "绿"(简体)映射不进 Shift-JIS → 整份落 UTF-8;非整 bpm 经文本往返仍相等。
        let path = tmp("rt2.ust");
        export("ust", &path, 89.9, [4, 4], vec![track("中文", vec![n(0, 480, 60, "绿")])]).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert!(contains(&bytes, b"Charset=UTF-8"));
        assert!(!contains(&bytes, b"Charset=Shift-JIS"));
        let s = import_back(&path);
        assert_eq!(s.bpm, Some(89.9));
        let t = &s.tracks[0];
        assert_eq!(t.start_tick, 0);
        assert_eq!(t.notes[0].lyric, "绿");
        assert_eq!(t.notes[0].duration, 480);
    }

    #[test]
    fn ust_multitrack_zip_members_sanitized_and_roundtrip() {
        use std::io::Read;
        // 两条同名轨(带 Windows 非法字符 ':')→ zip;成员名 sanitize + " (2)" 去重。
        let path = tmp("rt3.zip");
        export(
            "ust",
            &path,
            120.0,
            [4, 4],
            vec![track("Vo:Main", vec![n(0, 480, 60, "あ")]), track("Vo:Main", vec![n(480, 240, 67, "か")])],
        )
        .unwrap();
        let file = std::fs::File::open(&path).unwrap();
        let mut ar = zip::ZipArchive::new(file).unwrap();
        assert_eq!(ar.len(), 2);
        let names: Vec<String> = (0..ar.len()).map(|i| ar.by_index(i).unwrap().name().to_string()).collect();
        assert_eq!(names, vec!["Vo_Main.ust".to_string(), "Vo_Main (2).ust".to_string()]);
        // 逐成员解出 → parse_ust round-trip。
        for i in 0..2 {
            let mut buf: Vec<u8> = Vec::new();
            ar.by_index(i).unwrap().read_to_end(&mut buf).unwrap();
            let member_path = tmp(&format!("rt3_member{i}.ust"));
            std::fs::write(&member_path, &buf).unwrap();
            let s = import_back(&member_path);
            let t = &s.tracks[0];
            if i == 0 {
                assert_eq!(t.start_tick, 0);
                assert_eq!((t.notes[0].duration, t.notes[0].pitch, t.notes[0].lyric.as_str()), (480, 60, "あ"));
            } else {
                assert_eq!(t.start_tick, 480); // 首音符前的 gap → R 节 → 读回 offset
                assert_eq!((t.notes[0].duration, t.notes[0].pitch, t.notes[0].lyric.as_str()), (240, 67, "か"));
            }
        }
    }

    #[test]
    fn ustx_roundtrip_bpm_timesig_offset_and_vibrato_inverse() {
        let vib = ExportVibrato {
            depth_cents: 50.0,
            freq_hz: 5.0,
            phase: 0.25,
            start_ms: 192.0,
            ease_in_ms: 20.0,
            ease_out_ms: 30.0,
        };
        let mut n0 = n(1200, 480, 65, "よ");
        n0.vibrato = Some(vib);
        let path = tmp("rt4.ustx");
        export("ustx", &path, 156.0, [3, 8], vec![track("LEAD", vec![n0, n(1680, 240, 67, "い")])]).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[..3], &[0xEF, 0xBB, 0xBF]); // UTF-8 BOM(OpenUTAU 惯例)
        let s = import_back(&path);
        assert_eq!(s.bpm, Some(156.0));
        assert_eq!(s.time_sig, Some([3, 8]));
        assert_eq!(s.tracks.len(), 1);
        let t = &s.tracks[0];
        assert_eq!(t.name, "LEAD");
        assert_eq!(t.start_tick, 1200); // part.position = 首音符绝对 tick
        assert_eq!(t.notes.len(), 2);
        assert_eq!((t.notes[0].tick, t.notes[0].duration, t.notes[0].pitch, t.notes[0].lyric.as_str()), (0, 480, 65, "よ"));
        assert_eq!((t.notes[1].tick, t.notes[1].duration, t.notes[1].pitch, t.notes[1].lyric.as_str()), (480, 240, 67, "い"));
        // unmap_vibrato ∘ map_vibrato = id(浮点近似):各字段误差 < 0.05。
        let v = t.notes[0].vibrato.as_ref().expect("vibrato survives the round-trip");
        assert!((v.depth_cents - vib.depth_cents).abs() < 0.05);
        assert!((v.freq_hz - vib.freq_hz).abs() < 0.05);
        assert!((v.phase - vib.phase).abs() < 0.05);
        assert!((v.start_ms - vib.start_ms).abs() < 0.05);
        assert!((v.ease_in_ms - vib.ease_in_ms).abs() < 0.05);
        assert!((v.ease_out_ms - vib.ease_out_ms).abs() < 0.05);
        assert!(t.notes[1].vibrato.is_none()); // 没喂 vibrato 的音符不产出 vibrato
    }

    #[test]
    fn ustx_negative_phase_wraps_into_openutau_domain() {
        // 我们的 phase 是循环量(−0.25 ≡ +0.75);OpenUTAU shift 域 [0,100] 且硬钳 —— 导出必须
        // rem_euclid 包卷而不是让负值被钳成 0(丢相位)。
        let vib = ExportVibrato { depth_cents: 40.0, freq_hz: 5.0, phase: -0.25, start_ms: 100.0, ease_in_ms: 0.0, ease_out_ms: 0.0 };
        let out = unmap_vibrato(&vib, 480, 120.0).expect("vibrato survives");
        assert!((out.shift - 75.0).abs() < 1e-4, "shift = {}", out.shift);
        // 读回:map_vibrato 的 shift/100 落在 [0,1) —— 等价相位,不再是负数但正弦上同点。
        assert!(out.shift >= 0.0 && out.shift < 100.0);
    }

    #[test]
    fn ust_lossy_sjis_chars_fall_back_to_utf8() {
        // ¥(U+00A5)/‾(U+203E)在 WHATWG Shift_JIS 编码器里「成功」映射到 0x5C/0x7E —— had_errors
        // 抓不到,读回变成 \ 和 ~。encode_ust 的 decode 往返比对必须把这类文本落到 UTF-8 分支。
        let path = tmp("rt_yen.ust");
        export("ust", &path, 120.0, [4, 4], vec![track("", vec![n(0, 240, 60, "¥‾テスト")])]).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let text = String::from_utf8(bytes).expect("must be the UTF-8 branch");
        assert!(text.contains("Charset=UTF-8"));
        let s = import_back(&path);
        assert_eq!(s.tracks[0].notes[0].lyric, "¥‾テスト"); // 逐字符还原,零串改
    }

    #[test]
    fn midi_roundtrip_ppq480_one_to_one() {
        // PPQ 480 → 工程 tick 1:1;velocity=0 的音符必须写成 vel=1(vel 0 = note-off 语义)。
        let mut n1 = n(1440, 240, 72, "あ");
        n1.velocity = 0;
        let notes = vec![n(480, 480, 60, "か"), n1];
        let path = tmp("rt5.mid");
        export("midi", &path, 120.0, [3, 4], vec![track("melody", notes.clone())]).unwrap();
        let s = import_back(&path);
        assert_eq!(s.bpm, Some(120.0)); // 500000 µs/qn → 精确 120
        assert_eq!(s.time_sig, Some([3, 4]));
        assert_eq!(s.tracks.len(), 1); // conductor 轨无音符 → import 跳过
        let t = &s.tracks[0];
        assert_eq!(t.name, "melody");
        assert_eq!(t.start_tick, 480);
        assert_eq!(t.notes.len(), 2); // vel=0 的音符没有变成 note-off 被吞
        for (got, orig) in t.notes.iter().zip(&notes) {
            assert_eq!(got.tick, orig.tick - 480);
            assert_eq!(got.duration, orig.duration);
            assert_eq!(got.pitch, orig.pitch);
            assert_eq!(got.lyric, orig.lyric); // velocity 不比 —— import 不读
        }
    }

    #[test]
    fn overlap_truncation_and_engulfed_drop() {
        // 重叠:前者截到后者起点(半开区间)。
        let out = sanitize_notes(vec![n(0, 480, 60, "a"), n(240, 240, 62, "b")]);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].duration, 240);
        assert_eq!(out[1].tick, 240);
        // 完全吞没(同 tick 起步,前者截成 0)→ 丢弃前者。
        let out = sanitize_notes(vec![n(100, 400, 60, "a"), n(100, 200, 62, "b")]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].pitch, 62);
        // 长音符盖住多个后续音符 → 只截到紧随者起点,后续互不重叠保持原样。
        let out = sanitize_notes(vec![n(0, 1000, 60, "a"), n(100, 50, 61, "b"), n(200, 50, 62, "c")]);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].duration, 100);
        assert_eq!(out[1].duration, 50);
    }

    #[test]
    fn empty_tracks_and_unsupported_format_error_codes() {
        let path = tmp("rt7.ust");
        assert_eq!(export("ust", &path, 120.0, [4, 4], vec![]).unwrap_err(), "EXPORT_SCORE_EMPTY");
        assert_eq!(
            export("ust", &path, 120.0, [4, 4], vec![track("x", vec![])]).unwrap_err(),
            "EXPORT_SCORE_EMPTY"
        );
        let err = export("xml", &path, 120.0, [4, 4], vec![track("x", vec![n(0, 480, 60, "a")])]).unwrap_err();
        assert!(err.starts_with("EXPORT_SCORE_UNSUPPORTED"), "{err}");
        assert!(!path.exists()); // 全部早退,不该落盘
    }

    #[test]
    fn midi_non_pow2_time_sig_denominator_does_not_panic() {
        // den=6 不是 2 的幂 → 取最近的 2 幂(8)写 TimeSignature,绝不 panic。
        let path = tmp("rt8.mid");
        export("midi", &path, 120.0, [4, 6], vec![track("t", vec![n(0, 480, 60, "a")])]).unwrap();
        let s = import_back(&path);
        assert_eq!(s.time_sig, Some([4, 8]));
    }
}
