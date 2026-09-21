//! 音源(SoundFont)管理 + 乐器加载(阶段1+2)。
//!
//! 目录约定:`<app_dir>/soundfonts/`
//!   - SFZ:每个乐器一个子目录(内含 .sfz + 采样文件)
//!   - SF2:每个音源一个 .sf2 文件(preset 在文件内部)
//!
//! 不内置任何音源(工程规划:插件化,用户自行导入)。
//! 渲染:统一走 synth::Synth 离线 bake 成 WAV(16-bit),
//! 前端用 WebAudio 播放返回的文件。

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Serialize;

use super::sf2;
use super::sfz;
use super::synth::{LoadedInstrument, MidiEvent, pan_gains, Region, SampleData, Synth};

#[derive(Debug, Clone, Serialize)]
pub struct SoundfontPresetEntry {
    /// preset 标识(SFZ:文件名;SF2:"bank:program")。
    pub id: String,
    pub name: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SoundfontEntry {
    /// 音源标识(SFZ:目录名;SF2:文件 stem)。
    pub id: String,
    pub name: String,
    pub format: String, // "sfz" | "sf2"
    pub size_bytes: u64,
    pub presets: Vec<SoundfontPresetEntry>,
}

pub fn soundfonts_dir(app_dir: &Path) -> PathBuf {
    app_dir.join("soundfonts")
}

/// 扫描音源目录。SFZ 目录必须含至少一个 .sfz;SF2 直接解析 phdr 取 preset 列表
/// (解析失败时仍列出,标记空 preset 列表,前端显示导入异常)。
pub fn scan_soundfonts(dir: &Path) -> Vec<SoundfontEntry> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return out;
    };
    for entry in rd.flatten() {
        let path = entry.path();
        let Ok(meta) = entry.metadata() else { continue };
        if meta.is_dir() {
            // SFZ 目录:找 .sfz
            let sfz_files = find_sfz_files(&path);
            if sfz_files.is_empty() {
                continue;
            }
            let size = dir_size(&path);
            let id = entry.file_name().to_string_lossy().to_string();
            let presets = sfz_files
                .iter()
                .map(|p| SoundfontPresetEntry {
                    id: p.file_stem().unwrap_or_default().to_string_lossy().to_string(),
                    name: p.file_stem().unwrap_or_default().to_string_lossy().to_string(),
                })
                .collect();
            out.push(SoundfontEntry {
                id: id.clone(),
                name: id,
                format: "sfz".into(),
                size_bytes: size,
                presets,
            });
        } else if path.extension().and_then(|e| e.to_str()).map(|e| e.eq_ignore_ascii_case("sf2")).unwrap_or(false) {
            let id = path.file_stem().unwrap_or_default().to_string_lossy().to_string();
            let presets = match sf2::parse_sf2(&path) {
                Ok(font) => font
                    .presets
                    .iter()
                    .map(|p| SoundfontPresetEntry {
                        id: format!("{}:{}", p.info.bank, p.info.program),
                        name: p.info.name.clone(),
                    })
                    .collect(),
                Err(e) => {
                    tracing::warn!("SF2 parse failed '{}': {}", path.display(), e);
                    Vec::new()
                }
            };
            out.push(SoundfontEntry {
                id: id.clone(),
                name: id,
                format: "sf2".into(),
                size_bytes: meta.len(),
                presets,
            });
        }
    }
    out.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    out
}

fn find_sfz_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return out;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_file()
            && p.extension()
                .and_then(|x| x.to_str())
                .map(|x| x.eq_ignore_ascii_case("sfz"))
                .unwrap_or(false)
        {
            out.push(p);
        }
    }
    out.sort();
    out
}

fn dir_size(dir: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            if let Ok(m) = e.metadata() {
                if m.is_dir() {
                    total += dir_size(&e.path());
                } else {
                    total += m.len();
                }
            }
        }
    }
    total
}

/// 导入音源:
/// - path 是 .sf2 文件 → 复制为 `<dir>/<原文件名>`
/// - path 是目录(或 .sfz 文件)→ 整体复制为 `<dir>/<名字>/`
/// id 冲突时自动加后缀(不覆盖用户已有音源)。
pub fn import_soundfont(dir: &Path, src: &Path) -> Result<SoundfontEntry, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("创建音源目录失败: {}", e))?;
    let name = src
        .file_stem()
        .or_else(|| src.file_name())
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    if name.is_empty() {
        return Err("无效的音源路径".into());
    }

    let is_file = src.is_file();
    let target = if is_file {
        let ext = src.extension().and_then(|e| e.to_str()).unwrap_or("");
        unique_path(&dir.join(format!("{}.{}", sanitize(&name), ext)))
    } else {
        unique_path(&dir.join(sanitize(&name)))
    };

    if is_file {
        std::fs::copy(src, &target).map_err(|e| format!("复制文件失败: {}", e))?;
    } else {
        copy_dir_recursive(src, &target).map_err(|e| format!("复制目录失败: {}", e))?;
    }

    // 复扫返回新条目
    let id = target
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    Ok(SoundfontEntry {
        id: id.clone(),
        name: id,
        format: if is_file { "sf2".into() } else { "sfz".into() },
        size_bytes: if is_file {
            std::fs::metadata(&target).map(|m| m.len()).unwrap_or(0)
        } else {
            dir_size(&target)
        },
        presets: Vec::new(), // 调用方 rescan 填充
    })
}

pub(crate) fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            c => c,
        })
        .collect()
}

fn unique_path(p: &Path) -> PathBuf {
    if !p.exists() {
        return p.to_path_buf();
    }
    let stem = p.file_stem().unwrap_or_default().to_string_lossy().to_string();
    let ext = p.extension().map(|e| format!(".{}", e.to_string_lossy())).unwrap_or_default();
    let parent = p.parent().unwrap_or(Path::new("."));
    for i in 1..1000 {
        let cand = parent.join(format!("{}-{}{}", stem, i, ext));
        if !cand.exists() {
            return cand;
        }
    }
    p.to_path_buf()
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for e in std::fs::read_dir(src)? {
        let e = e?;
        let from = e.path();
        let to = dst.join(e.file_name());
        if from.is_dir() {
            copy_dir_recursive(&from, &to)?;
        } else {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

/// 删除音源(SFZ 目录整删;SF2 文件删除)。
pub fn delete_soundfont(dir: &Path, id: &str) -> Result<(), String> {
    let sid = sanitize(id);
    let sf2 = dir.join(format!("{}.sf2", sid));
    if sf2.is_file() {
        std::fs::remove_file(&sf2).map_err(|e| format!("删除失败: {}", e))?;
        return Ok(());
    }
    let d = dir.join(&sid);
    if d.is_dir() {
        std::fs::remove_dir_all(&d).map_err(|e| format!("删除失败: {}", e))?;
        return Ok(());
    }
    Err(format!("未找到音源: {}", id))
}

// ── 乐器加载 ─────────────────────────────────────────────────────────────

/// 按音源 id + preset id 加载乐器。
/// SFZ:preset id = .sfz 文件 stem;SF2:preset id = "bank:program"。
pub fn load_instrument(dir: &Path, font_id: &str, preset_id: &str) -> Result<LoadedInstrument, String> {
    let sid = sanitize(font_id);
    // SF2?
    let sf2_path = dir.join(format!("{}.sf2", sid));
    if sf2_path.is_file() {
        let font = sf2::parse_sf2(&sf2_path)?;
        let (bank, program) = parse_bank_program(preset_id)?;
        let preset = font
            .presets
            .iter()
            .find(|p| p.info.bank == bank && p.info.program == program)
            .ok_or_else(|| format!("SF2 中不存在 preset {}", preset_id))?;
        return Ok(preset.to_instrument());
    }
    // SFZ 目录
    let font_dir = dir.join(&sid);
    if !font_dir.is_dir() {
        return Err(format!("未找到音源: {}", font_id));
    }
    let sfz_files = find_sfz_files(&font_dir);
    let sfz_path = sfz_files
        .iter()
        .find(|p| {
            p.file_stem()
                .map(|s| s.to_string_lossy() == preset_id)
                .unwrap_or(false)
        })
        .cloned()
        .or_else(|| sfz_files.first().cloned()) // 兼容:单 preset SFZ 目录直接用第一个
        .ok_or_else(|| format!("SFZ 音源 '{}' 中没有 .sfz 文件", font_id))?;
    let regions = sfz::parse_sfz(&sfz_path)?;
    load_sfz_instrument(regions)
}

fn parse_bank_program(preset_id: &str) -> Result<(u16, u16), String> {
    let mut it = preset_id.splitn(2, ':');
    let bank = it
        .next()
        .and_then(|b| b.parse::<u16>().ok())
        .ok_or_else(|| format!("无效 preset 标识: {}", preset_id))?;
    let program = it
        .next()
        .and_then(|p| p.parse::<u16>().ok())
        .ok_or_else(|| format!("无效 preset 标识: {}", preset_id))?;
    Ok((bank, program))
}

/// SFZ regions → LoadedInstrument(解码采样,路径去重)。
fn load_sfz_instrument(regions: Vec<Region>) -> Result<LoadedInstrument, String> {
    let mut cache: HashMap<PathBuf, Arc<SampleData>> = HashMap::new();
    let mut samples = Vec::with_capacity(regions.len());
    for r in &regions {
        let data = match cache.get(&r.sample_path) {
            Some(d) => d.clone(),
            None => {
                let d = load_sample_file(&r.sample_path)?;
                cache.insert(r.sample_path.clone(), d.clone());
                d
            }
        };
        samples.push(data);
    }
    Ok(LoadedInstrument { regions, samples })
}

/// 解码一个采样文件为 SampleData(支持 WAV smpl 循环点)。
fn load_sample_file(path: &Path) -> Result<Arc<SampleData>, String> {
    let buf = super::load_audio(path).map_err(|e| format!("采样解码失败 '{}': {}", path.display(), e))?;
    let smpl_meta = if path.extension().and_then(|e| e.to_str()).map(|e| e.eq_ignore_ascii_case("wav")).unwrap_or(false) {
        parse_wav_smpl(path)
    } else {
        None
    };
    let (left, right) = match buf.channels {
        1 => (buf.samples.clone(), None),
        2 => {
            let mut l = Vec::with_capacity(buf.samples.len() / 2);
            let mut r = Vec::with_capacity(buf.samples.len() / 2);
            for pair in buf.samples.chunks_exact(2) {
                l.push(pair[0]);
                r.push(pair[1]);
            }
            (l, Some(r))
        }
        _ => {
            // 多声道:取前两个
            let ch = buf.channels as usize;
            let mut l = Vec::with_capacity(buf.samples.len() / ch);
            let mut r = Vec::with_capacity(buf.samples.len() / ch);
            for f in buf.samples.chunks_exact(ch) {
                l.push(f[0]);
                r.push(*f.get(1).unwrap_or(&f[0]));
            }
            (l, Some(r))
        }
    };
    let frames = left.len() as u32;
    let meta = smpl_meta.unwrap_or((None, None, 0.0));
    let (loop_range, root_key, fine_cents) = meta;
    let loop_range = loop_range.filter(|(s, e)| *e > *s && *e <= frames);
    Ok(Arc::new(SampleData {
        left,
        right,
        rate: buf.sample_rate,
        loop_range,
        root_key: root_key.filter(|k| *k <= 127),
        fine_tune_cents: fine_cents,
    }))
}

/// 解析 WAV 的 smpl chunk:(loop_start, loop_end), root_key, fine_cents。
/// 标准 RIFF 布局:manufacturer u32, product u32, period u32, unity u32, fraction u32,
///       smpteFormat u32, smpteOffset u32, numLoops u32, samplerData u32,
///       随后每个循环块 24 字节 { cue u32, type u32, start u32, end u32, fraction u32, plays u32 }。
fn parse_wav_smpl(path: &Path) -> Option<(Option<(u32, u32)>, Option<u8>, f32)> {
    let data = std::fs::read(path).ok()?;
    if data.len() < 12 || &data[0..4] != b"RIFF" {
        return None;
    }
    let u32le = |o: usize| -> u32 { u32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]) };
    let mut pos = 12usize;
    while pos + 8 <= data.len() {
        let id = &data[pos..pos + 4];
        let size = u32le(pos + 4) as usize;
        let body = pos + 8;
        if id == b"smpl" && body + 36 <= data.len() {
            let unity = u32le(body + 12);
            let fraction = u32le(body + 16);
            let num_loops = u32le(body + 28);
            let root = unity.min(127) as u8;
            let cents = fraction as f32 / 4294967296.0; // 分数半音 → 0..1
            // 取第一个循环块(cue/type 各占 4 字节,start=块内偏移 8,end=块内偏移 12)。
            let lp = if num_loops >= 1 && body + 52 <= data.len() {
                let ls = u32le(body + 44);
                let le = u32le(body + 48);
                if le > ls {
                    Some((ls, le))
                } else {
                    None
                }
            } else {
                None
            };
            return Some((lp, Some(root), cents * 100.0));
        }
        pos = body + size + (size & 1);
    }
    None
}

// ── 渲染 ─────────────────────────────────────────────────────────────────

/// 渲染请求里的音符。
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RenderNote {
    /// 开始(秒)。
    pub start: f64,
    /// 时长(秒;note-off 时刻 = start + dur)。
    pub dur: f64,
    /// MIDI 键 0..127。
    pub key: u8,
    /// 力度 0..127。
    pub vel: u8,
}

/// 渲染请求里的控制器事件(CC1 颤音 / CC11 表情 / CC64 踏板)。
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RenderCc {
    /// 时刻(秒)。
    pub start: f64,
    /// 控制器号。
    pub cc: u8,
    /// 值 0..127。
    pub value: u8,
}

/// 渲染请求里的弯音事件(value ∈ [-8192, 8191],0 = 居中)。
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RenderBend {
    /// 时刻(秒)。
    pub start: f64,
    pub value: i16,
}

/// 分层渲染的音源层(3-1):同一份 notes 喂 N 层,各层独立音源,输出按 gain/pan 混合。
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderLayer {
    pub font_id: String,
    pub preset_id: String,
    /// 层增益(0..8,1 = 原声,渲染时钳位)。
    pub gain: f32,
    /// 声像(-1 左 .. +1 右)。
    pub pan: f32,
}

/// 采样率白名单(44.1k..192k),越界回退 44.1k。
fn clamp_sr(sample_rate: u32) -> f32 {
    if (44100..=192000).contains(&sample_rate) {
        sample_rate as f32
    } else {
        44100.0
    }
}

/// notes/cc/bend → 事件序列(非法项丢弃)。空音符返回 Err。
fn build_events(
    notes: &[RenderNote],
    cc: &[RenderCc],
    bend: &[RenderBend],
    sr: f32,
) -> Result<Vec<MidiEvent>, String> {
    let mut events: Vec<MidiEvent> = Vec::with_capacity(notes.len() * 2 + cc.len() + bend.len());
    let mut has_note = false;
    for n in notes {
        if n.key > 127 || n.start < 0.0 || n.dur <= 0.0 {
            continue;
        }
        has_note = true;
        let on = (n.start * sr as f64).round() as u64;
        events.push(MidiEvent::NoteOn {
            frame: on,
            key: n.key,
            vel: n.vel.min(127),
        });
        if n.dur > 0.0 {
            let off = ((n.start + n.dur) * sr as f64).round() as u64;
            if off > on {
                events.push(MidiEvent::NoteOff { frame: off, key: n.key });
            }
        }
    }
    for c in cc {
        if c.start < 0.0 || c.value > 127 {
            continue;
        }
        events.push(MidiEvent::ControlChange {
            frame: (c.start * sr as f64).round() as u64,
            cc: c.cc,
            value: c.value,
        });
    }
    for b in bend {
        if b.start < 0.0 {
            continue;
        }
        events.push(MidiEvent::PitchBend {
            frame: (b.start * sr as f64).round() as u64,
            value: b.value.clamp(-8192, 8191),
        });
    }
    if !has_note {
        return Err("没有可渲染的音符".into());
    }
    Ok(events)
}

/// 渲染一份乐器 → 立体声 f32(不落盘)。单渲染与分层混合的公共核心。
pub fn render_notes_stereo(
    instr: &LoadedInstrument,
    notes: &[RenderNote],
    cc: &[RenderCc],
    bend: &[RenderBend],
    sr: f32,
) -> Result<Vec<f32>, String> {
    let mut events = build_events(notes, cc, bend, sr)?;
    let synth = Synth::new(instr, sr);
    let stereo = synth.render_offline(&mut events, 0.5);
    if stereo.is_empty() {
        return Err("渲染结果为空".into());
    }
    Ok(stereo)
}

/// 离线渲染乐器 → WAV(16-bit 立体声),返回文件路径。
pub fn render_to_wav(
    dir: &Path,
    font_id: &str,
    preset_id: &str,
    notes: &[RenderNote],
    cc: &[RenderCc],
    bend: &[RenderBend],
    sample_rate: u32,
    out_path: &Path,
) -> Result<(), String> {
    let instr = load_instrument(dir, font_id, preset_id)?;
    let sr = clamp_sr(sample_rate);
    let stereo = render_notes_stereo(&instr, notes, cc, bend, sr)?;
    write_wav16(out_path, &stereo, sr as u32).map_err(|e| format!("写 WAV 失败: {}", e))
}

/// 分层混合:stereo(L/R 交替)按 gain/pan 叠加进 mixed,不足处补零对齐。
fn mix_into(mixed: &mut Vec<f32>, stereo: &[f32], gain: f32, pan: f32) {
    let (lg, rg) = pan_gains(pan);
    let g = gain.clamp(0.0, 8.0);
    if mixed.len() < stereo.len() {
        mixed.resize(stereo.len(), 0.0);
    }
    for (i, s) in stereo.iter().enumerate() {
        let w = if i % 2 == 0 { lg * g } else { rg * g };
        mixed[i] += s * w;
    }
}

/// 分层渲染:同一份 notes 依次喂 N 个音源层,输出按各层 gain/pan 混合 → WAV。
/// 任一层加载/渲染失败 → 整体报错(带层序号上下文)。
pub fn render_layers_to_wav(
    dir: &Path,
    layers: &[RenderLayer],
    notes: &[RenderNote],
    cc: &[RenderCc],
    bend: &[RenderBend],
    sample_rate: u32,
    out_path: &Path,
) -> Result<(), String> {
    if layers.is_empty() {
        return Err("没有可渲染的音源层".into());
    }
    let sr = clamp_sr(sample_rate);
    let mut mixed: Vec<f32> = Vec::new();
    for (idx, layer) in layers.iter().enumerate() {
        let instr = load_instrument(dir, &layer.font_id, &layer.preset_id).map_err(|e| {
            format!("第 {} 层加载音源失败({}/{}): {}", idx + 1, layer.font_id, layer.preset_id, e)
        })?;
        let stereo = render_notes_stereo(&instr, notes, cc, bend, sr)?;
        mix_into(&mut mixed, &stereo, layer.gain, layer.pan);
    }
    write_wav16(out_path, &mixed, sr as u32).map_err(|e| format!("写 WAV 失败: {}", e))
}

/// 写 16-bit 立体声 WAV。
fn write_wav16(path: &Path, samples: &[f32], rate: u32) -> Result<(), String> {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec).map_err(|e| e.to_string())?;
    for s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        w.write_sample(v).map_err(|e| e.to_string())?;
    }
    w.finalize().map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bank_program_parse() {
        assert_eq!(parse_bank_program("0:0").unwrap(), (0, 0));
        assert_eq!(parse_bank_program("128:60").unwrap(), (128, 60));
        assert!(parse_bank_program("abc").is_err());
        assert!(parse_bank_program("1").is_err());
    }

    #[test]
    fn sanitize_blocks_traversal() {
        assert_eq!(sanitize("a/b\\c:d"), "a_b_c_d");
        assert!(!sanitize("../evil").contains('/'));
    }

    #[test]
    fn mix_into_applies_gain_and_pan() {
        // pan_gains(+1) = (0.25, 1.0):右声道满增益,左声道 -12 dB。
        let mut m: Vec<f32> = Vec::new();
        mix_into(&mut m, &[0.5, 0.5], 1.0, 1.0);
        assert!((m[0] - 0.125).abs() < 1e-6 && (m[1] - 0.5).abs() < 1e-6);

        // 两层叠加 = 线性混合;短层补零不改变总长。
        let mut m: Vec<f32> = Vec::new();
        mix_into(&mut m, &[0.5, 0.5, 0.2, 0.2], 1.0, 0.0);
        mix_into(&mut m, &[0.1, 0.1], 1.0, 0.0);
        assert_eq!(m.len(), 4);
        assert!((m[0] - 0.6).abs() < 1e-6 && (m[2] - 0.2).abs() < 1e-6);

        // gain 钳位到 0..8。
        let mut m: Vec<f32> = Vec::new();
        mix_into(&mut m, &[0.5, 0.5], 100.0, 0.0);
        assert!((m[0] - 4.0).abs() < 1e-6);
    }
}
