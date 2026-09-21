//! 通用 FluidSynth CLI 渲染后端（P3 3-9 通用化）。
//!
//! 原本只服务 AMT 导出（amt_export.rs 私有函数），现提升为 audio 层公共模块：
//!   - [`exe_path`] / [`env`] / [`render`]：FluidSynth 2.5.6 CLI 的发现与调用，
//!     命令行与参考实现逐字对齐（src/core/muscriptor_result_assets.py::_synthesize）：
//!     `fluidsynth -ni -o audio.file.format={s24|s16} -F out.wav -r {rate} soundfont mid`，
//!     并复刻其 PATH 注入（bin + lib 目录前置），保证 exe 同目录 DLL 可见。
//!   - [`write_notes_midi`]：音符/CC/弯音事件 → SMF 文件。PPQ 480 + 固定 120 BPM
//!     ⇒ 960 tick/秒，与 RenderNote 的秒制时间轴无损换算；通道 0 上以 bank select
//!     (CC0/CC32) + ProgramChange 落 `preset_id`（"bank:program"，用户音源库约定）。
//!   - [`render_notes_to_wav`]：`render_soundfont_notes` 的 "fluidsynth" 后端（对照组：
//!     内置 Rust 合成器 vs FluidSynth 官方合成器）。用户音源库的 SF2 经由
//!     临时 MIDI → CLI → WAV 完成渲染。FluidSynth 只认 SF2/SF3 —— SFZ 目录明确报错。
//!
//! 错误约定（i18n 铁律）：用户可见错误一律稳定 CODE ——
//!   EXPORT_FLUIDSYNTH_NOT_FOUND / EXPORT_SOUNDFONT_NOT_FOUND / SOUNDFONT_BACKEND_UNSUPPORTED /
//!   EXPORT_WRITE_FAILED: {detail} / EXPORT_RENDER_FAILED: {detail}

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use super::soundfont::{RenderBend, RenderCc, RenderNote};

/// FluidSynth 相对 models_dir 的受管安装路径（资源管理器下载的副本）。
const FLUIDSYNTH_REL: [&str; 4] = ["fluidsynth", "2.5.6", "bin", "fluidsynth.exe"];

/// 音频输出档位：采样率 + 位深（CLI `-r` / `-o audio.file.format`）。suffix 仅供
/// AMT 导出拼文件名，通用后端不使用。
pub struct AudioPreset {
    pub sample_rate: u32,
    pub format: &'static str,
    pub suffix: &'static str,
}

pub fn audio_preset(preset: &str) -> AudioPreset {
    match preset {
        "compat" => AudioPreset { sample_rate: 44_100, format: "s16", suffix: "16bit-44.1kHz" },
        _ => AudioPreset { sample_rate: 48_000, format: "s24", suffix: "24bit-48kHz" },
    }
}

/// 发现受管的 FluidSynth 可执行文件（models_dir/fluidsynth/2.5.6/bin/fluidsynth.exe）。
pub fn exe_path(models_dir: &Path) -> Option<PathBuf> {
    let exe: PathBuf = FLUIDSYNTH_REL.iter().fold(models_dir.to_path_buf(), |acc, p| acc.join(p));
    if exe.is_file() { Some(exe) } else { None }
}

/// 参考 get_fluidsynth_subprocess_env：bin + lib 目录前置进 PATH，保证同目录 DLL 可见。
pub fn env(exe: &Path) -> HashMap<String, String> {
    let mut env: HashMap<String, String> = std::env::vars().collect();
    let bin_dir = exe.parent().map(|p| p.to_path_buf()).unwrap_or_default();
    let lib_dir = bin_dir.parent().map(|p| p.join("lib")).unwrap_or_default();
    let mut entries: Vec<String> = Vec::new();
    if !bin_dir.as_os_str().is_empty() {
        entries.push(bin_dir.to_string_lossy().into_owned());
    }
    if lib_dir.is_dir() {
        entries.push(lib_dir.to_string_lossy().into_owned());
    }
    if let Some(current) = env.get("PATH") {
        entries.push(current.clone());
    }
    env.insert("PATH".to_string(), entries.join(";"));
    env
}

/// 用 FluidSynth 把一个 MIDI 渲染成 WAV。命令行与参考 _synthesize 逐字一致。
/// 管道用后台线程排空避免 pipe-full 死锁；整体 600s 超时保护（参考实现同值）。
pub fn render(exe: &Path, sf: &Path, midi: &Path, out: &Path, preset: &AudioPreset) -> Result<(), String> {
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("-ni")
        .arg("-o").arg(format!("audio.file.format={}", preset.format))
        .arg("-F").arg(out)
        .arg("-r").arg(preset.sample_rate.to_string())
        .arg(sf)
        .arg(midi)
        .envs(env(exe))
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(crate::util::CREATE_NO_WINDOW);
    }

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("EXPORT_RENDER_FAILED: spawn fluidsynth: {e}"))?;

    // 后台线程读尽 stdout/stderr —— FluidSynth 输出很少，但必须排空防止写满管道卡死。
    fn drain<R: std::io::Read + Send + 'static>(stream: Option<R>) -> Option<std::thread::JoinHandle<()>> {
        stream.map(|mut reader| {
            std::thread::spawn(move || {
                let mut buf = [0u8; 4096];
                while reader.read(&mut buf).map(|n| n > 0).unwrap_or(false) {}
            })
        })
    }
    let mut stdout_drain = drain(child.stdout.take());
    let mut stderr_drain = drain(child.stderr.take());
    let mut join_drains = || {
        if let Some(h) = stdout_drain.take() {
            let _ = h.join();
        }
        if let Some(h) = stderr_drain.take() {
            let _ = h.join();
        }
    };

    let deadline = std::time::Instant::now() + Duration::from_secs(600);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if std::time::Instant::now() > deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    join_drains();
                    return Err("EXPORT_RENDER_FAILED: FluidSynth timed out after 600s".into());
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => {
                join_drains();
                return Err(format!("EXPORT_RENDER_FAILED: wait fluidsynth: {e}"));
            }
        }
    };
    join_drains();

    let rendered = out.is_file() && std::fs::metadata(out).map(|m| m.len() > 0).unwrap_or(false);
    if !status.success() || !rendered {
        return Err(format!(
            "EXPORT_RENDER_FAILED: fluidsynth exit={:?} output_exists={}",
            status.code(),
            rendered
        ));
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// 音符事件 → SMF（render_soundfont_notes 的 "fluidsynth" 后端专用）
// ─────────────────────────────────────────────────────────────────────────────

/// PPQ 480 + 固定 120 BPM ⇒ 960 tick/秒；RenderNote 的秒制时间轴 × 960 即 tick，
/// 与内置合成器逐帧对齐（同样的秒 → 同样的毫秒级落点）。
const SEC_TO_TICK: f64 = 960.0;

/// notes/cc/bend → 单轨 SMF 文件。
///
/// 时间排序约定（同 tick 稳定优先级）：class 0 = NoteOff / bank·program，
/// class 1 = CC / 弯音，class 2 = NoteOn —— 同 tick 的收束先于发声、
/// 控制器先于它要影响的 NoteOn，重击同 tick 也是一个干净的断口。
///
/// preset_id 形如 "bank:program"（用户音源库 SF2 约定）；解析失败按 0:0 处理。
pub fn write_notes_midi(
    out_path: &Path,
    notes: &[RenderNote],
    cc: &[RenderCc],
    bend: &[RenderBend],
    preset_id: &str,
) -> Result<(), String> {
    use midly::{
        Format, Header, MetaMessage, MidiMessage, PitchBend, Smf, Timing, Track, TrackEvent,
        TrackEventKind,
    };
    use midly::num::{u14, u15, u24, u28, u4, u7};

    let mut bank = 0u16;
    let mut program = 0u8;
    let mut it = preset_id.splitn(2, ':');
    if let Some(b) = it.next().and_then(|s| s.parse::<u16>().ok()) {
        bank = b;
    }
    if let Some(p) = it.next().and_then(|s| s.parse::<u16>().ok()) {
        program = p.min(127) as u8;
    }

    // (tick, class, kind)：见函数注释的 class 约定。
    let mut events: Vec<(i64, u8, TrackEventKind)> = Vec::new();
    let ch = u4::new(0);
    let bank_msb = TrackEventKind::Midi {
        channel: ch,
        message: MidiMessage::Controller {
            controller: u7::new(0),
            value: u7::new((bank >> 7).min(127) as u8),
        },
    };
    let bank_lsb = TrackEventKind::Midi {
        channel: ch,
        message: MidiMessage::Controller {
            controller: u7::new(32),
            value: u7::new((bank & 0x7F) as u8),
        },
    };
    let prog = TrackEventKind::Midi {
        channel: ch,
        message: MidiMessage::ProgramChange { program: u7::new(program) },
    };
    events.push((0, 0, bank_msb));
    events.push((0, 0, bank_lsb));
    events.push((0, 0, prog));

    let mut has_note = false;
    for n in notes {
        if n.key > 127 || n.start < 0.0 || n.dur <= 0.0 {
            continue;
        }
        has_note = true;
        let on = (n.start * SEC_TO_TICK).round() as i64;
        let off = ((n.start + n.dur) * SEC_TO_TICK).round() as i64;
        let key = u7::new(n.key);
        let vel = u7::new(n.vel.min(127));
        events.push((
            on,
            2,
            TrackEventKind::Midi {
                channel: ch,
                message: MidiMessage::NoteOn { key, vel },
            },
        ));
        if off > on {
            events.push((
                off,
                0,
                TrackEventKind::Midi {
                    channel: ch,
                    message: MidiMessage::NoteOff { key, vel: u7::new(0) },
                },
            ));
        }
    }
    for c in cc {
        if c.start < 0.0 || c.cc > 127 || c.value > 127 {
            continue;
        }
        events.push((
            (c.start * SEC_TO_TICK).round() as i64,
            1,
            TrackEventKind::Midi {
                channel: ch,
                message: MidiMessage::Controller {
                    controller: u7::new(c.cc),
                    value: u7::new(c.value),
                },
            },
        ));
    }
    for b in bend {
        if b.start < 0.0 {
            continue;
        }
        // value ∈ [-8192, 8191]（0 = 居中）→ SMF 原始 14 bit（8192 = 居中）。
        let raw = (b.value.clamp(-8192, 8191) + 8192) as u16;
        events.push((
            (b.start * SEC_TO_TICK).round() as i64,
            1,
            TrackEventKind::Midi {
                channel: ch,
                message: MidiMessage::PitchBend { bend: PitchBend(u14::new(raw)) },
            },
        ));
    }
    if !has_note {
        return Err("没有可渲染的音符".into());
    }
    events.sort_by_key(|(t, cls, _)| (*t, *cls));

    let mut trk: Track = Vec::new();
    let mut cursor: i64 = 0;
    for (t, _, kind) in events {
        let t = t.max(0);
        trk.push(TrackEvent {
            delta: u28::new((t - cursor).max(0) as u32),
            kind,
        });
        cursor = t;
    }
    trk.push(TrackEvent {
        delta: u28::new(0),
        kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
    });

    // Track 0：tempo meta（120 BPM 是秒→tick 换算的另一半，必须与 SEC_TO_TICK 同轴）。
    let mut master: Track = Vec::new();
    master.push(TrackEvent {
        delta: u28::new(0),
        kind: TrackEventKind::Meta(MetaMessage::Tempo(u24::new(500_000))),
    });
    master.push(TrackEvent {
        delta: u28::new(0),
        kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
    });

    let mut smf = Smf::new(Header::new(Format::Parallel, Timing::Metrical(u15::new(480))));
    smf.tracks.push(master);
    smf.tracks.push(trk);
    smf.save(out_path)
        .map_err(|e| format!("EXPORT_WRITE_FAILED: write midi: {e} ({})", out_path.display()))?;
    Ok(())
}

/// `render_soundfont_notes` 的 FluidSynth 后端：用户音源库 SF2 → notes/cc/bend 经
/// 临时 SMF → CLI → WAV。SFZ 目录 / 非法音源在此显式报错（稳定 CODE，i18n 链路）。
/// `midi_tmp` 由调用方给定（通常与 WAV 同目录、同 hash 词干），渲染后无论成败都删除。
pub fn render_notes_to_wav(
    fonts_dir: &Path,
    font_id: &str,
    preset_id: &str,
    notes: &[RenderNote],
    cc: &[RenderCc],
    bend: &[RenderBend],
    sample_rate: u32,
    out_wav: &Path,
    models_dir: &Path,
    midi_tmp: &Path,
) -> Result<(), String> {
    let sid = super::soundfont::sanitize(font_id);
    let sf2 = fonts_dir.join(format!("{sid}.sf2"));
    if !sf2.is_file() {
        if fonts_dir.join(&sid).is_dir() {
            return Err("SOUNDFONT_BACKEND_UNSUPPORTED".into());
        }
        return Err("EXPORT_SOUNDFONT_NOT_FOUND".into());
    }
    let exe = exe_path(models_dir).ok_or("EXPORT_FLUIDSYNTH_NOT_FOUND")?;

    // 采样率白名单（44.1k..192k），越界回退 44.1k —— 与内置后端 clamp_sr 同约定。
    let sr = if (44100..=192000).contains(&sample_rate) { sample_rate } else { 44100 };
    let preset = AudioPreset {
        sample_rate: sr,
        format: if sr >= 48000 { "s24" } else { "s16" },
        suffix: "",
    };

    write_notes_midi(midi_tmp, notes, cc, bend, preset_id)?;
    let result = render(&exe, &sf2, midi_tmp, out_wav, &preset);
    let _ = std::fs::remove_file(midi_tmp);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn note(start: f64, dur: f64, key: u8, vel: u8) -> RenderNote {
        RenderNote { start, dur, key, vel }
    }

    /// 回读 SMF，抽出通道 0 的 (tick, MidiMessage) 平面序列（delta 已折叠为绝对 tick）。
    fn read_smf(path: &Path) -> Vec<(u32, midly::MidiMessage)> {
        let data = std::fs::read(path).unwrap();
        let smf = midly::Smf::parse(data.as_slice()).unwrap();
        assert_eq!(smf.tracks.len(), 2, "track0=tempo, track1=notes");
        let mut out = Vec::new();
        let mut cursor: u32 = 0;
        for ev in &smf.tracks[1] {
            cursor = cursor.wrapping_add(ev.delta.as_int());
            if let midly::TrackEventKind::Midi { channel, message } = ev.kind {
                assert_eq!(channel.as_int(), 0, "全部事件都该落在通道 0");
                out.push((cursor, message));
            }
        }
        out
    }

    #[test]
    fn notes_cc_bend_land_at_960_ticks_per_second() {
        let dir = std::env::temp_dir().join("utai_fs_midi_test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t1.mid");
        let notes = vec![note(0.0, 0.5, 60, 100), note(1.25, 1.0, 72, 64)];
        let ccs = vec![RenderCc { start: 0.0, cc: 11, value: 90 }];
        let bends = vec![RenderBend { start: 0.5, value: -300 }];
        write_notes_midi(&p, &notes, &ccs, &bends, "0:24").unwrap();

        let evs = read_smf(&p);
        // 头三件：bank MSB/LSB(=0) + ProgramChange(24)。
        assert!(matches!(&evs[0].1, midly::MidiMessage::Controller { controller, value: _ } if controller.as_int() == 0));
        assert!(matches!(&evs[1].1, midly::MidiMessage::Controller { controller, value: _ } if controller.as_int() == 32));
        assert!(matches!(&evs[2].1, midly::MidiMessage::ProgramChange { program } if program.as_int() == 24));
        // 0.5 秒 = 480 tick：CC11(0) → NoteOn60(0) → NoteOff60(480) → Bend(480) → NoteOn72(1200)。
        assert_eq!(evs[3].0, 0);
        assert!(matches!(&evs[3].1, midly::MidiMessage::Controller { controller, .. } if controller.as_int() == 11));
        assert_eq!(evs[4].0, 0);
        assert!(matches!(&evs[4].1, midly::MidiMessage::NoteOn { key, vel } if key.as_int() == 60 && vel.as_int() == 100));
        assert_eq!(evs[5].0, 480);
        assert!(matches!(&evs[5].1, midly::MidiMessage::NoteOff { key, .. } if key.as_int() == 60));
        assert_eq!(evs[6].0, 480);
        // as_int() 返回带符号偏移后的 i16（0 = 居中）：raw 7892 - 0x2000 = -300。
        assert!(matches!(&evs[6].1, midly::MidiMessage::PitchBend { bend } if bend.as_int() == -300));
        assert_eq!(evs[7].0, 1200);
        assert!(matches!(&evs[7].1, midly::MidiMessage::NoteOn { key, .. } if key.as_int() == 72));
        let _ = std::fs::remove_file(&p);
    }

    /// 同 tick 的收束/发声断口：前音 off 排在前音 class 0，后音 on 是 class 2 ——
    /// off 必须先于 on 出现（无重叠伪影），且 CC 恰好插在两者之间。
    #[test]
    fn same_tick_noteoff_precedes_noteon_and_cc_sits_between() {
        let dir = std::env::temp_dir().join("utai_fs_midi_test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t2.mid");
        let notes = vec![note(0.0, 0.25, 60, 100), note(0.25, 0.25, 67, 100)];
        let ccs = vec![RenderCc { start: 0.25, cc: 1, value: 40 }];
        write_notes_midi(&p, &notes, &ccs, &[], "3:1").unwrap();

        let evs = read_smf(&p);
        // 找 tick=240 的事件窗：顺序必须是 NoteOff(60) → CC1 → NoteOn(67)。
        let at_240: Vec<_> = evs.iter().filter(|(t, _)| *t == 240).collect();
        assert_eq!(at_240.len(), 3);
        assert!(matches!(&at_240[0].1, midly::MidiMessage::NoteOff { key, .. } if key.as_int() == 60));
        assert!(matches!(&at_240[1].1, midly::MidiMessage::Controller { controller, .. } if controller.as_int() == 1));
        assert!(matches!(&at_240[2].1, midly::MidiMessage::NoteOn { key, .. } if key.as_int() == 67));
        // bank:program = 3:1 → MSB 0 / LSB 3 / Program 1。
        assert!(matches!(&evs[1].1, midly::MidiMessage::Controller { controller, value } if controller.as_int() == 32 && value.as_int() == 3));
        assert!(matches!(&evs[2].1, midly::MidiMessage::ProgramChange { program } if program.as_int() == 1));
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn empty_or_all_invalid_notes_are_rejected() {
        let dir = std::env::temp_dir().join("utai_fs_midi_test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t3.mid");
        assert!(write_notes_midi(&p, &[], &[], &[], "0:0").is_err());
        // 全部越界（key > 127 / dur ≤ 0）→ 同样没有可渲染音符。
        let bad = vec![note(0.0, 0.5, 128, 100), note(0.0, 0.0, 60, 100)];
        assert!(write_notes_midi(&p, &bad, &[], &[], "0:0").is_err());
    }

    /// render_notes_to_wav 的音源解析分支：SFZ 目录 → SOUNDFONT_BACKEND_UNSUPPORTED；
    /// 完全不存在 → EXPORT_SOUNDFONT_NOT_FOUND。两者都必须先于 exe 检查（音源语义优先）。
    #[test]
    fn font_resolution_rejects_sfz_and_missing() {
        let dir = std::env::temp_dir().join("utai_fs_font_test");
        let _ = std::fs::remove_dir_all(&dir);
        let fonts = dir.join("fonts");
        std::fs::create_dir_all(fonts.join("piano_kit")).unwrap(); // SFZ 目录
        let models = dir.join("models");
        std::fs::create_dir_all(&models).unwrap();

        let wav = dir.join("out.wav");
        let mid = dir.join("out.mid");
        let notes = vec![note(0.0, 0.5, 60, 100)];

        let err = render_notes_to_wav(&fonts, "piano_kit", "0:0", &notes, &[], &[], 44100, &wav, &models, &mid)
            .unwrap_err();
        assert_eq!(err, "SOUNDFONT_BACKEND_UNSUPPORTED");

        let err = render_notes_to_wav(&fonts, "ghost_font", "0:0", &notes, &[], &[], 44100, &wav, &models, &mid)
            .unwrap_err();
        assert_eq!(err, "EXPORT_SOUNDFONT_NOT_FOUND");
        // 音源存在（哑文件即可过解析）但 exe 缺失 → 轮到 FluidSynth 检查报错。
        std::fs::write(fonts.join("tiny.sf2"), b"RIFFdummy").unwrap();
        let err = render_notes_to_wav(&fonts, "tiny", "0:0", &notes, &[], &[], 44100, &wav, &models, &mid)
            .unwrap_err();
        assert_eq!(err, "EXPORT_FLUIDSYNTH_NOT_FOUND");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
