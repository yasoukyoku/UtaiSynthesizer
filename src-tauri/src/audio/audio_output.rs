//! Muno 实时音频输出(阶段2,工程规划第七节)。
//!
//! cpal 默认输出设备上的**常驻实时流**,专用于音源试听(选音色 / 点击音符的
//! 低延迟反馈)。时间线播放仍走离线 bake → WAV → WebAudio(与人声轨同一管线,
//! `render_soundfont_notes`),本模块只承担「立刻听到这个音色」这一件事。
//!
//! 架构(命令线程 → 回调线程):
//! - 乐器加载(SFZ/SF2 解析 + 采样解码)全部在命令线程完成并缓存(`play_note`);
//!   回调里只做 note_on / note_off / `render_block`,无文件 I/O、无解码。
//! - 事件传递用 `std::sync::mpsc`:回调侧 `try_recv` 非阻塞;发送端(命令线程)
//!   分配消息节点、回调侧仅释放节点。共享模式设备(WASAPI shared / CoreAudio /
//!   ALSA)下安全,换来零自研无锁代码的 bug 面(用户铁律:没有任何 BUG)。
//! - `Synth` 在命令线程按设备采样率构造好再整体送入回调,回调零构造开销。
//!
//! 试听失败(无设备 / 格式不支持)→ `RtAudition::Unavailable`,命令层自动回退
//! `audition_soundfont_note` 的 WAV+WebAudio 路径;乐器加载失败则直接报错
//! (WAV 路径同样会失败,无回退意义)。

use std::collections::HashMap;
use std::path::Path;
use std::sync::mpsc::{self, Sender};
use std::sync::OnceLock;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use parking_lot::Mutex;

use super::soundfont;
use super::synth::{LoadedInstrument, Synth};

/// 回调内挂起的试听音符上限(超出丢最旧;整条丢弃,该音从不 note_on,永无卡音)。
const MAX_PENDING: usize = 128;
/// 已加载乐器缓存上限(超限逐出最旧且不在用的;LRU 简化版)。
const MAX_CACHED: usize = 4;
/// 回调内单次渲染分片(帧)。设备缓冲任意大小都按此分片渲染。
const CHUNK_FRAMES: usize = 512;

/// 命令线程 → 回调线程的消息。
enum CbMsg {
    /// 换乐器(命令线程已构造好;回调直接替换,旧音截断——试听语义可接受)。
    SetSynth(Synth),
    /// 试听一音(on 立即,off = on + dur)。
    NoteOn { key: u8, vel: u8, dur_secs: f32 },
}

/// 回调侧的音符触发目标(让调度器可脱离 Synth 单测)。
pub(super) trait NoteSink {
    fn note_on(&mut self, key: u8, vel: u8);
    fn note_off(&mut self, key: u8);
}

impl NoteSink for Synth {
    fn note_on(&mut self, key: u8, vel: u8) {
        Synth::note_on(self, key, vel);
    }
    fn note_off(&mut self, key: u8) {
        Synth::note_off(self, key);
    }
}

/// 已入队试听音符的调度(纯逻辑,可单测)。
#[derive(Debug)]
struct PendingNote {
    on_frame: u64,
    off_frame: u64,
    key: u8,
    vel: u8,
    started: bool,
}

#[derive(Debug, Default)]
struct NoteScheduler {
    notes: Vec<PendingNote>,
}

impl NoteScheduler {
    fn push(&mut self, on_frame: u64, dur_secs: f32, sample_rate: f32, key: u8, vel: u8) {
        if self.notes.len() >= MAX_PENDING {
            self.notes.remove(0); // 整条丢弃 → 从未 note_on → 永无悬挂 note_off
        }
        // dur 下限 0.02s:保证 off_frame > on_frame,且短于一个块也能被正确收尾。
        let dur_frames = (dur_secs.max(0.02) * sample_rate).round() as u64;
        self.notes.push(PendingNote {
            on_frame,
            off_frame: on_frame.saturating_add(dur_frames),
            key,
            vel,
            started: false,
        });
    }

    /// 触发到 `block_end` 为止到期的 on/off。
    /// 关键不变量:**每条被移除的音符都先收到过 note_off**——任何时序下都不会
    /// 产生卡音。同帧排序取 **off 先于 on**:同键重触发时,若 on 在前,紧随的
    /// off 会把刚出生的新 voice 直接打入 release(误杀);反序最多损失旧音的
    /// 一瞬重叠,不可闻。
    fn fire_due(&mut self, block_end: u64, sink: &mut dyn NoteSink) {
        // 收集到期动作:未开始的 on、以及**无论是否已开始**的 off
        // (迟到入队的音符可能 on 与 off 同块到期,off 不能因未 started 而漏检)。
        let mut due: Vec<(u64, bool, u8, u8)> = Vec::new(); // (frame, is_on, key, vel)
        for n in &self.notes {
            if !n.started && n.on_frame <= block_end {
                due.push((n.on_frame, true, n.key, n.vel));
            }
            if n.off_frame <= block_end {
                due.push((n.off_frame, false, n.key, 0));
            }
        }
        if due.is_empty() {
            return;
        }
        due.sort_by_key(|(f, is_on, _, _)| (*f, *is_on));
        for (_, is_on, key, vel) in due {
            if is_on {
                sink.note_on(key, vel);
            } else {
                sink.note_off(key);
            }
        }
        for n in &mut self.notes {
            if !n.started && n.on_frame <= block_end {
                n.started = true;
            }
        }
        self.notes.retain(|n| n.off_frame > block_end);
    }
}

/// 回调线程的全部状态(随闭包 move 进 cpal 流)。
struct CbState {
    rx: mpsc::Receiver<CbMsg>,
    sched: NoteScheduler,
    synth: Option<Synth>,
    frame: u64,
    sample_rate: f32,
    channels: usize,
}

/// cpal 数据回调:消息 → 调度触发 → 分片渲染 → 按设备声道分发。
/// 必须无 panic(回调栈跨越 FFI 边界);路径上无 unwrap/除零/越界索引。
fn process<T>(state: &mut CbState, data: &mut [T])
where
    T: cpal::Sample + cpal::FromSample<f32>,
{
    let mut scratch = [0f32; CHUNK_FRAMES * 2];
    let channels = state.channels.max(1);
    let total = data.len() / channels;
    let mut done = 0usize;
    while done < total {
        let n = CHUNK_FRAMES.min(total - done);
        // 取消息(非阻塞;每分片一次足够——人手速远低于块率)
        while let Ok(msg) = state.rx.try_recv() {
            match msg {
                CbMsg::SetSynth(s) => {
                    state.synth = Some(s);
                    state.sched.notes.clear(); // 换乐器:残留音符不跨乐器送 note_off
                }
                CbMsg::NoteOn { key, vel, dur_secs } => {
                    state.sched.push(state.frame, dur_secs, state.sample_rate, key, vel);
                }
            }
        }
        let block_end = state.frame + n as u64;
        // 拆字段借用:sched 与 synth 互不冲突
        let CbState { sched, synth, .. } = state;
        if let Some(s) = synth.as_mut() {
            sched.fire_due(block_end, s);
            let out = &mut scratch[..n * 2];
            out.fill(0.0); // render_block 是叠加语义,分片必须清零(防跨片累积)
            s.render_block(out);
        }
        // 按设备声道分发
        let base = done * channels;
        if channels == 2 {
            for f in 0..n {
                data[base + f * 2] = T::from_sample(scratch[f * 2]);
                data[base + f * 2 + 1] = T::from_sample(scratch[f * 2 + 1]);
            }
        } else if channels == 1 {
            for f in 0..n {
                data[base + f] = T::from_sample(0.5 * (scratch[f * 2] + scratch[f * 2 + 1]));
            }
        } else {
            for f in 0..n {
                for c in 0..channels {
                    let v = match c {
                        0 => scratch[f * 2],
                        1 => scratch[f * 2 + 1],
                        _ => 0.0,
                    };
                    data[base + f * channels + c] = T::from_sample(v);
                }
            }
        }
        state.frame += n as u64;
        done += n;
    }
}

/// 实时试听引擎:持有乐器缓存与消息发送端(全部 Send,可进全局静态)。
/// 仅经 `try_rt_audition` 使用。
pub struct AuditionEngine {
    tx: Sender<CbMsg>,
    sample_rate: f32,
    /// 乐器缓存与「当前已送进回调的乐器」(key = "fontId|presetId")。仅命令线程访问。
    cache: HashMap<String, ArcInstr>,
    cache_order: Vec<String>,
    current: String,
}

type ArcInstr = std::sync::Arc<LoadedInstrument>;

impl AuditionEngine {
    /// 打开默认输出设备,启动常驻回调。失败返回 Err(调用方回退 WAV 路径)。
    ///
    /// ⚠ cpal 0.15 的 `Stream` 刻意 `!Send`(全平台统一标记,ASIO 健全性),
    /// 不能进全局静态。而本引擎的生存期**本来就该等于进程生存期**(常驻试听
    /// 流,随用随响)——所以 `play()` 成功后 `mem::forget` 保活:不 Drop 就
    /// 永不停止,进程退出时由 OS 收回 WASAPI 会话。每次进程内只初始化一次
    /// (全局静态装 None→Some 一次),无重复泄漏。
    pub fn new() -> Result<Self, String> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| "未找到输出设备".to_string())?;
        let supported = device
            .default_output_config()
            .map_err(|e| format!("读取设备配置失败: {}", e))?;
        if supported.channels() == 0 {
            return Err("设备声道数无效".into());
        }
        let sample_rate = supported.sample_rate().0 as f32;
        let channels = supported.channels() as usize;
        let stream_cfg: cpal::StreamConfig = supported.clone().into();
        let (tx, rx) = mpsc::channel::<CbMsg>();
        let mut state = CbState {
            rx,
            sched: NoteScheduler::default(),
            synth: None,
            frame: 0,
            sample_rate,
            channels,
        };
        let err_cb = |e| tracing::warn!("[audition] 音频流错误: {}", e);
        // 三个采样格式臂各自 move state(互斥执行,borrowck 允许)。第 4 参 timeout=None:
        // 空闲时让回调自然按设备节奏走,不人为加唤醒超时。
        let stream = match supported.sample_format() {
            cpal::SampleFormat::F32 => device
                .build_output_stream(
                    &stream_cfg,
                    move |d: &mut [f32], _| process(&mut state, d),
                    err_cb,
                    None,
                )
                .map_err(|e| format!("打开音频流失败: {}", e))?,
            cpal::SampleFormat::I16 => device
                .build_output_stream(
                    &stream_cfg,
                    move |d: &mut [i16], _| process(&mut state, d),
                    err_cb,
                    None,
                )
                .map_err(|e| format!("打开音频流失败: {}", e))?,
            cpal::SampleFormat::U16 => device
                .build_output_stream(
                    &stream_cfg,
                    move |d: &mut [u16], _| process(&mut state, d),
                    err_cb,
                    None,
                )
                .map_err(|e| format!("打开音频流失败: {}", e))?,
            _ => return Err("设备采样格式不受支持".into()),
        };
        stream.play().map_err(|e| format!("启动音频流失败: {}", e))?;
        // 常驻保活(见类型注释)。play() 失败则正常 Drop 关闭,不 forget。
        std::mem::forget(stream);
        Ok(Self {
            tx,
            sample_rate,
            cache: HashMap::new(),
            cache_order: Vec::new(),
            current: String::new(),
        })
    }

    /// 试听一音:乐器加载(带缓存,大 SFZ 只在首次付出解码成本)→
    /// 换乐器时整体送入新 Synth → 送音符事件。发送失败视为流已死,由调用方回退。
    fn play_note(
        &mut self,
        dir: &Path,
        font_id: &str,
        preset_id: &str,
        key: u8,
        vel: u8,
        dur_secs: f32,
    ) -> Result<(), String> {
        let cache_key = format!("{}|{}", font_id, preset_id);
        let instr = match self.cache.get(&cache_key) {
            Some(i) => i.clone(),
            None => {
                let i = std::sync::Arc::new(soundfont::load_instrument(dir, font_id, preset_id)?);
                if self.cache.len() >= MAX_CACHED {
                    // 逐出最旧的**非当前**乐器(当前在回调里响着,不动)
                    if let Some(pos) = self.cache_order.iter().position(|k| *k != self.current) {
                        let old = self.cache_order.remove(pos);
                        self.cache.remove(&old);
                    }
                }
                self.cache_order.push(cache_key.clone());
                self.cache.insert(cache_key.clone(), i.clone());
                i
            }
        };
        if self.current != cache_key {
            let synth = Synth::new(&instr, self.sample_rate);
            self.tx
                .send(CbMsg::SetSynth(synth))
                .map_err(|_| "实时音频流已关闭".to_string())?;
            self.current = cache_key;
        }
        self.tx
            .send(CbMsg::NoteOn { key, vel, dur_secs })
            .map_err(|_| "实时音频流已关闭".to_string())?;
        Ok(())
    }
}

/// 实时试听结果。
pub enum RtAudition {
    /// 已通过 cpal 常驻流发声。
    Played,
    /// 实时引擎不可用(无设备 / 格式不支持)→ 调用方回退 WAV+WebAudio。
    Unavailable,
}

/// 全局唯一引擎(Mutex<Option<…>>:Stream 是 Send 但非 Sync,装 Mutex 后整静态可 Sync)。
static ENGINE: OnceLock<Mutex<Option<AuditionEngine>>> = OnceLock::new();

/// 实时试听入口(命令层调用)。引擎首次失败不缓存失败态——下次点击重试,
/// 设备后来插上即可自愈。
pub fn try_rt_audition(
    dir: &Path,
    font_id: &str,
    preset_id: &str,
    key: u8,
    vel: u8,
    dur_secs: f32,
) -> Result<RtAudition, String> {
    let m = ENGINE.get_or_init(|| Mutex::new(None));
    let mut g = m.lock();
    if g.is_none() {
        match AuditionEngine::new() {
            Ok(e) => *g = Some(e),
            Err(err) => {
                tracing::debug!("[audition] realtime engine unavailable, falling back to WAV playback: {}", err);
                return Ok(RtAudition::Unavailable);
            }
        }
    }
    let engine = g.as_mut().expect("引擎刚初始化");
    engine.play_note(dir, font_id, preset_id, key, vel, dur_secs)?;
    Ok(RtAudition::Played)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 记录型 NoteSink:按序记录 on/off,供断言。
    struct Rec {
        events: Vec<(u64, bool, u8, u8)>,
    }
    impl NoteSink for Rec {
        fn note_on(&mut self, key: u8, vel: u8) {
            self.events.push((0, true, key, vel));
        }
        fn note_off(&mut self, key: u8) {
            self.events.push((0, false, key, 0));
        }
    }

    #[test]
    fn scheduler_fires_on_then_off_across_blocks() {
        let mut s = NoteScheduler::default();
        s.push(0, 0.1, 48000.0, 60, 100); // on=0, off=4800
        let mut r = Rec { events: vec![] };
        s.fire_due(480, &mut r); // 第一块:on 到期
        assert_eq!(r.events.len(), 1);
        assert!(r.events[0].1, "第一块只有 on");
        s.fire_due(960, &mut r); // 第二块:无到期
        assert_eq!(r.events.len(), 1);
        s.fire_due(5280, &mut r); // 第三块:off 到期
        assert_eq!(r.events.len(), 2);
        assert!(!r.events[1].1, "off 第二个触发");
        assert!(s.notes.is_empty(), "结束后应清空");
    }

    #[test]
    fn scheduler_overdue_note_never_stuck() {
        // 音符入队时 on/off 都已过期(回调延迟)→ 同块内 on 先于 off 触发,无卡音。
        let mut s = NoteScheduler::default();
        s.push(0, 0.1, 48000.0, 60, 100); // on=0, off=4800
        let mut r = Rec { events: vec![] };
        s.fire_due(u64::MAX, &mut r);
        assert_eq!(r.events.len(), 2, "on 和 off 都要触发");
        assert!(r.events[0].1, "on 在前");
        assert!(!r.events[1].1, "off 在后");
        assert!(s.notes.is_empty());
    }

    #[test]
    fn scheduler_retrigger_order_same_frame() {
        // 同键重触发:旧音 off 与新音 on 同帧时,off 必须在前(on 在前会被
        // 紧随的 off 误杀新 voice)。dur 下限 0.02s → 48000Hz 下 960 帧。
        let mut s = NoteScheduler::default();
        s.push(0, 0.02, 48000.0, 60, 100); // on=0, off=960
        s.push(960, 0.02, 48000.0, 60, 110); // on=960, off=1920
        let mut r = Rec { events: vec![] };
        s.fire_due(960, &mut r);
        // 帧序:on(0) → off(960) → on(960):同帧 off 先于 on
        assert_eq!(r.events.len(), 3, "应触发 3 个动作");
        assert!(r.events[0].1 && r.events[0].3 == 100, "第一动是旧音的 on");
        assert!(!r.events[1].1, "旧音的 off 必须在新 on 之前");
        assert!(r.events[2].1 && r.events[2].3 == 110, "最后是新音的 on");
        assert_eq!(s.notes.len(), 1, "只留第二个音");
    }

    #[test]
    fn scheduler_cap_drops_oldest_whole() {
        let mut s = NoteScheduler::default();
        for i in 0..(MAX_PENDING + 8) {
            s.push(i as u64 * 48000, 0.1, 48000.0, 60, 100);
        }
        assert_eq!(s.notes.len(), MAX_PENDING, "挂起上限");
        // 全部到期触发:on 数 = 存活数(被丢弃的最旧条从未 on,也永远不卡)
        let mut r = Rec { events: vec![] };
        s.fire_due(u64::MAX, &mut r);
        let ons = r.events.iter().filter(|e| e.1).count();
        let offs = r.events.iter().filter(|e| !e.1).count();
        assert_eq!(ons, MAX_PENDING);
        assert_eq!(offs, MAX_PENDING, "每个 on 都有 off —— 无卡音");
        assert!(s.notes.is_empty());
    }

    #[test]
    fn scheduler_dur_floor_keeps_off_after_on() {
        let mut s = NoteScheduler::default();
        s.push(100, 0.0, 48000.0, 60, 100); // dur=0 → 下限 0.02s
        let n = &s.notes[0];
        assert!(n.off_frame > n.on_frame, "off 必须严格晚于 on");
    }
}
