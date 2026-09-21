//! Muno 原生合成引擎核心(阶段2)。
//!
//! 统一的 Region / Voice / Synth 模型:SFZ 与 SF2 两种音源格式都解析成
//! `Vec<Region>` + `Vec<Arc<SampleData>>`,由同一个多音色采样回放引擎渲染。
//! 离线渲染(bake 成 WAV,前端 WebAudio 播放)与实时试听共用本引擎。
//!
//! 设计约束(工程规划第十部分):
//! - 渲染线程无锁、无每帧分配:包络参数在 spawn 时拷贝进 Voice,
//!   渲染内循环只读 voices + samples,不碰 regions。
//! - 插值:4 点三次 Hermite(音质/成本折中,优于线性)。
//! - 包络:线性段 AHDSR(attack/hold/decay/sustain/release),
//!   release 段指数衰减(-100 dB / release,听感自然,线性"断得很硬")。
//! - 滤波:可选二阶低通(RBJ biquad),在包络之前(SF2 规范信号顺序)。
//! - 音色映射:key/vel 区间、pitch_keycenter、track、tune、transpose。

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

/// 循环模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LoopMode {
    #[default]
    NoLoop,
    /// 一直循环(loop_start..loop_end)。
    Continuous,
    /// 按住时循环,松开后走出循环播到采样尾。
    Sustain,
    /// 一次性播完采样,note-off 不截断。
    OneShot,
}

/// 触发条件。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Trigger {
    #[default]
    Attack,
    Release,
    First,
    Legato,
}

/// 一条映射规则(SFZ <region> / SF2 instrument zone 的统一内部表示)。
#[derive(Debug, Clone)]
pub struct Region {
    /// 采样来源描述(仅用于日志:SFZ 为文件路径,SF2 为伪路径)。
    pub sample_path: PathBuf,
    pub key_lo: u8,
    pub key_hi: u8,
    pub vel_lo: u8,
    pub vel_hi: u8,
    /// 根音(未设 = 60;SF2 用 overridingRootKey 或 shdr 原调)。
    pub keycenter: Option<u8>,
    /// 音高跟随(SFZ pitch_track / SF2 scaleTuning=0;默认 true)。
    pub track: bool,
    /// 整数半音移调。
    pub transpose: i32,
    /// 音分微调。
    pub tune: f32,
    /// 区域音量 dB。
    pub volume_db: f32,
    /// 声像 -1..1。
    pub pan: f32,
    /// 速度灵敏度 %(SFZ amp_veltrack;SF2 恒 100)。
    pub veltrack: f32,
    pub attack: f32,
    pub hold: f32,
    pub decay: f32,
    /// sustain 电平 0..1(1 = 持满)。
    pub sustain: f32,
    pub release: f32,
    pub loop_mode: LoopMode,
    /// 循环点(采样帧;None = 用采样自带循环)。
    pub loop_start: Option<u32>,
    pub loop_end: Option<u32>,
    /// 采样结束帧(None = 自然尾)。
    pub end: Option<u32>,
    /// 采样起始帧。
    pub offset: u32,
    pub seq_length: u32,
    pub seq_position: u32,
    /// 排他组(off_by)。
    pub off_by: Option<u32>,
    pub group: u32,
    pub trigger: Trigger,
    /// keyswitch(简化:仅记录最后按键)。
    pub sw_last: Option<u8>,
    /// SF2 额外:初始衰减(cB;与 volume_db 独立叠加)。
    pub attenuation_cb: f32,
    /// 二阶低通截止频率 Hz(None = 直通;SF2 initialFilterFc / SFZ cutoff)。
    pub cutoff_hz: Option<f32>,
    /// 低通谐振 Q(线性;SF2 initialFilterQ cB / SFZ resonance dB 换算)。
    pub resonance_q: f32,
    /// SF2 modulator:velocity → initialFilterFc 偏移(cents;vel=127 不偏移,vel=0 全量)。
    pub vel2filter_cents: f32,
    /// CC1 vibrato 深度上限(cents;实际深度 = 上限 × CC1;SF2 vibLfoToPitch / 默认 50)。
    pub vib_depth_cents: f32,
    /// SFZ 交叉淡入淡出:键位入口/出口 ramp 区间(hi <= lo = 未启用)。
    pub xf_key_in: (u8, u8),
    pub xf_key_out: (u8, u8),
    /// SFZ 交叉淡入淡出:速度入口/出口 ramp 区间。
    pub xf_vel_in: (u8, u8),
    pub xf_vel_out: (u8, u8),
    /// 交叉淡入曲线:false = linear,true = power(等功率 sin/cos)。
    pub xf_key_power: bool,
    pub xf_vel_power: bool,
}

impl Default for Region {
    fn default() -> Self {
        Self {
            sample_path: PathBuf::new(),
            key_lo: 0,
            key_hi: 127,
            vel_lo: 0,
            vel_hi: 127,
            keycenter: None,
            track: true,
            transpose: 0,
            tune: 0.0,
            volume_db: 0.0,
            pan: 0.0,
            veltrack: 100.0,
            attack: 0.0,
            hold: 0.0,
            decay: 0.0,
            sustain: 1.0,
            release: 0.1,
            loop_mode: LoopMode::NoLoop,
            loop_start: None,
            loop_end: None,
            end: None,
            offset: 0,
            seq_length: 1,
            seq_position: 1,
            off_by: None,
            group: 0,
            trigger: Trigger::Attack,
            sw_last: None,
            attenuation_cb: 0.0,
            cutoff_hz: None,
            resonance_q: 1.0,
            vel2filter_cents: 0.0,
            vib_depth_cents: 50.0,
            xf_key_in: (0, 0),
            xf_key_out: (127, 127),
            xf_vel_in: (0, 0),
            xf_vel_out: (127, 127),
            xf_key_power: false,
            xf_vel_power: false,
        }
    }
}

/// 已解码的采样数据(平面存储,mono 用 left)。
#[derive(Debug)]
pub struct SampleData {
    pub left: Vec<f32>,
    pub right: Option<Vec<f32>>,
    pub rate: u32,
    /// 采样自带循环点(wav smpl chunk / SF2 shdr)。
    pub loop_range: Option<(u32, u32)>,
    /// 原始根音(SF2 shdr byOriginalPitch / wav smpl)。
    pub root_key: Option<u8>,
    pub fine_tune_cents: f32,
}

impl SampleData {
    pub fn frame_count(&self) -> u32 {
        self.left.len() as u32
    }
}

/// 一个乐器:区域 + 与区域等长的采样。
#[derive(Clone)]
pub struct LoadedInstrument {
    pub regions: Vec<Region>,
    pub samples: Vec<Arc<SampleData>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EnvPhase {
    Attack,
    Hold,
    Decay,
    Sustain,
    Release,
    Done,
}

/// 一个正在发声的音符。包络/循环等区域参数在 spawn 时拷贝进来,
/// 让渲染内循环完全不借用 regions。
struct Voice {
    region_idx: usize,
    key: u8,
    /// 播放位置(采样帧,f64 便于小步进)。
    pos: f64,
    /// 每输出帧的采样步进。
    step: f64,
    phase: EnvPhase,
    /// 包络当前值。
    env: f32,
    /// hold 阶段累计时间。
    hold_t: f32,
    /// release 起始电平。
    release_level: f32,
    /// 速度增益(线性)。
    vel_gain: f32,
    /// 声像×音量增益。
    lgain: f32,
    rgain: f32,
    /// 包络参数(spawn 时拷贝)。
    atk: f32,
    hld: f32,
    dcy: f32,
    sus: f32,
    rel: f32,
    /// 采样结束帧(spawn 时解析)。
    end_frame: f64,
    /// 实际循环区间(已定)。
    loop_range: Option<(u32, u32)>,
    /// loop_mode == Continuous。
    loop_continuous: bool,
    /// loop_mode == Sustain。
    sustain_loop: bool,
    /// sustain 循环已松键(不再回环)。
    loop_broken: bool,
    /// one_shot:忽略 note-off,播到采样尾。
    one_shot: bool,
    /// 指数释放每帧乘子(进入 Release 时计算)。
    release_rate: f32,
    /// 二阶低通状态(None = 直通;spawn 时从 Region 初始化)。
    filter: Option<BiquadLp>,
    /// CC1 modulation LFO 相位(0..2π)。
    lfo_phase: f32,
    /// 本 voice 的 vibrato 深度上限(cents,spawn 时取自 region.vib_depth_cents)。
    lfo_depth: f32,
    /// sustain 踏板踩着时松的键:挂着等抬踏板。
    held_by_pedal: bool,
}

/// 事件(离线渲染前按 frame 排序)。
#[derive(Debug, Clone, Copy)]
pub enum MidiEvent {
    NoteOn { frame: u64, key: u8, vel: u8 },
    NoteOff { frame: u64, key: u8 },
    /// 弯音:value ∈ [-8192, 8191],0 = 居中,极值 = ±bend_range 半音。
    PitchBend { frame: u64, value: i16 },
    /// 控制器:CC1 modulation(颤音)/ CC11 expression / CC64 sustain。
    ControlChange { frame: u64, cc: u8, value: u8 },
}

impl MidiEvent {
    pub fn frame(&self) -> u64 {
        match self {
            MidiEvent::NoteOn { frame, .. }
            | MidiEvent::NoteOff { frame, .. }
            | MidiEvent::PitchBend { frame, .. }
            | MidiEvent::ControlChange { frame, .. } => *frame,
        }
    }
}

/// 合成器:一个乐器的区域/采样 + 活跃 Voice。
pub struct Synth {
    regions: Vec<Region>,
    samples: Vec<Arc<SampleData>>,
    voices: Vec<Voice>,
    /// (group, key) → round-robin 计数。
    seq_state: HashMap<(u32, u8), u32>,
    /// 最后按下的 keyswitch 键(全局)。
    sw_last_key: Option<u8>,
    sample_rate: f32,
    max_voices: usize,
    /// 弯音当前 cents(0 = 居中)。
    bend_cents: f32,
    /// 弯音范围(±半音,默认 ±2)。
    bend_range: f32,
    /// CC1 值(0..1;实际 vibrato 深度 = voice.lfo_depth × lfo_scale)。
    lfo_scale: f32,
    /// CC64 sustain 踏板。
    sustain_on: bool,
    /// CC11 expression 目标(0..1)。
    expr_target: f32,
    /// expression 平滑当前值(一阶低通,20 ms 时间常数)。
    expression: f32,
    /// expression 一阶平滑系数(1 − exp(−dt/τ))。
    expr_coef: f32,
}

fn vel_gain(vel: u8, veltrack: f32) -> f32 {
    // SFZ 规范:attenuation(dB)= 40·log10(vel/127)·(veltrack/100)
    let vt = (veltrack / 100.0).clamp(-1.0, 1.0);
    if vt == 0.0 {
        return 1.0;
    }
    let v = (vel.max(1) as f32 / 127.0).clamp(1.0 / 127.0, 1.0);
    v.powf(vt * 2.0)
}

pub fn pan_gains(pan: f32) -> (f32, f32) {
    let p = pan.clamp(-1.0, 1.0);
    if p >= 0.0 {
        (1.0 - 0.75 * p, 1.0)
    } else {
        (1.0, 1.0 + 0.75 * p)
    }
}

/// SF2 initialFilterFc(absolute cents)→ Hz(规范:0 cents = 8.176 Hz)。
pub fn sf2_fc_cents_to_hz(cents: f32) -> f32 {
    8.176 * 2f32.powf(cents / 1200.0)
}

/// SF2 initialFilterQ(cB)→ 线性 Q。
pub fn sf2_q_cb_to_linear(cb: f32) -> f32 {
    10f32.powf(cb / 200.0)
}

/// 指数释放每帧乘子:release 秒内衰减 100 dB(SF2 音量包络惯例)。
fn release_decay_rate(rel: f32, sample_rate: f32) -> f32 {
    let k = -5.0 * std::f32::consts::LN_10 / (rel.max(1e-4) * sample_rate);
    k.exp().clamp(0.0, 1.0)
}

/// 4 点 Hermite 插值。
#[inline]
fn interp(buf: &[f32], pos: f64) -> f32 {
    let len = buf.len();
    if len == 0 {
        return 0.0;
    }
    if pos <= 0.0 {
        return buf[0];
    }
    let i = pos as usize;
    if i >= len - 1 {
        return buf[len - 1];
    }
    let frac = (pos - i as f64) as f32;
    let y0 = if i > 0 { buf[i - 1] } else { buf[0] };
    let y1 = buf[i];
    let y2 = buf[i + 1];
    let y3 = if i + 2 < len { buf[i + 2] } else { y2 };
    let c1 = 0.5 * (y2 - y0);
    let c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
    let c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2);
    ((c3 * frac + c2) * frac + c1) * frac + y1
}

/// 二阶低通 biquad(RBJ cookbook,直接 I 型)。
/// 左右通道状态独立;mono 采样只走左通道状态(两路输出同值)。
#[derive(Debug, Clone, Copy)]
struct BiquadLp {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    lx1: f32,
    lx2: f32,
    ly1: f32,
    ly2: f32,
    rx1: f32,
    rx2: f32,
    ry1: f32,
    ry2: f32,
}

impl BiquadLp {
    fn new(cutoff_hz: f32, q: f32, sample_rate: f32) -> Self {
        let nyquist = sample_rate * 0.5;
        let fc = cutoff_hz.clamp(20.0, nyquist * 0.45);
        let w0 = 2.0 * std::f32::consts::PI * fc / sample_rate;
        let (sin_w, cos_w) = w0.sin_cos();
        let alpha = sin_w / (2.0 * q.max(0.05));
        let a0 = 1.0 + alpha;
        let two_cos = 2.0 * cos_w;
        Self {
            b0: (1.0 - cos_w) / 2.0 / a0,
            b1: (1.0 - cos_w) / a0,
            b2: (1.0 - cos_w) / 2.0 / a0,
            a1: -two_cos / a0,
            a2: (1.0 - alpha) / a0,
            lx1: 0.0,
            lx2: 0.0,
            ly1: 0.0,
            ly2: 0.0,
            rx1: 0.0,
            rx2: 0.0,
            ry1: 0.0,
            ry2: 0.0,
        }
    }

    /// 处理一帧;mono(stereo=false)时右路复用左路结果。
    #[inline]
    fn process(&mut self, stereo: bool, l_in: f32, r_in: f32) -> (f32, f32) {
        let l = self.b0 * l_in + self.b1 * self.lx1 + self.b2 * self.lx2
            - self.a1 * self.ly1
            - self.a2 * self.ly2;
        self.lx2 = self.lx1;
        self.lx1 = l_in;
        self.ly2 = self.ly1;
        self.ly1 = l;
        let r = if stereo {
            let r = self.b0 * r_in + self.b1 * self.rx1 + self.b2 * self.rx2
                - self.a1 * self.ry1
                - self.a2 * self.ry2;
            self.rx2 = self.rx1;
            self.rx1 = r_in;
            self.ry2 = self.ry1;
            self.ry1 = r;
            r
        } else {
            l
        };
        (l, r)
    }
}

/// SF2 modulator:velocity → initialFilterFc 的截止频率乘子
/// (vel=127 → 1 不偏移;vel 越低偏移越大,vel=0 全量;负 cents = 低速更暗)。
fn vel2filter_mul(cents: f32, vel: u8) -> f32 {
    2f32.powf(cents * (1.0 - vel as f32 / 127.0) / 1200.0)
}

/// SFZ 交叉淡入淡出增益:xfin(incoming)随值从 lo→hi 由 0 升到 1,xfout 反向 1→0。
/// hi <= lo 视为未启用(恒 1);power = 等功率曲线(sin/cos),否则线性。
fn xf_gain(v: u8, range: (u8, u8), incoming: bool, power: bool) -> f32 {
    let (lo, hi) = range;
    if hi <= lo {
        return 1.0;
    }
    let t = ((v as i32 - lo as i32) as f32 / (hi - lo) as f32).clamp(0.0, 1.0);
    match (incoming, power) {
        (true, false) => t,
        (false, false) => 1.0 - t,
        (true, true) => (t * std::f32::consts::FRAC_PI_2).sin(),
        (false, true) => (t * std::f32::consts::FRAC_PI_2).cos(),
    }
}

impl Synth {
    pub fn new(instr: &LoadedInstrument, sample_rate: f32) -> Self {
        Self {
            regions: instr.regions.clone(),
            samples: instr.samples.clone(),
            voices: Vec::new(),
            seq_state: HashMap::new(),
            sw_last_key: None,
            sample_rate,
            max_voices: 256,
            bend_cents: 0.0,
            bend_range: 2.0,
            lfo_scale: 0.0,
            sustain_on: false,
            expr_target: 1.0,
            expression: 1.0,
            expr_coef: 1.0 - (-1.0 / (0.02 * sample_rate)).exp(),
        }
    }

    fn match_region(&mut self, key: u8, vel: u8, is_release: bool) -> Option<usize> {
        let mut candidates: Vec<usize> = Vec::new();
        for (i, r) in self.regions.iter().enumerate() {
            if key < r.key_lo || key > r.key_hi || vel < r.vel_lo || vel > r.vel_hi {
                continue;
            }
            match r.trigger {
                Trigger::Attack | Trigger::First | Trigger::Legato => {
                    if is_release {
                        continue;
                    }
                }
                Trigger::Release => {
                    if !is_release {
                        continue;
                    }
                }
            }
            if let Some(sw) = r.sw_last {
                if self.sw_last_key != Some(sw) {
                    continue;
                }
            }
            candidates.push(i);
        }
        if candidates.is_empty() {
            return None;
        }
        // round-robin(seq_length>1 的组内轮转)
        let with_seq: Vec<usize> = candidates
            .iter()
            .copied()
            .filter(|&i| self.regions[i].seq_length > 1)
            .collect();
        if !with_seq.is_empty() {
            let r0 = &self.regions[with_seq[0]];
            let st = self.seq_state.entry((r0.group, key)).or_insert(0);
            let pos = (*st % r0.seq_length) + 1;
            *st = (*st + 1) % r0.seq_length;
            if let Some(&found) = with_seq.iter().find(|&&i| self.regions[i].seq_position == pos) {
                return Some(found);
            }
        }
        Some(candidates[0])
    }

    pub fn note_on(&mut self, key: u8, vel: u8) {
        let Some(ri) = self.match_region(key, vel, false) else {
            return;
        };
        self.spawn_voice(ri, key, vel);
    }

    /// note-off:对活跃 voice 送入 release(踩着 sustain 踏板则挂起);并触发 release 类区域。
    pub fn note_off(&mut self, key: u8) {
        let pedal = self.sustain_on;
        for v in self.voices.iter_mut() {
            if v.key == key && !v.one_shot && v.phase != EnvPhase::Release && v.phase != EnvPhase::Done {
                if pedal {
                    // 踏板踩着:挂起,等抬踏板再释放
                    v.held_by_pedal = true;
                    continue;
                }
                if v.sustain_loop && v.loop_range.is_some() {
                    // loop_sustain:走出循环,播到采样尾
                    v.loop_broken = true;
                }
                v.phase = EnvPhase::Release;
                v.release_level = v.env;
                v.release_rate = release_decay_rate(v.rel, self.sample_rate);
            }
        }
        // release 触发区域(松键音色,如钢琴踏板采样)
        if let Some(ri) = self.match_region(key, 100, true) {
            self.spawn_voice(ri, key, 100);
        }
    }

    /// 弯音:value ∈ [-8192, 8191],0 = 居中。
    pub fn pitch_bend(&mut self, value: i16) {
        let v = (value as f32 / 8192.0).clamp(-1.0, 1.0);
        self.bend_cents = v * self.bend_range * 100.0;
    }

    /// 控制器:CC1 modulation / CC11 expression / CC64 sustain。
    pub fn control_change(&mut self, cc: u8, value: u8) {
        match cc {
            1 => self.lfo_scale = value as f32 / 127.0,
            11 => self.expr_target = value as f32 / 127.0,
            64 => {
                let down = value >= 64;
                if !down && self.sustain_on {
                    // 抬踏板:释放所有被挂住的 voice
                    for v in self.voices.iter_mut() {
                        if v.held_by_pedal {
                            v.held_by_pedal = false;
                            if v.sustain_loop && v.loop_range.is_some() {
                                v.loop_broken = true;
                            }
                            v.phase = EnvPhase::Release;
                            v.release_level = v.env;
                            v.release_rate = release_decay_rate(v.rel, self.sample_rate);
                        }
                    }
                }
                self.sustain_on = down;
            }
            _ => {}
        }
    }

    fn spawn_voice(&mut self, region_idx: usize, key: u8, vel: u8) {
        let region = self.regions[region_idx].clone();
        let sample_rate = self.sample_rate;
        let sample = match self.samples.get(region_idx) {
            Some(s) => s.clone(),
            None => return,
        };
        // off_by:关闭同组其它 voice
        if let Some(off_group) = region.off_by {
            for v in self.voices.iter_mut() {
                if self.regions[v.region_idx].group == off_group
                    && v.phase != EnvPhase::Release
                    && v.phase != EnvPhase::Done
                {
                    v.phase = EnvPhase::Release;
                    v.release_level = v.env;
                    v.release_rate = release_decay_rate(v.rel, self.sample_rate);
                }
            }
        }
        let root = region.keycenter.or(sample.root_key).unwrap_or(60);
        let semis = if region.track {
            key as i32 - root as i32 + region.transpose
        } else {
            region.transpose
        } as f64;
        let cents = region.tune as f64 + sample.fine_tune_cents as f64;
        let ratio =
            2f64.powf((semis + cents / 100.0) / 12.0) * (sample.rate as f64 / self.sample_rate as f64);
        let end = region
            .end
            .filter(|e| (*e as usize) <= sample.left.len() && *e > region.offset)
            .unwrap_or_else(|| sample.frame_count());
        // 循环区间:region 覆盖 > 采样自带;必须落在 [offset, end] 内
        let loop_range = region
            .loop_start
            .zip(region.loop_end)
            .filter(|(s, e)| *e > *s && (*e as usize) <= sample.left.len() && *e <= end && *s >= region.offset)
            .or(sample.loop_range)
            .filter(|(s, e)| *e > *s && (*e as usize) <= sample.left.len() && *e <= end && *s >= region.offset);
        let attenuation = 10f32.powf(-region.attenuation_cb / 200.0);
        let vol = 10f32.powf(region.volume_db / 20.0) * attenuation;
        let (lgain, rgain) = pan_gains(region.pan);
        // SFZ 交叉淡入淡出:键位/速度 ramp 与区域主区间正交(刻意重叠,渐变交接)
        let xf = xf_gain(key, region.xf_key_in, true, region.xf_key_power)
            * xf_gain(key, region.xf_key_out, false, region.xf_key_power)
            * xf_gain(vel, region.xf_vel_in, true, region.xf_vel_power)
            * xf_gain(vel, region.xf_vel_out, false, region.xf_vel_power);
        let vel_g = vel_gain(vel, region.veltrack) * xf;
        // SF2 modulator:velocity → initialFilterFc(vel=127 不偏移,越低越暗)
        let cutoff =
            region.cutoff_hz.map(|fc| fc * vel2filter_mul(region.vel2filter_cents, vel));
        self.voices.push(Voice {
            region_idx,
            key,
            pos: region.offset as f64,
            step: ratio,
            phase: EnvPhase::Attack,
            env: if region.attack > 0.0 { 0.0 } else { 1.0 },
            hold_t: 0.0,
            release_level: 1.0,
            vel_gain: vel_g,
            lgain: lgain * vol,
            rgain: rgain * vol,
            atk: region.attack,
            hld: region.hold,
            dcy: region.decay,
            sus: region.sustain,
            rel: region.release,
            end_frame: end as f64,
            loop_range,
            loop_continuous: region.loop_mode == LoopMode::Continuous,
            sustain_loop: region.loop_mode == LoopMode::Sustain,
            loop_broken: false,
            one_shot: region.loop_mode == LoopMode::OneShot,
            release_rate: release_decay_rate(region.release, sample_rate),
            filter: cutoff.map(|fc| BiquadLp::new(fc, region.resonance_q, sample_rate)),
            lfo_phase: 0.0,
            lfo_depth: region.vib_depth_cents,
            held_by_pedal: false,
        });
        // 上限:窃取最旧 voice
        if self.voices.len() > self.max_voices {
            self.voices.drain(0..self.voices.len() - self.max_voices);
        }
    }

    /// 渲染一块(立体声交错输出,len = 帧数×2)。
    /// `base_frame` 是本块第一帧的全局帧号(事件调度用)。
    pub fn render(&mut self, out: &mut [f32], base_frame: u64, events: &[MidiEvent], ev_i: &mut usize) {
        let frames = out.len() / 2;
        for fi in 0..frames {
            let cur = base_frame + fi as u64;
            // 帧内事件(已排序)
            while *ev_i < events.len() && events[*ev_i].frame() <= cur {
                match events[*ev_i] {
                    MidiEvent::NoteOn { key, vel, .. } => self.note_on(key, vel),
                    MidiEvent::NoteOff { key, .. } => self.note_off(key),
                    MidiEvent::PitchBend { value, .. } => self.pitch_bend(value),
                    MidiEvent::ControlChange { cc, value, .. } => self.control_change(cc, value),
                }
                *ev_i += 1;
            }
            let write = fi * 2;
            self.render_frame(&mut out[write..write + 2]);
        }
        self.finish_block(out);
    }

    /// 渲染一块(实时入口,事件由调用方驱动 note_on/note_off)。
    /// 立体声交错输出,len = 帧数×2;输出**叠加**进 out(需预置零)。
    pub fn render_block(&mut self, out: &mut [f32]) {
        for fi in 0..out.len() / 2 {
            self.render_frame(&mut out[fi * 2..fi * 2 + 2]);
        }
        self.finish_block(out);
    }

    /// 渲染一帧(2 样本,叠加进 out[0..2])。
    fn render_frame(&mut self, out2: &mut [f32]) {
        let dt = 1.0 / self.sample_rate;
        // CC11 expression 一阶平滑(长音渐强/渐弱不跳变)
        self.expression += (self.expr_target - self.expression) * self.expr_coef;
        let expr = self.expression;
        let bend_cents = self.bend_cents;
        let lfo_scale = self.lfo_scale;
        let lfo_inc = 2.0 * std::f32::consts::PI * 5.5 / self.sample_rate;
        let mut vi = 0;
        while vi < self.voices.len() {
            // Done 的 voice 在块内直接跳过(清理要等 finish_block;
            // 否则采样中途结束的 voice 会把末样本当直流播放到块边界)
            if self.voices[vi].phase == EnvPhase::Done {
                vi += 1;
                continue;
            }
            // 借用拆分:先 clone Arc(便宜,引用计数),再独占借用 voice。
            let sample = self.samples[self.voices[vi].region_idx].clone();
            let v = &mut self.voices[vi];
            // 循环
            if let Some((ls, le)) = v.loop_range {
                let looping = v.loop_continuous || (v.sustain_loop && !v.loop_broken);
                if looping && v.pos >= le as f64 {
                    v.pos = ls as f64 + (v.pos - le as f64) % ((le - ls) as f64);
                }
            }
            // 滤波在包络之前(SF2 规范信号顺序:采样 → 滤波 → 放大包络 → 声像)
            let (mut l, mut r) = match v.filter.as_mut() {
                Some(f) => {
                    let lin = interp(&sample.left, v.pos);
                    let rin = sample.right.as_ref().map_or(0.0, |rr| interp(rr, v.pos));
                    f.process(sample.right.is_some(), lin, rin)
                }
                None => {
                    let l = interp(&sample.left, v.pos);
                    let r = sample.right.as_ref().map_or(l, |rr| interp(rr, v.pos));
                    (l, r)
                }
            };
            l *= v.env * v.vel_gain * expr;
            r *= v.env * v.vel_gain * expr;
            out2[0] += l * v.lgain;
            out2[1] += r * v.rgain;
            // 弯音 + CC1 vibrato:cents → 读取步进倍率(仍走 Hermite 插值)
            let depth = v.lfo_depth * lfo_scale;
            let cents = bend_cents + if depth > 0.0 { v.lfo_phase.sin() * depth } else { 0.0 };
            if depth > 0.0 {
                v.lfo_phase = (v.lfo_phase + lfo_inc) % (2.0 * std::f32::consts::PI);
            }
            if cents.abs() > 1e-6 {
                v.pos += v.step * 2f64.powf((cents / 1200.0) as f64);
            } else {
                v.pos += v.step;
            }
            // 自然结束 / 采样尾(循环已在上方处理,pos 到达 end 说明
            // 无循环 / 已走出 sustain 循环 / one_shot —— 一律结束)
            if v.pos >= v.end_frame {
                v.phase = EnvPhase::Done;
            } else {
                // 包络推进
                advance_env(v, dt);
            }
            vi += 1;
        }
    }

    /// 块收尾:清理结束 voice + 输出 clamp。
    fn finish_block(&mut self, out: &mut [f32]) {
        self.voices.retain(|v| v.phase != EnvPhase::Done);
        for s in out.iter_mut() {
            *s = s.clamp(-1.0, 1.0);
        }
    }

    /// 离线渲染:事件序列 → 立体声交错 f32。
    /// 返回长度 = 事件尾 + 尾音(自动 trim 静音)。
    pub fn render_offline(mut self, events: &mut [MidiEvent], extra_tail_secs: f32) -> Vec<f32> {
        events.sort_by_key(|e| e.frame());
        // 尾音上界:所有区域 release/decay 的最大值 + 用户余量(压缩到合理上限)
        let mut max_release = 0.5f32;
        for r in &self.regions {
            max_release = max_release.max(r.release.max(r.decay));
        }
        let tail = max_release.max(extra_tail_secs).min(30.0);
        let last = events.last().map(|e| e.frame()).unwrap_or(0);
        let total = (last as f64 + self.sample_rate as f64 * tail as f64) as usize;
        let mut out = vec![0.0f32; total * 2];
        let mut ev_i = 0usize;
        let block = 512usize;
        let mut base = 0usize;
        while base < total {
            let n = block.min(total - base);
            let slice = &mut out[base * 2..(base + n) * 2];
            self.render(slice, base as u64, events, &mut ev_i);
            base += n;
        }
        // 尾部 trim:从后往前找最后一个非静音帧
        let thresh = 1e-4f32;
        let mut last_nz = out.len() / 2;
        for fi in (0..out.len() / 2).rev() {
            if out[fi * 2].abs() > thresh || out[fi * 2 + 1].abs() > thresh {
                last_nz = fi + 1;
                break;
            }
        }
        let keep = (last_nz + (self.sample_rate as usize) / 20).min(out.len() / 2); // +50ms
        out.truncate(keep * 2);
        out
    }

    pub fn active_voices(&self) -> usize {
        self.voices.len()
    }
}

/// 包络推进(独立函数避免同时借用 self.voices 与 self.regions)。
#[inline]
fn advance_env(v: &mut Voice, dt: f32) {
    match v.phase {
        EnvPhase::Attack => {
            if v.atk <= 0.0 {
                v.env = 1.0;
                v.phase = EnvPhase::Hold;
            } else {
                v.env += dt / v.atk;
                if v.env >= 1.0 {
                    v.env = 1.0;
                    v.phase = EnvPhase::Hold;
                }
            }
        }
        EnvPhase::Hold => {
            if v.hld <= 0.0 || v.hold_t >= v.hld {
                v.phase = EnvPhase::Decay;
            } else {
                v.hold_t += dt;
            }
        }
        EnvPhase::Decay => {
            if v.dcy <= 0.0 || v.env <= v.sus {
                v.env = v.sus;
                v.phase = EnvPhase::Sustain;
            } else {
                v.env -= dt * (1.0 - v.sus) / v.dcy;
                if v.env <= v.sus {
                    v.env = v.sus;
                    v.phase = EnvPhase::Sustain;
                }
            }
        }
        EnvPhase::Sustain => {
            // 维持(note-off 转 Release)
        }
        EnvPhase::Release => {
            if v.rel <= 0.0 {
                v.env = 0.0;
                v.phase = EnvPhase::Done;
            } else {
                // 指数释放:rel 秒内衰减 100 dB(人耳对响度是对数感知,
                // 指数衰减听感平滑;线性释放会听到明显拐点)
                v.env *= v.release_rate;
                if v.env <= (v.release_level * 1e-5).max(1e-6) {
                    v.env = 0.0;
                    v.phase = EnvPhase::Done;
                }
            }
        }
        EnvPhase::Done => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_sample(rate: u32, hz: f32, secs: f32) -> Arc<SampleData> {
        let n = (rate as f32 * secs) as usize;
        let mut left = Vec::with_capacity(n);
        for i in 0..n {
            left.push((2.0 * std::f32::consts::PI * hz * i as f32 / rate as f32).sin() * 0.8);
        }
        Arc::new(SampleData {
            left,
            right: None,
            rate,
            loop_range: None,
            root_key: Some(60),
            fine_tune_cents: 0.0,
        })
    }

    fn instr(region: Region, sample: Arc<SampleData>) -> LoadedInstrument {
        LoadedInstrument {
            regions: vec![region],
            samples: vec![sample],
        }
    }

    #[test]
    fn renders_nonzero_for_matched_note() {
        let mut region = Region::default();
        region.sample_path = PathBuf::from("test.wav");
        region.release = 0.05;
        let synth = Synth::new(&instr(region, sine_sample(48000, 440.0, 0.5)), 48000.0);
        let mut events = vec![MidiEvent::NoteOn { frame: 0, key: 60, vel: 100 }];
        let out = synth.render_offline(&mut events, 1.0);
        assert!(out.iter().any(|s| s.abs() > 1e-4), "应有输出");
    }

    #[test]
    fn release_ends_note() {
        let mut region = Region::default();
        region.sample_path = PathBuf::from("test.wav");
        region.release = 0.02;
        let synth = Synth::new(&instr(region, sine_sample(48000, 440.0, 2.0)), 48000.0);
        let mut events = vec![
            MidiEvent::NoteOn { frame: 0, key: 60, vel: 100 },
            MidiEvent::NoteOff { frame: 48000, key: 60 }, // 1 秒后松键
        ];
        let out = synth.render_offline(&mut events, 1.0);
        let frames = out.len() / 2;
        // 松键 + release 0.02s + 50ms trim 余量后,应远短于采样全长(2s)
        assert!(frames < 48000 + 4800, "release 后应停止,实际 {frames} 帧");
        assert!(frames > 48000, "至少播到松键点,实际 {frames} 帧");
    }

    #[test]
    fn no_sound_when_no_region_matches() {
        let mut region = Region::default();
        region.sample_path = PathBuf::from("test.wav");
        region.key_lo = 40;
        region.key_hi = 50;
        let synth = Synth::new(&instr(region, sine_sample(48000, 440.0, 0.2)), 48000.0);
        let mut events = vec![MidiEvent::NoteOn { frame: 0, key: 90, vel: 100 }];
        let out = synth.render_offline(&mut events, 0.2);
        assert!(out.iter().all(|s| *s == 0.0));
    }

    #[test]
    fn loop_continuous_sustains() {
        let rate = 48000;
        // 0.1s 采样,循环前半段
        let n = (rate as f32 * 0.1) as usize;
        let mut left = Vec::with_capacity(n);
        for i in 0..n {
            left.push((2.0 * std::f32::consts::PI * 220.0 * i as f32 / rate as f32).sin() * 0.5);
        }
        let sample = Arc::new(SampleData {
            left,
            right: None,
            rate,
            loop_range: Some((0, n as u32 - 4)),
            root_key: Some(60),
            fine_tune_cents: 0.0,
        });
        let mut region = Region::default();
        region.sample_path = PathBuf::from("loop.wav");
        region.loop_mode = LoopMode::Continuous;
        region.loop_start = Some(0);
        region.loop_end = Some(n as u32 - 4);
        region.release = 0.01;
        let synth = Synth::new(&instr(region, sample), rate as f32);
        // 按住 1 秒(采样仅 0.1s):循环应让它持续出声
        let mut events = vec![
            MidiEvent::NoteOn { frame: 0, key: 60, vel: 100 },
            MidiEvent::NoteOff { frame: rate as u64, key: 60 },
        ];
        let out = synth.render_offline(&mut events, 0.2);
        // 检查 0.5s 处(远超采样长度)仍有声
        let probe = 24000 * 2;
        assert!(probe + 1 < out.len());
        assert!(
            out[probe].abs() > 1e-3 || out[probe + 1].abs() > 1e-3,
            "循环采样应在采样尾后仍出声"
        );
    }

    #[test]
    fn velocity_gain_monotonic() {
        assert!(vel_gain(20, 100.0) < vel_gain(100, 100.0));
        assert_eq!(vel_gain(127, 0.0), 1.0);
        assert_eq!(vel_gain(20, 0.0), 1.0);
    }

    #[test]
    fn pan_gains_symmetric() {
        let (l1, r1) = pan_gains(-0.5);
        let (l2, r2) = pan_gains(0.5);
        assert!((l1 - r2).abs() < 1e-6 && (r1 - l2).abs() < 1e-6);
        assert_eq!(pan_gains(0.0), (1.0, 1.0));
    }

    #[test]
    fn filter_conversions() {
        // SF2 规范:absolute cents 0 = 8.176 Hz,1200 = 一个八度
        assert!((sf2_fc_cents_to_hz(0.0) - 8.176).abs() < 1e-4);
        assert!((sf2_fc_cents_to_hz(1200.0) - 16.352).abs() < 1e-3);
        assert!((sf2_q_cb_to_linear(0.0) - 1.0).abs() < 1e-6);
        assert!((sf2_q_cb_to_linear(600.0) - 1000.0).abs() < 1e-2);
        // 释放系数:连乘 rel×sr 次后应衰减 -100 dB
        let rate = release_decay_rate(0.5, 48000.0);
        let db = 20.0 * rate.powf(0.5 * 48000.0).log10();
        assert!((db + 100.0).abs() < 0.5, "0.5s 释放应衰减 -100 dB,实际 {db} dB");
    }

    #[test]
    fn lowpass_attenuates_highs() {
        let render = |mut r: Region| {
            r.sample_path = PathBuf::from("t.wav");
            r.release = 0.05;
            let synth = Synth::new(&instr(r, sine_sample(48000, 8000.0, 0.3)), 48000.0);
            let mut events = vec![MidiEvent::NoteOn { frame: 0, key: 60, vel: 100 }];
            synth.render_offline(&mut events, 0.1)
        };
        let mut filtered = Region::default();
        filtered.cutoff_hz = Some(400.0);
        let rf = render(filtered);
        let rp = render(Region::default());
        let rms = |o: &[f32]| (o.iter().map(|s| s * s).sum::<f32>() / o.len().max(1) as f32).sqrt();
        // 8 kHz 正弦过 400 Hz 二阶低通:理论衰减约 -52 dB
        assert!(
            rms(&rf) < rms(&rp) * 0.05,
            "低通应大幅衰减 8 kHz:{} vs {}",
            rms(&rf),
            rms(&rp)
        );
    }

    #[test]
    fn lowpass_passes_lows() {
        let render = |mut r: Region| {
            r.sample_path = PathBuf::from("t.wav");
            r.release = 0.05;
            let synth = Synth::new(&instr(r, sine_sample(48000, 200.0, 0.3)), 48000.0);
            let mut events = vec![MidiEvent::NoteOn { frame: 0, key: 60, vel: 100 }];
            synth.render_offline(&mut events, 0.1)
        };
        let mut filtered = Region::default();
        filtered.cutoff_hz = Some(400.0);
        let rf = render(filtered);
        let rp = render(Region::default());
        let rms = |o: &[f32]| (o.iter().map(|s| s * s).sum::<f32>() / o.len().max(1) as f32).sqrt();
        // 截止频率以下的信号应基本无损
        assert!(
            rms(&rf) > rms(&rp) * 0.7,
            "低通不应衰减 200 Hz:{} vs {}",
            rms(&rf),
            rms(&rp)
        );
    }

    #[test]
    fn release_is_exponential() {
        let mut region = Region::default();
        region.sample_path = PathBuf::from("e.wav");
        region.release = 0.5;
        let synth = Synth::new(&instr(region, sine_sample(48000, 440.0, 2.0)), 48000.0);
        let off_frame = 48000u64;
        let mut events = vec![
            MidiEvent::NoteOn { frame: 0, key: 60, vel: 100 },
            MidiEvent::NoteOff { frame: off_frame, key: 60 },
        ];
        let out = synth.render_offline(&mut events, 0.2);
        let base = off_frame as usize;
        let rms_at = |t0: f32, t1: f32| -> f32 {
            let s = ((base as f32 + t0 * 48000.0) as usize).min(out.len() / 2) * 2;
            let e = ((base as f32 + t1 * 48000.0) as usize).min(out.len() / 2) * 2;
            let seg = &out[s..e.max(s)];
            (seg.iter().map(|x| x * x).sum::<f32>() / seg.len().max(1) as f32).sqrt()
        };
        // 释放 0.5s:前窗中心 0.05,后窗中心 0.3。
        // 指数(100 dB / rel)比值 ≈ 316;线性只有 ≈ 2.25
        let early = rms_at(0.025, 0.075);
        let late = rms_at(0.275, 0.325);
        assert!(
            early > late * 10.0,
            "释放应为指数曲线:early={early} late={late}"
        );
    }

    /// 上升过零间隔序列(频率/颤音估计用)。
    fn zc_intervals(out: &[f32], start: usize, end: usize) -> Vec<f32> {
        let mut iv = Vec::new();
        let mut last: Option<usize> = None;
        let hi = end.min(out.len() / 2);
        for fi in start..hi {
            if fi > 0 && out[(fi - 1) * 2] < 0.0 && out[fi * 2] >= 0.0 {
                if let Some(l) = last {
                    iv.push((fi - l) as f32);
                }
                last = Some(fi);
            }
        }
        iv
    }

    #[test]
    fn pitch_bend_changes_pitch() {
        let render = |bend: i16| {
            let mut r = Region::default();
            r.sample_path = PathBuf::from("t.wav");
            r.release = 0.05;
            let synth = Synth::new(&instr(r, sine_sample(48000, 440.0, 0.5)), 48000.0);
            let mut events = vec![
                MidiEvent::NoteOn { frame: 0, key: 60, vel: 100 },
                MidiEvent::PitchBend { frame: 0, value: bend },
            ];
            synth.render_offline(&mut events, 0.1)
        };
        let hz = |o: &[f32]| {
            let mut iv = zc_intervals(o, 4800, 16800);
            iv.sort_by(|a, b| a.partial_cmp(b).unwrap());
            48000.0 / iv[iv.len() / 2]
        };
        let hz0 = hz(&render(0));
        let hzb = hz(&render(8192));
        let ratio = hzb / hz0;
        assert!(
            (ratio - 2f32.powf(2.0 / 12.0)).abs() < 0.02,
            "满弯音应升 2 半音(×1.1225):ratio={ratio}"
        );
    }

    #[test]
    fn cc11_expression_scales_output() {
        let render = |expr: u8| {
            let mut r = Region::default();
            r.sample_path = PathBuf::from("t.wav");
            r.release = 0.05;
            let synth = Synth::new(&instr(r, sine_sample(48000, 440.0, 0.5)), 48000.0);
            let mut events = vec![
                MidiEvent::NoteOn { frame: 0, key: 60, vel: 100 },
                MidiEvent::ControlChange { frame: 0, cc: 11, value: expr },
            ];
            synth.render_offline(&mut events, 0.1)
        };
        let rms_win = |o: &[f32]| {
            let s = 9600 * 2;
            let e = (16800 * 2).min(o.len());
            let seg = &o[s..e];
            (seg.iter().map(|x| x * x).sum::<f32>() / seg.len().max(1) as f32).sqrt()
        };
        let full = rms_win(&render(127));
        let soft = rms_win(&render(32));
        let ratio = soft / full;
        assert!(
            (ratio - 32.0 / 127.0).abs() < 0.02,
            "CC11 应按 value/127 缩放音量:ratio={ratio}"
        );
    }

    #[test]
    fn cc64_sustain_holds_note() {
        let mut region = Region::default();
        region.sample_path = PathBuf::from("t.wav");
        region.release = 0.05;
        let synth = Synth::new(&instr(region, sine_sample(48000, 440.0, 1.0)), 48000.0);
        let mut events = vec![
            MidiEvent::NoteOn { frame: 0, key: 60, vel: 100 },
            MidiEvent::ControlChange { frame: 9600, cc: 64, value: 127 },
            MidiEvent::NoteOff { frame: 19200, key: 60 },
            MidiEvent::ControlChange { frame: 33600, cc: 64, value: 0 },
        ];
        let out = synth.render_offline(&mut events, 0.1);
        let rms_win = |t0: f32, t1: f32| {
            let s = ((t0 * 48000.0) as usize).min(out.len() / 2) * 2;
            let e = ((t1 * 48000.0) as usize).min(out.len() / 2) * 2;
            let seg = &out[s..e.max(s)];
            (seg.iter().map(|x| x * x).sum::<f32>() / seg.len().max(1) as f32).sqrt()
        };
        let held = rms_win(0.45, 0.6);
        let after = rms_win(0.77, 0.79);
        assert!(held > 0.05, "踩着踏板松键应继续发声:{held}");
        assert!(after < held * 0.01, "抬踏板后应释放:held={held} after={after}");
    }

    #[test]
    fn cc1_modulation_adds_vibrato() {
        let render = |modv: u8| {
            let mut r = Region::default();
            r.sample_path = PathBuf::from("t.wav");
            r.release = 0.05;
            let synth = Synth::new(&instr(r, sine_sample(48000, 440.0, 0.5)), 48000.0);
            let mut events = vec![
                MidiEvent::NoteOn { frame: 0, key: 60, vel: 100 },
                MidiEvent::ControlChange { frame: 0, cc: 1, value: modv },
            ];
            synth.render_offline(&mut events, 0.1)
        };
        let range = |o: &[f32]| {
            let iv = zc_intervals(o, 4800, 18400);
            let mn = iv.iter().cloned().fold(f32::MAX, f32::min);
            let mx = iv.iter().cloned().fold(f32::MIN, f32::max);
            mx - mn
        };
        let flat = range(&render(0));
        let vib = range(&render(127));
        assert!(
            vib > 4.0 && vib > flat * 2.0,
            "CC1 应加入颤音(过零间隔波动):flat={flat} vib={vib}"
        );
    }

    #[test]
    fn vel2filter_multiplier() {
        use super::vel2filter_mul;
        // 无 modulator:任何速度都不偏移
        assert_eq!(vel2filter_mul(0.0, 0), 1.0);
        assert_eq!(vel2filter_mul(0.0, 127), 1.0);
        // -1200 cents:vel=127 不偏移,vel=0 减半
        assert!((vel2filter_mul(-1200.0, 127) - 1.0).abs() < 1e-6);
        assert!((vel2filter_mul(-1200.0, 0) - 0.5).abs() < 1e-6);
        // +1200 cents:vel=0 加倍
        assert!((vel2filter_mul(1200.0, 0) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn xf_gain_curves() {
        use super::xf_gain;
        // hi <= lo = 未启用 → 恒 1
        assert_eq!(xf_gain(64, (0, 0), true, false), 1.0);
        assert_eq!(xf_gain(64, (127, 127), false, false), 1.0);
        // 线性 xfin:区间外钳制,区间内线性
        assert_eq!(xf_gain(60, (60, 72), true, false), 0.0);
        assert_eq!(xf_gain(72, (60, 72), true, false), 1.0);
        assert!((xf_gain(66, (60, 72), true, false) - 0.5).abs() < 1e-6);
        // xfout 反向:入口处 1,出口处 0
        assert_eq!(xf_gain(60, (60, 72), false, false), 1.0);
        assert_eq!(xf_gain(72, (60, 72), false, false), 0.0);
        assert!((xf_gain(66, (60, 72), false, false) - 0.5).abs() < 1e-6);
        // 越界钳制
        assert_eq!(xf_gain(10, (60, 72), true, false), 0.0);
        assert_eq!(xf_gain(120, (60, 72), true, false), 1.0);
        // power(等功率):中点 sin/cos(π/4)
        let mid = xf_gain(66, (60, 72), true, true);
        assert!((mid - std::f32::consts::FRAC_PI_4.sin()).abs() < 1e-6);
        let mid_out = xf_gain(66, (60, 72), false, true);
        assert!((mid_out - std::f32::consts::FRAC_PI_4.cos()).abs() < 1e-6);
    }
}
