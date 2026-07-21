//! S73 线2 — OpenUTAU ustx 音高渲染数学的忠实移植(「全量音高线导入=烤入 pitchDev」)。
//!
//! 真值 = stakira/OpenUtau master(2026-07 抓取):RenderPhrase.cs 构造(平台阶梯→vibrato
//! 【覆盖】→pitch points【加性 delta】→PITD 加性,5-tick 网格)/UNote.cs(PitchPoint x=ms
//! y=0.1半音;vibrato Evaluate;snap_first 运行时改写首点)/UCurve.cs Sample(线性+Round)/
//! MusicMath.cs InterpolateShape(io/i/o/l)/SplineInterpolate.cs(Catmull-Rom,仅
//! 点数>2 且段左端 shape=sp 且右端点非自动补点)。上游只借数学不抄代码。
//!
//! 我们的口径:单 tempo(无 tempo map,S56 定案);输出 = OU 最终曲线 − 我们的阶梯基线
//! (tone*100 + tuning¢ = 导入后 pitch+detune),即 pitchDev 语义(加性 cents);
//! 网格值经迭代 RDP(±1¢)压稀成折线 → 前端 evalCurveAt 线性插值还原。
//! 间隙(休止)沿 OU 平台语义取【后继音符】的平台 → dev 在下一音符 onset 处连续;
//! 休止帧在我们渲染里 voiced=false,曲线值只影响边界插值形状。
//!
//! ★已知边界(S73 审查记档,均为罕见/畸形外来文件,设计已认):
//! - OU Validate 的 duration=Max(10, duration) 钳位与 OverlapError→滤出渲染层不镜像
//!   (仅 <10 tick 畸形音符受影响,窗口 ≈10ms);
//! - 首音符之前的 pitd/负 x 前插滑音被网格起点(=首音符)截断——该区在我们渲染里
//!   无音符=unvoiced,只丢不可闻前导;
//! - 只动过点 x/shape 而 y 全零、无 vibrato/pitd 的「纯时机调教」会被判未调教跳过
//!   (part_is_untuned 以 y≠0 为调教痕迹;用户拍板口径)。

/// OU 音高点 shape。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OuShape {
    Io,
    I,
    O,
    L,
    Sp,
}

impl OuShape {
    pub fn parse(s: Option<&str>) -> OuShape {
        match s.unwrap_or("io") {
            "l" => OuShape::L,
            "i" => OuShape::I,
            "o" => OuShape::O,
            "sp" => OuShape::Sp,
            _ => OuShape::Io,
        }
    }
}

/// 一个 OU 音高点(文件原始域:x=相对音符起点 ms,y=0.1 半音)。
#[derive(Clone, Copy, Debug)]
pub struct OuPitchPoint {
    pub x_ms: f64,
    pub y_tenths: f64,
    pub shape: OuShape,
}

/// OU vibrato 原始字段(UNote.cs setter 的钳位在 `clamped()` 里镜像)。
#[derive(Clone, Copy, Debug, Default)]
pub struct OuVibrato {
    pub length: f64, // % of note(从尾部覆盖)
    pub period: f64, // ms
    pub depth: f64,  // cents
    pub fade_in: f64,
    pub fade_out: f64,
    pub shift: f64, // 周期相位 %
    pub drift: f64, // 整体偏移 %(×depth/100 cents)
}

impl OuVibrato {
    /// UNote.cs 属性 setter 钳位:length[0,100] period[5,500] depth[5,200] in/out[0,100]
    /// 且 in+out≤100。★互钳方向=【in 让位】(S73 审查):YamlDotNet 按文档键序调 setter,
    /// 规范键序 in 在 out 前 → out setter(收缩 _in)最后执行 → _out=clamp(out) 先定、
    /// _in=min(in, 100−_out)。OU 自产文件恒满足 in+out≤100,方向只影响外来手写文件。
    fn clamped(&self) -> OuVibrato {
        let fade_out = self.fade_out.clamp(0.0, 100.0);
        OuVibrato {
            length: self.length.clamp(0.0, 100.0),
            period: self.period.clamp(5.0, 500.0),
            depth: self.depth.clamp(5.0, 200.0),
            fade_in: self.fade_in.clamp(0.0, 100.0).min(100.0 - fade_out),
            fade_out,
            shift: self.shift.clamp(0.0, 100.0),
            drift: self.drift.clamp(-100.0, 100.0),
        }
    }
}

/// 一个音符(abs_tick = vp.position + note.position;已按 abs_tick 升序)。
#[derive(Clone, Debug)]
pub struct OuNote {
    pub abs_tick: i64,
    pub duration: i64,
    pub tone: i32,
    pub tuning_cents: f64,
    pub pitch_points: Vec<OuPitchPoint>,
    pub snap_first: bool,
    pub vibrato: Option<OuVibrato>,
}

impl OuNote {
    fn end(&self) -> i64 {
        self.abs_tick + self.duration
    }
    /// AdjustedTone(半音) = tone + tuning/100(UNote.cs L34)。
    fn adjusted_tone(&self) -> f64 {
        self.tone as f64 + self.tuning_cents / 100.0
    }
}

/// part 级 pitd 曲线(xs=part 相对 tick,ys=cents;稀疏控制点)。
#[derive(Clone, Debug, Default)]
pub struct OuPitd {
    pub xs: Vec<i64>,
    pub ys: Vec<i64>,
}

impl OuPitd {
    /// UCurve.Sample:命中点直取;区间内线性插值 + Math.Round(银行家舍入=C# 默认
    /// MidpointRounding.ToEven);范围外 → 默认 0。
    fn sample(&self, x: i64) -> i64 {
        if self.xs.is_empty() {
            return 0;
        }
        match self.xs.binary_search(&x) {
            Ok(i) => self.ys[i],
            Err(i) => {
                if i == 0 || i >= self.xs.len() {
                    0
                } else {
                    let (x0, x1) = (self.xs[i - 1] as f64, self.xs[i] as f64);
                    let (y0, y1) = (self.ys[i - 1] as f64, self.ys[i] as f64);
                    let v = y0 + (y1 - y0) * ((x as f64 - x0) / (x1 - x0));
                    round_half_even(v)
                }
            }
        }
    }
    pub fn is_silent(&self) -> bool {
        self.xs.is_empty() || self.ys.iter().all(|&y| y == 0)
    }
}

/// C# Math.Round 默认 = 银行家舍入(half-to-even);Rust round() 是 half-away-from-zero,不能直用。
/// round_ties_even = C# ToEven 精确同语义(S73 审查:手滚 EPSILON 窗版在 |v|<1 的 1-ulp 近半值上背离)。
fn round_half_even(v: f64) -> i64 {
    v.round_ties_even() as i64
}

const PITCH_INTERVAL: i64 = 5; // OU RenderPhrase pitchInterval
const RDP_EPS_CENTS: f64 = 1.0;
/// 与 TS vocalNotes.MAX_CURVE_POINTS 同值(超限时加粗 RDP 容差重压)。
const MAX_CURVE_POINTS: usize = 100_000;
/// 网格上限(≈33 分钟 part @120bpm):真实曲目远在其下;超限=Overflow(响亮告知,绝不静默当未调教)。
const MAX_GRID_POINTS: usize = 400_000;
/// RDP 段长强切阈值:超过即先中点锚定再细分——把最坏情况从 O(n²)(锯齿全保)钉到 O(n log n),
/// 代价=平直长段每 4096 网格点多留一个锚(可忽略;S73 审查:4M 网格病理 >14 分钟占死 import 单飞锁)。
const RDP_MAX_SEG: usize = 4096;

/// 烤制结果三态:Overflow 必须与「未调教」可区分——调教被静默丢弃=用户资产蒸发(S73 审查)。
pub enum BakeOutcome {
    Baked { xs: Vec<i64>, ys: Vec<i64> },
    /// 与基线无差(≈未调教或纯 tuning)。
    NoDiff,
    /// part 超出可烤网格上限,调教未导入(前端须提示)。
    Overflow,
}

/// MusicMath.InterpolateShape(ep=0.001:区间过窄直接 y1;sp 在此层退化为 io——真 Catmull-Rom
/// 特例由调用方判定后走 `catmull_rom`)。
fn interpolate_shape(x0: f64, x1: f64, y0: f64, y1: f64, x: f64, shape: OuShape) -> f64 {
    const EP: f64 = 0.001;
    if x1 - x0 < EP {
        return y1;
    }
    let s = (x - x0) / (x1 - x0);
    match shape {
        OuShape::L => y0 + (y1 - y0) * s,
        OuShape::I => y0 + (y1 - y0) * (1.0 - ((s * std::f64::consts::PI) / 2.0).cos()),
        OuShape::O => y0 + (y1 - y0) * ((s * std::f64::consts::PI) / 2.0).sin(),
        OuShape::Io | OuShape::Sp => y0 + (y1 - y0) * (1.0 - (s * std::f64::consts::PI).cos()) / 2.0,
    }
}

/// SplineInterpolate.cs Catmull-Rom(四点,端点邻近复制;x≤x0→y0,x≥x1→y1)。
#[allow(clippy::too_many_arguments)]
fn catmull_rom(x_1: f64, x0: f64, x1: f64, x2: f64, y_1: f64, y0: f64, y1: f64, y2: f64, x: f64) -> f64 {
    if x <= x0 {
        return y0;
    }
    if x >= x1 {
        return y1;
    }
    let m0 = (y1 - y_1) * (x1 - x0) / (x1 - x_1);
    let m1 = (y2 - y0) * (x1 - x0) / (x2 - x0);
    let a = 2.0 * y0 - 2.0 * y1 + m0 + m1;
    let b = -3.0 * y0 + 3.0 * y1 - 2.0 * m0 - m1;
    let t = (x - x0) / (x1 - x0);
    ((a * t + b) * t + m0) * t + y0
}

/// 换算件(单 tempo;MusicMath 简式,MsPosToTickPos 四舍五入到整 tick)。
struct TimeConv {
    ms_per_tick: f64,
    ticks_per_ms: f64,
}

impl TimeConv {
    fn new(bpm: f64) -> TimeConv {
        let bpm = if bpm.is_finite() && bpm > 0.0 { bpm } else { 120.0 };
        TimeConv {
            ms_per_tick: 60_000.0 / (bpm * 480.0),
            ticks_per_ms: bpm * 480.0 / 60_000.0,
        }
    }
    fn tick_to_ms(&self, tick: i64) -> f64 {
        tick as f64 * self.ms_per_tick
    }
    /// TimeAxis.MsPosToTickPos = round(ms * ticksPerMs)(C# Math.Round 银行家舍入)。
    fn ms_to_tick_round(&self, ms: f64) -> i64 {
        round_half_even(ms * self.ticks_per_ms)
    }
}

/// 展开后的绝对音高点(x=绝对 tick,y=绝对 cents)。
#[derive(Clone, Copy, Debug)]
struct AbsPoint {
    x: i64,
    y: f64,
    shape: OuShape,
    auto_completed: bool,
}

/// 一个音符的音高点展开:snap_first 改写 → ms→tick(round) + y→绝对 cents → 自动补点。
/// grid0 = 网格原点(绝对 tick;首音符的「前插」下界,OU pitchStart 对应物)。
fn expand_pitch_points(n: &OuNote, prev: Option<&OuNote>, is_first: bool, grid0: i64, tc: &TimeConv) -> Vec<AbsPoint> {
    let tone_cents = n.adjusted_tone() * 100.0;
    let mut pts: Vec<AbsPoint> = Vec::new();
    if n.pitch_points.is_empty() {
        // data 空 → 两个 io 零点(note.position / note.End),均 autoCompleted
        pts.push(AbsPoint { x: n.abs_tick, y: tone_cents, shape: OuShape::Io, auto_completed: true });
        pts.push(AbsPoint { x: n.end(), y: tone_cents, shape: OuShape::Io, auto_completed: true });
        return pts;
    }
    let node_ms = tc.tick_to_ms(n.abs_tick);
    for (i, p) in n.pitch_points.iter().enumerate() {
        // snap_first:首点 Y 运行时改写(UNote.Validate)——前音贴合 → (prev−this)*10,否则 0
        let y_tenths = if i == 0 && n.snap_first {
            match prev {
                Some(pv) if pv.end() == n.abs_tick => (pv.adjusted_tone() - n.adjusted_tone()) * 10.0,
                _ => 0.0,
            }
        } else {
            p.y_tenths
        };
        pts.push(AbsPoint {
            x: tc.ms_to_tick_round(node_ms + p.x_ms),
            y: y_tenths * 10.0 + tone_cents,
            shape: p.shape,
            auto_completed: false,
        });
    }
    // 自动补点(RenderPhrase L286-297):首点在下界之后 → 前插同 Y;末点在 End 前 → 后补同 Y
    let lower = if is_first { grid0 } else { n.abs_tick };
    if pts[0].x > lower {
        pts.insert(0, AbsPoint { x: lower, y: pts[0].y, shape: OuShape::Io, auto_completed: true });
    }
    if pts.last().unwrap().x < n.end() {
        let y = pts.last().unwrap().y;
        pts.push(AbsPoint { x: n.end(), y, shape: OuShape::Io, auto_completed: true });
    }
    pts
}

/// OU vibrato Evaluate(UNote.cs L354-370):返回音高偏移 cents(含渐入渐出与 drift),
/// nPos = 音符内归一化位置。不在振音区 → 0。
fn vibrato_cents(v: &OuVibrato, n_pos: f64, n_period: f64) -> f64 {
    let n_start = 1.0 - v.length / 100.0;
    if n_pos < n_start {
        return 0.0;
    }
    let n_in = v.length / 100.0 * v.fade_in / 100.0;
    let n_in_pos = n_start + n_in;
    let n_out = v.length / 100.0 * v.fade_out / 100.0;
    let n_out_pos = 1.0 - n_out;
    let t = (n_pos - n_start) / n_period + v.shift / 100.0;
    let mut y = (2.0 * std::f64::consts::PI * t).sin() * v.depth + v.depth / 100.0 * v.drift;
    if n_pos < n_in_pos && n_in > 0.0 {
        y *= (n_pos - n_start) / n_in;
    } else if n_pos > n_out_pos && n_out > 0.0 {
        y *= (1.0 - n_pos) / n_out;
    }
    y
}

/// 整 part 烤制:OU 最终曲线(5-tick 网格)− 我们的阶梯基线 → RDP 压稀。
/// 返回 (xs, ys):xs = segment 相对 tick(绝对 − base_abs,即首音符=0),ys = 整数 cents。
/// 全程 |dev|<0.5¢ → None(与基线无差,不烤)。
pub fn bake_pitch_dev(
    notes: &[OuNote], // 已按 abs_tick 升序
    pitd: Option<&OuPitd>,
    part_pos: i64,
    base_abs: i64,
    bpm: f64,
) -> BakeOutcome {
    if notes.is_empty() {
        return BakeOutcome::NoDiff;
    }
    let tc = TimeConv::new(bpm);
    let grid0 = base_abs;
    // 网格只到末音符尾 +1 格:末音符之后的 pitd 尾巴两边都不可闻(OU 无音可弯、我们 voiced=false),
    // 烤了=segment 时长外的不可见不可编辑数据(S73 审查),裁掉。
    let last_end = notes.last().map(|n| n.end()).unwrap_or(base_abs);
    let grid_end = last_end + PITCH_INTERVAL;
    let len = (((grid_end - grid0) / PITCH_INTERVAL) + 1).max(1) as usize;
    if len > MAX_GRID_POINTS {
        return BakeOutcome::Overflow;
    }
    let gx = |k: usize| grid0 + k as i64 * PITCH_INTERVAL;

    // ① 平台阶梯(= 我们的基线;间隙取后继音符,尾部延最后值 —— RenderPhrase L248-259)
    let mut base = vec![0.0f64; len];
    let mut idx = 0usize;
    for n in notes {
        let tone_cents = n.adjusted_tone() * 100.0;
        while idx < len && gx(idx) < n.end() {
            base[idx] = tone_cents;
            idx += 1;
        }
    }
    let last_val = notes.last().unwrap().adjusted_tone() * 100.0;
    while idx < len {
        base[idx] = last_val;
        idx += 1;
    }
    let mut cur = base.clone();

    // ② vibrato 覆盖写(L260-274;startIndex=ceil,endIndex=整除,均相对网格)
    for n in notes {
        let Some(v) = n.vibrato else { continue };
        let v = v.clamped();
        if !(n.vibrato.map(|raw| raw.length > 0.0).unwrap_or(false)) || n.duration <= 0 {
            continue; // length<=0 跳过(钳位前判定,L262 用原始 length)
        }
        let note_ms = n.duration as f64 * tc.ms_per_tick;
        let n_period = v.period / note_ms;
        let start_i = ((n.abs_tick - grid0) as f64 / PITCH_INTERVAL as f64).ceil().max(0.0) as usize;
        let end_i = (((n.end() - grid0) / PITCH_INTERVAL) as usize).min(len);
        for k in start_i..end_i {
            let n_pos = (gx(k) - n.abs_tick) as f64 / n.duration as f64;
            let y = vibrato_cents(&v, n_pos, n_period);
            cur[k] = n.adjusted_tone() * 100.0 + y;
        }
    }

    // ③ pitch points 加性 delta(L276-333;基线取「x 还在前音符区间内 → 前音平台」)
    for (ni, n) in notes.iter().enumerate() {
        let prev = if ni > 0 { Some(&notes[ni - 1]) } else { None };
        let pts = expand_pitch_points(n, prev, ni == 0, grid0, &tc);
        if pts.len() < 2 {
            continue;
        }
        let mut k = (((pts[0].x - grid0) as f64) / PITCH_INTERVAL as f64).floor().max(0.0) as usize;
        for w in 0..pts.len() - 1 {
            let (p0, p1) = (pts[w], pts[w + 1]);
            while k < len && gx(k) < p1.x {
                let x = gx(k) as f64;
                // ★Catmull-Rom 门用【文件原始点数】(RenderPhrase L307 data.Count>2)——自动补点
                //   进的是展开列表不进 data;用展开数会让「原始 2 点 sp」误走样条(S73 审查)
                let pitch = if n.pitch_points.len() > 2 && p0.shape == OuShape::Sp && !p1.auto_completed {
                    // 真 Catmull-Rom 特例(端点邻近复制)
                    let pm1 = if w > 0 { pts[w - 1] } else { p0 };
                    let pp2 = if w + 2 < pts.len() { pts[w + 2] } else { p1 };
                    catmull_rom(
                        pm1.x as f64, p0.x as f64, p1.x as f64, pp2.x as f64,
                        pm1.y, p0.y, p1.y, pp2.y, x,
                    )
                } else {
                    interpolate_shape(p0.x as f64, p1.x as f64, p0.y, p1.y, x, p0.shape)
                };
                let base_pitch = match prev {
                    Some(pv) if gx(k) < pv.end() => pv.adjusted_tone() * 100.0,
                    _ => n.adjusted_tone() * 100.0,
                };
                cur[k] += pitch - base_pitch;
                k += 1;
            }
        }
    }

    // ④ PITD 加性(L409-416;x = part 相对 tick)
    if let Some(c) = pitd {
        if !c.is_silent() {
            for k in 0..len {
                cur[k] += c.sample(gx(k) - part_pos) as f64;
            }
        }
    }

    // dev = OU 最终 − 我们的基线;全静 → 不烤
    let dev: Vec<f64> = cur.iter().zip(base.iter()).map(|(a, b)| a - b).collect();
    if dev.iter().all(|d| d.abs() < 0.5) {
        return BakeOutcome::NoDiff;
    }

    // RDP 压稀(迭代式,S73 chunker 溢栈同教训;超限逐倍加粗容差)
    let pts: Vec<(i64, f64)> = (0..len).map(|k| (gx(k) - base_abs, dev[k])).collect();
    let mut eps = RDP_EPS_CENTS;
    let mut kept = rdp_simplify(&pts, eps);
    while kept.len() > MAX_CURVE_POINTS {
        eps *= 2.0;
        kept = rdp_simplify(&pts, eps);
    }
    let xs: Vec<i64> = kept.iter().map(|&(x, _)| x).collect();
    let ys: Vec<i64> = kept.iter().map(|&(_, y)| y.round() as i64).collect();
    BakeOutcome::Baked { xs, ys }
}

/// 函数图折线的 Douglas-Peucker(x 单调 → 距离用「到弦的竖直偏差」,与前端 evalCurveAt 的
/// 线性插值误差同度量)。迭代 worklist 非递归。
fn rdp_simplify(pts: &[(i64, f64)], eps: f64) -> Vec<(i64, f64)> {
    let n = pts.len();
    if n <= 2 {
        return pts.to_vec();
    }
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;
    let mut work = vec![(0usize, n - 1)];
    while let Some((lo, hi)) = work.pop() {
        if hi <= lo + 1 {
            continue;
        }
        // 段长强切:先锚中点再细分 → 最坏 O(n log n)(锯齿曲线的逐点剥离病理钉死)
        if hi - lo > RDP_MAX_SEG {
            let mid = lo + (hi - lo) / 2;
            keep[mid] = true;
            work.push((lo, mid));
            work.push((mid, hi));
            continue;
        }
        let (x0, y0) = (pts[lo].0 as f64, pts[lo].1);
        let (x1, y1) = (pts[hi].0 as f64, pts[hi].1);
        let dx = x1 - x0;
        let mut best = lo;
        let mut best_d = 0.0f64;
        for i in (lo + 1)..hi {
            let t = if dx > 0.0 { (pts[i].0 as f64 - x0) / dx } else { 0.0 };
            let chord = y0 + (y1 - y0) * t;
            let d = (pts[i].1 - chord).abs();
            if d > best_d {
                best_d = d;
                best = i;
            }
        }
        if best_d > eps {
            keep[best] = true;
            work.push((lo, best));
            work.push((best, hi));
        }
    }
    (0..n).filter(|&i| keep[i]).map(|i| pts[i]).collect()
}

/// 「完全未调教」检测(用户拍板=智能跳过):无 pitd 信号、无可闻 vibrato、所有音高点 y==0
/// (OU 默认滑音形状=零 y 点 + snap_first 运行时锚定)。tuning 不参与判定(无损映射成 detune,
/// 保持旋钮可编辑)。⚠已知边界:只动过点 x/shape 而 y 全零的调教会被当默认跳过(罕见,记录在案)。
pub fn part_is_untuned(notes: &[OuNote], pitd: Option<&OuPitd>) -> bool {
    let pitd_silent = pitd.map_or(true, |c| c.is_silent());
    // vibrato 可闻性只看 length>0(与烤制步骤②同口径):OU setter 把 depth 钳到 ≥5,
    // 文件 depth:0 在 OU 里照样出 5¢ 颤音——按原始 depth 判静会把它误判未调教(S73 审查)。
    pitd_silent
        && notes.iter().all(|n| {
            n.vibrato.map_or(true, |v| !(v.length > 0.0))
                && n.pitch_points.iter().all(|p| p.y_tenths == 0.0)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain_note(abs_tick: i64, duration: i64, tone: i32) -> OuNote {
        OuNote {
            abs_tick,
            duration,
            tone,
            tuning_cents: 0.0,
            pitch_points: Vec::new(),
            snap_first: true,
            vibrato: None,
        }
    }

    fn baked(o: BakeOutcome) -> (Vec<i64>, Vec<i64>) {
        match o {
            BakeOutcome::Baked { xs, ys } => (xs, ys),
            BakeOutcome::NoDiff => panic!("expected Baked, got NoDiff"),
            BakeOutcome::Overflow => panic!("expected Baked, got Overflow"),
        }
    }

    /// 网格上重建 dev 值(折线线性插值 = 前端 evalCurveAt 同式)。
    fn eval_poly(xs: &[i64], ys: &[i64], x: i64) -> f64 {
        if xs.is_empty() {
            return 0.0;
        }
        if x <= xs[0] {
            return ys[0] as f64;
        }
        if x >= *xs.last().unwrap() {
            return *ys.last().unwrap() as f64;
        }
        let i = xs.partition_point(|&v| v <= x);
        let (x0, x1) = (xs[i - 1] as f64, xs[i] as f64);
        let (y0, y1) = (ys[i - 1] as f64, ys[i] as f64);
        y0 + (y1 - y0) * ((x as f64 - x0) / (x1 - x0))
    }

    #[test]
    fn untuned_default_part_is_skipped() {
        // OU 默认音符:零 y 两点 + snap_first;无 vibrato/pitd → 未调教
        let mut n1 = plain_note(0, 480, 60);
        n1.pitch_points = vec![
            OuPitchPoint { x_ms: -40.0, y_tenths: 0.0, shape: OuShape::Io },
            OuPitchPoint { x_ms: 40.0, y_tenths: 0.0, shape: OuShape::Io },
        ];
        let n2 = plain_note(480, 480, 64);
        assert!(part_is_untuned(&[n1.clone(), n2.clone()], None));
        // 任一点 y≠0 → 调教过
        n1.pitch_points[1].y_tenths = 3.0;
        assert!(!part_is_untuned(&[n1, n2], None));
    }

    #[test]
    fn pitd_only_bake_roundtrips_linearly() {
        // 纯 pitd(音符无点=自动补零点,无滑音贡献)→ dev 应≈pitd 线性插值
        let notes = vec![plain_note(0, 960, 60)];
        let pitd = OuPitd { xs: vec![100, 200, 300], ys: vec![0, 50, 0] };
        let (xs, ys) = baked(bake_pitch_dev(&notes, Some(&pitd), 0, 0, 120.0));
        // 峰值处 ±(网格 5t + RDP 1¢) 容差
        assert!((eval_poly(&xs, &ys, 200) - 50.0).abs() <= 2.0, "peak={}", eval_poly(&xs, &ys, 200));
        assert!(eval_poly(&xs, &ys, 600).abs() <= 1.5);
        // xs 严格递增
        assert!(xs.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn snap_first_portamento_crosses_boundary() {
        // 两贴合音符 C4→E4,默认零点 + snap_first + x=±40ms:边界处应有从 6000¢ 滑向 6400¢
        // 的过渡(即 dev 在 B 起点略负、在 A 尾部略正——绝对线连续)
        let mk = |abs: i64, tone: i32| {
            let mut n = plain_note(abs, 480, tone);
            n.pitch_points = vec![
                OuPitchPoint { x_ms: -40.0, y_tenths: 0.0, shape: OuShape::Io },
                OuPitchPoint { x_ms: 40.0, y_tenths: 0.0, shape: OuShape::Io },
            ];
            n
        };
        let notes = vec![mk(0, 60), mk(480, 64)];
        let (xs, ys) = baked(bake_pitch_dev(&notes, None, 0, 0, 120.0));
        // 绝对曲线 = base + dev:B 起点(t=480)处 OU 滑音应在中点附近 → dev(480)≈−200¢
        let dev_at_onset = eval_poly(&xs, &ys, 480);
        assert!(
            (-350.0..=-50.0).contains(&dev_at_onset),
            "boundary glide midpoint, dev={dev_at_onset}"
        );
        // A 中段(远离边界)应平(|dev|<2¢)
        assert!(eval_poly(&xs, &ys, 240).abs() < 2.0);
        // B 中段回到本音(40ms@120bpm=32t,240 远在滑音后)
        assert!(eval_poly(&xs, &ys, 720).abs() < 2.0);
    }

    #[test]
    fn vibrato_overwrites_platform_and_fades() {
        // 单音符全程 vibrato:中段应有 ±depth 摆动;渐入起点为 0
        let mut n = plain_note(0, 1920, 60); // 4 拍@120bpm = 2s
        n.vibrato = Some(OuVibrato {
            length: 100.0,
            period: 250.0,
            depth: 50.0,
            fade_in: 10.0,
            fade_out: 10.0,
            shift: 0.0,
            drift: 0.0,
        });
        let (xs, ys) = baked(bake_pitch_dev(&[n], None, 0, 0, 120.0));
        let mut max_dev = 0.0f64;
        for t in (0..1920).step_by(5) {
            max_dev = max_dev.max(eval_poly(&xs, &ys, t).abs());
        }
        assert!((45.0..=55.0).contains(&max_dev), "vibrato depth≈50, got {max_dev}");
        // 起点渐入:头一个网格值应远小于满深
        assert!(eval_poly(&xs, &ys, 5).abs() < 20.0);
    }

    #[test]
    fn tuning_maps_to_base_not_dev() {
        // tuning=+30¢:AdjustedTone 进平台=我们的 detune 基线 → dev 恒 0 → 不烤
        let mut n = plain_note(0, 480, 60);
        n.tuning_cents = 30.0;
        assert!(matches!(bake_pitch_dev(&[n], None, 0, 0, 120.0), BakeOutcome::NoDiff));
    }

    #[test]
    fn rdp_reconstruction_error_bounded() {
        // 长正弦 dev 经 RDP 后逐网格重建误差 ≤ eps+0.5(整数化)
        let notes = vec![plain_note(0, 19_200, 60)];
        let pitd = OuPitd {
            xs: (0..3840).map(|i| i * 5).collect(),
            ys: (0..3840)
                .map(|i| ((i as f64 * 0.05).sin() * 80.0).round() as i64)
                .collect(),
        };
        let (xs, ys) = baked(bake_pitch_dev(&notes, Some(&pitd), 0, 0, 120.0));
        assert!(xs.len() < 3840, "RDP 应显著压稀,得 {}", xs.len());
        for t in (0..19_200).step_by(35) {
            let want = pitd.sample(t) as f64;
            let got = eval_poly(&xs, &ys, t);
            assert!((got - want).abs() <= RDP_EPS_CENTS + 1.5, "t={t} want={want} got={got}");
        }
    }

    #[test]
    fn pitd_sample_matches_ucurve_semantics() {
        let c = OuPitd { xs: vec![10, 20], ys: vec![0, 10] };
        assert_eq!(c.sample(10), 0);
        assert_eq!(c.sample(20), 10);
        assert_eq!(c.sample(15), 5);
        assert_eq!(c.sample(5), 0); // 范围外 → 默认 0
        assert_eq!(c.sample(25), 0);
        // 银行家舍入:12.5 → 12(C# Math.Round)
        let c2 = OuPitd { xs: vec![0, 2], ys: vec![12, 13] };
        assert_eq!(c2.sample(1), 12);
    }

    #[test]
    fn vibrato_fade_clamp_direction_is_in_yields() {
        // S73 审查钉死:in=80/out=50 → OU 载入(out setter 最后执行)得 (in=50, out=50)
        let v = OuVibrato { length: 50.0, period: 200.0, depth: 50.0, fade_in: 80.0, fade_out: 50.0, shift: 0.0, drift: 0.0 };
        let c = v.clamped();
        assert_eq!((c.fade_in, c.fade_out), (50.0, 50.0));
    }

    #[test]
    fn two_point_sp_uses_sine_not_spline() {
        // S73 审查:Catmull 门=文件原始点数(data.Count>2)。原始 2 点(首点 sp)+末端自动补点
        // 展开成 3 点,但仍必须走 InterpolateShape(sp→io);io 中点值 = (y0+y1)/2。
        let mut n = plain_note(0, 480, 60);
        n.snap_first = false;
        n.pitch_points = vec![
            OuPitchPoint { x_ms: 0.0, y_tenths: -20.0, shape: OuShape::Sp }, // −200¢ 起
            OuPitchPoint { x_ms: 200.0, y_tenths: 0.0, shape: OuShape::Io },
        ];
        let (xs, ys) = baked(bake_pitch_dev(&[n], None, 0, 0, 120.0));
        // 200ms@120bpm = 192t(1ms=0.96t);io 中点(96t)= −100¢ 精确(Catmull 会偏离)
        let mid = eval_poly(&xs, &ys, 96);
        assert!((mid - (-100.0)).abs() <= 2.0, "sp-2点须走 sineInOut,中点={mid}");
    }

    #[test]
    fn zero_depth_vibrato_still_counts_as_tuned() {
        // 文件 depth:0 + length>0:OU setter 钳到 5¢ 实际出声 → 必须算调教痕迹(烤出 5¢ 颤音)
        let mut n = plain_note(0, 1920, 60);
        n.vibrato = Some(OuVibrato { length: 100.0, period: 250.0, depth: 0.0, fade_in: 0.0, fade_out: 0.0, shift: 0.0, drift: 0.0 });
        assert!(!part_is_untuned(&[n.clone()], None));
        let (xs, ys) = baked(bake_pitch_dev(&[n], None, 0, 0, 120.0));
        let max_dev = (0..1920).step_by(5).map(|t| eval_poly(&xs, &ys, t).abs()).fold(0.0f64, f64::max);
        assert!((4.0..=6.0).contains(&max_dev), "钳后 5¢ 颤音,得 {max_dev}");
    }

    #[test]
    fn overflow_is_distinguishable_from_untuned() {
        // 超网格上限(>400k 格 = 2M ticks)→ Overflow 而非 NoDiff(调教绝不静默蒸发)
        let n0 = {
            let mut n = plain_note(0, 2_100_000, 60);
            n.pitch_points = vec![
                OuPitchPoint { x_ms: 0.0, y_tenths: -20.0, shape: OuShape::Io },
                OuPitchPoint { x_ms: 100.0, y_tenths: 0.0, shape: OuShape::Io },
            ];
            n
        };
        assert!(matches!(bake_pitch_dev(&[n0], None, 0, 0, 120.0), BakeOutcome::Overflow));
    }
}
