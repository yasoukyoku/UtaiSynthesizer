//! S163 —— **谐波能量占比**:「这段音频到底有没有在【目标基频】上发声」的一把尺子。
//!
//! ## 为什么需要它(不是又一把指标)
//! 音域扩展的落点选法在 S162 之前只有**一根轴:`|rel|`(电平)**。而电平这根轴有两个
//! 结构性盲点,两个都在 S163 当场量到了:
//! * ⛔ **它需要一个参照**,而参照 = 「邻近**没被救**的唱音」—— 密集救援的乐句里
//!   一个都没有(实测 akiko × 炉心 +7:68 个带候选的窗里 **24 个**取不到参照)。
//! * ⛔ **它分不开「唱得响」与「唱得对」**。实测 5 处「み」(目标 MIDI 87):一个
//!   **根本没救到**的候选(落点仍在死区,与关掉扩展的 `base` 几乎逐格相同)在整窗电平上
//!   反而**更贴近**邻居 ⇒ 它赢了,救援被整个丢掉。
//!
//! 这把尺子**不需要参照**,而且**与电平无关**:它问的是「在 300 Hz-8 kHz 这一带里,
//! 有多少能量真的落在 `f0` 的各次谐波上」。没救到的那一版,它的实际音高**根本不是**
//! 目标音高 ⇒ 目标谐波位置上是**谷** ⇒ 读数掉到 −13…−19.5 dB;救到的那一版是 −0.5…−2.7。
//! 而**健康的一对候选**之间,逐音差的 p99 只有 **1.83 dB**(n=240)。
//! ⇒ 3 dB 的门槛把 5/5 灾难全抓住,而 240 个健康音里只碰 1 个(那 1 个也确实该改)。
//!
//! ## ⛔ 三条自己给自己下的硬约束(每一条都有判据)
//! ⒜ **必须与电平无关** —— S162 的 `subf0` 护栏就是栽在这:它是绝对带能量差,于是
//!    「整体变安静」被读成「面状变好了」。这里返回的是**比值**,判据里把信号乘 0.1 必须逐位不变。
//! ⒝ **必须对「音高不对」敏感** —— 判据里拿同一个谐波堆、把 `f0` 换成 1.5 倍去问,读数必须塌。
//! ⒞ **不许用它跨音区比绝对值** —— 谐波序号尺子跨 f0 不可比(S162 栽过三次)。
//!    它的**唯一**用法是:**同一个音、同一个目标 f0**,比两个候选。调用方必须保证这一点。

use rustfft::{num_complex::Complex, FftPlanner};

/// 分析帧长(样本)。⛔ 写成**样本数**而不是毫秒:两个候选的读数必须逐位可比,
/// 而它们本来就同一条采样率;固定样本数让 FFT 尺寸与窗函数完全相同。
/// 4096 @ 40-48 kHz = 85-102 ms,频率分辨率 9.8-11.7 Hz。
const FRAME: usize = 4096;
/// 帧移 = 半帧。
const HOP: usize = FRAME / 2;
/// 统计带的下沿(Hz)。⭐ 故意**低于**大多数被救音的 f0:基频以下那一片正是
/// 「面状伪影 / 没在唱目标音高」会去的地方,把它算进分母才有区分度。
const BAND_LO: f32 = 300.0;
/// 统计带的上沿(Hz),再被 0.45·sr 夹一次。
const BAND_HI: f32 = 8000.0;
/// 谐波邻域 = ±`TOL_REL`·f0,但至少 ±`TOL_MIN_BINS` 个 bin(否则低音上会切掉主瓣)。
const TOL_REL: f32 = 0.10;
const TOL_MIN_BINS: f32 = 3.0;
/// 比最响的一帧低这么多以上的帧不参与中位数(它们是音头/音尾的静音,信噪比不够)。
const FRAME_FLOOR_DB: f32 = 40.0;

/// 谐波能量占比(dB,≤0):`f0` 各次谐波邻域内的能量 / `[300, 8000)` Hz 的总能量。
///
/// 逐帧算、取**中位数**(⛔ 不是把功率谱先平均:滑音会把平均谱的谐波抹平,
/// 而逐帧的谐波是清楚的)。
///
/// 返回 `None`:样本不够一帧 · `f0_hz` 不在合理范围 · 带内没有能量。
pub fn harmonic_energy_fraction_db(x: &[f32], sample_rate: u32, f0_hz: f32) -> Option<f32> {
    let sr = sample_rate as f32;
    if !f0_hz.is_finite() || f0_hz <= 40.0 || f0_hz >= sr * 0.25 || x.len() < FRAME || sr <= 0.0 {
        return None;
    }
    let bin = sr / FRAME as f32;
    let hi = BAND_HI.min(sr * 0.45);
    if hi <= BAND_LO {
        return None;
    }
    let lo_k = (BAND_LO / bin).ceil() as usize;
    let hi_k = ((hi / bin).floor() as usize).min(FRAME / 2);
    if hi_k <= lo_k {
        return None;
    }
    // 谐波掩码 —— 与帧无关,只算一次。
    let tol = (TOL_REL * f0_hz).max(TOL_MIN_BINS * bin);
    let mut mask = vec![false; hi_k + 1];
    let mut k = 1usize;
    loop {
        let c = f0_hz * k as f32;
        if c - tol > hi {
            break;
        }
        let a = ((c - tol) / bin).ceil().max(lo_k as f32) as usize;
        let b = ((c + tol) / bin).floor().min(hi_k as f32) as usize;
        for m in mask.iter_mut().take(b + 1).skip(a) {
            *m = true;
        }
        k += 1;
        if k > 512 {
            break;
        }
    }
    if !mask[lo_k..=hi_k].iter().any(|&v| v) {
        return None;
    }
    let win: Vec<f32> = (0..FRAME)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / FRAME as f32).cos())
        .collect();
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FRAME);
    let mut buf = vec![Complex::new(0.0f32, 0.0); FRAME];
    let mut rows: Vec<(f32, f32)> = Vec::new(); // (带内总能量, 谐波占比 dB)
    let mut off = 0usize;
    while off + FRAME <= x.len() {
        for (i, b) in buf.iter_mut().enumerate() {
            *b = Complex::new(x[off + i] * win[i], 0.0);
        }
        fft.process(&mut buf);
        let mut tot = 0.0f64;
        let mut har = 0.0f64;
        for j in lo_k..=hi_k {
            let p = f64::from(buf[j].re) * f64::from(buf[j].re)
                + f64::from(buf[j].im) * f64::from(buf[j].im);
            tot += p;
            if mask[j] {
                har += p;
            }
        }
        if tot > 0.0 {
            rows.push((tot as f32, (10.0 * (har / tot).max(1e-12).log10()) as f32));
        }
        off += HOP;
    }
    if rows.is_empty() {
        return None;
    }
    let peak = rows.iter().fold(0.0f32, |a, r| a.max(r.0));
    let floor = peak * 10f32.powf(-FRAME_FLOOR_DB / 10.0);
    let mut v: Vec<f32> = rows.iter().filter(|r| r.0 >= floor).map(|r| r.1).collect();
    if v.is_empty() {
        v = rows.iter().map(|r| r.1).collect();
    }
    v.sort_by(f32::total_cmp);
    Some(v[v.len() / 2])
}


/// 谐波**梳深**(dB):`[COMB_LO, COMB_HI)` 里,谐波峰的中位 − 谐波**中点**的中位。
///
/// 「谐波之间有没有起雾」——大 = 谐波清晰,小 = 谐波间被噪声填满。
/// ⛔ 它与 [`harmonic_energy_fraction_db`] **测的不是一件事**,而且在真实缺陷上方向相反:
/// 实测 yuyuko 的「卡痰」那个音梳深 **−0.4**(谐波间填满)而谐波占比 −0.10(看起来没问题);
/// 而 akiko 的「ぴゃ」梳深 **40.6**(谐波很清晰)却 H2−H1 = −44.8(只剩一根)。
/// ⇒ **两根轴都要有。**
/// ⭐⭐⭐ S163 —— **谐波谱峰的宽度**（相对 f0 的 %，越大 = 谐波越糊）。
///
/// ## 它量的是另一件事
/// [`harmonic_energy_fraction_db`] 量「能量在不在谐波上」，
/// [`comb_depth_db`] 量「谐波与谐波之间差多少」，
/// 而这一根量的是**每一根谐波自己糊不糊** —— 频率抖动 / 相位噪声会把谱峰展宽，
/// 而前两根轴对它都是瞎的（实测：4:36 接缝两侧峰宽 12.33 vs 0.99（**差 12 倍**），
/// 而填充度读到 −18 vs −29——**方向还反了**）。
///
/// ## 为什么它值钱
/// yuyuko 短音的 `donor_pre` 峰宽按落点：**77 → 3.50 而 78 → 11.88**（差 **3.4 倍**），
/// 而两者只差 **1 个半音** ⇒ 正好在落点候选范围内。
/// ⛔ 而 sidecar 的 `low_ratio` 在同两格上是 **77 → 0.616（最差）/ 79 → 0.129（好）**
/// —— **与峰宽完全相反**，而 S157 那条排序用的就是 `low_ratio`。
///
/// ## 口径
/// 长窗（8192）保证频率分辨率远高于谐波间隔，才量得到**峰宽**而不是窗宽；
/// 取 H3..H8 的 −6 dB 宽度中位，归一化到 f0。
/// ⚠ 它需要 ≥ 4096 个样本（≈ 100 ms @40k）；更短返回 `None`。
/// ⛔ **只在同一个音的两个候选之间比**（同 f0）—— 跨音高时谐波密度不同，读数不可比。
pub fn harmonic_peak_width_pct(x: &[f32], sample_rate: u32, f0_hz: f32) -> Option<f32> {
    const KLO: usize = 3;
    const KHI: usize = 8;
    const HALF: f64 = 0.25; // −6 dB
    if !(f0_hz.is_finite() && f0_hz > 20.0) {
        return None;
    }
    let n = if x.len() >= 8192 {
        8192usize
    } else if x.len() >= 4096 {
        4096
    } else {
        return None;
    };
    let win: Vec<f64> = (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos())
        .collect();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let mut acc = vec![0.0f64; n / 2 + 1];
    let mut frames = 0usize;
    let mut pos = 0usize;
    while pos + n <= x.len() {
        let mut buf: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new(f64::from(x[pos + i]) * win[i], 0.0))
            .collect();
        fft.process(&mut buf);
        for (k, a) in acc.iter_mut().enumerate() {
            *a += buf[k].norm_sqr();
        }
        frames += 1;
        pos += n / 2;
    }
    if frames == 0 {
        return None;
    }
    let df = f64::from(sample_rate) / n as f64;
    let f0 = f64::from(f0_hz);
    let nyq = f64::from(sample_rate) * 0.45;
    let mut ws: Vec<f64> = Vec::new();
    for k in KLO..=KHI {
        let c = f0 * k as f64;
        if c > nyq {
            break;
        }
        let lo = ((c - f0 * 0.4) / df).floor().max(0.0) as usize;
        let hi = (((c + f0 * 0.4) / df).ceil() as usize).min(acc.len() - 1);
        if hi <= lo + 6 {
            continue;
        }
        let peak = acc[lo..=hi].iter().copied().fold(0.0f64, f64::max);
        if peak <= 0.0 {
            continue;
        }
        let wide = acc[lo..=hi].iter().filter(|v| **v >= peak * HALF).count();
        ws.push(wide as f64 * df / f0 * 100.0);
    }
    if ws.len() < 3 {
        return None;
    }
    ws.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(ws[ws.len() / 2] as f32)
}

pub fn comb_depth_db(x: &[f32], sample_rate: u32, f0_hz: f32) -> Option<f32> {
    const COMB_LO: f32 = 2000.0;
    const COMB_HI: f32 = 8000.0;
    let sr = sample_rate as f32;
    if !f0_hz.is_finite() || f0_hz <= 40.0 || f0_hz >= sr * 0.25 || x.len() < FRAME {
        return None;
    }
    let hi = COMB_HI.min(sr * 0.45);
    if hi <= COMB_LO + f0_hz {
        return None;
    }
    let win: Vec<f32> = (0..FRAME)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / FRAME as f32).cos())
        .collect();
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FRAME);
    let mut buf = vec![Complex::new(0.0f32, 0.0); FRAME];
    let bin = sr / FRAME as f32;
    let tol = 0.12 * f0_hz;
    let mut vals: Vec<f32> = Vec::new();
    let mut off = 0usize;
    while off + FRAME <= x.len() {
        for (i, b) in buf.iter_mut().enumerate() {
            *b = Complex::new(x[off + i] * win[i], 0.0);
        }
        fft.process(&mut buf);
        let p = |j: usize| -> f64 {
            f64::from(buf[j].re) * f64::from(buf[j].re) + f64::from(buf[j].im) * f64::from(buf[j].im)
        };
        let mut peaks: Vec<f64> = Vec::new();
        let mut valleys: Vec<f64> = Vec::new();
        let mut k = (COMB_LO / f0_hz).ceil() as usize;
        while (k as f32) * f0_hz < hi {
            for (c, out) in [((k as f32) * f0_hz, true), (((k as f32) + 0.5) * f0_hz, false)] {
                let a = (((c - tol) / bin).ceil().max(0.0)) as usize;
                let b = (((c + tol) / bin).floor()).min((FRAME / 2) as f32) as usize;
                if b <= a {
                    continue;
                }
                if out {
                    peaks.push((a..=b).map(&p).fold(0.0f64, f64::max));
                } else {
                    let mut v: Vec<f64> = (a..=b).map(&p).collect();
                    v.sort_by(f64::total_cmp);
                    valleys.push(v[v.len() / 2]);
                }
            }
            k += 1;
        }
        if peaks.len() >= 3 && valleys.len() >= 3 {
            peaks.sort_by(f64::total_cmp);
            valleys.sort_by(f64::total_cmp);
            let (pm, vm) = (peaks[peaks.len() / 2], valleys[valleys.len() / 2]);
            vals.push((10.0 * ((pm + 1e-30) / (vm + 1e-30)).log10()) as f32);
        }
        off += HOP;
    }
    if vals.is_empty() {
        return None;
    }
    vals.sort_by(f32::total_cmp);
    Some(vals[vals.len() / 2])
}

/// ⭐⭐⭐⭐ S163 —— **上方谐波在音【内】被抽干多少**(dB;负 = 中段比音头弱)。
///
/// 量 `2·f0 .. 4·f0` 这一带(H2-H4)的能量随时间怎么走:中段中位 − 音头中位。
///
/// # 它治什么(用户 2026-08-27 亲口把两件事合成一件)
/// 「ぴゃ那里……**也是中间塌缩了**」「**那个音的中间电平弱就是差在了上方谐波上**」。
/// 分频带实测 akiko `[685]ぴゃ`(MIDI 90):落点 78 时 H2-H3 在音内塌 **22.0 dB**、
/// 落点 76 时 **+0.7**,而 `base` 自己只塌 5.4 ⇒ **救援把它放大了 4 倍**。
///
/// # ⛔ 为什么不是别的量
/// * **整音 RMS / 谐波占比**:量整个音的标量,音**内**的形状结构上看不见;
/// * **`comb_depth_db`**:量谐波之间有多空,不看谐波自己随时间怎么变;
/// * **`harmonic_peak_width_pct`**:量每根谐波糊不糊,同样没有时间轴;
/// * **全带电平的音内 Δ**:被清辅音起头的音污染(か/が 这类读到 40-75 dB 全是辅音动态)。
///   ⇒ 只看 `2..4·f0` 一带,并且掐掉首帧与尾部 15%(起音瞬态 / 释放曲线)。
///
/// # 口径
/// * STFT 2048 点、40 ms 跳距;不足 8 帧 ⇒ `None`(短音上这个量是噪声);
/// * 音头 = 前 1/5 的中位(跳过第 0 帧,窗淡入落在那里);中段 = 25%-75% 的中位;
/// * **与绝对电平无关**(带内自比)、**不需要参照**。
///
/// # ⚠ 只在同一个音的多个候选之间比
/// donor 逆变换后回到原音高 ⇒ 各候选 f0 相同 ⇒ 合法。**跨音高比一律无效。**
pub fn upper_harmonic_sag_db(x: &[f32], sample_rate: u32, f0_hz: f32) -> Option<f32> {
    const N: usize = 2048;
    if !(f0_hz > 20.0) || sample_rate == 0 {
        return None;
    }
    let hop = (sample_rate as usize / 25).max(1); // 40 ms
    if x.len() < N + hop {
        return None;
    }
    let (lo, hi) = (2.0 * f64::from(f0_hz), 4.0 * f64::from(f0_hz));
    let bin = f64::from(sample_rate) / N as f64;
    let (k0, k1) = ((lo / bin).floor() as usize, (hi / bin).ceil() as usize);
    let k1 = k1.min(N / 2);
    if k1 <= k0 {
        return None;
    }
    let win: Vec<f64> = (0..N)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N as f64).cos())
        .collect();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(N);
    let mut row: Vec<f32> = Vec::new();
    let mut i = 0usize;
    while i + N <= x.len() {
        let mut buf: Vec<Complex<f64>> =
            (0..N).map(|j| Complex::new(f64::from(x[i + j]) * win[j], 0.0)).collect();
        fft.process(&mut buf);
        let e: f64 = (k0..k1).map(|k| buf[k].norm_sqr()).sum();
        row.push((10.0 * (e + 1e-30).log10()) as f32);
        i += hop;
    }
    if row.len() < 8 {
        return None;
    }
    // ⛔ 掐掉第 0 帧(窗淡入)与尾部 15%(释放曲线是设计行为,不是缺陷)
    let end = ((row.len() as f64) * 0.85) as usize;
    let row = &row[1..end.max(3).min(row.len())];
    if row.len() < 6 {
        return None;
    }
    let med = |v: &[f32]| -> f32 {
        let mut w = v.to_vec();
        w.sort_by(f32::total_cmp);
        w[w.len() / 2]
    };
    let head = med(&row[..(row.len() / 5).max(2)]);
    let mid = med(&row[row.len() / 4..(3 * row.len()) / 4]);
    Some(mid - head)
}

/// ⭐⭐⭐ S163 —— **上方谐波的绝对强度**:`2..8·f0` 的能量 − `0.7..1.6·f0`(基频区)的能量,dB。
///
/// # 它是 [`upper_harmonic_sag_db`] 的**对手轴**
/// `upper_harmonic_sag_db` 量上方谐波在音内**稳不稳**,这一根量**强不强**。
/// 只看前者会换来一个**平但闷**的档:实测 akiko `[687]く` 换档后音内塌陷只改善 **0.98 dB**
/// 却把上方谐波压掉 **6.25 dB**;而 S163 §11 早就记过同一个方向
/// (akiko ぴゃ 的落点 76 上方谐波比 77 弱 **6.6 dB**)。
///
/// ⇒ ⛔ **凡是用 `upper_harmonic_sag_db` 做决策的地方,都必须同时查这一根。**
///
/// # 口径
/// 整段一次(4096 点、半重叠、累加谱),**相对基频区**而不是绝对值 ⇒ 与电平无关。
pub fn upper_harmonic_level_db(x: &[f32], sample_rate: u32, f0_hz: f32) -> Option<f32> {
    const N: usize = 4096;
    if !(f0_hz > 20.0) || sample_rate == 0 || x.len() < N {
        return None;
    }
    let bin = f64::from(sample_rate) / N as f64;
    let k = |hz: f64| -> usize { ((hz / bin).round() as usize).min(N / 2) };
    let (lo0, lo1) = (k(0.7 * f64::from(f0_hz)), k(1.6 * f64::from(f0_hz)));
    let (hi0, hi1) = (k(2.0 * f64::from(f0_hz)), k(8.0 * f64::from(f0_hz)));
    if lo1 <= lo0 || hi1 <= hi0 {
        return None;
    }
    let win: Vec<f64> = (0..N)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N as f64).cos())
        .collect();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(N);
    let mut acc = vec![0f64; N / 2 + 1];
    let mut i = 0usize;
    let mut frames = 0usize;
    while i + N <= x.len() {
        let mut buf: Vec<Complex<f64>> =
            (0..N).map(|j| Complex::new(f64::from(x[i + j]) * win[j], 0.0)).collect();
        fft.process(&mut buf);
        for (a, b) in acc.iter_mut().zip(buf.iter()) {
            *a += b.norm_sqr();
        }
        i += N / 2;
        frames += 1;
    }
    if frames == 0 {
        return None;
    }
    let up: f64 = acc[hi0..hi1].iter().sum();
    let lo: f64 = acc[lo0..lo1].iter().sum();
    if !(lo > 0.0) {
        return None;
    }
    Some((10.0 * ((up + 1e-30) / lo).log10()) as f32)
}

/// ⭐⭐⭐ S163 —— **逐周期峰值的相对起伏**(shimmer),单位 dB。
///
/// 把信号按 `f0` 切成整周期,取每周期的峰值,报 `20·log10(peak / median)` 的
/// **绝对值 p75**。
///
/// # ⛔⛔ 这是一把**双向**的尺子 —— 唯一正确的用法是拿 `base` 当靶子
/// * **太高** = 用户 2026-08-26 点名的「卡痰」:`donor_pre` 上 **9.41**,而 `base` **7.24**;
/// * **太低** = 用户 2026-08-27 点破的另一头:「**那里除了 f0 剩下的部分都没声了,电都没了**」——
///   纯正弦的逐周期峰值完全相同 ⇒ shimmer 必然趋近 **0**。深救援 −11 上实测 **0.41**。
///
/// ⇒ 调用方**必须**算 `|shimmer(donor) − shimmer(base)|`,
///   ⛔ **绝不许写成「越小越好」** —— 那会把落点直接推进最坏的那一档。
///
/// # 口径
/// * 周期数 <8 ⇒ `None`(样本不够,读数是噪声);
/// * 丢掉峰值 <最大峰 5% 的周期(静音尾巴会把中位拖垮);
/// * 报 p75 而不是标准差:**离群的几个周期才是听得见的那部分**,而 std 会被它们拖着走。
///
/// # ⚠ 与另外三根轴测的不是一件事
/// `harmonic_energy_fraction_db` 量谐波占**总能量**的比例、`comb_depth_db` 量谐波**之间**有多空、
/// `harmonic_peak_width_pct` 量每根谐波**自己**糊不糊 —— 它们全是**频域**的。
/// shimmer 是**时域逐周期**的,一个频谱完全正常但每个周期忽大忽小的信号只有它看得见。
pub fn shimmer_db(x: &[f32], sample_rate: u32, f0_hz: f32) -> Option<f32> {
    if !(f0_hz > 20.0) || sample_rate == 0 {
        return None;
    }
    let p = (f64::from(sample_rate) / f64::from(f0_hz)).round() as usize;
    if p < 8 || x.len() < 8 * p {
        return None;
    }
    let k = x.len() / p;
    let mut pk: Vec<f64> = (0..k)
        .map(|i| {
            x[i * p..(i + 1) * p]
                .iter()
                .fold(0.0f64, |a, &v| a.max(f64::from(v).abs()))
        })
        .collect();
    let mx = pk.iter().fold(0.0f64, |a, &v| a.max(v));
    if !(mx > 0.0) {
        return None;
    }
    pk.retain(|&v| v > mx * 0.05);
    if pk.len() < 8 {
        return None;
    }
    let mut sorted = pk.clone();
    sorted.sort_by(f64::total_cmp);
    let med = sorted[sorted.len() / 2];
    if !(med > 0.0) {
        return None;
    }
    let mut d: Vec<f64> = pk.iter().map(|&v| (20.0 * (v / med).log10()).abs()).collect();
    d.sort_by(f64::total_cmp);
    let i = (d.len() * 3) / 4;
    Some(d[i.min(d.len() - 1)] as f32)
}

/// ⭐⭐⭐ S165 —— **第二谐波(2·f0)相对基频的电平**,单位 dB。
///
/// # 它是干什么的
/// 用户 2026-08-28 指着三张频谱图说 yachiyo 的谐波线「**非常小段的忽明忽暗、不是一条干净的
/// 连续线**」,断得最明显的是「**基频上面那条**」(= `2·f0`)。本场先后造了**八把**量
/// 「沿时间的变化」的尺子(归一包络 std/mean · 包络谱尖峰度 · STFT 条纹 · 短窗峰宽 ·
/// 谐波间填充 · 存在率 · 40-60 Hz 调制占比 · log 域绝对快抖),**八把全部读反或分不开**。
///
/// 换轴之后一次命中 —— 三条臂最大的差异根本不在「沿时间」,而在「**跨谐波的幅度分布**」:
///
/// | 目标音(炉心 `[794]あ`,f0 987.8 Hz) | `2·f0` 相对 `f0` | 偏离「左右两根连线」 |
/// |---|---|---|
/// | yachiyo(用户说**油**) | **−17.0 dB** | **−3.3**(凹进去) |
/// | yuyuko(用户说好) | −5.2 dB | +11.6(凸出来) |
/// | SV(用户说好) | +0.9 dB | +13.1(凸出来) |
///
/// 而 `3·f0` 上三条臂**一致**(−11.6 / −12.3 / −16.3 都是谷)⇒ **只有这一根把它们分开**,落差 15 dB。
/// ⇒ ⭐ **「忽明忽暗」是低电平的后果不是原因**:一根本来就暗 12-18 dB 的线叠上**正常**的抖动,
///   在频谱图上就是断续的;亮的线抖同样多却看着是实心的。
///
/// # ⛔ 为什么 [`upper_harmonic_level_db`] 看不见它
/// 那一根量的是 **`2..8·f0` 的整体能量**;`2·f0` 只是其中一根,
/// **12 dB 的缺口被 7 根谐波一平均只剩 ~1.7 dB** ⇒ 它结构上拦不住这件事。
///
/// # ⛔ 口径(每一条都是本场栽出来的)
/// * **窄窗 `±0.08·f0`**:实测同一份 `base`,窗宽从 `0.15` 放到 `0.20·f0`
///   让读数跳 **14.7 dB**(−21.9 → −7.2,后者是假的 —— 它抓到了 1800 Hz 附近的**非谐波**能量);
/// * **验峰位**:峰偏离 `2·f0` 超过 `1.5%` ⇒ 返回 `None`(**「这根谐波不存在」不许读成一个数**)。
///   实测 `base`(模型硬唱、音高不准)正是这种情况,偏离 −2.1%;
/// * 与响度无关(比值),所以不吃「谁轻谁赢」的亏。
///
/// # ⚠ 这**不是**一把「越高越好」的尺子
/// 用户 2026-08-28 说 yuyuko 整体「**像大电流音刺耳**」—— 那一条的 `2·f0` 恰恰是最强的。
/// 而实测把落点从 −8 换到 −13 虽然把这根从 −17.0 拉到 −5.2,却把 `5·f0` **弄坏 24 dB**。
/// ⇒ ⛔ **调用方必须配对手轴闸**;⛔ 绝不许写成单向的「越大越好」。
pub fn second_harmonic_level_db(x: &[f32], sample_rate: u32, f0_hz: f32) -> Option<f32> {
    const N: usize = 4096;
    const TOL: f64 = 0.08;
    const MAX_OFF: f64 = 0.015;
    if !(f0_hz > 20.0) || sample_rate == 0 || x.len() < N {
        return None;
    }
    let sr = f64::from(sample_rate);
    let f0 = f64::from(f0_hz);
    if 2.0 * f0 * (1.0 + TOL) >= sr / 2.0 {
        return None;
    }
    let bin = sr / N as f64;
    let win: Vec<f64> = (0..N)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N as f64).cos())
        .collect();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(N);
    let mut acc = vec![0f64; N / 2 + 1];
    let mut i = 0usize;
    let mut frames = 0usize;
    while i + N <= x.len() {
        let mut buf: Vec<Complex<f64>> =
            (0..N).map(|j| Complex::new(f64::from(x[i + j]) * win[j], 0.0)).collect();
        fft.process(&mut buf);
        for (a, b) in acc.iter_mut().zip(buf.iter()) {
            *a += b.norm_sqr();
        }
        i += N / 2;
        frames += 1;
    }
    if frames == 0 {
        return None;
    }
    // 在 `c·(1±TOL)` 里找峰,并把峰所在的频率一起带出来验位。
    let peak = |c: f64| -> Option<(f64, f64)> {
        let lo = ((c * (1.0 - TOL)) / bin).floor().max(1.0) as usize;
        let hi = (((c * (1.0 + TOL)) / bin).ceil() as usize).min(N / 2);
        if hi <= lo {
            return None;
        }
        let (mut bj, mut bv) = (lo, acc[lo]);
        for j in lo..=hi {
            if acc[j] > bv {
                bv = acc[j];
                bj = j;
            }
        }
        Some((bj as f64 * bin, bv))
    };
    let (f_lo, p_lo) = peak(f0)?;
    let (f_hi, p_hi) = peak(2.0 * f0)?;
    if !(p_lo > 0.0) || !(p_hi > 0.0) {
        return None;
    }
    // ⛔ 峰位验证:两根都得真的在它们该在的地方。
    if ((f_lo - f0) / f0).abs() > MAX_OFF || ((f_hi - 2.0 * f0) / (2.0 * f0)).abs() > MAX_OFF {
        return None;
    }
    Some((10.0 * (p_hi / p_lo).log10()) as f32)
}

#[cfg(test)]
mod tests {

    /// ⭐⭐⭐ S165 —— [`second_harmonic_level_db`] 的口径:**窄窗 + 验峰位 + 与响度无关**。
    ///
    /// 每一条都钉住本场实际栽过的坑,不是凑数的断言。
    #[test]
    fn the_second_harmonic_ruler_reads_that_one_harmonic_and_refuses_when_it_is_not_there() {
        let sr = 48_000u32;
        let f0 = 987.8f32;
        let n = 48_000usize;
        let mk = |h2_gain: f32, off_pct: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let t = i as f32 / sr as f32;
                    let a = (2.0 * std::f32::consts::PI * f0 * t).sin();
                    let b = (2.0 * std::f32::consts::PI * 2.0 * f0 * (1.0 + off_pct) * t).sin();
                    0.3 * (a + h2_gain * b)
                })
                .collect()
        };

        // ⑴ 等幅 ⇒ 0 dB;按已知增益缩放 ⇒ 读回那个增益。
        let v0 = second_harmonic_level_db(&mk(1.0, 0.0), sr, f0).expect("等幅该读得到");
        assert!(v0.abs() < 0.5, "等幅该读 0 dB,实际 {v0:.2}");
        for g in [0.5f32, 0.1, 0.03] {
            let v = second_harmonic_level_db(&mk(g, 0.0), sr, f0).expect("有峰就该读得到");
            let want = 20.0 * g.log10();
            assert!((v - want).abs() < 1.0, "增益 {g} 该读 {want:.1} dB,实际 {v:.1}");
        }

        // ⑵ ⛔ 峰位验证:第二谐波偏离 5% ⇒ 必须 `None`,**不许读成一个数**。
        //    实测 `base`(模型硬唱、音高不准)正是这种情况(偏离 −2.1%)。
        assert!(
            second_harmonic_level_db(&mk(1.0, 0.05), sr, f0).is_none(),
            "峰偏离 5% 还给读数 ⇒ 「这根谐波不存在」被读成了「它很强」"
        );

        // ⑶ ⛔ 与响度无关 —— 它是比值,不许变成「谁轻谁赢」的尺子。
        let loud = mk(0.3, 0.0);
        let quiet: Vec<f32> = loud.iter().map(|v| v * 0.05).collect();
        let a = second_harmonic_level_db(&loud, sr, f0).unwrap();
        let b = second_harmonic_level_db(&quiet, sr, f0).unwrap();
        assert!((a - b).abs() < 0.2, "读数被响度影响了:{a:.2} vs {b:.2}");

        // ⑷ ⭐ 与 `upper_harmonic_level_db` **测的不是一件事**:
        //    只把 2·f0 挖掉、3..8·f0 全部保留 ⇒ 这一根必须塌,而那一根几乎不动
        //    (这正是「12 dB 的缺口被 7 根谐波平均成 1.7 dB」的复现)。
        let full: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                let mut y = (2.0 * std::f32::consts::PI * f0 * t).sin();
                for k in 2..=8u32 {
                    y += (2.0 * std::f32::consts::PI * f0 * k as f32 * t).sin();
                }
                0.1 * y
            })
            .collect();
        let dug: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                let mut y = (2.0 * std::f32::consts::PI * f0 * t).sin();
                for k in 2..=8u32 {
                    let g = if k == 2 { 0.06 } else { 1.0 }; // 只把 2·f0 挖掉 ~24 dB
                    y += g * (2.0 * std::f32::consts::PI * f0 * k as f32 * t).sin();
                }
                0.1 * y
            })
            .collect();
        let (s_full, s_dug) = (
            second_harmonic_level_db(&full, sr, f0).unwrap(),
            second_harmonic_level_db(&dug, sr, f0).unwrap(),
        );
        let (u_full, u_dug) = (
            upper_harmonic_level_db(&full, sr, f0).unwrap(),
            upper_harmonic_level_db(&dug, sr, f0).unwrap(),
        );
        assert!(
            s_full - s_dug > 18.0,
            "只挖 2·f0,这一根该塌 >18 dB:{s_full:.1} → {s_dug:.1}"
        );
        assert!(
            (u_full - u_dug) < 6.0,
            "⭐ 这正是要钉的盲区:`upper_harmonic_level_db` 把 2·f0 的缺口平均掉了,\
             不该塌这么多({u_full:.1} → {u_dug:.1})—— 若它也塌了,说明两根轴重复了"
        );
    }

    /// ⛔⛔ 这条钉的是 [`shimmer_db`] **两个方向**都要动 —— 一把只在一头有反应的尺子,
    /// 拿来当「贴近 base」的判据时会静默地把落点推向另一头。
    ///
    /// 三条信号,同一个 `f0`、同一个平均电平:
    /// * `flat`  = 纯正弦 ⇒ 逐周期峰值完全相同 ⇒ shimmer ≈ 0 = **「电都没了」那一头**;
    /// * `mid`   = 峰值 ±12% 起伏 ⇒ 中间;
    /// * `rough` = 峰值 ±45% 起伏 ⇒ **「卡痰」那一头**。
    ///
    /// ⚠ 变异检查:把 p75 换成 p50、或者不丢静音周期,`flat` 与 `mid` 会挤到一起。
    #[test]
    fn shimmer_moves_in_both_directions_and_is_level_invariant() {
        let sr = 44_100u32;
        let f0 = 220.0f32;
        let p = (f64::from(sr) / f64::from(f0)).round() as usize;
        let build = |amp: &dyn Fn(usize) -> f64| -> Vec<f32> {
            let mut v = Vec::with_capacity(p * 40);
            for i in 0..p * 40 {
                let a = amp(i / p);
                v.push(
                    (a * (2.0 * std::f64::consts::PI * f64::from(f0) * i as f64
                        / f64::from(sr))
                        .sin()) as f32,
                );
            }
            v
        };
        let flat = build(&|_| 0.5);
        let mid = build(&|c| if c % 2 == 0 { 0.5 } else { 0.44 });
        let rough = build(&|c| if c % 2 == 0 { 0.5 } else { 0.275 });

        let s_flat = shimmer_db(&flat, sr, f0).expect("flat");
        let s_mid = shimmer_db(&mid, sr, f0).expect("mid");
        let s_rough = shimmer_db(&rough, sr, f0).expect("rough");

        assert!(
            s_flat < 0.2,
            "纯正弦的 shimmer 必须趋近 0（这一头就是「电都没了」）—— 实测 {s_flat:.3}"
        );
        assert!(
            s_mid > s_flat + 0.5 && s_rough > s_mid + 1.0,
            "三档必须单调且拉得开 —— flat {s_flat:.2} < mid {s_mid:.2} < rough {s_rough:.2}"
        );

        // ⛔ 电平不变性:整体乘 0.1 读数必须不动(否则它会变成一把「谁轻谁赢」的尺子 ——
        //    S163 §7 已经栽过一次)。
        let quiet: Vec<f32> = rough.iter().map(|v| v * 0.1).collect();
        let s_quiet = shimmer_db(&quiet, sr, f0).expect("quiet");
        assert!(
            (s_quiet - s_rough).abs() < 0.05,
            "shimmer 必须与电平无关 —— {s_rough:.3} vs {s_quiet:.3}"
        );

        // 周期数不够时必须 None，不许返回一个噪声读数当真话
        assert!(shimmer_db(&rough[..p * 4], sr, f0).is_none(), "周期不足必须 None");
    }


    /// ⛔ 钉住 [`upper_harmonic_level_db`] 与 [`upper_harmonic_sag_db`] **测的不是一件事** ——
    /// 这正是它作为「对手轴」的全部价值。
    ///
    /// * `bright` 与 `dark` 上方谐波强度差 ~12 dB,而两者**都恒定** ⇒ `sag` 读数应该都 ≈ 0;
    /// * `drained` 上方谐波随时间衰减 ⇒ `sag` 明显负,而**整段平均强度**介于两者之间。
    #[test]
    fn upper_level_and_upper_sag_measure_different_things() {
        let sr = 44_100u32;
        let f0 = 300.0f32;
        let n = sr as usize;
        let mk = |amp: &dyn Fn(f64) -> f64| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let t = i as f64 / f64::from(sr);
                    let p = 2.0 * std::f64::consts::PI * f64::from(f0) * t;
                    (0.3 * p.sin() + amp(t) * (2.0 * p).sin() + amp(t) * 0.7 * (3.0 * p).sin())
                        as f32
                })
                .collect()
        };
        let bright = mk(&|_t| 0.25f64);
        let dark = mk(&|_t| 0.0625f64);
        let drained = mk(&|t| 0.25f64 * 10f64.powf(-30.0 * t / 20.0));

        let (lb, ld) = (
            upper_harmonic_level_db(&bright, sr, f0).expect("bright"),
            upper_harmonic_level_db(&dark, sr, f0).expect("dark"),
        );
        assert!(
            lb - ld > 8.0,
            "强度轴必须分得开亮/暗 —— bright {lb:.1} vs dark {ld:.1} dB"
        );
        let (sb, sd) = (
            upper_harmonic_sag_db(&bright, sr, f0).expect("bright"),
            upper_harmonic_sag_db(&dark, sr, f0).expect("dark"),
        );
        assert!(
            sb.abs() < 1.0 && sd.abs() < 1.0,
            "⛔ 两条都恒定 ⇒ sag 必须都 ≈ 0（否则它就不是「音内」的量了）—— {sb:.2} / {sd:.2}"
        );
        let s_dr = upper_harmonic_sag_db(&drained, sr, f0).expect("drained");
        assert!(
            s_dr < -5.0,
            "而衰减的那条 sag 必须明显为负 —— {s_dr:.2} dB"
        );
    }


    /// ⛔⛔ 这条钉的是 [`upper_harmonic_sag_db`] **只对「上方谐波随时间被抽干」有反应**,
    /// 而对「整体音量的起伏」和「清辅音起头」**没有**反应 —— 后两者是本场已经栽过的两个坑:
    /// * 全带 p90−p10 被 か/が 这类清辅音起头的音读到 40-75 dB(那是辅音动态,不是缺陷);
    /// * 整音标量(RMS / 谐波占比)结构上看不见音**内**的形状。
    ///
    /// 三条信号,同一个 f0、同样的长度:
    /// * `flat`   —— 谐波恒定 ⇒ Δ ≈ 0;
    /// * `drained`—— **只有 H2/H3 随时间衰减**,H1 不动 ⇒ Δ 必须明显为负;
    /// * `dimmed` —— **整体**(含 H1)一起衰减同样的量 ⇒ 带内自比 ⇒ Δ 仍然明显为负
    ///               (这是对的:上方谐波确实弱了),但**幅度不该超过 `drained`**。
    /// * `louder` —— 整体乘 10 倍 ⇒ Δ 必须**逐位不变**(证明它与绝对电平无关)。
    #[test]
    fn upper_harmonic_sag_sees_drained_harmonics_not_loudness() {
        let sr = 44_100u32;
        let f0 = 300.0f32;
        let n = sr as usize; // 1 s
        let mk = |h1: &dyn Fn(f64) -> f64, up: &dyn Fn(f64) -> f64| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let t = i as f64 / f64::from(sr);
                    let p = 2.0 * std::f64::consts::PI * f64::from(f0) * t;
                    (h1(t) * p.sin()
                        + up(t) * (2.0 * p).sin() * 0.6
                        + up(t) * (3.0 * p).sin() * 0.4) as f32
                })
                .collect()
        };
        let one = |_t: f64| 0.3f64;
        // ⛔ 衰减率写陡一点是有理由的:STFT 窗本身在平均,而口径又掐掉了尾部 15%
        //    ⇒ 读出来的 Δ 必然**小于**信号真实的端到端衰减量。这里断言的是
        //    「与 flat 拉开距离」而不是某个硬数值 —— 硬数值会变成拿测试拟合实现。
        let fade = |t: f64| 0.3f64 * 10f64.powf(-30.0 * t / 20.0); // 1 s 内衰 30 dB
        let flat = mk(&one, &one);
        let drained = mk(&one, &fade);
        let dimmed = mk(&fade, &fade);

        let s_flat = upper_harmonic_sag_db(&flat, sr, f0).expect("flat");
        let s_drain = upper_harmonic_sag_db(&drained, sr, f0).expect("drained");
        let s_dim = upper_harmonic_sag_db(&dimmed, sr, f0).expect("dimmed");

        assert!(
            s_flat.abs() < 1.0,
            "恒定谐波的 Δ 必须 ≈ 0 —— 实测 {s_flat:.2} dB"
        );
        assert!(
            s_drain < s_flat - 3.0,
            "上方谐波被抽干必须与恒定谐波**拉开距离** —— flat {s_flat:.2} vs drained {s_drain:.2} dB"
        );
        assert!(
            s_drain < -5.0,
            "而且必须是明确的负值(不是噪声) —— 实测 {s_drain:.2} dB"
        );
        assert!(
            (s_dim - s_drain).abs() < 3.0,
            "带内自比 ⇒ 整体一起衰减与只衰上方谐波读数应接近 —— {s_dim:.2} vs {s_drain:.2}"
        );

        // ⛔ 电平不变性:整体乘 10,读数必须不动
        let louder: Vec<f32> = drained.iter().map(|v| v * 10.0).collect();
        let s_loud = upper_harmonic_sag_db(&louder, sr, f0).expect("louder");
        assert!(
            (s_loud - s_drain).abs() < 0.05,
            "必须与绝对电平无关 —— {s_drain:.3} vs {s_loud:.3}"
        );

        // 太短必须 None，不许拿噪声读数当真话
        assert!(
            upper_harmonic_sag_db(&drained[..4096], sr, f0).is_none(),
            "帧数不足必须 None"
        );
    }

    use super::*;

    fn stack(f0: f32, sr: u32, secs: f32, harmonics: usize, amp: f32) -> Vec<f32> {
        let n = (sr as f32 * secs) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                let mut v = 0.0;
                for k in 1..=harmonics {
                    v += (1.0 / k as f32)
                        * (2.0 * std::f32::consts::PI * f0 * k as f32 * t).sin();
                }
                v * amp
            })
            .collect()
    }

    fn noise(n: usize, seed: u32, amp: f32) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                ((s as f32 / u32::MAX as f32) * 2.0 - 1.0) * amp
            })
            .collect()
    }

    /// ⒜ **一个干净的谐波堆读接近 0 dB**;⒝ **白噪声读得很负**。
    /// 这两条一起说明它量的是「能量在不在谐波上」,不是「有多少能量」。
    #[test]
    fn it_separates_a_harmonic_stack_from_noise() {
        let sr = 44100;
        let h = harmonic_energy_fraction_db(&stack(440.0, sr, 0.6, 12, 0.2), sr, 440.0).unwrap();
        let n = harmonic_energy_fraction_db(&noise(sr as usize * 6 / 10, 7, 0.2), sr, 440.0).unwrap();
        assert!(h > -1.5, "谐波堆应当接近 0 dB,读到 {h}");
        assert!(n < -4.0, "白噪声应当明显负,读到 {n}");
        assert!(h - n > 4.0, "两者的差 {} dB 太小,这把尺子分不开东西", h - n);
    }

    /// ⒞ ⭐⭐ **承重的那一条:音高不对就要塌。**
    /// 同一个谐波堆,问它「你是不是在 660 Hz(= 1.5×)上发声」——必须大幅变负。
    /// 这正是落点选法要问的问题:**这个候选到底救到没有**。
    #[test]
    fn it_collapses_when_the_pitch_is_wrong() {
        let sr = 44100;
        let x = stack(440.0, sr, 0.6, 12, 0.2);
        let right = harmonic_energy_fraction_db(&x, sr, 440.0).unwrap();
        let wrong = harmonic_energy_fraction_db(&x, sr, 660.0).unwrap();
        assert!(right - wrong > 6.0, "问错音高应当塌:440 读 {right},660 读 {wrong}");
    }

    /// ⒟ ⛔⛔ **必须与电平无关。** S162 的 `subf0` 护栏就是栽在这一条上:
    /// 它是**绝对**带能量差,于是「整体变安静」被读成「面状变好了」,把一整轮结论读反号。
    /// ⇒ 同一段音频乘 0.1(−20 dB)必须给出**逐位相同**的读数。
    #[test]
    fn it_is_blind_to_level() {
        let sr = 44100;
        let x = stack(440.0, sr, 0.6, 12, 0.2);
        let quiet: Vec<f32> = x.iter().map(|v| v * 0.1).collect();
        let a = harmonic_energy_fraction_db(&x, sr, 440.0).unwrap();
        let b = harmonic_energy_fraction_db(&quiet, sr, 440.0).unwrap();
        assert!((a - b).abs() < 0.05, "电平变了读数就变 = 这把尺子会被响度骗:{a} vs {b}");
    }

    /// ⒠ 中间值也要有序:谐波堆里掺噪声,占比必须**单调**下降(不是二值)。
    #[test]
    fn it_is_monotone_in_how_much_noise_is_mixed_in() {
        let sr = 44100;
        let x = stack(440.0, sr, 0.6, 12, 0.2);
        let mut last = f32::INFINITY;
        for w in [0.0f32, 0.05, 0.2, 0.8] {
            let nz = noise(x.len(), 11, 0.2 * w);
            let y: Vec<f32> = x.iter().zip(nz.iter()).map(|(a, b)| a + b).collect();
            let v = harmonic_energy_fraction_db(&y, sr, 440.0).unwrap();
            assert!(v < last + 1e-3, "掺噪声 {w} 反而更谐波:{v} ≥ {last}");
            last = v;
        }
    }

    /// ⒡ 拒绝而不是瞎猜:样本不够一帧 / f0 荒谬 ⇒ `None`(调用方据此**弃权**,
    /// 而不是拿一个编出来的数去做决定)。
    #[test]
    fn it_declines_instead_of_guessing() {
        let sr = 44100;
        let x = stack(440.0, sr, 0.6, 12, 0.2);
        assert!(harmonic_energy_fraction_db(&x[..1000], sr, 440.0).is_none());
        assert!(harmonic_energy_fraction_db(&x, sr, 0.0).is_none());
        assert!(harmonic_energy_fraction_db(&x, sr, 30.0).is_none());
        assert!(harmonic_energy_fraction_db(&x, sr, sr as f32 * 0.3).is_none());
        assert!(harmonic_energy_fraction_db(&[], sr, 440.0).is_none());
    }


    /// ⛔⛔ S163 —— **梳深与谐波占比测的不是一件事**,而且在真实缺陷上方向相反。
    /// ⑴ 谐波堆 + 谐波之间填噪声 ⇒ **梳深塌而谐波占比几乎不动**(= 用户说的「卡痰」);
    /// ⑵ 只留 H1 一根 ⇒ **梳深仍然高**(= 「ぴゃ」);
    /// ⑶ 音高问错 ⇒ 塌;⑷ 与电平无关。
    /// ⭐⭐⭐ S163 —— [`harmonic_peak_width_pct`] 量的是**每一根谐波自己糊不糊**，
    /// 而另外两根轴对它都是瞎的。三件：
    /// ① 干净的谐波串 ⇒ 峰宽小；② 加频率抖动 ⇒ 峰宽变大（对照组）；
    /// ③ **与电平无关**（同一份波形 × 0.01 读数不变）。
    /// ⛔ ② 是承重那一格：没有它，“峰宽总是小”也能让 ① 绿。
    #[test]
    fn peak_width_sees_frequency_smear_that_the_other_two_axes_miss() {
        let sr = 44_100u32;
        let f0 = 440.0f32;
        let n = (sr as f32 * 0.40) as usize;
        // ① 干净：H1..H8 固定频率
        let clean: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                let mut y = 0.0f32;
                for k in 1..=8u32 {
                    y += (1.0 / k as f32)
                        * (2.0 * std::f32::consts::PI * f0 * k as f32 * t).sin();
                }
                0.2 * y
            })
            .collect();
        // ② 同一批谐波，但 f0 带 3% 的慢抖动（相位积分）
        let smeared: Vec<f32> = {
            let mut ph = 0.0f32;
            (0..n)
                .map(|i| {
                    let t = i as f32 / sr as f32;
                    let inst = f0 * (1.0 + 0.03 * (2.0 * std::f32::consts::PI * 7.0 * t).sin());
                    ph += 2.0 * std::f32::consts::PI * inst / sr as f32;
                    let mut y = 0.0f32;
                    for k in 1..=8u32 {
                        y += (1.0 / k as f32) * (ph * k as f32).sin();
                    }
                    0.2 * y
                })
                .collect()
        };
        let wc = harmonic_peak_width_pct(&clean, sr, f0).expect("clean");
        let ws = harmonic_peak_width_pct(&smeared, sr, f0).expect("smeared");
        assert!(wc < 3.0, "干净谐波串的峰宽应该很小（读到 {wc:.2}）");
        assert!(
            ws > wc * 2.0,
            "频率抖动必须把峰宽拉开（干净 {wc:.2} vs 抖动 {ws:.2}）——              否则上一格测的是「峰宽总是小」"
        );
        // ③ 与电平无关
        let quiet: Vec<f32> = clean.iter().map(|v| v * 0.01).collect();
        let wq = harmonic_peak_width_pct(&quiet, sr, f0).expect("quiet");
        assert!(
            (wq - wc).abs() < 0.51,
            "峰宽必须与电平无关（{wc:.2} vs {wq:.2}）"
        );
        // ⛔ 另外两根轴在同一份对照上**看不见**这件事——这才是它存在的理由
        let hc = harmonic_energy_fraction_db(&clean, sr, f0).unwrap_or(0.0);
        let hs = harmonic_energy_fraction_db(&smeared, sr, f0).unwrap_or(0.0);
        assert!(
            (hs - hc).abs() < (ws - wc) / 2.0,
            "谐波占比对频率展宽的反应必须远小于峰宽本身（             占比 Δ{:.2} vs 峰宽 Δ{:.2}）",
            hs - hc,
            ws - wc
        );
    }

    #[test]
    fn comb_depth_and_harmonic_fraction_measure_different_things() {
        let sr = 44100;
        let clean = stack(440.0, sr, 0.6, 12, 0.2);
        let c0 = comb_depth_db(&clean, sr, 440.0).unwrap();
        let h0 = harmonic_energy_fraction_db(&clean, sr, 440.0).unwrap();
        // ⑴ 谐波之间填噪声(带通到 2-8 kHz 的白噪,幅度只有谐波的 1/8)
        let nz = noise(clean.len(), 3, 0.025);
        let fogged: Vec<f32> = clean.iter().zip(nz.iter()).map(|(a, b)| a + b).collect();
        let c1 = comb_depth_db(&fogged, sr, 440.0).unwrap();
        let h1 = harmonic_energy_fraction_db(&fogged, sr, 440.0).unwrap();
        assert!(c0 - c1 > 6.0, "起雾之后梳深必须塌:{c0:.1} → {c1:.1}");
        assert!(h0 - h1 < c0 - c1, "谐波占比不该比梳深更敏感:{h0:.2} → {h1:.2}");
        // ⑵ ⭐ 「ぴゃ」那一类:H1 极强、H2 以上**弱 40 dB 但仍然存在**
        //    ⇒ 梳深仍然很高(它看的是谐波**位置** vs 谐波**中点**,不是谐波有多强)。
        //    ⛔ 第一版夹具写成「只留 H1 一根」,那样 2-8 kHz 里根本没有谐波,
        //    量到的是本底(读 7.1)—— 而真实的「ぴゃ」在 2-8 kHz 里是有谐波的(梳深 40.6)。
        let n2 = (sr as f32 * 0.6) as usize;
        let mut one = vec![0.0f32; n2];
        for (i, v) in one.iter_mut().enumerate() {
            let t = i as f32 / sr as f32;
            let mut y = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
            for k in 2..=12usize {
                y += (0.01 / k as f32)
                    * (2.0 * std::f32::consts::PI * 440.0 * k as f32 * t).sin();
            }
            *v = 0.2 * y;
        }
        let c2 = comb_depth_db(&one, sr, 440.0).unwrap();
        assert!(c2 > 20.0, "谐波很弱但存在时梳深不该塌({c2:.1})—— 那是另一根轴的事");
        // ⛔ 而**谐波占比**在同一份素材上必须**也没事**(它看的是能量在不在谐波上)
        let h2v = harmonic_energy_fraction_db(&one, sr, 440.0).unwrap();
        assert!(h2v > -3.0, "谐波占比在「只剩基频」上不该塌({h2v:.2})—— 这正是它的盲区");
        // ⑶ 问错音高 ⇒ 塌
        let c3 = comb_depth_db(&clean, sr, 660.0).unwrap();
        assert!(c0 - c3 > 6.0, "问错音高梳深该塌:{c0:.1} → {c3:.1}");
        // ⑷ ⛔ 与电平无关
        let quiet: Vec<f32> = clean.iter().map(|v| v * 0.1).collect();
        let c4 = comb_depth_db(&quiet, sr, 440.0).unwrap();
        assert!((c0 - c4).abs() < 0.05, "梳深被响度影响了:{c0:.2} vs {c4:.2}");
    }

    /// ⒢ 确定性:同一份输入两次逐位相同(⛔ 判据里要拿它当零噪声口径)。
    #[test]
    fn it_is_deterministic() {
        let sr = 48000;
        let x = stack(523.25, sr, 0.5, 10, 0.3);
        let a = harmonic_energy_fraction_db(&x, sr, 523.25).unwrap();
        let b = harmonic_energy_fraction_db(&x, sr, 523.25).unwrap();
        assert_eq!(a.to_bits(), b.to_bits());
    }
}
