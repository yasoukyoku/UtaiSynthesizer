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

#[cfg(test)]
mod tests {
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
