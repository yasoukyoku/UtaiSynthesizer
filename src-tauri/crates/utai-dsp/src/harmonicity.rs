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
