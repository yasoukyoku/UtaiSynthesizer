//! S163 —— **最小相位 high-shelf**（RBJ biquad），给「压回某个频带的电平」用。
//!
//! # 为什么需要它
//! 救援在**清辅音**区把 4-8 kHz 抬高了 6-16 dB（同 run 零噪声，四模型；
//! 窗覆盖 n=217 p90 **+8.1** dB / >6 dB 的 37 处，而**窗未覆盖 n=127 一处都不超 6 dB**
//! ⇒ 归因完全锁死在救援上）。用户听到的「音头咔哒竖线」就是这一族：
//! `3:37.972` +14.0 · `3:46.605` +12.6 · `3:55.266` +15.4 · `3:34.973` +11.6 · `1:23.081` +16.0。
//!
//! # 为什么是 shelf 而不是换内容
//! S163 三次栽在同一个根上：**往窗里插一段别的内容，两端必然造缝**
//! （dipfill 的填充台阶 p90 18.3 dB、休止 v9 的音头竖线 16.3 → 25.8 dB）。
//! **改增益不改内容才不造缝** —— 这里把「改增益」推广到「改某个频带的增益」。
//!
//! # 为什么是最小相位（IIR）而不是零相位
//! 零相位（前后各滤一次）会产生 **pre-ringing**，落在辅音这种瞬态上正是新的「咔哒」。
//! RBJ biquad 是最小相位，只有群延迟，没有前振铃。

/// RBJ high-shelf 的二阶系数 + 状态（Direct Form I）。
#[derive(Clone, Copy, Debug)]
pub struct HighShelf {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl HighShelf {
    /// `f0` = 拐点频率(Hz)，`gain_db` = 高频侧增益（**负 = 压**），`s` = 斜率(1.0 = 最陡不过冲)。
    ///
    /// 返回 `None`：采样率/频率不合法，或增益为 0（调用方应当直接跳过，别做恒等滤波）。
    #[must_use]
    pub fn new(sample_rate: u32, f0: f32, gain_db: f32, s: f32) -> Option<Self> {
        let sr = f64::from(sample_rate);
        let f0 = f64::from(f0);
        if !(sr > 0.0) || !(f0 > 0.0) || f0 >= sr * 0.5 || !gain_db.is_finite() {
            return None;
        }
        if gain_db.abs() < 1e-3 {
            return None; // 恒等 ⇒ 不要白跑一遍滤波器
        }
        let a = 10f64.powf(f64::from(gain_db) / 40.0); // 幅度域的 sqrt(gain)
        let w0 = 2.0 * std::f64::consts::PI * f0 / sr;
        let (sw, cw) = (w0.sin(), w0.cos());
        let s = f64::from(s).clamp(0.1, 2.0);
        let alpha = sw / 2.0 * ((a + 1.0 / a) * (1.0 / s - 1.0) + 2.0).sqrt();
        let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;

        let b0 = a * ((a + 1.0) + (a - 1.0) * cw + two_sqrt_a_alpha);
        let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cw);
        let b2 = a * ((a + 1.0) + (a - 1.0) * cw - two_sqrt_a_alpha);
        let a0 = (a + 1.0) - (a - 1.0) * cw + two_sqrt_a_alpha;
        let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cw);
        let a2 = (a + 1.0) - (a - 1.0) * cw - two_sqrt_a_alpha;
        if !(a0.abs() > 1e-12) {
            return None;
        }
        Some(Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        })
    }

    /// 逐样本。⛔ 状态是有记忆的：一段音频要从头到尾用同一个实例。
    pub fn tick(&mut self, x: f32) -> f32 {
        let x0 = f64::from(x);
        let y0 = self.b0 * x0 + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x0;
        self.y2 = self.y1;
        self.y1 = y0;
        y0 as f32
    }

    /// 用前 `warmup` 个样本先把状态喂饱再输出，避免段首的启动瞬态自己变成一个咔哒。
    /// ⛔ 这正是这一族缺陷本身 —— 别在修它的路上再造一个。
    pub fn filter_in_place(buf: &mut [f32], sample_rate: u32, f0: f32, gain_db: f32, warmup: usize) {
        let Some(mut f) = Self::new(sample_rate, f0, gain_db, 1.0) else {
            return;
        };
        let w = warmup.min(buf.len());
        for &v in buf.iter().take(w) {
            let _ = f.tick(v);
        }
        for v in buf.iter_mut() {
            *v = f.tick(*v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn band_db(x: &[f32], sr: u32, lo: f32, hi: f32) -> f32 {
        let n = 1usize << (x.len() as f32).log2().floor() as usize;
        let mut re: Vec<f64> = Vec::with_capacity(n);
        for (i, v) in x.iter().take(n).enumerate() {
            let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos();
            re.push(f64::from(*v) * w);
        }
        // 朴素 DFT 只在测试里用，n 取 4096 足够快
        let mut acc = 0.0f64;
        let bins = n / 2;
        for k in 1..bins {
            let f = k as f32 * sr as f32 / n as f32;
            if f < lo || f > hi {
                continue;
            }
            let (mut sr_, mut si) = (0.0f64, 0.0f64);
            let w = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
            for (i, v) in re.iter().enumerate() {
                sr_ += v * (w * i as f64).cos();
                si -= v * (w * i as f64).sin();
            }
            acc += sr_ * sr_ + si * si;
        }
        10.0 * (acc + 1e-30).log10() as f32
    }

    fn tone(n: usize, sr: u32, f: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * f * i as f32 / sr as f32).sin())
            .collect()
    }

    #[test]
    /// high-shelf 压 6 dB：**高频侧真的降了 ~6 dB，低频侧几乎不动**。
    fn high_shelf_cuts_the_top_and_leaves_the_bottom() {
        let sr = 44100u32;
        let n = 4096;
        for (f, want_cut) in [(6000.0f32, true), (300.0f32, false)] {
            let mut x = tone(n, sr, f);
            let before = band_db(&x, sr, f - 200.0, f + 200.0);
            HighShelf::filter_in_place(&mut x, sr, 4000.0, -6.0, 512);
            let after = band_db(&x, sr, f - 200.0, f + 200.0);
            let d = after - before;
            if want_cut {
                assert!(
                    (-8.0..=-4.0).contains(&d),
                    "6 kHz 应当被压 ~6 dB,实测 {d:.1}"
                );
            } else {
                assert!(d.abs() < 1.5, "300 Hz 不该被动,实测 {d:.1}");
            }
        }
    }

    #[test]
    /// ⛔ 增益 0 ⇒ **逐位不变**（调用方跳过滤波，不许做恒等运算引入舍入）。
    fn zero_gain_is_bit_for_bit_identity() {
        let sr = 44100u32;
        let orig = tone(2048, sr, 5000.0);
        let mut x = orig.clone();
        HighShelf::filter_in_place(&mut x, sr, 4000.0, 0.0, 256);
        assert_eq!(x, orig, "增益 0 必须逐位不变");
        assert!(HighShelf::new(sr, 4000.0, 0.0, 1.0).is_none());
    }

    #[test]
    /// ⛔ **段首不许有启动瞬态** —— 这一族缺陷本身就是「音头咔哒」，
    /// 修它的路上再造一个就白做了。喂饱状态之后，段首若干样本的包络必须平滑。
    fn warmup_kills_the_startup_transient() {
        let sr = 44100u32;
        let mut x = tone(4096, sr, 6000.0);
        HighShelf::filter_in_place(&mut x, sr, 4000.0, -9.0, 1024);
        let head: f32 = x[..64].iter().map(|v| v * v).sum::<f32>() / 64.0;
        let body: f32 = x[2048..2112].iter().map(|v| v * v).sum::<f32>() / 64.0;
        let d = 10.0 * (head / body.max(1e-20)).log10();
        assert!(d.abs() < 2.0, "段首相对段中差 {d:.1} dB —— 启动瞬态没被喂饱");
    }
}
