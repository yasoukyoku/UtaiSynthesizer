//! Multi-band parametric bus EQ (Phase 4-1, 配方 F ②).
//!
//! RBJ Audio EQ Cookbook biquads chained in band order. Supported band
//! types: `Hpf` / `Lpf` (12 dB/oct resonant 2-pole), `Peaking`,
//! `LowShelf`, `HighShelf`. The caller supplies any number of bands —
//! the workflow UI ships a 5-band default (HPF 30 Hz, low shelf 120 Hz,
//! peaking 500 Hz / 2.5 kHz, high shelf 10 kHz).
//!
//! Filters are link-stereo: both channels use identical coefficients with
//! independent state, so the stereo image is never smeared. Invalid bands
//! (freq outside the stable range, q ≤ 0, unknown kind) are skipped, not
//! errors — a partial chain still renders.

/// Band shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandKind {
    Hpf,
    Lpf,
    Peaking,
    LowShelf,
    HighShelf,
}

impl BandKind {
    /// Parse the wire-format tag produced by the TS layer.
    pub fn from_tag(tag: &str) -> Option<BandKind> {
        match tag {
            "hpf" => Some(BandKind::Hpf),
            "lpf" => Some(BandKind::Lpf),
            "peaking" => Some(BandKind::Peaking),
            "lowShelf" => Some(BandKind::LowShelf),
            "highShelf" => Some(BandKind::HighShelf),
            _ => None,
        }
    }
}

/// One EQ band. `gain_db` is ignored by `Hpf`/`Lpf`.
#[derive(Debug, Clone)]
pub struct EqBand {
    pub kind: BandKind,
    pub freq: f64,
    pub gain_db: f64,
    pub q: f64,
    pub enabled: bool,
}

/// Normalized biquad coefficients (Direct Form I, a0 = 1).
#[derive(Debug, Clone, Copy)]
struct BiquadCoeffs {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

impl BiquadCoeffs {
    /// RBJ cookbook design. Returns `None` for numerically unstable bands.
    fn design(kind: BandKind, sr: f64, f0: f64, q: f64, gain_db: f64) -> Option<BiquadCoeffs> {
        let nyq = sr * 0.5;
        if f0 <= 20.0 * f64::EPSILON || f0 >= nyq * 0.95 || q <= 0.0 || sr <= 0.0 {
            return None;
        }
        let a = 10.0f64.powf(gain_db / 40.0); // shelf & peaking gain term
        let w0 = 2.0 * std::f64::consts::PI * f0 / sr;
        let (s, c) = w0.sin_cos();
        let alpha = s / (2.0 * q);

        let (b0, b1, b2, a0, a1, a2);
        match kind {
            BandKind::Hpf => {
                b0 = (1.0 + c) / 2.0;
                b1 = -(1.0 + c);
                b2 = (1.0 + c) / 2.0;
                a0 = 1.0 + alpha;
                a1 = -2.0 * c;
                a2 = 1.0 - alpha;
            }
            BandKind::Lpf => {
                b0 = (1.0 - c) / 2.0;
                b1 = 1.0 - c;
                b2 = (1.0 - c) / 2.0;
                a0 = 1.0 + alpha;
                a1 = -2.0 * c;
                a2 = 1.0 - alpha;
            }
            BandKind::Peaking => {
                b0 = 1.0 + alpha * a;
                b1 = -2.0 * c;
                b2 = 1.0 - alpha * a;
                a0 = 1.0 + alpha / a;
                a1 = -2.0 * c;
                a2 = 1.0 - alpha / a;
            }
            BandKind::LowShelf => {
                let sq = 2.0 * a.sqrt() * alpha;
                b0 = a * ((a + 1.0) - (a - 1.0) * c + sq);
                b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * c);
                b2 = a * ((a + 1.0) - (a - 1.0) * c - sq);
                a0 = (a + 1.0) + (a - 1.0) * c + sq;
                a1 = -2.0 * ((a - 1.0) + (a + 1.0) * c);
                a2 = (a + 1.0) + (a - 1.0) * c - sq;
            }
            BandKind::HighShelf => {
                let sq = 2.0 * a.sqrt() * alpha;
                b0 = a * ((a + 1.0) + (a - 1.0) * c + sq);
                b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * c);
                b2 = a * ((a + 1.0) + (a - 1.0) * c - sq);
                a0 = (a + 1.0) - (a - 1.0) * c + sq;
                a1 = 2.0 * ((a - 1.0) - (a + 1.0) * c);
                a2 = (a + 1.0) - (a - 1.0) * c - sq;
            }
        }
        Some(BiquadCoeffs {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        })
    }
}

/// One filter instance (coefficients + per-channel state).
#[derive(Debug, Clone, Copy)]
struct Biquad {
    c: BiquadCoeffs,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl Biquad {
    fn new(c: BiquadCoeffs) -> Biquad {
        Biquad { c, x1: 0.0, x2: 0.0, y1: 0.0, y2: 0.0 }
    }

    #[inline]
    fn process(&mut self, x: f64) -> f64 {
        let y = self.c.b0 * x + self.c.b1 * self.x1 + self.c.b2 * self.x2
            - self.c.a1 * self.y1
            - self.c.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

/// Design the active band chain. Returns the coefficients actually in
/// circuit — disabled, unknown-kind and unstable bands are dropped.
pub fn design_chain(sr: f64, bands: &[EqBand]) -> Vec<BiquadCoeffs> {
    bands
        .iter()
        .filter(|b| b.enabled)
        .filter_map(|b| BiquadCoeffs::design(b.kind, sr, b.freq, b.q, b.gain_db))
        .collect()
}

/// Run one channel through the chain in place.
pub fn process_channel(channel: &mut [f32], chain: &[BiquadCoeffs]) {
    if chain.is_empty() || channel.is_empty() {
        return;
    }
    let mut filters: Vec<Biquad> = chain.iter().map(|&c| Biquad::new(c)).collect();
    for s in channel.iter_mut() {
        let mut v = *s as f64;
        for f in filters.iter_mut() {
            v = f.process(v);
        }
        *s = v as f32;
    }
}

/// Link-stereo EQ over interleaved channel planes. Returns the number of
/// bands actually applied (after dropping invalid ones).
pub fn apply_eq(
    left: &mut [f32],
    right: Option<&mut [f32]>,
    sr: f64,
    bands: &[EqBand],
) -> usize {
    let chain = design_chain(sr, bands);
    if chain.is_empty() {
        return 0;
    }
    process_channel(left, &chain);
    if let Some(r) = right {
        process_channel(r, &chain);
    }
    chain.len()
}

/// The 5-band default the workflow node ships with (all bands enabled,
/// 0 dB = transparent until the user moves a control).
pub fn default_five_band() -> Vec<EqBand> {
    vec![
        EqBand { kind: BandKind::Hpf, freq: 30.0, gain_db: 0.0, q: 0.707, enabled: true },
        EqBand { kind: BandKind::LowShelf, freq: 120.0, gain_db: 0.0, q: 0.707, enabled: true },
        EqBand { kind: BandKind::Peaking, freq: 500.0, gain_db: 0.0, q: 1.0, enabled: true },
        EqBand { kind: BandKind::Peaking, freq: 2500.0, gain_db: 0.0, q: 1.0, enabled: true },
        EqBand { kind: BandKind::HighShelf, freq: 10000.0, gain_db: 0.0, q: 0.707, enabled: true },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f64 = 44100.0;

    /// Steady-state RMS gain in dB for a sine at `freq` through `bands`
    /// (skips the first 10 cycles — capped at half the buffer so the
    /// measured window always holds ≥ 10 cycles).
    fn sine_gain_db(freq: f64, bands: &[EqBand]) -> f64 {
        let n = (SR * 0.2) as usize; // 0.2 s
        let skip = ((SR * 10.0 / freq) as usize).min(n / 2);
        let mut x: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / SR).sin() as f32 * 0.5)
            .collect();
        let chain = design_chain(SR, bands);
        process_channel(&mut x, &chain);
        let rms = |s: &[f32]| {
            (s.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / s.len() as f64).sqrt()
        };
        let in_rms = 0.5 / 2.0f64.sqrt();
        20.0 * (rms(&x[skip..]) / in_rms).log10()
    }

    #[test]
    fn peaking_boosts_center_frequency() {
        let bands = vec![EqBand {
            kind: BandKind::Peaking,
            freq: 1000.0,
            gain_db: 12.0,
            q: 1.0,
            enabled: true,
        }];
        let g = sine_gain_db(1000.0, &bands);
        assert!(
            (g - 12.0).abs() < 0.5,
            "peaking at f0 should give ≈ +12 dB, got {g:.2}"
        );
        // Far away the bell barely moves the signal.
        let g_far = sine_gain_db(100.0, &bands);
        assert!(g_far.abs() < 0.7, "100 Hz should be ≈ untouched, got {g_far:.2}");
    }

    #[test]
    fn hpf_blocks_low_keeps_high() {
        let bands = vec![EqBand {
            kind: BandKind::Hpf,
            freq: 1000.0,
            gain_db: 0.0,
            q: 0.707,
            enabled: true,
        }];
        let low = sine_gain_db(100.0, &bands);
        assert!(low < -20.0, "100 Hz through HPF@1k should be ≤ -20 dB, got {low:.2}");
        let high = sine_gain_db(5000.0, &bands);
        assert!(high.abs() < 0.5, "5 kHz should pass, got {high:.2}");
    }

    #[test]
    fn lpf_blocks_high_keeps_low() {
        let bands = vec![EqBand {
            kind: BandKind::Lpf,
            freq: 1000.0,
            gain_db: 0.0,
            q: 0.707,
            enabled: true,
        }];
        let high = sine_gain_db(8000.0, &bands);
        assert!(high < -20.0, "8 kHz through LPF@1k should be ≤ -20 dB, got {high:.2}");
        let low = sine_gain_db(100.0, &bands);
        assert!(low.abs() < 0.5, "100 Hz should pass, got {low:.2}");
    }

    #[test]
    fn shelf_boosts_one_side() {
        let low_shelf = vec![EqBand {
            kind: BandKind::LowShelf,
            freq: 200.0,
            gain_db: 6.0,
            q: 0.707,
            enabled: true,
        }];
        let g_low = sine_gain_db(50.0, &low_shelf);
        assert!((g_low - 6.0).abs() < 0.6, "50 Hz shelf ≈ +6 dB, got {g_low:.2}");
        let g_high = sine_gain_db(4000.0, &low_shelf);
        assert!(g_high.abs() < 0.7, "4 kHz ≈ untouched, got {g_high:.2}");

        let high_shelf = vec![EqBand {
            kind: BandKind::HighShelf,
            freq: 4000.0,
            gain_db: 6.0,
            q: 0.707,
            enabled: true,
        }];
        let g_high = sine_gain_db(12000.0, &high_shelf);
        assert!((g_high - 6.0).abs() < 0.6, "12 kHz shelf ≈ +6 dB, got {g_high:.2}");
    }

    #[test]
    fn disabled_and_invalid_bands_are_dropped() {
        let bands = vec![
            EqBand { kind: BandKind::Peaking, freq: 1000.0, gain_db: 6.0, q: 1.0, enabled: false },
            EqBand { kind: BandKind::Peaking, freq: 0.0, gain_db: 6.0, q: 1.0, enabled: true },
            EqBand { kind: BandKind::Peaking, freq: 1000.0, gain_db: 6.0, q: -1.0, enabled: true },
            EqBand { kind: BandKind::Peaking, freq: 1000.0, gain_db: 6.0, q: 1.0, enabled: true },
        ];
        assert_eq!(design_chain(SR, &bands).len(), 1, "only the valid enabled band survives");
        assert_eq!(apply_eq(&mut vec![0.0f32; 8], None, SR, &bands), 1);
    }

    #[test]
    fn empty_chain_is_identity() {
        let mut x: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let expect = x.clone();
        apply_eq(&mut x, None, SR, &[]);
        assert_eq!(x, expect);
    }

    #[test]
    fn link_stereo_processes_both_planes() {
        let bands = vec![EqBand {
            kind: BandKind::Peaking,
            freq: 1000.0,
            gain_db: 12.0,
            q: 1.0,
            enabled: true,
        }];
        let mk = || -> Vec<f32> {
            (0..4410).map(|i| (2.0 * std::f64::consts::PI * 1000.0 * i as f64 / SR).sin() as f32 * 0.5)
                .collect()
        };
        let (mut l, mut r) = (mk(), mk());
        apply_eq(&mut l, Some(&mut r), SR, &bands);
        // Same input → identical output (link-stereo, no image smear).
        let l_rms: f64 = l.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / l.len() as f64;
        let r_rms: f64 = r.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / r.len() as f64;
        assert!((l_rms - r_rms).abs() < 1e-12);
        assert!(l_rms.sqrt() > 0.9, "1 kHz sine boosted to ≈ 1.0 peak, rms {}", l_rms.sqrt());
    }

    #[test]
    fn band_kind_tags_round_trip() {
        for tag in ["hpf", "lpf", "peaking", "lowShelf", "highShelf"] {
            assert!(BandKind::from_tag(tag).is_some(), "{tag} parses");
        }
        assert!(BandKind::from_tag("notch").is_none());
    }
}
