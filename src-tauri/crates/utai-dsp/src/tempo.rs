//! tempo.rs — classical (non-ML) BPM + beat-grid detection for audio segments.
//!
//! Pipeline (S59, recipe distilled from Ellis 2007 "Beat Tracking by Dynamic Programming",
//! Percival & Tzanetakis 2014 "Streamlined Tempo Estimation", and the librosa 0.11 reference
//! implementation — all reimplemented here from the papers/docs, no GPL code involved):
//!
//!   mono f32 → onset strength envelope (STFT 2048/512 → Slaney mel 64 → dB → positive
//!   spectral flux → one-pole detrend) → autocorrelation × log-normal tempo prior (60–200 BPM,
//!   center 115) → octave disambiguation by P&T pulse-train rescoring ({T/2, 2T/3, T, 3T/2, 2T})
//!   → Ellis dynamic-programming beat phase (α=680 on the std-normalized envelope) → robust
//!   constant-grid fit (median IBI + least-squares regression + circular phase) → confidence +
//!   not_constant flag → downbeat by accent folding over the project meter (E_low/OSE/spectral
//!   difference, z-scored).
//!
//! The output is a CONSTANT grid (BPM + first-beat anchor), the DJ-software storage model —
//! deliberately: this app has a single global tempo scalar (no tempo map), and the target
//! material is steady-tempo pop/EDM/vocal mixes. Non-constant material is *reported* as such
//! (`not_constant` + low confidence) rather than given a fake authoritative grid.
//!
//! Everything is parameterized by the actual sample rate (frame rate = sr / HOP), so 44.1 k and
//! 48 k inputs both work without resampling.

use crate::stft::{StftConfig, StftProcessor};
use rustfft::{num_complex::Complex, FftPlanner};

const N_FFT: usize = 2048;
const HOP: usize = 512;
const N_MELS: usize = 64;
/// Analysis tempo range (BPM). Half/double candidates outside the range are discarded.
const BPM_MIN: f64 = 60.0;
const BPM_MAX: f64 = 200.0;
/// Log-normal tempo prior center/width (octaves) — Ellis §3.2 / librosa start_bpm & std_bpm.
const PRIOR_BPM: f64 = 115.0;
const PRIOR_STD_OCT: f64 = 1.0;
/// Ellis DP transition weight — §4.2: optimal 680, flat over ~20..2000.
const DP_ALPHA: f32 = 680.0;
/// Minimum analyzable window (shorter → TEMPO_TOO_SHORT).
const MIN_ANALYSIS_SECS: f64 = 5.0;
/// Minimum DP beats required to fit a grid.
const MIN_BEATS: usize = 4;

#[derive(Debug, Clone, PartialEq)]
pub enum TempoError {
    /// Input shorter than MIN_ANALYSIS_SECS.
    TooShort,
    /// Onset envelope is flat / no periodicity found (ambient, silence, speech...).
    NoBeat,
}

#[derive(Debug, Clone)]
pub struct TempoAnalysis {
    /// Grid tempo in BPM (regression-refined from the DP beat list).
    pub bpm: f64,
    /// First grid beat at/after the start of the analyzed audio, in ms (grid = anchor + k·period).
    pub grid_anchor_ms: f64,
    /// Which grid beat (0-based, counting from the anchor) is bar-beat 1 (downbeat).
    pub downbeat_index: u32,
    /// Downbeat fold margin ∈ [0,1] — small = the meter phase is a guess, expose a nudge UI.
    pub downbeat_margin: f32,
    /// Composite confidence ∈ [0,1] (phase concentration + tempo-peak saliency + IBI regularity
    /// + ACF↔regression agreement).
    pub confidence: f32,
    /// True when the material does not fit a constant grid (tempo drift / rubato / high residual).
    pub not_constant: bool,
    /// Alternative BPM readings (octave family of the winner, range-clamped, winner excluded) —
    /// the "BPM candidates" correction affordance.
    pub candidates: Vec<f64>,
    /// Raw DP beat times (ms) — diagnostics / future fine alignment.
    pub beats_ms: Vec<f64>,
}

/// Analyze a mono window. `beats_per_bar` comes from the project time signature (downbeat fold).
pub fn analyze_tempo(mono: &[f32], sr: u32, beats_per_bar: u32) -> Result<TempoAnalysis, TempoError> {
    let sr_f = sr as f64;
    if (mono.len() as f64) < MIN_ANALYSIS_SECS * sr_f {
        return Err(TempoError::TooShort);
    }
    let frame_rate = sr_f / HOP as f64;

    // ── (a) onset strength envelope + the mel dB matrix (kept for downbeat features) ──
    let (env, mel_db) = onset_envelope(mono, sr);
    let n_frames = env.len();
    if n_frames < 16 {
        return Err(TempoError::TooShort);
    }
    let env_std = std_ddof1(&env);
    if !(env_std > 1e-8) {
        return Err(TempoError::NoBeat); // flat envelope — silence / DC
    }
    let env_norm: Vec<f32> = env.iter().map(|v| v / env_std).collect();

    // ── (b) global tempo: harmonically-enhanced ACF × prior, then P&T pulse-train rescoring
    //        over the top ACF peaks ∪ the octave/dotted family of the best peak ──
    let (ranked, s_rel, tie) = estimate_bpm(&env, frame_rate).ok_or(TempoError::NoBeat)?;

    // ── (c) beat phase: Ellis DP on the pulse-ranked winner. (A "beat support referee" that
    //        re-ran the DP per metrical-level candidate and picked the best-supported one was
    //        tried and REVERTED: on busy real mixes even a wrong 3/2 level finds eighth-note
    //        energy under most of its beats, and the localscore RMS normalization varies with
    //        the candidate's kernel width — the measure is not comparable across candidates.
    //        The pulse-train max+var scoring is the more reliable level referee.) ──
    let bpm_pick = ranked[0].0;
    let track = ellis_dp(&env_norm, frame_rate, bpm_pick);
    if track.frames.len() < MIN_BEATS {
        return Err(TempoError::NoBeat);
    }
    let beat_support = track.support;

    // Strong-beat filter for the grid fit: the DP fills quiet intros/breakdowns with weak
    // period-spaced beats that wreck the constant-fit statistics of an otherwise steady song —
    // fit the grid on beats that carry real onset energy, fall back to all if too few survive.
    let strong_thr = {
        let mut s = track.strengths.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let top_half_mean =
            s[s.len() / 2..].iter().map(|v| *v as f64).sum::<f64>() / (s.len() - s.len() / 2) as f64;
        (0.3 * top_half_mean) as f32
    };
    let strong: Vec<f64> = track
        .frames
        .iter()
        .zip(&track.strengths)
        .filter(|(_, &st)| st >= strong_thr)
        .map(|(&f, _)| f as f64 * HOP as f64 / sr_f)
        .collect();
    let beats_sec: Vec<f64> = track.frames.iter().map(|&f| f as f64 * HOP as f64 / sr_f).collect();
    let fit_input: &[f64] =
        if strong.len() >= MIN_BEATS && strong.len() * 2 >= beats_sec.len() { &strong } else { &beats_sec };

    // ── (d) constant-grid fit + confidence ──
    let fit = fit_grid(fit_input).ok_or(TempoError::NoBeat)?;
    let agree = if (60.0 / fit.period - bpm_pick).abs() / bpm_pick <= 0.02 { 1.0f32 } else { 0.0 };
    let cv_term = (1.0 - fit.cv / 0.04).clamp(0.0, 1.0) as f32;
    let raw = (0.35 * fit.phase_r as f32 + 0.25 * s_rel + 0.2 * cv_term + 0.2 * agree).clamp(0.0, 1.0);
    // GATE by onset support: the DP *imposes* near-perfect periodicity even on noise (the beat
    // list is regular by construction), so beat-list regularity alone cannot certify a groove.
    // Only material whose beats actually capture envelope energy well above the floor may score
    // high. support ≈ 1 for structureless noise, ≥ 3–4 for anything with real onsets.
    let support_term = ((beat_support - 1.5) / 1.5).clamp(0.0, 1.0);
    let mut confidence = raw * (0.3 + 0.7 * support_term);
    if tie {
        confidence *= 0.7; // metrical-level vote was close — damp, but don't crush to 0
    }
    let not_constant = fit.drift > 0.01 || fit.cv > 0.04 || fit.resid_rms > 0.10 * fit.period;

    // Anchor = first grid beat at/after 0 (normalize the regression intercept into [0, period)).
    let anchor_sec = fit.t0.rem_euclid(fit.period);

    // ── (e) downbeat: accent folding over the meter on GRID beats ──
    let dur_sec = mono.len() as f64 / sr_f;
    // clamp matches the UI's max meter numerator (16) — a tighter clamp would fold the downbeat
    // over the wrong cycle in 13/8..16/x meters (audit)
    let l = beats_per_bar.clamp(2, 16) as usize;
    let (downbeat_index, downbeat_margin) =
        fold_downbeat(&env_norm, &mel_db, frame_rate, sr_f, anchor_sec, fit.period, dur_sec, l);

    // Octave/dotted-family candidates for the correction UI (winner excluded) — the Cubase
    // correction-button set: ×2, ÷2, 2/3, 3/2, 3/4, 4/3.
    let bpm = 60.0 / fit.period;
    let mut candidates: Vec<f64> = [0.5, 2.0 / 3.0, 0.75, 4.0 / 3.0, 1.5, 2.0]
        .iter()
        .map(|m| bpm * m)
        .filter(|b| (BPM_MIN..=BPM_MAX).contains(b))
        .collect();
    candidates.sort_by(|a, b| a.partial_cmp(b).unwrap());

    Ok(TempoAnalysis {
        bpm,
        grid_anchor_ms: anchor_sec * 1000.0,
        downbeat_index,
        downbeat_margin,
        confidence,
        not_constant,
        candidates,
        beats_ms: beats_sec.iter().map(|t| t * 1000.0).collect(),
    })
}

// ───────────────────────── (a) onset strength envelope ─────────────────────────

/// Returns (onset envelope [n_frames], mel dB matrix [N_MELS][n_frames]).
fn onset_envelope(mono: &[f32], sr: u32) -> (Vec<f32>, Vec<Vec<f32>>) {
    let proc = StftProcessor::new(StftConfig { n_fft: N_FFT, hop_length: HOP, win_length: N_FFT });
    let spec = proc.stft(mono); // [freq_bins, frames, 2]
    let n_frames = spec.shape()[1];

    // Slaney mel filterbank on the POWER spectrum.
    let fb = mel_filterbank(sr as f64, N_FFT, N_MELS);
    let mut mel = vec![vec![0.0f32; n_frames]; N_MELS];
    for t in 0..n_frames {
        for (b, filt) in fb.iter().enumerate() {
            let mut acc = 0.0f32;
            for &(k, w) in filt {
                let re = spec[[k, t, 0]];
                let im = spec[[k, t, 1]];
                acc += (re * re + im * im) * w;
            }
            mel[b][t] = acc;
        }
    }

    // power_to_db(ref=max, amin=1e-10, top_db=80) — librosa semantics.
    let mut ref_max = 1e-10f32;
    for band in &mel {
        for &v in band {
            ref_max = ref_max.max(v);
        }
    }
    let ref_db = 10.0 * ref_max.log10();
    for band in mel.iter_mut() {
        for v in band.iter_mut() {
            *v = 10.0 * v.max(1e-10).log10() - ref_db;
            *v = v.max(-80.0);
        }
    }

    // Positive spectral flux (band-mean of the half-wave-rectified first difference)...
    let mut flux = vec![0.0f32; n_frames];
    for t in 1..n_frames {
        let mut acc = 0.0f32;
        for band in mel.iter() {
            acc += (band[t] - band[t - 1]).max(0.0);
        }
        flux[t] = acc / N_MELS as f32;
    }
    // ...then a one-pole detrend (librosa lfilter([1,-1],[1,-0.99])): local zero-mean, may go
    // negative — negative values legitimately penalize placing beats in flat regions, don't clamp.
    let mut env = vec![0.0f32; n_frames];
    let mut prev_x = 0.0f32;
    let mut prev_y = 0.0f32;
    for t in 0..n_frames {
        let y = flux[t] - prev_x + 0.99 * prev_y;
        prev_x = flux[t];
        prev_y = y;
        env[t] = y;
    }
    (env, mel)
}

/// Slaney-style mel filterbank (librosa default): sparse per-band (bin, weight) lists,
/// triangles between n_mels+2 points uniform in Slaney mel, Slaney area normalization.
fn mel_filterbank(sr: f64, n_fft: usize, n_mels: usize) -> Vec<Vec<(usize, f32)>> {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
    let logstep: f64 = (6.4f64).ln() / 27.0;
    let hz_to_mel = |f: f64| if f < MIN_LOG_HZ { f / F_SP } else { MIN_LOG_MEL + (f / MIN_LOG_HZ).ln() / logstep };
    let mel_to_hz = |m: f64| if m < MIN_LOG_MEL { m * F_SP } else { MIN_LOG_HZ * (logstep * (m - MIN_LOG_MEL)).exp() };

    let mel_max = hz_to_mel(sr / 2.0);
    let pts: Vec<f64> = (0..n_mels + 2).map(|i| mel_to_hz(mel_max * i as f64 / (n_mels + 1) as f64)).collect();
    let bins = n_fft / 2 + 1;
    let bin_hz = sr / n_fft as f64;

    let mut fb = Vec::with_capacity(n_mels);
    for b in 0..n_mels {
        let (lo, c, hi) = (pts[b], pts[b + 1], pts[b + 2]);
        let enorm = 2.0 / (hi - lo);
        let mut filt = Vec::new();
        let k_lo = (lo / bin_hz).floor().max(0.0) as usize;
        let k_hi = ((hi / bin_hz).ceil() as usize).min(bins - 1);
        for k in k_lo..=k_hi {
            let f = k as f64 * bin_hz;
            let w = ((f - lo) / (c - lo).max(1e-9)).min((hi - f) / (hi - c).max(1e-9));
            if w > 0.0 {
                filt.push((k, (w * enorm) as f32));
            }
        }
        fb.push(filt);
    }
    fb
}

// ───────────────────────── (b) tempo estimation ─────────────────────────

/// FFT autocorrelation of the (detrended) envelope, positive lags.
fn autocorrelate(env: &[f32]) -> Vec<f32> {
    let n = env.len();
    let size = (2 * n).next_power_of_two();
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(size);
    let ifft = planner.plan_fft_inverse(size);
    let mut buf: Vec<Complex<f32>> = env.iter().map(|&v| Complex::new(v, 0.0)).collect();
    buf.resize(size, Complex::new(0.0, 0.0));
    fft.process(&mut buf);
    for v in buf.iter_mut() {
        *v = Complex::new(v.norm_sqr(), 0.0);
    }
    ifft.process(&mut buf);
    let norm = 1.0 / size as f32;
    (0..n).map(|i| buf[i].re * norm).collect()
}

/// Log-normal tempo prior weight for a BPM value.
fn prior_weight(bpm: f64) -> f64 {
    let d = (bpm / PRIOR_BPM).log2() / PRIOR_STD_OCT;
    (-0.5 * d * d).exp()
}

/// Global tempo estimate. Pipeline:
///   1. harmonically-enhanced ACF (P&T-style EA(τ) = A(τ) + ½·max A(2τ±1) + ¼·max A(4τ±2) —
///      boosts the true beat level, whose bar/half-bar multiples also carry energy, over the
///      dotted 4/3 level that plain ACF loves to lock on real pop mixes),
///   2. prior-weighted local maxima → top-8 peak candidates ∪ octave/dotted family of the best
///      ({1/2, 2/3, 3/4, 1, 4/3, 3/2, 2} — the Cubase correction-button set = the known failure
///      modes),
///   3. P&T pulse-train rescoring (max_φ + var_φ, each normalized across candidates) × prior,
///   4. refine each survivor by parabolic interpolation at its enhanced-ACF peak.
/// Returns (ranked (bpm, score) candidates — best first, saliency, tie_flag). The final
/// metrical-level decision happens in analyze_tempo via DP beat support.
fn estimate_bpm(env: &[f32], frame_rate: f64) -> Option<(Vec<(f64, f64)>, f32, bool)> {
    let acf = autocorrelate(env);
    let lag_min = (frame_rate * 60.0 / BPM_MAX).floor().max(2.0) as usize;
    let lag_max = ((frame_rate * 60.0 / BPM_MIN).ceil() as usize).min(acf.len().saturating_sub(2));
    if lag_min + 2 > lag_max {
        return None;
    }

    // harmonic enhancement + prior weighting
    let get = |l: isize| -> f64 {
        if l >= 1 && (l as usize) < acf.len() {
            acf[l as usize] as f64
        } else {
            0.0
        }
    };
    let enhanced = |l: usize| -> f64 {
        let l = l as isize;
        let h2 = (get(2 * l - 1)).max(get(2 * l)).max(get(2 * l + 1));
        let h4 = (-2..=2i8).map(|d| get(4 * l + d as isize)).fold(f64::MIN, f64::max);
        get(l) + 0.5 * h2 + 0.25 * h4
    };
    let weighted: Vec<f64> = (0..=lag_max + 1)
        .map(|l| if l < 2 { 0.0 } else { enhanced(l) * prior_weight(60.0 * frame_rate / l as f64) })
        .collect();

    // local maxima in range, top-8 by value
    let mut peaks: Vec<(usize, f64)> = Vec::new();
    for l in lag_min.max(2)..=lag_max {
        if weighted[l] > weighted[l - 1] && weighted[l] >= weighted[l + 1] {
            peaks.push((l, weighted[l]));
        }
    }
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let (best_lag, best_val) = *peaks.first()?;
    if !(best_val > 0.0) {
        return None;
    }

    // saliency: runner-up peak outside the winner's octave-kin neighborhoods (±4%)
    let kin: Vec<f64> =
        [1.0 / 3.0, 0.5, 2.0 / 3.0, 0.75, 1.0, 4.0 / 3.0, 1.5, 2.0, 3.0].iter().map(|m| best_lag as f64 * m).collect();
    let second = peaks
        .iter()
        .filter(|(l, _)| kin.iter().all(|k| (*l as f64 - k).abs() / k > 0.04))
        .map(|&(_, v)| v)
        .fold(0.0f64, f64::max);
    let s_rel = (1.0 - second / best_val).clamp(0.0, 1.0) as f32;

    // candidate set: top peaks ∪ octave/dotted family of the best peak
    let best_bpm = 60.0 * frame_rate / best_lag as f64;
    let mut cands: Vec<f64> = peaks.iter().take(8).map(|&(l, _)| 60.0 * frame_rate / l as f64).collect();
    for m in [0.5, 2.0 / 3.0, 0.75, 4.0 / 3.0, 1.5, 2.0] {
        cands.push(best_bpm * m);
    }
    cands.retain(|b| (BPM_MIN..=BPM_MAX).contains(b));
    cands.sort_by(|a, b| a.partial_cmp(b).unwrap());
    cands.dedup_by(|a, b| ((*a - *b) / *b).abs() < 0.015);
    if cands.is_empty() {
        return Some((vec![(best_bpm, 1.0)], s_rel, false));
    }

    // refine each candidate at its enhanced-ACF peak (±4% lag neighborhood), parabolic interp
    let refine = |bpm: f64| -> f64 {
        let lag_c = 60.0 * frame_rate / bpm;
        let lo = ((lag_c * 0.96).floor() as usize).clamp(2, weighted.len() - 2);
        let hi = ((lag_c * 1.04).ceil() as usize).clamp(lo, weighted.len() - 2);
        let mut ref_lag = lag_c;
        let mut ref_val = f64::MIN;
        for l in lo..=hi {
            if weighted[l] > ref_val {
                ref_val = weighted[l];
                ref_lag = l as f64;
            }
        }
        let l0 = ref_lag as usize;
        if l0 >= 1 && l0 + 1 < weighted.len() {
            let (ym1, y0, yp1) = (weighted[l0 - 1], weighted[l0], weighted[l0 + 1]);
            let denom = ym1 - 2.0 * y0 + yp1;
            if denom.abs() > 1e-12 {
                ref_lag = l0 as f64 + (0.5 * (ym1 - yp1) / denom).clamp(-0.5, 0.5);
            }
        }
        60.0 * frame_rate / ref_lag
    };

    // pulse-train rescoring: (normalized max_φ + normalized var_φ) × prior — P&T §II-B
    let pulses: Vec<(f64, f64)> = cands.iter().map(|&c| pulse_score(env, frame_rate, c)).collect();
    let sum_max: f64 = pulses.iter().map(|p| p.0).sum::<f64>().max(1e-12);
    let sum_var: f64 = pulses.iter().map(|p| p.1).sum::<f64>().max(1e-12);
    let mut sorted: Vec<(f64, f64)> = cands
        .iter()
        .zip(&pulses)
        .map(|(&c, &(mx, vr))| (refine(c), (mx / sum_max + vr / sum_var) * prior_weight(c)))
        .collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut tie = false;
    if sorted.len() > 1 && sorted[0].1 > 0.0 {
        if (sorted[0].1 - sorted[1].1) / sorted[0].1 < 0.10 {
            tie = true;
            let in_band = |b: f64| (90.0..=180.0).contains(&b);
            if !in_band(sorted[0].0) && in_band(sorted[1].0) {
                sorted.swap(0, 1);
            }
        }
    }
    Some((sorted, s_rel, tie))
}

/// P&T 2014-style pulse-train score for a candidate BPM: cross-correlate a pulse comb (period P
/// weight 1.0 ×4 pulses, period 2P weight 0.5, period 3P/2 weight 0.5) with the envelope over up
/// to 8 windows of ~6 s. Returns (mean-over-windows max_φ, mean-over-windows var_φ) — the best
/// phase correlation plus the variance across phases (P&T §II-B: a true beat level has a PEAKY
/// phase profile; a flat envelope or a wrong metrical level scores low variance). A doubled
/// tempo puts half its pulses on offbeats where the envelope has no energy.
fn pulse_score(env: &[f32], frame_rate: f64, bpm: f64) -> (f64, f64) {
    let p = frame_rate * 60.0 / bpm;
    let win = ((6.0 * frame_rate) as usize).min(env.len()).max((3.5 * p) as usize + 2);
    if win > env.len() || p < 2.0 {
        return (0.0, 0.0);
    }
    // pulse offsets (frames, fractional) + weights, relative to phase
    let mut pulses: Vec<(f64, f32)> = Vec::new();
    for k in 0..4 {
        pulses.push((k as f64 * p, 1.0));
    }
    for k in 0..2 {
        pulses.push((k as f64 * 2.0 * p, 0.5));
    }
    for k in 0..3 {
        pulses.push((k as f64 * 1.5 * p, 0.5));
    }

    let n_windows = 8.min(((env.len() - win) / win.max(1)) + 1).max(1);
    let mut total_max = 0.0f64;
    let mut total_var = 0.0f64;
    for w in 0..n_windows {
        let start = if n_windows == 1 { 0 } else { w * (env.len() - win) / (n_windows - 1).max(1) };
        let slice = &env[start..start + win];
        let p_int = p.ceil() as usize;
        let mut phase_scores = Vec::with_capacity(p_int);
        for phase in 0..p_int {
            let mut s = 0.0f64;
            for &(off, wt) in &pulses {
                let pos = (phase as f64 + off).round() as usize;
                if pos < slice.len() {
                    s += slice[pos] as f64 * wt as f64;
                }
            }
            phase_scores.push(s);
        }
        let best = phase_scores.iter().copied().fold(f64::MIN, f64::max);
        let mean = phase_scores.iter().sum::<f64>() / phase_scores.len() as f64;
        let var = phase_scores.iter().map(|s| (s - mean) * (s - mean)).sum::<f64>() / phase_scores.len() as f64;
        total_max += best;
        total_var += var;
    }
    ((total_max / n_windows as f64).max(0.0), (total_var / n_windows as f64).max(0.0))
}

// ───────────────────────── (c) Ellis DP beat phase ─────────────────────────

/// One DP beat-tracking result: frames + per-beat localscore strengths + the beat-support ratio
/// (mean localscore at the beats / RMS localscore — ≈1 for structureless noise, ≫1 for real
/// onsets; doubles as the metrical-level referee and the confidence gate).
struct BeatTrack {
    frames: Vec<usize>,
    strengths: Vec<f32>,
    support: f32,
}

/// Ellis 2007 dynamic-programming beat tracker (FMP-style recurrence with a fresh-start floor of
/// 0 so the first beat needs no predecessor). Input must be the STD-NORMALIZED envelope — the
/// α=680 calibration assumes it.
fn ellis_dp(env_norm: &[f32], frame_rate: f64, bpm: f64) -> BeatTrack {
    let n = env_norm.len();
    let tp = (frame_rate * 60.0 / bpm).round().max(2.0) as usize;

    // localscore: envelope convolved with a Gaussian of σ = τp/32 (librosa refinement — lets
    // onsets within ~τp/32 of a grid point attract the beat).
    let sigma = tp as f32 / 32.0;
    let radius = tp.min(n);
    let mut localscore = vec![0.0f32; n];
    let kernel: Vec<f32> = (-(radius as isize)..=radius as isize)
        .map(|d| (-0.5 * (d as f32 / sigma).powi(2)).exp())
        .collect();
    for t in 0..n {
        let mut acc = 0.0f32;
        for (i, &kw) in kernel.iter().enumerate() {
            let idx = t as isize + i as isize - radius as isize;
            if idx >= 0 && (idx as usize) < n {
                acc += env_norm[idx as usize] * kw;
            }
        }
        localscore[t] = acc;
    }

    let dmin = (tp / 2).max(1);
    let dmax = 2 * tp;
    let txcost: Vec<f32> =
        (0..=dmax).map(|d| if d < dmin { f32::MIN } else { -DP_ALPHA * ((d as f32 / tp as f32).ln()).powi(2) }).collect();

    let mut cumscore = vec![0.0f32; n];
    let mut backlink = vec![-1isize; n];
    for t in 0..n {
        let mut best = f32::MIN;
        let mut arg = -1isize;
        let lo = t.saturating_sub(dmax).max(0);
        let hi = t.saturating_sub(dmin);
        if t >= dmin {
            for prev in lo..=hi {
                let d = t - prev;
                let s = txcost[d] + cumscore[prev];
                if s > best {
                    best = s;
                    arg = prev as isize;
                }
            }
        }
        // fresh-start floor: a beat may start a new chain at no penalty (FMP C6S3 variant)
        if best > 0.0 {
            cumscore[t] = localscore[t] + best;
            backlink[t] = arg;
        } else {
            cumscore[t] = localscore[t];
            backlink[t] = -1;
        }
    }

    // endpoint: last local max of cumscore ≥ 0.5 × median(local-max values)
    let mut peaks: Vec<usize> = Vec::new();
    for t in 1..n.saturating_sub(1) {
        if cumscore[t] > cumscore[t - 1] && cumscore[t] >= cumscore[t + 1] {
            peaks.push(t);
        }
    }
    if peaks.is_empty() {
        return BeatTrack { frames: Vec::new(), strengths: Vec::new(), support: 0.0 };
    }
    let mut vals: Vec<f32> = peaks.iter().map(|&t| cumscore[t]).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = vals[vals.len() / 2];
    let end = *peaks.iter().rev().find(|&&t| cumscore[t] >= 0.5 * med).unwrap_or(peaks.last().unwrap());

    let mut beats = vec![end];
    while backlink[*beats.last().unwrap()] >= 0 {
        beats.push(backlink[*beats.last().unwrap()] as usize);
    }
    beats.reverse();

    // trim leading/trailing weak beats (below 0.5 × RMS(localscore))
    let rms = (localscore.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / n as f64).sqrt() as f32;
    let thr = 0.5 * rms;
    let first = beats.iter().position(|&b| localscore[b] >= thr).unwrap_or(0);
    let last = beats.iter().rposition(|&b| localscore[b] >= thr).unwrap_or(beats.len().saturating_sub(1));
    let beats = beats[first..=last.max(first)].to_vec();

    let strengths: Vec<f32> = beats.iter().map(|&b| localscore[b]).collect();
    let support = if beats.is_empty() || rms <= 1e-9 {
        0.0
    } else {
        let at_beats = strengths.iter().map(|&v| v as f64).sum::<f64>() / beats.len() as f64;
        (at_beats / rms as f64) as f32
    };
    BeatTrack { frames: beats, strengths, support }
}

// ───────────────────────── (d) constant-grid fit ─────────────────────────

struct GridFit {
    period: f64, // seconds per beat
    t0: f64,     // regression intercept (seconds; beat k sits at t0 + k·period)
    phase_r: f64,
    cv: f64,
    drift: f64,
    resid_rms: f64,
}

fn fit_grid(beats_sec: &[f64]) -> Option<GridFit> {
    if beats_sec.len() < MIN_BEATS {
        return None;
    }
    let mut ibis: Vec<f64> = beats_sec.windows(2).map(|w| w[1] - w[0]).collect();
    ibis.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let t_med = ibis[ibis.len() / 2];
    if !(t_med > 1e-4) {
        return None;
    }

    // GAP-AWARE pair test: the input may have holes (the strong-beat filter removes weak
    // DP-filled beats in intros/breakdowns), so an inter-beat interval spanning m grid periods
    // is fine as long as d/m ≈ the median period. A naive consecutive-IBI test here flagged
    // every filtered list as non-constant.
    let pair_ok = |a: f64, b: f64| {
        let d = b - a;
        let m = (d / t_med).round().max(1.0);
        (d / m - t_med).abs() <= 0.25 * t_med
    };
    let good: Vec<f64> = beats_sec
        .iter()
        .enumerate()
        .filter(|(i, _)| {
            let left = *i == 0 || pair_ok(beats_sec[i - 1], beats_sec[*i]);
            let right = *i == beats_sec.len() - 1 || pair_ok(beats_sec[*i], beats_sec[i + 1]);
            left && right
        })
        .map(|(_, &t)| t)
        .collect();
    if good.len() < MIN_BEATS {
        return None;
    }

    let regress = |ts: &[f64]| -> (f64, f64) {
        // integer beat indices via the median period (robust to skipped beats)
        let k: Vec<f64> = ts.iter().map(|t| ((t - ts[0]) / t_med).round()).collect();
        let n = ts.len() as f64;
        let km = k.iter().sum::<f64>() / n;
        let tm = ts.iter().sum::<f64>() / n;
        let mut num = 0.0;
        let mut den = 0.0;
        for (ki, ti) in k.iter().zip(ts.iter()) {
            num += (ki - km) * (ti - tm);
            den += (ki - km) * (ki - km);
        }
        let tau = if den > 0.0 { num / den } else { t_med };
        (tau, tm - tau * km)
    };
    let (mut tau, mut t0) = regress(&good);
    if !(tau > 1e-4) {
        return None;
    }

    // residuals on the fit; if poor, refit on the longest stable run (rubato intro case)
    let resid = |tau: f64, t0: f64| -> Vec<f64> {
        good.iter().map(|t| { let k = ((t - t0) / tau).round(); t - (t0 + k * tau) }).collect()
    };
    let mut r = resid(tau, t0);
    let rms = |r: &[f64]| (r.iter().map(|v| v * v).sum::<f64>() / r.len() as f64).sqrt();
    if rms(&r) > 0.10 * tau {
        let (mut best_s, mut best_e, mut s) = (0usize, 0usize, 0usize);
        for i in 0..r.len() {
            if r[i].abs() < 0.1 * tau {
                if i + 1 - s > best_e - best_s {
                    best_s = s;
                    best_e = i + 1;
                }
            } else {
                s = i + 1;
            }
        }
        if best_e - best_s >= MIN_BEATS {
            let (tau2, t02) = regress(&good[best_s..best_e]);
            if tau2 > 1e-4 {
                tau = tau2;
                t0 = t02;
                r = resid(tau, t0);
            }
        }
    }

    // drift: independent regressions on each half
    let half = good.len() / 2;
    let (tau_a, _) = regress(&good[..half.max(2)]);
    let (tau_b, _) = regress(&good[good.len() - half.max(2)..]);
    let drift = ((tau_a - tau_b) / tau).abs();

    // IBI regularity (gap-normalized: divide each interval by the integer period count it
    // spans) + circular phase concentration
    let good_ibis: Vec<f64> = good
        .windows(2)
        .map(|w| {
            let d = w[1] - w[0];
            d / (d / t_med).round().max(1.0)
        })
        .collect();
    let ibi_mean = good_ibis.iter().sum::<f64>() / good_ibis.len() as f64;
    let ibi_std =
        (good_ibis.iter().map(|v| (v - ibi_mean) * (v - ibi_mean)).sum::<f64>() / good_ibis.len() as f64).sqrt();
    let cv = ibi_std / t_med;
    let (mut sx, mut sy) = (0.0f64, 0.0f64);
    for t in &good {
        let th = 2.0 * std::f64::consts::PI * ((t / tau).fract());
        sx += th.cos();
        sy += th.sin();
    }
    let phase_r = (sx * sx + sy * sy).sqrt() / good.len() as f64;

    Some(GridFit { period: tau, t0, phase_r, cv, drift, resid_rms: rms(&r) })
}

// ───────────────────────── (e) downbeat folding ─────────────────────────

/// Score each meter phase p ∈ [0, L) by folding per-beat accent features (low-band energy, onset
/// strength, positive low-band spectral difference — all z-scored) over the grid. Returns
/// (best phase, margin). Non-ML downbeat tops out ~72% in the literature — the margin drives a
/// "this is a guess, nudge me" UI affordance.
#[allow(clippy::too_many_arguments)]
fn fold_downbeat(
    env_norm: &[f32],
    mel_db: &[Vec<f32>],
    frame_rate: f64,
    _sr: f64,
    anchor_sec: f64,
    period: f64,
    dur_sec: f64,
    l: usize,
) -> (u32, f32) {
    let n_frames = env_norm.len();
    // grid beats inside the analyzed window
    let mut beats: Vec<f64> = Vec::new();
    let mut t = anchor_sec;
    while t < dur_sec {
        beats.push(t);
        t += period;
    }
    if beats.len() < 2 * l {
        return (0, 0.0);
    }

    // low mel bands ≈ kick region: with Slaney-64 over 0..sr/2 the first bands cover <~200 Hz;
    // take the bottom 4 bands for E_low and the bottom half for the spectral difference.
    let low_bands = 4.min(mel_db.len());
    let half_bands = (mel_db.len() / 2).max(1);
    let frame_of = |sec: f64| ((sec * frame_rate).round() as usize).min(n_frames.saturating_sub(1));

    let band_mean = |b: usize, f: usize, rad: usize| -> f32 {
        let lo = f.saturating_sub(rad);
        let hi = (f + rad).min(n_frames - 1);
        let mut acc = 0.0f32;
        for t in lo..=hi {
            acc += mel_db[b][t];
        }
        acc / (hi - lo + 1) as f32
    };

    let nb = beats.len();
    let mut e_low = vec![0.0f32; nb];
    let mut onset = vec![0.0f32; nb];
    let mut sdiff = vec![0.0f32; nb];
    for (k, &bt) in beats.iter().enumerate() {
        let f = frame_of(bt);
        for b in 0..low_bands {
            e_low[k] += band_mean(b, f, 4);
        }
        e_low[k] /= low_bands as f32;
        let lo = f.saturating_sub(2);
        let hi = (f + 2).min(n_frames - 1);
        onset[k] = (lo..=hi).map(|t| env_norm[t]).fold(f32::MIN, f32::max);
        if k > 0 {
            let fp = frame_of(beats[k - 1]);
            for b in 0..half_bands {
                sdiff[k] += (band_mean(b, f, 2) - band_mean(b, fp, 2)).max(0.0);
            }
        }
    }
    if nb > 1 {
        sdiff[0] = sdiff[1..].iter().sum::<f32>() / (nb - 1) as f32; // neutral filler for k=0
    }

    let zscore = |v: &mut [f32]| {
        let n = v.len() as f32;
        let m = v.iter().sum::<f32>() / n;
        let sd = (v.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / n).sqrt().max(1e-9);
        for x in v.iter_mut() {
            *x = (*x - m) / sd;
        }
    };
    zscore(&mut e_low);
    zscore(&mut onset);
    zscore(&mut sdiff);

    let mut scores = vec![0.0f32; l];
    for k in 0..nb {
        scores[k % l] += 1.0 * e_low[k] + 0.5 * onset[k] + 1.0 * sdiff[k];
    }
    let mut order: Vec<usize> = (0..l).collect();
    order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
    let (s1, s2) = (scores[order[0]], scores[order[1]]);
    let margin = ((s1 - s2) / (s1.abs() + s2.abs() + 1e-9)).clamp(0.0, 1.0);
    (order[0] as u32, margin)
}

// ───────────────────────── helpers ─────────────────────────

fn std_ddof1(v: &[f32]) -> f32 {
    if v.len() < 2 {
        return 0.0;
    }
    let n = v.len() as f64;
    let m = v.iter().map(|&x| x as f64).sum::<f64>() / n;
    let var = v.iter().map(|&x| (x as f64 - m) * (x as f64 - m)).sum::<f64>() / (n - 1.0);
    var.sqrt() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 44100;

    /// Deterministic LCG noise in [-1, 1] (no rand dep; Date/random are banned in this codebase's
    /// test conventions anyway).
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        }
    }

    /// A percussive click: 10 ms noise burst with exponential decay (broadband → strong flux).
    fn add_click(buf: &mut [f32], at_sec: f64, amp: f32, rng: &mut Lcg) {
        let start = (at_sec * SR as f64) as usize;
        let len = (0.010 * SR as f64) as usize;
        for i in 0..len {
            if start + i < buf.len() {
                let decay = (-(i as f32) / (0.002 * SR as f32)).exp();
                buf[start + i] += amp * decay * rng.next();
            }
        }
    }

    /// Click track at `bpm`, optionally accenting every `accent_every`-th click (starting at
    /// `accent_phase`) with a louder, longer, low-heavy thump.
    fn click_track(bpm: f64, dur_sec: f64, first_beat_sec: f64, accent: Option<(usize, usize)>) -> Vec<f32> {
        let mut buf = vec![0.0f32; (dur_sec * SR as f64) as usize];
        let mut rng = Lcg(0x5EED);
        let period = 60.0 / bpm;
        let mut k = 0usize;
        let mut t = first_beat_sec;
        while t < dur_sec - 0.05 {
            let accented = accent.map(|(every, phase)| k % every == phase).unwrap_or(false);
            add_click(&mut buf, t, if accented { 0.9 } else { 0.4 }, &mut rng);
            if accented {
                // add a low sine thump (kick-ish) for the E_low feature
                let start = (t * SR as f64) as usize;
                let len = (0.060 * SR as f64) as usize;
                for i in 0..len {
                    if start + i < buf.len() {
                        let ph = 2.0 * std::f32::consts::PI * 60.0 * i as f32 / SR as f32;
                        let dec = (-(i as f32) / (0.02 * SR as f32)).exp();
                        buf[start + i] += 0.8 * dec * ph.sin();
                    }
                }
            }
            t += period;
            k += 1;
        }
        buf
    }

    #[test]
    fn steady_120_bpm() {
        let x = click_track(120.0, 25.0, 0.25, None);
        let a = analyze_tempo(&x, SR, 4).expect("should detect");
        assert!((a.bpm - 120.0).abs() / 120.0 < 0.01, "bpm={}", a.bpm);
        assert!(!a.not_constant, "steady clicks flagged not_constant");
        assert!(a.confidence > 0.6, "confidence={}", a.confidence);
        // grid anchor lands on a click (mod period; clicks start at 250 ms, period 500 ms)
        let period_ms = 60000.0 / a.bpm;
        let miss = (a.grid_anchor_ms - 250.0).rem_euclid(period_ms).min(
            (250.0 - a.grid_anchor_ms).rem_euclid(period_ms),
        );
        assert!(miss < 30.0, "anchor {} ms misses the click grid by {miss} ms", a.grid_anchor_ms);
    }

    #[test]
    fn slow_87_bpm_no_octave_error() {
        let x = click_track(87.0, 30.0, 0.1, None);
        let a = analyze_tempo(&x, SR, 4).expect("should detect");
        assert!((a.bpm - 87.0).abs() / 87.0 < 0.02, "bpm={} (octave error?)", a.bpm);
    }

    #[test]
    fn fast_174_bpm_stays_fast() {
        let x = click_track(174.0, 25.0, 0.2, None);
        let a = analyze_tempo(&x, SR, 4).expect("should detect");
        let fam_ok = [(174.0, 0.02), (87.0, 0.02)].iter().any(|(b, tol)| (a.bpm - b).abs() / b < *tol);
        assert!(fam_ok, "bpm={} not in the 87/174 family", a.bpm);
    }

    #[test]
    fn downbeat_accent_every_4() {
        // accents on click index 2, 6, 10, ... → the grid beat congruent to 2 (mod 4) is beat 1
        let x = click_track(120.0, 30.0, 0.25, Some((4, 2)));
        let a = analyze_tempo(&x, SR, 4).expect("should detect");
        assert!((a.bpm - 120.0).abs() / 120.0 < 0.01, "bpm={}", a.bpm);
        // map: grid beat k sits at anchor + k·period; click index 2 is at 0.25 + 2·0.5 = 1.25 s
        let period = 60.0 / a.bpm;
        let k_accent = ((1.25 - a.grid_anchor_ms / 1000.0) / period).round() as i64;
        let expected = k_accent.rem_euclid(4) as u32;
        assert_eq!(a.downbeat_index, expected, "downbeat phase wrong (margin={})", a.downbeat_margin);
        assert!(a.downbeat_margin > 0.05, "accented track should give a usable margin");
    }

    #[test]
    fn waltz_3_4() {
        let x = click_track(140.0, 30.0, 0.2, Some((3, 0)));
        let a = analyze_tempo(&x, SR, 3).expect("should detect");
        assert!((a.bpm - 140.0).abs() / 140.0 < 0.015, "bpm={}", a.bpm);
        let period = 60.0 / a.bpm;
        let k_accent = ((0.2 - a.grid_anchor_ms / 1000.0) / period).round() as i64;
        assert_eq!(a.downbeat_index, k_accent.rem_euclid(3) as u32);
    }

    #[test]
    fn accelerating_flags_not_constant() {
        // 120 → 138 BPM linear ramp over 30 s
        let mut buf = vec![0.0f32; (30.0 * SR as f64) as usize];
        let mut rng = Lcg(0xACCE1);
        let mut t = 0.2f64;
        while t < 29.8 {
            add_click(&mut buf, t, 0.4, &mut rng);
            let frac = t / 30.0;
            let bpm_now = 120.0 + 18.0 * frac;
            t += 60.0 / bpm_now;
        }
        let a = analyze_tempo(&buf, SR, 4).expect("should still produce a reading");
        assert!(a.not_constant, "7.5% ramp not flagged (cv={:.4})", a.confidence);
    }

    #[test]
    fn noise_only_low_confidence_or_nobeat() {
        let mut rng = Lcg(0xBEEF_5EED);
        let buf: Vec<f32> = (0..(20.0 * SR as f64) as usize).map(|_| 0.2 * rng.next()).collect();
        match analyze_tempo(&buf, SR, 4) {
            Err(TempoError::NoBeat) => {}
            Ok(a) => assert!(a.confidence < 0.5, "white noise got confidence {}", a.confidence),
            Err(e) => panic!("unexpected error {e:?}"),
        }
    }

    #[test]
    fn too_short_rejected() {
        let x = click_track(120.0, 3.0, 0.1, None);
        assert_eq!(analyze_tempo(&x, SR, 4).unwrap_err(), TempoError::TooShort);
    }

    #[test]
    fn swing_does_not_bait_1_5x() {
        // beat clicks + quieter swung 8ths at 2/3 of each beat: tempo must stay ~100, not 150
        let mut buf = vec![0.0f32; (30.0 * SR as f64) as usize];
        let mut rng = Lcg(0x517);
        let period = 60.0 / 100.0;
        let mut t = 0.3f64;
        while t < 29.5 {
            add_click(&mut buf, t, 0.5, &mut rng);
            add_click(&mut buf, t + period * 2.0 / 3.0, 0.18, &mut rng);
            t += period;
        }
        let a = analyze_tempo(&buf, SR, 4).expect("should detect");
        assert!((a.bpm - 100.0).abs() / 100.0 < 0.02, "swing pulled tempo to {}", a.bpm);
    }
}
