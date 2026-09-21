//! Leading/trailing silence measurement.
//!
//! Distribution spec: head silence < 1 s, tail silence < 3 s. "Silence" =
//! RMS below `threshold_db` (default -60 dBFS) over a sliding window.

/// Default silence threshold: -60 dBFS.
pub const DEFAULT_SILENCE_DB: f64 = -60.0;

/// Returns (head_seconds, tail_seconds) of silence.
///
/// `samples` is mono (use a down-mix for stereo). `sample_rate` in Hz.
pub fn measure_silence(samples: &[f32], sample_rate: u32, threshold_db: f64) -> (f64, f64) {
    let threshold_linear = 10.0_f64.powf(threshold_db / 20.0);
    // Use 10 ms windows for a robust "is this region silent" decision.
    let win = (sample_rate as usize / 100).max(1);
    let n = samples.len();
    if n == 0 {
        return (0.0, 0.0);
    }

    let is_silent_window = |start: usize| -> bool {
        let end = (start + win).min(n);
        if end <= start {
            return true;
        }
        let mut sum_sq = 0.0f64;
        for &s in &samples[start..end] {
            sum_sq += (s as f64) * (s as f64);
        }
        let rms = (sum_sq / (end - start) as f64).sqrt();
        rms < threshold_linear
    };

    // Head: count leading silent windows.
    let mut head_samples = 0usize;
    while head_samples < n && is_silent_window(head_samples) {
        head_samples += win;
    }
    head_samples = head_samples.min(n);

    // Tail: count trailing silent windows (scanning backwards).
    let mut tail_samples = 0usize;
    while tail_samples < n {
        let start = n.saturating_sub(tail_samples + win);
        if start >= n || !is_silent_window(start) {
            break;
        }
        tail_samples += win;
        if start == 0 {
            break;
        }
    }
    tail_samples = tail_samples.min(n);

    (
        head_samples as f64 / sample_rate as f64,
        tail_samples as f64 / sample_rate as f64,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measures_silence_at_edges() {
        let sr = 44100u32;
        let head = sr as usize / 2; // 0.5 s silence
        let tail = sr as usize; // 1 s silence
        let total = head + sr as usize + tail;
        let mut s = vec![0.0f32; total];
        for i in head..head + sr as usize {
            s[i] = ((i - head) as f32 * 0.01).sin() * 0.5;
        }
        let (h, t) = measure_silence(&s, sr, DEFAULT_SILENCE_DB);
        assert!((h - 0.5).abs() < 0.02, "head {h} should be ~0.5s");
        assert!((t - 1.0).abs() < 0.02, "tail {t} should be ~1.0s");
    }

    #[test]
    fn no_silence_returns_near_zero() {
        let sr = 44100u32;
        let s: Vec<f32> = (0..sr as usize * 2).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        let (h, t) = measure_silence(&s, sr, DEFAULT_SILENCE_DB);
        assert!(h < 0.05, "head {h} should be near 0");
        assert!(t < 0.05, "tail {t} should be near 0");
    }

    #[test]
    fn all_silence_reports_full_duration() {
        let sr = 44100u32;
        let s = vec![0.0f32; sr as usize * 2];
        let (h, t) = measure_silence(&s, sr, DEFAULT_SILENCE_DB);
        // Head + tail together cover the whole (or nearly whole) buffer.
        assert!(h + t >= 1.9, "all-silent buffer should report ~2s total, got {h}+{t}");
    }
}
