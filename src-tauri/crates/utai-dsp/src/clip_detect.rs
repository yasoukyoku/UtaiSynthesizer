//! Hard-clipping detection.
//!
//! Hard clipping = 3+ consecutive samples at or near full scale (|s| ≥ 0.999).
//! It produces odd-harmonic distortion (THD 1–10%) and is the single most
//! common reason distributors reject a master. Returns the list of clipped
//! (start, end) sample ranges.

/// Threshold for "at full scale". 0.999 ≈ -0.009 dBFS — samples above this
/// are effectively clamped by a following 16-bit/24-bit quantizer anyway.
pub const CLIP_THRESHOLD: f32 = 0.999;
/// Minimum consecutive samples to count as clipping (industry convention: ≥3).
pub const CLIP_MIN_RUN: usize = 3;

/// Returns `(start, end)` sample index pairs of clipping runs. `end` is
/// exclusive.
pub fn detect_clipping(samples: &[f32]) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut run_start: Option<usize> = None;
    for (i, &s) in samples.iter().enumerate() {
        let at_limit = s.abs() >= CLIP_THRESHOLD;
        match (at_limit, run_start) {
            (true, None) => run_start = Some(i),
            (false, Some(start)) => {
                let len = i - start;
                if len >= CLIP_MIN_RUN {
                    runs.push((start, i));
                }
                run_start = None;
            }
            _ => {}
        }
    }
    if let Some(start) = run_start {
        let len = samples.len() - start;
        if len >= CLIP_MIN_RUN {
            runs.push((start, samples.len()));
        }
    }
    runs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_clipping_run() {
        let mut s = vec![0.0f32; 100];
        for i in 10..15 {
            s[i] = 1.0;
        }
        let runs = detect_clipping(&s);
        assert_eq!(runs, vec![(10, 15)]);
    }

    #[test]
    fn ignores_short_runs() {
        let mut s = vec![0.0f32; 100];
        s[50] = 1.0;
        s[51] = 1.0;
        assert!(detect_clipping(&s).is_empty(), "2-sample run must not count");
    }

    #[test]
    fn handles_negative_clip() {
        let mut s = vec![0.0f32; 50];
        for i in 20..24 {
            s[i] = -1.0;
        }
        let runs = detect_clipping(&s);
        assert_eq!(runs, vec![(20, 24)]);
    }

    #[test]
    fn run_at_end_is_caught() {
        let mut s = vec![0.0f32; 30];
        for i in 27..30 {
            s[i] = 1.0;
        }
        let runs = detect_clipping(&s);
        assert_eq!(runs, vec![(27, 30)]);
    }

    #[test]
    fn clean_signal_has_no_clips() {
        let s: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.05).sin() * 0.8).collect();
        assert!(detect_clipping(&s).is_empty());
    }
}
