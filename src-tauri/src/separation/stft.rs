use ndarray::Array3;
use rustfft::{num_complex::Complex, FftPlanner};
use std::sync::Arc;
use rustfft::Fft;

pub struct StftConfig {
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
}

impl StftConfig {
    pub fn freq_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }
}

/// Reusable STFT processor — pre-computes window and FFT plan.
/// Thread-safe: can be shared across rayon tasks via Arc.
pub struct StftProcessor {
    config: StftConfig,
    window: Vec<f32>,
    fft_forward: Arc<dyn Fft<f32>>,
    fft_inverse: Arc<dyn Fft<f32>>,
    win_offset: usize,
}

impl StftProcessor {
    pub fn new(config: StftConfig) -> Self {
        let window = hann_window(config.win_length);
        let mut planner = FftPlanner::<f32>::new();
        let fft_forward = planner.plan_fft_forward(config.n_fft);
        let fft_inverse = planner.plan_fft_inverse(config.n_fft);
        let win_offset = (config.n_fft - config.win_length) / 2;
        Self { config, window, fft_forward, fft_inverse, win_offset }
    }

    pub fn config(&self) -> &StftConfig {
        &self.config
    }

    pub fn freq_bins(&self) -> usize {
        self.config.freq_bins()
    }

    /// STFT: mono signal [T] → complex spectrogram [freq_bins, frames, 2]
    pub fn stft(&self, signal: &[f32]) -> Array3<f32> {
        let freq_bins = self.freq_bins();
        let n_fft = self.config.n_fft;
        let hop = self.config.hop_length;

        let pad = n_fft / 2;
        let padded_len = pad + signal.len() + pad;
        let mut padded = vec![0.0f32; padded_len];
        padded[pad..pad + signal.len()].copy_from_slice(signal);
        for i in 0..pad.min(signal.len()) {
            padded[pad - 1 - i] = signal[i];
        }
        for i in 0..pad.min(signal.len()) {
            padded[pad + signal.len() + i] = signal[signal.len() - 1 - i];
        }

        let num_frames = if padded_len >= n_fft {
            (padded_len - n_fft) / hop + 1
        } else {
            0
        };

        let mut result = Array3::<f32>::zeros((freq_bins, num_frames, 2));
        let mut buffer = vec![Complex::new(0.0f32, 0.0f32); n_fft];

        for frame_idx in 0..num_frames {
            let start = frame_idx * hop;
            for b in buffer.iter_mut() {
                *b = Complex::new(0.0, 0.0);
            }
            for i in 0..self.config.win_length {
                let sample_idx = start + self.win_offset + i;
                if sample_idx < padded_len {
                    buffer[self.win_offset + i] = Complex::new(padded[sample_idx] * self.window[i], 0.0);
                }
            }
            self.fft_forward.process(&mut buffer);
            for bin in 0..freq_bins {
                result[[bin, frame_idx, 0]] = buffer[bin].re;
                result[[bin, frame_idx, 1]] = buffer[bin].im;
            }
        }
        result
    }

    /// iSTFT: complex spectrogram [freq_bins, frames, 2] → mono signal [T]
    pub fn istft(&self, spectrogram: &Array3<f32>, length: usize) -> Vec<f32> {
        let freq_bins = self.freq_bins();
        let n_fft = self.config.n_fft;
        let hop = self.config.hop_length;
        let num_frames = spectrogram.shape()[1];

        let pad = n_fft / 2;
        let output_len = (num_frames - 1) * hop + n_fft;
        let mut output = vec![0.0f32; output_len];
        let mut window_sum = vec![0.0f32; output_len];

        let mut buffer = vec![Complex::new(0.0f32, 0.0f32); n_fft];

        for frame_idx in 0..num_frames {
            for bin in 0..freq_bins {
                buffer[bin] = Complex::new(
                    spectrogram[[bin, frame_idx, 0]],
                    spectrogram[[bin, frame_idx, 1]],
                );
            }
            for bin in freq_bins..n_fft {
                let mirror = n_fft - bin;
                buffer[bin] = Complex::new(buffer[mirror].re, -buffer[mirror].im);
            }

            self.fft_inverse.process(&mut buffer);

            let norm = 1.0 / n_fft as f32;
            let start = frame_idx * hop;
            for i in 0..self.config.win_length {
                let pos = start + self.win_offset + i;
                if pos < output_len {
                    output[pos] += buffer[self.win_offset + i].re * norm * self.window[i];
                    window_sum[pos] += self.window[i] * self.window[i];
                }
            }
        }

        let tiny = 1e-8f32;
        for i in 0..output_len {
            if window_sum[i] > tiny {
                output[i] /= window_sum[i];
            }
        }

        let start = pad.min(output_len);
        let end = (start + length).min(output_len);
        output[start..end].to_vec()
    }

    /// Stereo STFT: [2, T] → [freq_bins * 2, frames, 2]
    /// Interleaves channels matching BSRoformer's expected layout.
    pub fn stft_stereo(&self, left: &[f32], right: &[f32]) -> Array3<f32> {
        let spec_l = self.stft(left);
        let spec_r = self.stft(right);

        let freq_bins = self.freq_bins();
        let num_frames = spec_l.shape()[1];

        let mut result = Array3::<f32>::zeros((freq_bins * 2, num_frames, 2));

        for f in 0..freq_bins {
            for t in 0..num_frames {
                result[[f * 2, t, 0]] = spec_l[[f, t, 0]];
                result[[f * 2, t, 1]] = spec_l[[f, t, 1]];
                result[[f * 2 + 1, t, 0]] = spec_r[[f, t, 0]];
                result[[f * 2 + 1, t, 1]] = spec_r[[f, t, 1]];
            }
        }
        result
    }

    /// Stereo iSTFT: [freq_bins * 2, frames, 2] → (left [T], right [T])
    pub fn istft_stereo(&self, spectrogram: &Array3<f32>, length: usize) -> (Vec<f32>, Vec<f32>) {
        let freq_bins = self.freq_bins();
        let num_frames = spectrogram.shape()[1];

        let mut spec_l = Array3::<f32>::zeros((freq_bins, num_frames, 2));
        let mut spec_r = Array3::<f32>::zeros((freq_bins, num_frames, 2));

        for f in 0..freq_bins {
            for t in 0..num_frames {
                spec_l[[f, t, 0]] = spectrogram[[f * 2, t, 0]];
                spec_l[[f, t, 1]] = spectrogram[[f * 2, t, 1]];
                spec_r[[f, t, 0]] = spectrogram[[f * 2 + 1, t, 0]];
                spec_r[[f, t, 1]] = spectrogram[[f * 2 + 1, t, 1]];
            }
        }

        let left = self.istft(&spec_l, length);
        let right = self.istft(&spec_r, length);
        (left, right)
    }
}

fn hann_window(length: usize) -> Vec<f32> {
    (0..length)
        .map(|i| {
            let phase = std::f32::consts::PI * 2.0 * i as f32 / length as f32;
            0.5 * (1.0 - phase.cos())
        })
        .collect()
}

/// Apply mask element-wise: result[f,t,ri] = stft[f,t,ri] * mask[f,t,ri]
/// BSRoformer/MelBandRoformer masks are NOT complex — each ri channel
/// is an independent scalar multiplier.
pub fn apply_complex_mask(stft_repr: &Array3<f32>, mask: &Array3<f32>) -> Array3<f32> {
    let shape = stft_repr.shape();
    let mut result = Array3::<f32>::zeros((shape[0], shape[1], 2));

    for f in 0..shape[0] {
        for t in 0..shape[1] {
            result[[f, t, 0]] = stft_repr[[f, t, 0]] * mask[[f, t, 0]];
            result[[f, t, 1]] = stft_repr[[f, t, 1]] * mask[[f, t, 1]];
        }
    }
    result
}

// ─── Legacy free-function API (used by existing code, delegates to StftProcessor) ──

pub fn stft(signal: &[f32], config: &StftConfig) -> Array3<f32> {
    let proc = StftProcessor::new(StftConfig {
        n_fft: config.n_fft,
        hop_length: config.hop_length,
        win_length: config.win_length,
    });
    proc.stft(signal)
}

pub fn istft(spectrogram: &Array3<f32>, config: &StftConfig, length: usize) -> Vec<f32> {
    let proc = StftProcessor::new(StftConfig {
        n_fft: config.n_fft,
        hop_length: config.hop_length,
        win_length: config.win_length,
    });
    proc.istft(spectrogram, length)
}

pub fn stft_stereo(left: &[f32], right: &[f32], config: &StftConfig) -> Array3<f32> {
    let proc = StftProcessor::new(StftConfig {
        n_fft: config.n_fft,
        hop_length: config.hop_length,
        win_length: config.win_length,
    });
    proc.stft_stereo(left, right)
}

pub fn istft_stereo(spectrogram: &Array3<f32>, config: &StftConfig, length: usize) -> (Vec<f32>, Vec<f32>) {
    let proc = StftProcessor::new(StftConfig {
        n_fft: config.n_fft,
        hop_length: config.hop_length,
        win_length: config.win_length,
    });
    proc.istft_stereo(spectrogram, length)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stft_istft_roundtrip() {
        let proc = StftProcessor::new(StftConfig {
            n_fft: 1024,
            hop_length: 256,
            win_length: 1024,
        });

        let sample_rate = 44100.0f32;
        let freq = 440.0f32;
        let duration = 0.1;
        let num_samples = (sample_rate * duration) as usize;
        let signal: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate).sin())
            .collect();

        let spec = proc.stft(&signal);
        let reconstructed = proc.istft(&spec, signal.len());

        assert_eq!(reconstructed.len(), signal.len());
        let max_err: f32 = signal
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-4, "STFT/iSTFT roundtrip error too large: {max_err}");
    }

    #[test]
    fn stereo_roundtrip() {
        let proc = StftProcessor::new(StftConfig {
            n_fft: 2048,
            hop_length: 512,
            win_length: 2048,
        });

        let sr = 44100.0f32;
        let n = (sr * 0.05) as usize;
        let left: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr).sin())
            .collect();
        let right: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 880.0 * i as f32 / sr).sin())
            .collect();

        let spec = proc.stft_stereo(&left, &right);
        assert_eq!(spec.shape()[0], proc.freq_bins() * 2);

        let (rec_l, rec_r) = proc.istft_stereo(&spec, n);

        let max_err_l: f32 = left.iter().zip(rec_l.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let max_err_r: f32 = right.iter().zip(rec_r.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);

        assert!(max_err_l < 1e-4, "Left channel roundtrip error: {max_err_l}");
        assert!(max_err_r < 1e-4, "Right channel roundtrip error: {max_err_r}");
    }

    #[test]
    fn identity_mask() {
        let proc = StftProcessor::new(StftConfig {
            n_fft: 512,
            hop_length: 128,
            win_length: 512,
        });

        let n = 2000;
        let signal: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();

        let spec = proc.stft(&signal);
        let freq_bins = spec.shape()[0];
        let frames = spec.shape()[1];

        // Identity mask: all 1.0 for both real and imag channels
        let mut mask = Array3::<f32>::zeros((freq_bins, frames, 2));
        for f in 0..freq_bins {
            for t in 0..frames {
                mask[[f, t, 0]] = 1.0;
                mask[[f, t, 1]] = 1.0;
            }
        }

        let masked = apply_complex_mask(&spec, &mask);
        let reconstructed = proc.istft(&masked, signal.len());

        let max_err: f32 = signal.iter().zip(reconstructed.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-4, "Identity mask roundtrip error: {max_err}");
    }
}
