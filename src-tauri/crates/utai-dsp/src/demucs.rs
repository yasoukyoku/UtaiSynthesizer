//! demucs `_spec` / `_ispec` STFT convention (HTDemucs).
//!
//! HTDemucs' spectral branch does NOT consume a plain centered STFT. demucs
//! (demucs4ht.py::_spec) reflect-pads the chunk by pad = hop/2*3 on the left and
//! pad + (le*hop - length) on the right (le = ceil(length/hop)) BEFORE the centered
//! STFT, then keeps frames [2 : 2+le] and drops the nyquist bin. Net effect: output
//! frame j is centered at j*hop + hop/2 — whereas a plain centered STFT centers frame
//! j at j*hop. This half-hop shift is what keeps the freq and time branches aligned
//! inside the model; feeding a plain STFT costs ~80 dB of stem SNR (measured 6.22 dB
//! vs 86.22 dB against the original torch pipeline).
//!
//! Scale invariant: demucs runs torch.stft(normalized=True), i.e. a 1/sqrt(n_fft)
//! factor. That factor is deliberately OMITTED here: the exported graph normalizes the
//! spec in-graph by its own mean/std and denormalizes its output, so a constant input
//! scale k scales mean/std by k and the model output by k — which our matching
//! unnormalized iSTFT then inverts exactly. Proven numerically (our spec == demucs
//! _spec × sqrt(n_fft) bit-exact; end-to-end 86.22 dB). Do NOT add the factor.

use ndarray::Array3;

use crate::reflect_pad_lr;
use crate::stft::StftProcessor;

/// demucs `_spec`: signal [T] → [n_fft/2 bins, ceil(T/hop) frames, 2] (nyquist dropped).
pub fn demucs_spec(proc: &StftProcessor, signal: &[f32]) -> Array3<f32> {
    let n_fft = proc.config().n_fft;
    let hop = proc.config().hop_length;
    debug_assert_eq!(n_fft, hop * 4, "demucs convention requires hop == n_fft/4");
    let t = signal.len();
    let le = (t + hop - 1) / hop;
    let pad = hop / 2 * 3;
    // Torch-exact asymmetric reflect pad (demucs pad1d); the extra le*hop - t on the
    // right rounds the signal up to a whole number of hops.
    let padded = reflect_pad_lr(signal, pad, pad + le * hop - t);
    // proc.stft additionally center-pads by n_fft/2 internally (stft.rs) — exactly like
    // torch.stft(center=True) inside demucs' spectro(). The demucs pad sits OUTSIDE that
    // center pad, so the padded signal yields exactly le + 4 frames.
    let full = proc.stft(&padded);
    debug_assert_eq!(full.shape()[1], le + 4);
    let bins = n_fft / 2; // drop the nyquist row
    let mut out = Array3::<f32>::zeros((bins, le, 2));
    for b in 0..bins {
        for fr in 0..le {
            out[[b, fr, 0]] = full[[b, fr + 2, 0]];
            out[[b, fr, 1]] = full[[b, fr + 2, 1]];
        }
    }
    out
}

/// demucs `_ispec`: spec [n_fft/2 bins, ceil(length/hop) frames, 2] → signal [length].
/// Re-adds a zero nyquist row + 2 zero frames per side (undoing the `[2 : 2+le]` crop),
/// runs the plain centered iSTFT over le*hop + 2*pad samples, then slices
/// [pad : pad+length]. The edge frames it re-inserts are zeros, so the outer pad region
/// is reconstructed from incomplete overlap-add — same as demucs; chunk overlap covers it.
pub fn demucs_ispec(proc: &StftProcessor, spec: &Array3<f32>, length: usize) -> Vec<f32> {
    let hop = proc.config().hop_length;
    let full_bins = proc.freq_bins(); // n_fft/2 + 1
    let bins = spec.shape()[0].min(full_bins);
    let frames = spec.shape()[1];
    debug_assert_eq!((length + hop - 1) / hop, frames, "frames must equal ceil(length/hop)");
    let pad = hop / 2 * 3;
    let mut padded = Array3::<f32>::zeros((full_bins, frames + 4, 2));
    for b in 0..bins {
        for fr in 0..frames {
            padded[[b, fr + 2, 0]] = spec[[b, fr, 0]];
            padded[[b, fr + 2, 1]] = spec[[b, fr, 1]];
        }
    }
    let le_len = hop * ((length + hop - 1) / hop) + 2 * pad;
    let x = proc.istft(&padded, le_len);
    let end = (pad + length).min(x.len());
    x[pad.min(end)..end].to_vec()
}

// ─── HTDemucs `_spec` convention tests ───────────────────────────
// These lock the demucs frame convention (frames centered at j*hop + hop/2, nyquist
// dropped, [2 : 2+le] crop) against future regressions. Verified against the torch
// reference numerically (86.22 dB end-to-end); the tests below are self-contained.
#[cfg(test)]
mod htdemucs_spec_tests {
    use super::*;
    use crate::stft::StftConfig;

    const N_FFT: usize = 4096;
    const HOP: usize = 1024;
    const SR: usize = 44100;

    fn make_proc() -> StftProcessor {
        StftProcessor::new(StftConfig {
            n_fft: N_FFT,
            hop_length: HOP,
            win_length: N_FFT,
        })
    }

    /// Deterministic test signal: sines + band-limited noise. The 2-tap averager on the
    /// noise nulls the response at nyquist, so the convention's dropped-nyquist-bin error
    /// stays negligible and the roundtrip bound below is meaningful.
    fn synth_signal(len: usize) -> Vec<f32> {
        let mut state = 0x1234_5678u32;
        let mut noise = Vec::with_capacity(len + 1);
        for _ in 0..=len {
            // xorshift32 — deterministic, no rand dependency
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            noise.push((state as f32 / u32::MAX as f32) - 0.5);
        }
        (0..len)
            .map(|i| {
                let t = i as f32 / SR as f32;
                let tau = 2.0 * std::f32::consts::PI;
                (tau * 440.0 * t).sin() * 0.4
                    + (tau * 1234.5 * t).sin() * 0.25
                    + (tau * 3210.0 * t).sin() * 0.15
                    + 0.05 * (noise[i] + noise[i + 1]) * 0.5
            })
            .collect()
    }

    #[test]
    fn reflect_pad_lr_matches_torch() {
        // torch.nn.functional.pad([0,1,2,3,4], (3,2), mode='reflect') == [3,2,1,0,1,2,3,4,3,2]
        let x = [0.0f32, 1.0, 2.0, 3.0, 4.0];
        assert_eq!(
            reflect_pad_lr(&x, 3, 2),
            vec![3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]
        );
    }

    /// Alignment property: demucs_spec frame j is centered at j*hop + hop/2, so it must
    /// equal frame j of a PLAIN centered STFT of the signal advanced by hop/2 samples
    /// (whose frame j is centered at j*hop in ITS coordinates = j*hop + hop/2 in ours) —
    /// sample-exactly on interior frames, where neither transform's window touches any
    /// padded region (the two paths then window bit-identical samples).
    #[test]
    fn demucs_spec_frame_alignment() {
        let proc = make_proc();
        let t = 3 * SR; // 3 s
        let x = synth_signal(t);
        let le = (t + HOP - 1) / HOP;

        let ds = demucs_spec(&proc, &x);
        assert_eq!(ds.shape(), &[N_FFT / 2, le, 2]);

        let advanced = &x[HOP / 2..];
        let plain = proc.stft(advanced);

        // Interior frames: window [j*hop + hop/2 - n_fft/2, j*hop + hop/2 + n_fft/2)
        // must lie inside [0, t - hop/2) so BOTH transforms window pure signal samples.
        let j_lo = (N_FFT / 2 - HOP / 2 + HOP - 1) / HOP; // ceil -> 2
        let j_hi = (t - HOP - N_FFT / 2) / HOP; // floor, inclusive
        assert!(j_hi > j_lo + 8, "test signal too short for interior frames");
        assert!(j_hi < le && j_hi < plain.shape()[1]);

        let mut max_diff = 0.0f32;
        for j in j_lo..=j_hi {
            for b in 0..N_FFT / 2 {
                for ri in 0..2 {
                    max_diff = max_diff.max((ds[[b, j, ri]] - plain[[b, j, ri]]).abs());
                }
            }
        }
        assert!(max_diff < 1e-6, "interior frame mismatch: {max_diff}");
    }

    /// demucs_spec → demucs_ispec must round-trip to identity on the interior. The first
    /// and last pad = 3*hop/2 samples are excluded: the `[2 : 2+le]` crop discards the
    /// analysis frames covering the outer reflect pad and _ispec re-inserts them as ZERO
    /// frames, so the pad region is reconstructed from incomplete overlap-add (exactly as
    /// in demucs — chunk overlap covers it in real use).
    #[test]
    fn demucs_spec_ispec_roundtrip() {
        let proc = make_proc();
        let t = 3 * SR;
        let x = synth_signal(t);
        let le = (t + HOP - 1) / HOP;

        let spec = demucs_spec(&proc, &x);
        let rec = demucs_ispec(&proc, &spec, t);
        assert_eq!(rec.len(), t);

        let pad = HOP / 2 * 3;
        let hi = (le * HOP - pad).min(t);
        let mut max_err = 0.0f32;
        for i in pad..hi {
            max_err = max_err.max((rec[i] - x[i]).abs());
        }
        assert!(max_err < 1e-4, "roundtrip interior error: {max_err}");
    }
}
