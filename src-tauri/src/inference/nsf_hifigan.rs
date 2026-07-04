//! NSF-HiFiGAN vocoder call + the so-vits 4.1 enhancer — faithful ports of
//! `diffusion/vocoder.py Vocoder.infer` and `modules/enhancer.py Enhancer.enhance`
//! (original repo D:\MyDev\so-vits-svc\so-vits-svc). The vocoder ONNX itself is the aux
//! `nsf_hifigan.onnx` exported by converter/export_nsf_hifigan.py (mel [1,128,T] ln-mel +
//! f0 [1,T] Hz → audio [1,1,T*512]); its sidecar json + slaney filterbank .npy live next
//! to it in models/aux/.
//!
//! S36 REWRITE: the previous content of this file was opus4.6-era dead scaffolding — a
//! placeholder "mel" (frame energy × linear ramp) and a VocoderMode effects API that no
//! caller could ever reach (effects always passed session=None). Trust nothing from it.
//!
//! DOCUMENTED deviations from the original enhancer:
//!   - resampling is scipy-exact resample_poly (features::resample), not torchaudio sinc
//!     Resample(lowpass_filter_width=128) — the S35 house resampler decision; the E2E
//!     reference quantifies it with a REFPOLY variant.
//!   - the original's output-resample guard `if adaptive_factor != 0` is always true; we
//!     skip the resample when adaptive_sr == vocoder sr instead (behaviorally identical —
//!     torchaudio's equal-rate Resample is a passthrough).
//!   - 44.1 kHz models only: the original silently mislabels the output rate for non-44.1k
//!     models (enhance returns 44100, infer() discards it) — the command layer rejects
//!     that combination instead of reproducing the bug.

use ndarray::Array2;

use super::engine::{InputTensor, OnnxEngine};
use super::features::{np_interp, resample};
use super::mel::nsf_mel;
use crate::{Result, UtaiError};

/// Runtime facts of the aux vocoder (from models/aux/nsf_hifigan.json). All fields are
/// validated against the voice model's geometry by the command layer before use.
#[derive(Debug, Clone)]
pub struct VocoderConfig {
    pub sample_rate: u32, // 44100
    pub hop_size: usize,  // 512
    pub num_mels: usize,  // 128
}

/// `Vocoder.infer(mel, f0)` (diffusion/vocoder.py:41-44): f0 is TRUNCATED to the mel frame
/// count (never the other way), mel goes in as [1, num_mels, T]. Returns the raw audio
/// samples (T*hop) at the vocoder sample rate.
pub fn vocode(
    engine: &OnnxEngine,
    session_id: &str,
    mel: &Array2<f32>, // [num_mels, T] ln-mel (mel::nsf_mel layout)
    f0: &[f32],
) -> Result<Vec<f32>> {
    let (num_mels, t) = (mel.nrows(), mel.ncols());
    if f0.len() < t {
        return Err(UtaiError::Inference(format!(
            "声码器 f0 帧数不足：{} < mel 帧数 {}",
            f0.len(),
            t
        )));
    }
    let mel_data: Vec<f32> = mel.iter().copied().collect(); // row-major == [num_mels, T]
    let outputs = engine.run(
        session_id,
        vec![
            (
                "mel",
                InputTensor::F32 {
                    data: mel_data,
                    shape: vec![1, num_mels as i64, t as i64],
                },
            ),
            (
                "f0",
                InputTensor::F32 {
                    data: f0[..t].to_vec(),
                    shape: vec![1, t as i64],
                },
            ),
        ],
    )?;
    outputs
        .into_iter()
        .next()
        .ok_or_else(|| UtaiError::Inference("NSF-HiFiGAN 声码器没有返回输出".into()))
}

/// `Enhancer.enhance` (modules/enhancer.py:25-77), silence_front=0 specialization (the svc
/// caller never passes it). `audio` = the (still-padded) VITS output at `sample_rate`
/// (== the model sr; command layer guarantees it equals the vocoder's 44100), `f0` = the
/// hop-grid f0 the synthesizer used (post auto-f0), `hop_size` = the MAIN model's hop.
///
/// adaptive_key mechanism (enhancer.py:40-42): upsample the audio by 2^(key/12) but let
/// the vocoder treat it as 44.1k (i.e. run k semitones lower, inside its trained range),
/// scale f0 by the reciprocal factor, then time-compress back — a pitch-neutral round trip.
pub fn enhance(
    engine: &OnnxEngine,
    session_id: &str,
    mel_filters: &Array2<f32>,
    voc: &VocoderConfig,
    audio: &[f32],
    sample_rate: u32,
    f0: &[f32],
    hop_size: usize,
    adaptive_key: i32,
) -> Result<Vec<f32>> {
    let enhancer_sr = voc.sample_rate as f64;

    // adaptive_factor = 2^(-key/12); adaptive_sr = 100 * round(enhancer_sr/factor/100)
    let adaptive_factor = 2f64.powf(-(adaptive_key as f64) / 12.0);
    let adaptive_sr = (100.0 * (enhancer_sr / adaptive_factor / 100.0).round()) as u32;
    let real_factor = enhancer_sr / adaptive_sr as f64;

    // input resample (enhancer.py:45-51), skipped when rates match
    let audio_res: Vec<f32> = if sample_rate != adaptive_sr {
        resample(audio, sample_rate, adaptive_sr)
    } else {
        audio.to_vec()
    };

    // n_frames = len // hop + 1 (enhancer.py:53) — ≥ the mel frame count len//hop; the
    // generator call below truncates f0 to the mel count (enhancer.py:106 semantics via
    // vocode()'s f0[..t]).
    let n_frames_enh = audio_res.len() / voc.hop_size + 1;

    // f0 → enhancer hop grid via np.interp with endpoint clamp (enhancer.py:56-61); raw
    // f0 including 0 Hz unvoiced values is interpolated directly (original behavior).
    if f0.is_empty() {
        return Err(UtaiError::Inference("增强器输入 f0 为空".into()));
    }
    let f0_scaled: Vec<f64> = f0.iter().map(|&v| v as f64 * real_factor).collect();
    let time_org: Vec<f64> = (0..f0.len())
        .map(|i| hop_size as f64 / sample_rate as f64 * i as f64 / real_factor)
        .collect();
    let time_frame: Vec<f64> = (0..n_frames_enh)
        .map(|j| voc.hop_size as f64 / enhancer_sr * j as f64)
        .collect();
    let f0_res: Vec<f32> = np_interp(&time_frame, &time_org, &f0_scaled)
        .into_iter()
        .map(|v| v as f32)
        .collect();

    // mel (fresh STFT per call in the original — pure DSP here) + vocoder
    let mel = nsf_mel(&audio_res, mel_filters);
    let enhanced = vocode(engine, session_id, &mel, &f0_res)?;

    // output resample back adaptive_sr → enhancer_sr (enhancer.py:67-71); the original's
    // `if adaptive_factor != 0` is always-true — equal-rate skip is the identity case.
    let out = if adaptive_sr != voc.sample_rate {
        resample(&enhanced, adaptive_sr, voc.sample_rate)
    } else {
        enhanced
    };
    Ok(out)
}
