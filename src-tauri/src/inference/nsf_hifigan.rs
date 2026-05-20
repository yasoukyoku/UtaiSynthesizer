use ndarray::Array2;

use super::engine::{InputTensor, OnnxEngine};
use super::SynthesisResult;
use crate::Result;

const NSF_HIFIGAN_SAMPLE_RATE: u32 = 44100;
const HOP_SIZE: usize = 512;
const N_MEL: usize = 128;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum VocoderMode {
    PitchShift { semitones: f32 },
    FormantShift { ratio: f32 },
    AudioEnhance,
}

pub fn synthesize(
    engine: &OnnxEngine,
    session_id: &str,
    mel: &Array2<f32>,
    f0: &[f32],
    mode: &VocoderMode,
) -> Result<SynthesisResult> {
    let f0_processed: Vec<f32> = match mode {
        VocoderMode::PitchShift { semitones } => {
            let ratio = 2.0f32.powf(*semitones / 12.0);
            f0.iter().map(|&x| if x > 0.0 { x * ratio } else { 0.0 }).collect()
        }
        VocoderMode::FormantShift { .. } => f0.to_vec(),
        VocoderMode::AudioEnhance => f0.to_vec(),
    };

    let n_mel = mel.shape()[1] as i64;
    let t = mel.shape()[0] as i64;

    // mel: [1, n_mel, T]
    let mel_transposed: Vec<f32> = {
        let mut data = vec![0.0f32; (n_mel * t) as usize];
        for frame in 0..t as usize {
            for bin in 0..n_mel as usize {
                data[bin * t as usize + frame] = mel[[frame, bin]];
            }
        }
        data
    };

    let mel_input = InputTensor::F32 {
        data: mel_transposed,
        shape: vec![1, n_mel, t],
    };

    let f0_input = InputTensor::F32 {
        data: f0_processed,
        shape: vec![1, t],
    };

    let outputs = engine.run(session_id, vec![("mel", mel_input), ("f0", f0_input)])?;
    let audio = outputs.into_iter().next().unwrap_or_default();

    Ok(SynthesisResult {
        audio,
        sample_rate: NSF_HIFIGAN_SAMPLE_RATE,
    })
}

pub fn audio_to_mel(audio: &[f32], _sample_rate: u32) -> Result<Array2<f32>> {
    // Simplified mel spectrogram — placeholder until proper STFT + mel filterbank
    let hop = HOP_SIZE;
    let n_frames = audio.len() / hop;

    let mut mel = Array2::zeros((n_frames, N_MEL));
    for frame_idx in 0..n_frames {
        let start = frame_idx * hop;
        let end = (start + hop * 4).min(audio.len());
        let frame = &audio[start..end];

        let energy = frame.iter().map(|x| x * x).sum::<f32>() / frame.len() as f32;
        let log_energy = (energy.max(1e-10)).log10() * 10.0;

        for mel_bin in 0..N_MEL {
            mel[[frame_idx, mel_bin]] = log_energy * ((mel_bin as f32 + 1.0) / N_MEL as f32);
        }
    }

    Ok(mel)
}
