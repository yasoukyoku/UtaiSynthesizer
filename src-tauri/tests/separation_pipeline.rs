use std::path::PathBuf;
use utai_lib::inference::engine::OnnxEngine;
use utai_lib::separation::pipeline::{self, NativePipeline};
use utai_lib::separation::stft::{self, StftConfig};

fn bs_roformer_onnx_path() -> PathBuf {
    PathBuf::from(r"D:\MyDev\TESTING\MSST\bs_roformer_vocals.onnx")
}

fn test_wav_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("converter")
        .join("test_output")
        .join("test_stereo.wav")
}

fn generate_test_wav(path: &PathBuf) {
    if path.exists() {
        return;
    }
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();

    let sr = 44100u32;
    let duration = 3.0f32;
    let n = (sr as f32 * duration) as usize;

    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: sr,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec).unwrap();

    for i in 0..n {
        let t = i as f32 / sr as f32;
        let left = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5;
        let right = (2.0 * std::f32::consts::PI * 554.37 * t).sin() * 0.5;
        writer.write_sample(left).unwrap();
        writer.write_sample(right).unwrap();
    }
    writer.finalize().unwrap();
}

#[test]
fn test_stft_roundtrip_quality() {
    let config = StftConfig {
        n_fft: 2048,
        hop_length: 512,
        win_length: 2048,
    };

    let sr = 44100.0f32;
    let n = (sr * 1.0) as usize;

    let signal: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / sr;
            (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.7
                + (2.0 * std::f32::consts::PI * 1000.0 * t).sin() * 0.3
        })
        .collect();

    let spec = stft::stft(&signal, &config);
    let reconstructed = stft::istft(&spec, &config, signal.len());

    assert_eq!(reconstructed.len(), signal.len());

    let max_err: f32 = signal
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let rms_err: f32 = (signal
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / signal.len() as f32)
        .sqrt();

    eprintln!(
        "STFT roundtrip (1s, 2048-fft): max_err={:.6e}, rms_err={:.6e}",
        max_err, rms_err
    );
    assert!(max_err < 1e-4, "Max error too large: {}", max_err);
    assert!(rms_err < 1e-5, "RMS error too large: {}", rms_err);
}

#[test]
fn test_load_wav() {
    let path = test_wav_path();
    generate_test_wav(&path);

    let audio = pipeline::load_wav(&path).unwrap();
    assert_eq!(audio.channels, 2);
    assert_eq!(audio.sample_rate, 44100);
    assert!(audio.left.len() > 0);
    assert_eq!(audio.left.len(), audio.right.len());

    let expected_samples = (44100.0 * 3.0) as usize;
    assert_eq!(audio.left.len(), expected_samples);
    eprintln!("Loaded WAV: {} channels, {} Hz, {} samples", audio.channels, audio.sample_rate, audio.left.len());
}

#[test]
fn test_bs_roformer_native_pipeline() {
    let model_path = bs_roformer_onnx_path();
    if !model_path.exists() {
        eprintln!(
            "Skipping BSRoformer test: {} not found",
            model_path.display()
        );
        return;
    }

    let wav_path = test_wav_path();
    generate_test_wav(&wav_path);

    let engine = OnnxEngine::new();
    let pipe = NativePipeline::new(&engine, &model_path).expect("Failed to create pipeline");

    let config = pipe.config();
    eprintln!(
        "BSRoformer config: type={}, sr={}, stereo={}, stems={}, fft={}, hop={}",
        config.model_type,
        config.sample_rate,
        config.stereo,
        config.num_stems,
        config.n_fft,
        config.hop_length
    );

    let audio = pipeline::load_wav(&wav_path).unwrap();

    let progress_count = std::sync::atomic::AtomicU32::new(0);
    let stems = pipe
        .separate(&audio, &|_p| {
            progress_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            true
        })
        .expect("Separation failed");

    assert!(progress_count.load(std::sync::atomic::Ordering::Relaxed) > 0, "Progress callback was never called");

    // num_stems=1 model outputs vocals + residual instrumental
    assert_eq!(stems.len(), 2, "Expected 2 stems (vocals + instrumental)");
    assert_eq!(stems[0].label, "vocals");
    assert_eq!(stems[1].label, "instrumental");
    assert_eq!(stems[0].left.len(), audio.left.len());
    assert_eq!(stems[1].left.len(), audio.left.len());

    for stem in &stems {
        assert!(
            stem.left.iter().all(|x| !x.is_nan()),
            "{} left has NaN",
            stem.label
        );
        let max_val = stem.left.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        eprintln!(
            "Stem '{}': {} samples, peak={:.4}",
            stem.label,
            stem.left.len(),
            max_val
        );
    }

    // Save output stems for manual inspection
    let output_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("converter")
        .join("test_output");
    for stem in &stems {
        let path = output_dir.join(format!("bsroformer_{}.wav", stem.label));
        pipeline::save_wav(&path, stem, config.sample_rate).unwrap();
        eprintln!("Saved: {}", path.display());
    }

    pipe.unload();
}
