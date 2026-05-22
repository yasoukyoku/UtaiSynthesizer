pub mod effects;
pub mod export;
pub mod resample;

use std::path::Path;

use crate::Result;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AudioBuffer {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

impl AudioBuffer {
    pub fn new_mono(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            samples,
            sample_rate,
            channels: 1,
        }
    }

    pub fn duration_secs(&self) -> f64 {
        self.samples.len() as f64 / (self.sample_rate as f64 * self.channels as f64)
    }

    pub fn frame_count(&self) -> usize {
        self.samples.len() / self.channels as usize
    }
}

/// Three-tier audio loading:
///   1. hound — WAV only, fastest, zero overhead
///   2. symphonia — MP3/FLAC/OGG/Vorbis/AAC, pure Rust, no external deps
///   3. ffmpeg — everything else (Opus/WMA/APE/WavPack/...), bundled binary
pub fn load_audio(path: &Path) -> Result<AudioBuffer> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    if ext == "wav" {
        if let Ok(buf) = load_wav(path) {
            return Ok(buf);
        }
    }

    match load_with_symphonia(path) {
        Ok(buf) => return Ok(buf),
        Err(e) => {
            tracing::debug!("symphonia cannot decode '{}': {} — trying ffmpeg", path.display(), e);
        }
    }

    match load_with_ffmpeg(path) {
        Ok(buf) => Ok(buf),
        Err(e) => {
            tracing::error!("Audio load failed for '{}': {}", path.display(), e);
            Err(e)
        }
    }
}

pub fn load_wav(path: &Path) -> Result<AudioBuffer> {
    let reader = hound::WavReader::open(path)
        .map_err(|e| crate::UtaiError::Audio(format!("Failed to open WAV: {}", e)))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .filter_map(|s| s.ok())
            .collect(),
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_val)
                .collect()
        }
    };

    Ok(AudioBuffer {
        samples,
        sample_rate,
        channels,
    })
}

fn load_with_symphonia(path: &Path) -> Result<AudioBuffer> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path)
        .map_err(|e| crate::UtaiError::Audio(format!("Failed to open: {}", e)))?;

    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| crate::UtaiError::Audio(format!("Unsupported format: {}", e)))?;

    let mut format = probed.format;

    let track = format
        .default_track()
        .ok_or_else(|| crate::UtaiError::Audio("No audio track found".into()))?;

    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| crate::UtaiError::Audio("Unknown sample rate".into()))?;
    let channels = track
        .codec_params
        .channels
        .map(|c| c.count() as u16)
        .unwrap_or(2);
    let track_id = track.id;

    let codec_id = track.codec_params.codec;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| {
            let codec_name = symphonia::core::codecs::CODEC_TYPE_NULL;
            let hint = if codec_id == codec_name {
                "unknown codec"
            } else {
                "codec not supported by decoder (Opus in WebM requires ffmpeg)"
            };
            crate::UtaiError::Audio(format!("Cannot decode audio — {}: {}", hint, e))
        })?;

    let mut all_samples: Vec<f32> = Vec::new();
    let mut decode_errors = 0u32;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => {
                tracing::debug!("Packet read ended: {}", e);
                break;
            }
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(e) => {
                decode_errors += 1;
                if decode_errors <= 3 {
                    tracing::warn!("Decode error ({}): {}", decode_errors, e);
                }
                continue;
            }
        };

        let spec = *decoded.spec();
        let n_frames = decoded.capacity();
        let mut sample_buf = SampleBuffer::<f32>::new(n_frames as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);
        all_samples.extend_from_slice(sample_buf.samples());
    }

    if all_samples.is_empty() {
        return Err(crate::UtaiError::Audio(format!(
            "No audio data decoded from '{}' ({} decode errors). The audio codec may not be supported — try converting to WAV or MP3 first.",
            path.display(), decode_errors,
        )));
    }

    tracing::info!(
        "Loaded audio via symphonia: {} ({} ch, {}Hz, {:.1}s)",
        path.display(),
        channels,
        sample_rate,
        all_samples.len() as f64 / (sample_rate as f64 * channels as f64),
    );

    Ok(AudioBuffer {
        samples: all_samples,
        sample_rate,
        channels,
    })
}

#[cfg(windows)]
use std::os::windows::process::CommandExt;

fn load_with_ffmpeg(path: &Path) -> Result<AudioBuffer> {
    let ffmpeg = find_ffmpeg().ok_or_else(|| {
        crate::UtaiError::Audio(format!(
            "Cannot decode '{}' — symphonia does not support this format and ffmpeg was not found. \
             Place ffmpeg.exe next to the application binary.",
            path.display()
        ))
    })?;

    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let temp_path = std::env::temp_dir().join(format!(
        "utai_decode_{}_{}.wav",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed),
    ));

    let result = (|| -> Result<AudioBuffer> {
        let output = std::process::Command::new(&ffmpeg)
            .args(["-i"])
            .arg(path)
            .args(["-f", "wav", "-acodec", "pcm_f32le", "-v", "error", "-y"])
            .arg(&temp_path)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .creation_flags(0x08000000)
            .output()
            .map_err(|e| {
                crate::UtaiError::Audio(format!(
                    "Failed to run ffmpeg ({}): {}",
                    ffmpeg.display(),
                    e
                ))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(crate::UtaiError::Audio(format!(
                "ffmpeg failed for '{}': {}",
                path.display(),
                stderr.trim()
            )));
        }

        let buf = load_wav(&temp_path)?;

        tracing::info!(
            "Loaded via ffmpeg: {} ({} ch, {}Hz, {:.1}s)",
            path.display(),
            buf.channels,
            buf.sample_rate,
            buf.duration_secs(),
        );

        Ok(buf)
    })();

    let _ = std::fs::remove_file(&temp_path);
    result
}

fn find_ffmpeg() -> Option<std::path::PathBuf> {
    // Next to the running binary (bundled — release mode)
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let p = dir.join("ffmpeg.exe");
            if p.exists() {
                return Some(p);
            }
        }
    }

    // Dev mode: project root / tools
    let cwd = std::env::current_dir().unwrap_or_default();
    for base in [&cwd, cwd.parent().unwrap_or(&cwd)] {
        for sub in ["", "tools", "bin"] {
            let p = if sub.is_empty() {
                base.join("ffmpeg.exe")
            } else {
                base.join(sub).join("ffmpeg.exe")
            };
            if p.exists() {
                return Some(p);
            }
        }
    }

    // Dev fallback: system PATH
    if let Ok(output) = std::process::Command::new("where")
        .arg("ffmpeg.exe")
        .output()
    {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if let Some(line) = stdout.lines().next() {
                let p = std::path::PathBuf::from(line.trim());
                if p.exists() {
                    return Some(p);
                }
            }
        }
    }

    None
}

pub fn load_audio_at_rate(path: &Path, target_sr: u32) -> Result<AudioBuffer> {
    let ffmpeg = find_ffmpeg().ok_or_else(|| {
        crate::UtaiError::Audio("ffmpeg not found — needed for sample rate conversion".into())
    })?;

    use std::sync::atomic::{AtomicU64, Ordering};
    static CTR: AtomicU64 = AtomicU64::new(0);
    let temp_path = std::env::temp_dir().join(format!(
        "utai_resample_{}_{}.wav",
        std::process::id(),
        CTR.fetch_add(1, Ordering::Relaxed),
    ));

    let result = (|| -> Result<AudioBuffer> {
        let output = std::process::Command::new(&ffmpeg)
            .args(["-i"])
            .arg(path)
            .args(["-ar", &target_sr.to_string(), "-f", "wav", "-acodec", "pcm_f32le", "-v", "error", "-y"])
            .arg(&temp_path)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .creation_flags(0x08000000)
            .output()
            .map_err(|e| crate::UtaiError::Audio(format!("ffmpeg: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(crate::UtaiError::Audio(format!("ffmpeg resample failed: {}", stderr.trim())));
        }

        load_wav(&temp_path)
    })();

    let _ = std::fs::remove_file(&temp_path);
    result
}

pub fn save_wav(path: &Path, buffer: &AudioBuffer) -> Result<()> {
    let spec = hound::WavSpec {
        channels: buffer.channels,
        sample_rate: buffer.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| crate::UtaiError::Audio(format!("Failed to create WAV: {}", e)))?;

    for &sample in &buffer.samples {
        writer
            .write_sample((sample.clamp(-1.0, 1.0) * 32767.0) as i16)
            .map_err(|e| crate::UtaiError::Audio(format!("Write error: {}", e)))?;
    }

    writer
        .finalize()
        .map_err(|e| crate::UtaiError::Audio(format!("Finalize error: {}", e)))?;

    Ok(())
}
