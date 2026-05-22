use std::path::PathBuf;
use std::sync::Arc;
use tauri::State;

use crate::audio::effects::Effect;
use crate::AppState;

#[derive(serde::Serialize)]
pub struct AudioFileInfo {
    pub duration_ms: f64,
    pub sample_rate: u32,
    pub channels: u16,
    pub peaks: Vec<f32>,
    /// Path to a WAV-format copy for playback (same as input if already WAV).
    /// Ensures browser decodeAudioData and Rust peaks use identical sample data.
    pub playback_path: String,
}

#[derive(serde::Deserialize)]
pub struct EffectRequest {
    pub audio_path: String,
    pub effects: Vec<Effect>,
    pub output_path: Option<String>,
}

#[derive(serde::Serialize)]
pub struct EffectResult {
    pub output_path: String,
    pub duration_secs: f64,
}

#[tauri::command]
pub async fn load_audio_file(
    state: State<'_, Arc<AppState>>,
    path: String,
) -> Result<AudioFileInfo, String> {
    let input_path = PathBuf::from(&path);
    let mut buf = crate::audio::load_audio(&input_path).map_err(|e| e.to_string())?;

    let ext = input_path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
    if ext != "wav" {
        buf = trim_codec_silence(buf);
    }

    let duration_ms = buf.duration_secs() * 1000.0;
    let peak_count = ((buf.duration_secs() * 200.0) as usize).clamp(4000, 64000);
    let peaks = extract_peaks(&buf.samples, buf.channels, peak_count);

    let playback_path = if ext == "wav" {
        path.clone()
    } else {
        let hash = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            path.hash(&mut h);
            h.finish()
        };
        let cache_dir = state.cache_dir.join("audio_cache");
        let _ = std::fs::create_dir_all(&cache_dir);
        let wav_path = cache_dir.join(format!("{:016x}.wav", hash));
        crate::audio::save_wav(&wav_path, &buf).map_err(|e| e.to_string())?;
        wav_path.to_string_lossy().to_string()
    };

    Ok(AudioFileInfo {
        duration_ms,
        sample_rate: buf.sample_rate,
        channels: buf.channels,
        peaks,
        playback_path,
    })
}

fn extract_peaks(samples: &[f32], channels: u16, target_count: usize) -> Vec<f32> {
    let frame_count = samples.len() / channels as usize;
    if frame_count == 0 {
        return vec![];
    }

    let count = target_count.min(frame_count);
    let frames_per_peak = frame_count as f64 / count as f64;
    let mut peaks = Vec::with_capacity(count);

    for i in 0..count {
        let start = (i as f64 * frames_per_peak) as usize;
        let end = (((i + 1) as f64) * frames_per_peak) as usize;
        let end = end.min(frame_count);

        let mut max_val = 0.0f32;
        for frame in start..end {
            let idx = frame * channels as usize;
            if idx < samples.len() {
                max_val = max_val.max(samples[idx].abs());
            }
        }
        peaks.push(max_val);
    }

    peaks
}

/// Trim leading near-zero samples from non-WAV audio.
/// Codec pre-skip (Opus, AAC) often leaves silence that ffmpeg doesn't fully strip.
/// Only trims if: amplitude < -74dB AND duration < 1s (avoids cutting artistic silence).
fn trim_codec_silence(mut buf: crate::audio::AudioBuffer) -> crate::audio::AudioBuffer {
    let channels = buf.channels as usize;
    let frame_count = buf.samples.len() / channels;
    if frame_count == 0 { return buf; }

    let threshold = 0.0002f32; // ~-74dB
    let max_trim_frames = buf.sample_rate as usize; // 1 second max

    let mut first_audible = 0usize;
    for frame in 0..frame_count.min(max_trim_frames) {
        let base = frame * channels;
        let mut peak = 0.0f32;
        for ch in 0..channels {
            peak = peak.max(buf.samples[base + ch].abs());
        }
        if peak > threshold {
            first_audible = frame;
            break;
        }
        if frame == frame_count.min(max_trim_frames) - 1 {
            return buf;
        }
    }

    if first_audible < 64 { return buf; }

    let trim_to = first_audible.saturating_sub(32);
    let trim_samples = trim_to * channels;
    buf.samples = buf.samples[trim_samples..].to_vec();
    tracing::debug!("Trimmed {} frames ({:.1}ms) of codec silence",
        trim_to, trim_to as f64 / buf.sample_rate as f64 * 1000.0);
    buf
}

#[tauri::command]
pub async fn process_effects(
    state: State<'_, Arc<AppState>>,
    request: EffectRequest,
) -> Result<EffectResult, String> {
    let mut buffer = crate::audio::load_audio(&PathBuf::from(&request.audio_path))
        .map_err(|e| e.to_string())?;

    let nsf_session: Option<&str> = None;

    for effect in &request.effects {
        buffer = crate::audio::effects::apply_effect(
            &buffer,
            effect,
            &state.inference.engine,
            nsf_session,
        )
        .map_err(|e| e.to_string())?;
    }

    let output_path = request.output_path.unwrap_or_else(|| {
        let input = PathBuf::from(&request.audio_path);
        let stem = input.file_stem().unwrap_or_default().to_string_lossy();
        input
            .with_file_name(format!("{}_processed.wav", stem))
            .to_string_lossy()
            .to_string()
    });

    crate::audio::save_wav(&PathBuf::from(&output_path), &buffer)
        .map_err(|e| e.to_string())?;

    Ok(EffectResult {
        output_path,
        duration_secs: buffer.duration_secs(),
    })
}

#[tauri::command]
pub async fn save_temp_audio(
    samples: Vec<f32>,
    sample_rate: u32,
    output_path: String,
) -> Result<String, String> {
    let buf = crate::audio::AudioBuffer {
        samples,
        sample_rate,
        channels: 1,
    };
    crate::audio::save_wav(&PathBuf::from(&output_path), &buf)
        .map_err(|e| e.to_string())?;
    Ok(output_path)
}

#[tauri::command]
pub async fn ensure_cache_dir(
    state: State<'_, Arc<AppState>>,
    segment_id: String,
) -> Result<String, String> {
    let cache_dir = state.cache_dir.join(&segment_id);
    std::fs::create_dir_all(&cache_dir).map_err(|e| e.to_string())?;
    Ok(cache_dir.to_string_lossy().to_string())
}

#[tauri::command]
pub async fn export_audio(
    state: State<'_, Arc<AppState>>,
    _output_path: String,
    _format: String,
    _sample_rate: u32,
    _normalize: bool,
) -> Result<(), String> {
    let proj = state.project.read();
    let _project = proj
        .as_ref()
        .ok_or_else(|| "No project open".to_string())?;

    Err("Export requires rendered tracks — pending full pipeline integration".to_string())
}
