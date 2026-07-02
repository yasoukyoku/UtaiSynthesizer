use std::path::PathBuf;
use std::sync::Arc;
use tauri::State;

use crate::audio::effects::Effect;
use crate::AppState;

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct AudioFileInfo {
    pub duration_ms: f64,
    pub sample_rate: u32,
    pub channels: u16,
    pub peaks: Vec<f32>,
    /// Path to a content-addressed WAV copy (in audio_cache) for playback. Ensures browser
    /// decodeAudioData and Rust peaks use identical sample data.
    pub playback_path: String,
    /// XXH3-64 hex of the original file's bytes — content identity (same content ⇒ same hash,
    /// regardless of path/name). Backend-internal: the decode cache is keyed by it (the frontend has
    /// no consumer — reserved for a future frontend-side dedup UI).
    #[serde(default)]
    pub content_hash: String,
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

/// Per-call unique suffix for the content-addressed WAV temp file (see load_audio_file).
static TMP_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[tauri::command]
pub async fn load_audio_file(
    state: State<'_, Arc<AppState>>,
    path: String,
) -> Result<AudioFileInfo, String> {
    let input_path = PathBuf::from(&path);

    // CONTENT IDENTITY: hash the raw file bytes (read once). Identical content under a different
    // path/name produces the same hash, so we cache decode results by CONTENT, not path — "stop
    // fighting file names". XXH3-64 is fast (~GB/s) and ample for dedup (not security).
    let bytes = std::fs::read(&input_path).map_err(|e| format!("read {}: {e}", input_path.display()))?;
    let content_hash = format!("{:016x}", xxhash_rust::xxh3::xxh3_64(&bytes));
    drop(bytes);

    let cache_dir = state.cache_dir.join("audio_cache");
    let _ = std::fs::create_dir_all(&cache_dir);
    let sidecar = cache_dir.join(format!("{content_hash}.json"));

    // CACHE HIT: same content was decoded before → return its metadata WITHOUT re-decoding. This is
    // the payoff: a re-import of the same audio (even renamed/moved) skips the expensive decode+peaks.
    if let Ok(text) = std::fs::read_to_string(&sidecar) {
        if let Ok(info) = serde_json::from_str::<AudioFileInfo>(&text) {
            if std::path::Path::new(&info.playback_path).exists() {
                return Ok(info);
            }
        }
    }

    // MISS: decode (load_audio re-reads by path — cheap after the hash read warms the OS page cache).
    let mut buf = crate::audio::load_audio(&input_path).map_err(|e| e.to_string())?;
    let ext = input_path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
    if ext != "wav" {
        buf = trim_codec_silence(buf);
    }

    let duration_ms = buf.duration_secs() * 1000.0;
    // 300 peaks/sec gives the waveform more horizontal detail. The ceiling stays high (the zoomed-in
    // per-pixel renderer draws from the full peaks, so long clips keep their detail; the cache
    // downsamples to its own column cap for the zoomed-out blit).
    let peak_count = ((buf.duration_secs() * 300.0) as usize).clamp(4000, 64000);
    let peaks = extract_peaks(&buf.samples, buf.channels, peak_count);

    // Content-addressed playback WAV (same content ⇒ one cache file, shared by all paths/names; also
    // means deleting the original doesn't break playback). For a WAV input, copy the bytes VERBATIM so
    // the original bit depth (24-bit / 32-bit float) is preserved — save_wav would requantize to 16-bit.
    // Non-WAV is decoded → 16-bit WAV (same as before content-addressing).
    //
    // Write via a UNIQUE temp + atomic rename, and skip if it already exists. Two decodes of the SAME
    // content can run CONCURRENTLY (e.g. a workflow run's collectOutputs + the live reconciler both
    // loading a freshly-rendered separation stem) — a direct copy/write to the shared `{hash}.wav` then
    // raced into "The process cannot access the file because it is being used by another process"
    // (os error 32) on Windows, which silently dropped a stem (e.g. instrumental) and fell the track back
    // to the original mix. Per-call temp + atomic publish removes the shared-destination contention.
    let wav_path = cache_dir.join(format!("{content_hash}.wav"));
    if !wav_path.exists() {
        let tmp = cache_dir.join(format!(
            "{content_hash}.{}.{}.tmp",
            std::process::id(),
            TMP_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        ));
        let write = if ext == "wav" {
            std::fs::copy(&input_path, &tmp).map(|_| ()).map_err(|e| format!("copy wav: {e}"))
        } else {
            crate::audio::save_wav(&tmp, &buf).map_err(|e| e.to_string())
        };
        if let Err(e) = write {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
        // Atomic publish. If a concurrent decode of the same content already published it, our rename
        // fails on Windows (it won't overwrite) — fine, the content is identical; drop our temp.
        if std::fs::rename(&tmp, &wav_path).is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
    }

    let info = AudioFileInfo {
        duration_ms,
        sample_rate: buf.sample_rate,
        channels: buf.channels,
        peaks,
        playback_path: wav_path.to_string_lossy().to_string(),
        content_hash,
    };
    // Persist the sidecar so the next import of this content short-circuits the whole decode.
    let _ = std::fs::write(&sidecar, serde_json::to_string(&info).unwrap_or_default());
    Ok(info)
}

/// Fast duration probe (milliseconds) without decoding samples — for the drag-import ghost
/// preview and loading-segment placeholder, where the file isn't loaded yet.
#[tauri::command]
pub async fn probe_audio_duration(path: String) -> Result<f64, String> {
    crate::audio::probe_duration_ms(&PathBuf::from(&path)).map_err(|e| e.to_string())
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

