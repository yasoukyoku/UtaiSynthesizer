use std::path::PathBuf;
use std::sync::Arc;
use tauri::State;

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

/// Per-call unique suffix for the content-addressed WAV temp file (see load_audio_file).
static TMP_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[tauri::command]
pub async fn load_audio_file(
    state: State<'_, Arc<AppState>>,
    path: String,
) -> Result<AudioFileInfo, String> {
    let t0 = std::time::Instant::now();
    let input_path = PathBuf::from(&path);

    // CONTENT IDENTITY: hash the raw file bytes (read once). Identical content under a different
    // path/name produces the same hash, so we cache decode results by CONTENT, not path — "stop
    // fighting file names". XXH3-64 is fast (~GB/s) and ample for dedup (not security).
    let bytes = std::fs::read(&input_path).map_err(|e| format!("read {}: {e}", input_path.display()))?;
    let content_hash = format!("{:016x}", xxhash_rust::xxh3::xxh3_64(&bytes));
    drop(bytes);
    let hash_ms = t0.elapsed().as_millis();

    let cache_dir = state.cache_dir.join("audio_cache");
    let _ = std::fs::create_dir_all(&cache_dir);
    let sidecar = cache_dir.join(format!("{content_hash}.json"));

    // CACHE HIT: same content was decoded before → return its metadata WITHOUT re-decoding. This is
    // the payoff: a re-import of the same audio (even renamed/moved) skips the expensive decode+peaks.
    if let Ok(text) = std::fs::read_to_string(&sidecar) {
        if let Ok(info) = serde_json::from_str::<AudioFileInfo>(&text) {
            if std::path::Path::new(&info.playback_path).exists() {
                tracing::debug!("[perf] load_audio_file HIT {}ms hash={}ms {}", t0.elapsed().as_millis(), hash_ms, input_path.display());
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
    // S59 deposit-perf O4: a WAV that ALREADY lives inside our cache tree (separation stems in
    // run dirs, stretch artifacts — all under state.cache_dir) is its own playback copy: skip the
    // ~85MB fs::copy per stem, point playback_path at the file itself (S32's stem-deposit
    // bottleneck #2 — the copy had no consumer on the deposit path; playback uses the run-dir
    // path). If the run dir is later swept, the sidecar's exists() check above misses and we
    // simply re-decode. External user files still get the durable content-addressed copy.
    //
    // Write via a UNIQUE temp + atomic rename, and skip if it already exists. Two decodes of the SAME
    // content can run CONCURRENTLY (e.g. a workflow run finishing + the live reconciler both
    // loading a freshly-rendered separation stem) — a direct copy/write to the shared `{hash}.wav` then
    // raced into "The process cannot access the file because it is being used by another process"
    // (os error 32) on Windows, which silently dropped a stem (e.g. instrumental) and fell the track back
    // to the original mix. Per-call temp + atomic publish removes the shared-destination contention.
    let t_copy = std::time::Instant::now();
    // usp_work extractions are EXCLUDED from the skip: prune_usp_work deletes them mid-session /
    // on the next open, so a sidecar pinning playback_path there would silently mute the track
    // (audit MAJOR). Archive media keeps the durable {hash}.wav copy — the pre-S59 behavior; run
    // dirs and stretch artifacts (the actual S32 bottleneck) keep the optimization, their sweep
    // lifecycle just re-decodes on a stale sidecar.
    let in_cache_wav = ext == "wav"
        && input_path.starts_with(&state.cache_dir)
        && !input_path.starts_with(state.cache_dir.join("usp_work"));
    let wav_path = if in_cache_wav {
        input_path.clone()
    } else {
        cache_dir.join(format!("{content_hash}.wav"))
    };
    if !in_cache_wav && !wav_path.exists() {
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
    tracing::debug!(
        "[perf] load_audio_file MISS total={}ms hash={}ms copy={}ms in_cache={} {}",
        t0.elapsed().as_millis(), hash_ms, t_copy.elapsed().as_millis(), in_cache_wav, input_path.display()
    );
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

/// Fidelity pitch-shift (transpose) of a whole file — the workflow Transpose node's engine.
/// Spectral-domain via Signalsmith Stretch (time_factor = 1.0 → sample-exact same length),
/// formant/tonality-aware, polyphonic-safe (built for instrumentals; the vocal path transposes
/// model-side instead). Writes a 32-bit float WAV to `output_path` (a run-unique workflow cache
/// path — path change IS the re-render signal, so no content-addressing here). Errors are stable
/// CODEs per the i18n rule: TRANSPOSE_RANGE / TRANSPOSE_INPUT_MISSING / STRETCH_ENGINE_FAILED.
#[tauri::command]
pub async fn transpose_audio(
    path: String,
    semitones: f64,
    output_path: String,
) -> Result<StretchResult, String> {
    if !(semitones.is_finite() && (-24.0..=24.0).contains(&semitones)) {
        return Err("TRANSPOSE_RANGE".to_string());
    }
    tauri::async_runtime::spawn_blocking(move || {
        let input = PathBuf::from(&path);
        let buf = crate::audio::load_audio(&input).map_err(|e| format!("TRANSPOSE_INPUT_MISSING: {e}"))?;
        let shifted = utai_stretch::stretch_interleaved(
            &buf.samples,
            buf.channels.max(1) as usize,
            buf.sample_rate,
            1.0,
            semitones,
        )?;
        let out_buf = crate::audio::AudioBuffer {
            samples: shifted,
            sample_rate: buf.sample_rate,
            channels: buf.channels.max(1),
        };
        let out = PathBuf::from(&output_path);
        if let Some(parent) = out.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        crate::audio::save_wav_f32(&out, &out_buf).map_err(|e| format!("STRETCH_ENGINE_FAILED: {e}"))?;
        Ok(StretchResult {
            output_path,
            duration_ms: out_buf.duration_secs() * 1000.0,
            sample_rate: out_buf.sample_rate,
            channels: out_buf.channels,
        })
    })
    .await
    .map_err(|e| format!("STRETCH_JOIN: {e}"))?
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


/// Write user-chosen export bytes (e.g. the training loss-curve PNG) to a path picked
/// via the save dialog — the fs plugin scope doesn't cover arbitrary user paths.
#[tauri::command]
pub async fn save_binary_file(path: String, data: Vec<u8>) -> Result<(), String> {
    std::fs::write(&path, data).map_err(|e| format!("write {}: {}", path, e))
}

// ─── S59: segment BPM/beat-grid analysis + Tempo Slider time-stretch ───

#[derive(serde::Serialize, Clone)]
pub struct TempoAnalysisResult {
    /// Constant-grid tempo (regression-refined).
    pub bpm: f64,
    /// First grid beat in SOURCE-audio ms (window offset already folded back in).
    pub grid_anchor_ms: f64,
    /// Which grid beat (0-based from the anchor) is bar-beat 1.
    pub downbeat_index: u32,
    pub downbeat_margin: f32,
    pub confidence: f32,
    pub not_constant: bool,
    /// Alternative BPM readings (octave/dotted family) for the correction UI.
    pub candidates: Vec<f64>,
}

/// Analyze the tempo/beat grid of a source-audio WINDOW (the segment's visible window, in
/// SOURCE ms — pass 0/0 to analyze the whole file). Classical DSP (utai-dsp tempo.rs), no ML.
/// Errors are stable CODEs per the i18n rule: TEMPO_LOAD_FAILED / TEMPO_TOO_SHORT / TEMPO_NO_BEAT.
#[tauri::command]
pub async fn analyze_segment_tempo(
    path: String,
    window_start_ms: f64,
    window_end_ms: f64,
    beats_per_bar: u32,
) -> Result<TempoAnalysisResult, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let input = PathBuf::from(&path);
        let buf = crate::audio::load_audio(&input).map_err(|e| format!("TEMPO_LOAD_FAILED: {e}"))?;
        let ch = buf.channels.max(1) as usize;
        let total_frames = buf.samples.len() / ch;
        let sr = buf.sample_rate as f64;
        let mut a = ((window_start_ms.max(0.0) / 1000.0) * sr).round() as usize;
        let mut b = (((window_end_ms.max(0.0) / 1000.0) * sr).round() as usize).min(total_frames);
        // whole-file analysis is ONLY the explicit 0/0 sentinel — a window lying entirely past
        // the source's end must error, not silently analyze audio outside the segment (audit).
        let whole_file = window_start_ms <= 0.0 && window_end_ms <= 0.0;
        if b <= a {
            if !whole_file {
                return Err("TEMPO_TOO_SHORT".to_string());
            }
            a = 0;
            b = total_frames;
        }
        let mono: Vec<f32> = buf.samples[a * ch..b * ch]
            .chunks_exact(ch)
            .map(|fr| fr.iter().sum::<f32>() / ch as f32)
            .collect();
        let res = utai_dsp::tempo::analyze_tempo(&mono, buf.sample_rate, beats_per_bar).map_err(|e| {
            match e {
                utai_dsp::tempo::TempoError::TooShort => "TEMPO_TOO_SHORT".to_string(),
                utai_dsp::tempo::TempoError::NoBeat => "TEMPO_NO_BEAT".to_string(),
            }
        })?;
        Ok(TempoAnalysisResult {
            bpm: res.bpm,
            // anchor comes back window-relative — express it in source coordinates so it is
            // stable under segment split/resize (both halves keep the same grid)
            grid_anchor_ms: (a as f64 / sr) * 1000.0 + res.grid_anchor_ms,
            downbeat_index: res.downbeat_index,
            downbeat_margin: res.downbeat_margin,
            confidence: res.confidence,
            not_constant: res.not_constant,
            candidates: res.candidates,
        })
    })
    .await
    .map_err(|e| format!("TEMPO_JOIN: {e}"))?
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct StretchResult {
    pub output_path: String,
    pub duration_ms: f64,
    pub sample_rate: u32,
    pub channels: u16,
}

/// Offline time-stretch (tempo change, pitch preserved) of a whole source/stem file.
/// `time_factor` = output duration / input duration. Output is a CONTENT-ADDRESSED 32-bit float
/// WAV under audio_cache/stretch/ ({content_hash}_r{factor}.wav) — same input content + factor
/// ⇒ same artifact, so undo/redo and .usp reload re-resolve without recomputing, and a source
/// overwritten in place can never serve a stale stretch (the hash changes). Errors are stable
/// CODEs: STRETCH_RATIO_RANGE / STRETCH_INPUT_MISSING / STRETCH_ENGINE_FAILED.
#[tauri::command]
pub async fn stretch_segment_audio(
    state: State<'_, Arc<AppState>>,
    path: String,
    time_factor: f64,
) -> Result<StretchResult, String> {
    if !(time_factor.is_finite() && (0.25..=4.0).contains(&time_factor)) {
        return Err("STRETCH_RATIO_RANGE".to_string());
    }
    let cache_dir = state.cache_dir.join("audio_cache").join("stretch");
    tauri::async_runtime::spawn_blocking(move || {
        let input = PathBuf::from(&path);
        let bytes = std::fs::read(&input).map_err(|e| format!("STRETCH_INPUT_MISSING: {e}"))?;
        let content_hash = format!("{:016x}", xxhash_rust::xxh3::xxh3_64(&bytes));
        drop(bytes);

        let _ = std::fs::create_dir_all(&cache_dir);
        let key = format!("{content_hash}_r{time_factor:.6}");
        let wav_path = cache_dir.join(format!("{key}.wav"));
        let sidecar = cache_dir.join(format!("{key}.json"));

        // cache hit (sidecar is written LAST = completion marker, per the export convention)
        if let Ok(text) = std::fs::read_to_string(&sidecar) {
            if let Ok(info) = serde_json::from_str::<StretchResult>(&text) {
                if std::path::Path::new(&info.output_path).exists() {
                    return Ok(info);
                }
            }
        }

        let buf = crate::audio::load_audio(&input).map_err(|e| format!("STRETCH_INPUT_MISSING: {e}"))?;
        let stretched = utai_stretch::stretch_interleaved(
            &buf.samples,
            buf.channels.max(1) as usize,
            buf.sample_rate,
            time_factor,
            0.0,
        )?;
        let out_buf = crate::audio::AudioBuffer {
            samples: stretched,
            sample_rate: buf.sample_rate,
            channels: buf.channels.max(1),
        };
        // unique temp + atomic publish (same pattern as load_audio_file: concurrent stretches of
        // the same (content, factor) must not collide on Windows)
        let tmp = cache_dir.join(format!(
            "{key}.{}.{}.tmp",
            std::process::id(),
            TMP_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        ));
        if let Err(e) = crate::audio::save_wav_f32(&tmp, &out_buf) {
            let _ = std::fs::remove_file(&tmp);
            return Err(format!("STRETCH_ENGINE_FAILED: {e}"));
        }
        if let Err(e) = std::fs::rename(&tmp, &wav_path) {
            let _ = std::fs::remove_file(&tmp);
            // a lost race is fine (identical content already published); any OTHER rename failure
            // must not return a path that does not exist (the sidecar is the completion marker)
            if !wav_path.exists() {
                return Err(format!("STRETCH_ENGINE_FAILED: publish: {e}"));
            }
        }
        let info = StretchResult {
            output_path: wav_path.to_string_lossy().to_string(),
            duration_ms: out_buf.duration_secs() * 1000.0,
            sample_rate: out_buf.sample_rate,
            channels: out_buf.channels,
        };
        let _ = std::fs::write(&sidecar, serde_json::to_string(&info).unwrap_or_default());
        Ok(info)
    })
    .await
    .map_err(|e| format!("STRETCH_JOIN: {e}"))?
}
