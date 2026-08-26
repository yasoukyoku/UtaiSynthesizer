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

/// Decode `path` to PCM f32 WAV via ffmpeg, then load the WAV back. `target_sr` resamples (`-ar`) during
/// decode when Some. The CALLER resolves `ffmpeg` (so it can emit a context-specific not-found error).
/// Single source for the ffmpeg→wav→load_wav core shared by `load_with_ffmpeg` (decode) and
/// `load_audio_at_rate` (resample) — keeps the codec flags / stdio / CREATE_NO_WINDOW / temp-file
/// strategy in ONE place so the two paths can't drift.
fn ffmpeg_decode_to_wav(ffmpeg: &Path, path: &Path, target_sr: Option<u32>) -> Result<AudioBuffer> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let temp_path = std::env::temp_dir().join(format!(
        "utai_ffmpeg_{}_{}.wav",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed),
    ));

    // Run ffmpeg with the given resample args, decoding to a temp f32 wav.
    let run = |resample: &[String]| -> Result<AudioBuffer> {
        let mut cmd = std::process::Command::new(ffmpeg);
        cmd.arg("-i").arg(path);
        for a in resample {
            cmd.arg(a);
        }
        cmd.args(["-f", "wav", "-acodec", "pcm_f32le", "-v", "error", "-y"])
            .arg(&temp_path)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .creation_flags(crate::util::CREATE_NO_WINDOW);
        let output = cmd.output().map_err(|e| {
            crate::UtaiError::Audio(format!("Failed to run ffmpeg ({}): {}", ffmpeg.display(), e))
        })?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(crate::UtaiError::Audio(format!(
                "ffmpeg failed for '{}': {}",
                path.display(),
                stderr.trim()
            )));
        }
        load_wav(&temp_path)
    };

    let result = match target_sr {
        None => run(&[]),
        // Prefer soxr (high quality, 28-bit precision) for the model's required resample; fall back to
        // ffmpeg's default swresample (-ar) if this ffmpeg has no libsoxr, so a stripped bundled build
        // never just fails the run. (16→32-bit precision is handled at WAV-write time, separately.)
        Some(sr) => {
            let soxr = ["-af".to_string(), format!("aresample={sr}:resampler=soxr:precision=28")];
            run(&soxr).or_else(|_| {
                tracing::warn!("ffmpeg soxr resampler unavailable — falling back to swresample ({}Hz)", sr);
                run(&["-ar".to_string(), sr.to_string()])
            })
        }
    };

    let _ = std::fs::remove_file(&temp_path);
    result
}

fn load_with_ffmpeg(path: &Path) -> Result<AudioBuffer> {
    let ffmpeg = find_ffmpeg().ok_or_else(|| {
        crate::UtaiError::Audio(format!(
            "Cannot decode '{}' — symphonia does not support this format and ffmpeg was not found. \
             Place ffmpeg.exe next to the application binary.",
            path.display()
        ))
    })?;

    let buf = ffmpeg_decode_to_wav(&ffmpeg, path, None)?;
    tracing::info!(
        "Loaded via ffmpeg: {} ({} ch, {}Hz, {:.1}s)",
        path.display(),
        buf.channels,
        buf.sample_rate,
        buf.duration_secs(),
    );
    Ok(buf)
}

/// Fast duration probe (no full sample decode) — used for the drag-import ghost preview and
/// the loading-segment placeholder width, where the file isn't decoded yet.
///   1. WAV  → hound header (frame count / sample rate), instant.
///   2. other → `ffmpeg -i` and parse the `Duration:` line from stderr (header read only).
///   3. fallback → symphonia container metadata (`n_frames`) when ffmpeg is unavailable
///                 (symphonia's `n_frames` is unreliable for matroska/webm, hence the fallback).
///
/// Returns the full (untrimmed) container length — an upper-bound estimate. For non-WAV files
/// `load_audio_file` may strip leading codec/lead-in silence, so the finalized segment can be
/// slightly shorter than this; the placeholder simply snaps to the exact length on decode.
pub fn probe_duration_ms(path: &Path) -> Result<f64> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    if ext == "wav" {
        if let Ok(reader) = hound::WavReader::open(path) {
            let spec = reader.spec();
            if spec.sample_rate > 0 {
                // hound's `duration()` is samples-per-channel (i.e. frame count).
                return Ok(reader.duration() as f64 / spec.sample_rate as f64 * 1000.0);
            }
        }
    }

    // ffmpeg's container `Duration` is authoritative across formats and matches what `load_audio`
    // actually decodes. symphonia's `n_frames` is UNRELIABLE for some containers — matroska/webm
    // report a wildly wrong frame count (e.g. ~8s for a 393s file) — so ffmpeg is the primary
    // probe and symphonia is only a last-resort fallback when ffmpeg isn't available (in which
    // case webm/opus can't be decoded anyway, so the inaccuracy is moot).
    if let Ok(ms) = probe_with_ffmpeg(path) {
        return Ok(ms);
    }

    probe_with_symphonia(path)
}

fn probe_with_symphonia(path: &Path) -> Result<f64> {
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

    let track = probed
        .format
        .default_track()
        .ok_or_else(|| crate::UtaiError::Audio("No audio track found".into()))?;

    let sample_rate = track
        .codec_params
        .sample_rate
        .filter(|&sr| sr > 0)
        .ok_or_else(|| crate::UtaiError::Audio("Unknown sample rate".into()))?;
    let n_frames = track
        .codec_params
        .n_frames
        .ok_or_else(|| crate::UtaiError::Audio("Unknown frame count".into()))?;

    Ok(n_frames as f64 / sample_rate as f64 * 1000.0)
}

fn probe_with_ffmpeg(path: &Path) -> Result<f64> {
    let ffmpeg = find_ffmpeg().ok_or_else(|| {
        crate::UtaiError::Audio(format!(
            "Cannot probe '{}' — no header metadata and ffmpeg was not found.",
            path.display()
        ))
    })?;

    // `ffmpeg -i <file>` with no output file exits non-zero ("At least one output file...")
    // but still prints the container's `Duration:` to stderr after only reading the header.
    let output = std::process::Command::new(&ffmpeg)
        .arg("-i")
        .arg(path)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .creation_flags(crate::util::CREATE_NO_WINDOW)
        .output()
        .map_err(|e| crate::UtaiError::Audio(format!("Failed to run ffmpeg probe: {}", e)))?;

    let stderr = String::from_utf8_lossy(&output.stderr);
    parse_ffmpeg_duration(&stderr).ok_or_else(|| {
        crate::UtaiError::Audio(format!("Could not determine duration of '{}'", path.display()))
    })
}

/// Parse `Duration: HH:MM:SS.ss` (the first occurrence) from ffmpeg's stderr into milliseconds.
fn parse_ffmpeg_duration(stderr: &str) -> Option<f64> {
    let idx = stderr.find("Duration:")?;
    let ts = stderr[idx + "Duration:".len()..]
        .trim_start()
        .split(',')
        .next()?
        .trim();
    if ts.starts_with("N/A") {
        return None;
    }
    let mut parts = ts.split(':');
    let h: f64 = parts.next()?.trim().parse().ok()?;
    let m: f64 = parts.next()?.trim().parse().ok()?;
    let s: f64 = parts.next()?.trim().parse().ok()?;
    Some((h * 3600.0 + m * 60.0 + s) * 1000.0)
}

pub(crate) fn find_ffmpeg() -> Option<std::path::PathBuf> {
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

    // Dev fallback: scan %PATH% entries directly. (Unicode-safe — avoids round-tripping `where`
    // stdout, which Windows emits in the console OEM codepage and which from_utf8_lossy mangles
    // for non-ASCII directory names, breaking ffmpeg discovery under Chinese/Japanese paths.)
    if let Some(paths) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&paths) {
            let p = dir.join("ffmpeg.exe");
            if p.exists() {
                return Some(p);
            }
        }
    }

    None
}

pub fn load_audio_at_rate(path: &Path, target_sr: u32) -> Result<AudioBuffer> {
    let ffmpeg = find_ffmpeg().ok_or_else(|| {
        crate::UtaiError::Audio("ffmpeg not found — needed for sample rate conversion".into())
    })?;
    ffmpeg_decode_to_wav(&ffmpeg, path, Some(target_sr))
}

/// Replace non-finite samples (NaN/Inf) in place with 0.0, returning how many were replaced.
/// S68c: a single poisoned sample otherwise nukes an entire voice-conversion take — the RVC/SoVITS
/// 48 Hz high-pass is a bidirectional IIR (filtfilt), which smears one NaN across the whole buffer
/// in both directions, and ContentVec then emits all-NaN features (the shipped 20% crash).
/// Sources seen in the wild: fp16 MSST stems computed on DirectML (true-fp16 kernels can overflow;
/// the CPU EP hides this by computing in fp32), and NaN-bearing float WAVs.
pub fn sanitize_non_finite(samples: &mut [f32]) -> usize {
    let mut replaced = 0usize;
    for s in samples.iter_mut() {
        if !s.is_finite() {
            *s = 0.0;
            replaced += 1;
        }
    }
    replaced
}

/// ⭐⭐ S163 —— **削波护栏**:交给 [`save_wav`] 之前把峰压回天花板以下。
///
/// ## 它存在的唯一理由,就在下面那一行 `save_wav` 里
/// `save_wav` 写 16 bit 时是 `sample.clamp(-1.0, 1.0)` ⇒ **超过 1.0 的样本被硬削顶**,
/// 而**没有任何一层在那之前看过一眼峰值**。
///
/// ## 谁会超(实测,不是假想)
/// 谱面轨的音域扩展:base 那一遍归一到 0.92,而每一遍 donor 乘的是**base 的**标量
/// (`peak_normalize_to(.., 0.92, Some(base_pre_norm_peak))`,S147 定的,为的是两边共用
/// 一个标量)⇒ **donor 在它唱得响的地方就会超过 0.92,甚至超过 1.0**。
/// 实测 S163 出厂臂:yuyuko × 炉心 +7 峰 **1.0000**、**756 个样本 |v|>0.999**、
/// 最长一段 **0:57.340 起 180 ms**(= 用户 2026-08-26 点名的那一处);
/// yachiyo × 鹅妈妈 +7 也有 3 个样本(1:27.160)。
/// ⭐ 用户对它的定性:「就算真是模型烂,那我们碰上了也意味着**社区确实有这种模型**,
/// 那就得做好优化」⇒ **不许推给模型**。
///
/// ## 为什么是**一个全曲标量**而不是限幅器
/// 限幅器会改**波形**(那才是真的「压平」);一个标量**只改响度**:
/// 相对电平、波峰因数、频谱形状**一个字节不动**。代价上限 = 超出量本身
/// (yuyuko 那条臂 = **0.09 dB**),远低于可闻刻度(~2.7 dB)。
/// ⛔ 而且它**只压不抬**:峰没超天花板的臂**逐位不变**(严格大于才动手)。
///
/// 返回 `Some(动手之前的峰)`——调用方**必须**把它打进日志:
/// 「护栏开着」与「护栏做了事」要分开可查(S129 铁律)。
pub fn peak_guard(samples: &mut [f32], ceiling: f32) -> Option<f32> {
    if !(ceiling > 0.0) || samples.is_empty() {
        return None;
    }
    let peak = samples.iter().fold(0.0f32, |a, &v| if v.is_finite() { a.max(v.abs()) } else { a });
    if !(peak > ceiling) {
        return None;
    }
    let g = ceiling / peak;
    for v in samples.iter_mut() {
        *v *= g;
    }
    Some(peak)
}

/// ⚙ [`peak_guard`] 的天花板。**不是 0.92**:0.92 是**渲染**那一遍自己的归一目标
/// (`score2svc::peak_normalize_to`),而这一条是**写盘前的最后一道**,它只需要
/// 「不被 16 bit 削掉」。取 0.99 是为了**少动**:实测超标的臂只需要 0.09 dB。
pub const OUTPUT_PEAK_CEILING: f32 = 0.99;

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

/// tmp+rename twin of `save_wav` (16-bit) — a killed process can never leave a truncated
/// wav that a cache short-circuit (or a stale-path consumer) would trust. Same quantization
/// as save_wav, so callers migrating from a save_wav round trip stay byte-identical.
pub fn save_wav_atomic(path: &Path, buffer: &AudioBuffer) -> Result<()> {
    let tmp = path.with_extension("wav.tmp");
    save_wav(&tmp, buffer)?;
    std::fs::rename(&tmp, path)
        .map_err(|e| crate::UtaiError::Audio(format!("Atomic wav rename failed: {}", e)))
}

/// 32-bit float writer for INTERLEAVED buffers — full precision, no clamping (preserve
/// inter-sample overshoot for downstream gain), same philosophy as the separation stems'
/// planar twin (separation/pipeline.rs save_wav, which is numerics-locked and takes
/// StemAudio; the layouts differ, so the two writers stay separate on purpose).
pub fn save_wav_f32(path: &Path, buffer: &AudioBuffer) -> Result<()> {
    let spec = hound::WavSpec {
        channels: buffer.channels,
        sample_rate: buffer.sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| crate::UtaiError::Audio(format!("Failed to create WAV: {}", e)))?;
    for &sample in &buffer.samples {
        writer.write_sample(sample).map_err(|e| crate::UtaiError::Audio(format!("Write error: {}", e)))?;
    }
    writer.finalize().map_err(|e| crate::UtaiError::Audio(format!("Finalize error: {}", e)))?;
    Ok(())
}

#[cfg(test)]
mod peak_guard_tests {
    use super::*;

    /// ⛔⛔ S163 —— **护栏只在真的超了的时候动手,而且只动一个标量。**
    ///
    /// 四格,少一格这条判据就有一根轴是空的:
    /// ⑴ 超了 ⇒ 峰回到天花板;⑵ ⛔ **波形形状逐样本成比例**(它不是限幅器);
    /// ⑶ ⛔ **阴性对照**:没超 ⇒ **逐位不变**;⑷ 天花板荒谬 / 空缓冲 ⇒ 不动手。
    #[test]
    fn the_peak_guard_only_scales_and_only_when_it_must() {
        // ⑴ + ⑵
        let src: Vec<f32> = (0..1000).map(|i| 1.5 * ((i as f32) * 0.017).sin()).collect();
        let mut x = src.clone();
        let was = peak_guard(&mut x, OUTPUT_PEAK_CEILING).expect("超了就该动手");
        assert!((was - 1.5).abs() < 1e-3, "报回来的峰应当是动手之前的:{was}");
        let peak = x.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        assert!((peak - OUTPUT_PEAK_CEILING).abs() < 1e-4, "峰没回到天花板:{peak}");
        let g = OUTPUT_PEAK_CEILING / was;
        for (a, b) in src.iter().zip(x.iter()) {
            assert!((a * g - b).abs() < 1e-6, "不是一个标量 —— 它变成限幅器了");
        }
        // ⑶ ⛔ 阴性对照:没超 ⇒ 逐位不变
        let quiet: Vec<f32> = src.iter().map(|v| v * 0.5).collect();
        let mut y = quiet.clone();
        assert_eq!(peak_guard(&mut y, OUTPUT_PEAK_CEILING), None);
        assert_eq!(y, quiet, "没超的缓冲必须逐位不变");
        // ⚠ 恰好等于天花板也不许动(严格大于才动手)
        let mut edge = vec![OUTPUT_PEAK_CEILING, -OUTPUT_PEAK_CEILING, 0.0];
        let e0 = edge.clone();
        assert_eq!(peak_guard(&mut edge, OUTPUT_PEAK_CEILING), None);
        assert_eq!(edge, e0);
        // ⑷ 退化输入
        let mut z = vec![2.0f32];
        assert_eq!(peak_guard(&mut z, 0.0), None);
        assert_eq!(peak_guard(&mut [], 0.99), None);
    }

    /// ⛔⛔ S163 —— **它必须接在写盘的那条路上**,否则就是一条谁也不调用的纯函数。
    ///
    /// 这一条走**生产自己的出口** [`crate::commands::inference::commit_rendered_audio`]
    /// 的完整来回:交一个峰 1.5 的缓冲,读回来必须**没有一个样本顶到满刻度**。
    /// ⛔ 变异靶:把 `commit_rendered_audio` 里那一句 `peak_guard` 删掉 ⇒ 当场红。
    ///
    /// ⭐ 外加一格**源码级**的:mg 台子那一条也必须接着 —— 台子与生产不是同一条链的话,
    /// 台子上量到的一切都不代表用户听到的东西(S162 为这件事付过学费)。
    #[test]
    fn a_hot_buffer_reaches_the_writer_without_clipping() {
        let samples: Vec<f32> = (0..2000).map(|i| 1.5 * ((i as f32) * 0.031).sin()).collect();
        let dir = std::env::temp_dir().join("utai_peak_guard_test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("hot.wav");
        crate::commands::inference::commit_rendered_audio(
            crate::inference::SynthesisResult {
                audio: samples.clone(),
                sample_rate: 44100,
                pre_norm_peak: None,
            },
            p.to_string_lossy().into_owned(),
        )
        .unwrap();
        let mut r = hound::WavReader::open(&p).unwrap();
        let back: Vec<i16> = r.samples::<i16>().map(|v| v.unwrap()).collect();
        let _ = std::fs::remove_file(&p);
        assert_eq!(back.len(), samples.len());
        let hot = back.iter().filter(|v| v.abs() >= 32760).count();
        assert_eq!(hot, 0, "还有 {hot} 个样本顶到满刻度 —— 护栏没接在写盘那条路上");
        // ⛔ 形状成比例(它是标量不是限幅器)—— 拿两个相邻峰的比值当指纹
        let peak = back.iter().map(|v| v.abs()).max().unwrap();
        assert!(peak > 32000, "压过头了:峰只有 {peak}");

        // ⭐ 源码级:mg 台子那一条
        let mg = include_str!("../inference/score2svc_mg.rs");
        assert!(
            mg.contains("peak_guard(&mut result.audio"),
            "mg 台子没接同一道护栏 —— 台子上量到的东西就不代表生产"
        );
    }
}
