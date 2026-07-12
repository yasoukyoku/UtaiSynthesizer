// S63 — Audio-export ENCODE stage. The frontend renders the whole-project mixdown OFFLINE
// (src/lib/audio/exportMixdown.ts — an OfflineAudioContext, i.e. the very engine playback uses, so the
// bounce is playback-parity by construction) and ships the interleaved stereo f32le PCM here in ONE
// raw-body IPC hop (InvokeBody::Raw). That deliberately avoids the run_rvc-style Vec<f32> JSON round
// trip (~100MB of JSON, the S59 "O5" debt) — raw bytes cross the IPC boundary uncopied by serde.
//
// Encoding split:
//   WAV  → hound, in-process (16 / 24 / 32-float). Zero external dependency, works even when ffmpeg is
//          absent — WAV export must never be hostage to the bundle.
//   FLAC / MP3 / OGG(Vorbis) / Opus / M4A(AAC) → the bundled ffmpeg, fed the raw f32le PCM directly
//          (-f f32le -ar .. -ac 2), so no intermediate WAV is materialized. Missing ffmpeg is a loud
//          stable CODE, never a silent fallback.
//
// Protocol (front-end drives, single-flight by UI): export_audio_pcm (raw body, stashes the PCM) →
// export_audio_encode (parameters incl. the non-ASCII-safe output path — which is why it can't ride the
// raw request's headers) → done. export_audio_discard frees the stash on dialog cancel / render error.
//
// Fixed-point conversion CLAMPS (mirrors audio::save_wav): the float sum can exceed ±1 and live playback
// clamps at the hardware — clamping here keeps "what you exported = what you heard". The frontend
// surfaces a peak>1 warning so the user can pull faders down if they want a clean-headroom bounce.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use tauri::ipc::{InvokeBody, Request};

use crate::commands::audition::{FlightGuard, BUSY_RETRY_MSG};

/// PCM stash between the two IPC hops. A plain module Mutex (the audition.rs static-state convention) —
/// the export dialog is single-flight, so one slot is the whole protocol.
static PENDING_PCM: Mutex<Option<Vec<u8>>> = Mutex::new(None);

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(serde::Serialize, Debug)]
pub struct ExportAudioResult {
    pub out_path: String,
    pub duration_ms: u64,
    pub file_bytes: u64,
}

/// Hop 1a: start a fresh PCM transfer. The PCM arrives in CHUNKS (hop 1b), NEVER as one giant raw
/// body: Tauri's IPC silently falls back from the custom-protocol fetch to postMessage when a fetch
/// rejects — and the postMessage path Array.from()s the payload into a JS number array before
/// JSON.stringify (tauri scripts/ipc-protocol.js + process-ipc-message-fn.js). For a ~100MB bounce
/// that fallback allocates a 100M-element array + a ~400MB JSON string and OOM-kills the WebView2
/// renderer (S63 live crash: silent process death, no panic/WER). Small chunks make the fallback
/// harmless even if it ever triggers. `total_bytes` lets hop 1b preallocate + sanity-cap.
#[tauri::command]
pub fn export_audio_pcm_begin(total_bytes: u64) -> Result<(), String> {
    // 8 = 4 bytes × 2 channels — anything else means a truncated/misaligned transfer plan.
    // Cap mirrors the frontend's 60-min guard (60min × 48k × 2ch × 4B ≈ 1.4GB) with headroom.
    if total_bytes == 0 || total_bytes % 8 != 0 || total_bytes > 2_000_000_000 {
        return Err(format!("EXPORT_BAD_PCM: plan {total_bytes} bytes"));
    }
    *PENDING_PCM.lock().unwrap() = Some(Vec::with_capacity(total_bytes as usize));
    Ok(())
}

/// Hop 1b: append one raw chunk (frontend sends ~8MB slices in order, awaiting each).
#[tauri::command]
pub fn export_audio_pcm_chunk(request: Request<'_>) -> Result<(), String> {
    let InvokeBody::Raw(bytes) = request.body() else {
        return Err("EXPORT_BAD_PCM: expected a raw chunk body".to_string());
    };
    if bytes.is_empty() {
        return Err("EXPORT_BAD_PCM: empty chunk".to_string());
    }
    let mut slot = PENDING_PCM.lock().unwrap();
    let Some(buf) = slot.as_mut() else {
        return Err("EXPORT_NO_PCM".to_string()); // chunk without begin (or after a discard)
    };
    if buf.len() + bytes.len() > buf.capacity() {
        // More data than the begin() plan — a desynced/duplicated transfer must not grow unbounded.
        *slot = None;
        return Err("EXPORT_BAD_PCM: overflow".to_string());
    }
    buf.extend_from_slice(bytes);
    Ok(())
}

/// Free the stash without encoding (dialog cancelled between the hops, or the encode errored and the
/// user closed the dialog). Idempotent.
#[tauri::command]
pub fn export_audio_discard() {
    *PENDING_PCM.lock().unwrap() = None;
}

/// Hop 2: encode the stashed PCM to `out_path`. Consumes the stash either way (a failed encode requires
/// a fresh render → re-stash; the frontend re-runs the whole export, which is cheap).
#[tauri::command]
pub async fn export_audio_encode(
    out_path: String,
    format: String,
    sample_rate: u32,
    bit_depth: String,
    bitrate_kbps: u32,
) -> Result<ExportAudioResult, String> {
    if sample_rate == 0 {
        return Err("EXPORT_ENCODE_FAIL: sample_rate 0".to_string());
    }
    tauri::async_runtime::spawn_blocking(move || {
        // Mutual exclusion with the storage cleanup / audition / training-start family: cleanup deletes
        // cache trees while we may still be reading nothing (the PCM is already in memory) — but holding
        // the guard keeps an export from overlapping the cleanup's own FlightGuard window, and cheaply
        // serializes concurrent exports (the UI is single-flight anyway; this is the TOCTOU backstop).
        // Guard FIRST, take SECOND (audit): a transient APP_BUSY rejection must leave the stash intact
        // so a retry doesn't force the user through a whole re-render + re-transfer.
        let _guard = FlightGuard::acquire(BUSY_RETRY_MSG)?;
        let pcm = PENDING_PCM
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| "EXPORT_NO_PCM".to_string())?;
        if pcm.is_empty() || pcm.len() % 8 != 0 {
            return Err(format!("EXPORT_BAD_PCM: {} bytes assembled", pcm.len()));
        }
        encode(&pcm, &out_path, &format, sample_rate, &bit_depth, bitrate_kbps)
    })
    .await
    .map_err(|e| format!("EXPORT_ENCODE_FAIL: join — {e}"))?
}

fn encode(
    pcm: &[u8],
    out_path: &str,
    format: &str,
    sample_rate: u32,
    bit_depth: &str,
    bitrate_kbps: u32,
) -> Result<ExportAudioResult, String> {
    let frames = (pcm.len() / 8) as u64;
    let duration_ms = frames * 1000 / sample_rate as u64;

    match format {
        "wav" => encode_wav(pcm, out_path, sample_rate, bit_depth)?,
        "flac" | "mp3" | "ogg" | "opus" | "m4a" => {
            encode_ffmpeg(pcm, out_path, format, sample_rate, bit_depth, bitrate_kbps)?
        }
        other => return Err(format!("EXPORT_FORMAT_UNSUPPORTED: {other}")),
    }

    let file_bytes = std::fs::metadata(out_path)
        .map_err(|e| format!("EXPORT_WRITE_FAIL: {e}"))?
        .len();
    Ok(ExportAudioResult { out_path: out_path.to_string(), duration_ms, file_bytes })
}

/// In-process WAV writer. Sample conversion mirrors audio::save_wav's `(clamp × scale) as iN` exactly —
/// ONE quantization convention across the app.
fn encode_wav(pcm: &[u8], out_path: &str, sample_rate: u32, bit_depth: &str) -> Result<(), String> {
    let (bits, float) = match bit_depth {
        "16" => (16, false),
        "24" => (24, false),
        "32f" => (32, true),
        other => return Err(format!("EXPORT_FORMAT_UNSUPPORTED: wav/{other}")),
    };
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: bits,
        sample_format: if float { hound::SampleFormat::Float } else { hound::SampleFormat::Int },
    };
    // Mirror the ffmpeg branch's partial-output cleanup (audit): hound's Drop best-effort patches the
    // header, so an aborted write would otherwise leave a VALID-LOOKING truncated wav at out_path.
    let write_all = || -> Result<(), String> {
        let mut writer =
            hound::WavWriter::create(out_path, spec).map_err(|e| format!("EXPORT_WRITE_FAIL: {e}"))?;
        let samples = pcm.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
        let write_err = |e: hound::Error| format!("EXPORT_WRITE_FAIL: {e}");
        if float {
            for s in samples {
                writer.write_sample(s).map_err(write_err)?;
            }
        } else if bits == 16 {
            for s in samples {
                writer.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16).map_err(write_err)?;
            }
        } else {
            for s in samples {
                writer.write_sample((s.clamp(-1.0, 1.0) * 8_388_607.0) as i32).map_err(write_err)?;
            }
        }
        writer.finalize().map_err(|e| format!("EXPORT_WRITE_FAIL: {e}"))
    };
    write_all().inspect_err(|_| {
        let _ = std::fs::remove_file(out_path);
    })
}

/// Compressed formats: write the raw f32le PCM to a temp file (same temp-dir convention as
/// ffmpeg_decode_to_wav) and let ffmpeg read it with an explicit raw-PCM demuxer. stdout is discarded,
/// stderr is captured into the error CODE so codec problems surface verbatim.
fn encode_ffmpeg(
    pcm: &[u8],
    out_path: &str,
    format: &str,
    sample_rate: u32,
    bit_depth: &str,
    bitrate_kbps: u32,
) -> Result<(), String> {
    let ffmpeg = crate::audio::find_ffmpeg().ok_or_else(|| "EXPORT_FFMPEG_MISSING".to_string())?;

    let tmp = std::env::temp_dir().join(format!(
        "utai_export_{}_{}.f32",
        std::process::id(),
        TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&tmp, pcm).map_err(|e| {
        // fs::write creates before it writes — a mid-write failure (disk full) must not leak a
        // multi-hundred-MB partial temp file (audit).
        let _ = std::fs::remove_file(&tmp);
        format!("EXPORT_WRITE_FAIL: temp — {e}")
    })?;

    let kbps = format!("{}k", bitrate_kbps.clamp(32, 512));
    let mut args: Vec<String> = vec![
        "-f".into(), "f32le".into(),
        "-ar".into(), sample_rate.to_string(),
        "-ac".into(), "2".into(),
        "-i".into(), tmp.to_string_lossy().into_owned(),
    ];
    match format {
        "flac" => {
            args.extend(["-c:a".into(), "flac".into()]);
            match bit_depth {
                "24" => args.extend([
                    "-sample_fmt".into(), "s32".into(),
                    "-bits_per_raw_sample".into(), "24".into(),
                ]),
                // 16 (and anything else) → plain s16 FLAC, the interchange default.
                _ => args.extend(["-sample_fmt".into(), "s16".into()]),
            }
        }
        "mp3" => args.extend(["-c:a".into(), "libmp3lame".into(), "-b:a".into(), kbps.clone()]),
        "ogg" => args.extend(["-c:a".into(), "libvorbis".into(), "-b:a".into(), kbps.clone()]),
        "opus" => args.extend(["-c:a".into(), "libopus".into(), "-b:a".into(), kbps.clone()]),
        "m4a" => args.extend([
            "-c:a".into(), "aac".into(), "-b:a".into(), kbps.clone(),
            "-movflags".into(), "+faststart".into(),
        ]),
        other => {
            let _ = std::fs::remove_file(&tmp);
            return Err(format!("EXPORT_FORMAT_UNSUPPORTED: {other}"));
        }
    }
    args.extend(["-v".into(), "error".into(), "-y".into(), out_path.to_string()]);

    let output = {
        use std::os::windows::process::CommandExt;
        std::process::Command::new(&ffmpeg)
            .args(&args)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .creation_flags(crate::util::CREATE_NO_WINDOW)
            .output()
    };
    let _ = std::fs::remove_file(&tmp);
    let output = output.map_err(|e| format!("EXPORT_ENCODE_FAIL: spawn — {e}"))?;
    if !output.status.success() {
        // A failed encode can leave a partial output file — don't let it masquerade as a bounce.
        let _ = std::fs::remove_file(Path::new(out_path));
        let stderr = String::from_utf8_lossy(&output.stderr);
        let tail: String = stderr.trim().chars().rev().take(400).collect::<Vec<_>>().into_iter().rev().collect();
        return Err(format!("EXPORT_ENCODE_FAIL: {tail}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_bytes(samples: &[f32]) -> Vec<u8> {
        samples.iter().flat_map(|s| s.to_le_bytes()).collect()
    }

    #[test]
    fn wav_16_roundtrip_and_clamp() {
        let dir = std::env::temp_dir();
        let out = dir.join(format!("utai_test_export16_{}.wav", std::process::id()));
        // L=0.5, R=-0.5 then an out-of-range pair that must clamp.
        let pcm = f32_bytes(&[0.5, -0.5, 1.5, -1.5]);
        encode_wav(&pcm, out.to_str().unwrap(), 44_100, "16").unwrap();
        let mut reader = hound::WavReader::open(&out).unwrap();
        let spec = reader.spec();
        assert_eq!(spec.channels, 2);
        assert_eq!(spec.sample_rate, 44_100);
        assert_eq!(spec.bits_per_sample, 16);
        let s: Vec<i16> = reader.samples::<i16>().map(|x| x.unwrap()).collect();
        assert_eq!(s, vec![16383, -16383, 32767, -32767]);
        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn wav_24_and_32f_write() {
        let dir = std::env::temp_dir();
        let pcm = f32_bytes(&[0.25, -0.25]);
        let out24 = dir.join(format!("utai_test_export24_{}.wav", std::process::id()));
        encode_wav(&pcm, out24.to_str().unwrap(), 48_000, "24").unwrap();
        let mut r24 = hound::WavReader::open(&out24).unwrap();
        assert_eq!(r24.spec().bits_per_sample, 24);
        let s24: Vec<i32> = r24.samples::<i32>().map(|x| x.unwrap()).collect();
        assert_eq!(s24, vec![2_097_151, -2_097_151]); // (0.25 × 8388607) truncated
        let _ = std::fs::remove_file(&out24);

        let out32 = dir.join(format!("utai_test_export32f_{}.wav", std::process::id()));
        encode_wav(&pcm, out32.to_str().unwrap(), 48_000, "32f").unwrap();
        let mut r32 = hound::WavReader::open(&out32).unwrap();
        assert_eq!(r32.spec().sample_format, hound::SampleFormat::Float);
        let s32: Vec<f32> = r32.samples::<f32>().map(|x| x.unwrap()).collect();
        assert_eq!(s32, vec![0.25, -0.25]);
        let _ = std::fs::remove_file(&out32);
    }

    #[test]
    fn unsupported_format_and_bit_depth_error() {
        let pcm = f32_bytes(&[0.0, 0.0]);
        let out = std::env::temp_dir().join("utai_test_export_bad.wav");
        let e = encode_wav(&pcm, out.to_str().unwrap(), 44_100, "8").unwrap_err();
        assert!(e.contains("EXPORT_FORMAT_UNSUPPORTED"));
        let e2 = encode(&pcm, out.to_str().unwrap(), "wma", 44_100, "16", 192).unwrap_err();
        assert!(e2.contains("EXPORT_FORMAT_UNSUPPORTED"));
    }

    /// Real ffmpeg encode smoke — needs ffmpeg on PATH (dev box). `cargo test -- --ignored export_audio`.
    #[test]
    #[ignore]
    fn ffmpeg_encodes_all_compressed_formats() {
        let dir = std::env::temp_dir();
        // 0.5s of a 440Hz-ish ramp, stereo.
        let n = 22_050usize;
        let mut samples = Vec::with_capacity(n * 2);
        for i in 0..n {
            let v = ((i as f32) * 0.05).sin() * 0.3;
            samples.push(v);
            samples.push(-v);
        }
        let pcm = f32_bytes(&samples);
        for (fmt, ext) in [("flac", "flac"), ("mp3", "mp3"), ("ogg", "ogg"), ("opus", "opus"), ("m4a", "m4a")] {
            let out = dir.join(format!("utai_test_export_{}_{}.{ext}", std::process::id(), fmt));
            let r = encode_ffmpeg(&pcm, out.to_str().unwrap(), fmt, 44_100, "16", 192);
            assert!(r.is_ok(), "{fmt}: {r:?}");
            let sz = std::fs::metadata(&out).unwrap().len();
            assert!(sz > 500, "{fmt}: suspiciously small ({sz} B)");
            let _ = std::fs::remove_file(&out);
        }
    }
}
