//! P0-B: Lyrics extraction (faster-whisper ASR via the AMT sidecar) and
//! MIDI lyric read/write (SMF Lyric meta events on a dedicated track).
//!
//! Extraction spawns `amt_sidecar.py lyrics`, streams `@@PROGRESS@@` lines as
//! the `amt-lyrics-progress` event and parses the `@@RESULT@@` payload.
//! Write-back serializes timed lyric lines as Lyric meta events on a dedicated
//! "Lyrics" track (previous lyrics are stripped first so re-saving is stable).
//! Read parses existing Lyric events back into seconds-timed lines so the
//! editor round-trips across app restarts.

use std::path::PathBuf;
use std::sync::Arc;

use tauri::{AppHandle, Emitter, State};

use crate::AppState;

use super::amt::{resolve_amt_python, resolve_sidecar_dir};

/// Node id under which the running lyrics sidecar is registered so
/// `cancel_amt_midi` can force-kill it from the frontend.
pub const LYRICS_NODE_ID: &str = "amt-lyrics";

/// midly's u28 has no MAX constant — 0x0FFF_FFFF is 2^28-1.
const U28_MAX: u64 = 0x0FFF_FFFF;

#[derive(serde::Serialize, Clone)]
pub struct AmtLyricSegment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

#[derive(serde::Serialize, Clone)]
pub struct AmtLyricsResult {
    pub language: Option<String>,
    pub language_probability: f64,
    pub duration: f64,
    pub device: String,
    pub model: String,
    pub segments: Vec<AmtLyricSegment>,
}

/// Extract timed lyrics from an audio file with faster-whisper (auto language
/// detection unless `language` is forced). The model must be installed via the
/// resource manager (`<models_dir>/whisper/<size>/model.bin`); a missing model
/// fails fast with `LYRICS_MODEL_NOT_INSTALLED` instead of letting
/// huggingface-hub hang on a blocked network.
#[tauri::command]
pub async fn amt_extract_lyrics(
    app: AppHandle,
    state: State<'_, Arc<AppState>>,
    audio_path: String,
    output_dir: String,
    model: Option<String>,
    language: Option<String>,
) -> Result<AmtLyricsResult, String> {
    let sidecar_dir = resolve_sidecar_dir(&state);
    let sidecar_script = sidecar_dir.join("amt_sidecar.py");
    let python = resolve_amt_python(&sidecar_dir, &state.app_dir);

    if !sidecar_script.exists() {
        return Err("AMT_SIDECAR_NOT_FOUND".into());
    }

    let model_size = model
        .as_deref()
        .map(str::trim)
        .filter(|m| !m.is_empty())
        .unwrap_or("small")
        .to_lowercase();

    // The resource-managed model must exist locally — no silent HF downloads.
    let model_dir = state.amt_models_dir.join("whisper").join(&model_size);
    if !model_dir.join("model.bin").is_file() || !model_dir.join("config.json").is_file() {
        return Err(format!("LYRICS_MODEL_NOT_INSTALLED: {model_size}"));
    }

    // Absolutize paths (the sidecar chdirs to its own folder at startup).
    let absolutize = |p: &str| -> String {
        let pb = PathBuf::from(p);
        if pb.is_absolute() {
            pb.to_string_lossy().into_owned()
        } else {
            std::path::absolute(&pb)
                .map(|a| a.to_string_lossy().into_owned())
                .unwrap_or_else(|_| pb.to_string_lossy().into_owned())
        }
    };
    let audio_path = absolutize(&audio_path);
    let output_dir = absolutize(&output_dir);
    if !std::path::Path::new(&audio_path).is_file() {
        return Err(format!("LYRICS_AUDIO_NOT_FOUND: {audio_path}"));
    }
    std::fs::create_dir_all(&output_dir)
        .map_err(|e| format!("LYRICS_OUTDIR_FAILED: {e}"))?;

    // Minimal config — the lyrics command doesn't read it, but the shared
    // parser requires a valid `--config`.
    let config_path = sidecar_dir.join("amt_lyrics_config.json");
    let config_text = serde_json::to_string(&serde_json::json!({}))
        .map_err(|e| format!("LYRICS_CONFIG_ERROR: {e}"))?;
    std::fs::write(&config_path, &config_text)
        .map_err(|e| format!("LYRICS_CONFIG_WRITE_FAILED: {e}"))?;

    // Prime PATH with the venv's CUDA runtime DLLs (cuBLAS 12 / cuDNN 9 ship
    // inside torch's lib dir) so ctranslate2 can run whisper on the GPU.
    let mut command = tokio::process::Command::new(&python);
    command
        .arg(&sidecar_script)
        .arg("lyrics")
        .arg("--audio")
        .arg(&audio_path)
        .arg("--outdir")
        .arg(&output_dir)
        .arg("--config")
        .arg(format!("@{}", config_path.to_string_lossy()))
        .arg("--model")
        .arg(&model_size)
        .arg("--language")
        .arg(language.as_deref().unwrap_or(""))
        .current_dir(&sidecar_dir)
        .env("PYTHONIOENCODING", "utf-8")
        .env(
            "MUSIC_TO_MIDI_MODELS_DIR",
            state.amt_models_dir.to_string_lossy().as_ref(),
        )
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    if let Some(extra_path) = venv_cuda_lib_dir(&sidecar_dir) {
        let old = std::env::var("PATH").unwrap_or_default();
        command.env("PATH", format!("{};{}", extra_path.display(), old));
    }

    let mut child = command
        .spawn()
        .map_err(|e| format!("LYRICS_SPAWN_FAILED: {e}"))?;

    if let Some(pid) = child.id() {
        state.active_amt.lock().insert(LYRICS_NODE_ID.to_string(), pid);
    }

    use tokio::io::{AsyncBufReadExt, BufReader};
    let stdout = child.stdout.take().ok_or("LYRICS_SPAWN_FAILED: cannot take stdout")?;
    let stderr = child.stderr.take().ok_or("LYRICS_SPAWN_FAILED: cannot take stderr")?;

    let mut result_json: Option<serde_json::Value> = None;
    let mut stdout_lines: Vec<String> = Vec::new();

    {
        let mut reader = BufReader::new(stdout).lines();
        while let Some(line) = reader.next_line().await.map_err(|e| format!("LYRICS_READ_FAILED: {e}"))? {
            if let Some(rest) = line.strip_prefix("@@PROGRESS@@") {
                if let Ok(payload) = serde_json::from_str::<serde_json::Value>(rest) {
                    let _ = app.emit("amt-lyrics-progress", payload);
                }
            } else if let Some(rest) = line.strip_prefix("@@RESULT@@") {
                if let Ok(payload) = serde_json::from_str::<serde_json::Value>(rest.trim()) {
                    result_json = Some(payload);
                }
            } else if !line.is_empty() {
                stdout_lines.push(line);
            }
        }
    }

    let status = child.wait().await.map_err(|e| format!("LYRICS_WAIT_FAILED: {e}"))?;
    state.active_amt.lock().remove(LYRICS_NODE_ID);
    let _ = std::fs::remove_file(&config_path);

    let stderr_text = {
        let mut reader = BufReader::new(stderr).lines();
        let mut out = String::new();
        while let Some(line) = reader.next_line().await.unwrap_or(None) {
            out.push_str(&line);
            out.push('\n');
        }
        out
    };

    if !status.success() {
        let combined = format!("{}\n{}", stdout_lines.join("\n"), stderr_text);
        if combined.contains("LYRICS_DEP_MISSING") {
            return Err("LYRICS_DEP_MISSING".into());
        }
        return Err(format!(
            "LYRICS_EXTRACT_FAILED: exit code {:?}\n{}",
            status.code(),
            combined
        ));
    }

    let res = result_json.ok_or_else(|| {
        format!(
            "LYRICS_NO_RESULT: sidecar finished without emitting @@RESULT@@\n{}\n{}",
            stdout_lines.join("\n"),
            stderr_text
        )
    })?;

    let segments = res
        .get("segments")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|s| {
                    let text = s.get("text").and_then(|t| t.as_str()).unwrap_or("").trim().to_string();
                    if text.is_empty() {
                        return None;
                    }
                    Some(AmtLyricSegment {
                        start: s.get("start").and_then(|v| v.as_f64()).unwrap_or(0.0),
                        end: s.get("end").and_then(|v| v.as_f64()).unwrap_or(0.0),
                        text,
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    Ok(AmtLyricsResult {
        language: res
            .get("language")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        language_probability: res
            .get("language_probability")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0),
        duration: res.get("duration").and_then(|v| v.as_f64()).unwrap_or(0.0),
        device: res
            .get("device")
            .and_then(|v| v.as_str())
            .unwrap_or("cpu")
            .to_string(),
        model: model_size,
        segments,
    })
}

/// venv site-packages torch lib dir, when it exists (Windows layout).
fn venv_cuda_lib_dir(sidecar_dir: &std::path::Path) -> Option<PathBuf> {
    let dir = sidecar_dir
        .join("venv")
        .join("Lib")
        .join("site-packages")
        .join("torch")
        .join("lib");
    if dir.is_dir() {
        Some(dir)
    } else {
        None
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MIDI lyric read / write
// ─────────────────────────────────────────────────────────────────────────────

#[derive(serde::Deserialize, Clone)]
pub struct LyricLineInput {
    pub start: f64,
    pub text: String,
}

/// Build a sorted tempo map (tick, us_per_quarter) across all tracks. Always
/// starts with (0, 500000) — the SMF default — so time conversion works even
/// when no tempo meta is present.
fn collect_tempo_map(smf: &midly::Smf) -> Vec<(u64, u64)> {
    use midly::{MetaMessage, TrackEventKind};

    let mut map: Vec<(u64, u64)> = Vec::new();
    for track in &smf.tracks {
        let mut tick: u64 = 0;
        for ev in track.iter() {
            tick += ev.delta.as_int() as u64;
            if let TrackEventKind::Meta(MetaMessage::Tempo(us)) = ev.kind {
                map.push((tick, us.as_int() as u64));
            }
        }
    }
    map.sort_by_key(|(t, _)| *t);
    // Later events at the same tick win.
    map.dedup_by(|a, b| a.0 == b.0);
    let mut map = map;
    if map.first().map(|(t, _)| *t) != Some(0) {
        map.insert(0, (0, 500_000));
    }
    map
}

fn seconds_to_ticks(tempo_map: &[(u64, u64)], ppq: u32, t: f64) -> u64 {
    let target_us = t * 1_000_000.0;
    let mut cum_us: f64 = 0.0;
    let mut last_tick: u64 = 0;
    let mut us_q = tempo_map[0].1 as f64;
    for &(tick, us) in tempo_map.iter().skip(1) {
        let boundary_us = cum_us + (tick - last_tick) as f64 * us_q / ppq as f64;
        if boundary_us >= target_us {
            return (last_tick as f64 + (target_us - cum_us) * ppq as f64 / us_q).round() as u64;
        }
        cum_us = boundary_us;
        last_tick = tick;
        us_q = us as f64;
    }
    (last_tick as f64 + (target_us - cum_us) * ppq as f64 / us_q).round() as u64
}

fn ticks_to_seconds(tempo_map: &[(u64, u64)], ppq: u32, tick: u64) -> f64 {
    let mut cum_us: f64 = 0.0;
    let mut last_tick: u64 = 0;
    let mut us_q = tempo_map[0].1 as f64;
    for &(t, us) in tempo_map.iter().skip(1) {
        if t >= tick {
            break;
        }
        cum_us += (t - last_tick) as f64 * us_q / ppq as f64;
        last_tick = t;
        us_q = us as f64;
    }
    (cum_us + (tick - last_tick) as f64 * us_q / ppq as f64) / 1_000_000.0
}

/// Remove all Lyric meta events from a track, folding their deltas into the
/// following retained events so timing stays exact.
fn strip_lyric_events<'a>(track: &mut midly::Track<'a>) {
    use midly::{MetaMessage, TrackEvent, TrackEventKind};
    use midly::num::u28;

    let mut rebuilt: Vec<TrackEvent<'a>> = Vec::with_capacity(track.len());
    let mut pending_delta: u64 = 0;
    for ev in track.drain(..) {
        if matches!(ev.kind, TrackEventKind::Meta(MetaMessage::Lyric(_))) {
            pending_delta += ev.delta.as_int() as u64;
        } else {
            let delta = (ev.delta.as_int() as u64 + pending_delta).min(U28_MAX);
            rebuilt.push(TrackEvent {
                delta: u28::new(delta as u32),
                kind: ev.kind,
            });
            pending_delta = 0;
        }
    }
    *track = rebuilt;
}

/// Write timed lyric lines into a MIDI file as Lyric meta events on a
/// dedicated "Lyrics" track. Existing lyric events are stripped first, so
/// re-saving is idempotent. `out_path` may equal `midi_path` (in-place).
/// Returns the absolute path written.
#[tauri::command]
pub fn amt_write_lyrics_to_midi(
    midi_path: String,
    out_path: String,
    lyrics: Vec<LyricLineInput>,
) -> Result<String, String> {
    use midly::{MetaMessage, Smf, Timing, Track, TrackEvent, TrackEventKind};
    use midly::num::u28;

    let lyrics: Vec<(f64, String)> = lyrics
        .into_iter()
        .map(|l| (l.start, l.text.trim().to_string()))
        .filter(|(_, text)| !text.is_empty())
        .collect();
    if lyrics.is_empty() {
        return Err("LYRICS_WRITE_EMPTY".into());
    }

    let bytes = std::fs::read(&midi_path).map_err(|e| format!("LYRICS_MIDI_READ_FAILED: {e}"))?;
    let mut smf = Smf::parse(&bytes).map_err(|e| format!("LYRICS_MIDI_PARSE_FAILED: {e}"))?;

    let ppq = match smf.header.timing {
        Timing::Metrical(t) => t.as_int() as u32,
        Timing::Timecode(_, _) => return Err("LYRICS_MIDI_BAD_TIMING".into()),
    };

    // Strip old lyrics from every track, then append a fresh Lyrics track.
    for track in smf.tracks.iter_mut() {
        strip_lyric_events(track);
    }

    let tempo_map = collect_tempo_map(&smf);
    let mut timed: Vec<(u64, String)> = lyrics
        .iter()
        .map(|(start, text)| (seconds_to_ticks(&tempo_map, ppq, *start), text.clone()))
        .collect();
    timed.sort_by_key(|(tick, _)| *tick);

    let mut lyric_track: Track<'static> = Vec::new();
    lyric_track.push(TrackEvent {
        delta: u28::new(0),
        kind: TrackEventKind::Meta(MetaMessage::TrackName(b"Lyrics")),
    });
    let mut prev_tick: u64 = 0;
    for (tick, text) in timed {
        let delta = tick.saturating_sub(prev_tick).min(U28_MAX);
        // Track<'static> requires 'static bytes — leak the (tiny) lyric text.
        let text_bytes: &'static [u8] = Box::leak(text.into_bytes().into_boxed_slice());
        lyric_track.push(TrackEvent {
            delta: u28::new(delta as u32),
            kind: TrackEventKind::Meta(MetaMessage::Lyric(text_bytes)),
        });
        prev_tick = tick;
    }
    lyric_track.push(TrackEvent {
        delta: u28::new(0),
        kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
    });
    smf.tracks.push(lyric_track);

    // Preserve the original header (format + PPQ).
    let out_pb = PathBuf::from(&out_path);
    if let Some(parent) = out_pb.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("LYRICS_WRITE_DIR_FAILED: {e}"))?;
    }
    smf.save(&out_pb)
        .map_err(|e| format!("LYRICS_WRITE_FAILED: {e}"))?;
    Ok(std::path::absolute(&out_pb)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or(out_path))
}

/// Read Lyric meta events from a MIDI file back into seconds-timed lines so
/// the lyrics editor round-trips. Returns lines sorted by start time.
#[tauri::command]
pub fn amt_read_lyrics_from_midi(midi_path: String) -> Result<Vec<AmtLyricSegment>, String> {
    use midly::{MetaMessage, Smf, Timing, TrackEventKind};

    let bytes = std::fs::read(&midi_path).map_err(|e| format!("LYRICS_MIDI_READ_FAILED: {e}"))?;
    let smf = Smf::parse(&bytes).map_err(|e| format!("LYRICS_MIDI_PARSE_FAILED: {e}"))?;

    let ppq = match smf.header.timing {
        Timing::Metrical(t) => t.as_int() as u32,
        Timing::Timecode(_, _) => return Err("LYRICS_MIDI_BAD_TIMING".into()),
    };
    let tempo_map = collect_tempo_map(&smf);

    let mut lines: Vec<(u64, String)> = Vec::new();
    for track in &smf.tracks {
        let mut tick: u64 = 0;
        for ev in track.iter() {
            tick += ev.delta.as_int() as u64;
            if let TrackEventKind::Meta(MetaMessage::Lyric(text)) = ev.kind {
                let text = String::from_utf8_lossy(text).trim().to_string();
                if text.is_empty() {
                    continue;
                }
                lines.push((tick, text));
            }
        }
    }
    lines.sort_by_key(|(tick, _)| *tick);
    // Lyric events carry no duration — each line ends where the next starts.
    let mut out: Vec<AmtLyricSegment> = lines
        .iter()
        .map(|(tick, text)| AmtLyricSegment {
            start: ticks_to_seconds(&tempo_map, ppq, *tick),
            end: 0.0,
            text: text.clone(),
        })
        .collect();
    for i in 0..out.len() {
        let next = out.get(i + 1).map(|n| n.start).unwrap_or(f64::MAX);
        out[i].end = next.max(out[i].start + 0.001);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal one-track MIDI with two tempo sections:
    /// ticks 0..4800 at 500000 us/q (120 bpm), then 250000 us/q (240 bpm).
    fn build_test_midi(path: &std::path::Path) {
        use midly::num::{u15, u24, u28};
        use midly::{Header, MetaMessage, Smf, Timing, TrackEvent, TrackEventKind};

        let mut smf = Smf {
            header: Header {
                format: midly::Format::Parallel,
                timing: Timing::Metrical(u15::new(480)),
            },
            tracks: vec![Vec::new()],
        };
        let track = &mut smf.tracks[0];
        track.push(TrackEvent {
            delta: u28::new(0),
            kind: TrackEventKind::Meta(MetaMessage::Tempo(u24::new(500_000))),
        });
        track.push(TrackEvent {
            delta: u28::new(4800),
            kind: TrackEventKind::Meta(MetaMessage::Tempo(u24::new(250_000))),
        });
        track.push(TrackEvent {
            delta: u28::new(0),
            kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
        });
        smf.save(path).unwrap();
    }

    #[test]
    fn lyrics_roundtrip_with_tempo_change() {
        let tmp = std::env::temp_dir().join("utai_lyrics_roundtrip_test.mid");
        build_test_midi(&tmp);
        let path = tmp.to_string_lossy().into_owned();

        let lyrics = vec![
            LyricLineInput { start: 2.5, text: "窗头灯亮到了天".into() },
            LyricLineInput { start: 7.5, text: "You turned away".into() },
            LyricLineInput { start: 0.5, text: "out of order line".into() },
        ];
        amt_write_lyrics_to_midi(path.clone(), path.clone(), lyrics.clone()).unwrap();

        let back = amt_read_lyrics_from_midi(path.clone()).unwrap();
        assert_eq!(back.len(), 3, "all three lines round-tripped");

        // Read-back is sorted by start; the 0.5s line comes first.
        assert_eq!(back[0].text, "out of order line");
        assert_eq!(back[1].text, "窗头灯亮到了天");
        assert_eq!(back[2].text, "You turned away");
        // Read-back is sorted; compare against sorted expected starts.
        let mut want_starts: Vec<f64> = lyrics.iter().map(|l| l.start).collect();
        want_starts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (want, got) in want_starts.into_iter().zip(back.iter().map(|l| l.start)) {
            assert!(
                (want - got).abs() < 0.05,
                "time drifted: wrote {want}s, read {got}s"
            );
        }
        // End of a line = start of the next.
        assert!(back[0].end >= back[0].start);
        assert!((back[0].end - back[1].start).abs() < 0.05);

        // Re-saving over the same file is idempotent (old lyrics stripped).
        let lyrics2 = vec![LyricLineInput { start: 1.0, text: "replaced".into() }];
        amt_write_lyrics_to_midi(path.clone(), path.clone(), lyrics2).unwrap();
        let back2 = amt_read_lyrics_from_midi(path).unwrap();
        assert_eq!(back2.len(), 1);
        assert_eq!(back2[0].text, "replaced");
        assert!((back2[0].start - 1.0).abs() < 0.05);

        let _ = std::fs::remove_file(&tmp);
    }
}
