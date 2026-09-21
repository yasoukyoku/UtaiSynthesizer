//! AMT (Automatic Music Transcription) MIDI sidecar commands.
//!
//! Spawns the Python-based amt_sidecar.py for audio-to-MIDI transcription.
//! The sidecar is expected at `data/amt/python/` with its venv.
//!
//! The sidecar emits two kinds of JSON-lines to stdout:
//!   `@@PROGRESS@@{"stage":...,"progress":...,"total":...,"message":...}`
//!   `@@RESULT@@  {"midi_path":...,"stem_midi_paths":...,"merged_midi_path":...,...}`
//! This command streams stdout line-by-line, re-emits progress as the Tauri
//! event `amt-progress`, parses the final result, SHA-256-verifies the main
//! MIDI artifact, and returns a rich result to the frontend.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use tauri::{AppHandle, Emitter, Manager, State};

use crate::AppState;

#[derive(serde::Serialize, Clone)]
pub struct AmtMidiResult {
    pub midi_path: String,
    pub processing_time_secs: f64,
    pub total_notes: Option<u64>,
    pub vocal_midi_path: Option<String>,
    pub accompaniment_midi_path: Option<String>,
    pub merged_midi_path: Option<String>,
    pub stem_midi_paths: Option<HashMap<String, String>>,
    pub separated_audio: Option<HashMap<String, String>>,
    pub transcription_backend: Option<String>,
    pub mode: Option<String>,
    pub sha256: String,
}

#[derive(serde::Serialize, Clone)]
struct AmtProgressEvent<'a> {
    node_id: Option<&'a str>,
    stage: Option<&'a str>,
    progress: f64,
    total: f64,
    message: Option<&'a str>,
}

/// All seven processing modes supported by the sidecar. Anything else is
/// rejected before the sidecar is spawned.
const VALID_MODES: &[&str] = &[
    "smart",
    "vocal_split",
    "six_stem_split",
    "piano_transkun",
    "piano_transkun_v2_aug",
    "piano_aria_amt",
    "piano_bytedance_pedal",
];

const VALID_BACKENDS: &[&str] = &["yourmt3", "miros", "muscriptor"];
// Must mirror YourMT3Model in the Python data_models EXACTLY. "miros_streaming_amt" was
// once accepted here but Config.validate() rejects it — it silently leaked through and
// killed the run inside the sidecar. The MIROS engine is selected by the backend, never
// by yourmt3_model.
const VALID_YOURMT3_MODELS: &[&str] = &[
    "ymt3_plus",
    "yptf_single_nops",
    "yptf_multi_ps",
    "yptf_moe_multi_nops",
    "yptf_moe_multi_ps",
    "mc13_256_all_cross_v6",
];
// Must mirror MuscriptorModel EXACTLY (small|medium|large). Catalog ids like
// "muscriptor_large" are stripped by the frontend; keep them OUT so a raw pass-through
// falls back to the default instead of reaching Python's validate() and failing.
const VALID_MUSCRIPTOR_MODELS: &[&str] = &["small", "medium", "large"];
const VALID_TRACK_MODES: &[&str] = &["multi_track", "single_track"];
// Python TempoMode only knows these three. There is no "none" tempo mode — the
// "don't quantize" intent is quantize_grid="none".
const VALID_TEMPO_MODES: &[&str] = &["adaptive", "fixed_auto", "fixed_manual"];
const VALID_QUANTIZE_GRIDS: &[&str] = &["none", "1/4", "1/8", "1/16", "1/32", "1/64"];

fn pick<'a>(value: &'a Option<String>, valid: &[&str], default: &'a str) -> &'a str {
    match value {
        Some(v) if valid.iter().any(|x| x == &v.to_lowercase()) => v,
        _ => default,
    }
}

pub(crate) fn resolve_sidecar_dir(state: &AppState) -> std::path::PathBuf {
    // Look in current data_dir first (custom location)
    let data_dir = state.cache_dir.parent().unwrap_or(&state.cache_dir);
    let sidecar_dir = data_dir.join("amt").join("python");
    if sidecar_dir.join("amt_sidecar.py").exists() {
        return sidecar_dir;
    }

    // Fallback to app_dir/data (default location)
    let app_data_dir = state.app_dir.join("data");
    let sidecar_dir = app_data_dir.join("amt").join("python");
    if sidecar_dir.join("amt_sidecar.py").exists() {
        return sidecar_dir;
    }

    // Fallback to app_dir/amt (alternative location)
    let sidecar_dir = state.app_dir.join("amt").join("python");
    sidecar_dir
}

pub(crate) fn resolve_amt_model_path(state: &AppState, filename: &str, alt_filename: Option<&str>) -> Option<PathBuf> {
    let local_path = state.amt_models_dir.join(filename);
    if local_path.exists() {
        return Some(local_path);
    }
    if let Some(alt) = alt_filename {
        let alt_local = state.amt_models_dir.join(alt);
        if alt_local.exists() {
            return Some(alt_local);
        }
    }
    None
}

/// Resolve the Python interpreter for the AMT sidecar.
///
/// The AMT sidecar ships with its OWN venv at `<sidecar_dir>/venv` containing
/// all transcription dependencies (PyTorch, librosa, transkun, beat_this, etc.).
/// That venv is the ONLY interpreter that can import the AMT modules — the
/// shared `converter_python()` resolves to the MSST/converter runtime venv (or
/// a bare `python` on PATH), which lacks these deps and fails on the first
/// import. Prefer the AMT venv; fall back to `converter_python` only when the
/// dedicated venv is absent (e.g. a fresh checkout before the runtime is set up).
pub(crate) fn resolve_amt_python(sidecar_dir: &std::path::Path, app_dir: &std::path::Path) -> PathBuf {
    // Windows: venv/Scripts/python.exe | Unix: venv/bin/python
    #[cfg(windows)]
    let rel = ["venv", "Scripts", "python.exe"];
    #[cfg(not(windows))]
    let rel = ["venv", "bin", "python"];

    let amt_venv_python = rel.iter().fold(sidecar_dir.to_path_buf(), |acc, p| acc.join(p));
    if amt_venv_python.exists() {
        return amt_venv_python;
    }
    // Fallback: the shared converter/runtime interpreter (will lack AMT deps,
    // but keeps the sidecar-spawn path working so the error surfaces from
    // Python itself rather than a "python not found" spawn failure).
    crate::pyenv::converter_python(app_dir)
}

/// Run the AMT sidecar to transcribe audio to MIDI via the `primary` command.
///
/// The sidecar must be at `data/amt/python/amt_sidecar.py` with its source
/// tree at `data/amt/python/src/`.  If the sidecar is not found, the command
/// returns `AMT_SIDECAR_NOT_FOUND`.
#[tauri::command]
pub async fn run_amt_midi(
    app: AppHandle,
    state: State<'_, Arc<AppState>>,
    audio_path: String,
    midi_mode: String,
    transcription_backend: Option<String>,
    yourmt3_model: Option<String>,
    muscriptor_model: Option<String>,
    midi_track_mode: Option<String>,
    tempo_mode: Option<String>,
    custom_bpm: Option<f64>,
    quantize_notes: Option<bool>,
    quantize_grid: Option<String>,
    use_gpu: bool,
    gpu_device: i32,
    output_dir: String,
    node_id: Option<String>,
    muscriptor_instruments: Option<Vec<String>>,
    muscriptor_processing_chain: Option<String>,
    vocal_split_merge_midi: Option<bool>,
    remove_duplicates: Option<bool>,
    velocity_smoothing: Option<bool>,
    max_polyphony: Option<u32>,
    ticks_per_beat: Option<u32>,
    default_velocity: Option<u8>,
    start_time_ms: Option<f64>,
    duration_ms: Option<f64>,
) -> Result<AmtMidiResult, String> {
    let mode = midi_mode.trim().to_lowercase();
    if !VALID_MODES.contains(&mode.as_str()) {
        return Err(format!("AMT_INVALID_MODE: unsupported midi_mode {midi_mode:?}"));
    }

    let sidecar_dir = resolve_sidecar_dir(&state);
    let sidecar_script = sidecar_dir.join("amt_sidecar.py");
    let python = resolve_amt_python(&sidecar_dir, &state.app_dir);

    if !sidecar_script.exists() {
        return Err(format!(
            "AMT_SIDECAR_NOT_FOUND: sidecar not found at {}. Please install the AMT runtime first.",
            sidecar_script.display()
        ));
    }

    // Resolve FluidSynth and SoundFont paths. The soundfont goes through the SHARED
    // active-soundfont resolver (P0-C): user selection in soundfonts/ wins, built-in
    // MuseScore_General.sf2 is the fallback — conversion playback, edit re-render and
    // every export path use the SAME source.
    let amt_models_dir = &state.amt_models_dir;
    let fluidsynth_exe = resolve_amt_model_path(&state, "fluidsynth/2.5.6/bin/fluidsynth.exe", None)
        .unwrap_or_else(|| amt_models_dir.join("fluidsynth").join("2.5.6").join("bin").join("fluidsynth.exe"));
    let soundfont_path = {
        let resolved = crate::commands::amt_export::resolve_active_soundfont(amt_models_dir);
        if resolved.is_file() {
            resolved
        } else {
            resolve_amt_model_path(&state, "MuseScore_General.sf2", None)
                .unwrap_or_else(|| amt_models_dir.join("MuseScore_General.sf2"))
        }
    };

    // Multi-instrument modes (smart / splits) require a real backend; piano
    // modes ignore it but Config.validate() still demands a valid value.
    let is_multi = matches!(
        mode.as_str(),
        "smart" | "vocal_split" | "six_stem_split"
    );
    let backend = if is_multi {
        pick(&transcription_backend, VALID_BACKENDS, "yourmt3").to_string()
    } else {
        "yourmt3".to_string()
    };

    // Build a sidecar-compatible Config JSON. Every field must satisfy
    // Config.validate() or the sidecar raises before doing any work.
    let config = serde_json::json!({
        "processing_mode": mode,
        "transcription_backend": backend,
        "multi_instrument_model": backend,
        "yourmt3_model": pick(&yourmt3_model, VALID_YOURMT3_MODELS, "yptf_moe_multi_nops"),
        "muscriptor_model": pick(&muscriptor_model, VALID_MUSCRIPTOR_MODELS, "large"),
        "muscriptor_instruments": muscriptor_instruments.unwrap_or_default(),
        "muscriptor_processing_chain": muscriptor_processing_chain.unwrap_or_else(|| "official".to_string()),
        "vocal_split_merge_midi": vocal_split_merge_midi.unwrap_or(false),
        "midi_track_mode": pick(&midi_track_mode, VALID_TRACK_MODES, "multi_track"),
        "tempo_mode": pick(&tempo_mode, VALID_TEMPO_MODES, "fixed_auto"),
        "custom_bpm": custom_bpm,
        "quantize_notes": quantize_notes.unwrap_or(true),
        "quantize_grid": pick(&quantize_grid, VALID_QUANTIZE_GRIDS, "1/32"),
        "remove_duplicates": remove_duplicates.unwrap_or(true),
        "velocity_smoothing": velocity_smoothing.unwrap_or(true),
        "max_polyphony": max_polyphony.unwrap_or(40),
        "ticks_per_beat": ticks_per_beat.unwrap_or(480),
        "default_velocity": default_velocity.unwrap_or(80),
        "use_gpu": use_gpu,
        "gpu_device": gpu_device,
        "output_dir": "",
        "save_separated_tracks": true,
        "start_time_ms": start_time_ms,
        "duration_ms": duration_ms,
    });

    let config_filename = format!("amt_run_config_{}.json", uuid::Uuid::new_v4());
    let config_path = sidecar_dir.join(&config_filename);
    let config_text = serde_json::to_string_pretty(&config)
        .map_err(|e| format!("AMT_CONFIG_ERROR: {e}"))?;
    std::fs::write(&config_path, &config_text)
        .map_err(|e| format!("AMT_CONFIG_WRITE_FAILED: {e}"))?;

    let start = std::time::Instant::now();

    // Ensure output dir exists so the sidecar can write {stem}.mid into it.
    std::fs::create_dir_all(&output_dir)
        .map_err(|e| format!("AMT_OUTPUT_DIR_FAILED: {e}"))?;

    let mut child = tokio::process::Command::new(&python)
        .arg(&sidecar_script)
        .arg("primary")
        .arg("--audio")
        .arg(&audio_path)
        .arg("--outdir")
        .arg(&output_dir)
        .arg("--config")
        .arg(format!("@{}", config_path.to_string_lossy()))
        .current_dir(&sidecar_dir)
        .env("PYTHONIOENCODING", "utf-8")
        .env("MUSIC_TO_MIDI_FLUIDSYNTH", fluidsynth_exe.to_string_lossy().as_ref())
        .env("MUSIC_TO_MIDI_SOUNDFONT", soundfont_path.to_string_lossy().as_ref())
        .env("MUSIC_TO_MIDI_MODELS_DIR", amt_models_dir.to_string_lossy().as_ref())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("AMT_SPAWN_FAILED: {e}"))?;

    // Register the running sidecar under `node_id` so `cancel_amt_midi` can force-kill it.
    if let (Some(nid), Some(pid)) = (&node_id, child.id()) {
        state.active_amt.lock().insert(nid.clone(), pid);
    }

    use tokio::io::{AsyncBufReadExt, BufReader};
    let stdout = child.stdout.take().ok_or("AMT_SPAWN_FAILED: cannot take stdout")?;
    let stderr = child.stderr.take().ok_or("AMT_SPAWN_FAILED: cannot take stderr")?;

    let mut result_json: Option<serde_json::Value> = None;
    let mut stdout_lines: Vec<String> = Vec::new();

    // Stream stdout: emit @@PROGRESS@@ as Tauri events, capture @@RESULT@@.
    {
        let mut reader = BufReader::new(stdout).lines();
        while let Some(line) = reader.next_line().await.map_err(|e| format!("AMT_READ_FAILED: {e}"))? {
            if let Some(rest) = line.strip_prefix("@@PROGRESS@@") {
                if let Ok(payload) = serde_json::from_str::<serde_json::Value>(rest) {
                    let stage = payload.get("stage").and_then(|v| v.as_str());
                    let progress = payload.get("progress").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let total = payload.get("total").and_then(|v| v.as_f64()).unwrap_or(1.0);
                    let message = payload.get("message").and_then(|v| v.as_str());
                    let _ = app.emit(
                        "amt-progress",
                        AmtProgressEvent {
                            node_id: node_id.as_deref(),
                            stage,
                            progress,
                            total,
                            message,
                        },
                    );
                }
            } else if let Some(rest) = line.strip_prefix("@@RESULT@@") {
                let rest = rest.trim_start();
                if let Ok(payload) = serde_json::from_str::<serde_json::Value>(rest) {
                    result_json = Some(payload);
                }
            } else if !line.is_empty() {
                stdout_lines.push(line);
            }
        }
    }

    let status = child.wait().await.map_err(|e| format!("AMT_WAIT_FAILED: {e}"))?;
    // AMT sidecar finished (success or failure) — release it from the cancel registry.
    if let Some(nid) = &node_id {
        state.active_amt.lock().remove(nid);
    }
    let elapsed = start.elapsed().as_secs_f64();

    // Clean up temp config
    let _ = std::fs::remove_file(&config_path);

    // Drain stderr for diagnostics.
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
        return Err(format!(
            "AMT_TRANSCRIPTION_FAILED: exit code {:?}\nstdout: {}\nstderr: {}",
            status.code(),
            stdout_lines.join("\n"),
            stderr_text,
        ));
    }

    let result = result_json.ok_or_else(|| {
        format!(
            "AMT_NO_RESULT: sidecar finished without emitting @@RESULT@@\nstdout: {}\nstderr: {}",
            stdout_lines.join("\n"),
            stderr_text,
        )
    })?;

    let midi_path = result
        .get("midi_path")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    // Separation-only modes (vocal_split / six_stem_split) return an empty
    // midi_path and populate separated_audio instead — that's valid, so only
    // SHA-verify when a MIDI artifact was actually produced.
    let mut sha256 = String::new();
    if !midi_path.is_empty() {
        let midi_pb = PathBuf::from(&midi_path);
        if !midi_pb.exists() {
            return Err(format!(
                "AMT_OUTPUT_NOT_FOUND: transcription completed but output file not found at {midi_path}"
            ));
        }
        sha256 = crate::download::sha256_file(&midi_pb)
            .map_err(|e| format!("AMT_SHA_FAILED: {e}"))?;
    }

    let get_opt_str = |key: &str| -> Option<String> {
        result.get(key).and_then(|v| v.as_str()).map(|s| s.to_string())
    };
    let get_opt_map = |key: &str| -> Option<HashMap<String, String>> {
        result.get(key).and_then(|v| v.as_object()).map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                .collect()
        })
    };

    Ok(AmtMidiResult {
        midi_path,
        processing_time_secs: result
            .get("processing_time")
            .and_then(|v| v.as_f64())
            .unwrap_or(elapsed),
        total_notes: result
            .get("total_notes")
            .and_then(|v| v.as_u64())
            .or_else(|| result.get("total_notes").and_then(|v| v.as_f64().map(|f| f as u64))),
        vocal_midi_path: get_opt_str("vocal_midi_path"),
        accompaniment_midi_path: get_opt_str("accompaniment_midi_path"),
        merged_midi_path: get_opt_str("merged_midi_path"),
        stem_midi_paths: get_opt_map("stem_midi_paths"),
        separated_audio: get_opt_map("separated_audio"),
        transcription_backend: get_opt_str("transcription_backend"),
        mode: get_opt_str("mode"),
        sha256,
    })
}

/// Force-stop a running AMT sidecar by its registered `node_id` (the id the
/// frontend passed to `run_amt_midi`). The OS terminates the child; the in-flight
/// `run_amt_midi` then settles with a non-success exit, which the engine treats as
/// a cancellation. Safe to call when nothing is registered (returns Ok).
#[tauri::command]
pub fn cancel_amt_midi(state: State<'_, Arc<AppState>>, node_id: String) -> Result<(), String> {
    let pid = state.active_amt.lock().remove(&node_id);
    match pid {
        Some(pid) => kill_pid(pid).map_err(|e| format!("AMT_KILL_FAILED: {e}")),
        None => Ok(()),
    }
}

/// Force-stop ALL active AMT sidecars. Used by the generic workflow Stop button
/// (it doesn't know which node is currently rendering AMT). Idempotent.
#[tauri::command]
pub fn cancel_amt_all(state: State<'_, Arc<AppState>>) -> Result<(), String> {
    let pids: Vec<u32> = state.active_amt.lock().values().copied().collect();
    for pid in pids {
        let _ = kill_pid(pid);
    }
    state.active_amt.lock().clear();
    Ok(())
}

pub(crate) fn kill_pid(pid: u32) -> std::io::Result<()> {
    // Windows: taskkill /F /T terminates the process tree (the sidecar may spawn
    // its own Python worker children). Unix: SIGKILL the process group would be
    // ideal but a plain SIGKILL on the sidecar's own pid is enough.
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("taskkill")
            .args(["/T", "/F", "/PID", &pid.to_string()])
            .status()?;
        Ok(())
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        std::process::Command::new("kill")
            .arg("-9")
            .arg(pid.to_string())
            .status()?;
        Ok(())
    }
    #[cfg(not(any(target_os = "windows", unix)))]
    {
        Err(std::io::Error::other("unsupported platform for process kill"))
    }
}

/// Alias for run_amt_midi to match frontend call pattern
#[derive(serde::Serialize, Clone)]
pub struct AmtPlaybackResult {
    pub duration: f64,
    pub transcription_wav: String,
    pub original_wav: String,
    pub midi_gain_db: f64,
    pub instrument_wavs: HashMap<String, String>,
}

/// Prepare audio assets for interactive MIDI playback via FluidSynth.
#[tauri::command]
pub async fn amt_prepare_playback(
    app: AppHandle,
    state: State<'_, Arc<AppState>>,
    midi_path: String,
    audio_path: String,
    output_dir: String,
) -> Result<AmtPlaybackResult, String> {
    let sidecar_dir = resolve_sidecar_dir(&state);
    let sidecar_script = sidecar_dir.join("amt_sidecar.py");
    let python = resolve_amt_python(&sidecar_dir, &state.app_dir);

    // The sidecar chdirs to its own folder at startup, so RELATIVE paths passed from the
    // frontend would resolve against the wrong base and fail with FileNotFoundError.
    // Absolutize everything against the app's current working directory first.
    let absolutize = |p: &str| -> String {
        let pb = std::path::PathBuf::from(p);
        if pb.is_absolute() {
            pb.to_string_lossy().into_owned()
        } else {
            std::path::absolute(&pb)
                .map(|a| a.to_string_lossy().into_owned())
                .unwrap_or_else(|_| pb.to_string_lossy().into_owned())
        }
    };
    let midi_path = absolutize(&midi_path);
    let audio_path = absolutize(&audio_path);
    let output_dir = absolutize(&output_dir);

    // Resolve FluidSynth and SoundFont paths — the soundfont honors the SHARED active
    // selection (P0-C) so interactive playback re-renders with the same source the
    // exports use.
    let amt_models_dir = &state.amt_models_dir;
    let fluidsynth_exe = resolve_amt_model_path(&state, "fluidsynth/2.5.6/bin/fluidsynth.exe", None)
        .unwrap_or_else(|| amt_models_dir.join("fluidsynth").join("2.5.6").join("bin").join("fluidsynth.exe"));
    let soundfont_path = {
        let resolved = crate::commands::amt_export::resolve_active_soundfont(amt_models_dir);
        if resolved.is_file() {
            resolved
        } else {
            resolve_amt_model_path(&state, "MuseScore_General.sf2", None)
                .unwrap_or_else(|| amt_models_dir.join("MuseScore_General.sf2"))
        }
    };

    if !sidecar_script.exists() {
        return Err("AMT_SIDECAR_NOT_FOUND".into());
    }

    // Prepare a minimal config for the sidecar
    let config = serde_json::json!({
        "processing_mode": "smart",
        "transcription_backend": "yourmt3",
    });

    let config_path = sidecar_dir.join("amt_playback_config.json");
    let config_text = serde_json::to_string(&config).unwrap();
    std::fs::write(&config_path, &config_text).map_err(|e| e.to_string())?;

    let mut child = tokio::process::Command::new(&python)
        .arg(&sidecar_script)
        .arg("prepare_playback")
        .arg("--midi")
        .arg(&midi_path)
        .arg("--audio")
        .arg(&audio_path)
        .arg("--outdir")
        .arg(&output_dir)
        .arg("--config")
        .arg(format!("@{}", config_path.to_string_lossy()))
        .current_dir(&sidecar_dir)
        .env("PYTHONIOENCODING", "utf-8")
        .env("MUSIC_TO_MIDI_FLUIDSYNTH", fluidsynth_exe.to_string_lossy().as_ref())
        .env("MUSIC_TO_MIDI_SOUNDFONT", soundfont_path.to_string_lossy().as_ref())
        .env("MUSIC_TO_MIDI_MODELS_DIR", amt_models_dir.to_string_lossy().as_ref())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("AMT_SPAWN_FAILED: {e}"))?;

    use tokio::io::{AsyncBufReadExt, BufReader};
    let stdout = child.stdout.take().ok_or("AMT_SPAWN_FAILED: cannot take stdout")?;
    let stderr = child.stderr.take().ok_or("AMT_SPAWN_FAILED: cannot take stderr")?;

    let mut result_json: Option<serde_json::Value> = None;
    let mut stdout_lines: Vec<String> = Vec::new();

    {
        let mut reader = BufReader::new(stdout).lines();
        while let Some(line) = reader.next_line().await.map_err(|e| e.to_string())? {
            if let Some(rest) = line.strip_prefix("@@PROGRESS@@") {
                if let Ok(payload) = serde_json::from_str::<serde_json::Value>(rest) {
                    let _ = app.emit("amt-progress", payload);
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

    let status = child.wait().await.map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&config_path);

    // Drain stderr for diagnostics.
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
        return Err(format!(
            "AMT_PLAYBACK_PREP_FAILED: exit code {:?}\nstdout: {}\nstderr: {}",
            status.code(),
            stdout_lines.join("\n"),
            stderr_text,
        ));
    }

    let res = result_json.ok_or("AMT_NO_RESULT")?;
    
    Ok(AmtPlaybackResult {
        duration: res.get("duration").and_then(|v| v.as_f64()).unwrap_or(0.0),
        transcription_wav: res.get("transcription_wav").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        original_wav: res.get("original_wav").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        midi_gain_db: res.get("midi_gain_db").and_then(|v| v.as_f64()).unwrap_or(0.0),
        instrument_wavs: res.get("instrument_wavs")
            .and_then(|v| v.as_object())
            .map(|obj| {
                obj.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect()
            })
            .unwrap_or_default(),
    })
}

#[tauri::command]
pub async fn amt_export_zip(
    midi_paths: Vec<String>,
    save_path: String,
) -> Result<(), String> {
    use std::io::Write;
    
    // Check if parent directory exists
    let save_pb = std::path::PathBuf::from(&save_path);
    if let Some(parent) = save_pb.parent() {
        if !parent.exists() {
            return Err(format!("EXPORT_DIR_NOT_FOUND: {}", parent.display()));
        }
    }

    let file = std::fs::File::create(&save_path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::PermissionDenied {
            "EXPORT_PERMISSION_DENIED: File might be in use by another program".to_string()
        } else {
            format!("CREATE_ZIP_FAILED: {e}")
        }
    })?;
    
    let mut zip = zip::ZipWriter::new(file);
    let opts = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);

    let mut added_count = 0;
    for path in midi_paths {
        let pb = std::path::PathBuf::from(&path);
        if !pb.exists() { continue; }
        
        let name = pb.file_name().unwrap_or_default().to_string_lossy();
        zip.start_file(name.to_string(), opts).map_err(|e| format!("ZIP_ENTRY_FAILED: {e}"))?;
        
        let bytes = std::fs::read(&pb).map_err(|e| format!("READ_MIDI_FAILED: {e} ({})", pb.display()))?;
        zip.write_all(&bytes).map_err(|e| format!("WRITE_ZIP_FAILED: {e}"))?;
        added_count += 1;
    }
    
    if added_count == 0 {
        return Err("EXPORT_NO_FILES: No valid MIDI files to export".to_string());
    }

    zip.finish().map_err(|e| format!("FINALIZE_ZIP_FAILED: {e}"))?;
    Ok(())
}

/// Export EVERY track of a merged MIDI file as its own separate .mid file into a chosen folder.
/// Smart-mode transcription returns ONE merged multi-track .mid (each track = one instrument), so
/// "导出全部轨道" needs this splitter. Files are named `<base>_<instrumentName>.mid`.
#[tauri::command]
pub fn export_midi_tracks_to_folder(
    midi_path: String,
    folder: String,
) -> Result<Vec<String>, String> {
    use midly::{MetaMessage, Smf, TrackEventKind};

    let src_pb = std::path::PathBuf::from(&midi_path);
    if !src_pb.is_file() {
        return Err(format!("EXPORT_SRC_NOT_FOUND: {}", src_pb.display()));
    }
    let bytes = std::fs::read(&src_pb).map_err(|e| format!("EXPORT_READ_FAILED: {e}"))?;
    let smf = Smf::parse(&bytes).map_err(|e| format!("EXPORT_PARSE_FAILED: {e}"))?;

    let folder_pb = std::path::PathBuf::from(&folder);
    std::fs::create_dir_all(&folder_pb).map_err(|e| format!("EXPORT_DIR_FAILED: {e}"))?;

    let base = src_pb
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "merged".to_string());

    let mut written: Vec<String> = Vec::new();
    let mut used_names: std::collections::HashSet<String> = std::collections::HashSet::new();

    for (idx, track) in smf.tracks.iter().enumerate() {
        if track.is_empty() {
            continue;
        }
        // Prefer the first TrackName meta as the instrument name.
        let mut name: Option<String> = None;
        for ev in track.iter() {
            if let TrackEventKind::Meta(MetaMessage::TrackName(n)) = ev.kind {
                if !n.is_empty() {
                    name = Some(String::from_utf8_lossy(n).into_owned());
                }
                break;
            }
        }
        let raw = name.unwrap_or_else(|| format!("track_{}", idx + 1));
        // Sanitize for a filename.
        let safe: String = raw
            .chars()
            .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' || c == ' ' { c } else { '_' })
            .collect::<String>()
            .trim()
            .to_string();
        let safe = if safe.is_empty() {
            format!("track_{}", idx + 1)
        } else {
            safe
        };

        // Avoid filename collision (two tracks with the same instrument name).
        let mut leaf = format!("{}_{}.mid", base, safe);
        let mut n = 2;
        while used_names.contains(&leaf) {
            leaf = format!("{}_{}_{}.mid", base, safe, n);
            n += 1;
        }
        used_names.insert(leaf.clone());

        let out_pb = folder_pb.join(&leaf);
        let out_smf = Smf {
            header: smf.header,
            tracks: vec![track.clone()],
        };
        out_smf
            .save(&out_pb)
            .map_err(|e| format!("EXPORT_WRITE_FAILED: {e} ({})", out_pb.display()))?;
        written.push(out_pb.to_string_lossy().into_owned());
    }

    if written.is_empty() {
        return Err("EXPORT_NO_FILES: No tracks found in the MIDI file".to_string());
    }
    Ok(written)
}

/// Add a directory to the asset-protocol scope at RUNTIME. A right-click "音乐转MIDI"
/// conversion writes its outputs NEXT TO the source file (e.g. Downloads/song_midi/),
/// which neither the static scope (AppData only) nor the boot-time data-root extension
/// can know. The result panel calls this with its outputDir so convertFileSrc() preview
/// URLs actually resolve instead of being silently blocked.
#[tauri::command]
pub fn allow_asset_dir(app: AppHandle, dir: String) -> Result<(), String> {
    let pb = std::path::PathBuf::from(&dir);
    app.asset_protocol_scope()
        .allow_directory(pb, true)
        .map_err(|e| format!("ASSET_SCOPE_FAILED: {e}"))
}

/// Check if the AMT sidecar is installed.
#[tauri::command]
pub fn amt_sidecar_installed(state: State<'_, Arc<AppState>>) -> bool {
    resolve_sidecar_dir(&state)
        .join("amt_sidecar.py")
        .exists()
}

/// Get the sidecar's expected directory path.
#[tauri::command]
pub fn amt_sidecar_path(state: State<'_, Arc<AppState>>) -> String {
    resolve_sidecar_dir(&state)
        .to_string_lossy()
        .to_string()
}

#[derive(serde::Serialize, Clone)]
pub struct AmtMidiTrackInfo {
    pub name: String,
    pub note_count: u64,
    pub program: Option<u8>,
    /// First MIDI channel the track emits on (0-15). Channel 9 = drums — the
    /// frontend uses this to map a track onto the sidecar's instrument WAV keys
    /// ("drums" vs "gm:NNN").
    pub channel: Option<u8>,
}

#[derive(serde::Serialize, Clone)]
pub struct AmtMidiMetadata {
    pub track_count: usize,
    pub total_notes: u64,
    pub ppq: u16,
    pub bpm: Option<f64>,
    pub time_signature: Option<[u32; 2]>,
    pub tracks: Vec<AmtMidiTrackInfo>,
    pub size_bytes: u64,
}

/// Read lightweight metadata from a MIDI file (SMF) for the result workbench.
#[tauri::command]
pub fn amt_midi_metadata(midi_path: String) -> Result<AmtMidiMetadata, String> {
    use midly::{MetaMessage, MidiMessage, Smf, Timing, TrackEventKind};
    use std::collections::HashMap;

    let bytes = std::fs::read(&midi_path).map_err(|e| format!("AMT_META_READ_FAILED: {e}"))?;
    let size_bytes = bytes.len() as u64;
    let smf = Smf::parse(&bytes).map_err(|e| format!("AMT_META_PARSE_FAILED: {e}"))?;

    let ppq = match smf.header.timing {
        Timing::Metrical(t) => t.as_int(),
        Timing::Timecode(_, _) => 0,
    };

    let mut bpm: Option<f64> = None;
    let mut time_signature: Option<[u32; 2]> = None;
    let mut total_notes: u64 = 0;
    let mut tracks: Vec<AmtMidiTrackInfo> = Vec::new();

    for track in &smf.tracks {
        let mut track_name = String::new();
        let mut note_count: u64 = 0;
        let mut program: Option<u8> = None;
        let mut channel: Option<u8> = None;
        let mut open: HashMap<u8, u8> = HashMap::new();

        for ev in track.iter() {
            match ev.kind {
                TrackEventKind::Meta(MetaMessage::Tempo(us)) => {
                    if bpm.is_none() {
                        let us = us.as_int() as f64;
                        if us > 0.0 {
                            bpm = Some(60_000_000.0 / us);
                        }
                    }
                }
                TrackEventKind::Meta(MetaMessage::TimeSignature(num, denom_pow, _, _)) => {
                    if time_signature.is_none() {
                        let den = 1u32 << (denom_pow as u32).min(30);
                        time_signature = Some([num as u32, den]);
                    }
                }
                TrackEventKind::Meta(MetaMessage::TrackName(name)) => {
                    if track_name.is_empty() {
                        track_name = String::from_utf8_lossy(name).to_string();
                    }
                }
                TrackEventKind::Midi { channel: ch, message } => {
                    if channel.is_none() {
                        channel = Some(ch.as_int());
                    }
                    match message {
                    MidiMessage::ProgramChange { program: p } => {
                        if program.is_none() {
                            program = Some(p.as_int());
                        }
                    }
                    MidiMessage::NoteOn { key, vel } => {
                        let k = key.as_int();
                        if vel.as_int() > 0 {
                            *open.entry(k).or_insert(0) += 1;
                        } else if let Some(c) = open.get_mut(&k) {
                            *c = c.saturating_sub(1);
                            if *c == 0 {
                                open.remove(&k);
                            }
                            note_count += 1;
                        }
                    }
                    MidiMessage::NoteOff { key, .. } => {
                        let k = key.as_int();
                        if let Some(c) = open.get_mut(&k) {
                            *c = c.saturating_sub(1);
                            if *c == 0 {
                                open.remove(&k);
                            }
                            note_count += 1;
                        }
                    }
                    _ => {}
                    }
                }
                _ => {}
            }
        }
        total_notes += note_count;
        if note_count > 0 || !track_name.is_empty() {
            tracks.push(AmtMidiTrackInfo {
                name: if track_name.is_empty() {
                    format!("Track {}", tracks.len() + 1)
                } else {
                    track_name
                },
                note_count,
                program,
                channel,
            });
        }
    }

    Ok(AmtMidiMetadata {
        track_count: tracks.len(),
        total_notes,
        ppq,
        bpm,
        time_signature,
        tracks,
        size_bytes,
    })
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────
// Edited-MIDI write-back: the result workbench's MIDI editor commits note edits and needs the edited
// score re-synthesized through the SAME FluidSynth pipeline (amt_prepare_playback). This command
// serializes the frontend's edited stems back into a multi-track SMF.
//
// TICK SPACE: the frontend loads notes via import_score_file, which scales to OUR 480-ppq space and
// REBASES each track to its first note (tick 0). The panel adds `start_tick` back when loading, so the
// ticks arriving here are ABSOLUTE in 480-ppq. We therefore always write PPQ=480 — round-trip stable
// with import_score_file, and amt_prepare_playback/FluidSynth honor the tempo meta for real time.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────

/// One edited note from the frontend (absolute 480-ppq ticks; velocity defaults to 100).
#[derive(serde::Deserialize)]
pub struct EditedNoteInput {
    pub tick: i64,
    pub duration: i64,
    pub pitch: i32,
    #[serde(default)]
    pub velocity: Option<u8>,
    /// §用户：歌词随编辑写回 MIDI（Lyric meta，位于 NoteOn 之前），
    /// import_score_file 会把它铺回音符上。
    #[serde(default)]
    pub lyric: Option<String>,
}

/// One edited track (stem): name + GM program/channel (from the ORIGINAL file's metadata, so the
/// re-synthesis maps back onto the same instrument buses) + its notes.
#[derive(serde::Deserialize)]
pub struct EditedTrackInput {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub program: Option<u8>,
    #[serde(default)]
    pub channel: Option<u8>,
    pub notes: Vec<EditedNoteInput>,
}

/// Write the edited score as a multi-track SMF (PPQ 480, tempo meta from `bpm`).
/// Returns the absolute path written. Mirrors the reference project's edited-MIDI publish path.
#[tauri::command]
pub fn amt_write_edited_midi(
    out_path: String,
    bpm: f64,
    tracks: Vec<EditedTrackInput>,
) -> Result<String, String> {
    use midly::{
        Format, Header, MetaMessage, MidiMessage, Smf, Timing, Track, TrackEvent, TrackEventKind,
    };
    use midly::num::{u15, u24, u28, u4, u7};

    if tracks.is_empty() {
        return Err("EDIT_WRITE_NO_TRACKS".into());
    }
    if !bpm.is_finite() || bpm <= 0.0 {
        return Err("EDIT_WRITE_BAD_BPM".into());
    }

    let out_pb = std::path::PathBuf::from(&out_path);
    if let Some(parent) = out_pb.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("EDIT_WRITE_DIR_FAILED: {e}"))?;
    }

    let ppq = 480u32;
    let mut smf = Smf::new(Header::new(
        Format::Parallel,
        Timing::Metrical(u15::new(ppq as u16)),
    ));

    // Track 0: tempo + time signature (4/4 default, matching the AMT outputs).
    let us_per_quarter = (60_000_000.0 / bpm).round().clamp(1.0, 16_777_215.0) as u32;
    let mut master: Track = Vec::new();
    master.push(TrackEvent {
        delta: u28::new(0),
        kind: TrackEventKind::Meta(MetaMessage::Tempo(u24::new(us_per_quarter))),
    });
    master.push(TrackEvent {
        delta: u28::new(0),
        kind: TrackEventKind::Meta(MetaMessage::TimeSignature(4, 2, 24, 8)),
    });
    master.push(TrackEvent {
        delta: u28::new(0),
        kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
    });
    smf.tracks.push(master);

    for track in &tracks {
        let channel = u4::new(track.channel.unwrap_or(0).min(15));
        let mut events: Vec<(i64, u8, TrackEventKind)> = Vec::new(); // (tick, is_note_off, kind)

        for note in &track.notes {
            let start = note.tick.max(0);
            let dur = note.duration.max(1);
            let key = u7::new(note.pitch.clamp(0, 127) as u8);
            let vel = u7::new(note.velocity.unwrap_or(100).clamp(1, 127));
            // §用户：歌词写为 Lyric meta，排在同 tick 的 NoteOn 之前（稳定排序保序）。
            if let Some(lyr) = note.lyric.as_deref().map(str::trim) {
                if !lyr.is_empty() {
                    events.push((start, 0, TrackEventKind::Meta(MetaMessage::Lyric(lyr.as_bytes()))));
                }
            }
            events.push((
                start,
                0,
                TrackEventKind::Midi {
                    channel,
                    message: MidiMessage::NoteOn { key, vel },
                },
            ));
            events.push((
                start + dur,
                1,
                TrackEventKind::Midi {
                    channel,
                    message: MidiMessage::NoteOff { key, vel: u7::new(0) },
                },
            ));
        }
        // Note-offs first at equal ticks so a re-attack at the same tick stays a clean gap.
        events.sort_by_key(|(t, off, _)| (*t, *off));

        let mut trk: Track = Vec::new();
        trk.push(TrackEvent {
            delta: u28::new(0),
            kind: TrackEventKind::Meta(MetaMessage::TrackName(track.name.as_bytes())),
        });
        if !track.channel.is_some_and(|c| c == 9) {
            // Drums (channel 9) ignore ProgramChange; melodic tracks keep their GM program.
            trk.push(TrackEvent {
                delta: u28::new(0),
                kind: TrackEventKind::Midi {
                    channel,
                    message: MidiMessage::ProgramChange {
                        program: u7::new(track.program.unwrap_or(0).min(127)),
                    },
                },
            });
        }
        let mut cursor: i64 = 0;
        for (tick, _, kind) in events {
            let t = tick.max(0);
            trk.push(TrackEvent {
                delta: u28::new((t - cursor).max(0) as u32),
                kind,
            });
            cursor = t;
        }
        trk.push(TrackEvent {
            delta: u28::new(0),
            kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
        });
        smf.tracks.push(trk);
    }

    smf.save(&out_pb)
        .map_err(|e| format!("EDIT_WRITE_FAILED: {e} ({})", out_pb.display()))?;
    Ok(out_pb.to_string_lossy().into_owned())
}
