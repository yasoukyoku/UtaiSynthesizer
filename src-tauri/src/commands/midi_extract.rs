//! S60-1 GAME MIDI-extraction commands: note extraction from a stem window + the
//! in-app engine downloader.
//!
//! The GAME weights are CC BY-NC-SA 4.0 (code is MIT) — they must NOT ship inside the
//! release bundle. Like OpenUtau, the user downloads them on demand: GitHub release is
//! the primary source, our HF mirror (same license, attributed) is the fallback.
//! Errors are stable CODEs per the i18n rule: MIDI_EXTRACT_NOT_INSTALLED /
//! MIDI_EXTRACT_LOAD_FAILED / MIDI_EXTRACT_TOO_SHORT / MIDI_EXTRACT_CANCELLED /
//! MIDI_EXTRACT_FAILED / GAME_DL_FAILED / GAME_DL_CHECKSUM / GAME_DL_EXTRACT.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use tauri::{Emitter, State};

use crate::inference::midi_extract;
use crate::AppState;

/// job_ids with a pending cancel request. Cancels for jobs that never start (the frontend
/// cancels a whole group at once; queued jobs after the first cancelled one are never
/// invoked) have no cleanup point — the size cap turns that from a slow process-lifetime
/// leak into bounded noise (audit S60).
static EXTRACT_CANCELS: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Extractions currently in flight — GAME sessions are shared (same canonical paths), so
/// the per-job tail unload must only fire when the LAST job finishes, or two concurrent
/// group extractions keep killing each other's sessions mid-run (audit S60 MAJOR).
static ACTIVE_EXTRACTS: AtomicUsize = AtomicUsize::new(0);

fn cancel_requested(job_id: &str) -> bool {
    EXTRACT_CANCELS.lock().iter().any(|j| j == job_id)
}

fn push_cancel(job_id: String) {
    let mut v = EXTRACT_CANCELS.lock();
    v.push(job_id);
    if v.len() > 256 {
        v.drain(..128);
    }
}

fn clear_cancel(job_id: &str) {
    EXTRACT_CANCELS.lock().retain(|j| j != job_id);
}

#[derive(serde::Serialize, Clone)]
pub struct MidiExtractStatus {
    pub installed: bool,
    /// A download is in flight (process-wide) — lets the manager UI restore its progress
    /// view after an unmount/remount instead of offering a second concurrent download.
    pub downloading: bool,
}

fn status(state: &AppState) -> MidiExtractStatus {
    MidiExtractStatus {
        installed: midi_extract::game_installed(state.models.models_dir()),
        downloading: GAME_DL_ACTIVE.load(Ordering::SeqCst),
    }
}

#[tauri::command]
pub async fn midi_extract_status(
    state: State<'_, Arc<AppState>>,
) -> Result<MidiExtractStatus, String> {
    Ok(status(&state))
}

/// Request cancellation of a running extraction (undo during inference / explicit cancel).
#[tauri::command]
pub async fn cancel_midi_extract(job_id: String) -> Result<(), String> {
    push_cancel(job_id);
    Ok(())
}

#[derive(serde::Serialize, Clone)]
pub struct ExtractedNote {
    /// SOURCE-audio ms (window offset folded back in, like analyze_segment_tempo).
    pub onset_ms: f64,
    pub offset_ms: f64,
    /// Float MIDI pitch (A4=69, carries cents) — the frontend rounds/quantizes.
    pub pitch: f32,
}

#[derive(serde::Serialize, Clone)]
struct ExtractProgress {
    job_id: String,
    progress: f32,
}

/// Extract MIDI notes from a SOURCE-audio window of a stem/file (window in source ms,
/// 0/0 = whole file). Runs GAME on CPU; progress via "midi-extract-progress" events.
#[tauri::command]
pub async fn extract_midi_from_audio(
    state: State<'_, Arc<AppState>>,
    app: tauri::AppHandle,
    path: String,
    window_start_ms: f64,
    window_end_ms: f64,
    job_id: String,
) -> Result<Vec<ExtractedNote>, String> {
    let _task = state.begin_task("midi-extract"); // listed in the close-flow warning
    let models_dir = state.models.models_dir().to_path_buf();
    let state_arc = state.inner().clone();
    let job = job_id.clone();
    let result = tauri::async_runtime::spawn_blocking(move || {
        let run = || -> Result<Vec<ExtractedNote>, String> {
            let config = midi_extract::load_game_config(&models_dir)?;
            let input = PathBuf::from(&path);
            let buf = crate::audio::load_audio(&input)
                .map_err(|e| format!("MIDI_EXTRACT_LOAD_FAILED: {e}"))?;
            let ch = buf.channels.max(1) as usize;
            let total_frames = buf.samples.len() / ch;
            let sr = buf.sample_rate as f64;
            // window slicing in source ms — same contract as analyze_segment_tempo
            let mut a = ((window_start_ms.max(0.0) / 1000.0) * sr).round() as usize;
            let mut b =
                (((window_end_ms.max(0.0) / 1000.0) * sr).round() as usize).min(total_frames);
            let whole_file = window_start_ms <= 0.0 && window_end_ms <= 0.0;
            if b <= a {
                if !whole_file {
                    return Err("MIDI_EXTRACT_TOO_SHORT".to_string());
                }
                a = 0;
                b = total_frames;
            }
            let mono: Vec<f32> = buf.samples[a * ch..b * ch]
                .chunks_exact(ch)
                .map(|fr| fr.iter().sum::<f32>() / ch as f32)
                .collect();
            drop(buf);
            let mono44 = crate::inference::features::resample(
                &mono,
                sr as u32,
                config.samplerate,
            );
            drop(mono);
            if mono44.len() < config.samplerate as usize / 10 {
                return Err("MIDI_EXTRACT_TOO_SHORT".to_string());
            }

            let mut last_pct: i32 = -1;
            let window_offset_ms = (a as f64 / sr) * 1000.0;
            let notes = midi_extract::extract_notes(
                &state_arc.inference.engine,
                &models_dir,
                &mono44,
                0, // universal — the stem's language is unknown; official/OpenUtau default
                &|| cancel_requested(&job),
                &mut |p| {
                    let pct = (p * 100.0) as i32;
                    if pct > last_pct {
                        last_pct = pct;
                        let _ = app.emit(
                            "midi-extract-progress",
                            ExtractProgress { job_id: job.clone(), progress: p },
                        );
                    }
                },
            )?;
            Ok(notes
                .into_iter()
                .map(|n| ExtractedNote {
                    onset_ms: window_offset_ms + n.onset_sec * 1000.0,
                    offset_ms: window_offset_ms + n.offset_sec * 1000.0,
                    pitch: n.pitch,
                })
                .collect())
        };
        ACTIVE_EXTRACTS.fetch_add(1, Ordering::SeqCst);
        let out = run();
        // free the CPU sessions (~0.5 GB RAM) — but ONLY when this was the last in-flight
        // extraction: the sessions are shared (same canonical paths), so an unconditional
        // unload here would kill a concurrently running group's sessions mid-run
        // (reload-on-miss saves correctness but thrashes; audit S60 MAJOR).
        if ACTIVE_EXTRACTS.fetch_sub(1, Ordering::SeqCst) == 1 {
            midi_extract::unload_sessions(&state_arc.inference.engine, &models_dir);
        }
        out
    })
    .await
    .map_err(|e| format!("MIDI_EXTRACT_FAILED: join: {e}"))?;
    clear_cancel(&job_id);
    result
}

// ─── downloader (GitHub → HF mirror fallback via the shared S42-audited engine) ──

const GAME_ZIP_SHA256: &str = "5b7a21e64c6310efac399f5d12838fffa70565be162436b5a4a65f290721e7d8";
const GAME_ZIP_SIZE: u64 = 179_775_226;
const GAME_SOURCES: [&str; 2] = [
    "https://github.com/openvpi/GAME/releases/download/v1.0.3/GAME-1.0.3-medium-onnx.zip",
    "https://huggingface.co/datasets/yasoukyoku/utai-runtimes/resolve/main/game/GAME-1.0.3-medium-onnx.zip",
];

/// Single-flight for the download (audit S60: the manager tab's `busy` is component-local
/// state — unmount/remount could start a second concurrent download writing the same .part).
static GAME_DL_ACTIVE: AtomicBool = AtomicBool::new(false);

struct DlGuard;
impl Drop for DlGuard {
    fn drop(&mut self) {
        GAME_DL_ACTIVE.store(false, Ordering::SeqCst);
    }
}

#[derive(serde::Serialize, Clone)]
struct GameDownloadProgress {
    stage: String, // download | extract | done
    downloaded: u64,
    total: u64,
}

/// Extract the package zip (entries live under a GAME-x.y.z-variant-onnx/ prefix —
/// flatten by final path component) into `game_tmp`, then atomically swap into `game`.
fn extract_game_zip(zip_path: &Path, aux_dir: &Path) -> Result<(), String> {
    let file = std::fs::File::open(zip_path).map_err(|e| format!("GAME_DL_EXTRACT: open: {e}"))?;
    let mut archive =
        zip::ZipArchive::new(file).map_err(|e| format!("GAME_DL_EXTRACT: zip: {e}"))?;
    let tmp = aux_dir.join("game.tmp");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).map_err(|e| format!("GAME_DL_EXTRACT: mkdir: {e}"))?;
    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|e| format!("GAME_DL_EXTRACT: entry: {e}"))?;
        if entry.is_dir() {
            continue;
        }
        let name = entry
            .enclosed_name()
            .and_then(|p| p.file_name().map(|f| f.to_string_lossy().to_string()))
            .ok_or_else(|| "GAME_DL_EXTRACT: bad entry name".to_string())?;
        let mut out = std::fs::File::create(tmp.join(&name))
            .map_err(|e| format!("GAME_DL_EXTRACT: create {name}: {e}"))?;
        std::io::copy(&mut entry, &mut out).map_err(|e| format!("GAME_DL_EXTRACT: {name}: {e}"))?;
    }
    for required in midi_extract::GAME_FILES {
        if !tmp.join(required).is_file() {
            let _ = std::fs::remove_dir_all(&tmp);
            return Err(format!("GAME_DL_EXTRACT: missing {required}"));
        }
    }
    let final_dir = aux_dir.join("game");
    if final_dir.exists() {
        std::fs::remove_dir_all(&final_dir).map_err(|e| format!("GAME_DL_EXTRACT: clear: {e}"))?;
    }
    std::fs::rename(&tmp, &final_dir).map_err(|e| format!("GAME_DL_EXTRACT: publish: {e}"))?;
    Ok(())
}

#[tauri::command]
pub async fn download_game_package(
    state: State<'_, Arc<AppState>>,
    app: tauri::AppHandle,
) -> Result<MidiExtractStatus, String> {
    if GAME_DL_ACTIVE.swap(true, Ordering::SeqCst) {
        return Err("GAME_DL_BUSY".to_string());
    }
    let _dl_guard = DlGuard;
    let _task = state.begin_task("game-download");
    let aux_dir = state.models.models_dir().join("aux");
    tokio::fs::create_dir_all(&aux_dir)
        .await
        .map_err(|e| format!("GAME_DL_FAILED: mkdir: {e}"))?;
    let zip_path = aux_dir.join("GAME-1.0.3-medium-onnx.zip");

    // shared engine: connect/read timeouts + per-chunk stall watchdog + .part resume +
    // mirror rotation + sha256 verification (the hand-rolled loop this replaced had none
    // of those — audit S60 MAJOR; the S42 audit already adjudicated the black-holing case)
    let client = crate::download::client().map_err(|e| format!("GAME_DL_FAILED: {e}"))?;
    let req = crate::download::DownloadRequest {
        urls: GAME_SOURCES.iter().map(|s| s.to_string()).collect(),
        dest: zip_path.clone(),
        sha256: Some(GAME_ZIP_SHA256.to_string()),
        expected_size: Some(GAME_ZIP_SIZE),
    };
    let cancel = Arc::new(AtomicBool::new(false));
    let app_emit = app.clone();
    crate::download::download(&client, &req, &cancel, move |done, total| {
        let _ = app_emit.emit(
            "game-download-progress",
            GameDownloadProgress {
                stage: "download".into(),
                downloaded: done,
                total: total.unwrap_or(GAME_ZIP_SIZE),
            },
        );
    })
    .await
    .map_err(|e| format!("GAME_DL_FAILED: {e}"))?;

    let _ = app.emit(
        "game-download-progress",
        GameDownloadProgress { stage: "extract".into(), downloaded: 0, total: 0 },
    );
    // replacing files under a loaded session's path — release them first (delete/import parity)
    state
        .inference
        .engine
        .unload_paths_with_prefix(&midi_extract::game_dir(state.models.models_dir()));
    let zp = zip_path.clone();
    let aux = aux_dir.clone();
    let extracted = tauri::async_runtime::spawn_blocking(move || extract_game_zip(&zp, &aux))
        .await
        .map_err(|e| format!("GAME_DL_EXTRACT: join: {e}"))?;
    if let Err(e) = extracted {
        // don't leave 180 MB of dead weight behind a failed unzip (audit S60)
        let _ = tokio::fs::remove_file(&zip_path).await;
        let _ = tokio::fs::remove_dir_all(aux_dir.join("game.tmp")).await;
        return Err(e);
    }
    let _ = tokio::fs::remove_file(&zip_path).await;
    let _ = app.emit(
        "game-download-progress",
        GameDownloadProgress { stage: "done".into(), downloaded: 0, total: 0 },
    );
    Ok(status(&state))
}

#[tauri::command]
pub async fn delete_game_package(
    state: State<'_, Arc<AppState>>,
) -> Result<MidiExtractStatus, String> {
    let dir = midi_extract::game_dir(state.models.models_dir());
    state.inference.engine.unload_paths_with_prefix(&dir);
    if dir.exists() {
        std::fs::remove_dir_all(&dir).map_err(|e| format!("GAME_DELETE_FAILED: {e}"))?;
    }
    Ok(status(&state))
}
