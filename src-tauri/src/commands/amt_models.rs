use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tauri::{Emitter, State};

use crate::AppState;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmtModelFile {
    pub id: Option<String>,
    pub filename: String,
    pub size: u64,
    pub architecture: String,
    pub is_installed: bool,
    pub sha256_ok: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DownloadProgress {
    id: String,
    downloaded: u64,
    total: u64,
    stage: String,
}

#[tauri::command]
pub fn get_amt_models_dir(state: State<'_, Arc<AppState>>) -> Result<String, String> {
    let dir = state.amt_models_dir.clone();
    std::fs::create_dir_all(&dir).map_err(|e| format!("Failed to create AMT models dir: {}", e))?;
    Ok(dir.to_string_lossy().to_string())
}

fn resolve_amt_model_path(dir: &Path, filename: &str) -> bool {
    dir.join(filename).exists()
}

#[tauri::command]
pub fn list_amt_models(state: State<'_, Arc<AppState>>) -> Result<Vec<AmtModelFile>, String> {
    let dir = &state.amt_models_dir;
    let mut models = Vec::new();
    let mut seen_filenames = std::collections::HashSet::new();

    // 1. Scan local app models dir
    if dir.exists() {
        let mut stack = vec![dir.to_path_buf()];
        while let Some(current_dir) = stack.pop() {
            if let Ok(entries) = std::fs::read_dir(current_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        stack.push(path);
                    } else {
                        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
                        if ext == "ckpt" || ext == "safetensors" || ext == "pth" || ext == "pt" || ext == "exe" || ext == "sf2" || ext == "onnx" || ext == "msi" || ext == "yaml" {
                            let rel_path = path.strip_prefix(dir).unwrap_or(&path);
                            let filename = rel_path.to_string_lossy().to_string().replace("\\", "/");
                            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                            
                            let architecture = detect_architecture(&filename);
                            seen_filenames.insert(filename.clone());

                            models.push(AmtModelFile {
                                id: None,
                                filename,
                                size,
                                architecture,
                                is_installed: true,
                                sha256_ok: true,
                            });
                        }
                    }
                }
            }
        }
    }

    // Explicitly check for FluidSynth and MuseScoreGeneral SoundFont at expected paths
    let fluidsynth_installed = resolve_amt_model_path(dir, "fluidsynth/2.5.6/bin/fluidsynth.exe");
    let soundfont_installed = resolve_amt_model_path(dir, "MuseScore_General.sf2");
    // FFmpeg (resource-manager copy) — <models>/amt/ffmpeg/bin/ffmpeg.exe
    let ffmpeg_installed = resolve_amt_model_path(dir, "ffmpeg/bin/ffmpeg.exe");

    // Add FluidSynth status
    if !seen_filenames.contains("fluidsynth/2.5.6/bin/fluidsynth.exe") {
        models.push(AmtModelFile {
            id: Some("fluidsynth_runtime".to_string()),
            filename: "fluidsynth/2.5.6/bin/fluidsynth.exe".to_string(),
            size: 0,
            architecture: "fluidsynth".to_string(),
            is_installed: fluidsynth_installed,
            sha256_ok: true,
        });
    }

    // Add FFmpeg status (essential on-demand codec; NOT bundled in the installer)
    if !seen_filenames.contains("ffmpeg/bin/ffmpeg.exe") {
        models.push(AmtModelFile {
            id: Some("ffmpeg_runtime".to_string()),
            filename: "ffmpeg/bin/ffmpeg.exe".to_string(),
            size: 0,
            architecture: "ffmpeg".to_string(),
            is_installed: ffmpeg_installed,
            sha256_ok: true,
        });
    }

    // Add MuseScoreGeneral SoundFont status
    if !seen_filenames.contains("MuseScore_General.sf2") {
        models.push(AmtModelFile {
            id: Some("musescore_general_sf2".to_string()),
            filename: "MuseScore_General.sf2".to_string(),
            size: 0,
            architecture: "soundfont".to_string(),
            is_installed: soundfont_installed,
            sha256_ok: true,
        });
    }

    // Whisper lyric-recognition models (P0-B): each size lives in
    // whisper/<size>/ and counts as installed when model.bin + config.json
    // are present (the two files faster-whisper requires).
    for (size, model_id) in [
        ("tiny", "whisper_tiny"),
        ("base", "whisper_base"),
        ("small", "whisper_small"),
        ("medium", "whisper_medium"),
    ] {
        let filename = format!("whisper/{size}/model.bin");
        let installed = dir
            .join("whisper")
            .join(size)
            .join("model.bin")
            .is_file()
            && dir.join("whisper").join(size).join("config.json").is_file();
        if !seen_filenames.contains(&filename) {
            models.push(AmtModelFile {
                id: Some(model_id.to_string()),
                filename,
                size: 0,
                architecture: "whisper".to_string(),
                is_installed: installed,
                sha256_ok: true,
            });
        }
    }

    // (Shared search roots logic has been removed as per user request to isolate the project)

    models.sort_by(|a, b| a.filename.cmp(&b.filename));
    Ok(models)
}

fn detect_architecture(filename: &str) -> String {
    if filename.contains("yourmt3") || filename.contains("yptf") || filename.contains("last.ckpt") {
        "yourmt3_plus".to_string()
    } else if filename.contains("transkun") {
        "transkun".to_string()
    } else if filename.contains("aria") {
        "aria_amt".to_string()
    } else if filename.contains("bytedance") || filename.contains("piano_transcription") {
        "bytedance_piano".to_string()
    } else if filename.contains("beat_this") || filename.contains("final0") {
        "beat_this".to_string()
    } else if filename.contains("fluidsynth") {
        "fluidsynth".to_string()
    } else if filename.starts_with("ffmpeg/") || filename.contains("ffmpeg") {
        "ffmpeg".to_string()
    } else if filename.contains(".sf2") || filename.contains("SoundFont") {
        "soundfont".to_string()
    } else if filename.contains("Roformer") || filename.contains("bs_6stem") || filename.contains("leap_xe") {
        "bs_roformer".to_string()
    } else if filename.contains("polarformer") {
        "polarformer".to_string()
    } else if filename.contains("MuseScore") || filename.contains("musescore") {
        "musescore".to_string()
    } else if filename.contains("miros") {
        "miros".to_string()
    } else {
        "unknown".to_string()
    }
}

#[tauri::command]
pub async fn download_amt_model(
    app: tauri::AppHandle,
    state: State<'_, Arc<AppState>>,
    urls: Vec<String>,
    id: String,
    filename: String,
    architecture: String,
    sha256: Option<String>,
) -> Result<String, String> {
    let dir = state.amt_models_dir.clone();
    std::fs::create_dir_all(&dir).map_err(|e| format!("Failed to create models dir: {}", e))?;

    let mut dest = dir.join(&filename);
    if architecture == "fluidsynth" {
        dest = dir.join("fluidsynth_runtime.zip");
    } else if architecture == "ffmpeg" {
        // FFmpeg ships as a release ZIP (gyan essentials build) whose root folder
        // ("ffmpeg-9.0.1-essentials_build/") is stripped on extract into <dir>/ffmpeg/.
        dest = dir.join("ffmpeg_runtime.zip");
    }

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    // Special handling for FluidSynth ZIP
    if architecture == "fluidsynth" {
        let exe_path = dir.join(&filename);
        if exe_path.exists() {
            return Ok(exe_path.to_string_lossy().to_string());
        }
    }

    // Special handling for FFmpeg ZIP — skip re-download when the extracted exe is present
    if architecture == "ffmpeg" {
        let exe_path = dir.join(&filename);
        if exe_path.exists() {
            return Ok(exe_path.to_string_lossy().to_string());
        }
    }

    // Special handling for TransKun V2 Aug ZIP
    if id == "transkun_v2_aug" {
        let pt_path = dir.join(&filename);
        if pt_path.exists() {
            return Ok(pt_path.to_string_lossy().to_string());
        }
    }

    // Special handling for MuseScore MSI
    if architecture == "musescore" {
        let exe_path = dir.join(&filename);
        if exe_path.exists() {
            return Ok(exe_path.to_string_lossy().to_string());
        }
    }

    if !["fluidsynth", "musescore", "ffmpeg"].contains(&architecture.as_str()) && dest.exists() {
        // If sha256 is provided, verify it.
        if let Some(expected_sha) = &sha256 {
            if expected_sha != "verified_by_runtime" {
                let actual_sha = crate::download::sha256_file(&dest).map_err(|e| e.to_string())?;
                if actual_sha == *expected_sha {
                    return Ok(dest.to_string_lossy().to_string());
                } else {
                    tracing::info!("SHA256 mismatch for {}, re-downloading", filename);
                }
            } else {
                return Ok(dest.to_string_lossy().to_string());
            }
        } else {
            return Ok(dest.to_string_lossy().to_string());
        }
    }

    let client = crate::download::client().map_err(|e| e.to_string())?;
    let app_emit = app.clone();
    let model_id = id.clone();
    let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut last_emit: u64 = 0;
    
    crate::download::download(
        &client,
        &crate::download::DownloadRequest {
            urls,
            dest: dest.clone(),
            sha256: if sha256.as_deref() == Some("verified_by_runtime") { None } else { sha256 },
            expected_size: None,
        },
        &cancel,
        move |done, total| {
            if done < last_emit {
                last_emit = 0;
            }
            if done.saturating_sub(last_emit) > 1_000_000 || Some(done) == total {
                last_emit = done;
                let _ = app_emit.emit(
                    "amt-download-progress",
                    DownloadProgress {
                        id: model_id.clone(),
                        downloaded: done,
                        total: total.unwrap_or(0),
                        stage: "download".into(),
                    },
                );
            }
        },
    )
    .await
    .map_err(|e| e.to_string())?;

    // FluidSynth / FFmpeg ZIP extraction (root folder stripped into their target dirs)
    if architecture == "fluidsynth" || architecture == "ffmpeg" || id == "transkun_v2_aug" {
        let zip_path = dest.clone();
        let target_dir = if id == "transkun_v2_aug" {
            dir.join("amt").join("transkun_v2_aug").join("checkpoints")
        } else if architecture == "ffmpeg" {
            dir.join("ffmpeg")
        } else {
            dir.join("fluidsynth").join("2.5.6")
        };
        std::fs::create_dir_all(&target_dir).ok();
        
        let app_emit = app.clone();
        let model_id = id.clone();
        let _ = app_emit.emit("amt-download-progress", DownloadProgress {
            id: model_id.clone(),
            downloaded: 100,
            total: 100,
            stage: "extract".into(),
        });

        let id_for_closure = id.clone();
        tokio::task::spawn_blocking(move || {
            let file = std::fs::File::open(&zip_path).map_err(|e| e.to_string())?;
            let mut archive = zip::ZipArchive::new(file).map_err(|e| e.to_string())?;
            
            for i in 0..archive.len() {
                let mut file = archive.by_index(i).map_err(|e| e.to_string())?;
                let outpath = match file.enclosed_name() {
                    Some(path) => {
                        // For transkun_v2_aug, we don't strip the root component
                        if id_for_closure == "transkun_v2_aug" {
                            target_dir.join(path.file_name().unwrap())
                        } else {
                            // The zip usually has a root folder like "fluidsynth-v2.5.6-win10-x64-cpp11/"
                            // We want to strip the first component.
                            let mut components = path.components();
                            components.next(); // skip root
                            target_dir.join(components.as_path())
                        }
                    },
                    None => continue,
                };

                if (*file.name()).ends_with('/') {
                    std::fs::create_dir_all(&outpath).ok();
                } else {
                    if let Some(p) = outpath.parent() {
                        std::fs::create_dir_all(p).ok();
                    }
                    let mut outfile = std::fs::File::create(&outpath).map_err(|e| e.to_string())?;
                    std::io::copy(&mut file, &mut outfile).map_err(|e| e.to_string())?;
                }
            }
            // Cleanup zip
            let _ = std::fs::remove_file(&zip_path);
            Ok::<(), String>(())
        }).await.map_err(|e| e.to_string())??;
    }

    // MuseScore MSI extraction
    if architecture == "musescore" {
        let msi_path = dest.clone();
        let target_dir = dir.join("musescore").join("4.7.4");
        std::fs::create_dir_all(&target_dir).ok();

        let app_emit = app.clone();
        let model_id = id.clone();
        let _ = app_emit.emit("amt-download-progress", DownloadProgress {
            id: model_id,
            downloaded: 100,
            total: 100,
            stage: "extract".into(),
        });

        tokio::task::spawn_blocking(move || {
            let temp_extract = target_dir.join("temp_msi_extract");
            std::fs::create_dir_all(&temp_extract).ok();

            #[cfg(target_os = "windows")]
            {
                let status = std::process::Command::new("msiexec")
                    .arg("/a")
                    .arg(&msi_path)
                    .arg("/qn")
                    .arg("/norestart")
                    .arg(format!("TARGETDIR={}", temp_extract.to_string_lossy()))
                    .status()
                    .map_err(|e| e.to_string())?;

                if !status.success() {
                    return Err(format!("MSI extraction failed with status: {}", status));
                }

                // Find MuseScore4.exe in temp_extract and move its parent's parent content to target_dir
                // Based on Python script, it's usually in extracted/PFiles/MuseScore 4/
                // We'll just search for MuseScore4.exe
                let mut found_exe = None;
                for entry in walkdir::WalkDir::new(&temp_extract) {
                    let entry = entry.map_err(|e| e.to_string())?;
                    if entry.file_name() == "MuseScore4.exe" {
                        found_exe = Some(entry.path().to_path_buf());
                        break;
                    }
                }

                if let Some(exe_path) = found_exe {
                    if let Some(dist_root) = exe_path.parent().and_then(|p| p.parent()) {
                        // Move everything from dist_root to target_dir
                        for entry in std::fs::read_dir(dist_root).map_err(|e| e.to_string())? {
                            let entry = entry.map_err(|e| e.to_string())?;
                            let dest_path = target_dir.join(entry.file_name());
                            if dest_path.exists() {
                                if dest_path.is_dir() {
                                    std::fs::remove_dir_all(&dest_path).ok();
                                } else {
                                    std::fs::remove_file(&dest_path).ok();
                                }
                            }
                            // Move file/dir
                            std::fs::rename(entry.path(), dest_path).map_err(|e| e.to_string())?;
                        }
                    }
                }
            }
            
            // Cleanup
            std::fs::remove_dir_all(&temp_extract).ok();
            std::fs::remove_file(&msi_path).ok();
            Ok::<(), String>(())
        }).await.map_err(|e| e.to_string())??;
    }

    tracing::info!("Downloaded AMT model: {}", filename);
    Ok(dest.to_string_lossy().to_string())
}

#[tauri::command]
pub fn delete_amt_model(
    state: State<'_, Arc<AppState>>,
    filename: String,
) -> Result<(), String> {
    let dir = &state.amt_models_dir;
    let path = dir.join(&filename);
    // Whisper models span a directory of small support files — remove the
    // whole whisper/<size>/ folder, not just model.bin.
    if filename.starts_with("whisper/") {
        if let Some(size_dir) = path.parent() {
            if size_dir.join("model.bin").is_file() {
                std::fs::remove_dir_all(size_dir)
                    .map_err(|e| format!("AMT_DELETE_FAILED: {e}"))?;
                tracing::info!("Deleted AMT whisper model dir: {}", size_dir.display());
                return Ok(());
            }
        }
    }
    // FFmpeg extracts into a multi-file tree (bin/ + docs) — remove <dir>/ffmpeg as a whole.
    if filename.starts_with("ffmpeg/") {
        let ffmpeg_dir = dir.join("ffmpeg");
        if ffmpeg_dir.is_dir() {
            std::fs::remove_dir_all(&ffmpeg_dir)
                .map_err(|e| format!("AMT_DELETE_FAILED: {e}"))?;
            tracing::info!("Deleted AMT ffmpeg dir: {}", ffmpeg_dir.display());
            return Ok(());
        }
    }
    if path.exists() {
        std::fs::remove_file(&path).map_err(|e| format!("AMT_DELETE_FAILED: {e}"))?;
        
        // Clean up empty parent directories
        let mut parent = path.parent();
        while let Some(p) = parent {
            if p == dir { break; }
            if let Ok(entries) = std::fs::read_dir(p) {
                if entries.count() == 0 {
                    let _ = std::fs::remove_dir(p);
                } else {
                    break;
                }
            } else {
                break;
            }
            parent = p.parent();
        }
    }

    tracing::info!("Deleted AMT model: {}", filename);
    Ok(())
}

