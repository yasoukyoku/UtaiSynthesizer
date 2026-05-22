use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tauri::{Emitter, State};

use crate::AppState;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MsstModelFile {
    pub filename: String,
    pub size: u64,
    pub architecture: String,
    pub has_onnx: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DownloadProgress {
    filename: String,
    downloaded: u64,
    total: u64,
    stage: String,
}

#[tauri::command]
pub fn get_msst_models_dir(state: State<'_, Arc<AppState>>) -> Result<String, String> {
    let dir = state.msst_models_dir.clone();
    std::fs::create_dir_all(&dir).map_err(|e| format!("Failed to create models dir: {}", e))?;
    Ok(dir.to_string_lossy().to_string())
}

#[tauri::command]
pub fn list_msst_models(state: State<'_, Arc<AppState>>) -> Result<Vec<MsstModelFile>, String> {
    let dir = &state.msst_models_dir;
    if !dir.exists() {
        return Ok(vec![]);
    }

    let mut models = Vec::new();
    let entries = std::fs::read_dir(dir).map_err(|e| e.to_string())?;

    for entry in entries.flatten() {
        let path = entry.path();
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if !matches!(ext, "ckpt" | "th" | "pth" | "onnx") {
            continue;
        }

        let filename = path.file_name().unwrap_or_default().to_string_lossy().to_string();
        let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
        let architecture = detect_architecture_from_name(&filename);

        let has_onnx = if ext == "onnx" {
            true
        } else {
            path.with_extension("onnx").exists()
        };

        models.push(MsstModelFile {
            filename,
            size,
            architecture,
            has_onnx,
        });
    }

    models.sort_by(|a, b| a.filename.cmp(&b.filename));
    Ok(models)
}

#[tauri::command]
pub async fn download_msst_model(
    app: tauri::AppHandle,
    state: State<'_, Arc<AppState>>,
    url: String,
    filename: String,
) -> Result<String, String> {
    let dir = state.msst_models_dir.clone();
    let app_dir = state.app_dir.clone();
    std::fs::create_dir_all(&dir).map_err(|e| format!("Failed to create models dir: {}", e))?;

    let dest = dir.join(&filename);
    if dest.exists() {
        tracing::info!("File already exists, skipping: {}", filename);
        return Ok(dest.to_string_lossy().to_string());
    }

    let tmp = dir.join(format!("{}.part", filename));
    if tmp.exists() {
        let _ = tokio::fs::remove_file(&tmp).await;
    }

    let client = reqwest::Client::builder()
        .user_agent("UTAI/2.0")
        .build()
        .map_err(|e| format!("HTTP client error: {}", e))?;

    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("Download request failed: {}", e))?;

    if !response.status().is_success() {
        return Err(format!("HTTP {}: {}", response.status(), url));
    }

    let total = response.content_length().unwrap_or(0);

    let mut file =
        tokio::fs::File::create(&tmp)
            .await
            .map_err(|e| format!("Failed to create file: {}", e))?;

    let mut stream = response.bytes_stream();
    let mut downloaded: u64 = 0;
    let mut last_emit: u64 = 0;

    use futures_util::StreamExt;
    use tokio::io::AsyncWriteExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Download stream error: {}", e))?;
        file.write_all(&chunk)
            .await
            .map_err(|e| format!("File write error: {}", e))?;
        downloaded += chunk.len() as u64;

        if downloaded - last_emit > 1_000_000 || downloaded == total {
            let _ = app.emit(
                "msst-download-progress",
                DownloadProgress {
                    filename: filename.clone(),
                    downloaded,
                    total,
                    stage: "download".into(),
                },
            );
            last_emit = downloaded;
        }
    }

    file.flush()
        .await
        .map_err(|e| format!("File flush error: {}", e))?;
    drop(file);

    tokio::fs::rename(&tmp, &dest)
        .await
        .map_err(|e| format!("Failed to rename temp file: {}", e))?;

    tracing::info!("Downloaded MSST model: {} ({} bytes)", filename, downloaded);

    // Auto-convert to ONNX
    let arch = detect_architecture_from_name(&filename);
    if arch != "unknown" {
        let _ = app.emit(
            "msst-download-progress",
            DownloadProgress {
                filename: filename.clone(),
                downloaded: 0,
                total: 0,
                stage: "converting".into(),
            },
        );

        match run_converter(&dest, &arch, &app_dir).await {
            Ok(onnx_path) => {
                tracing::info!("Converted {} -> {}", filename, onnx_path);
            }
            Err(e) => {
                tracing::warn!("Auto-conversion failed for {}: {} (model still usable via sidecar)", filename, e);
            }
        }
    }

    Ok(dest.to_string_lossy().to_string())
}

#[tauri::command]
pub async fn convert_msst_model(
    state: State<'_, Arc<AppState>>,
    filename: String,
) -> Result<String, String> {
    let dir = &state.msst_models_dir;
    let path = dir.join(&filename);
    if !path.exists() {
        return Err(format!("Model file not found: {}", filename));
    }

    let arch = detect_architecture_from_name(&filename);
    if arch == "unknown" {
        return Err("Cannot detect architecture from filename".into());
    }

    let onnx_path = run_converter(&path, &arch, &state.app_dir)
        .await
        .map_err(|e| format!("Conversion failed: {}", e))?;

    Ok(onnx_path)
}

#[tauri::command]
pub fn delete_msst_model(
    state: State<'_, Arc<AppState>>,
    filename: String,
) -> Result<(), String> {
    let dir = &state.msst_models_dir;
    let path = dir.join(&filename);
    if path.exists() {
        std::fs::remove_file(&path).map_err(|e| format!("Delete failed: {}", e))?;
    }

    let stem = PathBuf::from(&filename);
    let stem_name = stem.file_stem().unwrap_or_default().to_string_lossy();

    for ext in &["json", "yaml", "yml", "onnx"] {
        let related = dir.join(format!("{}.{}", stem_name, ext));
        if related.exists() {
            std::fs::remove_file(&related).ok();
        }
    }

    tracing::info!("Deleted MSST model: {}", filename);
    Ok(())
}

#[tauri::command]
pub fn import_local_msst_model(
    state: State<'_, Arc<AppState>>,
    source_path: String,
) -> Result<MsstModelFile, String> {
    let source = PathBuf::from(&source_path);
    if !source.exists() {
        return Err(format!("File not found: {}", source_path));
    }

    let filename = source.file_name().unwrap_or_default().to_string_lossy().to_string();
    let dir = &state.msst_models_dir;
    std::fs::create_dir_all(dir).map_err(|e| format!("Failed to create models dir: {}", e))?;

    let dest = dir.join(&filename);
    if dest.exists() {
        return Err(format!("File already exists: {}", filename));
    }

    std::fs::copy(&source, &dest).map_err(|e| format!("Copy failed: {}", e))?;

    for ext in &["yaml", "yml"] {
        let cfg = source.with_extension(ext);
        if cfg.exists() {
            let cfg_dest = dir.join(cfg.file_name().unwrap());
            std::fs::copy(&cfg, &cfg_dest).ok();
        }
    }

    let size = std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
    let architecture = detect_architecture_from_name(&filename);
    let has_onnx = dest.with_extension("onnx").exists();

    tracing::info!("Imported MSST model: {}", filename);
    Ok(MsstModelFile {
        filename,
        size,
        architecture,
        has_onnx,
    })
}

// ─── Converter ───────────────────────────────────────────────

async fn run_converter(model_path: &Path, arch: &str, app_dir: &Path) -> Result<String, String> {
    let python = find_converter_python(app_dir);
    let script = app_dir.join("converter").join("convert.py");

    if !script.exists() {
        return Err(format!("Converter script not found: {}", script.display()));
    }

    let onnx_path = model_path.with_extension("onnx");

    let mut cmd = tokio::process::Command::new(&python);
    cmd.arg(&script)
        .arg("--input").arg(model_path)
        .arg("--output").arg(&onnx_path)
        .arg("--type").arg(arch);

    if arch == "mdx23c" {
        for ext in &["yaml", "yml"] {
            let cfg = model_path.with_extension(ext);
            if cfg.exists() {
                cmd.arg("--config").arg(&cfg);
                break;
            }
        }
    }

    tracing::info!("Converting {} (arch={})...", model_path.display(), arch);

    let output = cmd
        .output()
        .await
        .map_err(|e| format!("Failed to run converter ({}): {}", python.display(), e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(stderr.trim().to_string());
    }

    if !onnx_path.exists() {
        return Err("Converter finished but .onnx file not found".into());
    }

    Ok(onnx_path.to_string_lossy().to_string())
}

fn find_converter_python(app_dir: &Path) -> PathBuf {
    let venv = app_dir
        .join("converter")
        .join(".venv")
        .join("Scripts")
        .join("python.exe");
    if venv.exists() {
        return venv;
    }

    let embedded = app_dir.join("python").join("python.exe");
    if embedded.exists() {
        return embedded;
    }

    PathBuf::from("python")
}

fn detect_architecture_from_name(filename: &str) -> String {
    let lower = filename.to_lowercase();
    if lower.contains("htdemucs") {
        "htdemucs".to_string()
    } else if lower.contains("mel_band") || lower.contains("melband") {
        "mel_band_roformer".to_string()
    } else if lower.contains("bs_roformer") || lower.contains("bsroformer") {
        "bs_roformer".to_string()
    } else if lower.contains("mdx23c") || lower.contains("tfc_tdf") {
        "mdx23c".to_string()
    } else {
        "unknown".to_string()
    }
}
