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
    /// The fp32 ONNX exists (usable natively).
    pub has_onnx: bool,
    /// The `<stem>.fp16.onnx` variant exists. Either precision alone is runnable — the node's
    /// precision selector only appears when BOTH do.
    pub has_fp16: bool,
    /// The model's TRUE output order from its json (converter reads it from ckpt kwargs/yaml).
    /// The frontend MUST label output ports from this, not from hand-written catalog lists:
    /// htdemucs_6s really outputs [drums,bass,other,vocals,guitar,piano] while its model card
    /// says drums/bass/guitar/piano/other/vocals — the catalog order put VOCALS on the Piano port.
    pub stem_names: Option<Vec<String>>,
    /// Residual (mix-minus-stem) label for single-stem models — the LAST output port.
    pub residual_name: Option<String>,
    /// The model json's ACTUAL num_overlap (converter wrote it from the training yaml).
    /// The node's overlap slider must display THIS as its default — the per-arch
    /// MSST_DEFAULT_NUM_OVERLAP constant is only the pre-install fallback and lies for
    /// models whose yaml carries a different value (e.g. Kim family yaml=2, arch default 4).
    pub num_overlap: Option<usize>,
}

/// Read stem_names/residual_name/num_overlap from the model's json sibling
/// (None when absent/unparseable).
fn read_json_fields(fp32_onnx: &Path) -> (Option<Vec<String>>, Option<String>, Option<usize>) {
    let json_path = fp32_onnx.with_extension("json");
    let Ok(text) = std::fs::read_to_string(&json_path) else { return (None, None, None) };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else { return (None, None, None) };
    let names = v.get("stem_names").and_then(|a| a.as_array()).map(|a| {
        a.iter().filter_map(|s| s.as_str().map(str::to_string)).collect::<Vec<_>>()
    }).filter(|n| !n.is_empty());
    let residual = v.get("residual_name").and_then(|s| s.as_str()).map(str::to_string);
    let num_overlap = v.get("num_overlap").and_then(|n| n.as_u64()).map(|n| n as usize);
    (names, residual, num_overlap)
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
        // fp16 variants are surfaced as `has_fp16` on their base entry, not as standalone rows.
        if filename.ends_with(".fp16.onnx") {
            continue;
        }
        let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
        let architecture = detect_architecture_from_name(&filename);

        let fp32_onnx = path.with_extension("onnx");
        let has_onnx = if ext == "onnx" { true } else { fp32_onnx.exists() };
        let has_fp16 = crate::separation::fp16_sibling(&fp32_onnx).exists();
        let (stem_names, residual_name, num_overlap) = read_json_fields(&fp32_onnx);

        models.push(MsstModelFile {
            filename,
            size,
            architecture,
            has_onnx,
            has_fp16,
            stem_names,
            residual_name,
            num_overlap,
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
    // Download-time precision choice ("fp32" | "fp16", None = fp32). fp16 exports through an
    // fp32 intermediate then deletes it — the other precision can be back-converted later.
    precision: Option<String>,
    // The catalog KNOWS the architecture — official demucs weights have hash filenames
    // (5c90dfd2-34c22ccb.th) that name-detection can't classify, which silently skipped
    // auto-convert and left the model unrunnable (native pipeline only since S42).
    // Name detection stays as the fallback for URL/local imports without catalog metadata.
    architecture: Option<String>,
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
        .user_agent(crate::download::APP_USER_AGENT)
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

    // Auto-convert to ONNX — model files only. Sidecar downloads (the catalog's configUrl yaml
    // lands next to the ckpt) match arch names too and used to get "converted" (traceback WARN).
    let is_model_file = matches!(
        dest.extension().and_then(|e| e.to_str()),
        Some("ckpt") | Some("pth") | Some("th") | Some("bin")
    );
    let arch = resolve_architecture(architecture.as_deref(), &filename);
    // Legacy MDX-Net models are ALREADY .onnx — the "conversion" validates the graph and
    // writes the sidecar json (without which the native pipeline refuses the model).
    // Only that arch may convert an .onnx.
    let is_mdx_net_onnx = arch == "mdx_net"
        && dest.extension().and_then(|e| e.to_str()) == Some("onnx");
    if arch != "unknown" && (is_model_file || is_mdx_net_onnx) {
        let _ = app.emit(
            "msst-download-progress",
            DownloadProgress {
                filename: filename.clone(),
                downloaded: 0,
                total: 0,
                stage: "converting".into(),
            },
        );

        match run_converter(&dest, &arch, &app_dir, precision.as_deref()).await {
            Ok(onnx_path) => {
                tracing::info!("Converted {} -> {}", filename, onnx_path);
            }
            Err(e) => {
                tracing::warn!("Auto-conversion failed for {}: {} (model unusable until converted — retry via the convert button)", filename, e);
            }
        }
    }

    Ok(dest.to_string_lossy().to_string())
}

#[tauri::command]
pub async fn convert_msst_model(
    state: State<'_, Arc<AppState>>,
    filename: String,
    // Target precision ("fp32" | "fp16", None = fp32). 补转 fast path: when fp16 is requested
    // and the fp32 ONNX already exists, only the cheap post-hoc fp16 conversion runs (no
    // torch re-export). fp32 (or fp16 without an fp32 on disk) does the full ckpt export.
    precision: Option<String>,
    // Catalog-provided architecture (hash-named official weights defeat name detection).
    architecture: Option<String>,
) -> Result<String, String> {
    let _task = state.begin_task("convert"); // listed in the close-flow's in-progress warning
    let dir = &state.msst_models_dir;
    let path = dir.join(&filename);
    if !path.exists() {
        return Err(format!("Model file not found: {}", filename));
    }

    if precision.as_deref() == Some("fp16") {
        let fp32_onnx = path.with_extension("onnx");
        if fp32_onnx.exists() {
            return run_fp16_converter(&fp32_onnx, &state.app_dir)
                .await
                .map_err(|e| format!("fp16 conversion failed: {}", e));
        }
    }

    let arch = resolve_architecture(architecture.as_deref(), &filename);
    if arch == "unknown" {
        return Err("Cannot detect architecture from filename".into());
    }

    let onnx_path = run_converter(&path, &arch, &state.app_dir, precision.as_deref())
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

    for ext in &["json", "yaml", "yml", "onnx", "fp16.onnx"] {
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
    let fp32_onnx = dest.with_extension("onnx");
    let has_onnx = fp32_onnx.exists();
    let has_fp16 = crate::separation::fp16_sibling(&fp32_onnx).exists();
    let (stem_names, residual_name, num_overlap) = read_json_fields(&fp32_onnx);

    tracing::info!("Imported MSST model: {}", filename);
    Ok(MsstModelFile {
        filename,
        size,
        architecture,
        has_onnx,
        has_fp16,
        stem_names,
        residual_name,
        num_overlap,
    })
}

// ─── Converter ───────────────────────────────────────────────

async fn run_converter(
    model_path: &Path,
    arch: &str,
    app_dir: &Path,
    precision: Option<&str>,
) -> Result<String, String> {
    let python = crate::pyenv::converter_python_checked(app_dir).map_err(|e| e.to_string())?;
    let script = app_dir.join("converter").join("convert.py");

    if !script.exists() {
        return Err(format!("Converter script not found: {}", script.display()));
    }

    let onnx_path = model_path.with_extension("onnx");

    // Shared python spawn hygiene (UTF-8 stdio + no console flash) — crate::util::python_command.
    let mut cmd = tokio::process::Command::from(crate::util::python_command(&python));
    cmd.arg(&script)
        .arg("--input").arg(model_path)
        .arg("--output").arg(&onnx_path)
        .arg("--type").arg(arch);

    if let Some(p) = precision {
        cmd.arg("--precision").arg(p);
    }

    // Pass the model's ORIGINAL yaml whenever it sits next to the ckpt (the frontend downloads
    // the catalog's configUrl there). Roformers take chunk_size/num_overlap from it; melband/
    // mdx23c also STFT/mel params. Archs that ignore --config just don't read it.
    for ext in &["yaml", "yml"] {
        let cfg = model_path.with_extension(ext);
        if cfg.exists() {
            cmd.arg("--config").arg(&cfg);
            break;
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

/// 补转 fast path: post-hoc fp16 conversion of an existing fp32 ONNX (converter/onnx_fp16.py,
/// the proven convert_float_to_float16 + Cast-retarget recipe). Writes `<stem>.fp16.onnx`
/// next to the input; the shared `<stem>.json` is untouched. Takes ~1-2 min, no torch export.
async fn run_fp16_converter(fp32_onnx: &Path, app_dir: &Path) -> Result<String, String> {
    let python = crate::pyenv::converter_python_checked(app_dir).map_err(|e| e.to_string())?;
    let script = app_dir.join("converter").join("onnx_fp16.py");
    if !script.exists() {
        return Err(format!("fp16 converter script not found: {}", script.display()));
    }

    let out_path = crate::separation::fp16_sibling(fp32_onnx);
    tracing::info!("Converting to fp16: {} -> {}", fp32_onnx.display(), out_path.display());

    let output = tokio::process::Command::from(crate::util::python_command(&python))
        .arg(&script)
        .arg(fp32_onnx)
        .arg(&out_path)
        .output()
        .await
        .map_err(|e| format!("Failed to run fp16 converter ({}): {}", python.display(), e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(stderr.trim().to_string());
    }
    if !out_path.exists() {
        return Err("fp16 converter finished but output file not found".into());
    }
    Ok(out_path.to_string_lossy().to_string())
}

// Converter interpreter resolution = crate::pyenv::converter_python_checked (S42) —
// see models/convert.rs. The TRAINING role uses crate::pyenv::training_interpreter.

/// Catalog-provided architecture wins (validated against the known set); filename heuristics
/// are only the fallback for URL/local imports without catalog metadata.
fn resolve_architecture(explicit: Option<&str>, filename: &str) -> String {
    if let Some(a) = explicit {
        if matches!(a, "bs_roformer" | "mel_band_roformer" | "mdx23c" | "htdemucs"
                       | "uvr_vr" | "mdx_net") {
            return a.to_string();
        }
        tracing::warn!("Ignoring unknown architecture hint '{}' — falling back to name detection", a);
    }
    detect_architecture_from_name(filename)
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
        // NOTE: must stay BEFORE the looser "mdxnet" check below.
        "mdx23c".to_string()
    } else if lower.contains("mdxnet") || lower.contains("mdx_net") {
        // Legacy UVR MDX-Net .onnx (e.g. UVR_MDXNET_KARA*.onnx).
        "mdx_net".to_string()
    } else if lower.contains("karaoke-uvr") || lower.contains("de-echo")
        || lower.contains("deecho") || lower.contains("denoise")
        || lower.contains("de-reverb") || lower.contains("wind_inst")
    {
        // UVR VR-arch .pth family (registry-gated in the converter — an unknown
        // VR model fails there with a clear message, not with garbage output).
        "uvr_vr".to_string()
    } else {
        "unknown".to_string()
    }
}
