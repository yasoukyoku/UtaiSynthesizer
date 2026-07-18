use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tauri::{Emitter, State};

use crate::AppState;

/// The fp16 conversion-recipe generation this build considers current.
/// MUST mirror converter/onnx_fp16.py FP16_RECIPE (the stamp's single writer).
const FP16_RECIPE: &str = "2";

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
    /// S68c: the fp16 file carries a current-recipe stamp (`<stem>.fp16.recipe` sidecar, written
    /// by onnx_fp16.py on success). false ⇒ converted by an older build (roformers: can go
    /// all-NaN on silent chunks on GPU EPs) ⇒ the manager shows the "重转 fp16" cure button;
    /// one successful reconvert stamps it and the prompt stops (§user).
    pub fp16_recipe_ok: bool,
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
        let fp16_path = crate::separation::fp16_sibling(&fp32_onnx);
        let has_fp16 = fp16_path.exists();
        let fp16_recipe_ok = has_fp16
            && std::fs::read_to_string(fp16_path.with_extension("recipe"))
                .map(|s| s.trim() == FP16_RECIPE)
                .unwrap_or(false);
        let (stem_names, residual_name, num_overlap) = read_json_fields(&fp32_onnx);

        models.push(MsstModelFile {
            filename,
            size,
            architecture,
            has_onnx,
            has_fp16,
            fp16_recipe_ok,
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
    // S66: full mirror-failover candidate list (frontend ghMirrorCandidates — chosen proxy →
    // direct → other presets), consumed in order by the unified engine.
    urls: Vec<String>,
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

    // S66: migrated onto the unified engine (the header note in download.rs named this file as
    // the last legacy holdout) — .part Range-resume across THIS invoke's mirror rotation,
    // stall watchdog, HTML-poison sniff. No sha256: third-party catalog files carry no digests,
    // so a stale .part from an EARLIER invoke is deleted up front (review S66: blind-resuming
    // un-hashed bytes of unknown origin could commit a corrupt model that no re-download would
    // ever repair — the dest-exists short-circuit above would trust it forever). Within one
    // invoke the sources are the same content, so mid-rotation resume stays safe.
    let part = crate::download::part_path(&dest);
    if part.exists() {
        let _ = std::fs::remove_file(&part);
    }
    let client = crate::download::client().map_err(|e| e.to_string())?;
    let app_emit = app.clone();
    let fname = filename.clone();
    let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false)); // no cancel UI (parity with the old flow)
    let mut last_emit: u64 = 0;
    crate::download::download(
        &client,
        &crate::download::DownloadRequest {
            urls,
            dest: dest.clone(),
            sha256: None,
            expected_size: None,
        },
        &cancel,
        move |done, total| {
            // done < last_emit = a restart-from-zero (server ignored Range) — reset the
            // high-water mark or the bar freezes at the stale value (review S66).
            if done < last_emit {
                last_emit = 0;
            }
            if done.saturating_sub(last_emit) > 1_000_000 || Some(done) == total {
                last_emit = done;
                let _ = app_emit.emit(
                    "msst-download-progress",
                    DownloadProgress {
                        filename: fname.clone(),
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

    tracing::info!("Downloaded MSST model: {}", filename);

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
        // S66: the auto-convert honors the convert slot. Busy → SKIP the conversion but keep the
        // finished download (the run-preflight / convert button covers it later) — a busy peer
        // must never fail a completed multi-GB download. The "converting" stage event is emitted
        // ONLY after the slot is held (review S66: an emit on the busy-skip path races the invoke
        // resolution — a late event re-adds the frontend's downloading record with no terminal
        // event to ever clear it, permanently graying every convert button).
        match state.acquire_convert_slot() {
            Ok(_task) => {
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
            Err(code) => {
                tracing::warn!(
                    "Auto-conversion for {} skipped: another heavy job is running ({}) — convert via the model manager button later",
                    filename,
                    code
                );
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
    // S66: single-flight + heavy-job interlock (also registers the close-flow "convert" entry).
    let _task = state.acquire_convert_slot()?;
    let dir = &state.msst_models_dir;
    let path = dir.join(&filename);
    if !path.exists() {
        return Err(format!("MSST_FILE_NOT_FOUND: {}", filename));
    }

    if precision.as_deref() == Some("fp16") {
        let fp32_onnx = path.with_extension("onnx");
        if fp32_onnx.exists() {
            let out = run_fp16_converter(&fp32_onnx, &state.app_dir)
                .await
                .map_err(|e| format!("MSST_CONVERT_FAILED: {}", e))?;
            // S68c: a RE-conversion may have replaced a file the engine still holds a cached
            // session for — evict by stem prefix (covers .onnx + .fp16.onnx) so the next run
            // builds from the fresh bytes instead of silently serving the old graph.
            state.inference.engine.unload_paths_with_prefix(&path.with_extension(""));
            return Ok(out);
        }
    }

    let arch = resolve_architecture(architecture.as_deref(), &filename);
    if arch == "unknown" {
        return Err(format!("MSST_ARCH_UNKNOWN: {}", filename));
    }

    let onnx_path = run_converter(&path, &arch, &state.app_dir, precision.as_deref())
        .await
        .map_err(|e| format!("MSST_CONVERT_FAILED: {}", e))?;
    state.inference.engine.unload_paths_with_prefix(&path.with_extension(""));

    Ok(onnx_path)
}

#[tauri::command]
pub fn delete_msst_model(
    state: State<'_, Arc<AppState>>,
    filename: String,
) -> Result<(), String> {
    // S66: a running conversion is writing .onnx/.json beside this file — deleting mid-export
    // would race the converter (half files, converter crash on a vanished input).
    if state.task_active("convert") {
        return Err("CONVERT_BUSY".into());
    }
    let dir = &state.msst_models_dir;
    let path = dir.join(&filename);
    if path.exists() {
        std::fs::remove_file(&path).map_err(|e| format!("MSST_DELETE_FAILED: {e}"))?;
    }

    let stem = PathBuf::from(&filename);
    let stem_name = stem.file_stem().unwrap_or_default().to_string_lossy();

    for ext in &["json", "yaml", "yml", "onnx", "fp16.onnx", "fp16.recipe"] {
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
    let fp16_path = crate::separation::fp16_sibling(&fp32_onnx);
    let has_fp16 = fp16_path.exists();
    let fp16_recipe_ok = has_fp16
        && std::fs::read_to_string(fp16_path.with_extension("recipe"))
            .map(|s| s.trim() == FP16_RECIPE)
            .unwrap_or(false);
    let (stem_names, residual_name, num_overlap) = read_json_fields(&fp32_onnx);

    tracing::info!("Imported MSST model: {}", filename);
    Ok(MsstModelFile {
        filename,
        size,
        architecture,
        has_onnx,
        has_fp16,
        fp16_recipe_ok,
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

    // Success check must match what the requested precision actually PRODUCES:
    // `--precision fp16` deliberately deletes the fp32 intermediate (convert.py keeps only
    // <stem>.fp16.onnx + the shared json) — blindly requiring .onnx here reported
    // "Converter finished but .onnx file not found" on fully SUCCESSFUL conversions
    // (the S68b YiMing bs_roformer mystery, and every fp16-default download/重转 since:
    // the model was usable the whole time, only this check cried failure).
    let fp16_path = crate::separation::fp16_sibling(&onnx_path);
    match precision {
        Some("fp16") => {
            if !fp16_path.exists() {
                return Err("Converter finished but .fp16.onnx file not found".into());
            }
            Ok(fp16_path.to_string_lossy().to_string())
        }
        Some("both") => {
            if !onnx_path.exists() || !fp16_path.exists() {
                return Err("Converter finished but .onnx/.fp16.onnx pair not found".into());
            }
            Ok(onnx_path.to_string_lossy().to_string())
        }
        _ => {
            if !onnx_path.exists() {
                return Err("Converter finished but .onnx file not found".into());
            }
            Ok(onnx_path.to_string_lossy().to_string())
        }
    }
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
