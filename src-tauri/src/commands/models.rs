use std::path::{Path, PathBuf};
use std::sync::Arc;
use tauri::State;

use crate::models::{ImportOutcome, ModelEntry, ModelType};
use crate::AppState;

/// THE mapping from the frontend's model-type vocabulary to a registry type. `pub(crate)` so
/// the training project page can resolve its export ledger's rows against the live registry
/// through the same table the import/delete path validates with — a second copy would drift
/// the moment a backend is added.
pub(crate) fn parse_voice_type(model_type: &str) -> Option<ModelType> {
    match model_type {
        "rvc" => Some(ModelType::Rvc),
        "sovits" => Some(ModelType::SoVits),
        // S68: training-page imports pass snapshot.backend verbatim; a 4.0-v2
        // training product is a SoVITS resource (converter auto-detects v2)
        "sovits_v2" => Some(ModelType::SoVits),
        // S40: the vocoder RESOURCE class (fine-tuned / imported NSF-HiFiGAN
        // vocoders under models/nsf_hifigan/); the aux default vocoder stays
        // aux-resolved and outside the registry
        "vocoder" => Some(ModelType::NsfHifigan),
        _ => None,
    }
}

#[tauri::command]
pub async fn list_models(
    state: State<'_, Arc<AppState>>,
    model_type: Option<String>,
) -> Result<Vec<ModelEntry>, String> {
    // Explicit rescan (not just the registry's lazy one): the manager UI calls this after
    // imports/deletes and must reflect on-disk reality.
    state.models.scan().map_err(|e| e.to_string())?;

    match model_type.as_deref() {
        Some("rvc") => Ok(state.models.list_by_type(&ModelType::Rvc)),
        Some("sovits") => Ok(state.models.list_by_type(&ModelType::SoVits)),
        Some("s2h") => Ok(state.models.list_by_type(&ModelType::S2H)),
        Some("f0") => Ok(state.models.list_by_type(&ModelType::F0)),
        Some("nsf_hifigan") => Ok(state.models.list_by_type(&ModelType::NsfHifigan)),
        // S40 alias — the frontend voice store speaks "vocoder" everywhere
        // (import/delete via parse_voice_type do too)
        Some("vocoder") => Ok(state.models.list_by_type(&ModelType::NsfHifigan)),
        _ => Ok(state.models.list()),
    }
}

/// Returns the created entry PLUS non-fatal warnings (failed index conversion, synthesized
/// sidecar config, avatar problems) — the frontend must surface these, not just "success".
///
/// `source_ckpt` (S76 batch 4) = the training CHECKPOINT this import really came from, when
/// `path` is not it. The training page's batch import hands us the audition cache's
/// already-converted `<slot>/audition/<stem>/model.onnx` to skip a 10-30s reconversion —
/// recording THAT in the export ledger would leave the real snapshot looking un-imported (so
/// batch 3's「清理未导入的快照」could delete it) and would file the row against a path that
/// 「清理试听缓存」removes.
#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn import_model(
    state: State<'_, Arc<AppState>>,
    name: String,
    path: String,
    model_type: String,
    index_path: Option<String>,
    diffusion_path: Option<String>,
    diffusion_config_path: Option<String>,
    avatar_path: Option<String>,
    vocoder_config_path: Option<String>,
    source_ckpt: Option<String>,
) -> Result<ImportOutcome, String> {
    let mt = parse_voice_type(&model_type)
        .ok_or_else(|| format!("Unsupported model type: {}", model_type))?;

    // S60-4 audit: block REPLACE while an audition render is in flight (see delete_model — the
    // stale audition wav would land beside the NEW model's files at completion).
    if crate::commands::audition::AUDITION_IN_FLIGHT.load(std::sync::atomic::Ordering::SeqCst) {
        return Err("MODEL_BUSY_AUDITION".to_string());
    }
    // S66: import runs the torch→ONNX conversion synchronously — take the app-wide convert
    // slot (single-flight + heavy-job interlock; also lists "convert" in the close-flow).
    let _convert = state.acquire_convert_slot()?;
    // A same-name re-import REPLACES the model on disk — drop any live inference session first,
    // or it would keep serving the stale ONNX (and leak the old RvcIndex RAM).
    state.inference.unload_voice(&name);
    // Vocoder resources are cached BY PATH (engine session + mel filterbank
    // npy), not by voice name — evict them too before the files are replaced
    // (设计红队 A18; a live session also holds a Windows file lock).
    if matches!(mt, ModelType::NsfHifigan) {
        if let Some(old) = state.models.get_by_type(&name, &mt) {
            state.inference.unload_model_file(&old.path);
        }
    }

    let src = PathBuf::from(&path);
    let idx = index_path.map(PathBuf::from);
    let diff = diffusion_path.map(PathBuf::from);
    let diff_cfg = diffusion_config_path.map(PathBuf::from);
    let avatar = avatar_path.map(PathBuf::from);
    let voc_cfg = vocoder_config_path.map(PathBuf::from);
    let outcome = state
        .models
        .import_file(
            &name,
            &src,
            mt,
            &state.app_dir,
            idx.as_deref(),
            diff.as_deref(),
            diff_cfg.as_deref(),
            avatar.as_deref(),
            voc_cfg.as_deref(),
        )
        .map_err(|e| {
            tracing::error!("Model import failed: {}", e);
            e.to_string()
        });
    // S76: note the export in the source project's ledger — THE single place this happens.
    //
    // It used to be a second RPC issued by the training page, which missed two paths entirely:
    // the resource manager's file picker (a user can browse straight into
    // `<data>/training/<proj>/<family>/weights/*.pth`) and any interruption between the import
    // and the follow-up call. Both left「已安装模型 + 账本无此行」— and the snapshot cleanup
    // reads a missing row as「没人要」, i.e. it fails OPEN on the one thing it must not.
    if let Ok(entry) = &outcome {
        let ledger_src = source_ckpt.as_deref().map(PathBuf::from).unwrap_or_else(|| src.clone());
        record_training_export(&state, &ledger_src, &entry.entry.name, &model_type);
    }
    outcome
}

/// Best-effort ledger write for a model imported straight out of a training slot. Never fails
/// the import: the model IS installed either way, and the cleanup is conservative when the
/// ledger is thin (it also protects anything a registry lookup can still reach).
fn record_training_export(state: &AppState, src: &Path, name: &str, model_type: &str) {
    let data_dir = state
        .models
        .models_dir()
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| state.app_dir.join("data"));
    let root = crate::training::tproject::training_root(&data_dir);
    let Ok(rel) = src.strip_prefix(&root) else { return };
    // `<project_id>/<family>/...` — the first component identifies the project.
    let Some(project_id) = rel.components().next().map(|c| c.as_os_str().to_string_lossy().into_owned())
    else {
        return;
    };
    if let Err(e) = crate::training::tproject::record_export(
        &data_dir,
        &project_id,
        name,
        model_type,
        &src.to_string_lossy(),
    ) {
        tracing::warn!("export ledger not updated for {}: {e}", src.display());
    }
}

/// Attach a TRAINED shallow-diffusion checkpoint (model_<step>.pt with its
/// config.yaml auto-resolved next to it) to an installed SoVITS model (S39).
/// Conversion + validation run into a temp dir BEFORE the model's live
/// sessions are dropped — a failure leaves an existing attachment untouched
/// and still loaded; the swap itself is rename-based with rollback.
#[tauri::command]
pub async fn attach_diffusion(
    state: State<'_, Arc<AppState>>,
    name: String,
    ckpt_path: String,
    config_path: Option<String>,
) -> Result<ModelEntry, String> {
    // S66: the attachment converts (encoder/denoiser export) — same convert slot as import.
    let _convert = state.acquire_convert_slot()?;
    let cfg = config_path.map(PathBuf::from);
    let tmp = state
        .models
        .prepare_diffusion_attachment(
            &name,
            &PathBuf::from(&ckpt_path),
            cfg.as_deref(),
            &state.app_dir,
        )
        .map_err(|e| {
            tracing::error!("Diffusion attach (prepare) failed: {}", e);
            e.to_string()
        })?;
    // sessions hold Windows file handles on the OLD attachment — drop them
    // only now, after everything that can fail has succeeded
    state.inference.unload_voice(&name);
    state
        .models
        .commit_diffusion_attachment(&name, &tmp)
        .map_err(|e| {
            tracing::error!("Diffusion attach (commit) failed: {}", e);
            e.to_string()
        })?;
    // S76: attach IS the export path for shallow diffusion — its checkpoints never go through
    // import_model, so without this the whole sovits_diff family reads as never-imported.
    // `sovits` (not `sovits_diff`) is deliberate: the ledger's model_type names a REGISTRY
    // type, and the thing that got installed is a SoVITS model's attachment.
    record_training_export(&state, &PathBuf::from(&ckpt_path), &name, "sovits");
    state
        .models
        .get(&name)
        .ok_or_else(|| format!("MODEL_NOT_FOUND: {}", name))
}

#[tauri::command]
pub async fn set_model_avatar(
    state: State<'_, Arc<AppState>>,
    name: String,
    avatar_path: String,
) -> Result<Option<String>, String> {
    state
        .models
        .set_avatar(&name, &PathBuf::from(avatar_path))
        .map(|p| p.map(|x| x.to_string_lossy().to_string()))
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn delete_model(
    state: State<'_, Arc<AppState>>,
    name: String,
    model_type: Option<String>,
) -> Result<(), String> {
    // Type-scoped when the caller knows the type (设计红队 A5): an untyped
    // first-match delete of a vocoder named after its singer would remove the
    // SINGER MODEL's files (scan order rvc→sovits→…→nsf_hifigan).
    let mt = model_type.as_deref().and_then(parse_voice_type);
    // S60-4 audit: an in-flight audition writes `<stem>.audition_spk{N}.wav` NEXT TO the model
    // at completion — deleting (or REPLACE-importing, below) mid-render would let that stale
    // wav land beside the new files and impersonate the new model forever. Same guard class as
    // start_training/reset_training_display (S41-INT-4 check-then-act).
    if crate::commands::audition::AUDITION_IN_FLIGHT.load(std::sync::atomic::Ordering::SeqCst) {
        return Err("MODEL_BUSY_AUDITION".to_string());
    }
    // S66: a running conversion (import/attach) may be writing this model's files right now.
    if state.task_active("convert") {
        return Err("CONVERT_BUSY".into());
    }
    // Unload BEFORE removing files: a loaded session would keep serving the deleted model (and
    // on Windows can hold the .onnx file open, blocking removal).
    state.inference.unload_voice(&name);
    if let Some(ModelType::NsfHifigan) = mt {
        if let Some(old) = state.models.get_by_type(&name, &ModelType::NsfHifigan) {
            state.inference.unload_model_file(&old.path);
        }
    }
    state.models.delete(&name, mt.as_ref()).map_err(|e| e.to_string())
}

/// S60-2: persist a model's tested vocal range into its sidecar (the frontend-orchestrated
/// range test writes this; the render layer reads it back via vocal_range::speaker_range).
#[tauri::command]
pub async fn set_model_vocal_range(
    state: State<'_, Arc<AppState>>,
    name: String,
    model_type: String,
    record: serde_json::Value,
) -> Result<(), String> {
    let mt = parse_voice_type(&model_type).ok_or("RANGE_BAD_TYPE")?;
    if !record.is_object() {
        return Err("RANGE_BAD_RECORD".to_string());
    }
    crate::inference::vocal_range::validate_range_record(&record)?;
    state
        .models
        .set_config_extra_key(&name, &mt, "vocal_range", record)
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn check_model_exists(
    state: State<'_, Arc<AppState>>,
    name: String,
    model_type: String,
) -> Result<bool, String> {
    match parse_voice_type(&model_type) {
        Some(mt) => Ok(state.models.exists(&name, &mt)),
        None => Ok(false),
    }
}

/// S78 batch 7: bundle one installed voice model's full portable stem family into a single `.zip`
/// at `dest_path` (user-picked save dialog) so it can be carried to another install and re-imported
/// losslessly via `import_model_package`. Read-only w.r.t. the model, but guarded against a live
/// audition/conversion that could be writing the family under us (same guard class as delete).
/// The zip write runs off the async executor (spawn_blocking) — models are 10s–100s of MB.
#[tauri::command]
pub async fn export_model(
    state: State<'_, Arc<AppState>>,
    name: String,
    model_type: String,
    dest_path: String,
) -> Result<crate::models::ExportOutcome, String> {
    let mt = parse_voice_type(&model_type)
        .ok_or_else(|| "EXPORT_FAILED: unsupported model type".to_string())?;
    // An in-flight audition writes `<stem>.audition_spk{N}.wav` beside the model; a conversion may
    // be rewriting the family — either would make the zip a moving target.
    if crate::commands::audition::AUDITION_IN_FLIGHT.load(std::sync::atomic::Ordering::SeqCst) {
        return Err("MODEL_BUSY_AUDITION".to_string());
    }
    // Hold the app-wide convert slot for the WHOLE export (returns CONVERT_BUSY if a conversion is
    // running). This is the SAME interlock import/attach/import_package take, so a concurrent
    // same-model delete or REPLACE (both check task_active("convert")) can't remove a family member
    // between collect_stem_family and the per-file open, which would spuriously fail the export
    // (审查 S78: export was read-only but LOCKLESS — one-directional interlock gap). TaskGuard is
    // Send, so it is held across the spawn_blocking await below.
    let _convert = state.acquire_convert_slot()?;
    let entry = state
        .models
        .get_by_type(&name, &mt)
        .ok_or_else(|| "EXPORT_MODEL_NOT_FOUND".to_string())?;
    let dest = PathBuf::from(&dest_path);
    tauri::async_runtime::spawn_blocking(move || {
        crate::models::write_model_package(&entry, &model_type, &dest)
    })
    .await
    .map_err(|e| format!("EXPORT_FAILED: task join: {e}"))?
}

/// S78 batch 7: re-import a `.zip` model package produced by `export_model` — the lossless
/// counterpart. Extracts (zip-slip guarded) to a temp dir under the cache, reads the manifest for
/// the display name / registry type / stem, then materializes the whole family under a fresh local
/// stem (registry `import_package`). Same interlocks as `import_model` (convert slot + audition
/// guard + drop live sessions before a same-name REPLACE). NOT a training-slot source, so it writes
/// no export ledger row.
#[tauri::command]
pub async fn import_model_package(
    state: State<'_, Arc<AppState>>,
    package_path: String,
) -> Result<ImportOutcome, String> {
    if crate::commands::audition::AUDITION_IN_FLIGHT.load(std::sync::atomic::Ordering::SeqCst) {
        return Err("MODEL_BUSY_AUDITION".to_string());
    }
    let _convert = state.acquire_convert_slot()?;

    let (work, name, mt_str, stem) = extract_package(&state.cache_dir, &package_path)?;
    let mt = match parse_voice_type(&mt_str) {
        Some(m) => m,
        None => {
            let _ = std::fs::remove_dir_all(&work);
            return Err("PACKAGE_INVALID: unsupported model type".to_string());
        }
    };

    // A same-name re-import REPLACES on disk — drop any live inference session first (it would keep
    // serving the stale ONNX and, for vocoders cached by path, hold a Windows file lock).
    state.inference.unload_voice(&name);
    if matches!(mt, ModelType::NsfHifigan) {
        if let Some(old) = state.models.get_by_type(&name, &mt) {
            state.inference.unload_model_file(&old.path);
        }
    }

    let outcome = state
        .models
        .import_package(&name, mt, &work, &stem)
        .map_err(|e| {
            tracing::error!("Package import failed: {}", e);
            e.to_string()
        });
    let _ = std::fs::remove_dir_all(&work); // the temp extraction is regenerable
    outcome
}

/// Extract a `.zip` model package to a fresh temp dir under the cache and read its manifest.
/// Returns (extracted_dir, display_name, model_type_str, pkg_stem). Zip-slip guarded via
/// `enclosed_name()` (the `.usp` opener's discipline). The caller owns cleanup of the returned dir.
fn extract_package(
    cache_dir: &Path,
    package_path: &str,
) -> Result<(PathBuf, String, String, String), String> {
    use std::io::Read;
    let file = std::fs::File::open(package_path)
        .map_err(|e| format!("PACKAGE_EXTRACT_FAILED: open {package_path}: {e}"))?;
    let mut archive =
        zip::ZipArchive::new(file).map_err(|e| format!("PACKAGE_INVALID: not a zip: {e}"))?;

    // Unique staging dir per invocation so a repeat/concurrent import can't collide.
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let work = cache_dir.join("model_import").join(format!("pkg_{nonce}"));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work)
        .map_err(|e| format!("PACKAGE_EXTRACT_FAILED: create work dir: {e}"))?;

    let mut extract = || -> Result<String, String> {
        let mut manifest_text = String::new();
        for i in 0..archive.len() {
            let mut entry = archive
                .by_index(i)
                .map_err(|e| format!("PACKAGE_EXTRACT_FAILED: entry {i}: {e}"))?;
            let name = entry.name().to_string();
            if name == crate::models::PACKAGE_MANIFEST {
                entry
                    .read_to_string(&mut manifest_text)
                    .map_err(|e| format!("PACKAGE_EXTRACT_FAILED: read manifest: {e}"))?;
                continue;
            }
            if name.ends_with('/') {
                continue; // directory marker
            }
            // enclosed_name() rejects zip-slip (`..` traversal) — only extract contained paths.
            let rel = match entry.enclosed_name() {
                Some(p) => p.to_path_buf(),
                None => return Err(format!("PACKAGE_INVALID: unsafe archive entry: {name}")),
            };
            let outpath = work.join(rel);
            if let Some(parent) = outpath.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("PACKAGE_EXTRACT_FAILED: create {}: {e}", parent.display()))?;
            }
            let mut out = std::fs::File::create(&outpath)
                .map_err(|e| format!("PACKAGE_EXTRACT_FAILED: create {}: {e}", outpath.display()))?;
            std::io::copy(&mut entry, &mut out)
                .map_err(|e| format!("PACKAGE_EXTRACT_FAILED: extract {}: {e}", outpath.display()))?;
        }
        Ok(manifest_text)
    };
    let manifest_text = match extract() {
        Ok(t) => t,
        Err(e) => {
            let _ = std::fs::remove_dir_all(&work);
            return Err(e);
        }
    };

    let fail = |work: &Path, msg: &str| -> String {
        let _ = std::fs::remove_dir_all(work);
        msg.to_string()
    };
    if manifest_text.is_empty() {
        return Err(fail(&work, "PACKAGE_INVALID: no utaimodel.json manifest"));
    }
    let manifest: serde_json::Value = match serde_json::from_str(&manifest_text) {
        Ok(v) => v,
        Err(e) => return Err(fail(&work, &format!("PACKAGE_INVALID: manifest JSON: {e}"))),
    };
    if manifest.get("format").and_then(|v| v.as_str()) != Some(crate::models::PACKAGE_FORMAT) {
        return Err(fail(&work, "PACKAGE_INVALID: not a Utai model package"));
    }
    let name = manifest.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let mt_str = manifest.get("model_type").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let stem = manifest.get("stem").and_then(|v| v.as_str()).unwrap_or("").to_string();
    if name.is_empty() || mt_str.is_empty() || stem.is_empty() {
        return Err(fail(&work, "PACKAGE_INVALID: manifest missing name/model_type/stem"));
    }
    Ok((work, name, mt_str, stem))
}
