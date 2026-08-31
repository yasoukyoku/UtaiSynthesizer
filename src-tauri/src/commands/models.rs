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
/// ★S120 — returns `ImportOutcome`, not a bare `ModelEntry`. Reason: attaching is one of the
/// three ways a shallow-diffusion attachment becomes live, and §F9's import-time vocoder hint
/// has to be able to reach the user from here too. Reusing `ImportOutcome` (rather than minting
/// a second `{entry, warnings}` shape) keeps the frontend on ONE warning-rendering path — the
/// same `backendErrorMessage(w) ?? w` funnel `import_model` already uses.
/// ⚠ Shape change: the previous return value was the entry itself. The only caller
/// (`TrainingPage.attachCkpt`) discarded it, so nothing read the old shape.
pub async fn attach_diffusion(
    state: State<'_, Arc<AppState>>,
    name: String,
    ckpt_path: String,
    config_path: Option<String>,
) -> Result<ImportOutcome, String> {
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
    let (_dir, warnings) = state
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
    let entry = state
        .models
        .get(&name)
        .ok_or_else(|| format!("MODEL_NOT_FOUND: {}", name))?;
    Ok(ImportOutcome { entry, warnings })
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

/// S167: resolve an INSTALLED model back to a live training checkpoint through the projects'
/// export ledgers. The registry keeps no provenance — installed voices are ONNX, and the
/// community `.pth` cannot be reconstructed from them — so the ledger row
/// (`ExportedModel { name, model_type, from_ckpt_rel, .. }`) is the only honest bridge.
/// Newest matching live row whose checkpoint still exists wins. Returns the checkpoint path and
/// the row's backend string (the family carrier — a "sovits_v2" row installs as SoVits but its
/// family dispatch differs from "rvc").
fn resolve_community_source(
    data_dir: &Path,
    name: &str,
    model_type: &str,
) -> Option<(PathBuf, String)> {
    let want = parse_voice_type(model_type)?;
    let mut best: Option<(u64, PathBuf, String)> = None;
    for meta in crate::training::tproject::list_projects(data_dir) {
        let proj = crate::training::tproject::project_dir(data_dir, &meta.id);
        for row in &meta.exported {
            if row.name != name || !row.source_live() {
                continue;
            }
            if parse_voice_type(&row.model_type) != Some(want.clone()) {
                continue;
            }
            let ckpt = proj.join(&row.from_ckpt_rel);
            if !ckpt.is_file() {
                continue;
            }
            if best.as_ref().is_none_or(|(at, _, _)| row.at_ms > *at) {
                best = Some((row.at_ms, ckpt, row.model_type.clone()));
            }
        }
    }
    best.map(|(_, p, mt)| (p, mt))
}

/// S167: does an installed model have a live training-side source for community export?
/// Drives the enabled/disabled state of the「community format」choice in the manager's
/// export flow — the honest answer is often "no" (imported packages carry no `.pth`).
#[tauri::command]
pub async fn has_community_source(
    state: State<'_, Arc<AppState>>,
    name: String,
    model_type: String,
) -> Result<bool, String> {
    if model_type == "vocoder" || parse_voice_type(&model_type).is_none() {
        return Ok(false);
    }
    let data_dir = crate::commands::training::data_root(&state);
    if resolve_community_source(&data_dir, &name, &model_type).is_some() {
        return Ok(true);
    }
    // S167c: the retained import source beside the ONNX (v0.12+ imports keep it)
    let Some(mt) = parse_voice_type(&model_type) else { return Ok(false) };
    let Some(entry) = state.models.get_by_type(&name, &mt) else { return Ok(false) };
    Ok(match retained_community_source(&entry.path) {
        Some((_, cfg)) => !matches!(mt, ModelType::SoVits) || cfg.is_some(),
        None => false,
    })
}

/// S167: export an INSTALLED model in the COMMUNITY-standard format — plain files into a
/// user-picked folder, no zip (user 2026-08-31). The `.pth` is resolved through the training
/// export ledger (see [`resolve_community_source`]); the file set itself comes from the same
/// shared builder as the training page (`training::community_export_files` — ONE contract).
/// Vocoder rows are refused: our fine-tuned NSF-HiFiGAN snapshots have no community standard.
#[tauri::command]
pub async fn export_model_community(
    state: State<'_, Arc<AppState>>,
    name: String,
    model_type: String,
    dest_dir: String,
) -> Result<Vec<String>, String> {
    if model_type == "vocoder" {
        return Err("EXPORT_COMMUNITY_UNSUPPORTED".to_string());
    }
    let mt = parse_voice_type(&model_type)
        .ok_or_else(|| "EXPORT_COMMUNITY_UNSUPPORTED".to_string())?;
    if crate::commands::audition::AUDITION_IN_FLIGHT.load(std::sync::atomic::Ordering::SeqCst) {
        return Err("MODEL_BUSY_AUDITION".to_string());
    }
    // per-row button in the manager ⇒ the registry entry must exist
    let entry = state
        .models
        .get_by_type(&name, &mt)
        .ok_or_else(|| "EXPORT_MODEL_NOT_FOUND".to_string())?;
    let dest = PathBuf::from(&dest_dir);
    if !dest.is_dir() {
        return Err("EXPORT_COMMUNITY_DEST_MISSING".to_string());
    }
    let data_dir = crate::commands::training::data_root(&state);
    let (ckpt, family, feat, cfg) = match resolve_community_source(&data_dir, &name, &model_type) {
        Some((ckpt, backend)) => {
            let ckpt = ckpt
                .canonicalize()
                .map_err(|e| format!("EXPORT_COMMUNITY_CKPT_MISSING: {e}"))?;
            let family = crate::training::backend_family(&backend).to_string();
            let (feat, cfg) = crate::commands::training::community_run_assets(&family, &ckpt)?;
            (ckpt, family, feat, cfg)
        }
        None => {
            // S167c: fall back to the RETAINED import source (`<stem>.src.*` beside the ONNX —
            // v0.12 imports keep it exactly for this; older imports have none ⇒ honest refusal,
            // one re-import unlocks it).
            let (ckpt, cfg) = retained_community_source(&entry.path)
                .ok_or_else(|| "EXPORT_COMMUNITY_NO_SOURCE".to_string())?;
            let family = match mt {
                ModelType::Rvc => "rvc",
                ModelType::SoVits => "sovits",
                _ => return Err("EXPORT_COMMUNITY_UNSUPPORTED".to_string()),
            };
            if matches!(mt, ModelType::SoVits) && cfg.is_none() {
                return Err("EXPORT_COMMUNITY_NO_SOURCE".to_string());
            }
            let feat = if matches!(mt, ModelType::Rvc) {
                let npy = entry.path.with_extension("npy");
                npy.is_file().then_some(npy)
            } else {
                None
            };
            (ckpt, family.to_string(), feat, cfg)
        }
    };
    let out_name = crate::models::sanitize_file_stem(name.trim());
    if out_name.is_empty() {
        return Err("TRAINING_NAME_EMPTY".to_string());
    }
    // Same interlock as export_model: the faiss index build runs under the converter role, and
    // holding the slot keeps a concurrent delete/REPLACE from pulling files mid-export.
    let _convert = state.acquire_convert_slot()?;
    crate::commands::training::community_export_files(
        state.app_dir.clone(),
        state.cache_dir.clone(),
        &family,
        ckpt,
        &out_name,
        dest,
        feat,
        cfg,
    )
    .await
}

/// S167c: the import-retained community source beside the installed ONNX — `<stem>.src.pth`
/// (or `.ckpt`/`.pt`) plus, for SoVITS, `<stem>.src.config.json` (see `Models::import_file`).
/// Pure over the ONNX path so the unit test needs no registry entry.
fn retained_community_source(onnx_path: &Path) -> Option<(PathBuf, Option<PathBuf>)> {
    let dir = onnx_path.parent()?;
    let stem = onnx_path.file_stem()?.to_string_lossy().to_string();
    let ckpt = ["pth", "ckpt", "pt"]
        .iter()
        .map(|e| dir.join(format!("{stem}.src.{e}")))
        .find(|p| p.is_file())?;
    let cfg = dir.join(format!("{stem}.src.config.json"));
    Some((ckpt, cfg.is_file().then_some(cfg)))
}

#[cfg(test)]
mod community_source_tests {
    use super::*;

    /// The resolver's four gates, each proven on a real on-disk ledger: name+type match,
    /// `source_live`, the checkpoint file existing, and newest-at_ms-wins.
    #[test]
    fn resolver_finds_only_live_matching_rows_with_existing_ckpts() {
        let root = std::env::temp_dir().join(format!("utai_commsrc_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let proj = root.join("training").join("p1_aaaabbbb");
        std::fs::create_dir_all(proj.join("rvc").join("weights")).unwrap();
        std::fs::write(proj.join("rvc").join("weights").join("m_e14_s147.pth"), b"x").unwrap();
        std::fs::write(proj.join("rvc").join("weights").join("m_e20_s200.pth"), b"y").unwrap();
        std::fs::write(
            proj.join("project.json"),
            r#"{"id":"p1_aaaabbbb","name":"p","exported":[
                {"name":"m","model_type":"rvc","from_ckpt_rel":"rvc/weights/m_e14_s147.pth","at_ms":9},
                {"name":"m","model_type":"rvc","from_ckpt_rel":"rvc/weights/m_e20_s200.pth","at_ms":20},
                {"name":"m","model_type":"rvc","from_ckpt_rel":"rvc/weights/gone.pth","at_ms":99},
                {"name":"dead","model_type":"rvc","from_ckpt_rel":"rvc/weights/m_e14_s147.pth","at_ms":5,"source_deleted_ms":7},
                {"name":"m","model_type":"sovits","from_ckpt_rel":"rvc/weights/m_e14_s147.pth","at_ms":50}
            ]}"#,
        )
        .unwrap();

        // newest LIVE row whose file exists wins — at_ms 99 points at a missing file and must lose
        let got = resolve_community_source(&root, "m", "rvc").expect("live rvc row");
        assert!(got.0.ends_with("m_e20_s200.pth"), "newest existing wins, got {:?}", got.0);
        assert_eq!(got.1, "rvc");
        // the sovits row for "m" resolves through the type table, not the raw string
        assert!(resolve_community_source(&root, "m", "sovits").is_some());
        // a tombstoned row must not resolve
        assert!(resolve_community_source(&root, "dead", "rvc").is_none());
        // vocoder / unknown types never resolve
        assert!(resolve_community_source(&root, "m", "vocoder").is_none());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// S167c: the retained-source fallback — `<stem>.src.*` beside the ONNX, SoVITS gated on
    /// its retained config (a `.pth` without the config it was converted with is not the
    /// community pair and must not light the button).
    #[test]
    fn retained_source_resolves_beside_the_onnx_and_sovits_needs_its_config() {
        let root = std::env::temp_dir().join(format!("utai_retained_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let onnx = root.join("たろう.onnx");
        std::fs::write(&onnx, b"o").unwrap();
        assert!(retained_community_source(&onnx).is_none(), "nothing retained ⇒ None");
        std::fs::write(root.join("たろう.src.pth"), b"p").unwrap();
        let (ckpt, cfg) = retained_community_source(&onnx).expect("pth retained");
        assert!(ckpt.ends_with("たろう.src.pth"));
        assert!(cfg.is_none(), "no config retained yet");
        std::fs::write(root.join("たろう.src.config.json"), b"{}").unwrap();
        let (_, cfg) = retained_community_source(&onnx).expect("pair retained");
        assert!(cfg.is_some(), "the retained config must surface for the sovits gate");
        // a prefix sibling must not leak into another stem's family
        let other = root.join("たろ.onnx");
        std::fs::write(&other, b"o").unwrap();
        assert!(retained_community_source(&other).is_none());
        let _ = std::fs::remove_dir_all(&root);
    }
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
