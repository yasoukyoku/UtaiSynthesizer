pub mod convert;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::Result;

pub struct ModelRegistry {
    models_dir: PathBuf,
    entries: RwLock<Vec<ModelEntry>>,
    /// Lazy self-scan flag: commands other than list_models (run_rvc / run_sovits /
    /// check_model_exists / import / delete) may be the FIRST registry access of a session, so
    /// every accessor calls `ensure_scanned` instead of relying on the UI listing models first.
    scanned: AtomicBool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub name: String,
    pub model_type: ModelType,
    pub format: ModelFormat,
    pub path: PathBuf,
    pub sample_rate: u32,
    pub config: ModelConfig,
    pub index_path: Option<PathBuf>,
    /// SoVITS S36: `<stem>.diffusion/` attachment dir (encoder+denoiser onnx +
    /// diffusion.json) — present iff the dir holds a diffusion.json. Gates 浅扩散.
    /// Mirrored by the TS `VoiceModelEntry` (src/store/voice-models.ts) — keep in sync.
    #[serde(default)]
    pub diffusion_path: Option<PathBuf>,
    pub avatar_path: Option<PathBuf>,
}

/// Result of an import: the created entry plus NON-FATAL problems (index conversion failure,
/// missing sidecar json, avatar copy failure). The model itself is usable — but the user must
/// see these instead of a silent "success" that quietly dropped the retrieval index.
#[derive(Debug, Clone, Serialize)]
pub struct ImportOutcome {
    pub entry: ModelEntry,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModelType {
    Rvc,
    SoVits,
    S2H,
    F0,
    NsfHifigan,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModelFormat {
    Onnx,
    Pth,
}

/// Models-dir subdirectory for a model type. Single source of truth — also the `--type` string
/// passed to converter/convert.py and the `type` field written into sidecar json.
pub(crate) fn type_subdir(model_type: &ModelType) -> &'static str {
    match model_type {
        ModelType::Rvc => "rvc",
        ModelType::SoVits => "sovits",
        ModelType::S2H => "s2h",
        ModelType::F0 => "f0",
        ModelType::NsfHifigan => "nsf_hifigan",
    }
}

/// Sidecar json contents. Every field is `#[serde(default)]`-tolerant: the converter workflow
/// extends this schema over time and older/foreign sidecars must still load. Unknown keys are
/// kept verbatim in `extra`, so `list_models` always exposes the FULL config to the frontend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Display name chosen at import time — written INTO the sidecar json so a disk rescan
    /// reconstructs the user's custom name losslessly (file stems are sanitized copies of it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default)]
    pub r#type: String,
    /// "v1" / "v2" (RVC) or "4.0" / "4.1" (SoVITS).
    #[serde(default = "default_version")]
    pub version: String,
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    #[serde(default = "default_features_dim")]
    pub features_dim: u32,
    #[serde(default)]
    pub n_speakers: u32,
    /// Speaker name → id. The v2 converter emits a map; legacy sidecars carried a plain list of
    /// names (mapped to their list index here). Tolerant of both, plus absent.
    #[serde(default = "default_speakers", deserialize_with = "de_speakers")]
    pub speakers: BTreeMap<String, u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speech_encoder: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hop_size: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vol_embedding: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit_interpolate_mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub noise: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inputs: Option<serde_json::Value>,
    #[serde(flatten)]
    pub extra: serde_json::Value,
}

impl ModelConfig {
    /// Effective ContentVec feature dim: speech_encoder wins when present (SoVITS sidecars;
    /// unknown encoder = error, never a silent fallback), else features_dim (RVC sidecars).
    /// THE single source — used by the inference command layer AND the import-time
    /// diffusion-attachment cross-check.
    pub fn resolved_features_dim(&self) -> std::result::Result<usize, String> {
        if let Some(enc) = self.speech_encoder.as_deref() {
            return match enc {
                "vec768l12" => Ok(768),
                "vec256l9" => Ok(256),
                other => Err(format!(
                    "不支持的 speech_encoder：{}（仅支持 vec768l12 / vec256l9）",
                    other
                )),
            };
        }
        Ok(self.features_dim as usize)
    }
}

fn default_version() -> String { "unknown".to_string() }
fn default_sample_rate() -> u32 { 40000 }
fn default_features_dim() -> u32 { 768 }
fn default_speakers() -> BTreeMap<String, u32> {
    BTreeMap::from([("default".to_string(), 0u32)])
}

/// Import-time default when the sidecar json carries no sample_rate at all.
fn default_sample_rate_for(model_type: &ModelType) -> u32 {
    match model_type {
        ModelType::SoVits => 44100,
        _ => default_sample_rate(),
    }
}

/// Accept `{"name": id, …}` (current converter), `["name", …]` (legacy list — index = id),
/// or anything else (→ default) without failing the whole config.
fn de_speakers<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<BTreeMap<String, u32>, D::Error> {
    let v = serde_json::Value::deserialize(deserializer)?;
    Ok(match v {
        serde_json::Value::Object(map) => map
            .iter()
            .filter_map(|(k, val)| val.as_u64().map(|id| (k.clone(), id as u32)))
            .collect(),
        serde_json::Value::Array(list) => list
            .iter()
            .enumerate()
            .filter_map(|(i, val)| val.as_str().map(|s| (s.to_string(), i as u32)))
            .collect(),
        _ => default_speakers(),
    })
}

const AVATAR_EXTS: &[&str] = &["png", "jpg", "jpeg", "bmp", "webp"];

impl ModelRegistry {
    pub fn new(models_dir: PathBuf) -> Self {
        Self {
            models_dir,
            entries: RwLock::new(Vec::new()),
            scanned: AtomicBool::new(false),
        }
    }

    /// Root models directory (aux voice models — ContentVec/RMVPE — live in <models_dir>/aux).
    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    /// Lazy self-scan: any read on a fresh session populates the registry from disk first.
    fn ensure_scanned(&self) {
        if !self.scanned.load(Ordering::Acquire) {
            if let Err(e) = self.scan() {
                tracing::warn!("Initial model scan failed: {}", e);
            }
        }
    }

    pub fn scan(&self) -> Result<()> {
        let mut fresh: Vec<ModelEntry> = Vec::new();

        for subdir in &["rvc", "sovits", "s2h", "f0", "nsf_hifigan"] {
            let dir = self.models_dir.join(subdir);
            if !dir.exists() {
                continue;
            }

            let model_type = match *subdir {
                "rvc" => ModelType::Rvc,
                "sovits" => ModelType::SoVits,
                "s2h" => ModelType::S2H,
                "f0" => ModelType::F0,
                "nsf_hifigan" => ModelType::NsfHifigan,
                _ => continue,
            };

            if let Ok(read_dir) = std::fs::read_dir(&dir) {
                for entry in read_dir.flatten() {
                    let path = entry.path();
                    if path.extension().map(|e| e == "onnx").unwrap_or(false) {
                        let stem = path
                            .file_stem()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .to_string();
                        // `<stem>.f0.onnx` is the SoVITS auto-f0 predictor COMPANION of
                        // `<stem>.onnx` (S36), not a model of its own — a phantom entry here
                        // would show up in the UI and crash on run. Only skip when the base
                        // model actually exists, so a model legitimately NAMED `…​.f0`
                        // (sanitize keeps dots) doesn't silently vanish from the list.
                        if let Some(base) = stem.strip_suffix(".f0") {
                            if dir.join(format!("{}.onnx", base)).exists() {
                                continue;
                            }
                        }
                        // `<stem>.selfcheck.onnx` = the vocoder exporter's deterministic
                        // twin (deleted in its finally; survives only a hard kill) —
                        // never a model, unconditionally invisible (S40 审查).
                        if stem.ends_with(".selfcheck") {
                            continue;
                        }

                        let config_path = path.with_extension("json");
                        let config = if config_path.exists() {
                            let content = std::fs::read_to_string(&config_path).ok();
                            content
                                .and_then(|c| serde_json::from_str(&c).ok())
                                .unwrap_or_else(default_config)
                        } else {
                            default_config()
                        };

                        // The user-chosen display name lives in the sidecar json ("name") so it
                        // survives rescans; pre-existing sidecars without it fall back to the stem.
                        let name = config.name.clone().unwrap_or_else(|| stem.clone());
                        let sample_rate = config.sample_rate;

                        // Every artifact shares the onnx stem — index/avatar reconstruct from it.
                        // SoVITS cluster/retrieval assets live in `<stem>.cluster/` instead of a
                        // flat sibling .npy (multiple models share one subdir — flat spk-id names
                        // would collide).
                        let index_path = path.with_extension("npy");
                        let index = if index_path.exists() {
                            Some(index_path)
                        } else if matches!(model_type, ModelType::SoVits) {
                            first_cluster_asset(&path.parent().unwrap_or(&dir).join(format!("{}.cluster", stem)))
                        } else {
                            None
                        };

                        let avatar = find_avatar(&path);

                        // SoVITS shallow-diffusion attachment: `<stem>.diffusion/` counts
                        // only when its diffusion.json exists (a half-written dir from a
                        // failed conversion must not light the 浅扩散 UI).
                        let diffusion = if matches!(model_type, ModelType::SoVits) {
                            let d = path
                                .parent()
                                .unwrap_or(&dir)
                                .join(format!("{}.diffusion", stem));
                            if d.join("diffusion.json").exists() { Some(d) } else { None }
                        } else {
                            None
                        };

                        fresh.push(ModelEntry {
                            name,
                            model_type: model_type.clone(),
                            format: ModelFormat::Onnx,
                            path,
                            sample_rate,
                            config,
                            index_path: index,
                            diffusion_path: diffusion,
                            avatar_path: avatar,
                        });
                    }
                }
            }
        }

        {
            let mut entries = self.entries.write();
            if entries.len() != fresh.len() {
                tracing::info!("Model scan: {} models found (was {})", fresh.len(), entries.len());
            }
            *entries = fresh;
        }
        self.scanned.store(true, Ordering::Release);
        Ok(())
    }

    pub fn list(&self) -> Vec<ModelEntry> {
        self.ensure_scanned();
        self.entries.read().clone()
    }

    pub fn list_by_type(&self, model_type: &ModelType) -> Vec<ModelEntry> {
        self.ensure_scanned();
        self.entries
            .read()
            .iter()
            .filter(|e| std::mem::discriminant(&e.model_type) == std::mem::discriminant(model_type))
            .cloned()
            .collect()
    }

    pub fn get(&self, name: &str) -> Option<ModelEntry> {
        self.ensure_scanned();
        self.entries.read().iter().find(|e| e.name == name).cloned()
    }

    /// Type-scoped lookup. Same-name entries across types are a STANDARD
    /// workflow here (an rvc+sovits pair per singer — S39 review F3; S40 adds
    /// vocoders named after the singer too), so any consumer that knows the
    /// type must use this instead of the first-match `get`.
    pub fn get_by_type(&self, name: &str, model_type: &ModelType) -> Option<ModelEntry> {
        self.ensure_scanned();
        self.entries
            .read()
            .iter()
            .find(|e| {
                e.name == name
                    && std::mem::discriminant(&e.model_type)
                        == std::mem::discriminant(model_type)
            })
            .cloned()
    }

    pub fn exists(&self, name: &str, model_type: &ModelType) -> bool {
        self.ensure_scanned();
        self.entries.read().iter().any(|e| {
            e.name == name
                && std::mem::discriminant(&e.model_type) == std::mem::discriminant(model_type)
        })
    }

    /// Import a model file (.pth → converted to ONNX; .onnx → copied directly). EVERYTHING is
    /// keyed on the user-chosen `name`: onnx / sidecar json / index npy / avatar share one
    /// sanitized stem, and the display name is written into the sidecar json so a rescan
    /// reconstructs the entry losslessly.
    ///
    /// A same-name same-type re-import REPLACES the old entry (files removed first — which also
    /// keeps the stem stable). The caller must unload any live inference session for `name`
    /// BEFORE calling (commands/models.rs does), or the old session would keep serving stale ONNX.
    #[allow(clippy::too_many_arguments)]
    pub fn import_file(
        &self,
        name: &str,
        src_path: &Path,
        model_type: ModelType,
        app_dir: &Path,
        index_file: Option<&Path>,
        diffusion_file: Option<&Path>,
        diffusion_config: Option<&Path>,
        avatar_file: Option<&Path>,
        vocoder_config: Option<&Path>,
    ) -> Result<ImportOutcome> {
        self.ensure_scanned();
        let mut warnings: Vec<String> = Vec::new();

        let subdir = self.models_dir.join(type_subdir(&model_type));
        std::fs::create_dir_all(&subdir)?;

        // ---- S40 vocoder PRE-FLIGHT (审查 HIGH 修复): everything that can fail
        // must fail BEFORE the destructive REPLACE below — a failed import must
        // neither destroy the old resource nor leave scan-able residue that
        // resurrects a self-check-rejected graph as a selectable ghost.
        //   direct .onnx  → validate the source triple in place;
        //   torch ckpt    → convert + ORT-self-check into a TEMP dir
        //                   (attach_diffusion pattern), then import the temp
        //                   output through the ordinary direct-onnx flow.
        let mut vocoder_tmp: Option<PathBuf> = None;
        let mut effective_src = src_path.to_path_buf();
        if matches!(model_type, ModelType::NsfHifigan) {
            let src_is_onnx = src_path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("onnx"))
                .unwrap_or(false);
            if src_is_onnx {
                read_vocoder_source(src_path)?; // json + fields + mini_nsf + npy
            } else {
                let tmp = subdir.join(".vocimport_tmp");
                if tmp.exists() {
                    // only ever populated mid-import — an existing one is a
                    // crashed run's residue
                    std::fs::remove_dir_all(&tmp).ok();
                }
                std::fs::create_dir_all(&tmp)?;
                if let Err(e) = convert::convert_vocoder_to_onnx(
                    src_path,
                    vocoder_config,
                    &tmp,
                    "vocoder",
                    app_dir,
                ) {
                    std::fs::remove_dir_all(&tmp).ok();
                    return Err(e);
                }
                effective_src = tmp.join("vocoder.onnx");
                vocoder_tmp = Some(tmp);
            }
        }
        let src_path = effective_src.as_path();

        // Same-name re-import = REPLACE, not a duplicate row + silent file overwrite.
        let old_entry = {
            let mut entries = self.entries.write();
            entries
                .iter()
                .position(|e| {
                    e.name == name
                        && std::mem::discriminant(&e.model_type)
                            == std::mem::discriminant(&model_type)
                })
                .map(|pos| entries.remove(pos))
        };
        if let Some(old) = old_entry {
            remove_entry_files(&old);
            tracing::info!("Re-import: replaced existing model '{}'", name);
        }

        // One sanitized stem for every artifact; different display names that sanitize to the
        // same filename get a numbered suffix instead of clobbering each other.
        let stem = unique_stem(&subdir, &sanitize_file_stem(name));
        let onnx_path = subdir.join(format!("{}.onnx", stem));

        let is_onnx = src_path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("onnx"))
            .unwrap_or(false);
        if is_onnx {
            // Direct ONNX import: no conversion. Bring a same-stem sidecar json along if the
            // source has one; otherwise finalize_sidecar synthesizes a minimal default below.
            // The materialization is guarded: a partial trio left behind by an IO failure
            // would resurrect on the next scan() as a broken ghost entry (审查 HIGH).
            let materialized: Result<()> = (|| {
                std::fs::copy(src_path, &onnx_path)?;
                let src_json = src_path.with_extension("json");
                if src_json.exists() {
                    std::fs::copy(&src_json, onnx_path.with_extension("json"))?;
                }
                // S40 vocoder resource: a converter-exported vocoder is a THREE-piece
                // set — the mel filterbank npy must travel and the sidecar's
                // mel_filters field must follow the (possibly re-sanitized) stem;
                // without them the vocoder is 100% unusable, so failures are FATAL
                // here, not warnings (设计红队 A6). The source was pre-flight
                // validated above, so failures here are raw IO only.
                if matches!(model_type, ModelType::NsfHifigan) {
                    import_vocoder_filterbank(src_path, &onnx_path, &stem)?;
                }
                Ok(())
            })();
            if let Err(e) = materialized {
                sweep_partial_import(&subdir, &stem);
                if let Some(tmp) = &vocoder_tmp {
                    std::fs::remove_dir_all(tmp).ok();
                }
                return Err(e);
            }
            // S36 companions travel with a converter-exported .onnx: the auto-f0 predictor
            // (`<stem>.f0.onnx` — the sidecar may claim auto_f0.available and the run would
            // hard-error without the file) and a converted `.diffusion/` attachment dir.
            let src_f0 = src_path.with_extension("f0.onnx");
            if src_f0.exists() {
                if let Err(e) = std::fs::copy(&src_f0, onnx_path.with_extension("f0.onnx")) {
                    warnings.push(format!("自动音高预测器复制失败：{}——自动f0不可用", e));
                }
            }
            if let (Some(src_dir), Some(src_stem)) = (src_path.parent(), src_path.file_stem()) {
                let src_diff = src_dir.join(format!("{}.diffusion", src_stem.to_string_lossy()));
                if src_diff.join("diffusion.json").exists() {
                    let dest_diff = subdir.join(format!("{}.diffusion", stem));
                    if let Err(e) = copy_dir_flat(&src_diff, &dest_diff) {
                        warnings.push(format!("扩散附件复制失败：{}——浅扩散不可用", e));
                        std::fs::remove_dir_all(&dest_diff).ok();
                    }
                }
            }
        } else {
            // (NsfHifigan never reaches here: its torch-ckpt sources were
            // converted into the temp dir during pre-flight, making
            // effective_src an .onnx — the branch above handles it.)
            convert::convert_pth_to_onnx(src_path, &onnx_path, &model_type, app_dir)?;
        }

        let config = match finalize_sidecar(&onnx_path, name, &model_type, &mut warnings) {
            Ok(c) => c,
            Err(e) => {
                if matches!(model_type, ModelType::NsfHifigan) {
                    sweep_partial_import(&subdir, &stem);
                }
                if let Some(tmp) = &vocoder_tmp {
                    std::fs::remove_dir_all(tmp).ok();
                }
                return Err(e);
            }
        };
        let sample_rate = config.sample_rate;
        if let Some(tmp) = &vocoder_tmp {
            std::fs::remove_dir_all(tmp).ok();
        }

        let index_path =
            self.resolve_index(&stem, &model_type, src_path, index_file, app_dir, &mut warnings);

        let diffusion_path = self
            .resolve_diffusion_assets(
                &stem,
                &model_type,
                &config,
                diffusion_file,
                diffusion_config,
                app_dir,
                &mut warnings,
            )
            .or_else(|| {
                // A direct-.onnx import may have brought an already-converted `.diffusion/`
                // dir along (copied above) — surface it on the entry like scan() would.
                let d = subdir.join(format!("{}.diffusion", stem));
                if d.join("diffusion.json").exists() { Some(d) } else { None }
            });

        let avatar_path = match avatar_file {
            Some(src) if src.exists() => match copy_avatar(&subdir, &stem, src) {
                Ok(p) => {
                    tracing::info!("Imported avatar for {}: {}", name, p.display());
                    Some(p)
                }
                Err(e) => {
                    tracing::warn!("Avatar import failed: {}", e);
                    warnings.push(format!("头像导入失败：{}", e));
                    None
                }
            },
            Some(src) => {
                warnings.push(format!("头像文件不存在：{}", src.display()));
                None
            }
            None => None,
        };

        let entry = ModelEntry {
            name: name.to_string(),
            model_type,
            format: ModelFormat::Onnx,
            path: onnx_path,
            sample_rate,
            config,
            index_path,
            diffusion_path,
            avatar_path,
        };

        self.entries.write().push(entry.clone());
        Ok(ImportOutcome { entry, warnings })
    }

    /// SoVITS shallow-diffusion attachment (S36): the user-picked diffusion `.pt` (+ optional
    /// `.yaml`) is converted (converter/export_diffusion.py) into `<subdir>/<stem>.diffusion/`
    /// (encoder.onnx + denoiser.onnx + diffusion.json). Failures are WARNINGS (model still
    /// imports, 浅扩散 unavailable — the cluster-asset posture). A successful conversion is
    /// additionally cross-checked against the MAIN model: the diffusion model's ContentVec
    /// dim must match, else the pair could never run together and the dir is dropped.
    fn resolve_diffusion_assets(
        &self,
        stem: &str,
        model_type: &ModelType,
        main_config: &ModelConfig,
        diffusion_file: Option<&Path>,
        diffusion_config: Option<&Path>,
        app_dir: &Path,
        warnings: &mut Vec<String>,
    ) -> Option<PathBuf> {
        let src = diffusion_file?;
        if !matches!(model_type, ModelType::SoVits) {
            warnings.push("扩散模型仅支持 SoVITS——已忽略".to_string());
            return None;
        }
        if !src.exists() {
            warnings.push(format!("扩散模型文件不存在：{}", src.display()));
            return None;
        }
        let diffusion_dir = self
            .models_dir
            .join(type_subdir(&ModelType::SoVits))
            .join(format!("{}.diffusion", stem));

        if let Err(e) =
            convert::convert_diffusion_assets(src, diffusion_config, &diffusion_dir, app_dir)
        {
            tracing::warn!("Diffusion conversion failed for {}: {}", src.display(), e);
            warnings.push(format!(
                "扩散模型转换失败（{}）：{}——模型已导入，浅扩散不可用",
                src.file_name().unwrap_or_default().to_string_lossy(),
                e
            ));
            std::fs::remove_dir_all(&diffusion_dir).ok(); // no half-written attachment
            return None;
        }

        // Cross-check: diffusion encoder dim vs the main model's ContentVec dim
        // (resolved_features_dim = the shared single source; an unknown speech_encoder
        // would already have failed the main import, treat it as unknown here).
        let enc_dim = diffusion_sidecar_dim(&diffusion_dir);
        let main_dim = main_config.resolved_features_dim().ok().map(|d| d as u64);
        if let (Some(ed), Some(md)) = (enc_dim, main_dim) {
            if ed != md {
                warnings.push(format!(
                    "扩散模型的特征维度（{}）与主模型（{}）不一致——两者无法配合使用，已移除该扩散附件",
                    ed, md
                ));
                std::fs::remove_dir_all(&diffusion_dir).ok();
                return None;
            }
        }
        Some(diffusion_dir)
    }

    /// S39 attach flow, step 1 — SAFE: the live attachment is untouched.
    /// Convert the trained diffusion .pt (+ auto-resolved yaml) into a TEMP
    /// dir next to the target model and cross-check the encoder dim. Any
    /// failure removes the temp dir and changes nothing else. Returns the
    /// temp dir for commit_diffusion_attachment.
    pub fn prepare_diffusion_attachment(
        &self,
        name: &str,
        ckpt: &Path,
        config: Option<&Path>,
        app_dir: &Path,
    ) -> Result<PathBuf> {
        self.ensure_scanned();
        // type-filtered lookup: an RVC model may share the display name (the
        // standard dual-backend workflow) — a name-only get() would hit it and
        // dead-end the attach (review F11)
        let entry = self
            .entries
            .read()
            .iter()
            .find(|e| e.name == name && matches!(e.model_type, ModelType::SoVits))
            .cloned()
            .ok_or_else(|| {
                crate::UtaiError::Model(format!("找不到 SoVITS 模型「{}」", name))
            })?;
        if !ckpt.exists() {
            return Err(crate::UtaiError::Model(format!(
                "扩散模型文件不存在：{}",
                ckpt.display()
            )));
        }
        let (subdir, stem) = entry_dir_and_stem(&entry)?;
        // unique temp dir per invocation: a re-entrant attach (the UI guard is
        // component-local state and dies on unmount) must not delete a
        // conversion already in flight. Stale .tmp*/.old* dirs (crashed or
        // failed attempts, ~hundreds of MB of onnx each) are swept first, but
        // only when older than an hour — a fresh one may be a live conversion.
        sweep_stale_attachment_dirs(&subdir, &stem);
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        let tmp_dir = subdir.join(format!("{}.diffusion.tmp{}", stem, nonce));
        if tmp_dir.exists() {
            std::fs::remove_dir_all(&tmp_dir)?;
        }
        if let Err(e) = convert::convert_diffusion_assets(ckpt, config, &tmp_dir, app_dir) {
            std::fs::remove_dir_all(&tmp_dir).ok();
            return Err(e);
        }
        let enc_dim = diffusion_sidecar_dim(&tmp_dir);
        let main_dim = entry.config.resolved_features_dim().ok().map(|d| d as u64);
        if let (Some(ed), Some(md)) = (enc_dim, main_dim) {
            if ed != md {
                std::fs::remove_dir_all(&tmp_dir).ok();
                return Err(crate::UtaiError::Model(format!(
                    "扩散模型的特征维度（{}）与主模型「{}」（{}）不一致——两者无法配合使用",
                    ed, name, md
                )));
            }
        }
        Ok(tmp_dir)
    }

    /// S39 attach flow, step 2 — the caller must have dropped the model's live
    /// sessions FIRST (they hold Windows file handles on the OLD attachment):
    /// swap the prepared temp dir into place, rolling back on failure.
    pub fn commit_diffusion_attachment(&self, name: &str, tmp_dir: &Path) -> Result<PathBuf> {
        self.ensure_scanned();
        let mut entries = self.entries.write();
        let entry = entries
            .iter_mut()
            .find(|e| e.name == name && matches!(e.model_type, ModelType::SoVits))
            .ok_or_else(|| crate::UtaiError::Model(format!("找不到模型「{}」", name)))?;
        let (subdir, stem) = entry_dir_and_stem(entry)?;
        let final_dir = subdir.join(format!("{}.diffusion", stem));
        let bak = subdir.join(format!("{}.diffusion.old", stem));
        if bak.exists() {
            std::fs::remove_dir_all(&bak)?;
        }
        let had_old = final_dir.exists();
        if had_old {
            std::fs::rename(&final_dir, &bak)
                .map_err(|e| crate::UtaiError::Model(format!("移出旧扩散附件失败: {}", e)))?;
        }
        if let Err(e) = std::fs::rename(tmp_dir, &final_dir) {
            if had_old {
                let _ = std::fs::rename(&bak, &final_dir); // rollback
            }
            let _ = std::fs::remove_dir_all(tmp_dir); // no half-attach residue
            return Err(crate::UtaiError::Model(format!("扩散附件替换失败: {}", e)));
        }
        if had_old {
            let _ = std::fs::remove_dir_all(&bak);
        }
        entry.diffusion_path = Some(final_dir.clone());
        tracing::info!("attached diffusion assets for {}: {}", name, final_dir.display());
        Ok(final_dir)
    }

    /// S60-2: write ONE extra key into a model's sidecar json (raw-map update — unknown keys
    /// are preserved verbatim, the finalize_sidecar discipline: never round-trip through the
    /// typed struct) and refresh the in-memory entry so `list_models` exposes it without a
    /// rescan. Type-scoped lookup (S40 red-team A5). The sidecar on disk is the single source
    /// of truth — a REPLACE re-import deletes it, which is the intended "record lost → 补做
    /// button" path for the vocal_range record.
    pub fn set_config_extra_key(
        &self,
        name: &str,
        model_type: &ModelType,
        key: &str,
        value: serde_json::Value,
    ) -> Result<()> {
        self.ensure_scanned();
        let mut entries = self.entries.write();
        let entry = entries
            .iter_mut()
            .find(|e| e.name == name && &e.model_type == model_type)
            .ok_or_else(|| crate::UtaiError::Model(format!("Model '{}' not found", name)))?;
        let json_path = entry.path.with_extension("json");
        let mut map: serde_json::Map<String, serde_json::Value> = std::fs::read_to_string(&json_path)
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default();
        map.insert(key.to_string(), value.clone());
        std::fs::write(
            &json_path,
            serde_json::to_string_pretty(&serde_json::Value::Object(map))
                .map_err(|e| crate::UtaiError::Model(format!("sidecar serialize: {e}")))?,
        )?;
        if let serde_json::Value::Object(extra) = &mut entry.config.extra {
            extra.insert(key.to_string(), value);
        } else {
            entry.config.extra = serde_json::json!({ key: value });
        }
        tracing::info!("Sidecar '{}' updated: {} written", json_path.display(), key);
        Ok(())
    }

    pub fn set_avatar(&self, name: &str, avatar_file: &Path) -> Result<Option<PathBuf>> {
        self.ensure_scanned();
        let mut entries = self.entries.write();
        let entry = entries
            .iter_mut()
            .find(|e| e.name == name)
            .ok_or_else(|| crate::UtaiError::Model(format!("Model '{}' not found", name)))?;
        let dir = entry
            .path
            .parent()
            .ok_or_else(|| crate::UtaiError::Model("Model path has no parent dir".to_string()))?
            .to_path_buf();
        let stem = entry
            .path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let dest = copy_avatar(&dir, &stem, avatar_file)?;
        entry.avatar_path = Some(dest.clone());
        tracing::info!("Set avatar for {}: {}", name, dest.display());
        Ok(Some(dest))
    }

    /// RVC retrieval index: copy a .npy or convert a faiss .index next to the model, keyed on
    /// `stem`. Any failure is pushed into `warnings` (surfaced by the import command) — a broken
    /// index must not look like a successful import that silently lost retrieval.
    fn resolve_index(
        &self,
        stem: &str,
        model_type: &ModelType,
        src_path: &Path,
        index_file: Option<&Path>,
        app_dir: &Path,
        warnings: &mut Vec<String>,
    ) -> Option<PathBuf> {
        if matches!(model_type, ModelType::SoVits) {
            return self.resolve_cluster_assets(stem, index_file, app_dir, warnings);
        }
        if !matches!(model_type, ModelType::Rvc) {
            return None;
        }

        let npy_dest = self
            .models_dir
            .join(type_subdir(model_type))
            .join(format!("{}.npy", stem));

        if let Some(idx_path) = index_file {
            if !idx_path.exists() {
                // The user explicitly picked this file — don't silently auto-detect another one.
                warnings.push(format!("索引文件不存在：{}", idx_path.display()));
                return None;
            }
            let ext = idx_path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if ext.eq_ignore_ascii_case("npy") {
                return copy_npy_index(idx_path, &npy_dest, warnings);
            }
            if ext.eq_ignore_ascii_case("index") {
                return try_index_conversion(idx_path, &npy_dest, app_dir, warnings);
            }
            warnings.push(format!("不支持的索引文件类型：{}", idx_path.display()));
            return None;
        }

        // Auto-detect next to the source model file (named after the SOURCE file's stem).
        let auto_npy = src_path.with_extension("npy");
        if auto_npy.exists() {
            return copy_npy_index(&auto_npy, &npy_dest, warnings);
        }

        let auto_index = src_path.with_extension("index");
        if auto_index.exists() {
            return try_index_conversion(&auto_index, &npy_dest, app_dir, warnings);
        }

        // Also check for added_*.index (RVC trainer naming) or an index carrying the source stem.
        let src_stem = src_path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        if let Some(parent) = src_path.parent() {
            if let Ok(entries) = std::fs::read_dir(parent) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.extension().and_then(|e| e.to_str()) == Some("index") {
                        let fname = p.file_name().unwrap_or_default().to_string_lossy().to_string();
                        if fname.starts_with("added_")
                            || (!src_stem.is_empty() && fname.contains(&src_stem))
                        {
                            return try_index_conversion(&p, &npy_dest, app_dir, warnings);
                        }
                    }
                }
            }
        }

        None
    }

    /// SoVITS companion assets (聚类/检索): a user-picked kmeans .pt / retrieval .pkl is
    /// converted (converter/export_cluster.py) into per-speaker .npy files under
    /// `<subdir>/<stem>.cluster/` — the directory the inference pipeline probes first.
    /// A pre-converted .npy is copied there verbatim (its filename must already follow the
    /// `<speaker_id>.index_vectors.npy` / `<speaker_name>.centers.npy` convention).
    /// Returns the first asset .npy so ModelEntry.index_path lights the UI 聚类 badge.
    fn resolve_cluster_assets(
        &self,
        stem: &str,
        index_file: Option<&Path>,
        app_dir: &Path,
        warnings: &mut Vec<String>,
    ) -> Option<PathBuf> {
        let src = index_file?;
        if !src.exists() {
            warnings.push(format!("聚类/检索模型文件不存在：{}", src.display()));
            return None;
        }
        let cluster_dir = self
            .models_dir
            .join(type_subdir(&ModelType::SoVits))
            .join(format!("{}.cluster", stem));
        let ext = src
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        match ext.as_str() {
            "npy" => {
                if let Err(e) = std::fs::create_dir_all(&cluster_dir) {
                    warnings.push(format!("聚类资产目录创建失败：{}", e));
                    return None;
                }
                let dest = cluster_dir.join(src.file_name()?);
                if let Err(e) = std::fs::copy(src, &dest) {
                    warnings.push(format!("聚类资产复制失败：{}", e));
                    return None;
                }
                // ①c: a multi-speaker RETRIEVAL model has ONE `<id>.index_vectors.npy` PER speaker
                // in the same source dir (exp_dir/cluster/), but the import flow only conveys ONE
                // of them. When the conveyed file is such a retrieval index, install EVERY sibling
                // so per-speaker (dominant-of-blend) inference can find each `<id>.index_vectors.npy`.
                // Single-speaker: only `0.index_vectors.npy` exists → this copies exactly that one
                // (byte-identical to pre-①c). A per-speaker copy failure is warned, not fatal
                // (partial retrieval degrades gracefully — the pipeline skips a missing index).
                let src_name = src.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if src_name.ends_with(".index_vectors.npy") {
                    if let Some(src_dir) = src.parent() {
                        if let Ok(read_dir) = std::fs::read_dir(src_dir) {
                            for sib in read_dir.flatten() {
                                let sp = sib.path();
                                if sp == src {
                                    continue; // already copied above
                                }
                                let matches = sp
                                    .file_name()
                                    .and_then(|n| n.to_str())
                                    .map(|n| n.ends_with(".index_vectors.npy"))
                                    .unwrap_or(false);
                                if matches {
                                    if let Some(fname) = sp.file_name() {
                                        if let Err(e) = std::fs::copy(&sp, cluster_dir.join(fname)) {
                                            warnings.push(format!(
                                                "多说话人检索资产 {} 复制失败：{}",
                                                sp.display(),
                                                e
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Some(dest)
            }
            "pt" | "pkl" | "pickle" => {
                match convert::convert_cluster_assets(src, &cluster_dir, app_dir) {
                    Ok(()) => first_cluster_asset(&cluster_dir).or_else(|| {
                        warnings.push(
                            "聚类模型转换完成但未生成任何 .npy——模型已导入，聚类/检索不可用"
                                .to_string(),
                        );
                        None
                    }),
                    Err(e) => {
                        tracing::warn!("Cluster conversion failed for {}: {}", src.display(), e);
                        warnings.push(format!(
                            "聚类/检索模型转换失败（{}）：{}——模型已导入，聚类增强不可用",
                            src.file_name().unwrap_or_default().to_string_lossy(),
                            e
                        ));
                        None
                    }
                }
            }
            _ => {
                warnings.push(format!(
                    "不支持的聚类/检索模型类型：{}（支持 .pt / .pkl / .npy）",
                    src.display()
                ));
                None
            }
        }
    }

    /// Delete by name, TYPE-SCOPED when the caller knows it (设计红队 A5: the
    /// scan order is rvc→sovits→…→nsf_hifigan, so an untyped delete of a
    /// vocoder named after its singer would remove the SINGER MODEL's files
    /// instead). `None` keeps the legacy first-match behavior for callers
    /// that genuinely have no type context.
    pub fn delete(&self, name: &str, model_type: Option<&ModelType>) -> Result<()> {
        self.ensure_scanned();
        let removed = {
            let mut entries = self.entries.write();
            entries
                .iter()
                .position(|e| {
                    e.name == name
                        && model_type.map_or(true, |t| {
                            std::mem::discriminant(&e.model_type)
                                == std::mem::discriminant(t)
                        })
                })
                .map(|idx| entries.remove(idx))
        };
        if let Some(entry) = removed {
            remove_entry_files(&entry);
            tracing::info!("Deleted model: {}", name);
        }
        Ok(())
    }
}

/// S40 vocoder source validation (设计红队 A6 + 审查 HIGH): parse the SOURCE
/// sidecar json, reject mini_nsf / missing recipe fields, and locate the mel
/// filterbank npy — WITHOUT touching the destination. Runs as the import
/// pre-flight (before the destructive REPLACE) and again inside the copy step.
/// Returns (sidecar_root, source_npy_path).
fn read_vocoder_source(src_path: &Path) -> Result<(serde_json::Value, PathBuf)> {
    let src_json = src_path.with_extension("json");
    if !src_json.is_file() {
        return Err(crate::UtaiError::Model(
            "声码器 .onnx 导入需要配套的 .json 配置文件（与 onnx 同名同目录）——缺少梅尔频谱参数的声码器无法使用"
                .to_string(),
        ));
    }
    let text = std::fs::read_to_string(&src_json)?;
    let root: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| crate::UtaiError::Model(format!("声码器配置 .json 解析失败：{}", e)))?;
    if root["mini_nsf"].as_bool().unwrap_or(false) {
        return Err(crate::UtaiError::Model(
            "暂不支持 PC-NSF（mini_nsf）架构的声码器——目前仅支持经典 NSF-HiFiGAN".to_string(),
        ));
    }
    for key in ["sample_rate", "hop_size", "num_mels", "n_fft", "win_size", "fmin", "fmax"] {
        if root.get(key).map(|v| v.is_null()).unwrap_or(true) {
            return Err(crate::UtaiError::Model(format!(
                "声码器配置缺少字段「{}」——无法确认梅尔频谱格式，拒绝导入",
                key
            )));
        }
    }
    // the sidecar's mel_filters name resolves against the SOURCE dir (that is
    // the field's contract), with the exporter's default naming as fallback
    let src_dir = src_path.parent().unwrap_or_else(|| Path::new("."));
    let src_stem = src_path.file_stem().unwrap_or_default().to_string_lossy();
    let mel_name = root["mel_filters"]
        .as_str()
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("{}_mel.npy", src_stem));
    let src_npy = src_dir.join(&mel_name);
    if !src_npy.is_file() {
        return Err(crate::UtaiError::Model(format!(
            "找不到声码器滤波器文件 {}（应与 onnx/json 同目录）——三件套缺一不可",
            src_npy.display()
        )));
    }
    Ok((root, src_npy))
}

/// S40 vocoder direct-.onnx import (设计红队 A6): carry the mel filterbank npy
/// over under OUR stem and rewrite the destination sidecar's `mel_filters`
/// field to match. Validation lives in read_vocoder_source (also the pre-flight).
fn import_vocoder_filterbank(src_path: &Path, onnx_path: &Path, stem: &str) -> Result<()> {
    let (mut root, src_npy) = read_vocoder_source(src_path)?;
    let dest_name = format!("{}_mel.npy", stem);
    let dest_npy = onnx_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(&dest_name);
    std::fs::copy(&src_npy, &dest_npy)?;
    root["mel_filters"] = serde_json::Value::from(dest_name);
    std::fs::write(
        onnx_path.with_extension("json"),
        serde_json::to_string_pretty(&root)?,
    )?;
    Ok(())
}

/// Remove a half-materialized import's files (审查 HIGH): a partial trio left
/// on disk would resurrect on the next scan() as a selectable ghost entry —
/// for vocoders that would even bypass the exporter's ORT self-check verdict.
fn sweep_partial_import(subdir: &Path, stem: &str) {
    for name in [
        format!("{}.onnx", stem),
        format!("{}.json", stem),
        format!("{}_mel.npy", stem),
    ] {
        std::fs::remove_file(subdir.join(name)).ok();
    }
}

/// Load (or synthesize) the sidecar json next to `onnx_path`, fill required keys (`type`,
/// `sample_rate`) with defaults when missing, inject the display `name`, and write it back.
/// Returns the parsed tolerant config. Missing/unreadable sidecars produce a user-visible warning.
fn finalize_sidecar(
    onnx_path: &Path,
    display_name: &str,
    model_type: &ModelType,
    warnings: &mut Vec<String>,
) -> Result<ModelConfig> {
    let json_path = onnx_path.with_extension("json");
    let mut root: serde_json::Map<String, serde_json::Value> = if json_path.exists() {
        match std::fs::read_to_string(&json_path)
            .ok()
            .and_then(|c| serde_json::from_str::<serde_json::Value>(&c).ok())
        {
            Some(serde_json::Value::Object(map)) => map,
            _ => {
                warnings.push("模型配置 .json 无法解析，已按默认参数重新生成".to_string());
                serde_json::Map::new()
            }
        }
    } else {
        warnings.push(format!(
            "未找到配套的 .json 配置，已按默认参数生成（{} / {}Hz）；若与模型实际参数不符，请附带配置文件重新导入",
            type_subdir(model_type).to_uppercase(),
            default_sample_rate_for(model_type)
        ));
        serde_json::Map::new()
    };

    if !root.contains_key("type") {
        root.insert("type".into(), serde_json::Value::from(type_subdir(model_type)));
    }
    if !root.contains_key("sample_rate") {
        root.insert(
            "sample_rate".into(),
            serde_json::Value::from(default_sample_rate_for(model_type)),
        );
    }
    // Display name INTO the sidecar — the disk rescan reads it back (file stems are sanitized).
    root.insert("name".into(), serde_json::Value::from(display_name));

    // 审查修复 S41-INT-1: auto_f0.file names the SOURCE stem's companion (an
    // audition conversion says "model.f0.onnx") — after a stem-renaming copy
    // the runtime lookup would miss the file or hit a foreign one. Normalize
    // from the actual on-disk companion next to the destination onnx; a
    // claimed-but-absent companion flips available=false (the run would
    // otherwise hard-error at render time).
    let dest_f0 = onnx_path.with_extension("f0.onnx");
    if let Some(auto) = root.get_mut("auto_f0").and_then(|v| v.as_object_mut()) {
        let claims = auto.get("available").and_then(|v| v.as_bool()).unwrap_or(false);
        if claims {
            if dest_f0.is_file() {
                let fname = dest_f0
                    .file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default();
                auto.insert("file".into(), serde_json::Value::from(fname));
            } else {
                auto.insert("available".into(), serde_json::Value::from(false));
                warnings.push(
                    "自动音高预测器文件缺失——该模型的自动f0已禁用（重新导入 .pth 可恢复）"
                        .to_string(),
                );
            }
        }
    }

    std::fs::write(
        &json_path,
        serde_json::to_string_pretty(&serde_json::Value::Object(root.clone()))?,
    )?;

    Ok(
        serde_json::from_value(serde_json::Value::Object(root)).unwrap_or_else(|e| {
            tracing::warn!("Sidecar config parse fell back to defaults: {}", e);
            let mut cfg = default_config();
            cfg.name = Some(display_name.to_string());
            cfg
        }),
    )
}

/// Copy a single-level asset dir (the `.diffusion/` attachment — flat files only).
fn copy_dir_flat(src: &Path, dest: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dest)?;
    for entry in std::fs::read_dir(src)?.flatten() {
        let p = entry.path();
        if p.is_file() {
            std::fs::copy(&p, dest.join(entry.file_name()))?;
        }
    }
    Ok(())
}

/// First per-speaker .npy inside a `<stem>.cluster/` dir (alphabetical for stability), or None.
fn first_cluster_asset(cluster_dir: &Path) -> Option<PathBuf> {
    let mut assets: Vec<PathBuf> = std::fs::read_dir(cluster_dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("npy"))
        .collect();
    assets.sort();
    assets.into_iter().next()
}

fn copy_npy_index(src: &Path, npy_dest: &Path, warnings: &mut Vec<String>) -> Option<PathBuf> {
    match std::fs::copy(src, npy_dest) {
        Ok(_) => Some(npy_dest.to_path_buf()),
        Err(e) => {
            tracing::warn!("Failed to copy .npy index {}: {}", src.display(), e);
            warnings.push(format!("索引文件复制失败：{}", e));
            None
        }
    }
}

fn try_index_conversion(
    index_path: &Path,
    npy_dest: &Path,
    app_dir: &Path,
    warnings: &mut Vec<String>,
) -> Option<PathBuf> {
    match convert::convert_index_to_npy(index_path, npy_dest, app_dir) {
        Ok(()) => Some(npy_dest.to_path_buf()),
        Err(e) => {
            tracing::warn!("Index conversion failed for {}: {}", index_path.display(), e);
            warnings.push(format!(
                "索引文件转换失败（{}）：{}——模型已导入，但检索增强不可用",
                index_path.file_name().unwrap_or_default().to_string_lossy(),
                e
            ));
            None
        }
    }
}

/// Remove leftover `<stem>.diffusion.tmp*` / `<stem>.diffusion.old*` dirs
/// older than an hour (crashed/failed attach attempts). Fresh ones are kept —
/// they may belong to a conversion still in flight.
fn sweep_stale_attachment_dirs(subdir: &Path, stem: &str) {
    let tmp_prefix = format!("{}.diffusion.tmp", stem);
    let old_prefix = format!("{}.diffusion.old", stem);
    let Ok(rd) = std::fs::read_dir(subdir) else { return };
    for e in rd.filter_map(|e| e.ok()) {
        let n = e.file_name().to_string_lossy().into_owned();
        if !(n.starts_with(&tmp_prefix) || n.starts_with(&old_prefix)) {
            continue;
        }
        let stale = e
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.elapsed().ok())
            .map(|age| age.as_secs() > 3600)
            .unwrap_or(true);
        if stale {
            let _ = std::fs::remove_dir_all(e.path());
        }
    }
}

/// `encoder_out_channels` from a `.diffusion/diffusion.json` sidecar — the
/// dim-compat cross-check input, shared by the import-time attachment
/// (resolve_diffusion_assets) and the S39 trained-checkpoint attach flow.
fn diffusion_sidecar_dim(diffusion_dir: &Path) -> Option<u64> {
    std::fs::read_to_string(diffusion_dir.join("diffusion.json"))
        .ok()
        .and_then(|c| serde_json::from_str::<serde_json::Value>(&c).ok())
        .and_then(|v| v.get("encoder_out_channels").and_then(|d| d.as_u64()))
}

/// The directory + file stem a model's companion assets key off
/// (`<dir>/<stem>.diffusion` etc.).
fn entry_dir_and_stem(entry: &ModelEntry) -> Result<(PathBuf, String)> {
    let dir = entry
        .path
        .parent()
        .ok_or_else(|| crate::UtaiError::Model("Model path has no parent dir".to_string()))?
        .to_path_buf();
    let stem = entry
        .path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    Ok((dir, stem))
}

/// Copy `avatar_src` to `<dir>/<stem>.avatar.<ext>`, clearing any previous `<stem>.avatar.*`
/// first — a re-set with a different image extension must not leave the old file behind to
/// shadow the new one on rescan.
fn copy_avatar(dir: &Path, stem: &str, avatar_src: &Path) -> Result<PathBuf> {
    let ext = avatar_src.extension().and_then(|e| e.to_str()).unwrap_or("png");
    remove_stem_avatars(dir, stem);
    let dest = dir.join(format!("{}.avatar.{}", stem, ext));
    std::fs::copy(avatar_src, &dest)
        .map_err(|e| crate::UtaiError::Model(format!("Avatar copy failed: {}", e)))?;
    Ok(dest)
}

fn remove_stem_avatars(dir: &Path, stem: &str) {
    for ext in AVATAR_EXTS {
        let p = dir.join(format!("{}.avatar.{}", stem, ext));
        if p.exists() {
            let _ = std::fs::remove_file(&p);
        }
    }
}

/// Remove every on-disk artifact keyed on the entry's stem (onnx + sidecar json + index npy +
/// f0-predictor onnx + diffusion dir + avatar). Shared by delete and the re-import REPLACE
/// path — one source of truth for "which files belong to a model".
fn remove_entry_files(entry: &ModelEntry) {
    std::fs::remove_file(&entry.path).ok();
    std::fs::remove_file(entry.path.with_extension("json")).ok();
    std::fs::remove_file(entry.path.with_extension("npy")).ok();
    // SoVITS auto-f0 predictor graph (converter writes `<stem>.f0.onnx`, S36).
    std::fs::remove_file(entry.path.with_extension("f0.onnx")).ok();
    // SoVITS cluster/retrieval asset dir (see resolve_cluster_assets) + the S36
    // shallow-diffusion attachment dir (see resolve_diffusion_assets) + the S40
    // vocoder mel filterbank (`<stem>_mel.npy` — underscore, NOT an extension:
    // `<stem>.npy` is the RVC index slot).
    if let (Some(dir), Some(stem)) = (entry.path.parent(), entry.path.file_stem()) {
        std::fs::remove_dir_all(dir.join(format!("{}.cluster", stem.to_string_lossy()))).ok();
        std::fs::remove_dir_all(dir.join(format!("{}.diffusion", stem.to_string_lossy()))).ok();
        std::fs::remove_file(dir.join(format!("{}_mel.npy", stem.to_string_lossy()))).ok();
    }
    if let Some(index) = &entry.index_path {
        std::fs::remove_file(index).ok();
    }
    if let Some(avatar) = &entry.avatar_path {
        std::fs::remove_file(avatar).ok();
    }
    if let (Some(dir), Some(stem)) = (entry.path.parent(), entry.path.file_stem()) {
        remove_stem_avatars(dir, &stem.to_string_lossy());
    }
}

/// Strip characters Windows can't have in filenames (CJK and other Unicode pass through
/// untouched), trim trailing dots/spaces, and dodge reserved device names. Empty → "model".
/// pub(crate): the S41 audition cache names its per-host wav with this (single source).
pub(crate) fn sanitize_file_stem(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .filter(|c| {
            !matches!(c, '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*') && !c.is_control()
        })
        .collect();
    let cleaned = cleaned
        .trim()
        .trim_end_matches(|c| c == '.' || c == ' ')
        .to_string();
    let upper = cleaned.to_ascii_uppercase();
    let reserved = matches!(upper.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || (upper.len() == 4
            && (upper.starts_with("COM") || upper.starts_with("LPT"))
            && upper.chars().nth(3).map(|c| c.is_ascii_digit()).unwrap_or(false));
    if cleaned.is_empty() {
        "model".to_string()
    } else if reserved {
        format!("{}_model", cleaned)
    } else {
        cleaned
    }
}

/// First stem (base, then "base (2)", "base (3)", …) whose artifact slots are free in `dir`.
/// A same-name re-import never reaches the suffix path: the REPLACE step already removed the
/// old files, so the base stem is free again and filenames stay stable across re-imports.
fn unique_stem(dir: &Path, base: &str) -> String {
    let taken = |stem: &str| {
        dir.join(format!("{}.onnx", stem)).exists() || dir.join(format!("{}.json", stem)).exists()
    };
    if !taken(base) {
        return base.to_string();
    }
    let mut i = 2u32;
    loop {
        let cand = format!("{} ({})", base, i);
        if !taken(&cand) {
            return cand;
        }
        i += 1;
    }
}

fn find_avatar(onnx_path: &Path) -> Option<PathBuf> {
    for ext in AVATAR_EXTS {
        let p = onnx_path.with_extension(format!("avatar.{}", ext));
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn default_config() -> ModelConfig {
    ModelConfig {
        name: None,
        r#type: String::new(),
        version: default_version(),
        sample_rate: default_sample_rate(),
        features_dim: default_features_dim(),
        n_speakers: 0,
        speakers: default_speakers(),
        speech_encoder: None,
        hop_size: None,
        vol_embedding: None,
        unit_interpolate_mode: None,
        noise: None,
        inputs: None,
        extra: serde_json::Value::Object(serde_json::Map::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_models_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("utai_registry_test_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// #1/#2/#5/#9: CJK display name keys every artifact on ONE sanitized stem; the sidecar json
    /// carries the display name so a FRESH registry (lazy self-scan) reconstructs the entry
    /// losslessly; same-name re-import replaces instead of duplicating; sanitize-collisions get
    /// a numbered suffix. Uses the direct-.onnx import path — no python / no ORT.
    #[test]
    fn import_rename_keying_and_rescan_roundtrip() {
        let models_dir = temp_models_dir();
        let app_dir = models_dir.clone(); // converter is never spawned on this path

        let src_dir = models_dir.join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        let src_onnx = src_dir.join("orig_export.onnx");
        std::fs::write(&src_onnx, b"fake onnx bytes").unwrap();
        let src_index = src_dir.join("picked.npy");
        std::fs::write(&src_index, b"fake npy").unwrap();
        let src_avatar = src_dir.join("face.png");
        std::fs::write(&src_avatar, b"fake png").unwrap();

        let display_name = "紫音・テスト v2.3";
        let reg = ModelRegistry::new(models_dir.clone());
        let outcome = reg
            .import_file(display_name, &src_onnx, ModelType::Rvc, &app_dir,
                Some(&src_index), None, None, Some(&src_avatar), None)
            .unwrap();

        // Every artifact shares the sanitized stem (CJK preserved; nothing to strip here).
        let rvc_dir = models_dir.join("rvc");
        let stem = outcome.entry.path.file_stem().unwrap().to_string_lossy().to_string();
        assert_eq!(stem, display_name);
        assert_eq!(outcome.entry.path, rvc_dir.join(format!("{}.onnx", stem)));
        assert!(outcome.entry.path.exists());
        assert_eq!(outcome.entry.index_path.as_deref(), Some(rvc_dir.join(format!("{}.npy", stem)).as_path()));
        assert!(outcome.entry.index_path.as_ref().unwrap().exists());
        assert_eq!(outcome.entry.avatar_path.as_deref(), Some(rvc_dir.join(format!("{}.avatar.png", stem)).as_path()));
        assert!(outcome.entry.avatar_path.as_ref().unwrap().exists());
        // Source had no sidecar json → minimal default synthesized + surfaced as a warning (#5/#6).
        assert!(!outcome.warnings.is_empty());
        let sidecar: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(outcome.entry.path.with_extension("json")).unwrap(),
        )
        .unwrap();
        assert_eq!(sidecar["name"], display_name);
        assert_eq!(sidecar["type"], "rvc");

        // FRESH registry (fresh session): lazy self-scan on get() + name restored from the sidecar.
        let reg2 = ModelRegistry::new(models_dir.clone());
        let entry = reg2.get(display_name).expect("rescan must rebuild the entry by display name");
        assert_eq!(entry.path, outcome.entry.path);
        assert_eq!(entry.index_path, outcome.entry.index_path);
        assert_eq!(entry.avatar_path, outcome.entry.avatar_path);
        assert_eq!(entry.config.name.as_deref(), Some(display_name));

        // Same-name re-import → REPLACE (single row, stable stem), old index/avatar cleaned up.
        let outcome2 = reg2
            .import_file(display_name, &src_onnx, ModelType::Rvc, &app_dir, None, None, None, None, None)
            .unwrap();
        assert_eq!(reg2.list_by_type(&ModelType::Rvc).len(), 1);
        assert_eq!(outcome2.entry.path, outcome.entry.path);
        assert!(outcome2.entry.index_path.is_none());
        assert!(!rvc_dir.join(format!("{}.npy", stem)).exists());
        assert!(!rvc_dir.join(format!("{}.avatar.png", stem)).exists());

        // Two DIFFERENT display names sanitizing to the same stem → numbered suffix, no clobber.
        let a = reg2.import_file("A/B", &src_onnx, ModelType::Rvc, &app_dir, None, None, None, None, None).unwrap();
        let b = reg2.import_file("A\\B", &src_onnx, ModelType::Rvc, &app_dir, None, None, None, None, None).unwrap();
        assert_eq!(a.entry.path, rvc_dir.join("AB.onnx"));
        assert_eq!(b.entry.path, rvc_dir.join("AB (2).onnx"));
        assert!(a.entry.path.exists() && b.entry.path.exists());
        // Rescan keeps both display names distinct (from their sidecars).
        let reg3 = ModelRegistry::new(models_dir.clone());
        assert!(reg3.get("A/B").is_some());
        assert!(reg3.get("A\\B").is_some());

        std::fs::remove_dir_all(&models_dir).ok();
    }

    /// #11: speakers field tolerates the converter's map form, the legacy list form, and absence.
    #[test]
    fn model_config_speakers_tolerant() {
        let map_form: ModelConfig =
            serde_json::from_str(r#"{"speakers": {"akiko": 0, "beta": 1}, "hop_size": 512}"#).unwrap();
        assert_eq!(map_form.speakers.get("beta"), Some(&1));
        assert_eq!(map_form.hop_size, Some(512));

        let list_form: ModelConfig =
            serde_json::from_str(r#"{"speakers": ["first", "second"]}"#).unwrap();
        assert_eq!(list_form.speakers.get("second"), Some(&1));

        let absent: ModelConfig = serde_json::from_str(r#"{"custom_key": 7}"#).unwrap();
        assert_eq!(absent.speakers, default_speakers());
        // Unknown keys survive into `extra` (full config flows through list_models).
        assert_eq!(absent.extra.get("custom_key").and_then(|v| v.as_u64()), Some(7));
    }
}
