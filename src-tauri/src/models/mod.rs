pub mod convert;

use std::path::PathBuf;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::Result;

pub struct ModelRegistry {
    models_dir: PathBuf,
    entries: RwLock<Vec<ModelEntry>>,
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
    pub avatar_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    #[serde(default)]
    pub r#type: String,
    #[serde(default = "default_version")]
    pub version: String,
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    #[serde(default = "default_features_dim")]
    pub features_dim: u32,
    #[serde(default)]
    pub n_speakers: u32,
    #[serde(default = "default_speakers")]
    pub speakers: Vec<String>,
    #[serde(flatten)]
    pub extra: serde_json::Value,
}

fn default_version() -> String { "unknown".to_string() }
fn default_sample_rate() -> u32 { 40000 }
fn default_features_dim() -> u32 { 768 }
fn default_speakers() -> Vec<String> { vec!["default".to_string()] }

impl ModelRegistry {
    pub fn new(models_dir: PathBuf) -> Self {
        Self {
            models_dir,
            entries: RwLock::new(Vec::new()),
        }
    }

    pub fn scan(&self) -> Result<()> {
        let mut entries = self.entries.write();
        let prev_count = entries.len();
        entries.clear();

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
                        let name = path
                            .file_stem()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .to_string();

                        let config_path = path.with_extension("json");
                        let config = if config_path.exists() {
                            let content = std::fs::read_to_string(&config_path).ok();
                            content
                                .and_then(|c| serde_json::from_str(&c).ok())
                                .unwrap_or_else(default_config)
                        } else {
                            default_config()
                        };

                        let sample_rate = config.sample_rate;

                        let index_path = path.with_extension("npy");
                        let index = if index_path.exists() {
                            Some(index_path)
                        } else {
                            None
                        };

                        let avatar = find_avatar(&path, &name);

                        entries.push(ModelEntry {
                            name,
                            model_type: model_type.clone(),
                            format: ModelFormat::Onnx,
                            path,
                            sample_rate,
                            config,
                            index_path: index,
                            avatar_path: avatar,
                        });
                    }
                }
            }
        }

        if entries.len() != prev_count {
            tracing::info!("Model scan: {} models found (was {})", entries.len(), prev_count);
        }
        Ok(())
    }

    pub fn list(&self) -> Vec<ModelEntry> {
        self.entries.read().clone()
    }

    pub fn list_by_type(&self, model_type: &ModelType) -> Vec<ModelEntry> {
        self.entries
            .read()
            .iter()
            .filter(|e| std::mem::discriminant(&e.model_type) == std::mem::discriminant(model_type))
            .cloned()
            .collect()
    }

    pub fn get(&self, name: &str) -> Option<ModelEntry> {
        self.entries.read().iter().find(|e| e.name == name).cloned()
    }

    pub fn exists(&self, name: &str, model_type: &ModelType) -> bool {
        self.entries.read().iter().any(|e| {
            e.name == name
                && std::mem::discriminant(&e.model_type) == std::mem::discriminant(model_type)
        })
    }

    pub fn import_pth(
        &self,
        name: &str,
        pth_path: &PathBuf,
        model_type: ModelType,
        app_dir: &PathBuf,
        index_file: Option<&PathBuf>,
        avatar_file: Option<&PathBuf>,
    ) -> Result<ModelEntry> {
        let onnx_path = convert::convert_pth_to_onnx(pth_path, &self.models_dir, &model_type, app_dir)?;

        let config_path = onnx_path.with_extension("json");
        let config = if config_path.exists() {
            std::fs::read_to_string(&config_path)
                .ok()
                .and_then(|c| serde_json::from_str(&c).ok())
                .unwrap_or_else(default_config)
        } else {
            default_config()
        };
        let sample_rate = config.sample_rate;

        let index_path = self.resolve_index(name, &model_type, pth_path, index_file, app_dir);
        let avatar_path = self.import_avatar(name, &model_type, avatar_file);

        let entry = ModelEntry {
            name: name.to_string(),
            model_type,
            format: ModelFormat::Onnx,
            path: onnx_path,
            sample_rate,
            config,
            index_path,
            avatar_path,
        };

        self.entries.write().push(entry.clone());
        Ok(entry)
    }

    fn import_avatar(
        &self,
        name: &str,
        model_type: &ModelType,
        avatar_file: Option<&PathBuf>,
    ) -> Option<PathBuf> {
        let avatar_src = avatar_file.filter(|p| p.exists())?;
        let ext = avatar_src.extension().and_then(|e| e.to_str()).unwrap_or("png");
        let subdir = match model_type {
            ModelType::Rvc => "rvc",
            ModelType::SoVits => "sovits",
            _ => return None,
        };
        let dest = self.models_dir.join(subdir).join(format!("{}.avatar.{}", name, ext));
        match std::fs::copy(avatar_src, &dest) {
            Ok(_) => {
                tracing::info!("Imported avatar for {}: {}", name, dest.display());
                Some(dest)
            }
            Err(e) => {
                tracing::warn!("Failed to copy avatar: {}", e);
                None
            }
        }
    }

    pub fn set_avatar(&self, name: &str, avatar_file: &PathBuf) -> Result<Option<PathBuf>> {
        let mut entries = self.entries.write();
        let entry = entries.iter_mut().find(|e| e.name == name)
            .ok_or_else(|| crate::UtaiError::Model(format!("Model '{}' not found", name)))?;
        let path = self.import_avatar(name, &entry.model_type, Some(avatar_file));
        entry.avatar_path = path.clone();
        Ok(path)
    }

    fn resolve_index(
        &self,
        name: &str,
        model_type: &ModelType,
        pth_path: &PathBuf,
        index_file: Option<&PathBuf>,
        app_dir: &PathBuf,
    ) -> Option<PathBuf> {
        if !matches!(model_type, ModelType::Rvc) {
            return None;
        }

        let subdir = self.models_dir.join("rvc");
        let npy_dest = subdir.join(format!("{}.npy", name));

        if let Some(idx_path) = index_file {
            if idx_path.exists() {
                let ext = idx_path.extension().and_then(|e| e.to_str()).unwrap_or("");
                if ext == "npy" {
                    if let Err(e) = std::fs::copy(idx_path, &npy_dest) {
                        tracing::warn!("Failed to copy .npy index: {}", e);
                        return None;
                    }
                    return Some(npy_dest);
                }
                if ext == "index" {
                    match convert::convert_index_to_npy(idx_path, &npy_dest, app_dir) {
                        Ok(p) => return Some(p),
                        Err(e) => {
                            tracing::warn!("Index conversion failed: {}", e);
                            return None;
                        }
                    }
                }
            }
        }

        // Auto-detect: check for .index or .npy next to the .pth
        let auto_npy = pth_path.with_extension("npy");
        if auto_npy.exists() {
            if let Err(e) = std::fs::copy(&auto_npy, &npy_dest) {
                tracing::warn!("Failed to copy auto-detected .npy: {}", e);
                return None;
            }
            return Some(npy_dest);
        }

        let auto_index = pth_path.with_extension("index");
        if auto_index.exists() {
            match convert::convert_index_to_npy(&auto_index, &npy_dest, app_dir) {
                Ok(p) => return Some(p),
                Err(e) => tracing::warn!("Auto index conversion failed: {}", e),
            }
        }

        // Also check for added_*.index in the same directory
        if let Some(parent) = pth_path.parent() {
            if let Ok(entries) = std::fs::read_dir(parent) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.extension().and_then(|e| e.to_str()) == Some("index") {
                        let fname = p.file_name().unwrap_or_default().to_string_lossy();
                        if fname.starts_with("added_") || fname.contains(name) {
                            match convert::convert_index_to_npy(&p, &npy_dest, app_dir) {
                                Ok(p) => return Some(p),
                                Err(e) => tracing::warn!("Index conversion failed for {}: {}", fname, e),
                            }
                        }
                    }
                }
            }
        }

        None
    }

    pub fn delete(&self, name: &str) -> Result<()> {
        let mut entries = self.entries.write();
        if let Some(idx) = entries.iter().position(|e| e.name == name) {
            let entry = entries.remove(idx);
            std::fs::remove_file(&entry.path).ok();
            if let Some(index) = &entry.index_path {
                std::fs::remove_file(index).ok();
            }
            if let Some(avatar) = &entry.avatar_path {
                std::fs::remove_file(avatar).ok();
            }
            let config_path = entry.path.with_extension("json");
            std::fs::remove_file(config_path).ok();
            tracing::info!("Deleted model: {}", name);
        }
        Ok(())
    }
}

fn find_avatar(onnx_path: &std::path::Path, name: &str) -> Option<PathBuf> {
    for ext in &["png", "jpg", "jpeg", "bmp", "webp"] {
        let p = onnx_path.with_extension(format!("avatar.{}", ext));
        if p.exists() { return Some(p); }
        if let Some(dir) = onnx_path.parent() {
            let p2 = dir.join(format!("{}.avatar.{}", name, ext));
            if p2.exists() { return Some(p2); }
        }
    }
    None
}

fn default_config() -> ModelConfig {
    ModelConfig {
        r#type: String::new(),
        version: default_version(),
        sample_rate: default_sample_rate(),
        features_dim: default_features_dim(),
        n_speakers: 0,
        speakers: default_speakers(),
        extra: serde_json::Value::Object(serde_json::Map::new()),
    }
}
