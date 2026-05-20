pub mod cache;
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
    pub fn new() -> Self {
        let models_dir = PathBuf::from("models");
        Self {
            models_dir,
            entries: RwLock::new(Vec::new()),
        }
    }

    pub fn set_models_dir(&self, dir: PathBuf) {
        // Re-scan would happen here
        let _ = dir;
    }

    pub fn scan(&self) -> Result<()> {
        let mut entries = self.entries.write();
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

                        entries.push(ModelEntry {
                            name,
                            model_type: model_type.clone(),
                            format: ModelFormat::Onnx,
                            path,
                            sample_rate,
                            config,
                            index_path: index,
                        });
                    }
                }
            }
        }

        tracing::info!("Model scan complete: {} models found", entries.len());
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
    ) -> Result<ModelEntry> {
        let onnx_path = convert::convert_pth_to_onnx(pth_path, &self.models_dir, &model_type)?;

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

        let index_path = pth_path.with_extension("npy");
        let index = if index_path.exists() { Some(index_path) } else { None };

        let entry = ModelEntry {
            name: name.to_string(),
            model_type,
            format: ModelFormat::Onnx,
            path: onnx_path,
            sample_rate,
            config,
            index_path: index,
        };

        self.entries.write().push(entry.clone());
        Ok(entry)
    }

    pub fn delete(&self, name: &str) -> Result<()> {
        let mut entries = self.entries.write();
        if let Some(idx) = entries.iter().position(|e| e.name == name) {
            let entry = entries.remove(idx);
            std::fs::remove_file(&entry.path).ok();
            if let Some(index) = &entry.index_path {
                std::fs::remove_file(index).ok();
            }
            let config_path = entry.path.with_extension("json");
            std::fs::remove_file(config_path).ok();
            tracing::info!("Deleted model: {}", name);
        }
        Ok(())
    }
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
