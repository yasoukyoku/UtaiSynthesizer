use std::collections::HashMap;
use std::path::PathBuf;

use parking_lot::RwLock;

use crate::{Result, UtaiError};

pub struct OnnxEngine {
    sessions: RwLock<HashMap<String, LoadedSession>>,
    device: RwLock<DeviceConfig>,
}

struct LoadedSession {
    path: PathBuf,
    // The actual ort::Session will be stored here once models are available
    // For now we track paths to validate the architecture compiles
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum DeviceConfig {
    Cpu,
    DirectMl { device_id: u32 },
    Cuda { device_id: u32 },
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self::DirectMl { device_id: 0 }
    }
}

impl OnnxEngine {
    pub fn new() -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
            device: RwLock::new(DeviceConfig::default()),
        }
    }

    pub fn set_device(&self, config: DeviceConfig) {
        *self.device.write() = config;
    }

    pub fn load_model(&self, path: &PathBuf) -> Result<String> {
        if !path.exists() {
            return Err(UtaiError::Inference(format!(
                "Model file not found: {}",
                path.display()
            )));
        }

        let id = uuid::Uuid::new_v4().to_string();
        let session = LoadedSession { path: path.clone() };
        self.sessions.write().insert(id.clone(), session);

        tracing::info!("Registered ONNX model: {} -> {}", path.display(), id);
        Ok(id)
    }

    pub fn unload_model(&self, id: &str) {
        self.sessions.write().remove(id);
        tracing::info!("Unloaded ONNX model: {}", id);
    }

    pub fn is_loaded(&self, id: &str) -> bool {
        self.sessions.read().contains_key(id)
    }

    pub fn run_f32(
        &self,
        session_id: &str,
        inputs: &[(&str, &[f32], &[usize])],
    ) -> Result<Vec<Vec<f32>>> {
        if !self.is_loaded(session_id) {
            return Err(UtaiError::Inference(format!(
                "Session '{}' not found",
                session_id
            )));
        }

        // ONNX Runtime inference will be connected here once models are available.
        // The actual implementation will:
        // 1. Create ort::Value from input arrays
        // 2. Run session.run() with named inputs
        // 3. Extract output tensors as Vec<f32>
        //
        // For now, return empty outputs to validate the architecture.
        tracing::debug!(
            "ONNX inference requested on session {} with {} inputs",
            session_id,
            inputs.len()
        );

        Ok(vec![vec![]])
    }
}
