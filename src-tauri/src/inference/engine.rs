use std::borrow::Cow;
use std::collections::HashMap;
use std::path::PathBuf;

use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;
use parking_lot::{Mutex, RwLock};

use crate::{Result, UtaiError};

pub struct OnnxEngine {
    sessions: RwLock<HashMap<String, LoadedSession>>,
    device: RwLock<DeviceConfig>,
}

struct LoadedSession {
    session: Mutex<Session>,
    _path: PathBuf,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum DeviceConfig {
    Cpu,
    DirectMl { device_id: u32 },
    Cuda { device_id: u32 },
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self::Cpu
    }
}

pub enum InputTensor {
    F32 { data: Vec<f32>, shape: Vec<i64> },
    I64 { data: Vec<i64>, shape: Vec<i64> },
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

        let device = self.device.read().clone();
        let session = build_session(path, &device)?;

        let id = uuid::Uuid::new_v4().to_string();
        let loaded = LoadedSession {
            session: Mutex::new(session),
            _path: path.clone(),
        };
        self.sessions.write().insert(id.clone(), loaded);

        tracing::info!("Loaded ONNX model: {} -> {}", path.display(), id);
        Ok(id)
    }

    pub fn unload_model(&self, id: &str) {
        self.sessions.write().remove(id);
        tracing::info!("Unloaded ONNX model: {}", id);
    }

    pub fn is_loaded(&self, id: &str) -> bool {
        self.sessions.read().contains_key(id)
    }

    pub fn run(
        &self,
        session_id: &str,
        inputs: Vec<(&str, InputTensor)>,
    ) -> Result<Vec<Vec<f32>>> {
        let sessions = self.sessions.read();
        let loaded = sessions.get(session_id).ok_or_else(|| {
            UtaiError::Inference(format!("Session '{}' not found", session_id))
        })?;

        let mut session = loaded.session.lock();

        let mut input_values: Vec<(Cow<str>, SessionInputValue)> = Vec::new();

        for (name, tensor) in inputs {
            let value: SessionInputValue = match tensor {
                InputTensor::F32 { data, shape } => {
                    Tensor::from_array((shape, data.into_boxed_slice()))
                        .map_err(|e| UtaiError::Inference(format!("Input '{}': {}", name, e)))?
                        .into()
                }
                InputTensor::I64 { data, shape } => {
                    Tensor::from_array((shape, data.into_boxed_slice()))
                        .map_err(|e| UtaiError::Inference(format!("Input '{}': {}", name, e)))?
                        .into()
                }
            };
            input_values.push((Cow::Borrowed(name), value));
        }

        let outputs = session.run(input_values)
            .map_err(|e| UtaiError::Inference(format!("Inference failed: {}", e)))?;

        let mut result = Vec::new();
        for i in 0..outputs.len() {
            let (_, data) = outputs[i]
                .try_extract_tensor::<f32>()
                .map_err(|e| UtaiError::Inference(format!("Output {}: {}", i, e)))?;
            result.push(data.to_vec());
        }

        Ok(result)
    }
}

fn build_session(path: &PathBuf, device: &DeviceConfig) -> Result<Session> {
    let mut builder = Session::builder()
        .map_err(|e| UtaiError::Inference(format!("Session builder: {}", e)))?;

    match device {
        DeviceConfig::Cpu => {
            builder = builder
                .with_execution_providers([ort::ep::CPU::default().build()])
                .map_err(|e| UtaiError::Inference(format!("CPU EP: {}", e)))?;
        }
        DeviceConfig::DirectMl { device_id } => {
            builder = builder
                .with_execution_providers([
                    ort::ep::DirectML::default()
                        .with_device_id(*device_id as i32)
                        .build(),
                    ort::ep::CPU::default().build(),
                ])
                .map_err(|e| UtaiError::Inference(format!("DirectML EP: {}", e)))?;
        }
        DeviceConfig::Cuda { device_id } => {
            builder = builder
                .with_execution_providers([
                    ort::ep::CUDA::default()
                        .with_device_id(*device_id as i32)
                        .build(),
                    ort::ep::CPU::default().build(),
                ])
                .map_err(|e| UtaiError::Inference(format!("CUDA EP: {}", e)))?;
        }
    }

    builder
        .commit_from_file(path)
        .map_err(|e| UtaiError::Inference(format!("Load model '{}': {}", path.display(), e)))
}
