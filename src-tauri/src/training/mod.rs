pub mod augment;
pub mod sidecar;

use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::{Result, UtaiError};

pub struct TrainingManager {
    active_process: Mutex<Option<sidecar::TrainingSidecar>>,
    python_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub model_name: String,
    pub backend: TrainingBackend,
    pub dataset_path: String,
    pub epochs: u32,
    pub batch_size: u32,
    pub sample_rate: u32,
    pub save_interval: u32,
    pub augmentation: Option<augment::AugmentConfig>,
    pub continuation: ContinuationMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrainingBackend {
    Rvc { version: RvcVersion },
    SoVits { shallow_diffusion: bool },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RvcVersion {
    V1,
    V2,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContinuationMode {
    Fresh,
    Continue { from_epoch: u32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStatus {
    pub state: TrainingState,
    pub current_epoch: u32,
    pub total_epochs: u32,
    pub loss: Option<f64>,
    pub elapsed_secs: u64,
    pub eta_secs: Option<u64>,
    pub model_name: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TrainingState {
    Idle,
    Preparing,
    Preprocessing,
    Training,
    GeneratingIndex,
    Completed,
    Stopped,
    Error(String),
}

impl TrainingManager {
    pub fn new() -> Self {
        let python_path = Self::find_python();
        Self {
            active_process: Mutex::new(None),
            python_path,
        }
    }

    pub fn start(&self, config: TrainingConfig) -> Result<()> {
        let mut process = self.active_process.lock();
        if process.is_some() {
            return Err(UtaiError::Training(
                "Training already in progress".to_string(),
            ));
        }

        let sidecar = sidecar::TrainingSidecar::spawn(&self.python_path, &config)?;
        *process = Some(sidecar);
        tracing::info!("Training started: {}", config.model_name);
        Ok(())
    }

    pub fn stop(&self) -> Result<()> {
        let mut process = self.active_process.lock();
        if let Some(ref mut sidecar) = *process {
            sidecar.request_stop()?;
            // Index generation happens in the Python side on stop signal
            tracing::info!("Training stop requested (will generate index before exit)");
        }
        Ok(())
    }

    pub fn force_stop(&self) -> Result<()> {
        let mut process = self.active_process.lock();
        if let Some(sidecar) = process.take() {
            sidecar.force_kill()?;
            tracing::warn!("Training force-killed");
        }
        Ok(())
    }

    pub fn status(&self) -> TrainingStatus {
        let process = self.active_process.lock();
        match &*process {
            Some(sidecar) => sidecar.status(),
            None => TrainingStatus {
                state: TrainingState::Idle,
                current_epoch: 0,
                total_epochs: 0,
                loss: None,
                elapsed_secs: 0,
                eta_secs: None,
                model_name: String::new(),
            },
        }
    }

    pub fn is_active(&self) -> bool {
        let process = self.active_process.lock();
        match &*process {
            Some(sidecar) => matches!(
                sidecar.status().state,
                TrainingState::Preparing
                    | TrainingState::Preprocessing
                    | TrainingState::Training
                    | TrainingState::GeneratingIndex
            ),
            None => false,
        }
    }

    fn find_python() -> PathBuf {
        // Priority: embedded portable Python
        let embedded = PathBuf::from("./python/python.exe");
        if embedded.exists() {
            return embedded;
        }

        // Fallback for development: look for system Python
        // (in production, always use embedded)
        PathBuf::from("python")
    }
}
