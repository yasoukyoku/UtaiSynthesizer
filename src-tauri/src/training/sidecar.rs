use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex;

use super::{TrainingConfig, TrainingState, TrainingStatus};
use crate::{Result, UtaiError};

pub struct TrainingSidecar {
    child: Arc<Mutex<Option<Child>>>,
    status: Arc<Mutex<TrainingStatus>>,
    start_time: Instant,
}

impl TrainingSidecar {
    pub fn spawn(python_path: &PathBuf, config: &TrainingConfig) -> Result<Self> {
        let config_json = serde_json::to_string(config)?;
        let script = match &config.backend {
            super::TrainingBackend::Rvc { .. } => "training/rvc/train.py",
            super::TrainingBackend::SoVits { .. } => "training/sovits/train.py",
        };

        let mut child = Command::new(python_path)
            .arg(script)
            .arg("--config")
            .arg(&config_json)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| UtaiError::Training(format!("Failed to spawn Python: {}", e)))?;

        let status = Arc::new(Mutex::new(TrainingStatus {
            state: TrainingState::Preparing,
            current_epoch: 0,
            total_epochs: config.epochs,
            loss: None,
            elapsed_secs: 0,
            eta_secs: None,
            model_name: config.model_name.clone(),
        }));

        let stdout = child.stdout.take();
        let status_clone = status.clone();

        if let Some(stdout) = stdout {
            std::thread::spawn(move || {
                let reader = BufReader::new(stdout);
                for line in reader.lines().flatten() {
                    if let Some(update) = parse_progress_line(&line) {
                        let mut s = status_clone.lock();
                        match update {
                            ProgressUpdate::State(state) => s.state = state,
                            ProgressUpdate::Epoch(e) => s.current_epoch = e,
                            ProgressUpdate::Loss(l) => s.loss = Some(l),
                            ProgressUpdate::Eta(eta) => s.eta_secs = Some(eta),
                        }
                    }
                }
                // Process ended — mark as completed or stopped
                let mut s = status_clone.lock();
                if s.state == TrainingState::Training || s.state == TrainingState::GeneratingIndex {
                    s.state = TrainingState::Completed;
                }
            });
        }

        Ok(Self {
            child: Arc::new(Mutex::new(Some(child))),
            status,
            start_time: Instant::now(),
        })
    }

    pub fn status(&self) -> TrainingStatus {
        let mut s = self.status.lock().clone();
        s.elapsed_secs = self.start_time.elapsed().as_secs();
        s
    }

    pub fn request_stop(&mut self) -> Result<()> {
        // Send graceful stop signal via stdin or signal
        // The Python process will:
        // 1. Finish current batch
        // 2. Save checkpoint
        // 3. Generate index file
        // 4. Exit cleanly
        self.status.lock().state = TrainingState::GeneratingIndex;

        if let Some(ref mut child) = *self.child.lock() {
            // On Windows, we write a stop signal to a file the Python process watches
            let pid = child.id();
            let stop_file = PathBuf::from(format!("training/.stop_{}", pid));
            std::fs::write(&stop_file, "stop").ok();
        }

        Ok(())
    }

    pub fn force_kill(self) -> Result<()> {
        if let Some(mut child) = self.child.lock().take() {
            child
                .kill()
                .map_err(|e| UtaiError::Training(format!("Kill failed: {}", e)))?;
        }
        Ok(())
    }
}

enum ProgressUpdate {
    State(TrainingState),
    Epoch(u32),
    Loss(f64),
    Eta(u64),
}

fn parse_progress_line(line: &str) -> Option<ProgressUpdate> {
    // Protocol: Python prints JSON lines with progress info
    // {"type": "state", "value": "training"}
    // {"type": "epoch", "value": 15}
    // {"type": "loss", "value": 0.0234}
    // {"type": "eta", "value": 3600}
    let json: serde_json::Value = serde_json::from_str(line).ok()?;
    let msg_type = json.get("type")?.as_str()?;

    match msg_type {
        "state" => {
            let state_str = json.get("value")?.as_str()?;
            let state = match state_str {
                "preparing" => TrainingState::Preparing,
                "preprocessing" => TrainingState::Preprocessing,
                "training" => TrainingState::Training,
                "generating_index" => TrainingState::GeneratingIndex,
                "completed" => TrainingState::Completed,
                "stopped" => TrainingState::Stopped,
                other => TrainingState::Error(other.to_string()),
            };
            Some(ProgressUpdate::State(state))
        }
        "epoch" => {
            let epoch = json.get("value")?.as_u64()? as u32;
            Some(ProgressUpdate::Epoch(epoch))
        }
        "loss" => {
            let loss = json.get("value")?.as_f64()?;
            Some(ProgressUpdate::Loss(loss))
        }
        "eta" => {
            let eta = json.get("value")?.as_u64()?;
            Some(ProgressUpdate::Eta(eta))
        }
        _ => None,
    }
}
