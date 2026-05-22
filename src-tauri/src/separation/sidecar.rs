use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;

use parking_lot::Mutex;

use super::{SeparationState, SeparationStatus, StemOutput};
use crate::{Result, UtaiError};

pub struct SeparationSidecar {
    child: Arc<Mutex<Option<Child>>>,
    status: Arc<Mutex<SeparationStatus>>,
}

impl SeparationSidecar {
    pub fn spawn(python_path: &Path, script_path: &Path, config_json: &str) -> Result<Self> {
        let mut child = Command::new(python_path)
            .arg(script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| UtaiError::Audio(format!("Failed to spawn MSST Python: {}", e)))?;

        // Write config via stdin to avoid CLI arg Unicode issues on Windows
        {
            use std::io::Write;
            let mut stdin = child.stdin.take()
                .ok_or_else(|| UtaiError::Audio("Failed to open stdin pipe".to_string()))?;
            stdin.write_all(config_json.as_bytes())
                .map_err(|e| UtaiError::Audio(format!("Failed to write config to stdin: {}", e)))?;
        }

        let status = Arc::new(Mutex::new(SeparationStatus {
            state: SeparationState::LoadingModel,
            stems: None,
            progress: 0.0,
        }));

        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let status_clone = status.clone();

        if let Some(stdout) = stdout {
            std::thread::spawn(move || {
                let reader = BufReader::new(stdout);
                for line in reader.lines().flatten() {
                    if let Some(update) = parse_line(&line) {
                        let mut s = status_clone.lock();
                        match update {
                            SepUpdate::State(state) => s.state = state,
                            SepUpdate::Progress(p) => s.progress = p,
                            SepUpdate::Stems(stems) => s.stems = Some(stems),
                            SepUpdate::Error(msg) => s.state = SeparationState::Error(msg),
                        }
                    }
                }
                let mut s = status_clone.lock();
                if matches!(s.state, SeparationState::LoadingModel | SeparationState::Separating) {
                    s.state = SeparationState::Completed;
                }
            });
        }

        // Consume stderr to prevent pipe buffer deadlock from PyTorch warnings
        if let Some(stderr) = stderr {
            std::thread::spawn(move || {
                let reader = BufReader::new(stderr);
                for line in reader.lines().flatten() {
                    tracing::debug!("[MSST] {}", line);
                }
            });
        }

        Ok(Self {
            child: Arc::new(Mutex::new(Some(child))),
            status,
        })
    }

    pub fn status(&self) -> SeparationStatus {
        self.status.lock().clone()
    }

    pub fn cancel(&self) -> Result<()> {
        if let Some(mut child) = self.child.lock().take() {
            child
                .kill()
                .map_err(|e| UtaiError::Audio(format!("Kill failed: {}", e)))?;
        }
        self.status.lock().state = SeparationState::Error("Cancelled".to_string());
        Ok(())
    }
}

enum SepUpdate {
    State(SeparationState),
    Progress(f32),
    Stems(Vec<StemOutput>),
    Error(String),
}

fn parse_line(line: &str) -> Option<SepUpdate> {
    let json: serde_json::Value = serde_json::from_str(line).ok()?;
    let msg_type = json.get("type")?.as_str()?;

    match msg_type {
        "state" => {
            let val = json.get("value")?.as_str()?;
            let state = match val {
                "loading_model" => SeparationState::LoadingModel,
                "separating" => SeparationState::Separating,
                "completed" => SeparationState::Completed,
                other => SeparationState::Error(other.to_string()),
            };
            Some(SepUpdate::State(state))
        }
        "progress" => {
            let p = json.get("value")?.as_f64()? as f32;
            Some(SepUpdate::Progress(p))
        }
        "stems" => {
            let arr = json.get("value")?.as_array()?;
            let stems: Vec<StemOutput> = arr
                .iter()
                .filter_map(|v| {
                    Some(StemOutput {
                        label: v.get("label")?.as_str()?.to_string(),
                        path: v.get("path")?.as_str()?.to_string(),
                    })
                })
                .collect();
            Some(SepUpdate::Stems(stems))
        }
        "error" => {
            let msg = json.get("value")?.as_str()?.to_string();
            Some(SepUpdate::Error(msg))
        }
        _ => None,
    }
}
