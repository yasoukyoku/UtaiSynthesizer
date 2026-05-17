use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;
use tokio::time;

use super::Project;
use crate::Result;

const AUTOSAVE_INTERVAL: Duration = Duration::from_secs(60);
const MAX_AUTOSAVES: usize = 5;

pub struct AutosaveManager {
    autosave_dir: PathBuf,
    running: Arc<parking_lot::Mutex<bool>>,
}

impl AutosaveManager {
    pub fn new(autosave_dir: PathBuf) -> Self {
        std::fs::create_dir_all(&autosave_dir).ok();
        Self {
            autosave_dir,
            running: Arc::new(parking_lot::Mutex::new(false)),
        }
    }

    pub fn start(&self, project: Arc<RwLock<Option<Project>>>) {
        let dir = self.autosave_dir.clone();
        let running = self.running.clone();

        *running.lock() = true;

        tokio::spawn(async move {
            let mut interval = time::interval(AUTOSAVE_INTERVAL);
            loop {
                interval.tick().await;

                if !*running.lock() {
                    break;
                }

                let proj = project.read();
                if let Some(ref p) = *proj {
                    if p.dirty {
                        let filename = format!("autosave_{}.utai", timestamp_suffix());
                        let path = dir.join(&filename);
                        if let Ok(json) = serde_json::to_string(p) {
                            if std::fs::write(&path, json).is_ok() {
                                tracing::debug!("Autosaved: {}", filename);
                                cleanup_old_autosaves(&dir);
                            }
                        }
                    }
                }
            }
        });
    }

    pub fn stop(&self) {
        *self.running.lock() = false;
    }
}

fn timestamp_suffix() -> u64 {
    use std::time::SystemTime;
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn cleanup_old_autosaves(dir: &PathBuf) {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_str()
                .map(|n| n.starts_with("autosave_"))
                .unwrap_or(false)
        })
        .collect();

    entries.sort_by_key(|e| std::cmp::Reverse(e.metadata().ok().and_then(|m| m.modified().ok())));

    for entry in entries.iter().skip(MAX_AUTOSAVES) {
        std::fs::remove_file(entry.path()).ok();
    }
}
