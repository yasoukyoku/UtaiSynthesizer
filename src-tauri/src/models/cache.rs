use std::collections::HashMap;
use std::path::PathBuf;

use parking_lot::RwLock;

use crate::inference::engine::OnnxEngine;
use crate::Result;

pub struct ModelCache {
    loaded: RwLock<HashMap<String, CachedModel>>,
    max_memory_mb: u64,
}

struct CachedModel {
    session_id: String,
    model_path: PathBuf,
    last_used: std::time::Instant,
    approx_size_mb: u64,
}

impl ModelCache {
    pub fn new(max_memory_mb: u64) -> Self {
        Self {
            loaded: RwLock::new(HashMap::new()),
            max_memory_mb,
        }
    }

    pub fn get_or_load(
        &self,
        name: &str,
        path: &PathBuf,
        engine: &OnnxEngine,
    ) -> Result<String> {
        // Check if already loaded
        {
            let mut cache = self.loaded.write();
            if let Some(entry) = cache.get_mut(name) {
                entry.last_used = std::time::Instant::now();
                return Ok(entry.session_id.clone());
            }
        }

        // Evict if necessary
        self.evict_if_needed(engine);

        // Load new model
        let session_id = engine.load_model(path)?;
        let approx_size = estimate_model_size(path);

        let entry = CachedModel {
            session_id: session_id.clone(),
            model_path: path.clone(),
            last_used: std::time::Instant::now(),
            approx_size_mb: approx_size,
        };

        self.loaded.write().insert(name.to_string(), entry);
        Ok(session_id)
    }

    pub fn unload(&self, name: &str, engine: &OnnxEngine) {
        let mut cache = self.loaded.write();
        if let Some(entry) = cache.remove(name) {
            engine.unload_model(&entry.session_id);
        }
    }

    pub fn clear(&self, engine: &OnnxEngine) {
        let mut cache = self.loaded.write();
        for (_, entry) in cache.drain() {
            engine.unload_model(&entry.session_id);
        }
    }

    fn evict_if_needed(&self, engine: &OnnxEngine) {
        let mut cache = self.loaded.write();
        let total_mb: u64 = cache.values().map(|e| e.approx_size_mb).sum();

        if total_mb <= self.max_memory_mb {
            return;
        }

        // Evict least recently used
        let mut entries: Vec<(&String, &CachedModel)> = cache.iter().collect();
        entries.sort_by_key(|(_, e)| e.last_used);

        let mut freed = 0u64;
        let target = total_mb - self.max_memory_mb;
        let mut to_remove: Vec<String> = Vec::new();

        for (name, entry) in &entries {
            if freed >= target {
                break;
            }
            freed += entry.approx_size_mb;
            engine.unload_model(&entry.session_id);
            to_remove.push(name.to_string());
        }

        for name in to_remove {
            cache.remove(&name);
        }
    }
}

fn estimate_model_size(path: &PathBuf) -> u64 {
    std::fs::metadata(path)
        .map(|m| m.len() / (1024 * 1024))
        .unwrap_or(50)
}
