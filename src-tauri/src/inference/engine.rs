use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;
use parking_lot::{Mutex, RwLock};

use crate::{Result, UtaiError};

/// Max number of ONNX sessions kept resident at once.
///
/// Sessions are cached by (path + device) and are NOT freed when a pipeline drops —
/// repeated runs of the same model reuse the loaded session (skips the multi-second
/// parse + graph optimization + EP init). To bound memory, the cache is an LRU capped
/// at this size; the least-recently-used session is evicted on overflow, and the whole
/// cache is cleared when the device changes. Keep this small: each MSST/voice session
/// can be hundreds of MB. 4 leaves headroom for a separation model + a couple of voice
/// models without unbounded growth.
const MAX_CACHED_SESSIONS: usize = 4;

pub struct OnnxEngine {
    sessions: RwLock<HashMap<String, LoadedSession>>,
    /// key → model path, KEPT even after a session is LRU-evicted, so `run()` can REBUILD a session a
    /// consumer still holds. The cache is shared by voices + the F0 model + MSST, so eviction of a key
    /// another layer still points at must NOT hard-fail with "Session not found".
    paths: RwLock<HashMap<String, PathBuf>>,
    device: RwLock<DeviceConfig>,
    clock: AtomicU64,
    /// Wall-clock of the last inference, for idle-release: a background sweeper frees the cached sessions
    /// (and the resident ORT CUDA arena) after a stretch of inactivity. A new run refreshes this + reloads
    /// on demand, so actively working never pays a reload.
    last_activity: Mutex<Instant>,
    /// `run()` calls currently executing. last_activity alone can't protect a single inference call
    /// that outlasts the idle window (it's only refreshed between calls) — the sweeper skips while
    /// this is non-zero.
    in_flight: AtomicUsize,
}

/// RAII guard for `in_flight` — decrements on drop so `run()`'s error paths count down too, and
/// refreshes last_activity on exit (the idle clock must restart from the END of a long call).
struct InFlightGuard<'a>(&'a OnnxEngine);

impl Drop for InFlightGuard<'_> {
    fn drop(&mut self) {
        *self.0.last_activity.lock() = Instant::now();
        self.0.in_flight.fetch_sub(1, Ordering::Relaxed);
    }
}

struct LoadedSession {
    session: Mutex<Session>,
    _path: PathBuf,
    last_used: AtomicU64,
    /// Human label of the EP actually backing this session ("CUDA (GPU)", "DirectML (GPU)", "CPU").
    /// Logged at inference time so the user can confirm what really ran (and catch silent fallbacks).
    actual_device: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceConfig {
    Cpu,
    #[serde(rename = "directml")]
    DirectMl { device_id: u32 },
    Cuda { device_id: u32 },
    Auto,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self::Auto
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
            paths: RwLock::new(HashMap::new()),
            device: RwLock::new(DeviceConfig::default()),
            clock: AtomicU64::new(0),
            last_activity: Mutex::new(Instant::now()),
            in_flight: AtomicUsize::new(0),
        }
    }

    pub fn set_device(&self, config: DeviceConfig) {
        let changed = *self.device.read() != config;
        tracing::info!("Device preference set to: {:?}", config);
        *self.device.write() = config;
        // Cached sessions were built for the previous execution provider — drop them so
        // they rebuild for the new device on next load (and free the old EP's memory).
        if changed {
            let n = {
                let mut s = self.sessions.write();
                let n = s.len();
                s.clear();
                n
            };
            // `paths` is KEPT: keys embed the device, so old-device entries are unreachable by new
            // loads, and run() needs the surviving entry to report a mid-job device change clearly.
            if n > 0 {
                tracing::info!("Cleared {} cached ONNX session(s) after device change", n);
            }
        }
    }

    pub fn device(&self) -> DeviceConfig {
        self.device.read().clone()
    }

    /// EP label actually backing a loaded session — for inference-time "what hardware ran" logging.
    pub fn resolved_device(&self, session_id: &str) -> Option<String> {
        self.sessions.read().get(session_id).map(|s| s.actual_device.clone())
    }

    fn make_key(path: &Path, device: &DeviceConfig) -> String {
        let canon = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        format!("{}|{:?}", canon.display(), device)
    }

    /// Load (or reuse) a session for `path` under the current device.
    ///
    /// Returns a stable cache key (used as the session id by `run`). A second call for the
    /// same (path, device) returns the cached session with no reload.
    pub fn load_model(&self, path: &PathBuf) -> Result<String> {
        *self.last_activity.lock() = Instant::now(); // loading counts as activity (don't release mid-load)
        if !path.exists() {
            return Err(UtaiError::Inference(format!(
                "Model file not found: {}",
                path.display()
            )));
        }

        let device = self.device.read().clone();
        let key = Self::make_key(path, &device);
        let tick = self.clock.fetch_add(1, Ordering::Relaxed);
        // Remember the path under this key (idempotent) so an evicted session can be rebuilt by run().
        self.paths.write().insert(key.clone(), path.clone());

        // Cache hit → reuse, no reload.
        if let Some(loaded) = self.sessions.read().get(&key) {
            loaded.last_used.store(tick, Ordering::Relaxed);
            tracing::info!("Reusing cached ONNX session: {}", path.display());
            return Ok(key);
        }

        // Miss → build outside the write lock (commit_from_file is the slow part).
        tracing::info!("Building ONNX session: {} (device={:?})", path.display(), device);
        let (session, actual_device) = build_session(path, &device)?;

        let mut sessions = self.sessions.write();
        // Another thread may have built the same model while we were compiling.
        if !sessions.contains_key(&key) {
            // Evict least-recently-used entries while at capacity (bounds memory).
            while sessions.len() >= MAX_CACHED_SESSIONS {
                let evict = sessions
                    .iter()
                    .min_by_key(|(_, v)| v.last_used.load(Ordering::Relaxed))
                    .map(|(k, _)| k.clone());
                match evict {
                    Some(k) => {
                        sessions.remove(&k);
                        tracing::info!(
                            "Evicted least-recently-used ONNX session (cache cap {})",
                            MAX_CACHED_SESSIONS
                        );
                    }
                    None => break,
                }
            }
            sessions.insert(
                key.clone(),
                LoadedSession {
                    session: Mutex::new(session),
                    _path: path.clone(),
                    last_used: AtomicU64::new(tick),
                    actual_device,
                },
            );
            tracing::info!(
                "Loaded ONNX model: {} ({} session(s) cached)",
                path.display(),
                sessions.len()
            );
        }
        Ok(key)
    }

    pub fn unload_model(&self, id: &str) {
        if self.sessions.write().remove(id).is_some() {
            tracing::info!("Unloaded ONNX session");
        }
    }

    /// Drop all cached sessions (frees their memory). Safe to call between runs.
    pub fn clear_sessions(&self) {
        let n = {
            let mut s = self.sessions.write();
            let n = s.len();
            s.clear();
            n
        };
        if n > 0 {
            tracing::info!("Cleared {} cached ONNX session(s)", n);
        }
    }

    pub fn is_loaded(&self, id: &str) -> bool {
        self.sessions.read().contains_key(id)
    }

    /// Free all cached sessions IF there's been no inference for `idle`. Returns the count released.
    /// A background sweeper calls this so the resident ORT CUDA arena returns to the driver after the
    /// user stops working; the next run reloads on demand (reload-on-miss in `run`, which is why we keep
    /// the `paths` map). Never fires mid-job: `in_flight` covers a single call that outlasts `idle`,
    /// and activity is re-checked under the write lock against a run() that started while acquiring it.
    pub fn release_if_idle(&self, idle: Duration) -> usize {
        if self.in_flight.load(Ordering::Relaxed) > 0 || self.last_activity.lock().elapsed() < idle {
            return 0;
        }
        let mut sessions = self.sessions.write();
        if self.in_flight.load(Ordering::Relaxed) > 0 || self.last_activity.lock().elapsed() < idle {
            return 0;
        }
        let n = sessions.len();
        if n == 0 {
            return 0;
        }
        sessions.clear(); // keep `paths` so run() can reload-on-miss
        tracing::info!("Idle — released {} cached ONNX session(s) to free GPU memory", n);
        n
    }

    pub fn run(
        &self,
        session_id: &str,
        inputs: Vec<(&str, InputTensor)>,
    ) -> Result<Vec<Vec<f32>>> {
        *self.last_activity.lock() = Instant::now();
        self.in_flight.fetch_add(1, Ordering::Relaxed);
        let _in_flight = InFlightGuard(self); // declared before `sessions` so its Drop runs lock-free
        // Reload-on-miss: the shared LRU cache may have evicted a session a consumer still holds.
        // Rebuild it from the remembered path (same key, same device) instead of hard-failing.
        if !self.sessions.read().contains_key(session_id) {
            let path = self.paths.read().get(session_id).cloned();
            if let Some(path) = path {
                // A device switch mid-job orphans the old-device key: load_model would rebuild under
                // the CURRENT device's key, never this one. Fail with a clear message instead of the
                // cryptic "Session not found".
                if Self::make_key(&path, &self.device.read()) != session_id {
                    return Err(UtaiError::Inference(
                        "Device preference changed while a job was running — re-run the job".to_string(),
                    ));
                }
                tracing::info!("ONNX session evicted — reloading: {}", path.display());
                self.load_model(&path)?;
            }
        }
        let sessions = self.sessions.read();
        let loaded = sessions.get(session_id).ok_or_else(|| {
            UtaiError::Inference(format!("Session '{}' not found", session_id))
        })?;

        loaded
            .last_used
            .store(self.clock.fetch_add(1, Ordering::Relaxed), Ordering::Relaxed);

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

        let mut result = Vec::with_capacity(outputs.len());
        for i in 0..outputs.len() {
            let (_, data) = outputs[i]
                .try_extract_tensor::<f32>()
                .map_err(|e| UtaiError::Inference(format!("Output {}: {}", i, e)))?;
            result.push(data.to_vec());
        }

        Ok(result)
    }
}

fn base_builder() -> Result<ort::session::builder::SessionBuilder> {
    Session::builder()
        .map_err(|e| UtaiError::Inference(format!("Session builder: {}", e)))?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
        .map_err(|e| UtaiError::Inference(format!("Optimization: {}", e)))
}

fn commit(mut builder: ort::session::builder::SessionBuilder, path: &Path) -> Result<Session> {
    builder
        .commit_from_file(path)
        .map_err(|e| UtaiError::Inference(format!("Load model '{}': {}", path.display(), e)))
}

/// CUDA execution provider with TF32 enabled (~1.5–2.5x on Ampere+ tensor cores). NOTE: this single
/// constructor is shared by EVERY CUDA session — separation AND the voice/vocoder pipeline (RVC, SoVITS,
/// ContentVec, F0, NSF-HiFiGAN) — so TF32 lowers precision for ALL of them, not just the transformer
/// separators. Low risk for the f32 separators (TF32 only perturbs the predicted mask; the iSTFT runs in
/// Rust f32); the on-GPU vocoder warrants an A/B listen before relying on it for fidelity. The
/// cuDNN-conv-search / bounded-workspace knobs stay OFF — they were in the option set that crashed the
/// real MSST run on 1.24.4 (examples/ort_cuda_test's 40-run probe is a ZERO-INPUT crash/stability check
/// over real frame SHAPES, NOT a fidelity check), so they wait until that crash is root-caused. Both
/// call sites stay in lockstep.
fn cuda_ep(device_id: Option<i32>) -> ort::ep::CUDA {
    let mut ep = ort::ep::CUDA::default().with_tf32(true);
    if let Some(id) = device_id {
        ep = ep.with_device_id(id);
    }
    ep
}

fn build_session(path: &PathBuf, device: &DeviceConfig) -> Result<(Session, String)> {
    // Auto resolves at runtime by probing each EP — handled separately so we can LOG the device
    // that actually ran (the old all-EPs-at-once registration could not report which one ORT picked).
    if matches!(device, DeviceConfig::Auto) {
        return build_session_auto(path);
    }

    let mut builder = base_builder()?;

    let label = match device {
        DeviceConfig::Cpu => {
            builder = builder
                .with_execution_providers([ort::ep::CPU::default().build()])
                .map_err(|e| UtaiError::Inference(format!("CPU EP: {}", e)))?;
            tracing::info!("ONNX device=CPU");
            "CPU".to_string()
        }
        // Explicit GPU selection: make registration failure a HARD error (no silent CPU
        // fallback). Otherwise a missing GPU runtime DLL would silently run on CPU and the
        // user would just see "inference is slow" with no clue. Auto keeps the silent CPU
        // fallback by design.
        DeviceConfig::DirectMl { device_id } => {
            builder = builder
                .with_execution_providers([ort::ep::DirectML::default()
                    .with_device_id(*device_id as i32)
                    .build()
                    .error_on_failure()])
                .map_err(|e| {
                    UtaiError::Inference(format!(
                        "DirectML GPU unavailable (device {}): {}. The bundled onnxruntime may \
                         not be the DirectML build, or DirectML.dll is missing. Switch the device \
                         to Auto or CPU in Settings.",
                        device_id, e
                    ))
                })?;
            tracing::info!("ONNX device=DirectML (GPU {})", device_id);
            format!("DirectML (GPU {})", device_id)
        }
        DeviceConfig::Cuda { device_id } => {
            builder = builder
                .with_execution_providers([cuda_ep(Some(*device_id as i32))
                    .build()
                    .error_on_failure()])
                .map_err(|e| {
                    UtaiError::Inference(format!(
                        "CUDA GPU unavailable (device {}): {}. Ensure the CUDA onnxruntime DLL \
                         plus cuDNN/cuBLAS are present (Settings → Download CUDA Runtime), or \
                         switch the device to Auto or CPU.",
                        device_id, e
                    ))
                })?;
            tracing::info!("ONNX device=CUDA (GPU {})", device_id);
            format!("CUDA (GPU {})", device_id)
        }
        DeviceConfig::Auto => unreachable!("Auto handled above"),
    };

    Ok((commit(builder, path)?, label))
}

/// Auto device: probe CUDA → DirectML → CPU in order, committing with the FIRST execution provider
/// that initializes, and LOG which one actually ran (the user's "auto" no longer hides the real
/// hardware). Each GPU attempt uses `.error_on_failure()` so a missing runtime/DLL fails
/// registration fast instead of silently dropping to CPU — that explicit failure is exactly what
/// lets us report the true device. The chosen EP still gets ORT's per-op CPU fallback for any
/// node it can't run.
fn build_session_auto(path: &Path) -> Result<(Session, String)> {
    let cuda = base_builder()
        .and_then(|b| {
            b.with_execution_providers([cuda_ep(None).build().error_on_failure()])
                .map_err(|e| UtaiError::Inference(format!("CUDA EP: {}", e)))
        })
        .and_then(|b| commit(b, path));
    match cuda {
        Ok(s) => {
            tracing::info!("ONNX device=Auto → using CUDA (GPU)");
            return Ok((s, "CUDA (GPU, Auto)".to_string()));
        }
        // Probe failures are debug-level noise (like ffmpeg internals) — only the resolved device
        // below is INFO. Keeps the default log clean; switch the panel to DEBUG to see why a GPU failed.
        Err(e) => tracing::debug!("ONNX device=Auto: CUDA unavailable ({e}); trying DirectML"),
    }

    let dml = base_builder()
        .and_then(|b| {
            b.with_execution_providers([ort::ep::DirectML::default().build().error_on_failure()])
                .map_err(|e| UtaiError::Inference(format!("DirectML EP: {}", e)))
        })
        .and_then(|b| commit(b, path));
    match dml {
        Ok(s) => {
            tracing::info!("ONNX device=Auto → using DirectML (GPU)");
            return Ok((s, "DirectML (GPU, Auto)".to_string()));
        }
        Err(e) => tracing::debug!("ONNX device=Auto: DirectML unavailable ({e}); falling back to CPU"),
    }

    let builder = base_builder()?
        .with_execution_providers([ort::ep::CPU::default().build()])
        .map_err(|e| UtaiError::Inference(format!("CPU EP: {}", e)))?;
    let session = commit(builder, path)?;
    tracing::info!("ONNX device=Auto → using CPU");
    Ok((session, "CPU (Auto)".to_string()))
}
