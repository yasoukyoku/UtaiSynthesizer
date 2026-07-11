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
/// cache is cleared when the device changes. Keep this small-ish: each MSST/voice session
/// can be hundreds of MB. S36 raised 4 → 8: the SoVITS quality path holds SIX sessions at
/// once (voice + contentvec + rmvpe + diffusion encoder + denoiser + nsf-hifigan vocoder)
/// and round-robins them per piece — at 4 the LRU would evict/rebuild a session EVERY
/// piece (multi-second stalls that read as a hang). Aux/vocoder sessions are small and the
/// aux ones sit on CPU; MSST still bounds its own footprint via release_others().
const MAX_CACHED_SESSIONS: usize = 8;

pub struct OnnxEngine {
    sessions: RwLock<HashMap<String, LoadedSession>>,
    /// key → load spec (path, mem_pattern, device override), KEPT even after a session is
    /// LRU-evicted so `run()` can REBUILD a session a consumer still holds with the SAME options.
    /// The cache is shared by voices + the F0/aux models + MSST, so eviction of a key another layer
    /// still points at must NOT hard-fail with "Session not found".
    paths: RwLock<HashMap<String, LoadSpec>>,
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

/// How a session was loaded — enough to rebuild it identically after LRU eviction.
#[derive(Clone)]
struct LoadSpec {
    path: PathBuf,
    mem_pattern: bool,
    /// Force a specific EP for this session (aux feature extractors → CPU to keep them off VRAM),
    /// or None to follow the global device preference.
    device: Option<DeviceConfig>,
}

struct LoadedSession {
    session: Mutex<Session>,
    path: PathBuf,
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
    /// Boolean tensor — the ScoreToCV `phone_mask` input (all-true at deploy, B=1). Added S48 Phase 1c.
    Bool { data: Vec<bool>, shape: Vec<i64> },
}

/// Typed output tensor with shape (S60). GAME's graphs return bool tensors (boundaries/
/// presence/masks) and the caller needs the dynamic dims (T frames / N notes) back.
pub enum OutputTensor {
    F32 { shape: Vec<i64>, data: Vec<f32> },
    Bool { shape: Vec<i64>, data: Vec<bool> },
    I64 { shape: Vec<i64>, data: Vec<i64> },
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
    /// Load with memory-pattern ENABLED (default; correct for static-shape models — MSST chunks).
    pub fn load_model(&self, path: &PathBuf) -> Result<String> {
        self.load_model_opts(path, true, None)
    }

    /// Load with explicit memory-pattern control, following the global device. Dynamic-shape
    /// models (voice: per-chunk T varies) pass `mem_pattern=false`.
    pub fn load_model_with(&self, path: &PathBuf, mem_pattern: bool) -> Result<String> {
        self.load_model_opts(path, mem_pattern, None)
    }

    /// Load on a FORCED device (e.g. aux feature extractors on CPU to keep them off VRAM), or
    /// `device_override = None` to follow the global preference.
    pub fn load_model_on(
        &self,
        path: &PathBuf,
        mem_pattern: bool,
        device_override: DeviceConfig,
    ) -> Result<String> {
        self.load_model_opts(path, mem_pattern, Some(device_override))
    }

    fn load_model_opts(
        &self,
        path: &PathBuf,
        mem_pattern: bool,
        device_override: Option<DeviceConfig>,
    ) -> Result<String> {
        *self.last_activity.lock() = Instant::now(); // loading counts as activity (don't release mid-load)
        if !path.exists() {
            return Err(UtaiError::Inference(format!(
                "Model file not found: {}",
                path.display()
            )));
        }

        // Effective device = the per-session override, else the global preference.
        let device = device_override
            .clone()
            .unwrap_or_else(|| self.device.read().clone());
        let key = Self::make_key(path, &device);
        let tick = self.clock.fetch_add(1, Ordering::Relaxed);
        // Remember the full spec under this key so an evicted session rebuilds identically.
        self.paths.write().insert(
            key.clone(),
            LoadSpec {
                path: path.clone(),
                mem_pattern,
                device: device_override,
            },
        );

        // Cache hit → reuse, no reload.
        if let Some(loaded) = self.sessions.read().get(&key) {
            loaded.last_used.store(tick, Ordering::Relaxed);
            tracing::info!("Reusing cached ONNX session: {}", path.display());
            return Ok(key);
        }

        // Miss → build outside the write lock (commit_from_file is the slow part).
        tracing::info!("Building ONNX session: {} (device={:?})", path.display(), device);
        let (session, actual_device) = build_session(path, &device, mem_pattern)?;

        let mut sessions = self.sessions.write();
        // Another thread may have built the same model while we were compiling.
        if !sessions.contains_key(&key) {
            // Evict least-recently-used entries while at capacity. The cap is PER DEVICE
            // CLASS (S60): GPU sessions keep the original MAX_CACHED_SESSIONS semantics
            // among themselves (the cap exists to bound VRAM — S36), while CPU sessions
            // (aux extractors + GAME's 4-graph fleet) count against their OWN cap. Without
            // the split, a minutes-long GAME extraction (4 CPU sessions, last_used
            // constantly refreshed) squeezes the SoVITS quality path's 6 GPU sessions out
            // of the shared cap → the S36 "evict/rebuild every piece, reads as a hang"
            // regression returns whenever both run concurrently.
            let new_is_cpu = actual_device == "CPU";
            let same_class = |v: &LoadedSession| (v.actual_device == "CPU") == new_is_cpu;
            while sessions.values().filter(|v| same_class(v)).count() >= MAX_CACHED_SESSIONS {
                let evict = sessions
                    .iter()
                    .filter(|(_, v)| same_class(v))
                    .min_by_key(|(_, v)| v.last_used.load(Ordering::Relaxed))
                    .map(|(k, _)| k.clone());
                match evict {
                    Some(k) => {
                        sessions.remove(&k);
                        tracing::info!(
                            "Evicted least-recently-used ONNX session (per-class cache cap {})",
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
                    path: path.clone(),
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

    /// Free every cached session EXCEPT the one for `keep_path`. Call before loading a
    /// separation-size model: two separation CUDA arenas cannot coexist on a 12 GB card
    /// (S31: BSRoformer's resident post-run arena + MelBand's T=1101 working set → WDDM
    /// shared-memory overcommit, per-chunk time explodes and the run looks hung; the idle
    /// sweeper would only reclaim it minutes later). Evicted sessions rebuild on demand via
    /// run()'s reload-on-miss — the `paths` map survives — so voice/vocoder models just pay
    /// a few seconds' reload on their next use.
    pub fn release_others(&self, keep_path: &Path) -> usize {
        let mut sessions = self.sessions.write();
        let before = sessions.len();
        sessions.retain(|_, s| s.path == keep_path);
        let n = before - sessions.len();
        if n > 0 {
            tracing::info!(
                "Released {} other cached ONNX session(s) before loading {} (GPU memory headroom)",
                n, keep_path.display()
            );
        }
        n
    }

    /// Free every GPU-bound cached session whose path is NOT under one of `keep`
    /// (prefix match). "GPU-bound" = LoadSpec.device ≠ forced-CPU — the CPU aux
    /// extractors hold no VRAM and stay warm. The symmetric counterpart of
    /// release_others(): separation evicts voice sessions before ITS big load; a voice
    /// run calls this with its OWN model family, so a leftover MSST arena / the other
    /// backend's fleet is released before the new load peaks (S36: cap 4→8 removed the
    /// LRU pressure that used to evict them incidentally — a stale SoVITS fleet + a
    /// full-GPU RVC run stacked to ~12 GB). The GPU AUX extractor sessions are
    /// deliberately NOT in any keep-set: ORT CUDA arenas only ever GROW, so a "kept
    /// warm" extractor carries the previous run's activation high-water mark into the
    /// next run (live-measured +1.4 GB) — every run's VRAM must equal its OWN
    /// footprint. Reload specs survive (reload-on-miss).
    pub fn release_gpu_sessions_except(&self, keep: &[PathBuf]) -> usize {
        let paths = self.paths.read();
        let mut sessions = self.sessions.write();
        let before = sessions.len();
        sessions.retain(|key, _| {
            let Some(spec) = paths.get(key) else { return true };
            if matches!(spec.device, Some(DeviceConfig::Cpu)) {
                return true; // forced-CPU sessions hold no VRAM
            }
            keep.iter().any(|k| spec.path.starts_with(k))
        });
        let n = before - sessions.len();
        if n > 0 {
            tracing::info!(
                "Released {} GPU session(s) not needed by this voice run (VRAM headroom)",
                n
            );
        }
        n
    }

    /// Unload every session (AND its reload spec) whose file path equals `prefix` or lives
    /// under it. Used when a model's on-disk artifacts are replaced/deleted (re-import):
    /// the engine caches by path, so without this a re-imported `<stem>.f0.onnx` /
    /// `<stem>.diffusion/*.onnx` would keep serving the OLD graph. Removing from `paths`
    /// too is deliberate — reload-on-miss must not resurrect a stale spec for a replaced
    /// file. Prefix matching is on the raw stored path (same base the loaders passed in).
    pub fn unload_paths_with_prefix(&self, prefix: &Path) -> usize {
        let keys: Vec<String> = {
            let paths = self.paths.read();
            paths
                .iter()
                .filter(|(_, spec)| spec.path.starts_with(prefix))
                .map(|(k, _)| k.clone())
                .collect()
        };
        if keys.is_empty() {
            return 0;
        }
        {
            let mut sessions = self.sessions.write();
            for k in &keys {
                sessions.remove(k);
            }
        }
        {
            let mut paths = self.paths.write();
            for k in &keys {
                paths.remove(k);
            }
        }
        tracing::info!(
            "Unloaded {} session spec(s) under {} (files replaced)",
            keys.len(),
            prefix.display()
        );
        keys.len()
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

    /// f32-only convenience wrapper over `run_typed` — every voice/aux model returns f32
    /// tensors, so this stays the pipeline-facing signature. Non-f32 graphs (GAME's bool
    /// boundaries/presence) go through `run_typed` directly.
    pub fn run(
        &self,
        session_id: &str,
        inputs: Vec<(&str, InputTensor)>,
    ) -> Result<Vec<Vec<f32>>> {
        let outputs = self.run_typed(session_id, inputs)?;
        let mut result = Vec::with_capacity(outputs.len());
        for (i, out) in outputs.into_iter().enumerate() {
            match out {
                OutputTensor::F32 { data, .. } => result.push(data),
                _ => {
                    return Err(UtaiError::Inference(format!(
                        "Output {}: expected an f32 tensor",
                        i
                    )))
                }
            }
        }
        Ok(result)
    }

    /// Run returning TYPED outputs with shapes. Needed by graphs whose outputs are not all
    /// f32 and whose dynamic dims (T frames / N notes) the caller must recover from the
    /// shape (GAME midi_extract, S60). Same session/reload/in-flight semantics as `run`.
    pub fn run_typed(
        &self,
        session_id: &str,
        inputs: Vec<(&str, InputTensor)>,
    ) -> Result<Vec<OutputTensor>> {
        *self.last_activity.lock() = Instant::now();
        self.in_flight.fetch_add(1, Ordering::Relaxed);
        let _in_flight = InFlightGuard(self); // declared before `sessions` so its Drop runs lock-free
        // Reload-on-miss: the shared LRU cache may have evicted a session a consumer still holds.
        // Rebuild it from the remembered path (same key, same device) instead of hard-failing.
        if !self.sessions.read().contains_key(session_id) {
            let spec = self.paths.read().get(session_id).cloned();
            if let Some(spec) = spec {
                // A device switch mid-job orphans the old-device key: load_model would rebuild under
                // the CURRENT device's key, never this one. Fail with a clear message instead of the
                // cryptic "Session not found". (Forced-device sessions — aux on CPU — are immune:
                // their effective device never depends on the global preference.)
                let effective = spec.device.clone().unwrap_or_else(|| self.device.read().clone());
                if Self::make_key(&spec.path, &effective) != session_id {
                    return Err(UtaiError::Inference(
                        "Device preference changed while a job was running — re-run the job".to_string(),
                    ));
                }
                tracing::info!("ONNX session evicted — reloading: {}", spec.path.display());
                self.load_model_opts(&spec.path, spec.mem_pattern, spec.device)?;
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
                InputTensor::Bool { data, shape } => {
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
            let out = &outputs[i];
            let typed = if let Ok((shape, data)) = out.try_extract_tensor::<f32>() {
                OutputTensor::F32 { shape: shape.to_vec(), data: data.to_vec() }
            } else if let Ok((shape, data)) = out.try_extract_tensor::<bool>() {
                OutputTensor::Bool { shape: shape.to_vec(), data: data.to_vec() }
            } else if let Ok((shape, data)) = out.try_extract_tensor::<i64>() {
                OutputTensor::I64 { shape: shape.to_vec(), data: data.to_vec() }
            } else {
                return Err(UtaiError::Inference(format!(
                    "Output {}: unsupported tensor dtype (expected f32/bool/i64)",
                    i
                )));
            };
            result.push(typed);
        }

        Ok(result)
    }
}

fn base_builder(mem_pattern: bool) -> Result<ort::session::builder::SessionBuilder> {
    let builder = Session::builder()
        .map_err(|e| UtaiError::Inference(format!("Session builder: {}", e)))?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
        .map_err(|e| UtaiError::Inference(format!("Optimization: {}", e)))?;
    if !mem_pattern {
        // Dynamic-shape models (voice: T varies per chunk / per song) make ORT's memory-pattern
        // planner reserve for the largest-seen shape and HOLD it — the runtime log showed 249 MB
        // reserved for an 82 MB tensor ("block in memory pattern size … fall back to default"),
        // ×many tensors ×3 big sessions = the VRAM exhaustion the user hit. Disabling only changes
        // ALLOCATION strategy (never numerics; MSST md5 gate unaffected). MSST keeps it ON — fixed
        // chunk length = static shapes, exactly where the pattern helps.
        return builder
            .with_memory_pattern(false)
            .map_err(|e| UtaiError::Inference(format!("Disable mem pattern: {}", e)));
    }
    Ok(builder)
}

fn commit(mut builder: ort::session::builder::SessionBuilder, path: &Path) -> Result<Session> {
    // ORT's CreateSession squeezes the path through the process ACP on Windows — any character
    // outside the active codepage (e.g. 东雪莲 on a cp932 system) fails with "No mapping for the
    // Unicode character". Non-ASCII paths therefore load via memory; ASCII paths keep the
    // file route (mmap-friendly, and the entire MSST catalog is ASCII-named).
    let ascii_path = path.to_str().map(|s| s.is_ascii()).unwrap_or(false);
    if ascii_path {
        builder
            .commit_from_file(path)
            .map_err(|e| UtaiError::Inference(format!("Load model '{}': {}", path.display(), e)))
    } else {
        let bytes = std::fs::read(path)
            .map_err(|e| UtaiError::Inference(format!("Read model '{}': {}", path.display(), e)))?;
        builder
            .commit_from_memory(&bytes)
            .map_err(|e| UtaiError::Inference(format!("Load model '{}': {}", path.display(), e)))
    }
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
    // SameAsRequested: grow the CUDA arena only by what each allocation needs, instead of the
    // default power-of-two doubling. Measured (S31 perf investigation): with the default strategy
    // the arena for the BSRoformer T=801 attention workload balloons to 11.9 of 12 GiB, WDDM
    // starts paging VRAM over PCIe, and per-chunk inference intermittently drops from ~890 ms to
    // 2.6-3.7 s (bit-identical outputs — pure residency thrash; the user-facing "5-minute song").
    // Numerics are unaffected by allocator strategy. NOT adding gpu_mem_limit: a hard cap turns
    // "slow" into "allocation failed" on smaller cards. (S23 HARD RULE: any new EP option must be
    // A/B-run on the real model against this exact ort/ORT pair before shipping — done for this one.)
    let mut ep = ort::ep::CUDA::default()
        .with_tf32(true)
        .with_arena_extend_strategy(ort::ep::ArenaExtendStrategy::SameAsRequested);
    if let Some(id) = device_id {
        ep = ep.with_device_id(id);
    }
    ep
}

fn build_session(path: &PathBuf, device: &DeviceConfig, mem_pattern: bool) -> Result<(Session, String)> {
    // Auto resolves at runtime by probing each EP — handled separately so we can LOG the device
    // that actually ran (the old all-EPs-at-once registration could not report which one ORT picked).
    if matches!(device, DeviceConfig::Auto) {
        return build_session_auto(path, mem_pattern);
    }

    let mut builder = base_builder(mem_pattern)?;

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
fn build_session_auto(path: &Path, mem_pattern: bool) -> Result<(Session, String)> {
    let cuda = base_builder(mem_pattern)
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

    let dml = base_builder(mem_pattern)
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

    let builder = base_builder(mem_pattern)?
        .with_execution_providers([ort::ep::CPU::default().build()])
        .map_err(|e| UtaiError::Inference(format!("CPU EP: {}", e)))?;
    let session = commit(builder, path)?;
    tracing::info!("ONNX device=Auto → using CPU");
    Ok((session, "CPU (Auto)".to_string()))
}
