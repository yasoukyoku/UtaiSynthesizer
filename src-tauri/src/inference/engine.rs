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
    /// Unique per BUILD of this entry (a fresh value every time the session is (re)built,
    /// never reused across rebuilds of the same key). The DML recycle revalidates against it
    /// under the write lock so a stale decision can never tear down a freshly rebuilt session.
    build_id: u64,
    /// S67b: DML shape-cache accounting — `Some` iff this session runs on DirectML.
    dml: Option<DmlShapeAccounting>,
}

/// S67b: the DirectML EP re-initializes its compiled operators + pools for every DISTINCT
/// input shape and keeps ALL of them alive for the session's lifetime (measured on the real
/// RVC net_g, tests\voice_mem_profile.rs: same shape re-run = 0 growth; each new shape =
/// +32 MB…+3.1 GB process commit in erratic allocation-bucket jumps; session drop returns
/// everything). Voice chunks are silence-seek cut, so every chunk is a NEW shape — a 4-min
/// song walked commit up to ~6.9 GB and a 16 GB community machine died mid-song (WDDM charges
/// GPU allocations against process commit, and on small-VRAM cards the pools spill RESIDENT
/// into system RAM). The engine therefore attributes pool growth to the session that caused
/// it — the commit delta across each of the session's OWN new-shape runs, excluding its first
/// shape (the unavoidable "ticket" any session must pay to run at all) — and recycles the
/// session (drop + reload-on-miss rebuild; sessions hold no cross-run state, so numerics are
/// unaffected — net_g is graph-stochastic run-to-run anyway) once that attributed growth
/// passes [`dml_extra_commit_cap_mb`]. Per-run deltas rather than a frozen process-global
/// baseline (S67b review): with several DML sessions alive (SoVITS quality path holds 4),
/// absolute baselines go stale the moment any peer recycles and the budget stops meaning
/// anything. Consequences of the delta scheme: static-shape sessions (MSST fixed chunks)
/// accumulate nothing after their first shape — the batched separation path's remainder
/// group (one extra shape) adds only its own small delta, never a recycle; same-shape loops
/// (diffusion denoiser ~100 runs/piece) measure nothing at all. App-side allocations can
/// pollute a delta only while that specific new-shape run is in flight — bounded noise,
/// worst case an early recycle.
struct DmlShapeAccounting {
    /// Hashes of every distinct input-shape signature COMPLETED since this session was
    /// built. Inserted only after a SUCCESSFUL run — a FAILED run instead drops the whole
    /// session incarnation (see run_typed's error arm): a failed compile leaves unknown
    /// pool/device state, and growth bookkeeping alone can never free headroom for the
    /// retry (post-run recycling keeps growth <= cap at every run start; review round 2).
    shapes: Mutex<std::collections::HashSet<u64>>,
    /// Σ process-commit deltas (MB) across this session's own new-shape runs, excluding the
    /// first shape. This is the reclaimable pool memory a recycle would return.
    pool_growth_mb: AtomicU64,
}

/// Process commit charge (private bytes) in MB — the number WDDM/D3D12 allocations count
/// against (and what exhausts a 16 GB machine long before working set does).
#[cfg(windows)]
fn process_commit_mb() -> u64 {
    #[repr(C)]
    struct Pmcex {
        cb: u32,
        page_fault_count: u32,
        peak_working_set: usize,
        working_set: usize,
        quota_peak_paged: usize,
        quota_paged: usize,
        quota_peak_non_paged: usize,
        quota_non_paged: usize,
        pagefile: usize,
        peak_pagefile: usize,
        private_usage: usize,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn K32GetProcessMemoryInfo(h: isize, p: *mut Pmcex, cb: u32) -> i32;
        fn GetCurrentProcess() -> isize;
    }
    unsafe {
        let mut c: Pmcex = std::mem::zeroed();
        c.cb = std::mem::size_of::<Pmcex>() as u32;
        if K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) != 0 {
            (c.private_usage / (1024 * 1024)) as u64
        } else {
            0
        }
    }
}
#[cfg(not(windows))]
fn process_commit_mb() -> u64 {
    0
}

/// System-wide memory snapshot via GlobalMemoryStatusEx: (total physical MB, available
/// COMMIT MB). Available commit = the system commit limit (RAM + current pagefile) minus
/// total committed — the pool a DML first-shape ticket must fit inside. WDDM charges GPU
/// allocations against commit, and when this pool runs dry Windows kills the process with
/// no panic and no log line (the community "silent crash at 20%"). (0, 0) on failure.
#[cfg(windows)]
pub(crate) fn system_memory_mb() -> (u64, u64) {
    #[repr(C)]
    struct MemStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page: u64,
        avail_page: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended: u64,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GlobalMemoryStatusEx(p: *mut MemStatusEx) -> i32;
    }
    unsafe {
        let mut m: MemStatusEx = std::mem::zeroed();
        m.length = std::mem::size_of::<MemStatusEx>() as u32;
        if GlobalMemoryStatusEx(&mut m) != 0 {
            (m.total_phys / (1024 * 1024), m.avail_page / (1024 * 1024))
        } else {
            (0, 0)
        }
    }
}
#[cfg(not(windows))]
pub(crate) fn system_memory_mb() -> (u64, u64) {
    (0, 0)
}

/// One-line memory snapshot for stage logs: "commit=1234 MB, sys avail=5678 MB". S67c
/// forensic thread — stage INFO/DEBUG lines carry these numbers so a community crash log
/// reads as a commit-exhaustion timeline without a debugger on the machine.
pub fn memory_stamp() -> String {
    let (_, avail) = system_memory_mb();
    format!("commit={} MB, sys avail={} MB", process_commit_mb(), avail)
}

/// DML pool budget BEYOND the first-shape ticket before a session is recycled:
/// total physical RAM / 8, clamped to [1024, 4096] MB (16 GB machine → 2 GB budget).
/// `UTAI_DML_COMMIT_CAP_MB` overrides (support/testing valve; 0 falls back to the formula).
fn dml_extra_commit_cap_mb() -> u64 {
    static CAP: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        if let Ok(raw) = std::env::var("UTAI_DML_COMMIT_CAP_MB") {
            match raw.trim().parse::<u64>() {
                Ok(mb) if mb > 0 => return mb,
                Ok(_) => {} // 0 = use the formula (documented)
                // cmd.exe preserves trailing spaces in `set X=…` values — a silently dead
                // support valve costs a whole diagnosis round; say why it was ignored.
                Err(_) => tracing::warn!(
                    "UTAI_DML_COMMIT_CAP_MB={raw:?} is not a whole MB number — using the default budget"
                ),
            }
        }
        let (total_mb, _) = system_memory_mb();
        if total_mb > 0 {
            return (total_mb / 8).clamp(1024, 4096);
        }
        2048
    })
}

/// S67c: minimum system-wide available commit (MB) required before a DirectML session may
/// compile a NEW input shape. The first-shape "ticket" of a big voice model is 1.7-3.4 GB
/// of commit paid blind inside session.run — on a machine already near the commit limit the
/// OS kills the process mid-allocation with zero diagnostics. This floor converts that
/// silent death into a loud, actionable INFERENCE_LOW_MEMORY error. Deliberately CONSERVATIVE
/// (well below any real ticket): a false positive here would block a run that could have
/// succeeded — never trade a working feature for a guard (feedback_v2_feature_parity).
/// `UTAI_INFER_MIN_AVAIL_MB` overrides (0 disables the check).
fn dml_min_avail_commit_mb() -> u64 {
    static MIN: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MIN.get_or_init(|| {
        if let Ok(raw) = std::env::var("UTAI_INFER_MIN_AVAIL_MB") {
            match raw.trim().parse::<u64>() {
                Ok(mb) => return mb, // 0 = disabled (documented)
                Err(_) => tracing::warn!(
                    "UTAI_INFER_MIN_AVAIL_MB={raw:?} is not a whole MB number — using the default floor"
                ),
            }
        }
        1024
    })
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

impl InputTensor {
    fn shape(&self) -> &[i64] {
        match self {
            InputTensor::F32 { shape, .. }
            | InputTensor::I64 { shape, .. }
            | InputTensor::Bool { shape, .. } => shape,
        }
    }
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
            // Taken out of the map under the lock, torn down AFTER it releases — a GB-scale
            // D3D12/EP teardown under the global sessions lock stalls every concurrent
            // inference (S67b; same discipline as drop_session_incarnation).
            let taken = std::mem::take(&mut *self.sessions.write());
            // `paths` is KEPT: keys embed the device, so old-device entries are unreachable by new
            // loads, and run() needs the surviving entry to report a mid-job device change clearly.
            if !taken.is_empty() {
                tracing::info!("Cleared {} cached ONNX session(s) after device change", taken.len());
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

        // Declared before the guard so LRU-evicted sessions tear down AFTER the write lock
        // releases (drop order is reverse declaration order).
        let mut evicted: Vec<LoadedSession> = Vec::new();
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
                        evicted.extend(sessions.remove(&k));
                        tracing::info!(
                            "Evicted least-recently-used ONNX session (per-class cache cap {})",
                            MAX_CACHED_SESSIONS
                        );
                    }
                    None => break,
                }
            }
            let dml = actual_device.starts_with("DirectML").then(|| DmlShapeAccounting {
                shapes: Mutex::new(std::collections::HashSet::new()),
                pool_growth_mb: AtomicU64::new(0),
            });
            sessions.insert(
                key.clone(),
                LoadedSession {
                    session: Mutex::new(session),
                    path: path.clone(),
                    last_used: AtomicU64::new(tick),
                    actual_device,
                    // The clock ticks on every load call, so this is unique per (re)build.
                    build_id: tick,
                    dml,
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
        // The write guard is a temporary — released at the end of this statement, so the
        // removed session (bound to a local) tears down AFTER the lock, like everywhere else.
        let removed = self.sessions.write().remove(id);
        if removed.is_some() {
            tracing::info!("Unloaded ONNX session");
        }
    }

    /// Take every session matching `evict` out of the cache under the write lock and hand
    /// them back for teardown AFTER the guard releases — a GB-scale D3D12/EP teardown under
    /// the global sessions lock would stall every concurrent inference (S67b introduced the
    /// discipline in drop_session_incarnation; S67c extends it to every eviction path).
    fn take_sessions_where(
        &self,
        mut evict: impl FnMut(&str, &LoadedSession) -> bool,
    ) -> Vec<LoadedSession> {
        let mut sessions = self.sessions.write();
        let keys: Vec<String> = sessions
            .iter()
            .filter(|(k, v)| evict(k, v))
            .map(|(k, _)| k.clone())
            .collect();
        keys.iter().filter_map(|k| sessions.remove(k)).collect()
    }

    /// Free every cached session EXCEPT the one for `keep_path`. Call before loading a
    /// separation-size model: two separation CUDA arenas cannot coexist on a 12 GB card
    /// (S31: BSRoformer's resident post-run arena + MelBand's T=1101 working set → WDDM
    /// shared-memory overcommit, per-chunk time explodes and the run looks hung; the idle
    /// sweeper would only reclaim it minutes later). Evicted sessions rebuild on demand via
    /// run()'s reload-on-miss — the `paths` map survives — so voice/vocoder models just pay
    /// a few seconds' reload on their next use.
    pub fn release_others(&self, keep_path: &Path) -> usize {
        let removed = self.take_sessions_where(|_, s| s.path != keep_path);
        let n = removed.len();
        if n > 0 {
            // Commit before/after the teardown: on a healthy driver the pools return in
            // full right here (measured, msst_then_rvc_probe) — a community log where the
            // "after" number barely moves is the direct diagnosis of a driver that defers
            // or withholds DML pool release (the prime suspect for the 16 GB crashes).
            let c0 = process_commit_mb();
            drop(removed); // teardown outside the sessions lock
            tracing::info!(
                "Released {} other cached ONNX session(s) before loading {} (GPU memory headroom; commit {}→{} MB)",
                n, keep_path.display(), c0, process_commit_mb()
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
        let removed = {
            let paths = self.paths.read();
            self.take_sessions_where(|key, _| {
                let Some(spec) = paths.get(key) else { return false };
                if matches!(spec.device, Some(DeviceConfig::Cpu)) {
                    return false; // forced-CPU sessions hold no VRAM
                }
                !keep.iter().any(|k| spec.path.starts_with(k))
            })
        };
        let n = removed.len();
        if n > 0 {
            // Same commit-return trace as release_others — this is the hop right before the
            // voice model pays its first-shape ticket, exactly where the community machines die.
            let c0 = process_commit_mb();
            drop(removed); // teardown outside the sessions lock
            tracing::info!(
                "Released {} GPU session(s) not needed by this voice run (VRAM headroom; commit {}→{} MB)",
                n, c0, process_commit_mb()
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
        let removed: Vec<LoadedSession> = {
            let mut sessions = self.sessions.write();
            keys.iter().filter_map(|k| sessions.remove(k)).collect()
        };
        {
            let mut paths = self.paths.write();
            for k in &keys {
                paths.remove(k);
            }
        }
        drop(removed); // teardown outside the sessions lock
        tracing::info!(
            "Unloaded {} session spec(s) under {} (files replaced)",
            keys.len(),
            prefix.display()
        );
        keys.len()
    }

    /// Drop all cached sessions (frees their memory). Safe to call between runs.
    pub fn clear_sessions(&self) {
        let taken = std::mem::take(&mut *self.sessions.write());
        let n = taken.len();
        drop(taken); // teardown outside the sessions lock
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
        let taken = {
            let mut sessions = self.sessions.write();
            if self.in_flight.load(Ordering::Relaxed) > 0
                || self.last_activity.lock().elapsed() < idle
            {
                return 0;
            }
            std::mem::take(&mut *sessions) // keep `paths` so run() can reload-on-miss
        };
        let n = taken.len();
        if n == 0 {
            return 0;
        }
        drop(taken); // teardown outside the sessions lock
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
        // S67b: input-shape signature for the DML shape-cache accounting (names + dims; the
        // pipelines build their input lists in a fixed order, so the hash is stable per shape).
        let shape_sig = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            for (name, t) in &inputs {
                name.hash(&mut h);
                t.shape().hash(&mut h);
            }
            h.finish()
        };
        // S67b: recycle an over-budget DML session BEFORE it compiles yet another shape
        // (see DmlShapeAccounting). Dropping it here lets the reload-on-miss below rebuild
        // it fresh — same path, same device, no cross-run state, ~1 s.
        self.maybe_recycle_dml_session(session_id, Some(shape_sig));
        // Reload-on-miss: the shared LRU cache may have evicted a session a consumer still holds.
        // Rebuild it from the remembered path (same key, same device) instead of hard-failing.
        self.reload_if_missing(session_id)?;
        let mut sessions = self.sessions.read();
        if !sessions.contains_key(session_id) {
            // A concurrent DML recycle (another thread's post-run check on this session id)
            // can remove the entry between the reload check above and this read (S67b
            // review). One in-place rebuild resolves it — and can't re-race: a freshly
            // rebuilt session has pool_growth 0, which no concurrent check can recycle.
            drop(sessions);
            self.reload_if_missing(session_id)?;
            sessions = self.sessions.read();
        }
        let loaded = sessions.get(session_id).ok_or_else(|| {
            UtaiError::Inference(format!("Session '{}' not found", session_id))
        })?;

        loaded
            .last_used
            .store(self.clock.fetch_add(1, Ordering::Relaxed), Ordering::Relaxed);

        let mut session = loaded.session.lock();
        // S67b: delta accounting for a NEW shape — measured around the run under the session
        // Mutex (concurrent same-sig runs serialize here, so a shape is measured once). The
        // sig is only recorded on SUCCESS: a failed compile must stay "new" so its retry
        // passes the pre-run budget check (see DmlShapeAccounting.shapes).
        let dml_new_shape: Option<u64> = match &loaded.dml {
            Some(dml) if !dml.shapes.lock().contains(&shape_sig) => {
                // S67c: a new DML shape is about to pay a pool/compile allocation — worst
                // case the multi-GB first-shape ticket, charged blind inside session.run.
                // On a machine at the commit limit the OS kills the process with zero
                // diagnostics; refuse LOUDLY below a conservative floor instead, and leave
                // a commit/headroom trace either way (the next community log then shows
                // the whole march toward exhaustion — instrument, don't theorize).
                let commit = process_commit_mb();
                let (_, avail) = system_memory_mb();
                let min = dml_min_avail_commit_mb();
                tracing::debug!(
                    "DML new shape for {}: process commit={} MB, system available commit={} MB",
                    loaded.path.display(), commit, avail
                );
                if min > 0 && avail > 0 && avail < min {
                    return Err(UtaiError::Inference(format!(
                        "INFERENCE_LOW_MEMORY: system available commit {avail} MB is below the \
                         {min} MB floor needed to compile a new GPU shape for {}",
                        loaded.path.display()
                    )));
                }
                Some(commit)
            }
            _ => None,
        };

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

        // Run + output extraction live in an immediately-called closure: NLL (Problem Case
        // #3) otherwise keeps `session` borrowed through BOTH match arms of run's Result,
        // and the failed-run arm below needs the guards released.
        let failed_dml_build = loaded.dml.as_ref().map(|_| loaded.build_id);
        let run_outcome: Result<Vec<OutputTensor>> = (|| {
            let outputs = session
                .run(input_values)
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
        })();

        let result = match run_outcome {
            Ok(result) => result,
            Err(e) => {
                // S67b (review round 2): a FAILED run on a DirectML session leaves unknown
                // allocator/device state — a half-committed pool from a failed compile, or
                // a wedged device (DXGI device-removed kills the session permanently).
                // Growth bookkeeping can never rescue this (every successful run restores
                // growth <= cap, so the retry's pre-run budget check structurally never
                // fires) — instead drop THIS session incarnation outright so the retry
                // rebuilds fresh with the session's whole footprint (ticket + pools)
                // returned as headroom. Non-DML sessions keep the plain error path.
                drop(session);
                drop(sessions);
                if let Some(build_id) = failed_dml_build {
                    if self.drop_session_incarnation(session_id, build_id) {
                        tracing::info!(
                            "Dropped DirectML session after a failed run (fresh rebuild on retry): {}",
                            session_id
                        );
                    }
                }
                return Err(e);
            }
        };

        // S67b: the run succeeded — record the shape and attribute its pool delta (first
        // shape excluded: it's the unavoidable ticket, not reclaimable growth).
        if let (Some(dml), Some(commit_before)) = (&loaded.dml, dml_new_shape) {
            let mut shapes = dml.shapes.lock();
            shapes.insert(shape_sig);
            if shapes.len() > 1 {
                let delta = process_commit_mb().saturating_sub(commit_before);
                dml.pool_growth_mb.fetch_add(delta, Ordering::Relaxed);
            }
        }

        drop(session);
        drop(sessions);
        // S67b post-run: reclaim a just-blown budget promptly so multi-GB pools don't linger
        // across the rest of the job (see DmlShapeAccounting).
        self.maybe_recycle_dml_session(session_id, None);
        Ok(result)
    }

    /// Reload-on-miss body shared by `run_typed`'s two lookup attempts: rebuild an evicted /
    /// recycled session from its remembered spec — or fail clearly when a mid-job device
    /// switch orphaned the key (load_model would rebuild under the CURRENT device's key,
    /// never this one; forced-device sessions — aux on CPU — are immune by construction).
    fn reload_if_missing(&self, session_id: &str) -> Result<()> {
        if self.sessions.read().contains_key(session_id) {
            return Ok(());
        }
        let spec = self.paths.read().get(session_id).cloned();
        if let Some(spec) = spec {
            let effective = spec.device.clone().unwrap_or_else(|| self.device.read().clone());
            if Self::make_key(&spec.path, &effective) != session_id {
                return Err(UtaiError::Inference(
                    "Device preference changed while a job was running — re-run the job".to_string(),
                ));
            }
            tracing::info!("ONNX session evicted — reloading: {}", spec.path.display());
            self.load_model_opts(&spec.path, spec.mem_pattern, spec.device)?;
        }
        Ok(())
    }

    /// S67b: DML shape-cache budget enforcement (rationale on [`DmlShapeAccounting`]).
    /// Called with `new_sig = Some(sig)` BEFORE a run — recycles only when `sig` is a shape
    /// this session has NOT compiled yet (known shapes add no memory, so same-shape loops
    /// like the diffusion denoiser never trip it) — and with `new_sig = None` AFTER a run
    /// (recycles immediately when the just-compiled shape blew the budget so multi-GB pools
    /// don't linger). The budget compares the session's OWN attributed pool growth
    /// (`pool_growth_mb`, Σ new-shape run deltas minus the first shape) against
    /// [`dml_extra_commit_cap_mb`]. Recycling only drops the session from the cache: the
    /// reload spec survives, so the next `run()` rebuilds it fresh via reload-on-miss
    /// (~1 s; no cross-run state, numerics unaffected).
    fn maybe_recycle_dml_session(&self, session_id: &str, new_sig: Option<u64>) {
        let (build_id, n_shapes, growth_mb, cap_mb, path) = {
            let sessions = self.sessions.read();
            let Some(loaded) = sessions.get(session_id) else { return };
            let Some(dml) = &loaded.dml else { return };
            if let Some(sig) = new_sig {
                if dml.shapes.lock().contains(&sig) {
                    return; // already-compiled shape — running it again costs nothing
                }
            }
            let growth = dml.pool_growth_mb.load(Ordering::Relaxed);
            let cap = dml_extra_commit_cap_mb();
            if growth <= cap {
                return;
            }
            let facts = (
                loaded.build_id,
                dml.shapes.lock().len(),
                growth,
                cap,
                loaded.path.display().to_string(),
            );
            facts
        };
        if self.drop_session_incarnation(session_id, build_id) {
            tracing::info!(
                "Recycled DirectML session: {} distinct input shape(s) grew ~{} MB of pools (budget {} MB) — {} rebuilds on next use to reclaim it",
                n_shapes,
                growth_mb,
                cap_mb,
                path
            );
        }
    }

    /// Remove ONE session incarnation from the cache, revalidated by `build_id` under the
    /// write lock — a concurrent thread may have recycled AND rebuilt the entry since the
    /// caller decided, and a stale decision must never tear down a fresh session (S67b
    /// review, TOCTOU). The removed entry (an ort Session owning potentially multi-GB DML
    /// pools) is dropped AFTER the write guard releases: a GB-scale D3D12 teardown under
    /// the global sessions lock would stall every concurrent inference (S67b review).
    /// Returns whether this incarnation was actually removed. Callers must hold NO engine
    /// locks. The reload spec in `paths` survives, so the next `run()` rebuilds on demand.
    fn drop_session_incarnation(&self, session_id: &str, build_id: u64) -> bool {
        let removed = {
            let mut sessions = self.sessions.write();
            match sessions.get(session_id) {
                Some(l) if l.build_id == build_id => sessions.remove(session_id),
                _ => None,
            }
        };
        removed.is_some() // `removed` drops here, outside the lock
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
/// S66: user-configurable CUDA arena cap in MB (Settings → CUDA runtime; 0 = unlimited =
/// the shipped default = pre-S66 behavior byte-for-byte). Set from config at startup and by
/// set_cuda_mem_limit (which also evicts GPU sessions so rebuilds pick the new value up).
/// OPT-IN by design: a hard cap turns "slow" into "allocation failed" on undersized values —
/// the Settings note says so — and per the S23 HARD RULE the option is A/B-verified against
/// the real MSST model before each release that touches it.
pub static CUDA_MEM_LIMIT_MB: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn cuda_ep(device_id: Option<i32>) -> ort::ep::CUDA {
    // SameAsRequested: grow the CUDA arena only by what each allocation needs, instead of the
    // default power-of-two doubling. Measured (S31 perf investigation): with the default strategy
    // the arena for the BSRoformer T=801 attention workload balloons to 11.9 of 12 GiB, WDDM
    // starts paging VRAM over PCIe, and per-chunk inference intermittently drops from ~890 ms to
    // 2.6-3.7 s (bit-identical outputs — pure residency thrash; the user-facing "5-minute song").
    // Numerics are unaffected by allocator strategy. gpu_mem_limit stays OFF unless the user
    // explicitly sets one (CUDA_MEM_LIMIT_MB above): a hard cap turns "slow" into "allocation
    // failed" on smaller cards. (S23 HARD RULE: any new EP option must be A/B-run on the real
    // model against this exact ort/ORT pair before shipping — done for SameAsRequested; the
    // user-set limit is A/B-verified per release.)
    let mut ep = ort::ep::CUDA::default()
        .with_tf32(true)
        .with_arena_extend_strategy(ort::ep::ArenaExtendStrategy::SameAsRequested);
    let limit_mb = CUDA_MEM_LIMIT_MB.load(std::sync::atomic::Ordering::Relaxed);
    if limit_mb > 0 {
        ep = ep.with_memory_limit((limit_mb as usize).saturating_mul(1024 * 1024));
    }
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
