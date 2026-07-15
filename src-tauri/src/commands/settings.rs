use std::sync::Arc;
use tauri::{Emitter, State};

use crate::inference::engine::DeviceConfig;
use crate::AppState;

#[derive(serde::Serialize)]
pub struct HardwareInfo {
    pub gpu_name: String,
    pub cuda_available: bool,
    pub directml_available: bool,
    pub current_device: String,
    /// Per-adapter vendor classification (S42, for runtime-pack recommendation).
    /// Vendor comes from PNPDeviceID's VEN_xxxx — NEVER from WMI AdapterRAM (a lying
    /// uint32: this dev box reports the 3080 Ti as 4 GB) and never from name heuristics.
    pub gpus: Vec<GpuAdapter>,
    /// Which runtime-pack variant this machine should default to
    /// ("nv-cu130" | "amd" | "xpu" | "cpu") — the user can always override.
    pub recommended_variant: String,
    /// Largest NVIDIA card's total VRAM in MB (nvidia-smi truth — NOT the lying WMI
    /// AdapterRAM), None = undetermined / no NVIDIA. Feeds the GPU-特征提取 gate (S66).
    pub nvidia_vram_mb: Option<u64>,
    /// The TRAINING device dropdown's list — values live in the accelerator's OWN
    /// namespace (NVIDIA UUID / vendor-relative index), never a WMI position. Empty
    /// = no trainable GPU on this box (the UI forces CPU). See training_gpu_list.
    pub training_gpus: Vec<TrainingGpu>,
}

#[derive(serde::Serialize, Clone)]
pub struct GpuAdapter {
    pub name: String,
    /// "nvidia" | "amd" | "intel" | "other"
    pub vendor: String,
}

/// One trainable GPU as the training-device dropdown offers it. `value` is what
/// run.json "gpu" carries into device.py's visibility env var.
#[derive(serde::Serialize, Clone)]
pub struct TrainingGpu {
    pub label: String,
    /// NVIDIA: the nvidia-smi UUID ("GPU-…" — CUDA_VISIBLE_DEVICES accepts it, exact
    /// identity, immune to enumeration-order drift). Fallbacks: vendor-relative index.
    pub value: String,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct AppConfig {
    #[serde(default)]
    pub device: DeviceConfig,
    /// User-set data root for the BIG growable files (models + cache). Empty/None → app_dir/data (next
    /// to the program, NOT C: AppData — those files reach tens of GB). See `resolve_data_dir`.
    #[serde(default)]
    pub data_dir: Option<String>,
    /// S66: user-set CUDA arena cap in MB (0 = unlimited = default). Shown only in the
    /// Settings CUDA section when a CUDA runtime is installed (user decision: the control
    /// is visible ⟺ it is effective; DirectML has no equivalent API).
    #[serde(default)]
    pub cuda_mem_limit_mb: u64,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            device: DeviceConfig::Auto,
            data_dir: None,
            cuda_mem_limit_mb: 0,
        }
    }
}

#[tauri::command]
pub fn get_hardware_info(state: State<'_, Arc<AppState>>) -> Result<HardwareInfo, String> {
    let current = state.inference.engine.device();
    let current_str = match &current {
        DeviceConfig::Cpu => "cpu".to_string(),
        DeviceConfig::DirectMl { .. } => "directml".to_string(),
        DeviceConfig::Cuda { .. } => "cuda".to_string(),
        DeviceConfig::Auto => "auto".to_string(),
    };

    let gpus = query_gpu_adapters();
    let gpu_name = if gpus.is_empty() {
        "Unknown GPU".to_string()
    } else {
        gpus.iter().map(|g| g.name.as_str()).collect::<Vec<_>>().join(", ")
    };
    Ok(HardwareInfo {
        gpu_name,
        // Vendor-guarded (S64c audit): the self-downloaded runtime/cuda DLLs satisfy the PATH probe
        // even on a box whose NVIDIA card is gone (migrated data dir) — the badge must track the GPU.
        cuda_available: gpus.iter().any(|g| g.vendor == "nvidia") && is_cuda_available(),
        directml_available: cfg!(windows),
        current_device: current_str,
        recommended_variant: recommend_variant(&gpus).to_string(),
        nvidia_vram_mb: if gpus.iter().any(|g| g.vendor == "nvidia") {
            nvidia_total_vram_mb()
        } else {
            None
        },
        training_gpus: training_gpu_list(&gpus),
        gpus,
    })
}

/// GPU list for the TRAINING device dropdown — values in the ACCELERATOR'S OWN ordinal
/// space, not WMI's. S67 (community bug): the dropdown used to store the raw
/// Win32_VideoController index, which device.py fed to CUDA_VISIBLE_DEVICES verbatim; on
/// an iGPU+NVIDIA box the NVIDIA card sits at WMI index 1 but CUDA ordinal 0, so
/// SELECTING the correct card masked every GPU and torch silently trained on CPU.
/// NVIDIA boxes get nvidia-smi UUIDs (exact identity); the fallbacks (nvidia-smi absent,
/// AMD/HIP, Intel/ZE_AFFINITY_MASK) keep vendor-relative indices — exact for the
/// dominant single-card case, and the sidecar's require_wanted_accelerator guard turns
/// any remaining mismatch into a loud TRAIN_GPU_UNAVAILABLE instead of silent CPU.
fn training_gpu_list(gpus: &[GpuAdapter]) -> Vec<TrainingGpu> {
    let vendor_indexed = |vendor: &str| -> Vec<TrainingGpu> {
        gpus.iter()
            .filter(|g| g.vendor == vendor)
            .enumerate()
            .map(|(i, g)| TrainingGpu { label: g.name.clone(), value: i.to_string() })
            .collect()
    };
    if gpus.iter().any(|g| g.vendor == "nvidia") {
        let smi = nvidia_gpu_uuids();
        if !smi.is_empty() {
            return smi;
        }
        return vendor_indexed("nvidia");
    }
    if gpus.iter().any(|g| g.vendor == "amd") {
        return vendor_indexed("amd");
    }
    if gpus.iter().any(|g| g.vendor == "intel") {
        return vendor_indexed("intel");
    }
    Vec::new()
}

/// NVIDIA cards as (name, UUID) via nvidia-smi — the only enumeration whose identity
/// CUDA itself understands. Empty on any failure (no smi / no driver): callers fall
/// back to vendor-relative indices.
#[cfg(windows)]
fn nvidia_gpu_uuids() -> Vec<TrainingGpu> {
    use std::os::windows::process::CommandExt;
    let out = match std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,uuid", "--format=csv,noheader"])
        .creation_flags(crate::util::CREATE_NO_WINDOW)
        .output()
    {
        Ok(o) if o.status.success() => o,
        _ => return Vec::new(),
    };
    let text = String::from_utf8_lossy(&out.stdout);
    text.lines()
        .filter_map(|l| {
            // "NVIDIA GeForce RTX 3080 Ti, GPU-8a2c…" — rsplit so a comma INSIDE the
            // name can't shear the row; a non-UUID tail drops the row instead of
            // feeding CUDA a garbage mask
            let (name, uuid) = l.rsplit_once(',')?;
            let uuid = uuid.trim();
            if !uuid.starts_with("GPU-") {
                return None;
            }
            Some(TrainingGpu { label: name.trim().to_string(), value: uuid.to_string() })
        })
        .collect()
}

#[cfg(not(windows))]
fn nvidia_gpu_uuids() -> Vec<TrainingGpu> {
    Vec::new()
}

/// Default runtime-pack variant for this machine. NVIDIA wins over everything (the
/// only fully-supported training path); AMD over Intel. iGPU-vs-dGPU is deliberately
/// NOT guessed — the pick is only a DEFAULT and the UI lets the user override
/// (Pinokio's silent wrong-variant installs are the anti-pattern we're avoiding).
fn recommend_variant(gpus: &[GpuAdapter]) -> &'static str {
    if gpus.iter().any(|g| g.vendor == "nvidia") {
        "nv-cu130"
    } else if gpus.iter().any(|g| g.vendor == "amd") {
        "amd"
    } else if gpus.iter().any(|g| g.vendor == "intel") {
        "xpu"
    } else {
        "cpu"
    }
}

/// Enumerate video adapters with PCI vendor ids via WMI. One query serves both the
/// display string and the vendor classification (single source — replaces the old
/// name-only `detect_gpu_name`).
pub(crate) fn query_gpu_adapters() -> Vec<GpuAdapter> {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let output = std::process::Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                "Get-CimInstance -ClassName Win32_VideoController | Select-Object Name, PNPDeviceID | ConvertTo-Json -Compress",
            ])
            .creation_flags(crate::util::CREATE_NO_WINDOW)
            .output();
        if let Ok(out) = output {
            let text = String::from_utf8_lossy(&out.stdout);
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(text.trim()) {
                // ConvertTo-Json yields an OBJECT for one adapter, an ARRAY for several.
                let items: Vec<&serde_json::Value> = match &val {
                    serde_json::Value::Array(a) => a.iter().collect(),
                    other => vec![other],
                };
                let adapters: Vec<GpuAdapter> = items
                    .into_iter()
                    .filter_map(|item| {
                        let name = item.get("Name")?.as_str()?.trim().to_string();
                        let pnp = item.get("PNPDeviceID").and_then(|v| v.as_str()).unwrap_or("");
                        let vendor = if pnp.contains("VEN_10DE") {
                            "nvidia"
                        } else if pnp.contains("VEN_1002") {
                            "amd"
                        } else if pnp.contains("VEN_8086") {
                            "intel"
                        } else {
                            "other"
                        };
                        Some(GpuAdapter { name, vendor: vendor.to_string() })
                    })
                    .collect();
                if !adapters.is_empty() {
                    return adapters;
                }
            }
        }
    }
    Vec::new()
}

/// Max NVIDIA compute capability across installed NVIDIA GPUs, via nvidia-smi
/// (authoritative — WMI/PNPDeviceID can't report it). `None` when nvidia-smi is
/// absent or unreadable (no driver, or a non-NVIDIA box): callers treat `None` as
/// "undetermined → do not architecture-gate" (fail open, envtest is the real gate).
#[cfg(windows)]
pub(crate) fn nvidia_max_compute_cap() -> Option<f32> {
    use std::os::windows::process::CommandExt;
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .creation_flags(crate::util::CREATE_NO_WINDOW)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let max = text
        .lines()
        .filter_map(|l| l.trim().parse::<f32>().ok())
        .fold(f32::NAN, f32::max); // f32::max ignores NaN → seeds cleanly, stays NaN if no rows
    if max.is_nan() {
        None
    } else {
        Some(max)
    }
}

#[cfg(not(windows))]
pub(crate) fn nvidia_max_compute_cap() -> Option<f32> {
    None
}

/// Total VRAM of the largest NVIDIA card in MB (nvidia-smi memory.total), None = undetermined
/// (no nvidia-smi / no NVIDIA card). S66: feeds the GPU-特征提取 gate — the feature's measured
/// steady peak is ~9.4 GB (user, two runs), so cards under 12 GB can't enable it. Undetermined
/// fails OPEN (the variant_supported convention: never hide a capability on a probe failure).
#[cfg(windows)]
pub(crate) fn nvidia_total_vram_mb() -> Option<u64> {
    use std::os::windows::process::CommandExt;
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .creation_flags(crate::util::CREATE_NO_WINDOW)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    text.lines().filter_map(|l| l.trim().parse::<u64>().ok()).max()
}

#[cfg(not(windows))]
pub(crate) fn nvidia_total_vram_mb() -> Option<u64> {
    None
}

/// Whether THIS machine's hardware can run a given runtime-pack VARIANT — the gate for
/// which download entries the settings UI offers (only expose packs the user can actually
/// use; a fresh box always sees CPU). Vendor comes from PNPDeviceID. The NVIDIA pack
/// ADDITIONALLY needs an sm_75+ card (compute cap ≥ 7.5): torch cu130's fatbin floor is
/// sm_75, so a GTX 10-series / Pascal card can't run it and must not be offered the pack.
/// AMD/Intel are gated on vendor presence ONLY (experimental tier — the on-device envtest
/// is the true capability gate, and robust RDNA/Arc arch detection needs tooling we don't
/// bundle). An UNDETERMINED NVIDIA compute cap (nvidia-smi absent) fails OPEN so a valid
/// RTX user is never hidden. NB: LOCAL-FILE install is deliberately NOT gated by this.
pub(crate) fn variant_supported(variant: &str, gpus: &[GpuAdapter], nv_cc: Option<f32>) -> bool {
    let has = |v: &str| gpus.iter().any(|g| g.vendor == v);
    match variant {
        "cpu" => true,
        "nv-cu130" => has("nvidia") && nv_cc.map_or(true, |cc| cc >= 7.5),
        "amd" => has("amd"),
        "xpu" => has("intel"),
        _ => false,
    }
}

#[tauri::command]
pub fn set_device_preference(
    state: State<'_, Arc<AppState>>,
    device: String,
) -> Result<(), String> {
    let config = match device.as_str() {
        "cuda" => DeviceConfig::Cuda { device_id: 0 },
        "directml" => DeviceConfig::DirectMl { device_id: 0 },
        "cpu" => DeviceConfig::Cpu,
        _ => DeviceConfig::Auto,
    };

    state.inference.engine.set_device(config.clone());

    // Persist — load-then-update so we never clobber the rest of the config (esp. data_dir).
    let mut cfg = load_config(&state.app_dir).unwrap_or_default();
    cfg.device = config;
    if let Err(e) = save_config(&state.app_dir, &cfg) {
        tracing::warn!("Failed to save config: {}", e);
    }

    Ok(())
}

#[tauri::command]
pub fn get_cuda_mem_limit(state: State<'_, Arc<AppState>>) -> u64 {
    let _ = &state; // config is the source of truth, but the live static is what sessions read
    crate::inference::engine::CUDA_MEM_LIMIT_MB.load(std::sync::atomic::Ordering::Relaxed)
}

/// S66: set the CUDA arena cap (MB; 0 = unlimited). Applies to sessions built from now on —
/// live GPU sessions are evicted so the next run rebuilds them under the new cap (reload-on-
/// miss restores them transparently). Persisted in config.json.
#[tauri::command]
pub fn set_cuda_mem_limit(state: State<'_, Arc<AppState>>, mb: u64) -> Result<(), String> {
    crate::inference::engine::CUDA_MEM_LIMIT_MB.store(mb, std::sync::atomic::Ordering::Relaxed);
    state.inference.engine.release_gpu_sessions_except(&[]);
    let mut cfg = load_config(&state.app_dir).unwrap_or_default();
    cfg.cuda_mem_limit_mb = mb;
    if let Err(e) = save_config(&state.app_dir, &cfg) {
        tracing::warn!("Failed to save config: {}", e);
    }
    tracing::info!(
        "CUDA memory limit set to {} (GPU sessions evicted; rebuilt under the new cap on next use)",
        if mb == 0 { "unlimited".to_string() } else { format!("{mb} MB") }
    );
    Ok(())
}

#[tauri::command]
pub fn get_device_preference(state: State<'_, Arc<AppState>>) -> Result<String, String> {
    let current = state.inference.engine.device();
    Ok(match current {
        DeviceConfig::Cpu => "cpu".to_string(),
        DeviceConfig::DirectMl { .. } => "directml".to_string(),
        DeviceConfig::Cuda { .. } => "cuda".to_string(),
        DeviceConfig::Auto => "auto".to_string(),
    })
}

pub fn load_and_apply_config(state: &AppState) {
    // Logging rules (S22 + S42): state FACTS, not the fallback chain — which ORT
    // build this process committed is already known (ORT_LOADED_BUILD), and the
    // per-inference "ONNX device=..." lines remain the truth source for what each
    // run executes on. Logs are English/standard format (Chinese belongs to the
    // user-facing error strings, not tracing). NB: an absent config.json MEANS the
    // preference IS Auto (the default is simply never written to disk) — the old
    // wording ("No config found") read like breakage and was mistaken for a CUDA
    // regression in the field.
    let build = crate::ORT_LOADED_BUILD.get().map(|s| s.as_str()).unwrap_or("?");
    if let Some(cfg) = load_config(&state.app_dir) {
        tracing::info!(
            "device preference: {:?} (config.json); ORT build loaded: {}; per-run EP is logged as \"ONNX device=...\"",
            cfg.device,
            build
        );
        state.inference.engine.set_device(cfg.device);
        if cfg.cuda_mem_limit_mb > 0 {
            crate::inference::engine::CUDA_MEM_LIMIT_MB
                .store(cfg.cuda_mem_limit_mb, std::sync::atomic::Ordering::Relaxed);
            tracing::info!("CUDA memory limit: {} MB (config.json)", cfg.cuda_mem_limit_mb);
        }
    } else {
        tracing::info!(
            "device preference: Auto (default; config.json is only written once changed in Settings); ORT build loaded: {}; per-run EP is logged as \"ONNX device=...\"",
            build
        );
    }
}

fn config_path(app_dir: &std::path::Path) -> std::path::PathBuf {
    app_dir.join("config.json")
}

fn save_config(app_dir: &std::path::Path, cfg: &AppConfig) -> std::io::Result<()> {
    let path = config_path(app_dir);
    let json = serde_json::to_string_pretty(cfg).unwrap_or_default();
    // Temp + rename so a crash mid-write can't truncate config.json (losing device pref + data_dir).
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, json)?;
    std::fs::rename(&tmp, &path)
}

fn load_config(app_dir: &std::path::Path) -> Option<AppConfig> {
    let path = config_path(app_dir);
    let content = std::fs::read_to_string(path).ok()?;
    match serde_json::from_str(&content) {
        Ok(cfg) => Some(cfg),
        Err(e) => {
            // A corrupt config silently falling back to defaults would look like lost settings.
            tracing::warn!("config.json exists but failed to parse ({}); using defaults", e);
            None
        }
    }
}

/// S64 portability: the data-dir override in config.json is an ABSOLUTE user-chosen path (the one
/// sanctioned absolute reference) — when its target vanishes (drive unplugged, dir deleted, install
/// copied to another machine) the old behavior was a SILENT empty library (models/dictionaries/
/// runtimes all "gone", zero warnings). This records what happened for the settings UI + a startup
/// toast; set at most once, at startup resolution.
#[derive(serde::Serialize, Clone)]
pub struct DataDirIssue {
    /// The configured (missing) override path.
    pub configured: String,
    /// The directory actually used this session.
    pub effective: String,
    /// true = override unusable (drive gone) → default next to the program; false = recreated empty.
    pub fell_back: bool,
}

pub static DATA_DIR_ISSUE: std::sync::OnceLock<DataDirIssue> = std::sync::OnceLock::new();

/// Startup warning for the frontend (null = the data dir resolved normally).
#[tauri::command]
pub fn get_data_dir_issue() -> Option<DataDirIssue> {
    DATA_DIR_ISSUE.get().cloned()
}

/// Data root for the big growable files (models + cache). User-set in config.json's `data_dir`; else
/// `app_dir/data` — NEXT TO THE PROGRAM, never C: AppData (those files reach tens of GB). Derived at
/// startup; changing it takes effect on restart. A configured-but-missing override is recreated on
/// its drive when possible (user intent wins), else falls back to the default — either way LOUDLY
/// (DATA_DIR_ISSUE), never a silent empty library.
pub fn resolve_data_dir(app_dir: &std::path::Path) -> std::path::PathBuf {
    if let Some(cfg) = load_config(app_dir) {
        if let Some(d) = cfg.data_dir {
            let d = d.trim();
            if !d.is_empty() {
                let p = std::path::PathBuf::from(d);
                if p.is_dir() {
                    return p;
                }
                if std::fs::create_dir_all(&p).is_ok() {
                    tracing::warn!("configured data_dir {} was missing — recreated (empty)", d);
                    let _ = DATA_DIR_ISSUE.set(DataDirIssue {
                        configured: d.to_string(),
                        effective: p.to_string_lossy().to_string(),
                        fell_back: false,
                    });
                    return p;
                }
                let fallback = app_dir.join("data");
                tracing::warn!(
                    "configured data_dir {} is unavailable — falling back to {}",
                    d,
                    fallback.display()
                );
                let _ = DATA_DIR_ISSUE.set(DataDirIssue {
                    configured: d.to_string(),
                    effective: fallback.to_string_lossy().to_string(),
                    fell_back: true,
                });
                return fallback;
            }
        }
    }
    app_dir.join("data")
}

/// The data root ACTUALLY in use this session — parent of cache_dir (cache_dir = data_root/cache,
/// models = data_root/models). May differ from `resolve_data_dir`: startup can pick the legacy
/// AppData fallback for upgraders (see lib.rs setup).
fn effective_data_root(state: &AppState) -> &std::path::Path {
    state.cache_dir.parent().unwrap_or(state.cache_dir.as_path())
}

/// Current data dir (for the settings UI).
#[tauri::command]
pub fn get_data_dir(state: State<'_, Arc<AppState>>) -> Result<String, String> {
    Ok(effective_data_root(&state).to_string_lossy().to_string())
}

/// Recursively copy a directory's contents into `dst` (creating it). Cross-drive safe (copy, not rename).
fn copy_dir_all(src: &std::path::Path, dst: &std::path::Path) -> std::io::Result<()> {
    if !src.exists() {
        return Ok(());
    }
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if from.is_dir() {
            copy_dir_all(&from, &to)?;
        } else {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

/// One-click migrate: copy the CURRENT models + cache into `new_dir`, then persist it as the data dir.
/// Takes effect on restart. Old data is LEFT in place (rollback = revert the setting / don't restart);
/// the user deletes the old copy manually once the new location is confirmed working.
#[tauri::command]
pub async fn migrate_data_dir(state: State<'_, Arc<AppState>>, new_dir: String) -> Result<(), String> {
    let new = std::path::PathBuf::from(new_dir.trim());
    if new.as_os_str().is_empty() {
        return Err("Empty target directory".into());
    }
    // S61: a live training run writes checkpoints/features mid-copy — the migrated tree would be
    // torn (and the workspace copy below is exactly what a running trainer mutates).
    if state.training.is_active() {
        return Err("TRAINING_ACTIVE".into());
    }
    let data_root = effective_data_root(&state).to_path_buf();
    let target = new.clone();
    // The copy reaches tens of GB — run it off the event loop so the UI stays responsive.
    tauri::async_runtime::spawn_blocking(move || -> Result<(), String> {
        std::fs::create_dir_all(&target).map_err(|e| format!("Create target: {e}"))?;
        // Refuse a target nested inside the data root (or vice versa) — copying a tree into itself
        // recurses forever.
        let canon_target = std::fs::canonicalize(&target).map_err(|e| format!("Resolve target: {e}"))?;
        let canon_root = std::fs::canonicalize(&data_root).unwrap_or_else(|_| data_root.clone());
        if canon_target.starts_with(&canon_root) || canon_root.starts_with(&canon_target) {
            return Err("Target directory overlaps the current data directory".into());
        }
        copy_dir_all(&data_root.join("models"), &target.join("models")).map_err(|e| format!("Copy models: {e}"))?;
        copy_dir_all(&data_root.join("cache"), &target.join("cache")).map_err(|e| format!("Copy cache: {e}"))?;
        // ② S58: the stage1 G2P dictionaries live under <data_root>/dictionaries — leaving them behind
        // would fake-OOV every zh/en/de/fr/es/it lyric after a migration (audit MAJOR).
        let dicts_src = data_root.join("dictionaries");
        if dicts_src.exists() {
            copy_dir_all(&dicts_src, &target.join("dictionaries")).map_err(|e| format!("Copy dictionaries: {e}"))?;
        }
        // Runtime packs (S42) live under <data_root>/runtimes and must MOVE WITH the
        // data dir — lib.rs roots pyenv on the resolved data dir, so leaving them
        // behind would make every installed pack "vanish" after migration (and strand
        // gigabytes on the old drive with no UI to reclaim them). `.staging` (torn
        // installs / resumable part files) is transient — skip it.
        let runtimes_src = data_root.join("runtimes");
        if runtimes_src.exists() {
            let runtimes_dst = target.join("runtimes");
            std::fs::create_dir_all(&runtimes_dst).map_err(|e| format!("Create runtimes: {e}"))?;
            for entry in std::fs::read_dir(&runtimes_src).map_err(|e| format!("Read runtimes: {e}"))?.flatten() {
                let name = entry.file_name();
                if name.to_string_lossy().starts_with('.') {
                    continue;
                }
                copy_dir_all(&entry.path(), &runtimes_dst.join(&name))
                    .map_err(|e| format!("Copy runtimes/{}: {e}", name.to_string_lossy()))?;
            }
        }
        // S61 (recon gap): training WORKSPACES live under <data_root>/training and resolve off the
        // SAME data dir (commands/training.rs data_root) — not copying them silently stranded every
        // checkpoint + dataset on the old drive while 续训/共享池 resolved against the NEW (empty)
        // tree after restart. GBs, but losing training progress is worse than a longer copy.
        copy_dir_all(&data_root.join("training"), &target.join("training"))
            .map_err(|e| format!("Copy training: {e}"))?;
        Ok(())
    })
    .await
    .map_err(|e| format!("Copy task failed: {e}"))??;
    let mut cfg = load_config(&state.app_dir).unwrap_or_default();
    cfg.data_dir = Some(new.to_string_lossy().to_string());
    save_config(&state.app_dir, &cfg).map_err(|e| format!("Save config: {e}"))?;
    tracing::info!("Migrated data dir → {} (restart to apply)", new.display());
    Ok(())
}

/// Whether CUDA is ACTUALLY usable, not just "files downloaded". Verifies that the CUDA ORT build is
/// present AND that the CUDA major it was built for (read from providers_cuda.dll's imports) matches a
/// cudart + cuDNN actually resolvable on this machine. This is what stops the old false "Ready" when a
/// CUDA-11-built ORT (1.21.x) sat on a CUDA-12 system — it now correctly reports NOT ready.
#[tauri::command]
pub fn is_cuda_runtime_ready(state: State<'_, Arc<AppState>>) -> Result<bool, String> {
    let cuda_dir = state.app_dir.join("runtime").join("ort").join("cuda");
    let ort_cuda_dll = cuda_dir.join("onnxruntime.dll");
    let providers = cuda_dir.join("onnxruntime_providers_cuda.dll");
    if !ort_cuda_dll.exists() || !providers.exists() {
        return Ok(false);
    }
    // Which CUDA major does this build actually need? (1.21.x wrongly needs 11 → unusable on a 12 box.)
    let major = cuda_build_major(&providers).unwrap_or(0);
    if major < 12 {
        return Ok(false); // CUDA 11 build (or unreadable) — treat as not ready
    }
    Ok(cuda_provider_deps_resolvable(&state.app_dir))
}

/// THE provider-dependency check (S64c): the FULL import set scanned from the 1.24.4
/// providers_cuda.dll, each resolvable from OUR runtime/cuda (self-contained download), PATH, or
/// CUDA_PATH (Toolkit users). Shared by is_cuda_runtime_ready AND lib.rs' Auto build pick — a
/// PARTIAL install must never flip Auto onto the CUDA build (it has no DirectML provider).
pub(crate) fn cuda_provider_deps_resolvable(app_dir: &std::path::Path) -> bool {
    const DEPS: [&str; 5] = [
        "cudart64_12.dll",
        "cublas64_12.dll",
        "cublasLt64_12.dll",
        "cufft64_11.dll",
        "cudnn64_9.dll",
    ];
    let cuda_dir = app_dir.join("runtime").join("cuda");
    DEPS.iter().all(|d| dll_on_path_or_dir(d, &cuda_dir))
}

/// Scan a providers_cuda.dll for its imported `cudart64_NNN.dll` string to learn the CUDA MAJOR it was
/// built against (110 → 11, 12 → 12, 118 → 11). Reads the whole DLL once; fine for an on-demand check.
fn cuda_build_major(providers_cuda: &std::path::Path) -> Option<u32> {
    use std::collections::HashMap;
    use std::sync::Mutex;
    // Cache keyed by (path, mtime, len) so repeated Settings opens don't re-read the DLL, while a
    // re-download replacing it in-session (do_download_cuda_runtime) is picked up without a restart.
    type CacheKey = (std::path::PathBuf, Option<std::time::SystemTime>, u64);
    static CACHE: Mutex<Option<HashMap<CacheKey, Option<u32>>>> = Mutex::new(None);
    let meta = std::fs::metadata(providers_cuda).ok();
    let key: CacheKey = (
        providers_cuda.to_path_buf(),
        meta.as_ref().and_then(|m| m.modified().ok()),
        meta.as_ref().map(|m| m.len()).unwrap_or(0),
    );
    if let Some(m) = CACHE.lock().unwrap().as_ref() {
        if let Some(v) = m.get(&key) {
            return *v;
        }
    }
    let result = scan_cuda_major(providers_cuda);
    CACHE
        .lock()
        .unwrap()
        .get_or_insert_with(HashMap::new)
        .insert(key, result);
    result
}

fn scan_cuda_major(providers_cuda: &std::path::Path) -> Option<u32> {
    use std::io::Read;
    // The "cudart64_NNN.dll" import string lives near the PE header, not in the hundreds-of-MB CUDA
    // kernel blob — read only the first 64MB instead of slurping the whole DLL into RAM.
    let mut data = Vec::new();
    std::fs::File::open(providers_cuda)
        .ok()?
        .take(64 * 1024 * 1024)
        .read_to_end(&mut data)
        .ok()?;
    let needle = b"cudart64_";
    let mut i = 0usize;
    while i + needle.len() + 1 < data.len() {
        if &data[i..i + needle.len()] == needle {
            let mut j = i + needle.len();
            let mut digits = String::new();
            while j < data.len() && data[j].is_ascii_digit() && digits.len() < 4 {
                digits.push(data[j] as char);
                j += 1;
            }
            if let Ok(n) = digits.parse::<u32>() {
                return Some(if n >= 100 { n / 10 } else { n });
            }
        }
        i += 1;
    }
    None
}

/// True if `name` is found on PATH or in the system CUDA Toolkit bin (CUDA_PATH may not be on PATH here).
fn dll_on_path(name: &str) -> bool {
    if let Ok(path) = std::env::var("PATH") {
        if std::env::split_paths(&path).any(|d| d.join(name).exists()) {
            return true;
        }
    }
    if let Ok(cuda) = std::env::var("CUDA_PATH") {
        if std::path::Path::new(&cuda).join("bin").join(name).exists() {
            return true;
        }
    }
    false
}

fn dll_on_path_or_dir(name: &str, extra: &std::path::Path) -> bool {
    extra.join(name).exists() || dll_on_path(name)
}

/// Remote mirror list (mirrors.json on the utai-runtimes HF dataset; hf-mirror twin).
/// Public GH proxies rot in 6-18 months — shipped builds refresh their preset list from
/// here (frontend caches it; builtin list is the offline fallback). Schema gate = `schema: 1`.
const MIRROR_LIST_URLS: [&str; 2] = [
    "https://huggingface.co/datasets/yasoukyoku/utai-runtimes/resolve/main/mirrors.json",
    "https://hf-mirror.com/datasets/yasoukyoku/utai-runtimes/resolve/main/mirrors.json",
];

#[tauri::command]
pub async fn fetch_mirror_list() -> Result<serde_json::Value, String> {
    let client = crate::download::client().map_err(|e| e.to_string())?;
    for url in MIRROR_LIST_URLS {
        let fut = client.get(url).send();
        match tokio::time::timeout(std::time::Duration::from_secs(8), fut).await {
            Ok(Ok(resp)) if resp.status().is_success() => {
                if let Ok(bytes) = resp.bytes().await {
                    if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                        if v.get("schema").and_then(|s| s.as_i64()) == Some(1) {
                            return Ok(v);
                        }
                        tracing::warn!("mirrors.json from {url}: unexpected schema — ignored");
                    }
                }
            }
            other => {
                if let Ok(Err(e)) = other {
                    tracing::debug!("mirrors.json fetch failed via {url}: {e}");
                }
            }
        }
    }
    Err("MIRROR_LIST_UNAVAILABLE".into())
}

/// Cooperative cancel for the in-flight CUDA runtime download (S66): the active download
/// stashes its cancel flag here; the command flips it. The unified engine keeps every
/// .part on cancel, so a resumed download loses nothing.
static CUDA_DL_CANCEL: parking_lot::Mutex<Option<Arc<std::sync::atomic::AtomicBool>>> =
    parking_lot::Mutex::new(None);

#[tauri::command]
pub fn cancel_cuda_download() {
    if let Some(flag) = CUDA_DL_CANCEL.lock().as_ref() {
        flag.store(true, std::sync::atomic::Ordering::SeqCst);
    }
}

/// Download CUDA ORT DLLs + cuDNN DLLs for CUDA EP support.
/// Emits `cuda-download-progress` events with {stage, progress, message}.
/// `prefer_cn_mirrors` (from the frontend HF-source choice) puts the Chinese PyPI/HF
/// mirrors ahead of the official hosts — mainland users time out on pythonhosted (S66).
#[tauri::command]
pub async fn download_cuda_runtime(
    app_handle: tauri::AppHandle,
    state: State<'_, Arc<AppState>>,
    prefer_cn_mirrors: Option<bool>,
) -> Result<(), String> {
    // S64c: the download is now fully self-contained (cudart/cublas/cufft/cudnn all fetched from
    // NVIDIA's official PyPI redistributables — no CUDA Toolkit needed, which beta testers proved
    // nobody has). The one hard requirement left is an NVIDIA GPU + its driver. FAIL-OPEN on an
    // EMPTY probe (WMI/PowerShell failure = undetermined, the variant_supported convention) —
    // refuse only on a POSITIVE non-NVIDIA determination.
    let gpus = query_gpu_adapters();
    if !gpus.is_empty() && !gpus.iter().any(|g| g.vendor == "nvidia") {
        return Err("CUDA_GPU_REQUIRED".to_string());
    }
    // Single-flight (S64c audit): begin_task is a refcount for the close-flow listing, not a mutex —
    // a remounted Settings panel re-enables the button mid-download, and a second click would run
    // two concurrent downloaders over the same files.
    if state.task_active("cuda_download") {
        return Err("CUDA_DOWNLOAD_BUSY".to_string());
    }
    let _task = state.begin_task("cuda_download"); // listed in the close-flow's in-progress warning
    let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
    *CUDA_DL_CANCEL.lock() = Some(cancel.clone());
    let app_dir = state.app_dir.clone();
    let handle = app_handle.clone();
    let prefer_cn = prefer_cn_mirrors.unwrap_or(false);

    let joined = tokio::task::spawn_blocking(move || {
        let rt = tokio::runtime::Handle::current();
        rt.block_on(async {
            do_download_cuda_runtime(&app_dir, &handle, prefer_cn, &cancel).await
        })
    })
    .await;
    // Clear the cancel slot BEFORE the join `?` (review S66: a JoinError early-return left a
    // stale Arc — a later cancel click would flip a dead flag while a fresh download ignores it).
    *CUDA_DL_CANCEL.lock() = None;
    let result = joined.map_err(|e| format!("Task failed: {}", e))?;

    // Surface the outcome into the tracing pipeline (log panel + file) — a failed download used to be
    // invisible there (only shown under the button), which is exactly what the user hit.
    match &result {
        Ok(()) => tracing::info!("CUDA runtime download complete"),
        Err(e) if e.to_string().contains("CANCELLED") => {
            tracing::info!("CUDA runtime download cancelled (resumable — every .part is kept)")
        }
        Err(e) => tracing::error!("CUDA runtime download failed: {}", e),
    }
    // Terminal event on failure/cancel too (review S66): the frontend clears its busy state on
    // invoke resolution, but a LATE buffered progress event would re-latch it with no terminal
    // event ever following — the panel wedged in a fake in-progress state.
    if let Err(e) = &result {
        let cancelled = e.to_string().contains("CANCELLED");
        let _ = app_handle.emit(
            "cuda-download-progress",
            serde_json::json!({
                "stage": "error", "progress": 0.0,
                "code": if cancelled { "CUDA_DL_CANCELLED" } else { "CUDA_DL_FAILED" },
                "label": "", "message": e.to_string(),
            }),
        );
    }
    result.map_err(|e| e.to_string())
}

// ── CUDA runtime sources (S66: unified engine + mainland-China mirrors + resume) ──

/// The ORT CUDA build. MIT-licensed, so it is legitimately mirrored on our HF dataset
/// (mainland reachability via hf-mirror); NuGet stays the canonical source. 1.24.4 MUST
/// match ort 2.0-rc.12 (API 24) AND the bundled DirectML build (see the Stage-1 note).
const ORT_GPU_NUPKG_URLS: [&str; 3] = [
    "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.Gpu.Windows/1.24.4",
    "https://huggingface.co/datasets/yasoukyoku/utai-runtimes/resolve/main/mirror/ort/Microsoft.ML.OnnxRuntime.Gpu.Windows.1.24.4.nupkg",
    "https://hf-mirror.com/datasets/yasoukyoku/utai-runtimes/resolve/main/mirror/ort/Microsoft.ML.OnnxRuntime.Gpu.Windows.1.24.4.nupkg",
];
const ORT_GPU_NUPKG_SHA256: &str = "e897a13d318483e71e1eef91005634846201ab50bc6a582ae913dc5a6ccc0240";
const ORT_GPU_NUPKG_SIZE: u64 = 172_417_405;

/// Full PyPI mirrors serving files.pythonhosted.org packages under the SAME content-addressed
/// `/packages/<h1>/<h2>/<hash>/<file>` path (bandersnatch layout) — live-verified to carry the
/// exact pinned NVIDIA wheels, including the 655 MB cuDNN one. Pure prefix swap; the NVIDIA
/// binaries themselves stay untouched (we never re-host them — EULA posture, S66 research).
const PYPI_MIRRORS: [&str; 3] = [
    "https://pypi.tuna.tsinghua.edu.cn",
    "https://mirrors.aliyun.com/pypi",
    "https://mirrors.cloud.tencent.com/pypi",
];

/// Candidate URL rotation for one pinned pythonhosted wheel. `prefer_cn` puts the Chinese
/// mirrors first (mainland users chronically time out on pythonhosted); sha256 verification
/// makes any source content-safe.
fn pypi_candidates(url: &str, prefer_cn: bool) -> Vec<String> {
    match url.strip_prefix("https://files.pythonhosted.org/") {
        Some(suffix) => {
            let mirrors = PYPI_MIRRORS.iter().map(|b| format!("{b}/{suffix}"));
            if prefer_cn {
                mirrors.chain([url.to_string()]).collect()
            } else {
                [url.to_string()].into_iter().chain(mirrors).collect()
            }
        }
        None => vec![url.to_string()],
    }
}

/// One NVIDIA runtime lane: a pinned official wheel + where its DLLs live inside it. Shared
/// by the network download AND install_cuda_runtime_local (file_prefix classifies user files).
pub(crate) struct CudaWheel {
    pub guard: &'static str,  // presence of this DLL marks the lane complete (renamed LAST)
    pub file_prefix: &'static str, // local-file classification (filename starts-with)
    pub url: &'static str,    // pinned pythonhosted wheel (cu12 family)
    pub sha256: &'static str, // official PyPI digest
    pub size: u64,
    pub filter: &'static str, // wheel-internal bin dir holding the DLLs
    pub label: &'static str,
    p0: f32,
    p1: f32,
}
pub(crate) const CUDA_WHEELS: [CudaWheel; 4] = [
    CudaWheel { guard: "cudart64_12.dll", file_prefix: "nvidia_cuda_runtime_cu12", url: "https://files.pythonhosted.org/packages/59/df/e7c3a360be4f7b93cee39271b792669baeb3846c58a4df6dfcf187a7ffab/nvidia_cuda_runtime_cu12-12.9.79-py3-none-win_amd64.whl", sha256: "8e018af8fa02363876860388bd10ccb89eb9ab8fb0aa749aaf58430a9f7c4891", size: 3_591_604, filter: "nvidia/cuda_runtime/bin", label: "CUDA runtime", p0: 0.25, p1: 0.28 },
    CudaWheel { guard: "cublas64_12.dll", file_prefix: "nvidia_cublas_cu12", url: "https://files.pythonhosted.org/packages/20/e2/fc9a0e985249d873150276d5afb02e39a66817fedbf1a385724393e505ed/nvidia_cublas_cu12-12.9.2.10-py3-none-win_amd64.whl", sha256: "623f43027d40d44ceadf0043f002bd25cf353e8f13ce90b9a87057019f560661", size: 553_162_896, filter: "nvidia/cublas/bin", label: "cuBLAS", p0: 0.28, p1: 0.55 },
    CudaWheel { guard: "cufft64_11.dll", file_prefix: "nvidia_cufft_cu12", url: "https://files.pythonhosted.org/packages/20/ee/29955203338515b940bd4f60ffdbc073428f25ef9bfbce44c9a066aedc5c/nvidia_cufft_cu12-11.4.1.4-py3-none-win_amd64.whl", sha256: "8e5bfaac795e93f80611f807d42844e8e27e340e0cde270dcb6c65386d795b80", size: 200_067_309, filter: "nvidia/cufft/bin", label: "cuFFT", p0: 0.55, p1: 0.65 },
    CudaWheel { guard: "cudnn64_9.dll", file_prefix: "nvidia_cudnn_cu12", url: "https://files.pythonhosted.org/packages/f2/a4/045f8d0ce6b99726d88e76bbb8ee147123f55e80111d89262762d8149abb/nvidia_cudnn_cu12-9.22.0.52-py3-none-win_amd64.whl", sha256: "5d10117314c861245992dbcf8a6f8ae1f54852137a7c9f80cc9de9fa596f7d62", size: 687_235_974, filter: "nvidia/cudnn/bin", label: "cuDNN", p0: 0.65, p1: 0.93 },
];

/// Extract the ORT CUDA build out of the nupkg into runtime/ort/cuda — STAGED and VALIDATED
/// before the swap (review S66 critical: the old wipe-then-extract destroyed a WORKING install
/// when a user handed the local-install flow a wrong/empty file, then reported success). The
/// staging must yield the core DLLs AND a CUDA-12 providers build (a CUDA-11 / wrong-API nupkg
/// installs cleanly but deadlocks ort's init later — the same major gate as the ready check).
fn place_ort_gpu(app_dir: &std::path::Path, nupkg: &std::path::Path) -> crate::Result<()> {
    let ort_cuda_dir = app_dir.join("runtime").join("ort").join("cuda");
    let staging = app_dir.join("runtime").join("ort").join("cuda.staging");
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging)?;
    let validated = (|| -> crate::Result<()> {
        crate::util::extract_zip_dlls(nupkg, &staging, |n| n.starts_with("runtimes/win-x64/native"))?;
        let providers = staging.join("onnxruntime_providers_cuda.dll");
        if !providers.exists() || !staging.join("onnxruntime.dll").exists() {
            return Err(crate::UtaiError::Download(format!(
                "CUDA_LOCAL_BAD_FILE: no ORT CUDA DLLs found in {}",
                nupkg.display()
            )));
        }
        if cuda_build_major(&providers) != Some(12) {
            return Err(crate::UtaiError::Download(format!(
                "CUDA_LOCAL_BAD_FILE: {} is not a CUDA-12 ORT build",
                nupkg.display()
            )));
        }
        Ok(())
    })();
    if let Err(e) = validated {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(e);
    }
    // Validated — swap in. The existing install is only gone between these two calls.
    let _ = std::fs::remove_dir_all(&ort_cuda_dir);
    std::fs::rename(&staging, &ort_cuda_dir)
        .map_err(|e| crate::UtaiError::Download(format!("ORT CUDA swap failed: {e}")))
}

/// Human labels of CUDA runtime lanes that are missing OR unusable (wrong-major ORT build) —
/// shared by cuda_runtime_paths (panel) and the local-install completion report. A lane counts
/// PRESENT when its guard DLL is resolvable the SAME way the loader resolves it (runtime/cuda
/// OR PATH OR CUDA_PATH — dll_on_path_or_dir, exactly like cuda_provider_deps_resolvable):
/// checking only runtime/cuda showed "missing: cuBLAS" right beside the "Ready" badge on a
/// machine whose DLLs come from an installed Toolkit (CDP 目检-caught contradiction).
fn cuda_missing_lanes(app_dir: &std::path::Path) -> Vec<String> {
    let ort_dir = app_dir.join("runtime").join("ort").join("cuda");
    let dll_dir = app_dir.join("runtime").join("cuda");
    let mut missing: Vec<String> = Vec::new();
    let providers = ort_dir.join("onnxruntime_providers_cuda.dll");
    if !providers.exists() || cuda_build_major(&providers) != Some(12) {
        missing.push("CUDA ORT".to_string());
    }
    for w in &CUDA_WHEELS {
        if !dll_on_path_or_dir(w.guard, &dll_dir) {
            missing.push(w.label.to_string());
        }
    }
    missing
}

/// ATOMIC per-lane placement of one NVIDIA wheel's DLLs into runtime/cuda (S64c audit MAJOR):
/// extract into a staging dir, then rename each DLL in with the GUARD LAST — guard presence ⇒
/// lane complete. A torn extraction can never wedge the skip guard or read as ready.
fn place_cuda_wheel_lane(
    app_dir: &std::path::Path,
    guard: &str,
    filter: &str,
    wheel_zip: &std::path::Path,
) -> crate::Result<()> {
    let cuda_dir = app_dir.join("runtime").join("cuda");
    std::fs::create_dir_all(&cuda_dir)?;
    let stage_dir = app_dir.join("runtime").join(format!("{}.extract", guard));
    let _ = std::fs::remove_dir_all(&stage_dir);
    std::fs::create_dir_all(&stage_dir)?;
    let placed = (|| -> crate::Result<()> {
        crate::util::extract_zip_dlls(wheel_zip, &stage_dir, |n| n.contains(filter))?;
        let mut names: Vec<std::ffi::OsString> = std::fs::read_dir(&stage_dir)?
            .flatten()
            .map(|e| e.file_name())
            .collect();
        if names.is_empty() {
            return Err(crate::UtaiError::Download(format!(
                "CUDA_LOCAL_BAD_FILE: no {} DLLs found in {}",
                guard,
                wheel_zip.display()
            )));
        }
        // Guard renames LAST — its presence must imply every sibling already moved.
        names.sort_by_key(|n| n.eq_ignore_ascii_case(guard));
        for name in names {
            let dest = cuda_dir.join(&name);
            let _ = std::fs::remove_file(&dest); // Windows rename refuses to overwrite
            std::fs::rename(stage_dir.join(&name), &dest)?;
        }
        Ok(())
    })();
    let _ = std::fs::remove_dir_all(&stage_dir);
    placed
}

async fn do_download_cuda_runtime(
    app_dir: &std::path::Path,
    handle: &tauri::AppHandle,
    prefer_cn: bool,
    cancel: &Arc<std::sync::atomic::AtomicBool>,
) -> crate::Result<()> {

    // code+label ride along for i18n (frontend maps code → localized line, label = proper noun;
    // message stays as the raw-English fallback — the S62 pyenv structured-progress pattern).
    let emit = |stage: &str, progress: f32, code: &str, label: &str, msg: &str| {
        let _ = handle.emit("cuda-download-progress", serde_json::json!({
            "stage": stage, "progress": progress, "code": code, "label": label, "message": msg,
        }));
    };

    // S66: everything below rides the unified downloader (download.rs) — .part resume
    // (a mainland user resuming the 655 MB cuDNN wheel loses nothing on a mid-transfer
    // block), mirror rotation, per-chunk stall watchdog, sha256-before-commit, cancel.
    let client = crate::download::client()?;

    // ── Stage 1: CUDA ORT DLLs (NuGet canonical; our HF mirror + hf-mirror as fallbacks) ──
    // 1.24.4 MUST match the ORT API version the `ort` crate (2.0-rc.12) targets — API 24 — AND the
    // bundled DirectML build (1.24.4). A mismatched CUDA build (e.g. 1.20.1 = API 20) makes ort's
    // init_from of the CUDA build DEADLOCK (ort calls API-24 ABI against an API-20 DLL). 1.24.4's
    // providers_cuda imports cudart64_12 / cublas64_12+Lt / cufft64_11 / cudnn64_9.
    // AVOID 1.21.x (mis-built against CUDA 11). Gpu.Windows has the actual DLLs.
    emit("ort", 0.0, "CUDA_DL_DOWNLOADING", "CUDA ORT", "Downloading CUDA ORT runtime...");
    let ort_cuda_dir = app_dir.join("runtime").join("ort").join("cuda");
    let ort_nupkg = app_dir.join("runtime").join("ort_gpu.nupkg.zip");

    let ort_urls: Vec<String> = if prefer_cn {
        // hf-mirror leads for mainland users; NuGet + HF close the chain.
        vec![ORT_GPU_NUPKG_URLS[2].into(), ORT_GPU_NUPKG_URLS[0].into(), ORT_GPU_NUPKG_URLS[1].into()]
    } else {
        ORT_GPU_NUPKG_URLS.iter().map(|s| s.to_string()).collect()
    };
    // Download FIRST, wipe after (S64c audit): the old wipe-then-download order destroyed a good
    // install before the replacement bytes were secured — a failed retry left NOTHING.
    crate::download::download(
        &client,
        &crate::download::DownloadRequest {
            urls: ort_urls,
            dest: ort_nupkg.clone(),
            sha256: Some(ORT_GPU_NUPKG_SHA256.into()),
            expected_size: Some(ORT_GPU_NUPKG_SIZE),
        },
        cancel,
        |done, total| {
            let p = total.map(|t| done as f32 / t.max(1) as f32).unwrap_or(0.0);
            emit("ort", p * 0.2, "CUDA_DL_DOWNLOADING", "CUDA ORT", "Downloading CUDA ORT...");
        },
    )
    .await?;
    emit("ort", 0.2, "CUDA_DL_EXTRACTING", "CUDA ORT", "Extracting CUDA ORT DLLs...");
    place_ort_gpu(app_dir, &ort_nupkg)?;
    let _ = std::fs::remove_file(&ort_nupkg);

    // ── Stage 2 (S64c): the provider's FULL import set from NVIDIA's official PyPI redistributables —
    //    cudart64_12 / cublas64_12+Lt / cufft64_11 / cudnn64_9 (the exact list scanned from the 1.24.4
    //    providers_cuda.dll). No CUDA Toolkit install needed; runtime/cuda sits FIRST in
    //    setup_cuda_dll_paths' search dirs, so our copies also win over a wrong-major Toolkit (e.g. 13).
    //    Each lane SKIPS when its DLL is already present (runtime/cuda is kept across re-downloads;
    //    a flaky/blocked network must not fail an otherwise-complete install). ──
    let cuda_dir = app_dir.join("runtime").join("cuda");
    std::fs::create_dir_all(&cuda_dir)?;
    for w in &CUDA_WHEELS {
        if cuda_dir.join(w.guard).exists() {
            emit("cuda", w.p1, "CUDA_DL_SKIP", w.label, &format!("{} already present — skipping", w.label));
            tracing::info!("CUDA download: {} already present, skipping", w.label);
            continue;
        }
        emit("cuda", w.p0, "CUDA_DL_DOWNLOADING", w.label, &format!("Downloading {}...", w.label));
        let tmp = app_dir.join("runtime").join(format!("{}.whl.zip", w.guard));
        // Unified engine: candidates = pinned pythonhosted + Chinese full-mirror twins (CN-first
        // when the user's download source says mainland), resumable .part kept across failures
        // AND cancels — never delete it here (the whole point over the legacy helper).
        crate::download::download(
            &client,
            &crate::download::DownloadRequest {
                urls: pypi_candidates(w.url, prefer_cn),
                dest: tmp.clone(),
                sha256: Some(w.sha256.into()),
                expected_size: Some(w.size),
            },
            cancel,
            |done, total| {
                let p = total.map(|t| done as f32 / t.max(1) as f32).unwrap_or(0.0);
                emit("cuda", w.p0 + p * (w.p1 - w.p0) * 0.9, "CUDA_DL_DOWNLOADING", w.label, &format!("Downloading {}...", w.label));
            },
        )
        .await?;
        emit("cuda", w.p0 + (w.p1 - w.p0) * 0.9, "CUDA_DL_EXTRACTING", w.label, &format!("Extracting {}...", w.label));
        // ATOMIC placement (S64c audit MAJOR) — see place_cuda_wheel_lane (shared with the
        // install-from-local-file flow): staging dir + guard-renamed-last.
        let placed = place_cuda_wheel_lane(app_dir, w.guard, w.filter, &tmp);
        let _ = std::fs::remove_file(&tmp);
        placed?;
    }

    // Make the fresh runtime resolvable IN-SESSION (S64c audit): runtime/cuda may not have existed
    // at startup, so it never got onto PATH — is_cuda_available's probe would stay false until a
    // restart while the runtime row says Installed. Re-running setup is idempotent.
    crate::setup_cuda_dll_paths(app_dir);

    // ── Stage 3 (DEV BUILDS ONLY): copy next to the debug exe. In release this polluted the
    // install root with the four CUDA DLLs (S64b beta report) — the installed app loads from
    // runtime/ort/cuda directly and needs no exe-side copies. lib.rs setup sweeps old strays. ──
    emit("copy", 0.95, "CUDA_DL_FINALIZING", "", "Finalizing...");
    #[cfg(debug_assertions)]
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let target_debug = exe_dir;
            // Copy CUDA ORT DLLs next to exe for dev convenience
            for entry in std::fs::read_dir(&ort_cuda_dir).into_iter().flatten().flatten() {
                let name = entry.file_name();
                let dest = target_debug.join(&name);
                // Overwrite unconditionally — a stale wrong-CUDA copy here would shadow the new one.
                let _ = std::fs::copy(entry.path(), &dest);
            }
        }
    }

    emit("done", 1.0, "CUDA_DL_DONE", "", "CUDA runtime ready. Restart to activate.");
    tracing::info!("CUDA runtime download complete: ORT={}, cuDNN={}", ort_cuda_dir.display(), cuda_dir.display());
    Ok(())
}

// The legacy no-resume download_file helper is GONE (S66) — every CUDA source now rides
// crate::download (resume + mirrors + sha256 + stall watchdog + cancel).
// extract_nupkg_dlls / extract_wheel_dlls moved to crate::util::extract_zip_dlls
// (callers pass a starts_with / contains closure for the path match).

/// S66 install-from-local-file for the CUDA runtime: the user picks the 4 NVIDIA wheels
/// and/or the ORT GPU nupkg (exact filenames shown in Settings — an offline escape hatch
/// when none of the download routes work). Each file is classified by name and placed
/// through the SAME staging/atomic lanes as the network download. Returns the labels of
/// the lanes installed; unrecognized files fail loudly (never silently skipped).
#[tauri::command]
pub async fn install_cuda_runtime_local(
    app_handle: tauri::AppHandle,
    state: State<'_, Arc<AppState>>,
    paths: Vec<String>,
) -> Result<Vec<String>, String> {
    if paths.is_empty() {
        return Err("CUDA_LOCAL_NO_FILES".to_string());
    }
    if state.task_active("cuda_download") {
        return Err("CUDA_DOWNLOAD_BUSY".to_string());
    }
    let _task = state.begin_task("cuda_download");
    let app_dir = state.app_dir.clone();
    let handle = app_handle.clone();

    let result = tokio::task::spawn_blocking(move || -> Result<Vec<String>, String> {
        let emit = |stage: &str, progress: f32, code: &str, label: &str, msg: &str| {
            let _ = handle.emit("cuda-download-progress", serde_json::json!({
                "stage": stage, "progress": progress, "code": code, "label": label, "message": msg,
            }));
        };
        let mut installed: Vec<String> = Vec::new();
        let n = paths.len() as f32;
        for (i, p) in paths.iter().enumerate() {
            let path = std::path::PathBuf::from(p);
            let name = path
                .file_name()
                .map(|f| f.to_string_lossy().to_ascii_lowercase())
                .unwrap_or_default();
            let frac = i as f32 / n;
            if name.contains("onnxruntime.gpu") && (name.ends_with(".nupkg") || name.ends_with(".zip")) {
                emit("local", frac, "CUDA_DL_EXTRACTING", "CUDA ORT", "Extracting CUDA ORT DLLs...");
                place_ort_gpu(&app_dir, &path).map_err(|e| e.to_string())?;
                installed.push("CUDA ORT".to_string());
                continue;
            }
            match CUDA_WHEELS.iter().find(|w| name.starts_with(w.file_prefix)) {
                Some(w) => {
                    if !name.contains("win_amd64") {
                        return Err(format!("CUDA_LOCAL_BAD_FILE: {} (need the win_amd64 wheel)", name));
                    }
                    emit("local", frac, "CUDA_DL_EXTRACTING", w.label, &format!("Extracting {}...", w.label));
                    place_cuda_wheel_lane(&app_dir, w.guard, w.filter, &path).map_err(|e| e.to_string())?;
                    installed.push(w.label.to_string());
                }
                None => {
                    return Err(format!("CUDA_LOCAL_UNRECOGNIZED: {}", name));
                }
            }
        }
        // In-session PATH refresh, same as the network flow.
        crate::setup_cuda_dll_paths(&app_dir);
        // Honest completion (review S66): a PARTIAL local install (e.g. only two wheels picked)
        // must not read as "runtime ready — restart to activate".
        let still_missing = cuda_missing_lanes(&app_dir);
        if still_missing.is_empty() {
            emit("done", 1.0, "CUDA_DL_DONE", "", "CUDA runtime files installed.");
        } else {
            emit(
                "done",
                1.0,
                "CUDA_DL_LOCAL_PARTIAL",
                &still_missing.join(" · "),
                "Some CUDA runtime parts are still missing.",
            );
        }
        Ok(installed)
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))?;

    match &result {
        Ok(lanes) => tracing::info!("CUDA local install complete: {:?}", lanes),
        Err(e) => tracing::error!("CUDA local install failed: {}", e),
    }
    // Same terminal-event discipline as the network flow (late "local" progress events must
    // never re-latch the panel after the invoke settled).
    if let Err(e) = &result {
        let _ = app_handle.emit(
            "cuda-download-progress",
            serde_json::json!({
                "stage": "error", "progress": 0.0,
                "code": "CUDA_DL_FAILED", "label": "", "message": e.clone(),
            }),
        );
    }
    result
}

/// S66: the exact on-disk CUDA runtime layout for the Settings panel (copyable paths =
/// inspection/support-friendly) + per-lane presence so a half install is visible at a glance.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CudaRuntimePaths {
    pub ort_dir: String,
    pub dll_dir: String,
    /// Human labels of lanes whose guard DLL is MISSING from runtime/cuda (empty = complete).
    pub missing: Vec<String>,
    /// Required local-install filenames (shown in the picker dialog).
    pub expected_files: Vec<String>,
}

#[tauri::command]
pub fn cuda_runtime_paths(state: State<'_, Arc<AppState>>) -> CudaRuntimePaths {
    let ort_dir = state.app_dir.join("runtime").join("ort").join("cuda");
    let dll_dir = state.app_dir.join("runtime").join("cuda");
    let missing = cuda_missing_lanes(&state.app_dir);
    let mut expected: Vec<String> = vec!["Microsoft.ML.OnnxRuntime.Gpu.Windows.1.24.4.nupkg".to_string()];
    expected.extend(CUDA_WHEELS.iter().map(|w| {
        w.url.rsplit('/').next().unwrap_or(w.file_prefix).to_string()
    }));
    CudaRuntimePaths {
        ort_dir: ort_dir.to_string_lossy().to_string(),
        dll_dir: dll_dir.to_string_lossy().to_string(),
        missing,
        expected_files: expected,
    }
}

pub(crate) fn is_cuda_available() -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // S64c self-contained runtime: cudart lives in runtime/cuda, which setup_cuda_dll_paths put
        // on PATH before any caller runs — a plain PATH scan covers it (and any real Toolkit).
        if dll_on_path("cudart64_12.dll") {
            return true;
        }
        // Check CUDA toolkit's standard install location first (fast)
        if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
            let bin = std::path::Path::new(&cuda_path).join("bin");
            if bin.exists() {
                if let Ok(entries) = std::fs::read_dir(&bin) {
                    for entry in entries.flatten() {
                        let name = entry.file_name().to_string_lossy().to_lowercase();
                        if name.starts_with("cudart64_") && name.ends_with(".dll") {
                            return true;
                        }
                    }
                }
            }
        }
        // Fallback: check if nvcc is on PATH (lightweight — just runs one command)
        if let Ok(output) = std::process::Command::new("where")
            .arg("nvcc.exe")
            .creation_flags(crate::util::CREATE_NO_WINDOW)
            .output()
        {
            if output.status.success() {
                return true;
            }
        }
    }
    false
}
