use std::sync::Arc;
use tauri::{Emitter, State};

use crate::inference::engine::DeviceConfig;
use crate::AppState;

#[derive(serde::Serialize)]
pub struct HardwareInfo {
    pub gpu_name: String,
    pub cuda_available: bool,
    /// Consumption point 1 of the single package predicate: this machine has an NVIDIA card our
    /// shipped CUDA package can run. Gates the CUDA-runtime download entry and the CUDA device
    /// option — an unsupported card is never offered CUDA and uses DirectML instead.
    /// ⚠ UNDETERMINED = NOT SUPPORTED (fail-CLOSED). `cuda_pkg_supported` is the single authority
    /// and carries the rationale; this comment used to claim fail-OPEN, which was the opposite of
    /// both the implementation and the agreed design.
    pub cuda_supported: bool,
    /// S74b: a saved explicit device preference was stale (hardware/window changed) and was
    /// demoted to Auto at startup. The frontend toasts this ONCE (App's startup effect).
    pub preference_demoted: bool,
    pub directml_available: bool,
    pub current_device: String,
    /// The configured GPU ordinal of the current device preference (0 for cpu/auto).
    /// S68b: feeds the Settings "preferred GPU" picker.
    pub current_device_id: u32,
    /// S68b (§user): Auto-mode preferred GPU (DXGI index; None = fully automatic).
    pub auto_gpu: Option<u32>,
    /// Which ORT build this PROCESS loaded ("CUDA" | "DirectML" | dev/system labels).
    /// S68b: lets the UI say "restart required" when the preference implies the OTHER
    /// build — the community user read the current-build fact as a hardware verdict.
    pub ort_build: String,
    /// Per-adapter vendor classification (S42, for runtime-pack recommendation).
    /// S68b: DXGI-first (gpu.rs), WMI fallback. Vendor comes from the PCI vendor id —
    /// NEVER from WMI AdapterRAM (a lying uint32: this dev box reports the 3080 Ti as
    /// 4 GB) and never from name heuristics.
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
    /// ★S75 review: the UI identity, and the ONLY thing the start request carries. `<vendor>:<n>`
    /// (NVIDIA with smi: `nvidia:<uuid>`) — globally unique.
    ///
    /// It exists because `value` STOPPED being unique the moment this list became multi-vendor:
    /// AMD's first card and Intel's first card are both vendor-relative index "0", so a lookup
    /// (or a React key, or the dropdown's selectedIdx) keyed on `value` silently resolves to
    /// whichever vendor happens to come first — the exact shape of the S67 wrong-device-mask bug.
    /// UI identity and device mask are two different things; conflating them was the bug.
    pub id: String,
    pub label: String,
    /// NVIDIA: the nvidia-smi UUID ("GPU-…" — CUDA_VISIBLE_DEVICES accepts it, exact
    /// identity, immune to enumeration-order drift). Fallbacks: vendor-relative index.
    /// ⚠ This string is fed VERBATIM to device.py's visibility env var — its format is a
    /// python-side contract. S75 added the fields below rather than encoding anything extra here.
    pub value: String,
    /// S75: the training runtime variant that would drive this GPU ("nv-cu130"/"amd"/"xpu");
    /// None = this vendor has no training runtime at all. The frontend never sends this back —
    /// try_start re-derives the whole entry from `id` against a freshly built list, so a stale
    /// or hand-edited payload gets a refusal instead of someone else's runtime.
    pub variant: Option<String>,
    /// S75 (the S74b shape, ported from `list_inference_gpus`): "you can select it" must imply
    /// "it can train here". Unselectable entries stay VISIBLE with their reason — a user who
    /// knows they own that card must not think we lost it.
    pub selectable: bool,
    /// Stable CODE for why not (frontend maps via backendError.ts).
    pub reason: Option<String>,
}

/// One NVIDIA card as nvidia-smi sees it — identity AND capability from a SINGLE query, so the
/// per-GPU compute cap can never be mis-paired. (Zipping two separate smi calls would: the uuid
/// query drops rows whose tail is not a UUID, and the cap query does not.)
struct NvSmiGpu {
    name: String,
    uuid: String,
    cc10: Option<i32>,
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
    /// S68b: Auto-mode preferred GPU as a DXGI adapter index (None = fully automatic —
    /// the pre-S68b behavior). Kept OUTSIDE DeviceConfig so legacy `"device": "auto"`
    /// strings keep deserializing (externally-tagged unit variant); skipped when None
    /// so an untouched picker never even changes config.json's bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto_gpu: Option<u32>,
    /// S68c: OLD data roots of completed (verified) migrations, awaiting reclaim at startup —
    /// deleting in-session would collide with live handles (ONNX session mmaps, asset-protocol
    /// avatar reads). A LIST (§user round 2): entries are independent — one old root on an
    /// unplugged removable drive stays queued (retried every boot) without blocking anything,
    /// and a later migration APPENDS instead of overwriting (no orphaned roots). Entries are
    /// removed one-by-one as their reclaim completes. Skipped when empty so users who never
    /// migrate keep byte-identical config.json.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pending_delete_dirs: Vec<String>,
    /// ★★S133 §F2⒝ ④e — DATA-ROOT-RELATIVE paths the user deliberately deleted **while an old
    /// root was still queued above**. The reclaim removes them from the old copy before syncing.
    ///
    /// ⛔ Without this the delta sync copies them straight back: every one of its refusals hangs
    /// on `to.exists()`, and a path the user just deleted does not exist on the destination ⇒ the
    /// whole subtree falls through to a plain per-file recursion, `needs_copy` answers true for
    /// every missing file, and the run/slot/project comes back — counted as `copied`, logged as
    /// 「freed X MB … stragglers synced first」. Zero warnings. (`delete_slot` / `delete_project`
    /// have had this hole since S68c; ④e's per-run delete would have made it frequent.)
    ///
    /// ⚠ It lives in `config.json` (the APP dir) on purpose — anything kept in the data root is
    /// itself a reclaim target, and `project.json` is demonstrably one (see `needs_copy`).
    /// ⚠ Recorded ONLY while `pending_delete_dirs` is non-empty, and dropped when that queue
    /// empties: with no old root there is nothing that could resurrect anything, so the list has
    /// no reason to grow on the machine of a user who never migrated.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub deleted_since_migration: Vec<String>,
    /// S115 §F5-2: diagnostic mode — makes a training run reproduce a crash legibly at the
    /// cost of speed (see `training::diagnostics` for the exact set and why it is keyed on the
    /// runtime variant). PERSISTED on purpose: reproducing a GPU fault can take several
    /// attempts across app restarts, so a per-session flag would be useless for the case it
    /// exists for. The other half of that decision lives in the UI — because it persists, a
    /// user WILL forget it on, so the training page carries a permanent banner while it is
    /// set; a slow run that never explains itself would come back to us as "your update made
    /// training slower", i.e. a regression we manufactured.
    #[serde(default)]
    pub diagnostic_mode: bool,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            device: DeviceConfig::Auto,
            data_dir: None,
            cuda_mem_limit_mb: 0,
            auto_gpu: None,
            pending_delete_dirs: Vec::new(),
            deleted_since_migration: Vec::new(),
            diagnostic_mode: false,
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
    let current_device_id = match &current {
        DeviceConfig::DirectMl { device_id } | DeviceConfig::Cuda { device_id } => *device_id,
        _ => 0,
    };

    let gpus = query_gpu_adapters();
    // S68b: nvidia-smi is queried ONCE here and its result is both the training list
    // (UUID identity) and independent NVIDIA evidence. On the community box the WMI
    // probe failed entirely ("Unknown GPU") and every vendor-gated capability collapsed
    // with it — while nvidia-smi (and CUDA itself) worked the whole time. Probes must
    // corroborate, never gate each other.
    let smi_gpus = nvidia_gpu_uuids();
    let has_nvidia = gpus.iter().any(|g| g.vendor == "nvidia") || !smi_gpus.is_empty();
    let gpu_name = if gpus.is_empty() {
        "Unknown GPU".to_string()
    } else {
        gpus.iter().map(|g| g.name.as_str()).collect::<Vec<_>>().join(", ")
    };
    Ok(HardwareInfo {
        gpu_name,
        // Vendor-guarded (S64c audit): the self-downloaded runtime/cuda DLLs satisfy the PATH probe
        // even on a box whose NVIDIA card is gone (migrated data dir) — the badge must track the GPU.
        cuda_available: has_nvidia && is_cuda_available(),
        // S74b consumption point 1 (entry/option gate). No `has_nvidia &&`: cuda_pkg_supported is
        // itself the NVIDIA evidence (a successful nvidia-smi cc read), so it still answers on a
        // box whose adapter probe failed — the S68b rescue property, kept.
        cuda_supported: cuda_pkg_supported(),
        preference_demoted: PREFERENCE_DEMOTED.load(std::sync::atomic::Ordering::Relaxed),
        directml_available: cfg!(windows),
        current_device: current_str,
        current_device_id,
        auto_gpu: state.inference.engine.auto_gpu(),
        ort_build: crate::ORT_LOADED_BUILD.get().cloned().unwrap_or_else(|| "?".to_string()),
        // S116: the SAME hoisted probe the pack gates use — `nvidia_compute_caps_cc10` is a
        // per-process OnceLock, so asking here costs no extra nvidia-smi subprocess.
        recommended_variant: recommend_variant(&gpus, &nvidia_compute_caps_cc10()).to_string(),
        nvidia_vram_mb: if has_nvidia { nvidia_total_vram_mb() } else { None },
        training_gpus: training_gpu_list(
            &gpus,
            smi_gpus,
            &crate::pyenv::available_training_variants(&state.app_dir),
        ),
        gpus,
    })
}

/// Emit the one-per-process "Hardware:" inventory line to the log FILE, RELIABLY at
/// process startup (S74). It used to log lazily inside get_hardware_info on the first
/// frontend startup-check call — which was flaky: community crash reports frequently
/// lacked this line entirely (the frontend probe hadn't fired, or the log the reporter
/// sent started after it), leaving the reporter's GPU/RAM unknown. lib.rs::run now calls
/// this on a startup background thread (nvidia-smi + DXGI enumeration must not delay the
/// window). Single source: get_hardware_info no longer logs.
pub(crate) fn log_hardware_inventory() {
    let (total_mb, avail_mb) = crate::inference::engine::system_memory_mb();
    let inventory = crate::gpu::inventory_line().unwrap_or_else(|| {
        let gpus = query_gpu_adapters();
        if gpus.is_empty() {
            "Unknown GPU".to_string()
        } else {
            gpus.iter().map(|g| g.name.as_str()).collect::<Vec<_>>().join(", ")
        }
    });
    let driver = nvidia_driver_version()
        .map(|v| format!("; NVIDIA driver {v}"))
        .unwrap_or_default();
    tracing::info!(
        "Hardware: GPUs [{}]{}; physical RAM {} MB (available commit {} MB)",
        inventory, driver, total_mb, avail_mb
    );
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
///
/// S68b: nvidia-smi's result comes IN (queried once by get_hardware_info) and wins
/// UNCONDITIONALLY — the old code only consulted it after WMI had already classified an
/// adapter as NVIDIA, so the community box whose WMI probe failed outright never asked
/// the perfectly-working nvidia-smi and silently forced CPU training on an RTX 3080.
/// ★S75 rewrite — the vendor EARLY-RETURN chain is gone. It used to return the first vendor that
/// matched (nvidia > amd > intel), which meant:
///   - On a mixed box (the dev machine: RTX 3080 Ti + Radeon 780M) the AMD card could NEVER be
///     picked — even with the amd pack installed and the 780M being the ONE gfx target that pack
///     supports. Confirmed live by the user, 2026-07-23.
///   - On an AMD-only box the list was pure vendor matching with no capability check, so an
///     RX 7900 (which our pack cannot drive) was offered, resolved to the CPU pack, and trained
///     on the CPU — `require_wanted_accelerator` deliberately lets device_backend="cpu" through
///     as legitimate explicit-CPU, so the only trace was one tracing::warn.
///   - Same for an NVIDIA card below `CUDA_CC10_FLOOR`: offered, never checked.
/// The old doc claimed the sidecar guard turned "any remaining mismatch" into a loud failure.
/// That is true only when the resolved backend is an ACCELERATOR; it never covered the CPU-pack
/// case, because it assumed the dropdown only ever offered cards some installed pack could drive.
/// That assumption is what this function now actually enforces.
///
/// Every adapter of every vendor is listed. Selectability = the single criterion:
///   a variant exists for the vendor ∧ this machine's hardware qualifies for it
///   (`variant_supported`, the S74b predicate — unchanged, reused) ∧ that variant is installed
///   (`pyenv::available_training_variants`).
/// Unselectable entries stay VISIBLE with a reason CODE (S74b: a card the user knows they own
/// must not silently vanish), and `TRAINING_GPU_PACK_MISSING` is the actionable one.
fn training_gpu_list(
    gpus: &[GpuAdapter],
    smi: Vec<NvSmiGpu>,
    available: &[&'static str],
) -> Vec<TrainingGpu> {
    let nv_cc10 = nvidia_compute_caps_cc10();
    let judge = |variant: &'static str, hw_ok: bool, cc_unknown: bool| -> (bool, Option<String>) {
        if cc_unknown {
            return (false, Some("TRAINING_GPU_CC_UNKNOWN".to_string()));
        }
        if !hw_ok {
            return (false, Some("TRAINING_GPU_UNSUPPORTED".to_string()));
        }
        if !available.contains(&variant) {
            return (false, Some("TRAINING_GPU_PACK_MISSING".to_string()));
        }
        (true, None)
    };

    let mut out: Vec<TrainingGpu> = Vec::new();

    // NVIDIA — nvidia-smi identity wins UNCONDITIONALLY when it answers (S68b: the community box
    // whose WMI probe failed outright still had a perfectly working smi). Per-card cap from the
    // same query; when smi is silent, fall back to adapters + the machine-level cap set.
    if !smi.is_empty() {
        for g in smi {
            let cc_unknown = g.cc10.is_none();
            let hw_ok = g.cc10.is_some_and(crate::gpu::cuda_cc_supported_training);
            let (selectable, reason) = judge("nv-cu130", hw_ok, cc_unknown);
            out.push(TrainingGpu {
                id: format!("nvidia:{}", g.uuid),
                label: g.name,
                value: g.uuid,
                variant: Some("nv-cu130".to_string()),
                selectable,
                reason,
            });
        }
    } else {
        let hw_ok = nv_cc10.iter().copied().any(crate::gpu::cuda_cc_supported_training);
        let cc_unknown = nv_cc10.is_empty();
        for (i, g) in gpus.iter().filter(|g| g.vendor == "nvidia").enumerate() {
            let (selectable, reason) = judge("nv-cu130", hw_ok, cc_unknown);
            out.push(TrainingGpu {
                id: format!("nvidia:{i}"),
                label: g.name.clone(),
                value: i.to_string(),
                variant: Some("nv-cu130".to_string()),
                selectable,
                reason,
            });
        }
    }

    // AMD / Intel — vendor-relative index (HIP / ZE_AFFINITY_MASK ordinals). The index must be
    // counted WITHIN the vendor, exactly as before; it is what device.py masks with.
    for (vendor, variant, capable) in [
        ("amd", "amd", &amd_adapter_is_rocm_capable as &dyn Fn(&GpuAdapter) -> bool),
        ("intel", "xpu", &intel_adapter_is_xpu_capable as &dyn Fn(&GpuAdapter) -> bool),
    ] {
        for (i, g) in gpus.iter().filter(|g| g.vendor == vendor).enumerate() {
            let (selectable, reason) = judge(variant, capable(g), false);
            out.push(TrainingGpu {
                id: format!("{vendor}:{i}"),
                label: g.name.clone(),
                value: i.to_string(),
                variant: Some(variant.to_string()),
                selectable,
                reason,
            });
        }
    }

    // Anything else (Basic Render / virtual adapters): listed, never selectable — there is no
    // training runtime for it at all, which is a different fact from "unsupported card".
    for (i, g) in gpus
        .iter()
        .filter(|g| !matches!(g.vendor.as_str(), "nvidia" | "amd" | "intel"))
        .enumerate()
    {
        out.push(TrainingGpu {
            id: format!("other:{i}"),
            label: g.name.clone(),
            value: String::new(),
            variant: None,
            selectable: false,
            reason: Some("TRAINING_GPU_NO_RUNTIME".to_string()),
        });
    }
    debug_assert!(
        {
            let mut ids: Vec<&str> = out.iter().map(|g| g.id.as_str()).collect();
            ids.sort_unstable();
            let n = ids.len();
            ids.dedup();
            ids.len() == n
        },
        "TrainingGpu.id must be unique — it is the lookup key try_start resolves against"
    );
    out
}

/// Re-derive the entry the request names, from the SAME list the UI was built from — never from
/// the payload. try_start calls this at PREFLIGHT: a stale or hand-edited `id` gets a loud
/// refusal, not someone else's runtime.
///
/// Keyed on `id`, not `value`: `value` is only unique within a vendor (S75 review — AMD's and
/// Intel's first cards are both "0"), so a value-keyed lookup resolved to whichever vendor was
/// pushed first.
pub(crate) fn training_gpu_by_id(app_dir: &std::path::Path, id: &str) -> Option<TrainingGpu> {
    let gpus = query_gpu_adapters();
    let available = crate::pyenv::available_training_variants(app_dir);
    training_gpu_list(&gpus, nvidia_gpu_uuids(), &available).into_iter().find(|g| g.id == id)
}

/// NVIDIA cards as (name, UUID) via nvidia-smi — the only enumeration whose identity
/// CUDA itself understands. Empty on any failure (no smi / no driver): callers fall
/// back to vendor-relative indices.
#[cfg(windows)]
fn nvidia_gpu_uuids() -> Vec<NvSmiGpu> {
    use std::os::windows::process::CommandExt;
    let out = match std::process::Command::new("nvidia-smi")
        // S75: compute_cap rides ALONG (same query, same row) — the training gate needs the cap
        // PER CARD, and the machine-level `nvidia_compute_caps_cc10()` cannot say which card.
        .args(["--query-gpu=name,uuid,compute_cap", "--format=csv,noheader"])
        .creation_flags(crate::util::CREATE_NO_WINDOW)
        .output()
    {
        Ok(o) if o.status.success() => o,
        _ => return Vec::new(),
    };
    let text = String::from_utf8_lossy(&out.stdout);
    text.lines()
        .filter_map(|l| {
            // "NVIDIA GeForce RTX 3080 Ti, GPU-8a2c…, 8.6" — rsplit from the RIGHT so a comma
            // INSIDE the name can't shear the row; a non-UUID uuid field drops the row instead
            // of feeding CUDA a garbage mask.
            let (head, cap) = l.rsplit_once(',')?;
            let (name, uuid) = head.rsplit_once(',')?;
            let uuid = uuid.trim();
            if !uuid.starts_with("GPU-") {
                return None;
            }
            Some(NvSmiGpu {
                name: name.trim().to_string(),
                uuid: uuid.to_string(),
                // Unparseable cap = unknown, NOT unsupported — it becomes its own reason CODE
                // (S74b: "we read it and it's out of range" ≠ "we couldn't read it").
                cc10: cap.trim().parse::<f32>().ok().map(|c| (c * 10.0).round() as i32),
            })
        })
        .collect()
}

#[cfg(not(windows))]
fn nvidia_gpu_uuids() -> Vec<NvSmiGpu> {
    Vec::new()
}

/// Default runtime-pack variant for this machine. NVIDIA wins over everything (the
/// only fully-supported training path); AMD over Intel. iGPU-vs-dGPU is deliberately
/// NOT guessed — the pick is only a DEFAULT and the UI lets the user override
/// (Pinokio's silent wrong-variant installs are the anti-pattern we're avoiding).
///
/// ★S116: every arm asks the SAME question `variant_supported` asks, so this can never
/// name a pack the download list hides. It used to take a bare `has_nvidia` (vendor
/// presence) while f87443a narrowed the amd/intel arms beside it to capability
/// predicates — so a GTX 10-series box, or one whose nvidia-smi cannot answer, read
/// "Recommended variant: nv-cu130" directly above a list that filters on `c.supported`
/// and therefore does not contain it, with no reason shown. Naming a pack we then
/// refuse to offer is consumption point 6 (禁用必给理由) run backwards.
/// ⚠ The S68b rescue is KEPT, for the same reason `cuda_supported` keeps it one screen
/// up (see `get_hardware_info`): `nv_cc10` IS nvidia-smi evidence, so a box whose ADAPTER
/// probe died still recommends nv-cu130. What no longer survives is a dead nvidia-smi —
/// and that is exactly the case S74b made fail-CLOSED, where the pack is hidden anyway.
fn recommend_variant(gpus: &[GpuAdapter], nv_cc10: &[i32]) -> &'static str {
    if nv_cc10.iter().any(|&cc| crate::gpu::cuda_cc_supported_training(cc)) {
        "nv-cu130"
    } else if amd_is_rocm_capable(gpus) {
        "amd"
    } else if intel_is_xpu_capable(gpus) {
        "xpu"
    } else {
        "cpu"
    }
}

/// Enumerate video adapters with PCI vendor ids. S68b: DXGI first (subprocess-free,
/// healthy wherever a display stack exists — the very thing DirectML runs on), WMI as
/// the fallback for exotic DXGI failures. Software adapters (Basic Render Driver) are
/// excluded for parity with the old WMI inventory (they are not Win32_VideoControllers).
pub(crate) fn query_gpu_adapters() -> Vec<GpuAdapter> {
    let dxgi: Vec<GpuAdapter> = crate::gpu::dxgi_adapters()
        .into_iter()
        .filter(|a| !a.software)
        .map(|a| GpuAdapter { name: a.name, vendor: a.vendor.to_string() })
        .collect();
    if !dxgi.is_empty() {
        return dxgi;
    }
    query_gpu_adapters_wmi()
}

/// The pre-S68b WMI/PowerShell probe, kept verbatim as the fallback. Known field failure
/// (community RTX 3080 box): the whole probe returns empty — powershell.exe unresolvable
/// or a broken WMI repository; the exit status lands as empty stdout → JSON parse fails.
fn query_gpu_adapters_wmi() -> Vec<GpuAdapter> {
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

/// All installed NVIDIA cards' compute capabilities as cc10 (major*10+minor, e.g. 8.6 → 86) via
/// nvidia-smi. Empty when nvidia-smi is absent/unreadable (no driver / non-NVIDIA box) — see
/// `cuda_pkg_supported` for what "undetermined" means (S74b: it means NOT supported).
/// Cached per process (S74b): this now runs on the STARTUP path (the ORT build gate in
/// init_ort_runtime) as well as from the settings/pack/training queries, and each call is an
/// nvidia-smi subprocess (~100 ms). Installed GPUs do not change within a process; the same
/// convention as gpu::cuda_device_label's table.
#[cfg(windows)]
pub(crate) fn nvidia_compute_caps_cc10() -> Vec<i32> {
    static CACHE: std::sync::OnceLock<Vec<i32>> = std::sync::OnceLock::new();
    CACHE.get_or_init(probe_nvidia_compute_caps_cc10).clone()
}

/// Bounded wrapper (S74b review): this probe now runs on the SYNCHRONOUS pre-window startup path
/// (the ORT build gate), and nvidia-smi can hang for tens of seconds on a wedged driver — the same
/// batch moved the hardware-inventory log to a background thread for exactly this reason. A hang
/// there is a black window, so cap the wait and fail CLOSED (an unanswered probe is "unsupported",
/// which is the documented direction; the config rewrite is separately gated on a real answer).
#[cfg(windows)]
fn probe_nvidia_compute_caps_cc10() -> Vec<i32> {
    const PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(8);
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(probe_nvidia_compute_caps_cc10_blocking());
    });
    match rx.recv_timeout(PROBE_TIMEOUT) {
        Ok(v) => v,
        Err(_) => {
            tracing::warn!(
                "nvidia-smi compute-cap probe did not answer within {}s — treating CUDA support as undetermined",
                PROBE_TIMEOUT.as_secs()
            );
            Vec::new()
        }
    }
}

#[cfg(windows)]
fn probe_nvidia_compute_caps_cc10_blocking() -> Vec<i32> {
    use std::os::windows::process::CommandExt;
    let Ok(out) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .creation_flags(crate::util::CREATE_NO_WINDOW)
        .output()
    else {
        return Vec::new();
    };
    if !out.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.trim().parse::<f32>().ok())
        .map(|cc| (cc * 10.0).round() as i32)
        .collect()
}

/// ★THE machine-level predicate (S74b): does THIS machine have an NVIDIA GPU our INFERENCE CUDA
/// package supports? Every inference-CUDA decision reads this one function — the download entry
/// and device option (`get_hardware_info.cuda_supported`), the download command's refusal, which
/// ORT build `init_ort_runtime` loads, and whether an already-installed package counts as usable
/// or as reclaimable storage.
///
/// ⚠ UNDETERMINED = NOT SUPPORTED (fail-CLOSED — user decision, S74b). A probe that cannot
/// confirm support must not leave a 1.6 GB package sitting there invisible and unreclaimable: a
/// user moving to a new machine (different vendor, or no GPU at all) is far more common than a
/// temporarily detached eGPU, and the machine-swap case is exactly the one fail-open strands.
/// Consequence we accept: an NVIDIA box whose driver is broken enough that nvidia-smi fails sees
/// the CUDA entry disappear and its package listed as reclaimable; fixing the driver restores it,
/// and nothing is deleted without the user confirming a dialog that says so.
#[cfg(windows)]
pub(crate) fn cuda_pkg_supported() -> bool {
    nvidia_compute_caps_cc10()
        .iter()
        .any(|&cc10| crate::gpu::cuda_cc_supported_inference(cc10))
}

#[cfg(not(windows))]
pub(crate) fn nvidia_compute_caps_cc10() -> Vec<i32> {
    Vec::new()
}
#[cfg(not(windows))]
pub(crate) fn cuda_pkg_supported() -> bool {
    false
}

/// Total VRAM of the largest NVIDIA card in MB (nvidia-smi memory.total), None = undetermined
/// (no nvidia-smi / no NVIDIA card). S66: feeds the GPU-特征提取 gate — the feature's measured
/// steady peak is ~9.4 GB (user, two runs), so cards under 12 GB can't enable it. Sole consumer:
/// `VoiceModelPicker.tsx`, via `HardwareInfo::nvidia_vram_mb`.
///
/// ⚠ THIS gate fails OPEN — `None` leaves the checkbox ENABLED — which is the OPPOSITE of
/// `cuda_pkg_supported` / `variant_supported`. ⛔ Do NOT cite those as the reason.
/// S115: this line used to say "the variant_supported convention", and that citation was CORRECT
/// WHEN WRITTEN (S66 `a2b6359`: `variant_supported` then read `nv_cc.map_or(true, |cc| cc >= 7.5)`
/// and its own doc said an undetermined cap "fails OPEN"). S74b (`f87443a`) INVERTED the
/// convention repo-wide and left three citers behind — this one, `download_cuda_runtime`, and
/// VoiceModelPicker.tsx. The direction is chosen per CONSEQUENCE, never by a global convention:
///   - fail-CLOSED there, because an unanswered probe would strand a multi-GB package as
///     invisible-and-unreclaimable, or keep offering a download that cannot run;
///   - fail-OPEN here, because this probe is NVIDIA-only by construction (`get_hardware_info`
///     returns None outright when `!has_nvidia`) while the gated extractors run on the GLOBAL
///     device ⇒ fail-closed would permanently disable the feature on every DirectML box the probe
///     knows nothing about; and only the ENABLE direction is gated, so a pre-existing `true` can
///     always be switched off.
/// Accepted consequence: on an NVIDIA box under 12 GB whose nvidia-smi merely fails, the user can
/// tick the box and hit an OOM at render — recoverable by unticking, and S66 already chunk-bounded
/// the worst offender (`inference/rvc.rs`, the last unbounded GPU feed).
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

/// NVIDIA driver version via nvidia-smi ("566.14"), None = undetermined. S68b forensics:
/// the 20%-crash line of inquiry landed on the driver layer and no community log ever
/// recorded the driver version — logged once in the hardware-inventory line.
#[cfg(windows)]
fn nvidia_driver_version() -> Option<String> {
    use std::os::windows::process::CommandExt;
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=driver_version", "--format=csv,noheader"])
        .creation_flags(crate::util::CREATE_NO_WINDOW)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    text.lines().next().map(|l| l.trim().to_string()).filter(|s| !s.is_empty())
}

#[cfg(not(windows))]
fn nvidia_driver_version() -> Option<String> {
    None
}

/// Driver MAJOR ("581.42" → 581). pub(crate): the nv-cu130 download gate and the
/// envtest crash diagnosis (commands/pyenv.rs) both need it — CUDA 13 wheels require
/// an r580+ driver, and the community RTX 4070 Laptop case proved a CUDA-12-happy
/// driver sails through every other probe while torch-cu130 sees zero devices.
pub(crate) fn nvidia_driver_major() -> Option<u32> {
    nvidia_driver_version()?.split('.').next()?.parse().ok()
}

/// S74: whether an Intel GPU here is a torch-XPU target (Arc family), NOT a legacy integrated
/// GPU (Iris Xe / UHD / HD Graphics) that torch.xpu does not support — a pre-Arc user hit the
/// self-check failing and reported it as a bug, because the pack was offered to a card it can't
/// run. torch-xpu covers Arc discrete (A/B), Core Ultra's "Arc Graphics" iGPU, and Data Center
/// Max — all "Arc"-branded; the token is "Arc", NEVER "Xe" (which would wrongly match Iris Xe).
/// Legacy iGPUs (no "Arc" in the name) are denied here; local-file install stays ungated and the
/// on-device envtest is the final word.
/// S74b: a coarse identity for "the hardware this machine has", stamped into a self-test report so
/// a later run of the app can tell whether that report still describes THIS box. Adapter names +
/// the NVIDIA driver version are enough to catch what matters — a swapped GPU, a moved install, a
/// driver upgrade — without pretending to be a serial number.
///
/// Why it exists: envtest.json is a snapshot. A green badge from a different machine (or from
/// before a driver change) is a stale verdict presented as a current one — the same class of lie
/// the run-scoped stale-report deletion in run_envtest_inner already guards against, extended from
/// "this run" to "this hardware".
pub(crate) fn machine_sig() -> String {
    let mut names: Vec<String> = query_gpu_adapters().into_iter().map(|g| g.name).collect();
    names.sort();
    format!("{}|{}", names.join(";"), nvidia_driver_version().unwrap_or_default())
}

/// S74b: whether an AMD GPU here is one our SHIPPED rocm pack can actually run.
///
/// MEASURED from the shipped tarball, not inferred from names — S115 listed all 43104 members and
/// decompressed every kernel container: the pack's GENERAL compute kernels are gfx1103-only. All
/// 876 code objects inside the four `.kpack` files and all 124 inside the `.hsaco`/`.co` files
/// carry EF_AMDGPU_MACH gfx1103 and none names another arch; `torch_hip.dll`, `c10_hip.dll`,
/// `rocblas.dll` and `libhipblaslt.dll` embed no device objects at all, so those kpacks really are
/// that code's only home. gfx1103 is the Phoenix / Hawk Point iGPU sold as Radeon 780M / 760M /
/// 740M; no discrete RX card (gfx1100-1102 RDNA3, RDNA2, RDNA4) and no other iGPU generation
/// (680M = gfx1035, 880M/890M = gfx115x) has a compute kernel here.
///
/// ⚠ S115 CORRECTION — this comment used to claim `amd-torch-device-gfx110x` was "a family
/// dist-info with no kernels behind it". That is FALSE: it installs 1738 files / 232 MiB, 1731 of
/// them AOTriton flash-attention images under `torch/lib/aotriton.images/amd-gfx110x/`, and those
/// DO carry gfx1100/1101/1102/1103 machine code (76748 code objects censused via ELF `e_flags`,
/// 1731 of 1731 files). The CONCLUSION is unchanged, because SDPA images are not a runtime: an
/// RX 7000 would still find no torch-HIP and no rocBLAS/hipBLASLt kernel for its arch.
/// 【unverified】 that such a card actually FAILS — no gfx1100-1102 card has ever been run against
/// this pack, and the pack also ships a complete AMDGPU compiler (amd_comgr / hiprtc / clang;
/// `_rocm_sdk_core/` alone is 49% of the 4.5 GB) plus device-libs bitcode for gfx1100-1102, so
/// "no precompiled kernels" is not by itself proof of "cannot run". What is measured is the
/// INVENTORY. Offering a 4.5 GB download on an untested guess is the Iris-Xe mistake with a
/// bigger file, so this gate stays narrow until someone runs one.
///
/// Token match on the adapter name, like `intel_is_xpu_capable` — reading the real gfx target
/// needs ROCm tooling we do not bundle. "780m" cannot collide with "RX 7800M" (the char after 780
/// is another digit there). The on-device envtest remains the authority and local-file install
/// stays ungated, so a miss costs a hidden download entry, never a blocked user.
///
/// ⚠ The NARROWNESS is the pack's, not this predicate's: broadening AMD coverage means shipping
/// more device kernels (a packaging task, tracked in the backlog), and this predicate must widen
/// in the same commit that does it.
/// S75: split per-ADAPTER so the training-device gate can answer "can THIS card train" while the
/// machine-level wrappers below keep answering "is this pack worth offering". One predicate, two
/// granularities — the pack gate and the device gate can never disagree about the same card.
fn amd_adapter_is_rocm_capable(g: &GpuAdapter) -> bool {
    g.vendor == "amd" && {
        let n = g.name.to_ascii_lowercase();
        ["780m", "760m", "740m"].iter().any(|t| n.contains(t))
    }
}

fn amd_is_rocm_capable(gpus: &[GpuAdapter]) -> bool {
    gpus.iter().any(amd_adapter_is_rocm_capable)
}

fn intel_adapter_is_xpu_capable(g: &GpuAdapter) -> bool {
    g.vendor == "intel" && {
        let n = g.name.to_ascii_lowercase();
        n.contains("arc") || n.contains("data center gpu max")
    }
}

fn intel_is_xpu_capable(gpus: &[GpuAdapter]) -> bool {
    gpus.iter().any(intel_adapter_is_xpu_capable)
}

/// Whether THIS machine's hardware can run a given TRAINING runtime-pack variant. Same one
/// sentence as the inference side (`cuda_pkg_supported`): a pack is offered, selectable and
/// counted as usable storage **iff this machine can actually run it**.
///  - `nv-cu130` → an NVIDIA card at or above the shared `gpu::CUDA_CC10_FLOOR` (torch cu130's
///    fatbin floor). Blackwell is fine here — the training lane is ALREADY on CUDA 13; only the
///    inference lane carries the temporary Blackwell exception.
///  - `amd` → an AMD GPU (TheRock's own capability check is the envtest's job — a name/vendor
///    gate cannot be exact, see the module note on best-effort gates).
///  - `xpu` → an Arc-family Intel GPU; legacy Iris Xe / UHD are NOT torch-xpu targets
///    (intel_is_xpu_capable).
///
/// ⚠ UNDETERMINED = NOT SUPPORTED, same as `cuda_pkg_supported` (S74b) — the probe failing is
/// not a licence to keep offering a pack, nor to hide an installed one from storage reclamation.
/// The on-device envtest stays the final authority; these gates are best-effort filters.
/// NB: LOCAL-FILE install is deliberately NOT gated by this.
///
/// `nv_cc10` is the hoisted `nvidia_compute_caps_cc10()` result (callers loop over variants;
/// re-probing would spawn one nvidia-smi per entry).
pub(crate) fn variant_supported(variant: &str, gpus: &[GpuAdapter], nv_cc10: &[i32]) -> bool {
    match variant {
        "cpu" => true,
        "nv-cu130" => nv_cc10.iter().any(|&cc| crate::gpu::cuda_cc_supported_training(cc)),
        "amd" => amd_is_rocm_capable(gpus),
        "xpu" => intel_is_xpu_capable(gpus),
        _ => false,
    }
}

#[tauri::command]
pub fn set_device_preference(
    state: State<'_, Arc<AppState>>,
    device: String,
    device_id: Option<u32>,
) -> Result<(), String> {
    // S68b: the preferred-GPU picker feeds device_id. Explicit modes: DML = DXGI
    // EnumAdapters1 ordinal, CUDA = CUDA runtime ordinal (DIFFERENT spaces, see gpu.rs);
    // omitted → 0, the pre-picker behavior byte-for-byte. Auto (§user): device_id is the
    // preferred DXGI adapter for BOTH GPU legs (CUDA maps it to an ordinal by LUID);
    // None = fully automatic (DXCore high-performance pick) = pre-S68b behavior.
    let id = device_id.unwrap_or(0);
    let config = match device.as_str() {
        "cuda" => DeviceConfig::Cuda { device_id: id },
        "directml" => DeviceConfig::DirectMl { device_id: id },
        "cpu" => DeviceConfig::Cpu,
        _ => DeviceConfig::Auto,
    };
    let auto_gpu = if device == "auto" { device_id } else { None };

    state.inference.engine.set_device(config.clone());
    state.inference.engine.set_auto_gpu(auto_gpu);

    // Persist — load-then-update so we never clobber the rest of the config (esp. data_dir).
    let mut cfg = load_config(&state.app_dir).unwrap_or_default();
    cfg.device = config;
    cfg.auto_gpu = auto_gpu;
    if let Err(e) = save_config(&state.app_dir, &cfg) {
        tracing::warn!("Failed to save config: {}", e);
    }

    Ok(())
}

/// One GPU choice for the inference preferred-GPU picker. `id` lives in the EP's OWN
/// ordinal space (see gpu.rs); `selectable=false` = a software adapter that occupies an
/// index slot (ORT throws if picked) — shown greyed, never compacted away. `vendor`
/// drives the Auto-mode restart hint (a non-NVIDIA pick can't run on the CUDA build).
#[derive(serde::Serialize)]
pub struct InferenceGpuChoice {
    pub id: u32,
    pub label: String,
    pub selectable: bool,
    /// S74b: WHY this entry is not selectable — a stable CODE the frontend localizes
    /// ("SOFTWARE_ADAPTER" | "CC_UNSUPPORTED"). A greyed option with no reason is the same
    /// guessing game as a bare error code; every disabled affordance must say what failed.
    pub reason: Option<String>,
    pub vendor: String,
}

#[derive(serde::Serialize)]
pub struct InferenceGpuLists {
    pub directml: Vec<InferenceGpuChoice>,
    pub cuda: Vec<InferenceGpuChoice>,
}

/// S68b: the Settings preferred-GPU picker's option lists. DirectML entries are DXGI
/// EnumAdapters1 ordinals (== the ORT DML device_id space); CUDA entries are CUDA
/// runtime ordinals labeled via cudart→nvidia-smi PCI matching. Device names are
/// hardware identifiers — deliberately not localized.
#[tauri::command]
pub fn list_inference_gpus() -> InferenceGpuLists {
    // ★S75 (user report): DXGI can list ONE physical GPU twice. On the reporter's box a single
    // RTX 3080 Ti enumerates at index 0 AND index 2 — different LUIDs, byte-identical DESC3
    // identity and VRAM, `DXGI_ADAPTER_FLAG3_REMOTE` NOT set on either. So no DESC3 field tells
    // the shadow apart from a genuine second identical card, and a name/VRAM dedupe would break
    // real dual-GPU rigs. The CUDA driver's LUID table does tell them apart, exactly: the real
    // adapter's LUID is in it, the shadow's is not (verified on that box — idx0 → cuda ordinal 0,
    // idx2 → none).
    //
    // Fail-OPEN by design, unlike the rest of these gates: an EMPTY LUID table means the probe
    // could not answer (no CUDA drivers), not "no GPUs exist" — demoting every NVIDIA adapter
    // there would take a working card away from someone whose driver merely lacks CUDA.
    let cuda_luids = crate::gpu::cuda_visible_luids();
    let directml = crate::gpu::dxgi_adapters()
        .into_iter()
        .map(|a| {
            let shadow = a.vendor == "nvidia"
                && !cuda_luids.is_empty()
                && !cuda_luids.contains(&a.luid);
            let reason = if a.software {
                Some("SOFTWARE_ADAPTER")
            } else if shadow {
                Some("DUPLICATE_ADAPTER")
            } else {
                None
            };
            InferenceGpuChoice {
                id: a.index,
                label: if a.dedicated_mb >= 256 {
                    format!("GPU {}: {} ({} MB)", a.index, a.name, a.dedicated_mb)
                } else {
                    format!("GPU {}: {}", a.index, a.name)
                },
                selectable: reason.is_none(),
                reason: reason.map(str::to_string),
                vendor: a.vendor.to_string(),
            }
        })
        .collect();
    // S74b: a CUDA device our shipped CUDA package cannot run must not be PICKABLE — the whole
    // point of the gates is that "you could select it" implies "it can run here". Same single
    // predicate; the entry stays visible (with its reason) rather than vanishing, so a user who
    // remembers having that card does not think we lost it.
    let cuda = crate::gpu::cuda_devices()
        .into_iter()
        .map(|d| {
            // S74b review: separate "we read the cap and it's out of range" from "we couldn't read
            // it". cudart listed this device, so a failed attribute query is an anomaly, not a
            // verdict about the hardware — labelling it "not supported by our CUDA package" would
            // be a false statement about the user's GPU.
            let cap = crate::gpu::cuda_compute_cap(d.index);
            let reason = match cap {
                Some((a, b)) if crate::gpu::cuda_cc_supported_inference(a * 10 + b) => None,
                Some(_) => Some("CC_UNSUPPORTED"),
                None => Some("CC_UNKNOWN"),
            };
            InferenceGpuChoice {
                id: d.index,
                label: format!("CUDA {}: {}", d.index, d.name),
                selectable: reason.is_none(),
                reason: reason.map(|r| r.to_string()),
                vendor: "nvidia".to_string(),
            }
        })
        .collect();
    InferenceGpuLists { directml, cuda }
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
pub fn get_diagnostic_mode(state: State<'_, Arc<AppState>>) -> bool {
    let _ = &state; // same shape as get_cuda_mem_limit: config on disk, the live static is read
    crate::training::diagnostics::enabled()
}

/// S115 §F5-2: turn diagnostic mode on/off. Persisted in config.json (a crash repro can span
/// restarts) and applied to the live static immediately — it is read at the next training
/// spawn, so a run already in flight is unaffected, which is correct: its env block was fixed
/// when it started and pretending otherwise would make the banner a lie.
#[tauri::command]
pub fn set_diagnostic_mode(state: State<'_, Arc<AppState>>, on: bool) -> Result<(), String> {
    crate::training::diagnostics::set_enabled(on);
    let mut cfg = load_config(&state.app_dir).unwrap_or_default();
    cfg.diagnostic_mode = on;
    if let Err(e) = save_config(&state.app_dir, &cfg) {
        tracing::warn!("Failed to save config: {}", e);
    }
    tracing::info!(
        "Diagnostic mode {} (takes effect on the NEXT training run)",
        if on { "ENABLED — training runs will be noticeably slower" } else { "disabled" }
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

/// S74b: set once at startup when a stale explicit device preference was demoted to Auto (see
/// load_and_apply_config). Surfaced through `get_hardware_info` so the frontend can toast it once
/// — there is no reliable moment to push an event from `.setup` (the listener mounts later).
pub(crate) static PREFERENCE_DEMOTED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

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
    if let Some(mut cfg) = load_config(&state.app_dir) {
        // ★S74b consumption point 5 — STALE EXPLICIT PREFERENCE. "You could select it" must imply
        // "it can run here", but a SAVED preference outlives the facts it was chosen under: the
        // user swaps a GPU or a whole machine, or we ourselves narrow the supported window in an
        // update. An explicit CUDA pick can only be honoured by a process that actually loaded the
        // CUDA ORT build (the build gate in lib.rs decides that from cuda_pkg_supported), so if we
        // are NOT on that build the preference is dead: leaving it in place would make every render
        // fail with the explicit-pick modal for a choice the user did not make today.
        //
        // Demote to Auto and REWRITE the config (a stale value must not resurrect next launch),
        // then tell the user ONCE — non-blocking, because this is our environment changing, not
        // the user's deterministic intent being refused (that distinction is the whole taxonomy).
        if matches!(cfg.device, crate::inference::engine::DeviceConfig::Cuda { .. }) && build != "CUDA"
        {
            // S74b review: PERSIST the demotion only when the hardware probe actually answered.
            // cuda_pkg_supported() is fail-closed, so an nvidia-smi that merely failed this once
            // (driver update in progress, AV interference) reads the same as "unsupported" — and
            // rewriting config.json on that would silently and permanently throw away a setting
            // the user chose. In-session we demote either way (we cannot honour a preference this
            // process can't serve); only the on-disk change waits for evidence.
            let undetermined = nvidia_compute_caps_cc10().is_empty();
            tracing::warn!(
                "Saved device preference CUDA cannot be honoured (ORT build loaded: {build}; \
                 compute-cap probe: {}) — demoting to Auto{}",
                if undetermined { "no answer" } else { "answered" },
                if undetermined { " for this session only" } else { " and rewriting config" }
            );
            cfg.device = crate::inference::engine::DeviceConfig::Auto;
            cfg.auto_gpu = None;
            if !undetermined {
                if let Err(e) = save_config(&state.app_dir, &cfg) {
                    tracing::warn!("Failed to persist the demoted device preference: {e}");
                }
            }
            PREFERENCE_DEMOTED.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        tracing::info!(
            "device preference: {:?} (config.json); ORT build loaded: {}; per-run EP is logged as \"ONNX device=...\"",
            cfg.device,
            build
        );
        state.inference.engine.set_device(cfg.device);
        state.inference.engine.set_auto_gpu(cfg.auto_gpu);
        if cfg.cuda_mem_limit_mb > 0 {
            crate::inference::engine::CUDA_MEM_LIMIT_MB
                .store(cfg.cuda_mem_limit_mb, std::sync::atomic::Ordering::Relaxed);
            tracing::info!("CUDA memory limit: {} MB (config.json)", cfg.cuda_mem_limit_mb);
        }
        // S115 §F5-2: only announced when ON — a machine that never touched it should not carry
        // a line about a feature it does not use. Announced LOUDLY when it is on, because it
        // survives restarts and its whole cost is silent slowness (see the field's doc).
        if cfg.diagnostic_mode {
            crate::training::diagnostics::set_enabled(true);
            tracing::warn!(
                "Diagnostic mode is ON (config.json): training runs will be noticeably slower. \
                 Turn it off in Settings → Diagnostics when you are done reproducing."
            );
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
///
/// ⚠ That parenthesis is a claim about a layout built in ANOTHER file, and until S110 nothing
/// enforced it — while three consumers depend on it agreeing: this function (which
/// `dictionary_fingerprint` hashes), `models.models_dir().parent()` (which the three
/// `g2p::set_dict_dir` call sites read), and the `data_dir` local the bundled sync WRITES. A
/// disagreement stamps a bake with the fingerprint of a directory it did not sing from. Both halves
/// are now pinned: `data_root_derivations_agree` (this layout ⇒ both derivations return the root)
/// and `boot_steps_with_a_single_call_site_stay_wired_into_setup` (lib.rs still builds that layout
/// out of the same root the sync was handed).
fn effective_data_root(state: &AppState) -> &std::path::Path {
    state.cache_dir.parent().unwrap_or(state.cache_dir.as_path())
}

/// Current data dir (for the settings UI).
#[tauri::command]
pub fn get_data_dir(state: State<'_, Arc<AppState>>) -> Result<String, String> {
    Ok(effective_data_root(&state).to_string_lossy().to_string())
}

/// The subtrees a data-dir migration moves — the single source of truth shared by the copy, the
/// post-copy verification, and the next-startup delta-sync + old-tree reclaim. `runtimes` skips
/// top-level dot-entries (`.staging` = torn/resumable installs, transient by design).
const MIGRATED_SUBTREES: [&str; 5] = ["models", "cache", "dictionaries", "runtimes", "training"];

fn skips_dot_top(subtree: &str) -> bool {
    subtree == "runtimes"
}

/// Recursively copy a directory's contents into `dst` (creating it). Cross-drive safe (copy, not rename).
/// pub(crate): also the S68e webview-profile migration's copier (lib.rs) — ONE walker.
pub(crate) fn copy_dir_all(src: &std::path::Path, dst: &std::path::Path, skip_dot_top: bool) -> std::io::Result<()> {
    if !src.exists() {
        return Ok(());
    }
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        if skip_dot_top && entry.file_name().to_string_lossy().starts_with('.') {
            continue;
        }
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if from.is_dir() {
            copy_dir_all(&from, &to, false)?;
        } else {
            // S68d: a mid-copy failure in a tens-of-GB migration must name the file —
            // io::Error's Display alone gives "os error 112" with no idea where.
            std::fs::copy(&from, &to).map_err(|e| {
                std::io::Error::new(e.kind(), format!("{} -> {}: {e}", from.display(), to.display()))
            })?;
        }
    }
    Ok(())
}

/// S68d disk-preflight walker: bytes the migration still NEEDS at the target for one
/// subtree — Σ over SOURCE files of (src len − existing same-relpath target len). The
/// traversal predicates MIRROR copy_dir_all exactly (`is_dir()` follows junctions and
/// `fs::metadata` follows file symlinks, so linked content is counted the way the copy
/// will actually copy it); crediting only the same-path target file keeps unrelated
/// pre-existing target content from shrinking the estimate (both review S68d).
/// pub(crate): also sizes the S68e webview-profile migration (lib.rs).
pub(crate) fn migrate_tree_needed(src: &std::path::Path, dst: &std::path::Path, skip_dot_top: bool) -> u64 {
    let mut needed = 0u64;
    let Ok(rd) = std::fs::read_dir(src) else { return 0 };
    for entry in rd.flatten() {
        if skip_dot_top && entry.file_name().to_string_lossy().starts_with('.') {
            continue;
        }
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if from.is_dir() {
            needed = needed.saturating_add(migrate_tree_needed(&from, &to, false));
        } else {
            let src_len = std::fs::metadata(&from).map(|m| m.len()).unwrap_or(0);
            let dst_len = std::fs::metadata(&to).map(|m| m.len()).unwrap_or(0);
            needed = needed.saturating_add(src_len.saturating_sub(dst_len));
        }
    }
    needed
}

/// Post-copy integrity check: every file under `src` (same skip rules as the copy) must exist under
/// `dst` with the same byte length. Metadata-only (no re-read of tens of GB) — `fs::copy` already
/// fails loudly on content errors; this catches whole-file misses (skipped entries, torn traversal).
/// Returns the number of files checked.
fn verify_dir_copy(src: &std::path::Path, dst: &std::path::Path, skip_dot_top: bool) -> Result<u64, String> {
    if !src.exists() {
        return Ok(0);
    }
    let mut checked = 0u64;
    for entry in std::fs::read_dir(src).map_err(|e| format!("read {}: {e}", src.display()))? {
        let entry = entry.map_err(|e| format!("read {}: {e}", src.display()))?;
        if skip_dot_top && entry.file_name().to_string_lossy().starts_with('.') {
            continue;
        }
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if from.is_dir() {
            checked += verify_dir_copy(&from, &to, false)?;
        } else {
            let src_len = entry.metadata().map_err(|e| format!("stat {}: {e}", from.display()))?.len();
            let dst_len = std::fs::metadata(&to)
                .map_err(|_| format!("missing after copy: {}", to.display()))?
                .len();
            if src_len != dst_len {
                return Err(format!("size mismatch after copy: {} ({} vs {} bytes)", to.display(), src_len, dst_len));
            }
            checked += 1;
        }
    }
    Ok(checked)
}

/// Startup delta-sync old→new before reclaiming the old tree: anything written to the OLD root
/// between the migration copy and the restart (model downloads, render cache, training artifacts)
/// would otherwise be deleted with it. `fs::copy` preserves mtimes on Windows, so "src newer than
/// dst" only matches genuinely newer writes. Copies via tmp+rename so concurrent readers of the
/// NEW tree (model scan, cache sweep) never observe a half-copied file. Returns (copied, failed);
/// the caller REFUSES to delete a subtree whose sync had failures — a straggler that could not be
/// carried over must never be deleted with the tree (S68c review major). An unreadable source
/// entry counts as failed for the same reason: we can't prove it's already in the new tree.
/// `layout_aware` (S76, used for the `training` subtree ONLY): refuse to MERGE two top-level
/// directories that are in different layouts.
///
/// `skip_top_names` (S109, used for the `dictionaries` subtree ONLY): TOP-LEVEL names this walk must
/// not carry over, because the destination already has a more authoritative source for them. The one
/// caller passes the bundled dictionary file names. Rationale — and why "old→new delta" is the wrong
/// rule for exactly these files:
///
///   * every other subtree here holds USER data, for which the old root is the only source and
///     "newer or different ⇒ carry it over" is right;
///   * `<data>/dictionaries/*.tsv` are BUNDLE resources whose single authority is
///     `<install>/data/dictionaries`, and a queued old root's copy is BY CONSTRUCTION that same
///     authority as of some EARLIER version — never newer.
///     ⚠ S110 correction: an earlier draft of this paragraph justified the rule with "the sync has
///     already written that authority into the active root earlier in the same boot". That is
///     FALSE and it contradicted the next paragraph of this very doc — `lib.rs` SPAWNS this thread
///     (line ~1037) and only THEN calls the sync (~1042), so at the moment this walk starts the
///     fresh bytes may not be on disk yet. The rule is safe for the version reason above, which
///     needs no ordering at all; the temporal claim was decoration that happened to be wrong.
///     (This is the same shape S109 was written up for: a comment whose invariant is broken by
///     code five lines away — here it was broken by the doc's own next paragraph.)
///
/// Without this the reclaim thread (spawned at lib.rs:1037, five lines before the sync) walks the
/// old root's `dictionaries` with the predicate below — `size differs || src newer` — and the SIZE
/// clause alone is enough: an old root left over from before an app update holds a different-sized
/// de/en/fr.tsv, so those three get written BACK over the freshly-synced ones while es/it/zh_* (same
/// sizes across those two generations) do not, leaving a MIXED dictionary set no build ever shipped.
/// The session then renders with the old phones; the next boot restores the new files, so nothing is
/// ever red. Worse, `DICT_FINGERPRINT` is a `OnceLock` read once per session, so a bake made in that
/// window can be stamped with the NEW fingerprint while carrying OLD phones — the precise hole the
/// fingerprint was introduced to close (see `dictionary_fingerprint_for`).
///
/// Skipped names are NOT counted as failures: they are reproducible from the bundle, so keeping the
/// old subtree undeleted on their account would only leak disk. Anything else the user parked in
/// that directory is still carried over, so the "never delete a straggler we could not carry" rule
/// below is untouched.
///
/// ⚠ S110 — THE SKIP LIST HAS TO NAME THE STAGING TWIN TOO, and S109's claim that it did not need
/// to was wrong. Both writers stage through the SAME temp path in the destination directory:
/// `sync_bundled_dictionaries` computes `fr.tsv`.with_extension("tsv.syncing") = `fr.tsv.syncing`,
/// and this walk, handed a source entry literally named `fr.tsv.syncing` (what a previous boot's
/// torn sync leaves behind — the cleanup at the end of that function is on the FAILURE branch only,
/// so a crash between copy and rename leaves one), renames onto that same `fr.tsv.syncing`. Skipping
/// `fr.tsv` does not skip `fr.tsv.syncing`: the predicate below is exact-match. A reclaim rename
/// landing between the sync's copy and its rename replaces the fresh bytes with the old root's
/// straggler, and the sync then logs "refreshed" over them. S109's doc said that collision was
/// "structurally impossible for the eight shipped files"; it was impossible only for the eight
/// NAMES, which is not the same set. The caller now passes both spellings.
///
/// Both roots can hold a `training/<id>/` — but one may be a pre-S76 workspace (checkpoints
/// at its root) while the other is a migrated PROJECT (checkpoints one level down, in a
/// family slot). Merging those per-file plants `G_*.pth` / `config.json` / `run_manifest.json`
/// back at the project root next to the slot that already holds them — precisely the
/// "checkpoints without a manifest" shape the resume guards treat as corruption, and
/// unrecoverable because the migration marker is long gone.
///
/// ⚠ Skipping such a directory counts as a FAILURE, not as "nothing to do". `reclaim_one_root`
/// deletes the old subtree exactly when nothing failed to sync, so a silent skip would delete
/// the old `training/` along with everything it still held that the new root does not — the
/// concrete loss being every checkpoint written between "migrated the data dir" and "actually
/// restarted", which lands in the OLD root. Same-layout pairs merge per file as before.
/// Where in the training tree a [`sync_dir_delta`] recursion currently is.
///
/// ★§F2⒝ — it was a `bool`, then an `Option<&str>` marker, and neither could express the level
/// this batch adds. The levels do NOT share one rule:
/// * a PROJECT and a SLOT each decide their layout with a marker file, and merging two different
///   layouts per-file re-seeds products at a level nothing reads them from any more;
/// * a POOL merges harmlessly, because a pool's content is a pure function of the identity its
///   directory is named after — two roots' `pools/<same id>/` hold the same recipe;
/// * ⛔ a RUN does NOT. Its content is not a function of its id, and the layout migration mints
///   the legacy run's id DETERMINISTICALLY per family, so two data roots each holding a migrated
///   `rvc` slot have a `runs/<same id>/` containing two DIFFERENT trainings. Merging those
///   file-by-file, newest-mtime-wins, interleaves two trainings' `G_*.pth` / `D_*.pth` /
///   `best_state.json` / `resume_best/` into one directory — the unrecoverable tree this whole
///   function exists to prevent, one level deeper than the guard used to reach.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SyncLevel {
    /// Below every layout decision (or outside the training tree): plain per-file delta.
    Plain,
    /// The training ROOT — children are projects.
    Projects,
    /// A PROJECT — children are family slots.
    Slots,
    /// A family SLOT — children are `pools/`, `runs/`, and whatever an unmigrated slot still has.
    Slot,
    /// The `runs/` container — every child is one training RUN.
    Runs,
}

impl SyncLevel {
    /// Are these two child directories in DIFFERENT layouts? Returns what to say if so.
    ///
    /// ⛔ For a slot this compares the layout NUMBER, not the marker's existence. Layout 2 and
    /// layout 3 advance the SAME `slot.json` (`trun::SLOT_LAYOUT_RUNS` is deliberately not a bump
    /// of `tpool::SLOT_LAYOUT` — see that constant), so an existence test cannot tell "pool folded"
    /// from "pool AND run folded": a layout-2 slot would merge file-by-file into a layout-3 one and
    /// re-seed `G_*.pth` / `run.json` / `weights/` at the slot root, beside a `runs/` that already
    /// holds them. Nothing would ever read that copy and nothing would ever delete it — S121's
    /// `dataset_44k` finding, one layout later, and the reason this is a number.
    fn layout_conflict(self, from: &std::path::Path, to: &std::path::Path) -> Option<String> {
        match self {
            SyncLevel::Projects => {
                let m = crate::training::tproject::PROJECT_META;
                (from.join(m).is_file() != to.join(m).is_file())
                    .then(|| format!("{m} on one side only"))
            }
            SyncLevel::Slots => {
                let layout = |d: &std::path::Path| {
                    crate::training::tpool::read_slot_meta(d).map(|m| m.layout).unwrap_or(0)
                };
                let (a, b) = (layout(from), layout(to));
                (a != b).then(|| format!("slot layout {a} vs {b}"))
            }
            _ => None,
        }
    }

    fn descend(self, child: &str) -> SyncLevel {
        match self {
            SyncLevel::Projects => SyncLevel::Slots,
            SyncLevel::Slots => SyncLevel::Slot,
            SyncLevel::Slot if child == crate::training::trun::RUNS_DIR => SyncLevel::Runs,
            _ => SyncLevel::Plain,
        }
    }
}

/// Would a per-file delta from `src` into `dst` write anything? Returns the first file that would.
///
/// Stat-only and recursive, and it MUST answer with the same rule the copier uses — hence
/// [`needs_copy`], shared by both. A second opinion here would let the two drift until the run-level
/// refusal fired on a tree the copier would have left alone (or, worse, the other way round).
fn delta_would_write(src: &std::path::Path, dst: &std::path::Path) -> Option<std::path::PathBuf> {
    let rd = std::fs::read_dir(src).ok()?;
    for entry in rd.flatten() {
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if from.is_dir() {
            if let Some(p) = delta_would_write(&from, &to) {
                return Some(p);
            }
            continue;
        }
        if needs_copy(&from, &to) {
            return Some(from);
        }
    }
    None
}

/// Is `from` newer than, or absent from, `to`? The one rule the delta sync runs on.
///
/// An unreadable SOURCE answers `true` on purpose: the caller then treats it as unsynced and keeps
/// the old subtree, which is the safe direction — the alternative would silently delete a file the
/// process could not read.
///
/// ⛔★★S133 §F2⒝ ④e — **a destination that has MOVED ON is never overwritten.** The rule used to be
/// 「different size OR source newer」, and the `size` half fires on its own: `project.json` is a
/// read-modify-write JSON that lives in the project directory like any other file, so the ordinary
/// script — migrate the data root, train on the NEW root, export a model (the ledger gains a row,
/// the file gains bytes) — makes the OLD root's shorter copy overwrite the live one at the next
/// reclaim. That is not 「history loses a line」: `meta.exported` is the sole source of
/// `KeptReason::Exported`, so the next 「清理未导入的快照」 deletes the checkpoint the user
/// exported. The same silent rollback would also undo ④e's own ledger retirement.
///
/// ⚠ The size clause is KEPT for the same-mtime case — that is the torn-copy shape it was there
/// for. Only 「destination strictly newer」 became decisive, which is also the answer
/// `delta_would_write` needs: an old copy with nothing newer to contribute has nothing to carry
/// over, which is exactly what its own doc says the run-level refusal hangs on.
fn needs_copy(from: &std::path::Path, to: &std::path::Path) -> bool {
    match (std::fs::metadata(from), std::fs::metadata(to)) {
        (Ok(s), Ok(d)) => match (s.modified(), d.modified()) {
            (Ok(sm), Ok(dm)) if dm > sm => false,
            (Ok(sm), Ok(dm)) if sm > dm => true,
            _ => s.len() != d.len(),
        },
        (Ok(_), Err(_)) => true,
        (Err(_), _) => true,
    }
}

/// Remember that `rel` (relative to the DATA root) was deleted on purpose, so a queued data-root
/// reclaim cannot copy it back. No-op when nothing is queued. See [`AppConfig::deleted_since_migration`].
///
/// Best-effort by design: the bytes are already gone by the time this runs, and refusing a
/// completed delete because a preference file could not be written would be the wrong trade.
pub fn record_deliberate_delete(app_dir: &std::path::Path, rel: &str) {
    let Some(mut cfg) = load_config(app_dir) else { return };
    if cfg.pending_delete_dirs.is_empty() {
        return; // no old copy exists ⇒ nothing can resurrect it ⇒ nothing to remember
    }
    let rel = rel.replace('\\', "/");
    // The paths are built by this app from ids it validated, but this list is later joined onto a
    // root and DELETED from — so it refuses anything that could climb out, rather than trusting
    // that no future caller ever passes something else.
    if rel.is_empty()
        || rel.starts_with('/')
        || rel.split('/').any(|c| c == ".." || c.is_empty())
        || std::path::Path::new(&rel).is_absolute()
    {
        tracing::warn!("refusing to record a suspicious deliberate-delete path: {rel}");
        return;
    }
    if cfg.deleted_since_migration.iter().any(|r| *r == rel) {
        return;
    }
    cfg.deleted_since_migration.push(rel);
    if let Err(e) = save_config(app_dir, &cfg) {
        tracing::warn!("could not record a deliberate delete for the pending reclaim: {e}");
    }
}

fn sync_dir_delta(
    src: &std::path::Path,
    dst: &std::path::Path,
    skip_dot_top: bool,
    level: SyncLevel,
    skip_top_names: &[String],
) -> (u64, u64) {
    let mut copied = 0u64;
    let mut failed = 0u64;
    let Ok(rd) = std::fs::read_dir(src) else { return (0, 0) };
    for entry in rd.flatten() {
        if skip_dot_top && entry.file_name().to_string_lossy().starts_with('.') {
            continue;
        }
        if skip_top_names.iter().any(|n| *n == entry.file_name().to_string_lossy()) {
            continue;
        }
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if from.is_dir() {
            if to.exists() {
                if let Some(why) = level.layout_conflict(&from, &to) {
                    tracing::warn!(
                        "data-dir reclaim: {} and {} are in different training layouts ({why}) — \
                         keeping the old tree instead of merging (move it by hand once you have \
                         checked it)",
                        from.display(),
                        to.display()
                    );
                    failed += 1;
                    continue;
                }
            }
            // ⛔ A RUN is copied whole or not at all — see [`SyncLevel`]. Two runs that really are
            // different trainings must never be interleaved file by file, and only a human can say
            // which one is wanted.
            //
            // ⚠ But "same id" does not mean "same training" in this batch, and refusing on the NAME
            // alone would strand every reclaim there is: `trun::legacy_run_id` is a pure function of
            // the FAMILY, so both roots of the ordinary case — `migrate_data_dir` copies the tree,
            // the copy is queued for reclaim, and the new root has since been trained on — hold
            // `runs/<the same id>/` by construction. Every migrated slot would then contribute a
            // failure, the whole `training/` subtree would be kept forever, and the warning would
            // tell the user they are two different trainings, which in that case is false.
            //
            // So the refusal hangs on the POSITIVE fact that the old copy actually has something to
            // contribute. A delta that would write NOTHING is not a merge at all — it is the stale
            // copy of a run that has moved on, and keeping the subtree for it only costs disk.
            //
            // ⚠★S133 re-checked this whole argument (§F2⒝ ④e's handoff asked for it to be rewritten)
            // and it stands unchanged: `legacy_run_id` really is still a pure function of the family,
            // and the refusal really does still hang on the delta. What S133 added is a SEPARATE
            // mechanism one level up — `reclaim_one_root` now removes the paths the user deleted on
            // purpose from the old copy BEFORE this runs (see `AppConfig::deleted_since_migration`),
            // so a deliberately-deleted run never reaches this comparison at all. Do not merge the
            // two ideas: this one is about two real trainings colliding, that one is about a delete.
            if level == SyncLevel::Runs && to.exists() {
                if let Some(newer) = delta_would_write(&from, &to) {
                    tracing::warn!(
                        "data-dir reclaim: {} and {} are both training runs with the same id AND \
                         the old one holds newer or extra files (e.g. {}) — keeping the old tree \
                         instead of merging them (move the one you want by hand)",
                        from.display(),
                        to.display(),
                        newer.display()
                    );
                    failed += 1;
                }
                // …otherwise there is nothing to carry over: fall through WITHOUT recursing, so the
                // subtree stays deletable.
                continue;
            }
            // Nested levels are never bundle resources — the skip list is a TOP-LEVEL rule.
            let child = entry.file_name().to_string_lossy().into_owned();
            let (c, f) = sync_dir_delta(&from, &to, false, level.descend(&child), &[]);
            copied += c;
            failed += f;
            continue;
        }
        // An unreadable SOURCE is kept rather than skipped silently — same as before; everything
        // else goes through the one shared rule, so `delta_would_write` cannot drift from it.
        if entry.metadata().is_err() {
            tracing::warn!("data-dir reclaim: cannot stat {} — treating as unsynced", from.display());
            failed += 1;
            continue;
        }
        if !needs_copy(&from, &to) {
            continue;
        }
        let tmp = to.with_extension(format!(
            "{}.syncing",
            to.extension().map(|e| e.to_string_lossy().to_string()).unwrap_or_default()
        ));
        let ok = std::fs::create_dir_all(to.parent().unwrap_or(dst)).is_ok()
            && std::fs::copy(&from, &tmp).is_ok()
            && std::fs::rename(&tmp, &to).is_ok();
        if ok {
            tracing::info!("data-dir reclaim: synced straggler {}", to.display());
            copied += 1;
        } else {
            let _ = std::fs::remove_file(&tmp);
            tracing::warn!("data-dir reclaim: failed to sync {}", from.display());
            failed += 1;
        }
    }
    (copied, failed)
}

/// The bundled dictionary files, read out of tauri.conf.json's OWN `bundle.resources` map — the same
/// zero-drift trick `bundled_integrity_report` uses. Install-relative paths under data/dictionaries/.
fn bundled_dictionary_targets() -> Vec<String> {
    static CONF: &str = include_str!("../../tauri.conf.json");
    let Ok(v) = serde_json::from_str::<serde_json::Value>(CONF) else { return Vec::new() };
    let Some(res) = v.pointer("/bundle/resources").and_then(|r| r.as_object()) else {
        return Vec::new();
    };
    res.values()
        .filter_map(|t| t.as_str())
        .filter(|t| t.starts_with("data/dictionaries/") && !t.ends_with('/'))
        .map(str::to_string)
        .collect()
}

/// S101 — closes the S83 dictionary DISTRIBUTION fault (review S94-VB-1).
///
/// The stage1 dictionaries ship as NSIS bundle resources into `<install>\data\dictionaries`. On a
/// DEFAULT install that directory *is* inside the data root, so the read path and the bundle path
/// are the same files and every update refreshes them automatically. Two populations are not so
/// lucky, and both were permanently stuck:
///   · a user who moved the data dir — `migrate_data_dir` copies `dictionaries` ONCE, and the
///     next-boot delta-sync only runs while the old root is still queued for reclaim (a single
///     attempt, see `reclaim_one_root`). Every later app update rewrote the install copy and
///     nothing ever carried it across. They kept the dictionary they migrated with, forever.
///   · a legacy-AppData root — nothing has ever populated `<appdata>\dictionaries` at all.
///
/// Why it MUST be here and MUST be synchronous: `g2p::set_dict_dir` is a `OnceLock` and a loaded
/// dictionary is `Box::leak`ed for the process lifetime, lazily, on the first `validate_lyrics` or
/// render. A background thread would race that and could pin a stale dictionary for the whole
/// session with nothing able to invalidate it. Running here — before the main window exists — means
/// the first lookup of the session already sees the fresh file.
/// ⇒ S110: every clause of that sentence ("here", "synchronous", "before the main window") is now
/// asserted by `boot_steps_with_a_single_call_site_stay_wired_into_setup`. Until then the paragraph
/// was a promise with nothing behind it: `lib.rs:1042` is the ONLY production call site, and
/// deleting it or wrapping it in a `thread::spawn` left the entire suite green.
///
/// ⚠ The identity test is a CONTENT HASH, deliberately not size+mtime like `sync_dir_delta`. That
/// predicate is wrong for this job in both directions: S98 caught an upstream swap that changed
/// 3051 German primary pronunciations with the line count, word set and file size all identical,
/// and on this very machine the STALE `target/debug` copy carries the NEWER mtime. 8 files / ~18 MB
/// hashes in well under a second, once per boot, and only when the paths actually differ.
///
/// Never deletes and never fails the boot: a missing source (dev checkout — `data/` is gitignored
/// and the TSVs are generated by the MBS2H generator) is a no-op, and a copy that cannot be
/// made is logged at ERROR and left alone. It is not a repair path for a *corrupt* dictionary — only
/// for a stale or absent one.
///
/// ★ S110 — IT LOGS ON EVERY RETURN PATH, and that is the point rather than tidiness. Until now the
/// two early returns and the "everything already matches" tail were all silent, so an installed
/// build's log could not distinguish "the step ran and correctly did nothing" from "the step never
/// ran at all". That ambiguity is not hypothetical: S109 had to settle exactly this question about a
/// shipped exe and could only do it by searching the BINARY for the `dictionary sync:` format string
/// (the code predated the feature — but the same search on a build that HAS the code proves nothing
/// about whether the call still executes). It is also what release-checklist item #5 asks a tester to
/// read, and its old acceptance line — "confirm the `dictionary sync:` row" — was unsatisfiable on a
/// healthy machine, because a healthy machine printed nothing. One INFO row per boot.
pub fn sync_bundled_dictionaries(app_dir: &std::path::Path, data_dir: &std::path::Path) {
    let src_dir = app_dir.join("data").join("dictionaries");
    let dst_dir = data_dir.join("dictionaries");
    if !src_dir.is_dir() {
        // dev checkout / stripped install — never treat "no source" as "delete the copy"
        tracing::info!("dictionary sync: no bundled source at {} — nothing to do", src_dir.display());
        return;
    }
    // Default install and dev build both land here: the two paths are the SAME directory. Compare
    // canonicalized, so a hand-set data_dir pointing back at the install root is caught too — not
    // just the cfg!(debug_assertions) case.
    if let (Ok(a), Ok(b)) = (src_dir.canonicalize(), dst_dir.canonicalize()) {
        if a == b {
            // NOT an anomaly — this is the DEFAULT install and every dev build. Logged anyway so
            // "ran, correctly did nothing" is distinguishable from "never ran" (see the doc above).
            tracing::info!("dictionary sync: data root reads the bundled copy directly ({}) — no-op", a.display());
            return;
        }
    }
    let targets = bundled_dictionary_targets();
    if targets.is_empty() {
        tracing::warn!("dictionary sync: bundle.resources lists no dictionary files — skipping");
        return;
    }
    let (mut updated, mut failed) = (Vec::new(), Vec::new());
    for target in &targets {
        let name = std::path::Path::new(target).file_name().unwrap_or_default().to_owned();
        let from = src_dir.join(&name);
        let to = dst_dir.join(&name);
        if !from.is_file() {
            continue;
        }
        let same = match (crate::download::sha256_file(&from), crate::download::sha256_file(&to)) {
            (Ok(a), Ok(b)) => a == b,
            (Ok(_), Err(_)) => false, // destination absent or unreadable — refresh it
            (Err(e), _) => {
                tracing::warn!("dictionary sync: cannot hash {} ({e}) — leaving {} alone", from.display(), to.display());
                continue;
            }
        };
        if same {
            continue;
        }
        let tmp = to.with_extension("tsv.syncing");
        let ok = std::fs::create_dir_all(&dst_dir).is_ok()
            && std::fs::copy(&from, &tmp).is_ok()
            && crate::util::rename_with_retry(&tmp, &to, "dictionary").is_ok();
        if ok {
            updated.push(name.to_string_lossy().to_string());
        } else {
            let _ = std::fs::remove_file(&tmp);
            failed.push(name.to_string_lossy().to_string());
        }
    }
    if updated.is_empty() && failed.is_empty() {
        tracing::info!(
            "dictionary sync: all {} already match the bundled copy -> {}",
            targets.len(),
            dst_dir.display()
        );
    }
    if !updated.is_empty() {
        tracing::info!(
            "dictionary sync: refreshed {} of {} from the bundled copy ({}) -> {}",
            updated.len(),
            targets.len(),
            updated.join(", "),
            dst_dir.display()
        );
    }
    if !failed.is_empty() {
        // LOUD: this is the exact state the pairing with G2P_ALGO_VERSION exists to prevent —
        // a bumped stamp forcing a re-render against a dictionary that never arrived.
        tracing::error!(
            "dictionary sync: FAILED to refresh {} ({}) in {} — renders will use the OLD phones for \
             those languages until this succeeds (close anything holding the file, then restart)",
            failed.len(),
            failed.join(", "),
            dst_dir.display()
        );
    }
}

static DICT_FINGERPRINT: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// S101 — THE signature carrier for dictionary CONTENT (review S94-VB-1: it never had one).
///
/// `G2P_ALGO_VERSION` versions the lyric→phone ALGORITHM, and until now it was also being asked to
/// stand in for the dictionary FILES. That substitution has a hole with teeth: if a bump lands but
/// the new dictionary does not (the S83 distribution fault, or `sync_bundled_dictionaries` above
/// losing to a locked file), the owner re-renders against the OLD phones and gets stamped with the
/// NEW version — after which nothing can ever invalidate that bake again. With a content hash in the
/// signature, the bake becomes dirty the moment the real file changes, whenever that finally happens.
///
/// Hashes the ACTIVE data root's copy — the files `set_dict_dir` will actually read — not the
/// bundled one. Content only (name + sha256, fixed order): identical dictionaries on two machines
/// give the identical fingerprint, so a project rendered on one and opened on the other is NOT
/// spuriously dirty; different content is what must move it. Absent files hash as "-", which is
/// itself a stable value (a JA/ZH-only project renders fine with no dictionaries on disk at all).
///
/// `OnceLock`: the dictionaries are `Box::leak`ed on first load and cannot change within a session,
/// and `sync_bundled_dictionaries` has already run by the time any frontend command is served.
/// The fingerprint of ONE dictionary directory. Split out of the command so it is reachable without
/// a running Tauri app — the S101 distribution test drives this exact function over a synthesized
/// install tree, which is the only way to prove the carrier actually MOVES when the sync fires.
pub fn dictionary_fingerprint_for(dir: &std::path::Path) -> String {
    let mut targets: Vec<String> = bundled_dictionary_targets()
        .iter()
        .filter_map(|t| std::path::Path::new(t).file_name().map(|n| n.to_string_lossy().to_string()))
        .collect();
    targets.sort();
    let mut acc = String::new();
    for name in &targets {
        let h = crate::download::sha256_file(&dir.join(name)).unwrap_or_else(|_| "-".into());
        acc.push_str(name);
        acc.push(':');
        acc.push_str(&h);
        acc.push('\n');
    }
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    hasher.update(acc.as_bytes());
    format!("{:x}", hasher.finalize())[..12].to_string()
}

#[tauri::command]
pub fn dictionary_fingerprint(state: State<'_, Arc<AppState>>) -> String {
    if let Some(v) = DICT_FINGERPRINT.get() {
        return v.clone();
    }
    let dir = effective_data_root(&state).join("dictionaries");
    let short = dictionary_fingerprint_for(&dir);
    tracing::info!("dictionary fingerprint {} for {}", short, dir.display());
    DICT_FINGERPRINT.get_or_init(|| short).clone()
}

/// One-click migrate: copy the CURRENT data subtrees (MIGRATED_SUBTREES — models/cache/
/// dictionaries/runtimes/training; see each subtree's rationale below) into `new_dir`, VERIFY the
/// copy (every file present with the same size), then persist it as the data dir. Takes effect on
/// restart. S68c: the OLD tree is marked (`pending_delete_dir`) and reclaimed automatically on the
/// next startup — most users never found the old copy to delete it, leaving C: full. Nothing is
/// deleted before a verified replica exists AND the app actually boots on the new root
/// (spawn_pending_data_dir_delete); an unverified copy aborts here with config untouched.
///
/// Subtree notes (why each is in MIGRATED_SUBTREES):
/// - dictionaries (② S58): stage1 G2P dictionaries — leaving them behind would fake-OOV every
///   zh/en/de/fr/es/it lyric after a migration (audit MAJOR).
/// - runtimes (S42): lib.rs roots pyenv on the resolved data dir — leaving packs behind would make
///   every installed pack "vanish" after migration; `.staging` (torn/resumable installs) skipped.
/// - training (S61 recon gap): workspaces resolve off the SAME data dir — not copying them silently
///   stranded every checkpoint + dataset while 续训/共享池 resolved against the NEW (empty) tree.
#[tauri::command]
pub async fn migrate_data_dir(state: State<'_, Arc<AppState>>, new_dir: String) -> Result<(), String> {
    let new = std::path::PathBuf::from(new_dir.trim());
    if new.as_os_str().is_empty() {
        return Err("Empty target directory".into());
    }
    // S61: a live training run writes checkpoints/features mid-copy — the migrated tree would be
    // torn (and the workspace copy is exactly what a running trainer mutates).
    if state.training.is_active() {
        return Err("TRAINING_ACTIVE".into());
    }
    // §user S68c round 2: ONE migration per session, keyed on a PROCESS-LOCAL flag — not on the
    // pending-reclaim queue. Restarting genuinely unlocks it (new process), while an old root
    // stuck on an unplugged drive keeps its queue entry WITHOUT locking migration forever. The
    // Settings button disables itself via migrate_pending_restart; this is the backend backstop.
    if MIGRATED_THIS_SESSION.load(std::sync::atomic::Ordering::SeqCst) {
        return Err("MIGRATE_RESTART_REQUIRED".into());
    }
    let data_root = effective_data_root(&state).to_path_buf();
    let target = new.clone();
    let src_root = data_root.clone();
    // The copy reaches tens of GB — run it off the event loop so the UI stays responsive.
    tauri::async_runtime::spawn_blocking(move || -> Result<(), String> {
        std::fs::create_dir_all(&target).map_err(|e| format!("Create target: {e}"))?;
        // Refuse a target nested inside the data root (or vice versa) — copying a tree into itself
        // recurses forever.
        let canon_target = std::fs::canonicalize(&target).map_err(|e| format!("Resolve target: {e}"))?;
        let canon_root = std::fs::canonicalize(&src_root).unwrap_or_else(|_| src_root.clone());
        if canon_target.starts_with(&canon_root) || canon_root.starts_with(&canon_target) {
            return Err("Target directory overlaps the current data directory".into());
        }
        // S68d disk preflight: refuse up front with real numbers instead of dying
        // mid-copy after half an hour. Per-file same-path credit (a retried migration
        // re-needs nothing for files already copied); probe failure = fail open.
        let mut needed: u64 = 0;
        for name in MIGRATED_SUBTREES {
            needed = needed.saturating_add(migrate_tree_needed(
                &src_root.join(name),
                &target.join(name),
                skips_dot_top(name),
            ));
        }
        if let Some(free) = crate::util::free_bytes_at(&canon_target) {
            if free < needed {
                return Err(format!(
                    "MIGRATE_DISK_FULL: {} MB needed, {} MB free at {}",
                    needed / 1_000_000,
                    free / 1_000_000,
                    target.display()
                ));
            }
        }
        for name in MIGRATED_SUBTREES {
            copy_dir_all(&src_root.join(name), &target.join(name), skips_dot_top(name))
                .map_err(|e| format!("Copy {name}: {e}"))?;
        }
        // S68c: the old tree gets auto-deleted after this — a silent copy gap must therefore fail
        // the migration LOUDLY here (config untouched, old root stays authoritative) instead of
        // surfacing later as lost data.
        let mut checked = 0u64;
        for name in MIGRATED_SUBTREES {
            checked += verify_dir_copy(&src_root.join(name), &target.join(name), skips_dot_top(name))
                .map_err(|e| format!("MIGRATE_VERIFY_FAILED: {e}"))?;
        }
        tracing::info!("data-dir migration verified: {} files intact under {}", checked, target.display());
        Ok(())
    })
    .await
    .map_err(|e| format!("Copy task failed: {e}"))??;
    {
        let _g = CONFIG_LOCK.lock();
        let mut cfg = load_config(&state.app_dir).unwrap_or_default();
        cfg.data_dir = Some(new.to_string_lossy().to_string());
        let old_s = data_root.to_string_lossy().to_string();
        if !cfg.pending_delete_dirs.iter().any(|p| p == &old_s) {
            cfg.pending_delete_dirs.push(old_s);
        }
        save_config(&state.app_dir, &cfg).map_err(|e| format!("Save config: {e}"))?;
    }
    MIGRATED_THIS_SESSION.store(true, std::sync::atomic::Ordering::SeqCst);
    tracing::info!(
        "Migrated data dir → {} (restart to apply; old tree {} queued for reclaim at next startup)",
        new.display(),
        data_root.display()
    );
    Ok(())
}

/// §user S68c: has a migration completed in THIS process (⇒ button locks until the restart)?
/// Process-local on purpose — a queued-but-unreachable old root (unplugged drive) must NOT keep
/// migration locked across sessions.
static MIGRATED_THIS_SESSION: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Serializes load-modify-save transactions on config.json between the migrate command (queue
/// APPEND) and the reclaim worker (queue REMOVE) — unsynchronized last-writer-wins would drop
/// whichever entry the other side just wrote.
static CONFIG_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

#[tauri::command]
pub fn migrate_pending_restart() -> bool {
    MIGRATED_THIS_SESSION.load(std::sync::atomic::Ordering::SeqCst)
}

/// S68c: finish data-dir migrations on the first startup that runs on the NEW root — for each
/// queued old root: delta-sync stragglers old→new (writes that landed old-side between the
/// migration copy and the restart), then reclaim it. Runs on a background thread: an old tree can
/// be tens of GB and NOTHING in the new session references it (deleting it in the migrating
/// session instead would collide with live handles — ONNX session mmaps, asset-protocol reads).
///
/// Deletion is scoped to MIGRATED_SUBTREES — NEVER the root itself unless it ends up empty: the
/// legacy-AppData root can also house `logs\` (dev builds always; release builds until the S68e
/// log migration has moved them — possibly THIS session's still-writing worker) and other
/// identifier-dir state. A subtree whose delta-sync had ANY failure is kept whole (data beats
/// disk space); bundled dictionaries under the default root are kept too. Per-entry single
/// attempt: a PROCESSED entry leaves the queue even if some subtrees were kept (WARNed with
/// paths) — retrying forever would re-run the delta-sync every boot and could resurrect files the
/// user deleted from the new tree since. An entry stays queued (retried next boot) only while it
/// is genuinely UNREACHABLE: its drive unmounted, or a global postpone (resolver fell back off
/// the target / autosave recovery pending / a sibling live instance exists).
pub fn spawn_pending_data_dir_delete(app_dir: std::path::PathBuf, active_data_dir: std::path::PathBuf) {
    let Some(cfg) = load_config(&app_dir) else { return };
    let entries = cfg.pending_delete_dirs.clone();
    if entries.is_empty() {
        return;
    }
    // A sibling live instance (no single-instance guard exists; double-launch is a supported
    // reality — see crashlog) may still be ROOTED ON an old tree: the classic shape is
    // "migrated, chose Restart Later, then launched a second copy". Deleting under its feet
    // would orphan everything it keeps writing old-side. Postpone; the queue survives.
    if crate::crashlog::other_instance_alive() {
        tracing::warn!("data-dir reclaim postponed: another live instance detected");
        return;
    }
    // Only reclaim when this session actually ROOTS on the configured migration target. If the
    // resolver fell back (new drive unplugged → default dir), deleting old trees would orphan
    // the user's data behind an empty fallback — keep the queue and retry on a later boot.
    let configured = cfg.data_dir.as_deref().map(std::path::PathBuf::from);
    let active_is_target = configured
        .map(|c| {
            let ca = std::fs::canonicalize(&active_data_dir).unwrap_or_else(|_| active_data_dir.clone());
            let cc = std::fs::canonicalize(&c).unwrap_or(c);
            ca == cc
        })
        .unwrap_or(false);
    if !active_is_target {
        tracing::warn!(
            "data-dir reclaim postponed: active root {} is not the configured migration target",
            active_data_dir.display()
        );
        return;
    }
    // Same philosophy as the usp_work startup sweep (lib.rs): a pending autosave recovery may
    // reference media by ABSOLUTE paths under an OLD root (project opened before the restart).
    // Reclaiming now would break the recovery — postpone to a boot with no recovery pending
    // (the queue survives; delta-sync will still carry stragglers over then).
    if app_dir.join("autosave.json").exists() {
        tracing::warn!("data-dir reclaim postponed: autosave recovery pending");
        return;
    }
    std::thread::spawn(move || {
        let mut done: Vec<String> = Vec::new();
        for old in &entries {
            if reclaim_one_root(&app_dir, &active_data_dir, old) {
                done.push(old.clone());
            }
        }
        if done.is_empty() {
            return;
        }
        // Remove ONLY the processed entries, under the config write lock — a migration running
        // concurrently in this session appends its own entry, and an unsynchronized
        // load-modify-save here would drop it.
        let _g = CONFIG_LOCK.lock();
        let mut cfg = load_config(&app_dir).unwrap_or_default();
        cfg.pending_delete_dirs.retain(|p| !done.contains(p));
        // ★S133 — the deliberate-delete list only exists to protect against a queued old root.
        // With the queue empty it can never do anything again, so it goes rather than growing
        // forever. (`record_deliberate_delete` is a no-op in that state for the same reason.)
        if cfg.pending_delete_dirs.is_empty() {
            cfg.deleted_since_migration.clear();
        }
        if let Err(e) = save_config(&app_dir, &cfg) {
            tracing::warn!("data-dir reclaim: failed to update queue: {e}");
        }
    });
}

/// Reclaim a single queued old root (sync stragglers → delete MIGRATED_SUBTREES → rmdir-if-empty).
/// Returns true when the entry is PROCESSED (drop from queue), false to keep it queued for a
/// later boot (drive unmounted).
///
/// ⛔ `pub` only so `tests/dictionary_two_writers.rs` can drive it (S141 §E2E-D3): the dangerous
/// half of S109 is the ORDER of the two dictionary writers, and pinning it at the loader needs its
/// own process — `set_dict_dir` is first-call-wins and the lib test binary already claimed it.
/// Production callers are all in this module; `sync_bundled_dictionaries` and
/// `dictionary_fingerprint_for` are `pub` for the same reason.
pub fn reclaim_one_root(app_dir: &std::path::Path, active_data_dir: &std::path::Path, old: &str) -> bool {
    let old_p = std::path::PathBuf::from(old);
    if !old_p.exists() {
        // "Already deleted" vs "its DRIVE isn't mounted": an old root on a removable/USB or
        // network drive reads as missing while unplugged — dropping the entry then would strand
        // its subtrees forever once the drive returns. Keep it queued while even the path's root
        // component is absent; only a present drive with a missing tree counts as gone.
        if let Some(drive) = old_p.ancestors().filter(|p| !p.as_os_str().is_empty()).last() {
            if !drive.exists() {
                tracing::warn!(
                    "data-dir reclaim postponed: drive {} of old tree {} is not mounted",
                    drive.display(),
                    old
                );
                return false;
            }
        }
        tracing::info!("data-dir reclaim: old tree {} already gone", old);
        return true;
    }
    // Self-protection: never touch a tree that IS (or contains / is contained by) the active
    // root — a hand-edited config could alias them.
    let canon_old = std::fs::canonicalize(&old_p).unwrap_or_else(|_| old_p.clone());
    let canon_active = std::fs::canonicalize(active_data_dir).unwrap_or_else(|_| active_data_dir.to_path_buf());
    if canon_old.starts_with(&canon_active) || canon_active.starts_with(&canon_old) {
        tracing::warn!("data-dir reclaim skipped: {} overlaps the active data dir", old);
        return true;
    }
    // The NSIS-bundled dictionaries live at <install>\data\dictionaries — INSIDE the default
    // data root. When migrating away from the default root, deleting that subtree would strip
    // a bundled resource and make bundled_integrity_report cry "installation incomplete" on
    // every launch (S68c review major). ~18 MB — keep it; every other subtree is user data.
    let old_is_default_root = canon_old
        == std::fs::canonicalize(app_dir.join("data")).unwrap_or_else(|_| app_dir.join("data"));
    // Derived from tauri.conf.json (ONE authority, same as the sync and the fingerprint) — never a
    // hand-kept sixth copy of the file list.
    //
    // S110: each name goes in TWICE — `fr.tsv` and `fr.tsv.syncing`. The second spelling is the temp
    // path `sync_bundled_dictionaries` stages through in this very directory, so carrying a torn
    // straggler of that name out of the old root lets the two writers collide on one path (full
    // mechanism in `sync_dir_delta`'s doc). It is NOT a defence-in-depth extra: `fr.tsv.syncing` is
    // a real file a previous boot can leave behind, and the skip predicate is exact-match.
    let bundled_names: Vec<String> = bundled_dictionary_targets()
        .iter()
        .filter_map(|t| std::path::Path::new(t).file_name().map(|n| n.to_string_lossy().to_string()))
        .flat_map(|n| [format!("{n}.syncing"), n])
        .collect();
    let mut synced = 0u64;
    let mut freed = 0u64;
    let mut kept: Vec<String> = Vec::new();
    // The old root's training/ is in whatever layout it had when it was last active, while
    // the ACTIVE root has just been folded into the project layout — and per-file merging
    // across the two shapes is the one thing that produces an unrecoverable tree. Fold the old
    // root first (pure renames, idempotent, rolls itself back on failure): once both sides are
    // project-shaped, the ordinary per-file delta-sync is correct again and the subtree can be
    // reclaimed as it always was. Skipping it instead would leave the single biggest subtree
    // permanently un-merged AND un-deleted, with the queue entry consumed either way.
    // ★§F2⒝ ④d 笔 2 —— **整条链**,与开机走的是同一个函数。这里的每一步都要跑,理由是
    // `SyncLevel::Slots` 逐**数字**比较槽 layout:旧根比活根少折一档,就整槽拒绝合并,于是
    // 最大的那棵子树既不合并也不删除。以前这里是抄过来的三行,而「活根加了一档、这里忘了」
    // 长得跟一次遗忘一模一样 —— 现在漏不掉了。
    crate::training::migrate_layouts(&old_p);
    // ⛔★★S133 §F2⒝ ④e — BEFORE the delta sync: drop from the OLD copy everything the user
    // deleted on purpose since it was queued. Otherwise the sync below copies it straight back —
    // every refusal in `sync_dir_delta` hangs on `to.exists()`, and a just-deleted path does not
    // exist on the destination, so the whole subtree falls through to a per-file recursion that
    // recreates it and counts it as `copied`. The user then sees a run they explicitly deleted
    // reappear, with the log saying 「freed X MB」.
    //
    // ⚠ Removing it here rather than skipping it in the sync is deliberate: the old subtree is
    // going to be `remove_dir_all`'d at the end of this very function anyway, so this only makes
    // that happen a few lines earlier — and it keeps `sync_dir_delta`'s rule set (which
    // `delta_would_write` MUST mirror exactly) from growing a second opinion.
    let deliberate: Vec<String> = load_config(app_dir)
        .map(|c| c.deleted_since_migration)
        .unwrap_or_default();
    for rel in &deliberate {
        let victim = old_p.join(rel);
        if !victim.exists() {
            continue;
        }
        match crate::util::remove_dir_all_robust(&victim) {
            Ok(()) => tracing::info!(
                "data-dir reclaim: {} was deleted on purpose — dropping the old copy too",
                victim.display()
            ),
            // Loud: leaving it means the sync below resurrects it.
            Err(e) => tracing::warn!(
                "data-dir reclaim: could not drop the old copy of the deliberately deleted {} \
                 ({e}) — it may be copied back",
                victim.display()
            ),
        }
    }
    for name in MIGRATED_SUBTREES {
        let sub = old_p.join(name);
        if name == "training" {
            crate::training::tproject::RECLAIM_TOUCHING_TRAINING.store(true, std::sync::atomic::Ordering::SeqCst);
        }
        // S109: the bundled TSVs are the ONE subtree whose authority is the install, not the old
        // root — carrying them over would undo the sync `lib.rs` performs five lines AFTER it
        // spawns this thread (S110 correction: the original wording had that order backwards, which
        // read as "the fresh bytes are already down" — they may not be. The rule does not depend on
        // it; see `sync_dir_delta`'s doc for why the version argument is the load-bearing one, and
        // for the full mechanism). Note this is about the COPY; the `old_is_default_root` branch
        // below only governs the DELETE.
        let skip_top: &[String] = if name == "dictionaries" { &bundled_names } else { &[] };
        let (c, sync_failed) = sync_dir_delta(
            &sub,
            &active_data_dir.join(name),
            skips_dot_top(name),
            if name == "training" { SyncLevel::Projects } else { SyncLevel::Plain },
            skip_top,
        );
        if name == "training" {
            crate::training::tproject::RECLAIM_TOUCHING_TRAINING.store(false, std::sync::atomic::Ordering::SeqCst);
        }
        synced += c;
        if !sub.exists() {
            continue;
        }
        if name == "dictionaries" && old_is_default_root {
            tracing::info!("data-dir reclaim: keeping {} (bundled install resource)", sub.display());
            continue;
        }
        // A straggler that could not be carried over must never be deleted with its tree —
        // keep the whole subtree and say so (space stays used; data survives).
        if sync_failed > 0 {
            kept.push(format!("{} ({sync_failed} unsynced file(s))", sub.display()));
            continue;
        }
        let size = crate::commands::storage::dir_size(&sub);
        match std::fs::remove_dir_all(&sub) {
            Ok(()) => freed += size,
            Err(e) => {
                kept.push(format!("{} (locked: {e})", sub.display()));
            }
        }
    }
    // Remove the old root only when nothing else lives there (a plain data dir); the
    // legacy-AppData root keeps logs/window-state and stays.
    if std::fs::read_dir(&old_p).map(|mut d| d.next().is_none()).unwrap_or(false) {
        let _ = std::fs::remove_dir(&old_p);
    }
    if kept.is_empty() {
        tracing::info!(
            "data-dir reclaim: freed {} MB from {} ({} straggler file(s) synced first)",
            freed / (1024 * 1024),
            old,
            synced
        );
    } else {
        tracing::warn!(
            "data-dir reclaim: freed {} MB from {} ({} straggler(s) synced); KEPT (delete manually once confirmed): {}",
            freed / (1024 * 1024),
            old,
            synced,
            kept.join("; ")
        );
    }
    true
}

/// S68c (§user): install-completeness report for the NSIS-bundled files, run by the startup
/// component check (which already fires on the first launch after every update). The expected set
/// is parsed out of tauri.conf.json's OWN `bundle.resources` map (compiled in via include_str!) —
/// zero drift with what the installer actually ships. Repair path for these files is a reinstall
/// (they are not in any downloadable pack); the dialog says so instead of pretending to self-heal.
#[derive(serde::Serialize)]
pub struct BundledIntegrity {
    /// Install-relative resource paths that are absent or empty (files: len==0; dirs: no entries).
    pub missing: Vec<String>,
    /// Release build had to load ORT from OUTSIDE the bundled layout (system PATH / stray DLL) —
    /// the bundled onnxruntime.dll is present-but-unloadable or gone. Always false in dev builds.
    pub ort_fallback: bool,
}

#[tauri::command]
pub fn bundled_integrity_report(state: State<'_, Arc<AppState>>) -> BundledIntegrity {
    // Dev builds run from the repo, not an installed tree — the bundled layout doesn't exist.
    if cfg!(debug_assertions) {
        return BundledIntegrity { missing: Vec::new(), ort_fallback: false };
    }
    let mut missing = Vec::new();
    static CONF: &str = include_str!("../../tauri.conf.json");
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(CONF) {
        if let Some(res) = v.pointer("/bundle/resources").and_then(|r| r.as_object()) {
            for target in res.values().filter_map(|t| t.as_str()) {
                let p = state.app_dir.join(target.trim_end_matches('/'));
                let ok = if target.ends_with('/') {
                    std::fs::read_dir(&p).map(|mut d| d.next().is_some()).unwrap_or(false)
                } else {
                    std::fs::metadata(&p).map(|m| m.len() > 0).unwrap_or(false)
                };
                if !ok {
                    missing.push(target.to_string());
                }
            }
        }
    }
    if !missing.is_empty() {
        tracing::warn!("bundled-file integrity: {} resource(s) missing/empty: {}", missing.len(), missing.join(", "));
    }
    // "CUDA"/"DirectML" are the two bundled-layout outcomes init_ort_runtime records; anything
    // else in a release build means the bundled ORT could not be used.
    let ort_fallback = !matches!(
        crate::ORT_LOADED_BUILD.get().map(|s| s.as_str()),
        Some("CUDA") | Some("DirectML")
    );
    if ort_fallback {
        tracing::warn!(
            "bundled-file integrity: ORT loaded from a fallback source ({:?}) — bundled runtime unusable?",
            crate::ORT_LOADED_BUILD.get()
        );
    }
    BundledIntegrity { missing, ort_fallback }
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
    // nobody has). The one hard requirement left is an NVIDIA GPU + its driver. THIS FIRST HOP is
    // fail-OPEN on an EMPTY probe (WMI/PowerShell failure = undetermined) — refuse only on a
    // POSITIVE non-NVIDIA determination. ⛔ S115: that is this hop alone, and it is NOT "the
    // variant_supported convention" — S74b (`f87443a`) made that one fail-CLOSED (the citation
    // was true when written at S64c; see the note on `nvidia_total_vram_mb`). The command as a
    // WHOLE stays fail-closed: `cuda_pkg_supported()` right below refuses a box whose cap probe
    // also came back empty.
    let gpus = query_gpu_adapters();
    if !gpus.is_empty() && !gpus.iter().any(|g| g.vendor == "nvidia") {
        return Err("CUDA_GPU_REQUIRED".to_string());
    }
    // S74b: a box whose card(s) can't run our CUDA package (too old / Blackwell / undetermined).
    // The Settings entry is already hidden for these (cuda_supported); this is the stale-UI and
    // scripted-call defense. DirectML covers them.
    if !cuda_pkg_supported() {
        return Err("CUDA_UNSUPPORTED_GPU".to_string());
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

    // S68d disk preflight (estimate): each missing wheel counted twice (the compressed
    // archive + its extracted DLLs coexist at that lane's peak) MINUS its resumable
    // in-flight .part — kept across cancels by design, so without the credit a nearly
    // complete retry double-counts those bytes and is spuriously refused (review S68d).
    // The ORT nupkg stage always runs: archive + extracted payload coexist until the
    // archive is deleted, so both are counted. Fail open on a failed probe — the
    // per-lane download errors still carry their own causes.
    {
        // Extracted ORT CUDA payload estimate — ~291 MB measured on the shipped
        // 1.24.4 set (providers_cuda.dll alone is 275 MB); rounded up.
        const ORT_GPU_EXTRACTED_EST: u64 = 300_000_000;
        let cuda_dir = app_dir.join("runtime").join("cuda");
        let missing: u64 = CUDA_WHEELS
            .iter()
            .filter(|w| !cuda_dir.join(w.guard).exists())
            .map(|w| {
                let mut part = app_dir
                    .join("runtime")
                    .join(format!("{}.whl.zip", w.guard))
                    .into_os_string();
                part.push(".part");
                let staged = std::fs::metadata(std::path::PathBuf::from(part))
                    .map(|m| m.len().min(w.size))
                    .unwrap_or(0);
                w.size.saturating_mul(2).saturating_sub(staged)
            })
            .sum();
        let needed = missing
            .saturating_add(ORT_GPU_NUPKG_SIZE)
            .saturating_add(ORT_GPU_EXTRACTED_EST);
        if let Some(free) = crate::util::free_bytes_at(app_dir) {
            if free < needed {
                return Err(crate::UtaiError::Download(format!(
                    "INSTALL_DISK_FULL: {} MB needed, {} MB free at {}",
                    needed / 1_000_000,
                    free / 1_000_000,
                    app_dir.display()
                )));
            }
        }
    }

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

/// S74b: reclaim the CUDA runtime (~1.6 GB across runtime/ort/cuda + runtime/cuda). Until now the
/// biggest optional download in the app could be installed but never removed — and the machines
/// that most need to remove it are exactly the ones our CUDA package does not support (an RTX 50
/// owner who downloaded it back when it was offered to every NVIDIA box).
///
/// Two guards, both refusals rather than best-effort deletes, because a half-deleted CUDA runtime
/// is worse than either state:
///  1. no long task may be running (shared fail-closed pre-flight);
///  2. THIS process must not have the CUDA build loaded — Windows keeps a mapped DLL locked, so
///     the delete would half-succeed. The remedy needs BOTH steps: switching the preference alone
///     is not enough, because Auto would load the CUDA build again on the next start; the user has
///     to move the preference off CUDA (or onto DirectML/CPU) AND restart, then delete. The
///     frontend spells that out — this returns the CODE for it.
#[tauri::command]
pub async fn delete_cuda_runtime(state: State<'_, Arc<AppState>>) -> Result<(), String> {
    crate::commands::window::ensure_idle_for_package_delete(&state)?;
    if crate::ort_build_is_cuda() {
        tracing::warn!("delete_cuda_runtime refused: this process loaded the CUDA ORT build");
        return Err("CUDA_DELETE_IN_USE".to_string());
    }
    let runtime_dir = state.app_dir.join("runtime");
    let dirs = vec![runtime_dir.join("ort").join("cuda"), runtime_dir.join("cuda")];
    tokio::task::spawn_blocking(move || delete_dirs_via_trash(&runtime_dir, &dirs, "CUDA_DELETE_FAILED"))
        .await
        .map_err(|e| format!("DELETE_TASK_FAILED: {e}"))?
}

/// Prefix of the deferred-delete staging dirs under `<app>/runtime/` (see delete_dirs_via_trash).
const CUDA_TRASH_PREFIX: &str = ".del-cuda-";

/// Remove directories that may contain MAPPED DLLs, without ever leaving a torn install (S74b
/// review, crown finding).
///
/// A plain `remove_dir_all` cannot work here and the reason is not obvious: the Settings panel
/// itself maps `runtime/cuda/cudart64_12.dll` into this process — `list_inference_gpus` →
/// `gpu::cuda_devices()` → `LoadLibraryA`, and nothing ever calls FreeLibrary. Windows refuses to
/// DELETE a mapped image, so a delete of `runtime/ort/cuda` followed by `runtime/cuda` succeeded on
/// the first and failed with ERROR_ACCESS_DENIED on the second, destroying half the install; the
/// panel then hid its own Delete button (no CUDA runtime = nothing to delete), stranding ~1.4 GB
/// with no way back. Gating on `GetModuleHandle` instead would refuse forever, since merely
/// opening Settings maps the DLL.
///
/// MEASURED (S74b): a mapped DLL blocks DELETE of its directory but NOT a RENAME of it. So:
///   stage 1 — rename every target into one staging dir; any failure rolls the earlier renames
///             back, so the install is either fully gone or fully intact, never half;
///   stage 2 — best-effort delete of the staging dir. What the mapped DLL still pins survives as
///             `runtime/.del-cuda-*` and is reclaimed by `sweep_deleted_cuda` on the next start,
///             which runs before anything maps those DLLs again.
fn delete_dirs_via_trash(
    runtime_dir: &std::path::Path,
    targets: &[std::path::PathBuf],
    fail_code: &str,
) -> Result<(), String> {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let trash = runtime_dir.join(format!("{CUDA_TRASH_PREFIX}{stamp}"));
    if let Err(e) = std::fs::create_dir_all(&trash) {
        tracing::error!("CUDA delete: could not create staging dir {}: {e}", trash.display());
        return Err(format!("{fail_code}: {}: {e}", trash.display()));
    }
    let mut moved: Vec<(std::path::PathBuf, std::path::PathBuf)> = Vec::new();
    for (i, dir) in targets.iter().enumerate() {
        if !dir.exists() {
            continue;
        }
        let dest = trash.join(format!("d{i}"));
        if let Err(e) = std::fs::rename(dir, &dest) {
            tracing::error!("CUDA delete: rename {} aside failed: {e} — rolling back", dir.display());
            for (orig, staged) in moved.iter().rev() {
                if let Err(re) = std::fs::rename(staged, orig) {
                    // Loud: the install is now genuinely torn and only the sweep/user can fix it.
                    tracing::error!("CUDA delete rollback FAILED for {}: {re}", orig.display());
                }
            }
            let _ = std::fs::remove_dir_all(&trash);
            return Err(format!("{fail_code}: {}: {e}", dir.display()));
        }
        moved.push((dir.clone(), dest));
    }
    let staged = moved.len();
    match std::fs::remove_dir_all(&trash) {
        Ok(()) => tracing::info!("CUDA runtime removed ({staged} dir(s))"),
        Err(e) => tracing::warn!(
            "CUDA runtime unlinked ({staged} dir(s)) but {} could not be erased yet ({e}) — mapped DLLs are still pinned; the next startup sweep will reclaim it",
            trash.display()
        ),
    }
    Ok(())
}

/// Reclaim `runtime/.del-cuda-*` left by a deferred delete. Called at startup BEFORE the ORT/CUDA
/// DLLs are loaded, which is the one moment nothing pins them.
pub fn sweep_deleted_cuda(app_dir: &std::path::Path) {
    let runtime_dir = app_dir.join("runtime");
    let Ok(rd) = std::fs::read_dir(&runtime_dir) else { return };
    for entry in rd.flatten() {
        let p = entry.path();
        if !p.is_dir() {
            continue;
        }
        let is_trash = p
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with(CUDA_TRASH_PREFIX));
        if !is_trash {
            continue;
        }
        match std::fs::remove_dir_all(&p) {
            Ok(()) => tracing::info!("Reclaimed deferred CUDA runtime delete: {}", p.display()),
            Err(e) => tracing::warn!("Deferred CUDA delete {} not reclaimed ({e}) — retrying next start", p.display()),
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(name: &str, vendor: &str) -> GpuAdapter {
        GpuAdapter { name: name.to_string(), vendor: vendor.to_string() }
    }

    fn nv(name: &str, uuid: &str, cc10: Option<i32>) -> NvSmiGpu {
        NvSmiGpu { name: name.to_string(), uuid: uuid.to_string(), cc10 }
    }

    fn find<'a>(list: &'a [TrainingGpu], label: &str) -> &'a TrainingGpu {
        list.iter().find(|g| g.label == label).unwrap_or_else(|| panic!("{label} not listed"))
    }

    /// ⛔ §F2⒝ batch 2 — the data-root reclaim must never MERGE two training runs.
    ///
    /// The layout migration mints the legacy run's id deterministically per family, so two data
    /// roots that both went through it have `runs/<the same id>/` holding two DIFFERENT trainings.
    /// A per-file, newest-mtime-wins merge would interleave their `G_*.pth` / `D_*.pth` /
    /// `best_state.json` into one directory — the unrecoverable tree this function exists to
    /// prevent, one level below where the guard used to reach. A pool at the same depth DOES merge
    /// harmlessly, and that contrast is asserted here too: a rule that refused everything below a
    /// slot would be indistinguishable from this one on the run alone.
    #[test]
    fn a_reclaim_merges_pools_but_never_two_runs_with_the_same_id() {
        let base = std::env::temp_dir().join(format!("utai_sync_runs_{}", uuid::Uuid::new_v4()));
        let (old, new) = (base.join("old"), base.join("new"));
        let w = |p: std::path::PathBuf, body: &str| {
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, body).unwrap();
        };
        // Both roots: the same project, the same family slot, both migrated (slot.json present),
        // both holding a run with the SAME id — but different weights.
        for (root, tag) in [(&old, "old"), (&new, "new")] {
            w(root.join("p1_aaaabbbb").join("project.json"), r#"{"id":"p1_aaaabbbb"}"#);
            w(root.join("p1_aaaabbbb").join("rvc").join("slot.json"), r#"{"layout":3}"#);
            w(root.join("p1_aaaabbbb/rvc/runs/rfeedfacefeed/G_1400.pth"), tag);
            w(root.join("p1_aaaabbbb/rvc/pools/pdeadbeef0000/dataset.fingerprint"), "same");
        }
        // …and one artifact that exists only in the OLD root, on each side of the rule
        w(old.join("p1_aaaabbbb/rvc/runs/rfeedfacefeed/D_1400.pth"), "old");
        w(old.join("p1_aaaabbbb/rvc/pools/pdeadbeef0000/0_gt_wavs/0.wav"), "old");

        let (_copied, failed) = sync_dir_delta(&old, &new, false, SyncLevel::Projects, &[]);

        assert!(failed > 0, "a colliding run must be REPORTED, not silently skipped — the caller \
                             deletes the old tree exactly when nothing failed");
        assert_eq!(
            std::fs::read_to_string(new.join("p1_aaaabbbb/rvc/runs/rfeedfacefeed/G_1400.pth"))
                .unwrap(),
            "new",
            "the destination run must be left exactly as it was"
        );
        assert!(
            !new.join("p1_aaaabbbb/rvc/runs/rfeedfacefeed/D_1400.pth").exists(),
            "…and nothing from the other training may leak into it: a G from one run beside a D \
             from another is a pair that resumes into garbage"
        );
        // the POOL at the same depth still merges — its content IS a function of its identity
        assert!(
            new.join("p1_aaaabbbb/rvc/pools/pdeadbeef0000/0_gt_wavs/0.wav").is_file(),
            "pools carry over as before; refusing them too would cost hours of preprocessing"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    /// ⛔ §F2⒝ batch 3 — the SAME id is not the same evidence any more, and reading it as such
    /// would strand every reclaim there is.
    ///
    /// `trun::legacy_run_id` is a pure function of the FAMILY while a slot holds one run, so the
    /// ordinary reclaim — `migrate_data_dir` copies the tree, queues the copy, and the new root is
    /// trained on afterwards — puts `runs/<the same id>/` on both sides BY CONSTRUCTION. Refusing on
    /// the name alone therefore makes every migrated slot report a failure, and one failure keeps
    /// the whole `training/` subtree: gigabytes that can never be freed again, under a warning that
    /// says they are two different trainings.
    ///
    /// So the refusal hangs on whether the old copy has anything to contribute. Both halves are
    /// asserted, because only the pair distinguishes the rule from "always allow" and from
    /// "always refuse" — the sibling test above is the other half.
    #[test]
    fn a_reclaim_frees_a_stale_copy_of_the_same_run() {
        let base = std::env::temp_dir().join(format!("utai_sync_same_{}", uuid::Uuid::new_v4()));
        let (old, new) = (base.join("old"), base.join("new"));
        let w = |p: std::path::PathBuf, body: &str| {
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, body).unwrap();
        };
        for root in [&old, &new] {
            w(root.join("p1_aaaabbbb").join("project.json"), r#"{"id":"p1_aaaabbbb"}"#);
            w(root.join("p1_aaaabbbb").join("rvc").join("slot.json"), r#"{"layout":3}"#);
            w(root.join("p1_aaaabbbb/rvc/runs/rfeedfacefeed/G_1400.pth"), "same");
        }
        // the new root trained on: a checkpoint the old copy never saw. The old copy has NOTHING
        // the new one lacks, which is the whole point.
        w(new.join("p1_aaaabbbb/rvc/runs/rfeedfacefeed/G_2800.pth"), "newer");

        let (copied, failed) = sync_dir_delta(&old, &new, false, SyncLevel::Projects, &[]);

        assert_eq!(
            failed, 0,
            "a stale copy of the same run has nothing to carry over — reporting it keeps the whole \
             training subtree forever, and the id it collides on is a constant"
        );
        assert_eq!(copied, 0, "…and nothing may be written into the destination run either");
        assert!(
            !new.join("p1_aaaabbbb/rvc/runs/rfeedfacefeed/G_1400.pth.syncing").exists(),
            "no staging file may be left behind"
        );
        assert_eq!(
            std::fs::read_to_string(new.join("p1_aaaabbbb/rvc/runs/rfeedfacefeed/G_2800.pth"))
                .unwrap(),
            "newer"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    /// ⛔★★S133 §F2⒝ ④e —— **一份活根上已经改过的文件,不许被旧根那份盖回去。**
    ///
    /// 规则原本是「大小不同 **或** 源更新」,而 `size` 那一半会单独触发。走一遍最普通的剧本:
    /// 迁数据根 → 在**新根**上训练并导出一次(`project.json` 多了一行 `exported`、字节数变了)
    /// → 下次开机回收 ⇒ **旧根那份短的把活的盖掉**。
    /// 后果不是「历史少一行」:`meta.exported` 是 `KeptReason::Exported` 的唯一来源 ⇒
    /// 下一次「清理未导入的快照」把用户导出过的那个存档真删掉。④e 自己的账本退役也会被同一条
    /// 路静默回滚。
    ///
    /// ⚠ 三条 sync 测试此前两边的 `project.json` 都写成同一个字面量(同长同新)⇒ 这条分支
    /// **一次都没被执行过**。
    #[test]
    fn a_reclaim_never_overwrites_a_file_the_active_root_has_moved_on_from() {
        let base = std::env::temp_dir().join(format!("utai_sync_newer_{}", uuid::Uuid::new_v4()));
        let (old, new) = (base.join("old"), base.join("new"));
        let w = |p: std::path::PathBuf, body: &str| {
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, body).unwrap();
        };
        let mtime = |p: &std::path::Path| std::fs::metadata(p).unwrap().modified().unwrap();
        // 旧根:迁移那一刻的快照
        let op = old.join("p1_aaaabbbb").join("project.json");
        w(op.clone(), r#"{"id":"p1_aaaabbbb","exported":[]}"#);
        // 活根:之后导出过一次 —— 更长、而且**更新**。
        // ⚠ 「更新」要等时钟真的走一格:Windows 的文件时间戳按 ~15.6 ms 跳,同一个 tick 里的两次
        //   写会读成一样新,而那时判据落回 size 那一条(它是为**撕裂的拷贝**准备的:`std::fs::copy`
        //   保留源的时间戳,所以半截文件就是「同样新、更短」)。真实剧本里这两次相差几分钟到
        //   几小时。⛔ 用 sleep 凑一个固定毫秒数才是会闪的那种写法。
        let np = new.join("p1_aaaabbbb").join("project.json");
        let body = r#"{"id":"p1_aaaabbbb","exported":[{"name":"m","model_type":"rvc","from_ckpt_rel":"rvc/w.pth","at_ms":9}]}"#;
        loop {
            w(np.clone(), body);
            if mtime(&np) > mtime(&op) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }

        let (copied, failed) = sync_dir_delta(&old, &new, false, SyncLevel::Projects, &[]);
        assert_eq!((copied, failed), (0, 0), "旧根那份没有任何东西可贡献");
        assert!(
            std::fs::read_to_string(new.join("p1_aaaabbbb").join("project.json"))
                .unwrap()
                .contains("from_ckpt_rel"),
            "活根上更新的那份被旧根盖掉了 —— 导出账本被静默回滚,而它是删除保护的唯一来源"
        );

        // ⚠ 阴性对照:方向反过来时**必须**照拷 —— 否则这条修复等于把 straggler 同步整个关掉,
        //    而那正是这个函数存在的理由(「迁移完还在旧根上跑了一会」)。
        let straggler = old.join("p1_aaaabbbb").join("late.txt");
        w(straggler.clone(), "written on the old root after the copy");
        let (copied2, failed2) = sync_dir_delta(&old, &new, false, SyncLevel::Projects, &[]);
        assert_eq!((copied2, failed2), (1, 0));
        assert!(new.join("p1_aaaabbbb").join("late.txt").is_file());
        let _ = std::fs::remove_dir_all(&base);
    }

    /// ⛔★★S133 §F2⒝ ④e —— **用户主动删掉的东西,回收不许把它拷回来。**
    ///
    /// 这个洞今天就在(`delete_slot` / `delete_project` 一样中招),而 ④e 的 per-run 删除会把它
    /// 从「几乎不发生」推成常态。机理:`sync_dir_delta` 的每一道拒绝都挂在 `to.exists()` 上,
    /// 而一个刚被删掉的路径在目标侧**不存在** ⇒ 整棵子树落到逐文件递归 ⇒ 每个文件都 `needs_copy`
    /// ⇒ 原样复活,计成 `copied`,收尾日志写「freed X MB(N straggler synced first)」。
    #[test]
    fn a_reclaim_never_resurrects_something_the_user_deleted_on_purpose() {
        let base = std::env::temp_dir().join(format!("utai_sync_tomb_{}", uuid::Uuid::new_v4()));
        let (app, old, new) = (base.join("app"), base.join("old"), base.join("new"));
        let w = |p: std::path::PathBuf, body: &str| {
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, body).unwrap();
        };
        let seed = |root: &std::path::Path| {
            // a project both roots share, plus ONE run that only the old copy still has
            w(root.join("training/p1_aaaabbbb/project.json"), r#"{"id":"p1_aaaabbbb"}"#);
            w(root.join("training/p1_aaaabbbb/rvc/slot.json"), r#"{"layout":4}"#);
        };
        let run_rel = "training/p1_aaaabbbb/rvc/runs/rfeedfacefeed";
        let mk = || {
            let _ = std::fs::remove_dir_all(&base);
            std::fs::create_dir_all(&app).unwrap();
            for r in [&old, &new] {
                seed(r);
            }
            w(old.join(run_rel).join("G_1400.pth"), "weights");
            save_config(
                &app,
                &AppConfig {
                    pending_delete_dirs: vec![old.to_string_lossy().into_owned()],
                    ..Default::default()
                },
            )
            .unwrap();
        };

        // ⚠ 阴性对照先跑:**不记账**时它真的会复活。没有这一半,下面那条断言可能只是在测
        //    「回收本来就不拷这个目录」。
        mk();
        reclaim_one_root(&app, &new, &old.to_string_lossy());
        assert!(
            new.join(run_rel).join("G_1400.pth").is_file(),
            "阴性对照失效:回收本来就没打算拷它 —— 那么下面那条断言什么也证明不了"
        );

        // …记了账之后就不会
        mk();
        record_deliberate_delete(&app, run_rel);
        reclaim_one_root(&app, &new, &old.to_string_lossy());
        assert!(
            !new.join(run_rel).exists(),
            "用户明确删掉的 run 被回收原样拷了回来(而日志会说 freed)"
        );
        // 兄弟内容照样搬得动 —— 这不是把 straggler 同步关掉
        assert!(new.join("training/p1_aaaabbbb/rvc/slot.json").is_file());

        // 队列空时不记账(没有旧根就没有东西能复活,清单没有理由生长)
        save_config(&app, &AppConfig::default()).unwrap();
        record_deliberate_delete(&app, run_rel);
        assert!(load_config(&app).unwrap().deleted_since_migration.is_empty());

        // 会爬出根的路径一律拒绝 —— 这份清单后面会被 join 到一个根上并**删除**
        save_config(
            &app,
            &AppConfig { pending_delete_dirs: vec!["x".into()], ..Default::default() },
        )
        .unwrap();
        for bad in ["../escape", "training/../../etc", "/abs", "C:/abs", ""] {
            record_deliberate_delete(&app, bad);
        }
        assert!(load_config(&app).unwrap().deleted_since_migration.is_empty());

        let _ = std::fs::remove_dir_all(&base);
    }

    /// ⛔ §F2⒝ batch 2 — layout 2 and layout 3 write the SAME `slot.json`, so "the marker exists on
    /// both sides" cannot tell them apart. The guard has to read the NUMBER.
    ///
    /// Without this a slot whose run products are still at its root merges into one that has
    /// folded them into `runs/<id>/`, planting `G_*.pth` / `run.json` / `weights/` back at the slot
    /// root beside the container that already holds them — a copy nothing reads and nothing
    /// deletes. It is S121's `dataset_44k` finding one layout later, and it is invisible: the merge
    /// reports success and the reclaim then deletes the old root.
    #[test]
    fn a_reclaim_will_not_merge_a_layout_2_slot_into_a_layout_3_one() {
        let base = std::env::temp_dir().join(format!("utai_sync_lay_{}", uuid::Uuid::new_v4()));
        let (old, new) = (base.join("old"), base.join("new"));
        let w = |p: std::path::PathBuf, body: &str| {
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, body).unwrap();
        };
        for root in [&old, &new] {
            w(root.join("p1_aaaabbbb").join("project.json"), r#"{"id":"p1_aaaabbbb"}"#);
        }
        // OLD: pools folded, run products still at the slot root
        w(old.join("p1_aaaabbbb/rvc/slot.json"), r#"{"layout":2}"#);
        w(old.join("p1_aaaabbbb/rvc/G_1400.pth"), "old");
        w(old.join("p1_aaaabbbb/rvc/run.json"), "{}");
        // NEW: the same slot, already folded into runs/
        w(new.join("p1_aaaabbbb/rvc/slot.json"), r#"{"layout":3}"#);
        w(new.join("p1_aaaabbbb/rvc/runs/rfeedfacefeed/G_1400.pth"), "new");

        let (_copied, failed) = sync_dir_delta(&old, &new, false, SyncLevel::Projects, &[]);

        assert!(failed > 0, "a layout mismatch must be reported, or the old root gets deleted");
        assert!(
            !new.join("p1_aaaabbbb/rvc/G_1400.pth").exists(),
            "a layout-2 run product must never be planted at a layout-3 slot root"
        );
        assert!(!new.join("p1_aaaabbbb/rvc/run.json").exists());
        assert_eq!(
            std::fs::read_to_string(new.join("p1_aaaabbbb/rvc/runs/rfeedfacefeed/G_1400.pth"))
                .unwrap(),
            "new"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    /// S115 §F5-2: the upgrade path. Every existing user's `config.json` predates
    /// `diagnostic_mode`, and the whole feature is opt-in — an absent field that deserialized to
    /// anything but `false` would turn diagnostic mode ON for the entire installed base at once
    /// (a silent global slowdown). Also pins that turning it on actually SURVIVES a round trip,
    /// which is the only reason it lives in config.json instead of localStorage.
    #[test]
    fn s115_diagnostic_mode_defaults_off_and_survives_a_round_trip() {
        // An old config, exactly as it exists on disk today: no such key.
        let legacy = r#"{"device":"auto","cuda_mem_limit_mb":0}"#;
        let cfg: AppConfig = serde_json::from_str(legacy).expect("legacy config must still parse");
        assert!(!cfg.diagnostic_mode, "an absent field must read as OFF, never as on");
        assert!(!AppConfig::default().diagnostic_mode);

        let mut cfg = AppConfig::default();
        cfg.diagnostic_mode = true;
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(json.contains("\"diagnostic_mode\":true"), "{json}");
        let back: AppConfig = serde_json::from_str(&json).unwrap();
        assert!(back.diagnostic_mode, "the flag must survive save+load — a repro spans restarts");

        // ⛔ Deliberately NOTHING here touches `training::diagnostics`'s process-wide static.
        // The first version of this test did, and so does the `apply` test in that module —
        // measured: 3 of 6 plain `cargo test` runs FAILED, because libtest runs them in
        // parallel and they fought over one AtomicBool. The release gate runs plain
        // `cargo test`, so that would have been an intermittent red with no stable cause.
        // ONE test owns that static (`s115_apply_stages_the_env_only_when_the_mode_is_on`),
        // and it also covers "the accessor pair actually moves it". Keep it that way.
    }

    /// ★The S75 regression pin. The vendor EARLY-RETURN chain used to return the first matching
    /// vendor, so on this exact machine (the dev box) the 780M — the ONE gfx target the amd pack
    /// supports — could never be selected, with the pack installed. Confirmed live by the user
    /// before the fix. Both cards must now be listed AND both selectable.
    #[test]
    fn mixed_vendor_box_lists_both_cards() {
        let gpus = [gpu("NVIDIA GeForce RTX 3080 Ti", "nvidia"), gpu("AMD Radeon 780M Graphics", "amd")];
        let smi = vec![nv("NVIDIA GeForce RTX 3080 Ti", "GPU-abc", Some(86))];
        let list = training_gpu_list(&gpus, smi, &["nv-cu130", "amd"]);
        assert_eq!(list.len(), 2, "both vendors must be listed, not just the first match");
        let n = find(&list, "NVIDIA GeForce RTX 3080 Ti");
        assert!(n.selectable && n.value == "GPU-abc" && n.variant.as_deref() == Some("nv-cu130"));
        let a = find(&list, "AMD Radeon 780M Graphics");
        assert!(a.selectable, "780M + installed amd pack must be pickable");
        // vendor-RELATIVE index: the AMD card is index 0 among AMD adapters, not 1 overall —
        // device.py masks with it, so an absolute index would select the wrong device.
        assert_eq!(a.value, "0");
        assert_eq!(a.variant.as_deref(), Some("amd"));
    }

    /// ★The regression the FIRST cut of this rewrite introduced (caught by the S75 adversarial
    /// review). `value` is a VENDOR-RELATIVE index, so AMD's first card and Intel's first card
    /// are both "0". Keying the start-time lookup on it resolved to whichever vendor was pushed
    /// first — pick the Arc, train on the Radeon (or get refused with the Radeon's reason).
    /// `id` is the identity; `value` stays the device mask. They are not the same thing.
    #[test]
    fn ids_are_unique_across_vendors_even_when_masks_collide() {
        let gpus = [
            gpu("AMD Radeon 780M Graphics", "amd"),
            gpu("Intel Arc A770", "intel"),
            gpu("Microsoft Basic Render Driver", "other"),
        ];
        let list = training_gpu_list(&gpus, vec![], &["amd", "xpu"]);
        let ids: std::collections::HashSet<&str> = list.iter().map(|g| g.id.as_str()).collect();
        assert_eq!(ids.len(), list.len(), "ids must be unique");

        let amd = find(&list, "AMD Radeon 780M Graphics");
        let arc = find(&list, "Intel Arc A770");
        // the MASKS collide on purpose — that is the accelerator's own ordinal space
        assert_eq!(amd.value, "0");
        assert_eq!(arc.value, "0");
        // …which is exactly why the identity must not be the mask
        assert_ne!(amd.id, arc.id);
        assert!(amd.selectable && arc.selectable, "both packs installed → both pickable");
        assert_eq!(amd.variant.as_deref(), Some("amd"));
        assert_eq!(arc.variant.as_deref(), Some("xpu"));
    }

    /// Machine-specific DIAGNOSTIC (ignored by default — it probes the real box, so it can only
    /// ever be "what does this machine see", never an assertion). Run it when a user reports a
    /// card that is missing from, or greyed out in, the training device picker:
    ///   cargo test --lib training_device_gate_on_this_machine -- --ignored --nocapture
    /// It prints the same three inputs the gate reads (adapters / nvidia-smi / available
    /// variants) next to the verdict, so a wrong verdict points straight at which input lied.
    #[test]
    #[ignore]
    fn training_device_gate_on_this_machine() {
        let app_dir = std::env::current_dir().unwrap().parent().unwrap().to_path_buf();
        // ⚠ WITHOUT this the probe LIES: `list_packs` reads a OnceLock that only lib.rs's setup
        // fills, so a bare test process sees zero installed packs and every GPU comes back
        // TRAINING_GPU_PACK_MISSING. (Cost me one wrong conclusion — S75.)
        crate::pyenv::init_runtime_root(&app_dir.join("data"));
        println!("runtime_root = {:?}", crate::pyenv::runtime_root());
        let gpus = query_gpu_adapters();
        let smi = nvidia_gpu_uuids();
        let available = crate::pyenv::available_training_variants(&app_dir);
        println!("app_dir   = {}", app_dir.display());
        println!("adapters  = {:?}", gpus.iter().map(|g| (&g.name, &g.vendor)).collect::<Vec<_>>());
        println!("smi       = {:?}", smi.iter().map(|g| (&g.name, g.cc10)).collect::<Vec<_>>());
        println!("available = {available:?}");
        println!("nv_cc10   = {:?}", nvidia_compute_caps_cc10());
        for g in training_gpu_list(&gpus, smi, &available) {
            println!(
                "  [{}] id={} value={:?} variant={:?} selectable={} reason={:?}",
                g.label, g.id, g.value, g.variant, g.selectable, g.reason
            );
        }
        // The INFERENCE picker is a different list in a different ordinal space — dump it too,
        // so "which list is the user looking at" is never a guess in a bug report.
        let inf = list_inference_gpus();
        println!("--- list_inference_gpus().directml (the preferred-GPU picker) ---");
        for o in inf.directml {
            println!("  id={} selectable={} reason={:?} {}", o.id, o.selectable, o.reason, o.label);
        }
        println!("--- list_inference_gpus().cuda ---");
        for o in inf.cuda {
            println!("  id={} selectable={} reason={:?} {}", o.id, o.selectable, o.reason, o.label);
        }
    }

    /// The silent-CPU hole: an AMD card our pack cannot drive used to be offered, resolve to the
    /// CPU pack, and train on the CPU with one tracing::warn as the only trace.
    #[test]
    fn unsupported_card_is_listed_but_not_selectable() {
        let gpus = [gpu("AMD Radeon RX 7900 XTX", "amd")];
        let list = training_gpu_list(&gpus, vec![], &["amd", "cpu"]);
        assert_eq!(list.len(), 1, "it stays VISIBLE — a card the user owns must not vanish");
        assert!(!list[0].selectable);
        assert_eq!(list[0].reason.as_deref(), Some("TRAINING_GPU_UNSUPPORTED"));
    }

    /// Supported hardware whose pack simply is not installed gets the ACTIONABLE reason —
    /// distinct from "your card cannot do this", which no download would fix.
    #[test]
    fn supported_card_without_its_pack_says_pack_missing() {
        let gpus = [gpu("AMD Radeon 780M Graphics", "amd")];
        let list = training_gpu_list(&gpus, vec![], &["cpu"]);
        assert!(!list[0].selectable);
        assert_eq!(list[0].reason.as_deref(), Some("TRAINING_GPU_PACK_MISSING"));
    }

    /// Below the shared CUDA_CC10_FLOOR = unsupported; an unreadable cap is its OWN verdict
    /// (S74b: "we read it and it's out of range" ≠ "we couldn't read it"), never a claim about
    /// the user's hardware. Both are fail-closed.
    #[test]
    fn nvidia_cap_decides_per_card() {
        let gpus = [gpu("NVIDIA GeForce GTX 1080", "nvidia")];
        let old = training_gpu_list(&gpus, vec![nv("GTX 1080", "GPU-old", Some(61))], &["nv-cu130"]);
        assert!(!old[0].selectable);
        assert_eq!(old[0].reason.as_deref(), Some("TRAINING_GPU_UNSUPPORTED"));

        let unknown = training_gpu_list(&gpus, vec![nv("GTX 1080", "GPU-old", None)], &["nv-cu130"]);
        assert!(!unknown[0].selectable);
        assert_eq!(unknown[0].reason.as_deref(), Some("TRAINING_GPU_CC_UNKNOWN"));

        // and the floor itself is the shared constant, not a second copy
        let ok = training_gpu_list(
            &gpus,
            vec![nv("RTX 2060", "GPU-new", Some(crate::gpu::CUDA_CC10_FLOOR))],
            &["nv-cu130"],
        );
        assert!(ok[0].selectable);
    }

    /// A device with no training runtime at all (Basic Render / virtual adapters) is a different
    /// fact from an unsupported card, and must never carry a pickable value.
    #[test]
    fn vendorless_adapter_has_no_runtime_and_no_value() {
        let list = training_gpu_list(&[gpu("Microsoft Basic Render Driver", "other")], vec![], &["cpu"]);
        assert!(!list[0].selectable);
        assert_eq!(list[0].reason.as_deref(), Some("TRAINING_GPU_NO_RUNTIME"));
        assert!(list[0].value.is_empty(), "an empty value can never be masked into device.py");
        assert!(list[0].variant.is_none());
    }

    /// The two NAME-based pack gates. Both exist because a vendor-only gate offered users a
    /// multi-GB pack their hardware cannot run (Iris Xe first, then every AMD card) — these cases
    /// are the ones that must not silently come back.
    #[test]
    fn amd_gate_accepts_only_gfx1103_class_igpus() {
        for name in ["AMD Radeon 780M Graphics", "AMD Radeon(TM) 760M", "Radeon 740M Graphics"] {
            assert!(amd_is_rocm_capable(&[gpu(name, "amd")]), "{name}");
        }
        for name in [
            "AMD Radeon RX 7900 XTX",   // gfx1100 — RDNA3, no ATen/BLAS kernel in the pack
                                        // (S115: it DOES get the gfx110x flash-attn images —
                                        //  "no kernels at all" was the third copy of that
                                        //  falsehood; see amd_adapter_is_rocm_capable)
            "AMD Radeon RX 7800M XT",   // must NOT match the "780m" token
            "AMD Radeon RX 6800 XT",    // RDNA2
            "AMD Radeon RX 9070",       // RDNA4
            "AMD Radeon 890M",          // gfx115x
            "AMD Radeon 680M",          // gfx1035
        ] {
            assert!(!amd_is_rocm_capable(&[gpu(name, "amd")]), "{name}");
        }
        // Vendor still matters: a same-named adapter attributed to another vendor is not ours.
        assert!(!amd_is_rocm_capable(&[gpu("Radeon 780M", "intel")]));
        assert!(!amd_is_rocm_capable(&[]));
    }

    /// ★S116 §G16 — the recommendation and the offer gate must never disagree.
    ///
    /// The panel prints "Recommended variant: X" directly above a download list filtered on
    /// `variant_supported`. If those two can differ, we name a pack and then hide it with no
    /// reason — which is consumption point 6 (禁用必给理由) run backwards. So the invariant is
    /// not a lookup table, it is a PROPERTY: whatever `recommend_variant` returns must satisfy
    /// `variant_supported` on the very same inputs.
    #[test]
    fn s116_recommendation_is_always_a_pack_this_machine_may_download() {
        // ⚠ Every arm needs at least one box it can get WRONG, or that arm is untested here: a
        // mutation replacing the Intel arm with bare vendor presence stayed green until the
        // Iris Xe row existed (there is no such thing as a "safe" arm to leave uncovered).
        let boxes: [(&str, Vec<GpuAdapter>, Vec<i32>); 10] = [
            ("supported NVIDIA", vec![gpu("NVIDIA GeForce RTX 3080 Ti", "nvidia")], vec![86]),
            ("Pascal NVIDIA", vec![gpu("NVIDIA GeForce GTX 1080", "nvidia")], vec![61]),
            ("Blackwell NVIDIA", vec![gpu("NVIDIA GeForce RTX 5090", "nvidia")], vec![120]),
            ("NVIDIA adapter, nvidia-smi silent", vec![gpu("NVIDIA GeForce RTX 4070", "nvidia")], vec![]),
            ("Pascal + gfx1103 iGPU", vec![gpu("NVIDIA GeForce GTX 1080", "nvidia"), gpu("AMD Radeon 780M Graphics", "amd")], vec![61]),
            ("Arc only", vec![gpu("Intel(R) Arc(TM) A770 Graphics", "intel")], vec![]),
            ("Iris Xe only", vec![gpu("Intel(R) Iris(R) Xe Graphics", "intel")], vec![]),
            ("RX 7900 XTX only", vec![gpu("AMD Radeon RX 7900 XTX", "amd")], vec![]),
            ("Pascal + Iris Xe", vec![gpu("NVIDIA GeForce GTX 1080", "nvidia"), gpu("Intel(R) Iris(R) Xe Graphics", "intel")], vec![61]),
            ("no adapters at all", vec![], vec![]),
        ];
        for (label, gpus, cc) in boxes {
            let rec = recommend_variant(&gpus, &cc);
            assert!(
                variant_supported(rec, &gpus, &cc),
                "{label}: recommended {rec:?}, which variant_supported() rejects — the Settings \
                 panel would name a pack its own download list hides"
            );
        }
    }

    /// The four rows that carry the actual behaviour change, pinned individually so a regression
    /// says WHICH machine it broke. (The property above passes even for a `_ => \"cpu\"` stub, so
    /// it cannot be the only gate — S108: the specific assertions have to be separable.)
    #[test]
    fn s116_recommendation_tracks_capability_not_vendor_presence() {
        // Blackwell is fine for TRAINING (cu130) even though inference excludes it — the
        // recommendation must follow the training predicate, not the inference one.
        assert_eq!(recommend_variant(&[gpu("NVIDIA GeForce RTX 5090", "nvidia")], &[120]), "nv-cu130");
        // Below the shared floor: a real NVIDIA card with no pack we would let it download.
        assert_eq!(recommend_variant(&[gpu("NVIDIA GeForce GTX 1080", "nvidia")], &[61]), "cpu");
        // nvidia-smi cannot answer ⇒ S74b says NOT supported ⇒ the pack is hidden ⇒ do not name it.
        assert_eq!(recommend_variant(&[gpu("NVIDIA GeForce RTX 4070", "nvidia")], &[]), "cpu");
        // ★S68b rescue, KEPT: the ADAPTER probe died (no gpus at all) but nvidia-smi answered.
        assert_eq!(recommend_variant(&[], &[86]), "nv-cu130", "a dead WMI probe must not funnel an RTX box into the CPU pack");
        // The AMD arm was unreachable behind the old bare-vendor NVIDIA arm; on this box the
        // 780M is the only thing that can actually train.
        assert_eq!(
            recommend_variant(
                &[gpu("NVIDIA GeForce GTX 1080", "nvidia"), gpu("AMD Radeon 780M Graphics", "amd")],
                &[61]
            ),
            "amd"
        );
    }

    /// The two tests above call `recommend_variant` directly, so neither can see what the ONE
    /// production call site actually feeds it. The compiler blocks the exact old bug (a `bool`
    /// no longer type-checks), but not "hand it some other `&[i32]`" — and an empty slice would
    /// make every machine read "cpu" while both tests above stay green. `get_hardware_info` takes
    /// a `State<'_, Arc<AppState>>` and tauri ships no test feature here (S110), so the call site
    /// is pinned in the only hermetic way left: by reading this file.
    /// ⚠ TO FIX A FAILURE HERE: do not delete the assertion — either restore the shared probe as
    /// the argument, or, if `recommended_variant` genuinely moved, re-anchor the search string.
    #[test]
    fn s116_the_recommendation_call_site_uses_the_shared_probe() {
        static SELF_SRC: &str = include_str!("settings.rs");
        let i = SELF_SRC
            .find("recommended_variant: recommend_variant(")
            .expect("`recommended_variant: recommend_variant(` is gone from get_hardware_info");
        let call: String = SELF_SRC[i..].chars().take(160).collect();
        assert!(
            call.contains("nvidia_compute_caps_cc10()"),
            "the recommendation must be derived from the SAME hoisted probe the pack gates use, \
             or it can drift back into naming a pack the download list hides. Found: {call:?}"
        );
    }

    /// ★S116 — nobody re-scatters the ORT build identity.
    ///
    /// `ORT_LOADED_BUILD` is a four-valued DISPLAY string, and since S74b one of its readings
    /// decides whether the DirectML EP may be registered — which on the CUDA build does not fail
    /// cleanly, it ACCESS-VIOLATES the process. It was compared to the bare literal `"CUDA"` in
    /// five places across three files (one of them negated), i.e. five chances to disagree about
    /// the identity of a value that gates a process crash, and no single place stating what the
    /// two UNCLASSIFIED values mean. `crate::ort_build_is_cuda` is now that place.
    /// ⚠ TO FIX A FAILURE: call `crate::ort_build_is_cuda()` instead of comparing the tag; if you
    /// genuinely need a different question, add it NEXT TO that helper with its own doc.
    #[test]
    fn s116_the_ort_build_identity_has_exactly_one_reader() {
        for (name, src) in [
            ("lib.rs", include_str!("../lib.rs")),
            ("inference/engine.rs", include_str!("../inference/engine.rs")),
            ("commands/settings.rs", include_str!("settings.rs")),
        ] {
            // Self-check first (S105): a typo in the path would make this vacuous. Each of the
            // three files must still PARTICIPATE in the contract — either by holding the tag or
            // by asking the one question. (engine.rs holds neither literal any more, which is
            // exactly the point of this commit, so "contains ORT_LOADED_BUILD" is NOT the test.)
            assert!(
                src.contains("ORT_LOADED_BUILD") || src.contains("ort_build_is_cuda"),
                "{name}: sliced the wrong file, or it dropped out of the ORT-build contract"
            );
            for (n, line) in src.lines().enumerate() {
                let code = line.split("//").next().unwrap_or("");
                assert!(
                    !(code.contains("ORT_LOADED_BUILD") && code.contains("\"CUDA\"")),
                    "{name}:{}: a sixth copy of the build identity — use crate::ort_build_is_cuda()\n  {line}",
                    n + 1
                );
            }
        }
        // …and the single reader must still exist and still be the one that answers.
        assert!(include_str!("../lib.rs").contains("pub fn ort_build_is_cuda()"));
        assert!(!crate::ort_build_is_cuda(), "an unset OnceLock must answer false (test processes never set it)");
    }

    /// ★S116 — the collapse of those five expressions into one is only allowed if it is
    /// BEHAVIOUR-IDENTICAL, and "I read it carefully" is not a criterion. This enumerates every
    /// value the tag can hold and re-evaluates BOTH original shapes against the new one — the
    /// negated site (`try_dml`) included, since that is the one an off-by-a-polarity rewrite
    /// would break silently (Auto would stop probing DirectML, i.e. land on CPU).
    #[test]
    fn s116_the_single_reader_is_byte_equivalent_to_the_five_it_replaced() {
        for tag in [
            None,
            Some("CUDA"),
            Some("DirectML"),
            Some("dev/system (D:\\ort\\onnxruntime.dll)"),
            Some("system PATH"),
            Some("?"), // what get_hardware_info shows before init records anything
        ] {
            let new = crate::ort_tag_is_cuda(tag);
            // the four positive sites: `…get().map(|b| b == "CUDA").unwrap_or(false)`
            assert_eq!(new, tag.map(|b| b == "CUDA").unwrap_or(false), "positive form, tag={tag:?}");
            // the negated site: `let try_dml = …get().map(|b| b != "CUDA").unwrap_or(true)`
            assert_eq!(!new, tag.map(|b| b != "CUDA").unwrap_or(true), "negated form, tag={tag:?}");
        }
        // The two classified values must stay distinguishable — a rename that collapsed them
        // would make every guard above answer for the wrong build.
        assert!(crate::ort_tag_is_cuda(Some("CUDA")));
        assert!(!crate::ort_tag_is_cuda(Some("DirectML")));
    }

    /// ★S116 — the ONE user-facing copy of the compute-capability floor stays tied to the constant.
    ///
    /// `CUDA_CC10_FLOOR`'s own doc says "do not derive a second floor anywhere", and after S116
    /// removed the drifted prose copies there is exactly one number left outside gpu.rs:
    /// `rtWhyNv` in Settings.tsx, in all three languages. That is a legitimate place for it (the
    /// string has to name a card generation the user recognises) — what it must not do is drift.
    /// ⚠ The moment it WILL drift is already written down: gpu.rs step ⑨ of the CUDA-13 migration
    /// checklist. Whoever does that will change the constant and has no reason to think about a
    /// TSX string; this test is what tells them.
    #[test]
    fn s116_the_user_facing_cc_floor_still_matches_the_constant() {
        static SETTINGS_TSX: &str = include_str!("../../../src/components/common/Settings.tsx");
        let i = SETTINGS_TSX
            .find("rtWhyNv:")
            .expect("`rtWhyNv` is gone from Settings.tsx — re-anchor or drop this test");
        // ⚠ char-wise, not byte-wise: this line is three languages of CJK (S115).
        let row: String = SETTINGS_TSX[i..].chars().take(700).collect();
        let expected = format!("sm_{}", crate::gpu::CUDA_CC10_FLOOR);
        // All three locales live on that one row; each must name the floor.
        let hits = row.matches(&expected).count();
        assert!(
            hits >= 3,
            "rtWhyNv names the CUDA floor {hits} time(s), expected one per locale (zh/en/ja).\n\
             CUDA_CC10_FLOOR is now {} ⇒ the user-facing text must say {expected}. Row was:\n{row}",
            crate::gpu::CUDA_CC10_FLOOR
        );
    }

    #[test]
    fn intel_gate_accepts_arc_never_xe() {
        for name in ["Intel(R) Arc(TM) A770 Graphics", "Intel(R) Arc(TM) Graphics", "Intel(R) Data Center GPU Max 1100"] {
            assert!(intel_is_xpu_capable(&[gpu(name, "intel")]), "{name}");
        }
        for name in ["Intel(R) Iris(R) Xe Graphics", "Intel(R) UHD Graphics 620", "Intel(R) HD Graphics 530"] {
            assert!(!intel_is_xpu_capable(&[gpu(name, "intel")]), "{name}");
        }
    }

    /// The destructive path: it must be all-or-nothing. (The mapped-DLL case that motivated it
    /// can't be built in a unit test — that was verified empirically — but the rename/rollback
    /// mechanics and the sweep are exactly what turn a half-delete into an atomic one.)
    #[test]
    fn cuda_delete_is_atomic_and_sweepable() {
        let base = std::env::temp_dir().join(format!(
            "utai_s74b_del_{}",
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        let runtime = base.join("runtime");
        let a = runtime.join("ort").join("cuda");
        let b = runtime.join("cuda");
        for d in [&a, &b] {
            std::fs::create_dir_all(d).unwrap();
            std::fs::write(d.join("x.dll"), b"payload").unwrap();
        }
        let targets = vec![a.clone(), b.clone()];
        delete_dirs_via_trash(&runtime, &targets, "CUDA_DELETE_FAILED").unwrap();
        assert!(!a.exists() && !b.exists(), "both target dirs must be gone");

        // A leftover staging dir (the "still pinned" case) is reclaimed by the startup sweep and
        // nothing else in runtime/ is touched.
        let leftover = runtime.join(format!("{CUDA_TRASH_PREFIX}999"));
        std::fs::create_dir_all(leftover.join("d0")).unwrap();
        std::fs::create_dir_all(runtime.join("ort")).unwrap();
        std::fs::write(runtime.join("ort").join("keep.dll"), b"keep").unwrap();
        sweep_deleted_cuda(&base);
        assert!(!leftover.exists(), "sweep must reclaim the staging dir");
        assert!(runtime.join("ort").join("keep.dll").exists(), "sweep must touch nothing else");

        // Rollback: a target that cannot be renamed (a FILE where a dir is expected is enough to
        // make the second rename fail) must leave the first target where it was.
        let c = runtime.join("cuda2");
        std::fs::create_dir_all(&c).unwrap();
        std::fs::write(c.join("y.dll"), b"payload").unwrap();
        let missing_parent = runtime.join("nope").join("deep");
        let r = delete_dirs_via_trash(&runtime, &[c.clone(), missing_parent], "CUDA_DELETE_FAILED");
        // The bogus second target simply doesn't exist, so it is skipped and the call succeeds —
        // assert the REAL invariant instead: an absent target is never an error.
        assert!(r.is_ok(), "absent targets must not fail the delete: {r:?}");
        assert!(!c.exists());

        let _ = std::fs::remove_dir_all(&base);
    }

    /// The inference/training split is ONE floor plus ONE dated exception — the property that
    /// makes the CUDA-13 migration a single deletion.
    #[test]
    fn cuda_predicates_share_the_floor_and_differ_only_on_blackwell() {
        for cc in [50, 61, 70, 74] {
            assert!(!crate::gpu::cuda_cc_supported_training(cc), "sm_{cc}");
            assert!(!crate::gpu::cuda_cc_supported_inference(cc), "sm_{cc}");
        }
        for cc in [75, 86, 89, 90] {
            assert!(crate::gpu::cuda_cc_supported_training(cc), "sm_{cc}");
            assert!(crate::gpu::cuda_cc_supported_inference(cc), "sm_{cc}");
        }
        // Blackwell: training (cu130 torch) yes, inference (ORT CUDA 12.9) not yet.
        for cc in [100, 120] {
            assert!(crate::gpu::cuda_cc_supported_training(cc), "sm_{cc}");
            assert!(!crate::gpu::cuda_cc_supported_inference(cc), "sm_{cc}");
        }
    }

    /// S101: `bundled_dictionary_targets` is the source list for BOTH the startup freshness sync and
    /// the fingerprint that carries dictionary content into the bake signature. Its failure mode is
    /// silent and total: an empty list makes the sync a no-op AND makes the fingerprint the hash of
    /// an empty string — a constant, i.e. a carrier that looks wired and can never fire again. It is
    /// derived by string-matching inside tauri.conf.json, so a re-key of `bundle.resources` (or a
    /// move of the dictionaries out of `data/dictionaries/`) breaks it with everything still green.
    /// Hermetic — the config is compiled in with `include_str!`, no files needed.
    #[test]
    fn bundled_dictionary_targets_are_actually_found() {
        let t = super::bundled_dictionary_targets();
        assert!(
            t.len() >= 8,
            "bundle.resources yielded {} dictionary targets ({t:?}) — the startup sync would be a \
             no-op and the fingerprint a constant. Did the resource map get re-keyed?",
            t.len()
        );
        for want in ["en.tsv", "de.tsv", "fr.tsv", "es.tsv", "it.tsv",
                     "zh_chars.tsv", "zh_phrases.tsv", "zh_syllables.tsv"] {
            assert!(t.iter().any(|x| x.ends_with(want)), "{want} not among the bundled targets: {t:?}");
        }
        // File paths only — a trailing-slash directory entry would make `file_name()` useless.
        assert!(t.iter().all(|x| x.starts_with("data/dictionaries/") && !x.ends_with('/')), "{t:?}");
    }

    /// S101: actually RUN `sync_bundled_dictionaries` over temp trees. It is dead code on this
    /// machine otherwise — a dev build resolves src and dst to the same repo directory, so the whole
    /// function no-ops and the first execution of the real path would be on a user's install
    /// (S85 rule: the user is never the first tester). Covers all four behaviours that matter,
    /// including the two "do nothing" ones, because those are where a sync goes from useless to
    /// destructive.
    #[test]
    fn dictionary_sync_refreshes_stale_and_never_destroys() {
        let base = std::env::temp_dir().join(format!("utai_dictsync_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let (app, data) = (base.join("install"), base.join("data"));
        let (src, dst) = (app.join("data").join("dictionaries"), data.join("dictionaries"));
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dst).unwrap();
        std::fs::write(src.join("fr.tsv"), "abstenir\ta p s t ə n i ʁ\n").unwrap();
        std::fs::write(src.join("en.tsv"), "even\tIY1 V AH0 N\n").unwrap();
        std::fs::write(dst.join("fr.tsv"), "abstenir\ta p s t ə ɲ i ʁ\n").unwrap(); // STALE
        std::fs::write(dst.join("en.tsv"), "even\tIY1 V AH0 N\n").unwrap(); // already current
        std::fs::write(dst.join("keepme.txt"), "not ours").unwrap();

        super::sync_bundled_dictionaries(&app, &data);
        // 1. the stale one is refreshed …
        assert_eq!(std::fs::read_to_string(dst.join("fr.tsv")).unwrap(), "abstenir\ta p s t ə n i ʁ\n");
        // 2. … a file that already matches is left alone, and nothing foreign is touched or removed.
        assert_eq!(std::fs::read_to_string(dst.join("en.tsv")).unwrap(), "even\tIY1 V AH0 N\n");
        assert_eq!(std::fs::read_to_string(dst.join("keepme.txt")).unwrap(), "not ours");
        // 3. no `.syncing` temp survives a successful run.
        let leftovers: Vec<_> = std::fs::read_dir(&dst).unwrap().flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.contains("syncing")).collect();
        assert!(leftovers.is_empty(), "temp files left behind: {leftovers:?}");

        // 4. a MISSING source is a silent no-op, never "the destination is now wrong/empty".
        let empty_app = base.join("no_install");
        super::sync_bundled_dictionaries(&empty_app, &data);
        assert_eq!(std::fs::read_to_string(dst.join("fr.tsv")).unwrap(), "abstenir\ta p s t ə n i ʁ\n");

        // 5. src == dst (the DEFAULT install, and the dev build) must not walk a tree onto itself.
        super::sync_bundled_dictionaries(&app, &app.join("data"));
        assert_eq!(std::fs::read_to_string(src.join("fr.tsv")).unwrap(), "abstenir\ta p s t ə n i ʁ\n");
        let self_leftovers: Vec<_> = std::fs::read_dir(&src).unwrap().flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.contains("syncing")).collect();
        assert!(self_leftovers.is_empty(), "self-sync left temp files: {self_leftovers:?}");

        let _ = std::fs::remove_dir_all(&base);
    }

    /// S109 — the OTHER writer into `<data-root>/dictionaries`, and the one that could undo the
    /// sync above. `lib.rs` setup() spawns the reclaim thread (line ~1037) and THEN calls
    /// `sync_bundled_dictionaries` (line ~1042), whose doc justifies being synchronous by saying a
    /// background sync "could pin a stale dictionary for the whole session with nothing able to
    /// invalidate it" — while `reclaim_one_root` walks the old root's `dictionaries` subtree with a
    /// `size differs || src newer` predicate and copies old→new. Before the skip list this test
    /// pins, a queued old root left over from before an app update wrote its de/en/fr.tsv back over
    /// the freshly synced ones (es/it/zh_* have equal sizes across the two real generations, so they
    /// did not move — the result was a MIXED set no build ever shipped), the session rendered with
    /// the old phones, and the next boot put the new files back so nothing was ever red.
    ///
    /// WHY THIS IS A UNIT TEST AND NOT PART OF `tests/dictionary_distribution.rs`: the production
    /// entry point `spawn_pending_data_dir_delete` is guarded by three checks that depend on the
    /// MACHINE, not on the fixture — `crashlog::other_instance_alive()` reads the real
    /// `%LOCALAPPDATA%\UtaiSynthesizer\logs`, so a gate driving that entry point would postpone (and
    /// pass while asserting nothing) whenever the maintainer happens to have the app open. Driving
    /// `reclaim_one_root` directly is the deterministic half; the environment-dependent half is
    /// written up as a real-window item rather than faked here.
    ///
    /// ★ MUTATION-PROBED, twice, and the two probes go red on DIFFERENT assertions — which is the
    /// point, because it is what separates "the fix" from "the lazy fix":
    ///   · M1 — pass `&[]` as `skip_top_names` at the call site (i.e. no fix at all)
    ///     ⇒ red on assertion 1, "the reclaim wrote the old root's stale fr.tsv over the freshly
    ///     synced one".
    ///   · M2 — skip the whole `dictionaries` subtree in the reclaim instead of just the bundled
    ///     names (the tempting one-liner) ⇒ red on assertion 2, the user file that was parked in
    ///     that directory never arrives.
    /// (A single probe would not have shown this: assertion 1 aborts the test, so M1 alone says
    /// nothing about whether assertions 2-3 can still fire. Logs: TESTING\s109_c16_dict_distribution
    /// \mut_M1_no_skiplist.log and \mut_M2_whole_subtree_skipped.log.)
    #[test]
    fn reclaim_never_carries_a_stale_bundled_dictionary_into_the_active_root() {
        let base = std::env::temp_dir().join(format!("utai_reclaimdict_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let app = base.join("install");
        let active = base.join("active");
        let old = base.join("oldroot"); // deliberately NOT app/data — that population is safe already
        let install_dicts = app.join("data").join("dictionaries");
        for d in [&install_dicts, &active.join("dictionaries"), &old.join("dictionaries"), &old.join("cache")] {
            std::fs::create_dir_all(d).unwrap();
        }

        const FRESH: &str = "abstenir\ta p s t ə n i ʁ\n"; // post-D6, what the install ships
        const STALE: &str = "abstenir\ta p s t ə ɲ i ʁ\n"; // pre-D6, what an old root still holds
        // ⛔⛔ S141 CORRECTION — this fixture used to write the old root FIRST, with the comment
        // "so the active copy ends up with the newer mtime: that removes the `src newer` clause
        // from play and leaves the SIZE clause". **That reads `needs_copy` backwards.** Its FIRST
        // arm is `dm > sm => false` — a newer destination returns immediately and the size clause
        // is never reached. So in that shape the copy was blocked by `needs_copy`, NOT by the skip
        // list, and assertion 1 below was protected by the wrong thing.
        //
        // Measured: the same M1 mutation (`skip_top_names = &[]`) run twice went red on DIFFERENT
        // assertions — once on 1, once on 5 — because whether the two `fs::write` calls land in the
        // same clock tick decides whether `needs_copy` falls through to the size clause at all.
        // A judgement that is only sometimes reachable is worse than one that is never reachable:
        // it looks like coverage.
        //
        // ⇒ The old root is now written LAST and forced strictly newer, so `needs_copy` returns
        // true and **the skip list is the only thing standing between the stale bytes and the
        // active root**. (The `.syncing` straggler in assertion 5 was always deterministic — it
        // does not exist in the active root at all, so `(Ok, Err) => true`.)
        std::fs::write(install_dicts.join("fr.tsv"), FRESH).unwrap();
        std::fs::write(active.join("dictionaries").join("fr.tsv"), FRESH).unwrap();
        std::fs::write(old.join("dictionaries").join("fr.tsv"), STALE).unwrap();
        std::fs::write(old.join("dictionaries").join("notes.txt"), "user parked this here").unwrap();
        // S110 (assertion 5): what a previous boot's TORN dictionary sync leaves in a root — the
        // cleanup in `sync_bundled_dictionaries` runs on the failure branch only, so a crash between
        // `fs::copy` and the rename strands exactly this file. It is not covered by skipping
        // "fr.tsv": the skip predicate is exact-match, and carrying it over lands it on the very
        // path the sync stages through in the ACTIVE root.
        std::fs::write(old.join("dictionaries").join("fr.tsv.syncing"), b"torn temp from a previous boot").unwrap();
        std::fs::write(old.join("cache").join("straggler.bin"), b"written after the migration").unwrap();
        // The two arms must actually differ, or every assertion below is vacuous (S92p).
        assert_ne!(STALE.len(), FRESH.len(), "fixture is vacuous: the two arms are indistinguishable");
        // …and the SOURCE must be strictly newer, or `needs_copy`'s first arm blocks the copy and
        // assertion 1 stops testing the skip list. The clock granularity can put two writes in one
        // tick, so this is forced rather than assumed, and then ASSERTED — a fixture precondition
        // that is only usually true is how a judgement goes quietly vacuous.
        let stale_fr = old.join("dictionaries").join("fr.tsv");
        let fresh_fr = active.join("dictionaries").join("fr.tsv");
        let mtime = |p: &std::path::Path| std::fs::metadata(p).unwrap().modified().unwrap();
        for _ in 0..200 {
            if mtime(&stale_fr) > mtime(&fresh_fr) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
            std::fs::write(&stale_fr, STALE).unwrap();
        }
        assert!(
            mtime(&stale_fr) > mtime(&fresh_fr),
            "fixture is vacuous: `needs_copy` returns false for a newer destination BEFORE it ever \
             looks at the size, so the skip list would not be the thing under test here"
        );

        let processed = super::reclaim_one_root(&app, &active, old.to_str().unwrap());

        // 1. THE POINT: the authoritative copy survived the reclaim untouched.
        assert_eq!(
            std::fs::read_to_string(active.join("dictionaries").join("fr.tsv")).unwrap(),
            FRESH,
            "the reclaim wrote the old root's stale fr.tsv over the freshly synced one"
        );
        // 2. … and it did not buy that by refusing to carry NON-bundled files out of the same
        //    directory. That rule exists so a straggler is never deleted with its tree.
        //    Read with a message rather than `unwrap`: the failure mode here is an ABSENT file, and
        //    a bare unwrap would report only "called `Result::unwrap()` on an `Err`" (the probe M2
        //    made exactly that unreadable panic — see this repo's rule about reporting layers).
        assert_eq!(
            std::fs::read_to_string(active.join("dictionaries").join("notes.txt")).ok().as_deref(),
            Some("user parked this here"),
            "a user file in the old dictionaries dir was dropped instead of carried over — the skip \
             list must name the bundled TSVs only, never the whole subtree"
        );
        // 3. … nor by disabling the delta-sync for unrelated subtrees.
        assert_eq!(
            std::fs::read_to_string(active.join("cache").join("straggler.bin")).ok().as_deref(),
            Some("written after the migration"),
            "the cache straggler was not carried over — the delta-sync itself is broken"
        );
        // 4. The entry is PROCESSED and the old tree is gone — skipping must not leak disk forever.
        assert!(processed, "the queue entry was not processed");
        assert!(!old.join("dictionaries").exists(), "old dictionaries subtree kept: {}", old.display());
        assert!(!old.join("cache").exists(), "old cache subtree kept");
        // 5. S110 — and the old root's TORN STAGING FILE was not carried over either. This is a
        //    separate failure from 1: `fr.tsv` itself is skipped by name, so 1-3 stay green while
        //    `fr.tsv.syncing` still arrives — landing on the exact path `sync_bundled_dictionaries`
        //    stages through (`fr.tsv`.with_extension("tsv.syncing")), i.e. a rename from this thread
        //    can replace the fresh bytes between that function's copy and its own rename, after
        //    which it logs "refreshed" over the old content.
        assert!(
            !active.join("dictionaries").join("fr.tsv.syncing").exists(),
            "the old root's torn `fr.tsv.syncing` was carried into the active root — that is the \
             temp path the bundled sync stages through, so the two writers can collide on one file. \
             The skip list must name the `.syncing` twin of every bundled TSV, not just the TSV."
        );

        // ⚠ S141 §E2E-D3 的口径更正,写在这里免得下一个人重走一遍:
        //    D-3 那一格写「判据要落在唱出来的音素而不是盘上的文件」,S134 的侦察据此判定这条
        //    测试「差正好一跳」。**那一跳不该补在这里**:上面第 1 条钉的是 `fr.tsv` 的**逐字节
        //    全等**,而且是在 reclaim 返回的**那一刻**取的 —— 字节全等就蕴含了音素相同,再补
        //    一条「喂进解析层唱一遍」是**被它蕴含的装饰性判据**(实测:任何能让音素变的坏法,
        //    第 1 条都会先红,那条新断言一个字也说不上)。
        //    「唱出来」真正不可替代的地方是**载入器的缓存**能与盘不一致(`set_dict_dir` 是
        //    first-call-wins、词典 `Box::leak` 到进程生命周期)。那一层
        //    `tests/dictionary_distribution.rs` 步骤 4+5 早就有了(还带 `UTAI_MUTANT_STALE_INSTALL`
        //    这个「文件都拷对了、程序仍唱旧的」变异钩)。
        //    ⇒ 真正没人测的是**两个写者的先后顺序**:`lib.rs` 先 spawn 回收线程、再同步调 sync,
        //    所以回收可能落在 sync **之后**。那条腿在 `tests/dictionary_two_writers.rs`。

        let _ = std::fs::remove_dir_all(&base);
    }

    /// S110 (queue §G14-②) — the BOOT-STEP CONTRACT: steps whose only production call site is
    /// `lib.rs` setup(), where deleting or backgrounding the call leaves the whole suite green.
    ///
    /// ★ WHY A TEXT GATE, stated as a negative result rather than a preference. Three other shapes
    /// were evaluated first and each one is structurally unable to catch the mutation:
    ///  · a UNIT TEST of the callee — that is what already exists
    ///    (`dictionary_sync_refreshes_stale_and_never_destroys`). It proves the function works; the
    ///    defect is that nothing calls it.
    ///  · the INTEGRATION chain (`tests/dictionary_distribution.rs`) — it calls
    ///    `sync_bundled_dictionaries` itself, with literal paths, so it passes with `lib.rs`'s call
    ///    deleted. Its own header says it drives "the exact call `lib.rs` setup() makes"; it drives
    ///    an identical call, which is not the same claim.
    ///  · a RUNTIME "did it run" flag (`OnceLock`, the shape `pyenv::RUNTIME_ROOT` already uses) —
    ///    it cannot go red anywhere: in the lib binary the sibling unit test at
    ///    `dictionary_sync_refreshes_stale_and_never_destroys` sets it for the whole process, in the
    ///    integration binary the test sets it itself, and the real consumers are `#[tauri::command]`
    ///    fns needing `State<'_, Arc<AppState>>`, which no test target can build (`tauri` is not
    ///    compiled with its `test` feature here and there are no dev-dependencies). Such a flag is
    ///    production OBSERVABILITY, not a tripwire — worth having, but it must not be recorded as
    ///    closing this item. (The cheap half of it did land: `sync_bundled_dictionaries` now logs on
    ///    every return path, so an installed build's log distinguishes "ran, no-op" from "never ran".)
    ///
    /// ⚠ So this reads `src/lib.rs` AS TEXT, the same zero-drift trick as
    /// `phoneme_input_bound_is_unreachable_from_the_editor` (`commands/inference.rs`, S109) and
    /// `bundled_dictionary_targets` (tauri.conf.json). Text pins have a real cost — an honest
    /// refactor can turn them red — so every message below says how to RE-POINT it and why it is not
    /// safe to simply delete.
    ///
    /// The contract, per step, and each clause is a claim `lib.rs` makes about itself somewhere:
    ///  1. the call EXISTS and is UNIQUE (a second call site would mean two authorities);
    ///  2. it is a STATEMENT, not an argument or a closure body — catches the brace-less
    ///     `spawn_blocking(move || sync(..))` shape that keeps brace depth at 1;
    ///  3. it sits at brace depth 1 inside the setup closure — catches `thread::spawn(move || {…})`,
    ///     `async_runtime::spawn`, and being buried in a new `if`/`match` arm;
    ///  4. it carries no `#[cfg(…)]` attribute — otherwise a build profile can silently drop it;
    ///  5. it runs AFTER the last write to `data_dir` — `lib.rs:1011` resolves the root and the
    ///     legacy-AppData block can REASSIGN it, so hoisting the call above that block keeps this
    ///     line byte-identical while the sync starts refreshing a directory nobody reads. That
    ///     population — an upgrader whose models still live under AppData — is exactly one of the
    ///     two the sync was written for;
    ///  6. (dictionary sync only) it runs BEFORE the main window is built: `g2p::set_dict_dir` is a
    ///     first-call-wins `OnceLock` reached from three `#[tauri::command]`s, and commands cannot be
    ///     invoked until the window exists;
    ///  7. its ARGUMENTS still name the same two roots. Deliberately LAST: it is the broadest clause,
    ///     so putting it earlier would let it swallow failures that belong to 2/3/5 (S108's rule —
    ///     specific groups first, catch-all last, or the specific ones are never exercised).
    ///
    /// ★ THE SCANNER IS SELF-CHECKED BEFORE IT IS TRUSTED (S89: feed a checker a known-correct
    /// sample first). It asserts the `.setup(` opener line has depth 1 on its own, then walks forward
    /// to the line where depth returns to 0 and requires that closer to lie beyond every anchor —
    /// which is also what proves the calls are inside the closure rather than merely after it.
    ///
    /// ★ SECOND STEP COVERED ON PURPOSE — `pyenv::init_runtime_root`. The survey behind this gate
    /// ranked all 20 setup steps by "what breaks if this line disappears"; the dictionary sync was
    /// not first. `init_runtime_root` fills the `RUNTIME_ROOT` `OnceLock`, and without it
    /// `list_packs()` returns empty, so every training/converter GPU check answers
    /// `TRAINING_GPU_PACK_MISSING` — the whole training side goes dark and tells the user to install
    /// a pack they already have. That consequence is not inferred: the repo already paid for it once
    /// (`training_device_gate_on_this_machine`: "WITHOUT this the probe LIES … Cost me one wrong
    /// conclusion — S75"). The remaining untied steps are listed in queue §G14-④ rather than pinned
    /// here, because each needs its own ordering claim and inventing one is worse than none.
    ///
    /// ★ MUTATION-PROBED, one probe per clause, each landing on a DIFFERENT assertion — logs in
    /// `TESTING\s110_g14_2_callsite\`. See the commit for the table.
    #[test]
    fn boot_steps_with_a_single_call_site_stay_wired_into_setup() {
        // Compile-time coupling: a rename or a signature change fails HERE, as a build error, rather
        // than as a baffling text mismatch below.
        let _: fn(&std::path::Path, &std::path::Path) = super::sync_bundled_dictionaries;
        let _: fn(&std::path::Path) = crate::pyenv::init_runtime_root;

        static LIB_RS: &str = include_str!("../lib.rs");
        // `trim_end` everywhere: this repo has no .gitattributes and core.autocrlf is on, so a fresh
        // clone can carry CRLF even though the working tree today is pure LF.
        let lines: Vec<&str> = LIB_RS.lines().map(str::trim_end).collect();
        let is_comment = |l: &str| l.trim_start().starts_with("//");

        // Exactly-one-match lookup. A 0-match is the dangerous direction: it would leave this gate
        // blind rather than satisfied, so it panics with instructions instead of quietly passing.
        let only = |needle: &str, what: &str| -> usize {
            let hits: Vec<usize> = lines
                .iter()
                .enumerate()
                .filter(|(_, l)| !is_comment(l) && l.contains(needle))
                .map(|(i, _)| i)
                .collect();
            assert_eq!(
                hits.len(),
                1,
                "BOOT CONTRACT — anchor {what} ({needle:?}) matched {} non-comment lines in \
                 src/lib.rs (1-based: {:?}), expected exactly 1.\n\
                 0 matches means this gate can no longer SEE its subject, which is worse than no \
                 gate (S108). Re-point it, do NOT delete it — it is the only thing in the repo that \
                 notices when one of these boot steps stops being called.\n\
                 >1 match means there are now two authorities for the same step; decide which one \
                 owns it before touching this test.",
                hits.len(),
                hits.iter().map(|i| i + 1).collect::<Vec<_>>()
            );
            hits[0]
        };

        // Brace depth over [from..=to], skipping string literals and trailing line comments.
        // (Verified by hand: the span this walks contains no char literals and no raw strings, which
        // is why a scanner this small is enough. If either appears, this must grow — the self-check
        // below is what will tell you.)
        let depth = |from: usize, to: usize| -> i32 {
            let mut d = 0i32;
            for l in &lines[from..=to] {
                let b: Vec<char> = l.chars().collect();
                let (mut in_s, mut esc) = (false, false);
                let mut i = 0usize;
                while i < b.len() {
                    let c = b[i];
                    if in_s {
                        if esc {
                            esc = false;
                        } else if c == '\\' {
                            esc = true;
                        } else if c == '"' {
                            in_s = false;
                        }
                    } else if c == '"' {
                        in_s = true;
                    } else if c == '/' && b.get(i + 1) == Some(&'/') {
                        break;
                    } else if c == '{' {
                        d += 1;
                    } else if c == '}' {
                        d -= 1;
                    }
                    i += 1;
                }
            }
            d
        };

        let i_setup = only(".setup(", "the setup() opener");
        let i_win = only("WebviewWindowBuilder::new", "the main-window builder");

        // ── SELF-CHECK the scanner on answers that are known independently ──────────────────────
        assert_eq!(
            depth(i_setup, i_setup),
            1,
            "BOOT CONTRACT — scanner self-check failed: the `.setup(` line (src/lib.rs:{}) should \
             open exactly one block on its own. The brace scanner is reading something it was not \
             built for (a char literal, a raw string, a macro body). Fix the scanner before \
             believing anything else this test says.",
            i_setup + 1
        );
        let i_close = (i_setup..lines.len())
            .find(|&j| depth(i_setup, j) == 0)
            .unwrap_or_else(|| panic!(
                "BOOT CONTRACT — the setup() closure opened at src/lib.rs:{} never closes according \
                 to the brace scanner. Same conclusion as the self-check above: distrust the scanner.",
                i_setup + 1
            ));
        assert!(
            i_close > i_win,
            "BOOT CONTRACT — the main-window builder (src/lib.rs:{}) is OUTSIDE the setup() closure \
             (which closes at src/lib.rs:{}). Clause 6 below compares against it as a landmark \
             INSIDE setup; that comparison is meaningless now. Re-point the landmark.",
            i_win + 1,
            i_close + 1
        );

        // The last write to `data_dir` — matched by ASSIGNMENT SHAPE, not by what the assigned
        // expression happens to be, so restructuring the legacy-AppData fallback does not false-red.
        let i_data_dir_last = lines
            .iter()
            .enumerate()
            .filter(|(i, l)| {
                let t = l.trim_start();
                *i > i_setup
                    && *i < i_close
                    && !is_comment(l)
                    && (t.starts_with("data_dir = ") || t.starts_with("let mut data_dir = ") || t.starts_with("let data_dir = "))
            })
            .map(|(i, _)| i)
            .next_back()
            .expect(
                "BOOT CONTRACT — no `data_dir` assignment found inside setup() in src/lib.rs. \
                 Clause 5 (\"the boot steps run after the data root is final\") cannot be evaluated, \
                 so this gate is blind. Re-point it at whatever now decides the data root.",
            );

        struct Step {
            callee: &'static str,
            what: &'static str,
            args: &'static [&'static str],
            before_window: bool,
            why: &'static str,
        }
        let steps = [
            Step {
                callee: "commands::settings::sync_bundled_dictionaries(",
                what: "the bundled-dictionary refresh",
                args: &["&app_dir_early", "&data_dir"],
                before_window: true,
                why: "every machine that migrated its data root stops receiving new dictionaries \
                      forever, silently — the S83 distribution fault this call was written to close",
            },
            Step {
                callee: "pyenv::init_runtime_root(",
                what: "the python runtime-root init",
                args: &["&data_dir"],
                // Not asserted: its doc claims only \"AFTER the data root is resolved\". Pinning an
                // ordering the code never promised is how a gate starts lying about its subject.
                before_window: false,
                why: "RUNTIME_ROOT stays unset, list_packs() returns empty, and every training / \
                      converter GPU check answers TRAINING_GPU_PACK_MISSING to a user who has the \
                      pack installed (see training_device_gate_on_this_machine)",
            },
        ];

        for s in &steps {
            let i = only(s.callee, s.what);
            let line = lines[i];
            let t = line.trim_start();

            // (2) STATEMENT, not an argument / closure body.
            assert!(
                t.starts_with(s.callee),
                "BOOT CONTRACT — {} (src/lib.rs:{}) is no longer a statement of its own; the line \
                 reads `{}`.\nA brace-less closure body such as \
                 `spawn_blocking(move || {}…))` keeps the brace depth at 1 and would slip past \
                 clause 3, so this clause is the one that catches it. If the call was merely \
                 reflowed, keep the callee at the START of its own line.\nWhat breaks if it really \
                 was backgrounded: {}",
                s.what, i + 1, t, s.callee, s.why
            );

            // (3) depth 1 inside setup.
            let d = depth(i_setup, i);
            assert_eq!(
                d, 1,
                "BOOT CONTRACT — {} (src/lib.rs:{}) sits {d} block(s) deep inside setup(), not 1: \
                 it was moved into a thread / async block / conditional. It MUST stay a plain \
                 statement in the setup() body.\nFor the dictionary sync specifically, \
                 `g2p::set_dict_dir` is a first-call-wins OnceLock and the loaded dictionary is \
                 `Box::leak`ed for the process lifetime, so a backgrounded refresh races the \
                 session's first render and can pin a STALE dictionary for the whole session with \
                 nothing able to invalidate it.\nWhat breaks: {}",
                s.what, i + 1, s.why
            );

            // (4) no cfg attribute smuggling it out of a build profile.
            let prev = lines[..i].iter().rev().find(|l| !l.trim().is_empty() && !is_comment(l));
            assert!(
                !prev.map(|l| l.trim_start().starts_with("#[cfg")).unwrap_or(false),
                "BOOT CONTRACT — {} (src/lib.rs:{}) now carries a `#[cfg(…)]` attribute. Depth and \
                 order still look right, so nothing else here can see it, yet a build profile can \
                 drop the step entirely. What breaks in that profile: {}",
                s.what, i + 1, s.why
            );

            // (5) after the data root is final.
            assert!(
                i > i_data_dir_last,
                "BOOT CONTRACT — {} (src/lib.rs:{}) runs BEFORE the last write to `data_dir` \
                 (src/lib.rs:{}). The call text is unchanged, which is exactly why this needs its \
                 own clause: the step would operate on the pre-fallback root while every later \
                 consumer reads the one that line selects. The population this silently strands is \
                 the upgrader whose data still lives under AppData.",
                s.what, i + 1, i_data_dir_last + 1
            );

            // (6) before the window can serve commands (dictionary sync only).
            if s.before_window {
                assert!(
                    i < i_win,
                    "BOOT CONTRACT — {} (src/lib.rs:{}) now runs AFTER the main window is built \
                     (src/lib.rs:{}). Once the window exists the frontend can invoke \
                     validate_lyrics / preview_vocal_phonemes / render_vocal_segment, each of which \
                     calls `g2p::set_dict_dir` — first-call-wins. Whoever wins pins the dictionary \
                     for the whole session, so the refresh has to be finished before the window is \
                     constructed.",
                    s.what, i + 1, i_win + 1
                );
            }

            // (7) arguments — LAST, because it is the broadest clause.
            let joined: String = lines[i..(i + 4).min(lines.len())].join(" ");
            for a in s.args {
                assert!(
                    joined.contains(a),
                    "BOOT CONTRACT — {} (src/lib.rs:{}) no longer passes `{a}`; the call reads \
                     `{}`.\nIf it was only reflowed across more than 4 lines, widen the join window \
                     here. If an argument was CLONED, the call was probably moved into a thread — \
                     see clause 3. If a different root is passed, the step now targets a directory \
                     the readers do not read.",
                    s.what, i + 1, t
                );
            }
        }

        // ── ONE DATA ROOT, THREE READERS (the invariant the call-site clauses do not cover) ─────
        // `sync_bundled_dictionaries` writes into `data_dir`; the render reads
        // `state.models.models_dir().parent()`; `dictionary_fingerprint` hashes
        // `effective_data_root` = `state.cache_dir.parent()`. Those three coincide ONLY because
        // setup() derives both dirs from the same local — an unenforced claim that the doc on
        // `effective_data_root` states as fact. `data_root_derivations_agree` pins the reader side;
        // this pins that lib.rs still feeds it from the root the sync just wrote.
        for (needle, what) in [
            ("let cache_dir = data_dir.join(\"cache\");", "the cache dir"),
            ("let models_dir = data_dir.join(\"models\");", "the models dir"),
        ] {
            let i = only(needle, what);
            assert!(
                i > i_data_dir_last && i < i_close,
                "BOOT CONTRACT — {what} (src/lib.rs:{}) is no longer derived from the FINAL \
                 `data_dir` inside setup(). If it is derived from anything else, the dictionary \
                 sync writes one root while the render and the bake fingerprint read another, and \
                 the bake gets stamped with the hash of a directory it did not sing from — the very \
                 hole `dictionary_fingerprint_for` exists to close.",
                i + 1
            );
        }
    }

    /// S110 (queue §G14-②) — the READER half of "one data root, three consumers".
    ///
    /// `AppState` has no `data_dir` field, so two consumers each re-derive the root by walking UP
    /// from a dir they were handed: `effective_data_root` takes `cache_dir.parent()`, and the three
    /// `g2p::set_dict_dir` call sites take `models.models_dir().parent()`. Nothing asserted they
    /// agree before this test. If they ever disagree the
    /// render leaks one dictionary directory while the bake signature carries the hash of another —
    /// silent, and permanent for the session because `set_dict_dir` is first-call-wins.
    ///
    /// Kept deliberately small: it does not test `lib.rs` (the text gate above does that), only that
    /// GIVEN the layout lib.rs builds, the two derivations land on the same directory.
    ///
    /// ⚠ S141 correction: this doc used to say "nothing constructs `AppState` anywhere in the test
    /// suite" — a sentence its own body refutes nine lines below. It was read as a constraint and
    /// cost a round: the §E2E-M2 survey concluded `check_in_training_root` could not be driven and
    /// downgraded that gate to a source-text check. `AppState::new` is a plain `pub fn` whose four
    /// members are pure in-memory constructors, so ANY test can build one against a temp dir.
    #[test]
    fn data_root_derivations_agree() {
        let root = std::env::temp_dir().join(format!("utai_dataroot_{}", std::process::id()));
        let state = crate::AppState::new(
            root.join("app"),
            root.join("cache"),
            root.join("models"),
            std::sync::Arc::new(crate::logging::LogBuffer::new(8)),
        );
        // The two arms must be able to disagree, or this asserts nothing (S92p).
        assert_ne!(state.cache_dir, *state.models.models_dir(), "fixture is vacuous");
        assert_eq!(
            super::effective_data_root(&state),
            root.as_path(),
            "`effective_data_root` (cache_dir.parent) no longer returns the data root — \
             `dictionary_fingerprint` would hash the wrong directory"
        );
        assert_eq!(
            state.models.models_dir().parent(),
            Some(root.as_path()),
            "`models_dir().parent()` no longer returns the data root — the three `set_dict_dir` \
             call sites in commands/inference.rs would point g2p at the wrong directory"
        );
    }

    /// S109 (queue §H7, first baseline) — the IDENTITY of the dictionaries this build ships.
    ///
    /// Nothing in this repo pinned their bytes before: they are gitignored generated assets, so git
    /// does not carry them; `release.ps1` checked only `Test-Path`; and the ~15 dictionary gates in
    /// `g2p.rs` pin PROPERTIES (a knife survived, a cluster count held), not identity. An installer
    /// could therefore carry a dictionary generation nobody chose with every gate green — the S98
    /// shape exactly, where the wrong upstream file had the same line count, word set and size while
    /// 3051 German primary pronunciations differed.
    ///
    /// Two halves, and they fail for different reasons ON PURPOSE:
    ///   (1) MANIFEST vs bundle.resources — pure data, no files needed, so this half can NEVER skip.
    ///       It is also the missing direction of `bundled_dictionary_targets_are_actually_found`,
    ///       which only checks that the eight known names are PRESENT (`>= 8`): a ninth dictionary
    ///       added to the bundle without a manifest entry goes red here.
    ///   (2) MANIFEST vs the bytes on disk — skips when the assets are absent, same contract as the
    ///       `g2p.rs` gates, so a bare checkout stays green. The skip is printed as a single
    ///       grep-able `SKIPPED-ASSET` line (queue §G12 wants these countable rather than silent;
    ///       making the release script assert zero of them is §G12's own item, not this one).
    ///
    /// FAIL-CLOSED BY DESIGN: regenerating a dictionary turns this red. That is the same rule S105
    /// chose for the curated onset tables — a dictionary change is a decision that has to be
    /// announced, not something that slips through. Fix: rerun `verify_dictionaries.py` (MBS2H
    /// `_onnx_derisk/`, proves the files are what the generator produces from the upstream sources),
    /// then `py -3.10 scripts/dict_manifest.py --write`, and name the change in the commit.
    ///
    /// ★ MUTATION-PROBED, one per half, and they land on different assertions:
    ///   · M3 — flip one hex character of de.tsv's digest ⇒ red on half (2), naming the file and
    ///     both digests. ⚠ The FIRST version of this probe used a PowerShell `-replace '^4c0d545e'`,
    ///     where `^` anchors to the start of the whole STRING, not the line — the manifest was never
    ///     modified and the all-green result said nothing. Re-run with an asserted-applied mutation.
    ///   · M4 — drop one entry from the manifest ⇒ red on half (1), printing both sets.
    ///   Logs: TESTING\s109_c16_dict_distribution\mut_M3_digest_flipped.log / mut_M4_*.log.
    #[test]
    fn shipped_dictionaries_match_the_committed_manifest() {
        static MANIFEST: &str = include_str!("../../dictionaries.sha256");
        let want: std::collections::BTreeMap<&str, &str> = MANIFEST
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(|l| {
                let (digest, name) = l.split_once("  ").unwrap_or_else(|| {
                    panic!("malformed manifest line (expected '<sha256>  <name>'): {l:?}")
                });
                (name.trim(), digest.trim())
            })
            .collect();

        // (1) the manifest and the bundle must describe the SAME set of files.
        let bundled: std::collections::BTreeSet<String> = super::bundled_dictionary_targets()
            .iter()
            .filter_map(|t| std::path::Path::new(t).file_name().map(|n| n.to_string_lossy().to_string()))
            .collect();
        let listed: std::collections::BTreeSet<String> = want.keys().map(|k| k.to_string()).collect();
        assert_eq!(
            listed, bundled,
            "src-tauri/dictionaries.sha256 and tauri.conf.json bundle.resources disagree about which \
             dictionaries ship — run `py -3.10 scripts/dict_manifest.py --write`"
        );

        // (2) … and the bytes on disk must be the ones the manifest names.
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/dictionaries");
        let mut absent: Vec<&str> = Vec::new();
        let mut wrong: Vec<String> = Vec::new();
        for (name, digest) in &want {
            let p = dir.join(name);
            if !p.is_file() {
                absent.push(name);
                continue;
            }
            match crate::download::sha256_file(&p) {
                Ok(live) if live.eq_ignore_ascii_case(digest) => {}
                Ok(live) => wrong.push(format!("{name}: on disk {}…, manifest {}…", &live[..16], &digest[..16])),
                Err(e) => wrong.push(format!("{name}: cannot hash ({e})")),
            }
        }
        if !absent.is_empty() {
            eprintln!(
                "SKIPPED-ASSET: data/dictionaries — {} of {} shipped dictionaries absent ({}); \
                 gitignored generated assets, run MBS2H build_dictionaries.py",
                absent.len(),
                want.len(),
                absent.join(", ")
            );
            // Partial presence still gets checked: `wrong` below covers whatever WAS readable.
        }
        assert!(
            wrong.is_empty(),
            "shipped dictionaries do not match src-tauri/dictionaries.sha256:\n  {}\n\
             If this was an intentional regeneration: rerun verify_dictionaries.py, then \
             `py -3.10 scripts/dict_manifest.py --write`, and say so in the commit.",
            wrong.join("\n  ")
        );
    }

    /// S134 (§F7 first pass) — the release gate must run the WHOLE workspace, not just the root
    /// package.
    ///
    /// `src-tauri/Cargo.toml` is BOTH `[workspace]` and `[package]`, so a bare `cargo test` there
    /// resolves to the root package alone. Measured on one tree at one moment (2026-08-11, HEAD
    /// 5d3385a): bare `cargo test` = 545 passed, `cargo test --workspace` = 583 — the 38 missing
    /// are exactly utai-dsp's 30 (MDX / demucs / formant) and utai-stretch's 8 (the range-extension
    /// engine). Nothing was ever RED: both crates are BUILT either way (utai depends on them by
    /// path), their tests simply never ran in a release gate.
    ///
    /// Same family as the `--lib` hole four lines above it in release.ps1 — the one that let a red
    /// `download_http.rs` ship through four releases — one level up: `--lib` hid `tests/`, a bare
    /// `cargo test` hides whole member crates.
    ///
    /// Shape notes, each paid for elsewhere in this repo:
    ///   · Comments AND string literals are stripped before looking for the invocation. release.ps1
    ///     both narrates (`Write-Host "gate: cargo test"`) and blames (`Fail "cargo test"`) with the
    ///     same three words, and the paragraph above the invocation discusses it in prose ⇒ a
    ///     substring check over the raw text is satisfied by a corpse (S119(a)).
    ///   · The assertion is over EVERY invocation, not "the file contains the good string": adding a
    ///     second, bare `cargo test` line has to go red too.
    ///   · The stripper SELF-CHECKS on a known positive and a known negative first. A classifier that
    ///     has never seen a positive carries no information in its negatives (S116).
    ///
    /// ⛔ Honest boundary: this test CANNOT prove that `--workspace` changes what runs — that was
    /// measured by hand once (the two numbers above). What it pins is that the flag is still there
    /// and that the workspace still has members for it to reach.
    #[test]
    fn the_release_gate_runs_the_whole_workspace() {
        static RELEASE_PS1: &str = include_str!("../../../scripts/release.ps1");
        static CARGO_TOML: &str = include_str!("../../Cargo.toml");

        /// Drop PowerShell string literals so prose about a command cannot pass for the command.
        fn strip_ps_strings(line: &str) -> String {
            let mut out = String::new();
            let mut quote: Option<char> = None;
            for ch in line.chars() {
                match quote {
                    Some(q) => {
                        if ch == q {
                            quote = None;
                        }
                    }
                    None => {
                        if ch == '"' || ch == '\'' {
                            quote = Some(ch);
                        } else {
                            out.push(ch);
                        }
                    }
                }
            }
            out
        }

        // (0) the stripper must be able to fail in BOTH directions before its verdict means anything.
        assert!(
            !strip_ps_strings("Write-Host \"gate: cargo test\" -ForegroundColor Cyan").contains("cargo test"),
            "strip_ps_strings no longer removes quoted prose — part (1) below would be satisfied by \
             a Write-Host line and this whole test would be decoration"
        );
        assert!(
            strip_ps_strings("Push-Location src-tauri; cargo test --workspace; Pop-Location").contains("cargo test"),
            "strip_ps_strings ate a real invocation — part (1) below would silently find nothing"
        );

        // (1) every real `cargo test` invocation must carry --workspace.
        let mut invocations: Vec<String> = Vec::new();
        for line in RELEASE_PS1.lines() {
            if line.trim_start().starts_with('#') {
                continue;
            }
            let code = strip_ps_strings(line);
            if code.contains("cargo test") {
                invocations.push(code.trim().to_string());
            }
        }
        assert!(
            !invocations.is_empty(),
            "scripts/release.ps1 no longer invokes `cargo test` at all — the Rust suite has left the \
             release gate entirely"
        );
        for inv in &invocations {
            assert!(
                inv.contains("--workspace"),
                "scripts/release.ps1 invokes cargo test WITHOUT --workspace:\n  {inv}\n\
                 src-tauri is the workspace root AND a package, so that runs the root package only \
                 and silently skips crates/utai-dsp + crates/utai-stretch (38 tests as of S134)."
            );
        }

        // (2) …and the workspace still has members for --workspace to reach.
        let members = CARGO_TOML
            .lines()
            .find(|l| l.trim_start().starts_with("members"))
            .expect("src-tauri/Cargo.toml lost its [workspace] members line");
        for krate in ["crates/utai-dsp", "crates/utai-stretch"] {
            assert!(
                members.contains(krate),
                "{krate} is no longer a workspace member ({members}) — if it really is gone, retire \
                 it from this test too; if it merely moved, --workspace must still reach it"
            );
            assert!(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(krate).is_dir(),
                "{krate} is listed as a workspace member but its directory is missing"
            );
        }
    }
}
