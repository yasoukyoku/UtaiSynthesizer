//! GPU adapter enumeration + VRAM forensics — THE single source (S68b).
//!
//! Why this exists: the WMI/PowerShell probe (`query_gpu_adapters`) failed ENTIRELY on a
//! community RTX 3080 box ("Hardware: GPUs [Unknown GPU]") and one probe failure cascaded
//! into: CUDA greyed out in Settings, the NVIDIA training pack hidden from downloads, and
//! silent CPU training — while DirectML inference worked the whole time, proving the DXGI
//! stack was healthy. DXGI is subprocess-free and works wherever a display stack exists,
//! so it is now the PRIMARY enumeration; WMI stays as a fallback (settings.rs).
//!
//! ORDERING CONTRACT (do not break): `DxgiAdapterInfo.index` is the raw
//! `IDXGIFactory1::EnumAdapters1` ordinal — exactly the id space ORT's DirectML EP
//! consumes for an explicit device_id (verified in ONNX Runtime v1.24.4
//! dml_provider_factory.cc: `CreateDXGIFactory2 → EnumAdapters1(device_id)`, software
//! adapters occupy indices and merely THROW if picked). Never compact or re-sort the
//! list. NOTE the asymmetry: Auto-DML (no device_id) goes through the DML2/DXCore
//! "high performance" path and may resolve to a DIFFERENT adapter than index 0 — Auto
//! behavior is deliberately untouched by everything in this module.

#![cfg_attr(not(windows), allow(dead_code))]

/// One DXGI adapter, in EnumAdapters1 order (== ORT DML device_id space).
#[derive(Clone, Debug)]
pub struct DxgiAdapterInfo {
    /// Raw EnumAdapters1 ordinal — the value `DeviceConfig::DirectMl { device_id }` takes.
    pub index: u32,
    pub name: String,
    /// "nvidia" | "amd" | "intel" | "other" (PCI vendor id — same classification as
    /// settings.rs's WMI PNPDeviceID path).
    pub vendor: &'static str,
    /// Dedicated video memory in MB (0 for virtual/software adapters).
    pub dedicated_mb: u64,
    /// DXGI_ADAPTER_FLAG_SOFTWARE or the Basic Render Driver (VEN 0x1414 DEV 0x8c) —
    /// ORT's DML EP refuses these; the picker greys them out but keeps the index slot.
    pub software: bool,
    /// AdapterLuid as raw little-endian bytes (LowPart then HighPart) — the EXACT
    /// cross-API identity: cudaDeviceGetLuid returns the same 8 bytes, which is how an
    /// Auto-mode preferred DXGI adapter maps to a CUDA ordinal without name guessing.
    pub luid: [u8; 8],
}

pub fn vendor_from_pci_id(vendor_id: u32) -> &'static str {
    match vendor_id {
        0x10DE => "nvidia",
        0x1002 => "amd",
        0x8086 => "intel",
        _ => "other",
    }
}

/// Enumerate DXGI adapters in EnumAdapters1 order. Empty on any failure — callers fall
/// back to the WMI probe. Enumeration is a few COM calls (~µs); no caching, so eGPU
/// hotplug / driver resets never serve a stale list.
#[cfg(windows)]
pub fn dxgi_adapters() -> Vec<DxgiAdapterInfo> {
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory1, IDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE,
    };

    let factory: IDXGIFactory1 = match unsafe { CreateDXGIFactory1() } {
        Ok(f) => f,
        Err(e) => {
            tracing::debug!("DXGI factory creation failed: {e}");
            return Vec::new();
        }
    };
    let mut out = Vec::new();
    for index in 0..64u32 {
        let adapter = match unsafe { factory.EnumAdapters1(index) } {
            Ok(a) => a,
            Err(_) => break, // DXGI_ERROR_NOT_FOUND = end of list
        };
        let desc = match unsafe { adapter.GetDesc1() } {
            Ok(d) => d,
            Err(_) => continue,
        };
        let len = desc.Description.iter().position(|&c| c == 0).unwrap_or(desc.Description.len());
        let name = String::from_utf16_lossy(&desc.Description[..len]).trim().to_string();
        // Same rule as ORT's IsSoftwareAdapter (dml_provider_factory.cc).
        let software = (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0
            || (desc.VendorId == 0x1414 && desc.DeviceId == 0x8c);
        let mut luid = [0u8; 8];
        luid[..4].copy_from_slice(&desc.AdapterLuid.LowPart.to_le_bytes());
        luid[4..].copy_from_slice(&desc.AdapterLuid.HighPart.to_le_bytes());
        out.push(DxgiAdapterInfo {
            index,
            name,
            vendor: vendor_from_pci_id(desc.VendorId),
            dedicated_mb: (desc.DedicatedVideoMemory / (1024 * 1024)) as u64,
            software,
            luid,
        });
    }
    out
}

#[cfg(not(windows))]
pub fn dxgi_adapters() -> Vec<DxgiAdapterInfo> {
    Vec::new()
}

/// Adapter name for an explicit DML device_id (session-build logging).
pub fn adapter_name(index: u32) -> Option<String> {
    dxgi_adapters().into_iter().find(|a| a.index == index).map(|a| a.name)
}

/// Hardware-inventory fragment for the startup "Hardware:" log line:
/// "0: NVIDIA GeForce RTX 3070 Laptop GPU (8192 MB), 1: Intel(R) UHD Graphics 730 (128 MB)".
/// Indices ARE the DML device_id space — a community log now records both identity and
/// ordering, closing the "which adapter did DML use" blind spot. None when DXGI failed.
pub fn inventory_line() -> Option<String> {
    let fragments: Vec<String> = dxgi_adapters()
        .iter()
        .filter(|a| !a.software)
        .map(|a| format!("{}: {} ({} MB)", a.index, a.name, a.dedicated_mb))
        .collect();
    if fragments.is_empty() {
        // Also covers "only software adapters enumerate" — the caller's WMI-derived
        // fallback name must win over an empty bracket pair (review round 1).
        return None;
    }
    Some(fragments.join(", "))
}

/// Per-adapter LOCAL video-memory usage/budget, one compact fragment for forensic log
/// lines: "vram GPU0 2151/7222 MB, GPU1 120/8032 MB". Empty string when unavailable.
///
/// This is the observable S67c never had: the community 20%-crash log proved system
/// commit returns in full after MSST teardown (2994→223 MB) with 23 GB avail at death —
/// the GPU side (driver residency / budget) was the one unwatched state. Usage/Budget
/// come from IDXGIAdapter3::QueryVideoMemoryInfo (OS ground truth, per process-visible
/// segment; cheap enough for per-new-shape use). All hardware adapters are reported
/// because Auto-DML doesn't tell us which one it picked — the one that climbs is the one
/// in use.
#[cfg(windows)]
pub fn vram_stamp() -> String {
    use windows::core::Interface;
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory1, IDXGIAdapter3, IDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE,
        DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
    };

    let factory: IDXGIFactory1 = match unsafe { CreateDXGIFactory1() } {
        Ok(f) => f,
        Err(_) => return String::new(),
    };
    let mut parts = Vec::new();
    for index in 0..64u32 {
        let adapter = match unsafe { factory.EnumAdapters1(index) } {
            Ok(a) => a,
            Err(_) => break,
        };
        let Ok(desc) = (unsafe { adapter.GetDesc1() }) else { continue };
        let software = (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0
            || (desc.VendorId == 0x1414 && desc.DeviceId == 0x8c);
        if software {
            continue;
        }
        let Ok(a3) = adapter.cast::<IDXGIAdapter3>() else { continue };
        let mut info = Default::default();
        if unsafe { a3.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info) }.is_ok() {
            parts.push(format!(
                "GPU{} {}/{} MB",
                index,
                info.CurrentUsage / (1024 * 1024),
                info.Budget / (1024 * 1024)
            ));
        }
    }
    if parts.is_empty() {
        String::new()
    } else {
        format!("vram {}", parts.join(", "))
    }
}

#[cfg(not(windows))]
pub fn vram_stamp() -> String {
    String::new()
}

/// One CUDA device as the inference picker offers it. `index` is the CUDA RUNTIME
/// ordinal — the value ORT's CUDA EP feeds to cudaSetDevice.
#[derive(Clone, Debug)]
pub struct CudaDeviceInfo {
    pub index: u32,
    pub name: String,
}

/// CUDA devices in CUDA-runtime ordinal order, labeled with nvidia-smi names.
///
/// The ordinal space is CUDA's own (default CUDA_DEVICE_ORDER=FASTEST_FIRST) which does
/// NOT match nvidia-smi's PCI order on multi-card boxes — the S67 WMI-index lesson, one
/// namespace over. We deliberately do NOT set CUDA_DEVICE_ORDER=PCI_BUS_ID: that would
/// silently change which card `device_id: 0` means for every existing multi-GPU install
/// (a 1050Ti+3080 box would flip its default from the 3080 to whichever sits first on
/// the PCI bus). Instead the mapping goes ordinal → cudaDeviceGetPCIBusId (two stable
/// C-ABI cudart calls, no context creation) → nvidia-smi pci.bus_id → name. If cudart
/// isn't loadable (CUDA EP can't run anyway), we fall back to nvidia-smi order with a
/// positional caveat only multi-card boxes could ever notice.
#[cfg(windows)]
pub fn cuda_devices() -> Vec<CudaDeviceInfo> {
    let smi = nvidia_smi_name_by_pci();
    match cudart_pci_bus_ids() {
        Some(pcis) => pcis
            .into_iter()
            .enumerate()
            .map(|(i, pci)| CudaDeviceInfo {
                index: i as u32,
                name: smi
                    .iter()
                    .find(|(p, _)| *p == pci)
                    .map(|(_, n)| n.clone())
                    .unwrap_or_else(|| format!("CUDA device {i}")),
            })
            .collect(),
        None => smi
            .into_iter()
            .enumerate()
            .map(|(i, (_, name))| CudaDeviceInfo { index: i as u32, name })
            .collect(),
    }
}

#[cfg(not(windows))]
pub fn cuda_devices() -> Vec<CudaDeviceInfo> {
    Vec::new()
}

/// Compute capability (major, minor) for a CUDA RUNTIME ordinal (the ORT CUDA EP's
/// device space — same ordinal cudaSetDevice uses), via cudart cudaDeviceGetAttribute.
/// None = cudart unloadable / the call failed. ⚠ Callers consume `None` as NOT SUPPORTED
/// (fail-CLOSED — the S74b package rule), except `cuda_device_label`, which only formats a log
/// suffix. Do not reintroduce a `map_or(true, …)` here: that is the fail-open this predicate was
/// changed away from.
#[cfg(windows)]
pub fn cuda_compute_cap(ordinal: u32) -> Option<(i32, i32)> {
    type GetAttr = unsafe extern "C" fn(*mut i32, i32, i32) -> i32;
    // cudart ABI: cudaDevAttrComputeCapabilityMajor = 75, ...Minor = 76 (stable enum values).
    unsafe {
        let f: GetAttr = std::mem::transmute(cudart_proc(b"cudaDeviceGetAttribute\0")?);
        let (mut major, mut minor) = (0i32, 0i32);
        if f(&mut major, 75, ordinal as i32) != 0 {
            return None;
        }
        if f(&mut minor, 76, ordinal as i32) != 0 {
            return None;
        }
        Some((major, minor))
    }
}

#[cfg(not(windows))]
pub fn cuda_compute_cap(_ordinal: u32) -> Option<(i32, i32)> {
    None
}

/// Compute-capability FLOOR shared by BOTH CUDA lanes — `cc10` = major*10+minor (sm_8.6 → 86,
/// sm_12.0 → 120). sm_75 (Turing) is not "what CUDA 12.9 happens to contain": it is the
/// CUDA-13 target configuration both sides are deliberately aligned on (the nv-cu130 TRAINING
/// wheels already use it). Running training and inference on two different CUDA configurations
/// is the thing this constant exists to prevent — do not derive a second floor anywhere.
pub const CUDA_CC10_FLOOR: i32 = 75;

/// ★TEMPORARY, INFERENCE ONLY. Microsoft ships no CUDA-13 ONNX Runtime yet, and the CUDA-12.9
/// build we are stuck on has only broken PTX for Blackwell (sm_100 / sm_120 — ORT #26177,
/// #26245), so RTX 50 cards must be kept off the INFERENCE CUDA lane and use DirectML. Training
/// is already on cu130 PyTorch and has no such limit.
///
/// ⚠ This is step ⑨ of the CUDA-13 migration checklist: when inference moves to an ORT with a
/// CUDA-13 build, DELETE this constant and its single use in `cuda_cc_supported_inference`.
/// Both lanes are then literally the same predicate and RTX 50 is released everywhere at once.
/// Do NOT delete it earlier because "the two sides should be aligned" — they are aligned on the
/// FLOOR; this line is the one honest, dated exception.
const INFERENCE_ORT_HAS_NO_BLACKWELL_YET: bool = true;

/// Can the nv-cu130 TRAINING pack run on this compute capability?
pub fn cuda_cc_supported_training(cc10: i32) -> bool {
    cc10 >= CUDA_CC10_FLOOR
}

/// Can the shipped INFERENCE CUDA package (ORT CUDA build + runtime/cuda DLLs) run on this
/// compute capability? = the shared floor, minus the temporary Blackwell exception above.
/// THE single predicate behind every inference-CUDA decision: whether the download entry and
/// the device option are shown at all, whether `init_ort_runtime` loads the CUDA ORT build, and
/// whether an already-installed package counts as usable or as reclaimable storage.
pub fn cuda_cc_supported_inference(cc10: i32) -> bool {
    cuda_cc_supported_training(cc10) && !(INFERENCE_ORT_HAS_NO_BLACKWELL_YET && cc10 >= 100)
}

/// Cached ": <name>, sm_XY" suffix for logging WHICH CUDA device an ORT session bound to
/// (S74: CUDA build previously logged only the ordinal number — a community CUDA-fail log
/// told us nothing about the card). Empty when the ordinal is unknown / cudart+smi
/// unavailable. cuda_devices() shells nvidia-smi, so the whole ordinal→(name,cc) table is
/// built ONCE per process (session builds can be frequent). Only ever called after a CUDA
/// session commits, so cudart is loaded by then.
pub fn cuda_device_label(ordinal: u32) -> String {
    static CACHE: std::sync::OnceLock<Vec<(u32, String, Option<(i32, i32)>)>> =
        std::sync::OnceLock::new();
    let table = CACHE.get_or_init(|| {
        cuda_devices()
            .into_iter()
            .map(|d| {
                let cc = cuda_compute_cap(d.index);
                (d.index, d.name, cc)
            })
            .collect()
    });
    table
        .iter()
        .find(|(i, _, _)| *i == ordinal)
        .map(|(_, name, cc)| {
            let sm = cc.map(|(a, b)| format!(", sm_{a}{b}")).unwrap_or_default();
            format!(": {name}{sm}")
        })
        .unwrap_or_default()
}

/// Proc address from a named DLL via plain kernel32 FFI. cudart64_12.dll reaches PATH
/// through setup_cuda_dll_paths whenever the CUDA runtime is installed; nvcuda.dll
/// (the CUDA DRIVER API) lives in System32 whenever an NVIDIA driver is installed.
#[cfg(windows)]
unsafe fn dll_proc(dll: &[u8], name: &[u8]) -> Option<isize> {
    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryA(name: *const u8) -> isize;
        fn GetProcAddress(module: isize, name: *const u8) -> isize;
    }
    let module = LoadLibraryA(dll.as_ptr());
    if module == 0 {
        return None;
    }
    let f = GetProcAddress(module, name.as_ptr());
    if f == 0 {
        None
    } else {
        Some(f)
    }
}

#[cfg(windows)]
unsafe fn cudart_proc(name: &[u8]) -> Option<isize> {
    dll_proc(b"cudart64_12.dll\0", name)
}

#[cfg(windows)]
fn cudart_device_count() -> Option<i32> {
    type GetCount = unsafe extern "C" fn(*mut i32) -> i32;
    unsafe {
        let count_fn: GetCount = std::mem::transmute(cudart_proc(b"cudaGetDeviceCount\0")?);
        let mut count: i32 = 0;
        if count_fn(&mut count) != 0 || count <= 0 {
            return None;
        }
        Some(count)
    }
}

/// PCI bus ids per CUDA ordinal via cudart. None = cudart not loadable / calls failed.
#[cfg(windows)]
fn cudart_pci_bus_ids() -> Option<Vec<String>> {
    type GetPciBusId = unsafe extern "C" fn(*mut u8, i32, i32) -> i32;
    unsafe {
        let pci_fn: GetPciBusId = std::mem::transmute(cudart_proc(b"cudaDeviceGetPCIBusId\0")?);
        let count = cudart_device_count()?;
        let mut out = Vec::with_capacity(count as usize);
        for dev in 0..count {
            let mut buf = [0u8; 64];
            if pci_fn(buf.as_mut_ptr(), buf.len() as i32, dev) != 0 {
                return None;
            }
            let len = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
            out.push(normalize_pci_bus_id(&String::from_utf8_lossy(&buf[..len])));
        }
        Some(out)
    }
}

/// CUDA ordinal for a DXGI adapter. Two exact hops, ZERO enumeration-order assumptions:
/// the DRIVER API (nvcuda.dll) links LUID ↔ PCI bus id on one device handle
/// (cuDeviceGetLuid + cuDeviceGetPCIBusId), and the RUNTIME ordinal links to the same
/// PCI id via cudaDeviceGetPCIBusId. NB: cudart exports NO LUID query — probed on the
/// dev box, `cudaDeviceGetLuid` isn't in cudart64_12.dll at all (docs notwithstanding);
/// only the driver API carries it. None = non-CUDA adapter / drivers unavailable — the
/// Auto probe then falls back to ORT's default device (pre-S68b behavior).
#[cfg(windows)]
pub fn cuda_ordinal_for_dxgi(dxgi_index: u32) -> Option<u32> {
    let target = dxgi_adapters().into_iter().find(|a| a.index == dxgi_index)?.luid;
    let pci = driver_pci_for_luid(target)?;
    cudart_pci_bus_ids()?.into_iter().position(|p| p == pci).map(|i| i as u32)
}

/// PCI bus id of the CUDA device whose LUID matches, via the CUDA driver API.
#[cfg(windows)]
fn driver_pci_for_luid(target: [u8; 8]) -> Option<String> {
    type CuInit = unsafe extern "C" fn(u32) -> i32;
    type CuCount = unsafe extern "C" fn(*mut i32) -> i32;
    type CuGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
    type CuLuid = unsafe extern "C" fn(*mut u8, *mut u32, i32) -> i32;
    type CuPci = unsafe extern "C" fn(*mut u8, i32, i32) -> i32;
    const NVCUDA: &[u8] = b"nvcuda.dll\0";
    unsafe {
        let init: CuInit = std::mem::transmute(dll_proc(NVCUDA, b"cuInit\0")?);
        let count_fn: CuCount = std::mem::transmute(dll_proc(NVCUDA, b"cuDeviceGetCount\0")?);
        let get_fn: CuGet = std::mem::transmute(dll_proc(NVCUDA, b"cuDeviceGet\0")?);
        let luid_fn: CuLuid = std::mem::transmute(dll_proc(NVCUDA, b"cuDeviceGetLuid\0")?);
        let pci_fn: CuPci = std::mem::transmute(dll_proc(NVCUDA, b"cuDeviceGetPCIBusId\0")?);
        if init(0) != 0 {
            return None; // idempotent; ORT's CUDA EP calls it too
        }
        let mut count: i32 = 0;
        if count_fn(&mut count) != 0 || count <= 0 {
            return None;
        }
        for i in 0..count {
            let mut dev: i32 = 0;
            if get_fn(&mut dev, i) != 0 {
                continue;
            }
            let mut luid = [0u8; 8];
            let mut node_mask: u32 = 0;
            if luid_fn(luid.as_mut_ptr(), &mut node_mask, dev) != 0 || luid != target {
                continue;
            }
            let mut buf = [0u8; 64];
            if pci_fn(buf.as_mut_ptr(), buf.len() as i32, dev) != 0 {
                return None;
            }
            let len = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
            return Some(normalize_pci_bus_id(&String::from_utf8_lossy(&buf[..len])));
        }
    }
    None
}

#[cfg(not(windows))]
pub fn cuda_ordinal_for_dxgi(_dxgi_index: u32) -> Option<u32> {
    None
}

/// Vendor of a USABLE (present, non-software) DXGI adapter by index. None ⇒ the pick
/// can't be honored (stale index / software adapter): the startup ORT-build pick then
/// treats it as no-preference, and build_session_auto falls back to the default
/// adapter with a WARN instead of letting ORT throw and land Auto on CPU.
pub fn adapter_vendor(index: u32) -> Option<&'static str> {
    dxgi_adapters()
        .into_iter()
        .find(|a| a.index == index && !a.software)
        .map(|a| a.vendor)
}

/// (pci_bus_id, name) rows from nvidia-smi. Empty on any failure.
#[cfg(windows)]
fn nvidia_smi_name_by_pci() -> Vec<(String, String)> {
    use std::os::windows::process::CommandExt;
    let out = match std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=pci.bus_id,name", "--format=csv,noheader"])
        .creation_flags(crate::util::CREATE_NO_WINDOW)
        .output()
    {
        Ok(o) if o.status.success() => o,
        _ => return Vec::new(),
    };
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| {
            // pci.bus_id is first so a comma inside the NAME can't shear the row.
            let (pci, name) = l.split_once(',')?;
            Some((normalize_pci_bus_id(pci.trim()), name.trim().to_string()))
        })
        .collect()
}

/// Canonicalize a PCI bus id for comparison: nvidia-smi prints an 8-hex-digit domain
/// ("00000000:2B:00.0"), cudart a 4-digit one ("0000:2b:00.0"); case differs too.
fn normalize_pci_bus_id(raw: &str) -> String {
    let lower = raw.trim().to_ascii_lowercase();
    match lower.split_once(':') {
        Some((domain, rest)) => {
            let dom = u32::from_str_radix(domain, 16).unwrap_or(0);
            format!("{dom:04x}:{rest}")
        }
        None => lower,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pci_bus_id_normalization_bridges_smi_and_cudart() {
        assert_eq!(normalize_pci_bus_id("00000000:2B:00.0"), "0000:2b:00.0");
        assert_eq!(normalize_pci_bus_id("0000:2b:00.0"), "0000:2b:00.0");
        assert_eq!(normalize_pci_bus_id(" 00000000:01:00.0 "), "0000:01:00.0");
    }

    // Dev machine ground truth (single NVIDIA card, healthy display stack): DXGI must
    // enumerate at least one hardware adapter — this is the fallback-ordering contract
    // the picker and the DML device_id space both stand on.
    #[cfg(windows)]
    #[test]
    fn dxgi_enumerates_hardware_adapters() {
        let adapters = dxgi_adapters();
        assert!(!adapters.is_empty(), "DXGI enumeration returned nothing");
        assert!(adapters.iter().any(|a| !a.software), "no hardware adapter found");
        // Indices must be the raw enumeration ordinals (contract with ORT DML device_id).
        for (i, a) in adapters.iter().enumerate() {
            assert_eq!(a.index as usize, i);
        }
    }

    // vram_stamp rides on every memory_stamp() forensic line — it must yield per-GPU
    // usage/budget on a healthy display stack (empty only when DXGI itself is down).
    #[cfg(windows)]
    #[test]
    fn vram_stamp_reports_hardware_adapters() {
        let stamp = vram_stamp();
        assert!(stamp.starts_with("vram GPU"), "unexpected stamp: {stamp:?}");
        assert!(stamp.contains("/"), "no usage/budget pair: {stamp:?}");
    }

    // The Auto-mode preferred-GPU bridge: DXGI LUID ↔ cudaDeviceGetLuid must agree on
    // the dev box (single NVIDIA card = CUDA ordinal 0), and a non-NVIDIA adapter must
    // never map. Skips (with a note) when cudart isn't resolvable in the bare test PATH.
    #[cfg(windows)]
    #[test]
    fn cuda_ordinal_maps_by_luid() {
        if cudart_device_count().is_none() {
            eprintln!("cudart64_12.dll not resolvable in test PATH — LUID mapping not exercised");
            return;
        }
        let adapters = dxgi_adapters();
        let nvidia = adapters.iter().find(|a| a.vendor == "nvidia");
        if let Some(a) = nvidia {
            assert_eq!(cuda_ordinal_for_dxgi(a.index), Some(0), "single-N box maps to ordinal 0");
        }
        for a in adapters.iter().filter(|a| a.vendor != "nvidia") {
            assert_eq!(cuda_ordinal_for_dxgi(a.index), None, "non-NVIDIA adapter must not map: {}", a.name);
        }
    }
}
