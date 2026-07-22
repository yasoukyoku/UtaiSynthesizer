//! Embedded Python runtime packs (S42 Phase A).
//!
//! A *pack* = one fully self-contained CPython (python-build-standalone, msvc-shared,
//! flat `Lib\site-packages` — NO venv, see s42_training_env_design.md §2.2) plus a
//! `pack.json` describing it. One directory per pack under `<data_root>/runtimes/`:
//!
//!   <data_root>/runtimes/
//!     runtime-cpu-v1/
//!       pack.json            ← presence = installed (scan-based discovery, no registry)
//!       envtest.json         ← latest self-test report (written by utai_train.envtest)
//!       python/python.exe    ← the interpreter (invoked as `python.exe -m ...`)
//!     .staging/              ← in-flight extractions; the staging→final DIRECTORY
//!                              RENAME is the install commit point, so a torn install
//!                              can never be mistaken for a pack
//!
//! Distribution: `<id>.manifest.json` + `<id>.tar.zst` (split into `.partNN` volumes
//! when a host caps file size — GH releases: 2 GiB) hosted on HF/GH. The manifest
//! carries per-part sha256; parts stream through MultiFileReader → zstd → tar with no
//! joined intermediate copy.
//!
//! Variant strategy (design §2.1): ONE unified dependency set (training ∪ converter),
//! built per torch backend: cpu / nv-cu130 / xpu / amd. Any installed pack can serve
//! the CONVERTER role (CPU-bound scripts); the TRAINING role stays on the dev .venv
//! until Phase B wires it to packs.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use serde::{Deserialize, Serialize};

use crate::{Result, UtaiError};

fn err(msg: impl Into<String>) -> UtaiError {
    UtaiError::Pyenv(msg.into())
}

/// Render `e` with its full `source()` chain — "outer: cause: root (os error N)".
/// tar's TarError Display prints ONLY its own desc; the real io cause (os error
/// 5/87/112…) lives exclusively in `source()`, so a plain `{e}` drops it — S68d: the
/// E:\ EXTRACT_FAILED field report ended at the file path and left the actual OS
/// failure a guess. A link whose text is already present is skipped (some wrappers
/// duplicate their cause's Display). Appended text is io-error prose — it can never
/// contain another SCREAMING_SNAKE code or the CANCELLED sentinel, so the frontend's
/// substring code matcher keeps working (backendError.ts).
fn error_chain(e: &(dyn std::error::Error + 'static)) -> String {
    let mut acc = e.to_string();
    let mut cur = e.source();
    while let Some(src) = cur {
        let s = src.to_string();
        if !acc.contains(&s) {
            acc.push_str(": ");
            acc.push_str(&s);
        }
        cur = src.source();
    }
    acc
}

// ─── runtime root ───────────────────────────────────────────────────────────

static RUNTIME_ROOT: OnceLock<PathBuf> = OnceLock::new();

/// Called once from lib.rs setup AFTER the data root is resolved (incl. the legacy
/// AppData fallback) — pack discovery/installation is rooted here. Harnesses that
/// never call it (unit tests, bare cargo test) simply see "no packs".
pub fn init_runtime_root(data_root: &Path) {
    let _ = RUNTIME_ROOT.set(data_root.join("runtimes"));
}

pub fn runtime_root() -> Option<&'static PathBuf> {
    RUNTIME_ROOT.get()
}

/// Non-ASCII install paths are the single most reproducible way to break an embedded
/// CPython + torch on Windows (DLL loader + multiprocessing spawn both choke) — refuse
/// early with an actionable message instead of failing later with `DLL load failed`.
pub fn ensure_ascii_path(p: &Path) -> Result<()> {
    let ok = p.to_str().map(|s| s.is_ascii()).unwrap_or(false);
    if ok {
        Ok(())
    } else {
        Err(err(format!(
            "RUNTIME_PATH_NON_ASCII: {}",
            p.display()
        )))
    }
}

// ─── pack model ─────────────────────────────────────────────────────────────

/// `pack.json` written by the pack builder (training/packs/build_pack.py) into the
/// archive root. Tolerant deserialization — future builders may add keys.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackMeta {
    #[serde(default)]
    pub schema: u32,
    pub id: String,
    pub variant: String,
    /// Monotonic per-variant version (the vN in the id) — same-variant coexistence
    /// picks the HIGHEST (upgrade path: install v2 next to v1, delete v1 after its
    /// envtest passes). Older pack.json without the field reads as 0.
    #[serde(default)]
    pub version: u32,
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub python: String,
    #[serde(default)]
    pub torch: String,
    #[serde(default)]
    pub disk_bytes: u64,
    #[serde(default)]
    pub built: String,
}

/// Whether `s` is safe as a SINGLE path component under our control dirs. Pack ids
/// and manifest part names come from REMOTE json — without this, a hostile/corrupt
/// manifest could rename-commit outside the runtimes root or write parts through
/// `..`/absolute paths (audit S42). Also enforces the ASCII invariant.
pub fn is_safe_component(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 120
        && !s.starts_with('.')
        && s.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// An installed pack as reported to the frontend.
#[derive(Debug, Clone, Serialize)]
pub struct PackStatus {
    #[serde(flatten)]
    pub meta: PackMeta,
    pub path: String,
    /// Parsed envtest.json (None = self-test never ran). The frontend reads
    /// `overall` ("pass"/"fail") for the badge.
    pub envtest: Option<serde_json::Value>,
    /// S74b: can THIS machine actually run this installed pack (settings::variant_supported)?
    /// Filled by get_runtime_env_info, which is where the hardware facts live — list_packs reads
    /// only on-disk facts and leaves it `true`. An installed-but-unsupported pack is NOT hidden:
    /// it keeps its card, its self-test button (a driver fix must be re-testable) and its delete
    /// button, and says WHY it can't be used. Hiding it would strip exactly the affordances the
    /// user needs, which is how a package becomes invisible dead weight.
    pub supported: bool,
    /// S74b: the self-test report was produced on DIFFERENT hardware than this machine has now
    /// (its `machine` stamp disagrees). The badge then says "re-run the self-test" instead of
    /// showing a verdict that no longer describes anything. Reports predating the stamp are NOT
    /// called stale — absence of evidence is not evidence of a swap, and flagging every existing
    /// install once would be noise. Filled by get_runtime_env_info.
    pub envtest_stale: bool,
}

pub fn pack_python(pack_dir: &Path) -> PathBuf {
    pack_dir.join("python").join("python.exe")
}

/// Scan-based discovery: every `<root>/<dir>/pack.json` (skipping dot-dirs like
/// `.staging`) is an installed pack. No registry file to drift out of sync.
pub fn list_packs() -> Vec<PackStatus> {
    let Some(root) = runtime_root() else { return vec![] };
    let Ok(entries) = std::fs::read_dir(root) else { return vec![] };
    let mut packs = Vec::new();
    for entry in entries.flatten() {
        let dir = entry.path();
        if !dir.is_dir() {
            continue;
        }
        if dir
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.starts_with('.'))
            .unwrap_or(true)
        {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(dir.join("pack.json")) else { continue };
        let Ok(meta) = serde_json::from_str::<PackMeta>(&text) else {
            tracing::warn!("unparseable pack.json in {} — ignoring", dir.display());
            continue;
        };
        let envtest = std::fs::read_to_string(dir.join("envtest.json"))
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok());
        packs.push(PackStatus {
            meta,
            path: dir.to_string_lossy().to_string(),
            envtest,
            supported: true, // hardware-independent here; get_runtime_env_info decides
            envtest_stale: false, // ditto
        });
    }
    packs.sort_by(|a, b| a.meta.id.cmp(&b.meta.id));
    packs
}

pub fn find_pack(id: &str) -> Option<PackStatus> {
    list_packs().into_iter().find(|p| p.meta.id == id)
}

/// The CONVERTER-role interpreter (S42 — replaces the 7 scattered
/// `find_python(app_dir/converter, app_dir)` call sites). Priority:
///   1. dev venv `converter/.venv` (dev machines keep their known-good env);
///   2. best installed runtime pack — any variant runs the CPU-bound converter
///      scripts; GPU variants first so a machine holding only `nv-cu130` needs no
///      extra cpu pack;
///   3. the manual `<app_dir>/python/python.exe` slot;
///   4. bare `python` on PATH (dev fallback).
pub fn converter_python(app_dir: &Path) -> PathBuf {
    let venv = app_dir
        .join("converter")
        .join(".venv")
        .join("Scripts")
        .join("python.exe");
    if venv.exists() {
        return venv;
    }
    const CONVERTER_PRIORITY: [&str; 4] = ["nv-cu130", "amd", "xpu", "cpu"];
    let packs = list_packs();
    for variant in CONVERTER_PRIORITY {
        // Same-variant coexistence (v1 + v2 during an upgrade): the NEWEST version
        // wins — id lexicographic order would pick v1 forever (and sort v10 < v2).
        if let Some(p) = packs
            .iter()
            .filter(|p| p.meta.variant == variant)
            .max_by_key(|p| p.meta.version)
        {
            let py = pack_python(Path::new(&p.path));
            if py.exists() {
                return py;
            }
        }
    }
    let embedded = crate::util::manual_python_slot(app_dir);
    if embedded.exists() {
        return embedded;
    }
    PathBuf::from("python")
}

/// `converter_python`, but a bare-PATH fallback becomes a LOUD, actionable error
/// instead of a doomed spawn ("系统找不到指定的文件" pointing at nothing). On dev
/// machines the venv always resolves first, so this only fires on end-user machines
/// with no runtime pack installed — exactly where the guidance is needed.
pub fn converter_python_checked(app_dir: &Path) -> Result<PathBuf> {
    let py = converter_python(app_dir);
    if py == Path::new("python") {
        return Err(err("RUNTIME_PACK_REQUIRED"));
    }
    Ok(py)
}

/// nv-cu130 and amd(torch-hip) both drive the `torch.cuda.*` namespace, so both map
/// to the "cuda" backend for device.py's shim; xpu and cpu map to themselves.
fn variant_backend(variant: &str) -> &'static str {
    match variant {
        "xpu" => "xpu",
        "cpu" => "cpu",
        _ => "cuda",
    }
}

/// The TRAINING-role interpreter AND its device backend. Unlike the converter role
/// (any variant runs its CPU-bound scripts), the pack VARIANT drives `device_backend`
/// for device.py's shim. `force_cpu` pins "cpu" regardless (the train-on-CPU-anyway
/// path). Priority: dev venv (unchanged dev experience — backend from the box's own
/// GPU) → best installed pack (GPU variants first, newest version) → manual slot →
/// bare python. Returns `(interpreter, device_backend)`.
pub fn training_interpreter(app_dir: &Path, force_cpu: bool) -> (PathBuf, String) {
    let default_gpu = || if force_cpu { "cpu" } else { "cuda" }.to_string();

    let venv = app_dir
        .join("training")
        .join(".venv")
        .join("Scripts")
        .join("python.exe");
    if venv.exists() {
        return (venv, default_gpu());
    }
    // Same GPU-first order as the converter role; here the variant also fixes the
    // backend, so an NVIDIA box holding only nv-cu130 trains on GPU (device=cuda).
    const TRAINING_PRIORITY: [&str; 4] = ["nv-cu130", "amd", "xpu", "cpu"];
    let packs = list_packs();
    for variant in TRAINING_PRIORITY {
        if let Some(p) = packs
            .iter()
            .filter(|p| p.meta.variant == variant)
            .max_by_key(|p| p.meta.version)
        {
            let py = pack_python(Path::new(&p.path));
            if py.exists() {
                let backend = if force_cpu {
                    "cpu".to_string()
                } else {
                    variant_backend(variant).to_string()
                };
                return (py, backend);
            }
        }
    }
    let embedded = crate::util::manual_python_slot(app_dir);
    if embedded.exists() {
        return (embedded, default_gpu());
    }
    (PathBuf::from("python"), default_gpu())
}

// ─── catalog ────────────────────────────────────────────────────────────────

/// A downloadable pack the app knows about. The catalog deliberately carries NO
/// hashes/part lists — those live in the published `<id>.manifest.json` next to the
/// archive, so pack rebuilds don't require an app update.
#[derive(Debug, Clone, Serialize)]
pub struct CatalogEntry {
    pub id: &'static str,
    pub variant: &'static str,
    pub label: &'static str,
    /// Rough sizes for the UI (真源 = manifest once fetched).
    pub download_bytes: u64,
    pub disk_bytes: u64,
    pub experimental: bool,
    /// Published manifest URLs (mirrors). Empty until the pack is uploaded —
    /// the dev override UTAI_PACK_BASE_URL (comma-separated base URLs) extends
    /// this at runtime for local end-to-end testing against `python -m http.server`.
    pub manifest_urls: &'static [&'static str],
}

pub const CATALOG: &[CatalogEntry] = &[CatalogEntry {
    id: "runtime-cpu-v1",
    variant: "cpu",
    label: "CPU runtime (model conversion base + CPU training)",
    // Real numbers from the published pack (S42): 236 MB download / 1.18 GB on disk.
    download_bytes: 236_000_000,
    disk_bytes: 1_180_000_000,
    experimental: false,
    // Published S42 (datasets/yasoukyoku/utai-runtimes). Order matters: official
    // first, hf-mirror second — the downloader walks them with resume carried
    // across, so CN users blocked from hf.co fail over automatically. Both
    // verified live: anonymous resolve ✓, Range→206 ✓, mirror manifest ✓.
    manifest_urls: &[
        "https://huggingface.co/datasets/yasoukyoku/utai-runtimes/resolve/main/runtime-cpu-v1.manifest.json",
        "https://hf-mirror.com/datasets/yasoukyoku/utai-runtimes/resolve/main/runtime-cpu-v1.manifest.json",
    ],
}, CatalogEntry {
    id: "runtime-nv-cu130-v1",
    variant: "nv-cu130",
    label: "NVIDIA runtime (cu130; RTX 20-50 training + model conversion)",
    // Real numbers from the Phase B build: 1.84 GB download / 3.59 GB on disk (single
    // part — under the 1.9 GiB split cap).
    download_bytes: 1_838_043_109,
    disk_bytes: 3_587_520_577,
    experimental: false,
    // Same HF dataset repo as cpu; goes live once uploaded (`hf upload
    // yasoukyoku/utai-runtimes <dist> . --repo-type dataset`). Until then the dev
    // override UTAI_PACK_BASE_URL serves it for local end-to-end tests.
    manifest_urls: &[
        "https://huggingface.co/datasets/yasoukyoku/utai-runtimes/resolve/main/runtime-nv-cu130-v1.manifest.json",
        "https://hf-mirror.com/datasets/yasoukyoku/utai-runtimes/resolve/main/runtime-nv-cu130-v1.manifest.json",
    ],
}, CatalogEntry {
    id: "runtime-amd-v1",
    variant: "amd",
    label: "AMD runtime (TheRock ROCm; RDNA3/4 training + model conversion, experimental)",
    // Real numbers from the S44 build: 1.172 GB download / 4.50 GB on disk (single
    // part, under the 1.9 GiB split cap). Validated end-to-end on the dev machine's
    // Radeon 780M (gfx1103): flat-PBS-layout torch-rocm import + full envtest 20/20
    // PASS (tiny_gan conv/convT converges identically to NVIDIA; fp16+GradScaler OK).
    download_bytes: 1_171_991_360,
    disk_bytes: 4_504_503_445,
    // EXPERIMENTAL tier (design §4.3): TheRock ROCm is a pinned nightly; MIOpen ships
    // no precompiled conv DB for gfx1103 (#6335) → first-encounter conv configs pay a
    // one-time kernel-compile cost, then cache. envtest is the release gate.
    experimental: true,
    manifest_urls: &[
        "https://huggingface.co/datasets/yasoukyoku/utai-runtimes/resolve/main/runtime-amd-v1.manifest.json",
        "https://hf-mirror.com/datasets/yasoukyoku/utai-runtimes/resolve/main/runtime-amd-v1.manifest.json",
    ],
}, CatalogEntry {
    id: "runtime-xpu-v1",
    variant: "xpu",
    label: "Intel runtime (XPU; Arc training + model conversion, experimental)",
    // Real numbers from the S45 build: 1.356 GB download / 5.27 GB on disk (single part,
    // under the 1.9 GiB split cap). Bulkier than cpu/nv because the Intel SYCL/oneMKL
    // runtime + triton-xpu (359 MB) ship as separate wheels. Built + verified on the dev
    // machine (RTX 3080 Ti, NO Intel GPU): flat-PBS torch-2.11.0+xpu import OK + envtest
    // --device cpu 20/20 PASS (tiny_gan 0.6350→0.4609 bitwise-identical to cpu/nv/amd).
    download_bytes: 1_355_876_805,
    disk_bytes: 5_273_762_795,
    // EXPERIMENTAL tier (design §4.3/§4.5). xpu runs fp32 (no bf16). Four of the five training
    // objects (RVC / SoVITS 4.0 / 4.1 / shallow-diffusion) use the Intel GPU via the device
    // shim; the vocoder finetune trains on CPU (Lightning 2.6.5 ships no XPU accelerator — it
    // warns loudly rather than silently fall back). torch 2.11.0+xpu is on the
    // download.pytorch.org/whl/xpu RELEASE channel (permanent, no HF wheel mirror). No numeric
    // xpu gate is possible without Intel silicon — correctness is by-construction + the
    // on-device envtest, validated by the first community reports (keep the flag until then).
    experimental: true,
    // Published S45 to datasets/yasoukyoku/utai-runtimes. Verified live: HF LFS oid ==
    // local sha256 (6e9610e0…, byte-identical 1.36 GB), manifest HTTP 200 on both sources,
    // tar Range→206 (resumable), served manifest id/variant/sha correct. Official first,
    // hf-mirror second (downloader fails over with resume carried).
    manifest_urls: &[
        "https://huggingface.co/datasets/yasoukyoku/utai-runtimes/resolve/main/runtime-xpu-v1.manifest.json",
        "https://hf-mirror.com/datasets/yasoukyoku/utai-runtimes/resolve/main/runtime-xpu-v1.manifest.json",
    ],
}];

/// Manifest URL candidates for a catalog entry: published URLs + the dev override.
pub fn manifest_url_candidates(entry: &CatalogEntry) -> Vec<String> {
    let mut urls: Vec<String> = entry.manifest_urls.iter().map(|s| s.to_string()).collect();
    if let Ok(bases) = std::env::var("UTAI_PACK_BASE_URL") {
        for base in bases.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            urls.push(format!("{}/{}.manifest.json", base.trim_end_matches('/'), entry.id));
        }
    }
    urls
}

/// Published distribution manifest (written by the pack builder next to the parts).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackManifest {
    #[serde(default)]
    pub schema: u32,
    pub id: String,
    pub variant: String,
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub disk_bytes: u64,
    pub parts: Vec<ManifestPart>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestPart {
    pub name: String,
    pub size: u64,
    pub sha256: String,
}

/// Fetch the manifest from the first reachable candidate. Returns (manifest,
/// base_url_of_the_winning_candidate) — part URLs resolve against that base first,
/// with every other candidate base as fallback mirror.
pub async fn fetch_manifest(
    client: &reqwest::Client,
    candidates: &[String],
) -> Result<(PackManifest, Vec<String>)> {
    if candidates.is_empty() {
        return Err(err("PACK_NO_DOWNLOAD_SOURCE"));
    }
    let mut last: Option<String> = None;
    for url in candidates {
        match client.get(url).send().await {
            Ok(resp) if resp.status().is_success() => match resp.text().await {
                Ok(text) => match serde_json::from_str::<PackManifest>(&text) {
                    Ok(man) => {
                        let mut bases: Vec<String> = Vec::new();
                        // Winning base first, then the rest (mirror order preserved).
                        for u in std::iter::once(url).chain(candidates.iter().filter(|u| *u != url)) {
                            if let Some(pos) = u.rfind('/') {
                                bases.push(u[..pos].to_string());
                            }
                        }
                        return Ok((man, bases));
                    }
                    Err(e) => last = Some(format!("MANIFEST_PARSE_FAILED: {e} ({url})")),
                },
                Err(e) => last = Some(format!("MANIFEST_READ_FAILED: {e} ({url})")),
            },
            Ok(resp) => last = Some(format!("MANIFEST_REQUEST_FAILED: HTTP {} ({url})", resp.status())),
            Err(e) => last = Some(format!("MANIFEST_REQUEST_FAILED: {e} ({url})")),
        }
    }
    Err(err(last.unwrap_or_else(|| "MANIFEST_FETCH_FAILED".into())))
}

// ─── install ────────────────────────────────────────────────────────────────

/// Single concurrent install/download (the UI drives one at a time; a second request
/// while busy is a hard error, not a queue). Holds the cooperative cancel flag of the
/// in-flight install so `cancel_runtime_install` can reach it.
static ACTIVE_INSTALL: parking_lot::Mutex<Option<Arc<AtomicBool>>> = parking_lot::Mutex::new(None);

pub struct InstallGuard;

impl InstallGuard {
    pub fn acquire() -> Result<(Self, Arc<AtomicBool>)> {
        let mut slot = ACTIVE_INSTALL.lock();
        if slot.is_some() {
            return Err(err("INSTALL_BUSY"));
        }
        let flag = Arc::new(AtomicBool::new(false));
        *slot = Some(Arc::clone(&flag));
        Ok((InstallGuard, flag))
    }
}

impl Drop for InstallGuard {
    fn drop(&mut self) {
        *ACTIVE_INSTALL.lock() = None;
    }
}

pub fn cancel_active_install() -> bool {
    match ACTIVE_INSTALL.lock().as_ref() {
        Some(flag) => {
            flag.store(true, Ordering::SeqCst);
            true
        }
        None => false,
    }
}

/// An install/download is currently in flight (drives the frontend's busy-state
/// rebuild when the settings panel is reopened mid-install — audit S42).
pub fn install_active() -> bool {
    ACTIVE_INSTALL.lock().is_some()
}

/// Envtest single-flight — lives HERE (not the command layer) so `delete_pack` can
/// refuse to pull a pack out from under a running self-test (audit S42: a partial
/// remove_dir_all deletes pack.json first, then fails on the locked python.exe,
/// leaving an invisible undeletable orphan).
static ENVTEST_BUSY: AtomicBool = AtomicBool::new(false);

pub struct EnvtestGuard;

impl EnvtestGuard {
    pub fn acquire() -> Result<Self> {
        if ENVTEST_BUSY.swap(true, Ordering::SeqCst) {
            return Err(err("ENVTEST_BUSY"));
        }
        Ok(EnvtestGuard)
    }
}

impl Drop for EnvtestGuard {
    fn drop(&mut self) {
        ENVTEST_BUSY.store(false, Ordering::SeqCst);
    }
}

pub fn envtest_active() -> bool {
    ENVTEST_BUSY.load(Ordering::SeqCst)
}

/// Validate everything a REMOTE manifest feeds into filesystem paths — one gate,
/// called right after fetch (covers the download flow; local installs derive part
/// paths from a real directory listing and extract_and_commit re-validates the id).
pub fn validate_manifest(man: &PackManifest) -> Result<()> {
    if !is_safe_component(&man.id) {
        return Err(err(format!("MANIFEST_BAD_ID: {:?}", man.id)));
    }
    if man.parts.is_empty() {
        return Err(err("MANIFEST_NO_PARTS"));
    }
    for p in &man.parts {
        if !is_safe_component(&p.name) {
            return Err(err(format!("MANIFEST_BAD_PART_NAME: {:?}", p.name)));
        }
        if p.sha256.len() != 64 || !p.sha256.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(err(format!("MANIFEST_BAD_SHA256: {}", p.name)));
        }
    }
    Ok(())
}

/// Ensured + ASCII-checked runtime root (install-time entry point).
pub fn install_root() -> Result<PathBuf> {
    let root = runtime_root().ok_or_else(|| err("RUNTIME_ROOT_UNINIT"))?;
    ensure_ascii_path(root)?;
    std::fs::create_dir_all(root)?;
    Ok(root.clone())
}

/// Verify each part against the manifest (blocking; wrap in spawn_blocking).
pub fn verify_parts(manifest: &PackManifest, dir: &Path) -> Result<()> {
    for part in &manifest.parts {
        let p = dir.join(&part.name);
        let meta = std::fs::metadata(&p)
            .map_err(|e| err(format!("PART_MISSING: {}: {e}", part.name)))?;
        if meta.len() != part.size {
            return Err(err(format!(
                "PART_SIZE_MISMATCH: {} (expected {}, got {})",
                part.name, part.size, meta.len()
            )));
        }
        let got = crate::download::sha256_file(&p)?;
        if !got.eq_ignore_ascii_case(&part.sha256) {
            return Err(err(format!("PART_SHA256_MISMATCH: {}", part.name)));
        }
    }
    Ok(())
}

/// `\\?\`-prefix an absolute path so tar extraction of deep site-packages trees
/// (torch easily exceeds 200 chars below the root) survives MAX_PATH on systems
/// without the LongPathsEnabled policy. canonicalize() returns the prefixed form.
fn long_path(p: &Path) -> PathBuf {
    std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf())
}

/// Rename with exponential-backoff retry on ACCESS_DENIED(5) / SHARING_VIOLATION(32).
/// Used for SINGLE-FILE renames (the pack.json commit marker) where a hold, if any,
/// is on that one fresh file and clears fast. NOT a fix for renaming big freshly-
/// extracted DIRECTORY trees — Defender's async inspection of thousands of new PE
/// files holds handles below such trees continuously for minutes, far beyond any
/// sane retry window (live-failed S42 even with 10 s of backoff; that's why the
/// install commit protocol is a marker FILE, not a directory rename — see
/// extract_and_commit).
/// `remove_dir_all` with one assisted retry: strip READONLY attributes throughout
/// (files planted by user tooling/backup restores may carry them; a read-only file
/// fails both the in-tree delete and tar's overwrite path — `remove_file` before
/// re-create — with os error 5), pause briefly for transient handle races (AV
/// inspection), then try once more. Call sites stay best-effort — see the torn-dir
/// comment in extract_and_commit.
fn remove_dir_all_robust(dir: &Path) -> std::io::Result<()> {
    fn clear_readonly(dir: &Path) {
        // The ROOT dir's own attribute matters too (review S68d): std deletes children
        // first and the root last, so a readonly root alone defeats both attempts.
        if let Ok(md) = std::fs::metadata(dir) {
            let mut perm = md.permissions();
            if perm.readonly() {
                perm.set_readonly(false);
                let _ = std::fs::set_permissions(dir, perm);
            }
        }
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            if let Ok(md) = e.metadata() {
                let mut perm = md.permissions();
                if perm.readonly() {
                    perm.set_readonly(false);
                    let _ = std::fs::set_permissions(&e.path(), perm);
                }
                if md.is_dir() {
                    clear_readonly(&e.path());
                }
            }
        }
    }
    match std::fs::remove_dir_all(dir) {
        Ok(()) => Ok(()),
        Err(_) => {
            clear_readonly(dir);
            std::thread::sleep(std::time::Duration::from_millis(300));
            std::fs::remove_dir_all(dir)
        }
    }
}

/// Best-effort recursive byte count (metadata only) — credits a torn install tree
/// that the extract is about to clear in the disk-space preflight.
fn dir_size(dir: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            if let Ok(md) = e.metadata() {
                if md.is_dir() {
                    total = total.saturating_add(dir_size(&e.path()));
                } else {
                    total = total.saturating_add(md.len());
                }
            }
        }
    }
    total
}

fn rename_with_retry(from: &Path, to: &Path, what: &str) -> Result<()> {
    let mut delay_ms = 250u64;
    let mut last: Option<std::io::Error> = None;
    for attempt in 0..8 {
        match std::fs::rename(from, to) {
            Ok(()) => {
                if attempt > 0 {
                    // `what` is a short English phase token (PACK_MOVE_OUT / INSTALL_COMMIT /
                    // PACK_ROLLBACK / INSTALL_RECOVERY) carried as the CODE's detail payload;
                    // the paths carry the context here.
                    tracing::info!("rename succeeded on retry {attempt}: {} -> {}", from.display(), to.display());
                }
                return Ok(());
            }
            Err(e) => {
                let retriable = matches!(e.raw_os_error(), Some(5) | Some(32));
                if !retriable {
                    return Err(err(format!("RENAME_FAILED: {what} ({} -> {}): {e}", from.display(), to.display())));
                }
                tracing::warn!("rename denied (attempt {attempt}) {} -> {}: {e} — retrying in {delay_ms}ms", from.display(), to.display());
                last = Some(e);
                std::thread::sleep(std::time::Duration::from_millis(delay_ms));
                delay_ms = (delay_ms * 2).min(2000);
            }
        }
    }
    Err(err(format!(
        "RENAME_RETRY_EXHAUSTED: {what} ({} -> {}): {}",
        from.display(),
        to.display(),
        last.map(|e| e.to_string()).unwrap_or_default()
    )))
}

/// Extract a (possibly multi-part) `.tar.zst` pack archive DIRECTLY into its final
/// directory and commit with a single-FILE marker write. Blocking — call in
/// spawn_blocking. `progress(entries_done)` ticks as tar entries land.
///
/// WHY NOT staging + directory rename (the first design, replaced S42 after a live
/// failure): renaming a directory on Windows fails with ACCESS_DENIED while ANY
/// process holds ANY handle anywhere below it — and right after extracting ~10k
/// brand-new files (hundreds of them PE binaries) Defender's async inspection queue
/// holds handles somewhere in the tree essentially CONTINUOUSLY, for minutes; a
/// retry window can't outlast it (10 s of backoff still failed on the dev box while
/// the identical code passed into %TEMP%). File CREATES are never blocked that way —
/// the extraction itself writing 10k files is the proof — so the commit point is
/// the exact thing discovery already keys on: **pack.json presence**, written LAST
/// via a same-dir tmp+rename of one fresh file. A torn install is a marker-less
/// directory: invisible to list_packs, reclaimed by sweep_staging.
///
/// Requires pack.json to be the FIRST tar entry (build_pack.py writes entries
/// sorted: "pack.json" < "python/") — the id inside decides the target directory
/// before anything touches disk.
pub fn extract_and_commit(
    parts: &[PathBuf],
    cancel: &AtomicBool,
    mut progress: impl FnMut(u64),
) -> Result<PackMeta> {
    use std::io::Read;

    let root = install_root()?;
    let reader = crate::download::MultiFileReader::new(parts.to_vec());
    let decoder = zstd::stream::read::Decoder::new(reader)
        .map_err(|e| err(format!("ZSTD_INIT_FAILED: {e}")))?;
    let mut archive = tar::Archive::new(decoder);
    // S68d ROOT-CAUSE FIX (the E:\ EXTRACT_FAILED report): the pack builder's staging
    // trees carry mtime=0, so every archived file says "1970" — and restoring that via
    // SetFileTime is REJECTED by FAT32/exFAT volumes (timestamp epoch 1980) with
    // os error 87, killing the very FIRST file of every extract on such drives. NTFS
    // (epoch 1601) accepts it, which is why dev machines never saw this. The archived
    // mtimes are worthless anyway — don't write them at all; extracted files carry the
    // extraction time on every filesystem. (build_pack.py now also clamps mtimes for
    // future packs, but THIS line is what rescues the four already-published ones.)
    archive.set_preserve_mtime(false);
    let mut entries = archive
        .entries()
        .map_err(|e| err(format!("TAR_READ_FAILED: {}", error_chain(&e))))?;

    // ── entry 0: pack.json → memory (determines the target dir) ──
    let mut first = entries
        .next()
        .ok_or_else(|| err("PACK_EMPTY"))?
        .map_err(|e| err(format!("TAR_ENTRY_CORRUPT: {}", error_chain(&e))))?;
    let first_path = first
        .path()
        .map_err(|e| err(format!("TAR_ENTRY_BAD_PATH: {}", error_chain(&e))))?
        .into_owned();
    if first_path != Path::new("pack.json") {
        return Err(err(format!(
            "PACK_FORMAT_INVALID: first entry is {:?}, expected pack.json (rebuild with the latest build_pack.py)",
            first_path
        )));
    }
    let mut meta_text = String::new();
    first
        .read_to_string(&mut meta_text)
        .map_err(|e| err(format!("PACK_JSON_READ_FAILED: {e}")))?;
    let meta: PackMeta = serde_json::from_str(&meta_text)
        .map_err(|e| err(format!("PACK_JSON_PARSE_FAILED: {e}")))?;
    // The id becomes a directory under the runtimes root — an id like "..\\evil"
    // or "包名" would escape the root / break the ASCII invariant.
    if !is_safe_component(&meta.id) {
        return Err(err(format!("PACK_JSON_BAD_ID: {:?}", meta.id)));
    }

    let final_dir = root.join(&meta.id);
    let marker = final_dir.join("pack.json");

    // S68d disk preflight — BEFORE anything touches an existing install, so a refusal
    // leaves the working pack in place. Needed = the extracted tree size (pack.json
    // disk_bytes, real numbers from the builders). A TORN remnant (dir without marker)
    // is about to be cleared, so its bytes count as available again; a reinstall's old
    // tree is only moved ASIDE and stays occupied until commit, so it earns no credit.
    // disk_bytes==0 (older builder, serde default) or a failed probe skips the check —
    // fail open: a residual ENOSPC still surfaces with its os error via error_chain.
    if meta.disk_bytes > 0 {
        if let Some(free) = crate::util::free_bytes_at(&root) {
            let torn_credit =
                if !marker.exists() && final_dir.exists() { dir_size(&final_dir) } else { 0 };
            let avail = free.saturating_add(torn_credit);
            if avail < meta.disk_bytes {
                return Err(err(format!(
                    "INSTALL_DISK_FULL: {} MB needed, {} MB free at {}",
                    meta.disk_bytes / 1_000_000,
                    avail / 1_000_000,
                    root.display()
                )));
            }
        }
    }

    let mut old_backup: Option<PathBuf> = None;
    let mut preclean_err: Option<String> = None;
    if marker.exists() {
        // Reinstall over an INSTALLED same-id pack: move the old tree ASIDE, never
        // destroy it up front — a failed install must roll the working pack back
        // (audit S42-r2). This dir rename targets a COLD tree (no fresh-file scan
        // storm — that only ever hit the just-extracted side), so retry suffices.
        let staging_parent = root.join(".staging");
        std::fs::create_dir_all(&staging_parent)?;
        let moved = staging_parent.join(format!(
            ".old-{}-{}",
            meta.id,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0)
        ));
        rename_with_retry(&final_dir, &moved, "PACK_MOVE_OUT")?;
        old_backup = Some(moved);
    } else if final_dir.exists() {
        // Torn earlier attempt. Prefer a clean slate; extracting over identical
        // content is the fallback when something still holds a subdir. Failing LOUD
        // here would re-create the S42 pain (Defender's scan queue pins fresh-file
        // handles for minutes right after a failed extract, while file CREATES keep
        // working) — so the fallback stays, but the failure is REMEMBERED: if the
        // extract-over then fails too, both causes surface in one message instead of
        // an unexplainable unpack error (S68d).
        if let Err(e) = remove_dir_all_robust(&final_dir) {
            preclean_err = Some(e.to_string());
            tracing::warn!("torn install dir not fully cleared ({e}) — extracting over it");
        }
    }
    std::fs::create_dir_all(&final_dir)?;

    let result = (|| -> Result<PackMeta> {
        let dest = long_path(&final_dir);
        let mut count: u64 = 0;
        for entry in entries {
            if cancel.load(Ordering::SeqCst) {
                return Err(err("INSTALL_CANCELLED"));
            }
            let mut entry = entry.map_err(|e| err(format!("TAR_ENTRY_CORRUPT: {}", error_chain(&e))))?;
            // unpack_in refuses paths escaping dest (tar 路径穿越防护).
            entry.unpack_in(&dest).map_err(|e| {
                let mut msg = format!("EXTRACT_FAILED: {}", error_chain(&e));
                if let Some(pc) = &preclean_err {
                    // Both causes in one line: an extract-over failure is usually a
                    // symptom of whatever blocked the pre-clean (locked/readonly
                    // remnant) — alone, the unpack error is a riddle.
                    msg.push_str(&format!(" [pre-clean incomplete: {pc}]"));
                }
                err(msg)
            })?;
            count += 1;
            if count % 100 == 0 {
                progress(count);
            }
        }
        progress(count);

        if !pack_python(&final_dir).exists() {
            return Err(err("PACK_NO_PYTHON: python/python.exe missing from the archive"));
        }

        // ── the commit: one fresh file, same-dir rename ──
        let tmp = final_dir.join("pack.json.tmp");
        std::fs::write(&tmp, meta_text.as_bytes())
            .map_err(|e| err(format!("INSTALL_COMMIT_WRITE_FAILED: {}: {e}", tmp.display())))?;
        rename_with_retry(&tmp, &marker, "INSTALL_COMMIT")?;
        Ok(meta)
    })();

    match &result {
        Ok(_) => {
            // New pack committed — the old backup (reinstall case) is now redundant.
            if let Some(old) = &old_backup {
                let _ = std::fs::remove_dir_all(old);
            }
        }
        Err(_) => {
            // Marker never landed — the dir is invisible to discovery either way.
            // Best-effort reclaim; sweep_staging finishes the job on next startup.
            let _ = std::fs::remove_file(final_dir.join("pack.json.tmp"));
            if let Err(e) = std::fs::remove_dir_all(&final_dir) {
                tracing::warn!("failed install not fully reclaimed ({e}) — sweep will finish");
            }
            // Reinstall case: roll the previous pack back IN-SESSION — waiting for
            // the next startup's sweep would leave the user packless until restart.
            if let Some(old) = &old_backup {
                match rename_with_retry(old, &final_dir, "PACK_ROLLBACK") {
                    Ok(()) => tracing::warn!("reinstall failed — previous pack rolled back"),
                    Err(e) => tracing::error!("old-pack rollback failed ({e}) — startup sweep will restore it"),
                }
            }
        }
    }
    result
}

/// Resolve the part files for a LOCAL archive pick: a single `.tar.zst`, or the
/// first `.partNN`/`.NNN` volume (siblings collected by numeric suffix). When a
/// `<id>.manifest.json` sits next to the pick it is loaded for verification;
/// otherwise (dev convenience) verification is skipped with a warning.
pub fn resolve_local_parts(picked: &Path) -> Result<(Vec<PathBuf>, Option<PackManifest>)> {
    let dir = picked
        .parent()
        .ok_or_else(|| err("LOCAL_FILE_BAD_DIR"))?;
    let fname = picked
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| err("LOCAL_FILE_BAD_NAME"))?;

    // Split-volume naming: <stem>.tar.zst.partNN (builder convention).
    let (stem, parts) = if let Some(idx) = fname.find(".tar.zst.part") {
        let stem = &fname[..idx + ".tar.zst".len()]; // "<id>.tar.zst"
        let prefix = format!("{stem}.part");
        let mut vols: Vec<(u32, PathBuf)> = Vec::new();
        for e in std::fs::read_dir(dir)?.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            if let Some(numpart) = name.strip_prefix(&prefix) {
                if let Ok(n) = numpart.parse::<u32>() {
                    vols.push((n, e.path()));
                }
            }
        }
        if vols.is_empty() {
            return Err(err("LOCAL_PARTS_NOT_FOUND"));
        }
        vols.sort_by_key(|(n, _)| *n);
        // Volumes must be contiguous from 1 — a missing middle volume would
        // otherwise silently produce a corrupt stream.
        for (i, (n, _)) in vols.iter().enumerate() {
            if *n != (i as u32) + 1 {
                return Err(err(format!("LOCAL_PARTS_GAP: missing {prefix}{:02}", i + 1)));
            }
        }
        (stem.to_string(), vols.into_iter().map(|(_, p)| p).collect())
    } else if fname.ends_with(".tar.zst") {
        (fname.to_string(), vec![picked.to_path_buf()])
    } else {
        return Err(err("LOCAL_FILE_BAD_TYPE"));
    };

    let manifest_name = format!("{}.manifest.json", stem.trim_end_matches(".tar.zst"));
    let manifest = std::fs::read_to_string(dir.join(&manifest_name))
        .ok()
        .and_then(|t| serde_json::from_str::<PackManifest>(&t).ok());
    if manifest.is_none() {
        tracing::warn!("no {manifest_name} next to the archive — installing WITHOUT hash verification");
    }
    Ok((parts, manifest))
}

pub fn delete_pack(id: &str) -> Result<()> {
    // Interlocks first: deleting under a live install/self-test tears files out
    // from under a running python.exe.
    if install_active() {
        return Err(err("DELETE_WHILE_INSTALLING"));
    }
    if envtest_active() {
        return Err(err("DELETE_WHILE_ENVTEST"));
    }
    let root = runtime_root().ok_or_else(|| err("RUNTIME_ROOT_UNINIT"))?;
    if !is_safe_component(id) {
        return Err(err(format!("PACK_BAD_ID: {id:?}")));
    }
    let dir = root.join(id);
    let marker = dir.join("pack.json");
    if !marker.exists() {
        return Err(err(format!("PACK_NOT_FOUND: {id}")));
    }
    // Marker-FIRST delete — the mirror image of the install commit: removing ONE
    // closed file is never blocked by scanner handles elsewhere in the tree, and
    // once the marker is gone the pack is de-listed everywhere (discovery keys on
    // it). The tree itself is best-effort now + startup sweep later — the old
    // "rename the whole dir out" approach hit the same ACCESS_DENIED wall as the
    // install commit (see extract_and_commit).
    let mut last: Option<std::io::Error> = None;
    for attempt in 0..5u64 {
        match std::fs::remove_file(&marker) {
            Ok(()) => {
                last = None;
                break;
            }
            Err(e) => {
                last = Some(e);
                std::thread::sleep(std::time::Duration::from_millis(150 * (attempt + 1)));
            }
        }
    }
    if let Some(e) = last {
        return Err(err(format!(
            "PACK_DELETE_FAILED: {}: {e}",
            marker.display()
        )));
    }
    if let Err(e) = std::fs::remove_dir_all(&dir) {
        // Invisible already; the startup sweep reclaims marker-less dirs.
        tracing::warn!("pack tree removal deferred ({e}) — sweep will reclaim {}", dir.display());
    }
    Ok(())
}

/// Startup reclamation of `.staging` (audit S42 — nothing else ever GC'd it):
///   - `.old-<id>-<ts>`: the previous pack moved out during a reinstall. If the
///     final dir went MISSING (crash between the two commit renames), RESTORE it —
///     the user's working pack must not silently vanish. Otherwise delete.
///   - `.del-*`: deferred deletes — remove.
///   - uuid extraction dirs: torn installs — remove.
///   - `dl-<id>`: KEEP (resumable downloaded parts).
pub fn sweep_staging() {
    // Hold the install slot for the whole sweep: without it an install starting
    // mid-sweep can have its (deliberately marker-less) target dir reclaimed out
    // from under the extracting tar, or a mid-reinstall .old- backup "recovered"
    // into the commit target (audit S42-r2). If somehow busy, just skip.
    let Ok((_guard, _flag)) = InstallGuard::acquire() else {
        tracing::info!("pyenv sweep skipped — an install is in flight");
        return;
    };
    let Some(root) = runtime_root() else { return };
    let staging = root.join(".staging");
    if let Ok(entries) = std::fs::read_dir(&staging) {
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()).map(str::to_string) else {
                continue;
            };
            if name.starts_with("dl-") {
                continue; // resumable download parts
            }
            if let Some(rest) = name.strip_prefix(".old-") {
                // ".old-<id>-<millis>" — id may itself contain '-'.
                if let Some((id, _ts)) = rest.rsplit_once('-') {
                    let final_dir = root.join(id);
                    if !final_dir.exists() && path.join("pack.json").exists() {
                        match rename_with_retry(&path, &final_dir, "INSTALL_RECOVERY") {
                            Ok(()) => {
                                tracing::warn!("recovered pack {id} from interrupted reinstall");
                            }
                            Err(e) => {
                                // NEVER fall through to deletion here: this backup can
                                // be the ONLY remaining copy of the user's pack (the
                                // fall-through was audit S42-r2's HIGH finding). Leave
                                // it for a later sweep.
                                tracing::warn!("recovery of {id} failed ({e}) — keeping backup for next sweep");
                            }
                        }
                        continue;
                    }
                    // final dir exists (or backup is incomplete) → stale backup: delete below.
                }
            }
            match std::fs::remove_dir_all(&path) {
                Ok(()) => tracing::info!("pyenv sweep: removed stale {}", path.display()),
                Err(e) => tracing::warn!("pyenv sweep: could not remove {} ({e})", path.display()),
            }
        }
    }

    // Marker-less dirs under the ROOT = torn installs / deferred deletes of the
    // marker-file commit protocol — reclaim them too (safe: we HOLD the install
    // slot, so no live extraction target can be among them).
    let Ok(entries) = std::fs::read_dir(root) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else { continue };
        if name.starts_with('.') || !path.is_dir() {
            continue;
        }
        if !path.join("pack.json").exists() {
            match std::fs::remove_dir_all(&path) {
                Ok(()) => tracing::info!("pyenv sweep: reclaimed marker-less {}", path.display()),
                Err(e) => tracing::warn!("pyenv sweep: could not reclaim {} ({e})", path.display()),
            }
        }
    }
}
