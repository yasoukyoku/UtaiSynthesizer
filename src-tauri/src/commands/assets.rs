// S64 — Model-asset pack downloader: the aux inference models + training base models that used to be
// hand-placed on the dev machine (pending_cleanups "aux 模型分发 / 训练底模分发"). Three packs, each a
// flat list of files mirrored on HF at models/<rel> (datasets/yasoukyoku/utai-runtimes) with the SAME
// relative path they occupy under <data>/models/<rel> locally — one catalog, no per-file mapping.
//
// Download rides the shared S42 engine (download.rs: .part resume + mirror rotation + stall watchdog
// + sha256-before-rename), one file at a time. Sources per file: the user's custom HF mirror base
// (Settings 下载源, applyMirror semantics — host prefix replacement) first when set, then
// huggingface.co, then hf-mirror.com; sha256 makes any source content-safe. Files already present are
// skipped (existence check — the rename-commit protocol means a present dest is a complete download,
// and a user's own hand-placed variant of an asset must not be clobbered by a hash mismatch).
//
// Stable CODEs: ASSET_DL_BUSY / ASSET_DL_FAILED (+ the engine's DOWNLOAD_CANCELLED cancel sentinel).

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use tauri::{Emitter, State};

use crate::AppState;

const HF_HOST: &str = "https://huggingface.co";
const HF_MIRROR_HOST: &str = "https://hf-mirror.com";
const REPO_PATH: &str = "datasets/yasoukyoku/utai-runtimes/resolve/main/models";

struct AssetFile {
    /// Path under `<data>/models/` AND under `models/` in the HF repo (forward slashes).
    rel: &'static str,
    size: u64,
    sha256: &'static str,
}

struct AssetPack {
    id: &'static str,
    files: &'static [AssetFile],
}

// ─── catalog (sizes + sha256 computed from the dev-machine originals at upload time, S64) ───

const AUX_FILES: &[AssetFile] = &[
    AssetFile { rel: "auxiliary/contentvec_256l9.onnx", size: 293_312_060, sha256: "d1ce3a3ce3d39c3e12f7c618ebeee631e089fe6606eb2cb414f2b9af74b95314" },
    AssetFile { rel: "auxiliary/contentvec_768l12.onnx", size: 377_602_470, sha256: "3a7db9b31ec297378bcfa8ec78c00968bee84ba0491efdca52aa7044150d92c5" },
    AssetFile { rel: "auxiliary/rmvpe_e2e.onnx", size: 361_704_910, sha256: "2c2a08416dcd9790c8837e9fabe9fcab54b3657c76029d3cd709d17bdbaf6200" },
    AssetFile { rel: "auxiliary/rmvpe_mel_filters.npy", size: 262_784, sha256: "cb277cc6da0f8d217cbfedc3513ad234de9e69919764259f4e7699f75518e1eb" },
    AssetFile { rel: "auxiliary/nsf_hifigan.onnx", size: 56_829_864, sha256: "5597601d628e54fea8382e64a781f67f0379a6d37c8fccf22c72004dac1d7a20" },
    AssetFile { rel: "auxiliary/nsf_hifigan.json", size: 207, sha256: "0f8cdcb28624e4e1a30acb0abecab8e2e98d7a79c49c4c602164b2b6ce6007a3" },
    AssetFile { rel: "auxiliary/nsf_hifigan_mel.npy", size: 524_928, sha256: "a5b709d52d0ad9182fddaf3c9136f89620d11c92beafad433e9c813f42da0e6c" },
    AssetFile { rel: "auxiliary/score2cv_256.onnx", size: 180_361_506, sha256: "8464168c6400e389ad448b676436e4664ea7732177af000b86830f08f981a31e" },
    AssetFile { rel: "auxiliary/score2cv_256.json", size: 407, sha256: "c23afe19801d9bc544b88401ccd54e740e8edcd993a146b59d8f61ef5014c2f3" },
    AssetFile { rel: "auxiliary/score2cv_768.onnx", size: 181_416_226, sha256: "35d081de21595e0f95dd36b67f22763ffc5b24dfe34520c8e3f03a73c57756bc" },
    AssetFile { rel: "auxiliary/score2cv_768.json", size: 413, sha256: "cc4490132ade28d1bb88a8721deefc52d136d26baad77cabf1f854f0b8698dd3" },
    // NSF-HiFiGAN weights are CC BY-NC-SA — the attribution NOTICE travels with the onnx derivative.
    AssetFile { rel: "auxiliary/NOTICE.txt", size: 3_104, sha256: "a393b44505ccb6d1da63c2c73ccbbdaeb9b877a5227bf41b1b1e4a8429a51dd6" },
    AssetFile { rel: "auxiliary/NOTICE.zh-CN.txt", size: 3_046, sha256: "ea5511e12932a33481c212c1c19f6225af90ff6dc6f3e34a41050f028823ebb5" },
];

const RVC_TRAIN_FILES: &[AssetFile] = &[
    AssetFile { rel: "training/rvc/pretrained/f0G32k.pth", size: 72_795_627, sha256: "285f524bf48bb692c76ad7bd0bc654c12bd9e5edeb784dddf7f61a789a608574" },
    AssetFile { rel: "training/rvc/pretrained/f0G40k.pth", size: 72_909_665, sha256: "9115654aeef1995f7dd3c6fc4140bebbef0ca9760bed798105a2380a34299831" },
    AssetFile { rel: "training/rvc/pretrained/f0G48k.pth", size: 73_008_619, sha256: "78bc9cab27e34bcfc194f93029374d871d8b3e663ddedea32a9709e894cc8fe8" },
    AssetFile { rel: "training/rvc/pretrained/f0D32k.pth", size: 109_978_943, sha256: "294db3087236e2c75260d6179056791c9231245daf5d0485545d9e54c4057c77" },
    AssetFile { rel: "training/rvc/pretrained/f0D40k.pth", size: 109_978_943, sha256: "7d4f5a441594b470d67579958b2fd4c6b992852ded28ff9e72eda67abcebe423" },
    AssetFile { rel: "training/rvc/pretrained/f0D48k.pth", size: 109_978_943, sha256: "1b84c8bf347ad1e539c842e8f2a4c36ecd9e7fb23c16041189e4877e9b07925c" },
    AssetFile { rel: "training/rvc/pretrained_v2/f0G32k.pth", size: 73_950_049, sha256: "2332611297b8d88c7436de8f17ef5f07a2119353e962cd93cda5806d59a1133d" },
    AssetFile { rel: "training/rvc/pretrained_v2/f0G40k.pth", size: 73_106_273, sha256: "3b2c44035e782c4b14ddc0bede9e2f4a724d025cd073f736d4f43708453adfcb" },
    AssetFile { rel: "training/rvc/pretrained_v2/f0G48k.pth", size: 75_465_569, sha256: "b5d51f589cc3632d4eae36a315b4179397695042edc01d15312e1bddc2b764a4" },
    AssetFile { rel: "training/rvc/pretrained_v2/f0D32k.pth", size: 142_875_703, sha256: "bd7134e7793674c85474d5145d2d982e3c5d8124fc7bb6c20f710ed65808fa8a" },
    AssetFile { rel: "training/rvc/pretrained_v2/f0D40k.pth", size: 142_875_703, sha256: "6b6ab091e70801b28e3f41f335f2fc5f3f35c75b39ae2628d419644ec2b0fa09" },
    AssetFile { rel: "training/rvc/pretrained_v2/f0D48k.pth", size: 142_875_703, sha256: "2269b73c7a4cf34da09aea99274dabf99b2ddb8a42cbfb065fb3c0aa9a2fc748" },
    // RVC training f0 extractor (the aux/ RVC-blood rmvpe.pt — a DIFFERENT architecture from the
    // sovits-blood training/sovits/rmvpe.pt below; never interchangeable, see pending_cleanups S37).
    AssetFile { rel: "auxiliary/rmvpe.pt", size: 181_184_272, sha256: "6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193" },
];

const SOVITS_TRAIN_FILES: &[AssetFile] = &[
    AssetFile { rel: "training/sovits/rmvpe.pt", size: 368_492_925, sha256: "19dc1809cf4cdb0a18db93441816bc327e14e5644b72eeaae5220560c6736fe2" },
    AssetFile { rel: "training/sovits/vec768/G_0.pth", size: 209_268_661, sha256: "9d3e408786013590bb3574ade2831ab62c989d303834742fe73ca8d5552d2f03" },
    AssetFile { rel: "training/sovits/vec768/D_0.pth", size: 187_027_770, sha256: "60b6936d55d2cfaa717033eafe9d98dbe44d322e6adaf7be7c1c5a835ebb7177" },
    AssetFile { rel: "training/sovits/vec256/G_0.pth", size: 180_628_517, sha256: "20a327c54e5731bed377bd38404bc32ab98e66a1b2777b0af4cc034d4d6914b0" },
    AssetFile { rel: "training/sovits/vec256/D_0.pth", size: 187_018_591, sha256: "635be5c3409aaf3eec4135a1f5a771595683f3a6461ffc5bdea43441e50269a9" },
    AssetFile { rel: "training/sovits/nsf_hifigan/model", size: 56_825_430, sha256: "2c576b63b7ed952161b70fad34e0562ace502ce689195520d8a2a6c051de29d6" },
    AssetFile { rel: "training/sovits/nsf_hifigan/config.json", size: 845, sha256: "9707614b59c299766a91ea25b5ec62cfd813a45a902766c454f75b6868118684" },
    AssetFile { rel: "training/sovits/nsf_hifigan/NOTICE.txt", size: 3_104, sha256: "a393b44505ccb6d1da63c2c73ccbbdaeb9b877a5227bf41b1b1e4a8429a51dd6" },
    AssetFile { rel: "training/sovits/nsf_hifigan/NOTICE.zh-CN.txt", size: 3_046, sha256: "ea5511e12932a33481c212c1c19f6225af90ff6dc6f3e34a41050f028823ebb5" },
    AssetFile { rel: "training/sovits/diffusion/vec768/model_0.pt", size: 220_890_164, sha256: "d8b7cc5a94a57f7e5772c3f5f48fd458684235b8d98f38e0feff134fafad93dd" },
];

// S68: SoVITS 4.0-v2 (VISinger2) base pair — its own pack (~1GB; the ckpts are
// not interchangeable with the 4.x vec256/vec768 bases, and 4.x-only users
// shouldn't pay the download). rmvpe.pt rides the training-sovits pack (same
// yxlllc lineage file, resolve_training_assets points v2 at it).
const SOVITS_V2_TRAIN_FILES: &[AssetFile] = &[
    AssetFile { rel: "training/sovits_v2/G_0.pth", size: 424_574_162, sha256: "8bb021019d65aef34755ac4006d27d9eda4244faabd63d546fa069902e668f27" },
    AssetFile { rel: "training/sovits_v2/D_0.pth", size: 561_070_439, sha256: "028b7db89f184327cfa1c8ee701887e1cb513b9eaa21b4b573bbbd6f10ad38de" },
];

const PACKS: &[AssetPack] = &[
    AssetPack { id: "aux-inference", files: AUX_FILES },
    AssetPack { id: "training-rvc", files: RVC_TRAIN_FILES },
    AssetPack { id: "training-sovits", files: SOVITS_TRAIN_FILES },
    AssetPack { id: "training-sovits-v2", files: SOVITS_V2_TRAIN_FILES },
];

/// Which asset pack distributes the file at `rel` (forward-slash path under `<data>/models/`),
/// None = not pack-distributed (e.g. the CC BY-NC-SA vocoder base = user self-download).
/// S66: the "missing base model → one-click download" dialog maps files to packs through
/// THIS table so the mapping can never drift from what download_asset_pack actually fetches.
pub(crate) fn pack_for_rel(rel: &str) -> Option<&'static str> {
    PACKS
        .iter()
        .find(|p| p.files.iter().any(|f| f.rel == rel))
        .map(|p| p.id)
}

// ─── state ───────────────────────────────────────────────────────────────────

/// Single-flight for the whole subsystem (one pack downloads at a time — the UI queues intent, and
/// two concurrent multi-GB streams help nobody). The GAME_DL_ACTIVE pattern.
static ASSET_DL_ACTIVE: AtomicBool = AtomicBool::new(false);
/// The pack id currently downloading — asset_pack_status stamps it per-pack so a remounted
/// Settings panel reattaches its cancel/progress UI immediately, not on the next chunk event.
static ASSET_DL_PACK: Mutex<Option<String>> = Mutex::new(None);
/// The in-flight download's cancel flag (cancel_asset_pack_download flips it; cooperative,
/// .part survives for resume).
static ASSET_DL_CANCEL: Mutex<Option<Arc<AtomicBool>>> = Mutex::new(None);

struct DlGuard;
impl Drop for DlGuard {
    fn drop(&mut self) {
        *ASSET_DL_CANCEL.lock() = None;
        *ASSET_DL_PACK.lock() = None;
        ASSET_DL_ACTIVE.store(false, Ordering::SeqCst);
    }
}

fn dest_path(models_dir: &std::path::Path, rel: &str) -> PathBuf {
    rel.split('/').fold(models_dir.to_path_buf(), |p, seg| p.join(seg))
}

#[derive(serde::Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct AssetPackStatus {
    pub id: String,
    pub file_count: usize,
    pub missing: usize,
    pub total_bytes: u64,
    pub missing_bytes: u64,
    pub downloading: bool,
}

#[derive(serde::Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct AssetPackProgress {
    pack: String,
    stage: String, // "download" | "done" | "failed" | "cancelled" (terminal event ALWAYS fires —
    // a remounted panel must never wedge on a phantom in-flight state, audit S64)
    file: String,
    file_index: usize,
    file_count: usize,
    downloaded: u64,
    total: u64,
    /// Stable CODE detail for stage "failed" (frontend maps via backendError).
    error: Option<String>,
}

#[tauri::command]
pub fn asset_pack_status(state: State<'_, Arc<AppState>>) -> Vec<AssetPackStatus> {
    let models_dir = state.models.models_dir();
    let active = ASSET_DL_PACK.lock().clone();
    PACKS
        .iter()
        .map(|p| {
            let missing: Vec<&AssetFile> =
                p.files.iter().filter(|f| !dest_path(&models_dir, f.rel).exists()).collect();
            AssetPackStatus {
                id: p.id.to_string(),
                file_count: p.files.len(),
                missing: missing.len(),
                total_bytes: p.files.iter().map(|f| f.size).sum(),
                missing_bytes: missing.iter().map(|f| f.size).sum(),
                downloading: active.as_deref() == Some(p.id),
            }
        })
        .collect()
}

/// Download every missing file of one pack, sequentially, through the shared engine.
/// `hf_base` = the user's custom HF host replacement (Settings 下载源; applyMirror semantics —
/// replaces the `https://huggingface.co` prefix). Tried first when set; huggingface.co and
/// hf-mirror.com always follow, sha256 verifying whatever answers.
#[tauri::command]
pub async fn download_asset_pack(
    state: State<'_, Arc<AppState>>,
    app: tauri::AppHandle,
    id: String,
    hf_base: Option<String>,
) -> Result<(), String> {
    if ASSET_DL_ACTIVE.swap(true, Ordering::SeqCst) {
        return Err("ASSET_DL_BUSY".to_string());
    }
    let _guard = DlGuard;
    let _task = state.begin_task("asset-download"); // close-flow in-progress listing

    let pack = PACKS
        .iter()
        .find(|p| p.id == id)
        .ok_or_else(|| format!("ASSET_DL_FAILED: unknown pack {id}"))?;
    *ASSET_DL_PACK.lock() = Some(pack.id.to_string());
    let models_dir = state.models.models_dir();
    let client = crate::download::client().map_err(|e| format!("ASSET_DL_FAILED: {e}"))?;
    let cancel = Arc::new(AtomicBool::new(false));
    *ASSET_DL_CANCEL.lock() = Some(cancel.clone());

    // Host-replacement base from Settings 下载源 ("https://hf-mirror.com" for the preset, or the
    // custom URL). The fixed rotation below dedupes against it so the chosen base is tried first
    // exactly once.
    let custom_base = hf_base
        .as_deref()
        .map(|b| b.trim().trim_end_matches('/').to_string())
        .filter(|b| (b.starts_with("https://") || b.starts_with("http://")) && b != HF_HOST);

    let total: u64 = pack.files.iter().map(|f| f.size).sum();
    let file_count = pack.files.len();
    let mut done_before: u64 = 0;

    let result: Result<(), String> = async {
        // S68d disk preflight: refuse before the first byte, with real numbers — a
        // 3.6 GB pack dying on ENOSPC mid-way used to surface as a bare engine error.
        // Missing files only; in-flight .part bytes are credited (resume). Runs INSIDE
        // the result funnel so the terminal failed event still fires. Fail open when
        // the probe fails.
        {
            let mut needed: u64 = 0;
            for f in pack.files.iter() {
                let dest = dest_path(&models_dir, f.rel);
                if dest.exists() {
                    continue;
                }
                let mut part = dest.into_os_string();
                part.push(".part");
                let inflight = std::fs::metadata(std::path::PathBuf::from(part))
                    .map(|m| m.len().min(f.size))
                    .unwrap_or(0);
                needed = needed.saturating_add(f.size - inflight);
            }
            if needed > 0 {
                if let Some(free) = crate::util::free_bytes_at(&models_dir) {
                    if free < needed {
                        return Err(format!(
                            "INSTALL_DISK_FULL: {} MB needed, {} MB free at {}",
                            needed / 1_000_000,
                            free / 1_000_000,
                            models_dir.display()
                        ));
                    }
                }
            }
        }
        for (i, f) in pack.files.iter().enumerate() {
            let dest = dest_path(&models_dir, f.rel);
            if dest.exists() {
                done_before += f.size;
                continue;
            }
            let mut urls = Vec::with_capacity(3);
            if let Some(b) = &custom_base {
                urls.push(format!("{b}/{REPO_PATH}/{}", f.rel));
            }
            for host in [HF_HOST, HF_MIRROR_HOST] {
                if custom_base.as_deref() != Some(host) {
                    urls.push(format!("{host}/{REPO_PATH}/{}", f.rel));
                }
            }

            let req = crate::download::DownloadRequest {
                urls,
                dest,
                sha256: Some(f.sha256.to_string()),
                expected_size: Some(f.size),
            };
            let app_emit = app.clone();
            let pack_id = pack.id.to_string();
            let rel = f.rel.to_string();
            // Throttle: 2MB advance or file completion (the pyenv pattern) — per-chunk emits on a
            // 3.6GB pack would flood the IPC + React setState path (audit S64).
            let mut last_emitted: u64 = 0;
            crate::download::download(&client, &req, &cancel, move |done, _| {
                let abs = done_before + done;
                let complete = done >= f.size;
                if abs.saturating_sub(last_emitted) < 2_000_000 && !complete && abs != 0 {
                    return;
                }
                last_emitted = abs;
                let _ = app_emit.emit(
                    "asset-pack-progress",
                    AssetPackProgress {
                        pack: pack_id.clone(),
                        stage: "download".into(),
                        file: rel.clone(),
                        file_index: i,
                        file_count,
                        downloaded: abs,
                        total,
                        error: None,
                    },
                );
            })
            .await
            .map_err(|e| {
                let msg = e.to_string();
                // Preserve the engine's cancel sentinel (frontend isCancelError swallows it silently).
                if msg.contains("CANCELLED") { msg } else { format!("ASSET_DL_FAILED: {msg}") }
            })?;
            done_before += f.size;
        }
        Ok(())
    }
    .await;

    // A terminal event ALWAYS fires (done/cancelled/failed) — a Settings panel remounted mid-run
    // has only these events + asset_pack_status to reattach with; a swallowed failure would leave
    // it wedged on a phantom "downloading" (audit S64).
    let (stage, error) = match &result {
        Ok(()) => ("done", None),
        Err(e) if e.contains("CANCELLED") => ("cancelled", None),
        Err(e) => ("failed", Some(e.clone())),
    };
    let _ = app.emit(
        "asset-pack-progress",
        AssetPackProgress {
            pack: pack.id.to_string(),
            stage: stage.into(),
            file: String::new(),
            file_index: file_count,
            file_count,
            downloaded: done_before,
            total,
            error,
        },
    );
    if result.is_ok() {
        tracing::info!("asset pack {} complete ({} files)", pack.id, file_count);
    }
    result
}

/// Cooperative cancel of the in-flight pack download (the .part stays for a later resume).
#[tauri::command]
pub fn cancel_asset_pack_download() {
    if let Some(c) = ASSET_DL_CANCEL.lock().as_ref() {
        c.store(true, Ordering::SeqCst);
    }
}
