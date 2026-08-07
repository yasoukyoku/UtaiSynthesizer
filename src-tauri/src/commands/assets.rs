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
    /// SPDX-ish license id when the pack's weights carry terms of their own (S75: the
    /// NSF-HiFiGAN finetune base is CC BY-NC-SA 4.0). `None` = nothing extra to surface.
    /// Data, not copy — the UI decides how to render it; nothing here is user-facing prose.
    license: Option<&'static str>,
    /// The ORIGINAL upstream release page. We mirror those weights, we do not own them:
    /// the attribution link must stay reachable from the UI, and it doubles as the offline
    /// escape hatch when neither HF host answers.
    upstream: Option<&'static str>,
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

// S73 自动音高调教(旋钮线 Phase A/runA3;SVC2SVS pitch/export_onnx.py 产)。
// ⚠ S100 更正:这里原本写着「自训无许可负担」——**错的,别照抄**。它不来自别人的 checkpoint
// (那一半对),但训练数据与 score2cv 是同一批(SVC2SVS pitch/dataprep/build_clip_map.py 读的就是
// MBS2H processed/splits/{train,val}_final.jsonl)⇒ **93.96% 是 NC 语料**(GTSinger / M4Singer
// CC BY-NC-SA 4.0 + 四个日语声库非商用),权重随之只能非商用分发。清单与出处 = 仓根 NOTICE.md。
// ★独立 pack,绝不并入 aux-inference:渲染预检(preflightVocalModels)按 aux-inference 的
// missing 计数硬挡 Play——往老 pack 加新文件会让所有升级用户的存量工程被缺模型对话框卡死
// (断网即死锁,S73 审查 HIGH)。可选功能的模型 = 自己的 pack,用到时才提示下载。
// ★换版本必须换文件名(下载按存在性跳过)——下一版叫 autotune_a2.onnx。
// ★发版前置:两文件先 `hf upload` 到 datasets/yasoukyoku/utai-runtimes models/auxiliary/,否则下载 404。
const AUTOTUNE_FILES: &[AssetFile] = &[
    AssetFile { rel: "auxiliary/autotune_a1.onnx", size: 5_945_317, sha256: "e726cdf32e5ff08ee7c1ec65f3c7e0508df000f0aa8719a9f51ff72eb832ce05" },
    AssetFile { rel: "auxiliary/autotune_a1.json", size: 892, sha256: "f807abfc94eaa8836034fad0046608b647af3286bce452d9778caedc2f96c49e" },
];

// S75 声码器微调底模。★这是 app 里唯一带自有许可条款的资产包 —— 见 license/upstream 字段。
//
// 立场（用户 2026-07-23 定案，与 GAME 引擎同构 [[reference_game_engine]]）：权重 CC BY-NC-SA 4.0
// **绝不进 installer bundle**，我们只做「镜像 + 下载器」。CC BY-NC-SA 允许再分发，义务是署名 /
// 非商用 / 相同方式共享 —— 三条我们都履行：NOTICE 双语随权重同包下载（不是可选项，别为了省
// 7KB 把它们摘出去），UI 打许可徽章 + 上游链接，微调产物继承 NC 的提示留在参数页。
//
// 血统实证（S75 现场核验，别再重推）：上游 release 只有一个 zip
// github.com/openvpi/SingingVocoders/releases/download/v0.0.2/nsf_hifigan_44.1k_hop512_128bin_2024.02.zip
// (377,707,972 B)，内含这三个文件，**内部 ckpt 文件名与我们期望的一字不差**（曾怀疑是 `..._train.ckpt`
// —— 读 zip 中央目录证伪了）。dev 机这三份与 release 内容 **CRC32 + 大小逐字节同一**
// (ckpt crc32=0x3ec4e407 / zh NOTICE 0xb3004cf3 / NOTICE 0x34c73950)，故直接以它们为镜像原件。
//
// ⚠️ 这份 ckpt ≠ 推理用的「默认声码器」(auxiliary/nsf_hifigan.onnx)：后者 56.8MB、**只有
// generator**、且是 2022.12 代；微调要的是 2024.02 代的 lightning 训练档。两者格式类相同但
// 不可互换 —— 别再想着「复用已分发的那个」。
// ⛔★S119 更正：这里原本写着「generator.* 457 键 + discriminator.* 299 键」，那是**假的**。
// 逐件复算 sha256 与 size 都与上表逐位相同（= 字节没问题），而真值是 `{'state_dict': …}` 一个
// 顶层键、**509 项 = generator 303 + discriminator 206**，与我们生产 config 建出来的模型
// missing=0 / unexpected=0 逐键对上。那句错的散文的用途恰恰是「让下一个人认出这是不是对的
// 文件」，所以它是一条会误导人的证据链，不是排版。工具：`TESTING\s119_vocoder\probe_finetune_load.py`。
const VOCODER_TRAIN_FILES: &[AssetFile] = &[
    AssetFile { rel: "training/vocoder/nsf_hifigan_44.1k_hop512_128bin_2024.02.ckpt", size: 405_661_921, sha256: "5f4b4eb097b6e8126ada72651e32986908903b2780b478e3dfd05a5615f57fe2" },
    AssetFile { rel: "training/vocoder/NOTICE.txt", size: 3_725, sha256: "31005a94c1e591d3a09dd0702bc21082b2aaa2f36d3689fe248a651cdd83ebf6" },
    AssetFile { rel: "training/vocoder/NOTICE.zh-CN.txt", size: 3_653, sha256: "f62af3088f53d928db85092959b7b4dfe822ee1f63f3abb7c0f65e49a0c5b2b7" },
];

const PACKS: &[AssetPack] = &[
    AssetPack { id: "aux-inference", files: AUX_FILES, license: None, upstream: None },
    AssetPack { id: "aux-autotune", files: AUTOTUNE_FILES, license: None, upstream: None },
    AssetPack { id: "training-rvc", files: RVC_TRAIN_FILES, license: None, upstream: None },
    AssetPack { id: "training-sovits", files: SOVITS_TRAIN_FILES, license: None, upstream: None },
    AssetPack { id: "training-sovits-v2", files: SOVITS_V2_TRAIN_FILES, license: None, upstream: None },
    AssetPack {
        id: "training-vocoder",
        files: VOCODER_TRAIN_FILES,
        license: Some("CC BY-NC-SA 4.0"),
        upstream: Some("https://github.com/openvpi/SingingVocoders/releases/tag/v0.0.2"),
    },
];

/// License + upstream of the pack distributing `rel` — the missing-asset dialog needs both so a
/// license-bound download can never be offered as if it were just another silent fetch.
/// Same table as `pack_for_rel` on purpose: one catalog, no second mapping to drift.
pub(crate) fn pack_terms_for_rel(rel: &str) -> (Option<&'static str>, Option<&'static str>) {
    PACKS
        .iter()
        .find(|p| p.files.iter().any(|f| f.rel == rel))
        .map(|p| (p.license, p.upstream))
        .unwrap_or((None, None))
}

/// Which asset pack distributes the file at `rel` (forward-slash path under `<data>/models/`),
/// None = no catalog entry for this rel. As of S75 every asset `resolve_training_assets` can
/// demand is pack-distributed — the CC BY-NC-SA vocoder base was the last self-download holdout
/// and now rides `training-vocoder`, so a None here means a typo'd rel, not a manual asset.
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
    /// S75: license id when the pack's weights carry their own terms (rendered as a badge +
    /// upstream link). None for everything we can hand out without conditions.
    pub license: Option<String>,
    pub upstream: Option<String>,
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
                license: p.license.map(str::to_string),
                upstream: p.upstream.map(str::to_string),
            }
        })
        .collect()
}

/// Download every missing file of one pack, sequentially, through the shared engine.
/// `hf_base` = the user's custom HF host replacement (Settings 下载源; applyMirror semantics —
/// replaces the `https://huggingface.co` prefix). Tried first when set; huggingface.co and
/// hf-mirror.com always follow, sha256 verifying whatever answers.
/// S74b: reclaim an asset pack's files. Until now these packs could only ever be downloaded —
/// a user who stops using an architecture ("I'm never training RVC again") had no way to get the
/// gigabytes back, and the storage page could only watch them sit there.
///
/// Deletes exactly the files the CATALOG lists for the pack (never a directory sweep): the models
/// tree also holds the user's OWN models, and a recursive delete there would be unrecoverable.
/// Empty parent directories left behind are harmless and are what the next download re-fills.
///
/// Guards: the shared fail-closed idle pre-flight, plus a refusal while an asset download is in
/// flight (deleting the files it is writing produces a half-pack that passes no check).
/// The caller confirms first — including for `aux-inference`, whose removal disables voice
/// conversion until it is downloaded again (the dialog says so; refusing outright would just be
/// another "the app won't let me" dead end).
#[tauri::command]
pub async fn delete_asset_pack(
    state: State<'_, Arc<AppState>>,
    id: String,
) -> Result<u64, String> {
    crate::commands::window::ensure_idle_for_package_delete(&state)?;
    if ASSET_DL_ACTIVE.load(Ordering::SeqCst) {
        return Err("ASSET_DL_BUSY".to_string());
    }
    let pack = PACKS
        .iter()
        .find(|p| p.id == id)
        .ok_or_else(|| format!("ASSET_PACK_UNKNOWN: {id}"))?;
    let models_dir = state.models.models_dir().to_path_buf();
    let rels: Vec<&'static str> = pack.files.iter().map(|f| f.rel).collect();
    tokio::task::spawn_blocking(move || {
        let mut freed = 0u64;
        for rel in rels {
            let p = dest_path(&models_dir, rel);
            // S74b: the downloader keeps `<dest>.part` across cancellations on purpose (resume), so
            // a partially-downloaded pack's bytes live mostly in .part files — and the UI offers
            // Delete for exactly that state. Reclaiming only the finished files would leave the
            // gigabytes the user is trying to free.
            for target in [crate::download::part_path(&p), p.clone()] {
                let size = std::fs::metadata(&target).map(|m| m.len()).unwrap_or(0);
                match std::fs::remove_file(&target) {
                    Ok(()) => freed += size,
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    // Report the FIRST hard failure instead of silently leaving a half-deleted pack.
                    Err(e) => {
                        tracing::warn!("asset delete failed at {}: {e}", target.display());
                        return Err(format!("ASSET_DELETE_FAILED: {}: {e}", target.display()));
                    }
                }
            }
        }
        tracing::info!("Deleted asset pack {} ({} bytes reclaimed)", id, freed);
        Ok(freed)
    })
    .await
    .map_err(|e| format!("DELETE_TASK_FAILED: {e}"))?
}

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

#[cfg(test)]
mod tests {
    use super::*;

    /// A `rel` in two packs makes `pack_for_rel` answer arbitrarily and lets `delete_asset_pack`
    /// of one pack silently gut another. Nothing else enforces uniqueness — the catalog is a
    /// hand-written table.
    #[test]
    fn catalog_rels_are_unique_across_packs() {
        let mut seen = std::collections::HashSet::new();
        for p in PACKS {
            for f in p.files {
                assert!(seen.insert(f.rel), "duplicate rel across packs: {}", f.rel);
            }
        }
    }

    /// A typo'd or truncated hash is invisible until a user has downloaded the whole file and
    /// watched it fail verification — for the vocoder base that is 405 MB of wasted transfer.
    #[test]
    fn catalog_hashes_are_well_formed() {
        for p in PACKS {
            for f in p.files {
                assert_eq!(f.sha256.len(), 64, "sha256 not 64 chars: {}", f.rel);
                assert!(
                    f.sha256.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
                    "sha256 not lowercase hex: {}",
                    f.rel
                );
                assert!(f.size > 0, "zero size: {}", f.rel);
                assert!(!f.rel.starts_with('/') && !f.rel.contains(".."), "unsafe rel: {}", f.rel);
            }
        }
    }

    /// Mirroring someone else's weights is only lawful WITH the attribution link. Declaring a
    /// license and dropping the upstream would leave the UI showing terms it can't point anywhere.
    #[test]
    fn licensed_packs_carry_their_upstream() {
        for p in PACKS {
            if p.license.is_some() {
                assert!(p.upstream.is_some(), "licensed pack without upstream: {}", p.id);
            }
        }
    }

    /// S75 compliance pin: the vocoder base is CC BY-NC-SA and its NOTICE files must ship WITH the
    /// weights (attribution travels with the artifact — dropping them to "save 7 KB" breaks the
    /// license, and nothing at runtime would notice).
    #[test]
    fn vocoder_pack_is_licensed_and_ships_its_notices() {
        let p = PACKS.iter().find(|p| p.id == "training-vocoder").expect("training-vocoder pack");
        assert_eq!(p.license, Some("CC BY-NC-SA 4.0"));
        assert!(p.upstream.is_some_and(|u| u.contains("SingingVocoders")));
        for want in ["training/vocoder/NOTICE.txt", "training/vocoder/NOTICE.zh-CN.txt"] {
            assert!(p.files.iter().any(|f| f.rel == want), "missing {want}");
        }
    }

    /// `pack_for_rel` and `pack_terms_for_rel` read the same table on purpose — pin that they
    /// stay one lookup, so a file can never be offered for download without its terms.
    #[test]
    fn pack_lookup_and_terms_agree() {
        for p in PACKS {
            for f in p.files {
                assert_eq!(pack_for_rel(f.rel), Some(p.id), "{}", f.rel);
                assert_eq!(pack_terms_for_rel(f.rel), (p.license, p.upstream), "{}", f.rel);
            }
        }
        assert_eq!(pack_for_rel("nope/not/here"), None);
        assert_eq!(pack_terms_for_rel("nope/not/here"), (None, None));
    }
}
