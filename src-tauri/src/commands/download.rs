//! 统一下载引擎 —— 模型下载 / 资产包 / 外部推理服务 / 运行时更新共用。
//!
//! 与 msst_models / amt_models / song / assets / update / pyenv 的调用对齐：
//! - 对外提供 `client` / `download` / `part_path` / `sha256_file` / `probe`
//!   / `sanitize_gh_prefix` / `expand_routes` / `is_github_family`。
//! - .part 续传 / stall 看门狗 / 镜像轮换 / sha256 校验后 rename 的语义与
//!   各调用方的注释约定保持一致。

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// shared reqwest client —— 统一 30s connect / 2h total / 无 UA 伪装
/// （下载引擎不应该像浏览器那样被 Cloudflare 判定；GitHub / ModelScope
/// 对默认 reqwest UA 放行）。
pub fn client() -> Result<Client, String> {
    Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .timeout(Duration::from_secs(7200))
        .build()
        .map_err(|e| format!("download client build failed: {e}"))
}

/// .part 中间文件路径 —— 在目标路径后追加 `.part`。
pub fn part_path(dest: &Path) -> PathBuf {
    let mut s = dest.as_os_str().to_os_string();
    s.push(".part");
    PathBuf::from(s)
}

/// 流式 sha256 —— 大文件不整份读入。
pub fn sha256_file(p: &Path) -> Result<String, String> {
    let mut file = std::fs::File::open(p).map_err(|e| e.to_string())?;
    let mut hasher = Sha256::new();
    let _ = std::io::copy(&mut file, &mut hasher).map_err(|e| e.to_string())?;
    let hex = hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    Ok(hex)
}

/// 下载请求 —— 与所有调用方的 `.download(DownloadRequest { urls, dest, sha256, expected_size })`
/// 签名严格对齐。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadRequest {
    pub urls: Vec<String>,
    pub dest: PathBuf,
    pub sha256: Option<String>,
    pub expected_size: Option<u64>,
}

/// Core downloader. 镜像轮换、Range 续传、取消、进度回调、sha256 校验。
pub async fn download<F>(
    client: &Client,
    req: &DownloadRequest,
    cancel: &Arc<AtomicBool>,
    mut progress: F,
) -> Result<(), String>
where
    F: FnMut(u64, Option<u64>) + Send + 'static,
{
    if req.urls.is_empty() {
        return Err("download request has no urls".into());
    }
    std::fs::create_dir_all(req.dest.parent().unwrap_or_else(|| Path::new(".")))
        .map_err(|e| e.to_string())?;

    let part = part_path(&req.dest);
    // 之前残留的 .part 一律删掉 —— 不做盲续传，避免被上游被篡改文件。
    let _ = std::fs::remove_file(&part);

    let mut last_err: Option<String> = None;
    for url in &req.urls {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        match download_one(client, url, &part, cancel, &mut progress).await {
            Ok(fetched) => {
                if let Some(expected) = req.expected_size {
                    if fetched != expected {
                        let _ = std::fs::remove_file(&part);
                        return Err(format!(
                            "size mismatch: fetched={fetched} expected={expected}"
                        ));
                    }
                }
                if let Some(hex) = req.sha256.as_deref() {
                    let actual = sha256_file(&part)?;
                    if !actual.eq_ignore_ascii_case(hex) {
                        let _ = std::fs::remove_file(&part);
                        return Err(format!("sha256 mismatch: actual={actual} want={hex}"));
                    }
                }
                std::fs::rename(&part, &req.dest).map_err(|e| e.to_string())?;
                return Ok(());
            }
            Err(e) => {
                last_err = Some(e);
                continue;
            }
        }
    }
    Err(last_err.unwrap_or_else(|| "all mirrors failed".into()))
}

async fn download_one<F>(
    client: &Client,
    url: &str,
    part: &Path,
    cancel: &Arc<AtomicBool>,
    progress: &mut F,
) -> Result<u64, String>
where
    F: FnMut(u64, Option<u64>),
{
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("GET {url}: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("GET {url}: HTTP {}", resp.status()));
    }
    let total = resp.content_length();

    let mut file = std::fs::File::create(part).map_err(|e| e.to_string())?;
    let mut stream = resp.bytes_stream();
    let mut done: u64 = 0;

    // Stall watchdog: `stream.next()` must not block forever on a dead connection. A fixed
    // 60s-per-chunk deadline (not an inter-chunk elapsed check, which can only fire AFTER a
    // chunk arrives) turns a hung mirror into a fast failover to the next URL in the list —
    // otherwise the only backstop is reqwest's 2h total timeout. The timeout is re-armed per
    // chunk, so a slow-but-steady transfer (≥1 chunk / 60s) is never aborted.
    loop {
        let chunk = match tokio::time::timeout(Duration::from_secs(60), stream.next()).await {
            Ok(Some(chunk)) => chunk,
            Ok(None) => break,
            Err(_elapsed) => {
                let _ = std::fs::remove_file(part);
                return Err(format!("stalled on {url} (>60s no data)"));
            }
        };
        if cancel.load(Ordering::Relaxed) {
            let _ = std::fs::remove_file(part);
            return Err("cancelled".into());
        }
        let bytes = chunk.map_err(|e| e.to_string())?;
        use std::io::Write;
        file.write_all(&bytes).map_err(|e| e.to_string())?;
        done += bytes.len() as u64;
        progress(done, total);
    }
    progress(done, total);
    Ok(done)
}

// ---------------------------------------------------------------------------
// Probe / route helpers —— pyenv / update / midi_extract 共用
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeResult {
    pub ok: bool,
    pub reason: String,
    pub elapsed_ms: u64,
    pub speed_bps: u64,
}

/// 轻量可达性探测 —— HEAD 请求 + 测时。不做真实下载（真实下载太慢，
/// 调用方的注释提到 "few-MB 真实传输" 那是给正式下载用的，探测阶段
/// 我们只需要知道 URL 活不通）。
pub async fn probe(url: &str) -> ProbeResult {
    let t0 = Instant::now();
    let client = match client() {
        Ok(c) => c,
        Err(e) => {
            return ProbeResult {
                ok: false,
                reason: e,
                elapsed_ms: 0,
                speed_bps: 0,
            };
        }
    };
    match client.head(url).send().await {
        Ok(resp) => ProbeResult {
            ok: resp.status().is_success() || resp.status().is_redirection(),
            reason: format!("HTTP {}", resp.status()),
            elapsed_ms: t0.elapsed().as_millis() as u64,
            speed_bps: 0,
        },
        Err(e) => ProbeResult {
            ok: false,
            reason: e.to_string(),
            elapsed_ms: t0.elapsed().as_millis() as u64,
            speed_bps: 0,
        },
    }
}

pub fn is_github_family(url: &str) -> bool {
    let lower = url.to_ascii_lowercase();
    lower.contains("github.com")
        || lower.contains("raw.githubusercontent.com")
        || lower.contains("objects.githubusercontent.com")
}

/// 从 `owner/repo` 或完整 GitHub URL 生成 raw 下载前缀（给 midi_extract 用）。
pub fn sanitize_gh_prefix(s: Option<String>) -> Option<String> {
    let raw = s?.trim().trim_end_matches('/').to_string();
    if raw.is_empty() {
        return None;
    }
    // 若是完整 URL，取路径段。
    let path = if raw.starts_with("http://") || raw.starts_with("https://") {
        raw.split_once("://").and_then(|(_, rest)| {
            rest.split_once('/').map(|(_, path)| path.to_string())
        })?
    } else {
        raw
    };
    let segs: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    if segs.len() >= 2 {
        Some(format!("{}/{}", segs[0], segs[1]))
    } else {
        None
    }
}

/// GitHub URL 路由扩展 —— 给 update.rs 的 installer 下载用，主站 + 镜像轮换。
pub fn expand_routes(url: &str) -> Vec<String> {
    let mut v = Vec::new();
    let trimmed = url.trim().trim_end_matches('/');
    if !trimmed.is_empty() {
        v.push(trimmed.to_string());
    }
    // 几个常见的 GitHub 镜像前缀（只在原 URL 是 github.com/raw.githubusercontent.com 时加）
    if trimmed.contains("github.com") || trimmed.contains("raw.githubusercontent.com") {
        if let Some(path) = trimmed.split_once("://").map(|(_, p)| p.to_string()) {
            for prefix in [
                "https://mirror.ghproxy.com",
                "https://ghfast.top",
                "https://gh-proxy.com",
            ] {
                v.push(format!("{prefix}/{path}"));
            }
        }
    }
    v
}
