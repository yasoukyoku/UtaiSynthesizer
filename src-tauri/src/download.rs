//! Unified large-file downloader — the single source of truth for resumable, verified
//! downloads (S42). Built for multi-GB runtime packs; the two earlier ad-hoc
//! `download_file` copies (commands/settings.rs, commands/msst_models.rs) predate this
//! module and are slated to migrate onto it.
//!
//! What the ad-hoc copies lack, all REQUIRED at runtime-pack scale:
//!   - mirror list: the SAME file behind several URLs (HF / hf-mirror / GH release),
//!     tried in order, with the partial file carried across mirror switches;
//!   - HTTP Range resume: progress lives in `<dest>.part` and survives failures,
//!     cancels and app restarts — a retry never restarts a 2 GB transfer from zero;
//!   - stall-based timeout: a whole-request timeout (settings.rs uses 600 s) kills any
//!     big download on a slow link; here only "no bytes for STALL_TIMEOUT" fails;
//!   - sha256 verification BEFORE the `.part` → dest rename — the rename is the commit
//!     point, so a half-written or corrupted file can never pass as complete.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::{Result, UtaiError};

/// No-bytes-arriving window after which the current attempt is abandoned (the next
/// attempt resumes from the .part). Deliberately generous: slow links make progress,
/// dead links don't.
const STALL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);
/// Full passes over the mirror list before giving up.
const MIRROR_ROUNDS: usize = 2;

pub struct DownloadRequest {
    /// Mirror URLs for the SAME content, tried in order. The .part is kept across
    /// mirror switches (byte-identical content assumed; the final sha256 is the
    /// backstop if a mirror ever serves something else).
    pub urls: Vec<String>,
    pub dest: PathBuf,
    /// Lowercase hex sha256 of the COMPLETE file. None skips verification (only
    /// acceptable for dev/local flows; catalog manifests always carry hashes).
    pub sha256: Option<String>,
    /// Expected size when the manifest knows it (drives progress before the server
    /// responds and sanity-checks over-长 downloads).
    pub expected_size: Option<u64>,
}

fn err(msg: impl Into<String>) -> UtaiError {
    UtaiError::Download(msg.into())
}

pub fn part_path(dest: &Path) -> PathBuf {
    let mut os = dest.as_os_str().to_owned();
    os.push(".part");
    PathBuf::from(os)
}

/// Streaming sha256 of a file (blocking — call sites off the async runtime wrap in
/// spawn_blocking). Lowercase hex.
pub fn sha256_file(path: &Path) -> Result<String> {
    use sha2::Digest;
    let mut file = std::fs::File::open(path)?;
    let mut hasher = sha2::Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

/// Shared client: NO whole-request timeout (multi-GB bodies). `read_timeout` bounds
/// every socket read — INCLUDING the wait for response headers, which is where a
/// black-holing proxy/CDN would otherwise hang `send().await` forever with the
/// cancel flag unreachable (audit S42: that wedges the InstallGuard until restart).
/// A slow-but-progressing transfer never trips it; the per-chunk STALL_TIMEOUT in
/// the read loop stays as the second, explicit layer.
pub fn client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .user_agent("UTAI/2.0")
        .connect_timeout(std::time::Duration::from_secs(30))
        .read_timeout(STALL_TIMEOUT)
        .build()
        .map_err(|e| err(format!("HTTP client: {e}")))
}

/// Download `req` with resume + mirrors + verification. `progress(done, total)` is
/// called with ABSOLUTE byte counts (total None until any server reports one).
/// `cancel` is cooperative (checked between chunks); a cancelled download keeps its
/// .part for a later resume and returns an error mentioning 取消.
pub async fn download(
    client: &reqwest::Client,
    req: &DownloadRequest,
    cancel: &Arc<AtomicBool>,
    mut progress: impl FnMut(u64, Option<u64>),
) -> Result<()> {
    if req.urls.is_empty() {
        return Err(err("没有可用的下载源（URL 列表为空）"));
    }
    if let Some(parent) = req.dest.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Idempotence: a complete, hash-matching dest is DONE (re-entry after a crash
    // between rename and caller bookkeeping).
    if req.dest.exists() {
        match &req.sha256 {
            Some(want) => {
                let dest = req.dest.clone();
                let got = tokio::task::spawn_blocking(move || sha256_file(&dest))
                    .await
                    .map_err(|e| err(format!("hash task: {e}")))??;
                if got.eq_ignore_ascii_case(want) {
                    return Ok(());
                }
                // Wrong content under the final name — replace it.
                std::fs::remove_file(&req.dest)?;
            }
            None => return Ok(()),
        }
    }

    let part = part_path(&req.dest);
    let mut last_err: Option<UtaiError> = None;

    'attempts: for round in 0..MIRROR_ROUNDS {
        for url in &req.urls {
            if cancel.load(Ordering::SeqCst) {
                return Err(err("下载已取消（进度已保留，可续传）"));
            }
            match stream_one(client, url, &part, req, cancel, &mut progress).await {
                Ok(()) => {
                    last_err = None;
                    break 'attempts;
                }
                Err(e) => {
                    tracing::warn!("download attempt failed (round {round}, {url}): {e}");
                    last_err = Some(e);
                }
            }
        }
    }
    if let Some(e) = last_err {
        return Err(e);
    }

    // Verify BEFORE the commit rename.
    if let Some(want) = &req.sha256 {
        let p = part.clone();
        let got = tokio::task::spawn_blocking(move || sha256_file(&p))
            .await
            .map_err(|e| err(format!("hash task: {e}")))??;
        if !got.eq_ignore_ascii_case(want) {
            // A corrupt part can never resume into a correct file — discard it.
            let _ = std::fs::remove_file(&part);
            return Err(err(format!(
                "sha256 校验失败（期望 {want}，实际 {got}）——已删除损坏的下载缓存，请重试"
            )));
        }
    }
    std::fs::rename(&part, &req.dest)?;
    Ok(())
}

/// One attempt against one URL: open (with Range when the .part has bytes), stream to
/// the .part, per-chunk stall timeout. Ok(()) = the file is COMPLETE in the .part.
async fn stream_one(
    client: &reqwest::Client,
    url: &str,
    part: &Path,
    req: &DownloadRequest,
    cancel: &Arc<AtomicBool>,
    progress: &mut impl FnMut(u64, Option<u64>),
) -> Result<()> {
    use futures_util::StreamExt;
    use tokio::io::AsyncWriteExt;

    let mut offset = std::fs::metadata(part).map(|m| m.len()).unwrap_or(0);

    // Already have every expected byte (e.g. crash after the last chunk, before
    // rename)? Skip the network round-trip; the caller's sha check decides.
    if let Some(total) = req.expected_size {
        if offset >= total {
            return Ok(());
        }
    }

    let mut request = client.get(url);
    if offset > 0 {
        request = request.header(reqwest::header::RANGE, format!("bytes={offset}-"));
    }
    let resp = request
        .send()
        .await
        .map_err(|e| err(format!("请求失败: {e}")))?;

    let status = resp.status();
    let total: Option<u64> = match status.as_u16() {
        206 => {
            // Content-Range: bytes <start>-<end>/<total>
            resp.headers()
                .get(reqwest::header::CONTENT_RANGE)
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.rsplit('/').next())
                .and_then(|t| t.parse::<u64>().ok())
        }
        200 => {
            // Server ignored (or wasn't sent) Range — restart from zero.
            if offset > 0 {
                tracing::info!("server ignored Range — restarting {}", part.display());
            }
            offset = 0;
            let _ = std::fs::remove_file(part);
            resp.content_length()
        }
        416 => {
            // Requested range not satisfiable — the .part may already be complete
            // (server total == offset) or it's garbage. Let the sha check decide;
            // when sizes are known and disagree, discard.
            if let Some(expected) = req.expected_size {
                if offset == expected {
                    return Ok(());
                }
            }
            let _ = std::fs::remove_file(part);
            return Err(err(format!("HTTP 416（续传范围无效，已清除缓存重试）: {url}")));
        }
        _ => {
            return Err(err(format!("HTTP {status}: {url}")));
        }
    };
    let total = total.or(req.expected_size);

    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(part)
        .await?;
    let mut stream = resp.bytes_stream();
    let mut done = offset;
    progress(done, total);

    loop {
        if cancel.load(Ordering::SeqCst) {
            file.flush().await?;
            return Err(err("下载已取消（进度已保留，可续传）"));
        }
        let chunk = match tokio::time::timeout(STALL_TIMEOUT, stream.next()).await {
            Err(_) => return Err(err(format!("下载停滞超过 {}s: {url}", STALL_TIMEOUT.as_secs()))),
            Ok(None) => break,
            Ok(Some(Err(e))) => return Err(err(format!("下载流中断: {e}"))),
            Ok(Some(Ok(c))) => c,
        };
        file.write_all(&chunk).await?;
        done += chunk.len() as u64;
        progress(done, total);
    }
    file.flush().await?;

    // Short body = treat as a failed attempt so the next mirror/round resumes it.
    if let Some(t) = total {
        if done < t {
            return Err(err(format!("下载不完整（{done}/{t} 字节）: {url}")));
        }
        if done > t {
            // Over-long can never be right — poison, discard.
            let _ = std::fs::remove_file(part);
            return Err(err(format!("下载超长（{done}/{t} 字节，已清除缓存）: {url}")));
        }
    }
    Ok(())
}

/// A `Read` that concatenates several files in order — lets multi-part pack archives
/// stream straight into zstd+tar without materializing a joined copy on disk.
pub struct MultiFileReader {
    paths: Vec<PathBuf>,
    idx: usize,
    current: Option<std::io::BufReader<std::fs::File>>,
}

impl MultiFileReader {
    pub fn new(paths: Vec<PathBuf>) -> Self {
        Self { paths, idx: 0, current: None }
    }
}

impl std::io::Read for MultiFileReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        loop {
            if self.current.is_none() {
                if self.idx >= self.paths.len() {
                    return Ok(0);
                }
                let f = std::fs::File::open(&self.paths[self.idx])?;
                self.idx += 1;
                self.current = Some(std::io::BufReader::new(f));
            }
            let n = self.current.as_mut().unwrap().read(buf)?;
            if n > 0 {
                return Ok(n);
            }
            self.current = None; // EOF on this part — advance
        }
    }
}
