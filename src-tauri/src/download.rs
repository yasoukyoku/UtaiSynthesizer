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
/// .part for a later resume and returns a DOWNLOAD_CANCELLED error (contains the
/// "CANCELLED" sentinel — frontend isCancelError treats it as a user cancel).
pub async fn download(
    client: &reqwest::Client,
    req: &DownloadRequest,
    cancel: &Arc<AtomicBool>,
    mut progress: impl FnMut(u64, Option<u64>),
) -> Result<()> {
    if req.urls.is_empty() {
        return Err(err("DOWNLOAD_NO_SOURCE: empty URL list"));
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
                return Err(err("DOWNLOAD_CANCELLED"));
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
                "DOWNLOAD_SHA256_MISMATCH: expected {want}, got {got}"
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
        .map_err(|e| err(format!("DOWNLOAD_REQUEST_FAILED: {e}")))?;

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
            return Err(err(format!("DOWNLOAD_RANGE_INVALID: HTTP 416, cache cleared: {url}")));
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
            return Err(err("DOWNLOAD_CANCELLED"));
        }
        let chunk = match tokio::time::timeout(STALL_TIMEOUT, stream.next()).await {
            Err(_) => return Err(err(format!("DOWNLOAD_STALLED: no data for {}s: {url}", STALL_TIMEOUT.as_secs()))),
            Ok(None) => break,
            Ok(Some(Err(e))) => return Err(err(format!("DOWNLOAD_STREAM_INTERRUPTED: {e}"))),
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
            return Err(err(format!("DOWNLOAD_INCOMPLETE: {done}/{t} bytes: {url}")));
        }
        if done > t {
            // Over-long can never be right — poison, discard.
            let _ = std::fs::remove_file(part);
            return Err(err(format!("DOWNLOAD_OVERSIZE: {done}/{t} bytes, cache cleared: {url}")));
        }
    }
    Ok(())
}

// ─── connection probe (download-source health test, S43) ─────────────────────

/// Result of a download-source throughput probe. Deliberately a REAL transfer, not a
/// ping/HEAD: the GFW commonly lets small packets through the handshake but throttles
/// or resets sustained transfers, so a ping/HEAD false-positives ("放小包不放打包").
#[derive(serde::Serialize)]
pub struct ProbeResult {
    /// Got enough bytes to call the transfer real (not a reset-after-handshake).
    pub reachable: bool,
    /// "ok" | "slow" | "throttled" | "http_error" | "unreachable"
    pub verdict: String,
    pub mbps: f64,
    pub ttfb_ms: u64,
    pub bytes: u64,
    pub http_status: Option<u16>,
    pub error: Option<String>,
}

/// Probe `url` by transferring its first ~4 MB and measuring SUSTAINED throughput.
/// 4 MB is chosen to cross the GFW's small-packet allowance — a HEAD/small GET passes
/// even when big downloads get throttled/reset. Bounded by a 15 s total window and a
/// 5 s per-read stall: a slow-but-progressing link reports a low mbps (not an error);
/// a black-holed/reset one reports few bytes → "throttled"/"unreachable".
pub async fn probe(url: &str) -> ProbeResult {
    use futures_util::StreamExt;
    const WANT: u64 = 4_000_000;
    const TOTAL_DEADLINE: std::time::Duration = std::time::Duration::from_secs(15);
    const READ_STALL: std::time::Duration = std::time::Duration::from_secs(5);

    let unreachable = |e: String| ProbeResult {
        reachable: false,
        verdict: "unreachable".into(),
        mbps: 0.0,
        ttfb_ms: 0,
        bytes: 0,
        http_status: None,
        error: Some(e),
    };
    let client = match client() {
        Ok(c) => c,
        Err(e) => return unreachable(e.to_string()),
    };
    let t0 = std::time::Instant::now();
    // The header-wait must honor the total window too: a source that accepts the TCP/TLS
    // handshake but black-holes the HTTP response (a classic interference pattern) would
    // otherwise hang on the client's 60 s read_timeout, ~4x past the advertised 15 s.
    let send_fut = client
        .get(url)
        .header(reqwest::header::RANGE, format!("bytes=0-{}", WANT - 1))
        .send();
    let resp = match tokio::time::timeout(TOTAL_DEADLINE, send_fut).await {
        Err(_) => return unreachable("PROBE_TIMEOUT".to_string()),
        Ok(Err(e)) => {
            let msg = if e.is_timeout() {
                "PROBE_CONNECT_TIMEOUT".to_string()
            } else if e.is_connect() {
                "PROBE_CONNECT_FAILED".to_string()
            } else {
                format!("{e}")
            };
            return unreachable(msg);
        }
        Ok(Ok(r)) => r,
    };
    let status = resp.status();
    if !status.is_success() {
        return ProbeResult {
            reachable: false,
            verdict: "http_error".into(),
            mbps: 0.0,
            ttfb_ms: t0.elapsed().as_millis() as u64,
            bytes: 0,
            http_status: Some(status.as_u16()),
            error: Some(format!("PROBE_HTTP_ERROR: {}", status.as_u16())),
        };
    }
    let ttfb = t0.elapsed();
    let mut bytes: u64 = 0;
    let mut stream = resp.bytes_stream();
    // How the transfer ENDED matters as much as the byte count: a stream Err/None/stall
    // BEFORE WANT is the "burst then cut" pattern (>200 KB delivered fast, then RST) —
    // a real multi-GB download would fail the same way, so it must NOT grade ok/slow.
    // Only a deadline break at the top of the loop = "genuinely slow but progressing".
    let mut early_cut = false;
    loop {
        if t0.elapsed() > TOTAL_DEADLINE {
            break;
        }
        match tokio::time::timeout(READ_STALL, stream.next()).await {
            Ok(Some(Ok(c))) => {
                bytes += c.len() as u64;
                if bytes >= WANT {
                    break;
                }
            }
            Ok(Some(Err(_))) => {
                early_cut = true;
                break;
            } // stream reset mid-transfer
            Ok(None) => {
                early_cut = true;
                break;
            } // stream ended before WANT
            Err(_) => {
                early_cut = true;
                break;
            } // 5 s with no chunk = stopped delivering
        }
    }
    let reached = bytes >= WANT;
    let elapsed = t0.elapsed().as_secs_f64().max(0.001);
    let mbps = (bytes as f64 / 1_000_000.0) / elapsed;
    let verdict = if bytes < 200_000 {
        "throttled" // barely any data past the handshake (放小包不放打包)
    } else if early_cut && !reached {
        "throttled" // burst then cut short — the exact false-positive this probe catches
    } else if mbps < 0.1 {
        "throttled"
    } else if mbps < 1.0 {
        "slow"
    } else {
        "ok"
    };
    ProbeResult {
        // reachable = delivered the full sample, or was still progressing at the deadline
        // (slow but usable). A cut-short transfer is NOT reachable.
        reachable: reached || (!early_cut && bytes >= 200_000),
        verdict: verdict.into(),
        mbps,
        ttfb_ms: ttfb.as_millis() as u64,
        bytes,
        http_status: Some(status.as_u16()),
        error: None,
    }
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
