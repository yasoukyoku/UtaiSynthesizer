//! Automated E2E of the unified downloader (S42) against a REAL local HTTP server:
//! full transfer, Range resume, sha256 rejection, mirror fallback. Runs in normal
//! `cargo test` (binds an ephemeral localhost port; no network).
//!
//! The server is a ~60-line hand-rolled responder because `python -m http.server`
//! does NOT implement Range — and the 206 resume path is exactly what needs proving.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

/// Serve `payload` at every path except `/missing` (404). Records the Range start
/// (None = no Range header) of each request for assertions.
fn spawn_server(payload: Arc<Vec<u8>>, ranges: Arc<Mutex<Vec<Option<u64>>>>) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut s) = stream else { break };
            let payload = Arc::clone(&payload);
            let ranges = Arc::clone(&ranges);
            std::thread::spawn(move || {
                let mut buf = Vec::new();
                let mut byte = [0u8; 1];
                // Read headers until CRLFCRLF (requests here are tiny).
                while !buf.ends_with(b"\r\n\r\n") && s.read(&mut byte).map(|n| n == 1).unwrap_or(false) {
                    buf.push(byte[0]);
                }
                let text = String::from_utf8_lossy(&buf);
                let path = text.split_whitespace().nth(1).unwrap_or("/").to_string();
                let range_start = text
                    .lines()
                    .find(|l| l.to_ascii_lowercase().starts_with("range:"))
                    .and_then(|l| l.split('=').nth(1))
                    .and_then(|v| v.trim().trim_end_matches('-').parse::<u64>().ok());
                ranges.lock().unwrap().push(range_start);

                if path == "/missing" {
                    let _ = s.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
                    return;
                }
                let total = payload.len() as u64;
                match range_start {
                    Some(start) if start < total => {
                        let body = &payload[start as usize..];
                        let head = format!(
                            "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {}-{}/{}\r\nConnection: close\r\n\r\n",
                            body.len(), start, total - 1, total
                        );
                        let _ = s.write_all(head.as_bytes());
                        let _ = s.write_all(body);
                    }
                    _ => {
                        let head = format!(
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            total
                        );
                        let _ = s.write_all(head.as_bytes());
                        let _ = s.write_all(&payload);
                    }
                }
            });
        }
    });
    port
}

fn test_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("utai_dl_test").join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn payload_bytes() -> Vec<u8> {
    // 3 MB deterministic non-trivial pattern.
    (0..3_000_000u32).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect()
}

fn sha_of(data: &[u8], dir: &std::path::Path) -> String {
    let p = dir.join("payload.ref");
    std::fs::write(&p, data).unwrap();
    utai_lib::download::sha256_file(&p).unwrap()
}

fn run_download(req: &utai_lib::download::DownloadRequest) -> utai_lib::Result<()> {
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    let client = utai_lib::download::client().unwrap();
    let cancel = Arc::new(AtomicBool::new(false));
    rt.block_on(utai_lib::download::download(&client, req, &cancel, |_done, _total| {}))
}

#[test]
fn downloader_full_resume_sha_and_mirrors() {
    let payload = Arc::new(payload_bytes());
    let ranges = Arc::new(Mutex::new(Vec::new()));
    let port = spawn_server(Arc::clone(&payload), Arc::clone(&ranges));
    let base = format!("http://127.0.0.1:{port}");
    let dir = test_dir("main");
    let good_sha = sha_of(&payload, &dir);

    // 1. full download + hash verify
    let dest = dir.join("full.bin");
    run_download(&utai_lib::download::DownloadRequest {
        urls: vec![format!("{base}/pack.bin")],
        dest: dest.clone(),
        sha256: Some(good_sha.clone()),
        expected_size: Some(payload.len() as u64),
    })
    .unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), *payload, "full download content");

    // 2. RESUME: pre-seed a truncated .part — the server must see Range: bytes=1000000-
    //    and the result must still be byte-perfect.
    let dest2 = dir.join("resume.bin");
    std::fs::write(utai_lib::download::part_path(&dest2), &payload[..1_000_000]).unwrap();
    ranges.lock().unwrap().clear();
    run_download(&utai_lib::download::DownloadRequest {
        urls: vec![format!("{base}/pack.bin")],
        dest: dest2.clone(),
        sha256: Some(good_sha.clone()),
        expected_size: Some(payload.len() as u64),
    })
    .unwrap();
    assert_eq!(std::fs::read(&dest2).unwrap(), *payload, "resumed content");
    let seen = ranges.lock().unwrap().clone();
    assert!(
        seen.contains(&Some(1_000_000)),
        "server never saw the resume Range request: {seen:?}"
    );

    // 3. sha mismatch → loud error AND the poisoned .part is discarded
    let dest3 = dir.join("badsha.bin");
    let err = run_download(&utai_lib::download::DownloadRequest {
        urls: vec![format!("{base}/pack.bin")],
        dest: dest3.clone(),
        sha256: Some("0".repeat(64)),
        expected_size: Some(payload.len() as u64),
    })
    .unwrap_err();
    // The backend emits the stable CODE (S62 i18n discipline) — assert on it, not on prose.
    assert!(
        err.to_string().contains("DOWNLOAD_SHA256_MISMATCH"),
        "unexpected error: {err}"
    );
    assert!(!dest3.exists(), "dest must not exist after sha failure");
    assert!(
        !utai_lib::download::part_path(&dest3).exists(),
        "corrupt .part must be discarded"
    );

    // 4. mirror fallback: first URL 404s, second succeeds
    let dest4 = dir.join("mirror.bin");
    run_download(&utai_lib::download::DownloadRequest {
        urls: vec![format!("{base}/missing"), format!("{base}/pack.bin")],
        dest: dest4.clone(),
        sha256: Some(good_sha),
        expected_size: Some(payload.len() as u64),
    })
    .unwrap();
    assert_eq!(std::fs::read(&dest4).unwrap(), *payload, "mirror fallback content");
}
