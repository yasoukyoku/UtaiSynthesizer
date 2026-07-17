//! Crash forensics (S68b): panic hook, unclean-exit sentinels, and a startup autopsy
//! that reads the Windows Event Log.
//!
//! Motivation: the community "RVC 20%" crash dies with ZERO diagnostics — release
//! builds are `panic = "abort"` with no hook (a Rust panic prints to a stderr nobody
//! sees), the lossy non_blocking log channel drops its tail on abrupt death, and an
//! OS-level kill (commit exhaustion) or a GPU driver reset (TDR) leaves nothing in our
//! file at all. Windows, however, usually records something: Application log Event 1000
//! ("Application Error", faulting module + exception code) for native faults, and
//! System log Event 4101 ("Display driver stopped responding and has recovered") for
//! TDRs. Reading those on the restart after an unclean exit turns every community
//! "it just closed" report into a self-contained diagnosis — the two event classes
//! disambiguate exactly the silent-death mechanisms we cannot tell apart from our log.
//!
//! Sentinels are PER-PID (`session.<pid>.alive`) with an is-the-pid-alive check:
//! a concurrent second instance (or the dev restart cycle) must neither fire a false
//! autopsy about a healthy sibling nor delete the sibling's sentinel on its own clean
//! exit (adversarial review, round 1).

use std::path::{Path, PathBuf};

/// The lossy-channel worker guard, parked here so quit paths can drop it (= flush up to
/// 1 s of buffered lines). `app.exit(0)` never runs Drop (window.rs), so without this
/// even a CLEAN quit relies on the worker having kept up.
pub static LOG_GUARD: parking_lot::Mutex<Option<tracing_appender::non_blocking::WorkerGuard>> =
    parking_lot::Mutex::new(None);

fn sentinel_path(log_dir: &Path, pid: u32) -> PathBuf {
    log_dir.join(format!("session.{pid}.alive"))
}

fn now_epoch() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Is the pid a live process? OpenProcess + STILL_ACTIVE — a reused pid can read as
/// "alive" (skips one autopsy, rare and harmless); a dead pid never reads alive.
#[cfg(windows)]
fn pid_alive(pid: u32) -> bool {
    #[link(name = "kernel32")]
    extern "system" {
        fn OpenProcess(access: u32, inherit: i32, pid: u32) -> isize;
        fn CloseHandle(h: isize) -> i32;
        fn GetExitCodeProcess(h: isize, code: *mut u32) -> i32;
    }
    const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;
    const STILL_ACTIVE: u32 = 259;
    unsafe {
        let h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if h == 0 {
            return false;
        }
        let mut code: u32 = 0;
        let ok = GetExitCodeProcess(h, &mut code) != 0;
        CloseHandle(h);
        ok && code == STILL_ACTIVE
    }
}

#[cfg(not(windows))]
fn pid_alive(_pid: u32) -> bool {
    false
}

pub struct PrevSession {
    pub pid: u64,
    pub started_epoch: u64,
}

/// Collect sentinels from DEAD previous sessions (deleting their files), leave live
/// siblings' sentinels alone, then stamp our own. Call once at startup, after tracing
/// init. Returned sessions = unclean exits to autopsy (usually 0 or 1).
pub fn rotate_session_sentinel(log_dir: &Path) -> Vec<PrevSession> {
    let mut dead = Vec::new();
    if let Ok(rd) = std::fs::read_dir(log_dir) {
        for entry in rd.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let Some(pid) = name
                .strip_prefix("session.")
                .and_then(|s| s.strip_suffix(".alive"))
                .and_then(|s| s.parse::<u32>().ok())
            else {
                continue;
            };
            if pid == std::process::id() || pid_alive(pid) {
                continue; // ourselves / a live concurrent instance — not ours to touch
            }
            let started_epoch = std::fs::read_to_string(entry.path())
                .ok()
                .and_then(|s| s.split_whitespace().nth(1).and_then(|w| w.parse().ok()))
                .unwrap_or(0);
            let _ = std::fs::remove_file(entry.path());
            dead.push(PrevSession { pid: pid as u64, started_epoch });
        }
    }
    let pid = std::process::id();
    let _ = std::fs::write(sentinel_path(log_dir, pid), format!("{} {}", pid, now_epoch()));
    dead
}

/// Remove OUR sentinel — the exit ahead is deliberate. Used by quit_app and by the
/// updater right before install (its NSIS run exits the process without Drop).
pub fn remove_sentinel() {
    let _ = std::fs::remove_file(sentinel_path(
        &crate::logging::get_log_dir(),
        std::process::id(),
    ));
}

/// Re-stamp our sentinel after an aborted deliberate exit (e.g. updater install failed
/// and the session keeps running).
pub fn restore_sentinel() {
    let pid = std::process::id();
    let path = sentinel_path(&crate::logging::get_log_dir(), pid);
    let _ = std::fs::write(path, format!("{} {}", pid, now_epoch()));
}

/// Clean-quit bookkeeping: drop the sentinel AND flush the log worker. Only call when
/// the process is about to exit — after the guard drops, file logging is closed.
pub fn mark_clean_exit() {
    remove_sentinel();
    LOG_GUARD.lock().take();
}

/// Panic hook: raw-append the panic to today's log file FIRST (pure std — with
/// panic=abort the lossy tracing channel may never flush), then mirror into tracing
/// for the dev console / panel, then chain to the previous hook (stderr). The line is
/// assembled whole and written with ONE write_all so a concurrently-flushing logging
/// worker cannot interleave mid-line, and it carries the same timestamp format as
/// every tracing line (shared logging::LINE_TIME_FORMAT).
pub fn install_panic_hook(log_dir: PathBuf, offset: time::UtcOffset) {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let now = time::OffsetDateTime::now_utc().to_offset(offset);
        let stamp = now.format(crate::logging::LINE_TIME_FORMAT).unwrap_or_default();
        let name = crate::logging::log_file_name(crate::logging::LOG_PREFIX, now.date());
        let line = format!("{stamp} PANIC utai_lib::crashlog: {info}\n");
        if let Ok(mut f) =
            std::fs::OpenOptions::new().create(true).append(true).open(log_dir.join(name))
        {
            use std::io::Write;
            let _ = f.write_all(line.as_bytes());
        }
        tracing::error!("PANIC: {info}");
        prev(info);
    }));
}

/// Detached-thread autopsy after unclean previous exits. Queries are read-only
/// wevtutil calls; results land in the normal log (file + panel).
pub fn spawn_autopsy(prev: Vec<PrevSession>) {
    if prev.is_empty() {
        return;
    }
    std::thread::spawn(move || {
        let pids = prev.iter().map(|p| p.pid.to_string()).collect::<Vec<_>>().join(", ");
        tracing::warn!(
            "crash autopsy: previous session (pid {pids}) did not shut down cleanly (crash, forced kill, or an OS shutdown with the app open) — checking Windows error records"
        );
        // Records must postdate the oldest dead session's start. The 7-day cap applies
        // ONLY when no sentinel carried a valid epoch (review round 1: max()-ing the cap
        // over a valid epoch silently discarded true records after a >7-day pause).
        let cutoff = prev
            .iter()
            .map(|p| p.started_epoch)
            .filter(|&e| e > 0)
            .min()
            .unwrap_or_else(|| now_epoch().saturating_sub(7 * 24 * 3600));
        let mut found = false;

        // Native faults: Application log, "Application Error" (Event 1000), restricted
        // to OUR exe names SERVER-SIDE (review round 1: a newest-8 cap over all apps let
        // busy boxes crowd our record out and mislabel a recorded fault as an OS kill).
        // The client-side "utai" match stays as a belt.
        for ev in query_events(
            "Application",
            "*[System[Provider[@Name='Application Error'] and (EventID=1000)] and \
             EventData[Data='UtaiSynthesizer.exe' or Data='utai.exe']]",
            8,
        ) {
            if ev.epoch < cutoff || !ev.data.to_ascii_lowercase().contains("utai") {
                continue;
            }
            found = true;
            tracing::warn!("crash autopsy: Application Error (1000) at {}: {}", ev.time, ev.data);
        }

        // GPU driver resets: System log, Display 4101 (TDR). Not per-process — any hit
        // in the window is reported (a TDR at the crash timestamp = the smoking gun for
        // the black-screen deaths; commit-kills leave NO record in either log).
        for ev in query_events(
            "System",
            "*[System[Provider[@Name='Display'] and (EventID=4101)]]",
            4,
        ) {
            if ev.epoch < cutoff {
                continue;
            }
            found = true;
            tracing::warn!(
                "crash autopsy: display driver reset / TDR (4101) at {}: {}",
                ev.time,
                ev.data
            );
        }

        if !found {
            tracing::info!(
                "crash autopsy: no Application Error (1000) or display-reset (4101) records since the previous session start — a task-manager kill, OS shutdown, or commit-limit kill leaves none"
            );
        }
    });
}

struct EventRecord {
    time: String,
    epoch: u64,
    data: String,
}

/// Run `wevtutil qe <log> /q:<xpath> /rd:true /c:<n> /f:xml` and parse the events with
/// plain string scanning (structure-only XML from a system tool; no parser dependency).
#[cfg(windows)]
fn query_events(log: &str, xpath: &str, count: u32) -> Vec<EventRecord> {
    use std::os::windows::process::CommandExt;
    let out = match std::process::Command::new("wevtutil")
        .args(["qe", log, &format!("/q:{xpath}"), "/rd:true", &format!("/c:{count}"), "/f:xml"])
        .creation_flags(crate::util::CREATE_NO_WINDOW)
        .output()
    {
        Ok(o) if o.status.success() => o,
        _ => return Vec::new(),
    };
    let text = String::from_utf8_lossy(&out.stdout);
    text.split("</Event>").filter_map(parse_event).collect()
}

#[cfg(not(windows))]
fn query_events(_log: &str, _xpath: &str, _count: u32) -> Vec<EventRecord> {
    Vec::new()
}

fn parse_event(chunk: &str) -> Option<EventRecord> {
    // <TimeCreated SystemTime='2026-07-17T04:50:30.1234567Z'/> — wevtutil emits single
    // quotes; accept double quotes defensively.
    let time = ["SystemTime='", "SystemTime=\""].iter().find_map(|pat| {
        let start = chunk.find(pat)? + pat.len();
        let end = chunk[start..].find(['\'', '"'])? + start;
        Some(chunk[start..end].to_string())
    })?;
    let epoch = time::OffsetDateTime::parse(&time, &time::format_description::well_known::Rfc3339)
        .map(|t| t.unix_timestamp().max(0) as u64)
        .unwrap_or(0);

    // EventData <Data> values (Event 1000: app, version, module, exception code, …).
    let mut data: Vec<&str> = Vec::new();
    let mut rest = chunk;
    while let Some(open) = rest.find("<Data") {
        let after_open = &rest[open..];
        let Some(gt) = after_open.find('>') else { break };
        if after_open[..gt].ends_with('/') {
            rest = &after_open[gt + 1..];
            continue; // self-closing <Data/>
        }
        let content_start = gt + 1;
        let Some(close) = after_open[content_start..].find("</Data>") else { break };
        let val = after_open[content_start..content_start + close].trim();
        if !val.is_empty() {
            data.push(val);
        }
        rest = &after_open[content_start + close..];
    }
    let mut joined = data.join(" | ");
    if joined.len() > 700 {
        // Keep log lines bounded (the head carries the identifying fields). Floor to a
        // char boundary: a naive truncate(700) PANICS when a multibyte char straddles
        // the cut — with panic=abort that was a startup crash LOOP fed by any CJK-path
        // event record (adversarial review, round 1: the forensics feature crashing the
        // app it is meant to diagnose).
        let mut n = 700;
        while n > 0 && !joined.is_char_boundary(n) {
            n -= 1;
        }
        joined.truncate(n);
        joined.push_str(" …");
    }
    Some(EventRecord { time, epoch, data: joined })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_wevtutil_event_chunk() {
        let chunk = "<Event xmlns='http://schemas.microsoft.com/win/2004/08/events/event'>\
            <System><Provider Name='Application Error'/><EventID>1000</EventID>\
            <TimeCreated SystemTime='2026-07-17T04:50:30.1234567Z'/></System>\
            <EventData><Data>utai.exe</Data><Data>0.4.0</Data><Data/>\
            <Data>nvwgf2umx.dll</Data><Data>c0000005</Data></EventData>";
        let ev = parse_event(chunk).expect("parse");
        assert_eq!(ev.time, "2026-07-17T04:50:30.1234567Z");
        assert!(ev.epoch > 1_700_000_000);
        assert_eq!(ev.data, "utai.exe | 0.4.0 | nvwgf2umx.dll | c0000005");
    }

    #[test]
    fn event_without_timestamp_is_dropped() {
        assert!(parse_event("<Event><System></System>").is_none());
    }

    // The review-round-1 major: a multibyte char straddling byte 700 must not panic
    // the truncation (release panic=abort turned this into a startup crash loop).
    #[test]
    fn truncation_floors_to_char_boundary() {
        let mut payload = "a".repeat(699);
        payload.push('好'); // 3 bytes: 699..702 straddles the 700 cut
        let chunk = format!(
            "<Event><System><TimeCreated SystemTime='2026-07-17T04:50:30.1234567Z'/></System>\
             <EventData><Data>{payload}</Data></EventData>"
        );
        let ev = parse_event(&chunk).expect("parse");
        assert!(ev.data.ends_with(" …"));
        assert_eq!(&ev.data[..699], &"a".repeat(699));
    }
}
