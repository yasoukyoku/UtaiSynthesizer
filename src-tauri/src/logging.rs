use std::collections::VecDeque;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;
use serde::Serialize;
use tracing::field::{Field, Visit};
use tracing::Subscriber;
use tracing_subscriber::layer::Context;
use tracing_subscriber::Layer;

#[derive(Debug, Clone, Serialize)]
pub struct LogEntry {
    pub timestamp: String,
    pub level: String,
    pub module: String,
    pub message: String,
}

/// The app's log-file prefix — `<prefix>.<YYYY-MM-DD>` files under get_log_dir().
pub const LOG_PREFIX: &str = "utai.log";

/// Per-line timestamp format — ONE source shared by the two fmt layers (lib.rs
/// OffsetTime) and the panic hook's raw append (crashlog.rs), so a PANIC line looks
/// like every other line. S68b (§user): the offset moved into a "(UTC+08:00)"
/// parenthetical — the bare RFC3339 "+08:00" suffix read as "add 8 hours".
pub const LINE_TIME_FORMAT: &[time::format_description::BorrowedFormatItem<'static>] =
    time::macros::format_description!(
        "[year]-[month]-[day]T[hour]:[minute]:[second].[subsecond digits:6] (UTC[offset_hour sign:mandatory]:[offset_minute])"
    );

/// `<prefix>.<YYYY-MM-DD>` — THE file-name scheme (S67). Single source shared by
/// LocalDailyFile and the panic hook's raw append (crashlog.rs), which must hit the
/// same file the logging worker writes.
pub fn log_file_name(prefix: &str, date: time::Date) -> String {
    format!("{}.{:04}-{:02}-{:02}", prefix, date.year(), u8::from(date.month()), date.day())
}

/// Daily-rolling log file writer whose file-name DATE and roll boundary follow the
/// LOCAL offset (S67 follow-up, §user): tracing-appender 0.2's rolling::daily is
/// hardwired to UTC, so a UTC+8/+9 user's evening lines landed in "yesterday's"
/// file and the day flipped at 08:00/09:00 local — we know the offset, so we roll
/// ourselves. Same naming scheme as rolling::daily (`<prefix>.<YYYY-MM-DD>`) and
/// append mode (an app restart on the same day continues the file). The offset is
/// captured once at startup — consistent with the per-line OffsetTime timestamps.
/// Wrapped in tracing_appender::non_blocking, so the per-write date check runs on
/// the logging worker thread (one now_utc() call per batched write). Open failures
/// degrade to discarding file output — never a panic in the log path.
pub struct LocalDailyFile {
    dir: PathBuf,
    prefix: &'static str,
    offset: time::UtcOffset,
    date: time::Date,
    file: Option<std::fs::File>,
    open_warned: bool,
}

impl LocalDailyFile {
    pub fn new(dir: PathBuf, prefix: &'static str, offset: time::UtcOffset) -> Self {
        let date = Self::today(offset);
        let mut this = Self { dir, prefix, offset, date, file: None, open_warned: false };
        this.file = this.open(date);
        this
    }

    fn today(offset: time::UtcOffset) -> time::Date {
        time::OffsetDateTime::now_utc().to_offset(offset).date()
    }

    fn open(&mut self, date: time::Date) -> Option<std::fs::File> {
        let name = log_file_name(self.prefix, date);
        match std::fs::OpenOptions::new().create(true).append(true).open(self.dir.join(&name)) {
            Ok(f) => {
                self.open_warned = false;
                Some(f)
            }
            Err(e) => {
                // stderr only, once per failure episode — this IS the log sink
                if !self.open_warned {
                    self.open_warned = true;
                    eprintln!("log file open failed ({}): {}", name, e);
                }
                None
            }
        }
    }

    fn roll_if_needed(&mut self) {
        let today = Self::today(self.offset);
        if today != self.date || self.file.is_none() {
            self.date = today;
            self.file = self.open(today);
        }
    }
}

impl std::io::Write for LocalDailyFile {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.roll_if_needed();
        match self.file.as_mut() {
            Some(f) => f.write(buf),
            // no sink — swallow so the non_blocking worker never wedges on errors
            None => Ok(buf.len()),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self.file.as_mut() {
            Some(f) => f.flush(),
            None => Ok(()),
        }
    }
}

pub struct LogBuffer {
    entries: Mutex<VecDeque<LogEntry>>,
    capacity: usize,
}

impl LogBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Mutex::new(VecDeque::with_capacity(capacity)),
            capacity,
        }
    }

    pub fn push(&self, entry: LogEntry) {
        let mut buf = self.entries.lock();
        if buf.len() >= self.capacity {
            buf.pop_front();
        }
        buf.push_back(entry);
    }

    pub fn recent(&self, count: usize) -> Vec<LogEntry> {
        let buf = self.entries.lock();
        let start = buf.len().saturating_sub(count);
        buf.iter().skip(start).cloned().collect()
    }

    pub fn since(&self, after_timestamp: &str) -> Vec<LogEntry> {
        let buf = self.entries.lock();
        buf.iter()
            .filter(|e| e.timestamp.as_str() > after_timestamp)
            .cloned()
            .collect()
    }
}

pub struct BufferLayer {
    buffer: Arc<LogBuffer>,
}

impl BufferLayer {
    pub fn new(buffer: Arc<LogBuffer>) -> Self {
        Self { buffer }
    }
}

impl<S: Subscriber> Layer<S> for BufferLayer {
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        let meta = event.metadata();

        // User log panel: utai messages only. ort/symphonia go to file log.
        let module = meta.module_path().unwrap_or("");
        if !module.starts_with("utai") {
            return;
        }
        // Skip excessive model scan logs etc.
        if *meta.level() > tracing::Level::DEBUG {
            return;
        }
        let mut visitor = MessageVisitor::default();
        event.record(&mut visitor);

        let now = std::time::SystemTime::now();
        let timestamp = format_timestamp(now);

        self.buffer.push(LogEntry {
            timestamp,
            level: meta.level().to_string().to_uppercase(),
            module: meta
                .module_path()
                .unwrap_or("")
                .strip_prefix("utai_lib::")
                .or_else(|| meta.module_path())
                .unwrap_or("")
                .to_string(),
            message: visitor.message,
        });
    }
}

#[derive(Default)]
struct MessageVisitor {
    message: String,
}

impl Visit for MessageVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        if field.name() == "message" {
            self.message = format!("{:?}", value);
        } else if self.message.is_empty() {
            self.message = format!("{:?}", value);
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message = value.to_string();
        } else if self.message.is_empty() {
            self.message = value.to_string();
        }
    }
}

fn format_timestamp(_time: std::time::SystemTime) -> String {
    #[cfg(windows)]
    {
        #[repr(C)]
        #[allow(non_snake_case)]
        struct SYSTEMTIME {
            wYear: u16, wMonth: u16, wDayOfWeek: u16, wDay: u16,
            wHour: u16, wMinute: u16, wSecond: u16, wMilliseconds: u16,
        }
        extern "system" {
            fn GetLocalTime(lpSystemTime: *mut SYSTEMTIME);
        }
        let mut st = SYSTEMTIME {
            wYear: 0, wMonth: 0, wDayOfWeek: 0, wDay: 0,
            wHour: 0, wMinute: 0, wSecond: 0, wMilliseconds: 0,
        };
        unsafe { GetLocalTime(&mut st); }
        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}",
            st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond, st.wMilliseconds
        )
    }
    #[cfg(not(windows))]
    {
        let duration = _time
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        let secs = duration.as_secs();
        let millis = duration.subsec_millis();
        let total_days = secs / 86400;
        let day_secs = secs % 86400;
        let h = day_secs / 3600;
        let m = (day_secs % 3600) / 60;
        let s = day_secs % 60;
        let (year, month, day) = days_to_ymd(total_days as i64 + 719468);
        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}",
            year, month, day, h, m, s, millis
        )
    }
}

#[cfg(not(windows))]
fn days_to_ymd(days: i64) -> (i64, u32, u32) {
    let era = if days >= 0 { days } else { days - 146096 } / 146097;
    let doe = (days - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// S68e fullportable: RELEASE builds root the logs next to the exe (`<install>\logs`)
/// so the whole install is one carry-able folder; the legacy identifier dir stays as
/// the fallback when the exe dir is unwritable (folder moved under a protected path,
/// read-only media). Dev builds keep the legacy location on purpose — the dev app
/// root is the REPO, and a repo-local logs/ would pollute the working tree. Resolved
/// ONCE per process (the probe writes a marker file; 8 call sites — crash sentinels,
/// log panel, storage stats, open-folder — must all agree anyway).
pub fn get_log_dir() -> PathBuf {
    static LOG_DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    LOG_DIR
        .get_or_init(|| {
            #[cfg(all(target_os = "windows", not(debug_assertions)))]
            if let Some(dir) = portable_log_dir() {
                return dir;
            }
            legacy_log_dir()
        })
        .clone()
}

/// `<exe dir>\logs` when creatable AND writable (probed with a marker write — a
/// merely-creatable dir on read-only media would lose every log line silently).
#[cfg(all(target_os = "windows", not(debug_assertions)))]
fn portable_log_dir() -> Option<PathBuf> {
    let dir = std::env::current_exe().ok()?.parent()?.join("logs");
    std::fs::create_dir_all(&dir).ok()?;
    let probe = dir.join(".writable");
    std::fs::write(&probe, b"").ok()?;
    let _ = std::fs::remove_file(&probe);
    Some(dir)
}

/// The pre-S68e log home (`%LOCALAPPDATA%\com.utaisynthesizer.app\logs`) — still the
/// dev-build location, the unwritable-root fallback, and the migration SOURCE.
pub fn legacy_log_dir() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        if let Ok(local) = std::env::var("LOCALAPPDATA") {
            return PathBuf::from(local)
                .join("com.utaisynthesizer.app")
                .join("logs");
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        if let Some(data) = dirs_next::data_local_dir() {
            return data.join("com.utaisynthesizer.app").join("logs");
        }
    }
    PathBuf::from(".").join("logs")
}

/// S68e one-time move of the legacy log home into the merged location — log files AND
/// crash sentinels (`session.<pid>.alive`), so the crash-autopsy chain stays unbroken
/// across the update. Same-name targets are left alone (re-runs are no-ops); rename
/// first, copy+delete as the cross-volume fallback; everything best-effort and SILENT
/// (runs before tracing exists). No-op when the locations coincide (dev builds).
pub fn migrate_legacy_logs(new_dir: &PathBuf) {
    let old = legacy_log_dir();
    if old == *new_dir || !old.is_dir() {
        return;
    }
    // DEFER the whole migration while any OTHER live session still logs into the
    // legacy home (a pre-update copy running elsewhere) — review S68e: moving its
    // OPEN daily file cross-volume sends the file delete-pending (every line it
    // writes after our copy snapshot silently dies), and moving its live
    // session.<pid>.alive sentinel fires a FALSE crash autopsy on our next boot.
    // Retried automatically on every boot; once that session ends, files move.
    if foreign_live_session_in(&old) {
        return;
    }
    let Ok(rd) = std::fs::read_dir(&old) else { return };
    for e in rd.flatten() {
        let from = e.path();
        if !from.is_file() {
            continue;
        }
        let to = new_dir.join(e.file_name());
        if to.exists() {
            continue;
        }
        if std::fs::rename(&from, &to).is_err() {
            if std::fs::copy(&from, &to).is_ok() {
                let _ = std::fs::remove_file(&from);
            }
        }
    }
    let _ = std::fs::remove_dir(&old); // only succeeds once fully emptied
}

/// TRUE when `dir` holds a `session.<pid>.alive` sentinel of a LIVE process other
/// than ours — i.e. a pre-merge copy of the app is running right now and still uses
/// the legacy homes. Shared by the log migration (deferring the file moves) and the
/// S68e.1 webview-profile reclaim (a live old instance's profile is only PARTIALLY
/// lock-protected: remove_dir_all would tear out its closed files — leveldb tables,
/// Preferences — before hitting the first locked one, corrupting the live session).
pub(crate) fn foreign_live_session_in(dir: &PathBuf) -> bool {
    let me = std::process::id();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if let Some(pid) = name
                .strip_prefix("session.")
                .and_then(|s| s.strip_suffix(".alive"))
                .and_then(|s| s.parse::<u32>().ok())
            {
                if pid != me && crate::crashlog::pid_alive(pid) {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    /// Pins the LocalDailyFile contract: rolling::daily's naming scheme
    /// (`<prefix>.<YYYY-MM-DD>`) with the DATE taken in the given offset, append mode.
    #[test]
    fn local_daily_file_names_by_offset_date_and_appends() {
        let dir = std::env::temp_dir().join(format!("utai_localdaily_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let offset = time::UtcOffset::from_hms(8, 0, 0).unwrap();
        let date = time::OffsetDateTime::now_utc().to_offset(offset).date();
        let name = format!(
            "utai.log.{:04}-{:02}-{:02}",
            date.year(),
            u8::from(date.month()),
            date.day()
        );

        let mut w = LocalDailyFile::new(dir.clone(), "utai.log", offset);
        w.write_all(b"one\n").unwrap();
        w.flush().unwrap();
        drop(w);
        // restart on the same day must APPEND, not truncate
        let mut w2 = LocalDailyFile::new(dir.clone(), "utai.log", offset);
        w2.write_all(b"two\n").unwrap();
        w2.flush().unwrap();
        drop(w2);

        let text = std::fs::read_to_string(dir.join(&name)).expect("local-date file exists");
        assert_eq!(text, "one\ntwo\n");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
