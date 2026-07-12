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

pub fn get_log_dir() -> PathBuf {
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
