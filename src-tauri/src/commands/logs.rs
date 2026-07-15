use std::sync::Arc;
use tauri::State;

use crate::logging::LogEntry;
use crate::AppState;

#[tauri::command]
pub fn get_recent_logs(
    state: State<'_, Arc<AppState>>,
    count: Option<usize>,
) -> Vec<LogEntry> {
    state.log_buffer.recent(count.unwrap_or(200))
}

#[tauri::command]
pub fn get_logs_since(
    state: State<'_, Arc<AppState>>,
    after: String,
) -> Vec<LogEntry> {
    state.log_buffer.since(&after)
}

#[tauri::command]
pub fn get_log_file_path() -> String {
    let dir = crate::logging::get_log_dir();
    dir.to_string_lossy().to_string()
}

/// Open the log DIRECTORY in the OS file browser — the log panel's jump button (S67:
/// users hunted for the folder by hand). Fire-and-forget by design: explorer.exe
/// exits 1 even on success, so only a failed SPAWN is meaningful (log-only, no user
/// error surface — this never blocks anything). plugin-shell's JS open() is not an
/// option here: its default validator only allows mailto/tel/http(s).
#[tauri::command]
pub fn open_log_dir() {
    let dir = crate::logging::get_log_dir();
    #[cfg(windows)]
    let spawned = std::process::Command::new("explorer").arg(&dir).spawn();
    #[cfg(target_os = "macos")]
    let spawned = std::process::Command::new("open").arg(&dir).spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let spawned = std::process::Command::new("xdg-open").arg(&dir).spawn();
    if let Err(e) = spawned {
        tracing::warn!("open_log_dir: failed to open {}: {}", dir.display(), e);
    }
}

/// Bridge FRONTEND logs into the same tracing pipeline (panel buffer + file + stdout). Without this
/// the frontend's console.error is invisible to the log panel/file, so workflow/UI failures (e.g. an
/// MSST timeout) were silent. Captured by BufferLayer because this is a `utai` module.
#[tauri::command]
pub fn log_message(level: String, message: String) {
    match level.as_str() {
        "error" => tracing::error!("[UI] {}", message),
        "warn" => tracing::warn!("[UI] {}", message),
        "debug" => tracing::debug!("[UI] {}", message),
        _ => tracing::info!("[UI] {}", message),
    }
}
