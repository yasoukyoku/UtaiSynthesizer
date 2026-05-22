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
