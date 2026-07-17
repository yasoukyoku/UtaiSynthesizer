use std::sync::Arc;

use tauri::menu::{Menu, MenuItem};
use tauri::{AppHandle, Manager, State};

use crate::separation::SeparationState;
use crate::AppState;

/// Quit the whole app (not just close the window). The frontend close-flow has already confirmed any
/// in-progress work + unsaved changes. Before exiting we KILL child processes the OS won't reap: the
/// training sidecar especially — on Windows the child Python isn't reaped when the parent exits, and
/// app.exit(0) doesn't run Drop, so it would orphan a GPU-pinning headless process. (Separation / MSST
/// native work runs in-process and dies with the process.)
#[tauri::command]
pub fn quit_app(app: AppHandle, state: State<'_, Arc<AppState>>) {
    let _ = state.training.force_stop();
    // Deliberate exit: drop the unclean-exit sentinel + flush the log worker (S68b).
    crate::crashlog::mark_clean_exit();
    app.exit(0);
}

/// The Rust-visible long tasks currently running, as stable string ids the frontend maps to localized
/// labels (`close.task_<id>`) and LISTS in the quit warning. The frontend adds what only IT can see
/// (workflow node executions, MSST downloads). When a new long task gains a queryable running flag, push
/// its id HERE (and add a `close.task_<id>` label in the locales).
#[tauri::command]
pub fn running_tasks(state: State<'_, Arc<AppState>>) -> Vec<String> {
    let mut tasks: Vec<String> = state.active_tasks.lock().keys().cloned().collect();
    if state.training.is_active() {
        tasks.push("training".to_string());
    }
    if matches!(
        state.separation.status().state,
        SeparationState::LoadingModel | SeparationState::Separating
    ) {
        tasks.push("separation".to_string());
    }
    tasks
}

/// Update the tray menu labels so the menu follows the UI language (called by the frontend on mount and
/// on language change — the tray is built with English fallback labels before the frontend is ready).
#[tauri::command]
pub fn set_tray_labels(app: AppHandle, show: String, quit: String) -> Result<(), String> {
    let show_i = MenuItem::with_id(&app, "show", show, true, None::<&str>).map_err(|e| e.to_string())?;
    let quit_i = MenuItem::with_id(&app, "quit", quit, true, None::<&str>).map_err(|e| e.to_string())?;
    let menu = Menu::with_items(&app, &[&show_i, &quit_i]).map_err(|e| e.to_string())?;
    if let Some(tray) = app.tray_by_id("main") {
        tray.set_menu(Some(menu)).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Show, unminimize and focus the main window (tray click, and before the tray-Quit close dialogs so
/// they're actually visible).
pub fn show_main(app: &AppHandle) {
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.show();
        let _ = w.unminimize();
        let _ = w.set_focus();
    }
}
