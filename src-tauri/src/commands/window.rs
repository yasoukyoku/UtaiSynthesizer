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

/// Quit and relaunch (S68c — the data-dir migration dialog's "restart now"). The frontend runs the
/// SAME close-flow gates as quit (in-progress work + unsaved changes) before calling this. Mirrors
/// quit_app's cleanup, plus an explicit window-state save: `AppHandle::restart()` leaves via
/// std::process::exit, so the window-state plugin's own exit-time save never runs (same reasoning
/// as the updater's pre-install save in update.rs).
#[tauri::command]
pub fn restart_app(app: AppHandle, state: State<'_, Arc<AppState>>) {
    let _ = state.training.force_stop();
    let _ = tauri_plugin_window_state::AppHandleExt::save_window_state(&app, crate::window_state_flags());
    crate::crashlog::mark_clean_exit();
    app.restart();
}

/// The Rust-visible long tasks currently running, as stable string ids the frontend maps to localized
/// labels (`close.task_<id>`) and LISTS in the quit warning. The frontend adds what only IT can see
/// (workflow node executions, MSST downloads). When a new long task gains a queryable running flag, push
/// its id HERE (and add a `close.task_<id>` label in the locales).
#[tauri::command]
pub fn running_tasks(state: State<'_, Arc<AppState>>) -> Vec<String> {
    running_tasks_of(&state)
}

/// Same list, callable from plain Rust (the delete pre-flight below) — one source, so a task that
/// starts blocking the quit flow automatically starts blocking destructive package removals too.
pub fn running_tasks_of(state: &AppState) -> Vec<String> {
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
    // S74b: voice conversion / vocal rendering and the audition family register NO entry in
    // active_tasks (run_rvc, run_sovits, render_vocal_segment, run_autotune have no begin_task), so
    // a list built only from active_tasks silently omitted the very jobs that hold the inference
    // model files open — the delete pre-flight below would have called a machine "idle" mid-render.
    // These two flags are the same ones acquire_convert_slot (lib.rs) checks; keep the three in
    // step. Both are also genuine quit-warning material, which is why they live here.
    if crate::commands::inference::voice_render_active() {
        tasks.push("render".to_string());
    }
    if crate::commands::audition::AUDITION_IN_FLIGHT.load(std::sync::atomic::Ordering::SeqCst) {
        tasks.push("audition".to_string());
    }
    tasks
}

/// S74b PRE-FLIGHT for every destructive package removal (runtime packs, asset packs, the CUDA
/// runtime): deleting files out from under a running job breaks it in ways that surface far from
/// the cause — a training run whose interpreter vanishes, an inference session whose DLLs are
/// unmapped mid-render.
///
/// Deliberately FAIL-CLOSED and coarse: ANY running task blocks ANY package delete. A per-package
/// "which family could be using this" matrix would be an enumeration, and an enumeration that
/// misses a newly added task id fails OPEN — silently allowing exactly the delete that corrupts a
/// run (the S61 cleanup-protection-set lesson: enumerate every holder, or don't enumerate).
/// The cost of being coarse is only that the user waits for a job to finish.
///
/// Returns the running task ids so the caller can name them; the frontend localizes them through
/// the SAME `close.task_<id>` keys the quit warning uses (no second vocabulary).
pub fn ensure_idle_for_package_delete(state: &AppState) -> Result<(), String> {
    let tasks = running_tasks_of(state);
    if tasks.is_empty() {
        Ok(())
    } else {
        Err(format!("DELETE_WHILE_BUSY: {}", tasks.join(",")))
    }
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
