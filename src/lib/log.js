import { invoke } from "@tauri-apps/api/core";
/** Forward a frontend log line into the Rust tracing pipeline so it appears in the log panel AND the
 *  log file. The panel only mirrors Rust logs (BufferLayer captures `utai` modules), so a bare
 *  console.error is invisible there — that was why workflow/MSST failures looked "silent".
 *  Fire-and-forget; never throws. */
export function logToBackend(level, message) {
    invoke("log_message", { level, message }).catch(() => { });
}
