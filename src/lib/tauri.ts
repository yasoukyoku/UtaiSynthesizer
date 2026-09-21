/** Environment-safe wrappers around Tauri APIs.
 *
 *  Tauri desktop apps run inside a WebView2 that injects `window.__TAURI_INTERNALS__`.
 *  When the same frontend is opened in a plain browser (dev preview, E2E test, or
 *  accidentally), that global is absent and every Tauri call throws. This module
 *  checks the environment up-front so the rest of the code base can remain
 *  browser-compatible without sprinkling `try/catch` everywhere.
 */

declare global {
  interface Window {
    __TAURI_INTERNALS__?: unknown;
  }
}

/** True when the runtime is an actual Tauri WebView (WebView2 / WKWebView / Lynx). */
export function isTauri(): boolean {
  return typeof window !== "undefined" && !!window.__TAURI_INTERNALS__;
}

/** Safely call a Tauri function, returning a no-op fallback when not in a Tauri
 *  environment. `fn` is a lazy getter so the import (e.g. `getCurrentWindow`) is
 *  only touched when we know the runtime is Tauri — importing @tauri-apps/api in a
 *  plain browser throws during module evaluation in some older bundlers. */
export async function tauriTry<T>(fn: () => T | Promise<T>, fallback: T): Promise<T> {
  if (!isTauri()) return fallback;
  try {
    return await fn();
  } catch {
    return fallback;
  }
}

/** Synchronous variant for the tiny subset of Tauri APIs that are sync
 *  (e.g. `getCurrentWindow()` before `.onCloseRequested()`). */
export function tauriTrySync<T>(fn: () => T, fallback: T): T {
  if (!isTauri()) return fallback;
  try {
    return fn();
  } catch {
    return fallback;
  }
}
