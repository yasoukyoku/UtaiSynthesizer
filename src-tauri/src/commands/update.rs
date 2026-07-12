// S64 — In-app update: check + download + install from GitHub Releases (tauri-plugin-updater,
// driven ENTIRELY from Rust so the endpoint list can follow the user's GH-mirror setting at runtime —
// the JS plugin API can't override endpoints per-call).
//
// Flow (frontend drives, single-flight by UI):
//   update_check(gh_proxy?) → Some(UpdateInfo) | None (up to date). On success the plugin's `Update`
//   handle is stashed module-side; update_install() then downloads (progress → "update-progress"
//   window events, stall-watchdog + update_cancel supported) and installs. On Windows install()
//   spawns the NSIS setup with /P /R (passive UI, relaunch after install) and exits this process —
//   update_install never returns on success.
//
// GH mirror (audit-hardened, S64):
//   - the prefix comes through download::sanitize_gh_prefix — **https only**, because the plugin's
//     endpoint validation hard-rejects http in RELEASE builds (warn-only in dev: the one shape of
//     breakage `tauri dev` can never reproduce). An http prefix degrades to the direct route.
//   - check endpoints = [<prefix>/<direct-json>, <direct-json>] tried in order (2XX wins).
//   - the announced download_url (a github.com release asset) is rewritten through the same prefix,
//     and the ORIGINAL url is kept: if the mirror download fails, ONE direct-route retry runs before
//     giving up (public proxies die without notice — a dead mirror must never make an update
//     uninstallable while the direct route works). minisign verifies file CONTENT, so any source is
//     tamper-safe.
//
// Errors are stable CODEs (i18n via src/lib/backendError.ts): UPDATE_CHECK_FAILED / UPDATE_NO_PENDING /
// UPDATE_DOWNLOAD_FAILED / UPDATE_INSTALL_FAILED; cancels carry the CANCELLED sentinel (swallowed
// silently by the frontend's isCancelError, per the app-wide convention).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tauri::{Emitter, State};
use tauri_plugin_updater::UpdaterExt;

use crate::AppState;

/// The one update source (single source of truth — tauri.conf.json deliberately carries only the
/// pubkey). GitHub's `latest/download` redirect always points at the newest NON-prerelease release,
/// so publishing a release with a `latest.json` asset is the whole update-push protocol.
const UPDATE_ENDPOINT: &str =
    "https://github.com/yasoukyoku/UtaiSynthesizer/releases/latest/download/latest.json";

/// No-bytes window after which the download attempt is abandoned (same posture as download.rs's
/// stall watchdog: slow links make progress, dead links don't — never a whole-request timeout,
/// which would kill legitimately slow transfers of a ~100MB installer).
const DOWNLOAD_STALL: Duration = Duration::from_secs(60);

/// Update handle between check and install, plus the DIRECT download URL kept when the announced
/// one was rewritten through the GH mirror (the install's fallback route). install() takes it out
/// and RESTORES it on failure so a retry needs no re-check.
static PENDING_UPDATE: Mutex<Option<(tauri_plugin_updater::Update, Option<tauri::Url>)>> =
    Mutex::new(None);

/// Cooperative cancel for the in-flight download (update_cancel flips it; the watchdog loop drops
/// the download future). Reset at each update_install start.
static UPDATE_CANCEL: AtomicBool = AtomicBool::new(false);

#[derive(serde::Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct UpdateInfo {
    pub version: String,
    pub current_version: String,
    pub notes: Option<String>,
}

#[derive(serde::Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct UpdateProgress {
    downloaded: u64,
    total: Option<u64>,
}

/// Same host family as the frontend's applyGhMirror (precise set + subdomain fallback) — the
/// download_url in OUR latest.json is always a github.com release asset, the redirect targets
/// (objects.githubusercontent.com) are chased by the proxy itself.
fn is_github_family(url: &tauri::Url) -> bool {
    match url.host_str() {
        Some(h) => {
            h == "github.com"
                || h == "codeload.github.com"
                || h.ends_with(".github.com")
                || h.ends_with(".githubusercontent.com")
        }
        None => false,
    }
}

#[tauri::command]
pub async fn update_check(
    app: tauri::AppHandle,
    gh_proxy: Option<String>,
) -> Result<Option<UpdateInfo>, String> {
    let proxy = crate::download::sanitize_gh_prefix(gh_proxy);
    let mut endpoints: Vec<tauri::Url> = Vec::new();
    if let Some(p) = &proxy {
        // Mirror first, direct second — endpoints are tried in order until one answers 2XX.
        if let Ok(u) = tauri::Url::parse(&format!("{p}/{UPDATE_ENDPOINT}")) {
            endpoints.push(u);
        }
    }
    endpoints.push(tauri::Url::parse(UPDATE_ENDPOINT).map_err(|e| format!("UPDATE_CHECK_FAILED: {e}"))?);

    let updater = app
        .updater_builder()
        .endpoints(endpoints)
        .map_err(|e| format!("UPDATE_CHECK_FAILED: {e}"))?
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|e| format!("UPDATE_CHECK_FAILED: {e}"))?;

    match updater.check().await {
        Ok(Some(mut update)) => {
            // Rewrite the download through the mirror, KEEPING the direct URL as install-time
            // fallback (the check may well have succeeded via the direct endpoint because the
            // mirror is dead — an unconditional rewrite would then fail every install).
            let mut direct: Option<tauri::Url> = None;
            if let Some(p) = &proxy {
                if is_github_family(&update.download_url) {
                    if let Ok(u) = tauri::Url::parse(&format!("{p}/{}", update.download_url)) {
                        direct = Some(std::mem::replace(&mut update.download_url, u));
                    }
                }
            }
            tracing::info!(
                "update available: {} -> {} (download {})",
                update.current_version,
                update.version,
                update.download_url
            );
            let info = UpdateInfo {
                version: update.version.clone(),
                current_version: update.current_version.clone(),
                notes: update.body.clone(),
            };
            *PENDING_UPDATE.lock().unwrap() = Some((update, direct));
            Ok(Some(info))
        }
        Ok(None) => {
            *PENDING_UPDATE.lock().unwrap() = None;
            Ok(None)
        }
        Err(e) => Err(format!("UPDATE_CHECK_FAILED: {e}")),
    }
}

/// One download attempt with progress events + stall watchdog + cooperative cancel. Dropping the
/// plugin's download future (on stall/cancel) aborts the underlying transfer.
async fn run_download(app: &tauri::AppHandle, update: &tauri_plugin_updater::Update) -> Result<Vec<u8>, String> {
    let last_chunk = Arc::new(Mutex::new(Instant::now()));
    let progress_app = app.clone();
    let watchdog_clock = last_chunk.clone();
    let mut downloaded: u64 = 0;
    // First emit fires immediately (Instant set in the past) so the UI leaves "starting…" state fast.
    let mut last_emit = Instant::now() - Duration::from_secs(1);
    let dl = update.download(
        move |chunk, total| {
            *watchdog_clock.lock().unwrap() = Instant::now();
            downloaded += chunk as u64;
            if last_emit.elapsed() >= Duration::from_millis(200) {
                last_emit = Instant::now();
                let _ = progress_app.emit("update-progress", UpdateProgress { downloaded, total });
            }
        },
        || {},
    );
    tokio::pin!(dl);
    loop {
        tokio::select! {
            res = &mut dl => {
                return res.map_err(|e| format!("UPDATE_DOWNLOAD_FAILED: {e}"));
            }
            _ = tokio::time::sleep(Duration::from_secs(1)) => {
                if UPDATE_CANCEL.load(Ordering::SeqCst) {
                    return Err("UPDATE_DOWNLOAD_CANCELLED".to_string());
                }
                if last_chunk.lock().unwrap().elapsed() > DOWNLOAD_STALL {
                    return Err("UPDATE_DOWNLOAD_FAILED: stalled (no data for 60s)".to_string());
                }
            }
        }
    }
}

/// Download + install the update stashed by update_check. Emits "update-progress" (throttled) while
/// downloading and "update-installing" right before handing off to the installer. On Windows a
/// successful install EXITS THE PROCESS (NSIS /P /R relaunches the new version) — the frontend must
/// finish anything it wants persisted BEFORE invoking this.
#[tauri::command]
pub async fn update_install(
    app: tauri::AppHandle,
    state: State<'_, Arc<AppState>>,
) -> Result<(), String> {
    let (mut update, mut direct) = PENDING_UPDATE
        .lock()
        .unwrap()
        .take()
        .ok_or_else(|| "UPDATE_NO_PENDING".to_string())?;
    let _task = state.begin_task("update-download"); // close-flow in-progress listing
    UPDATE_CANCEL.store(false, Ordering::SeqCst);

    let bytes = match run_download(&app, &update).await {
        Ok(b) => b,
        Err(e) if e.contains("CANCELLED") => {
            *PENDING_UPDATE.lock().unwrap() = Some((update, direct));
            return Err(e);
        }
        Err(e) => {
            // Mirror route failed → ONE direct-route retry (the check's own fallback posture).
            match direct.take().filter(|d| *d != update.download_url) {
                Some(d) => {
                    tracing::warn!("update download failed via mirror ({e}); retrying direct: {d}");
                    update.download_url = d;
                    match run_download(&app, &update).await {
                        Ok(b) => b,
                        Err(e2) => {
                            // Restore with the direct URL in place — later retries stay direct.
                            *PENDING_UPDATE.lock().unwrap() = Some((update, None));
                            return Err(e2);
                        }
                    }
                }
                None => {
                    *PENDING_UPDATE.lock().unwrap() = Some((update, None));
                    return Err(e);
                }
            }
        }
    };

    // A cancel that raced the download's completion still wins — install is the point of no return.
    if UPDATE_CANCEL.load(Ordering::SeqCst) {
        *PENDING_UPDATE.lock().unwrap() = Some((update, direct));
        return Err("UPDATE_DOWNLOAD_CANCELLED".to_string());
    }

    let _ = app.emit("update-installing", ());
    // install() exits via std::process::exit — the window-state plugin's own exit-time save never
    // runs, so flush it here (same flags as the plugin registration; shared helper, NO-dup).
    let _ = tauri_plugin_window_state::AppHandleExt::save_window_state(&app, crate::window_state_flags());
    if let Err(e) = update.install(bytes) {
        *PENDING_UPDATE.lock().unwrap() = Some((update, direct));
        return Err(format!("UPDATE_INSTALL_FAILED: {e}"));
    }
    // Unreachable on Windows (install() exits the process); kept for non-Windows correctness.
    Ok(())
}

/// Cooperative cancel of the in-flight update download (the watchdog loop drops the transfer;
/// the stashed update handle is restored for a later retry).
#[tauri::command]
pub fn update_cancel() {
    UPDATE_CANCEL.store(true, Ordering::SeqCst);
}
