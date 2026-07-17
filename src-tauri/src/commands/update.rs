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
// GH mirror (audit-hardened S64; generalized to an ordered ROUTE LIST S66):
//   - the frontend sends ghRouteOrder(): proxy prefixes with an "" marker at the DIRECT position
//     (chosen proxy before direct, remaining presets after) — public proxies die without notice,
//     so every consumer gets the FULL failover chain, not one mirror + one retry.
//   - every prefix passes download::sanitize_gh_prefix — **https only**, because the plugin's
//     endpoint validation hard-rejects http in RELEASE builds (warn-only in dev: the one shape of
//     breakage `tauri dev` can never reproduce). An invalid prefix degrades to the direct route.
//   - check endpoints = the same ordered chain (2XX wins); the announced download_url (a github.com
//     release asset) becomes candidate[0] with the REST kept as install-time fallbacks — a dead
//     mirror must never make an update uninstallable while any route works. minisign verifies file
//     CONTENT, so any source is tamper-safe.
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

/// Update handle between check and install, plus the REMAINING download-route candidates
/// (ordered; consumed head-first when a route fails). install() takes the pair out and
/// RESTORES it on failure so a retry needs no re-check.
static PENDING_UPDATE: Mutex<Option<(tauri_plugin_updater::Update, Vec<tauri::Url>)>> =
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

/// Expand an ordered route list ("" = direct, else proxy prefix) over `base`, sanitizing each
/// prefix and deduping — the SINGLE builder for both the check endpoints and the download
/// candidates, so their orders can never drift.
fn expand_routes(routes: &Option<Vec<String>>, base: &str) -> Vec<tauri::Url> {
    let mut out: Vec<tauri::Url> = Vec::new();
    let mut push = |u: Option<tauri::Url>| {
        if let Some(u) = u {
            if !out.contains(&u) {
                out.push(u);
            }
        }
    };
    let direct = tauri::Url::parse(base).ok();
    let mut had_direct = false;
    if let Some(rs) = routes {
        for r in rs {
            if r.is_empty() {
                had_direct = true;
                push(direct.clone());
            } else if let Some(p) = crate::download::sanitize_gh_prefix(Some(r.clone())) {
                push(tauri::Url::parse(&format!("{p}/{base}")).ok());
            }
        }
    }
    if !had_direct {
        push(direct);
    }
    out
}

#[tauri::command]
pub async fn update_check(
    app: tauri::AppHandle,
    gh_routes: Option<Vec<String>>,
) -> Result<Option<UpdateInfo>, String> {
    let endpoints = expand_routes(&gh_routes, UPDATE_ENDPOINT);
    if endpoints.is_empty() {
        return Err("UPDATE_CHECK_FAILED: no valid endpoint".into());
    }

    // ONE endpoint per check() call (review S66): the plugin's own endpoint list only falls
    // through on transport/non-2XX errors — a poisoned proxy answering 200 + an HTML page
    // (ghproxy.site, live-verified) parse-fails and ABORTS the whole check, masking every
    // later (healthy) route. Looping ourselves restores "first WORKING route wins".
    let mut check_result: std::result::Result<Option<tauri_plugin_updater::Update>, String> =
        Err("UPDATE_CHECK_FAILED: no endpoint reachable".into());
    for ep in endpoints {
        let updater = app
            .updater_builder()
            .endpoints(vec![ep.clone()])
            .map_err(|e| format!("UPDATE_CHECK_FAILED: {e}"))?
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| format!("UPDATE_CHECK_FAILED: {e}"))?;
        match updater.check().await {
            Ok(r) => {
                check_result = Ok(r);
                break;
            }
            Err(e) => {
                tracing::warn!("update check failed via {ep}: {e}");
                check_result = Err(format!("UPDATE_CHECK_FAILED: {e}"));
            }
        }
    }

    match check_result {
        Ok(Some(mut update)) => {
            // Expand the announced download_url over the SAME ordered route chain: candidate[0]
            // becomes the active URL, the rest are install-time fallbacks (a dead mirror must
            // never make an update uninstallable while any route works).
            let mut fallbacks: Vec<tauri::Url> = Vec::new();
            if is_github_family(&update.download_url) {
                let mut candidates = expand_routes(&gh_routes, update.download_url.as_str());
                if !candidates.is_empty() {
                    update.download_url = candidates.remove(0);
                    fallbacks = candidates;
                }
            }
            tracing::info!(
                "update available: {} -> {} (download {}, {} fallback route(s))",
                update.current_version,
                update.version,
                update.download_url,
                fallbacks.len()
            );
            let info = UpdateInfo {
                version: update.version.clone(),
                current_version: update.current_version.clone(),
                notes: update.body.clone(),
            };
            *PENDING_UPDATE.lock().unwrap() = Some((update, fallbacks));
            Ok(Some(info))
        }
        Ok(None) => {
            *PENDING_UPDATE.lock().unwrap() = None;
            Ok(None)
        }
        Err(e) => Err(e), // already UPDATE_CHECK_FAILED-prefixed by the loop above
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
    let (mut update, mut fallbacks) = PENDING_UPDATE
        .lock()
        .unwrap()
        .take()
        .ok_or_else(|| "UPDATE_NO_PENDING".to_string())?;
    let _task = state.begin_task("update-download"); // close-flow in-progress listing
    UPDATE_CANCEL.store(false, Ordering::SeqCst);

    // Snapshot the FULL chain up front: when every route fails, restore the whole thing —
    // pinning later retries to the last (deliberately least-preferred) route would be a
    // regression vs the old direct-restore behavior (review S66).
    let orig_url = update.download_url.clone();
    let orig_fallbacks = fallbacks.clone();

    // Walk the route chain head-first: a failed route is consumed (later retries resume from
    // the next one), a cancel restores everything untouched.
    let bytes = loop {
        match run_download(&app, &update).await {
            Ok(b) => break b,
            Err(e) if e.contains("CANCELLED") => {
                *PENDING_UPDATE.lock().unwrap() = Some((update, fallbacks));
                return Err(e);
            }
            Err(e) => {
                if fallbacks.is_empty() {
                    update.download_url = orig_url;
                    *PENDING_UPDATE.lock().unwrap() = Some((update, orig_fallbacks));
                    return Err(e);
                }
                let next = fallbacks.remove(0);
                tracing::warn!(
                    "update download failed via {} ({e}); trying next route: {next}",
                    update.download_url
                );
                update.download_url = next;
            }
        }
    };

    // A cancel that raced the download's completion still wins — install is the point of no return.
    if UPDATE_CANCEL.load(Ordering::SeqCst) {
        *PENDING_UPDATE.lock().unwrap() = Some((update, fallbacks));
        return Err("UPDATE_DOWNLOAD_CANCELLED".to_string());
    }

    let _ = app.emit("update-installing", ());
    // install() exits via std::process::exit — the window-state plugin's own exit-time save never
    // runs, so flush it here (same flags as the plugin registration; shared helper, NO-dup).
    let _ = tauri_plugin_window_state::AppHandleExt::save_window_state(&app, crate::window_state_flags());
    // Deliberate exit ahead (S68b): drop the unclean-exit sentinel so the post-update
    // start doesn't run a crash autopsy. Sentinel only — the log worker must stay
    // alive in case install() fails and the session continues.
    crate::crashlog::remove_sentinel();
    if let Err(e) = update.install(bytes) {
        crate::crashlog::restore_sentinel();
        *PENDING_UPDATE.lock().unwrap() = Some((update, fallbacks));
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
