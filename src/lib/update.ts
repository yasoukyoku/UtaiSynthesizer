// S64 — In-app update flow, frontend half. THE single source for the update commands, the
// auto-check setting and the progress event names; Settings, App (startup check) and UpdateDialog all
// import from here. The Rust half is src-tauri/src/commands/update.rs (tauri-plugin-updater driven
// from Rust so the endpoint order can follow the user's GH-mirror setting).
import { invoke } from "@tauri-apps/api/core";
import { loadSetting, saveSetting } from "./settings";
import { ghRouteOrder } from "./models/msst-catalog";
import { useMsstModelStore } from "../store/msst-models";
import { flushAutosaveNow } from "./project/autosave";

export interface UpdateInfo {
  version: string;
  currentVersion: string;
  notes: string | null;
}

export interface UpdateProgress {
  downloaded: number;
  total: number | null;
}

/** Window events emitted by update_install (Rust). */
export const UPDATE_PROGRESS_EVENT = "update-progress";
export const UPDATE_INSTALLING_EVENT = "update-installing";

const AUTO_CHECK_KEY = "utai.autoUpdateCheck";

export function autoUpdateCheckEnabled(): boolean {
  return loadSetting(AUTO_CHECK_KEY, true);
}

export function setAutoUpdateCheckEnabled(v: boolean): void {
  saveSetting(AUTO_CHECK_KEY, v);
}

/** Ask GitHub Releases for a newer version. Returns null when up to date; throws
 *  "UPDATE_CHECK_FAILED: …" (backendError-mapped) on network/endpoint failure. The user's full
 *  GH route order (chosen proxy → direct → other presets, S66) is passed through so both the
 *  check and the later download walk the whole failover chain. */
export async function checkForUpdate(): Promise<UpdateInfo | null> {
  const { ghMirror, ghPresets } = useMsstModelStore.getState();
  return await invoke<UpdateInfo | null>("update_check", { ghRoutes: ghRouteOrder(ghMirror, ghPresets) });
}

/** Download + install the update found by checkForUpdate. On success the process EXITS (the NSIS
 *  updater relaunches the new version), so this only ever returns by throwing. An autosave snapshot is
 *  flushed first — unsaved work comes back through the normal crash-recovery prompt after the restart. */
export async function installUpdate(): Promise<void> {
  await flushAutosaveNow();
  await invoke("update_install");
}
