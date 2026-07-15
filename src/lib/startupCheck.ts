// S66 — startup missing-component check. The post-download design ships a small installer and
// fetches the big pieces later; before this, a fresh install only learned about the missing
// converter runtime / core inference models when something FAILED (RUNTIME_PACK_REQUIRED /
// AUX_FILE_MISSING toasts). This check turns that into one friendly dialog with a one-click
// download at first launch. App.tsx runs it AFTER the update-check window with the same
// confirm-collision discipline; the toggle lives in Settings → Model Assets.
import { invoke } from "@tauri-apps/api/core";
import i18n from "../i18n";
import { loadSetting, saveSetting } from "./settings";
import { hfBaseForMirror } from "./models/msst-catalog";
import { useAppStore } from "../store/app";
import { useMsstModelStore } from "../store/msst-models";

const KEY = "utai.startupComponentCheck";

export function startupComponentCheckEnabled(): boolean {
  return loadSetting(KEY, true);
}

export function setStartupComponentCheckEnabled(v: boolean): void {
  saveSetting(KEY, v);
}

interface AssetPackStatus {
  id: string;
  missing: number;
  downloading: boolean;
}

interface CatalogItem {
  id: string;
  variant: string;
  installed: boolean;
  supported: boolean;
  downloadable: boolean;
  experimental: boolean;
}

/** Detect the missing must-have components (converter runtime pack + core inference models)
 *  and offer the one-click download. The caller guarantees no other modal is open. */
export async function runStartupComponentCheck(): Promise<void> {
  const [convOk, packs, hw] = await Promise.all([
    invoke<boolean>("converter_env_ready"),
    invoke<AssetPackStatus[]>("asset_pack_status"),
    invoke<{ recommended_variant?: string }>("get_hardware_info").catch(
      () => ({}) as { recommended_variant?: string },
    ),
  ]);
  const aux = packs.find((p) => p.id === "aux-inference");
  const auxMissing = (aux?.missing ?? 0) > 0 && !(aux?.downloading ?? false);
  if (convOk && !auxMissing) return;

  const lines: string[] = [];
  if (!convOk) lines.push(`· ${i18n.t("startup.compRuntime")}`);
  if (auxMissing) lines.push(`· ${i18n.t("startup.compAux")}`);
  const c = await useAppStore.getState().showConfirm({
    title: i18n.t("startup.compTitle"),
    body: `${i18n.t("startup.compBody")}\n${lines.join("\n")}\n\n${i18n.t("startup.compHint")}`,
    buttons: [
      { id: "never", label: i18n.t("startup.compNever") },
      { id: "later", label: i18n.t("startup.compLater") },
      { id: "dl", label: i18n.t("startup.compDl"), kind: "primary" },
    ],
  });
  if (c === "never") {
    setStartupComponentCheckEnabled(false);
    return;
  }
  if (c !== "dl") return;

  // Open Settings so the existing progress UIs (Model Assets / Training Runtime sections)
  // show the downloads the click just started — no new progress surface to maintain.
  if (!useAppStore.getState().settingsOpen) useAppStore.getState().toggleSettings();
  const hfBase = hfBaseForMirror(useMsstModelStore.getState().mirror);
  void (async () => {
    if (auxMissing) {
      try {
        await invoke("download_asset_pack", { id: "aux-inference", hfBase });
      } catch {
        /* busy/cancel/fail all surface in the Settings asset section */
      }
    }
    if (!convOk) {
      try {
        // The runtime pack matching this machine: hardware-recommended variant first, then any
        // supported stable pack, then any supported one (mirrors the Settings list's gating).
        const env = await invoke<{ catalog: CatalogItem[] }>("get_runtime_env_info");
        const rec = hw.recommended_variant;
        const candidates = env.catalog.filter((e) => !e.installed && e.supported && e.downloadable);
        const pick =
          candidates.find((e) => e.variant === rec) ??
          candidates.find((e) => !e.experimental) ??
          candidates[0];
        if (pick) await invoke("download_runtime_pack", { id: pick.id });
      } catch {
        /* surfaced in the Settings runtime section */
      }
    }
  })();
}
