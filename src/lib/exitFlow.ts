import { invoke } from "@tauri-apps/api/core";
import i18n from "../i18n";
import { useAppStore } from "../store/app";
import { useWorkflowStore } from "../store/workflow";
import { useMsstModelStore } from "../store/msst-models";
import { saveProjectFile } from "./project/projectFile";
import { clearAutosave, hasUnsavedWork, rearmAutosave, setRecoveryPending } from "./project/autosave";

/** The long tasks currently running, as localized labels to LIST in the quit warning (so the user sees
 *  exactly what's running and can decide). Workflow node executions (which cover inference / separation /
 *  effects run through the graph) + MSST downloads are frontend-visible; training + standalone separation
 *  come from Rust `running_tasks` (stable ids → `close.task_<id>`). Extend in BOTH places as tasks grow. */
async function getRunningTasks(): Promise<string[]> {
  const tasks: string[] = [];
  let rustTasks: string[] = [];
  try {
    rustTasks = await invoke<string[]>("running_tasks"); // training / separation / cuda_download / convert
  } catch {
    /* best-effort guard */
  }
  // Workflow node executions cover inference/effects; but a SEPARATION node ALSO surfaces as the Rust
  // "separation" task, so only add the generic "workflow" label when the running node isn't already a more
  // specific Rust task — otherwise the same separation lists twice ("workflow" + "audio separation").
  const wfRunning = Object.values(useWorkflowStore.getState().executions).some((e) => e.status === "running");
  if (wfRunning && !rustTasks.includes("separation")) tasks.push(i18n.t("close.task_workflow"));
  for (const id of rustTasks) tasks.push(i18n.t(`close.task_${id}`));
  if (Object.keys(useMsstModelStore.getState().downloading).length > 0) tasks.push(i18n.t("close.task_download"));
  return tasks;
}

/** The shared QUIT path (window-close "Quit" + tray "Quit" + the migration dialog's "restart now"):
 *  warn about in-progress work, then unsaved changes, then exit the whole app. Any cancel/dismiss
 *  aborts (the app keeps running). `mode: "restart"` runs the SAME gates but relaunches via
 *  `restart_app` instead of exiting — a restart is a quit with a relaunch, so it must never skip
 *  the busy/unsaved confirmations. */
export async function runExitFlow(mode: "quit" | "restart" = "quit"): Promise<void> {
  if (useAppStore.getState().confirm) return; // a dialog is already open (recovery / another close) — don't stack
  // S64: a busy update (downloading/installing) must not be silently abandoned; an IDLE update
  // dialog just closes and the quit proceeds (the user can update after the next launch).
  if (useAppStore.getState().updateBusy) {
    useAppStore.getState().showToast(i18n.t("update.quitBlocked"), "info");
    return;
  }
  if (useAppStore.getState().updateDialog) useAppStore.getState().closeUpdateDialog();
  setRecoveryPending(true); // freeze autosave while deciding (no mid-dialog write/clear race)
  try {
    const running = await getRunningTasks();
    if (running.length > 0) {
      const body = `${i18n.t("close.busyIntro")}\n\n${running.map((t) => `• ${t}`).join("\n")}\n\n${i18n.t("close.busyOutro")}`;
      const c = await useAppStore.getState().showConfirm({
        title: i18n.t("close.busyTitle"),
        body,
        buttons: [
          { id: "cancel", label: i18n.t("common.cancel") },
          { id: "quit", label: i18n.t("close.quitAnyway"), kind: "danger" },
        ],
      });
      if (c !== "quit") return;
    }
    if (hasUnsavedWork()) {
      const c = await useAppStore.getState().showConfirm({
        title: i18n.t("project.discardTitle"),
        body: i18n.t("project.closeBody"),
        buttons: [
          { id: "cancel", label: i18n.t("common.cancel") },
          { id: "dontSave", label: i18n.t("project.dontSave"), kind: "danger" },
          { id: "save", label: i18n.t("menu.save"), kind: "primary" },
        ],
      });
      if (c === "save") {
        const ok = await saveProjectFile();
        if (!ok) return; // save cancelled/failed → abort quit
      } else if (c !== "dontSave") {
        return; // cancel / dismiss → abort quit
      }
    }
    await clearAutosave(); // exiting cleanly → drop the recovery file
    await invoke(mode === "restart" ? "restart_app" : "quit_app");
  } finally {
    setRecoveryPending(false); // re-enable autosave if we aborted (quit_app/restart_app exit anyway)
    rearmAutosave(); // capture any edits made during the (aborted) exit dialogs, don't wait for next change
  }
}
