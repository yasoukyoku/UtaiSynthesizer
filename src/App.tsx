import { useEffect } from "react";
import { useAppStore } from "./store/app";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { invoke } from "@tauri-apps/api/core";
import i18n from "./i18n";
import { installHistory, routeUndo, routeRedo } from "./store/history";
import { newProjectFile, openProjectFile, saveProjectFile, saveProjectFileAs, restoreAutosave } from "./lib/project/projectFile";
import { installAutosave, clearAutosave, readAutosave, setRecoveryPending, hasUnsavedWork, rearmAutosave } from "./lib/project/autosave";
import { Titlebar } from "./components/common/Titlebar";
import { DawWorkflowSplit } from "./components/synth/DawWorkflowSplit";
import { TrainingPanel } from "./components/training/TrainingPanel";
import { MsstModelManager } from "./components/models/MsstModelManager";
import { LogViewer } from "./components/common/LogViewer";
import { Settings } from "./components/common/Settings";
import { useTrainingStore } from "./store/training";
import { ToastContainer } from "./components/common/Toast";
import { HistoryBanner } from "./components/common/HistoryBanner";
import { ConfirmDialog } from "./components/common/ConfirmDialog";
import { RenderLinkWatcher } from "./components/workflow/RenderLinkWatcher";
import { listen } from "@tauri-apps/api/event";
import { useWorkflowStore } from "./store/workflow";
import { useMsstModelStore } from "./store/msst-models";
import "./App.css";

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

/** The shared QUIT path (window-close "Quit" + tray "Quit"): warn about in-progress work, then unsaved
 *  changes, then exit the whole app. Any cancel/dismiss aborts the quit (the app keeps running). */
async function runExitFlow(): Promise<void> {
  if (useAppStore.getState().confirm) return; // a dialog is already open (recovery / another close) — don't stack
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
    await invoke("quit_app");
  } finally {
    setRecoveryPending(false); // re-enable autosave if we aborted (quit_app exits anyway)
    rearmAutosave(); // capture any edits made during the (aborted) exit dialogs, don't wait for next change
  }
}

export function App() {
  const { trainingPanelOpen, modelManagerOpen, toggleModelManager, logViewerOpen, toggleLogViewer, settingsOpen, toggleSettings } = useAppStore();
  const { fetchStatus } = useTrainingStore();

  useEffect(() => {
    const interval = setInterval(fetchStatus, 2000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  // Install the undo/redo auto-capture subscription (cleanup unsubscribes — HMR-safe).
  useEffect(() => installHistory(), []);

  // Autosave the document (debounced) for crash recovery — cleanup unsubscribes (HMR-safe).
  useEffect(() => installAutosave(), []);

  // Window close (X) → ask minimize-to-tray / quit / cancel. We ALWAYS preventDefault and decide
  // ourselves; Rust no longer guards close/exit, the frontend owns the whole flow. "Quit" runs the shared
  // exit flow (in-progress + unsaved prompts → quit_app); "minimize" HIDES the window into the tray.
  useEffect(() => {
    const win = getCurrentWindow();
    let unlisten: (() => void) | undefined;
    let disposed = false;
    void win
      .onCloseRequested(async (event) => {
        event.preventDefault();
        // Don't stack on top of an already-open dialog (the startup "Recover?" prompt, or an in-flight
        // close/exit) — settling it via a new showConfirm would clobber that decision. Let it resolve first.
        if (useAppStore.getState().confirm) return;
        const choice = await useAppStore.getState().showConfirm({
          title: i18n.t("close.title"),
          body: i18n.t("close.body"),
          buttons: [
            { id: "cancel", label: i18n.t("common.cancel") },
            { id: "quit", label: i18n.t("close.quit"), kind: "danger" },
            { id: "minimize", label: i18n.t("close.minimize"), kind: "primary" },
          ],
        });
        if (choice === "minimize") void win.hide();
        else if (choice === "quit") await runExitFlow();
        // cancel / dismiss → stay
      })
      .then((u) => {
        if (disposed) u();
        else unlisten = u;
        // Reveal the window only AFTER the close listener exists — the window starts hidden
        // (tauri.conf visible:false) so a click on the native X can never slip through before the
        // frontend owns the close flow (which would let Tauri silently destroy + quit).
        void getCurrentWindow().show();
      });
    return () => {
      disposed = true;
      unlisten?.();
    };
  }, []);

  // Tray "Quit" → Rust shows the window first (so the dialogs are visible), then emits this; run the
  // same shared exit flow. (Tray "Show" + tray icon click are handled entirely Rust-side.)
  useEffect(() => {
    let unlisten: (() => void) | undefined;
    let disposed = false;
    void listen("tray-quit", () => void runExitFlow()).then((u) => {
      if (disposed) u();
      else unlisten = u;
    });
    return () => {
      disposed = true;
      unlisten?.();
    };
  }, []);

  // Keep the tray menu labels in the UI language (the tray is built with English fallback labels).
  useEffect(() => {
    const update = () =>
      void invoke("set_tray_labels", { show: i18n.t("tray.show"), quit: i18n.t("tray.quit") }).catch(() => {});
    update();
    i18n.on("languageChanged", update);
    return () => i18n.off("languageChanged", update);
  }, []);

  // On startup, offer to recover an autosave left by an unclean exit (crash / kill / closing the process).
  useEffect(() => {
    void (async () => {
      const env = await readAutosave();
      if (!env) return;
      setRecoveryPending(true); // don't let an autosave write (e.g. an OS file-drop) clobber the slot
      const choice = await useAppStore.getState().showConfirm({
        title: i18n.t("project.recoverTitle"),
        body: i18n.t("project.recoverBody"),
        buttons: [
          { id: "discard", label: i18n.t("project.discard"), kind: "danger" },
          { id: "recover", label: i18n.t("project.recover"), kind: "primary" },
        ],
      });
      setRecoveryPending(false);
      if (choice === "recover") restoreAutosave(env);
      else await clearAutosave(); // discard OR dismiss — the single recovery slot can't be deferred
    })();
  }, []);

  useEffect(() => {
    const block = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && e.key === "C") {
        e.preventDefault();
      }
      // A lone Alt press activates the Windows window/system menu and steals keyboard focus from the
      // WebView (after which Space pops the restore/minimize/close menu and the app feels frozen).
      // We only use Alt as a wheel modifier (vertical zoom), so suppress its default menu activation.
      if (e.key === "Alt") {
        e.preventDefault();
      }
      // Undo / Redo. Skip while typing in a field (so Ctrl+Z reaches the input's own text undo).
      // Routes to the workflow editor's modal-local history when it's open, else the timeline.
      const el = e.target as HTMLElement | null;
      const editable =
        !!el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT" || el.isContentEditable);
      if (e.ctrlKey || e.metaKey) {
        const k = e.key.toLowerCase();
        if (k === "z" && !e.shiftKey) {
          if (editable) return; // let the focused input do its own text-undo
          e.preventDefault();
          routeUndo();
        } else if (k === "y" || (k === "z" && e.shiftKey)) {
          if (editable) return;
          e.preventDefault();
          routeRedo();
        } else if (k === "s") {
          // Save / Open / New are global app actions — fire regardless of focus. Blur first so an
          // in-progress inline rename / BPM edit commits before the save.
          e.preventDefault();
          (document.activeElement as HTMLElement | null)?.blur?.();
          if (e.shiftKey) void saveProjectFileAs();
          else void saveProjectFile();
        } else if (k === "o" && !e.shiftKey) {
          e.preventDefault();
          void openProjectFile();
        } else if (k === "n" && !e.shiftKey) {
          e.preventDefault();
          void newProjectFile();
        }
      }
    };
    document.addEventListener("keydown", block);
    return () => document.removeEventListener("keydown", block);
  }, []);

  return (
    <div className="app-shell">
      <Titlebar />
      <div className="app-content">
        <DawWorkflowSplit />
        {trainingPanelOpen && <TrainingPanel />}
        {logViewerOpen && <LogViewer onClose={toggleLogViewer} />}
        {settingsOpen && <Settings onClose={toggleSettings} />}
        {modelManagerOpen && <MsstModelManager onClose={toggleModelManager} />}
      </div>
      <ToastContainer />
      <HistoryBanner />
      <ConfirmDialog />
      <RenderLinkWatcher />
    </div>
  );
}
