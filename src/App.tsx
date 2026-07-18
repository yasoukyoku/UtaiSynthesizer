import { useEffect } from "react";
import { useAppStore } from "./store/app";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { invoke } from "@tauri-apps/api/core";
import i18n from "./i18n";
import { installHistory, routeUndo, routeRedo } from "./store/history";
import { newProjectFile, openProjectFile, saveProjectFile, saveProjectFileAs, restoreAutosave } from "./lib/project/projectFile";
import { installAutosave, clearAutosave, readAutosave, setRecoveryPending } from "./lib/project/autosave";
import { runExitFlow } from "./lib/exitFlow";
import { installOovWatch } from "./lib/vocal/oovWatch";
import { Titlebar } from "./components/common/Titlebar";
import { DawWorkflowSplit } from "./components/synth/DawWorkflowSplit";
import { TrainingPage } from "./components/training/TrainingPage";
import { MsstModelManager } from "./components/models/MsstModelManager";
import { LogViewer } from "./components/common/LogViewer";
import { Settings } from "./components/common/Settings";
import { setupTrainingListeners, useTrainingStore } from "./store/training";
import { ToastContainer } from "./components/common/Toast";
import { HistoryBanner } from "./components/common/HistoryBanner";
import { ConfirmDialog } from "./components/common/ConfirmDialog";
import { UpdateDialog } from "./components/common/UpdateDialog";
import { MissingModelsDialog } from "./components/common/MissingModelsDialog";
import { RenderLinkWatcher } from "./components/workflow/RenderLinkWatcher";
import { autoUpdateCheckEnabled, checkForUpdate } from "./lib/update";
import { runStartupComponentCheck, runBundledIntegrityCheck, startupComponentCheckEnabled } from "./lib/startupCheck";
import { backendErrorMessage } from "./lib/backendError";
import { listen } from "@tauri-apps/api/event";
import "./App.css";

export function App() {
  const { trainingPageOpen, modelManagerOpen, toggleModelManager, logViewerOpen, toggleLogViewer, settingsOpen, toggleSettings } = useAppStore();

  // Training is event-driven (no polling): install the global listeners once and
  // resync — an app reload during a run reattaches to the still-running Rust side.
  useEffect(() => {
    void setupTrainingListeners();
    void useTrainingStore.getState().refresh();
  }, []);

  // Install the undo/redo auto-capture subscription (cleanup unsubscribes — HMR-safe).
  useEffect(() => installHistory(), []);

  // Autosave the document (debounced) for crash recovery — cleanup unsubscribes (HMR-safe).
  useEffect(() => installAutosave(), []);

  // ② S58 OOV validation watcher (debounced Rust validate_lyrics → red notes / segment badge / track
  // warning) — cleanup unsubscribes (HMR-safe).
  useEffect(() => installOovWatch(), []);

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
        // S64: same update-busy discipline as runExitFlow (a confirm opened here would paint UNDER
        // the update overlay and be mouse-unreachable — the X would look dead).
        if (useAppStore.getState().updateBusy) {
          useAppStore.getState().showToast(i18n.t("update.quitBlocked"), "info");
          return;
        }
        if (useAppStore.getState().updateDialog) useAppStore.getState().closeUpdateDialog();
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
      if (choice === "recover") void restoreAutosave(env);
      else await clearAutosave(); // discard OR dismiss — the single recovery slot can't be deferred
    })();
  }, []);

  // S64 portability: the configured data dir was missing at startup and Rust recovered (recreated
  // empty / fell back to the default) — surface it, or the empty model library reads as data loss.
  // Details + the paths live persistently in Settings → Storage; the toast is the attention hook.
  useEffect(() => {
    void invoke<{ fell_back: boolean } | null>("get_data_dir_issue")
      .then((issue) => {
        if (issue) useAppStore.getState().showToast(i18n.t("startup.dataDirIssue"), "error");
      })
      .catch(() => {});
  }, []);

  // S64: startup update check (GitHub Releases; Settings-toggleable, default ON). Failure is a MODAL
  // by design — a toast is too easy to miss, and anyone annoyed can turn the auto-check off (user
  // decision). Collision guard: showConfirm auto-settles an already-open dialog as DISMISSED, and
  // dismissing the startup "Recover?" prompt DELETES the autosave slot — so wait until no confirm
  // dialog is open before surfacing anything, and fall back to a toast if one raced in anyway.
  useEffect(() => {
    if (!autoUpdateCheckEnabled()) return;
    let disposed = false;
    void (async () => {
      await new Promise((r) => setTimeout(r, 3000));
      for (let i = 0; i < 60 && !disposed && useAppStore.getState().confirm; i++) {
        await new Promise((r) => setTimeout(r, 1000));
      }
      if (disposed || useAppStore.getState().confirm) return;
      try {
        const info = await checkForUpdate();
        if (disposed || !info) return;
        // Same collision guard as the failure path (audit S64): a confirm may have opened DURING
        // the network check (e.g. the recovery prompt) — wait it out, and if one still holds,
        // downgrade to a toast rather than stacking a second modal.
        for (let i = 0; i < 60 && !disposed && useAppStore.getState().confirm; i++) {
          await new Promise((r) => setTimeout(r, 1000));
        }
        if (disposed) return;
        if (useAppStore.getState().confirm) {
          useAppStore.getState().showToast(`${i18n.t("update.title")} · v${info.version}`, "info");
          return;
        }
        useAppStore.getState().openUpdateDialog(info);
      } catch (e) {
        if (disposed) return;
        const detail = backendErrorMessage(e) ?? String(e);
        if (useAppStore.getState().confirm) {
          useAppStore.getState().showToast(`${i18n.t("update.checkFailedTitle")} — ${detail}`, "error");
          return;
        }
        void useAppStore.getState().showConfirm({
          title: i18n.t("update.checkFailedTitle"),
          body: `${detail}\n\n${i18n.t("update.checkFailedHint")}`,
          buttons: [{ id: "ok", label: i18n.t("common.ok"), kind: "primary" }],
        });
      }
    })();
    return () => {
      disposed = true;
    };
  }, []);

  // S66: startup missing-component check (converter runtime + core inference models →
  // one-click download). Runs AFTER the update-check window with the same collision
  // discipline: never stack on an open confirm OR the update dialog; the check itself
  // is silent when everything is installed (the common case).
  useEffect(() => {
    let disposed = false;
    void (async () => {
      await new Promise((r) => setTimeout(r, 6000));
      for (
        let i = 0;
        i < 120 && !disposed && (useAppStore.getState().confirm || useAppStore.getState().updateDialog);
        i++
      ) {
        await new Promise((r) => setTimeout(r, 1000));
      }
      if (disposed || useAppStore.getState().confirm || useAppStore.getState().updateDialog) return;
      // S68c: install-integrity first, and OUTSIDE the component-check master switch — that
      // switch is a permanent opt-out of the download nagging (compNever); integrity has its
      // own per-version mute and must re-arm after every update even for opted-out users.
      try {
        await runBundledIntegrityCheck();
      } catch {
        /* never block startup on this check */
      }
      if (disposed || !startupComponentCheckEnabled()) return;
      try {
        await runStartupComponentCheck();
      } catch {
        /* never block startup on this check */
      }
    })();
    return () => {
      disposed = true;
    };
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
      // Suppress browser-CHROME accelerator keys that leak through the WebView (Find / Find-next / Print /
      // View-source / Downloads / Reload / Zoom / Caret-browsing / history Back-Forward) — no place in a
      // desktop-app window. preventDefault ONLY (no stopPropagation), so the app's own keydown handlers
      // still receive the event and app shortcuts keep working; only the browser's default is cancelled.
      // DevTools (F12 / Ctrl+Shift+I) is deliberately LEFT alone so it stays usable in dev. Ctrl+S/O/N and
      // the editor's Ctrl+A/C/X/V/D + Ctrl+Z/Y are app-owned (handled elsewhere) and NOT blocked here.
      {
        const bk = e.key.toLowerCase();
        const mod = e.ctrlKey || e.metaKey;
        if (
          (mod && !e.altKey && ["f", "g", "p", "u", "j", "r"].includes(bk)) || // find/find-next/print/view-source/downloads/reload
          (mod && ["+", "-", "=", "0"].includes(e.key)) || // browser zoom
          e.key === "F3" || e.key === "F5" || e.key === "F7" || // find / reload / caret-browsing
          (e.altKey && (e.key === "ArrowLeft" || e.key === "ArrowRight")) // history back/forward
        ) {
          e.preventDefault();
        }
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
          // the full-screen training page covers the DAW — undo would silently
          // mutate the invisible timeline (Ctrl+S/O/N stay: explicit app actions)
          if (useAppStore.getState().trainingPageOpen) return;
          e.preventDefault();
          routeUndo();
        } else if (k === "y" || (k === "z" && e.shiftKey)) {
          if (editable) return;
          if (useAppStore.getState().trainingPageOpen) return;
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
    // Kill Chromium's middle-click autoscroll (the round anchor + drift) — a browser artifact in a native
    // DAW window. preventDefault on the button-1 mousedown suppresses it app-wide.
    const noMiddleAutoscroll = (e: MouseEvent) => { if (e.button === 1) e.preventDefault(); };
    document.addEventListener("keydown", block);
    document.addEventListener("mousedown", noMiddleAutoscroll);
    return () => {
      document.removeEventListener("keydown", block);
      document.removeEventListener("mousedown", noMiddleAutoscroll);
    };
  }, []);

  return (
    <div className="app-shell">
      <Titlebar />
      <div className="app-content">
        <DawWorkflowSplit />
        {trainingPageOpen && <TrainingPage />}
        {logViewerOpen && <LogViewer onClose={toggleLogViewer} />}
        {settingsOpen && <Settings onClose={toggleSettings} />}
        {modelManagerOpen && <MsstModelManager onClose={toggleModelManager} />}
      </div>
      <ToastContainer />
      <HistoryBanner />
      <ConfirmDialog />
      <UpdateDialog />
      <MissingModelsDialog />
      <RenderLinkWatcher />
    </div>
  );
}
