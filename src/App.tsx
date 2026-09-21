import { Suspense, lazy, useEffect, useState } from "react";
import { useAppStore } from "./store/app";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { invoke } from "@tauri-apps/api/core";
import { isTauri } from "./lib/tauri";
import i18n from "./i18n";
import { installHistory, routeUndo, routeRedo } from "./store/history";
import { newProjectFile, openProjectFile, saveProjectFile, saveProjectFileAs } from "./lib/project/projectFile";
import { installAutosave, readAutosave, setRecoveryPending } from "./lib/project/autosave";
import { runExitFlow } from "./lib/exitFlow";
import { installOovWatch } from "./lib/vocal/oovWatch";
import { ensureDictionarySig } from "./lib/vocal/vocalRender";
import { Titlebar } from "./components/common/Titlebar";
import { OnboardingTour } from "./components/common/OnboardingTour";
import { DawWorkflowSplit } from "./components/synth/DawWorkflowSplit";
import { SoundfontManager } from "./components/soundfont/SoundfontManager";
import { LogViewer } from "./components/common/LogViewer";
import { setupTrainingListeners, useTrainingStore } from "./store/training";
import { ToastContainer } from "./components/common/Toast";
import { HistoryBanner } from "./components/common/HistoryBanner";
import { ConfirmDialog } from "./components/common/ConfirmDialog";
import { UpdateDialog } from "./components/common/UpdateDialog";
import { SplashWizard } from "./components/common/SplashWizard";
import { LyricToVocalWizard } from "./components/common/LyricToVocalWizard";
import { MissingModelsDialog } from "./components/common/MissingModelsDialog";
import { RenderLinkWatcher } from "./components/workflow/RenderLinkWatcher";
import { useAmtStore } from "./store/amt";
import { AutoTuneWatcher } from "./components/synth/AutoTuneWatcher";
import { autoUpdateCheckEnabled, checkForUpdate } from "./lib/update";
import { runStartupComponentCheck, runBundledIntegrityCheck, startupComponentCheckEnabled } from "./lib/startupCheck";
import { backendErrorMessage } from "./lib/backendError";
import { listen } from "@tauri-apps/api/event";
import "./App.css";

// Heavy overlay surfaces (Settings 158KB / TrainingPage 155KB / MsstModelManager 150KB+ sources,
// each dragging its own CSS + deps) are code-split: they only load when first opened, cutting the
// initial bundle roughly in half. Fallback is null — the chunks are local and resolve in a frame.
const TrainingPage = lazy(() => import("./components/training/TrainingPage").then((m) => ({ default: m.TrainingPage })));
const MsstModelManager = lazy(() => import("./components/models/MsstModelManager").then((m) => ({ default: m.MsstModelManager })));
const Settings = lazy(() => import("./components/common/Settings").then((m) => ({ default: m.Settings })));
const MidiWorkbench = lazy(() => import("./components/workflow/nodes/MidiWorkbench").then((m) => ({ default: m.MidiWorkbench })));
const AmtConversionDialog = lazy(() => import("./components/models/AmtConversionDialog").then((m) => ({ default: m.AmtConversionDialog })));
const SongStudioDialog = lazy(() => import("./components/song/SongStudioDialog").then((m) => ({ default: m.SongStudioDialog })));

export function App() {
  // 启动欢迎页(Studio Pro 式):每次启动出现 — 新建/模板卡片 + 最近打开列表。
  // ESC/遮罩点击/任一动作后关闭;不再用 localStorage 首启标记(欢迎页即仪表盘)。
  const [splashOpen, setSplashOpen] = useState(true);
  const [lyricWizardOpen, setLyricWizardOpen] = useState(false);
  
  // Listen for lyric wizard request from Titlebar / other components
  useEffect(() => {
    const handler = () => setLyricWizardOpen(true);
    window.addEventListener("utai:open-lyric-wizard", handler);
    return () => window.removeEventListener("utai:open-lyric-wizard", handler);
  }, []);
  
  // 精确 selector + 稳定 action:App 是根组件,整 store 订阅会让任意 appStore 变化
  // (尤其滚动期间的 scrollX/scrollY 每帧更新)引发整棵组件树重渲染 —— 滚动/播放卡顿的
  // 主要根源之一。action 引用稳定,从 getState 取一次;开关状态走 selector。
  const { toggleModelManager, toggleSongStudio, toggleLogViewer, toggleSettings } = useAppStore.getState();
  const trainingPageOpen = useAppStore((s) => s.trainingPageOpen);
  const modelManagerOpen = useAppStore((s) => s.modelManagerOpen);
  const soundfontManagerOpen = useAppStore((s) => s.soundfontManagerOpen);
  const songStudioOpen = useAppStore((s) => s.songStudioOpen);
  const logViewerOpen = useAppStore((s) => s.logViewerOpen);
  const settingsOpen = useAppStore((s) => s.settingsOpen);

  // Apply the selected built-in skin by writing <html data-skin="…"> (CSS vars in theme.css cascade
  // from it). Defaults to the violet "junzi" skin; persisted via utai.skin.
  const skin = useAppStore((s) => s.skin);
  useEffect(() => {
    document.documentElement.dataset.skin = skin;
  }, [skin]);

  // ── 4.4 首次启动: 中文路径检测 + autosave 7天清理 ──
  // 首启快捷键弹窗由 Titlebar.tsx 里的 useEffect 单独负责 (那里能直接 setShortcutsOpen)
  useEffect(() => {
    // 中文路径检测 (内嵌 Python 训练环境限制)
    try {
      const installDir = window.location.pathname.replace(/^\//, "").split("/").slice(0, 4).join("\\");
      const hasNonAscii = /[^\x00-\x7F]/.test(installDir);
      if (hasNonAscii) {
        // 等 UI 初始化完再弹 (防 toast 容器还没 mount)
        setTimeout(() => {
          try {
            useAppStore.getState().showToast(
              "⚠️ 安装路径含中文/非 ASCII 字符，内嵌 Python 训练环境可能无法正常工作。建议迁移到纯英文路径后再使用训练功能。",
              "error"
            );
          } catch {
            console.warn("⚠️ Chinese path detected — Python training may not work");
          }
        }, 1500);
      }
    } catch { /* noop */ }

    // autosave.json 定期清理 (7 天)
    try {
      const lastCleanup = localStorage.getItem("utai.autosaveCleanup");
      const now = Date.now();
      if (!lastCleanup || now - parseInt(lastCleanup, 10) > 7 * 24 * 60 * 60 * 1000) {
        localStorage.setItem("utai.autosaveCleanup", String(now));
        import("./lib/project/autosave").then((m) => {
          try { m.clearAutosave?.(); } catch { /* noop */ }
        });
      }
    } catch { /* noop */ }
  }, []);

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
    // Skip window event hooks when not running in a Tauri WebView so the app shell still
    // mounts in a plain browser (dev preview / E2E / accidental open).
    let disposed = false;
    let unlisten: (() => void) | undefined;
    try {
      const win = getCurrentWindow();
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
          try { void getCurrentWindow().show(); } catch { /* browser */ }
        })
        .catch(() => { /* not a Tauri WebView — skip */ });
    } catch {
      /* getCurrentWindow() itself throws outside Tauri — skip all window hooks */
    }
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
    try {
      void listen("tray-quit", () => void runExitFlow())
        .then((u) => {
          if (disposed) u();
          else unlisten = u;
        })
        .catch(() => { /* not a Tauri WebView — skip tray event */ });
    } catch {
      /* listen() itself throws outside Tauri */
    }
    return () => {
      disposed = true;
      unlisten?.();
    };
  }, []);

  // S74b: a saved explicit inference device that this machine can no longer honour was demoted to
  // Auto during startup (GPU swapped, machine changed, or our supported window narrowed in an
  // update). Non-blocking — the user's stored intent was invalidated by the environment, not
  // refused. Asked once here rather than pushed from Rust: the backend decides it during setup,
  // before any frontend listener exists, so an event would be lost.
  useEffect(() => {
    if (!isTauri()) return;
    let cancelled = false;
    invoke<{ preference_demoted?: boolean }>("get_hardware_info")
      .then((h) => {
        if (!cancelled && h.preference_demoted) {
          useAppStore.getState().showToast(i18n.t("common.devicePreferenceDemoted"), "info");
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  // S158: on startup, surface which GPU inference backend is actually active (CUDA / DirectML) — a
  // single non-blocking toast so the "自动选择推理后端" result is visible and the user isn't left
  // guessing whether the accelerator is working. CPU / unknown fallback builds stay silent (nothing
  // to advertise, and a CPU toast on every launch would just be noise).
  useEffect(() => {
    if (!isTauri()) return;
    let cancelled = false;
    invoke<{ ort_build?: string }>("get_hardware_info")
      .then((h) => {
        if (cancelled) return;
        const build = h.ort_build ?? "";
        // Exact match against the four literals init_ort_runtime can set — substring matching
        // would misreport a dev/system path that happens to contain "CUDA"/"DirectML" (audit L4).
        const backend =
          build === "CUDA"
            ? "CUDA"
            : build === "DirectML"
              ? "DirectML"
              : null;
        if (backend) {
          useAppStore.getState().showToast(i18n.t("common.gpuAccelStartup", { backend }), "info");
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  // S74: the inference engine transparently fell back CUDA → DirectML after a CUDA run failure
  // (an in-window card that still can't run CUDA, or a leaked/old install) — inform the user
  // (non-blocking; the failure itself is WARN-logged backend-side for debugging).
  useEffect(() => {
    if (!isTauri()) return;
    let unlisten: (() => void) | undefined;
    let disposed = false;
    void listen("auto-cuda-fallback", () => {
      useAppStore.getState().showToast(i18n.t("common.autoCudaFallback"), "info");
    }).then((u) => {
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
    if (!isTauri()) return;
    const update = () =>
      void invoke("set_tray_labels", { show: i18n.t("tray.show"), quit: i18n.t("tray.quit") }).catch(() => {});
    update();
    i18n.on("languageChanged", update);
    return () => i18n.off("languageChanged", update);
  }, []);

  // ── 崩溃恢复:启动欢迎页自己处理(读 autosave → 恢复卡片),这里不再弹独立 ConfirmDialog ──
  useEffect(() => {
    void (async () => {
      const env = await readAutosave();
      if (!env) return;
      setRecoveryPending(true); // 冻结 autosave 写入直到用户在欢迎页做出选择
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

  // S101: pull the dictionary-content fingerprint into the bake signature as early as possible.
  // Rust has already refreshed the data root's dictionaries synchronously in setup(), so this reads
  // a settled value. `ensureDictionarySig` is idempotent and never rejects; until it resolves,
  // isVocalDirty deliberately reports "not dirty" rather than compare against a placeholder.
  useEffect(() => {
    void ensureDictionarySig();
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
      <Titlebar splashLocked={splashOpen} />
      <div className="app-content">
        {!splashOpen && (
          <>
            <DawWorkflowSplit />
            {trainingPageOpen && (
              <Suspense fallback={null}>
                <TrainingPage />
              </Suspense>
            )}
            {logViewerOpen && <LogViewer onClose={toggleLogViewer} />}
            {settingsOpen && (
              <Suspense fallback={null}>
                <Settings onClose={toggleSettings} />
              </Suspense>
            )}
            {modelManagerOpen && (
              <Suspense fallback={null}>
                <MsstModelManager onClose={toggleModelManager} />
              </Suspense>
            )}
            {soundfontManagerOpen && <SoundfontManager />}
            {songStudioOpen && <SongStudioDialog onClose={toggleSongStudio} />}
          </>
        )}
      </div>
      <ToastContainer />
      {!splashOpen && <HistoryBanner />}
      <ConfirmDialog />
      <UpdateDialog />
      <MissingModelsDialog />
      {!splashOpen && <OnboardingTour />}
      {splashOpen && <SplashWizard onClose={() => setSplashOpen(false)} />}
      {!splashOpen && <>
        {lyricWizardOpen && <LyricToVocalWizard onClose={() => setLyricWizardOpen(false)} />}
        <RenderLinkWatcher />
        <AutoTuneWatcher />
        <AmtWorkbenchHost />
        <AmtConversionHost />
      </>}
    </div>
  );
}

/** Renders any open AMT MIDI workbench modals (one per node id). */
function AmtWorkbenchHost() {
  const workbenchOpen = useAmtStore((s) => s.workbenchOpen);
  const closeWorkbench = useAmtStore((s) => s.closeWorkbench);
  const openIds = Object.keys(workbenchOpen).filter((id) => workbenchOpen[id]);
  return (
    <>
      {openIds.map((id) => (
        <Suspense key={id} fallback={null}>
          <MidiWorkbench nodeId={id} onClose={() => closeWorkbench(id)} />
        </Suspense>
      ))}
    </>
  );
}

/** Renders the AMT (Audio-to-MIDI) conversion dialog if active. */
function AmtConversionHost() {
  const trackId = useAppStore((s) => s.amtConversionTrackId);
  const close = useAppStore((s) => s.closeAmtConversion);
  if (!trackId) return null;
  return (
    <Suspense fallback={null}>
      <AmtConversionDialog trackId={trackId} onClose={close} />
    </Suspense>
  );
}


