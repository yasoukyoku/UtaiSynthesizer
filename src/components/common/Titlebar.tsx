import { useEffect, useState } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { getVersion } from "@tauri-apps/api/app";
import { open } from "@tauri-apps/plugin-dialog";
import { projectExportItems, anyProjectExportable, exportProjectTracksToFolder, laneExportErrorMessage } from "../../lib/audio/exportLaneAudio";
import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import { routeUndo, routeRedo, routeCanUndo, routeCanRedo } from "../../store/history";
import { useTrainingStore } from "../../store/training";
import { isRunningState } from "../../lib/training/liveRun";
import { useTranslation } from "react-i18next";
import { ContextMenu, type MenuItem } from "./ContextMenu";
import { ShortcutsDialog } from "./ShortcutsDialog";
import { UserGuideDialog } from "./UserGuideDialog";
import { HistoryPanel } from "./HistoryPanel";
import { newProjectFile, openProjectFile, saveProjectFile, saveProjectFileAs, loadDemoProject } from "../../lib/project/projectFile";
import { importScoreFile } from "../../lib/vocal/import";
import { scoreExportableTracks } from "../../lib/vocal/exportScore";
import { copySelectedSegments, cutSelectedSegments, pasteWithFeedback, clipboardKind } from "../../lib/clipboard";
import { ExportAudioDialog } from "./ExportAudioDialog";
import { ExportScoreDialog } from "./ExportScoreDialog";
import { SuperOriginalWizard } from "./SuperOriginalWizard";
// Muno 阶段4:AI 菜单 —— 自动编曲 / 识别和弦 / 识别鼓点 / 人声转MIDI 的统一入口。
import { analyzeTrackChords, detectTrackDrums } from "../../lib/analysis/trackAnalysisActions";
import { resolveMelodyTrackId } from "../../lib/arrangement/autoArrange";
import { resolveTrackSourceAudio } from "../../lib/amtSource";
import { ArrangeDialog } from "../synth/ArrangeDialog";
// Muno 阶段5:和弦 MIDI(配和声)弹窗 —— AI 菜单「生成和弦 MIDI」的 UI。
import { ChordMidiDialog } from "../synth/ChordMidiDialog";
import "./Titlebar.css";

export function Titlebar({ splashLocked }: { splashLocked?: boolean } = {}) { const { t } = useTranslation();
  // 精确 selector:整 store 订阅会让 Titlebar 在播放期间随 playheadTick 每帧重渲染
  // (projectStore),滚动期间随 scrollX/scrollY 每帧重渲染 (appStore)。action 是稳定引用,
  // 从 getState 取一次即可,不参与订阅。
  const name = useProjectStore((s) => s.name);
  const dirty = useProjectStore((s) => s.dirty);
  const trackCount = useProjectStore((s) => s.tracks.length);
  const {
    toggleTrainingPage, toggleModelManager, toggleSoundfontManager, toggleSongStudio,
    toggleLogViewer, toggleSettings,
  } = useAppStore.getState();
  const settingsOpen = useAppStore((s) => s.settingsOpen);
  const trainingState = useTrainingStore((s) => s.snapshot.state);
  // The Edit menu's enablement is read via routeCanUndo/routeCanRedo when the menu opens (opening it
  // sets editMenu → re-render), so it reflects whichever stack is active (the workflow editor's
  // modal-local stack while open, else the timeline) without a live subscription.
  const [editMenu, setEditMenu] = useState<{ x: number; y: number } | null>(null);
  const [fileMenu, setFileMenu] = useState<{ x: number; y: number } | null>(null);
  const [helpMenu, setHelpMenu] = useState<{ x: number; y: number } | null>(null);
  // Muno 阶段4:AI 菜单(下拉)+ 自动编曲弹窗的目标轨(打开时锁定)。
  const [aiMenu, setAiMenu] = useState<{ x: number; y: number } | null>(null);
  const [aiArrangeTarget, setAiArrangeTarget] = useState<string | null>(null);
  // Muno 阶段5:和弦 MIDI(配和声)弹窗的目标轨。
  const [aiChordTarget, setAiChordTarget] = useState<string | null>(null);
  const [toolsMenu, setToolsMenu] = useState<{ x: number; y: number } | null>(null);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const [userGuideOpen, setUserGuideOpen] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [exportAudioOpen, setExportAudioOpen] = useState(false);
  const [exportScoreOpen, setExportScoreOpen] = useState(false);
    const [wizardOpen, setWizardOpen] = useState(false);
  const [appVersion, setAppVersion] = useState("");
  useEffect(() => { void getVersion().then(setAppVersion).catch(() => {}); }, []);

  // Ctrl+Shift+A → 打开智能编曲面板(解析活跃/选中/第一条音符轨),与时间线快捷键互补。
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (e.ctrlKey && e.shiftKey && e.key.toLowerCase() === "a") {
        e.preventDefault();
        const tid = resolveMelodyTrackId();
        if (tid) setAiArrangeTarget(tid);
        else useAppStore.getState().showToast("先选一条旋律/音符轨(或先把音频 AI 转谱)", "info");
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  // 底部栏等外部入口通过事件打开超级原创向导(与 utai:open-lyric-wizard 同模式)
  useEffect(() => {
    const handler = () => setWizardOpen(true);
    window.addEventListener("utai:open-super-wizard", handler);
    return () => window.removeEventListener("utai:open-super-wizard", handler);
  }, []);

  // ── 首次启动: 2s 后自动弹快捷键帮助 ──
  useEffect(() => {
    const shown = localStorage.getItem("utai.firstRunShortcutsShown");
    if (!shown) {
      localStorage.setItem("utai.firstRunShortcutsShown", "true");
      const t = setTimeout(() => setShortcutsOpen(true), 2000);
      return () => clearTimeout(t);
    }
  }, []);

  // 帮助/社区：所有地址链接已删除，只保留版本行 + 纯文本「QQ：202112」。
  const helpItems: MenuItem[] = [
    { label: t("help.userGuide"), icon: "📖", onClick: () => setUserGuideOpen(true) },
    { label: `Muno ${appVersion ? `v${appVersion}` : ""}`.trim(), disabled: true, onClick: () => {} },
    { label: t("help.qq"), disabled: true, onClick: () => {} },
  ];

  const isTraining = isRunningState(trainingState);

  // File→轨道另存为:把工程里所有轨道(导入音频 / 分离 stems / 克隆 bake)一起复制进一个文件夹。
  const showToast = useAppStore((s) => s.showToast);
  const exportAllTracks = async () => {
    const tracks = useProjectStore.getState().tracks;
    const items = projectExportItems(tracks);
    if (items.length === 0) {
      showToast(t("tracks.exportNothing"), "info");
      return;
    }
    const out = await open({ directory: true, title: t("menu.exportTracks") });
    if (!out || typeof out !== "string") return;
    try {
      const dests = await exportProjectTracksToFolder(tracks, out);
      showToast(`${t("tracks.exportDone")} ${dests.length} ${t("tracks.exportCopied")} ${out}`, "success");
    } catch (e) {
      showToast(laneExportErrorMessage(e), "error");
    }
  };

  const fileItems: MenuItem[] = [
    { label: t("menu.new"), icon: "🆕", shortcut: "Ctrl+N", onClick: () => void newProjectFile() },
    // 一键示例工程：新手模板（8 小节四轨），立即能看/能播/能导出。
    { label: t("menu.demo"), icon: "🎁", onClick: () => void loadDemoProject() },
    { label: t("menu.open"), icon: "📂", shortcut: "Ctrl+O", onClick: () => void openProjectFile() },
    { label: t("menu.save"), icon: "💾", shortcut: "Ctrl+S", disabled: trackCount === 0, onClick: () => void saveProjectFile() },
    { label: t("menu.saveAs"), icon: "📝", shortcut: "Ctrl+Shift+S", disabled: trackCount === 0, onClick: () => void saveProjectFileAs() },
    { label: t("menu.import"), icon: "🎼", onClick: () => void importScoreFile() },
    // S63 export entries. Enablement is read lazily on menu open (the same pattern as the clipboard
    // items below): audio needs any track at all, score needs ≥1 vocal track with notes — via THE
    // same predicate the dialog lists tracks with (scoreExportableTracks), so they can't disagree.
    { label: t("menu.exportAudio"), icon: "🔊", disabled: trackCount === 0, onClick: () => setExportAudioOpen(true) },
    {
      label: t("menu.exportScore"),
      icon: "🎵",
      disabled: scoreExportableTracks(useProjectStore.getState().tracks).length === 0,
      onClick: () => void setExportScoreOpen(true),
    },
    // 轨道另存为:整工程所有可导出音频复制进用户选定的一个文件夹。
    {
      label: t("menu.exportTracks"),
      icon: "⬇",
      disabled: !anyProjectExportable(useProjectStore.getState().tracks),
      title: anyProjectExportable(useProjectStore.getState().tracks) ? undefined : t("tracks.exportNothing"),
      onClick: () => void exportAllTracks(),
    },
  ];

  // Clipboard entries act on the ARRANGEMENT selection (the vocal editor owns note copy/paste via its
  // own Ctrl+C/V while focused) — so they enable only while the timeline pane is active. Read lazily on
  // menu open, same as undo/redo enablement above.
  const timelineActive = useAppStore.getState().activePane === "timeline";
  const hasSelection = useAppStore.getState().selectedSegments.length > 0 || useAppStore.getState().selectedSegment !== null;
  const editItems: MenuItem[] = [
    {
      label: t("menu.undo"),
      icon: "↩",
      shortcut: "Ctrl+Z",
      disabled: !routeCanUndo(),
      onClick: () => routeUndo(),
    },
    {
      label: t("menu.redo"),
      icon: "↪",
      shortcut: "Ctrl+Y",
      disabled: !routeCanRedo(),
      onClick: () => routeRedo(),
    },
    {
      label: t("history.panelTitle"),
      icon: "🕘",
      onClick: () => setHistoryOpen(true),
    },
    {
      label: t("menu.copy"),
      icon: "📋",
      shortcut: "Ctrl+C",
      disabled: !timelineActive || !hasSelection,
      onClick: () => { copySelectedSegments(); },
    },
    {
      label: t("menu.cut"),
      icon: "✂️",
      shortcut: "Ctrl+X",
      disabled: !timelineActive || !hasSelection,
      onClick: () => { cutSelectedSegments(); },
    },
    {
      label: t("menu.paste"),
      icon: "📌",
      shortcut: "Ctrl+V",
      disabled: !timelineActive || clipboardKind() === null,
      onClick: () => pasteWithFeedback(),
    },
  ];

  // Muno 阶段4:AI 菜单 —— 目标轨在打开菜单时解析(与 Edit 菜单同一惰性模式):
  // 旋律类动作用活跃/选中/第一条音符轨;音频类动作用第一条有音频源的轨。
  const aiMelodyId = resolveMelodyTrackId();
  const aiAudioTrack = useProjectStore
    .getState()
    .tracks.find((tr) => resolveTrackSourceAudio(tr)?.path);
  const aiAudioSrc = aiAudioTrack ? resolveTrackSourceAudio(aiAudioTrack) : null;
  const aiItems: MenuItem[] = [
    {
      label: t("arrange.entry"),
      icon: "✨",
      disabled: !aiMelodyId,
      title: aiMelodyId ? undefined : t("arrange.noNotes"),
      onClick: () => { if (aiMelodyId) setAiArrangeTarget(aiMelodyId); },
    },
    {
      label: t("wizard.titlebarBtn"),
      icon: "✨",
      onClick: () => setWizardOpen(true),
    },
    {
      label: t("menu.lyric"),
      icon: "🎤",
      onClick: () => window.dispatchEvent(new CustomEvent("utai:open-lyric-wizard")),
    },
    {
      label: t("menu.autoTune"),
      icon: "🎵",
      onClick: () => {
        const targetId = resolveMelodyTrackId();
        if (!targetId) { showToast("请先选中一条人声/旋律轨", "error"); return; }
        const store = useProjectStore.getState();
        const track = store.tracks.find((t) => t.id === targetId);
        const cur = track?.vocalParams?.autoTuneFollow;
        store.setVocalParams(targetId, { autoTuneFollow: cur === false ? true : false });
        showToast(cur === false ? "自动修音：开" : "自动修音：关", "info");
      },
    },
    {
      label: t("chordTrack.analyze"),
      icon: "🎼",
      disabled: !aiMelodyId,
      title: aiMelodyId ? undefined : t("chordTrack.analyzeHint"),
      onClick: () => { if (aiMelodyId) analyzeTrackChords(aiMelodyId); },
    },
    {
      label: t("chordTrack.detectDrums"),
      icon: "🥁",
      disabled: !aiAudioSrc?.path,
      title: aiAudioSrc?.path ? undefined : t("amt.noSource", "轨道没有音频源，无法转换"),
      onClick: () => {
        if (aiAudioTrack && aiAudioSrc?.path) void detectTrackDrums(aiAudioTrack.id, aiAudioSrc.path);
      },
    },
    {
      label: t("tracks.convertToMidi"),
      icon: "🎶",
      disabled: !aiAudioSrc?.path,
      title: aiAudioSrc?.path ? undefined : t("amt.noSource", "轨道没有音频源，无法转换"),
      onClick: () => {
        if (!aiAudioTrack || !aiAudioSrc?.path) return;
        useAppStore.getState().setAmtSource(aiAudioSrc.path, aiAudioSrc.name || aiAudioTrack.name);
        useAppStore.getState().openAmtConversion(aiAudioTrack.id);
      },
    },
    {
      label: t("menu.chordMidi"),
      icon: "🎹",
      disabled: !aiMelodyId,
      title: aiMelodyId ? undefined : t("chordMidi.noNotes"),
      onClick: () => { if (aiMelodyId) setAiChordTarget(aiMelodyId); },
    },
  ];

  // 工具菜单：收纳模型、音色、歌曲工作室、训练、日志等次要功能
  const toolsItems: MenuItem[] = [
    { label: t("titlebar.models"), icon: "📦", onClick: toggleModelManager },
    { label: t("titlebar.soundfonts"), icon: "🎹", onClick: toggleSoundfontManager },
    { label: t("titlebar.songStudio"), icon: "🎧", onClick: toggleSongStudio },
    { label: t("titlebar.training"), icon: "🎓", onClick: toggleTrainingPage },
    { label: t("titlebar.log"), icon: "📋", onClick: toggleLogViewer },
  ];

  return (
    <header className={`titlebar${splashLocked ? " splash-locked" : ""}`} data-tauri-drag-region>
      {/* Row 1: 品牌区 + 项目名 + 窗口控制 */}
      <div className="titlebar-row1">
        <div className="titlebar-left">
          <span className="titlebar-brand">
            <img className="titlebar-logo" src="/logo-64.png" alt="MunoAI" width="22" height="22" />
            <span className="titlebar-brand-name">MunoAI</span>
            <span className="titlebar-brand-sep">·</span>
            <span className="titlebar-brand-tag">造乐之地</span>
          </span>
        </div>

        <div className="titlebar-center">
          {!splashLocked && (
            <span className="project-name">
              {name || t("untitled")}
              {dirty && <span className="dirty-dot" />}
            </span>
          )}
        </div>

        <div className="titlebar-right">
          {isTraining && (
            <span className="training-indicator">
              <span className="pulse-dot" />
              {t("training.active")}
            </span>
          )}

        {/* 窗口控制按钮 (无边框窗口) */}
        <div className="window-controls" data-tauri-drag-region="false">
          <button
            className="win-btn win-min"
            title="最小化"
            onClick={() => getCurrentWindow().minimize()}
          >
            <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
              <rect x="2" y="5.5" width="8" height="1" fill="currentColor" />
            </svg>
          </button>
          <button
            className="win-btn win-max"
            title="最大化 / 还原"
            onClick={async () => {
              const w = getCurrentWindow();
              if (await w.isMaximized()) await w.unmaximize(); else await w.maximize();
            }}
          >
            <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
              <rect x="2" y="2" width="8" height="8" fill="none" stroke="currentColor" strokeWidth="1.2" />
            </svg>
          </button>
          <button
            className="win-btn win-close"
            title="关闭"
            onClick={() => getCurrentWindow().close()}
          >
            <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
              <path d="M3 3l6 6M9 3l-6 6" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
            </svg>
          </button>
        </div>
        </div>
      </div>

      {/* Row 2: 导航菜单 (仅进入软件后显示) */}
      {!splashLocked && (
      <div className="titlebar-row2" data-tauri-drag-region="false">
        <nav className="titlebar-menu">
          <button
            className="menu-item"
            onClick={(e) => {
              const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
              setFileMenu({ x: r.left, y: r.bottom });
            }}
          >
            {t("menu.file")}
          </button>
          <button
            className="menu-item"
            onClick={(e) => {
              const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
              setEditMenu({ x: r.left, y: r.bottom });
            }}
          >
            {t("menu.edit")}
          </button>
          <button
            className={`menu-item ${aiMenu ? "active" : ""}`}
            onClick={(e) => {
              const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
              setAiMenu({ x: r.left, y: r.bottom });
            }}
          >
            {t("menu.ai")}
          </button>
          <button
            className="menu-item menu-item-accent"
            disabled={!aiMelodyId}
            title={aiMelodyId ? "打开智能编曲面板(乐器自由组合 · 实时试听)" : "先选一条旋律/音符轨,或先把音频 AI 转谱"}
            onClick={() => { if (aiMelodyId) setAiArrangeTarget(aiMelodyId); }}
          >
            智能编曲
          </button>
          <button
            className={`menu-item ${toolsMenu ? "active" : ""}`}
            onClick={(e) => {
              const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
              setToolsMenu({ x: r.left, y: r.bottom });
            }}
          >
            工具
          </button>
          <button className={`menu-item ${settingsOpen ? "active" : ""}`} onClick={toggleSettings}>{t("menu.settings")}</button>
          <button
            className={`menu-item ${helpMenu ? "active" : ""}`}
            onClick={(e) => {
              const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
              setHelpMenu({ x: r.left, y: r.bottom });
            }}
          >
            {t("help.title")}
          </button>
        </nav>
      </div>
      )}

      {fileMenu && (
        <ContextMenu x={fileMenu.x} y={fileMenu.y} items={fileItems} onClose={() => setFileMenu(null)} />
      )}
      {editMenu && (
        <ContextMenu x={editMenu.x} y={editMenu.y} items={editItems} onClose={() => setEditMenu(null)} />
      )}
      {aiMenu && (
        <ContextMenu x={aiMenu.x} y={aiMenu.y} items={aiItems} onClose={() => setAiMenu(null)} />
      )}
      {toolsMenu && (
        <ContextMenu x={toolsMenu.x} y={toolsMenu.y} items={toolsItems} onClose={() => setToolsMenu(null)} />
      )}
      {helpMenu && (
        <ContextMenu x={helpMenu.x} y={helpMenu.y} items={helpItems} onClose={() => setHelpMenu(null)} />
      )}
      {exportAudioOpen && <ExportAudioDialog onClose={() => setExportAudioOpen(false)} />}
      {exportScoreOpen && <ExportScoreDialog onClose={() => setExportScoreOpen(false)} />}
      {wizardOpen && <SuperOriginalWizard onClose={() => setWizardOpen(false)} />}
      {aiArrangeTarget && (
        <ArrangeDialog trackId={aiArrangeTarget} onClose={() => setAiArrangeTarget(null)} />
      )}
      {aiChordTarget && (
        <ChordMidiDialog trackId={aiChordTarget} onClose={() => setAiChordTarget(null)} />
      )}
      {shortcutsOpen && <ShortcutsDialog onClose={() => setShortcutsOpen(false)} />}
      {userGuideOpen && <UserGuideDialog onClose={() => setUserGuideOpen(false)} />}
      {historyOpen && <HistoryPanel onClose={() => setHistoryOpen(false)} />}
    </header>
  );
}


