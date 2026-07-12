import { useEffect, useState } from "react";
import { getVersion } from "@tauri-apps/api/app";
import { open as openUrl } from "@tauri-apps/plugin-shell";
import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import { routeUndo, routeRedo, routeCanUndo, routeCanRedo } from "../../store/history";
import { useTrainingStore } from "../../store/training";
import { useTranslation } from "react-i18next";
import { ContextMenu, type MenuItem } from "./ContextMenu";
import { newProjectFile, openProjectFile, saveProjectFile, saveProjectFileAs } from "../../lib/project/projectFile";
import { importScoreFile } from "../../lib/vocal/import";
import { scoreExportableTracks } from "../../lib/vocal/exportScore";
import { copySelectedSegments, cutSelectedSegments, pasteWithFeedback, clipboardKind } from "../../lib/clipboard";
import { ExportAudioDialog } from "./ExportAudioDialog";
import { ExportScoreDialog } from "./ExportScoreDialog";
import "./Titlebar.css";

// S64 — Help/community links (the "?" titlebar control). External URLs open in the system browser
// via plugin-shell; the project GitHub is also the update source (commands/update.rs).
const HELP_LINKS = {
  qq: "https://qun.qq.com/universal-share/share?ac=1&authKey=3uD5AoM8e50y00vhOYOZsa2VI341dBNfr07S2IK9wraewz0rcFHpSzONYJ9QrTP7&busi_data=eyJncm91cENvZGUiOiIxMDU4MjI3MjEyIiwidG9rZW4iOiJONGpqQ2MzM3h3N3BDMVBMRzZiSUFOU05YWnRnbHBxdTZDUElZYlZOSGN3VnhCaEc5eWludlJBYlltK3hkdlFwIiwidWluIjoiMjc2Njc2NDM1NSJ9&data=VyWCaG06iaMLBFcfEx_fjE2Tme2X7YvJsUIUjJ51zk6XymaED6Z6TEC_zOvAdm9q2MbzbYbpuO4ukQHZ1GBHLw&svctype=4&tempid=h5_group_info",
  discord: "https://discord.com/invite/p3fGh942fJ",
  score2convec: "https://github.com/yasoukyoku/Score2ConVec",
  repo: "https://github.com/yasoukyoku/UtaiSynthesizer",
} as const;

export function Titlebar() {
  const { t } = useTranslation();
  const { name, dirty } = useProjectStore();
  const trackCount = useProjectStore((s) => s.tracks.length);
  const { toggleTrainingPage, trainingPageOpen, toggleModelManager, modelManagerOpen, toggleLogViewer, logViewerOpen, toggleSettings, settingsOpen } = useAppStore();
  const trainingState = useTrainingStore((s) => s.snapshot.state);
  // The Edit menu's enablement is read via routeCanUndo/routeCanRedo when the menu opens (opening it
  // sets editMenu → re-render), so it reflects whichever stack is active (the workflow editor's
  // modal-local stack while open, else the timeline) without a live subscription.
  const [editMenu, setEditMenu] = useState<{ x: number; y: number } | null>(null);
  const [fileMenu, setFileMenu] = useState<{ x: number; y: number } | null>(null);
  const [helpMenu, setHelpMenu] = useState<{ x: number; y: number } | null>(null);
  const [exportAudioOpen, setExportAudioOpen] = useState(false);
  const [exportScoreOpen, setExportScoreOpen] = useState(false);
  const [appVersion, setAppVersion] = useState("");
  useEffect(() => { void getVersion().then(setAppVersion).catch(() => {}); }, []);

  const helpItems: MenuItem[] = [
    { label: `UtaiSynthesizer ${appVersion ? `v${appVersion}` : ""}`.trim(), disabled: true, onClick: () => {} },
    { label: t("help.qq"), onClick: () => void openUrl(HELP_LINKS.qq).catch(() => {}) },
    { label: t("help.discord"), onClick: () => void openUrl(HELP_LINKS.discord).catch(() => {}) },
    { label: t("help.score2convec"), onClick: () => void openUrl(HELP_LINKS.score2convec).catch(() => {}) },
    { label: t("help.repo"), onClick: () => void openUrl(HELP_LINKS.repo).catch(() => {}) },
  ];

  const isTraining = trainingState === "running" || trainingState === "starting";

  const fileItems: MenuItem[] = [
    { label: t("menu.new"), shortcut: "Ctrl+N", onClick: () => void newProjectFile() },
    { label: t("menu.open"), shortcut: "Ctrl+O", onClick: () => void openProjectFile() },
    { label: t("menu.save"), shortcut: "Ctrl+S", disabled: trackCount === 0, onClick: () => void saveProjectFile() },
    { label: t("menu.saveAs"), shortcut: "Ctrl+Shift+S", disabled: trackCount === 0, onClick: () => void saveProjectFileAs() },
    { label: t("menu.import"), onClick: () => void importScoreFile() },
    // S63 export entries. Enablement is read lazily on menu open (the same pattern as the clipboard
    // items below): audio needs any track at all, score needs ≥1 vocal track with notes — via THE
    // same predicate the dialog lists tracks with (scoreExportableTracks), so they can't disagree.
    { label: t("menu.exportAudio"), disabled: trackCount === 0, onClick: () => setExportAudioOpen(true) },
    {
      label: t("menu.exportScore"),
      disabled: scoreExportableTracks(useProjectStore.getState().tracks).length === 0,
      onClick: () => setExportScoreOpen(true),
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
      shortcut: "Ctrl+Z",
      disabled: !routeCanUndo(),
      onClick: () => routeUndo(),
    },
    {
      label: t("menu.redo"),
      shortcut: "Ctrl+Y",
      disabled: !routeCanRedo(),
      onClick: () => routeRedo(),
    },
    {
      label: t("menu.copy"),
      shortcut: "Ctrl+C",
      disabled: !timelineActive || !hasSelection,
      onClick: () => { copySelectedSegments(); },
    },
    {
      label: t("menu.cut"),
      shortcut: "Ctrl+X",
      disabled: !timelineActive || !hasSelection,
      onClick: () => { cutSelectedSegments(); },
    },
    {
      label: t("menu.paste"),
      shortcut: "Ctrl+V",
      disabled: !timelineActive || clipboardKind() === null,
      onClick: () => pasteWithFeedback(),
    },
  ];

  return (
    <header className="titlebar">
      <div className="titlebar-left">
        <span className="titlebar-brand">
          UTAI<span className="titlebar-brand-sub">SYNTHESIZER</span>
        </span>
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
          <button className={`menu-item ${modelManagerOpen ? "active" : ""}`} onClick={toggleModelManager}>{t("titlebar.models")}</button>
          <button className={`menu-item ${settingsOpen ? "active" : ""}`} onClick={toggleSettings}>{t("menu.settings")}</button>
          <button className={`menu-item ${logViewerOpen ? "active" : ""}`} onClick={toggleLogViewer}>{t("titlebar.log")}</button>
        </nav>
      </div>

      <div className="titlebar-center">
        <span className="project-name">
          {name || t("untitled")}
          {dirty && <span className="dirty-dot" />}
        </span>
      </div>

      <div className="titlebar-right">
        {isTraining && (
          <span className="training-indicator">
            <span className="pulse-dot" />
            {t("training.active")}
          </span>
        )}
        <button
          className={`titlebar-btn ${trainingPageOpen ? "active" : ""}`}
          onClick={toggleTrainingPage}
        >
          {t("titlebar.training")}
        </button>
        <button
          className={`titlebar-btn titlebar-help ${helpMenu ? "active" : ""}`}
          title={t("help.title")}
          onClick={(e) => {
            const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
            setHelpMenu({ x: r.left, y: r.bottom });
          }}
        >
          {/* Angular question mark — square caps/joins, themed stroke (house SVG style, no raw emoji). */}
          <svg viewBox="0 0 24 24" width="13" height="13" aria-hidden="true">
            <path
              d="M8 9 V8 a4 4 0 0 1 4-4 a4 4 0 0 1 4 4 v0.5 c0 1.8-1.4 2.6-2.6 3.4 c-1 0.65-1.4 1.3-1.4 2.6 v0.5"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.2"
              strokeLinecap="square"
            />
            <rect x="10.9" y="18" width="2.4" height="2.4" fill="currentColor" />
          </svg>
        </button>
      </div>

      {fileMenu && (
        <ContextMenu x={fileMenu.x} y={fileMenu.y} items={fileItems} onClose={() => setFileMenu(null)} />
      )}
      {editMenu && (
        <ContextMenu x={editMenu.x} y={editMenu.y} items={editItems} onClose={() => setEditMenu(null)} />
      )}
      {helpMenu && (
        <ContextMenu x={helpMenu.x} y={helpMenu.y} items={helpItems} onClose={() => setHelpMenu(null)} />
      )}
      {exportAudioOpen && <ExportAudioDialog onClose={() => setExportAudioOpen(false)} />}
      {exportScoreOpen && <ExportScoreDialog onClose={() => setExportScoreOpen(false)} />}
    </header>
  );
}
