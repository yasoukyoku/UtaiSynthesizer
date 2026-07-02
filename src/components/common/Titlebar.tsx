import { useState } from "react";
import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import { routeUndo, routeRedo, routeCanUndo, routeCanRedo } from "../../store/history";
import { useTrainingStore } from "../../store/training";
import { useTranslation } from "react-i18next";
import { ContextMenu, type MenuItem } from "./ContextMenu";
import { newProjectFile, openProjectFile, saveProjectFile, saveProjectFileAs } from "../../lib/project/projectFile";
import "./Titlebar.css";

export function Titlebar() {
  const { t } = useTranslation();
  const { name, dirty } = useProjectStore();
  const trackCount = useProjectStore((s) => s.tracks.length);
  const { toggleTrainingPanel, trainingPanelOpen, toggleModelManager, modelManagerOpen, toggleLogViewer, logViewerOpen, toggleSettings, settingsOpen } = useAppStore();
  const { status } = useTrainingStore();
  // The Edit menu's enablement is read via routeCanUndo/routeCanRedo when the menu opens (opening it
  // sets editMenu → re-render), so it reflects whichever stack is active (the workflow editor's
  // modal-local stack while open, else the timeline) without a live subscription.
  const [editMenu, setEditMenu] = useState<{ x: number; y: number } | null>(null);
  const [fileMenu, setFileMenu] = useState<{ x: number; y: number } | null>(null);

  const isTraining =
    status.state === "Training" ||
    status.state === "Preprocessing" ||
    status.state === "Preparing";

  const fileItems: MenuItem[] = [
    { label: t("menu.new"), shortcut: "Ctrl+N", onClick: () => void newProjectFile() },
    { label: t("menu.open"), shortcut: "Ctrl+O", onClick: () => void openProjectFile() },
    { label: t("menu.save"), shortcut: "Ctrl+S", disabled: trackCount === 0, onClick: () => void saveProjectFile() },
    { label: t("menu.saveAs"), shortcut: "Ctrl+Shift+S", disabled: trackCount === 0, onClick: () => void saveProjectFileAs() },
  ];

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
  ];

  return (
    <header className="titlebar">
      <div className="titlebar-left">
        <span className="titlebar-brand">UTAI</span>
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
          <button className="menu-item">{t("menu.tools")}</button>
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
          className={`titlebar-btn ${trainingPanelOpen ? "active" : ""}`}
          onClick={toggleTrainingPanel}
        >
          {t("titlebar.training")}
        </button>
      </div>

      {fileMenu && (
        <ContextMenu x={fileMenu.x} y={fileMenu.y} items={fileItems} onClose={() => setFileMenu(null)} />
      )}
      {editMenu && (
        <ContextMenu x={editMenu.x} y={editMenu.y} items={editItems} onClose={() => setEditMenu(null)} />
      )}
    </header>
  );
}
