import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import { useTrainingStore } from "../../store/training";
import { useTranslation } from "react-i18next";
import "./Titlebar.css";

export function Titlebar() {
  const { t } = useTranslation();
  const { name, dirty } = useProjectStore();
  const { toggleTrainingPanel, trainingPanelOpen, toggleModelManager, modelManagerOpen, toggleLogViewer, logViewerOpen, toggleSettings, settingsOpen } = useAppStore();
  const { status } = useTrainingStore();

  const isTraining =
    status.state === "Training" ||
    status.state === "Preprocessing" ||
    status.state === "Preparing";

  return (
    <header className="titlebar">
      <div className="titlebar-left">
        <span className="titlebar-brand">UTAI</span>
        <nav className="titlebar-menu">
          <button className="menu-item">{t("menu.file")}</button>
          <button className="menu-item">{t("menu.edit")}</button>
          <button className="menu-item">{t("menu.view")}</button>
          <button className="menu-item">{t("menu.tools")}</button>
          <button className={`menu-item ${settingsOpen ? "active" : ""}`} onClick={toggleSettings}>{t("menu.settings")}</button>
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
          className={`titlebar-btn ${logViewerOpen ? "active" : ""}`}
          onClick={toggleLogViewer}
        >
          {t("titlebar.log")}
        </button>
        <button
          className={`titlebar-btn ${modelManagerOpen ? "active" : ""}`}
          onClick={toggleModelManager}
        >
          {t("titlebar.models")}
        </button>
        <button
          className={`titlebar-btn ${trainingPanelOpen ? "active" : ""}`}
          onClick={toggleTrainingPanel}
        >
          {t("titlebar.training")}
        </button>
      </div>
    </header>
  );
}
