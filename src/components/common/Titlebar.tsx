import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import { useTrainingStore } from "../../store/training";
import { useTranslation } from "react-i18next";
import "./Titlebar.css";

export function Titlebar() {
  const { t } = useTranslation();
  const { name, dirty } = useProjectStore();
  const { toggleTrainingPanel, trainingPanelOpen } = useAppStore();
  const { status } = useTrainingStore();

  const isTraining =
    status.state === "Training" ||
    status.state === "Preprocessing" ||
    status.state === "Preparing";

  return (
    <header className="titlebar" data-tauri-drag-region>
      <div className="titlebar-left">
        <span className="titlebar-brand">UTAI</span>
        <nav className="titlebar-menu">
          <button className="menu-item">{t("menu.file")}</button>
          <button className="menu-item">{t("menu.edit")}</button>
          <button className="menu-item">{t("menu.view")}</button>
          <button className="menu-item">{t("menu.tools")}</button>
        </nav>
      </div>

      <div className="titlebar-center" data-tauri-drag-region>
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
          data-tooltip={t("training.panel")}
        >
          ⚡
        </button>
      </div>
    </header>
  );
}
