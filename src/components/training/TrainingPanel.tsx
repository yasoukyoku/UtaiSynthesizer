import { useState } from "react";
import { useTrainingStore, type TrainingState } from "../../store/training";
import { useTranslation } from "react-i18next";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import "./TrainingPanel.css";

export function TrainingPanel() {
  const { t } = useTranslation();
  const { status, config, updateConfig, startTraining, stopTraining } =
    useTrainingStore();
  const [showOverwriteDialog, setShowOverwriteDialog] = useState(false);

  const isActive =
    status.state === "Training" ||
    status.state === "Preparing" ||
    status.state === "Preprocessing" ||
    status.state === "GeneratingIndex";

  const handleStart = async () => {
    if (!config.model_name || !config.dataset_path) return;

    try {
      const exists = await invoke<boolean>("check_model_exists", {
        name: config.model_name,
        modelType: config.backend,
      });

      if (exists) {
        setShowOverwriteDialog(true);
      } else {
        updateConfig({ continuation_mode: "fresh" });
        await startTraining();
      }
    } catch (e) {
      console.error("Failed to start training:", e);
    }
  };

  const handleOverwriteChoice = async (mode: "fresh" | "continue") => {
    setShowOverwriteDialog(false);
    updateConfig({ continuation_mode: mode });
    await startTraining();
  };

  const handleSelectDataset = async () => {
    const path = await open({
      directory: true,
      title: t("training.selectDataset"),
    });
    if (path) {
      updateConfig({ dataset_path: path as string });
    }
  };

  return (
    <aside className="training-panel">
      <div className="panel-header">
        <span className="panel-title">{t("training.title")}</span>
        <span className="panel-subtitle mono">
          {isActive ? formatState(status.state) : "IDLE"}
        </span>
      </div>

      {isActive ? (
        <TrainingProgress status={status} onStop={stopTraining} />
      ) : (
        <TrainingConfig
          config={config}
          onUpdate={updateConfig}
          onStart={handleStart}
          onSelectDataset={handleSelectDataset}
        />
      )}

      {showOverwriteDialog && (
        <OverwriteDialog
          modelName={config.model_name}
          onContinue={() => handleOverwriteChoice("continue")}
          onFresh={() => handleOverwriteChoice("fresh")}
          onCancel={() => setShowOverwriteDialog(false)}
        />
      )}
    </aside>
  );
}

interface TrainingStatusData {
  state: TrainingState;
  current_epoch: number;
  total_epochs: number;
  loss: number | null;
  elapsed_secs: number;
  eta_secs: number | null;
  model_name: string;
}

function TrainingProgress({
  status,
  onStop,
}: {
  status: TrainingStatusData;
  onStop: () => void;
}) {
  const { t } = useTranslation();
  const progress =
    status.total_epochs > 0
      ? (status.current_epoch / status.total_epochs) * 100
      : 0;

  return (
    <div className="training-progress">
      <div className="progress-header">
        <span className="mono">{status.model_name}</span>
      </div>

      <div className="progress-bar-container">
        <div className="progress-bar" style={{ width: `${progress}%` }} />
      </div>

      <div className="progress-stats">
        <div className="stat">
          <span className="stat-label">{t("training.epoch")}</span>
          <span className="stat-value mono">
            {status.current_epoch} / {status.total_epochs}
          </span>
        </div>
        {status.loss != null && (
          <div className="stat">
            <span className="stat-label">{t("training.loss")}</span>
            <span className="stat-value mono">{status.loss.toFixed(4)}</span>
          </div>
        )}
        <div className="stat">
          <span className="stat-label">{t("training.elapsed")}</span>
          <span className="stat-value mono">
            {formatDuration(status.elapsed_secs)}
          </span>
        </div>
        {status.eta_secs != null && (
          <div className="stat">
            <span className="stat-label">{t("training.eta")}</span>
            <span className="stat-value mono">
              {formatDuration(status.eta_secs)}
            </span>
          </div>
        )}
      </div>

      <button className="danger stop-btn" onClick={onStop}>
        {t("training.stop")}
      </button>
    </div>
  );
}

interface TrainingConfigData {
  model_name: string;
  backend: "rvc" | "sovits";
  dataset_path: string;
  epochs: number;
  batch_size: number;
  sample_rate: number;
  augmentation_intensity: number;
}

function TrainingConfig({
  config,
  onUpdate,
  onStart,
  onSelectDataset,
}: {
  config: TrainingConfigData;
  onUpdate: (u: Partial<TrainingConfigData>) => void;
  onStart: () => void;
  onSelectDataset: () => void;
}) {
  const { t } = useTranslation();

  return (
    <div className="training-config">
      <div className="config-field">
        <label>{t("training.modelName")}</label>
        <input
          type="text"
          value={config.model_name}
          onChange={(e) => onUpdate({ model_name: e.target.value })}
          placeholder={t("training.modelNamePlaceholder")}
        />
      </div>

      <div className="config-field">
        <label>{t("training.backend")}</label>
        <select
          value={config.backend}
          onChange={(e) =>
            onUpdate({ backend: e.target.value as "rvc" | "sovits" })
          }
        >
          <option value="rvc">RVC v2</option>
          <option value="sovits">SoVITS 4.1</option>
        </select>
      </div>

      <div className="config-field">
        <label>{t("training.dataset")}</label>
        <div className="field-with-btn">
          <input
            type="text"
            value={config.dataset_path}
            readOnly
            placeholder={t("training.datasetPlaceholder")}
          />
          <button onClick={onSelectDataset}>...</button>
        </div>
      </div>

      <div className="config-row">
        <div className="config-field half">
          <label>{t("training.epochs")}</label>
          <input
            type="number"
            value={config.epochs}
            min={10}
            max={10000}
            onChange={(e) => onUpdate({ epochs: Number(e.target.value) })}
          />
        </div>
        <div className="config-field half">
          <label>{t("training.batchSize")}</label>
          <input
            type="number"
            value={config.batch_size}
            min={1}
            max={64}
            onChange={(e) => onUpdate({ batch_size: Number(e.target.value) })}
          />
        </div>
      </div>

      <div className="config-field">
        <label>{t("training.sampleRate")}</label>
        <select
          value={config.sample_rate}
          onChange={(e) => onUpdate({ sample_rate: Number(e.target.value) })}
        >
          <option value={32000}>32000 Hz</option>
          <option value={40000}>40000 Hz</option>
          <option value={48000}>48000 Hz</option>
        </select>
      </div>

      <div className="config-field augment-section">
        <label>
          {t("training.augmentation")}
          <span className="mono augment-value">
            {(config.augmentation_intensity * 100).toFixed(0)}%
          </span>
        </label>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={config.augmentation_intensity}
          onChange={(e) =>
            onUpdate({ augmentation_intensity: Number(e.target.value) })
          }
        />
        <div className="augment-labels">
          <span className="text-muted">{t("training.augNone")}</span>
          <span className="text-muted">{t("training.augStrong")}</span>
        </div>
      </div>

      <button
        className="primary start-btn"
        onClick={onStart}
        disabled={!config.model_name || !config.dataset_path}
      >
        {t("training.start")}
      </button>
    </div>
  );
}

function OverwriteDialog({
  modelName,
  onContinue,
  onFresh,
  onCancel,
}: {
  modelName: string;
  onContinue: () => void;
  onFresh: () => void;
  onCancel: () => void;
}) {
  const { t } = useTranslation();

  return (
    <div className="dialog-overlay" onClick={onCancel}>
      <div className="dialog" onClick={(e) => e.stopPropagation()}>
        <h3>{t("training.modelExists")}</h3>
        <p>
          {t("training.modelExistsDesc", { name: modelName })}
        </p>
        <div className="dialog-actions">
          <button onClick={onContinue}>{t("training.continueTraining")}</button>
          <button className="danger" onClick={onFresh}>
            {t("training.deleteRetrain")}
          </button>
          <button onClick={onCancel}>{t("common.cancel")}</button>
        </div>
      </div>
    </div>
  );
}

function formatDuration(secs: number): string {
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = secs % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function formatState(state: TrainingState): string {
  if (typeof state === "string") return state;
  if (typeof state === "object" && "Error" in state) return `Error: ${state.Error}`;
  return "Unknown";
}
