import { create } from "zustand";
import { invoke } from "@tauri-apps/api/core";

export type TrainingState =
  | "Idle"
  | "Preparing"
  | "Preprocessing"
  | "Training"
  | "GeneratingIndex"
  | "Completed"
  | "Stopped"
  | { Error: string };

interface TrainingStatus {
  state: TrainingState;
  current_epoch: number;
  total_epochs: number;
  loss: number | null;
  elapsed_secs: number;
  eta_secs: number | null;
  model_name: string;
}

interface TrainingConfig {
  model_name: string;
  backend: "rvc" | "sovits";
  dataset_path: string;
  epochs: number;
  batch_size: number;
  sample_rate: number;
  save_interval: number;
  augmentation_intensity: number;
  continuation_mode: "fresh" | "continue";
}

interface TrainingStore {
  status: TrainingStatus;
  config: TrainingConfig;

  fetchStatus: () => Promise<void>;
  startTraining: () => Promise<void>;
  stopTraining: () => Promise<void>;
  updateConfig: (updates: Partial<TrainingConfig>) => void;
}

const defaultConfig: TrainingConfig = {
  model_name: "",
  backend: "rvc",
  dataset_path: "",
  epochs: 200,
  batch_size: 8,
  sample_rate: 40000,
  save_interval: 25,
  augmentation_intensity: 0.0,
  continuation_mode: "fresh",
};

const idleStatus: TrainingStatus = {
  state: "Idle",
  current_epoch: 0,
  total_epochs: 0,
  loss: null,
  elapsed_secs: 0,
  eta_secs: null,
  model_name: "",
};

export const useTrainingStore = create<TrainingStore>((set, get) => ({
  status: idleStatus,
  config: defaultConfig,

  fetchStatus: async () => {
    try {
      const status = await invoke<TrainingStatus>("get_training_status");
      set({ status });
    } catch {
      // Backend not ready yet
    }
  },

  startTraining: async () => {
    const { config } = get();
    const backendConfig =
      config.backend === "rvc"
        ? { Rvc: { version: "V2" } }
        : { SoVits: { shallow_diffusion: true } };

    const augmentation =
      config.augmentation_intensity > 0
        ? { intensity: config.augmentation_intensity }
        : null;

    await invoke("start_training", {
      config: {
        model_name: config.model_name,
        backend: backendConfig,
        dataset_path: config.dataset_path,
        epochs: config.epochs,
        batch_size: config.batch_size,
        sample_rate: config.sample_rate,
        save_interval: config.save_interval,
        augmentation,
        continuation:
          config.continuation_mode === "fresh"
            ? "Fresh"
            : { Continue: { from_epoch: 0 } },
      },
    });
  },

  stopTraining: async () => {
    await invoke("stop_training");
  },

  updateConfig: (updates) =>
    set((s) => ({ config: { ...s.config, ...updates } })),
}));
