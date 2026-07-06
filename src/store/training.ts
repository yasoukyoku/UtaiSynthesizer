/**
 * Training store — mirrors the Rust TrainingManager (protocol v2, S37 rewrite).
 *
 * Event-driven (training-stage / training-step / training-ckpt / training-done),
 * NOT polled: install the module-level listeners once via setupTrainingListeners()
 * (msst-models.ts pattern — global, so progress survives the page being closed).
 * The Rust side keeps the authoritative loss history (get_training_history) so a
 * re-mounted page reconstructs the curve; live points append via training-step.
 */
import { create } from "zustand";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import i18n from "../i18n";
import { useAppStore } from "./app";

export interface DatasetFile {
  path: string;
  name: string;
  durationMs: number | null;
}

export interface StageInfo {
  stage: string;
  done?: number | null;
  total?: number | null;
  progress?: number | null;
  message?: string | null;
}

export interface StepInfo {
  step: number;
  total_steps: number;
  epoch: number;
  total_epochs: number;
  lr: number;
  losses: Record<string, number>;
  eta_secs?: number | null;
}

export interface CkptInfo {
  kind: "periodic" | "best" | "final" | "stop";
  path: string;
  step: number;
  epoch: number;
  metric?: number | null;
}

export interface StepPoint {
  step: number;
  lr: number;
  losses: Record<string, number>;
}

/** Mirror of Rust `training::WorkspaceInfo` (get_training_workspace_info). */
export interface WorkspaceInfo {
  exists: boolean;
  /** manifest family ("rvc"/"sovits"); "" when absent */
  family: string;
  /** manifest version ("v1"/"v2"/"4.1"/"4.0"); "" when absent */
  version: string;
  /** manifest sample rate ("32k"/"40k"/"48k"/"44k"); "" when absent */
  sample_rate: string;
  has_main_progress: boolean;
  /** max diffusion checkpoint step; 0 = none/base only */
  diff_steps: number;
}

export interface TrainingSnapshot {
  state: "idle" | "starting" | "running" | "completed" | "stopped" | "error";
  error?: string | null;
  backend: string;
  model_name: string;
  model_slug: string;
  workspace: string;
  total_epochs: number;
  stage?: StageInfo | null;
  step?: StepInfo | null;
  ckpts: CkptInfo[];
  summary?: Record<string, unknown> | null;
  stop_requested: boolean;
  elapsed_secs: number;
  stderr_tail: string[];
}

export interface TrainingFormConfig {
  modelName: string;
  backend: "rvc" | "sovits" | "sovits_diff";
  version: "v1" | "v2";
  sampleRate: "32k" | "40k" | "48k";
  totalEpoch: number;
  batchSize: number;
  saveEveryEpoch: number;
  saveEveryWeights: boolean;
  keepOnlyLatest: boolean;
  cacheGpu: boolean;
  fp16: boolean;
  gpu: number;
  forceCpu: boolean;
  // ---- SoVITS (44.1kHz fixed; separate fields so switching cards never
  // clobbers the RVC values with SoVITS-scaled ones) ----
  sovitsVersion: "4.1" | "4.0";
  sovitsTotalEpoch: number;
  sovitsBatchSize: number;
  /** ckpt/eval cadence in global steps (upstream eval_interval) */
  sovitsSaveEverySteps: number;
  /** G_/D_ checkpoints kept on disk (upstream keep_ckpts) */
  sovitsKeepCkpts: number;
  sovitsFp16: boolean;
  /** 响度嵌入 — 4.1 only (couples vol_embedding + vol_aug like upstream --vol_aug) */
  sovitsVolEmbedding: boolean;
  /** resample 响度归一 — upstream default ON, ours OFF (lossy per upstream README) */
  sovitsLoudnorm: boolean;
  /** kmeans cluster centers instead of the retrieval matrix */
  sovitsKmeans: boolean;
  sovitsAllInMem: boolean;
  // ---- 浅扩散 sovits_diff (separate fields — card switches must not clobber;
  // no loudnorm/vol_embedding here: a diff run INHERITS them from the
  // workspace manifest, flipping them would wipe the shared caches) ----
  diffVersion: "4.1" | "4.0";
  /** completion target in global steps (diffusion progress is step-based) */
  diffTotalSteps: number;
  diffBatchSize: number;
  /** save + validation cadence in steps (upstream interval_val) */
  diffSaveEverySteps: number;
  /** milestone keep cadence; Rust normalizes to a multiple of diffSaveEverySteps */
  diffForceSaveSteps: number;
  /** 0 = full diffusion (train all 1000 t) — most capable; 100/200/300 = shallow-only */
  diffKStepMax: number;
  /** amp fp16 (upstream amp_dtype; default fp32) */
  diffFp16: boolean;
  diffCacheAllData: boolean;
}

const IDLE_SNAPSHOT: TrainingSnapshot = {
  state: "idle",
  backend: "",
  model_name: "",
  model_slug: "",
  workspace: "",
  total_epochs: 0,
  ckpts: [],
  stop_requested: false,
  elapsed_secs: 0,
  stderr_tail: [],
};

const DEFAULT_CONFIG: TrainingFormConfig = {
  modelName: "",
  backend: "rvc",
  version: "v2",
  sampleRate: "48k",
  totalEpoch: 200,
  batchSize: 6,
  saveEveryEpoch: 25,
  saveEveryWeights: true,
  keepOnlyLatest: true,
  cacheGpu: false,
  fp16: true,
  gpu: 0,
  forceCpu: false,
  sovitsVersion: "4.1",
  sovitsTotalEpoch: 1000,
  sovitsBatchSize: 6,
  sovitsSaveEverySteps: 800,
  sovitsKeepCkpts: 3,
  sovitsFp16: false,
  sovitsVolEmbedding: true,
  sovitsLoudnorm: false,
  sovitsKmeans: false,
  sovitsAllInMem: false,
  diffVersion: "4.1",
  diffTotalSteps: 100000,
  diffBatchSize: 48,
  diffSaveEverySteps: 2000,
  diffForceSaveSteps: 10000,
  diffKStepMax: 0,
  diffFp16: false,
  diffCacheAllData: true,
};

/** Client-side mirror of the Rust history cap: thin to half when exceeded. */
const HISTORY_CAP = 40000;

interface TrainingStoreState {
  snapshot: TrainingSnapshot;
  /** Wall-clock ms when `snapshot` was received — RunStep's elapsed ticker extrapolates from it. */
  snapshotAt: number;
  history: StepPoint[];
  dataset: DatasetFile[];
  config: TrainingFormConfig;
  wizard: 1 | 2 | 3 | 4;
  starting: boolean;

  setWizard: (w: 1 | 2 | 3 | 4) => void;
  updateConfig: (u: Partial<TrainingFormConfig>) => void;
  addFiles: (paths: string[]) => Promise<void>;
  removeFile: (path: string) => void;
  refresh: () => Promise<void>;
  start: (fresh: boolean) => Promise<void>;
  stop: () => Promise<void>;
  forceStop: () => Promise<void>;
  /** Clear the finished run's display state (snapshot + curve) back to idle.
   *  Files are untouched — the workspace stays resumable. */
  resetRun: () => Promise<void>;
}

export const useTrainingStore = create<TrainingStoreState>((set, get) => ({
  snapshot: IDLE_SNAPSHOT,
  snapshotAt: Date.now(),
  history: [],
  dataset: [],
  config: { ...DEFAULT_CONFIG },
  wizard: 1,
  starting: false,

  setWizard: (w) => set({ wizard: w }),
  updateConfig: (u) => set((s) => ({ config: { ...s.config, ...u } })),

  addFiles: async (paths) => {
    const existing = new Set(get().dataset.map((f) => f.path));
    const fresh = paths.filter((p) => !existing.has(p));
    if (!fresh.length) return;
    const probed: DatasetFile[] = await Promise.all(
      fresh.map(async (path) => {
        let durationMs: number | null = null;
        try {
          durationMs = await invoke<number>("probe_audio_duration", { path });
        } catch {
          durationMs = null;
        }
        const name = path.replace(/\\/g, "/").split("/").pop() ?? path;
        return { path, name, durationMs };
      }),
    );
    set((s) => ({ dataset: [...s.dataset, ...probed] }));
  },
  removeFile: (path) =>
    set((s) => ({ dataset: s.dataset.filter((f) => f.path !== path) })),

  refresh: async () => {
    try {
      const snapshot = await invoke<TrainingSnapshot>("get_training_status");
      const history = await invoke<StepPoint[]>("get_training_history");
      set((s) => {
        // during a live run, step events may have appended points while we
        // awaited — keep the local tail newer than the fetched copy instead of
        // clobbering it (it re-syncs fully at done anyway)
        const lastFetched = history.length ? history[history.length - 1]!.step : -1;
        const localTail = s.history.filter((p) => p.step > lastFetched);
        return {
          snapshot,
          history: localTail.length ? [...history, ...localTail] : history,
          snapshotAt: Date.now(),
        };
      });
    } catch (e) {
      console.error("training refresh failed", e);
    }
  },

  start: async (fresh) => {
    const { config, dataset } = get();
    set({ starting: true });
    try {
      const base = {
        model_name: config.modelName.trim(),
        backend: config.backend,
        dataset_files: dataset.map((f) => f.path),
        gpu: config.gpu,
        force_cpu: config.forceCpu,
        spk_id: 0,
        fresh,
      };
      const request =
        config.backend === "rvc"
          ? {
              ...base,
              version: config.version,
              sample_rate: config.sampleRate,
              total_epoch: config.totalEpoch,
              batch_size: config.batchSize,
              save_every_epoch: config.saveEveryEpoch,
              save_every_weights: config.saveEveryWeights,
              keep_only_latest: config.keepOnlyLatest,
              cache_gpu: config.cacheGpu,
              fp16: config.fp16,
            }
          : config.backend === "sovits_diff"
            ? {
                ...base,
                version: config.diffVersion,
                sample_rate: "44k",
                // sentinel: diffusion progress is step-based; the UI hides
                // epoch displays when total_epochs is 0
                total_epoch: 0,
                batch_size: config.diffBatchSize,
                save_every_steps: config.diffSaveEverySteps,
                total_steps: config.diffTotalSteps,
                k_step_max: config.diffKStepMax,
                interval_force_save: config.diffForceSaveSteps,
                cache_all_data: config.diffCacheAllData,
                fp16: config.diffFp16,
              }
            : {
              ...base,
              version: config.sovitsVersion,
              sample_rate: "44k",
              total_epoch: config.sovitsTotalEpoch,
              batch_size: config.sovitsBatchSize,
              save_every_steps: config.sovitsSaveEverySteps,
              keep_ckpts: config.sovitsKeepCkpts,
              fp16: config.sovitsFp16,
              // 响度嵌入 is a 4.1 feature — the 4.0 card trains ecosystem-
              // compatible checkpoints, so it stays structurally off there
              vol_embedding:
                config.sovitsVersion === "4.1" ? config.sovitsVolEmbedding : false,
              loudnorm: config.sovitsLoudnorm,
              kmeans: config.sovitsKmeans,
              all_in_mem: config.sovitsAllInMem,
            };
      await invoke("start_training", { request });
      set({ wizard: 4, history: [] });
      useAppStore.getState().showToast(i18n.t("training.started"), "info");
      await get().refresh();
    } catch (e) {
      useAppStore.getState().showToast(String(e), "error");
      throw e;
    } finally {
      set({ starting: false });
    }
  },

  stop: async () => {
    try {
      await invoke("stop_training");
      set((s) => ({ snapshot: { ...s.snapshot, stop_requested: true } }));
    } catch (e) {
      useAppStore.getState().showToast(String(e), "error");
    }
  },

  forceStop: async () => {
    try {
      await invoke("force_stop_training");
    } catch (e) {
      useAppStore.getState().showToast(String(e), "error");
    }
  },

  resetRun: async () => {
    try {
      await invoke("reset_training_display");
      // only clear locally once the backend agreed (it refuses while running)
      set({ snapshot: IDLE_SNAPSHOT, history: [], snapshotAt: Date.now() });
    } catch (e) {
      useAppStore.getState().showToast(String(e), "error");
    }
  },
}));

let unlistens: UnlistenFn[] | null = null;
let installing = false;

/** Idempotent global listener install (App mount) — keeps the titlebar indicator
 *  and the loss history live even while the training page is closed. The sync
 *  sentinel closes the await window (StrictMode double-mount would double-install
 *  and duplicate every history point + toast). */
export async function setupTrainingListeners() {
  if (unlistens || installing) return;
  installing = true;
  unlistens = await Promise.all([
    listen<StageInfo>("training-stage", (e) => {
      useTrainingStore.setState((s) => ({
        snapshot: { ...s.snapshot, stage: e.payload },
      }));
    }),
    listen<StepInfo>("training-step", (e) => {
      useTrainingStore.setState((s) => {
        let history = s.history;
        if (history.length >= HISTORY_CAP) {
          history = history.filter((_, i) => i % 2 === 0);
        }
        // NB: snapshotAt is NOT touched here — it anchors the elapsed
        // extrapolation to the last full refresh (elapsed_secs base); resetting
        // it per step would freeze the displayed elapsed at the base value
        return {
          snapshot: { ...s.snapshot, state: "running", step: e.payload },
          history: [
            ...history,
            { step: e.payload.step, lr: e.payload.lr, losses: e.payload.losses },
          ],
        };
      });
    }),
    listen<CkptInfo>("training-ckpt", (e) => {
      useTrainingStore.setState((s) => {
        const kept =
          e.payload.kind === "best" || e.payload.kind === "final"
            ? s.snapshot.ckpts.filter((c) => c.kind !== e.payload.kind)
            : s.snapshot.ckpts;
        return { snapshot: { ...s.snapshot, ckpts: [...kept, e.payload] } };
      });
    }),
    listen<TrainingSnapshot>("training-done", (e) => {
      useTrainingStore.setState({ snapshot: e.payload, snapshotAt: Date.now() });
      const t = i18n.t.bind(i18n);
      const app = useAppStore.getState();
      if (e.payload.state === "completed") {
        app.showToast(t("training.doneCompleted"), "success");
      } else if (e.payload.state === "stopped") {
        app.showToast(t("training.doneStopped"), "info");
      } else if (e.payload.state === "error") {
        app.showToast(`${t("training.doneError")}: ${e.payload.error ?? ""}`, "error");
      }
      // the final force-emitted step may have landed Rust-side only — resync once
      void useTrainingStore.getState().refresh();
    }),
    listen<string>("training-state", (e) => {
      if (e.payload === "running") {
        useTrainingStore.setState((s) => ({
          snapshot: { ...s.snapshot, state: "running" },
        }));
      }
    }),
  ]);
}
