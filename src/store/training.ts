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

/** ①c multi-speaker co-training: one co-trained singer = a display name + its
 *  own files. Drives the SoVITS card's singer list (1 singer = single-speaker,
 *  the degenerate case). `id` is a stable React key (names may be blank /
 *  duplicate mid-edit). */
export interface SpeakerGroupDraft {
  id: string;
  name: string;
  files: DatasetFile[];
}

let spkSeq = 0;
let flashSeq = 0;

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
  /** manifest 数据增强份数 (S41) — what a diff run will inherit */
  aug_copies: number;
  /** a reusable shared slice pool exists — diff may start without importing */
  has_dataset: boolean;
  /** ①c resume config-diff: manifest vol_embedding (SoVITS); null when absent/not-sovits */
  vol_embedding: boolean | null;
  /** ①c: manifest n_speakers (multi-speaker); 1 when single-speaker */
  n_speakers: number;
  /** ①c: ordered speaker display names (index = emb_g id); empty for single-speaker */
  speakers: string[];
  /** ①c: manifest diff_k_step_max (sovits_diff); 0 when absent */
  diff_k_step_max: number;
}

/** S41 共享池模式 — THE single predicate for "a diff run may start without
 *  importing data" (root tab gating, DataStep next button, RunStep start
 *  guard all share it; Rust start_training re-verifies authoritatively). */
export function diffPoolReady(backend: string, info: WorkspaceInfo | null): boolean {
  return (
    backend === "sovits_diff" &&
    !!info?.exists &&
    info.family === "sovits" &&
    info.has_dataset
  );
}

/** ①c: which backends take a SINGER LIST (multi-speaker co-train) — SoVITS (α) + RVC (α′).
 *  THE single source for the DataStep singer-list gating so the store + page never drift.
 *  Shallow-diffusion / vocoder stay flat-dataset (their loaders assume one speaker). */
export function backendSupportsMultiSpeaker(backend: string): boolean {
  return backend === "sovits" || backend === "rvc";
}

/** THE single predicate for "step 2 (data) is satisfied" — shared by the root
 *  wizard gating (step3Ok) AND the DataStep next button so they never drift.
 *  ①c: SoVITS/RVC data is a SINGER LIST (default 1 singer = single-speaker, the
 *  degenerate case of N). Every singer needs files; with ≥2 singers each also
 *  needs a (unique) name. Other backends keep the flat-dataset / shared-pool rule. */
export function trainingDataOk(
  backend: string,
  dataset: DatasetFile[],
  speakerGroups: SpeakerGroupDraft[],
  diffPool: boolean,
): boolean {
  if (backendSupportsMultiSpeaker(backend)) {
    const allHaveFiles =
      speakerGroups.length > 0 && speakerGroups.every((g) => g.files.length > 0);
    if (speakerGroups.length <= 1) return allHaveFiles;
    // ≥2 singers: each needs a name, and names must be UNIQUE (the Rust side
    // hard-rejects duplicates — gate here so Next/Start never advertise a state
    // that would only fail with an error toast)
    const names = speakerGroups.map((g) => g.name.trim());
    const allNamed = names.every((n) => n !== "");
    const uniqueNames = new Set(names).size === names.length;
    return allHaveFiles && allNamed && uniqueNames;
  }
  return dataset.length > 0 || diffPool;
}

/** Probe duration + derive the display name for a batch of picked paths.
 *  Single source for the flat dataset add AND the per-speaker add. */
async function probeFiles(paths: string[]): Promise<DatasetFile[]> {
  return Promise.all(
    paths.map(async (path) => {
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
  /** ①c: ordered speaker display names for a multi-speaker run (index = emb_g id); empty for
   *  single-speaker. Reflects the RUN (frozen at start), used by the audition speaker picker. */
  speakers?: string[];
}

export interface TrainingFormConfig {
  modelName: string;
  backend: "rvc" | "sovits" | "sovits_diff" | "vocoder";
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
  /** S41 PSOLA 数据增强份数 (0-3, 0=off) — rvc card (per-card fields so
   *  switching cards never clobbers; diff has NO field: it inherits the
   *  workspace manifest's like loudnorm/vol_embedding) */
  augCopies: number;
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
  /** S41 PSOLA 数据增强份数 (0-3, 0=off) — sovits card; the value a later
   *  diff run inherits via the workspace manifest */
  sovitsAugCopies: number;
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
  // ---- 声码器微调 vocoder (S40; separate fields — card switches must not
  // clobber). Steps are REAL optimizer rounds (the lightning GAN counts D+G
  // separately internally — sidecar handles the 2× mapping). ----
  /** completion target in REAL steps (official guidance: ~2000 finishes a fine-tune) */
  vocTotalSteps: number;
  /** save + validation cadence in REAL steps */
  vocSaveEverySteps: number;
  vocBatchSize: number;
  /** workspace lightning checkpoints kept (weights/ snapshots are never pruned) */
  vocKeepCkpts: number;
  /** dataset crop window in mel frames (32 = upstream 16G preset, 48 = 24G) */
  vocCropMelFrames: number;
  /** freeze the MPD discriminator (upstream README: may help small-step fine-tunes) */
  vocFreezeMpd: boolean;
  /** S41 PSOLA 数据增强份数 (0-3, 0=off) — vocoder card */
  vocAugCopies: number;
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
  augCopies: 0,
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
  sovitsAugCopies: 0,
  diffVersion: "4.1",
  diffTotalSteps: 100000,
  diffBatchSize: 48,
  diffSaveEverySteps: 2000,
  diffForceSaveSteps: 10000,
  diffKStepMax: 0,
  diffFp16: false,
  diffCacheAllData: true,
  vocTotalSteps: 2000,
  vocSaveEverySteps: 500,
  vocBatchSize: 8,
  vocKeepCkpts: 5,
  vocCropMelFrames: 32,
  vocFreezeMpd: false,
  vocAugCopies: 0,
};

/** Client-side mirror of the Rust history cap: thin to half when exceeded. */
const HISTORY_CAP = 40000;

interface TrainingStoreState {
  snapshot: TrainingSnapshot;
  /** Wall-clock ms when `snapshot` was received — RunStep's elapsed ticker extrapolates from it. */
  snapshotAt: number;
  history: StepPoint[];
  dataset: DatasetFile[];
  /** ①c: the SoVITS card's singer list (always ≥1; 1 = single-speaker). */
  speakerGroups: SpeakerGroupDraft[];
  config: TrainingFormConfig;
  wizard: 1 | 2 | 3 | 4;
  starting: boolean;
  /** workspace info for the CURRENT diff host pick (null when backend≠diff or
   *  no pick) — fetched by the TrainingPage root effect, consumed everywhere
   *  via diffPoolReady() */
  diffWsInfo: WorkspaceInfo | null;

  setWizard: (w: 1 | 2 | 3 | 4) => void;
  setDiffWsInfo: (info: WorkspaceInfo | null) => void;
  updateConfig: (u: Partial<TrainingFormConfig>) => void;
  addFiles: (paths: string[]) => Promise<void>;
  removeFile: (path: string) => void;
  addSpeaker: () => void;
  removeSpeaker: (id: string) => void;
  setSpeakerName: (id: string, name: string) => void;
  addSpeakerFiles: (id: string, paths: string[]) => Promise<void>;
  removeSpeakerFile: (id: string, path: string) => void;
  /** ①c: which singer card a file-drag is currently over (highlight target);
   *  null = none. Set by the drag handler, read by the DataStep cards. */
  dragOverSpeakerId: string | null;
  setDragOverSpeakerId: (id: string | null) => void;
  /** ①c: transient "files were just added to this singer" pulse — id + nonce so
   *  the SAME singer re-flashes on repeat adds (the nonce forces the animated
   *  node to remount). Only set with ≥2 singers; cleared on animation end. */
  flashSpeaker: { id: string; nonce: number } | null;
  /** clear the pulse, but only if `nonce` still matches — so a stale animationend
   *  from one singer cannot wipe a pulse just started on another. */
  clearFlashSpeaker: (nonce: number) => void;
  /** ①c: move the imported file list between the flat dataset (rvc/vocoder/diff)
   *  and the first singer (sovits) when the training object changes, so a switch
   *  never leaves already-imported files stranded/invisible. */
  migrateOnBackendSwitch: (prev: string, next: string) => void;
  refresh: () => Promise<void>;
  start: (fresh: boolean) => Promise<void>;
  stop: () => Promise<void>;
  forceStop: () => Promise<void>;
  /** Clear the finished run's display state (snapshot + curve) back to idle.
   *  Files are untouched — the workspace stays resumable. Resolves true only
   *  when the backend accepted (it refuses while running / audition in
   *  flight) — the caller's wizard jump must not fire on a refused clear. */
  resetRun: () => Promise<boolean>;
}

export const useTrainingStore = create<TrainingStoreState>((set, get) => ({
  snapshot: IDLE_SNAPSHOT,
  snapshotAt: Date.now(),
  history: [],
  dataset: [],
  // ①c: always ≥1 singer (default 1 = single-speaker). removeSpeaker keeps ≥1.
  speakerGroups: [{ id: `spk${++spkSeq}`, name: "", files: [] }],
  dragOverSpeakerId: null,
  flashSpeaker: null,
  config: { ...DEFAULT_CONFIG },
  wizard: 1,
  starting: false,
  diffWsInfo: null,

  setWizard: (w) => set({ wizard: w }),
  setDiffWsInfo: (info) => set({ diffWsInfo: info }),
  updateConfig: (u) => set((s) => ({ config: { ...s.config, ...u } })),

  addFiles: async (paths) => {
    const existing = new Set(get().dataset.map((f) => f.path));
    const fresh = paths.filter((p) => !existing.has(p));
    if (!fresh.length) return;
    const probed = await probeFiles(fresh);
    set((s) => ({ dataset: [...s.dataset, ...probed] }));
  },
  removeFile: (path) =>
    set((s) => ({ dataset: s.dataset.filter((f) => f.path !== path) })),

  addSpeaker: () =>
    set((s) => ({
      speakerGroups: [
        ...s.speakerGroups,
        { id: `spk${++spkSeq}`, name: "", files: [] },
      ],
    })),
  removeSpeaker: (id) =>
    set((s) =>
      // keep ≥1 singer (the single-speaker degenerate case) — the last group
      // is not removable
      s.speakerGroups.length > 1
        ? { speakerGroups: s.speakerGroups.filter((g) => g.id !== id) }
        : {},
    ),
  setSpeakerName: (id, name) =>
    set((s) => ({
      speakerGroups: s.speakerGroups.map((g) =>
        g.id === id ? { ...g, name } : g,
      ),
    })),
  addSpeakerFiles: async (id, paths) => {
    const grp = get().speakerGroups.find((g) => g.id === id);
    if (!grp) return;
    const existing = new Set(grp.files.map((f) => f.path));
    const fresh = paths.filter((p) => !existing.has(p));
    if (!fresh.length) return;
    const probed = await probeFiles(fresh);
    // the group may have been removed during the async probe — bail rather than
    // discard the files (and leave a flash pointing at a dead id)
    if (!get().speakerGroups.some((g) => g.id === id)) return;
    set((s) => {
      const speakerGroups = s.speakerGroups.map((g) =>
        g.id === id ? { ...g, files: [...g.files, ...probed] } : g,
      );
      return {
        speakerGroups,
        // pulse the target card — only with ≥2 singers (a lone singer has no
        // card to pulse; setting it would leave an uncleared flash that fires
        // late when a second singer is added)
        flashSpeaker:
          speakerGroups.length > 1 ? { id, nonce: ++flashSeq } : s.flashSpeaker,
      };
    });
  },
  removeSpeakerFile: (id, path) =>
    set((s) => ({
      speakerGroups: s.speakerGroups.map((g) =>
        g.id === id
          ? { ...g, files: g.files.filter((f) => f.path !== path) }
          : g,
      ),
    })),
  setDragOverSpeakerId: (id) => set({ dragOverSpeakerId: id }),
  clearFlashSpeaker: (nonce) =>
    set((s) => (s.flashSpeaker?.nonce === nonce ? { flashSpeaker: null } : {})),
  migrateOnBackendSwitch: (prev, next) =>
    set((s) => {
      // ①c: SoVITS (α) + RVC (α′) both use the singer list; diff/vocoder use the flat dataset.
      const wasMulti = backendSupportsMultiSpeaker(prev);
      const isMulti = backendSupportsMultiSpeaker(next);
      if (isMulti === wasMulti) return {}; // both singer-list / both flat: same list, no migration
      // only ever migrate a LONE singer (never clobber a real multi-singer setup)
      const g0 = s.speakerGroups[0];
      if (!g0 || s.speakerGroups.length !== 1) return {};
      if (isMulti) {
        // flat dataset -> the (empty default) first singer
        if (s.dataset.length > 0 && g0.files.length === 0) {
          return { dataset: [], speakerGroups: [{ ...g0, files: s.dataset }] };
        }
      } else if (s.dataset.length === 0 && g0.files.length > 0) {
        // leaving a singer-list backend with a lone singer -> the (empty) flat dataset
        return { dataset: g0.files, speakerGroups: [{ ...g0, files: [] }] };
      }
      return {};
    }),

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
    const { config, dataset, speakerGroups } = get();
    set({ starting: true });
    try {
      // ①c: SoVITS (α) + RVC (α′) data is the singer list. 1 singer = single-speaker: send its
      // files as the flat dataset_files, NO `speakers` key -> byte-identical to pre-①c. ≥2
      // singers: send `speakers` (matching Rust StartTrainingRequest.speakers:
      // Vec<SpeakerGroup{name,files}>) + empty dataset_files. diff/vocoder keep the flat `dataset`.
      const isMulti = backendSupportsMultiSpeaker(config.backend);
      const multi = isMulti && speakerGroups.length > 1;
      const datasetFiles = multi
        ? []
        : isMulti
          ? (speakerGroups[0]?.files ?? []).map((f) => f.path)
          : dataset.map((f) => f.path);
      const base = {
        model_name: config.modelName.trim(),
        backend: config.backend,
        dataset_files: datasetFiles,
        ...(multi
          ? {
              speakers: speakerGroups.map((g) => ({
                name: g.name.trim(),
                files: g.files.map((f) => f.path),
              })),
            }
          : {}),
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
              aug_copies: config.augCopies,
            }
          : config.backend === "vocoder"
            ? {
                ...base,
                // fixed markers, not user choices (一期单格式类); total_epoch 0
                // = the step-based sentinel (the UI hides epoch displays)
                version: "nsf_hifigan",
                sample_rate: "44k",
                total_epoch: 0,
                batch_size: config.vocBatchSize,
                total_steps: config.vocTotalSteps,
                save_every_steps: config.vocSaveEverySteps,
                keep_ckpts: config.vocKeepCkpts,
                crop_mel_frames: config.vocCropMelFrames,
                freeze_mpd: config.vocFreezeMpd,
                aug_copies: config.vocAugCopies,
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
              aug_copies: config.sovitsAugCopies,
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
      return true;
    } catch (e) {
      useAppStore.getState().showToast(String(e), "error");
      return false;
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
