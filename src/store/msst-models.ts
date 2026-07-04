import { create } from "zustand";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { type MirrorSource, DEFAULT_MIRROR, applyMirror, type MsstCatalogEntry, type MsstPrecision } from "../lib/models/msst-catalog";
import { loadSetting, saveSetting } from "../lib/settings";

export interface InstalledModel {
  filename: string;
  size: number;
  architecture: string;
  /** fp32 `<stem>.onnx` exists. */
  has_onnx: boolean;
  /** fp16 `<stem>.fp16.onnx` sibling exists (never listed as its own row). */
  has_fp16: boolean;
  /** TRUE output order from the model json — ports/lane labels must follow this, not catalog lists. */
  stem_names?: string[] | null;
  /** Residual (mix-minus-stem) label for single-stem models — the LAST output port. */
  residual_name?: string | null;
  /** The model json's ACTUAL num_overlap (from its training yaml). The overlap slider's
   *  display default must use this; MSST_DEFAULT_NUM_OVERLAP is only the pre-install fallback
   *  (it lies for models whose yaml differs, e.g. Kim family yaml=2 vs arch default 4). */
  num_overlap?: number | null;
}

interface DownloadState {
  filename: string;
  downloaded: number;
  total: number;
  stage: "download" | "converting";
}

interface MsstModelStore {
  installed: InstalledModel[];
  modelsDir: string;
  downloading: Record<string, DownloadState>;
  error: string | null;
  mirror: MirrorSource;

  fetchInstalled: () => Promise<void>;
  fetchModelsDir: () => Promise<void>;
  downloadEntry: (entry: MsstCatalogEntry, precision?: MsstPrecision) => Promise<void>;
  downloadUrl: (url: string, filename: string, precision?: MsstPrecision, architecture?: string) => Promise<void>;
  convertPrecision: (filename: string, precision?: MsstPrecision, architecture?: string) => Promise<void>;
  deleteModel: (filename: string) => Promise<void>;
  importLocal: (path: string) => Promise<void>;
  clearError: () => void;
  setMirror: (mirror: MirrorSource) => void;
  updateDownloadProgress: (filename: string, downloaded: number, total: number, stage?: string) => void;
  removeDownload: (filename: string) => void;
}

export const useMsstModelStore = create<MsstModelStore>((set, get) => ({
  installed: [],
  modelsDir: "",
  downloading: {},
  error: null,
  mirror: loadSetting<MirrorSource>("utai.mirror", DEFAULT_MIRROR),

  fetchInstalled: async () => {
    try {
      const models = await invoke<InstalledModel[]>("list_msst_models");
      set({ installed: models, error: null });
    } catch (e) {
      set({ error: String(e) });
    }
  },

  fetchModelsDir: async () => {
    try {
      const dir = await invoke<string>("get_msst_models_dir");
      set({ modelsDir: dir });
    } catch (e) {
      set({ error: String(e) });
    }
  },

  downloadEntry: async (entry, precision) => {
    const mirror = get().mirror;
    // The original yaml must land BEFORE the ckpt and be named <ckpt stem>.yaml:
    // the ckpt download auto-converts on completion, and the converter reads the SIBLING
    // yaml (chunk/overlap, stem labels). The URL basename can differ from the ckpt name
    // (e.g. config_melbandroformer_inst_v2.yaml), so always rename to the ckpt stem.
    if (entry.configUrl) {
      try {
        const cfgUrl = applyMirror(entry.configUrl, mirror);
        const stem = entry.filename.replace(/\.[^.]+$/, "");
        await get().downloadUrl(cfgUrl, `${stem}.yaml`);
      } catch {
        // config download failure is non-fatal
      }
    }
    const url = applyMirror(entry.downloadUrl, mirror);
    // precision/architecture only apply to the MODEL download (auto-convert target), never the
    // yaml sidecar. Passing the catalog architecture matters for hash-named official weights
    // (e.g. demucs 5c90dfd2-34c22ccb.th) that Rust's name detection cannot classify.
    await get().downloadUrl(url, entry.filename, precision, entry.architecture);
  },

  downloadUrl: async (url, filename, precision, architecture) => {
    set((s) => ({
      downloading: {
        ...s.downloading,
        [filename]: { filename, downloaded: 0, total: 0, stage: "download" as const },
      },
      error: null,
    }));

    try {
      await invoke("download_msst_model", { url, filename, precision, architecture });
      set((s) => {
        const { [filename]: _, ...rest } = s.downloading;
        return { downloading: rest };
      });
      await get().fetchInstalled();
    } catch (e) {
      set((s) => {
        const { [filename]: _, ...rest } = s.downloading;
        return { downloading: rest, error: String(e) };
      });
    }
  },

  // 补转/convert an INSTALLED model to the given precision (undefined = full fp32 export from
  // ckpt). fp16 with the fp32 onnx on disk is the fast post-hoc path (~1-2 min); fp32 (or fp16
  // without the fp32 file) is a full re-export. Reuses the download record's "converting" stage
  // so the existing DownloadBar / indicators track it.
  convertPrecision: async (filename, precision, architecture) => {
    set((s) => ({
      downloading: {
        ...s.downloading,
        [filename]: { filename, downloaded: 0, total: 0, stage: "converting" as const },
      },
      error: null,
    }));
    try {
      await invoke("convert_msst_model", { filename, precision, architecture });
      await get().fetchInstalled();
    } catch (e) {
      set({ error: String(e) });
    }
    get().removeDownload(filename);
  },

  deleteModel: async (filename) => {
    try {
      await invoke("delete_msst_model", { filename });
      await get().fetchInstalled();
    } catch (e) {
      set({ error: String(e) });
    }
  },

  importLocal: async (path) => {
    try {
      await invoke("import_local_msst_model", { sourcePath: path });
      await get().fetchInstalled();
    } catch (e) {
      set({ error: String(e) });
    }
  },

  clearError: () => set({ error: null }),

  setMirror: (mirror) => { saveSetting("utai.mirror", mirror); set({ mirror }); },

  updateDownloadProgress: (filename, downloaded, total, stage) =>
    set((s) => ({
      downloading: {
        ...s.downloading,
        [filename]: {
          filename,
          downloaded,
          total,
          stage: (stage === "converting" ? "converting" : "download") as DownloadState["stage"],
        },
      },
    })),

  removeDownload: (filename) =>
    set((s) => {
      const { [filename]: _, ...rest } = s.downloading;
      return { downloading: rest };
    }),
}));

let progressUnlisten: UnlistenFn | null = null;

export async function setupDownloadListener() {
  if (progressUnlisten) return;
  progressUnlisten = await listen<{ filename: string; downloaded: number; total: number; stage?: string }>(
    "msst-download-progress",
    (event) => {
      useMsstModelStore
        .getState()
        .updateDownloadProgress(event.payload.filename, event.payload.downloaded, event.payload.total, event.payload.stage);
    },
  );
}
