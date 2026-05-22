import { create } from "zustand";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { type MirrorSource, DEFAULT_MIRROR, applyMirror, type MsstCatalogEntry } from "../lib/models/msst-catalog";

export interface InstalledModel {
  filename: string;
  size: number;
  architecture: string;
  has_onnx: boolean;
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
  downloadEntry: (entry: MsstCatalogEntry) => Promise<void>;
  downloadUrl: (url: string, filename: string) => Promise<void>;
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
  mirror: DEFAULT_MIRROR,

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

  downloadEntry: async (entry) => {
    const mirror = get().mirror;
    const url = applyMirror(entry.downloadUrl, mirror);
    await get().downloadUrl(url, entry.filename);

    if (entry.configUrl) {
      try {
        const cfgUrl = applyMirror(entry.configUrl, mirror);
        const cfgFilename = entry.configUrl.split("/").pop() ?? "";
        if (cfgFilename) await get().downloadUrl(cfgUrl, cfgFilename);
      } catch {
        // config download failure is non-fatal
      }
    }
  },

  downloadUrl: async (url, filename) => {
    set((s) => ({
      downloading: {
        ...s.downloading,
        [filename]: { filename, downloaded: 0, total: 0, stage: "download" as const },
      },
      error: null,
    }));

    try {
      await invoke("download_msst_model", { url, filename });
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

  setMirror: (mirror) => set({ mirror }),

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
