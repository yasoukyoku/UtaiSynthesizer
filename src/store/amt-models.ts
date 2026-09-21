import { create } from "zustand";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { isTauri } from "../lib/tauri";
import {
  type MirrorSource, DEFAULT_MIRROR, applyMirror,
  type GhMirror, type GhPreset, DEFAULT_GH_MIRROR, BUILTIN_GH_PRESETS,
  migrateGhMirror, ghMirrorCandidates,
} from "../lib/models/msst-catalog";
import { loadSetting } from "../lib/settings";
import { AmtCatalogEntry, AMT_CATALOG } from "../lib/models/amt-catalog";
import { useAppStore } from "./app";

export interface InstalledAmtModel {
  id: string;
  filename: string;
  size: number;
  architecture: string;
  is_installed: boolean;
  is_available: boolean;
}

interface DownloadState {
  id: string;
  filename: string;
  downloaded: number;
  total: number;
  stage: string;
}

interface AmtModelStore {
  installed: InstalledAmtModel[];
  modelsDir: string;
  downloading: Record<string, DownloadState>;
  error: string | null;
  mirror: MirrorSource;
  ghMirror: GhMirror;
  ghPresets: GhPreset[];

  fetchInstalled: () => Promise<void>;
  fetchModelsDir: () => Promise<void>;
  downloadEntry: (entry: AmtCatalogEntry) => Promise<void>;
  deleteModel: (id: string) => Promise<void>;
  clearError: () => void;
  updateDownloadProgress: (id: string, downloaded: number, total: number, stage: string) => void;
  removeDownload: (id: string) => void;
}

export const useAmtModelStore = create<AmtModelStore>((set, get) => ({
  installed: [],
  modelsDir: "",
  downloading: {},
  error: null,
  mirror: loadSetting<MirrorSource>("utai.mirror", DEFAULT_MIRROR),
  ghMirror: migrateGhMirror(loadSetting<GhMirror>("utai.ghMirror", DEFAULT_GH_MIRROR)),
  ghPresets: BUILTIN_GH_PRESETS, // Simple fallback

  fetchInstalled: async () => {
    if (!isTauri()) return; // No Rust backend in browser preview
    try {
      const models = await invoke<InstalledAmtModel[]>("list_amt_models");
      // Match with catalog to get proper IDs
      const mapped = models.map(m => {
        // Find by filename first, then by architecture for special cases
        const entry = AMT_CATALOG.find(e => 
          e.filename === m.filename || 
          (m.id && e.id === m.id) ||
          (m.architecture === e.architecture && ["fluidsynth", "soundfont"].includes(m.architecture))
        );
        return {
          ...m,
          id: entry?.id || m.id || m.filename,
          is_available: m.is_installed // use the backend's is_installed flag
        };
      });
      set({ installed: mapped, error: null });
    } catch (e) {
      set({ error: String(e) });
    }
  },


  fetchModelsDir: async () => {
    try {
      const dir = await invoke<string>("get_amt_models_dir");
      set({ modelsDir: dir });
    } catch (e) {
      set({ error: String(e) });
    }
  },

  downloadEntry: async (entry) => {
    const { mirror, ghMirror, ghPresets } = get();
    const urls = ghMirrorCandidates(applyMirror(entry.downloadUrl, mirror), ghMirror, ghPresets);

    set((s) => ({
      downloading: {
        ...s.downloading,
        [entry.id]: { id: entry.id, filename: entry.filename, downloaded: 0, total: entry.fileSize, stage: "download" },
      },
      error: null,
    }));

    try {
      await invoke("download_amt_model", {
        urls,
        id: entry.id,
        filename: entry.filename,
        architecture: entry.architecture,
        sha256: entry.sha256,
      });
      // Multi-file models (e.g. Whisper): fetch small companion files sequentially
      // under the same progress card. download_amt_model skips files already on
      // disk, so re-downloads stay cheap.
      for (const extra of entry.extraFiles ?? []) {
        await invoke("download_amt_model", {
          urls: ghMirrorCandidates(applyMirror(extra.downloadUrl, mirror), ghMirror, ghPresets),
          id: entry.id,
          filename: extra.filename,
          architecture: entry.architecture,
          sha256: null,
        });
      }
      set((s) => {
        const { [entry.id]: _, ...rest } = s.downloading;
        return { downloading: rest };
      });
      await get().fetchInstalled();
    } catch (e) {
      set((s) => {
        const { [entry.id]: _, ...rest } = s.downloading;
        return { downloading: rest, error: String(e) };
      });
      // Display the error to the user via global toast
      useAppStore.getState().showToast(String(e), "error");
    }
  },

  deleteModel: async (filename) => {
    try {
      await invoke("delete_amt_model", { filename });
      await get().fetchInstalled();
    } catch (e) {
      set({ error: String(e) });
    }
  },


  clearError: () => set({ error: null }),

  updateDownloadProgress: (id, downloaded, total, stage) =>
    set((s) => {
      const existing = s.downloading[id];
      if (!existing) return {};
      return {
        downloading: {
          ...s.downloading,
          [id]: {
            ...existing,
            downloaded,
            total,
            stage,
          },
        },
      };
    }),

  removeDownload: (id) =>
    set((s) => {
      const { [id]: _, ...rest } = s.downloading;
      return { downloading: rest };
    }),
}));

let progressUnlisten: UnlistenFn | null = null;

export async function setupAmtDownloadListener() {
  if (!isTauri()) return; // No Rust backend in browser preview
  if (progressUnlisten) return;
  progressUnlisten = await listen<{ id: string; downloaded: number; total: number; stage: string }>(
    "amt-download-progress",
    (event) => {
      useAmtModelStore
        .getState()
        .updateDownloadProgress(event.payload.id, event.payload.downloaded, event.payload.total, event.payload.stage);
    },
  );
}


