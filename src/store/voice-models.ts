import { create } from "zustand";
import { invoke } from "@tauri-apps/api/core";

/**
 * Faithful TS mirror of Rust `ModelConfig` in src-tauri/src/models/mod.rs (the model's sidecar
 * .json, written by converter/convert.py at import time). All fields optional/additive: the
 * backend fills serde defaults today, and a parallel backend agent may grow the shape — never
 * hard-depend on presence.
 */
export interface VoiceModelConfig {
  /** "rvc" | "sovits" (converter json `type`; may be empty on hand-written jsons). */
  type?: string;
  /** RVC: "v1"/"v2"; SoVITS: "4.0"/"4.1". Serde default "unknown" when the json lacks it. */
  version?: string;
  sample_rate?: number;
  /** ContentVec/HuBERT feature width — 768 = RVC v2 / SoVITS 4.0-768, 256 = RVC v1 / SoVITS 4.0. */
  features_dim?: number;
  n_speakers?: number;
  /** Speaker NAME → id map (Rust BTreeMap<String,u32>). NOT a list — and NOT the emb_g table
   * size (that's n_speakers, which is huge/phantom for typical single-speaker models). */
  /** Speaker display names; serde default ["default"]. */
  speakers?: Record<string, number>;
  /** SoVITS S36: automatic-f0 predictor export info written by the converter when the
   * checkpoint carries f0_decoder weights: { available: boolean, file?: string, inputs?: string[] }. */
  auto_f0?: { available?: boolean; file?: string; inputs?: string[] };
  /** serde(flatten)-ed extras from the json pass through untyped. */
  [extra: string]: unknown;
}

/**
 * Faithful TS mirror of Rust `ModelEntry` in src-tauri/src/models/mod.rs — the `list_models`
 * payload. Keep in sync when the Rust struct changes.
 */
export interface VoiceModelEntry {
  name: string;
  /** Serde enum variant name: "Rvc" | "SoVits" | "S2H" | "F0" | "NsfHifigan". */
  model_type: string;
  /** "Onnx" | "Pth". */
  format: string;
  /** Absolute path to the .onnx (Unicode-safe — may contain Chinese/Japanese). */
  path: string;
  sample_rate: number;
  config?: VoiceModelConfig;
  /** RVC: converted KNN index (.npy). Also the SoVITS cluster/index asset when present. */
  index_path: string | null;
  /** SoVITS S36: `<stem>.diffusion/` attachment dir (encoder+denoiser onnx + diffusion.json)
   * when the model has a converted shallow-diffusion model — gates the 浅扩散 UI. */
  diffusion_path?: string | null;
  avatar_path: string | null;
}

export type VoiceType = "rvc" | "sovits";

interface VoiceModelStore {
  models: Record<VoiceType, VoiceModelEntry[]>;
  error: string | null;

  /** Refresh BOTH lists (backend scan() runs inside list_models, so this picks up disk changes). */
  fetchModels: () => Promise<void>;
  deleteModel: (name: string) => Promise<void>;
  setAvatar: (name: string, avatarPath: string) => Promise<void>;
  clearError: () => void;
}

export const useVoiceModelStore = create<VoiceModelStore>((set, get) => ({
  models: { rvc: [], sovits: [] },
  error: null,

  fetchModels: async () => {
    try {
      const [rvc, sovits] = await Promise.all([
        invoke<VoiceModelEntry[]>("list_models", { modelType: "rvc" }),
        invoke<VoiceModelEntry[]>("list_models", { modelType: "sovits" }),
      ]);
      set({ models: { rvc, sovits } });
    } catch (e) {
      set({ error: String(e) });
    }
  },

  deleteModel: async (name) => {
    try {
      await invoke("delete_model", { name });
      await get().fetchModels();
    } catch (e) {
      set({ error: String(e) });
    }
  },

  setAvatar: async (name, avatarPath) => {
    try {
      await invoke("set_model_avatar", { name, avatarPath });
      await get().fetchModels();
    } catch (e) {
      set({ error: String(e) });
    }
  },

  clearError: () => set({ error: null }),
}));

/**
 * Display version badge: RVC "v1"/"v2", SoVITS "4.0"/"4.1" — straight from the model json's
 * `version`; RVC falls back to the features_dim heuristic (768 = v2, 256 = v1) for jsons
 * without one. Null = render no badge (never show "unknown").
 */
export function voiceVersionBadge(m: VoiceModelEntry): string | null {
  const v = m.config?.version;
  if (v && v !== "unknown") return v;
  if (m.model_type === "Rvc") {
    if (m.config?.features_dim === 768) return "v2";
    if (m.config?.features_dim === 256) return "v1";
  }
  return null;
}

/**
 * Speaker dropdown options — EMPTY unless the model really is multi-speaker, driven by the
 * `speakers` NAME→id map (NOT n_speakers: that's the emb_g embedding-table row count, e.g. 109
 * for a single-speaker RVC voice / 200 for akiko — selecting a phantom id gathers an UNTRAINED
 * embedding row → silent garbage). RVC sidecars carry no map → single-speaker → no dropdown.
 * Options are sorted by id and labelled with the real speaker name.
 */
export function voiceSpeakerOptions(m: VoiceModelEntry): { id: number; label: string }[] {
  const map = m.config?.speakers;
  if (!map || typeof map !== "object" || Array.isArray(map)) return [];
  const entries = Object.entries(map).map(([label, id]) => ({ id: Number(id), label }));
  if (entries.length <= 1) return []; // single real speaker → default 0, no dropdown
  return entries.sort((a, b) => a.id - b.id);
}

/** "40kHz" / "44.1kHz" — shared by the node meta rows and the resource manager list. */
export function formatSampleRateKhz(sr: number): string {
  return sr % 1000 === 0 ? `${sr / 1000}kHz` : `${(sr / 1000).toFixed(1)}kHz`;
}

/** Whether the model has a converted `.diffusion/` attachment (gates 浅扩散/仅扩散 UI). */
export function voiceHasDiffusion(m: VoiceModelEntry | undefined): boolean {
  return !!m?.diffusion_path;
}

/** Whether the model's export includes the automatic-f0 predictor (gates 自动音高预测 UI).
 * Truth source = converter-written sidecar key (weight-derived), NOT the model config. */
export function voiceHasAutoF0(m: VoiceModelEntry | undefined): boolean {
  return m?.config?.auto_f0?.available === true;
}
