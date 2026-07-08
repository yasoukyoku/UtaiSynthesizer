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
  /** ①c S46: present ONLY for a genuine multi-speaker export (the ONNX graph carries a
   * "spk_mix" input replacing scalar "sid"). `n_spk` = emb_g table width (the blend vector
   * length). Absent on single-speaker / pre-①c models — gates the δ blend stack UI. */
  spk_mix?: { available?: boolean; n_spk?: number };
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

/** "vocoder" = the S40 NSF-HiFiGAN vocoder RESOURCE class (fine-tuned/imported,
 * models/nsf_hifigan/ on disk, Rust ModelType::NsfHifigan) — shared across every
 * SoVITS model of the same singer; consumed by the SoVITS node's vocoder dropdown. */
export type VoiceType = "rvc" | "sovits" | "vocoder";

interface VoiceModelStore {
  models: Record<VoiceType, VoiceModelEntry[]>;
  error: string | null;

  /** Refresh ALL lists (backend scan() runs inside list_models, so this picks up disk changes). */
  fetchModels: () => Promise<void>;
  /** Type-scoped delete — same-name entries across types are a standard workflow
   * (an rvc+sovits pair per singer + a vocoder named after the singer), an
   * untyped delete would hit the first scan match instead (Rust 红队 A5). */
  deleteModel: (name: string, voiceType: VoiceType) => Promise<void>;
  setAvatar: (name: string, avatarPath: string) => Promise<void>;
  clearError: () => void;
}

export const useVoiceModelStore = create<VoiceModelStore>((set, get) => ({
  models: { rvc: [], sovits: [], vocoder: [] },
  error: null,

  fetchModels: async () => {
    try {
      const [rvc, sovits, vocoder] = await Promise.all([
        invoke<VoiceModelEntry[]>("list_models", { modelType: "rvc" }),
        invoke<VoiceModelEntry[]>("list_models", { modelType: "sovits" }),
        invoke<VoiceModelEntry[]>("list_models", { modelType: "vocoder" }),
      ]);
      set({ models: { rvc, sovits, vocoder } });
    } catch (e) {
      set({ error: String(e) });
    }
  },

  deleteModel: async (name, voiceType) => {
    try {
      await invoke("delete_model", { name, modelType: voiceType });
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
 * A model's ContentVec feature dim (speech_encoder wins, features_dim
 * fallback) — the pairing key for shallow-diffusion companions: the diffusion
 * card derives its training version from it and the attach flow filters
 * candidates by it. Null = unknown (let the Rust side validate).
 */
export function voiceFeatureDim(m: VoiceModelEntry): number | null {
  if (m.config?.speech_encoder === "vec768l12") return 768;
  if (m.config?.speech_encoder === "vec256l9") return 256;
  return m.config?.features_dim ?? null;
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

/** ①c: whether the model is a GENUINE multi-speaker export whose ONNX graph takes a `spk_mix`
 * blend input (converter writes `spk_mix.available` ONLY for len(speakers) > 1). This — NOT
 * merely voiceSpeakerOptions().length > 1 — gates the blend-stack UI: a pre-①c multi-speaker
 * model has a speaker map but a scalar `sid` input, so it uses the plain SpeakerSelect instead.
 * Requires ≥2 named speakers too (the blend needs real names to pick from). */
export function voiceHasSpkMix(m: VoiceModelEntry | undefined): boolean {
  return m?.config?.spk_mix?.available === true && voiceSpeakerOptions(m).length > 1;
}

/** 一期唯一声码器格式类 = the OpenVPI standard (the aux default vocoder's recipe;
 * every SoVITS diffusion attachment / the enhancer mel is anchored to it).
 * MIRRORS the Rust constants in commands/inference.rs (VOCODER_STD_*) — the Rust
 * side re-validates strictly at run time, this filter only decides visibility. */
export const VOCODER_STD_FORMAT = {
  sample_rate: 44100,
  hop_size: 512,
  n_fft: 2048,
  win_size: 2048,
  num_mels: 128,
  fmin: 40,
  fmax: 16000,
} as const;

/** Whether an installed vocoder's sidecar recipe matches the standard format class —
 * mismatches are HIDDEN from the node dropdown (不能选隐藏), the resource list
 * explains the format instead. Missing fields = unverifiable = no match. */
export function vocoderFormatMatches(m: VoiceModelEntry): boolean {
  const c = m.config;
  if (!c) return false;
  return (Object.keys(VOCODER_STD_FORMAT) as (keyof typeof VOCODER_STD_FORMAT)[]).every(
    (k) => Number(c[k]) === VOCODER_STD_FORMAT[k],
  );
}

/** "44.1kHz · hop 512 · 128 mel" — the resource-list format line for a vocoder row. */
export function vocoderFormatLabel(m: VoiceModelEntry): string {
  const c = m.config ?? {};
  const sr = typeof c.sample_rate === "number" ? c.sample_rate : m.sample_rate;
  const hop = c.hop_size ?? "?";
  const mels = (c as Record<string, unknown>).num_mels ?? "?";
  return `${formatSampleRateKhz(sr)} · hop ${hop} · ${mels} mel`;
}
