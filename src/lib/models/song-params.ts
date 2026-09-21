import type { SongModelFamily } from "./song-catalog";

export interface ParamConfig {
  key: string;
  label: { zh: string; en: string };
  type: "number" | "select" | "boolean" | "text";
  default: any;
  min?: number;
  max?: number;
  step?: number;
  options?: Array<{ label: string; value: any }>;
  description?: { zh: string; en: string };
  recommended?: string;
  example?: { zh: string; en: string };
  group?: "common" | "model";
}

const YUE2_PARAMS: ParamConfig[] = [
  {
    key: "cfg_scale",
    label: { zh: "CFG强度", en: "CFG Scale" },
    type: "number",
    default: 3,
    min: 0,
    max: 20,
    step: 0.5,
    description: { zh: "控制歌曲对风格提示词的贴合程度。", en: "Controls adherence to the style prompt." },
    recommended: "3.0",
    group: "model",
  },
  {
    key: "cot",
    label: { zh: "推理模式", en: "CoT Mode" },
    type: "select",
    default: "full",
    options: [
      { label: "关闭", value: "off" },
      { label: "旋律", value: "melody" },
      { label: "完整", value: "full" },
    ],
    description: { zh: "控制 YuE2 的歌词和旋律规划方式。", en: "Controls YuE2 lyric and melody planning." },
    recommended: "完整",
    group: "model",
  },
  {
    key: "temperature",
    label: { zh: "采样温度", en: "Temperature" },
    type: "number",
    default: 1,
    min: 0,
    max: 5,
    step: 0.1,
    description: { zh: "控制生成的随机程度。", en: "Controls generation randomness." },
    recommended: "1.0",
    group: "model",
  },
  {
    key: "top_p",
    label: { zh: "Top-P", en: "Top-P" },
    type: "number",
    default: 0.95,
    min: 0,
    max: 1,
    step: 0.05,
    description: { zh: "控制采样候选范围。", en: "Controls the nucleus sampling range." },
    recommended: "0.95",
    group: "model",
  },
  {
    key: "top_k",
    label: { zh: "Top-K", en: "Top-K" },
    type: "number",
    default: 50,
    min: 1,
    max: 200,
    step: 1,
    description: { zh: "控制每一步保留的候选数量。", en: "Controls the number of candidates kept per step." },
    recommended: "50",
    group: "model",
  },
  {
    key: "repetition_penalty",
    label: { zh: "重复惩罚", en: "Repetition Penalty" },
    type: "number",
    default: 1,
    min: 0.5,
    max: 2,
    step: 0.1,
    description: { zh: "减少歌词或旋律重复。", en: "Reduces repeated lyrics or melodies." },
    recommended: "1.0",
    group: "model",
  },
  {
    key: "ode_steps",
    label: { zh: "ODE步数", en: "ODE Steps" },
    type: "number",
    default: 32,
    min: 8,
    max: 100,
    step: 1,
    description: { zh: "YuE2 音频解码的求解步数。", en: "YuE2 audio decoding solver steps." },
    recommended: "32",
    group: "model",
  },
  {
    key: "ode_method",
    label: { zh: "ODE方法", en: "ODE Method" },
    type: "select",
    default: "midpoint",
    options: [
      { label: "中点法", value: "midpoint" },
      { label: "欧拉法", value: "euler" },
    ],
    description: { zh: "选择 YuE2 音频解码方法。", en: "Selects the YuE2 audio decoding method." },
    recommended: "中点法",
    group: "model",
  },
];

const ACESTEP_PARAMS: ParamConfig[] = [
  {
    key: "vram_mode",
    label: { zh: "显存模式", en: "VRAM Mode" },
    type: "select",
    default: "auto",
    options: [
      { label: "自动检测", value: "auto" },
      { label: "手动：快速（8G）", value: "fast_8gb" },
      { label: "手动：均衡（12G）", value: "balanced_12gb" },
      { label: "手动：高质量（16G）", value: "quality_16gb" },
    ],
    description: { zh: "自动按显存选择策略，也可以手动覆盖。", en: "Automatically selects a VRAM strategy or lets you override it." },
    recommended: "自动检测",
    group: "model",
  },
  {
    key: "inference_steps",
    label: { zh: "推理步数", en: "Inference Steps" },
    type: "number",
    default: 8,
    min: 4,
    max: 8,
    step: 1,
    description: { zh: "ACE-Step Turbo 的真实扩散步数。Turbo 超过 8 步不会生效。", en: "Real ACE-Step Turbo diffusion steps. Turbo does not support values above 8." },
    recommended: "8",
    group: "model",
  },
  {
    key: "guidance_scale",
    label: { zh: "引导系数", en: "Guidance Scale" },
    type: "number",
    default: 4,
    min: 1,
    max: 10,
    step: 0.5,
    description: { zh: "控制 ACE-Step 对提示词的引导强度。Turbo 推荐 4；Base 任务会使用更高范围。", en: "Controls prompt guidance. 4 is recommended for Turbo; Base tasks support a higher range." },
    recommended: "4.0",
    group: "model",
  },
  {
    key: "audio_format",
    label: { zh: "输出格式", en: "Audio Format" },
    type: "select",
    default: "flac",
    options: [
      { label: "FLAC（无损，推荐）", value: "flac" },
      { label: "WAV", value: "wav" },
      { label: "WAV 32-bit", value: "wav32" },
      { label: "MP3", value: "mp3" },
      { label: "AAC", value: "aac" },
      { label: "Opus", value: "opus" },
    ],
    description: { zh: "由 ACE-Step 真正使用的输出格式。FLAC 适合保留最高质量。", en: "The output format actually used by ACE-Step. FLAC preserves the highest quality." },
    recommended: "FLAC",
    group: "model",
  },
  {
    key: "mp3_bitrate",
    label: { zh: "MP3码率", en: "MP3 Bitrate" },
    type: "select",
    default: "192k",
    options: [
      { label: "128 kbps", value: "128k" },
      { label: "192 kbps", value: "192k" },
      { label: "256 kbps", value: "256k" },
      { label: "320 kbps", value: "320k" },
    ],
    description: { zh: "只在输出格式为 MP3 时生效。", en: "Only applies when the output format is MP3." },
    recommended: "192 kbps",
    group: "model",
  },
  {
    key: "mp3_sample_rate",
    label: { zh: "MP3采样率", en: "MP3 Sample Rate" },
    type: "select",
    default: 48000,
    options: [
      { label: "48 kHz", value: 48000 },
      { label: "44.1 kHz", value: 44100 },
    ],
    description: { zh: "只在输出格式为 MP3 时生效，不改变模型内部生成采样率。", en: "Only applies to MP3 output and does not change the model's internal sample rate." },
    recommended: "48 kHz",
    group: "model",
  },
];

const HEARTMULA_PARAMS: ParamConfig[] = [
  {
    key: "cfg_scale",
    label: { zh: "CFG强度", en: "CFG Scale" },
    type: "number",
    default: 1.5,
    min: 1,
    max: 5,
    step: 0.1,
    description: { zh: "提示词引导强度。", en: "Prompt guidance strength." },
    recommended: "1.5",
    group: "model",
  },
  {
    key: "temperature",
    label: { zh: "采样温度", en: "Temperature" },
    type: "number",
    default: 1,
    min: 0.1,
    max: 2,
    step: 0.1,
    description: { zh: "生成随机性。", en: "Generation randomness." },
    recommended: "1.0",
    group: "model",
  },
  {
    key: "topk",
    label: { zh: "Top-K", en: "Top-K" },
    type: "number",
    default: 50,
    min: 10,
    max: 200,
    step: 10,
    description: { zh: "保留的候选数量。", en: "Number of candidates kept." },
    recommended: "50",
    group: "model",
  },
];

export const MODEL_PARAMS_MAP: Record<SongModelFamily, ParamConfig[]> = {
  yue2: YUE2_PARAMS,
  acestep: ACESTEP_PARAMS,
  heartmula: HEARTMULA_PARAMS,
};

export function getModelParams(family: SongModelFamily): ParamConfig[] {
  return MODEL_PARAMS_MAP[family] || [];
}

export function getDefaultParams(family: SongModelFamily): Record<string, any> {
  return Object.fromEntries(getModelParams(family).map((param) => [param.key, param.default]));
}
