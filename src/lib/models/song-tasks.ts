// 歌曲任务（模式）定义 —— 以"任务"为中心的统一数据层（规划 4.1）。
// 纯数据，不含 React/IPC 依赖；UI（模式条/表单）与运行器（runSongTask）都从这里取定义。
import type { I18nText } from "./msst-catalog";

export type SongTaskId =
  | "generate"      // 写歌（歌词+提示词 → 整首）
  | "instrumental"  // 纯伴奏生成
  | "cover"         // 风格翻唱
  | "repaint"       // 局部重绘（起止秒）
  | "complete"      // 音轨补全 / 人声转伴奏
  | "extract"       // ACE 原生分轨提取（逐轨）
  | "lego"          // 叠加音轨
  | "stems"         // demucs 快速四分轨
  | "sheet";        // YuE2 乐谱（人声/旋律 → ABC/MIDI）

export type SongTaskOutputKind = "audio" | "stems" | "midi" | "lrc" | "abc";

export interface SongTaskInputSpec {
  lyrics: "required" | "optional" | "none";
  prompt: "required" | "optional" | "none";
  srcAudio: "required" | "optional" | "none";
  /** repaint 需要时间区间（repaint_start/end） */
  range?: boolean;
  /** lego/extract 需要单选轨种 */
  trackClass?: boolean;
  /** complete / extract 批量：可多选轨种 */
  trackClassesMulti?: boolean;
}

export interface SongTaskDef {
  id: SongTaskId;
  icon: string;
  name: I18nText;
  tagline: I18nText;
  /** 能执行该任务的模型目录 id，按优先级排序；UI 自动选第 1 个已就绪模型 */
  models: string[];
  input: SongTaskInputSpec;
  /** 可能产出（能力声明；实际以运行结果 outputs 为准） */
  outputs: SongTaskOutputKind[];
  /** 生成按钮文案随模式变化（写进 name 之外的动作用词） */
  action: I18nText;
  longRunning?: boolean;
}

// ── 任务 → 模型映射与定义表 ────────────────────────────────────────────────
// 模型 id 对应 song-catalog.ts：yue2-3b（YuE2）、acestep-v1.5（含 turbo/xl-turbo/base 变体，
// sidecar 按 task 自动路由：cover/repaint→XL-Turbo，lego/extract/complete→Base）。

export const SONG_TASKS: Record<SongTaskId, SongTaskDef> = {
  generate: {
    id: "generate",
    icon: "✍",
    name: { zh: "生成歌曲", en: "Create", ja: "作曲" },
    tagline: { zh: "歌词 + 风格描述 → 完整歌曲", en: "Lyrics + style → full song", ja: "歌詞+スタイル → 楽曲" },
    models: ["acestep-v1.5", "yue2-3b"],
    input: { lyrics: "required", prompt: "required", srcAudio: "none" },
    outputs: ["audio", "stems", "midi", "lrc"],
    action: { zh: "生成歌曲", en: "Generate Song", ja: "楽曲を生成" },
  },
  instrumental: {
    id: "instrumental",
    icon: "🎚",
    name: { zh: "纯伴奏", en: "Instrumental", ja: "伴奏のみ" },
    tagline: { zh: "无歌词，直接生成伴奏", en: "Lyrics-free backing track", ja: "歌詞なしで伴奏を生成" },
    models: ["acestep-v1.5"],
    input: { lyrics: "none", prompt: "required", srcAudio: "none" },
    outputs: ["audio", "stems"],
    action: { zh: "生成伴奏", en: "Generate Instrumental", ja: "伴奏を生成" },
  },
  cover: {
    id: "cover",
    icon: "🎤",
    name: { zh: "翻唱", en: "Cover", ja: "カバー" },
    tagline: { zh: "保留旋律骨架，换风格/唱法", en: "Keep the melody, restyle it", ja: "メロディを保ちカバー" },
    models: ["acestep-v1.5"],
    input: { lyrics: "optional", prompt: "optional", srcAudio: "required" },
    outputs: ["audio", "stems", "lrc"],
    action: { zh: "开始翻唱", en: "Start Cover", ja: "カバー開始" },
    longRunning: true,
  },
  repaint: {
    id: "repaint",
    icon: "🖌",
    name: { zh: "局部重绘", en: "Repaint", ja: "部分リペイント" },
    tagline: { zh: "只重画选定起止秒的片段", en: "Redraw a time range", ja: "区間だけ描き直す" },
    models: ["acestep-v1.5"],
    input: { lyrics: "optional", prompt: "optional", srcAudio: "required", range: true },
    outputs: ["audio"],
    action: { zh: "重绘所选区间", en: "Repaint Range", ja: "区間をリペイント" },
    longRunning: true,
  },
  complete: {
    id: "complete",
    icon: "🧩",
    name: { zh: "音轨补全", en: "Complete", ja: "トラック補完" },
    tagline: { zh: "人声转伴奏 / 补全缺失声部", en: "Voice-to-accompaniment / fill tracks", ja: "伴奏化・音部補完" },
    models: ["acestep-v1.5"],
    input: { lyrics: "none", prompt: "optional", srcAudio: "required", trackClassesMulti: true },
    outputs: ["audio", "stems"],
    action: { zh: "开始补全", en: "Start Completion", ja: "補完を開始" },
    longRunning: true,
  },
  extract: {
    id: "extract",
    icon: "✂",
    name: { zh: "分轨", en: "Extract Stems", ja: "トラック抽出" },
    tagline: { zh: "任意音频（含外部歌曲）提取吉他/钢琴等轨", en: "Extract stems from any audio", ja: "任意の音源から抽出" },
    models: ["acestep-v1.5"],
    input: { lyrics: "none", prompt: "optional", srcAudio: "required", trackClass: true, trackClassesMulti: true },
    outputs: ["stems"],
    action: { zh: "开始分轨", en: "Extract Stems", ja: "抽出開始" },
    longRunning: true,
  },
  lego: {
    id: "lego",
    icon: "🧱",
    name: { zh: "叠加音轨", en: "Add Track (Lego)", ja: "トラック重ね" },
    tagline: { zh: "在成品上叠加一件新乐器", en: "Layer a new instrument onto a mix", ja: "新乐器を重ねる" },
    models: ["acestep-v1.5"],
    input: { lyrics: "none", prompt: "required", srcAudio: "required", trackClass: true },
    outputs: ["audio", "stems"],
    action: { zh: "生成叠加轨", en: "Layer Track", ja: "トラックを重ねる" },
    longRunning: true,
  },
  stems: {
    id: "stems",
    icon: "🎚",
    name: { zh: "快速四分轨", en: "Quick 4-Stems", ja: "簡易4トラック" },
    tagline: { zh: "demucs 一键分出人声/鼓/贝斯/其他", en: "One-click demucs 4-stem split", ja: "demucsで4分割" },
    models: ["acestep-v1.5", "yue2-3b"],
    input: { lyrics: "none", prompt: "none", srcAudio: "required" },
    outputs: ["stems"],
    action: { zh: "开始分轨", en: "Split Stems", ja: "分離開始" },
    longRunning: true,
  },
  sheet: {
    id: "sheet",
    icon: "🎹",
    name: { zh: "乐谱", en: "Sheet/MIDI", ja: "楽譜" },
    tagline: { zh: "YuE2 生成式乐谱（支持 ABC 续写）→ ABC/MIDI", en: "Generative ABC score & MIDI (supports ABC continue)", ja: "ABC/MIDI を生成" },
    models: ["yue2-3b"],
    input: { lyrics: "none", prompt: "optional", srcAudio: "optional" },
    outputs: ["abc", "midi"],
    action: { zh: "生成乐谱/MIDI", en: "Transcribe to Sheet", ja: "楽譜を生成" },
    longRunning: true,
  },
};

/** 模式条展示顺序 */
export const SONG_TASK_ORDER: SongTaskId[] = [
  "generate", "instrumental", "cover", "repaint", "complete", "extract", "lego", "stems", "sheet",
];

// ── ACE-Step 任务路由（与 song_sidecar.py ACE_TASKS 对应）─────────────────
// generate/instrumental 之外的任务在提交 songGenerate 时需要携带的 task 参数值。

export const ACE_TASK_OF: Partial<Record<SongTaskId, string>> = {
  cover: "cover",
  repaint: "repaint",
  complete: "complete",
  extract: "extract",
  lego: "lego",
};

/** extract / complete 可用的轨种（与 sidecar 类名一致，展示名走 i18n key songTrack.*） */
export const ACE_TRACK_CLASSES = [
  "vocals", "drums", "bass", "guitar", "piano", "other",
] as const;

export type AceTrackClass = (typeof ACE_TRACK_CLASSES)[number];

/** demucs 四轨固定轨种（want_stems 路径产出） */
export const DEMUCS_STEM_KINDS = ["vocals", "drums", "bass", "other"] as const;

/** 任务是否需要源音频 */
export function taskNeedsSource(task: SongTaskId): boolean {
  return SONG_TASKS[task].input.srcAudio !== "none";
}
