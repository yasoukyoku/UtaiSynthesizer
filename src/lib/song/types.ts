// 歌曲制作统一类型层（规划 13.1/4.4）：
//  - SongSource：三来源统一对象（结果库 / 工程轨道 / 本地文件）
//  - SongHistoryEntry：从 SongStudioDialog.tsx 迁出的历史条目（含 task/source/capabilities 扩展，全部可选以兼容旧 JSON）
//  - deriveCapabilities / migrateHistoryEntry：纯函数 + 旧数据迁移
import type { SongOutput, SongGenRequest } from "../backendSong";
import type { SongTaskId } from "../models/song-tasks";

// ── 统一"源"对象（规划 4.2）───────────────────────────────────────────────

export type SongSource =
  | { kind: "history"; songId: string; outputIndex?: number }  // 结果库
  | { kind: "track"; trackId: string; segmentId?: string }     // 工程轨道/片段（运行时渲染为临时 WAV）
  | { kind: "file"; path: string };                            // 本地文件

// ── 历史条目 ───────────────────────────────────────────────────────────────

export interface SongHistoryCapabilities {
  hasStems: boolean;
  hasMidi: boolean;
  hasLrc: boolean;
  hasAbc: boolean;
  stemKinds: string[];
}

export interface SongHistoryEntry {
  uuid: string;
  modelId: string;
  modelFamily: "yue2" | "acestep" | "heartmula";
  license?: string;
  /** 任务模式；旧数据无此字段时迁移兜底为 "generate" */
  task?: SongTaskId;
  settings: SongGenRequest & { songName?: string };
  outputs: SongOutput[];
  timestamp: number;
  label: string;
  trackId?: string;
  /** 来源链（从哪个结果/轨道/文件再创作而来） */
  source?: { kind: "history" | "track" | "file"; ref: string; label: string; range?: [number, number] };
  /** 前端推导产物能力（保存时计算） */
  capabilities?: SongHistoryCapabilities;
  /** 失败条目记录错误信息供重试 */
  error?: string;
}

// ── capabilities 推导（唯一依据是 outputs 实际内容，不认模型静态标签）────────

export function deriveCapabilities(entry: Pick<SongHistoryEntry, "outputs">): SongHistoryCapabilities {
  let hasStems = false, hasMidi = false, hasLrc = false, hasAbc = false;
  const stemKinds = new Set<string>();
  for (const o of entry.outputs ?? []) {
    if (o.audio_path) { /* 主音频单独不算能力；stems 才算 */ }
    if (o.midi_path) hasMidi = true;
    if (o.lrc_path) hasLrc = true;
    if (o.abc_path) hasAbc = true;
    for (const kind of Object.keys(o.stems ?? {})) {
      hasStems = true;
      stemKinds.add(kind);
    }
  }
  return { hasStems, hasMidi, hasLrc, hasAbc, stemKinds: [...stemKinds] };
}

// ── 旧历史 JSON 迁移（不允许旧文件打不开）─────────────────────────────────

/** 解析 + 迁移：task 缺省 → "generate"，capabilities 现算（覆盖脏值）。返回 null 表示 JSON 非法。 */
export function migrateHistoryEntries(raw: string): SongHistoryEntry[] | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!Array.isArray(parsed)) return null;
  return (parsed as SongHistoryEntry[]).map((e) => ({
    ...e,
    task: e.task ?? "generate",
    capabilities: deriveCapabilities(e),
  }));
}
