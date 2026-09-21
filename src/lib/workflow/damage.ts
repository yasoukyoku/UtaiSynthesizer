/**
 * 节点损伤等级表（规划 6-5）—— 每个节点类型必填（Record<WorkflowNodeType, …> 穷举，
 * 新增节点类型漏配直接编译报错），用于两件事：
 *   1. NodeShell 在节点卡片上挂损伤徽标（🟢/🟡/🟠/🔴），让用户一眼看到这条链的保真度；
 *   2. 引擎按 SNR 公式累计整条链的近似信噪比，显示在工作流标头：
 *        总SNR(dB) ≈ -10 · log10( Σ 10^(-SNR_i / 10) )
 *      零损伤节点（L0 = ∞）对 Σ 贡献 10^(-∞) = 0，天然不影响总 SNR。
 *
 * 等级口径与规划 8.6 母带三档预设一致：
 *   L0 ∞dB（分析/纯数据/零处理） → L1 60dB（抖动/直流） → L2 35dB（主流模型与 EQ 类）
 *   → L3 25dB（饱和激励/深度原创） → L4 15dB（变速）。
 */

import type { WorkflowNodeType } from "../../types/project";

/** 损伤等级 0-4（0 = 零损伤）。做成必填字段，漏了编译不过。 */
export type DamageLevel = 0 | 1 | 2 | 3 | 4;

export interface DamageInfo {
  level: DamageLevel;
  /** 该节点单独作用的近似信噪比（dB）；L0 为 Infinity。 */
  snrDb: number;
}

const L0: DamageInfo = { level: 0, snrDb: Infinity };
const L1: DamageInfo = { level: 1, snrDb: 60 };
const L2: DamageInfo = { level: 2, snrDb: 35 };
const L3: DamageInfo = { level: 3, snrDb: 25 };
const L4: DamageInfo = { level: 4, snrDb: 15 };

export const NODE_DAMAGE: Record<WorkflowNodeType, DamageInfo> = {
  // I/O 与零处理：原样透传
  input: L0,
  output: L0,
  split: L0,
  merge: L0,
  complianceCheck: L0,
  lufsNormalize: L0,
  // Phase 5 分析可视化：音频透传、报告走旁路端口
  spectrogram: L0,
  f0Curve: L0,
  timbreMetrics: L0,
  harmonicityCheck: L0,
  spectralCompare: L0,
  dtwAlign: L0,
  abCompare: L0,
  lufsAnalyze: L0,
  // 纯数据/符号域节点：不碰音频波形
  amtMidi: L0,
  chordDetect: L0,
  autoArrange: L0,
  midiFileIn: L0,
  chordBlockIn: L0,
  harmonizer: L0,
  melodyGen: L0,
  // 音源渲染是「音频的起点」而非途经站：它合成波形，不存在把已有音频弄脏的问题。
  soundfontRender: L0,
  melodySimilarity: L0,
  songLyrics: L0,
  songPrompt: L0,
  songSheet: L0,
  midiHumanize: L0,
  velocityCurve: L0,
  swingQuantize: L0,
  melodyReharm: L0,
  rhythmRestructure: L0,
  contourMorph: L0,
  motifDevelop: L0,
  reharmonize: L0,
  rhythmVariation: L0,
  structureEdit: L0,
  breathPlanner: L0,
  // L1 60dB：近无损（1 位抖动 / 直流去除）
  dither: L1,
  dcRemove: L1,
  // L2 35dB：主流生成模型与宽泛 EQ 类
  rvc: L2,
  sovits: L2,
  transpose: L2,
  busEq: L2,
  stereoWidth: L2,
  phaseRotate: L2,
  msstSeparation: L2,
  songStems: L2,
  songGen: L2,
  songCover: L2,
  songRepaint: L2,
  songComplete: L2,
  songExtract: L2,
  songLego: L2,
  songGenYue2: L2,
  songGenAceStep: L2,
  // L3 25dB：饱和激励 / 深度原创（多级模型级联）
  saturate: L3,
  deepOriginal: L3,
  // L4 15dB：变速（重采样式时域处理，最伤）
  speedShift: L4,
};

/** 损伤等级徽标（L0 零损伤不在节点卡上显示，保留供未来使用）。 */
export const DAMAGE_ICONS: Record<DamageLevel, string> = {
  0: "🟢",
  1: "🟡",
  2: "🟡",
  3: "🟠",
  4: "🔴",
};

/**
 * 级联 SNR 近似：-10·log10(Σ 10^(-db/10))。跳过非有限值（L0 / 未知）；
 * 全部为零损伤（Σ=0）时返回 null —— 调用方以「无意义」处理，不显示。
 */
export function cumulativeSnrDb(snrDbs: number[]): number | null {
  let sum = 0;
  for (const db of snrDbs) {
    if (Number.isFinite(db)) sum += Math.pow(10, -db / 10);
  }
  if (sum <= 0) return null;
  return -10 * Math.log10(sum);
}
