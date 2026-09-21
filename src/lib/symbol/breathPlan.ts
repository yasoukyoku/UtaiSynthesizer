/**
 * 换气点规划 — 符号域部分（规划 2-11，对应 8.7 换气层）。
 * 从旋律音符的间隙 + 可选歌词标点推断换气插入点，输出结构化 JSON。
 * 音频域的实际换气声采样插入属 Phase 7；本模块只产出"哪里换气、换多大"的计划。
 */
import type { ChordAnalysisNote } from "../analysis/chordAnalysis";
import { TICKS_PER_BEAT } from "../constants";
import { byTick } from "./common";

export interface BreathPoint {
  tick: number;
  /** 0-1：1 = 全呼吸（句末），0.6 = 短换气。 */
  weight: number;
  reason: "gap" | "punct";
}

export interface BreathPlanOptions {
  /** 触发换气的最小间隙（拍数），默认 1。 */
  minGapBeats?: number;
  ppq?: number;
  /** 可选歌词：与音符一一对应时，标点位置追加换气点。 */
  lyrics?: string;
}

/** 每个音节（中/日/韩字符各一音节，拉丁字母聚合成词）的标点权重；0 = 无标点。 */
function splitSyllablePuncts(text: string): number[] {
  const PUNCT: Record<string, number> = {
    "。": 1, "！": 1, "？": 1, "，": 0.7, "、": 0.7, "；": 0.7,
    ".": 1, "!": 1, "?": 1, ",": 0.7, ";": 0.7,
  };
  const weights: number[] = [];
  let lastLatin = false;
  for (const ch of text) {
    const w = PUNCT[ch];
    if (w != null) {
      if (weights.length > 0) weights[weights.length - 1] = Math.max(weights[weights.length - 1]!, w);
      lastLatin = false;
      continue;
    }
    if (/\s/.test(ch)) {
      lastLatin = false;
      continue;
    }
    if (/[A-Za-z0-9]/.test(ch)) {
      if (!lastLatin) {
        weights.push(0);
        lastLatin = true;
      }
      continue;
    }
    if (/[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]/.test(ch)) {
      weights.push(0);
      lastLatin = false;
    }
  }
  return weights;
}

export function planBreathPoints(notes: ChordAnalysisNote[], o: BreathPlanOptions = {}): BreathPoint[] {
  if (notes.length === 0) return [];
  const ppq = o.ppq ?? TICKS_PER_BEAT;
  const minGap = Math.max(0, o.minGapBeats ?? 1) * ppq;
  const sorted = [...notes].sort(byTick);
  const points: BreathPoint[] = [];
  // 间隙驱动：音符间静默 ≥ minGap 即换气，间隙越长权重越大。
  for (let i = 0; i + 1 < sorted.length; i++) {
    const cur = sorted[i]!;
    const next = sorted[i + 1]!;
    const end = cur.tick + cur.duration;
    const gap = next.tick - end;
    if (gap >= minGap) {
      points.push({
        tick: Math.round(end + gap / 2),
        weight: gap >= 2 * ppq ? 1 : 0.6,
        reason: "gap",
      });
    }
  }
  // 歌词标点驱动：仅当音节数与音符数一致时对齐（错位则整体放弃，不猜）。
  const lyrics = o.lyrics?.trim();
  if (lyrics) {
    const puncts = splitSyllablePuncts(lyrics);
    if (puncts.length === sorted.length) {
      puncts.forEach((w, i) => {
        if (w > 0) {
          const cur = sorted[i]!;
          points.push({ tick: Math.round(cur.tick + cur.duration + ppq / 4), weight: w, reason: "punct" });
        }
      });
    }
  }
  return points.sort((a, b) => a.tick - b.tick);
}
