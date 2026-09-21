/**
 * 符号域共享小工具：数值钳制、音符排序、音级取模。
 */
import type { ChordAnalysisNote } from "../analysis/chordAnalysis";

export const SINGABLE_LOW = 36; // C2
export const SINGABLE_HIGH = 84; // C6

export function clamp01(v: number): number {
  return Math.min(1, Math.max(0, v));
}

export function clampVel(v: number): number {
  return Math.min(127, Math.max(1, Math.round(v)));
}

export function mod12p(v: number): number {
  return ((v % 12) + 12) % 12;
}

export function byTick(a: ChordAnalysisNote, b: ChordAnalysisNote): number {
  return a.tick - b.tick || a.pitch - b.pitch;
}
