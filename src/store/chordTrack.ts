/**
 * Muno 阶段3「和弦轨道」——时间线顶部固定一行的和弦标签数据。
 *
 * 会话级 UI 状态（zustand）：analyzeChords 的结果 + 来源轨。
 * 刻意**不进 .usp / meaningfulSig**（纯视图覆盖层，与播放头同族），
 * 工程文件格式零改动 —— 打开旧工程、保存、撤销都不会触碰它。
 * 想要新的分析：轨道右键 →「识别和弦」再跑一次（确定性，同输入同输出）。
 */
import { create } from "zustand";
import type { ChordSegment, KeyEstimate } from "../lib/analysis/chordAnalysis";
import { TICKS_PER_BEAT } from "../lib/constants";

export interface ChordAnalysisPayload {
  sourceTrackId: string;
  sourceTrackName: string;
  segments: ChordSegment[];
  key: KeyEstimate;
}

interface ChordTrackState extends ChordAnalysisPayload {
  /** 顶部标签行有没有内容（空 = 显示引导文案）。 */
  hasAnalysis: () => boolean;
  setAnalysis: (v: ChordAnalysisPayload) => void;
  /** 就地改名（点击标签的输入框提交）。只改 label 显示，不动音乐内容。 */
  renameLabel: (index: number, label: string) => void;
  /** 拖拽调整分段边界：把第 index 段与下一段的共享边界挪到 tick ——
   *  吸附到拍（480 tick），两端各保 ≥1 拍宽（标签挤没=拒改）。会话态覆盖层，
   *  与 renameLabel 同族：不进 .usp / meaningfulSig / undo。 */
  moveBoundary: (index: number, tick: number) => void;
  clear: () => void;
}

const EMPTY: ChordAnalysisPayload = {
  sourceTrackId: "",
  sourceTrackName: "",
  segments: [],
  key: { tonic: 0, major: true, confidence: 0, label: "C" },
};

export const useChordTrackStore = create<ChordTrackState>((set, get) => ({
  ...EMPTY,
  hasAnalysis: () => get().segments.length > 0,
  setAnalysis: (v) => set({ ...v }),
  renameLabel: (index, label) =>
    set((s) => {
      const segs = s.segments;
      if (index < 0 || index >= segs.length) return s;
      const trimmed = label.trim();
      if (!trimmed || trimmed === segs[index]!.label) return s;
      const next = segs.slice();
      next[index] = { ...next[index]!, label: trimmed };
      return { segments: next };
    }),
  moveBoundary: (index, tick) =>
    set((s) => {
      const segs = s.segments;
      if (index < 0 || index + 1 >= segs.length) return s;
      const a = segs[index]!;
      const b = segs[index + 1]!;
      if (a.endTick !== b.startTick) return s; // 非连续分段（识别输出不会出现）——保守不动
      const snapped = Math.round(tick / TICKS_PER_BEAT) * TICKS_PER_BEAT;
      const lo = a.startTick + TICKS_PER_BEAT;
      const hi = b.endTick - TICKS_PER_BEAT;
      if (snapped < lo || snapped > hi || snapped === a.endTick) return s;
      const next = segs.slice();
      next[index] = { ...a, endTick: snapped };
      next[index + 1] = { ...b, startTick: snapped };
      return { segments: next };
    }),
  clear: () => set({ ...EMPTY }),
}));
