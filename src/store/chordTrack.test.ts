import { describe, expect, it, beforeEach } from "vitest";

import { useChordTrackStore } from "./chordTrack";
import type { ChordSegment, KeyEstimate } from "../lib/analysis/chordAnalysis";

/** 两段连续和弦（0–1920 / 1920–3840 tick，即各一小节 4/4）—— moveBoundary 的最小夹具。 */
function twoSegments(): ChordSegment[] {
  return [
    { startTick: 0, endTick: 1920, root: 0, quality: "maj", bass: null, label: "C" },
    { startTick: 1920, endTick: 3840, root: 9, quality: "min", bass: null, label: "Am" },
  ];
}

function load(segments: ChordSegment[]) {
  useChordTrackStore.getState().setAnalysis({
    sourceTrackId: "t1",
    sourceTrackName: "Melody",
    segments,
    key: { tonic: 0, major: true, confidence: 1, label: "C" } as KeyEstimate,
  });
}

describe("chordTrack moveBoundary (拖拽调整分段边界)", () => {
  beforeEach(() => {
    load(twoSegments());
  });

  it("把共享边界挪到吸附后的 tick：两段同步收/放，标签不动", () => {
    useChordTrackStore.getState().moveBoundary(0, 1440); // 3 拍处
    const segs = useChordTrackStore.getState().segments;
    expect(segs[0]).toMatchObject({ startTick: 0, endTick: 1440, label: "C" });
    expect(segs[1]).toMatchObject({ startTick: 1440, endTick: 3840, label: "Am" });
  });

  it("吸附到最近的拍（480 tick 网格）", () => {
    useChordTrackStore.getState().moveBoundary(0, 1100); // → 吸附 960（2 拍）
    expect(useChordTrackStore.getState().segments[0]!.endTick).toBe(960);
    useChordTrackStore.getState().moveBoundary(0, 1000); // → 吸附仍 960 = 当前边界（no-op）
    expect(useChordTrackStore.getState().segments[0]!.endTick).toBe(960);
  });

  it("两端各保 ≥1 拍：越界拖拽被拒（段挤没 = 不动）", () => {
    useChordTrackStore.getState().moveBoundary(0, 200); // < 左段 startTick + 480
    expect(useChordTrackStore.getState().segments[0]!.endTick).toBe(1920);
    useChordTrackStore.getState().moveBoundary(0, 3700); // > 右段 endTick - 480
    expect(useChordTrackStore.getState().segments[0]!.endTick).toBe(1920);
  });

  it("末段边界（没有下一段）与不连续分段：保守拒改", () => {
    useChordTrackStore.getState().moveBoundary(1, 2400); // index 1 是最后一段 —— 无共享边界
    expect(useChordTrackStore.getState().segments[1]!.endTick).toBe(3840);
    // 非连续（识别输出不会出现，防御）：a.endTick ≠ b.startTick 时不动。
    load([
      { startTick: 0, endTick: 1920, root: 0, quality: "maj", bass: null, label: "C" },
      { startTick: 2000, endTick: 3840, root: 9, quality: "min", bass: null, label: "Am" },
    ]);
    useChordTrackStore.getState().moveBoundary(0, 1440);
    expect(useChordTrackStore.getState().segments[0]!.endTick).toBe(1920);
    expect(useChordTrackStore.getState().segments[1]!.startTick).toBe(2000);
  });
});
