// S73 自动调教——调教所有权谓词的纯函数单测(应用链路的 store/undo 行为在 vocalData.test.ts)。
import { describe, it, expect, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: () => Promise.resolve() }));
vi.mock("../../i18n", () => ({ default: { t: (k: string) => k } }));

import { curveHasSignal, isUserTuned, TUNED_DEV_EPS_CENTS } from "./autoTune";
import type { Note, PitchCurve } from "../../types/project";

const note = (extra: Partial<Note> = {}): Note => ({
  id: "n1", tick: 480, duration: 480, pitch: 60, lyric: "か", velocity: 100, ...extra,
});

describe("curveHasSignal — 折线在窗口内的信号检测", () => {
  it("空/undefined 曲线无信号", () => {
    expect(curveHasSignal(undefined, 0, 1000, 2)).toBe(false);
    expect(curveHasSignal({ xs: [], ys: [] }, 0, 1000, 2)).toBe(false);
  });
  it("窗口内控制点 ≥eps 命中;<eps 的噪声不命中", () => {
    const c: PitchCurve = { xs: [500, 600], ys: [0, 5] };
    expect(curveHasSignal(c, 480, 960, 2)).toBe(true);
    expect(curveHasSignal({ xs: [500, 600], ys: [1, -1] }, 480, 960, 2)).toBe(false);
  });
  it("控制点全在窗外但折线段穿过窗口 → 端点插值命中", () => {
    // 100→(y=40) …… 2000→(y=40):窗口 [480,960] 内无控制点,但插值恒 40
    const c: PitchCurve = { xs: [100, 2000], ys: [40, 40] };
    expect(curveHasSignal(c, 480, 960, 2)).toBe(true);
  });
  it("窗口外的大信号不误伤(首尾外持平语义下窗口值为 0)", () => {
    const c: PitchCurve = { xs: [1500, 1600, 1700], ys: [0, 80, 0] };
    expect(curveHasSignal(c, 0, 960, 2)).toBe(false);
  });
});

describe("isUserTuned — 自动调教的绕行谓词", () => {
  it("裸音符 = 未调教", () => {
    expect(isUserTuned(note(), undefined)).toBe(false);
  });
  it("手设 vibrato / transition = 用户调教", () => {
    expect(isUserTuned(note({ vibrato: { depthCents: 80, freqHz: 5.5, phase: 0, startMs: 0, easeInMs: 80, easeOutMs: 120 } }), undefined)).toBe(true);
    expect(isUserTuned(note({ transition: { durLeftMs: 0 } }), undefined)).toBe(true);
  });
  it("autoTuned 标记 = 机器调教,即使带 vibrato 也可被机器改写", () => {
    expect(isUserTuned(note({ autoTuned: true, vibrato: { depthCents: 80, freqHz: 5.5, phase: 0, startMs: 0, easeInMs: 80, easeOutMs: 120 } }), undefined)).toBe(false);
  });
  it("★机器调教之上又手绘了 dev → 用户地盘(dev 检查先于 autoTuned 短路,S73 审查)", () => {
    const dev: PitchCurve = { xs: [480, 700, 960], ys: [0, 30, 0] };
    expect(isUserTuned(note({ autoTuned: true }), dev)).toBe(true);
  });
  it("pitchDev 覆盖音符范围 = 用户调教(手绘/ustx 烤入);别的音符不受牵连", () => {
    const dev: PitchCurve = { xs: [480, 700, 960], ys: [0, 30, 0] };
    expect(isUserTuned(note(), dev)).toBe(true);
    expect(isUserTuned(note({ id: "n2", tick: 2000 }), dev)).toBe(false);
  });
  it("eps 阈内的微小残留不算调教", () => {
    const dev: PitchCurve = { xs: [480, 960], ys: [TUNED_DEV_EPS_CENTS - 1, 0] };
    expect(isUserTuned(note(), dev)).toBe(false);
  });
});
