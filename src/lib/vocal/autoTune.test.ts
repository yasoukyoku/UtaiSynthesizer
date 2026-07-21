// S73 自动调教——调教所有权谓词的纯函数单测(应用链路的 store/undo 行为在 vocalData.test.ts)。
// S73c 语义:pitchDev 不参与 θ 资格(手绘=独立叠加层,机器永不写;基线在其下照常再生成=SV 同构)。
import { describe, it, expect, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: () => Promise.resolve() }));
vi.mock("../../i18n", () => ({ default: { t: (k: string) => k } }));

import { isUserTuned, autoTuneScalesOf, phaseForTake } from "./autoTune";
import type { Note } from "../../types/project";

const note = (extra: Partial<Note> = {}): Note => ({
  id: "n1", tick: 480, duration: 480, pitch: 60, lyric: "か", velocity: 100, ...extra,
});

describe("isUserTuned — θ 维度的自动调教绕行谓词(S73c:不看 pitchDev)", () => {
  it("裸音符 = 未调教(机器可调)", () => {
    expect(isUserTuned(note())).toBe(false);
  });
  it("手设 vibrato / transition = 用户调教(含 ustx 烤入的显式零 transition)", () => {
    expect(isUserTuned(note({ vibrato: { depthCents: 80, freqHz: 5.5, phase: 0, startMs: 0, easeInMs: 80, easeOutMs: 120 } }))).toBe(true);
    expect(isUserTuned(note({ transition: { durLeftMs: 0 } }))).toBe(true);
  });
  it("autoTuned 标记 = 机器调教,即使带 vibrato/transition 也可被机器改写", () => {
    expect(
      isUserTuned(
        note({
          autoTuned: true,
          transition: { durLeftMs: 120 },
          vibrato: { depthCents: 80, freqHz: 5.5, phase: 0, startMs: 0, easeInMs: 80, easeOutMs: 120 },
        }),
      ),
    ).toBe(false);
  });
});

describe("autoTuneScalesOf — 旋钮单一读取点", () => {
  it("absent 默认 = expr 2 / vib 1 / take 0(S73c/d 拍板)", () => {
    expect(autoTuneScalesOf(undefined)).toEqual({ expr: 2, vib: 1, take: 0 });
  });
  it("显式值透传", () => {
    expect(
      autoTuneScalesOf({ autoTuneExpr: 0.5, autoTuneVib: 1.5, autoTuneTake: 7 } as never),
    ).toEqual({ expr: 0.5, vib: 1.5, take: 7 });
  });
});

describe("phaseForTake — 确定性唱法版本(S73d,替代 Retake 抽奖)", () => {
  it("take 0 = 基准相位 0(KA3 耳测口径)", () => {
    expect(phaseForTake(0, "any-id")).toBe(0);
  });
  it("同 (take, id) 恒同相位;换 take/换 id 相位不同;域 [-0.5, 0.5)", () => {
    const a = phaseForTake(3, "n1");
    expect(phaseForTake(3, "n1")).toBe(a);
    expect(phaseForTake(4, "n1")).not.toBe(a);
    expect(phaseForTake(3, "n2")).not.toBe(a);
    for (let t = 1; t <= 20; t++) {
      const p = phaseForTake(t, "note-uuid-xyz");
      expect(p).toBeGreaterThanOrEqual(-0.5);
      expect(p).toBeLessThan(0.5);
    }
  });
});
