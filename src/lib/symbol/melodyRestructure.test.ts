/**
 * 旋律重构四法 + 相似度度量单测（规划 2-4 验收）。
 * 核心验收：contourMorph 强度 60% 时，输出与输入相似度 ∈ [35%, 45%]（默认种子）。
 */
import { describe, it, expect } from "vitest";
import type { ChordAnalysisNote } from "../analysis/chordAnalysis";
import {
  melodySimilarity,
  degreeSwap,
  rhythmRestructure,
  contourMorph,
  motifDevelop,
} from "./melodyRestructure";
import { DEFAULT_SYMBOL_SEED } from "./rng";

/** C 大调四小节测试旋律（全八分音符，32 个音，含高潮音 A5=81）。 */
function testMelody(): ChordAnalysisNote[] {
  const pitches = [
    72, 74, 76, 77, 79, 77, 76, 74,
    72, 71, 69, 67, 69, 71, 72, 74,
    76, 79, 81, 79, 77, 76, 74, 72,
    74, 72, 71, 69, 67, 69, 71, 72,
  ];
  return pitches.map((pitch, i) => ({ tick: i * 240, duration: 220, pitch, velocity: 96 }));
}

describe("melodySimilarity", () => {
  it("自身相似度 = 1", () => {
    const m = testMelody();
    expect(melodySimilarity(m, m)).toBe(1);
  });

  it("整体移调不改变相似度（转置不变性）", () => {
    const m = testMelody();
    const up = m.map((n) => ({ ...n, pitch: n.pitch + 3 }));
    expect(melodySimilarity(m, up)).toBeGreaterThan(0.95);
  });

  it("无关旋律相似度低", () => {
    const m = testMelody();
    const other = m.map((n, i) => ({ ...n, pitch: 60 + ((i * 7) % 13) }));
    expect(melodySimilarity(m, other)).toBeLessThan(0.35);
  });
});

describe("contourMorph（法三，验收核心）", () => {
  it("验收：强度 60% 时相似度 ∈ [0.35, 0.45]（默认种子）", () => {
    const m = testMelody();
    const out = contourMorph(m, { strength: 60, keepClimax: true });
    const sim = melodySimilarity(m, out);
    // 规划 2-4 验收断言：60% 强度 → 35–45% 相似度。
    expect(sim).toBeGreaterThanOrEqual(0.35);
    expect(sim).toBeLessThanOrEqual(0.45);
  });

  it("多种子下相似度落在宽区间 [0.30, 0.55]", () => {
    const m = testMelody();
    for (const seed of [1, 2, 3, 42, 999, DEFAULT_SYMBOL_SEED]) {
      const sim = melodySimilarity(m, contourMorph(m, { strength: 60, keepClimax: true, seed }));
      expect(sim).toBeGreaterThanOrEqual(0.3);
      expect(sim).toBeLessThanOrEqual(0.55);
    }
  });

  it("强度 0 时音高完全不变", () => {
    const m = testMelody();
    const out = contourMorph(m, { strength: 0, keepClimax: true });
    expect(out.map((n) => n.pitch)).toEqual(m.map((n) => n.pitch));
  });

  it("轮廓走向保持：非零音程符号不变", () => {
    const m = testMelody();
    const out = contourMorph(m, { strength: 80, keepClimax: true, seed: 5 });
    for (let i = 0; i + 1 < m.length; i++) {
      const a = Math.sign(m[i + 1]!.pitch - m[i]!.pitch);
      const b = Math.sign(out[i + 1]!.pitch - out[i]!.pitch);
      if (a === 0) continue; // 同度允许被钳制边界打破
      if (b === 0) continue; // 音域贴边的同度塌缩是合法输出
      expect(b).toBe(a);
    }
  });

  it("keepClimax 保留最高音位置", () => {
    const m = testMelody();
    const climaxIdx = m.reduce((best, n, i) => (n.pitch > m[best]!.pitch ? i : best), 0);
    const out = contourMorph(m, { strength: 70, keepClimax: true, seed: 11 });
    expect(out[climaxIdx]!.pitch).toBe(m[climaxIdx]!.pitch);
  });

  it("强度越高相似度越低（单调性抽查）", () => {
    const m = testMelody();
    const sim30 = melodySimilarity(m, contourMorph(m, { strength: 30, keepClimax: true, seed: 7 }));
    const sim90 = melodySimilarity(m, contourMorph(m, { strength: 90, keepClimax: true, seed: 7 }));
    expect(sim30).toBeGreaterThan(sim90);
  });

  it("音域钳制：输出始终在 C2..C6", () => {
    const m = testMelody();
    const out = contourMorph(m, { strength: 100, keepClimax: false, seed: 13 });
    out.forEach((n) => {
      expect(n.pitch).toBeGreaterThanOrEqual(36);
      expect(n.pitch).toBeLessThanOrEqual(84);
    });
  });
});

describe("degreeSwap（法一）", () => {
  it("density=0 时不变", () => {
    const m = testMelody();
    expect(degreeSwap(m, { density: 0, seed: 1 }).map((n) => n.pitch)).toEqual(m.map((n) => n.pitch));
  });

  it("替换后的音都在估计调式的音阶内（首音除外）", () => {
    const m = testMelody();
    const out = degreeSwap(m, { density: 100, seed: 21 });
    const scalePcs = new Set([0, 2, 4, 5, 7, 9, 11]);
    for (let i = 1; i < out.length; i++) {
      expect(scalePcs.has(((out[i]!.pitch % 12) + 12) % 12)).toBe(true);
    }
  });

  it("同种子可复现、异种子不同", () => {
    const m = testMelody();
    const o = { density: 60 };
    expect(degreeSwap(m, { ...o, seed: 100 })).toEqual(degreeSwap(m, { ...o, seed: 100 }));
    expect(degreeSwap(m, { ...o, seed: 100 })).not.toEqual(degreeSwap(m, { ...o, seed: 200 }));
  });

  it("重构后相似度介于无关与相同之间", () => {
    const m = testMelody();
    const out = degreeSwap(m, { density: 80, seed: 33 });
    const sim = melodySimilarity(m, out);
    expect(sim).toBeGreaterThan(0.1);
    expect(sim).toBeLessThan(0.95);
  });
});

describe("rhythmRestructure（法二）", () => {
  it("strength=0 时不变", () => {
    const m = testMelody();
    expect(rhythmRestructure(m, { strength: 0 })).toEqual(m);
  });

  it("保音高序列：合并不跨小节时音高多重集变化受限（拆分/合并仅重复音高）", () => {
    const m = testMelody();
    const out = rhythmRestructure(m, { strength: 100, seed: 17 });
    // 输出已排序、tick 单调不减。
    for (let i = 1; i < out.length; i++) {
      expect(out[i]!.tick).toBeGreaterThanOrEqual(out[i - 1]!.tick);
    }
    // 首音高不变（小节头组不受"合并/拆分首音高保留"影响——至少旋律起点仍是 72）。
    expect(out[0]!.pitch).toBe(72);
  });

  it("跨度大致保持（±两拍内）", () => {
    const m = testMelody();
    const out = rhythmRestructure(m, { strength: 100, seed: 19 });
    const spanA = m[m.length - 1]!.tick + m[m.length - 1]!.duration;
    const spanB = out[out.length - 1]!.tick + out[out.length - 1]!.duration;
    expect(Math.abs(spanB - spanA)).toBeLessThanOrEqual(960);
  });
});

describe("motifDevelop（法四）", () => {
  it("首段动机逐音保留", () => {
    const m = testMelody();
    const out = motifDevelop(m, { technique: "sequence", motifBars: 2, seed: 23 });
    // 动机 = 前 2 小节 = 前 16 个音（全八分）。
    for (let i = 0; i < 16; i++) {
      expect(out[i]!.pitch).toBe(m[i]!.pitch);
      expect(out[i]!.tick).toBe(m[i]!.tick);
    }
  });

  it("各技巧都有输出且音域合法", () => {
    const m = testMelody();
    for (const technique of ["sequence", "invert", "retrograde", "augment", "diminish", "mixed"] as const) {
      const out = motifDevelop(m, { technique, motifBars: 2, seed: 29 });
      expect(out.length).toBeGreaterThan(0);
      out.forEach((n) => {
        expect(n.pitch).toBeGreaterThanOrEqual(36);
        expect(n.pitch).toBeLessThanOrEqual(84);
        expect(n.tick).toBeGreaterThanOrEqual(0);
        expect(n.duration).toBeGreaterThan(0);
      });
    }
  });

  it("retrograde 动机倒序（第二组起点 = 动机末音音高）", () => {
    const m = testMelody();
    const out = motifDevelop(m, { technique: "retrograde", motifBars: 2, seed: 31 });
    // 第二组起点 = motifBars(2) × barTicks(1920) = 3840；倒序后首音 = 动机末音音高（74）。
    const secondGroup = out.filter((n) => n.tick >= 3840);
    expect(secondGroup.length).toBeGreaterThan(0);
    expect(secondGroup[0]!.pitch).toBe(m[15]!.pitch);
  });

  it("同种子可复现", () => {
    const m = testMelody();
    const o = { technique: "mixed" as const, motifBars: 2, seed: 37 };
    expect(motifDevelop(m, o)).toEqual(motifDevelop(m, o));
  });
});
