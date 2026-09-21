import { describe, it, expect } from "vitest";
import { NODE_DAMAGE, DAMAGE_ICONS, cumulativeSnrDb, type DamageLevel } from "./damage";

describe("cumulativeSnrDb", () => {
  it("单节点：-10·log10(10^(-snr/10)) 还原为原值", () => {
    expect(cumulativeSnrDb([35])).toBeCloseTo(35, 6);
    expect(cumulativeSnrDb([15])).toBeCloseTo(15, 6);
  });

  it("级联总 SNR 低于任意单级（损伤叠加单调劣化）", () => {
    const two = cumulativeSnrDb([60, 60])!;
    expect(two).toBeCloseTo(60 - 10 * Math.log10(2), 6);
    expect(two).toBeLessThan(60);
    expect(cumulativeSnrDb([35, 35, 25])!).toBeLessThan(25);
  });

  it("跳过非有限值（L0 = ∞ 零损伤节点不影响总 SNR）", () => {
    expect(cumulativeSnrDb([Infinity, 35])).toBeCloseTo(35, 6);
    expect(cumulativeSnrDb([35, Infinity, NaN, 35])).toBeCloseTo(35 - 10 * Math.log10(2), 6);
  });

  it("全零损伤或空链返回 null（调用方按「无意义」不显示）", () => {
    expect(cumulativeSnrDb([])).toBeNull();
    expect(cumulativeSnrDb([Infinity])).toBeNull();
    expect(cumulativeSnrDb([Infinity, Infinity])).toBeNull();
  });
});

describe("NODE_DAMAGE 表", () => {
  it("I/O 与纯数据节点为零损伤（∞dB）", () => {
    expect(NODE_DAMAGE.input).toEqual({ level: 0, snrDb: Infinity });
    expect(NODE_DAMAGE.output).toEqual({ level: 0, snrDb: Infinity });
    expect(NODE_DAMAGE.amtMidi.snrDb).toBe(Infinity);
    expect(NODE_DAMAGE.midiFileIn.snrDb).toBe(Infinity);
  });

  it("等级锚点：L1 60 / L2 35 / L3 25 / L4 15", () => {
    expect(NODE_DAMAGE.dither).toEqual({ level: 1, snrDb: 60 });
    expect(NODE_DAMAGE.rvc).toEqual({ level: 2, snrDb: 35 });
    expect(NODE_DAMAGE.saturate).toEqual({ level: 3, snrDb: 25 });
    expect(NODE_DAMAGE.speedShift).toEqual({ level: 4, snrDb: 15 });
  });

  it("全表自洽：L0 ↔ ∞dB，L1-4 ↔ 有限正 dB，等级与图标映射齐备", () => {
    const levels = new Set<DamageLevel>();
    for (const [type, info] of Object.entries(NODE_DAMAGE)) {
      expect(info.level, type).toBeGreaterThanOrEqual(0);
      expect(info.level, type).toBeLessThanOrEqual(4);
      if (info.level === 0) {
        expect(info.snrDb, type).toBe(Infinity);
      } else {
        expect(Number.isFinite(info.snrDb), type).toBe(true);
        expect(info.snrDb, type).toBeGreaterThan(0);
      }
      levels.add(info.level);
      expect(DAMAGE_ICONS[info.level], type).toBeTruthy();
    }
    // 0-4 五档全部被至少一个节点使用，DAMAGE_ICONS 无缺档
    expect([...levels].sort()).toEqual([0, 1, 2, 3, 4]);
    expect(Object.keys(DAMAGE_ICONS)).toHaveLength(5);
  });
});
