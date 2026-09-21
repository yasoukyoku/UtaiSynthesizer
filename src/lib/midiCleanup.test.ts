import { describe, expect, it } from "vitest";
import {
  CleanableNote,
  CleanableStem,
  cleanChangedTotal,
  detectKeyPitchClasses,
  smartCleanStems,
} from "./midiCleanup";

const OPTS = { bpm: 120, ppq: 480, durationTicks: 1920 * 10 };
const note = (tick: number, duration: number, pitch: number, velocity = 100, id?: string): CleanableNote =>
  ({ tick, duration, pitch, velocity, id });

describe("smartCleanStems v3 — 新增 Pass", () => {
  it("双八度（+24）和声幻觉被删除", () => {
    const stems: CleanableStem[] = [
      { channel: 0, notes: [note(0, 480, 60, 100), note(0, 480, 84, 50)] },
    ];
    const stats = smartCleanStems(stems, OPTS);
    expect(stems[0]!.notes.map((n) => n.pitch)).toEqual([60]);
    expect(stats.removedGhost).toBe(1);
  });

  it("力度轮廓压缩：局部离群向中位数靠一半", () => {
    const stems: CleanableStem[] = [
      {
        channel: 0,
        notes: [
          note(0, 120, 60, 100), note(240, 120, 62, 140), note(480, 120, 64, 100),
          note(720, 120, 65, 100), note(960, 120, 67, 100),
        ],
      },
    ];
    const stats = smartCleanStems(stems, OPTS);
    expect(stats.fixedVelocity).toBe(1);
    expect(stems[0]!.notes[1]!.velocity).toBe(120); // (140+100)/2
  });

  it("拖尾截断：鼓轨超过半小节封顶", () => {
    const stems: CleanableStem[] = [{ channel: 9, notes: [note(0, 3000, 38)] }];
    const stats = smartCleanStems(stems, OPTS);
    expect(stems[0]!.notes[0]!.duration).toBe(960); // 1920/2
    expect(stats.trimmedHang).toBe(1);
  });

  it("拖尾截断：旋律轨（重合率低）收束为单声部", () => {
    const stems: CleanableStem[] = [
      {
        channel: 0,
        notes: [
          note(0, 500, 60), note(480, 120, 64), note(720, 120, 67), note(960, 120, 72),
          note(1200, 120, 74), note(1440, 120, 76), note(1680, 120, 77),
        ],
      },
    ];
    const stats = smartCleanStems(stems, OPTS);
    // 第一个音 500 拖到下一音起点 480 之外 → 截到 480。
    expect(stems[0]!.notes[0]!.duration).toBe(480);
    expect(stats.trimmedHang).toBeGreaterThanOrEqual(1);
  });

  it("节拍量化：贴近 1/32 网格的起点被吸附", () => {
    const stems: CleanableStem[] = [
      { channel: 0, notes: [note(490, 240, 60), note(1210, 240, 62)] },
    ];
    const stats = smartCleanStems(stems, OPTS);
    expect(stems[0]!.notes.map((n) => n.tick)).toEqual([480, 1200]);
    expect(stats.snapped).toBe(2);
  });

  it("节拍量化：远离网格的切分音不受影响", () => {
    const stems: CleanableStem[] = [{ channel: 0, notes: [note(505, 240, 60)] }];
    smartCleanStems(stems, OPTS);
    // 505 距网格 480 差 25 > 容差 24 → 不动。
    expect(stems[0]!.notes[0]!.tick).toBe(505);
  });

  it("周期性补漏（v3 规则二）：非紧邻重复的空小节按周期回填", () => {
    const C4 = 60, D4 = 62, E4 = 64, F4 = 65, G4 = 67, A4 = 69;
    // 乐句 0-3 小节 = C D E F，4-7 小节重复该乐句但模型漏检了 4、5 两小节
    //（bars 6-7 = bars 2-3 的拷贝，bars 8-9 走向新内容）。
    // 空档前一格（bar3=F）与再前一格（bar2=E）不相似 → v2 规则不触发；
    // 周期 d=4 的自相似度 = 1.0（bars6-7 ≈ bars2-3）→ v3 回填 bars 4←0、5←1。
    const notes: CleanableNote[] = [];
    const layout = [C4, D4, E4, F4, null, null, E4, F4, G4, A4];
    layout.forEach((p, bar) => {
      if (p != null) notes.push(note(bar * 1920, 480, p));
    });
    const stems: CleanableStem[] = [{ channel: 0, notes }];
    const stats = smartCleanStems(stems, OPTS);
    expect(stats.filledNotes).toBe(2); // bar4←bar0(C)，bar5←bar1(D)
    const filled = stems[0]!.notes.filter((n) => n.tick === 4 * 1920 || n.tick === 5 * 1920);
    expect(filled.map((n) => n.pitch)).toEqual([C4, D4]);
  });

  it("v2 补漏规则回归：前两小节相同 → 平铺回填", () => {
    const notes: CleanableNote[] = [];
    // bar0-2 内容相同、bar3 空、bar4 又出现（音乐仍在继续）。
    [0, 1, 2, 4].forEach((bar) => notes.push(note(bar * 1920, 480, 60)));
    const stems: CleanableStem[] = [{ channel: 0, notes }];
    const stats = smartCleanStems(stems, OPTS);
    expect(stats.filledNotes).toBe(1);
    expect(stems[0]!.notes.some((n) => n.tick === 3 * 1920)).toBe(true);
  });
});

describe("v2 行为回归", () => {
  it("超短伪音删除 + 同音重叠合并", () => {
    const stems: CleanableStem[] = [
      {
        channel: 0,
        notes: [note(0, 20, 60), note(100, 300, 62), note(200, 300, 62)],
      },
    ];
    const stats = smartCleanStems(stems, OPTS);
    expect(stats.removedShort).toBe(1);
    expect(stats.mergedOverlap).toBe(1);
  });

  it("音域外音高删除", () => {
    const stems: CleanableStem[] = [{ channel: 0, notes: [note(0, 480, 120)] }];
    const stats = smartCleanStems(stems, OPTS);
    expect(stats.removedRange).toBe(1);
    expect(stems[0]!.notes).toHaveLength(0);
  });

  it("cleanChangedTotal 汇总含 v3 新统计", () => {
    expect(cleanChangedTotal({
      removedShort: 0, mergedOverlap: 0, removedGhost: 0, removedKey: 0,
      fixedVelocity: 0, removedRange: 0, filledNotes: 0,
      trimmedHang: 1, snapped: 2, touched: [],
    })).toBe(3);
  });

  it("detectKeyPitchClasses 音符太少返回 null", () => {
    const stems: CleanableStem[] = [{ channel: 0, notes: [note(0, 480, 60)] }];
    expect(detectKeyPitchClasses(stems)).toBeNull();
  });
});
