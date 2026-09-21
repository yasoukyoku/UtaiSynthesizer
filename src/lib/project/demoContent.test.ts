import { describe, expect, it } from "vitest";

import { buildDemoTracks } from "./demoContent";
import { TICKS_PER_BEAT } from "../constants";

const BARS = 8;
const BAR_TICKS = TICKS_PER_BEAT * 4; // 4/4

describe("demoContent buildDemoTracks (示例工程模板)", () => {
  const tracks = buildDemoTracks();

  it("四条乐器轨，各一个 notes 段从 0 起，段长覆盖全部音符", () => {
    expect(tracks).toHaveLength(4);
    for (const tr of tracks) {
      expect(tr.trackType).toBe("instrument");
      expect(tr.segments).toHaveLength(1);
      const seg = tr.segments[0]!;
      expect(seg.startTick).toBe(0);
      expect(seg.content.type).toBe("notes");
      const notes = seg.content.type === "notes" ? seg.content.notes : [];
      expect(notes.length).toBeGreaterThan(0);
      let maxTick = 0;
      for (const n of notes) maxTick = Math.max(maxTick, n.tick + n.duration);
      expect(seg.durationTicks).toBeGreaterThanOrEqual(maxTick);
    }
  });

  it("鼓轨名含 \"Drums\"（GM channel 10 的命名约定）", () => {
    expect(tracks.some((tr) => tr.name.includes("Drums"))).toBe(true);
  });

  it("全部音符在 8 小节内，音高合法，note id 唯一", () => {
    const ids = new Set<string>();
    for (const tr of tracks) {
      for (const seg of tr.segments) {
        if (seg.content.type !== "notes") continue;
        for (const n of seg.content.notes) {
          expect(n.tick).toBeGreaterThanOrEqual(0);
          expect(n.tick + n.duration).toBeLessThanOrEqual(BARS * BAR_TICKS);
          expect(n.pitch).toBeGreaterThanOrEqual(0);
          expect(n.pitch).toBeLessThanOrEqual(127);
          expect(n.velocity).toBeGreaterThan(0);
          ids.add(n.id);
        }
      }
    }
    expect(ids.size).toBeGreaterThan(100); // 四轨合计，且 id 无碰撞
  });

  it("主旋律末小节是全音符收尾（第 8 小节 60 持续 4 拍）", () => {
    const melody = tracks[0]!;
    const notes = melody.segments[0]!.content.type === "notes"
      ? melody.segments[0]!.content.notes
      : [];
    const last = notes[notes.length - 1]!;
    expect(last.pitch).toBe(60);
    expect(last.tick).toBe(7 * BAR_TICKS);
    expect(last.duration).toBe(BAR_TICKS);
  });
});
