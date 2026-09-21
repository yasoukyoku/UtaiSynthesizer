import type { Note, Track } from "../../types/project";
import { TICKS_PER_BEAT } from "../constants";
import { blankTrack } from "../trackFactory";

/**
 * 示例工程内容（新手一键模板）：C 大调 · 120 BPM · 4/4 · 8 小节，四条乐器轨——
 * 主旋律 / 和弦伴奏 / 贝斯 / 鼓。纯数据构造（不碰 store），供 loadDemoProject() 落轨，
 * 也可直接单测。tick 域：一拍 = 480，一小节 = 1920。
 */

const TICKS_PER_BAR = 1920;
const BARS = 8;

function note(tick: number, duration: number, pitch: number, velocity: number): Note {
  return { id: crypto.randomUUID(), tick, duration, pitch, lyric: "", velocity };
}

/** 主旋律（简谱 1=C）：do mi sol mi | fa la sol mi | re fa la sol | sol si do(2拍) |
 *  do' si la sol | fa mi re do | re fa mi re | do(全音符)。欢快的小儿歌轮廓，新手一眼能看懂。 */
function demoMelody(): Note[] {
  const bars: number[][] = [
    [60, 64, 67, 64],
    [65, 69, 67, 64],
    [62, 65, 69, 67],
    [67, 71, 72, -1], // 末音 2 拍
    [72, 71, 69, 67],
    [65, 64, 62, 60],
    [62, 65, 64, 62],
    [60, -2, -2, -2], // 全音符收尾
  ];
  const notes: Note[] = [];
  bars.forEach((pitches, bar) => {
    const base = bar * TICKS_PER_BAR;
    let beat = 0;
    for (const p of pitches) {
      if (p < 0) break; // -1/-2 标记「本音延长到小节尾」
      const isLast = beat === 3 || pitches[beat + 1]! < 0;
      const dur = isLast ? (BAR_END(beat, pitches)) : TICKS_PER_BEAT;
      notes.push(note(base + beat * TICKS_PER_BEAT, dur, p, 92));
      beat += 1;
    }
  });
  return notes;
}

/** 末音时长：小节 4/8 的长音占 2 拍，小节 8 的全音符占 4 拍。 */
function BAR_END(beat: number, pitches: number[]): number {
  if (pitches[0] === 60 && pitches[1] === -2) return TICKS_PER_BAR; // 全音符小节
  return TICKS_PER_BAR - beat * TICKS_PER_BEAT;
}

/** 和弦伴奏（每小节两次柱式，各 2 拍）：C / Am / F / G 循环两遍。 */
function demoChords(): Note[] {
  const voicing: number[][] = [
    [60, 64, 67], // C
    [57, 60, 64], // Am
    [53, 57, 60], // F
    [55, 59, 62], // G
  ];
  const notes: Note[] = [];
  for (let bar = 0; bar < BARS; bar++) {
    const base = bar * TICKS_PER_BAR;
    const chord = voicing[bar % 4]!;
    for (const half of [0, 960]) {
      for (const p of chord) notes.push(note(base + half, 940, p, 70));
    }
  }
  return notes;
}

/** 贝斯（根音 2 拍 + 五度 2 拍）：C / Am / F / G 循环两遍。 */
function demoBass(): Note[] {
  const roots: Array<[number, number]> = [
    [36, 43], // C2 / G2
    [33, 40], // A1 / E2
    [41, 48], // F2 / C3
    [43, 50], // G2 / D3
  ];
  const notes: Note[] = [];
  for (let bar = 0; bar < BARS; bar++) {
    const base = bar * TICKS_PER_BAR;
    const [root, fifth] = roots[bar % 4]!;
    notes.push(note(base, 940, root, 88));
    notes.push(note(base + 960, 940, fifth, 82));
  }
  return notes;
}

/** 鼓（名字含 "Drums" → GM channel 10）：每拍踩镲，1/3 拍底鼓，2/4 拍军鼓；末小节镲片收尾。 */
function demoDrums(): Note[] {
  const HAT = 42, KICK = 36, SNARE = 38, CRASH = 49;
  const notes: Note[] = [];
  for (let bar = 0; bar < BARS; bar++) {
    const base = bar * TICKS_PER_BAR;
    for (let beat = 0; beat < 4; beat++) {
      const t = base + beat * TICKS_PER_BEAT;
      notes.push(note(t, 240, HAT, 58));
      if (beat % 2 === 0) notes.push(note(t, 240, KICK, 104));
      else notes.push(note(t, 240, SNARE, 94));
    }
  }
  // 末小节起点叠一枚镲片（4 拍延音）收尾。
  notes.push(note((BARS - 1) * TICKS_PER_BAR, TICKS_PER_BAR, CRASH, 108));
  return notes;
}

/** 组装示例工程轨道（旋律 → 和弦 → 贝斯 → 鼓）。不含音色分配（loadDemoProject 按已导入音源补）。 */
export function buildDemoTracks(): Track[] {
  const defs: Array<{ name: string; notes: Note[] }> = [
    { name: "Demo Melody · 主旋律", notes: demoMelody() },
    { name: "Demo Chords · 和弦", notes: demoChords() },
    { name: "Demo Bass · 贝斯", notes: demoBass() },
    { name: "Demo Drums · 鼓", notes: demoDrums() },
  ];
  return defs.map((d) => {
    let maxTick = 0;
    for (const n of d.notes) maxTick = Math.max(maxTick, n.tick + n.duration);
    const t = blankTrack(crypto.randomUUID(), d.name, "instrument");
    t.segments = [{
      id: crypto.randomUUID(),
      startTick: 0,
      durationTicks: Math.max(1, maxTick),
      content: { type: "notes", notes: d.notes },
    }];
    return t;
  });
}
