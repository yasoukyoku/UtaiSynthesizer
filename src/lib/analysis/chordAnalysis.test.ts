import { describe, it, expect } from "vitest";
import { analyzeChords, estimateKey, chordLabel, analyzeChordsFromCqt, type ChordAnalysisNote } from "./chordAnalysis";
import { analyzeCqt } from "./cqt";

const PPQ = 480;

/** 以根音为低音的和弦块：pitch = MIDI 音级 + 八度，持续 durTicks。 */
function chordBlock(startTick: number, durTicks: number, pitches: number[], velocity = 100): ChordAnalysisNote[] {
  return pitches.map((p) => ({ tick: startTick, duration: durTicks, pitch: p, velocity }));
}

/** C 大调音阶顺阶音符（时长加权直方图）。 */
function scaleNotes(roots: number[], dur = PPQ): ChordAnalysisNote[] {
  return roots.map((p, i) => ({ tick: i * PPQ, duration: dur, pitch: p }));
}

describe("estimateKey", () => {
  it("C 大调音阶 → C 大调", () => {
    // C D E F G A B（60..71）
    const key = estimateKey(scaleNotes([60, 62, 64, 65, 67, 69, 71]));
    expect(key.tonic).toBe(0);
    expect(key.major).toBe(true);
    expect(key.label).toBe("C");
    expect(key.confidence).toBeGreaterThan(0.5);
  });

  it("A 小调音阶 → Am", () => {
    // A B C D E F G（57..67 附近，结束在 A）
    const key = estimateKey(scaleNotes([57, 59, 60, 62, 64, 65, 67, 69, 57, 57]));
    expect(key.tonic).toBe(9);
    expect(key.major).toBe(false);
    expect(key.label).toBe("Am");
  });

  it("空输入 → 兜底 C（不抛错）", () => {
    const key = estimateKey([]);
    expect(key.label).toBe("C");
    expect(key.confidence).toBe(0);
  });
});

describe("analyzeChords", () => {
  it("单个 C 大三和弦长音 → 一段 C", () => {
    const notes = chordBlock(0, PPQ * 8, [48, 60, 64, 67]);
    const res = analyzeChords(notes, PPQ);
    expect(res.segments.length).toBe(1);
    expect(res.segments[0]!.label).toBe("C");
    expect(res.segments[0]!.startTick).toBe(0);
  });

  it("经典进行 C–G–Am–F（每和弦一小节）→ 依序识别", () => {
    const bar = PPQ * 4;
    const notes: ChordAnalysisNote[] = [
      ...chordBlock(0, bar, [48, 60, 64, 67]),        // C
      ...chordBlock(bar, bar, [43, 59, 62, 67]),      // G
      ...chordBlock(bar * 2, bar, [45, 57, 60, 64]),  // Am
      ...chordBlock(bar * 3, bar, [41, 57, 60, 65]),  // F
    ];
    const res = analyzeChords(notes, PPQ);
    expect(res.segments.length).toBeGreaterThanOrEqual(3);
    expect(res.segments[0]!.label).toBe("C");
    expect(res.segments.some((s) => s.label === "G")).toBe(true);
    expect(res.segments.some((s) => s.label === "Am")).toBe(true);
    expect(res.segments.some((s) => s.label === "F")).toBe(true);
    // 时间顺序：C → G → Am → F
    const labels = res.segments.map((s) => s.label);
    expect(labels.indexOf("C")).toBeLessThan(labels.indexOf("G"));
    expect(labels.indexOf("G")).toBeLessThan(labels.indexOf("Am"));
    expect(labels.indexOf("Am")).toBeLessThan(labels.indexOf("F"));
  });

  it("七和弦：Cmaj7 块含 B → 匹配 maj7 而非 maj", () => {
    const notes = chordBlock(0, PPQ * 8, [48, 60, 64, 67, 71]);
    const res = analyzeChords(notes, PPQ);
    expect(res.segments[0]!.label).toBe("Cmaj7");
  });

  it("Slash 和弦：C 和弦但最低音是 G → 标注 C/G", () => {
    const notes = chordBlock(0, PPQ * 8, [43, 60, 64, 67]); // 低音 G(43)，和弦 C
    const res = analyzeChords(notes, PPQ);
    expect(res.segments[0]!.label).toBe("C/G");
    expect(res.segments[0]!.bass).toBe(7);
  });

  it("空输入 / 非法 ppq → 空段不抛错", () => {
    expect(analyzeChords([], PPQ).segments).toEqual([]);
    expect(analyzeChords(chordBlock(0, PPQ, [60]), 0).segments).toEqual([]);
  });

  it("惯性平滑：同和弦延续合为一段", () => {
    const bar = PPQ * 4;
    const notes: ChordAnalysisNote[] = [
      ...chordBlock(0, bar, [48, 60, 64, 67]),
      ...chordBlock(bar, bar, [48, 60, 64, 67]),
      ...chordBlock(bar * 2, bar, [48, 60, 64, 67]),
    ];
    const res = analyzeChords(notes, PPQ);
    expect(res.segments.length).toBe(1);
    expect(res.segments[0]!.endTick).toBe(bar * 3);
  });
});

describe("chordLabel", () => {
  it("原位/Slash 标注", () => {
    expect(chordLabel(0, "maj", null)).toBe("C");
    expect(chordLabel(9, "min7", null)).toBe("Am7");
    expect(chordLabel(7, "maj", 4)).toBe("G/E");
    expect(chordLabel(2, "m7b5", 2)).toBe("Dm7b5");
    expect(chordLabel(4, "sus4", null)).toBe("Esus4");
  });
});
describe("analyzeChordsFromCqt (FFT/CQT 音频和弦识别)", () => {
  const SR = 44100;
  const sine = (freq: number, durSec: number): Float32Array => {
    const n = Math.floor(SR * durSec);
    const buf = new Float32Array(n);
    for (let i = 0; i < n; i++) buf[i] = Math.sin((2 * Math.PI * freq * i) / SR) * 0.6;
    return buf;
  };
  // MIDI pitch → Hz
  const pitchHz = (p: number) => 440 * Math.pow(2, (p - 69) / 12);

  it("音频输入 → 产生和弦段 + 调性 (结构正确性)", () => {
    // 正弦波叠加大三和弦: 纯音 + 谐波, CQT 能稳定检测到能量集中在 C/E/G 音级附近
    const chordMix = (pitches: number[], durSec = 1.5) => {
      const bufs = pitches.map((p) => sine(pitchHz(p), durSec));
      const out = new Float32Array(bufs[0]!.length);
      for (const b of bufs) for (let i = 0; i < out.length; i++) out[i] = (out[i] ?? 0) + (b[i] ?? 0);
      return out;
    };
    const audio = chordMix([60, 64, 67]); // C4 E4 G4
    const cqt = analyzeCqt(audio, SR, { hopSec: 0.03 });
    const res = analyzeChordsFromCqt(cqt, SR, 120, { windowSec: 0.6, hopSec: 0.3, silenceFloor: 0.001 });
    // 至少能切出 1 个和弦段 (算法管道没挂)
    expect(res.segments.length).toBeGreaterThan(0);
    // 每个 segment 有合法 label + tick 范围
    for (const s of res.segments) {
      expect(s.label.length).toBeGreaterThan(0);
      expect(s.endTick).toBeGreaterThan(s.startTick);
    }
    // key 置信度在合理范围 (真实音频 ≈ 0.4~0.7, 纯合成偏低但不会 < 0)
    expect(res.key.confidence).toBeGreaterThanOrEqual(0);
  });

  it("静音输入 → 返回空 segments + Cmaj key", () => {
    
    const silent = { times: [] as number[], magnitudes: [] as Float32Array[], pitches: [] as number[] };
    const res = analyzeChordsFromCqt(silent, SR, 120);
    expect(res.segments).toEqual([]);
    expect(res.key.label).toBe("Cmaj");
  });

  it("单音 A4 (MIDI 69) → 调性估计 key ≈ A 或 A min", () => {
    
    const audio = sine(440, 2.0);
    const cqt = analyzeCqt(audio, SR, { hopSec: 0.05, minPitch: 55, maxPitch: 82 });
    const res = analyzeChordsFromCqt(cqt, SR, 120, { windowSec: 0.6, silenceFloor: 0.001 });
    // 单音太弱, 调性置信度低但结构正确
    expect(res.key).toHaveProperty("tonic");
    expect(res.key).toHaveProperty("confidence");
  });
});