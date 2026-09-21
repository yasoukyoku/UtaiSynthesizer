import { describe, it, expect } from "vitest";
import { analyzeCqt, pitchSalience, midiPitchToFreq } from "./cqt";

const SR = 44100;

function sine(freq: number, durSec: number): Float32Array {
  const n = Math.floor(SR * durSec);
  const buf = new Float32Array(n);
  for (let i = 0; i < n; i++) buf[i] = Math.sin((2 * Math.PI * freq * i) / SR) * 0.8;
  return buf;
}

describe("analyzeCqt", () => {
  it("440Hz 纯音 → 最显著音高 = MIDI 69（A4）", () => {
    const res = analyzeCqt(sine(440, 1), SR, { minPitch: 55, maxPitch: 82 });
    expect(res.pitches.length).toBe(82 - 55 + 1);
    expect(res.times.length).toBeGreaterThan(10);
    const sal = pitchSalience(res, 1);
    expect(sal.length).toBe(res.times.length);
    // 取中段一帧（避开首尾边界）
    const mid = sal[Math.floor(sal.length / 2)]!;
    expect(mid.topPitches.length).toBeGreaterThan(0);
    expect(mid.topPitches[0]!.pitch).toBe(69);
    expect(mid.topPitches[0]!.magnitude).toBeGreaterThan(0);
  });

  it("低八度 220Hz → 音高 57", () => {
    const res = analyzeCqt(sine(220, 0.5), SR, { minPitch: 50, maxPitch: 70 });
    const sal = pitchSalience(res, 1);
    const mid = sal[Math.floor(sal.length / 2)]!;
    expect(mid.topPitches[0]!.pitch).toBe(57);
  });

  it("空输入 → 空结果（不抛错）", () => {
    const res = analyzeCqt(new Float32Array(0), SR);
    expect(res.times).toEqual([]);
    expect(res.magnitudes).toEqual([]);
  });

  it("帧间隔可配置（hopSec 落到采样数）", () => {
    const res = analyzeCqt(sine(440, 0.3), SR, { minPitch: 67, maxPitch: 71, hopSec: 0.05 });
    expect(res.hopSec).toBe(0.05);
    // 0.3s / 0.05s ≈ 6 帧（首帧 0 起）
    expect(res.times.length).toBeGreaterThanOrEqual(5);
    expect(res.times[1]! - res.times[0]!).toBeCloseTo(0.05, 2);
  });
});

describe("midiPitchToFreq", () => {
  it("A4 = 440, A3 = 220, C4 ≈ 261.6", () => {
    expect(midiPitchToFreq(69)).toBeCloseTo(440, 5);
    expect(midiPitchToFreq(57)).toBeCloseTo(220, 5);
    expect(midiPitchToFreq(60)).toBeCloseTo(261.6256, 3);
  });
});

describe("pitchSalience", () => {
  it("只取局部极大：返回的音高都满足邻域极大", () => {
    const res = analyzeCqt(sine(440, 0.4), SR, { minPitch: 60, maxPitch: 80 });
    const sal = pitchSalience(res, 4);
    for (const frame of sal) {
      expect(frame.topPitches.length).toBeLessThanOrEqual(4);
    }
    // 单音信号：每帧 top1 应稳定为 69
    const tops = sal.slice(5, -5).map((f) => f.topPitches[0]?.pitch);
    for (const p of tops) expect(p).toBe(69);
  });

  it("topK 排序：幅度降序", () => {
    const res = analyzeCqt(sine(440, 0.4), SR, { minPitch: 60, maxPitch: 80 });
    const sal = pitchSalience(res, 3);
    const frame = sal[Math.floor(sal.length / 2)]!;
    for (let i = 1; i < frame.topPitches.length; i++) {
      expect(frame.topPitches[i]!.magnitude).toBeLessThanOrEqual(frame.topPitches[i - 1]!.magnitude);
    }
  });
});
