// S60-2 音域测试 — classification math gates (the v1 criteria must not drift).
import { describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn() }));
vi.mock("@tauri-apps/api/event", () => ({ listen: vi.fn() }));
vi.mock("../../i18n", () => ({ default: { t: (k: string) => k } }));

import {
  buildScaleScore,
  classifySemitones,
  deriveRanges,
  buildSpeakerRecord,
  midiToHz,
  midiName,
  RANGE_MIDI_LO,
  RANGE_MIDI_HI,
  type SemitoneStat,
} from "./rangeTest";

function stat(midi: number, errCents: number, voicedRatio: number): SemitoneStat {
  return { midi, errCents, voicedRatio };
}

describe("buildScaleScore", () => {
  it("covers C2..C8 with contiguous frames and aligned 100fps spans", () => {
    const { triples, spans } = buildScaleScore();
    expect(spans.length).toBe(RANGE_MIDI_HI - RANGE_MIDI_LO + 1);
    // triples tile the timeline: Σframes*2 == the last span end + trailing rest
    const total50 = triples.reduce((a, t) => a + t.frames, 0);
    expect(spans[spans.length - 1]!.end100).toBe((total50 - 6) * 2);
    // spans are note-only windows, monotonically increasing
    for (let i = 1; i < spans.length; i++) {
      expect(spans[i]!.start100).toBeGreaterThan(spans[i - 1]!.end100 - 1);
    }
  });
});

describe("classifySemitones", () => {
  it("measures a perfect note as ~0 cents with full voicing, and erodes edges", () => {
    const { spans } = buildScaleScore();
    const span = spans[10]!;
    const f0 = new Array(spans[spans.length - 1]!.end100 + 8).fill(0);
    for (let i = span.start100; i < span.end100; i++) f0[i] = midiToHz(span.midi);
    // poison the edge frames — erosion must exclude them from the stats
    f0[span.start100] = midiToHz(span.midi) * 2;
    f0[span.end100 - 1] = midiToHz(span.midi) / 2;
    const stats = classifySemitones(f0, spans);
    expect(stats[10]!.errCents).toBeLessThan(1);
    expect(stats[10]!.voicedRatio).toBe(1);
    // a fully silent note reads unvoiced/Infinity
    expect(stats[0]!.voicedRatio).toBe(0);
    expect(stats[0]!.errCents).toBe(Infinity);
  });
});

describe("deriveRanges (v1 criteria)", () => {
  it("usable=<100¢&voiced>50%, comfort=<50¢&voiced>80%, contiguous runs", () => {
    const stats: SemitoneStat[] = [];
    for (let m = 36; m <= 96; m++) {
      if (m >= 48 && m <= 84) {
        const comfy = m >= 52 && m <= 79;
        stats.push(stat(m, comfy ? 20 : 80, comfy ? 0.95 : 0.6));
      } else {
        stats.push(stat(m, 500, 0.1));
      }
    }
    const r = deriveRanges(stats)!;
    expect(r.usable).toEqual([48, 84]);
    expect(r.comfort).toEqual([52, 79]);
  });

  it("takes the LONGEST contiguous usable run (a stray good semitone far away doesn't win)", () => {
    const stats: SemitoneStat[] = [];
    for (let m = 36; m <= 96; m++) {
      const inMain = m >= 55 && m <= 75;
      const stray = m === 40;
      stats.push(inMain || stray ? stat(m, 10, 1) : stat(m, 400, 0.2));
    }
    const r = deriveRanges(stats)!;
    expect(r.usable).toEqual([55, 75]);
  });

  it("comfort falls back to usable when nothing reaches comfort grade; null when nothing usable", () => {
    const stats: SemitoneStat[] = [];
    for (let m = 36; m <= 96; m++) {
      stats.push(m >= 60 && m <= 70 ? stat(m, 90, 0.6) : stat(m, 900, 0));
    }
    const r = deriveRanges(stats)!;
    expect(r.comfort).toEqual(r.usable);
    expect(deriveRanges(stats.map((s) => stat(s.midi, 900, 0)))).toBeNull();
  });
});

describe("buildSpeakerRecord", () => {
  it("stores the raw per-semitone scan and comfort_auto = detected comfort", () => {
    const stats: SemitoneStat[] = [];
    for (let m = 36; m <= 96; m++) stats.push(m >= 50 && m <= 80 ? stat(m, 15, 0.97) : stat(m, Infinity, 0));
    const rec = buildSpeakerRecord(stats)!;
    expect(rec.comfort).toEqual(rec.comfort_auto);
    expect(rec.semitones["50"]).toEqual([15, 0.97]);
    expect(rec.semitones["36"]![0]).toBe(9999); // Infinity is stored finitely (JSON-safe)
  });
});

describe("midiName", () => {
  it("labels C4=60 and friends", () => {
    expect(midiName(60)).toBe("C4");
    expect(midiName(48)).toBe("C3");
    expect(midiName(69)).toBe("A4");
    expect(midiName(61)).toBe("C#4");
  });
});
