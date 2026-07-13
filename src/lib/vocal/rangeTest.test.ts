// S60-2 音域测试 — classification math gates (the v1 criteria must not drift).
import { describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn() }));
vi.mock("@tauri-apps/api/event", () => ({ listen: vi.fn() }));
vi.mock("../../i18n", () => ({ default: { t: (k: string) => k } }));

import {
  buildScaleScore,
  classifySemitones,
  deriveRanges,
  deriveCautionZones,
  buildSpeakerRecord,
  clampComfort,
  effectiveComfort,
  midiToHz,
  midiName,
  MIN_COMFORT_SPAN,
  RANGE_MIDI_LO,
  RANGE_MIDI_HI,
  type SemitoneStat,
  type SpeakerRangeRecord,
} from "./rangeTest";

function stat(midi: number, errCents: number, voicedRatio: number): SemitoneStat {
  return { midi, errCents, voicedRatio };
}

describe("buildScaleScore", () => {
  it("covers C2..C7 with contiguous frames and aligned 100fps spans", () => {
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

describe("deriveRanges noise bridging (S60d)", () => {
  it("bridges isolated 1-wide octave-flip dropouts (the lengv2.3 field case)", () => {
    // clean passes 36..77 except single 1180¢ points at 57 and 61 — without bridging the
    // longest run is [36,56] (ceiling truncated by 21 st); bridged it must be [36,77]
    const stats: SemitoneStat[] = [];
    for (let m = 36; m <= 96; m++) {
      if (m === 57 || m === 61) stats.push(stat(m, 1180, 1));
      else if (m <= 77) stats.push(stat(m, 5, 1));
      else stats.push(stat(m, 3800, 0));
    }
    const r = deriveRanges(stats)!;
    expect(r.usable).toEqual([36, 77]);
    expect(r.comfort).toEqual([36, 77]);
  });

  it("does NOT bridge a real saturation gap (>2 wide) or leading/trailing failures", () => {
    // passes 42..70 and 76..79, real 5-wide failure at 71..75 (the 風音サヨ field case)
    const stats: SemitoneStat[] = [];
    for (let m = 36; m <= 96; m++) {
      if ((m >= 42 && m <= 70) || (m >= 76 && m <= 79)) stats.push(stat(m, 5, 1));
      else stats.push(stat(m, 1500, 1));
    }
    const r = deriveRanges(stats)!;
    expect(r.usable).toEqual([42, 70]); // the island at 76-79 stays a separate (losing) run
  });
});

describe("comfort guards (S60d)", () => {
  it("clampComfort enforces the minimum span within usable", () => {
    // the field disaster verbatim: both sliders at the usable floor
    expect(clampComfort([42, 70], [42, 42])).toEqual([42, 42 + MIN_COMFORT_SPAN]);
    // span enforced against the ceiling too
    expect(clampComfort([42, 70], [70, 70])).toEqual([70 - MIN_COMFORT_SPAN, 70]);
    // honest wide zones pass through (sorted + clamped only)
    expect(clampComfort([42, 70], [60, 50])).toEqual([50, 60]);
    // usable narrower than the minimum → the whole usable zone
    expect(clampComfort([60, 63], [61, 61])).toEqual([60, 63]);
  });

  it("effectiveComfort mirrors the Rust read-side healing chain", () => {
    const rec = (comfort: [number, number], auto: [number, number]): SpeakerRangeRecord => ({
      usable: [42, 70], comfort, comfort_auto: auto, semitones: {}, tested_at: "2026-07-12",
    });
    expect(effectiveComfort(rec([42, 42], [42, 70]))).toEqual([42, 70]); // degenerate → auto
    expect(effectiveComfort(rec([42, 42], [50, 52]))).toEqual([42, 70]); // auto degenerate → usable
    expect(effectiveComfort(rec([45, 60], [42, 70]))).toEqual([45, 60]); // healthy stored value wins
  });
});

describe("deriveCautionZones (S60d3 model-quirk chips)", () => {
  it("finds 'sings confidently wrong' artifact runs near usable (風音サヨ shape)", () => {
    // usable [42,70]; 71-73 voiced but 1223-2410¢ off; 74 at 187¢ (below the 200¢ bar);
    // 80-82 the saturation ramp start (327-535¢, voiced); everything past 82 out of window
    const semis: Record<string, [number, number]> = {};
    for (let m = 36; m <= 96; m++) {
      if (m >= 42 && m <= 70) semis[m] = [5, 1];
      else if (m >= 71 && m <= 73) semis[m] = [1223 + (m - 71) * 590, 1];
      else if (m === 74) semis[m] = [187, 0.75];
      else if (m >= 75 && m <= 79) semis[m] = [9999, 0];
      else semis[m] = [327 + (m - 80) * 100, 1];
    }
    const z = deriveCautionZones(semis, [42, 70]);
    expect(z.artifact).toEqual([[71, 73], [80, 82]]); // window caps at usable[1]+12 = 82
    expect(z.weak).toEqual([]);
  });

  it("finds in-usable bridged weak notes (lengv2.3 shape) and ignores single outside points", () => {
    const semis: Record<string, [number, number]> = {};
    for (let m = 36; m <= 96; m++) {
      if (m === 57 || m === 61) semis[m] = [1180, 1]; // octave-flip points INSIDE usable
      else if (m <= 77) semis[m] = [5, 1];
      else if (m === 80) semis[m] = [3655, 0.75]; // isolated (81+ unvoiced) → no ≥2 run
      else semis[m] = [9999, 0];
    }
    const z = deriveCautionZones(semis, [36, 77]);
    expect(z.weak).toEqual([57, 61]);
    expect(z.artifact).toEqual([]);
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
