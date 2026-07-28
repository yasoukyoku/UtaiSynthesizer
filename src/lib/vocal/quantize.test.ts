// S87 — the shared grid-quantization contract used by score import (ust/ustx/midi) and GAME vocal→MIDI
// extraction. The properties that matter, and why:
//   1. legato survives — a shared boundary between two contiguous notes rounds identically (BOUNDARY
//      quantization, not per-note duration rounding, which accumulates drift);
//   2. nothing vanishes silently — under "widen" (import: every note carries a lyric) a note shorter than
//      half a cell becomes one cell and the follower yields that cell from its START ONLY; under "drop"
//      (extraction: pitch-tracker artifacts) it is discarded and COUNTED, never a bare `continue`;
//   3. a chord stays a chord — the anti-overlap push is anchored to the last WIDENED note, so quantizing a
//      polyphonic .mid can neither arpeggiate it nor stack disjoint notes onto one span.
import { describe, it, expect } from "vitest";
import { quantizeSpans, offGridCount, GRID_QUANT_TICKS, type QuantSpan } from "./quantize";

const Q = GRID_QUANT_TICKS; // 40t

describe("quantizeSpans — boundary rounding", () => {
  it("already-on-grid material is untouched (no false 'moved')", () => {
    const input: QuantSpan[] = [{ start: 0, end: 240 }, { start: 480, end: 720 }];
    const r = quantizeSpans(input);
    expect(r.spans).toEqual(input);
    expect(r.moved).toBe(0);
    expect(r.widened).toBe(0);
    expect(r.dropped).toBe(0);
  });

  it("contiguous notes stay contiguous — the shared boundary rounds ONCE", () => {
    const r = quantizeSpans([{ start: 0, end: 137 }, { start: 137, end: 400 }]);
    expect(r.spans[0]!.end).toBe(r.spans[1]!.start); // legato preserved
    expect(r.spans[0]!.end).toBe(120); // 137 -> nearest 40
  });

  it("a gap stays a gap", () => {
    const r = quantizeSpans([{ start: 0, end: 100 }, { start: 300, end: 500 }]);
    expect(r.spans[0]!.end).toBe(120);
    expect(r.spans[1]!.start).toBe(320);
    expect(r.spans[1]!.start).toBeGreaterThan(r.spans[0]!.end);
  });

  it("index order of the OUTPUT matches the INPUT even for unsorted input", () => {
    const r = quantizeSpans([{ start: 480, end: 720 }, { start: 0, end: 240 }]);
    expect(r.spans[0]).toEqual({ start: 480, end: 720 });
    expect(r.spans[1]).toEqual({ start: 0, end: 240 });
  });
});

describe('quantizeSpans "widen" — collapse is widened, never dropped', () => {
  it("a sub-half-cell note becomes one cell and the follower yields only its START", () => {
    // [0,15) is 15t — both edges round to 0. Pre-S87 midiExtract dropped exactly this note.
    const r = quantizeSpans([{ start: 0, end: 15 }, { start: 15, end: 500 }], "widen");
    expect(r.spans).toHaveLength(2); // nothing disappeared
    expect(r.spans[0]).toEqual({ start: 0, end: Q });
    expect(r.spans[1]!.start).toBe(Q); // gave up the cell the widened note took
    expect(r.spans[1]!.end).toBe(520); // ...but its END is untouched: downstream does not shift
    expect(r.widened).toBe(1);
    expect(r.dropped).toBe(0);
  });

  it("a RUN of collapsed notes lays out one cell each; the cascade stops at the first note with room", () => {
    const r = quantizeSpans([{ start: 0, end: 10 }, { start: 10, end: 20 }, { start: 20, end: 500 }], "widen");
    expect(r.spans[0]).toEqual({ start: 0, end: Q });
    expect(r.spans[1]).toEqual({ start: Q, end: 2 * Q });
    expect(r.spans[2]).toEqual({ start: 2 * Q, end: 520 }); // end still the rounded original
    // only the FIRST was genuinely under half a cell: [10,20) rounds to [0,40) on its own (Math.round(0.5)=1),
    // so it is merely pushed — the count must not claim it was unsingable.
    expect(r.widened).toBe(1);
  });

  it("every surviving span is at least one cell long (nothing can round to zero length)", () => {
    const r = quantizeSpans([{ start: 3, end: 4 }, { start: 1000, end: 1001 }], "widen");
    for (const s of r.spans) expect(s!.end - s!.start).toBeGreaterThanOrEqual(Q);
  });

  it("the 'widened' count means TOO SHORT — a note merely pushed by a neighbour is not counted", () => {
    // [15,55) rounds to exactly one cell on its own; it is only re-widened because the note before it
    // claimed its first cell. Counting it would make the toast claim 2 unsingable notes when there was 1.
    const r = quantizeSpans([{ start: 0, end: 15 }, { start: 15, end: 55 }], "widen");
    expect(r.widened).toBe(1);
    expect(r.spans[1]).toEqual({ start: Q, end: 2 * Q });
  });
});

describe('quantizeSpans "drop" — extraction policy', () => {
  it("collapsed notes are dropped and counted; the survivors keep the timing HEAD gave them", () => {
    const r = quantizeSpans([{ start: 0, end: 300 }, { start: 300, end: 310 }, { start: 310, end: 800 }], "drop");
    expect(r.spans[1]).toBeNull();
    expect(r.dropped).toBe(1);
    expect(r.widened).toBe(0);
    // the survivor's start is its OWN rounded onset — no cell was stolen from it, so no drift
    expect(r.spans[0]).toEqual({ start: 0, end: 320 }); // round(300/40) = round(7.5) = 8
    expect(r.spans[2]).toEqual({ start: 320, end: 800 });
  });

  it("a long run of micro-notes causes ZERO cumulative drift for the following real note", () => {
    // eight contiguous 20t artifacts then a real sustain — under "widen" the sustain's onset would be
    // shoved 8 cells late; under "drop" it lands exactly where its own onset rounds to.
    const spans: QuantSpan[] = [];
    for (let i = 0; i < 8; i++) spans.push({ start: i * 20, end: (i + 1) * 20 });
    spans.push({ start: 160, end: 760 });
    const r = quantizeSpans(spans, "drop");
    expect(r.dropped).toBe(4); // the ones whose two edges land on the same line
    expect(r.spans[8]).toEqual({ start: 160, end: 760 });
  });
});

describe("quantizeSpans — overlapping input is left overlapping (no arpeggiation, no stacking)", () => {
  it("a chord (identical spans) quantizes to the same span, not to a sequence", () => {
    const r = quantizeSpans([{ start: 0, end: 240 }, { start: 0, end: 240 }, { start: 0, end: 240 }]);
    expect(r.spans[0]).toEqual({ start: 0, end: 240 });
    expect(r.spans[1]).toEqual({ start: 0, end: 240 });
    expect(r.spans[2]).toEqual({ start: 0, end: 240 });
  });

  it("partially overlapping notes keep their overlap", () => {
    const r = quantizeSpans([{ start: 0, end: 300 }, { start: 100, end: 400 }]);
    expect(r.spans[1]!.start).toBe(120); // round(100/40) = round(2.5) = 3 → 120; NOT pushed to the first note's end
    expect(r.spans[1]!.start).toBeLessThan(r.spans[0]!.end);
  });

  it("a HELD note must not disable the anti-overlap push for the short notes under it", () => {
    // Review finding: with the push anchored to a running max over ALL previous ends, the held note's end
    // (2000) suppressed the push for every following short note, and widening then stacked all three onto
    // ONE identical span — a three-note melodic run became three lyrics on the same instant.
    const r = quantizeSpans([
      { start: 0, end: 2000 },
      { start: 100, end: 115 },
      { start: 115, end: 130 },
      { start: 130, end: 145 },
    ], "widen");
    expect(r.spans[0]).toEqual({ start: 0, end: 2000 }); // the held note is untouched
    const shorts = [r.spans[1]!, r.spans[2]!, r.spans[3]!];
    expect(new Set(shorts.map((s) => s.start)).size).toBe(3); // three DISTINCT onsets
    expect(shorts).toEqual([
      { start: 120, end: 160 },
      { start: 160, end: 200 },
      { start: 200, end: 240 },
    ]);
  });
});

describe("offGridCount — the number the import dialog shows", () => {
  it("counts starts that are not on a line", () => {
    expect(offGridCount([{ start: 0, end: 40 }, { start: 37, end: 80 }, { start: 80, end: 120 }])).toBe(1);
    expect(offGridCount([])).toBe(0);
  });
});
