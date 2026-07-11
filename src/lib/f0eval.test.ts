// S51 Phase-5 gate — the pure pitch evaluators (SynthV-aligned layering, §10.3). Structural + exact-value
// tests; the COMPOSED golden-vector parity gate (overlay==Rust) lands in Phase 6 once the Rust render
// command exists (Option A: Rust consumes the TS cents array).
import { describe, it, expect } from "vitest";
import { interpShape } from "./interpolateShape";
import { evalF0CentsAt, evalF0CentsFrames, paintedDev, paintedParamCurve, sliceCurveAtTick, evalCurveAt } from "./f0eval";
import type { Note, NoteTransition } from "../types/project";

describe("interpolateShape", () => {
  it("easings hit boundaries + known midpoints, and clamp out-of-range t", () => {
    for (const s of ["linear", "sineIn", "sineOut", "sineInOut"] as const) {
      expect(interpShape(0, s)).toBeCloseTo(0, 6);
      expect(interpShape(1, s)).toBeCloseTo(1, 6);
    }
    expect(interpShape(0.5, "linear")).toBeCloseTo(0.5, 6);
    expect(interpShape(0.5, "sineIn")).toBeCloseTo(1 - Math.cos(Math.PI / 4), 6); // ≈0.2929 (ease-in)
    expect(interpShape(0.5, "sineOut")).toBeCloseTo(Math.sin(Math.PI / 4), 6); // ≈0.7071 (ease-out)
    expect(interpShape(0.5, "sineInOut")).toBeCloseTo(0.5, 6);
    expect(interpShape(-1, "sineIn")).toBe(0);
    expect(interpShape(2, "sineOut")).toBe(1);
  });
});

describe("evalF0Cents (SynthV-aligned layering §10.3)", () => {
  const mk = (id: string, tick: number, dur: number, pitch: number, extra: Partial<Note> = {}): Note =>
    ({ id, tick, duration: dur, pitch, lyric: "あ", velocity: 100, ...extra });
  // Pure staircase (all transition durations 0) = the back-end parity anchor.
  // openEdgeCents 0 on flat/smooth so the EXISTING tests keep their pre-§10.5 flat isolated onsets/releases.
  const flat: Required<NoteTransition> = { offsetMs: 0, durLeftMs: 0, durRightMs: 0, depthLeftCents: 0, depthRightCents: 0, openEdgeCents: 0 };
  const smooth: Required<NoteTransition> = { offsetMs: 0, durLeftMs: 100, durRightMs: 70, depthLeftCents: 15, depthRightCents: 15, openEdgeCents: 0 };
  const optsFlat = { tempo: 120, defaultTransition: flat };
  const optsSmooth = { tempo: 120, defaultTransition: smooth };
  // scoop: the §10.5 open-edge behaviour ON (100¢ reference below target) for the lead/release tests.
  const scoop: Required<NoteTransition> = { offsetMs: 0, durLeftMs: 100, durRightMs: 100, depthLeftCents: 15, depthRightCents: 15, openEdgeCents: 100 };
  const optsScoop = { tempo: 120, defaultTransition: scoop };

  it("flat transition = stepped base (pitch*100 + detune); a gap tick = unvoiced", () => {
    const notes = [mk("a", 0, 480, 60), mk("b", 480, 480, 62, { detune: 20 })];
    expect(evalF0CentsAt(notes, undefined, 100, optsFlat)).toEqual({ cents: 6000, voiced: true });
    expect(evalF0CentsAt(notes, undefined, 500, optsFlat)).toEqual({ cents: 6220, voiced: true }); // 62*100+20
    expect(evalF0CentsAt(notes, undefined, 2000, optsFlat)).toEqual({ cents: 0, voiced: false }); // past both = rest
  });

  it("PARITY ANCHOR: flat transition (duration=0) is a pure staircase even at an abutting boundary", () => {
    const notes = [mk("a", 0, 480, 60), mk("b", 480, 480, 72)];
    expect(evalF0CentsAt(notes, undefined, 479, optsFlat).cents).toBe(6000); // last tick of A
    expect(evalF0CentsAt(notes, undefined, 480, optsFlat).cents).toBe(7200); // onset of B (half-open, B wins)
  });

  it("smooth transition glides across the boundary (interval-replace, rises, lands on each tone)", () => {
    const notes = [mk("a", 0, 480, 60), mk("b", 480, 480, 72)]; // 6000 → 7200
    expect(evalF0CentsAt(notes, undefined, 100, optsSmooth).cents).toBeCloseTo(6000, 6); // stable A before span
    expect(evalF0CentsAt(notes, undefined, 800, optsSmooth).cents).toBeCloseTo(7200, 6); // stable B after span
    const mid = evalF0CentsAt(notes, undefined, 480, optsSmooth).cents; // at boundary = mid-glide
    expect(mid).toBeGreaterThan(6000);
    expect(mid).toBeLessThan(7200);
    const early = evalF0CentsAt(notes, undefined, 430, optsSmooth).cents; // in A's departure
    const late = evalF0CentsAt(notes, undefined, 540, optsSmooth).cents; // in B's arrival
    expect(early).toBeLessThan(late); // rising glide
    expect(early).toBeGreaterThanOrEqual(6000 - 20); // small departure overshoot allowed
    expect(late).toBeLessThanOrEqual(7200 + 20);
  });

  it("no glide across a large gap (real rest) — each note holds flat to its own edge", () => {
    const notes = [mk("a", 0, 240, 60), mk("b", 2000, 240, 72)]; // gap ≫ legato
    expect(evalF0CentsAt(notes, undefined, 230, optsSmooth).cents).toBeCloseTo(6000, 6); // A holds
    expect(evalF0CentsAt(notes, undefined, 2005, optsSmooth).cents).toBeCloseTo(7200, 6); // B holds
    expect(evalF0CentsAt(notes, undefined, 1000, optsSmooth).voiced).toBe(false); // the gap is unvoiced
  });

  it("per-note transition override extends B's arrival glide deeper into B", () => {
    const base = [mk("a", 0, 480, 60), mk("b", 480, 960, 72)];
    const over = [mk("a", 0, 480, 60), mk("b", 480, 960, 72, { transition: { durLeftMs: 400 } })];
    const at = 600; // 120 ticks past B's onset
    expect(evalF0CentsAt(base, undefined, at, optsSmooth).cents).toBeCloseTo(7200, 0); // default durLeft → arrived
    expect(evalF0CentsAt(over, undefined, at, optsSmooth).cents).toBeLessThan(7200 - 10); // longer durLeft → still arriving
  });

  it("same-pitch abutting notes have NO transition wobble (flat through the seam — no phantom bump)", () => {
    const same = [mk("a", 0, 480, 60), mk("b", 480, 480, 60)];
    for (const at of [430, 480, 540]) expect(evalF0CentsAt(same, undefined, at, optsSmooth).cents).toBeCloseTo(6000, 6);
  });

  it("overshoot follows glide direction: a down-glide dips below the target (not above)", () => {
    const down = [mk("a", 0, 480, 72), mk("b", 480, 480, 60)]; // 7200 → 6000
    const noOver = { tempo: 120, defaultTransition: { ...smooth, depthLeftCents: 0, depthRightCents: 0 } };
    const withOver = { tempo: 120, defaultTransition: { ...smooth, depthLeftCents: 40, depthRightCents: 40 } };
    const at = 500; // in the arrival part
    expect(evalF0CentsAt(down, undefined, at, withOver).cents).toBeLessThan(evalF0CentsAt(down, undefined, at, noOver).cents);
  });

  it("offsetMs shifts the whole transition crossover in time (NOT a dead knob)", () => {
    const notes = [mk("a", 0, 480, 60), mk("b", 480, 480, 72)];
    const shifted = { tempo: 120, defaultTransition: { ...smooth, offsetMs: 120 } }; // crossover moves later
    // at the boundary, a later crossover means the glide hasn't begun → still near A, below the un-shifted mid-glide
    expect(evalF0CentsAt(notes, undefined, 480, shifted).cents).toBeLessThan(evalF0CentsAt(notes, undefined, 480, optsSmooth).cents);
  });

  it("pitchDev is additive on top of the base", () => {
    const notes = [mk("a", 0, 480, 60)];
    const dev = { xs: [0, 480], ys: [0, 100] };
    const r = evalF0CentsAt(notes, dev, 240, optsFlat); // base 6000 + pitchDev(50)
    expect(r.voiced).toBe(true);
    expect(r.cents).toBeCloseTo(6050, 4);
  });

  it("vibrato: 0 before the onset delay, nonzero after, bounded by depth", () => {
    const vib = { depthCents: 50, freqHz: 5.5, phase: 0, startMs: 100, easeInMs: 0, easeOutMs: 0 };
    const notes = [mk("a", 0, 1920, 60, { vibrato: vib })]; // long note; startMs 100ms@120bpm ≈ 96 ticks
    expect(evalF0CentsAt(notes, undefined, 48, optsFlat).cents).toBe(6000); // before onset = base only
    const inside = evalF0CentsAt(notes, undefined, 960, optsFlat).cents;
    expect(Math.abs(inside - 6000)).toBeLessThanOrEqual(50 + 1e-6); // within ±depth
  });

  it("an off vibrato (depthCents 0) contributes nothing", () => {
    const notes = [mk("a", 0, 1920, 60, { vibrato: { depthCents: 0, freqHz: 5.5, phase: 0, startMs: 0, easeInMs: 0, easeOutMs: 0 } })];
    expect(evalF0CentsAt(notes, undefined, 960, optsFlat).cents).toBe(6000);
  });

  it("REGRESSION: vibrato on a lone/tail note (no next) modulates the line in its body (§user 尾音加不了颤音)", () => {
    // A note with NO following note must still wiggle. Uses the shipped default's small onset/ease so it's
    // visible without a multi-beat note (the old 250 ms onset + 200 ms fades suppressed it below ~2 beats).
    const vib = { depthCents: 100, freqHz: 5.5, phase: 0, startMs: 0, easeInMs: 80, easeOutMs: 120 };
    const notes = [mk("a", 0, 480, 60, { vibrato: vib })]; // a single 1-beat note, no neighbor
    expect(Math.abs(evalF0CentsAt(notes, undefined, 240, optsFlat).cents - 6000)).toBeGreaterThan(20); // not flat
  });

  it("evalF0CentsFrames yields the per-frame cents array + voiced mask", () => {
    const notes = [mk("a", 0, 100, 60)];
    const { cents, voiced } = evalF0CentsFrames(notes, undefined, { frameStartTick: 0, ticksPerFrame: 50, frameCount: 4 }, optsFlat);
    expect(cents.length).toBe(4);
    expect(Array.from(voiced)).toEqual([1, 1, 0, 0]); // ticks 0,50 ∈ [0,100); 100,150 rest
    expect(cents[0]).toBe(6000);
  });

  // ── §10.5 open-edge scoop-in / drift-out (isolated onset & release) ──
  it("open-edge ONSET: an isolated first note scoops UP from below, then SNAPS to pitch (snappier than a glide)", () => {
    const notes = [mk("a", 0, 480, 60)]; // single isolated note; openEdge 100¢, durLeft 100ms ≈ 96 ticks
    const onset = evalF0CentsAt(notes, undefined, 4, optsScoop).cents; // just after the start
    expect(onset).toBeLessThan(6000); // scooped BELOW the written pitch
    expect(onset).toBeGreaterThan(6000 - 100 - 1); // but not past the reference (tone − 100)
    // SNAP: the scoop is durLeft·OPEN_EDGE_SNAP (a fraction of the ~96-tick glide) — fully settled by tick 60,
    // long before a note-to-note glide (durLeft ≈96) would. A full-duration scoop would still be mid-rise here.
    expect(evalF0CentsAt(notes, undefined, 60, optsScoop).cents).toBeCloseTo(6000, 6);
  });

  it("open-edge RELEASE: an isolated last note drifts DOWN below its written pitch at the very end", () => {
    const notes = [mk("a", 0, 480, 60)];
    expect(evalF0CentsAt(notes, undefined, 479, optsScoop).cents).toBeLessThan(6000); // drifted below at the end
    expect(evalF0CentsAt(notes, undefined, 200, optsScoop).cents).toBeCloseTo(6000, 6); // body still flat
  });

  it("REGRESSION F1: a large negative offset can't tear the lead↔cross handoff (no pitch jump)", () => {
    // short A (isolated onset → leadCents) abuts B; B's big durLeft + very negative offset used to pull the
    // A→B glide back into A's scoop region → a huge one-tick jump. The midpoint-bounded offset clamp fixes it.
    const A = mk("a", 0, 100, 60);
    const B = mk("b", 100, 480, 67, { transition: { durLeftMs: 400, offsetMs: -400 } });
    const notes = [A, B];
    let prev = evalF0CentsAt(notes, undefined, 0, optsScoop).cents;
    let maxJump = 0;
    for (let x = 1; x <= 100; x++) { // sweep A's whole span
      const c = evalF0CentsAt(notes, undefined, x, optsScoop).cents;
      maxJump = Math.max(maxJump, Math.abs(c - prev));
      prev = c;
    }
    expect(maxJump).toBeLessThan(30); // continuous (a real glide moves <30¢ per tick here); pre-fix was ~665¢
  });

  it("open-edge scoop is OFF at 0 (parity) and never fires at a CONNECTED boundary", () => {
    const notes = [mk("a", 0, 480, 60)];
    const off = { tempo: 120, defaultTransition: { ...scoop, openEdgeCents: 0 } };
    expect(evalF0CentsAt(notes, undefined, 4, off).cents).toBeCloseTo(6000, 6); // openEdge 0 → flat onset
    // b abuts p → b's onset GLIDES from p (crossCents, above b's pitch), NOT a scoop from below
    const seq = [mk("p", 0, 240, 62), mk("b", 240, 240, 60)];
    expect(evalF0CentsAt(seq, undefined, 244, optsScoop).cents).toBeGreaterThan(6000);
  });

  // ── paintedDev: SynthV-pencil interval-replace with a fixed non-bleeding edge reconnect ──
  it("paintedDev: stores the painted delta; edges return to 0 (no 整条平移 bleed)", () => {
    const notes = [mk("a", 0, 480, 60)]; // flat auto-line = 6000¢
    const c = paintedDev(notes, { xs: [240], ys: [6200] }, undefined, optsFlat); // +200¢ at tick 240
    expect(c.ys[0]).toBe(0); // leftmost edge = original pitch (zero anchor a pad outside)
    expect(c.ys[c.ys.length - 1]).toBe(0); // rightmost edge = original pitch
    expect(Math.max(...c.ys)).toBe(200); // painted delta stored exactly (interval-replace)
    expect(c.xs.length).toBe(3); // loA(0), painted(240,+200), hiA(0)
  });

  it("paintedDev: a multi-point stroke keeps its interior points + zero edges", () => {
    const notes = [mk("a", 0, 480, 60)];
    const c = paintedDev(notes, { xs: [200, 240, 280], ys: [6100, 6300, 6100] }, undefined, optsFlat);
    expect(c.ys[0]).toBe(0);
    expect(c.ys[c.ys.length - 1]).toBe(0);
    expect(Math.max(...c.ys)).toBe(300); // +300¢ peak preserved exactly
  });
});

describe("② param lanes — paintedParamCurve + sliceCurveAtTick", () => {
  it("paintedParamCurve: stores the ABSOLUTE drawn value (no delta), zero-anchored outside the span", () => {
    const c = paintedParamCurve({ xs: [200, 240, 280], ys: [3, 5, 3] }, undefined);
    expect(c.ys[0]).toBe(0); // loA zero anchor
    expect(c.ys[c.ys.length - 1]).toBe(0); // hiA zero anchor
    expect(Math.max(...c.ys)).toBe(5); // absolute painted value (unlike paintedDev, no baseline subtraction)
    expect(evalCurveAt(c, 240)).toBeCloseTo(5, 6); // inside the span → the painted value
    expect(evalCurveAt(c, 0)).toBe(0); // outside → neutral 0 (no 整条平移 bleed)
  });

  it("sliceCurveAtTick: partitions + rebases each half with a lossless seam", () => {
    const { left, right } = sliceCurveAtTick({ xs: [0, 600, 1000], ys: [0, 50, 0] }, 300);
    expect(left).toBeDefined();
    expect(evalCurveAt(left, 0)).toBeCloseTo(0, 6);
    expect(evalCurveAt(left, 300)).toBeCloseTo(25, 6); // interp of 0@0..50@600 at the seam
    expect(right).toBeDefined();
    expect(evalCurveAt(right, 0)).toBeCloseTo(25, 6); // rebased seam sample at x=0
    expect(evalCurveAt(right, 300)).toBeCloseTo(50, 6); // 600 − 300
    expect(evalCurveAt(right, 700)).toBeCloseTo(0, 6); // 1000 − 300
  });

  it("sliceCurveAtTick: undefined curve → empty halves", () => {
    expect(sliceCurveAtTick(undefined, 100)).toEqual({});
  });

  it("sliceCurveAtTick: a flat held region survives on BOTH halves — no reset to neutral (§user split)", () => {
    // a single point at 1000 (+3) → the curve holds +3 EVERYWHERE (flat). Split at 500 → both halves flat +3
    // (the new segment's lane keeps +3dB, instead of the point-less left half dropping to neutral 0).
    const { left, right } = sliceCurveAtTick({ xs: [1000], ys: [3] }, 500);
    expect(left).toBeDefined();
    expect(evalCurveAt(left, 0)).toBeCloseTo(3, 6);
    expect(evalCurveAt(left, 500)).toBeCloseTo(3, 6);
    expect(right).toBeDefined();
    expect(evalCurveAt(right, 0)).toBeCloseTo(3, 6);
    expect(evalCurveAt(right, 500)).toBeCloseTo(3, 6);
  });
});
