// S51 Phase-5 gate — the pure pitch evaluators (SynthV-aligned layering, §10.3). Structural + exact-value
// tests; the COMPOSED golden-vector parity gate (overlay==Rust) lands in Phase 6 once the Rust render
// command exists (Option A: Rust consumes the TS cents array).
import { describe, it, expect } from "vitest";
import { interpShape } from "./interpolateShape";
import { evalF0CentsAt, evalF0CentsFrames } from "./f0eval";
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
  const flat: Required<NoteTransition> = { offsetMs: 0, durLeftMs: 0, durRightMs: 0, depthLeftCents: 0, depthRightCents: 0 };
  const smooth: Required<NoteTransition> = { offsetMs: 0, durLeftMs: 100, durRightMs: 70, depthLeftCents: 15, depthRightCents: 15 };
  const optsFlat = { tempo: 120, defaultTransition: flat };
  const optsSmooth = { tempo: 120, defaultTransition: smooth };

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

  it("evalF0CentsFrames yields the per-frame cents array + voiced mask", () => {
    const notes = [mk("a", 0, 100, 60)];
    const { cents, voiced } = evalF0CentsFrames(notes, undefined, { frameStartTick: 0, ticksPerFrame: 50, frameCount: 4 }, optsFlat);
    expect(cents.length).toBe(4);
    expect(Array.from(voiced)).toEqual([1, 1, 0, 0]); // ticks 0,50 ∈ [0,100); 100,150 rest
    expect(cents[0]).toBe(6000);
  });
});
