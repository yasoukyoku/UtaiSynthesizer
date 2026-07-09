// ② Vocal pitch — evalF0Cents, THE single f0 evaluator (S51 SynthV-aligned rewrite, §10.3, Option A).
// One source of truth for the sounding pitch line, feeding: the editor OVERLAY draw + the light PREVIEW
// oscillator (→ centsToHz) + (Phase 6) the Rust render array. Option A = TS computes this array and Rust
// consumes it (+transpose → Hz → resample), so "what you see == hear == render" holds by construction.
// Returns WRITTEN-pitch cents (transpose is applied ONLY in the Rust render, §9.3).
//
// LAYERS (SynthV two-level add, §10.3):
//   ① base staircase — the covering note's tone*100 + detune.
//   ② note-to-note TRANSITION — a smooth glide across a boundary (SynthV durLeft/durRight + depth overshoot)
//      that OVERRIDES the staircase inside the transition span (interval-replace, NOT additive — avoids
//      double-counting the step). Times are ABSOLUTE ms so a glide sounds the same at any tempo.
//   ③ Pitch Deviation — hand-drawn segment-relative cents curve — ADDITIVE.
//   ④ Vibrato — mid/tail LFO (SynthV freqHz/depthCents/start/ease/phase) — ADDITIVE.
// Pure. Notes MUST be tick-sorted (normalizeNotesArray); the overlay sorts its live-preview array too.
import type { Note, PitchCurve, NoteTransition } from "../types/project";
import { interpShape } from "./interpolateShape";
import { ticksToMs, msToTicks } from "./audio/laneOps";

export interface F0EvalOpts {
  /** Tempo (BPM) — transition/vibrato times are ABSOLUTE ms, so ms↔tick needs it. */
  tempo: number;
  /** Track-level DEFAULT transition (VocalTrackParams.transition); a note's partial transition overrides it. */
  defaultTransition: Required<NoteTransition>;
}

/** Linear-interpolate a PitchCurve (parallel strictly-increasing xs / ys) at `x`; hold flat outside; 0 if empty. */
function evalCurveAt(c: PitchCurve | undefined, x: number): number {
  if (!c || c.xs.length === 0) return 0;
  const { xs, ys } = c;
  const n = xs.length;
  if (x <= xs[0]!) return ys[0]!;
  if (x >= xs[n - 1]!) return ys[n - 1]!;
  let i = 0;
  while (i < n - 1 && xs[i + 1]! <= x) i++;
  const span = xs[i + 1]! - xs[i]!;
  const t = span > 0 ? (x - xs[i]!) / span : 0;
  return ys[i]! + (ys[i + 1]! - ys[i]!) * t;
}

/** Index of the note covering segment-relative `relTick` (half-open [tick, tick+duration)); -1 = rest.
 *  Full linear scan returning the LAST match (no early break) so an unsorted live-preview array can't drop a
 *  note into a false rest-gap (审查 #2); on the sorted array the last match = the higher-onset note, so an
 *  abutting note's onset correctly wins over the prior note's end. */
function findNoteAt(notes: readonly Note[], relTick: number): number {
  let found = -1;
  for (let i = 0; i < notes.length; i++) {
    const n = notes[i]!;
    if (relTick >= n.tick && relTick < n.tick + n.duration) found = i;
  }
  return found;
}

/** Note's WRITTEN base pitch in cents (tone + fine detune). */
const noteCents = (n: Note): number => n.pitch * 100 + (n.detune ?? 0);

/** Effective transition for a note = track default ⊕ per-note override (per field). */
function effTransition(n: Note, def: Required<NoteTransition>): Required<NoteTransition> {
  const t = n.transition;
  if (!t) return def;
  return {
    offsetMs: t.offsetMs ?? def.offsetMs,
    durLeftMs: t.durLeftMs ?? def.durLeftMs,
    durRightMs: t.durRightMs ?? def.durRightMs,
    depthLeftCents: t.depthLeftCents ?? def.depthLeftCents,
    depthRightCents: t.depthRightCents ?? def.depthRightCents,
  };
}

// Overshoot bumps: 0 at both ends (s=0,1) so the transition still lands exactly on each note's tone; the
// peak is biased toward the arriving end (→1) / departing end (→0). This is OUR clean design (手感 aligned
// with SynthV's "slight overshoot on landing", not a bit-for-bit port).
const arriveBump = (s: number): number => Math.sin(Math.PI * s * s); // peak ≈ s=0.707 (toward B)
const departBump = (s: number): number => Math.sin(Math.PI * (1 - s) * (1 - s)); // peak ≈ s=0.293 (toward A)

/**
 * The single cross-boundary transition curve A→B evaluated at `relTick`, or null if `relTick` is outside the
 * transition span or the two notes are too far apart to connect (gap > LEGATO). The span is
 * [A.end − durRight(A), B.start + durLeft(B)] (each side clamped to half its own note so short notes don't
 * overrun), centred on the boundary: a sineInOut glide A.tone→B.tone + signed depth overshoot at each end.
 */
function crossCents(A: Note, B: Note, relTick: number, opts: F0EvalOpts): number | null {
  // Only ABUTTING notes glide (B starts exactly where A ends). A short-GAP legato glide (承前元音) needs a
  // carrier in the gap, which is Phase-6 render territory (§3.4); drawing a glide over an uncovered gap would
  // leave a broken "half-slide + silent hole", so until then each note holds flat to its own edge.
  if (B.tick !== A.tick + A.duration) return null;
  const tA = effTransition(A, opts.defaultTransition);
  const tB = effTransition(B, opts.defaultTransition);
  const durR = Math.min(msToTicks(tA.durRightMs, opts.tempo), A.duration / 2); // A leaves early, ≤ half of A
  const durL = Math.min(msToTicks(tB.durLeftMs, opts.tempo), B.duration / 2); // B arrives late, ≤ half of B
  // SynthV Offset shifts the crossover in time (B owns the seam); CLAMP it so the glide always straddles the
  // boundary — a larger offset would detach the glide from the note edge and jump discontinuously.
  const off = Math.max(-durL, Math.min(durR, msToTicks(tB.offsetMs, opts.tempo)));
  const t0 = A.tick + A.duration - durR + off;
  const t1 = B.tick + durL + off;
  const span = t1 - t0;
  if (span <= 0 || relTick < t0 || relTick > t1) return null; // durLeft=durRight=0 → pure staircase (parity)
  const cA = noteCents(A), cB = noteCents(B);
  const s = (relTick - t0) / span;
  const base = cA + (cB - cA) * interpShape(s, "sineInOut");
  // Overshoot follows the GLIDE DIRECTION: an up-glide overshoots above the target, a down-glide below; a
  // SAME-pitch seam (dir 0) gets NO bump, so a sustained/tie seam stays flat (审查 #2/#4).
  const dir = Math.sign(cB - cA);
  return base + dir * (tA.depthRightCents * departBump(s) + tB.depthLeftCents * arriveBump(s));
}

/** ④ Vibrato (SynthV): starts after `startMs` (short notes stay flat), oscillates at `freqHz` with `phase`
 *  offset and `depthCents` amplitude, linearly faded in/out over `easeIn/easeOut` ms of the active span. */
function evalVibrato(note: Note, noteRel: number, tempo: number): number {
  const v = note.vibrato!;
  if (v.depthCents <= 0 || v.freqHz <= 0) return 0;
  const startTicks = msToTicks(v.startMs, tempo);
  const activeTicks = note.duration - startTicks; // vibrato-active span [start, note end]
  const inSpan = noteRel - startTicks;
  if (inSpan < 0 || activeTicks <= 0) return 0; // before onset delay / no room
  const elapsedMs = ticksToMs(inSpan, tempo);
  const amp = v.depthCents * Math.sin(2 * Math.PI * ((elapsedMs / 1000) * v.freqHz + v.phase));
  const easeIn = msToTicks(v.easeInMs, tempo);
  const easeOut = msToTicks(v.easeOutMs, tempo);
  let env = 1;
  if (easeIn > 0 && inSpan < easeIn) env = inSpan / easeIn; // fade-in
  const remain = activeTicks - inSpan; // ticks to the note end
  if (easeOut > 0 && remain < easeOut) env = Math.min(env, remain / easeOut); // fade-out
  return amp * Math.max(0, env);
}

/**
 * ★ evalF0Cents at one segment-relative tick → { WRITTEN-pitch cents, voiced }. Layered per §10.3
 * (① base staircase / ② transition interval-replace / ③ pitchDev additive / ④ vibrato additive).
 * A rest (no covering note) → voiced:false (the overlay breaks the line; Rust render mirrors via the mask).
 */
export function evalF0CentsAt(
  notes: readonly Note[],
  pitchDev: PitchCurve | undefined,
  relTick: number,
  opts: F0EvalOpts,
): { cents: number; voiced: boolean } {
  const dev = evalCurveAt(pitchDev, relTick); // ③ (segment-relative; also over rests, but they're unvoiced)
  const idx = findNoteAt(notes, relTick);
  if (idx < 0) return { cents: dev, voiced: false };
  const note = notes[idx]!;
  // ② transition OVERRIDES the staircase in its span: try the incoming (prev→note) then outgoing (note→next)
  //    boundary; the half-note clamps keep the two spans from overlapping inside the note, so at most one
  //    applies. (Neighbor selection notes[idx±1] relies on tick-sorted input, as findNoteAt does.)
  let cents: number | null = null;
  if (idx > 0) cents = crossCents(notes[idx - 1]!, note, relTick, opts);
  if (cents === null && idx < notes.length - 1) cents = crossCents(note, notes[idx + 1]!, relTick, opts);
  if (cents === null) cents = noteCents(note); // ① stable staircase
  cents += dev; // ③
  if (note.vibrato) cents += evalVibrato(note, relTick - note.tick, opts.tempo); // ④
  return { cents, voiced: true };
}

export interface F0Frames {
  cents: Float32Array;
  voiced: Uint8Array;
}

/**
 * Batch: sample evalF0Cents at `frameCount` frames, `ticksPerFrame` apart from `frameStartTick` (segment-
 * relative). The canonical per-frame f0 array — the overlay samples it to draw the line, and (Option A)
 * Phase 6 hands it to the Rust render (which adds transpose, converts →Hz with the voiced mask, resamples).
 */
export function evalF0CentsFrames(
  notes: readonly Note[],
  pitchDev: PitchCurve | undefined,
  frame: { frameStartTick: number; ticksPerFrame: number; frameCount: number },
  opts: F0EvalOpts,
): F0Frames {
  const cents = new Float32Array(frame.frameCount);
  const voiced = new Uint8Array(frame.frameCount);
  for (let f = 0; f < frame.frameCount; f++) {
    const r = evalF0CentsAt(notes, pitchDev, frame.frameStartTick + f * frame.ticksPerFrame, opts);
    cents[f] = r.cents;
    voiced[f] = r.voiced ? 1 : 0;
  }
  return { cents, voiced };
}
