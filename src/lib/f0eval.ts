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

/** Linear-interpolate a PitchCurve (parallel strictly-increasing xs / ys) at `x`; hold flat outside; 0 if empty.
 *  Exported (single interpolator, no fork) so the ② param lanes (loudness/formant) sample the SAME way pitchDev
 *  does — the lane draw + the render feed both call it. Generic (no cents assumption); 0 = the neutral value
 *  for both current params (loudness dB, formant semitones). */
export function evalCurveAt(c: PitchCurve | undefined, x: number): number {
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

/** Effective transition for a note = track default ⊕ per-note override (per field). Exported so the property
 *  sidebar shows the SAME effective values it evaluates with (one definition, no drift). */
export function effTransition(n: Note, def: Required<NoteTransition>): Required<NoteTransition> {
  const t = n.transition;
  if (!t) return def;
  return {
    offsetMs: t.offsetMs ?? def.offsetMs,
    durLeftMs: t.durLeftMs ?? def.durLeftMs,
    durRightMs: t.durRightMs ?? def.durRightMs,
    depthLeftCents: t.depthLeftCents ?? def.depthLeftCents,
    depthRightCents: t.depthRightCents ?? def.depthRightCents,
    openEdgeCents: t.openEdgeCents ?? def.openEdgeCents,
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
  // SynthV Offset shifts the crossover in time (B owns the seam). CLAMP it so the glide stays within BOTH notes'
  // INNER halves — never past A's or B's midpoint. Beyond simply keeping the glide on the boundary, this stops a
  // departure/arrival from overrunning into the neighbour's §10.5 onset-scoop / release-drift region, which would
  // otherwise leave a discontinuity at the lead/release↔cross handoff (verify F1). For normal (long) notes it is
  // identical to the old [-durL, durR] clamp; it only tightens when a short note's half is the binding limit.
  const loOff = Math.max(-durL, durR - A.duration / 2);
  const hiOff = Math.min(durR, B.duration / 2 - durL);
  const off = Math.max(loOff, Math.min(hiOff, msToTicks(tB.offsetMs, opts.tempo)));
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

// The open-edge scoop/drift is much SNAPPIER than a note-to-note glide (§user; SynthV's AI attack is ~30–80 ms
// vs the ~100 ms portamento). It reuses durLeft/durRight but scaled by this factor AND with a front-loaded ease,
// so a phrase's first/last note SNAPS to pitch instead of a slow portamento (which reads as out-of-tune). The
// note↔note crossCents glide is untouched — it keeps the full duration + smooth sineInOut S-curve.
const OPEN_EDGE_SNAP = 0.2;

/**
 * ② Isolated-ONSET scoop (§10.5): a note with NO connected previous note doesn't start flat at its written pitch
 * — over a SHORT durLeft·OPEN_EDGE_SNAP it scoops UP (fast/near-vertical `sineOut`) from `tone − openEdgeCents`
 * to `tone` with the depthLeft overshoot (SynthV's AI attack, synthesized). Confined to the scoop span (≤ half
 * the note); null outside / when disabled (durLeft=0 or openEdgeCents=0 → flat onset = pre-§10.5 parity).
 */
function leadCents(note: Note, relTick: number, opts: F0EvalOpts): number | null {
  const t = effTransition(note, opts.defaultTransition);
  const durL = Math.min(msToTicks(t.durLeftMs, opts.tempo) * OPEN_EDGE_SNAP, note.duration / 2);
  if (durL <= 0 || t.openEdgeCents <= 0) return null;
  const t1 = note.tick + durL;
  if (relTick < note.tick || relTick > t1) return null;
  const tone = noteCents(note);
  const from = tone - t.openEdgeCents; // scoop reference below the target
  const s = (relTick - note.tick) / durL;
  const base = from + (tone - from) * interpShape(s, "sineOut"); // FAST rise (near-vertical) then settle
  return base + t.depthLeftCents * arriveBump(s); // dir = +1 (scoop up) → overshoot above target, settle at s=1
}

/**
 * ② Isolated-RELEASE drift (§10.5): a note with NO connected next note doesn't end flat — over a SHORT
 * durRight·OPEN_EDGE_SNAP it holds, then falls STEEPLY (`sineIn`) from `tone` DOWN to `tone − openEdgeCents` with
 * the depthRight bump (SynthV's AI release, synthesized; vibrato rides on top). Confined to the release span
 * (≤ half the note); null outside / when disabled.
 */
function releaseCents(note: Note, relTick: number, opts: F0EvalOpts): number | null {
  const t = effTransition(note, opts.defaultTransition);
  const durR = Math.min(msToTicks(t.durRightMs, opts.tempo) * OPEN_EDGE_SNAP, note.duration / 2);
  if (durR <= 0 || t.openEdgeCents <= 0) return null;
  const end = note.tick + note.duration, t0 = end - durR;
  if (relTick < t0 || relTick > end) return null;
  const tone = noteCents(note);
  const to = tone - t.openEdgeCents; // drift reference below the target
  const s = (relTick - t0) / durR; // 0 at the start of the release, 1 at the note end
  const base = tone + (to - tone) * interpShape(s, "sineIn"); // long hold then STEEP fall at the very end
  return base - t.depthRightCents * departBump(s); // dir = −1 (drift down) → overshoot below mid-release
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
  // ② transition OVERRIDES the staircase near a boundary. CONNECTED (abutting) boundary → crossCents glides
  //    to/from the real neighbour; OPEN boundary (no abutting neighbour) → lead/releaseCents synthesize the
  //    SynthV scoop-in / drift-out from the openEdge reference (§10.5). The half-note clamps keep this note's own
  //    onset & release spans from overlapping, and crossCents' midpoint-bounded offset keeps a neighbour's glide
  //    off this note's scoop/drift region, so the hand-offs stay continuous. (Neighbour selection needs sorted input.)
  const prevAbut = idx > 0 && notes[idx - 1]!.tick + notes[idx - 1]!.duration === note.tick;
  const nextAbut = idx < notes.length - 1 && notes[idx + 1]!.tick === note.tick + note.duration;
  let cents: number | null = prevAbut ? crossCents(notes[idx - 1]!, note, relTick, opts) : leadCents(note, relTick, opts);
  if (cents === null) cents = nextAbut ? crossCents(note, notes[idx + 1]!, relTick, opts) : releaseCents(note, relTick, opts);
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

const PAINT_EDGE_PAD = 30; // ticks — fixed reconnection pad: a painted pitchDev eases (linearly) back to the
// original pitch over this pad on each side, so it never bleeds a constant offset out to the segment start.

/**
 * Merge a pitch-paint drag into a segment's pitchDev curve (SynthV Pencil = interval-replace). Each drawn
 * (relTick, absCents) point becomes a DELTA vs the AUTOMATIC line (evalF0Cents with pitchDev=undefined —
 * base ⊕ transition + vibrato); the drawn x-span REPLACES that interval, and a zero anchor a fixed pad outside
 * each end makes the deviation ease (linearly, via evalCurveAt) back to the original pitch — so a painted
 * deviation can never bleed a constant offset out to the segment start (the S51 整条平移 bug). Base points
 * outside the padded span are kept; the store's normalizeCurve does the final round/quantize/dedup. Pure.
 */
export function paintedDev(
  notes: readonly Note[],
  paint: { xs: number[]; ys: number[] },
  base: PitchCurve | undefined,
  opts: F0EvalOpts,
): PitchCurve {
  // Each drawn (relTick, absCents) → the DELTA vs the automatic line (pitchDev=undefined); last sample per x.
  const map = new Map<number, number>();
  for (let i = 0; i < paint.xs.length; i++) {
    const x = Math.round(paint.xs[i]!);
    map.set(x, paint.ys[i]! - evalF0CentsAt(notes, undefined, x, opts).cents);
  }
  return mergePaintedInterval(map, base);
}

/**
 * ② Merge a param-lane paint drag (loudness / formant) into a segment's param curve. Unlike paintedDev, each
 * drawn (relTick, value) point is the ABSOLUTE param value (no delta-vs-baseline subtraction). Interval-replace
 * + the same zero-anchor ease-back to neutral 0 outside the painted span (so a locally-drawn hump can't bleed a
 * constant offset out to the segment start). Pure — the store's normalizeCurve does the final round/dedup.
 */
export function paintedParamCurve(paint: { xs: number[]; ys: number[] }, base: PitchCurve | undefined): PitchCurve {
  const map = new Map<number, number>();
  for (let i = 0; i < paint.xs.length; i++) map.set(Math.round(paint.xs[i]!), paint.ys[i]!);
  return mergePaintedInterval(map, base);
}

/**
 * ② Slice a segment-relative curve (pitchDev / a param lane, X = ticks) at `boundary` into {left, right} for
 * splitSegment. `left` keeps points with x < boundary + a lossless boundary SAMPLE (so it reproduces the
 * source over [0, boundary]); `right` keeps points with x > boundary REBASED by −boundary + a lossless x=0
 * SAMPLE. A point exactly on the seam goes to both. Our painted curves zero-anchor their ends, so a half with
 * no source points is correctly `undefined` (= flat neutral). Reuses evalCurveAt for the seam value. Pure.
 */
export function sliceCurveAtTick(curve: PitchCurve | undefined, boundary: number): { left?: PitchCurve; right?: PitchCurve } {
  if (!curve || curve.xs.length === 0) return {};
  const bVal = evalCurveAt(curve, boundary);
  const lxs: number[] = [], lys: number[] = [], rxs: number[] = [], rys: number[] = [];
  for (let i = 0; i < curve.xs.length; i++) {
    const x = curve.xs[i]!, y = curve.ys[i]!;
    if (x < boundary) { lxs.push(x); lys.push(y); }
    else if (x > boundary) { rxs.push(x - boundary); rys.push(y); }
    else { lxs.push(x); lys.push(y); rxs.push(0); rys.push(y); } // exactly on the seam → both halves
  }
  // ALWAYS give each half a boundary seam sample so a half with NO explicit points still holds the source's
  // value there — a flat non-zero region (e.g. a single +3dB point held across the back half) SURVIVES the
  // split on BOTH halves (§user: the new segment's lane keeps +3dB), instead of dropping to neutral 0.
  if (!lxs.length || lxs[lxs.length - 1] !== boundary) { lxs.push(boundary); lys.push(bVal); }
  if (!rxs.length || rxs[0] !== 0) { rxs.unshift(0); rys.unshift(bVal); }
  return { left: { xs: lxs, ys: lys }, right: { xs: rxs, ys: rys } };
}

/** Interval-replace merge shared by paintedDev/paintedParamCurve: the painted x-span (padded PAINT_EDGE_PAD,
 *  zero-anchored just outside → linear ease back to 0, no 整条平移 bleed) replaces `base` there; base points
 *  outside are kept. `paint` maps rounded relTick → the curve VALUE at that tick. Pure. */
function mergePaintedInterval(paint: Map<number, number>, base: PitchCurve | undefined): PitchCurve {
  const dxs = [...paint.keys()].sort((a, b) => a - b);
  if (dxs.length === 0) return base ? { xs: [...base.xs], ys: [...base.ys] } : { xs: [], ys: [] };
  const xMin = dxs[0]!, xMax = dxs[dxs.length - 1]!;
  const loA = Math.max(0, xMin - PAINT_EDGE_PAD), hiA = xMax + PAINT_EDGE_PAD;
  const merged = new Map<number, number>();
  if (base) for (let i = 0; i < base.xs.length; i++) {
    const x = base.xs[i]!;
    if (x < loA || x > hiA) merged.set(x, base.ys[i]!);
  }
  merged.set(loA, 0); // zero anchors just outside → linear ease back to neutral (no 整条平移 bleed)
  merged.set(hiA, 0);
  for (const x of dxs) merged.set(x, paint.get(x)!); // painted points (interval-replace); an anchor at xMin is overridden
  const xs = [...merged.keys()].sort((a, b) => a - b);
  return { xs, ys: xs.map((x) => merged.get(x)!) };
}
