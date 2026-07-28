// ② Vocal piano-roll GEOMETRY — the SINGLE coordinate authority for the vocal editor (§9.4). Draw AND
// hit-test both go through here, so tick↔x / pitch↔y can never be defined twice and drift. Horizontal is
// ABSOLUTE tick space (identical to the arrangement, so drawBeatGrid / the TimeAxis grid reuse unchanged);
// part-local note ticks convert to absolute in exactly two adapters. Vertical has TWO deliberately separate
// mappings: pitchToY/yToPitch FLOOR to a whole-semitone row (note placement), and the CONTINUOUS
// centsToY/yToCents (NO floor) for sub-semitone pitch editing — so micro-tuning a pitch line is possible
// (the floored version made it impossible, §9.4 blocker). Higher MIDI = higher on screen.
//
// The editor keeps its OWN view state (not the arrangement's global scroll/zoom, §9.4); every function
// takes it explicitly so this module stays pure + testable.

export const V_PITCH_MIN = 0;
export const V_PITCH_MAX = 127;
export const V_ROW_H_MIN = 6;
export const V_ROW_H_MAX = 40;
export const V_ROW_H_DEFAULT = 16;

/** Vocal-editor view: horizontal scroll (px) + pixels-per-tick, vertical scroll (px) + per-semitone row px.
 *  `top` = the y where the note rows begin (below a bar-number ruler); pitch↔y all offset by it. */
export interface VocalView {
  scrollX: number;
  scrollY: number;
  ppt: number; // pixels per tick (PIXELS_PER_TICK * horizontal zoom)
  rowH: number; // pixels per semitone row
  top?: number; // px reserved at the top (bar ruler); default 0
}

// ── horizontal (absolute tick space — same formula the arrangement uses) ──
export const tickToX = (tick: number, v: VocalView): number => tick * v.ppt - v.scrollX;
export const xToTick = (x: number, v: VocalView): number => (x + v.scrollX) / v.ppt;

// ── part-local ↔ absolute (the ONLY 2 adapters; note ticks are segment-relative, the grid is absolute) ──
export const noteTickToX = (relTick: number, partStartTick: number, v: VocalView): number =>
  tickToX(relTick + partStartTick, v);
export const xToNoteTick = (x: number, partStartTick: number, v: VocalView): number =>
  xToTick(x, v) - partStartTick;

// ── vertical, WHOLE-SEMITONE (note-row placement). All offset by `top` (the bar-ruler height). ──
/** Screen Y of the TOP edge of pitch `p`'s row (note blocks are drawn from here, `rowH` tall). */
export const pitchToY = (pitch: number, v: VocalView): number => (v.top ?? 0) + (V_PITCH_MAX - pitch) * v.rowH - v.scrollY;
/** The whole-semitone pitch whose row contains screen `y` (clamped to MIDI range). */
export const yToPitch = (y: number, v: VocalView): number =>
  Math.min(V_PITCH_MAX, Math.max(V_PITCH_MIN, V_PITCH_MAX - Math.floor((y - (v.top ?? 0) + v.scrollY) / v.rowH)));

// ── vertical, CONTINUOUS cents (sub-semitone pitch editing / f0 line) — NO floor ──
/** Screen Y for an absolute-cents pitch value (pitch*100 + detune). Row-CENTER aligned: a 0-detune note's
 *  f0 line runs through the MIDDLE of its block. */
export const centsToY = (absCents: number, v: VocalView): number =>
  (v.top ?? 0) + (V_PITCH_MAX + 0.5 - absCents / 100) * v.rowH - v.scrollY;
/** Inverse of centsToY — the continuous absolute-cents value at screen `y` (for dragging a pitch point). */
export const yToCents = (y: number, v: VocalView): number =>
  (V_PITCH_MAX + 0.5 - (y - (v.top ?? 0) + v.scrollY) / v.rowH) * 100;

/** Total content height of all 128 rows (for vertical scroll clamps). */
export const rowsContentHeight = (rowH: number): number => (V_PITCH_MAX - V_PITCH_MIN + 1) * rowH;

// ── ② bottom automation lane (loudness / formant) value↔y — the band is a FIXED strip [laneTop, laneTop+laneH]
//    at the canvas bottom (NOT scrolled with the note rows). Higher value = higher on screen; neutral (0) sits
//    at whatever y its [min,max] maps to. Pure (laneTop/laneH passed explicitly, like the rest of this module). ──
/** Loudness-lane dB range (±): ONE constant for the vocal editor's lane (LANE_PARAMS) AND the S59
 *  audio-track loudness band, so the two surfaces can never drift apart. */
export const LOUDNESS_DB_RANGE = 12;
/** Param value in [min,max] → screen Y within the lane band. Clamped so the line never leaves the band. */
export const paramToY = (value: number, min: number, max: number, laneTop: number, laneH: number): number => {
  const t = max > min ? (value - min) / (max - min) : 0.5;
  return laneTop + (1 - Math.min(1, Math.max(0, t))) * laneH;
};
/** Screen Y within the lane band → the param value in [min,max] (clamped) — inverse of paramToY (for painting). */
export const yToParam = (y: number, min: number, max: number, laneTop: number, laneH: number): number => {
  const t = laneH > 0 ? 1 - (y - laneTop) / laneH : 0.5;
  return min + Math.min(1, Math.max(0, t)) * (max - min);
};

// ── snapping (grid IS the snap unit, §9.1). floor for placement, round for move. ──
export const snapFloor = (relTick: number, snapTicks: number): number =>
  snapTicks > 0 ? Math.floor(relTick / snapTicks) * snapTicks : Math.round(relTick);
export const snapRound = (relTick: number, snapTicks: number): number =>
  snapTicks > 0 ? Math.round(relTick / snapTicks) * snapTicks : Math.round(relTick);

/**
 * S87 — THE two grid-snap entry points for the vocal editor. Note ticks are PART-RELATIVE while the grid
 * lines are drawn in ABSOLUTE tick space (TimeAxis), so a part whose `partStart` is not a multiple of the
 * unit (every imported score / MIDI-extracted part — their part starts at the first note) would otherwise
 * snap notes to lines that are NOT the ones on screen. Both convert to absolute, snap, and convert back.
 *
 * `on = false` (user turned snapping off) = CONTINUOUS placement: the raw tick, rounded to a whole tick
 * (note ticks are integers by normalizeNote anyway). This is an explicit branch on purpose — `snapFloor(x, 0)`
 * falls back to Math.ROUND, so "pass 0 to disable" would silently swap floor semantics for round.
 *
 * Pure, so the snap contract is unit-testable without mounting the editor.
 */
/** Placement (drawing a NEW note): FLOOR to the cell the cursor is inside. */
export const snapPlaceTick = (relTick: number, partStart: number, unit: number, on: boolean): number =>
  on ? snapFloor(partStart + relTick, unit) - partStart : Math.round(relTick);
/** Adjustment (moving / resizing an EXISTING note, or dragging a new note's end): ROUND to the nearest line. */
export const snapEdgeTick = (relTick: number, partStart: number, unit: number, on: boolean): number =>
  on ? snapRound(partStart + relTick, unit) - partStart : Math.round(relTick);

/**
 * S87 — the tick delta a MOVE gesture applies to the WHOLE selection. `anchorRel` is the grabbed note's
 * original tick and `cursorDelta` the continuous tick distance the cursor has travelled; the delta is chosen
 * so the ANCHOR lands on a grid line, and every other selected note rides the SAME delta (spacing preserved,
 * which snapping each note independently would destroy).
 *
 * ⚠ This replaces the pre-S87 form `snapRound(cursorNow) - snapRound(cursorDown)`, which snapped the CURSOR
 * at both ends and subtracted: always a whole number of cells, so it CONSERVED any off-grid offset — an
 * imported off-grid note could never be dragged back onto the grid (§user 死结).
 */
export const snapMoveDelta = (
  anchorRel: number,
  cursorDelta: number,
  partStart: number,
  unit: number,
  on: boolean,
): number => snapEdgeTick(anchorRel + cursorDelta, partStart, unit, on) - anchorRel;

/**
 * S87 — the new END tick of a note being resized by its right edge.
 *  · snapping ON  = the end lands ON a grid line (absolute space), floored at `minLen`. Byte-identical to
 *    the pre-S87 formula for an on-grid part.
 *  · snapping OFF = the end follows the HAND: the ORIGINAL end plus the cursor's travel since the grab.
 *    ⚠ NOT the raw cursor position. Absolute-cursor semantics without a grid means a mere CLICK inside the
 *    6px edge hotzone (zero motion) resolves to a tick strictly inside the note and silently truncates it —
 *    with the grid on, the 1/12 rounding used to absorb the whole hotzone and hide that. Keeping the grab
 *    offset makes a zero-travel gesture a mathematical no-op, in every mode and at every zoom.
 */
export const resizeEndTick = (
  origTick: number,
  origDuration: number,
  cursorRel: number,
  grabRel: number,
  partStart: number,
  unit: number,
  minLen: number,
  on: boolean,
): number =>
  Math.max(
    origTick + minLen,
    on ? snapEdgeTick(cursorRel, partStart, unit, true) : origTick + origDuration + Math.round(cursorRel - grabRel),
  );

// ── piano-key helpers (key column + row striping + note names) ──
const BLACK = new Set([1, 3, 6, 8, 10]);
const NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
/** A black (sharp) key — its row is drawn slightly darker so the rows don't read as one blur (§9.4). */
export const isBlackKey = (pitch: number): boolean => BLACK.has(((pitch % 12) + 12) % 12);
/** Scientific pitch name, C4 = 60 (e.g. 60→"C4", 0→"C-1", 127→"G9"). */
export const pitchName = (pitch: number): string => `${NAMES[((pitch % 12) + 12) % 12]}${Math.floor(pitch / 12) - 1}`;
/** MIDI pitch → Hz (A4=440), for the light WebAudio preview oscillator (§9.7). */
export const pitchToHz = (pitch: number): number => 440 * Math.pow(2, (pitch - 69) / 12);
/** Absolute-cents → Hz (continuous, for the f0-line preview). */
export const centsToHz = (absCents: number): number => 440 * Math.pow(2, (absCents - 6900) / 1200);
