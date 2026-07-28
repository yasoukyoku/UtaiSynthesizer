// S87 — the vocal editor's GRID-SNAP contract. Two defects motivated pulling it out of the component:
//   (a) snapping was computed in PART-RELATIVE tick space while the grid LINES are drawn in ABSOLUTE space,
//       so for any part whose start is off-grid (every imported score / MIDI-extracted part — their part
//       starts AT the first note) a "snapped" note did not sit on a line the user can see;
//   (b) a MOVE snapped the cursor DELTA (`snapRound(now) - snapRound(down)`) = always a whole number of
//       cells, which CONSERVES an off-grid offset — an imported off-grid note could never be dragged back
//       onto the grid (§user 死结).
// These functions are pure precisely so that contract has a regression net without mounting the editor
// (before S87 NOTHING in the suite asserted note placement / snapping at all).
import { describe, it, expect } from "vitest";
import { snapFloor, snapRound, snapPlaceTick, snapEdgeTick, snapMoveDelta, resizeEndTick } from "./vocalGeometry";
import { TICKS_PER_BEAT } from "./constants";

const CELL = TICKS_PER_BEAT / 12; // 40t — the finest grid, and the fixed MOVE/RESIZE quantum
const GRID_8 = TICKS_PER_BEAT / 2; // 240t — a "1/8" creation grid (the editor's default)
const OFF = 137; // a deliberately off-grid part start / note tick (137 % 40 !== 0)

describe("snapFloor / snapRound — untouched primitives", () => {
  it("floors and rounds to the unit", () => {
    expect(snapFloor(239, GRID_8)).toBe(0);
    expect(snapFloor(241, GRID_8)).toBe(240);
    expect(snapRound(239, GRID_8)).toBe(240);
    expect(snapRound(119, GRID_8)).toBe(0);
  });
  it("unit <= 0 falls back to whole-tick ROUND, not floor — why 'pass 0 to disable' is banned", () => {
    expect(snapFloor(239.6, 0)).toBe(240); // NOT 239 — the fallback silently swaps floor for round
    expect(snapRound(239.6, 0)).toBe(240);
  });
});

describe("snapPlaceTick — creation floors into the cell, in ABSOLUTE space", () => {
  it("on-grid part start: same answer as the old part-relative floor", () => {
    expect(snapPlaceTick(250, 4 * TICKS_PER_BEAT, GRID_8, true)).toBe(240);
    expect(snapPlaceTick(239, 0, GRID_8, true)).toBe(0);
  });
  it("OFF-grid part start: the note lands on a line that is actually DRAWN", () => {
    const rel = snapPlaceTick(250, OFF, GRID_8, true);
    expect(OFF + rel).toBe(240); // absolute position is a real 1/8 line
    expect((OFF + rel) % GRID_8).toBe(0);
    expect(rel).toBe(103); // ...which is NOT a multiple of the unit in part-relative space
  });
  it("snapping OFF: continuous placement, rounded to a whole tick", () => {
    expect(snapPlaceTick(250.4, OFF, GRID_8, false)).toBe(250);
    expect(snapPlaceTick(0.6, 0, GRID_8, false)).toBe(1);
  });
});

describe("snapEdgeTick — adjustment rounds to the nearest line, in ABSOLUTE space", () => {
  it("on-grid part start", () => {
    expect(snapEdgeTick(250, 0, CELL, true)).toBe(240);
    expect(snapEdgeTick(261, 0, CELL, true)).toBe(280);
  });
  it("OFF-grid part start lands on a drawn line", () => {
    const rel = snapEdgeTick(250, OFF, CELL, true);
    expect((OFF + rel) % CELL).toBe(0);
    expect(OFF + rel).toBe(400);
  });
  it("snapping OFF: continuous", () => {
    expect(snapEdgeTick(250.6, OFF, CELL, false)).toBe(251);
  });
});

describe("resizeEndTick — a CLICK must never resize (the regression the review caught)", () => {
  // Note [0,480). The Arrow tool's edge hotzone is 6 screen px, so a bare click lands the cursor INSIDE the
  // note (rel 470 at the default zoom; up to 300 ticks inside when zoomed out). With snapping ON the 1/12
  // rounding used to absorb that; with it OFF a raw-cursor formula would silently truncate the note.
  const N = { tick: 0, duration: 480 };
  it("snap OFF: zero pointer travel is a no-op for ANY grab position (the delta form guarantees it)", () => {
    for (const grab of [479, 470, 300, 180, 0]) {
      expect(resizeEndTick(N.tick, N.duration, grab, grab, 0, CELL, 8, false)).toBe(480);
    }
  });
  it("snap ON: zero travel is a no-op only WITHIN half a cell of the end — hence the component's d.moved guard", () => {
    expect(resizeEndTick(N.tick, N.duration, 470, 470, 0, CELL, CELL, true)).toBe(480); // inside half a cell
    // …but a grab far inside the note (the 6px hotzone is 300 ticks at the min zoom, and the Pen tool makes
    // the WHOLE note a resize target) still snaps to a different line. That is HEAD's behavior too, which is
    // why the fix lives in onPointerUp ("no motion ⇒ never commit"), not only in this formula.
    expect(resizeEndTick(N.tick, N.duration, 180, 180, 0, CELL, CELL, true)).toBe(200);
  });
  it("snap ON: the end lands on a line (HEAD's absolute-cursor rule, unchanged)", () => {
    expect(resizeEndTick(0, 480, 470, 470, 0, CELL, CELL, true)).toBe(480);
    expect(resizeEndTick(0, 480, 505, 470, 0, CELL, CELL, true)).toBe(520);
    expect(resizeEndTick(0, 480, 12, 470, 0, CELL, CELL, true)).toBe(CELL); // floored at one cell
  });
  it("snap OFF: the end follows the HAND, keeping the grab offset", () => {
    expect(resizeEndTick(0, 480, 475, 470, 0, CELL, 8, false)).toBe(485); // +5 ticks of travel
    expect(resizeEndTick(0, 480, 465, 470, 0, CELL, 8, false)).toBe(475); // -5
  });
  it("snap OFF: the floor is the caller's minLen (one render frame), never 1 tick", () => {
    expect(resizeEndTick(100, 480, 0, 570, 0, CELL, 19, false)).toBe(119); // 100 + 19
  });
});

describe("snapMoveDelta — the 死结: an off-grid note CAN be dragged back onto the grid", () => {
  it("off-grid note + a nudge => it lands ON a grid line (the pre-S87 delta form could not)", () => {
    const d = snapMoveDelta(OFF, 5, 0, CELL, true);
    expect((OFF + d) % CELL).toBe(0);
    expect(OFF + d).toBe(160);
    // the pre-S87 form was always a whole number of cells, so it conserved the 137 % 40 = 17 offset
    expect(d % CELL).not.toBe(0);
  });
  it("...also when the PART start is what is off-grid (an imported part, note at rel 0)", () => {
    const d = snapMoveDelta(0, 5, OFF, CELL, true);
    expect((OFF + 0 + d) % CELL).toBe(0);
    expect(OFF + d).toBe(160);
  });
  it("an ON-grid note keeps the old feel: whole-cell steps, and jitter moves nothing", () => {
    expect(snapMoveDelta(160, 3, 0, CELL, true)).toBe(0); // sub-half-cell jitter => no move
    expect(snapMoveDelta(160, 40, 0, CELL, true)).toBe(CELL);
    expect(snapMoveDelta(160, -40, 0, CELL, true)).toBe(-CELL);
    expect(snapMoveDelta(160, 25, 0, CELL, true)).toBe(CELL); // past the half-cell => one cell
  });
  it("snapping OFF: the delta is the raw cursor travel (continuous), off-grid notes stay put", () => {
    expect(snapMoveDelta(OFF, 5.4, 0, CELL, false)).toBe(5);
    expect(snapMoveDelta(OFF, -5.4, 0, CELL, false)).toBe(-5); // round(137 - 5.4) = round(131.6) = 132
    expect(snapMoveDelta(OFF, 0, 0, CELL, false)).toBe(0);
  });
  it("ONE delta for the whole selection: applying it keeps relative spacing exactly", () => {
    const notes = [OFF, OFF + 240, OFF + 500];
    const d = snapMoveDelta(notes[0]!, 5, 0, CELL, true);
    const moved = notes.map((t) => t + d);
    expect(moved[1]! - moved[0]!).toBe(240);
    expect(moved[2]! - moved[1]!).toBe(260);
  });
});
