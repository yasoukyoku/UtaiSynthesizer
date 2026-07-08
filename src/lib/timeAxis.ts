import { TICKS_PER_BEAT } from "./constants";

/**
 * TimeAxis — the SINGLE authority for bar/beat/tick geometry (S48 Phase 0).
 *
 * Before this, `ticksPerBar = TICKS_PER_BEAT * timeSignature[0]` was duplicated across ≥4 files
 * (Arrangement, TimelineRuler, Toolbar, canvasDraw) and the time-signature DENOMINATOR was ignored
 * entirely — so 3/4 and 6/8 rendered identically (both `480*num`), which is wrong (6/8 = 6 eighth-beats
 * = 1440 ticks with the beat every 240, not 4 quarter-beats). This centralizes all of it and makes the
 * denominator live: `ticksPerBeat = TICKS_PER_BEAT * 4 / den` (den=4 → 480 quarter-beat; den=8 → 240).
 *
 * INVARIANT (S48, user 2026-07-09): tempo & meter are TIMELINE-level (project-global), shared by every
 * track — never per-track. This axis is the one place bar geometry is computed; all tracks read it.
 *
 * MAP-READY BY DESIGN: the public API is POSITION-based (`*At(tick)`), built from a list of
 * `TimeSigChange`. Phase 0 constructs it from the single global meter (`TimeAxis.global(num, den)` = one
 * change at bar 0), so every method returns the constant-meter answer today. When per-section meter lands
 * (a real `timeSignatures[]` with changes at later bars), only THIS file's construction changes —
 * consumers that call `ticksPerBarAt`/`gridLinesInRange`/`tickToBarBeat` are already position-correct.
 *
 * tick↔ms is deliberately NOT here — that is tempo-only (`laneOps.ticksToMs`) and meter never enters it.
 *
 * ASSUMES tick ≥ 0 (the app clamps every scroll/playhead/note tick to ≥0). Its bar/beat answers match the
 * pre-Phase-0 formulas bit-for-bit ONLY in that non-negative regime — `gridLinesInRange` floors its start
 * to 0 and `tickToBarBeat` uses `Math.floor`, so a NEGATIVE tick would diverge from the old sign-preserving
 * `%`. If a future caller ever feeds signed ticks, handle the sign here rather than assuming parity.
 */

/** A meter change anchored at a bar index (bar 0 = arrangement start). `den` = beat unit (4=quarter,
 *  8=eighth, …). Phase 0 uses exactly one, at bar 0; the type is the seam for per-section meter later. */
export interface TimeSigChange {
  bar: number;
  num: number;
  den: number;
}

interface Seg {
  bar: number; // first bar of this meter section
  startTick: number; // tick at that bar
  num: number;
  den: number;
  ticksPerBeat: number;
  ticksPerBar: number;
}

/** Quarter-of-a-beat is the finest sub reported by `tickToBarBeat` (matches the pre-Phase-0 Toolbar
 *  position readout, which showed sixteenths for a quarter beat = TICKS_PER_BEAT/4). */
const SUBS_PER_BEAT = 4;

export class TimeAxis {
  /** Precomputed meter sections, sorted by tick; ALWAYS non-empty (constructor guarantees ≥1 at bar 0). */
  private readonly segs: readonly Seg[];
  private readonly first: Seg;

  constructor(changes: TimeSigChange[]) {
    const src = changes.length === 0 ? [{ bar: 0, num: 4, den: 4 }] : changes;
    const sorted = [...src].sort((a, b) => a.bar - b.bar);
    const head = sorted[0]!; // src is non-empty ⇒ sorted[0] defined
    if (head.bar !== 0) sorted.unshift({ bar: 0, num: head.num, den: head.den });

    const segs: Seg[] = [];
    for (let i = 0; i < sorted.length; i++) {
      const c = sorted[i]!;
      const ticksPerBeat = Math.round((TICKS_PER_BEAT * 4) / c.den);
      const ticksPerBar = ticksPerBeat * c.num;
      const prev = segs[i - 1]; // undefined when i === 0
      const startTick = prev ? prev.startTick + (c.bar - prev.bar) * prev.ticksPerBar : 0;
      segs.push({ bar: c.bar, startTick, num: c.num, den: c.den, ticksPerBeat, ticksPerBar });
    }
    this.segs = segs;
    this.first = segs[0]!; // ≥1 by construction
  }

  /** Convenience for the single global meter (Phase 0). */
  static global(num: number, den: number): TimeAxis {
    return new TimeAxis([{ bar: 0, num, den }]);
  }

  /** The meter section covering `tick` (last section is open-ended). Never undefined. */
  private segAt(tick: number): Seg {
    let s = this.first;
    for (let i = 1; i < this.segs.length; i++) {
      const seg = this.segs[i]!;
      if (seg.startTick <= tick) s = seg;
      else break;
    }
    return s;
  }

  ticksPerBeatAt(tick: number): number {
    return this.segAt(tick).ticksPerBeat;
  }

  ticksPerBarAt(tick: number): number {
    return this.segAt(tick).ticksPerBar;
  }

  /** Bar/beat/sub at a tick. `bar` and `beat` are 1-based (display); `sub` is a 0-based quarter-of-beat.
   *  Mirrors the old Toolbar formula exactly for a 4/4 project. */
  tickToBarBeat(tick: number): { bar: number; beat: number; sub: number } {
    const s = this.segAt(tick);
    const rel = tick - s.startTick;
    const barsInto = Math.floor(rel / s.ticksPerBar);
    const tickInBar = rel - barsInto * s.ticksPerBar;
    const beat = Math.floor(tickInBar / s.ticksPerBeat);
    const tickInBeat = tickInBar - beat * s.ticksPerBeat;
    const sub = Math.floor((tickInBeat / s.ticksPerBeat) * SUBS_PER_BEAT);
    return { bar: s.bar + barsInto + 1, beat: beat + 1, sub };
  }

  /** First tick of a given 0-based bar index. */
  tickAtBar(bar: number): number {
    let s = this.first;
    for (let i = 1; i < this.segs.length; i++) {
      const seg = this.segs[i]!;
      if (seg.bar <= bar) s = seg;
      else break;
    }
    return s.startTick + (bar - s.bar) * s.ticksPerBar;
  }

  /** Total tick span of `nBars` bars from the start — used by `computeTotalTicks`'s scroll-headroom
   *  floor. With a single meter this is `nBars * ticksPerBar`; map-correct across sections too. */
  ticksForBars(nBars: number): number {
    return this.tickAtBar(nBars);
  }

  /**
   * The vertical grid lines (bar downbeats + in-between beats) covering `[startTick, endTick)`, each
   * tagged `isBar`. THE source for the arrangement grid + timeline ruler (replaces the per-file
   * `tick % ticksPerBar` loops, which couldn't vary the beat spacing per meter). The first line is
   * FLOORED to the beat at or below `startTick` so the left-edge partial line is covered — this makes
   * a 4/4 project yield EXACTLY the pre-Phase-0 loop's line set (`startTick - startTick%480 … <endTick`,
   * beats every 480, bars every 1920), so the drawn grid is bit-for-bit identical. Callers clip lines
   * left of x=0 themselves (drawing at x≤0 renders nothing).
   */
  gridLinesInRange(startTick: number, endTick: number): { tick: number; isBar: boolean }[] {
    const lines: { tick: number; isBar: boolean }[] = [];
    if (endTick <= startTick) return lines;
    for (let si = 0; si < this.segs.length; si++) {
      const s = this.segs[si]!;
      const next = this.segs[si + 1]; // Seg | undefined (last section is open-ended)
      const segEnd = next ? next.startTick : endTick;
      if (segEnd <= startTick) continue;
      if (s.startTick >= endTick) break;
      // Start at the beat line at or BELOW max(startTick, s.startTick) (floor to this section's beat
      // grid) — matches the old loop's `startTick - startTick%TICKS_PER_BEAT` left-edge coverage. For a
      // later section (startTick < s.startTick) `off` is 0, so it begins right at the section downbeat.
      const from = Math.max(startTick, s.startTick);
      const off = (from - s.startTick) % s.ticksPerBeat;
      const stop = Math.min(endTick, segEnd);
      for (let tick = from - off; tick < stop; tick += s.ticksPerBeat) {
        const isBar = (tick - s.startTick) % s.ticksPerBar === 0;
        lines.push({ tick, isBar });
      }
    }
    return lines;
  }
}
