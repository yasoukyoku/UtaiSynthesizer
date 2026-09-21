import { TICKS_PER_BEAT } from "./constants";
/** Quarter-of-a-beat is the finest sub reported by `tickToBarBeat` (matches the pre-Phase-0 Toolbar
 *  position readout, which showed sixteenths for a quarter beat = TICKS_PER_BEAT/4). */
const SUBS_PER_BEAT = 4;
export class TimeAxis {
    constructor(changes) {
        /** Precomputed meter sections, sorted by tick; ALWAYS non-empty (constructor guarantees ≥1 at bar 0). */
        Object.defineProperty(this, "segs", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "first", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        const src = changes.length === 0 ? [{ bar: 0, num: 4, den: 4 }] : changes;
        const sorted = [...src].sort((a, b) => a.bar - b.bar);
        const head = sorted[0]; // src is non-empty ⇒ sorted[0] defined
        if (head.bar !== 0)
            sorted.unshift({ bar: 0, num: head.num, den: head.den });
        const segs = [];
        for (let i = 0; i < sorted.length; i++) {
            const c = sorted[i];
            const ticksPerBeat = Math.round((TICKS_PER_BEAT * 4) / c.den);
            const ticksPerBar = ticksPerBeat * c.num;
            const prev = segs[i - 1]; // undefined when i === 0
            const startTick = prev ? prev.startTick + (c.bar - prev.bar) * prev.ticksPerBar : 0;
            segs.push({ bar: c.bar, startTick, num: c.num, den: c.den, ticksPerBeat, ticksPerBar });
        }
        this.segs = segs;
        this.first = segs[0]; // ≥1 by construction
    }
    /** Convenience for the single global meter (Phase 0). */
    static global(num, den) {
        return new TimeAxis([{ bar: 0, num, den }]);
    }
    /** The meter section covering `tick` (last section is open-ended). Never undefined. */
    segAt(tick) {
        let s = this.first;
        for (let i = 1; i < this.segs.length; i++) {
            const seg = this.segs[i];
            if (seg.startTick <= tick)
                s = seg;
            else
                break;
        }
        return s;
    }
    ticksPerBeatAt(tick) {
        return this.segAt(tick).ticksPerBeat;
    }
    ticksPerBarAt(tick) {
        return this.segAt(tick).ticksPerBar;
    }
    /** Bar/beat/sub at a tick. `bar` and `beat` are 1-based (display); `sub` is a 0-based quarter-of-beat.
     *  Mirrors the old Toolbar formula exactly for a 4/4 project. */
    tickToBarBeat(tick) {
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
    tickAtBar(bar) {
        let s = this.first;
        for (let i = 1; i < this.segs.length; i++) {
            const seg = this.segs[i];
            if (seg.bar <= bar)
                s = seg;
            else
                break;
        }
        return s.startTick + (bar - s.bar) * s.ticksPerBar;
    }
    /** Total tick span of `nBars` bars from the start — used by `computeTotalTicks`'s scroll-headroom
     *  floor. With a single meter this is `nBars * ticksPerBar`; map-correct across sections too. */
    ticksForBars(nBars) {
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
    gridLinesInRange(startTick, endTick) {
        const lines = [];
        if (endTick <= startTick)
            return lines;
        for (let si = 0; si < this.segs.length; si++) {
            const s = this.segs[si];
            const next = this.segs[si + 1]; // Seg | undefined (last section is open-ended)
            const segEnd = next ? next.startTick : endTick;
            if (segEnd <= startTick)
                continue;
            if (s.startTick >= endTick)
                break;
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
    /**
     * ② SUB-BEAT grid for the vocal piano-roll (S48 Phase 4) — divides each BEAT into `div` equal parts and
     * yields every line over `[startTick, endTick)`, tagged with a `level` ("bar" downbeat / "beat" / "sub")
     * and its 0-based sub index within the beat (`sub`), so the draw can shade coarser vs finer subdivisions.
     * `div` SUPPORTS TRIPLETS (÷3/÷6/÷12, not just powers of two) — the base vocal grid is 12/beat, so both
     * binary (8th=÷2, 16th=÷4) and triplet (8th-T=÷3, 16th-T=÷6) snap targets land on it, which is exactly
     * why v2 uses a constant six-based grid instead of a switchable triplet mode (the v1 position-corruption
     * trap). Floors the first line to the left edge like `gridLinesInRange`. Leaves `gridLinesInRange`
     * (the arrangement's bar/beat set) BIT-FOR-BIT untouched — this is a new, additive method.
     */
    subGridLinesInRange(startTick, endTick, div) {
        const out = [];
        if (endTick <= startTick || !Number.isFinite(div) || div < 1)
            return out;
        for (let si = 0; si < this.segs.length; si++) {
            const s = this.segs[si];
            const next = this.segs[si + 1];
            const segEnd = next ? next.startTick : endTick;
            if (segEnd <= startTick)
                continue;
            if (s.startTick >= endTick)
                break;
            const step = s.ticksPerBeat / div;
            const from = Math.max(startTick, s.startTick);
            const stop = Math.min(endTick, segEnd);
            let k = Math.floor((from - s.startTick) / step); // floor to the sub-line at/below the left edge
            for (;; k++) {
                const tick = s.startTick + Math.round(k * step);
                if (tick >= stop)
                    break;
                const rel = tick - s.startTick;
                const level = rel % s.ticksPerBar === 0 ? "bar" : rel % s.ticksPerBeat === 0 ? "beat" : "sub";
                out.push({ tick, level, sub: ((k % div) + div) % div });
            }
        }
        return out;
    }
}
/** Format an absolute tick as a "bar:beat:sub" transport readout (1-based bar/beat, 2-digit sub). ONE format,
 *  shared by the Toolbar position display and the ② vocal-editor playhead readout (no drift). */
export function formatBarBeat(axis, tick) {
    const { bar, beat, sub } = axis.tickToBarBeat(tick);
    return `${bar}:${beat}:${sub.toString().padStart(2, "0")}`;
}
