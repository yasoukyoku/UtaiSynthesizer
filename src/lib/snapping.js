/** Snap distance in SCREEN pixels. Converted to ticks at the call site via the current ppt, so the
 *  snap "feel" stays constant across zoom levels (a fixed tick tolerance would feel huge when zoomed
 *  in and tiny when zoomed out). */
export const SNAP_PX = 8;
/** Multiples of `gridStep` that cover the interval `[from, to]` plus one step either side — the grid
 *  targets that could plausibly win a snap for a clip/caret spanning `[from,to]`. Bounded (a fixed
 *  chunk around the span), so injecting them into the magnetic target list costs ~span/step which is
 *  tiny for a typical clip. Empty when `gridStep <= 0` (grid snapping off). */
export function collectGridTicks(from, to, gridStep) {
    if (gridStep <= 0)
        return [];
    const first = Math.floor(from / gridStep) * gridStep;
    const out = [];
    for (let t = first; t <= to + gridStep; t += gridStep)
        out.push(t);
    return out;
}
/** Collect snap-target ticks: every (non-loading) segment's start and end, plus tick 0, optionally
 *  excluding some segment ids (the ones being dragged so they don't snap to themselves) and adding
 *  extra points (e.g. the playhead, when snapping clips). */
export function collectSnapTicks(tracks, excludeIds, ...extra) {
    const pts = [0, ...extra];
    for (const tk of tracks) {
        for (const s of tk.segments) {
            if (s.loading)
                continue; // loading placeholders have no real duration yet
            if (excludeIds?.has(s.id))
                continue;
            pts.push(s.startTick, s.startTick + s.durationTicks);
        }
    }
    return pts;
}
/** Snap a single tick to the nearest target within `tol` ticks; returns the original tick if none.
 *  When a grid `gridStep` (>0) is given its multiples are merged into the target list, so a caret or
 *  resizing edge also magnetizes onto the grid. */
export function snapTick(tick, targets, tol, gridStep = 0) {
    const all = gridStep > 0 ? [...targets, ...collectGridTicks(tick - tol, tick + tol, gridStep)] : targets;
    let best = tick;
    let bestDist = tol;
    for (const t of all) {
        const d = Math.abs(t - tick);
        if (d <= bestDist) {
            bestDist = d;
            best = t;
        }
    }
    return best;
}
/** Snap a moving clip by whichever of its edges (start OR end) is closest to a target — so a clip
 *  snaps when either edge lines up. Returns the adjusted start tick (unchanged if neither edge is
 *  within `tol`). When a grid `gridStep` (>0) is given its multiples are merged into the target list
 *  so the clip also aligns onto the grid. */
export function snapMovedStart(startTick, durationTicks, targets, tol, gridStep = 0) {
    let best = startTick;
    let bestDist = tol;
    const end = startTick + durationTicks;
    const all = gridStep > 0
        ? [...targets, ...collectGridTicks(startTick - tol, end + tol, gridStep)]
        : targets;
    for (const t of all) {
        const ds = Math.abs(t - startTick);
        if (ds <= bestDist) {
            bestDist = ds;
            best = t;
        }
        const de = Math.abs(t - end);
        if (de <= bestDist) {
            bestDist = de;
            best = t - durationTicks;
        }
    }
    return best;
}
