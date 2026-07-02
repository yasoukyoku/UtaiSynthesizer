import type { Track } from "../types/project";

/** Snap distance in SCREEN pixels. Converted to ticks at the call site via the current ppt, so the
 *  snap "feel" stays constant across zoom levels (a fixed tick tolerance would feel huge when zoomed
 *  in and tiny when zoomed out). */
export const SNAP_PX = 8;

/** Collect snap-target ticks: every (non-loading) segment's start and end, plus tick 0, optionally
 *  excluding some segment ids (the ones being dragged so they don't snap to themselves) and adding
 *  extra points (e.g. the playhead, when snapping clips). */
export function collectSnapTicks(tracks: Track[], excludeIds?: Set<string>, ...extra: number[]): number[] {
  const pts: number[] = [0, ...extra];
  for (const tk of tracks) {
    for (const s of tk.segments) {
      if (s.loading) continue; // loading placeholders have no real duration yet
      if (excludeIds?.has(s.id)) continue;
      pts.push(s.startTick, s.startTick + s.durationTicks);
    }
  }
  return pts;
}

/** Snap a single tick to the nearest target within `tol` ticks; returns the original tick if none. */
export function snapTick(tick: number, targets: number[], tol: number): number {
  let best = tick;
  let bestDist = tol;
  for (const t of targets) {
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
 *  within `tol`). */
export function snapMovedStart(startTick: number, durationTicks: number, targets: number[], tol: number): number {
  let best = startTick;
  let bestDist = tol;
  const end = startTick + durationTicks;
  for (const t of targets) {
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
