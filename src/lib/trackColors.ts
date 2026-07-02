import type { Track } from "../types/project";

/**
 * Single source of truth for track-type colors. Previously the same three colors were independently
 * encoded as RGB arrays in Arrangement's canvas draw, as a `--track-*` CSS-var ternary in TrackList,
 * and as CSS vars in theme.css — three copies that could silently drift. Canvas code uses the RGB
 * tuple (+ `rgba`); React/CSS code uses `trackTypeCssVar`. (Keep these in sync with the
 * `--track-audio/--track-vocal/--track-instrument` vars in theme.css.)
 */
type TrackType = Track["trackType"];

export const TRACK_RGB: Record<TrackType, [number, number, number]> = {
  audio: [96, 165, 250],
  vocal: [57, 197, 187],
  instrument: [167, 139, 250],
};

export function trackRgb(type: TrackType): [number, number, number] {
  return TRACK_RGB[type] ?? TRACK_RGB.instrument;
}

export function trackTypeCssVar(type: TrackType): string {
  return type === "vocal"
    ? "var(--track-vocal)"
    : type === "audio"
      ? "var(--track-audio)"
      : "var(--track-instrument)";
}

/** Build an rgba() string from an [r,g,b] tuple (or any number[] of length ≥ 3) + alpha. */
export function rgba(c: readonly number[], a: number): string {
  return `rgba(${c[0]},${c[1]},${c[2]},${a})`;
}

/** Theme accent (--accent-primary, #39c5bb = rgb(57,197,187)) as an [r,g,b] tuple, so canvas code can
 *  build alpha variants via `rgba(ACCENT_RGB, a)`. The grid lines, the minimap viewport box and the
 *  drag-over wash all derive from this ONE hue (don't re-hardcode `57,197,187`). NOTE: this equals
 *  `TRACK_RGB.vocal` by coincidence only — they are semantically distinct; keep them independent. */
export const ACCENT_RGB: [number, number, number] = [57, 197, 187];

/** Theme accent as a canvas hex literal (= `rgba(ACCENT_RGB, 1)`) — used by the loading spinner etc. */
export const ACCENT = "#39c5bb";

/** Sub-lane GROUP palette ("r,g,b" strings for canvas rgba() + the header's `--lane-rgb` CSS var),
 *  cycled by the group-run index within a track (all rows of one 组 share the hue, so grouping reads
 *  at a glance). ONE source for the canvas lane rows AND the header column's group bar/bracket. */
export const LANE_COLORS = ["78,205,196", "255,184,108", "168,130,255", "255,107,129", "114,224,175", "255,214,102"];

/** Selection gold ("r,g,b") — the segment AND sub-lane-group selection glow build their stroke/shadow
 *  alpha variants from this ONE hue (they must read identical; the pair was drifting by copy-paste).
 *  NOTE: equals LANE_COLORS[5] by coincidence — a 6th lane group's hue matching the selection cue is a
 *  known (accepted) collision; change the palette entry, not this, if it ever bites. */
export const SELECTION_GLOW_RGB = "255,214,102";
