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

/** 画布绘制用轨道色: 优先轨道头自定义色(track.color, 用户在色条上选的 hex),
 *  保持「音轨块颜色 = 轨道头颜色」一致; 未自定义时回退到类型默认色。 */
export function trackDrawRgb(track: Pick<Track, "color" | "trackType">): [number, number, number] {
  if (track.color && track.color.startsWith("#")) {
    return hexToRgb(track.color, trackRgb(track.trackType));
  }
  return trackRgb(track.trackType);
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

/** Parse a 3/6-digit hex color ("#a78bfa") to an [r,g,b] tuple; on any parse failure return `fallback`
 *  (default: the theme accent). Used by canvas draw loops that follow the active skin by reading
 *  `--accent-primary` from computed styles (ChordTrack/TimelineRuler). */
export function hexToRgb(hex: string, fallback: [number, number, number] = ACCENT_RGB): [number, number, number] {
  const m = /^#?([\da-f]{3}|[\da-f]{6})$/i.exec(hex.trim());
  if (!m) return fallback;
  let s = m[1]!;
  if (s.length === 3) s = s[0]! + s[0] + s[1]! + s[1] + s[2]! + s[2];
  return [parseInt(s.slice(0, 2), 16), parseInt(s.slice(2, 4), 16), parseInt(s.slice(4, 6), 16)];
}

/** Theme accent as a canvas hex literal (= `rgba(ACCENT_RGB, 1)`) — used by the loading spinner etc. */
export const ACCENT = "#39c5bb";

/** Random color palette for new tracks (when no audio is added yet).
 *  These colors are visually distinct and work well with the dark theme. */
export const RANDOM_TRACK_COLORS = [
  "#60A5FA", // blue
  "#39C5BB", // teal
  "#A78BFA", // purple
  "#F59E0B", // amber
  "#10B981", // emerald
  "#F472B6", // pink
  "#8B5CF6", // violet
  "#EF4444", // red
  "#14B8A6", // cyan
  "#F97316", // orange
];

/** Get a random color from the track color palette */
export function getRandomTrackColor(): string {
  return RANDOM_TRACK_COLORS[Math.floor(Math.random() * RANDOM_TRACK_COLORS.length)]!;
}

/** 挑一个「未被现有轨道占用」的随机轨道色 —— 所有新建轨(音频/乐器/人声/粘贴)都走这里,
 *  保证全局不出现相同颜色; 10 色用满后回退到使用次数最少的颜色。 */
export function pickUniqueTrackColor(existing: { color?: string }[]): string {
  const used = new Map<string, number>();
  for (const t of existing) {
    if (t.color && t.color.startsWith("#")) used.set(t.color, (used.get(t.color) ?? 0) + 1);
  }
  let min = Infinity;
  for (const c of RANDOM_TRACK_COLORS) min = Math.min(min, used.get(c) ?? 0);
  const least = RANDOM_TRACK_COLORS.filter((c) => (used.get(c) ?? 0) === min);
  return least[Math.floor(Math.random() * least.length)]!;
}

/** Sub-lane GROUP palette ("r,g,b" strings for canvas rgba() + the header's `--lane-rgb` CSS var),
 *  cycled by the group-run index within a track (all rows of one 组 share the hue, so grouping reads
 *  at a glance). ONE source for the canvas lane rows AND the header column's group bar/bracket. */
export const LANE_COLORS = ["78,205,196", "255,184,108", "168,130,255", "255,107,129", "114,224,175", "255,214,102"];

/** Selection gold ("r,g,b") — the segment AND sub-lane-group selection glow build their stroke/shadow
 *  alpha variants from this ONE hue (they must read identical; the pair was drifting by copy-paste).
 *  NOTE: equals LANE_COLORS[5] by coincidence — a 6th lane group's hue matching the selection cue is a
 *  known (accepted) collision; change the palette entry, not this, if it ever bites. */
export const SELECTION_GLOW_RGB = "255,214,102";

/** S59c: the sub-lane loudness ENVELOPE line — a dedicated near-white, always drawn over a dark
 *  halo under-stroke. Deliberately NOT hue-based: LANE_COLORS[0] ≈ the accent teal and [3] ≈ the
 *  playhead pink, so any palette-derived envelope color eventually sinks into the very waveform
 *  it rides (§user: 真串色了). White-on-halo reads on every hue. */
export const ENVELOPE_LINE = "#e8edf5";
export const ENVELOPE_HALO = "rgba(10, 14, 24, 0.85)";
/** S73b 调教所有权标记(VocalEditor 音符左缘竖条):用户调教=金——与 note 青/选中紫/OOV 红
 *  都不冲突;机器调教(autoTuned)不标。SV1「Manual 音符角标」的本家版。 */
export const TUNED_MARKER = "#e6c25a";
