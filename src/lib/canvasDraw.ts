import { TICKS_PER_BEAT } from "./constants";
import { rgba, ACCENT_RGB } from "./trackColors";

/**
 * Single source of truth for the timeline chrome that the arrangement canvas, the timeline ruler and the
 * minimap each used to draw with their own copies of the same literals + loops. A 2D context can't read
 * CSS `var()`s, so the colors are concrete literals mirrored from theme.css (noted per-constant) — keep
 * them in sync. The accent hue itself lives in trackColors (`ACCENT_RGB`), reused here.
 */

// Canvas-chrome colors (mirror theme.css). The accent teal comes from `rgba(ACCENT_RGB, a)`, not here.
export const PLAYHEAD = "#ff6b9d"; // --accent-tertiary
export const PLAYHEAD_HOVER = "#ffadc8"; // brighter near-hover playhead (no theme var)
export const CANVAS_BORDER = "#2a3a5c"; // --border-default
export const SEPARATOR_RGB: [number, number, number] = [30, 42, 69]; // track/lane separator (#1e2a45 = --border-subtle)

type AnyCtx = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;

export interface BeatGridOpts {
  /** pixels-per-tick (PIXELS_PER_TICK * zoom). */
  ppt: number;
  scrollX: number;
  width: number;
  height: number;
  /** TICKS_PER_BEAT * timeSignature[0]. */
  ticksPerBar: number;
  /** Accent alpha for bar (downbeat) lines. */
  barAlpha: number;
  /** Accent alpha for the in-between beat lines. */
  beatAlpha: number;
  /** y at which non-bar beat lines start (bar lines always start at 0). Default 0 → full-height beat
   *  lines (arrangement); pass `height - n` for short ruler ticks (timeline ruler). */
  beatTop?: number;
}

/**
 * Draw the vertical bar/beat grid over the visible tick range. ONE source for the grid-line loop shared
 * by the arrangement canvas and the timeline ruler — each passes its own alphas + beat-tick top. Aligns
 * the first line to the beat at/below the left edge (lines left of x=0 are simply clipped).
 */
export function drawBeatGrid(ctx: AnyCtx, o: BeatGridOpts) {
  const beatTop = o.beatTop ?? 0;
  const barColor = rgba(ACCENT_RGB, o.barAlpha);
  const beatColor = rgba(ACCENT_RGB, o.beatAlpha);
  const startTick = Math.floor(o.scrollX / o.ppt);
  const endTick = Math.ceil((o.scrollX + o.width) / o.ppt);
  for (let tick = startTick - (startTick % TICKS_PER_BEAT); tick < endTick; tick += TICKS_PER_BEAT) {
    const x = tick * o.ppt - o.scrollX;
    const isBar = tick % o.ticksPerBar === 0;
    ctx.strokeStyle = isBar ? barColor : beatColor;
    ctx.lineWidth = isBar ? 1 : 0.5;
    ctx.beginPath();
    ctx.moveTo(x, isBar ? 0 : beatTop);
    ctx.lineTo(x, o.height);
    ctx.stroke();
  }
}

export interface PlayheadOpts {
  /** Pixel x of the playhead (already mapped to canvas space by the caller). */
  x: number;
  height: number;
  /** Draw the full-height vertical line (arrangement, minimap). Omit for a cap-only marker (ruler). */
  line?: boolean;
  /** Line width — default 1.5 (arrangement); the minimap passes 1. */
  lineWidth?: number;
  /** Brighten + soft-glow the playhead (the arrangement's "pointer near" affordance). */
  glow?: boolean;
  /** Triangle marker: "top" points down from y=0 (arrangement), "bottom" points up from y=height
   *  (ruler). Omit for no marker (minimap). */
  cap?: "top" | "bottom";
  /** Half-width of the triangle marker — default 6 (arrangement); the ruler passes 5. */
  capHalfWidth?: number;
  /** Depth of the triangle marker — default 8 (arrangement); the ruler passes 6. */
  capDepth?: number;
}

/**
 * Draw the playhead — ONE source for the line + triangle marker that the arrangement canvas, the ruler
 * and the minimap each drew separately. Visibility / off-screen guarding stays at the call site (each
 * surface clips slightly differently). The pink + hover-pink come from PLAYHEAD / PLAYHEAD_HOVER.
 */
export function drawPlayhead(ctx: AnyCtx, o: PlayheadOpts) {
  const color = o.glow ? PLAYHEAD_HOVER : PLAYHEAD;
  if (o.line) {
    if (o.glow) {
      ctx.save();
      ctx.shadowColor = PLAYHEAD;
      ctx.shadowBlur = 12;
      ctx.strokeStyle = PLAYHEAD_HOVER;
      ctx.lineWidth = o.lineWidth ?? 1.5;
      ctx.beginPath();
      ctx.moveTo(o.x, 0);
      ctx.lineTo(o.x, o.height);
      ctx.stroke();
      ctx.restore();
    } else {
      ctx.strokeStyle = PLAYHEAD;
      ctx.lineWidth = o.lineWidth ?? 1.5;
      ctx.beginPath();
      ctx.moveTo(o.x, 0);
      ctx.lineTo(o.x, o.height);
      ctx.stroke();
    }
  }
  if (o.cap) {
    const hw = o.capHalfWidth ?? 6;
    const d = o.capDepth ?? 8;
    ctx.fillStyle = color;
    ctx.beginPath();
    if (o.cap === "top") {
      ctx.moveTo(o.x - hw, 0);
      ctx.lineTo(o.x + hw, 0);
      ctx.lineTo(o.x, d);
    } else {
      ctx.moveTo(o.x - hw, o.height);
      ctx.lineTo(o.x + hw, o.height);
      ctx.lineTo(o.x, o.height - d);
    }
    ctx.closePath();
    ctx.fill();
  }
}
