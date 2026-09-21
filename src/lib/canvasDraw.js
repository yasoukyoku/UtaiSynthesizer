import { rgba, ACCENT_RGB } from "./trackColors";
/**
 * Single source of truth for the timeline chrome that the arrangement canvas, the timeline ruler and the
 * minimap each used to draw with their own copies of the same literals + loops. A 2D context can't read
 * CSS `var()`s, so the colors are concrete literals mirrored from theme.css (noted per-constant) — keep
 * them in sync. The accent hue itself lives in trackColors (`ACCENT_RGB`), reused here; skin-aware
 * surfaces (TimelineRuler) override it per-draw via `BeatGridOpts.accentRgb`.
 */
// Canvas-chrome colors (mirror theme.css). The accent teal comes from `rgba(ACCENT_RGB, a)`, not here.
export const PLAYHEAD = "#ff6b9d"; // --accent-tertiary
export const PLAYHEAD_HOVER = "#ffadc8"; // brighter near-hover playhead (no theme var)
export const CANVAS_BORDER = "#2a3a5c"; // --border-default
export const SEPARATOR_RGB = [30, 42, 69]; // track/lane separator (#1e2a45 = --border-subtle)
let themeVarsCache = null;
export function canvasThemeVars() {
    const root = document.documentElement;
    const key = `${root.dataset.skin ?? "default"}|${root.dataset.theme ?? "dark"}`;
    if (themeVarsCache && themeVarsCache.key === key)
        return themeVarsCache.v;
    const cs = getComputedStyle(root);
    const v = {
        bgBase: cs.getPropertyValue("--bg-base").trim() || "#0d1220",
        bgSurface: cs.getPropertyValue("--bg-surface").trim() || "#131a2b",
        textPrimary: cs.getPropertyValue("--text-primary").trim() || "#e8ecf4",
        textMuted: cs.getPropertyValue("--text-muted").trim() || "#556b94",
        borderDefault: cs.getPropertyValue("--border-default").trim() || "#2a3a5c",
        accentPrimary: cs.getPropertyValue("--accent-primary").trim() || "#39c5bb",
    };
    themeVarsCache = { key, v };
    return v;
}
/**
 * Draw the vertical bar/beat grid over the visible tick range. ONE source for the grid-line loop shared
 * by the arrangement canvas and the timeline ruler — each passes its own alphas + beat-tick top. The
 * lines (positions + which are downbeats) come from the TimeAxis, which floors the first line to the
 * beat at/below the left edge (lines left of x=0 are simply clipped). For a 4/4 project this is the exact
 * pre-Phase-0 line set (beats every 480, bars every 1920) — the grid is bit-for-bit unchanged.
 */
export function drawBeatGrid(ctx, o) {
    const beatTop = o.beatTop ?? 0;
    const accent = o.accentRgb ?? ACCENT_RGB;
    const barColor = rgba(accent, o.barAlpha);
    const beatColor = rgba(accent, o.beatAlpha);
    const startTick = Math.floor(o.scrollX / o.ppt);
    const endTick = Math.ceil((o.scrollX + o.width) / o.ppt);
    for (const { tick, isBar } of o.axis.gridLinesInRange(startTick, endTick)) {
        const x = tick * o.ppt - o.scrollX;
        ctx.strokeStyle = isBar ? barColor : beatColor;
        ctx.lineWidth = isBar ? 1 : 0.5;
        ctx.beginPath();
        ctx.moveTo(x, isBar ? 0 : beatTop);
        ctx.lineTo(x, o.height);
        ctx.stroke();
    }
}
/**
 * Draw the playhead — ONE source for the line + triangle marker that the arrangement canvas, the ruler
 * and the minimap each drew separately. Visibility / off-screen guarding stays at the call site (each
 * surface clips slightly differently). The pink + hover-pink come from PLAYHEAD / PLAYHEAD_HOVER.
 */
export function drawPlayhead(ctx, o) {
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
        }
        else {
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
        }
        else {
            ctx.moveTo(o.x - hw, o.height);
            ctx.lineTo(o.x + hw, o.height);
            ctx.lineTo(o.x, o.height - d);
        }
        ctx.closePath();
        ctx.fill();
    }
}
