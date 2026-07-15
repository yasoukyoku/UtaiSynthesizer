import { useEffect, useRef, useState } from "react";
import type React from "react";
import { loadSetting, saveSetting } from "./settings";

export interface PanelRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export type ResizeDir = "n" | "s" | "e" | "w" | "ne" | "nw" | "se" | "sw";

/** Keep the drag clamp identical to the old useDraggable: at least a header sliver
 *  stays reachable so the panel can never be lost off-screen. */
const KEEP_X = 120;
const KEEP_Y = 40;
/** Viewport margin the panel size is clamped into. */
const EDGE = 16;

/**
 * Floating-panel chrome for the fixed-position DOM panels (log viewer / settings /
 * resource manager): header drag + 8-direction edge resize + a persisted rect.
 * ONE source — supersedes useDraggable (S67, panels grew resize because the fixed
 * 480px log panel's 11px text was unreadable). The rect persists per `storageKey`
 * (settings.ts localStorage, the utai.workflowPanelHeight precedent) and is
 * re-clamped into the viewport on load so a monitor change can't strand it.
 * Pair the returned `startResize` with <PanelResizeHandles> (self-drawn hit zones —
 * house rule: never the native CSS `resize` control).
 */
export function useFloatingPanel(opts: {
  storageKey: string;
  initial: () => PanelRect;
  minW: number;
  minH: number;
}) {
  const { storageKey, minW, minH } = opts;
  const [rect, setRect] = useState<PanelRect>(() =>
    clampRect(loadSetting<PanelRect | null>(storageKey, null) ?? opts.initial(), minW, minH),
  );
  const rectRef = useRef(rect);
  rectRef.current = rect;

  const persist = () => saveSetting(storageKey, rectRef.current);

  // the old vh-sized panels tracked window shrinks for free — a px rect must
  // re-clamp live or the footer/south edge ends up unreachable off-screen
  useEffect(() => {
    const onWindowResize = () => setRect((r) => clampRect(r, minW, minH));
    window.addEventListener("resize", onWindowResize);
    return () => window.removeEventListener("resize", onWindowResize);
  }, [minW, minH]);

  const startDrag = (e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest("button")) return;
    const offX = e.clientX - rectRef.current.x;
    const offY = e.clientY - rectRef.current.y;
    const onMove = (ev: MouseEvent) => {
      setRect((r) => ({
        ...r,
        x: Math.min(Math.max(0, ev.clientX - offX), window.innerWidth - KEEP_X),
        y: Math.min(Math.max(0, ev.clientY - offY), window.innerHeight - KEEP_Y),
      }));
    };
    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      persist();
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  };

  const startResize = (dir: ResizeDir) => (e: React.PointerEvent) => {
    // no preventDefault: cancelling pointerdown suppresses the compat mousedown that
    // dismisses open menus/popups (review S67); selection during the drag is
    // suppressed via a temporary body user-select instead
    e.stopPropagation();
    const from = { ...rectRef.current };
    const sx = e.clientX;
    const sy = e.clientY;
    const prevSelect = document.body.style.userSelect;
    document.body.style.userSelect = "none";
    const onMove = (ev: PointerEvent) => {
      const dx = ev.clientX - sx;
      const dy = ev.clientY - sy;
      setRect(() => {
        let { x, y, w, h } = from;
        if (dir.includes("e")) w = from.w + dx;
        if (dir.includes("s")) h = from.h + dy;
        if (dir.includes("w")) w = from.w - dx;
        if (dir.includes("n")) h = from.h - dy;
        let cw = Math.min(Math.max(minW, w), Math.max(minW, window.innerWidth - EDGE));
        let ch = Math.min(Math.max(minH, h), Math.max(minH, window.innerHeight - EDGE));
        // west/north drags move the origin so the OPPOSITE edge stays pinned; when
        // the origin hits the viewport edge the SIZE gives way — never the far edge
        // (review S67: a Math.max(0,x) alone shifted the pinned east/south edge)
        if (dir.includes("w")) {
          const right = from.x + from.w;
          x = right - cw;
          if (x < 0) {
            x = 0;
            cw = right;
          }
        }
        if (dir.includes("n")) {
          const bottom = from.y + from.h;
          y = bottom - ch;
          if (y < 0) {
            y = 0;
            ch = bottom;
          }
        }
        return { x, y, w: cw, h: ch };
      });
    };
    const onUp = () => {
      document.removeEventListener("pointermove", onMove);
      document.removeEventListener("pointerup", onUp);
      document.body.style.userSelect = prevSelect;
      persist();
    };
    document.addEventListener("pointermove", onMove);
    document.addEventListener("pointerup", onUp);
  };

  /** Spread onto the panel root — position AND size are state-driven (the size
   *  overrides the stylesheet's pre-S67 fixed default). */
  const style: React.CSSProperties = { left: rect.x, top: rect.y, width: rect.w, height: rect.h };

  return { rect, style, startDrag, startResize };
}

function clampRect(r: PanelRect, minW: number, minH: number): PanelRect {
  const w = Math.min(Math.max(minW, r.w), Math.max(minW, window.innerWidth - EDGE));
  const h = Math.min(Math.max(minH, r.h), Math.max(minH, window.innerHeight - EDGE));
  return {
    x: Math.min(Math.max(0, r.x), Math.max(0, window.innerWidth - KEEP_X)),
    y: Math.min(Math.max(0, r.y), Math.max(0, window.innerHeight - KEEP_Y)),
    w,
    h,
  };
}
