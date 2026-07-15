import type React from "react";
import type { ResizeDir } from "../../lib/useFloatingPanel";
import "./PanelResizeHandles.css";

const DIRS: ResizeDir[] = ["n", "s", "e", "w", "ne", "nw", "se", "sw"];

/** The 8 self-drawn resize hit zones for a floating panel — pairs with
 *  useFloatingPanel (ONE source for all panels; invisible zones + directional
 *  cursors, never the native CSS `resize` control). Render as the LAST child of
 *  the panel root so the zones sit above the content. */
export function PanelResizeHandles({
  start,
}: {
  start: (dir: ResizeDir) => (e: React.PointerEvent) => void;
}) {
  return (
    <>
      {DIRS.map((d) => (
        <div key={d} className={`panel-resize panel-resize-${d}`} onPointerDown={start(d)} />
      ))}
    </>
  );
}
