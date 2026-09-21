import { jsx as _jsx, Fragment as _Fragment } from "react/jsx-runtime";
import "./PanelResizeHandles.css";
const DIRS = ["n", "s", "e", "w", "ne", "nw", "se", "sw"];
/** The 8 self-drawn resize hit zones for a floating panel — pairs with
 *  useFloatingPanel (ONE source for all panels; invisible zones + directional
 *  cursors, never the native CSS `resize` control). Render as the LAST child of
 *  the panel root so the zones sit above the content. */
export function PanelResizeHandles({ start, }) {
    return (_jsx(_Fragment, { children: DIRS.map((d) => (_jsx("div", { className: `panel-resize panel-resize-${d}`, onPointerDown: start(d) }, d))) }));
}
