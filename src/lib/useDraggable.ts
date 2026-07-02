import { useState } from "react";
import type React from "react";

/**
 * Drag a floating panel by its header. Returns the current top-left position and a mousedown
 * handler to attach to the drag handle (e.g. the panel header). Mousedown on a <button> inside
 * the handle (like the close button) does not start a drag. Position is clamped so the window
 * can't be dragged fully off-screen.
 */
export function useDraggable(initial: () => { x: number; y: number }) {
  const [pos, setPos] = useState(initial);

  const startDrag = (e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest("button")) return;
    const offX = e.clientX - pos.x;
    const offY = e.clientY - pos.y;
    const onMove = (ev: MouseEvent) => {
      setPos({
        x: Math.min(Math.max(0, ev.clientX - offX), window.innerWidth - 120),
        y: Math.min(Math.max(0, ev.clientY - offY), window.innerHeight - 40),
      });
    };
    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  };

  return { pos, startDrag };
}
