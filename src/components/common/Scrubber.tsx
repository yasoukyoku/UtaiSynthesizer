import { useRef } from "react";
import "./Scrubber.css";

/** Thin div scrubber (house style: square 4×12 head — never a solid dot, never a native
 *  range input). Click/drag to seek; the parent drives `value` (0..1) via its own rAF
 *  ticker. Extracted from TrainingPage (S41) in S66 so the workflow node output preview
 *  shares the ONE implementation. */
export function Scrubber({
  value,
  onSeek,
  className,
}: {
  value: number;
  onSeek: (frac: number) => void;
  className?: string;
}) {
  const trackRef = useRef<HTMLDivElement>(null);
  const seekAt = (clientX: number) => {
    const el = trackRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    onSeek((clientX - r.left) / Math.max(1, r.width));
  };
  const onDown = (e: React.PointerEvent) => {
    e.stopPropagation();
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    seekAt(e.clientX);
  };
  const onMove = (e: React.PointerEvent) => {
    if (e.buttons & 1) seekAt(e.clientX);
  };
  return (
    <div
      className={`ui-scrubber ${className ?? ""}`}
      ref={trackRef}
      onPointerDown={onDown}
      onPointerMove={onMove}
    >
      <div className="ui-scrubber-fill" style={{ width: `${Math.round(value * 100)}%` }} />
      <div className="ui-scrubber-head" style={{ left: `${Math.round(value * 100)}%` }} />
    </div>
  );
}
