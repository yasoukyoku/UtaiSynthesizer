import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useRef } from "react";
import "./Scrubber.css";
/** Thin div scrubber (house style: square 4×12 head — never a solid dot, never a native
 *  range input). Click/drag to seek; the parent drives `value` (0..1) via its own rAF
 *  ticker. Extracted from TrainingPage (S41) in S66 so the workflow node output preview
 *  shares the ONE implementation. */
export function Scrubber({ value, onSeek, className, }) {
    const trackRef = useRef(null);
    const seekAt = (clientX) => {
        const el = trackRef.current;
        if (!el)
            return;
        const r = el.getBoundingClientRect();
        onSeek((clientX - r.left) / Math.max(1, r.width));
    };
    const onDown = (e) => {
        e.stopPropagation();
        e.target.setPointerCapture(e.pointerId);
        seekAt(e.clientX);
    };
    const onMove = (e) => {
        if (e.buttons & 1)
            seekAt(e.clientX);
    };
    return (_jsxs("div", { className: `ui-scrubber ${className ?? ""}`, ref: trackRef, onPointerDown: onDown, onPointerMove: onMove, children: [_jsx("div", { className: "ui-scrubber-fill", style: { width: `${Math.round(value * 100)}%` } }), _jsx("div", { className: "ui-scrubber-head", style: { left: `${Math.round(value * 100)}%` } })] }));
}
