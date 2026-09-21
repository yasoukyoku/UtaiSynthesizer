import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useRef, useCallback, useEffect } from "react";
import "./VolumeFader.css";
/** Pan tooltip text ("L50" / "C" / "R50") for a −1..1 fader — the ONE pan display convention. */
export function formatPan(v) {
    if (v === 0)
        return "C";
    return v < 0 ? `L${Math.round(-v * 100)}` : `R${Math.round(v * 100)}`;
}
/** Volume-fader dB text ("-∞ dB" at/below the floor, else "+1.5 dB" / "-6.0 dB"). The ONE dB
 *  formatter shared by the fader tooltip AND the always-on TrackList numeric readout. */
export function formatDb(v, min) {
    return v <= min ? "-∞ dB" : `${v > 0 ? "+" : ""}${v.toFixed(1)} dB`;
}
export function VolumeFader({ value, min, max, onChange, onGestureStart, onGestureEnd, width = 48, height = 80, orientation = "horizontal", step = 0.5, fillFrom = "left", format, tip }) {
    const trackRef = useRef(null);
    const dragging = useRef(false);
    const isVertical = orientation === "vertical";
    // Everything the document listeners need lives in refs so calcValue/the listener effect are STABLE
    // (empty deps). If calcValue depended on `value` (a per-frame controlled prop), the listener effect
    // would tear down + re-create on every drag frame — and its cleanup would fire mid-drag.
    const onChangeRef = useRef(onChange);
    onChangeRef.current = onChange;
    const onStartRef = useRef(onGestureStart);
    onStartRef.current = onGestureStart;
    const onEndRef = useRef(onGestureEnd);
    onEndRef.current = onGestureEnd;
    const minRef = useRef(min);
    minRef.current = min;
    const maxRef = useRef(max);
    maxRef.current = max;
    const stepRef = useRef(step);
    stepRef.current = step;
    const valueRef = useRef(value);
    valueRef.current = value;
    const ratio = (value - min) / (max - min);
    const calcValue = useCallback((clientX, clientY) => {
        const el = trackRef.current;
        if (!el)
            return valueRef.current;
        const rect = el.getBoundingClientRect();
        let r;
        if (isVertical) {
            // Vertical fader: bottom = max, top = min (traditional mixer convention).
            r = Math.max(0, Math.min(1, 1 - (clientY - rect.top) / rect.height));
        }
        else {
            r = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
        }
        const raw = minRef.current + r * (maxRef.current - minRef.current);
        // Quantize to `step`, then snap away float dust (0.1-steps yield 0.30000000000000004).
        return Math.round((Math.round(raw / stepRef.current) * stepRef.current) * 1000) / 1000;
    }, [isVertical]);
    const handleDown = useCallback((e) => {
        e.stopPropagation();
        e.preventDefault();
        document.activeElement?.blur?.();
        dragging.current = true;
        onStartRef.current?.();
        onChangeRef.current(calcValue(e.clientX, e.clientY));
    }, [calcValue]);
    useEffect(() => {
        const onMove = (e) => {
            if (!dragging.current)
                return;
            onChangeRef.current(calcValue(e.clientX, e.clientY));
        };
        const onUp = () => {
            if (!dragging.current)
                return;
            dragging.current = false;
            onEndRef.current?.();
        };
        document.addEventListener("mousemove", onMove);
        document.addEventListener("mouseup", onUp);
        return () => {
            document.removeEventListener("mousemove", onMove);
            document.removeEventListener("mouseup", onUp);
            if (dragging.current) {
                dragging.current = false;
                onEndRef.current?.();
            }
        };
    }, [calcValue]);
    const zeroRatio = (0 - min) / (max - min);
    const fillLeft = fillFrom === "center" ? Math.min(ratio, zeroRatio) : 0;
    const fillWidth = fillFrom === "center" ? Math.abs(ratio - zeroRatio) : ratio;
    // Vertical fader: use top/height instead of left/width.
    const trackStyle = isVertical
        ? { height, width: Math.max(12, Math.min(20, width * 0.4)) }
        : { width, height: undefined };
    return (_jsxs("div", { className: `vol-fader ${isVertical ? "vol-fader-vertical" : ""}`, ref: trackRef, style: trackStyle, onMouseDown: handleDown, onClick: (e) => e.stopPropagation(), title: `${tip ? `${tip} — ` : ""}${format ? format(value) : formatDb(value, min)}`, children: [_jsx("div", { className: "vol-track" }), _jsx("div", { className: "vol-zero", style: isVertical
                    ? { bottom: `${zeroRatio * 100}%`, left: undefined, top: undefined }
                    : { left: `${zeroRatio * 100}%` } }), _jsx("div", { className: "vol-fill", style: isVertical
                    ? { bottom: `${fillLeft * 100}%`, height: `${fillWidth * 100}%`, left: undefined, width: undefined, top: undefined }
                    : { left: `${fillLeft * 100}%`, width: `${fillWidth * 100}%` } }), _jsx("div", { className: "vol-thumb", style: isVertical
                    ? { bottom: `${ratio * 100}%`, left: undefined, top: undefined }
                    : { left: `${ratio * 100}%` } })] }));
}
