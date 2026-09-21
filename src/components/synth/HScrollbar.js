import { jsx as _jsx, Fragment as _Fragment, jsxs as _jsxs } from "react/jsx-runtime";
import { useRef, useCallback, useEffect } from "react";
import { useAppStore } from "../../store/app";
import "./HScrollbar.css";
const MIN_ZOOM = 0.1;
const MAX_ZOOM = 10;
const clampZoom = (z) => Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, z));
/** S73e: the CONTROLLED scrollbar view (two styled divs + drag/click logic). Shared by the DAW
 *  (store-backed wrapper below) and the vocal editor (off-React viewRef-backed) — ONE drag/geometry
 *  implementation, two scroll sources (NO-Dup). */
export function HScrollbarView({ scrollX, totalWidth, viewWidth, onChange, currentZoom, onZoomChange }) {
    const trackRef = useRef(null);
    const isDragging = useRef(false);
    const justDragged = useRef(false);
    const dragStartMouseX = useRef(0);
    const dragStartScrollX = useRef(0);
    // Thumb-edge drag → zoom (only active when the caller supplies currentZoom + onZoomChange).
    const edgeDrag = useRef(null);
    // Keep latest values in refs so event listeners always see fresh data
    const onChangeRef = useRef(onChange);
    const totalWidthRef = useRef(totalWidth);
    const maxScrollRef = useRef(0);
    const scrollXRef = useRef(scrollX);
    const onZoomChangeRef = useRef(onZoomChange);
    const currentZoomRef = useRef(0);
    onChangeRef.current = onChange;
    totalWidthRef.current = totalWidth;
    maxScrollRef.current = Math.max(0, totalWidth - viewWidth);
    scrollXRef.current = scrollX;
    onZoomChangeRef.current = onZoomChange;
    currentZoomRef.current = currentZoom ?? 0;
    const maxScroll = maxScrollRef.current;
    const thumbRatio = Math.min(1, viewWidth / Math.max(1, totalWidth));
    const thumbLeft = maxScroll > 0 ? (scrollX / maxScroll) * (1 - thumbRatio) * 100 : 0;
    const handleTrackClick = useCallback((e) => {
        if (justDragged.current) {
            justDragged.current = false;
            return;
        }
        if (edgeDrag.current)
            return; // an edge-drag click shouldn't re-position the thumb
        const track = trackRef.current;
        if (!track)
            return;
        const rect = track.getBoundingClientRect();
        const clickRatio = (e.clientX - rect.left) / rect.width;
        onChangeRef.current(Math.round(clickRatio * maxScrollRef.current));
    }, []);
    const handleThumbDown = useCallback((e) => {
        e.stopPropagation();
        e.preventDefault();
        isDragging.current = true;
        dragStartMouseX.current = e.clientX;
        dragStartScrollX.current = scrollXRef.current;
    }, []);
    const handleEdgeDown = useCallback((e, side) => {
        // Only zoom from an edge when the caller enabled it.
        if (!onZoomChangeRef.current)
            return;
        e.stopPropagation();
        e.preventDefault();
        edgeDrag.current = { side, startX: e.clientX, startZoom: currentZoomRef.current || 1 };
        justDragged.current = true; // swallow the track-mouseup click that follows
    }, []);
    useEffect(() => {
        const onMove = (e) => {
            if (!isDragging.current || !trackRef.current)
                return;
            const trackWidth = trackRef.current.getBoundingClientRect().width;
            const dx = e.clientX - dragStartMouseX.current;
            // 标准滚动条方向: 滑块跟随鼠标 — 向右拖, 滑块右移, 视口右移(scrollX增大); 向左拖相反
            const scrollDelta = (dx / trackWidth) * totalWidthRef.current;
            const newScroll = Math.max(0, Math.min(maxScrollRef.current, dragStartScrollX.current + scrollDelta));
            onChangeRef.current(Math.round(newScroll));
        };
        const onUp = () => {
            if (isDragging.current) {
                isDragging.current = false;
                justDragged.current = true;
                setTimeout(() => { justDragged.current = false; }, 0);
            }
        };
        document.addEventListener("mousemove", onMove);
        document.addEventListener("mouseup", onUp);
        return () => {
            document.removeEventListener("mousemove", onMove);
            document.removeEventListener("mouseup", onUp);
        };
    }, []);
    // Separate move/up pass for the edge-drag → zoom gesture (independent of the thumb move).
    useEffect(() => {
        const onMove = (e) => {
            const ed = edgeDrag.current;
            if (!ed)
                return;
            const dx = e.clientX - ed.startX;
            const dz = dx / 300; // 300px across the track ≈ doubling/halving around 1×
            // 方向: 手柄边缘跟随鼠标 — 右手柄向右拖 / 左手柄向左拖 → 滑块变大 = 缩小(轨道变密);
            // 反向拖 → 滑块缩小 = 放大(内容变疏)
            let z = ed.startZoom * (ed.side === "right" ? 1 - dz : 1 + dz);
            z = clampZoom(z);
            onZoomChangeRef.current?.(z);
        };
        const onUp = () => {
            if (edgeDrag.current) {
                edgeDrag.current = null;
                setTimeout(() => { justDragged.current = false; }, 0);
            }
        };
        document.addEventListener("mousemove", onMove);
        document.addEventListener("mouseup", onUp);
        return () => {
            document.removeEventListener("mousemove", onMove);
            document.removeEventListener("mouseup", onUp);
        };
    }, []);
    return (_jsx("div", { className: "hscrollbar-track", ref: trackRef, onClick: handleTrackClick, children: _jsx("div", { className: "hscrollbar-thumb", style: {
                left: `${thumbLeft}%`,
                width: `${Math.max(5, thumbRatio * 100)}%`,
            }, onMouseDown: handleThumbDown, children: onZoomChange && (_jsxs(_Fragment, { children: [_jsx("div", { className: "hscroll-handle hscroll-handle-l", onMouseDown: (e) => handleEdgeDown(e, "left") }), _jsx("div", { className: "hscroll-handle hscroll-handle-r", onMouseDown: (e) => handleEdgeDown(e, "right") })] })) }) }));
}
/** The DAW arrangement scrollbar — self-subscribes app scrollX (and zoom) so horizontal scroll re-renders
 *  ONLY this tiny component, not the whole DawView subtree (original S1 behavior, unchanged). */
export function HScrollbar(props) {
    const scrollX = useAppStore((s) => s.scrollX);
    const zoom = useAppStore((s) => s.zoom);
    const setZoom = useAppStore((s) => s.setZoom);
    return _jsx(HScrollbarView, { scrollX: scrollX, currentZoom: zoom, onZoomChange: setZoom, ...props });
}
