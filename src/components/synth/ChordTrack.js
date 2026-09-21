import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useRef, useState, useCallback, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { useProjectStore, useTimeAxis } from "../../store/project";
import { useAppStore } from "../../store/app";
import { useChordTrackStore } from "../../store/chordTrack";
import { PIXELS_PER_TICK } from "../../lib/constants";
import { hexToRgb, rgba } from "../../lib/trackColors";
import { ContextMenu } from "../common/ContextMenu";
import "./ChordTrack.css";
/** 和弦行高度（DawView grid 的第 2 行同款常量；CSS 里 100% 填满）。 */
export const CHORD_TRACK_H = 26;
/**
 * Muno 阶段3「和弦轨道」——时间线标尺正下方的一行固定高度（26px）和弦标签条。
 *
 * · 数据：chordTrack store（会话态覆盖层，识别入口 = 轨道右键「识别和弦」）。
 * · 绘制：canvas 命令式重绘（同 TimelineRuler 模式）——scrollX/zoom/playhead 走 store
 *   订阅改 ref + rAF 合并，不重渲染 React。
 * · 交互：点击标签 → 就地改名单行输入；拖拽相邻段边界 → col-resize 调整分段
 *   （吸附到拍，两端保 ≥1 拍）；右键 → 复制和弦进行 / 清空；空态 → 引导文案。
 */
export function ChordTrack() {
    const canvasRef = useRef(null);
    const segments = useChordTrackStore((s) => s.segments);
    const keyEst = useChordTrackStore((s) => s.key);
    const timeAxis = useTimeAxis();
    const { t } = useTranslation();
    const [ctxMenu, setCtxMenu] = useState(null);
    // 就地编辑：命中段的序号 + 输入框落点（canvas 内局部 x/y，输入框 absolute 定位）。
    const [editing, setEditing] = useState(null);
    // 边界拖拽：正在拖的边界（左段 index；null = 未拖）+ hover 高亮的边界。
    const dragIdxRef = useRef(null);
    const hoverIdxRef = useRef(null);
    // 绘制输入走 ref（draw 每帧调用，不能闭包过期 React state）。
    const segsRef = useRef(segments);
    segsRef.current = segments;
    const keyRef = useRef(keyEst);
    keyRef.current = keyEst;
    // 视图 refs —— 与 TimelineRuler 相同的命令式订阅模式。
    const scrollXRef = useRef(useAppStore.getState().scrollX);
    const zoomRef = useRef(useAppStore.getState().zoom);
    const pptRef = useRef(PIXELS_PER_TICK * zoomRef.current);
    const playheadRef = useRef(useProjectStore.getState().playheadTick);
    const drawRef = useRef(() => { });
    const redrawRafRef = useRef(0);
    const requestRedraw = useCallback(() => {
        if (redrawRafRef.current)
            return;
        redrawRafRef.current = requestAnimationFrame(() => {
            redrawRafRef.current = 0;
            drawRef.current();
        });
    }, []);
    const draw = useCallback(() => {
        const canvas = canvasRef.current;
        if (!canvas)
            return;
        const ctx = canvas.getContext("2d");
        if (!ctx)
            return;
        const dpr = devicePixelRatio;
        const { width, height } = canvas.getBoundingClientRect();
        const cw = Math.round(width * dpr);
        const ch = Math.round(height * dpr);
        if (canvas.width !== cw || canvas.height !== ch) {
            canvas.width = cw;
            canvas.height = ch;
        }
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        const scrollX = scrollXRef.current;
        const ppt = pptRef.current;
        const segs = segsRef.current;
        // 背景 + 底边框（与标尺一致的深色面板）。
        const css = getComputedStyle(canvas);
        const panel = css.getPropertyValue("--bg-panel").trim() || "#1a2236";
        const border = css.getPropertyValue("--border-subtle").trim() || "#2a3a5c";
        const textMain = css.getPropertyValue("--text-primary").trim() || "#e8ecf4";
        const textMuted = css.getPropertyValue("--text-muted").trim() || "#556b94";
        const accent = css.getPropertyValue("--accent-primary").trim() || "#7c5cff";
        const accentRgb = hexToRgb(accent);
        ctx.fillStyle = panel;
        ctx.fillRect(0, 0, width, height);
        // 小节下拍参考线（只画 bar，beat 在 26px 行里只是噪声）。
        const startTick = Math.floor(scrollX / ppt);
        const endTick = Math.ceil((scrollX + width) / ppt);
        ctx.strokeStyle = rgba(accentRgb, 0.14);
        ctx.lineWidth = 1;
        for (const g of timeAxis.gridLinesInRange(startTick, endTick)) {
            if (!g.isBar)
                continue;
            const x = Math.round(g.tick * ppt - scrollX) + 0.5;
            ctx.beginPath();
            ctx.moveTo(x, 6);
            ctx.lineTo(x, height - 2);
            ctx.stroke();
        }
        if (segs.length === 0) {
            // 空态引导：一行灰字，告诉用户入口在哪。
            ctx.fillStyle = textMuted;
            ctx.font = "10px system-ui, sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(t("chordTrack.empty"), width / 2, height / 2);
            ctx.textAlign = "left";
        }
        else {
            // 和弦段：块 + 居中标签。x 裁剪到视口，1px 圆角太碎——直角块即可。
            ctx.font = "bold 11px system-ui, sans-serif";
            ctx.textBaseline = "middle";
            for (const s of segs) {
                const x0 = s.startTick * ppt - scrollX;
                const x1 = s.endTick * ppt - scrollX;
                if (x1 <= 0 || x0 >= width)
                    continue;
                const cx0 = Math.max(1, x0);
                const cx1 = Math.min(width - 1, x1);
                const w = cx1 - cx0;
                if (w < 2)
                    continue;
                // 块底：半透明主色；正在编辑的那段亮一档。
                ctx.fillStyle = rgba(accentRgb, 0.18);
                ctx.fillRect(cx0, 3, w, height - 8);
                // 标签居中；块太窄时左对齐溢出显示（不截断——乐谱标签宁溢勿缺）。
                ctx.fillStyle = textMain;
                const tw = ctx.measureText(s.label).width;
                const lx = w >= tw + 6 ? cx0 + (w - tw) / 2 : cx0 + 2;
                ctx.fillText(s.label, lx, height / 2);
            }
            // 调性标签：左端固定徽章（不随滚动 —— 是全曲属性，不是时间位置）。
            const badge = `${t("chordTrack.key")} ${keyRef.current.label}`;
            ctx.font = "9px system-ui, sans-serif";
            const bw = ctx.measureText(badge).width + 10;
            ctx.fillStyle = rgba(accentRgb, 0.32);
            ctx.fillRect(0, 3, bw, height - 8);
            ctx.fillStyle = accent;
            ctx.fillText(badge, 5, height / 2);
            // hover / 拖拽中的分段边界：accent 竖线（拖拽中实时跟随提交后的 endTick）。
            const hiIdx = dragIdxRef.current ?? hoverIdxRef.current;
            if (hiIdx !== null && hiIdx >= 0 && hiIdx + 1 < segs.length) {
                const x = Math.round(segs[hiIdx].endTick * ppt - scrollX) + 0.5;
                if (x >= 0 && x <= width) {
                    ctx.strokeStyle = dragIdxRef.current !== null ? accent : rgba(accentRgb, 0.6);
                    ctx.lineWidth = dragIdxRef.current !== null ? 2 : 1.5;
                    ctx.beginPath();
                    ctx.moveTo(x, 1);
                    ctx.lineTo(x, height - 1);
                    ctx.stroke();
                }
            }
        }
        // 播放头细线（整行高，跟随全局播放头）。
        const phx = playheadRef.current * ppt - scrollX;
        if (phx >= 0 && phx <= width) {
            ctx.strokeStyle = "rgba(232,236,244,0.7)";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(Math.round(phx) + 0.5, 2);
            ctx.lineTo(Math.round(phx) + 0.5, height - 1);
            ctx.stroke();
        }
        // 底边框收尾。
        ctx.strokeStyle = border;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, height - 0.5);
        ctx.lineTo(width, height - 0.5);
        ctx.stroke();
    }, [timeAxis, t]);
    drawRef.current = draw;
    // 内容/标尺变化 → 直接重绘（segments 也在内：rename/拖拽边界后立即反映）；滚动/缩放/播放头 → 订阅改 ref + rAF 合并。
    useEffect(() => { draw(); }, [draw, segments]);
    useEffect(() => {
        const unsubApp = useAppStore.subscribe((s) => {
            let changed = false;
            if (s.scrollX !== scrollXRef.current) {
                scrollXRef.current = s.scrollX;
                changed = true;
            }
            if (s.zoom !== zoomRef.current) {
                zoomRef.current = s.zoom;
                pptRef.current = PIXELS_PER_TICK * s.zoom;
                changed = true;
            }
            if (changed)
                requestRedraw();
        });
        const unsubProj = useProjectStore.subscribe((s) => {
            if (s.playheadTick !== playheadRef.current) {
                playheadRef.current = s.playheadTick;
                requestRedraw();
            }
        });
        return () => { unsubApp(); unsubProj(); cancelAnimationFrame(redrawRafRef.current); };
    }, [requestRedraw]);
    // 画布尺寸变化 → 重绘（ResizeObserver 比窗口事件准）。
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas)
            return;
        const observer = new ResizeObserver(() => drawRef.current());
        observer.observe(canvas);
        return () => observer.disconnect();
    }, []);
    // 命中测试：xLocal 附近（±4px）的相邻段共享边界 → 左段 index；否则 -1。
    const hitBoundary = useCallback((xLocal) => {
        const segs = segsRef.current;
        for (let i = 0; i + 1 < segs.length; i++) {
            const bx = segs[i].endTick * pptRef.current - scrollXRef.current;
            if (Math.abs(xLocal - bx) <= 4)
                return i;
        }
        return -1;
    }, []);
    // hover：边界附近 → col-resize + 高亮（离开/移开复位）。
    const handleMouseMove = useCallback((e) => {
        if (dragIdxRef.current !== null)
            return;
        const canvas = canvasRef.current;
        if (!canvas)
            return;
        const hit = hitBoundary(e.clientX - canvas.getBoundingClientRect().left);
        if (hit !== hoverIdxRef.current) {
            hoverIdxRef.current = hit;
            canvas.style.cursor = hit >= 0 ? "col-resize" : "";
            requestRedraw();
        }
    }, [hitBoundary, requestRedraw]);
    const handleMouseLeave = useCallback(() => {
        if (dragIdxRef.current !== null || hoverIdxRef.current === null)
            return;
        hoverIdxRef.current = null;
        const canvas = canvasRef.current;
        if (canvas)
            canvas.style.cursor = "";
        requestRedraw();
    }, [requestRedraw]);
    // 点击（左键）：边界 → 拖拽调整分段；段内 → 就地编辑输入框。
    const handleMouseDown = useCallback((e) => {
        if (e.button !== 0)
            return;
        const canvas = canvasRef.current;
        if (!canvas)
            return;
        const rect = canvas.getBoundingClientRect();
        // 先测边界（优先于段编辑命中；±4px 的手柄区）。
        const boundary = hitBoundary(e.clientX - rect.left);
        if (boundary >= 0) {
            dragIdxRef.current = boundary;
            hoverIdxRef.current = null;
            requestRedraw();
            const onMove = (ev) => {
                const c = canvasRef.current;
                if (!c)
                    return;
                const r = c.getBoundingClientRect();
                const tick = (ev.clientX - r.left + scrollXRef.current) / pptRef.current;
                useChordTrackStore.getState().moveBoundary(boundary, tick);
            };
            const onUp = () => {
                dragIdxRef.current = null;
                window.removeEventListener("mousemove", onMove);
                window.removeEventListener("mouseup", onUp);
                requestRedraw();
            };
            window.addEventListener("mousemove", onMove);
            window.addEventListener("mouseup", onUp);
            return;
        }
        const tick = (e.clientX - rect.left + scrollXRef.current) / pptRef.current;
        const segs = segsRef.current;
        let hit = -1;
        for (let i = 0; i < segs.length; i++) {
            const s = segs[i];
            if (tick >= s.startTick && tick < s.endTick) {
                hit = i;
                break;
            }
        }
        if (hit < 0)
            return;
        const s = segs[hit];
        const left = Math.max(1, s.startTick * pptRef.current - scrollXRef.current);
        const right = Math.min(rect.width - 1, s.endTick * pptRef.current - scrollXRef.current);
        setEditing({ index: hit, left, width: Math.max(48, right - left), value: s.label });
    }, [hitBoundary, requestRedraw]);
    const commitEdit = useCallback(() => {
        setEditing((ed) => {
            if (ed) {
                const label = ed.value;
                useChordTrackStore.getState().renameLabel(ed.index, label);
            }
            return null;
        });
    }, []);
    const ctxItems = [
        {
            label: t("chordTrack.copy"),
            disabled: segments.length === 0,
            onClick: () => {
                void navigator.clipboard
                    .writeText(segments.map((s) => s.label).join(" "))
                    .then(() => useAppStore.getState().showToast(t("chordTrack.copied"), "success"))
                    .catch(() => useAppStore.getState().showToast(t("chordTrack.copyFailed"), "error"));
            },
        },
        {
            label: t("chordTrack.clear"),
            disabled: segments.length === 0,
            danger: true,
            onClick: () => {
                useChordTrackStore.getState().clear();
                useAppStore.getState().showToast(t("chordTrack.cleared"), "info");
            },
        },
    ];
    return (_jsxs("div", { className: "chord-track", children: [_jsx("canvas", { ref: canvasRef, className: "chord-track-canvas", onMouseDown: handleMouseDown, onMouseMove: handleMouseMove, onMouseLeave: handleMouseLeave, onContextMenu: (e) => {
                    e.preventDefault();
                    setCtxMenu({ x: e.clientX, y: e.clientY });
                } }), editing && (_jsx("input", { className: "chord-track-edit", style: { left: editing.left, width: editing.width }, value: editing.value, autoFocus: true, onChange: (e) => setEditing((ed) => (ed ? { ...ed, value: e.target.value } : ed)), onBlur: commitEdit, onKeyDown: (e) => {
                    if (e.key === "Enter")
                        commitEdit();
                    else if (e.key === "Escape")
                        setEditing(null);
                } })), ctxMenu && _jsx(ContextMenu, { x: ctxMenu.x, y: ctxMenu.y, items: ctxItems, onClose: () => setCtxMenu(null) })] }));
}
