import { jsxs as _jsxs, jsx as _jsx } from "react/jsx-runtime";
import { useCallback, useEffect, useRef, useState } from "react";
import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import { useHistoryStore } from "../../store/history";
import { playSingleNote } from "../../lib/audio/previewNote";
import { resolveMelodyTrackId } from "../../lib/arrangement/autoArrange";
import { ArrangeDialog } from "./ArrangeDialog";
import { MixerConsole } from "./MixerConsole";
import "./VirtualPiano.css";
/** MIDI note number for C3 (中央 C 下八度的 C) */
const C3 = 48;
const OCTAVES = 2;
const WHITE_KEYS_PER_OCTAVE = 7;
const TOTAL_WHITE = OCTAVES * WHITE_KEYS_PER_OCTAVE; // 14 keys
/** 键盘 → MIDI 映射 */
const KEYBOARD_MAP_LOWER = {
    z: C3, s: C3 + 1, x: C3 + 2, d: C3 + 3, c: C3 + 4, v: C3 + 5, g: C3 + 6,
    b: C3 + 7, h: C3 + 8, n: C3 + 9, j: C3 + 10, m: C3 + 11,
};
const KEYBOARD_MAP_UPPER = {
    q: C3 + 12, "2": C3 + 13, w: C3 + 14, "3": C3 + 15, e: C3 + 16,
    r: C3 + 17, "5": C3 + 18, t: C3 + 19, "6": C3 + 20, y: C3 + 21,
    "7": C3 + 22, u: C3 + 23,
};
const KEYBOARD_MAP = { ...KEYBOARD_MAP_LOWER, ...KEYBOARD_MAP_UPPER };
/** 白键序号 → note number */
function whiteIndexToNote(idx) {
    const octave = Math.floor(idx / WHITE_KEYS_PER_OCTAVE);
    const step = idx % WHITE_KEYS_PER_OCTAVE;
    const whiteSteps = [0, 2, 4, 5, 7, 9, 11];
    return C3 + octave * 12 + whiteSteps[step];
}
export function VirtualPiano() {
    const [open, setOpen] = useState(false);
    const [activeNotes, setActiveNotes] = useState(new Set());
    const [arrangeTarget, setArrangeTarget] = useState(null);
    const [consoleOpen, setConsoleOpen] = useState(false);
    const pressedKeys = useRef(new Set());
    const inspectorTrackId = useAppStore((s) => s.inspectorTrackId);
    // —— 底部工作流快捷按钮 ——
    const toast = (msg, kind = "info") => useAppStore.getState().showToast(msg, kind);
    const openSongStudio = () => useAppStore.getState().toggleSongStudio();
    const openSuperWizard = () => window.dispatchEvent(new CustomEvent("utai:open-super-wizard"));
    const openSmartArrange = () => {
        const tid = resolveMelodyTrackId();
        if (!tid) {
            toast("请先选中或创建一条有音符的旋律轨", "error");
            return;
        }
        setArrangeTarget(tid);
    };
    // 注: 转MIDI / 和弦MIDI / 识别和弦 / 识别鼓点 已集成到轨道右键菜单 (TrackList)
    // 键盘事件 — 全局监听
    useEffect(() => {
        if (!open)
            return;
        const onDown = (e) => {
            const el = e.target;
            if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA"))
                return;
            // 修饰键组合 (Ctrl/Cmd/Alt) 属于应用快捷键或系统手势 —— 别把 Ctrl+Z/S/C/V… 误触发成琴键音。
            if (e.ctrlKey || e.metaKey || e.altKey)
                return;
            const k = e.key.toLowerCase();
            if (pressedKeys.current.has(k))
                return;
            const note = KEYBOARD_MAP[k];
            if (note !== undefined) {
                pressedKeys.current.add(k);
                setActiveNotes((s) => new Set([...s, note]));
                playSingleNote(note);
            }
        };
        const onUp = (e) => {
            const k = e.key.toLowerCase();
            pressedKeys.current.delete(k);
            const note = KEYBOARD_MAP[k];
            if (note !== undefined) {
                setActiveNotes((s) => {
                    const next = new Set(s);
                    next.delete(note);
                    return next;
                });
            }
        };
        window.addEventListener("keydown", onDown);
        window.addEventListener("keyup", onUp);
        return () => {
            window.removeEventListener("keydown", onDown);
            window.removeEventListener("keyup", onUp);
            // 收起/卸载时清掉挂起的按键与高亮 —— 否则按住键时合上钢琴，keyup 监听已移除，
            // pressedKeys 里的键会「卡死」（重开后按它不再发声），activeNotes 残留的高亮也不会消失。
            pressedKeys.current.clear();
            setActiveNotes(new Set());
        };
    }, [open]);
    const writeNoteToTrack = useCallback((note) => {
        const trackId = inspectorTrackId;
        if (!trackId || !inspectorTrackId)
            return;
        const track = useProjectStore.getState().tracks.find((t) => t.id === trackId);
        if (!track || (track.trackType !== "instrument" && track.trackType !== "vocal"))
            return;
        const { playheadTick, updateTrack } = useProjectStore.getState();
        const seg = {
            id: "vp-" + Date.now() + "-" + Math.random().toString(36).slice(2, 6),
            startTick: playheadTick,
            lengthTicks: 120,
            durationTicks: 120,
            content: { type: "notes", notes: [{ pitch: note, startTick: 0, lengthTicks: 120, velocity: 80 }] },
        };
        useHistoryStore.getState().beginTransaction();
        updateTrack(trackId, { segments: [...track.segments, seg] });
        useHistoryStore.getState().commitTransaction();
    }, [inspectorTrackId]);
    const onMouseDown = useCallback((note) => {
        setActiveNotes((s) => new Set([...s, note]));
        playSingleNote(note);
        writeNoteToTrack(note);
    }, [writeNoteToTrack]);
    const onMouseUp = useCallback((note) => {
        setActiveNotes((s) => {
            const next = new Set(s);
            next.delete(note);
            return next;
        });
    }, []);
    // 构建黑键列表 (白键之间, 跳过 E-F 和 B-C)
    const blackKeys = [];
    const whiteSteps = [0, 2, 4, 5, 7, 9, 11];
    for (let o = 0; o < OCTAVES; o++) {
        for (let w = 0; w < 7; w++) {
            if (w === 2 || w === 6)
                continue; // E/B 后无黑键
            const whiteNote = C3 + o * 12 + whiteSteps[w];
            const blackNote = whiteNote + 1;
            const leftPct = ((o * 7 + w + 1) / TOTAL_WHITE) * 100;
            blackKeys.push({ note: blackNote, leftPct });
        }
    }
    const whiteNoteNames = ["C", "D", "E", "F", "G", "A", "B"];
    return (_jsxs("div", { className: `vp-container ${open ? "vp-open" : ""}`, children: [_jsxs("div", { className: "vp-bar", children: [_jsxs("button", { className: "vp-toggle", onClick: () => setOpen((o) => !o), title: "\u865A\u62DF\u94A2\u7434 (2 \u516B\u5EA6 C3-C5)", children: ["\uD83C\uDFB9 ", open ? "收起钢琴" : "虚拟钢琴"] }), _jsx("span", { className: "vp-sep" }), _jsx("button", { className: "vp-quick", onClick: openSongStudio, title: "\u6253\u5F00\u6B4C\u66F2\u5236\u4F5C\u5DE5\u4F5C\u53F0", children: "\uD83C\uDFA7 \u6B4C\u66F2\u5236\u4F5C" }), _jsx("button", { className: "vp-quick", onClick: openSuperWizard, title: "\u9009\u6B4C + \u6A21\u677F \u2192 \u4E00\u952E\u751F\u6210\u539F\u521B\u6B4C\u66F2", children: "\uD83E\uDE84 \u8D85\u7EA7\u539F\u521B" }), _jsx("button", { className: "vp-quick", onClick: openSmartArrange, title: "AI \u81EA\u52A8\u7F16\u66F2 (\u9F13/\u8D1D\u65AF/\u94A2\u7434/\u94FA\u5E95)", children: "\u2728 \u667A\u80FD\u7F16\u66F2" }), _jsx("span", { className: "vp-sep" }), _jsx("button", { className: "vp-quick", onClick: () => setConsoleOpen(true), title: "\u63A7\u5236\u53F0: \u6240\u6709\u8F68\u9053\u97F3\u91CF/\u5E73\u8861/\u9759\u97F3\u72EC\u594F + \u603B\u8F93\u51FA", children: "\uD83C\uDF9B \u63A7\u5236\u53F0" })] }), open && (_jsxs("div", { className: "vp-keyboard", children: [_jsx("div", { className: "vp-white-row", children: Array.from({ length: TOTAL_WHITE }, (_, i) => {
                            const note = whiteIndexToNote(i);
                            const isActive = activeNotes.has(note);
                            const pc = note % 12;
                            const name = whiteNoteNames[pc === 0 ? 0 : pc === 2 ? 1 : pc === 4 ? 2 : pc === 5 ? 3 : pc === 7 ? 4 : pc === 9 ? 5 : 6];
                            return (_jsx("button", { className: `vp-white ${isActive ? "vp-active" : ""}`, onMouseDown: () => onMouseDown(note), onMouseUp: () => onMouseUp(note), onMouseLeave: () => onMouseUp(note), children: _jsxs("span", { className: "vp-white-label", children: [name, Math.floor(note / 12) - 1] }) }, i));
                        }) }), _jsx("div", { className: "vp-black-row", children: blackKeys.map(({ note, leftPct }) => {
                            const isActive = activeNotes.has(note);
                            return (_jsx("button", { className: `vp-black ${isActive ? "vp-active-black" : ""}`, style: { left: `${leftPct}%` }, onMouseDown: () => onMouseDown(note), onMouseUp: () => onMouseUp(note), onMouseLeave: () => onMouseUp(note) }, note));
                        }) }), _jsxs("div", { className: "vp-hint", children: ["\uD83D\uDCA1 \u952E\u76D8: ", _jsx("code", { children: "Z S X D C V G B H N J M" }), " (\u4F4E\u516B\u5EA6) \u00B7 ", _jsx("code", { children: "Q 2 W 3 E R 5 T 6 Y 7 U" }), " (\u9AD8\u516B\u5EA6) \u00B7 \u70B9\u7434\u952E \u2192 \u5199\u5165\u5F53\u524D\u9009\u4E2D\u8F68"] })] })), arrangeTarget && _jsx(ArrangeDialog, { trackId: arrangeTarget, onClose: () => setArrangeTarget(null) }), consoleOpen && _jsx(MixerConsole, { onClose: () => setConsoleOpen(false) })] }));
}
