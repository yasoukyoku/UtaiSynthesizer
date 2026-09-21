import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
/**
 * MidiEditor - 轻量 MIDI 编辑器（功能 A）
 *
 * 定位：多轨工作室的"符号层"下游——把生成的 MIDI 打开成钢琴卷帘窗，做基础编辑
 * （拖动改音高/位置、拖右缘改时值、双击空白添加、删除、量化），然后：
 *   1. 导出 MIDI（DAW 可直接打开）
 *   2. 渲染为音频（FluidSynth 音源 → WAV）并落回工程（MIDI→音频回流）
 *   3. 直接把音符落成工程的乐器轨
 *
 * 刻度空间：与工程一致 480 ppq。import_score_file 会把每轨 rebase 到首音符，
 * 因此绝对 tick = track.start_tick + note.tick；写回走 amt_write_edited_midi（同为 480 ppq）。
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open, save } from "@tauri-apps/plugin-dialog";
import "./MidiEditor.css";
const TICKS_PER_BEAT = 480;
const MIN_PITCH = 21;
const MAX_PITCH = 108;
const ROW_H = 12;
const NOTE_H = ROW_H - 1;
const PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
function pitchName(p) {
    const n = PITCH_NAMES[p % 12] ?? "C";
    return `${n}${Math.floor(p / 12) - 1}`;
}
function isBlackKey(p) {
    return [1, 3, 6, 8, 10].includes(p % 12);
}
/** 网格吸附（0 = 关闭）。 */
function snapTick(t, grid) {
    if (grid <= 0)
        return Math.max(0, Math.round(t));
    return Math.max(0, Math.round(t / grid) * grid);
}
const GRID_OPTIONS = [
    { label: "1/2 拍", value: TICKS_PER_BEAT / 2 },
    { label: "1/4 拍", value: TICKS_PER_BEAT / 4 },
    { label: "1/8 拍", value: TICKS_PER_BEAT / 8 },
    { label: "1/16 拍", value: TICKS_PER_BEAT / 16 },
    { label: "关闭", value: 0 },
];
export function MidiEditor({ midiPath, initialTitle, onClose, onDepositNotes, onDepositAudio, }) {
    const [tracks, setTracks] = useState([]);
    const [activeTrackId, setActiveTrackId] = useState("");
    const [bpm, setBpm] = useState(120);
    const [timeSig, setTimeSig] = useState([4, 4]);
    const [title, setTitle] = useState(initialTitle ?? "MIDI 编辑");
    const [pxPerBeat, setPxPerBeat] = useState(48);
    const [grid, setGrid] = useState(TICKS_PER_BEAT / 4);
    const [selectedIds, setSelectedIds] = useState(new Set());
    const [dragging, setDragging] = useState(false);
    const [busy, setBusy] = useState(false);
    const [status, setStatus] = useState(null);
    const [error, setError] = useState(null);
    const canvasRef = useRef(null);
    const dragRef = useRef(null);
    const pxPerTick = pxPerBeat / TICKS_PER_BEAT;
    const activeTrack = tracks.find((t) => t.id === activeTrackId) ?? null;
    const allNotes = useMemo(() => tracks.flatMap((t) => t.notes), [tracks]);
    // 纵向自适应：包住所有音符，最少跨 24 个半音（不下滑，符合"能不用滑动就不用"）
    const { lowPitch, highPitch } = useMemo(() => {
        let lo = MAX_PITCH;
        let hi = MIN_PITCH;
        for (const n of allNotes) {
            if (n.pitch < lo)
                lo = n.pitch;
            if (n.pitch > hi)
                hi = n.pitch;
        }
        if (allNotes.length === 0)
            return { lowPitch: 48, highPitch: 72 };
        lo -= 3;
        hi += 3;
        while (hi - lo < 24) {
            lo -= 1;
            hi += 1;
        }
        return {
            lowPitch: Math.max(MIN_PITCH, lo),
            highPitch: Math.min(MAX_PITCH, hi),
        };
    }, [allNotes]);
    const rowCount = highPitch - lowPitch + 1;
    const contentHeight = rowCount * ROW_H;
    const totalTicks = useMemo(() => {
        let maxEnd = 0;
        for (const n of allNotes)
            maxEnd = Math.max(maxEnd, n.tick + n.duration);
        return Math.max(TICKS_PER_BEAT * 8, maxEnd + TICKS_PER_BEAT);
    }, [allNotes]);
    const contentWidth = Math.max(320, Math.ceil(totalTicks * pxPerTick));
    // ── 载入 ────────────────────────────────────────────────────────────────
    async function loadFrom(path, fallbackName) {
        setBusy(true);
        setError(null);
        setStatus(null);
        try {
            const score = await invoke("import_score_file", { path });
            const trks = (score.tracks ?? [])
                .filter((t) => Array.isArray(t.notes) && t.notes.length > 0)
                .map((t) => ({
                id: crypto.randomUUID(),
                name: t.name || fallbackName || "Track",
                program: null,
                channel: null,
                notes: t.notes.map((n) => ({
                    id: crypto.randomUUID(),
                    tick: Math.max(0, (t.start_tick ?? 0) + n.tick),
                    duration: Math.max(1, n.duration),
                    pitch: Math.min(MAX_PITCH, Math.max(MIN_PITCH, n.pitch)),
                    velocity: 100,
                    ...(n.lyric ? { lyric: n.lyric } : {}),
                })),
            }));
            if (trks.length === 0) {
                setError("该文件中没有可用音符");
                return;
            }
            setTracks(trks);
            setActiveTrackId(trks[0].id);
            setSelectedIds(new Set());
            if (score.bpm && Number.isFinite(score.bpm) && score.bpm > 0)
                setBpm(score.bpm);
            if (score.time_sig)
                setTimeSig(score.time_sig);
            if (fallbackName)
                setTitle(fallbackName);
            const total = trks.reduce((s, t) => s + t.notes.length, 0);
            setStatus(`已载入 ${trks.length} 轨 / ${total} 个音符`);
        }
        catch (e) {
            setError(`载入失败：${e instanceof Error ? e.message : String(e)}`);
        }
        finally {
            setBusy(false);
        }
    }
    useEffect(() => {
        if (midiPath)
            void loadFrom(midiPath, initialTitle);
        // 初次挂载按传入路径载入一次
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);
    async function pickAndLoad() {
        const picked = await open({
            title: "打开 MIDI",
            multiple: false,
            filters: [{ name: "MIDI", extensions: ["mid", "midi", "ustx", "ust"] }],
        });
        if (typeof picked === "string" && picked) {
            const base = picked.replace(/^.*[\\/]/, "").replace(/\.[^.]+$/, "");
            await loadFrom(picked, base);
        }
    }
    // ── 编辑操作 ────────────────────────────────────────────────────────────
    function mutateActive(fn) {
        setTracks((prev) => prev.map((t) => (t.id === activeTrackId ? { ...t, notes: fn(t.notes) } : t)));
    }
    function xyToTickPitch(x, y) {
        return {
            tick: x / pxPerTick,
            pitch: highPitch - Math.floor(y / ROW_H),
        };
    }
    function noteRect(n) {
        return {
            x: n.tick * pxPerTick,
            y: (highPitch - n.pitch) * ROW_H,
            w: Math.max(2, n.duration * pxPerTick),
            h: NOTE_H,
        };
    }
    function hitTest(x, y) {
        if (!activeTrack)
            return null;
        const notes = activeTrack.notes;
        for (let i = notes.length - 1; i >= 0; i--) {
            const n = notes[i];
            const r = noteRect(n);
            if (x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h)
                return n;
        }
        return null;
    }
    function onCanvasMouseDown(e) {
        const x = e.nativeEvent.offsetX;
        const y = e.nativeEvent.offsetY;
        const hit = hitTest(x, y);
        if (!hit) {
            if (!e.ctrlKey && !e.metaKey)
                setSelectedIds(new Set());
            dragRef.current = null;
            return;
        }
        let selection = new Set(selectedIds);
        if (e.ctrlKey || e.metaKey) {
            if (selection.has(hit.id))
                selection.delete(hit.id);
            else
                selection.add(hit.id);
        }
        else if (!selection.has(hit.id)) {
            selection = new Set([hit.id]);
        }
        setSelectedIds(selection);
        const r = noteRect(hit);
        const mode = x >= r.x + r.w - 6 ? "resize" : "move";
        const orig = {};
        if (activeTrack) {
            for (const n of activeTrack.notes) {
                if (selection.has(n.id)) {
                    orig[n.id] = { tick: n.tick, duration: n.duration, pitch: n.pitch };
                }
            }
        }
        dragRef.current = { mode, startX: x, startY: y, ids: [...selection], orig };
        setDragging(true);
    }
    function onCanvasDoubleClick(e) {
        const x = e.nativeEvent.offsetX;
        const y = e.nativeEvent.offsetY;
        if (hitTest(x, y))
            return;
        const { tick, pitch } = xyToTickPitch(x, y);
        if (pitch < MIN_PITCH || pitch > MAX_PITCH)
            return;
        const step = grid > 0 ? grid : TICKS_PER_BEAT / 4;
        const note = {
            id: crypto.randomUUID(),
            tick: snapTick(tick, step),
            duration: step,
            pitch,
            velocity: 100,
        };
        mutateActive((notes) => [...notes, note]);
        setSelectedIds(new Set([note.id]));
    }
    // 拖拽：window 级监听，脱离画布也能继续
    useEffect(() => {
        if (!dragging)
            return;
        const onMove = (e) => {
            const drag = dragRef.current;
            if (!drag)
                return;
            const canvas = canvasRef.current;
            if (!canvas)
                return;
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            if (drag.mode === "move") {
                const dtick = Math.round((x - drag.startX) / pxPerTick);
                const dpitch = -Math.round((y - drag.startY) / ROW_H);
                mutateActive((notes) => notes.map((n) => {
                    const o = drag.orig[n.id];
                    if (!o)
                        return n;
                    return {
                        ...n,
                        tick: snapTick(o.tick + dtick, grid),
                        pitch: Math.min(MAX_PITCH, Math.max(MIN_PITCH, o.pitch + dpitch)),
                    };
                }));
            }
            else {
                const dtick = Math.round((x - drag.startX) / pxPerTick);
                mutateActive((notes) => notes.map((n) => {
                    const o = drag.orig[n.id];
                    if (!o)
                        return n;
                    const end = snapTick(o.tick + o.duration + dtick, grid);
                    const minDur = grid > 0 ? grid : 30;
                    return { ...n, duration: Math.max(minDur, end - n.tick) };
                }));
            }
        };
        const onUp = () => {
            dragRef.current = null;
            setDragging(false);
        };
        window.addEventListener("mousemove", onMove);
        window.addEventListener("mouseup", onUp);
        return () => {
            window.removeEventListener("mousemove", onMove);
            window.removeEventListener("mouseup", onUp);
        };
        // grid/pxPerTick 参与闭包，编辑期间变化极少；随 dragging 重挂即可
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [dragging, grid, pxPerTick, activeTrackId]);
    function deleteSelected() {
        if (selectedIds.size === 0)
            return;
        mutateActive((notes) => notes.filter((n) => !selectedIds.has(n.id)));
        setSelectedIds(new Set());
    }
    function quantize() {
        const step = grid > 0 ? grid : TICKS_PER_BEAT / 4;
        const ids = selectedIds;
        mutateActive((notes) => notes
            .map((n) => ids.size === 0 || ids.has(n.id)
            ? { ...n, tick: snapTick(n.tick, step), duration: Math.max(step, Math.round(n.duration / step) * step) }
            : n)
            .sort((a, b) => a.tick - b.tick));
    }
    // ── 渲染到 canvas ───────────────────────────────────────────────────────
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas)
            return;
        const ctx = canvas.getContext("2d");
        if (!ctx)
            return;
        const dpr = window.devicePixelRatio || 1;
        canvas.width = Math.ceil(contentWidth * dpr);
        canvas.height = Math.ceil(contentHeight * dpr);
        canvas.style.width = `${contentWidth}px`;
        canvas.style.height = `${contentHeight}px`;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, contentWidth, contentHeight);
        ctx.fillStyle = "#14161b";
        ctx.fillRect(0, 0, contentWidth, contentHeight);
        // 行底色（黑键行更深）
        for (let i = 0; i < rowCount; i++) {
            const pitch = highPitch - i;
            ctx.fillStyle = isBlackKey(pitch) ? "#191c22" : "#1d2027";
            ctx.fillRect(0, i * ROW_H, contentWidth, ROW_H);
        }
        // 横向网格（拍线 + 小节线）
        const beats = Math.ceil(totalTicks / TICKS_PER_BEAT);
        for (let b = 0; b <= beats; b++) {
            const x = Math.round(b * TICKS_PER_BEAT * pxPerTick) + 0.5;
            const isBar = b % timeSig[0] === 0;
            ctx.strokeStyle = isBar ? "#3a4050" : "#262b34";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, contentHeight);
            ctx.stroke();
        }
        // 行分隔线
        ctx.strokeStyle = "#20242b";
        ctx.lineWidth = 1;
        for (let i = 0; i <= rowCount; i++) {
            const y = i * ROW_H + 0.5;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(contentWidth, y);
            ctx.stroke();
        }
        // 音符
        if (activeTrack) {
            for (const n of activeTrack.notes) {
                const r = noteRect(n);
                const selected = selectedIds.has(n.id);
                ctx.fillStyle = selected ? "#8b9bff" : "#5a6af0";
                ctx.fillRect(r.x, r.y, r.w, r.h);
                ctx.strokeStyle = selected ? "#c9d2ff" : "#3d4bb5";
                ctx.lineWidth = 1;
                ctx.strokeRect(r.x + 0.5, r.y + 0.5, r.w - 1, r.h - 1);
                // 右缘把柄（可拖拽改时值）
                if (selected || r.w > 14) {
                    ctx.fillStyle = selected ? "#e6e8ee" : "#40499a";
                    ctx.fillRect(r.x + r.w - 3, r.y + 2, 2, r.h - 4);
                }
            }
        }
    }, [activeTrack, selectedIds, contentWidth, contentHeight, rowCount, highPitch, totalTicks, pxPerTick, timeSig]);
    // ── 导出 / 回流 ─────────────────────────────────────────────────────────
    function buildTrackPayload() {
        return tracks
            .filter((t) => t.notes.length > 0)
            .map((t) => ({
            name: t.name,
            program: t.program,
            channel: t.channel,
            notes: t.notes.map((n) => ({
                tick: Math.round(n.tick),
                duration: Math.max(1, Math.round(n.duration)),
                pitch: n.pitch,
                velocity: n.velocity,
                ...(n.lyric ? { lyric: n.lyric } : {}),
            })),
        }));
    }
    async function exportMidi() {
        if (tracks.length === 0)
            return;
        setBusy(true);
        setError(null);
        setStatus(null);
        try {
            const out = await save({
                title: "导出 MIDI",
                defaultPath: `${title.replace(/[\\/:*?"<>|]/g, "_")}.mid`,
                filters: [{ name: "MIDI", extensions: ["mid"] }],
            });
            if (!out)
                return;
            const path = out.toLowerCase().endsWith(".mid") || out.toLowerCase().endsWith(".midi") ? out : `${out}.mid`;
            await invoke("amt_write_edited_midi", {
                outPath: path,
                bpm,
                tracks: buildTrackPayload(),
            });
            setStatus(`已导出 MIDI：${path}`);
        }
        catch (e) {
            setError(`导出失败：${e instanceof Error ? e.message : String(e)}`);
        }
        finally {
            setBusy(false);
        }
    }
    async function renderAudio() {
        if (tracks.length === 0)
            return;
        setBusy(true);
        setError(null);
        setStatus(null);
        try {
            const out = await save({
                title: "渲染为音频",
                defaultPath: `${title.replace(/[\\/:*?"<>|]/g, "_")}.wav`,
                filters: [{ name: "WAV", extensions: ["wav"] }],
            });
            if (!out)
                return;
            const wavPath = out.toLowerCase().endsWith(".wav") ? out : `${out}.wav`;
            const sep = Math.max(wavPath.lastIndexOf("/"), wavPath.lastIndexOf("\\"));
            const dir = sep >= 0 ? wavPath.slice(0, sep) : ".";
            const tempMidi = `${dir}/_midi_editor_snapshot_${Date.now()}.mid`;
            await invoke("amt_write_edited_midi", {
                outPath: tempMidi,
                bpm,
                tracks: buildTrackPayload(),
            });
            const rendered = await invoke("amt_export_midi_audio", {
                midiPath: tempMidi,
                outPath: wavPath,
                preset: "hq",
            });
            const durationSec = Math.max(1, (totalTicks / TICKS_PER_BEAT) * (60 / Math.max(20, bpm)));
            onDepositAudio(rendered, title, durationSec);
            setStatus(`已渲染并落到工程：${rendered}`);
        }
        catch (e) {
            const msg = e instanceof Error ? e.message : String(e);
            setError(msg.includes("EXPORT_FLUIDSYNTH_NOT_FOUND") || msg.includes("EXPORT_SOUNDFONT_NOT_FOUND")
                ? "渲染需要 FluidSynth 与音源（SoundFont），请先到「资源管理 → 音频渲染」下载。"
                : `渲染失败：${msg}`);
        }
        finally {
            setBusy(false);
        }
    }
    const noteCount = activeTrack?.notes.length ?? 0;
    return (_jsx("div", { className: "mid-editor-overlay", onClick: onClose, children: _jsxs("div", { className: "mid-editor-dialog", onClick: (e) => e.stopPropagation(), children: [_jsxs("div", { className: "mid-editor-header", children: [_jsx("div", { className: "mid-editor-title", children: "\uD83C\uDFB9 MIDI \u7F16\u8F91\u5668" }), _jsx("div", { className: "mid-editor-subtitle", children: "\u94A2\u7434\u5377\u5E18\u7A97 \u00B7 \u62D6\u52A8\u6539\u97F3\u9AD8/\u4F4D\u7F6E \u00B7 \u62D6\u53F3\u7F18\u6539\u65F6\u503C \u00B7 \u53CC\u51FB\u7A7A\u767D\u6DFB\u52A0" }), _jsx("button", { className: "mid-editor-close", onClick: onClose, children: "\u2715" })] }), _jsxs("div", { className: "mid-editor-toolbar", children: [_jsx("button", { className: "mid-btn mid-btn-ghost", onClick: pickAndLoad, disabled: busy, children: "\uD83D\uDCC2 \u8F7D\u5165 MIDI" }), _jsx("div", { className: "mid-editor-sep" }), _jsx("label", { className: "mid-inline-label", children: "BPM" }), _jsx("input", { className: "mid-input mid-input-num", type: "number", min: 20, max: 400, value: bpm, onChange: (e) => setBpm(Number(e.target.value) || 120) }), _jsx("label", { className: "mid-inline-label", children: "\u7F51\u683C" }), _jsx("select", { className: "mid-select", value: grid, onChange: (e) => setGrid(Number(e.target.value)), children: GRID_OPTIONS.map((g) => (_jsx("option", { value: g.value, children: g.label }, g.label))) }), _jsx("div", { className: "mid-editor-sep" }), _jsx("button", { className: "mid-btn mid-btn-ghost", onClick: () => setPxPerBeat((v) => Math.max(12, Math.round(v / 1.3))), title: "\u7F29\u5C0F", children: "\u2796" }), _jsx("button", { className: "mid-btn mid-btn-ghost", onClick: () => setPxPerBeat((v) => Math.min(320, Math.round(v * 1.3))), title: "\u653E\u5927", children: "\u2795" }), _jsx("div", { className: "mid-editor-sep" }), _jsx("button", { className: "mid-btn mid-btn-ghost", onClick: quantize, disabled: busy, title: "\u6309\u7F51\u683C\u5BF9\u9F50\uFF08\u9009\u4E2D\u4F18\u5148\uFF0C\u672A\u9009\u5219\u5168\u90E8\uFF09", children: "\uD83C\uDFAF \u91CF\u5316" }), _jsxs("button", { className: "mid-btn mid-btn-danger", onClick: deleteSelected, disabled: busy || selectedIds.size === 0, children: ["\uD83D\uDDD1 \u5220\u9664", selectedIds.size > 0 ? `(${selectedIds.size})` : ""] }), tracks.length > 1 && (_jsxs(_Fragment, { children: [_jsx("div", { className: "mid-editor-sep" }), _jsx("select", { className: "mid-select", value: activeTrackId, onChange: (e) => { setActiveTrackId(e.target.value); setSelectedIds(new Set()); }, children: tracks.map((t) => (_jsxs("option", { value: t.id, children: [t.name, "\uFF08", t.notes.length, "\uFF09"] }, t.id))) })] })), _jsx("div", { className: "mid-editor-spacer" }), _jsxs("span", { className: "mid-editor-count", children: [noteCount, " \u97F3\u7B26"] })] }), tracks.length === 0 ? (_jsxs("div", { className: "mid-editor-empty", children: [_jsx("div", { className: "mid-editor-empty-icon", children: "\uD83C\uDFBC" }), _jsx("div", { className: "mid-editor-empty-text", children: "\u8F7D\u5165\u4E00\u4E2A MIDI \u6587\u4EF6\u5F00\u59CB\u7F16\u8F91" }), _jsx("button", { className: "mid-btn mid-btn-primary", onClick: pickAndLoad, disabled: busy, children: "\uD83D\uDCC2 \u9009\u62E9 MIDI \u6587\u4EF6" })] })) : (_jsxs("div", { className: "mid-editor-roll", children: [_jsx("div", { className: "mid-editor-keys", style: { height: contentHeight }, children: Array.from({ length: rowCount }, (_, i) => {
                                const pitch = highPitch - i;
                                const black = isBlackKey(pitch);
                                const showLabel = pitch % 12 === 0;
                                return (_jsx("div", { className: `mid-editor-key ${black ? "black" : ""} ${showLabel ? "labeled" : ""}`, style: { height: ROW_H }, children: showLabel ? pitchName(pitch) : "" }, pitch));
                            }) }), _jsx("div", { className: "mid-editor-canvas-wrap", children: _jsx("canvas", { ref: canvasRef, className: "mid-editor-canvas", style: { cursor: dragging ? "grabbing" : "crosshair" }, onMouseDown: onCanvasMouseDown, onDoubleClick: onCanvasDoubleClick }) })] })), error && _jsx("div", { className: "mid-editor-error", children: error }), status && _jsx("div", { className: "mid-editor-status", children: status }), _jsxs("div", { className: "mid-editor-footer", children: [_jsx("button", { className: "mid-btn mid-btn-ghost", onClick: onClose, disabled: busy, children: "\u5173\u95ED" }), _jsx("div", { className: "mid-editor-spacer" }), _jsx("button", { className: "mid-btn mid-btn-ghost", onClick: () => activeTrack && onDepositNotes(activeTrack.name || title, activeTrack.notes, bpm), disabled: busy || !activeTrack || activeTrack.notes.length === 0, title: "\u628A\u5F53\u524D\u8F68\u97F3\u7B26\u843D\u6210\u5DE5\u7A0B\u7684\u4E50\u5668\u8F68", children: "\uD83C\uDFB9 \u53D1\u9001\u97F3\u7B26\u5230\u8F68\u9053" }), _jsx("button", { className: "mid-btn mid-btn-ghost", onClick: exportMidi, disabled: busy || tracks.length === 0, children: "\uD83D\uDCBE \u5BFC\u51FA MIDI" }), _jsx("button", { className: "mid-btn mid-btn-primary", onClick: renderAudio, disabled: busy || tracks.length === 0, children: "\uD83D\uDD0A \u6E32\u67D3\u97F3\u9891\u5E76\u56DE\u6D41" })] }), busy && _jsx("div", { className: "mid-editor-working", children: "\u5904\u7406\u4E2D\u2026" })] }) }));
}
