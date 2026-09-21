import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useState, useCallback, useRef, useEffect } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";
import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import { useSoundfontStore } from "../../store/soundfont";
import { useHistoryStore } from "../../store/history";
import { useVoiceModelStore } from "../../store/voice-models";
import { useRecording } from "../../lib/audio/recorder";
import { VOCAL_LANGUAGES, langById } from "../../lib/vocal/languages";
import { VolumeFader, formatDb, formatPan } from "../common/VolumeFader";
import { OutLevelMeter } from "../common/OutLevelMeter";
import { updateTrackVolume, updateTrackPan, updateTrackAudibility } from "../../lib/audio/playback";
import { FADER_MIN_DB, FADER_MAX_DB } from "../../lib/constants";
import { trackTypeCssVar } from "../../lib/trackColors";
import "./TrackInspectorPanel.css";
/** Studio Pro 风格底部 docked Inspector 面板
 *  v3: 删掉顶部 AI 按钮组(与右键菜单/底部栏重复) 与假"输入路由"行;
 *      录音按钮放大醒目; 音源/人物声集中在顶部一行;
 *      主区 = 平衡旋钮 + 竖直音量推子(带 dB 刻度) + 实时电平表 + 插入槽位/发送;
 *      全局效果总线移入"🎛 控制台"(MixerConsole 全轨道混音台)。
 */
const PLUGIN_CATALOG = [
    { id: "eq", name: "EQ 均衡器", icon: "🎚️" },
    { id: "compressor", name: "压缩器", icon: "📦" },
    { id: "reverb", name: "混响", icon: "🌊" },
    { id: "delay", name: "延迟", icon: "⏱️" },
    { id: "chorus", name: "合唱", icon: "🎭" },
    { id: "limiter", name: "限制器", icon: "🔒" },
];
const loadInspectorHeight = () => {
    try {
        return parseInt(localStorage.getItem("utai.inspectorHeight") || "280", 10) || 280;
    }
    catch {
        return 280;
    }
};
export function TrackInspectorPanel() {
    const tracks = useProjectStore((s) => s.tracks);
    const inspectorTrackId = useAppStore((s) => s.inspectorTrackId);
    const closeInspector = useAppStore((s) => s.closeInspector);
    const openInspector = useAppStore((s) => s.openInspector);
    const updateTrack = useProjectStore((s) => s.updateTrack);
    const setTrackFxSend = useProjectStore((s) => s.setTrackFxSend);
    const setTrackSoundfont = useProjectStore((s) => s.setTrackSoundfont);
    const refreshFonts = useSoundfontStore((s) => s.refresh);
    const fonts = useSoundfontStore((s) => s.fonts);
    // 录音态 + 歌手/语言 (从轨道头移入参数面板)
    const recording = useRecording((s) => (inspectorTrackId ? s.recordingTrackId === inspectorTrackId : false));
    const voiceModels = useVoiceModelStore((s) => s.models);
    const setVocalParams = useProjectStore((s) => s.setVocalParams);
    const [insertOpen, setInsertOpen] = useState(false);
    const [activeSlot, setActiveSlot] = useState(0); // -1 = 「添加新插槽」追加模式
    const [panelHeight, setPanelHeight] = useState(loadInspectorHeight);
    const dockRef = useRef(null);
    const handleRef = useRef(null);
    const targetHRef = useRef(panelHeight);
    const rafRef = useRef(null);
    const draggingRef = useRef(false);
    const startYRef = useRef(0);
    const startHRef = useRef(0);
    const activePointerIdRef = useRef(null);
    // Keep DOM in sync with ref during normal (non-drag) renders
    useEffect(() => { targetHRef.current = panelHeight; }, [panelHeight]);
    // Lazily refresh soundfonts ONCE when they're empty. Must be in useEffect, NOT render body —
    // calling zustand setters during render triggers React's "Cannot update while rendering" error.
    useEffect(() => {
        if (fonts.length === 0)
            refreshFonts().catch(() => { });
    }, [fonts.length, refreshFonts]);
    const track = tracks.find((t) => t.id === inspectorTrackId) ?? null;
    if (!track)
        return null;
    const currentFont = track.soundfont
        ? fonts.find((f) => f.id === track.soundfont?.fontId)
        : null;
    // 插件槽位数据：从 Track.pluginSlots 读（动态数量）。
    // 旧工程存的是 6 个 null 的占位数组, 这里过滤成纯字符串列表 → 槽位随插入自动增长。
    const plugins = (track.pluginSlots ?? []).filter((s) => typeof s === "string");
    const updatePlugins = (slots) => {
        useHistoryStore.getState().beginTransaction();
        updateTrack(track.id, { pluginSlots: slots });
        useHistoryStore.getState().commitTransaction();
    };
    // activeSlot >= 0 → 替换该槽位; activeSlot === -1 → 追加一个新插槽
    const onInsertPlugin = (pluginId) => {
        const next = [...plugins];
        if (activeSlot >= 0)
            next[activeSlot] = pluginId;
        else
            next.push(pluginId);
        updatePlugins(next);
        setInsertOpen(false);
    };
    // 动态槽位: 清除 = 直接删掉这一格 (后面的插槽前移)
    const onClearSlot = (slotIdx) => {
        const next = [...plugins];
        next.splice(slotIdx, 1);
        updatePlugins(next);
    };
    // 音量/平衡: store + 播放引擎双写 (与轨道头推子完全一致, 播放中拖动即时生效)
    // 历史事务由 VolumeFader 的 onGestureStart/End 统一包裹, 这里不再嵌套 begin/commit
    // (旧代码每次 mousemove 各开一层事务, 与手势外层事务嵌套错配, 撤销历史被刷屏)
    const updateVolume = useCallback((v) => {
        updateTrack(track.id, { volumeDb: v });
        updateTrackVolume(track.id, v);
    }, [track.id, updateTrack]);
    const updatePan = useCallback((v) => {
        updateTrack(track.id, { pan: v });
        updateTrackPan(track.id, v);
    }, [track.id, updateTrack]);
    const toggleMute = () => {
        useHistoryStore.getState().beginTransaction();
        updateTrack(track.id, { muted: !track.muted });
        useHistoryStore.getState().commitTransaction();
        // 播放中立即反映静音状态 (与轨道头 M/S 按钮一致)
        updateTrackAudibility(useProjectStore.getState().tracks);
    };
    const toggleSolo = () => {
        useHistoryStore.getState().beginTransaction();
        updateTrack(track.id, { solo: !track.solo });
        useHistoryStore.getState().commitTransaction();
        updateTrackAudibility(useProjectStore.getState().tracks);
    };
    const onFontChange = (fontId) => {
        const font = fonts.find((f) => f.id === fontId);
        if (!font) {
            setTrackSoundfont(track.id, undefined);
            return;
        }
        const firstPreset = font.presets[0];
        useHistoryStore.getState().beginTransaction();
        setTrackSoundfont(track.id, {
            fontId: font.id,
            presetId: firstPreset?.id ?? "0:0",
            presetName: firstPreset?.name ?? font.name,
        });
        useHistoryStore.getState().commitTransaction();
    };
    // 旋钮指针角度: pan∈[-1,1] → 指针从左(-90°)经上(0°=居中)到右(+90°)。
    // 旧公式 -90*(1-pan) 把范围压在 -180°..0°, 居中时指左、最右时朝上, 视觉完全错位。
    const panDeg = track.pan * 90;
    // 平衡旋钮: 按住左右拖动调节 (旧版只有一个点击回中, 鼠标拖不动)
    // 灵敏度 1px = 0.01 pan; 单击不动 = 回正中; 拖动期间只开一次撤销事务
    const onPanPointerDown = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        const startX = e.clientX;
        const startPan = track.pan;
        let moved = false;
        useHistoryStore.getState().beginTransaction();
        const onMove = (ev) => {
            const dx = ev.clientX - startX;
            if (!moved && Math.abs(dx) < 3)
                return;
            moved = true;
            const next = Math.max(-1, Math.min(1, Math.round((startPan + dx / 100) * 100) / 100));
            updatePan(next);
        };
        const onUp = () => {
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", onUp);
            if (!moved)
                updatePan(0); // 单击 = 回到正中 (保留原交互)
            useHistoryStore.getState().commitTransaction();
        };
        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onUp);
    }, [track.pan, updatePan]);
    // ── Height resize drag — Pointer Events + rAF direct DOM write ──
    // 关键点: mousemove 只更新 ref, rAF 里直接写 dock.style.height,
    // 全程不触发 React re-render, 只在 pointerup 时 commit 到 state.
    const flushRaf = useCallback(() => {
        if (!rafRef.current)
            return;
        rafRef.current = null;
        const dock = dockRef.current;
        if (dock)
            dock.style.height = `${targetHRef.current}px`;
    }, []);
    const onResizePointerDown = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        const el = handleRef.current;
        if (el && el.setPointerCapture) {
            try {
                el.setPointerCapture(e.pointerId);
            }
            catch { /* ignore */ }
        }
        activePointerIdRef.current = e.pointerId;
        draggingRef.current = true;
        startYRef.current = e.clientY;
        startHRef.current = targetHRef.current;
        document.body.style.cursor = "ns-resize";
        document.body.style.userSelect = "none";
        document.body.style.touchAction = "none";
        el?.classList.add("is-dragging");
        const onMove = (ev) => {
            if (!draggingRef.current)
                return;
            if (activePointerIdRef.current !== null && ev.pointerId !== activePointerIdRef.current)
                return;
            const dy = startYRef.current - ev.clientY;
            const next = Math.max(180, Math.min(560, startHRef.current + dy));
            targetHRef.current = next;
            if (!rafRef.current)
                rafRef.current = requestAnimationFrame(flushRaf);
        };
        const finish = () => {
            if (!draggingRef.current)
                return;
            draggingRef.current = false;
            activePointerIdRef.current = null;
            // Cancel any pending rAF and commit last value
            if (rafRef.current) {
                cancelAnimationFrame(rafRef.current);
                rafRef.current = null;
            }
            flushRaf();
            document.body.style.cursor = "";
            document.body.style.userSelect = "";
            document.body.style.touchAction = "";
            handleRef.current?.classList.remove("is-dragging");
            // Persist to localStorage + commit state (only ONCE at end)
            const final = Math.round(targetHRef.current);
            localStorage.setItem("utai.inspectorHeight", String(final));
            setPanelHeight((prev) => (prev !== final ? final : prev));
        };
        const onUp = (ev) => {
            if (activePointerIdRef.current !== null && ev.pointerId !== activePointerIdRef.current)
                return;
            const el2 = handleRef.current;
            if (el2 && el2.releasePointerCapture) {
                try {
                    el2.releasePointerCapture(ev.pointerId);
                }
                catch { /* ignore */ }
            }
            finish();
        };
        const onCancel = () => finish();
        // pointer capture ensures we get move/up even when pointer leaves the handle
        el?.addEventListener("pointermove", onMove);
        el?.addEventListener("pointerup", onUp);
        el?.addEventListener("pointercancel", onCancel);
        // Also bind on window as safety net (capture already handles it, but belt-and-suspenders)
        window.addEventListener("pointerup", finish);
        window.addEventListener("blur", finish);
        // Cleanup is tied to pointerup/cancel — but also protect against unmount mid-drag
        handleRef.current.__cleanupResize = () => {
            el?.removeEventListener("pointermove", onMove);
            el?.removeEventListener("pointerup", onUp);
            el?.removeEventListener("pointercancel", onCancel);
            window.removeEventListener("pointerup", finish);
            window.removeEventListener("blur", finish);
        };
    }, [flushRaf]);
    // Close insert popover when clicking outside
    useEffect(() => {
        if (!insertOpen)
            return;
        const onDocClick = () => setInsertOpen(false);
        const t = setTimeout(() => document.addEventListener("click", onDocClick), 0);
        return () => { clearTimeout(t); document.removeEventListener("click", onDocClick); };
    }, [insertOpen]);
    return (_jsxs("aside", { ref: dockRef, className: "inspector-dock", role: "dialog", "aria-label": "\u8F68\u9053\u5C5E\u6027", style: { height: panelHeight }, children: [_jsx("div", { ref: handleRef, className: "inspector-resize-handle", onPointerDown: onResizePointerDown, title: "\u62D6\u52A8\u8C03\u6574\u9AD8\u5EA6" }), _jsxs("div", { className: "inspector-header", children: [_jsx("button", { className: "inspector-close", onClick: closeInspector, title: "\u5173\u95ED", children: "\u2715" }), _jsx("span", { className: "inspector-track-name", title: track.name, children: track.name }), _jsx("div", { className: "inspector-header-spacer" })] }), _jsxs("div", { className: "inspector-rec-row", children: [_jsx("button", { className: `inspector-rec-btn inspector-rec-big ${recording ? "recording" : ""}`, onClick: () => { void useRecording.getState().toggle(track.id); }, title: recording ? "停止录音" : "开始在此轨道录音", children: recording ? "■ 停止录音" : "● 录音" }), _jsx("button", { className: `inspector-state-btn inspector-state-big ${track.muted ? "active-mute" : ""}`, onClick: toggleMute, title: "\u9759\u97F3\u8FD9\u6761\u8F68\u9053", children: "\u9759\u97F3" }), _jsx("button", { className: `inspector-state-btn inspector-state-big ${track.solo ? "active-solo" : ""}`, onClick: toggleSolo, title: "\u72EC\u594F \u2014 \u53EA\u542C\u8FD9\u6761\u8F68\u9053", children: "\u72EC\u594F" }), _jsx("span", { className: "inspector-db-readout", title: "\u5F53\u524D\u8F68\u9053\u97F3\u91CF", children: formatDb(track.volumeDb, FADER_MIN_DB) }), track.trackType === "vocal" && (() => {
                        const singers = [...voiceModels.sovits, ...voiceModels.rvc];
                        const currentSinger = singers.find((s) => s.name === track.voiceModel) ?? null;
                        return (_jsxs(_Fragment, { children: [_jsx("span", { className: "inspector-field-label", children: "\uD83C\uDFA4 \u4EBA\u7269\u58F0\u97F3" }), _jsxs("select", { className: "inspector-input-select", value: track.voiceModel ?? "", onChange: (e) => {
                                        const m = singers.find((s) => s.name === e.target.value);
                                        useHistoryStore.getState().beginTransaction();
                                        updateTrack(track.id, {
                                            voiceModel: m?.name,
                                            voiceModelAvatar: m?.avatar_path ? convertFileSrc(m.avatar_path) : undefined,
                                        });
                                        useHistoryStore.getState().commitTransaction();
                                    }, title: "\u9009\u62E9\u6B4C\u624B\u6A21\u578B (\u7FFB\u5531\u4EBA\u7269\u5361)", children: [_jsx("option", { value: "", children: "\uD83C\uDFA4 \u9009\u62E9\u6B4C\u624B\u2026" }), singers.map((m) => (_jsx("option", { value: m.name, children: m.name }, `${m.model_type}-${m.path}`)))] }), currentSinger?.avatar_path && (_jsx("img", { className: "inspector-singer-avatar", src: convertFileSrc(currentSinger.avatar_path), alt: currentSinger.name, title: currentSinger.name })), _jsx("span", { className: "inspector-field-label", children: "\uD83C\uDF10 \u8BED\u8A00" }), _jsx("select", { className: "inspector-input-select inspector-lang-select", value: track.vocalParams?.langId ?? 0, onChange: (e) => setVocalParams(track.id, { langId: +e.target.value }), title: `发音语言: ${langById(track.vocalParams?.langId ?? 0).code.toUpperCase()}`, children: VOCAL_LANGUAGES.map((l) => (_jsxs("option", { value: l.id, children: [l.short, " \u2014 ", l.code] }, l.id))) })] }));
                    })(), track.trackType === "instrument" && (_jsxs(_Fragment, { children: [_jsx("span", { className: "inspector-field-label", children: "\uD83C\uDFB9 \u97F3\u6E90" }), _jsxs("select", { className: "inspector-input-select", value: track.soundfont?.fontId ?? "", onChange: (e) => onFontChange(e.target.value), title: "\u66FF\u6362\u8FD9\u6761 MIDI \u8F68\u7684\u4E50\u5668\u97F3\u6E90", children: [_jsx("option", { value: "", children: "\u2014 \u9ED8\u8BA4\u5408\u6210 \u2014" }), fonts.map((f) => (_jsxs("option", { value: f.id, children: [f.name, " (", f.format.toUpperCase(), ")"] }, f.id)))] }), currentFont && currentFont.presets.length > 1 && (_jsx("select", { className: "inspector-input-select", value: track.soundfont?.presetId ?? "", onChange: (e) => {
                                    if (!track.soundfont)
                                        return;
                                    const preset = currentFont.presets.find((p) => p.id === e.target.value);
                                    useHistoryStore.getState().beginTransaction();
                                    setTrackSoundfont(track.id, {
                                        ...track.soundfont,
                                        presetId: e.target.value,
                                        presetName: preset?.name ?? e.target.value,
                                    });
                                    useHistoryStore.getState().commitTransaction();
                                }, title: "\u97F3\u8272 (\u97F3\u6E90\u5185\u7684\u9884\u8BBE)", children: currentFont.presets.map((p) => (_jsx("option", { value: p.id, children: p.name || p.id }, p.id))) })), currentFont?.format === "sf2" && track.soundfont && (_jsxs("select", { className: "inspector-input-select", value: track.soundfont.backend === "fluidsynth" ? "fluidsynth" : "builtin", onChange: (e) => {
                                    const cur = track.soundfont;
                                    if (!cur)
                                        return;
                                    useHistoryStore.getState().beginTransaction();
                                    setTrackSoundfont(track.id, {
                                        fontId: cur.fontId,
                                        presetId: cur.presetId,
                                        presetName: cur.presetName,
                                        ...(e.target.value === "fluidsynth" ? { backend: "fluidsynth" } : {}),
                                    });
                                    useHistoryStore.getState().commitTransaction();
                                }, title: "\u6E32\u67D3\u5F15\u64CE(\u4EC5 SF2 \u97F3\u6E90\u53EF\u9009)", children: [_jsx("option", { value: "builtin", children: "\u2699 \u5185\u7F6E\u5408\u6210\u5668" }), _jsx("option", { value: "fluidsynth", children: "\uD83C\uDF9B FluidSynth" })] }))] }))] }), _jsxs("div", { className: "inspector-main", children: [_jsxs("div", { className: "inspector-pan-group", children: [_jsx("span", { className: "inspector-pan-label", children: "\u5E73\u8861 (\u5DE6\u00B7\u53F3)" }), _jsx("div", { className: "inspector-pan-knob", onPointerDown: onPanPointerDown, title: "\u5DE6\u53F3\u58F0\u9053\u5E73\u8861 \u2014 \u6309\u4F4F\u5DE6\u53F3\u62D6\u52A8, \u5355\u51FB\u56DE\u5230\u6B63\u4E2D", children: _jsx("div", { className: "inspector-pan-indicator", style: { transform: `translateX(-50%) rotate(${panDeg}deg)` } }) }), _jsx("span", { className: "inspector-pan-value", children: formatPan(track.pan) })] }), _jsxs("div", { className: "inspector-volume-group", children: [_jsx("span", { className: "inspector-volume-label", children: "\u97F3\u91CF" }), _jsxs("div", { className: "inspector-volume-faderbox", children: [_jsx("div", { className: "inspector-fader-scale", "aria-hidden": true, children: [FADER_MAX_DB, 0, -12, FADER_MIN_DB].map((db) => {
                                            const r = (db - FADER_MIN_DB) / (FADER_MAX_DB - FADER_MIN_DB);
                                            // 钳制到盒内 — 旧版 +6 标签会越过推子台顶部, 飘出一个小方块
                                            const clamped = Math.min(0.94, Math.max(0.03, r));
                                            return _jsx("span", { style: { bottom: `${clamped * 100}%` }, children: db > 0 ? `+${db}` : db }, db);
                                        }) }), _jsx(VolumeFader, { value: track.volumeDb, min: FADER_MIN_DB, max: FADER_MAX_DB, orientation: "vertical", width: 28, height: 150, onChange: updateVolume, onGestureStart: () => useHistoryStore.getState().beginTransaction(), onGestureEnd: () => useHistoryStore.getState().commitTransaction(), tip: "\u8F68\u9053\u97F3\u91CF (\u4E0A\u4E0B\u62D6\u52A8)" }), _jsx(OutLevelMeter, { trackId: track.id, width: 14, height: 150 })] })] }), _jsxs("div", { className: "inspector-plugins-area", onClick: (e) => e.stopPropagation(), children: [_jsxs("div", { className: "inspector-plugin-insert-row", children: [_jsx("div", { className: "inspector-plugin-slot inspector-font-slot", title: "\u56FA\u5B9A\u97F3\u6E90\u63D2\u69FD \u2014 \u5728\u4E0A\u65B9\u300C\u97F3\u6E90\u300D\u4E0B\u62C9\u91CC\u66F4\u6362; MIDI \u8F68\u64AD\u653E\u65F6\u8F93\u51FA\u8FD9\u4E2A\u97F3\u6E90\u7684\u58F0\u97F3", children: track.soundfont
                                            ? _jsxs("span", { children: ["\uD83C\uDFB9 ", currentFont?.name ?? "已选音源"] })
                                            : _jsx("span", { children: "\uD83C\uDFB9 \u97F3\u6E90 \u00B7 \u9ED8\u8BA4\u5408\u6210" }) }), plugins.map((pluginId, i) => {
                                        const p = PLUGIN_CATALOG.find((pc) => pc.id === pluginId);
                                        return (_jsx("div", { className: `inspector-plugin-slot ${activeSlot === i ? "inspector-plugin-slot-active" : ""}`, onClick: () => { setActiveSlot(i); setInsertOpen(true); }, title: p ? `${p.name} — 点击更换/移除` : `效果器插槽 ${i + 1}（空）— 点击插入效果器`, children: p ? (_jsxs("span", { children: [p.icon, " ", p.name, _jsx("button", { className: "inspector-plugin-clear", onClick: (e) => { e.stopPropagation(); onClearSlot(i); }, title: "\u79FB\u9664", children: "\u00D7" })] })) : (_jsxs("span", { style: { color: "var(--text-tertiary, #666)" }, children: ["\uFF0B \u7A7A\u69FD ", i + 1] })) }, i));
                                    }), _jsx("div", { className: "inspector-plugin-slot inspector-plugin-add", onClick: () => { setActiveSlot(-1); setInsertOpen(true); }, title: "\u6DFB\u52A0\u4E00\u4E2A\u65B0\u6548\u679C\u5668\u63D2\u69FD", children: _jsx("span", { children: "\uFF0B \u6DFB\u52A0\u6548\u679C\u5668" }) })] }), insertOpen && (_jsxs("div", { className: "inspector-insert-menu", children: [_jsx("div", { className: "inspector-insert-menu-title", children: activeSlot >= 0
                                            ? `替换插槽 ${activeSlot + 1}（${plugins[activeSlot] ? PLUGIN_CATALOG.find(p => p.id === plugins[activeSlot])?.name : "空"}）`
                                            : "添加新效果器插槽" }), PLUGIN_CATALOG.map((p) => (_jsxs("button", { className: "inspector-insert-item", onClick: () => onInsertPlugin(p.id), children: [_jsx("span", { className: "inspector-insert-item-icon", children: p.icon }), _jsx("span", { children: p.name })] }, p.id))), plugins[activeSlot] && (_jsx("button", { className: "inspector-insert-item inspector-insert-clear", onClick: () => onClearSlot(activeSlot), children: "\uD83D\uDDD1\uFE0F \u6E05\u9664\u6B64\u69FD\u4F4D" }))] })), _jsxs("div", { className: "inspector-sends-row", children: [_jsx("span", { className: "inspector-sends-label", children: "\uD83C\uDF9B \u53D1\u9001" }), _jsxs("div", { className: "inspector-send-vert", title: "\u6DF7\u54CD\u53D1\u9001 \u2014 \u53D1\u5230\u5168\u5C40\u6DF7\u54CD\u603B\u7EBF\u7684\u91CF (\u4E0A\u4E0B\u62D6\u52A8)", children: [_jsx(VolumeFader, { value: track.reverbSend ?? 0, min: 0, max: 1, orientation: "vertical", width: 22, height: 84, step: 0.01, onChange: (v) => setTrackFxSend(track.id, "reverbSend", v), onGestureStart: () => useHistoryStore.getState().beginTransaction(), onGestureEnd: () => useHistoryStore.getState().commitTransaction(), tip: "\u6DF7\u54CD\u53D1\u9001 (\u4E0A\u4E0B\u62D6\u52A8)" }), _jsxs("span", { className: "inspector-send-vert-name", children: ["\uD83C\uDF0A ", Math.round((track.reverbSend ?? 0) * 100), "%"] })] }), _jsxs("div", { className: "inspector-send-vert", title: "\u5EF6\u8FDF\u53D1\u9001 \u2014 \u53D1\u5230\u5168\u5C40\u5EF6\u8FDF\u603B\u7EBF\u7684\u91CF (\u4E0A\u4E0B\u62D6\u52A8)", children: [_jsx(VolumeFader, { value: track.delaySend ?? 0, min: 0, max: 1, orientation: "vertical", width: 22, height: 84, step: 0.01, onChange: (v) => setTrackFxSend(track.id, "delaySend", v), onGestureStart: () => useHistoryStore.getState().beginTransaction(), onGestureEnd: () => useHistoryStore.getState().commitTransaction(), tip: "\u5EF6\u8FDF\u53D1\u9001 (\u4E0A\u4E0B\u62D6\u52A8)" }), _jsxs("span", { className: "inspector-send-vert-name", children: ["\u23F1 ", Math.round((track.delaySend ?? 0) * 100), "%"] })] }), _jsx("button", { className: "inspector-sends-reset", onClick: () => {
                                            useHistoryStore.getState().beginTransaction();
                                            setTrackFxSend(track.id, "reverbSend", 0);
                                            setTrackFxSend(track.id, "delaySend", 0);
                                            useHistoryStore.getState().commitTransaction();
                                        }, title: "\u6E05\u96F6\u6240\u6709\u53D1\u9001", children: "\u00D7 \u6E05\u96F6" })] })] })] }), _jsxs("div", { className: "inspector-track-tabs", children: [tracks.map((t, i) => {
                        const color = trackTypeCssVar(t.trackType); // 统一走 trackColors（与 TrackList/主题 --track-* 一致）
                        const isActive = t.id === inspectorTrackId;
                        return (_jsxs("button", { className: `inspector-track-tab ${isActive ? "active" : ""}`, onClick: () => openInspector(t.id), style: {
                                borderColor: isActive ? color : "transparent",
                                borderLeftColor: isActive ? color : color,
                                borderLeftWidth: isActive ? "4px" : "3px",
                            }, title: t.name, children: [_jsx("span", { style: { color, fontWeight: 700, marginRight: 4 }, children: i + 1 }), t.name] }, t.id));
                    }), tracks.length === 0 && (_jsx("span", { style: { color: "var(--text-tertiary, #666)", fontSize: 11, padding: "4px 10px" }, children: "\u5DE5\u7A0B\u91CC\u8FD8\u6CA1\u6709\u8F68\u9053" }))] })] }));
}
