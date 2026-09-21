import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState, useCallback, useEffect, useRef, Fragment } from "react";
import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import { useHistoryStore } from "../../store/history";
import { useTranslation } from "react-i18next";
import { open } from "@tauri-apps/plugin-dialog";
import { LANE_HEIGHT, LANE_GROUP_BAR_HEIGHT, LOUDNESS_LANE_HEIGHT, TRACK_HEADER_HEIGHT, FADER_MIN_DB, FADER_MAX_DB, AUDIO_EXTENSIONS } from "../../lib/constants";
import { computeTrackHeight, computeTrackYOffsets, computeTotalTracksHeight, findTrackAtY, hiddenTrackIds, getLanes, getLaneLayout, isLaneRowMuted, laneControlFor, loudnessBandH } from "../../lib/trackLayout";
import { laneLabelParts } from "../../lib/audio/laneOps";
import { trackTypeCssVar, LANE_COLORS } from "../../lib/trackColors";
import { importAudioToNewTrack } from "../../lib/audio/import";
import { blankTrack } from "../../lib/trackFactory";
import { copyTrackToClipboard, pasteWithFeedback, clipboardKind } from "../../lib/clipboard";
import { VolumeFader, formatPan, formatDb } from "../common/VolumeFader";
import { ContextMenu } from "../common/ContextMenu";
import { songTaskSubmenuItems } from "../../lib/song/daw-menu";
import * as playback from "../../lib/audio/playback";
import { useVoiceModelStore } from "../../store/voice-models";
import { backendOf, backendLabel, pickVoiceForTrack } from "../../lib/vocal/voicePick";
import { VOCAL_LANGUAGES, DEFAULT_LANG_ID } from "../../lib/vocal/languages";
import { isTauri } from "../../lib/tauri";
import { useAmtModelStore } from "../../store/amt-models";
import { AMT_CATALOG } from "../../lib/models/amt-catalog";
// —— AI 引擎（右键菜单新增）——
import { analyzeTrackChords, detectTrackDrums } from "../../lib/analysis/trackAnalysisActions";
import { ArrangeDialog } from "./ArrangeDialog";
import { ChordMidiDialog } from "./ChordMidiDialog";
import { AmtConversionDialog } from "../models/AmtConversionDialog";
import i18n from "../../i18n";
import "./TrackList.css";
import { TrackColorPicker } from "./TrackColorPicker";
import { useRecording } from "../../lib/audio/recorder";
import { TrackWaveform } from "./TrackWaveform";
export function TrackList({ width }) {
    const { t } = useTranslation();
    // Per-field selectors so this column re-renders only on values it shows — NOT on playheadTick
    // (every playback frame) or scrollX (horizontal scroll). scrollY self-subscribed for the
    // vertical transform.
    const tracks = useProjectStore((s) => s.tracks);
    const updateTrack = useProjectStore((s) => s.updateTrack);
    const removeTrack = useProjectStore((s) => s.removeTrack);
    const toggleTrackExpanded = useProjectStore((s) => s.toggleTrackExpanded);
    const updateLaneControl = useProjectStore((s) => s.updateLaneControl);
    const setLaneMute = useProjectStore((s) => s.setLaneMute);
    const setTrackPlayOriginal = useProjectStore((s) => s.setTrackPlayOriginal);
    const addTrack = useProjectStore((s) => s.addTrack);
    const activeTrackId = useAppStore((s) => s.activeTrackId);
    const setActiveTrack = useAppStore((s) => s.setActiveTrack);
    const scrollY = useAppStore((s) => s.scrollY);
    const vZoom = useAppStore((s) => s.vZoom);
    const ghostInsert = useAppStore((s) => s.ghostInsert);
    const [menu, setMenu] = useState(null);
    const [hoverBoundary, setHoverBoundary] = useState(null);
    const [editingTrackId, setEditingTrackId] = useState(null);
    const [draggingTrackId, setDraggingTrackId] = useState(null);
    // —— 轨道颜色选择器 state ——
    const [colorPicker, setColorPicker] = useState(null);
    // —— AI 弹窗 state (右键菜单触发) ——
    const [arrangeTarget, setArrangeTarget] = useState(null);
    const [chordMidiTarget, setChordMidiTarget] = useState(null);
    const [amtTarget, setAmtTarget] = useState(null);
    const [amtMissingOpen, setAmtMissingOpen] = useState(null);
    // 一键扒带:音频转谱完成后自动打开智能编曲面板(指向新转出的第一条旋律轨)。
    const [arrangeAfterAmt, setArrangeAfterAmt] = useState(false);
    // ── AMT 依赖预检: 右键点 AI 转谱时, 先检查 Tauri 环境 + 模型是否齐全 ──
    const tryOpenAmt = useCallback(async (trackId) => {
        // 1) 不是 Tauri → 轻量提示 (没有 invoke 后端)
        if (!isTauri()) {
            useAppStore.getState().showToast("⚠️ AI 转谱需要在 Tauri 桌面应用里运行 (当前是浏览器预览)", "info");
            setAmtMissingOpen(trackId);
            return;
        }
        // 2) 拉已安装模型清单 → 算缺了啥
        try {
            const store = useAmtModelStore.getState();
            await store.fetchInstalled();
            const needed = AMT_CATALOG.filter((m) => ["fluidsynth", "soundfont", "yourmt3_plus"].includes(m.architecture));
            const missing = needed.filter((m) => {
                const installed = store.installed.find((i) => i.id === m.id || i.architecture === m.architecture);
                return !installed || !installed.is_available;
            });
            if (missing.length > 0) {
                setAmtMissingOpen(trackId);
                useAppStore.getState().showToast(`⚠️ 缺少 ${missing.length} 个组件, 请在弹窗里点下载`, "info");
                return;
            }
            // 3) 全部齐了 → 正常打开
            setAmtTarget(trackId);
        }
        catch (e) {
            // fetchInstalled 也会因为无后端失败 → 直接进缺失弹窗
            useAppStore.getState().showToast("⚠️ 无法连接后端, 请确认 Tauri 应用正常启动", "info");
            setAmtMissingOpen(trackId);
        }
    }, []);
    const listRef = useRef(null);
    const trackDragRef = useRef(null);
    // The boundary hint is computed from the cursor vs. the track layout; when the layout shifts
    // under a stationary cursor (vertical zoom or scroll), drop the stale hint until the pointer
    // moves again — otherwise the line sticks to the wrong boundary.
    useEffect(() => { setHoverBoundary(null); }, [vZoom, scrollY]);
    // 快捷键支持: R录音、M静音、S独奏
    useEffect(() => {
        const handleKeyDown = (e) => {
            // 忽略在输入框中的按键
            if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement)
                return;
            // 忽略有修饰键的按键
            if (e.ctrlKey || e.metaKey || e.altKey)
                return;
            const activeTrack = tracks.find((t) => t.id === activeTrackId);
            if (!activeTrack)
                return;
            switch (e.key.toLowerCase()) {
                case 'r':
                    // R键: 录音开关(与轨道头 R 按钮同一 toggle)
                    e.preventDefault();
                    void useRecording.getState().toggle(activeTrack.id);
                    break;
                case 'm':
                    // M键: 切换静音
                    e.preventDefault();
                    updateTrack(activeTrack.id, { muted: !activeTrack.muted });
                    playback.updateTrackAudibility(useProjectStore.getState().tracks);
                    break;
                case 's':
                    // S键: 切换独奏
                    e.preventDefault();
                    updateTrack(activeTrack.id, { solo: !activeTrack.solo });
                    playback.updateTrackAudibility(useProjectStore.getState().tracks);
                    break;
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [tracks, activeTrackId, updateTrack]);
    // Drag-reorder tracks by their header. Starts only past a small threshold (so a plain click still
    // selects); live-reorders as the cursor crosses track midpoints.
    const onTrackHeaderMouseDown = useCallback((e, trackId) => {
        if (e.button !== 0)
            return;
        const el = e.target;
        if (el.closest(".track-btn, .track-expand-btn, .track-name, .track-name-input, .vol-fader, .track-row-bot"))
            return;
        trackDragRef.current = { trackId, startY: e.clientY, dragging: false };
    }, []);
    // ── 轨道底边纵向拖拽(Studio Pro 式):拖任一条轨的底边 → 全局 vZoom 等比缩放(与 Alt+滚轮同
    //    一通路)。边缘跟随光标:每拖 TRACK_HEADER_HEIGHT 像素 ≈ vZoom ±1。pointer capture + rAF
    //    合帧,拖完不持久化(vZoom 本就不入 localStorage)。 ─────────────────────────────────
    const vDragRef = useRef(null);
    const vDragRafRef = useRef(0);
    useEffect(() => () => cancelAnimationFrame(vDragRafRef.current), []);
    const onTrackResizePointerDown = useCallback((e) => {
        if (e.button !== 0)
            return;
        e.preventDefault();
        e.stopPropagation();
        try {
            e.currentTarget.setPointerCapture(e.pointerId);
        }
        catch { /* ignore */ }
        const st = useAppStore.getState();
        vDragRef.current = { startY: e.clientY, startZoom: st.vZoom };
        document.body.style.cursor = "row-resize";
        document.body.style.userSelect = "none";
        const onMove = (ev) => {
            const d = vDragRef.current;
            if (!d)
                return;
            const next = d.startZoom + (ev.clientY - d.startY) / TRACK_HEADER_HEIGHT;
            if (!vDragRafRef.current) {
                vDragRafRef.current = requestAnimationFrame(() => {
                    vDragRafRef.current = 0;
                    useAppStore.getState().setVZoom(next);
                });
            }
        };
        const onUp = () => {
            vDragRef.current = null;
            if (vDragRafRef.current) {
                cancelAnimationFrame(vDragRafRef.current);
                vDragRafRef.current = 0;
            }
            document.body.style.cursor = "";
            document.body.style.userSelect = "";
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", onUp);
        };
        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onUp);
    }, []);
    useEffect(() => {
        const onMove = (e) => {
            const d = trackDragRef.current;
            if (!d)
                return;
            if (!d.dragging) {
                if (Math.abs(e.clientY - d.startY) <= 4)
                    return;
                d.dragging = true;
                setDraggingTrackId(d.trackId);
                document.body.style.cursor = "grabbing";
                document.body.style.userSelect = "none"; // no stray text selection across track names
                // Reorder fires live per midpoint cross — coalesce the whole drag into one undo step.
                useHistoryStore.getState().beginTransaction();
            }
            const el = listRef.current;
            if (!el)
                return;
            const proj = useProjectStore.getState();
            const trks = proj.tracks;
            const fromIdx = trks.findIndex((t) => t.id === d.trackId);
            if (fromIdx < 0)
                return;
            const vz = useAppStore.getState().vZoom;
            const contentY = e.clientY - el.getBoundingClientRect().top + useAppStore.getState().scrollY;
            // Target = how many OTHER tracks' midpoints sit above the cursor. Excluding the dragged track
            // gives proper hysteresis (using its own slot as the only dead-band oscillates when it is
            // shorter than the track it crosses).
            const offs = computeTrackYOffsets(trks, vz);
            let target = 0;
            for (let i = 0; i < trks.length; i++) {
                if (i === fromIdx)
                    continue;
                if (contentY >= offs[i] + computeTrackHeight(trks[i], vz) / 2)
                    target++;
            }
            if (target !== fromIdx)
                proj.reorderTrack(fromIdx, target);
        };
        const onUp = () => {
            if (trackDragRef.current) {
                const wasDragging = trackDragRef.current.dragging;
                trackDragRef.current = null;
                setDraggingTrackId(null);
                document.body.style.cursor = "";
                document.body.style.userSelect = "";
                if (wasDragging)
                    useHistoryStore.getState().commitTransaction();
            }
        };
        document.addEventListener("mousemove", onMove);
        document.addEventListener("mouseup", onUp);
        return () => {
            document.removeEventListener("mousemove", onMove);
            document.removeEventListener("mouseup", onUp);
            document.body.style.cursor = "";
            document.body.style.userSelect = "";
        };
    }, []);
    const commitRename = useCallback((trackId, name) => {
        const n = name.trim();
        if (n)
            updateTrack(trackId, { name: n });
        setEditingTrackId(null);
    }, [updateTrack]);
    const offsets = computeTrackYOffsets(tracks, vZoom);
    const totalH = computeTotalTracksHeight(tracks, vZoom);
    const hidden = hiddenTrackIds(tracks);
    // ── 文件夹分组操作 ──────────────────────────────────────────────
    // 在某轨上方插入一个空的文件夹容器轨, 并把该轨作为第一个子轨移入。
    const createFolderAbove = useCallback((trackId) => {
        const idx = tracks.findIndex((tk) => tk.id === trackId);
        const fid = crypto.randomUUID();
        const folder = {
            id: fid, name: "新建分组", trackType: "audio", segments: [],
            volumeDb: 0, pan: 0, muted: false, solo: false, expanded: false,
            laneControls: {}, isFolder: true, folderCollapsed: false,
        };
        addTrack(folder, idx < 0 ? undefined : idx);
        updateTrack(trackId, { folderId: fid });
    }, [tracks, addTrack, updateTrack]);
    // 解散分组: 子轨 folderId 全部清掉(保留轨道与内容), 再移除文件夹轨本身。
    const dissolveFolder = useCallback((folderId) => {
        useHistoryStore.getState().beginTransaction();
        for (const tk of tracks) {
            if (tk.folderId === folderId)
                updateTrack(tk.id, { folderId: undefined });
        }
        removeTrack(folderId);
        useHistoryStore.getState().commitTransaction();
    }, [tracks, updateTrack, removeTrack]);
    // 移入上方最近的分组文件夹(向上扫描 tracks 数组)。
    const moveIntoFolderAbove = useCallback((trackId) => {
        const idx = tracks.findIndex((tk) => tk.id === trackId);
        for (let j = idx - 1; j >= 0; j--) {
            if (tracks[j]?.isFolder) {
                updateTrack(trackId, { folderId: tracks[j].id });
                return;
            }
        }
        useAppStore.getState().showToast("上方没有分组文件夹 — 先右键某轨「新建分组文件夹」", "info");
    }, [tracks, updateTrack]);
    // Reusable track-creation actions. `insertIndex` positions the new track at a boundary (the
    // right-click "add material" menu); omitting it appends at the bottom (the "+" menu).
    const createAudioTrack = useCallback((insertIndex) => {
        const n = useProjectStore.getState().tracks.filter((tk) => tk.trackType === "audio").length + 1;
        addTrack(blankTrack(crypto.randomUUID(), `Audio ${n}`, "audio"), insertIndex);
    }, [addTrack]);
    const createVocalTrack = useCallback((insertIndex) => {
        const n = useProjectStore.getState().tracks.filter((tk) => tk.trackType === "vocal").length + 1;
        addTrack(blankTrack(crypto.randomUUID(), `Vocal ${n}`, "vocal"), insertIndex);
    }, [addTrack]);
    const importAudioAt = useCallback(async (insertIndex) => {
        const path = await open({
            title: t("toolbar.importAudio"),
            filters: [{ name: "Audio", extensions: AUDIO_EXTENSIONS }],
        });
        if (!path)
            return;
        // import.ts creates the track + loading segment immediately + owns decode/error handling.
        void importAudioToNewTrack(path, useProjectStore.getState().playheadTick, undefined, insertIndex);
    }, [t]);
    // Boundary nearest the cursor (insert index 0..tracks.length) within a small threshold, else null.
    const boundaryAt = useCallback((clientY) => {
        const el = listRef.current;
        if (!el)
            return null;
        const contentY = clientY - el.getBoundingClientRect().top + scrollY;
        for (let i = 0; i <= tracks.length; i++) {
            const by = i < tracks.length ? offsets[i] : totalH;
            if (Math.abs(contentY - by) <= 5)
                return i;
        }
        return null;
    }, [tracks.length, offsets, totalH, scrollY]);
    const handleContextMenu = useCallback((e) => {
        e.preventDefault();
        const b = boundaryAt(e.clientY);
        if (b !== null) {
            setMenu({ kind: "add", index: b, x: e.clientX, y: e.clientY });
            return;
        }
        const el = listRef.current;
        if (!el)
            return;
        const contentY = e.clientY - el.getBoundingClientRect().top + scrollY;
        const idx = findTrackAtY(offsets, contentY);
        if (idx >= 0 && idx < tracks.length && contentY <= totalH) {
            setMenu({ kind: "track", trackId: tracks[idx].id, x: e.clientX, y: e.clientY });
        }
        else {
            // Empty area below the last track → "add material", appending at the bottom.
            setMenu({ kind: "add", index: tracks.length, x: e.clientX, y: e.clientY });
        }
    }, [boundaryAt, offsets, totalH, scrollY, tracks]);
    const voiceModels = useVoiceModelStore((s) => s.models);
    const setVocalParams = useProjectStore((s) => s.setVocalParams);
    const vocalOov = useAppStore((s) => s.vocalOov); // ② S58 track-level OOV warning
    const vocalUnknownPhone = useAppStore((s) => s.vocalUnknownPhone); // S109 §C15: 音素写错 ≠ 歌词不认识
    const vocalDropped = useAppStore((s) => s.vocalDropped); // S85b: too-short dropped notes(独立文案)
    const vocalShort = useAppStore((s) => s.vocalShort); // S87: rescued notes — advisory, amber badge
    const vocalAliasHint = useAppStore((s) => s.vocalAliasHint); // S113 §C14: 别名形状提示 — advisory
    const menuItems = (() => {
        if (!menu)
            return [];
        if (menu.kind === "track") {
            const tr = tracks.find((tt) => tt.id === menu.trackId);
            if (!tr)
                return [];
            // DEBUG: 真实 track 结构
            console.log("[AI MENU DEBUG] track:", tr?.name, "type:", tr?.trackType, "segments:", tr?.segments?.map((s) => ({ type: s.content?.type, hasSource: !!s.content?.sourcePath })));
            const isMelodyLike = !!tr && (tr.trackType === "instrument" || tr.trackType === "vocal");
            const isAudio = !!tr && tr.trackType === "audio";
            // Audio path for audio-track analysis (drums, chords, AMT)
            const audioPath = tr
                ? tr.segments.find((s) => s.content.type === "audioClip")?.content?.sourcePath ?? null
                : null;
            const hasNotes = isMelodyLike && tr.segments.some((s) => s.content.type === "notes" && s.content.notes.length > 0);
            // Per-item enable rules (smarter than a single `hasNotes` gate):
            //  - autoArrange / chordMidi: need NOTES (MIDI content to arrange FROM)
            //  - detectChords: works on BOTH notes-track (pitch-class histogram) AND audio-track (onboard FFT)
            //  - detectDrums: works ONLY on audio tracks with source file
            const arrangeDisabled = !hasNotes;
            const chordMidiDisabled = !hasNotes;
            const chordAnalyzeDisabled = !hasNotes && !(isAudio && !!audioPath);
            const detectDrumsDisabled = !isAudio || !audioPath;
            const amtDisabled = !isAudio || !audioPath;
            // Labels & tooltips adapt so users understand WHY an item is grayed out:
            const tipNoNotes = "需要有音符内容的旋律/乐器轨";
            const tipNoAudio = "需要音频轨 (带音频文件)";
            return [
                // —— AI 快捷入口（最简右键直达）——
                { label: "🎵 转MIDI (AI 转谱)…", disabled: amtDisabled, title: amtDisabled ? tipNoAudio : "用 AI 模型把音频转成 MIDI 音符轨 (需要 Tauri 后端)",
                    onClick: () => { void tryOpenAmt(menu.trackId); } },
                { label: "⚡ 一键扒带编曲 (音频→转谱→配器)", disabled: amtDisabled, title: amtDisabled ? tipNoAudio : "音频先 AI 转谱,完成后自动打开智能编曲面板(自动定速/定调/出和弦)",
                    onClick: () => {
                        setArrangeAfterAmt(true);
                        void tryOpenAmt(menu.trackId);
                    } },
                { label: "✨ 智能编曲…", disabled: arrangeDisabled, title: arrangeDisabled ? tipNoNotes : "打开编曲面板:11 类乐器自由组合,实时试听后一键生成多轨",
                    onClick: () => setArrangeTarget(menu.trackId) },
                { label: "🎹 和弦MIDI…", disabled: chordMidiDisabled, title: chordMidiDisabled ? tipNoNotes : "生成一条和弦轨 (C Am F G …)",
                    onClick: () => setChordMidiTarget(menu.trackId) },
                { label: "🎼 识别和弦", disabled: chordAnalyzeDisabled, title: chordAnalyzeDisabled ? (isAudio ? tipNoAudio : tipNoNotes) : "分析这条轨的和弦进行 (显示在和弦轨)",
                    onClick: () => { analyzeTrackChords(menu.trackId); useAppStore.getState().showToast(i18n.t("chordTrack.analyzeDone"), "info"); } },
                { label: "🥁 识别鼓点", disabled: detectDrumsDisabled, title: detectDrumsDisabled ? tipNoAudio : "把这条音频轨的鼓点变成 MIDI 鼓轨",
                    onClick: async () => { if (audioPath) {
                        await detectTrackDrums(menu.trackId, audioPath);
                        useAppStore.getState().showToast("鼓点识别完成", "success");
                    } } },
                // —— 规划 10.2：歌曲模型子菜单（DAW → 歌曲制作带参打开；不改变现有项与顺序）——
                ...(!!audioPath
                    ? [{
                            type: "submenu",
                            label: i18n.t("songMenu.remixGroup"),
                            icon: "🎤",
                            items: songTaskSubmenuItems({
                                source: { kind: "track", trackId: menu.trackId },
                                sourceLabel: tr?.name,
                                audioPath,
                                trackId: menu.trackId,
                            }),
                        }]
                    : []),
                // —— 文件夹分组 ——
                ...(tr.isFolder
                    ? [
                        { label: tr.folderCollapsed ? "📂 展开分组" : "📁 折叠分组",
                            onClick: () => updateTrack(menu.trackId, { folderCollapsed: !tr.folderCollapsed }) },
                        { label: "🗂️ 解散分组 (保留子轨)", onClick: () => dissolveFolder(menu.trackId) },
                    ]
                    : [
                        ...(tr.folderId
                            ? [{ label: "📤 移出分组", onClick: () => updateTrack(menu.trackId, { folderId: undefined }) }]
                            : [{ label: "📁 移入上方分组", onClick: () => moveIntoFolderAbove(menu.trackId) }]),
                        { label: "🗂️ 新建分组文件夹 (于此轨上方)", onClick: () => createFolderAbove(menu.trackId) },
                    ]),
                // —— 原有轨道操作 ——
                { label: t("tracks.rename"), onClick: () => setEditingTrackId(menu.trackId) },
                { label: t("tracks.copyTrack"), onClick: () => { copyTrackToClipboard(menu.trackId); } },
                {
                    label: t("tracks.pasteTrack"), disabled: clipboardKind() !== "track",
                    onClick: () => pasteWithFeedback(menu.trackId),
                },
                { label: t("tracks.delete"), danger: true, onClick: () => removeTrack(menu.trackId) },
            ];
        }
        // ② S58 header pickers — quick whole-track setup without opening the segment editor (and a visible
        // cue that singer/language are TRACK-level, not per-segment). Details (quality/vocoder…) stay in
        // the editor sidebar. The pick path is the SHARED pickVoiceForTrack (same as the sidebar — NO-dup).
        if (menu.kind === "voice") {
            const all = [...voiceModels.sovits, ...voiceModels.rvc];
            const cur = tracks.find((tr) => tr.id === menu.trackId);
            if (all.length === 0)
                return [{ label: t("tracks.noVoices"), disabled: true, onClick: () => { } }];
            return all.map((m) => ({
                label: `${m.name} · ${backendLabel(m)}`,
                active: cur?.voiceModel === m.name && cur?.vocalParams?.backend === backendOf(m),
                onClick: () => pickVoiceForTrack(menu.trackId, m),
            }));
        }
        if (menu.kind === "lang") {
            const cur = tracks.find((tr) => tr.id === menu.trackId);
            const curId = cur?.vocalParams?.langId ?? DEFAULT_LANG_ID;
            return VOCAL_LANGUAGES.map((l) => ({
                label: `${l.short} · ${t(`langs.${l.code}`)}`,
                active: l.id === curId,
                onClick: () => setVocalParams(menu.trackId, { langId: l.id }),
            }));
        }
        return [
            { label: t("toolbar.importAudio"), onClick: () => importAudioAt(menu.index) },
            { label: t("toolbar.addAudio"), onClick: () => createAudioTrack(menu.index) },
            { label: t("toolbar.addMidi"), onClick: () => createVocalTrack(menu.index) },
        ];
    })();
    return (_jsxs("div", { className: "track-list", style: { width }, ref: listRef, onContextMenu: handleContextMenu, onMouseMove: (e) => {
            if (trackDragRef.current || vDragRef.current) {
                if (hoverBoundary !== null)
                    setHoverBoundary(null);
                return;
            }
            const b = boundaryAt(e.clientY);
            setHoverBoundary((prev) => (prev === b ? prev : b));
        }, onMouseLeave: () => setHoverBoundary(null), children: [_jsxs("div", { className: "track-list-scroll", style: { transform: `translateY(${-(Math.round(scrollY * (window.devicePixelRatio || 1)) / (window.devicePixelRatio || 1))}px)` }, children: [tracks.length === 0 && (_jsx("div", { className: "track-list-empty", onClick: async () => {
                            // 点击空区域 → 打开文件选择器，添加音频文件
                            const path = await open({
                                title: t("toolbar.importAudio"),
                                filters: [{ name: "Audio", extensions: AUDIO_EXTENSIONS }],
                            });
                            if (!path)
                                return;
                            // 第一个文件从起始位置（tick 0）开始
                            void importAudioToNewTrack(path, 0);
                        }, style: { cursor: "pointer" }, title: "\u70B9\u51FB\u4E0A\u4F20\u97F3\u4E50\u6587\u4EF6", children: _jsxs("div", { style: { display: "flex", flexDirection: "column", alignItems: "center", gap: "8px" }, children: [_jsx("span", { className: "text-muted", children: t("tracks.empty") }), _jsx("span", { style: { fontSize: "12px", opacity: 0.6 }, children: "\u70B9\u51FB\u4E0A\u4F20\u97F3\u4E50" })] }) })), hoverBoundary !== null && (_jsx("div", { className: "track-boundary-hint", style: { top: hoverBoundary < tracks.length ? offsets[hoverBoundary] : totalH } })), tracks.map((track, i) => (_jsxs(Fragment, { children: [hidden.has(track.id) ? null : ghostInsert && ghostInsert.index === i && (_jsx("div", { className: "track-ghost-slot", style: { height: ghostInsert.count * TRACK_HEADER_HEIGHT * vZoom } })), hidden.has(track.id) ? null : (_jsx(TrackItem, { track: track, index: i, childCount: track.isFolder ? tracks.filter((tk) => tk.folderId === track.id).length : undefined, vZoom: vZoom, hasSolo: tracks.some((tk) => tk.solo), active: track.id === activeTrackId, dragging: track.id === draggingTrackId, editing: track.id === editingTrackId, onHeaderMouseDown: (e) => onTrackHeaderMouseDown(e, track.id), onStartRename: () => setEditingTrackId(track.id), onCommitRename: (name) => commitRename(track.id, name), onCancelRename: () => setEditingTrackId(null), onSelect: () => setActiveTrack(track.id), onMute: () => {
                                    updateTrack(track.id, { muted: !track.muted });
                                    playback.updateTrackAudibility(useProjectStore.getState().tracks);
                                }, onSolo: () => {
                                    updateTrack(track.id, { solo: !track.solo });
                                    playback.updateTrackAudibility(useProjectStore.getState().tracks);
                                }, onVolumeChange: (v) => {
                                    updateTrack(track.id, { volumeDb: v });
                                    playback.updateTrackVolume(track.id, v);
                                }, onPanChange: (v) => {
                                    updateTrack(track.id, { pan: v });
                                    playback.updateTrackPan(track.id, v);
                                }, onToggleExpand: () => toggleTrackExpanded(track.id), onTogglePlayOriginal: () => setTrackPlayOriginal(track.id, !track.playOriginal), onLaneMute: (members) => {
                                    // A merged row toggles as ONE: "muted" reads as all-members-muted, and the write fans
                                    // out over every member rowKey — inside one transaction so the click is one undo step
                                    // (each setLaneMute is a separate store set that would otherwise auto-capture).
                                    const newMuted = !members.every((m) => isLaneRowMuted(track, m.rowKey, m.laneId));
                                    useHistoryStore.getState().beginTransaction();
                                    for (const m of members) {
                                        setLaneMute(track.id, m.rowKey, newMuted);
                                        playback.updateLaneMute(track.id, m.rowKey, newMuted, laneControlFor(track, m.groupId, m.laneId)?.volumeDb ?? 0);
                                    }
                                    useHistoryStore.getState().commitTransaction();
                                }, onLaneVolumeChange: (run, v) => {
                                    // Fan out over every member 组 under this bar (merged rows stay in lockstep; the legacy
                                    // laneId seed applies only to the primary 组 — other members' legacy entries key by
                                    // THEIR laneIds and simply start fresh, converging on this first touch).
                                    for (const gid of run.groupIds) {
                                        updateLaneControl(track.id, gid, { volumeDb: v }, gid === run.groupId ? run.laneId : undefined);
                                        playback.updateLaneVolume(track.id, gid, v);
                                    }
                                }, onLanePanChange: (run, v) => {
                                    for (const gid of run.groupIds) {
                                        updateLaneControl(track.id, gid, { pan: v }, gid === run.groupId ? run.laneId : undefined);
                                        playback.updateLanePan(track.id, gid, v);
                                    }
                                }, onOpenColorPicker: (x, y) => setColorPicker({ trackId: track.id, x, y }), onResizePointerDown: onTrackResizePointerDown, hasOov: track.segments.some((sg) => (vocalOov[sg.id]?.length ?? 0) > (vocalUnknownPhone[sg.id]?.length ?? 0)), hasUnknownPhone: track.segments.some((sg) => (vocalUnknownPhone[sg.id]?.length ?? 0) > 0), hasDropped: track.segments.some((sg) => (vocalDropped[sg.id]?.length ?? 0) > 0), hasShort: track.segments.some((sg) => (vocalShort[sg.id]?.length ?? 0) > 0), hasAliasHint: track.segments.some((sg) => (vocalAliasHint[sg.id]?.length ?? 0) > 0) }))] }, track.id)))] }), menu && (_jsx(ContextMenu, { x: menu.x, y: menu.y, items: menuItems, onClose: () => setMenu(null) })), amtTarget && (_jsx(AmtConversionDialog, { trackId: amtTarget, onClose: () => { setAmtTarget(null); setArrangeAfterAmt(false); }, onImported: arrangeAfterAmt ? (ids) => {
                    // 转谱完成 → 自动打开智能编曲面板,以第一条新轨为编曲源(它含全部音符)。
                    const first = ids[0];
                    if (first)
                        setArrangeTarget(first);
                } : undefined })), amtMissingOpen && (() => {
                const store = useAmtModelStore();
                const needed = AMT_CATALOG.filter((m) => ["fluidsynth", "soundfont", "yourmt3_plus"].includes(m.architecture));
                const missing = needed.filter((m) => {
                    const installed = store.installed.find((i) => i.id === m.id || i.architecture === m.architecture);
                    return !installed || !installed.is_available;
                });
                return (_jsx("div", { className: "arrange-overlay", onClick: () => setAmtMissingOpen(null), children: _jsxs("div", { className: "arrange-dialog", onClick: (e) => e.stopPropagation(), style: { borderColor: "var(--color-error)" }, children: [_jsxs("div", { className: "arrange-header", children: [_jsx("span", { className: "arrange-title", style: { color: "var(--color-error)" }, children: "\u26A0\uFE0F AI \u8F6C\u8C31\u7EC4\u4EF6\u7F3A\u5931" }), _jsx("button", { className: "arrange-close", onClick: () => setAmtMissingOpen(null), children: "\u2715" })] }), _jsxs("div", { style: { padding: "12px 16px" }, children: [_jsxs("p", { style: { fontSize: 12, color: "var(--text-secondary)", marginBottom: 12 }, children: ["\u68C0\u6D4B\u5230 ", missing.length, " \u4E2A\u5FC5\u8981\u7EC4\u4EF6\u8FD8\u6CA1\u88C5. \u70B9\u6BCF\u4E2A\u7EC4\u4EF6\u53F3\u8FB9\u7684\u300C\u4E0B\u8F7D\u300D\u6309\u94AE\u5373\u53EF. \u5168\u90E8\u4E0B\u8F7D\u5B8C\u540E\u53F3\u952E\u90A3\u6761\u97F3\u9891\u8F68 \u2192 \u518D\u70B9\u300CAI \u8F6C\u8C31\u300D."] }), !isTauri() && (_jsx("div", { style: { padding: 8, background: "rgba(239,68,68,0.12)", border: "1px solid var(--color-error)", borderRadius: 6, fontSize: 12, color: "var(--color-error)", marginBottom: 12 }, children: "\uD83D\uDCA1 \u5F53\u524D\u5728\u6D4F\u89C8\u5668\u9884\u89C8\u91CC\u8FD0\u884C \u2014 AI \u8F6C\u8C31\u4F9D\u8D56 Rust \u540E\u7AEF, \u9700\u8981\u7528 Tauri \u684C\u9762\u7248\u624D\u80FD\u7528. \u4E0D\u8FC7 Lo-Fi / Trap / \u6C11\u8C23 / \u6D69\u5BA4 \u7B49 22 \u79CD\u98CE\u683C\u7684**\u81EA\u52A8\u7F16\u66F2/\u548C\u5F26\u751F\u6210/\u9F13\u70B9\u8BC6\u522B**\u90FD\u662F\u7EAF\u524D\u7AEF, \u73B0\u5728\u5C31\u80FD\u8DD1!" })), missing.map((m) => {
                                        const downloading = !!store.downloading[m.id];
                                        return (_jsxs("div", { style: { display: "flex", alignItems: "center", gap: 10, padding: "8px 0", borderBottom: "1px solid rgba(255,255,255,0.06)" }, children: [_jsxs("div", { style: { flex: 1 }, children: [_jsx("div", { style: { fontSize: 13, fontWeight: 600, color: "#fff" }, children: m.name.zh }), _jsxs("div", { style: { fontSize: 11, color: "#888" }, children: [m.architecture, " \u00B7 ", m.description.zh] })] }), _jsx("button", { disabled: downloading || !isTauri(), className: "inspector-insert-btn", style: { opacity: downloading || !isTauri() ? 0.4 : 1, cursor: downloading ? "progress" : "pointer" }, onClick: () => { void store.downloadEntry(m); }, children: downloading ? "⏳ 下载中…" : "⬇️ 下载" })] }, m.id));
                                    })] }), _jsxs("div", { style: { padding: "8px 16px 14px", display: "flex", justifyContent: "flex-end", gap: 8 }, children: [_jsx("button", { className: "arrange-btn", onClick: () => setAmtMissingOpen(null), children: "\u7A0D\u540E" }), !missing.some((m) => store.installed.find((i) => (i.id === m.id || i.architecture === m.architecture) && i.is_available)) && isTauri() && (_jsx("button", { className: "arrange-btn", style: { background: "var(--color-error)", borderColor: "var(--color-error)" }, onClick: () => { setAmtTarget(amtMissingOpen); setAmtMissingOpen(null); }, children: "\u4ECD\u7136\u5C1D\u8BD5\u6253\u5F00 (\u53EF\u80FD\u4F1A\u62A5\u9519)" })), missing.length === 0 && isTauri() && (_jsx("button", { className: "arrange-btn", style: { background: "var(--color-success)", borderColor: "var(--color-success)" }, onClick: () => { setAmtTarget(amtMissingOpen); setAmtMissingOpen(null); }, children: "\u2705 \u5168\u90E8\u5C31\u7EEA \u2014 \u6253\u5F00\u8F6C\u8C31" }))] })] }) }));
            })(), arrangeTarget && (_jsx(ArrangeDialog, { trackId: arrangeTarget, onClose: () => setArrangeTarget(null) })), chordMidiTarget && (_jsx(ChordMidiDialog, { trackId: chordMidiTarget, onClose: () => setChordMidiTarget(null) })), colorPicker && (_jsx(TrackColorPicker, { currentColor: tracks.find((t) => t.id === colorPicker.trackId)?.color || trackTypeCssVar(tracks.find((t) => t.id === colorPicker.trackId)?.trackType || "audio"), onColorChange: (color) => {
                    updateTrack(colorPicker.trackId, { color });
                }, onClose: () => setColorPicker(null), position: { x: colorPicker.x, y: colorPicker.y } }))] }));
}
function TrackItem({ track, index, childCount, vZoom, active, dragging, editing, onHeaderMouseDown, onStartRename, onCommitRename, onCancelRename, onSelect, onResizePointerDown, onMute, onSolo, onVolumeChange: _onVolumeChange, onPanChange: _onPanChange, onToggleExpand, onTogglePlayOriginal, onLaneMute, onLaneVolumeChange, onLanePanChange, onOpenColorPicker, hasOov, hasUnknownPhone, hasDropped, hasShort, hasAliasHint, }) {
    const { t } = useTranslation();
    const colorVar = track.color || trackTypeCssVar(track.trackType);
    const rendering = useAppStore((s) => s.renderingVocalTrackId) === track.id; // ② spinner while this track re-renders
    // 属性面板态(参数按钮高亮)——必须订阅而非 getState(),否则不触发重渲染
    const inspectorOpen = useAppStore((s) => s.inspectorTrackId === track.id);
    const lanes = getLanes(track);
    const laneLayout = getLaneLayout(track);
    const hasLanes = lanes.length > 0;
    const totalHeight = computeTrackHeight(track, vZoom);
    const isEmpty = track.segments.length === 0;
    // The left "indicator light" comment above described the old light; dimming now uses the
    // `.track-item-group.empty` class (see CSS).
    // ── 文件夹分组轨: 简化头部(折叠箭头 + 文件夹图标 + 名称 + 子轨数), 无推子/按钮 ──
    if (track.isFolder) {
        return (_jsx("div", { className: `track-item-group folder ${active ? "active" : ""}`, style: { height: totalHeight }, children: _jsxs("div", { className: `track-item folder-track ${track.folderCollapsed ? "collapsed" : ""}`, onClick: onSelect, style: { height: TRACK_HEADER_HEIGHT * vZoom }, children: [_jsx("div", { className: "track-color-bar", style: { background: colorVar, cursor: "pointer" }, onClick: (e) => {
                            e.stopPropagation();
                            const r = e.currentTarget.getBoundingClientRect();
                            onOpenColorPicker(r.right + 4, r.top);
                        }, title: "\u70B9\u51FB\u9009\u62E9\u8F68\u9053\u989C\u8272" }), _jsx("button", { className: "track-expand-btn", onClick: (e) => {
                            e.stopPropagation();
                            useProjectStore.getState().updateTrack(track.id, { folderCollapsed: !track.folderCollapsed });
                        }, title: track.folderCollapsed ? "展开分组" : "折叠分组", children: track.folderCollapsed ? "▶" : "▼" }), _jsx("svg", { className: "folder-glyph", viewBox: "0 0 24 24", width: "16", height: "16", fill: "currentColor", "aria-hidden": "true", children: _jsx("path", { d: "M3 5a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5z" }) }), _jsx("div", { className: "track-info", children: editing ? (_jsx(RenameInput, { initial: track.name, onCommit: onCommitRename, onCancel: onCancelRename })) : (_jsx("span", { className: "track-name", title: track.name, onDoubleClick: (e) => { e.stopPropagation(); onStartRename(); }, children: track.name })) }), _jsxs("span", { className: "folder-count", children: [childCount ?? 0, " \u8F68"] })] }) }));
    }
    return (_jsxs("div", { className: `track-item-group ${active ? "active" : ""} ${isEmpty ? "empty" : ""} ${dragging ? "dragging" : ""} ${track.playOriginal ? "play-original" : ""}`, style: { height: totalHeight }, children: [_jsxs("div", { className: `track-item ${track.trackType}-track`, onClick: onSelect, style: { height: TRACK_HEADER_HEIGHT * vZoom }, children: [_jsx("div", { className: "track-color-bar", style: { background: colorVar, cursor: 'pointer' }, onClick: (e) => {
                            e.stopPropagation();
                            const r = e.currentTarget.getBoundingClientRect();
                            onOpenColorPicker(r.right + 4, r.top);
                        }, title: "\u70B9\u51FB\u9009\u62E9\u8F68\u9053\u989C\u8272" }), hasLanes && (_jsx("button", { className: "track-expand-btn", onClick: (e) => { e.stopPropagation(); onToggleExpand(); }, title: track.expanded ? "折叠" : "展开", children: track.expanded ? "▼" : "▶" })), _jsxs("div", { className: "track-main", children: [_jsxs("div", { className: "track-row-top", children: [_jsx("span", { className: "track-no-badge", style: { background: colorVar }, title: `轨道 ${index + 1}`, children: index + 1 }), _jsx("button", { className: `track-state-btn ${track.muted ? "active-mute" : ""}`, onClick: (e) => { e.stopPropagation(); onMute(); }, title: "\u9759\u97F3", children: "M" }), _jsx("button", { className: `track-state-btn ${track.solo ? "active-solo" : ""}`, onClick: (e) => { e.stopPropagation(); onSolo(); }, title: "\u72EC\u594F", children: "S" }), _jsx("button", { className: `track-state-btn ${inspectorOpen ? "active-inspector" : ""}`, onClick: (e) => { e.stopPropagation(); useAppStore.getState().toggleInspector(track.id); }, title: "\u8F68\u9053\u53C2\u6570", children: _jsxs("svg", { viewBox: "0 0 16 16", width: "13", height: "13", fill: "none", stroke: "currentColor", strokeWidth: "1.5", strokeLinecap: "round", children: [_jsx("line", { x1: "4", y1: "2", x2: "4", y2: "14" }), _jsx("line", { x1: "2", y1: "5", x2: "6", y2: "5" }), _jsx("line", { x1: "8", y1: "2", x2: "8", y2: "14" }), _jsx("line", { x1: "6", y1: "10", x2: "10", y2: "10" }), _jsx("line", { x1: "12", y1: "2", x2: "12", y2: "14" }), _jsx("line", { x1: "10", y1: "7", x2: "14", y2: "7" })] }) }), hasLanes && (_jsx("button", { className: `track-state-btn track-src-btn ${track.playOriginal ? "active-src" : ""}`, onClick: (e) => { e.stopPropagation(); onTogglePlayOriginal(); }, title: track.playOriginal ? "正在播放原始音频 (子轨不进输出) — 点击切回子轨" : "正在播放子轨 — 点击切到原始音频", children: track.playOriginal ? "SRC" : "SUB" })), _jsx("div", { className: "track-info", children: editing ? (_jsx(RenameInput, { initial: track.name, onCommit: onCommitRename, onCancel: onCancelRename })) : (_jsx("span", { className: "track-name", title: track.name, onDoubleClick: (e) => { e.stopPropagation(); onStartRename(); }, children: track.name })) }), rendering && (_jsx("span", { className: "track-render-spinner", title: t("vocalEditor.render.rendering"), children: _jsx("svg", { viewBox: "0 0 24 24", width: "12", height: "12", children: _jsx("path", { fill: "none", stroke: "currentColor", strokeWidth: "3", strokeLinecap: "round", d: "M12 3a9 9 0 1 0 9 9" }) }) })), (hasOov || hasUnknownPhone || hasDropped || hasShort || hasAliasHint) && (_jsx("span", { className: hasOov || hasUnknownPhone || hasDropped ? "track-oov-badge" : "track-oov-badge advisory", title: [
                                            hasOov ? t("tracks.oovWarning") : null,
                                            hasUnknownPhone ? t("tracks.unknownPhoneWarning") : null,
                                            hasDropped ? t("tracks.droppedWarning") : null,
                                            hasShort ? t("tracks.shortWarning") : null,
                                            hasAliasHint ? t("tracks.aliasHintWarning") : null,
                                        ]
                                            .filter(Boolean)
                                            .join("\n"), children: _jsx("svg", { viewBox: "0 0 24 24", width: "12", height: "12", children: _jsx("path", { fill: "currentColor", d: "M12 3 2 21h20L12 3zm-1 7h2v6h-2v-6zm0 7h2v2h-2v-2z" }) }) }))] }), _jsxs("div", { className: "track-row-volume", children: [_jsx(VolumeFader, { value: track.volumeDb, min: FADER_MIN_DB, max: FADER_MAX_DB, onChange: _onVolumeChange, onGestureStart: () => useHistoryStore.getState().beginTransaction(), onGestureEnd: () => useHistoryStore.getState().commitTransaction() }), _jsx("input", { type: "text", className: "track-volume-input", value: track.volumeDb > FADER_MIN_DB ? `${track.volumeDb > 0 ? "+" : ""}${track.volumeDb.toFixed(1)}` : "-∞", onChange: (e) => {
                                            const val = e.target.value.trim();
                                            if (val === "-∞" || val === "-inf") {
                                                _onVolumeChange(FADER_MIN_DB);
                                            }
                                            else {
                                                const num = parseFloat(val);
                                                if (!isNaN(num)) {
                                                    _onVolumeChange(Math.max(FADER_MIN_DB, Math.min(FADER_MAX_DB, num)));
                                                }
                                            }
                                        }, onFocus: (e) => e.target.select(), onClick: (e) => e.stopPropagation(), title: "\u70B9\u51FB\u8F93\u5165\u97F3\u91CF(dB)" })] }), _jsx("div", { className: "track-row-input", children: _jsxs("button", { className: "track-input-label-btn", onClick: (e) => {
                                        e.stopPropagation();
                                        // TODO: 打开输入源选择菜单
                                        console.log("点击输入源选择");
                                    }, title: "\u70B9\u51FB\u9009\u62E9\u8F93\u5165\u6E90", children: [_jsx("span", { className: "track-input-label", children: "\u8F93\u5165 L+R \u7ACB\u4F53\u58F0" }), _jsxs("div", { className: "track-input-icons", children: [_jsxs("svg", { viewBox: "0 0 24 24", width: "14", height: "14", fill: "none", stroke: "currentColor", strokeWidth: "2", children: [_jsx("circle", { cx: "8", cy: "12", r: "3" }), _jsx("circle", { cx: "16", cy: "12", r: "3" }), _jsx("line", { x1: "8", y1: "12", x2: "16", y2: "12" })] }), _jsx("svg", { viewBox: "0 0 24 24", width: "10", height: "10", fill: "currentColor", style: { marginLeft: "2px" }, children: _jsx("path", { d: "M7 10l5 5 5-5z" }) })] })] }) })] }), _jsx("div", { className: "track-waveform-thumb", children: _jsx(TrackWaveform, { track: track, height: TRACK_HEADER_HEIGHT * vZoom }) }), _jsx("div", { className: "track-light", onMouseDown: onHeaderMouseDown, title: "\u62D6\u52A8\u91CD\u65B0\u6392\u5217\u8F68\u9053" })] }), track.expanded && laneLayout.runs.map((run) => {
                // One GROUP BLOCK per 组+名 run (getLaneLayout — same geometry the canvas rows use): a slim
                // group BAR carrying the 轨道组 name + the group-level volume/pan (keyed by the 组 via
                // laneControlFor — all rows of one 组 share the mix; 解组 for independent control), then the
                // member rows with just the stem name + per-ROW mute (isLaneRowMuted, loose row semantics).
                const ctrl = laneControlFor(track, run.groupId, run.laneId);
                const laneRgb = LANE_COLORS[run.colorIndex % LANE_COLORS.length];
                return (_jsxs("div", { className: "lane-group", style: { "--lane-rgb": laneRgb }, children: [_jsxs("div", { className: "lane-group-bar", style: { height: LANE_GROUP_BAR_HEIGHT * vZoom }, children: [_jsx("span", { className: "lane-group-swatch" }), track.trackType === "audio" && (_jsx("button", { className: `track-btn lane-db-btn ${track.laneLoudnessOpen?.[run.groupId] ? "active-orig" : ""}`, title: t("tracks.loudnessLane"), onClick: (e) => { e.stopPropagation(); useProjectStore.getState().toggleLaneLoudnessOpen(track.id, run.groupId); }, children: "dB" })), _jsx("span", { className: "lane-group-name", title: run.name, children: run.name }), _jsxs("div", { className: "track-controls lane-group-controls", children: [_jsxs("div", { className: "fader-row", children: [_jsx("span", { className: "fader-tag", children: "V" }), _jsx(VolumeFader, { value: ctrl?.volumeDb ?? 0, min: FADER_MIN_DB, max: FADER_MAX_DB, width: 42, onChange: (v) => onLaneVolumeChange(run, v), onGestureStart: () => useHistoryStore.getState().beginTransaction(), onGestureEnd: () => useHistoryStore.getState().commitTransaction() }), _jsx("span", { className: "fader-val", children: formatDb(ctrl?.volumeDb ?? 0, FADER_MIN_DB) })] }), _jsxs("div", { className: "fader-row", children: [_jsx("span", { className: "fader-tag", children: "P" }), _jsx(VolumeFader, { value: ctrl?.pan ?? 0, min: -1, max: 1, step: 0.1, fillFrom: "center", format: formatPan, width: 28, onChange: (v) => onLanePanChange(run, v), onGestureStart: () => useHistoryStore.getState().beginTransaction(), onGestureEnd: () => useHistoryStore.getState().commitTransaction() }), _jsx("span", { className: "fader-val", children: formatPan(ctrl?.pan ?? 0) })] })] })] }), lanes.slice(run.start, run.start + run.count).map(({ id, label, members }) => {
                            // A merged row reads muted only when ALL members are (the toggle fans out, so they
                            // only diverge via legacy state — the canvas dims each piece by its own member truth).
                            const muted = members.every((m) => isLaneRowMuted(track, m.rowKey, m.laneId));
                            // Show only the sub-name within the group (labels are "Group · stem") — the group name
                            // lives on the bar above, and the bracket ties the rows to it (members, not peers).
                            const subName = laneLabelParts(label).stem ?? label;
                            return (_jsxs("div", { className: "lane-item", style: { height: LANE_HEIGHT * vZoom }, children: [_jsx("span", { className: "lane-label", title: label, children: subName }), _jsx("div", { className: "track-controls", children: _jsx("button", { className: `track-btn ${muted ? "active-mute" : ""}`, onClick: (e) => { e.stopPropagation(); onLaneMute(members); }, children: "M" }) })] }, id));
                        })] }, run.key));
            }), loudnessBandH(track) > 0 && (_jsx("div", { className: "loudness-band-header", style: { height: LOUDNESS_LANE_HEIGHT * vZoom }, children: _jsx("span", { className: "loudness-band-label", children: t("tracks.loudnessLane") }) })), _jsx("div", { className: "track-resize-handle", onPointerDown: onResizePointerDown, title: "\u62D6\u62FD\u8C03\u6574\u8F68\u9053\u9AD8\u5EA6" })] }));
}
/** Inline track-name editor. Commits once on blur (Enter blurs → commit; Escape cancels). */
function RenameInput({ initial, onCommit, onCancel }) {
    const ref = useRef(null);
    const doneRef = useRef(false);
    return (_jsx("input", { ref: ref, className: "track-name-input", autoFocus: true, defaultValue: initial, onMouseDown: (e) => e.stopPropagation(), onClick: (e) => e.stopPropagation(), onKeyDown: (e) => {
            if (e.key === "Enter")
                ref.current?.blur();
            else if (e.key === "Escape") {
                doneRef.current = true;
                onCancel();
            }
        }, onBlur: (e) => { if (doneRef.current)
            return; doneRef.current = true; onCommit(e.target.value); } }));
}
