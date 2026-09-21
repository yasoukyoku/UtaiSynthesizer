import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import React, { useEffect, useRef, useState, useMemo } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useTranslation } from "react-i18next";
import { useAppStore } from "../../store/app";
import { useProjectStore } from "../../store/project";
import { useHistoryStore } from "../../store/history";
import { useAudioStore } from "../../store/audio";
import { matchInstrumentWav } from "../../lib/amtSource";
import { SynchronizedPlayer } from "../../lib/amt/syncPlayer";
import * as playback from "../../lib/audio/playback";
import { distributeLyricsToNotes } from "../../lib/amt/lyricDistribute";
import { smartCleanStems, cleanChangedTotal } from "../../lib/midiCleanup";
import { isVocalTrackName, VOCAL_STEM_NAME } from "../../lib/gmInstruments";
import { ContextMenu } from "../common/ContextMenu";
import { AmtLyricsPanel } from "./AmtLyricsPanel";
import "./AmtResultPanel.css";
const STEM_COLORS = [
    "#00e5ff", // Neon Cyan
    "#ff4081", // Pink
    "#7c4dff", // Purple
    "#00e676", // Green
    "#ffea00", // Yellow
    "#ff9100", // Orange
    "#f44336", // Red
    "#2196f3", // Blue
];
/** import_score_file 对无歌词 MIDI 每音符填的占位符（现行「啊」/旧版「あ」）—— 全轨只剩它 = 无歌词。 */
const PLACEHOLDER_LYRICS = new Set(["あ", "啊"]);
/**
 * import_score_file 的轨内音符 → 工作台 NoteData（绝对 480-ppq 时间轴 + 歌词贯通）。
 * 真实歌词（≠「あ」占位）全轨透传；否则不带 lyric（乐器轨 → 编辑器显示音名）。
 * 加载与自动歌词写回后的重导共用，保证两条路径行为一致。
 */
function buildNotesFromTrack(rawNotes, startTick, stemIdx) {
    const hasLyrics = rawNotes.some((n) => String(n.lyric ?? "").trim() !== "" && !PLACEHOLDER_LYRICS.has(String(n.lyric ?? "")));
    return rawNotes.map((n, ni) => ({
        id: `${stemIdx}-${ni}`,
        tick: n.tick + startTick,
        duration: n.duration,
        pitch: n.pitch,
        velocity: n.velocity ?? 100,
        lyric: hasLyrics ? String(n.lyric ?? "") || "啊" : undefined,
    }));
}
/** Compact waveform canvas for one separation stem row (peaks + playhead). */
function SepWaveCanvas({ peaks, color, active, playFrac }) {
    const canvasRef = useRef(null);
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas)
            return;
        const ctx = canvas.getContext("2d");
        if (!ctx)
            return;
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        const w = Math.floor(rect.width * dpr);
        const h = Math.floor(rect.height * dpr);
        if (canvas.width !== w || canvas.height !== h) {
            canvas.width = w;
            canvas.height = h;
        }
        if (w === 0 || h === 0)
            return;
        ctx.clearRect(0, 0, w, h);
        const midY = h / 2;
        // Backdrop
        ctx.fillStyle = active ? "#131824" : "#0d1117";
        ctx.fillRect(0, 0, w, h);
        ctx.strokeStyle = "rgba(255,255,255,0.06)";
        ctx.lineWidth = 1 * dpr;
        ctx.beginPath();
        ctx.moveTo(0, midY);
        ctx.lineTo(w, midY);
        ctx.stroke();
        if (peaks && peaks.length > 0) {
            const numCols = Math.min(peaks.length, Math.floor(w / (2 * dpr)) || 1);
            const colWidth = w / Math.max(1, numCols);
            const per = peaks.length / numCols;
            const amp = h / 2 - 3 * dpr;
            for (let c = 0; c < numCols; c++) {
                const start = Math.floor(c * per);
                const end = Math.min(peaks.length, Math.max(start + 1, Math.floor((c + 1) * per)));
                let p = 0;
                for (let i = start; i < end; i++) {
                    const v = Math.abs(peaks[i] ?? 0);
                    if (v > p)
                        p = v;
                }
                const isPast = c / numCols <= playFrac;
                ctx.fillStyle = isPast ? color : "rgba(96,165,250,0.45)";
                const barH = Math.max(2 * dpr, p * amp);
                const barW = Math.max(1 * dpr, colWidth - 1 * dpr);
                ctx.fillRect(c * colWidth, midY - barH, barW, barH * 2);
            }
        }
        else {
            ctx.strokeStyle = "rgba(0,229,255,0.3)";
            ctx.setLineDash([4 * dpr, 4 * dpr]);
            ctx.beginPath();
            ctx.moveTo(0, midY);
            ctx.lineTo(w, midY);
            ctx.stroke();
            ctx.setLineDash([]);
        }
        // Playhead
        if (playFrac > 0) {
            const px = playFrac * w;
            ctx.strokeStyle = "#00e5ff";
            ctx.lineWidth = 1.5 * dpr;
            ctx.beginPath();
            ctx.moveTo(px, 0);
            ctx.lineTo(px, h);
            ctx.stroke();
        }
    }, [peaks, color, active, playFrac]);
    return _jsx("canvas", { ref: canvasRef, className: "amt-sep-wave-canvas" });
}
export function AmtResultPanel({ result, onClose, style }) {
    const { t } = useTranslation();
    const showToast = useAppStore((s) => s.showToast);
    const addTrack = useProjectStore((s) => s.addTrack);
    const [stems, setStems] = useState([]);
    const [activeStemIdx, setActiveStemIdx] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [isLoading, setIsLoading] = useState(true);
    const [playSource, setPlaySource] = useState("midi");
    const [bpm, setBpm] = useState(120);
    const [ppq, setPpq] = useState(480);
    // ── Tempo (reference: set_bpm_context / bpm_spin / speed_spin) ──
    // `bpm` = DETECTED source BPM — drives every tick→second mapping (grid, roll,
    // duration) and NEVER changes. `projBpm`/`speed` are the linked user controls:
    // rate = projBpm / detectedBpm applies LIVE to the synchronized player.
    const [detectedBpm, setDetectedBpm] = useState(null);
    const [projBpm, setProjBpm] = useState(120);
    const [speed, setSpeed] = useState(1);
    // ── MIDI editor (reference: editor_panel) ──
    const [editMode, setEditMode] = useState(false);
    const [selNotes, setSelNotes] = useState(new Set());
    const [activeInstr, setActiveInstr] = useState("");
    const [velSpin, setVelSpin] = useState(100);
    const [quantScope, setQuantScope] = useState("all_tracks");
    /** Quantize grid denominator (4,8,16,32,64 → 1/4 … 1/64). Reference default: 1/32. */
    const [quantGrid, setQuantGrid] = useState(32);
    const [editRendering, setEditRendering] = useState(false);
    const [editDirty, setEditDirty] = useState(false);
    const [followMode, setFollowMode] = useState(true);
    // ── Download menu (reference download_menu: MIDI / 乐谱 / 转录WAV / 分轨WAV / 立体声) ──
    const [downloadOpen, setDownloadOpen] = useState(false);
    const downloadRef = useRef(null);
    // ── Lyrics drawer (P0-B: Whisper extraction → edit → write back to MIDI) ──
    const [lyricsOpen, setLyricsOpen] = useState(false);
    const [activeTab, setActiveTab] = useState("edit");
    /** Which export is rendering — disables the whole menu and shows busy on the trigger. */
    const [exportBusy, setExportBusy] = useState(null);
    /** Backend tool availability (drives menu-item enable/disable like the reference). */
    const [exportTools, setExportTools] = useState(null);
    // ── SoundFont switcher (P0-C 多音源挂载) ──
    const [soundfonts, setSoundfonts] = useState([]);
    const [sfOpen, setSfOpen] = useState(false);
    const sfMenuRef = useRef(null);
    const originalStemsRef = useRef(null);
    const undoRef = useRef([]);
    const redoRef = useRef([]);
    const clipboardRef = useRef([]);
    /** Marquee selection rect in roll-canvas pixels (reference rubber-band select). */
    const [marquee, setMarquee] = useState(null);
    /** Note drag/resize in progress: original + draft geometry per note id. */
    const dragRef = useRef(null);
    /** Redraw counter bumped while dragging (drafts are rendered from the ref). */
    const [dragRev, setDragRev] = useState(0);
    /** Frozen source-seconds position — the player re-seeks here after every reconfigure. */
    const frozenPosRef = useRef(0);
    const rateRef = useRef(1);
    /** Lazy FluidSynth re-synthesis results: filled when the dialog's best-effort
     *  playback prep failed and left audioPreview/originalAudio empty. */
    const [prepAudio, setPrepAudio] = useState({});
    const [preparingAudio, setPreparingAudio] = useState(false);
    /** §user "干音都要有歌词"：每个 result 只自动提取一次歌词（防循环 / 防重复跑 Whisper）。 */
    const autoLyricsRef = useRef(false);
    /** 一键修复差异高亮：被修过且留存的音符 id → 6 秒内描琥珀色描边。 */
    const [repairHighlight, setRepairHighlight] = useState(new Set());
    const repairHighlightTimer = useRef(null);
    /** 快捷键速查卡（§建议4）：「?」按钮弹出，随时可查钢琴卷帘的全部操作。 */
    const [rollHelpOpen, setRollHelpOpen] = useState(false);
    /** 音符右键菜单（右键菜单统一）：右键不再直接删除，而是就近弹出常用操作。 */
    const [rollMenu, setRollMenu] = useState(null);
    /** §用户「双击音符改歌词」：正在编辑歌词的音符（canvas 浮层输入框）。 */
    const [lyricEdit, setLyricEdit] = useState(null);
    const lyricEditRef = useRef(lyricEdit);
    lyricEditRef.current = lyricEdit;
    /** Effective playable sources (dialog-provided first, lazily-synthesized fallback). */
    const effPreview = result.audioPreview || prepAudio.preview || "";
    const effOriginal = result.originalAudio || prepAudio.original || "";
    /** Separation-only mode: audio stems, no MIDI. Renders per-stem waveform rows. */
    const isSeparation = !!result.separationOnly ||
        (result.midiPaths.length === 0 && Object.values(result.stems ?? {}).some((s) => s?.audio));
    /** Per-stem waveform peaks for separation view. */
    const [stemPeaks, setStemPeaks] = useState({});
    // Top playable "track": waveform of the file currently being played (MIDI synthesis / original).
    const [masterPeaks, setMasterPeaks] = useState([]);
    const [masterDur, setMasterDur] = useState(0);
    const [masterHover, setMasterHover] = useState(false);
    const [masterHoverFrac, setMasterHoverFrac] = useState(0);
    const masterCanvasRef = useRef(null);
    const canvasRef = useRef(null);
    const viewportRef = useRef(null);
    /** Piano-roll zoom (0.3×–3×): scales BOTH the horizontal (pixels/second) and the
     *  vertical (row height) axes so notes and piano keys grow/shrink together. */
    const [zoom, setZoom] = useState(1);
    const clampZoom = (z) => Math.min(3, Math.max(0.3, z));
    const ROW_HEIGHT = Math.max(8, Math.round(18 * zoom));
    const PIXELS_PER_SECOND = Math.max(30, Math.round(120 * zoom));
    const PIANO_WIDTH = 60;
    // Load MIDI data and metadata
    useEffect(() => {
        autoLyricsRef.current = false; // 新 result / 重开面板 → 允许再自动提取一次
        async function loadData() {
            setIsLoading(true);
            try {
                // Playback previews come from convertFileSrc() <audio> URLs. A right-click
                // conversion's outputs (transcription/original/instrument WAVs) live NEXT TO the
                // source file — outside the static asset-protocol scope — so ask the backend to
                // runtime-allow that folder BEFORE any audio element tries to load from it.
                if (result.outputDir) {
                    await invoke("allow_asset_dir", { dir: result.outputDir }).catch(() => { });
                }
                const loadedStems = [];
                let maxTick = 0;
                let vocalCount = 0;
                // Merged per-instrument WAV map: dialog-provided stems + lazily re-synthesized
                // ones (prepAudio.wavs) — either may be the only populated source.
                const wavMap = {};
                for (const [k, v] of Object.entries(result.stems ?? {})) {
                    if (v?.audio)
                        wavMap[k] = v.audio;
                }
                for (const [k, v] of Object.entries(prepAudio.wavs ?? {})) {
                    if (v)
                        wavMap[k] = v;
                }
                // Separation-only result (vocal/six-stem split): build audio-only stems and
                // take the duration from the first stem's audio metadata.
                if (isSeparation) {
                    const sepEntries = Object.entries(result.stems ?? {}).filter(([, v]) => !!v?.audio);
                    let sepIdx = 0;
                    for (const [name, v] of sepEntries) {
                        loadedStems.push({
                            name,
                            notes: [],
                            midiPath: "",
                            audioPath: v.audio,
                            muted: false,
                            soloed: false,
                            color: STEM_COLORS[sepIdx % STEM_COLORS.length] || "#00e5ff",
                        });
                        sepIdx++;
                    }
                    setStems(loadedStems);
                    // Per-stem peaks for the waveform rows + authoritative duration.
                    const peaksMap = {};
                    let sepDur = 0;
                    for (const st of loadedStems) {
                        if (!st.audioPath)
                            continue;
                        try {
                            const data = await useAudioStore.getState().loadAudioFile(st.audioPath);
                            if (data.peaks && data.peaks.length > 0)
                                peaksMap[st.name] = data.peaks;
                            if (data.durationMs > 0 && data.durationMs / 1000 > sepDur)
                                sepDur = data.durationMs / 1000;
                        }
                        catch { /* best-effort waveform */ }
                    }
                    setStemPeaks(peaksMap);
                    if (sepDur > 0)
                        setDuration(sepDur);
                    return;
                }
                // Fetch metadata from the first MIDI file (main one usually)
                if (result.midiPaths.length > 0) {
                    try {
                        const meta = await invoke("amt_midi_metadata", { midiPath: result.midiPaths[0] });
                        if (meta.bpm) {
                            setBpm(meta.bpm);
                            setDetectedBpm(meta.bpm);
                            setProjBpm(meta.bpm);
                        }
                        if (meta.ppq)
                            setPpq(meta.ppq);
                    }
                    catch (e) {
                        console.warn("Failed to fetch MIDI metadata", e);
                    }
                }
                for (let i = 0; i < result.midiPaths.length; i++) {
                    const path = result.midiPaths[i];
                    const score = await invoke("import_score_file", { path });
                    // Per-file GM metadata: gives program+channel per track so we can map a
                    // track onto the sidecar's "gm:NNN"/"drums" WAV keys (names don't match).
                    let metaTracks = [];
                    try {
                        const meta = await invoke("amt_midi_metadata", { midiPath: path });
                        metaTracks = meta?.tracks ?? [];
                    }
                    catch { /* best-effort — name matching still runs below */ }
                    for (const track of score.tracks) {
                        const mt = metaTracks.find((m) => m.name === track.name) || metaTracks[0];
                        let audioPath = matchInstrumentWav(wavMap, track.name || "", mt?.program ?? null, mt?.channel ?? null);
                        // import_score_file REBASES each track to its first note (tick 0) —
                        // add start_tick back so every stem lives on ONE absolute 480-ppq
                        // timeline (display, playhead and edit write-back all share it).
                        const startTick = track.start_tick || 0;
                        const rawNotes = track.notes;
                        const notes = buildNotesFromTrack(rawNotes, startTick, loadedStems.length);
                        // §user "演唱的干音都命名 vocal"：人声干音轨（不管模型给起的名是
                        // "Voice"/"Singing Voice"/"人声"…）统一改叫 Vocal，第 2 条起加序号。
                        // 命名在 WAV 匹配之后做，不影响 matchInstrumentWav 的原名匹配。
                        const rawName = track.name || path.split(/[/\\]/).pop()?.replace(".midi", "").replace(".mid", "") || `${t("amt.tracks")} ${i + 1}`;
                        const displayName = isVocalTrackName(rawName)
                            ? (vocalCount === 0 ? VOCAL_STEM_NAME : `${VOCAL_STEM_NAME} ${vocalCount + 1}`)
                            : rawName;
                        if (isVocalTrackName(rawName))
                            vocalCount++;
                        // §用户「干声转MIDI试听要是人声」：干音轨没匹配到独立 bus 时挂干音本身
                        //（转换源就是人声）——纯人声结果的播放器配置据此用干音做 MIDI 侧音源。
                        if (!audioPath) {
                            const dryVocal = result.vocalAudioPath || result.sourceAudioPath || result.originalAudio;
                            if (dryVocal && isVocalTrackName(rawName))
                                audioPath = dryVocal;
                        }
                        loadedStems.push({
                            name: displayName,
                            notes,
                            midiPath: path,
                            audioPath,
                            muted: false,
                            soloed: false,
                            color: STEM_COLORS[loadedStems.length % STEM_COLORS.length] || "#00e5ff",
                            program: mt?.program ?? null,
                            channel: mt?.channel ?? null,
                            midiTrackIdx: score.tracks.indexOf(track),
                        });
                        const trackMax = notes.reduce((max, n) => Math.max(max, n.tick + n.duration), 0);
                        maxTick = Math.max(maxTick, trackMax);
                    }
                }
                setStems(loadedStems);
                // Editor baseline ("还原" target + dirty comparison) — deep snapshot.
                originalStemsRef.current = loadedStems.map((s) => ({
                    ...s,
                    notes: s.notes.map((n) => ({ ...n })),
                }));
                // First detected instrument = the editor's initial active instrument.
                const firstStem = loadedStems[0];
                if (firstStem)
                    setActiveInstr(firstStem.name);
                // Authoritative duration from the transcribed note ticks, in our 480-ppq space:
                //   seconds = maxTick / (480 * (bpm/60)). This stays exactly aligned with the
                //   note→pixel mapping below, so playhead, timeline width and note positions
                //   always line up. (The synthesized transcription audio plays this same tempo,
                //   so the two stay in sync; we do NOT read it from the audio metadata, which
                //   avoids bogus values like a 1-hour duration.)
                const safePpq = ppq > 0 ? ppq : 480;
                const safeBpm = bpm > 0 ? bpm : 120;
                setDuration(maxTick / (safePpq * (safeBpm / 60)));
                setPpq(safePpq);
            }
            catch (e) {
                showToast(String(e), "error");
            }
            finally {
                setIsLoading(false);
            }
        }
        loadData();
    }, [result, showToast]);
    // ── §user "只要是干音，不管名字是什么，都要有歌词" ─────────────────────────────
    // 结果工作台加载后：发现人声干音轨（已统一命名 Vocal）且整份结果无任何真实歌词时，
    // 自动跑一次 Whisper 提取 → 写回干音轨的 MIDI → 按 (midiPath, trackIdx) 重导同一条轨，
    // 把歌词铺到音符上（导入 DAW 后编辑器打开即歌词）。每个 result 只跑一次。
    useEffect(() => {
        if (isSeparation || isLoading || autoLyricsRef.current)
            return;
        const vocalStems = stems.filter((s) => isVocalTrackName(s.name) && s.midiPath);
        if (vocalStems.length === 0)
            return;
        const anyLyrics = stems.some((s) => s.notes.some((n) => String(n.lyric ?? "").trim() !== ""));
        if (anyLyrics)
            return;
        autoLyricsRef.current = true;
        let alive = true;
        (async () => {
            try {
                // 1. 已装 Whisper 模型里挑最大的（小模型在歌词任务上明显更稳）。
                const installed = await invoke("list_amt_models");
                const sizes = (installed ?? [])
                    .filter((m) => m.filename.startsWith("whisper/") && m.is_installed)
                    .map((m) => m.filename.split("/")[1] ?? "")
                    .filter(Boolean);
                if (!alive)
                    return;
                const rank = { tiny: 0, base: 1, small: 2, medium: 3 };
                const model = sizes.reduce((best, cur) => ((rank[cur] ?? -1) >= (rank[best] ?? -1) ? cur : best), sizes[0] ?? "");
                if (!model) {
                    showToast(t("amt.autoLyricsNoModel") || "人声轨暂无歌词：请先在资源管理下载 Whisper 模型，再到歌词面板提取", "info");
                    return;
                }
                showToast(t("amt.autoLyricsRunning") || "检测到人声干音轨，正在自动提取歌词…", "info");
                // 2. 提取源：干音 WAV（人声分离产物）最准；没有才退回整曲混音。
                const dryVocal = result.vocalAudioPath;
                const source = dryVocal || result.sourceAudioPath || result.originalAudio || "";
                if (!source)
                    return;
                const res = await invoke("amt_extract_lyrics", {
                    audioPath: source,
                    outputDir: result.outputDir,
                    model,
                    language: "",
                });
                const lines = (res.segments ?? []).filter((l) => l.text.trim().length > 0);
                if (!alive || lines.length === 0)
                    return;
                // 3. §user「Whisper 提取后按音节切分铺到每个音符」：按每条干音轨自己的
                //    音符起点把句级歌词切成音节逐个对齐（每个音符一个词），再写回各自的
                //    MIDI（多文件各自写，互不覆盖）。没有音符信息时回退为句级写回。
                const targets = [...new Set(vocalStems.map((s) => s.midiPath))];
                const safePpqL = ppq > 0 ? ppq : 480;
                const safeBpmL = bpm > 0 ? bpm : 120;
                const secPerTick = 1 / (safePpqL * (safeBpmL / 60));
                let noteCount = 0;
                for (const midiPath of targets) {
                    const onsets = vocalStems
                        .filter((s) => s.midiPath === midiPath)
                        .flatMap((s) => s.notes.map((n) => n.tick * secPerTick));
                    const perNote = onsets.length > 0 ? distributeLyricsToNotes(lines, onsets) : lines;
                    noteCount += perNote.length;
                    await invoke("amt_write_lyrics_to_midi", {
                        midiPath,
                        outPath: midiPath,
                        lyrics: perNote.map((l) => ({ start: l.start, text: l.text.trim() })),
                    });
                }
                // 4. 按 (midiPath, trackIdx) 重导干音轨，把歌词铺回音符（保持绝对时间轴）。
                for (const midiPath of targets) {
                    const score = await invoke("import_score_file", { path: midiPath });
                    setStems((prev) => prev.map((s) => {
                        if (s.midiPath !== midiPath || s.midiTrackIdx == null)
                            return s;
                        const track = score.tracks?.[s.midiTrackIdx];
                        if (!track)
                            return s;
                        const startTick = track.start_tick || 0;
                        const notes = buildNotesFromTrack(track.notes ?? [], startTick, prev.indexOf(s));
                        return { ...s, notes };
                    }));
                    originalStemsRef.current = (originalStemsRef.current ?? []).map((s) => {
                        if (s.midiPath !== midiPath || s.midiTrackIdx == null)
                            return s;
                        const track = score.tracks?.[s.midiTrackIdx];
                        if (!track)
                            return s;
                        const startTick = track.start_tick || 0;
                        return { ...s, notes: buildNotesFromTrack(track.notes ?? [], startTick, 0) };
                    });
                }
                showToast(noteCount > 0
                    ? (t("amt.autoLyricsDonePerNote", { count: noteCount }) || `人声歌词已按音符写回（${noteCount} 个音符，可双击改词）`)
                    : (t("amt.autoLyricsDone", { count: lines.length }) || `人声歌词已自动写回（${lines.length} 句）`), "success");
            }
            catch (e) {
                const msg = String(e);
                if (msg.includes("LYRICS_MODEL_NOT_INSTALLED")) {
                    showToast(t("amt.autoLyricsNoModel") || "人声轨暂无歌词：请先在资源管理下载 Whisper 模型，再到歌词面板提取", "info");
                }
                else {
                    showToast(t("amt.autoLyricsFail") || "人声歌词自动提取失败，请打开歌词面板手动提取", "info");
                }
            }
        })();
        return () => { alive = false; };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [stems, isLoading, isSeparation, result, showToast, t]);
    // §user "确定以后也可以更改歌词"：歌词面板写回 MIDI 后，按 (midiPath, trackIdx)
    // 重导对应轨（与上方自动提取写回的同步逻辑一致），音符上的歌词即时刷新。
    const syncLyricsFromMidi = React.useCallback(async (midiPath) => {
        try {
            const score = await invoke("import_score_file", { path: midiPath });
            setStems((prev) => prev.map((s) => {
                if (s.midiPath !== midiPath || s.midiTrackIdx == null)
                    return s;
                const track = score.tracks?.[s.midiTrackIdx];
                if (!track)
                    return s;
                const startTick = track.start_tick || 0;
                const notes = buildNotesFromTrack(track.notes ?? [], startTick, prev.indexOf(s));
                return { ...s, notes };
            }));
            originalStemsRef.current = (originalStemsRef.current ?? []).map((s) => {
                if (s.midiPath !== midiPath || s.midiTrackIdx == null)
                    return s;
                const track = score.tracks?.[s.midiTrackIdx];
                if (!track)
                    return s;
                const startTick = track.start_tick || 0;
                return { ...s, notes: buildNotesFromTrack(track.notes ?? [], startTick, 0) };
            });
        }
        catch { /* best-effort：同步失败不影响已写回的 MIDI 文件 */ }
    }, []);
    // Export tool availability (reference: download actions start disabled, enable when
    // FluidSynth / MuseScore / soundfont resolve). Drives menu-item enable/disable.
    useEffect(() => {
        invoke("amt_export_tools_status")
            .then((s) => setExportTools(s))
            .catch(() => { });
    }, []);
    // SoundFont list (P0-C) — refresh helper shared by switch/import/delete.
    const refreshSoundfonts = React.useCallback(() => {
        invoke("amt_list_soundfonts")
            .then((list) => setSoundfonts(list ?? []))
            .catch(() => { });
    }, []);
    useEffect(() => {
        refreshSoundfonts();
    }, [refreshSoundfonts]);
    // 拖拽全通（§user "SF2 拖进音源列表即装"）：任何位置的 SF2 拖放安装后（Arrangement
    // 统一处理并 emit），这里即时刷新列表——打开中的下拉立即可见、可选。
    useEffect(() => {
        let un;
        let cancelled = false;
        import("@tauri-apps/api/event").then(({ listen }) => listen("amt-soundfont-installed", () => refreshSoundfonts())).then((u) => {
            if (cancelled)
                u();
            else
                un = u;
        }).catch(() => { });
        return () => {
            cancelled = true;
            if (un)
                un();
        };
    }, [refreshSoundfonts]);
    // Click-outside closes the soundfont menu.
    useEffect(() => {
        if (!sfOpen)
            return;
        const onDown = (e) => {
            if (sfMenuRef.current && !sfMenuRef.current.contains(e.target)) {
                setSfOpen(false);
            }
        };
        document.addEventListener("pointerdown", onDown, true);
        return () => document.removeEventListener("pointerdown", onDown, true);
    }, [sfOpen]);
    // Click-outside closes the download menu.
    useEffect(() => {
        if (!downloadOpen)
            return;
        const onDown = (e) => {
            if (downloadRef.current && !downloadRef.current.contains(e.target)) {
                setDownloadOpen(false);
            }
        };
        document.addEventListener("pointerdown", onDown, true);
        return () => document.removeEventListener("pointerdown", onDown, true);
    }, [downloadOpen]);
    // Attach per-instrument WAVs (from the self-heal synthesis or an EDIT re-render)
    // WITHOUT rebuilding the stems — notes are the editor's source of truth and must
    // survive every audio refresh (reference: assets replace, notes persist).
    useEffect(() => {
        if (isSeparation)
            return;
        const wavMap = {};
        for (const [k, v] of Object.entries(result.stems ?? {})) {
            if (v?.audio)
                wavMap[k] = v.audio;
        }
        for (const [k, v] of Object.entries(prepAudio.wavs ?? {})) {
            if (v)
                wavMap[k] = v;
        }
        if (Object.keys(wavMap).length === 0)
            return;
        setStems((prev) => {
            if (prev.length === 0)
                return prev;
            let changed = false;
            const next = prev.map((s) => {
                const audioPath = matchInstrumentWav(wavMap, s.name, s.program ?? null, s.channel ?? null);
                if (audioPath && audioPath !== s.audioPath) {
                    changed = true;
                    return { ...s, audioPath };
                }
                return s;
            });
            return changed ? next : prev;
        });
    }, [result.stems, prepAudio.wavs, isSeparation]);
    // Duration follows the CURRENT notes (edits can extend the last note; the
    // re-rendered audio then matches the new length on the same tick math).
    useEffect(() => {
        if (isSeparation)
            return;
        const safePpq = ppq > 0 ? ppq : 480;
        const safeBpm = bpm > 0 ? bpm : 120;
        const maxTick = stems.reduce((max, s) => s.notes.reduce((m, n) => Math.max(m, n.tick + n.duration), max), 0);
        setDuration(maxTick / (safePpq * (safeBpm / 60)));
    }, [stems, isSeparation, ppq, bpm]);
    // SELF-HEAL: the dialog's playback prep is best-effort and may have failed (leaving
    // audioPreview/originalAudio empty → play button dead). Re-run FluidSynth synthesis here
    // so the workbench ALWAYS ends up playable. Rust absolutizes paths for the sidecar.
    useEffect(() => {
        if (isSeparation)
            return;
        const needPreview = !result.audioPreview;
        const needOriginal = !result.originalAudio;
        const needWavs = !result.stems || Object.values(result.stems).every((s) => !s?.audio);
        if (!needPreview && !needOriginal && !needWavs)
            return;
        const srcAudio = result.sourceAudioPath || result.originalAudio;
        const midi = result.midiPaths[0];
        if (!srcAudio || !midi || !result.outputDir)
            return;
        let cancelled = false;
        setPreparingAudio(true);
        invoke("amt_prepare_playback", {
            midiPath: midi,
            audioPath: srcAudio,
            outputDir: result.outputDir,
        })
            .then((res) => {
            if (cancelled)
                return;
            setPrepAudio({
                preview: res?.transcription_wav || "",
                original: res?.original_wav || "",
                wavs: (res?.instrument_wavs ?? {}),
            });
        })
            .catch((e) => {
            console.warn("Lazy playback prep failed:", e);
        })
            .finally(() => {
            if (!cancelled)
                setPreparingAudio(false);
        });
        return () => {
            cancelled = true;
        };
    }, [result, isSeparation]);
    // Load waveform peaks for the actively-played source file (MIDI synthesis → audioPreview,
    // original → originalAudio) so the top "track" shows what the transport is actually playing.
    useEffect(() => {
        let cancelled = false;
        // Separation view: stems and the original share the same timeline, so the master
        // waveform always shows the ORIGINAL mix regardless of the source switch.
        const src = isSeparation
            ? effOriginal
            : playSource === "midi" ? effPreview : effOriginal;
        if (!src) {
            setMasterPeaks([]);
            setMasterDur(duration);
            return;
        }
        useAudioStore
            .getState()
            .loadAudioFile(src)
            .then((data) => {
            if (cancelled)
                return;
            if (data.peaks && data.peaks.length > 0)
                setMasterPeaks(data.peaks);
            if (data.durationMs > 0)
                setMasterDur(data.durationMs / 1000);
        })
            .catch(() => {
            if (!cancelled) {
                setMasterPeaks([]);
                setMasterDur(duration);
            }
        });
        return () => {
            cancelled = true;
        };
    }, [result, playSource, duration, effPreview, effOriginal]);
    // Draw the top master-track waveform (playable "音轨"): bars + playhead + hover cue.
    useEffect(() => {
        const canvas = masterCanvasRef.current;
        if (!canvas)
            return;
        const ctx = canvas.getContext("2d");
        if (!ctx)
            return;
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        const w = Math.floor(rect.width * dpr);
        const h = Math.floor(rect.height * dpr);
        if (canvas.width !== w || canvas.height !== h) {
            canvas.width = w;
            canvas.height = h;
        }
        if (w === 0 || h === 0)
            return;
        ctx.clearRect(0, 0, w, h);
        const midY = h / 2;
        const totalDur = masterDur || duration || 1;
        const playFrac = Math.min(1, Math.max(0, currentTime / totalDur));
        // Track backdrop.
        const bg = ctx.createLinearGradient(0, 0, 0, h);
        bg.addColorStop(0, "#161a22");
        bg.addColorStop(1, "#0e1116");
        ctx.fillStyle = bg;
        ctx.fillRect(0, 0, w, h);
        // Baseline.
        ctx.strokeStyle = "rgba(255,255,255,0.06)";
        ctx.lineWidth = 1 * dpr;
        ctx.beginPath();
        ctx.moveTo(0, midY);
        ctx.lineTo(w, midY);
        ctx.stroke();
        if (masterPeaks && masterPeaks.length > 0) {
            const numCols = Math.min(masterPeaks.length, Math.floor(w / (2 * dpr)));
            const colWidth = w / Math.max(1, numCols);
            const per = masterPeaks.length / numCols;
            const amp = h / 2 - 4 * dpr;
            for (let c = 0; c < numCols; c++) {
                const start = Math.floor(c * per);
                const end = Math.min(masterPeaks.length, Math.max(start + 1, Math.floor((c + 1) * per)));
                let p = 0;
                for (let i = start; i < end; i++) {
                    const v = Math.abs(masterPeaks[i] ?? 0);
                    if (v > p)
                        p = v;
                }
                const barX = c * colWidth;
                const isPast = c / numCols <= playFrac;
                ctx.fillStyle = isPast ? "#00e5ff" : "rgba(96,165,250,0.55)";
                const barH = Math.max(2 * dpr, p * amp);
                const barW = Math.max(1 * dpr, colWidth - 1 * dpr);
                ctx.fillRect(barX, midY - barH, barW, barH * 2);
            }
        }
        else {
            ctx.strokeStyle = "rgba(0,229,255,0.35)";
            ctx.setLineDash([4 * dpr, 4 * dpr]);
            ctx.beginPath();
            ctx.moveTo(0, midY);
            ctx.lineTo(w, midY);
            ctx.stroke();
            ctx.setLineDash([]);
        }
        // Playhead.
        if (totalDur > 0) {
            const px = playFrac * w;
            ctx.shadowBlur = 8;
            ctx.shadowColor = "#00e5ff";
            ctx.strokeStyle = "#00e5ff";
            ctx.lineWidth = 2 * dpr;
            ctx.beginPath();
            ctx.moveTo(px, 0);
            ctx.lineTo(px, h);
            ctx.stroke();
            ctx.shadowBlur = 0;
            ctx.fillStyle = "#00e5ff";
            ctx.beginPath();
            ctx.moveTo(px - 7 * dpr, 0);
            ctx.lineTo(px + 7 * dpr, 0);
            ctx.lineTo(px, 11 * dpr);
            ctx.closePath();
            ctx.fill();
        }
        // Hover cursor.
        if (masterHover) {
            const hx = masterHoverFrac * w;
            ctx.strokeStyle = "rgba(255,255,255,0.5)";
            ctx.lineWidth = 1 * dpr;
            ctx.setLineDash([3 * dpr, 3 * dpr]);
            ctx.beginPath();
            ctx.moveTo(hx, 0);
            ctx.lineTo(hx, h);
            ctx.stroke();
            ctx.setLineDash([]);
        }
    }, [masterPeaks, masterDur, duration, currentTime, masterHover, masterHoverFrac]);
    const seekMasterTrack = (frac) => {
        const totalDur = masterDur || duration || 1;
        handleSeek(Math.max(0, Math.min(1, frac)) * totalDur);
    };
    // ── Sample-synchronized playback (100% port of music-to-midi's SynchronizedPcmPlayer) ──
    // ONE audio graph mixes original + MIDI mix + per-instrument buses at a shared
    // clock, so nothing can drift. mix slider crossfades, stereo toggles A/B
    // (original left / MIDI right), per-stem M/S buttons rebuild the MIDI side from
    // the unmuted instrument buses.
    /** 0 = 原声, 100 = MIDI/分轨. Reference default: 75. */
    const [mixPercent, setMixPercent] = useState(75);
    /** A/B mode: original on LEFT, MIDI on RIGHT. */
    const [stereoMode, setStereoMode] = useState(false);
    const syncPlayerRef = useRef(null);
    // Player lifecycle: configure whenever the playable asset set changes.
    const playerAssetsKey = JSON.stringify([
        effPreview,
        effOriginal,
        stems.filter(s => s.audioPath).map(s => [s.name, s.audioPath]),
    ]);
    useEffect(() => {
        const player = new SynchronizedPlayer();
        syncPlayerRef.current = player;
        player.onEnded = () => {
            setIsPlaying(false);
            setCurrentTime(0);
            frozenPosRef.current = 0;
        };
        (async () => {
            const instrumentPaths = {};
            for (const s of stems) {
                if (s.audioPath)
                    instrumentPaths[s.name] = s.audioPath;
            }
            // §用户「干声转MIDI后试听要是对应的声音」：纯人声结果（所有音符轨都是干音轨）的
            // MIDI 侧不再用 GM 合成的 midiMix，而是干音 bus —— 试听始终是人声本身。
            const noteStems = stems.filter((s) => s.notes.length > 0);
            const pureVocal = !isSeparation && noteStems.length > 0 &&
                noteStems.every((s) => isVocalTrackName(s.name)) &&
                Object.keys(instrumentPaths).length > 0;
            await player.configure({
                originalPath: effOriginal || undefined,
                midiMixPath: pureVocal ? undefined : isSeparation ? undefined : effPreview || undefined,
                instrumentPaths,
            });
            // Restore the frozen transport position + rate across asset swaps
            // (reference keeps _frozen_position_frame / playback_rate on replace).
            player.setPlaybackRate(rateRef.current);
            if (frozenPosRef.current > 0)
                player.seek(frozenPosRef.current);
        })();
        return () => {
            player.dispose();
            if (syncPlayerRef.current === player)
                syncPlayerRef.current = null;
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [playerAssetsKey, isSeparation]);
    // Live playback rate: project BPM ÷ detected BPM (reference _apply_result_playback_rate).
    const playbackRate = detectedBpm && detectedBpm > 0 ? projBpm / detectedBpm : 1;
    useEffect(() => {
        rateRef.current = playbackRate;
        syncPlayerRef.current?.setPlaybackRate(playbackRate);
    }, [playbackRate, playerAssetsKey]);
    // Live mix state: crossfade slider, A/B toggle, per-stem mute.
    // SOLO SEMANTICS (reference _toggle_solo): soloing an instrument MUTES every
    // other one (the mute flags themselves carry it) — see toggleSolo below.
    useEffect(() => {
        const player = syncPlayerRef.current;
        if (!player)
            return;
        const muted = new Set(stems.filter(s => s.audioPath && s.muted).map(s => s.name));
        player.setMixState({
            mix: mixPercent / 100,
            stereo: stereoMode,
            muted,
        });
    }, [stems, mixPercent, stereoMode]);
    // Playhead clock: rAF-driven while playing (sample-accurate position).
    useEffect(() => {
        if (!isPlaying)
            return;
        let raf = 0;
        const tick = () => {
            const player = syncPlayerRef.current;
            if (player?.isPlaying) {
                const pos = player.position;
                setCurrentTime(pos);
                frozenPosRef.current = pos;
                raf = requestAnimationFrame(tick);
            }
        };
        raf = requestAnimationFrame(tick);
        return () => cancelAnimationFrame(raf);
    }, [isPlaying]);
    const togglePlay = () => {
        const player = syncPlayerRef.current;
        if (!player || !player.isConfigured) {
            if (preparingAudio) {
                showToast(t("amt.preparingPlayback") || "正在合成播放音频，请稍候...", "info");
            }
            else {
                showToast(t("amt.playbackUnavailable") || "播放音频不可用（未生成试听文件）", "error");
            }
            return;
        }
        if (isPlaying) {
            player.pause(); // freezes position internally (startOffset = current)
            frozenPosRef.current = player.position;
            setIsPlaying(false);
        }
        else {
            player.play(currentTime).then(() => {
                if (player.isPlaying)
                    setIsPlaying(true);
            });
        }
    };
    const stopPlay = () => {
        syncPlayerRef.current?.pause();
        syncPlayerRef.current?.seek(0);
        frozenPosRef.current = 0;
        setIsPlaying(false);
        setCurrentTime(0);
        if (viewportRef.current)
            viewportRef.current.scrollLeft = 0;
    };
    const handleSeek = (time) => {
        const player = syncPlayerRef.current;
        const target = Math.max(0, Math.min(duration || time, time));
        setCurrentTime(target);
        frozenPosRef.current = target;
        player?.seek(target);
    };
    // Handle Canvas Drawing — renders the FULL timeline (not only the visible viewport), so
    // scrolling right no longer shows a blank area. The backing store is scaled for HiDPI
    // while the CSS size stays at the logical `rollWidth`, keeping the scrollbar correct.
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || stems.length === 0)
            return;
        const ctx = canvas.getContext("2d");
        if (!ctx)
            return;
        const dpr = window.devicePixelRatio || 1;
        // Logical (CSS px) dimensions of the whole roll.
        const rollWidth = Math.max(1200, Math.ceil((duration || 1) * PIXELS_PER_SECOND));
        const rollHeight = 128 * ROW_HEIGHT;
        // Backing store = logical * dpr; CSS size = logical (drives the scroll width).
        const pxW = Math.ceil(rollWidth * dpr);
        const pxH = Math.ceil(rollHeight * dpr);
        if (canvas.width !== pxW || canvas.height !== pxH) {
            canvas.width = pxW;
            canvas.height = pxH;
        }
        canvas.style.width = `${rollWidth}px`;
        canvas.style.height = `${rollHeight}px`;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, rollWidth, rollHeight);
        // Draw Grid
        ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
        ctx.lineWidth = 1;
        // Horizontal lines (Semitones)
        for (let i = 0; i <= 128; i++) {
            const y = rollHeight - i * ROW_HEIGHT;
            const isC = i % 12 === 0;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(rollWidth, y);
            if (isC) {
                ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
            }
            else {
                ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
            }
            ctx.stroke();
        }
        // Vertical lines (Beats)
        const pixelsPerBeat = PIXELS_PER_SECOND * (60 / bpm);
        for (let x = 0; x <= rollWidth; x += pixelsPerBeat) {
            const isBar = Math.round(x / pixelsPerBeat) % 4 === 0;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, rollHeight);
            ctx.strokeStyle = isBar ? "rgba(255, 255, 255, 0.2)" : "rgba(255, 255, 255, 0.07)";
            ctx.lineWidth = isBar ? 1.5 : 1;
            ctx.stroke();
        }
        const hasSolo = stems.some(s => s.soloed);
        const activeName = activeInstr || stems[activeStemIdx]?.name || "";
        // Draw Notes (during a drag the draft overrides the dragged notes' geometry)
        const drawStems = stems;
        drawStems.forEach((stem, idx) => {
            if (stem.muted)
                return;
            if (hasSolo && !stem.soloed)
                return;
            ctx.fillStyle = stem.color;
            const isActive = !editMode && activeStemIdx === idx;
            const isEditInstr = editMode && stem.name === activeName;
            ctx.globalAlpha = isActive || isEditInstr ? 1.0 : editMode ? 0.45 : 0.25;
            const safePpq = ppq > 0 ? ppq : 480;
            const safeBpm = bpm > 0 ? bpm : 120;
            const pixelsPerTick = PIXELS_PER_SECOND / (safePpq * (safeBpm / 60));
            stem.notes.forEach(note => {
                const draftNote = dragRef.current?.draft?.get(note.id ?? "");
                const n = draftNote ?? note;
                const x = n.tick * pixelsPerTick;
                const w = Math.max(3, n.duration * pixelsPerTick - 1);
                const y = rollHeight - ((n.pitch + 1) * ROW_HEIGHT);
                const selected = editMode && selNotes.has(note.id ?? "");
                // Neon-glow effect for active stem notes
                if (isActive || isEditInstr || selected) {
                    ctx.shadowBlur = 8;
                    ctx.shadowColor = selected ? "#ffffff" : stem.color;
                }
                else {
                    ctx.shadowBlur = 0;
                }
                ctx.beginPath();
                ctx.roundRect(x, y + 1, w, ROW_HEIGHT - 2, 3);
                ctx.fill();
                ctx.shadowBlur = 0;
                // Inner highlight
                if (isActive || isEditInstr) {
                    ctx.fillStyle = "rgba(255, 255, 255, 0.3)";
                    ctx.fillRect(x, y + 1, w, 1);
                    ctx.fillStyle = stem.color;
                }
                // Selection outline (reference: selected notes get a bright frame)
                if (selected) {
                    ctx.strokeStyle = "#ffffff";
                    ctx.lineWidth = 1.5;
                    ctx.beginPath();
                    ctx.roundRect(x - 1, y, w + 2, ROW_HEIGHT, 3);
                    ctx.stroke();
                }
                // Repair-diff outline（§建议1）：一键修复动过的音符 → 琥珀色描边 + 柔光，
                // 与选中态（白色）区分开。高亮集合由 smartCleanup 写入、6 秒后自动清除。
                if (repairHighlight.has(note.id ?? "")) {
                    ctx.save();
                    ctx.strokeStyle = "#ffb300";
                    ctx.lineWidth = 2;
                    ctx.shadowBlur = 6;
                    ctx.shadowColor = "rgba(255, 179, 0, 0.9)";
                    ctx.beginPath();
                    ctx.roundRect(x - 1.5, y - 0.5, w + 3, ROW_HEIGHT + 1, 3);
                    ctx.stroke();
                    ctx.restore();
                }
                // §用户「MIDI音符都要有对应的歌词」：激活/编辑中的人声轨音符上直接画歌词（占位不画）。
                if (isActive || isEditInstr) {
                    const lyr = (note.lyric ?? "").trim();
                    if (lyr && !PLACEHOLDER_LYRICS.has(lyr) && ROW_HEIGHT >= 12) {
                        ctx.save();
                        ctx.beginPath();
                        ctx.rect(x, y, w, ROW_HEIGHT);
                        ctx.clip();
                        ctx.fillStyle = "rgba(255, 255, 255, 0.92)";
                        ctx.font = `${Math.max(8, Math.min(11, ROW_HEIGHT - 5))}px sans-serif`;
                        ctx.textBaseline = "middle";
                        ctx.fillText(lyr, x + 3, y + ROW_HEIGHT / 2);
                        ctx.restore();
                    }
                }
            });
        });
        ctx.globalAlpha = 1.0;
        // Marquee selection rectangle
        if (marquee) {
            const x = Math.min(marquee.x0, marquee.x1);
            const y = Math.min(marquee.y0, marquee.y1);
            const w = Math.abs(marquee.x1 - marquee.x0);
            const h = Math.abs(marquee.y1 - marquee.y0);
            ctx.fillStyle = "rgba(0, 229, 255, 0.08)";
            ctx.fillRect(x, y, w, h);
            ctx.strokeStyle = "rgba(0, 229, 255, 0.7)";
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 3]);
            ctx.strokeRect(x, y, w, h);
            ctx.setLineDash([]);
        }
        // Draw playhead
        const playheadX = currentTime * PIXELS_PER_SECOND;
        // Playhead glow
        const gradient = ctx.createLinearGradient(playheadX - 20, 0, playheadX + 20, 0);
        gradient.addColorStop(0, "rgba(0, 229, 255, 0)");
        gradient.addColorStop(0.5, "rgba(0, 229, 255, 0.1)");
        gradient.addColorStop(1, "rgba(0, 229, 255, 0)");
        ctx.fillStyle = gradient;
        ctx.fillRect(playheadX - 20, 0, 40, rollHeight);
        ctx.strokeStyle = "#00e5ff";
        ctx.lineWidth = 2;
        ctx.shadowBlur = 10;
        ctx.shadowColor = "#00e5ff";
        ctx.beginPath();
        ctx.moveTo(playheadX, 0);
        ctx.lineTo(playheadX, rollHeight);
        ctx.stroke();
        ctx.shadowBlur = 0;
        // Playhead handle
        ctx.fillStyle = "#00e5ff";
        ctx.beginPath();
        ctx.moveTo(playheadX - 8, 0);
        ctx.lineTo(playheadX + 8, 0);
        ctx.lineTo(playheadX, 12);
        ctx.fill();
    }, [stems, activeStemIdx, activeInstr, editMode, selNotes, marquee, dragRev, currentTime, duration, bpm, ppq, ROW_HEIGHT, PIXELS_PER_SECOND, repairHighlight]);
    // Sync scroll with playhead (reference: follow_checkbox, default ON)
    useEffect(() => {
        if (isPlaying && followMode && viewportRef.current) {
            const x = currentTime * PIXELS_PER_SECOND;
            const vw = viewportRef.current.clientWidth - PIANO_WIDTH;
            const scrollLeft = viewportRef.current.scrollLeft;
            if (x > scrollLeft + vw * 0.7) {
                viewportRef.current.scrollLeft = x - vw * 0.3;
            }
        }
    }, [currentTime, isPlaying, followMode, PIXELS_PER_SECOND]);
    // ═══════════════════════════════════════════════════════════════════════════
    // MIDI EDITOR — 100% port of the reference project's editor_panel:
    //   select (click / marquee / Ctrl+Shift multi), drag move & edge-resize with
    //   grid snap (Alt = no snap), double-click add, right-click delete, keyboard
    //   transforms, cut/copy/paste/duplicate, quantize (scope + grid), undo/redo/
    //   reset, per-note velocity — and every commit re-renders the FluidSynth
    //   audition audio so the edited MIDI is always audible.
    // ═══════════════════════════════════════════════════════════════════════════
    /** Quantize grid in 480-ppq ticks (1/4 → 480, 1/32 → 60 …). Uses the DETECTED
     *  bpm space — the roll's x-axis is source time (reference _editor_grid_seconds). */
    const gridTicks = useMemo(() => {
        const safePpq = ppq > 0 ? ppq : 480;
        return Math.max(1, Math.round((safePpq * 4) / quantGrid));
    }, [ppq, quantGrid]);
    const durationTicks = useMemo(() => {
        const safePpq = ppq > 0 ? ppq : 480;
        const safeBpm = bpm > 0 ? bpm : 120;
        return Math.max(1, Math.round((duration || 1) * safePpq * (safeBpm / 60)));
    }, [duration, ppq, bpm]);
    const pixelsPerTick = useMemo(() => {
        const safePpq = ppq > 0 ? ppq : 480;
        const safeBpm = bpm > 0 ? bpm : 120;
        return PIXELS_PER_SECOND / (safePpq * (safeBpm / 60));
    }, [ppq, bpm, PIXELS_PER_SECOND]);
    const idSeqRef = useRef(1);
    const newNoteId = () => `n${idSeqRef.current++}`;
    const cloneStems = (src) => src.map((s) => ({ ...s, notes: s.notes.map((n) => ({ ...n })) }));
    const findNoteAt = (x, y) => {
        const rollHeight = 128 * ROW_HEIGHT;
        const hasSolo = stems.some((s) => s.soloed);
        for (let si = stems.length - 1; si >= 0; si--) {
            const stem = stems[si];
            if (!stem)
                continue;
            if (stem.muted)
                continue;
            if (hasSolo && !stem.soloed)
                continue;
            for (let ni = stem.notes.length - 1; ni >= 0; ni--) {
                const n = stem.notes[ni];
                if (!n)
                    continue;
                const nx = n.tick * pixelsPerTick;
                const nw = Math.max(3, n.duration * pixelsPerTick - 1);
                const ny = rollHeight - ((n.pitch + 1) * ROW_HEIGHT);
                if (x >= nx && x <= nx + nw && y >= ny && y <= ny + ROW_HEIGHT) {
                    return { si, ni, note: n, nx, ny, nw };
                }
            }
        }
        return null;
    };
    const pitchForY = (y) => {
        const rollHeight = 128 * ROW_HEIGHT;
        return Math.max(0, Math.min(127, Math.floor((rollHeight - y) / ROW_HEIGHT) - 1));
    };
    const tickForX = (x) => Math.max(0, x / pixelsPerTick);
    const snapTick = (tick) => Math.max(0, Math.round(tick / gridTicks) * gridTicks);
    /** Selection as {si, ni} pairs resolved against the CURRENT stems. */
    const selectionEntries = useMemo(() => {
        const out = [];
        stems.forEach((s, si) => s.notes.forEach((n, ni) => {
            if (n.id && selNotes.has(n.id))
                out.push({ si, ni, note: n });
        }));
        return out;
    }, [stems, selNotes]);
    // ── Edit commit pipeline (reference _record_editor_commit + _queue_editor_audio_render) ──
    const resynthTimerRef = useRef(null);
    const renderGenRef = useRef(0);
    const scheduleResynthesis = (next) => {
        if (isSeparation)
            return;
        const srcAudio = result.sourceAudioPath || result.originalAudio;
        if (!srcAudio || !result.outputDir)
            return; // no re-render source — edits stay visual-only
        if (resynthTimerRef.current)
            window.clearTimeout(resynthTimerRef.current);
        resynthTimerRef.current = window.setTimeout(async () => {
            const gen = ++renderGenRef.current;
            setEditRendering(true);
            try {
                const outDir = `${result.outputDir.replace(/[\/\\]+$/, "")}/edit_render`;
                const midiPath = `${outDir}/edited_${Date.now()}.mid`;
                await invoke("amt_write_edited_midi", {
                    outPath: midiPath,
                    // Render at the DETECTED tempo: the roll timeline is source-time and the
                    // transport's playbackRate handles speed — audio and grid stay aligned.
                    bpm: detectedBpm ?? bpm,
                    tracks: next.map((s) => ({
                        name: s.name,
                        program: s.program ?? null,
                        channel: s.channel ?? null,
                        notes: s.notes.map((n) => ({
                            tick: Math.round(n.tick),
                            duration: Math.round(n.duration),
                            pitch: n.pitch,
                            velocity: n.velocity ?? 100,
                            lyric: n.lyric ?? undefined,
                        })),
                    })),
                });
                const res = await invoke("amt_prepare_playback", {
                    midiPath,
                    audioPath: srcAudio,
                    outputDir: outDir,
                });
                if (gen !== renderGenRef.current)
                    return; // superseded by a newer edit
                setPrepAudio({
                    preview: res?.transcription_wav || "",
                    original: res?.original_wav || "",
                    wavs: (res?.instrument_wavs ?? {}),
                });
            }
            catch (e) {
                showToast(t("amt.editorAudioFailed", { error: String(e) }) || `试听音频更新失败：${e}`, "error");
            }
            finally {
                if (gen === renderGenRef.current)
                    setEditRendering(false);
            }
        }, 400);
    };
    const applyEdit = (next, record = true) => {
        if (record) {
            // Reference pauses playback on every committed edit (_record_editor_commit).
            const player = syncPlayerRef.current;
            if (player?.isPlaying) {
                player.pause();
                frozenPosRef.current = player.position;
                setIsPlaying(false);
            }
            undoRef.current.push(cloneStems(stems));
            if (undoRef.current.length > 100)
                undoRef.current.shift();
            redoRef.current = [];
        }
        setStems(next);
        setEditDirty(true);
        scheduleResynthesis(next);
    };
    const undoEdit = () => {
        if (undoRef.current.length === 0)
            return;
        const prev = undoRef.current.pop();
        redoRef.current.push(cloneStems(stems));
        setStems(prev);
        setSelNotes(new Set());
        scheduleResynthesis(prev);
    };
    const redoEdit = () => {
        if (redoRef.current.length === 0)
            return;
        const nxt = redoRef.current.pop();
        undoRef.current.push(cloneStems(stems));
        setStems(nxt);
        setSelNotes(new Set());
        scheduleResynthesis(nxt);
    };
    const resetEdits = () => {
        if (!originalStemsRef.current)
            return;
        applyEdit(cloneStems(originalStemsRef.current));
        setSelNotes(new Set());
        setEditDirty(false);
    };
    // ── SoundFont actions (P0-C): switch re-renders the CURRENT score with the new source ──
    // §user「切换音源时 2 秒预览试听」：切换成功后立刻用新音源渲染开头 ~2 秒
    // 的音符并播放（preview_notes_render → 共享 AudioContext），不用整曲验证。
    const sfPreviewSrcRef = useRef(null);
    const stopSfPreview = React.useCallback(() => {
        const s = sfPreviewSrcRef.current;
        sfPreviewSrcRef.current = null;
        if (s) {
            try {
                s.stop();
            }
            catch { /* already ended */ }
        }
    }, []);
    const previewSoundfontSnippet = React.useCallback(async () => {
        try {
            const stem = stems.find((s) => s.notes.length > 0);
            if (!stem)
                return;
            const safePpq = ppq > 0 ? ppq : 480;
            const safeBpm = bpm > 0 ? bpm : 120;
            // 前 2 秒内的音符；前奏稀疏时至少取前 4 个，上限 32 个防极端密集。
            const twoSecTicks = 2 * safePpq * (safeBpm / 60);
            let picked = stem.notes.filter((n) => n.tick < twoSecTicks);
            if (picked.length === 0)
                picked = stem.notes.slice(0, 4);
            const notes = picked.slice(0, 32).map((n) => ({
                tick: Math.round(n.tick),
                duration: Math.round(Math.max(n.duration, 60)),
                pitch: n.pitch,
                velocity: n.velocity ?? 100,
            }));
            if (notes.length === 0)
                return;
            const wavPath = await invoke("preview_notes_render", {
                bpm: safeBpm,
                notes,
                program: stem.program ?? null,
                channel: stem.channel ?? null,
            });
            const ctx = playback.getPreviewContext();
            const buf = await playback.loadAudioBuffer(wavPath);
            stopSfPreview();
            const src = ctx.createBufferSource();
            src.buffer = buf;
            src.connect(ctx.destination);
            src.onended = () => { if (sfPreviewSrcRef.current === src)
                sfPreviewSrcRef.current = null; };
            sfPreviewSrcRef.current = src;
            src.start(0, 0, 2); // 最多播 2 秒
        }
        catch { /* 预览失败不影响切换：整曲重渲染仍会给出结果 */ }
    }, [stems, ppq, bpm, stopSfPreview]);
    const handleSwitchSoundfont = async (filename, name) => {
        setSfOpen(false);
        try {
            await invoke("amt_set_active_soundfont", { filename });
            refreshSoundfonts();
            invoke("amt_export_tools_status").then(setExportTools).catch(() => { });
            showToast(t("amt.sfSwitched", { name }) || `已切换音源：${name}`, "success");
            // Re-render the preview with the new soundfont (same pipeline as an edit).
            if (!isSeparation && stems.length > 0)
                scheduleResynthesis(stems);
            // 2 秒预览试听（静默失败—— FluidSynth/音源缺失时整曲重渲染会报具体错误）。
            void previewSoundfontSnippet();
        }
        catch (e) {
            showToast(mapExportError(String(e)), "error");
        }
    };
    const handleImportSoundfont = async () => {
        try {
            const { open } = await import("@tauri-apps/plugin-dialog");
            const src = await open({
                multiple: false,
                filters: [{ name: "SoundFont", extensions: ["sf2", "sf3"] }],
                title: t("amt.sfImportTitle") || "选择 SoundFont 文件（SF2 / SF3）",
            });
            if (!src || Array.isArray(src))
                return;
            const info = await invoke("amt_import_soundfont", { sourcePath: src });
            refreshSoundfonts();
            showToast(t("amt.sfImported", { name: info?.name ?? "" }) || "音源导入成功", "success");
        }
        catch (e) {
            showToast(mapExportError(String(e)), "error");
        }
    };
    const handleDeleteSoundfont = async (filename, name) => {
        try {
            await invoke("amt_delete_soundfont", { filename });
            refreshSoundfonts();
            invoke("amt_export_tools_status").then(setExportTools).catch(() => { });
            showToast(t("amt.sfDeleted", { name }) || `已删除音源：${name}`, "success");
            // The deleted source may have been the active one — re-render with the fallback.
            if (!isSeparation && stems.length > 0)
                scheduleResynthesis(stems);
        }
        catch (e) {
            showToast(mapExportError(String(e)), "error");
        }
    };
    const selectAllNotes = () => {
        if (!editMode)
            return;
        setSelNotes(new Set(stems.flatMap((s) => s.notes.map((n) => n.id ?? ""))));
    };
    const deleteSelection = () => {
        if (!editMode || selNotes.size === 0)
            return;
        const next = cloneStems(stems).map((s) => ({
            ...s,
            notes: s.notes.filter((n) => !n.id || !selNotes.has(n.id)),
        }));
        applyEdit(next);
        setSelNotes(new Set());
    };
    const addNoteAt = (tick, pitch) => {
        if (!editMode || !activeInstr)
            return;
        const si = stems.findIndex((s) => s.name === activeInstr);
        if (si < 0)
            return;
        // Reference template: 0.5 s duration, pitch clamped 21..108, velocity from the spin.
        const halfSecondTicks = Math.max(1, Math.round(0.5 * (ppq || 480) * (bpm / 60)));
        const clampedPitch = Math.max(21, Math.min(108, Math.round(pitch)));
        const start = Math.max(0, Math.min(Math.round(tick), durationTicks - 1));
        const dur = Math.max(1, Math.min(halfSecondTicks, durationTicks - start));
        const note = {
            id: newNoteId(),
            tick: start,
            duration: dur,
            pitch: clampedPitch,
            velocity: velSpin,
        };
        const next = cloneStems(stems);
        next[si]?.notes.push(note);
        applyEdit(next);
        setSelNotes(new Set([note.id]));
    };
    /** Reference _add_editor_note_at_playhead: new note at the playhead, pitch 60 (C4). */
    const addNoteAtPlayhead = () => {
        if (!editMode)
            return;
        const playheadSec = currentTime / (rateRef.current || 1);
        addNoteAt(playheadSec * (ppq || 480) * (bpm / 60), 60);
    };
    const copySelection = () => {
        if (!editMode || selectionEntries.length === 0)
            return;
        clipboardRef.current = selectionEntries.map(({ note }) => ({ ...note }));
    };
    const cutSelection = () => {
        if (!editMode || selectionEntries.length === 0)
            return;
        copySelection();
        deleteSelection();
    };
    const pasteClipboard = () => {
        if (!editMode || clipboardRef.current.length === 0)
            return;
        const clip = clipboardRef.current;
        const sourceStart = Math.min(...clip.map((n) => n.tick));
        const sourceEnd = Math.max(...clip.map((n) => n.tick + n.duration));
        const span = sourceEnd - sourceStart;
        // Paste at the PLAYHEAD (reference _paste_editor_notes).
        const playheadTick = Math.round(currentTime * (ppq || 480) * (bpm / 60));
        const targetStart = Math.max(0, Math.min(playheadTick, durationTicks - span));
        const offset = targetStart - sourceStart;
        const byInstr = new Map();
        for (const n of clip) {
            const instr = stems.find((s) => s.notes.some((x) => x.id === n.id))?.name ?? activeInstr;
            const copy = {
                ...n,
                id: newNoteId(),
                tick: Math.max(0, n.tick + offset),
            };
            const list = byInstr.get(instr) ?? [];
            list.push(copy);
            byInstr.set(instr, list);
        }
        const next = cloneStems(stems);
        const newIds = [];
        for (const [instr, list] of byInstr) {
            const si = next.findIndex((s) => s.name === instr);
            if (si >= 0 && next[si]) {
                next[si].notes.push(...list);
                newIds.push(...list.map((n) => n.id));
            }
        }
        applyEdit(next);
        setSelNotes(new Set(newIds));
    };
    const duplicateSelection = () => {
        if (!editMode || selectionEntries.length === 0)
            return;
        const selStart = Math.min(...selectionEntries.map(({ note }) => note.tick));
        const selEnd = Math.max(...selectionEntries.map(({ note }) => note.tick + note.duration));
        const offset = Math.max(gridTicks, selEnd - selStart);
        const clamped = Math.min(offset, durationTicks - selEnd);
        if (clamped <= 0)
            return;
        const next = cloneStems(stems);
        const newIds = [];
        for (const { si, note } of selectionEntries) {
            const copy = { ...note, id: newNoteId(), tick: note.tick + clamped };
            next[si]?.notes.push(copy);
            newIds.push(copy.id);
        }
        applyEdit(next);
        setSelNotes(new Set(newIds));
    };
    const quantizeSelection = () => {
        if (!editMode)
            return;
        const all = quantScope === "all_tracks";
        const targets = all
            ? stems.flatMap((s) => s.notes)
            : selectionEntries.map(({ note }) => note);
        if (targets.length === 0)
            return;
        const grid = gridTicks;
        const next = cloneStems(stems);
        let changed = false;
        next.forEach((s) => s.notes.forEach((n) => {
            if (!all && (!n.id || !selNotes.has(n.id)))
                return;
            const qDur = Math.min(durationTicks, Math.max(grid, Math.round(n.duration / grid) * grid));
            const start = Math.max(0, Math.min(durationTicks - qDur, Math.round(n.tick / grid) * grid));
            if (start !== n.tick || qDur !== n.duration) {
                n.tick = start;
                n.duration = qDur;
                changed = true;
            }
        }));
        if (changed)
            applyEdit(next);
    };
    /** One-click fully-automatic repair (smart cleanup v2) — shared core lives in
     *  src/lib/midiCleanup.ts; this wrapper clones, runs it, and reports the stats
     *  as ONE undo step so 撤销 reverts everything. See the module doc for the
     *  7-pass strategy (ghost deletion + missed-bar fill). */
    const smartCleanup = () => {
        if (!editMode)
            return;
        const next = cloneStems(stems);
        const stats = smartCleanStems(next, { bpm, ppq, durationTicks });
        if (cleanChangedTotal(stats) === 0) {
            showToast(t("amt.editorCleanNoChange") || "已经非常干净，没有需要修复的音符", "success");
            return;
        }
        applyEdit(next);
        // 差异高亮（§建议1）：被修过且留存的音符 6 秒内描琥珀边，看清"AI 修了哪里"。
        if (stats.touched.length > 0) {
            setRepairHighlight(new Set(stats.touched));
            if (repairHighlightTimer.current)
                window.clearTimeout(repairHighlightTimer.current);
            repairHighlightTimer.current = window.setTimeout(() => setRepairHighlight(new Set()), 6000);
        }
        const removedExtra = stats.removedShort + stats.removedRange + stats.removedGhost + stats.removedKey;
        showToast(t("amt.editorCleanReport", {
            removed: removedExtra,
            merged: stats.mergedOverlap,
            velocity: stats.fixedVelocity,
            filled: stats.filledNotes,
            trimmed: stats.trimmedHang,
            snapped: stats.snapped,
        }) || "一键修复完成", "success");
    };
    /** Keyboard/drag transforms — mirrors _transform_selected_editor_notes clamping. */
    const transformSelection = (command, amount) => {
        if (!editMode || selectionEntries.length === 0)
            return;
        const next = cloneStems(stems);
        const entries = selectionEntries
            .map(({ si, ni }) => ({ si, ni, note: next[si]?.notes[ni] }))
            .filter((e) => e.note != null);
        if (command === "move_time") {
            let delta = gridTicks * amount;
            delta = Math.max(-Math.min(...entries.map((e) => e.note.tick)), Math.min(durationTicks - Math.max(...entries.map((e) => e.note.tick + e.note.duration)), delta));
            for (const e of entries)
                e.note.tick += delta;
        }
        else if (command === "resize_time") {
            let delta = gridTicks * amount;
            delta = Math.max(-(Math.min(...entries.map((e) => e.note.duration)) - 1), Math.min(durationTicks - Math.max(...entries.map((e) => e.note.tick + e.note.duration)), delta));
            for (const e of entries)
                e.note.duration = Math.max(1, e.note.duration + delta);
        }
        else if (command === "pitch") {
            const delta = Math.max(21 - Math.min(...entries.map((e) => e.note.pitch)), Math.min(108 - Math.max(...entries.map((e) => e.note.pitch)), amount));
            for (const e of entries)
                e.note.pitch += delta;
        }
        else if (command === "velocity") {
            const delta = Math.max(1 - Math.min(...entries.map((e) => e.note.velocity ?? 100)), Math.min(127 - Math.max(...entries.map((e) => e.note.velocity ?? 100)), amount));
            for (const e of entries)
                e.note.velocity = Math.max(1, Math.min(127, (e.note.velocity ?? 100) + delta));
        }
        applyEdit(next);
    };
    /** Velocity spin: sets the ABSOLUTE velocity on the selection (reference behavior). */
    const onVelocitySpin = (v) => {
        setVelSpin(v);
        if (!editMode || selectionEntries.length === 0)
            return;
        const next = cloneStems(stems);
        for (const { si, ni } of selectionEntries) {
            const target = next[si]?.notes[ni];
            if (target)
                target.velocity = v;
        }
        applyEdit(next);
    };
    // ── Roll mouse interactions (edit mode) ──
    const rollMouseDown = (e) => {
        if (!editMode || e.button !== 0)
            return;
        const canvas = canvasRef.current;
        if (!canvas)
            return;
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const hit = findNoteAt(x, y);
        if (hit) {
            const id = hit.note.id ?? "";
            let nextSel;
            if (e.shiftKey || e.ctrlKey || e.metaKey) {
                nextSel = new Set(selNotes);
                if (nextSel.has(id))
                    nextSel.delete(id);
                else
                    nextSel.add(id);
            }
            else if (selNotes.has(id)) {
                nextSel = selNotes;
            }
            else {
                nextSel = new Set([id]);
            }
            setSelNotes(nextSel);
            // Single selection drives the instrument combo + velocity spin
            // (reference _on_roll_selection_changed).
            if (nextSel.size === 1) {
                const hitStem = stems[hit.si];
                if (hitStem)
                    setActiveInstr(hitStem.name);
                setVelSpin(hit.note.velocity ?? 100);
            }
            const nearRight = x > hit.nx + hit.nw - Math.max(6, hit.nw * 0.25);
            const orig = new Map();
            const entries = nextSel.size > 0
                ? stems.flatMap((s) => s.notes).filter((n) => n.id && nextSel.has(n.id))
                : [hit.note];
            for (const n of entries) {
                if (n.id)
                    orig.set(n.id, { tick: n.tick, duration: n.duration, pitch: n.pitch });
            }
            dragRef.current = {
                mode: nearRight ? "resize" : "move",
                snap: !e.altKey,
                startX: x,
                startY: y,
                orig,
                draft: new Map(),
            };
        }
        else {
            const additive = e.shiftKey || e.ctrlKey || e.metaKey;
            setMarquee({ x0: x, y0: y, x1: x, y1: y, additive });
            if (!additive)
                setSelNotes(new Set());
        }
    };
    const rollMouseMove = (e) => {
        const drag = dragRef.current;
        const canvas = canvasRef.current;
        if (!canvas)
            return;
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        if (drag) {
            let dTick = tickForX(x) - tickForX(drag.startX);
            if (drag.snap)
                dTick = Math.round(dTick / gridTicks) * gridTicks;
            const dPitch = pitchForY(y) - pitchForY(drag.startY);
            const vals = [...drag.orig.values()];
            const draft = new Map();
            if (drag.mode === "move") {
                let delta = Math.max(-Math.min(...vals.map((v) => v.tick)), Math.min(durationTicks - Math.max(...vals.map((v) => v.tick + v.duration)), dTick));
                if (!Number.isFinite(delta))
                    delta = dTick;
                for (const [id, v] of drag.orig) {
                    draft.set(id, {
                        tick: Math.max(0, v.tick + delta),
                        duration: v.duration,
                        pitch: Math.max(21, Math.min(108, v.pitch + dPitch)),
                    });
                }
            }
            else {
                let delta = Math.max(-(Math.min(...vals.map((v) => v.duration)) - 1), dTick);
                const room = durationTicks - Math.max(...vals.map((v) => v.tick + v.duration));
                if (Number.isFinite(room))
                    delta = Math.min(room, delta);
                for (const [id, v] of drag.orig) {
                    draft.set(id, {
                        tick: v.tick,
                        duration: Math.max(1, v.duration + delta),
                        pitch: v.pitch,
                    });
                }
            }
            drag.draft = draft;
            setDragRev((r) => r + 1);
        }
        else if (marquee) {
            setMarquee((m) => (m ? { ...m, x1: x, y1: y } : m));
        }
    };
    const rollMouseUp = () => {
        const drag = dragRef.current;
        if (drag) {
            dragRef.current = null;
            let changed = false;
            for (const [id, d] of drag.draft) {
                const o = drag.orig.get(id);
                if (o && (o.tick !== d.tick || o.duration !== d.duration || o.pitch !== d.pitch)) {
                    changed = true;
                    break;
                }
            }
            if (changed) {
                const next = cloneStems(stems);
                next.forEach((s) => s.notes.forEach((n) => {
                    const d = drag.draft.get(n.id ?? "");
                    if (d) {
                        n.tick = d.tick;
                        n.duration = d.duration;
                        n.pitch = d.pitch;
                    }
                }));
                applyEdit(next);
            }
            else {
                setDragRev((r) => r + 1);
            }
            return;
        }
        if (marquee) {
            const x0 = Math.min(marquee.x0, marquee.x1);
            const x1 = Math.max(marquee.x0, marquee.x1);
            const y0 = Math.min(marquee.y0, marquee.y1);
            const y1 = Math.max(marquee.y0, marquee.y1);
            const rollHeight = 128 * ROW_HEIGHT;
            const hits = new Set();
            stems.forEach((s) => s.notes.forEach((n) => {
                const nx = n.tick * pixelsPerTick;
                const nw = Math.max(3, n.duration * pixelsPerTick - 1);
                const ny = rollHeight - ((n.pitch + 1) * ROW_HEIGHT);
                if (nx <= x1 && nx + nw >= x0 && ny <= y1 && ny + ROW_HEIGHT >= y0 && n.id) {
                    hits.add(n.id);
                }
            }));
            setSelNotes((prev) => (marquee.additive ? new Set([...prev, ...hits]) : hits));
            setMarquee(null);
        }
    };
    const rollDoubleClick = (e) => {
        const canvas = canvasRef.current;
        if (!canvas)
            return;
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        // §用户「点音符改词」：双击人声轨的已有音符 → 浮层改歌词（查看/编辑模式均可）；
        // 其余情况保持原行为（编辑模式双击空白 = 添加音符）。
        const hit = findNoteAt(x, y);
        const hitStem = hit ? stems[hit.si] : undefined;
        if (hit && hit.note.id !== undefined && hitStem && isVocalTrackName(hitStem.name)) {
            setActiveInstr(hitStem.name);
            setSelNotes(new Set([hit.note.id]));
            setLyricEdit({ stemIdx: hit.si, noteId: hit.note.id, x, y, value: hit.note.lyric ?? "" });
            return;
        }
        if (!editMode)
            return;
        addNoteAt(snapTick(tickForX(x)), pitchForY(y));
    };
    /** §用户：歌词浮层提交 —— 写回该音符（进撤销栈），只动歌词不动几何。 */
    const commitRollLyric = (value) => {
        const le = lyricEditRef.current;
        setLyricEdit(null);
        if (!le)
            return;
        const v = value.trim();
        if (!v || v === (le.value ?? "").trim())
            return;
        const next = cloneStems(stems);
        const note = next[le.stemIdx]?.notes.find((n) => n.id === le.noteId);
        if (!note)
            return;
        note.lyric = v;
        applyEdit(next);
    };
    /** 右键菜单统一：右键音符/空白不再直接删除，而是就近弹出常用操作菜单（剪切/复制/
     *  粘贴/向右复制/量化/一键修复/删除）。点在音符上会先选中它。 */
    const rollContextMenu = (e) => {
        if (!editMode)
            return;
        e.preventDefault();
        const canvas = canvasRef.current;
        if (!canvas)
            return;
        const rect = canvas.getBoundingClientRect();
        const hit = findNoteAt(e.clientX - rect.left, e.clientY - rect.top);
        if (hit && hit.note.id) {
            // 右键一个未选中的音符 = 单选它；右键已选中的音符 = 保留整个选区。
            if (!selNotes.has(hit.note.id))
                setSelNotes(new Set([hit.note.id]));
            const hitStem = stems[hit.si];
            if (hitStem)
                setActiveInstr(hitStem.name);
        }
        setRollMenu({ x: e.clientX, y: e.clientY });
    };
    /** 右键菜单条目：与工具栏/快捷键共用同一批操作函数，保证行为一致。 */
    const rollMenuItems = () => {
        const hasSel = selectionEntries.length > 0;
        const hasClip = clipboardRef.current.length > 0;
        return [
            { label: t("amt.editorCut") || "剪切", icon: "✂", shortcut: "Ctrl+X", disabled: !hasSel, onClick: cutSelection },
            { label: t("amt.editorCopy") || "复制", icon: "⧉", shortcut: "Ctrl+C", disabled: !hasSel, onClick: copySelection },
            { label: t("amt.editorPaste") || "粘贴", icon: "📋", shortcut: "Ctrl+V", disabled: !hasClip, onClick: pasteClipboard },
            { label: t("amt.editorDuplicate") || "向右复制", icon: "⇥", shortcut: "Ctrl+D", disabled: !hasSel, onClick: duplicateSelection },
            { label: t("amt.editorQuantize") || "量化（对齐拍子）", icon: "⌗", shortcut: "Alt+Q", onClick: quantizeSelection },
            { label: t("amt.editorClean") || "一键修复", icon: "🪄", onClick: smartCleanup, title: t("amt.editorCleanTooltip") },
            { label: t("amt.editorUndo") || "撤销", icon: "↶", shortcut: "Ctrl+Z", disabled: undoRef.current.length === 0, onClick: undoEdit },
            { label: t("amt.editorRedo") || "重做", icon: "↷", shortcut: "Ctrl+Y", disabled: redoRef.current.length === 0, onClick: redoEdit },
            { label: t("amt.editorDelete") || "删除", icon: "🗑", shortcut: "Del", danger: true, disabled: !hasSel, onClick: deleteSelection },
        ];
    };
    /** Roll keyboard map — 1:1 with the reference _PianoRollCanvas.keyPressEvent. */
    const rollKeyDown = (e) => {
        // 「?」切换 / Esc 关闭快捷键速查卡（优先于其他手势）。
        if (e.key === "?" || (e.shiftKey && e.key === "/")) {
            e.preventDefault();
            setRollHelpOpen((o) => !o);
            return;
        }
        if (e.key === "Escape" && rollHelpOpen) {
            e.preventDefault();
            setRollHelpOpen(false);
            return;
        }
        if (!editMode)
            return;
        const mod = e.ctrlKey || e.metaKey;
        const key = e.key.toLowerCase();
        const handle = (fn) => {
            e.preventDefault();
            fn();
        };
        if (mod && !e.shiftKey && key === "z")
            return handle(undoEdit);
        if ((mod && key === "y") || (mod && e.shiftKey && key === "z"))
            return handle(redoEdit);
        if (mod && key === "a")
            return handle(selectAllNotes);
        if (mod && key === "x")
            return handle(cutSelection);
        if (mod && key === "c")
            return handle(copySelection);
        if (mod && key === "v")
            return handle(pasteClipboard);
        if (mod && (key === "d" || key === "b"))
            return handle(duplicateSelection);
        if ((e.altKey && key === "q") || (mod && key === "u"))
            return handle(quantizeSelection);
        if (key === "delete" || key === "backspace")
            return handle(deleteSelection);
        if (key === "escape")
            return handle(() => setSelNotes(new Set()));
        if (key === "arrowleft" || key === "arrowright") {
            const dir = key === "arrowleft" ? -1 : 1;
            return handle(() => transformSelection(e.shiftKey ? "resize_time" : "move_time", dir));
        }
        if (key === "arrowup" || key === "arrowdown") {
            const dir = key === "arrowup" ? 1 : -1;
            if (mod)
                return handle(() => transformSelection("velocity", dir));
            return handle(() => transformSelection("pitch", dir * (e.shiftKey ? 12 : 1)));
        }
    };
    // ── Tempo controls (reference bpm_spin / speed_spin linkage) ──
    const clampRate = (r) => Math.min(20, Math.max(0.05, r));
    const onProjBpmChange = (v) => {
        const clamped = Math.min(400, Math.max(4, v));
        setProjBpm(clamped);
        if (detectedBpm && detectedBpm > 0)
            setSpeed(clampRate(clamped / detectedBpm));
    };
    const onSpeedChange = (v) => {
        if (!detectedBpm || detectedBpm <= 0)
            return;
        const target = Math.min(400, Math.max(4, detectedBpm * Math.min(20, Math.max(0.05, v))));
        setProjBpm(target);
        setSpeed(clampRate(target / detectedBpm));
    };
    // ── Solo/mute: EXACT reference semantics ──
    // _toggle_mute: any mute click CLEARS solo. _toggle_solo: solo(i) mutes every
    // other detected instrument; un-solo clears all mutes.
    const toggleStemMute = (idx) => {
        const next = stems.map((s) => ({ ...s, soloed: false, muted: s.muted }));
        const target = next[idx];
        if (!target || !stems[idx])
            return;
        target.muted = !stems[idx].muted;
        setStems(next);
    };
    const toggleStemSolo = (idx) => {
        if (stems[idx]?.soloed) {
            setStems(stems.map((s) => ({ ...s, soloed: false, muted: false })));
        }
        else {
            setStems(stems.map((s, i) => ({ ...s, soloed: i === idx, muted: i !== idx })));
        }
    };
    // "转写另一首" — close the workbench and reopen the conversion dialog.
    const handleAnother = () => {
        const store = useAppStore.getState();
        onClose();
        store.openAmtConversion(result.trackId);
    };
    const handleImportAll = async () => {
        try {
            // Coalesce into one undo step
            useHistoryStore.getState().beginTransaction();
            for (const stem of stems) {
                // We already have the notes in the stem object, no need to re-invoke import_score_file
                // unless we want to be absolutely sure about the structure.
                // But stem.notes is already populated during loadData.
                const newTrackId = crypto.randomUUID();
                const segmentId = crypto.randomUUID();
                const maxTick = Math.max(1, stem.notes.reduce((max, n) => Math.max(max, n.tick + n.duration), 0));
                // INSTRUMENT tracks (not "vocal"): a vocal track requires a SoVITS voice model, which
                // surfaces as a "缺少模型" error and produces no sound on import. MIDI tracks are
                // instruments — keyed to their synth stems. To make the imported track AUDIBLY play its
                // instrument, attach the per-instrument synthesized WAV as a lane output (the same
                // processedOutputs channel the DAW playback engine schedules).
                let processedOutputs = [];
                if (stem.audioPath) {
                    const safePpq = ppq > 0 ? ppq : 480;
                    const safeBpm = bpm > 0 ? bpm : 120;
                    let totalDurationMs = Math.max(1, (maxTick / (safePpq * (safeBpm / 60))) * 1000);
                    let waveformPeaks;
                    try {
                        const data = await useAudioStore.getState().loadAudioFile(stem.audioPath);
                        if (data.durationMs > 0)
                            totalDurationMs = data.durationMs;
                        if (data.peaks && data.peaks.length > 0)
                            waveformPeaks = data.peaks;
                    }
                    catch { /* best-effort: fall back to the note-derived duration */ }
                    processedOutputs.push({
                        laneId: segmentId,
                        laneLabel: stem.name,
                        group: stem.name,
                        audioPath: stem.audioPath,
                        totalDurationMs,
                        waveformPeaks,
                    });
                }
                // Vocal stems import as VOCAL tracks (§user "干音都要有歌词")：有歌词 → 编辑器
                // 打开就是歌词；干音轨即使自动歌词没跑成（未装 Whisper）也以人声轨导入，
                // 编辑器里照样能填词 / 看音素。乐器轨不带 lyric → 编辑器显示音名 C4 / D#3。
                const stemHasLyrics = stem.notes.some((n) => String(n.lyric ?? "").trim() !== "");
                const stemIsVocal = isVocalTrackName(stem.name);
                addTrack({
                    id: newTrackId,
                    name: stem.name,
                    trackType: stemIsVocal || stemHasLyrics ? "vocal" : "instrument",
                    volumeDb: 0,
                    pan: 0,
                    muted: false,
                    solo: false,
                    expanded: true,
                    laneControls: {},
                    segments: [{
                            id: segmentId,
                            startTick: 0,
                            durationTicks: maxTick,
                            content: {
                                type: "notes",
                                notes: stem.notes.map(n => ({
                                    id: crypto.randomUUID(),
                                    tick: n.tick,
                                    duration: n.duration,
                                    pitch: n.pitch,
                                    lyric: (n.lyric ?? ""),
                                    velocity: n.velocity ?? 100
                                }))
                            },
                            processedOutputs: processedOutputs.length > 0 ? processedOutputs : undefined,
                        }]
                });
            }
            useHistoryStore.getState().commitTransaction();
            showToast(t("amt.importSuccess") || "Import successful", "success");
            onClose();
        }
        catch (e) {
            useHistoryStore.getState().commitTransaction();
            showToast(String(e), "error");
        }
    };
    // ── Download menu exports (reference: _save_asset / _start_midi_audio_export / _start_sheet_music_export) ──
    /** Base file name for export suggestions (first conversion output's stem). */
    const exportSourceName = () => {
        const first = result.midiPaths[0] || "transcription";
        return (first
            .split(/[/\\]/)
            .pop()
            ?.replace(/\.(midi?|MIDI?)$/i, "")
            .replace(/[^\w\-\u4e00-\u9fff ]+/g, "_") || "transcription");
    };
    /** Map a stable backend EXPORT_* code onto a localized user message. */
    const mapExportError = (msg) => {
        if (msg.includes("EXPORT_FLUIDSYNTH_NOT_FOUND"))
            return t("amt.exportErrFluidsynth") || "FluidSynth 未安装，请先在资源管理中下载";
        if (msg.includes("EXPORT_MUSESCORE_NOT_FOUND"))
            return t("amt.exportErrMusescore") || "MuseScore 未安装，请先在资源管理中下载";
        if (msg.includes("EXPORT_SOUNDFONT_NOT_FOUND"))
            return t("amt.exportErrSoundfont") || "音源缺失，请先在资源管理中下载音源";
        if (msg.includes("EXPORT_SOUNDFONT_BAD_TYPE"))
            return t("amt.sfErrBadType") || "仅支持 SF2 / SF3 格式的音源文件";
        if (msg.includes("EXPORT_SOUNDFONT_BUILTIN"))
            return t("amt.sfErrBuiltin") || "内置音源不可删除";
        if (msg.startsWith("EXPORT_SRC_NOT_FOUND"))
            return t("amt.exportErrSrc") || "源文件不存在，请重新转换";
        if (msg.startsWith("EXPORT_SHEET_FAILED"))
            return `${t("amt.exportErrSheet") || "乐谱导出失败"}：${msg.replace(/^EXPORT_SHEET_FAILED:?\s*/, "")}`;
        if (msg.startsWith("EXPORT_RENDER_FAILED"))
            return `${t("amt.exportErrRender") || "音频渲染失败"}：${msg.replace(/^EXPORT_RENDER_FAILED:?\s*/, "")}`;
        if (msg.startsWith("EXPORT_WRITE_FAILED"))
            return `${t("amt.exportErrWrite") || "文件写入失败"}：${msg.replace(/^EXPORT_WRITE_FAILED:?\s*/, "")}`;
        return msg;
    };
    /** Write the CURRENT score (notes + edits + project tempo) to a snapshot MIDI and return
     *  its path. Every audio/sheet export renders the exact state the user hears — NOT the
     *  original conversion output (reference snapshot semantics). */
    const writeCurrentMidiSnapshot = async () => {
        const base = result.outputDir?.replace(/[\/\\]+$/, "") ||
            (stems[0]?.midiPath || "").replace(/[/\\][^/\\]+$/, "");
        if (!base)
            throw new Error("EXPORT_SRC_NOT_FOUND: no output dir");
        const outDir = `${base}/export_snapshots`;
        const midiPath = `${outDir}/snapshot_${Date.now()}.mid`;
        await invoke("amt_write_edited_midi", {
            outPath: midiPath,
            // Project tempo (what playback actually uses) — the reference commits the tempo
            // edit into every exported MIDI.
            bpm: projBpm || detectedBpm || bpm || 120,
            tracks: stems.map((s) => ({
                name: s.name,
                program: s.program ?? null,
                channel: s.channel ?? null,
                notes: s.notes.map((n) => ({
                    tick: Math.round(n.tick),
                    duration: Math.round(n.duration),
                    pitch: n.pitch,
                    velocity: n.velocity ?? 100,
                })),
            })),
        });
        return midiPath;
    };
    const handleExportZip = async () => {
        setDownloadOpen(false);
        if (exportBusy)
            return;
        try {
            // MIDI view: export EVERY instrument as its own .mid file (NOT a single merged file).
            // `export_midi_tracks_to_folder` splits a merged multi-track MIDI into per-instrument
            // files in a user-chosen folder — one file per instrument, one instrument per file.
            if (!isSeparation) {
                const { open } = await import("@tauri-apps/plugin-dialog");
                let dir = await open({ directory: true, title: t("amt.exportFolderTitle") });
                if (!dir)
                    return;
                if (Array.isArray(dir))
                    dir = dir[0];
                if (typeof dir !== "string")
                    return;
                setExportBusy("midi");
                // Edits present → export the CURRENT snapshot instead of the raw conversion output.
                const sources = editDirty
                    ? [await writeCurrentMidiSnapshot()]
                    : result.midiPaths.filter(Boolean);
                let exported = 0;
                for (const midiPath of sources) {
                    try {
                        const written = await invoke("export_midi_tracks_to_folder", {
                            midiPath,
                            folder: dir,
                        });
                        exported += written.length;
                    }
                    catch (e) {
                        // Fall through so other files still export; surface the first real failure below.
                        console.warn("export_midi_tracks_to_folder failed for", midiPath, e);
                    }
                }
                showToast(exported > 0
                    ? (t("amt.exportTracksDone", { count: exported }) || `已导出 ${exported} 个乐器 MIDI 文件`)
                    : (t("amt.exportNoFiles") || "没有可导出的 MIDI 文件"), exported > 0 ? "success" : "error");
                return;
            }
            // Separation view exports the stem WAVs as a ZIP.
            const { save } = await import("@tauri-apps/plugin-dialog");
            const savePath = await save({
                filters: [{ name: "ZIP Archive", extensions: ["zip"] }],
                defaultPath: "separated_stems.zip"
            });
            if (!savePath)
                return;
            const pathsToZip = stems.map((s) => s.audioPath).filter(Boolean);
            await invoke("amt_export_zip", {
                midiPaths: pathsToZip,
                savePath
            });
            showToast(t("amt.exportSuccess") || "Export successful", "success");
        }
        catch (e) {
            const msg = String(e);
            let userMsg = msg;
            if (msg.includes("EXPORT_PERMISSION_DENIED")) {
                userMsg = t("amt.exportPermissionDenied") || "Permission denied: The target file might be in use by another program.";
            }
            else if (msg.includes("EXPORT_DIR_NOT_FOUND")) {
                userMsg = t("amt.exportDirNotFound") || "Target directory not found.";
            }
            else if (msg.includes("EXPORT_NO_FILES")) {
                userMsg = t("amt.exportNoFiles") || "No valid MIDI files found to export.";
            }
            else {
                userMsg = mapExportError(msg);
            }
            showToast(userMsg, "error");
        }
        finally {
            setExportBusy(null);
        }
    };
    /** 乐谱导出：MuseScore 刻写 MusicXML + 全谱 PDF，打 zip（reference sheet music package）。 */
    const handleExportSheetMusic = async () => {
        setDownloadOpen(false);
        if (exportBusy)
            return;
        setExportBusy("sheet");
        try {
            const { save } = await import("@tauri-apps/plugin-dialog");
            const out = await save({
                filters: [{ name: "ZIP", extensions: ["zip"] }],
                defaultPath: `${exportSourceName()}_sheet_music.zip`,
            });
            if (!out)
                return;
            const midi = await writeCurrentMidiSnapshot();
            const res = await invoke("amt_export_sheet_music", {
                midiPath: midi,
                outPath: out.toLowerCase().endsWith(".zip") ? out : `${out}.zip`,
            });
            showToast(t("amt.exportSheetSaved", { path: res.zipPath }) || "乐谱包已导出", "success");
        }
        catch (e) {
            showToast(mapExportError(String(e)), "error");
        }
        finally {
            setExportBusy(null);
        }
    };
    /** 转录音频导出：整首 MIDI → 单个 WAV（HQ 24bit/48kHz | 兼容 16bit/44.1kHz）。 */
    const handleExportTranscriptionAudio = async (preset) => {
        setDownloadOpen(false);
        if (exportBusy)
            return;
        setExportBusy(`transcription-${preset}`);
        try {
            const { save } = await import("@tauri-apps/plugin-dialog");
            const suffix = preset === "hq" ? "24bit-48kHz" : "16bit-44.1kHz";
            const out = await save({
                filters: [{ name: "WAV", extensions: ["wav"] }],
                defaultPath: `${exportSourceName()}_transcription_${suffix}.wav`,
            });
            if (!out)
                return;
            const midi = await writeCurrentMidiSnapshot();
            const res = await invoke("amt_export_midi_audio", {
                midiPath: midi,
                outPath: out.toLowerCase().endsWith(".wav") ? out : `${out}.wav`,
                preset,
            });
            showToast(t("amt.exportAudioSaved", { path: res }) || "转录音频已导出", "success");
        }
        catch (e) {
            showToast(mapExportError(String(e)), "error");
        }
        finally {
            setExportBusy(null);
        }
    };
    /** 分轨音频导出：每乐器一个 WAV，打 zip（reference stem_archive）。 */
    const handleExportStemsAudio = async (preset) => {
        setDownloadOpen(false);
        if (exportBusy)
            return;
        setExportBusy(`stems-${preset}`);
        try {
            const { save } = await import("@tauri-apps/plugin-dialog");
            const suffix = preset === "hq" ? "24bit-48kHz" : "16bit-44.1kHz";
            const out = await save({
                filters: [{ name: "ZIP", extensions: ["zip"] }],
                defaultPath: `${exportSourceName()}_instrument-stems_${suffix}.zip`,
            });
            if (!out)
                return;
            const midi = await writeCurrentMidiSnapshot();
            const res = await invoke("amt_export_stems_audio", {
                midiPath: midi,
                outPath: out.toLowerCase().endsWith(".zip") ? out : `${out}.zip`,
                preset,
            });
            showToast(t("amt.exportStemsSaved", { count: res.members.length, path: res.zipPath }) ||
                `已导出 ${res.members.length} 个乐器 WAV`, "success");
        }
        catch (e) {
            showToast(mapExportError(String(e)), "error");
        }
        finally {
            setExportBusy(null);
        }
    };
    /** 立体声 A/B 导出：左=原声、右=MIDI 合成（reference stereo mix）。 */
    const handleExportStereoWav = async () => {
        setDownloadOpen(false);
        if (exportBusy)
            return;
        setExportBusy("stereo");
        try {
            const original = result.sourceAudioPath || effOriginal;
            if (!original) {
                showToast(t("amt.exportErrNoOriginal") || "缺少原声文件，无法导出立体声对比", "error");
                return;
            }
            const { save } = await import("@tauri-apps/plugin-dialog");
            const out = await save({
                filters: [{ name: "WAV", extensions: ["wav"] }],
                defaultPath: `${exportSourceName()}_stereo.wav`,
            });
            if (!out)
                return;
            const midi = await writeCurrentMidiSnapshot();
            const res = await invoke("amt_export_stereo_wav", {
                originalAudioPath: original,
                midiPath: midi,
                outPath: out.toLowerCase().endsWith(".wav") ? out : `${out}.wav`,
            });
            showToast(t("amt.exportStereoSaved", { path: res }) || "立体声对比已导出", "success");
        }
        catch (e) {
            showToast(mapExportError(String(e)), "error");
        }
        finally {
            setExportBusy(null);
        }
    };
    const formatTime = (secs) => {
        const m = Math.floor(secs / 60);
        const s = Math.floor(secs % 60);
        const ms = Math.floor((secs % 1) * 100);
        return `${m}:${s.toString().padStart(2, "0")}.${ms.toString().padStart(2, "0")}`;
    };
    const pianoKeys = useMemo(() => {
        const keys = [];
        for (let i = 127; i >= 0; i--) {
            const isBlack = [1, 3, 6, 8, 10].includes(i % 12);
            const noteNames = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
            const noteName = noteNames[i % 12];
            const octave = Math.floor(i / 12) - 1;
            keys.push(_jsx("div", { className: `amt-key ${isBlack ? "black" : ""}`, style: { height: ROW_HEIGHT }, children: i % 12 === 0 ? _jsx("span", { className: "amt-key-label", children: `${noteName}${octave}` }) : "" }, i));
        }
        return keys;
    }, [ROW_HEIGHT]);
    return (_jsxs("div", { className: "amt-result-panel", style: style, children: [isLoading && (_jsx("div", { className: "amt-loading-overlay", children: _jsx("div", { className: "amt-spinner" }) })), _jsxs("div", { className: "amt-result-header", children: [_jsxs("div", { className: "amt-header-row-top", children: [_jsxs("div", { className: "amt-header-left", children: [_jsx("button", { className: "amt-close-btn", onClick: onClose, title: t("common.close"), children: "\u2715" }), _jsxs("div", { className: "amt-title-group", children: [_jsx("h3", { children: t("amt.resultWorkbench") }), _jsx("span", { className: "amt-subtitle", children: isSeparation
                                                    ? t("amt.separationDesc", { count: stems.length })
                                                    : t("amt.resultDesc", { count: result.midiPaths.length }) })] })] }), _jsxs("div", { className: "amt-header-actions", children: [_jsxs("button", { className: "amt-action-btn secondary", onClick: handleAnother, title: t("amt.transcribeAnotherTooltip") || "关闭当前结果并重新打开转换对话框", children: [_jsx("span", { className: "icon", children: "\uD83C\uDFB5" }), " ", t("amt.transcribeAnother") || "再转一首"] }), !isSeparation && (_jsxs("button", { className: `amt-action-btn secondary ${lyricsOpen ? "active" : ""}`, onClick: () => setLyricsOpen((v) => !v), title: t("amt.lyricsTooltip") || "从歌曲中识别歌词（自动检测语言），编辑后写回 MIDI", children: [_jsx("span", { className: "icon", children: "\uD83C\uDFA4" }), " ", t("amt.lyricsBtn") || "歌词"] })), !isSeparation && (_jsxs("div", { className: "amt-download-menu", ref: sfMenuRef, children: [_jsxs("button", { className: "amt-action-btn secondary", onClick: () => setSfOpen((v) => !v), title: t("amt.sfTooltip") || "切换 MIDI 合成音源（SoundFont），试听与导出共用", children: [_jsx("span", { className: "icon", children: "\uD83C\uDFB9" }), " ", t("amt.sfLabel") || "音源", _jsx("span", { className: "amt-download-caret", children: "\u25BE" })] }), sfOpen && (_jsxs("div", { className: "amt-download-panel amt-sf-panel", role: "menu", children: [_jsx("div", { className: "amt-download-group-title", children: t("amt.sfListTitle") || "选择音源" }), soundfonts.length === 0 && (_jsx("div", { className: "amt-sf-empty", children: t("amt.sfEmpty") || "未找到音源，请先下载内置音源" })), soundfonts.map((sf) => (_jsxs("div", { className: `amt-sf-row ${sf.active ? "active" : ""}`, children: [_jsxs("button", { className: "amt-download-item", role: "menuitemradio", "aria-checked": sf.active, onClick: () => handleSwitchSoundfont(sf.filename, sf.name), title: sf.filename, children: [_jsx("span", { className: "amt-download-icon", children: sf.active ? "●" : "○" }), _jsxs("span", { className: "amt-download-label", children: [sf.name, sf.isBuiltin ? `（${t("amt.sfBuiltin") || "内置"}）` : ""] })] }), !sf.isBuiltin && (_jsx("button", { className: "amt-sf-delete", title: t("amt.sfDeleteTooltip") || "删除该音源", onClick: () => handleDeleteSoundfont(sf.filename, sf.name), children: "\u2715" }))] }, sf.filename))), _jsxs("button", { className: "amt-download-item", onClick: handleImportSoundfont, children: [_jsx("span", { className: "amt-download-icon", children: "\uFF0B" }), _jsx("span", { className: "amt-download-label", children: t("amt.sfImport") || "导入音源（SF2 / SF3）…" })] })] }))] })), isSeparation ? (_jsxs("button", { className: "amt-action-btn secondary", onClick: handleExportZip, children: [_jsx("span", { className: "icon", children: "\uD83D\uDCE6" }), " ", t("amt.exportStems") || "导出分轨"] })) : (_jsxs("div", { className: "amt-download-menu", ref: downloadRef, children: [_jsxs("button", { className: `amt-action-btn secondary ${exportBusy ? "busy" : ""}`, onClick: () => setDownloadOpen((v) => !v), disabled: !!exportBusy, title: t("amt.downloadTooltip") || "导出 MIDI / 乐谱 / 音频", children: [_jsx("span", { className: "icon", children: "\uD83D\uDCE6" }), " ", exportBusy
                                                        ? (t("amt.exportRendering") || "渲染中…")
                                                        : t("amt.download") || "下载", !exportBusy && _jsx("span", { className: "amt-download-caret", children: "\u25BE" })] }), downloadOpen && (_jsxs("div", { className: "amt-download-panel", role: "menu", children: [_jsxs("button", { className: "amt-download-item", role: "menuitem", disabled: !!exportBusy, onClick: handleExportZip, title: t("amt.downloadMidiTooltip") || "每个乐器一个 .mid 文件（含当前编辑）", children: [_jsx("span", { className: "amt-download-icon", children: "\uD83C\uDFBC" }), _jsx("span", { className: "amt-download-label", children: t("amt.downloadMidi") || "导出可编辑 MIDI" }), editDirty && _jsx("span", { className: "amt-download-badge", children: t("amt.downloadEdited") || "已编辑" })] }), _jsxs("button", { className: "amt-download-item", role: "menuitem", disabled: !!exportBusy || !exportTools?.musescoreAvailable, title: exportTools?.musescoreAvailable
                                                            ? (t("amt.downloadSheetTooltip") || "MusicXML + PDF 乐谱包")
                                                            : (t("amt.exportErrMusescore") || "MuseScore 未安装，请先在资源管理中下载"), onClick: handleExportSheetMusic, children: [_jsx("span", { className: "amt-download-icon", children: "\uD83D\uDCC4" }), _jsx("span", { className: "amt-download-label", children: t("amt.downloadSheetMusic") || "乐谱包（MusicXML / PDF）" })] }), _jsxs("div", { className: "amt-download-group", children: [_jsx("div", { className: "amt-download-group-title", children: t("amt.downloadTranscription") || "转写合成 WAV" }), _jsxs("button", { className: "amt-download-item", role: "menuitem", disabled: !!exportBusy || !exportTools?.fluidsynthAvailable || !exportTools?.soundfontAvailable, onClick: () => handleExportTranscriptionAudio("hq"), children: [_jsx("span", { className: "amt-download-icon", children: "\uD83C\uDFA7" }), _jsx("span", { className: "amt-download-label", children: t("amt.audioHq") || "24-bit / 48 kHz（推荐）" })] }), _jsxs("button", { className: "amt-download-item", role: "menuitem", disabled: !!exportBusy || !exportTools?.fluidsynthAvailable || !exportTools?.soundfontAvailable, onClick: () => handleExportTranscriptionAudio("compat"), children: [_jsx("span", { className: "amt-download-icon", children: "\uD83C\uDFA7" }), _jsx("span", { className: "amt-download-label", children: t("amt.audioCompat") || "16-bit / 44.1 kHz（兼容）" })] })] }), _jsxs("div", { className: "amt-download-group", children: [_jsx("div", { className: "amt-download-group-title", children: t("amt.downloadStemsAudio") || "下载分乐器 WAV（ZIP）" }), _jsxs("button", { className: "amt-download-item", role: "menuitem", disabled: !!exportBusy || !exportTools?.fluidsynthAvailable || !exportTools?.soundfontAvailable, onClick: () => handleExportStemsAudio("hq"), children: [_jsx("span", { className: "amt-download-icon", children: "\uD83C\uDF9A" }), _jsx("span", { className: "amt-download-label", children: t("amt.audioHq") || "24-bit / 48 kHz（推荐）" })] }), _jsxs("button", { className: "amt-download-item", role: "menuitem", disabled: !!exportBusy || !exportTools?.fluidsynthAvailable || !exportTools?.soundfontAvailable, onClick: () => handleExportStemsAudio("compat"), children: [_jsx("span", { className: "amt-download-icon", children: "\uD83C\uDF9A" }), _jsx("span", { className: "amt-download-label", children: t("amt.audioCompat") || "16-bit / 44.1 kHz（兼容）" })] })] }), _jsxs("button", { className: "amt-download-item", role: "menuitem", disabled: !!exportBusy || !exportTools?.fluidsynthAvailable || !exportTools?.soundfontAvailable, title: t("amt.downloadStereoTooltip") || "左声道原声 / 右声道 MIDI 的对比音频", onClick: handleExportStereoWav, children: [_jsx("span", { className: "amt-download-icon", children: "\u2194" }), _jsx("span", { className: "amt-download-label", children: t("amt.downloadStereo") || "原音左声道 + MIDI 右声道 WAV" })] })] }))] })), !isSeparation && (_jsxs("button", { className: "amt-action-btn primary", onClick: handleImportAll, children: [_jsx("span", { className: "icon", children: "\uD83D\uDCE5" }), " ", t("amt.importToTracks")] }))] })] }), !isSeparation && (_jsxs("div", { className: "amt-tabs-row", children: [_jsxs("button", { className: `amt-tab-btn ${activeTab === "edit" ? "active" : ""}`, onClick: () => setActiveTab("edit"), title: t("amt.tabEditTooltip") || "编辑 MIDI 分轨、试听合成效果", children: ["\uD83C\uDFB5 ", t("amt.tabEdit", "编辑试听")] }), _jsxs("button", { className: `amt-tab-btn ${activeTab === "export" ? "active" : ""}`, onClick: () => setActiveTab("export"), title: t("amt.tabExportTooltip") || "一键导入工程或下载导出文件", children: ["\uD83D\uDCE5 ", t("amt.tabExport", "导出")] }), _jsxs("button", { className: `amt-tab-btn ${activeTab === "lyrics" ? "active" : ""}`, onClick: () => { setActiveTab("lyrics"); setLyricsOpen(true); }, title: t("amt.tabLyricsTooltip") || "识别并编辑歌词", children: ["\uD83C\uDFA4 ", t("amt.tabLyrics", "歌词")] })] })), _jsxs("div", { className: "amt-transport-bar", children: [_jsxs("div", { className: "amt-transport-group amt-transport-left", children: [_jsxs("div", { className: "amt-source-switch", children: [_jsx("button", { className: `amt-switch-btn ${playSource === "midi" ? "active" : ""}`, onClick: () => { setPlaySource("midi"); setMixPercent(100); }, children: isSeparation ? (t("amt.stemsPlayback") || "分轨") : "MIDI" }), _jsx("button", { className: `amt-switch-btn ${playSource === "original" ? "active" : ""}`, onClick: () => { setPlaySource("original"); setMixPercent(0); }, children: t("amt.original") || "原声" })] }), _jsxs("div", { className: "amt-mix-control", title: t("amt.mixTooltip") || "原声 ↔ MIDI 混音比例", children: [_jsx("span", { className: "amt-mix-label", children: t("amt.original") || "原声" }), _jsx("input", { type: "range", className: "amt-mix-slider", min: 0, max: 100, step: 1, value: mixPercent, disabled: stereoMode, onChange: (e) => {
                                                    const v = parseInt(e.target.value, 10);
                                                    setMixPercent(v);
                                                    setPlaySource(v >= 50 ? "midi" : "original");
                                                } }), _jsx("span", { className: "amt-mix-label", children: isSeparation ? (t("amt.stemsPlayback") || "分轨") : "MIDI" })] }), _jsxs("label", { className: "amt-stereo-toggle", title: t("amt.stereoTooltip") || "左声道原声 / 右声道 MIDI 对比", children: [_jsx("input", { type: "checkbox", checked: stereoMode, onChange: (e) => setStereoMode(e.target.checked) }), t("amt.stereoAB") || "A/B"] })] }), _jsxs("div", { className: "amt-transport-center", children: [_jsx("button", { className: "amt-transport-btn", onClick: stopPlay, title: t("common.stop") || "Stop", children: "\u23F9" }), _jsx("button", { className: `amt-transport-btn play ${isPlaying ? "playing" : ""}`, onClick: togglePlay, children: isPlaying ? "⏸" : "▶" }), _jsxs("div", { className: "amt-time-display", children: [_jsx("span", { className: "current", children: formatTime(currentTime) }), _jsx("span", { className: "divider", children: "/" }), _jsx("span", { className: "total", children: formatTime(duration) })] })] }), _jsxs("div", { className: "amt-transport-group amt-transport-right", children: [!isSeparation && (_jsxs("label", { className: "amt-follow-toggle", title: t("amt.followTooltip") || "播放时自动滚动钢琴卷帘跟随播放头", children: [_jsx("input", { type: "checkbox", checked: followMode, onChange: (e) => setFollowMode(e.target.checked) }), t("amt.follow") || "跟随"] })), !isSeparation && detectedBpm != null && detectedBpm > 0 && (_jsxs("div", { className: "amt-tempo-controls", children: [_jsxs("label", { className: "amt-tempo-field", title: t("amt.projBpmTooltip") || "设置试听和导出 MIDI 的 BPM。提高 BPM 会加快播放、缩短时长。", children: [_jsx("span", { className: "amt-tempo-label", children: t("amt.projectTempo") || "工程速度" }), _jsx("input", { type: "number", min: 4, max: 400, step: 0.1, value: Number(projBpm.toFixed(1)), onChange: (e) => onProjBpmChange(parseFloat(e.target.value) || detectedBpm) })] }), _jsxs("label", { className: "amt-tempo-field", title: t("amt.speedTooltip") || "当前工程 BPM 对应的播放倍率（工程 BPM ÷ 检测 BPM）。修改该倍率会反向更新工程 BPM。", children: [_jsx("span", { className: "amt-tempo-label", children: t("amt.playbackSpeed") || "工程倍速" }), _jsx("input", { type: "number", min: 0.05, max: 20, step: 0.05, value: Number(speed.toFixed(3)), onChange: (e) => onSpeedChange(parseFloat(e.target.value) || 1) })] }), _jsx("span", { className: "amt-tempo-status", children: t("amt.tempoDetected", { bpm: Math.round(detectedBpm * 10) / 10 }) || `检测 BPM ${detectedBpm.toFixed(1)}` })] }))] })] }), _jsx("div", { className: "amt-header-progress", children: _jsx("input", { type: "range", className: "amt-progress-slider", min: 0, max: duration || 1, step: 0.01, value: currentTime, onChange: (e) => handleSeek(parseFloat(e.target.value)) }) })] }), _jsxs("div", { className: "amt-master-track", children: [_jsxs("div", { className: "amt-master-track-label", children: [_jsx("span", { className: "amt-master-track-name", children: isSeparation
                                    ? (playSource === "midi" ? (t("amt.stemsPlayback") || "分轨播放") : (t("amt.original") || "原声"))
                                    : playSource === "midi" ? (t("amt.midiMaster") || "MIDI 合成") : (t("amt.original") || "原声") }), _jsxs("span", { className: "amt-master-track-meta", children: [formatTime(currentTime), " / ", formatTime(duration)] })] }), _jsx("div", { className: "amt-master-track-canvas-wrap", onMouseMove: (e) => {
                            const rect = e.currentTarget.getBoundingClientRect();
                            if (rect.width <= 0)
                                return;
                            setMasterHover(true);
                            setMasterHoverFrac(Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width)));
                        }, onMouseLeave: () => setMasterHover(false), onMouseDown: (e) => {
                            const rect = e.currentTarget.getBoundingClientRect();
                            if (rect.width <= 0)
                                return;
                            seekMasterTrack((e.clientX - rect.left) / rect.width);
                        }, children: _jsx("canvas", { ref: masterCanvasRef, className: "amt-master-track-canvas" }) })] }), _jsxs("div", { className: "amt-result-content", children: [_jsxs("div", { className: "amt-sidebar", children: [_jsx("div", { className: "amt-sidebar-header", children: t("amt.tracks") }), _jsx("div", { className: "amt-stem-list", children: stems.map((stem, idx) => (_jsxs("div", { className: `amt-stem-item ${activeStemIdx === idx ? "active" : ""}`, onClick: () => setActiveStemIdx(idx), children: [_jsx("div", { className: "amt-stem-color-bar", style: { backgroundColor: stem.color } }), _jsxs("div", { className: "amt-stem-main", children: [_jsx("div", { className: "amt-stem-name", style: { color: activeStemIdx === idx ? stem.color : "" }, children: stem.name }), _jsx("div", { className: "amt-stem-meta", children: stem.notes.length > 0
                                                        ? _jsxs(_Fragment, { children: [stem.notes.length, " ", t("amt.notesCount")] })
                                                        : (stem.audioPath ? (t("amt.audioStem") || "音频分轨") : "—") })] }), _jsxs("div", { className: "amt-stem-controls", children: [isSeparation && stem.audioPath && (_jsx("button", { className: "amt-mini-btn convert", title: t("amt.stemToMidi") || "该分轨转 MIDI", onClick: (e) => {
                                                        e.stopPropagation();
                                                        // Same forced-source flow as the track right-click "音乐转MIDI":
                                                        // set the stem audio as the AMT source, then open the conversion dialog.
                                                        const store = useAppStore.getState();
                                                        store.setAmtSource(stem.audioPath, stem.name);
                                                        store.openAmtConversion(result.trackId);
                                                    }, children: "\u266A\u2192M" })), _jsx("button", { className: `amt-mini-btn mute ${stem.muted ? "active" : ""}`, title: t("amt.stemMuteTooltip") || "静音此乐器（点击任意静音会清除独奏）", onClick: (e) => {
                                                        e.stopPropagation();
                                                        toggleStemMute(idx);
                                                    }, children: "M" }), _jsx("button", { className: `amt-mini-btn solo ${stem.soloed ? "active" : ""}`, title: t("amt.stemSoloTooltip") || "独奏（静音其他乐器）", onClick: (e) => {
                                                        e.stopPropagation();
                                                        toggleStemSolo(idx);
                                                    }, children: "S" })] })] }, idx))) })] }), isSeparation ? (
                    /* Separation view: one waveform row per stem, time-aligned, click to seek. */
                    _jsx("div", { className: "amt-sep-rows", children: stems.map((stem, idx) => (_jsxs("div", { className: `amt-sep-row ${activeStemIdx === idx ? "active" : ""} ${stem.muted ? "muted" : ""}`, onClick: () => setActiveStemIdx(idx), children: [_jsx("div", { className: "amt-sep-row-name", style: { color: stem.color }, children: stem.name }), _jsx("div", { className: "amt-sep-row-wave", onMouseDown: (e) => {
                                        const rect = e.currentTarget.getBoundingClientRect();
                                        if (rect.width <= 0)
                                            return;
                                        seekMasterTrack((e.clientX - rect.left) / rect.width);
                                    }, children: _jsx(SepWaveCanvas, { peaks: stemPeaks[stem.name] ?? [], color: stem.color, active: activeStemIdx === idx && !stem.muted, playFrac: duration > 0 ? Math.min(1, currentTime / duration) : 0 }) }), _jsx("div", { className: "amt-sep-row-badge", children: stem.muted ? t("amt.muted") || "静音" : (stem.soloed ? "SOLO" : "WAV") })] }, idx))) })) : (_jsxs("div", { className: "amt-piano-roll-container", children: [_jsxs("div", { className: `amt-editor-toolbar ${editMode ? "active" : ""}`, children: [_jsxs("div", { className: "amt-editor-row", children: [_jsx("button", { className: `amt-ed-btn toggle ${editMode ? "on" : ""}`, onClick: () => setEditMode((v) => !v), title: t("amt.editorHelp"), children: t("amt.editorToggle") || "编辑 MIDI" }), editMode && (_jsxs(_Fragment, { children: [_jsx("button", { className: "amt-ed-btn", onClick: addNoteAtPlayhead, children: t("amt.editorAdd") || "新增音符" }), _jsx("button", { className: "amt-ed-btn", disabled: selNotes.size === 0, onClick: deleteSelection, children: t("amt.editorDelete") || "删除" }), _jsx("button", { className: "amt-ed-btn", disabled: undoRef.current.length === 0, onClick: undoEdit, children: t("amt.editorUndo") || "撤销" }), _jsx("button", { className: "amt-ed-btn", disabled: redoRef.current.length === 0, onClick: redoEdit, children: t("amt.editorRedo") || "重做" }), _jsx("button", { className: "amt-ed-btn", disabled: !editDirty, onClick: resetEdits, children: t("amt.editorReset") || "还原" }), _jsx("button", { className: "amt-ed-btn", onClick: selectAllNotes, children: t("amt.editorSelectAll") || "全选" }), _jsx("button", { className: "amt-ed-btn", disabled: selNotes.size === 0, onClick: cutSelection, children: t("amt.editorCut") || "剪切" }), _jsx("button", { className: "amt-ed-btn", disabled: selNotes.size === 0, onClick: copySelection, children: t("amt.editorCopy") || "复制" }), _jsx("button", { className: "amt-ed-btn", disabled: clipboardRef.current.length === 0, onClick: pasteClipboard, children: t("amt.editorPaste") || "粘贴" }), _jsx("button", { className: "amt-ed-btn", disabled: selNotes.size === 0, onClick: duplicateSelection, children: t("amt.editorDuplicate") || "向右复制" })] })), _jsxs("div", { className: "amt-editor-row-relative", children: [_jsx("button", { className: `amt-ed-btn${rollHelpOpen ? " on" : ""}`, title: t("amt.rollHelpTitle"), onClick: () => setRollHelpOpen((o) => !o), children: "?" }), rollHelpOpen && (_jsxs("div", { className: "amt-help-card", role: "dialog", "aria-label": t("amt.rollHelpTitle"), children: [_jsxs("div", { className: "amt-help-card-head", children: [_jsx("span", { children: t("amt.rollHelpTitle") }), _jsx("button", { className: "amt-help-close", onClick: () => setRollHelpOpen(false), children: "\u00D7" })] }), _jsx("table", { className: "amt-help-table", children: _jsx("tbody", { children: [
                                                                        ["rollHelpUndo", "rollHelpUndoKey"],
                                                                        ["rollHelpSelect", "rollHelpSelectKey"],
                                                                        ["rollHelpClip", "rollHelpClipKey"],
                                                                        ["rollHelpQuantize", "rollHelpQuantizeKey"],
                                                                        ["rollHelpDelete", "rollHelpDeleteKey"],
                                                                        ["rollHelpTime", "rollHelpTimeKey"],
                                                                        ["rollHelpPitch", "rollHelpPitchKey"],
                                                                        ["rollHelpVel", "rollHelpVelKey"],
                                                                        ["rollHelpZoom", "rollHelpZoomKey"],
                                                                        ["rollHelpEscape", "rollHelpEscapeKey"],
                                                                        ["rollHelpMouseAdd", "rollHelpMouseAddKey"],
                                                                        ["rollHelpMouseResize", "rollHelpMouseResizeKey"],
                                                                    ].map(([k, kk]) => (_jsxs("tr", { children: [_jsx("td", { className: "amt-help-key", children: _jsx("kbd", { children: t(`amt.${kk}`) }) }), _jsx("td", { children: t(`amt.${k}`) })] }, k))) }) })] }))] })] }), editMode && (_jsxs("div", { className: "amt-editor-row fields", children: [_jsxs("label", { className: "amt-ed-field", children: [_jsx("span", { children: t("amt.editorQuantizeScope") || "量化范围" }), _jsxs("select", { value: quantScope, title: t("amt.editorQuantizeScopeTooltip"), onChange: (e) => setQuantScope(e.target.value), children: [_jsx("option", { value: "all_tracks", children: t("amt.editorQuantizeScopeAll") || "全部轨道" }), _jsx("option", { value: "selected_notes", children: t("amt.editorQuantizeScopeSelected") || "所选音符" })] })] }), _jsxs("label", { className: "amt-ed-field", children: [_jsx("span", { children: t("amt.editorQuantizeGrid") || "量化网格" }), _jsxs("select", { value: quantGrid, title: t("amt.editorQuantizeGridTooltip"), onChange: (e) => setQuantGrid(parseInt(e.target.value, 10)), children: [_jsx("option", { value: 4, children: "1/4" }), _jsx("option", { value: 8, children: "1/8" }), _jsx("option", { value: 16, children: "1/16" }), _jsx("option", { value: 32, children: "1/32" }), _jsx("option", { value: 64, children: "1/64" })] })] }), _jsx("button", { className: "amt-ed-btn accent", onClick: quantizeSelection, disabled: quantScope === "selected_notes" && selNotes.size === 0, children: t("amt.editorQuantize") || "量化" }), _jsx("button", { className: "amt-ed-btn accent", onClick: smartCleanup, title: t("amt.editorCleanTooltip"), children: t("amt.editorClean") || "一键修复" }), _jsxs("label", { className: "amt-ed-field", children: [_jsx("span", { children: t("amt.editorInstrument") || "当前编辑乐器" }), _jsx("select", { value: activeInstr, onChange: (e) => setActiveInstr(e.target.value), children: stems.filter((s) => s.notes.length > 0).map((s) => (_jsx("option", { value: s.name, children: s.name }, s.name))) })] }), _jsxs("label", { className: "amt-ed-field", children: [_jsx("span", { children: t("amt.editorVelocity") || "力度" }), _jsx("input", { type: "number", min: 1, max: 127, value: velSpin, disabled: selNotes.size === 0, onChange: (e) => onVelocitySpin(Math.max(1, Math.min(127, parseInt(e.target.value, 10) || 100))) })] }), _jsxs("span", { className: "amt-ed-summary", children: [t("amt.editorSummary", { count: stems.reduce((a, s) => a + s.notes.length, 0), changes: undoRef.current.length }), editRendering ? ` · ${t("amt.editorAudioRendering") || "正在更新试听音频…"}` : ""] })] }))] }), _jsx("div", { className: "amt-piano-roll-viewport", ref: viewportRef, tabIndex: editMode ? 0 : -1, onKeyDown: rollKeyDown, onWheel: (e) => {
                                    // Ctrl + wheel = zoom (both axes). Plain wheel stays native scroll.
                                    if (!e.ctrlKey)
                                        return;
                                    e.preventDefault();
                                    const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
                                    setZoom((z) => clampZoom(z * factor));
                                }, children: _jsxs("div", { className: "amt-piano-roll-canvas-wrap", style: { width: PIANO_WIDTH + Math.max(1200, Math.ceil((duration || 1) * PIXELS_PER_SECOND)) }, children: [_jsx("div", { className: "amt-piano-keys", style: { width: PIANO_WIDTH }, children: pianoKeys }), _jsx("canvas", { ref: canvasRef, className: `amt-piano-roll-canvas ${editMode ? "editing" : ""}`, onMouseDown: rollMouseDown, onMouseMove: rollMouseMove, onMouseUp: rollMouseUp, onMouseLeave: rollMouseUp, onDoubleClick: rollDoubleClick, onContextMenu: rollContextMenu, onClick: (e) => {
                                                if (editMode)
                                                    return; // editing clicks are handled by rollMouseDown
                                                const rect = e.currentTarget.getBoundingClientRect();
                                                const x = e.clientX - rect.left;
                                                handleSeek(x / PIXELS_PER_SECOND);
                                            } }), lyricEdit && (_jsx("input", { className: "amt-lyric-input", style: { left: PIANO_WIDTH + Math.max(8, lyricEdit.x - 24), top: Math.max(0, lyricEdit.y - 6) }, defaultValue: lyricEdit.value, autoFocus: true, onKeyDown: (e) => {
                                                if (e.key === "Enter")
                                                    commitRollLyric(e.target.value);
                                                else if (e.key === "Escape")
                                                    setLyricEdit(null);
                                            }, onBlur: (e) => commitRollLyric(e.target.value) }, lyricEdit.noteId))] }) }), _jsxs("div", { className: "amt-zoom-toolbar", children: [_jsx("button", { className: "amt-zoom-btn", title: t("amt.zoomOut") || "缩小", onClick: () => setZoom((z) => clampZoom(z / 1.25)), children: "\u2212" }), _jsxs("span", { className: "amt-zoom-value", children: [Math.round(zoom * 100), "%"] }), _jsx("button", { className: "amt-zoom-btn", title: t("amt.zoomIn") || "放大", onClick: () => setZoom((z) => clampZoom(z * 1.25)), children: "+" }), _jsx("button", { className: "amt-zoom-btn", title: t("amt.zoomReset") || "重置缩放", onClick: () => setZoom(1), children: "\u21BA" })] }), preparingAudio && (_jsxs("div", { className: "amt-prep-badge", children: [_jsx("span", { className: "amt-spinner-sm" }), t("amt.preparingPlayback") || "正在合成播放音频..."] }))] }))] }), lyricsOpen && !isSeparation && (_jsx(AmtLyricsPanel, { result: result, onClose: () => setLyricsOpen(false), onSavedToMidi: syncLyricsFromMidi })), rollMenu && (_jsx(ContextMenu, { x: rollMenu.x, y: rollMenu.y, items: rollMenuItems(), onClose: () => setRollMenu(null) }))] }));
}
