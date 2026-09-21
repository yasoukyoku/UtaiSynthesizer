import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useState, useEffect, useCallback, useRef } from "react";
import { invoke, convertFileSrc } from "@tauri-apps/api/core";
import { open as openDialog } from "@tauri-apps/plugin-dialog";
import { listen } from "@tauri-apps/api/event";
import { useTranslation } from "react-i18next";
import { useAmtModelStore } from "../../store/amt-models";
import { useMsstModelStore } from "../../store/msst-models";
import { useAudioStore } from "../../store/audio";
import { AMT_CATALOG } from "../../lib/models/amt-catalog";
import { useAppStore } from "../../store/app";
import { useProjectStore } from "../../store/project";
import { useHistoryStore } from "../../store/history";
import { blankTrack } from "../../lib/trackFactory";
import { t18, ghRouteOrder, hfBaseForMirror } from "../../lib/models/msst-catalog";
import { matchInstrumentWav } from "../../lib/amtSource";
import { MUSCRIPTOR_INSTRUMENTS, MUSCRIPTOR_ZH_LABELS } from "../../lib/models/muscriptor-instruments";
import { fmtDur } from "../../lib/constants";
import "./AmtConversionDialog.css";
export function AmtConversionDialog({ trackId, onClose, onImported }) {
    const { t, i18n } = useTranslation();
    const lang = i18n.language;
    const showToast = useAppStore((s) => s.showToast);
    const track = useProjectStore((s) => s.tracks.find(t => t.id === trackId));
    const amtConversionSegmentId = useAppStore((s) => s.amtConversionSegmentId);
    const forcedAudioPath = useAppStore((s) => s.amtSourceAudioPath);
    const forcedTrackName = useAppStore((s) => s.amtSourceTrackName);
    const { installed: amtInstalled, downloading: amtDownloading, downloadEntry: downloadAmtEntry } = useAmtModelStore();
    const addTrack = useProjectStore((s) => s.addTrack);
    const openAmtResult = useAppStore((s) => s.openAmtResult);
    const toggleModelManager = useAppStore((s) => s.toggleModelManager);
    const [selectedModelId, setSelectedModelId] = useState(AMT_CATALOG[0]?.id || "");
    const [selectedMode, setSelectedMode] = useState("smart");
    const [selectedBackend, setSelectedBackend] = useState("yourmt3");
    const [selectedMuScriptorInstruments, setSelectedMuScriptorInstruments] = useState([]);
    // Source contract (data_models.MuscriptorProcessingChain): ONLY "official" / "telknet".
    // "sustain_connect" was a merged-in value the Python side rejects outright.
    const [selectedMuScriptorChain, setSelectedMuScriptorChain] = useState("official");
    const [midiTrackMode, setMidiTrackMode] = useState("multi_track");
    const [vocalSplitMergeMidi, setVocalSplitMergeMidi] = useState(false);
    const [removeDuplicates, setRemoveDuplicates] = useState(true);
    const [velocitySmoothing, setVelocitySmoothing] = useState(true);
    const [maxPolyphony, setMaxPolyphony] = useState(40);
    const [ticksPerBeat, setTicksPerBeat] = useState(480);
    const [defaultVelocity, setDefaultVelocity] = useState(80);
    const [hardwareInfo, setany] = useState(null);
    const [isConverting, setIsConverting] = useState(false);
    const [progress, setProgress] = useState(null);
    const [result, setResult] = useState(null);
    /** Playback synthesis assets from the last successful conversion: per-instrument
     *  WAVs keyed "gm:NNN"/"drums" + the source audio + merged MIDI path. The
     *  import-to-tracks flow uses these so every imported MIDI track carries a
     *  playable audio lane (imported notes alone are silent in the DAW). */
    const [playbackAssets, setPlaybackAssets] = useState(null);
    const [sidecarInstalled, setSidecarInstalled] = useState(null);
    const [pyenvProgress, setPyenvProgress] = useState(null);
    const [isInstallingPyenv, setIsInstallingPyenv] = useState(false);
    // Manual fallback when the source audio path cannot be auto-resolved from the track/segment.
    const [manualSourcePath, setManualSourcePath] = useState(null);
    // Advanced options
    const [useGpu, setUseGpu] = useState(true);
    // 默认不量化（保留原曲节奏）；默认速度方案 = 跟随原曲。
    const [quantizeGrid, setQuantizeGrid] = useState("none");
    const [tempoMode, setTempoMode] = useState("adaptive");
    const [customBpm, setCustomBpm] = useState(120);
    const [showAdvanced, setShowAdvanced] = useState(false);
    // Trimming options
    const [startTimeMs, setStartTimeMs] = useState(0);
    const [durationMs, setDurationMs] = useState(0);
    const [isTrimming, setIsTrimming] = useState(false);
    const audioRef = useRef(null);
    const canvasRef = useRef(null);
    // Set when the user clicks "取消" mid-conversion. run_amt_midi then rejects because the
    // sidecar is killed; we settle as an info toast instead of a red error.
    const cancelledRef = useRef(false);
    const [audioDuration, setAudioDuration] = useState(0);
    const [currentTime, setCurrentTime] = useState(0);
    const [isPlayingAudio, setIsPlayingAudio] = useState(false);
    const [peaksData, setPeaksData] = useState([]);
    const [isHoveringWave, setIsHoveringWave] = useState(false);
    const [hoverFrac, setHoverFrac] = useState(0);
    // In-scope playable URL (content-addressed cache WAV) for the <audio> element. The raw
    // source path can be OUTSIDE the asset-protocol scope — see the load effect above.
    const [playbackSrc, setPlaybackSrc] = useState(null);
    useEffect(() => {
        invoke("amt_sidecar_installed").then(setSidecarInstalled).catch(() => setSidecarInstalled(false));
        invoke("get_hardware_info").then(setany).catch(() => { });
        useAmtModelStore.getState().fetchInstalled();
    }, []);
    useEffect(() => {
        const unlisten = listen("pyenv-progress", (event) => {
            setPyenvProgress(event.payload);
            if (event.payload.phase === "done") {
                setSidecarInstalled(true);
                setIsInstallingPyenv(false);
                showToast(t("amt.envInstalled") || "AMT 运行环境安装成功", "success");
                setTimeout(() => setPyenvProgress(null), 3000);
            }
            else if (event.payload.phase === "error") {
                setIsInstallingPyenv(false);
                showToast(event.payload.message, "error");
            }
        });
        return () => { void unlisten.then(fn => fn()); };
    }, [t, showToast]);
    const handleDownloadRuntime = useCallback(async () => {
        if (isInstallingPyenv)
            return;
        setIsInstallingPyenv(true);
        setPyenvProgress({ id: "init", phase: "init", progress: 0, message: "正在初始化...", params: [] });
        try {
            const { mirror, ghMirror, ghPresets } = useMsstModelStore.getState();
            const hf_base = hfBaseForMirror(mirror) || undefined;
            const gh_routes = ghRouteOrder(ghMirror, ghPresets);
            // S68: detect GPU for runtime selection. If NVIDIA r580+ is found, offer/default to CUDA runtime.
            // For simplicity in this dialog's quick-fix button, we use the unified downloader which
            // the backend handles.
            await invoke("download_runtime_pack", {
                id: "runtime-cpu-v1", // backend may upgrade this to gpu if appropriate
                hf_base,
                gh_routes
            });
        }
        catch (e) {
            showToast(String(e), "error");
            setIsInstallingPyenv(false);
        }
    }, [isInstallingPyenv, showToast]);
    const requiredComponents = AMT_CATALOG.filter(m => ["fluidsynth", "soundfont"].includes(m.architecture));
    // Find components that are not marked as available in amtInstalled
    const missingComponents = requiredComponents.filter(m => {
        const installedItem = amtInstalled.find(i => i.id === m.id || i.architecture === m.architecture);
        return !installedItem || !installedItem.is_available;
    });
    // The real `amt-progress` payload is {node_id, progress(0..1), total, message}, NOT
    // {stage, percent, message}. Map it correctly so the bar shows a real percentage.
    useEffect(() => {
        const unlisten = listen("amt-progress", (event) => {
            const p = event.payload;
            setProgress({
                stage: p.stage ?? "",
                percent: Math.min(100, Math.round(((p.total > 0 ? p.progress / p.total : p.progress) || 0) * 100)),
                message: p.message ?? "",
            });
        });
        return () => { void unlisten.then(fn => fn()); };
    }, []);
    // Find the source audio path. The path/name captured at right-click time is the GUARANTEED
    // source — it wins over any segment scan (which could pick up a different processed output).
    let resolvedAudioPath = undefined;
    let resolvedSegmentName = "No Source";
    let resolvedSegment = null;
    // Highest priority: the path/name captured at right-click time (guaranteed source).
    if (forcedAudioPath) {
        resolvedAudioPath = forcedAudioPath;
        resolvedSegmentName = forcedTrackName || t("amt.unknownTrack", "未命名轨道");
    }
    // Only fall back to scanning the track's segments when NO explicit source was captured.
    if (!resolvedAudioPath && track) {
        const candidates = [];
        // Prioritize explicitly requested segment
        if (amtConversionSegmentId) {
            const s = track.segments.find(sg => sg.id === amtConversionSegmentId);
            if (s)
                candidates.push(s);
        }
        // Add selected segment if different from explicit
        const selSegId = useAppStore.getState().selectedSegment?.segmentId;
        if (selSegId && selSegId !== amtConversionSegmentId) {
            const s = track.segments.find(sg => sg.id === selSegId);
            if (s)
                candidates.push(s);
        }
        // Add all other segments
        if (track.segments) {
            candidates.push(...track.segments.filter(s => s.id !== amtConversionSegmentId && s.id !== selSegId));
        }
        for (const seg of candidates) {
            if (!seg)
                continue;
            let p = undefined;
            // Helper to get path from various properties
            const getPath = (obj) => {
                if (!obj)
                    return undefined;
                return obj.audioPath || obj.sourcePath || obj.filePath;
            };
            // 1. Check segment content (audioClip, notes with processedOutputs, vocal type)
            if (seg.content) {
                if (seg.content.type === "audioClip" && seg.content.sourcePath) {
                    p = seg.content.sourcePath;
                }
                else if (seg.content.type === "notes" && seg.processedOutputs && seg.processedOutputs.length > 0) {
                    for (const out of seg.processedOutputs) {
                        p = getPath(out);
                        if (p)
                            break;
                    }
                }
                else if (seg.content.type === "vocal") {
                    p = getPath(seg.content);
                }
                // Fallback: dig into content just in case
                if (!p) {
                    p = getPath(seg.content);
                }
            }
            // Absolute fallback: check root segment properties
            if (!p) {
                p = getPath(seg);
            }
            if (p) {
                resolvedAudioPath = p;
                resolvedSegment = seg;
                // 优先显示片段自带名字；否则用轨道名；再退回通用文案 —— 保证来源轨道/乐器名始终可见。
                resolvedSegmentName =
                    seg.content.label ||
                        seg.content.name ||
                        track.name ||
                        t("amt.unknownTrack", "未命名轨道");
                break; // Found audio path, stop searching
            }
        }
    }
    const audioPath = manualSourcePath || resolvedAudioPath || undefined;
    const sourceSegment = resolvedSegment;
    const segmentName = resolvedSegmentName;
    const displayTrackName = track?.name || forcedTrackName || t("amt.unknownTrack", "未知轨道");
    console.log("AmtConversionDialog debug info:");
    console.log(" - trackId:", trackId);
    console.log(" - track found:", !!track);
    console.log(" - track segments:", track?.segments?.length);
    console.log(" - amtConversionSegmentId:", amtConversionSegmentId);
    console.log(" - selectedSegment:", useAppStore.getState().selectedSegment);
    console.log(" - sourceSegment resolved:", sourceSegment);
    console.log(" - audioPath resolved:", audioPath);
    // Show a toast if audioPath is not found, providing user feedback
    useEffect(() => {
        if (!audioPath) {
            showToast(t("amt.noSourceAudio", "无法找到音频源文件，请确保轨道包含有效音频。"), "error");
        }
    }, [audioPath, showToast, t]);
    // Load audio duration and peaks when path changes
    useEffect(() => {
        if (audioPath) {
            // Reset the cache-copy URL so a previous source's preview can't bleed into this one.
            setPlaybackSrc(null);
            // §user "默认整首歌": a NEW source must not inherit the previous song's trim
            // window — reset the range so it re-initializes to the new song's full length.
            setStartTimeMs(0);
            setDurationMs(0);
            setPeaksData([]);
            // 1. Fetch peaks and duration via useAudioStore.loadAudioFile (high-performance backend cache)
            useAudioStore.getState().loadAudioFile(audioPath).then((data) => {
                if (data.peaks && data.peaks.length > 0) {
                    setPeaksData(data.peaks);
                }
                if (data.durationMs > 0) {
                    const durSec = data.durationMs / 1000;
                    setAudioDuration(durSec);
                    if (durationMs === 0)
                        setDurationMs(Math.floor(data.durationMs));
                }
                // PLAYBACK SOURCE: the ORIGINAL file may live anywhere on disk (Downloads/Music/…),
                // OUTSIDE the asset-protocol scope — a convertFileSrc() URL on it is silently blocked
                // and the <audio> element never plays. The backend's content-addressed WAV copy
                // (data root, runtime-added to the scope) is always playable: prefer it.
                if (data.playbackPath) {
                    setPlaybackSrc(convertFileSrc(data.playbackPath.replace(/\\/g, "/")));
                }
            }).catch((e) => {
                console.warn("loadAudioFile failed, fallback to native audio metadata:", e);
            });
            // 2. Metadata probe (best effort — the real <audio> uses playbackSrc above when ready)
            const safeUrl = convertFileSrc(audioPath.replace(/\\/g, "/"));
            const audio = new Audio(safeUrl);
            audio.onloadedmetadata = () => {
                setAudioDuration(audio.duration);
                if (durationMs === 0)
                    setDurationMs(Math.floor(audio.duration * 1000));
            };
            audio.onerror = (e) => {
                console.error("Failed to load audio for preview:", safeUrl, e);
            };
        }
    }, [audioPath]);
    // Audio play/pause and timeupdate synchronization
    const togglePlayAudio = () => {
        const el = audioRef.current;
        if (!el)
            return;
        if (el.paused) {
            el.play().catch(e => console.warn("Audio play error:", e));
            setIsPlayingAudio(true);
        }
        else {
            el.pause();
            setIsPlayingAudio(false);
        }
    };
    const handleAudioTimeUpdate = () => {
        const el = audioRef.current;
        if (el) {
            setCurrentTime(el.currentTime);
        }
    };
    const handleAudioEnded = () => {
        setIsPlayingAudio(false);
    };
    const seekAudio = (frac) => {
        const el = audioRef.current;
        if (!el || !audioDuration)
            return;
        const target = Math.max(0, Math.min(1, frac)) * audioDuration;
        el.currentTime = target;
        setCurrentTime(target);
    };
    // Draw Waveform on Canvas
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
        ctx.clearRect(0, 0, w, h);
        // Background gradient
        const bgGrad = ctx.createLinearGradient(0, 0, 0, h);
        bgGrad.addColorStop(0, "#14171d");
        bgGrad.addColorStop(1, "#0d0f12");
        ctx.fillStyle = bgGrad;
        ctx.fillRect(0, 0, w, h);
        // Center baseline
        const midY = h / 2;
        ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
        ctx.lineWidth = 1 * dpr;
        ctx.beginPath();
        ctx.moveTo(0, midY);
        ctx.lineTo(w, midY);
        ctx.stroke();
        const totalDur = audioDuration || 1;
        const trimStartSec = isTrimming ? startTimeMs / 1000 : 0;
        const trimDurSec = isTrimming && durationMs > 0 ? durationMs / 1000 : totalDur;
        const trimEndSec = Math.min(totalDur, trimStartSec + trimDurSec);
        // Waveform rendering
        if (peaksData && peaksData.length > 0) {
            const numCols = Math.min(peaksData.length, Math.floor(w / (2 * dpr)));
            const colWidth = w / Math.max(1, numCols);
            const per = peaksData.length / numCols;
            const amp = (h / 2) - 4 * dpr;
            for (let c = 0; c < numCols; c++) {
                const start = Math.floor(c * per);
                const end = Math.min(peaksData.length, Math.max(start + 1, Math.floor((c + 1) * per)));
                let p = 0;
                for (let i = start; i < end; i++) {
                    const val = Math.abs(peaksData[i] ?? 0);
                    if (val > p)
                        p = val;
                }
                const barX = c * colWidth;
                const colCenterSec = (c / numCols) * totalDur;
                const isInTrim = !isTrimming || (colCenterSec >= trimStartSec && colCenterSec <= trimEndSec);
                const isPastPlayhead = colCenterSec <= currentTime;
                // Determine column bar color
                if (isPastPlayhead) {
                    ctx.fillStyle = isInTrim ? "#00e5ff" : "rgba(0, 229, 255, 0.35)";
                }
                else {
                    ctx.fillStyle = isInTrim ? "rgba(96, 165, 250, 0.75)" : "rgba(255, 255, 255, 0.2)";
                }
                const barH = Math.max(2 * dpr, p * amp);
                const barW = Math.max(1 * dpr, colWidth - 1 * dpr);
                ctx.fillRect(barX, midY - barH, barW, barH * 2);
            }
        }
        else {
            // Empty waveform placeholder line
            ctx.strokeStyle = "rgba(0, 229, 255, 0.4)";
            ctx.setLineDash([4 * dpr, 4 * dpr]);
            ctx.beginPath();
            ctx.moveTo(0, midY);
            ctx.lineTo(w, midY);
            ctx.stroke();
            ctx.setLineDash([]);
        }
        // Trimming mask & boundaries
        if (isTrimming && totalDur > 0) {
            const startX = (trimStartSec / totalDur) * w;
            const endX = (trimEndSec / totalDur) * w;
            // Darken outside regions
            ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
            if (startX > 0) {
                ctx.fillRect(0, 0, startX, h);
            }
            if (endX < w) {
                ctx.fillRect(endX, 0, w - endX, h);
            }
            // Trim boundary lines
            ctx.strokeStyle = "#38bdf8";
            ctx.lineWidth = 2 * dpr;
            ctx.beginPath();
            ctx.moveTo(startX, 0);
            ctx.lineTo(startX, h);
            ctx.moveTo(endX, 0);
            ctx.lineTo(endX, h);
            ctx.stroke();
            // Highlight trim region frame
            ctx.strokeStyle = "rgba(56, 189, 248, 0.4)";
            ctx.lineWidth = 1 * dpr;
            ctx.strokeRect(startX, 1 * dpr, endX - startX, h - 2 * dpr);
        }
        // Playhead cursor
        if (totalDur > 0) {
            const playheadX = (currentTime / totalDur) * w;
            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = 2 * dpr;
            ctx.shadowColor = "rgba(0, 229, 255, 0.8)";
            ctx.shadowBlur = 6 * dpr;
            ctx.beginPath();
            ctx.moveTo(playheadX, 0);
            ctx.lineTo(playheadX, h);
            ctx.stroke();
            ctx.shadowBlur = 0;
            // Playhead triangle handle on top
            ctx.fillStyle = "#ffffff";
            ctx.beginPath();
            ctx.moveTo(playheadX - 4 * dpr, 0);
            ctx.lineTo(playheadX + 4 * dpr, 0);
            ctx.lineTo(playheadX, 6 * dpr);
            ctx.closePath();
            ctx.fill();
        }
        // Hover cursor
        if (isHoveringWave && totalDur > 0) {
            const hoverX = hoverFrac * w;
            ctx.strokeStyle = "rgba(255, 255, 255, 0.5)";
            ctx.lineWidth = 1 * dpr;
            ctx.setLineDash([2 * dpr, 2 * dpr]);
            ctx.beginPath();
            ctx.moveTo(hoverX, 0);
            ctx.lineTo(hoverX, h);
            ctx.stroke();
            ctx.setLineDash([]);
        }
    }, [peaksData, audioDuration, currentTime, isTrimming, startTimeMs, durationMs, isHoveringWave, hoverFrac]);
    // Infer MuScriptor instruments from filename
    useEffect(() => {
        if (audioPath && selectedBackend === "muscriptor" && selectedMuScriptorInstruments.length === 0) {
            const filename = audioPath.split(/[/\\]/).pop() || "";
            const stem = filename.replace(/\.[^.]+$/, "").trim().toLowerCase();
            const normalized = stem.replace(/[\s.\-]+/g, "_").replace(/^_+|_+$/g, "");
            const aliases = {
                guitar: ["acoustic_guitar", "clean_electric_guitar", "distorted_electric_guitar"],
                guitars: ["acoustic_guitar", "clean_electric_guitar", "distorted_electric_guitar"],
                bass: ["acoustic_bass", "electric_bass"],
                basses: ["acoustic_bass", "electric_bass"],
                piano: ["acoustic_piano", "electric_piano"],
                keys: ["acoustic_piano", "electric_piano"],
                keyboards: ["acoustic_piano", "electric_piano"],
                vocal: ["voice"],
                vocals: ["voice"],
                voice: ["voice"],
                drum: ["drums"],
                drums: ["drums"],
            };
            let inferred = [];
            const tokens = normalized.split("_");
            const lastToken = tokens[tokens.length - 1];
            if (aliases[normalized])
                inferred = aliases[normalized];
            else if (lastToken && aliases[lastToken])
                inferred = aliases[lastToken];
            else {
                // Try exact matches
                const exact = MUSCRIPTOR_INSTRUMENTS.find(inst => normalized === inst || normalized.endsWith(`_${inst}`));
                if (exact)
                    inferred = [exact];
            }
            if (inferred.length > 0) {
                setSelectedMuScriptorInstruments(inferred);
            }
        }
    }, [audioPath, selectedBackend]);
    /** §user「只要是 MIDI 文件打开就可以修复」：选本地 .mid/.midi → 验证可解析 →
     *  直接打开修复工作台（不经过音频转换）。outputDir 取文件所在目录，导出成品就近落盘。 */
    const handleOpenLocalMidi = async () => {
        const picked = await openDialog({
            multiple: false,
            filters: [{ name: "MIDI", extensions: ["mid", "midi", "MID", "MIDI"] }],
        });
        const midiPath = typeof picked === "string" ? picked : null;
        if (!midiPath)
            return;
        try {
            const score = await invoke("import_score_file", { path: midiPath });
            if (!score || (score.tracks?.length ?? 0) === 0) {
                throw new Error("empty score");
            }
            openAmtResult({
                trackId: `midi-repair-` + Date.now(),
                outputDir: midiPath.replace(/[\\/][^\\/]+$/, "") || ".",
                midiPaths: [midiPath],
            });
            onClose();
        }
        catch (e) {
            console.warn("Local MIDI open failed:", e);
            showToast(t("amt.localMidiInvalid") || "无法读取该 MIDI 文件（可能已损坏或不是标准 MIDI）", "error");
        }
    };
    const handleStartConversion = async () => {
        if (!track || !audioPath) {
            showToast(t("amt.noSource"), "error");
            return;
        }
        const model = AMT_CATALOG.find(m => m.id === selectedModelId);
        if (!model)
            return;
        const isInstalled = amtInstalled.find(m => m.id === selectedModelId)?.is_available;
        if (!isInstalled) {
            showToast(t("amt.modelNotInstalled"), "error");
            return;
        }
        setIsConverting(true);
        cancelledRef.current = false;
        setProgress({ stage: "init", percent: 0, message: t("amt.starting") });
        setResult(null);
        try {
            // 1. Run MIDI transcription
            // Source contract (Config.validate in data_models.py):
            //   - custom_bpm is ONLY valid with tempo_mode="fixed_manual" — otherwise the sidecar
            //     raises "custom_bpm is only valid for fixed_manual tempo_mode".
            //   - yourmt3_model must be one of the 6 official YourMT3 checkpoints (catalog id for
            //     yourmt3_plus happens to be one). MIROS/MuScriptor models pass a different id that
            //     Python rejects — send null so Rust applies the safe default.
            //   - muscriptor_model must be small|medium|large — strip the catalog's "muscriptor_"
            //     prefix. The yourmt3 fix above means Rust's VALID lists see only contract values.
            const effCustomBpm = tempoMode === "fixed_manual" ? customBpm : null;
            const effYourmt3Model = selectedBackend === "yourmt3" ? selectedModelId : null;
            const effMuscriptorModel = selectedBackend === "muscriptor"
                ? (selectedModelId.replace(/^muscriptor_/, "") || "large")
                : null;
            const res = await invoke("run_amt_midi", {
                audioPath: audioPath,
                midiMode: selectedMode,
                transcriptionBackend: selectedBackend,
                yourmt3Model: effYourmt3Model,
                muscriptorModel: effMuscriptorModel,
                midiTrackMode: midiTrackMode,
                tempoMode: tempoMode,
                customBpm: effCustomBpm,
                quantizeNotes: quantizeGrid !== "none",
                quantizeGrid: quantizeGrid,
                useGpu: useGpu,
                gpuDevice: 0,
                outputDir: audioPath.replace(/\.[^/.]+$/, "") + "_midi",
                nodeId: trackId,
                muscriptorInstruments: selectedMuScriptorInstruments,
                muscriptorProcessingChain: selectedMuScriptorChain,
                vocalSplitMergeMidi: vocalSplitMergeMidi,
                removeDuplicates: removeDuplicates,
                velocitySmoothing: velocitySmoothing,
                maxPolyphony: maxPolyphony,
                ticksPerBeat: ticksPerBeat,
                defaultVelocity: defaultVelocity,
                startTimeMs: isTrimming ? startTimeMs : undefined,
                durationMs: isTrimming ? durationMs : undefined
            });
            const midiPath = res.midi_path;
            // Separation modes (vocal_split / six_stem_split) legitimately return an empty
            // midi_path — the sidecar emits separated WAV stems via `separated_audio` instead.
            // Treat that as success and open the workbench in separation view.
            const separatedAudio = (res.separated_audio ?? {});
            const isSeparationOnly = !midiPath && Object.values(separatedAudio).filter(Boolean).length > 0;
            if (!midiPath && !isSeparationOnly) {
                throw new Error("No MIDI path returned from sidecar");
            }
            if (isSeparationOnly) {
                const sepStems = {};
                for (const [name, path] of Object.entries(separatedAudio)) {
                    if (path)
                        sepStems[name] = { midi: "", audio: path };
                }
                openAmtResult({
                    trackId,
                    outputDir: audioPath.replace(/\.[^/.]+$/, "") + "_midi",
                    midiPaths: [],
                    audioPreview: "",
                    originalAudio: audioPath,
                    sourceAudioPath: audioPath,
                    stems: sepStems,
                    separationOnly: true
                });
                showToast(t("amt.separationSuccess", "分离完成，可试听分轨"), "success");
                onClose();
                return;
            }
            // 2. Prepare playback assets (FluidSynth synthesis). This is BEST-EFFORT: preview/playspeed
            //    requires FluidSynth + SoundFont, which may be missing. Converting succeeded regardless, so
            //    NEVER let a playback-prep failure block the result page from opening — the panel falls back
            //    to showing the MIDI stems (piano roll) with an empty/linear top track.
            let playbackRes = { transcription_wav: "", original_wav: "", instrument_wavs: {} };
            try {
                setProgress({ stage: "synthesize", percent: 90, message: t("amt.preparingPlayback") || "正在合成预览音频..." });
                playbackRes = await invoke("amt_prepare_playback", {
                    midiPath: midiPath,
                    audioPath: audioPath,
                    outputDir: audioPath.replace(/\.[^/.]+$/, "") + "_midi"
                });
            }
            catch (e) {
                console.warn("Playback prep failed (best-effort — opening result anyway):", e);
            }
            // Adapt result to ConversionResult interface. Always include the primary MIDI (the per-track
            // midis may be absent if the backend returned only a merged file — import_score_file then exposes
            // each instrument as a separate track in the panel).
            const midiPaths = res.stem_midi_paths
                ? Object.values(res.stem_midi_paths).filter(Boolean)
                : [midiPath];
            const adaptedResult = {
                midi_paths: midiPaths.length > 0 ? midiPaths : [midiPath],
                output_dir: audioPath.replace(/\.[^/.]+$/, "") + "_midi"
            };
            setResult(adaptedResult);
            setPlaybackAssets({
                audioPath,
                mergedMidiPath: midiPath,
                wavs: (playbackRes?.instrument_wavs ?? {}),
            });
            showToast(t("amt.conversionSuccess"), "success");
            // 3. Automatically open the result workbench at the bottom.
            const instrumentWavs = (playbackRes.instrument_wavs ?? {});
            // §user "干音都要有歌词"：人声分离出的干音 WAV —— 结果工作台自动歌词提取的最准源。
            const dryVocalAudio = separatedAudio["vocals"] || separatedAudio["vocal"] || "";
            openAmtResult({
                trackId,
                outputDir: adaptedResult.output_dir,
                midiPaths: adaptedResult.midi_paths,
                audioPreview: playbackRes.transcription_wav,
                originalAudio: playbackRes.original_wav,
                sourceAudioPath: audioPath,
                vocalAudioPath: dryVocalAudio,
                stems: Object.entries(instrumentWavs).reduce((acc, [k, v]) => {
                    acc[k] = { midi: "", audio: v };
                    return acc;
                }, {})
            });
            onClose();
        }
        catch (e) {
            const errStr = String(e);
            console.error("AMT Conversion failed:", errStr);
            if (cancelledRef.current) {
                // User cancelled — the sidecar was force-killed. Report as a clean cancel.
                await invoke("cancel_amt_midi", { nodeId: trackId }).catch(() => { });
                showToast(t("amt.cancelled", "转换已取消"), "info");
                return;
            }
            // Extract just the first line for the toast if it's a long error
            const shortErr = errStr.split('\n')[0] ?? errStr;
            showToast(shortErr, "error");
        }
        finally {
            setIsConverting(false);
            setProgress(null);
        }
    };
    /** Force-stop the running AMT sidecar for this dialog's conversion. */
    const handleCancelConversion = () => {
        cancelledRef.current = true;
        setProgress((prev) => ({ stage: "cancelling", percent: (prev?.percent ?? 0), message: t("amt.cancelling", "正在取消...") }));
        invoke("cancel_amt_midi", { nodeId: trackId }).catch(() => { });
    };
    const handleImportToTracks = async () => {
        if (!result)
            return;
        try {
            useHistoryStore.getState().beginTransaction();
            const importedIds = [];
            // Guarantee per-instrument WAVs exist: the dialog's playback prep is best-effort,
            // so if it failed (or produced no instrument WAVs) synthesize NOW — imported
            // notes-only tracks would otherwise be silent in the DAW.
            let wavs = playbackAssets?.wavs ?? {};
            const srcAudio = playbackAssets?.audioPath;
            const mergedMidi = playbackAssets?.mergedMidiPath || result.midi_paths[0];
            if (Object.keys(wavs).length === 0 && srcAudio && mergedMidi) {
                try {
                    const pb = await invoke("amt_prepare_playback", {
                        midiPath: mergedMidi,
                        audioPath: srcAudio,
                        outputDir: result.output_dir,
                    });
                    wavs = (pb?.instrument_wavs ?? {});
                }
                catch (e) {
                    console.warn("Import-time playback synthesis failed:", e);
                }
            }
            if (result.output_dir) {
                await invoke("allow_asset_dir", { dir: result.output_dir }).catch(() => { });
            }
            for (const path of result.midi_paths) {
                const score = await invoke("import_score_file", { path });
                // Per-file GM metadata (program+channel per track) maps track names onto
                // the sidecar's "gm:NNN"/"drums" WAV keys — the names alone never match.
                let metaTracks = { track_count: 0, total_notes: 0, ppq: 480, tracks: [], size_bytes: 0 };
                try {
                    const meta = await invoke("amt_midi_metadata", { midiPath: path });
                    metaTracks = meta?.tracks ?? [];
                }
                catch { /* best-effort — name matching still runs below */ }
                for (const it of score.tracks) {
                    const newTrackId = crypto.randomUUID();
                    importedIds.push(newTrackId);
                    const segmentId = crypto.randomUUID();
                    const maxTick = Math.max(1, it.notes.reduce((max, n) => Math.max(max, n.tick + n.duration), 0));
                    const newTrack = blankTrack(newTrackId, it.name || path.split(/[/\\]/).pop() || "MIDI Track", "instrument");
                    // Attach the per-instrument synthesized WAV as a playable audio lane so the
                    // DAW transport produces sound for this track (notes alone are silent).
                    const mt = metaTracks.tracks.find((m) => m.name === it.name) || metaTracks[0];
                    const wav = matchInstrumentWav(wavs, it.name || "", mt?.program ?? null, mt?.channel ?? null);
                    const processedOutputs = [];
                    if (wav) {
                        const safePpq = 480;
                        const safeBpm = 120;
                        let totalDurationMs = Math.max(1, (maxTick / (safePpq * (safeBpm / 60))) * 1000);
                        let waveformPeaks;
                        try {
                            const data = await useAudioStore.getState().loadAudioFile(wav);
                            if (data.durationMs > 0)
                                totalDurationMs = data.durationMs;
                            if (data.peaks && data.peaks.length > 0)
                                waveformPeaks = data.peaks;
                        }
                        catch { /* best-effort waveform */ }
                        processedOutputs.push({
                            laneId: segmentId,
                            laneLabel: it.name || "MIDI",
                            group: it.name || "MIDI",
                            audioPath: wav,
                            totalDurationMs,
                            waveformPeaks,
                        });
                    }
                    newTrack.segments = [{
                            id: segmentId,
                            startTick: it.start_tick,
                            durationTicks: maxTick,
                            content: {
                                type: "notes",
                                notes: it.notes.map((n) => ({
                                    ...n,
                                    id: crypto.randomUUID(),
                                    lyric: n.lyric || "La",
                                    velocity: n.velocity ?? 100
                                })),
                                pitchDev: it.pitch_dev
                            },
                            processedOutputs: processedOutputs.length > 0 ? processedOutputs : undefined,
                        }];
                    addTrack(newTrack);
                }
            }
            useHistoryStore.getState().commitTransaction();
            const silent = wavs && Object.keys(wavs).length === 0;
            showToast(silent
                ? (t("amt.importSuccessNoAudio") || "已导入轨道（未找到试听音频）")
                : t("amt.importSuccess"), silent ? "info" : "success");
            if (importedIds.length > 0)
                onImported?.(importedIds);
            handleClose();
        }
        catch (e) {
            useHistoryStore.getState().commitTransaction();
            showToast(String(e), "error");
        }
    };
    const handleExportToFolder = async () => {
        if (!result)
            return;
        try {
            const { save } = await import("@tauri-apps/plugin-dialog");
            const savePath = await save({
                filters: [{ name: "ZIP Archive", extensions: ["zip"] }],
                defaultPath: `${track?.name || "converted"}_midi.zip`
            });
            if (!savePath || typeof savePath !== "string")
                return;
            await invoke("amt_export_zip", {
                midiPaths: result.midi_paths,
                savePath
            });
            showToast(t("amt.exportSuccess"), "success");
        }
        catch (e) {
            showToast(String(e), "error");
        }
    };
    // The stars feature is not currently used, but keeping it commented out for future use
    // const renderStars = (rating: number) => {
    //   return (
    //     <div className="amt-stars-container">
    //       {[...Array(5)].map((_, i) => (
    //         <span key={i} className={`amt-star ${i < rating ? "filled" : ""}`}>★</span>
    //       ))}
    //     </div>
    //   );
    // };
    const pianoModels = AMT_CATALOG.filter(m => ["transkun", "aria_amt", "bytedance_piano"].includes(m.architecture));
    const isMultiInstrumentMode = ["smart", "vocal_split", "six_stem_split"].includes(selectedMode);
    const handleClose = () => {
        useAppStore.setState({ amtConversionSegmentId: undefined });
        onClose();
    };
    return (_jsx("div", { className: "amt-dialog-overlay", onClick: handleClose, children: _jsxs("div", { className: "amt-dialog", onClick: e => e.stopPropagation(), children: [_jsxs("div", { className: "amt-dialog-header", children: [_jsxs("div", { className: "amt-title-with-icon", children: [_jsx("span", { className: "amt-header-icon", children: "\uD83C\uDFB9" }), _jsx("h2", { children: t("amt.dialogTitle") })] }), _jsx("button", { className: "amt-close-btn", onClick: handleClose, children: "\u00D7" })] }), _jsxs("div", { className: "amt-dialog-toolbar", children: [isConverting && progress && (_jsxs("div", { className: "amt-floating-progress", children: [_jsx("div", { className: "progress-bar-bg", children: _jsx("div", { className: "progress-bar-fill", style: { width: `${progress.percent}%` } }) }), _jsxs("div", { className: "progress-info", children: [_jsx("span", { className: "msg", children: progress.message }), _jsxs("span", { className: "pct", children: [progress.percent, "%"] })] })] })), !result && (_jsx("div", { className: "amt-actions-v2", children: isConverting ? (_jsxs(_Fragment, { children: [_jsxs("button", { className: "amt-main-convert-btn", disabled: true, style: { opacity: 0.55, cursor: "default" }, children: [_jsx("span", { className: "spinner" }), t("amt.converting", "正在转换...")] }), _jsxs("button", { className: "amt-cancel-convert-btn", onClick: handleCancelConversion, title: t("amt.cancelConversion", "终止转换"), children: ["\u2715 ", t("amt.cancelConversion", "终止")] })] })) : (_jsxs(_Fragment, { children: [_jsxs("button", { className: "amt-main-convert-btn", onClick: handleStartConversion, children: [_jsx("span", { className: "icon", children: "\uD83D\uDE80" }), t("amt.startConversion", "开始转换")] }), _jsxs("button", { className: "amt-local-midi-btn", onClick: handleOpenLocalMidi, title: t("amt.repairLocalMidiTip") || "打开任意 .mid/.midi 文件，直接进入修复工作台：一键修复多余/缺失音符、换音源、导出", children: ["\uD83E\uDE79 ", t("amt.repairLocalMidi") || "修复本地 MIDI 文件…"] })] })) }))] }), _jsx("div", { className: "amt-dialog-content", children: !result ? (_jsxs(_Fragment, { children: [_jsxs("div", { className: "amt-source-card", children: [_jsxs("div", { className: "amt-source-header", children: [_jsxs("div", { className: "amt-source-info", children: [_jsx("div", { className: "amt-source-label", children: t("amt.sourceTrack", "源轨道 / 片段") }), _jsx("div", { className: "amt-source-track", title: displayTrackName, children: `🎵 ${displayTrackName}` }), _jsx("div", { className: "amt-source-name", title: segmentName, children: segmentName }), _jsx("div", { className: "amt-source-path", title: audioPath, children: audioPath || t("amt.noSource", "轨道没有音频源，无法转换") })] }), _jsxs("div", { className: "amt-source-actions", children: [_jsxs("button", { className: `amt-trim-toggle ${isTrimming ? "active" : ""}`, onClick: () => setIsTrimming(!isTrimming), children: ["\u2702\uFE0F ", t("amt.trimSegment", "剪切片段")] }), !audioPath && (_jsxs("button", { className: "amt-trim-toggle", onClick: async () => {
                                                            const { open } = await import("@tauri-apps/plugin-dialog");
                                                            const picked = await open({
                                                                multiple: false,
                                                                filters: [{ name: "Audio", extensions: ["wav", "mp3", "m4a", "flac", "ogg", "aac", "wma", "opus", "mid", "midi"] }],
                                                            });
                                                            if (picked && !Array.isArray(picked))
                                                                setManualSourcePath(picked);
                                                        }, children: ["\uD83D\uDCC2 ", t("amt.chooseSourceFile", "选择音频文件")] }))] })] }), audioPath && (_jsxs("div", { className: "amt-waveform-player-card", children: [_jsxs("div", { className: "amt-waveform-canvas-wrap", onMouseEnter: () => setIsHoveringWave(true), onMouseLeave: () => setIsHoveringWave(false), onMouseMove: (e) => {
                                                    const rect = e.currentTarget.getBoundingClientRect();
                                                    const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / Math.max(1, rect.width)));
                                                    setHoverFrac(frac);
                                                }, onClick: (e) => {
                                                    const rect = e.currentTarget.getBoundingClientRect();
                                                    const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / Math.max(1, rect.width)));
                                                    seekAudio(frac);
                                                }, children: [_jsx("canvas", { ref: canvasRef, className: "amt-waveform-canvas" }), isHoveringWave && (_jsx("div", { className: "amt-waveform-hover-time", style: { left: `${hoverFrac * 100}%` }, children: fmtDur(hoverFrac * (audioDuration || 0)) }))] }), _jsxs("div", { className: "amt-player-controls-bar", children: [_jsxs("div", { className: "amt-player-left", children: [_jsx("button", { className: `amt-play-btn ${isPlayingAudio ? "playing" : ""}`, onClick: togglePlayAudio, title: isPlayingAudio ? t("common.pause", "暂停") : t("common.play", "播放"), children: isPlayingAudio ? "⏸" : "▶" }), _jsxs("div", { className: "amt-player-time-display", children: [_jsx("span", { className: "current", children: fmtDur(currentTime) }), _jsx("span", { className: "divider", children: "/" }), _jsx("span", { className: "total", children: fmtDur(audioDuration) })] })] }), _jsx("div", { className: "amt-player-center", children: _jsxs("div", { className: "amt-scrubber-track", onClick: (e) => {
                                                                const rect = e.currentTarget.getBoundingClientRect();
                                                                const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / Math.max(1, rect.width)));
                                                                seekAudio(frac);
                                                            }, children: [_jsx("div", { className: "amt-scrubber-fill", style: { width: `${audioDuration > 0 ? (currentTime / audioDuration) * 100 : 0}%` } }), _jsx("div", { className: "amt-scrubber-handle", style: { left: `${audioDuration > 0 ? (currentTime / audioDuration) * 100 : 0}%` } })] }) }), _jsx("div", { className: "amt-player-right", children: isTrimming ? (_jsxs("span", { className: "amt-trim-summary-badge", children: [t("amt.rangeLabel", "转换范围"), ": ", fmtDur(startTimeMs / 1000), " ~ ", fmtDur(Math.min(audioDuration, (startTimeMs + durationMs) / 1000))] })) : (audioDuration > 0 && (_jsxs("span", { className: "amt-trim-summary-badge", children: [t("amt.fullSongRange", "整首歌"), " \u00B7 ", fmtDur(0), " ~ ", fmtDur(audioDuration)] }))) })] }), _jsx("audio", { ref: audioRef, src: playbackSrc || convertFileSrc(audioPath.replace(/\\/g, "/")), onTimeUpdate: handleAudioTimeUpdate, onEnded: handleAudioEnded, style: { display: "none" } })] })), isTrimming && (_jsxs("div", { className: "amt-trim-controls", children: [_jsxs("div", { className: "amt-trim-field", children: [_jsx("label", { children: t("amt.startTime", "开始时间 (秒)") }), _jsx("input", { type: "number", step: "0.1", min: "0", max: audioDuration, value: startTimeMs / 1000, onChange: (e) => setStartTimeMs(parseFloat(e.target.value) * 1000) })] }), _jsxs("div", { className: "amt-trim-field", children: [_jsx("label", { children: t("amt.duration", "持续时长 (秒)") }), _jsx("input", { type: "number", step: "0.1", min: "0.1", max: audioDuration - (startTimeMs / 1000), value: durationMs / 1000, onChange: (e) => setDurationMs(parseFloat(e.target.value) * 1000) })] })] }))] }), !sidecarInstalled && sidecarInstalled !== null && (_jsxs("div", { className: "amt-sidecar-warning", children: [_jsx("div", { className: "amt-warning-icon", children: "\u26A0\uFE0F" }), _jsxs("div", { className: "amt-warning-content", children: [_jsx("p", { children: t("amt.sidecarNotInstalled", "AMT 运行环境未安装，请先下载安装") }), pyenvProgress ? (_jsxs("div", { className: "amt-pyenv-progress-container", children: [_jsxs("div", { className: "download-progress-container", children: [_jsx("div", { className: "download-progress-bar", style: { width: `${pyenvProgress.progress * 100}%` } }), _jsx("span", { className: "download-progress-text", children: pyenvProgress.phase === "download" ? `${(pyenvProgress.progress * 100).toFixed(1)}%` : pyenvProgress.message })] }), _jsx("p", { className: "amt-pyenv-msg", children: pyenvProgress.message })] })) : (_jsxs("div", { className: "amt-warning-actions", children: [_jsx("button", { className: "amt-dl-btn", onClick: handleDownloadRuntime, disabled: isInstallingPyenv, children: isInstallingPyenv ? "正在启动..." : (t("amt.downloadEnv", "下载 AMT 运行环境 (约 1.2GB)")) }), _jsx("button", { className: "amt-link-btn", onClick: toggleModelManager, children: t("missingModels.openManager", "打开资源管理") })] }))] })] })), _jsxs("div", { className: "amt-components-card", children: [_jsxs("div", { className: "amt-card-header", children: [_jsx("span", { className: "icon", children: "\uD83C\uDFB5" }), _jsx("span", { children: t("amt.playbackComponents", "播放组件状态") })] }), _jsx("div", { className: "amt-components-list", children: requiredComponents.map(comp => {
                                            const installedItem = amtInstalled.find(i => i.id === comp.id || i.architecture === comp.architecture);
                                            const isInstalled = installedItem?.is_available;
                                            const isDl = !!amtDownloading[comp.id];
                                            return (_jsxs("div", { className: `amt-comp-item ${isInstalled ? "installed" : "missing"}`, children: [_jsxs("div", { className: "amt-comp-info", children: [_jsx("span", { className: "amt-comp-name", children: t18(comp.name, lang) }), _jsx("span", { className: "amt-comp-status", children: isInstalled ? "✅" : "❌" })] }), !isInstalled ? (_jsx("button", { className: `amt-comp-dl-btn-sm ${isDl ? "loading" : ""}`, disabled: isDl, onClick: () => {
                                                            console.log("Triggering download for component:", comp);
                                                            downloadAmtEntry(comp);
                                                        }, children: isDl ? "..." : t("missingModels.download", "下载") })) : (_jsx("span", { className: "amt-comp-installed-text", style: { fontSize: "12px", color: "var(--success-color)", fontWeight: 500 }, children: t("amt.installed", "已安装") }))] }, comp.id));
                                        }) }), missingComponents.length > 0 && (_jsx("p", { className: "amt-comp-tip", children: t("amt.playbackTip", "安装上述组件以启用转换后的高保真预览与导出") }))] }), _jsx("div", { className: "amt-config-sections", children: _jsxs("div", { className: "amt-config-section", children: [_jsxs("div", { className: "amt-section-title", children: [_jsx("span", { className: "icon", children: "\uD83C\uDFAF" }), " ", t("amt.section.model", "模型与模式")] }), _jsxs("div", { className: "amt-config-grid-v3", children: [_jsx("div", { className: "amt-mode-cards", children: [
                                                        { id: "smart", icon: "🎸", descKey: "smart" },
                                                        { id: "vocal_split", icon: "🎤", descKey: "vocal_split" },
                                                        { id: "six_stem_split", icon: "🎚", descKey: "six_stem_split" },
                                                        { id: "piano_transkun", icon: "🎹", descKey: "piano_transkun" },
                                                        { id: "piano_transkun_v2_aug", icon: "🎹", descKey: "piano_transkun_v2_aug" },
                                                        { id: "piano_aria_amt", icon: "🎹", descKey: "piano_aria_amt" },
                                                        { id: "piano_bytedance_pedal", icon: "🎹", descKey: "piano_bytedance_pedal" },
                                                    ].map((m) => (_jsxs("button", { type: "button", className: `amt-mode-card ${selectedMode === m.id ? "active" : ""}`, onClick: () => setSelectedMode(m.id), children: [_jsx("span", { className: "amt-mode-icon", children: m.icon }), _jsx("span", { className: "amt-mode-label", children: t(`amt.mode.${m.descKey}`) })] }, m.id))) }), isMultiInstrumentMode && (_jsxs("div", { className: "amt-field", children: [_jsx("label", { children: t("amt.backend", "转谱后端") }), _jsxs("select", { value: selectedBackend, onChange: (e) => {
                                                                setSelectedBackend(e.target.value);
                                                                if (e.target.value === "yourmt3")
                                                                    setSelectedModelId("yptf_moe_multi_nops");
                                                                else if (e.target.value === "miros")
                                                                    setSelectedModelId("mc13_256_all_cross_v6");
                                                                else if (e.target.value === "muscriptor")
                                                                    setSelectedModelId("muscriptor_large");
                                                            }, children: [_jsxs("option", { value: "muscriptor", children: [t18({ zh: "MuScriptor（法国 Kyutai）", en: "MuScriptor (Kyutai, France)", ja: "MuScriptor（Kyutai・フランス）" }, lang), " \u2605\u2605\u2605\u2605\u2605"] }), _jsxs("option", { value: "miros", children: [t18({ zh: "MIROS MusicFM（字节跳动）", en: "MIROS MusicFM (ByteDance)", ja: "MIROS MusicFM（ByteDance）" }, lang), " \u2605\u2605\u2605\u2605"] }), _jsxs("option", { value: "yourmt3", children: [t18({ zh: "YourMT3+（谷歌）", en: "YourMT3+ (Google)", ja: "YourMT3+（Google）" }, lang), " \u2605\u2605\u2605"] })] })] })), isMultiInstrumentMode && selectedBackend === "muscriptor" && (_jsxs("div", { className: "amt-field full-width", children: [_jsx("label", { children: t("amt.speedMode", "转谱速度") }), _jsxs("div", { className: "amt-speed-buttons", children: [_jsxs("button", { type: "button", className: `amt-speed-btn${selectedModelId === "muscriptor_small" ? " active" : ""}`, onClick: () => setSelectedModelId("muscriptor_small"), title: t18({ zh: "Small 模型（约 0.4GB）快速出草稿，试错时间大幅缩短", en: "Small model (~0.4GB) fast draft — much shorter trial-and-error", ja: "Small モデル（約0.4GB）で高速ドラフト" }, lang), children: ["\u26A1 ", t18({ zh: "快速模式", en: "Quick", ja: "高速" }, lang), amtInstalled.find((m) => m.id === "muscriptor_small")?.is_available ? "" : t18({ zh: "（未安装）", en: " (not installed)", ja: "（未導入）" }, lang)] }), _jsxs("button", { type: "button", className: `amt-speed-btn${selectedModelId === "muscriptor_large" ? " active" : ""}`, onClick: () => setSelectedModelId("muscriptor_large"), title: t18({ zh: "默认 Large 模型（约 5.5GB）精修，精度最高", en: "Default Large model (~5.5GB) for best accuracy", ja: "Large モデル（約5.5GB）で最高精度" }, lang), children: ["\uD83C\uDFAF ", t18({ zh: "精细模式", en: "Detailed", ja: "高精細" }, lang), amtInstalled.find((m) => m.id === "muscriptor_large")?.is_available ? "" : t18({ zh: "（未安装）", en: " (not installed)", ja: "（未導入）" }, lang)] })] })] })), selectedMode === "vocal_split" && (_jsx("div", { className: "amt-field full-width", children: _jsx("div", { className: "amt-settings-grid", style: { paddingTop: 0, border: 'none' }, children: _jsx("div", { className: "setting-item checkbox", children: _jsxs("label", { children: [_jsx("input", { type: "checkbox", checked: vocalSplitMergeMidi, onChange: (e) => setVocalSplitMergeMidi(e.target.checked) }), _jsx("span", { children: t("amt.vocalSplitMergeMidi", "合并人声与伴奏 MIDI") })] }) }) }) })), isMultiInstrumentMode && selectedBackend === "yourmt3" && (_jsxs("div", { className: "amt-field", children: [_jsxs("label", { children: ["YourMT3+ ", t("amt.model", "模型")] }), _jsx("select", { value: selectedModelId, onChange: (e) => setSelectedModelId(e.target.value), children: AMT_CATALOG.filter(m => m.architecture === "yourmt3_plus").map(m => (_jsx("option", { value: m.id, children: t18(m.name, lang) }, m.id))) })] })), isMultiInstrumentMode && selectedBackend === "miros" && (_jsxs("div", { className: "amt-field", children: [_jsxs("label", { children: ["MIROS ", t("amt.model", "模型")] }), _jsx("select", { value: selectedModelId, onChange: (e) => setSelectedModelId(e.target.value), children: AMT_CATALOG.filter(m => m.architecture === "miros" && m.id.startsWith("miros")).map(m => (_jsx("option", { value: m.id, children: t18(m.name, lang) }, m.id))) })] })), (isMultiInstrumentMode && selectedBackend === "muscriptor") && (_jsxs(_Fragment, { children: [_jsxs("div", { className: "amt-field", children: [_jsxs("label", { children: ["MuScriptor ", t("amt.model", "模型")] }), _jsx("select", { value: selectedModelId, onChange: (e) => setSelectedModelId(e.target.value), children: AMT_CATALOG.filter(m => m.id.startsWith("muscriptor_")).map(m => (_jsxs("option", { value: m.id, children: [t18(m.name, lang), " ", "★".repeat(m.rating ?? 3)] }, m.id))) })] }), _jsxs("div", { className: "amt-field", children: [_jsx("label", { children: t("amt.processingChain", "MuScriptor 分段衔接") }), _jsxs("select", { value: selectedMuScriptorChain, onChange: (e) => setSelectedMuScriptorChain(e.target.value), children: [_jsx("option", { value: "official", children: t("amt.chain.official", "标准分段 (官方)") }), _jsx("option", { value: "telknet", children: t("amt.chain.telknet", "Telk-Net 增强 (实验)") })] })] }), _jsxs("div", { className: "amt-field full-width", children: [_jsxs("div", { className: "amt-instruments-header", children: [_jsx("label", { children: t("amt.instruments", "输出乐器 (留空自动识别)") }), _jsxs("div", { className: "amt-instruments-actions", children: [_jsx("button", { className: "amt-action-btn-sm", onClick: () => setSelectedMuScriptorInstruments([...MUSCRIPTOR_INSTRUMENTS]), children: t("common.selectAll", "全选") }), _jsx("button", { className: "amt-action-btn-sm", onClick: () => setSelectedMuScriptorInstruments([]), children: t("common.clearAll", "清空") })] })] }), _jsx("div", { className: "amt-instruments-multi", children: MUSCRIPTOR_INSTRUMENTS.map(id => (_jsxs("label", { className: `amt-instrument-chip ${selectedMuScriptorInstruments.includes(id) ? "active" : ""}`, children: [_jsx("input", { type: "checkbox", checked: selectedMuScriptorInstruments.includes(id), onChange: (e) => {
                                                                                    if (e.target.checked) {
                                                                                        setSelectedMuScriptorInstruments([...selectedMuScriptorInstruments, id]);
                                                                                    }
                                                                                    else {
                                                                                        setSelectedMuScriptorInstruments(selectedMuScriptorInstruments.filter(x => x !== id));
                                                                                    }
                                                                                } }), MUSCRIPTOR_ZH_LABELS[id] || id] }, id))) })] })] })), !isMultiInstrumentMode && (_jsxs("div", { className: "amt-field", children: [_jsx("label", { children: t("amt.pianoModel", "钢琴模型") }), _jsx("select", { value: selectedModelId, onChange: (e) => setSelectedModelId(e.target.value), children: pianoModels.map(m => (_jsx("option", { value: m.id, children: t18(m.name, lang) }, m.id))) })] }))] })] }) }), _jsxs("div", { className: "amt-advanced-section", children: [_jsxs("button", { className: "amt-advanced-toggle-btn", onClick: () => setShowAdvanced(!showAdvanced), children: [_jsx("span", { className: `arrow ${showAdvanced ? "open" : ""}`, children: "\u25B6" }), showAdvanced ? t("amt.hideAdvanced", "隐藏高级设置") : t("amt.showAdvanced", "显示高级设置")] }), showAdvanced && (_jsxs("div", { className: "amt-advanced-panel", children: [_jsxs("div", { className: "amt-config-section", children: [_jsxs("div", { className: "amt-section-title", children: [_jsx("span", { className: "icon", children: "\uD83C\uDFBC" }), " ", t("amt.section.midi", "MIDI 输出选项")] }), _jsxs("div", { className: "amt-config-grid-v3", children: [_jsxs("div", { className: "amt-field", children: [_jsx("label", { children: t("amt.trackLayout", "音轨布局") }), _jsxs("select", { value: midiTrackMode, onChange: (e) => setMidiTrackMode(e.target.value), children: [_jsx("option", { value: "multi_track", children: t("amt.layout.multi", "多轨道 (按乐器拆分)") }), _jsx("option", { value: "single_track", children: t("amt.layout.single", "单轨道 (所有乐器合并)") })] })] }), _jsxs("div", { className: "amt-field", children: [_jsx("label", { children: t("amt.tempoMode", "速度方案") }), _jsxs("select", { value: tempoMode, onChange: (e) => setTempoMode(e.target.value), children: [_jsx("option", { value: "adaptive", children: t("amt.tempo.adaptive", "跟随原曲 (变速)") }), _jsx("option", { value: "fixed_auto", children: t("amt.tempo.auto", "自动 (固定 BPM)") }), _jsx("option", { value: "fixed_manual", children: t("amt.tempo.manual", "手动设置 BPM") })] })] }), tempoMode === "fixed_manual" && (_jsxs("div", { className: "amt-field", children: [_jsx("label", { children: t("amt.manualBpm", "BPM") }), _jsx("input", { type: "number", value: customBpm, onChange: (e) => setCustomBpm(parseInt(e.target.value)), min: 20, max: 300 })] })), _jsxs("div", { className: "amt-field", children: [_jsx("label", { children: t("amt.quantization", "量化音符 (全部轨道)") }), _jsxs("select", { value: quantizeGrid, onChange: (e) => setQuantizeGrid(e.target.value), title: t18({
                                                                            zh: "不量化 = 保留原曲节奏（默认，转换整首歌）；量化 = 把音符对齐到节拍网格，节奏更规整",
                                                                            en: "Off = keep the original rhythm (default, whole song); Quantize = snap notes to the beat grid for a tidier rhythm",
                                                                            ja: "オフ = 元のリズムを保持（デフォルト・曲全体）；量子化 = 音符を拍グリッドに揃えてリズムを整えます",
                                                                        }, lang), children: [_jsx("option", { value: "none", children: t("amt.quantize.none", "不量化") }), _jsx("option", { value: "1/4", children: "1/4" }), _jsx("option", { value: "1/8", children: "1/8" }), _jsx("option", { value: "1/16", children: "1/16" }), _jsx("option", { value: "1/32", children: "1/32" }), _jsx("option", { value: "1/64", children: "1/64" })] })] }), _jsx("div", { className: "amt-field amt-field-hint full-width", children: _jsx("span", { className: "amt-hint-text", children: t18({
                                                                        zh: "量化 = 把音符对齐到节拍网格，节奏更规整；不量化 = 保留原曲节奏（默认，转换整首歌）",
                                                                        en: "Quantize snaps notes to the beat grid for a tidier rhythm; Off keeps the original timing (default, whole song)",
                                                                        ja: "量子化は音符を拍グリッドに揃えます。オフなら元のリズムのまま（デフォルト・曲全体）",
                                                                    }, lang) }) })] })] }), _jsxs("div", { className: "amt-config-section", children: [_jsxs("div", { className: "amt-section-title", children: [_jsx("span", { className: "icon", children: "\uD83D\uDCBB" }), " ", t("amt.section.compute", "计算与硬件")] }), _jsx("div", { className: "amt-config-grid-v3", children: _jsxs("div", { className: "amt-field device-info", children: [_jsx("label", { children: t("amt.computeDevice", "计算设备") }), _jsx("div", { className: "device-name", children: hardwareInfo ? (_jsxs(_Fragment, { children: [_jsx("span", { className: "dot online" }), hardwareInfo.gpu_name || "CPU", " (", hardwareInfo.ort_build, ")"] })) : "..." })] }) })] }), _jsxs("div", { className: "amt-settings-grid", children: [_jsx("div", { className: "setting-item checkbox", children: _jsxs("label", { children: [_jsx("input", { type: "checkbox", checked: useGpu, onChange: (e) => setUseGpu(e.target.checked) }), _jsx("span", { children: t("amt.useGpu", "使用 GPU 加速") })] }) }), _jsx("div", { className: "setting-item checkbox", children: _jsxs("label", { children: [_jsx("input", { type: "checkbox", checked: removeDuplicates, onChange: (e) => setRemoveDuplicates(e.target.checked) }), _jsx("span", { children: t("amt.removeDuplicates", "移除重复音符") })] }) }), _jsx("div", { className: "setting-item checkbox", children: _jsxs("label", { children: [_jsx("input", { type: "checkbox", checked: velocitySmoothing, onChange: (e) => setVelocitySmoothing(e.target.checked) }), _jsx("span", { children: t("amt.velocitySmoothing", "力度平滑") })] }) })] }), _jsxs("div", { className: "amt-advanced-grid-v2", children: [_jsxs("div", { className: "amt-field", children: [_jsx("label", { children: t("amt.maxPolyphony", "最大复音数") }), _jsx("input", { type: "number", value: maxPolyphony, onChange: e => setMaxPolyphony(parseInt(e.target.value)), min: 1, max: 256 })] }), _jsxs("div", { className: "amt-field", children: [_jsx("label", { children: t("amt.ticksPerBeat", "每拍 Tick 数 (分辨率)") }), _jsx("input", { type: "number", value: ticksPerBeat, onChange: e => setTicksPerBeat(parseInt(e.target.value)), min: 24, max: 960 })] }), _jsxs("div", { className: "amt-field", children: [_jsx("label", { children: t("amt.defaultVelocity", "默认力度") }), _jsx("input", { type: "number", value: defaultVelocity, onChange: e => setDefaultVelocity(parseInt(e.target.value)), min: 1, max: 127 })] })] })] }))] })] })) : (_jsxs("div", { className: "amt-success-workbench", children: [_jsx("div", { className: "success-icon", children: "\uD83C\uDF89" }), _jsx("h3", { children: t("amt.conversionFinished") || "转换已完成！" }), _jsx("p", { className: "desc", children: t("amt.resultDesc", { count: result.midi_paths.length }) }), _jsx("div", { className: "amt-result-list-v2", children: result.midi_paths.map((path, idx) => (_jsxs("div", { className: "amt-result-card", children: [_jsx("span", { className: "file-icon", children: "\uD83D\uDCC4" }), _jsx("span", { className: "file-name", children: path.split(/[/\\]/).pop() })] }, idx))) }), _jsxs("div", { className: "amt-workbench-actions", children: [_jsxs("button", { className: "wb-btn primary", onClick: handleImportToTracks, children: [_jsx("span", { className: "icon", children: "\uD83D\uDCE5" }), " ", t("amt.importToTracks")] }), _jsxs("button", { className: "wb-btn", onClick: handleExportToFolder, children: [_jsx("span", { className: "icon", children: "\uD83D\uDCE6" }), " ", t("amt.exportAll")] }), _jsx("button", { className: "wb-btn secondary", onClick: () => setResult(null), children: t("amt.back") })] })] })) })] }) }));
}
