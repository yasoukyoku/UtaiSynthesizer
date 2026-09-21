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

interface AmtConversionDialogProps {
  trackId: string;
  onClose: () => void;
  /** 转谱轨道导入工程后回调,传出新建轨道 id(用于一键扒带:转完自动开编曲面板)。 */
  onImported?: (newTrackIds: string[]) => void;
}

interface ConversionProgress {
  stage: string;
  percent: number;
  message: string;
}

interface PyenvProgress {
  id: string;
  phase: string;
  progress: number;
  message: string;
  code?: string;
  params: string[];
}

interface ConversionResult {
  midi_paths: string[];
  output_dir: string;
}

export function AmtConversionDialog({ trackId, onClose, onImported }: AmtConversionDialogProps) {
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

  const [selectedModelId, setSelectedModelId] = useState<string>(AMT_CATALOG[0]?.id || "");
  const [selectedMode, setSelectedMode] = useState<string>("smart");
  const [selectedBackend, setSelectedBackend] = useState<string>("yourmt3");
  const [selectedMuScriptorInstruments, setSelectedMuScriptorInstruments] = useState<string[]>([]);
  // Source contract (data_models.MuscriptorProcessingChain): ONLY "official" / "telknet".
  // "sustain_connect" was a merged-in value the Python side rejects outright.
  const [selectedMuScriptorChain, setSelectedMuScriptorChain] = useState<string>("official");
  const [midiTrackMode, setMidiTrackMode] = useState<string>("multi_track");
  const [vocalSplitMergeMidi, setVocalSplitMergeMidi] = useState(false);
  const [removeDuplicates, setRemoveDuplicates] = useState(true);
  const [velocitySmoothing, setVelocitySmoothing] = useState(true);
  const [maxPolyphony, setMaxPolyphony] = useState(40);
  const [ticksPerBeat, setTicksPerBeat] = useState(480);
  const [defaultVelocity, setDefaultVelocity] = useState(80);
  const [hardwareInfo, setany] = useState<any | null>(null);
  const [isConverting, setIsConverting] = useState(false);
  const [progress, setProgress] = useState<ConversionProgress | null>(null);
  const [result, setResult] = useState<ConversionResult | null>(null);
  /** Playback synthesis assets from the last successful conversion: per-instrument
   *  WAVs keyed "gm:NNN"/"drums" + the source audio + merged MIDI path. The
   *  import-to-tracks flow uses these so every imported MIDI track carries a
   *  playable audio lane (imported notes alone are silent in the DAW). */
  const [playbackAssets, setPlaybackAssets] = useState<{
    audioPath: string;
    mergedMidiPath: string;
    wavs: Record<string, string>;
  } | null>(null);
  const [sidecarInstalled, setSidecarInstalled] = useState<boolean | null>(null);
  const [pyenvProgress, setPyenvProgress] = useState<PyenvProgress | null>(null);
  const [isInstallingPyenv, setIsInstallingPyenv] = useState(false);
  // Manual fallback when the source audio path cannot be auto-resolved from the track/segment.
  const [manualSourcePath, setManualSourcePath] = useState<string | null>(null);
  
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
  
  const audioRef = useRef<HTMLAudioElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // Set when the user clicks "取消" mid-conversion. run_amt_midi then rejects because the
  // sidecar is killed; we settle as an info toast instead of a red error.
  const cancelledRef = useRef(false);
  const [audioDuration, setAudioDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [peaksData, setPeaksData] = useState<number[]>([]);
  const [isHoveringWave, setIsHoveringWave] = useState(false);
  const [hoverFrac, setHoverFrac] = useState(0);
  // In-scope playable URL (content-addressed cache WAV) for the <audio> element. The raw
  // source path can be OUTSIDE the asset-protocol scope — see the load effect above.
  const [playbackSrc, setPlaybackSrc] = useState<string | null>(null);

  useEffect(() => {
    invoke<any>("amt_sidecar_installed").then(setSidecarInstalled).catch(() => setSidecarInstalled(false));
    invoke<any>("get_hardware_info").then(setany).catch(() => {});
    useAmtModelStore.getState().fetchInstalled();
  }, []);

  useEffect(() => {
    const unlisten = listen<PyenvProgress>("pyenv-progress", (event) => {
      setPyenvProgress(event.payload);
      if (event.payload.phase === "done") {
        setSidecarInstalled(true);
        setIsInstallingPyenv(false);
        showToast(t("amt.envInstalled") || "AMT 运行环境安装成功", "success");
        setTimeout(() => setPyenvProgress(null), 3000);
      } else if (event.payload.phase === "error") {
        setIsInstallingPyenv(false);
        showToast(event.payload.message, "error");
      }
    });
    return () => { void unlisten.then(fn => fn()); };
  }, [t, showToast]);

  const handleDownloadRuntime = useCallback(async () => {
    if (isInstallingPyenv) return;
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
    } catch (e) {
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
    type Prog = { node_id: string | null; stage: string | null; progress: number; total: number; message: string | null };
    const unlisten = listen<Prog>("amt-progress", (event) => {
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
  let resolvedAudioPath: string | undefined = undefined;
  let resolvedSegmentName = "No Source";
  let resolvedSegment: { content?: unknown } | null = null;

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
      if (s) candidates.push(s);
    }

    // Add selected segment if different from explicit
    const selSegId = useAppStore.getState().selectedSegment?.segmentId;
    if (selSegId && selSegId !== amtConversionSegmentId) {
      const s = track.segments.find(sg => sg.id === selSegId);
      if (s) candidates.push(s);
    }

    // Add all other segments
    if (track.segments) {
      candidates.push(...track.segments.filter(s =>
        s.id !== amtConversionSegmentId && s.id !== selSegId
      ));
    }

    for (const seg of candidates) {
      if (!seg) continue;

      let p: string | undefined = undefined;

      // Helper to get path from various properties
      const getPath = (obj: any): string | undefined => {
        if (!obj) return undefined;
        return obj.audioPath || obj.sourcePath || obj.filePath;
      };

      // 1. Check segment content (audioClip, notes with processedOutputs, vocal type)
      if (seg.content) {
        if (seg.content.type === "audioClip" && (seg.content as { sourcePath?: string }).sourcePath) {
          p = (seg.content as { sourcePath?: string }).sourcePath;
        } else if (seg.content.type === "notes" && seg.processedOutputs && seg.processedOutputs.length > 0) {
          for (const out of seg.processedOutputs) {
            p = getPath(out);
            if (p) break;
          }
        } else if ((seg.content as { type?: string }).type === "vocal") {
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
          (seg.content as { label?: string }).label ||
          (seg.content as { name?: string }).name ||
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
          if (durationMs === 0) setDurationMs(Math.floor(data.durationMs));
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
        if (durationMs === 0) setDurationMs(Math.floor(audio.duration * 1000));
      };
      audio.onerror = (e) => {
        console.error("Failed to load audio for preview:", safeUrl, e);
      };
    }
  }, [audioPath]);

  // Audio play/pause and timeupdate synchronization
  const togglePlayAudio = () => {
    const el = audioRef.current;
    if (!el) return;
    if (el.paused) {
      el.play().catch(e => console.warn("Audio play error:", e));
      setIsPlayingAudio(true);
    } else {
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

  const seekAudio = (frac: number) => {
    const el = audioRef.current;
    if (!el || !audioDuration) return;
    const target = Math.max(0, Math.min(1, frac)) * audioDuration;
    el.currentTime = target;
    setCurrentTime(target);
  };

  // Draw Waveform on Canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

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
          if (val > p) p = val;
        }

        const barX = c * colWidth;
        const colCenterSec = (c / numCols) * totalDur;
        const isInTrim = !isTrimming || (colCenterSec >= trimStartSec && colCenterSec <= trimEndSec);
        const isPastPlayhead = colCenterSec <= currentTime;

        // Determine column bar color
        if (isPastPlayhead) {
          ctx.fillStyle = isInTrim ? "#00e5ff" : "rgba(0, 229, 255, 0.35)";
        } else {
          ctx.fillStyle = isInTrim ? "rgba(96, 165, 250, 0.75)" : "rgba(255, 255, 255, 0.2)";
        }

        const barH = Math.max(2 * dpr, p * amp);
        const barW = Math.max(1 * dpr, colWidth - 1 * dpr);
        ctx.fillRect(barX, midY - barH, barW, barH * 2);
      }
    } else {
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
      
      const aliases: Record<string, string[]> = {
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

      let inferred: string[] = [];
      const tokens = normalized.split("_");
      const lastToken = tokens[tokens.length - 1] as string | undefined;
      
      if (aliases[normalized]) inferred = aliases[normalized];
      else if (lastToken && aliases[lastToken]) inferred = aliases[lastToken];
      else {
        // Try exact matches
        const exact = MUSCRIPTOR_INSTRUMENTS.find(inst => normalized === inst || normalized.endsWith(`_${inst}`));
        if (exact) inferred = [exact];
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
    if (!midiPath) return;
    try {
      const score = await invoke<any>("import_score_file", { path: midiPath });
      if (!score || (score.tracks?.length ?? 0) === 0) {
        throw new Error("empty score");
      }
      openAmtResult({
        trackId: `midi-repair-` + Date.now(),
        outputDir: midiPath.replace(/[\\/][^\\/]+$/, "") || ".",
        midiPaths: [midiPath],
      });
      onClose();
    } catch (e) {
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
    if (!model) return;

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
      const res = await invoke<any>("run_amt_midi", {
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
      const separatedAudio = (res.separated_audio ?? {}) as Record<string, string>;
      const isSeparationOnly = !midiPath && Object.values(separatedAudio).filter(Boolean).length > 0;
      if (!midiPath && !isSeparationOnly) {
        throw new Error("No MIDI path returned from sidecar");
      }

      if (isSeparationOnly) {
        const sepStems: Record<string, { midi: string; audio?: string }> = {};
        for (const [name, path] of Object.entries(separatedAudio)) {
          if (path) sepStems[name] = { midi: "", audio: path };
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
      let playbackRes: any = { transcription_wav: "", original_wav: "", instrument_wavs: {} } as any;
      try {
        setProgress({ stage: "synthesize", percent: 90, message: t("amt.preparingPlayback") || "正在合成预览音频..." });
        playbackRes = await invoke<any>("amt_prepare_playback", {
          midiPath: midiPath,
          audioPath: audioPath,
          outputDir: audioPath.replace(/\.[^/.]+$/, "") + "_midi"
        });
      } catch (e) {
        console.warn("Playback prep failed (best-effort — opening result anyway):", e);
      }

      // Adapt result to ConversionResult interface. Always include the primary MIDI (the per-track
      // midis may be absent if the backend returned only a merged file — import_score_file then exposes
      // each instrument as a separate track in the panel).
      const midiPaths = res.stem_midi_paths
        ? Object.values(res.stem_midi_paths).filter(Boolean)
        : [midiPath];
      const adaptedResult: ConversionResult = {
        midi_paths: midiPaths.length > 0 ? midiPaths : [midiPath],
        output_dir: audioPath.replace(/\.[^/.]+$/, "") + "_midi"
      };
      
      setResult(adaptedResult);
      setPlaybackAssets({
        audioPath,
        mergedMidiPath: midiPath,
        wavs: (playbackRes?.instrument_wavs ?? {}) as Record<string, string>,
      });
      showToast(t("amt.conversionSuccess"), "success");

      // 3. Automatically open the result workbench at the bottom.
      const instrumentWavs = (playbackRes.instrument_wavs ?? {}) as Record<string, string>;
      // §user "干音都要有歌词"：人声分离出的干音 WAV —— 结果工作台自动歌词提取的最准源。
      const dryVocalAudio =
        separatedAudio["vocals"] || separatedAudio["vocal"] || "";
      openAmtResult({
        trackId,
        outputDir: adaptedResult.output_dir,
        midiPaths: adaptedResult.midi_paths,
        audioPreview: playbackRes.transcription_wav,
        originalAudio: playbackRes.original_wav,
        sourceAudioPath: audioPath,
        vocalAudioPath: dryVocalAudio,
        stems: Object.entries(instrumentWavs).reduce((acc, [k, v]) => {
          acc[k] = { midi: "", audio: v as string };
          return acc;
        }, {} as any)
      });
      onClose();
    } catch (e: unknown) {
      const errStr = String(e);
      console.error("AMT Conversion failed:", errStr);
      if (cancelledRef.current) {
        // User cancelled — the sidecar was force-killed. Report as a clean cancel.
        await invoke("cancel_amt_midi", { nodeId: trackId }).catch(() => {});
        showToast(t("amt.cancelled", "转换已取消"), "info");
        return;
      }
      // Extract just the first line for the toast if it's a long error
      const shortErr = errStr.split('\n')[0] ?? errStr;
      showToast(shortErr, "error");
    } finally {
      setIsConverting(false);
      setProgress(null);
    }
  };

  /** Force-stop the running AMT sidecar for this dialog's conversion. */
  const handleCancelConversion = () => {
    cancelledRef.current = true;
    setProgress((prev) => ({ stage: "cancelling", percent: (prev?.percent ?? 0), message: t("amt.cancelling", "正在取消...") }));
    invoke("cancel_amt_midi", { nodeId: trackId }).catch(() => {});
  };

  const handleImportToTracks = async () => {
    if (!result) return;
    try {
      useHistoryStore.getState().beginTransaction();
      const importedIds: string[] = [];

      // Guarantee per-instrument WAVs exist: the dialog's playback prep is best-effort,
      // so if it failed (or produced no instrument WAVs) synthesize NOW — imported
      // notes-only tracks would otherwise be silent in the DAW.
      let wavs: Record<string, string> = playbackAssets?.wavs ?? {};
      const srcAudio = playbackAssets?.audioPath;
      const mergedMidi = playbackAssets?.mergedMidiPath || result.midi_paths[0];
      if (Object.keys(wavs).length === 0 && srcAudio && mergedMidi) {
        try {
          const pb = await invoke<any>("amt_prepare_playback", {
            midiPath: mergedMidi,
            audioPath: srcAudio,
            outputDir: result.output_dir,
          });
          wavs = (pb?.instrument_wavs ?? {}) as Record<string, string>;
        } catch (e) {
          console.warn("Import-time playback synthesis failed:", e);
        }
      }
      if (result.output_dir) {
        await invoke("allow_asset_dir", { dir: result.output_dir }).catch(() => {});
      }

      for (const path of result.midi_paths) {
        const score = await invoke<any>("import_score_file", { path });
        // Per-file GM metadata (program+channel per track) maps track names onto
        // the sidecar's "gm:NNN"/"drums" WAV keys — the names alone never match.
        let metaTracks: any = { track_count: 0, total_notes: 0, ppq: 480, tracks: [], size_bytes: 0 } as any;
        try {
          const meta = await invoke<any>("amt_midi_metadata", { midiPath: path });
          metaTracks = meta?.tracks ?? [];
        } catch { /* best-effort — name matching still runs below */ }
        for (const it of score.tracks) {
          const newTrackId = crypto.randomUUID();
          importedIds.push(newTrackId);
          const segmentId = crypto.randomUUID();
          const maxTick = Math.max(1, it.notes.reduce((max: number, n: any) => Math.max(max, n.tick + n.duration), 0));

          const newTrack = blankTrack(newTrackId, it.name || path.split(/[/\\]/).pop() || "MIDI Track", "instrument");

          // Attach the per-instrument synthesized WAV as a playable audio lane so the
          // DAW transport produces sound for this track (notes alone are silent).
          const mt = metaTracks.tracks.find((m: any) => m.name === it.name) || metaTracks[0];
          const wav = matchInstrumentWav(wavs, it.name || "", mt?.program ?? null, mt?.channel ?? null);
          const processedOutputs: any[] = [];
          if (wav) {
            const safePpq = 480;
            const safeBpm = 120;
            let totalDurationMs = Math.max(1, (maxTick / (safePpq * (safeBpm / 60))) * 1000);
            let waveformPeaks: number[] | undefined;
            try {
              const data = await useAudioStore.getState().loadAudioFile(wav);
              if (data.durationMs > 0) totalDurationMs = data.durationMs;
              if (data.peaks && data.peaks.length > 0) waveformPeaks = data.peaks;
            } catch { /* best-effort waveform */ }
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
              notes: it.notes.map((n: any) => ({
                ...n,
                id: crypto.randomUUID(),
                lyric: n.lyric || "La",
                velocity: (n as { velocity?: number }).velocity ?? 100
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
      showToast(
        silent
          ? (t("amt.importSuccessNoAudio") || "已导入轨道（未找到试听音频）")
          : t("amt.importSuccess"),
        silent ? "info" : "success",
      );
      if (importedIds.length > 0) onImported?.(importedIds);
      handleClose();
    } catch (e) {
      useHistoryStore.getState().commitTransaction();
      showToast(String(e), "error");
    }
  };

  const handleExportToFolder = async () => {
    if (!result) return;
    try {
      const { save } = await import("@tauri-apps/plugin-dialog");
      const savePath = await save({
        filters: [{ name: "ZIP Archive", extensions: ["zip"] }],
        defaultPath: `${track?.name || "converted"}_midi.zip`
      });
      
      if (!savePath || typeof savePath !== "string") return;

      await invoke("amt_export_zip", {
        midiPaths: result.midi_paths,
        savePath
      });
      showToast(t("amt.exportSuccess"), "success");
    } catch (e) {
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

  return (
    <div className="amt-dialog-overlay" onClick={handleClose}>
      <div className="amt-dialog" onClick={e => e.stopPropagation()}>
        <div className="amt-dialog-header">
          <div className="amt-title-with-icon">
            <span className="amt-header-icon">🎹</span>
            <h2>{t("amt.dialogTitle")}</h2>
          </div>
          <button className="amt-close-btn" onClick={handleClose}>×</button>
        </div>

        <div className="amt-dialog-toolbar">
          {isConverting && progress && (
            <div className="amt-floating-progress">
              <div className="progress-bar-bg">
                <div className="progress-bar-fill" style={{ width: `${progress.percent}%` }} />
              </div>
              <div className="progress-info">
                <span className="msg">{progress.message}</span>
                <span className="pct">{progress.percent}%</span>
              </div>
            </div>
          )}
          {!result && (
            <div className="amt-actions-v2">
              {isConverting ? (
                <>
                  <button
                    className="amt-main-convert-btn"
                    disabled
                    style={{ opacity: 0.55, cursor: "default" }}
                  >
                    <span className="spinner"></span>
                    {t("amt.converting", "正在转换...")}
                  </button>
                  <button
                    className="amt-cancel-convert-btn"
                    onClick={handleCancelConversion}
                    title={t("amt.cancelConversion", "终止转换")}
                  >
                    ✕ {t("amt.cancelConversion", "终止")}
                  </button>
                </>
              ) : (
                <>
                  <button
                    className="amt-main-convert-btn"
                    onClick={handleStartConversion}
                  >
                    <span className="icon">🚀</span>
                    {t("amt.startConversion", "开始转换")}
                  </button>
                  {/* §user「只要是 MIDI 文件打开就可以修复」：不经过音频转换，直接把
                      本地 MIDI 载入修复工作台（钢琴卷帘 + 一键修复 + 换音源 + 导出）。 */}
                  <button
                    className="amt-local-midi-btn"
                    onClick={handleOpenLocalMidi}
                    title={t("amt.repairLocalMidiTip") || "打开任意 .mid/.midi 文件，直接进入修复工作台：一键修复多余/缺失音符、换音源、导出"}
                  >
                    🩹 {t("amt.repairLocalMidi") || "修复本地 MIDI 文件…"}
                  </button>
                </>
              )}
            </div>
          )}
        </div>

        <div className="amt-dialog-content">
          {!result ? (
            <>
              <div className="amt-source-card">
                <div className="amt-source-header">
                  <div className="amt-source-info">
                    <div className="amt-source-label">{t("amt.sourceTrack", "源轨道 / 片段")}</div>
                    <div className="amt-source-track" title={displayTrackName}>{`🎵 ${displayTrackName}`}</div>
                    <div className="amt-source-name" title={segmentName}>{segmentName}</div>
                    <div className="amt-source-path" title={audioPath}>{audioPath || t("amt.noSource", "轨道没有音频源，无法转换")}</div>
                  </div>
                  <div className="amt-source-actions">
                    <button 
                      className={`amt-trim-toggle ${isTrimming ? "active" : ""}`}
                      onClick={() => setIsTrimming(!isTrimming)}
                    >
                      ✂️ {t("amt.trimSegment", "剪切片段")}
                    </button>
                    {!audioPath && (
                      <button className="amt-trim-toggle" onClick={async () => {
                        const { open } = await import("@tauri-apps/plugin-dialog");
                        const picked = await open({
                          multiple: false,
                          filters: [{ name: "Audio", extensions: ["wav", "mp3", "m4a", "flac", "ogg", "aac", "wma", "opus", "mid", "midi"] }],
                        });
                        if (picked && !Array.isArray(picked)) setManualSourcePath(picked);
                      }}>
                        📂 {t("amt.chooseSourceFile", "选择音频文件")}
                      </button>
                    )}
                  </div>
                </div>

                {audioPath && (
                  <div className="amt-waveform-player-card">
                    <div 
                      className="amt-waveform-canvas-wrap"
                      onMouseEnter={() => setIsHoveringWave(true)}
                      onMouseLeave={() => setIsHoveringWave(false)}
                      onMouseMove={(e) => {
                        const rect = e.currentTarget.getBoundingClientRect();
                        const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / Math.max(1, rect.width)));
                        setHoverFrac(frac);
                      }}
                      onClick={(e) => {
                        const rect = e.currentTarget.getBoundingClientRect();
                        const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / Math.max(1, rect.width)));
                        seekAudio(frac);
                      }}
                    >
                      <canvas ref={canvasRef} className="amt-waveform-canvas" />
                      {isHoveringWave && (
                        <div 
                          className="amt-waveform-hover-time" 
                          style={{ left: `${hoverFrac * 100}%` }}
                        >
                          {fmtDur(hoverFrac * (audioDuration || 0))}
                        </div>
                      )}
                    </div>

                    <div className="amt-player-controls-bar">
                      <div className="amt-player-left">
                        <button 
                          className={`amt-play-btn ${isPlayingAudio ? "playing" : ""}`}
                          onClick={togglePlayAudio}
                          title={isPlayingAudio ? t("common.pause", "暂停") : t("common.play", "播放")}
                        >
                          {isPlayingAudio ? "⏸" : "▶"}
                        </button>
                        <div className="amt-player-time-display">
                          <span className="current">{fmtDur(currentTime)}</span>
                          <span className="divider">/</span>
                          <span className="total">{fmtDur(audioDuration)}</span>
                        </div>
                      </div>

                      <div className="amt-player-center">
                        <div 
                          className="amt-scrubber-track"
                          onClick={(e) => {
                            const rect = e.currentTarget.getBoundingClientRect();
                            const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / Math.max(1, rect.width)));
                            seekAudio(frac);
                          }}
                        >
                          <div 
                            className="amt-scrubber-fill" 
                            style={{ width: `${audioDuration > 0 ? (currentTime / audioDuration) * 100 : 0}%` }} 
                          />
                          <div 
                            className="amt-scrubber-handle" 
                            style={{ left: `${audioDuration > 0 ? (currentTime / audioDuration) * 100 : 0}%` }} 
                          />
                        </div>
                      </div>

                      <div className="amt-player-right">
                        {/* §user "任何转换MIDI默认整首歌": the conversion range is ALWAYS stated —
                            full song by default (trim off), the chosen window when trimming. */}
                        {isTrimming ? (
                          <span className="amt-trim-summary-badge">
                            {t("amt.rangeLabel", "转换范围")}: {fmtDur(startTimeMs / 1000)} ~ {fmtDur(Math.min(audioDuration, (startTimeMs + durationMs) / 1000))}
                          </span>
                        ) : (
                          audioDuration > 0 && (
                            <span className="amt-trim-summary-badge">
                              {t("amt.fullSongRange", "整首歌")} · {fmtDur(0)} ~ {fmtDur(audioDuration)}
                            </span>
                          )
                        )}
                      </div>
                    </div>

                    {/* Hidden Native Audio Element to Handle Real Audio Playback & Decoding.
                        playbackSrc (cache WAV, in-scope) wins over the raw source path —
                        see the load effect for why the raw convertFileSrc can be blocked. */}
                    <audio
                      ref={audioRef}
                      src={playbackSrc || convertFileSrc(audioPath.replace(/\\/g, "/"))}
                      onTimeUpdate={handleAudioTimeUpdate}
                      onEnded={handleAudioEnded}
                      style={{ display: "none" }}
                    />
                  </div>
                )}

                {isTrimming && (
                  <div className="amt-trim-controls">
                    <div className="amt-trim-field">
                      <label>{t("amt.startTime", "开始时间 (秒)")}</label>
                      <input 
                        type="number" 
                        step="0.1"
                        min="0"
                        max={audioDuration}
                        value={startTimeMs / 1000}
                        onChange={(e) => setStartTimeMs(parseFloat(e.target.value) * 1000)}
                      />
                    </div>
                    <div className="amt-trim-field">
                      <label>{t("amt.duration", "持续时长 (秒)")}</label>
                      <input 
                        type="number" 
                        step="0.1"
                        min="0.1"
                        max={audioDuration - (startTimeMs / 1000)}
                        value={durationMs / 1000}
                        onChange={(e) => setDurationMs(parseFloat(e.target.value) * 1000)}
                      />
                    </div>
                  </div>
                )}
              </div>

              {!sidecarInstalled && sidecarInstalled !== null && (
                <div className="amt-sidecar-warning">
                  <div className="amt-warning-icon">⚠️</div>
                  <div className="amt-warning-content">
                    <p>{t("amt.sidecarNotInstalled", "AMT 运行环境未安装，请先下载安装")}</p>
                    {pyenvProgress ? (
                      <div className="amt-pyenv-progress-container">
                        <div className="download-progress-container">
                          <div className="download-progress-bar" style={{ width: `${pyenvProgress.progress * 100}%` }} />
                          <span className="download-progress-text">
                            {pyenvProgress.phase === "download" ? `${(pyenvProgress.progress * 100).toFixed(1)}%` : pyenvProgress.message}
                          </span>
                        </div>
                        <p className="amt-pyenv-msg">{pyenvProgress.message}</p>
                      </div>
                    ) : (
                      <div className="amt-warning-actions">
                        <button className="amt-dl-btn" onClick={handleDownloadRuntime} disabled={isInstallingPyenv}>
                          {isInstallingPyenv ? "正在启动..." : (t("amt.downloadEnv", "下载 AMT 运行环境 (约 1.2GB)"))}
                        </button>
                        <button className="amt-link-btn" onClick={toggleModelManager}>
                          {t("missingModels.openManager", "打开资源管理")}
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              )}

              <div className="amt-components-card">
                <div className="amt-card-header">
                  <span className="icon">🎵</span>
                  <span>{t("amt.playbackComponents", "播放组件状态")}</span>
                </div>
                <div className="amt-components-list">
                  {requiredComponents.map(comp => {
                    const installedItem = amtInstalled.find(i => i.id === comp.id || i.architecture === comp.architecture);
                    const isInstalled = installedItem?.is_available;
                    const isDl = !!amtDownloading[comp.id];
                    return (
                      <div key={comp.id} className={`amt-comp-item ${isInstalled ? "installed" : "missing"}`}>
                        <div className="amt-comp-info">
                          <span className="amt-comp-name">{t18(comp.name, lang)}</span>
                          <span className="amt-comp-status">{isInstalled ? "✅" : "❌"}</span>
                        </div>
                        {!isInstalled ? (
                          <button 
                            className={`amt-comp-dl-btn-sm ${isDl ? "loading" : ""}`}
                            disabled={isDl}
                            onClick={() => {
                              console.log("Triggering download for component:", comp);
                              downloadAmtEntry(comp);
                            }}
                          >
                            {isDl ? "..." : t("missingModels.download", "下载")}
                          </button>
                        ) : (
                          <span className="amt-comp-installed-text" style={{ fontSize: "12px", color: "var(--success-color)", fontWeight: 500 }}>
                            {t("amt.installed", "已安装")}
                          </span>
                        )}
                      </div>
                    );
                  })}
                </div>
                {missingComponents.length > 0 && (
                  <p className="amt-comp-tip">{t("amt.playbackTip", "安装上述组件以启用转换后的高保真预览与导出")}</p>
                )}
              </div>

              <div className="amt-config-sections">
                <div className="amt-config-section">
                  <div className="amt-section-title">
                    <span className="icon">🎯</span> {t("amt.section.model", "模型与模式")}
                  </div>
                  <div className="amt-config-grid-v3">
                    <div className="amt-mode-cards">
                      {[
                        { id: "smart", icon: "🎸", descKey: "smart" },
                        { id: "vocal_split", icon: "🎤", descKey: "vocal_split" },
                        { id: "six_stem_split", icon: "🎚", descKey: "six_stem_split" },
                        { id: "piano_transkun", icon: "🎹", descKey: "piano_transkun" },
                        { id: "piano_transkun_v2_aug", icon: "🎹", descKey: "piano_transkun_v2_aug" },
                        { id: "piano_aria_amt", icon: "🎹", descKey: "piano_aria_amt" },
                        { id: "piano_bytedance_pedal", icon: "🎹", descKey: "piano_bytedance_pedal" },
                      ].map((m) => (
                        <button
                          type="button"
                          key={m.id}
                          className={`amt-mode-card ${selectedMode === m.id ? "active" : ""}`}
                          onClick={() => setSelectedMode(m.id)}
                        >
                          <span className="amt-mode-icon">{m.icon}</span>
                          <span className="amt-mode-label">{t(`amt.mode.${m.descKey}`)}</span>
                        </button>
                      ))}
                    </div>

                    {isMultiInstrumentMode && (
                      <div className="amt-field">
                        <label>{t("amt.backend", "转谱后端")}</label>
                        <select value={selectedBackend} onChange={(e) => {
                          setSelectedBackend(e.target.value);
                          if (e.target.value === "yourmt3") setSelectedModelId("yptf_moe_multi_nops");
                          else if (e.target.value === "miros") setSelectedModelId("mc13_256_all_cross_v6");
                          else if (e.target.value === "muscriptor") setSelectedModelId("muscriptor_large");
                        }}>
                          <option value="muscriptor">{t18({ zh: "MuScriptor（法国 Kyutai）", en: "MuScriptor (Kyutai, France)", ja: "MuScriptor（Kyutai・フランス）" }, lang)} ★★★★★</option>
                          <option value="miros">{t18({ zh: "MIROS MusicFM（字节跳动）", en: "MIROS MusicFM (ByteDance)", ja: "MIROS MusicFM（ByteDance）" }, lang)} ★★★★</option>
                          <option value="yourmt3">{t18({ zh: "YourMT3+（谷歌）", en: "YourMT3+ (Google)", ja: "YourMT3+（Google）" }, lang)} ★★★</option>
                        </select>
                      </div>
                    )}

                    {/* §user「快速模式/精细模式双按钮」：Small 模型出草稿 + 默认模型精修。
                        仅 MuScriptor 有大小分级（Small ~0.4GB / Large ~5.5GB），其它后端单模型。 */}
                    {isMultiInstrumentMode && selectedBackend === "muscriptor" && (
                      <div className="amt-field full-width">
                        <label>{t("amt.speedMode", "转谱速度")}</label>
                        <div className="amt-speed-buttons">
                          <button
                            type="button"
                            className={`amt-speed-btn${selectedModelId === "muscriptor_small" ? " active" : ""}`}
                            onClick={() => setSelectedModelId("muscriptor_small")}
                            title={t18({ zh: "Small 模型（约 0.4GB）快速出草稿，试错时间大幅缩短", en: "Small model (~0.4GB) fast draft — much shorter trial-and-error", ja: "Small モデル（約0.4GB）で高速ドラフト" }, lang)}
                          >
                            ⚡ {t18({ zh: "快速模式", en: "Quick", ja: "高速" }, lang)}
                            {amtInstalled.find((m) => m.id === "muscriptor_small")?.is_available ? "" : t18({ zh: "（未安装）", en: " (not installed)", ja: "（未導入）" }, lang)}
                          </button>
                          <button
                            type="button"
                            className={`amt-speed-btn${selectedModelId === "muscriptor_large" ? " active" : ""}`}
                            onClick={() => setSelectedModelId("muscriptor_large")}
                            title={t18({ zh: "默认 Large 模型（约 5.5GB）精修，精度最高", en: "Default Large model (~5.5GB) for best accuracy", ja: "Large モデル（約5.5GB）で最高精度" }, lang)}
                          >
                            🎯 {t18({ zh: "精细模式", en: "Detailed", ja: "高精細" }, lang)}
                            {amtInstalled.find((m) => m.id === "muscriptor_large")?.is_available ? "" : t18({ zh: "（未安装）", en: " (not installed)", ja: "（未導入）" }, lang)}
                          </button>
                        </div>
                      </div>
                    )}

                    {selectedMode === "vocal_split" && (
                      <div className="amt-field full-width">
                        <div className="amt-settings-grid" style={{ paddingTop: 0, border: 'none' }}>
                          <div className="setting-item checkbox">
                            <label>
                              <input 
                                type="checkbox" 
                                checked={vocalSplitMergeMidi} 
                                onChange={(e) => setVocalSplitMergeMidi(e.target.checked)}
                              />
                              <span>{t("amt.vocalSplitMergeMidi", "合并人声与伴奏 MIDI")}</span>
                            </label>
                          </div>
                        </div>
                      </div>
                    )}

                    {isMultiInstrumentMode && selectedBackend === "yourmt3" && (
                      <div className="amt-field">
                        <label>YourMT3+ {t("amt.model", "模型")}</label>
                        <select value={selectedModelId} onChange={(e) => setSelectedModelId(e.target.value)}>
                          {AMT_CATALOG.filter(m => m.architecture === "yourmt3_plus").map(m => (
                            <option key={m.id} value={m.id}>{t18(m.name, lang)}</option>
                          ))}
                        </select>
                      </div>
                    )}

                    {isMultiInstrumentMode && selectedBackend === "miros" && (
                      <div className="amt-field">
                        <label>MIROS {t("amt.model", "模型")}</label>
                        <select value={selectedModelId} onChange={(e) => setSelectedModelId(e.target.value)}>
                          {AMT_CATALOG.filter(m => m.architecture === "miros" && m.id.startsWith("miros")).map(m => (
                            <option key={m.id} value={m.id}>{t18(m.name, lang)}</option>
                          ))}
                        </select>
                      </div>
                    )}

                    {(isMultiInstrumentMode && selectedBackend === "muscriptor") && (
                      <>
                        <div className="amt-field">
                          <label>MuScriptor {t("amt.model", "模型")}</label>
                          <select value={selectedModelId} onChange={(e) => setSelectedModelId(e.target.value)}>
                            {AMT_CATALOG.filter(m => m.id.startsWith("muscriptor_")).map(m => (
                              <option key={m.id} value={m.id}>{t18(m.name, lang)} {"★".repeat(m.rating ?? 3)}</option>
                            ))}
                          </select>
                        </div>
                        <div className="amt-field">
                          <label>{t("amt.processingChain", "MuScriptor 分段衔接")}</label>
                          <select value={selectedMuScriptorChain} onChange={(e) => setSelectedMuScriptorChain(e.target.value)}>
                            <option value="official">{t("amt.chain.official", "标准分段 (官方)")}</option>
                            <option value="telknet">{t("amt.chain.telknet", "Telk-Net 增强 (实验)")}</option>
                          </select>
                        </div>
                        <div className="amt-field full-width">
                          <div className="amt-instruments-header">
                            <label>{t("amt.instruments", "输出乐器 (留空自动识别)")}</label>
                            <div className="amt-instruments-actions">
                              <button className="amt-action-btn-sm" onClick={() => setSelectedMuScriptorInstruments([...MUSCRIPTOR_INSTRUMENTS])}>
                                {t("common.selectAll", "全选")}
                              </button>
                              <button className="amt-action-btn-sm" onClick={() => setSelectedMuScriptorInstruments([])}>
                                {t("common.clearAll", "清空")}
                              </button>
                            </div>
                          </div>
                          <div className="amt-instruments-multi">
                            {MUSCRIPTOR_INSTRUMENTS.map(id => (
                              <label key={id} className={`amt-instrument-chip ${selectedMuScriptorInstruments.includes(id) ? "active" : ""}`}>
                                <input 
                                  type="checkbox" 
                                  checked={selectedMuScriptorInstruments.includes(id)}
                                  onChange={(e) => {
                                    if (e.target.checked) {
                                      setSelectedMuScriptorInstruments([...selectedMuScriptorInstruments, id]);
                                    } else {
                                      setSelectedMuScriptorInstruments(selectedMuScriptorInstruments.filter(x => x !== id));
                                    }
                                  }}
                                />
                                {MUSCRIPTOR_ZH_LABELS[id] || id}
                              </label>
                            ))}
                          </div>
                        </div>
                      </>
                    )}

                    {!isMultiInstrumentMode && (
                      <div className="amt-field">
                        <label>{t("amt.pianoModel", "钢琴模型")}</label>
                        <select value={selectedModelId} onChange={(e) => setSelectedModelId(e.target.value)}>
                          {pianoModels.map(m => (
                            <option key={m.id} value={m.id}>{t18(m.name, lang)}</option>
                          ))}
                        </select>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              <div className="amt-advanced-section">
                <button className="amt-advanced-toggle-btn" onClick={() => setShowAdvanced(!showAdvanced)}>
                  <span className={`arrow ${showAdvanced ? "open" : ""}`}>▶</span>
                  {showAdvanced ? t("amt.hideAdvanced", "隐藏高级设置") : t("amt.showAdvanced", "显示高级设置")}
                </button>
                
                {showAdvanced && (
                  <div className="amt-advanced-panel">
                  <div className="amt-config-section">
                  <div className="amt-section-title">
                    <span className="icon">🎼</span> {t("amt.section.midi", "MIDI 输出选项")}
                  </div>
                  <div className="amt-config-grid-v3">
                    <div className="amt-field">
                      <label>{t("amt.trackLayout", "音轨布局")}</label>
                      <select value={midiTrackMode} onChange={(e) => setMidiTrackMode(e.target.value)}>
                        <option value="multi_track">{t("amt.layout.multi", "多轨道 (按乐器拆分)")}</option>
                        <option value="single_track">{t("amt.layout.single", "单轨道 (所有乐器合并)")}</option>
                      </select>
                    </div>

                    <div className="amt-field">
                      <label>{t("amt.tempoMode", "速度方案")}</label>
                      <select value={tempoMode} onChange={(e) => setTempoMode(e.target.value)}>
                        <option value="adaptive">{t("amt.tempo.adaptive", "跟随原曲 (变速)")}</option>
                        <option value="fixed_auto">{t("amt.tempo.auto", "自动 (固定 BPM)")}</option>
                        <option value="fixed_manual">{t("amt.tempo.manual", "手动设置 BPM")}</option>
                      </select>
                    </div>

                    {tempoMode === "fixed_manual" && (
                      <div className="amt-field">
                        <label>{t("amt.manualBpm", "BPM")}</label>
                        <input 
                          type="number" 
                          value={customBpm} 
                          onChange={(e) => setCustomBpm(parseInt(e.target.value))} 
                          min={20} 
                          max={300}
                        />
                      </div>
                    )}

                    <div className="amt-field">
                      <label>{t("amt.quantization", "量化音符 (全部轨道)")}</label>
                      <select
                        value={quantizeGrid}
                        onChange={(e) => setQuantizeGrid(e.target.value)}
                        title={t18({
                          zh: "不量化 = 保留原曲节奏（默认，转换整首歌）；量化 = 把音符对齐到节拍网格，节奏更规整",
                          en: "Off = keep the original rhythm (default, whole song); Quantize = snap notes to the beat grid for a tidier rhythm",
                          ja: "オフ = 元のリズムを保持（デフォルト・曲全体）；量子化 = 音符を拍グリッドに揃えてリズムを整えます",
                        }, lang)}
                      >
                        <option value="none">{t("amt.quantize.none", "不量化")}</option>
                        <option value="1/4">1/4</option>
                        <option value="1/8">1/8</option>
                        <option value="1/16">1/16</option>
                        <option value="1/32">1/32</option>
                        <option value="1/64">1/64</option>
                      </select>
                    </div>

                    <div className="amt-field amt-field-hint full-width">
                      <span className="amt-hint-text">
                        {t18({
                          zh: "量化 = 把音符对齐到节拍网格，节奏更规整；不量化 = 保留原曲节奏（默认，转换整首歌）",
                          en: "Quantize snaps notes to the beat grid for a tidier rhythm; Off keeps the original timing (default, whole song)",
                          ja: "量子化は音符を拍グリッドに揃えます。オフなら元のリズムのまま（デフォルト・曲全体）",
                        }, lang)}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="amt-config-section">
                <div className="amt-section-title">
                  <span className="icon">💻</span> {t("amt.section.compute", "计算与硬件")}
                </div>
                <div className="amt-config-grid-v3">
                  <div className="amt-field device-info">
                    <label>{t("amt.computeDevice", "计算设备")}</label>
                    <div className="device-name">
                      {hardwareInfo ? (
                        <>
                          <span className="dot online"></span>
                          {hardwareInfo.gpu_name || "CPU"} ({hardwareInfo.ort_build})
                        </>
                      ) : "..."}
                    </div>
                  </div>
                </div>
              </div>

                  <div className="amt-settings-grid">
                      <div className="setting-item checkbox">
                        <label>
                          <input 
                            type="checkbox" 
                            checked={useGpu} 
                            onChange={(e) => setUseGpu(e.target.checked)}
                          />
                          <span>{t("amt.useGpu", "使用 GPU 加速")}</span>
                        </label>
                      </div>
                      <div className="setting-item checkbox">
                        <label>
                          <input 
                            type="checkbox" 
                            checked={removeDuplicates} 
                            onChange={(e) => setRemoveDuplicates(e.target.checked)}
                          />
                          <span>{t("amt.removeDuplicates", "移除重复音符")}</span>
                        </label>
                      </div>
                      <div className="setting-item checkbox">
                        <label>
                          <input 
                            type="checkbox" 
                            checked={velocitySmoothing} 
                            onChange={(e) => setVelocitySmoothing(e.target.checked)}
                          />
                          <span>{t("amt.velocitySmoothing", "力度平滑")}</span>
                        </label>
                      </div>
                    </div>

                    <div className="amt-advanced-grid-v2">
                      <div className="amt-field">
                        <label>{t("amt.maxPolyphony", "最大复音数")}</label>
                        <input type="number" value={maxPolyphony} onChange={e => setMaxPolyphony(parseInt(e.target.value))} min={1} max={256} />
                      </div>
                      <div className="amt-field">
                        <label>{t("amt.ticksPerBeat", "每拍 Tick 数 (分辨率)")}</label>
                        <input type="number" value={ticksPerBeat} onChange={e => setTicksPerBeat(parseInt(e.target.value))} min={24} max={960} />
                      </div>
                      <div className="amt-field">
                        <label>{t("amt.defaultVelocity", "默认力度")}</label>
                        <input type="number" value={defaultVelocity} onChange={e => setDefaultVelocity(parseInt(e.target.value))} min={1} max={127} />
                      </div>
                    </div>
                  </div>
                )}
              </div>

            </>
          ) : (
            <div className="amt-success-workbench">
              <div className="success-icon">🎉</div>
              <h3>{t("amt.conversionFinished") || "转换已完成！"}</h3>
              <p className="desc">{t("amt.resultDesc", { count: result.midi_paths.length })}</p>
              
              <div className="amt-result-list-v2">
                {result.midi_paths.map((path, idx) => (
                  <div key={idx} className="amt-result-card">
                    <span className="file-icon">📄</span>
                    <span className="file-name">{path.split(/[/\\]/).pop()}</span>
                  </div>
                ))}
              </div>

              <div className="amt-workbench-actions">
                <button className="wb-btn primary" onClick={handleImportToTracks}>
                  <span className="icon">📥</span> {t("amt.importToTracks")}
                </button>
                <button className="wb-btn" onClick={handleExportToFolder}>
                  <span className="icon">📦</span> {t("amt.exportAll")}
                </button>
                <button className="wb-btn secondary" onClick={() => setResult(null)}>
                  {t("amt.back")}
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

