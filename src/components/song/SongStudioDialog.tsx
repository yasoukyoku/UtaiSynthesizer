/**
 * Song Studio (歌曲制作) - 左右横屏布局 SUNO 风格
 * 
 * 布局:
 *   左侧(50%): 横向模型选择 + 歌词(40%高度) + 提示词(4-5行) + 紧凑多列参数 + 生成按钮
 *   右侧(50%): 歌曲列表 + 选中歌曲详情
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { invoke, convertFileSrc } from "@tauri-apps/api/core";
import { open as openUrl } from "@tauri-apps/plugin-shell";
import { useProjectStore } from "../../store/project";
import { useAppStore, type PendingSongTask } from "../../store/app";
import { useAudioStore } from "../../store/audio";
import { GenerationCompleteDialog } from "./GenerationCompleteDialog";
import {
  loadSongHistory,
  saveSongHistory,
  listSongModels,
  abcToMidi,
  type SongOutput,
} from "../../lib/backendSong";
import {
  SONG_MODEL_CATALOG,
  getSongInstallStatus,
  songModelById,
  type SongModelFamily,
} from "../../lib/models/song-catalog";
import { getDefaultParams } from "../../lib/models/song-params";
import {
  SONG_TASKS,
  SONG_TASK_ORDER,
  ACE_TRACK_CLASSES,
  type SongTaskId,
} from "../../lib/models/song-tasks";
import {
  deriveCapabilities,
  type SongHistoryCapabilities,
} from "../../lib/song/types";
import {
  type SongTaskPayload,
} from "../../lib/song/runSongTask";
import {
  enqueueSongTask,
  useSongTaskQueue,
  MAX_WAITING as MAX_QUEUE_WAITING,
} from "../../store/song-task-queue";
import { open as openDialog, save as saveDialog } from "@tauri-apps/plugin-dialog";
import { readTextFile } from "@tauri-apps/plugin-fs";
import { ContextMenu, type MenuItem } from "../common/ContextMenu";
import i18n from "../../i18n";
import { useAmtStore } from "../../store/amt";
import { DynamicParamsPanel } from "./DynamicParamsPanel";
import { ParamsTutorialDialog } from "./ParamsTutorialDialog";
import { SendToTrackDialog, type SendFormat, type SendDestination } from "./SendToTrackDialog";
import { MultiTrackStudio, type MultiTrackResult } from "./MultiTrackStudio";
import { CreativeAssistant, type CreativeTab } from "./CreativeAssistant";
import { MidiEditor, type MidiEditNote } from "./MidiEditor";
import { OriginalityPanel } from "./OriginalityPanel";
import type { Track, SegmentContent } from "../../types/project";
import "./SongStudioDialog.css";

export interface SongStudioDialogProps {
  onClose: () => void;
}

/** 按当前语言取 SONG_TASKS 三语字段（zh/en/ja），缺省回退中文 */
function tn(map: { zh: string; en: string; ja: string }): string {
  const lang = i18n.language as keyof typeof map;
  return map[lang] ?? map.zh;
}

export interface SongGenParams {
  songName: string;
  lyrics: string;
  prompt: string;
  negative_prompt: string;
  bpm: number;
  time_signature: string;
  language: string;
  audio_duration: number;
  want_stems: boolean;
  want_midi: boolean;
  want_lrc: boolean;
  service_url: string;
  vocal_gender?: string;
  vocal_type?: string;
  vocal_range?: string;
  cot?: string;
  checkpoint?: string;
  guidance_scale?: number;
  shift?: number;
  num_inference_steps?: number;
  cfg_scale?: number;
  seed?: number;
  format?: string;
}

export interface SongHistoryEntry {
  uuid: string;
  modelId: string;
  modelFamily: SongModelFamily;
  license: string | "unknown";
  settings: SongGenParams;
  outputs: SongOutput[];
  trackId?: string;
  timestamp: number;
  label?: string;
  /** 本条目由哪个任务模式产生（缺省视为 generate，兼容旧数据） */
  task?: SongTaskId;
  /** 产物能力描述（发送轨道对话框按此显示可用动作） */
  capabilities?: SongHistoryCapabilities;
  /** 失败原因（保留生成中失败占位时使用） */
  error?: string;
  /** 来源链（从哪个结果/轨道/文件再创作而来），与 lib/song/types 同构 */
  source?: { kind: "history" | "track" | "file"; ref: string; label: string; range?: [number, number] };
}

const STYLE_TAGS = [
  { label: "流行", value: "pop" },
  { label: "摇滚", value: "rock" },
  { label: "电子", value: "electronic" },
  { label: "民谣", value: "folk" },
  { label: "爵士", value: "jazz" },
  { label: "说唱", value: "rap" },
  { label: "古风", value: "traditional" },
  { label: "轻音乐", value: "light" },
];

function formatTime(ts: number): string {
  const d = new Date(ts);
  const pad = (n: number) => n.toString().padStart(2, "0");
  return `${d.getFullYear()}/${pad(d.getMonth() + 1)}/${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

const DEFAULT_PARAMS: Omit<SongGenParams, "service_url"> = {
  songName: "",
  lyrics: "",
  prompt: "",
  negative_prompt: "",
  bpm: 120,
  time_signature: "4/4",
  language: "zh",
  audio_duration: 180,
  want_stems: false,
  want_midi: false,
  want_lrc: false,
  vocal_gender: "auto",
  vocal_type: "soprano",
  cot: "full",
  guidance_scale: 7,
  shift: 0,
  cfg_scale: 7,
  num_inference_steps: 50,
  seed: Math.floor(Math.random() * 2147483647) + 1,
  format: "wav",
};

// 一键测试所有模型使用的固定参数

/**
 * 根据vocal_gender参数在prompt中添加性别描述
 * YuE2模型不支持独立的gender参数，需要在style/prompt中描述
 */
function buildPromptWithGender(
  prompt: string | undefined,
  vocalGender: string | undefined,
  modelFamily: SongModelFamily
): string | undefined {
  if (!prompt) return undefined;

  if (modelFamily !== "yue2") return prompt;

  if (!vocalGender || vocalGender === "auto") return prompt;

  const genderDesc = vocalGender === "male" ? "male vocal" : "female vocal";

  if (
    prompt.toLowerCase().includes("vocal") ||
    prompt.toLowerCase().includes("voice") ||
    prompt.toLowerCase().includes("男") ||
    prompt.toLowerCase().includes("女")
  ) {
    return prompt;
  }

  return `${prompt}, ${genderDesc}`.trim();
}

// 根据模型家族返回对应的默认服务 URL
function getDefaultServiceUrl(family: SongModelFamily): string {
  switch (family) {
    case "yue2": return "http://127.0.0.1:8080";
    case "heartmula": return "http://127.0.0.1:7861";
    case "acestep": return "http://127.0.0.1:7861";
    default: return "http://127.0.0.1:7861";
  }
}

function randomSeed() {
  return Math.floor(Math.random() * 2147483647) + 1;
}

export function SongStudioDialog({ onClose }: SongStudioDialogProps) {
  const [modelFamily, setModelFamily] = useState<SongModelFamily>("yue2");
  const [selectedModel, setSelectedModel] = useState<string>("yue2-3b");
  const [commonParams, setCommonParams] = useState({ seed: randomSeed() });
  const [modelParams, setModelParams] = useState<Record<SongModelFamily, Record<string, any>>>(() => ({
    yue2: getDefaultParams("yue2"),
    acestep: getDefaultParams("acestep"),
    heartmula: getDefaultParams("heartmula"),
  }));

  const dynamicParams = modelParams[modelFamily];
  const updateModelParam = (key: string, value: any) => {
    setModelParams((prev) => ({
      ...prev,
      [modelFamily]: { ...prev[modelFamily], [key]: value },
    }));
  };
  // 从 localStorage 加载保存的歌词、提示词、歌名
  const loadSavedFormData = () => {
    try {
      const saved = localStorage.getItem("song_form_data");
      if (saved) {
        const data = JSON.parse(saved);
        return {
          songName: data.songName || "",
          lyrics: data.lyrics || "",
          prompt: data.prompt || "",
        };
      }
    } catch (e) {
      console.warn("Failed to load saved form data:", e);
    }
    return { songName: "", lyrics: "", prompt: "" };
  };

  const savedData = loadSavedFormData();
  const [params, setParams] = useState<SongGenParams>({
    ...DEFAULT_PARAMS,
    songName: savedData.songName,
    lyrics: savedData.lyrics,
    prompt: savedData.prompt,
    service_url: getDefaultServiceUrl("yue2"),
  });
  
  const generating = useSongTaskQueue((s) => !!s.current);
  const taskProgress = useSongTaskQueue((s) => s.progress);
  const taskError = useSongTaskQueue((s) => s.taskError);
  const queueWaiting = useSongTaskQueue((s) => s.waiting);
  const generatingEntryId = useSongTaskQueue((s) => s.generatingEntryId);
  const setTaskError = useSongTaskQueue((s) => s.setTaskError);
  const setGeneratingEntryId = useSongTaskQueue((s) => s.setGeneratingEntryId);
  
  const [history, setHistory] = useState<SongHistoryEntry[]>([]);
  // 多首歌排队时，异步回调里必须读到最新历史，不能用闭包快照
  const historyRef = useRef<SongHistoryEntry[]>([]);
  historyRef.current = history;
  const [historyLoading, setHistoryLoading] = useState(false);
  const [selectedSong, setSelectedSong] = useState<SongHistoryEntry | null>(null);
  const [contextMenuSong, setContextMenuSong] = useState<{ song: SongHistoryEntry; x: number; y: number } | null>(null);
  const [playingAudio, setPlayingAudio] = useState<HTMLAudioElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [showParamsTutorial, setShowParamsTutorial] = useState(false);
  const [showSendDialog, setShowSendDialog] = useState<SongHistoryEntry | null>(null);
  const [showMultiTrack, setShowMultiTrack] = useState(false);
  const [multiTrackSrc, setMultiTrackSrc] = useState<string | undefined>(undefined);
  // 功能 B/C/D：创作助手（简化入口）与功能 A：MIDI 编辑器
  const [showCreative, setShowCreative] = useState(false);
  const [creativeSrc, setCreativeSrc] = useState<string | undefined>(undefined);
  const [creativeTab, setCreativeTab] = useState<CreativeTab>("instrument");
  const [completion, setCompletion] = useState<{
    songName: string;
    modelId: string;
    processingTime?: number;
  } | null>(null);
  const [midiEditor, setMidiEditor] = useState<{ path?: string; label?: string } | null>(null);
  const [showAdvancedParams, setShowAdvancedParams] = useState(false);

  // ── 任务模式状态（规划 6.x / 5.3）────────────────────────────
  const [activeTask, setActiveTask] = useState<SongTaskId>("generate");
  // 模式共享表单：切换模式时隐藏字段保留，避免丢数据（规划 6.9）
  const [modeForm, setModeForm] = useState({
    sourceKind: "history" as "history" | "track" | "file",
    sourcePath: "",
    sourceLabel: "",
    srcDurationSec: 0,
    coverStrength: 0.5,
    repaintStart: 0,
    repaintEnd: 30,
    trackClasses: [] as string[],
    legoTrack: "vocals",
    extractEngine: "ace" as "ace" | "demucs",
    sheetAbc: "",
  });
  const lyricsAreaRef = useRef<HTMLTextAreaElement | null>(null);
  const patchModeForm = (patch: Partial<typeof modeForm>) =>
    setModeForm((f) => ({ ...f, ...patch }));

  const closeRequested = useRef(false);
  const addTrack = useProjectStore((s) => s.addTrack);
  const tracks = useProjectStore((s) => s.tracks);
  const setAmtRun = useAmtStore((s) => s.setRun);
  const openAmtWorkbench = useAmtStore((s) => s.openWorkbench);

  function translateStage(stage: string): string {
    const translations: Record<string, string> = {
      'init': '初始化',
      'initialize': '初始化',
      'prepare': '准备中',
      'plan': '规划中',
      'loading': '加载中',
      'processing': '处理中',
      'semantic': '语义分析',
      'synthesize': '合成中',
      'generate': '生成中',
      'generating': '生成中',
      'decode': '解码中',
      'composing': '作曲中',
      'singing': '演唱中',
      'mixing': '混音中',
      'rendering': '渲染中',
      'save': '保存中',
      'midi': '生成MIDI',
      'stems': '分轨处理',
      'finalizing': '完成中',
      'done': '完成',
      'complete': '完成',
      'extract': '分轨中',
    };
    return translations[stage] || stage;
  }

  // 当前任务定义与生效模型（所选模型不在任务支持列表时自动跳第一个）
  const taskDef = SONG_TASKS[activeTask];
  const effectiveModel =
    taskDef.models.includes(selectedModel) ? selectedModel : taskDef.models[0] ?? selectedModel;
  const needsSource = taskDef.input.srcAudio !== "none";

  // 结果库中带音频的条目（作为翻唱/重绘等模式的源）
  const audioHistory = history.filter((h) => h.outputs.some((o) => o.audio_path));

  // 历史歌曲 → 工作室/创作助手共用的数据模型（含原曲歌词供翻唱预填）
  const studioSongs = useMemo(
    () =>
      history.map((h) => ({
        uuid: h.uuid,
        label: h.label,
        audioPath: h.outputs.find((o) => o.audio_path)?.audio_path,
        abcPath: h.outputs.find((o) => o.abc_path)?.abc_path,
        durationSec: h.settings.audio_duration,
        bpm: h.settings.bpm,
        styleHint: h.settings.prompt,
        modelId: h.modelId,
        lyrics: h.settings.lyrics,
      })),
    [history],
  );

  // 工程中的音频轨道片段（作为源）
  const trackSources = tracks
    .filter((t) => t.trackType === "audio" && !t.isFolder)
    .flatMap((t) =>
      t.segments
        .map((seg) => (seg.content.type === "audioClip" ? seg.content : null))
        .filter((c): c is Extract<SegmentContent, { type: "audioClip" }> => c !== null)
        .map((c) => ({
          path: c.sourcePath,
          durationSec: (c.totalDurationMs - c.offsetMs) / 1000,
          label: `${t.name}`,
        })),
    );

  function switchTask(id: SongTaskId) {
    // 载荷在入队那一刻已快照，生成中改表单不影响在跑的任务 → 允许边生成边配置下一首
    setActiveTask(id);
    setTaskError(null);
    const def = SONG_TASKS[id];
    if (!def.models.includes(selectedModel)) {
      const nextModel = def.models[0] ?? "acestep-v1.5";
      const nextFamily = nextModel.startsWith("yue2") ? "yue2" : "acestep";
      setModelFamily(nextFamily as SongModelFamily);
      setSelectedModel(nextModel);
      setParams((p) => ({ ...p, service_url: getDefaultServiceUrl(nextFamily) }));
    }
  }

  // 规划 13.2：消费全局待启动任务（DAW 右键 → 带参打开）。挂载后执行一次。
  const consumedPendingRef = useRef(false);
  const pendingSongTask = useAppStore((s) => s.pendingSongTask);
  const clearPendingSongTask = useAppStore((s) => s.clearPendingSongTask);
  const [returnTarget, setReturnTarget] = useState<PendingSongTask["returnTarget"]>(undefined);
  // 规划 12.4：原创度自检面板状态
  const [originalityCheck, setOriginalityCheck] = useState<{ sourcePath: string | null; resultPath: string | null } | null>(null);
  useEffect(() => {
    if (consumedPendingRef.current || !pendingSongTask) return;
    consumedPendingRef.current = true;
    const p = pendingSongTask;
    setReturnTarget(p.returnTarget);
    switchTask(p.task);
    let pendingSourcePath: string | undefined;
    if (p.source.kind === "file") {
      pendingSourcePath = p.source.path;
      patchModeForm({ sourceKind: "file", sourcePath: p.source.path, sourceLabel: p.sourceLabel ?? p.source.path });
    } else if (p.source.kind === "track") {
      const seg = tracks
        .flatMap((t) => t.segments)
        .find((s) => s.id === (p.source as { kind: "track"; trackId: string; segmentId?: string }).segmentId);
      const clip = seg?.content.type === "audioClip" ? seg.content : null;
      if (clip) {
        pendingSourcePath = clip.sourcePath;
        patchModeForm({
          sourceKind: "track",
          sourcePath: clip.sourcePath,
          sourceLabel: p.sourceLabel ?? "工程片段",
          srcDurationSec: (clip.totalDurationMs - clip.offsetMs) / 1000,
        });
      }
    } else if (p.source.kind === "history") {
      // history 源：从结果库找到对应条目并填入
      const entry = audioHistory.find(
        (h) => h.uuid === (p.source as { kind: "history"; songId: string }).songId,
      );
      const audioPath = entry?.outputs.find((o) => o.audio_path)?.audio_path;
      if (entry && audioPath) {
        pendingSourcePath = audioPath;
        patchModeForm({
          sourceKind: "history",
          sourcePath: audioPath,
          sourceLabel: entry.label,
          srcDurationSec: entry.settings.audio_duration || 0,
        });
      }
    }
    if (p.range) patchModeForm({ repaintStart: p.range[0], repaintEnd: p.range[1] });
    if (p.trackClasses) patchModeForm({ trackClasses: p.trackClasses });
    if (p.trackClass) patchModeForm({ legoTrack: p.trackClass });
    if (p.presetPrompt) setParams((prev) => ({ ...prev, prompt: p.presetPrompt! }));
    if (p.presetLyrics) setParams((prev) => ({ ...prev, lyrics: p.presetLyrics! }));
    if (p.tool) {
      if (p.tool === "multiTrack") {
        setMultiTrackSrc(pendingSourcePath);
        setShowMultiTrack(true);
      } else if (p.tool === "creative" || p.tool === "cover") {
        setCreativeSrc(pendingSourcePath);
        setCreativeTab(p.tool === "cover" ? "cover" : "instrument");
        setShowCreative(true);
      } else {
        setMidiEditor({ label: p.sourceLabel });
      }
    }
    clearPendingSongTask();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingSongTask]);

  // 自动保存歌词、提示词、歌名到 localStorage
  useEffect(() => {
    const timer = setTimeout(() => {
      try {
        localStorage.setItem("song_form_data", JSON.stringify({
          songName: params.songName,
          lyrics: params.lyrics,
          prompt: params.prompt,
        }));
      } catch (e) {
        console.warn("Failed to save form data:", e);
      }
    }, 500); // 防抖 500ms
    return () => clearTimeout(timer);
  }, [params.songName, params.lyrics, params.prompt]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !closeRequested.current) {
        closeRequested.current = true;
        onClose();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  useEffect(() => {
    void loadHistory();
    setTaskError(null);
  }, []);

  async function loadHistory() {
    setHistoryLoading(true);
    try {
      const raw = await loadSongHistory();
      if (!raw) { setHistory([]); return; }
      const parsed = JSON.parse(raw) as SongHistoryEntry[];
      // 生成中的占位条目（outputs 为空）会落盘，以便关闭弹窗后重开仍显示进度；但
      // generatingEntryId 只在内存 store 里，应用重启/页面 reload 后归 null，占位条目
      // 就会变成永不消失、点了也放不出声的幽灵行（一次性 sidecar 也已随进程死掉）。
      // 因此这里对账：非当前任务的空 outputs 条目一律丢弃并回写。
      // 排队中的任务同样有空 outputs 占位条目，必须一并保留。
      const q = useSongTaskQueue.getState();
      const activeIds = new Set(
        [q.generatingEntryId, q.current?.entryId, ...q.waiting.map((w) => w.entryId)].filter(
          (id): id is string => !!id,
        ),
      );
      const cleaned = parsed.filter(
        (e) => (e.outputs?.length ?? 0) > 0 || activeIds.has(e.uuid),
      );
      const sorted = cleaned.sort((a, b) => b.timestamp - a.timestamp);
      setHistory(sorted);
      historyRef.current = sorted;
      if (cleaned.length !== parsed.length) void persistHistory(sorted);
    } catch {
      setHistory([]);
    } finally {
      setHistoryLoading(false);
    }
  }

  async function persistHistory(entries: SongHistoryEntry[]) {
    try {
      await saveSongHistory(entries);
    } catch {}
  }

  // 队列允许多首歌同时排队，任何一首结束时都必须基于"最新"历史做增量更新；
  // 直接用闭包里捕获的旧快照回写，会把其他任务刚写入的结果整条抹掉。
  function mutateHistory(
    updater: (prev: SongHistoryEntry[]) => SongHistoryEntry[],
  ): SongHistoryEntry[] {
    const next = updater(historyRef.current);
    historyRef.current = next;
    setHistory(next);
    void persistHistory(next);
    return next;
  }

  function depositToTrack(
    audioPath: string,
    label: string,
    durationMs: number = 60000,
    startTick: number = 0,
    insertIndex?: number,
  ): string {
    const proj = useProjectStore.getState();
    const bpm = proj.tempo ?? params.bpm ?? 120;
    const ticksPerBeat = 480;
    const durationTicks = Math.max(1, Math.round((durationMs / 60000) * bpm * ticksPerBeat));
    const trackId = crypto.randomUUID();
    const segmentId = crypto.randomUUID();

    const track: Track = {
      id: trackId,
      name: `🎧 ${label}`,
      trackType: "audio",
      segments: [
        {
          id: segmentId,
          startTick,
          durationTicks,
          content: {
            type: "audioClip",
            sourcePath: audioPath,
            offsetMs: 0,
            totalDurationMs: durationMs,
          },
        },
      ],
      volumeDb: 0,
      pan: 0,
      muted: false,
      solo: false,
      expanded: true,
      laneControls: {},
      laneMutes: {},
      autoArrangeNodeId: undefined,
    };
    addTrack(track, insertIndex);
    return trackId;
  }

  // 规划 8.2：落点选项 → (startTick, insertIndex) 换算
  function resolveDest(dest: SendDestination): { startTick: number; insertIndex?: number } {
    const proj = useProjectStore.getState();
    if (dest.kind === "playhead") {
      return { startTick: Math.max(0, Math.round(proj.playheadTick)) };
    }
    if (dest.kind === "afterTrack") {
      const idx = proj.tracks.findIndex((t) => t.id === dest.trackId);
      return idx >= 0 ? { startTick: 0, insertIndex: idx + 1 } : { startTick: 0 };
    }
    return { startTick: 0 };
  }

  // 规划 8.2：MIDI 直接落轨——import_score_file 解析后建 instrument 音符轨（最佳努力）
  async function depositMidiAsNotesTrack(
    midiPath: string,
    label: string,
    startTick: number = 0,
    insertIndex?: number,
  ) {
    try {
      const score = await invoke<any>("import_score_file", { path: midiPath });
      const parts = (score?.tracks ?? []).filter(
        (t: any) => Array.isArray(t.notes) && t.notes.length > 0,
      );
      if (parts.length === 0) {
        alert("MIDI 中没有可用的音符");
        return;
      }
      let maxTick = 0;
      const notes = parts.flatMap((it: any) =>
        it.notes.map((n: any) => {
          maxTick = Math.max(maxTick, n.tick + n.duration);
          return {
            id: crypto.randomUUID(),
            tick: n.tick,
            duration: n.duration,
            pitch: n.pitch,
            lyric: n.lyric || "La",
            velocity: n.velocity ?? 100,
          };
        }),
      );
      useProjectStore.getState().addTrack(
        {
          id: crypto.randomUUID(),
          name: `🎹 ${label}`,
          trackType: "instrument",
          volumeDb: 0,
          pan: 0,
          muted: false,
          solo: false,
          expanded: false,
          laneControls: {},
          segments: [
            {
              id: crypto.randomUUID(),
              startTick,
              durationTicks: Math.max(1, maxTick),
              content: { type: "notes", notes },
            },
          ],
        } as Track,
        insertIndex,
      );
      alert(`已添加 MIDI 音符轨: ${label}`);
    } catch (err) {
      console.error("midi deposit failed:", err);
      alert(`MIDI 落轨失败: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  // 规划 10.5：结果回流——把音频追加到既有轨道的指定 tick（默认对齐源片段起点）
  function depositToExistingTrack(
    trackId: string,
    audioPath: string,
    _label: string,
    durationMs: number,
    startTick: number = 0,
  ) {
    const proj = useProjectStore.getState();
    const bpm = proj.tempo ?? params.bpm ?? 120;
    const durationTicks = Math.max(1, Math.round((durationMs / 60000) * bpm * 480));
    const target = proj.tracks.find((t) => t.id === trackId);
    if (!target) return;
    const seg = {
      id: crypto.randomUUID(),
      startTick,
      durationTicks,
      content: {
        type: "audioClip",
        sourcePath: audioPath,
        offsetMs: 0,
        totalDurationMs: durationMs,
      } as SegmentContent,
    };
    useProjectStore.getState().updateTrack(trackId, {
      segments: [...target.segments, seg],
    });
  }

  useEffect(() => {
    return () => {
      if (playingAudio) {
        playingAudio.pause();
        playingAudio.src = "";
      }
    };
  }, [playingAudio]);

  function handlePlaySong(song: SongHistoryEntry) {
    console.log("[handlePlaySong] 开始播放，song.outputs:", song.outputs);
    const primaryAudio = song.outputs.find(o => o.audio_path);
    console.log("[handlePlaySong] primaryAudio:", primaryAudio);
    
    if (!primaryAudio?.audio_path) {
      console.error("[handlePlaySong] 没有找到 audio_path");
      alert("没有可播放的音频文件");
      return;
    }

    // 如果是同一首歌，切换播放/暂停
    if (playingAudio && selectedSong?.uuid === song.uuid) {
      togglePlayPause();
      return;
    }

    const originalPath = primaryAudio.audio_path;
    const convertedPath = convertFileSrc(originalPath);
    console.log("[handlePlaySong] 原始路径:", originalPath);
    console.log("[handlePlaySong] 转换后路径:", convertedPath);

    if (playingAudio) {
      playingAudio.pause();
      playingAudio.src = "";
    }

    const audio = new Audio(convertedPath);
    
    audio.addEventListener("loadedmetadata", () => {
      console.log("[handlePlaySong] 音频加载成功, duration:", audio.duration);
      setDuration(audio.duration);
    });

    audio.addEventListener("play", () => {
      setIsPlaying(true);
    });

    audio.addEventListener("pause", () => {
      setIsPlaying(false);
    });

    audio.addEventListener("timeupdate", () => {
      setCurrentTime(audio.currentTime);
    });

    audio.addEventListener("ended", () => {
      setCurrentTime(0);
      setIsPlaying(false);
    });

    audio.addEventListener("error", (e) => {
      console.error("[handlePlaySong] Audio 元素错误:", e);
      console.error("[handlePlaySong] Audio error code:", audio.error?.code);
      console.error("[handlePlaySong] Audio error message:", audio.error?.message);
    });

    setPlayingAudio(audio);
    setSelectedSong(song);

    audio.play().catch(err => {
      console.error("[handlePlaySong] 播放失败:", err);
      console.error("[handlePlaySong] 错误详情:", JSON.stringify(err, null, 2));
      alert(`播放失败: ${err.message || "未知错误"}`);
    });
  }

  function togglePlayPause() {
    if (!playingAudio) return;
    
    if (playingAudio.paused) {
      playingAudio.play().catch(err => {
        console.error("播放失败:", err);
      });
    } else {
      playingAudio.pause();
    }
  }

  function formatDuration(seconds: number): string {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  }

  function handlePreviousSong() {
    if (!selectedSong) return;
    const currentIndex = history.findIndex(s => s.uuid === selectedSong.uuid);
    if (currentIndex > 0) {
      const prevSong = history[currentIndex - 1];
      if (prevSong) handlePlaySong(prevSong);
    }
  }

  function handleNextSong() {
    if (!selectedSong) return;
    const currentIndex = history.findIndex(s => s.uuid === selectedSong.uuid);
    if (currentIndex < history.length - 1) {
      const nextSong = history[currentIndex + 1];
      if (nextSong) handlePlaySong(nextSong);
    }
  }

  function handleSeek(newTime: number) {
    if (playingAudio) {
      playingAudio.currentTime = newTime;
      setCurrentTime(newTime);
    }
  }



  async function handleDownloadSong(song: SongHistoryEntry) {
    const primaryAudio = song.outputs.find(o => o.audio_path);
    if (!primaryAudio?.audio_path) {
      alert("没有可下载的音频文件");
      return;
    }
    
    try {
      const link = document.createElement("a");
      link.href = `file:///${primaryAudio.audio_path.replace(/\\/g, "/")}`;
      link.download = `${song.label || "song"}.wav`;
      link.click();
    } catch (err) {
      console.error("下载失败:", err);
      alert("下载失败，请检查文件路径");
    }
  }

  // 规划 12.6：ZIP 发行包（01_母带/02_分轨/03_工程/04_资料 + README.txt）
  async function handleExportReleasePack(song: SongHistoryEntry) {
    const t = (k: string, opts?: Record<string, unknown>) => i18n.t(k, opts);
    const safeLabel = (song.label || "song").replace(/[\\/:*?"<>|]/g, "_");
    const extOf = (p: string) => (p.lastIndexOf(".") >= 0 ? p.slice(p.lastIndexOf(".")) : "");
    const entries: { path: string; name: string }[] = [];

    const master = song.outputs.find((o) => o.audio_path)?.audio_path;
    if (master) entries.push({ path: master, name: `01_母带/${safeLabel}${extOf(master)}` });
    for (const o of song.outputs) {
      for (const [k, p] of Object.entries(o.stems ?? {})) {
        if (p) entries.push({ path: p, name: `02_分轨/${safeLabel} - ${k}${extOf(p)}` });
      }
      if (o.midi_path) entries.push({ path: o.midi_path, name: `03_工程/${safeLabel}.mid` });
      if (o.abc_path) entries.push({ path: o.abc_path, name: `03_工程/${safeLabel}.abc` });
      if (o.lrc_path) entries.push({ path: o.lrc_path, name: `04_资料/${safeLabel}.lrc` });
    }
    if (entries.length === 0) {
      alert(t("songPack.noFiles"));
      return;
    }

    try {
      const outPath = await saveDialog({
        title: t("songMenu.exportPack"),
        defaultPath: `${safeLabel}_${t("songPack.suffix")}.zip`,
        filters: [{ name: "ZIP", extensions: ["zip"] }],
      });
      if (!outPath) return;

      // README.txt 写进歌曲产物目录（避免污染导出位置），随包打包
      const anchor = entries[0]!.path;
      const readmePath = anchor.replace(/[/\\][^/\\]+$/, "") + `/pack_${Date.now()}_README.txt`;
      const { writeTextFile } = await import("@tauri-apps/plugin-fs");
      await writeTextFile(
        readmePath,
        t("songPack.readme", {
          label: song.label || safeLabel,
          model: `${song.modelFamily} (${song.modelId})`,
          date: new Date().toISOString().slice(0, 10),
        }),
      );
      entries.push({ path: readmePath, name: "README.txt" });

      await invoke("zip_files", { entries, outPath });
      alert(t("songPack.done", { path: outPath }));
    } catch (err) {
      console.error("export pack failed:", err);
      alert(t("songPack.failed", { error: err instanceof Error ? err.message : String(err) }));
    }
  }

  // 规划 8.1/8.2：产物驱动发送（single/multi/liveExtract/midi/midiTrack + 落点）
  async function handleAddToTrack(
    song: SongHistoryEntry,
    format: SendFormat = "single",
    extractEngine: "ace" | "demucs" = "demucs",
    dest: SendDestination = { kind: "end" },
  ) {
    const primaryAudio = song.outputs.find(o => o.audio_path);
    if (!primaryAudio?.audio_path) {
      alert("没有可添加的音频文件");
      return;
    }
    const label = song.label || "song";
    const durationMs = Math.round(song.settings.audio_duration * 1000);
    const { startTick, insertIndex } = resolveDest(dest);

    // 多轨：只看实际 stems 产物（8.1）
    if (format === "multi") {
      const stems = Object.entries(primaryAudio.stems ?? {});
      if (stems.length > 0) {
        stems.forEach(([name, path], k) => {
          depositToTrack(
            path,
            `${label} - ${name}`,
            durationMs,
            startTick,
            insertIndex === undefined ? undefined : insertIndex + k,
          );
        });
        alert(`已添加 ${stems.length} 个分轨到轨道`);
        return;
      }
    }

    // 现场分轨兜底（8.1）：demucs 快速四轨 / ACE 全轨
    if (format === "liveExtract") {
      try {
        const result = await enqueueSongTask({
          task: extractEngine === "demucs" ? "stems" : "extract",
          label,
          payload: {
            songName: label,
            model: effectiveModel,
            srcAudioPath: primaryAudio.audio_path,
            srcDurationSec: song.settings.audio_duration,
            trackClasses: extractEngine === "ace" ? [...ACE_TRACK_CLASSES] : undefined,
          },
        });
        const stemOut = result.outputs.find((o) => o.stems);
        const stems = Object.entries(stemOut?.stems ?? {});
        if (stems.length === 0) throw new Error("分轨未产出任何音轨");
        stems.forEach(([name, path], k) => {
          depositToTrack(
            path,
            `${label} - ${name}`,
            durationMs,
            startTick,
            insertIndex === undefined ? undefined : insertIndex + k,
          );
        });
        // 分轨产物写回历史条目，下次可直接"多轨发送"
        const nextOutputs = stemOut ? [...song.outputs, stemOut] : song.outputs;
        const updated = history.map(h =>
          h.uuid === song.uuid
            ? { ...h, outputs: nextOutputs, capabilities: deriveCapabilities({ outputs: nextOutputs }) }
            : h,
        );
        setHistory(updated);
        void persistHistory(updated);
        setSelectedSong(updated.find(h => h.uuid === song.uuid) ?? null);
        alert(`已分轨并添加 ${stems.length} 个分轨到轨道`);
      } catch (err) {
        console.error("live extract failed:", err);
        alert(`现场分轨失败: ${err instanceof Error ? err.message : String(err)}`);
      }
      return;
    }

    // MIDI：midi_path 优先，仅 abc 时先转换（8.1）
    if (format === "midi" || format === "midiTrack") {
      let midiPath = song.outputs.find(o => o.midi_path)?.midi_path;
      if (!midiPath) {
        const abcOutput = song.outputs.find(o => o.abc_path);
        if (abcOutput?.abc_path) {
          try {
            const outPath = abcOutput.abc_path.replace(/\.[^.]+$/, "") + ".mid";
            midiPath = await abcToMidi(abcOutput.abc_path, outPath);
          } catch (err) {
            alert(`ABC 转 MIDI 失败: ${err instanceof Error ? err.message : String(err)}`);
            return;
          }
        }
      }
      if (midiPath) {
        // 规划 8.2 midiTrack：直接落为可编辑的 MIDI 音符轨（最佳努力，不阻断）
        if (format === "midiTrack") {
          await depositMidiAsNotesTrack(midiPath, `${label} (MIDI)`, startTick, insertIndex);
          return;
        }
        // 在 MIDI 工作台中打开生成的 MIDI 文件（可查看/编辑元数据）
        const nodeId = `song-midi-${song.uuid}`;
        setAmtRun({
          nodeId,
          audioPath: primaryAudio.audio_path,
          midiPath,
          mode: "smart",
          backend: "yourmt3",
          totalNotes: null,
          processingTimeSecs: 0,
        });
        openAmtWorkbench(nodeId);
        alert("已打开 MIDI 工作台，关闭本窗口即可查看 MIDI 内容");
        return;
      }
    }

    // 默认单轨添加
    const trackId = depositToTrack(
      primaryAudio.audio_path,
      label,
      durationMs,
      startTick,
      insertIndex,
    );
    
    const updated = history.map(h => 
      h.uuid === song.uuid ? { ...h, trackId } : h
    );
    setHistory(updated);
    void persistHistory(updated);
    
    setSelectedSong({ ...song, trackId });
    alert(`已添加到轨道: ${label}`);
  }

  function handleLocateTrack(_trackId: string) {
    onClose();
  }

  // 删除歌曲
  function handleDeleteSong(song: SongHistoryEntry) {
    if (!confirm(`确定要删除歌曲「${song.label || "Untitled"}」吗？`)) {
      return;
    }
    
    const next = history.filter(h => h.uuid !== song.uuid);
    setHistory(next);
    void persistHistory(next);
    
    if (selectedSong?.uuid === song.uuid) {
      setSelectedSong(null);
    }
    
    setContextMenuSong(null);
  }

  // 关闭右键菜单（点击其他地方时）
  useEffect(() => {
    const handleClick = () => setContextMenuSong(null);
    if (contextMenuSong) {
      window.addEventListener("click", handleClick);
      return () => window.removeEventListener("click", handleClick);
    }
  }, [contextMenuSong]);


  async function handleGenerate() {
    // 生成中允许继续提交：只有等待队列满了才拦（与队列上限一致）
    if (useSongTaskQueue.getState().waiting.length >= MAX_QUEUE_WAITING) {
      setTaskError(`等待队列已满（最多 ${MAX_QUEUE_WAITING} 个任务）`);
      return;
    }
    setTaskError(null);
    const model = songModelById(effectiveModel);
    if (!model) { setTaskError("模型不存在"); return; }

    // ── 模式级校验（规划 6.1-6.9）────────────────────────────
    const mf = modeForm;
    const label = params.songName.trim() || defaultLabelFor(activeTask);
    if (needsSource && !mf.sourcePath) {
      setTaskError("请先选择源音频（结果库 / 工程轨道 / 本地文件）");
      return;
    }
    if (activeTask === "extract" && mf.extractEngine === "ace" && mf.trackClasses.length === 0) {
      setTaskError("请至少选择一个要分出的音轨");
      return;
    }
    if (activeTask === "repaint") {
      const srcDur = mf.srcDurationSec || params.audio_duration;
      if (!(mf.repaintStart >= 0 && mf.repaintEnd > mf.repaintStart && mf.repaintEnd <= srcDur)) {
        setTaskError(`重绘区间无效：需满足 0 ≤ 起点 < 终点 ≤ 源时长（${srcDur.toFixed(1)}s）`);
        return;
      }
    }
    if (activeTask === "complete" && mf.trackClasses.length === 0) {
      setTaskError("请至少选择一个要补全的音轨");
      return;
    }
    if (activeTask === "lego" && !params.prompt.trim()) {
      setTaskError("叠加模式需要填写提示词描述新元素");
      return;
    }

    const generatingUUID = crypto.randomUUID();
    // 队列空闲时这首歌马上开跑；已有任务在跑时它是"排队中"，
    // generatingEntryId 由队列在真正开跑那一刻切换，避免覆盖正在生成的那首。
    if (!useSongTaskQueue.getState().current) setGeneratingEntryId(generatingUUID);
    const generatingEntry: SongHistoryEntry = {
      uuid: generatingUUID,
      modelId: effectiveModel,
      modelFamily: model.family,
      license: model.license,
      settings: { ...params, songName: label },
      outputs: [],
      trackId: undefined,
      timestamp: Date.now(),
      label: label.slice(0, 40),
      task: activeTask,
    };
    // 立即落盘占位条目，确保关闭对话框后重新打开仍能看到生成/排队状态
    mutateHistory((prev) => [generatingEntry, ...prev]);
    setSelectedSong(generatingEntry);

    try {
      // 组装统一任务载荷（规划 6.x → runSongTask）
      // extract + demucs 引擎 → stems 快速分轨任务；其余按 activeTask 路由
      const runTaskId: SongTaskId =
        activeTask === "extract" && mf.extractEngine === "demucs" ? "stems" : activeTask;
      const payload: SongTaskPayload = {
        songName: label,
        model: effectiveModel,
        srcAudioPath: needsSource ? mf.sourcePath : undefined,
        srcDurationSec: mf.srcDurationSec || undefined,
        trackName: activeTask === "lego" ? mf.legoTrack : undefined,
        trackClasses:
          activeTask === "extract" || activeTask === "complete" ? mf.trackClasses : undefined,
        repaintStart: activeTask === "repaint" ? mf.repaintStart : undefined,
        repaintEnd: activeTask === "repaint" ? mf.repaintEnd : undefined,
        coverStrength: activeTask === "cover" ? mf.coverStrength : undefined,
        wantStems: activeTask === "extract" ? true : params.want_stems,
        wantMidi: activeTask === "generate" || activeTask === "instrumental" || activeTask === "sheet" ? params.want_midi : undefined,
        wantLrc: activeTask === "generate" || activeTask === "instrumental" ? params.want_lrc : undefined,
        durationSec: activeTask === "generate" || activeTask === "instrumental" ? params.audio_duration : undefined,
        seed: commonParams.seed === -1 || commonParams.seed == null ? undefined : Number(commonParams.seed),
        abc: activeTask === "sheet" && mf.sheetAbc.trim() ? mf.sheetAbc.trim() : undefined,
        extra: {
          lyrics:
            activeTask === "generate" || activeTask === "instrumental"
              ? params.lyrics
              : undefined,
          prompt: buildPromptWithGender(params.prompt, params.vocal_gender, model.family),
          ...dynamicParams,
        },
      };

      // 规划 13.3：后端单任务 → 队列忙时先提示"已加入队列"（进度回调在真正开跑后才来）
      const result = await enqueueSongTask({
        task: runTaskId,
        label,
        payload,
        entryId: generatingUUID,
      });

      const capabilities = deriveCapabilities({ outputs: result.outputs });
      const updatedEntry: SongHistoryEntry = {
        ...generatingEntry,
        modelId: result.modelId,
        modelFamily: result.modelId.startsWith("yue2") ? "yue2" : model.family,
        outputs: result.outputs,
        label: result.label.slice(0, 40),
        task: activeTask,
        capabilities,
      };
      // 注册音频 metadata，让播放器/波形可用
      const audioPath = result.outputs.find((o) => o.audio_path)?.audio_path;
      if (audioPath) {
        try { await useAudioStore.getState().loadAudioFile(audioPath); }
        catch (metaErr) { console.warn("[song] failed to register audio metadata:", metaErr); }
      }
      mutateHistory((prev) =>
        prev.map((h) => (h.uuid === generatingUUID ? updatedEntry : h)),
      );
      setSelectedSong((cur) =>
        !cur || cur.uuid === generatingUUID ? updatedEntry : cur,
      );

      if (audioPath) {
        setCompletion({
          songName: result.label.slice(0, 40),
          modelId: result.modelId,
          processingTime: result.outputs.find((output) => output.processing_time_secs !== undefined)?.processing_time_secs,
        });
      }
      // 规划 10.5/13.2：DAW 发起的任务带 returnTarget → 结果回流到原轨道（对齐源片段起点）
      if (returnTarget && audioPath) {
        const proj = useProjectStore.getState();
        const srcSeg = proj.tracks
          .flatMap((t) => (t.id === returnTarget.trackId ? t.segments : []))
          .find((s) => s.id === returnTarget.segmentId);
        depositToExistingTrack(
          returnTarget.trackId,
          audioPath,
          result.label.slice(0, 40),
          Math.round((mf.srcDurationSec || params.audio_duration || 60) * 1000),
          returnTarget.align ? srcSeg?.startTick ?? 0 : 0,
        );
      }
      // generatingEntryId 由队列在任务结束时自行清理／切给下一首，这里不能手动置空，
      // 否则会把队列刚切上来的下一首的"生成中"标记抹掉。
    } catch (err) {
      console.error("song task failed:", err);
      const errorMsg =
        err instanceof Error ? err.message : typeof err === "string" ? err : JSON.stringify(err);
      // 占位条目开跑时已落盘，失败后必须同步回写，否则磁盘上会留下残条目
      mutateHistory((prev) => prev.filter((h) => h.uuid !== generatingUUID));
      setSelectedSong((cur) => (cur?.uuid === generatingUUID ? null : cur));
      setTaskError(errorMsg);
    }
  }

  function defaultLabelFor(task: SongTaskId): string {
    const base = SONG_TASKS[task].name.zh;
    return `${base}_${Date.now().toString(36)}`;
  }

  // 规划 7.2：以某结果为源，切换到对应再创作模式
  function openRetaskFromSong(song: SongHistoryEntry, task: SongTaskId) {
    const audioPath = song.outputs.find((o) => o.audio_path)?.audio_path;
    setSelectedSong(song);
    setModeForm((f) => ({
      ...f,
      sourceKind: "history",
      sourcePath: audioPath ?? "",
      sourceLabel: song.label || "未命名",
      srcDurationSec: song.settings.audio_duration || 0,
    }));
    switchTask(task);
  }

  // 规划 7.2 结果卡片菜单（P0 第一组：播放/再创作/发送/下载/删除）
  function buildSongMenu(song: SongHistoryEntry): MenuItem[] {
    const t = (k: string) => i18n.t(k);
    const audioPath = song.outputs.find((o) => o.audio_path)?.audio_path;
    const retask = (task: SongTaskId, label: string, icon: string): MenuItem => ({
      label,
      icon,
      disabled: !audioPath,
      onClick: () => openRetaskFromSong(song, task),
    });
    return [
      {
        label: t("songMenu.play"),
        icon: "▶",
        disabled: !audioPath,
        onClick: () => handlePlaySong(song),
      },
      { type: "divider" },
      {
        type: "submenu",
        label: t("songMenu.remixGroup"),
        icon: "✨",
        disabled: !audioPath,
        items: [
          retask("cover", t("songMenu.remixCover"), "🎤"),
          retask("repaint", t("songMenu.remixRepaint"), "🖌"),
          retask("complete", t("songMenu.remixComplete"), "🧩"),
          retask("lego", t("songMenu.remixLego"), "🧱"),
          retask("extract", t("songMenu.remixExtract"), "✂"),
          retask("sheet", t("songMenu.remixSheet"), "🎹"),
        ],
      },
      { type: "divider" },
      {
        label: t("songResult.sendToTrack"),
        icon: "📤",
        disabled: !audioPath,
        onClick: () => {
          setSelectedSong(song);
          setShowSendDialog(song);
        },
      },
      {
        label: t("songMenu.download"),
        icon: "⬇",
        disabled: !audioPath,
        onClick: () => handleDownloadSong(song),
      },
      // 规划 12.6：ZIP 发行包
      {
        label: t("songMenu.exportPack"),
        icon: "📦",
        disabled: !audioPath,
        onClick: () => handleExportReleasePack(song),
      },
      // 规划 9.1：带源预选打开高级多轨工作室（过渡期入口）
      {
        label: t("songMenu.openInStudio"),
        icon: "🎛",
        disabled: !audioPath,
        onClick: () => {
          setSelectedSong(song);
          setMultiTrackSrc(audioPath);
          setShowMultiTrack(true);
        },
      },
      // 功能 B/C/D：创作助手（单乐器叠加 / 乐谱续写 / 翻唱助手）
      {
        label: "🪄 创作助手",
        icon: "🪄",
        disabled: !audioPath,
        onClick: () => {
          setSelectedSong(song);
          setCreativeTab("instrument");
          setShowCreative(true);
        },
      },
      // 功能 A：MIDI 编辑器
      {
        label: "🎹 MIDI 编辑",
        icon: "🎹",
        disabled: !audioPath,
        onClick: () => {
          setSelectedSong(song);
          void handleOpenMidiEditor(
            song.outputs.find((o) => o.midi_path)?.midi_path,
            song.label,
            song.outputs.find((o) => o.abc_path)?.abc_path,
          );
        },
      },
      // 规划 12.4：原创度自检（源优先取来源链里的源音频，无源则只分析结果自身）
      {
        label: t("songOriginality.menuItem"),
        icon: "🛡",
        disabled: !audioPath,
        onClick: () => {
          let src: string | null = null;
          const s = song.source;
          if (s?.kind === "file") {
            src = s.ref;
          } else if (s?.kind === "history") {
            const entry = history.find((h) => h.uuid === s.ref);
            src = entry?.outputs.find((o) => o.audio_path)?.audio_path ?? null;
          } else if (s?.kind === "track") {
            for (const tr of useProjectStore.getState().tracks) {
              for (const seg of tr.segments) {
                if (seg.id === s.ref && seg.content.type === "audioClip") {
                  src =
                    seg.processedOutputs?.filter((o) => !o.loading).slice(-1)[0]?.audioPath ??
                    seg.content.sourcePath;
                }
              }
            }
          }
          setOriginalityCheck({ sourcePath: src, resultPath: audioPath ?? null });
        },
      },
      { type: "divider" },
      {
        label: t("songMenu.delete"),
        icon: "🗑",
        danger: true,
        onClick: () => handleDeleteSong(song),
      },
    ];
  }

  // 在歌词光标处插入结构标签（规划 5.5）
  function insertLyricTag(tag: string) {
    const text = params.lyrics;
    const token = `[${tag}]`;
    const ta = lyricsAreaRef.current;
    if (!ta) {
      setParams({ ...params, lyrics: text ? `${text}\n${token}` : token });
      return;
    }
    const s = ta.selectionStart ?? text.length;
    const e = ta.selectionEnd ?? s;
    const before = text.slice(0, s);
    const after = text.slice(e);
    const needNL = before.length > 0 && !before.endsWith("\n");
    const next = `${before}${needNL ? "\n" : ""}${token}\n${after}`;
    setParams({ ...params, lyrics: next });
    const pos = before.length + (needNL ? 1 : 0) + token.length + 1;
    requestAnimationFrame(() => {
      ta.focus();
      ta.setSelectionRange(pos, pos);
    });
  }

  // 多轨工作室 / 创作助手：把生成结果保存为新的历史歌曲（共用）
  function saveStudioResult(r: MultiTrackResult) {
    const entry: SongHistoryEntry = {
      uuid: crypto.randomUUID(),
      modelId: r.modelId,
      modelFamily: r.modelId === "yue2-3b" ? "yue2" : "acestep",
      license: r.modelId === "yue2-3b" ? "unknown" : "apache-2.0",
      settings: { ...params, songName: r.label, audio_duration: r.durationSec, want_midi: r.wantMidi },
      outputs: r.outputs,
      timestamp: Date.now(),
      label: r.label.slice(0, 40),
    };
    const next = [entry, ...history];
    setHistory(next);
    setSelectedSong(entry);
    void persistHistory(next);
    const audioPath = r.outputs.find((o) => o.audio_path)?.audio_path;
    if (audioPath) {
      useAudioStore.getState().loadAudioFile(audioPath).catch((e) =>
        console.warn("[studio] failed to register audio metadata:", e));
    }
    return entry;
  }

  // 多轨工作室：把生成结果保存为新的历史歌曲
  function handleSaveMultiTrackResult(r: MultiTrackResult) {
    saveStudioResult(r);
    setShowMultiTrack(false);
  }

  // 创作助手：保存结果并关闭面板
  function handleCreativeGenerated(r: MultiTrackResult) {
    const entry = saveStudioResult(r);
    setShowCreative(false);
    alert(`已保存到歌曲列表：${entry.label}`);
  }

  // 多轨工作室 / 创作助手：把结果音频直接落到工程轨道（新轨道）
  function handleSendMultiTrackToTrack(audioPath: string, label: string, durationSec: number) {
    depositToTrack(audioPath, label, Math.round(durationSec * 1000));
    useAudioStore.getState().loadAudioFile(audioPath).catch((e) =>
      console.warn("[studio] failed to register audio metadata:", e));
    alert(`已添加为新的工程轨道：${label}`);
  }

  // 功能 A：打开 MIDI 编辑器（优先用歌曲自带 MIDI；否则用 ABC 现场转一份；再否则打开空编辑器）
  // abcPath 由调用方显式传入：右键菜单在 setSelectedSong 之后同步调用，闭包里的 selectedSong 还是旧值。
  async function handleOpenMidiEditor(midiPath?: string, label?: string, abcPath?: string) {
    if (midiPath) {
      setMidiEditor({ path: midiPath, label });
      return;
    }
    if (!abcPath) {
      setMidiEditor({ label });
      return;
    }
    try {
      const abc = await readTextFile(abcPath);
      const base = (label || "score").replace(/[\\/:*?"<>|]/g, "_");
      const out = await saveDialog({
        title: "ABC 转 MIDI",
        defaultPath: `${base}.mid`,
        filters: [{ name: "MIDI", extensions: ["mid"] }],
      });
      if (!out) return;
      const target = out.toLowerCase().endsWith(".mid") ? out : `${out}.mid`;
      await abcToMidi(abc, target);
      setMidiEditor({ path: target, label });
    } catch (e) {
      alert(`ABC 转 MIDI 失败：${e instanceof Error ? e.message : String(e)}`);
    }
  }

  // 功能 A：把编辑器里的音符落成工程的乐器轨
  function handleMidiDepositNotes(label: string, notes: MidiEditNote[]) {
    let maxTick = 0;
    const mapped = notes.map((n) => {
      maxTick = Math.max(maxTick, n.tick + n.duration);
      return {
        id: crypto.randomUUID(),
        tick: Math.round(n.tick),
        duration: Math.max(1, Math.round(n.duration)),
        pitch: n.pitch,
        lyric: n.lyric || "La",
        velocity: n.velocity ?? 100,
      };
    });
    useProjectStore.getState().addTrack(
      {
        id: crypto.randomUUID(),
        name: `🎹 ${label}`,
        trackType: "instrument",
        volumeDb: 0,
        pan: 0,
        muted: false,
        solo: false,
        expanded: false,
        laneControls: {},
        segments: [
          {
            id: crypto.randomUUID(),
            startTick: 0,
            durationTicks: Math.max(1, maxTick),
            content: { type: "notes", notes: mapped },
          },
        ],
      } as Track,
    );
    alert(`已添加 MIDI 音符轨：${label}`);
  }

  // 功能 A：把编辑器渲染出的音频落到工程（新音频轨）
  function handleMidiDepositAudio(audioPath: string, label: string, durationSec: number) {
    depositToTrack(audioPath, label, Math.round(durationSec * 1000));
    useAudioStore.getState().loadAudioFile(audioPath).catch((e) =>
      console.warn("[midi-editor] failed to register audio metadata:", e));
    alert(`已添加为新轨道：${label}`);
  }

  // 一键测试所有模型：用相同的歌词/提示词/时长为每个模型各生成一首测试歌曲


  const currentModel = songModelById(selectedModel);

  // 安装状态（下拉里标注未下载的模型；生成前还会再实时校验一次）
  const [songInstall, setSongInstall] = useState<Record<string, { installed: boolean; ready: number; required: number }>>({});
  useEffect(() => {
    void listSongModels()
      .then((files) => {
        console.log("[歌曲创作] 检测到的模型文件:", files);
        const map: typeof songInstall = {};
        for (const m of SONG_MODEL_CATALOG) {
          const s = getSongInstallStatus(files, m);
          console.log(`[歌曲创作] 模型 ${m.id} 检测结果:`, s);
          map[m.id] = { installed: s.installed, ready: s.ready, required: s.required };
        }
        console.log("[歌曲创作] 最终安装状态:", map);
        setSongInstall(map);
      })
      .catch((err) => {
        console.error("[歌曲创作] 模型检测失败:", err);
      });
  }, []);



  return (
    <div className="ss-overlay" onClick={onClose}>
      <div className="ss-panel" role="dialog" onClick={(e) => e.stopPropagation()}>
        <header className="ss-head">
          <span className="ss-title">歌曲制作工作室</span>
          <div className="ss-player-controls">
            {selectedSong && (
              <>
                <button 
                  className="ss-player-btn"
                  onClick={handlePreviousSong}
                  disabled={!selectedSong || history.findIndex(s => s.uuid === selectedSong.uuid) === 0}
                  title="上一曲"
                >
                  ⏮
                </button>
                <button 
                  className="ss-player-btn ss-player-btn-main"
                  onClick={togglePlayPause}
                  disabled={!playingAudio}
                  title={isPlaying ? "暂停" : "播放"}
                >
                  {isPlaying ? "⏸" : "▶"}
                </button>
                <button 
                  className="ss-player-btn"
                  onClick={handleNextSong}
                  disabled={!selectedSong || history.findIndex(s => s.uuid === selectedSong.uuid) === history.length - 1}
                  title="下一曲"
                >
                  ⏭
                </button>
                <div className="ss-player-progress-container">
                  <span className="ss-player-time">{formatDuration(currentTime)}</span>
                  <input 
                    type="range"
                    className="ss-player-progress"
                    min="0"
                    max={duration || 0}
                    value={currentTime}
                    onChange={(e) => handleSeek(Number(e.target.value))}
                  />
                  <span className="ss-player-time">{formatDuration(duration)}</span>
                </div>
                <span className="ss-player-song-name">{selectedSong.label || "Untitled"}</span>
              </>
            )}
          </div>
          <button 
            className="ss-help" 
            onClick={async () => {
              try {
                await openUrl("歌曲制作完整教程.md");
              } catch (err) {
                alert("无法打开帮助文档：" + err);
              }
            }}
            title="查看帮助文档"
          >
            ❓
          </button>
          <button className="ss-close" onClick={onClose}>✕</button>
        </header>

        <div className="ss-body">
          {/* 左侧配置区域 - 三段式：顶栏 / 滚动区 / 底栏（规划 5.3） */}
          <aside className="ss-col ss-left">
            {/* 顶栏：模型选择 + 任务模式条 */}
            <div className="ss-topbar">
              <div className="ss-quickbar">
                <div className="ss-quickbar-row">
                  <select
                    className="ss-model-select"
                    value={effectiveModel}
                    disabled={taskDef.models.length <= 1}
                    onChange={(e) => {
                      const selected = songModelById(e.target.value);
                      if (selected) {
                        setSelectedModel(selected.id);
                        setModelFamily(selected.family);
                        setParams((prev) => ({
                          ...prev,
                          service_url: getDefaultServiceUrl(selected.family),
                        }));
                      }
                    }}
                  >
                    {taskDef.models.map((id) => {
                      const m = songModelById(id);
                      const inst = songInstall[id];
                      const statusIcon = inst && inst.installed ? "🟢" : "🔴";
                      return (
                        <option key={id} value={id}>
                          {statusIcon} {m?.label.zh ?? id}
                        </option>
                      );
                    })}
                  </select>
                  {effectiveModel && (songInstall[effectiveModel]?.installed === false) && (
                    <button
                      className="ss-download-btn"
                      onClick={() => {
                        onClose();
                        window.dispatchEvent(new CustomEvent("navigate-to-resource-manager", {
                          detail: { tab: "song" }
                        }));
                      }}
                      title="下载模型"
                    >
                      ⬇️
                    </button>
                  )}
                  
                  <select
                    className="ss-quickbar-select"
                    value={params.language}
                    onChange={(e) => setParams({ ...params, language: e.target.value })}
                  >
                    <option value="zh">🇨🇳 中文</option>
                    <option value="en">🇺🇸 English</option>
                    <option value="ja">🇯🇵 日本語</option>
                    <option value="ko">🇰🇷 한국어</option>
                  </select>

                  <div className="ss-gender-buttons">
                    <button
                      className={`ss-gender-btn ${params.vocal_gender === "male" ? "active" : ""}`}
                      onClick={() => setParams({ ...params, vocal_gender: "male" })}
                      title="男声"
                    >
                      👨 <span className="ss-gender-label">男生演唱</span>
                    </button>
                    <button
                      className={`ss-gender-btn ${params.vocal_gender === "female" ? "active" : ""}`}
                      onClick={() => setParams({ ...params, vocal_gender: "female" })}
                      title="女声"
                    >
                      👩 <span className="ss-gender-label">女生演唱</span>
                    </button>
                    <button
                      className={`ss-gender-btn ${(!params.vocal_gender || params.vocal_gender === "auto") ? "active" : ""}`}
                      onClick={() => setParams({ ...params, vocal_gender: "auto" })}
                      title="自动"
                    >
                      ⚡ <span className="ss-gender-label">自动</span>
                    </button>
                  </div>

                  <button
                    className="ss-params-help-btn"
                    onClick={() => setShowParamsTutorial(true)}
                    title="参数教程"
                  >
                    ❓
                  </button>
                </div>
              </div>
              <div className="ss-mode-tabs">
                <select
                  className="ss-task-select"
                  value={activeTask}
                  onChange={(e) => switchTask(e.target.value as SongTaskId)}
                >
                  {SONG_TASK_ORDER.map((id) => {
                    const def = SONG_TASKS[id];
                    return (
                      <option key={id} value={id}>
                        {def.icon} {tn(def.name)}
                      </option>
                    );
                  })}
                </select>

                {/* 翻唱强度 - 仅翻唱模式显示，紧跟任务选择框 */}
                {activeTask === "cover" && (
                  <div className="ss-cover-strength-inline">
                    <label className="ss-field-label-small" title="0 = 保留原曲，1 = 完全跟随提示词">
                      翻唱强度
                    </label>
                    <input
                      className="ss-slider-inline"
                      type="range"
                      min={0}
                      max={1}
                      step={0.05}
                      value={modeForm.coverStrength}
                      onChange={(e) => patchModeForm({ coverStrength: Number(e.target.value) })}
                    />
                    <span className="ss-slider-value">{modeForm.coverStrength.toFixed(2)}</span>
                  </div>
                )}

                {/* 输出产物勾选 - 紧跟任务选择框 */}
                {(activeTask === "generate" || activeTask === "instrumental") && (
                  <div className="ss-output-toggles-inline">
                    <button
                      className={`ss-output-toggle ${params.want_stems ? 'active' : ''} ${!currentModel?.supportsMultiTrack ? 'disabled' : ''}`}
                      onClick={() => currentModel?.supportsMultiTrack && setParams({ ...params, want_stems: !params.want_stems })}
                      disabled={!currentModel?.supportsMultiTrack}
                      title={!currentModel?.supportsMultiTrack ? '当前模型不支持' : '输出多轨分离版本'}
                    >
                      <span className="ss-toggle-icon">🎚️</span>
                      <span className="ss-toggle-label">多轨</span>
                    </button>
                    <button
                      className={`ss-output-toggle ${params.want_midi ? 'active' : ''} ${!currentModel?.supportsMidi ? 'disabled' : ''}`}
                      onClick={() => currentModel?.supportsMidi && setParams({ ...params, want_midi: !params.want_midi })}
                      disabled={!currentModel?.supportsMidi}
                      title={!currentModel?.supportsMidi ? '当前模型不支持' : '输出MIDI乐谱文件'}
                    >
                      <span className="ss-toggle-icon">🎹</span>
                      <span className="ss-toggle-label">MIDI</span>
                    </button>
                    <button
                      className={`ss-output-toggle ${params.want_lrc ? 'active' : ''}`}
                      onClick={() => setParams({ ...params, want_lrc: !params.want_lrc })}
                      title="输出歌词字幕文件"
                    >
                      <span className="ss-toggle-icon">📝</span>
                      <span className="ss-toggle-label">字幕</span>
                    </button>
                  </div>
                )}
              </div>
            </div>

            {/* 滚动区：源选择 + 模式参数 + 创作内容 */}
            <div className="ss-scroll">
            {needsSource && (
              <section className="ss-field-section">
                <div className="ss-textarea-label">源音频</div>
                <div className="ss-source-unified">
                  <select
                    className="ss-input ss-source-selector"
                    value={modeForm.sourcePath || "_placeholder_"}
                    onChange={(e) => {
                      const val = e.target.value;
                      if (val === "_placeholder_" || val === "_file_") return;
                      
                      if (val.startsWith("history:")) {
                        const path = val.substring(8);
                        const entry = audioHistory.find((h) => h.outputs.some((o) => o.audio_path === path));
                        patchModeForm({
                          sourceKind: "history",
                          sourcePath: path,
                          sourceLabel: entry?.label || "结果库音频",
                          srcDurationSec: entry?.settings.audio_duration || 0,
                        });
                      } else if (val.startsWith("track:")) {
                        const path = val.substring(6);
                        const src = trackSources.find((t) => t.path === path);
                        patchModeForm({
                          sourceKind: "track",
                          sourcePath: path,
                          sourceLabel: src?.label || "工程音频",
                          srcDurationSec: src?.durationSec || 0,
                        });
                      }
                    }}
                  >
                    <option value="_placeholder_">请选择音频来源…</option>
                    {audioHistory.length > 0 && (
                      <optgroup label="📚 结果库">
                        {audioHistory.map((h) => {
                          const audioPath = h.outputs.find((o) => o.audio_path)?.audio_path ?? "";
                          return (
                            <option key={h.uuid} value={`history:${audioPath}`}>
                              {h.label || "Untitled"}（{h.settings.audio_duration}s）
                            </option>
                          );
                        })}
                      </optgroup>
                    )}
                    {trackSources.length > 0 && (
                      <optgroup label="🎵 工程轨道">
                        {trackSources.map((t) => (
                          <option key={t.path} value={`track:${t.path}`}>
                            {t.label}（{t.durationSec.toFixed(1)}s）
                          </option>
                        ))}
                      </optgroup>
                    )}
                    <option value="_file_" disabled>────────────</option>
                  </select>
                  <button
                    className="ss-btn-secondary ss-source-file-btn"
                    onClick={async () => {
                      const picked = await openDialog({
                        multiple: false,
                        filters: [{ name: "音频", extensions: ["wav", "mp3", "flac", "ogg", "m4a", "aac"] }],
                      });
                      if (typeof picked === "string") {
                        patchModeForm({
                          sourceKind: "file",
                          sourcePath: picked,
                          sourceLabel: picked.split(/[\\/]/).pop() || picked,
                          srcDurationSec: 0,
                        });
                      }
                    }}
                  >
                    📂 本地文件
                  </button>
                </div>
                {modeForm.sourcePath && (
                  <div className="ss-source-summary">
                    已选：{modeForm.sourceLabel}
                    {modeForm.srcDurationSec > 0 ? `（${modeForm.srcDurationSec.toFixed(1)}s）` : ""}
                  </div>
                )}
              </section>
            )}

            {/* 模式专属参数 */}
            {activeTask === "repaint" && (
              <section className="ss-field-section">
                <div className="ss-textarea-label">重绘区间（秒）</div>
                <div className="ss-repaint-row">
                  <input
                    className="ss-input"
                    type="number"
                    min={0}
                    step={0.5}
                    value={modeForm.repaintStart}
                    onChange={(e) => patchModeForm({ repaintStart: Number(e.target.value) })}
                  />
                  <span className="ss-repaint-dash">→</span>
                  <input
                    className="ss-input"
                    type="number"
                    min={0}
                    step={0.5}
                    value={modeForm.repaintEnd}
                    onChange={(e) => patchModeForm({ repaintEnd: Number(e.target.value) })}
                  />
                </div>
                {modeForm.srcDurationSec > 0 && (
                  <div className="ss-field-hint">源时长 {modeForm.srcDurationSec.toFixed(1)}s，需满足 0 ≤ 起点 &lt; 终点 ≤ 源时长</div>
                )}
              </section>
            )}

            {(activeTask === "complete" || activeTask === "extract") && (
              <section className="ss-field-section">
                {activeTask === "extract" && (
                  <div className="ss-extract-engines">
                    <label className={`ss-radio-card ${modeForm.extractEngine === "ace" ? "active" : ""}`}>
                      <input
                        type="radio"
                        checked={modeForm.extractEngine === "ace"}
                        onChange={() => patchModeForm({ extractEngine: "ace" })}
                      />
                      <div>
                        <div className="ss-radio-title">ACE 音轨分离</div>
                        <div className="ss-radio-sub">按音轨类型顺序分出，效果更可控</div>
                      </div>
                    </label>
                    <label className={`ss-radio-card ${modeForm.extractEngine === "demucs" ? "active" : ""}`}>
                      <input
                        type="radio"
                        checked={modeForm.extractEngine === "demucs"}
                        onChange={() => patchModeForm({ extractEngine: "demucs" })}
                      />
                      <div>
                        <div className="ss-radio-title">Demucs 快速分轨</div>
                        <div className="ss-radio-sub">人声/鼓/贝斯/其他，免费无需生成模型</div>
                      </div>
                    </label>
                  </div>
                )}
                {(activeTask === "complete" || (activeTask === "extract" && modeForm.extractEngine === "ace")) && (
                  <>
                    <div className="ss-textarea-label">
                      {activeTask === "complete" ? "要补全的音轨（至少一项）" : "要分出的音轨（可多选，按顺序执行）"}
                    </div>
                    <div className="ss-class-chips">
                      {ACE_TRACK_CLASSES.map((cls) => {
                        const on = modeForm.trackClasses.includes(cls);
                        return (
                          <button
                            key={cls}
                            className={`ss-class-chip ${on ? "on" : ""}`}
                            onClick={() =>
                              patchModeForm({
                                trackClasses: on
                                  ? modeForm.trackClasses.filter((c) => c !== cls)
                                  : [...modeForm.trackClasses, cls],
                              })
                            }
                          >
                            {cls}
                          </button>
                        );
                      })}
                    </div>
                  </>
                )}
              </section>
            )}

            {activeTask === "lego" && (
              <section className="ss-field-section">
                <div className="ss-textarea-label">叠加的音轨元素</div>
                <select
                  className="ss-input"
                  value={modeForm.legoTrack}
                  onChange={(e) => patchModeForm({ legoTrack: e.target.value })}
                >
                  {ACE_TRACK_CLASSES.map((cls) => (
                    <option key={cls} value={cls}>{cls}</option>
                  ))}
                </select>
                <div className="ss-field-hint">在下方提示词中描述要叠加的新元素</div>
              </section>
            )}

            {activeTask === "sheet" && (
              <section className="ss-field-section">
                <div className="ss-textarea-label">ABC 乐谱（可选，留空则全新生成）</div>
                <textarea
                  className="ss-lyrics-textarea ss-abc-textarea"
                  placeholder="粘贴 ABC 乐谱以续写/变奏…"
                  value={modeForm.sheetAbc}
                  onChange={(e) => patchModeForm({ sheetAbc: e.target.value })}
                />
              </section>
            )}

            {/* 歌词 */}
            {taskDef.input.lyrics !== "none" && (
              <section className="ss-lyrics-section">
                <div className="ss-textarea-label">歌词输入</div>
                <div className="ss-lyrics-wrapper">
                  <div className="ss-structure-chips">
                    {["Intro", "Verse", "Chorus", "Bridge", "Outro"].map((tag) => (
                      <button
                        key={tag}
                        className="ss-structure-chip"
                        onClick={() => insertLyricTag(tag)}
                        title={`在光标处插入 [${tag}]`}
                      >
                        {tag}
                      </button>
                    ))}
                  </div>
                  <textarea
                    ref={lyricsAreaRef}
                    className="ss-lyrics-textarea"
                    placeholder="输入或粘贴歌词...&#10;&#10;支持 [Verse] [Chorus] 等结构标签"
                    value={params.lyrics}
                    onChange={(e) => setParams({ ...params, lyrics: e.target.value })}
                  />
                  <div className="ss-textarea-actions">
                    <button
                      className="ss-btn-text"
                      title={i18n.t("songStudio.importLyricsTooltip")}
                      onClick={async () => {
                        try {
                          const path = await openDialog({
                            title: i18n.t("songStudio.importFromFile"),
                            filters: [{ name: "文本文件", extensions: ["txt", "lrc", "md"] }],
                            multiple: false,
                          });
                          if (path) {
                            const text = await readTextFile(path as string);
                            setParams({ ...params, lyrics: text });
                          }
                        } catch (err) {
                          console.error("导入歌词文件失败:", err);
                        }
                      }}
                    >
                      📄
                    </button>
                    <button
                      className="ss-btn-text"
                      onClick={() => {
                        navigator.clipboard.readText().then((text) => {
                          setParams({ ...params, lyrics: text });
                        });
                      }}
                    >
                      📋
                    </button>
                    <button
                      className="ss-btn-text"
                      onClick={() => setParams({ ...params, lyrics: "" })}
                    >
                      🗑️
                    </button>
                  </div>
                </div>
              </section>
            )}

            {/* 提示词区域 - 4-5排文字高度 */}
            {taskDef.input.prompt !== "none" && (
            <section className="ss-prompt-section">
              <div className="ss-textarea-label">
                {activeTask === "lego" ? "新元素描述（必填）" : "音乐风格与描述"}
              </div>
              <div className="ss-prompt-wrapper">
                <textarea
                  className="ss-prompt-input"
                  placeholder={activeTask === "lego"
                    ? "描述要叠加的元素，如：明亮的萨克斯独奏"
                    : "描述音乐风格、情绪、节奏等...&#10;例如: 轻快的流行风格，温暖阳光的感觉"}
                  value={params.prompt}
                  onChange={(e) => setParams({ ...params, prompt: e.target.value })}
                />
                <div className="ss-quick-tags">
                  <button
                    className="ss-quick-tag"
                    style={{ pointerEvents: "auto" }}
                    title={i18n.t("songStudio.importPromptTooltip")}
                    onClick={async () => {
                      try {
                        const path = await openDialog({
                          title: i18n.t("songStudio.importFromFile"),
                          filters: [{ name: "文本文件", extensions: ["txt", "md"] }],
                          multiple: false,
                        });
                        if (path) {
                          const text = await readTextFile(path as string);
                          setParams({ ...params, prompt: text });
                        }
                      } catch (err) {
                        console.error("导入提示词文件失败:", err);
                      }
                    }}
                  >
                    📄
                  </button>
                  {STYLE_TAGS.map((tag) => (
                    <button
                      key={tag.value}
                      className="ss-quick-tag"
                      onClick={() => {
                        const current = params.prompt;
                        if (current.includes(tag.label)) return;
                        setParams({ ...params, prompt: current ? `${current}, ${tag.label}` : tag.label });
                      }}
                    >
                      {tag.label}
                    </button>
                  ))}
                </div>
              </div>
            </section>
            )}

            {/* 参数设置 - 仅生成/伴奏模式显示 */}
            {(activeTask === "generate" || activeTask === "instrumental") && (
            <div className="ss-params-section">
              <div className="ss-params-header">
                <span className="ss-params-title">参数设置</span>
                <div className="ss-params-actions">
                  <button
                    className="ss-params-more-btn"
                    onClick={() => setShowAdvancedParams(!showAdvancedParams)}
                    title="更多设置"
                  >
                    {showAdvancedParams ? "收起" : "更多设置"}
                  </button>
                  <button
                    className="ss-params-help-btn"
                    onClick={() => setShowParamsTutorial(true)}
                    title="参数教程"
                  >
                    ❓
                  </button>
                </div>
              </div>
              
              <div className="ss-params-grid-compact">
                <div className="ss-field-compact">
                  <label className="ss-field-label" title="每分钟节拍数，控制歌曲速度">
                    <span className="ss-field-label-icon">🎼</span>
                    节奏速度(BPM)
                  </label>
                  <input
                    className="ss-input"
                    type="number"
                    min={40}
                    max={240}
                    value={params.bpm}
                    onChange={(e) => setParams({ ...params, bpm: Number(e.target.value) })}
                  />
                </div>

                <div className="ss-field-compact">
                  <label className="ss-field-label" title="生成歌曲的总时长">
                    <span className="ss-field-label-icon">⏱️</span>
                    时长(秒)
                  </label>
                  <input
                    className="ss-input"
                    type="number"
                    min={10}
                    max={300}
                    value={params.audio_duration}
                    onChange={(e) => setParams({ ...params, audio_duration: Number(e.target.value) })}
                  />
                </div>

                <div className="ss-field-compact ss-field-with-dice">
                  <label className="ss-field-label" title="随机种子，控制生成的随机性">
                    <span className="ss-field-label-icon">🎲</span>
                    随机种子
                  </label>
                  <div className="ss-seed-input-group">
                    <input
                      className="ss-input"
                      type="number"
                      min={1}
                      max={2147483647}
                      value={commonParams.seed ?? ""}
                      onChange={(e) => setCommonParams({ seed: Number(e.target.value) })}
                    />
                    <button
                      className="ss-dice-btn"
                      onClick={() => setCommonParams({ seed: randomSeed() })}
                      title="生成随机种子"
                    >
                      🎲
                    </button>
                  </div>
                </div>
              </div>

              {showAdvancedParams && (
                <div className="ss-advanced-params-modal" onClick={() => setShowAdvancedParams(false)}>
                  <div className="ss-advanced-params-content" onClick={(e) => e.stopPropagation()}>
                    <div className="ss-advanced-params-header">
                      <span className="ss-advanced-params-title">更多设置</span>
                      <button className="ss-advanced-params-close" onClick={() => setShowAdvancedParams(false)}>✕</button>
                    </div>
                    <div className="ss-advanced-params-body">
                      <DynamicParamsPanel
                        modelFamily={modelFamily}
                        params={dynamicParams}
                        onChange={updateModelParam}
                        lang="zh"
                      />
                    </div>
                  </div>
                </div>
              )}
            </div>
            )}
            </div>{/* /ss-scroll */}

            {/* 底栏：歌名 + 生成按钮 + 进度（规划 5.3：始终可见） */}
            <div className="ss-bottombar">
              <div className="ss-songname-inline">
                <label className="ss-field-label-small" title="为你的歌曲命名">
                  🎵 歌曲名称
                </label>
                <input
                  className="ss-input-inline"
                  placeholder={taskDef.name.zh}
                  value={params.songName}
                  onChange={(e) => setParams({ ...params, songName: e.target.value })}
                />
              </div>

              <div className="ss-generate-row">
                <button
                  className="ss-btn-generate"
                  disabled={queueWaiting.length >= 3 || !effectiveModel}
                  onClick={handleGenerate}
                  title={queueWaiting.length >= 3 ? "等待队列已满（最多3个任务）" : undefined}
                >
                  {`${taskDef.icon} ${tn(taskDef.action)}`}
                </button>

                {queueWaiting.length > 0 && (
                  <div className="ss-queue-status">
                    <span className="ss-queue-icon">⏳</span>
                    <span className="ss-queue-text">队列中: {queueWaiting.length} 个任务</span>
                  </div>
                )}
              </div>

              {taskError && (
                <div className="ss-task-error" onClick={() => setTaskError(null)}>
                  ⚠️ {taskError}
                </div>
              )}
            </div>
          </aside>

          {/* 右侧歌曲区域 - 50% */}
          <aside className="ss-col ss-right">
            <div className="ss-songs-list">
              {historyLoading && (
                <div className="ss-empty-state">
                  <div className="ss-empty-icon">⏳</div>
                  <div className="ss-empty-text">加载中...</div>
                </div>
              )}
              {!historyLoading && history.length === 0 && (
                <div className="ss-empty-state">
                  <div className="ss-empty-icon">🎵</div>
                  <div className="ss-empty-text">还没有生成任何歌曲<br/>开始创作你的第一首歌曲吧</div>
                </div>
              )}
              {!historyLoading && history.length > 0 && (
                <div className="ss-songs-list-items">
                  {history.map((song) => {
                    const isGenerating = song.uuid === generatingEntryId;
                    // 排队中：占位条目已落盘但任务还没开跑，没有进度可显示
                    const isQueued =
                      !isGenerating && queueWaiting.some((w) => w.entryId === song.uuid);
                    const isBusy = isGenerating || isQueued;

                    return (
                      <div
                        key={song.uuid}
                        className={`ss-song-item ${selectedSong?.uuid === song.uuid ? 'selected' : ''}`}
                        onClick={() => setSelectedSong(song)}
                        onContextMenu={(e) => {
                          e.preventDefault();
                          setContextMenuSong({ song, x: e.clientX, y: e.clientY });
                        }}
                      >
                        <div className="ss-song-item-cover">
                          <div className="ss-song-item-cover-icon">🎵</div>
                          {isGenerating ? (
                            <button 
                              className="ss-song-item-cancel-btn"
                              onClick={(e) => {
                                e.stopPropagation();
                                useSongTaskQueue.getState().cancelCurrent();
                              }}
                              title="取消生成"
                            >
                              ✕
                            </button>
                          ) : isQueued ? (
                            <div className="ss-song-item-queued-badge" title="排队中">⏳</div>
                          ) : (
                            <button 
                              className={`ss-song-item-play-btn ${isPlaying && selectedSong?.uuid === song.uuid ? 'playing' : ''}`}
                              onClick={(e) => {
                                e.stopPropagation();
                                handlePlaySong(song);
                              }}
                            >
                              {isPlaying && selectedSong?.uuid === song.uuid ? "⏸" : "▶"}
                            </button>
                          )}
                          {isPlaying && selectedSong?.uuid === song.uuid && (
                            <div className="ss-playing-indicator">
                              <div className="ss-wave-bar"></div>
                              <div className="ss-wave-bar"></div>
                              <div className="ss-wave-bar"></div>
                            </div>
                          )}
                        </div>
                        <div className="ss-song-item-info">
                          <div className="ss-song-item-title">{song.label || "Untitled"}</div>
                          <div className="ss-song-item-meta">
                            {formatTime(song.timestamp)} · {song.modelFamily.toUpperCase()} · {song.settings.audio_duration}s
                          </div>
                          {isGenerating && taskProgress && (
                            <div className="ss-song-item-progress">
                              <div className="ss-progress-bar">
                                <div 
                                  className="ss-progress-bar-fill" 
                                  style={{ width: `${taskProgress.percent}%` }}
                                />
                              </div>
                              <div className="ss-progress-text">
                                {translateStage(taskProgress.stage)} - {taskProgress.percent}%
                              </div>
                            </div>
                          )}
                          {isGenerating && !taskProgress && (
                            <div className="ss-song-item-progress">
                              <div className="ss-progress-text">准备中...</div>
                            </div>
                          )}
                          {isQueued && (
                            <div className="ss-song-item-progress">
                              <div className="ss-progress-text">排队中，等待前一首完成</div>
                            </div>
                          )}
                        </div>
                        <div className="ss-song-item-actions">
                          {!isBusy && (
                            <>
                              <button 
                                className="ss-song-item-action-btn"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleDeleteSong(song);
                                }}
                                title="删除"
                              >
                                �️
                              </button>
                              <button 
                                className="ss-song-item-action-btn"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setShowSendDialog(song);
                                }}
                                title="发送到轨道"
                              >
                                📤
                              </button>
                            </>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            <div className="ss-song-detail-panel">
              {selectedSong ? (
                <>
                  {/* 标题 */}
                  <div className="ss-detail-title-large">{selectedSong.label || "Untitled"}</div>
                  
                  {/* 详情信息紧跟标题 */}
                  <div className="ss-detail-info-inline">
                    <span className="ss-info-item">
                      <span className="ss-info-label">模型</span>
                      <span className="ss-info-value">{selectedSong.modelFamily.toUpperCase()}</span>
                    </span>
                    <span className="ss-info-divider">·</span>
                    <span className="ss-info-item">
                      <span className="ss-info-label">BPM</span>
                      <span className="ss-info-value">{selectedSong.settings.bpm}</span>
                    </span>
                    <span className="ss-info-divider">·</span>
                    <span className="ss-info-item">
                      <span className="ss-info-label">时长</span>
                      <span className="ss-info-value">{selectedSong.settings.audio_duration}s</span>
                    </span>
                  </div>

                  {/* 播放器 + 操作行（原左栏结果区迁移至此） */}
                  {selectedSong.outputs.length > 0 && (
                    <div className="ss-detail-player-row">
                      {playingAudio && (
                        <>
                          <button className="ss-audio-btn-play-large" onClick={togglePlayPause}>
                            {isPlaying ? "⏸" : "▶"}
                          </button>
                          <span className="ss-player-time">
                            {formatDuration(currentTime)} / {formatDuration(duration)}
                          </span>
                        </>
                      )}
                      <div className="ss-detail-actions">
                        <button className="ss-btn-result-action" onClick={() => handleDownloadSong(selectedSong)}>
                          💾 下载
                        </button>
                        <button className="ss-btn-result-action" onClick={() => handleAddToTrack(selectedSong)}>
                          🎵 添加到轨道
                        </button>
                        {selectedSong.trackId && (
                          <button className="ss-btn-result-action" onClick={() => handleLocateTrack(selectedSong.trackId!)}>
                            📍 定位轨道
                          </button>
                        )}
                      </div>
                    </div>
                  )}

                  {/* 创作助手 / 高级多轨 / MIDI 编辑器（紧凑次级按钮） */}
                  {selectedSong.outputs.some((o) => o.audio_path) && (
                    <div className="ss-detail-mtrick-row">
                      <button
                        className="ss-mtrick-btn ss-mtrick-compact"
                        disabled={generating}
                        onClick={() => {
                          setMultiTrackSrc(selectedSong?.outputs.find((o) => o.audio_path)?.audio_path);
                          setShowMultiTrack(true);
                        }}
                      >
                        🎛 高级多轨
                      </button>
                      <button
                        className="ss-mtrick-btn ss-mtrick-compact"
                        disabled={generating}
                        onClick={() => { setCreativeTab("instrument"); setShowCreative(true); }}
                      >
                        🪄 创作助手
                      </button>
                      <button
                        className="ss-mtrick-btn ss-mtrick-compact"
                        disabled={generating}
                        onClick={() => { setCreativeTab("cover"); setShowCreative(true); }}
                      >
                        🎤 翻唱助手
                      </button>
                      <button
                        className="ss-mtrick-btn ss-mtrick-compact"
                        disabled={generating}
                        onClick={() => void handleOpenMidiEditor(
                          selectedSong.outputs.find((o) => o.midi_path)?.midi_path,
                          selectedSong.label,
                          selectedSong.outputs.find((o) => o.abc_path)?.abc_path,
                        )}
                      >
                        🎹 MIDI 编辑
                      </button>
                    </div>
                  )}

                  {/* 提示词 */}
                  {selectedSong.settings.prompt && (
                    <div className="ss-detail-prompt-compact">
                      <div className="ss-detail-section-title">提示词</div>
                      <div className="ss-detail-prompt">
                        {selectedSong.settings.prompt}
                      </div>
                    </div>
                  )}

                  {/* 歌词区域 - 占据更多空间 */}
                  {selectedSong.settings.lyrics && (
                    <div className="ss-detail-lyrics-area">
                      <div className="ss-detail-section-title">歌词</div>
                      <div className="ss-detail-lyrics">
                        {selectedSong.settings.lyrics.split(/\r?\n/).map((line, idx) => (
                          <div key={idx} className="ss-lyric-line">
                            {line || "\u00A0"}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="ss-empty-state">
                  <div className="ss-empty-icon">👈</div>
                  <div className="ss-empty-text">选择一首歌曲<br/>查看详细信息</div>
                </div>
              )}
            </div>
          </aside>
        </div>

        {/* 参数教程弹窗 */}
        {showParamsTutorial && (
          <ParamsTutorialDialog
            modelFamily={modelFamily}
            onClose={() => setShowParamsTutorial(false)}
          />
        )}

        {/* 右键菜单（规划 7.2：通用 ContextMenu + 再创作子菜单） */}
        {contextMenuSong && (
          <ContextMenu
            x={contextMenuSong.x}
            y={contextMenuSong.y}
            items={buildSongMenu(contextMenuSong.song)}
            onClose={() => setContextMenuSong(null)}
          />
        )}

        {/* 发送到轨道对话框 */}
        {showSendDialog && (
          <SendToTrackDialog
            song={showSendDialog}
            onConfirm={(format, engine, dest) => {
              handleAddToTrack(showSendDialog, format, engine, dest);
              setShowSendDialog(null);
            }}
            onClose={() => setShowSendDialog(null)}
          />
        )}

        {/* 多轨工作室 */}
        {showMultiTrack && (
          <MultiTrackStudio
            songs={studioSongs}
            initialAudioPath={multiTrackSrc ?? selectedSong?.outputs.find((o) => o.audio_path)?.audio_path}
            onClose={() => setShowMultiTrack(false)}
            onGenerated={handleSaveMultiTrackResult}
            onSendToTrack={handleSendMultiTrackToTrack}
          />
        )}

        {/* 创作助手（功能 B/C/D：单乐器叠加 / 乐谱续写 / 翻唱助手） */}
        {showCreative && (
          <CreativeAssistant
            songs={studioSongs}
            initialTab={creativeTab}
            initialAudioPath={creativeSrc}
            onClose={() => setShowCreative(false)}
            onGenerated={handleCreativeGenerated}
            onSendToTrack={handleSendMultiTrackToTrack}
            onOpenMidi={(midiPath, label) => setMidiEditor({ path: midiPath, label })}
          />
        )}

        {/* MIDI 编辑器（功能 A：钢琴卷帘窗 + 基础编辑 + 导出 DAW + MIDI→音频回流） */}
        {midiEditor && (
          <MidiEditor
            midiPath={midiEditor.path}
            initialTitle={midiEditor.label}
            onClose={() => setMidiEditor(null)}
            onDepositNotes={handleMidiDepositNotes}
            onDepositAudio={handleMidiDepositAudio}
          />
        )}

        {completion && (
          <GenerationCompleteDialog
            songName={completion.songName}
            modelId={completion.modelId}
            processingTime={completion.processingTime}
            onClose={() => setCompletion(null)}
          />
        )}

        {/* 规划 12.4：原创度自检面板 */}
        {originalityCheck && (
          <OriginalityPanel
            sourcePath={originalityCheck.sourcePath}
            resultPath={originalityCheck.resultPath}
            onClose={() => setOriginalityCheck(null)}
          />
        )}
      </div>
    </div>
  );
}
