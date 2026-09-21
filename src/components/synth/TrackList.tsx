import { useState, useCallback, useEffect, useRef, Fragment } from "react";
import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import { useHistoryStore } from "../../store/history";
import { useTranslation } from "react-i18next";
import { open } from "@tauri-apps/plugin-dialog";
import { LANE_HEIGHT, LANE_GROUP_BAR_HEIGHT, LOUDNESS_LANE_HEIGHT, TRACK_HEADER_HEIGHT, FADER_MIN_DB, FADER_MAX_DB, AUDIO_EXTENSIONS } from "../../lib/constants";
import { computeTrackHeight, computeTrackYOffsets, computeTotalTracksHeight, findTrackAtY, hiddenTrackIds, getLanes, getLaneLayout, isLaneRowMuted, laneControlFor, loudnessBandH, type LaneGroupRun, type LaneMember } from "../../lib/trackLayout";
import { laneLabelParts } from "../../lib/audio/laneOps";
import { trackTypeCssVar, LANE_COLORS } from "../../lib/trackColors";
import { importAudioToNewTrack } from "../../lib/audio/import";
import { blankTrack } from "../../lib/trackFactory";
import { copyTrackToClipboard, pasteWithFeedback, clipboardKind } from "../../lib/clipboard";
import { VolumeFader, formatPan, formatDb } from "../common/VolumeFader";
import { ContextMenu, type MenuItem } from "../common/ContextMenu";
import { songTaskSubmenuItems } from "../../lib/song/daw-menu";
import * as playback from "../../lib/audio/playback";
import { useVoiceModelStore } from "../../store/voice-models";
import { backendOf, backendLabel, pickVoiceForTrack } from "../../lib/vocal/voicePick";
import { VOCAL_LANGUAGES, DEFAULT_LANG_ID } from "../../lib/vocal/languages";
import type { Track } from "../../types/project";
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

interface Props {
  width: number;
}

/** Context menu in the track-header column: per-track actions, "add material" at a boundary, or the
 *  ② vocal-track header pickers (S58): "voice" = singer list, "lang" = the track's default language. */
type Menu = { x: number; y: number } & (
  | { kind: "track"; trackId: string }
  | { kind: "add"; index: number }
  | { kind: "voice"; trackId: string }
  | { kind: "lang"; trackId: string }
);

export function TrackList({ width }: Props) {
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
  const [menu, setMenu] = useState<Menu | null>(null);
  const [hoverBoundary, setHoverBoundary] = useState<number | null>(null);
  const [editingTrackId, setEditingTrackId] = useState<string | null>(null);
  const [draggingTrackId, setDraggingTrackId] = useState<string | null>(null);
  // —— 轨道颜色选择器 state ——
  const [colorPicker, setColorPicker] = useState<{ trackId: string; x: number; y: number } | null>(null);
  // —— AI 弹窗 state (右键菜单触发) ——
  const [arrangeTarget, setArrangeTarget] = useState<string | null>(null);
  const [chordMidiTarget, setChordMidiTarget] = useState<string | null>(null);
  const [amtTarget, setAmtTarget] = useState<string | null>(null);
  const [amtMissingOpen, setAmtMissingOpen] = useState<string | null>(null);
  // 一键扒带:音频转谱完成后自动打开智能编曲面板(指向新转出的第一条旋律轨)。
  const [arrangeAfterAmt, setArrangeAfterAmt] = useState(false);

  // ── AMT 依赖预检: 右键点 AI 转谱时, 先检查 Tauri 环境 + 模型是否齐全 ──
  const tryOpenAmt = useCallback(async (trackId: string) => {
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
    } catch (e) {
      // fetchInstalled 也会因为无后端失败 → 直接进缺失弹窗
      useAppStore.getState().showToast("⚠️ 无法连接后端, 请确认 Tauri 应用正常启动", "info");
      setAmtMissingOpen(trackId);
    }
  }, []);
  const listRef = useRef<HTMLDivElement>(null);
  const trackDragRef = useRef<{ trackId: string; startY: number; dragging: boolean } | null>(null);

  // The boundary hint is computed from the cursor vs. the track layout; when the layout shifts
  // under a stationary cursor (vertical zoom or scroll), drop the stale hint until the pointer
  // moves again — otherwise the line sticks to the wrong boundary.
  useEffect(() => { setHoverBoundary(null); }, [vZoom, scrollY]);

  // 快捷键支持: R录音、M静音、S独奏
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // 忽略在输入框中的按键
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      // 忽略有修饰键的按键
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      
      const activeTrack = tracks.find((t) => t.id === activeTrackId);
      if (!activeTrack) return;
      
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
  const onTrackHeaderMouseDown = useCallback((e: React.MouseEvent, trackId: string) => {
    if (e.button !== 0) return;
    const el = e.target as HTMLElement;
    if (el.closest(".track-btn, .track-expand-btn, .track-name, .track-name-input, .vol-fader, .track-row-bot")) return;
    trackDragRef.current = { trackId, startY: e.clientY, dragging: false };
  }, []);

  // ── 轨道底边纵向拖拽(Studio Pro 式):拖任一条轨的底边 → 全局 vZoom 等比缩放(与 Alt+滚轮同
  //    一通路)。边缘跟随光标:每拖 TRACK_HEADER_HEIGHT 像素 ≈ vZoom ±1。pointer capture + rAF
  //    合帧,拖完不持久化(vZoom 本就不入 localStorage)。 ─────────────────────────────────
  const vDragRef = useRef<{ startY: number; startZoom: number } | null>(null);
  const vDragRafRef = useRef(0);
  useEffect(() => () => cancelAnimationFrame(vDragRafRef.current), []);
  const onTrackResizePointerDown = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    try { e.currentTarget.setPointerCapture(e.pointerId); } catch { /* ignore */ }
    const st = useAppStore.getState();
    vDragRef.current = { startY: e.clientY, startZoom: st.vZoom };
    document.body.style.cursor = "row-resize";
    document.body.style.userSelect = "none";
    const onMove = (ev: PointerEvent) => {
      const d = vDragRef.current;
      if (!d) return;
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
      if (vDragRafRef.current) { cancelAnimationFrame(vDragRafRef.current); vDragRafRef.current = 0; }
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  }, []);

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      const d = trackDragRef.current;
      if (!d) return;
      if (!d.dragging) {
        if (Math.abs(e.clientY - d.startY) <= 4) return;
        d.dragging = true;
        setDraggingTrackId(d.trackId);
        document.body.style.cursor = "grabbing";
        document.body.style.userSelect = "none"; // no stray text selection across track names
        // Reorder fires live per midpoint cross — coalesce the whole drag into one undo step.
        useHistoryStore.getState().beginTransaction();
      }
      const el = listRef.current;
      if (!el) return;
      const proj = useProjectStore.getState();
      const trks = proj.tracks;
      const fromIdx = trks.findIndex((t) => t.id === d.trackId);
      if (fromIdx < 0) return;
      const vz = useAppStore.getState().vZoom;
      const contentY = e.clientY - el.getBoundingClientRect().top + useAppStore.getState().scrollY;
      // Target = how many OTHER tracks' midpoints sit above the cursor. Excluding the dragged track
      // gives proper hysteresis (using its own slot as the only dead-band oscillates when it is
      // shorter than the track it crosses).
      const offs = computeTrackYOffsets(trks, vz);
      let target = 0;
      for (let i = 0; i < trks.length; i++) {
        if (i === fromIdx) continue;
        if (contentY >= offs[i]! + computeTrackHeight(trks[i]!, vz) / 2) target++;
      }
      if (target !== fromIdx) proj.reorderTrack(fromIdx, target);
    };
    const onUp = () => {
      if (trackDragRef.current) {
        const wasDragging = trackDragRef.current.dragging;
        trackDragRef.current = null;
        setDraggingTrackId(null);
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
        if (wasDragging) useHistoryStore.getState().commitTransaction();
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

  const commitRename = useCallback((trackId: string, name: string) => {
    const n = name.trim();
    if (n) updateTrack(trackId, { name: n });
    setEditingTrackId(null);
  }, [updateTrack]);

  const offsets = computeTrackYOffsets(tracks, vZoom);
  const totalH = computeTotalTracksHeight(tracks, vZoom);
  const hidden = hiddenTrackIds(tracks);

  // ── 文件夹分组操作 ──────────────────────────────────────────────
  // 在某轨上方插入一个空的文件夹容器轨, 并把该轨作为第一个子轨移入。
  const createFolderAbove = useCallback((trackId: string) => {
    const idx = tracks.findIndex((tk) => tk.id === trackId);
    const fid = crypto.randomUUID();
    const folder: Track = {
      id: fid, name: "新建分组", trackType: "audio", segments: [],
      volumeDb: 0, pan: 0, muted: false, solo: false, expanded: false,
      laneControls: {}, isFolder: true, folderCollapsed: false,
    };
    addTrack(folder, idx < 0 ? undefined : idx);
    updateTrack(trackId, { folderId: fid });
  }, [tracks, addTrack, updateTrack]);

  // 解散分组: 子轨 folderId 全部清掉(保留轨道与内容), 再移除文件夹轨本身。
  const dissolveFolder = useCallback((folderId: string) => {
    useHistoryStore.getState().beginTransaction();
    for (const tk of tracks) {
      if (tk.folderId === folderId) updateTrack(tk.id, { folderId: undefined });
    }
    removeTrack(folderId);
    useHistoryStore.getState().commitTransaction();
  }, [tracks, updateTrack, removeTrack]);

  // 移入上方最近的分组文件夹(向上扫描 tracks 数组)。
  const moveIntoFolderAbove = useCallback((trackId: string) => {
    const idx = tracks.findIndex((tk) => tk.id === trackId);
    for (let j = idx - 1; j >= 0; j--) {
      if (tracks[j]?.isFolder) {
        updateTrack(trackId, { folderId: tracks[j]!.id });
        return;
      }
    }
    useAppStore.getState().showToast("上方没有分组文件夹 — 先右键某轨「新建分组文件夹」", "info");
  }, [tracks, updateTrack]);

  // Reusable track-creation actions. `insertIndex` positions the new track at a boundary (the
  // right-click "add material" menu); omitting it appends at the bottom (the "+" menu).
  const createAudioTrack = useCallback((insertIndex?: number) => {
    const n = useProjectStore.getState().tracks.filter((tk) => tk.trackType === "audio").length + 1;
    addTrack(blankTrack(crypto.randomUUID(), `Audio ${n}`, "audio"), insertIndex);
  }, [addTrack]);

  const createVocalTrack = useCallback((insertIndex?: number) => {
    const n = useProjectStore.getState().tracks.filter((tk) => tk.trackType === "vocal").length + 1;
    addTrack(blankTrack(crypto.randomUUID(), `Vocal ${n}`, "vocal"), insertIndex);
  }, [addTrack]);

  const importAudioAt = useCallback(async (insertIndex?: number) => {
    const path = await open({
      title: t("toolbar.importAudio"),
      filters: [{ name: "Audio", extensions: AUDIO_EXTENSIONS }],
    });
    if (!path) return;
    // import.ts creates the track + loading segment immediately + owns decode/error handling.
    void importAudioToNewTrack(path as string, useProjectStore.getState().playheadTick, undefined, insertIndex);
  }, [t]);

  // Boundary nearest the cursor (insert index 0..tracks.length) within a small threshold, else null.
  const boundaryAt = useCallback((clientY: number): number | null => {
    const el = listRef.current;
    if (!el) return null;
    const contentY = clientY - el.getBoundingClientRect().top + scrollY;
    for (let i = 0; i <= tracks.length; i++) {
      const by = i < tracks.length ? offsets[i]! : totalH;
      if (Math.abs(contentY - by) <= 5) return i;
    }
    return null;
  }, [tracks.length, offsets, totalH, scrollY]);

  const handleContextMenu = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      const b = boundaryAt(e.clientY);
      if (b !== null) {
        setMenu({ kind: "add", index: b, x: e.clientX, y: e.clientY });
        return;
      }
      const el = listRef.current;
      if (!el) return;
      const contentY = e.clientY - el.getBoundingClientRect().top + scrollY;
      const idx = findTrackAtY(offsets, contentY);
      if (idx >= 0 && idx < tracks.length && contentY <= totalH) {
        setMenu({ kind: "track", trackId: tracks[idx]!.id, x: e.clientX, y: e.clientY });
      } else {
        // Empty area below the last track → "add material", appending at the bottom.
        setMenu({ kind: "add", index: tracks.length, x: e.clientX, y: e.clientY });
      }
    },
    [boundaryAt, offsets, totalH, scrollY, tracks],
  );

  const voiceModels = useVoiceModelStore((s) => s.models);
  const setVocalParams = useProjectStore((s) => s.setVocalParams);
  const vocalOov = useAppStore((s) => s.vocalOov); // ② S58 track-level OOV warning
  const vocalUnknownPhone = useAppStore((s) => s.vocalUnknownPhone); // S109 §C15: 音素写错 ≠ 歌词不认识
  const vocalDropped = useAppStore((s) => s.vocalDropped); // S85b: too-short dropped notes(独立文案)
  const vocalShort = useAppStore((s) => s.vocalShort); // S87: rescued notes — advisory, amber badge
  const vocalAliasHint = useAppStore((s) => s.vocalAliasHint); // S113 §C14: 别名形状提示 — advisory

  const menuItems: MenuItem[] = (() => {
    if (!menu) return [];
    if (menu.kind === "track") {
      const tr = tracks.find((tt) => tt.id === menu.trackId);
      if (!tr) return [];
      // DEBUG: 真实 track 结构
      console.log("[AI MENU DEBUG] track:", tr?.name, "type:", tr?.trackType, "segments:", tr?.segments?.map((s: any) => ({ type: s.content?.type, hasSource: !!(s.content as any)?.sourcePath })));
      const isMelodyLike = !!tr && (tr.trackType === "instrument" || tr.trackType === "vocal");
      const isAudio = !!tr && tr.trackType === "audio";
      // Audio path for audio-track analysis (drums, chords, AMT)
      const audioSegment = tr?.segments.find((s) => s.content.type === "audioClip");
      const audioPath = audioSegment?.content.type === "audioClip" ? audioSegment.content.sourcePath : null;
      const hasNotes = isMelodyLike && tr!.segments.some((s) => s.content.type === "notes" && s.content.notes.length > 0);
      const songSource = {
        kind: "track" as const,
        trackId: menu.trackId,
        segmentId: audioSegment?.id,
      };
      const songReturnTarget = {
        kind: "track" as const,
        trackId: menu.trackId,
        segmentId: audioSegment?.id,
        align: true,
      };
      const openSongTool = (tool: "multiTrack" | "creative" | "cover" | "midi") => {
        useAppStore.getState().setPendingSongTask({
          task: tool === "midi" ? "sheet" : tool === "cover" ? "cover" : "lego",
          source: songSource,
          sourceLabel: tr.name,
          returnTarget: songReturnTarget,
          tool,
        });
        const app = useAppStore.getState();
        if (!app.songStudioOpen) app.toggleSongStudio();
      };

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
          onClick: async () => { if (audioPath) { await detectTrackDrums(menu.trackId, audioPath); useAppStore.getState().showToast("鼓点识别完成", "success"); } } },
        // —— 规划 10.2：歌曲模型子菜单（DAW → 歌曲制作带参打开；不改变现有项与顺序）——
        ...(!!audioPath
          ? [{
              type: "submenu" as const,
              label: i18n.t("songMenu.remixGroup"),
              icon: "🎤",
              items: songTaskSubmenuItems({
                source: songSource,
                sourceLabel: tr.name,
                audioPath,
                trackId: menu.trackId,
                segmentId: audioSegment?.id,
              }),
            }]
          : []),
        {
          type: "submenu" as const,
          label: "🎵 歌曲生成工具",
          icon: "🎵",
          title: "AI 歌曲生成与编辑工具",
          items: [
            {
              label: "🎛️ 高级多轨工作室",
              title: "专业多轨编辑和制作",
              disabled: !audioPath,
              onClick: () => openSongTool("multiTrack"),
            },
            {
              label: "🪄 创作助手",
              title: "单乐器叠加、乐谱续写等快捷功能",
              disabled: !audioPath,
              onClick: () => openSongTool("creative"),
            },
            {
              label: "🎤 翻唱助手",
              title: "快速翻唱和风格转换",
              disabled: !audioPath,
              onClick: () => openSongTool("cover"),
            },
            {
              label: "🎹 MIDI 编辑器",
              title: "编辑和导出 MIDI 文件",
              disabled: !audioPath && !hasNotes,
              onClick: () => openSongTool("midi"),
            },
          ],
        },
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
      if (all.length === 0) return [{ label: t("tracks.noVoices"), disabled: true, onClick: () => {} }];
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

  return (
    <div
      className="track-list"
      style={{ width }}
      ref={listRef}
      onContextMenu={handleContextMenu}
      onMouseMove={(e) => {
        if (trackDragRef.current || vDragRef.current) { if (hoverBoundary !== null) setHoverBoundary(null); return; }
        const b = boundaryAt(e.clientY);
        setHoverBoundary((prev) => (prev === b ? prev : b));
      }}
      onMouseLeave={() => setHoverBoundary(null)}
    >
      {/* DEVICE-pixel snapping: scrollY is routinely fractional (raw wheel deltas + vZoom-scaled clamp
          values), and a translateY that lands off the DEVICE pixel grid rasterizes the whole DOM text
          column blurry in WebView2. Plain Math.round is not enough on scaled Windows displays (125% ⇒
          integer CSS px = fractional device px), so snap to device pixels via devicePixelRatio.
          Display-only: the store value (and the canvas, whose own drawing consumes it) stays unrounded,
          so shared scroll math is untouched; max divergence from the canvas is < 1 device px. */}
      <div
        className="track-list-scroll"
        style={{ transform: `translateY(${-(Math.round(scrollY * (window.devicePixelRatio || 1)) / (window.devicePixelRatio || 1))}px)` }}
      >
        {tracks.length === 0 && (
          <div 
            className="track-list-empty"
            onClick={async () => {
              // 点击空区域 → 打开文件选择器，添加音频文件
              const path = await open({
                title: t("toolbar.importAudio"),
                filters: [{ name: "Audio", extensions: AUDIO_EXTENSIONS }],
              });
              if (!path) return;
              // 第一个文件从起始位置（tick 0）开始
              void importAudioToNewTrack(path as string, 0);
            }}
            style={{ cursor: "pointer" }}
            title="点击上传音乐文件"
          >
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "8px" }}>
              <span className="text-muted">{t("tracks.empty")}</span>
              <span style={{ fontSize: "12px", opacity: 0.6 }}>点击上传音乐</span>
            </div>
          </div>
        )}
        {hoverBoundary !== null && (
          <div
            className="track-boundary-hint"
            style={{ top: hoverBoundary < tracks.length ? offsets[hoverBoundary]! : totalH }}
          />
        )}
        {tracks.map((track, i) => (
          <Fragment key={track.id}>
            {hidden.has(track.id) ? null : ghostInsert && ghostInsert.index === i && (
              <div
                className="track-ghost-slot"
                style={{ height: ghostInsert.count * TRACK_HEADER_HEIGHT * vZoom }}
              />
            )}
          {hidden.has(track.id) ? null : (
          <TrackItem
            track={track}
            index={i}
            childCount={track.isFolder ? tracks.filter((tk) => tk.folderId === track.id).length : undefined}
            vZoom={vZoom}
            hasSolo={tracks.some((tk) => tk.solo)}
            active={track.id === activeTrackId}
            dragging={track.id === draggingTrackId}
            editing={track.id === editingTrackId}
            onHeaderMouseDown={(e) => onTrackHeaderMouseDown(e, track.id)}
            onStartRename={() => setEditingTrackId(track.id)}
            onCommitRename={(name) => commitRename(track.id, name)}
            onCancelRename={() => setEditingTrackId(null)}
            onSelect={() => setActiveTrack(track.id)}
            onMute={() => {
              updateTrack(track.id, { muted: !track.muted });
              playback.updateTrackAudibility(useProjectStore.getState().tracks);
            }}
            onSolo={() => {
              updateTrack(track.id, { solo: !track.solo });
              playback.updateTrackAudibility(useProjectStore.getState().tracks);
            }}
            onVolumeChange={(v) => {
              updateTrack(track.id, { volumeDb: v });
              playback.updateTrackVolume(track.id, v);
            }}
            onPanChange={(v) => {
              updateTrack(track.id, { pan: v });
              playback.updateTrackPan(track.id, v);
            }}
            onToggleExpand={() => toggleTrackExpanded(track.id)}
            onTogglePlayOriginal={() => setTrackPlayOriginal(track.id, !track.playOriginal)}
            onLaneMute={(members) => {
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
            }}
            onLaneVolumeChange={(run, v) => {
              // Fan out over every member 组 under this bar (merged rows stay in lockstep; the legacy
              // laneId seed applies only to the primary 组 — other members' legacy entries key by
              // THEIR laneIds and simply start fresh, converging on this first touch).
              for (const gid of run.groupIds) {
                updateLaneControl(track.id, gid, { volumeDb: v }, gid === run.groupId ? run.laneId : undefined);
                playback.updateLaneVolume(track.id, gid, v);
              }
            }}
            onLanePanChange={(run, v) => {
              for (const gid of run.groupIds) {
                updateLaneControl(track.id, gid, { pan: v }, gid === run.groupId ? run.laneId : undefined);
                playback.updateLanePan(track.id, gid, v);
              }
            }}
            onOpenColorPicker={(x, y) => setColorPicker({ trackId: track.id, x, y })}
            onResizePointerDown={onTrackResizePointerDown}
            hasOov={track.segments.some(
              (sg) => (vocalOov[sg.id]?.length ?? 0) > (vocalUnknownPhone[sg.id]?.length ?? 0),
            )}
            hasUnknownPhone={track.segments.some((sg) => (vocalUnknownPhone[sg.id]?.length ?? 0) > 0)}
            hasDropped={track.segments.some((sg) => (vocalDropped[sg.id]?.length ?? 0) > 0)}
            hasShort={track.segments.some((sg) => (vocalShort[sg.id]?.length ?? 0) > 0)}
            hasAliasHint={track.segments.some((sg) => (vocalAliasHint[sg.id]?.length ?? 0) > 0)}
          />
          )}
          </Fragment>
        ))}
      </div>

      {menu && (
        <ContextMenu
          x={menu.x}
          y={menu.y}
          items={menuItems}
          onClose={() => setMenu(null)}
        />
      )}

      {/* —— AI 弹窗（右键菜单触发）—— */}
      {amtTarget && (
        <AmtConversionDialog
          trackId={amtTarget}
          onClose={() => { setAmtTarget(null); setArrangeAfterAmt(false); }}
          onImported={arrangeAfterAmt ? (ids) => {
            // 转谱完成 → 自动打开智能编曲面板,以第一条新轨为编曲源(它含全部音符)。
            const first = ids[0];
            if (first) setArrangeTarget(first);
          } : undefined}
        />
      )}
      {/* AMT 依赖缺失 → 红色提示卡 + 一键下载按钮 */}
      {amtMissingOpen && (() => {
        const store = useAmtModelStore();
        const needed = AMT_CATALOG.filter((m) => ["fluidsynth", "soundfont", "yourmt3_plus"].includes(m.architecture));
        const missing = needed.filter((m) => {
          const installed = store.installed.find((i) => i.id === m.id || i.architecture === m.architecture);
          return !installed || !installed.is_available;
        });
        return (
          <div className="arrange-overlay" onClick={() => setAmtMissingOpen(null)}>
            <div className="arrange-dialog" onClick={(e) => e.stopPropagation()} style={{ borderColor: "var(--color-error)" }}>
              <div className="arrange-header">
                <span className="arrange-title" style={{ color: "var(--color-error)" }}>⚠️ AI 转谱组件缺失</span>
                <button className="arrange-close" onClick={() => setAmtMissingOpen(null)}>✕</button>
              </div>
              <div style={{ padding: "12px 16px" }}>
                <p style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 12 }}>
                  检测到 {missing.length} 个必要组件还没装. 点每个组件右边的「下载」按钮即可.
                  全部下载完后右键那条音频轨 → 再点「AI 转谱」.
                </p>
                {!isTauri() && (
                  <div style={{ padding: 8, background: "rgba(239,68,68,0.12)", border: "1px solid var(--color-error)", borderRadius: 6, fontSize: 12, color: "var(--color-error)", marginBottom: 12 }}>
                    💡 当前在浏览器预览里运行 — AI 转谱依赖 Rust 后端, 需要用 Tauri 桌面版才能用. 不过 Lo-Fi / Trap / 民谣 / 浩室 等 22 种风格的**自动编曲/和弦生成/鼓点识别**都是纯前端, 现在就能跑!
                  </div>
                )}
                {missing.map((m) => {
                  const downloading = !!store.downloading[m.id];
                  return (
                    <div key={m.id} style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 0", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
                      <div style={{ flex: 1 }}>
                        <div style={{ fontSize: 13, fontWeight: 600, color: "#fff" }}>{m.name.zh}</div>
                        <div style={{ fontSize: 11, color: "#888" }}>{m.architecture} · {m.description.zh}</div>
                      </div>
                      <button
                        disabled={downloading || !isTauri()}
                        className="inspector-insert-btn"
                        style={{ opacity: downloading || !isTauri() ? 0.4 : 1, cursor: downloading ? "progress" : "pointer" }}
                        onClick={() => { void store.downloadEntry(m); }}
                      >
                        {downloading ? "⏳ 下载中…" : "⬇️ 下载"}
                      </button>
                    </div>
                  );
                })}
              </div>
              <div style={{ padding: "8px 16px 14px", display: "flex", justifyContent: "flex-end", gap: 8 }}>
                <button className="arrange-btn" onClick={() => setAmtMissingOpen(null)}>稍后</button>
                {!missing.some((m) => store.installed.find((i) => (i.id === m.id || i.architecture === m.architecture) && i.is_available)) && isTauri() && (
                  <button
                    className="arrange-btn"
                    style={{ background: "var(--color-error)", borderColor: "var(--color-error)" }}
                    onClick={() => { setAmtTarget(amtMissingOpen); setAmtMissingOpen(null); }}
                  >
                    仍然尝试打开 (可能会报错)
                  </button>
                )}
                {missing.length === 0 && isTauri() && (
                  <button className="arrange-btn" style={{ background: "var(--color-success)", borderColor: "var(--color-success)" }}
                    onClick={() => { setAmtTarget(amtMissingOpen); setAmtMissingOpen(null); }}>
                    ✅ 全部就绪 — 打开转谱
                  </button>
                )}
              </div>
            </div>
          </div>
        );
      })()}
      {arrangeTarget && (
        <ArrangeDialog trackId={arrangeTarget} onClose={() => setArrangeTarget(null)} />
      )}
      {chordMidiTarget && (
        <ChordMidiDialog trackId={chordMidiTarget} onClose={() => setChordMidiTarget(null)} />
      )}
      
      {/* 轨道颜色选择器 */}
      {colorPicker && (
        <TrackColorPicker
          currentColor={tracks.find((t) => t.id === colorPicker.trackId)?.color || trackTypeCssVar(tracks.find((t) => t.id === colorPicker.trackId)?.trackType || "audio")}
          onColorChange={(color) => {
            updateTrack(colorPicker.trackId, { color });
          }}
          onClose={() => setColorPicker(null)}
          position={{ x: colorPicker.x, y: colorPicker.y }}
        />
      )}
    </div>
  );
}

interface TrackItemProps {
  track: Track;
  index: number;
  /** 文件夹轨的子轨数(仅 isFolder 时传入, 用于头部计数徽标)。 */
  childCount?: number;
  vZoom: number;
  hasSolo: boolean;
  active: boolean;
  dragging: boolean;
  editing: boolean;
  onHeaderMouseDown: (e: React.MouseEvent) => void;
  onStartRename: () => void;
  onCommitRename: (name: string) => void;
  onCancelRename: () => void;
  onSelect: () => void;
  /** 轨道底边拖拽 → 纵向调整(全局 vZoom,Studio Pro 式;拖哪条轨都等比缩放全部轨道)。 */
  onResizePointerDown: (e: React.PointerEvent<HTMLDivElement>) => void;
  onMute: () => void;
  onSolo: () => void;
  onVolumeChange: (v: number) => void;
  onPanChange: (v: number) => void;
  onToggleExpand: () => void;
  onTogglePlayOriginal: () => void;
  onLaneMute: (members: LaneMember[]) => void;
  onLaneVolumeChange: (run: LaneGroupRun, v: number) => void;
  onLanePanChange: (run: LaneGroupRun, v: number) => void;
  onOpenColorPicker: (x: number, y: number) => void;
  /** ② S58: some segment on this track has OOV lyrics → header warning badge.
   *  ⚠ S109 (§C15): this is now "…an OOV LYRIC", i.e. EXCLUDING the notes whose failure was a
   *  mistyped phoneme — computed as `vocalOov.length > vocalUnknownPhone.length` per segment,
   *  because the latter is a SUBSET of the former (see the store's doc). Per SEGMENT and not per
   *  track: one segment can hold both kinds, and a track-level subtraction would hide one. */
  hasOov: boolean;
  /** S109 (§C15): some segment has a note whose BRACKET HINT / phoneme override names a phoneme that
   *  is not in the inventory. Same red badge as `hasOov`, different sentence — telling that user to
   *  "check the lyric or the language" points at the two things that are fine. Third instance of the
   *  split S85b/S87 made for `hasDropped`/`hasShort`. */
  hasUnknownPhone: boolean;
  /** S85b: some note rounded to zero frames(过短被跳过)→ same badge, truthful tooltip. */
  hasDropped: boolean;
  /** S87: some note was rescued by a frame borrow — ADVISORY (it sounds), so the badge turns amber
   *  rather than red unless a blocking finding is also present. */
  hasShort: boolean;
  /** S113 (§C14): some note's ALIAS resolved to a shape its convention cannot produce (an English
   *  word typed into an alias track). ADVISORY like `hasShort` — the note sounds, and it sounds
   *  exactly as it did before this channel existed. Fourth line of the same tooltip. */
  hasAliasHint: boolean;
}

function TrackItem({
  track,
  index,
  childCount,
  vZoom,
  active,
  dragging,
  editing,
  onHeaderMouseDown,
  onStartRename,
  onCommitRename,
  onCancelRename,
  onSelect,
  onResizePointerDown,
  onMute,
  onSolo,
  onVolumeChange: _onVolumeChange,
  onPanChange: _onPanChange,
  onToggleExpand,
  onTogglePlayOriginal,
  onLaneMute,
  onLaneVolumeChange,
  onLanePanChange,
  onOpenColorPicker,
  hasOov,
  hasUnknownPhone,
  hasDropped,
  hasShort,
  hasAliasHint,
}: TrackItemProps) {
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
      return (
        <div className={`track-item-group folder ${active ? "active" : ""}`} style={{ height: totalHeight }}>
          <div
            className={`track-item folder-track ${track.folderCollapsed ? "collapsed" : ""}`}
            onClick={onSelect}
            style={{ height: TRACK_HEADER_HEIGHT * vZoom }}
          >
            <div
              className="track-color-bar"
              style={{ background: colorVar, cursor: "pointer" }}
              onClick={(e) => {
                e.stopPropagation();
                const r = e.currentTarget.getBoundingClientRect();
                onOpenColorPicker(r.right + 4, r.top);
              }}
              title="点击选择轨道颜色"
            />
            <button
              className="track-expand-btn"
              onClick={(e) => {
                e.stopPropagation();
                useProjectStore.getState().updateTrack(track.id, { folderCollapsed: !track.folderCollapsed });
              }}
              title={track.folderCollapsed ? "展开分组" : "折叠分组"}
            >
              {track.folderCollapsed ? "▶" : "▼"}
            </button>
            <svg className="folder-glyph" viewBox="0 0 24 24" width="16" height="16" fill="currentColor" aria-hidden="true">
              <path d="M3 5a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5z" />
            </svg>
            <div className="track-info">
              {editing ? (
                <RenameInput initial={track.name} onCommit={onCommitRename} onCancel={onCancelRename} />
              ) : (
                <span
                  className="track-name"
                  title={track.name}
                  onDoubleClick={(e) => { e.stopPropagation(); onStartRename(); }}
                >
                  {track.name}
                </span>
              )}
            </div>
            <span className="folder-count">{childCount ?? 0} 轨</span>
          </div>
        </div>
      );
    }

  return (
    <div
      className={`track-item-group ${active ? "active" : ""} ${isEmpty ? "empty" : ""} ${dragging ? "dragging" : ""} ${track.playOriginal ? "play-original" : ""}`}
      style={{ height: totalHeight }}
    >
      <div
        className={`track-item ${track.trackType}-track`}
        onClick={onSelect}
        style={{ height: TRACK_HEADER_HEIGHT * vZoom }}
      >
        {/* Studio Pro 风格：左侧彩色色条（按轨道类型染色） + 展开箭头 */}
        <div 
          className="track-color-bar" 
          style={{ background: colorVar, cursor: 'pointer' }}
          onClick={(e) => {
            e.stopPropagation();
            const r = e.currentTarget.getBoundingClientRect();
            onOpenColorPicker(r.right + 4, r.top);
          }}
          title="点击选择轨道颜色"
        />
        {hasLanes && (
          <button
            className="track-expand-btn"
            onClick={(e) => { e.stopPropagation(); onToggleExpand(); }}
            title={track.expanded ? "折叠" : "展开"}
          >
            {track.expanded ? "▼" : "▶"}
          </button>
        )}
        {/* 参考图片2的3行布局：
            第1行：轨道号 + M/S/录音/声道按钮 + 歌曲标题
            第2行：音量推子（完整长条）
            第3行：输入源标签 + 下拉选择 */}
        <div className="track-main">
          {/* 第1行：轨道号 + 按钮组 + 标题 */}
          <div className="track-row-top">
            {/* 轨道号 */}
            <span className="track-no-badge" style={{ background: colorVar }} title={`轨道 ${index + 1}`}>{index + 1}</span>

            {/* M按钮 */}
            <button
              className={`track-state-btn ${track.muted ? "active-mute" : ""}`}
              onClick={(e) => { e.stopPropagation(); onMute(); }}
              title="静音"
            >
              M
            </button>
            
            {/* S按钮 */}
            <button
              className={`track-state-btn ${track.solo ? "active-solo" : ""}`}
              onClick={(e) => { e.stopPropagation(); onSolo(); }}
              title="独奏"
            >
              S
            </button>
            
            {/* 属性按钮 - 参数(调音台垂直推子)图标 */}
            <button
              className={`track-state-btn ${inspectorOpen ? "active-inspector" : ""}`}
              onClick={(e) => { e.stopPropagation(); useAppStore.getState().toggleInspector(track.id); }}
              title="轨道参数"
            >
              <svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
                <line x1="4" y1="2" x2="4" y2="14" />
                <line x1="2" y1="5" x2="6" y2="5" />
                <line x1="8" y1="2" x2="8" y2="14" />
                <line x1="6" y1="10" x2="10" y2="10" />
                <line x1="12" y1="2" x2="12" y2="14" />
                <line x1="10" y1="7" x2="14" y2="7" />
              </svg>
            </button>

            {/* 原音频/子轨切换按钮 (仅展开子轨的轨有) */}
            {hasLanes && (
              <button
                className={`track-state-btn track-src-btn ${track.playOriginal ? "active-src" : ""}`}
                onClick={(e) => { e.stopPropagation(); onTogglePlayOriginal(); }}
                title={track.playOriginal ? "正在播放原始音频 (子轨不进输出) — 点击切回子轨" : "正在播放子轨 — 点击切到原始音频"}
              >
                {track.playOriginal ? "SRC" : "SUB"}
              </button>
            )}
            
            {/* 歌曲标题 */}
            <div className="track-info">
              {editing ? (
                <RenameInput initial={track.name} onCommit={onCommitRename} onCancel={onCancelRename} />
              ) : (
                <span
                  className="track-name"
                  title={track.name}
                  onDoubleClick={(e) => { e.stopPropagation(); onStartRename(); }}
                >
                  {track.name}
                </span>
              )}
            </div>
            
            {rendering && (
              <span className="track-render-spinner" title={t("vocalEditor.render.rendering")}>
                <svg viewBox="0 0 24 24" width="12" height="12"><path fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" d="M12 3a9 9 0 1 0 9 9" /></svg>
              </span>
            )}
            
            {(hasOov || hasUnknownPhone || hasDropped || hasShort || hasAliasHint) && (
              <span
                className={
                  hasOov || hasUnknownPhone || hasDropped ? "track-oov-badge" : "track-oov-badge advisory"
                }
                title={[
                  hasOov ? t("tracks.oovWarning") : null,
                  hasUnknownPhone ? t("tracks.unknownPhoneWarning") : null,
                  hasDropped ? t("tracks.droppedWarning") : null,
                  hasShort ? t("tracks.shortWarning") : null,
                  hasAliasHint ? t("tracks.aliasHintWarning") : null,
                ]
                  .filter(Boolean)
                  .join("\n")}
              >
                <svg viewBox="0 0 24 24" width="12" height="12"><path fill="currentColor" d="M12 3 2 21h20L12 3zm-1 7h2v6h-2v-6zm0 7h2v2h-2v-2z" /></svg>
              </span>
            )}
          </div>
          
          {/* 第2行：音量推子（完整长条） */}
          <div className="track-row-volume">
            <VolumeFader
              value={track.volumeDb}
              min={FADER_MIN_DB}
              max={FADER_MAX_DB}
              onChange={_onVolumeChange}
              onGestureStart={() => useHistoryStore.getState().beginTransaction()}
              onGestureEnd={() => useHistoryStore.getState().commitTransaction()}
            />
            <input
              type="text"
              className="track-volume-input"
              value={track.volumeDb > FADER_MIN_DB ? `${track.volumeDb > 0 ? "+" : ""}${track.volumeDb.toFixed(1)}` : "-∞"}
              onChange={(e) => {
                const val = e.target.value.trim();
                if (val === "-∞" || val === "-inf") {
                  _onVolumeChange(FADER_MIN_DB);
                } else {
                  const num = parseFloat(val);
                  if (!isNaN(num)) {
                    _onVolumeChange(Math.max(FADER_MIN_DB, Math.min(FADER_MAX_DB, num)));
                  }
                }
              }}
              onFocus={(e) => e.target.select()}
              onClick={(e) => e.stopPropagation()}
              title="点击输入音量(dB)"
            />
          </div>
          
          {/* 第3行：输入源 + 下拉选择 */}
          <div className="track-row-input">
            <button 
              className="track-input-label-btn"
              onClick={(e) => {
                e.stopPropagation();
                // TODO: 打开输入源选择菜单
                console.log("点击输入源选择");
              }}
              title="点击选择输入源"
            >
              <span className="track-input-label">输入 L+R 立体声</span>
              <div className="track-input-icons">
                <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="8" cy="12" r="3" />
                  <circle cx="16" cy="12" r="3" />
                  <line x1="8" y1="12" x2="16" y2="12" />
                </svg>
                <svg viewBox="0 0 24 24" width="10" height="10" fill="currentColor" style={{ marginLeft: "2px" }}>
                  <path d="M7 10l5 5 5-5z" />
                </svg>
              </div>
            </button>
          </div>
        </div>
        {/* 轨道波形缩略图(真实 peaks) */}
        <div className="track-waveform-thumb">
          <TrackWaveform track={track} height={TRACK_HEADER_HEIGHT * vZoom} />
        </div>
        {/* Track light = drag-reorder HANDLE: 拖拽手柄(波纹图标已移除) */}
        <div
          className="track-light"
          onMouseDown={onHeaderMouseDown}
          title="拖动重新排列轨道"
        />
      </div>
      {track.expanded && laneLayout.runs.map((run) => {
        // One GROUP BLOCK per 组+名 run (getLaneLayout — same geometry the canvas rows use): a slim
        // group BAR carrying the 轨道组 name + the group-level volume/pan (keyed by the 组 via
        // laneControlFor — all rows of one 组 share the mix; 解组 for independent control), then the
        // member rows with just the stem name + per-ROW mute (isLaneRowMuted, loose row semantics).
        const ctrl = laneControlFor(track, run.groupId, run.laneId);
        const laneRgb = LANE_COLORS[run.colorIndex % LANE_COLORS.length]!;
        return (
          <div key={run.key} className="lane-group" style={{ "--lane-rgb": laneRgb } as React.CSSProperties}>
            <div className="lane-group-bar" style={{ height: LANE_GROUP_BAR_HEIGHT * vZoom }}>
              <span className="lane-group-swatch" />
              {/* S59b: the GROUP's loudness-envelope toggle. Placed LEFT of the name on purpose:
                  the name is the flex-shrink absorber (flex:1 min-width:0) — as the LAST child of
                  the controls this button sat past the bar's right edge and overflow:hidden
                  clipped it invisible (the reported "看不到 dB 按钮"). AUDIO tracks only. */}
              {track.trackType === "audio" && (
                <button
                  className={`track-btn lane-db-btn ${track.laneLoudnessOpen?.[run.groupId] ? "active-orig" : ""}`}
                  title={t("tracks.loudnessLane")}
                  onClick={(e) => { e.stopPropagation(); useProjectStore.getState().toggleLaneLoudnessOpen(track.id, run.groupId); }}
                >
                  dB
                </button>
              )}
              <span className="lane-group-name" title={run.name}>{run.name}</span>
              <div className="track-controls lane-group-controls">
                <div className="fader-row">
                  <span className="fader-tag">V</span>
                  <VolumeFader
                    value={ctrl?.volumeDb ?? 0}
                    min={FADER_MIN_DB}
                    max={FADER_MAX_DB}
                    width={42}
                    onChange={(v) => onLaneVolumeChange(run, v)}
                    onGestureStart={() => useHistoryStore.getState().beginTransaction()}
                    onGestureEnd={() => useHistoryStore.getState().commitTransaction()}
                  />
                  <span className="fader-val">{formatDb(ctrl?.volumeDb ?? 0, FADER_MIN_DB)}</span>
                </div>
                <div className="fader-row">
                  <span className="fader-tag">P</span>
                  <VolumeFader
                    value={ctrl?.pan ?? 0}
                    min={-1}
                    max={1}
                    step={0.1}
                    fillFrom="center"
                    format={formatPan}
                    width={28}
                    onChange={(v) => onLanePanChange(run, v)}
                    onGestureStart={() => useHistoryStore.getState().beginTransaction()}
                    onGestureEnd={() => useHistoryStore.getState().commitTransaction()}
                  />
                  <span className="fader-val">{formatPan(ctrl?.pan ?? 0)}</span>
                </div>
              </div>
            </div>
            {lanes.slice(run.start, run.start + run.count).map(({ id, label, members }) => {
              // A merged row reads muted only when ALL members are (the toggle fans out, so they
              // only diverge via legacy state — the canvas dims each piece by its own member truth).
              const muted = members.every((m) => isLaneRowMuted(track, m.rowKey, m.laneId));
              // Show only the sub-name within the group (labels are "Group · stem") — the group name
              // lives on the bar above, and the bracket ties the rows to it (members, not peers).
              const subName = laneLabelParts(label).stem ?? label;
              return (
                <div key={id} className="lane-item" style={{ height: LANE_HEIGHT * vZoom }}>
                  <span className="lane-label" title={label}>{subName}</span>
                  <div className="track-controls">
                    <button
                      className={`track-btn ${muted ? "active-mute" : ""}`}
                      onClick={(e) => { e.stopPropagation(); onLaneMute(members); }}
                    >
                      M
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        );
      })}
      {/* S59 loudness-band header block — MUST render whenever the canvas band is open, else the
          header column's total height drifts from computeTrackHeight and every row below desyncs. */}
      {loudnessBandH(track) > 0 && (
        <div className="loudness-band-header" style={{ height: LOUDNESS_LANE_HEIGHT * vZoom }}>
          <span className="loudness-band-label">{t("tracks.loudnessLane")}</span>
        </div>
      )}
      {/* 轨道底边纵向拖拽条:拖动 → 全局 vZoom 等比缩放(与 Alt+滚轮同一通路),Studio Pro 式。
          只拦截左键按下,右键仍走列表级右键菜单(轨道间"添加素材"边界)。 */}
      <div className="track-resize-handle" onPointerDown={onResizePointerDown} title="拖拽调整轨道高度" />
    </div>
  );
}

/** Inline track-name editor. Commits once on blur (Enter blurs → commit; Escape cancels). */
function RenameInput({ initial, onCommit, onCancel }: {
  initial: string;
  onCommit: (name: string) => void;
  onCancel: () => void;
}) {
  const ref = useRef<HTMLInputElement>(null);
  const doneRef = useRef(false);
  return (
    <input
      ref={ref}
      className="track-name-input"
      autoFocus
      defaultValue={initial}
      onMouseDown={(e) => e.stopPropagation()}
      onClick={(e) => e.stopPropagation()}
      onKeyDown={(e) => {
        if (e.key === "Enter") ref.current?.blur();
        else if (e.key === "Escape") { doneRef.current = true; onCancel(); }
      }}
      onBlur={(e) => { if (doneRef.current) return; doneRef.current = true; onCommit(e.target.value); }}
    />
  );
}




