import { useEffect, useRef, useCallback, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useProjectStore, useTimeAxis } from "../../store/project";
import { useAudioStore } from "../../store/audio";
import { useAppStore } from "../../store/app";
import { useHistoryStore } from "../../store/history";
import { useWorkflowStore } from "../../store/workflow";
import { useTranslation } from "react-i18next";
import * as playback from "../../lib/audio/playback";
import { collectDirtyVocals, renderDirtyVocals, splitSegmentVocalAware, preflightVocalModels } from "../../lib/vocal/vocalRender";
import { collectDirtyInstruments, renderDirtyInstruments, instrumentTracksMissingFont } from "../../lib/soundfont/instrumentRender";
import { copySelectedSegments, cutSelectedSegments, pasteWithFeedback } from "../../lib/clipboard";
import { formatBarBeat, type TimeAxis } from "../../lib/timeAxis";
import { contentEndTick } from "../../lib/trackLayout";
import { sliceLaneGroupAtPlayhead, deleteLanePiece, liveSelectedLane } from "../../lib/laneEdit";
import { Dropdown } from "../common/Dropdown";
import { OverviewMap } from "./OverviewMap";
import { ShortcutsDialog } from "../common/ShortcutsDialog";
import { HistoryPanel } from "../common/HistoryPanel";
import "./Toolbar.css";

// Editable time-signature options (house-styled custom Dropdown — no native <select>). Numerator 1–16
// covers every common meter (5/4, 7/8, 12/8, …); denominator is restricted to powers of two — the only
// values for which TICKS_PER_BEAT*4/den is a whole tick count (den=8 ⇒ 240 ticks/beat, 1440/bar).
const TS_NUM_OPTIONS = Array.from({ length: 16 }, (_, i) => ({ value: i + 1, label: String(i + 1) }));
const TS_DEN_OPTIONS = [2, 4, 8, 16].map((d) => ({ value: d, label: String(d) }));

// ③ B8: 吸附步长选项。value = 栅格步进 tick 数（0 = 关闭栅格吸附，仅保留磁吸）；labelKey 指向 i18n。
const SNAP_GRID_OPTIONS = [
  { value: 0, labelKey: "toolbar.snapGridOff" },
  { value: 480, labelKey: "toolbar.snapGridWhole" }, // 1 拍
  { value: 240, labelKey: "toolbar.snapGridHalf" }, // 1/2
  { value: 120, labelKey: "toolbar.snapGridQuarter" }, // 1/4
  { value: 60, labelKey: "toolbar.snapGridEighth" }, // 1/8
  { value: 30, labelKey: "toolbar.snapGridSixteenth" }, // 1/16
];

export function Toolbar() {
  const { t } = useTranslation();
  // 整 store 订阅是播放头每帧触发 Toolbar 完整 React 重渲染的元凶 —— 把 playheadTick 单独拆出来
  // 走 ref + 订阅路径(命令式更新位置显示 DOM,不走 React reconciliation),其他字段保持 selector
  // 订阅(不会因 playheadTick 变化而重渲染)。
  const tempo = useProjectStore((s) => s.tempo);
  const playheadTick = useProjectStore((s) => s.playheadTick);
  const setTempo = useProjectStore((s) => s.setTempo);
  const setPlayhead = useProjectStore((s) => s.setPlayhead);
  const timeSignature = useProjectStore((s) => s.timeSignature);
  const setTimeSignature = useProjectStore((s) => s.setTimeSignature);
  const deleteSegments = useProjectStore((s) => s.deleteSegments);
  
  // 新增：拍号和调性弹窗状态
  const [timeSigOpen, setTimeSigOpen] = useState(false);
  const [keyOpen, setKeyOpen] = useState(false);
  const [selectedKey, setSelectedKey] = useState<string>(""); // 空字符串表示未检测到调性
  const timeAxis = useTimeAxis();
  const isPlaying = useAudioStore((s) => s.isPlaying);
  const setPlaying = useAudioStore((s) => s.setPlaying);
  const seeking = useAudioStore((s) => s.seeking);
  const scheduleVersion = useAudioStore((s) => s.scheduleVersion);
  const preparing = useAudioStore((s) => s.preparing);
  const selectedSegment = useAppStore((s) => s.selectedSegment);
  const clearSelection = useAppStore((s) => s.clearSelection);
  const snapSegments = useAppStore((s) => s.snapSegments);
  const snapPlayhead = useAppStore((s) => s.snapPlayhead);
  const toggleSnapSegments = useAppStore((s) => s.toggleSnapSegments);
  const toggleSnapPlayhead = useAppStore((s) => s.toggleSnapPlayhead);
  const snapGridStep = useAppStore((s) => s.snapGridStep);
  const setSnapGridStep = useAppStore((s) => s.setSnapGridStep);
  const skin = useAppStore((s) => s.skin);
  const setSkin = useAppStore((s) => s.setSkin);
  const theme = useAppStore((s) => s.theme);
  const setTheme = useAppStore((s) => s.setTheme);

  // —— 搬来的弹窗 state (撤销历史 / 快捷键 / 主题) ——
  const [historyOpen, setHistoryOpen] = useState(false);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const [skinOpen, setSkinOpen] = useState(false);
  const [skinPos, setSkinPos] = useState<{ top: number; right: number } | null>(null);

  const THEMES: { theme: "dark" | "light"; skin: string; label: string; color: string }[] = [
    { theme: "light", skin: "default", label: "白天 · 默认", color: "#3B82F6" },
    { theme: "dark", skin: "default", label: "黑夜 · 默认", color: "#60A5FA" },
    { theme: "dark", skin: "junzi", label: "黑夜 · 君紫", color: "#A78BFA" },
    { theme: "dark", skin: "sakura", label: "黑夜 · 樱粉", color: "#F472B6" },
    { theme: "dark", skin: "mint", label: "黑夜 · 薄荷", color: "#6EE7B7" },
    { theme: "dark", skin: "sunset", label: "黑夜 · 熔橙", color: "#FB923C" },
    { theme: "dark", skin: "gray", label: "黑夜 · 灰色", color: "#9CA3AF" },
  ];

  // playheadTick: ref + imperative subscription — 播放期间每帧变化,走 selector 会触发整 Toolbar 重渲。
  // 改用 ref 读值 + 命令式更新位置显示 DOM,彻底消除每帧 React reconciliation 开销。
  const playheadRef = useRef(useProjectStore.getState().playheadTick);
  const timeAxisRef = useRef(timeAxis);
  timeAxisRef.current = timeAxis; // 保持 timeAxis 最新(换调性/拍号时需要刷新位置格式)
  const positionDisplayRef = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    const unsub = useProjectStore.subscribe((s) => {
      if (s.playheadTick !== playheadRef.current) {
        playheadRef.current = s.playheadTick;
        if (positionDisplayRef.current) {
          positionDisplayRef.current.textContent = formatPosition(s.playheadTick, timeAxisRef.current);
        }
      }
    });
    return unsub;
  }, []);
  const animRef = useRef<number>(0);
  const baseTickRef = useRef(0);
  const baseTimeRef = useRef(0);
  const animatingRef = useRef(false);
  const wasPlayingRef = useRef(false);
  const seekingRef = useRef(false);
  // Content extent (last segment box end) the playhead runs to — the transport stops when the PLAYHEAD
  // reaches here, NOT when the audio sources end, so a ② vocal stem shorter than its box plays through the
  // silent tail to the segment end instead of pausing mid-segment (the premature-pause ghost). Cached +
  // refreshed at play-start / on a structural (scheduleVersion) edit so the rAF doesn't recompute per frame.
  const contentEndRef = useRef(0);

  // 订阅 seeking 状态变化，避免每帧读取 store
  useEffect(() => {
    const unsub = useAudioStore.subscribe((s) => {
      seekingRef.current = s.seeking;
    });
    seekingRef.current = useAudioStore.getState().seeking;
    return unsub;
  }, []);

  // Playhead advance loop during playback.
  useEffect(() => {
    if (!isPlaying) {
      animatingRef.current = false;
      cancelAnimationFrame(animRef.current);
      wasPlayingRef.current = false;
      return;
    }

    baseTickRef.current = playheadTick;
    baseTimeRef.current = !wasPlayingRef.current
      ? playback.getScheduleTimeOrigin()
      : playback.getContextTime();
    wasPlayingRef.current = true;
    animatingRef.current = true;
    contentEndRef.current = contentEndTick(useProjectStore.getState().tracks);

    const animate = () => {
      if (!animatingRef.current) return;
      if (useAudioStore.getState().seeking) {
        // The user is dragging the playhead — pin the baseline to the dragged position so
        // the rAF doesn't clobber it; audio reschedules from here once the drag is released.
        baseTickRef.current = useProjectStore.getState().playheadTick;
        baseTimeRef.current = playback.getContextTime();
      } else {
        const elapsed = playback.getContextTime() - baseTimeRef.current;
        const tick = Math.round(baseTickRef.current + playback.secondsToTicks(elapsed, tempo));
        setPlayhead(tick);
        // Natural end = the PLAYHEAD reaching the content extent (segment box end), not the audio sources
        // ending. This plays THROUGH a silent tail (a ② vocal stem shorter than its box) to the segment
        // end instead of pausing the instant the last note finished (the reported premature-pause ghost).
        const end = contentEndRef.current;
        if (end > 0 && tick >= end) {
          animatingRef.current = false;
          onPlaybackEnded();
          return;
        }
      }
      animRef.current = requestAnimationFrame(animate);
    };
    animRef.current = requestAnimationFrame(animate);

    return () => {
      animatingRef.current = false;
      cancelAnimationFrame(animRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isPlaying, tempo]);

  // Reschedule playback from the CURRENT playhead with the latest tracks (read fresh via getState).
  // Web Audio sources are scheduled once at play-start, so anything that changes what plays when —
  // a seek, or a committed clip move/resize/delete — needs the graph torn down and rescheduled.
  // Natural end of playback: pause AND rest the playhead exactly at the content end. The rAF advance
  // stops a few ticks short of the true end (undershoot), which would make the next Play schedule a
  // ~0-length tail and instantly re-pause (the visible "play→pause" flicker); snapping removes that.
  const onPlaybackEnded = useCallback(() => {
    // Ignore a natural end that fires mid-seek — the seek reschedules from the new position on
    // release; stopping/snapping here would clobber the drag and silently kill playback.
    if (useAudioStore.getState().seeking) return;
    const end = contentEndTick(useProjectStore.getState().tracks);
    // The natural end is the PLAYHEAD reaching the content extent, NOT the audio sources ending. When the
    // audio finishes BEFORE the content extent (a ② vocal stem shorter than its segment box), the sources'
    // onAllEnded fires mid-segment — IGNORE it and keep playing; the rAF advances the playhead through the
    // silent tail and calls back here once it actually reaches `end`. This fixes the premature pause +
    // playhead jump the instant the last note finished.
    if (end > 0 && useProjectStore.getState().playheadTick < end - 1) return;
    animatingRef.current = false;
    setPlaying(false);
    if (end > 0) setPlayhead(end);
  }, [setPlaying, setPlayhead]);

  const rescheduleNow = useCallback((overrideTick?: number) => {
    if (!useAudioStore.getState().isPlaying) return;
    const tp = useProjectStore.getState().tempo;
    // Live position from the AUDIO clock, NOT the store playhead: under main-thread jank (e.g. loading a
    // new track) the rAF that advances the store playhead lags behind the audio, and rescheduling from a
    // STALE (behind) tick replays a sliver of already-heard audio — the "ghosting" under load. baseTick/
    // baseTime are kept current by the playhead rAF (incl. while seeking), so this stays correct.
    // EXCEPTION — a SEEK release passes the store playhead as overrideTick: a click-seek landing inside
    // a jank burst may never get a rAF frame to pin baseTickRef, and extrapolating would reschedule
    // from the stale pre-seek position. The drag's final position is authoritative there.
    const tick = overrideTick !== undefined
      ? overrideTick
      : Math.round(baseTickRef.current + playback.secondsToTicks(playback.getContextTime() - baseTimeRef.current, tp));
    if (overrideTick !== undefined) {
      baseTickRef.current = overrideTick;
      baseTimeRef.current = playback.getContextTime();
    }
    const tr = useProjectStore.getState().tracks;
    const af = useAudioStore.getState().audioFiles;
    playback.playAllTracks(tr, af, tick, tp, onPlaybackEnded).then((result) => {
      if (result === "started") {
        // Anchor the playhead to the audio's ACTUAL origin — playAllTracks' scheduleTimeOrigin (the `now`
        // it scheduled from), NOT this resolve time. playAllTracks may AWAIT an async buffer decode (e.g. a
        // lane Output just reconnected mid-playback), so getContextTime() here lands well AFTER the audio's
        // `now`; anchoring to it left the playhead lagging the audio by the decode time — the playhead
        // "jumped back" while the audio kept going. Using the schedule origin keeps them locked together.
        const origin = playback.getScheduleTimeOrigin();
        baseTickRef.current = tick;
        baseTimeRef.current = origin;
        wasPlayingRef.current = true;
      } else if (result === "empty") {
        // Nothing left to play (content deleted/moved, or everything naturally ended during the
        // scheduling awaits) → stop, don't run away. EXCEPT mid-seek: onPlaybackEnded deliberately
        // no-ops while seeking (the release reschedules from the drop position) — stopping here would
        // defeat that; the release's own rescheduleNow re-evaluates with seeking false.
        if (!useAudioStore.getState().seeking) setPlaying(false);
      }
      // result === "superseded": a NEWER reschedule already bumped the generation and now owns playback — do
      // NOTHING (don't stop, don't re-anchor). Stopping here flipped isPlaying off while the newer schedule's
      // audio kept playing — the "playback stops in place but audio keeps going" ghost, especially at
      // render-completion when several deposits reschedule in quick succession and overlap.
    });
  }, [onPlaybackEnded, setPlaying]);

  // When a seek ends during playback, reschedule audio from the new playhead position (passed as the
  // override — see rescheduleNow — so a jank-delayed click-seek can't extrapolate a stale tick).
  const prevSeekingRef = useRef(false);
  useEffect(() => {
    const wasSeeking = prevSeekingRef.current;
    prevSeekingRef.current = seeking;
    if (wasSeeking && !seeking && isPlaying) rescheduleNow(useProjectStore.getState().playheadTick);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seeking]);

  // Reschedule when a committed clip edit (move/resize/delete) changed segment timing mid-playback.
  useEffect(() => {
    if (scheduleVersion === 0) return; // initial mount — nothing scheduled yet
    contentEndRef.current = contentEndTick(useProjectStore.getState().tracks); // structural edit moved the end
    if (useAudioStore.getState().seeking) return; // an active seek reschedules on its own release
    rescheduleNow();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scheduleVersion]);

  const playPendingRef = useRef(false);
  // ② Auto-render-on-Play (S55): while the pre-play batch of changed vocal tracks bakes, the play button
  // shows a "rendering" state and a second press CANCELS it (renders can be long). abortRef flips on that
  // second press; renderingRef gates the button into cancel-mode; the local state drives the pulse class.
  const autoRenderingRef = useRef(false);
  const autoRenderAbortRef = useRef(false);
  const [autoRendering, setAutoRendering] = useState(false);
  const handleTogglePlay = async () => {
    if (isPlaying) {
      animatingRef.current = false;
      playback.stopPlayback();
      setPlaying(false);
      return;
    }
    // A second Play/Space press WHILE the pre-play auto-render runs → cancel the batch and DON'T start
    // playback. cancel_voice aborts the in-flight GPU render; renderDirtyVocals bails between/within items.
    if (autoRenderingRef.current) {
      autoRenderAbortRef.current = true;
      void invoke("cancel_voice").catch(() => {});
      return;
    }

    if (playPendingRef.current) return;
    playPendingRef.current = true;
    // S60-3: visible "preparing" state for the whole request→sound window (auto-render +
    // stretch regeneration + buffer decode) — the first play of a big project sat here for
    // seconds with zero feedback, reading as a hang (§user).
    useAudioStore.getState().setPreparing(true);

    try {
      // Bake vocal tracks whose notes/params CHANGED since their last render (skip unchanged — the v1
      // render-on-play convenience). Skipped entirely when nothing is dirty (zero added latency), or when a
      // workflow (audio-clip) render is running — its SVC node shares the Rust voice guard, so starting a
      // vocal render would cross-kill it (better to play with the existing bakes than break that render).
      const dirty = collectDirtyVocals(tempo);
      const workflowBusy = Object.values(useWorkflowStore.getState().executions).some((e) => e?.status === "running");
      if (dirty.length > 0 && !workflowBusy) {
        // S66: missing core models → the one-click dialog; don't start playback either (the
        // stale/missing bakes would mislead — the user asked to hear the CURRENT notes).
        if (!(await preflightVocalModels())) return;
        autoRenderAbortRef.current = false;
        autoRenderingRef.current = true;
        setAutoRendering(true);
        useAppStore.getState().showToast(t("vocalEditor.render.autoRendering"), "info");
        try {
          await renderDirtyVocals(dirty, tempo, t("vocalEditor.render.laneLabel"), {
            shouldCancel: () => autoRenderAbortRef.current,
          });
        } finally {
          autoRenderingRef.current = false;
          setAutoRendering(false);
        }
        if (autoRenderAbortRef.current) return; // user cancelled the batch → don't start playback
      }

      // Muno 阶段2:乐器轨音源烘焙,与人声同一「Play 前自动渲染」约定(签名判脏,空 ⇒ 零延迟)。
      // 原生 SFZ/SF2 合成不走 GPU voice 守卫,workflow 运行中也可安全烘焙(与人声块刻意不同)。
      // 有音符但没选音源的轨只提醒不阻塞——播放继续(该轨静默),用户看到提示后去轨道头选音色。
      const dirtyIns = collectDirtyInstruments(tempo);
      if (dirtyIns.length > 0) {
        const missing = instrumentTracksMissingFont();
        if (missing.length > 0) {
          useAppStore
            .getState()
            .showToast(t("soundfont.playMissingFont", { names: missing.map((m) => m.name).join("、") }), "info");
        }
        autoRenderAbortRef.current = false;
        autoRenderingRef.current = true;
        setAutoRendering(true);
        useAppStore.getState().showToast(t("vocalEditor.render.autoRendering"), "info");
        try {
          await renderDirtyInstruments(dirtyIns, tempo, {
            shouldCancel: () => autoRenderAbortRef.current,
          });
        } finally {
          autoRenderingRef.current = false;
          setAutoRendering(false);
        }
        if (autoRenderAbortRef.current) return; // user cancelled the batch → don't start playback
      }

      // Read FRESH state — a bake just deposited (it changes what plays / the content extent), and the
      // playhead / tempo may have changed during the await. tempo MUST be fresh too: the playhead-advance
      // effect uses the reactive (post-await) tempo, so scheduling with the stale closure tempo would
      // desync the audio from the playhead if the user edited BPM mid-render.
      const st = useProjectStore.getState();
      const freshTracks = st.tracks;
      const ph = st.playheadTick;
      const freshTempo = st.tempo;
      // If the playhead is at/after the end of all content, restart from the beginning (rather than
      // starting at the end with nothing to play → instant auto-pause flicker).
      const end = contentEndTick(freshTracks);
      const startTick = end > 0 && ph >= end ? 0 : ph;
      if (startTick !== ph) setPlayhead(startTick);
      const result = await playback.playAllTracks(
        freshTracks,
        useAudioStore.getState().audioFiles,
        startTick,
        freshTempo,
        onPlaybackEnded,
      );
      if (result === "started") {
        setPlaying(true);
      }
    } finally {
      playPendingRef.current = false;
      useAudioStore.getState().setPreparing(false);
    }
  };

  const handleReturnToStart = () => {
    animatingRef.current = false;
    playback.stopPlayback();
    setPlaying(false);
    setPlayhead(0);
  };

  const handleSplit = () => {
    // A LIVE selected sub-lane group takes priority: Ctrl+K slices the LANE (non-destructive, at the
    // playhead) rather than the parent segment — the main-track split gesture, constrained to the lane.
    // liveSelectedLane() guards against a stale lane (track collapsed / render cleared) silently slicing an
    // invisible lane instead of splitting the segment; it clears the stale selection and returns null.
    const lane = liveSelectedLane();
    if (lane) {
      sliceLaneGroupAtPlayhead(lane.trackId, lane.segmentId, lane.outputNodeId);
      return;
    }
    if (!selectedSegment) return;
    // ② A notes (vocal) segment now splits too: the store partitions its notes + pitchDev/param curves at the
    // playhead, giving fresh ids + rebased ticks, and SNAPS a mid-note split to that note's end (§user). The
    // baked stem is CARRIED + windowed (no re-render) via splitSegmentVocalAware, which also applies the DIRTY
    // guard (a stale bake is never windowed clean — it re-renders). audioClip + notes both go through it.
    splitSegmentVocalAware(selectedSegment.trackId, selectedSegment.segmentId, playheadRef.current, tempo);
  };

  const handleDelete = () => {
    // A LIVE selected sub-lane group takes priority: Delete removes the clicked PIECE → silence
    // (non-destructive). Same stale-lane guard as split (else it would silence an invisible lane instead
    // of deleting the segment).
    const lane = liveSelectedLane();
    if (lane) {
      deleteLanePiece(lane.trackId, lane.segmentId, lane.outputNodeId, lane.clipIndex);
      return;
    }
    // Delete the ENTIRE multi-selection (keyboard Del / toolbar button); fall back to the PRIMARY
    // selection when the multi-set is empty — selectLane anchors only selectedSegment, so a stale-lane
    // Delete must fall through to the SEGMENT (the documented contract, matching Ctrl+K) instead of a
    // silent no-op.
    const selSet = useAppStore.getState().selectedSegments;
    const primary = useAppStore.getState().selectedSegment;
    const targets = selSet.length > 0 ? selSet : primary ? [primary] : [];
    if (targets.length === 0) return;
    deleteSegments(targets.map((s) => ({ trackId: s.trackId, segmentId: s.segmentId })));
    clearSelection();
    if (useAudioStore.getState().isPlaying) useAudioStore.getState().bumpSchedule();
  };

  const togglePlayRef = useRef(handleTogglePlay);
  togglePlayRef.current = handleTogglePlay;
  const splitRef = useRef(handleSplit);
  splitRef.current = handleSplit;
  const deleteRef = useRef(handleDelete);
  deleteRef.current = handleDelete;
  const selRef = useRef(selectedSegment);
  selRef.current = selectedSegment;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Don't hijack keys while typing in a field (the BPM input, dialogs, etc.) — otherwise
      // forward-Delete in a text field would wipe the whole segment selection (no undo).
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT" || el.isContentEditable)) {
        return;
      }
      // The full-screen training page covers the DAW: space/Delete/Ctrl+K must not
      // start playback of, or edit, the invisible timeline underneath (live-test hit:
      // space during dataset preview started hidden-DAW playback on top of it).
      if (useAppStore.getState().trainingPageOpen) {
        return;
      }
      // Delete is pane-scoped: when a bottom-dock editor pane is FOCUSED (workflow OR ② vocal), Delete
      // acts THERE (ReactFlow node / vocal note, each gated on the same activePane) — never ALSO on a
      // timeline segment. ⚠ The guard is `!== "timeline"` (NOT `=== "workflow"`): with the vocal pane a
      // third value, an `=== "workflow"` check would let the timeline Delete FIRE while the vocal editor is
      // focused → it would delete the whole segment being edited (catastrophic silent loss, §9.6 blocker).
      // Ctrl+K has NO node-editor meaning so it is not gated for workflow, but IS bailed for vocal (below).
      if (useAppStore.getState().activePane !== "timeline" && e.key === "Delete") {
        return;
      }
      if (e.key === " ") {
        e.preventDefault();
        togglePlayRef.current();
      } else if (e.key === "Delete" && selRef.current) {
        deleteRef.current();
      } else if (e.key === "k" && e.ctrlKey && selRef.current) {
        // ② In the vocal pane Ctrl+K belongs to the editor (note split, future) — never slice the timeline
        // segment underneath (§9.6 "Ctrl+K 对 vocal pane 关掉").
        if (useAppStore.getState().activePane === "vocal") return;
        // Ctrl+K is ungated across panes (no node-editor meaning), but in the workflow pane only slice
        // when the SELECTED segment IS the one whose workflow is open. Selection and the open segment can
        // diverge (click another clip, then click back into the panel) — without this guard Ctrl+K would
        // silently slice a different, possibly off-screen segment the panel isn't even showing, and the
        // split wouldn't be Ctrl+Z-undoable while the pane owns undo. Opening a workflow selects that same
        // segment, so the normal "slice the segment I'm editing" case still works.
        if (
          useAppStore.getState().activePane === "workflow" &&
          selRef.current.segmentId !== useAppStore.getState().workflowSegmentId
        ) {
          return;
        }
        e.preventDefault();
        splitRef.current();
      } else if (e.ctrlKey && !e.shiftKey && !e.altKey && ["c", "x", "v"].includes(e.key.toLowerCase())) {
        // S61 arrangement clipboard. Timeline-pane ONLY: the vocal editor owns note copy/paste under
        // activePane === "vocal" (VocalEditor keydown), and the workflow pane deliberately has no
        // clipboard — same `!== "timeline"` posture as Delete above (never fire under another pane).
        // toLowerCase: CapsLock reports "C"/"X"/"V" (the VocalEditor twin already normalizes).
        if (useAppStore.getState().activePane !== "timeline") return;
        e.preventDefault();
        const k = e.key.toLowerCase();
        if (k === "c") copySelectedSegments();
        else if (k === "x") cutSelectedSegments();
        else pasteWithFeedback();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className="toolbar">
      {/* 左侧区域：吸附控制 + 新建轨道 */}
      <div className="toolbar-left">
        <div className="toolbar-section snap-section">
          <button
            className={`snap-toggle ${snapSegments ? "active" : ""}`}
            onClick={toggleSnapSegments}
            title={t("toolbar.snapClipTip")}
          >
            <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true">
              <path fill="currentColor" d="M3 7v6a9 9 0 0 0 18 0V7h-4v6a5 5 0 0 1-10 0V7z M3 3h4v4H3z M17 3h4v4h-4z" />
            </svg>
          </button>
          <button
            className={`snap-toggle ${snapPlayhead ? "active" : ""}`}
            onClick={toggleSnapPlayhead}
            title={t("toolbar.snapPlayheadTip")}
          >
            <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true">
              <path fill="currentColor" d="M6 3h12l-6 7z M11 9h2v12h-2z" />
            </svg>
          </button>
          <Dropdown
            className="snap-grid-dropdown"
            value={snapGridStep}
            options={SNAP_GRID_OPTIONS.map(({ value, labelKey }) => ({ value, label: t(labelKey) }))}
            onChange={setSnapGridStep}
          />
        </div>

        <div className="toolbar-divider" />


      </div>

      {/* 中心区域：播放控制 + 概览图（视觉焦点） */}
      <div className="toolbar-center">
        <div className="toolbar-section transport">
          <button className="transport-btn transport-btn-sm" onClick={handleReturnToStart} title="回到开头">
            <span className="transport-icon icon-return" />
          </button>
          <button
            className={`transport-btn transport-btn-lg play ${isPlaying ? "playing" : ""} ${autoRendering ? "rendering" : ""} ${preparing && !autoRendering ? "preparing" : ""}`}
            onClick={handleTogglePlay}
            title={autoRendering ? t("vocalEditor.render.autoRenderingCancel") : preparing ? t("transport.preparing") : isPlaying ? "暂停 (空格)" : "播放 (空格)"}
          >
            <span className={`transport-icon ${isPlaying ? "icon-pause" : "icon-play"}`} />
          </button>
        </div>

        <div className="toolbar-overlay">
          <OverviewMap />
        </div>
      </div>

      {/* 右侧区域：BPM + 拍号 + 调性 + 位置 + 工具按钮 */}
      <div className="toolbar-right">
        <div className="toolbar-section tempo-section">
          <label className="toolbar-label">{t("toolbar.bpm")}</label>
          <input
            type="number"
            className="tempo-input mono"
            value={tempo}
            min={20}
            max={400}
            step={1}
            onFocus={() => { useHistoryStore.getState().beginTransaction(); useProjectStore.getState().beginTempoScale(); }}
            onBlur={() => { useProjectStore.getState().endTempoScale(); useHistoryStore.getState().commitTransaction(); }}
            onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }}
            onChange={(e) => setTempo(Number(e.target.value))}
          />
        </div>

        <div className="toolbar-section time-sig-clickable" onClick={() => setTimeSigOpen(true)} title={t("toolbar.timeSignature")}>
          <span className="mono time-sig-display">{timeSignature[0]}/{timeSignature[1]}</span>
        </div>

        <div className="toolbar-section key-selector" onClick={() => setKeyOpen(true)} title={t("toolbar.key")}>
          {selectedKey ? (
            <span className="key-display">{selectedKey}</span>
          ) : (
            <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">
              <path d="M3 8a5 5 0 0 1 10 0M8 8v5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" fill="none" />
              <circle cx="8" cy="8" r="1.5" fill="currentColor" />
            </svg>
          )}
        </div>

        <div className="toolbar-divider" />

        <div className="toolbar-section position-section">
          <label className="toolbar-label">{t("toolbar.position")}</label>
          <span ref={positionDisplayRef} className="mono position-display">
            {formatPosition(playheadRef.current, timeAxis)}
          </span>
        </div>

        <div className="toolbar-divider" />

        {/* 工具按钮：撤销历史 / 快捷键 / 主题 */}
        <div className="toolbar-section utility-section">
          <button className="toolbar-btn icon-btn" title={t("history.panelTitle")} onClick={() => setHistoryOpen(true)}>
            <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
              <path d="M8 1.5a6.5 6.5 0 1 1-6.5 6.5" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
              <path d="M1.5 8V3.5M1.5 3.5H6" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
          <button className="toolbar-btn icon-btn" title={t("titlebar.shortcuts")} onClick={() => setShortcutsOpen(true)}>
            <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
              <rect x="1" y="2" width="14" height="11" rx="2" fill="none" stroke="currentColor" strokeWidth="1.2" />
              <path d="M4 12.5v-2M4 8.5v-1M12 12.5v-6M10 9.5v-1M10 13v-.4M2.5 4.5h3M11.5 4.5h2" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
            </svg>
          </button>
          <button className="toolbar-btn icon-btn" title={t("titlebar.skin")} onClick={(e) => {
            const r = e.currentTarget.getBoundingClientRect();
            setSkinPos({ top: r.bottom + 6, right: window.innerWidth - r.right });
            setSkinOpen((v) => !v);
          }}>
            <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
              <circle cx="5" cy="4" r="1.6" fill="#fdba74" />
              <circle cx="9.6" cy="3.2" r="1.6" fill="#4ade80" />
              <circle cx="12.6" cy="6.4" r="1.6" fill="#60a5fa" />
              <circle cx="11.6" cy="10.4" r="1.6" fill="#a78bfa" />
              <circle cx="7.4" cy="11.4" r="1.6" fill="#f87171" />
            </svg>
          </button>
        </div>
      </div>

      {/* 弹窗集合 */}
      {skinOpen && (
        <div className="skin-popover" style={skinPos ?? undefined}>
          <div className="skin-popover-title">{t("titlebar.themeTitle")}</div>
          <div className="theme-grid">
            {THEMES.map((item, idx) => (
              <button
                key={idx}
                className={`theme-item ${item.theme === theme && item.skin === skin ? "active" : ""}`}
                onClick={() => {
                  setTheme(item.theme);
                  setSkin(item.skin);
                  setSkinOpen(false);
                }}
              >
                <span className="theme-color-dot" style={{ backgroundColor: item.color }} />
                <span className="theme-label">{item.label}</span>
              </button>
            ))}
          </div>
        </div>
      )}
      
      {timeSigOpen && (
        <>
          <div className="popover-backdrop" onClick={() => setTimeSigOpen(false)} />
          <div className="time-sig-popover">
            <div className="popover-title">{t("toolbar.timeSignature")}</div>
            <div className="time-sig-editor">
              <div className="time-sig-row">
                <label>分子</label>
                <div className="time-sig-options">
                  {TS_NUM_OPTIONS.slice(0, 8).map(opt => (
                    <button
                      key={opt.value}
                      className={`time-sig-option ${timeSignature[0] === opt.value ? "active" : ""}`}
                      onClick={() => {
                        setTimeSignature(opt.value, timeSignature[1]);
                        setTimeSigOpen(false);
                      }}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
              </div>
              <div className="time-sig-row">
                <label>分母</label>
                <div className="time-sig-options">
                  {TS_DEN_OPTIONS.map(opt => (
                    <button
                      key={opt.value}
                      className={`time-sig-option ${timeSignature[1] === opt.value ? "active" : ""}`}
                      onClick={() => {
                        setTimeSignature(timeSignature[0], opt.value);
                        setTimeSigOpen(false);
                      }}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </>
      )}
      
      {keyOpen && (
        <>
          <div className="popover-backdrop" onClick={() => setKeyOpen(false)} />
          <div className="key-popover">
            <div className="popover-title">{t("toolbar.key")}</div>
            <div className="key-section">
              <div className="key-section-title">{t("toolbar.keyMajor")}</div>
              <div className="key-grid">
                {["C", "G", "D", "A", "E", "B", "F♯", "C♯", "F", "B♭", "E♭", "A♭"].map(key => (
                  <button
                    key={key}
                    className={`key-option ${selectedKey === key ? "active" : ""}`}
                    onClick={() => {
                      setSelectedKey(key);
                      setKeyOpen(false);
                    }}
                  >
                    {key}
                  </button>
                ))}
              </div>
            </div>
            <div className="key-section">
              <div className="key-section-title">{t("toolbar.keyMinor")}</div>
              <div className="key-grid">
                {["Am", "Em", "Bm", "F♯m", "C♯m", "G♯m", "D♯m", "A♯m", "Dm", "Gm", "Cm", "Fm"].map(key => (
                  <button
                    key={key}
                    className={`key-option ${selectedKey === key ? "active" : ""}`}
                    onClick={() => {
                      setSelectedKey(key);
                      setKeyOpen(false);
                    }}
                  >
                    {key}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </>
      )}
      
      {shortcutsOpen && <ShortcutsDialog onClose={() => setShortcutsOpen(false)} />}
      {historyOpen && <HistoryPanel onClose={() => setHistoryOpen(false)} />}
    </div>
  );
}

// bar:beat:sub via the meter authority — identical to the old fixed 480-based math for 4/4, but a 6/8 bar
// now reads 6 beats of 240 ticks. `sub` is a 0-based quarter-of-beat. Shared with the ② vocal-editor playhead.
const formatPosition = (tick: number, axis: TimeAxis): string => formatBarBeat(axis, tick);
