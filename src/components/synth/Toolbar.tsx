import { useEffect, useRef, useCallback } from "react";
import { useProjectStore } from "../../store/project";
import { useAudioStore } from "../../store/audio";
import { useAppStore } from "../../store/app";
import { useHistoryStore } from "../../store/history";
import { useTranslation } from "react-i18next";
import * as playback from "../../lib/audio/playback";
import { TICKS_PER_BEAT } from "../../lib/constants";
import { contentEndTick } from "../../lib/trackLayout";
import { sliceLaneGroupAtPlayhead, deleteLanePiece, liveSelectedLane } from "../../lib/laneEdit";
import { OverviewMap } from "./OverviewMap";
import "./Toolbar.css";

export function Toolbar() {
  const { t } = useTranslation();
  const { tempo, setTempo, playheadTick, setPlayhead, timeSignature, tracks } =
    useProjectStore();
  const { isPlaying, setPlaying, audioFiles, seeking, scheduleVersion } = useAudioStore();
  const { selectedSegment, clearSelection, snapSegments, snapPlayhead, toggleSnapSegments, toggleSnapPlayhead } = useAppStore();
  const { splitSegment, deleteSegments } = useProjectStore();
  const animRef = useRef<number>(0);
  const baseTickRef = useRef(0);
  const baseTimeRef = useRef(0);
  const animatingRef = useRef(false);
  const wasPlayingRef = useRef(false);

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

    const animate = () => {
      if (!animatingRef.current) return;
      if (useAudioStore.getState().seeking) {
        // The user is dragging the playhead — pin the baseline to the dragged position so
        // the rAF doesn't clobber it; audio reschedules from here once the drag is released.
        baseTickRef.current = useProjectStore.getState().playheadTick;
        baseTimeRef.current = playback.getContextTime();
      } else {
        const elapsed = playback.getContextTime() - baseTimeRef.current;
        setPlayhead(Math.round(baseTickRef.current + playback.secondsToTicks(elapsed, tempo)));
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
    setPlaying(false);
    const end = contentEndTick(useProjectStore.getState().tracks);
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
    if (useAudioStore.getState().seeking) return; // an active seek reschedules on its own release
    rescheduleNow();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scheduleVersion]);

  const playPendingRef = useRef(false);
  const handleTogglePlay = async () => {
    if (isPlaying) {
      animatingRef.current = false;
      playback.stopPlayback();
      setPlaying(false);
      return;
    }

    if (playPendingRef.current) return;
    playPendingRef.current = true;

    try {
      // If the playhead is at/after the end of all content, restart from the beginning (rather than
      // starting at the end with nothing to play → instant auto-pause flicker).
      const end = contentEndTick(tracks);
      const startTick = end > 0 && playheadTick >= end ? 0 : playheadTick;
      if (startTick !== playheadTick) setPlayhead(startTick);
      const result = await playback.playAllTracks(
        tracks,
        audioFiles,
        startTick,
        tempo,
        onPlaybackEnded,
      );
      if (result === "started") {
        setPlaying(true);
      }
    } finally {
      playPendingRef.current = false;
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
    splitSegment(selectedSegment.trackId, selectedSegment.segmentId, playheadTick);
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
      // Delete is pane-scoped: when the workflow editor pane is FOCUSED, Delete removes a NODE there
      // (ReactFlow, gated on the same activePane), so don't ALSO delete a timeline segment. Ctrl+K, by
      // contrast, has NO meaning in the node editor, so it is NEVER gated — it always slices the selected
      // timeline segment at the playhead, even while the panel is focused (fixes the reported "Ctrl+K
      // stops working when the workflow is open" dead-zone). Focus-based, not just "open": the editor is a
      // persistent bottom panel co-visible with the tracks.
      if (useAppStore.getState().activePane === "workflow" && e.key === "Delete") {
        return;
      }
      if (e.key === " ") {
        e.preventDefault();
        togglePlayRef.current();
      } else if (e.key === "Delete" && selRef.current) {
        deleteRef.current();
      } else if (e.key === "k" && e.ctrlKey && selRef.current) {
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
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className="toolbar">
      <div className="toolbar-section transport">
        <button
          className="transport-btn"
          onClick={handleReturnToStart}
        >
          <span className="transport-icon icon-return" />
        </button>
        <button
          className={`transport-btn play ${isPlaying ? "playing" : ""}`}
          onClick={handleTogglePlay}
        >
          {isPlaying
            ? <span className="transport-icon icon-pause" />
            : <span className="transport-icon icon-play" />
          }
        </button>
      </div>

      <OverviewMap />

      <div className="toolbar-divider" />

      <div className="toolbar-section tempo-section">
        <label className="toolbar-label">{t("toolbar.bpm")}</label>
        <input
          type="number"
          className="tempo-input mono"
          value={tempo}
          min={20}
          max={400}
          step={1}
          // BPM typing/spinning fires onChange repeatedly and rescales every clip each time — coalesce
          // the whole edit-session into ONE undo step (focus → begin, blur/Enter → commit).
          onFocus={() => useHistoryStore.getState().beginTransaction()}
          onBlur={() => useHistoryStore.getState().commitTransaction()}
          onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }}
          onChange={(e) => setTempo(Number(e.target.value))}
        />
      </div>

      <div className="toolbar-section time-sig">
        <span className="mono time-display">
          {timeSignature[0]}/{timeSignature[1]}
        </span>
      </div>

      <div className="toolbar-divider" />

      <div className="toolbar-section position-section">
        <label className="toolbar-label">{t("toolbar.position")}</label>
        <span className="mono position-display">
          {formatPosition(playheadTick, timeSignature)}
        </span>
      </div>

      <div className="toolbar-divider" />

      <div className="toolbar-section snap-section">
        <label className="toolbar-label">{t("toolbar.snap")}</label>
        <button
          className={`snap-toggle ${snapSegments ? "active" : ""}`}
          onClick={toggleSnapSegments}
          aria-label={t("toolbar.snapClipTip")}
        >
          {/* magnet — clip snapping */}
          <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true">
            <path fill="currentColor" d="M3 7v6a9 9 0 0 0 18 0V7h-4v6a5 5 0 0 1-10 0V7z M3 3h4v4H3z M17 3h4v4h-4z" />
          </svg>
        </button>
        <button
          className={`snap-toggle ${snapPlayhead ? "active" : ""}`}
          onClick={toggleSnapPlayhead}
          aria-label={t("toolbar.snapPlayheadTip")}
        >
          {/* playhead marker — playhead snapping */}
          <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true">
            <path fill="currentColor" d="M6 3h12l-6 7z M11 9h2v12h-2z" />
          </svg>
        </button>
      </div>

      <div className="toolbar-divider" />

      <div className="toolbar-section edit-section">
        <button
          className="toolbar-btn"
          onClick={handleSplit}
          disabled={!selectedSegment}
        >
          {t("toolbar.split")}
        </button>
        <button
          className="toolbar-btn"
          onClick={handleDelete}
          disabled={!selectedSegment}
        >
          {t("toolbar.delete")}
        </button>
      </div>

      <div className="toolbar-spacer" />
    </div>
  );
}

function formatPosition(tick: number, timeSig: [number, number]): string {
  const ticksPerBar = TICKS_PER_BEAT * timeSig[0];
  const bar = Math.floor(tick / ticksPerBar) + 1;
  const beat = Math.floor((tick % ticksPerBar) / TICKS_PER_BEAT) + 1;
  const sub = Math.floor(((tick % ticksPerBar) % TICKS_PER_BEAT) / (TICKS_PER_BEAT / 4));
  return `${bar}:${beat}:${sub.toString().padStart(2, "0")}`;
}
