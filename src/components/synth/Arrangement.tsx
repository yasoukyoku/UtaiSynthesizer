import { useRef, useEffect, useCallback, useState } from "react";
import { useProjectStore, useTimeAxis } from "../../store/project";
import type { TimeAxis } from "../../lib/timeAxis";
import { useAppStore } from "../../store/app";
import { useAudioStore } from "../../store/audio";
import { useHistoryStore } from "../../store/history";
import { useTranslation } from "react-i18next";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { TICKS_PER_BEAT, PIXELS_PER_TICK, TRACK_HEADER_HEIGHT, LANE_HEIGHT, LANE_GROUP_BAR_HEIGHT, TRACK_ADD_FOOTER, AUDIO_EXT_RE } from "../../lib/constants";
import { computeTrackYOffsets, computeTrackHeight, computeTotalTracksHeight, findTrackAtY, getLanes, getLaneLayout, laneRowAtY, isLaneRowMuted, laneControlFor, segmentPlaysLanes, segmentLaneSumPeaks, laneSumSig } from "../../lib/trackLayout";
import { importAudioToNewTrack, importAudioToExistingTrack, probeAudioDuration, DEFAULT_DURATION_MS } from "../../lib/audio/import";
import { durationMsToTicks } from "../../lib/audio/playback";
import {
  laneGroupId, laneVisiblePieces, clipIndexAtMs, trimClip, isWholeClips, ticksToMs, laneOpsSig,
  materializeClips, laneRowKey, laneLabelParts, laneReachesSeam, flooredDurationTicks, MIN_LANE_PIECE_MS,
} from "../../lib/audio/laneOps";
import { drawWaveform, beginWaveformFrame, peaksSignature } from "../../lib/waveformCache";
import { collectSnapTicks, snapTick, snapMovedStart, SNAP_PX } from "../../lib/snapping";
import { trackRgb, rgba, ACCENT, ACCENT_RGB, LANE_COLORS, SELECTION_GLOW_RGB } from "../../lib/trackColors";
import { drawBeatGrid, drawPlayhead, SEPARATOR_RGB } from "../../lib/canvasDraw";
import { ContextMenu, type MenuItem } from "../common/ContextMenu";
import { sliceLaneGroupAtPlayhead, deleteLanePiece } from "../../lib/laneEdit";
import type { Track, Segment, LaneClip } from "../../types/project";
import "./Arrangement.css";

const EDGE_ZONE = 6;
// Slack on the viewport-cull bounds: a selected segment's outline has an ~8px shadow-blur glow that
// extends past its rect, so cull a touch beyond the edges to avoid the glow popping off early.
const CULL_PAD = 12;
const AUTOSCROLL_ZONE = 48;
const AUTOSCROLL_SPEED = 14; // px/frame at the edge; ramps up to 2× when the pointer goes past the edge
// During drag-import, a cursor within this many px of a track boundary creates a NEW track there
// (vs. inserting into the track body) — mirrors the track-header boundary "add track" affordance.
const DROP_BOUNDARY_PX = 7;

/** Where a drag/drop lands. "insert" = add a segment onto the existing audio track at `index`;
 *  "new" = create a new track at position `index` (the dragged spot); "none" = not a drop target. */
interface Placement {
  target: "insert" | "new" | "none";
  index: number;
  tick: number;
}

/** Insert index (0..tracks.length) of the track boundary within `pad` px of `contentY`, else null.
 *  Boundaries are each track's top edge (offsets[i]) and the bottom of the stack (totalH). */
function boundaryIndexAt(contentY: number, trks: Track[], offsets: number[], totalH: number, pad: number): number | null {
  for (let i = 0; i <= trks.length; i++) {
    const by = i < trks.length ? offsets[i]! : totalH;
    if (Math.abs(contentY - by) <= pad) return i;
  }
  return null;
}

type DragMode = null | "playhead" | "move" | "resizeL" | "resizeR" | "laneResizeL" | "laneResizeR" | "laneBoundary";

interface DragState {
  mode: DragMode;
  trackIdx: number;
  segId: string;
  startMouseX: number;
  startMouseY: number;
  /** Content-space X (clientX - canvasLeft + scrollX) at grab — so move/resize follow scroll
   *  changes during edge auto-scroll, not just raw pointer movement. */
  startContentX: number;
  origStartTick: number;
  origDurationTicks: number;
  origOffsetMs: number;
  /** All segments moving together (1 for single, N for a multi-selection drag). */
  moving: { trackId: string; segId: string; origStartTick: number }[];
  /** Set once the pointer actually moves, to distinguish a click from a drag. */
  dragged: boolean;
  /** On a plain click (no drag) of an already-multi-selected segment, collapse to just this one. */
  collapseTo: { trackId: string; segId: string } | null;
  /** Sub-lane trim (laneResizeL/R): the Output-node group + clip index being trimmed + the materialized
   *  clip list captured at grab (each frame re-trims from this stable base, like origStartTick). */
  laneGroup?: string;
  laneClipIndex?: number;
  origClips?: LaneClip[];
  /** Sub-lane SHARED-boundary drag (laneBoundary): the two touching pieces at a coincident slice edge.
   *  The drag DIRECTION picks which piece shrinks (drag right past the boundary → shrink the RIGHT piece's
   *  front; left → shrink the LEFT piece's back), so you never have to hover-target the exact side. */
  laneBoundaryMs?: number;
  laneLeftClipIndex?: number;
  laneRightClipIndex?: number;
  /** Mixer/mute entries COPIED onto tracks by cross-track hops during THIS gesture — reverted at
   *  release for every track the segment merely passed over (else a round-trip drag that visually did
   *  nothing leaves sig-visible residue → a phantom undo step + a spurious unsaved-changes prompt). */
  mixCopies?: { trackId: string; controlKeys: string[]; muteKeys: string[] }[];
}

interface CtxState {
  x: number;
  y: number;
  trackIdx: number;
  segId: string | null;
  /** Present when the right-click landed on a sub-lane row → the menu offers lane slice/delete. */
  lane?: { outputNodeId: string; clipIndex: number } | null;
}

export function Arrangement() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // Content fields drive React re-render + redraw. scrollX/scrollY/zoom (app) and playheadTick
  // (project) are deliberately NOT subscribed via React — they live in refs updated by store
  // subscriptions and drive the canvas imperatively (see the subscription effect below), so
  // scrolling and playback don't re-render this component or its DAW siblings.
  const tracks = useProjectStore((s) => s.tracks);
  const timeSignature = useProjectStore((s) => s.timeSignature); // scalar kept for the static-layer cache key
  const timeAxis = useTimeAxis();
  const tempo = useProjectStore((s) => s.tempo);
  const setPlayhead = useProjectStore((s) => s.setPlayhead);
  const updateTrack = useProjectStore((s) => s.updateTrack);
  const splitSegment = useProjectStore((s) => s.splitSegment);
  const deleteSegment = useProjectStore((s) => s.deleteSegment);
  const updateSegmentLaneOps = useProjectStore((s) => s.updateSegmentLaneOps);
  const openWorkflow = useAppStore((s) => s.openWorkflow);
  const selectedSegments = useAppStore((s) => s.selectedSegments);
  const selectedLane = useAppStore((s) => s.selectedLane);
  const selectSegment = useAppStore((s) => s.selectSegment);
  const selectLane = useAppStore((s) => s.selectLane);
  const toggleSegment = useAppStore((s) => s.toggleSegment);
  const clearSelection = useAppStore((s) => s.clearSelection);
  const audioFiles = useAudioStore((s) => s.audioFiles);
  const loadingPaths = useAudioStore((s) => s.loadingPaths);
  const { t } = useTranslation();
  const [cursor, setCursor] = useState("default");
  const [dragOver, setDragOver] = useState(false);
  const [ctxMenu, setCtxMenu] = useState<CtxState | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const mouseXRef = useRef(-9999);
  const mouseClientXRef = useRef(0);
  const mouseClientYRef = useRef(0);
  const autoScrollRef = useRef<number>(0);
  const drawRef = useRef(() => {});
  // Tick of the clip edge the active drag last snapped to (drawn as a dashed guide). null = no snap.
  const snapHintRef = useRef<number | null>(null);

  // OS drag-import state (driven by Tauri drag events, not React) — the ghost preview rect,
  // the audio paths being dragged, and their probed durations (path → ms) for ghost sizing.
  const ghostRef = useRef<Placement | null>(null);
  const dragPathsRef = useRef<string[]>([]);
  const dragDurationsRef = useRef<Record<string, number>>({});
  const lastGhostInsertRef = useRef(""); // guards redundant store writes for the shared placeholder gap
  const dragPosRef = useRef<{ x: number; y: number } | null>(null); // last OS-drag PHYSICAL position
  const dragScrollRef = useRef(0); // rAF id for the OS-drag edge auto-scroll
  const headerElRef = useRef<Element | null>(null); // cached .daw-tracklist-wrap (re-queried if detached)

  // Content refs (synced each render — sourced from React-subscribed values).
  const tracksRef = useRef(tracks);
  tracksRef.current = tracks;
  const tempoRef = useRef(tempo);
  tempoRef.current = tempo;

  // View refs — driven imperatively by store subscriptions, NOT by React render.
  const scrollXRef = useRef(useAppStore.getState().scrollX);
  const scrollYRef = useRef(useAppStore.getState().scrollY);
  const zoomRef = useRef(useAppStore.getState().zoom);
  const vZoomRef = useRef(useAppStore.getState().vZoom);
  const playheadRef = useRef(useProjectStore.getState().playheadTick);
  const pptRef = useRef(PIXELS_PER_TICK * zoomRef.current);

  // rAF-coalesced redraw so scroll + playhead changes in the same frame collapse to one draw.
  const redrawRafRef = useRef(0);
  const requestRedraw = useCallback(() => {
    if (redrawRafRef.current) return;
    redrawRafRef.current = requestAnimationFrame(() => {
      redrawRafRef.current = 0;
      drawRef.current();
    });
  }, []);

  // Subscribe to scroll/zoom (app store) + playhead (project store); update refs and repaint
  // the canvas imperatively — without re-rendering React.
  useEffect(() => {
    const unsubApp = useAppStore.subscribe((s) => {
      let changed = false;
      if (s.scrollX !== scrollXRef.current) { scrollXRef.current = s.scrollX; changed = true; }
      if (s.scrollY !== scrollYRef.current) { scrollYRef.current = s.scrollY; changed = true; }
      if (s.zoom !== zoomRef.current) {
        zoomRef.current = s.zoom;
        pptRef.current = PIXELS_PER_TICK * s.zoom;
        changed = true;
      }
      if (s.vZoom !== vZoomRef.current) { vZoomRef.current = s.vZoom; changed = true; }
      if (changed) requestRedraw();
    });
    const unsubProj = useProjectStore.subscribe((s) => {
      if (s.playheadTick !== playheadRef.current) {
        playheadRef.current = s.playheadTick;
        requestRedraw();
      }
    });
    return () => { unsubApp(); unsubProj(); cancelAnimationFrame(redrawRafRef.current); };
  }, [requestRedraw]);

  const canvasToTick = useCallback(
    (clientX: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return 0;
      const rect = canvas.getBoundingClientRect();
      return Math.max(0, Math.round((clientX - rect.left + scrollXRef.current) / pptRef.current));
    },
    [],
  );

  // Playhead tick at a pointer X — snapped to clip edges when playhead-snap is enabled.
  const playheadTickAt = useCallback(
    (clientX: number) => {
      let tick = canvasToTick(clientX);
      if (useAppStore.getState().snapPlayhead) {
        tick = snapTick(tick, collectSnapTicks(tracksRef.current), SNAP_PX / pptRef.current);
      }
      return Math.max(0, tick);
    },
    [canvasToTick],
  );

  const hitTest = useCallback(
    (clientX: number, clientY: number):
      { trackIdx: number; segId: string; zone: "body" | "left" | "right"; lane?: { group: string; clipIndex: number } } | null => {
      const canvas = canvasRef.current;
      if (!canvas) return null;
      const rect = canvas.getBoundingClientRect();
      const x = clientX - rect.left + scrollXRef.current;
      const y = clientY - rect.top + scrollYRef.current;
      const ppt = pptRef.current;
      const scale = vZoomRef.current;
      const tp = tempoRef.current;
      const yOffsets = computeTrackYOffsets(tracks, scale);
      const headerH = TRACK_HEADER_HEIGHT * scale;

      for (let i = 0; i < tracks.length; i++) {
        const track = tracks[i];
        if (!track) continue;
        const trackY = yOffsets[i]!;
        const trackH = computeTrackHeight(track, scale);
        if (y < trackY || y > trackY + trackH) continue;

        // Header row → the main segment (move / resize-edge).
        if (y <= trackY + headerH) {
          for (let si = track.segments.length - 1; si >= 0; si--) {
            const seg = track.segments[si]!;
            // Loading placeholders are non-interactive — not selectable, movable, resizable, or
            // splittable until decode finishes (finalizeSegment is keyed to this track + a
            // user resize/split during decode would be clobbered or strand the segment).
            if (seg.loading) continue;
            const sx = seg.startTick * ppt;
            const sw = seg.durationTicks * ppt;
            if (x >= sx && x <= sx + sw) {
              if (x - sx < EDGE_ZONE) return { trackIdx: i, segId: seg.id, zone: "left" };
              if (sx + sw - x < EDGE_ZONE) return { trackIdx: i, segId: seg.id, zone: "right" };
              return { trackIdx: i, segId: seg.id, zone: "body" };
            }
          }
          return null;
        }

        // Sub-lane row (only when the track is EXPANDED) → non-destructive slice/edge-stretch on a lane
        // GROUP. The row index maps to a distinct laneId; the hit segment's matching Output lane gives its
        // group (outputNodeId) + the clicked piece; an edge zone on that piece is a trim, the body a select.
        if (!track.expanded) return null;
        const lanes = getLanes(track);
        // Row lookup through the SHARED lane layout (group bars shift the rows); a Y on a group BAR
        // (or past the last row) is claimed by the track but interactive-empty.
        const li = laneRowAtY(track, (y - trackY) / scale);
        const laneInfo = lanes[li];
        if (!laneInfo) return null;
        for (let si = track.segments.length - 1; si >= 0; si--) {
          const seg = track.segments[si]!;
          if (seg.loading || seg.content.type !== "audioClip") continue;
          const sx = seg.startTick * ppt;
          const sw = seg.durationTicks * ppt;
          if (x < sx || x > sx + sw) continue;
          // Members: a merged visual row (equivalent 组s) resolves to whichever member THIS segment
          // carries — the clicked piece then edits/selects its own 组, exactly as unmerged.
          const out = seg.processedOutputs?.find((o) => !o.loading && laneInfo.members.some((m) => m.rowKey === laneRowKey(o)));
          if (!out) return null; // the row exists for the track, but THIS segment has no such lane → empty
          const group = laneGroupId(out);
          const stemDurMs = seg.content.totalDurationMs;
          const stored = seg.laneOps?.[group];
          const xTick = x / ppt;
          const clickMs = seg.content.offsetMs + ticksToMs(xTick - seg.startTick, tp);
          const pieces = laneVisiblePieces(seg, stored, stemDurMs, tp);
          const clipAt = (pc: { startMs: number; endMs: number }) => clipIndexAtMs(stored, stemDurMs, (pc.startMs + pc.endMs) / 2);
          // 1) Piece whose BODY the cursor is over → its near edge. Containment disambiguates a shared slice
          //    boundary by the SIDE the cursor is on (cursor left of the cut → left piece's RIGHT edge; right
          //    of the cut → right piece's LEFT edge), so you always grab the piece you're pointing at — fixes
          //    "wanted the right piece but it grabbed the left piece's pinned edge and wouldn't drag".
          for (const pc of pieces) {
            const leftPx = pc.startTick * ppt, rightPx = pc.endTick * ppt;
            if (x < leftPx || x > rightPx) continue;
            const clipIndex = clipAt(pc);
            const dl = x - leftPx, dr = rightPx - x;
            // Nearer edge wins so even a narrow piece exposes BOTH edges (left half → trim start, right → end).
            if (dl <= EDGE_ZONE && dl <= dr) return { trackIdx: i, segId: seg.id, zone: "left", lane: { group, clipIndex } };
            if (dr <= EDGE_ZONE) return { trackIdx: i, segId: seg.id, zone: "right", lane: { group, clipIndex } };
            return { trackIdx: i, segId: seg.id, zone: "body", lane: { group, clipIndex } };
          }
          // 2) In a silent gap but within reach of a piece edge → grab it to stretch that piece into the gap.
          for (const pc of pieces) {
            const leftPx = pc.startTick * ppt, rightPx = pc.endTick * ppt;
            if (Math.abs(x - leftPx) <= EDGE_ZONE) return { trackIdx: i, segId: seg.id, zone: "left", lane: { group, clipIndex: clipAt(pc) } };
            if (Math.abs(x - rightPx) <= EDGE_ZONE) return { trackIdx: i, segId: seg.id, zone: "right", lane: { group, clipIndex: clipAt(pc) } };
          }
          // 3) Deep in a gap → body-select the nearest piece for context (Ctrl+K/Delete still target it).
          return { trackIdx: i, segId: seg.id, zone: "body", lane: { group, clipIndex: clipIndexAtMs(stored, stemDurMs, clickMs) } };
        }
        return null;
      }
      return null;
    },
    [tracks],
  );

  const hitTrackIdx = useCallback(
    (clientY: number): number => {
      const canvas = canvasRef.current;
      if (!canvas) return -1;
      const rect = canvas.getBoundingClientRect();
      const y = clientY - rect.top + scrollYRef.current;
      return findTrackAtY(computeTrackYOffsets(tracks, vZoomRef.current), y);
    },
    [tracks],
  );

  // Map an OS-drag PHYSICAL position (devicePixelRatio applied) to a placement, with a UNIFIED model
  // so the drop never jumps: cursor Y picks the track (within a boundary band → a NEW track there; in
  // a track body → insert onto it, or a new track above/below on collision); cursor X picks the tick
  // (clamped to 0 when the cursor is left of the canvas / over the header column — that's how you land
  // on bar 1). Horizontal scrolling during the drag is handled by the edge auto-scroll, so the header
  // column is no longer a special "always new track" case. Reads refs only → stable across renders.
  const placementFromPhysical = useCallback((position: { x: number; y: number }): Placement => {
    const canvas = canvasRef.current;
    if (!canvas) return { target: "none", index: -1, tick: 0 };
    const trks = tracksRef.current;
    const scale = vZoomRef.current;
    const offsets = computeTrackYOffsets(trks, scale);
    const totalH = computeTotalTracksHeight(trks, scale);
    const lx = position.x / devicePixelRatio;
    const ly = position.y / devicePixelRatio;
    const rect = canvas.getBoundingClientRect();

    // Drop band = the rows' vertical extent (the canvas and the header column are vertically aligned).
    // Accept the canvas X-range AND the header column to its left, so dragging past the left edge
    // still drops (at tick 0). Outside that band → not a target.
    const overCanvasX = lx >= rect.left && lx <= rect.right;
    if (!headerElRef.current || !headerElRef.current.isConnected) {
      headerElRef.current = document.querySelector(".daw-tracklist-wrap");
    }
    const hr = headerElRef.current?.getBoundingClientRect();
    const overHeaderX = !!hr && lx >= hr.left && lx < rect.left;
    if (ly < rect.top || ly > rect.bottom || (!overCanvasX && !overHeaderX)) {
      return { target: "none", index: -1, tick: 0 };
    }

    let tick = overCanvasX
      ? Math.max(0, Math.round((lx - rect.left + scrollXRef.current) / pptRef.current))
      : 0; // left of the canvas (header column) → bar 1
    const contentY = ly - rect.top + scrollYRef.current;

    // Dragged clip duration (primary file) — used for both the collision test and edge snapping.
    const primary = dragPathsRef.current[0];
    const durMs = (primary ? dragDurationsRef.current[primary] : undefined) ?? DEFAULT_DURATION_MS;
    const durTicks = flooredDurationTicks(durMs, tempoRef.current);

    // Clip snapping (when enabled): snap the new clip's start OR end to existing clip edges / the
    // playhead, exactly like dragging an existing clip. Records the snapped edge for the dashed guide.
    snapHintRef.current = null;
    if (useAppStore.getState().snapSegments) {
      const targets = collectSnapTicks(trks, undefined, useProjectStore.getState().playheadTick);
      const snapped = Math.max(0, snapMovedStart(tick, durTicks, targets, SNAP_PX / pptRef.current));
      if (snapped !== tick) {
        snapHintRef.current = targets.includes(snapped) ? snapped : targets.includes(snapped + durTicks) ? snapped + durTicks : null;
        tick = snapped;
      }
    }

    // Near a track boundary → create a NEW track there (an empty placeholder row opens via the gap).
    const boundary = boundaryIndexAt(contentY, trks, offsets, totalH, DROP_BOUNDARY_PX);
    if (boundary !== null) return { target: "new", index: boundary, tick };

    // Over a track body.
    const idx = findTrackAtY(offsets, contentY);
    if (idx >= 0 && idx < trks.length && contentY <= totalH) {
      const track = trks[idx]!;
      const half = offsets[idx]! + computeTrackHeight(track, scale) / 2;
      if (track.trackType === "audio") {
        // Insert onto this audio track unless the clip would collide with existing content there —
        // then drop a new track just above/below (by cursor half) so clips never overlap on import.
        // (An empty audio track has no content → never collides → the clip lands in it.)
        const end = tick + durTicks;
        const collides = track.segments.some((s) => !s.loading && tick < s.startTick + s.durationTicks && s.startTick < end);
        if (!collides) return { target: "insert", index: idx, tick };
        return { target: "new", index: contentY < half ? idx : idx + 1, tick };
      }
      // Non-audio track body → new track above/below by cursor half.
      return { target: "new", index: contentY < half ? idx : idx + 1, tick };
    }

    // Below the last track → new track at the end.
    return { target: "new", index: trks.length, tick };
  }, []);

  // Apply the active move/resize for a pointer position. Delta is CONTENT-space (includes scrollX),
  // so the segment keeps moving/extending during edge auto-scroll even when the pointer is held
  // still (scrollX advances → delta grows). Called from the document mousemove AND the autoscroll tick.
  const applyDrag = useCallback((clientX: number, clientY: number) => {
    const drag = dragRef.current;
    if (!drag || drag.mode === "playhead") return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const ppt = pptRef.current;
    const deltaTicks = Math.round((clientX - rect.left + scrollXRef.current - drag.startContentX) / ppt);
    const trks = tracksRef.current;
    const srcTrack = trks[drag.trackIdx];
    if (!srcTrack) return;

    // Require a real drag (>3px screen movement) before moving OR resizing — a click, or a click-hold
    // that happens to sit in an edge auto-scroll zone, must not nudge/resize the segment.
    if (!drag.dragged && Math.hypot(clientX - drag.startMouseX, clientY - drag.startMouseY) > 3) drag.dragged = true;
    if (!drag.dragged) return;

    // Sub-lane trim (non-destructive edge-stretch): re-trim the clip list captured at grab to the pointer
    // position (stem-ms), then write the recipe. No snapping / no move — the lane is pinned to its parent.
    if (drag.mode === "laneResizeL" || drag.mode === "laneResizeR" || drag.mode === "laneBoundary") {
      const seg = srcTrack.segments.find((s) => s.id === drag.segId);
      if (!seg || seg.content.type !== "audioClip" || drag.laneGroup === undefined || drag.origClips === undefined) return;
      const tp = tempoRef.current;
      const offset = seg.content.offsetMs;
      const stemDurMs = seg.content.totalDurationMs;
      const winEnd = offset + ticksToMs(seg.durationTicks, tp);
      const edgeTick = (clientX - rect.left + scrollXRef.current) / ppt;
      const newMs = offset + ticksToMs(edgeTick - seg.startTick, tp);
      // A shared-boundary drag: the FIRST dragged frame's DIRECTION locks which piece shrinks for the rest
      // of this drag (right of the boundary → the right piece's front = its start; left → the left piece's
      // back = its end), by resolving into a plain laneResizeL/R. Locking — vs re-deciding at M every frame —
      // lets you pull a gap back CLOSED (the moved edge clamps exactly to the boundary) without the OTHER
      // piece taking over at M (the reported "can't fully close" micro-op). Re-grab to edit the other piece.
      if (drag.mode === "laneBoundary") {
        if (newMs >= (drag.laneBoundaryMs ?? newMs)) { drag.mode = "laneResizeL"; drag.laneClipIndex = drag.laneRightClipIndex ?? 0; }
        else { drag.mode = "laneResizeR"; drag.laneClipIndex = drag.laneLeftClipIndex ?? 0; }
      }
      const clips = trimClip(
        drag.origClips, stemDurMs, drag.laneClipIndex ?? 0,
        drag.mode === "laneResizeL" ? "start" : "end", newMs, offset, winEnd,
      );
      updateSegmentLaneOps(srcTrack.id, seg.id, drag.laneGroup, isWholeClips(clips, stemDurMs) ? undefined : clips);
      return;
    }

    // Clip snapping: targets are other clips' edges + the playhead (the moving clips are excluded so
    // they don't snap to themselves). Tolerance is SNAP_PX screen px expressed in ticks.
    const snapTargets = useAppStore.getState().snapSegments
      ? collectSnapTicks(trks, new Set(drag.moving.map((m) => m.segId)), useProjectStore.getState().playheadTick)
      : null;
    const snapTol = SNAP_PX / ppt;
    snapHintRef.current = null; // recomputed below; set to the snapped edge tick when an edge snaps

    if (drag.mode === "move") {
      // Snap the move by whichever edge of the primary clip lands closest to a target.
      let moveDelta = deltaTicks;
      if (snapTargets) {
        const snappedStart = snapMovedStart(drag.origStartTick + deltaTicks, drag.origDurationTicks, snapTargets, snapTol);
        moveDelta = snappedStart - drag.origStartTick;
        // Record which edge landed on a target so draw() shows a dashed guide there.
        if (snapTargets.includes(snappedStart)) snapHintRef.current = snappedStart;
        else if (snapTargets.includes(snappedStart + drag.origDurationTicks)) snapHintRef.current = snappedStart + drag.origDurationTicks;
      }
      // Multi-selection move: shift every selected segment by the same delta in ONE store update.
      if (drag.moving.length > 1) {
        const origBySeg: Record<string, number> = {};
        for (const m of drag.moving) origBySeg[m.segId] = m.origStartTick;
        useProjectStore.getState().moveSegmentsBy(origBySeg, moveDelta);
        return;
      }

      // Single-segment move with cross-track detection.
      const scrollYNow = useAppStore.getState().scrollY;
      const mouseY = clientY - rect.top + scrollYNow;
      const yOffsets = computeTrackYOffsets(trks, vZoomRef.current);
      const targetIdx = Math.max(0, Math.min(trks.length - 1, findTrackAtY(yOffsets, mouseY)));
      const targetTrack = trks[targetIdx];
      if (targetTrack && targetIdx !== drag.trackIdx && srcTrack.trackType === "audio" && targetTrack.trackType === "audio") {
        const seg = srcTrack.segments.find((s) => s.id === drag.segId);
        if (seg) {
          const movedSeg = { ...seg, startTick: Math.max(0, drag.origStartTick + moveDelta) };
          // The 组 mixer + row mutes are TRACK-level state — carry the moved segment's entries onto the
          // target track (only when absent there: an existing entry is the target's own mix for that
          // group). Without this, deposited lanes moved cross-track silently played at the default
          // volume/pan and unmuted. Source entries are kept (other segments may share the group; entries
          // are never GC'd by design). Mutes copy TRUTHY only — an explicit-false entry would MASK the
          // target's legacy per-laneId muted flag (isLaneRowMuted's fallback) and unmute its row, and
          // the serializer normalizes plain false away anyway. Every copy is recorded in drag.mixCopies
          // so tracks the segment merely passes over get theirs reverted at release.
          let laneControls = targetTrack.laneControls;
          let laneMutes = targetTrack.laneMutes;
          const copied = { trackId: targetTrack.id, controlKeys: [] as string[], muteKeys: [] as string[] };
          for (const o of seg.processedOutputs ?? []) {
            const gid = laneGroupId(o);
            if (laneControls[gid] === undefined) {
              const src = laneControlFor(srcTrack, gid, o.laneId);
              if (src) {
                laneControls = { ...laneControls, [gid]: { ...src } };
                copied.controlKeys.push(gid);
              }
            }
            const rk = laneRowKey(o);
            if (srcTrack.laneMutes?.[rk] && laneMutes?.[rk] === undefined) {
              laneMutes = { ...(laneMutes ?? {}), [rk]: true };
              copied.muteKeys.push(rk);
            }
          }
          if (copied.controlKeys.length > 0 || copied.muteKeys.length > 0) {
            (drag.mixCopies ??= []).push(copied);
          }
          updateTrack(srcTrack.id, { segments: srcTrack.segments.filter((s) => s.id !== drag.segId) });
          updateTrack(targetTrack.id, { segments: [...targetTrack.segments, movedSeg], laneControls, laneMutes });
          drag.trackIdx = targetIdx;
          useAppStore.getState().selectSegment(targetTrack.id, drag.segId);
        }
        return;
      }
      updateTrack(srcTrack.id, {
        segments: srcTrack.segments.map((seg) =>
          seg.id === drag.segId ? { ...seg, startTick: Math.max(0, drag.origStartTick + moveDelta) } : seg,
        ),
      });
      return;
    }

    // Resize (no cross-track)
    updateTrack(srcTrack.id, {
      segments: srcTrack.segments.map((seg) => {
        if (seg.id !== drag.segId) return seg;
        if (drag.mode === "resizeL") {
          let snapped = drag.origStartTick + deltaTicks;
          if (snapTargets) {
            const s = snapTick(snapped, snapTargets, snapTol);
            if (s !== snapped) snapHintRef.current = s;
            snapped = s;
          }
          const newStart = Math.max(0, snapped);
          const shrink = newStart - drag.origStartTick;
          const newDur = Math.max(TICKS_PER_BEAT / 4, drag.origDurationTicks - shrink);
          const newOff =
            seg.content.type === "audioClip"
              ? Math.max(0, drag.origOffsetMs + ticksToMs(shrink, tempoRef.current))
              : 0;
          return {
            ...seg, startTick: newStart, durationTicks: newDur,
            content: seg.content.type === "audioClip" ? { ...seg.content, offsetMs: newOff } : seg.content,
          };
        }
        if (drag.mode === "resizeR") {
          let newEnd = drag.origStartTick + drag.origDurationTicks + deltaTicks;
          if (snapTargets) {
            const s = snapTick(newEnd, snapTargets, snapTol);
            if (s !== newEnd) snapHintRef.current = s;
            newEnd = s;
          }
          return { ...seg, durationTicks: Math.max(TICKS_PER_BEAT / 4, newEnd - drag.origStartTick) };
        }
        return seg;
      }),
    });
  }, [updateTrack, updateSegmentLaneOps]);

  // Auto-scroll during drag near edges
  const startAutoScroll = useCallback(() => {
    cancelAnimationFrame(autoScrollRef.current);
    const tick = () => {
      if (!dragRef.current) return;
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const localX = mouseClientXRef.current - rect.left;
      let dx = 0;
      if (localX < AUTOSCROLL_ZONE) {
        dx = -AUTOSCROLL_SPEED * Math.min(2, (AUTOSCROLL_ZONE - localX) / AUTOSCROLL_ZONE);
      } else if (localX > rect.width - AUTOSCROLL_ZONE) {
        dx = AUTOSCROLL_SPEED * Math.min(2, (localX - (rect.width - AUTOSCROLL_ZONE)) / AUTOSCROLL_ZONE);
      }
      // Auto-scroll only for a committed gesture: a playhead drag always scrub-scrolls, but a
      // move/resize must have actually started (a click-hold that merely sits in the edge zone must
      // not scroll or change anything). Skip when the scroll is already clamped (no change) to avoid
      // per-frame churn at the tick-0 boundary.
      const committed = dragRef.current.mode === "playhead" || dragRef.current.dragged;
      if (dx !== 0 && committed) {
        const newX = Math.max(0, scrollXRef.current + dx);
        if (newX !== scrollXRef.current) {
          useAppStore.getState().setScroll(newX, useAppStore.getState().scrollY);
          // Re-apply the active op against the NEW scroll position (the pointer is held still at the
          // edge, so without this the segment wouldn't keep moving/extending while it scrolls).
          if (dragRef.current.mode === "playhead") {
            useProjectStore.getState().setPlayhead(playheadTickAt(mouseClientXRef.current));
          } else {
            applyDrag(mouseClientXRef.current, mouseClientYRef.current);
          }
        }
      }
      autoScrollRef.current = requestAnimationFrame(tick);
    };
    autoScrollRef.current = requestAnimationFrame(tick);
  }, [playheadTickAt, applyDrag]);

  const stopAutoScroll = useCallback(() => {
    cancelAnimationFrame(autoScrollRef.current);
  }, []);

  // Document-level drag handlers
  useEffect(() => {
    const onDocMove = (e: MouseEvent) => {
      const drag = dragRef.current;
      mouseClientXRef.current = e.clientX;
      mouseClientYRef.current = e.clientY;
      if (!drag) return;
      if (drag.mode === "playhead") {
        // Playback may START mid-drag (Space is global): pin `seeking` so the transport rAF doesn't
        // fight the drag, and so the release reschedules from the dragged position — the mousedown
        // only set it if playback was ALREADY active (mirrors OverviewMap's drag handler).
        const a = useAudioStore.getState();
        if (a.isPlaying && !a.seeking) a.setSeeking(true);
        setPlayhead(playheadTickAt(e.clientX));
        return;
      }
      applyDrag(e.clientX, e.clientY);
    };

    const onDocUp = () => {
      const drag = dragRef.current;
      if (drag) {
        // Plain click (no actual drag) on an already-multi-selected segment → collapse to just it.
        if (drag.mode === "move" && !drag.dragged && drag.collapseTo) {
          useAppStore.getState().selectSegment(drag.collapseTo.trackId, drag.collapseTo.segId);
        }
        // Revert mixer/mute entries copied onto tracks the segment merely PASSED OVER during a
        // cross-track drag (see mixCopies) — only the track it ends on keeps its copies. Runs inside
        // the still-open transaction, so a round-trip drag commits as the no-op it visually is.
        if (drag.mixCopies) {
          const finalTrackId = tracksRef.current[drag.trackIdx]?.id;
          for (const c of drag.mixCopies) {
            if (c.trackId === finalTrackId) continue;
            const tr = tracksRef.current.find((t) => t.id === c.trackId);
            if (!tr) continue;
            const laneControls = { ...tr.laneControls };
            for (const k of c.controlKeys) delete laneControls[k];
            let laneMutes = tr.laneMutes;
            if (laneMutes && c.muteKeys.length > 0) {
              laneMutes = { ...laneMutes };
              for (const k of c.muteKeys) delete laneMutes[k];
            }
            useProjectStore.getState().updateTrack(c.trackId, { laneControls, laneMutes });
          }
        }
        // A lane trim that collapsed its piece to a sliver leaves an invisible, un-grabbable phantom
        // clip in the recipe (trimClip clamps at 1ms; the draw drops it but it persists + still sigs).
        // Dragging an edge all the way across a piece reads as "remove it" — finish the job on release,
        // inside the same transaction so trim+splice commit as ONE undo step.
        if ((drag.mode === "laneResizeL" || drag.mode === "laneResizeR") && drag.dragged && drag.laneGroup !== undefined) {
          const track = tracksRef.current[drag.trackIdx];
          const seg = track?.segments.find((s) => s.id === drag.segId);
          if (track && seg && seg.content.type === "audioClip") {
            const stored = seg.laneOps?.[drag.laneGroup];
            const kept = stored?.filter((c) => c.end - c.start > MIN_LANE_PIECE_MS);
            if (stored && kept && kept.length !== stored.length) {
              updateSegmentLaneOps(track.id, seg.id, drag.laneGroup,
                isWholeClips(kept, seg.content.totalDurationMs) ? undefined : kept);
            }
          }
        }
        // A committed clip move/resize changed segment timing → reschedule playback so it doesn't
        // keep playing the old layout (playhead drags reschedule via the seeking flag instead).
        if (drag.dragged && drag.mode !== "playhead" && useAudioStore.getState().isPlaying) {
          useAudioStore.getState().bumpSchedule();
        }
        // Close the undo transaction opened at mousedown for a move/resize — commits ONE step iff
        // the clip actually moved/resized (a no-drag click makes no change → discarded). A playhead
        // drag never opened a transaction (it only moves the non-undoable transport cursor).
        if (drag.mode !== "playhead") useHistoryStore.getState().commitTransaction();
        dragRef.current = null;
        stopAutoScroll();
        if (snapHintRef.current !== null) {
          snapHintRef.current = null; // erase the snap guide once the drag ends
          requestRedraw();
        }
        if (useAudioStore.getState().seeking) {
          useAudioStore.getState().setSeeking(false);
        }
      }
    };

    document.addEventListener("mousemove", onDocMove);
    document.addEventListener("mouseup", onDocUp);
    return () => {
      document.removeEventListener("mousemove", onDocMove);
      document.removeEventListener("mouseup", onDocUp);
    };
  }, [playheadTickAt, setPlayhead, applyDrag, stopAutoScroll, requestRedraw]);

  // Canvas mousedown (starts drag)
  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (e.button !== 0) return;
      const ctrl = e.ctrlKey || e.metaKey;
      const rect = canvasRef.current?.getBoundingClientRect();
      const startContentX = rect ? e.clientX - rect.left + scrollXRef.current : 0;
      const hit = hitTest(e.clientX, e.clientY);
      if (hit) {
        const track = tracks[hit.trackIdx];
        const seg = track?.segments.find((s) => s.id === hit.segId);
        if (!track || !seg) return;
        const offsetMs = seg.content.type === "audioClip" ? seg.content.offsetMs : 0;

        // Sub-lane hit (track expanded, clicked a lane row): select the GROUP; an edge starts a
        // non-destructive trim, the body just selects (Ctrl+K slices / Delete removes the clicked piece).
        // No multi-select / move / cross-track for lanes (they follow the parent) — Ctrl is ignored here.
        if (hit.lane) {
          selectLane(track.id, seg.id, hit.lane.group, hit.lane.clipIndex);
          if ((hit.zone === "left" || hit.zone === "right") && seg.content.type === "audioClip") {
            const clips = materializeClips(seg.laneOps?.[hit.lane.group], seg.content.totalDurationMs);
            const ci = hit.lane.clipIndex;
            // A COINCIDENT slice boundary (two touching pieces, no gap) → a DIRECTION-driven boundary drag:
            // no need to hover the exact side — the drag direction decides which piece shrinks (right → the
            // right piece's front, left → the left piece's back). Detect the touching neighbour on the
            // grabbed side (right-edge → next piece; left-edge → prev piece).
            let boundary: { ms: number; leftIdx: number; rightIdx: number } | null = null;
            if (hit.zone === "right" && ci + 1 < clips.length && Math.abs(clips[ci]!.end - clips[ci + 1]!.start) < 1) {
              boundary = { ms: clips[ci]!.end, leftIdx: ci, rightIdx: ci + 1 };
            } else if (hit.zone === "left" && ci - 1 >= 0 && Math.abs(clips[ci - 1]!.end - clips[ci]!.start) < 1) {
              boundary = { ms: clips[ci]!.start, leftIdx: ci - 1, rightIdx: ci };
            }
            dragRef.current = {
              mode: boundary ? "laneBoundary" : hit.zone === "left" ? "laneResizeL" : "laneResizeR",
              trackIdx: hit.trackIdx, segId: seg.id,
              startMouseX: e.clientX, startMouseY: e.clientY, startContentX,
              origStartTick: seg.startTick, origDurationTicks: seg.durationTicks, origOffsetMs: offsetMs,
              moving: [], dragged: false, collapseTo: null,
              laneGroup: hit.lane.group, laneClipIndex: ci, origClips: clips,
              laneBoundaryMs: boundary?.ms, laneLeftClipIndex: boundary?.leftIdx, laneRightClipIndex: boundary?.rightIdx,
            };
            useHistoryStore.getState().beginTransaction(); // coalesce the trim into one undo step
            startAutoScroll();
          }
          return;
        }

        // Ctrl+click adjusts the multi-selection only (file-manager style) — no drag.
        if (ctrl) {
          toggleSegment(track.id, seg.id);
          return;
        }

        // Resize is single-segment — select just this one.
        if (hit.zone === "left" || hit.zone === "right") {
          selectSegment(track.id, seg.id);
          dragRef.current = {
            mode: hit.zone === "left" ? "resizeL" : "resizeR",
            trackIdx: hit.trackIdx, segId: hit.segId,
            startMouseX: e.clientX, startMouseY: e.clientY, startContentX,
            origStartTick: seg.startTick, origDurationTicks: seg.durationTicks, origOffsetMs: offsetMs,
            moving: [{ trackId: track.id, segId: seg.id, origStartTick: seg.startTick }],
            dragged: false, collapseTo: null,
          };
          useHistoryStore.getState().beginTransaction(); // coalesce the resize into one undo step
          startAutoScroll();
          return;
        }

        // Body → move. Keep the multi-selection if this segment is already in it (so a drag moves
        // all of them); a plain click that doesn't drag collapses to just this one on mouse-up.
        const sel = useAppStore.getState().selectedSegments;
        const already = sel.some((x) => x.trackId === track.id && x.segmentId === seg.id);
        let collapseTo: { trackId: string; segId: string } | null = null;
        if (already) collapseTo = { trackId: track.id, segId: seg.id };
        else selectSegment(track.id, seg.id);

        const trks = tracksRef.current;
        const moving = useAppStore.getState().selectedSegments
          .map((x) => {
            const tk = trks.find((t) => t.id === x.trackId);
            const sg = tk?.segments.find((s) => s.id === x.segmentId);
            return sg ? { trackId: x.trackId, segId: x.segmentId, origStartTick: sg.startTick } : null;
          })
          .filter(Boolean) as { trackId: string; segId: string; origStartTick: number }[];

        dragRef.current = {
          mode: "move", trackIdx: hit.trackIdx, segId: hit.segId,
          startMouseX: e.clientX, startMouseY: e.clientY, startContentX,
          origStartTick: seg.startTick, origDurationTicks: seg.durationTicks, origOffsetMs: offsetMs,
          moving, dragged: false, collapseTo,
        };
        useHistoryStore.getState().beginTransaction(); // coalesce the move (incl. cross-track) into one step
        startAutoScroll();
      } else {
        if (!ctrl) clearSelection();
        dragRef.current = {
          mode: "playhead", trackIdx: -1, segId: "",
          startMouseX: e.clientX, startMouseY: e.clientY, startContentX,
          origStartTick: 0, origDurationTicks: 0, origOffsetMs: 0,
          moving: [], dragged: false, collapseTo: null,
        };
        setPlayhead(playheadTickAt(e.clientX));
        if (useAudioStore.getState().isPlaying) {
          useAudioStore.getState().setSeeking(true);
        }
        startAutoScroll();
      }
    },
    [hitTest, tracks, setPlayhead, playheadTickAt, selectSegment, selectLane, toggleSegment, clearSelection, startAutoScroll],
  );

  // Canvas mousemove (cursor only, drag is handled at document level)
  const handleCanvasMouseMove = useCallback(
    (e: React.MouseEvent) => {
      const canvas = canvasRef.current;
      if (canvas) mouseXRef.current = e.clientX - canvas.getBoundingClientRect().left;

      if (!dragRef.current) {
        const hit = hitTest(e.clientX, e.clientY);
        // Lane edge → trim (ew-resize); lane body → select (pointer, no move); segment edge → resize; body → move.
        setCursor(!hit ? "crosshair" : hit.zone !== "body" ? "ew-resize" : hit.lane ? "pointer" : "grab");
        drawRef.current();
      }
    },
    [hitTest],
  );

  const handleMouseLeave = useCallback(() => {
    mouseXRef.current = -9999;
    drawRef.current();
  }, []);

  const handleDoubleClick = useCallback(
    (e: React.MouseEvent) => {
      const hit = hitTest(e.clientX, e.clientY);
      if (hit) openWorkflow(hit.segId);
    },
    [hitTest, openWorkflow],
  );

  // Wheel: scroll = horizontal, shift = vertical, ctrl = zoom.
  // Scroll deltas are coalesced and flushed once per animation frame, so a burst of wheel
  // events (high-res mice fire many per frame) causes at most one state update + one redraw
  // per frame instead of one each — fixes scroll jank / CPU spikes.
  const wheelDxRef = useRef(0);
  const wheelDyRef = useRef(0);
  const wheelRafRef = useRef(0);
  const vZoomFactorRef = useRef(1);
  const vZoomRafRef = useRef(0);
  const maxScrollY = () =>
    Math.max(0, computeTotalTracksHeight(tracksRef.current, vZoomRef.current) + TRACK_ADD_FOOTER - useAppStore.getState().canvasHeight);
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.stopPropagation();
    if (e.ctrlKey) {
      e.preventDefault();
      // Horizontal zoom ANCHORED at the cursor — keep the tick under the pointer fixed (instead of
      // zooming relative to bar 1, which slid the view when zoomed into a later region).
      const st = useAppStore.getState();
      const rect = canvasRef.current?.getBoundingClientRect();
      const cursorX = rect ? e.clientX - rect.left : st.canvasWidth / 2;
      const oldPpt = PIXELS_PER_TICK * st.zoom;
      const newZoom = Math.max(0.1, Math.min(10, st.zoom * (e.deltaY > 0 ? 0.9 : 1.1)));
      const tickAtCursor = (cursorX + st.scrollX) / oldPpt;
      const newScrollX = Math.max(0, tickAtCursor * PIXELS_PER_TICK * newZoom - cursorX);
      st.setZoom(newZoom);
      st.setScroll(newScrollX, st.scrollY);
      return;
    }
    if (e.altKey) {
      e.preventDefault();
      // Vertical (track-height) zoom — coalesced to one update per frame (smooth, not sluggish).
      vZoomFactorRef.current *= e.deltaY > 0 ? 0.9 : 1.1;
      if (!vZoomRafRef.current) {
        vZoomRafRef.current = requestAnimationFrame(() => {
          vZoomRafRef.current = 0;
          const st = useAppStore.getState();
          st.setVZoom(st.vZoom * vZoomFactorRef.current);
          vZoomFactorRef.current = 1;
          st.setScroll(st.scrollX, Math.min(st.scrollY, maxScrollY()));
        });
      }
      return;
    }
    if (e.shiftKey) wheelDyRef.current += e.deltaY;
    else wheelDxRef.current += e.deltaY;
    if (wheelRafRef.current) return;
    wheelRafRef.current = requestAnimationFrame(() => {
      wheelRafRef.current = 0;
      const dx = wheelDxRef.current;
      const dy = wheelDyRef.current;
      wheelDxRef.current = 0;
      wheelDyRef.current = 0;
      const st = useAppStore.getState();
      // Clamp Y to content so a short track stack can't scroll out of view.
      st.setScroll(Math.max(0, st.scrollX + dx), Math.max(0, Math.min(maxScrollY(), st.scrollY + dy)));
    });
  }, []);

  // Right-click context menu
  const handleContextMenu = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      const hit = hitTest(e.clientX, e.clientY);
      const trackIdx = hitTrackIdx(e.clientY);
      if (hit) {
        const track = tracks[hit.trackIdx];
        if (hit.lane) {
          if (track) selectLane(track.id, hit.segId, hit.lane.group, hit.lane.clipIndex);
          setCtxMenu({ x: e.clientX, y: e.clientY, trackIdx: hit.trackIdx, segId: hit.segId, lane: { outputNodeId: hit.lane.group, clipIndex: hit.lane.clipIndex } });
        } else {
          if (track) selectSegment(track.id, hit.segId);
          setCtxMenu({ x: e.clientX, y: e.clientY, trackIdx: hit.trackIdx, segId: hit.segId });
        }
      } else if (trackIdx >= 0 && trackIdx < tracks.length) {
        setCtxMenu({ x: e.clientX, y: e.clientY, trackIdx, segId: null });
      }
    },
    [hitTest, hitTrackIdx, tracks, selectSegment, selectLane],
  );

  const ctxItems: MenuItem[] = (() => {
    if (!ctxMenu) return [];
    const track = tracks[ctxMenu.trackIdx];
    if (!track) return [];
    const items: MenuItem[] = [];
    // Sub-lane right-click → non-destructive slice (at playhead) / delete-piece for the lane group.
    if (ctxMenu.lane && ctxMenu.segId) {
      const seg = track.segments.find((s) => s.id === ctxMenu.segId);
      const ph = useProjectStore.getState().playheadTick;
      const canSlice = !!seg && seg.content.type === "audioClip" && ph > seg.startTick && ph < seg.startTick + seg.durationTicks;
      const group = ctxMenu.lane.outputNodeId;
      items.push({
        label: t("toolbar.split"), shortcut: "Ctrl+K", disabled: !canSlice,
        onClick: () => sliceLaneGroupAtPlayhead(track.id, ctxMenu.segId!, group),
      });
      items.push({
        label: t("toolbar.delete"), shortcut: "Del",
        onClick: () => deleteLanePiece(track.id, ctxMenu.segId!, group, ctxMenu.lane!.clipIndex),
      });
      // "Detach": split a MULTI-input Output node into one Output (group) per inbound edge. Executed by
      // the segment's workflow editor (ONE code path + node-graph undo) — open it, then hand the request.
      const fanIn = seg?.workflow?.connections.filter((c) => c.toNode === group).length ?? 0;
      if (fanIn >= 2) {
        items.push({
          label: t("workflow.detachGroup"),
          onClick: () => {
            useAppStore.getState().openWorkflow(ctxMenu.segId!);
            useAppStore.getState().requestLaneDetach(ctxMenu.segId!, group);
          },
        });
      }
      return items;
    }
    if (ctxMenu.segId) {
      const seg = track.segments.find((s) => s.id === ctxMenu.segId);
      const ph = useProjectStore.getState().playheadTick;
      const canSplit = seg && ph > seg.startTick && ph < seg.startTick + seg.durationTicks;
      items.push({
        label: t("toolbar.split"), shortcut: "Ctrl+K", disabled: !canSplit,
        onClick: () => { if (canSplit) splitSegment(track.id, ctxMenu.segId!, useProjectStore.getState().playheadTick); },
      });
      items.push({
        label: t("toolbar.delete"), shortcut: "Del",
        onClick: () => {
          deleteSegment(track.id, ctxMenu.segId!);
          clearSelection();
          if (useAudioStore.getState().isPlaying) useAudioStore.getState().bumpSchedule();
        },
      });
      // Clear a render → reveal the original audio again (non-destructive: content + node graph are
      // preserved, so it can be re-rendered any time). processedOutputs is a non-undoable overlay, so
      // this — like rendering — isn't an undo step; to get the render back, re-run the workflow.
      if (seg?.processedOutputs?.some((o) => !o.loading)) {
        items.push({
          label: t("workflow.clearRender"),
          onClick: () => {
            useProjectStore.getState().clearProcessedOutputs(track.id, ctxMenu.segId!);
            if (useAudioStore.getState().isPlaying) useAudioStore.getState().bumpSchedule();
          },
        });
      }
    }
    items.push({
      label: t("tracks.delete"), danger: true,
      onClick: () => useProjectStore.getState().removeTrack(track.id),
    });
    return items;
  })();

  // Remove the placeholder gap (canvas ref + the store flag the header column reads). Guarded.
  const clearGhostGap = useCallback(() => {
    if (lastGhostInsertRef.current !== "") {
      lastGhostInsertRef.current = "";
      useAppStore.getState().setGhostInsert(null);
    }
  }, []);

  // Drag-and-drop (OS file drop). In Tauri 2's WebView2 the HTML5 `File.path` is always
  // undefined, so OS drops MUST be received via the Tauri drag-drop event (which carries real
  // filesystem paths). Smart placement: the first file lands as a SEGMENT on the audio track
  // under the cursor (if any) at the hovered tick; otherwise — and every additional file — gets
  // its own new track. import.ts owns the loading placeholder + decode/error handling.
  const importDroppedFiles = useCallback(
    async (paths: string[], place: Placement, durations: Record<string, number>) => {
      const audioPaths = paths.filter((p) => AUDIO_EXT_RE.test(p));
      if (place.target === "none" || audioPaths.length === 0) {
        clearGhostGap();
        return;
      }

      // Resolve every duration UP FRONT (cached from the drag, else probe now). With a known
      // duration each import runs synchronously up to addTrack, so the indexed inserts below land
      // in file order; otherwise an un-cached probe await would defer addTrack and the splices
      // would interleave in probe-resolution order.
      const durs = await Promise.all(audioPaths.map((p) => durations[p] ?? probeAudioDuration(p)));

      // Clear the placeholder gap in the SAME tick as the synchronous inserts below, so the
      // gap→track handoff is a single paint (no up-then-down jump if a probe was still pending).
      clearGhostGap();
      const trks = tracksRef.current; // re-read after the await
      // One undo step for the whole drop: the placeholder inserts below are synchronous (durations
      // pre-resolved), so they all land inside this transaction; the async decode-populate runs
      // silently afterwards (finalizeSegment uses history.runSilent) and never adds a second step.
      const hist = useHistoryStore.getState();
      hist.beginTransaction();
      try {
        for (let i = 0; i < audioPaths.length; i++) {
          const filePath = audioPaths[i]!;
          const known = durs[i]!;
          if (place.target === "insert") {
            // First file inserts onto the hovered audio track; extra files get their own new tracks.
            if (i === 0) {
              const track = trks[place.index];
              if (track && track.trackType === "audio") {
                void importAudioToExistingTrack(filePath, track.id, place.tick, known);
                continue;
              }
            }
            void importAudioToNewTrack(filePath, place.tick, known);
          } else {
            // New track at the dragged position; multiple files stack in order from there.
            void importAudioToNewTrack(filePath, place.tick, known, place.index + i);
          }
        }
      } finally {
        hist.commitTransaction();
      }
    },
    [clearGhostGap],
  );

  // Publish the placeholder-gap insertion (index + file count) to the store so the track-header
  // column (DOM) opens the SAME gap as the canvas — otherwise the two columns misalign mid-drag.
  // Only for a NEW track dropped BETWEEN existing tracks. Guarded to avoid re-render churn.
  const syncGhostInsert = useCallback((place: Placement | null) => {
    const trks = tracksRef.current;
    const gi =
      place && place.target === "new" && place.index < trks.length
        ? { index: place.index, count: Math.max(1, dragPathsRef.current.length) }
        : null;
    const key = gi ? `${gi.index}_${gi.count}` : "";
    if (key !== lastGhostInsertRef.current) {
      lastGhostInsertRef.current = key;
      useAppStore.getState().setGhostInsert(gi);
    }
  }, []);

  // Edge auto-scroll while dragging an OS file over the arrangement: near the canvas's left/right
  // edge, scroll horizontally and re-place the ghost against the new scroll. The OS drag fires no
  // "over" events while the cursor is held still at the edge, so this rAF keeps it going (mirrors the
  // segment-drag auto-scroll). This is what makes the track-header column unnecessary for positioning.
  const startDragAutoScroll = useCallback(() => {
    cancelAnimationFrame(dragScrollRef.current);
    const tick = () => {
      const pos = dragPosRef.current;
      const canvas = canvasRef.current;
      if (!pos || !canvas || dragPathsRef.current.length === 0) return; // drag ended / no audio drag
      const rect = canvas.getBoundingClientRect();
      const ly = pos.y / devicePixelRatio;
      // Only auto-scroll while the cursor is within the rows' vertical band — not over the ruler above
      // or below the panel (where there's no drop target / ghost).
      if (ly >= rect.top && ly <= rect.bottom) {
        const localX = pos.x / devicePixelRatio - rect.left;
        let dx = 0;
        if (localX < AUTOSCROLL_ZONE) dx = -AUTOSCROLL_SPEED * Math.min(2, (AUTOSCROLL_ZONE - localX) / AUTOSCROLL_ZONE);
        else if (localX > rect.width - AUTOSCROLL_ZONE) dx = AUTOSCROLL_SPEED * Math.min(2, (localX - (rect.width - AUTOSCROLL_ZONE)) / AUTOSCROLL_ZONE);
        if (dx !== 0) {
          // No content-based max clamp: allow scrolling PAST the last clip into empty space so a clip
          // can be dropped a few bars after the end. The minimap/scrollbar stay content-based and just
          // slide out of view (fine). The rAF still stops when the drag ends (guards above) / on leave.
          const newX = Math.max(0, scrollXRef.current + dx);
          if (newX !== scrollXRef.current) {
            useAppStore.getState().setScroll(newX, useAppStore.getState().scrollY);
            const place = placementFromPhysical(pos);
            ghostRef.current = place.target === "none" ? null : place;
            syncGhostInsert(place);
            requestRedraw();
          }
        }
      }
      dragScrollRef.current = requestAnimationFrame(tick);
    };
    dragScrollRef.current = requestAnimationFrame(tick);
  }, [placementFromPhysical, syncGhostInsert, requestRedraw]);

  const stopDragAutoScroll = useCallback(() => {
    cancelAnimationFrame(dragScrollRef.current);
    dragPosRef.current = null;
  }, []);

  const clearDragState = useCallback(() => {
    stopDragAutoScroll();
    ghostRef.current = null;
    snapHintRef.current = null;
    dragPathsRef.current = [];
    dragDurationsRef.current = {};
    clearGhostGap();
    setDragOver(false);
    requestRedraw();
  }, [requestRedraw, clearGhostGap, stopDragAutoScroll]);

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    let cancelled = false;
    getCurrentWebview()
      .onDragDropEvent((event) => {
        const p = event.payload;
        // The webview drag event is GLOBAL: while the full-screen training page is
        // open (covering the timeline), it owns the drop as dataset import. Ignore
        // it here, but first tear down any affordance a drag started before the
        // page opened would have left behind (ghost gap / highlight).
        if (useAppStore.getState().trainingPageOpen) {
          if (dragPathsRef.current.length > 0) clearDragState();
          return;
        }
        if (p.type === "enter") {
          const audioPaths = p.paths.filter((pp) => AUDIO_EXT_RE.test(pp));
          dragPathsRef.current = audioPaths;
          dragDurationsRef.current = {};
          if (audioPaths.length === 0) return; // non-audio drag: no affordance
          setDragOver(true);
          dragPosRef.current = p.position;
          startDragAutoScroll();
          const place = placementFromPhysical(p.position);
          ghostRef.current = place.target === "none" ? null : place;
          syncGhostInsert(place);
          requestRedraw();
          // Probe durations to size the ghost — refresh once each resolves.
          for (const fp of audioPaths) {
            void probeAudioDuration(fp).then((ms) => {
              dragDurationsRef.current[fp] = ms;
              // Re-evaluate placement with the now-known real duration — the first placement sized the
              // clip with the fallback, so a held-still cursor could preview a stale collision decision.
              if (dragPosRef.current) {
                const place = placementFromPhysical(dragPosRef.current);
                ghostRef.current = place.target === "none" ? null : place;
                syncGhostInsert(place);
              }
              requestRedraw();
            });
          }
        } else if (p.type === "over") {
          if (dragPathsRef.current.length === 0) return; // non-audio drag
          dragPosRef.current = p.position;
          const place = placementFromPhysical(p.position);
          ghostRef.current = place.target === "none" ? null : place;
          syncGhostInsert(place);
          requestRedraw();
        } else if (p.type === "leave") {
          clearDragState();
        } else if (p.type === "drop") {
          stopDragAutoScroll();
          const place = placementFromPhysical(p.position);
          const durations = { ...dragDurationsRef.current };
          const paths = p.paths;
          // Tear down the moving ghost box + highlight now, but KEEP the placeholder gap open until
          // importDroppedFiles inserts the tracks (it clears the gap in the same tick as the
          // inserts) so the gap→track handoff is seamless. A drop outside any target clears it too.
          ghostRef.current = null;
          snapHintRef.current = null;
          dragPathsRef.current = [];
          dragDurationsRef.current = {};
          setDragOver(false);
          requestRedraw();
          void importDroppedFiles(paths, place, durations);
        }
      })
      .then((u) => {
        if (cancelled) u();
        else unlisten = u;
      });
    return () => {
      cancelled = true;
      if (unlisten) unlisten();
      stopDragAutoScroll();
    };
  }, [importDroppedFiles, placementFromPhysical, clearDragState, requestRedraw, syncGhostInsert, startDragAutoScroll, stopDragAutoScroll]);

  // Drawing — static content cached in an OffscreenCanvas (re-rendered on scroll/content change,
  // blitted as-is while only the playhead moves). Playhead drawn per frame.
  const offscreenRef = useRef<OffscreenCanvas | null>(null);
  const staticDepsRef = useRef("");

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = devicePixelRatio;
    const { width, height } = canvas.getBoundingClientRect();
    const cw = Math.round(width * dpr);
    const ch = Math.round(height * dpr);
    // Only reallocate the backing store when the size actually changes. Assigning
    // canvas.width every frame forces a clear + GPU realloc — wasteful during scroll and
    // playback. Use setTransform (absolute) since we no longer reset the matrix via resize.
    if (canvas.width !== cw || canvas.height !== ch) {
      canvas.width = cw;
      canvas.height = ch;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Snapshot view state from refs (driven by store subscriptions, not React render).
    const ppt = pptRef.current;
    const scale = vZoomRef.current;
    const scrollX = scrollXRef.current;
    const scrollY = scrollYRef.current;
    const playheadTick = playheadRef.current;

    const selectedKeys = new Set(selectedSegments.map((s) => `${s.trackId}:${s.segmentId}`));
    const selKeyStr = [...selectedKeys].sort().join(",");
    // Selected sub-lane GROUP key (`track:seg:outputNodeId`) — drives the gold row highlight in the static layer.
    const selectedLaneKey = selectedLane
      ? `${selectedLane.trackId}:${selectedLane.segmentId}:${selectedLane.outputNodeId}`
      : null;

    // While dragging a NEW track in BETWEEN existing tracks, open a placeholder gap (push the tracks
    // below down) so the ghost preview sits in a real empty row instead of overlapping a track.
    let ghostGap: { index: number; height: number } | null = null;
    {
      const g = ghostRef.current;
      if (g && g.target === "new" && g.index < tracks.length) {
        ghostGap = { index: g.index, height: Math.max(1, dragPathsRef.current.length) * TRACK_HEADER_HEIGHT * scale };
      }
    }

    const staticKey = `${ppt}:${scale}:${scrollX}:${scrollY}:${tempo}:${timeSignature[0]}/${timeSignature[1]}:${selKeyStr}:${selectedLaneKey ?? ""}:${dragOver}:${ghostGap ? `${ghostGap.index}_${ghostGap.height}` : ""}:${Object.keys(audioFiles).length}:${tracks.map(t => {
      // Both mute sources: per-row laneMutes + the legacy per-laneId muted flag (isLaneRowMuted reads both).
      const laneMutes = Object.entries(t.laneControls).map(([k, v]) => `${k}${v?.muted ? 1 : 0}`).join("|")
        + "/" + Object.entries(t.laneMutes ?? {}).map(([k, v]) => `${k}${v ? 1 : 0}`).join("|");
      // Include a content signature of each output's peaks so a workflow re-run that overwrites an
      // output at the SAME path (same length, new content) still changes the key → static redraw.
      const segs = t.segments.map(s => {
        // Include the SOURCE clip's peak count so the static layer re-bakes when an opened clip's
        // peaks finish loading (the key count alone doesn't change when an existing audioFiles entry
        // gets populated) — otherwise the real waveform never repaints over the empty box.
        const srcPeaks = s.content.type === "audioClip" ? (audioFiles[s.content.sourcePath]?.peaks.length ?? 0) : 0;
        // laneLabel + group are in the key so a group RENAME (relabel-in-place, same audioPath) still
        // repaints the canvas row text — without them the header updates but the canvas text goes stale.
        // laneId too: a DETACH rewrites laneId/outputNodeId in place with the same path/label/group,
        // but it splits one group RUN into several → new bars/row positions must repaint (P5).
        // o.loading: a lane turning ready flips the main row original-waveform → lane-sum switch.
        return `${s.startTick}.${s.durationTicks}.${s.loading ? 1 : 0}.${srcPeaks}.${s.processedOutputs?.map(o => `${o.audioPath}:${o.laneId}:${o.laneLabel}:${o.group ?? ""}:${o.loading ? 1 : 0}:${peaksSignature(o.waveformPeaks ?? [])}`).join(",") ?? ""}.${laneOpsSig(s.laneOps)}`;
      }).join(";");
      // playOriginal: flips the main-row waveform (sum ↔ original) + dims every lane row.
      return `${t.segments.length}:${t.expanded}:${t.muted}:${t.playOriginal ? 1 : 0}:${laneMutes}:${segs}`;
    }).join(",")}`;

    const sizeChanged = !offscreenRef.current
      || offscreenRef.current.width !== cw
      || offscreenRef.current.height !== ch;
    const needsStaticRedraw = sizeChanged || staticDepsRef.current !== staticKey;

    if (needsStaticRedraw) {
      if (sizeChanged) {
        offscreenRef.current = new OffscreenCanvas(cw, ch);
      }
      const oc = offscreenRef.current!.getContext("2d")!;
      oc.setTransform(dpr, 0, 0, dpr, 0, 0);
      drawStaticContent(oc, width, height, tracks, audioFiles, ppt, scrollX, scrollY, timeAxis, tempo, selectedKeys, selectedLaneKey, dragOver, scale, ghostGap);
      staticDepsRef.current = staticKey;
    }

    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(offscreenRef.current!, 0, 0);
    ctx.restore();

    // Loading indicator — drawn per-frame (so the stripes sweep + the spinner spins) for ANY loading
    // clip via the shared isSegLoading predicate + drawLoadingIndicator, so a fresh import placeholder
    // and an opened clip whose peaks are still loading look IDENTICAL. The rAF loop (below) runs while
    // anything is loading. (Static layer paints the base box; this overlays the loading look.)
    {
      const loadOffsets = computeTrackYOffsets(tracks, scale);
      const now = performance.now();
      const headerH = TRACK_HEADER_HEIGHT * scale;
      const laneH = LANE_HEIGHT * scale - 2;
      const label = t("tracks.loading");
      for (let i = 0; i < tracks.length; i++) {
        const track = tracks[i];
        if (!track) continue;
        // Match drawStaticContent's track Y, including the drag-import ghost-gap shift, so the loading
        // overlay stays glued to the static box while a new-track gap is open below/at this track.
        const trackY = loadOffsets[i]! - scrollY + (ghostGap && i >= ghostGap.index ? ghostGap.height : 0);
        // Cull by the FULL track height (matches drawStaticContent) — an expanded track's lane rows
        // extend well below the header, and culling by header alone froze their loading indicators
        // once the header scrolled off the top while the rows were still visible (review-caught).
        if (trackY + computeTrackHeight(track, scale) < 0 || trackY > height) continue;
        const c = trackRgb(track.trackType);
        for (const seg of track.segments) {
          const sx = seg.startTick * ppt - scrollX;
          const sw = seg.durationTicks * ppt;
          if (sx > width || sx + sw < 0) continue;
          // Segment-level loading (import placeholder / source peaks still in flight).
          if (isSegLoading(seg, loadingPaths)) {
            drawLoadingIndicator(ctx as CanvasRenderingContext2D, sx, trackY + 2, sw, headerH - 4, c, now, label);
          }
          // Lane-level loading: an Output-node deposit is decoding this lane's audio. Same look, on the
          // expanded lane row, until the real waveform merges in.
          if (track.expanded && seg.processedOutputs) {
            const layout = getLaneLayout(track);
            for (const out of seg.processedOutputs) {
              if (!out.loading) continue;
              const li = layout.rowByKey.get(laneRowKey(out)) ?? -1;
              if (li < 0) continue;
              const laneY = trackY + layout.rowY[li]! * scale;
              drawLoadingIndicator(ctx as CanvasRenderingContext2D, sx, laneY, sw, laneH, c, now, label);
            }
          }
        }
      }
    }

    // Drag-import ghost (drawn every frame off refs — moves with the OS drag, so it can't live in
    // the cached static layer). One dashed box per dragged file:
    //   • "insert" → file 0 on the hovered audio-track row; extra files preview at the bottom (where
    //     importDroppedFiles appends them).
    //   • "new" → boxes sit in the placeholder gap the static layer opened (between tracks) or in the
    //     empty area below the last track (at end). The opened gap reads as the new track row.
    const ghost = ghostRef.current;
    if (ghost && ghost.target !== "none") {
      const trks = tracksRef.current;
      const offsets = computeTrackYOffsets(trks, scale);
      const paths = dragPathsRef.current;
      const fileCount = Math.max(1, paths.length);
      const rowH = TRACK_HEADER_HEIGHT * scale;
      const boxH = rowH - 4;
      const gx = ghost.tick * ppt - scrollX;
      const durAt = (i: number) =>
        (paths[i] ? dragDurationsRef.current[paths[i]!] : undefined) ?? DEFAULT_DURATION_MS;
      const drawBox = (gy: number, durMs: number) => {
        const w = Math.max(6, durationMsToTicks(durMs, tempoRef.current) * ppt);
        ctx.fillStyle = "rgba(96,165,250,0.22)";
        ctx.fillRect(gx, gy, w, boxH);
        ctx.strokeStyle = "rgba(96,165,250,0.9)";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 4]);
        ctx.strokeRect(gx, gy, w, boxH);
        ctx.setLineDash([]);
      };

      ctx.save();
      if (ghost.target === "insert") {
        drawBox(offsets[ghost.index]! - scrollY + 2, durAt(0));
        if (fileCount > 1) {
          const bottom = computeTotalTracksHeight(trks, scale) - scrollY;
          for (let i = 1; i < fileCount; i++) drawBox(bottom + 2 + (i - 1) * rowH, durAt(i));
        }
      } else {
        const rowTop =
          (ghost.index < trks.length ? offsets[ghost.index]! : computeTotalTracksHeight(trks, scale)) - scrollY;
        for (let i = 0; i < fileCount; i++) drawBox(rowTop + 2 + i * rowH, durAt(i));
      }
      ctx.restore();
    }

    // Snap guide — a dashed vertical line at the clip edge the active drag snapped to (transient,
    // off the cached static layer since it changes every drag frame).
    if ((dragRef.current || ghostRef.current) && snapHintRef.current !== null) {
      const snx = snapHintRef.current * ppt - scrollX;
      if (snx >= -1 && snx <= width + 1) {
        ctx.save();
        ctx.strokeStyle = "rgba(120,220,255,0.85)";
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(snx + 0.5, 0);
        ctx.lineTo(snx + 0.5, height);
        ctx.stroke();
        ctx.restore();
      }
    }

    // Playhead (drawn every frame — just one line + triangle, trivially cheap)
    const phx = playheadTick * ppt - scrollX;
    if (phx >= -1 && phx <= width + 1) {
      const near = Math.abs(mouseXRef.current - phx) < 10;
      drawPlayhead(ctx, { x: phx, height, line: true, glow: near, cap: "top" });
    }
  }, [tracks, audioFiles, loadingPaths, timeSignature, timeAxis, tempo, selectedSegments, selectedLane, dragOver, t]);

  drawRef.current = draw;

  // While anything is still decoding (an import placeholder OR an opened clip's peaks in flight), drive
  // per-frame redraws so the loading spinner animates (the cached static layer just gets blitted each
  // frame — cheap). Stops once nothing is loading.
  useEffect(() => {
    if (
      loadingPaths.length === 0 &&
      !tracks.some((tk) => tk.segments.some((s) => s.loading || s.processedOutputs?.some((o) => o.loading)))
    ) return;
    let raf = 0;
    const tick = () => { requestRedraw(); raf = requestAnimationFrame(tick); };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [tracks, loadingPaths, requestRedraw]);

  useEffect(() => {
    draw();
  }, [draw]);

  // Observe canvas size once — rebuilding the observer on every `draw` change (i.e. every
  // scroll/playback frame, since `draw` depends on scrollX/playheadTick) was per-frame churn.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const observer = new ResizeObserver(() => drawRef.current());
    observer.observe(canvas);
    return () => observer.disconnect();
  }, []);

  return (
    <div className="arrangement">
      <canvas
        ref={canvasRef}
        className="arrangement-canvas"
        style={{ cursor }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleCanvasMouseMove}
        onMouseLeave={handleMouseLeave}
        onDoubleClick={handleDoubleClick}
        onContextMenu={handleContextMenu}
        onWheel={handleWheel}
      />
      {ctxMenu && <ContextMenu x={ctxMenu.x} y={ctxMenu.y} items={ctxItems} onClose={() => setCtxMenu(null)} />}
    </div>
  );
}

function drawStaticContent(
  ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
  width: number, height: number,
  tracks: import("../../types/project").Track[],
  audioFiles: Record<string, import("../../store/audio").AudioTrackData>,
  ppt: number, scrollX: number, scrollY: number,
  axis: TimeAxis, tempo: number,
  selectedKeys: Set<string>,
  selectedLaneKey: string | null,
  dragOver: boolean,
  scale: number,
  ghostGap: { index: number; height: number } | null,
) {
  // Open a new waveform-cache generation: every getWaveformCache call below is protected from
  // eviction for this pass, so the visible working set never thrashes (rebuilds) frame-to-frame.
  beginWaveformFrame();

  ctx.fillStyle = "#0d1220";
  ctx.fillRect(0, 0, width, height);

  if (dragOver) {
    ctx.fillStyle = rgba(ACCENT_RGB, 0.06);
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = rgba(ACCENT_RGB, 0.4);
    ctx.lineWidth = 1;
    ctx.strokeRect(1.5, 1.5, width - 3, height - 3);
  }

  drawBeatGrid(ctx, { ppt, scrollX, width, height, axis, barAlpha: 0.18, beatAlpha: 0.05 });

  const yOffsets = computeTrackYOffsets(tracks, scale);
  const headerH = TRACK_HEADER_HEIGHT * scale;
  const laneH = LANE_HEIGHT * scale - 2;
  for (let i = 0; i < tracks.length; i++) {
    const track = tracks[i];
    if (!track) continue;
    // Tracks at/after the ghost gap shift down to open a placeholder row for the dragged new track.
    const y = yOffsets[i]! - scrollY + (ghostGap && i >= ghostGap.index ? ghostGap.height : 0);
    const trackH = computeTrackHeight(track, scale);
    // Viewport-cull tracks scrolled fully out of view — skip all their box/waveform/lane/crossfade work.
    if (y + trackH < -CULL_PAD || y > height + CULL_PAD) continue;

    ctx.strokeStyle = rgba(SEPARATOR_RGB, 0.8); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, y + trackH); ctx.lineTo(width, y + trackH); ctx.stroke();

    const c = trackRgb(track.trackType);

    // Shared per-track lane geometry (rows + group bars) — the header column uses the SAME layout,
    // so canvas rows and header rows stay pixel-aligned by construction.
    const lanes = getLanes(track);
    const layout = getLaneLayout(track);

    for (const seg of track.segments) {
      const sx = seg.startTick * ppt - scrollX;
      const sw = seg.durationTicks * ppt;
      // Viewport-cull segments fully left/right of the visible range (box + waveform + lanes).
      if (sx > width + CULL_PAD || sx + sw < -CULL_PAD) continue;
      const sy = y + 2;
      const sh = headerH - 4;

      const isSel = selectedKeys.has(`${track.id}:${seg.id}`);

      ctx.fillStyle = isSel ? `rgba(${c[0]},${c[1]},${c[2]},0.28)` : `rgba(${c[0]},${c[1]},${c[2]},0.15)`;
      ctx.fillRect(sx, sy, sw, sh);
      // Selected segments get a soft gold glow outline — distinct from any track colour and the
      // pink playhead, so multi-selection reads clearly without a harsh hard line.
      if (isSel) {
        ctx.save();
        ctx.shadowColor = `rgba(${SELECTION_GLOW_RGB},0.9)`;
        ctx.shadowBlur = 8;
        ctx.strokeStyle = `rgba(${SELECTION_GLOW_RGB},0.7)`;
        ctx.lineWidth = 1.5;
        ctx.strokeRect(sx, sy, sw, sh);
        ctx.restore();
      } else {
        ctx.strokeStyle = `rgba(${c[0]},${c[1]},${c[2]},0.5)`;
        ctx.lineWidth = 1;
        ctx.strokeRect(sx, sy, sw, sh);
      }

      ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},0.3)`;
      ctx.fillRect(sx, sy, 3, sh);
      ctx.fillRect(sx + sw - 3, sy, 3, sh);

      // The loading look (stripes + spinner) is drawn per-frame in the main draw() via
      // drawLoadingIndicator (keyed on isSegLoading) — NOT here — so it can animate. The static layer
      // draws the waveform once peaks are ready; an as-yet-undecoded clip is just an empty box that the
      // per-frame indicator overlays.
      if (seg.content.type === "audioClip" && sw > 2 && seg.content.totalDurationMs > 0) {
        const offMs = seg.content.offsetMs;
        const totalMs = seg.content.totalDurationMs;
        const segMs = ticksToMs(seg.durationTicks, tempo);
        const startRatio = offMs / totalMs;
        const endRatio = Math.min(1, (offMs + segMs) / totalMs);
        // WHAT PLAYS IS WHAT SHOWS: when the sub-lanes are the source (segmentPlaysLanes), the main
        // row shows their REAL audible sum (row mutes + slice recipes respected — stem space, so the
        // same window ratios apply); the original waveform shows only when the original audio plays
        // (no ready lanes, or the track's playOriginal bypass).
        const sumPeaks = segmentPlaysLanes(track, seg) ? segmentLaneSumPeaks(track, seg) : null;
        if (sumPeaks) {
          // The cache id carries the EXACT input signature (laneSumSig): the generic 5-point
          // peaksSignature in the bitmap-cache key collides for localized edits (a slice/delete
          // between its sampled buckets), which would blit a stale pre-edit bitmap (review-caught).
          drawWaveform(ctx, `lanesum:${track.id}:${seg.id}:${laneSumSig(track, seg)}`, sumPeaks, `rgba(${c[0]},${c[1]},${c[2]},0.6)`, sx, sy, sw, sh, startRatio, endRatio, width);
        } else {
          const audio = audioFiles[seg.content.sourcePath];
          if (audio && audio.peaks.length > 0) {
            drawWaveform(ctx, seg.content.sourcePath, audio.peaks, `rgba(${c[0]},${c[1]},${c[2]},0.6)`, sx, sy, sw, sh, startRatio, endRatio, width);
          }
        }
      }

      if (seg.processedOutputs && seg.processedOutputs.length > 0) {
        ctx.fillStyle = "#4ade80";
        ctx.beginPath();
        ctx.arc(sx + 8, sy + 8, 3, 0, Math.PI * 2);
        ctx.fill();
      }

      if (track.muted) {
        ctx.fillStyle = "rgba(13, 18, 32, 0.5)";
        ctx.fillRect(sx, sy, sw, sh);
      }

      if (track.expanded && seg.processedOutputs) {
        for (const out of seg.processedOutputs) {
          // Position through the shared rowByKey map (covers MERGED rows: every member rowKey resolves
          // to its visual row) — never the array index, so the row always matches computeTrackHeight.
          const li = layout.rowByKey.get(laneRowKey(out)) ?? -1;
          if (li < 0) continue;
          const laneY = y + layout.rowY[li]! * scale;
          // Color by the 轨道组 NAME (run.colorIndex — same name shares a hue; 组 boundaries are the
          // bars); the header column's group bar uses the same LANE_COLORS entry, so the sides echo.
          const laneColor = LANE_COLORS[layout.runs[layout.rowRun[li]!]!.colorIndex % LANE_COLORS.length]!;
          const group = laneGroupId(out);
          const stored = seg.laneOps?.[group];

          // Faint row background spanning the whole segment = the lane's "slot"; any carved-out (sliced /
          // trimmed / deleted) region stays just this bg → reads as silence.
          ctx.fillStyle = `rgba(${laneColor},0.08)`;
          ctx.fillRect(sx, laneY, sw, laneH);
          ctx.strokeStyle = `rgba(${laneColor},0.3)`;
          ctx.lineWidth = 0.5;
          ctx.strokeRect(sx, laneY, sw, laneH);

          if (seg.content.type === "audioClip" && out.waveformPeaks && out.waveformPeaks.length > 0 && sw > 2) {
            const lTotalMs = seg.content.totalDurationMs;
            // Light + thin lane-colour edge handles at a piece's two edges — mirrors the main-track segment's
            // edge bars, so EVERY cut reads as a left piece meeting a right piece (each hoverable) whether it
            // came from a laneOps slice OR a main-track SPLIT boundary between two segments' sub-lanes (each
            // half draws its own handles, so their meeting looks the same as a slice). Kept subtle so it's
            // not abrupt. NB: purely visual — hit-detection is unchanged.
            const drawEdgeBars = (px: number, pw: number) => {
              ctx.fillStyle = `rgba(${laneColor},0.3)`;
              ctx.fillRect(px, laneY + 1, 1.5, laneH - 2);
              ctx.fillRect(px + pw - 1.5, laneY + 1, 1.5, laneH - 2);
            };
            if (!stored) {
              // UNEDITED lane → the exact pre-P3 single-window draw (byte-identical, zero regression). The
              // stem spans the whole (trimmed) source, so draw the SAME [offset, offset+segment] window as
              // the original audio above (whole stem 0..1 would shift the lane vs the main track).
              const lSegMs = ticksToMs(seg.durationTicks, tempo);
              const lStart = lTotalMs > 0 ? seg.content.offsetMs / lTotalMs : 0;
              const lEnd = lTotalMs > 0 ? Math.min(1, (seg.content.offsetMs + lSegMs) / lTotalMs) : 1;
              drawWaveform(ctx, out.audioPath, out.waveformPeaks, `rgba(${laneColor},0.6)`, sx, laneY, sw, laneH, lStart, lEnd, width);
              drawEdgeBars(sx, sw);
            } else {
              // EDITED lane → draw each kept piece's waveform window + its edge handles; the gaps between
              // pieces stay the faint bg (silence). Ratios are into the stem peaks with the SAME source-total
              // denominator as the main track, so lane + main stay aligned.
              for (const p of laneVisiblePieces(seg, stored, lTotalMs, tempo)) {
                const px = p.startTick * ppt - scrollX;
                const pw = (p.endTick - p.startTick) * ppt;
                if (pw < 0.5 || px > width + CULL_PAD || px + pw < -CULL_PAD) continue;
                const lStart = lTotalMs > 0 ? p.startMs / lTotalMs : 0;
                const lEnd = lTotalMs > 0 ? Math.min(1, p.endMs / lTotalMs) : 1;
                drawWaveform(ctx, out.audioPath, out.waveformPeaks, `rgba(${laneColor},0.6)`, px, laneY, pw, laneH, lStart, lEnd, width);
                drawEdgeBars(px, pw);
              }
            }
          }

          ctx.fillStyle = `rgba(${laneColor},0.7)`;
          ctx.font = "9px monospace";
          // Show the ROW's sub-name (labels are "Group · stem") — the row label (not the raw per-segment
          // laneLabel) so the canvas text matches the header column exactly, numbering included.
          const rowLabel = lanes[li]!.label;
          ctx.fillText(laneLabelParts(rowLabel).stem ?? rowLabel, sx + 4, laneY + 10);

          // Dim when the row is out of the audible output: its own MUTE, or the track's playOriginal
          // bypass (all lanes leave the output — the header column dims its group blocks the same way).
          if (track.playOriginal || isLaneRowMuted(track, laneRowKey(out), out.laneId)) {
            ctx.fillStyle = "rgba(13, 18, 32, 0.5)";
            ctx.fillRect(sx, laneY, sw, laneH);
          }

          // Selected sub-lane GROUP → a soft gold outline on every row of that group (matches the segment
          // selection glow), so clicking one lane / its Output node lights up the whole group.
          if (selectedLaneKey === `${track.id}:${seg.id}:${group}`) {
            ctx.save();
            ctx.shadowColor = `rgba(${SELECTION_GLOW_RGB},0.9)`;
            ctx.shadowBlur = 6;
            ctx.strokeStyle = `rgba(${SELECTION_GLOW_RGB},0.7)`;
            ctx.lineWidth = 1.25;
            ctx.strokeRect(sx + 0.5, laneY + 0.5, sw - 1, laneH - 1);
            ctx.restore();
          }
        }
      }
    }

    if (track.expanded && lanes.length > 0) {
      // Group delineation: one slim BAND per 组+名 run (aligned 1:1 with the header column's group
      // bar), a stronger line at each band top (= the GROUP boundary), and light separators only
      // BETWEEN rows within a run — so grouping reads at a glance without upstaging the waveforms.
      for (let r = 0; r < layout.runs.length; r++) {
        const run = layout.runs[r]!;
        const bandY = y + run.barY * scale;
        const bandH = LANE_GROUP_BAR_HEIGHT * scale;
        const rc = LANE_COLORS[run.colorIndex % LANE_COLORS.length]!;
        ctx.fillStyle = `rgba(${rc},0.05)`;
        ctx.fillRect(0, bandY, width, bandH);
        ctx.strokeStyle = rgba(SEPARATOR_RGB, 0.8); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(0, bandY); ctx.lineTo(width, bandY); ctx.stroke();
        ctx.strokeStyle = rgba(SEPARATOR_RGB, 0.4); ctx.lineWidth = 0.5;
        ctx.beginPath(); ctx.moveTo(0, bandY + bandH); ctx.lineTo(width, bandY + bandH); ctx.stroke();
        for (let k = 1; k < run.count; k++) {
          const ry = y + layout.rowY[run.start + k]! * scale;
          ctx.strokeStyle = rgba(SEPARATOR_RGB, 0.5); ctx.lineWidth = 0.5;
          ctx.beginPath(); ctx.moveTo(0, ry); ctx.lineTo(width, ry); ctx.stroke();
        }
      }
    }

    const timeSorted = [...track.segments].filter((s) => !s.loading).sort((a, b) => a.startTick - b.startTick);
    const xfLanes = track.expanded ? lanes : null;
    for (let si = 0; si + 1 < timeSorted.length; si++) {
      const seg = timeSorted[si]!;
      const next = timeSorted[si + 1]!;
      const segEnd = seg.startTick + seg.durationTicks;
      if (next.startTick < segEnd) {
        const overlapEnd = Math.min(segEnd, next.startTick + next.durationTicks);
        const ox = next.startTick * ppt - scrollX;
        const ow = (overlapEnd - next.startTick) * ppt;
        if (ox > width + CULL_PAD || ox + ow < -CULL_PAD) continue; // cull crossfades scrolled out of view
        drawCrossfade(ctx as CanvasRenderingContext2D, ox, y + 2, ow, headerH - 4, c);
        // Mirror the X onto each sub-lane row the two halves SHARE (ready, non-loading) — the row
        // crossfades in playback too, so it should read the same as the main clip.
        if (xfLanes) {
          for (const out of seg.processedOutputs ?? []) {
            if (out.loading) continue;
            const rowKey = laneRowKey(out);
            // Same gate as playback (laneReachesSeam): draw the X only when BOTH sides' kept pieces
            // actually reach the seam — a piece trimmed away from the boundary fades nothing, so the
            // mark would promise a crossfade the audio doesn't do.
            if (!laneReachesSeam(seg, rowKey, tempo, "end") || !laneReachesSeam(next, rowKey, tempo, "start")) continue;
            const li = layout.rowByKey.get(rowKey) ?? -1;
            if (li < 0) continue;
            drawCrossfade(ctx as CanvasRenderingContext2D, ox, y + layout.rowY[li]! * scale, ow, laneH, c);
          }
        }
      }
    }
  }

  // Placeholder row for a new track being dragged in between existing tracks (the gap opened above).
  if (ghostGap) {
    const gy = (yOffsets[ghostGap.index] ?? computeTotalTracksHeight(tracks, scale)) - scrollY;
    ctx.fillStyle = "rgba(96,165,250,0.07)";
    ctx.fillRect(0, gy, width, ghostGap.height);
    ctx.strokeStyle = "rgba(96,165,250,0.35)";
    ctx.lineWidth = 1;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(1, gy + 1, width - 2, ghostGap.height - 2);
    ctx.setLineDash([]);
  }
}

/** One predicate for "this clip's audio isn't ready" — covers BOTH a fresh import placeholder
 *  (seg.loading, State A) AND an opened clip whose source peaks are still in flight (State B). Used
 *  for the VISUAL loading indicator only. NB: seg.loading alone (not this) stays the test for
 *  interactivity/collision (a State-B clip has a real duration and IS interactive). */
function isSegLoading(seg: Segment, loadingPaths: string[]): boolean {
  if (seg.loading) return true;
  return seg.content.type === "audioClip" && loadingPaths.includes(seg.content.sourcePath);
}

/** Unified loading visual for ANY loading segment (State A or B): sweeping diagonal stripes + a
 *  rotating spinner + "加载中…" near the start. One code path so every loading clip looks identical.
 *  Drawn per-frame (the rAF loop runs while anything loads) so the stripes sweep + spinner spins. */
function drawLoadingIndicator(
  ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, c: number[], now: number, label: string,
) {
  if (w < 2) return;
  ctx.save();
  ctx.beginPath();
  ctx.rect(x, y, w, h);
  ctx.clip();
  // sweeping diagonal stripes
  ctx.strokeStyle = rgba(c, 0.28);
  ctx.lineWidth = 5;
  const gap = 14;
  const shift = (now / 35) % gap; // slow sweep so it clearly reads as "in progress"
  for (let dx = -h - gap + shift; dx < w + h; dx += gap) {
    ctx.beginPath();
    ctx.moveTo(x + dx, y + h);
    ctx.lineTo(x + dx + h, y);
    ctx.stroke();
  }
  // rotating spinner + label near the start
  const ph = now / 130;
  const text = `${label}${".".repeat(1 + (Math.floor(now / 400) % 3))}`;
  ctx.font = "600 11px sans-serif";
  ctx.textBaseline = "middle";
  const withText = w > ctx.measureText(text).width + 34;
  const cx = withText ? x + 13 : x + w / 2;
  const cy = withText ? y + h - 8 : y + h / 2;
  ctx.lineWidth = 2;
  ctx.lineCap = "round";
  ctx.strokeStyle = ACCENT;
  ctx.beginPath();
  ctx.arc(cx, cy, 5, ph, ph + Math.PI * 1.4);
  ctx.stroke();
  if (withText) {
    ctx.shadowColor = "rgba(0,0,0,0.85)";
    ctx.shadowBlur = 4;
    ctx.fillStyle = "#dbe6f0";
    ctx.fillText(text, cx + 11, cy);
  }
  ctx.restore();
}

function drawCrossfade(
  ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, c: number[],
) {
  if (w < 2) return;
  ctx.save(); ctx.globalAlpha = 0.25;
  ctx.strokeStyle = `rgb(${c[0]},${c[1]},${c[2]})`; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + w, y + h);
  ctx.moveTo(x, y + h); ctx.lineTo(x + w, y); ctx.stroke();
  ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},0.08)`; ctx.fillRect(x, y, w, h);
  ctx.restore();
}
