import { useProjectStore } from "../store/project";
import { useAudioStore } from "../store/audio";
import { useAppStore } from "../store/app";
import {
  sliceClips, deleteClip, isWholeClips, ticksToMs, clipIndexAtMs, materializeClips, laneGroupId, segStretch,
} from "./audio/laneOps";
import type { Segment } from "../types/project";

/**
 * Imperative sub-lane edit commands (P3) — the ONE place Ctrl+K / Delete (Toolbar) AND the right-click
 * menu (Arrangement) route through, so the two entry points can never drift. Each reads the live project
 * store, mutates the segment's laneOps (undoable — laneOps is in meaningfulSig), keeps the app-store
 * selection in sync, and reschedules playback. The heavy lifting is the pure helpers in ./audio/laneOps.
 */

function findSeg(trackId: string, segmentId: string): Segment | undefined {
  return useProjectStore.getState().tracks.find((t) => t.id === trackId)?.segments.find((s) => s.id === segmentId);
}

function reschedule(): void {
  if (useAudioStore.getState().isPlaying) useAudioStore.getState().bumpSchedule();
}

/** After an index-shifting edit (slice inserts a piece / delete removes one), re-point the sub-lane
 *  selection's clipIndex to the piece now at `anchorMs`, so a subsequent keyboard Delete never targets a
 *  stale positional index (it would otherwise silently silence a piece the user never selected). No-op
 *  unless the current selection is exactly this segment+group. */
function resyncSelectedClip(trackId: string, segmentId: string, outputNodeId: string, stemDurMs: number, anchorMs: number): void {
  const lane = useAppStore.getState().selectedLane;
  if (!lane || lane.trackId !== trackId || lane.segmentId !== segmentId || lane.outputNodeId !== outputNodeId) return;
  const stored = findSeg(trackId, segmentId)?.laneOps?.[outputNodeId];
  useAppStore.setState({ selectedLane: { ...lane, clipIndex: clipIndexAtMs(stored, stemDurMs, anchorMs) } });
}

/** Slice a sub-lane group at the current playhead: splits the piece the playhead sits in. No-op if the
 *  playhead is outside the segment span (matches the main-track split gate). */
export function sliceLaneGroupAtPlayhead(trackId: string, segmentId: string, outputNodeId: string): void {
  const st = useProjectStore.getState();
  const seg = findSeg(trackId, segmentId);
  if (!seg || seg.content.type !== "audioClip") return;
  const ph = st.playheadTick;
  if (ph <= seg.startTick || ph >= seg.startTick + seg.durationTicks) return;
  const stemDurMs = seg.content.totalDurationMs;
  // S59: the cut position is a SOURCE (stem) coordinate — a stretched box's tick distance covers /r source ms
  const cutMs = seg.content.offsetMs + ticksToMs(ph - seg.startTick, st.tempo) / segStretch(seg);
  const clips = sliceClips(seg.laneOps?.[outputNodeId], stemDurMs, cutMs);
  st.updateSegmentLaneOps(trackId, segmentId, outputNodeId, isWholeClips(clips, stemDurMs) ? undefined : clips);
  resyncSelectedClip(trackId, segmentId, outputNodeId, stemDurMs, cutMs);
  reschedule();
}

/** Delete one piece of a sub-lane group → a silent gap there (deleting the sole piece silences the whole
 *  lane). Non-destructive: only the recipe changes, the rendered stem is untouched. */
export function deleteLanePiece(trackId: string, segmentId: string, outputNodeId: string, clipIndex: number): void {
  const seg = findSeg(trackId, segmentId);
  if (!seg || seg.content.type !== "audioClip") return;
  const stemDurMs = seg.content.totalDurationMs;
  // Anchor at the START of the piece being removed so the selection lands on the adjacent remaining piece.
  const anchorMs = materializeClips(seg.laneOps?.[outputNodeId], stemDurMs)[clipIndex]?.start ?? seg.content.offsetMs;
  const clips = deleteClip(seg.laneOps?.[outputNodeId], stemDurMs, clipIndex);
  useProjectStore.getState().updateSegmentLaneOps(trackId, segmentId, outputNodeId, isWholeClips(clips, stemDurMs) ? undefined : clips);
  resyncSelectedClip(trackId, segmentId, outputNodeId, stemDurMs, anchorMs);
  reschedule();
}

/** The current sub-lane selection IF it is still LIVE — its track is EXPANDED and the segment still
 *  carries a ready lane whose group matches — with its clipIndex CLAMPED to the current recipe. Otherwise
 *  clears the stale selection and returns null so Ctrl+K / Delete fall through to the SEGMENT op. Without
 *  this, a lane selection surviving a track collapse / Clear Render (nothing clears it there) would make
 *  Ctrl+K/Delete/the Split button silently edit an INVISIBLE, inaudible lane instead of splitting/deleting
 *  the segment — a regression of the pre-P3 keyboard behaviour. */
export function liveSelectedLane(): { trackId: string; segmentId: string; outputNodeId: string; clipIndex: number } | null {
  const lane = useAppStore.getState().selectedLane;
  if (!lane) return null;
  const track = useProjectStore.getState().tracks.find((t) => t.id === lane.trackId);
  const seg = track?.segments.find((s) => s.id === lane.segmentId);
  const live = !!track?.expanded && !!seg && seg.content.type === "audioClip"
    && !!seg.processedOutputs?.some((o) => !o.loading && laneGroupId(o) === lane.outputNodeId);
  if (!live || !seg || seg.content.type !== "audioClip") {
    if (lane) useAppStore.setState({ selectedLane: null });
    return null;
  }
  const clips = materializeClips(seg.laneOps?.[lane.outputNodeId], seg.content.totalDurationMs);
  return { ...lane, clipIndex: Math.max(0, Math.min(lane.clipIndex, clips.length - 1)) };
}
