import { invoke } from "@tauri-apps/api/core";
import * as playback from "./playback";
import { flooredDurationTicks } from "./laneOps";
import type { Segment } from "../../types/project";
import { useProjectStore } from "../../store/project";
import { useAudioStore } from "../../store/audio";
import { useAppStore } from "../../store/app";
import { useHistoryStore } from "../../store/history";
import { flushAutosaveNow } from "../project/autosave";

/** Fallback placeholder duration (ms) when a fast probe can't determine the real length. */
export const DEFAULT_DURATION_MS = 8000;

/**
 * Fast duration probe (no full decode) — drives the drag ghost preview and the loading
 * segment's initial width before the file is decoded. Never throws: on failure it falls back
 * to {@link DEFAULT_DURATION_MS} so the UI still shows something sensible.
 */
export async function probeAudioDuration(filePath: string): Promise<number> {
  try {
    const ms = await invoke<number>("probe_audio_duration", { path: filePath });
    return ms > 0 ? ms : DEFAULT_DURATION_MS;
  } catch {
    return DEFAULT_DURATION_MS;
  }
}

function fileName(filePath: string): string {
  return filePath.split(/[/\\]/).pop() ?? "audio";
}

function msToTicks(ms: number): number {
  // flooredDurationTicks = THE shared 1-tick-floor conversion, so the placeholder width always matches
  // the final decoded segment (finalizeSegment uses the same fn) for sub-beat clips.
  return flooredDurationTicks(ms, useProjectStore.getState().tempo);
}

/**
 * Import an audio file onto a brand-new audio track. A striped loading segment appears
 * immediately at `startTick` (sized to `knownDurationMs` if supplied, else a fast probe),
 * then the file is decoded asynchronously and the segment is populated. On failure the
 * placeholder (and its just-created track) is removed and a toast is shown.
 *
 * Used by the click-import ("+") menu (appends) and drag-import (`insertIndex` positions the new
 * track at the dragged spot in the track-header column instead of always appending).
 */
export async function importAudioToNewTrack(
  filePath: string,
  startTick: number,
  knownDurationMs?: number,
  insertIndex?: number,
): Promise<void> {
  const trackId = crypto.randomUUID();
  const segId = crypto.randomUUID();
  const durMs = knownDurationMs ?? (await probeAudioDuration(filePath));

  const seg: Segment = {
    id: segId,
    startTick,
    durationTicks: msToTicks(durMs),
    loading: true,
    content: { type: "audioClip", sourcePath: filePath, offsetMs: 0, totalDurationMs: durMs },
  };

  useProjectStore.getState().addTrack({
    id: trackId,
    name: fileName(filePath),
    trackType: "audio",
    segments: [seg],
    volumeDb: 0,
    pan: 0,
    muted: false,
    solo: false,
    expanded: false,
    laneControls: {},
  }, insertIndex);

  await finalizeSegment(filePath, trackId, segId, true);
}

/**
 * Import an audio file as a new segment on an EXISTING audio track (drag smart-insert under
 * the cursor). Falls back to a new track if the target track has vanished. Same loading →
 * decode → populate / remove-on-failure flow as {@link importAudioToNewTrack}.
 */
export async function importAudioToExistingTrack(
  filePath: string,
  trackId: string,
  startTick: number,
  knownDurationMs?: number,
): Promise<void> {
  const durMs = knownDurationMs ?? (await probeAudioDuration(filePath));
  const track = useProjectStore.getState().tracks.find((t) => t.id === trackId);
  if (!track) {
    await importAudioToNewTrack(filePath, startTick, durMs);
    return;
  }

  const segId = crypto.randomUUID();
  const seg: Segment = {
    id: segId,
    startTick,
    durationTicks: msToTicks(durMs),
    loading: true,
    content: { type: "audioClip", sourcePath: filePath, offsetMs: 0, totalDurationMs: durMs },
  };
  useProjectStore.getState().updateTrack(trackId, { segments: [...track.segments, seg] });

  await finalizeSegment(filePath, trackId, segId, false);
}

/**
 * Decode the file and replace the loading placeholder with the real waveform/duration.
 * If the user deleted the segment (or its track) mid-load, do nothing. On decode failure
 * remove the segment — and, when `removeTrackOnFail` and the track is now empty, the track —
 * then surface the error (single owner of import errors, no re-throw).
 */
async function finalizeSegment(
  filePath: string,
  trackId: string,
  segId: string,
  removeTrackOnFail: boolean,
): Promise<void> {
  const { updateTrack, removeTrack } = useProjectStore.getState();
  try {
    const audioData = await useAudioStore.getState().loadAudioFile(filePath);
    const tempo = useProjectStore.getState().tempo;
    const durationTicks = flooredDurationTicks(audioData.durationMs, tempo);

    const track = useProjectStore.getState().tracks.find((t) => t.id === trackId);
    if (!track || !track.segments.some((s) => s.id === segId)) return; // deleted mid-load

    // The loading→loaded transition is a SYSTEM update (post-decode), not a user edit — record it
    // silently so undo collapses the whole import to its originating step (and undoing mid-decode
    // can't strand a half-loaded segment as its own undo state).
    useHistoryStore.getState().runSilent(() =>
      updateTrack(trackId, {
        segments: track.segments.map((s) =>
          s.id === segId
            ? {
                ...s,
                durationTicks,
                loading: false,
                content:
                  s.content.type === "audioClip"
                    ? { ...s.content, totalDurationMs: audioData.durationMs }
                    : s.content,
              }
            : s,
        ),
      }),
    );

    // Import is a milestone (decode just finished) — snapshot to disk NOW so a fast reload right after
    // doesn't lose the new audio to the 1.5s autosave debounce. (Renders do the same in WorkflowEditor.)
    flushAutosaveNow();

    // Non-blocking waveform prewarm — swallow failures (the segment is already populated).
    playback.loadAudioBuffer(audioData.playbackPath || filePath).catch(() => {});
  } catch (e) {
    const track = useProjectStore.getState().tracks.find((t) => t.id === trackId);
    if (track) {
      const remaining = track.segments.filter((s) => s.id !== segId);
      // Error cleanup is not a user edit either — record silently (never an undo step).
      useHistoryStore.getState().runSilent(() => {
        if (removeTrackOnFail && remaining.length === 0) {
          removeTrack(trackId);
        } else {
          updateTrack(trackId, { segments: remaining });
        }
      });
    }
    useAppStore.getState().showToast(e instanceof Error ? e.message : String(e), "error");
  }
}
