import { readFile } from "@tauri-apps/plugin-fs";
import { laneGroupId, laneReachesSeam, laneRowKey, laneVisiblePieces, msToTicks, ticksToMs } from "./laneOps";
import { isLaneRowMuted, laneControlFor, segmentPlaysLanes } from "../trackLayout";
import { useProjectStore } from "../../store/project";
import type { Track } from "../../types/project";
import type { AudioTrackData } from "../../store/audio";

let audioCtx: AudioContext | null = null;
const loadedBuffers = new Map<string, AudioBuffer>();

interface ScheduledSource {
  trackId: string;
  /** The lane's 组 id (`laneGroupId` = producing Output node) — the laneControls (volume/pan) identity,
   *  so live volume updates from the header faders (which key by 组) target every source of the 组. */
  groupId?: string;
  /** The lane ROW key (`laneRowKey`) — live per-row MUTE updates match on this (mute is row-scoped). */
  rowKey?: string;
  source: AudioBufferSourceNode;
  trackGain: GainNode;
  laneGain?: GainNode;
  fadeIn?: GainNode;
  fadeOut?: GainNode;
  panner: StereoPannerNode;
  /** The two pan layers COMPOSE (sum, clamped to ±1): panner.pan = clampPan(trackPan + lanePan), so the
   *  track pan keeps shifting lane sources even after a 组 pan is set. Kept on the source so a live
   *  update to EITHER layer can recompute the composite without re-reading the stores. */
  trackPan: number;
  lanePan: number;
  silenced: boolean;
  laneSilenced: boolean;
}

let scheduledSources: ScheduledSource[] = [];
let playGeneration = 0;
let scheduleTimeOrigin = 0;

function getContext(): AudioContext {
  if (!audioCtx || audioCtx.state === "closed") {
    audioCtx = new AudioContext();
  }
  if (audioCtx.state === "suspended") {
    audioCtx.resume();
  }
  return audioCtx;
}

export async function loadAudioBuffer(filePath: string): Promise<AudioBuffer> {
  if (loadedBuffers.has(filePath)) {
    return loadedBuffers.get(filePath)!;
  }
  const ctx = getContext();
  const bytes = await readFile(filePath);
  const arrayBuffer = bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  );
  const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
  loadedBuffers.set(filePath, audioBuffer);
  return audioBuffer;
}

export async function playAllTracks(
  tracks: Track[],
  audioFiles: Record<string, AudioTrackData>,
  playheadTick: number,
  tempo: number,
  onAllEnded: () => void,
): Promise<"started" | "empty" | "superseded"> {
  stopPlayback();
  const gen = ++playGeneration;

  const ctx = getContext();
  const now = ctx.currentTime;
  scheduleTimeOrigin = now;
  const hasSolo = tracks.some((t) => t.solo);
  let totalScheduled = 0;
  let endedCount = 0;
  // Scheduling is INCREMENTAL and can `await` (buffer decode) mid-loop, so a short/late source can END
  // before the loop finishes counting. Defer onAllEnded until every source is scheduled — otherwise an
  // early end fires `endedCount >= totalScheduled` against a partial count, flipping the UI to "stopped" +
  // snapping the playhead to the end while the rest of the audio keeps playing (the split-point ghost).
  let schedulingDone = false;

  // Live-mix read for source creation: the loop can `await` a buffer decode; a fader drag during that
  // window updates the ALREADY-scheduled sources (updateTrack*/updateLane*), but a source created AFTER
  // the await would bake the stale snapshot value in. Mix values (volume/pan/mute/solo/lane controls)
  // are therefore resolved from the LIVE store per source; the schedule STRUCTURE (segments/pieces/
  // crossfades) stays on the snapshot — structural edits bump scheduleVersion and reschedule wholesale.
  const liveMix = (snap: Track) => {
    const live = useProjectStore.getState().tracks;
    const t = live.find((x) => x.id === snap.id) ?? snap;
    const solo = live.length > 0 ? live.some((x) => x.solo) : hasSolo;
    return { t, audible: !t.muted && (!solo || t.solo) };
  };

  for (const track of tracks) {
    const sorted = [...track.segments]
      .filter((s) => s.content.type === "audioClip" && !s.loading)
      .sort((a, b) => a.startTick - b.startTick);

    for (let si = 0; si < sorted.length; si++) {
      const seg = sorted[si]!;
      if (seg.content.type !== "audioClip") continue;

      const segEnd = seg.startTick + seg.durationTicks;
      if (segEnd <= playheadTick) continue;

      // THE source-selection predicate (shared with the main-row waveform + future mixdown): ready
      // lanes play unless the track's playOriginal bypass is on. NOT gated on track.expanded —
      // collapse is pure view state. (`.some(!loading)` inside: while a deposit is still decoding ALL
      // lanes, fall through to the original audio instead of going silent.)
      const hasLaneOutputs = segmentPlaysLanes(track, seg);

      if (hasLaneOutputs) {
        for (const out of seg.processedOutputs!) {
          if (out.loading) continue; // a still-decoding deposit placeholder — nothing to play yet
          const rowKey = laneRowKey(out); // ROW identity — crossfade pairing + per-row mute
          const groupId = laneGroupId(out); // 组 identity — volume/pan ("recorded on the Output node")

          let buf = loadedBuffers.get(out.audioPath);
          if (!buf) {
            try { buf = await loadAudioBuffer(out.audioPath); } catch (e) {
              console.error(`Failed to load lane audio: ${out.audioPath}`, e);
              // The failed await was still an await — without this check a superseded loop would keep
              // scheduling stale sources into the NEW generation's scheduledSources.
              if (gen !== playGeneration) return "superseded";
              continue;
            }
            if (gen !== playGeneration) return "superseded"; // a NEWER playAllTracks superseded this one
          }
          if (!buf) continue;

          // Resolve mix values LIVE, after the possible await (see liveMix above).
          const { t: lt, audible } = liveMix(track);
          const laneCtrl = laneControlFor(lt, groupId, out.laneId);
          const laneMuted = isLaneRowMuted(lt, rowKey, out.laneId);
          const laneVol = laneMuted ? 0 : dbToLinear(laneCtrl?.volumeDb ?? 0);
          const trackVol = audible ? dbToLinear(lt.volumeDb) : 0;

          // Non-destructive sub-lane recipe (D2): the lane plays its KEPT pieces (stem-ms windows). An
          // UNEDITED lane yields exactly ONE whole-window piece → identical scheduling to the pre-P3 single
          // source. Sliced/trimmed/deleted regions simply produce no piece → silence; the stem is untouched.
          const group = laneGroupId(out);
          const stemDurMs = seg.content.type === "audioClip" ? seg.content.totalDurationMs : out.totalDurationMs;
          const pieces = laneVisiblePieces(seg, seg.laneOps?.[group], stemDurMs, tempo);

          for (const piece of pieces) {
            if (piece.endTick <= playheadTick) continue;

            let startDelay: number;
            let audioOffset: number;
            let playDuration: number;

            // A piece reads from its stem-ms offset (== the source offset, since the stem spans the whole
            // trimmed source): seg.content.offsetMs for a whole/first piece, a later position for a sliced one.
            const pieceOffsetSec = piece.startMs / 1000;
            if (playheadTick >= piece.startTick) {
              const secsInto = ticksToSeconds(playheadTick - piece.startTick, tempo);
              audioOffset = pieceOffsetSec + secsInto;
              startDelay = 0;
              playDuration = ticksToSeconds(piece.endTick - piece.startTick, tempo) - secsInto;
            } else {
              audioOffset = pieceOffsetSec;
              startDelay = ticksToSeconds(piece.startTick - playheadTick, tempo);
              playDuration = ticksToSeconds(piece.endTick - piece.startTick, tempo);
            }

            audioOffset = Math.max(0, Math.min(audioOffset, buf.duration));
            playDuration = Math.min(playDuration, buf.duration - audioOffset);
            if (playDuration <= 0) continue;
            // Sync-correct a LATE schedule: a buffer that loaded slowly (heavy track added mid-playback)
            // pushes `now + startDelay` into the PAST; without this it would start from the stale offset and
            // lag the on-time tracks (the desync "ghosting"). Advance the offset by the lateness so it plays
            // in sync instead.
            {
              const late = ctx.currentTime - (now + startDelay);
              if (late > 0) {
                audioOffset = Math.min(buf.duration, audioOffset + late);
                playDuration -= late;
                startDelay += late;
                if (playDuration <= 0) continue;
              }
            }

            const laneGainNode = ctx.createGain();
            laneGainNode.gain.setValueAtTime(laneVol, now);
            const trackGainNode = ctx.createGain();
            trackGainNode.gain.setValueAtTime(trackVol, now);
            const panner = ctx.createStereoPanner();
            const lanePan = laneCtrl?.pan ?? 0;
            panner.pan.value = clampPan(lt.pan + lanePan);

            // Crossfade only the piece that TOUCHES a segment boundary against a neighbour clip carrying the
            // SAME lane ROW (a genuine same-row overlap) — interior slice edges are hard cuts (no fade), and a
            // trimmed-back edge that no longer reaches the boundary doesn't fade. Fades are isolated from
            // laneGain so live volume/mute changes don't fight the ramps (mirrors the original-audio branch).
            const fadeInNode = ctx.createGain();
            const fadeOutNode = ctx.createGain();
            const atSegEnd = Math.abs(piece.endTick - segEnd) < 1;
            const atSegStart = Math.abs(piece.startTick - seg.startTick) < 1;
            if (atSegEnd && si + 1 < sorted.length) {
              const next = sorted[si + 1]!;
              if (next.startTick < segEnd && laneReachesSeam(next, rowKey, tempo, "start")) {
                applyFadeOut(fadeOutNode, next.startTick, Math.min(segEnd, next.startTick + next.durationTicks), playheadTick, tempo, now);
              }
            }
            if (atSegStart && si > 0) {
              const prev = sorted[si - 1]!;
              const prevEnd = prev.startTick + prev.durationTicks;
              if (seg.startTick < prevEnd && laneReachesSeam(prev, rowKey, tempo, "end")) {
                applyFadeIn(fadeInNode, seg.startTick, Math.min(prevEnd, segEnd), playheadTick, tempo, now, startDelay);
              }
            }

            const source = ctx.createBufferSource();
            source.buffer = buf;
            source.connect(fadeInNode).connect(fadeOutNode).connect(laneGainNode).connect(trackGainNode).connect(panner).connect(ctx.destination);
            source.onended = () => { if (gen !== playGeneration) return; endedCount++; if (schedulingDone && endedCount >= totalScheduled) onAllEnded(); };
            source.start(now + startDelay, audioOffset, playDuration);
            scheduledSources.push({
              trackId: track.id, groupId, rowKey, source,
              trackGain: trackGainNode, laneGain: laneGainNode, panner,
              trackPan: lt.pan, lanePan,
              fadeIn: fadeInNode, fadeOut: fadeOutNode,
              silenced: !audible, laneSilenced: laneMuted,
            });
            totalScheduled++;
          }
        }
        continue;
      }

      // Original audio scheduling (no lane outputs)
      const filePath = seg.content.sourcePath;
      const audioMeta = audioFiles[filePath];
      if (!audioMeta) continue;

      // Use WAV cache for non-WAV files to avoid browser codec delay mismatch
      const bufPath = audioMeta.playbackPath || filePath;
      let buf = loadedBuffers.get(bufPath);
      if (!buf) {
        try { buf = await loadAudioBuffer(bufPath); } catch (e) {
          console.error(`[playback] failed to load ${bufPath}:`, e);
          if (gen !== playGeneration) return "superseded"; // see the lane branch's catch
          continue;
        }
        if (gen !== playGeneration) return "superseded"; // a NEWER playAllTracks superseded this one
      }
      if (!buf) continue;

      // Resolve mix values LIVE, after the possible await (see liveMix above).
      const { t: lt, audible } = liveMix(track);
      const trackVol = audible ? dbToLinear(lt.volumeDb) : 0;

      let audioOffset: number;
      let startDelay: number;
      let playDuration: number;

      if (playheadTick >= seg.startTick) {
        const ticksInto = playheadTick - seg.startTick;
        const secsInto = ticksToSeconds(ticksInto, tempo);
        audioOffset = seg.content.offsetMs / 1000 + secsInto;
        startDelay = 0;
        playDuration = ticksToSeconds(seg.durationTicks, tempo) - secsInto;
      } else {
        audioOffset = seg.content.offsetMs / 1000;
        startDelay = ticksToSeconds(seg.startTick - playheadTick, tempo);
        playDuration = ticksToSeconds(seg.durationTicks, tempo);
      }

      audioOffset = Math.max(0, Math.min(audioOffset, buf.duration));
      playDuration = Math.min(playDuration, buf.duration - audioOffset);
      if (playDuration <= 0) continue;
      // Sync-correct a LATE schedule (see the lane branch): advance the offset by how late we are so a
      // buffer that loaded slowly mid-reschedule plays in sync with the on-time tracks, not lagging.
      {
        const late = ctx.currentTime - (now + startDelay);
        if (late > 0) {
          audioOffset = Math.min(buf.duration, audioOffset + late);
          playDuration -= late;
          startDelay += late;
          if (playDuration <= 0) continue;
        }
      }

      const trackGainNode = ctx.createGain();
      trackGainNode.gain.setValueAtTime(trackVol, now);
      const panner = ctx.createStereoPanner();
      panner.pan.value = lt.pan;

      // Separate fade nodes: crossfade automation isolated from track volume. The fade-in/out envelopes are
      // shared with the sub-lane branch via applyFadeIn/applyFadeOut (one source of truth — see bottom).
      const fadeInNode = ctx.createGain();
      const fadeOutNode = ctx.createGain();

      // Crossfade fade-out: overlap with the next segment
      if (si + 1 < sorted.length) {
        const next = sorted[si + 1]!;
        if (next.startTick < segEnd) {
          applyFadeOut(fadeOutNode, next.startTick, Math.min(segEnd, next.startTick + next.durationTicks), playheadTick, tempo, now);
        }
      }

      // Crossfade fade-in: overlap with the previous segment
      if (si > 0) {
        const prev = sorted[si - 1]!;
        const prevEnd = prev.startTick + prev.durationTicks;
        if (seg.startTick < prevEnd) {
          applyFadeIn(fadeInNode, seg.startTick, Math.min(prevEnd, segEnd), playheadTick, tempo, now, startDelay);
        }
      }

      const source = ctx.createBufferSource();
      source.buffer = buf;
      source.connect(fadeInNode).connect(fadeOutNode).connect(trackGainNode).connect(panner).connect(ctx.destination);

      source.onended = () => {
        // Ignore end events from a superseded generation — stopPlayback() (called when a new
        // playback/seek starts) fires onended on the old sources, which would otherwise trip the
        // previous onAllEnded and flip isPlaying off (freezing the playhead during a seek).
        if (gen !== playGeneration) return;
        endedCount++;
        if (schedulingDone && endedCount >= totalScheduled) onAllEnded();
      };

      source.start(now + startDelay, audioOffset, playDuration);
      scheduledSources.push({
        trackId: track.id, source, trackGain: trackGainNode,
        fadeIn: fadeInNode, fadeOut: fadeOutNode, panner,
        trackPan: lt.pan, lanePan: 0,
        silenced: !audible, laneSilenced: false,
      });
      totalScheduled++;
    }
  }

  // Count is final now — honor any end that already happened during the loop (so a genuinely finished
  // playback still ends), without ever having fired against a partial count mid-scheduling. A stale
  // loop must not report at all (its onAllEnded would stop the WINNING generation's UI).
  schedulingDone = true;
  if (gen !== playGeneration) return "superseded";
  if (totalScheduled > 0 && endedCount >= totalScheduled) {
    // Everything finished DURING the scheduling awaits (a sliver right at the content end while a
    // later buffer decoded slowly). onAllEnded has already stopped + snapped the playhead — report
    // "empty" so handleTogglePlay doesn't flip isPlaying back on for a playback with zero live
    // sources (runaway playhead past the end).
    onAllEnded();
    return "empty";
  }

  return totalScheduled > 0 ? "started" : "empty";
}

export function stopPlayback() {
  // Invalidate the current generation so the stopped sources' onended callbacks are gen-guarded out.
  // Otherwise a MANUAL stop (pause) would fire onAllEnded (which now snaps the playhead to the end),
  // making pause jump back to the end. onAllEnded must fire only on a NATURAL end (no stopPlayback).
  playGeneration++;
  for (const s of scheduledSources) {
    try { s.source.stop(); } catch { /* already stopped */ }
  }
  scheduledSources = [];
}

export function updateTrackVolume(trackId: string, volumeDb: number) {
  const vol = dbToLinear(volumeDb);
  for (const s of scheduledSources) {
    if (s.trackId === trackId && !s.silenced) {
      s.trackGain.gain.setValueAtTime(vol, s.trackGain.context.currentTime);
    }
  }
}

export function updateTrackPan(trackId: string, pan: number) {
  for (const s of scheduledSources) {
    if (s.trackId === trackId) {
      s.trackPan = pan;
      s.panner.pan.setValueAtTime(clampPan(s.trackPan + s.lanePan), s.panner.context.currentTime);
    }
  }
}

export function updateTrackAudibility(tracks: Array<{ id: string; muted: boolean; solo: boolean; volumeDb: number }>) {
  const hasSolo = tracks.some((t) => t.solo);
  for (const s of scheduledSources) {
    const track = tracks.find((t) => t.id === s.trackId);
    if (!track) continue;
    const audible = !track.muted && (!hasSolo || track.solo);
    s.silenced = !audible;
    s.trackGain.gain.setValueAtTime(
      audible ? dbToLinear(track.volumeDb) : 0,
      s.trackGain.context.currentTime,
    );
  }
}

export function updateLaneVolume(trackId: string, groupId: string, volumeDb: number) {
  const vol = dbToLinear(volumeDb);
  for (const s of scheduledSources) {
    if (s.trackId === trackId && s.groupId === groupId && s.laneGain && !s.laneSilenced) {
      s.laneGain.gain.setValueAtTime(vol, s.laneGain.context.currentTime);
    }
  }
}

export function updateLanePan(trackId: string, groupId: string, pan: number) {
  for (const s of scheduledSources) {
    if (s.trackId === trackId && s.groupId === groupId) {
      s.lanePan = pan;
      s.panner.pan.setValueAtTime(clampPan(s.trackPan + s.lanePan), s.panner.context.currentTime);
    }
  }
}

export function updateLaneMute(trackId: string, rowKey: string, muted: boolean, volumeDb: number) {
  for (const s of scheduledSources) {
    if (s.trackId === trackId && s.rowKey === rowKey && s.laneGain) {
      s.laneSilenced = muted;
      s.laneGain.gain.setValueAtTime(
        muted ? 0 : dbToLinear(volumeDb),
        s.laneGain.context.currentTime,
      );
    }
  }
}

export function clearBufferCache(filePath?: string) {
  if (filePath) {
    loadedBuffers.delete(filePath);
  } else {
    loadedBuffers.clear();
  }
}

export function getContextTime(): number {
  return audioCtx?.currentTime ?? 0;
}

export function getScheduleTimeOrigin(): number {
  return scheduleTimeOrigin;
}

// Second-based wrappers around THE tick<->ms conversions in laneOps.ts (the single formula source).
export function ticksToSeconds(ticks: number, tempo: number): number {
  return ticksToMs(ticks, tempo) / 1000;
}

export function secondsToTicks(secs: number, tempo: number): number {
  return msToTicks(secs * 1000, tempo);
}

export function durationMsToTicks(ms: number, tempo: number): number {
  return msToTicks(ms, tempo);
}

function dbToLinear(db: number): number {
  return Math.pow(10, db / 20);
}

/** StereoPannerNode.pan is [-1, 1] — the composed track+lane pan must stay in range. */
function clampPan(p: number): number {
  return Math.max(-1, Math.min(1, p));
}

/** Linear crossfade FADE-OUT envelope on `node` for a source overlapping a LATER neighbour over
 *  [fadeStartTick, fadeEndTick]. Handles the playhead landing past / inside / before the overlap. Shared by
 *  the original-audio AND sub-lane branches so they crossfade identically (one source of truth). */
function applyFadeOut(
  node: GainNode, fadeStartTick: number, fadeEndTick: number, playheadTick: number, tempo: number, now: number,
) {
  const fadeLenTicks = fadeEndTick - fadeStartTick;
  if (fadeLenTicks <= 0) return;
  if (playheadTick >= fadeEndTick) {
    node.gain.setValueAtTime(0, now);
  } else if (playheadTick > fadeStartTick) {
    const progress = (playheadTick - fadeStartTick) / fadeLenTicks;
    node.gain.setValueAtTime(1 - progress, now);
    node.gain.linearRampToValueAtTime(0, now + ticksToSeconds(fadeEndTick - playheadTick, tempo));
  } else {
    node.gain.setValueAtTime(1, now + ticksToSeconds(fadeStartTick - playheadTick, tempo));
    node.gain.linearRampToValueAtTime(0, now + ticksToSeconds(fadeEndTick - playheadTick, tempo));
  }
}

/** Linear crossfade FADE-IN envelope on `node` for a source overlapping an EARLIER neighbour over
 *  [segStartTick, fadeEndTick]. `startDelay` is the source's scheduled start offset (a not-yet-started
 *  source ramps from its real start). Shared by the original-audio AND sub-lane branches. */
function applyFadeIn(
  node: GainNode, segStartTick: number, fadeEndTick: number, playheadTick: number, tempo: number, now: number, startDelay: number,
) {
  const fadeLenTicks = fadeEndTick - segStartTick;
  if (fadeLenTicks <= 0 || playheadTick >= fadeEndTick) return;
  if (playheadTick > segStartTick) {
    const progress = (playheadTick - segStartTick) / fadeLenTicks;
    node.gain.setValueAtTime(progress, now);
    node.gain.linearRampToValueAtTime(1, now + ticksToSeconds(fadeEndTick - playheadTick, tempo));
  } else {
    node.gain.setValueAtTime(0, now + startDelay);
    node.gain.linearRampToValueAtTime(1, now + ticksToSeconds(fadeEndTick - playheadTick, tempo));
  }
}
