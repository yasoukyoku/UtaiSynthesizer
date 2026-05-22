import { readFile } from "@tauri-apps/plugin-fs";
import { TICKS_PER_BEAT } from "../constants";
import type { Track } from "../../types/project";
import type { AudioTrackData } from "../../store/audio";

let audioCtx: AudioContext | null = null;
const loadedBuffers = new Map<string, AudioBuffer>();

interface ScheduledSource {
  trackId: string;
  laneLabel?: string;
  source: AudioBufferSourceNode;
  trackGain: GainNode;
  laneGain?: GainNode;
  fadeIn?: GainNode;
  fadeOut?: GainNode;
  panner: StereoPannerNode;
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
): Promise<boolean> {
  stopPlayback();
  const gen = ++playGeneration;

  const ctx = getContext();
  const now = ctx.currentTime;
  scheduleTimeOrigin = now;
  const hasSolo = tracks.some((t) => t.solo);
  let totalScheduled = 0;
  let endedCount = 0;

  for (const track of tracks) {
    const trackAudible = !track.muted && (!hasSolo || track.solo);
    const trackVol = trackAudible ? dbToLinear(track.volumeDb) : 0;

    const sorted = [...track.segments]
      .filter((s) => s.content.type === "audioClip")
      .sort((a, b) => a.startTick - b.startTick);

    for (let si = 0; si < sorted.length; si++) {
      const seg = sorted[si]!;
      if (seg.content.type !== "audioClip") continue;

      const segEnd = seg.startTick + seg.durationTicks;
      if (segEnd <= playheadTick) continue;

      const hasLaneOutputs = seg.processedOutputs && seg.processedOutputs.length > 0 && track.expanded;

      if (hasLaneOutputs) {
        for (const out of seg.processedOutputs!) {
          const laneCtrl = track.laneControls[out.laneLabel];
          const laneMuted = laneCtrl?.muted ?? false;
          const laneVol = laneMuted ? 0 : dbToLinear(laneCtrl?.volumeDb ?? 0);

          let buf = loadedBuffers.get(out.audioPath);
          if (!buf) {
            try { buf = await loadAudioBuffer(out.audioPath); } catch (e) {
              console.error(`Failed to load lane audio: ${out.audioPath}`, e);
              continue;
            }
            if (gen !== playGeneration) return false;
          }
          if (!buf) continue;

          let startDelay: number;
          let audioOffset: number;
          let playDuration: number;

          if (playheadTick >= seg.startTick) {
            const secsInto = ticksToSeconds(playheadTick - seg.startTick, tempo);
            audioOffset = secsInto;
            startDelay = 0;
            playDuration = ticksToSeconds(seg.durationTicks, tempo) - secsInto;
          } else {
            audioOffset = 0;
            startDelay = ticksToSeconds(seg.startTick - playheadTick, tempo);
            playDuration = ticksToSeconds(seg.durationTicks, tempo);
          }

          audioOffset = Math.max(0, Math.min(audioOffset, buf.duration));
          playDuration = Math.min(playDuration, buf.duration - audioOffset);
          if (playDuration <= 0) continue;

          const laneGainNode = ctx.createGain();
          laneGainNode.gain.setValueAtTime(laneVol, now);
          const trackGainNode = ctx.createGain();
          trackGainNode.gain.setValueAtTime(trackVol, now);
          const panner = ctx.createStereoPanner();
          panner.pan.value = laneCtrl?.pan ?? track.pan;

          const source = ctx.createBufferSource();
          source.buffer = buf;
          source.connect(laneGainNode).connect(trackGainNode).connect(panner).connect(ctx.destination);
          source.onended = () => { endedCount++; if (endedCount >= totalScheduled) onAllEnded(); };
          source.start(now + startDelay, audioOffset, playDuration);
          scheduledSources.push({
            trackId: track.id, laneLabel: out.laneLabel, source,
            trackGain: trackGainNode, laneGain: laneGainNode, panner,
            silenced: !trackAudible, laneSilenced: laneMuted,
          });
          totalScheduled++;
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
          continue;
        }
        if (gen !== playGeneration) return false;
      }
      if (!buf) continue;

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

      const trackGainNode = ctx.createGain();
      trackGainNode.gain.setValueAtTime(trackVol, now);
      const panner = ctx.createStereoPanner();
      panner.pan.value = track.pan;

      // Separate fade nodes: crossfade automation isolated from track volume
      const fadeInNode = ctx.createGain();
      const fadeOutNode = ctx.createGain();

      // Crossfade fade-out: overlap with next segment
      if (si + 1 < sorted.length) {
        const next = sorted[si + 1]!;
        if (next.startTick < segEnd) {
          const fadeStartTick = next.startTick;
          const fadeEndTick = Math.min(segEnd, next.startTick + next.durationTicks);
          const fadeLenTicks = fadeEndTick - fadeStartTick;
          if (fadeLenTicks > 0) {
            if (playheadTick >= fadeEndTick) {
              fadeOutNode.gain.setValueAtTime(0, now);
            } else if (playheadTick > fadeStartTick) {
              const progress = (playheadTick - fadeStartTick) / fadeLenTicks;
              fadeOutNode.gain.setValueAtTime(1 - progress, now);
              fadeOutNode.gain.linearRampToValueAtTime(0, now + ticksToSeconds(fadeEndTick - playheadTick, tempo));
            } else {
              const fadeStartTime = now + ticksToSeconds(fadeStartTick - playheadTick, tempo);
              const fadeEndTime = now + ticksToSeconds(fadeEndTick - playheadTick, tempo);
              fadeOutNode.gain.setValueAtTime(1, fadeStartTime);
              fadeOutNode.gain.linearRampToValueAtTime(0, fadeEndTime);
            }
          }
        }
      }

      // Crossfade fade-in: overlap with previous segment
      if (si > 0) {
        const prev = sorted[si - 1]!;
        const prevEnd = prev.startTick + prev.durationTicks;
        if (seg.startTick < prevEnd) {
          const fadeEndTick = Math.min(prevEnd, segEnd);
          const fadeLenTicks = fadeEndTick - seg.startTick;
          if (fadeLenTicks > 0 && playheadTick < fadeEndTick) {
            if (playheadTick > seg.startTick) {
              const progress = (playheadTick - seg.startTick) / fadeLenTicks;
              fadeInNode.gain.setValueAtTime(progress, now);
              fadeInNode.gain.linearRampToValueAtTime(1, now + ticksToSeconds(fadeEndTick - playheadTick, tempo));
            } else {
              const fadeStartTime = now + startDelay;
              const fadeEndTime = now + ticksToSeconds(fadeEndTick - playheadTick, tempo);
              fadeInNode.gain.setValueAtTime(0, fadeStartTime);
              fadeInNode.gain.linearRampToValueAtTime(1, fadeEndTime);
            }
          }
        }
      }

      const source = ctx.createBufferSource();
      source.buffer = buf;
      source.connect(fadeInNode).connect(fadeOutNode).connect(trackGainNode).connect(panner).connect(ctx.destination);

      source.onended = () => {
        endedCount++;
        if (endedCount >= totalScheduled) onAllEnded();
      };

      source.start(now + startDelay, audioOffset, playDuration);
      scheduledSources.push({
        trackId: track.id, source, trackGain: trackGainNode,
        fadeIn: fadeInNode, fadeOut: fadeOutNode, panner,
        silenced: !trackAudible, laneSilenced: false,
      });
      totalScheduled++;
    }
  }

  return totalScheduled > 0;
}

export function stopPlayback() {
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
    if (s.trackId === trackId && !s.laneLabel) {
      s.panner.pan.setValueAtTime(pan, s.panner.context.currentTime);
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

export function updateLaneVolume(trackId: string, laneLabel: string, volumeDb: number) {
  const vol = dbToLinear(volumeDb);
  for (const s of scheduledSources) {
    if (s.trackId === trackId && s.laneLabel === laneLabel && s.laneGain && !s.laneSilenced) {
      s.laneGain.gain.setValueAtTime(vol, s.laneGain.context.currentTime);
    }
  }
}

export function updateLanePan(trackId: string, laneLabel: string, pan: number) {
  for (const s of scheduledSources) {
    if (s.trackId === trackId && s.laneLabel === laneLabel) {
      s.panner.pan.setValueAtTime(pan, s.panner.context.currentTime);
    }
  }
}

export function updateLaneMute(trackId: string, laneLabel: string, muted: boolean, volumeDb: number) {
  for (const s of scheduledSources) {
    if (s.trackId === trackId && s.laneLabel === laneLabel && s.laneGain) {
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

export function ticksToSeconds(ticks: number, tempo: number): number {
  return (ticks / TICKS_PER_BEAT) * (60.0 / tempo);
}

export function secondsToTicks(secs: number, tempo: number): number {
  return ((secs * tempo) / 60.0) * TICKS_PER_BEAT;
}

export function durationMsToTicks(ms: number, tempo: number): number {
  return secondsToTicks(ms / 1000.0, tempo);
}

function dbToLinear(db: number): number {
  return Math.pow(10, db / 20);
}
