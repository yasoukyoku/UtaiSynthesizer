import { readFile } from "@tauri-apps/plugin-fs";
import { TICKS_PER_BEAT } from "../constants";
import type { Track } from "../../types/project";
import type { AudioTrackData } from "../../store/audio";

let audioCtx: AudioContext | null = null;
const loadedBuffers = new Map<string, AudioBuffer>();

interface ScheduledSource {
  source: AudioBufferSourceNode;
  gain: GainNode;
  panner: StereoPannerNode;
}

let scheduledSources: ScheduledSource[] = [];

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

  const ctx = getContext();
  const now = ctx.currentTime;
  const hasSolo = tracks.some((t) => t.solo);
  let totalScheduled = 0;
  let endedCount = 0;

  for (const track of tracks) {
    if (track.muted) continue;
    if (hasSolo && !track.solo) continue;

    const vol = dbToLinear(track.volumeDb);
    const sorted = [...track.segments]
      .filter((s) => s.content.type === "audioClip")
      .sort((a, b) => a.startTick - b.startTick);

    for (let si = 0; si < sorted.length; si++) {
      const seg = sorted[si]!;
      if (seg.content.type !== "audioClip") continue;

      const segEnd = seg.startTick + seg.durationTicks;
      if (segEnd <= playheadTick) continue;

      const filePath = seg.content.sourcePath;
      if (!audioFiles[filePath]) continue;

      let buf = loadedBuffers.get(filePath);
      if (!buf) {
        try { buf = await loadAudioBuffer(filePath); } catch { continue; }
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

      const gain = ctx.createGain();
      gain.gain.setValueAtTime(vol, now + startDelay);

      const panner = ctx.createStereoPanner();
      panner.pan.value = track.pan;

      // Crossfade: fade out if overlapping with next segment
      if (si + 1 < sorted.length) {
        const next = sorted[si + 1]!;
        if (next.startTick < segEnd) {
          const fadeStart = Math.max(next.startTick, playheadTick);
          const fadeEnd = Math.min(segEnd, next.startTick + next.durationTicks);
          const fadeStartTime = now + ticksToSeconds(Math.max(0, fadeStart - playheadTick), tempo);
          const fadeEndTime = now + ticksToSeconds(Math.max(0, fadeEnd - playheadTick), tempo);
          if (fadeEndTime > fadeStartTime) {
            gain.gain.setValueAtTime(vol, fadeStartTime);
            gain.gain.linearRampToValueAtTime(0, fadeEndTime);
          }
        }
      }

      // Crossfade: fade in if overlapping with previous segment
      if (si > 0) {
        const prev = sorted[si - 1]!;
        const prevEnd = prev.startTick + prev.durationTicks;
        if (seg.startTick < prevEnd) {
          const fadeStart = Math.max(seg.startTick, playheadTick);
          const fadeEnd = Math.min(prevEnd, segEnd);
          const fadeStartTime = now + ticksToSeconds(Math.max(0, fadeStart - playheadTick), tempo);
          const fadeEndTime = now + ticksToSeconds(Math.max(0, fadeEnd - playheadTick), tempo);
          if (fadeEndTime > fadeStartTime) {
            gain.gain.setValueAtTime(0, fadeStartTime);
            gain.gain.linearRampToValueAtTime(vol, fadeEndTime);
          }
        }
      }

      const source = ctx.createBufferSource();
      source.buffer = buf;
      source.connect(gain).connect(panner).connect(ctx.destination);

      source.onended = () => {
        endedCount++;
        if (endedCount >= totalScheduled) onAllEnded();
      };

      source.start(now + startDelay, audioOffset, playDuration);
      scheduledSources.push({ source, gain, panner });
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

export function updateTrackVolume(_trackId: string, _volumeDb: number) {}
export function updateTrackPan(_trackId: string, _pan: number) {}

export function getContextTime(): number {
  return audioCtx?.currentTime ?? 0;
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
