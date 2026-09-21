/**
 * Resolve the BEST playable source audio + display name for a track / segment
 * so the 音乐转MIDI dialog ALWAYS has something to show (track name → waveform
 * → play button) even when the specific segment has no direct `sourcePath`.
 *
 * Priority order per segment:
 *   1. audioClip  content.sourcePath        (raw dropped file)
 *   2. notes  content + processedOutputs    (cloned / separated / rendered WAVs)
 *   3. any content.audioPath/filePath/path
 *   4. segment-level audioPath/sourcePath
 *
 * Returns `{ path, name }`; both optional. Zero imports → safe to use from the
 * track list OR the timeline right-click without touching Zustand.
 */

function getAudioPath(obj: any): string | undefined {
  if (!obj) return undefined;
  return obj.audioPath || obj.sourcePath || obj.filePath || obj.path || undefined;
}

export interface AudioSourceCandidate {
  path?: string;
  name?: string;
}

/** Resolve the source audio for ONE segment (right-click 转 MIDI target). */
export function resolveSegmentSourceAudio(seg: any): AudioSourceCandidate {
  if (!seg) return {};
  let path: string | undefined;
  let name: string | undefined;

  if (seg.content) {
    if (seg.content.type === "audioClip" && seg.content.sourcePath) {
      path = seg.content.sourcePath;
    } else if (seg.content.type === "notes") {
      // notes: prefer a render/clone/sep result WAV before the (absent) source.
      const outs = seg.processedOutputs || [];
      for (const o of outs) {
        path = getAudioPath(o);
        if (path) break;
      }
      if (!path) path = getAudioPath(seg.content);
    } else {
      path = getAudioPath(seg.content);
    }
    name = seg.content.name || seg.content.label;
  }
  if (!path) path = getAudioPath(seg);

  return { path, name };
}

/** Resolve the source audio for a WHOLE track (track-list right-click 转 MIDI).
 *  §user "任何转换MIDI默认整首歌": the track-level conversion must cover the WHOLE
 *  SONG, not the first segment it finds. An audioClip's `sourcePath` IS the full
 *  original file (split clips still reference it) → highest priority. A notes
 *  track's processedOutputs are per-SEGMENT renders — picking the first one would
 *  convert only a small piece, so among those we pick the LONGEST output (the
 *  best "whole song" proxy available without decoding). */
export function resolveTrackSourceAudio(track: any): AudioSourceCandidate {
  if (!track) return {};
  const segs = track.segments || [];
  // 1. audioClip → the raw source file behind the clip (full song even after splits).
  for (const seg of segs) {
    if (seg?.content?.type === "audioClip" && seg.content.sourcePath) {
      return { path: seg.content.sourcePath, name: seg.content.name || seg.content.label || track.name };
    }
  }
  // 2. notes / vocal segments → the longest processed output across ALL segments.
  let best: { path: string; name?: string; durMs: number } | null = null;
  for (const seg of segs) {
    const outs: any[] = seg?.processedOutputs || [];
    for (const o of outs) {
      const p = getAudioPath(o);
      if (!p) continue;
      const durMs = Number(o?.totalDurationMs) || 0;
      if (!best || durMs > best.durMs) best = { path: p, name: o?.laneLabel || o?.group, durMs };
    }
  }
  if (best) return { path: best.path, name: best.name || track.name };
  // 3. any content-level audio path (e.g. vocal renders), longest-duration first.
  for (const seg of segs) {
    const r = resolveSegmentSourceAudio(seg);
    if (r.path) return { path: r.path, name: r.name || track.name };
  }
  const path = getAudioPath(track);
  return { path, name: track.name };
}

/**
 * The sidecar's playback synthesis keys per-instrument WAVs by GM identity:
 * "drums" for channel-10 percussion, otherwise "gm:NNN" (zero-padded program).
 * The MERGED MIDI's human track names ("Acoustic Piano") never match those keys
 * directly — compute the key from the track's program/channel instead.
 */
export function amtInstrumentKey(program?: number | null, channel?: number | null): string | undefined {
  if (channel === 9) return "drums";
  if (program === null || program === undefined) return undefined;
  return `gm:${String(program).padStart(3, "0")}`;
}

/**
 * Find the per-instrument WAV for an imported MIDI track. Tries, in order:
 *   1. the computed GM key ("drums" / "gm:NNN") from program+channel
 *   2. the raw track name ("Acoustic Piano" ↔ instrument-labeled maps)
 *   3. case-insensitive / drum-name fuzzy match
 */
export function matchInstrumentWav(
  wavs: Record<string, string> | undefined,
  trackName: string,
  program?: number | null,
  channel?: number | null,
): string | undefined {
  if (!wavs) return undefined;
  const key = amtInstrumentKey(program, channel);
  if (key && wavs[key]) return wavs[key];
  if (trackName && wavs[trackName]) return wavs[trackName];
  const lower = (trackName || "").toLowerCase();
  if (lower) {
    const direct = Object.entries(wavs).find(([k]) => k.toLowerCase() === lower);
    if (direct) return direct[1];
    if (lower.includes("drum") && wavs["drums"]) return wavs["drums"];
  }
  return undefined;
}