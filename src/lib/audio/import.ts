import * as playback from "./playback";
import type { Track, Segment } from "../../types/project";
import type { AudioTrackData } from "../../store/audio";

export async function importAudioToTrack(
  filePath: string,
  tempo: number,
  startTick: number,
  existingTracks: Track[],
  addTrack: (track: Track) => void,
  loadAudioFile: (filePath: string) => Promise<AudioTrackData>,
  updateTrack: (id: string, updates: Partial<Track>) => void,
) {
  const fileName = filePath.split(/[/\\]/).pop() ?? "audio";

  const audioData = await loadAudioFile(filePath);
  const durationTicks = Math.round(playback.durationMsToTicks(audioData.durationMs, tempo));

  const seg: Segment = {
    id: `seg-${Date.now()}`,
    startTick,
    durationTicks,
    content: {
      type: "audioClip",
      sourcePath: filePath,
      offsetMs: 0,
      totalDurationMs: audioData.durationMs,
    },
  };

  const targetTrack = findTrackWithSpace(existingTracks, startTick, durationTicks);
  if (targetTrack) {
    updateTrack(targetTrack.id, {
      segments: [...targetTrack.segments, seg],
    });
  } else {
    addTrack({
      id: `track-${Date.now()}`,
      name: fileName,
      trackType: "audio",
      segments: [seg],
      volumeDb: 0,
      pan: 0,
      muted: false,
      solo: false,
    });
  }

  playback.loadAudioBuffer(filePath);
}

function findTrackWithSpace(
  tracks: Track[],
  startTick: number,
  durationTicks: number,
): Track | null {
  const endTick = startTick + durationTicks;
  for (const track of tracks) {
    if (track.trackType !== "audio") continue;
    const hasOverlap = track.segments.some((seg) => {
      const segEnd = seg.startTick + seg.durationTicks;
      return startTick < segEnd && endTick > seg.startTick;
    });
    if (!hasOverlap) return track;
  }
  return null;
}
