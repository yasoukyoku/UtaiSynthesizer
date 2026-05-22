import type { Track } from "../types/project";
import { TRACK_HEADER_HEIGHT, LANE_HEIGHT } from "./constants";

export function getMaxLaneCount(track: Track): number {
  let max = 0;
  for (const seg of track.segments) {
    const n = seg.processedOutputs?.length ?? 0;
    if (n > max) max = n;
  }
  return max;
}

export function getLaneLabels(track: Track): string[] {
  const labels = new Set<string>();
  for (const seg of track.segments) {
    if (seg.processedOutputs) {
      for (const out of seg.processedOutputs) {
        labels.add(out.laneLabel);
      }
    }
  }
  return Array.from(labels);
}

export function computeTrackHeight(track: Track): number {
  if (!track.expanded) return TRACK_HEADER_HEIGHT;
  const lanes = getMaxLaneCount(track);
  return lanes > 0 ? TRACK_HEADER_HEIGHT + lanes * LANE_HEIGHT : TRACK_HEADER_HEIGHT;
}

export function computeTrackYOffsets(tracks: Track[]): number[] {
  const offsets: number[] = [];
  let y = 0;
  for (const track of tracks) {
    offsets.push(y);
    y += computeTrackHeight(track);
  }
  return offsets;
}

export function computeTotalTracksHeight(tracks: Track[]): number {
  let h = 0;
  for (const track of tracks) {
    h += computeTrackHeight(track);
  }
  return h;
}

export function findTrackAtY(offsets: number[], y: number): number {
  for (let i = offsets.length - 1; i >= 0; i--) {
    if (y >= offsets[i]!) return i;
  }
  return -1;
}
