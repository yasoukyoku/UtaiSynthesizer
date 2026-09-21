import type { Track } from "../types/project";
import { getRandomTrackColor } from "./trackColors";

/** A blank track of `trackType` — THE one source for the Track literal built by TrackList's add-track
 *  buttons AND the score importer (src/lib/vocal/import.ts). Keeping it here means a new Track field
 *  can't be added to one construction site and forgotten in the other (the drift the NO-duplication
 *  rule guards against). Caller supplies the id + display name. */
export function blankTrack(id: string, name: string, trackType: Track["trackType"]): Track {
  return {
    id,
    name,
    trackType,
    color: getRandomTrackColor(),
    segments: [],
    volumeDb: 0,
    pan: 0,
    muted: false,
    solo: false,
    expanded: false,
    laneControls: {},
  };
}
