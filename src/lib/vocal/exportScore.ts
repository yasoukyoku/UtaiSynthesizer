// S63 — Score export (ust / ustx / midi): the TS side flattens each chosen vocal track's notes onto the
// ABSOLUTE timeline (seg.startTick + note.tick — the exact inverse of import.ts's rebase) and hands the
// plain data to Rust (commands/export_score.rs), which owns the format serialization — mirroring the
// import split ("Rust owns parsing, TS only maps").
//
// Scope mirrors the IMPORT scope deliberately (S56 symmetry): notes (tick/duration/pitch/lyric) + BPM +
// time signature + (ustx only) vibrato. Pitch is the SCORE pitch — vocalParams.transpose is a RENDER
// knob and must not bake into the exported notes. Notes outside the segment box are exported too: they
// are user data the render also consumes (buildScoreTriples takes the full array).
import { invoke } from "@tauri-apps/api/core";
import type { Track, VibratoSpec } from "../../types/project";
import { useProjectStore } from "../../store/project";

export type ScoreFormat = "ust" | "ustx" | "midi";

export interface ScoreTrackChoice {
  trackId: string;
  name: string;
  noteCount: number;
}

interface ExportNotePayload {
  tick: number;
  duration: number;
  pitch: number;
  lyric: string;
  velocity: number;
  vibrato?: VibratoSpec;
}

/** Vocal tracks that have at least one note — the dialog's checkbox list AND the menu-item enablement
 *  (one predicate, so the menu can never offer an export the dialog would show empty). */
export function scoreExportableTracks(tracks: Track[]): ScoreTrackChoice[] {
  const out: ScoreTrackChoice[] = [];
  for (const t of tracks) {
    if (t.trackType !== "vocal") continue;
    let n = 0;
    for (const s of t.segments) if (s.content.type === "notes") n += s.content.notes.length;
    if (n > 0) out.push({ trackId: t.id, name: t.name, noteCount: n });
  }
  return out;
}

/** Flatten one track's notes to absolute ticks, sorted. (Overlap truncation across segment boundaries
 *  is Rust's defensive job — the editor invariant keeps notes disjoint WITHIN a segment only.) */
function flattenTrack(track: Track): ExportNotePayload[] {
  const notes: ExportNotePayload[] = [];
  for (const seg of track.segments) {
    if (seg.content.type !== "notes") continue;
    for (const n of seg.content.notes) {
      notes.push({
        tick: seg.startTick + n.tick,
        duration: n.duration,
        pitch: n.pitch,
        lyric: n.lyric,
        velocity: n.velocity,
        ...(n.vibrato ? { vibrato: n.vibrato } : {}),
      });
    }
  }
  notes.sort((a, b) => a.tick - b.tick);
  return notes;
}

/** Invoke the Rust writer. Throws the backend's stable EXPORT_SCORE_* codes on failure. */
export async function runScoreExport(format: ScoreFormat, outPath: string, trackIds: string[]): Promise<void> {
  const st = useProjectStore.getState();
  const chosen = st.tracks
    .filter((t) => trackIds.includes(t.id))
    .map((t) => ({ name: t.name, notes: flattenTrack(t) }))
    .filter((t) => t.notes.length > 0);
  await invoke("export_score_files", {
    format,
    path: outPath,
    tempo: st.tempo,
    timeSig: st.timeSignature,
    tracks: chosen,
  });
}
