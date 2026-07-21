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
  /** S73: detune(cents)→ ustx tuning(import 之逆;ust/midi 忽略)。 */
  detune?: number;
}

/** S73: 轨内各 notes 段 pitchDev 展平到绝对 tick 的合并折线(→ ustx part 级 pitd)。 */
interface ExportCurvePayload {
  xs: number[];
  ys: number[];
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
        ...(n.detune ? { detune: n.detune } : {}),
      });
    }
  }
  notes.sort((a, b) => a.tick - b.tick);
  return notes;
}

/** S73: 各段 pitchDev 展平成一条绝对 tick 折线(段间不重叠,按段起点排序拼接);无点 → undefined。
 *  这是「烤入导入→再导出」不丢调教的闭环——手绘 pitchDev 一并受益(ustx pitd 通道)。 */
function flattenPitchDev(track: Track): ExportCurvePayload | undefined {
  const xs: number[] = [];
  const ys: number[] = [];
  const segs = [...track.segments]
    .filter((s) => s.content.type === "notes")
    .sort((a, b) => a.startTick - b.startTick);
  for (const seg of segs) {
    if (seg.content.type !== "notes") continue;
    const c = seg.content.pitchDev;
    if (!c || c.xs.length === 0) continue;
    for (let i = 0; i < c.xs.length; i++) {
      const x = seg.startTick + (c.xs[i] ?? 0);
      // 段间防御:绝不让 xs 逆序(段重叠时丢弃倒退点;编辑器不变量下不触发)
      if (xs.length > 0 && x <= (xs[xs.length - 1] ?? -Infinity)) continue;
      xs.push(x);
      ys.push(c.ys[i] ?? 0);
    }
  }
  return xs.length > 0 ? { xs, ys } : undefined;
}

/** Invoke the Rust writer. Throws the backend's stable EXPORT_SCORE_* codes on failure. */
export async function runScoreExport(format: ScoreFormat, outPath: string, trackIds: string[]): Promise<void> {
  const st = useProjectStore.getState();
  const chosen = st.tracks
    .filter((t) => trackIds.includes(t.id))
    .map((t) => {
      const pitchDev = flattenPitchDev(t);
      // 嵌套结构不经 Tauri 参数名转换 → 键名必须精确匹配 Rust serde(snake_case)
      return { name: t.name, notes: flattenTrack(t), ...(pitchDev ? { pitch_dev: pitchDev } : {}) };
    })
    .filter((t) => t.notes.length > 0);
  await invoke("export_score_files", {
    format,
    path: outPath,
    tempo: st.tempo,
    timeSig: st.timeSignature,
    tracks: chosen,
  });
}
