// ② Vocal render (S48 Phase 6, §11 / §10.1). Turn a segment's EDITED notes into singing:
//   1. build the score triples (per-note lyric + RAW MIDI + 50fps frames) covering [0, lastNoteEnd],
//      with a leading rest so the stem starts at the segment start and explicit gap rests (§3.4 — a rest
//      is NEVER inferred from pitch==0), and the whole-segment Option-A f0 (evalF0CentsFrames);
//   2. invoke the Rust `render_vocal_segment` (ScoreToCV → SVC net_g, +transpose Rust-side);
//   3. deposit the baked wav as a processedOutputs OVERLAY — non-undoable (sig-invisible) and it rides
//      the SAME lane machinery as an audio track's sub-lanes, so it plays back / persists / mutes for free.
import { invoke } from "@tauri-apps/api/core";
import { evalF0CentsFrames } from "../f0eval";
import { msToTicks } from "../audio/laneOps";
import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import type { Note, PitchCurve, NoteTransition, ProcessedOutput } from "../../types/project";

/** Thrown by the global single-flight backstop — the caller shows a "busy" message instead of "failed". */
export const VOCAL_RENDER_BUSY = "VOCAL_RENDER_BUSY";

/** ScoreToCV native frame rate — the triple `frames` and the f0 array share this one grid so they align. */
const RENDER_FPS = 50;
/** Stable lane identity for the single baked vocal stem (mirrors an Output-node id — one lane per segment). */
export const VOCAL_LANE_ID = "vocal";

export interface ScoreTriple {
  lyric: string;
  note_num: number;
  frames: number;
}

/** Wire options mirroring Rust `VocalRenderOptions` (snake_case — Tauri passes them through verbatim). */
export interface VocalRenderOptions {
  backend: "sovits" | "rvc";
  cv_speaker_id: number;
  lang_id: number;
  transpose: number;
  noise_scale: number;
  seed: number;
  gpu_extract: boolean;
}

/**
 * Build the render input from a segment's notes. Score triples cover [0, lastNoteEnd] contiguously: a
 * leading rest (so stem-ms 0 == the segment start → the lane plays 1:1 aligned), each note (RAW MIDI —
 * transpose is Rust-side, §9.3), and explicit gap rests between notes. The whole-segment 50fps Option-A f0
 * (WRITTEN-pitch cents + voiced mask) is sampled on the SAME `ticksPerFrame` grid, so Σ(triple frames) ==
 * the f0 frame count and `build_note_hz`'s cv↔DAW map is exact. Pure.
 */
export function buildVocalScore(
  notes: readonly Note[],
  pitchDev: PitchCurve | undefined,
  tempo: number,
  defaultTransition: Required<NoteTransition>,
): { triples: ScoreTriple[]; f0Cents: number[]; f0Voiced: number[] } {
  const ticksPerFrame = msToTicks(1000 / RENDER_FPS, tempo); // 20 ms per 50fps frame
  const frameOf = (relTick: number) => Math.round(relTick / ticksPerFrame);
  const sorted = [...notes].sort((a, b) => a.tick - b.tick || (a.id < b.id ? -1 : 1));

  const triples: ScoreTriple[] = [];
  let cursor = 0; // segment-relative tick covered so far
  for (const n of sorted) {
    const start = Math.max(cursor, n.tick);
    const end = n.tick + n.duration;
    if (end <= cursor) continue; // fully swallowed by a previous note (defensive — notes don't overlap)
    if (start > cursor) {
      const restFrames = frameOf(start) - frameOf(cursor);
      if (restFrames > 0) triples.push({ lyric: "R", note_num: 0, frames: restFrames });
    }
    const noteFrames = frameOf(end) - frameOf(start);
    if (noteFrames > 0) triples.push({ lyric: n.lyric, note_num: n.pitch, frames: noteFrames });
    cursor = end;
  }

  const frameCount = frameOf(cursor); // cursor == last note end == Σ(triple frames)
  const { cents, voiced } = evalF0CentsFrames(
    sorted,
    pitchDev,
    { frameStartTick: 0, ticksPerFrame, frameCount },
    { tempo, defaultTransition },
  );
  return { triples, f0Cents: Array.from(cents), f0Voiced: Array.from(voiced) };
}

interface AudioFileInfo {
  duration_ms: number;
  peaks: number[];
}
let vocalRunSeq = 0;

/**
 * Render a vocal segment and deposit the baked stem as a processedOutputs overlay.
 *  - GLOBAL single-flight: only one vocal render at a time (the shared ORT engine +
 *    release_gpu_sessions_except would make concurrent renders evict each other mid-inference). Throws
 *    VOCAL_RENDER_BUSY as a backstop (the button also gates on useAppStore.vocalRenderActive).
 *  - Never destroys a good bake before its replacement: a RE-render keeps the old lane playing until the
 *    new one lands; a FIRST render shows a loading placeholder for feedback. On error/cancel the PRIOR
 *    state is restored (re-render) or the spinner cleared (first render) — never a lost/broken render.
 *  - Deposits only if the segment still exists (a delete/project-load mid-render drops it). Throws on
 *    failure so the caller can toast; the deposit is sig-invisible (non-undoable overlay).
 */
export async function renderVocalSegment(req: {
  trackId: string;
  segmentId: string;
  laneLabel: string;
  voiceName: string;
  modelPath: string;
  triples: ScoreTriple[];
  f0Cents: number[];
  f0Voiced: number[];
  options: VocalRenderOptions;
}): Promise<void> {
  const { trackId, segmentId, laneLabel } = req;
  if (useAppStore.getState().vocalRenderActive) throw new Error(VOCAL_RENDER_BUSY);
  useAppStore.getState().setVocalRenderActive(true);

  const seg = () =>
    useProjectStore.getState().tracks.find((t) => t.id === trackId)?.segments.find((s) => s.id === segmentId);
  const prevOutputs = seg()?.processedOutputs; // the current bake (if any) — kept while rendering, restored on failure
  const deposit = (outs: ProcessedOutput[] | undefined) =>
    useProjectStore.getState().replaceProcessedOutputs(trackId, segmentId, outs ?? []);

  // FIRST render (no bake yet) → loading placeholder for feedback; RE-render → keep the old bake playing
  // until the new stem lands (mirrors the audio path — never wipe a good render before its replacement).
  if (!prevOutputs || prevOutputs.length === 0) {
    deposit([{ laneId: VOCAL_LANE_ID, laneLabel, group: laneLabel, audioPath: "", totalDurationMs: 0, waveformPeaks: [], outputNodeId: VOCAL_LANE_ID, loading: true }]);
  }
  try {
    const raw = await invoke<string>("ensure_cache_dir", {
      segmentId: `${segmentId}/v${Date.now().toString(36)}${(vocalRunSeq++).toString(36)}`,
    });
    const outputPath = `${raw.replace(/\\/g, "/")}/vocal.wav`;
    const result = await invoke<{ audio: number[]; sample_rate: number }>("render_vocal_segment", {
      voiceName: req.voiceName,
      modelPath: req.modelPath,
      nodeId: segmentId,
      score: req.triples,
      f0Cents: req.f0Cents,
      f0Voiced: req.f0Voiced,
      options: req.options,
    });
    await invoke("save_temp_audio", { samples: result.audio, sampleRate: result.sample_rate, outputPath });
    const info = await invoke<AudioFileInfo>("load_audio_file", { path: outputPath });
    if (seg()) {
      deposit([{ laneId: VOCAL_LANE_ID, laneLabel, group: laneLabel, audioPath: outputPath, totalDurationMs: info.duration_ms, waveformPeaks: info.peaks, outputNodeId: VOCAL_LANE_ID }]);
    }
  } catch (e) {
    if (seg()) deposit(prevOutputs); // restore the prior bake (re-render) or clear the first-render spinner
    throw e;
  } finally {
    useAppStore.getState().setVocalRenderActive(false);
  }
}
