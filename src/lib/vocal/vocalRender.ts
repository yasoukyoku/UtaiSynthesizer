// ② Vocal render (S48 Phase 6, §11 / §10.1). Turn a segment's EDITED notes into singing:
//   1. build the score triples (per-note lyric + RAW MIDI + 50fps frames) covering [0, lastNoteEnd],
//      with a leading rest so the stem starts at the segment start and explicit gap rests (§3.4 — a rest
//      is NEVER inferred from pitch==0), and the whole-segment Option-A f0 (evalF0CentsFrames);
//   2. invoke the Rust `render_vocal_segment` (ScoreToCV → SVC net_g, +transpose Rust-side);
//   3. deposit the baked wav as a processedOutputs OVERLAY — non-undoable (sig-invisible) and it rides
//      the SAME lane machinery as an audio track's sub-lanes, so it plays back / persists / mutes for free.
import { invoke } from "@tauri-apps/api/core";
import { RVC_DEFAULTS, SOVITS_DEFAULTS, type RvcOptions, type SovitsOptions } from "../workflow/voiceDefaults";
import { evalF0CentsFrames } from "../f0eval";
import { isBreathLyric } from "../vocalNotes";
import { msToTicks } from "../audio/laneOps";
import { useProjectStore, DEFAULT_VOCAL_PARAMS } from "../../store/project";
import { useAppStore } from "../../store/app";
import { useVoiceModelStore } from "../../store/voice-models";
import { contentSig, vocalParamsSig } from "../../store/history";
import type { Note, PitchCurve, NoteTransition, ProcessedOutput, Track, Segment } from "../../types/project";

/** Thrown by the global single-flight backstop — the caller shows a "busy" message instead of "failed". */
export const VOCAL_RENDER_BUSY = "VOCAL_RENDER_BUSY";
/** Thrown by renderVocalPart when the track's singer can't be resolved / the segment has no renderable notes.
 *  The manual-render caller maps these to their own toasts; the auto-render batch pre-filters both away. */
export const VOCAL_NO_VOICE = "VOCAL_NO_VOICE";
export const VOCAL_EMPTY = "VOCAL_EMPTY";

/** ScoreToCV native frame rate — the triple `frames` and the f0 array share this one grid so they align. */
const RENDER_FPS = 50;
/** Stable lane identity for the single baked vocal stem (mirrors an Output-node id — one lane per segment). */
export const VOCAL_LANE_ID = "vocal";

export interface ScoreTriple {
  lyric: string;
  note_num: number;
  frames: number;
}

/** Wire options mirroring Rust `VocalRenderOptions` (snake_case — Tauri passes them through verbatim).
 *  Item-1: the backend-specific quality knobs REUSE the SoVITS/RVC contracts (`sovits`/`rvc`), the FULL
 *  object each (defaults + the user's overrides). The command force-neutralizes auto_f0/f0_shift/loudness/
 *  only_diffusion/rms_mix (they'd break the ② render). */
export interface VocalRenderOptions {
  backend: "sovits" | "rvc";
  cv_speaker_id: number;
  lang_id: number;
  transpose: number;
  sovits: SovitsOptions;
  rvc: RvcOptions;
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
  breathToken: string,
): { triples: ScoreTriple[]; f0Cents: number[]; f0Voiced: number[] } {
  const ticksPerFrame = msToTicks(1000 / RENDER_FPS, tempo); // 20 ms per 50fps frame
  const frameOf = (relTick: number) => Math.round(relTick / ticksPerFrame);
  const sorted = [...notes].sort((a, b) => a.tick - b.tick || (a.id < b.id ? -1 : 1));
  // M3 breath: a breath lyric (canonical AP/ap or the track's trigger) maps to the `AP` phone Rust
  // recognizes. It is also UNVOICED — dropped from the pitch chain below so it breaks the line and the
  // neighbours get the §10.5 release/scoop (段中尾音) instead of gliding into/out of the breath.
  const mapLyric = (l: string) => (isBreathLyric(l, breathToken) ? "AP" : l);

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
    if (noteFrames > 0) triples.push({ lyric: mapLyric(n.lyric), note_num: n.pitch, frames: noteFrames });
    cursor = end;
  }

  const frameCount = frameOf(cursor); // cursor == last note end == Σ(triple frames)
  // f0 sees the notes WITHOUT breaths (they're unvoiced): a breath's frames fall in a rest gap → voiced 0,
  // and its neighbours become phrase edges (release-drift before, onset-scoop after). frameCount is unchanged
  // (it's the triple cursor incl. breath frames), so the array length still equals Σ(triple frames).
  const pitchNotes = sorted.filter((n) => !isBreathLyric(n.lyric, breathToken));
  const { cents, voiced } = evalF0CentsFrames(
    pitchNotes,
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
  /** The render-input signature this bake corresponds to — stamped on the deposited lane so a later Play
   *  can skip re-rendering an unchanged segment (see vocalRenderSig / isVocalDirty). */
  renderedSig?: string;
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
      deposit([{ laneId: VOCAL_LANE_ID, laneLabel, group: laneLabel, audioPath: outputPath, totalDurationMs: info.duration_ms, waveformPeaks: info.peaks, outputNodeId: VOCAL_LANE_ID, renderedSig: req.renderedSig }]);
    }
  } catch (e) {
    if (seg()) deposit(prevOutputs); // restore the prior bake (re-render) or clear the first-render spinner
    throw e;
  } finally {
    useAppStore.getState().setVocalRenderActive(false);
  }
}

// ── Auto-render-on-Play (S55): render vocal tracks whose notes/params CHANGED since their last bake, and
//    skip the unchanged ones. The dirty test compares a render-input signature against the one stamped on
//    the bake (renderedSig). One shared render path (renderVocalPart) backs BOTH the sidebar's manual
//    Render button and this batch, so they can never drift. ──

/** The full set of inputs a bake depends on, as one string: the segment content (notes + pitchDev +
 *  paramCurves, via contentSig), the track's vocal params (backend/speaker/lang/transpose/transition/
 *  sovits/rvc/breathToken, via vocalParamsSig), the singer (voiceModel), and the tempo — buildVocalScore
 *  derives its 50fps grid + f0 from the tempo, so a BPM change alters the bake even with identical notes.
 *  Reuses the history helpers (single source — never fork a sig). */
export function vocalRenderSig(track: Track, seg: Segment, tempo: number): string {
  return `${contentSig(seg.content)}|vp:${vocalParamsSig(track.vocalParams)}|vm:${track.voiceModel ?? ""}|bpm:${tempo}`;
}

/** Resolve a track's singer + build its score + invoke the render, stamping the render-input sig on the
 *  deposit. The ONE render code path (the sidebar button and the Play batch both call this). Throws
 *  VOCAL_NO_VOICE / VOCAL_EMPTY (caller maps to a toast); VOCAL_RENDER_BUSY bubbles from renderVocalSegment. */
export async function renderVocalPart(track: Track, seg: Segment, tempo: number, laneLabel: string): Promise<void> {
  if (seg.content.type !== "notes") return;
  const vp = track.vocalParams ?? DEFAULT_VOCAL_PARAMS;
  const entry = useVoiceModelStore.getState().models[vp.backend]?.find((m) => m.name === track.voiceModel);
  if (!entry) throw new Error(VOCAL_NO_VOICE);
  const { triples, f0Cents, f0Voiced } = buildVocalScore(
    seg.content.notes, seg.content.pitchDev, tempo, vp.transition, vp.breathToken ?? "AP",
  );
  if (triples.length === 0) throw new Error(VOCAL_EMPTY);
  await renderVocalSegment({
    trackId: track.id,
    segmentId: seg.id,
    laneLabel,
    voiceName: entry.name,
    modelPath: entry.path,
    triples,
    f0Cents,
    f0Voiced,
    options: {
      backend: vp.backend,
      cv_speaker_id: vp.speakerId,
      lang_id: vp.langId,
      transpose: vp.transpose,
      sovits: { ...SOVITS_DEFAULTS, ...(vp.sovits ?? {}) },
      rvc: { ...RVC_DEFAULTS, ...(vp.rvc ?? {}) },
    },
    renderedSig: vocalRenderSig(track, seg, tempo),
  });
}

/** True when a notes segment needs a (re-)bake: it has notes, a resolvable singer, and either no bake yet
 *  or a bake whose stamped sig no longer matches the current inputs. A segment with no resolvable singer is
 *  NOT dirty (we can't render it — skip silently rather than fail the batch). Ignores loading placeholders. */
export function isVocalDirty(track: Track, seg: Segment, tempo: number): boolean {
  if (seg.content.type !== "notes") return false;
  const bake = seg.processedOutputs?.find((o) => o.laneId === VOCAL_LANE_ID && !o.loading);
  // A segment emptied of ALL its notes but still holding a bake reconciles to SILENCE — else the old
  // singing keeps playing for a segment that has no notes (segmentPlaysLanes still schedules the stem).
  // There's nothing to bake, so renderDirtyVocals clears the overlay instead of rendering.
  if (seg.content.notes.length === 0) return !!bake;
  const vp = track.vocalParams;
  if (!vp) return false;
  const entry = useVoiceModelStore.getState().models[vp.backend]?.find((m) => m.name === track.voiceModel);
  if (!entry) return false;
  if (!bake) return true;
  return bake.renderedSig !== vocalRenderSig(track, seg, tempo);
}

/** Every dirty vocal segment across all tracks (read live). Empty ⇒ Play proceeds with zero added latency. */
export function collectDirtyVocals(tempo: number): Array<{ trackId: string; segmentId: string }> {
  const out: Array<{ trackId: string; segmentId: string }> = [];
  for (const tr of useProjectStore.getState().tracks) {
    for (const sg of tr.segments) if (isVocalDirty(tr, sg, tempo)) out.push({ trackId: tr.id, segmentId: sg.id });
  }
  return out;
}

/** Poll until no vocal render is in flight (the backend single-flights; a manual Render started just before
 *  Play must finish + deposit its fresh sig before the batch re-tests dirtiness). Bounded so a wedged render
 *  can't hang Play forever. */
async function waitVocalIdle(timeoutMs = 120000): Promise<void> {
  const start = performance.now();
  while (useAppStore.getState().vocalRenderActive) {
    if (performance.now() - start > timeoutMs) return;
    await new Promise((r) => setTimeout(r, 50));
  }
}

export interface DirtyRenderResult { rendered: number; failed: number; cancelled: boolean; }

/** Render a list of dirty segments SEQUENTIALLY (the backend voice guard cross-kills concurrent runs, so
 *  parallel is unsafe). Each item is re-read + re-tested live before rendering (it may have been deleted, or
 *  already re-baked by a racing trigger). `shouldCancel` (a second Play press) aborts between items and, via
 *  cancel_voice, mid-render (the in-flight invoke throws → caught → we bail). */
export async function renderDirtyVocals(
  list: Array<{ trackId: string; segmentId: string }>,
  tempo: number,
  laneLabel: string,
  opts?: { shouldCancel?: () => boolean },
): Promise<DirtyRenderResult> {
  await waitVocalIdle();
  let rendered = 0;
  let failed = 0;
  for (const { trackId, segmentId } of list) {
    if (opts?.shouldCancel?.()) return { rendered, failed, cancelled: true };
    const tr = useProjectStore.getState().tracks.find((t) => t.id === trackId);
    const sg = tr?.segments.find((s) => s.id === segmentId);
    if (!tr || !sg || !isVocalDirty(tr, sg, tempo)) continue;
    // Emptied-of-notes segment with a stale bake → clear the overlay (reconcile to silence, nothing to render).
    if (sg.content.type === "notes" && sg.content.notes.length === 0) {
      useProjectStore.getState().replaceProcessedOutputs(tr.id, sg.id, []);
      rendered++;
      continue;
    }
    try {
      await renderVocalPart(tr, sg, tempo, laneLabel);
      rendered++;
    } catch {
      if (opts?.shouldCancel?.()) return { rendered, failed, cancelled: true };
      failed++;
    }
  }
  return { rendered, failed, cancelled: false };
}
