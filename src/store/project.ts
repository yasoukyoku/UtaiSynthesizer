import { create } from "zustand";
import { useMemo } from "react";
import { invoke } from "@tauri-apps/api/core";
import type {
  Track,
  Segment,
  SegmentContent,
  LaneControl,
  ProcessedOutput,
  LaneClip,
  Note,
  PitchCurve,
  TempoDetect,
  VocalTrackParams,
} from "../types/project";
import { TimeAxis } from "../lib/timeAxis";
import { normalizeNotesArray, normalizeCurve, DEFAULT_TRANSITION } from "../lib/vocalNotes";
import { sliceCurveAtTick } from "../lib/f0eval";
import { orderProcessedOutputs, laneControlFor } from "../lib/trackLayout";
import {
  flooredDurationTicks,
  laneGroupName,
  laneLabelParts,
  recordDetachLineage,
  ticksToMs,
} from "../lib/audio/laneOps";
import { MAX_DOWNBEAT } from "../lib/constants";
import { useWorkflowStore } from "./workflow";
import { useAudioStore } from "./audio";

/** Cancel any in-flight render for the given segment ids — used when a segment/track is DELETED, the only
 *  op besides the Stop button that should cancel a render (split/move/rename/etc. must NOT). The still-
 *  running engine loop polls isCancelled and then stops the (single, global) Rust separation; the
 *  execution SETTLES (leaves 'running', un-sticking the quit/busy warning) when that loop obeys the
 *  cancel — cancelExecution itself only flags it (S62b: an instant flip made the UI claim "stopped"
 *  while the backend was still working). The JS flag is per segment id; the cancel_voice invoke below
 *  is app-GLOBAL (same semantics as cancel_separation) — a concurrent voice render of ANOTHER segment
 *  would also abort. Accepted trade-off: voice invokes carry no job id, and the deleted segment's run
 *  must not keep burning GPU minutes. */
function cancelRunningRenders(segmentIds: string[]) {
  const wf = useWorkflowStore.getState();
  let hadRunning = false;
  for (const id of segmentIds) {
    if (wf.executions[id]?.status === "running") { wf.cancelExecution(id); hadRunning = true; }
  }
  // Separation stops via the engine loop's isCancelled polling, but voice invokes
  // (run_rvc/run_sovits) are direct awaits — only the Rust-side flag can abort them mid-run.
  if (hadRunning) void invoke("cancel_voice").catch(() => {});
}

/** Gesture base for the BPM edit session (beginTempoScale/endTempoScale): setTempo scales geometry
 *  from THIS fixed snapshot so per-keystroke intermediate values can't accumulate rounding error. */
let tempoScaleBase: { tempo: number; tracks: Track[]; playheadTick: number } | null = null;

// ─── ② Vocal-note editing (S48 Phase 3) — data-layer store actions (no editor UI yet) ─────────────

/** Seed for a track's first vocal-param write (partial updates merge onto this). */
export const DEFAULT_VOCAL_PARAMS: VocalTrackParams = { backend: "sovits", speakerId: 49, langId: 2, transpose: 0, formant: 0, transition: { ...DEFAULT_TRANSITION }, breathToken: "AP", autoTuneExpr: 2, autoTuneVib: 1, autoTuneTake: 0 };

// `normalizeNote` / `normalizeNotesArray` / `normalizeCurve` — the canonical write-hygiene funnel — now
// live in `../lib/vocalNotes` (the SINGLE source shared by the store, the .usp loader, and the editor;
// §9.5). Every note-mutating action below passes its result array through `normalizeNotesArray` so storage
// order is a pure function of content (sort-on-write), and curves through `normalizeCurve`.

type NotesContent = Extract<SegmentContent, { type: "notes" }>;

/** Immutably transform ONE notes-segment's content (other tracks / segments / non-notes segments left as
 *  the same ref). Returns a fresh tracks[] with new refs only along the matched path — installHistory +
 *  autosave both key off ref changes, so this auto-captures + auto-persists (never mutate a note in place).
 *  A mis-targeted call (wrong id / an audioClip segment) changes nothing meaningful → the sig is unchanged
 *  → installHistory early-outs (no phantom undo step). */
function mapNotesContent(
  tracks: Track[],
  trackId: string,
  segmentId: string,
  fn: (c: NotesContent) => NotesContent,
): Track[] {
  return tracks.map((t) => {
    if (t.id !== trackId) return t;
    return {
      ...t,
      segments: t.segments.map((seg) =>
        seg.id === segmentId && seg.content.type === "notes"
          ? { ...seg, content: fn(seg.content) }
          : seg,
      ),
    };
  });
}

type AudioClipContent = Extract<SegmentContent, { type: "audioClip" }>;

/** audioClip twin of mapNotesContent — same immutable path-replacement contract. */
function mapAudioContent(
  tracks: Track[],
  trackId: string,
  segmentId: string,
  fn: (c: AudioClipContent) => AudioClipContent,
): Track[] {
  return tracks.map((t) => {
    if (t.id !== trackId) return t;
    return {
      ...t,
      segments: t.segments.map((seg) =>
        seg.id === segmentId && seg.content.type === "audioClip"
          ? { ...seg, content: fn(seg.content) }
          : seg,
      ),
    };
  });
}

/** Apply a transform to EVERY curve in a bag with a canonical rebuild (sorted keys, transforms
 *  returning undefined drop the entry, empty bag → undefined). THE shared shape for the S59
 *  geometry ops that must move ALL box-relative curves together (clip paramCurves + S59b
 *  laneLoudness): setTempo ×k, setSegmentStretch ×rNew/rOld, split slice, resizeL rebase. */
export function mapCurveBag(
  bag: Record<string, PitchCurve> | undefined,
  fn: (cv: PitchCurve) => PitchCurve | undefined,
): Record<string, PitchCurve> | undefined {
  if (!bag) return undefined;
  const out: Record<string, PitchCurve> = {};
  for (const k of Object.keys(bag).sort()) {
    const next = fn(bag[k]!);
    if (next) out[k] = next;
  }
  return Object.keys(out).length ? out : undefined;
}

/** Slice every curve in a paramCurves bag at a split boundary (ticks, segment-relative) — THE
 *  shared split helper for notes parts AND audio clips (both carry box-relative curves). Each half
 *  gets a boundary seam sample (sliceCurveAtTick) so a held non-zero region survives on both.
 *  Exported: Arrangement's resizeL rebases the audio loudness curve through this same helper. */
export function sliceParamCurves(
  pc: Record<string, PitchCurve> | undefined,
  boundary: number,
  which: "left" | "right",
): Record<string, PitchCurve> | undefined {
  if (!pc) return undefined;
  const out: Record<string, PitchCurve> = {};
  for (const k of Object.keys(pc).sort()) {
    const norm = normalizeCurve(sliceCurveAtTick(pc[k], boundary)[which], "param");
    if (norm) out[k] = norm;
  }
  return Object.keys(out).length ? out : undefined;
}

/** The ONE paramCurves write funnel (notes vocal lanes + audioClip loudness lane): normalizeCurve
 *  ("param" quantum) + sorted-key canonical Record + delete-when-empty — the sig↔serialize rule. */
function withParamCurve<C extends NotesContent | AudioClipContent>(
  c: C,
  param: string,
  curve: PitchCurve | undefined,
): C {
  const src = { ...(c.paramCurves ?? {}) };
  const norm = normalizeCurve(curve, "param");
  if (norm) src[param] = norm;
  else delete src[param];
  const next = { ...c };
  const keys = Object.keys(src);
  if (keys.length > 0) {
    const canonical: Record<string, PitchCurve> = {};
    for (const k of keys.sort()) canonical[k] = src[k]!;
    next.paramCurves = canonical;
  } else {
    delete next.paramCurves;
  }
  return next;
}

interface ProjectState {
  name: string;
  dirty: boolean;
  filePath: string | null;
  tracks: Track[];
  tempo: number;
  timeSignature: [number, number];
  selectedNotes: string[];
  playheadTick: number;

  addTrack: (track: Track, index?: number) => void;
  removeTrack: (id: string) => void;
  /** Move a track from one position to another (drag-reorder in the track-header column). */
  reorderTrack: (from: number, to: number) => void;
  updateTrack: (id: string, updates: Partial<Track>) => void;
  /** Move many segments at once (multi-selection drag) in a single update — keyed by segment id
   *  to its captured original startTick, shifted by `deltaTicks`. One set ⇒ one redraw per frame. */
  moveSegmentsBy: (origBySeg: Record<string, number>, deltaTicks: number) => void;
  /** Split a segment at `atTick`. Returns the NEW (right-half) segment id, or null if the split was a
   *  no-op (atTick outside the span). Clones the render cache onto the new half (post-render inheritance). */
  splitSegment: (trackId: string, segmentId: string, atTick: number) => string | null;
  deleteSegment: (trackId: string, segmentId: string) => void;
  /** Delete many segments at once (multi-selection) in one update. */
  deleteSegments: (items: { trackId: string; segmentId: string }[]) => void;
  setTempo: (bpm: number) => void;
  /** Set the GLOBAL time signature (tempo & meter are timeline-level, shared by every track — never
   *  per-track). Keeps the scalar `[num,den]` shape (no serialization/undo-snapshot migration): the
   *  new array reference makes installHistory capture one undo step and marks the project dirty. Unlike
   *  setTempo this moves NO ticks — notes/clips keep their positions; only the bar/beat GRID re-derives
   *  (via TimeAxis). Denominator finally goes live (den=8 ⇒ 240 ticks/beat, 1440/bar for 6/8). */
  setTimeSignature: (num: number, den: number) => void;
  /** Open/close a BPM edit session (the Toolbar input's focus/blur) — setTempo scales from the
   *  session-start geometry so intermediate keystroke values can't compound rounding error. */
  beginTempoScale: () => void;
  endTempoScale: () => void;
  setPlayhead: (tick: number) => void;
  selectNotes: (ids: string[]) => void;
  toggleTrackExpanded: (trackId: string) => void;
  /** S59: toggle the audio track's loudness-lane band (view state, mirrors toggleTrackExpanded). */
  toggleLoudnessLane: (trackId: string) => void;
  /** Flip the track's SOURCE selector (play original audio vs deposited sub-lanes — see
   *  Track.playOriginal / segmentPlaysLanes). Reschedules live playback so the audible source
   *  switches immediately (mirrors removeTrack's in-store bump — one path for every call site). */
  setTrackPlayOriginal: (trackId: string, playOriginal: boolean) => void;
  /** Set a 组's volume/pan (keyed by the producing Output node id — see Track.laneControls).
   *  `legacyLaneId` (the group's first-row laneId) seeds a MISSING group entry from the legacy
   *  pre-S28 per-laneId control the faders display via laneControlFor — without it, a pan-only first
   *  write would seed volumeDb: 0 and silently shadow a legacy saved volume (review-caught). */
  updateLaneControl: (trackId: string, groupId: string, updates: Partial<LaneControl>, legacyLaneId?: string) => void;
  /** Toggle a lane ROW's mute (keyed by laneRowKey — see Track.laneMutes; loose per-row semantics). */
  setLaneMute: (trackId: string, rowKey: string, muted: boolean) => void;
  /** Set (or clear, when `clips` is undefined) a segment's non-destructive sub-lane recipe for one
   *  Output-node group (keyed by outputNodeId). Undoable — laneOps is in the history meaningfulSig. */
  updateSegmentLaneOps: (trackId: string, segmentId: string, outputNodeId: string, clips: LaneClip[] | undefined) => void;

  // ── ② Vocal-note data-layer actions (S48 Phase 3). All mutate `seg.content.notes` of a `notes`
  //    segment (no-op on an audioClip segment), immutably + `dirty:true`, so installHistory captures ONE
  //    undo step and autosave persists. `normalizeNote` strips default optionals on write (§5). ──
  /** Append a vocal note to a notes-segment (normalized on write). */
  addVocalNote: (trackId: string, segmentId: string, note: Note) => void;
  /** Patch one vocal note by id (partial merge, re-normalized). */
  updateVocalNote: (trackId: string, segmentId: string, noteId: string, updates: Partial<Note>) => void;
  /** Delete one or many vocal notes by id (box-select delete). */
  deleteVocalNotes: (trackId: string, segmentId: string, noteIds: string[]) => void;
  /** Atomic BATCH note edit = ONE undo step (§9.5). Every editor gesture (create/move/resize/truncate/
   *  paste/delete) routes through here; the result array is sort-canonicalized (normalizeNotesArray). */
  applyNoteEdits: (
    trackId: string,
    segmentId: string,
    edits: { add?: Note[]; update?: Record<string, Partial<Note>>; remove?: string[] },
  ) => void;
  /** Create an EMPTY notes segment (part) on a vocal track and return its id (the editor then opens it).
   *  createVocalTrack seeds no segments — this is the missing "insert a notes part" action (§9.5). */
  createVocalPart: (trackId: string, startTick: number, durationTicks: number) => string;
  /** Set (or clear, when `curve` is undefined/empty) the part-level hand-drawn pitch deviation. */
  setSegmentPitchDev: (trackId: string, segmentId: string, curve: PitchCurve | undefined) => void;
  /** Set (or clear) one named parameter automation lane on a segment — notes parts (render-domain
   *  vocal lanes) AND audio clips (S59 playback-domain loudness lane) share this single funnel. */
  setSegmentParamCurve: (trackId: string, segmentId: string, param: string, curve: PitchCurve | undefined) => void;
  /** S59: set (or clear) an audio clip's detected BPM/beat grid. The SINGLE canonical write point —
   *  values are rounded here so serialize stays byte-stable. Undoable (contentSig). */
  setSegmentTempoDetect: (trackId: string, segmentId: string, detect: TempoDetect | undefined) => void;
  /** S59b: set (or clear) one GROUP's sub-lane loudness envelope on an audio segment (canonical
   *  funnel: normalizeCurve param + sorted keys + delete-when-empty). Undoable (meaningfulSig). */
  setSegmentLaneLoudness: (trackId: string, segmentId: string, groupId: string, curve: PitchCurve | undefined) => void;
  /** S59b: toggle a GROUP's envelope display/edit mode (view state; canonical true-only record). */
  toggleLaneLoudnessOpen: (trackId: string, groupId: string) => void;
  /** S59 Tempo Slider: set an audio clip's stretch factor (played/source duration). Rescales
   *  durationTicks so the SOURCE window stays fixed (left edge anchored); r≈1 deletes the field
   *  (false-dirty rule). One set = one undo step. Caller pre-generates the stretched artifacts
   *  (ensureStretched) BEFORE committing so playback never blocks on a cold stretch. */
  setSegmentStretch: (trackId: string, segmentId: string, stretch: number) => void;
  /** Merge a partial vocal-params update onto the track (seeds defaults on first write). Undoable. */
  setVocalParams: (trackId: string, updates: Partial<VocalTrackParams>) => void;
  /** Per-lane deposit: replace only the lanes present in `outputs` (keyed by outputNodeId), keep sibling
   *  lanes. Used by the live Output reconciler so depositing one lane never clobbers the others. */
  mergeProcessedOutputs: (trackId: string, segmentId: string, outputs: ProcessedOutput[]) => void;
  /** Remove all lanes contributed by one Output node — e.g. to clear loading placeholders when a
   *  deposit fails mid-decode, so the lane doesn't spin forever. */
  removeProcessedOutputsForNode: (trackId: string, segmentId: string, outputNodeId: string) => void;
  clearProcessedOutputs: (trackId: string, segmentId: string) => void;
  /** Replace a segment's lanes wholesale (empty ⇒ undefined). Used by RenderLinkWatcher to mirror a
   *  split source's final lanes onto the split-out half once its render settles. */
  replaceProcessedOutputs: (trackId: string, segmentId: string, outputs: ProcessedOutput[]) => void;
  /** Apply the project-store half of an Output-group DETACH (see engine.planDetachGroup): rewrite the
   *  deposited lanes to the new per-edge Output nodes IN PLACE (no re-decode — same audio), COPY the
   *  group's laneOps recipe to each new node key (the OLD key is kept so the graph-undo of the detach
   *  converges back with the recipe intact), and COPY each lane's mixer entry to its new row key.
   *  Caller wraps this in history.runSilent — laneOps/laneControls are in the meaningfulSig, and the
   *  detach's undo home is the NODE-GRAPH stack (the reconciler converges the lanes back on undo). */
  applyLaneDetach: (
    trackId: string,
    segmentId: string,
    oldNodeId: string,
    mapping: { oldLaneId: string; newLaneId: string; newNodeId: string; group: string; laneLabel: string }[],
  ) => void;
}

export const useProjectStore = create<ProjectState>((set, get) => ({
  name: "",
  dirty: false,
  filePath: null,
  tracks: [],
  tempo: 120,
  timeSignature: [4, 4],
  selectedNotes: [],
  playheadTick: 0,

  addTrack: (track, index) =>
    set((s) => {
      const tracks = [...s.tracks];
      if (index === undefined || index < 0 || index >= tracks.length) {
        tracks.push(track); // append (click-import, or out-of-range)
      } else {
        tracks.splice(index, 0, track); // insert at the dragged position
      }
      return { tracks, dirty: true };
    }),

  removeTrack: (id) => {
    // Deleting a track must STOP any render in flight for its segments (the only non-Stop op that cancels).
    const track = get().tracks.find((t) => t.id === id);
    if (track) cancelRunningRenders(track.segments.map((seg) => seg.id));
    set((s) => ({
      tracks: s.tracks.filter((t) => t.id !== id),
      dirty: true,
    }));
    // Reschedule playback so the removed track's scheduled Web-Audio sources + the playhead don't keep
    // running — deleting the LAST track then reschedules to "empty" and the Toolbar stops playback. (Unlike
    // segment deletes, whose UI call sites bump, removeTrack's call sites did NOT — so a track delete during
    // playback previously left audio + playhead running with an empty canvas.)
    if (useAudioStore.getState().isPlaying) useAudioStore.getState().bumpSchedule();
  },

  reorderTrack: (from, to) =>
    set((s) => {
      if (from === to || from < 0 || to < 0 || from >= s.tracks.length || to >= s.tracks.length) return {};
      const tracks = [...s.tracks];
      const [moved] = tracks.splice(from, 1);
      tracks.splice(to, 0, moved!);
      return { tracks, dirty: true };
    }),

  updateTrack: (id, updates) =>
    set((s) => ({
      tracks: s.tracks.map((t) => (t.id === id ? { ...t, ...updates } : t)),
      dirty: true,
    })),

  moveSegmentsBy: (origBySeg, deltaTicks) =>
    set((s) => {
      // Group clamp: bound the shared delta so the LEFTMOST selected segment stops at tick 0 and the
      // rest keep their relative offsets (clamping each segment independently would bunch them up).
      const origs = Object.values(origBySeg);
      const d = origs.length ? Math.max(deltaTicks, -Math.min(...origs)) : deltaTicks;
      return {
        dirty: true,
        tracks: s.tracks.map((t) => {
          if (!t.segments.some((seg) => origBySeg[seg.id] !== undefined)) return t;
          return {
            ...t,
            segments: t.segments.map((seg) =>
              origBySeg[seg.id] !== undefined
                ? { ...seg, startTick: origBySeg[seg.id]! + d }
                : seg,
            ),
          };
        }),
      };
    }),

  splitSegment: (trackId, segmentId, atTick) => {
    const newId = crypto.randomUUID();
    // Upfront no-op guard (§5 false-dirty): an out-of-range tick / missing segment are no-ops — return WITHOUT
    // entering set() so `dirty` never falsely flips. ② NOTES snap: if the split falls INSIDE a note, SNAP it
    // to that note's END so the note stays WHOLE on the left half (§user — a mid-note cut can't cleanly halve
    // the 1/12 grid and a straddling note would poke out of its box). A snap that empties either half = no-op.
    let splitAt = atTick;
    {
      const st = get();
      const seg = st.tracks.find((t) => t.id === trackId)?.segments.find((sg) => sg.id === segmentId);
      if (!seg || atTick <= seg.startTick || atTick >= seg.startTick + seg.durationTicks) return null;
      if (seg.content.type === "notes") {
        const rel = atTick - seg.startTick;
        const straddler = seg.content.notes.find((n) => n.tick < rel && rel < n.tick + n.duration);
        if (straddler) splitAt = seg.startTick + straddler.tick + straddler.duration;
        if (splitAt <= seg.startTick || splitAt >= seg.startTick + seg.durationTicks) return null;
      }
    }
    // Is a render in flight for this segment? If so the split must NOT drop the in-progress sub-lanes:
    // carry the loading placeholders to the right half too + link it, so the single ongoing render lands
    // on BOTH halves once it settles (RenderLinkWatcher mirrors the source's final lanes onto the new half).
    // A segment that is ITSELF a pending link target (you split a still-loading right half AGAIN before the
    // render settles) has no execution of its own — treat it as rendering too, and chain the new piece to
    // the ULTIMATE render owner so every descendant inherits when the single global render settles.
    const wfState = useWorkflowStore.getState();
    const linkedSource: string | undefined = wfState.renderLinks[segmentId];
    const rendering = wfState.executions[segmentId]?.status === "running" || linkedSource !== undefined;
    let didSplit = false;
    let splitNotes = false;
    set((s) => ({
      dirty: true,
      tracks: s.tracks.map((t) => {
        if (t.id !== trackId) return t;
        const segIdx = t.segments.findIndex((seg) => seg.id === segmentId);
        if (segIdx < 0) return t;
        const seg = t.segments[segIdx]!;

        if (splitAt <= seg.startTick || splitAt >= seg.startTick + seg.durationTicks) return t;

        const leftDuration = splitAt - seg.startTick;
        const rightDuration = seg.durationTicks - leftDuration;

        // ── ② NOTES split (§9.6): partition by ONSET (the straddler was snapped to its end upfront, so no
        //    note crosses the seam), give the RIGHT notes FRESH ids + rebase ticks by −leftDuration (SAME ids
        //    = corruption), slice+rebase pitchDev/paramCurves, run each half through the write-hygiene funnel
        //    (normalizeNotesArray/normalizeCurve — canonical order, omit-empty → no false-dirty), and CARRY +
        //    WINDOW the baked stem onto both halves (a split is NOT a re-render — see the deposit block below). ──
        if (seg.content.type === "notes") {
          const c = seg.content;
          const leftNotes: Note[] = [];
          const rightNotes: Note[] = [];
          for (const n of c.notes) {
            if (n.tick < leftDuration) leftNotes.push(n);
            else rightNotes.push({ ...n, id: crypto.randomUUID(), tick: n.tick - leftDuration });
          }
          const dev = sliceCurveAtTick(c.pitchDev, leftDuration);
          const devL = normalizeCurve(dev.left, "cents");
          const devR = normalizeCurve(dev.right, "cents");
          const paramsL = sliceParamCurves(c.paramCurves, leftDuration, "left");
          const paramsR = sliceParamCurves(c.paramCurves, leftDuration, "right");
          const leftContent: SegmentContent = { type: "notes", notes: normalizeNotesArray(leftNotes), ...(devL ? { pitchDev: devL } : {}), ...(paramsL ? { paramCurves: paramsL } : {}) };
          const rightContent: SegmentContent = { type: "notes", notes: normalizeNotesArray(rightNotes), ...(devR ? { pitchDev: devR } : {}), ...(paramsR ? { paramCurves: paramsR } : {}) };
          // ② CARRY + WINDOW the baked stem — a split is NOT a re-render (§user: "把已有整段在切点切开"). Both halves
          // share the parent stem; the left keeps its offset (its shorter box windows [offset, offset+leftDur]),
          // the right advances the stem offset by the left duration (ms) → plays [offset+leftDur, …]. Loading
          // placeholders are dropped. The bakes keep their ORIGINAL renderedSig (the whole-stem content); the
          // frontend caller then stamps each half's `windowSig` = vocalRenderSig(this half's content) so
          // isVocalDirty accepts the window with NO re-render (dual-sig: renderedSig OR windowSig must match).
          // Both sigs live on the OVERLAY (never undoable) so they can't desync — an undo-of-split leaves the
          // whole-stem renderedSig matching the restored full content (clean), and any real drift fails BOTH.
          const leftMs = ticksToMs(leftDuration, s.tempo);
          const settled = (seg.processedOutputs ?? []).filter((o) => !o.loading);
          const leftOuts = settled.map((o) => ({ ...o }));
          const rightOuts = settled.map((o) => ({ ...o, offsetMs: (o.offsetMs ?? 0) + leftMs }));
          const leftSeg: Segment = { ...seg, durationTicks: leftDuration, content: leftContent, processedOutputs: leftOuts.length ? leftOuts : undefined };
          const rightSeg: Segment = { id: newId, startTick: splitAt, durationTicks: rightDuration, content: rightContent, ...(rightOuts.length ? { processedOutputs: rightOuts } : {}) };
          didSplit = true;
          splitNotes = true;
          const newSegments = [...t.segments];
          newSegments.splice(segIdx, 1, leftSeg, rightSeg);
          return { ...t, segments: newSegments };
        }

        let leftClipContent = seg.content;
        let rightContent = seg.content;
        if (seg.content.type === "audioClip") {
          const c = seg.content;
          // S59: offsetMs is a SOURCE coordinate — a stretched clip's left half covers
          // leftDuration/r source ms, so the right window starts that much later, not leftMs.
          const splitOffsetMs = ticksToMs(leftDuration, s.tempo) / (c.stretch ?? 1);
          // S59: the loudness lane (box-relative ticks) splits like the vocal curves — each half
          // keeps its own points + a seam sample (a held offset survives on both halves).
          const pcL = sliceParamCurves(c.paramCurves, leftDuration, "left");
          const pcR = sliceParamCurves(c.paramCurves, leftDuration, "right");
          leftClipContent = { ...c, ...(pcL ? { paramCurves: pcL } : {}) };
          if (!pcL) delete leftClipContent.paramCurves;
          rightContent = { ...c, offsetMs: c.offsetMs + splitOffsetMs, ...(pcR ? { paramCurves: pcR } : {}) };
          if (!pcR) delete rightContent.paramCurves;
        }
        // S59b: the per-group lane envelopes are box-relative like the clip curves — slice them
        // for both halves the same way (seam sample keeps a held offset alive on both sides).
        const llL = mapCurveBag(seg.laneLoudness, (cv) => normalizeCurve(sliceCurveAtTick(cv, leftDuration).left, "param"));
        const llR = mapCurveBag(seg.laneLoudness, (cv) => normalizeCurve(sliceCurveAtTick(cv, leftDuration).right, "param"));
        const leftSeg: Segment = { ...seg, durationTicks: leftDuration, content: leftClipContent };
        if (llL) leftSeg.laneLoudness = llL;
        else delete leftSeg.laneLoudness;
        // Carry the baked sub-lane render onto the right half too (the left half keeps it via the spread).
        // POST-render (settled): drop loading placeholders — a stale one would spin forever (no reconciler
        // on the new id finalizes it). MID-render (rendering): KEEP the placeholders — the new half is
        // linked below and RenderLinkWatcher finalizes them when the shared render settles.
        const rightOutputs = (rendering
          ? (seg.processedOutputs ?? [])
          : (seg.processedOutputs ?? []).filter((o) => !o.loading)
        ).map((o) => ({ ...o }));
        const rightSeg: Segment = {
          id: newId,
          startTick: splitAt,
          durationTicks: rightDuration,
          content: rightContent,
          // Carry the per-segment node graph AND the baked sub-lane render to the right half — dropping
          // either on split was latent data loss (the newly split-out half lost all its processing). The
          // stem spans the WHOLE (trimmed) source and every lane windows it by content.offsetMs (advanced
          // above) + durationTicks — exactly like the original audio — so each half plays/draws its own
          // [offset, offset+dur] slice of the SAME stem with zero re-separation.
          workflow: seg.workflow,
          processedOutputs: rightOutputs.length > 0 ? rightOutputs : undefined,
          // Sub-lane edit recipe rides the split UNCHANGED: laneOps are in STEM MS (invariant under the
          // split's offset advance), so both halves reference the same recipe and each intersects it with
          // its own visible window at read time. The left half keeps it via the {...seg} spread above.
          laneOps: seg.laneOps,
          // S59b lane envelopes are box-relative → SLICED (unlike laneOps), see llL/llR above.
          ...(llR ? { laneLoudness: llR } : {}),
        };

        didSplit = true;
        const newSegments = [...t.segments];
        newSegments.splice(segIdx, 1, leftSeg, rightSeg);
        return { ...t, segments: newSegments };
      }),
    }));
    if (didSplit && !splitNotes) {
      // audioClip only: clone the render CACHE (+ settled node badges/execution) onto the new half so a
      // POST-render split "remembers" it was rendered: deleting + reconnecting an Output edge re-deposits
      // from the cache instead of a full re-run. NOTES have no workflow cache/execution (their bake is a
      // vocalRender overlay, CARRIED + windowed above via offsetMs+windowSig → no re-render), so this is skipped.
      useWorkflowStore.getState().cloneSegmentState(segmentId, newId);
      // Mid-render: link the new half to the ULTIMATE render owner (chain through an existing link) so the
      // single ongoing render deposits onto it too when it settles.
      if (rendering) useWorkflowStore.getState().linkRender(newId, linkedSource ?? segmentId);
    }
    return didSplit ? newId : null;
  },

  deleteSegment: (trackId, segmentId) => {
    cancelRunningRenders([segmentId]); // deleting a rendering segment stops its render (see helper)
    set((s) => ({
      dirty: true,
      tracks: s.tracks.map((t) => {
        if (t.id !== trackId) return t;
        return { ...t, segments: t.segments.filter((seg) => seg.id !== segmentId) };
      }),
    }));
  },

  deleteSegments: (items) => {
    cancelRunningRenders(items.map((it) => it.segmentId));
    set((s) => {
      const byTrack = new Map<string, Set<string>>();
      for (const it of items) {
        if (!byTrack.has(it.trackId)) byTrack.set(it.trackId, new Set());
        byTrack.get(it.trackId)!.add(it.segmentId);
      }
      return {
        dirty: true,
        tracks: s.tracks.map((t) => {
          const ids = byTrack.get(t.id);
          return ids ? { ...t, segments: t.segments.filter((seg) => !ids.has(seg.id)) } : t;
        }),
      };
    });
  },

  setTempo: (bpm) => {
    // REAL-TIME-preserving rescale: audio clips keep their absolute positions AND lengths in SECONDS —
    // startTick and durationTicks both scale by newBpm/baseTempo (end computed from the scaled edges so
    // touching pieces stay touching). The old formula rescaled ONLY durations, and from the FULL source
    // length (`totalDurationMs`) — correct back when every clip spanned its whole source, but a SPLIT
    // piece was blown up to "the whole song" long: pieces overlapped, stacked audibly, and gap lengths
    // changed. Note segments (MIDI) are musical — their ticks deliberately do NOT move with tempo.
    // Scaled from a stable BASE captured at gesture start (beginTempoScale) so per-keystroke edits
    // ("1" → "12" → "120") can't accumulate rounding damage; a call outside a gesture uses the current
    // state as a one-shot base. Geometry is applied onto the CURRENT tracks (never the base objects) so
    // concurrent overlay writes — a deposit landing mid-gesture — are not rolled back.
    const base = tempoScaleBase ?? { tempo: get().tempo, tracks: get().tracks, playheadTick: get().playheadTick };
    const k = bpm / base.tempo;
    if (!Number.isFinite(k) || k <= 0) return;
    const geo = new Map<string, { start: number; dur: number; pc?: Record<string, PitchCurve>; ll?: Record<string, PitchCurve> }>();
    for (const t of base.tracks) {
      for (const s of t.segments) {
        if (s.content.type !== "audioClip") continue;
        const start = Math.round(s.startTick * k);
        const end = Math.round((s.startTick + s.durationTicks) * k);
        // S59: the clip's loudness curves are box-relative TICKS glued to second-anchored audio —
        // xs must scale with the box (from the BASE, like the geometry, so per-keystroke edits
        // can't accumulate rounding). BOTH bags: clip paramCurves + S59b per-group laneLoudness.
        // Notes segments' curves stay musical (unscaled).
        const scaleXs = (cv: PitchCurve) =>
          normalizeCurve({ xs: cv.xs.map((x) => x * k), ys: [...cv.ys] }, "param");
        geo.set(s.id, {
          start,
          dur: Math.max(1, end - start),
          pc: mapCurveBag(s.content.paramCurves, scaleXs),
          ll: mapCurveBag(s.laneLoudness, scaleXs),
        });
      }
    }
    set((st) => ({
      tempo: bpm,
      dirty: true,
      // The playhead is second-anchored too, so the audible position survives a tempo change.
      playheadTick: Math.max(0, Math.round(base.playheadTick * k)),
      tracks: st.tracks.map((t) => ({
        ...t,
        segments: t.segments.map((seg) => {
          const g = geo.get(seg.id);
          if (!g) return seg;
          let content = seg.content;
          if (content.type === "audioClip") {
            const nc = { ...content };
            if (g.pc) nc.paramCurves = g.pc;
            else delete nc.paramCurves;
            content = nc;
          }
          const next = { ...seg, startTick: g.start, durationTicks: g.dur, content };
          if (g.ll) next.laneLoudness = g.ll;
          else delete next.laneLoudness;
          return next;
        }),
      })),
    }));
    // Tempo changes the tick↔seconds mapping — the already-scheduled Web Audio sources keep the old
    // layout while the playhead advances at the new rate. Reschedule like every other timing edit.
    if (useAudioStore.getState().isPlaying) useAudioStore.getState().bumpSchedule();
  },

  setTimeSignature: (num, den) => {
    // Sanitize to a valid meter: numerator ≥1 integer; denominator a power of two (the only values
    // for which TICKS_PER_BEAT*4/den is an exact tick count). The Toolbar only offers valid options,
    // so this is defensive. A NEW array reference is required — installHistory & autosave both key off
    // `timeSignature !== prev.timeSignature`. No tick geometry moves (meter only re-grids).
    const n = Math.max(1, Math.round(num));
    const ALLOWED = [1, 2, 4, 8, 16, 32];
    const d = ALLOWED.includes(den) ? den : 4;
    const cur = get().timeSignature;
    if (cur[0] === n && cur[1] === d) return; // no-op guard: don't churn undo/dirty on a same-value set
    set({ timeSignature: [n, d], dirty: true });
  },

  beginTempoScale: () => {
    tempoScaleBase = { tempo: get().tempo, tracks: get().tracks, playheadTick: get().playheadTick };
  },
  endTempoScale: () => {
    tempoScaleBase = null;
  },
  setPlayhead: (tick) => set({ playheadTick: tick }),
  selectNotes: (ids) => set({ selectedNotes: ids }),

  toggleTrackExpanded: (trackId) =>
    set((s) => ({
      tracks: s.tracks.map((t) =>
        t.id === trackId ? { ...t, expanded: !t.expanded } : t,
      ),
    })),

  // S59 loudness lane band toggle — view state exactly like expanded (no dirty, no undo step)
  toggleLoudnessLane: (trackId) =>
    set((s) => ({
      tracks: s.tracks.map((t) =>
        t.id === trackId ? { ...t, loudnessLaneOpen: !t.loudnessLaneOpen } : t,
      ),
    })),

  setTrackPlayOriginal: (trackId, playOriginal) => {
    set((s) => ({
      dirty: true, // a Mute/Solo-class output state — persisted + undoable
      tracks: s.tracks.map((t) => (t.id === trackId ? { ...t, playOriginal } : t)),
    }));
    // Source selection changes WHICH buffers are scheduled — a live gain tweak can't express it, so
    // reschedule mid-playback (the Toolbar watcher picks the bump up from the new playhead).
    if (useAudioStore.getState().isPlaying) useAudioStore.getState().bumpSchedule();
  },

  updateLaneControl: (trackId, groupId, updates, legacyLaneId) =>
    set((s) => ({
      dirty: true, // lane mix is persisted document state — must mark the project dirty (and be undoable)
      tracks: s.tracks.map((t) => {
        if (t.id !== trackId) return t;
        // Seed a missing group entry through THE read accessor (laneControlFor) so the first write
        // inherits exactly the VOLUME/PAN the fader displayed (incl. the legacy per-laneId fallback)
        // instead of resetting the other field to its default. `muted` is deliberately forced FALSE:
        // the group entry's muted is INERT (isLaneRowMuted reads laneMutes[rowKey] ?? the per-LANEID
        // control, never the groupId one), so inheriting a legacy laneId's muted=true here would both
        // be meaningless AND make a pure V/P drag flip a muted flag → describeDelta (which checks muted
        // before volume) mislabels the step "回退·子轨道静音" for an edit that only moved a fader.
        const base = legacyLaneId ? laneControlFor(t, groupId, legacyLaneId) : t.laneControls[groupId];
        const existing = { volumeDb: base?.volumeDb ?? 0, pan: base?.pan ?? 0, muted: false };
        return {
          ...t,
          laneControls: { ...t.laneControls, [groupId]: { ...existing, ...updates } },
        };
      }),
    })),

  setLaneMute: (trackId, rowKey, muted) =>
    set((s) => ({
      dirty: true,
      tracks: s.tracks.map((t) =>
        t.id === trackId ? { ...t, laneMutes: { ...(t.laneMutes ?? {}), [rowKey]: muted } } : t,
      ),
    })),

  updateSegmentLaneOps: (trackId, segmentId, outputNodeId, clips) =>
    set((s) => ({
      dirty: true,
      tracks: s.tracks.map((t) => {
        if (t.id !== trackId) return t;
        return {
          ...t,
          segments: t.segments.map((seg) => {
            if (seg.id !== segmentId) return seg;
            const next = { ...(seg.laneOps ?? {}) };
            if (clips === undefined) delete next[outputNodeId];
            else next[outputNodeId] = clips;
            return { ...seg, laneOps: Object.keys(next).length > 0 ? next : undefined };
          }),
        };
      }),
    })),

  addVocalNote: (trackId, segmentId, note) =>
    set((s) => ({
      dirty: true,
      tracks: mapNotesContent(s.tracks, trackId, segmentId, (c) => ({
        ...c,
        notes: normalizeNotesArray([...c.notes, note]), // append then sort-on-write (§9.5 single funnel)
      })),
    })),

  updateVocalNote: (trackId, segmentId, noteId, updates) =>
    set((s) => ({
      dirty: true,
      tracks: mapNotesContent(s.tracks, trackId, segmentId, (c) => ({
        ...c,
        // Re-normalize the WHOLE array so a field set to its default drops out AND order stays canonical
        // (a pitch edit can move a note past a neighbor → tick-sort funnel keeps storage order pure).
        notes: normalizeNotesArray(c.notes.map((n) => (n.id === noteId ? { ...n, ...updates } : n))),
      })),
    })),

  deleteVocalNotes: (trackId, segmentId, noteIds) =>
    set((s) => {
      const ids = new Set(noteIds);
      return {
        dirty: true,
        tracks: mapNotesContent(s.tracks, trackId, segmentId, (c) => ({
          ...c,
          notes: normalizeNotesArray(c.notes.filter((n) => !ids.has(n.id))),
        })),
      };
    }),

  applyNoteEdits: (trackId, segmentId, edits) =>
    set((s) => {
      // ONE atomic batch = ONE undo step (§9.5): remove, then patch, then add, then the single sort/
      // canonicalize funnel. All of create/move/resize/truncate/paste/delete/lyric route through here.
      const track = s.tracks.find((t) => t.id === trackId);
      const seg = track?.segments.find((sg) => sg.id === segmentId);
      if (!seg || seg.content.type !== "notes") return {};
      const removeSet = edits.remove && edits.remove.length > 0 ? new Set(edits.remove) : null;
      let next = removeSet ? seg.content.notes.filter((n) => !removeSet.has(n.id)) : seg.content.notes;
      if (edits.update) {
        const u = edits.update;
        next = next.map((n) => {
          const patch = u[n.id];
          if (!patch) return n;
          const merged = { ...n, ...patch };
          // S58 invariant: a phoneme override belongs to the LYRIC it was written for — changing the
          // lyric without explicitly re-supplying the override drops it (else a stale pinyin/ARPABET
          // silently keeps overriding the NEW lyric's pronunciation).
          if (patch.lyric !== undefined && patch.lyric !== n.lyric) {
            if (patch.phonemeInput === undefined) delete merged.phonemeInput;
            if (patch.phoneme === undefined) delete merged.phoneme;
          }
          return merged;
        });
      }
      if (edits.add && edits.add.length > 0) next = [...next, ...edits.add];
      const nextNotes = normalizeNotesArray(next);
      // No-op guard (§5 false-dirty, the user's #1 pain): a same-value edit — e.g. re-confirming a lyric
      // unchanged — must NOT set dirty (a stuck dirty flag never reconciles). Both arrays are canonical
      // (normalizeNotesArray), so JSON equality is reliable; identical → change nothing (empty set).
      if (JSON.stringify(seg.content.notes) === JSON.stringify(nextNotes)) return {};
      return { dirty: true, tracks: mapNotesContent(s.tracks, trackId, segmentId, (c) => ({ ...c, notes: nextNotes })) };
    }),

  createVocalPart: (trackId, startTick, durationTicks) => {
    const id = crypto.randomUUID();
    set((s) => ({
      dirty: true,
      tracks: s.tracks.map((t) =>
        t.id === trackId
          ? {
              ...t,
              segments: [
                ...t.segments,
                {
                  id,
                  startTick: Math.max(0, Math.round(startTick)),
                  durationTicks: Math.max(1, Math.round(durationTicks)),
                  content: { type: "notes", notes: [] } as SegmentContent,
                },
              ],
            }
          : t,
      ),
    }));
    return id;
  },

  setSegmentPitchDev: (trackId, segmentId, curve) =>
    set((s) => ({
      dirty: true,
      tracks: mapNotesContent(s.tracks, trackId, segmentId, (c) => {
        const next: NotesContent = { ...c };
        const norm = normalizeCurve(curve, "cents"); // pitchDev = integer cents
        if (norm) next.pitchDev = norm;
        else delete next.pitchDev;
        return next;
      }),
    })),

  setSegmentParamCurve: (trackId, segmentId, param, curve) =>
    set((s) => ({
      dirty: true,
      // one funnel, two content variants: withParamCurve owns the canonical write (sorted keys,
      // delete-when-empty) so notes vocal lanes and the audio loudness lane cannot drift apart.
      tracks: s.tracks.map((t) => {
        if (t.id !== trackId) return t;
        return {
          ...t,
          segments: t.segments.map((seg) =>
            seg.id === segmentId && (seg.content.type === "notes" || seg.content.type === "audioClip")
              ? { ...seg, content: withParamCurve(seg.content, param, curve) }
              : seg,
          ),
        };
      }),
    })),

  setSegmentStretch: (trackId, segmentId, stretch) =>
    set((s) => {
      const rNew = Math.round(Math.min(4, Math.max(0.25, stretch)) * 1e6) / 1e6; // canon (matches Rust {:.6})
      return {
        dirty: true,
        tracks: s.tracks.map((t) => {
          if (t.id !== trackId) return t;
          return {
            ...t,
            segments: t.segments.map((seg) => {
              if (seg.id !== segmentId || seg.content.type !== "audioClip") return seg;
              const rOld = seg.content.stretch ?? 1;
              if (Math.abs(rNew - rOld) < 1e-9) return seg;
              // the SOURCE window is the invariant; the box width in ticks follows the factor
              const winSrcMs = ticksToMs(seg.durationTicks, s.tempo) / rOld;
              const durationTicks = flooredDurationTicks(winSrcMs * rNew, s.tempo);
              const content = { ...seg.content };
              if (Math.abs(rNew - 1) < 1e-9) delete content.stretch;
              else content.stretch = rNew;
              // The loudness envelopes' box-relative xs are glued to second-anchored audio — the
              // box just changed by rNew/rOld, so the xs must scale with it (the same conversion
              // setTempo performs, audit MAJOR: without it an envelope silently ducks the WRONG
              // audio after a stretch change). BOTH bags move together: the clip-wide curves AND
              // the S59b per-group lane envelopes. One-shot Apply gesture → scaling from the
              // stored canon-rounded rOld accumulates no rounding.
              const scaleXs = (cv: PitchCurve) =>
                normalizeCurve({ xs: cv.xs.map((x) => (x * rNew) / rOld), ys: [...cv.ys] }, "param");
              const pc = mapCurveBag(content.paramCurves, scaleXs);
              if (pc) content.paramCurves = pc;
              else delete content.paramCurves;
              const ll = mapCurveBag(seg.laneLoudness, scaleXs);
              const next = { ...seg, durationTicks, content };
              if (ll) next.laneLoudness = ll;
              else delete next.laneLoudness;
              return next;
            }),
          };
        }),
      };
    }),

  setSegmentLaneLoudness: (trackId, segmentId, groupId, curve) =>
    set((s) => ({
      dirty: true,
      tracks: s.tracks.map((t) => {
        if (t.id !== trackId) return t;
        return {
          ...t,
          segments: t.segments.map((seg) => {
            if (seg.id !== segmentId || seg.content.type !== "audioClip") return seg;
            const src = { ...(seg.laneLoudness ?? {}) };
            const norm = normalizeCurve(curve, "param");
            if (norm) src[groupId] = norm;
            else delete src[groupId];
            const next = { ...seg };
            const keys = Object.keys(src);
            if (keys.length > 0) {
              const canonical: Record<string, PitchCurve> = {};
              for (const k of keys.sort()) canonical[k] = src[k]!;
              next.laneLoudness = canonical;
            } else {
              delete next.laneLoudness;
            }
            return next;
          }),
        };
      }),
    })),

  toggleLaneLoudnessOpen: (trackId, groupId) =>
    set((s) => ({
      tracks: s.tracks.map((t) => {
        if (t.id !== trackId) return t;
        const cur = { ...(t.laneLoudnessOpen ?? {}) };
        if (cur[groupId]) delete cur[groupId];
        else cur[groupId] = true;
        const next = { ...t };
        const keys = Object.keys(cur);
        if (keys.length > 0) {
          const canonical: Record<string, boolean> = {};
          for (const k of keys.sort()) canonical[k] = true;
          next.laneLoudnessOpen = canonical;
        } else {
          delete next.laneLoudnessOpen;
        }
        return next;
      }),
    })),

  setSegmentTempoDetect: (trackId, segmentId, detect) =>
    set((s) => ({
      dirty: true,
      tracks: mapAudioContent(s.tracks, trackId, segmentId, (c) => {
        const next = { ...c };
        if (detect) {
          // canonical rounding + fixed literal key order = byte-stable serialize (§5 false-dirty)
          next.tempoDetect = {
            bpm: Math.round(detect.bpm * 1000) / 1000,
            anchorMs: Math.round(detect.anchorMs * 100) / 100,
            downbeat: Math.max(0, Math.min(MAX_DOWNBEAT, Math.round(detect.downbeat))),
            conf: Math.round(detect.conf * 1000) / 1000,
            ...(detect.notConstant ? { notConstant: true as const } : {}),
          };
        } else {
          delete next.tempoDetect;
        }
        return next;
      }),
    })),

  setVocalParams: (trackId, updates) =>
    set((s) => ({
      dirty: true,
      tracks: s.tracks.map((t) => {
        if (t.id !== trackId) return t;
        const vp = { ...(t.vocalParams ?? DEFAULT_VOCAL_PARAMS), ...updates };
        // canonical write (sig↔serialize, S48 Phase 3): rangeExtend's default (OFF since S62c —
        // extension is opt-in) is stored as ABSENCE — an explicit `false` would false-dirty the
        // close/autosave byte-compare without an undo step (vocalParamsSig folds it either way).
        if (vp.rangeExtend === false) delete vp.rangeExtend;
        // S73b autoTuneFollow 极性相反(默认=开):true 折为 ABSENCE(sanitize/sig 双侧同折,
        // 否则 关→开 往返后 serialize 带 concrete true = close/autosave 字节假脏,审查)。
        if (vp.autoTuneFollow === true) delete vp.autoTuneFollow;
        return { ...t, vocalParams: vp };
      }),
    })),

  mergeProcessedOutputs: (trackId, segmentId, outputs) =>
    set((s) => {
      // Replace by SOURCE NODE identity (not laneLabel) so two Output nodes sharing a lane name don't
      // clobber each other and a deposit replaces only the depositing node's own prior contribution
      // (parity with a full Run, which layers same-label siblings). Legacy entries with no outputNodeId
      // fall back to laneLabel matching so loading an old project then depositing leaves no duplicate.
      const ids = new Set(outputs.map((o) => o.outputNodeId));
      // Legacy (no-outputNodeId) lane replacement matches by label — BASE-aware: the single-edge stem
      // suffix means the same graph now labels a lane "Main · vocals" where a legacy save has bare
      // "Main"; matching the base too lets the deposit REPLACE the stale legacy lane instead of leaving
      // a duplicate row that double-plays. (Legacy lanes have no node identity — label matching was
      // always first-writer-wins; the base match is the same rule, suffix-tolerant.)
      // NOTE deliberately NOT copying laneControls to a renamed lane's new row key here: laneControls IS
      // in the history meaningfulSig, and this merge runs on the DEPOSIT path — writing it here pushed a
      // phantom timeline undo step and washed the redo stack (review-caught HIGH). A group rename simply
      // starts the row's mixer fresh.
      const labels = new Set(outputs.flatMap((o) => [o.laneLabel, laneGroupName(o)]));
      const legacyGone = (o: ProcessedOutput) =>
        labels.has(o.laneLabel) || labels.has(laneLabelParts(o.laneLabel).base);
      return {
        dirty: true,
        tracks: s.tracks.map((t) => {
          if (t.id !== trackId) return t;
          return {
            ...t,
            segments: t.segments.map((seg) =>
              seg.id === segmentId
                ? {
                    ...seg,
                    // Normalize to workflow (Output-node) order so lanes don't reshuffle by deposit timing.
                    processedOutputs: orderProcessedOutputs(
                      [
                        ...(seg.processedOutputs ?? []).filter((o) =>
                          o.outputNodeId !== undefined ? !ids.has(o.outputNodeId) : !legacyGone(o),
                        ),
                        ...outputs,
                      ],
                      seg.workflow,
                    ),
                  }
                : seg,
            ),
          };
        }),
      };
    }),

  applyLaneDetach: (trackId, segmentId, oldNodeId, mapping) => {
    set((s) => ({
      dirty: true,
      tracks: s.tracks.map((t) => {
        if (t.id !== trackId) return t;
        const seg = t.segments.find((sg) => sg.id === segmentId);
        if (!seg) return t;
        // Mixer inheritance: laneControls key by the 组 (the Output node id), so the ungroup copies the
        // group's entry to EACH new node (COPY — the old entry stays for the graph-undo convergence
        // path, and a split sibling may still read it). Parallel to the laneOps copy below.
        let laneControls = t.laneControls;
        const groupCtrl = t.laneControls[oldNodeId];
        if (groupCtrl) {
          for (const m of mapping) {
            if (!laneControls[m.newNodeId]) {
              laneControls = { ...laneControls, [m.newNodeId]: { ...groupCtrl } };
            }
          }
        }
        return {
          ...t,
          laneControls,
          segments: t.segments.map((sg) => {
            if (sg.id !== segmentId) return sg;
            const processedOutputs = sg.processedOutputs?.map((o) => {
              if (o.outputNodeId !== oldNodeId) return o;
              const m = mapping.find((mm) => mm.oldLaneId === o.laneId);
              return m
                ? { ...o, laneId: m.newLaneId, outputNodeId: m.newNodeId, group: m.group, laneLabel: m.laneLabel }
                : o;
            });
            // The shared group recipe is INHERITED by every detached node (deep-copied per key — laneOps
            // values must never alias across keys/segments). The old key is deliberately KEPT.
            let laneOps = sg.laneOps;
            const recipe = laneOps?.[oldNodeId];
            if (recipe) {
              const next = { ...laneOps };
              for (const m of mapping) next[m.newNodeId] = recipe.map((c) => ({ ...c }));
              laneOps = next;
            }
            return { ...sg, processedOutputs, laneOps };
          }),
        };
      }),
    }));
    // Record the detach lineage AFTER the store write (runtime-only, not history state): applySnapshot
    // re-derives the machine copies above when a timeline undo crosses this detach — see laneOps
    // detachLineage for the snapshot-seq rule that keeps post-detach deletions intentional.
    for (const m of mapping) recordDetachLineage(m.newNodeId, oldNodeId);
  },

  removeProcessedOutputsForNode: (trackId, segmentId, outputNodeId) =>
    set((s) => ({
      dirty: true,
      tracks: s.tracks.map((t) => {
        if (t.id !== trackId) return t;
        return {
          ...t,
          segments: t.segments.map((seg) =>
            seg.id === segmentId
              ? { ...seg, processedOutputs: (seg.processedOutputs ?? []).filter((o) => o.outputNodeId !== outputNodeId) }
              : seg,
          ),
        };
      }),
    })),

  clearProcessedOutputs: (trackId, segmentId) =>
    set((s) => ({
      dirty: true,
      tracks: s.tracks.map((t) => {
        if (t.id !== trackId) return t;
        return {
          ...t,
          segments: t.segments.map((seg) =>
            seg.id === segmentId ? { ...seg, processedOutputs: undefined } : seg,
          ),
        };
      }),
    })),

  replaceProcessedOutputs: (trackId, segmentId, outputs) =>
    set((s) => ({
      dirty: true,
      tracks: s.tracks.map((t) => {
        if (t.id !== trackId) return t;
        return {
          ...t,
          segments: t.segments.map((seg) =>
            seg.id === segmentId
              ? { ...seg, processedOutputs: outputs.length > 0 ? orderProcessedOutputs(outputs, seg.workflow) : undefined }
              : seg,
          ),
        };
      }),
    })),
}));

/**
 * THE single construction site for the current project's TimeAxis (S48 Phase 0) — the one place bar
 * geometry is derived from the global meter, so no component re-implements `ticksPerBar` math. Rebuilds
 * only when the meter changes (the store's `timeSignature` array is reference-stable between meter edits,
 * so the memo holds). Phase 0 = a single meter section at bar 0; when per-section meter (`timeSignatures[]`)
 * lands, ONLY this hook's construction changes — every consumer already calls position-based TimeAxis
 * methods, so they need no edit.
 */
export function useTimeAxis(): TimeAxis {
  const ts = useProjectStore((s) => s.timeSignature);
  return useMemo(() => TimeAxis.global(ts[0], ts[1]), [ts]);
}
