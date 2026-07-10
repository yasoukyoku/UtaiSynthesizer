import { TICKS_PER_BEAT } from "../constants";
import type { LaneClip, ProcessedOutput, Segment } from "../../types/project";

/**
 * Pure helpers for sub-lane non-destructive edits (P3). A lane GROUP's recipe is a list of kept
 * `LaneClip`s in STEM MILLISECONDS (see LaneClip). The design that makes everything simple:
 *
 *   - Ops live in the STEM's own coordinate (ms), so they are INVARIANT under the parent segment's
 *     move / split / resize / tempo change — those only shift the visible window into the stem.
 *   - The visible window of a segment's lane = [offsetMs, offsetMs + durationMs]; read-time we
 *     INTERSECT each clip with the window (`laneVisiblePieces`). Untrimmed outer edges are stored as
 *     0 / stemDurationMs (≥ any window) so they auto-follow the window (resize/split "just work").
 *   - A missing entry = whole lane plays (implicit); `[]` = explicitly silenced.
 *
 * One source of truth for slice / trim / delete / draw-and-playback geometry / the history signature.
 */

/** 1ms epsilon — collapse sub-millisecond slivers so a slice/trim at a piece edge is a no-op, not a
 *  zero-width phantom piece. */
const EPS_MS = 1;

/** Pieces narrower than this after a trim RELEASE are spliced out of the recipe (Arrangement onDocUp):
 *  trimClip clamps a drag to EPS_MS wide, which draws as sub-pixel and can't be re-grabbed — dragging an
 *  edge all the way across a piece reads as "remove it", so the release finishes the job. */
export const MIN_LANE_PIECE_MS = 5;

export function ticksToMs(ticks: number, tempo: number): number {
  return (ticks / TICKS_PER_BEAT) * (60000 / tempo);
}

export function msToTicks(ms: number, tempo: number): number {
  return (ms / 60000) * tempo * TICKS_PER_BEAT;
}

/** Duration-ms → ticks with the shared minimum-1-tick floor (a 0-tick segment would be invisible and
 *  un-interactable). THE one definition — import placeholders/finalize, history's loading-reconcile,
 *  and probe-based resizes all round through here so they can never disagree by a rounding rule. */
export function flooredDurationTicks(ms: number, tempo: number): number {
  return Math.max(1, Math.round(msToTicks(ms, tempo)));
}

/** The Output-node group a lane belongs to = its `outputNodeId`, or (legacy / defensive) the laneId
 *  prefix before "::" (laneId is ALWAYS `${outputNodeId}::${fromNode}:${fromPort}`). laneOps is keyed
 *  by this, so all lanes fanned into one Output node share ONE recipe ("group-operate"). */
export function laneGroupId(out: Pick<ProcessedOutput, "laneId" | "outputNodeId">): string {
  return out.outputNodeId ?? out.laneId.split("::")[0]!;
}

/** Split a display laneLabel ("Group" / "Group · stem") into its parts. THE one definition of the
 *  " · " convention (produced by engine.ts laneLabelFor) — header column, canvas row text, and the
 *  legacy group fallback all parse through here so the convention can never drift. */
export function laneLabelParts(label: string): { base: string; stem: string | null } {
  const i = label.indexOf(" · ");
  return i < 0 ? { base: label, stem: null } : { base: label.slice(0, i), stem: label.slice(i + 3) };
}

/** The lane's GROUP NAME (display): the persisted `group` field, or (legacy saves) the laneLabel base. */
export function laneGroupName(out: Pick<ProcessedOutput, "group" | "laneLabel">): string {
  return out.group ?? laneLabelParts(out.laneLabel).base;
}

/** ROW identity for track lane rows + crossfade pairing = 轨道组 name + laneId. Two split halves
 *  share a laneId (a split copies the graph, so both halves' Output nodes have the same node id);
 *  including the NAME lets a half whose Output node was renamed move to its own row (and stop
 *  crossfade-pairing across the seam) instead of being swallowed by the sibling's row. ALSO the key of
 *  `Track.laneMutes` (per-row mute — loose, resets on rename). NOT the volume/pan key — laneControls
 *  key by laneId (the node identity), so a rename never re-keys the mix.
 *  Opaque — never parsed by consumers (bundle.ts strips an interim rowKey-keyed laneControls save). */
export function laneRowKey(out: Pick<ProcessedOutput, "group" | "laneLabel" | "laneId">): string {
  return `${laneGroupName(out)}\u0000${out.laneId}`;
}

/** The rowKey separator, derived from laneRowKey itself rather than repeating the raw escape literal
 *  (typing that escape in editing tools writes a REAL control byte — a known silent-corruption trap:
 *  laneRowKey of an all-empty output is "" + SEP + "" = the bare separator). */
const ROW_KEY_SEP = laneRowKey({ group: "", laneLabel: "", laneId: "" });

/** The laneId part of a laneRowKey — THE one parser for consumers that must relate a row-mute key back
 *  to the lane identity (the serializer's legacy-mask check; keep bundle's interim-key strip in sync). */
export function rowKeyLaneId(rowKey: string): string {
  const i = rowKey.indexOf(ROW_KEY_SEP);
  return i < 0 ? rowKey : rowKey.slice(i + ROW_KEY_SEP.length);
}

/** Materialize the stored recipe into a concrete clip list. `undefined` (never edited) → the whole
 *  stem [0, stemDurMs]; the whole-stem outer bounds keep untrimmed edges window-following. `[]`
 *  (explicitly silenced) → []. Always returns fresh objects (safe to mutate by the caller). */
export function materializeClips(stored: LaneClip[] | undefined, stemDurMs: number): LaneClip[] {
  if (stored === undefined) return [{ start: 0, end: Math.max(0, stemDurMs) }];
  return stored.map((c) => ({ ...c }));
}

/** A visible piece of a lane within its segment: absolute tick span on the timeline + the stem-ms
 *  window it reads (startMs = the audio offset to play from; the [startMs,endMs] fraction of the stem
 *  peaks to draw). Produced by intersecting the recipe with the segment's visible window. */
export interface LanePiece {
  startTick: number;
  endTick: number;
  startMs: number;
  endMs: number;
}

/** The lane's kept pieces within a segment's visible window [offsetMs, offsetMs+durationMs], mapped to
 *  absolute timeline ticks. `undefined` recipe → one whole-window piece (== the pre-P3 behaviour, so an
 *  unedited lane is byte-identical). Empty result = fully silent. */
export function laneVisiblePieces(
  seg: Segment,
  stored: LaneClip[] | undefined,
  stemDurMs: number,
  tempo: number,
): LanePiece[] {
  // ② vocal render: a notes segment's baked stem starts at the segment start (stem-ms 0 = segStart) and
  // plays 1:1 — offset 0 (an audioClip's offsetMs is where its window sits inside a longer source stem).
  const offset = seg.content.type === "audioClip" ? seg.content.offsetMs : 0;
  const winStart = offset;
  const winEnd = offset + ticksToMs(seg.durationTicks, tempo);
  const clips = materializeClips(stored, stemDurMs);
  const pieces: LanePiece[] = [];
  for (const c of clips) {
    const s = Math.max(c.start, winStart);
    const e = Math.min(c.end, winEnd);
    if (e - s <= EPS_MS) continue;
    pieces.push({
      startTick: seg.startTick + msToTicks(s - offset, tempo),
      endTick: seg.startTick + msToTicks(e - offset, tempo),
      startMs: s,
      endMs: e,
    });
  }
  return pieces;
}

/** Index of the materialized clip whose stem-ms range contains `ms` (for selecting / targeting a piece
 *  by click). Falls back to the nearest clip (by gap) so a click in a silent gap still targets a piece
 *  to trim/extend. -1 only when there are no clips (fully silenced). */
export function clipIndexAtMs(stored: LaneClip[] | undefined, stemDurMs: number, ms: number): number {
  const clips = materializeClips(stored, stemDurMs);
  if (clips.length === 0) return -1;
  for (let i = 0; i < clips.length; i++) {
    if (ms >= clips[i]!.start && ms <= clips[i]!.end) return i;
  }
  let best = 0;
  let bestGap = Infinity;
  for (let i = 0; i < clips.length; i++) {
    const gap = ms < clips[i]!.start ? clips[i]!.start - ms : ms - clips[i]!.end;
    if (gap < bestGap) { bestGap = gap; best = i; }
  }
  return best;
}

/** Slice the recipe at stem-ms `cutMs`: split the clip that contains the cut into two adjacent pieces.
 *  A cut at/near a piece edge or in a gap is a no-op. Returns the new recipe (materialized). */
export function sliceClips(stored: LaneClip[] | undefined, stemDurMs: number, cutMs: number): LaneClip[] {
  const clips = materializeClips(stored, stemDurMs);
  const out: LaneClip[] = [];
  for (const c of clips) {
    if (cutMs > c.start + EPS_MS && cutMs < c.end - EPS_MS) {
      out.push({ start: c.start, end: cutMs }, { start: cutMs, end: c.end });
    } else {
      out.push(c);
    }
  }
  return out;
}

/** Trim/stretch one edge of clip `index` to stem-ms `newMs`, clamped to the segment's visible window
 *  [winStartMs, winEndMs] and to the neighbouring clips (no overlap; a gap between neighbours = silence).
 *  When an OUTERMOST edge is stretched back to the window boundary it snaps to the stem bound (0 /
 *  stemDurMs) so it resumes following the window (un-trim restores resize-follows). */
export function trimClip(
  stored: LaneClip[] | undefined,
  stemDurMs: number,
  index: number,
  edge: "start" | "end",
  newMs: number,
  winStartMs: number,
  winEndMs: number,
): LaneClip[] {
  const clips = materializeClips(stored, stemDurMs);
  const c = clips[index];
  if (!c) return clips;
  if (edge === "start") {
    const lo = index > 0 ? clips[index - 1]!.end : winStartMs;
    let v = Math.min(Math.max(newMs, lo), c.end - EPS_MS);
    if (index === 0 && v <= winStartMs + EPS_MS) v = 0; // un-trimmed front → follow the window again
    c.start = v;
  } else {
    const hi = index < clips.length - 1 ? clips[index + 1]!.start : winEndMs;
    let v = Math.max(Math.min(newMs, hi), c.start + EPS_MS);
    if (index === clips.length - 1 && v >= winEndMs - EPS_MS) v = stemDurMs; // un-trimmed end → follow window
    c.end = v;
  }
  return clips;
}

/** Delete clip `index` → a silent gap there. Deleting the only clip yields `[]` (whole lane silent). */
export function deleteClip(stored: LaneClip[] | undefined, stemDurMs: number, index: number): LaneClip[] {
  const clips = materializeClips(stored, stemDurMs);
  if (index >= 0 && index < clips.length) clips.splice(index, 1);
  return clips;
}

/** True iff the recipe is equivalent to "whole lane plays" — one clip spanning the full stem. Lets the
 *  gesture code drop a redundant entry (store `undefined`) so an un-edited lane keeps a tiny history sig
 *  and stays window-following. */
export function isWholeClips(clips: LaneClip[], stemDurMs: number): boolean {
  return clips.length === 1 && clips[0]!.start <= EPS_MS && clips[0]!.end >= stemDurMs - EPS_MS;
}

/** Does `seg` carry a ready lane on this ROW whose KEPT audio actually reaches the given segment edge?
 *  THE crossfade gate, shared by playback and the canvas crossfade marks: a neighbour that merely HAS
 *  the row but whose piece was trimmed away from the seam would fade against silence (a one-sided
 *  volume dip into nothing). An unedited lane's single whole-window piece reaches both edges, so this
 *  is identical to the old presence-only check for the common case. */
export function laneReachesSeam(
  seg: Segment,
  rowKey: string,
  tempo: number,
  edge: "start" | "end",
): boolean {
  if (seg.content.type !== "audioClip") return false;
  const out = seg.processedOutputs?.find((o) => laneRowKey(o) === rowKey && !o.loading);
  if (!out) return false;
  const pieces = laneVisiblePieces(seg, seg.laneOps?.[laneGroupId(out)], seg.content.totalDurationMs, tempo);
  // |delta| < 1, matching playback's own-side atSegStart/atSegEnd gates EXACTLY — a looser bound here
  // would draw a crossfade mark (or fade one side) for a piece the other gate rejects (one-ramp fade).
  if (edge === "start") return pieces.some((p) => Math.abs(p.startTick - seg.startTick) < 1);
  return pieces.some((p) => Math.abs(seg.startTick + seg.durationTicks - p.endTick) < 1);
}

// ---------------------------------------------------------------------------
// UNGROUP (解组) lineage — runtime-only. applyLaneDetach copies the shared 组's laneOps recipe +
// laneControls mixer entry onto each NEW single-edge Output node. Those copies are sig-visible user
// state, so a timeline undo ACROSS the detach restores a snapshot that predates them — while the
// deposited lanes (a history overlay) keep the post-detach node ids. The lineage (new node id → the
// node it was detached from) lets applySnapshot re-derive the machine copies against the RESTORED
// state, so detached rows keep their recipe/mix instead of transiently playing the full stem at the
// default mix until redo.
//
// Each entry carries the SNAPSHOT SEQUENCE current when the detach landed (history stamps every
// snapshot with a monotonic seq at creation): a key missing from a snapshot NEWER than the detach is
// an INTENTIONAL deletion (e.g. an un-trim stored `undefined`), not pre-detach state — re-copying the
// ancestor recipe there would make the redo of that un-trim silently no-op forever. Only snapshots at
// or before the detach's seq get the re-derivation.
//
// Session-scoped by design: the undo stacks reset on project load/new, so a mapping never needs to
// survive a reload (history.ts reset() clears it). See history.applySnapshot.
// ---------------------------------------------------------------------------
const detachLineage = new Map<string, { oldId: string; seq: number }>();
let snapshotSeq = 0;

/** Monotonic snapshot sequence — history.ts stamps every Snapshot with this at creation time. */
export function nextSnapshotSeq(): number {
  return ++snapshotSeq;
}

/** Record `newNodeId` as detached from `oldNodeId`. MUST be called AFTER the detach's store write (its
 *  auto-captured pre-detach snapshot must receive a seq ≤ this entry's, so it still reconciles). */
export function recordDetachLineage(newNodeId: string, oldNodeId: string) {
  detachLineage.set(newNodeId, { oldId: oldNodeId, seq: snapshotSeq });
}

export function clearDetachLineage() {
  detachLineage.clear();
}

/** Nearest detach ANCESTOR of `nodeId` for which `has` is true, walking chained detaches
 *  (cycle-guarded). Returns null when the node has no lineage, when no ancestor satisfies `has`, or
 *  when the snapshot POST-dates a link in the chain (its missing key is an intentional deletion). */
export function resolveDetachAncestor(
  nodeId: string,
  snapSeq: number,
  has: (id: string) => boolean,
): string | null {
  const seen = new Set<string>([nodeId]);
  let entry = detachLineage.get(nodeId);
  while (entry && !seen.has(entry.oldId)) {
    if (snapSeq > entry.seq) return null;
    if (has(entry.oldId)) return entry.oldId;
    seen.add(entry.oldId);
    entry = detachLineage.get(entry.oldId);
  }
  return null;
}

/** Deterministic signature of a segment's laneOps for the history meaningfulSig. Sorted by group id so
 *  key order never spuriously creates an undo step; ms rounded to defeat float jitter. */
export function laneOpsSig(laneOps: Record<string, LaneClip[]> | undefined): string {
  if (!laneOps) return "";
  return Object.keys(laneOps)
    .sort()
    .map((g) => `${g}:${laneOps[g]!.map((c) => `${Math.round(c.start)}-${Math.round(c.end)}`).join(",")}`)
    .join("|");
}
