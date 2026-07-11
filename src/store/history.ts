import { create } from "zustand";
import { useProjectStore } from "./project";
import { useAppStore } from "./app";
import { useAudioStore } from "./audio";
import { useWorkflowStore } from "./workflow";
import { logToBackend } from "../lib/log";

/** TEMP diagnostic probes for the reported "undo banner shows but nothing reverts" (timeline-stack
 *  disorder after output/sub-lane ops). Logs land in the dev stdout as `[UI] [undoDbg] …` — grep the
 *  dev output. REMOVE once the phantom source is pinned. */
const UNDO_DBG = true;
function dbg(msg: string) {
  if (UNDO_DBG) logToBackend("info", `[undoDbg] ${msg}`);
}
import i18n from "../i18n";
import {
  clearDetachLineage,
  flooredDurationTicks,
  laneGroupId,
  laneOpsSig,
  nextSnapshotSeq,
  resolveDetachAncestor,
} from "../lib/audio/laneOps";
import type { Track, Segment, SegmentContent, LaneControl, Note, PitchCurve, VocalTrackParams } from "../types/project";

// ── HMR safety (DEV ONLY — a no-op in the production build) ──
// This module owns the undo state machine as module-level mutable state (past/future/txnDepth/lastSig below).
// A hot-swap spawns a SECOND module instance while ref-captured beginTransaction/commitTransaction closures (the
// VolumeFader gesture brackets) + the installHistory subscription still point at the OLD one — a split-brain
// where undo()/auto-capture read a stale txnDepth/past and silently NO-OP (it "heals" only on the next full
// reload). Self-accept + full reload so any edit here re-inits every consumer against ONE state machine.
if (import.meta.hot) import.meta.hot.accept(() => location.reload());

/**
 * Global (timeline) undo/redo for the audio-arrangement document.
 *
 * Model: SNAPSHOT-based with explicit gesture TRANSACTIONS. Every project mutation already replaces
 * `tracks[]` immutably (structural sharing), so a snapshot is just a reference to the prior
 * `{tracks, tempo, timeSignature}` — nearly free in memory. A linear `past`/`future` stack gives
 * Word-style semantics (a new edit after an undo discards the redo branch).
 *
 * What is undoable (the "recipe"): track add/remove/reorder, segment move/resize/split/delete,
 * track volume/pan/mute/solo, rename, lane volume/pan/mute, tempo (+ its segment-duration rescale),
 * timeSignature, segment geometry/content, and sub-lane slice/edge-stretch edits (segment.laneOps —
 * NOT the baked render they edit, which stays an overlay).
 *
 * What is deliberately EXCLUDED and treated as a non-history OVERLAY (never rolled back by timeline
 * undo): `track.expanded` and `segment.loading` (view/runtime), and — per the agreed "a render is a
 * commit" rule — `segment.processedOutputs` (the baked audio-track render result) and
 * `segment.workflow` (the per-segment node graph, which has its own modal-local undo). These ride
 * along at their CURRENT value across every undo/redo: they are stripped from the meaningful diff so
 * they never CREATE a step, and re-merged from the live store on restore so they are never reverted.
 * (NOTE: this covers ONLY audio-track node-workflow rendering. Vocal-track rendering will be a
 * separate, more complex rule set added later — design hook: extend the overlay/strip sets below.)
 *
 * Continuous gestures (drag move/resize, faders, reorder, tempo typing) are coalesced into ONE step
 * via beginTransaction()/commitTransaction(): the pre-gesture state is the previous state, the
 * on-release state is the next — never one step per frame.
 */

interface SegSel {
  trackId: string;
  segmentId: string;
}

interface Selection {
  selectedSegment: SegSel | null;
  selectedSegments: SegSel[];
  activeTrackId: string | null;
}

interface Snapshot {
  tracks: Track[]; // reference (immutable); overlay fields re-merged on apply
  tempo: number;
  timeSignature: [number, number];
  selection: Selection;
  /** Monotonic creation stamp (laneOps nextSnapshotSeq) — orders this snapshot's CONTENT relative to
   *  each recorded detach, so applySnapshot only re-derives detach machine-copies into snapshots that
   *  PREDATE the detach (a newer snapshot's missing key is an intentional deletion). */
  seq: number;
}


const MAX_DEPTH = 200;

// --- module-level history state (kept OUT of the zustand store so a push doesn't re-render the UI;
//     only the canUndo/canRedo booleans live in the store for menu enablement) ---
let past: Snapshot[] = [];
let future: Snapshot[] = [];
/** True while WE are writing the store from undo/redo — suppresses auto-capture re-entrancy. */
let applying = false;
/** Depth of the active gesture transaction; auto-capture is suppressed while > 0. */
let txnDepth = 0;
let txnBefore: Snapshot | null = null;
let txnSigBefore = "";
/** Meaningful signature of the last COMMITTED state — compared against to detect real changes. */
let lastSig = "";
/** Meaningful signature at the last save (markSaved); null = never saved this session. */
let savedSig: string | null = null;
let unsubscribe: (() => void) | null = null;

// ---------------------------------------------------------------------------
// Meaningful signature — a projection of the document that EXCLUDES the overlay
// fields (expanded / loading / processedOutputs / workflow / waveformPeaks) but INCLUDES segment.laneOps
// (the sub-lane slice/edge-stretch recipe is an arrangement edit, so it IS undoable even though the
// processedOutputs it edits are not). Two states with the same signature are "the same edit" for undo.
// ---------------------------------------------------------------------------
/** Deterministic signature of a pitch/param curve (ordered parallel arrays). */
function curveSig(c?: PitchCurve): string {
  return c ? `${c.xs.join(",")}:${c.ys.join(",")}` : "";
}

/** Deterministic signature of one vocal note: the 7 base fields + the optional pitch/expression edits.
 *  Every optional folds to its DEFAULT so "absent" reads identical to "default-valued" — the store omits
 *  defaults, so a note that never gained a `detune` and one whose `detune` returned to 0 are the same edit
 *  (no phantom undo step / no false-dirty). Order is fixed; arrays serialize in element order. */
function noteSig(n: Note): string {
  const t = n.transition;
  const tr = t ? `${t.offsetMs ?? ""}|${t.durLeftMs ?? ""}|${t.durRightMs ?? ""}|${t.depthLeftCents ?? ""}|${t.depthRightCents ?? ""}|${t.openEdgeCents ?? ""}` : "";
  const v = n.vibrato;
  const vib = v ? `${v.depthCents},${v.freqHz},${v.phase},${v.startMs},${v.easeInMs},${v.easeOutMs}` : "";
  return (
    `${n.id}.${n.tick}.${n.duration}.${n.pitch}.${n.lyric}.${n.phoneme ?? ""}.${n.velocity}` +
    `.${n.detune ?? 0}.${n.tie ? 1 : 0}.${n.pitchAuto === false ? 0 : 1}.${n.lang ?? ""}.${n.phonemeInput ?? ""}` +
    `.${tr}.${vib}`
  );
}

/** Sorted-key sig of a paramCurves bag (shared by both content variants). */
function paramCurvesSig(pc?: Record<string, PitchCurve>): string {
  if (!pc) return "";
  return Object.keys(pc).sort().map((k) => `${k}=${curveSig(pc[k])}`).join("&");
}

export function contentSig(c: SegmentContent): string {
  if (c.type === "audioClip") {
    // stretch folds to 1 and tempoDetect to "" so untouched clips keep their pre-S59 identity
    // (no phantom undo step / false-dirty on old projects).
    const td = c.tempoDetect;
    const tdSig = td ? `${td.bpm},${td.anchorMs},${td.downbeat},${td.conf},${td.notConstant ? 1 : 0}` : "";
    return `a:${c.sourcePath}:${c.offsetMs}:${c.totalDurationMs}:${c.stretch ?? 1}:${tdSig}:${paramCurvesSig(c.paramCurves)}`;
  }
  const notes = c.notes.map(noteSig).join("|");
  const dev = curveSig(c.pitchDev);
  // paramCurves is a Record — key order is not guaranteed, so SORT (like laneSig) for a stable signature.
  const params = paramCurvesSig(c.paramCurves);
  return `n:${notes}#${dev}#${params}`;
}

/** Deterministic signature of a track's vocal params (undoable, like voiceModel). */
/** Canonical (sorted-key) sig of a quality-param override bag so an undo/redo of a knob is caught and a
 *  re-serialized-but-equal bag never reads dirty (§Phase-3 假脏 discipline). */
function sigOpts(o?: Record<string, unknown>): string {
  if (!o) return "";
  return Object.keys(o)
    .sort()
    .map((k) => `${k}=${JSON.stringify(o[k])}`)
    .join(",");
}
export function vocalParamsSig(p?: VocalTrackParams): string {
  if (!p) return "";
  const t = p.transition;
  const tr = t ? `${t.offsetMs},${t.durLeftMs},${t.durRightMs},${t.depthLeftCents},${t.depthRightCents},${t.openEdgeCents}` : "";
  return `${p.backend},${p.speakerId},${p.langId},${p.transpose},${p.formant ?? 0},${tr}|sv:${sigOpts(p.sovits as Record<string, unknown> | undefined)}|rv:${sigOpts(p.rvc as Record<string, unknown> | undefined)}|bt:${p.breathToken ?? ""}`;
}

function laneSig(lc: Record<string, LaneControl>, mutes?: Record<string, boolean>): string {
  const controls = Object.keys(lc)
    .sort()
    .map((k) => {
      const v = lc[k]!;
      return `${k}=${v.volumeDb},${v.pan},${v.muted ? 1 : 0}`;
    })
    .join("|");
  const muteSig = mutes
    ? Object.keys(mutes).sort().map((k) => `${k}${mutes[k] ? 1 : 0}`).join("|")
    : "";
  return `${controls}/${muteSig}`;
}

function meaningfulSig(tracks: Track[], tempo: number, timeSig: [number, number]): string {
  return (
    `${tempo}|${timeSig[0]}/${timeSig[1]}|` +
    tracks
      .map(
        (t) =>
          `${t.id}~${t.name}~${t.trackType}~${t.volumeDb}~${t.pan}~${t.muted ? 1 : 0}~${t.solo ? 1 : 0}~` +
          `${t.playOriginal ? 1 : 0}~` +
          `${t.voiceModel ?? ""}~${t.voiceModelAvatar ?? ""}~${vocalParamsSig(t.vocalParams)}~${laneSig(t.laneControls, t.laneMutes)}~` +
          t.segments
            .map((s) => `${s.id}.${s.startTick}.${s.durationTicks}.${contentSig(s.content)}.${laneOpsSig(s.laneOps)}`)
            .join(";"),
      )
      .join("||")
  );
}

function currentSig(): string {
  const p = useProjectStore.getState();
  return meaningfulSig(p.tracks, p.tempo, p.timeSignature);
}

function currentSelection(): Selection {
  const a = useAppStore.getState();
  return {
    selectedSegment: a.selectedSegment,
    selectedSegments: a.selectedSegments,
    activeTrackId: a.activeTrackId,
  };
}

function snapshotCurrent(): Snapshot {
  const p = useProjectStore.getState();
  return {
    tracks: p.tracks,
    tempo: p.tempo,
    timeSignature: p.timeSignature,
    selection: currentSelection(),
    seq: nextSnapshotSeq(),
  };
}

// ---------------------------------------------------------------------------
// Apply a snapshot back into the stores. Re-merges the overlay fields from the LIVE store (so a
// render / expand-state done after the snapshot is not rolled back), reconciles the restored
// selection against the merged tracks, recomputes dirty vs the saved point, reschedules playback,
// and reveals the changed region.
// ---------------------------------------------------------------------------
function applySnapshot(snap: Snapshot) {
  const liveTracks = useProjectStore.getState().tracks;
  const liveTrackById = new Map(liveTracks.map((t) => [t.id, t]));
  const liveSegById = new Map<string, Segment>();
  for (const t of liveTracks) for (const s of t.segments) liveSegById.set(s.id, s);
  const audioFiles = useAudioStore.getState().audioFiles;

  // Render sources a RESTORED mid-render split half can re-link to: live segments that own a render
  // (running, or settled with a warm cache). Node-id overlap is THE split signature — a split copies
  // the graph, so a half's loading lanes carry outputNodeIds present in the source's workflow.
  const wfState = useWorkflowStore.getState();
  const renderSources: { id: string; nodeIds: Set<string> }[] = [];
  for (const [id, exec] of Object.entries(wfState.executions)) {
    const src = liveSegById.get(id);
    if (!src?.workflow) continue;
    const warm = wfState.nodeOutputs[id] && Object.keys(wfState.nodeOutputs[id]!).length > 0;
    if (exec.status !== "running" && !warm) continue;
    renderSources.push({ id, nodeIds: new Set(src.workflow.nodes.map((n) => n.id)) });
  }
  const relinks: { toId: string; fromId: string }[] = [];
  // Snapshot-fallback lanes (segment not live): a restored segment with mid-render LOADING placeholders
  // is a redone split half — if a live split SIBLING still owns the render, RE-LINK it instead of
  // stripping: RenderLinkWatcher then finishes it exactly like an original split half (tracks the run,
  // or — already settled — immediately clones the cache and headless-deposits). With no such source the
  // placeholders can never resolve, so they are stripped (a stuck spinner drives a permanent rAF).
  const fallbackOutputs = (s: Segment) => {
    const outs = s.processedOutputs;
    if (!outs?.some((o) => o.loading)) return outs;
    const src = renderSources.find(
      (r) => r.id !== s.id && outs.some((o) => o.loading && o.outputNodeId && r.nodeIds.has(o.outputNodeId)),
    );
    if (src) {
      relinks.push({ toId: s.id, fromId: src.id });
      return outs;
    }
    return outs.filter((o) => !o.loading);
  };

  const merged: Track[] = snap.tracks.map((t) => {
    const lt = liveTrackById.get(t.id);
    return {
      ...t,
      // expanded is a view overlay — keep the current value rather than the snapshot's.
      expanded: lt ? lt.expanded : t.expanded,
      // S59 loudness-lane band open state — same view-overlay treatment as expanded.
      loudnessLaneOpen: lt ? lt.loudnessLaneOpen : t.loudnessLaneOpen,
      segments: t.segments.map((s) => {
        const ls = liveSegById.get(s.id);
        // Overlay the runtime/render fields from the LIVE segment if it still exists, else fall back
        // to the snapshot's own (which is what makes undoing the DELETE of a rendered segment restore
        // its lanes).
        const seg = {
          ...s,
          loading: ls ? ls.loading : s.loading,
          // Snapshot fallback (segment not live): loading placeholders either RE-LINK to a live render
          // source (redone mid-render split half — see fallbackOutputs above) or are stripped. A live
          // segment keeps its current overlay (a genuine in-flight deposit). Non-loading lanes (a
          // fully-rendered restored segment) are untouched, so undoing a delete still restores real lanes.
          processedOutputs: ls ? ls.processedOutputs : fallbackOutputs(s),
          workflow: ls ? ls.workflow : s.workflow,
        } as Segment;
        // Reconcile a restored LOADING placeholder against the decode cache: if its source is already
        // decoded (e.g. redo of an import whose decode finished during the undo window), finalize it
        // here — otherwise it comes back as a permanently-stuck, non-interactive striped block (the
        // one-shot finalizeSegment already no-op'd). If not yet decoded, the in-flight finalizeSegment
        // (keyed by the same ids) completes it.
        if (seg.loading && seg.content.type === "audioClip") {
          const af = audioFiles[seg.content.sourcePath];
          if (af) {
            return {
              ...seg,
              loading: false,
              // S59: a stretched clip's box is source-duration × r — reconciling without the
              // factor would silently snap a stretched segment back to 1:1 length (the recon's
              // top silent-regression risk for the stretch feature).
              durationTicks: flooredDurationTicks(af.durationMs * (seg.content.stretch ?? 1), snap.tempo),
              content: { ...seg.content, totalDurationMs: af.durationMs },
            } as Segment;
          }
        }
        return seg;
      }),
    };
  });

  // UNGROUP (解组) machine-copy reconciliation: the deposited lanes (an overlay, kept LIVE above) may
  // reference post-detach Output-node ids that this snapshot PREDATES — the laneOps/laneControls copies
  // applyLaneDetach made for them are sig-visible state the snapshot lacks, so without this pass an undo
  // across the detach point transiently plays the detached rows as the full stem at the default mix
  // until redo. Re-derive each MISSING key from its nearest restored detach ancestor (same semantics as
  // applyLaneDetach: copy only what the ancestor actually has). Copy-on-write — `merged` objects are
  // fresh, but their laneOps/laneControls still reference the snapshot's objects.
  for (let ti = 0; ti < merged.length; ti++) {
    const t = merged[ti]!;
    let laneControls = t.laneControls;
    let lcChanged = false;
    let segsChanged = false;
    const segments = t.segments.map((seg) => {
      let laneOps = seg.laneOps;
      let opsChanged = false;
      for (const o of seg.processedOutputs ?? []) {
        const gid = laneGroupId(o);
        if (laneOps?.[gid] === undefined) {
          const anc = resolveDetachAncestor(gid, snap.seq, (id) => laneOps?.[id] !== undefined);
          if (anc) {
            laneOps = { ...(laneOps ?? {}), [gid]: laneOps![anc]!.map((c) => ({ ...c })) };
            opsChanged = true;
          }
        }
        if (laneControls[gid] === undefined) {
          const anc = resolveDetachAncestor(gid, snap.seq, (id) => laneControls[id] !== undefined);
          if (anc) {
            laneControls = { ...laneControls, [gid]: { ...laneControls[anc]! } };
            lcChanged = true;
          }
        }
      }
      if (!opsChanged) return seg;
      segsChanged = true;
      return { ...seg, laneOps };
    });
    if (lcChanged || segsChanged) merged[ti] = { ...t, laneControls, segments };
  }

  const sig = meaningfulSig(merged, snap.tempo, snap.timeSignature);

  // Reconcile selection: drop ids that no longer exist after the restore.
  const segKeys = new Set<string>();
  const trackIds = new Set<string>();
  for (const t of merged) {
    trackIds.add(t.id);
    for (const s of t.segments) segKeys.add(`${t.id}:${s.id}`);
  }
  const validSegs = snap.selection.selectedSegments.filter((x) => segKeys.has(`${x.trackId}:${x.segmentId}`));
  const primary =
    snap.selection.selectedSegment && segKeys.has(`${snap.selection.selectedSegment.trackId}:${snap.selection.selectedSegment.segmentId}`)
      ? snap.selection.selectedSegment
      : validSegs[validSegs.length - 1] ?? null;

  // Reconcile the vocal NOTE selection (§9.5): note ids are globally-unique UUIDs (segment-independent),
  // so an undo/redo that deleted notes must drop their now-dangling ids from projectStore.selectedNotes —
  // else the highlight lingers and a subsequent Delete/nudge acts on ghosts. Only write when it shrank
  // (a new array ref during applying is inert to installHistory, but avoid needless churn).
  const liveNoteIds = new Set<string>();
  for (const t of merged) for (const s of t.segments) {
    if (s.content.type === "notes") for (const nn of s.content.notes) liveNoteIds.add(nn.id);
  }
  const prevSelectedNotes = useProjectStore.getState().selectedNotes;
  const nextSelectedNotes = prevSelectedNotes.filter((id) => liveNoteIds.has(id));

  applying = true;
  useProjectStore.setState({
    tracks: merged,
    tempo: snap.tempo,
    timeSignature: snap.timeSignature,
    dirty: savedSig === null ? true : sig !== savedSig,
    ...(nextSelectedNotes.length !== prevSelectedNotes.length ? { selectedNotes: nextSelectedNotes } : {}),
  });
  useAppStore.setState({
    selectedSegments: validSegs,
    selectedSegment: primary,
    activeTrackId:
      snap.selection.activeTrackId && trackIds.has(snap.selection.activeTrackId)
        ? snap.selection.activeTrackId
        : useAppStore.getState().activeTrackId,
  });
  applying = false;

  // Re-link redone mid-render split halves AFTER the tracks landed, so RenderLinkWatcher's effect
  // (subscribed to renderLinks + tracks) sees the restored segment when it fires.
  for (const l of relinks) useWorkflowStore.getState().linkRender(l.toId, l.fromId);

  lastSig = sig;

  // A committed edit changed segment timing → reschedule the Web Audio graph if playing.
  if (useAudioStore.getState().isPlaying) useAudioStore.getState().bumpSchedule();
}

/**
 * Describe (as an i18n key under "history.") the single operation that transforms `from`→`to`. Used
 * to tell the user WHAT was undone/redone via a transient banner — instead of yanking the viewport to
 * the change (jarring; replaced per user feedback). Returns the most salient category; an op only ever
 * changes one of these, so first-match is fine.
 */
function describeDelta(from: Snapshot, to: Snapshot): string {
  if (from.tempo !== to.tempo) return "tempo";
  if (from.timeSignature[0] !== to.timeSignature[0] || from.timeSignature[1] !== to.timeSignature[1]) return "timeSignature";

  const fromById = new Map(from.tracks.map((t) => [t.id, t]));
  const toById = new Map(to.tracks.map((t) => [t.id, t]));
  if (to.tracks.some((t) => !fromById.has(t.id))) return "addedTrack";
  if (from.tracks.some((t) => !toById.has(t.id))) return "removedTrack";
  if (from.tracks.length === to.tracks.length && from.tracks.some((t, i) => to.tracks[i]?.id !== t.id)) return "reorderedTrack";

  for (const ft of from.tracks) {
    const tt = toById.get(ft.id);
    if (!tt) continue;
    if (ft.name !== tt.name) return "renamedTrack";
    if (ft.muted !== tt.muted) return "mute";
    if (ft.solo !== tt.solo) return "solo";
    if ((ft.playOriginal ?? false) !== (tt.playOriginal ?? false)) return "playOriginal";
    if (ft.volumeDb !== tt.volumeDb) return "volume";
    if (ft.pan !== tt.pan) return "pan";
    const laneKeys = new Set([...Object.keys(ft.laneControls), ...Object.keys(tt.laneControls)]);
    for (const lk of laneKeys) {
      const a = ft.laneControls[lk];
      const b = tt.laneControls[lk];
      if ((a?.muted ?? false) !== (b?.muted ?? false)) return "laneMute";
      if ((a?.volumeDb ?? 0) !== (b?.volumeDb ?? 0)) return "laneVolume";
      if ((a?.pan ?? 0) !== (b?.pan ?? 0)) return "lanePan";
    }
    const muteKeys = new Set([...Object.keys(ft.laneMutes ?? {}), ...Object.keys(tt.laneMutes ?? {})]);
    for (const mk of muteKeys) {
      if ((ft.laneMutes?.[mk] ?? false) !== (tt.laneMutes?.[mk] ?? false)) return "laneMute";
    }
    if (vocalParamsSig(ft.vocalParams) !== vocalParamsSig(tt.vocalParams)) return "vocalParams"; // ② vocal
  }

  // Segment-level, GLOBAL (so a cross-track move reads as a move, not a delete+add).
  const fG = new Map<string, { trackId: string; startTick: number; durationTicks: number }>();
  const tG = new Map<string, { trackId: string; startTick: number; durationTicks: number }>();
  for (const t of from.tracks) for (const s of t.segments) fG.set(s.id, { trackId: t.id, startTick: s.startTick, durationTicks: s.durationTicks });
  for (const t of to.tracks) for (const s of t.segments) tG.set(s.id, { trackId: t.id, startTick: s.startTick, durationTicks: s.durationTicks });
  const added = [...tG.keys()].filter((id) => !fG.has(id)).length;
  const removed = [...fG.keys()].filter((id) => !tG.has(id)).length;
  let moved = false;
  let resized = false;
  for (const [id, f] of fG) {
    const tg = tG.get(id);
    if (!tg) continue;
    if (tg.trackId !== f.trackId || tg.startTick !== f.startTick) moved = true;
    if (tg.durationTicks !== f.durationTicks) resized = true;
  }
  if (added && !removed) return resized ? "splitClip" : "addedClip"; // split = a clip added + a sibling shortened
  if (removed) return "deletedClip";
  if (moved) return "movedClip";
  if (resized) return "resizedClip";

  // ② Vocal-note content edit (editor): same segment geometry, different notes/pitch/param curves.
  for (const ft of from.tracks) {
    const tt = toById.get(ft.id);
    if (!tt) continue;
    for (const s of ft.segments) {
      if (s.content.type !== "notes") continue;
      const ts = tt.segments.find((x) => x.id === s.id);
      if (ts && ts.content.type === "notes" && contentSig(s.content) !== contentSig(ts.content)) return "notes";
    }
  }
  return "change";
}

/** Show the transient "Undone/Redone · <what>" banner for an op that transforms opFrom→opTo. */
function announce(opFrom: Snapshot, opTo: Snapshot, kind: "undo" | "redo") {
  const key = describeDelta(opFrom, opTo);
  const verb = i18n.t(kind === "undo" ? "history.undone" : "history.redone");
  useAppStore.getState().showBanner(`${verb} · ${i18n.t(`history.${key}`)}`, kind);
}

function pushPast(snap: Snapshot) {
  past.push(snap);
  if (past.length > MAX_DEPTH) past.shift();
}

interface HistoryState {
  canUndo: boolean;
  canRedo: boolean;
  undo: () => void;
  redo: () => void;
  /** Open a coalescing transaction (gesture start). Nestable; only the outermost commit records. */
  beginTransaction: () => void;
  /** Close the active transaction; records ONE step iff the document meaningfully changed. */
  commitTransaction: () => void;
  /** Close the active transaction without recording (does not revert state). */
  cancelTransaction: () => void;
  /** Run a mutation that must NOT create an undo step (e.g. async import finalize), updating the
   *  baseline so a later real edit still captures the right "before". */
  runSilent: <T>(fn: () => T) => T;
  /** Clear history (call when a project is opened / created). */
  reset: () => void;
  /** Mark the current state as the saved baseline (call after a successful save). */
  markSaved: () => void;
}

function syncFlags() {
  useHistoryStore.setState({ canUndo: past.length > 0, canRedo: future.length > 0 });
}

export const useHistoryStore = create<HistoryState>(() => ({
  canUndo: false,
  canRedo: false,

  undo: () => {
    // Never run mid-gesture (a mouse-held drag focuses no input, so Ctrl+Z isn't otherwise blocked)
    // or re-entrantly during an apply — both would corrupt the past/future/lastSig state machine.
    if (applying || txnDepth > 0) return;
    if (past.length === 0) return;
    const before = past.pop()!;
    const cur = snapshotCurrent();
    if (UNDO_DBG) {
      const beforeSig = meaningfulSig(before.tracks, before.tempo, before.timeSignature);
      const curSig = currentSig();
      dbg(`undo pop depth=${past.length} delta=${describeDelta(before, cur)}${beforeSig === curSig ? " *** PHANTOM (popped sig == current sig — applying changes nothing) ***" : ""}`);
    }
    future.push(cur);
    applySnapshot(before);
    if (UNDO_DBG) dbg(`undo applied — sig ${currentSig() === meaningfulSig(cur.tracks, cur.tempo, cur.timeSignature) ? "UNCHANGED vs pre-undo (no visible revert!)" : "changed (reverted ok)"}`);
    syncFlags();
    announce(before, cur, "undo"); // the undone op transformed before→cur
  },

  redo: () => {
    if (applying || txnDepth > 0) return;
    if (future.length === 0) return;
    const after = future.pop()!;
    const cur = snapshotCurrent();
    if (UNDO_DBG) {
      const afterSig = meaningfulSig(after.tracks, after.tempo, after.timeSignature);
      dbg(`redo pop depth=${future.length} delta=${describeDelta(cur, after)}${afterSig === currentSig() ? " *** PHANTOM ***" : ""}`);
    }
    past.push(cur);
    applySnapshot(after);
    syncFlags();
    announce(cur, after, "redo"); // the redone op transforms cur→after
  },

  beginTransaction: () => {
    if (applying) return;
    if (txnDepth === 0) {
      txnBefore = snapshotCurrent();
      txnSigBefore = currentSig();
    }
    txnDepth++;
  },

  commitTransaction: () => {
    if (txnDepth === 0) return;
    txnDepth--;
    if (txnDepth > 0) return;
    const before = txnBefore;
    txnBefore = null;
    if (!before) return;
    const sig = currentSig();
    if (sig === txnSigBefore) return; // gesture made no real change (a click, or returned to start)
    if (UNDO_DBG) dbg(`txn-commit depth=${past.length + 1} delta=${describeDelta(before, snapshotCurrent())}`);
    pushPast(before);
    future = [];
    lastSig = sig;
    syncFlags();
  },

  cancelTransaction: () => {
    if (txnDepth === 0) return;
    txnDepth--;
    if (txnDepth > 0) return;
    txnBefore = null;
  },

  runSilent: (fn) => {
    const prev = applying;
    applying = true;
    try {
      return fn();
    } finally {
      applying = prev;
      lastSig = currentSig();
    }
  },

  reset: () => {
    past = [];
    future = [];
    txnDepth = 0;
    txnBefore = null;
    lastSig = currentSig();
    savedSig = null;
    // Detach lineage exists only to serve undo-across-detach — no stacks, no lineage needed.
    clearDetachLineage();
    syncFlags();
  },

  markSaved: () => {
    savedSig = currentSig();
  },
}));

// --- Undo SCOPE routing (FOCUS-based) --------------------------------------
// The per-segment workflow editor registers its modal-local node-graph undo here while mounted. But
// it is now a PERSISTENT bottom panel co-visible with the track timeline, so being "registered" is no
// longer enough to own Ctrl+Z — that would starve the timeline of undo whenever the panel is open.
// Routing is FOCUS-based: the editor's stack wins ONLY when the workflow pane is the active pane
// (app.activePane === "workflow"); otherwise Ctrl+Z/Y act on the timeline. (A render inside the editor
// is still a commit barrier — it clears its own stack on render, see WorkflowEditor.)
interface UndoScope {
  undo: () => void;
  redo: () => void;
  canUndo: () => boolean;
  canRedo: () => boolean;
}
let scopedHandler: UndoScope | null = null;

export function setUndoScope(h: UndoScope | null) {
  scopedHandler = h;
  if (UNDO_DBG) dbg(`setUndoScope(${h ? "register" : "CLEAR"}) — workflowSeg=${useAppStore.getState().workflowSegmentId ?? "null"}`);
}

/** Is the WORKFLOW pane the ACTIVE undo surface? Requires the panel to actually be OPEN
 *  (workflowSegmentId) — NOT just `activePane`, which can go stale at "workflow" after the editor
 *  closed (observed: a phantom timeline undo fired from a "workflow"-marked pane whose editor was gone).
 *  The panel-open truth (workflowSegmentId) gates it: no panel ⇒ the timeline owns Ctrl+Z. */
function workflowUndoActive(): boolean {
  const a = useAppStore.getState();
  return a.workflowSegmentId != null && a.activePane === "workflow";
}

export function routeUndo() {
  const wf = workflowUndoActive();
  if (UNDO_DBG) dbg(`routeUndo → ${wf ? (scopedHandler ? "WORKFLOW node stack" : "WORKFLOW (no scope → no-op)") : "timeline"} (activePane=${useAppStore.getState().activePane}, workflowSeg=${useAppStore.getState().workflowSegmentId ?? "null"})`);
  // In the workflow pane, Ctrl+Z acts ONLY on the node stack (or no-ops if none) — it must NEVER revert
  // the timeline arrangement (that was the phantom "回退·子轨道静音" from a stale-pane / gone-editor state).
  if (wf) scopedHandler?.undo();
  else useHistoryStore.getState().undo();
}

export function routeRedo() {
  const wf = workflowUndoActive();
  if (UNDO_DBG) dbg(`routeRedo → ${wf ? (scopedHandler ? "WORKFLOW node stack" : "WORKFLOW (no scope → no-op)") : "timeline"} (activePane=${useAppStore.getState().activePane}, workflowSeg=${useAppStore.getState().workflowSegmentId ?? "null"})`);
  if (wf) scopedHandler?.redo();
  else useHistoryStore.getState().redo();
}

/** canUndo / canRedo for whichever stack Ctrl+Z would act on RIGHT NOW, so the Edit menu's enablement
 *  matches. In the workflow pane with no scope yet, nothing is undoable (mirrors routeUndo's no-op). */
export function routeCanUndo(): boolean {
  if (workflowUndoActive()) return scopedHandler ? scopedHandler.canUndo() : false;
  return useHistoryStore.getState().canUndo;
}

export function routeCanRedo(): boolean {
  if (workflowUndoActive()) return scopedHandler ? scopedHandler.canRedo() : false;
  return useHistoryStore.getState().canRedo;
}

/**
 * Install the auto-capture subscription on the project store. Discrete document mutations push the
 * pre-change snapshot. Continuous gestures are suppressed (txnDepth) and recorded once at commit.
 * Idempotent — re-installing (HMR) tears down the previous subscription first. Returns an unsubscribe.
 */
export function installHistory(): () => void {
  if (unsubscribe) unsubscribe();
  lastSig = currentSig();
  unsubscribe = useProjectStore.subscribe((next, prev) => {
    if (applying || txnDepth > 0) return;
    // Cheap early-out: playhead/selection-only sets don't touch the undoable refs.
    if (next.tracks === prev.tracks && next.tempo === prev.tempo && next.timeSignature === prev.timeSignature) return;
    const sig = meaningfulSig(next.tracks, next.tempo, next.timeSignature);
    if (sig === lastSig) return; // only overlay fields changed (expand / render / workflow / loading)
    if (UNDO_DBG) {
      const from: Snapshot = { tracks: prev.tracks, tempo: prev.tempo, timeSignature: prev.timeSignature, selection: currentSelection(), seq: 0 };
      const to: Snapshot = { tracks: next.tracks, tempo: next.tempo, timeSignature: next.timeSignature, selection: currentSelection(), seq: 0 };
      dbg(`capture depth=${past.length + 1} delta=${describeDelta(from, to)}`);
    }
    pushPast({
      tracks: prev.tracks,
      tempo: prev.tempo,
      timeSignature: prev.timeSignature,
      selection: currentSelection(),
      seq: nextSnapshotSeq(),
    });
    future = [];
    lastSig = sig;
    syncFlags();
  });
  return () => {
    if (unsubscribe) {
      unsubscribe();
      unsubscribe = null;
    }
  };
}
