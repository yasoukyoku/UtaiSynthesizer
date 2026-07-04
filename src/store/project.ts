import { create } from "zustand";
import { invoke } from "@tauri-apps/api/core";
import type { Track, Segment, LaneControl, ProcessedOutput, LaneClip } from "../types/project";
import { orderProcessedOutputs, laneControlFor } from "../lib/trackLayout";
import {
  laneGroupName,
  laneLabelParts,
  recordDetachLineage,
  ticksToMs,
} from "../lib/audio/laneOps";
import { useWorkflowStore } from "./workflow";
import { useAudioStore } from "./audio";

/** Cancel any in-flight render for the given segment ids — used when a segment/track is DELETED, the only
 *  op besides the Stop button that should cancel a render (split/move/rename/etc. must NOT). The still-
 *  running engine loop polls isCancelled and then stops the (single, global) Rust separation; flipping the
 *  execution off 'running' also un-sticks the quit/busy warning that scans for running executions. The JS
 *  execution flip is per segment id; the cancel_voice invoke below is app-GLOBAL (same semantics as
 *  cancel_separation) — a concurrent voice render of ANOTHER segment would also abort. Accepted trade-off:
 *  voice invokes carry no job id, and the deleted segment's run must not keep burning GPU minutes. */
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
  /** Open/close a BPM edit session (the Toolbar input's focus/blur) — setTempo scales from the
   *  session-start geometry so intermediate keystroke values can't compound rounding error. */
  beginTempoScale: () => void;
  endTempoScale: () => void;
  setPlayhead: (tick: number) => void;
  selectNotes: (ids: string[]) => void;
  toggleTrackExpanded: (trackId: string) => void;
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
    set((s) => ({
      dirty: true,
      tracks: s.tracks.map((t) => {
        if (t.id !== trackId) return t;
        const segIdx = t.segments.findIndex((seg) => seg.id === segmentId);
        if (segIdx < 0) return t;
        const seg = t.segments[segIdx]!;

        if (atTick <= seg.startTick || atTick >= seg.startTick + seg.durationTicks) return t;

        const leftDuration = atTick - seg.startTick;
        const rightDuration = seg.durationTicks - leftDuration;
        const leftSeg: Segment = { ...seg, durationTicks: leftDuration };

        let rightContent = seg.content;
        if (seg.content.type === "audioClip") {
          const splitOffsetMs = ticksToMs(leftDuration, s.tempo);
          rightContent = {
            ...seg.content,
            offsetMs: seg.content.offsetMs + splitOffsetMs,
          };
        }
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
          startTick: atTick,
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
        };

        didSplit = true;
        const newSegments = [...t.segments];
        newSegments.splice(segIdx, 1, leftSeg, rightSeg);
        return { ...t, segments: newSegments };
      }),
    }));
    if (didSplit) {
      // Clone the render CACHE (+ settled node badges/execution) onto the new half so a POST-render split
      // "remembers" it was rendered: deleting + reconnecting an Output edge re-deposits from the cache
      // instead of treating the upstream as never-executed and forcing a full re-run. The cache paths
      // point at the original segment's cache dir, whose stem files (whole-source) both halves window into.
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
    const geo = new Map<string, { start: number; dur: number }>();
    for (const t of base.tracks) {
      for (const s of t.segments) {
        if (s.content.type !== "audioClip") continue;
        const start = Math.round(s.startTick * k);
        const end = Math.round((s.startTick + s.durationTicks) * k);
        geo.set(s.id, { start, dur: Math.max(1, end - start) });
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
          return g ? { ...seg, startTick: g.start, durationTicks: g.dur } : seg;
        }),
      })),
    }));
    // Tempo changes the tick↔seconds mapping — the already-scheduled Web Audio sources keep the old
    // layout while the playhead advances at the new rate. Reschedule like every other timing edit.
    if (useAudioStore.getState().isPlaying) useAudioStore.getState().bumpSchedule();
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
