/**
 * S61 — arrangement clipboard (segments + whole tracks). In-memory, app-internal (like every DAW's
 * clip clipboard; the OS clipboard has no meaningful representation for render caches).
 *
 * Semantics (user-specified):
 * - Paste = insert copies at the CURRENT PLAYHEAD on the target track (the right-clicked track, else
 *   the selected/active track). Multi-segment copies keep their relative tick offsets (anchor = the
 *   earliest copied start) AND their relative track layout (per source track).
 * - Single-clip paste onto a mismatching track TYPE = no-op (with a toast).
 * - Target occupied (any overlap) → a NEW track of that type is created right below, like drag-import
 *   placement. A new VOCAL track created this way inherits the reference track's singer config, so the
 *   spill-over clip keeps singing (the "clip follows its track's config" rule still holds — the new
 *   track's config IS the reference's).
 * - Copies carry the per-segment node graph + rendered sub-lanes + the runtime render cache
 *   (SegmentRenderSnapshot) so a pasted copy is NOT "un-rendered": node badges read completed, and
 *   disconnect→reconnect re-deposits from cache instead of swallowing the audio (the S24-era trap).
 * - Vocal bakes ride along under the S57 dual-sig discipline: the copy's notes get FRESH ids (mandatory
 *   — duplicated note ids corrupt selection/editing), which permanently invalidates the carried
 *   renderedSig, so validity is re-established via windowSig: stamped when the source bake was CLEAN
 *   and the destination track's config (vocalTrackSig) matches the copy source's; cleared (→ dirty →
 *   auto re-render on Play) when the destination differs; the bake is DROPPED entirely when the
 *   destination has no resolvable singer (never let a stale stem play as false-clean — S57 §crown).
 */
import type { LaneControl, Note, Segment, Track } from "../types/project";
import { useProjectStore } from "../store/project";
import { useAppStore } from "../store/app";
import { useAudioStore } from "../store/audio";
import { useHistoryStore } from "../store/history";
import { useWorkflowStore, type SegmentRenderSnapshot } from "../store/workflow";
import { blankTrack } from "./trackFactory";
import { clipCollides, laneControlFor } from "./trackLayout";
import { laneGroupId, laneRowKey } from "./audio/laneOps";
import {
  VOCAL_LANE_ID,
  isVocalDirty,
  resolveTrackVoice,
  stampSplitWindowSigs,
  vocalTrackSig,
} from "./vocal/vocalRender";
import i18n from "../i18n";

interface VocalBakeMeta {
  /** The source bake passed isVocalDirty at copy time (dual-sig accepted). */
  bakeClean: boolean;
  /** vocalTrackSig(sourceTrack, tempoAtCopy) — the carried stem is valid for a destination iff the
   *  destination's vocalTrackSig is byte-equal (same singer/params/range record/tempo). */
  trackSigAtCopy: string;
}

interface ClipSegItem {
  trackType: Track["trackType"];
  /** Index of the source track at copy time — preserves relative track layout on multi-track paste. */
  srcTrackIndex: number;
  /** Deep clone; `loading` segments are never copied and loading lanes are stripped. Ids are the
   *  SOURCE ids — freshened per paste (each paste must mint its own). */
  seg: Segment;
  /** Runtime render cache snapshot (audio workflow segments only). */
  renderState: SegmentRenderSnapshot | null;
  /** The source track's mixer entries for this segment's lane groups (installed absent-only). */
  laneControls: Record<string, LaneControl>;
  laneMutes: Record<string, boolean>;
  vocal?: VocalBakeMeta;
}

export type ClipboardContent =
  | { kind: "segments"; items: ClipSegItem[]; anchorTick: number }
  | { kind: "track"; track: Track; renderStates: Record<string, SegmentRenderSnapshot>; vocalMeta: Record<string, VocalBakeMeta> };

let clipboard: ClipboardContent | null = null;

export function clipboardKind(): "segments" | "track" | null {
  return clipboard?.kind ?? null;
}

/** Dropped on project new/open/recover (teardownForLoad): carried render paths / laneControls / sigs
 *  belong to the OLD document; cross-document paste would resurrect its caches half-broken. */
export function clearClipboard(): void {
  clipboard = null;
}

/** S61 cleanup support: every audio path the CLIPBOARD still references (cut segments' sources,
 *  carried rendered lanes, render-cache snapshots). The Settings render-cache sweep must NOT delete
 *  these — a paste after cleanup would otherwise stamp a bake "valid" over a deleted stem
 *  (permanently silent false-clean, the S57 crown violation the audit caught). */
export function clipboardReferencedPaths(): string[] {
  if (!clipboard) return [];
  const out = new Set<string>();
  const addSeg = (seg: Segment) => {
    if (seg.content.type === "audioClip") out.add(seg.content.sourcePath);
    for (const o of seg.processedOutputs ?? []) out.add(o.audioPath);
  };
  const addSnap = (snap: SegmentRenderSnapshot | null | undefined) => {
    for (const paths of Object.values(snap?.nodeOutputs ?? {})) {
      for (const p of paths) if (p) out.add(p);
    }
  };
  if (clipboard.kind === "segments") {
    for (const it of clipboard.items) {
      addSeg(it.seg);
      addSnap(it.renderState);
    }
  } else {
    for (const seg of clipboard.track.segments) addSeg(seg);
    for (const snap of Object.values(clipboard.renderStates)) addSnap(snap);
  }
  return [...out];
}

/** The current segment selection, primary-fallback (the exact Toolbar-delete contract). */
function selectionTargets(): Array<{ trackId: string; segmentId: string }> {
  const app = useAppStore.getState();
  const set = app.selectedSegments;
  const primary = app.selectedSegment;
  return set.length > 0 ? set : primary ? [primary] : [];
}

/** Copy the selected segment(s) into the clipboard. Returns how many were copied (0 = nothing). */
export function copySelectedSegments(): number {
  return copySelection().length;
}

/** The copy core: returns exactly WHICH selections were copied (cut deletes precisely these —
 *  a mid-decode `loading` clip is skipped by copy and must NOT be deleted by cut, or "move via
 *  cut/paste" silently drops it; audit S61). */
function copySelection(): Array<{ trackId: string; segmentId: string }> {
  const st = useProjectStore.getState();
  const tempo = st.tempo;
  const copied: Array<{ trackId: string; segmentId: string }> = [];
  const items: ClipSegItem[] = [];
  for (const sel of selectionTargets()) {
    const trackIndex = st.tracks.findIndex((t) => t.id === sel.trackId);
    const track = st.tracks[trackIndex];
    const seg = track?.segments.find((s) => s.id === sel.segmentId);
    if (!track || !seg || seg.loading) continue; // mid-decode clips have no copyable content yet
    const clone = structuredClone(seg);
    delete clone.loading;
    // Strip loading lanes: a carried placeholder would spin forever (no reconciler/watcher finalizes a
    // clipboard copy of a mid-render run) — mirror the split-settled filter.
    clone.processedOutputs = clone.processedOutputs?.filter((o) => !o.loading);
    if (clone.processedOutputs?.length === 0) delete clone.processedOutputs;
    const laneControls: Record<string, LaneControl> = {};
    const laneMutes: Record<string, boolean> = {};
    for (const o of clone.processedOutputs ?? []) {
      const gid = laneGroupId(o);
      const ctrl = laneControlFor(track, gid, o.laneId);
      if (ctrl && !(gid in laneControls)) laneControls[gid] = structuredClone(ctrl);
      const rk = laneRowKey(o);
      if (track.laneMutes?.[rk]) laneMutes[rk] = true;
    }
    const hasBake = seg.content.type === "notes" && !!clone.processedOutputs?.some((o) => o.laneId === VOCAL_LANE_ID);
    const item: ClipSegItem = {
      trackType: track.trackType,
      srcTrackIndex: trackIndex,
      seg: clone,
      renderState:
        seg.content.type === "audioClip"
          ? structuredClone(useWorkflowStore.getState().snapshotSegmentState(seg.id))
          : null,
      laneControls,
      laneMutes,
      // bakeClean REQUIRES a resolvable singer: with the model missing, isVocalDirty returns false
      // ("can't render" ≠ "clean") — treating that as clean would let a stale stem paste as valid
      // once the model is later installed with an identical trackSig (audit S61).
      ...(hasBake
        ? { vocal: { bakeClean: !!resolveTrackVoice(track) && !isVocalDirty(track, seg, tempo), trackSigAtCopy: vocalTrackSig(track, tempo) } }
        : {}),
    };
    items.push(item);
    copied.push({ trackId: track.id, segmentId: seg.id });
  }
  if (items.length === 0) return [];
  const anchorTick = Math.min(...items.map((i) => i.seg.startTick));
  clipboard = { kind: "segments", items, anchorTick };
  return copied;
}

/** Copy a whole track (header right-click 复制轨道). Verbatim clone incl. config/mixer/view state;
 *  loading segments dropped (their decode finalizer targets the ORIGINAL ids — a copy would stay
 *  stuck loading forever), loading lanes stripped. */
export function copyTrackToClipboard(trackId: string): boolean {
  const st = useProjectStore.getState();
  const tempo = st.tempo;
  const track = st.tracks.find((t) => t.id === trackId);
  if (!track) return false;
  const clone = structuredClone(track);
  clone.segments = clone.segments.filter((s) => !s.loading);
  const renderStates: Record<string, SegmentRenderSnapshot> = {};
  const vocalMeta: Record<string, VocalBakeMeta> = {};
  for (const seg of clone.segments) {
    seg.processedOutputs = seg.processedOutputs?.filter((o) => !o.loading);
    if (seg.processedOutputs?.length === 0) delete seg.processedOutputs;
    if (seg.content.type === "audioClip") {
      const snap = useWorkflowStore.getState().snapshotSegmentState(seg.id);
      if (snap) renderStates[seg.id] = structuredClone(snap);
    } else if (seg.processedOutputs?.some((o) => o.laneId === VOCAL_LANE_ID)) {
      const live = track.segments.find((s) => s.id === seg.id)!;
      // same bakeClean rule as the segment copy: unresolvable singer ⇒ never "clean"
      vocalMeta[seg.id] = {
        bakeClean: !!resolveTrackVoice(track) && !isVocalDirty(track, live, tempo),
        trackSigAtCopy: vocalTrackSig(track, tempo),
      };
    }
  }
  clipboard = { kind: "track", track: clone, renderStates, vocalMeta };
  return true;
}

/** Cut = copy + delete EXACTLY the copied originals (one undo step — the delete). A `loading`
 *  clip in the selection is skipped by copy, so it must survive the cut too. */
export function cutSelectedSegments(): number {
  const copied = copySelection();
  if (copied.length === 0) return 0;
  useProjectStore.getState().deleteSegments(copied);
  useAppStore.getState().clearSelection();
  if (useAudioStore.getState().isPlaying) useAudioStore.getState().bumpSchedule();
  return copied.length;
}

/** Mint a paste-instance of a clipboard segment: fresh segment id, fresh note ids (duplicated note ids
 *  = selection/editing corruption — the splitSegment precedent), repositioned to the playhead. */
function materializeSegment(item: ClipSegItem, playhead: number, anchorTick: number): Segment {
  const seg = structuredClone(item.seg);
  seg.id = crypto.randomUUID();
  seg.startTick = Math.max(0, playhead + (item.seg.startTick - anchorTick));
  if (seg.content.type === "notes") {
    seg.content.notes = seg.content.notes.map((n: Note): Note => ({ ...n, id: crypto.randomUUID() }));
  }
  return seg;
}

/** Append `seg` to `trackId`, installing the carried mixer entries ABSENT-ONLY (the cross-track-move
 *  carry rule: never clobber the destination's existing group settings). */
function insertSegmentIntoTrack(trackId: string, seg: Segment, item: ClipSegItem): void {
  const st = useProjectStore.getState();
  const track = st.tracks.find((t) => t.id === trackId);
  if (!track) return;
  const patch: Partial<Track> = { segments: [...track.segments, seg] };
  const ctrlKeys = Object.keys(item.laneControls).filter((k) => !(k in track.laneControls));
  if (ctrlKeys.length > 0) {
    const laneControls = { ...track.laneControls };
    for (const k of ctrlKeys) laneControls[k] = structuredClone(item.laneControls[k]!);
    patch.laneControls = laneControls;
  }
  const muteKeys = Object.keys(item.laneMutes).filter((k) => !track.laneMutes?.[k]);
  if (muteKeys.length > 0) {
    const laneMutes = { ...(track.laneMutes ?? {}) };
    for (const k of muteKeys) laneMutes[k] = true;
    patch.laneMutes = laneMutes;
  }
  st.updateTrack(trackId, patch);
}

/** Post-insert bookkeeping shared by segment- and track-paste: render-cache install + the vocal
 *  dual-sig stamp/clear (see the header comment). `stripInsteadOfCarry` was applied BEFORE insert. */
function finalizePastedSegment(trackId: string, segId: string, renderState: SegmentRenderSnapshot | null, vocal: VocalBakeMeta | undefined): void {
  if (renderState) useWorkflowStore.getState().installSegmentState(segId, structuredClone(renderState));
  if (!vocal) return;
  const st = useProjectStore.getState();
  const track = st.tracks.find((t) => t.id === trackId);
  const seg = track?.segments.find((s) => s.id === segId);
  if (!track || !seg || seg.content.type !== "notes") return;
  if (!seg.processedOutputs?.some((o) => o.laneId === VOCAL_LANE_ID && !o.loading)) return;
  const valid = vocal.bakeClean && vocal.trackSigAtCopy === vocalTrackSig(track, st.tempo);
  // valid → windowSig = this copy's own sig (clean, plays the carried stem); invalid → windowSig
  // cleared (renderedSig can never match the fresh note ids) → dirty → Play auto re-renders under the
  // destination's config. Same single stamp helper as split (parentWasDirty := !valid).
  stampSplitWindowSigs(trackId, [segId], st.tempo, !valid);
}

/** Drop the carried vocal bake when the destination can't render (no resolvable singer): with no
 *  config the dirty machinery is inert (isVocalDirty=false) and the stale stem would play as
 *  false-clean forever — the exact laundering S57 forbids. */
function stripBakeIfUnrenderable(seg: Segment, destTrack: Track | undefined): void {
  if (seg.content.type !== "notes" || !seg.processedOutputs) return;
  if (destTrack && resolveTrackVoice(destTrack)) return;
  seg.processedOutputs = seg.processedOutputs.filter((o) => o.laneId !== VOCAL_LANE_ID);
  if (seg.processedOutputs.length === 0) delete seg.processedOutputs;
}

export type PasteResult = "ok" | "empty" | "typeMismatch";

/** Paste the clipboard at the playhead. `targetTrackId` = the right-clicked track (context menu);
 *  omitted = the selected/active track. See the module header for the placement rules. */
export function pasteClipboard(targetTrackId?: string): PasteResult {
  if (!clipboard) return "empty";
  if (clipboard.kind === "track") return pasteTrackClipboard(targetTrackId);
  const { items, anchorTick } = clipboard;
  const st = useProjectStore.getState();
  const app = useAppStore.getState();
  const origTracks = st.tracks;
  const playhead = st.playheadTick;

  // Group items by their source track (ascending) — each group lands on one destination track.
  const bySrc = new Map<number, ClipSegItem[]>();
  for (const it of items) {
    const list = bySrc.get(it.srcTrackIndex) ?? [];
    list.push(it);
    bySrc.set(it.srcTrackIndex, list);
  }
  const groups = [...bySrc.entries()].sort((a, b) => a[0] - b[0]).map(([srcIdx, list]) => ({ srcIdx, list }));

  const baseId = targetTrackId ?? app.selectedSegment?.trackId ?? app.activeTrackId;
  const baseIdx = baseId ? origTracks.findIndex((t) => t.id === baseId) : -1;

  // The single-clip rule: pasting one clip onto an explicitly-targeted track of the WRONG type is a
  // no-op (the user aimed at that track). Multi-track pastes instead spill mismatches to new tracks.
  if (groups.length === 1 && baseIdx >= 0 && origTracks[baseIdx]!.trackType !== groups[0]!.list[0]!.trackType) {
    return "typeMismatch";
  }

  // Plan each group's destination against the PRE-PASTE track list (later insertions shift indices, so
  // candidates are resolved by id at insert time).
  const plans = groups.map((g) => {
    const type = g.list[0]!.trackType;
    const candidateIdx = baseIdx >= 0 ? baseIdx + (g.srcIdx - groups[0]!.srcIdx) : -1;
    const candidate = candidateIdx >= 0 && candidateIdx < origTracks.length ? origTracks[candidateIdx] : undefined;
    const fits =
      candidate &&
      candidate.trackType === type &&
      g.list.every((it) => {
        const start = Math.max(0, playhead + (it.seg.startTick - anchorTick));
        return !clipCollides(candidate, start, it.seg.durationTicks);
      });
    return { group: g, type, candidate, fits: !!fits };
  });

  const hist = useHistoryStore.getState();
  const pasted: Array<{ trackId: string; segmentId: string; renderState: SegmentRenderSnapshot | null; vocal?: VocalBakeMeta }> = [];
  hist.beginTransaction();
  try {
    for (const plan of plans) {
      let destId: string;
      if (plan.fits && plan.candidate) {
        destId = plan.candidate.id;
      } else {
        // New track of the group's type, right below the candidate (or at the end). A same-type
        // reference (the collided candidate) seeds a vocal track's singer config so the spill-over
        // clip keeps its renderable context.
        const live = useProjectStore.getState().tracks;
        const refIdx = plan.candidate ? live.findIndex((t) => t.id === plan.candidate!.id) : -1;
        const insertIdx = refIdx >= 0 ? refIdx + 1 : undefined;
        const n = live.filter((tk) => tk.trackType === plan.type).length + 1;
        const nt = blankTrack(crypto.randomUUID(), `${plan.type === "audio" ? "Audio" : "Vocal"} ${n}`, plan.type);
        if (plan.type === "vocal" && plan.candidate?.trackType === "vocal") {
          if (plan.candidate.vocalParams) nt.vocalParams = structuredClone(plan.candidate.vocalParams);
          if (plan.candidate.voiceModel) nt.voiceModel = plan.candidate.voiceModel;
          if (plan.candidate.voiceModelAvatar) nt.voiceModelAvatar = plan.candidate.voiceModelAvatar;
        }
        useProjectStore.getState().addTrack(nt, insertIdx);
        destId = nt.id;
      }
      const destTrack = useProjectStore.getState().tracks.find((t) => t.id === destId);
      for (const it of plan.group.list) {
        const seg = materializeSegment(it, playhead, anchorTick);
        stripBakeIfUnrenderable(seg, destTrack);
        insertSegmentIntoTrack(destId, seg, it);
        pasted.push({ trackId: destId, segmentId: seg.id, renderState: it.renderState, vocal: it.vocal });
      }
    }
  } finally {
    hist.commitTransaction();
  }
  // Overlay bookkeeping AFTER the undoable step: render-cache install (workflow store, not undoable)
  // + the vocal windowSig stamp (replaceProcessedOutputs — overlay, sig-invisible).
  for (const p of pasted) finalizePastedSegment(p.trackId, p.segmentId, p.renderState, p.vocal);
  useAppStore.getState().selectSegments(pasted.map((p) => ({ trackId: p.trackId, segmentId: p.segmentId })));
  if (useAudioStore.getState().isPlaying) useAudioStore.getState().bumpSchedule();
  return "ok";
}

function pasteTrackClipboard(belowTrackId?: string): PasteResult {
  if (!clipboard || clipboard.kind !== "track") return "empty";
  const { track: src, renderStates, vocalMeta } = clipboard;
  const st = useProjectStore.getState();
  const app = useAppStore.getState();
  const clone = structuredClone(src);
  clone.id = crypto.randomUUID();
  clone.name = `${src.name} ${i18n.t("tracks.copySuffix")}`;
  const finalize: Array<{ segId: string; renderState: SegmentRenderSnapshot | null; vocal?: VocalBakeMeta }> = [];
  clone.segments = clone.segments.map((seg) => {
    const oldId = seg.id;
    seg.id = crypto.randomUUID();
    if (seg.content.type === "notes") {
      seg.content.notes = seg.content.notes.map((n: Note): Note => ({ ...n, id: crypto.randomUUID() }));
    }
    finalize.push({ segId: seg.id, renderState: renderStates[oldId] ?? null, vocal: vocalMeta[oldId] });
    return seg;
  });
  const anchorId = belowTrackId ?? app.activeTrackId ?? app.selectedSegment?.trackId;
  const anchorIdx = anchorId ? st.tracks.findIndex((t) => t.id === anchorId) : -1;
  const insertIdx = anchorIdx >= 0 ? anchorIdx + 1 : undefined;
  const hist = useHistoryStore.getState();
  hist.beginTransaction();
  try {
    useProjectStore.getState().addTrack(clone, insertIdx);
  } finally {
    hist.commitTransaction();
  }
  // Identical config ⇒ vocalTrackSig matches (unless the tempo/model registry changed since copy) ⇒
  // clean bakes stay clean; a dirty/stale source bake stays dirty and re-renders on Play — exactly the
  // source track's own behavior. An unconfigured source track carries verbatim (no meta recorded).
  for (const f of finalize) finalizePastedSegment(clone.id, f.segId, f.renderState, f.vocal);
  useAppStore.getState().setActiveTrack(clone.id);
  if (useAudioStore.getState().isPlaying) useAudioStore.getState().bumpSchedule();
  return "ok";
}

/** Shared paste entry with user feedback (toast on type mismatch). Used by keyboard + menus. */
export function pasteWithFeedback(targetTrackId?: string): void {
  const res = pasteClipboard(targetTrackId);
  if (res === "typeMismatch") {
    useAppStore.getState().showToast(i18n.t("clipboard.typeMismatch"), "info");
  }
}
