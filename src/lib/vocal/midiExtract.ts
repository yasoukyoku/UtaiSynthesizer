// midiExtract.ts — S60 GAME 人声→MIDI extraction glue (single funnel for the lane context menu).
//
// Inference runs in Rust (inference/midi_extract.rs, the official GAME 1.0.3 ONNX pipeline on CPU)
// over each stem of the clicked lane GROUP; results come back in SOURCE ms (window folded in, like
// analyze_segment_tempo). Here we map source ms → timeline ticks through the ONE conversion chain
// (segStretch × msToTicks, laneOps semantics), quantize note BOUNDARIES to the 1/12-beat grid
// (boundary quantization, not per-note duration rounding — the v1 frame-drift rule), and build one
// NEW vocal track per stem in a single undo transaction (importScoreFile pattern).
//
// UX contracts (§user):
//   - a group of N stems → N new MIDI tracks, inserted right below the source track;
//   - placeholder lyrics (the track language's defaultLyric — extraction has no lyrics by design,
//     the feature exists for 翻唱改词);
//   - the whole extraction = ONE Ctrl+Z; Ctrl+Z WHILE extracting cancels the inference —
//     but ONLY when the pending add-track is what Ctrl+Z would logically hit: the interceptor
//     stands down while the workflow pane owns undo, and while timeline edits NEWER than the
//     extraction exist (those undo normally; the extraction keeps running). Audit S60.
//   - a visible per-lane "extracting" indicator the whole time (never look frozen).

import { invoke } from "@tauri-apps/api/core";
import { logToBackend } from "../log";
import { listen } from "@tauri-apps/api/event";
import i18n from "../../i18n";
import { backendErrorMessage, isCancelError } from "../backendError";
import type { Note, ProcessedOutput, Segment } from "../../types/project";
import { blankTrack } from "../trackFactory";
import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import {
  useHistoryStore,
  registerUndoInterceptor,
  timelineUndoDepth,
  inGestureTransaction,
} from "../../store/history";
import { getLoadEpoch } from "../project/projectFile";
import { flushAutosaveNow } from "../project/autosave";
import { laneGroupId, laneVisiblePieces, msToTicks, segStretch, ticksToMs } from "../audio/laneOps";
import { DEFAULT_LANG_ID, langById } from "./languages";
import { loadSetting, saveSetting } from "../settings";
import { quantizeSpans, QUANTIZE_IMPORT_KEY, type QuantSpan } from "./quantize";

const t = (k: string, vars?: Record<string, unknown>) => i18n.t(k, vars ?? {}) as string;

/** Mirror of the Rust ExtractedNote (serde snake_case comes through invoke verbatim). */
interface ExtractedNote {
  onset_ms: number;
  offset_ms: number;
  pitch: number;
}

export function extractKey(segId: string, group: string): string {
  return `${segId}:${group}`;
}

// ── module job state (progress is read by the Arrangement draw each frame — no store churn) ──
const progressByKey = new Map<string, number>();
const jobMetaByJobId = new Map<string, { key: string; index: number; total: number }>();
/** Timeline undo depth at each group's start — the interceptor only consumes Ctrl+Z when no
 *  NEWER timeline steps exist (else the user is undoing their own later edit). */
const startDepthByKey = new Map<string, number>();
/** Groups the user cancelled — closes the cancel-vs-complete race: a Ctrl+Z landing after the
 *  last chunk finished (Rust returns Ok) must still drop the result, or the user sees an
 *  "已取消" toast AND new tracks (audit S60). */
const cancelledKeys = new Set<string>();

/** Current 0..1 progress of a lane group's extraction (for the canvas indicator label). */
export function midiExtractProgress(key: string): number {
  return progressByKey.get(key) ?? 0;
}

let progressListenerInstalled = false;
function ensureProgressListener() {
  if (progressListenerInstalled) return;
  progressListenerInstalled = true;
  void listen<{ job_id: string; progress: number }>("midi-extract-progress", (e) => {
    const meta = jobMetaByJobId.get(e.payload.job_id);
    // group progress = (finished stems + this stem's fraction) / stems — a per-job 0..1
    // written raw would snap back to 0% on every stem switch (audit S60)
    if (meta) progressByKey.set(meta.key, (meta.index + e.payload.progress) / meta.total);
  });
}

// ── undo-cancels-extraction (one registration while ANY job is active) ──
let unregisterInterceptor: (() => void) | null = null;

function interceptorWouldConsume(): boolean {
  const app = useAppStore.getState();
  const jobs = app.midiExtracting;
  const keys = Object.keys(jobs);
  if (!keys.length) return false;
  // the workflow pane owns Ctrl+Z while open+active — never hijack the node stack (the
  // routeUndo isolation invariant; audit S60 MAJOR)
  if (app.workflowSegmentId != null && app.activePane === "workflow") return false;
  // edits newer than the newest extraction must undo normally (extraction keeps running)
  const newestStart = Math.max(...keys.map((k) => startDepthByKey.get(k) ?? 0));
  return timelineUndoDepth() <= newestStart;
}

function cancelJobs(keys: string[]): void {
  const jobs = useAppStore.getState().midiExtracting;
  for (const k of keys) {
    cancelledKeys.add(k);
    for (const jobId of jobs[k]?.jobIds ?? []) void invoke("cancel_midi_extract", { jobId });
  }
}

function interceptorConsume(): void {
  cancelJobs(Object.keys(useAppStore.getState().midiExtracting));
  useAppStore.getState().showToast(t("midiExtract.cancelled"), "info");
}

function syncInterceptor() {
  const active = Object.keys(useAppStore.getState().midiExtracting).length > 0;
  if (active && !unregisterInterceptor) {
    unregisterInterceptor = registerUndoInterceptor({
      wouldConsume: interceptorWouldConsume,
      consume: interceptorConsume,
    });
  } else if (!active && unregisterInterceptor) {
    unregisterInterceptor();
    unregisterInterceptor = null;
  }
}

/** Cancel every in-flight extraction and clear all job state WITHOUT toasting — called from
 *  teardownForLoad (new/open/recover): the old document's jobs must not keep burning CPU nor
 *  leave the interceptor armed to eat the new document's first Ctrl+Z (audit S60). */
export function cancelExtractionsForTeardown(): void {
  const app = useAppStore.getState();
  const keys = Object.keys(app.midiExtracting);
  cancelJobs(keys);
  for (const k of keys) {
    app.setMidiExtracting(k, null);
    progressByKey.delete(k);
    startDepthByKey.delete(k);
  }
  syncInterceptor();
}

/** Stable-CODE → localized message (tempoErrorMessage pattern). */
export function midiExtractErrorMessage(e: unknown): string {
  const msg = e instanceof Error ? e.message : String(e);
  if (msg.includes("MIDI_EXTRACT_NOT_INSTALLED")) return t("midiExtract.errNotInstalled");
  if (msg.includes("MIDI_EXTRACT_TOO_SHORT")) return t("midiExtract.errTooShort");
  if (msg.includes("MIDI_EXTRACT_LOAD_FAILED")) return t("midiExtract.errLoad");
  if (isCancelError(msg)) return t("midiExtract.cancelled"); // e.g. GAME_DL_FAILED: DOWNLOAD_CANCELLED
  // App-wide codes riding inside wrappers (GAME_DL_FAILED: DOWNLOAD_* from the model downloader).
  const shared = backendErrorMessage(msg);
  if (shared) return `${t("midiExtract.errFailed")}: ${shared}`;
  return `${t("midiExtract.errFailed")}: ${msg}`;
}

/** One stem's audible notes as ABSOLUTE tick spans (+ pitch). NO rounding here — since S87 quantization is
 *  the USER's choice (the shared `quantizeSpans`), so the raw mapping and the rounding decision are
 *  separate steps. Honors the lane's non-destructive edits: a note whose midpoint falls in a trimmed-out
 *  range is silent in playback and must not be transcribed. */
function absNotes(
  raw: ExtractedNote[],
  seg: Segment,
  out: ProcessedOutput,
  tempo: number,
): { span: QuantSpan; pitch: number }[] {
  if (seg.content.type !== "audioClip") return [];
  const r = segStretch(seg);
  const winStart = seg.content.offsetMs;
  const pieces = laneVisiblePieces(seg, seg.laneOps?.[laneGroupId(out)], out.totalDurationMs, tempo);
  const audible = (srcMs: number) => pieces.some((p) => srcMs >= p.startMs && srcMs <= p.endMs);
  const items: { span: QuantSpan; pitch: number }[] = [];
  for (const n of raw) {
    if (!audible((n.onset_ms + n.offset_ms) / 2)) continue;
    // source ms → absolute timeline ticks (played time = source distance × r past the window start)
    items.push({
      span: {
        start: seg.startTick + msToTicks((n.onset_ms - winStart) * r, tempo),
        end: seg.startTick + msToTicks((n.offset_ms - winStart) * r, tempo),
      },
      pitch: n.pitch,
    });
  }
  return items;
}

/** S87 — round the raw spans to whole ticks WITHOUT touching the grid (the "don't quantize" branch). Keeps
 *  the ≥1 tick floor the store would otherwise apply silently, so no note can vanish here either. */
function wholeTicks(spans: readonly QuantSpan[]): QuantSpan[] {
  return spans.map((s) => {
    const start = Math.round(s.start);
    return { start, end: Math.max(start + 1, Math.round(s.end)) };
  });
}

/** Map absolute spans + pitches onto part-relative Note[] (rebased to `partStart`). */
function buildNotes(
  spans: readonly QuantSpan[],
  items: { pitch: number }[],
  partStart: number,
  lyric: string,
): Note[] {
  return spans.map((s, i) => ({
    id: crypto.randomUUID(),
    tick: s.start - partStart,
    duration: s.end - s.start,
    pitch: Math.min(127, Math.max(0, Math.round(items[i]!.pitch))),
    lyric,
    velocity: 100,
  }));
}

/** Wait for any held gesture transaction (drag/slider) to close, so the landing transaction
 *  isn't folded into the user's gesture step (audit S60). Bounded — proceeds after 5 s. */
async function whenNoGestureTransaction(): Promise<void> {
  for (let i = 0; i < 50 && inGestureTransaction(); i++) {
    await new Promise((res) => setTimeout(res, 100));
  }
}

/**
 * Extract MIDI from every stem of a lane GROUP into new vocal tracks. Fire-and-forget: failures
 * toast, stale results (segment edited/removed/reloaded mid-inference) are dropped with a toast,
 * Ctrl+Z during inference cancels (scoped — see header). One invocation per (segment, group).
 */
export async function extractMidiForLaneGroup(trackId: string, segId: string, group: string): Promise<void> {
  const key = extractKey(segId, group);
  const app = useAppStore.getState();
  if (app.midiExtracting[key]) return;
  const p = useProjectStore.getState();
  const track = p.tracks.find((tk) => tk.id === trackId);
  const seg = track?.segments.find((s) => s.id === segId);
  if (!track || !seg || seg.content.type !== "audioClip" || seg.loading) return;
  const outs = (seg.processedOutputs ?? []).filter((o) => !o.loading && laneGroupId(o) === group);
  if (!outs.length) return;

  // engine installed? (menu building is sync — check here, route to the resource manager if missing)
  try {
    const st = await invoke<{ installed: boolean }>("midi_extract_status");
    if (!st.installed) {
      app.showToast(t("midiExtract.errNotInstalled"), "info");
      if (!useAppStore.getState().modelManagerOpen) useAppStore.getState().toggleModelManager();
      return;
    }
  } catch {
    // status probe failing is non-fatal — the extraction call itself reports properly
  }
  if (useAppStore.getState().midiExtracting[key]) return; // double-trigger across the await

  // S87 — the same grid-rounding choice the score importer offers, asked BEFORE the (minutes-long) run.
  // Extraction onsets come from a pitch tracker, so rounding is almost always wanted — but "almost always"
  // is not "always", and the pre-S87 code rounded unconditionally AND silently dropped every note that
  // collapsed. One shared preference key with the importer (§user: default ON).
  let quantize = loadSetting(QUANTIZE_IMPORT_KEY, true);
  const choice = await useAppStore.getState().showConfirm({
    title: t("midiExtract.optionsTitle"),
    body: t("midiExtract.optionsBody"),
    check: {
      label: t("import.options.quantize"),
      initial: quantize,
      onChange: (v) => {
        quantize = v;
      },
    },
    buttons: [
      { id: "cancel", label: t("common.cancel") },
      { id: "ok", label: t("common.confirm"), kind: "primary" },
    ],
  });
  if (choice !== "ok") return; // Cancel / Esc / backdrop — nothing started
  saveSetting(QUANTIZE_IMPORT_KEY, quantize);
  if (useAppStore.getState().midiExtracting[key]) return; // double-trigger across the dialog await

  const tempo = p.tempo;
  const c = seg.content;
  const r = segStretch(seg);
  const winStart = c.offsetMs;
  const winEnd = c.offsetMs + ticksToMs(seg.durationTicks, tempo) / r;
  // the load epoch catches "same .usp reopened mid-inference" — ids + values alone would match
  const winSig = `${getLoadEpoch()}:${c.sourcePath}:${c.offsetMs}:${seg.startTick}:${seg.durationTicks}:${r}:${tempo}`;
  const segStillValid = (): boolean => {
    const p2 = useProjectStore.getState();
    const s2 = p2.tracks.find((tk) => tk.id === trackId)?.segments.find((s) => s.id === segId);
    if (!s2 || s2.content.type !== "audioClip") return false;
    const c2 = s2.content;
    return (
      `${getLoadEpoch()}:${c2.sourcePath}:${c2.offsetMs}:${s2.startTick}:${s2.durationTicks}:${segStretch(s2)}:${p2.tempo}` ===
      winSig
    );
  };

  ensureProgressListener();
  const jobs = outs.map((o, i) => ({ o, jobId: crypto.randomUUID(), index: i }));
  for (const j of jobs) jobMetaByJobId.set(j.jobId, { key, index: j.index, total: jobs.length });
  progressByKey.set(key, 0);
  startDepthByKey.set(key, timelineUndoDepth());
  useAppStore.getState().setMidiExtracting(key, { trackId, segId, group, jobIds: jobs.map((j) => j.jobId) });
  syncInterceptor();
  try {
    // sequential — the jobs share the CPU and the GAME sessions; progress stays readable
    const results: { o: ProcessedOutput; notes: ExtractedNote[] }[] = [];
    for (const { o, jobId, index } of jobs) {
      // the source vanished / changed mid-run → stop burning CPU on the remaining stems
      if (cancelledKeys.has(key) || !segStillValid()) break;
      const notes = await invoke<ExtractedNote[]>("extract_midi_from_audio", {
        path: o.audioPath,
        windowStartMs: winStart,
        windowEndMs: winEnd,
        jobId,
      });
      results.push({ o, notes });
      progressByKey.set(key, (index + 1) / jobs.length);
    }

    if (cancelledKeys.has(key)) return; // user cancelled — even if the last stem raced to Ok
    if (!segStillValid() || results.length < jobs.length) {
      useAppStore.getState().showToast(t("midiExtract.stale"), "info");
      return;
    }

    await whenNoGestureTransaction();
    if (cancelledKeys.has(key) || !segStillValid()) {
      useAppStore.getState().showToast(t("midiExtract.stale"), "info");
      return;
    }

    const p2 = useProjectStore.getState();
    const lyric = langById(DEFAULT_LANG_ID).defaultLyric;
    const history = useHistoryStore.getState();
    history.beginTransaction();
    let made = 0;
    let droppedTotal = 0;
    try {
      let insertAt = p2.tracks.findIndex((tk) => tk.id === trackId) + 1;
      for (const { o, notes: raw } of results) {
        const seg2 = p2.tracks.find((tk) => tk.id === trackId)?.segments.find((s) => s.id === segId);
        if (!seg2 || seg2.content.type !== "audioClip") break;
        const segEnd = seg2.startTick + seg2.durationTicks;
        const items = absNotes(raw, seg2, o, p2.tempo);
        if (!items.length) continue;
        // S87: rounding is the user's choice. A note that collapses is still DROPPED here (policy "drop":
        // GAME emits an exactly-contiguous chain, so a micro-note is a pitch-tracker transition artifact,
        // and widening it would push every following REAL note one cell later — cumulative drift against
        // the source audio, which is the one thing transcription must preserve). The change vs pre-S87 is
        // that the drop is now COUNTED and reported instead of a silent `continue`.
        // partStart also derives from the SAME spans the notes are built from — the old form re-derived it
        // from the UNFILTERED raw list with its own inline quantization, so the two could disagree.
        let spans: (QuantSpan | null)[];
        if (quantize) {
          const q = quantizeSpans(items.map((x) => x.span), "drop");
          spans = q.spans;
          droppedTotal += q.dropped;
        } else {
          spans = wholeTicks(items.map((x) => x.span));
        }
        const kept = spans
          .map((s, i) => (s ? { span: s, pitch: items[i]!.pitch } : null))
          .filter((x): x is { span: QuantSpan; pitch: number } => x !== null);
        if (!kept.length) continue;
        // part box aligned to the source segment, extended so the notes always fit
        const partStart = kept.reduce((m, k) => Math.min(m, k.span.start), seg2.startTick);
        const notes = buildNotes(kept.map((k) => k.span), kept, partStart, lyric);
        if (!notes.length) continue;
        const lastEnd = notes.reduce((m, n) => Math.max(m, n.tick + n.duration), 0);
        const newTrackId = crypto.randomUUID();
        p2.addTrack(blankTrack(newTrackId, `${o.laneLabel} MIDI`, "vocal"), insertAt++);
        const partId = p2.createVocalPart(newTrackId, partStart, Math.max(segEnd - partStart, lastEnd));
        p2.applyNoteEdits(newTrackId, partId, { add: notes });
        made++;
      }
    } finally {
      history.commitTransaction();
    }
    if (made > 0) {
      flushAutosaveNow();
      useAppStore.getState().showBanner(`${t("midiExtract.done")} · ${made}`, "info");
      // S87: the collapse drop is reported, never silent (pre-S87 it was a bare `continue`).
      if (droppedTotal > 0) useAppStore.getState().showToast(t("midiExtract.quantDropped", { count: droppedTotal }), "info");
    } else {
      useAppStore.getState().showToast(t("midiExtract.noNotes"), "info");
    }
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    if (!msg.includes("MIDI_EXTRACT_CANCELLED")) {
      // S74: GAME extraction is a blocking inference op — leave a copyable log trace.
      // This catch is OUTSIDE the per-stem loop, so it fires once per invocation (flood-safe).
      logToBackend("error", `MIDI extract failed: ${msg}`);
      useAppStore.getState().showToast(midiExtractErrorMessage(e), "error");
    } // cancellation already toasted by the interceptor
  } finally {
    for (const j of jobs) jobMetaByJobId.delete(j.jobId);
    progressByKey.delete(key);
    startDepthByKey.delete(key);
    cancelledKeys.delete(key);
    useAppStore.getState().setMidiExtracting(key, null);
    syncInterceptor();
  }
}
