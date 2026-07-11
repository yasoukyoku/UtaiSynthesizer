// tempoDetect.ts — S59 BPM/beat-grid detection glue (single funnel for the Arrangement menu).
//
// Detection runs in Rust (utai-dsp tempo.rs, classical DSP) over the segment's SOURCE window;
// the result is anchored in SOURCE ms so split halves share one grid and resize/stretch never
// invalidates it. The correction transforms below are the DJ/Cubase affordance set (×2 / ÷2 /
// downbeat nudge) — pure functions so they're unit-testable.

import { invoke } from "@tauri-apps/api/core";
import { useProjectStore } from "../../store/project";
import { useAudioStore } from "../../store/audio";
import { useAppStore } from "../../store/app";
import i18n from "../../i18n";
import { ticksToMs } from "./laneOps";
import { getLoadEpoch } from "../project/projectFile";
import type { TempoDetect } from "../../types/project";

/** Mirror of the Rust TempoAnalysisResult (serde snake_case comes through invoke verbatim). */
interface AnalyzeResult {
  bpm: number;
  grid_anchor_ms: number;
  downbeat_index: number;
  downbeat_margin: number;
  confidence: number;
  not_constant: boolean;
  candidates: number[];
}

/** Detected-BPM sanity range (matches the Rust analyzer's [60,200] plus correction headroom). */
export const TEMPO_BPM_MIN = 20;
export const TEMPO_BPM_MAX = 400;

/** Double-trigger guard: one analysis per segment at a time (headlessDeposit-Set pattern). */
const inFlight = new Set<string>();

/** Run detection for an audio segment's current window and store the grid. Fire-and-forget:
 *  failures toast, a stale result (segment edited/removed mid-analysis) is dropped. */
export async function detectSegmentTempo(trackId: string, segmentId: string, beatsPerBar: number): Promise<void> {
  if (inFlight.has(segmentId)) return;
  const p = useProjectStore.getState();
  const seg = p.tracks.find((t) => t.id === trackId)?.segments.find((s) => s.id === segmentId);
  if (!seg || seg.content.type !== "audioClip" || seg.loading) return;
  const c = seg.content;
  // the load epoch catches "same .usp reopened mid-analysis" — ids + values alone would match
  const winSig = `${getLoadEpoch()}:${c.sourcePath}:${c.offsetMs}:${seg.durationTicks}:${c.stretch ?? 1}:${p.tempo}`;
  const af = useAudioStore.getState().audioFiles[c.sourcePath];
  const path = af?.playbackPath ?? c.sourcePath;
  const windowStartMs = c.offsetMs;
  // the visible window's length in SOURCE ms is the tick width divided by the stretch factor
  const windowEndMs = c.offsetMs + ticksToMs(seg.durationTicks, p.tempo) / (c.stretch ?? 1);

  inFlight.add(segmentId);
  try {
    const res = await invoke<AnalyzeResult>("analyze_segment_tempo", {
      path,
      windowStartMs,
      windowEndMs,
      beatsPerBar,
    });
    // stale guard (oovWatch pattern): the await may outlive an edit — re-read and compare
    const p2 = useProjectStore.getState();
    const seg2 = p2.tracks.find((t) => t.id === trackId)?.segments.find((s) => s.id === segmentId);
    if (!seg2 || seg2.content.type !== "audioClip") return;
    const c2 = seg2.content;
    if (`${getLoadEpoch()}:${c2.sourcePath}:${c2.offsetMs}:${seg2.durationTicks}:${c2.stretch ?? 1}:${p2.tempo}` !== winSig) return;
    p2.setSegmentTempoDetect(trackId, segmentId, {
      bpm: res.bpm,
      anchorMs: res.grid_anchor_ms,
      downbeat: res.downbeat_index % Math.max(1, beatsPerBar),
      conf: res.confidence,
      ...(res.not_constant ? { notConstant: true as const } : {}),
    });
    if (res.not_constant) useAppStore.getState().showToast(i18n.t("tempo.notConstant"), "info");
  } catch (e) {
    useAppStore.getState().showToast(tempoErrorMessage(e));
  } finally {
    inFlight.delete(segmentId);
  }
}

/** ×2: beats double in density; the anchor stays on-grid and the downbeat MOMENT is preserved
 *  (old beat d = new beat 2d). The phase is stored mod (2·bpb) — NOT mod bpb — so a following ÷2
 *  recovers the ORIGINAL downbeat class (floor(2d/2)=d); reducing here lost the parity and a
 *  ×2→÷2 round trip shifted bar lines by half a bar (audit). Draw/nudge reduce mod bpb anyway. */
export function doubleTempoDetect(td: TempoDetect, beatsPerBar: number): TempoDetect | null {
  const bpm = td.bpm * 2;
  if (bpm > TEMPO_BPM_MAX) return null;
  const bpb = Math.max(1, beatsPerBar);
  return { ...td, bpm, downbeat: (td.downbeat * 2) % (2 * bpb) };
}

/** ÷2: keep every second beat. When the downbeat sits on an ODD old beat, the anchor must shift
 *  by one old period so the downbeat moment stays ON the new grid. Returns null out of range. */
export function halveTempoDetect(td: TempoDetect, beatsPerBar: number): TempoDetect | null {
  const bpm = td.bpm / 2;
  if (bpm < TEMPO_BPM_MIN) return null;
  const bpb = Math.max(1, beatsPerBar);
  const parity = td.downbeat % 2;
  return {
    ...td,
    bpm,
    anchorMs: td.anchorMs + parity * (60000 / td.bpm),
    downbeat: Math.floor(td.downbeat / 2) % bpb,
  };
}

/** Shift which grid beat is bar-beat 1 by +1 (cyclic) — the pragmatic downbeat correction. */
export function nudgeDownbeat(td: TempoDetect, beatsPerBar: number): TempoDetect {
  const bpb = Math.max(1, beatsPerBar);
  return { ...td, downbeat: (td.downbeat + 1) % bpb };
}

/** Stable-CODE → localized message (VOCAL_NO_VOICE pattern). Covers BOTH the detection codes
 *  (TEMPO_*) and the time-stretch codes (STRETCH_*) — the stretch panel and playback degrade
 *  paths share this single mapping (NO-dup). */
export function tempoErrorMessage(e: unknown): string {
  const msg = e instanceof Error ? e.message : String(e);
  if (msg.includes("TEMPO_TOO_SHORT")) return i18n.t("tempo.errTooShort");
  if (msg.includes("TEMPO_NO_BEAT")) return i18n.t("tempo.errNoBeat");
  if (msg.includes("STRETCH_INPUT_MISSING")) return i18n.t("tempo.errStretchMissing");
  if (msg.includes("STRETCH_")) return `${i18n.t("tempo.errStretchFailed")}: ${msg}`;
  return `${i18n.t("tempo.errFailed")}: ${msg}`;
}
