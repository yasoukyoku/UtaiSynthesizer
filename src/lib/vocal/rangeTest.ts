// rangeTest.ts — S60-2 音域测试 (the v1 session20 recipe, frontend-orchestrated).
//
// The scale renders through the EXISTING `render_vocal_segment` command (the ONE render
// source — no second Rust render path to drift), f0 is measured by the EXISTING `detect_f0`
// (rmvpe), and only the classification lives here (pure functions, vitest-covered). The
// record persists into the model's sidecar via `set_model_vocal_range` and is read back by
// the Rust render layer (inference/vocal_range.rs) for the three-tier shift.
//
// v1 criteria (session20, verbatim): usable = median |err| < 100¢ AND voiced > 50%;
// comfort = median |err| < 50¢ AND voiced > 80%. Sweep = every semitone C2..C7 (MIDI 36-96),
// 1/16 @ 120 bpm (= 6 frames @ 50 fps) 「あ」 notes with equal rests between. Ranges are the
// longest CONTIGUOUS runs (comfort within usable). v1's numbers are per-model — never reuse.

import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import i18n from "../../i18n";
import { backendErrorMessage, isBusyError, isCancelError } from "../backendError";
import { useAppStore } from "../../store/app";
import { useVoiceModelStore, MIN_COMFORT_SPAN, type VoiceType } from "../../store/voice-models";
import { SOVITS_DEFAULTS, RVC_DEFAULTS } from "../workflow/voiceDefaults";
import { VOCAL_RENDER_BUSY, type ScoreTriple } from "./vocalRender";

const t = (k: string) => i18n.t(k);

export const RANGE_MIDI_LO = 36; // C2
export const RANGE_MIDI_HI = 96; // C8
const NOTE_FRAMES = 6; // 1/16 @ 120 bpm @ 50 fps
const REST_FRAMES = 6;
/** rmvpe legally disagrees by octaves on note-edge frames — erode each note's measured span. */
const EDGE_ERODE_100FPS = 2;

export interface SemitoneStat {
  midi: number;
  /** median |cents error| over voiced frames (Infinity when nothing voiced). */
  errCents: number;
  voicedRatio: number;
}

export interface SpeakerRangeRecord {
  usable: [number, number];
  comfort: [number, number];
  comfort_auto: [number, number];
  /** midi → [errCents, voicedRatio] raw scan (Reset + UI display re-derive from this). */
  semitones: Record<string, [number, number]>;
  tested_at: string;
}

// ── pure pieces (vitest) ──────────────────────────────────────────────────────

/** The scale score: leading rest, then per semitone 「あ」(6f) + rest(6f). Spans are each
 *  note's frame window at 100 fps (2× the 50 fps grid) for the detect_f0 alignment. */
export function buildScaleScore(): { triples: ScoreTriple[]; spans: { midi: number; start100: number; end100: number }[] } {
  const triples: ScoreTriple[] = [{ lyric: "R", note_num: 0, frames: REST_FRAMES, lang: 2 }];
  const spans: { midi: number; start100: number; end100: number }[] = [];
  let cursor50 = REST_FRAMES;
  for (let midi = RANGE_MIDI_LO; midi <= RANGE_MIDI_HI; midi++) {
    triples.push({ lyric: "あ", note_num: midi, frames: NOTE_FRAMES, lang: 2 });
    spans.push({ midi, start100: cursor50 * 2, end100: (cursor50 + NOTE_FRAMES) * 2 });
    cursor50 += NOTE_FRAMES;
    triples.push({ lyric: "R", note_num: 0, frames: REST_FRAMES, lang: 2 });
    cursor50 += REST_FRAMES;
  }
  return { triples, spans };
}

export function midiToHz(midi: number): number {
  return 440 * Math.pow(2, (midi - 69) / 12);
}

/** Per-semitone stats from the rendered scale's rmvpe track (100 fps Hz, unvoiced = 0). */
export function classifySemitones(
  f0: number[],
  spans: { midi: number; start100: number; end100: number }[],
): SemitoneStat[] {
  return spans.map(({ midi, start100, end100 }) => {
    const a = start100 + EDGE_ERODE_100FPS;
    const b = Math.min(end100 - EDGE_ERODE_100FPS, f0.length);
    const window = a < b ? f0.slice(a, b) : [];
    const voiced = window.filter((v) => v > 0);
    const voicedRatio = window.length ? voiced.length / window.length : 0;
    if (!voiced.length) return { midi, errCents: Infinity, voicedRatio };
    const expected = midiToHz(midi);
    const errs = voiced.map((v) => Math.abs(1200 * Math.log2(v / expected))).sort((x, y) => x - y);
    return { midi, errCents: errs[Math.floor(errs.length / 2)]!, voicedRatio };
  });
}

const isUsable = (s: SemitoneStat) => s.errCents < 100 && s.voicedRatio > 0.5;
const isComfort = (s: SemitoneStat) => s.errCents < 50 && s.voicedRatio > 0.8;

/** Isolated measurement dropouts (rmvpe octave flips, a one-note synthesis hiccup) are 1-2
 *  semitones wide with clean passes on BOTH sides; true out-of-range failure is contiguous
 *  (saturation errs grow monotonically). Without bridging, one octave-flipped note truncates
 *  the whole ceiling (S60d: lengv2.3 lost 57–77 to a single 1180¢ point at 57). */
const BRIDGE_MAX_GAP = 2;

/** Minimum comfort span the UI lets the user commit — the constant now lives in
 *  voice-models.ts (the range-record gate needs it and rangeTest already imports that store;
 *  re-exported here so existing consumers keep their import path). */
export { MIN_COMFORT_SPAN };

/** Pass-flags with interior fail-gaps of ≤ BRIDGE_MAX_GAP (flanked by passes) bridged. */
function bridgedFlags(stats: SemitoneStat[], flag: (s: SemitoneStat) => boolean): boolean[] {
  const f = stats.map(flag);
  let i = 0;
  while (i < f.length) {
    if (f[i]) { i++; continue; }
    let j = i;
    while (j < f.length && !f[j]) j++;
    if (i > 0 && j < f.length && j - i <= BRIDGE_MAX_GAP) f.fill(true, i, j);
    i = j;
  }
  return f;
}

/** Longest contiguous true-run of `flag` over the stats (ties → the first), noise-bridged. */
function longestRun(stats: SemitoneStat[], flag: (s: SemitoneStat) => boolean): [number, number] | null {
  const flags = bridgedFlags(stats, flag);
  let best: [number, number] | null = null;
  let start = -1;
  for (let i = 0; i <= flags.length; i++) {
    const ok = i < flags.length && flags[i]!;
    if (ok && start < 0) start = i;
    if (!ok && start >= 0) {
      if (!best || i - 1 - start > best[1] - best[0]) best = [start, i - 1];
      start = -1;
    }
  }
  return best === null ? null : [stats[best[0]]!.midi, stats[best[1]]!.midi];
}

/** The comfort zone the RENDER layer will actually target — mirrors the Rust read-side
 *  healing in vocal_range.rs::speaker_range (degenerate comfort → comfort_auto → usable).
 *  UI display and slider seeding must show THIS, not the raw stored value. */
export function effectiveComfort(sp: SpeakerRangeRecord): [number, number] {
  const wide = (c: [number, number]) =>
    c[1] - c[0] >= MIN_COMFORT_SPAN && c[0] >= sp.usable[0] && c[1] <= sp.usable[1];
  if (wide(sp.comfort)) return sp.comfort;
  if (wide(sp.comfort_auto)) return sp.comfort_auto;
  return sp.usable;
}

/** usable = longest contiguous usable run; comfort = longest contiguous comfort run WITHIN it
 *  (falls back to the usable run when no semitone reaches comfort grade). null = model unusable. */
export function deriveRanges(stats: SemitoneStat[]): { usable: [number, number]; comfort: [number, number] } | null {
  const usable = longestRun(stats, isUsable);
  if (!usable) return null;
  const inside = stats.filter((s) => s.midi >= usable[0] && s.midi <= usable[1]);
  const comfort = longestRun(inside, isComfort) ?? usable;
  return { usable, comfort };
}

export function buildSpeakerRecord(stats: SemitoneStat[]): SpeakerRangeRecord | null {
  const ranges = deriveRanges(stats);
  if (!ranges) return null;
  const semitones: Record<string, [number, number]> = {};
  for (const s of stats) {
    semitones[String(s.midi)] = [Number.isFinite(s.errCents) ? Math.round(s.errCents) : 9999, Math.round(s.voicedRatio * 100) / 100];
  }
  return {
    usable: ranges.usable,
    comfort: ranges.comfort,
    comfort_auto: ranges.comfort,
    semitones,
    tested_at: new Date().toISOString().slice(0, 10),
  };
}

export interface CautionZones {
  /** Contiguous runs (≥2 st, within an octave of usable) where the model SINGS but lands
   *  ≥200¢ off pitch — "confidently wrong" model artifacts (S60d: 風音サヨ 71–75 at
   *  1223–2410¢ with full voicing). Labeled so a weird render reads as a model quirk,
   *  not a program/algorithm bug. */
  artifact: [number, number][];
  /** Isolated weak notes INSIDE usable (failed the probe but bridged over when deriving
   *  the range) — the exact "谨慎使用" notes. */
  weak: number[];
}

/** Model-quirk annotation derived from the STORED per-semitone scan — no new measurement.
 *  Takes the sidecar's raw `semitones` map (midi → [errCents, voicedRatio]). */
export function deriveCautionZones(
  semitones: Record<string, [number, number]>,
  usable: [number, number],
): CautionZones {
  const stats: SemitoneStat[] = Object.entries(semitones)
    .map(([k, v]) => ({ midi: Number(k), errCents: v[0], voicedRatio: v[1] }))
    .sort((a, b) => a.midi - b.midi);
  const weak = stats
    .filter((s) => s.midi > usable[0] && s.midi < usable[1] && !isUsable(s))
    .map((s) => s.midi);
  // voiced but far off pitch, outside usable yet within an octave of it (past that
  // everything fails anyway — the label stays musically relevant)
  const singsWrong = (s: SemitoneStat) =>
    s.voicedRatio > 0.5 &&
    s.errCents >= 200 &&
    s.errCents < 9999 &&
    (s.midi < usable[0] || s.midi > usable[1]) &&
    s.midi >= usable[0] - 12 &&
    s.midi <= usable[1] + 12;
  const artifact: [number, number][] = [];
  let run: [number, number] | null = null;
  for (const s of stats) {
    if (!singsWrong(s)) continue;
    if (run && s.midi === run[1] + 1) {
      run[1] = s.midi;
    } else {
      if (run && run[1] > run[0]) artifact.push(run);
      run = [s.midi, s.midi];
    }
  }
  if (run && run[1] > run[0]) artifact.push(run);
  return { artifact, weak };
}

/** MIDI → note name (C4 = 60) for the range labels. */
export function midiName(midi: number): string {
  const names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
  return `${names[((midi % 12) + 12) % 12]}${Math.floor(midi / 12) - 1}`;
}

// ── orchestration ─────────────────────────────────────────────────────────────

let runSeq = 0;

/** Run the scale test for one (model, speaker) and persist the record. Fire-and-forget with
 *  progress in the voice-models store; single-flight per model; shares the vocal render
 *  guard (VoiceRunGuard Rust-side + vocalRenderActive UI-side). */
export async function runRangeTest(
  name: string,
  backend: Exclude<VoiceType, "vocoder">,
  modelPath: string,
  speakerId = 0,
): Promise<void> {
  const store = useVoiceModelStore.getState();
  if (store.rangeTesting[name] !== undefined) return;
  if (useAppStore.getState().vocalRenderActive) {
    useAppStore.getState().showToast(t("rangeTest.busy"), "info");
    return;
  }
  useAppStore.getState().setVocalRenderActive(true);
  useVoiceModelStore.getState().setRangeTesting(name, 0);
  const nodeId = `range-test:${name}:${runSeq++}`;
  const unlisten = await listen<{ node_id: string; progress: number }>("voice-progress", (e) => {
    if (e.payload.node_id === nodeId) {
      useVoiceModelStore.getState().setRangeTesting(name, e.payload.progress * 0.85);
    }
  });
  try {
    const { triples, spans } = buildScaleScore();
    const result = await invoke<{ audio: number[]; sample_rate: number }>("render_vocal_segment", {
      voiceName: name,
      modelPath,
      nodeId,
      score: triples,
      f0Cents: [],
      f0Voiced: [],
      loudnessEnv: [],
      formantEnv: [],
      options: {
        backend,
        cv_speaker_id: 49,
        lang_id: 2,
        transpose: 0,
        range_extend: false, // measuring the RAW model — never shift the probe itself
        sovits: { ...SOVITS_DEFAULTS, speaker_id: speakerId },
        rvc: { ...RVC_DEFAULTS, speaker_id: speakerId },
      },
    });
    useVoiceModelStore.getState().setRangeTesting(name, 0.88);
    const dir = (await invoke<string>("ensure_cache_dir", { segmentId: "range_test" })).replace(/\\/g, "/");
    const wavPath = `${dir}/scale_${Date.now().toString(36)}.wav`;
    await invoke("save_temp_audio", { samples: result.audio, sampleRate: result.sample_rate, outputPath: wavPath });
    const f0 = await invoke<number[]>("detect_f0", { audioPath: wavPath });
    useVoiceModelStore.getState().setRangeTesting(name, 0.96);

    const record = buildSpeakerRecord(classifySemitones(f0, spans));
    if (!record) {
      useAppStore.getState().showToast(t("rangeTest.noUsable"), "error");
      return;
    }
    // merge into the existing record (other speakers' entries survive)
    const entry = useVoiceModelStore.getState().models[backend]?.find((m) => m.name === name);
    const existing = (entry?.config as { vocal_range?: { speakers?: Record<string, unknown> } } | undefined)?.vocal_range;
    const merged = { speakers: { ...(existing?.speakers ?? {}), [String(speakerId)]: record } };
    await invoke("set_model_vocal_range", { name, modelType: backend, record: merged });
    await useVoiceModelStore.getState().fetchModels();
    useAppStore.getState().showToast(
      `${t("rangeTest.done")} ${midiName(record.comfort[0])}–${midiName(record.comfort[1])}`,
      "success",
    );
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    const shared = backendErrorMessage(e); // app-wide CODEs (APP_BUSY from the VoiceRunGuard, …)
    if (isCancelError(e)) { /* user cancelled — not an error, no toast */ }
    else if (msg.includes(VOCAL_RENDER_BUSY)) useAppStore.getState().showToast(t("rangeTest.busy"), "info");
    else if (shared) useAppStore.getState().showToast(shared, isBusyError(e) ? "info" : "error");
    else useAppStore.getState().showToast(`${t("rangeTest.failed")}: ${msg}`, "error");
  } finally {
    unlisten();
    useVoiceModelStore.getState().setRangeTesting(name, null);
    useAppStore.getState().setVocalRenderActive(false);
  }
}

/** S60c: one training CANDIDATE's range test — the scale renders through render_candidate_scale
 *  (converts on demand; Rust FlightGuard = single-flight), measurement + classification reuse the
 *  SAME pure functions as the installed-model test (single source), and the record persists into
 *  the candidate's audition sidecar (the audition render pre-shifts with it). Speaker 0 only.
 *  Throws on failure (caller decides whether to toast or skip silently). */
export async function runCandidateRangeTest(
  workspace: string,
  backend: "rvc" | "sovits",
  ckptPath: string,
  candidateId: string,
): Promise<{ usable: [number, number]; comfort: [number, number] } | null> {
  const { triples, spans } = buildScaleScore();
  const result = await invoke<{ audio: number[]; sample_rate: number }>("render_candidate_scale", {
    backend,
    ckptPath,
    workspace,
    candidateId,
    score: triples,
  });
  const dir = (await invoke<string>("ensure_cache_dir", { segmentId: "range_test" })).replace(/\\/g, "/");
  const wavPath = `${dir}/cand_${Date.now().toString(36)}.wav`;
  await invoke("save_temp_audio", { samples: result.audio, sampleRate: result.sample_rate, outputPath: wavPath });
  const f0 = await invoke<number[]>("detect_f0", { audioPath: wavPath });
  const record = buildSpeakerRecord(classifySemitones(f0, spans));
  if (!record) return null; // nothing usable — an undertrained checkpoint; audition stays unshifted
  await invoke("set_candidate_vocal_range", { workspace, ckptPath, record: { speakers: { "0": record } } });
  return { usable: record.usable, comfort: record.comfort };
}

/** Clamp a requested comfort pair into `usable` and enforce MIN_COMFORT_SPAN (expanding
 *  around the requested low bound; a usable zone narrower than the minimum becomes the whole
 *  usable zone). Pure — the single source for commit-time comfort sanitation (vitest). */
export function clampComfort(usable: [number, number], comfort: [number, number]): [number, number] {
  let lo = Math.max(usable[0], Math.min(usable[1], Math.min(comfort[0], comfort[1])));
  let hi = Math.max(usable[0], Math.min(usable[1], Math.max(comfort[0], comfort[1])));
  if (hi - lo < MIN_COMFORT_SPAN) {
    hi = Math.min(usable[1], lo + MIN_COMFORT_SPAN);
    lo = Math.max(usable[0], hi - MIN_COMFORT_SPAN);
  }
  return [lo, hi];
}

/** Persist a user-adjusted comfort range (clamped inside usable, minimum span enforced —
 *  S60d: a degenerate committed zone centered whole songs onto a single MIDI note). */
export async function setComfortRange(
  name: string,
  backend: Exclude<VoiceType, "vocoder">,
  speakerId: number,
  comfort: [number, number],
): Promise<void> {
  const entry = useVoiceModelStore.getState().models[backend]?.find((m) => m.name === name);
  const existing = (entry?.config as { vocal_range?: { speakers?: Record<string, SpeakerRangeRecord> } } | undefined)
    ?.vocal_range;
  const sp = existing?.speakers?.[String(speakerId)];
  if (!sp) return;
  const merged = {
    speakers: { ...existing!.speakers, [String(speakerId)]: { ...sp, comfort: clampComfort(sp.usable, comfort) } },
  };
  await invoke("set_model_vocal_range", { name, modelType: backend, record: merged });
  await useVoiceModelStore.getState().fetchModels();
}
