// rangeTest.ts — S60-2 音域测试 (the v1 session20 recipe, frontend-orchestrated).
//
// The scale renders through the EXISTING `render_vocal_segment` command (the ONE render
// source — no second Rust render path to drift), f0 is measured by the EXISTING `detect_f0`
// (rmvpe), and only the classification lives here (pure functions, vitest-covered). The
// record persists into the model's sidecar via `set_model_vocal_range` and is read back by
// the Rust render layer (inference/vocal_range.rs) for the three-tier shift.
//
// v1 criteria (session20, verbatim): usable = median |err| < 100¢ AND voiced > 50%;
// comfort = median |err| < 50¢ AND voiced > 80%. Sweep = every semitone C2..C8 (MIDI 36-96),
// 1/16 @ 120 bpm (= 6 frames @ 50 fps) 「あ」 notes with equal rests between. Ranges are the
// longest CONTIGUOUS runs (comfort within usable). v1's numbers are per-model — never reuse.

import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import i18n from "../../i18n";
import { useAppStore } from "../../store/app";
import { useVoiceModelStore, type VoiceType } from "../../store/voice-models";
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

/** Longest contiguous true-run of `flag` over the stats (ties → the first). */
function longestRun(stats: SemitoneStat[], flag: (s: SemitoneStat) => boolean): [number, number] | null {
  let best: [number, number] | null = null;
  let start = -1;
  for (let i = 0; i <= stats.length; i++) {
    const ok = i < stats.length && flag(stats[i]!);
    if (ok && start < 0) start = i;
    if (!ok && start >= 0) {
      if (!best || i - 1 - start > best[1] - best[0]) best = [start, i - 1];
      start = -1;
    }
  }
  return best === null ? null : [stats[best[0]]!.midi, stats[best[1]]!.midi];
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
    if (msg.includes(VOCAL_RENDER_BUSY)) useAppStore.getState().showToast(t("rangeTest.busy"), "info");
    else useAppStore.getState().showToast(`${t("rangeTest.failed")}: ${msg}`, "error");
  } finally {
    unlisten();
    useVoiceModelStore.getState().setRangeTesting(name, null);
    useAppStore.getState().setVocalRenderActive(false);
  }
}

/** Persist a user-adjusted comfort range (clamped inside usable; the v1 dual-slider semantics). */
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
  const lo = Math.max(sp.usable[0], Math.min(comfort[0], comfort[1]));
  const hi = Math.min(sp.usable[1], Math.max(comfort[0], comfort[1]));
  const merged = {
    speakers: { ...existing!.speakers, [String(speakerId)]: { ...sp, comfort: [lo, hi] as [number, number] } },
  };
  await invoke("set_model_vocal_range", { name, modelType: backend, record: merged });
  await useVoiceModelStore.getState().fetchModels();
}
