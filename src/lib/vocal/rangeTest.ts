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
import { maybeShowErrorModal } from "../errorDisplay";
import { logToBackend } from "../log";
import { useAppStore } from "../../store/app";
import {
  useVoiceModelStore,
  voiceSpeakerOptions,
  MIN_COMFORT_SPAN,
  type VoiceModelEntry,
  type VoiceType,
} from "../../store/voice-models";
import { SOVITS_DEFAULTS, RVC_DEFAULTS } from "../workflow/voiceDefaults";
import { VOCAL_RENDER_BUSY, type ScoreTriple } from "./vocalRender";
import { boundsPayload, type RangeBoundsEdit } from "./rangeBounds";

const t = (k: string) => i18n.t(k);

export const RANGE_MIDI_LO = 36; // C2
export const RANGE_MIDI_HI = 96; // C8
/** S81 F1: 400 ms per note (was 120 ms = 6 frames). The old probe never reached steady state —
 *  for a vol_embedding model the whole note sat inside the phrase ADSR ramp and peaked at
 *  0.8875x, so the hardest condition for a high note (SUSTAINED, at full level) was never
 *  measured. Everything downstream reads spans, so this is the only place the duration lives. */
const NOTE_FRAMES = 20; // 400 ms @ 50 fps
const REST_FRAMES = 6;
/** S146b — the second pass. The 400 ms 「あ」 above is the EASIEST thing a model ever sings: no
 *  onset consonant, and long enough to reach steady state (which is exactly why S81 lengthened
 *  it — that reason still stands, so this is an ADDITIONAL pass, not a replacement).
 *  What it cannot see is the attack. Measured on akiko at MIDI 80 — same pitch, same duration,
 *  only the onset differs: 「ま」(voiced) −0.2 dB / voiced 1.00 vs 「か」(voiceless) −7.9 dB /
 *  voiced 0.33. The song agrees at scale (unrescued MIDI 80, n=45: voiceless onset 64% unvoiced,
 *  voiced 23%, none 8%; the entire low register 0/548). Only the LANDING grade reads this pass —
 *  a rescue must not aim at a slot that only survives when nothing has to start it. */
const ONSET_PROBE_LYRIC = "か";
const ONSET_PROBE_FRAMES = 9; // 180 ms — the duration the failing notes in real songs have
/** rmvpe legally disagrees by octaves on note-edge frames — erode each note's measured span. */
const EDGE_ERODE_100FPS = 2;

export interface SemitoneStat {
  midi: number;
  /** median |cents error| over voiced frames (Infinity when nothing voiced). */
  errCents: number;
  voicedRatio: number;
  /** S81 F1 — measured from the AUDIO (analyze_scale_quality), absent on pre-S81 records:
   *  level relative to this scale's loudest note (dB, ≤ 0). */
  rmsDb?: number;
  /** Energy below 1.5*f0 over total (to 8 kHz). A sung vowel spreads across harmonics
   *  (~0.1-0.55); a collapsed one is a near-pure sine (0.9+). THE dimension the f0-only
   *  criteria are structurally blind to. */
  lowRatio?: number;
}

export interface SpeakerRangeRecord {
  usable: [number, number];
  comfort: [number, number];
  comfort_auto: [number, number];
  /** S146e — 扫描量出来的可用域,与 `comfort_auto` 对称。缺列 = S146e 之前的记录,那时
   *  `usable` 就**是**扫描的答案(没有任何 UI 改得动它)⇒ 读侧拿 `usable` 顶替。
   *  存在的理由:可用上界成了一个用户要来回调的旋钮,没有它就调不回去。 */
  usable_auto?: [number, number];
  /** midi → [errCents, voicedRatio] (pre-S81) or [errCents, voicedRatio, rmsDb, lowRatio]
   *  (S81+). Readers MUST tolerate the 2-tuple: old records stay usable by design, they just
   *  cannot contribute the timbre dimension (§user 2026-07-25). */
  semitones: Record<string, [number, number] | [number, number, number, number]>;
  /** S146b — the SECOND probe pass, shaped like real singing (short note + voiceless onset):
   *  midi → [errCents, voicedRatio]. Absent on every pre-S146b record, and the Rust reader
   *  treats absent as "no veto", so old sidecars decide exactly as they did.
   *  ⛔ It may only ever NARROW the landing band (see vocal_range.rs::speaker_range) — the dead
   *  set is decided by `semitones` alone, so a second pass can never rescue an extra note. */
  semitones_onset?: Record<string, [number, number]>;
  tested_at: string;
  /** Scan format/probe version. Absent = pre-S81 (120 ms probe, f0-only). Bump whenever the
   *  probe conditions or the stored dimensions change, so the UI can say "worth re-testing"
   *  instead of silently mixing incomparable measurements. */
  scan_version?: number;
}

/** Current probe + scan format.
 *  2 = S81 F1: 400 ms probe + the timbre pair (rmsDb, lowRatio).
 *  3 = S146b: a SECOND pass with a singing-shaped probe (「か」, 180 ms, voiceless onset) whose
 *      readings land in `semitones_onset` and may veto unsafe LANDING slots. Every record made
 *      before this carries no such pass, so it keeps the 「あ」-only landing verdict — correct,
 *      but it means the fix does not reach a model until it is re-tested. Bumping is how the user
 *      finds that out: `MsstModelManager` marks `scan_version < SCAN_VERSION` as stale and the
 *      batch collector picks those models up. */
export const SCAN_VERSION = 3;

// ── pure pieces (vitest) ──────────────────────────────────────────────────────

/** The scale score: leading rest, then per semitone one probe note + rest. Spans are each
 *  note's frame window at 100 fps (2× the 50 fps grid) for the detect_f0 alignment. */
export function buildScaleScore(
  lyric: string = "あ",
  frames: number = NOTE_FRAMES,
): { triples: ScoreTriple[]; spans: { midi: number; start100: number; end100: number }[] } {
  const triples: ScoreTriple[] = [{ lyric: "R", note_num: 0, frames: REST_FRAMES, lang: 2 }];
  const spans: { midi: number; start100: number; end100: number }[] = [];
  let cursor50 = REST_FRAMES;
  for (let midi = RANGE_MIDI_LO; midi <= RANGE_MIDI_HI; midi++) {
    triples.push({ lyric, note_num: midi, frames, lang: 2 });
    spans.push({ midi, start100: cursor50 * 2, end100: (cursor50 + frames) * 2 });
    cursor50 += frames;
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
  /** S81 F1: `[rmsDb, lowRatio]` per span from analyze_scale_quality. Omitted → f0-only stats,
   *  which is exactly what a pre-S81 record carries (and stays supported). */
  quality?: [number, number][],
): SemitoneStat[] {
  return spans.map(({ midi, start100, end100 }, i) => {
    const q = quality?.[i];
    const extra = q ? { rmsDb: q[0], lowRatio: q[1] } : {};
    const a = start100 + EDGE_ERODE_100FPS;
    const b = Math.min(end100 - EDGE_ERODE_100FPS, f0.length);
    const window = a < b ? f0.slice(a, b) : [];
    const voiced = window.filter((v) => v > 0);
    const voicedRatio = window.length ? voiced.length / window.length : 0;
    if (!voiced.length) return { midi, errCents: Infinity, voicedRatio, ...extra };
    const expected = midiToHz(midi);
    const errs = voiced.map((v) => Math.abs(1200 * Math.log2(v / expected))).sort((x, y) => x - y);
    return { midi, errCents: errs[Math.floor(errs.length / 2)]!, voicedRatio, ...extra };
  });
}

/** Energy-below-1.5*f0 above which the note has collapsed toward a bare fundamental.
 *  Calibrated against the probe wavs on disk (S81 forensics, OLD 120 ms probe — re-check once
 *  the 400 ms probe has run): akiko MIDI 74 = 0.109 and 78 = 0.549 sound fine, 79 = 0.747 and
 *  80 = 0.887 are the audibly dead ones; lengv2.3 74 = 0.467 is fine while 75 = 0.983 and
 *  76 = 0.927 are near-pure sines. 0.70 separates those two populations with margin on both
 *  sides. Deliberately a COMFORT gate only — see below. */
const COMFORT_MAX_LOW_RATIO = 0.7;

/** "Can this model produce a pitched sound here at all" — the FLOOR, and the label the range
 *  chips show. Left at the v1 thresholds on purpose: the decision layer no longer trusts it
 *  (with a scan present it weighs every frame by measured damage, where voiced=0.5 already
 *  scores maximum), and tightening it would silently move the tier-2 short-circuit, the
 *  phantom-island mask and every displayed range at once. */
const isUsable = (s: SemitoneStat) => s.errCents < 100 && s.voicedRatio > 0.5;

/** "Does this model sound GOOD here" — the zone the render actually aims at, so it has to mean
 *  something. The f0 pair alone could never say: across a model's whole healthy span the scan
 *  reads err 1-37 cents and voiced 1.00, so those axes do not discriminate — they are binary
 *  (sings / doesn't). The timbre term is what makes comfort comfortable.
 *  Level is NOT gated here: a model is simply quieter at the edges of its range and a hard dB
 *  cut would punch holes everywhere; it feeds the continuous damage curve instead. */
const isComfort = (s: SemitoneStat) =>
  s.errCents < 50 &&
  s.voicedRatio > 0.8 &&
  (s.lowRatio === undefined || s.lowRatio < COMFORT_MAX_LOW_RATIO);

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
  const semitones: SpeakerRangeRecord["semitones"] = {};
  let measured = false;
  for (const s of stats) {
    const err = Number.isFinite(s.errCents) ? Math.round(s.errCents) : 9999;
    const voiced = Math.round(s.voicedRatio * 100) / 100;
    // 4-tuple only when the audio was actually analysed, so a record's SHAPE tells readers
    // whether the timbre dimension exists rather than storing a fabricated default.
    if (s.rmsDb !== undefined && s.lowRatio !== undefined) {
      measured = true;
      semitones[String(s.midi)] = [
        err,
        voiced,
        Math.round(s.rmsDb * 10) / 10,
        Math.round(s.lowRatio * 1000) / 1000,
      ];
    } else {
      semitones[String(s.midi)] = [err, voiced];
    }
  }
  return {
    usable: ranges.usable,
    comfort: ranges.comfort,
    comfort_auto: ranges.comfort,
    // S146e: the scan's own answer, kept so the user's usable knob has a "还原" to go back to.
    // A re-test overwrites it, which is right — it IS the new measurement.
    usable_auto: ranges.usable,
    semitones,
    tested_at: new Date().toISOString().slice(0, 10),
    ...(measured ? { scan_version: SCAN_VERSION } : {}),
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
  semitones: SpeakerRangeRecord["semitones"],
  usable: [number, number],
  comfort?: [number, number],
): CautionZones {
  const stats: SemitoneStat[] = Object.entries(semitones)
    .map(([k, v]) => ({
      midi: Number(k),
      errCents: v[0],
      voicedRatio: v[1],
      // 2-tuple (pre-S81) leaves these undefined, which every criterion treats as "unknown,
      // don't judge" rather than "fine" — old records keep their old verdicts exactly.
      ...(v.length >= 4 ? { rmsDb: v[2], lowRatio: v[3] } : {}),
    }))
    .sort((a, b) => a.midi - b.midi);
  // A weak point is a semitone the bridging carried INTO a zone even though it failed that
  // zone's own criterion. Reporting only the usable-grade ones (pre-S81) missed the case that
  // actually matters: comfort is what the render AIMS AT, so a semitone bridged into comfort
  // while failing the comfort bar is a note we deliberately transpose material onto and then
  // label a defect — the program contradicting itself. Both grades are reported now.
  // (Endpoints cannot be weak: bridgedFlags only fills gaps flanked by genuine passes.)
  const weak = stats
    .filter(
      (s) =>
        (s.midi > usable[0] && s.midi < usable[1] && !isUsable(s)) ||
        (comfort !== undefined &&
          s.midi > comfort[0] &&
          s.midi < comfort[1] &&
          !isComfort(s)),
    )
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
  // S146b: two probe passes now share this node id, so the bar maps each render into its own
  // window instead of replaying 0→0.85 twice (a bar that visibly restarts reads as "it hung and
  // started over" — the same观感 complaint S85e's windowed donors produced).
  const win = { lo: 0, hi: 0.5 };
  const unlisten = await listen<{ node_id: string; progress: number }>("voice-progress", (e) => {
    if (e.payload.node_id === nodeId) {
      useVoiceModelStore
        .getState()
        .setRangeTesting(name, win.lo + e.payload.progress * (win.hi - win.lo));
    }
  });
  try {
    // S66/O5: the render writes the probe wav Rust-side — compute the path up front.
    const dir = (await invoke<string>("ensure_cache_dir", { segmentId: "range_test" })).replace(/\\/g, "/");

    /** One probe pass: render the scale, measure it, classify. The two passes differ ONLY in the
     *  probe note (lyric + length) — same render command, same f0 detector, same analyzer, so
     *  they can never disagree about anything except what was sung. */
    const pass = async (
      lyric: string,
      frames: number,
      lo: number,
      hi: number,
      withQuality: boolean,
    ): Promise<SemitoneStat[]> => {
      const { triples, spans } = buildScaleScore(lyric, frames);
      win.lo = lo;
      win.hi = hi;
      const wavPath = `${dir}/scale_${Date.now().toString(36)}.wav`;
      await invoke<{ path: string; sample_rate: number }>("render_vocal_segment", {
        voiceName: name,
        modelPath,
        nodeId,
        score: triples,
        f0Cents: [],
        f0Voiced: [],
        loudnessEnv: [],
        formantEnv: [],
        outputPath: wavPath,
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
      const f0 = await invoke<number[]>("detect_f0", { audioPath: wavPath });
      // S81 F1: the timbre dimension the f0 criteria are structurally blind to. Measured from the
      // SAME probe wav, so it can never disagree with the f0 stats about what was rendered.
      // The onset pass does not need it — it only ever feeds the f0-axes LANDING veto.
      const quality = withQuality
        ? await invoke<[number, number][]>("analyze_scale_quality", {
            audioPath: wavPath,
            spans: spans.map((s) => [s.start100, s.end100]),
            expectedHz: spans.map((s) => midiToHz(s.midi)),
          })
        : undefined;
      return classifySemitones(f0, spans, quality);
    };

    const record = buildSpeakerRecord(await pass("あ", NOTE_FRAMES, 0, 0.5, true));
    if (!record) {
      useAppStore.getState().showToast(t("rangeTest.noUsable"), "error");
      return;
    }
    // S146b second pass — shaped like real singing. It can only ever narrow the landing band
    // (the Rust reader never SETS the bit from this map), so a failure here is not worth aborting
    // a finished scan over: the record simply keeps the pre-S146b landing verdict.
    try {
      const onset = await pass(ONSET_PROBE_LYRIC, ONSET_PROBE_FRAMES, 0.5, 0.95, false);
      const map: NonNullable<SpeakerRangeRecord["semitones_onset"]> = {};
      for (const st of onset) {
        map[String(st.midi)] = [
          Number.isFinite(st.errCents) ? Math.round(st.errCents) : 9999,
          Math.round(st.voicedRatio * 100) / 100,
        ];
      }
      record.semitones_onset = map;
    } catch (e) {
      void logToBackend(
        "warn",
        `range test: onset probe pass failed, keeping the 「あ」-only landing verdict: ${String(e)}`,
      );
    }
    useVoiceModelStore.getState().setRangeTesting(name, 0.97);

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
    // S74: this invokes the SAME render_vocal_segment as vocal render, so it can surface the
    // identical raw ONNX inference failures (Greater/ReduceSum) — log them so a report always
    // leaves a copyable trace in the log file. Once per test (never in a loop). Cancel skipped.
    if (!isCancelError(e)) logToBackend(isBusyError(e) ? "warn" : "error", `Range test failed (${name}): ${msg}`);
    if (isCancelError(e)) { /* user cancelled — not an error, no toast */ }
    else if (msg.includes(VOCAL_RENDER_BUSY)) useAppStore.getState().showToast(t("rangeTest.busy"), "info");
    // S67c: fatal modal-class errors (INFERENCE_LOW_MEMORY) open the alert dialog instead of a toast.
    else if (shared && maybeShowErrorModal(e, shared)) { /* modal shown */ }
    else if (shared) useAppStore.getState().showToast(shared, isBusyError(e) ? "info" : "error");
    else useAppStore.getState().showToast(`${t("rangeTest.failed")}: ${msg}`, "error");
  } finally {
    unlisten();
    useVoiceModelStore.getState().setRangeTesting(name, null);
    useAppStore.getState().setVocalRenderActive(false);
  }
}

/** One unit of work for the batch: a (model, speaker) whose range should be measured. */
export interface RangeTestTarget {
  name: string;
  backend: Exclude<VoiceType, "vocoder">;
  path: string;
  speakerId: number;
  /** Why it is listed — drives the "N 个待更新" count and nothing else. */
  reason: "missing" | "stale";
}

/** Every (model, speaker) that has no record, or one measured before the timbre dimension.
 *  PURE over the store snapshot so the count in the UI and the work the batch does can never
 *  disagree (they call this same function). */
export function collectRangeTestTargets(models: {
  rvc?: VoiceModelEntry[];
  sovits?: VoiceModelEntry[];
}): RangeTestTarget[] {
  const out: RangeTestTarget[] = [];
  for (const backend of ["sovits", "rvc"] as const) {
    for (const m of models[backend] ?? []) {
      const rec = (m.config as { vocal_range?: { speakers?: Record<string, SpeakerRangeRecord> } } | undefined)
        ?.vocal_range;
      // a single-speaker model is speaker 0 only; voiceSpeakerOptions returns [] for those
      const opts = voiceSpeakerOptions(m);
      const ids = opts.length > 1 ? opts.map((s) => s.id) : [0];
      for (const id of ids) {
        const sp = rec?.speakers?.[String(id)];
        if (!sp) out.push({ name: m.name, backend, path: m.path, speakerId: id, reason: "missing" });
        else if ((sp.scan_version ?? 0) < SCAN_VERSION)
          out.push({ name: m.name, backend, path: m.path, speakerId: id, reason: "stale" });
      }
    }
  }
  return out;
}

/** Measure a whole list of (model, speaker) targets, one at a time.
 *
 *  ★ THE invariant: `runRangeTest` takes and RELEASES the vocal-render guard per target, and
 *  this loop deliberately awaits between targets instead of holding anything across them. A
 *  batch that held the guard for its whole run would refuse the user's Play button for minutes,
 *  which is exactly the kind of long silent lock this codebase has been bitten by before.
 *  Cancellation is checked between targets — a target already in flight finishes (the Rust side
 *  owns its own cancel path); we never abandon a half-written record. */
export async function runRangeTestBatch(targets: RangeTestTarget[]): Promise<void> {
  const store = useVoiceModelStore.getState();
  if (store.rangeBatch && !store.rangeBatch.finished) return; // single-flight (a finished-with-failures summary may be replaced)
  useVoiceModelStore
    .getState()
    .setRangeBatch({ done: 0, total: targets.length, failed: [], cancel: false, finished: false });
  for (const tgt of targets) {
    if (useVoiceModelStore.getState().rangeBatch?.cancel) break;
    const before = useVoiceModelStore
      .getState()
      .models[tgt.backend]?.find((m) => m.name === tgt.name);
    await runRangeTest(tgt.name, tgt.backend, tgt.path, tgt.speakerId);
    // runRangeTest swallows its own errors (it toasts), so detect failure by asking whether the
    // record actually moved — a silent "0 failed" summary after a broken run would be a lie.
    const after = useVoiceModelStore.getState().models[tgt.backend]?.find((m) => m.name === tgt.name);
    const ok = JSON.stringify(readRangeRecord(after, tgt.speakerId)) !== JSON.stringify(readRangeRecord(before, tgt.speakerId));
    useVoiceModelStore.getState().bumpRangeBatch(ok ? null : `${tgt.name}${tgt.speakerId ? ` (${tgt.speakerId})` : ""}`);
  }
  useVoiceModelStore.getState().finishRangeBatch();
}

function readRangeRecord(m: VoiceModelEntry | undefined, speakerId: number): SpeakerRangeRecord | undefined {
  return (m?.config as { vocal_range?: { speakers?: Record<string, SpeakerRangeRecord> } } | undefined)
    ?.vocal_range?.speakers?.[String(speakerId)];
}

/** S60c: one training CANDIDATE's range test — the scale renders through render_candidate_scale
 *  (converts on demand; Rust FlightGuard = single-flight), measurement + classification reuse the
 *  SAME pure functions as the installed-model test (single source), and the record persists into
 *  the candidate's audition sidecar (the audition render pre-shifts with it). Speaker 0 only.
 *  Throws on failure (caller decides whether to toast or skip silently). */
export async function runCandidateRangeTest(
  workspace: string,
  backend: "rvc" | "sovits" | "sovits_v2",
  ckptPath: string,
  candidateId: string,
): Promise<{ usable: [number, number]; comfort: [number, number] } | null> {
  const { triples, spans } = buildScaleScore();
  // S66/O5: the render writes the probe wav Rust-side — compute the path up front.
  const dir = (await invoke<string>("ensure_cache_dir", { segmentId: "range_test" })).replace(/\\/g, "/");
  const wavPath = `${dir}/cand_${Date.now().toString(36)}.wav`;
  await invoke<{ path: string; sample_rate: number }>("render_candidate_scale", {
    backend,
    ckptPath,
    workspace,
    candidateId,
    score: triples,
    outputPath: wavPath,
  });
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

/* S146e: `setComfortRange`(只写 comfort 一个边界)已被下面的 `setVocalRangeBounds` 取代并
   删除 —— 它的唯一调用点是资源管理器那个编辑器,而那个编辑器现在与人声侧栏共用
   `RangeBoundsEditor`。留着一个「只写一半」的写入口是主动的危险:收窄 usable 那一半单独
   落盘时 comfort 还在外面 ⇒ 后端 RANGE_INVALID,而那条错误在 UI 上完全不可见。 */

/** S146e — 一次写下 usable **与** comfort 两个边界。
 *
 *  ⛔ 不许拆成两次调用:后端 `validate_range_record` 要求 comfort ⊆ usable,而收窄 usable
 *  的那一半单独落盘时 comfort 还在外面 ⇒ 当场 `RANGE_INVALID`,而这条错误今天在 UI 上
 *  完全不可见(无 catch、无文案)。夹取在 `boundsPayload` 里,后端只是最后一道闸。
 *
 *  返回真正落盘的那一对,让调用方可以把本地暂存对齐到被夹取后的值(否则滑条会弹回)。 */
export async function setVocalRangeBounds(
  name: string,
  backend: Exclude<VoiceType, "vocoder">,
  speakerId: number,
  usable: [number, number],
  comfort: [number, number],
): Promise<RangeBoundsEdit | null> {
  const entry = useVoiceModelStore.getState().models[backend]?.find((m) => m.name === name);
  const existing = (entry?.config as { vocal_range?: { speakers?: Record<string, SpeakerRangeRecord> } } | undefined)
    ?.vocal_range;
  const sp = existing?.speakers?.[String(speakerId)];
  if (!sp) return null;
  const next = boundsPayload(sp, usable, comfort);
  await invoke("set_model_vocal_range", {
    name,
    modelType: backend,
    record: { speakers: { ...existing!.speakers, [String(speakerId)]: next } },
  });
  await useVoiceModelStore.getState().fetchModels();
  return { usable: next.usable, comfort: next.comfort };
}
