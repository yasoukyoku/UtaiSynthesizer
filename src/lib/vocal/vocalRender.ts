// ② Vocal render (S48 Phase 6, §11 / §10.1). Turn a segment's EDITED notes into singing:
//   1. build the score triples (per-note lyric + RAW MIDI + 50fps frames) covering [0, lastNoteEnd],
//      with a leading rest so the stem starts at the segment start and explicit gap rests (§3.4 — a rest
//      is NEVER inferred from pitch==0), and the whole-segment Option-A f0 (evalF0CentsFrames);
//   2. invoke the Rust `render_vocal_segment` (ScoreToCV → SVC net_g, +transpose Rust-side);
//   3. deposit the baked wav as a processedOutputs OVERLAY — non-undoable (sig-invisible) and it rides
//      the SAME lane machinery as an audio track's sub-lanes, so it plays back / persists / mutes for free.
import { invoke } from "@tauri-apps/api/core";
import { RVC_DEFAULTS, SOVITS_DEFAULTS, type RvcOptions, type SovitsOptions } from "../workflow/voiceDefaults";
import { evalF0CentsFrames, evalCurveAt } from "../f0eval";
import { isBreathLyric } from "../vocalNotes";
import { DEFAULT_LANG_ID, effLangId } from "./languages";
import { msToTicks } from "../audio/laneOps";
import { useProjectStore, DEFAULT_VOCAL_PARAMS } from "../../store/project";
import { useAppStore } from "../../store/app";
import i18n from "../../i18n";
import { backendErrorMessage, isCancelError } from "../backendError";
import { useVoiceModelStore } from "../../store/voice-models";
import { contentSig, vocalParamsSig } from "../../store/history";
import type { Note, PitchCurve, NoteTransition, ProcessedOutput, Track, Segment } from "../../types/project";

/** Thrown by the global single-flight backstop — the caller shows a "busy" message instead of "failed". */
export const VOCAL_RENDER_BUSY = "VOCAL_RENDER_BUSY";
/** Thrown by renderVocalPart when the track's singer can't be resolved / the segment has no renderable notes.
 *  The manual-render caller maps these to their own toasts; the auto-render batch pre-filters both away. */
export const VOCAL_NO_VOICE = "VOCAL_NO_VOICE";
export const VOCAL_EMPTY = "VOCAL_EMPTY";
/** Backend guard code: a genuine multi-speaker spk_mix BLEND was combined with a diffusion companion, whose
 *  condition encoder ignores the blend (renders toward one speaker). Rust returns the CODE; the frontend
 *  maps it to an i18n toast (no hardcoded Chinese in Rust — S56 rule). */
export const VOCAL_SPK_MIX_DIFFUSION = "SPK_MIX_DIFFUSION";
/** Rust G2P codes (S58): `VOCAL_OOV: <lyric>` — a lyric has no phoneme mapping in its effective language
 *  (LOUD, never a silent SP fallback); `VOCAL_PHONE_MISSING: <phone>` — a mapped phone fell outside the
 *  210-token ScoreToCV vocab (internal invariant; should be impossible with audited dictionaries). */
export const VOCAL_OOV = "VOCAL_OOV";
export const VOCAL_PHONE_MISSING = "VOCAL_PHONE_MISSING";

/** Map a vocal-render failure to its user-facing message. THE single error→text mapping for BOTH render
 *  paths (the sidebar's manual Render button and the Play-time auto-render batch) — never fork it. Codes
 *  carrying a detail payload (`CODE: detail`) interpolate it into the i18n string. */
export function vocalRenderErrorMessage(e: unknown): string {
  const msg = String(e);
  // Payload-carrying codes FIRST: their detail is user content (a lyric), and a lyric that happens to
  // CONTAIN another code string ("VOCAL_EMPTY" as a lyric…) must not hijack the match (audit).
  const dict = msg.match(/VOCAL_DICT_MISSING:\s*(.*)$/);
  if (dict) return i18n.t("vocalEditor.render.dictMissing", { file: dict[1] });
  const oov = msg.match(/VOCAL_OOV:\s*(.*)$/);
  if (oov) return i18n.t("vocalEditor.render.oov", { lyric: oov[1] });
  const ph = msg.match(/VOCAL_PHONE_MISSING:\s*(.*)$/);
  if (ph) return i18n.t("vocalEditor.render.phoneMissing", { phone: ph[1] });
  if (msg.includes(VOCAL_NO_VOICE)) return i18n.t("vocalEditor.render.noVoice");
  if (msg.includes(VOCAL_EMPTY)) return i18n.t("vocalEditor.render.empty");
  if (msg.includes(VOCAL_RENDER_BUSY)) return i18n.t("vocalEditor.render.busy");
  if (msg.includes(VOCAL_SPK_MIX_DIFFUSION)) return i18n.t("vocalEditor.render.spkMixDiffusion");
  // App-wide backend CODEs (APP_BUSY from the VoiceRunGuard etc.) — the shared mapper, consulted AFTER
  // the payload regexes above so a lyric containing a code string can't hijack the match.
  const shared = backendErrorMessage(e);
  if (shared) return shared;
  return `${i18n.t("vocalEditor.render.failed")}: ${msg}`;
}

/** Cancel check for the vocal-render funnels. Payload-carrying codes (VOCAL_OOV / VOCAL_DICT_MISSING /
 *  VOCAL_PHONE_MISSING) embed the user's LYRIC verbatim — a lyric that happens to contain a
 *  cancel-sentinel substring ("已取消" / "CANCELLED") must not silently swallow the real error (same
 *  ordering rationale as vocalRenderErrorMessage's payload-first rule). */
export function isVocalCancelError(e: unknown): boolean {
  if (/VOCAL_(OOV|DICT_MISSING|PHONE_MISSING):/.test(String(e))) return false;
  return isCancelError(e);
}

/** ScoreToCV native frame rate — the triple `frames` and the f0 array share this one grid so they align. */
const RENDER_FPS = 50;
/** Stable lane identity for the single baked vocal stem (mirrors an Output-node id — one lane per segment). */
export const VOCAL_LANE_ID = "vocal";

export interface ScoreTriple {
  lyric: string;
  note_num: number;
  frames: number;
  /** S58: the note's EFFECTIVE lang id (note.lang override ?? track default) — snake-free wire name
   *  matching Rust `ScoreNote.lang`. Rests/sustains carry it too (Rust run-assignment refines them). */
  lang: number;
  /** §3.7 traditional-phoneme override (pinyin/kana/ARPABET/MFA — never raw IPA). */
  phoneme_input?: string;
}

/** Wire options mirroring Rust `VocalRenderOptions` (snake_case — Tauri passes them through verbatim).
 *  Item-1: the backend-specific quality knobs REUSE the SoVITS/RVC contracts (`sovits`/`rvc`), the FULL
 *  object each (defaults + the user's overrides). The command force-neutralizes auto_f0/f0_shift/loudness/
 *  only_diffusion/rms_mix (they'd break the ② render). */
export interface VocalRenderOptions {
  backend: "sovits" | "rvc";
  cv_speaker_id: number;
  lang_id: number;
  transpose: number;
  /** S60-2 音域扩展 (track-level): no-op without a sidecar vocal_range record. */
  range_extend: boolean;
  sovits: SovitsOptions;
  rvc: RvcOptions;
}

/**
 * Build the render input from a segment's notes. Score triples cover [0, lastNoteEnd] contiguously: a
 * leading rest (so stem-ms 0 == the segment start → the lane plays 1:1 aligned), each note (RAW MIDI —
 * transpose is Rust-side, §9.3), and explicit gap rests between notes. The whole-segment 50fps Option-A f0
 * (WRITTEN-pitch cents + voiced mask) is sampled on the SAME `ticksPerFrame` grid, so Σ(triple frames) ==
 * the f0 frame count and `build_note_hz`'s cv↔DAW map is exact. Pure.
 */
export function buildVocalScore(
  notes: readonly Note[],
  pitchDev: PitchCurve | undefined,
  tempo: number,
  defaultTransition: Required<NoteTransition>,
  breathToken: string,
  paramCurves?: Record<string, PitchCurve>,
  formantScalar = 0,
  defaultLangId: number = DEFAULT_LANG_ID,
): { triples: ScoreTriple[]; f0Cents: number[]; f0Voiced: number[]; loudnessEnv: number[]; formantEnv: number[] } {
  const { triples, sorted, ticksPerFrame, frameCount } = buildScoreTriples(notes, tempo, breathToken, defaultLangId);
  // f0 sees the notes WITHOUT breaths (they're unvoiced): a breath's frames fall in a rest gap → voiced 0,
  // and its neighbours become phrase edges (release-drift before, onset-scoop after). frameCount is unchanged
  // (it's the triple cursor incl. breath frames), so the array length still equals Σ(triple frames).
  const pitchNotes = sorted.filter((n) => !isBreathLyric(n.lyric, breathToken));
  const { cents, voiced } = evalF0CentsFrames(
    pitchNotes,
    pitchDev,
    { frameStartTick: 0, ticksPerFrame, frameCount },
    { tempo, defaultTransition },
  );
  // ② loudness + formant per-frame envelopes on the SAME 50fps grid as f0 (so Rust aligns them via the same
  // note-group remap). loudness = dB curve → linear multiplier (evalCurveAt returns 0 dB when absent → ×1);
  // formant = track scalar + lane semitones (additive). An absent loudness lane, or a 0 formant scalar with
  // no formant lane, yields an EMPTY array → Rust treats it as "no lane" (flat = exact-parity no-op). §M-defer.
  const loudCurve = paramCurves?.["loudness"];
  const formantCurve = paramCurves?.["formant"];
  const loudnessEnv = loudCurve ? sampleParamFrames(loudCurve, ticksPerFrame, frameCount, (db) => Math.pow(10, db / 20)) : [];
  const formantEnv =
    formantScalar !== 0 || formantCurve
      ? sampleParamFrames(formantCurve, ticksPerFrame, frameCount, (semi) => formantScalar + semi)
      : [];
  return { triples, f0Cents: Array.from(cents), f0Voiced: Array.from(voiced), loudnessEnv, formantEnv };
}

/** The score-triple construction shared by the RENDER (buildVocalScore) and the OOV VALIDATION watcher
 *  (oovWatch) — the validation payload must be STRUCTURALLY IDENTICAL to what renders (same breath
 *  mapping, same inserted gap rests — a rest breaks a zh phrase window, so its presence changes the
 *  polyphone verdict) or the editor's marking could drift from the render's judgment. `tripleNoteIds`
 *  is parallel to `triples` (null = an inserted gap rest) so verdicts map back to notes. Pure. */
export function buildScoreTriples(
  notes: readonly Note[],
  tempo: number,
  breathToken: string,
  defaultLangId: number,
): { triples: ScoreTriple[]; tripleNoteIds: (string | null)[]; sorted: Note[]; ticksPerFrame: number; frameCount: number } {
  const ticksPerFrame = msToTicks(1000 / RENDER_FPS, tempo); // 20 ms per 50fps frame
  const frameOf = (relTick: number) => Math.round(relTick / ticksPerFrame);
  const sorted = [...notes].sort((a, b) => a.tick - b.tick || (a.id < b.id ? -1 : 1));
  // M3 breath: a breath lyric (canonical AP/ap or the track's trigger) maps to the `AP` phone Rust
  // recognizes. It is also UNVOICED — dropped from the pitch chain by the caller so it breaks the line
  // and the neighbours get the §10.5 release/scoop (段中尾音) instead of gliding into/out of the breath.
  const mapLyric = (l: string) => (isBreathLyric(l, breathToken) ? "AP" : l);

  const triples: ScoreTriple[] = [];
  const tripleNoteIds: (string | null)[] = [];
  let cursor = 0; // segment-relative tick covered so far
  for (const n of sorted) {
    const start = Math.max(cursor, n.tick);
    const end = n.tick + n.duration;
    if (end <= cursor) continue; // fully swallowed by a previous note (defensive — notes don't overlap)
    // S58: per-note effective language (note override ?? track default). Gap rests take the default —
    // Rust's run assignment attaches them to the surrounding run anyway (a rest's lang is only read
    // when the whole score has no sung note).
    const lang = effLangId(n.lang, defaultLangId);
    if (start > cursor) {
      const restFrames = frameOf(start) - frameOf(cursor);
      if (restFrames > 0) {
        triples.push({ lyric: "R", note_num: 0, frames: restFrames, lang: defaultLangId });
        tripleNoteIds.push(null);
      }
    }
    const noteFrames = frameOf(end) - frameOf(start);
    if (noteFrames > 0) {
      const t: ScoreTriple = { lyric: mapLyric(n.lyric), note_num: n.pitch, frames: noteFrames, lang };
      if (n.phonemeInput) t.phoneme_input = n.phonemeInput;
      triples.push(t);
      tripleNoteIds.push(n.id);
    }
    cursor = end;
  }
  return { triples, tripleNoteIds, sorted, ticksPerFrame, frameCount: frameOf(cursor) };
}

/** Sample a segment-relative param curve at each of `frameCount` 50fps frames (`f·ticksPerFrame`), applying
 *  `transform` (dB→linear / +scalar). Mirrors evalF0CentsFrames' grid so the envelope aligns with f0. */
function sampleParamFrames(
  curve: PitchCurve | undefined,
  ticksPerFrame: number,
  frameCount: number,
  transform: (v: number) => number,
): number[] {
  const out = new Array<number>(frameCount);
  for (let f = 0; f < frameCount; f++) out[f] = transform(evalCurveAt(curve, f * ticksPerFrame));
  return out;
}

interface AudioFileInfo {
  duration_ms: number;
  peaks: number[];
}
let vocalRunSeq = 0;

/**
 * Render a vocal segment and deposit the baked stem as a processedOutputs overlay.
 *  - GLOBAL single-flight: only one vocal render at a time (the shared ORT engine +
 *    release_gpu_sessions_except would make concurrent renders evict each other mid-inference). Throws
 *    VOCAL_RENDER_BUSY as a backstop (the button also gates on useAppStore.vocalRenderActive).
 *  - Never destroys a good bake before its replacement: a RE-render keeps the old lane playing until the
 *    new one lands; a FIRST render shows a loading placeholder for feedback. On error/cancel the PRIOR
 *    state is restored (re-render) or the spinner cleared (first render) — never a lost/broken render.
 *  - Deposits only if the segment still exists (a delete/project-load mid-render drops it). Throws on
 *    failure so the caller can toast; the deposit is sig-invisible (non-undoable overlay).
 */
export async function renderVocalSegment(req: {
  trackId: string;
  segmentId: string;
  laneLabel: string;
  voiceName: string;
  modelPath: string;
  triples: ScoreTriple[];
  f0Cents: number[];
  f0Voiced: number[];
  /** ② per-frame @50fps loudness (linear multiplier) + formant (semitones) envelopes; empty = no lane. */
  loudnessEnv: number[];
  formantEnv: number[];
  options: VocalRenderOptions;
  /** The render-input signature this bake corresponds to — stamped on the deposited lane so a later Play
   *  can skip re-rendering an unchanged segment (see vocalRenderSig / isVocalDirty). */
  renderedSig?: string;
}): Promise<void> {
  const { trackId, segmentId, laneLabel } = req;
  if (useAppStore.getState().vocalRenderActive) throw new Error(VOCAL_RENDER_BUSY);
  useAppStore.getState().setVocalRenderActive(true);
  useAppStore.getState().setRenderingVocalTrackId(trackId); // ② spinner on this track's header while rendering (§user)

  const seg = () =>
    useProjectStore.getState().tracks.find((t) => t.id === trackId)?.segments.find((s) => s.id === segmentId);
  const prevOutputs = seg()?.processedOutputs; // the current bake (if any) — kept while rendering, restored on failure
  const deposit = (outs: ProcessedOutput[] | undefined) =>
    useProjectStore.getState().replaceProcessedOutputs(trackId, segmentId, outs ?? []);

  // FIRST render (no bake yet) → loading placeholder for feedback; RE-render → keep the old bake playing
  // until the new stem lands (mirrors the audio path — never wipe a good render before its replacement).
  if (!prevOutputs || prevOutputs.length === 0) {
    deposit([{ laneId: VOCAL_LANE_ID, laneLabel, group: laneLabel, audioPath: "", totalDurationMs: 0, waveformPeaks: [], outputNodeId: VOCAL_LANE_ID, loading: true }]);
  }
  try {
    const raw = await invoke<string>("ensure_cache_dir", {
      segmentId: `${segmentId}/v${Date.now().toString(36)}${(vocalRunSeq++).toString(36)}`,
    });
    const outputPath = `${raw.replace(/\\/g, "/")}/vocal.wav`;
    const result = await invoke<{ audio: number[]; sample_rate: number }>("render_vocal_segment", {
      voiceName: req.voiceName,
      modelPath: req.modelPath,
      nodeId: segmentId,
      score: req.triples,
      f0Cents: req.f0Cents,
      f0Voiced: req.f0Voiced,
      loudnessEnv: req.loudnessEnv,
      formantEnv: req.formantEnv,
      options: req.options,
    });
    await invoke("save_temp_audio", { samples: result.audio, sampleRate: result.sample_rate, outputPath });
    const info = await invoke<AudioFileInfo>("load_audio_file", { path: outputPath });
    if (seg()) {
      deposit([{ laneId: VOCAL_LANE_ID, laneLabel, group: laneLabel, audioPath: outputPath, totalDurationMs: info.duration_ms, waveformPeaks: info.peaks, outputNodeId: VOCAL_LANE_ID, renderedSig: req.renderedSig }]);
    }
  } catch (e) {
    if (seg()) deposit(prevOutputs); // restore the prior bake (re-render) or clear the first-render spinner
    throw e;
  } finally {
    useAppStore.getState().setVocalRenderActive(false);
    useAppStore.getState().setRenderingVocalTrackId(null);
  }
}

// ── Auto-render-on-Play (S55): render vocal tracks whose notes/params CHANGED since their last bake, and
//    skip the unchanged ones. The dirty test compares a render-input signature against the one stamped on
//    the bake (renderedSig). One shared render path (renderVocalPart) backs BOTH the sidebar's manual
//    Render button and this batch, so they can never drift. ──

/** The full set of inputs a bake depends on, as one string: the segment content (notes + pitchDev +
 *  paramCurves, via contentSig), the track's vocal params (backend/speaker/lang/transpose/transition/
 *  sovits/rvc/breathToken, via vocalParamsSig), the singer (voiceModel), and the tempo — buildVocalScore
 *  derives its 50fps grid + f0 from the tempo, so a BPM change alters the bake even with identical notes.
 *  Reuses the history helpers (single source — never fork a sig). */
export function vocalRenderSig(track: Track, seg: Segment, tempo: number): string {
  return `${contentSig(seg.content)}|${vocalTrackSig(track, tempo)}`;
}

/** The TRACK-level + tempo terms of vocalRenderSig, alone (no segment content). S61 copy/paste uses it
 *  to decide whether a carried bake is still valid on the DESTINATION track: the pasted copy's notes get
 *  fresh ids (contentSig can never match the source's), so validity = "source bake was clean AND the
 *  track/tempo terms are byte-equal between copy-source and paste-destination". Kept HERE so it can never
 *  drift from vocalRenderSig (same string, single construction). */
export function vocalTrackSig(track: Track, tempo: number): string {
  return `vp:${vocalParamsSig(track.vocalParams)}|vm:${track.voiceModel ?? ""}|bpm:${tempo}|rr:${rangeRecordSig(track)}`;
}

/** Resolve a track's configured singer to its installed model entry (undefined = no vocalParams, no
 *  voiceModel, or the model is gone). THE one "is this track renderable" probe (isVocalDirty + paste). */
export function resolveTrackVoice(track: Track): { name: string; path: string } | undefined {
  const vp = track.vocalParams;
  if (!vp) return undefined;
  return useVoiceModelStore.getState().models[vp.backend]?.find((m) => m.name === track.voiceModel);
}

/** S60-2 audit: the model's vocal_range record IS a render input (it decides the tier shift),
 *  so a re-test / comfort adjustment must dirty the bakes that used it — else Play keeps
 *  serving audio rendered under the OLD zone. Only the usable/comfort bounds matter (the raw
 *  per-semitone scan doesn't feed the render); gated off when the track opted out. */
function rangeRecordSig(track: Track): string {
  const vp = track.vocalParams ?? DEFAULT_VOCAL_PARAMS;
  if (vp.rangeExtend !== true || !track.voiceModel) return ""; // S62c: extension is opt-in (absent = OFF)
  const entry = useVoiceModelStore.getState().models[vp.backend]?.find((m) => m.name === track.voiceModel);
  const rec = (entry?.config as { vocal_range?: { speakers?: Record<string, { usable?: unknown; comfort?: unknown }> } } | undefined)
    ?.vocal_range;
  if (!rec?.speakers) return "";
  return Object.entries(rec.speakers)
    .map(([id, sp]) => `${id}=${JSON.stringify(sp?.usable)}~${JSON.stringify(sp?.comfort)}`)
    .sort()
    .join(",");
}

/** Split a segment (audioClip OR notes) at `tick`, carrying + windowing a CLEAN vocal bake so the split needs
 *  no re-render (§user: split is not a re-render). THE single split entry point for the toolbar + context menu
 *  (the dirty guard below must never be duplicated / forgotten). Returns the new right-half id (null = no-op).
 *
 *  THE DIRTY GUARD (audit): window-stamp ONLY when the parent bake was CLEAN. A DIRTY parent (edited / tempo /
 *  param / singer changed but not yet re-rendered) carries a STALE stem; stamping its CURRENT-content windowSig
 *  would launder that stale audio into false-clean → both halves play the pre-edit stem forever (silent wrong
 *  audio — the exact mirror of the split-then-edit case). So we compute `wasDirty` on the WHOLE parent BEFORE
 *  the split, and when dirty we CLEAR windowSig (see stampSplitWindowSigs) so only the parent `renderedSig`
 *  governs → both halves mismatch current content → dirty → Play re-renders them correctly. */
export function splitSegmentVocalAware(trackId: string, segId: string, tick: number, tempo: number): string | null {
  const track = useProjectStore.getState().tracks.find((t) => t.id === trackId);
  const seg = track?.segments.find((s) => s.id === segId);
  const parentWasDirty = !!(track && seg) && isVocalDirty(track, seg, tempo);
  const newId = useProjectStore.getState().splitSegment(trackId, segId, tick);
  if (newId) stampSplitWindowSigs(trackId, [segId, newId], tempo, parentWasDirty);
  return newId;
}

/** After a notes SPLIT carries + windows the baked stem, mark each half's window validity. When the parent was
 *  CLEAN, stamp `windowSig` = vocalRenderSig of THIS half's (windowed) content, so isVocalDirty accepts the
 *  window (dual-sig) with no re-render. When the parent was DIRTY, CLEAR windowSig (a stale carried stem must
 *  never read clean — the carried window does NOT match this half's content). The bake's `renderedSig` (the
 *  PARENT whole-stem content) is LEFT UNCHANGED either way so an undo-of-split still matches the restored full
 *  content. Both sigs ride the OVERLAY (never undoable → can never desync from the bake — unlike an undoable
 *  flag). No-op for a non-notes / un-baked half. Callers: splitSegmentVocalAware, and S61 paste (a pasted
 *  copy's fresh note ids make the carried renderedSig permanently stale — the SAME stamp/clear discipline
 *  marks the carried stem valid-for-destination or leaves it dirty; see lib/clipboard.ts). */
export function stampSplitWindowSigs(trackId: string, segIds: string[], tempo: number, parentWasDirty: boolean): void {
  for (const segId of segIds) {
    const track = useProjectStore.getState().tracks.find((t) => t.id === trackId);
    const seg = track?.segments.find((s) => s.id === segId);
    if (!track || !seg || seg.content.type !== "notes" || !seg.processedOutputs?.length) continue;
    if (!seg.processedOutputs.some((o) => o.laneId === VOCAL_LANE_ID && !o.loading)) continue;
    // Clean → this half's own sig accepts the window; Dirty → undefined so only the (mismatching) parent
    // renderedSig governs and both halves stay dirty → re-render (never launder a stale stem clean).
    const sig = parentWasDirty ? undefined : vocalRenderSig(track, seg, tempo);
    const outs = seg.processedOutputs.map((o) => (o.laneId === VOCAL_LANE_ID ? { ...o, windowSig: sig } : o));
    useProjectStore.getState().replaceProcessedOutputs(trackId, segId, outs);
  }
}


/** Resolve a track's singer + build its score + invoke the render, stamping the render-input sig on the
 *  deposit. The ONE render code path (the sidebar button and the Play batch both call this). Throws
 *  VOCAL_NO_VOICE / VOCAL_EMPTY (caller maps to a toast); VOCAL_RENDER_BUSY bubbles from renderVocalSegment. */
export async function renderVocalPart(track: Track, seg: Segment, tempo: number, laneLabel: string): Promise<void> {
  if (seg.content.type !== "notes") return;
  const vp = track.vocalParams ?? DEFAULT_VOCAL_PARAMS;
  const entry = useVoiceModelStore.getState().models[vp.backend]?.find((m) => m.name === track.voiceModel);
  if (!entry) throw new Error(VOCAL_NO_VOICE);
  const { triples, f0Cents, f0Voiced, loudnessEnv, formantEnv } = buildVocalScore(
    seg.content.notes, seg.content.pitchDev, tempo, vp.transition, vp.breathToken ?? "AP",
    seg.content.paramCurves, vp.formant ?? 0, vp.langId,
  );
  if (triples.length === 0) throw new Error(VOCAL_EMPTY);
  await renderVocalSegment({
    trackId: track.id,
    segmentId: seg.id,
    laneLabel,
    voiceName: entry.name,
    modelPath: entry.path,
    triples,
    f0Cents,
    f0Voiced,
    loudnessEnv,
    formantEnv,
    options: {
      backend: vp.backend,
      cv_speaker_id: vp.speakerId,
      lang_id: vp.langId,
      transpose: vp.transpose,
      // S60-2: absent = ON (no-op until the model carries a vocal_range record)
      range_extend: vp.rangeExtend === true, // S62c: opt-in (absent = OFF)
      sovits: { ...SOVITS_DEFAULTS, ...(vp.sovits ?? {}) },
      rvc: { ...RVC_DEFAULTS, ...(vp.rvc ?? {}) },
    },
    renderedSig: vocalRenderSig(track, seg, tempo),
  });
}

/** True when a notes segment needs a (re-)bake: it has notes, a resolvable singer, and either no bake yet
 *  or a bake whose stamped sig no longer matches the current inputs. A segment with no resolvable singer is
 *  NOT dirty (we can't render it — skip silently rather than fail the batch). Ignores loading placeholders. */
export function isVocalDirty(track: Track, seg: Segment, tempo: number): boolean {
  if (seg.content.type !== "notes") return false;
  const bake = seg.processedOutputs?.find((o) => o.laneId === VOCAL_LANE_ID && !o.loading);
  // A segment emptied of ALL its notes but still holding a bake reconciles to SILENCE — else the old
  // singing keeps playing for a segment that has no notes (segmentPlaysLanes still schedules the stem).
  // There's nothing to bake, so renderDirtyVocals clears the overlay instead of rendering.
  if (seg.content.notes.length === 0) return !!bake;
  if (!track.vocalParams) return false;
  if (!resolveTrackVoice(track)) return false;
  if (!bake) return true;
  // ② DUAL-SIG acceptance (§user: split is not a re-render). A carried SPLIT-WINDOW bake keeps the PARENT's
  // whole-stem `renderedSig` but is windowed to this half; `windowSig` is THIS half's own content sig (stamped
  // by stampSplitWindowSigs). Accept when EITHER matches the current content: `renderedSig` matches after an
  // undo-of-split (the full stem == the restored full content) OR `windowSig` matches right after the split
  // (the window == this half). Any REAL drift (edit / tempo / param / singer) changes vocalRenderSig → fails
  // BOTH → re-render. Both sigs ride the OVERLAY, so they can never desync from the bake (the undoable-flag
  // desync the audit caught — silent wrong audio on split→edit→render→undo / tempo — is structurally gone).
  const cur = vocalRenderSig(track, seg, tempo);
  return bake.renderedSig !== cur && bake.windowSig !== cur;
}

/** Every dirty vocal segment across all tracks (read live). Empty ⇒ Play proceeds with zero added latency. */
export function collectDirtyVocals(tempo: number): Array<{ trackId: string; segmentId: string }> {
  const out: Array<{ trackId: string; segmentId: string }> = [];
  for (const tr of useProjectStore.getState().tracks) {
    for (const sg of tr.segments) if (isVocalDirty(tr, sg, tempo)) out.push({ trackId: tr.id, segmentId: sg.id });
  }
  return out;
}

/** Poll until no vocal render is in flight (the backend single-flights; a manual Render started just before
 *  Play must finish + deposit its fresh sig before the batch re-tests dirtiness). Bounded so a wedged render
 *  can't hang Play forever. */
async function waitVocalIdle(timeoutMs = 120000): Promise<void> {
  const start = performance.now();
  while (useAppStore.getState().vocalRenderActive) {
    if (performance.now() - start > timeoutMs) return;
    await new Promise((r) => setTimeout(r, 50));
  }
}

export interface DirtyRenderResult { rendered: number; failed: number; cancelled: boolean; }

/** Render a list of dirty segments SEQUENTIALLY (the backend voice guard cross-kills concurrent runs, so
 *  parallel is unsafe). Each item is re-read + re-tested live before rendering (it may have been deleted, or
 *  already re-baked by a racing trigger). `shouldCancel` (a second Play press) aborts between items and, via
 *  cancel_voice, mid-render (the in-flight invoke throws → caught → we bail). */
export async function renderDirtyVocals(
  list: Array<{ trackId: string; segmentId: string }>,
  tempo: number,
  laneLabel: string,
  opts?: { shouldCancel?: () => boolean },
): Promise<DirtyRenderResult> {
  await waitVocalIdle();
  let rendered = 0;
  let failed = 0;
  const MAX_FAILURE_TOASTS = 3; // loud but bounded — a many-track failure aggregates past this
  for (const { trackId, segmentId } of list) {
    if (opts?.shouldCancel?.()) return { rendered, failed, cancelled: true };
    const tr = useProjectStore.getState().tracks.find((t) => t.id === trackId);
    const sg = tr?.segments.find((s) => s.id === segmentId);
    if (!tr || !sg || !isVocalDirty(tr, sg, tempo)) continue;
    // Emptied-of-notes segment with a stale bake → clear the overlay (reconcile to silence, nothing to render).
    if (sg.content.type === "notes" && sg.content.notes.length === 0) {
      useProjectStore.getState().replaceProcessedOutputs(tr.id, sg.id, []);
      rendered++;
      continue;
    }
    try {
      await renderVocalPart(tr, sg, tempo, laneLabel);
      rendered++;
    } catch (e) {
      if (opts?.shouldCancel?.()) return { rendered, failed, cancelled: true };
      // Backend-side cancel rejection (CANCELLED / legacy 已取消): same silent settle as shouldCancel —
      // a user cancel must never toast as a per-track failure. Payload-aware check: a VOCAL_OOV lyric
      // containing a sentinel substring is a REAL error, not a cancel.
      if (isVocalCancelError(e)) return { rendered, failed, cancelled: true };
      // VOCAL_EMPTY = a degenerate no-renderable-content segment (every note rounds to 0 frames):
      // nothing to bake AND it can never converge — treat like the emptied-segment case above (the
      // manual Render button still reports it loudly), instead of re-toasting on every Play (audit).
      if (String(e).includes(VOCAL_EMPTY)) continue;
      failed++;
      // LOUD failure (§user: Play's auto-render must report exactly like the manual Render button — never
      // swallow). Same shared mapping, prefixed with the track name so the user knows WHICH track failed;
      // capped so a project-wide failure (e.g. missing dictionary) doesn't storm one toast per segment.
      if (failed <= MAX_FAILURE_TOASTS) {
        useAppStore.getState().showToast(`${tr.name}: ${vocalRenderErrorMessage(e)}`, "error");
      } else if (failed === MAX_FAILURE_TOASTS + 1) {
        useAppStore.getState().showToast(i18n.t("vocalEditor.render.moreFailures"), "error");
      }
    }
  }
  return { rendered, failed, cancelled: false };
}
