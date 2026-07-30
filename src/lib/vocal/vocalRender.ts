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
import { isBreathLyric, isRestLyric, isSilentLyric, vocalTokens, ZERO_TRANSITION, DEFAULT_CONSONANT_EMPHASIS_DB, DEFAULT_CONSONANT_VALLEY, type VocalTokens } from "../vocalNotes";
import { DEFAULT_LANG_ID, effLangId } from "./languages";
import { msToTicks } from "../audio/laneOps";
import { useProjectStore, DEFAULT_VOCAL_PARAMS } from "../../store/project";
import { useAppStore } from "../../store/app";
import i18n from "../../i18n";
import { backendErrorMessage, isCancelError } from "../backendError";
import { maybeShowErrorModal } from "../errorDisplay";
import { logToBackend } from "../log";
import { useVoiceModelStore } from "../../store/voice-models";
import { contentSig, vocalParamsSig } from "../../store/history";
import type { Note, PitchCurve, NoteTransition, ProcessedOutput, Track, Segment, VocalTrackParams } from "../../types/project";

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
/** S91 `VOCAL_ALIAS: <convention> <lyric>` — a lyric on a track using a UTAU alias convention is not a
 *  legal alias in it. Its OWN code, not `VOCAL_OOV`: that message tells the user to check the lyric or
 *  the LANGUAGE, which is the wrong advice here (the language is fine — the convention is). The
 *  convention token comes first and is always whitespace-free, so the lyric (arbitrary user text) can
 *  be taken as the rest of the line without a separator that a lyric could contain. */
export const VOCAL_ALIAS = "VOCAL_ALIAS";

/** S66 pre-render model check for the vocal track: the core aux pack (ScoreToCV / ContentVec /
 *  RMVPE / vocoder onnx) must be present or the render dies mid-flight with AUX_FILE_MISSING —
 *  instead, open the one-click MissingModelsDialog and return false so the caller aborts. Shared
 *  by the sidebar Render button and the Play-time auto-render batch (never fork). Best-effort:
 *  an IPC failure never blocks the render (Rust still errors loudly). */
export async function preflightVocalModels(): Promise<boolean> {
  return preflightAuxPack("aux-inference");
}

/** Generic pack preflight (S73): the auto-tune button checks its OWN pack ("aux-autotune") through the
 *  same funnel — never fold optional-feature files into aux-inference (that would hard-block Play for
 *  every upgraded user until they download the new file; offline = deadlock. S73 审查 HIGH). */
export async function preflightAuxPack(packId: string): Promise<boolean> {
  try {
    const packs = await invoke<Array<{ id: string; missing: number; downloading: boolean }>>(
      "asset_pack_status",
    );
    const pack = packs.find((p) => p.id === packId);
    if ((pack?.missing ?? 0) > 0 && !(pack?.downloading ?? false)) {
      useAppStore.getState().openMissingModels([{ kind: "auxPack", label: packId }]);
      return false;
    }
  } catch {
    /* best-effort */
  }
  return true;
}

/** Map a vocal-render failure to its user-facing message. THE single error→text mapping for BOTH render
 *  paths (the sidebar's manual Render button and the Play-time auto-render batch) — never fork it. Codes
 *  carrying a detail payload (`CODE: detail`) interpolate it into the i18n string. */
export function vocalRenderErrorMessage(e: unknown): string {
  const msg = String(e);
  // Payload-carrying codes FIRST: their detail is user content (a lyric), and a lyric that happens to
  // CONTAIN another code string ("VOCAL_EMPTY" as a lyric…) must not hijack the match (audit).
  const dict = msg.match(/VOCAL_DICT_MISSING:\s*(.*)$/);
  if (dict) return i18n.t("vocalEditor.render.dictMissing", { file: dict[1] });
  // ⚠ The convention token is whitelisted, not `\S+`: the payload after it is the user's LYRIC, and a
  // lyric that merely CONTAINS the string "VOCAL_ALIAS:" would otherwise hijack this branch away from
  // the code that really failed — a review reproduced exactly that with `VOCAL_OOV: VOCAL_ALIAS: x y`.
  // Requiring a known convention narrows the hijack to a lyric literally spelled like a real alias
  // failure; that residue is inherent to the "CODE: user-content" wire shape (the dict/oov pair has
  // it too) and is accepted, not solved.
  const alias = msg.match(/VOCAL_ALIAS:\s*(arpasing|xsampa|vccv|words)\s([\s\S]*)$/);
  if (alias) {
    return i18n.t("vocalEditor.render.aliasBad", {
      lyric: alias[2],
      set: i18n.t(`vocalEditor.sidebar.phonemeSet_${alias[1]}`, { defaultValue: alias[1] }),
    });
  }
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
  if (/VOCAL_(OOV|DICT_MISSING|PHONE_MISSING|ALIAS):/.test(String(e))) return false;
  return isCancelError(e);
}

/** ScoreToCV native frame rate — the triple `frames` and the f0 array share this one grid so they align. */
const RENDER_FPS = 50;
/** S87 — ONE render frame expressed in ticks at `tempo`, rounded UP. The vocal editor uses it as the length
 *  floor when grid snapping is OFF, and as the "this note is too short" test everywhere else.
 *  ⚠ CEIL, not round: `frameOf(end) − frameOf(start) ≥ 1` holds for EVERY start phase only once the span is
 *  at least a whole frame. Rounding 19.2 down to 19 left a note dragged to the editor's own floor still able
 *  to land on zero frames at 20 of every 100 start ticks (audit-caught).
 *  Exported from HERE because this module owns the render grid — the floor must never drift from it. */
export const renderFrameTicks = (tempo: number): number => Math.max(1, Math.ceil(msToTicks(1000 / RENDER_FPS, tempo)));
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
  /** S83 knife 6b: voiceless-onset emphasis (dB; 0 = off; SynthV "consonant strength" analogue). */
  consonant_emphasis_db: number;
  /** S84 C 刀: chain-internal consonant-valley scale (×measured per-class depth; 0 = off). */
  consonant_valley: number;
  /** S84 E 刀: vowel-clarity articulation oversampling (absent-in-params ≡ true). */
  vowel_clarity: boolean;
  /** S89 「自动咬字时序」: onset consonants pre-roll ahead of the beat (absent-in-params ≡ true). */
  consonant_preroll: boolean;
  /** S91 「音素约定」: which UTAU alias convention this track's ENGLISH lyrics use. Omitted/`null` =
   *  words through the dictionary (Rust's `#[serde(default)]` lands there, so an older caller — e.g.
   *  the range-scan literal in rangeTest.ts — is unaffected by construction). */
  phoneme_set?: string | null;
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
  tokens: VocalTokens,
  paramCurves?: Record<string, PitchCurve>,
  formantScalar = 0,
  defaultLangId: number = DEFAULT_LANG_ID,
): { triples: ScoreTriple[]; f0Cents: number[]; f0Voiced: number[]; loudnessEnv: number[]; formantEnv: number[] } {
  const { triples, f0Notes, ticksPerFrame, frameCount } = buildScoreTriples(notes, tempo, tokens, defaultLangId);
  // f0 sees the notes WITHOUT the SILENT ones (breath = unvoiced inhale, rest = silence): their frames fall
  // in a rest gap → voiced 0, and their neighbours become phrase edges (release-drift before, onset-scoop
  // after). frameCount is unchanged (it's the triple cursor incl. those frames), so the array length still
  // equals Σ(triple frames).
  // ⚠ S88: a REST note used to stay in this chain — it rendered as `R` (Rust: no phones) while the f0 feed
  // still declared it voiced at whatever pitch it was drawn on, so the line glided INTO and OUT of a
  // silence instead of breaking there. Written as a gap the same music behaved differently; that split is
  // what `isSilentLyric` closes.
  // S87 #3: `f0Notes`, not `sorted` — a borrowed frame must carry the BORROWER's pitch (see the borrow pass).
  const pitchNotes = f0Notes.filter((n) => !isSilentLyric(n.lyric, tokens));
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
 *  (oovWatch) — the validation payload must be STRUCTURALLY IDENTICAL to what renders (same breath/rest
 *  token mapping, same inserted gap rests — a rest breaks a zh phrase window, so its presence changes the
 *  polyphone verdict) or the editor's marking could drift from the render's judgment. `tripleNoteIds`
 *  is parallel to `triples` (null = an inserted gap rest) so verdicts map back to notes. Pure. */
export function buildScoreTriples(
  notes: readonly Note[],
  tempo: number,
  tokens: VocalTokens,
  defaultLangId: number,
): {
  triples: ScoreTriple[];
  tripleNoteIds: (string | null)[];
  /** S84 D 刀: sung notes whose frame span rounded to ZERO on the 50fps grid (e.g. a 30t note at
   *  tempo 222 ≈ 0.85 frames landing inside one frame cell) — they emit NO triple and would have
   *  vanished SILENTLY (く/ず in the S84 audit). Surfaced so oovWatch can mark them like OOV
   *  ("this note will not sound"). Forcing a minimum frame instead is FORBIDDEN — it would break
   *  the absolute-diff frame conservation (Σframes == frameOf(cursor)). */
  droppedNoteIds: string[];
  /** S87 #3: sung notes that rounded to zero frames and were RESCUED by borrowing one frame from a
   *  neighbour. A NON-blocking condition (the note sounds — it was merely nudged), as opposed to
   *  `droppedNoteIds`, which is blocking (the note will not sound at all). Kept as a separate list so the
   *  UI can grade the warning instead of painting both the same red (§user; mirrors the S85b OOV/dropped
   *  split, whose whole point was that a merged channel makes the wording lie). */
  borrowedNoteIds: string[];
  /** S87 — sung notes whose DURATION is under one render frame. This is the honest "too short" fact and the
   *  one the UI marks, because unlike `dropped`/`borrowed` it is a property of the NOTE ALONE: those two
   *  depend on the note's PHASE against a frame grid that restarts at the segment start, so merely SPLITTING
   *  a part re-rolls them (measured: 40 of 51 split offsets flip the verdict) — the marks would blink out
   *  when you cut, while a lyric-OOV mark, which depends on nothing positional, survives (§user, and they
   *  were right that the two ought to behave the same).
   *  Nesting: dropped ⊆ short and borrowed ⊆ short (a note at least a frame long always gets a frame). */
  shortNoteIds: string[];
  sorted: Note[];
  /** S87 #3: `sorted`, retimed for the F0 FEED only — see the borrow pass. Identical to `sorted` (same
   *  object identities) when nothing was borrowed. */
  f0Notes: Note[];
  ticksPerFrame: number;
  frameCount: number;
} {
  const ticksPerFrame = msToTicks(1000 / RENDER_FPS, tempo); // 20 ms per 50fps frame
  const frameOf = (relTick: number) => Math.round(relTick / ticksPerFrame);
  const sorted = [...notes].sort((a, b) => a.tick - b.tick || (a.id < b.id ? -1 : 1));
  // The two per-track triggers map onto the tokens Rust hard-wires, so a user's convenient glyph never has
  // to be stolen from real lyric material: a breath lyric → the `AP` phone, a rest lyric → `R`. Both are
  // also dropped from the pitch chain by the caller (isSilentLyric) so they break the line and the
  // neighbours get the §10.5 release/scoop (段中尾音) instead of gliding into/out of the silence.
  // Rest is tested first — the isSilentLyric tie-break, kept identical here so the triple and the pitch
  // chain can never disagree about a note whose two tokens were set to the same string.
  const mapLyric = (l: string) => (isRestLyric(l, tokens.rest) ? "R" : isBreathLyric(l, tokens.breath) ? "AP" : l);

  // ── PASS 1 — tile the segment into frame-bounded items (a gap rest, then its note, …). Boundaries are
  //    SHARED: item[k].endF === item[k+1].startF. That is the whole trick: Σframes telescopes to the last
  //    endF BY CONSTRUCTION, so the borrow pass below — which only ever moves a shared boundary — cannot
  //    break `Σframes == frameOf(cursor)` no matter what it decides. ──
  interface Item {
    note: Note | null; // null = an inserted gap rest
    startF: number;
    endF: number;
    lang: number;
    /** SILENCE, whether written as a gap or as a note carrying the rest token. Every rule below keys on
     *  THIS, not on `note === null`: a written rest is a rest, and treating it as a sung note is what made
     *  it lend like one, block a rescue like one, and get warned about like one. */
    rest: boolean;
  }
  const items: Item[] = [];
  const shortNoteIds: string[] = [];
  let cursor = 0; // segment-relative tick covered so far
  for (const n of sorted) {
    const start = Math.max(cursor, n.tick);
    const end = n.tick + n.duration;
    if (end <= cursor) continue; // fully swallowed by a previous note (defensive — notes don't overlap)
    const isRest = isRestLyric(n.lyric, tokens.rest);
    // phase-INDEPENDENT "too short" fact — but only for notes that were meant to SOUND. Warning that a
    // rest "will not be heard" is a false alarm: not being heard is the whole point of writing one.
    if (!isRest && n.duration < ticksPerFrame) shortNoteIds.push(n.id);
    // S58: per-note effective language (note override ?? track default). Gap rests take the default —
    // Rust's run assignment attaches them to the surrounding run anyway (a rest's lang is only read
    // when the whole score has no sung note).
    const cursorF = frameOf(cursor);
    const startF = frameOf(start);
    if (startF > cursorF) items.push({ note: null, startF: cursorF, endF: startF, lang: defaultLangId, rest: true });
    items.push({ note: n, startF, endF: frameOf(end), lang: effLangId(n.lang, defaultLangId), rest: isRest });
    cursor = end;
  }
  const frameCount = frameOf(cursor);

  // ── PASS 2 — S87 #3 借帧刀. A note that rounds to ZERO frames borrows exactly ONE, BACKWARD by
  //    preference (its start moves earlier: every boundary downstream — i.e. every beat the singer is
  //    aiming at — stays put; same direction as S83's consonant pre-borrow). Only if the item before it
  //    cannot pay does it take from the one after (that DOES delay the follower's onset, hence second).
  //    LENDER FLOORS (S83's范式, at note granularity): a rest must survive with ≥1 frame — it is the
  //    phrase boundary the f0 line, the zh polyphone window and the rest-gate all key on; a sung note
  //    must survive with ≥2, because a 1-frame sung note is the very pathology this knife exists to
  //    remove (and §user: "小心前面本身就是极短音 → 元音被吃"). Unlendable on both sides ⇒ still dropped,
  //    loudly. ──
  const MIN_KEEP_REST = 1;
  const MIN_KEEP_NOTE = 2;
  const canLend = (it: Item) => it.endF - it.startF >= (it.rest ? MIN_KEEP_REST : MIN_KEEP_NOTE) + 1;
  /** ★ The knife may only fire when it CANNOT HARM A NEIGHBOUR. A rescued note is exactly ONE frame long,
   *  and Rust's allocator pre-rolls the NEXT sung note's onset consonant out of the phone before it
   *  (score2cv.rs `assemble_arrays`, S83): a 1-frame lender yields `avail = (1 - SUNG_KEEP_MIN) = 0`, so the
   *  follower either loses its consonant outright (short follower — 「た」 comes out as 「あ」) or falls back
   *  to placing it INSIDE the note, putting its vowel ~40 ms after the beat. That is precisely the pathology
   *  S83 was built to remove, and trading a full written note's consonant for a ~5 ms grace note is a bad
   *  deal. So: rescue only when nothing needs to pre-roll from us — the next item is a REST (an SP lender is
   *  fine) or there is no next item. Otherwise the note is still reported as too short, exactly as before
   *  this knife existed. Never better than baseline is acceptable; WORSE than baseline is not.
   *  ⚠ S88 review — the lender must be a rest that SURVIVES ONTO THE WIRE. Until rest NOTES existed this
   *  was free: a gap rest is only pushed when `startF > cursorF` (PASS 1), so it always has ≥1 frame, and
   *  `!next.note` therefore implied "an SP triple exists to pre-roll from". A WRITTEN rest has no such
   *  floor — a sub-frame one emits nothing (PASS 3 skips frames ≤ 0) and can never acquire a frame (PASS 2
   *  only rescues sung notes), so the phone actually following the borrower would be the next SUNG note,
   *  which then pre-rolls its onset out of a 1-frame nucleus: `avail = 0`, consonant lost. Measured on the
   *  first cut of this diff: 78.7% of rescues in that shape. So require a NON-EMPTY rest; anything else
   *  (a sung note, or any item that vanishes) falls back to the baseline refusal. */
  const rescueIsSafe = (next: Item | null) => !next || (next.rest && next.endF > next.startF);
  const borrowedNoteIds: string[] = [];
  const droppedNoteIds: string[] = [];
  /** noteId → the span the F0 feed must use (see PASS 3). `grace` = this note IS the rescued one. */
  const retimed = new Map<string, { start: number; end: number; grace: boolean }>();
  /** ★ The EXACT tick evalF0CentsFrames samples frame `f` at — deliberately the SAME expression
   *  (`frameStartTick + f * ticksPerFrame`, frameStartTick = 0), so the two agree bit-for-bit.
   *  ⚠ ROUNDING this is a trap the review caught: `Math.round(f*tpf)` lands AFTER the sample point
   *  whenever the fraction is ≥ 0.5, the borrower then does not contain its own frame, findNoteAt
   *  picks someone else, and the "rescued" note comes out UNVOICED (rest lender) — silent, while the
   *  UI cheerfully reports it as rescued. Tempo-dependent: never at 100/125 bpm, ALWAYS at 300. */
  const frameTick = (f: number) => f * ticksPerFrame;
  const spanOf = (n: Note) => retimed.get(n.id) ?? { start: n.tick, end: n.tick + n.duration, grace: false };
  const setSpan = (n: Note, start: number, end: number, grace: boolean) =>
    retimed.set(n.id, { start, end: Math.max(start + ticksPerFrame * 1e-3, end), grace });
  for (let i = 0; i < items.length; i++) {
    const it = items[i]!;
    // Nothing to rescue for a REST that rounded away (silence is what it asked for) — only sung notes.
    if (!it.note || it.rest || it.endF > it.startF) continue; // not a sung note, or it already sounds
    const prev = i > 0 ? items[i - 1]! : null;
    const next = i + 1 < items.length ? items[i + 1]! : null;
    if (!rescueIsSafe(next)) {
      droppedNoteIds.push(it.note.id); // a sung note follows: rescuing would starve ITS onset (see above)
      continue;
    }
    if (prev && canLend(prev)) {
      prev.endF -= 1;
      it.startF -= 1; // the SAME boundary — Σ is untouched
      borrowedNoteIds.push(it.note.id);
    } else if (next && canLend(next)) {
      next.startF += 1;
      it.endF += 1; // likewise the SAME boundary
      borrowedNoteIds.push(it.note.id);
    } else {
      // S84 D 刀: nothing in reach could pay → the note will NOT sound. Record it loudly (oovWatch marks
      // it) instead of the pre-S84 silent vanish. Forcing a minimum frame here is still FORBIDDEN.
      droppedNoteIds.push(it.note.id);
      continue;
    }
    // The rescued note's F0 span becomes EXACTLY the frames it now owns, and any adjacent NOTE is clamped
    // to abut it — so every frame's sample point has exactly ONE owner and the array stays ordered.
    // (The neighbour that did NOT lend is clamped too: the borrower's span now snaps to frame boundaries,
    // which can reach a hair past its original edge, and two notes claiming one tick is how findNoteAt —
    // last match wins — silently hands a frame to the wrong pitch.)
    const bs = frameTick(it.startF);
    const be = frameTick(it.endF);
    // (`!rest` on both sides: a rest note is filtered out of the pitch chain, so it owns no sample point
    //  to fight over — retiming it would only put a never-read entry in the map.)
    setSpan(it.note, bs, be, true);
    if (prev?.note && !prev.rest) setSpan(prev.note, spanOf(prev.note).start, bs, spanOf(prev.note).grace);
    if (next?.note && !next.rest) setSpan(next.note, be, spanOf(next.note).end, spanOf(next.note).grace);
  }

  // ── PASS 3 — emit the triples, and the F0 feed. The pitch line is sampled by TICK (evalF0CentsFrames),
  //    NOT by these frame counts, so a borrowed frame would otherwise still carry the LENDER's pitch — or,
  //    when the lender is a rest, be UNVOICED, leaving the rescued note silent anyway. That is the
  //    note-level twin of the trap S83 hit with pre-borrowed consonants (anchor_voiced_phone_f0). Only the
  //    notes a borrow actually touched are retimed; every other note keeps its exact sub-frame tick, so a
  //    score with nothing to rescue feeds a byte-identical f0 array. ──
  const triples: ScoreTriple[] = [];
  const tripleNoteIds: (string | null)[] = [];
  for (const it of items) {
    const frames = it.endF - it.startF;
    if (frames <= 0) continue; // an empty gap rest, or a note that could not borrow (already reported)
    if (!it.note) {
      triples.push({ lyric: "R", note_num: 0, frames, lang: defaultLangId });
      tripleNoteIds.push(null);
      continue;
    }
    // A REST note carries no pitch, so it emits the same `note_num: 0` a gap rest does — "written as a
    // note" and "written as a gap" become the SAME payload, byte for byte. Rust already zeroes it
    // internally (score2svc's npitch, via is_silent_token); matching here means no consumer of the wire
    // has to remember to, which is exactly how the dead-zone planner came to read a silence as a sung
    // note. A BREATH keeps its drawn pitch: it is a real phone whose note_num merely goes unused.
    const t: ScoreTriple = { lyric: mapLyric(it.note.lyric), note_num: it.rest ? 0 : it.note.pitch, frames, lang: it.lang };
    if (it.note.phonemeInput) t.phoneme_input = it.note.phonemeInput;
    triples.push(t);
    tripleNoteIds.push(it.note.id);
  }
  const f0Notes = retimed.size === 0 ? sorted : sorted.map((n) => {
    const r = retimed.get(n.id);
    if (!r) return n;
    // A rescued note owns ONE frame, and that frame is sampled exactly at its own onset — where the §10.5
    // open-edge scoop is at its FLOOR. With the default transition it would sing `tone − openEdgeCents`
    // (200 ¢ = two semitones flat) for its entire life. A grace note has no room for a portamento, so it
    // gets its written pitch, flat. Lenders keep their own transition — they only gave up 20 ms.
    const span = { ...n, tick: r.start, duration: r.end - r.start };
    return r.grace ? { ...span, transition: { ...ZERO_TRANSITION } } : span;
  });
  return { triples, tripleNoteIds, droppedNoteIds, borrowedNoteIds, shortNoteIds, sorted, f0Notes, ticksPerFrame, frameCount };
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
    // S66/O5: Rust writes the wav to outputPath and returns just the path (the old samples-JSON
    // response + save_temp_audio write-back peaked at ~200MB of IPC per render).
    await invoke<{ path: string; sample_rate: number }>("render_vocal_segment", {
      voiceName: req.voiceName,
      modelPath: req.modelPath,
      nodeId: segmentId,
      score: req.triples,
      f0Cents: req.f0Cents,
      f0Voiced: req.f0Voiced,
      loudnessEnv: req.loudnessEnv,
      formantEnv: req.formantEnv,
      outputPath,
      options: req.options,
    });
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
 *  sovits/rvc/breath+rest tokens, via vocalParamsSig), the singer (voiceModel), and the tempo — buildVocalScore
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
  // forRender=true:autoTune 三元组不进渲染 sig(它们经 θ→contentSig 间接生效;直接进会让
  // 切 follow 开关/拖缩放凭空判废整段 bake——S73b 审查假脏)。
  return `vp:${vocalParamsSig(track.vocalParams, true)}|vm:${track.voiceModel ?? ""}|bpm:${tempo}|rr:${rangeRecordSig(track)}|g2p:${G2P_ALGO_VERSION}|st:${SCORE_TIMING_VERSION}`;
}

/** Resolve a track's configured singer to its installed model entry (undefined = no vocalParams, no
 *  voiceModel, or the model is gone). THE one "is this track renderable" probe (isVocalDirty + paste). */
export function resolveTrackVoice(track: Track): { name: string; path: string } | undefined {
  const vp = track.vocalParams;
  if (!vp) return undefined;
  return useVoiceModelStore.getState().models[vp.backend]?.find((m) => m.name === track.voiceModel);
}

/** Version of the range-extension DECISION algorithm (engine changes count too — anything that
 *  makes the same decision sound different). Bumping it invalidates every bake that was
 *  rendered under the old version — without it, changing the algorithm produces "I changed it
 *  and the user hears nothing", which reads as a failed fix (S81 audit).
 *  ★ Any change to the decision functions in src-tauri/src/inference/vocal_range.rs (or to the
 *  inverse engine) MUST bump this AND the matching literal in
 *  commands/audition.rs::audition_cache_tag. s82 = Signalsmith 1.3.2 native-formant inverse;
 *  s83 = quiet-damage capped at 1.0 (a loudness-tilted scale no longer freezes the optimizer
 *  at 0 while an unsingable climax stays broken — the chika_v2 case);
 *  s85 = SCORE path switched to dead-only (whole-piece shift abolished — only rest-delimited
 *  phrases containing truly-dead notes render at a minimal local shift + inverse; everything
 *  else stays at written pitch; memory S85 三轮耳判);
 *  s85b = the S83 quiet-cap + escape-valve REVERT (cover decision math back to v0.11.0);
 *  s85c = tiered search depth (superseded within the same night);
 *  s85d = the whole-piece shift machinery is RETIRED everywhere — cover/audition now run the
 *  same dead-only philosophy as the score path (user verdict, memory S85 七轮): only sustained
 *  regions the model literally cannot phonate get a local minimal-landing shift + inverse
 *  (as deep as THAT region truly needs, ±24 cap), everything else renders at its own pitch,
 *  bit-identical to extension-off;
 *  s85e = windowed donors (cover donors render only dead-region neighbourhoods ±1.5 s instead
 *  of K whole-song passes — the 5-6 min render regression) + level-match now scoped to the
 *  score path only (cover has no per-render normalization; a global RMS pull was measuring
 *  climax loudness against whole-song average). Audition tag bumps in lockstep (_s85e_). */
export const RANGE_ALGO_VERSION = "s85e";

/** Version of the LYRIC → PHONE layer (g2p.rs / score2cv.rs). Bump it whenever the phones a given
 *  lyric resolves to change — otherwise every already-baked segment keeps its OLD audio forever and
 *  the fix reads as "I changed it and the user hears nothing", exactly the failure RANGE_ALGO_VERSION
 *  exists to prevent (S81 audit; S86 review R3 caught this one missing).
 *  s86 = 「に」 resolves to `n i` (the phones the model was trained on) instead of `ɲ i`; whole-kana
 *  multi-mora parsing (ずっと/っと/あー sing in full instead of being truncated to the head mora);
 *  `rest` freed from the reserved rest tokens; tolerant dictionary lookup (ß→ss, typographic
 *  apostrophes, glued punctuation).
 *  s90 = OpenUtau phonetic hints (`[dh ae dh]` / `read[r iy d]` in the lyric pin that note's phones);
 *  stressless ARPABET finally carries a syllable nucleus, so a hint spreads over its `+` notes instead
 *  of collapsing onto the first one; a bare `ah` (no stress digit) reads as ə, not ʌ.
 *  ⚠ On the SHIPPED dictionaries the phone output is unchanged to the byte — all 69 ARPABET tokens /
 *  863018 instances of en.tsv judge identically under the old and new nucleus rule, and neither en.tsv
 *  nor the golden vectors contain a digit-less AH. What moves is user-typed input, which is the point.
 *  ★ There is NO Rust twin to keep in lockstep any more: the audition cache tag dropped its g2p term in
 *  S90 (that pipeline renders a fixed clip with no lyrics — see commands/audition.rs). THIS is the one
 *  place the lyric→phone layer is versioned. */
export const G2P_ALGO_VERSION = "s90";

/** Version of the note → FRAME allocation layer (buildScoreTriples). Bump it whenever the frame counts a
 *  given note set resolves to change — the timing twin of G2P_ALGO_VERSION, and for the same reason: a
 *  segment already baked keeps its OLD audio forever unless its signature moves, so the fix reads as
 *  "I changed it and the user hears nothing" (S86 review R3).
 *  s87 = the frame-borrow knife: a note whose span rounds to ZERO frames borrows one from a neighbour
 *  instead of being dropped, and the f0 feed is retimed with it.
 *  ⚠ FRONTEND-ONLY on purpose — no Rust counterpart to keep in lockstep: the allocation happens here, and
 *  the audition cache (audition.rs) renders its own fixed probe score that this cannot touch.
 *  (S83/S84's timing knives predate this token and shipped with "re-render old projects by hand"; new
 *  timing work should bump this instead.)
 *  s88 = a note carrying a REST token is silence everywhere, not only in Rust: it leaves the f0 pitch
 *  chain (the line breaks over it instead of gliding through), it lends frames at the rest floor, it no
 *  longer blocks a neighbour's rescue, and it is no longer warned about for being short. Written as a
 *  gap, the same music always behaved this way — this is the two spellings converging.
 *  ⚠ ALSO the invalidation carrier for the Rust-side fix in the same round (commands/inference.rs: the
 *  dead-zone planner's note list now zeroes silent tokens instead of reading their drawn pitch). That
 *  fix only reaches the SCORE render path, whose signature is exactly this one — but it does mean the
 *  "frontend-only" line above describes the TOKEN, not the round.
 *  s92 = the coda CLUSTER split (Rust `allocate_in_note`): with two or more coda consonants the
 *  outermost one no longer takes the whole budget and silently deletes the inner one — `don't` sang
 *  "dote", `means` "meez", `things` "thiz" (6 such notes on a real 283-note English track). Frame
 *  totals are unchanged; WHICH phones exist is not, so every existing bake must be re-judged.
 *  ⚠ zh/ja bakes re-render to BYTE-IDENTICAL audio (the ja probe song's lane dump is byte-identical
 *  before/after — n_coda ≥ 2 cannot occur in zh/ja/UTAU-alias material), so for those projects the
 *  re-render is wasted work rather than a change. Bumping anyway follows the user's S89 ruling:
 *  "re-running an existing bake is no big deal; triggering a re-render where nothing should have
 *  changed is the bug" — a per-language token would be the distortion, not the protection. */
export const SCORE_TIMING_VERSION = "s92";

/** 32-bit rolling hash — keeps the per-semitone scan in the signature without pasting ~1 KB of
 *  JSON into every dirty-check string. */
export function hash32(s: string): string {
  let h = 5381;
  for (let i = 0; i < s.length; i++) h = ((h * 33) ^ s.charCodeAt(i)) >>> 0;
  return h.toString(36);
}

/** S60-2 audit: the model's vocal_range record IS a render input (it decides the tier shift),
 *  so a re-test / comfort adjustment must dirty the bakes that used it — else Play keeps
 *  serving audio rendered under the OLD zone. Covers the usable/comfort bounds, the RAW
 *  per-semitone scan (S81: the decision layer now reads it as a continuous damage curve, so a
 *  re-test that moves only the scan must dirty too) and the algorithm version; gated off when
 *  the track opted out. */
function rangeRecordSig(track: Track): string {
  const vp = track.vocalParams ?? DEFAULT_VOCAL_PARAMS;
  if (vp.rangeExtend !== true || !track.voiceModel) return ""; // S62c: extension is opt-in (absent = OFF)
  const entry = useVoiceModelStore.getState().models[vp.backend]?.find((m) => m.name === track.voiceModel);
  const rec = (entry?.config as {
    vocal_range?: { speakers?: Record<string, { usable?: unknown; comfort?: unknown; semitones?: unknown }> };
  } | undefined)?.vocal_range;
  if (!rec?.speakers) return "";
  const body = Object.entries(rec.speakers)
    .map(([id, sp]) => `${id}=${JSON.stringify(sp?.usable)}~${JSON.stringify(sp?.comfort)}~${hash32(JSON.stringify(sp?.semitones ?? null))}`)
    .sort()
    .join(",");
  return `v${RANGE_ALGO_VERSION}|${body}`;
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
    seg.content.notes, seg.content.pitchDev, tempo, vp.transition, vocalTokens(vp),
    seg.content.paramCurves, vp.formant ?? 0, vp.langId,
  );
  // Nothing to sing → say so, LOUDLY. A segment whose triples are ALL rests would otherwise render
  // successfully and deposit a bake of pure silence with no mark anywhere: a rest can never be OOV (Rust
  // validates the canonical `R` we send) and is deliberately excluded from the short/dropped channels.
  // That is one keystroke away now — the rest token is free text, and `a` is the default lyric of five of
  // the seven languages — so "the whole track went quiet and nothing told me why" has to be impossible.
  // (Pre-S88 an all-`R` segment did exactly that; the knob only made it reachable.)
  if (!triples.some((t) => t.lyric !== "R")) throw new Error(VOCAL_EMPTY);
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
    options: vocalRenderOptions(vp),
    renderedSig: vocalRenderSig(track, seg, tempo),
  });
}

/** THE track-params → wire-options mapping, extracted so it can be tested at all.
 *
 *  ⚠ Every knob here is a place a per-track setting can go MISSING silently: the Rust struct is
 *  `#[serde(default)]`, so a field the frontend forgets lands on the production default while the
 *  sidebar happily shows the user's choice — "the switch does nothing" with no error anywhere. It was
 *  an inline literal until S91 and had ZERO test coverage; `vocalRender.test.ts` now pins it. */
export function vocalRenderOptions(vp: VocalTrackParams): VocalRenderOptions {
  return {
    backend: vp.backend,
    cv_speaker_id: vp.speakerId,
    lang_id: vp.langId,
    transpose: vp.transpose,
    // S60-2: absent = ON (no-op until the model carries a vocal_range record)
    range_extend: vp.rangeExtend === true, // S62c: opt-in (absent = OFF)
    consonant_emphasis_db: vp.consonantEmphasis ?? DEFAULT_CONSONANT_EMPHASIS_DB,
    consonant_valley: vp.consonantValley ?? DEFAULT_CONSONANT_VALLEY,
    vowel_clarity: vp.vowelClarity !== false,
    consonant_preroll: vp.consonantPreroll !== false,
    phoneme_set: vp.phonemeSet ?? null,
    sovits: { ...SOVITS_DEFAULTS, ...(vp.sovits ?? {}) },
    rvc: { ...RVC_DEFAULTS, ...(vp.rvc ?? {}) },
  };
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
      // S67c: vocal failures now reach the backend log (they were toast-only — invisible
      // in crash forensics); fatal modal-class errors open the alert dialog once and stop
      // repeating per track (the guidance is machine-wide, not per-segment).
      logToBackend("error", `Vocal auto-render failed (${tr.name}): ${String(e)}`);
      const display = `${tr.name}: ${vocalRenderErrorMessage(e)}`;
      if (maybeShowErrorModal(e, display)) {
        return { rendered, failed, cancelled: false };
      }
      // LOUD failure (§user: Play's auto-render must report exactly like the manual Render button — never
      // swallow). Same shared mapping, prefixed with the track name so the user knows WHICH track failed;
      // capped so a project-wide failure (e.g. missing dictionary) doesn't storm one toast per segment.
      if (failed <= MAX_FAILURE_TOASTS) {
        useAppStore.getState().showToast(display, "error");
      } else if (failed === MAX_FAILURE_TOASTS + 1) {
        useAppStore.getState().showToast(i18n.t("vocalEditor.render.moreFailures"), "error");
      }
    }
  }
  return { rendered, failed, cancelled: false };
}
