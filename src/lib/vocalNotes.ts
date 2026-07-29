// ② Vocal-note DATA HYGIENE — the SINGLE source of truth for normalizing / sanitizing / clamping vocal
// notes and their curves (S48 Phase 4a, §9.5/§9.8). Every write path funnels through here so the store,
// the .usp loader, AND the editor share ONE definition of "a canonical note":
//   - canonical shape (omit-default + fixed key/element order) keeps the raw-JSON save/autosave compare
//     byte-stable (§5 false-dirty) and the undo sig (history.contentSig) phantom-free (absent ≡ default),
//   - finite/bounds clamps + string sanitization treat a hand-edited or hostile `.usp` as untrusted input
//     (§9.8.1): a NaN tick / out-of-range pitch / control-char lyric can never reach the canvas or render.
// PURE (no store / no i18n / no async) so it is trivially unit-testable and importable from both the store
// and the loader without a cycle. The reserved-token / OOV classifier that decides rest/sustain/绿-render
// is Rust `validate_lyrics` (§9.5) — NOT here; the tie-mirror invariant lands in Phase 6 with render.
import type { Note, PitchCurve, NoteTransition, VocalTrackParams } from "../types/project";
import { RVC_DEFAULTS, SOVITS_DEFAULTS } from "./workflow/voiceDefaults";
import { isVocalLangCode } from "./vocal/languages"; // S58: Note.lang whitelist (id↔code single source)

// ── bounds (defensive; valid editor input never trips them, a corrupt file does) ──
export const PITCH_MIN = 0;
export const PITCH_MAX = 127;
/** Fine-detune bound in cents (±1 octave) — a loaded |detune| beyond this is clamped, not trusted. */
export const DETUNE_CAP = 1200;
export const MAX_LYRIC_LEN = 64;
/** DoS caps: a single note's control points, a segment's note count, a curve's point count. A malicious
 *  `.usp` with a million points can't blow up the canvas / a Phase-6 render index. */
export const MAX_POINTS_PER_NOTE = 512;
export const MAX_NOTES_PER_SEGMENT = 100_000;
export const MAX_CURVE_POINTS = 100_000;

const HUGE = 1e9;

/** Track-level DEFAULT note transition — every field concrete (a note's partial NoteTransition overrides
 *  per field). This is why every note glides smoothly by default (SynthV, §10.3). */
export const DEFAULT_TRANSITION = { offsetMs: 0, durLeftMs: 100, durRightMs: 70, depthLeftCents: 15, depthRightCents: 15, openEdgeCents: 200 } as const;

/** S83 knife 6b: default voiceless-onset emphasis in dB (the SynthV "consonant strength" analogue).
 *  Mirrors Rust score2svc::DEFAULT_VOICELESS_ONSET_EMPHASIS_DB — the wire default for old payloads
 *  and the model-audition path. 0 = off (exact no-op render-side). */
export const DEFAULT_CONSONANT_EMPHASIS_DB = 2.5;
/** S84 C 刀: default chain-internal consonant-valley scale (×measured per-class depth — the fast-run
 *  粘连 treatment). Mirrors Rust score2svc::DEFAULT_CONSONANT_VALLEY_SCALE. 0 = off (exact no-op). */
export const DEFAULT_CONSONANT_VALLEY = 1;
/** 纯阶梯覆盖(全 0 = 无滑音/无起收):S73 ustx 烤入 part 的音符显式置它——OU 的全部音高
 *  动态已在 pitchDev 曲线里,任何默认滑音再叠加=双重。显式 0 ≠ 继承(normalizeTransition 保留)。 */
export const ZERO_TRANSITION = { offsetMs: 0, durLeftMs: 0, durRightMs: 0, depthLeftCents: 0, depthRightCents: 0, openEdgeCents: 0 } as const;
const TRANSITION_KEYS = ["offsetMs", "durLeftMs", "durRightMs", "depthLeftCents", "depthRightCents", "openEdgeCents"] as const;
const TRANSITION_BOUNDS: Record<(typeof TRANSITION_KEYS)[number], readonly [number, number]> = {
  offsetMs: [-2000, 2000], durLeftMs: [0, 2000], durRightMs: [0, 2000], depthLeftCents: [-1200, 1200], depthRightCents: [-1200, 1200], openEdgeCents: [0, 1200],
};
/** Canonicalize a PARTIAL NoteTransition: keep only finite fields (clamped) in a FIXED key order so a
 *  per-note override serializes byte-stable; drop non-finite. Empty object if nothing survives (caller omits). */
function normalizeTransition(t: NoteTransition): NoteTransition {
  const out: NoteTransition = {};
  for (const k of TRANSITION_KEYS) {
    const v = t[k];
    if (typeof v === "number" && Number.isFinite(v)) out[k] = clampNum(v, TRANSITION_BOUNDS[k][0], TRANSITION_BOUNDS[k][1], 0);
  }
  return out;
}

/** Round to int + clamp to [lo,hi]; a non-finite value falls back (never propagates NaN/Infinity). */
function clampInt(v: number, lo: number, hi: number, fallback: number): number {
  return Number.isFinite(v) ? Math.min(hi, Math.max(lo, Math.round(v))) : fallback;
}
/** Clamp a finite float to [lo,hi]; non-finite → fallback. */
function clampNum(v: number, lo: number, hi: number, fallback: number): number {
  return Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : fallback;
}

/**
 * THE single string-sanitization choke (§9.5/§9.8.4): NFC-normalize, strip every Unicode Control/Format/
 * Surrogate/Private-Use/Unassigned code point (`\p{C}` — covers C0/C1 controls incl. NUL, zero-width,
 * bidi overrides, the BOM, lone surrogates), and cap the length. Applied to every user string that enters
 * a Note (lyric / phoneme / phonemeInput / lang) on inline edit, batch, paste AND load — so a hostile
 * `.usp` can't inject control/bidi glyphs or an unbounded string, and the same input always canonicalizes
 * the same way (byte-stable). A space (Zs) is kept; tab/newline (Cc) are stripped — lyrics are single-line.
 */
export function sanitizeText(s: string, maxLen: number = MAX_LYRIC_LEN): string {
  return (typeof s === "string" ? s : "").normalize("NFC").replace(/\p{C}/gu, "").slice(0, maxLen);
}

/**
 * Canonicalize ONE vocal note. Rebuilds the 6 base fields (finite-clamped) then re-adds each optional
 * ONLY when non-default, in a FIXED key/element order — so the same logical note always serializes to the
 * same bytes regardless of how a caller built it (the omit-default + canonical-order rule the undo sig in
 * history.noteSig folds to match). Also the load-safety clamp point (§9.8.1): out-of-range/NaN numbers are
 * clamped, strings sanitized, bad shapes coerced. Idempotent (normalizeNote(normalizeNote(n)) === …).
 * NOTE tie is kept as-is (settable) in Phase 4; the §9.5 tie-mirror (tie ≡ isSustainToken(lyric)) enforces
 * in Phase 6 when render reads it — forcing it now would break nothing at render but churn Phase-3 tests
 * for zero editor benefit.
 */
export function normalizeNote(n: Note): Note {
  const out: Note = {
    id: String(n.id),
    tick: clampInt(n.tick, 0, HUGE, 0),
    duration: clampInt(n.duration, 1, HUGE, 1),
    pitch: clampInt(n.pitch, PITCH_MIN, PITCH_MAX, 60),
    lyric: sanitizeText(n.lyric),
    velocity: clampInt(n.velocity, 0, 127, 100),
  };
  if (n.phoneme) out.phoneme = sanitizeText(n.phoneme);
  if (n.detune && Number.isFinite(n.detune)) out.detune = clampNum(n.detune, -DETUNE_CAP, DETUNE_CAP, 0);
  // ② Per-note transition override (SynthV): keep only finite fields (clamped) in fixed key order; all-
  //    absent → omit entirely (§5 omit-default, so a partial override can't false-dirty).
  if (n.transition) {
    const tr = normalizeTransition(n.transition);
    if (Object.keys(tr).length > 0) out.transition = tr;
  }
  // ④ Vibrato (SynthV): all fields finite AND a real amplitude (depthCents > 0), else canonicalize to
  //    ABSENT — a zero-amplitude vibrato and no vibrato must be byte-identical (§5 false-dirty; 审查 #3).
  if (n.vibrato) {
    const v = n.vibrato;
    if ([v.depthCents, v.freqHz, v.phase, v.startMs, v.easeInMs, v.easeOutMs].every((x) => Number.isFinite(x)) && v.depthCents > 0) {
      out.vibrato = {
        depthCents: clampNum(v.depthCents, 0, 2400, 0), freqHz: clampNum(v.freqHz, 0.1, 40, 5.5), phase: clampNum(v.phase, -1, 1, 0),
        startMs: clampNum(v.startMs, 0, 60000, 0), easeInMs: clampNum(v.easeInMs, 0, 10000, 0), easeOutMs: clampNum(v.easeOutMs, 0, 10000, 0),
      };
    }
  }
  if (n.pitchAuto === false) out.pitchAuto = false; // absent/true (default) → absent
  if (n.tie) out.tie = true; // false → absent (Phase-6 mirror enforces tie ≡ sustain-lyric)
  if (n.autoTuned === true) out.autoTuned = true; // S73 机器调教所有权;false → absent
  // S58: lang is WHITELISTED to the 7 codes (anything else → absent = follow the track default) so a
  // corrupt/hand-edited .usp can never smuggle an arbitrary string into the render's lang resolution.
  if (n.lang && isVocalLangCode(n.lang)) out.lang = n.lang;
  if (n.phonemeInput) out.phonemeInput = sanitizeText(n.phonemeInput);
  return out;
}

/**
 * THE single write funnel for a notes array (§9.5): drop malformed (id-less) notes, cap the count (load
 * DoS guard), normalize each, and SORT by (tick, id). Storage order becomes a pure function of content →
 * no false-dirty from insertion order, and draw / hit-test can rely on the stored order (no per-frame
 * copy-sort). Every write entrypoint (the store note actions, applyNoteEdits, createVocalPart, and the
 * `.usp` loader) passes through here, so a saved file's baseline sig is captured in canonical order too
 * (a legacy file's first edit can't false-dirty). Idempotent.
 */
export function normalizeNotesArray(notes: Note[], cap: number = MAX_NOTES_PER_SEGMENT): Note[] {
  return (Array.isArray(notes) ? notes : [])
    .filter((n): n is Note => !!n && typeof n.id === "string" && n.id.length > 0)
    .slice(0, cap)
    .map(normalizeNote)
    .sort((a, b) => a.tick - b.tick || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

/**
 * Canonicalize a pitch/param curve (§9.5): drop non-finite points, round X→int tick, round Y by KIND
 * (`cents` = integer cents for pitchDev; `param` = 0.001 quantum for loudness/tension/… — NEVER integer
 * cents, which would flatten a loudness lane into a staircase), sort + dedup by X (strictly increasing),
 * cap the count. Empty → undefined (the caller clears the field). Keeps sig↔serialize byte-stable when the
 * Phase-5 editor writes float curves.
 */
export function normalizeCurve(curve: PitchCurve | undefined, kind: "cents" | "param"): PitchCurve | undefined {
  if (!curve || !Array.isArray(curve.xs) || !Array.isArray(curve.ys)) return undefined;
  const roundY = kind === "cents" ? (y: number) => Math.round(y) : (y: number) => Math.round(y * 1000) / 1000;
  const pts: { x: number; y: number }[] = [];
  const n = Math.min(curve.xs.length, curve.ys.length, MAX_CURVE_POINTS);
  for (let i = 0; i < n; i++) {
    const x = curve.xs[i]!;
    const y = curve.ys[i]!;
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    pts.push({ x: Math.round(x), y: roundY(y) });
  }
  pts.sort((a, b) => a.x - b.x);
  const xs: number[] = [];
  const ys: number[] = [];
  for (const p of pts) {
    if (xs.length > 0 && p.x === xs[xs.length - 1]) {
      ys[ys.length - 1] = p.y; // duplicate x → last value wins (a paint pass overwrites)
      continue;
    }
    xs.push(p.x);
    ys.push(p.y);
  }
  return xs.length > 0 ? { xs, ys } : undefined;
}

/**
 * Retime a note by a start delta and a duration delta. ONE helper shared by user resize AND the one-note-
 * per-position truncation (§9.5). SynthV transition/vibrato are keyed in ABSOLUTE ms (NOT note-relative
 * ticks), so moving/resizing a note needs NO pitch-shape rebase — just clamp tick/duration + re-normalize.
 */
export function retimeNote(note: Note, dTickStart: number, dDuration: number): Note {
  const tick = Math.max(0, note.tick + dTickStart);
  const duration = Math.max(1, note.duration + dDuration);
  return normalizeNote({ ...note, tick, duration });
}

/**
 * One-position-one-note (§9.2, §5): given the post-edit notes and the set of ACTIVE (just added/moved/
 * resized) note ids, clip every PASSIVE note so it no longer overlaps ANY active interval, over the WHOLE
 * array in tick order — NOT just each active note's immediate neighbor (the multi-select-resize
 * committed-overlap bug). Half-open intervals `[tick, tick+duration)`: an abutting boundary
 * (`passive.end === active.start`) is legato-legal, not overlap. A passive note swallowed by an active
 * one, or clipped below `minTicks`, is dropped. Active notes pass through untouched (they win). Pure.
 */
export function resolveOverlaps(notes: Note[], activeIds: Set<string>, minTicks: number): Note[] {
  const active = notes
    .filter((n) => activeIds.has(n.id))
    .map((n) => ({ start: n.tick, end: n.tick + n.duration }))
    .sort((a, b) => a.start - b.start);
  if (active.length === 0) return notes;
  const out: Note[] = [];
  for (const n of notes) {
    if (activeIds.has(n.id)) {
      out.push(n);
      continue;
    }
    let start = n.tick;
    let end = n.tick + n.duration;
    for (const a of active) {
      if (a.end <= start || a.start >= end) continue; // disjoint / abutting (legato-legal)
      if (a.start <= start)
        start = Math.max(start, a.end); // active covers/precedes head → push head to its end
      else end = Math.min(end, a.start); // active starts inside → truncate tail to its start (keep head)
    }
    const dur = end - start;
    if (dur < minTicks) continue; // swallowed / clipped below the audible floor → drop
    out.push(start === n.tick && dur === n.duration ? n : retimeNote(n, start - n.tick, dur - n.duration));
  }
  return out;
}

/** The canonical breath token Rust hard-wires (`AP`/`ap`, g2p.rs `token_class`) — also the default value
 *  of the per-track trigger, so the default is a no-op over the hard-wired behaviour. */
export const DEFAULT_BREATH_TOKEN = "AP";
/** The canonical rest token Rust hard-wires (`R`/`r`/empty, same place). S86 deliberately NARROWED that
 *  set — `rest`/`sil`/`pau` are ordinary singable words again — and this per-track trigger is how a user
 *  gets a convenient rest glyph back without any common word being stolen from every language at once. */
export const DEFAULT_REST_TOKEN = "R";

/** A track's two lyric triggers, resolved. NAMED on purpose: `f(lyric, breathToken, restToken)` would put
 *  two same-typed strings side by side through several call layers, and a swap there is invisible to the
 *  compiler, to tests that use the same value for both, and to review (S85 血训: ≥2 same-typed semantic
 *  fields must be a named struct). */
export interface VocalTokens {
  /** Lyric that renders as an audible inhale. Canonical `AP`/`ap` always work too. */
  breath: string;
  /** Lyric that renders as a rest. Canonical `R`/`r`/empty always work too. */
  rest: string;
}

/** Resolve a track's params to its tokens — THE single place the two defaults live, so no call site can
 *  drift by forgetting a `?? "AP"`. An explicitly EMPTY token stays empty (only the canonical forms then
 *  trigger); that is exactly what clearing the sidebar box should mean. */
export function vocalTokens(p?: VocalTrackParams): VocalTokens {
  return { breath: p?.breathToken ?? DEFAULT_BREATH_TOKEN, rest: p?.restToken ?? DEFAULT_REST_TOKEN };
}

/**
 * M3 breath: whether a lyric token renders as an audible INHALE — the canonical `AP`/`ap` the Rust
 * classifier hard-wires, OR the track's user-chosen breath trigger (VocalTrackParams.breathToken). A breath
 * is UNVOICED and breaks the pitch line: f0eval + the render drop it from the connected pitch chain, so the
 * previous note gets the §10.5 release-drift (段中尾音) and the next note the onset-scoop. Dynamic — changing
 * breathToken re-classifies the OLD token's notes (they become normal, connected lyrics again).
 */
export function isBreathLyric(lyric: string, breathToken: string): boolean {
  const l = lyric.trim();
  const bt = breathToken.trim();
  return CANONICAL_BREATH.includes(l) || (bt !== "" && l === bt);
}

/** The breath spellings Rust hard-wires (g2p.rs `token_class`) — always live, whatever the trigger says. */
const CANONICAL_BREATH: readonly string[] = ["AP", "ap"];
/** Ditto for rests. An EMPTY lyric is in the set because that is what Rust classifies, not because the
 *  editor can produce one (clearing a lyric substitutes the language's default lyric). */
const CANONICAL_REST: readonly string[] = ["R", "r", ""];

/**
 * The part of a trigger that can actually change classification: EMPTY when the token adds nothing beyond
 * the hard-wired set — i.e. absent, blank, or one of the canonical spellings themselves.
 *
 * ★ THE canonicalizer any SIGNATURE must run the token through. `isRestLyric` trims the token and ORs it
 * onto the canonical set, so `""`, `"R"`, `"r"` and `"R "` classify byte-identically; hashing the raw
 * string instead declares every bake on the track dirty for a change that cannot alter one sample. Worse,
 * it does not even stay decided: `sanitizeVocalParams` rewrites a blank token back to the canonical one on
 * load, so the project reads dirty before saving and clean after reopening (S88 review, CONFIRMED).
 */
export function restTokenKey(restToken: string | undefined): string {
  const t = (restToken ?? DEFAULT_REST_TOKEN).trim();
  return CANONICAL_REST.includes(t) ? "" : t;
}
/** The breath twin of `restTokenKey` — same reasoning, same wart it removes (the pre-S88 `bt:` term
 *  hashed the raw string unconditionally, so clearing the breath box false-dirtied a whole track too). */
export function breathTokenKey(breathToken: string | undefined): string {
  const t = (breathToken ?? DEFAULT_BREATH_TOKEN).trim();
  return CANONICAL_BREATH.includes(t) || t === "" ? "" : t;
}

/**
 * Whether a lyric token means A REST — the canonical `R`/`r`/empty Rust hard-wires, OR the track's chosen
 * trigger (VocalTrackParams.restToken). A rest note is the exact equivalent of leaving a GAP: no sung
 * phones (Rust's `token_class` sees the canonical `R` the render maps it to) AND no pitch, so the line
 * breaks over it and its neighbours become phrase edges. Dynamic, like the breath token.
 */
export function isRestLyric(lyric: string, restToken: string): boolean {
  const l = lyric.trim();
  const rt = restToken.trim();
  return CANONICAL_REST.includes(l) || (rt !== "" && l === rt);
}

/** Does this lyric carry NO pitch — a rest (silent) or a breath (unvoiced inhale)? The frontend twin of
 *  Rust `g2p::is_silent_token`, and THE single predicate for "this note is out of the pitch chain".
 *  Consumers, all of which must agree or the drawn line lies about the audio: the render f0 feed
 *  (vocalRender), the editor overlay + preview tone + PITCH-PAINT COMMIT (VocalEditor's `pitchChain` —
 *  the commit stores deltas against this line, so it diverging writes a deviation nobody drew), and the
 *  auto-tune payload (autoTune — the θ model infers phrase edges from note adjacency alone).
 *  Rest is tested FIRST so that setting both tokens to the same string resolves to silence: a rest is
 *  silent, a breath is audible, and the quieter reading is the safe one. */
export function isSilentLyric(lyric: string, tokens: VocalTokens): boolean {
  return isRestLyric(lyric, tokens.rest) || isBreathLyric(lyric, tokens.breath);
}

/**
 * Sanitize loaded track vocal params (§9.8.1): a corrupt `.usp` can carry a bad backend / out-of-range
 * speaker or lang id / non-finite transpose that later mis-indexes a ScoreToCV input. Absent → undefined.
 */
export function sanitizeVocalParams(p: VocalTrackParams | undefined): VocalTrackParams | undefined {
  if (!p || typeof p !== "object") return undefined;
  const tr = (p.transition && typeof p.transition === "object" ? p.transition : {}) as Partial<NoteTransition>;
  return {
    backend: p.backend === "rvc" ? "rvc" : "sovits",
    speakerId: clampInt(p.speakerId, 0, 76, 49),
    langId: clampInt(p.langId, 0, 6, 2),
    transpose: clampInt(p.transpose, -48, 48, 0),
    formant: clampNum(p.formant ?? NaN, -24, 24, 0),
    transition: {
      offsetMs: clampNum(tr.offsetMs ?? NaN, -2000, 2000, DEFAULT_TRANSITION.offsetMs),
      durLeftMs: clampNum(tr.durLeftMs ?? NaN, 0, 2000, DEFAULT_TRANSITION.durLeftMs),
      durRightMs: clampNum(tr.durRightMs ?? NaN, 0, 2000, DEFAULT_TRANSITION.durRightMs),
      depthLeftCents: clampNum(tr.depthLeftCents ?? NaN, -1200, 1200, DEFAULT_TRANSITION.depthLeftCents),
      depthRightCents: clampNum(tr.depthRightCents ?? NaN, -1200, 1200, DEFAULT_TRANSITION.depthRightCents),
      openEdgeCents: clampNum(tr.openEdgeCents ?? NaN, 0, 1200, DEFAULT_TRANSITION.openEdgeCents),
    },
    sovits: sanitizeOpts(p.sovits, SOVITS_DEFAULTS),
    rvc: sanitizeOpts(p.rvc, RVC_DEFAULTS),
    // The two lyric triggers (§9.8.1). They are compared against note lyrics, which ALWAYS pass through
    // sanitizeText — so a token carrying control/format code points could never match anything; sanitizing
    // it here keeps the stored value comparable (and bounded) without changing which notes classify.
    breathToken: sanitizeToken(p.breathToken, DEFAULT_BREATH_TOKEN),
    restToken: sanitizeToken(p.restToken, DEFAULT_REST_TOKEN),
    ...(p.rangeExtend === true ? { rangeExtend: true } : {}),
    // S73b/c 自动音高:follow 只存 false(absent≡true=默认常开);表现力 0–4 默认 2、
    // 颤音 0–2 默认 1(S73c 用户拍板:双乘后 4×4 太恐怖)。
    ...(p.autoTuneFollow === false ? { autoTuneFollow: false } : {}),
    autoTuneExpr: clampNum(p.autoTuneExpr ?? NaN, 0, 4, 2),
    autoTuneVib: clampNum(p.autoTuneVib ?? NaN, 0, 2, 1),
    autoTuneTake: clampInt(p.autoTuneTake ?? NaN, 0, 99, 0),
    // S83 knife 6b: consonant strength knob (dB on voiceless onsets; 0 = off).
    consonantEmphasis: clampNum(p.consonantEmphasis ?? NaN, 0, 12, DEFAULT_CONSONANT_EMPHASIS_DB),
    // S84 C 刀: consonant-valley scale knob (×per-class depth on chain-internal boundaries; 0 = off).
    consonantValley: clampNum(p.consonantValley ?? NaN, 0, 2, DEFAULT_CONSONANT_VALLEY),
    // S84 E 刀: vowel clarity toggle — only false is stored (absent≡true, autoTuneFollow pattern).
    ...(p.vowelClarity === false ? { vowelClarity: false } : {}),
  };
}

/** A lyric trigger from an untrusted file: sanitized like a lyric, and falling back to the canonical token
 *  when nothing printable survives (an all-blank token would otherwise mean "no trigger" silently). */
function sanitizeToken(raw: unknown, fallback: string): string {
  const s = typeof raw === "string" ? sanitizeText(raw) : "";
  return s.trim() ? s : fallback;
}

/** Keep only known contract keys from a quality-override bag + drop non-finite numbers; the Rust serde
 *  re-validates the full shape (an unknown-shape value errors loudly there). Absent → undefined. */
function sanitizeOpts<T extends object>(raw: unknown, defaults: T): Partial<T> | undefined {
  if (!raw || typeof raw !== "object") return undefined;
  const src = raw as Record<string, unknown>;
  const out: Record<string, unknown> = {};
  for (const k of Object.keys(defaults)) {
    if (!(k in src)) continue;
    const v = src[k];
    if (typeof v === "number" && !Number.isFinite(v)) continue;
    out[k] = v;
  }
  return Object.keys(out).length ? (out as Partial<T>) : undefined;
}
