// S87 — grid quantization for IMPORTED / EXTRACTED notes. ONE implementation, shared by score import
// (import.ts: ust / ustx / midi) and GAME vocal→MIDI extraction (midiExtract.ts), so the two can never
// drift apart the way they had (import did no rounding at all, extraction rounded unconditionally).
//
// WHY it is OPTIONAL (§user; baselines measured over four real scores in S86): rounding is the CURE for
// one class of file and POISON for another.
//   · 鹅妈妈 (ja .ust): 2.0% of starts off-grid — rounding fixes its two 30t-@-tempo-222 notes, which are
//     16.9 ms, i.e. SHORTER THAN ONE 20 ms render frame, so they round to zero frames and never sound.
//   · Main.ust (English X-SAMPA CVVC): 82.6% of starts off-grid and 93.5% of notes would MOVE — those
//     offsets are the alias author's hand-made preutterance compensation, i.e. the musical intent itself.
// Hence a per-import checkbox, never an unconditional rule.
//
// WHAT it does — BOUNDARY quantization in ABSOLUTE tick space (the v1 frame-drift rule, inherited from
// midiExtract's header): each edge rounds independently, so the SHARED boundary of two contiguous notes
// rounds identically and legato stays legato. Rounding each note's DURATION instead would accumulate drift.
//
// COLLAPSE is NOT a drop. A note shorter than half a cell has both edges land on the same line; midiExtract
// used to `continue` past it — a silent vanish, the exact failure class this project banned after S84/S85.
// Here it is WIDENED to exactly one cell, and the next note yields that cell from its START only (its END
// is untouched, so nothing downstream shifts). The caller gets counts to report out loud.

import { TICKS_PER_BEAT } from "../constants";

/** The quantum: 1/12 of a beat = 40t @ 480 tpq. Covers binary (16th = 3 units) AND ternary (8th-triplet =
 *  4 units) subdivisions, so straight and swung material both land on-grid. It is also the finest line the
 *  vocal editor draws, the editor's move/resize snap unit, and the shortest note the UI offers. */
export const GRID_QUANT_TICKS = TICKS_PER_BEAT / 12;

/** The remembered default of the "round to the 1/12 grid" checkbox (§user: default ON). ONE key for both
 *  prompts — score import and vocal→MIDI extraction are the same user intent, so they share the memory. */
export const QUANTIZE_IMPORT_KEY = "utai.quantizeImport";

/** A note's half-open span `[start, end)` in ABSOLUTE timeline ticks. */
export interface QuantSpan {
  start: number;
  end: number;
}

/**
 * What to do with a note whose two edges land on the SAME line (shorter than half a cell). The right answer
 * depends on the PRODUCER, so the caller must say — it is never a silent drop either way:
 *  · "widen" — score import. Every note is authored and carries a LYRIC; losing one loses a word. The note
 *    becomes exactly one cell and the next note yields that cell from its start.
 *  · "drop" — vocal→MIDI extraction. The notes are pitch-tracker output with placeholder lyrics, and GAME
 *    emits an exactly-contiguous chain, so a micro-note is a transition artifact, not a word. Widening those
 *    would push every following REAL note one cell later, accumulating across a run (a 20 ms transition pair
 *    at tempo 60 drifted the following sustain by 300+ ms in review) — precisely the sync with the source
 *    audio that transcription exists to preserve. Dropped notes are COUNTED and reported.
 */
export type CollapsePolicy = "widen" | "drop";

export interface QuantResult {
  /** Quantized spans, in the SAME index order as the input (zip them back onto your own note objects).
   *  `null` = the note collapsed under the "drop" policy; never null under "widen". */
  spans: (QuantSpan | null)[];
  /** How many surviving notes had a boundary changed at all (⊇ `widened`). */
  moved: number;
  /** How many notes were GENUINELY shorter than half a cell and got widened to one cell ("widen" only).
   *  A note merely pushed by a preceding widen is NOT counted here — it was never too short. */
  widened: number;
  /** How many notes collapsed and were dropped ("drop" only). */
  dropped: number;
}

/**
 * Quantize absolute note spans to `unit`. Input may be in any order and may overlap.
 *
 * Overlapping input (a chord in a .mid) is deliberately left overlapping. Independent edge rounding is
 * monotone, so for a DISJOINT pair it can never create an overlap; the only thing that can is the
 * collapse-widening, and the anti-overlap step is anchored to exactly that — see `pushTo` below.
 */
export function quantizeSpans(
  spans: readonly QuantSpan[],
  policy: CollapsePolicy = "widen",
  unit: number = GRID_QUANT_TICKS,
): QuantResult {
  const q = (t: number) => Math.round(t / unit) * unit;
  const order = spans
    .map((_, i) => i)
    .sort((a, b) => spans[a]!.start - spans[b]!.start || spans[a]!.end - spans[b]!.end || a - b);
  const out = new Array<QuantSpan | null>(spans.length);
  let moved = 0;
  let widened = 0;
  let dropped = 0;
  // The anti-overlap push is anchored to the last note that CLAIMED space it did not naturally own (a
  // widen): `pushTo` = the end it claimed, `pushFrom` = its ORIGINAL end (the disjointness test).
  // ⚠ Anchoring this to a running max over ALL notes instead is wrong: one long held note raises the
  // watermark above every following short note's start, the push is skipped for notes that WERE disjoint
  // from each other, and after widening three of them land on ONE identical span (review finding — a
  // polyphonic .mid turned a three-note run into three stacked notes at the same tick).
  let pushTo = -Infinity;
  let pushFrom = Infinity;
  for (const i of order) {
    const s = spans[i]!;
    const qsRaw = q(s.start);
    const qeRaw = q(s.end);
    const collapsed = qeRaw <= qsRaw; // genuinely under half a cell — judged BEFORE any push
    if (collapsed && policy === "drop") {
      out[i] = null;
      dropped++;
      continue;
    }
    const qs = s.start >= pushFrom ? Math.max(qsRaw, pushTo) : qsRaw;
    let qe = qeRaw;
    if (qe < qs + unit) qe = qs + unit;
    if (collapsed) widened++;
    if (qs !== s.start || qe !== s.end) moved++;
    out[i] = { start: qs, end: qe };
    if (qe > qeRaw) {
      pushTo = qe; // it took space past its own rounded end ⇒ the next disjoint note must yield
      pushFrom = s.end;
    }
  }
  return { spans: out, moved, widened, dropped };
}

/** How many of these notes rounding would actually MOVE — the number shown in the import options dialog so
 *  the choice is informed (a CVVC score reads ~80%+ here; a quantized ja UST reads ~2%).
 *  ⚠ BOTH edges: counting only starts made a DAW export whose starts are on-grid but whose lengths carry a
 *  90% note gate report "0 (0%) off the grid" and then rewrite every duration (audit-caught). */
export function offGridCount(spans: readonly QuantSpan[], unit: number = GRID_QUANT_TICKS): number {
  let n = 0;
  for (const s of spans) if (s.start % unit !== 0 || s.end % unit !== 0) n++;
  return n;
}
