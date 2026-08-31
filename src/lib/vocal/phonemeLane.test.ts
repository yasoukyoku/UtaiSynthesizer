import { describe, it, expect } from "vitest";
import { phonemeLaneSig, phonemeLaneRequest, type PhonemeLaneInputs } from "./phonemeLane";
import { DEFAULT_BREATH_TOKEN, DEFAULT_REST_TOKEN } from "../vocalNotes";
import type { Note } from "../../types/project";

const note = (id: string, tick: number, duration: number, pitch: number, lyric: string): Note => ({
  id, tick, duration, pitch, lyric, velocity: 100,
});

function base(): PhonemeLaneInputs {
  return {
    notes: [note("a", 0, 480, 60, "か"), note("b", 480, 480, 62, "た")],
    tempo: 120,
    tokens: { breath: DEFAULT_BREATH_TOKEN, rest: DEFAULT_REST_TOKEN },
    langId: 2,
    consonantPreroll: true,
    // explicitly present-but-undefined: the default (words), while keeping the KEY visible to the
    // completeness check below — an optional field that is simply omitted would slip past it.
    phonemeSet: undefined,
    esDialect: undefined,
  };
}

describe("phoneme lane — the cache key and the IPC payload come from ONE inputs object", () => {
  // ★ THE structural test. The lane skips its IPC whenever the signature is unchanged, so any input
  // that reaches `preview_vocal_phonemes` without also reaching the signature leaves the lane painting
  // a stale answer. S88 shipped that exact shape once (AutoTuneWatcher's skip-sig omitted the lyric
  // tokens ⇒ changing a token never re-tuned), and a mutation test on the S89 preroll switch proved
  // the old inline signature was green with the switch removed from it.
  //
  // Rather than one assertion per field (which the NEXT field would silently skip), this enumerates
  // the input object's own keys: adding a field to PhonemeLaneInputs without perturbing it here fails
  // the completeness check below, and a field that does not move the signature fails its own case.
  const perturb: Record<keyof PhonemeLaneInputs, (i: PhonemeLaneInputs) => PhonemeLaneInputs> = {
    notes: (i) => ({ ...i, notes: [...i.notes, note("c", 960, 480, 64, "な")] }),
    tempo: (i) => ({ ...i, tempo: 222 }),
    tokens: (i) => ({ ...i, tokens: { ...i.tokens, rest: "休" } }),
    langId: (i) => ({ ...i, langId: 1 }),
    consonantPreroll: (i) => ({ ...i, consonantPreroll: false }),
    phonemeSet: (i) => ({ ...i, phonemeSet: "vccv" }),
    esDialect: (i) => ({ ...i, esDialect: "latam" }),
  };

  it("every field of PhonemeLaneInputs moves the signature", () => {
    const b = base();
    const sig = phonemeLaneSig(b);
    for (const [key, mutate] of Object.entries(perturb)) {
      expect(phonemeLaneSig(mutate(b)), `field ${key} does not reach the lane's cache key`).not.toBe(sig);
    }
  });

  it("★ the perturbation table covers EVERY field (a new input cannot be added untested)", () => {
    expect(Object.keys(perturb).sort()).toEqual(Object.keys(base()).sort());
  });

  it("the signature is stable for identical inputs (no ordering / identity leakage)", () => {
    expect(phonemeLaneSig(base())).toBe(phonemeLaneSig(base()));
  });

  it("S167: a note's phoneTiming edit moves the signature (the preview returns a different split)", () => {
    const b = base();
    const edited = {
      ...b,
      notes: [{ ...b.notes[0]!, phoneTiming: { phones: ["t", "a"], scale: [2, 1] } }, b.notes[1]!],
    };
    expect(phonemeLaneSig(edited)).not.toBe(phonemeLaneSig(b));
    // …and rides the wire (through the triples the request builds)
    const req = phonemeLaneRequest(edited);
    const carried = req.args.score.some((t) => t.phone_edit?.phones.join(",") === "t,a");
    expect(carried, "the edit must reach the preview payload").toBe(true);
  });

  it("the IPC payload carries the switch, and the notes' own timing", () => {
    const on = phonemeLaneRequest(base());
    expect(on.args.consonantPreroll).toBe(true);
    expect(on.args.defaultLang).toBe(2);
    expect(on.args.score.length).toBeGreaterThan(0);
    expect(on.args.phonemeSet).toBe(null); // S91: absent → the words default, explicit on the wire
    expect(on.args.esDialect).toBe(null); // S167: absent → the dictionary default, explicit on the wire
    expect(phonemeLaneRequest({ ...base(), esDialect: "castilian" }).args.esDialect).toBe("castilian");
    const off = phonemeLaneRequest({ ...base(), consonantPreroll: false });
    expect(off.args.consonantPreroll).toBe(false);
    expect(phonemeLaneRequest({ ...base(), phonemeSet: "xsampa" }).args.phonemeSet).toBe("xsampa");
    // the switch is a BACKEND allocation decision — the triples themselves are identical, which is
    // why the lane cannot show it without actually re-issuing the call (hence the signature).
    expect(off.args.score).toEqual(on.args.score);
  });
});

import { boundaryDraggableAfter, redistributeConserving } from "./phonemeLane";

describe("S167c: redistributeConserving is a faithful mirror of Rust redistribute_conserving", () => {
  it("★ the SHARED vectors (score2cv.rs::redistribute_conserving_floors_sums_and_is_deterministic)", () => {
    // ⛔ 改这里必须同步 Rust 侧同名测试的向量 —— 两份向量一致是「预览 == 提交」的承重判据。
    expect(redistributeConserving(10, [6, 8])).toEqual([4, 6]);
    expect(redistributeConserving(3, [100, 1, 1])).toEqual([1, 1, 1]); // spare 0 → floor 1 each
    expect(redistributeConserving(7, [0, 0])).toEqual([1, 6]); // degenerate weights fail safe
    for (const w of [[1, 1, 1], [5, 1, 1], [0.1, 0.1, 9]]) {
      const out = redistributeConserving(17, w);
      expect(out.reduce((a, b) => a + b, 0)).toBe(17);
      expect(out.every((d) => d >= 1)).toBe(true);
    }
  });
  it("documents the old snap-back: a pair dragged to (1,11) commits as (2,10) — preview must show (2,10)", () => {
    // base [6,6], drag left to 1 frame ⇒ scales (1/6, 11/6) ⇒ weights (6×0.167, 6×1.833)
    const w = [6 * Math.max(0.1, Math.round((1 / 6) * 1000) / 1000), 6 * Math.round((11 / 6) * 1000) / 1000];
    expect(redistributeConserving(12, w)).toEqual([2, 10]);
  });
  it("deterministic tie-break follows ascending index (Rust: .then(a.cmp(&b)))", () => {
    expect(redistributeConserving(5, [1, 1, 1])).toEqual([2, 2, 1]);
  });
});

describe("S167c: boundaryDraggableAfter — hit-test and painted handles share ONE predicate", () => {
  const ids: { [evt: number]: string | undefined } = { 0: "n0", 1: "n1", 2: undefined };
  it("a same-note junction is draggable", () => {
    const spans = [{ evt: 0, frames: 4 }, { evt: 0, frames: 3 }];
    expect(boundaryDraggableAfter(spans, ids, 0)).toBe(true);
  });
  it("a note edge is NOT draggable (Rust conserves per-note totals)", () => {
    const spans = [{ evt: 0, frames: 4 }, { evt: 1, frames: 3 }];
    expect(boundaryDraggableAfter(spans, ids, 0)).toBe(false);
  });
  it("a dropped zero-width marker between two phones of one note must not hide their boundary", () => {
    const spans = [{ evt: 0, frames: 4 }, { evt: 0, frames: 0 }, { evt: 0, frames: 3 }];
    expect(boundaryDraggableAfter(spans, ids, 0)).toBe(true);
    expect(boundaryDraggableAfter(spans, ids, 1)).toBe(false); // the marker itself is no handle
  });
  it("a trailing dropped marker with no real phone after it is NOT draggable", () => {
    const spans = [{ evt: 0, frames: 4 }, { evt: 0, frames: 0 }, { evt: 1, frames: 3 }];
    expect(boundaryDraggableAfter(spans, ids, 0)).toBe(false);
  });
  it("a gap rest (no tripleNoteId) is never editable", () => {
    const spans = [{ evt: 2, frames: 4 }, { evt: 2, frames: 3 }];
    expect(boundaryDraggableAfter(spans, ids, 0)).toBe(false);
  });
});
