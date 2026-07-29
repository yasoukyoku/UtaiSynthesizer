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

  it("the IPC payload carries the switch, and the notes' own timing", () => {
    const on = phonemeLaneRequest(base());
    expect(on.args.consonantPreroll).toBe(true);
    expect(on.args.defaultLang).toBe(2);
    expect(on.args.score.length).toBeGreaterThan(0);
    const off = phonemeLaneRequest({ ...base(), consonantPreroll: false });
    expect(off.args.consonantPreroll).toBe(false);
    // the switch is a BACKEND allocation decision — the triples themselves are identical, which is
    // why the lane cannot show it without actually re-issuing the call (hence the signature).
    expect(off.args.score).toEqual(on.args.score);
  });
});
