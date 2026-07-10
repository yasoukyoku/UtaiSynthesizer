// ② Vocal render (S48 Phase 6) — buildVocalScore alignment gate. The score triples' `frames` and the
// Option-A f0 array MUST share one 50fps grid so Σ(triple frames) == f0 length (build_note_hz maps cv↔DAW
// by cumulative frames — a length disagreement silently drifts pitch, the class the user has been burned by).
import { describe, it, expect, vi } from "vitest";

// buildVocalScore is pure, but the module also imports invoke/store (for renderVocalSegment) — mock so the
// module loads headless (mirrors store/vocalData.test.ts).
vi.mock("@tauri-apps/api/core", () => ({ invoke: () => Promise.resolve() }));
vi.mock("../../i18n", () => ({ default: { t: (k: string) => k } }));

import { buildVocalScore } from "./vocalRender";
import { DEFAULT_TRANSITION } from "../vocalNotes";
import type { Note } from "../../types/project";

const mkNote = (id: string, tick: number, duration: number, pitch: number, lyric = "あ"): Note => ({
  id, tick, duration, pitch, lyric, velocity: 100,
});

describe("buildVocalScore", () => {
  const tempo = 120;
  const def = DEFAULT_TRANSITION;

  it("aligns Σ(triple frames) == f0 length == voiced length (build_note_hz cv↔DAW invariant)", () => {
    const notes = [mkNote("a", 0, 480, 60), mkNote("b", 960, 480, 62)]; // gap 480..960
    const { triples, f0Cents, f0Voiced } = buildVocalScore(notes, undefined, tempo, def);
    const sum = triples.reduce((s, t) => s + t.frames, 0);
    expect(f0Cents.length).toBe(sum);
    expect(f0Voiced.length).toBe(sum);
    expect(f0Cents.length).toBeGreaterThan(0);
  });

  it("inserts a leading rest + explicit gap rests (§3.4 — never inferred from pitch==0)", () => {
    const notes = [mkNote("a", 480, 480, 60), mkNote("b", 1440, 480, 62)]; // starts at 480; gap 960..1440
    const { triples } = buildVocalScore(notes, undefined, tempo, def);
    expect(triples[0]!.lyric).toBe("R"); // leading rest so stem-ms 0 == segment start
    expect(triples[0]!.note_num).toBe(0);
    expect(triples.filter((t) => t.lyric === "R").length).toBe(2); // leading + inter-note gap
    expect(triples.filter((t) => t.lyric !== "R").map((t) => t.note_num)).toEqual([60, 62]);
  });

  it("abutting notes glide with NO rest between", () => {
    const notes = [mkNote("a", 0, 480, 60), mkNote("b", 480, 480, 62)]; // abut at 480
    const { triples } = buildVocalScore(notes, undefined, tempo, def);
    expect(triples.filter((t) => t.lyric === "R").length).toBe(0);
    expect(triples.map((t) => t.note_num)).toEqual([60, 62]);
  });

  it("passes RAW pitch (transpose is applied Rust-side, §9.3)", () => {
    const notes = [mkNote("a", 0, 480, 60)];
    const { triples } = buildVocalScore(notes, undefined, tempo, def);
    expect(triples.find((t) => t.lyric !== "R")!.note_num).toBe(60);
  });

  it("keeps each note's lyric (JA kana), sorts by tick", () => {
    const notes = [mkNote("b", 480, 480, 62, "き"), mkNote("a", 0, 480, 60, "か")]; // unsorted input
    const { triples } = buildVocalScore(notes, undefined, tempo, def);
    expect(triples.map((t) => t.lyric)).toEqual(["か", "き"]);
  });

  it("empty notes → empty score + empty f0", () => {
    const { triples, f0Cents } = buildVocalScore([], undefined, tempo, def);
    expect(triples.length).toBe(0);
    expect(f0Cents.length).toBe(0);
  });
});
