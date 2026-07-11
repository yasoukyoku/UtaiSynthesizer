// ② Vocal render (S48 Phase 6) — buildVocalScore alignment gate. The score triples' `frames` and the
// Option-A f0 array MUST share one 50fps grid so Σ(triple frames) == f0 length (build_note_hz maps cv↔DAW
// by cumulative frames — a length disagreement silently drifts pitch, the class the user has been burned by).
import { describe, it, expect, vi } from "vitest";

// buildVocalScore is pure, but the module also imports invoke/store (for renderVocalSegment) — mock so the
// module loads headless (mirrors store/vocalData.test.ts).
vi.mock("@tauri-apps/api/core", () => ({ invoke: () => Promise.resolve() }));
vi.mock("../../i18n", () => ({ default: { t: (k: string) => k } }));
// isVocalDirty resolves the singer via the voice-model store — mock ONE installed "V" so `entry` is found.
vi.mock("../../store/voice-models", () => ({ useVoiceModelStore: { getState: () => ({ models: { sovits: [{ name: "V", path: "p" }] } }) } }));

import { buildVocalScore, isVocalDirty, vocalRenderSig, splitSegmentVocalAware } from "./vocalRender";
import { useProjectStore } from "../../store/project";
import { DEFAULT_TRANSITION } from "../vocalNotes";
import type { Note, Track, Segment, ProcessedOutput, VocalTrackParams } from "../../types/project";

const mkNote = (id: string, tick: number, duration: number, pitch: number, lyric = "あ"): Note => ({
  id, tick, duration, pitch, lyric, velocity: 100,
});

// ── ② isVocalDirty dual-sig: a carried SPLIT-WINDOW bake is accepted (no re-render) via renderedSig OR
//    windowSig; both live on the OVERLAY so they can't desync from the bake (the audit's bakeSplit-flag
//    desync — silent wrong audio on split→edit→render→undo / tempo — is structurally impossible). ──
describe("isVocalDirty — dual-sig split-window acceptance", () => {
  const vp: VocalTrackParams = { backend: "sovits", speakerId: 49, langId: 2, transpose: 0, formant: 0, transition: DEFAULT_TRANSITION, breathToken: "AP" };
  const mkSeg = (out?: Partial<ProcessedOutput>): Segment => ({
    id: "s", startTick: 0, durationTicks: 480,
    content: { type: "notes", notes: [mkNote("a", 0, 240, 60)] },
    ...(out ? { processedOutputs: [{ laneId: "vocal", laneLabel: "V", group: "V", audioPath: "x", totalDurationMs: 500, waveformPeaks: [0.1], outputNodeId: "vocal", ...out }] } : {}),
  });
  const mkTrack = (seg: Segment): Track => ({ id: "t", name: "V", trackType: "vocal", segments: [seg], volumeDb: 0, pan: 0, muted: false, solo: false, expanded: true, laneControls: {}, voiceModel: "V", vocalParams: vp });
  const curSig = () => { const b = mkSeg({}); return vocalRenderSig(mkTrack(b), b, 120); };

  it("renderedSig matches the content → clean (fresh render / undo-of-split whole stem)", () => {
    const seg = mkSeg({ renderedSig: curSig() });
    expect(isVocalDirty(mkTrack(seg), seg, 120)).toBe(false);
  });
  it("windowSig matches (renderedSig is the PARENT's, ≠) → clean (a carried split window still matching this half)", () => {
    const seg = mkSeg({ renderedSig: "parent-full-sig", windowSig: curSig() });
    expect(isVocalDirty(mkTrack(seg), seg, 120)).toBe(false);
  });
  it("NEITHER matches → dirty (real drift: edit / tempo / param / a stale post-render overlay)", () => {
    const seg = mkSeg({ renderedSig: "stale", windowSig: "also-stale" });
    expect(isVocalDirty(mkTrack(seg), seg, 120)).toBe(true);
  });
  it("no bake → dirty (never rendered)", () => {
    const seg = mkSeg();
    expect(isVocalDirty(mkTrack(seg), seg, 120)).toBe(true);
  });

  // ── ② THE DIRTY-SPLIT GUARD (audit MAJOR): splitSegmentVocalAware windows a CLEAN bake (no re-render) but
  //    must NEVER window a DIRTY one clean — else split-then-Play plays the stale pre-edit stem forever. Two
  //    notes so BOTH halves are non-empty (an empty half reconciles to silence, a different branch). ──
  const twoNoteSeg = (renderedSig: string): Segment => ({
    id: "p", startTick: 0, durationTicks: 960,
    content: { type: "notes", notes: [mkNote("a", 0, 240, 60), mkNote("b", 480, 240, 62)] },
    processedOutputs: [{ laneId: "vocal", laneLabel: "V", group: "V", audioPath: "stem.wav", totalDurationMs: 1000, waveformPeaks: [0.1], outputNodeId: "vocal", renderedSig }],
  });

  it("split of a DIRTY bake does NOT launder it clean — both halves stay dirty (→ re-render, not stale audio)", () => {
    const seg = twoNoteSeg("STALE-sig-does-not-match-content"); // renderedSig ≠ current content → dirty
    const track = mkTrack(seg);
    useProjectStore.setState({ tracks: [track], tempo: 120 } as never);
    expect(isVocalDirty(track, seg, 120)).toBe(true); // precondition: parent is dirty before the split
    const newId = splitSegmentVocalAware("t", "p", 300, 120); // 300 in the rest between the notes → leftDur 300
    expect(newId).toBeTruthy();
    const halves = useProjectStore.getState().tracks[0]!.segments;
    expect(halves).toHaveLength(2);
    for (const h of halves) {
      expect(h.processedOutputs?.[0]?.windowSig).toBeUndefined(); // guard CLEARED it — not laundered clean
      expect(isVocalDirty(useProjectStore.getState().tracks[0]!, h, 120)).toBe(true); // → Play re-renders
    }
  });

  it("split of a CLEAN bake DOES window both halves clean (no re-render — the guard doesn't over-fire)", () => {
    const full = twoNoteSeg("placeholder");
    const cleanSig = vocalRenderSig(mkTrack(full), full, 120); // the PARENT whole-stem sig → parent is clean
    const seg = twoNoteSeg(cleanSig);
    const track = mkTrack(seg);
    useProjectStore.setState({ tracks: [track], tempo: 120 } as never);
    expect(isVocalDirty(track, seg, 120)).toBe(false); // precondition: parent is clean before the split
    const newId = splitSegmentVocalAware("t", "p", 300, 120);
    expect(newId).toBeTruthy();
    const halves = useProjectStore.getState().tracks[0]!.segments;
    expect(halves).toHaveLength(2);
    for (const h of halves) {
      expect(h.processedOutputs?.[0]?.windowSig).toBeTruthy(); // windowed to THIS half's content
      expect(isVocalDirty(useProjectStore.getState().tracks[0]!, h, 120)).toBe(false); // no re-render
    }
  });
});

describe("buildVocalScore", () => {
  const tempo = 120;
  const def = DEFAULT_TRANSITION;

  it("aligns Σ(triple frames) == f0 length == voiced length (build_note_hz cv↔DAW invariant)", () => {
    const notes = [mkNote("a", 0, 480, 60), mkNote("b", 960, 480, 62)]; // gap 480..960
    const { triples, f0Cents, f0Voiced } = buildVocalScore(notes, undefined, tempo, def, "AP");
    const sum = triples.reduce((s, t) => s + t.frames, 0);
    expect(f0Cents.length).toBe(sum);
    expect(f0Voiced.length).toBe(sum);
    expect(f0Cents.length).toBeGreaterThan(0);
  });

  it("inserts a leading rest + explicit gap rests (§3.4 — never inferred from pitch==0)", () => {
    const notes = [mkNote("a", 480, 480, 60), mkNote("b", 1440, 480, 62)]; // starts at 480; gap 960..1440
    const { triples } = buildVocalScore(notes, undefined, tempo, def, "AP");
    expect(triples[0]!.lyric).toBe("R"); // leading rest so stem-ms 0 == segment start
    expect(triples[0]!.note_num).toBe(0);
    expect(triples.filter((t) => t.lyric === "R").length).toBe(2); // leading + inter-note gap
    expect(triples.filter((t) => t.lyric !== "R").map((t) => t.note_num)).toEqual([60, 62]);
  });

  it("abutting notes glide with NO rest between", () => {
    const notes = [mkNote("a", 0, 480, 60), mkNote("b", 480, 480, 62)]; // abut at 480
    const { triples } = buildVocalScore(notes, undefined, tempo, def, "AP");
    expect(triples.filter((t) => t.lyric === "R").length).toBe(0);
    expect(triples.map((t) => t.note_num)).toEqual([60, 62]);
  });

  it("passes RAW pitch (transpose is applied Rust-side, §9.3)", () => {
    const notes = [mkNote("a", 0, 480, 60)];
    const { triples } = buildVocalScore(notes, undefined, tempo, def, "AP");
    expect(triples.find((t) => t.lyric !== "R")!.note_num).toBe(60);
  });

  it("keeps each note's lyric (JA kana), sorts by tick", () => {
    const notes = [mkNote("b", 480, 480, 62, "き"), mkNote("a", 0, 480, 60, "か")]; // unsorted input
    const { triples } = buildVocalScore(notes, undefined, tempo, def, "AP");
    expect(triples.map((t) => t.lyric)).toEqual(["か", "き"]);
  });

  it("empty notes → empty score + empty f0", () => {
    const { triples, f0Cents } = buildVocalScore([], undefined, tempo, def, "AP");
    expect(triples.length).toBe(0);
    expect(f0Cents.length).toBe(0);
  });

  // helper: the [start,end) frame span of each triple.
  const spans = (triples: { lyric: string; frames: number }[]) => {
    let c = 0;
    return triples.map((t) => { const s = c; c += t.frames; return { lyric: t.lyric, s, e: c }; });
  };

  it("breath note → AP phone + UNVOICED f0 (breaks the pitch chain, §M3)", () => {
    // か—AP—き, all abutting. The AP breath is emitted as the AP phone and its frames are UNVOICED (so the
    // か releases / the き scoops rather than gliding into/out of the breath).
    const notes = [mkNote("a", 0, 480, 60, "か"), mkNote("br", 480, 240, 62, "AP"), mkNote("c", 720, 480, 64, "き")];
    const { triples, f0Voiced } = buildVocalScore(notes, undefined, tempo, def, "AP");
    const ap = spans(triples).find((x) => x.lyric === "AP")!;
    expect(ap).toBeTruthy(); // breath kept as the AP phone (not silence, not "か")
    for (let f = ap.s; f < ap.e; f++) expect(f0Voiced[f]).toBe(0); // breath frames unvoiced
    expect(Array.from(f0Voiced).some((v) => v === 1)).toBe(true); // the sung notes are voiced
  });

  it("custom breath token is unvoiced; renaming it re-voices the OLD token (§user dynamic)", () => {
    const notes = [mkNote("a", 0, 480, 60, "呼")];
    // breathToken "呼" → the note IS a breath → AP phone, all-unvoiced.
    const asBreath = buildVocalScore(notes, undefined, tempo, def, "呼");
    expect(asBreath.triples.some((t) => t.lyric === "AP")).toBe(true);
    expect(Array.from(asBreath.f0Voiced).every((v) => v === 0)).toBe(true);
    // change the token away → "呼" is a normal lyric again → sent literally + VOICED (connected pitch).
    const asLyric = buildVocalScore(notes, undefined, tempo, def, "AP");
    expect(asLyric.triples.some((t) => t.lyric === "呼")).toBe(true);
    expect(Array.from(asLyric.f0Voiced).some((v) => v === 1)).toBe(true);
  });

  // ── ② M-defer: loudness + formant per-frame envelopes (aligned to the SAME 50fps grid as f0) ──
  it("no loudness/formant lane + 0 formant scalar → EMPTY envelopes (Rust reads flat = exact-parity no-op)", () => {
    const { loudnessEnv, formantEnv } = buildVocalScore([mkNote("a", 0, 480, 60)], undefined, tempo, def, "AP");
    expect(loudnessEnv).toEqual([]);
    expect(formantEnv).toEqual([]);
  });

  it("loudness lane → per-frame LINEAR multiplier (dB→10^(dB/20)), aligned to f0 length, rising with the curve", () => {
    const { f0Cents, loudnessEnv } = buildVocalScore([mkNote("a", 0, 480, 60)], undefined, tempo, def, "AP", { loudness: { xs: [0, 480], ys: [0, 6] } }, 0);
    expect(loudnessEnv.length).toBe(f0Cents.length);
    expect(loudnessEnv[0]).toBeCloseTo(1, 5); // frame 0 @ tick 0 → 0 dB → ×1 (exact)
    const last = loudnessEnv[loudnessEnv.length - 1]!;
    expect(last).toBeGreaterThan(1.5); // rising toward +6 dB (×1.995); the last frame is < the note end so < 1.995
    expect(last).toBeLessThan(Math.pow(10, 6 / 20) + 1e-6);
  });

  it("formant SCALAR (no lane) → all-scalar semitone array; scalar + lane fold ADDITIVELY (one summation site)", () => {
    const scalarOnly = buildVocalScore([mkNote("a", 0, 480, 60)], undefined, tempo, def, "AP", undefined, -3);
    expect(scalarOnly.formantEnv.length).toBe(scalarOnly.f0Cents.length);
    expect(scalarOnly.formantEnv.every((v) => v === -3)).toBe(true); // flat = scalar everywhere
    const withLane = buildVocalScore([mkNote("a", 0, 480, 60)], undefined, tempo, def, "AP", { formant: { xs: [0, 480], ys: [0, 4] } }, 2);
    expect(withLane.formantEnv[0]).toBeCloseTo(2, 5); // frame 0: scalar 2 + lane 0 (exact)
    const flast = withLane.formantEnv[withLane.formantEnv.length - 1]!;
    expect(flast).toBeGreaterThan(5); // scalar 2 + lane rising toward 4 (last frame < note end → < 6)
    expect(flast).toBeLessThan(6 + 1e-6);
  });
});
