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

import { buildVocalScore, buildScoreTriples, isVocalDirty, vocalRenderSig, vocalTrackSig, splitSegmentVocalAware, vocalRenderOptions, vocalRenderErrorMessage, isVocalCancelError, G2P_ALGO_VERSION, SCORE_TIMING_VERSION } from "./vocalRender";
import { aliasDefaultLyric } from "./languages";
import { useProjectStore } from "../../store/project";
import { DEFAULT_TRANSITION } from "../vocalNotes";
import type { Note, Track, Segment, ProcessedOutput, VocalTrackParams } from "../../types/project";

const mkNote = (id: string, tick: number, duration: number, pitch: number, lyric = "あ"): Note => ({
  id, tick, duration, pitch, lyric, velocity: 100,
});

/** S88: the two lyric triggers travel as ONE named struct (never two bare strings side by side — a swap
 *  there would be invisible to the compiler and to a test that passes the same value for both).
 *  `TK()` = the canonical defaults, `TK("呼")` = a custom breath, `TK(undefined, "休")` = a custom rest. */
const TK = (breath = "AP", rest = "R") => ({ breath, rest });

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
    const { triples, f0Cents, f0Voiced } = buildVocalScore(notes, undefined, tempo, def, TK());
    const sum = triples.reduce((s, t) => s + t.frames, 0);
    expect(f0Cents.length).toBe(sum);
    expect(f0Voiced.length).toBe(sum);
    expect(f0Cents.length).toBeGreaterThan(0);
  });

  it("inserts a leading rest + explicit gap rests (§3.4 — never inferred from pitch==0)", () => {
    const notes = [mkNote("a", 480, 480, 60), mkNote("b", 1440, 480, 62)]; // starts at 480; gap 960..1440
    const { triples } = buildVocalScore(notes, undefined, tempo, def, TK());
    expect(triples[0]!.lyric).toBe("R"); // leading rest so stem-ms 0 == segment start
    expect(triples[0]!.note_num).toBe(0);
    expect(triples.filter((t) => t.lyric === "R").length).toBe(2); // leading + inter-note gap
    expect(triples.filter((t) => t.lyric !== "R").map((t) => t.note_num)).toEqual([60, 62]);
  });

  it("abutting notes glide with NO rest between", () => {
    const notes = [mkNote("a", 0, 480, 60), mkNote("b", 480, 480, 62)]; // abut at 480
    const { triples } = buildVocalScore(notes, undefined, tempo, def, TK());
    expect(triples.filter((t) => t.lyric === "R").length).toBe(0);
    expect(triples.map((t) => t.note_num)).toEqual([60, 62]);
  });

  it("passes RAW pitch (transpose is applied Rust-side, §9.3)", () => {
    const notes = [mkNote("a", 0, 480, 60)];
    const { triples } = buildVocalScore(notes, undefined, tempo, def, TK());
    expect(triples.find((t) => t.lyric !== "R")!.note_num).toBe(60);
  });

  it("keeps each note's lyric (JA kana), sorts by tick", () => {
    const notes = [mkNote("b", 480, 480, 62, "き"), mkNote("a", 0, 480, 60, "か")]; // unsorted input
    const { triples } = buildVocalScore(notes, undefined, tempo, def, TK());
    expect(triples.map((t) => t.lyric)).toEqual(["か", "き"]);
  });

  it("empty notes → empty score + empty f0", () => {
    const { triples, f0Cents } = buildVocalScore([], undefined, tempo, def, TK());
    expect(triples.length).toBe(0);
    expect(f0Cents.length).toBe(0);
  });

  // ── S58 per-note language + phoneme override wire ──
  it("triples carry the per-note EFFECTIVE lang (override ?? track default) + phoneme_input", () => {
    const notes = [
      mkNote("a", 480, 480, 60, "长"), // leading rest before it takes the DEFAULT lang
      { ...mkNote("b", 960, 480, 62, "light"), lang: "en", phonemeInput: "L AY1 T" },
    ];
    const { triples } = buildVocalScore(notes, undefined, tempo, def, TK(), undefined, 0, 0 /* zh default */);
    expect(triples[0]!.lyric).toBe("R");
    expect(triples[0]!.lang).toBe(0); // rest → track default (Rust re-attaches it to the run)
    const sung = triples.filter((t) => t.lyric !== "R");
    expect(sung[0]).toMatchObject({ lyric: "长", lang: 0 }); // no override → track default (zh)
    expect(sung[0]!.phoneme_input).toBeUndefined();
    expect(sung[1]).toMatchObject({ lyric: "light", lang: 1, phoneme_input: "L AY1 T" }); // en override
  });

  it("an INVALID note lang code falls back to the track default (whitelist mirror)", () => {
    const notes = [{ ...mkNote("a", 0, 480, 60), lang: "xx" }];
    const { triples } = buildVocalScore(notes, undefined, tempo, def, TK(), undefined, 0, 5 /* es */);
    expect(triples.find((t) => t.lyric !== "R")!.lang).toBe(5);
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
    const { triples, f0Voiced } = buildVocalScore(notes, undefined, tempo, def, TK());
    const ap = spans(triples).find((x) => x.lyric === "AP")!;
    expect(ap).toBeTruthy(); // breath kept as the AP phone (not silence, not "か")
    for (let f = ap.s; f < ap.e; f++) expect(f0Voiced[f]).toBe(0); // breath frames unvoiced
    expect(Array.from(f0Voiced).some((v) => v === 1)).toBe(true); // the sung notes are voiced
  });

  it("custom breath token is unvoiced; renaming it re-voices the OLD token (§user dynamic)", () => {
    const notes = [mkNote("a", 0, 480, 60, "呼")];
    // breathToken "呼" → the note IS a breath → AP phone, all-unvoiced.
    const asBreath = buildVocalScore(notes, undefined, tempo, def, TK("呼"));
    expect(asBreath.triples.some((t) => t.lyric === "AP")).toBe(true);
    expect(Array.from(asBreath.f0Voiced).every((v) => v === 0)).toBe(true);
    // change the token away → "呼" is a normal lyric again → sent literally + VOICED (connected pitch).
    const asLyric = buildVocalScore(notes, undefined, tempo, def, TK());
    expect(asLyric.triples.some((t) => t.lyric === "呼")).toBe(true);
    expect(Array.from(asLyric.f0Voiced).some((v) => v === 1)).toBe(true);
  });

  // ── ② M-defer: loudness + formant per-frame envelopes (aligned to the SAME 50fps grid as f0) ──
  it("no loudness/formant lane + 0 formant scalar → EMPTY envelopes (Rust reads flat = exact-parity no-op)", () => {
    const { loudnessEnv, formantEnv } = buildVocalScore([mkNote("a", 0, 480, 60)], undefined, tempo, def, TK());
    expect(loudnessEnv).toEqual([]);
    expect(formantEnv).toEqual([]);
  });

  it("loudness lane → per-frame LINEAR multiplier (dB→10^(dB/20)), aligned to f0 length, rising with the curve", () => {
    const { f0Cents, loudnessEnv } = buildVocalScore([mkNote("a", 0, 480, 60)], undefined, tempo, def, TK(), { loudness: { xs: [0, 480], ys: [0, 6] } }, 0);
    expect(loudnessEnv.length).toBe(f0Cents.length);
    expect(loudnessEnv[0]).toBeCloseTo(1, 5); // frame 0 @ tick 0 → 0 dB → ×1 (exact)
    const last = loudnessEnv[loudnessEnv.length - 1]!;
    expect(last).toBeGreaterThan(1.5); // rising toward +6 dB (×1.995); the last frame is < the note end so < 1.995
    expect(last).toBeLessThan(Math.pow(10, 6 / 20) + 1e-6);
  });

  it("formant SCALAR (no lane) → all-scalar semitone array; scalar + lane fold ADDITIVELY (one summation site)", () => {
    const scalarOnly = buildVocalScore([mkNote("a", 0, 480, 60)], undefined, tempo, def, TK(), undefined, -3);
    expect(scalarOnly.formantEnv.length).toBe(scalarOnly.f0Cents.length);
    expect(scalarOnly.formantEnv.every((v) => v === -3)).toBe(true); // flat = scalar everywhere
    const withLane = buildVocalScore([mkNote("a", 0, 480, 60)], undefined, tempo, def, TK(), { formant: { xs: [0, 480], ys: [0, 4] } }, 2);
    expect(withLane.formantEnv[0]).toBeCloseTo(2, 5); // frame 0: scalar 2 + lane 0 (exact)
    const flast = withLane.formantEnv[withLane.formantEnv.length - 1]!;
    expect(flast).toBeGreaterThan(5); // scalar 2 + lane rising toward 4 (last frame < note end → < 6)
    expect(flast).toBeLessThan(6 + 1e-6);
  });
});

// ── S84 D 刀: a sung note whose span rounds to ZERO 50fps frames must surface in droppedNoteIds
//    (pre-S84 it vanished silently — the 30t く/ず audit case); the timeline stays conserved
//    (forcing a minimum frame would break Σframes == frameOf(cursor)). ──
describe("buildScoreTriples — zero-frame note drop is loud", () => {
  it("a note that CANNOT borrow is still dropped loudly, and the frame timeline stays conserved", () => {
    // tempo 120 → ticksPerFrame 19.2. `b` spans ticks 20..25 → frames 1..1 = ZERO, and it is boxed in by
    // two 1-frame notes: neither can pay (a sung lender must keep ≥2), so it must still be reported.
    const notes = [mkNote("a", 0, 20, 60), mkNote("b", 20, 5, 62), mkNote("c", 25, 20, 64)];
    const { triples, tripleNoteIds, droppedNoteIds, borrowedNoteIds, frameCount } = buildScoreTriples(notes, 120, TK(), 2);
    expect(droppedNoteIds).toEqual(["b"]);
    expect(borrowedNoteIds).toEqual([]);
    for (const id of droppedNoteIds) expect(tripleNoteIds).not.toContain(id);
    expect(triples.reduce((s, t) => s + t.frames, 0)).toBe(frameCount);
  });
});

// ── S87 #3 借帧刀 — a note whose span rounds to ZERO 50fps frames borrows ONE frame from a neighbour
//    instead of vanishing. THE invariant: a borrow moves a SHARED frame boundary (the lender's end and the
//    borrower's start are the same number), so Σframes == frameOf(cursor) is preserved BY CONSTRUCTION —
//    "force a minimum frame" would break it and is why the pre-S87 code just dropped the note. ──
describe("buildScoreTriples — S87 frame borrowing", () => {
  const sumFrames = (t: { frames: number }[]) => t.reduce((s, x) => s + x.frames, 0);

  it("★ CONSERVATION holds in every case (the one rule the knife may never break)", () => {
    const cases: Note[][] = [
      [mkNote("n1", 0, 480, 60)],
      [mkNote("a", 0, 480, 60), mkNote("b", 480, 30, 62), mkNote("c", 510, 480, 64)], // sub-frame in the middle
      [mkNote("a", 0, 3, 60), mkNote("b", 3, 480, 62)], // sub-frame at the very START (no previous item)
      [mkNote("a", 0, 480, 60), mkNote("b", 480, 3, 62)], // sub-frame at the very END (no next item)
      [mkNote("a", 0, 30, 60), mkNote("b", 30, 30, 62), mkNote("c", 60, 30, 64)], // a run of short notes
      [mkNote("a", 0, 480, 60), mkNote("b", 900, 3, 62), mkNote("c", 1400, 480, 64)], // gaps around it
      // S88 — rest NOTES belong in this sweep too: every rest-item branch (the lender floor, the PASS-2
      // skip, rescueIsSafe) keys on frameOf(end)-frameOf(start), i.e. on the frame PHASE, which is exactly
      // the class S87 measured flipping on 27.7% of tempi from a single fixture. A review found the
      // zero-frame-rest hole below precisely because these cases were missing here.
      [mkNote("a", 0, 480, 60), mkNote("r", 480, 480, 71, "R"), mkNote("c", 960, 480, 64)], // rest as a note
      [mkNote("a", 0, 96, 60), mkNote("b", 96, 1, 62), mkNote("r", 97, 3, 60, "R"), mkNote("c", 100, 480, 64)], // zero-frame rest AFTER a zero-frame note
      [mkNote("b", 0, 1, 72), mkNote("r", 1, 39, 60, "R")], // the only lender is a rest note at its floor
      [mkNote("r", 0, 3, 60, "R"), mkNote("a", 3, 480, 60)], // a sub-frame rest at the very start
      [mkNote("a", 0, 480, 60), mkNote("r", 480, 3, 60, "R")], // …and at the very end
      [mkNote("r1", 0, 240, 60, "R"), mkNote("r2", 240, 240, 62, "R")], // nothing but rests
    ];
    let borrows = 0;
    let drops = 0;
    for (const notes of cases) {
      for (const tempo of [60, 120, 222, 300]) {
        const r = buildScoreTriples(notes, tempo, TK(), 2);
        expect(sumFrames(r.triples)).toBe(r.frameCount);
        // and every emitted triple must actually occupy time
        for (const t of r.triples) expect(t.frames).toBeGreaterThan(0);
        borrows += r.borrowedNoteIds.length;
        drops += r.droppedNoteIds.length;
      }
    }
    // …and the sweep must actually EXERCISE both outcomes, or "conservation holds" would just be a
    // statement about a knife that never fired (a review caught this loop being 20/24 no-ops).
    expect(borrows).toBeGreaterThan(0);
    expect(drops).toBeGreaterThan(0);
  });

  it("★ REFUSES to rescue when a SUNG note follows — it would starve that note's onset consonant", () => {
    // Rust pre-rolls the next note's onset out of the phone before it; a 1-frame lender yields nothing, so
    // the follower loses its consonant (「た」 sings as 「あ」) or its vowel lands ~40 ms late. Never worse
    // than baseline: the sub-frame note goes back to being reported as too short.
    const notes = [mkNote("a", 0, 480, 60), mkNote("b", 480, 30, 62), mkNote("c", 510, 480, 64)];
    const r = buildScoreTriples(notes, 222, TK(), 2);
    expect(r.borrowedNoteIds).toEqual([]);
    expect(r.droppedNoteIds).toEqual(["b"]);
    expect(r.shortNoteIds).toEqual(["b"]); // …but it is STILL reported as short
    // the follower keeps every frame it had without b — nothing was taken from anyone
    const base = buildScoreTriples([mkNote("a", 0, 480, 60), mkNote("c", 510, 480, 64)], 222, TK(), 2);
    expect(r.triples[r.tripleNoteIds.indexOf("a")]!.frames).toBe(base.triples[base.tripleNoteIds.indexOf("a")]!.frames);
    expect(r.triples[r.tripleNoteIds.indexOf("c")]!.frames).toBe(base.triples[base.tripleNoteIds.indexOf("c")]!.frames);
    expect(sumFrames(r.triples)).toBe(r.frameCount);
  });

  it("borrows ONE frame BACKWARD from the previous note — downstream boundaries do not move", () => {
    // tempo 222 (ticksPerFrame 35.52): b spans ticks 480..510 → frames 14..14 = zero. A REST follows it, so
    // the rescue is safe (an SP lender serves the next note's onset just fine).
    const notes = [mkNote("a", 0, 480, 60), mkNote("b", 480, 30, 62), mkNote("c", 700, 480, 64)];
    const base = buildScoreTriples([mkNote("a", 0, 480, 60), mkNote("c", 700, 480, 64)], 222, TK(), 2);
    const r = buildScoreTriples(notes, 222, TK(), 2);
    expect(r.borrowedNoteIds).toEqual(["b"]);
    expect(r.droppedNoteIds).toEqual([]);
    const tb = r.triples[r.tripleNoteIds.indexOf("b")]!;
    expect(tb.frames).toBe(1); // it sounds now
    // the lender paid exactly one frame…
    const aNew = r.triples[r.tripleNoteIds.indexOf("a")]!.frames;
    const aOld = base.triples[base.tripleNoteIds.indexOf("a")]!.frames;
    expect(aNew).toBe(aOld - 1);
    // …and NOTHING downstream shifted: c keeps its own frame count, and the total is unchanged
    expect(r.triples[r.tripleNoteIds.indexOf("c")]!.frames).toBe(base.triples[base.tripleNoteIds.indexOf("c")]!.frames);
    expect(sumFrames(r.triples)).toBe(r.frameCount);
    expect(r.frameCount).toBe(base.frameCount);
  });

  it("falls FORWARD when the previous item is too short to lend — and it is the FOLLOWER that pays", () => {
    // tempo 222 (ticksPerFrame 35.52): items are R[0,1] (1 frame), a[1,1] (ZERO), R[1,14] (13), long[14,27].
    // The LEADING rest has only 1 frame, so it may not pay (a rest must keep ≥1) → the borrow goes forward.
    const notes = [mkNote("a", 20, 30, 66), mkNote("long", 480, 480, 60)];
    const r = buildScoreTriples(notes, 222, TK(), 2);
    expect(r.droppedNoteIds).toEqual([]);
    expect(r.borrowedNoteIds).toEqual(["a"]);
    // ★ pin WHO paid and WHERE the frame landed — without these the same assertions pass for an illegal
    // backward borrow that deletes the leading rest (a review caught exactly that hole).
    expect(r.triples[0]).toMatchObject({ lyric: "R", frames: 1 }); // the leading rest is UNTOUCHED
    expect(r.tripleNoteIds.indexOf("a")).toBe(1); // …so a sits at frame index 1, not 0
    expect(r.triples[1]!.frames).toBe(1);
    expect(r.triples[2]!.frames).toBe(12); // the FOLLOWING rest paid: 13 → 12
    expect(r.triples[3]!.frames).toBe(13); // and the sung note after it is untouched
    expect(sumFrames(r.triples)).toBe(r.frameCount);
  });

  it("★ the rest-lender floor is exactly 2 frames — 1 refuses, 2 pays", () => {
    // tempo 120 (ticksPerFrame 19.2). PAYS: a 2-frame gap (frames 5..7) may drop to 1.
    const pays = [mkNote("a", 0, 96, 60), mkNote("b", 134, 1, 72)];
    const r1 = buildScoreTriples(pays, 120, TK(), 2);
    expect(r1.borrowedNoteIds).toEqual(["b"]);
    expect(r1.triples.filter((t) => t.lyric === "R").map((t) => t.frames)).toEqual([1]);
    expect(sumFrames(r1.triples)).toBe(r1.frameCount);
    // REFUSES: a 1-frame gap (frames 5..6) is already at the floor, and there is no next item → dropped.
    const refuses = [mkNote("a", 0, 96, 60), mkNote("b", 115, 1, 72)];
    const r2 = buildScoreTriples(refuses, 120, TK(), 2);
    expect(r2.droppedNoteIds).toEqual(["b"]);
    expect(r2.borrowedNoteIds).toEqual([]);
    expect(r2.triples.filter((t) => t.lyric === "R").map((t) => t.frames)).toEqual([1]); // rest kept its frame
    expect(sumFrames(r2.triples)).toBe(r2.frameCount);
  });

  it("borrows from a REST when one sits before it (silence is the cheapest lender)", () => {
    // tempo 120 → ticksPerFrame 19.2. n2 spans 600..603 → frames 31..31 = zero; the gap rest before it is
    // 6 frames (25..31), so it can pay and keep its own ≥1.
    const notes = [mkNote("n1", 0, 480, 60), mkNote("n2", 600, 3, 62)];
    const r = buildScoreTriples(notes, 120, TK(), 2);
    expect(r.borrowedNoteIds).toEqual(["n2"]);
    expect(r.droppedNoteIds).toEqual([]);
    const restFrames = r.triples.filter((t) => t.lyric === "R").map((t) => t.frames);
    expect(restFrames).toEqual([5]); // 6 → 5: the rest paid exactly one frame
    expect(r.triples[r.tripleNoteIds.indexOf("n1")]!.frames).toBe(25); // the sung note before it is UNTOUCHED
    expect(r.triples[r.tripleNoteIds.indexOf("n2")]!.frames).toBe(1);
    expect(sumFrames(r.triples)).toBe(r.frameCount);
  });

  it("★ the sung-lender floor is exactly 3 frames — 2 refuses, 3 pays", () => {
    // tempo 120 (ticksPerFrame 19.2). REFUSE: both neighbours are exactly 2 frames (0..38 and 41..79).
    const boxed = [mkNote("a", 0, 38, 60), mkNote("b", 38, 3, 62), mkNote("c", 41, 38, 64)];
    const r1 = buildScoreTriples(boxed, 120, TK(), 2);
    expect(r1.droppedNoteIds).toEqual(["b"]);
    expect(r1.borrowedNoteIds).toEqual([]);
    expect(r1.triples[r1.tripleNoteIds.indexOf("a")]!.frames).toBe(2); // lender untouched, still ≥ its floor
    expect(sumFrames(r1.triples)).toBe(r1.frameCount);
    // PAY: the previous note is 3 frames (0..57) — it may drop to 2, which is exactly the floor.
    const ok = [mkNote("a", 0, 57, 60), mkNote("b", 57, 3, 62)];
    const r2 = buildScoreTriples(ok, 120, TK(), 2);
    expect(r2.borrowedNoteIds).toEqual(["b"]);
    expect(r2.triples[r2.tripleNoteIds.indexOf("a")]!.frames).toBe(2);
    expect(r2.triples[r2.tripleNoteIds.indexOf("b")]!.frames).toBe(1);
    expect(sumFrames(r2.triples)).toBe(r2.frameCount);
  });

  it("the very FIRST item can borrow (there is no previous item at all → forward, into the rest)", () => {
    const notes = [mkNote("head", 0, 3, 60), mkNote("body", 200, 480, 62)];
    const r = buildScoreTriples(notes, 120, TK(), 2);
    expect(r.borrowedNoteIds).toEqual(["head"]);
    expect(r.droppedNoteIds).toEqual([]);
    expect(r.triples[r.tripleNoteIds.indexOf("head")]!.frames).toBe(1);
    expect(r.triples.filter((t) => t.lyric === "R").map((t) => t.frames)).toEqual([9]); // the rest paid: 10 → 9
    expect(r.triples[r.tripleNoteIds.indexOf("body")]!.frames).toBe(25); // untouched (tempo 120: 480t = 25 frames)
    expect(sumFrames(r.triples)).toBe(r.frameCount);
  });

  // ★★ THE test the first cut of this knife got WRONG. The f0 line is sampled by TICK — frame `f` at
  // exactly `f * ticksPerFrame` — so the rescued note's retimed span must CONTAIN that tick. The first
  // version rounded the frame boundary to a whole tick (`Math.round(f * tpf)`), which lands AFTER the
  // sample point whenever the fraction is ≥ 0.5: the borrower then does not own its own frame, findNoteAt
  // picks a neighbour (or nothing), and the "rescued" note comes out at the wrong pitch or UNVOICED —
  // i.e. still silent, while the UI reports it as rescued. That defect was invisible to a single-tempo
  // test (an adversarial review found it), so this one SWEEPS tempi: it was 100% broken at 300 bpm and
  // 0% broken at 100 and 125.
  describe("★ a rescued note actually OWNS its borrowed frame in the f0 feed (swept over tempo)", () => {
    const TEMPI = [60, 90, 100, 120, 125, 140, 160, 180, 200, 222, 240, 250, 280, 300];
    const frameIndexOf = (triples: { frames: number }[], upTo: number) =>
      triples.slice(0, upTo).reduce((s, t) => s + t.frames, 0);

    // A 1-tick note is NOT reliably sub-frame (at some tempi a frame boundary falls inside it), so the
    // pathological note is placed per tempo at the CENTRE of frame 40's cell, where both of its edges
    // round to the same frame for every tempo in the sweep.
    const tpf = (tempo: number) => (1000 / 50 / 60000) * tempo * 480;
    const build = (kind: string, tempo: number): Note[] => {
      const b0 = Math.round(40 * tpf(tempo));
      return kind === "rest lender"
        ? // lender = the gap REST before it (a miss here means UNVOICED — dead silence)
          [mkNote("a", 0, 120, 60), mkNote("b", b0, 1, 72), mkNote("c", b0 + 200, 480, 64)]
        : // lender = the sung note before it (a miss here means the LENDER's pitch)
          [mkNote("a", 0, b0, 60), mkNote("b", b0, 1, 72), mkNote("c", b0 + 200, 480, 64)];
    };

    for (const kind of ["rest lender", "note lender"]) {
      it(kind + ": voiced, at its OWN written pitch, at every tempo", () => {
        for (const tempo of TEMPI) {
          const notes = build(kind, tempo);
          const r = buildScoreTriples(notes, tempo, TK(), 2);
          expect(r.borrowedNoteIds, `tempo ${tempo}`).toEqual(["b"]);
          const idx = r.tripleNoteIds.indexOf("b");
          expect(r.triples[idx]!.frames, `tempo ${tempo}`).toBe(1);
          const sv = buildVocalScore(notes, undefined, tempo, DEFAULT_TRANSITION, TK());
          expect(sv.f0Cents.length).toBe(sumFrames(sv.triples));
          const f = frameIndexOf(r.triples, idx);
          expect(sv.f0Voiced[f], `tempo ${tempo}: the rescued frame must be VOICED`).toBe(1);
          // A grace note gets ZERO transition (no room for a portamento, and its single sample sits on
          // its own onset where the §10.5 open-edge scoop bottoms out) ⇒ its exact written pitch.
          expect(sv.f0Cents[f]! / 100, `tempo ${tempo}: pitch of the rescued frame`).toBeCloseTo(72, 3);
        }
      });
    }
  });

  it("★ SWEEP: over 600 pseudo-random scores, EVERY borrowed frame is voiced and at its own pitch", () => {
    // The adversarial review measured the first cut at 27.7% of borrows landing on an UNVOICED frame and
    // 35% carrying a neighbour's pitch — a rate no single-fixture test could have shown. Deterministic LCG
    // (no Math.random: a flaky gate is worse than no gate).
    let seed = 20260729;
    const rnd = (n: number) => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) % n);
    let borrows = 0;
    let unvoiced = 0;
    let wrongPitch = 0;
    for (let i = 0; i < 600; i++) {
      const tempo = 60 + rnd(241); // 60..300
      const gap = rnd(2) === 0; // rest lender vs note lender
      const head = 200 + rnd(400);
      const b0 = head + (gap ? 100 + rnd(400) : 0);
      const notes = [
        mkNote("a", 0, head, 60),
        mkNote("b", b0, 1 + rnd(3), 72),
        mkNote("c", b0 + 200 + rnd(200), 480, 64),
      ];
      const r = buildScoreTriples(notes, tempo, TK(), 2);
      expect(sumFrames(r.triples), `case ${i} tempo ${tempo}`).toBe(r.frameCount); // conservation, always
      if (!r.borrowedNoteIds.includes("b")) continue;
      borrows++;
      const idx = r.tripleNoteIds.indexOf("b");
      const f = r.triples.slice(0, idx).reduce((s, t) => s + t.frames, 0);
      const sv = buildVocalScore(notes, undefined, tempo, DEFAULT_TRANSITION, TK());
      if (sv.f0Voiced[f] !== 1) unvoiced++;
      else if (Math.abs(sv.f0Cents[f]! / 100 - 72) > 0.01) wrongPitch++;
    }
    expect(borrows).toBeGreaterThan(50); // the sweep must actually exercise the knife
    expect({ unvoiced, wrongPitch }).toEqual({ unvoiced: 0, wrongPitch: 0 });
  });

  // ── S88 rest token. The claim being tested is ONE sentence: a note carrying the rest trigger is the
  //    same silence as leaving a gap. Everything below is that sentence at a different layer. ──
  describe("the rest token — a written rest IS a gap", () => {
    /** [start,end) frame span of each triple (local copy — the buildVocalScore block's one is out of scope). */
    const spans = (triples: { lyric: string; frames: number }[]) => {
      let c = 0;
      return triples.map((t) => { const s = c; c += t.frames; return { lyric: t.lyric, s, e: c }; });
    };

    it("★ EQUIVALENCE: the same music written as a rest NOTE and as a GAP produces identical output", () => {
      const asGap = [mkNote("a", 0, 480, 60, "か"), mkNote("c", 960, 480, 64, "き")];
      // the rest note fills the gap exactly, and is drawn at pitch 71 — a pitch nothing may ever sing
      const asNote = [mkNote("a", 0, 480, 60, "か"), mkNote("r", 480, 480, 71, "R"), mkNote("c", 960, 480, 64, "き")];
      const g = buildVocalScore(asGap, undefined, 120, DEFAULT_TRANSITION, TK());
      const n = buildVocalScore(asNote, undefined, 120, DEFAULT_TRANSITION, TK());
      expect(n.triples).toEqual(g.triples); // payload: same lyric, same note_num 0, same frames, same lang
      expect(n.f0Cents).toEqual(g.f0Cents);
      expect(n.f0Voiced).toEqual(g.f0Voiced);
      // …and specifically: the drawn pitch of the rest is nowhere in the pitch line. Before S88 the line
      // GLIDED into and out of it (the note stayed in the chain), so か released toward 71 and き scooped
      // out of it — audibly wrong, and impossible to see from the score.
      expect(Array.from(n.f0Cents).some((c) => Math.abs(c / 100 - 71) < 0.5)).toBe(false);
      expect(Array.from(n.f0Voiced).some((v) => v === 1)).toBe(true); // か / き still sing
    });

    it("a CUSTOM trigger is that same silence; pointing it elsewhere makes the word sing again", () => {
      const notes = [mkNote("a", 0, 480, 60, "か"), mkNote("x", 480, 480, 71, "休"), mkNote("c", 960, 480, 64, "き")];
      const asRest = buildVocalScore(notes, undefined, 120, DEFAULT_TRANSITION, TK(undefined, "休"));
      expect(asRest.triples.map((t) => t.lyric)).toEqual(["か", "R", "き"]); // mapped to the canonical token
      expect(asRest.triples[1]!.note_num).toBe(0);
      const s = spans(asRest.triples).find((x) => x.lyric === "R")!;
      for (let f = s.s; f < s.e; f++) expect(asRest.f0Voiced[f]).toBe(0);
      // …and with the trigger pointed at the default, 休 is an ordinary lyric: sent literally, voiced.
      const asLyric = buildVocalScore(notes, undefined, 120, DEFAULT_TRANSITION, TK());
      expect(asLyric.triples.map((t) => t.lyric)).toEqual(["か", "休", "き"]);
      expect(asLyric.triples[1]!.note_num).toBe(71);
      const s2 = spans(asLyric.triples).find((x) => x.lyric === "休")!;
      expect(asLyric.f0Voiced[s2.s]).toBe(1);
    });

    it("a sub-frame rest is NOT warned about and NOT rescued (not sounding is what it asked for)", () => {
      // tempo 120 → ticksPerFrame 19.2; the rest spans 480..483, which rounds to zero frames.
      const notes = [mkNote("a", 0, 480, 60, "か"), mkNote("r", 480, 3, 60, "R"), mkNote("c", 483, 480, 64, "き")];
      const r = buildScoreTriples(notes, 120, TK(), 2);
      expect(r.shortNoteIds).toEqual([]); // ← a rest in this list is a false alarm ("will not sound": good)
      expect(r.droppedNoteIds).toEqual([]);
      expect(r.borrowedNoteIds).toEqual([]);
      expect(r.triples.reduce((s, t) => s + t.frames, 0)).toBe(r.frameCount);
    });

    it("★ a rest note LENDS at the rest floor (1), not the sung floor (2)", () => {
      // b spans 0..1 → zero frames, and there is no previous item, so the only possible lender is the
      // rest note after it: 2 frames, which may drop to 1. Written as a NOTE it used to be held to the
      // sung floor of 2 and refused — the same music written as a gap paid without complaint.
      const notes = [mkNote("b", 0, 1, 72, "か"), mkNote("r", 1, 39, 60, "R")];
      const r = buildScoreTriples(notes, 120, TK(), 2);
      expect(r.borrowedNoteIds).toEqual(["b"]);
      expect(r.droppedNoteIds).toEqual([]);
      expect(r.triples.map((t) => [t.lyric, t.frames])).toEqual([["か", 1], ["R", 1]]);
      expect(r.triples.reduce((s, t) => s + t.frames, 0)).toBe(r.frameCount);
    });

    it("★ a rest note does not BLOCK a rescue — nothing pre-rolls a consonant out of silence", () => {
      // The knife only fires when the next item needs nothing from the borrower (S87). A rest note is a
      // rest: it has no onset consonant to starve, so it must not veto the rescue the way a sung note does.
      const notes = [mkNote("a", 0, 96, 60, "か"), mkNote("b", 96, 3, 72, "き"), mkNote("r", 99, 401, 60, "R")];
      const r = buildScoreTriples(notes, 120, TK(), 2);
      expect(r.borrowedNoteIds).toEqual(["b"]);
      expect(r.droppedNoteIds).toEqual([]);
      expect(r.triples.reduce((s, t) => s + t.frames, 0)).toBe(r.frameCount);
      // the same shape with a SUNG note in that slot must still refuse (the S87 rule is untouched)
      const sung = [mkNote("a", 0, 96, 60, "か"), mkNote("b", 96, 3, 72, "き"), mkNote("c", 99, 401, 64, "く")];
      expect(buildScoreTriples(sung, 120, TK(), 2).droppedNoteIds).toEqual(["b"]);
    });

    it("★ a ZERO-FRAME rest note may NOT be treated as an available lender (S88 review, real regression)", () => {
      // tempo 120 → ticksPerFrame 19.2. き(480..484) AND the rest(484..488) both round to frame 25, i.e.
      // BOTH are zero-frame. The rest emits no triple and can never acquire one, so the phone actually
      // following a rescued き would be く — which then pre-rolls its onset consonant out of a 1-frame
      // nucleus (avail = 0) and loses it. That is the S87 pathology, so the rescue must refuse and き must
      // go back to being reported, exactly as it was before this feature existed.
      const notes = [mkNote("a", 0, 480, 60, "か"), mkNote("b", 480, 4, 62, "き"), mkNote("r", 484, 4, 60, "R"), mkNote("c", 488, 480, 64, "く")];
      const r = buildScoreTriples(notes, 120, TK(), 2);
      expect(r.borrowedNoteIds).toEqual([]);
      expect(r.droppedNoteIds).toEqual(["b"]); // reported, not silently rescued into damage
      // the wire is untouched: か keeps all 25 frames, so く's onset still pre-rolls out of a real vowel
      expect(r.triples.map((t) => [t.lyric, t.frames])).toEqual([["か", 25], ["く", 25]]);
      expect(r.triples.reduce((s, t) => s + t.frames, 0)).toBe(r.frameCount);
    });

    it("★ the breath/rest tie-break is decided in the TRIPLE, not just in the boolean", () => {
      // isSilentLyric is an OR, so asserting it alone cannot tell the two orderings apart (a review
      // caught that). mapLyric is where the order is observable: with both triggers on the same word the
      // note must come out as the REST token — silence, note_num 0 — not as an AP breath phone.
      const notes = [mkNote("x", 0, 480, 71, "同")];
      const r = buildScoreTriples(notes, 120, { breath: "同", rest: "同" }, 2);
      expect(r.triples.map((t) => [t.lyric, t.note_num])).toEqual([["R", 0]]);
    });

    it("an ordinary lyric is untouched by the rest machinery (no false silencing)", () => {
      const notes = [mkNote("a", 0, 480, 60, "ら"), mkNote("b", 480, 480, 62, "rest")];
      const r = buildVocalScore(notes, undefined, 120, DEFAULT_TRANSITION, TK());
      expect(r.triples.map((t) => t.lyric)).toEqual(["ら", "rest"]); // S86 freed `rest` — it still sings
      expect(Array.from(r.f0Voiced).every((v) => v === 1)).toBe(true);
    });
  });

  it("leaves a score with NO sub-frame note byte-identical (no false borrowing)", () => {
    const notes = [mkNote("a", 0, 480, 60), mkNote("b", 480, 480, 62), mkNote("c", 1200, 480, 64)];
    const r = buildScoreTriples(notes, 120, TK(), 2);
    expect(r.borrowedNoteIds).toEqual([]);
    expect(r.droppedNoteIds).toEqual([]);
    // tempo 120 ⇒ ticksPerFrame 19.2: a 480t note is 25 frames, the 240t gap is 13 (63−50, both rounded).
    expect(r.triples.map((t) => [t.lyric, t.note_num, t.frames])).toEqual([
      ["あ", 60, 25], ["あ", 62, 25], ["R", 0, 13], ["あ", 64, 25],
    ]);
    expect(sumFrames(r.triples)).toBe(r.frameCount);
  });
});

// S90 — the ALGORITHM-VERSION terms of the bake signature, pinned as a LITERAL. A sig test that compares
// two sigs to each other passes even when both sides lose the same term (S88 lesson: "sig 自比不是测试"),
// and these two tokens are the whole mechanism by which a lyric→phone or note→frame fix reaches an
// already-baked segment. If a round changes what a lyric sings and this literal did not move, the user
// hears the OLD audio forever and reads it as "the fix did nothing" (S81/S86 R3).
describe("vocalTrackSig — the version terms are present and literal", () => {
  const vp: VocalTrackParams = { backend: "sovits", speakerId: 49, langId: 2, transpose: 0, formant: 0, transition: DEFAULT_TRANSITION, breathToken: "AP" };
  const track: Track = { id: "t", name: "V", trackType: "vocal", segments: [], volumeDb: 0, pan: 0, muted: false, solo: false, expanded: true, laneControls: {}, voiceModel: "V", vocalParams: vp };

  // WIRING: the two tokens are actually IN the signature, with their current values. Robust against
  // unrelated additions to vocalParamsSig — it is not this test's job to notice those.
  it("carries the g2p + timing tokens", () => {
    expect(vocalTrackSig(track, 120)).toContain("|g2p:s94|st:s96e");
    expect(G2P_ALGO_VERSION).toBe("s94"); // S94 dictionary re-audit batch: EN onset vote gate + en.tsv regeneration knives
    expect(SCORE_TIMING_VERSION).toBe("s96e"); // S96e: review round — stress redistributes, coda ratios bucket-stable, knife ②b reverted
  });

  // SHAPE: the whole string, pinned. When this one goes red the question to answer is "did I mean to
  // invalidate every existing bake?" — every term here is part of the dirty-check for stored renders.
  // ⚠ It cannot notice a round that CHANGES lyric→phone behaviour and forgets to bump (both sides would
  // move together); that failure mode has no cheap test, only the review checklist.
  it("has exactly this shape — a change here invalidates every stored bake", () => {
    expect(vocalTrackSig(track, 120)).toBe(
      "vp:sovits,49,2,0,0,0,100,70,15,15,200|sv:|rv:|re:0|vm:V|bpm:120|rr:|g2p:s94|st:s96e",
    );
  });
});

// ── S91: a new/emptied ENGLISH note on an alias track must start out LEGAL in that convention. The
//    language default `a` is not an ARPABET symbol, so on an ARPAsing track the pen tool would mint
//    notes that hard-fail the whole segment render with VOCAL_ALIAS (review S91). ──
describe("aliasDefaultLyric — a new note is never born failing", () => {
  it("is legal in every convention", () => {
    expect(aliasDefaultLyric("arpasing")).toBe("aa"); // `a` is not ARPABET
    expect(aliasDefaultLyric("xsampa")).toBe("a"); // `a` = AA in both alias tables
    expect(aliasDefaultLyric("vccv")).toBe("a");
    expect(aliasDefaultLyric(undefined)).toBe("a"); // words track — the language default, unchanged
  });
});

// ── S91: the error→message mapping for the new VOCAL_ALIAS code. The function had NO test at all
//    (review S91), and its own doc warns that a payload-carrying code must be matched BEFORE the
//    substring checks or a lyric containing a code string hijacks the branch. ──
describe("vocalRenderErrorMessage — VOCAL_ALIAS and the payload-first ordering", () => {
  it("splits the convention from the lyric, and does not hijack other codes", () => {
    // i18n is mocked to echo the key, so the assertions are about WHICH branch fires.
    expect(vocalRenderErrorMessage(new Error("VOCAL_ALIAS: vccv &m"))).toContain("aliasBad");
    // a lyric may contain spaces, brackets, CJK — everything after the convention token is the lyric
    expect(vocalRenderErrorMessage(new Error("VOCAL_ALIAS: xsampa [k ae] 休"))).toContain("aliasBad");
    // an OOV whose LYRIC happens to spell another code must still be an OOV
    expect(vocalRenderErrorMessage(new Error("VOCAL_OOV: VOCAL_ALIAS: x y"))).toContain("oov");
    expect(vocalRenderErrorMessage(new Error("VOCAL_OOV: zzz"))).toContain("oov");
    // …and a VOCAL_ALIAS failure is never mistaken for a user cancel
    expect(isVocalCancelError(new Error("VOCAL_ALIAS: vccv 已取消"))).toBe(false);
  });
});

// ── S91: the track-params → wire-options mapping. It was an inline literal with ZERO coverage until
//    now, which is the shape where a per-track switch silently never reaches Rust: the Rust struct is
//    `#[serde(default)]`, so a forgotten field lands on the production default while the sidebar shows
//    the user's choice — "the switch does nothing", no error anywhere. ──
describe("vocalRenderOptions — every per-track knob must actually reach the wire", () => {
  const vp = { backend: "sovits", speakerId: 49, langId: 2, transpose: 0, formant: 0, transition: DEFAULT_TRANSITION } as const;
  const base = () => ({ ...vp } as unknown as import("../../types/project").VocalTrackParams);

  it("the DEFAULT payload, pinned", () => {
    const o = vocalRenderOptions(base());
    expect({ ...o, sovits: "…", rvc: "…" }).toEqual({
      backend: "sovits", cv_speaker_id: 49, lang_id: 2, transpose: 0,
      range_extend: false, consonant_emphasis_db: 2.5, consonant_valley: 1,
      vowel_clarity: true, consonant_preroll: true, phoneme_set: null,
      sovits: "…", rvc: "…",
    });
  });

  it("★ each knob changes its own field (a dropped one is a switch that silently does nothing)", () => {
    const cases: Array<[Partial<import("../../types/project").VocalTrackParams>, keyof ReturnType<typeof vocalRenderOptions>, unknown]> = [
      [{ rangeExtend: true }, "range_extend", true],
      [{ consonantEmphasis: 0 }, "consonant_emphasis_db", 0],
      [{ consonantValley: 0 }, "consonant_valley", 0],
      [{ vowelClarity: false }, "vowel_clarity", false],
      [{ consonantPreroll: false }, "consonant_preroll", false],
      [{ phonemeSet: "vccv" }, "phoneme_set", "vccv"],
      [{ transpose: -3 }, "transpose", -3],
      [{ langId: 1 }, "lang_id", 1],
      [{ speakerId: 32 }, "cv_speaker_id", 32],
    ];
    for (const [patch, field, want] of cases) {
      expect(vocalRenderOptions({ ...base(), ...patch })[field], `${field}`).toEqual(want);
    }
  });
});
