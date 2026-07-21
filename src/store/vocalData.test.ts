// S48 Phase 3 GATE — vocal DATA MODEL / store / undo / .usp, driven headless (no editor UI).
// Verifies the three Phase-3 claims: (A) undo captures a vocal-note edit incl. a NEW field, (B) it
// reverts cleanly to the saved baseline (dirty recomputed), (C) .usp save/load round-trips every vocal
// field byte-identically, (D) no false-dirty (normalizeNote strips default optionals; determinism).
import { describe, it, expect, beforeEach, vi } from "vitest";

// Fire-and-forget backend invokes (e.g. cancel_voice) must not throw in a headless run;
// i18n (announce banner) is mocked so react-i18next/JSON never load.
vi.mock("@tauri-apps/api/core", () => ({ invoke: () => Promise.resolve() }));
vi.mock("../i18n", () => ({ default: { t: (k: string) => k } }));

import { useProjectStore } from "./project";
import { useHistoryStore, installHistory } from "./history";
import { useAppStore } from "./app";
import { buildSaveBundle, buildAutosaveJson, parseLoadedBundle } from "../lib/project/bundle";
import { normalizeCurve, resolveOverlaps, sanitizeText, DEFAULT_TRANSITION } from "../lib/vocalNotes";
import type { Track, Segment, SegmentContent, Note, VocalTrackParams } from "../types/project";

type NotesContent = Extract<SegmentContent, { type: "notes" }>;

const T = "t1";
const S = "seg1";

function plainNote(): Note {
  return { id: "n1", tick: 0, duration: 240, pitch: 60, lyric: "か", velocity: 100 };
}
function notesSeg(notes: Note[], extra: Partial<NotesContent> = {}): Segment {
  return { id: S, startTick: 0, durationTicks: 1920, content: { type: "notes", notes, ...extra } };
}
function vocalTrack(seg: Segment, params?: VocalTrackParams): Track {
  return {
    id: T, name: "Vocal", trackType: "vocal", segments: [seg],
    volumeDb: 0, pan: 0, muted: false, solo: false, expanded: true, laneControls: {},
    ...(params ? { vocalParams: params } : {}),
  };
}
function seed(track: Track) {
  useProjectStore.setState({
    name: "P", tracks: [track], tempo: 120, timeSignature: [4, 4],
    dirty: false, filePath: null, selectedNotes: [], playheadTick: 0,
  });
}
function notes(): Note[] {
  return (useProjectStore.getState().tracks[0]!.segments[0]!.content as NotesContent).notes;
}
function content(): NotesContent {
  return useProjectStore.getState().tracks[0]!.segments[0]!.content as NotesContent;
}

let uninstall: (() => void) | null = null;
beforeEach(() => {
  useAppStore.setState({ selectedSegment: null, selectedSegments: [], activeTrackId: null } as never);
  seed(vocalTrack(notesSeg([plainNote()])));
  uninstall?.();
  uninstall = installHistory();
  useHistoryStore.getState().reset();
  useHistoryStore.getState().markSaved(); // baseline = the seeded doc
});

describe("Phase 3 — undo captures + reverts vocal edits (GATE A/B)", () => {
  it("captures an edit to a NEW field (detune) and reverts it clean", () => {
    expect(useProjectStore.getState().dirty).toBe(false);
    useProjectStore.getState().updateVocalNote(T, S, "n1", { detune: 30 });
    expect(notes()[0]!.detune).toBe(30);
    expect(useHistoryStore.getState().canUndo).toBe(true); // the new field was in contentSig → captured
    expect(useProjectStore.getState().dirty).toBe(true);

    useHistoryStore.getState().undo();
    expect(notes()[0]!.detune).toBeUndefined(); // reverted to the seeded (no-detune) note
    expect(useProjectStore.getState().dirty).toBe(false); // sig back to savedSig
    expect(useHistoryStore.getState().canRedo).toBe(true);
  });

  it("captures add / delete note and vocalParams as distinct undo steps", () => {
    useProjectStore.getState().addVocalNote(T, S, { ...plainNote(), id: "n2", tick: 480, pitch: 62 });
    expect(notes()).toHaveLength(2);
    useProjectStore.getState().setVocalParams(T, { transpose: 3 });
    expect(useProjectStore.getState().tracks[0]!.vocalParams).toMatchObject({ transpose: 3, backend: "sovits" });

    useHistoryStore.getState().undo(); // undo setVocalParams
    expect(useProjectStore.getState().tracks[0]!.vocalParams?.transpose ?? 0).toBe(0);
    expect(notes()).toHaveLength(2);
    useHistoryStore.getState().undo(); // undo addVocalNote
    expect(notes()).toHaveLength(1);
    expect(useProjectStore.getState().dirty).toBe(false);
  });

  it("captures pitch curves (pitchDev / paramCurves)", () => {
    useProjectStore.getState().setSegmentPitchDev(T, S, { xs: [0, 240], ys: [0, 50] });
    expect(content().pitchDev).toEqual({ xs: [0, 240], ys: [0, 50] });
    expect(useHistoryStore.getState().canUndo).toBe(true);
    useHistoryStore.getState().undo();
    expect(content().pitchDev).toBeUndefined();
    expect(useProjectStore.getState().dirty).toBe(false);
  });
});

describe("Phase 3 — .usp save/load round-trips every vocal field (GATE C)", () => {
  const rich = vocalTrack(
    notesSeg(
      [
        {
          id: "n1", tick: 0, duration: 240, pitch: 60, lyric: "か", velocity: 100,
          detune: 30, tie: true, pitchAuto: false, lang: "ja", phonemeInput: "ka",
          transition: { offsetMs: -10, durLeftMs: 120, durRightMs: 50, depthLeftCents: 20, depthRightCents: 10 },
          vibrato: { depthCents: 50, freqHz: 5.5, phase: 0, startMs: 250, easeInMs: 200, easeOutMs: 200 },
        },
      ],
      { pitchDev: { xs: [0, 240], ys: [0, 50] }, paramCurves: { loudness: { xs: [0, 480], ys: [0, -3] } } },
    ),
    { backend: "sovits", speakerId: 49, langId: 2, transpose: 2, formant: -3, breathToken: "AP",
      transition: { offsetMs: 0, durLeftMs: 100, durRightMs: 70, depthLeftCents: 15, depthRightCents: 15, openEdgeCents: 50 } },
  );

  it("preserves vocalParams + all note/curve fields through save→load", () => {
    const { projectJson } = buildSaveBundle("P", [rich], 120, [4, 4]);
    const loaded = parseLoadedBundle(projectJson, "C:/proj.usp");
    // S73b:sanitize 载入时补 concrete 的 autoTuneExpr/Vib(默认 1)——夹具没写它们,期望值补齐
    expect(loaded.tracks[0]!.vocalParams).toEqual({ ...rich.vocalParams, autoTuneExpr: 1, autoTuneVib: 1 });
    expect(loaded.tracks[0]!.segments[0]!.content).toEqual(rich.segments[0]!.content);
  });

  it("S73:rangeExtend=true 存读不再被 sanitize 丢弃(存量 bug 修复)", () => {
    const t = { ...rich, vocalParams: { ...rich.vocalParams!, rangeExtend: true as const } };
    const loaded = parseLoadedBundle(buildAutosaveJson("P", [t], 120, [4, 4]), "C:/proj.usp");
    expect(loaded.tracks[0]!.vocalParams?.rangeExtend).toBe(true);
  });

  it("load→serialize is byte-identical (autosave form)", () => {
    // A store-produced doc is always canonical; the hand-written fixture is not (field order), and load now
    // normalizes untrusted input (§9.8.1). Normalize once via a load pass to the canonical baseline, then
    // assert re-loading is a FIXPOINT (reopening never false-dirties). Real app data is already canonical.
    const canonical = parseLoadedBundle(buildAutosaveJson("P", [rich], 120, [4, 4]), "C:/proj.usp");
    const auto1 = buildAutosaveJson("P", canonical.tracks, 120, [4, 4]);
    const reloaded = parseLoadedBundle(auto1, "C:/proj.usp");
    const auto2 = buildAutosaveJson("P", reloaded.tracks, reloaded.tempo, reloaded.timeSignature);
    expect(auto2).toBe(auto1);
  });
});

describe("Phase 3 — no false-dirty (GATE D)", () => {
  it("serializing the same doc twice is byte-identical", () => {
    const t = vocalTrack(notesSeg([plainNote()]));
    expect(buildAutosaveJson("P", [t], 120, [4, 4])).toBe(buildAutosaveJson("P", [t], 120, [4, 4]));
  });

  it("normalizeNote strips default optionals on write (no JSON growth)", () => {
    // A note added with explicit default values must NOT store them (detune:0 / tie:false → absent); an
    // empty transition object and a no-op (zero-amplitude) vibrato canonicalize to absent too.
    useProjectStore.getState().addVocalNote(T, S, {
      ...plainNote(), id: "n2", tick: 480, pitch: 62, detune: 0, tie: false, pitchAuto: true,
      transition: {}, vibrato: { depthCents: 0, freqHz: 5.5, phase: 0, startMs: 0, easeInMs: 0, easeOutMs: 0 },
    });
    const n2 = notes().find((n) => n.id === "n2")!;
    expect(n2.detune).toBeUndefined();
    expect(n2.tie).toBeUndefined();
    expect(n2.pitchAuto).toBeUndefined();
    expect(n2.transition).toBeUndefined();
    expect(n2.vibrato).toBeUndefined();
  });

  it("setting a field back to its default returns to the byte-identical baseline", () => {
    const base = buildAutosaveJson("P", useProjectStore.getState().tracks, 120, [4, 4]);
    useProjectStore.getState().updateVocalNote(T, S, "n1", { detune: 15 });
    expect(buildAutosaveJson("P", useProjectStore.getState().tracks, 120, [4, 4])).not.toBe(base);
    useProjectStore.getState().updateVocalNote(T, S, "n1", { detune: 0 }); // back to default
    expect(buildAutosaveJson("P", useProjectStore.getState().tracks, 120, [4, 4])).toBe(base);
  });

  it("paramCurves key order is canonical — delete-then-readd does not false-dirty", () => {
    const P = () => useProjectStore.getState();
    P().setSegmentParamCurve(T, S, "loudness", { xs: [0], ys: [0] });
    P().setSegmentParamCurve(T, S, "tension", { xs: [0], ys: [0] });
    const baseline = buildAutosaveJson("P", P().tracks, 120, [4, 4]);
    // delete then re-add loudness → without sorted keys the Record would reorder to {tension, loudness}
    P().setSegmentParamCurve(T, S, "loudness", undefined);
    P().setSegmentParamCurve(T, S, "loudness", { xs: [0], ys: [0] });
    expect(buildAutosaveJson("P", P().tracks, 120, [4, 4])).toBe(baseline);
    expect(Object.keys(content().paramCurves!)).toEqual(["loudness", "tension"]); // sorted
  });

  it("normalizeNote canonicalizes vibrato/transition key order (input order can't false-dirty)", () => {
    const P = () => useProjectStore.getState();
    // same values, NON-canonical key order (a future editor might build objects either way)
    P().updateVocalNote(T, S, "n1", {
      vibrato: { easeOutMs: 200, easeInMs: 200, startMs: 250, phase: 0, freqHz: 5.5, depthCents: 50 },
      transition: { depthRightCents: 10, depthLeftCents: 20, durRightMs: 50, durLeftMs: 120, offsetMs: -10 },
    });
    const jsonA = buildAutosaveJson("P", P().tracks, 120, [4, 4]);
    // same values, canonical key order
    P().updateVocalNote(T, S, "n1", {
      vibrato: { depthCents: 50, freqHz: 5.5, phase: 0, startMs: 250, easeInMs: 200, easeOutMs: 200 },
      transition: { offsetMs: -10, durLeftMs: 120, durRightMs: 50, depthLeftCents: 20, depthRightCents: 10 },
    });
    expect(buildAutosaveJson("P", P().tracks, 120, [4, 4])).toBe(jsonA); // normalized → identical bytes
  });
});

// ─── Phase 4a — data-layer additions (sort-on-write / applyNoteEdits / createVocalPart / load sanitize /
//     selectedNotes reconcile) + pure vocalNotes helpers ──────────────────────────────────────────────
describe("Phase 4a — store additions", () => {
  const P = () => useProjectStore.getState();

  it("addVocalNote sorts on write (insert-in-middle keeps tick order)", () => {
    P().addVocalNote(T, S, { ...plainNote(), id: "n3", tick: 960 });
    P().addVocalNote(T, S, { ...plainNote(), id: "n2", tick: 480 });
    expect(notes().map((n) => n.id)).toEqual(["n1", "n2", "n3"]); // n1@0, n2@480, n3@960
  });

  it("applyNoteEdits is ONE atomic undo step (batch add + update + remove)", () => {
    P().addVocalNote(T, S, { ...plainNote(), id: "n2", tick: 480 }); // undo step 1
    P().applyNoteEdits(T, S, {
      add: [{ ...plainNote(), id: "n3", tick: 960 }],
      update: { n1: { pitch: 72 } },
      remove: ["n2"],
    });
    expect(notes().map((n) => n.id)).toEqual(["n1", "n3"]);
    expect(notes().find((n) => n.id === "n1")!.pitch).toBe(72);

    useHistoryStore.getState().undo(); // reverts the WHOLE batch as one step
    expect(notes().map((n) => n.id)).toEqual(["n1", "n2"]);
    expect(notes().find((n) => n.id === "n1")!.pitch).toBe(60);
  });

  it("createVocalPart adds an empty notes segment (returns id) and is undoable", () => {
    const id = P().createVocalPart(T, 1920, 1920);
    const track = P().tracks[0]!;
    expect(track.segments).toHaveLength(2);
    const seg = track.segments.find((s) => s.id === id)!;
    expect(seg.content.type).toBe("notes");
    expect((seg.content as NotesContent).notes).toHaveLength(0);
    expect(seg.startTick).toBe(1920);
    useHistoryStore.getState().undo();
    expect(P().tracks[0]!.segments).toHaveLength(1);
  });

  it("undo reconciles selectedNotes (drops ids whose note no longer exists)", () => {
    P().addVocalNote(T, S, { ...plainNote(), id: "n2", tick: 480 });
    P().selectNotes(["n1", "n2"]);
    useHistoryStore.getState().undo(); // reverts the add → n2 gone
    expect(notes().map((n) => n.id)).toEqual(["n1"]);
    expect(P().selectedNotes).toEqual(["n1"]); // dangling n2 dropped
  });

  it("applyNoteEdits with a no-op (same-value) edit does NOT set dirty (§5 false-dirty)", () => {
    expect(P().dirty).toBe(false);
    P().applyNoteEdits(T, S, { update: { n1: { lyric: notes()[0]!.lyric, pitch: notes()[0]!.pitch } } }); // re-set SAME values
    expect(P().dirty).toBe(false); // canonical-identical → change nothing (no stuck dirty)
    expect(useHistoryStore.getState().canUndo).toBe(false); // no phantom undo step
  });

  // ── S58 language fields ──
  it("Note.lang is WHITELISTED to the 7 codes (junk strips to follow-track); valid codes persist", () => {
    P().applyNoteEdits(T, S, { update: { n1: { lang: "en" } } });
    expect(notes()[0]!.lang).toBe("en");
    P().applyNoteEdits(T, S, { update: { n1: { lang: "klingon" } } });
    expect(notes()[0]!.lang).toBeUndefined(); // invalid → absent = follow the track default
  });

  it("changing the LYRIC drops a stale phonemeInput/phoneme override (S58 invariant)", () => {
    P().applyNoteEdits(T, S, { update: { n1: { phonemeInput: "L AY1 T", phoneme: "x" } } });
    expect(notes()[0]!.phonemeInput).toBe("L AY1 T");
    P().applyNoteEdits(T, S, { update: { n1: { lyric: "night" } } }); // lyric changed, no new override
    expect(notes()[0]!.lyric).toBe("night");
    expect(notes()[0]!.phonemeInput).toBeUndefined(); // the old pinyin/ARPABET no longer applies
    expect(notes()[0]!.phoneme).toBeUndefined();
    // …but supplying BOTH in one edit keeps the new override (an explicit pair is intentional)
    P().applyNoteEdits(T, S, { update: { n1: { lyric: "light", phonemeInput: "L AY1 T" } } });
    expect(notes()[0]!.phonemeInput).toBe("L AY1 T");
    // …and a NON-lyric edit never touches it
    P().applyNoteEdits(T, S, { update: { n1: { pitch: 65 } } });
    expect(notes()[0]!.phonemeInput).toBe("L AY1 T");
  });

  it("splitSegment on a notes part: partitions notes + rebases + slices curves + one undo step (§user)", () => {
    // fixture: n1@[0,240]. Add n2@[480,960] + a pitchDev/loudness curve spanning the part.
    P().addVocalNote(T, S, { ...plainNote(), id: "n2", tick: 480, duration: 480, pitch: 64 });
    P().setSegmentPitchDev(T, S, { xs: [0, 600, 1000], ys: [0, 50, 0] });
    P().setSegmentParamCurve(T, S, "loudness", { xs: [0, 600, 1000], ys: [0, -3, 0] });
    const r = P().splitSegment(T, S, 300); // 300 = in the REST between n1(ends 240) and n2(starts 480) → clean
    expect(r).not.toBeNull();
    const segs = P().tracks[0]!.segments;
    expect(segs).toHaveLength(2);
    const [left, right] = segs;
    expect(left!.durationTicks).toBe(300);
    expect(right!.startTick).toBe(300);
    expect(right!.durationTicks).toBe(1920 - 300);
    // n1(0-240) → left; n2(480-960) → right, tick rebased to 180 with a FRESH id (no shared-id corruption).
    expect((left!.content as NotesContent).notes.map((n) => n.id)).toEqual(["n1"]);
    const rn = (right!.content as NotesContent).notes;
    expect(rn).toHaveLength(1);
    expect(rn[0]!.tick).toBe(180); // 480 − 300
    expect(rn[0]!.id).not.toBe("n2"); // fresh uuid
    // curves sliced+rebased onto both halves (segment-relative). No bake in THIS fixture → processedOutputs
    // stays undefined (a real bake is CARRIED + windowed via offsetMs — see the dedicated split-window test).
    expect((left!.content as NotesContent).pitchDev).toBeDefined();
    expect((right!.content as NotesContent).pitchDev).toBeDefined();
    expect((left!.content as NotesContent).paramCurves?.loudness).toBeDefined();
    expect((right!.content as NotesContent).paramCurves?.loudness).toBeDefined();
    expect(left!.processedOutputs).toBeUndefined(); // no bake in this fixture → nothing to carry
    // ONE undo step reverts the whole split back to a single segment.
    useHistoryStore.getState().undo();
    expect(P().tracks[0]!.segments).toHaveLength(1);
  });

  it("splitSegment CARRIES + WINDOWS a baked stem (offsetMs, renderedSig unchanged); UNDO restores it clean (§user)", () => {
    // give the fixture a baked vocal stem; n1 is [0,240], split at 300 (in the rest) → leftDur 300 ticks.
    P().replaceProcessedOutputs(T, S, [{ laneId: "vocal", laneLabel: "V", group: "V", audioPath: "stem.wav", totalDurationMs: 2000, waveformPeaks: [0.1], outputNodeId: "vocal", renderedSig: "sigFull" }]);
    P().splitSegment(T, S, 300);
    const [left, right] = P().tracks[0]!.segments;
    // BOTH halves carry the SAME stem (not cleared) — split windows it like an audioClip, no re-bake.
    expect(left!.processedOutputs?.[0]?.audioPath).toBe("stem.wav");
    expect(right!.processedOutputs?.[0]?.audioPath).toBe("stem.wav");
    // left keeps offset 0; right advances by the left duration in ms (ticksToMs(300,120)=312.5).
    expect(left!.processedOutputs?.[0]?.offsetMs ?? 0).toBe(0);
    expect(right!.processedOutputs?.[0]?.offsetMs).toBeCloseTo(312.5, 1);
    // renderedSig stays the PARENT whole-stem sig on both (no re-stamp) — so an undo-of-split matches the full
    // content. windowSig (this half's own sig) is stamped frontend-side (stampSplitWindowSigs) — not tested here.
    expect(left!.processedOutputs?.[0]?.renderedSig).toBe("sigFull");
    expect(right!.processedOutputs?.[0]?.renderedSig).toBe("sigFull");
    // UNDO → back to ONE full segment; its whole-stem bake (offset 0, renderedSig=full) is restored, so the
    // normal dirty-check matches the full content → no re-render on undo (dual-sig, real-window verified).
    useHistoryStore.getState().undo();
    const un = P().tracks[0]!.segments;
    expect(un).toHaveLength(1);
    expect(un[0]!.processedOutputs?.[0]?.offsetMs ?? 0).toBe(0);
    expect(un[0]!.processedOutputs?.[0]?.renderedSig).toBe("sigFull");
  });

  it("splitSegment SNAPS a mid-note split to the note's END, keeping the note whole (§user)", () => {
    const r = P().splitSegment(T, S, 100); // 100 is INSIDE n1[0,240] → snaps to 240
    expect(r).not.toBeNull();
    const segs = P().tracks[0]!.segments;
    expect(segs[0]!.durationTicks).toBe(240); // snapped to n1's end
    expect((segs[0]!.content as NotesContent).notes).toHaveLength(1); // n1 whole, on the left
    expect((segs[1]!.content as NotesContent).notes).toHaveLength(0); // right half is the trailing rest
  });

  it("splitSegment at a segment edge / straddle-that-empties-a-half is a no-op (no false-dirty)", () => {
    // With the fixture's single n1[0,240], splitting at 0 (edge) is a no-op; splitting inside n1 near the
    // segment END would snap past it — but here n1 ends at 240 (mid-segment) so that path can't trigger; test
    // the edge no-op which must NOT enter set() (dirty stays false).
    expect(P().dirty).toBe(false);
    expect(P().splitSegment(T, S, 0)).toBeNull();
    expect(P().dirty).toBe(false);
    expect(P().tracks[0]!.segments).toHaveLength(1);
  });
});

describe("Phase 4a — untrusted .usp load boundary (§9.8.1)", () => {
  it("clamps/sanitizes hostile notes + vocalParams; drops id-less notes", () => {
    // Build control-char lyric at RUNTIME (no literal control chars in source): NUL + RLO bidi override.
    const badLyric = "a" + String.fromCharCode(0) + "b" + String.fromCharCode(0x202e) + "c";
    const hostileTrack = {
      id: "t1", name: "V", trackType: "vocal",
      segments: [{
        id: "s1", startTick: 0, durationTicks: 1920,
        content: {
          type: "notes",
          notes: [
            {
              id: "bad", tick: 1e12, duration: 0, pitch: 999, lyric: badLyric, velocity: 5000, detune: 99999,
              transition: { durLeftMs: 1e9, depthLeftCents: 1e9, offsetMs: NaN },
              vibrato: { depthCents: 1e9, freqHz: 1e9, phase: 9, startMs: -5, easeInMs: 1e9, easeOutMs: 1e9 },
            },
            { id: "", tick: 100, duration: 240, pitch: 60, lyric: "x", velocity: 100 }, // id-less → dropped
          ],
        },
      }],
      volumeDb: 0, pan: 0, muted: false, solo: false, expanded: true, laneControls: {},
      vocalParams: { backend: "evil", speakerId: 9999, langId: -5, transpose: 1e9 },
    };
    const json = JSON.stringify({ format: "usp", version: 1, name: "H", tempo: 120, timeSignature: [4, 4], tracks: [hostileTrack] });
    const loaded = parseLoadedBundle(json, "C:/h.usp");
    const seg = loaded.tracks[0]!.segments[0]!.content as NotesContent;

    expect(seg.notes).toHaveLength(1); // id-less note dropped
    const b = seg.notes[0]!;
    expect(Number.isFinite(b.tick)).toBe(true);
    expect(b.tick).toBeLessThanOrEqual(1e9);
    expect(b.duration).toBe(1); // 0 → clamped to ≥1
    expect(b.pitch).toBe(127); // 999 → clamped to MIDI max
    expect(b.velocity).toBe(127);
    expect(b.lyric).toBe("abc"); // control/bidi stripped
    expect(b.detune).toBe(1200); // clamped to ±DETUNE_CAP
    expect(b.transition!.durLeftMs).toBe(2000); // 1e9 → clamped [0,2000]
    expect(b.transition!.depthLeftCents).toBe(1200); // 1e9 → clamped [-1200,1200]
    expect(b.transition!.offsetMs).toBeUndefined(); // NaN dropped (finite-only, canonical)
    expect(b.vibrato!.depthCents).toBe(2400); // 1e9 → clamped [0,2400]
    expect(b.vibrato!.freqHz).toBe(40); // 1e9 → clamped [0.1,40]
    expect(b.vibrato!.phase).toBe(1); // 9 → clamped [-1,1]

    const vp = loaded.tracks[0]!.vocalParams!;
    expect(vp.backend).toBe("sovits"); // "evil" → default
    expect(vp.speakerId).toBe(76); // clamped [0,76]
    expect(vp.langId).toBe(0); // clamped [0,6]
    expect(vp.transpose).toBe(48); // clamped [-48,48]
  });
});

describe("Phase 4a — pure vocalNotes helpers", () => {
  const mk = (id: string, tick: number, dur: number): Note => ({ id, tick, duration: dur, pitch: 60, lyric: "あ", velocity: 100 });

  it("resolveOverlaps truncates a passive tail-overlap and drops a swallowed note; active untouched", () => {
    const out = resolveOverlaps([mk("p", 0, 500), mk("s", 520, 60), mk("a", 480, 480)], new Set(["a"]), 60);
    const byId = Object.fromEntries(out.map((n) => [n.id, n]));
    expect(byId.a!.duration).toBe(480); // active passes through
    expect(byId.p!.duration).toBe(480); // [0,500) truncated to active.start=480 → [0,480)
    expect(byId.s).toBeUndefined(); // [520,580) swallowed by [480,960) → dropped
  });

  it("resolveOverlaps keeps an abutting (legato) neighbor", () => {
    const out = resolveOverlaps([mk("p", 0, 480), mk("a", 480, 480)], new Set(["a"]), 60);
    expect(out.find((n) => n.id === "p")!.duration).toBe(480); // p.end === a.start → not overlap
  });

  it("normalizeCurve: param keeps 0.001 quantum + dedups x; cents rounds to int", () => {
    expect(normalizeCurve({ xs: [0, 10, 10], ys: [1.5, 2.25, 9] }, "param")).toEqual({ xs: [0, 10], ys: [1.5, 9] });
    expect(normalizeCurve({ xs: [0.4, 5.6], ys: [10.9, -3.2] }, "cents")).toEqual({ xs: [0, 6], ys: [11, -3] });
  });

  it("sanitizeText strips control/bidi chars and NFC-normalizes", () => {
    expect(sanitizeText("a" + String.fromCharCode(0) + "b" + String.fromCharCode(0x202e) + "c")).toBe("abc");
    // decomposed か + ゛ (U+304B U+3099) → composed が (U+304C)
    expect(sanitizeText(String.fromCharCode(0x304b, 0x3099))).toBe(String.fromCharCode(0x304c));
  });
});

describe("Phase 5 — property sidebar data-layer (transition override / vibrato)", () => {
  it("per-note transition override stores a PARTIAL; resetting a field re-inherits; both undoable", () => {
    const { applyNoteEdits } = useProjectStore.getState();
    // set ONE field → an explicit partial override (the sidebar's editTransition path — not a full 5-field object)
    applyNoteEdits(T, S, { update: { n1: { transition: { durLeftMs: 200 } } } });
    expect(notes()[0]!.transition).toEqual({ durLeftMs: 200 });
    expect(useHistoryStore.getState().canUndo).toBe(true);
    // reset that field to inherit → normalizeTransition drops the non-finite → transition omitted entirely
    applyNoteEdits(T, S, { update: { n1: { transition: { durLeftMs: undefined } } } });
    expect(notes()[0]!.transition).toBeUndefined();
    useHistoryStore.getState().undo(); // undo the reset
    expect(notes()[0]!.transition).toEqual({ durLeftMs: 200 });
  });

  it("vibrato: add a default spec then depth→0 canonicalizes to no-vibrato (= remove, byte-identical)", () => {
    const { applyNoteEdits } = useProjectStore.getState();
    const spec = { depthCents: 100, freqHz: 5.5, phase: 0, startMs: 250, easeInMs: 200, easeOutMs: 200 };
    applyNoteEdits(T, S, { update: { n1: { vibrato: spec } } });
    expect(notes()[0]!.vibrato).toEqual(spec);
    applyNoteEdits(T, S, { update: { n1: { vibrato: { ...spec, depthCents: 0 } } } });
    expect(notes()[0]!.vibrato).toBeUndefined(); // depthCents≤0 → stripped → absent
  });

  it("REPRO track-default undo: note edit THEN transaction-wrapped track-default edit → undo reverts the track-default (not the note)", () => {
    const hist = () => useHistoryStore.getState();
    const { applyNoteEdits, setVocalParams } = useProjectStore.getState();
    // 1. a note edit, transaction-wrapped like a note-override slider drag
    hist().beginTransaction();
    applyNoteEdits(T, S, { update: { n1: { detune: 20 } } });
    hist().commitTransaction();
    // 2. a track-default transition edit, transaction-wrapped exactly like the sidebar track slider
    hist().beginTransaction();
    setVocalParams(T, { transition: { ...DEFAULT_TRANSITION, durLeftMs: 250 } });
    hist().commitTransaction();
    expect(useProjectStore.getState().tracks[0]!.vocalParams!.transition.durLeftMs).toBe(250);
    // 2b. a SECOND track-default edit with MANY mid-drag sets inside one transaction (real fader drag) → 1 step
    hist().beginTransaction();
    for (const v of [260, 275, 300, 330, 360]) setVocalParams(T, { transition: { ...DEFAULT_TRANSITION, durLeftMs: v } });
    hist().commitTransaction();
    expect(useProjectStore.getState().tracks[0]!.vocalParams!.transition.durLeftMs).toBe(360);
    // 3. undo → reverts the 2b drag to 250 (the whole multi-set drag = ONE step); note intact
    hist().undo();
    expect(useProjectStore.getState().tracks[0]!.vocalParams!.transition.durLeftMs).toBe(250);
    expect(notes()[0]!.detune).toBe(20);
    // 3b. undo again → reverts the first track-default edit; 3c. redo re-applies it
    hist().undo();
    expect(useProjectStore.getState().tracks[0]!.vocalParams?.transition?.durLeftMs ?? 100).toBe(100);
    expect(useHistoryStore.getState().canRedo).toBe(true);
    hist().redo();
    expect(useProjectStore.getState().tracks[0]!.vocalParams!.transition.durLeftMs).toBe(250);
  });

  it("track default transition edit is captured by vocalParamsSig + reverts", () => {
    const { setVocalParams } = useProjectStore.getState();
    setVocalParams(T, { transition: { offsetMs: 0, durLeftMs: 250, durRightMs: 70, depthLeftCents: 15, depthRightCents: 15, openEdgeCents: 50 } });
    expect(useProjectStore.getState().tracks[0]!.vocalParams!.transition.durLeftMs).toBe(250);
    expect(useHistoryStore.getState().canUndo).toBe(true); // vocalParamsSig captures the transition
    useHistoryStore.getState().undo();
    expect(useProjectStore.getState().tracks[0]!.vocalParams?.transition?.durLeftMs).toBeUndefined(); // seeded had no params
  });
});


describe("S73 — autoTuned 调教所有权标记(假脏铁律全套)", () => {
  it("autoTuned:true 进 sig(可撤销)、false/absent 归一为 absent(无假脏)", () => {
    // absent → 设 false = 归一后逐字节同 → applyNoteEdits 的 JSON no-op 守卫吞掉整步
    // (applyNoteEdits = 侧栏/autoTune 的真实写入路径)
    useProjectStore.getState().applyNoteEdits(T, S, { update: { n1: { autoTuned: false } } });
    expect(notes()[0]!.autoTuned).toBeUndefined();
    expect(useHistoryStore.getState().canUndo).toBe(false);
    expect(useProjectStore.getState().dirty).toBe(false);

    // 设 true = 一步 undo + dirty
    useProjectStore.getState().applyNoteEdits(T, S, { update: { n1: { autoTuned: true } } });
    expect(notes()[0]!.autoTuned).toBe(true);
    expect(useHistoryStore.getState().canUndo).toBe(true);
    expect(useProjectStore.getState().dirty).toBe(true);

    useHistoryStore.getState().undo();
    expect(notes()[0]!.autoTuned).toBeUndefined();
    expect(useProjectStore.getState().dirty).toBe(false);
  });

  it("autoTuned 随 .usp 往返字节一致", () => {
    useProjectStore.getState().updateVocalNote(T, S, "n1", { autoTuned: true, vibrato: { depthCents: 80, freqHz: 5.5, phase: 0, startMs: 0, easeInMs: 80, easeOutMs: 120 } });
    const a = buildAutosaveJson("P", useProjectStore.getState().tracks, 120, [4, 4]);
    const parsed = parseLoadedBundle(a, "C:/proj.usp");
    expect((parsed.tracks[0]!.segments[0]!.content as NotesContent).notes[0]!.autoTuned).toBe(true);
    expect(buildAutosaveJson("P", parsed.tracks, 120, [4, 4])).toBe(a); // 幂等 = sig↔serialize 恒一致
  });

  it("S73b:autoTuneFollow 关→开往返 = 折回 absence(serialize 字节回到基线,无假脏)", () => {
    const base = buildAutosaveJson("P", useProjectStore.getState().tracks, 120, [4, 4]);
    useProjectStore.getState().setVocalParams(T, { autoTuneFollow: false });
    expect(useProjectStore.getState().tracks[0]!.vocalParams?.autoTuneFollow).toBe(false);
    useProjectStore.getState().setVocalParams(T, { autoTuneFollow: true });
    expect(useProjectStore.getState().tracks[0]!.vocalParams?.autoTuneFollow).toBeUndefined();
    // vocalParams 现在存在(seed 时没有)→ 序列化会多出该对象;与「同参数直接构造」对齐即无假脏
    const now = buildAutosaveJson("P", useProjectStore.getState().tracks, 120, [4, 4]);
    expect(now).not.toBe(base); // vocalParams 从无到有=真变化
    useProjectStore.getState().setVocalParams(T, { autoTuneFollow: false });
    useProjectStore.getState().setVocalParams(T, { autoTuneFollow: true });
    expect(buildAutosaveJson("P", useProjectStore.getState().tracks, 120, [4, 4])).toBe(now); // 往返=字节不动
  });

  it("手动 vibrato/transition 编辑剥 autoTuned(所有权移交,侧栏语义的 store 层镜像)", () => {
    useProjectStore.getState().updateVocalNote(T, S, "n1", { autoTuned: true, transition: { durLeftMs: 120 } });
    expect(notes()[0]!.autoTuned).toBe(true);
    // 侧栏手动编辑 = update 带 autoTuned: undefined
    useProjectStore.getState().applyNoteEdits(T, S, { update: { n1: { transition: { durLeftMs: 150 }, autoTuned: undefined } } });
    expect(notes()[0]!.transition?.durLeftMs).toBe(150);
    expect(notes()[0]!.autoTuned).toBeUndefined();
  });
});
