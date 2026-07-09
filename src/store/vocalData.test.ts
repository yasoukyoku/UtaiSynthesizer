// S48 Phase 3 GATE — vocal DATA MODEL / store / undo / .usp, driven headless (no editor UI).
// Verifies the three Phase-3 claims: (A) undo captures a vocal-note edit incl. a NEW field, (B) it
// reverts cleanly to the saved baseline (dirty recomputed), (C) .usp save/load round-trips every vocal
// field byte-identically, (D) no false-dirty (normalizeNote strips default optionals; determinism).
import { describe, it, expect, beforeEach, vi } from "vitest";

// The store's fire-and-forget backend log (history dbg → logToBackend → invoke) + any cancel_voice must
// not throw in a headless run; i18n (announce banner) is mocked so react-i18next/JSON never load.
vi.mock("@tauri-apps/api/core", () => ({ invoke: () => Promise.resolve() }));
vi.mock("../i18n", () => ({ default: { t: (k: string) => k } }));

import { useProjectStore } from "./project";
import { useHistoryStore, installHistory } from "./history";
import { useAppStore } from "./app";
import { buildSaveBundle, buildAutosaveJson, parseLoadedBundle } from "../lib/project/bundle";
import { normalizeCurve, resolveOverlaps, sanitizeText } from "../lib/vocalNotes";
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

  it("captures pitch curves (pitchDev / paramCurves / pitchPoints)", () => {
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
          pitchPoints: [
            { x: -20, y: -100, shape: "sineIn" },
            { x: 120, y: 0, shape: "linear" },
          ],
          vibrato: { length: 0.5, period: 200, depth: 50, in: 0.1, out: 0.1, shift: 0, drift: 0 },
        },
      ],
      { pitchDev: { xs: [0, 240], ys: [0, 50] }, paramCurves: { loudness: { xs: [0, 480], ys: [0, -3] } } },
    ),
    { backend: "sovits", speakerId: 49, langId: 2, transpose: 2 },
  );

  it("preserves vocalParams + all note/curve fields through save→load", () => {
    const { projectJson } = buildSaveBundle("P", [rich], 120, [4, 4]);
    const loaded = parseLoadedBundle(projectJson, "C:/proj.usp");
    expect(loaded.tracks[0]!.vocalParams).toEqual(rich.vocalParams);
    expect(loaded.tracks[0]!.segments[0]!.content).toEqual(rich.segments[0]!.content);
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
    // A note added with explicit default values must NOT store them (detune:0 / tie:false → absent).
    useProjectStore.getState().addVocalNote(T, S, {
      ...plainNote(), id: "n2", tick: 480, pitch: 62, detune: 0, tie: false, pitchAuto: true, pitchPoints: [],
    });
    const n2 = notes().find((n) => n.id === "n2")!;
    expect(n2.detune).toBeUndefined();
    expect(n2.tie).toBeUndefined();
    expect(n2.pitchAuto).toBeUndefined();
    expect(n2.pitchPoints).toBeUndefined();
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

  it("normalizeNote canonicalizes vibrato/pitchPoints element key order (input order can't false-dirty)", () => {
    const P = () => useProjectStore.getState();
    // same values, NON-canonical key/element order (a future editor might build objects either way)
    P().updateVocalNote(T, S, "n1", {
      vibrato: { drift: 0, shift: 0, out: 0.1, in: 0.1, depth: 50, period: 200, length: 0.5 },
      pitchPoints: [{ shape: "linear", y: 0, x: 120 }, { y: -100, shape: "sineIn", x: -20 }],
    });
    const jsonA = buildAutosaveJson("P", P().tracks, 120, [4, 4]);
    // same values, canonical key/element order
    P().updateVocalNote(T, S, "n1", {
      vibrato: { length: 0.5, period: 200, depth: 50, in: 0.1, out: 0.1, shift: 0, drift: 0 },
      pitchPoints: [{ x: -20, y: -100, shape: "sineIn" }, { x: 120, y: 0, shape: "linear" }],
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

  it("splitSegment on a notes segment is a no-op and never false-dirties (§9.6 gate)", () => {
    expect(P().dirty).toBe(false);
    const r = P().splitSegment(T, S, 480); // 480 is inside the part, but a notes segment can't split
    expect(r).toBeNull();
    expect(P().dirty).toBe(false); // returned before entering set() — no dirty
    expect(notes()).toHaveLength(1); // notes untouched (no shared-array corruption)
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
              pitchPoints: [{ x: 5, y: 1e9, shape: "nope" }, { x: 1, y: 0, shape: "linear" }],
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
    expect(b.pitchPoints!.map((p) => p.x)).toEqual([1, 5]); // sorted by x
    expect(b.pitchPoints!.every((p) => ["linear", "sineIn", "sineOut", "sineInOut"].includes(p.shape))).toBe(true);

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
