// S88 — score export canonicalizes the two per-track lyric TRIGGERS. A .ust/.ustx/.mid carries no track
// params, so a custom trigger arriving at the other end (or back here, into a fresh track) would be read as
// an ordinary word: the rests would be SUNG and the breaths would be OOV. Exporting the canonical tokens
// means the file says what it means to every UTAU-family tool. Runs through the REAL store, faking only the
// Rust `export_score_files` invoke.
import { describe, it, expect, beforeEach, vi } from "vitest";

const invokeMock = vi.fn();
vi.mock("@tauri-apps/api/core", () => ({ invoke: (...a: unknown[]) => invokeMock(...a) }));
vi.mock("../../i18n", () => ({ default: { t: (k: string) => k } }));

import { runScoreExport } from "./exportScore";
import { useProjectStore } from "../../store/project";
import type { Note, Track, VocalTrackParams } from "../../types/project";

const mkNote = (id: string, tick: number, lyric: string, pitch = 60): Note => ({
  id, tick, duration: 240, pitch, lyric, velocity: 100,
});

function installTrack(notes: Note[], vocalParams?: Partial<VocalTrackParams>): void {
  const track: Track = {
    id: "t", name: "V", trackType: "vocal", volumeDb: 0, pan: 0, muted: false, solo: false,
    expanded: true, laneControls: {}, voiceModel: "V",
    vocalParams: {
      backend: "sovits", speakerId: 49, langId: 2, transpose: 0, formant: 0,
      transition: { offsetMs: 0, durLeftMs: 100, durRightMs: 70, depthLeftCents: 15, depthRightCents: 15, openEdgeCents: 200 },
      ...vocalParams,
    },
    segments: [{ id: "s", startTick: 0, durationTicks: 1920, content: { type: "notes", notes } }],
  };
  useProjectStore.setState({ tracks: [track], tempo: 120, timeSignature: [4, 4] });
}

/** The lyrics actually handed to Rust, in order. */
async function exportedLyrics(): Promise<string[]> {
  await runScoreExport("ust", "D:/out.ust", ["t"]);
  const [, args] = invokeMock.mock.calls[0] as [string, { tracks: { notes: { lyric: string }[] }[] }];
  return args.tracks[0]!.notes.map((n) => n.lyric);
}

describe("runScoreExport — lyric trigger canonicalization", () => {
  beforeEach(() => {
    invokeMock.mockReset();
    invokeMock.mockResolvedValue(undefined);
  });

  it("a CUSTOM rest token goes out as R (re-importing it must not sing the rest)", async () => {
    installTrack([mkNote("a", 0, "か"), mkNote("r", 240, "休"), mkNote("c", 480, "き")], { restToken: "休" });
    expect(await exportedLyrics()).toEqual(["か", "R", "き"]);
  });

  it("a CUSTOM breath token goes out as AP (re-importing it must not be OOV)", async () => {
    installTrack([mkNote("a", 0, "か"), mkNote("b", 240, "呼")], { breathToken: "呼" });
    expect(await exportedLyrics()).toEqual(["か", "AP"]);
  });

  it("the canonical tokens and ordinary lyrics pass through untouched", async () => {
    // `rest` is an ordinary word since S86 — exporting it as `R` would be the very theft that round undid.
    installTrack([mkNote("a", 0, "R"), mkNote("b", 240, "AP"), mkNote("c", 480, "rest"), mkNote("d", 720, "か")]);
    expect(await exportedLyrics()).toEqual(["R", "AP", "rest", "か"]);
  });

  it("a BLANK lyric also goes out as R — it is a rest everywhere downstream, and blank means nothing to UTAU", async () => {
    installTrack([mkNote("a", 0, "か"), mkNote("blank", 240, "")]);
    expect(await exportedLyrics()).toEqual(["か", "R"]);
  });

  it("with the DEFAULT tokens the payload is exactly the written lyrics (no silent rewriting)", async () => {
    const lyrics = ["か", "き", "く", "け"];
    installTrack(lyrics.map((l, i) => mkNote(`n${i}`, i * 240, l)));
    expect(await exportedLyrics()).toEqual(lyrics);
  });

  it("pitch/tick/duration are untouched by the mapping (only the lyric is canonicalized)", async () => {
    installTrack([mkNote("r", 240, "休", 71)], { restToken: "休" });
    await runScoreExport("ust", "D:/out.ust", ["t"]);
    const [, args] = invokeMock.mock.calls[0] as [string, { tracks: { notes: { tick: number; duration: number; pitch: number }[] }[] }];
    expect(args.tracks[0]!.notes[0]).toMatchObject({ tick: 240, duration: 240, pitch: 71 });
  });
});
