// S87 — END-TO-END gate for score import (ust / ustx / midi) through the REAL store, with only the two
// process boundaries faked: the native file picker and the Rust `import_score_file` invoke. The point is
// that the new grid-rounding option is exercised on the actual production path (§user blood-lesson: the
// user must never be the first to run a newly wired path), not just on the pure quantizer beneath it.
//
// The two cases that matter are the two real-world scores from the S86 baseline:
//   · a ja .ust with a 30t note @ tempo 222 (16.9 ms — SHORTER than one 20 ms render frame, so it rounds to
//     zero frames and never sounds): rounding must rescue it, and must not disturb the rest of the score;
//   · a UTAU CVVC score whose starts are hand-shifted preutterance offsets: rounding OFF must leave the
//     import byte-identical to the pre-S87 behavior.
import { describe, it, expect, beforeEach, vi } from "vitest";

const invokeMock = vi.fn();
const openMock = vi.fn();
vi.mock("@tauri-apps/api/core", () => ({ invoke: (...a: unknown[]) => invokeMock(...a) }));
vi.mock("@tauri-apps/plugin-dialog", () => ({ open: (...a: unknown[]) => openMock(...a) }));
vi.mock("../../i18n", () => ({ default: { t: (k: string) => k } }));
vi.mock("../project/autosave", () => ({ flushAutosaveNow: () => {} }));

import { importScoreFile } from "./import";
import { QUANTIZE_IMPORT_KEY, GRID_QUANT_TICKS } from "./quantize";
import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import { installHistory, useHistoryStore, timelineUndoDepth } from "../../store/history";
import type { Note, SegmentContent } from "../../types/project";

const Q = GRID_QUANT_TICKS; // 40t

interface RawNote { tick: number; duration: number; pitch: number; lyric: string; detune?: number | null }
function score(notes: RawNote[], startTick = 0, pitchDev: { xs: number[]; ys: number[] } | null = null) {
  return {
    tracks: [
      {
        name: "V",
        start_tick: startTick,
        notes: notes.map((n) => ({ detune: null, ...n })),
        pitch_dev: pitchDev,
        pitch_dev_dropped: false,
      },
    ],
    bpm: 222,
    time_sig: [4, 4] as [number, number],
  };
}

type ConfirmOpts = Parameters<ReturnType<typeof useAppStore.getState>["showConfirm"]>[0];

/** Drive the options dialog: answer OK and set the checkbox to `quantize`. */
function answerDialog(quantize: boolean) {
  useAppStore.setState({
    showConfirm: async (opts: ConfirmOpts) => {
      opts.check?.onChange(quantize);
      return "ok";
    },
  });
}

function imported(): { start: number; dur: number; notes: Note[] } {
  const tracks = useProjectStore.getState().tracks;
  const seg = tracks[tracks.length - 1]!.segments[0]!;
  const content = seg.content as Extract<SegmentContent, { type: "notes" }>;
  return { start: seg.startTick, dur: seg.durationTicks, notes: content.notes };
}

describe("importScoreFile — S87 grid-rounding option", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    invokeMock.mockReset();
    openMock.mockReset();
    openMock.mockResolvedValue("D:/songs/nursery.ust");
    useProjectStore.setState({ name: "P", tracks: [], tempo: 120, timeSignature: [4, 4], dirty: false, filePath: null, selectedNotes: [], playheadTick: 0 });
    installHistory();
    try {
      localStorage.removeItem(QUANTIZE_IMPORT_KEY);
    } catch {
      /* node env — loadSetting falls back to the default anyway */
    }
  });

  it("rounding OFF imports the authored timing verbatim (the CVVC case)", async () => {
    // Hand-shifted starts, exactly the shape a UTAU CVVC .ust has.
    invokeMock.mockResolvedValue(score([
      { tick: 0, duration: 137, pitch: 60, lyric: "a" },
      { tick: 137, duration: 223, pitch: 62, lyric: "i" },
      { tick: 360, duration: 480, pitch: 64, lyric: "u" },
    ], 1000));
    answerDialog(false);
    await importScoreFile();
    const r = imported();
    expect(r.start).toBe(1000); // part start straight from the file
    expect(r.notes.map((n) => [n.tick, n.duration])).toEqual([[0, 137], [137, 223], [360, 480]]);
    expect(r.dur).toBe(840);
  });

  it("rounding ON snaps boundaries in ABSOLUTE space and keeps legato", async () => {
    invokeMock.mockResolvedValue(score([
      { tick: 0, duration: 137, pitch: 60, lyric: "a" },
      { tick: 137, duration: 223, pitch: 62, lyric: "i" },
      { tick: 360, duration: 480, pitch: 64, lyric: "u" },
    ], 1000));
    answerDialog(true);
    await importScoreFile();
    const r = imported();
    // absolute starts were 1000 / 1137 / 1360 → 1000 / 1120 / 1360 (all multiples of 40)
    expect(r.start % Q).toBe(0);
    for (const n of r.notes) expect((r.start + n.tick) % Q).toBe(0);
    // legato: every note still begins exactly where the previous one ended
    const ends = r.notes.map((n) => n.tick + n.duration);
    expect(r.notes[1]!.tick).toBe(ends[0]);
    expect(r.notes[2]!.tick).toBe(ends[1]);
  });

  it("rounding ON RESCUES a sub-frame note instead of letting it vanish (the 30t @ tempo 222 case)", async () => {
    invokeMock.mockResolvedValue(score([
      { tick: 0, duration: 480, pitch: 60, lyric: "く" },
      { tick: 480, duration: 30, pitch: 62, lyric: "ず" }, // 16.9 ms @ 222bpm — under one 20 ms frame
      { tick: 510, duration: 480, pitch: 64, lyric: "あ" },
    ], 0));
    answerDialog(true);
    await importScoreFile();
    const r = imported();
    expect(r.notes).toHaveLength(3); // it did NOT disappear
    const rescued = r.notes.find((n) => n.lyric === "ず")!;
    expect(rescued.duration).toBeGreaterThanOrEqual(Q); // widened to a full cell = audible again
    // the follower yielded the cell from its START; nothing after it moved
    const last = r.notes.find((n) => n.lyric === "あ")!;
    expect(last.tick).toBe(rescued.tick + rescued.duration);
    expect(last.tick + last.duration).toBe(1000); // 990 rounds to 1000 — the tail is where rounding put it
  });

  it("a BAKED pitch curve exempts its track from rounding (curve stays keyed to its notes)", async () => {
    invokeMock.mockResolvedValue(score(
      [{ tick: 0, duration: 137, pitch: 60, lyric: "a" }, { tick: 137, duration: 223, pitch: 62, lyric: "i" }],
      1000,
      { xs: [0, 100, 300], ys: [0, 50, -20] },
    ));
    answerDialog(true);
    await importScoreFile();
    const r = imported();
    expect(r.start).toBe(1000);
    expect(r.notes.map((n) => [n.tick, n.duration])).toEqual([[0, 137], [137, 223]]);
  });

  it("Cancel touches NOTHING (no track, no undo step)", async () => {
    invokeMock.mockResolvedValue(score([{ tick: 0, duration: 480, pitch: 60, lyric: "a" }]));
    useAppStore.setState({ showConfirm: async () => "" } as never);
    const before = timelineUndoDepth();
    await importScoreFile();
    expect(useProjectStore.getState().tracks).toHaveLength(0);
    expect(timelineUndoDepth()).toBe(before);
  });

  it("the whole import is ONE undo step, rounding on or off", async () => {
    invokeMock.mockResolvedValue(score([
      { tick: 0, duration: 137, pitch: 60, lyric: "a" },
      { tick: 137, duration: 223, pitch: 62, lyric: "i" },
    ], 1000));
    answerDialog(true);
    const before = timelineUndoDepth();
    await importScoreFile();
    expect(useProjectStore.getState().tracks).toHaveLength(1);
    expect(timelineUndoDepth()).toBe(before + 1);
    useHistoryStore.getState().undo();
    expect(useProjectStore.getState().tracks).toHaveLength(0);
  });
});
