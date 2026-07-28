// S87 — the note-warning WATCHER, driven end-to-end through the real project store.
//
// Two things it must never do, both user-reported classes:
//   1. lose a verdict across a SPLIT — the right half gets a new segment id AND fresh note ids
//      (store/project.ts: "SAME ids = corruption"), so the marks have to be re-published against the new
//      identities, and the left half must keep only what it still owns;
//   2. gate the FRAME verdicts (too-short / rescued-by-borrow) behind `validate_lyrics`. Those come
//      straight out of buildScoreTriples and need no backend; the catch below only stamps `validated`, so
//      a failing classifier used to drop them PERMANENTLY (never retried).
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

let invokeFails = false;
const invokeMock = vi.fn((_cmd: string, args: { notes?: unknown[] }) =>
  invokeFails
    ? Promise.reject(new Error("validate_lyrics exploded"))
    : Promise.resolve((args?.notes ?? []).map(() => ({ kind: "ok" }))),
);
vi.mock("@tauri-apps/api/core", () => ({ invoke: (c: string, a: never) => invokeMock(c, a) }));
vi.mock("../../i18n", () => ({ default: { t: (k: string) => k } }));
vi.mock("../../store/voice-models", () => ({
  useVoiceModelStore: { getState: () => ({ models: { rvc: [{ name: "V", path: "p" }] } }) },
}));

import { splitSegmentVocalAware } from "./vocalRender";
import { installOovWatch } from "./oovWatch";
import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import { installHistory } from "../../store/history";
import { DEFAULT_TRANSITION } from "../vocalNotes";
import type { Note, Track, Segment, SegmentContent } from "../../types/project";

const TEMPO = 120; // ticksPerFrame 19.2
const mk = (id: string, tick: number, duration: number, pitch: number): Note =>
  ({ id, tick, duration, pitch, lyric: "あ", velocity: 100 });
/** `b` spans 768..769 → frames 40..40 = ZERO ⇒ it must be rescued by a borrow. */
const NOTES = () => [mk("a", 0, 700, 60), mk("b", 768, 1, 72), mk("c", 1000, 480, 64)];

async function settle() {
  for (let i = 0; i < 3; i++) {
    await vi.advanceTimersByTimeAsync(400);
    for (let k = 0; k < 5; k++) await Promise.resolve();
  }
}

function seed() {
  const seg: Segment = { id: SEG, startTick: 0, durationTicks: 2000, content: { type: "notes", notes: NOTES() } as SegmentContent };
  const track: Track = {
    id: "T", name: "V", trackType: "vocal", segments: [seg],
    volumeDb: 0, pan: 0, muted: false, solo: false, expanded: true, laneControls: {},
  };
  useProjectStore.setState({ name: "P", tracks: [track], tempo: TEMPO, timeSignature: [4, 4], dirty: false, filePath: null, selectedNotes: [], playheadTick: 0 });
  installHistory();
}

let uninstall: (() => void) | null = null;
// oovWatch keeps a module-level `validated` map that survives between tests — give every test its OWN
// segment id so a previous test's stamp can never skip this one's first pass.
let seq = 0;
let SEG = "S0";

describe("oovWatch — frame warnings survive a split and a backend failure", () => {
  beforeEach(() => {
    (globalThis as unknown as { window: unknown }).window = globalThis;
    vi.useFakeTimers();
    invokeFails = false;
    useAppStore.setState({ vocalOov: {}, vocalDropped: {}, vocalShort: {} });
    SEG = `S${++seq}`;
    seed();
  });
  afterEach(() => {
    uninstall?.();
    uninstall = null;
    vi.useRealTimers();
  });

  it("publishes the rescued note, then FOLLOWS it into the right half of a split (new ids and all)", async () => {
    uninstall = installOovWatch();
    await settle();
    expect(useAppStore.getState().vocalShort).toEqual({ [SEG]: ["b"] });

    // split BEFORE b ⇒ b moves to the new right half and is given a FRESH note id
    const newId = splitSegmentVocalAware("T", SEG, 750, TEMPO)!;
    expect(newId).toBeTruthy();
    await settle();
    const map = useAppStore.getState().vocalShort;
    expect(Object.keys(map)).toEqual([newId]); // the left half no longer owns it…
    const rightNotes = (useProjectStore.getState().tracks[0]!.segments.find((s) => s.id === newId)!
      .content as Extract<SegmentContent, { type: "notes" }>).notes;
    expect(map[newId]).toEqual([rightNotes.find((n) => n.pitch === 72)!.id]); // …and the id is the NEW one
  });

  it("keeps the verdict on the LEFT half when the note stays there", async () => {
    uninstall = installOovWatch();
    await settle();
    const newId = splitSegmentVocalAware("T", SEG, 900, TEMPO)!; // b (768) stays left
    await settle();
    expect(useAppStore.getState().vocalShort).toEqual({ [SEG]: ["b"] });
    expect(useAppStore.getState().vocalShort[newId]).toBeUndefined();
  });

  it("★ publishes the FRAME verdicts even when validate_lyrics fails (they need no backend)", async () => {
    invokeFails = true;
    uninstall = installOovWatch();
    await settle();
    expect(useAppStore.getState().vocalShort).toEqual({ [SEG]: ["b"] });
    expect(useAppStore.getState().vocalOov).toEqual({}); // only the LYRIC verdict is lost to the failure
  });

  // §user control experiment: with TWO sub-frame notes, cutting BETWEEN them left the first half marked
  // and the second half blank — "只判断了第一片" — and it never recovered.
  it("★ marks BOTH halves when each half owns a flagged note", async () => {
    const two = [
      mk("a", 0, 700, 60), mk("b1", 768, 1, 72), mk("c", 1000, 480, 64),
      mk("b2", 1600, 1, 74), mk("d", 1800, 480, 62),
    ];
    // ★ WITH A BAKE — the real-world shape: the user had rendered, so the split also runs
    // stampSplitWindowSigs → replaceProcessedOutputs, i.e. extra store writes right after the split.
    const seg: Segment = {
      id: SEG, startTick: 0, durationTicks: 2400,
      content: { type: "notes", notes: two } as SegmentContent,
      processedOutputs: [{ laneId: "vocal", laneLabel: "V", group: "V", audioPath: "x.wav", totalDurationMs: 5000, waveformPeaks: [0.1], outputNodeId: "vocal" }],
    };
    const track: Track = {
      id: "T", name: "V", trackType: "vocal", segments: [seg], voiceModel: "V",
      vocalParams: { backend: "rvc", speakerId: 0, langId: 2, transpose: 0, formant: 0, transition: DEFAULT_TRANSITION, breathToken: "AP" },
      volumeDb: 0, pan: 0, muted: false, solo: false, expanded: true, laneControls: {},
    };
    useProjectStore.setState({ name: "P", tracks: [track], tempo: TEMPO, timeSignature: [4, 4], dirty: false, filePath: null, selectedNotes: [], playheadTick: 0 });
    installHistory();
    uninstall = installOovWatch();
    await settle();
    expect(useAppStore.getState().vocalShort[SEG]).toEqual(["b1", "b2"]);

    const rightId = splitSegmentVocalAware("T", SEG, 1500, TEMPO)!; // between b1 and b2
    await settle();
    const map = useAppStore.getState().vocalShort;
    expect(map[SEG], "LEFT half keeps b1").toEqual(["b1"]);
    expect(map[rightId], "RIGHT half must be marked too").toHaveLength(1);
  });

  it("clears every channel when the segment goes away", async () => {
    uninstall = installOovWatch();
    await settle();
    expect(useAppStore.getState().vocalShort).toEqual({ [SEG]: ["b"] });
    useProjectStore.getState().deleteSegment("T", SEG);
    await settle();
    expect(useAppStore.getState().vocalShort).toEqual({});
    expect(useAppStore.getState().vocalDropped).toEqual({});
  });
});
