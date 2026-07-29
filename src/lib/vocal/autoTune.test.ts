// S73 自动调教——调教所有权谓词的纯函数单测(应用链路的 store/undo 行为在 vocalData.test.ts)。
// S73c 语义:pitchDev 不参与 θ 资格(手绘=独立叠加层,机器永不写;基线在其下照常再生成=SV 同构)。
import { describe, it, expect, vi } from "vitest";

/** run_autotune calls captured by the test — the payload IS the contract (what the model gets to see). */
let autotunePayloads: { startMs: number; durMs: number; pitch: number }[][] = [];
vi.mock("@tauri-apps/api/core", () => ({
  invoke: (cmd: string, args: { notes?: { startMs: number; durMs: number; pitch: number }[] }) => {
    if (cmd !== "run_autotune") return Promise.resolve();
    const notes = args?.notes ?? [];
    autotunePayloads.push(notes);
    // one θ row per payload note, all fields concrete (shape mirrors AutotuneTheta)
    return Promise.resolve(
      notes.map(() => ({
        transition: { offsetMs: 0, durLeftMs: 90, durRightMs: 60, depthLeftCents: 12, depthRightCents: 12, openEdgeCents: 150 },
        vibrato: { depthCents: 40, freqHz: 5.5, phase: 0, startMs: 200, easeInMs: 100, easeOutMs: 100 },
      })),
    );
  },
}));
vi.mock("../../i18n", () => ({ default: { t: (k: string) => k } }));

import { isUserTuned, autoTuneScalesOf, phaseForTake, applyAutoTune } from "./autoTune";
import { useProjectStore } from "../../store/project";
import type { Note, Track, Segment, SegmentContent } from "../../types/project";

const note = (extra: Partial<Note> = {}): Note => ({
  id: "n1", tick: 480, duration: 480, pitch: 60, lyric: "か", velocity: 100, ...extra,
});

describe("isUserTuned — θ 维度的自动调教绕行谓词(S73c:不看 pitchDev)", () => {
  it("裸音符 = 未调教(机器可调)", () => {
    expect(isUserTuned(note())).toBe(false);
  });
  it("手设 vibrato / transition = 用户调教(含 ustx 烤入的显式零 transition)", () => {
    expect(isUserTuned(note({ vibrato: { depthCents: 80, freqHz: 5.5, phase: 0, startMs: 0, easeInMs: 80, easeOutMs: 120 } }))).toBe(true);
    expect(isUserTuned(note({ transition: { durLeftMs: 0 } }))).toBe(true);
  });
  it("autoTuned 标记 = 机器调教,即使带 vibrato/transition 也可被机器改写", () => {
    expect(
      isUserTuned(
        note({
          autoTuned: true,
          transition: { durLeftMs: 120 },
          vibrato: { depthCents: 80, freqHz: 5.5, phase: 0, startMs: 0, easeInMs: 80, easeOutMs: 120 },
        }),
      ),
    ).toBe(false);
  });
});

describe("autoTuneScalesOf — 旋钮单一读取点", () => {
  it("absent 默认 = expr 2 / vib 1 / take 0(S73c/d 拍板)", () => {
    expect(autoTuneScalesOf(undefined)).toEqual({ expr: 2, vib: 1, take: 0 });
  });
  it("显式值透传", () => {
    expect(
      autoTuneScalesOf({ autoTuneExpr: 0.5, autoTuneVib: 1.5, autoTuneTake: 7 } as never),
    ).toEqual({ expr: 0.5, vib: 1.5, take: 7 });
  });
});

describe("phaseForTake — 确定性唱法版本(S73d,替代 Retake 抽奖)", () => {
  it("take 0 = 基准相位 0(KA3 耳测口径)", () => {
    expect(phaseForTake(0, "any-id")).toBe(0);
  });
  it("同 (take, id) 恒同相位;换 take/换 id 相位不同;域 [-0.5, 0.5)", () => {
    const a = phaseForTake(3, "n1");
    expect(phaseForTake(3, "n1")).toBe(a);
    expect(phaseForTake(4, "n1")).not.toBe(a);
    expect(phaseForTake(3, "n2")).not.toBe(a);
    for (let t = 1; t <= 20; t++) {
      const p = phaseForTake(t, "note-uuid-xyz");
      expect(p).toBeGreaterThanOrEqual(-0.5);
      expect(p).toBeLessThan(0.5);
    }
  });
});

// ── S88: the model is asked about the SUNG line only ────────────────────────────────────────────────
// Found by the S88 review as a MISSED CONSUMER: run_autotune only ever receives {startMs,durMs,pitch}, so
// a silent note left in the payload tells the model "these two abut" exactly where the render now sees a
// phrase edge — it then predicts a legato portamento for an edge that gets rendered as a release, and the
// "a written rest IS a gap" promise fails on any auto-tuned track (the default). `applyAutoTune` had NO
// test at all before this, which is why the gap survived.
describe("applyAutoTune — silent notes are not part of the tuned line", () => {
  const mkNote = (id: string, tick: number, pitch: number, lyric: string): Note =>
    ({ id, tick, duration: 480, pitch, lyric, velocity: 100 });

  function seed(notes: Note[], restToken?: string): void {
    const seg: Segment = { id: "S", startTick: 0, durationTicks: 2400, content: { type: "notes", notes } as SegmentContent };
    const track: Track = {
      id: "T", name: "V", trackType: "vocal", segments: [seg], voiceModel: "V",
      vocalParams: {
        backend: "sovits", speakerId: 49, langId: 2, transpose: 0, formant: 0,
        transition: { offsetMs: 0, durLeftMs: 100, durRightMs: 70, depthLeftCents: 15, depthRightCents: 15, openEdgeCents: 200 },
        ...(restToken ? { restToken } : {}),
      },
      volumeDb: 0, pan: 0, muted: false, solo: false, expanded: true, laneControls: {},
    };
    useProjectStore.setState({ name: "P", tracks: [track], tempo: 120, timeSignature: [4, 4], dirty: false, filePath: null, selectedNotes: [], playheadTick: 0 });
    autotunePayloads = [];
  }

  const notesNow = () =>
    (useProjectStore.getState().tracks[0]!.segments[0]!.content as Extract<SegmentContent, { type: "notes" }>).notes;

  it("★ a rest note is left OUT of the payload — the model sees the gap the render will render", async () => {
    // か(0..480) · R(480..960) · き(960..1440), all abutting on the page.
    seed([mkNote("a", 0, 60, "か"), mkNote("r", 480, 71, "R"), mkNote("c", 960, 64, "き")]);
    const res = await applyAutoTune("T", "S", { expr: 1, vib: 1, take: 0 });
    expect(autotunePayloads).toHaveLength(1);
    expect(autotunePayloads[0]!.map((n) => n.pitch), "only the sung notes").toEqual([60, 64]);
    // …and they arrive SEPARATED, exactly as the same music written as a gap would: か ends at 1000 ms
    // (tempo 120) while き starts at 2000 ms. A payload containing the rest would have said abut_next=1.
    expect(autotunePayloads[0]![0]!.startMs + autotunePayloads[0]![0]!.durMs).toBeLessThan(autotunePayloads[0]![1]!.startMs);
    expect(res.applied).toBe(2);

    const after = notesNow();
    expect(after.find((n) => n.id === "r")!.autoTuned, "a silence has nothing to tune").toBeUndefined();
    expect(after.find((n) => n.id === "r")!.transition, "…and writing one would only churn contentSig").toBeUndefined();
    expect(after.find((n) => n.id === "a")!.autoTuned).toBe(true);
    expect(after.find((n) => n.id === "c")!.autoTuned).toBe(true);
  });

  it("the same holds for a CUSTOM rest token and for a breath", async () => {
    seed([mkNote("a", 0, 60, "か"), mkNote("r", 480, 71, "休"), mkNote("b", 960, 65, "AP"), mkNote("c", 1440, 64, "き")], "休");
    await applyAutoTune("T", "S", { expr: 1, vib: 1, take: 0 });
    expect(autotunePayloads[0]!.map((n) => n.pitch)).toEqual([60, 64]);
    const after = notesNow();
    expect(after.find((n) => n.id === "r")!.autoTuned).toBeUndefined();
    expect(after.find((n) => n.id === "b")!.autoTuned).toBeUndefined();
  });

  it("a segment with nothing but rests tunes nothing and calls no model", async () => {
    seed([mkNote("r1", 0, 60, "R"), mkNote("r2", 480, 62, "R")]);
    const res = await applyAutoTune("T", "S", { expr: 1, vib: 1, take: 0 });
    expect(autotunePayloads).toHaveLength(0);
    expect(res).toEqual({ applied: 0, skipped: 2 });
  });
});
