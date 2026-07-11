// S59 GATE — tempo-grid correction transforms (×2 / ÷2 / downbeat nudge) + the store write's
// fake-dirty discipline for the new audioClip fields (tempoDetect / stretch / paramCurves).
import { describe, it, expect, beforeEach, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: () => Promise.resolve() }));
vi.mock("../../i18n", () => ({ default: { t: (k: string) => k } }));
// the store imports ../i18n relative to src/store — mock that path too
vi.mock("../i18n", () => ({ default: { t: (k: string) => k } }));

import { doubleTempoDetect, halveTempoDetect, nudgeDownbeat } from "./tempoDetect";
import { useProjectStore } from "../../store/project";
import { contentSig } from "../../store/history";
import type { TempoDetect, Track, Segment } from "../../types/project";

const BPB = 4;

function td(over: Partial<TempoDetect> = {}): TempoDetect {
  return { bpm: 120, anchorMs: 250, downbeat: 3, conf: 0.8, ...over };
}

/** The downbeat MOMENT in ms — every correction must keep this moment ON its (new) grid. */
function downbeatMoment(d: TempoDetect): number {
  return d.anchorMs + d.downbeat * (60000 / d.bpm);
}
/** Is `ms` on the grid of `d` (within float noise)? */
function onGrid(d: TempoDetect, ms: number): boolean {
  const p = 60000 / d.bpm;
  const k = (ms - d.anchorMs) / p;
  return Math.abs(k - Math.round(k)) < 1e-6 && Math.round(k) >= 0;
}

describe("tempo-grid corrections", () => {
  it("×2 keeps the anchor and the downbeat moment on-grid", () => {
    const a = td();
    const b = doubleTempoDetect(a, BPB)!;
    expect(b.bpm).toBe(240);
    expect(b.anchorMs).toBe(a.anchorMs);
    expect(onGrid(b, downbeatMoment(a))).toBe(true);
    // the old downbeat moment is congruent to the new downbeat phase
    const k = Math.round((downbeatMoment(a) - b.anchorMs) / (60000 / b.bpm));
    expect(((k - b.downbeat) % BPB + BPB) % BPB).toBe(0);
  });

  it("÷2 with an ODD downbeat shifts the anchor so the downbeat stays on-grid", () => {
    const a = td({ downbeat: 3 });
    const b = halveTempoDetect(a, BPB)!;
    expect(b.bpm).toBe(60);
    expect(b.anchorMs).toBeCloseTo(a.anchorMs + 60000 / a.bpm, 6); // parity shift by ONE old period
    expect(onGrid(b, downbeatMoment(a))).toBe(true);
  });

  it("÷2 with an EVEN downbeat keeps the anchor", () => {
    const a = td({ downbeat: 2 });
    const b = halveTempoDetect(a, BPB)!;
    expect(b.anchorMs).toBe(a.anchorMs);
    expect(onGrid(b, downbeatMoment(a))).toBe(true);
  });

  it("×2 then ÷2 is a perfect round trip (downbeat class preserved via mod 2·bpb)", () => {
    for (const d of [0, 1, 2, 3]) {
      const a = td({ downbeat: d });
      const back = halveTempoDetect(doubleTempoDetect(a, BPB)!, BPB)!;
      expect(back.bpm).toBe(a.bpm);
      expect(back.anchorMs).toBeCloseTo(a.anchorMs, 6);
      expect(onGrid(back, downbeatMoment(a))).toBe(true);
      // the recovered downbeat marks the SAME moment (congruent mod bpb)
      const k = Math.round((downbeatMoment(a) - back.anchorMs) / (60000 / back.bpm));
      expect(((k - back.downbeat) % BPB + BPB) % BPB).toBe(0);
    }
  });

  it("range guards return null instead of leaving [20,400]", () => {
    expect(doubleTempoDetect(td({ bpm: 220 }), BPB)).toBeNull();
    expect(halveTempoDetect(td({ bpm: 35 }), BPB)).toBeNull();
  });

  it("nudge cycles the downbeat phase", () => {
    let d = td({ downbeat: 2 });
    d = nudgeDownbeat(d, BPB);
    expect(d.downbeat).toBe(3);
    d = nudgeDownbeat(d, BPB);
    expect(d.downbeat).toBe(0);
  });
});

// ─── store write discipline for the new audioClip fields ───

const T = "t1";
const S = "s1";

function audioSeg(): Segment {
  return {
    id: S, startTick: 0, durationTicks: 1920,
    content: { type: "audioClip", sourcePath: "C:/x/a.wav", offsetMs: 0, totalDurationMs: 4000 },
  };
}
function audioTrack(seg: Segment): Track {
  return {
    id: T, name: "A", trackType: "audio", segments: [seg],
    volumeDb: 0, pan: 0, muted: false, solo: false, expanded: false, laneControls: {},
  };
}

describe("setSegmentTempoDetect", () => {
  beforeEach(() => {
    useProjectStore.setState({ tracks: [audioTrack(audioSeg())], dirty: false });
  });

  function clip() {
    const c = useProjectStore.getState().tracks[0]!.segments[0]!.content;
    if (c.type !== "audioClip") throw new Error("not audioClip");
    return c;
  }

  it("writes canonically rounded values and clears cleanly", () => {
    useProjectStore.getState().setSegmentTempoDetect(T, S, {
      bpm: 123.456789, anchorMs: 250.123456, downbeat: 2, conf: 0.876543,
    });
    const td1 = clip().tempoDetect!;
    expect(td1.bpm).toBe(123.457);
    expect(td1.anchorMs).toBe(250.12);
    expect(td1.conf).toBe(0.877);
    expect("notConstant" in td1).toBe(false); // false → key omitted (byte-stable serialize)
    expect(useProjectStore.getState().dirty).toBe(true);

    useProjectStore.getState().setSegmentTempoDetect(T, S, undefined);
    expect("tempoDetect" in clip()).toBe(false); // delete-when-cleared, not undefined-valued
  });

  it("contentSig folds absent stretch/tempoDetect to the pre-S59 identity", () => {
    const before = contentSig(clip());
    expect(before.endsWith(":1::")).toBe(true); // stretch=1, no tempoDetect, no paramCurves
    useProjectStore.getState().setSegmentTempoDetect(T, S, { bpm: 120, anchorMs: 0, downbeat: 0, conf: 0.5 });
    expect(contentSig(clip())).not.toBe(before);
    useProjectStore.getState().setSegmentTempoDetect(T, S, undefined);
    expect(contentSig(clip())).toBe(before); // clear returns to the exact original identity
  });

  it("audio loudness lane goes through the shared paramCurves funnel", () => {
    useProjectStore.getState().setSegmentParamCurve(T, S, "loudness", { xs: [0, 960], ys: [0, -6] });
    const pc = clip().paramCurves;
    expect(pc?.loudness?.xs).toEqual([0, 960]);
    useProjectStore.getState().setSegmentParamCurve(T, S, "loudness", undefined);
    expect("paramCurves" in clip()).toBe(false);
  });
});

describe("setSegmentStretch (S59 Tempo Slider)", () => {
  beforeEach(() => {
    useProjectStore.setState({ tracks: [audioTrack(audioSeg())], tempo: 120, dirty: false });
  });

  function seg() {
    return useProjectStore.getState().tracks[0]!.segments[0]!;
  }
  function clip() {
    const c = seg().content;
    if (c.type !== "audioClip") throw new Error("not audioClip");
    return c;
  }

  it("rescales durationTicks around a FIXED source window; r=1 deletes the field", () => {
    // 1920 ticks @120 BPM = 2000 ms played = 2000 ms source (r=1)
    useProjectStore.getState().setSegmentStretch(T, S, 1.25);
    expect(seg().durationTicks).toBe(2400); // same 2000 ms source × 1.25 played
    expect(clip().stretch).toBe(1.25);
    expect(clip().offsetMs).toBe(0); // source window untouched

    useProjectStore.getState().setSegmentStretch(T, S, 1);
    expect(seg().durationTicks).toBe(1920); // round-trips back
    expect("stretch" in clip()).toBe(false); // omitted at 1 (old projects byte-stable)
  });

  it("split of a stretched clip advances offsetMs in SOURCE ms (÷r), and both halves keep r", () => {
    useProjectStore.getState().setSegmentStretch(T, S, 1.25); // box now 2400 ticks / 2000 ms source
    const rightId = useProjectStore.getState().splitSegment(T, S, 1200); // 1200 ticks = 1250 played ms = 1000 source ms
    expect(rightId).toBeTruthy();
    const [left, right] = useProjectStore.getState().tracks[0]!.segments;
    const lc = left!.content, rc = right!.content;
    if (lc.type !== "audioClip" || rc.type !== "audioClip") throw new Error("not audioClip");
    expect(rc.offsetMs).toBeCloseTo(1000, 6); // NOT 1250 (the naive played-ms advance)
    expect(lc.stretch).toBe(1.25);
    expect(rc.stretch).toBe(1.25);
  });
});
