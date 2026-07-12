// S61 GATE — arrangement clipboard (copy/cut/paste of segments + tracks), driven headless.
// Verifies: (A) paste lands at the playhead on the selected track with fresh segment/note ids,
// (B) collision → NEW track below (a vocal spill-over inherits the reference singer config),
// (C) single-clip type mismatch = no-op, (D) multi-segment pastes keep relative tick offsets,
// (E) the vocal bake dual-sig rules — same-config paste stays CLEAN (windowSig stamped), a
// differently-configured target reads DIRTY (auto re-render on Play), an unconfigured target
// gets the bake STRIPPED (no false-clean stale stem), (F) cut = copy + delete in one undo step,
// (G) whole-track paste duplicates everything under fresh ids, (H) undo removes the paste cleanly.
import { describe, it, expect, beforeEach, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: () => Promise.resolve() }));
vi.mock("../i18n", () => ({ default: { t: (k: string) => k } }));

import { useProjectStore } from "../store/project";
import { useHistoryStore, installHistory } from "../store/history";
import { useAppStore } from "../store/app";
import { useAudioStore } from "../store/audio";
import { useWorkflowStore } from "../store/workflow";
import { useVoiceModelStore } from "../store/voice-models";
import {
  clearClipboard, clipboardKind, copySelectedSegments, copyTrackToClipboard,
  cutSelectedSegments, pasteClipboard,
} from "./clipboard";
import { isVocalDirty, vocalRenderSig, VOCAL_LANE_ID } from "./vocal/vocalRender";
import type { Note, ProcessedOutput, Segment, Track, VocalTrackParams } from "../types/project";

function audioSeg(id: string, startTick: number, durationTicks = 960): Segment {
  return {
    id, startTick, durationTicks,
    content: { type: "audioClip", sourcePath: `C:/audio/${id}.wav`, offsetMs: 0, totalDurationMs: 2000 },
  };
}
function audioTrack(id: string, segs: Segment[]): Track {
  return {
    id, name: id, trackType: "audio", segments: segs,
    volumeDb: 0, pan: 0, muted: false, solo: false, expanded: false, laneControls: {},
  };
}
const VP: VocalTrackParams = {
  backend: "sovits", speakerId: 49, langId: 2, transpose: 0, formant: 0,
  transition: { durMs: 120, offsetMs: 0, depthCents: 0 }, breathToken: "AP",
} as unknown as VocalTrackParams;

function vocalNote(id: string, tick: number): Note {
  return { id, tick, duration: 240, pitch: 60, lyric: "あ", velocity: 100 };
}
function vocalBake(sig: string): ProcessedOutput {
  return {
    laneId: VOCAL_LANE_ID, laneLabel: "Vocal", group: "Vocal", outputNodeId: VOCAL_LANE_ID,
    audioPath: "C:/cache/seg/v1/vocal.wav", totalDurationMs: 1000, renderedSig: sig,
  };
}
function vocalTrack(id: string, segs: Segment[], voiceModel?: string): Track {
  return {
    id, name: id, trackType: "vocal", segments: segs,
    volumeDb: 0, pan: 0, muted: false, solo: false, expanded: false, laneControls: {},
    ...(voiceModel ? { voiceModel, vocalParams: structuredClone(VP) } : {}),
  };
}
function seed(tracks: Track[], playhead = 0) {
  useProjectStore.setState({
    name: "P", tracks, tempo: 120, timeSignature: [4, 4],
    dirty: false, filePath: null, selectedNotes: [], playheadTick: playhead,
  });
}
function tracks(): Track[] {
  return useProjectStore.getState().tracks;
}
function select(trackId: string, segmentId: string) {
  useAppStore.getState().selectSegment(trackId, segmentId);
}

let uninstall: (() => void) | null = null;
beforeEach(() => {
  clearClipboard();
  useAppStore.setState({ selectedSegment: null, selectedSegments: [], selectedLane: null, activeTrackId: null } as never);
  useAudioStore.setState({ isPlaying: false } as never);
  useWorkflowStore.setState({ executions: {}, nodeStatuses: {}, nodeOutputs: {}, nodeProgress: {}, nodeErrors: {}, renderLinks: {} });
  useVoiceModelStore.setState({ models: { sovits: [{ name: "singerA", path: "C:/m/a.onnx" }, { name: "singerB", path: "C:/m/b.onnx" }], rvc: [] } } as never);
  uninstall?.();
  uninstall = installHistory();
  useHistoryStore.getState().reset();
  useHistoryStore.getState().markSaved();
});

describe("S61 clipboard — audio segments", () => {
  it("A: pastes at the playhead on the selected track with a fresh id (free space)", () => {
    const t = audioTrack("A", [audioSeg("s1", 0)]);
    seed([t], 2000);
    select("A", "s1");
    expect(copySelectedSegments()).toBe(1);
    expect(pasteClipboard()).toBe("ok");
    const segs = tracks()[0]!.segments;
    expect(segs.length).toBe(2);
    const pasted = segs[1]!;
    expect(pasted.startTick).toBe(2000);
    expect(pasted.id).not.toBe("s1");
    expect(tracks().length).toBe(1);
    // paste selected its output
    expect(useAppStore.getState().selectedSegment?.segmentId).toBe(pasted.id);
  });

  it("B: collision → new track of the same type right below", () => {
    const t = audioTrack("A", [audioSeg("s1", 0)]);
    seed([t, audioTrack("Z", [])], 100); // playhead inside s1 → collides
    select("A", "s1");
    copySelectedSegments();
    expect(pasteClipboard()).toBe("ok");
    expect(tracks().length).toBe(3);
    const nt = tracks()[1]!; // inserted right below A
    expect(nt.trackType).toBe("audio");
    expect(nt.segments.length).toBe(1);
    expect(nt.segments[0]!.startTick).toBe(100);
    expect(tracks()[2]!.id).toBe("Z");
  });

  it("C: single-clip paste onto a mismatching track type is a no-op", () => {
    const a = audioTrack("A", [audioSeg("s1", 0)]);
    const v = vocalTrack("V", [], "singerA");
    seed([a, v], 0);
    select("A", "s1");
    copySelectedSegments();
    select("V", "nonexistent"); // selects track V as the target anchor
    useAppStore.setState({ selectedSegment: { trackId: "V", segmentId: "x" }, selectedSegments: [] } as never);
    expect(pasteClipboard()).toBe("typeMismatch");
    expect(tracks().length).toBe(2);
    expect(tracks()[1]!.segments.length).toBe(0);
  });

  it("D: multi-segment paste keeps relative offsets (anchor = earliest start)", () => {
    const t = audioTrack("A", [audioSeg("s1", 480, 240), audioSeg("s2", 1200, 240)]);
    seed([t], 4800);
    useAppStore.getState().selectSegments([
      { trackId: "A", segmentId: "s1" },
      { trackId: "A", segmentId: "s2" },
    ]);
    expect(copySelectedSegments()).toBe(2);
    expect(pasteClipboard()).toBe("ok");
    const segs = tracks()[0]!.segments;
    expect(segs.length).toBe(4);
    const starts = segs.slice(2).map((s) => s.startTick).sort((x, y) => x - y);
    expect(starts).toEqual([4800, 4800 + (1200 - 480)]);
  });

  it("F: cut removes the originals and pastes them back elsewhere", () => {
    const t = audioTrack("A", [audioSeg("s1", 0)]);
    seed([t], 5000);
    select("A", "s1");
    expect(cutSelectedSegments()).toBe(1);
    expect(tracks()[0]!.segments.length).toBe(0);
    useAppStore.getState().setActiveTrack("A");
    expect(pasteClipboard()).toBe("ok");
    expect(tracks()[0]!.segments.length).toBe(1);
    expect(tracks()[0]!.segments[0]!.startTick).toBe(5000);
  });

  it("F2: cut deletes ONLY what it copied — a loading clip in the selection survives", () => {
    const t = audioTrack("A", [audioSeg("s1", 0), { ...audioSeg("s2", 2000), loading: true }]);
    seed([t], 5000);
    useAppStore.getState().selectSegments([
      { trackId: "A", segmentId: "s1" },
      { trackId: "A", segmentId: "s2" },
    ]);
    expect(cutSelectedSegments()).toBe(1); // only s1 copied
    const ids = tracks()[0]!.segments.map((s) => s.id);
    expect(ids).toEqual(["s2"]); // the mid-decode clip was NOT deleted
  });

  it("H: undo removes the pasted segment cleanly (one step)", () => {
    const t = audioTrack("A", [audioSeg("s1", 0)]);
    seed([t], 3000);
    select("A", "s1");
    copySelectedSegments();
    pasteClipboard();
    expect(tracks()[0]!.segments.length).toBe(2);
    useHistoryStore.getState().undo();
    expect(tracks()[0]!.segments.length).toBe(1);
    expect(tracks()[0]!.segments[0]!.id).toBe("s1");
  });
});

describe("S61 clipboard — vocal bake dual-sig", () => {
  function bakedVocal(id: string, voiceModel: string): Track {
    const seg: Segment = {
      id: `${id}-seg`, startTick: 0, durationTicks: 960,
      content: { type: "notes", notes: [vocalNote(`${id}-n1`, 0)] },
    };
    const tr = vocalTrack(id, [seg], voiceModel);
    // stamp a CLEAN bake (renderedSig = current sig)
    const sig = vocalRenderSig(tr, seg, 120);
    seg.processedOutputs = [vocalBake(sig)];
    return tr;
  }

  it("E1: same-config paste keeps the bake CLEAN (windowSig stamped for fresh note ids)", () => {
    const src = bakedVocal("V1", "singerA");
    const dst = vocalTrack("V2", [], "singerA");
    seed([src, dst], 2000);
    expect(isVocalDirty(tracks()[0]!, tracks()[0]!.segments[0]!, 120)).toBe(false);
    select("V1", "V1-seg");
    copySelectedSegments();
    useAppStore.setState({ selectedSegment: { trackId: "V2", segmentId: "x" }, selectedSegments: [] } as never);
    expect(pasteClipboard()).toBe("ok");
    const pasted = tracks()[1]!.segments[0]!;
    expect(pasted.processedOutputs?.some((o) => o.laneId === VOCAL_LANE_ID)).toBe(true);
    // fresh note ids → renderedSig can't match; windowSig must carry validity
    expect(isVocalDirty(tracks()[1]!, pasted, 120)).toBe(false);
  });

  it("E2: different-singer paste reads DIRTY (auto re-render on Play)", () => {
    const src = bakedVocal("V1", "singerA");
    const dst = vocalTrack("V2", [], "singerB");
    seed([src, dst], 2000);
    select("V1", "V1-seg");
    copySelectedSegments();
    useAppStore.setState({ selectedSegment: { trackId: "V2", segmentId: "x" }, selectedSegments: [] } as never);
    expect(pasteClipboard()).toBe("ok");
    const pasted = tracks()[1]!.segments[0]!;
    expect(pasted.processedOutputs?.some((o) => o.laneId === VOCAL_LANE_ID)).toBe(true);
    expect(isVocalDirty(tracks()[1]!, pasted, 120)).toBe(true);
  });

  it("E3: unconfigured target strips the bake (no false-clean stale stem)", () => {
    const src = bakedVocal("V1", "singerA");
    const dst = vocalTrack("V2", []); // no singer
    seed([src, dst], 2000);
    select("V1", "V1-seg");
    copySelectedSegments();
    useAppStore.setState({ selectedSegment: { trackId: "V2", segmentId: "x" }, selectedSegments: [] } as never);
    expect(pasteClipboard()).toBe("ok");
    const pasted = tracks()[1]!.segments[0]!;
    expect(pasted.processedOutputs?.some((o) => o.laneId === VOCAL_LANE_ID) ?? false).toBe(false);
  });

  it("E4: a DIRTY source bake never pastes clean, even same-config", () => {
    const src = bakedVocal("V1", "singerA");
    // dirty the source: change its renderedSig away from current content
    src.segments[0]!.processedOutputs![0]!.renderedSig = "stale";
    const dst = vocalTrack("V2", [], "singerA");
    seed([src, dst], 2000);
    expect(isVocalDirty(tracks()[0]!, tracks()[0]!.segments[0]!, 120)).toBe(true);
    select("V1", "V1-seg");
    copySelectedSegments();
    useAppStore.setState({ selectedSegment: { trackId: "V2", segmentId: "x" }, selectedSegments: [] } as never);
    expect(pasteClipboard()).toBe("ok");
    expect(isVocalDirty(tracks()[1]!, tracks()[1]!.segments[0]!, 120)).toBe(true);
  });

  it("B-vocal: collision spill-over inherits the reference track's singer config", () => {
    const src = bakedVocal("V1", "singerA");
    seed([src], 0); // playhead 0 collides with the source seg itself
    select("V1", "V1-seg");
    copySelectedSegments();
    expect(pasteClipboard()).toBe("ok");
    expect(tracks().length).toBe(2);
    const nt = tracks()[1]!;
    expect(nt.trackType).toBe("vocal");
    expect(nt.voiceModel).toBe("singerA");
    // inherited config = same vocalTrackSig → the carried bake stays clean
    expect(isVocalDirty(nt, nt.segments[0]!, 120)).toBe(false);
  });
});

describe("S61 clipboard — whole track", () => {
  it("G: track paste duplicates segments/config under fresh ids, inserted below the anchor", () => {
    const t = audioTrack("A", [audioSeg("s1", 0), audioSeg("s2", 2000)]);
    t.volumeDb = -6;
    seed([t, audioTrack("B", [])]);
    expect(copyTrackToClipboard("A")).toBe(true);
    expect(clipboardKind()).toBe("track");
    useAppStore.getState().setActiveTrack("A");
    expect(pasteClipboard()).toBe("ok");
    expect(tracks().length).toBe(3);
    const copy = tracks()[1]!;
    expect(copy.id).not.toBe("A");
    expect(copy.name).toContain("A");
    expect(copy.volumeDb).toBe(-6);
    expect(copy.segments.length).toBe(2);
    expect(copy.segments.map((s) => s.id)).not.toContain("s1");
    expect(tracks()[2]!.id).toBe("B");
  });

  it("G2: render cache snapshot installs under the new segment id", () => {
    const t = audioTrack("A", [audioSeg("s1", 0)]);
    seed([t]);
    useWorkflowStore.getState().setNodeOutputs("s1", "node1", ["C:/cache/s1/r1/out.wav"]);
    useWorkflowStore.getState().completeExecution("s1");
    select("A", "s1");
    copySelectedSegments();
    useProjectStore.setState({ playheadTick: 5000 });
    expect(pasteClipboard()).toBe("ok");
    const pasted = tracks()[0]!.segments[1]!;
    const outs = useWorkflowStore.getState().nodeOutputs[pasted.id];
    expect(outs?.["node1"]).toEqual(["C:/cache/s1/r1/out.wav"]);
    expect(useWorkflowStore.getState().executions[pasted.id]?.status).toBe("completed");
  });
});
