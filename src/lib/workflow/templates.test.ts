import { describe, it, expect } from "vitest";
import {
  buildWorkflow,
  buildTranscribeWorkflow,
  buildCloneWorkflow,
  buildOriginalWorkflow,
  resolveVocalPorts,
  stemCount,
  templateNeedsSeparation,
  templateNeedsVoiceModel,
  templateStemCount,
  type WizardTemplateContext,
} from "./templates";
import { parseWorkflowGraph } from "./graph";

const CTX: WizardTemplateContext = {
  vocalStemNames: ["Vocals", "Instrumental"],
  separationModelFile: "mel_band_roformer_vocals.ckpt",
  voiceModel: { name: "我的声音", path: "C:/models/my_voice.onnx" },
  transposeSemitones: 2,
  amtBackend: "muscriptor",
};

describe("resolveVocalPorts", () => {
  it("标准 Vocals/Instrumental → 0/1", () => {
    expect(resolveVocalPorts(["Vocals", "Instrumental"])).toEqual({ vocalPort: 0, instPort: 1 });
  });

  it("伴奏在前 → 1/0", () => {
    expect(resolveVocalPorts(["Instrumental", "Vocals"])).toEqual({ vocalPort: 1, instPort: 0 });
  });

  it("大小写不敏感 + no_vocal 变体", () => {
    expect(resolveVocalPorts(["vocals", "no_vocals"])).toEqual({ vocalPort: 0, instPort: 1 });
    expect(resolveVocalPorts(["karaoke", "VOICE"])).toEqual({ vocalPort: 1, instPort: 0 });
  });

  it("空/未知名 → 兜底 0/1", () => {
    expect(resolveVocalPorts([])).toEqual({ vocalPort: 0, instPort: 1 });
    expect(resolveVocalPorts(undefined)).toEqual({ vocalPort: 0, instPort: 1 });
    expect(resolveVocalPorts(["Bass", "Drums"])).toEqual({ vocalPort: 0, instPort: 1 });
  });
});

describe("buildWorkflow", () => {
  it("transcribe：input → amtMidi → output，muscriptor 后端", () => {
    const wf = buildWorkflow("transcribe", CTX);
    expect(wf.nodes.map((n) => n.nodeType)).toEqual(["input", "amtMidi", "output"]);
    expect(wf.connections).toEqual([
      { fromNode: "input", fromPort: 0, toNode: "amt", toPort: 0 },
      { fromNode: "amt", fromPort: 0, toNode: "out", toPort: 0 },
    ]);
    const amt = wf.nodes[1]!;
    expect(amt.params.backend).toBe("muscriptor");
    expect(amt.params.midiMode).toBe("smart");
    expect(amt.params.midiTrackMode).toBe("multi_track");
  });

  it("clone：分离→RVC→输出 + 伴奏直出，modelFile 写入分离节点", () => {
    const wf = buildWorkflow("clone", CTX);
    const types = wf.nodes.map((n) => n.nodeType).sort();
    expect(types).toEqual(["input", "msstSeparation", "output", "rvc"]);
    const sep = wf.nodes.find((n) => n.nodeType === "msstSeparation")!;
    expect(sep.params.modelFile).toBe("mel_band_roformer_vocals.ckpt");
    expect(sep.params.stemLabels).toEqual(["Vocals", "Instrumental"]);
    const rvc = wf.nodes.find((n) => n.nodeType === "rvc")!;
    expect(rvc.params.voiceName).toBe("我的声音");
    expect(rvc.params.modelPath).toBe("C:/models/my_voice.onnx");
    // 人声（端口 0）→ RVC，伴奏（端口 1）→ 输出
    expect(wf.connections).toContainEqual({ fromNode: "sep", fromPort: 0, toNode: "rvc", toPort: 0 });
    expect(wf.connections).toContainEqual({ fromNode: "sep", fromPort: 1, toNode: "out", toPort: 1 });
    expect(wf.connections).toContainEqual({ fromNode: "rvc", fromPort: 0, toNode: "out", toPort: 0 });
  });

  it("clone：无声音模型时 RVC 节点仍在（面板挂载后自选第一个）", () => {
    const wf = buildWorkflow("clone", { vocalStemNames: CTX.vocalStemNames });
    const rvc = wf.nodes.find((n) => n.nodeType === "rvc")!;
    expect(rvc.params).toEqual({});
  });

  it("original：6 节点全图（含移调 + 平行 AMT），移调默认 +2", () => {
    const wf = buildWorkflow("original", CTX);
    expect(wf.nodes.length).toBe(6);
    const tp = wf.nodes.find((n) => n.nodeType === "transpose")!;
    expect(tp.params.semitones).toBe(2);
    // 平行扒带：input 直连 amt
    expect(wf.connections).toContainEqual({ fromNode: "input", fromPort: 0, toNode: "amt", toPort: 0 });
    // 伴奏 → 移调 → 输出
    expect(wf.connections).toContainEqual({ fromNode: "sep", fromPort: 1, toNode: "transpose", toPort: 0 });
    expect(wf.connections).toContainEqual({ fromNode: "transpose", fromPort: 0, toNode: "out", toPort: 1 });
  });

  it("端口跟随真实 stem 顺序（Inst 在前时人声走端口 1）", () => {
    const wf = buildWorkflow("clone", { ...CTX, vocalStemNames: ["Instrumental", "Vocals"] });
    expect(wf.connections).toContainEqual({ fromNode: "sep", fromPort: 1, toNode: "rvc", toPort: 0 });
    expect(wf.connections).toContainEqual({ fromNode: "sep", fromPort: 0, toNode: "out", toPort: 1 });
  });

  it("三个模板都能被引擎的图解析器接受（无环/连通）", () => {
    for (const t of ["transcribe", "clone", "original"] as const) {
      const wf = buildWorkflow(t, CTX);
      const g = parseWorkflowGraph(wf); // throws on cycle/broken graph
      expect(g.sorted.length).toBe(wf.nodes.length);
    }
  });

  it("P2-14 歌曲模板：结构正确且能被图解析器接受", () => {
    // 词曲一键成歌：歌词源/风格源 → songGen → 输出
    const l2s = buildWorkflow("lyrics2song", {});
    expect(l2s.nodes.filter((n) => n.nodeType !== "input").map((n) => n.nodeType))
      .toEqual(["songLyrics", "songPrompt", "songGen", "output"]);
    expect(l2s.connections).toContainEqual({ fromNode: "lyr", fromPort: 0, toNode: "gen", toPort: 0 });
    expect(l2s.connections).toContainEqual({ fromNode: "sty", fromPort: 0, toNode: "gen", toPort: 1 });
    // 片段重绘：input → songRepaint ← songPrompt → 输出
    const sr = buildWorkflow("segRepaint", {});
    expect(sr.connections).toContainEqual({ fromNode: "input", fromPort: 0, toNode: "rep", toPort: 0 });
    expect(sr.connections).toContainEqual({ fromNode: "rep", fromPort: 0, toNode: "out", toPort: 0 });
    // 人声转伴奏：songComplete 混音 + 3 声部分轨 → 输出 4 口
    const v2a = buildWorkflow("vocal2acc", {});
    expect(v2a.connections.filter((c) => c.fromNode === "comp").map((c) => c.fromPort)).toEqual([0, 1, 2, 3]);
    // 分轨重建：songStems 四轨 → 输出 4 口
    const st = buildWorkflow("stemsRebuild", {});
    expect(st.nodes.find((n) => n.nodeType === "songStems")).toBeTruthy();
    expect(st.connections.filter((c) => c.fromNode === "st").map((c) => c.fromPort)).toEqual([0, 1, 2, 3]);
    // 四个模板都能被图解析器接受（无环/连通）
    for (const t of ["lyrics2song", "segRepaint", "vocal2acc", "stemsRebuild"] as const) {
      const wf = buildWorkflow(t, {});
      const g = parseWorkflowGraph(wf);
      expect(g.sorted.length).toBe(wf.nodes.length);
    }
  });
});

describe("template helpers", () => {
  it("needsVoiceModel / needsSeparation", () => {
    expect(templateNeedsVoiceModel("transcribe")).toBe(false);
    expect(templateNeedsVoiceModel("clone")).toBe(true);
    expect(templateNeedsVoiceModel("original")).toBe(true);
    expect(templateNeedsSeparation("transcribe")).toBe(false);
    expect(templateNeedsSeparation("original")).toBe(true);
  });

  it("stemCount / templateStemCount", () => {
    expect(stemCount(undefined)).toBe(2);
    expect(stemCount(["a"])).toBe(2); // 兜底至少 2
    expect(stemCount(["a", "b", "c"])).toBe(3);
    expect(templateStemCount("transcribe", CTX)).toBe(0);
    expect(templateStemCount("original", { vocalStemNames: ["Vocals", "Bass", "Drums", "Other"] })).toBe(4);
  });

  it("默认上下文（全部缺省）也不抛错", () => {
    expect(() => buildTranscribeWorkflow({})).not.toThrow();
    expect(() => buildCloneWorkflow({})).not.toThrow();
    expect(() => buildOriginalWorkflow({})).not.toThrow();
    const wf = buildOriginalWorkflow({});
    const tp = wf.nodes.find((n) => n.nodeType === "transpose")!;
    expect(tp.params.semitones).toBe(2); // 默认移调
  });
});
