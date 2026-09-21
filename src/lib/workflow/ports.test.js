import { describe, it, expect } from "vitest";
import { NODE_PORTS, inputPortKind, outputPortKind, portsCompatible, canConnect, } from "./ports";
describe("portsCompatible", () => {
    it("同型相连放行", () => {
        expect(portsCompatible("audio", "audio")).toBe(true);
        expect(portsCompatible("midi", "midi")).toBe(true);
        expect(portsCompatible("chords", "chords")).toBe(true);
        expect(portsCompatible("lyrics", "lyrics")).toBe(true);
        expect(portsCompatible("report", "report")).toBe(true);
    });
    it("任一端 any 放行（双向）", () => {
        expect(portsCompatible("any", "audio")).toBe(true);
        expect(portsCompatible("audio", "any")).toBe(true);
        expect(portsCompatible("any", "report")).toBe(true);
        expect(portsCompatible("any", "any")).toBe(true);
    });
    it("异型拒绝（audio↔midi、report→audio、chords→audio）", () => {
        expect(portsCompatible("audio", "midi")).toBe(false);
        expect(portsCompatible("midi", "audio")).toBe(false);
        expect(portsCompatible("report", "audio")).toBe(false);
        expect(portsCompatible("chords", "audio")).toBe(false);
        expect(portsCompatible("lyrics", "midi")).toBe(false);
    });
});
describe("inputPortKind / outputPortKind", () => {
    it("界内端口取表值", () => {
        expect(inputPortKind("rvc", 0)).toBe("audio");
        expect(inputPortKind("songGen", 0)).toBe("lyrics");
        expect(inputPortKind("songGen", 2)).toBe("audio");
        expect(outputPortKind("amtMidi", 0)).toBe("midi");
        expect(outputPortKind("amtMidi", 1)).toBe("report");
        expect(outputPortKind("msstSeparation", 4)).toBe("audio");
    });
    it("越界回退 any（动态端口：6-stem 分离、split/merge 扩展）", () => {
        expect(outputPortKind("msstSeparation", 9)).toBe("any");
        expect(outputPortKind("split", 5)).toBe("any");
        expect(inputPortKind("merge", 7)).toBe("any");
        expect(inputPortKind("rvc", 3)).toBe("any");
    });
    it("output 节点入口 any / input 节点出口 audio", () => {
        expect(inputPortKind("output", 0)).toBe("any");
        expect(outputPortKind("input", 0)).toBe("audio");
    });
});
describe("canConnect（规划 6-2 isValidConnection 核心判定）", () => {
    it("常规音频链放行：input→rvc→sovits", () => {
        expect(canConnect("input", 0, "rvc", 0)).toBe(true);
        expect(canConnect("rvc", 0, "sovits", 0)).toBe(true);
        expect(canConnect("sovits", 0, "output", 0)).toBe(true);
    });
    it("哑线拒绝：audio→midi、report→audio、chords→audio", () => {
        expect(canConnect("input", 0, "chordDetect", 0)).toBe(false);
        expect(canConnect("spectrogram", 1, "rvc", 0)).toBe(false);
        expect(canConnect("chordBlockIn", 0, "rvc", 0)).toBe(false);
    });
    it("报告/和弦/歌词旁路端口放行到对应消费方", () => {
        expect(canConnect("spectrogram", 1, "output", 0)).toBe(true); // 报告接 any 入口存档
        expect(canConnect("chordDetect", 0, "melodyGen", 0)).toBe(true); // chords→chords
        expect(canConnect("songLyrics", 0, "songGen", 0)).toBe(true); // lyrics→lyrics
        expect(canConnect("midiFileIn", 0, "harmonizer", 0)).toBe(true); // midi→any
    });
    it("越界动态端口经 any 回退放行", () => {
        expect(canConnect("msstSeparation", 9, "rvc", 0)).toBe(true);
        expect(canConnect("split", 5, "transpose", 0)).toBe(true);
    });
    it("双音频入口节点第二口同型放行", () => {
        expect(canConnect("input", 0, "spectralCompare", 1)).toBe(true);
        expect(canConnect("input", 0, "spectralCompare", 0)).toBe(true);
    });
});
/** soundfontRender 是符号域与声音域之间**唯一**的桥。它的价值全在「进口只收 MIDI、出口
 *  只发 audio」这一条上：进口放宽一格,音频就能绕过渲染直连进来(引擎当场抛 "needs MIDI
 *  input");出口写错一格,它就接不上任何后续音频链,节点等于白做。两个方向都钉住。 */
describe("canConnect：soundfontRender 的符号域↔声音域边界", () => {
    it("MIDI 上游都能进：midiFileIn / amtMidi 的 midi 口 / harmonizer 双出 / melodyGen", () => {
        expect(canConnect("midiFileIn", 0, "soundfontRender", 0)).toBe(true);
        expect(canConnect("amtMidi", 0, "soundfontRender", 0)).toBe(true);
        expect(canConnect("harmonizer", 0, "soundfontRender", 0)).toBe(true);
        expect(canConnect("harmonizer", 1, "soundfontRender", 0)).toBe(true);
        expect(canConnect("melodyGen", 0, "soundfontRender", 0)).toBe(true);
    });
    it("非 MIDI 上游一律拒绝：音频源 / amtMidi 的报告口 / 和弦流", () => {
        expect(canConnect("input", 0, "soundfontRender", 0)).toBe(false);
        expect(canConnect("amtMidi", 1, "soundfontRender", 0)).toBe(false);
        expect(canConnect("chordBlockIn", 0, "soundfontRender", 0)).toBe(false);
    });
    it("出口是货真价实的音频：能进音频处理链和 output,进不了和弦入口", () => {
        expect(canConnect("soundfontRender", 0, "rvc", 0)).toBe(true);
        expect(canConnect("soundfontRender", 0, "output", 0)).toBe(true);
        expect(canConnect("soundfontRender", 0, "melodyGen", 0)).toBe(false);
    });
});
/** melodySimilarity 是原创性闸门,接线口径必须跟 spectralCompare/dtwAlign 一族一致:
 *  A(端口 0)是链路里正在走的那一路并原样透传,B(端口 1)是参考旋律。两个入口都只收 MIDI ——
 *  放宽成 any 的话音频能直连进来,引擎会在 resolveMidiNotes 里才炸,报错点离现场很远。
 *  出口 0 必须还是 midi(否则闸门一插进链路就把下游全断开),出口 1 必须是 report。 */
describe("canConnect：melodySimilarity 的双 MIDI 入口与透传出口", () => {
    it("A/B 两个入口都收 MIDI 上游", () => {
        expect(canConnect("midiFileIn", 0, "melodySimilarity", 0)).toBe(true);
        expect(canConnect("midiFileIn", 0, "melodySimilarity", 1)).toBe(true);
        expect(canConnect("harmonizer", 0, "melodySimilarity", 0)).toBe(true);
        expect(canConnect("amtMidi", 0, "melodySimilarity", 1)).toBe(true);
    });
    it("非 MIDI 上游两个口都拒绝：音频 / 和弦 / 报告", () => {
        expect(canConnect("input", 0, "melodySimilarity", 0)).toBe(false);
        expect(canConnect("input", 0, "melodySimilarity", 1)).toBe(false);
        expect(canConnect("chordBlockIn", 0, "melodySimilarity", 0)).toBe(false);
        expect(canConnect("amtMidi", 1, "melodySimilarity", 1)).toBe(false);
    });
    it("出口 0 是透传的 MIDI：能继续接下游符号域节点和音源渲染", () => {
        expect(canConnect("melodySimilarity", 0, "soundfontRender", 0)).toBe(true);
        expect(canConnect("melodySimilarity", 0, "harmonizer", 0)).toBe(true);
        expect(canConnect("melodySimilarity", 0, "melodySimilarity", 1)).toBe(true);
        expect(canConnect("melodySimilarity", 0, "rvc", 0)).toBe(false);
    });
    it("出口 1 是报告：能存档进 output,进不了 MIDI 入口", () => {
        expect(canConnect("melodySimilarity", 1, "output", 0)).toBe(true);
        expect(canConnect("melodySimilarity", 1, "soundfontRender", 0)).toBe(false);
    });
});
describe("NODE_PORTS 完整性抽样", () => {
    it("msstSeparation 5 出口 / songComplete 4 出口 / merge 4 入口", () => {
        expect(NODE_PORTS.msstSeparation.outputs).toHaveLength(5);
        expect(NODE_PORTS.songComplete.outputs).toHaveLength(4);
        expect(NODE_PORTS.merge.inputs).toHaveLength(4);
    });
    it("I/O 节点为边界：input 无入口、output 无出口", () => {
        expect(NODE_PORTS.input.inputs).toHaveLength(0);
        expect(NODE_PORTS.output.outputs).toHaveLength(0);
    });
    it("soundfontRender 恰好 1 进 1 出（渲染是 N 音符→1 波形，不该有第二个口）", () => {
        expect(NODE_PORTS.soundfontRender.inputs).toEqual(["midi"]);
        expect(NODE_PORTS.soundfontRender.outputs).toEqual(["audio"]);
    });
    it("melodySimilarity 是 2 进 2 出（A/B 双入，透传 + 报告双出）", () => {
        expect(NODE_PORTS.melodySimilarity.inputs).toEqual(["midi", "midi"]);
        expect(NODE_PORTS.melodySimilarity.outputs).toEqual(["midi", "report"]);
    });
});
