/**
 * 和弦域与结构域核心库单测（规划 2-5 / 2-7 / 2-11）。
 * 覆盖：和弦标签解析往返、和弦重配四策略、段落结构四操作、换气点规划。
 */
import { describe, it, expect } from "vitest";
import { parseChordLabel, chordLabel, isMinorishQuality, chordBlockToSegments, segmentsToChordBlock, } from "./chords";
import { reharmonizeSegments } from "./reharmonize";
import { editStructure } from "./structure";
import { planBreathPoints } from "./breathPlan";
function seg(i, root, quality) {
    return {
        startTick: i * 1920,
        endTick: (i + 1) * 1920,
        root,
        quality,
        bass: null,
        label: chordLabel(root, quality),
    };
}
/** 四小节 I-V-vi-IV 进行（C-G-Am-F）。 */
function fourBars() {
    return [seg(0, 0, "maj"), seg(1, 7, "maj"), seg(2, 9, "min"), seg(3, 5, "maj")];
}
function note(tick, duration, pitch = 72) {
    return { tick, duration, pitch, velocity: 96 };
}
describe("chords：和弦标签解析", () => {
    it("parseChordLabel：常见标签", () => {
        expect(parseChordLabel("Am7")).toEqual({ root: 9, quality: "min7", bass: null });
        expect(parseChordLabel("Cmaj7")).toEqual({ root: 0, quality: "maj7", bass: null });
        expect(parseChordLabel("G/B")).toEqual({ root: 7, quality: "maj", bass: 11 });
        expect(parseChordLabel("F#m7b5")).toEqual({ root: 6, quality: "m7b5", bass: null });
        expect(parseChordLabel("Bb")).toEqual({ root: 10, quality: "maj", bass: null });
        expect(parseChordLabel("C7")).toEqual({ root: 0, quality: "7", bass: null });
    });
    it("chordLabel：反构标签", () => {
        expect(chordLabel(9, "min7")).toBe("Am7");
        expect(chordLabel(6, "m7b5")).toBe("F#m7b5");
        expect(chordLabel(7, "maj", 11)).toBe("G/B");
        expect(chordLabel(0, "maj")).toBe("C");
    });
    it("isMinorishQuality", () => {
        for (const q of ["min", "min7", "m7b5", "dim", "dim7"]) {
            expect(isMinorishQuality(q)).toBe(true);
        }
        for (const q of ["maj", "7", "maj7", "sus4", "6"]) {
            expect(isMinorishQuality(q)).toBe(false);
        }
    });
    it("chordBlockToSegments：每和弦一小节", () => {
        const block = {
            type: "chordBlock",
            chords: [
                { label: "C", root: 0 },
                { label: "Am", root: 9 },
            ],
            bpm: 120,
        };
        const out = chordBlockToSegments(block, 4);
        expect(out).toHaveLength(2);
        expect(out[0].startTick).toBe(0);
        expect(out[0].endTick).toBe(1920);
        expect(out[1].startTick).toBe(1920);
        expect(out[1].endTick).toBe(3840);
        expect(out[1].root).toBe(9);
        expect(out[1].quality).toBe("min");
        expect(out[1].label).toBe("Am");
    });
    it("segmentsToChordBlock：往返保留 root/quality", () => {
        const block = segmentsToChordBlock(fourBars(), 90);
        expect(block.type).toBe("chordBlock");
        expect(block.bpm).toBe(90);
        expect(block.chords).toHaveLength(4);
        const back = chordBlockToSegments(block, 4);
        expect(back.map((s) => s.label)).toEqual(["C", "G", "Am", "F"]);
        expect(back.map((s) => s.root)).toEqual([0, 7, 9, 5]);
        expect(back.map((s) => s.quality)).toEqual(["maj", "maj", "min", "maj"]);
    });
});
describe("reharmonize：和弦重配", () => {
    it("密度 0：完全不变（L0 零损伤）", () => {
        const src = fourBars();
        const out = reharmonizeSegments(src, { strategies: ["tritone", "diatonic", "borrowed", "extension"], density: 0 });
        expect(out.map((s) => s.label)).toEqual(["C", "G", "Am", "F"]);
    });
    it("tritone：C → F#7（属功能替代）", () => {
        const out = reharmonizeSegments([seg(0, 0, "maj")], { strategies: ["tritone"], density: 100 });
        expect(out).toHaveLength(1);
        expect(out[0].root).toBe(6);
        expect(out[0].quality).toBe("7");
        expect(out[0].label).toBe("F#7");
    });
    it("tritone：小和弦不适用（原样返回）", () => {
        const out = reharmonizeSegments([seg(0, 9, "min")], { strategies: ["tritone"], density: 100 });
        expect(out[0].root).toBe(9);
        expect(out[0].quality).toBe("min");
    });
    it("diatonic：C→Am，Am→C（调内相对）", () => {
        const cToAm = reharmonizeSegments([seg(0, 0, "maj")], { strategies: ["diatonic"], density: 100 });
        expect(cToAm[0].root).toBe(9);
        expect(cToAm[0].quality).toBe("min");
        const amToC = reharmonizeSegments([seg(0, 9, "min")], { strategies: ["diatonic"], density: 100 });
        expect(amToC[0].root).toBe(0);
        expect(amToC[0].quality).toBe("maj");
    });
    it("borrowed：同根大小互换", () => {
        const out = reharmonizeSegments([seg(0, 5, "maj")], { strategies: ["borrowed"], density: 100 });
        expect(out[0].root).toBe(5);
        expect(out[0].quality).toBe("min");
    });
    it("extension：C7 → C9（唯一候选）", () => {
        const out = reharmonizeSegments([seg(0, 0, "7")], { strategies: ["extension"], density: 100 });
        expect(out[0].root).toBe(0);
        expect(out[0].quality).toBe("9");
    });
    it("extension：C 的候选 ∈ {maj7, add9, 6}，root 不变", () => {
        const out = reharmonizeSegments([seg(0, 0, "maj")], { strategies: ["extension"], density: 100 });
        expect(out[0].root).toBe(0);
        expect(["maj7", "add9", "6"]).toContain(out[0].quality);
    });
    it("同种子完全可复现", () => {
        const a = reharmonizeSegments(fourBars(), { strategies: ["tritone", "diatonic"], density: 60, seed: 42 });
        const b = reharmonizeSegments(fourBars(), { strategies: ["tritone", "diatonic"], density: 60, seed: 42 });
        expect(JSON.stringify(a)).toBe(JSON.stringify(b));
    });
});
describe("structure：段落结构编辑", () => {
    it("repeatTail：尾段复制追加", () => {
        const out = editStructure(fourBars(), { op: "repeatTail", sectionBars: 1 });
        expect(out).toHaveLength(5);
        expect(out.map((s) => s.startTick)).toEqual([0, 1920, 3840, 5760, 7680]);
        expect(out[4].endTick).toBe(9600);
        expect(out[4].label).toBe(out[3].label);
    });
    it("dropTail：删除尾段", () => {
        const out = editStructure(fourBars(), { op: "dropTail", sectionBars: 1 });
        expect(out).toHaveLength(3);
        expect(out[2].endTick).toBe(5760);
    });
    it("transposeTail：仅尾段移调", () => {
        const out = editStructure(fourBars(), { op: "transposeTail", sectionBars: 1, semitones: 2 });
        expect(out).toHaveLength(4);
        expect(out[0].label).toBe("C");
        expect(out[3].root).toBe(7);
        expect(out[3].label).toBe("G");
    });
    it("lengthenTail：末和弦延长一段", () => {
        const out = editStructure(fourBars(), { op: "lengthenTail", sectionBars: 1 });
        expect(out).toHaveLength(4);
        expect(out[3].endTick).toBe(9600);
    });
    it("空输入不崩溃", () => {
        for (const op of ["repeatTail", "dropTail", "transposeTail", "lengthenTail"]) {
            expect(editStructure([], { op, sectionBars: 1 })).toEqual([]);
        }
    });
});
describe("breathPlan：换气点规划", () => {
    it("间隙驱动：默认 minGap 1 拍，短间隙权重 0.6", () => {
        const notes = [note(0, 480, 72), note(1200, 480, 74), note(2400, 480, 76)];
        const pts = planBreathPoints(notes);
        expect(pts).toHaveLength(2);
        expect(pts[0]).toEqual({ tick: 840, weight: 0.6, reason: "gap" });
        expect(pts[1]).toEqual({ tick: 2040, weight: 0.6, reason: "gap" });
    });
    it("间隙 ≥ 2 拍：权重 1", () => {
        const pts = planBreathPoints([note(0, 480, 72), note(1440, 480, 74)]);
        expect(pts).toHaveLength(1);
        expect(pts[0].weight).toBe(1);
        expect(pts[0].tick).toBe(960);
    });
    it("minGapBeats 阈值：0.5 拍只捕获 ≥240 tick 的间隙", () => {
        const notes = [
            note(0, 480, 72),
            note(480, 240, 74),
            note(960, 240, 76),
            note(1200, 240, 77),
        ];
        const pts = planBreathPoints(notes, { minGapBeats: 0.5 });
        expect(pts).toHaveLength(1);
        expect(pts[0].tick).toBe(840);
    });
    it("无间隙连奏：无换气点", () => {
        const notes = [0, 240, 480, 720].map((t) => note(t, 240, 72));
        expect(planBreathPoints(notes, { minGapBeats: 1 })).toHaveLength(0);
    });
    it("歌词标点：音节数对齐时插入换气点", () => {
        // "你好，世界。" = 4 个演唱音节（标点并入前一音节槽，换气在携带标点的音之后）。
        const notes = [0, 240, 480, 720].map((t) => note(t, 240, 72));
        const pts = planBreathPoints(notes, { lyrics: "你好，世界。" });
        expect(pts).toEqual([
            { tick: 600, weight: 0.7, reason: "punct" },
            { tick: 1080, weight: 1, reason: "punct" },
        ]);
    });
    it("拉丁歌词：单词聚合 + 标点", () => {
        // "Hello, world." = 2 个词槽（Hello→0.7，world→1）。
        const notes = [0, 240].map((t) => note(t, 240, 72));
        const pts = planBreathPoints(notes, { lyrics: "Hello, world." });
        expect(pts).toHaveLength(2);
        expect(pts[0].weight).toBe(0.7);
        expect(pts[1].weight).toBe(1);
    });
    it("音节数错位：放弃标点对齐，不猜", () => {
        const notes = [0, 240, 480].map((t) => note(t, 240, 72));
        const pts = planBreathPoints(notes, { lyrics: "你好，世界。" });
        expect(pts).toHaveLength(0);
    });
});
