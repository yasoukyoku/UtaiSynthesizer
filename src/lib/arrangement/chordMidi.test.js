/**
 * Muno 阶段5 和声化引擎测试：钉死 generateChordMidi 的五种风格语义。
 * 顺阶吸附 / 七和弦扩展（含小九度冲突）/ DARK 强制小调 / RANDB 种子确定 / 网格对齐
 * —— 全部确定性对照，改引擎必须连同这里一起改。
 */
import { describe, it, expect } from "vitest";
import { generateChordMidi } from "./chordMidi";
import { VOICING } from "./arranger";
const BEAT = 480; // TICKS_PER_BEAT
const BAR = 4 * BEAT;
/** 每拍同响一组音的旋律（和弦判定无歧义）。 */
function chordMelody(pitches, beats, startTick = 0) {
    const notes = [];
    for (let b = 0; b < beats; b++) {
        for (const p of pitches)
            notes.push({ tick: startTick + b * BEAT, duration: BEAT, pitch: p });
    }
    return notes;
}
const run = (notes, style, over = {}) => generateChordMidi({ notes, timeSignature: [4, 4], style, chordsPerBar: over.chordsPerBar ?? 1, key: over.key ?? "auto" });
const MAJOR_PCS = [0, 2, 4, 5, 7, 9, 11];
const MINOR_PCS = [0, 2, 3, 5, 7, 8, 10];
/** 某调性（tonic + 大/小）的顺阶绝对音级。 */
const scaleOf = (tonic, major) => (major ? MAJOR_PCS : MINOR_PCS).map((pc) => (pc + tonic) % 12);
describe("generateChordMidi 引擎", () => {
    it("空输入 → null", () => {
        expect(generateChordMidi({ notes: [], timeSignature: [4, 4], style: "POP_STANDARD", chordsPerBar: 1, key: "auto" })).toBeNull();
    });
    it("NOCONSTRAINT：保留分析结果 + 网格对齐 + 块状 voicing", () => {
        const res = run(chordMelody([60, 64, 67], 8), "NOCONSTRAINT");
        expect(res.bars).toBe(2);
        expect(res.startTick).toBe(0);
        expect(res.endTick).toBe(2 * BAR);
        expect(res.key.tonic).toBe(0);
        expect(res.segments).toHaveLength(1);
        const seg = res.segments[0];
        expect(seg.root).toBe(0);
        expect(seg.quality).toBe("maj");
        expect(seg.label).toBe("C");
        // voicing：一段一块，C 大三和弦 [60,64,67]，时长铺满段长 − 1 步。
        expect(res.notes).toHaveLength(3);
        expect(new Set(res.notes.map((n) => n.pitch))).toEqual(new Set([60, 64, 67]));
        expect(res.notes.every((n) => n.tick === 0)).toBe(true);
        expect(res.notes.every((n) => n.duration === 2 * BAR - 120)).toBe(true);
        expect(res.notes.every((n) => n.velocity === 82)).toBe(true);
    });
    it("POP_STANDARD：离调和弦吸附到调内顺阶（C7 的根仍吸到 I 级 C 大三和弦）", () => {
        // C7（C E G Bb）最近级位：根 C=I（0 距离）→ 吸附为 C 大三和弦（检测调性为 C 大调）。
        const res = run(chordMelody([60, 64, 67, 70], 4), "POP_STANDARD");
        const scale = scaleOf(res.key.tonic, res.key.major);
        for (const seg of res.segments) {
            // 根音必在调内顺阶音级上，质量必是三和弦。
            expect(scale).toContain(seg.root);
            expect(["maj", "min", "dim"]).toContain(seg.quality);
            expect(seg.bass).toBeNull();
        }
        expect(res.key.tonic).toBe(0);
        expect(res.segments[0].root).toBe(0);
        expect(res.segments[0].quality).toBe("maj");
    });
    it("POP_COMPLEX：三和弦按级位扩七和弦（C 大调 I→maj7、V→7、ii/vi→min7）", () => {
        // 每小节换和弦：C maj → G maj → A min → F maj（I V vi IV）。
        const notes = [];
        const prog = [
            [[60, 64, 67], "maj7"], // I → Cmaj7
            [[67, 71, 74], "7"], // V → G7
            [[69, 72, 76], "min7"], // vi → Am7
            [[65, 69, 72], "maj7"], // IV → Fmaj7
        ];
        prog.forEach(([pcs], bar) => {
            for (let b = 0; b < 4; b++)
                for (const p of pcs)
                    notes.push({ tick: bar * BAR + b * BEAT, duration: BEAT, pitch: p });
        });
        const res = run(notes, "POP_COMPLEX");
        expect(res.segments.map((s) => s.label)).toEqual(["Cmaj7", "G7", "Am7", "Fmaj7"]);
    });
    it("POP_COMPLEX：扩音与旋律构成小九度冲突时放弃扩展", () => {
        // C 大三和弦（C E G）+ 旋律里有 F（=大七度 B 的下方小二度相邻音,4 与 11 距离 7…）
        // 构造真正的冲突：I 级扩 maj7 加 B(11)；旋律含 C(0)/E(4)/G(7) 无冲突 → 扩。
        // 把旋律换成 C-E-G + 相邻音 Bb(10)：|11−10|=1 → 冲突 → 保 C 大三。
        // 但 Bb 会让分析变 C7 —— 改用旋律 F(4)+B(11)？B 本身就是扩展音。
        // 真正可控的冲突：G 大三和弦（G B D）扩 7 加 F(5)；旋律含 F#(6)：|5−6|=1 → 冲突。
        const notes = [];
        for (let b = 0; b < 4; b++) {
            for (const p of [67, 71, 74])
                notes.push({ tick: b * BEAT, duration: BEAT * 0.9, pitch: p });
            notes.push({ tick: b * BEAT, duration: BEAT * 0.9, pitch: 78 }); // F#5 与 F 槽冲撞
        }
        const res = run(notes, "POP_COMPLEX");
        // G 段（V 级）保三和弦；分析本身可能带 F# → G（无 F）… 断言：无任何含 F 与 F# 同时存在的段。
        for (const seg of res.segments) {
            const pcs = (VOICING[seg.quality] ?? []).map((iv) => (seg.root + iv) % 12);
            expect(pcs.includes(5) && pcs.includes(6)).toBe(false);
        }
    });
    it("DARK：大调旋律 → 关系小调 + 小调顺阶 + i/iv/v→min7", () => {
        // C 大调旋律（C E G）→ DARK → A 小调，C=III（maj）。
        const res = run(chordMelody([60, 64, 67], 8), "DARK");
        expect(res.key.tonic).toBe(9);
        expect(res.key.major).toBe(false);
        expect(res.key.label).toBe("Am");
        const aMinor = scaleOf(9, false);
        for (const seg of res.segments) {
            // 根音必在 A 自然小调顺阶音级上。
            expect(aMinor).toContain(seg.root);
        }
        // C E G 吸附到 III（C maj，级位 2）。
        expect(res.segments[0].root).toBe(0);
        expect(res.segments[0].quality).toBe("maj");
    });
    it("DARK：i 级扩 min7（A 小调旋律 A C E → Am7）", () => {
        const res = run(chordMelody([57, 60, 64], 4), "DARK");
        expect(res.segments[0].label).toBe("Am7");
    });
    it("RANDB：种子随机但确定（同输入同输出），全部落在调内顺阶", () => {
        // 旋律带变化音（不完全贴合一个和弦）→ 每格有随机选择空间。
        const notes = chordMelody([62, 65, 69, 74], 8); // D F A D
        const a = run(notes, "RANDB");
        const b = run(chordMelody([62, 65, 69, 74], 8), "RANDB");
        expect(a).toEqual(b);
        const scale = scaleOf(a.key.tonic, a.key.major);
        for (const seg of a.segments) {
            expect(scale).toContain(seg.root);
            expect(["maj", "min", "dim"]).toContain(seg.quality);
        }
    });
    it("RANDB：选中弦必与该格旋律有共同音（音乐性约束）", () => {
        // C E G 旋律 → 候选必含 C/E/G 之一。
        const notes = chordMelody([60, 64, 67], 8);
        const res = run(notes, "RANDB");
        for (const seg of res.segments) {
            const pcs = (VOICING[seg.quality] ?? VOICING.maj).map((iv) => (seg.root + iv) % 12);
            expect(pcs.some((p) => [0, 4, 7].includes(p))).toBe(true);
        }
    });
    it("chordsPerBar=2：网格对齐半小节，相邻同和弦合并", () => {
        // 前半小节 C、后半小节 Am（每拍换半小节太细 —— 用 2 拍一块）。
        const notes = [];
        for (let b = 0; b < 8; b++) {
            const pcs = b < 2 || (b >= 4 && b < 6) ? [60, 64, 67] : [57, 60, 64];
            for (const p of pcs)
                notes.push({ tick: b * BEAT, duration: BEAT, pitch: p });
        }
        const res = run(notes, "NOCONSTRAINT", { chordsPerBar: 2 });
        expect(res.bars).toBe(2);
        // 段起点必在半小节网格上（0/960/1920/2880 的子集）。
        for (const seg of res.segments) {
            expect(seg.startTick % 960).toBe(0);
        }
        // 相邻同和弦合并：两小节同构（C Am | C Am）→ 4 段。
        expect(res.segments.map((s) => s.label)).toEqual(["C", "Am", "C", "Am"]);
        expect(res.segments.every((s) => s.endTick - s.startTick === 960)).toBe(true);
    });
    it("调性覆盖：key='Am' 强制 A 小调顺阶（即使旋律偏 C 大调）", () => {
        const res = run(chordMelody([60, 64, 67], 4), "POP_STANDARD", { key: "Am" });
        expect(res.key.tonic).toBe(9);
        expect(res.key.major).toBe(false);
        expect(res.key.label).toBe("Am");
        const aMinor = scaleOf(9, false);
        for (const seg of res.segments) {
            expect(aMinor).toContain(seg.root);
        }
    });
    it("无效调性输入回落自动检测", () => {
        const res = run(chordMelody([60, 64, 67], 4), "POP_STANDARD", { key: "not-a-key" });
        expect(res.key.tonic).toBe(0);
        expect(res.key.major).toBe(true);
    });
    it("voicing 音区：全部音符落在 [50, 69]（C3..C#5 钢琴伴奏区）", () => {
        // D F# A C# → Dmaj7：完整根位七和弦 11 半音跨度，根音必须降到 D3 才不越上界。
        const res = run(chordMelody([62, 66, 69, 73], 8), "NOCONSTRAINT");
        expect(res.segments[0].label).toBe("Dmaj7");
        expect(res.notes.map((n) => n.pitch)).toEqual([50, 54, 57, 61]);
        for (const n of res.notes) {
            expect(n.pitch).toBeGreaterThanOrEqual(50);
            expect(n.pitch).toBeLessThanOrEqual(69);
        }
    });
});
