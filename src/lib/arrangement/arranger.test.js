/**
 * Muno 阶段4 自动编曲引擎测试：钉死 styles.ts 模板 → arranger.ts 生成的关键骨架。
 * 鼓 GM 音高 / swing 位移 / 情绪密度 / 拍号降级 / 和弦跟随 —— 全部确定性对照，
 * 改模板或引擎必须连带这里一起改，且永远有对照。
 */
import { describe, it, expect } from "vitest";
import { arrange } from "./arranger";
import { STEP_TICKS } from "./styles";
// GM 鼓组音高（与 arranger.ts 一致）。
const KICK = 36, SNARE = 38, HIHAT = 42, CRASH = 49;
const BEAT = 480; // TICKS_PER_BEAT
const BAR = 4 * BEAT; // 4/4 一小节
/** 每拍同时给一组和弦音的旋律（窗口内和弦判定无歧义）。 */
function chordMelody(pitches, beats, startTick = 0) {
    const notes = [];
    for (let b = 0; b < beats; b++) {
        for (const p of pitches)
            notes.push({ tick: startTick + b * BEAT, duration: BEAT, pitch: p });
    }
    return notes;
}
/** 某小节内某轨音符的 (tick 相对小节, pitch) 对，按序。 */
function pairsInBar(notes, bar, barTicks = BAR) {
    return notes
        .filter((n) => n.tick >= bar * barTicks && n.tick < (bar + 1) * barTicks)
        .map((n) => [n.tick - bar * barTicks, n.pitch])
        .sort((a, b) => a[0] - b[0] || a[1] - b[1]);
}
const base = (notes, over = {}) => ({ notes, tempo: 100, timeSignature: [4, 4], style: "pop", mood: "neutral", ...over });
describe("arrange 引擎", () => {
    it("空输入 → null", () => {
        expect(arrange(base([]))).toBeNull();
    });
    it("pop 骨架：鼓/贝斯/钢琴/铺底的逐拍对照（C 大调 2 小节）", () => {
        // 每拍 C-E-G 同响 → 和弦判定 C maj、调性 C。
        const res = arrange(base(chordMelody([60, 64, 67], 8)));
        expect(res.bars).toBe(2);
        expect(res.startTick).toBe(0);
        expect(res.endTick).toBe(2 * BAR);
        expect(res.key.tonic).toBe(0);
        expect(res.chords[0].root).toBe(0);
        expect(res.chords[0].quality).toBe("maj");
        // 鼓：第 1 小节 —— 底鼓 0/8 步、军鼓 4/12、闭镲 8 分、开头吊镲。
        expect(pairsInBar(res.drums, 0)).toEqual([
            [0, KICK], [0, HIHAT], [0, CRASH],
            [240, HIHAT], [480, SNARE], [480, HIHAT], [720, HIHAT],
            [960, KICK], [960, HIHAT], [1200, HIHAT], [1440, SNARE], [1440, HIHAT], [1680, HIHAT],
        ]);
        // 第 2 小节：同骨架但无吊镲。
        const b1 = pairsInBar(res.drums, 1);
        expect(b1).toEqual([
            [0, KICK], [0, HIHAT],
            [240, HIHAT], [480, SNARE], [480, HIHAT], [720, HIHAT],
            [960, KICK], [960, HIHAT], [1200, HIHAT], [1440, SNARE], [1440, HIHAT], [1680, HIHAT],
        ]);
        expect(b1.some(([, p]) => p === CRASH)).toBe(false);
        // 贝斯：每拍根音 C（pitch%12===0，E1..E2 区）。
        expect(res.bass).toHaveLength(8);
        for (const n of res.bass) {
            expect(n.pitch % 12).toBe(0);
            expect(n.pitch).toBeGreaterThanOrEqual(28);
            expect(n.pitch).toBeLessThanOrEqual(45);
        }
        expect(res.bass.map((n) => n.tick)).toEqual([0, 480, 960, 1440, 1920, 2400, 2880, 3360]);
        expect(res.bass.every((n) => n.duration === BEAT)).toBe(true);
        // 钢琴：整块 C大三和弦 [60,64,67]，第 1/3 拍各一下。
        expect(res.piano.map((n) => n.tick)).toEqual([0, 0, 0, 960, 960, 960, 1920, 1920, 1920, 2880, 2880, 2880]);
        expect(new Set(res.piano.map((n) => n.pitch))).toEqual(new Set([60, 64, 67]));
        // 铺底：每小节 3 音、铺满（barTicks - 1 步）。
        expect(res.pad).toHaveLength(6);
        expect(res.pad.filter((n) => n.tick === 0)).toHaveLength(3);
        expect(res.pad.every((n) => n.duration === BAR - STEP_TICKS)).toBe(true);
        // 全轨有序且力度合法。
        for (const track of [res.drums, res.bass, res.piano, res.pad]) {
            const sorted = [...track].sort((a, b) => a.tick - b.tick || a.pitch - b.pitch);
            expect(track).toEqual(sorted);
            for (const n of track) {
                expect(n.velocity).toBeGreaterThanOrEqual(1);
                expect(n.velocity).toBeLessThanOrEqual(127);
            }
        }
    });
    it("同输入同输出（确定性）", () => {
        const notes = chordMelody([62, 65, 69, 74], 12); // D-F-A-D
        const a = arrange(base(notes, { style: "funk", mood: "energetic" }));
        const b = arrange(base(chordMelody([62, 65, 69, 74], 12), { style: "funk", mood: "energetic" }));
        expect(a).toEqual(b);
    });
    it("jazz swing：奇数八分（步%4===2）右移 swing×拍/6", () => {
        const res = arrange(base(chordMelody([60, 64, 67], 4), { style: "jazz" }));
        // swing 0.9 → 位移 = round(0.9 × 480/6) = 72 tick。
        const swing = Math.round(0.9 * (BEAT / 6));
        // 军鼓幽灵音在第 2/6/10/14 步 → 都右移。
        const ghostTicks = res.drums
            .filter((n) => n.pitch === SNARE && n.tick < BAR)
            .map((n) => n.tick);
        expect(ghostTicks).toEqual([2 * STEP_TICKS + swing, 6 * STEP_TICKS + swing, 10 * STEP_TICKS + swing, 14 * STEP_TICKS + swing]);
        // 踏镲 2/4 拍（步 4/12）→ 正拍不位移。
        const hatTicks = res.drums.filter((n) => n.pitch === HIHAT && n.tick < BAR).map((n) => n.tick);
        expect(hatTicks).toEqual([4 * STEP_TICKS, 12 * STEP_TICKS]);
        // 行走贝斯每拍一音、正拍不位移。
        expect(res.bass.filter((n) => n.tick < BAR).map((n) => n.tick)).toEqual([0, BEAT, 2 * BEAT, 3 * BEAT]);
        // 钢琴 Charleston：步 0 正拍 + 步 6 奇数八分（右移）。
        const pT = [...new Set(res.piano.filter((n) => n.tick < BAR).map((n) => n.tick))];
        expect(pT).toEqual([0, 6 * STEP_TICKS + swing]);
    });
    it("sad 情绪（density 0）：闭镲减半拍、力度 -12", () => {
        const res = arrange(base(chordMelody([60, 64, 67], 4), { mood: "sad" }));
        const hats = res.drums.filter((n) => n.pitch === HIHAT && n.tick < BAR);
        // 只留正拍步（%4===0）。
        expect(hats.map((n) => n.tick)).toEqual([0, 4 * STEP_TICKS, 8 * STEP_TICKS, 12 * STEP_TICKS]);
        // 力度 = 74 - 12。
        expect(hats.every((n) => n.velocity === 62)).toBe(true);
    });
    it("energetic 情绪（density 2）：闭镲填满 16 分", () => {
        const res = arrange(base(chordMelody([60, 64, 67], 4), { mood: "energetic" }));
        const hats = res.drums.filter((n) => n.pitch === HIHAT && n.tick < BAR);
        expect(hats).toHaveLength(16);
    });
    it("3/4 拍号安全降级：越出小节的步被丢弃，永不越界", () => {
        const res = arrange(base(chordMelody([60, 64, 67], 6), { style: "edm", timeSignature: [3, 4] }));
        const barTicks = 3 * BEAT;
        expect(res.bars).toBe(2);
        // edm kick [0,4,8,12] → 12 步越界丢弃 → 每小节 3 个。
        for (const bar of [0, 1]) {
            const kicks = pairsInBar(res.drums, bar, barTicks).filter(([, p]) => p === KICK);
            expect(kicks.map(([t]) => t)).toEqual([0, 4 * STEP_TICKS, 8 * STEP_TICKS]);
            // 全部鼓点都在小节内。
            for (const [t] of pairsInBar(res.drums, bar, barTicks))
                expect(t).toBeLessThan(barTicks);
        }
    });
    it("跨度对齐：首音符所在小节起、末音符结束小节止", () => {
        // 从 tick 100 开始的 1 小节旋律 → 对齐为整小节。
        const notes = [];
        for (let b = 0; b < 3; b++)
            for (const p of [60, 64, 67])
                notes.push({ tick: 100 + b * BEAT, duration: 400, pitch: p });
        const res = arrange(base(notes));
        expect(res.startTick).toBe(0);
        expect(res.endTick).toBe(BAR);
        expect(res.bars).toBe(1);
    });
    it("贝斯跟随和弦：A 小三和弦 → 根音 A、钢琴 voicing [57,60,64]", () => {
        const res = arrange(base(chordMelody([57, 60, 64], 4)));
        expect(res.chords[0].root).toBe(9);
        expect(res.chords[0].quality).toBe("min");
        // 贝斯根音 = A1(33)。
        expect(res.bass[0].pitch % 12).toBe(9);
        expect(new Set(res.piano.map((n) => n.pitch))).toEqual(new Set([57, 60, 64]));
    });
    it("country boom-chick：根音-五度交替", () => {
        const res = arrange(base(chordMelody([60, 64, 67], 4), { style: "country" }));
        const bar0 = res.bass.filter((n) => n.tick < BAR);
        expect(bar0).toHaveLength(4);
        // 第 1/3 拍根音、第 2/4 拍五度（+7 或 -5）。
        expect(bar0[0].pitch % 12).toBe(0);
        expect(bar0[2].pitch % 12).toBe(0);
        expect((bar0[1].pitch - bar0[0].pitch) % 12).toBe(7);
        expect((bar0[3].pitch - bar0[2].pitch) % 12).toBe(7);
    });
    it("padBars=2 的风格：铺底每 2 小节才起一组", () => {
        const res = arrange(base(chordMelody([60, 64, 67], 8), { style: "rock" }));
        expect(res.bars).toBe(2);
        // 第 1 小节起、第 2 小节不起。
        expect(res.pad.filter((n) => n.tick === 0)).toHaveLength(3);
        expect(res.pad.filter((n) => n.tick === BAR)).toHaveLength(0);
        // 时值铺满 2 小节。
        expect(res.pad.every((n) => n.duration === 2 * BAR - STEP_TICKS)).toBe(true);
    });
    it("rnb sustain 钢琴：整小节延音一块", () => {
        const res = arrange(base(chordMelody([60, 64, 67], 4), { style: "rnb" }));
        const bar0 = res.piano.filter((n) => n.tick === 0);
        expect(bar0).toHaveLength(3);
        expect(bar0.every((n) => n.duration === BAR)).toBe(true);
    });
    it("扩充乐器：6 类新件全部非空、有序、力度合法（C 大调 2 小节）", () => {
        const res = arrange(base(chordMelody([60, 64, 67], 8)));
        const byTickPitch = (a, b) => {
            for (let i = 1; i < a.length; i++) {
                const prev = a[i - 1], cur = a[i];
                expect(cur.tick >= prev.tick).toBe(true);
            }
            return a.length > 0 && b.length >= 0;
        };
        // 吉他分解：每拍 2 个八分单音 → 2 小节 16。
        expect(res.guitarArp.length).toBe(16);
        // 吉他扫弦：每小节 2 次 × 3 和音 = 6/小节 → 12。
        expect(res.guitarStrum.length).toBe(12);
        // 弦乐：每 2 小节起一组 3 和音。
        expect(res.strings.length).toBe(3);
        // 电钢琴：每拍一个 → 8。
        expect(res.epiano.length).toBe(8);
        // 合成铺底：每 2 小节一组 3 音,且比普通 pad 高八度(≥ 60 起)。
        expect(res.synthPad.length).toBe(3);
        expect(res.synthPad.every((n) => n.pitch >= 60)).toBe(true);
        // Pluck：每 16 步一个 → 32。
        expect(res.pluck.length).toBe(32);
        for (const track of [res.guitarArp, res.guitarStrum, res.strings, res.epiano, res.synthPad, res.pluck]) {
            byTickPitch(track, []);
            for (const n of track) {
                expect(n.velocity).toBeGreaterThanOrEqual(1);
                expect(n.velocity).toBeLessThanOrEqual(127);
                expect(n.pitch).toBeGreaterThanOrEqual(28);
                expect(n.pitch).toBeLessThanOrEqual(96);
            }
        }
    });
    it.each([
        ["folk"], ["citypop"], ["hiphop"], ["house"], ["cinematic"],
    ])("新风格 %s：鼓/贝斯/新乐器均能产出", (style) => {
        const res = arrange(base(chordMelody([60, 64, 67], 8), { style }));
        expect(res.bars).toBe(2);
        expect(res.drums.length).toBeGreaterThan(0);
        expect(res.bass.length).toBeGreaterThan(0);
        expect(res.guitarArp.length).toBeGreaterThan(0);
        expect(res.strings.length).toBeGreaterThan(0);
    });
    it("外部骨架优先：传入 A 小调和弦时,输出和弦/贝斯根音跟随外部而非自估", () => {
        // 旋律本身是 C-E-G(自估会得 C 大调),但外部指定 A 小调和弦进行。
        const extChords = [
            { startTick: 0, endTick: BAR, root: 9, quality: "min", bass: null, label: "Am" },
            { startTick: BAR, endTick: 2 * BAR, root: 9, quality: "min", bass: null, label: "Am" },
        ];
        const res = arrange(base(chordMelody([60, 64, 67], 8), {
            externalKey: { tonic: 9, major: false, confidence: 1, label: "Am" },
            externalChords: extChords,
        }));
        expect(res.key.tonic).toBe(9);
        expect(res.key.major).toBe(false);
        expect(res.chords[0].root).toBe(9);
        // 贝斯根音 = A(pitch%12===9)。
        expect(res.bass.every((n) => n.pitch % 12 === 9)).toBe(true);
    });
    it("AI 旋律:小三和弦上不弹大三度,A 自然小调内无离调音", () => {
        // A-C-E → A 小三,key=Am。旋律音只能落在 A 自然小调音集(和弦音+经过音)。
        const res = arrange(base(chordMelody([57, 60, 64], 32)));
        expect(res.key.major).toBe(false);
        expect(res.key.tonic).toBe(9);
        const aMinorPcs = new Set([9, 11, 0, 2, 4, 5, 7]); // A B C D E F G
        expect(res.melody.length).toBeGreaterThan(0);
        for (const n of res.melody)
            expect(aMinorPcs.has(n.pitch % 12)).toBe(true);
    });
    it("AI 旋律:经过音按调性移调,D 大调歌曲不出现 C/F 等 C 调残留离调音", () => {
        // D-F#-A → D 大调,tonic=2。修复前经过音写死在 C 调会混入 C(0)/F(5)。
        const res = arrange(base(chordMelody([62, 66, 69], 32)));
        expect(res.key.tonic).toBe(2);
        const dMajorPcs = new Set([2, 4, 6, 7, 9, 11, 1]); // D E F# G A B C#
        expect(res.melody.length).toBeGreaterThan(0);
        for (const n of res.melody)
            expect(dMajorPcs.has(n.pitch % 12)).toBe(true);
    });
});
