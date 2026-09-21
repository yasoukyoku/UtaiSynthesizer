/**
 * 符号域节奏/力度变换单测（规划 2-1 / 2-2 / 2-3 / 2-6）。
 * 核心验收：同种子 → 同输出（可复现）；异种子 → 异输出。
 */
import { describe, it, expect } from "vitest";
import { makeRng } from "./rng";
import { humanizeNotes, applyVelocityCurve, swingQuantizeNotes, varyRhythm, msToTicks } from "./midiTransforms";
function notesOf(pitches, step = 240, dur = 220, vel = 100) {
    return pitches.map((pitch, i) => ({ tick: i * step, duration: dur, pitch, velocity: vel }));
}
describe("makeRng 确定性", () => {
    it("同种子产生相同序列，异种子产生不同序列", () => {
        const a1 = makeRng(12345);
        const a2 = makeRng(12345);
        const b = makeRng(54321);
        const seqA1 = Array.from({ length: 16 }, () => a1());
        const seqA2 = Array.from({ length: 16 }, () => a2());
        const seqB = Array.from({ length: 16 }, () => b());
        expect(seqA1).toEqual(seqA2);
        expect(seqA1).not.toEqual(seqB);
    });
    it("种子 0 回落到默认种子，值域 [0,1)", () => {
        const rng = makeRng(0);
        for (let i = 0; i < 1000; i++) {
            const v = rng();
            expect(v).toBeGreaterThanOrEqual(0);
            expect(v).toBeLessThan(1);
        }
    });
});
describe("msToTicks", () => {
    it("120BPM 下 500ms = 一拍 = 480 tick", () => {
        expect(msToTicks(500, 120)).toBeCloseTo(480, 6);
    });
});
describe("humanizeNotes", () => {
    const base = notesOf([72, 74, 76, 77, 79, 77, 76, 74], 240, 220, 100);
    it("同种子完全可复现", () => {
        const o = { timingMs: 15, velocityJitter: 10, tempo: 120, seed: 777 };
        expect(humanizeNotes(base, o)).toEqual(humanizeNotes(base, o));
    });
    it("异种子产生不同结果", () => {
        const o = { timingMs: 15, velocityJitter: 10, tempo: 120 };
        const a = humanizeNotes(base, { ...o, seed: 1 });
        const b = humanizeNotes(base, { ...o, seed: 2 });
        expect(a).not.toEqual(b);
    });
    it("幅度约束：起音偏移 ≤ ±ms 换算值，力度 ∈ [1,127]", () => {
        const out = humanizeNotes(base, { timingMs: 20, velocityJitter: 8, tempo: 120, seed: 42 });
        const maxDt = Math.ceil(msToTicks(20, 120)) + 1;
        base.forEach((n, i) => {
            expect(Math.abs(out[i].tick - n.tick)).toBeLessThanOrEqual(maxDt);
            expect(out[i].tick).toBeGreaterThanOrEqual(0);
            expect(out[i].velocity).toBeGreaterThanOrEqual(1);
            expect(out[i].velocity).toBeLessThanOrEqual(127);
        });
    });
    it("不修改输入数组", () => {
        const snapshot = JSON.stringify(base);
        humanizeNotes(base, { timingMs: 20, velocityJitter: 8, tempo: 120, seed: 1 });
        expect(JSON.stringify(base)).toBe(snapshot);
    });
});
describe("applyVelocityCurve", () => {
    const base = notesOf([60, 62, 64, 65, 67, 69, 71, 72], 480, 440, 90);
    it("crescendo 后半句平均力度高于前半句", () => {
        const out = applyVelocityCurve(base, { curve: "crescendo", intensity: 80 });
        const firstHalf = out.slice(0, 4).reduce((s, n) => s + n.velocity, 0) / 4;
        const secondHalf = out.slice(4).reduce((s, n) => s + n.velocity, 0) / 4;
        expect(secondHalf).toBeGreaterThan(firstHalf);
    });
    it("decrescendo 首音高于末音", () => {
        const out = applyVelocityCurve(base, { curve: "decrescendo", intensity: 80 });
        expect(out[0].velocity).toBeGreaterThan(out[out.length - 1].velocity);
    });
    it("arch 中段最高", () => {
        const out = applyVelocityCurve(base, { curve: "arch", intensity: 80 });
        const mid = out.slice(2, 6).reduce((s, n) => s + n.velocity, 0) / 4;
        const ends = (out[0].velocity + out[out.length - 1].velocity) / 2;
        expect(mid).toBeGreaterThan(ends);
    });
    it("intensity=0 时力度不变（形状偏移为零）", () => {
        const out = applyVelocityCurve(base, { curve: "crescendo", intensity: 0 });
        out.forEach((n, i) => expect(n.velocity).toBe(base[i].velocity));
    });
    it("custom 形状按位置取样", () => {
        // 4 个音等距铺满 → 位置 t = i/3 → shape 节点恰好逐一对齐 [0, 1, 0, 1]。
        const four = notesOf([60, 62, 64, 65], 480, 440, 90);
        const out = applyVelocityCurve(four, { curve: "custom", intensity: 100, shape: [0, 1, 0, 1] });
        expect(out[0].velocity).toBeLessThan(out[1].velocity);
        expect(out[1].velocity).toBeGreaterThan(out[2].velocity);
    });
});
describe("swingQuantizeNotes", () => {
    it("八分网格：奇数八分位 swing=100 延迟 480/6", () => {
        // 240 = 奇数八分位（第二八分），期望 240 + 80 = 320。
        const out = swingQuantizeNotes([{ tick: 240, duration: 200, pitch: 60, velocity: 100 }], {
            grid: 8,
            swing: 100,
            quantize: 0,
        });
        expect(out[0].tick).toBe(320);
    });
    it("八分网格：偶数位不动", () => {
        const out = swingQuantizeNotes(notesOf([60, 62], 480, 440), { grid: 8, swing: 100, quantize: 0 });
        expect(out[0].tick).toBe(0);
        expect(out[1].tick).toBe(480);
    });
    it("十六分网格：奇数十六分位 swing=100 延迟 480/12", () => {
        const out = swingQuantizeNotes([{ tick: 120, duration: 100, pitch: 60, velocity: 100 }], {
            grid: 16,
            swing: 100,
            quantize: 0,
        });
        expect(out[0].tick).toBe(160);
    });
    it("quantize=100 吸附到网格，swing=0 时为纯量化", () => {
        const out = swingQuantizeNotes([{ tick: 130, duration: 100, pitch: 60, velocity: 100 }], { grid: 8, swing: 0, quantize: 100 });
        expect(out[0].tick).toBe(240);
    });
    it("swing=0 且 quantize=0 时完全透传", () => {
        const base = notesOf([60, 63, 66], 247, 200);
        const out = swingQuantizeNotes(base, { grid: 8, swing: 0, quantize: 0 });
        expect(out.map((n) => n.tick)).toEqual(base.map((n) => n.tick));
    });
});
describe("varyRhythm", () => {
    const base = notesOf([72, 74, 76, 77, 79, 77, 76, 74], 240, 220);
    it("push 提前、layBack 拖后，量随 amount 线性", () => {
        const push100 = varyRhythm(base, { mode: "push", amount: 100 });
        const push50 = varyRhythm(base, { mode: "push", amount: 50 });
        expect(push100[1].tick).toBe(base[1].tick - 120);
        expect(push50[1].tick).toBe(base[1].tick - 60);
        const back = varyRhythm(base, { mode: "layBack", amount: 100 });
        expect(back[1].tick).toBe(base[1].tick + 120);
    });
    it("push 不产生负 tick", () => {
        const out = varyRhythm(base, { mode: "push", amount: 100 });
        out.forEach((n) => expect(n.tick).toBeGreaterThanOrEqual(0));
    });
    it("syncopate 同种子可复现，正拍有概率提前", () => {
        const o = { mode: "syncopate", amount: 100, seed: 9 };
        expect(varyRhythm(base, o)).toEqual(varyRhythm(base, o));
    });
    it("sparse 删除弱位但保留每小节头", () => {
        // base 的 240/720 等 = 偶数八分（正拍间的八分位，isWeak 判定含 120*2=240 偏移）。
        const dense = notesOf([72, 74, 76, 77], 120, 100);
        const out = varyRhythm(dense, { mode: "sparse", amount: 100, seed: 3 });
        expect(out.length).toBeLessThan(dense.length);
        expect(out[0].tick).toBe(0);
    });
});
