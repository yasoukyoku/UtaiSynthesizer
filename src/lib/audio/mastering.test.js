import { describe, expect, it } from "vitest";
import { applyMasterChain } from "./mastering";
const SR = 44100;
function sine(seconds, freq, amp) {
    const n = Math.round(seconds * SR);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++)
        out[i] = Math.sin((2 * Math.PI * freq * i) / SR) * amp;
    return out;
}
describe("applyMasterChain — 发行级母带链", () => {
    it("热信号被限制到 -1 dBFS 天花板以下", () => {
        const L = sine(1, 440, 0.99);
        const R = sine(1, 440, 0.99);
        const peak = applyMasterChain(L, R, SR);
        expect(peak).toBeLessThanOrEqual(10 ** (-1 / 20) + 1e-6); // -1 dBFS = 0.8912509
        expect(peak).toBeGreaterThan(0.5); // 不是无脑压死：响度仍在
    });
    it("安静信号基本保持原样（小增益打磨，不炸不糊）", () => {
        const L = sine(1, 440, 0.05);
        const R = sine(1, 440, 0.05);
        const peak = applyMasterChain(L, R, SR);
        expect(peak).toBeGreaterThan(0.03);
        expect(peak).toBeLessThan(0.5);
    });
    it("静音进静音出（无 NaN）", () => {
        const L = new Float32Array(SR);
        const R = new Float32Array(SR);
        const peak = applyMasterChain(L, R, SR);
        expect(peak).toBe(0);
        for (let i = 0; i < L.length; i++) {
            expect(Number.isFinite(L[i])).toBe(true);
            expect(Number.isFinite(R[i])).toBe(true);
        }
    });
    it("压缩器把忽大忽小的电平拢住（峰值差收窄）", () => {
        // 前 0.5s 很响、后 0.5s 很轻 —— 母带链应抬轻压响，两段峰值差距收窄。
        const n = SR;
        const L = new Float32Array(n);
        const R = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            const amp = i < n / 2 ? 0.9 : 0.15;
            const v = Math.sin((2 * Math.PI * 220 * i) / SR) * amp;
            L[i] = v;
            R[i] = v;
        }
        const loudBefore = 0.9;
        const quietBefore = 0.15;
        applyMasterChain(L, R, SR);
        let loudAfter = 0;
        let quietAfter = 0;
        for (let i = 0; i < n / 2; i++)
            loudAfter = Math.max(loudAfter, Math.abs(L[i]));
        for (let i = n / 2; i < n; i++)
            quietAfter = Math.max(quietAfter, Math.abs(L[i]));
        const before = loudBefore / quietBefore;
        const after = loudAfter / Math.max(quietAfter, 1e-9);
        expect(after).toBeLessThan(before); // 动态范围被温和收窄
    });
    it("确定性：同一输入两次处理结果逐位一致", () => {
        const a = sine(0.5, 440, 0.8);
        const b = sine(0.5, 440, 0.8);
        applyMasterChain(a, a, SR);
        applyMasterChain(b, b, SR);
        for (let i = 0; i < a.length; i++)
            expect(a[i]).toBe(b[i]);
    });
    it("空输入不抛异常", () => {
        expect(applyMasterChain(new Float32Array(0), new Float32Array(0), SR)).toBe(0);
    });
});
