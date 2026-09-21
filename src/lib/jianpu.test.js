/**
 * Muno 阶段3 简谱映射测试：钉死 jianpuForPitch 的首调唱名表与八度点规则。
 * 规则本体在 jianpu.ts 头注释 —— 这里是「文档即测试」的钉子：
 * 改映射必须连同惯例一起改，且永远有对照。
 */
import { describe, it, expect } from "vitest";
import { jianpuForPitch } from "./jianpu";
describe("jianpuForPitch", () => {
    it("C 大调（tonic=0）：C4=1、白键 1-7、黑键变音", () => {
        expect(jianpuForPitch(60, 0)).toEqual({ text: "1", dotsAbove: 0, dotsBelow: 0 });
        expect(jianpuForPitch(62, 0)).toEqual({ text: "2", dotsAbove: 0, dotsBelow: 0 });
        expect(jianpuForPitch(64, 0)).toEqual({ text: "3", dotsAbove: 0, dotsBelow: 0 });
        expect(jianpuForPitch(65, 0)).toEqual({ text: "4", dotsAbove: 0, dotsBelow: 0 });
        expect(jianpuForPitch(67, 0)).toEqual({ text: "5", dotsAbove: 0, dotsBelow: 0 });
        expect(jianpuForPitch(69, 0)).toEqual({ text: "6", dotsAbove: 0, dotsBelow: 0 });
        expect(jianpuForPitch(71, 0)).toEqual({ text: "7", dotsAbove: 0, dotsBelow: 0 });
        // 黑键：#1 / b3 / #4 / b6 / b7（含 C#5=73=同八度）
        expect(jianpuForPitch(61, 0).text).toBe("#1");
        expect(jianpuForPitch(63, 0).text).toBe("b3");
        expect(jianpuForPitch(66, 0).text).toBe("#4");
        expect(jianpuForPitch(68, 0).text).toBe("b6");
        expect(jianpuForPitch(70, 0).text).toBe("b7");
    });
    it("八度点：每高/低一个八度加一，钳到 3", () => {
        expect(jianpuForPitch(72, 0)).toEqual({ text: "1", dotsAbove: 1, dotsBelow: 0 });
        expect(jianpuForPitch(84, 0)).toEqual({ text: "1", dotsAbove: 2, dotsBelow: 0 });
        expect(jianpuForPitch(48, 0)).toEqual({ text: "1", dotsAbove: 0, dotsBelow: 1 });
        expect(jianpuForPitch(36, 0)).toEqual({ text: "1", dotsAbove: 0, dotsBelow: 2 });
        // 96 = C6，96-60=36 → 3 个上点
        expect(jianpuForPitch(96, 0).dotsAbove).toBe(3);
        // 108 = C7 → 4 个八度，钳 3
        expect(jianpuForPitch(108, 0).dotsAbove).toBe(3);
    });
    it("G 大调（tonic=7）：G4=1、F#4=7、D5 同八度 5、D6 高八度 5", () => {
        expect(jianpuForPitch(67, 7)).toEqual({ text: "1", dotsAbove: 0, dotsBelow: 0 });
        expect(jianpuForPitch(66, 7).text).toBe("7");
        expect(jianpuForPitch(69, 7).text).toBe("2");
        // D5 = G4+7 半音，仍在 1 的八度内 → 无点；D6 才是高八度 5。
        expect(jianpuForPitch(74, 7)).toEqual({ text: "5", dotsAbove: 0, dotsBelow: 0 });
        expect(jianpuForPitch(86, 7)).toEqual({ text: "5", dotsAbove: 1, dotsBelow: 0 });
    });
    it("B 调（tonic=11）：B4=1、C5 同八度 #1、C4 是低八度 #1", () => {
        expect(jianpuForPitch(71, 11)).toEqual({ text: "1", dotsAbove: 0, dotsBelow: 0 });
        expect(jianpuForPitch(72, 11)).toEqual({ text: "#1", dotsAbove: 0, dotsBelow: 0 });
        // C4 = B3+1 半音 → 低八度的升 1（#1 下点），不是 b7（B 调 b7 是 A）。
        expect(jianpuForPitch(60, 11)).toEqual({ text: "#1", dotsAbove: 0, dotsBelow: 1 });
        // A4 = B4-2 半音 → 低八度的 b7（B 大调 7 是 A#）。
        expect(jianpuForPitch(69, 11)).toEqual({ text: "b7", dotsAbove: 0, dotsBelow: 1 });
    });
    it("A 小调主音（tonic=9）：A4=1", () => {
        expect(jianpuForPitch(69, 9)).toEqual({ text: "1", dotsAbove: 0, dotsBelow: 0 });
        expect(jianpuForPitch(72, 9).text).toBe("b3");
    });
    it("越界输入收敛不抛错（绘制路径每帧调用）", () => {
        expect(() => jianpuForPitch(-5, -3)).not.toThrow();
        expect(() => jianpuForPitch(999, 99)).not.toThrow();
        // 钳位后等价于 (0,0) / (127,11)
        expect(jianpuForPitch(-5, -3)).toEqual(jianpuForPitch(0, 0));
        expect(jianpuForPitch(999, 99).text).toBe(jianpuForPitch(127, 11).text);
    });
});
