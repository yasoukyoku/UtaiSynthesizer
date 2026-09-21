import { describe, it, expect } from "vitest";
import { detectDrums, drumHitsToNotes, GM_DRUM_NOTE } from "./drumDetection";
const SR = 44100;
/** 合成敲击：t 秒处一个 attack 极快的衰减正弦（模拟鼓的瞬态）。 */
function synthHit(buf, atSec, freq, decaySec = 0.08) {
    const start = Math.floor(atSec * SR);
    const len = Math.floor(decaySec * SR);
    for (let i = 0; i < len && start + i < buf.length; i++) {
        const env = Math.exp(-i / (SR * decaySec * 0.4));
        buf[start + i] = (buf[start + i] ?? 0) + Math.sin((2 * Math.PI * freq * i) / SR) * env;
    }
}
function near(hits, type, timeSec, tolSec = 0.05) {
    return hits.some((h) => h.type === type && Math.abs(h.time - timeSec) <= tolSec);
}
describe("detectDrums", () => {
    it("三频段敲击分离：kick / snare / hat 各归各的频段", () => {
        const dur = 2;
        const buf = new Float32Array(SR * dur);
        synthHit(buf, 0.5, 60); // kick：60Hz 低频
        synthHit(buf, 1.0, 320); // snare：中频段中心
        synthHit(buf, 1.5, 8000); // hat：高频
        const hits = detectDrums([buf], { sampleRate: SR });
        expect(hits.length).toBeGreaterThan(0);
        expect(near(hits, "kick", 0.5)).toBe(true);
        expect(near(hits, "snare", 1.0)).toBe(true);
        expect(near(hits, "hat", 1.5)).toBe(true);
    });
    it("力度归一：velocity ∈ [0.2, 1]", () => {
        const buf = new Float32Array(SR);
        synthHit(buf, 0.3, 60);
        synthHit(buf, 0.6, 60, 0.2);
        const hits = detectDrums([buf], { sampleRate: SR });
        for (const h of hits) {
            expect(h.velocity).toBeGreaterThanOrEqual(0.2);
            expect(h.velocity).toBeLessThanOrEqual(1);
        }
    });
    it("静音输入 → 无鼓点", () => {
        const buf = new Float32Array(SR);
        expect(detectDrums([buf], { sampleRate: SR })).toEqual([]);
    });
    it("空输入 → 空结果（不抛错）", () => {
        expect(detectDrums([], { sampleRate: SR })).toEqual([]);
        expect(detectDrums([new Float32Array(0)], { sampleRate: SR })).toEqual([]);
    });
    it("双触发抑制：10Hz 连续低频敲击不会每 50ms 报一次", () => {
        const dur = 2;
        const buf = new Float32Array(SR * dur);
        for (let k = 0; k < 20; k++)
            synthHit(buf, 0.1 * k, 60, 0.02); // 100ms 间隔
        const hits = detectDrums([buf], { sampleRate: SR, minGap: { kick: 0.5 } });
        const kicks = hits.filter((h) => h.type === "kick");
        // minGap 0.5s → 2s 内至多 ~4 个
        expect(kicks.length).toBeLessThanOrEqual(5);
    });
    it("立体声两声道都被降混", () => {
        const L = new Float32Array(SR * 2);
        const R = new Float32Array(SR * 2);
        synthHit(L, 0.5, 60);
        synthHit(R, 1.0, 60);
        const hits = detectDrums([L, R], { sampleRate: SR });
        expect(near(hits, "kick", 0.5)).toBe(true);
        expect(near(hits, "kick", 1.0)).toBe(true);
    });
});
describe("drumHitsToNotes", () => {
    it("秒 → tick 换算 + GM 通道 10 音符", () => {
        const bpm = 120;
        const ppq = 480;
        const notes = drumHitsToNotes([
            { time: 0.5, type: "kick", velocity: 1 },
            { time: 1.0, type: "snare", velocity: 0.5 },
            { time: 1.25, type: "hat", velocity: 0.9 },
        ], bpm, ppq);
        expect(notes.length).toBe(3);
        // 0.5s @120bpm = 1 拍 = 480 ticks
        expect(notes[0].tick).toBe(480);
        expect(notes[0].pitch).toBe(GM_DRUM_NOTE.kick);
        expect(notes[0].channel).toBe(9);
        expect(notes[1].pitch).toBe(GM_DRUM_NOTE.snare);
        expect(notes[1].velocity).toBe(Math.round(0.5 * 127));
        expect(notes[2].pitch).toBe(GM_DRUM_NOTE.hat);
        expect(notes[2].tick).toBe(1200);
        for (const n of notes) {
            expect(n.duration).toBeGreaterThan(0);
            expect(n.velocity).toBeGreaterThanOrEqual(1);
            expect(n.velocity).toBeLessThanOrEqual(127);
        }
    });
});
