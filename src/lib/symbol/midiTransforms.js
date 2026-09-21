/**
 * 节奏与力度域变换（规划 2-1 / 2-2 / 2-3 / 2-6，对应 4.4 节奏层面原创化）。
 * 全部为纯函数：输入 ChordAnalysisNote[] + 参数，输出新数组，不改写输入。
 * 所有随机成分走 makeRng 种子化，保证同参重跑可复现。
 */
import { TICKS_PER_BEAT } from "../constants";
import { makeRng } from "./rng";
import { clamp01, clampVel, byTick } from "./common";
/** 毫秒 → tick（用于 humanize 的"±毫秒"手感参数）。 */
export function msToTicks(ms, tempo) {
    return (ms / 1000) * (tempo / 60) * TICKS_PER_BEAT;
}
export function humanizeNotes(notes, o) {
    if (notes.length === 0)
        return [];
    const rng = makeRng(o.seed);
    const ticksPerMs = msToTicks(1, o.tempo);
    return notes
        .map((n) => {
        const dt = Math.round((rng() * 2 - 1) * o.timingMs * ticksPerMs);
        const dv = Math.round((rng() * 2 - 1) * o.velocityJitter);
        return {
            ...n,
            tick: Math.max(0, n.tick + dt),
            duration: Math.max(1, n.duration),
            velocity: clampVel((n.velocity ?? 100) + dv),
        };
    })
        .sort(byTick);
}
export function applyVelocityCurve(notes, o) {
    if (notes.length === 0)
        return [];
    const span = (clamp01(o.intensity / 100) * 64);
    const sorted = [...notes].sort(byTick);
    const first = sorted[0].tick;
    const lastTick = Math.max(...sorted.map((n) => n.tick));
    const total = Math.max(1, lastTick - first);
    const customShape = o.curve === "custom" && o.shape && o.shape.length >= 2 ? o.shape : null;
    const shape = customShape
        ? (t) => {
            // custom 采样序列按位置线性插值（与离散步进曲线等价，行为同原实现）
            const pos = t * (customShape.length - 1);
            const i = Math.min(customShape.length - 2, Math.floor(pos));
            const f = pos - i;
            const a = clamp01(customShape[i] ?? 0);
            const b = clamp01(customShape[i + 1] ?? 0);
            return a + (b - a) * f;
        }
        : o.curve === "decrescendo"
            ? (t) => 1 - t
            : o.curve === "arch"
                ? (t) => Math.sin(Math.PI * t)
                : (t) => t;
    return sorted.map((n) => {
        const t = (n.tick - first) / total;
        const offset = (clamp01(shape(t)) - 0.5) * span;
        return { ...n, velocity: clampVel((n.velocity ?? 100) + offset) };
    });
}
export function swingQuantizeNotes(notes, o) {
    const step = o.grid === 16 ? TICKS_PER_BEAT / 4 : TICKS_PER_BEAT / 2;
    const maxShift = o.grid === 16 ? TICKS_PER_BEAT / 12 : TICKS_PER_BEAT / 6;
    const q = clamp01(o.quantize / 100);
    const sw = clamp01(o.swing / 100);
    return notes
        .map((n) => {
        let tick = n.tick;
        if (q > 0) {
            const snapped = Math.round(tick / step) * step;
            tick += (snapped - tick) * q;
        }
        const idx = Math.round(tick / step);
        if (((idx % 2) + 2) % 2 === 1)
            tick += maxShift * sw;
        return { ...n, tick: Math.max(0, Math.round(tick)) };
    })
        .sort(byTick);
}
export function varyRhythm(notes, o) {
    if (notes.length === 0)
        return [];
    const amount = clamp01(o.amount / 100);
    const rng = makeRng(o.seed);
    const step16 = TICKS_PER_BEAT / 4;
    const barTicks = Math.max(1, o.beatsPerBar ?? 4) * TICKS_PER_BEAT;
    const sorted = [...notes].sort(byTick);
    switch (o.mode) {
        case "push":
            return sorted.map((n) => ({ ...n, tick: Math.max(0, Math.round(n.tick - amount * step16)) }));
        case "layBack":
            return sorted.map((n) => ({ ...n, tick: Math.round(n.tick + amount * step16) }));
        case "syncopate":
            // 正拍上的音以 amount 概率提前一个十六分（先行切分）；每小节头不动。
            return sorted.map((n) => {
                const inBar = n.tick % barTicks;
                const onBeat = inBar % TICKS_PER_BEAT === 0 && inBar !== 0;
                if (onBeat && rng() < amount)
                    return { ...n, tick: Math.max(0, n.tick - step16) };
                return { ...n };
            });
        case "sparse": {
            // 弱位（八分反拍 + 十六分弱位）以 amount 概率删除；小节头与正拍永远保留。
            const kept = [];
            sorted.forEach((n, i) => {
                const inBar = n.tick % barTicks;
                const isBarHead = i === 0 || inBar === 0;
                const isWeak = inBar % (TICKS_PER_BEAT / 2) === TICKS_PER_BEAT / 4 || inBar % (TICKS_PER_BEAT / 4) !== 0;
                if (!isBarHead && isWeak && rng() < amount)
                    return;
                kept.push({ ...n });
            });
            return kept;
        }
    }
}
