/**
 * §user「鼓点识别」——离线鼓转录（kick / snare / hi-hat 三件套）。
 *
 * 纯前端 DSP、确定性。方法（经典能量包络法，ADRess/librosa 同思路）：
 *
 *   立体声→单声道 → 三频段双二阶分离（kick <130Hz / snare 180-500Hz / hat >5kHz）
 *   → 包络跟随（RMS 平滑）→ 谱通量（正向差分）→ 自适应阈值峰选
 *   → 鼓点 {时间秒, 类型, 力度} → （可选）GM 通道 10 鼓音符
 *
 * 不依赖模型下载，秒级出结果；配合 MSST 分离出的鼓轨使用效果最佳，
 * 直接吃全曲混音也能用（人声瞬态多数落在 snare 频段之外）。
 */
/** GM 鼓音符号：channel 10（索引 9）。 */
export const GM_DRUM_NOTE = { kick: 36, snare: 38, hat: 42 };
/** 极简双二阶（Direct Form I），系数离线算好。 */
class Biquad {
    constructor(b0, b1, b2, a1, a2) {
        Object.defineProperty(this, "b0", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: b0
        });
        Object.defineProperty(this, "b1", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: b1
        });
        Object.defineProperty(this, "b2", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: b2
        });
        Object.defineProperty(this, "a1", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: a1
        });
        Object.defineProperty(this, "a2", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: a2
        });
        Object.defineProperty(this, "x1", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: 0
        });
        Object.defineProperty(this, "x2", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: 0
        });
        Object.defineProperty(this, "y1", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: 0
        });
        Object.defineProperty(this, "y2", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: 0
        });
    }
    process(x) {
        const y = this.b0 * x + this.b1 * this.x1 + this.b2 * this.x2 - this.a1 * this.y1 - this.a2 * this.y2;
        this.x2 = this.x1;
        this.x1 = x;
        this.y2 = this.y1;
        this.y1 = y;
        return y;
    }
}
function lowpass(sr, f0) {
    const w0 = (2 * Math.PI * Math.min(f0, sr * 0.49)) / sr;
    const c = Math.cos(w0), s = Math.sin(w0);
    const alpha = s / Math.SQRT2;
    const a0 = 1 + alpha;
    return new Biquad(((1 - c) / 2) / a0, (1 - c) / a0, ((1 - c) / 2) / a0, (-2 * c) / a0, (1 - alpha) / a0);
}
function highpass(sr, f0) {
    const w0 = (2 * Math.PI * Math.min(f0, sr * 0.49)) / sr;
    const c = Math.cos(w0), s = Math.sin(w0);
    const alpha = s / Math.SQRT2;
    const a0 = 1 + alpha;
    return new Biquad(((1 + c) / 2) / a0, (-(1 + c)) / a0, ((1 + c) / 2) / a0, (-2 * c) / a0, (1 - alpha) / a0);
}
function bandpass(sr, f0, q) {
    const w0 = (2 * Math.PI * Math.min(f0, sr * 0.49)) / sr;
    const c = Math.cos(w0), s = Math.sin(w0);
    const alpha = s / (2 * q);
    const a0 = 1 + alpha;
    return new Biquad(alpha / a0, 0, -alpha / a0, (-2 * c) / a0, (1 - alpha) / a0);
}
/** 包络 + 正向通量 + 自适应峰选，单频段 onset 检测。 */
function onsetsInBand(band, sr, type, beta, localMeanSec, minGapSec) {
    const n = band.length;
    if (n === 0)
        return [];
    // 1) RMS 包络，10ms 窗。
    const hop = Math.max(1, Math.round(0.01 * sr));
    const frames = Math.floor(n / hop);
    const env = new Float32Array(frames);
    for (let f = 0; f < frames; f++) {
        let sum = 0;
        const s = f * hop;
        const e = Math.min(n, s + hop);
        for (let i = s; i < e; i++) {
            const v = band[i] ?? 0;
            sum += v * v;
        }
        env[f] = Math.sqrt(sum / Math.max(1, e - s));
    }
    // 2) 正向通量。
    const flux = new Float32Array(frames);
    for (let f = 1; f < frames; f++) {
        const d = (env[f] ?? 0) - (env[f - 1] ?? 0);
        flux[f] = d > 0 ? d : 0;
    }
    // 3) 自适应阈值 + 峰选（局部均值滑动窗）。
    const meanWin = Math.max(1, Math.round((localMeanSec / 0.01)));
    const hits = [];
    let lastHitFrame = -Infinity;
    for (let f = 1; f < frames; f++) {
        const v = flux[f] ?? 0;
        if (v <= 1e-6)
            continue;
        let sum = 0;
        let cnt = 0;
        for (let k = Math.max(0, f - meanWin); k < Math.min(frames, f + meanWin); k++) {
            sum += flux[k] ?? 0;
            cnt++;
        }
        const mean = cnt > 0 ? sum / cnt : 0;
        if (v < mean * beta)
            continue;
        // 局部极大：不比相邻帧小
        if (f > 0 && v < (flux[f - 1] ?? 0))
            continue;
        if (f + 1 < frames && v < (flux[f + 1] ?? 0))
            continue;
        if (f - lastHitFrame < minGapSec / 0.01)
            continue; // 双触发抑制
        lastHitFrame = f;
        // 力度：峰通量相对局部最大值的比例，clamp 到 [0.2, 1]。
        let maxFlux = 1e-9;
        for (let k = Math.max(0, f - meanWin); k < Math.min(frames, f + meanWin); k++) {
            if ((flux[k] ?? 0) > maxFlux)
                maxFlux = flux[k] ?? 0;
        }
        const rel = maxFlux > 1e-9 ? v / maxFlux : 1;
        hits.push({
            time: (f * hop) / sr,
            type,
            velocity: Math.min(1, Math.max(0.2, rel)),
        });
    }
    return hits;
}
/**
 * 主入口：立体声/单声道采样 → 鼓点列表。
 * @param channels 一个或多个声道平面（多声道取平均降混）。
 */
export function detectDrums(channels, opts) {
    const sr = opts.sampleRate;
    const n = channels.reduce((m, c) => Math.max(m, c.length), 0);
    if (n === 0 || channels.length === 0)
        return [];
    // 降混单声道
    const mono = new Float32Array(n);
    for (const c of channels) {
        for (let i = 0; i < c.length; i++)
            mono[i] = (mono[i] ?? 0) + (c[i] ?? 0) / channels.length;
    }
    const beta = opts.beta ?? 2.2;
    const localMeanSec = opts.localMeanSec ?? 0.6;
    const gaps = {
        kick: opts.minGap?.kick ?? 0.1,
        snare: opts.minGap?.snare ?? 0.08,
        hat: opts.minGap?.hat ?? 0.05,
    };
    // 三频段并行滤波（每频段独立跑一遍全信号——双二阶有状态，不能混用）。
    const bands = [
        { type: "kick", data: runFilter(mono, lowpass(sr, 130)) },
        { type: "snare", data: runFilter(mono, bandpass(sr, 320, 1.0)) },
        { type: "hat", data: runFilter(mono, highpass(sr, 5000)) },
    ];
    const hits = [];
    for (const b of bands) {
        hits.push(...onsetsInBand(b.data, sr, b.type, beta, localMeanSec, gaps[b.type]));
    }
    hits.sort((a, b) => a.time - b.time);
    return hits;
}
function runFilter(x, filter) {
    const out = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++)
        out[i] = filter.process(x[i] ?? 0);
    return out;
}
/** 鼓点 → GM 通道 10 音符（tick 域，供直接铺到轨道）。 */
export function drumHitsToNotes(hits, bpm, ppq) {
    const secToTick = (sec) => Math.round((sec * bpm * ppq) / 60);
    return hits.map((h) => ({
        tick: secToTick(h.time),
        duration: Math.max(1, Math.round(ppq / 4)), // 16 分音符长度
        pitch: GM_DRUM_NOTE[h.type],
        velocity: Math.round(Math.min(127, Math.max(1, h.velocity * 127))),
        channel: 9,
    }));
}
