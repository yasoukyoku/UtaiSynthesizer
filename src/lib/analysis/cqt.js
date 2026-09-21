/**
 * §user「CQT 离散频谱分析」——NoteDigger 同源核心（常数Q变换）的 TypeScript 移植。
 *
 * CQT 与普通 STFT 的区别：滤波器带宽随频率指数变化（每个半音一个滤波器，
 * 低频窗长、高频窗短），音乐信号的对数频率结构下分辨率远优于线性 FFT。
 *
 * 实现：每个 MIDI 音高一个 Goertzel 谐振器（复数增益版），窗长 = Q 个周期
 * （Q = 1/(2^(1/12)−1) ≈ 16.8，半音分辨率），逐帧滑动。纯离线、确定性。
 *
 *   analyzeCqt(samples, sampleRate) → { times[], pitches[], magnitudes[][] }
 *
 * 典型用途：音高显著性热图（配合 AMT 的转谱校验 / NoteDigger 式可视化扒谱）。
 */
/** MIDI 音高 → 频率（Hz）。 */
export function midiPitchToFreq(p) {
    return 440 * Math.pow(2, (p - 69) / 12);
}
/** 单帧单频点 Goertzel 幅度。窗 Hann 加权，返回线性幅度。 */
function goertzelMagnitude(x, start, len, freq, sampleRate) {
    const w = (2 * Math.PI * freq) / sampleRate;
    const coeff = 2 * Math.cos(w);
    let s1 = 0;
    let s2 = 0;
    // Hann 窗同步内联（避免额外分配）
    for (let i = 0; i < len; i++) {
        const idx = start + i;
        const t = i / (len - 1);
        const win = 0.5 - 0.5 * Math.cos(2 * Math.PI * t);
        const v = (x[idx] ?? 0) * win;
        const s0 = v + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    // 幅度：|X|² = s1² + s2² − 2·cos(w)·s1·s2
    const power = s1 * s1 + s2 * s2 - coeff * s1 * s2;
    return Math.sqrt(Math.max(0, power));
}
/**
 * 主入口：单声道采样 → CQT 幅度矩阵。
 * 复杂度 O(frames × pitches × window)，88 音高 × 30s 音频 ≈ 秒级（离线可接受）。
 */
export function analyzeCqt(samples, sampleRate, opts = {}) {
    const minPitch = opts.minPitch ?? 21;
    const maxPitch = Math.min(opts.maxPitch ?? 108, 21 + 87);
    const hopSec = opts.hopSec ?? 0.02;
    const q = opts.q ?? 1 / (Math.pow(2, 1 / 12) - 1);
    const pitches = [];
    const freqs = [];
    const windowLens = [];
    for (let p = minPitch; p <= maxPitch; p++) {
        const f = midiPitchToFreq(p);
        const win = Math.min(samples.length, Math.max(8, Math.round((q * sampleRate) / f)));
        pitches.push(p);
        freqs.push(f);
        windowLens.push(win);
    }
    if (pitches.length === 0 || samples.length === 0) {
        return { times: [], magnitudes: [], pitches: [], hopSec };
    }
    const hopSamples = Math.max(1, Math.round(hopSec * sampleRate));
    const frameCount = Math.max(1, Math.floor((samples.length - Math.min(...windowLens)) / hopSamples) + 1);
    const magnitudes = pitches.map(() => new Float32Array(frameCount));
    const times = [];
    for (let fr = 0; fr < frameCount; fr++) {
        const center = fr * hopSamples;
        times.push(center / sampleRate);
        for (let pi = 0; pi < pitches.length; pi++) {
            const win = windowLens[pi];
            const half = win >> 1;
            const start = Math.max(0, Math.min(samples.length - win, center - half));
            magnitudes[pi][fr] = goertzelMagnitude(samples, start, win, freqs[pi], sampleRate);
        }
    }
    return { times, magnitudes, pitches, hopSec };
}
/** 从 CQT 结果提取逐帧音高显著性 top-K（NoteDigger 扒谱的候选音符来源）。 */
export function pitchSalience(result, topK = 4) {
    const frames = result.times.length;
    const out = [];
    for (let fr = 0; fr < frames; fr++) {
        const cands = [];
        for (let pi = 1; pi < result.pitches.length - 1; pi++) {
            const m = result.magnitudes[pi][fr] ?? 0;
            // 局部极大（相邻音高都比它小）
            if (m > (result.magnitudes[pi - 1][fr] ?? 0) && m > (result.magnitudes[pi + 1][fr] ?? 0)) {
                cands.push({ pitch: result.pitches[pi], magnitude: m });
            }
        }
        cands.sort((a, b) => b.magnitude - a.magnitude);
        out.push({ time: result.times[fr], topPitches: cands.slice(0, topK) });
    }
    return out;
}
