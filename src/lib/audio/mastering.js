/**
 * §user「母带处理」—— 纯 DSP 母带链（确定性、无 WebAudio 依赖，可在离线导出与
 * 单元测试中逐位复现）。链路（每声道）：
 *
 *   EQ（RBJ 双二阶）→ 立体声联动压缩器 → 前瞻峰值限制器 → 安全钳位
 *
 * 设计取向「发行级、但绝不毁歌」：
 *   - EQ 三段全是大宽度、小增益（±1.5~2 dB）：低频托底、中频人声存在感、
 *     高频开一点“空气感”——不是修色，是整体打磨。
 *   - 压缩器软拐点、2.5:1 温和比率、立体声联动（不歪像）：把干声/分轨
 *     合成后忽大忽小的电平拢住，不做“响度战争”。
 *   - 限制器 -1 dBFS 天花板 + 3ms 前瞻（峰值前开始压，无过冲）+ 80ms 释放。
 */
const LN10_OVER_20 = Math.LN10 / 20;
function dbToLin(db) {
    return Math.exp(db * LN10_OVER_20);
}
/** RBJ Audio EQ Cookbook 双二阶（Direct Form I），系数离线算好，逐样本恒定。 */
class Biquad {
    constructor() {
        Object.defineProperty(this, "b0", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: 1
        });
        Object.defineProperty(this, "b1", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: 0
        });
        Object.defineProperty(this, "b2", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: 0
        });
        Object.defineProperty(this, "a1", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: 0
        });
        Object.defineProperty(this, "a2", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: 0
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
    static lowShelf(sr, f0, gainDb, s = 1) {
        const A = Math.pow(10, gainDb / 40);
        const w0 = (2 * Math.PI * Math.min(f0, sr * 0.49)) / sr;
        const cw = Math.cos(w0);
        const sw = Math.sin(w0);
        const alpha = (sw / 2) * Math.sqrt((A + 1 / A) * (1 / s - 1) + 2);
        const twoSqrtAAlpha = 2 * Math.sqrt(A) * alpha;
        const b = new Biquad();
        // RBJ cookbook：b 侧用 −(A−1)cos、a 侧用 +(A−1)cos（镜像）。
        const a0 = A + 1 + (A - 1) * cw + twoSqrtAAlpha;
        b.b0 = (A * (A + 1 - (A - 1) * cw + twoSqrtAAlpha)) / a0;
        b.b1 = (2 * A * (A - 1 - (A + 1) * cw)) / a0;
        b.b2 = (A * (A + 1 - (A - 1) * cw - twoSqrtAAlpha)) / a0;
        b.a1 = (-2 * (A - 1 + (A + 1) * cw)) / a0;
        b.a2 = (A + 1 + (A - 1) * cw - twoSqrtAAlpha) / a0;
        return b;
    }
    static peaking(sr, f0, q, gainDb) {
        const A = Math.pow(10, gainDb / 40);
        const w0 = (2 * Math.PI * Math.min(f0, sr * 0.49)) / sr;
        const cw = Math.cos(w0);
        const sw = Math.sin(w0);
        const alpha = sw / (2 * q);
        const b = new Biquad();
        const a0 = 1 + alpha / A;
        b.b0 = (1 + alpha * A) / a0;
        b.b1 = (-2 * cw) / a0;
        b.b2 = (1 - alpha * A) / a0;
        b.a1 = (-2 * cw) / a0;
        b.a2 = (1 - alpha / A) / a0;
        return b;
    }
    static highShelf(sr, f0, gainDb, s = 1) {
        const A = Math.pow(10, gainDb / 40);
        const w0 = (2 * Math.PI * Math.min(f0, sr * 0.49)) / sr;
        const cw = Math.cos(w0);
        const sw = Math.sin(w0);
        const alpha = (sw / 2) * Math.sqrt((A + 1 / A) * (1 / s - 1) + 2);
        const twoSqrtAAlpha = 2 * Math.sqrt(A) * alpha;
        const b = new Biquad();
        const a0 = A + 1 - (A - 1) * cw + twoSqrtAAlpha;
        b.b0 = (A * (A + 1 + (A - 1) * cw + twoSqrtAAlpha)) / a0;
        b.b1 = (-2 * A * (A - 1 + (A + 1) * cw)) / a0;
        b.b2 = (A * (A + 1 + (A - 1) * cw - twoSqrtAAlpha)) / a0;
        b.a1 = (2 * (A - 1 - (A + 1) * cw)) / a0;
        b.a2 = (A + 1 - (A - 1) * cw - twoSqrtAAlpha) / a0;
        return b;
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
/**
 * 就地处理立体声平面。返回输出真实峰值。所有内部状态独立于调用——同一输入
 * 永远得到同一输出（确定性），导出与测试共用。
 */
export function applyMasterChain(L, R, sampleRate, opts = {}) {
    const n = Math.min(L.length, R.length);
    if (n === 0)
        return 0;
    const ceiling = opts.ceiling ?? dbToLin(-1);
    const compThreshDb = opts.compThresholdDb ?? -14;
    const ratio = opts.compRatio ?? 2.5;
    const makeupDb = opts.makeupDb ?? 1.5;
    const makeupLin = dbToLin(makeupDb);
    // 1) EQ：低频托底 / 人声存在感 / 高频空气感（同参双声道，保持定位）。
    const eqsL = [
        Biquad.lowShelf(sampleRate, 90, 1.5),
        Biquad.peaking(sampleRate, 3000, 0.9, 1.5),
        Biquad.highShelf(sampleRate, 10000, 2.0),
    ];
    const eqsR = [
        Biquad.lowShelf(sampleRate, 90, 1.5),
        Biquad.peaking(sampleRate, 3000, 0.9, 1.5),
        Biquad.highShelf(sampleRate, 10000, 2.0),
    ];
    for (let i = 0; i < n; i++) {
        let l = L[i];
        let r = R[i];
        for (let k = 0; k < eqsL.length; k++) {
            l = eqsL[k].process(l);
            r = eqsR[k].process(r);
        }
        L[i] = l;
        R[i] = r;
    }
    // 2) 立体声联动压缩器：包络取双声道最大绝对值，attack 15ms / release 150ms。
    const atkC = Math.exp(-1 / (0.015 * sampleRate));
    const relC = Math.exp(-1 / (0.15 * sampleRate));
    const slopeAbove = 1 - 1 / ratio;
    const envToGainDb = (envLin) => {
        if (envLin <= 1e-6)
            return 0;
        const envDb = 20 * Math.log10(envLin);
        if (envDb <= compThreshDb)
            return 0;
        // 软拐点：阈值上方 6dB 内从 1:1 平滑过渡到设定比率。
        const over = envDb - compThreshDb;
        const knee = 6;
        let reduction;
        if (over < knee) {
            // 平方插值软拐点（过渡区斜率从 0 → slopeAbove）。
            reduction = slopeAbove * (over * over) / (2 * knee);
        }
        else {
            reduction = slopeAbove * (over - knee / 2);
        }
        return -reduction;
    };
    let env = 0;
    let gainDb = 0;
    for (let i = 0; i < n; i++) {
        const rect = Math.max(Math.abs(L[i]), Math.abs(R[i]));
        const coef = rect > env ? atkC : relC;
        env = coef * env + (1 - coef) * rect;
        const targetDb = envToGainDb(env * makeupLin);
        // 增益自身也做平滑（attack 快释放慢的镜像：压快放慢）。
        gainDb = targetDb < gainDb
            ? targetDb + (gainDb - targetDb) * atkC
            : targetDb + (gainDb - targetDb) * relC;
        const g = dbToLin(gainDb) * makeupLin;
        L[i] = L[i] * g;
        R[i] = R[i] * g;
    }
    // 3) 前瞻峰值限制器：3ms 前瞻滑窗最大（单调队列 O(1) 均摊）+ 80ms 释放。
    //    三段式：先取包络 rect[]（此后 L/R 在施加前不再被读），再按窗口
    //    [i, i+look] 滑出每个样本的增益（压前不压后 = 无过冲），最后统一施加。
    const look = Math.max(1, Math.round(0.003 * sampleRate));
    const relL = Math.exp(-1 / (0.08 * sampleRate));
    const rect = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        rect[i] = Math.max(Math.abs(L[i]), Math.abs(R[i]));
    }
    const gainArr = new Float32Array(n).fill(1);
    const qIdx = []; // 单调队列（下标，值从 rect 只读取）
    const qVal = [];
    let gainLin = 1;
    for (let j = 0; j < n + look; j++) {
        if (j < n) {
            const v = rect[j];
            while (qVal.length > 0 && qVal[qVal.length - 1] <= v) {
                qVal.pop();
                qIdx.pop();
            }
            qIdx.push(j);
            qVal.push(v);
        }
        const i = j - look; // 窗口 [i, i+look] 恰好已全部入队
        if (i < 0)
            continue;
        while (qIdx.length > 0 && qIdx[0] < i) {
            qIdx.shift();
            qVal.shift();
        }
        const winMax = qVal[0] ?? 0;
        const target = winMax > ceiling ? ceiling / winMax : 1;
        gainLin = target < gainLin
            ? target
            : Math.min(gainLin + (1 - gainLin) * relL, target);
        gainArr[i] = gainLin;
    }
    let peak = 0;
    for (let i = 0; i < n; i++) {
        const l = Math.max(-1, Math.min(1, L[i] * gainArr[i]));
        const r = Math.max(-1, Math.min(1, R[i] * gainArr[i]));
        L[i] = l;
        R[i] = r;
        const a = Math.abs(l);
        if (a > peak)
            peak = a;
        const b = Math.abs(r);
        if (b > peak)
            peak = b;
    }
    return peak;
}
