/**
 * 符号域确定性随机源 — xorshift32。
 * 种子约定与 effectsBus.ts 一致（0x2f6e2b1）：相同种子 → 相同输出序列，
 * 保证"同参重跑结果可复现"（规划 2-1 / 4.4 验收要求）。
 */
export const DEFAULT_SYMBOL_SEED = 0x2f6e2b1;
/** 创建 [0,1) 均匀分布的确定性随机函数。种子 0 视为未设置，回落到默认种子。 */
export function makeRng(seed) {
    let s = (seed === undefined || seed === 0 ? DEFAULT_SYMBOL_SEED : seed) >>> 0;
    return () => {
        s ^= s << 13;
        s ^= s >>> 17;
        s ^= s << 5;
        return (s >>> 0) / 4294967296;
    };
}
/** FNV-1a 字符串散列 — 把文本（如节点 id）折叠成稳定种子。 */
export function hashStringToSeed(str) {
    let h = 2166136261 >>> 0;
    for (let i = 0; i < str.length; i++) {
        h ^= str.charCodeAt(i);
        h = Math.imul(h, 16777619) >>> 0;
    }
    return h >>> 0;
}
/** [min, max] 整数闭区间取值。 */
export function rngInt(rng, min, max) {
    return min + Math.floor(rng() * (max - min + 1));
}
/** 以 probability 概率返回 true。 */
export function rngChance(rng, probability) {
    return rng() < probability;
}
