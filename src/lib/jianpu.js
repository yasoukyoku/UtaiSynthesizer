/**
 * Muno 阶段3「简谱标尺」——MIDI 音高 → 简谱数字（1-7 + 八度点 + 变化记号）。
 *
 * 约定（首调唱名，主音 = 1）：
 *   · tonic 为音级 0-11（C=0），来自 chordAnalysis 的 estimateKey（默认 C）。
 *   · 调内自然音级 → 数字 1..7；变化音用固定映射（#1 b3 #4 b6 b7 …），
 *     与国内简谱惯例一致（升 4 / 降 7 最常见）。
 *   · 八度点：1 的基准八度 = 算 C4..B4 那一圈（MIDI 60..71）里的 tonic，
 *     每高一个八度加一个上点、低一个八度加一个下点，各最多 3 个（更多省略）。
 *
 * 纯函数、确定性 —— 单测钉死映射表（jianpu.test.ts）。
 */
/** 半音偏移（相对 tonic）→ 简谱文本。0..11 全覆盖。 */
const DEGREE_TEXT = [
    "1", "#1", "2", "b3", "3", "4", "#4", "5", "b6", "6", "b7", "7",
];
const mod12 = (n) => ((n % 12) + 12) % 12;
const clampDots = (n) => Math.max(0, Math.min(3, n));
/**
 * pitch（MIDI 0-127）在主音 tonic（0-11，C=0）下的简谱唱名。
 * 越界的 tonic/异常 pitch 都收敛到安全值（不抛错——绘制路径每帧调用）。
 */
export function jianpuForPitch(pitch, tonic) {
    const t = Math.max(0, Math.min(11, Math.round(tonic)));
    const p = Math.max(0, Math.min(127, Math.round(pitch)));
    const rel = mod12(p - t);
    // 1 的基准：MIDI 60..71 那一圈里的 tonic（C 调 → C4 = 1；B 调 → B4 = 1）。
    const base = 60 + t;
    const octave = Math.floor((p - base) / 12);
    return {
        text: DEGREE_TEXT[rel] ?? "1",
        dotsAbove: clampDots(octave),
        dotsBelow: clampDots(-octave),
    };
}
