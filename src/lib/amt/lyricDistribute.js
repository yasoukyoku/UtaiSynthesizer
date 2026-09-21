import { splitLyricTokens } from "../vocalNotes";
/**
 * §用户「Whisper 提取后按音节切分铺到每个音符（现在按句）」：
 * 句级歌词 → 逐音符歌词。每句文本先用 splitLyricTokens 切成音节
 * （中文一字一音节、英文按空格、日文按 mora），再按时间把音节落到
 * 该句区间内的音符起点上——每个音符一个词，改词粒度到单音。
 *
 * - 音节多于音符：剩余音节合并到最后一个音符（一个音符唱多个字）。
 * - 音符多于音节：多余音符不生成歌词行（写回 MIDI 后显示占位符）。
 * - 该句区间内没有音符：整句原样保留（时间轴信息不丢）。
 *
 * @param segments Whisper 句级段（秒，含 start/end/text）
 * @param onsets   音符起点（秒，乱序容忍，内部会排序）
 * @param fallback 空文本兜底字（默认「啊」）
 */
export function distributeLyricsToNotes(segments, onsets, fallback = "啊") {
    const sorted = [...onsets].filter((t) => Number.isFinite(t) && t >= 0).sort((a, b) => a - b);
    const out = [];
    for (const seg of segments) {
        const text = seg.text.trim();
        if (!text)
            continue;
        // 该句区间内的音符（容差 30ms 抵消 Whisper 切句与音符起点的微小错位）。
        const inSeg = sorted.filter((t) => t >= seg.start - 0.03 && t < seg.end + 0.03);
        if (inSeg.length === 0) {
            out.push({ start: seg.start, end: seg.end, text });
            continue;
        }
        const toks = splitLyricTokens(text, fallback);
        const cjk = /[\u4e00-\u9fff\u3040-\u30ff]/.test(text);
        for (let i = 0; i < inSeg.length; i++) {
            const tok = toks[i];
            const onset = inSeg[i];
            if (!tok)
                break; // 音符多于音节：后续音符留空
            const isLast = i === inSeg.length - 1;
            const merged = isLast && toks.length > inSeg.length
                ? toks.slice(i).join(cjk ? "" : " ")
                : tok;
            out.push({
                start: onset,
                end: isLast
                    ? Math.max(seg.end, onset + 0.05)
                    : inSeg[i + 1],
                text: merged,
            });
        }
    }
    return out.sort((a, b) => a.start - b.start);
}
/** 拼接两个词：任一侧是 CJK/假名则直接连写，否则加空格（英文单词边界）。 */
function joinToken(a, b) {
    const cjk = /[\u4e00-\u9fff\u3040-\u30ff\u3000-\u303f\uff00-\uffef]/;
    return cjk.test(a.slice(-1)) || cjk.test(b.charAt(0)) ? a + b : `${a} ${b}`;
}
/**
 * LRC 导出的「按句」模式：把逐音符行合并回句（相邻行间隔 ≤ gapSec 视为
 * 同一句）。相邻判断基于「下一行 start − 上一行 end」；行的时间轴来自
 * 音符起点，所以同一句内的间隔天然很小，句间停顿则明显更大。
 */
export function mergeLyricLines(lines, gapSec = 0.5) {
    const sorted = [...lines]
        .filter((l) => l.text.trim().length > 0)
        .sort((a, b) => a.start - b.start);
    const out = [];
    for (const l of sorted) {
        const last = out[out.length - 1];
        if (last && l.start - last.end <= gapSec) {
            last.end = Math.max(last.end, l.end);
            last.text = joinToken(last.text, l.text.trim());
        }
        else {
            out.push({ start: l.start, end: l.end, text: l.text.trim() });
        }
    }
    return out;
}
