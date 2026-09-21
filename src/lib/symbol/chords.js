/**
 * 和弦域共享工具：和弦标签 ⇄ (root, quality) 解析、chordBlock JSON ⇄ ChordSegment 互转。
 * 供 reharmonize / structureEdit / engine 的 melodyGen 重做共用。
 */
import { TICKS_PER_BEAT } from "../constants";
import { mod12p } from "./common";
export const PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const LETTER_TO_PC = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
// 长后缀优先，避免 "m7b5" 被 "m" 截走、"maj7" 被 "maj" 截走、"m7" 被 "m" 截走。
const QUALITY_SUFFIXES = [
    { suffix: "m7b5", quality: "m7b5" },
    { suffix: "maj7", quality: "maj7" },
    { suffix: "min7", quality: "min7" },
    { suffix: "maj", quality: "maj" },
    { suffix: "min", quality: "min" },
    { suffix: "dim7", quality: "dim7" },
    { suffix: "dim", quality: "dim" },
    { suffix: "aug", quality: "aug" },
    { suffix: "sus4", quality: "sus4" },
    { suffix: "sus2", quality: "sus2" },
    { suffix: "add9", quality: "add9" },
    { suffix: "m7", quality: "min7" },
    { suffix: "m", quality: "min" },
    { suffix: "6", quality: "6" },
    { suffix: "7", quality: "7" },
    { suffix: "9", quality: "9" },
];
const QUALITY_TO_SUFFIX = {
    maj: "",
    min: "m",
    dim: "dim",
    aug: "aug",
    sus2: "sus2",
    sus4: "sus4",
    "6": "6",
    "7": "7",
    maj7: "maj7",
    min7: "m7",
    m7b5: "m7b5",
    dim7: "dim7",
    "9": "9",
    add9: "add9",
};
/** 解析 "Am7" / "Cmaj7" / "G/B" / "F#m7b5" → root 音级 + quality + 低音音级。 */
export function parseChordLabel(label) {
    const fail = { root: 0, quality: "maj", bass: null };
    const trimmed = (label ?? "").trim();
    const slashIdx = trimmed.indexOf("/");
    const main = slashIdx >= 0 ? trimmed.slice(0, slashIdx) : trimmed;
    const bassText = slashIdx >= 0 ? trimmed.slice(slashIdx + 1) : "";
    const m = /^([A-G])(#|b)?/.exec(main);
    if (!m)
        return fail;
    let root = LETTER_TO_PC[m[1]] ?? 0;
    if (m[2] === "#")
        root = mod12p(root + 1);
    if (m[2] === "b")
        root = mod12p(root - 1);
    const rest = main.slice(m[0].length);
    let quality = "maj";
    for (const { suffix, quality: q } of QUALITY_SUFFIXES) {
        if (rest.includes(suffix)) {
            quality = q;
            break;
        }
    }
    let bass = null;
    const bm = /^([A-G])(#|b)?/.exec(bassText.trim());
    if (bm) {
        let bp = LETTER_TO_PC[bm[1]] ?? 0;
        if (bm[2] === "#")
            bp = mod12p(bp + 1);
        if (bm[2] === "b")
            bp = mod12p(bp - 1);
        bass = bp;
    }
    return { root, quality, bass };
}
/** 由 (root, quality[, bass]) 反构和弦标签，如 (9, min7) → "Am7"。 */
export function chordLabel(root, quality, bass = null) {
    const name = PITCH_NAMES[mod12p(root)] ?? "C";
    let label = name + (QUALITY_TO_SUFFIX[quality] ?? "");
    if (bass != null && mod12p(bass) !== mod12p(root))
        label += "/" + (PITCH_NAMES[mod12p(bass)] ?? "C");
    return label;
}
export function isMinorishQuality(q) {
    return q === "min" || q === "min7" || q === "m7b5" || q === "dim" || q === "dim7";
}
/** chordBlock JSON → 等宽小节和弦段（每个和弦占一小节）。 */
export function chordBlockToSegments(block, beatsPerBar = 4) {
    const ppq = block.ppq ?? TICKS_PER_BEAT;
    const barTicks = Math.max(1, beatsPerBar) * ppq;
    return block.chords.map((c, i) => {
        const parsed = parseChordLabel(c.label);
        const root = mod12p(c.root ?? parsed.root);
        const quality = c.quality ?? parsed.quality;
        return {
            startTick: i * barTicks,
            endTick: (i + 1) * barTicks,
            root,
            quality,
            bass: parsed.bass,
            label: chordLabel(root, quality, parsed.bass),
        };
    });
}
/** ChordSegment[] → chordBlock JSON（供 autoArrange / melodyGen / harmonizer 消费）。 */
export function segmentsToChordBlock(segments, bpm, ppq = TICKS_PER_BEAT) {
    return {
        type: "chordBlock",
        chords: segments.map((s) => ({ label: s.label, root: s.root, quality: s.quality })),
        bpm,
        ppq,
    };
}
