import { TICKS_PER_BEAT } from "../constants";
import { mod12p } from "./common";
import { chordLabel } from "./chords";
function transposeSegment(s, semis) {
    const root = mod12p(s.root + semis);
    const bass = s.bass == null ? null : mod12p(s.bass + semis);
    return { ...s, root, bass, label: chordLabel(root, s.quality, bass) };
}
export function editStructure(segments, o) {
    if (segments.length === 0)
        return [];
    const ppq = o.ppq ?? TICKS_PER_BEAT;
    const barTicks = Math.max(1, o.beatsPerBar ?? 4) * ppq;
    const totalTicks = Math.max(...segments.map((s) => s.endTick));
    const sectionTicks = Math.max(1, o.sectionBars) * barTicks;
    // 尾段边界：最后一节 sectionBars 小节。
    const boundary = Math.max(0, totalTicks - sectionTicks);
    const head = segments.filter((s) => s.startTick < boundary);
    const tail = segments.filter((s) => s.startTick >= boundary);
    switch (o.op) {
        case "repeatTail": {
            // 尾段原样复制追加（重复终段），tick 整体后移一个段长。
            const shifted = tail.map((s) => ({ ...s, startTick: s.startTick + sectionTicks, endTick: s.endTick + sectionTicks }));
            return [...head, ...tail, ...shifted];
        }
        case "dropTail": {
            // 删除尾段；若整首短于一段则原样返回（避免空输出）。
            return head.length > 0 ? head : segments;
        }
        case "transposeTail": {
            // 尾段整体移调（调内离调均可，root/bass 音级取模）。
            return [...head, ...tail.map((s) => transposeSegment(s, o.semitones ?? 2))];
        }
        case "lengthenTail": {
            // 最后一个和弦延长一个段长（终止延音）。
            return segments.map((s, i) => (i === segments.length - 1 ? { ...s, endTick: s.endTick + sectionTicks } : s));
        }
    }
}
