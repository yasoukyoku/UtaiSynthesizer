/**
 * Muno 阶段5「和弦 MIDI 生成（配和声）」——原生和声化引擎（纯前端，无模型，确定性）。
 *
 * 链路：
 *   旋律音符 → analyzeChords（复用阶段3，半拍窗分析）
 *            → 对齐到「每小节和弦数」网格（1 = 整小节，2 = 半小节）
 *            → 风格变换（MIDI-SAG 五风格的确定性原生语义，见下）
 *            → 块状 voicing（根位原位和弦，一段一块，时长铺满段长 − 1 步防同音重触发）
 *
 * 风格语义（同输入同输出）：
 *   · NOCONSTRAINT 原样分析结果（仅对齐网格）——最尊重旋律的实际音响。
 *   · POP_STANDARD 吸附到调内顺阶和弦（大调 I ii iii IV V vi vii° / 自然小调 i ii° III iv v VI VII）。
 *   · POP_COMPLEX  保留分析结果，三和弦按级位扩成七和弦（I/IV→maj7、V→7、ii/iii/vi→min7、
 *                  vii°→m7b5）；扩音与该小节旋律构成小九度冲突（±1 半音）时放弃扩展。
 *   · DARK          强制小调性（大调 → 关系小调），吸附小调顺阶，i/iv/v → min7（含冲突检查）。
 *   · RANDB         种子随机（mulberry32 + 音符内容哈希）：每格从与该格旋律有共同音的
 *                  顺阶和弦里按支持度前 3 随机取一 —— 随机但永远音乐性成立，且可复现。
 *
 * 双输出：segments（和弦行显示 + 和弦标签）与 notes（可播和声轨）。
 * 落轨/音色分配/和弦行写入/toast 走动作层（autoArrange.ts 的 runChordMidi）。
 */
import { TICKS_PER_BEAT } from "../constants";
import { analyzeChords, chordLabel, estimateKey, } from "../analysis/chordAnalysis";
import { STEP_TICKS } from "./styles";
import { VOICING, pcInRange } from "./arranger";
export const CHORD_MIDI_STYLES = [
    "POP_STANDARD", "POP_COMPLEX", "DARK", "RANDB", "NOCONSTRAINT",
];
// ── 顺阶表 ──────────────────────────────────────────────────────────────────
const MAJOR_DEGREES = [0, 2, 4, 5, 7, 9, 11];
const MAJOR_QUALITIES = ["maj", "min", "min", "maj", "maj", "min", "dim"];
const MINOR_DEGREES = [0, 2, 3, 5, 7, 8, 10];
const MINOR_QUALITIES = ["min", "dim", "maj", "min", "min", "maj", "maj"];
const PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const mod12 = (n) => ((n % 12) + 12) % 12;
/** 音级圆距离（0..6）。 */
const circDist = (a, b) => {
    const d = Math.abs(mod12(a - b));
    return Math.min(d, 12 - d);
};
/** 离根音最近的顺阶级位序号（平票取低级位，稳定）。 */
function nearestDegree(root, degrees) {
    let best = 0;
    let bestD = circDist(root, degrees[0]);
    for (let i = 1; i < degrees.length; i++) {
        const d = circDist(root, degrees[i]);
        if (d < bestD) {
            best = i;
            bestD = d;
        }
    }
    return best;
}
/** 解析 "auto"/"C"/"Cm"/"f#"… → 调性；无效 → null（回落自动检测）。 */
function parseKeyOption(key) {
    const k = key.trim().toLowerCase();
    if (!k || k === "auto")
        return null;
    const minor = k.endsWith("m");
    const name = (minor ? k.slice(0, -1) : k);
    const idx = PITCH_NAMES.findIndex((p) => p.toLowerCase() === name);
    if (idx < 0)
        return null;
    return { tonic: idx, major: !minor };
}
/** mulberry32：种子 PRNG（RANDB 用；同输入同种子 → 同随机序列）。 */
function mulberry32(seed) {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6d2b79f5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}
/** 音符内容哈希（FNV-1a）→ RANDB 的种子来源。 */
function hashNotes(notes) {
    let h = 2166136261;
    for (const n of notes) {
        h ^= n.tick;
        h = Math.imul(h, 16777619);
        h ^= n.duration;
        h = Math.imul(h, 16777619);
        h ^= n.pitch;
        h = Math.imul(h, 16777619);
    }
    return h >>> 0;
}
/** 主引擎。输入为空时返回 null（调用层负责 toast）。 */
export function generateChordMidi(input) {
    const { notes, style, chordsPerBar, timeSignature } = input;
    if (notes.length === 0)
        return null;
    const beatsPerBar = Math.max(1, Math.min(16, timeSignature[0] || 4));
    const barTicks = beatsPerBar * TICKS_PER_BEAT;
    const cells = Math.max(1, Math.min(2, chordsPerBar));
    const cellTicks = barTicks / cells;
    // 调性：显式指定优先，否则旋律检测。DARK 强制小调（大调 → 关系小调）。
    const est = estimateKey(notes);
    const forced = parseKeyOption(input.key);
    const dark = style === "DARK";
    let tonic = forced ? forced.tonic : est.tonic;
    let major = forced ? forced.major : est.major;
    if (dark && major) {
        tonic = (tonic + 9) % 12;
        major = false;
    }
    const key = dark
        ? { tonic, major: false, confidence: est.confidence, label: `${PITCH_NAMES[tonic]}m` }
        : forced
            ? { tonic: forced.tonic, major: forced.major, confidence: 1, label: `${PITCH_NAMES[forced.tonic]}${forced.major ? "" : "m"}` }
            : est;
    // 跨度：对齐小节（与 arrange 同规则）。
    let firstTick = Infinity;
    let lastTick = 0;
    for (const n of notes) {
        if (n.tick < firstTick)
            firstTick = n.tick;
        const end = n.tick + Math.max(0, n.duration);
        if (end > lastTick)
            lastTick = end;
    }
    const startTick = Math.floor(firstTick / barTicks) * barTicks;
    const endTick = Math.max(startTick + barTicks, Math.ceil(lastTick / barTicks) * barTicks);
    const bars = Math.round((endTick - startTick) / barTicks);
    // 半拍窗分析（最细粒度），随后对齐到网格。
    const analysis = analyzeChords(notes, TICKS_PER_BEAT, beatsPerBar, { windowsPerBeat: 2 });
    const chordAt = (tick) => {
        let hit = null;
        for (const c of analysis.segments) {
            if (c.startTick <= tick)
                hit = c;
            else
                break;
        }
        return hit ?? analysis.segments[0] ?? null;
    };
    const degrees = major ? MAJOR_DEGREES : MINOR_DEGREES;
    const qualities = major ? MAJOR_QUALITIES : MINOR_QUALITIES;
    // 顺阶音级的绝对音级（主音 + 级数音程）——吸附/查询都在绝对域上做。
    const scalePcs = degrees.map((d) => mod12(tonic + d));
    const rng = mulberry32(hashNotes(notes));
    /** 网格 [start, end) 内活跃的旋律音级集合。 */
    const cellPcs = (start, end) => {
        const s = new Set();
        for (const n of notes) {
            if (n.tick >= end || n.tick + Math.max(0, n.duration) <= start)
                continue;
            s.add(mod12(n.pitch));
        }
        return s;
    };
    const pcsOfChord = (root, quality) => (VOICING[quality] ?? VOICING.maj).map((iv) => mod12(root + iv));
    /**
     * 三和弦 → 七和弦（按级位）。扩音与该格旋律构成小九度冲突（±1 半音）时放弃扩展
     * —— 例外：冲突的旋律音本身是扩展后和弦的和弦音（如 Cmaj7 配旋律 C），
     * 和弦内色彩不算冲突，照常扩展。
     * @param extendMaj POP_COMPLEX 扩大三和弦（I/IV→maj7、V→7）；DARK 只加深小三和弦（i/iv/v→min7）。
     */
    const extendTriad = (root, quality, deg, pcs, extendMaj) => {
        let next = null;
        if (major) {
            if (extendMaj && (deg === 0 || deg === 3) && quality === "maj")
                next = "maj7";
            else if (extendMaj && deg === 4 && quality === "maj")
                next = "7";
            else if (deg === 1 || deg === 2 || deg === 5) {
                if (quality === "min")
                    next = "min7";
            }
            else if (deg === 6) {
                if (quality === "dim")
                    next = "m7b5";
            }
        }
        else {
            if (deg === 0 || deg === 3 || deg === 4) {
                if (quality === "min")
                    next = "min7";
            }
            else if (extendMaj && (deg === 2 || deg === 5 || deg === 6)) {
                if (quality === "maj")
                    next = "maj7";
            }
            else if (deg === 1) {
                if (quality === "dim")
                    next = "m7b5";
            }
        }
        if (next === null)
            return quality;
        const base = VOICING[quality] ?? VOICING.maj;
        const ext = VOICING[next] ?? VOICING.maj;
        const extPcs = new Set(ext.map((iv) => mod12(root + iv)));
        for (const iv of ext) {
            if (base.includes(iv))
                continue;
            const a = mod12(root + iv);
            for (const m of pcs) {
                const d = Math.abs(a - m);
                if ((d === 1 || d === 11) && !extPcs.has(m))
                    return quality; // 非和弦音的小九度冲突 → 保三和弦
            }
        }
        return next;
    };
    // 逐格定和弦（网格对齐，相邻同和弦合并）。
    const cellsOut = [];
    for (let bar = 0; bar < bars; bar++) {
        for (let c = 0; c < cells; c++) {
            const start = startTick + bar * barTicks + c * cellTicks;
            const end = start + cellTicks;
            const pcs = cellPcs(start, end);
            const an = chordAt(start);
            let root = an ? an.root : tonic;
            let quality = an ? an.quality : major ? "maj" : "min";
            const deg = nearestDegree(root, scalePcs);
            if (style === "POP_STANDARD") {
                root = scalePcs[deg];
                quality = qualities[deg];
            }
            else if (style === "DARK") {
                root = scalePcs[deg];
                quality = qualities[deg];
                quality = extendTriad(root, quality, deg, pcs, false);
            }
            else if (style === "POP_COMPLEX") {
                quality = extendTriad(root, quality, deg, pcs, true);
            }
            else if (style === "RANDB") {
                const ranked = scalePcs
                    .map((pc, i) => ({ pc, i, overlap: pcsOfChord(pc, qualities[i]).filter((p) => pcs.has(p)).length }))
                    .filter((r) => r.overlap > 0)
                    .sort((a, b) => b.overlap - a.overlap);
                if (ranked.length > 0) {
                    const pick = ranked[Math.min(ranked.length - 1, Math.floor(rng() * Math.min(3, ranked.length)))];
                    root = pick.pc;
                    quality = qualities[pick.i];
                }
                else {
                    // 无共同音候选 → 顺阶吸附兜底。
                    root = scalePcs[deg];
                    quality = qualities[deg];
                }
            }
            // NOCONSTRAINT：分析结果原样。
            const prev = cellsOut[cellsOut.length - 1];
            if (!(prev && prev.root === root && prev.quality === quality)) {
                cellsOut.push({ start, root, quality });
            }
        }
    }
    // 相邻同和弦已合并 —— 段边界 = 下一段起点（末段铺到 endTick）。
    const segments = cellsOut.map((cell, i) => ({
        startTick: cell.start,
        endTick: i + 1 < cellsOut.length ? cellsOut[i + 1].start : endTick,
        root: cell.root,
        quality: cell.quality,
        bass: null,
        label: chordLabel(cell.root, cell.quality, null),
    }));
    // 块状 voicing：根位形态（三和弦根音限 [55,62]、七和弦限 [50,58]，叠音程后全部落在
    // [50,69] = C3..C#5 钢琴伴奏区，严格上行不压人声主旋律），一段一块（时长 = 段长 − 1 步）。
    const voiced = [];
    let id = 0;
    for (const seg of segments) {
        const ivs = VOICING[seg.quality] ?? VOICING.maj;
        const maxIv = ivs[ivs.length - 1];
        const rootP = maxIv > 7 ? pcInRange(seg.root, 50, 58) : pcInRange(seg.root, 55, 62);
        const dur = Math.max(1, seg.endTick - seg.startTick - STEP_TICKS);
        for (const iv of ivs) {
            voiced.push({
                id: `c${++id}`,
                tick: seg.startTick,
                duration: dur,
                pitch: rootP + iv,
                lyric: "",
                velocity: 82,
            });
        }
    }
    return { key, segments, notes: voiced, startTick, endTick, bars };
}
