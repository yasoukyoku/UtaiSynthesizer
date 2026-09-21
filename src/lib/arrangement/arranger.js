/**
 * Muno 阶段4「原生自动编曲」——确定性编曲引擎(纯前端,无模型无随机)。
 *
 * 链路:
 *   旋律音符 → estimateKey(全曲调性) + analyzeChords(和弦段,复用阶段3)
 *            → 逐小节查和弦 → 按风格模板生成 4 轨:
 *               · Drums(GM 鼓:底鼓/军鼓/闭镲/开镲/边击/吊镐 + 每 8 小节 fill)
 *               · Bass(根音走句:jazz 行走贝斯 / country 根五交替 / 其余按模板)
 *               · Piano(和弦 voicing:整块 stabs / ballad 琶音 / rnb 延音)
 *               · Pad(和弦铺底,每 padBars 小节换)
 *            → 情绪(力度/密度/八度微调) + swing(奇数八分右移)
 *
 * 节拍假设:模板按 4/4 的 16 步网格定义;其他拍号安全降级 —— 步序号 ≥ 拍数×4
 * 的鼓点/击奏被丢弃,行走贝斯与琶音按真实拍数截断,永不越出小节。
 *
 * 同输入同输出 —— 单测(arranger.test.ts)钉死关键风格骨架;生成落轨走
 * autoArrange.ts(轨道创建 + 音色自动分配在那一层)。
 */
import { TICKS_PER_BEAT } from "../constants";
import { analyzeChords, estimateKey, } from "../analysis/chordAnalysis";
import { MOODS, STEP_TICKS, STYLE_TEMPLATES } from "./styles";
// ── GM 鼓组音高 ─────────────────────────────────────────────────────────────
const KICK = 36;
const SNARE = 38;
const RIM = 37;
const HIHAT = 42;
const OPEN_HAT = 46;
const CRASH = 49;
/** 和弦 → voicing 音级(根音=0)。钢琴/铺底共用;3~4 音,手型稳定。阶段5 和弦 MIDI 亦复用。 */
export const VOICING = {
    maj: [0, 4, 7], min: [0, 3, 7], dim: [0, 3, 6], aug: [0, 4, 8],
    sus2: [0, 2, 7], sus4: [0, 5, 7],
    "6": [0, 4, 7, 9], "7": [0, 4, 7, 10], maj7: [0, 4, 7, 11],
    min7: [0, 3, 7, 10], m7b5: [0, 3, 6, 10], dim7: [0, 3, 6, 9],
    "9": [0, 4, 10, 14], add9: [0, 4, 7, 14],
};
const mod12 = (n) => ((n % 12) + 12) % 12;
/** 把音级 pc 放进 [lo, hi] 音区(取该音区里的那个八度位置)。 */
export function pcInRange(pc, lo, hi) {
    let p = lo + mod12(pc - lo);
    if (p > hi)
        p -= 12;
    return p;
}
/** voicing:根音放 [lo, lo+12] 后直接叠音级(保证严格上行,不超过 lo+26)。 */
function voice(rootPc, quality, lo) {
    const rootP = pcInRange(rootPc, lo, lo + 12);
    return (VOICING[quality] ?? VOICING.maj).map((iv) => rootP + iv);
}
const clampVel = (v) => Math.max(1, Math.min(127, Math.round(v)));
/** 命中 tick 处的和弦段(取 startTick ≤ tick 的最后一段;之前没有则取第一段)。 */
function chordAt(chords, tick) {
    if (chords.length === 0)
        return null;
    let hit = null;
    for (const c of chords) {
        if (c.startTick <= tick)
            hit = c;
        else
            break;
    }
    return hit ?? chords[0];
}
/** 主引擎。输入为空时返回 null(调用层负责 toast)。 */
export function arrange(input) {
    const { notes, style: styleId, mood: moodId, timeSignature } = input;
    if (notes.length === 0)
        return null;
    const style = STYLE_TEMPLATES[styleId];
    const mood = MOODS[moodId];
    const beatsPerBar = Math.max(1, Math.min(16, timeSignature[0] || 4));
    const barTicks = beatsPerBar * TICKS_PER_BEAT;
    const maxStep = beatsPerBar * 4; // 该拍号下 16 步网格的有效步数
    // 硬骨架优先:一键扒带传入的调性/和弦优先于自估(保证与干音严格同调同和声)。
    const key = input.externalKey ?? estimateKey(notes);
    const chords = input.externalChords && input.externalChords.length > 0
        ? input.externalChords
        : analyzeChords(notes, TICKS_PER_BEAT, beatsPerBar).segments;
    // 跨度:首音符所在小节起,末音符结束所在小节止(对齐小节)。
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
    // 摇摆:奇数八分(步 %4===2)右移 swing×拍/6(1 ≈ 三连音)。
    const swingDelay = Math.round(style.swing * (TICKS_PER_BEAT / 6));
    const drums = [];
    const bass = [];
    const piano = [];
    const pad = [];
    // 扩充乐器输出缓冲。
    const guitarArp = [];
    const guitarStrum = [];
    const strings = [];
    const epiano = [];
    const synthPad = [];
    const pluck = [];
    // 扩充乐器的稳定自增 id(同输入同输出)。
    let gaId = 0, gsId = 0, stId = 0, epId = 0, spId = 0, plId = 0;
    let dId = 0, bId = 0, pId = 0, sId = 0;
    for (let bar = 0; bar < bars; bar++) {
        const barTick = startTick + bar * barTicks;
        const cs = chordAt(chords, barTick);
        const root = cs ? cs.root : key.tonic;
        const quality = cs ? cs.quality : "maj";
        const bassPc = cs && cs.bass !== null ? cs.bass : root;
        const nextCs = chordAt(chords, Math.min(endTick - 1, barTick + barTicks));
        const nextRoot = nextCs && nextCs !== cs ? nextCs.root : null;
        // ── Drums ──
        const d = style.drums;
        const fill = bar % 8 === 7 && styleId !== "ballad"; // 每 8 小节句尾 fill
        const pushDrum = (pitch, step, vel) => {
            if (step >= maxStep)
                return; // 非 4/4:越出小节的步直接丢弃
            const off = step % 4 === 2 ? swingDelay : 0;
            drums.push({
                id: `d${++dId}`, tick: barTick + step * STEP_TICKS + off,
                duration: STEP_TICKS, pitch, lyric: "", velocity: clampVel(vel + mood.velocityGain),
            });
        };
        for (const s of d.kick)
            pushDrum(KICK, s, 100);
        for (const s of d.snare)
            pushDrum(SNARE, s, 108);
        if (mood.density !== 0)
            for (const s of d.ghost ?? [])
                pushDrum(SNARE, s, 36);
        let hatSteps = d.hihat;
        if (mood.density === 0)
            hatSteps = hatSteps.filter((s) => s % 4 === 0);
        else if (mood.density === 2 && hatSteps.length > 0)
            hatSteps = Array.from({ length: maxStep }, (_, i) => i);
        for (const s of hatSteps)
            pushDrum(HIHAT, s, s % 4 === 0 ? 74 : 62);
        for (const s of d.openHat ?? [])
            pushDrum(OPEN_HAT, s, 78);
        for (const s of d.rim ?? [])
            pushDrum(RIM, s, 72);
        if (d.crashOnFirstBar && bar === 0)
            pushDrum(CRASH, 0, 112);
        if (fill) {
            // 句尾 fill:末拍军鼓 16 分爬升(整段最后一小节的最后一步换吊镲收束)。
            for (let s = Math.max(0, maxStep - 4); s < maxStep; s++) {
                if (s === maxStep - 1 && bar === bars - 1)
                    pushDrum(CRASH, s, 112);
                else
                    pushDrum(SNARE, s, 90 + (s - (maxStep - 4)) * 10);
            }
        }
        // ── Bass ──
        const bassRootPitch = pcInRange(bassPc, 28, 40); // E1..E2
        if (styleId === "jazz") {
            // 行走贝斯:根 → 三 → 五 →(下个和弦根的半音邻接;无下个则根音高八度)。
            const iv = VOICING[quality] ?? VOICING.maj;
            // 和弦音放根音 ±6 半音内(上行不跳远,下行不低穿)。
            const near = (interval) => bassRootPitch + (interval > 6 ? interval - 12 : interval);
            const third = near(iv[1] ?? 4);
            const fifth = near(7);
            const nextRootPitch = nextRoot !== null ? pcInRange(nextRoot, 28, 40) : null;
            const approach = nextRootPitch !== null
                ? (nextRootPitch > 28 ? nextRootPitch - 1 : nextRootPitch + 1)
                : bassRootPitch + 12;
            const walk = [bassRootPitch, third, fifth, approach];
            for (let i = 0; i < beatsPerBar; i++) {
                bass.push({
                    id: `b${++bId}`, tick: barTick + i * TICKS_PER_BEAT,
                    duration: TICKS_PER_BEAT, pitch: walk[i % walk.length], lyric: "",
                    velocity: clampVel(96 + mood.velocityGain),
                });
            }
        }
        else {
            for (const bn of style.bass.notes) {
                if (bn.offset >= maxStep)
                    continue;
                const off = bn.offset % 4 === 2 ? swingDelay : 0;
                let pitch = bassRootPitch;
                if (bn.useFifth)
                    pitch = bassRootPitch + 7 <= 45 ? bassRootPitch + 7 : bassRootPitch - 5;
                else if (bn.octaveUp)
                    pitch += 12;
                bass.push({
                    id: `b${++bId}`,
                    tick: barTick + bn.offset * STEP_TICKS + off,
                    duration: bn.dur * STEP_TICKS,
                    pitch, lyric: "", velocity: clampVel(96 + mood.velocityGain),
                });
            }
        }
        // ── Piano ──
        const voicing = voice(root, quality, 55); // 根音 C4 附近 [55,67)
        const pushPianoChord = (tick, dur, vel) => {
            for (const p of voicing) {
                piano.push({
                    id: `p${++pId}`, tick, duration: Math.max(1, dur),
                    pitch: p, lyric: "", velocity: clampVel(vel + mood.velocityGain),
                });
            }
        };
        const pk = style.piano;
        if (pk.kind === "hits") {
            for (const s of pk.steps ?? []) {
                if (s >= maxStep)
                    continue;
                const off = s % 4 === 2 ? swingDelay : 0;
                pushPianoChord(barTick + s * STEP_TICKS + off, STEP_TICKS * 2, 90);
            }
        }
        else if (pk.kind === "quarterArp") {
            // ballad:每拍一个 voicing 音(不足拍数补根音高八度,超出截断)。
            const seq = voicing.length >= beatsPerBar
                ? voicing
                : [...voicing, ...Array.from({ length: beatsPerBar - voicing.length }, (_, i) => voicing[i % voicing.length] + 12)];
            for (let i = 0; i < beatsPerBar; i++) {
                piano.push({
                    id: `p${++pId}`, tick: barTick + i * TICKS_PER_BEAT,
                    duration: TICKS_PER_BEAT, pitch: seq[i], lyric: "",
                    velocity: clampVel((i === 0 ? 92 : 84) + mood.velocityGain),
                });
            }
        }
        else {
            // sustain:整小节延音。
            pushPianoChord(barTick, barTicks, 82);
        }
        // ── Pad(和弦铺底)── 每 padBars 小节起一次,长度铺满到下个边界。
        if (bar % style.padBars === 0) {
            const padLen = Math.min(style.padBars, bars - bar) * barTicks;
            for (const iv of VOICING[quality] ?? VOICING.maj) {
                pad.push({
                    id: `s${++sId}`, tick: barTick,
                    duration: Math.max(1, padLen - STEP_TICKS),
                    pitch: pcInRange(mod12(root + iv), 55, 67) + mood.padOctave * 12,
                    lyric: "", velocity: clampVel(68 + mood.velocityGain),
                });
            }
        }
        // ── 扩充乐器(通用生成,全部自动对齐本小节 root/quality 骨架)──
        const step8 = TICKS_PER_BEAT / 2; // 八分音符 tick
        // 1) 木吉他分解:八分拨弦,根-三-五-高根-五-三 波浪型,低把位。
        {
            const gv = voice(root, quality, 50);
            const arpSeq = [gv[0], gv[1], gv[2], (gv[0] ?? 0) + 12, gv[2], gv[1]];
            const count = beatsPerBar * 2;
            for (let i = 0; i < count; i++) {
                const pitch = arpSeq[i % arpSeq.length];
                if (pitch == null)
                    continue;
                guitarArp.push({
                    id: `ga${++gaId}`, tick: barTick + i * step8,
                    duration: step8, pitch, lyric: "",
                    velocity: clampVel(72 + mood.velocityGain),
                });
            }
        }
        // 2) 扫弦人性化:扫弦方向(↓/↑)决定 voicing 音的 tick 顺序,每个音约 10ms spread,
        //    力度从起始音向末音线性衰减 (-9dB), 每个音有 ±4 velocity + ±2 tick 微抖。
        {
            const gv = voice(root, quality, 50);
            const strumPos = [0, Math.min(beatsPerBar * 2, maxStep - 2)];
            const spreadTick = Math.max(8, Math.round(STEP_TICKS * 0.35)); // ~35ms @120BPM
            const strumDur = STEP_TICKS * 2;
            for (let si = 0; si < strumPos.length; si++) {
                const startStep = strumPos[si];
                if (startStep + 1 >= maxStep)
                    continue;
                const dir = si % 2 === 0 ? "down" : "up";
                const ordered = dir === "up" ? [...gv].sort((a, b) => b - a) : gv;
                const baseTick = barTick + startStep * STEP_TICKS;
                const baseVel = 72 + mood.velocityGain;
                ordered.forEach((p, i) => {
                    const tick = baseTick + i * spreadTick;
                    const velFalloff = 1 - (i / Math.max(1, ordered.length - 1)) * 0.25;
                    const jitter = (((bar * 37 + si * 7 + i * 13) % 7) - 3);
                    const vel = clampVel(baseVel * velFalloff + jitter);
                    guitarStrum.push({
                        id: `gs${++gsId}`, tick: tick + (((bar * 11 + si * 5 + i * 3) % 3) - 1),
                        duration: Math.max(2, strumDur - i * 2), pitch: p, lyric: "",
                        velocity: vel,
                    });
                });
            }
        }
        // 3) 弦乐组长音:每 2 小节换弓,低中音区,柔和铺底。
        if (bar % 2 === 0) {
            const sv = voice(root, quality, 48);
            const len = Math.min(2, bars - bar) * barTicks;
            for (const p of sv) {
                strings.push({
                    id: `st${++stId}`, tick: barTick,
                    duration: Math.max(1, len - STEP_TICKS), pitch: p, lyric: "",
                    velocity: clampVel(60 + mood.velocityGain),
                });
            }
        }
        // 4) 电钢琴:每拍一个柔和 voicing 音(quarterArp 型),比钢琴更轻。
        {
            const ev = voice(root, quality, 55);
            for (let i = 0; i < beatsPerBar; i++) {
                const pitch = ev[i % ev.length] ?? ev[0];
                if (pitch == null)
                    continue;
                epiano.push({
                    id: `ep${++epId}`, tick: barTick + i * TICKS_PER_BEAT,
                    duration: TICKS_PER_BEAT, pitch, lyric: "",
                    velocity: clampVel(74 + mood.velocityGain),
                });
            }
        }
        // 5) 合成铺底:每 2 小节,高八度宽 voicing,空间感。
        if (bar % 2 === 0) {
            const pv = voice(root, quality, 60);
            const len = Math.min(2, bars - bar) * barTicks;
            for (const p of pv) {
                synthPad.push({
                    id: `sp${++spId}`, tick: barTick,
                    duration: Math.max(1, len - STEP_TICKS), pitch: p + 12, lyric: "",
                    velocity: clampVel(56 + mood.velocityGain),
                });
            }
        }
        // 6) Pluck 十六分琶音:voicing 上行循环,短促颗粒感(稀疏情绪只留正拍)。
        {
            const pv = voice(root, quality, 60);
            for (let s = 0; s < maxStep; s++) {
                if (mood.density === 0 && s % 4 !== 0)
                    continue;
                const pitch = pv[s % pv.length] ?? pv[0];
                if (pitch == null)
                    continue;
                pluck.push({
                    id: `pl${++plId}`, tick: barTick + s * STEP_TICKS,
                    duration: STEP_TICKS, pitch, lyric: "",
                    velocity: clampVel(66 + mood.velocityGain),
                });
            }
        }
    }
    // ── 自动旋律生成 (可选, 给没旋律的用户一条 lead 轨) ──
    // 策略: 每个 bar 的 chord tones (root / 3rd / 5th / 7th) + 经过音
    //       用确定性 seed = 风格+和弦序列 hash, 所以同输入→同输出 (非随机)
    const melody = [];
    // 确定性 seed: 基于 chord 序列 + style id 拼字符串算 hash (同输入 → 同输出)
    let _seed = 0;
    const seedStr = style.id + ":" + chords.map((c) => String(c.root) + c.quality).join("|");
    for (let i = 0; i < seedStr.length; i++)
        _seed = (_seed * 31 + seedStr.charCodeAt(i)) | 0;
    const melRand = () => { _seed = (_seed * 1664525 + 1013904223) | 0; return ((_seed >>> 0) % 10000) / 10000; };
    // 大调音阶 (Ionian) 或 自然小调 —— 必须按 key.tonic 移调到本调,
    // 否则经过音会永远落在 C 调,非 C 调歌曲出现离调音。
    const scalePcs = (key.major ? [0, 2, 4, 5, 7, 9, 11] : [0, 2, 3, 5, 7, 8, 10])
        .map((iv) => mod12(key.tonic + iv));
    for (let bar = 0; bar < bars; bar++) {
        const barTick = startTick + bar * barTicks;
        const cs = chordAt(chords, barTick);
        // 和弦内音严格按和弦性质取(小三→小三度、大七→大七度),避免在小和弦上弹出大三度错音。
        const chordQuality = cs ? cs.quality : (key.major ? "maj" : "min");
        const chordRootPc = cs ? cs.root : key.tonic;
        const chordPcs = (VOICING[chordQuality] ?? VOICING.maj)
            .slice(0, 4)
            .map((iv) => mod12(chordRootPc + iv));
        const notesPerBar = 6 + Math.floor(melRand() * 4); // 6-9 notes per bar
        for (let n = 0; n < notesPerBar; n++) {
            const nt = barTick + Math.floor((barTicks / notesPerBar) * n);
            if (nt >= barTick + barTicks)
                break;
            const isChordTone = melRand() < 0.65;
            const pc = (isChordTone
                ? chordPcs[Math.floor(melRand() * chordPcs.length)]
                : scalePcs[Math.floor(melRand() * scalePcs.length)]) ?? 0;
            const octave = melRand() < 0.25 ? 1 : 0;
            const pitch = pc + 60 + octave * 12;
            const durRoll = melRand();
            const durTicks = durRoll < 0.25 ? 24 : durRoll < 0.75 ? 48 : 96;
            melody.push({
                id: 'mel' + bar + '_' + n,
                tick: nt,
                duration: durTicks,
                pitch,
                lyric: "",
                velocity: clampVel(70 + Math.floor(melRand() * 20)),
            });
        }
    }
    // === OPT1: 音量/音区自动平衡 ===
    // 只在多轨场景 (≥5 轨) 下启用, 单轨/少轨保留原始 velocity 不动。
    const allParts = { drums, bass, piano, pad, melody, guitarArp, guitarStrum, epiano, strings, synthPad, pluck };
    const activeCount = Object.values(allParts).filter((a) => a.length > 0).length;
    if (activeCount >= 5) {
        const BALANCE_BIG = {
            drums: 95, bass: 82, piano: 68, pad: 55, melody: 88,
            guitarArp: 65, guitarStrum: 68, epiano: 66, strings: 58, synthPad: 52, pluck: 62,
        };
        const BALANCE_SMALL = {
            drums: 100, bass: 90, piano: 85, pad: 72, melody: 92,
            guitarArp: 88, guitarStrum: 88, epiano: 85, strings: 75, synthPad: 65, pluck: 85,
        };
        const isBigEnsemble = activeCount >= 7;
        const target = isBigEnsemble ? BALANCE_BIG : BALANCE_SMALL;
        Object.keys(allParts).forEach((k) => {
            if (k === "drums")
                return; // drums 保留原始 velocity (MIDI punchy), 永远不做归一化
            const part = allParts[k];
            if (!part.length)
                return;
            const tgt = target[k] ?? 72;
            for (const n of part) {
                const delta = tgt - n.velocity;
                n.velocity = clampVel(n.velocity + delta * 0.5);
            }
        });
        // 音区防堆叠:贝斯在 ≥9 轨场景下,50% 低把位音 +12
        if (activeCount >= 9) {
            for (const n of bass) {
                const h = ((n.id.charCodeAt(0) * 7 + n.tick + n.pitch) % 3);
                if (h === 0 && n.pitch < 40)
                    n.pitch = n.pitch + 12;
            }
        }
    }
    // 稳定排序:同轨按 tick 再按 pitch,保证任何环境下列表序一致(同输入同输出)。
    const byTickPitch = (a, b) => a.tick - b.tick || a.pitch - b.pitch;
    drums.sort(byTickPitch);
    bass.sort(byTickPitch);
    piano.sort(byTickPitch);
    pad.sort(byTickPitch);
    melody.sort(byTickPitch);
    guitarArp.sort(byTickPitch);
    guitarStrum.sort(byTickPitch);
    epiano.sort(byTickPitch);
    strings.sort(byTickPitch);
    synthPad.sort(byTickPitch);
    pluck.sort(byTickPitch);
    return {
        key, chords, startTick, endTick, bars,
        drums, bass, piano, pad, melody,
        guitarArp, guitarStrum, strings, epiano, synthPad, pluck,
    };
}
