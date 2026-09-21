/**
 * 旋律重构四法 + 旋律相似度度量（规划 2-4 / 7.1 法一~法四，对应 4.2 旋律层面原创化）。
 *
 * 相似度度量（验收指标）：音程 3-gram Jaccard + 音高 3-gram Jaccard + 音程序列 Pearson 相关
 * 三者平均，值域 [0,1]。验收断言：contourMorph 强度 60% 时，输出与输入相似度落在 35–45%。
 *
 * 四法（法五 = melodyGen 重做，见 engine.ts）：
 *   法一 degreeSwap        音阶内度数置换（保轮廓、换音级）
 *   法二 rhythmRestructure 节奏移位与切分重组（保音高序列）
 *   法三 contourMorph      轮廓保形（保上下行走向、改音程幅度）
 *   法四 motifDevelop      动机发展（模进/倒影/逆行/增值/减值）
 */
import { TICKS_PER_BEAT } from "../constants";
import type { ChordAnalysisNote, KeyEstimate } from "../analysis/chordAnalysis";
import { estimateKey } from "../analysis/chordAnalysis";
import { makeRng, rngInt, rngChance } from "./rng";
import { clamp01, mod12p, byTick, SINGABLE_LOW, SINGABLE_HIGH } from "./common";

// ══ 相似度度量 ══

function intervalsOf(notes: ChordAnalysisNote[]): number[] {
  const out: number[] = [];
  for (let i = 1; i < notes.length; i++) out.push(notes[i]!.pitch - notes[i - 1]!.pitch);
  return out;
}

function ngramSet(seq: number[], n: number): Set<string> {
  const s = new Set<string>();
  for (let i = 0; i + n <= seq.length; i++) s.add(seq.slice(i, i + n).join(","));
  return s;
}

function jaccard(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 && b.size === 0) return 1;
  let inter = 0;
  for (const g of a) if (b.has(g)) inter++;
  return inter / (a.size + b.size - inter);
}

function pearson(x: number[], y: number[]): number {
  const len = Math.min(x.length, y.length);
  if (len < 2) return 0;
  let sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
  for (let i = 0; i < len; i++) {
    sx += x[i]!;
    sy += y[i]!;
    sxx += x[i]! * x[i]!;
    syy += y[i]! * y[i]!;
    sxy += x[i]! * y[i]!;
  }
  const cov = sxy / len - (sx / len) * (sy / len);
  const vx = sxx / len - (sx / len) ** 2;
  const vy = syy / len - (sy / len) ** 2;
  if (vx <= 1e-9 || vy <= 1e-9) return 0;
  return cov / Math.sqrt(vx * vy);
}

/** 轮廓符号序列（Parsons 编码）：+1 上行 / 0 同度 / -1 下行。 */
function signsOf(notes: ChordAnalysisNote[]): number[] {
  const out: number[] = [];
  for (let i = 1; i < notes.length; i++) out.push(Math.sign(notes[i]!.pitch - notes[i - 1]!.pitch));
  return out;
}

/** 旋律相似度 [0,1]：1 = 完全一致，0 = 完全无关。音高项用相对首音的音高（转置不变）。 */
export function melodySimilarity(a: ChordAnalysisNote[], b: ChordAnalysisNote[]): number {
  if (a.length < 2 || b.length < 2) return a.length === b.length ? 1 : 0;
  const ia = intervalsOf(a);
  const ib = intervalsOf(b);
  const jacInt = jaccard(ngramSet(ia, 3), ngramSet(ib, 3));
  const relA = a.map((n) => n.pitch - a[0]!.pitch);
  const relB = b.map((n) => n.pitch - b[0]!.pitch);
  const jacPitch = jaccard(ngramSet(relA, 3), ngramSet(relB, 3));
  const corr = Math.max(0, pearson(signsOf(a), signsOf(b)));
  return (jacInt + jacPitch + corr) / 3;
}

// ══ 法一 degreeSwap — 音阶内度数置换 ══

const MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11];
const MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10];

export interface DegreeSwapOptions {
  /** 0-100，每个音符被替换的概率。 */
  density: number;
  seed?: number;
  /** 不传则 estimateKey 自动估计。 */
  key?: KeyEstimate | null;
  low?: number;
  high?: number;
}

export function degreeSwap(notes: ChordAnalysisNote[], o: DegreeSwapOptions): ChordAnalysisNote[] {
  if (notes.length === 0) return [];
  const density = clamp01(o.density / 100);
  const rng = makeRng(o.seed);
  const key = o.key ?? estimateKey(notes);
  const scale = key.major ? MAJOR_SCALE : MINOR_SCALE;
  const pcs = new Set(scale.map((d) => mod12p(d + key.tonic)));
  const low = o.low ?? SINGABLE_LOW;
  const high = o.high ?? SINGABLE_HIGH;
  return notes.map((n, i) => {
    // 首音是调性锚点，永不替换。
    if (i === 0 || !rngChance(rng, density)) return { ...n };
    const candidates: number[] = [];
    for (let d = -4; d <= 4; d++) {
      if (d === 0) continue;
      const p = n.pitch + d;
      if (p < low || p > high) continue;
      if (!pcs.has(mod12p(p))) continue;
      candidates.push(p);
    }
    if (candidates.length === 0) return { ...n };
    return { ...n, pitch: candidates[Math.floor(rng() * candidates.length)]! };
  });
}

// ══ 法二 rhythmRestructure — 节奏移位与切分重组（保音高序列） ══

export interface RhythmRestructureOptions {
  /** 0-100，小节被重组的概率与幅度来源。 */
  strength: number;
  beatsPerBar?: number;
  seed?: number;
}

export function rhythmRestructure(notes: ChordAnalysisNote[], o: RhythmRestructureOptions): ChordAnalysisNote[] {
  if (notes.length === 0) return [];
  const strength = clamp01(o.strength / 100);
  const rng = makeRng(o.seed);
  const barTicks = Math.max(1, o.beatsPerBar ?? 4) * TICKS_PER_BEAT;
  const step16 = TICKS_PER_BEAT / 4;
  const sorted = [...notes].sort(byTick);
  const out: ChordAnalysisNote[] = [];
  let i = 0;
  while (i < sorted.length) {
    const barIdx = Math.floor(sorted[i]!.tick / barTicks);
    let j = i;
    while (j < sorted.length && Math.floor(sorted[j]!.tick / barTicks) === barIdx) j++;
    const bar = sorted.slice(i, j);
    if (rng() < strength && bar.length >= 2) {
      const op = rngInt(rng, 0, 2);
      if (op === 0) {
        // 合并相邻两音（保音高首位，时值相加）— 音符数 -1。
        const k = rngInt(rng, 0, bar.length - 2);
        const a = bar[k]!;
        const b = bar[k + 1]!;
        out.push({ ...a, duration: b.tick + b.duration - a.tick });
        for (let m = 0; m < bar.length; m++) if (m !== k && m !== k + 1) out.push({ ...bar[m]! });
      } else if (op === 1) {
        // 拆分一个较长音为 60/40（同音高重复）— 音符数 +1。
        const k = rngInt(rng, 0, bar.length - 1);
        const a = bar[k]!;
        const d1 = Math.round(a.duration * 0.6);
        if (a.duration >= 2 * step16) {
          out.push({ ...a, duration: d1 });
          out.push({ ...a, tick: a.tick + d1, duration: a.duration - d1 });
        } else {
          out.push({ ...a });
        }
        for (let m = 0; m < bar.length; m++) if (m !== k) out.push({ ...bar[m]! });
      } else {
        // 小节内 ±十六分移位（不跨小节线）。
        const k = rngInt(rng, 0, bar.length - 1);
        const a = bar[k]!;
        const dir = rng() < 0.5 ? -1 : 1;
        const shifted = a.tick + dir * step16;
        const lo = barIdx * barTicks;
        const hi = lo + barTicks - 1;
        out.push({ ...a, tick: Math.min(hi, Math.max(lo, shifted)) });
        for (let m = 0; m < bar.length; m++) if (m !== k) out.push({ ...bar[m]! });
      }
    } else {
      for (const n of bar) out.push({ ...n });
    }
    i = j;
  }
  return out.sort(byTick);
}

// ══ 法三 contourMorph — 轮廓保形 ══

export interface ContourMorphOptions {
  /** 0-100。验收：60 时相似度 ∈ [35%, 45%]。 */
  strength: number;
  /** 保留高潮音（原旋律最高音不被改动）。 */
  keepClimax: boolean;
  seed?: number;
  low?: number;
  high?: number;
}

export function contourMorph(notes: ChordAnalysisNote[], o: ContourMorphOptions): ChordAnalysisNote[] {
  if (notes.length < 2) return notes.map((n) => ({ ...n }));
  const strength = clamp01(o.strength / 100);
  if (strength === 0) return notes.map((n) => ({ ...n }));
  const rng = makeRng(o.seed);
  const low = o.low ?? SINGABLE_LOW;
  const high = o.high ?? SINGABLE_HIGH;
  const orig = notes.map((n) => n.pitch);
  const pitches = [...orig];
  let climaxIdx = 0;
  for (let i = 1; i < orig.length; i++) if (orig[i]! > orig[climaxIdx]!) climaxIdx = i;
  const maxAlter = 1 + 5 * strength;
  for (let i = 0; i + 1 < orig.length; i++) {
    // 音程符号取自原始轮廓：逐音构建时若用已修改的当前音，符号会被漂移翻转。
    const iv = orig[i + 1]! - orig[i]!;
    if (iv === 0) continue; // 同度保持：轮廓 "=" 不变
    const sign = iv > 0 ? 1 : -1;
    if (o.keepClimax && i + 1 === climaxIdx) {
      // 高潮音锚定不动：把当前音钳到高潮音的正确一侧，保住进入高潮的走向。
      if (sign > 0) pitches[i] = Math.min(pitches[i]!, orig[i + 1]! - 1);
      else pitches[i] = Math.max(pitches[i]!, orig[i + 1]! + 1);
      continue;
    }
    const mag = Math.abs(iv);
        // strength = 每个音程被重构的概率（打 0.75 折，让 60% 强度的相似度落在验收带 35–45%）；
        // 未被选中的音程相对当前音保持原音程。
        let newMag = mag;
        if (rng() < strength * 0.75) {
      newMag = Math.round(mag + (rng() * 2 - 1) * maxAlter);
      if (newMag === mag) {
        const dir = mag >= 14 ? -1 : mag === 1 ? 1 : rng() < 0.5 ? 1 : -1;
        newMag = mag + dir;
      }
      newMag = Math.min(14, Math.max(1, newMag));
    }
    let next = pitches[i]! + sign * newMag;
    // 音域钳制：越界取边界；贴边导致的同度塌缩是合法输出（相似度与符号测试均容忍）。
    if (next > high) next = high;
    if (next < low) next = low;
    pitches[i + 1] = next;
  }
  return notes.map((n, i) => ({ ...n, pitch: pitches[i]! }));
}

// ══ 法四 motifDevelop — 动机发展 ══

export type MotifTechnique = "sequence" | "invert" | "retrograde" | "augment" | "diminish" | "mixed";

export interface MotifDevelopOptions {
  technique: MotifTechnique;
  /** 动机长度（小节），默认 2。 */
  motifBars?: number;
  beatsPerBar?: number;
  seed?: number;
}

export function motifDevelop(notes: ChordAnalysisNote[], o: MotifDevelopOptions): ChordAnalysisNote[] {
  if (notes.length === 0) return [];
  const barTicks = Math.max(1, o.beatsPerBar ?? 4) * TICKS_PER_BEAT;
  const motifBars = Math.max(1, o.motifBars ?? 2);
  const groupTicks = motifBars * barTicks;
  const rng = makeRng(o.seed);
  const sorted = [...notes].sort(byTick);
  const startTime = sorted[0]!.tick;
  const endTime = sorted[sorted.length - 1]!.tick + sorted[sorted.length - 1]!.duration;
  const motif = sorted.filter((n) => n.tick < startTime + groupTicks);
  if (motif.length === 0) return sorted;
  const motifRel = motif.map((n) => ({ ...n, tick: n.tick - startTime }));
  const totalGroups = Math.max(1, Math.ceil((endTime - startTime) / groupTicks));
  const TECHS = ["sequence", "invert", "retrograde", "augment", "diminish"] as const;
  const out: ChordAnalysisNote[] = [];
  for (let g = 0; g < totalGroups; g++) {
    if (g === 0) {
      for (const n of motifRel) out.push({ ...n, tick: n.tick + startTime });
      continue;
    }
    const tech: MotifTechnique =
      o.technique === "mixed" ? TECHS[rngInt(rng, 0, TECHS.length - 1)]! : o.technique;
    const gStart = g * groupTicks;
    let varNotes: ChordAnalysisNote[];
    switch (tech) {
      case "sequence": {
        // 模进：按循环音程移调，保持音程结构。
        const shift = [2, -3, 4, -5][(g - 1) % 4]!;
        varNotes = motifRel.map((n) => ({
          ...n,
          pitch: Math.min(SINGABLE_HIGH, Math.max(SINGABLE_LOW, n.pitch + shift)),
        }));
        break;
      }
      case "invert": {
        // 倒影：以动机首音为轴做音程镜像。
        const anchor = motifRel[0]!.pitch;
        varNotes = motifRel.map((n) => ({
          ...n,
          pitch: Math.min(SINGABLE_HIGH, Math.max(SINGABLE_LOW, anchor - (n.pitch - anchor))),
        }));
        break;
      }
      case "retrograde": {
        // 逆行：音符序列倒序，onset 按倒序时值从头铺。
        varNotes = [...motifRel].reverse();
        let t = 0;
        varNotes = varNotes.map((n) => {
          const nt = { ...n, tick: t };
          t += n.duration;
          return nt;
        });
        break;
      }
      case "augment": {
        // 增值：时值加倍（可溢出组尾，与下一组衔接）。
        varNotes = motifRel.map((n) => ({ ...n, duration: n.duration * 2 }));
        break;
      }
      case "diminish": {
        // 减值：时值减半并重复两遍填满组。
        const half = motifRel.map((n) => ({ ...n, duration: Math.max(1, Math.floor(n.duration / 2)) }));
        varNotes = [...half, ...half.map((n) => ({ ...n, tick: n.tick + groupTicks / 2 }))];
        break;
      }
    }
    for (const n of varNotes) {
      if (n.tick >= groupTicks) continue;
      out.push({ ...n, tick: n.tick + startTime + gStart });
    }
  }
  return out.sort(byTick);
}

export type RestructureMethod = "degreeSwap" | "rhythmRestructure" | "contourMorph" | "motifDevelop";

/** 统一入口（供引擎/测试按法分发）。 */
export function restructureMelody(
  notes: ChordAnalysisNote[],
  method: RestructureMethod,
  opts: DegreeSwapOptions & RhythmRestructureOptions & ContourMorphOptions & MotifDevelopOptions,
): ChordAnalysisNote[] {
  switch (method) {
    case "degreeSwap":
      return degreeSwap(notes, opts);
    case "rhythmRestructure":
      return rhythmRestructure(notes, opts);
    case "contourMorph":
      return contourMorph(notes, opts);
    case "motifDevelop":
      return motifDevelop(notes, opts);
  }
}
