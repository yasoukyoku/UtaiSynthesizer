/**
 * §user「MIDI Lens 式和弦分析」——从 AMT 转谱产出的音符集识别和弦进行。
 *
 * 纯前端、确定性（同输入同输出）、零依赖。链路：
 *
 *   音符集 → 按拍窗切分 → 时长加权音级直方图 → 和弦模板匹配（含转位/Slash）
 *          → 惯性平滑（偏好强拍换和弦 + 延续前和弦）→ 和弦段 + 全曲调性
 *
 * 支持的和弦类型（对齐 MIDI Lens 的识别面）：
 *   maj, min, dim, aug, sus2, sus4, 6, 7, maj7, min7, m7b5, dim7, 9, add9,
 *   以及 Slash 和弦（原位/转位检测，标注低音）。
 */

/** 分析输入音符（tick 域，与工程 Note/CleanableNote 同构）。 */
export interface ChordAnalysisNote {
  tick: number;
  duration: number;
  pitch: number;
  velocity?: number;
}

/** 一段和弦。tick 域；root 用音级 0-11（C=0），bass 为低音音级（Slash 检测），无则为 null。 */
export interface ChordSegment {
  startTick: number;
  endTick: number;
  root: number;
  quality: ChordQuality;
  /** 转位/Slash 的低音音级；与 root 相同（原位）时为 null。 */
  bass: number | null;
  /** 如 "C", "Am7", "G/B"。 */
  label: string;
}

export type ChordQuality =
  | "maj" | "min" | "dim" | "aug"
  | "sus2" | "sus4"
  | "6" | "7" | "maj7" | "min7" | "m7b5" | "dim7"
  | "9" | "add9";

const PITCH_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

/** 和弦模板：以根音为 0 的音级集合。顺序无关（穷举全根音），仅影响平票时的选择。 */
const CHORD_TEMPLATES: ReadonlyArray<{ quality: ChordQuality; intervals: readonly number[] }> = [
  { quality: "maj7", intervals: [0, 4, 7, 11] },
  { quality: "min7", intervals: [0, 3, 7, 10] },
  { quality: "m7b5", intervals: [0, 3, 6, 10] },
  { quality: "dim7", intervals: [0, 3, 6, 9] },
  { quality: "7", intervals: [0, 4, 7, 10] },
  { quality: "6", intervals: [0, 4, 7, 9] },
  { quality: "9", intervals: [0, 4, 7, 10, 14] },
  { quality: "add9", intervals: [0, 4, 7, 14] },
  { quality: "sus4", intervals: [0, 5, 7] },
  { quality: "sus2", intervals: [0, 2, 7] },
  { quality: "dim", intervals: [0, 3, 6] },
  { quality: "aug", intervals: [0, 4, 8] },
  { quality: "maj", intervals: [0, 4, 7] },
  { quality: "min", intervals: [0, 3, 7] },
];

const QUALITY_SUFFIX: Record<ChordQuality, string> = {
  maj: "", min: "m", dim: "dim", aug: "aug",
  sus2: "sus2", sus4: "sus4",
  "6": "6", "7": "7", maj7: "maj7", min7: "m7", m7b5: "m7b5", dim7: "dim7",
  "9": "9", add9: "add9",
};

/** Krumhansl–Kessler 调性模板（时长加权音级直方图相关）。 */
const KK_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
const KK_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];

export interface KeyEstimate {
  /** 主音音级 0-11。 */
  tonic: number;
  major: boolean;
  /** 相关系数 [-1,1]，越高越可信。 */
  confidence: number;
  label: string;
}

export interface ChordAnalysisResult {
  segments: ChordSegment[];
  key: KeyEstimate;
}

export interface ChordAnalysisOptions {
  /** 每拍切几个分析窗（1=按拍，2=半拍）。默认 1。 */
  windowsPerBeat?: number;
  /** 匹配阈值 [0,1]：窗内最佳模板得分低于它时输出 N.C.（延续前和弦）。默认 0.55。 */
  matchThreshold?: number;
  /** 平滑惯性：非强拍新和弦需超过 阈值+inertia 才切换。默认 0.15。 */
  inertia?: number;
}

const mod12 = (n: number): number => ((n % 12) + 12) % 12;

function pearson(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = 12;
  let ma = 0;
  let mb = 0;
  for (let i = 0; i < n; i++) {
    ma += a[i] ?? 0;
    mb += b[i] ?? 0;
  }
  ma /= n;
  mb /= n;
  let num = 0;
  let da = 0;
  let db = 0;
  for (let i = 0; i < n; i++) {
    const x = (a[i] ?? 0) - ma;
    const y = (b[i] ?? 0) - mb;
    num += x * y;
    da += x * x;
    db += y * y;
  }
  if (da < 1e-9 || db < 1e-9) return 0;
  return num / Math.sqrt(da * db);
}

/** 全曲调性：所有音符时长加权 → 音级直方图 → KK 大小调模板相关。 */
export function estimateKey(notes: ChordAnalysisNote[]): KeyEstimate {
  const hist = new Float64Array(12);
  for (const n of notes) {
    const idx = mod12(n.pitch);
    hist[idx] = (hist[idx] ?? 0) + Math.max(0, n.duration);
  }
  let best: KeyEstimate = { tonic: 0, major: true, confidence: -2, label: "C" };
  for (let tonic = 0; tonic < 12; tonic++) {
    const rotated = new Float64Array(12);
    for (let i = 0; i < 12; i++) rotated[i] = hist[mod12(i + tonic)] ?? 0;
    const rMaj = pearson(rotated, KK_MAJOR);
    const rMin = pearson(rotated, KK_MINOR);
    if (rMaj > best.confidence) {
      best = { tonic, major: true, confidence: rMaj, label: PITCH_NAMES_SHARP[tonic]! };
    }
    if (rMin > best.confidence) {
      best = { tonic, major: false, confidence: rMin, label: `${PITCH_NAMES_SHARP[tonic]}m` };
    }
  }
  return best;
}

interface WindowMatch {
  root: number;
  quality: ChordQuality;
  score: number;
}

/**
 * 单窗匹配：给定音级能量向量，穷举 12 个根音 × 全部模板。
 * 得分 = 模板音级覆盖能量占比 − 非模板音级能量占比 ×外音惩罚（复杂模板惩罚更低）。
 */
function matchWindow(pc: Float64Array): WindowMatch | null {
  let total = 0;
  for (let i = 0; i < 12; i++) total += pc[i] ?? 0;
  if (total < 1e-9) return null;

  let best: WindowMatch | null = null;
  for (const tpl of CHORD_TEMPLATES) {
    const ivs = tpl.intervals;
    for (let root = 0; root < 12; root++) {
      const inSet = new Set<number>();
      for (const iv of ivs) inSet.add(mod12(root + iv));
      let covered = 0;
      for (let i = 0; i < 12; i++) {
        if (inSet.has(i)) covered += pc[i] ?? 0;
      }
      const outside = total - covered;
      // 覆盖率打分：超集模板（maj7/6/add9…）同样能 100% 覆盖纯三和弦输入（平票 1.0），
      // 因此按模板音数施加微小复杂度惩罚，平票时永远偏好最简解释（奥卡姆剃刀）。
      const score =
        covered / total - (outside / total) * (ivs.length >= 4 ? 0.55 : 0.9) - ivs.length * 1e-3;
      if (!best || score > best.score) {
        best = { root, quality: tpl.quality, score };
      }
    }
  }
  return best;
}

/** 窗内最低音符音级（Slash/转位检测用）。 */
function lowestPcInWindow(notes: ChordAnalysisNote[], startTick: number, endTick: number): number | null {
  let lowest = Infinity;
  for (const n of notes) {
    if (n.tick >= endTick || n.tick + n.duration <= startTick) continue;
    if (n.pitch < lowest) lowest = n.pitch;
  }
  return Number.isFinite(lowest) ? mod12(lowest) : null;
}

export function chordLabel(root: number, quality: ChordQuality, bass: number | null): string {
  const base = `${PITCH_NAMES_SHARP[root]}${QUALITY_SUFFIX[quality]}`;
  return bass != null && bass !== root ? `${base}/${PITCH_NAMES_SHARP[bass]}` : base;
}

/**
 * 主入口：音符集 → 和弦段列表 + 调性。
 *
 * @param notes       tick 域音符（AMT 产出的任意乐器轨；建议伴奏/合并轨）
 * @param ppq         每 4 分音符 tick 数（MIDI 头；工程内通常 480）
 * @param beatsPerBar 拍号分子（默认 4）——只影响"强拍"判定
 */
export function analyzeChords(
  notes: ChordAnalysisNote[],
  ppq: number,
  beatsPerBar = 4,
  opts: ChordAnalysisOptions = {},
): ChordAnalysisResult {
  const windowsPerBeat = Math.max(1, opts.windowsPerBeat ?? 1);
  const threshold = opts.matchThreshold ?? 0.55;
  const inertia = opts.inertia ?? 0.15;

  if (notes.length === 0 || ppq <= 0) {
    return { segments: [], key: { tonic: 0, major: true, confidence: 0, label: "C" } };
  }

  let minTick = Infinity;
  let maxTick = -Infinity;
  for (const n of notes) {
    if (n.tick < minTick) minTick = n.tick;
    const end = n.tick + n.duration;
    if (end > maxTick) maxTick = end;
  }

  const windowTicks = Math.max(1, Math.floor(ppq / windowsPerBeat));
  const firstWindow = Math.floor(minTick / windowTicks);
  const lastWindow = Math.floor(Math.max(maxTick - 1, minTick) / windowTicks);
  const windowsPerBar = beatsPerBar * windowsPerBeat;

  interface Smoothed {
    startTick: number;
    endTick: number;
    root: number;
    quality: ChordQuality;
    bass: number | null;
  }
  const out: Smoothed[] = [];

  for (let w = firstWindow; w <= lastWindow; w++) {
    const start = w * windowTicks;
    const end = start + windowTicks;
    const pc = new Float64Array(12);
    for (const n of notes) {
      const s = Math.max(n.tick, start);
      const e = Math.min(n.tick + n.duration, end);
      if (e <= s) continue;
      const idx = mod12(n.pitch);
      pc[idx] = (pc[idx] ?? 0) + (e - s); // 时长加权（在本窗内的部分）
    }
    const m = matchWindow(pc);
    const isBarStart = ((w % windowsPerBar) + windowsPerBar) % windowsPerBar === 0;
    const prev = out[out.length - 1];

    if (!m) {
      // 无可信匹配：延续前和弦（避免 N.C. 碎段）；曲首无前和弦则跳过该窗。
      if (prev && prev.endTick === start) prev.endTick = end;
      continue;
    }
    const low = lowestPcInWindow(notes, start, end);
    const bass = low != null && low !== m.root ? low : null;

    if (prev && prev.endTick === start) {
      const sameChord = prev.root === m.root && prev.quality === m.quality;
      // 惯性：非强拍、新和弦、得分未显著超阈值 → 延续前和弦
      if (sameChord) {
        prev.endTick = end;
        if (bass != null) prev.bass = bass;
        continue;
      }
      if (!isBarStart && inertia > 0 && m.score < threshold + inertia) {
        prev.endTick = end;
        continue;
      }
    }
    out.push({ startTick: start, endTick: end, root: m.root, quality: m.quality, bass });
  }

  const segments: ChordSegment[] = out.map((s) => ({
    ...s,
    label: chordLabel(s.root, s.quality, s.bass),
  }));
  return { segments, key: estimateKey(notes) };
}

// ═══════════════════════════════════════════════════════════════════════
// OPT5: FFT/CQT 音频和弦识别 — 直接在频谱域做 pitch-class histogram
// 复用 CHORD_TEMPLATES + matchWindow() — 无需任何 MIDI 音符输入。
//
//   AudioBuffer / Float32Array samples → analyzeCqt → 每帧 12 维 histogram
//   → 滑窗聚合成 per-bar pc → matchWindow 识别和弦 + KK 调性
//
// 纯前端, 零 Rust 依赖, 浏览器里也能跑。
// ═══════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════
// OPT5: FFT/CQT 音频和弦识别 — 直接在频谱域做 pitch-class histogram
// 复用 CHORD_TEMPLATES + matchWindow() — 无需任何 MIDI 音符输入。
// 纯前端, 零 Rust 依赖, 浏览器里也能跑。
// ═══════════════════════════════════════════════════════════════════════

export interface AudioChordOptions {
  /** 分析窗口, 秒。默认 0.5s (4/4 拍@120BPM ≈ 半小节)。 */
  windowSec?: number;
  /** 窗口间 hop, 秒。默认 windowSec/2 (50% 重叠)。 */
  hopSec?: number;
  /** 检测静音阈值 (相对 RMS)。低于此值不输出和弦。默认 0.01。 */
  silenceFloor?: number;
  /** 和弦切换惯性, 抑制碎段。默认 0.3。 */
  inertia?: number;
}
export function analyzeChordsFromCqt(
  cqt: { times: number[]; magnitudes: Float32Array[]; pitches: number[] },
  _sampleRate: number,
  tempo: number,
  opts: AudioChordOptions = {},
): { segments: ChordSegment[]; key: KeyEstimate } {
  const windowSec = opts.windowSec ?? 0.5;
  const hopSec = opts.hopSec ?? windowSec * 0.5;
  const silenceFloor = opts.silenceFloor ?? 0.01;
  const inertia = opts.inertia ?? 0.3;

  if (cqt.times.length === 0 || cqt.pitches.length === 0) {
    return { segments: [], key: { tonic: 0, major: true, confidence: 0, label: "Cmaj" } };
  }

  // Step 1: 把 CQT magnitude (88 pitches × N frames) 逐帧聚合到 12 维 pitch-class histogram
  const pcHistPerFrame: Float64Array[] = [];
  for (let fr = 0; fr < cqt.times.length; fr++) {
    const pc = new Float64Array(12);
    for (let pi = 0; pi < cqt.pitches.length; pi++) {
      const midi = cqt.pitches[pi]!;
      const pcIdx = mod12(midi);
      const row = cqt.magnitudes[pi];
      const mag = row != null ? (row[fr] ?? 0) : 0;
      if (mag > 0) pc[pcIdx] = (pc[pcIdx] ?? 0) + Math.log1p(mag);
    }
    // 归一化 (L2 norm)
    let norm = 0;
    for (let i = 0; i < 12; i++) norm += (pc[i] ?? 0) * (pc[i] ?? 0);
    norm = Math.sqrt(norm);
    if (norm > 1e-9) for (let i = 0; i < 12; i++) pc[i] = (pc[i] ?? 0) / norm;
    pcHistPerFrame.push(pc);
  }

  // Step 2: 滑窗聚合成 per-window pc + 静音检测
  const hopFrameDur = (cqt.times[1] ?? 0.02) - (cqt.times[0] ?? 0);
  const frameHop = Math.max(1, Math.round(hopSec / (hopFrameDur || 0.02)));
  const frameWin = Math.max(2, Math.round(windowSec / (hopFrameDur || 0.02)));

  const beatSec = 60 / tempo;
  const tickPerSec = 480 / beatSec; // 工程 PPQ = 480
  const out: Array<{ startSec: number; endSec: number; startTick: number; endTick: number; pc: Float64Array }> = [];

  for (let fr = 0; fr + frameWin < pcHistPerFrame.length; fr += frameHop) {
    const agg = new Float64Array(12);
    let frameCount = 0;
    let rms = 0;
    for (let k = 0; k < frameWin && (fr + k) < pcHistPerFrame.length; k++) {
      const f = pcHistPerFrame[fr + k]!;
      for (let i = 0; i < 12; i++) agg[i] = (agg[i] ?? 0) + (f[i] ?? 0);
      let frRms = 0;
      for (let i = 0; i < 12; i++) { const v = f[i] ?? 0; frRms += v * v; }
      rms += Math.sqrt(frRms / 12);
      frameCount++;
    }
    rms /= frameCount;
    if (rms < silenceFloor) continue;

    let n = 0;
    for (let i = 0; i < 12; i++) n += (agg[i] ?? 0) * (agg[i] ?? 0);
    n = Math.sqrt(n);
    if (n > 1e-9) for (let i = 0; i < 12; i++) agg[i] = (agg[i] ?? 0) / n;

    const startSec = cqt.times[fr] ?? 0;
    const endSec = cqt.times[Math.min(fr + frameWin, cqt.times.length - 1)] ?? startSec;
    out.push({
      startSec, endSec,
      startTick: Math.round(startSec * tickPerSec),
      endTick: Math.round(endSec * tickPerSec),
      pc: agg,
    });
  }

  // Step 3: matchWindow + 惯性平滑
  const segments: ChordSegment[] = [];
  let prev: ChordSegment | null = null;
  for (const w of out) {
    const m = matchWindow(w.pc);
    if (!m) {
      if (prev && prev.endTick === w.startTick) prev.endTick = w.endTick;
      continue;
    }
    if (prev && prev.endTick === w.startTick) {
      const same = prev.root === m.root && prev.quality === m.quality;
      if (same) {
        prev.endTick = w.endTick;
        continue;
      }
      // 惯性平滑：不同和弦仅当置信度足够高时才切换；低于 inertia 视为噪声/转位碎片，
      // 延续前和弦抑制碎段。原实现把 score（≤1 的归一化得分）与
      // `prev.endTick - prev.startTick`（tick 数，通常数百）比较，量纲错位恒为真，
      // 使得 inertia 彻底失效、任意低分不同和弦一律被合并。
      if (m.score < inertia) {
        prev.endTick = w.endTick;
        continue;
      }
    }
    prev = {
      startTick: w.startTick, endTick: w.endTick,
      root: m.root, quality: m.quality, bass: null,
      label: chordLabel(m.root, m.quality, null),
    };
    segments.push(prev);
  }

  // Step 4: 调性估计 (KK 模板)
  const totalPc = new Float64Array(12);
  for (const w of out) for (let i = 0; i < 12; i++) totalPc[i] = (totalPc[i] ?? 0) + (w.pc[i] ?? 0);
  let key: KeyEstimate = { tonic: 0, major: true, confidence: 0, label: "Cmaj" };
  {
    let maxR = -Infinity, bestT = 0, bestMajor = true;
    for (let tonic = 0; tonic < 12; tonic++) {
      const shiftedMaj = new Float64Array(12);
      const shiftedMin = new Float64Array(12);
      for (let i = 0; i < 12; i++) {
        shiftedMaj[i] = totalPc[mod12(i + tonic)] ?? 0;
        shiftedMin[i] = totalPc[mod12(i + tonic)] ?? 0;
      }
      const rM = pearson(shiftedMaj, KK_MAJOR);
      const rm = pearson(shiftedMin, KK_MINOR);
      if (rM > maxR) { maxR = rM; bestT = tonic; bestMajor = true; }
      if (rm > maxR) { maxR = rm; bestT = tonic; bestMajor = false; }
    }
    const KK_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"];
    key = {
      tonic: bestT, major: bestMajor,
      confidence: Math.max(0, maxR),
      label: KK_NAMES[bestT]! + (bestMajor ? "maj" : "min"),
    };
  }

  return { segments, key };
}