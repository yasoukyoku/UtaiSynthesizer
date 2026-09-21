/**
 * 和弦重配（规划 2-5，对应 4.3 和声层面原创化）。
 * 策略：tritone 三全音替代 / diatonic 调内相对替换 / borrowed 同主音借用 / extension 延伸音。
 * 纯函数，种子化随机，密度参数控制被替换的和弦比例。
 */
import type { ChordQuality, ChordSegment } from "../analysis/chordAnalysis";
import { makeRng, rngInt, rngChance } from "./rng";
import { clamp01, mod12p } from "./common";
import { chordLabel } from "./chords";

export type ReharmStrategy = "tritone" | "diatonic" | "borrowed" | "extension";

export interface ReharmonizeOptions {
  /** 启用的策略集合（引擎按节点参数选择一个）。 */
  strategies: ReharmStrategy[];
  /** 0-100，每个和弦被处理的概率。 */
  density: number;
  seed?: number;
}

/** 延伸音候选：同功能加色彩，不改变进行逻辑。 */
const EXTENSION_CANDIDATES: Partial<Record<ChordQuality, ChordQuality[]>> = {
  maj: ["maj7", "add9", "6"],
  maj7: ["add9", "6"],
  "7": ["9"],
  min: ["min7"],
};

function seg(s: ChordSegment, root: number, quality: ChordQuality): ChordSegment {
  return { ...s, root: mod12p(root), quality, bass: null, label: chordLabel(root, quality) };
}

function applyStrategy(s: ChordSegment, strat: ReharmStrategy, rng: ReturnType<typeof makeRng>): ChordSegment | null {
  switch (strat) {
    case "extension": {
      const opts = EXTENSION_CANDIDATES[s.quality] ?? [];
      if (opts.length === 0) return null;
      return seg(s, s.root, opts[rngInt(rng, 0, opts.length - 1)]!);
    }
    case "tritone": {
      // 属功能替代：C7 → Gb7；大三和弦按属色彩处理 C → Gb7。
      if (s.quality !== "maj" && s.quality !== "7") return null;
      return seg(s, s.root + 6, "7");
    }
    case "diatonic": {
      // 调内相对替换：共享两个和弦音。C → Am（vi），Am → C（III）。
      if (s.quality === "maj") return seg(s, s.root + 9, "min");
      if (s.quality === "min") return seg(s, s.root + 3, "maj");
      return null;
    }
    case "borrowed": {
      // 同主音借用：平行大小互换（色彩突变）。
      if (s.quality === "maj") return seg(s, s.root, "min");
      if (s.quality === "min") return seg(s, s.root, "maj");
      return null;
    }
  }
}

export function reharmonizeSegments(segments: ChordSegment[], o: ReharmonizeOptions): ChordSegment[] {
  if (segments.length === 0) return [];
  const rng = makeRng(o.seed);
  const density = clamp01(o.density / 100);
  const strategies = o.strategies.length > 0 ? o.strategies : (["extension"] as ReharmStrategy[]);
  return segments.map((s) => {
    if (!rngChance(rng, density)) return { ...s };
    // Fisher-Yates 打乱策略顺序，让多策略节点每次随机挑一个可行的。
    const order = [...strategies];
    for (let i = order.length - 1; i > 0; i--) {
      const j = rngInt(rng, 0, i);
      const tmp = order[i]!;
      order[i] = order[j]!;
      order[j] = tmp;
    }
    for (const strat of order) {
      const sub = applyStrategy(s, strat, rng);
      if (sub) return sub;
    }
    return { ...s };
  });
}
