/**
 * Muno 阶段4「原生自动编曲」——风格模板(纯数据)。
 *
 * 9 种风格 × 6 种情绪,全部是确定性模板(无模型、无随机):同一旋律 + 同一风格
 * 永远生成完全相同的伴奏(可撤销、可重复、单测钉死)。
 *
 * 网格约定:每小节 16 步(4 拍 × 每 4 步一拍,1 步 = 1/16 = TICKS_PER_BEAT/4)。
 * 鼓组音高 = GM 标准打击乐(36 底鼓 / 38 军鼓 / 42 闭镲 / 46 开镲 …),
 * 名称含英文 "Drums" 的轨道在渲染/编辑器里自动走 channel 10 + 鼓件名形态。
 */

/** 编曲风格。 */
export type ArrangeStyle =
  | "pop" | "rock" | "jazz" | "edm" | "latin" | "funk" | "rnb" | "ballad" | "country"
  | "lofi" | "trap" | "reggae" | "bossanova" | "kpop" | "chinese" | "ambient" | "techno"
  // ── 主流扩充 5 种 ──
  | "folk" | "citypop" | "hiphop" | "house" | "cinematic";

/** 情绪(只调力度/密度/八度,不改节奏骨架)。 */
export type ArrangeMood = "happy" | "sad" | "energetic" | "chill" | "epic" | "neutral";

/** 一步的 tick 数(1/16)。 */
export const STEP_TICKS = 120; // TICKS_PER_BEAT / 4 — 常量直接内联(与 constants.ts 的 480 对齐,避免循环依赖)

export interface DrumPattern {
  /** 底鼓(GM 36)的步序号。 */
  kick: readonly number[];
  /** 军鼓(GM 38)的步序号。 */
  snare: readonly number[];
  /** 军鼓幽灵音(低力度,Ghost note)的步序号。 */
  ghost?: readonly number[];
  /** 闭镲(GM 42)的步序号。 */
  hihat: readonly number[];
  /** 开镲(GM 46)的步序号。 */
  openHat?: readonly number[];
  /** 边击(GM 37,Side stick)。 */
  rim?: readonly number[];
  /** 小节开头加吊镲(GM 49)。 */
  crashOnFirstBar: boolean;
}

export interface BassPattern {
  /**
   * [offsetStep, durSteps] 列表 —— 贝斯音符的位置与时值。
   * pitch 取和弦根音;`useFifth` 的音符用五度(乡村 boom-chick / 流行点缀)。
   */
  notes: readonly { offset: number; dur: number; useFifth?: boolean; octaveUp?: boolean }[];
}

export interface PianoPattern {
  /**
   * 击奏点:步序号列表(和弦整块同时发声);ballad 例外走琶音(quarterArp)。
   * sustainWholeBar = 整小节延音(铺底形态)。
   */
  kind: "hits" | "quarterArp" | "sustain";
  steps?: readonly number[];
}

export interface StyleTemplate {
  id: ArrangeStyle;
  drums: DrumPattern;
  bass: BassPattern;
  piano: PianoPattern;
  /** 和弦铺底轨每几小节换一次和弦(1 = 每小节)。 */
  padBars: 1 | 2;
  /** swing 强度 0~1:奇数八分(第 2,6,10,14 步)右移 swing×拍/6(1 = 三连音摇摆)。 */
  swing: number;
  /** 参考速度(仅 UI 展示提示,不改工程 tempo)。 */
  bpmHint: number;
}

// ── 风格模板 ────────────────────────────────────────────────────────────────
// 16 步/小节。步 → 拍:0/4/8/12 = 第 1/2/3/4 拍正拍。

export const STYLE_TEMPLATES: Readonly<Record<ArrangeStyle, StyleTemplate>> = {
  pop: {
    id: "pop",
    drums: {
      kick: [0, 8],
      snare: [4, 12],
      hihat: [0, 2, 4, 6, 8, 10, 12, 14],
      crashOnFirstBar: true,
    },
    bass: {
      notes: [
        { offset: 0, dur: 4 }, { offset: 4, dur: 4 },
        { offset: 8, dur: 4 }, { offset: 12, dur: 4 },
      ],
    },
    piano: { kind: "hits", steps: [0, 8] },
    padBars: 1,
    swing: 0,
    bpmHint: 100,
  },
  rock: {
    id: "rock",
    drums: {
      kick: [0, 6, 8],
      snare: [4, 12],
      hihat: [0, 2, 4, 6, 8, 10, 12, 14],
      crashOnFirstBar: true,
    },
    bass: {
      notes: [
        { offset: 0, dur: 2 }, { offset: 2, dur: 2 }, { offset: 4, dur: 2 }, { offset: 6, dur: 2 },
        { offset: 8, dur: 2 }, { offset: 10, dur: 2 }, { offset: 12, dur: 2 }, { offset: 14, dur: 2 },
      ],
    },
    piano: { kind: "hits", steps: [0, 2, 4, 6, 8, 10, 12, 14] },
    padBars: 2,
    swing: 0,
    bpmHint: 130,
  },
  jazz: {
    id: "jazz",
    drums: {
      // Ride 摇摆八度近似(生成时奇数八分右移 swing)。
      kick: [],
      snare: [],
      ghost: [2, 6, 10, 14], // 军鼓补偿幽灵音(轻)
      hihat: [4, 12], // 踏镲 2/4 拍
      openHat: [],
      crashOnFirstBar: false,
    },
    bass: {
      // 行走贝斯:每拍一个音,走向由 arranger 计算(根/三/五/邻接)。
      notes: [
        { offset: 0, dur: 4 }, { offset: 4, dur: 4 },
        { offset: 8, dur: 4 }, { offset: 12, dur: 4 },
      ],
    },
    piano: { kind: "hits", steps: [0, 6] }, // Charleston 节奏
    padBars: 1,
    swing: 0.9,
    bpmHint: 120,
  },
  edm: {
    id: "edm",
    drums: {
      kick: [0, 4, 8, 12],
      snare: [4, 12],
      hihat: [],
      openHat: [2, 6, 10, 14],
      crashOnFirstBar: true,
    },
    bass: {
      notes: [
        { offset: 2, dur: 2 }, { offset: 6, dur: 2 },
        { offset: 10, dur: 2 }, { offset: 14, dur: 2 },
      ],
    },
    piano: { kind: "hits", steps: [2, 6, 10, 14] }, // 反拍 stabs
    padBars: 2,
    swing: 0,
    bpmHint: 128,
  },
  latin: {
    id: "latin",
    drums: {
      kick: [0, 3, 8, 11],
      snare: [],
      rim: [3, 11],
      hihat: [0, 2, 4, 6, 8, 10, 12, 14],
      crashOnFirstBar: false,
    },
    bass: {
      // Tumbao 感:长音 + 切分。
      notes: [{ offset: 0, dur: 3 }, { offset: 6, dur: 4 }, { offset: 12, dur: 4 }],
    },
    piano: { kind: "hits", steps: [0, 6, 8, 14] }, // Montuno 感
    padBars: 1,
    swing: 0,
    bpmHint: 105,
  },
  funk: {
    id: "funk",
    drums: {
      kick: [0, 3, 6, 10],
      snare: [4, 12],
      ghost: [7, 15],
      hihat: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
      crashOnFirstBar: false,
    },
    bass: {
      notes: [
        { offset: 0, dur: 2 }, { offset: 3, dur: 1 }, { offset: 6, dur: 2 },
        { offset: 10, dur: 2 }, { offset: 14, dur: 1 },
      ],
    },
    piano: { kind: "hits", steps: [0, 3, 6, 10, 14] },
    padBars: 1,
    swing: 0.15,
    bpmHint: 110,
  },
  rnb: {
    id: "rnb",
    drums: {
      kick: [0, 10],
      snare: [4, 12],
      hihat: [0, 2, 4, 6, 8, 10, 12, 14],
      crashOnFirstBar: false,
    },
    bass: {
      notes: [{ offset: 0, dur: 6 }, { offset: 8, dur: 8 }],
    },
    piano: { kind: "sustain" },
    padBars: 1,
    swing: 0.2,
    bpmHint: 90,
  },
  ballad: {
    id: "ballad",
    drums: {
      kick: [0, 8],
      snare: [12],
      rim: [4],
      hihat: [0, 4, 8, 12],
      crashOnFirstBar: false,
    },
    bass: {
      notes: [{ offset: 0, dur: 16 }],
    },
    piano: { kind: "quarterArp" },
    padBars: 2,
    swing: 0,
    bpmHint: 70,
  },
  country: {
    id: "country",
    drums: {
      kick: [0, 8],
      snare: [4, 12],
      ghost: [5, 13], // 火车节奏感
      hihat: [0, 2, 4, 6, 8, 10, 12, 14],
      crashOnFirstBar: false,
    },
    bass: {
      // Boom-chick:根音-五度交替。
      notes: [
        { offset: 0, dur: 4 }, { offset: 4, dur: 4, useFifth: true },
        { offset: 8, dur: 4 }, { offset: 12, dur: 4, useFifth: true },
      ],
    },
    piano: { kind: "hits", steps: [0, 4, 8, 12] },
    padBars: 1,
    swing: 0.1,
    bpmHint: 120,
  },
  // ── 新增 8 种风格 ──────────────────────────────────────
  lofi: {
    id: "lofi",
    drums: {
      kick: [0, 10],                 // 稀:主 kick + 延后 kick
      snare: [4, 12],
      ghost: [5, 7, 13],             // Lo-Fi 幽灵军鼓 + 随机感
      hihat: [0, 3, 6, 9, 12, 15],   // 稀疏 6 步闭镲 (不是满 8 步)
      openHat: [7],                  // 偶尔开镲
      crashOnFirstBar: false,
    },
    bass: {
      notes: [
        { offset: 0, dur: 6 },       // 长音根音
        { offset: 8, dur: 4, octaveUp: true }, // 八度点缀
      ],
    },
    piano: { kind: "quarterArp" },   // 琶音 = 钢琴加温柔感
    padBars: 2,
    swing: 0.25,                     // Lo-Fi 典型的 swing
    bpmHint: 85,
  },
  trap: {
    id: "trap",
    drums: {
      kick: [0, 3, 8, 11],           // 808 kick: 正拍 + 切分点
      snare: [6, 14],                // Trap 军鼓落在 6 和 14
      hihat: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], // 满 16 步 (滚镲感)
      rim: [2, 10],                  // Rim click 点缀
      crashOnFirstBar: false,
    },
    bass: {
      notes: [
        { offset: 0, dur: 4 },
        { offset: 6, dur: 2 },       // 短切分
        { offset: 8, dur: 4 },
        { offset: 14, dur: 2, useFifth: true },
      ],
    },
    piano: { kind: "hits", steps: [0, 4, 8, 12] },
    padBars: 1,
    swing: 0.15,
    bpmHint: 140,
  },
  reggae: {
    id: "reggae",
    drums: {
      kick: [4, 12],                 // One Drop: kick on 2+4 (反拍!)
      snare: [6, 14],
      hihat: [0, 2, 4, 6, 8, 10, 12, 14],
      rim: [4, 12],                  // Rim on 反拍 + kick
      crashOnFirstBar: true,
    },
    bass: {
      // 反拍切分贝斯 (Reggae 灵魂)
      notes: [
        { offset: 0, dur: 2 },
        { offset: 4, dur: 2 },
        { offset: 6, dur: 2, useFifth: true },
        { offset: 8, dur: 2 },
        { offset: 12, dur: 2 },
      ],
    },
    piano: { kind: "hits", steps: [2, 6, 10, 14] }, // 反拍 stab
    padBars: 1,
    swing: 0.15,
    bpmHint: 100,
  },
  bossanova: {
    id: "bossanova",
    drums: {
      kick: [0, 6, 10],              // 稀疏 Bossa kick
      snare: [],                     // Bossa Nova 几乎不用军鼓 (side stick 是核心)
      rim: [2, 6, 10, 14],           // Side stick = Bossa 灵魂
      hihat: [0, 4, 8, 12, 14],      // 轻 hihat
      crashOnFirstBar: false,
    },
    bass: {
      // 八分 walking bass (Bossa Nova 标准)
      notes: [
        { offset: 0, dur: 2 }, { offset: 2, dur: 2, useFifth: true },
        { offset: 4, dur: 2 }, { offset: 6, dur: 2, useFifth: true },
        { offset: 8, dur: 2 }, { offset: 10, dur: 2, useFifth: true },
        { offset: 12, dur: 2 }, { offset: 14, dur: 2, useFifth: true },
      ],
    },
    piano: { kind: "quarterArp" },   // Bossa 吉他风格琶音
    padBars: 2,
    swing: 0.0,
    bpmHint: 120,
  },
  kpop: {
    id: "kpop",
    drums: {
      kick: [0, 8],
      snare: [4, 12],
      ghost: [5, 13],
      hihat: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], // 满 16 步 = K-Pop 能量
      openHat: [6, 14],              // 反拍开镲爆发
      crashOnFirstBar: true,
    },
    bass: {
      notes: [
        { offset: 0, dur: 4 }, { offset: 4, dur: 2 },
        { offset: 8, dur: 4 }, { offset: 12, dur: 2, octaveUp: true },
      ],
    },
    piano: { kind: "hits", steps: [0, 4, 8, 12] }, // Power chord hit
    padBars: 1,
    swing: 0.0,
    bpmHint: 128,
  },
  chinese: {
    id: "chinese",
    drums: {
      kick: [0, 8],                  // 简单中鼓点
      snare: [4, 12],
      rim: [2, 6, 10, 14],           // 加边击模拟木鱼
      hihat: [0, 4, 8, 12],          // 稀: 每拍一下
      crashOnFirstBar: true,
    },
    bass: {
      notes: [
        { offset: 0, dur: 4 },
        { offset: 4, dur: 4, octaveUp: true }, // 五声音阶感
        { offset: 8, dur: 4 },
        { offset: 12, dur: 4 },
      ],
    },
    piano: { kind: "quarterArp" },   // 琶音 = 模拟古筝
    padBars: 2,
    swing: 0.0,
    bpmHint: 90,
  },
  ambient: {
    id: "ambient",
    drums: {
      kick: [],                      // Ambient 几乎没鼓
      snare: [],
      openHat: [0, 8],               // 只有极稀疏的开镲
      hihat: [8],
      crashOnFirstBar: true,
    },
    bass: {
      notes: [
        { offset: 0, dur: 16 },      // 整小节根音 drone
      ],
    },
    piano: { kind: "quarterArp" },   // 长延音琶音
    padBars: 2,
    swing: 0.0,
    bpmHint: 70,
  },
  techno: {
    id: "techno",
    drums: {
      kick: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], // 满 16 步 4/4 on the floor!
      hihat: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], // 满 16 步
      openHat: [0,4,8,12],           // 每拍开镲重音
      snare: [4, 12],
      crashOnFirstBar: true,
    },
    bass: {
      // 16 分 straight Techno bassline
      notes: [
        { offset: 0, dur: 1 }, { offset: 2, dur: 1 },
        { offset: 4, dur: 1, useFifth: true }, { offset: 6, dur: 1 },
        { offset: 8, dur: 1 }, { offset: 10, dur: 1 },
        { offset: 12, dur: 1, useFifth: true }, { offset: 14, dur: 1 },
      ],
    },
    piano: { kind: "hits", steps: [0, 8] }, // 每小节两下 stab
    padBars: 2,
    swing: 0.0,
    bpmHint: 128,
  },
  // ── 主流扩充 5 种风格 ──────────────────────────────────────
  folk: {
    id: "folk",
    // 民谣:稳拍 + 轻军鼓/边击,突出木吉他分解(引擎层 guitarArp)。
    drums: {
      kick: [0, 8],
      snare: [12],
      rim: [4],
      hihat: [0, 4, 8, 12],
      crashOnFirstBar: false,
    },
    bass: {
      // 根五交替,平稳。
      notes: [
        { offset: 0, dur: 4 }, { offset: 4, dur: 4, useFifth: true },
        { offset: 8, dur: 4 }, { offset: 12, dur: 4, useFifth: true },
      ],
    },
    piano: { kind: "quarterArp" },
    padBars: 2,
    swing: 0.05,
    bpmHint: 96,
  },
  citypop: {
    id: "citypop",
    // 都市流行:groove 反拍 + 十六分闭镲,电钢/铺底为主。
    drums: {
      kick: [0, 6, 10],
      snare: [4, 12],
      ghost: [7, 15],
      hihat: [0, 2, 4, 6, 8, 10, 12, 14],
      openHat: [14],
      crashOnFirstBar: true,
    },
    bass: {
      notes: [
        { offset: 0, dur: 2 }, { offset: 3, dur: 1 }, { offset: 6, dur: 2 },
        { offset: 8, dur: 2 }, { offset: 11, dur: 1 }, { offset: 14, dur: 2 },
      ],
    },
    piano: { kind: "hits", steps: [0, 6, 8, 14] },
    padBars: 1,
    swing: 0.18,
    bpmHint: 108,
  },
  hiphop: {
    id: "hiphop",
    // 嘻哈(区别于 trap):boom-bap 经典骨架,军鼓 4/12,boom kick。
    drums: {
      kick: [0, 6, 10],
      snare: [4, 12],
      ghost: [13],
      hihat: [0, 2, 4, 6, 8, 10, 12, 14],
      crashOnFirstBar: false,
    },
    bass: {
      notes: [
        { offset: 0, dur: 6 }, { offset: 8, dur: 4 },
        { offset: 12, dur: 4, octaveUp: true },
      ],
    },
    piano: { kind: "sustain" },
    padBars: 2,
    swing: 0.22,
    bpmHint: 92,
  },
  house: {
    id: "house",
    // 浩室:four-on-the-floor 正拍踩镲 off-beat(2/6/10/14)。
    drums: {
      kick: [0, 4, 8, 12],
      snare: [],
      rim: [4, 12],
      hihat: [0, 4, 8, 12],   // 正拍闭镲,与底鼓齐
      openHat: [2, 6, 10, 14], // 反拍开镲(off-beat),House 标志性律动
      crashOnFirstBar: true,
    },
    bass: {
      notes: [
        { offset: 0, dur: 2 }, { offset: 4, dur: 2 },
        { offset: 8, dur: 2 }, { offset: 12, dur: 2 },
      ],
    },
    piano: { kind: "hits", steps: [0, 6, 10] }, // 和弦 stab
    padBars: 2,
    swing: 0.0,
    bpmHint: 124,
  },
  cinematic: {
    id: "cinematic",
    // 史诗影视:极简鼓 + 长音铺底,弦乐(引擎层 strings)为主体。
    drums: {
      kick: [0, 8],
      snare: [12],
      hihat: [0, 8],
      crashOnFirstBar: true,
    },
    bass: {
      notes: [{ offset: 0, dur: 16 }], // 整小节根音 drone
    },
    piano: { kind: "sustain" },
    padBars: 2,
    swing: 0.0,
    bpmHint: 80,
  },
};

export const ARRANGE_STYLES: readonly ArrangeStyle[] = Object.keys(STYLE_TEMPLATES) as ArrangeStyle[];

export interface MoodSpec {
  /** 力度整体增减(1-127 钳位)。 */
  velocityGain: number;
  /** 0=稀疏(镲减半拍) 1=正常 2=加密(闭镲 16 分)。 */
  density: 0 | 1 | 2;
  /** 铺底八度偏移(epic 上移一个八度更开阔)。 */
  padOctave: -1 | 0 | 1;
}

export const MOODS: Readonly<Record<ArrangeMood, MoodSpec>> = {
  happy:     { velocityGain: 8,   density: 1, padOctave: 0 },
  sad:       { velocityGain: -12, density: 0, padOctave: 0 },
  energetic: { velocityGain: 14,  density: 2, padOctave: 0 },
  chill:     { velocityGain: -8,  density: 0, padOctave: 0 },
  epic:      { velocityGain: 10,  density: 1, padOctave: 1 },
  neutral:   { velocityGain: 0,   density: 1, padOctave: 0 },
};

export const ARRANGE_MOODS: readonly ArrangeMood[] = Object.keys(MOODS) as ArrangeMood[];
