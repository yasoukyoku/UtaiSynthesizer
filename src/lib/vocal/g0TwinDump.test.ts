// G0 对拍 fixtures dump(SVC2SVS 旋钮线首刀)— 诊断工具,不是 gate。
// 目的:把 f0eval.ts(THE single f0 evaluator)在一批定向边角 + 种子随机谱上的真实产出 dump 成
// JSON,给 SVC2SVS 仓的 PyTorch 可微渲染器孪生(renderer/f0_twin.py)做数值对拍(G0 gate)。
// 用真前端代码 = 零复制(NO-duplication 铁律:孪生的「真值」只能来自这里,绝不在 Python 手抄期望值)。
// 语义要点(孪生消费口径):
//   - theta = effTransition(note, defaultTransition) 合并后的【有效】transition —— 孪生不实现
//     per-field 默认合并,直接吃 12 维有效 θ(transition 6 + vibrato 6);vibrato: null = 无(≡贡献 0)。
//   - pitchDev 恒 undefined(旋钮线训练恒 0,孪生不实现 ③ 层)。
//   - cents 期望值 = Float32Array 量化后的值(f32 舍入 ≤ half-ulp ≈ 4.9e-4¢ @ 现夹具幅值 → 对拍容差锚点)。
//   - 帧网格 = 生产口径 ticksPerFrame = msToTicks(1000/50, tempo)(50fps,分数 tick),另含少量
//     变网格 case(非零 frameStartTick / 100fps)测孪生的网格泛化。
// ★ S71 对抗审查补的三个盲区(动 case 前先懂为什么在):
//   - empty_score:N=0 纯休止窗(训练切窗撞纯伴奏段),真值=全休止,孪生曾在此硬崩;
//   - offset_interior:offsetMs 的【未钳直通】区间——其余 case 的非零 offset 全被钳到端点,
//     换算尺度错(k≥0.7 的膨胀类)会静默绿灯;本 case 生成时自检 raw 严格落在 [loOff,hiOff] 内;
//   - integer_grid_boundaries 的 vibrato 取 phase≠0 且 easeIn=0:让 in_span=0 的整点帧成为
//     真不连续(O(depth) 级),钉死起点门的 >= 语义(phase=0×easeIn>0 会双重湮灭成空断言)。
// 默认 SKIP(UTAI_G0_DUMP=1 才跑),不影响常规 vitest 套件。
// 运行:$env:UTAI_G0_DUMP='1'; npx vitest run src/lib/vocal/g0TwinDump.test.ts
import { describe, it, expect } from "vitest";
// 前端 tsconfig 无 @types/node(include=src 全量过 tsc)。本文件是 node 专用诊断件,用「变量说明符
// 的动态 import」拿 fs(TS 对非字面量说明符不做模块解析,无 TS2307;运行时 node 原生解析)。
type NodeFs = {
  writeFileSync(p: string, data: string): void;
  mkdirSync(p: string, opts?: { recursive?: boolean }): void;
};
declare const process: { env: Record<string, string | undefined> };
const importFs = (): Promise<NodeFs> => {
  const spec = "node:fs";
  return import(/* @vite-ignore */ spec) as Promise<NodeFs>;
};

import { evalF0CentsFrames, effTransition, type F0EvalOpts } from "../f0eval";
import { DEFAULT_TRANSITION } from "../vocalNotes";
import { msToTicks } from "../audio/laneOps";
import type { Note, NoteTransition } from "../../types/project";

const OUT_DIR = process.env.UTAI_G0_OUT ?? "D:\\MyDev\\SVC2SVS\\gates\\fixtures";
const RENDER_FPS = 50; // 生产口径(vocalRender.ts RENDER_FPS)

// ── 确定性 PRNG(mulberry32)— dump 必须可复现(同 seed 同字节),孪生侧靠 case name 定位回归 ──
function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rf = (rng: () => number, lo: number, hi: number) => lo + rng() * (hi - lo);
const ri = (rng: () => number, lo: number, hi: number) => Math.floor(rf(rng, lo, hi + 1));

type VibratoSpec = NonNullable<Note["vibrato"]>;
let nextId = 0;
const mkNote = (
  tick: number, duration: number, pitch: number,
  extra?: { detune?: number; transition?: NoteTransition; vibrato?: VibratoSpec },
): Note => ({
  id: `n${nextId++}`, tick, duration, pitch, lyric: "あ", velocity: 100,
  ...(extra?.detune !== undefined ? { detune: extra.detune } : {}),
  ...(extra?.transition ? { transition: extra.transition } : {}),
  ...(extra?.vibrato ? { vibrato: extra.vibrato } : {}),
});

interface G0Case {
  name: string;
  tempo: number;
  defaultTransition: Required<NoteTransition>;
  notes: Note[];
  /** 缺省 = 生产网格 {frameStartTick:0, ticksPerFrame:msToTicks(20ms), frameCount:cover(lastEnd+pad)} */
  frame?: { frameStartTick: number; ticksPerFrame: number; frameCount: number };
}

const lastEnd = (notes: Note[]) => notes.reduce((m, n) => Math.max(m, n.tick + n.duration), 0);
const prodFrame = (notes: Note[], tempo: number) => {
  const ticksPerFrame = msToTicks(1000 / RENDER_FPS, tempo);
  return { frameStartTick: 0, ticksPerFrame, frameCount: Math.ceil((lastEnd(notes) + 240) / ticksPerFrame) };
};

// ── 定向边角(每个都对应 f0eval 的一条分支/一处 clamp)──
function directedCases(): G0Case[] {
  const cases: G0Case[] = [];

  // ⓪ N=0 空谱(纯休止窗):真值=全帧 {cents:0, voiced:false};孪生的 gather/规约路径必须短路
  cases.push({
    name: "empty_score", tempo: 98,
    defaultTransition: { ...DEFAULT_TRANSITION },
    notes: [],
    frame: { frameStartTick: 0, ticksPerFrame: msToTicks(20, 98), frameCount: 40 },
  });

  // ① durLeft=durRight=0 → 纯阶梯(cross span<=0 / lead·release durL<=0 全灭);openEdge>0 仍无效
  cases.push({
    name: "staircase_zero_durs", tempo: 120,
    defaultTransition: { offsetMs: 0, durLeftMs: 0, durRightMs: 0, depthLeftCents: 0, depthRightCents: 0, openEdgeCents: 200 },
    notes: [mkNote(0, 480, 60), mkNote(480, 480, 64, { detune: -37.5 }), mkNote(1200, 480, 67), mkNote(1680, 240, 55)],
  });

  // ② 贴合上/下行滑音链 + 尾部隔离音(默认 transition;tempo 98 = 分数 tick 网格)
  cases.push({
    name: "abut_updown_default", tempo: 98,
    defaultTransition: { ...DEFAULT_TRANSITION },
    notes: [mkNote(0, 480, 60), mkNote(480, 480, 67), mkNote(960, 480, 62), mkNote(1440, 240, 62), mkNote(1980, 480, 69)],
  });

  // ③ 同音高 seam(dir=0 无 bump)+ 同 pitch 异 detune(dir 来自 cents 差含 detune!)
  cases.push({
    name: "same_pitch_seam", tempo: 110,
    defaultTransition: { ...DEFAULT_TRANSITION },
    notes: [mkNote(0, 480, 64), mkNote(480, 480, 64), mkNote(960, 480, 64, { detune: 30 }), mkNote(1440, 480, 64, { detune: 30 })],
  });

  // ④ 短音符:durR>A/2、durL>B/2、offset clamp 双向绑定(±2000ms 极值)
  cases.push({
    name: "short_note_clamps", tempo: 98,
    defaultTransition: { ...DEFAULT_TRANSITION },
    notes: [
      mkNote(0, 36, 60), mkNote(36, 24, 65, { transition: { offsetMs: 2000 } }),
      mkNote(60, 48, 58, { transition: { offsetMs: -2000 } }), mkNote(108, 480, 63),
      mkNote(588, 30, 66, { transition: { durLeftMs: 2000, durRightMs: 2000 } }), mkNote(618, 30, 61),
    ],
  });

  // ⑤ offset 极值在正常长度音符上(span 整体平移进 B / 进 A;必被钳到端点)
  cases.push({
    name: "offset_extremes_long", tempo: 98,
    defaultTransition: { ...DEFAULT_TRANSITION },
    notes: [
      mkNote(0, 960, 60), mkNote(960, 960, 65, { transition: { offsetMs: 2000 } }),
      mkNote(1920, 960, 58, { transition: { offsetMs: -2000 } }), mkNote(2880, 960, 62),
    ],
  });

  // ⑤b offset 界内直通(S71 盲区补):长贴合链+默认 durs,raw=msToTicks(±30/−50ms) 严格落在
  //    [loOff,hiOff]≈[-78.4,+54.88]t 内不被钳 → 把 offset 的 ms→tick 换算+直通语义钉进 parity
  //    (界内性由 dump 末尾的生成自检硬断言,防未来改 case 悄悄退回全钳盲区)
  cases.push({
    name: "offset_interior", tempo: 98,
    defaultTransition: { ...DEFAULT_TRANSITION },
    notes: [
      mkNote(0, 960, 60), mkNote(960, 960, 65, { transition: { offsetMs: 30 } }),
      mkNote(1920, 960, 58, { transition: { offsetMs: -50 } }), mkNote(2880, 960, 63, { transition: { offsetMs: 30 } }),
    ],
  });

  // ⑥ 开边 scoop/drift:openEdge 变体 + durL=0(lead 灭 release 活)+ openEdge=0(全灭)+ 微型音符 clamp
  cases.push({
    name: "open_edges_suite", tempo: 98,
    defaultTransition: { ...DEFAULT_TRANSITION },
    notes: [
      mkNote(0, 480, 60),
      mkNote(720, 480, 66, { transition: { openEdgeCents: 1200, depthLeftCents: -300, depthRightCents: -300 } }),
      mkNote(1440, 480, 63, { transition: { durLeftMs: 0 } }),
      mkNote(2160, 480, 59, { transition: { openEdgeCents: 0 } }),
      mkNote(2880, 30, 61),
      mkNote(3150, 480, 64, { transition: { durLeftMs: 2000, durRightMs: 2000, openEdgeCents: 400 } }),
    ],
  });

  // ⑦ 负 depth 滑音(undershoot)+ 单侧 0
  cases.push({
    name: "neg_depth_glides", tempo: 98,
    defaultTransition: { ...DEFAULT_TRANSITION, depthLeftCents: -80, depthRightCents: 0 },
    notes: [mkNote(0, 480, 60), mkNote(480, 480, 67, { transition: { depthRightCents: -120 } }), mkNote(960, 480, 61)],
  });

  // ⑧ vibrato 套件:基本 / ease 和越界 / startMs≥dur(无空间)/ 极值域角 / 贴合+隔离叠加
  const vib = (d: number, f: number, p: number, s: number, ei: number, eo: number): VibratoSpec =>
    ({ depthCents: d, freqHz: f, phase: p, startMs: s, easeInMs: ei, easeOutMs: eo });
  cases.push({
    name: "vibrato_suite", tempo: 98,
    defaultTransition: { ...DEFAULT_TRANSITION },
    notes: [
      mkNote(0, 960, 60, { vibrato: vib(50, 5.5, 0, 200, 100, 150) }),
      mkNote(960, 720, 64, { vibrato: vib(120, 6, 0.25, 0, 4000, 4000) }), // ease 和远超活跃段 → env=min 路径
      mkNote(1920, 480, 62, { vibrato: vib(80, 5, 0, 60000, 0, 0) }),      // startMs≥dur → 恒 0
      mkNote(2640, 960, 66, { vibrato: vib(2400, 40, -1, 0, 0, 0) }),      // clamp 域极值角
      mkNote(3600, 960, 59, { vibrato: vib(60, 4.5, 1, 300, 200, 0) }),
    ],
  });

  // ⑨ 整数网格边界命中:tempo 125(1 tick = 1ms,msToTicks 全整)+ tpf=20 → 帧精确落在
  //    note 起点/终点(半开!)/cross t0·t1(闭!)/lead t1/release t0/vibrato start 上。
  //    vibrato 取 phase=0.3+easeIn=0:in_span=0 整点帧携带 40·sin(0.6π)≈38¢ 真不连续,
  //    孪生把起点门 >= 写成 > 会当场差 38¢(S71:phase=0×easeIn>0 时两侧同 0=空断言)
  cases.push({
    name: "integer_grid_boundaries", tempo: 125,
    defaultTransition: { ...DEFAULT_TRANSITION },
    notes: [
      mkNote(0, 400, 60, { transition: { durRightMs: 60 } }),
      mkNote(400, 400, 64, { transition: { durLeftMs: 100 } }), // cross span=[340,500] 两端都在帧上
      mkNote(900, 400, 67, {
        transition: { durLeftMs: 100, durRightMs: 100, openEdgeCents: 200 }, // lead=[900,920] release=[1280,1300]
        vibrato: vib(40, 5, 0.3, 100, 0, 0), // vib 起点 abs tick 1000 = 帧 50(真不连续点)
      }),
      mkNote(1400, 200, 65),
    ],
  });

  // ⑩ tempo 极值(ms↔tick 缩放两端)
  for (const tempo of [55, 200]) {
    cases.push({
      name: `tempo_${tempo}`, tempo,
      defaultTransition: { ...DEFAULT_TRANSITION },
      notes: [
        mkNote(0, 480, 60), mkNote(480, 480, 66, { vibrato: vib(70, 5.5, 0.5, 150, 120, 120) }),
        mkNote(1200, 480, 63), mkNote(1680, 36, 58), mkNote(1716, 480, 61),
      ],
    });
  }

  // ⑪ 变网格:非零(且分数)frameStartTick + 100fps —— 孪生的网格泛化(生产恒 0 起点 50fps)
  {
    const tempo = 98;
    const notes = [mkNote(0, 480, 60), mkNote(480, 480, 65), mkNote(1200, 480, 62, { vibrato: vib(60, 5.5, 0, 100, 100, 100) })];
    const tpf = msToTicks(1000 / 100, tempo);
    cases.push({
      name: "offgrid_100fps_midwindow", tempo,
      defaultTransition: { ...DEFAULT_TRANSITION },
      notes,
      frame: { frameStartTick: 250.5, ticksPerFrame: tpf, frameCount: Math.ceil((lastEnd(notes) - 250.5 + 120) / tpf) },
    });
  }

  // ⑫ 休止密集:首帧前置休止 + 长间隙 + 尾部越界帧(voiced=0 / cents=0 路径)
  cases.push({
    name: "rest_heavy", tempo: 98,
    defaultTransition: { ...DEFAULT_TRANSITION },
    notes: [mkNote(500, 240, 60), mkNote(1400, 240, 64), mkNote(2600, 240, 67)],
  });

  return cases;
}

// ── 种子随机谱(覆盖组合空间;θ 全域采样 = vocalNotes 的 clamp 域)──
function randomCases(count: number): G0Case[] {
  const cases: G0Case[] = [];
  for (let c = 0; c < count; c++) {
    const rng = mulberry32(0xc0ffee + c * 7919);
    const tempo = c % 3 === 0 ? ri(rng, 60, 180) : rf(rng, 60, 180);
    const defaultTransition: Required<NoteTransition> =
      rng() < 0.7 ? { ...DEFAULT_TRANSITION } : {
        offsetMs: rf(rng, -300, 300), durLeftMs: rf(rng, 0, 400), durRightMs: rf(rng, 0, 400),
        depthLeftCents: rf(rng, -200, 200), depthRightCents: rf(rng, -200, 200), openEdgeCents: rf(rng, 0, 600),
      };
    const notes: Note[] = [];
    let cursor = ri(rng, 0, 200);
    let pitch = ri(rng, 52, 76);
    const n = ri(rng, 8, 20);
    for (let i = 0; i < n; i++) {
      const bucket = rng();
      const duration = bucket < 0.2 ? ri(rng, 24, 60) : bucket < 0.75 ? ri(rng, 120, 480) : ri(rng, 480, 1440);
      pitch = Math.min(79, Math.max(48, pitch + ri(rng, -7, 7)));
      const extra: { detune?: number; transition?: NoteTransition; vibrato?: VibratoSpec } = {};
      if (rng() < 0.3) extra.detune = rf(rng, -100, 100);
      if (rng() < 0.5) {
        const t: NoteTransition = {};
        if (rng() < 0.4) t.offsetMs = rf(rng, -2000, 2000);
        if (rng() < 0.4) t.durLeftMs = rf(rng, 0, 2000);
        if (rng() < 0.4) t.durRightMs = rf(rng, 0, 2000);
        if (rng() < 0.4) t.depthLeftCents = rf(rng, -1200, 1200);
        if (rng() < 0.4) t.depthRightCents = rf(rng, -1200, 1200);
        if (rng() < 0.4) t.openEdgeCents = rf(rng, 0, 1200);
        if (Object.keys(t).length > 0) extra.transition = t;
      }
      if (rng() < 0.35) {
        extra.vibrato = {
          depthCents: rng() < 0.85 ? rf(rng, 5, 400) : rf(rng, 400, 2400),
          freqHz: rng() < 0.85 ? rf(rng, 0.5, 12) : rf(rng, 12, 40),
          phase: rf(rng, -1, 1), startMs: rf(rng, 0, 1200), easeInMs: rf(rng, 0, 800), easeOutMs: rf(rng, 0, 800),
        };
      }
      notes.push(mkNote(cursor, duration, pitch, extra));
      const g = rng();
      cursor += duration + (g < 0.6 ? 0 : g < 0.85 ? ri(rng, 30, 300) : ri(rng, 1, 5)); // 贴合/休止/擦边非贴合
    }
    cases.push({ name: `random_${c}`, tempo, defaultTransition, notes });
  }
  return cases;
}

describe.skipIf(!process.env.UTAI_G0_DUMP)("G0 twin-parity dump (diagnostic, not a gate)", () => {
  it("dumps directed + seeded-random f0eval fixtures for the SVC2SVS renderer twin", async () => {
    const fs = await importFs();
    const cases = [...directedCases(), ...randomCases(12)];
    const dumped = [];
    for (const c of cases) {
      const opts: F0EvalOpts = { tempo: c.tempo, defaultTransition: c.defaultTransition };
      const frame = c.frame ?? prodFrame(c.notes, c.tempo);
      const { cents, voiced } = evalF0CentsFrames(c.notes, undefined, frame, opts);
      // 生成器自检(抓 case 构造 bug,非孪生对拍):全有限 + 有声帧存在(空谱除外)+ 休止帧 cents=0
      expect([...cents].every(Number.isFinite), `${c.name}: non-finite cents`).toBe(true);
      const voicedN = voiced.reduce((s: number, v: number) => s + v, 0);
      if (c.notes.length > 0) expect(voicedN, `${c.name}: no voiced frames`).toBeGreaterThan(0);
      else expect(voicedN, `${c.name}: empty score must be all-rest`).toBe(0);
      expect(voicedN, `${c.name}: no rest frames`).toBeLessThan(frame.frameCount);
      for (let f = 0; f < frame.frameCount; f++) {
        if (!voiced[f]) expect(cents[f], `${c.name}: rest frame ${f} cents != 0`).toBe(0);
      }
      dumped.push({
        name: c.name, tempo: c.tempo, frame,
        notes: c.notes.map((nt) => ({
          tick: nt.tick, duration: nt.duration, pitch: nt.pitch, detune: nt.detune ?? 0,
          theta: effTransition(nt, c.defaultTransition), // 有效 θ(默认合并在 TS 侧完成)
          vibrato: nt.vibrato ?? null,
        })),
        cents: Array.from(cents), // Float32Array → f32 量化后的精确值(JSON 无损往返)
        voiced: Array.from(voiced),
      });
      // eslint-disable-next-line no-console
      console.log(`[g0dump] ${c.name}: ${c.notes.length} notes, ${frame.frameCount} frames, voiced ${voicedN}`);
    }
    // 阶梯 case 解析自检:span 外内部帧 == noteCents 精确(容差=f32 量化)
    {
      const st = dumped.find((d) => d.name === "staircase_zero_durs")!;
      const tpf = st.frame.ticksPerFrame;
      const inNote1 = Math.round(700 / tpf); // note1=[480,960) pitch64 detune-37.5,700 tick 在内部
      expect(Math.abs(st.cents[inNote1]! - (64 * 100 - 37.5))).toBeLessThan(2e-3);
    }
    // ⑤b 界内直通自检(S71 盲区的守门断言):offset_interior 每个非零 offset 的 raw=msToTicks(offsetMs)
    // 必须【严格】落在该贴合对的 [loOff,hiOff] 开区间内(f0eval crossCents 的钳公式在此复算)——
    // 一旦未来有人改 case 让它退回被钳,G0 对 offset 换算的覆盖会静默消失,这里直接红
    {
      const oc = cases.find((c) => c.name === "offset_interior")!;
      for (let i = 1; i < oc.notes.length; i++) {
        const A = oc.notes[i - 1]!, B = oc.notes[i]!;
        const tA = effTransition(A, oc.defaultTransition), tB = effTransition(B, oc.defaultTransition);
        const durR = Math.min(msToTicks(tA.durRightMs, oc.tempo), A.duration / 2);
        const durL = Math.min(msToTicks(tB.durLeftMs, oc.tempo), B.duration / 2);
        const loOff = Math.max(-durL, durR - A.duration / 2);
        const hiOff = Math.min(durR, B.duration / 2 - durL);
        const raw = msToTicks(tB.offsetMs, oc.tempo);
        expect(raw, `offset_interior pair ${i}: offset 必须非零`).not.toBe(0);
        expect(raw > loOff && raw < hiOff, `offset_interior pair ${i}: raw=${raw} 必须严格在 (${loOff},${hiOff}) 内`).toBe(true);
      }
    }
    fs.mkdirSync(OUT_DIR, { recursive: true });
    const out = `${OUT_DIR}\\g0_cases.json`;
    fs.writeFileSync(out, JSON.stringify({
      meta: {
        generator: "Utai_v2-dev/src/lib/vocal/g0TwinDump.test.ts",
        semantics: "evalF0CentsFrames(notes, pitchDev=undefined, frame, {tempo, defaultTransition});"
          + " theta=effTransition 合并后有效值; cents=Float32 量化; vibrato:null=无(≡0); TICKS_PER_BEAT=480",
        renderFps: RENDER_FPS, caseCount: dumped.length,
      },
      cases: dumped,
    }));
    // eslint-disable-next-line no-console
    console.log(`[g0dump] wrote ${dumped.length} cases -> ${out}`);
  });
});
