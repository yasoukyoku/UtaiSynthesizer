// E1 交叉实验(S70)— 参数化 f0 dump 工具,不是 gate。
// 目的:把「生产默认调教」的 Option-A f0(buildVocalScore → evalF0CentsFrames,默认 transition/无
// vibrato/无 pitchDev)对 E1 测试段真实产出一遍,dump 成 JSON 给 Rust 侧 e1_cross_probe harness 消费
// (D:\MyDev\TESTING\e1_cross_probe\{seg}_score.json)。用真前端代码 = 零复制(NO-duplication 铁律:
// 绝不在 Rust/Python 重写 f0 求值)。默认 SKIP(UTAI_E1_DUMP=1 才跑),因此不影响常规 vitest 套件。
// 运行:$env:UTAI_E1_DUMP='1'; npx vitest run src/lib/vocal/e1CrossDump.test.ts
import { describe, it, expect, vi } from "vitest";
// 前端 tsconfig 无 @types/node(include=src 全量过 tsc)。本文件是 node 专用诊断件(vitest 的
// node env 运行时真有 node:fs),故用「变量说明符的动态 import」拿 fs——TS 对非字面量说明符
// 不做模块解析(无 TS2307),运行时由 node 原生解析;不引 @types/node、不动 tsconfig。
type NodeFs = {
  readFileSync(p: string, enc: string): string;
  writeFileSync(p: string, data: string): void;
  existsSync(p: string): boolean;
};
declare const process: { env: Record<string, string | undefined> };
const importFs = (): Promise<NodeFs> => {
  const spec = "node:fs";
  return import(/* @vite-ignore */ spec) as Promise<NodeFs>;
};

// vocalRender.ts 也导入 invoke/store/i18n(为 renderVocalSegment)— mock 使模块可无头加载
// (与 vocalRender.test.ts 同款)。
vi.mock("@tauri-apps/api/core", () => ({ invoke: () => Promise.resolve() }));
vi.mock("../../i18n", () => ({ default: { t: (k: string) => k } }));
vi.mock("../../store/voice-models", () => ({ useVoiceModelStore: { getState: () => ({ models: { sovits: [] } }) } }));

import { buildVocalScore } from "./vocalRender";
import { DEFAULT_TRANSITION } from "../vocalNotes";
import type { Note } from "../../types/project";

const WORK_DIR = "D:\\MyDev\\TESTING\\e1_cross_probe";
const SEGMENTS = ["verse", "chorus"];

describe.skipIf(!process.env.UTAI_E1_DUMP)("E1 param-f0 dump (diagnostic, not a gate)", () => {
  it("dumps production-default Option-A f0 + triples for each E1 segment", async () => {
    const fs = await importFs();
    const join = (...parts: string[]) => parts.join("\\");
    const manifest = JSON.parse(fs.readFileSync(join(WORK_DIR, "e1_manifest.json"), "utf-8"));
    const tempo: number = manifest.tempo; // 98 — the svp's (and thus the wav slice's) tempo
    for (const name of SEGMENTS) {
      const metaPath = join(WORK_DIR, `${name}_notes.json`);
      expect(fs.existsSync(metaPath), `missing ${metaPath} (run e1_emit_segments.py first)`).toBe(true);
      const meta = JSON.parse(fs.readFileSync(metaPath, "utf-8"));
      const notes: Note[] = meta.notes.map((n: { tick: number; duration: number; pitch: number; lyric: string }, i: number) => ({
        id: `n${i}`, tick: n.tick, duration: n.duration, pitch: n.pitch, lyric: n.lyric, velocity: 100,
      }));
      // 生产口径:默认 transition(track default)、无 vibrato、无 pitchDev、无参数泳道、JA(lang 2 默认)。
      const { triples, f0Cents, f0Voiced, loudnessEnv, formantEnv } = buildVocalScore(
        notes, undefined, tempo, DEFAULT_TRANSITION, "AP",
      );
      const sum = triples.reduce((s, t) => s + t.frames, 0);
      expect(f0Cents.length).toBe(sum); // render_vocal_segment 的硬校验同款不变量
      expect(f0Voiced.length).toBe(sum);
      expect(loudnessEnv.length).toBe(0); // 无泳道 → 空数组(flat no-op)
      expect(formantEnv.length).toBe(0);
      expect(triples.every((t) => t.frames > 0)).toBe(true);
      const voicedN = f0Voiced.reduce((s, v) => s + v, 0);
      // eslint-disable-next-line no-console
      console.log(
        `[e1dump] ${name}: ${meta.n_notes} notes -> ${triples.length} triples, ${sum} frames ` +
        `(${(sum / 50).toFixed(2)}s @50fps), voiced ${voicedN}/${sum}`,
      );
      fs.writeFileSync(
        join(WORK_DIR, `${name}_score.json`),
        JSON.stringify({ name, tempo, triples, f0Cents, f0Voiced }),
      );
    }
  });
});
