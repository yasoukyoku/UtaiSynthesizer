// S84 鹅妈妈快段探针第 2 段(diagnostic,不是 gate)—— mg_notes.json(Rust parse_ust dump)→
// 真前端 buildVocalScore(默认 transition/无 vibrato/无 pitchDev = 装机版 UST 导入后实际状态,
// S83:PBS/PBW 未导入)→ mg_score.json(triples + 生产口径 f0)。零复制铁律同 e1CrossDump。
// 链全景见 src-tauri\src\inference\score2svc_mg.rs 头注。默认 SKIP(UTAI_MG_DUMP=1 才跑)。
// 运行:$env:UTAI_MG_DUMP='1'; npx vitest run src/lib/vocal/mgScoreDump.test.ts
import { describe, it, expect, vi } from "vitest";
// 前端 tsconfig 无 @types/node —— 变量说明符动态 import 拿 fs(e1CrossDump 同款,TS 对非字面量
// 说明符不做模块解析,运行时 node 原生解析)。
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

// vocalRender.ts 也导入 invoke/store/i18n(为 renderVocalSegment)— mock 使模块可无头加载。
vi.mock("@tauri-apps/api/core", () => ({ invoke: () => Promise.resolve() }));
vi.mock("../../i18n", () => ({ default: { t: (k: string) => k } }));
vi.mock("../../store/voice-models", () => ({ useVoiceModelStore: { getState: () => ({ models: { sovits: [] } }) } }));

import { buildVocalScore } from "./vocalRender";
import { DEFAULT_TRANSITION } from "../vocalNotes";
import type { Note } from "../../types/project";

const PROBE_DIR = "D:\\MyDev\\TESTING\\不为人所知的鹅妈妈童谣\\probe";

describe.skipIf(!process.env.UTAI_MG_DUMP)("MG score dump (diagnostic, not a gate)", () => {
  it("dumps production triples + default-tuning f0 for the Mother Goose UST", async () => {
    const fs = await importFs();
    const join = (...parts: string[]) => parts.join("\\");
    // S89: `UTAI_MG_NOTES` / `UTAI_MG_OUT` point the dump at any notes file in the same shape, so a
    // score exported straight out of the running app can be pushed through the SAME production
    // buildVocalScore — verifying on real user material without re-deriving triples anywhere.
    const notesPath = process.env.UTAI_MG_NOTES ?? join(PROBE_DIR, "mg_notes.json");
    const outPath = process.env.UTAI_MG_OUT ?? join(PROBE_DIR, "mg_score.json");
    expect(fs.existsSync(notesPath), `missing ${notesPath} (run dump_mg_notes first)`).toBe(true);
    const meta = JSON.parse(fs.readFileSync(notesPath, "utf-8"));
    const tempo: number = meta.bpm ?? 222;
    const notes: Note[] = meta.notes.map(
      (n: { tick: number; duration: number; pitch: number; lyric: string }, i: number) => ({
        id: `n${i}`, tick: n.tick, duration: n.duration, pitch: n.pitch, lyric: n.lyric, velocity: 100,
      }),
    );
    // 生产口径:默认 transition(track default)、无 vibrato、无 pitchDev、无参数泳道、JA 默认。
    const { triples, f0Cents, f0Voiced, loudnessEnv, formantEnv } = buildVocalScore(
      notes, undefined, tempo, DEFAULT_TRANSITION, { breath: "AP", rest: "R" }, // 生产默认的两个标记
    );
    const sum = triples.reduce((s, t) => s + t.frames, 0);
    expect(f0Cents.length).toBe(sum); // render_vocal_segment 的硬校验同款不变量
    expect(f0Voiced.length).toBe(sum);
    expect(loudnessEnv.length).toBe(0);
    expect(formantEnv.length).toBe(0);
    const voicedN = f0Voiced.reduce((s, v) => s + v, 0);
    // eslint-disable-next-line no-console
    console.log(
      `[mgdump] ${meta.n_notes} notes -> ${triples.length} triples, ${sum} frames ` +
      `(${(sum / 50).toFixed(2)}s @50fps), voiced ${voicedN}/${sum}, tempo ${tempo}`,
    );
    fs.writeFileSync(outPath, JSON.stringify({ name: "mg", tempo, triples, f0Cents, f0Voiced }));
  });
});
