// E1 K 臂(S71 旋钮线第四刀)— 预测 θ 的参数化 f0 dump 工具,不是 gate。
// 目的:把 SVC2SVS Phase A 模型对 E1 谱段预测的 note 级 θ(labels/karm/{seg}_theta.json)
// 套进真前端 Note.transition/vibrato,走生产 buildVocalScore 出 Option-A f0,dump 成
// {seg}_score_K.json 给 Rust e1 harness 的 K 臂消费(与 D 臂唯一差异=调教参数;零复制铁律)。
// 默认 SKIP(UTAI_E1K_DUMP=1 才跑)。运行:
//   $env:UTAI_E1K_DUMP='1'; npx vitest run src/lib/vocal/e1KarmDump.test.ts
// S71+1 第二轮多变体:UTAI_E1K_SRC=theta 目录(默认 SVC2SVS labels/karm)、
// UTAI_E1K_TAG=输出后缀(默认 K → {seg}_score_{TAG}.json;如 KA / KAB72,与 Rust 侧
// UTAI_E1K_TAG 配对使用)。
import { describe, it, expect, vi } from "vitest";
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

vi.mock("@tauri-apps/api/core", () => ({ invoke: () => Promise.resolve() }));
vi.mock("../../i18n", () => ({ default: { t: (k: string) => k } }));
vi.mock("../../store/voice-models", () => ({ useVoiceModelStore: { getState: () => ({ models: { sovits: [] } }) } }));

import { buildVocalScore } from "./vocalRender";
import { DEFAULT_TRANSITION } from "../vocalNotes";
import type { Note } from "../../types/project";

const WORK_DIR = "D:\\MyDev\\TESTING\\e1_cross_probe";
const KARM_DIR = process.env.UTAI_E1K_SRC ?? "D:\\MyDev\\SVC2SVS\\labels\\karm";
const KTAG = process.env.UTAI_E1K_TAG ?? "K";
const SEGMENTS = ["verse", "chorus"];

describe.skipIf(!process.env.UTAI_E1K_DUMP)("E1 K-arm param-f0 dump (diagnostic, not a gate)", () => {
  it("dumps knob-model Option-A f0 for each E1 segment", async () => {
    const fs = await importFs();
    const join = (...parts: string[]) => parts.join("\\");
    const manifest = JSON.parse(fs.readFileSync(join(WORK_DIR, "e1_manifest.json"), "utf-8"));
    const tempo: number = manifest.tempo;
    for (const name of SEGMENTS) {
      const meta = JSON.parse(fs.readFileSync(join(WORK_DIR, `${name}_notes.json`), "utf-8"));
      const thetaPath = join(KARM_DIR, `${name}_theta.json`);
      expect(fs.existsSync(thetaPath), `missing ${thetaPath} (run pitch/export_karm.py first)`).toBe(true);
      const theta = JSON.parse(fs.readFileSync(thetaPath, "utf-8"));
      expect(theta.notes.length).toBe(meta.notes.length);
      const notes: Note[] = meta.notes.map(
        (n: { tick: number; duration: number; pitch: number; lyric: string }, i: number) => {
          const th = theta.notes[i]!;
          // 索引对账:θ 行必须锚在同一个 note 上(tickStart 来自 export 时的同一份 notes.json)
          expect(th.tickStart, `${name} n${i}: theta/notes 错位`).toBe(n.tick);
          return {
            id: `n${i}`, tick: n.tick, duration: n.duration, pitch: n.pitch, lyric: n.lyric,
            velocity: 100,
            transition: { ...th.transition }, // 全 6 字段=完全覆盖 track default(per-field 合并)
            vibrato: { ...th.vibrato },       // 全 6 字段(phase=0)
          };
        },
      );
      const { triples, f0Cents, f0Voiced, loudnessEnv, formantEnv } = buildVocalScore(
        notes, undefined, tempo, DEFAULT_TRANSITION, "AP",
      );
      const sum = triples.reduce((s, t) => s + t.frames, 0);
      expect(f0Cents.length).toBe(sum);
      expect(f0Voiced.length).toBe(sum);
      expect(loudnessEnv.length).toBe(0);
      expect(formantEnv.length).toBe(0);
      // 与 D 臂 dump 的谱面一致性:θ 只动调教,triples(歌词/音高/帧数)必须逐项等同
      const dPath = join(WORK_DIR, `${name}_score.json`);
      if (fs.existsSync(dPath)) {
        const d = JSON.parse(fs.readFileSync(dPath, "utf-8"));
        expect(JSON.stringify(triples), `${name}: triples 漂移(θ 不该动谱面)`).toBe(
          JSON.stringify(d.triples),
        );
      }
      fs.writeFileSync(
        join(WORK_DIR, `${name}_score_${KTAG}.json`),
        JSON.stringify({ name, tempo, ckpt: theta.ckpt, triples, f0Cents, f0Voiced }),
      );
      // eslint-disable-next-line no-console
      console.log(`[e1k] ${name} -> ${name}_score_${KTAG}.json (${notes.length} notes, ckpt ${theta.ckpt})`);
    }
  });
});
