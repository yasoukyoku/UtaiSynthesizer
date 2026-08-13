/**
 * §E2E-M10 —— 参数页那条「改这一项会重跑预处理」的提示,第一次有判据。
 *
 * ⛔ 这份文件**不**证明那句话到达了屏幕。五个挂点还在不在、每个 backend 的分支有没有漏挂,
 * 由 `components/training/rowIdentityWiring.test.ts` 的源码闸看着 —— 那条是第一位的,因为
 * 在它之前,把整段提示删光全仓一条判据都不会红。
 *
 * ## 这里的两条阴性对照分别在防什么
 *
 * ⒜ **合成锁表**(`poolCostIdsIn`)—— 唯一能对 `scope` 那一半说话的入口。真表里三条 costly
 *    全是 `scope: "pool"`,所以在真 backend 上 tier 与 scope 给出**同一个答案**,把 scope
 *    那一半删掉一个像素都不变(S128 的变异 L9 就是这么「存活」的:它是一条**等价变异**)。
 * ⒝ **差分模糊**(搬家前的原文 vs 今天的两个函数,随机语料逐个对拍)—— S142 这一笔声称是
 *    纯提取,而「我读了代码,一样」不是证据。⚠ 配套的**语料自检**同样是承重的:一个走不到
 *    有分辨力形状的生成器,跑两千次也只是把同一格重复了两千次。
 */
import { describe, expect, it } from "vitest";
import { poolAtStake, poolCostFieldIds, poolCostIdsIn } from "./costlyNote";
import {
  lockedFieldIds,
  poolInvalidatingIds,
  resumeLockedFields,
  resumeWouldBeGuarded,
  type LockedField,
} from "../resumeLock";
import type { TrainingBackend, WorkspaceInfo } from "../../store/training";

/** 一个**跑过**的槽(与 `formForSlot.test.ts:19` 同形:`version` 非空 = manifest 存在)。 */
function ws(over: Partial<WorkspaceInfo> = {}): WorkspaceInfo {
  return {
    exists: true,
    family: "sovits",
    version: "4.1",
    sample_rate: "44k",
    has_main_progress: true,
    diff_steps: 0,
    best_resume_step: null,
    diff_best_resume_step: null,
    aug_copies: 0,
    loudnorm: null,
    has_dataset: true,
    vol_embedding: null,
    n_speakers: 1,
    speakers: [],
    diff_k_step_max: 0,
    ...over,
  };
}

const ALL_BACKENDS: TrainingBackend[] = ["rvc", "sovits", "sovits_v2", "sovits_diff", "vocoder"];

const ids = (s: Set<string>): string[] => [...s].sort();

// ─── ⑴ 搬家之前的原文(`TrainingPage.tsx` 的 :876-884,S142 之前)────────────────────────
//     ⛔ 原样抄下来,一个字都不许「顺手改好」—— 它是对照臂,不是产品代码。
function legacyCostly(
  backend: TrainingBackend,
  info: WorkspaceInfo | null | undefined,
  retrainIntent: boolean,
): Set<string> {
  const slotHasPreprocessing =
    !retrainIntent && !!info?.exists && (info.version !== "" || info.sample_rate !== "");
  const poolIds = poolInvalidatingIds(backend);
  return slotHasPreprocessing
    ? new Set([...lockedFieldIds(backend, "costly")].filter((id) => poolIds.has(id)))
    : new Set<string>();
}

// ─── ⑵ 语料 ──────────────────────────────────────────────────────────────────────────
interface Sample {
  backend: TrainingBackend;
  info: WorkspaceInfo | null;
  retrainIntent: boolean;
}

/** 固定种子:同一份语料每次跑逐个相同,红了能重放。 */
function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function corpus(n: number, seed = 20260813): Sample[] {
  const r = rng(seed);
  const pick = <T,>(xs: readonly T[]): T => xs[Math.floor(r() * xs.length)]!;
  const out: Sample[] = [];
  for (let i = 0; i < n; i++) {
    const backend = pick(ALL_BACKENDS);
    // `null` = 探针失败(`TrainingPage.tsx` 的 catch 把它吞成 null)。它是一个真实形状,不是补白。
    const info = r() < 0.12
      ? null
      : ws({
          exists: r() < 0.85,
          version: pick(["", "4.1", "4.0", "v2", "nsf_hifigan"]),
          sample_rate: pick(["", "44k", "40k", "48k"]),
          diff_steps: pick([0, 0, 1200]),
          has_main_progress: r() < 0.5,
        });
    out.push({ backend, info, retrainIntent: r() < 0.3 });
  }
  return out;
}

/**
 * ★ 语料自检 —— 一份**没有分辨力**的语料会让上面那条差分对拍变成一条恒真的断言。
 *
 * 它自己也要有阴性对照(见下面那条 it):把语料塌成一格,这个函数必须当场抛。
 */
function assertCorpusHasPower(samples: Sample[]): void {
  const seen = {
    backends: new Set(samples.map((s) => s.backend)),
    retrain: new Set(samples.map((s) => s.retrainIntent)),
    nullInfo: samples.some((s) => s.info === null),
    // 已跑过 + 已存在 ⇒ 提示应该挂出来的那一格
    live: samples.some((s) => !s.retrainIntent && s.info?.exists && s.info.version !== ""),
    // ★ diff-first:跑过、有池,但 manifest 的两个键都空、进度记在 diff_steps 上。
    //   这一格同时是「不许把 poolAtStake 合并进 resumeWouldBeGuarded」的唯一守卫。
    diffFirst: samples.some(
      (s) =>
        s.info?.exists === true &&
        s.info.version === "" &&
        s.info.sample_rate === "" &&
        s.info.diff_steps > 0,
    ),
  };
  const sigs = new Set(
    samples.map((s) => `${s.backend}:${ids(poolCostFieldIds(s.backend, poolAtStake(s.info, s.retrainIntent))).join(",")}`),
  );
  if (seen.backends.size !== ALL_BACKENDS.length) throw new Error(`语料只覆盖了 ${seen.backends.size} 个 backend`);
  if (seen.retrain.size !== 2) throw new Error("语料里 retrainIntent 只有一种取值");
  if (!seen.nullInfo) throw new Error("语料里没有探针失败(info=null)那一格");
  if (!seen.live) throw new Error("语料里没有一格是「该挂提示」的 —— 那条对拍只在比空集");
  if (!seen.diffFirst) throw new Error("语料里没有 diff-first 那一格 —— 合并守卫是空的");
  if (sigs.size < 6) throw new Error(`语料只产出 ${sigs.size} 种结果签名`);
}

// ─── ⑶ 判据 ──────────────────────────────────────────────────────────────────────────

describe("poolCostIdsIn —— 只按 scope 说话(唯一能驱动那一半的入口)", () => {
  /**
   * ★★ 这条是**阴性对照③**,而它必须吃合成表:真表里三条 costly 全是 `scope: "pool"`,
   * 所以在任何真 backend 上「按 tier 选」与「按 tier+scope 选」给出同一个答案。
   */
  it("★ 一行 costly 但 run 级的字段【必须不挂】,一行 pool 级但 locked 的也不挂", () => {
    const synthetic: LockedField[] = [
      { id: "poolCostly", tier: "costly", scope: "pool" },
      { id: "bothCostly", tier: "costly", scope: "both" },
      { id: "runCostly", tier: "costly", scope: "run" }, // ← scope 那一半的唯一看守
      { id: "poolLocked", tier: "locked", scope: "pool" }, // ← tier 那一半的唯一看守
    ];
    expect(ids(poolCostIdsIn(synthetic))).toEqual(["bothCostly", "poolCostly"]);
  });

  it("空表 ⇒ 空集(别让「一个都没有」和「没在算」长得一样)", () => {
    expect(poolCostIdsIn([]).size).toBe(0);
  });

  /**
   * ⚠ 这条钉的是一个**巧合**,不是一条定义 —— `resumeLock.ts` 自己也这么写着。
   * 它在这里的用处是:巧合被打破的那天(有人加了一行 costly + run),这条会红,
   * 而红的时候上面那条合成表判据已经说清了正确行为是什么。
   */
  it("★ 今天真表上 tier 与 scope 恰好等价 —— 钉住这个巧合,别让它静默变化", () => {
    for (const b of ALL_BACKENDS) {
      expect(ids(poolCostIdsIn(resumeLockedFields(b))), b).toEqual(ids(lockedFieldIds(b, "costly")));
    }
  });
});

describe("poolCostFieldIds —— 槽里没有池就一个都不挂", () => {
  it("有池 ⇒ 逐 backend 给出该 backend 的 costly 池级集合", () => {
    expect(ids(poolCostFieldIds("rvc", true))).toEqual(["augCopies", "dataset"]);
    expect(ids(poolCostFieldIds("sovits", true))).toEqual(["augCopies", "dataset", "loudnorm"]);
    expect(ids(poolCostFieldIds("sovits_v2", true))).toEqual(["augCopies", "dataset", "loudnorm"]);
    expect(ids(poolCostFieldIds("sovits_diff", true))).toEqual(["augCopies", "dataset"]);
    expect(ids(poolCostFieldIds("vocoder", true))).toEqual(["augCopies", "dataset"]);
  });

  // ★ 与上一条**成对**:只有上面那条,一个恒返回全集的实现也全绿;只有下面这条,
  //   一个恒返回空集的实现也全绿。分开它们的是同一个 backend 的两个 poolAtStake 取值。
  it("★ 没池 ⇒ 每个 backend 都是空集", () => {
    for (const b of ALL_BACKENDS) expect(poolCostFieldIds(b, false).size, b).toBe(0);
  });
});

describe("poolAtStake —— 今天那个近似,以及它已知偏窄的两格", () => {
  it("跑过的槽 + 不是重训 ⇒ 有池", () => {
    expect(poolAtStake(ws(), false)).toBe(true);
  });

  it("★ 槽从没跑过(manifest 不存在)⇒ 没池 —— 阴性对照②", () => {
    expect(poolAtStake(ws({ version: "", sample_rate: "" }), false)).toBe(false);
    expect(poolAtStake(ws({ exists: false }), false)).toBe(false);
  });

  /**
   * ⛔⛔ **双向 pin,不是「它是对的」** —— S110 的形状。
   *
   * 这一臂的原始理由是「再训一个会清空整槽」,而 S132 的 flip 之后那句话是假的:旧 run 原样
   * 留着,池是**槽级、内容寻址、跨 run 共享**的 ⇒ 重训路径上改 `augCopies` 照样铸新池、整份
   * 预处理重跑,而屏幕上一个字都没有。同一个标志还把 `locked` 那一档(含 `scope: "both"` 的
   * `sampleRate`)一起关掉 ⇒ **唯一被设计成可以改这些字段的那条路,恰好两档提示同时消失。**
   *
   * 钉住它是为了让「改它」必须是一次自觉的改动。修法与判据在队列 §E2E-M10-⒜。
   */
  it("⛔ 已知偏窄⒜:重训路径上答 false —— 而那时池还在,代价也还在", () => {
    expect(poolAtStake(ws(), true)).toBe(false);
  });

  /**
   * ⛔⛔ 双向 pin⒝:`version` / `sample_rate` 来自 manifest,而扩散的进度记在 `diff_steps`
   * 上 ⇒ 一个已经跑过、已经有池的 diff-first 槽在这里读起来像「从没跑过」。
   */
  it("⛔ 已知偏窄⒝:diff-first 的槽(manifest 两键为空、diff_steps>0)答 false", () => {
    const diffFirst = ws({ version: "", sample_rate: "", diff_steps: 1200 });
    expect(poolAtStake(diffFirst, false)).toBe(false);
  });

  /**
   * ★ 这条是「**不许合并**」的唯一守卫:最省力的抽法是直接复用 `resumeWouldBeGuarded`,
   * 而那是一次**静默的行为改动** —— 两个谓词在 sovits_diff 上有意分叉。
   */
  it("★ 同一格上 resumeWouldBeGuarded 与 poolAtStake 给出【相反】的答案", () => {
    const diffFirst = ws({ version: "", sample_rate: "", diff_steps: 1200 });
    expect(resumeWouldBeGuarded("sovits_diff", diffFirst)).toBe(true);
    expect(poolAtStake(diffFirst, false)).toBe(false);
  });

  /**
   * ⚠ 探针失败(`get_training_slot_info` 抛 ⇒ 组件 catch 成 `null`)⇒ 提示整体消失。
   * 这是一条**被写下来的取舍**(fail-open),不是一条被覆盖了的分支:对 `info=null` 返回
   * false 正是这个函数的正确行为,所以它永远打绿,而屏幕上那几小时的代价没有人说。
   * ⇒ 真正的判据不在这里,在「探针失败要留下可见的 CODE」那一侧(队列 §E2E-M10-⒞)。
   */
  it("探针失败(info=null)⇒ 答 false(fail-open,取舍已记)", () => {
    expect(poolAtStake(null, false)).toBe(false);
    expect(poolAtStake(undefined, false)).toBe(false);
  });
});

describe("★ 搬家前后逐个输入对拍(纯提取的证据形状)", () => {
  it("★ 语料自检:它必须真的走得到有分辨力的形状", () => {
    expect(() => assertCorpusHasPower(corpus(2000))).not.toThrow();
  });

  // ★ 自检自己的阴性对照:把语料塌成一格,自检必须当场抛。少了这条,自检可能是一句空话。
  it("★ 语料塌成一格时,自检必须抛(否则它什么也没检)", () => {
    const flat: Sample[] = Array.from({ length: 2000 }, () => ({
      backend: "rvc" as TrainingBackend,
      info: ws(),
      retrainIntent: false,
    }));
    expect(() => assertCorpusHasPower(flat)).toThrow();
  });

  it("★ 2000 个随机槽上,新旧两条路逐个给出同一个集合", () => {
    const samples = corpus(2000);
    assertCorpusHasPower(samples);
    for (const s of samples) {
      const now = poolCostFieldIds(s.backend, poolAtStake(s.info, s.retrainIntent));
      const before = legacyCostly(s.backend, s.info, s.retrainIntent);
      expect(ids(now), JSON.stringify({ backend: s.backend, retrainIntent: s.retrainIntent, info: s.info })).toEqual(
        ids(before),
      );
    }
  });
});
