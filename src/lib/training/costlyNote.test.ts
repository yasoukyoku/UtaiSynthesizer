/**
 * §E2E-M10 —— 参数页那条「改这一项会重跑预处理」的提示。
 *
 * ⛔ 这份文件**不**证明那句话到达了屏幕。五个挂点还在不在、每个 backend 的分支有没有漏挂,
 * 由 `components/training/rowIdentityWiring.test.ts` 的源码闸看着 —— 那条是第一位的,因为
 * 在它之前,把整段提示删光全仓一条判据都不会红。
 *
 * ## 这里的三条阴性对照分别在防什么
 *
 * ⒜ **合成锁表**(`poolCostIdsIn`)—— 唯一能对 `scope` 那一半说话的入口。真表里三条 costly
 *    全是 `scope: "pool"`,所以在真 backend 上 tier 与 scope 给出**同一个答案**,把 scope
 *    那一半删掉一个像素都不变(S128 的变异 L9 就是这么「存活」的:它是一条**等价变异**)。
 * ⒝ **邻居字段**(`poolAtStake`)—— `has_preprocessing` 与 `has_dataset` 在同一个 struct 里
 *    隔了几行,而 Rust 那边用整段中文写着「别混」。**一段没有闸的警告等于没写**:所以语料
 *    把这两个字段放在**互相独立**的轴上,并有两格具名的反向夹具。
 * ⒞ **差分**(搬家前 vs 今天)—— 笔 3 **有意**改了行为,所以这一条不再是「处处相同」,而是
 *    「**只在这两格分歧,其余逐个相同**」,并且要求两个分歧方向在语料里都真的出现过。
 *    ⚠ 配套的**语料自检**同样承重:一个走不到分歧形状的生成器,跑两千次也只是把同一格
 *    重复了两千次。
 */
import { describe, expect, it } from "vitest";
import { poolAtStake, poolCostFieldIds, poolCostIdsIn } from "./costlyNote";
import { lockedFieldIds, poolInvalidatingIds, resumeLockedFields, resumeWouldBeGuarded, type LockedField } from "../resumeLock";
import type { TrainingBackend, WorkspaceInfo } from "../../store/training";

/** 一个**跑过**的槽(与 `formForSlot.test.ts` 同形)。 */
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
    has_preprocessing: true,
    vol_embedding: null,
    n_speakers: 1,
    speakers: [],
    diff_k_step_max: 0,
    ...over,
  };
}

const ALL_BACKENDS: TrainingBackend[] = ["rvc", "sovits", "sovits_v2", "sovits_diff", "vocoder"];

const ids = (s: Set<string>): string[] => [...s].sort();

// ─── ⑴ 搬家之前的原文(`TrainingPage.tsx` 的 :876-884,S142 笔 1 之前)──────────────────
//     ⛔ 原样抄下来,一个字都不许「顺手改好」—— 它是对照臂,不是产品代码。
function legacyPoolAtStake(info: WorkspaceInfo | null | undefined, retrainIntent: boolean): boolean {
  return !retrainIntent && !!info?.exists && (info.version !== "" || info.sample_rate !== "");
}

function legacyCostly(
  backend: TrainingBackend,
  info: WorkspaceInfo | null | undefined,
  retrainIntent: boolean,
): Set<string> {
  const poolIds = poolInvalidatingIds(backend);
  return legacyPoolAtStake(info, retrainIntent)
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
          // ★ 两条**互相独立**的轴 —— 这是「读成邻居字段」那条手误的唯一看守(见头注⒝)。
          has_preprocessing: r() < 0.5,
          has_dataset: r() < 0.5,
        });
    out.push({ backend, info, retrainIntent: r() < 0.3 });
  }
  return out;
}

/**
 * ★ 语料自检 —— 一份**没有分辨力**的语料会让下面那条差分对拍变成一句恒真的话。
 *
 * 它自己也要有阴性对照(见下面那条 it):把语料塌成一格,这个函数必须当场抛。
 */
function assertCorpusHasPower(samples: Sample[]): void {
  const has = (f: (s: Sample) => boolean) => samples.some(f);
  const diverges = (s: Sample) => legacyPoolAtStake(s.info, s.retrainIntent) !== poolAtStake(s.info);

  const backends = new Set(samples.map((s) => s.backend));
  if (backends.size !== ALL_BACKENDS.length) throw new Error(`语料只覆盖了 ${backends.size} 个 backend`);
  if (new Set(samples.map((s) => s.retrainIntent)).size !== 2) throw new Error("retrainIntent 只有一种取值");
  if (!has((s) => s.info === null)) throw new Error("语料里没有探针失败(info=null)那一格");
  // ★ 新旧两个谓词的**输入轴**都必须活着
  if (!has((s) => s.info?.has_preprocessing === true)) throw new Error("语料里没有「有池」那一格");
  if (!has((s) => s.info?.has_preprocessing === false)) throw new Error("语料里没有「没池」那一格");
  // ★ 邻居字段必须出现**两个方向**的不一致,否则读错字段的实现照样全绿
  if (!has((s) => s.info?.has_preprocessing === true && s.info.has_dataset === false)) {
    throw new Error("语料里没有「有池但项目没数据集」那一格 —— 读错邻居字段就没人抓了");
  }
  if (!has((s) => s.info?.has_preprocessing === false && s.info.has_dataset === true)) {
    throw new Error("语料里没有「有数据集但没池」那一格");
  }
  // ★★ 笔 3 改了行为 ⇒ 分歧必须在语料里**两个方向都真的出现过**,否则「只在这两格分歧」
  //    这句话是靠没走到那些格子成立的。
  if (!has((s) => diverges(s) && poolAtStake(s.info))) {
    throw new Error("语料里没有「旧的说不挂、新的说要挂」那一格(⒜/⒝ 修好的正是它)");
  }
  if (!has((s) => diverges(s) && !poolAtStake(s.info))) {
    throw new Error("语料里没有「旧的说要挂、新的说不挂」那一格(manifest 在但盘上没池)");
  }
  // ★ diff-first:跑过、有池,但 manifest 两键为空、进度记在 diff_steps 上。
  //   这一格同时是「不许把 poolAtStake 合并进 resumeWouldBeGuarded」的唯一守卫。
  if (
    !has(
      (s) =>
        s.info?.exists === true &&
        s.info.version === "" &&
        s.info.sample_rate === "" &&
        s.info.diff_steps > 0,
    )
  ) {
    throw new Error("语料里没有 diff-first 那一格 —— 合并守卫是空的");
  }
  const sigs = new Set(
    samples.map((s) => `${s.backend}:${ids(poolCostFieldIds(s.backend, poolAtStake(s.info))).join(",")}`),
  );
  if (sigs.size < 6) throw new Error(`语料只产出 ${sigs.size} 种结果签名`);
}

// ─── ⑶ 判据 ──────────────────────────────────────────────────────────────────────────

describe("poolCostIdsIn —— 只按 scope 说话(唯一能驱动那一半的入口)", () => {
  /**
   * ★★ 这条是**阴性对照⒜**,而它必须吃合成表:真表里三条 costly 全是 `scope: "pool"`,
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

describe("poolAtStake —— 直接问盘,而不是从 manifest 猜", () => {
  it("盘上有预处理 ⇒ true;没有 ⇒ false", () => {
    expect(poolAtStake(ws({ has_preprocessing: true }))).toBe(true);
    expect(poolAtStake(ws({ has_preprocessing: false }))).toBe(false);
  });

  /**
   * ★★ 阴性对照⒝ —— 「读成隔壁那个字段」是这一跳最像会真发生的手误:两个字段在同一个
   * struct 里隔几行,而 Rust 那边用整段中文写着「别混」。**这两格让那段话有了闸。**
   */
  it("★ 有池但项目没数据集 ⇒ 仍然要挂(读成 has_dataset 会答反)", () => {
    expect(poolAtStake(ws({ has_preprocessing: true, has_dataset: false }))).toBe(true);
  });

  it("★ 有数据集但这个槽还没预处理过 ⇒ 不挂(读成 has_dataset 会答反)", () => {
    expect(poolAtStake(ws({ has_preprocessing: false, has_dataset: true }))).toBe(false);
  });

  /**
   * ★ **决定性夹具**:manifest 在(旧谓词答「跑过」)而盘上没有池。
   * ⛔ 别把它写成「两个 manifest 键一起抹掉」—— 那一格新旧两个谓词**答案相同**,于是一个
   * 收下了 `has_preprocessing` 却继续用 manifest 的实现照样全绿。
   */
  it("★ manifest 在、盘上没池(刚起步就被杀掉的 run)⇒ 不挂", () => {
    expect(poolAtStake(ws({ version: "4.1", sample_rate: "44k", has_preprocessing: false }))).toBe(false);
  });

  /**
   * ★★ ⒜ 修好了。⛔ **但这条判据的夹具是承重的**:第一版用的是 `ws({has_preprocessing:true})`,
   * 而那份夹具的 `version` 是 `"4.1"` ⇒ **旧谓词在那一格上也答 true** ⇒ 把 `poolAtStake` 整个
   * 退回旧式子,那条测试**照样绿**(变异 C4 实测:它红在别的断言上)。一条测不出被测改动的
   * 断言就是一条装饰件。⇒ 夹具必须落在两个谓词**分歧**的那一侧:盘上有池,而 manifest 说不
   * 上话(刚折完 layout 的槽 / diff-first 的槽都是这个形状)。
   *
   * ⚠ 而「答案不再随 `retrainIntent` 变」这件事在**类型层**就成立了(它不再是入参)——
   * 单测说不出这句话,守它的是 `rowIdentityWiring` 里那条「调用点不许再把 retrainIntent
   * 喂进来」的源码闸。
   */
  it("★ 盘上有池而 manifest 说不上话 ⇒ 照样挂(旧谓词在这一格是瞎的)", () => {
    const info = ws({ version: "", sample_rate: "", has_preprocessing: true });
    expect(poolAtStake(info)).toBe(true);
    // 双向对照:旧行为在这一格答 false,**而且与是不是重训无关** —— 它压根看不见盘。
    expect(legacyPoolAtStake(info, false)).toBe(false);
    expect(legacyPoolAtStake(info, true)).toBe(false);
  });

  /**
   * ★★ ⒝ 修好了:diff-first 的槽(manifest 两键为空、进度在 diff_steps 上)有池就挂。
   * ★ 同一格仍然是「**不许合并** `resumeWouldBeGuarded`」的守卫 —— 两个谓词依然分叉。
   */
  it("★ diff-first 的槽有池 ⇒ 挂;而 resumeWouldBeGuarded 在同一格上答的是另一件事", () => {
    const diffFirst = ws({ version: "", sample_rate: "", diff_steps: 1200, has_preprocessing: true });
    expect(poolAtStake(diffFirst)).toBe(true);
    expect(legacyPoolAtStake(diffFirst, false)).toBe(false); // 旧行为在这里是瞎的
    // 两个谓词依然不是同一件事:这一格它们恰好同为 true,而下面那一格它们相反。
    const diffNoPool = ws({ version: "", sample_rate: "", diff_steps: 1200, has_preprocessing: false });
    expect(resumeWouldBeGuarded("sovits_diff", diffNoPool)).toBe(true);
    expect(poolAtStake(diffNoPool)).toBe(false);
  });

  /**
   * ⚠ 探针失败(`get_training_slot_info` 抛 ⇒ 组件 catch 成 `null`)⇒ 提示整体消失。
   * 这是一条**被写下来的取舍**(fail-open),不是一条被覆盖了的分支:对 `info=null` 返回
   * false 正是这个函数的正确行为,所以它永远打绿,而屏幕上那几小时的代价没有人说。
   * ⇒ 真正的判据不在这里,在「探针失败要留下可见的 CODE」那一侧(队列 §E2E-M10-⒞)。
   * ⚠ 反过来 Rust 那一侧是 fail-**closed** 的(`pools/` 读不动 ⇒ 答「有」)。
   */
  it("探针失败(info=null)⇒ 答 false(fail-open,取舍已记)", () => {
    expect(poolAtStake(null)).toBe(false);
    expect(poolAtStake(undefined)).toBe(false);
  });
});

describe("★ 笔 3 到底改了什么:新旧逐个输入对拍", () => {
  it("★ 语料自检:它必须真的走得到有分辨力的形状(含两个分歧方向)", () => {
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

  /**
   * ★★ 笔 3 **有意**改了行为,所以这一条不是「处处相同」,而是**「只在谓词分歧的那些格子上
   * 不同,其余 2000 格逐个相同」** —— 它把这次改动的边界钉成了一条会红的判据。
   */
  it("★ 两个谓词一致时集合逐个相同;不一致时【只】由 poolAtStake 说了算", () => {
    const samples = corpus(2000);
    assertCorpusHasPower(samples);
    for (const s of samples) {
      const now = poolCostFieldIds(s.backend, poolAtStake(s.info));
      const before = legacyCostly(s.backend, s.info, s.retrainIntent);
      const why = JSON.stringify({ backend: s.backend, retrainIntent: s.retrainIntent, info: s.info });
      if (legacyPoolAtStake(s.info, s.retrainIntent) === poolAtStake(s.info)) {
        expect(ids(now), why).toEqual(ids(before));
      } else {
        // 分歧格:新答案完全由「盘上有没有池」决定,与 manifest / retrainIntent 无关。
        expect(ids(now), why).toEqual(
          poolAtStake(s.info) ? ids(poolCostIdsIn(resumeLockedFields(s.backend))) : [],
        );
        expect(ids(now), why).not.toEqual(ids(before));
      }
    }
  });
});
