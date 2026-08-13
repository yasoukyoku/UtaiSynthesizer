/**
 * 参数页上「改这一项会重跑预处理」那句话,到底该挂在哪几个字段上 —— §E2E-M10。
 *
 * ## 为什么要把它从组件体里搬出来
 *
 * 这个决策此前是 `TrainingPage.tsx` 里 `ParamsStep` 组件体内的三行内联表达式,vitest
 * **结构上**够不着(`vitest.config.ts` 明写不做渲染)。S128 建它的时候就知道这一点,并留下了
 * 一条**如实存活**的变异(L9:把 scope 那一半的过滤去掉)——而 S142 量清楚了那条变异为什么
 * 杀不死,见下面「⛔ 这一层为什么要吃一张表」。
 *
 * 它守的东西:一个 costly 字段改下去**不会被拒**,而是让下一次运行落到**另一个**预处理池上,
 * 把切片与全部伴生特征重跑一遍(数据集越大越久)。屏幕上这句话是用户在按下去之前
 * **唯一**能知道这件事的地方 —— 而它整段消失时没有任何异常、没有 CODE、没有 toast。
 *
 * ## ⛔ 这一层为什么要吃一张表,而不是只吃一个 backend 名字
 *
 * 判据是 **scope 不是 tier**:`tier=costly` 说的是「续训改它不会被拒」,而屏幕上这句话说的是
 * 「改了会重跑预处理」—— 那是 `scope` 回答的(`resumeLock.ts` 的 `scopeInvalidatesPool`)。
 * 两者**今天恰好等价**:锁表里三条 costly 全是 `scope: "pool"`,所以
 * `lockedFieldIds(b, "costly")` 对五个 backend 都已经是 `poolInvalidatingIds(b)` 的子集。
 *
 * ⇒ **把 scope 那一半整个删掉,今天一个像素都不会变。** 也就是说任何只喂真 backend 的判据
 * ——不管它写得多细——**结构上**都杀不死那条变异,而它看起来像覆盖。
 * 唯一能让那一跳有话说的,是喂一张**合成的**锁表(带一行 `tier=costly, scope=run`),
 * 所以 [`poolCostIdsIn`] 吃 `rows` 而不是吃 `backend`。
 *
 * ⛔ 而且**两半必须读同一份 `rows`**:如果 tier 那一半吃注入的表、scope 那一半仍然回头去调
 * `poolInvalidatingIds(backend)`,那么注入的那一行会因为**名字不在真表里**被剔掉 ——
 * 断言照样绿,但绿的理由与 scope 无关,那条阴性对照就成了装饰。
 *
 * ⛔ 也**不许**为了做这条对照去真的往 `resume_lock.rs` / `resumeLock.ts` 里加一行:
 * 那会同时打红跨语言对拍(`resumeLockParity.test.ts` 的 `id:tier:scope` 钥匙)与
 * `formForSlot.test.ts` 的两条棘轮。表那一层已经有判据了(S128 的变异 L6 把
 * `costly("augCopies", LockScope::Pool)` 改成 `Run`,cargo 侧实测 RED)。
 *
 * ## 这一层【不做】决定的事
 *
 * * **不决定这句话有没有到达屏幕。** 本模块只算集合;「五个挂点还在不在、每个 backend 的分支
 *   有没有漏挂」由 `rowIdentityWiring.test.ts` 的源码闸看着 —— 那条闸是**第一位**的,因为
 *   今天把整段提示删掉,全仓没有一条判据会红(i18n 闸只钉结构、死键闭合只管 `backend.*`)。
 * * **不合并 `resumeWouldBeGuarded`。** 那个谓词在 `sovits_diff` 上走 `diff_steps > 0`,
 *   而这里走 manifest —— 两者**有意分叉**(锁问「这条链练到哪了」,代价问「这个槽的预处理」)。
 *   顺手复用它是一次**静默的行为改动**,`costlyNote.test.ts` 的语料专门有一格守着这条。
 */

import { resumeLockedFields, scopeInvalidatesPool, type LockedField } from "../resumeLock";
import { hasManifest } from "./formForSlot";
import type { TrainingBackend, WorkspaceInfo } from "../../store/training";

/**
 * 一张锁表里,**改了就得重跑预处理**的那些 id。
 *
 * 只按 `tier` + `scope` 说话,不认识 backend、不回头查任何别的表 —— 这是上面那条
 * 「两半必须读同一份 rows」的落点,也是阴性对照③唯一驱得动的入口。
 */
export function poolCostIdsIn(rows: readonly LockedField[]): Set<string> {
  return new Set(
    rows.filter((f) => f.tier === "costly" && scopeInvalidatesPool(f.scope)).map((f) => f.id),
  );
}

/**
 * 这个 backend 上,此刻要挂代价提示的那些字段。
 *
 * @param poolAtStake 这个槽里有没有一份**会被换掉**的预处理池。没有就一个都不挂:
 *   一个从没跑过预处理的槽,改什么都不需要付第二遍。
 */
export function poolCostFieldIds(
  backend: TrainingBackend,
  poolAtStake: boolean,
): Set<string> {
  return poolAtStake ? poolCostIdsIn(resumeLockedFields(backend)) : new Set<string>();
}

/**
 * 今天用来回答「这个槽里有没有一份会被换掉的预处理池」的谓词。
 *
 * ⛔⛔ **它是一个近似,而且已知偏窄 —— 钉住它不是因为它对。**
 *
 * 真正的问题是「这个**槽**有没有池」,而 `WorkspaceInfo` 今天不带任何池字段
 * (盘上那个事实是有的:Rust 的 `tpool::slot_has_pool`,以及项目页那条
 * `SlotDetail.prepPoolCount` —— 只是都没走到参数页够得着的对象上)。于是这里退而用
 * 「这个 **run** 的 manifest 说它跑过」来近似,而那个近似在两处偏窄:
 *
 * ⒜ **重训路径**(`retrainIntent`)—— 这一臂的原始理由是「再训一个会清空整槽,那时不存在
 *    换池这回事」。**S132 的 flip 之后那句话是假的**:旧 run 原样留着,而池是**槽级、
 *    内容寻址、跨 run 共享**的。所以在重训路径上改 `augCopies` / `sampleRate` 照样铸新池、
 *    整份预处理重跑,而屏幕上一个字都没有 —— 并且同一个标志还把 `locked` 那一档
 *    (里面装着 `scope: "both"` 的 `sampleRate`)一起关掉了,也就是说**唯一被设计成可以改
 *    这些字段的那条路,恰好是两档提示同时消失的那条**。
 * ⒝ **diff-first 的槽** —— `version` / `sample_rate` 来自 manifest,而扩散的进度记在
 *    `diff_steps` 上 ⇒ 一个已经跑过、已经有池的 diff-first 槽在这里读起来像「从没跑过」。
 *
 * ⇒ 单测把这两格**钉在今天的答案上**(双向 pin,S110 的形状):不是宣布它们对,而是让
 * 「改它」必须是一次**自觉的**改动,而不是一次谁也没注意到的漂移。修法与判据记在队列。
 */
export function poolAtStake(
  info: WorkspaceInfo | null | undefined,
  retrainIntent: boolean,
): boolean {
  return !retrainIntent && hasManifest(info);
}
