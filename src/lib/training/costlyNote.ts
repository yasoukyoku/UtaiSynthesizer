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
 * 这个 backend 的参数页上**真的改得动**的池级字段 —— ★S144 §E2E-M10-⒜′。
 *
 * ## ⛔ 为什么这是一张手写的表,而不是从锁表推出来的
 *
 * 仓里**已经写下过这条警告**,而它正好挡在最诱人的那条路上 ——
 * `src-tauri/tests/pool_identity_formula.rs`:「rvc 的 `sampleRate` 是 `Locked` 不是
 * `Costly`,所以上面那条按 tier 过滤的循环**看不见它** …… ⚠ 不能靠把 Locked 也拉进那条
 * 循环来补:`LockScope` 对「这个 family 恒定不变的项」填的是**假设性**答案(sovits 家的
 * 44k、vocoder 的 version 都写着 `Both`/`Run` 却根本改不动),拉进去就会要求给一堆永远不会
 * 变的常量各发一个 token。」
 * ⇒ 「locked ∧ 作废池」会把**四个 backend 的 `sampleRate`** 一起算进来,而参数页上
 * 结构性地没有它们的控件(全文 `locked.has(` 只有四处,`updateConfig({ sampleRate` 只有一处)。
 *
 * ⛔ 也**不许**改成「`POOL_FORM_FIELDS[backend]` 里有没有这个键」:那张表是
 * `rowIdentityWiring` 那道逐 backend 逐字段闸的**期望来源** —— 集合一旦按它定义,
 * 那条断言就恒真,而它是这一面**唯一**的存在性看守。⚠ 这不是推理:S144 实测过一次
 * ——把整段提示改成不渲染,全仓 437 条判据**一条都不红**。
 *
 * ⇒ 所以这里是一条**独立的事实**:这个 family 的参数页上有一个控件承载它,而且这个 family
 * 真的改得动它。加一行的代价是 `formForSlot.test.ts` 的棘轮②会要求一条 probe。
 *
 * ## 今天为什么只有 rvc 一行(四条边界,写出来是为了下一个人不必重推)
 *
 * * **rvc `sampleRate`** —— `scope: "both"` ⇒ 改了必然换池、整份预处理重跑(还多一道 16k
 *   重采样),而参数页**主栅格**里就是一个可编辑 Dropdown。这是这一格唯一的净收益。
 * * **另外四个 backend 的 `sampleRate`** —— 锁表同样写 `Both`,但那是**假设性**的:
 *   44.1k 是常量,参数页上只有只读展示,没有控件。
 * * **sovits 家的 `version`** —— 它**确实**是池级(`resume_lock.rs` 的 `pool_ids("sovits")`
 *   含 `version`),但选择发生在**项目页的对话框**里,参数页上只有一行只读文字。
 *   这张表管的是**参数页**,所以它不在这里。⚠ 写出来是因为下一个人读 `pool_ids("sovits")`
 *   会以为提示已经覆盖了它 —— 没有,那一格只能靠眼睛(devServer 清单)。
 * * **rvc 的 `version`**(就在 sampleRate 上面一行、同样可编辑)—— `scope: "run"`:换它
 *   **不换池**,而是在**同一个池里**重跑一遍 ContentVec(`3_feature256` ↔ `3_feature768`)。
 *   `resumeCostlyTip` 那句「会换到另一份预处理产物上」对它是**假的** ⇒ 它需要的是另一句话,
 *   不是这一句。⛔ 别顺手把它加进来:那会让屏幕说一句不成立的话,而两个控件紧挨着。
 */
export const EDITABLE_POOL_FIELDS: Record<TrainingBackend, readonly string[]> = {
  rvc: ["sampleRate"],
  sovits: [],
  sovits_v2: [],
  sovits_diff: [],
  vocoder: [],
};

/**
 * 这个 backend 上,此刻要挂代价提示的那些字段。
 *
 * @param poolAtStake 这个槽里有没有一份**会被换掉**的预处理池。没有就一个都不挂:
 *   一个从没跑过预处理的槽,改什么都不需要付第二遍。
 *
 * ★S144 —— 返回的是**两个来源的并集**:⑴ 锁表里 tier=costly ∧ 作废池的那些(上面那条,
 * 与 backend 无关地按 rows 算)⑵ [`EDITABLE_POOL_FIELDS`] 里这个 backend 明写的那些。
 * ⛔ 并集只发生在**这一层**:`poolCostIdsIn` 的函数头规矩(「不认识 backend、不回头查任何
 * 别的表」)是阴性对照③唯一的入口,不许为了少写一行而破掉它。
 */
export function poolCostFieldIds(
  backend: TrainingBackend,
  poolAtStake: boolean,
): Set<string> {
  if (!poolAtStake) return new Set<string>();
  const out = poolCostIdsIn(resumeLockedFields(backend));
  for (const id of EDITABLE_POOL_FIELDS[backend]) out.add(id);
  return out;
}

/**
 * 这个槽里有没有一份**会被换掉**的预处理池。
 *
 * ★S142 笔 3 —— 它现在直接问盘:`WorkspaceInfo.has_preprocessing` 由 Rust 的
 * `training::slot_has_preprocessing` 算出,与**擦除同意闸**问的是同一个谓词。
 *
 * ⛔⛔ **在此之前这里是一个近似,而那个近似有两处是错的**(留在这里是因为它们解释了
 * 为什么这个字段值得跨一次 IPC):
 * ⒜ 旧式子的第一个合取项是 `!retrainIntent`,理由写着「再训一个会清空整槽,那时不存在
 *    换池这回事」——**S132 的 flip 之后那句话是假的**:旧 run 原样留着,而池是**槽级、
 *    内容寻址、跨 run 共享**的 ⇒ 重训路径上改 `augCopies` 照样铸新池、整份预处理重跑,
 *    而屏幕上一个字都没有。
 * ⒝ 旧式子的第二个合取项是「这个 **run** 的 manifest 说它跑过」,而 manifest 写在预处理
 *    **之前** ⇒ 一个刚起步就被杀掉的 run 会答「跑过」而盘上其实没有池;反过来,一个
 *    diff-first 的槽(进度记在 `diff_steps` 上、manifest 两键为空)已经有池却答「没跑过」。
 *
 * ⚠ **保留的取舍(不是疏漏)**:探针失败时 `info` 是 `null`(组件的 catch 把它吞成 null),
 * 这里答 `false` ⇒ 提示整体消失,静默,而代价是几小时。这一跳**对它结构上是瞎的**
 * (返回空集正是它的正确行为)⇒ 真正的判据必须在「探针失败要留下可见的 CODE」那一侧,
 * 记在队列 §E2E-M10-⒞。
 * ⚠ 反过来,Rust 那一侧是 **fail-closed** 的:`pools/` 读不动时答「有」,理由写在
 * `slot_has_preprocessing` 的函数头上(不确定时**说出代价**比**静默**安全)。
 */
export function poolAtStake(info: WorkspaceInfo | null | undefined): boolean {
  return !!info && info.has_preprocessing;
}
