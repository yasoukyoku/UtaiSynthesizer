/**
 * 项目详情页上**一个槽画成什么样**的全部决策 —— 画哪几行 run、「尚未开始」出不出现、
 * 浅扩散挂到哪个 run、以及「预处理 N 份 · X」那一行。
 *
 * ## 为什么要把它们从组件体里搬出来(§E2E-M3/M4/M5,S141)
 *
 * 这五个决策此前是 `ProjectDetail.tsx` 里 `FAMILIES.map` **内部**的内联表达式,没有导出 ——
 * vitest **结构上**够不着,所以它们只有源码在守,任何变异都会存活。而它们各自守着的东西:
 *
 * * `visibleRuns`：④e 刚修的那条 ——「铸了但没练成」的 run 必须画得出来。它此前被
 *   `runs.filter(startedRun)` **结构性地藏起来**,而那种 run 是一个**满强度的冻结源**
 *   (manifest 里有 `speakers` ⇒ 锁住整个项目的数据集),卡片却写着「尚未开始」:
 *   同一屏两句话互相打脸,而新文案里「去逐个 run 删掉」的指引对它执行不下去。
 * * `pickDiffHost`：它决定浅扩散**在哪个 run 目录里训练**,并据此显示版本与步数。
 * * `prepPoolLine`：屏幕上唯一说「预处理占了多少盘」的地方,而那是用户用来决定删什么的读数。
 *
 * ⛔ **不要用 DOM 快照去测这些**：快照对样式敏感、对逻辑不敏感,而这里每一条都是逻辑。
 *
 * ## 这一层【不做】决定的事
 *
 * 本模块是 S141 的**纯提取**:每个函数体与它在组件里的原样逐字等价,单测里另存了一份
 * 搬家前的表达式做差分对拍(`slotRows.test.ts` 的 legacy 组)。唯一的新增是几个
 * **只被判据读**的字段(`source` / `withMainProgress`),它们不改变任何 UI 行为 ——
 * 改 UI 是 B2-⑤ 的事,不是这一笔的。
 */

import type { RunDetail, SlotDetail } from "../../store/training";

/** 这个 run 练出过东西吗 —— 有断点,或者已经有主模型。 */
export function startedRun(r: RunDetail): boolean {
  return r.hasResumePoint || r.info.has_main_progress;
}

/**
 * 画哪几行。
 *
 * ⛔ 真正的 run(`id` 非空)**一律画出来**,哪怕它两个条件都假 —— 那正是「铸了但没练成」
 * 的形状(`run_manifest.json` 在 spawn 之前就写好,随后的切片/f0/特征要跑几小时)。
 * 只有 `id === ""` 那条**伪造行**(后端在**零 run** 时补的,寻址槽根)仍然按
 * 「练出过东西」过滤 —— 那种槽本来就没有 run 可挑。
 */
export function visibleRuns(runs: readonly RunDetail[]): RunDetail[] {
  return runs.filter((r) => r.id !== "" || startedRun(r));
}

/**
 * 「尚未开始」与槽级「开始」按钮跟着**看得见的行**走。
 *
 * ⚠ 它必须由 `visibleRuns` 定义,不是由 `runs`：两者一分家就会出现「有一行 run」同时
 * 「尚未开始」的自相矛盾,而那是这条决策存在的全部理由。
 */
export function slotStarted(runs: readonly RunDetail[]): boolean {
  return visibleRuns(runs).length > 0;
}

/** 浅扩散挂在哪个 run 上,以及**这个答案是怎么来的**。 */
export interface DiffHostChoice {
  host: RunDetail | undefined;
  /**
   * 这个选择是怎么来的。
   *
   * ⛔ 判据**需要**它:一个槽只有一个 run 且它带主模型时,`find(...)` 与 `runs[0]`
   * 返回的是**同一个对象** ⇒ 把 `find` 换成 `[0]` 的变异在 `host` 上完全不可见。
   * 这是本仓 S125/S126 那族「装饰性判据」的原样形状,所以答案自己要带上出处。
   */
  source: "main-progress" | "fallback-first" | "none";
  /**
   * 有几个 run 带主模型。
   *
   * ⛔★★S133 —— 这里此前有一句注释说「这条 `find` 是一个肯定事实,不是『挑第一个』」,
   * 而 ④e 的 flip 让「再训一个」真的铸第二个 run 之后**那句话就是假的**:两个 run 都练出
   * 主模型时,`find` 返回的是 `list_runs` 的排序(run id 的**字典序**)里的第一个 ——
   * 一个与「用户想挂到哪个」毫无关系的答案。
   * ⇒ 真正的修法是给这张卡一个 run 选择器,那是 **B2-⑤**(队列里已排)。在那之前
   * **行为保持现状**,但这个数让歧义变成一个可断言的事实,而不是一句注释。
   */
  withMainProgress: number;
  /**
   * 这个项目的 sovits 槽已经承诺给哪个 ContentVec 空间。浅扩散训练在那个槽的缓存特征上,
   * 所以 manifest 里的版本把它 **PIN** 住了。
   */
  pinnedVersion: "4.1" | "4.0" | undefined;
  /** 最大浅扩散存档步数;0 = 没有 / 只有底模。 */
  steps: number;
}

export function pickDiffHost(slot: SlotDetail | undefined): DiffHostChoice {
  const runs = slot?.runs ?? [];
  const withMain = runs.filter((r) => r.info.has_main_progress);
  const host = withMain[0] ?? runs[0];
  const version = host?.info.version;
  return {
    host,
    source: withMain.length > 0 ? "main-progress" : host ? "fallback-first" : "none",
    withMainProgress: withMain.length,
    pinnedVersion: version === "4.1" || version === "4.0" ? version : undefined,
    steps: host?.info.diff_steps ?? 0,
  };
}

/** 「预处理 N 份 · X」那一行:出不出现,以及出现时的两个数。 */
export interface PrepPoolLine {
  show: boolean;
  count: number;
  bytes: number;
}

/**
 * ⛔ 它是**槽级**的:池由这个槽的每个 run 共享,而那正是 layout 2 的全部意义。
 * `show` 与 `count` 必须来自**同一个**读数 —— 分头取的话,「有池但写着 0 份」与
 * 「没池却画出这一行」会各自成为可能。
 */
export function prepPoolLine(slot: SlotDetail | undefined): PrepPoolLine {
  const count = slot?.prepPoolCount ?? 0;
  return { show: count > 0, count, bytes: slot?.prepPoolBytes ?? 0 };
}

/**
 * ★★§F2⒝-B2-⑤ / §E2E-M25 ⑷ —— run 行的**展示序**:正在跑的置顶,其余按名字。
 *
 * ## 为什么要排,以及为什么后端那个顺序不值得保
 *
 * 后端 `trun::list_runs` 以 `a.id.cmp(&b.id)` 收尾,而 run id 是 **sha256 派生**的
 * (`minted_run_id` 的 doc 亲口写着「Not sortable, deliberately: sha256 destroys the ordering
 * of its seed」)⇒ 今天用户看到的行序**既不是创建序也不是任何可解释的序**,它是随机的。
 * 折叠(`foldRunRows`)取的又是**头部** `limit` 条 ⇒ 被收起来的是一个**任意**子集。
 * ⇒ ⑷ 是把一个随机序换成一个可解释序,不是改一个既有约定。
 *
 * ⚠ **按更新日期排不做**(用户 2026-08-13 拍板):`RunDetail` 两侧都没有时间字段,加它要动
 * IPC 与 Rust 的 run 列举,而收益是纯观感 ⇒ 等有人真要的那天再加载体。
 *
 * ## ⛔ 三条硬约束(每一条都有判据,别当成风格)
 *
 * 1. **不许把排序塞进 `visibleRuns`**。那个函数有一条 S141 留下的**差分对拍**
 *    (2000 次随机槽 vs 搬家前原样抄下来的 `legacyVisibleRuns`),它证明的是「纯提取、行为
 *    逐位不变」。排序是**新行为** ⇒ 塞进去会打红它,而「顺手改 legacy 那一半」就把那条证据
 *    变成了自证。⇒ 新开一个导出函数,`visibleRuns` 与 legacy 保持逐位相等。
 * 2. **必须返回新数组**。`slot.runs` 同时被 `pickDiffHost`(它读 `runs[0]` 当回落)与 store
 *    里那一份(`setProjectInfo(d)`)共用 ⇒ 一次原地 `sort()` 会**静默改掉浅扩散训练进哪个 run
 *    目录**、用谁的名字、还原谁的表单,而三条后果都要几小时之后才看得出来。
 *    ⚠ 判据必须断言**入参的顺序**逐位不变,不是 `length` —— 原地排序不改长度。
 * 3. **`pickDiffHost` 不跟排序走**。它吃的是组件顶层那个 `SlotDetail`(在 `FAMILIES.map` 之外),
 *    与这里排的是两条不相交的链。所以「排序前后 `pickDiffHost` 给同一个答案」写成单测是一句
 *    **恒真的话**;真正要钉的是**它的实参**,那是一道源码闸(见 `rowIdentityWiring`)。
 *
 * ## 排序规则(逐条都有一格判据)
 *
 * ⑴ 正在跑的那一行置顶(`liveRunId` 逐字节相等;⛔ `""` 是合法 id,不许用真值判断);
 * ⑵ 起过名的排在没起名的前面 —— 没起名 = 这个 run 还没练完过,它不是用户在找的东西;
 * ⑶ 按名字(`numeric` ⇒ `run2` 在 `run10` 前面,那是人读数字的方式);
 * ⑷ 同名时按 id —— **同名是可达状态**(改名那条路今天没有同名闸),没有这一条,两条同名 run
 *    的先后就靠 `Array#sort` 的稳定性,而那是一条没人声明过的巧合。
 */
export function sortRunRows(
  rows: readonly RunDetail[],
  liveRunId: string | null,
): RunDetail[] {
  const named = (r: RunDetail) => (r.modelName ?? "").trim();
  return [...rows].sort((a, b) => {
    const aLive = liveRunId !== null && a.id === liveRunId;
    const bLive = liveRunId !== null && b.id === liveRunId;
    if (aLive !== bLive) return aLive ? -1 : 1;
    const an = named(a);
    const bn = named(b);
    if (!an !== !bn) return an ? -1 : 1;
    const byName = an.localeCompare(bn, undefined, { numeric: true, sensitivity: "base" });
    if (byName !== 0) return byName;
    return a.id.localeCompare(b.id);
  });
}

/** 超过这么多条 run 才开始折叠。 */
export const RUN_ROWS_BEFORE_FOLD = 2;

export interface RunRowFold {
  /** 这一次要画出来的行。 */
  rows: RunDetail[];
  /** 被收起来的条数;0 = 没有折叠,连那行「还有 N 个」都不该出现。 */
  hidden: number;
}

/**
 * ★S141(用户实机提的)—— run 多了之后那一长条怎么收。
 *
 * 每个 run 行是「名字+改名 / 事实 / 三颗按钮」三段,五个 run 就是一根很长的柱子;而
 * `.tproj-slots` 是两列 grid,一个高卡片会把**同一行那个空槽**也拉成同样高(那半边已经在
 * CSS 里治了:`align-items: start`)。
 *
 * ⛔ 用户的原话是「现在这样直接显示确实很清楚」⇒ **少量 run 时必须逐像素保持今天的样子**,
 * 只有真的变多才收。所以阈值判据是 `rows.length > limit`,不是「永远只画一条 + 管理入口」。
 *
 * ## ★★§E2E-M25 ⑴ —— 「折叠不许藏起正在跑的那条」由**复合**保证,这里一个字都不用改
 *
 * ⚠ 原文这里写着「这里不认识哪个 run 正在跑 …… 所以正在训练的那个**可能**落在折叠里。要改成
 * 『正在跑的永远展开』得先把实时 run 的身份穿到这一层(那是 B2-⑤ 的面)」。**身份穿过来了**
 * (`TrainingSnapshot.run_id` → `liveRunIdFor`),而那句话的**结论**不再成立,理由与当初设想的
 * 不一样:调用点现在喂进来的是 [`sortRunRows`] 的输出,**正在跑的那一行恒在 index 0**,
 * 而这里取的是**头部** `limit` 条(`limit >= 1`)⇒ 它必然被画出来。
 *
 * ⛔ **所以这里没有 `keepId` 参数,那是有意的。** 加一个「保底把它捞回来」的分支,在今天的接线上
 * **没有任何输入走得到**:它会是一条读起来像守卫、而任何变异都杀不掉的死分支(本轮在
 * `trun::run_id_of` 上刚删过一条同形的)。⇒ 保证写成**复合判据**
 * (`sortRunRows → foldRunRows` 一口气驱动,输入是**未排序**且 live 落在 `limit` 之后),
 * 顺序由 `rowIdentityWiring` 那道「排序排在折叠之前」的源码闸守。
 * ⚠ 那条复合判据是这条性质**唯一**的看守:单独喂 `foldRunRows` 一份**已排序**的行去断言
 * 「正在跑的还在」是一句恒真的话(S128 的 L9 同族),它绿的理由与折叠无关。
 */
export function foldRunRows(
  rows: readonly RunDetail[],
  expanded: boolean,
  limit: number = RUN_ROWS_BEFORE_FOLD,
): RunRowFold {
  if (expanded || rows.length <= limit) {
    return { rows: [...rows], hidden: 0 };
  }
  return { rows: rows.slice(0, limit), hidden: rows.length - limit };
}

/** 「再训一个」那个新名字有什么毛病;`null` = 可以用。 */
export type NewRunNameProblem = "empty" | "taken" | null;

/**
 * ★§E2E-M24 —— 「再训一个」的新名字不许与**同槽已有 run** 重名。
 *
 * ⛔ 为什么重名是灾难而不是不方便:名字是**产物前缀**(`weights/<slug>*`、`audition/<slug>_*`)。
 * 两个 run 同名 ⇒ 同 slug ⇒ 存档页两行同名,而 `plan_cleanup` 的 `installed_stem` 按 file_stem
 * 判「这份快照还装着」—— 于是它会把**另一个** run 的快照也判成 StillInstalled 而永久保留。
 *
 * ⛔ 比对的是 **trim 之后**的串,两边都是:落库时写的就是 `newName.trim()`,所以一个只差首尾
 * 空格的名字**不是**一个新名字 —— 只 trim 一边的话,` 歌姫 ` 会顺利通过,然后落成 `歌姫`。
 *
 * ⚠ 没有名字的 run(铸了但没练成)不占名额:它的 `modelName` 是 `""`,而空名字走的是
 * `"empty"` 那一档。这条过滤是防御性的,不是由某条判据买回来的 —— 说清楚免得下一个人以为它有。
 */
export function newRunNameProblem(
  raw: string,
  runs: readonly RunDetail[],
  /** ★S143 §E2E-M25 笔 5 —— 改名时**排除这一行自己**(不然改成自己已有的名字会被自己挡住)。
   *  ⛔ 类型是 `string | null` 而不是 `string`:`""` 是一个**合法 run id**(未迁移槽的槽根就是
   *  那个 run),用 `""` 当「不排除」会让那种槽的唯一一行永远被跳过。`null` = 不排除。 */
  exceptId: string | null = null,
): NewRunNameProblem {
  const name = raw.trim();
  if (!name) return "empty";
  const taken = new Set(
    runs
      .filter((r) => exceptId === null || r.id !== exceptId)
      .map((r) => r.modelName?.trim())
      .filter((n): n is string => !!n),
  );
  return taken.has(name) ? "taken" : null;
}
