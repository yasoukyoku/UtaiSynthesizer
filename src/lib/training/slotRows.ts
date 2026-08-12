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
