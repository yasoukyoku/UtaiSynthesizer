/**
 * 「哪个 run 正在跑」与「此刻能不能动这一行」——§F2⒝-B2-⑤ / §E2E-M25 的决策层。
 *
 * ## ⛔⛔ 这里有【两个】谓词,而且它们的作用域是相反的。合成一个必然错一边
 *
 * * **`trainingIsLive`(全局、跨项目)** —— 后端的训练互斥是**进程级单槽**
 *   (`training/mod.rs` 的 `running.compare_exchange`,拒绝码 `TRAINING_ALREADY_RUNNING`)。
 *   别的项目在训练时,这个项目一样开不了训。⇒「继续训练 / 再训一个」跟它走,**不许按项目过滤**。
 * * **`liveRunIdFor`(本项目、本槽、逐行)** —— 用来**说出是哪一个**(徽章 / 置顶 / 折叠保底)。
 *   它必须按项目过滤,否则别的项目在跑时会给本页的行贴上徽章。
 *
 * 写成同一个谓词的两种错法都实测得出来:不过滤 ⇒ 给别人的 run 贴徽章;过滤了 ⇒ 别的项目在跑时
 * 本页一颗按钮都不禁,用户走完对话框再撞上一颗没有解释的禁用开始按钮。
 *
 * ## ⛔ 为什么「正在跑」不能写成 `state !== "idle"`
 *
 * 快照**不在训练结束时清**:唯一的清点是 `reset_display`,由用户点「清空结果」触发
 * (`training/mod.rs`)。所以一个 `completed` / `stopped` 的 run,它的 `workspace`/`run_id`
 * **仍然指着这一行**。照 `state !== "idle"` 写(仓里有两份现成抄本会诱你这么做:
 * `TrainingPage.tsx` 的 `runIsHere`、`store/training.ts` 的 `isLiveRun`),做出来的是
 * **跑完之后按钮永久禁死**的界面 —— 而任何只造 running 形状的夹具都杀不掉那个实现。
 *
 * ## ⛔ 为什么载体里那个空串不能当「没有」
 *
 * `TrainingSnapshot.run_id` 的 `""` 有两种含义:idle 时什么也没说;运行中它是**肯定事实**
 * ——「槽根就是这个 run」(未迁移槽,layout ≤ 2),而那正是 `RunDetail.id` 给那一行的值。
 * 所以这里一律先判 `state`,再谈 id;对外返回 `string | null`,`null` 才是「没有」。
 * (`rowIdentity.ts` 的头注早把这条缺席推断禁掉了,这只是同一条规矩换了个字段。)
 *
 * ## ⛔ 为什么要 `backendFamily`
 *
 * 浅扩散(`sovits_diff`)**跑在 sovits 槽的某个 run 目录里**,而 `sovits_diff` 不是一个 family
 * (`ProjectDetail` 的 `FAMILIES` 里没有它)。写成 `snap.backend === family` 的比较在**所有
 * 单 backend 的夹具上与正确写法同值**,只有浅扩散那一格分得开 —— 而浅扩散一跑就是几小时,
 * 那正是最需要这四件事的一屏。⇒ 语料自检显式要求见过这一格。
 *
 * ## ⛔ 「有没有长任务在跑」比「有没有训练在跑」宽,而删除/改名跟的是前者
 *
 * 后端 `ensure_safe_to_delete` / `ensure_idle_for_run_rename` 走的都是 `running_tasks_of`,
 * 它把 training / separation / render / audition / 全部 `active_tasks` 一起算,doc 明写
 * 「Deliberately FAIL-CLOSED and coarse」。⇒ 前端只禁「正在跑的那一行」会比后端**更宽**:
 * 用户去删同槽的兄弟 run、或在试听一段音频时删,照样被拒 ——「不让用户白点一次」这条承诺
 * 在最容易发生的那一格上直接落空。所以这一层**镜像后端的粗粒度**,per-run 身份只用来解释原因。
 */

import { backendFamily } from "../../store/training";

/** 决策要读的快照事实。⚠ 只列**用得到**的字段:组件那一侧必须用窄 selector 订阅,
 *  订整个 snapshot 会让每个训练 step 重渲染整片卡片墙(store 每一步都换一个新对象)。 */
export interface LiveFacts {
  state: string;
  project_id: string;
  backend: string;
  run_id: string;
}

/** 「正在跑」= starting | running。⛔ 不是 `!== "idle"`,理由见文件头。 */
export function isRunningState(state: string): boolean {
  return state === "starting" || state === "running";
}

/**
 * 此刻有没有**任何**训练在跑(跨项目、跨槽)—— 后端那道进程级互斥的前端镜像。
 *
 * `pendingStart` = store 的 `starting`(本机那次 `start_training` 的 invoke 在飞)。它不是
 * 冗余的:后端的 `running` 标志在 `compare_exchange` 那一刻就置位,而快照的 `state = "starting"`
 * 要到 450 行之后才写(中间还有真搬目录的 `migrate_one_slot`)⇒ 只看 `state` 的禁用在这段窗口里
 * 是**开着**的,而后端已经在拒了。
 */
export function trainingIsLive(snap: LiveFacts, pendingStart: boolean): boolean {
  return pendingStart || isRunningState(snap.state);
}

/**
 * 这个项目、这个 family 的**哪个 run** 正在跑。`null` = 没有(而 `""` 是一个合法答案:
 * 槽根就是那个 run)。
 *
 * ⚠ 不吃 `pendingStart`:那一档还没有 run 身份可言(目录都还没解析),而把它算进来会让
 * 徽章在一个**还不知道是谁**的行上先亮起来。「有训练在跑」由 `trainingIsLive` 回答。
 */
export function liveRunIdFor(snap: LiveFacts, projectId: string, family: string): string | null {
  if (!isRunningState(snap.state)) return null;
  if (snap.project_id !== projectId) return null;
  if (backendFamily(snap.backend) !== family) return null;
  return snap.run_id;
}

/** 一行 run 上,某颗按钮为什么不能点。空串 = 能点。 */
export type RowGateReason =
  | ""
  /** 项目被标记(`needsAttention`)—— 后端在任何东西开始之前就拒。 */
  | "flagged"
  /** 有训练在跑(任何项目、任何槽)—— 后端进程级单槽。 */
  | "training"
  /** 有长任务在跑(训练/分离/渲染/试听/…)—— 后端删除与改名的前置。 */
  | "tasks"
  /** 本页自己有一次 invoke 在飞(防重复点击)。 */
  | "inflight";

export interface RowGate {
  disabled: boolean;
  reason: RowGateReason;
}

export interface RunRowInput {
  /** 项目被标记。 */
  blocked: boolean;
  /** 本页有一次 invoke 在飞。 */
  inFlight: boolean;
  /** 有训练在跑(跨项目)。 */
  trainingLive: boolean;
  /** 有**任何**长任务在跑 —— 后端 `running_tasks_of` 的镜像。⚠ 训练也算一种长任务,所以
   *  正常情况下 `trainingLive ⇒ anyTaskLive`;两者分开是因为它们守的按钮不同。 */
  anyTaskLive: boolean;
  /** 这一行有没有 run id(`""` = 未迁移槽的伪造行 ⇒ 删除按钮本来就不画)。 */
  hasRunId: boolean;
}

export interface RunRowGates {
  /** 「继续训练」 */
  cont: RowGate;
  /** 「再训一个」 */
  retrain: RowGate;
  /** 「删除此 run」 */
  del: RowGate;
  /** 「✎ 改名」 */
  rename: RowGate;
}

const gate = (reason: RowGateReason): RowGate => ({ disabled: reason !== "", reason });

/**
 * 禁用原因 → i18n 键。`null` = 没被禁,没什么好说的。
 *
 * ⛔ 映射放在这里而不是组件里,是为了让接线闸能**由这张表推出期望**(而不是手抄一份字面串:
 * 同一个串在两边各写一遍,打错两遍会一起绿)。三语的存在由 i18n parity 闸钉,而
 * 「组件真的把它显示出来了」由 `rowIdentityWiring` 那道源码闸钉 —— 两者缺一都不够:
 * `training.*` 今天**没有死键闭合**(那道闸硬编只扫 `backend.*`),所以一个没人用的键会以
 * 「三语齐全」的姿态活下去。
 */
export function gateReasonKey(reason: RowGateReason): string | null {
  switch (reason) {
    case "":
      return null;
    case "flagged":
      return "training.gateFlagged";
    case "training":
      return "training.gateTraining";
    case "tasks":
      return "training.gateTasks";
    case "inflight":
      return "training.gateInflight";
  }
}

/** 上面那张表的全部键 —— 判据与接线闸都从它推期望,别再手抄。 */
export const GATE_REASON_KEYS: readonly string[] = (
  ["flagged", "training", "tasks", "inflight"] as const
).map((r) => gateReasonKey(r)!);

/**
 * 每颗按钮该不该禁、以及**为什么** —— 原因是要上屏的(tooltip),不是给日志看的。
 *
 * ⛔ 四颗按钮跟的**不是同一个谓词**,这正是这个函数存在的理由:
 * * 继续 / 再训 —— 「有训练在跑」(后端 `TRAINING_ALREADY_RUNNING`,跨项目);
 * * 删除 / 改名 —— 「有任何长任务在跑」(后端 `DELETE_WHILE_BUSY` / `RENAME_WHILE_BUSY`,
 *   经同一个 `running_tasks_of`)。
 *
 * ⚠ 把它们塌成一个 `disabled: boolean` 会让判据分不开这两条规则 —— 而分不开的那一刻,
 * 极性写反(禁错一半)不会被任何后端判据抓到:用户要么能点一颗必被拒的按钮,要么被永久锁死。
 *
 * ⚠ 顺序是承重的:先报**最解释得通**的那一条。项目被标记时说「有训练在跑」是一句会把人引去
 * 等训练结束的假话。
 */
export function runRowActions(input: RunRowInput): RunRowGates {
  const { blocked, inFlight, trainingLive, anyTaskLive, hasRunId } = input;
  const start: RowGateReason = blocked ? "flagged" : trainingLive ? "training" : "";
  const mutate: RowGateReason = blocked
    ? "flagged"
    : anyTaskLive
      ? "tasks"
      : inFlight
        ? "inflight"
        : "";
  return {
    cont: gate(start),
    retrain: gate(start),
    // 未迁移槽那一行没有 run id,删除按钮按设计不画;这里如实答「不可用」,让调用点
    // 不必再自己判一次(⛔ 但**不画**仍然由调用点决定 —— 这一层不管画不画)。
    del: gate(hasRunId ? mutate : "flagged"),
    rename: gate(mutate),
  };
}
