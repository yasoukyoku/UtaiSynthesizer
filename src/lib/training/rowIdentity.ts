/**
 * 存档列表里**一行**的产物身份 ——「这一行叫什么名字」与「它的工作区在哪」的**唯一**决策。
 *
 * ## 为什么名字与工作区必须由同一个函数、在同一次解析里回答(§F2⒝ 批 2 ④b)
 *
 * 一个槽从此可以有多个 run,而这两个读数以前来自**两个不同的取值口径**:
 * 名字来自槽级的 `slotCtx.modelName`,工作区来自实时 run 的 `snapshot.workspace`。
 * 只要它们分头回答,就会出现这一种失败:
 *
 * > 这一行标着 **run B 的名字**,试听放出来的却是 **run A 的缓存**。
 *
 * 而它是**静默**的:试听缓存的命中判据只看 `<dir>/model.json` 在不在
 * (`audition.rs:155-157`),连那道结构闸也放行 —— `workspace_is_a_slot` 问的是
 * 「目录里有没有 `runs/`」,而一个**兄弟 run** 的目录里当然没有。
 * 症状是「这个存档听起来像那个存档」,正是耳朵**判不出来**的那一类。
 *
 * ## 实时 run 的读数什么时候可信
 *
 * ⛔ 判据必须是**肯定事实**:实时 run 的工作区路径与这一行解析出来的工作区**相等**。
 * 不能写成「这一行没有 run id ⇒ 就当它是实时那个」——那是缺席推断,与「有人忘了把 run id
 * 穿下来」长得一模一样(本仓 `utai_train/pool.py:30-32` 明写禁止的形状)。
 * 唯一的例外是这一行**结构上就没有 run id**(实时候选行:它还没被盘上扫描看见),
 * 那时「本段显示的就是它」是由构造成立的,不是推断出来的。
 *
 * ⚠ 今天(每槽恒一个 run)两条路给出同一个答案 —— 所以判据必须**造出两个 run** 才有分辨力。
 */

/** Rust `get_slot_export_context` 对【这一行的 run】(或退回槽)的回答;`null` = 没拿到。 */
export interface RowExportContext {
  modelName: string;
  workspace: string;
  indexPath: string | null;
}

/** 本段正在显示的那次实时运行;`null` = 独立的存档中心视图(没有实时身份可用)。 */
export interface LiveRunIdentity {
  modelName: string;
  workspace: string;
  /** 这次运行自己报告的索引产物 —— 比探测出来的更权威,但**只对它自己那一行**成立。 */
  summaryIndex?: string | null;
}

export interface ResolveRowIdentityInput {
  /** 这一行指名的 run。`null`/`undefined` = 这一行没有 run id(实时候选行)。 */
  runId?: string | null;
  ctx: RowExportContext | null;
  live: LiveRunIdentity | null;
  /** 兜底名字(参数表单里的「本次训练名」)。 */
  fallbackName: string;
}

export interface RowIdentity {
  name: string;
  workspace: string;
  indexPath: string | null | undefined;
  summaryIndex: string | undefined;
  /** 工作区是谁答的。⛔ 判据需要它:`live` 与 `run` 在 N=1 时**给出同一个字符串**,
   *  没有这个字段就分不开「用了实时读数」和「用了这一行自己那个 run」。 */
  source: "live" | "run" | "none";
}

export function resolveRowIdentity(input: ResolveRowIdentityInput): RowIdentity {
  const { runId, ctx, live, fallbackName } = input;
  // 见文件头:肯定事实(路径相等),或者「这一行结构上就没有 run id」这个由构造成立的例外。
  const liveIsThisRow =
    !!live && (runId == null || (!!live.workspace && live.workspace === ctx?.workspace));
  const name = (liveIsThisRow ? live!.modelName : "") || ctx?.modelName || fallbackName;
  const workspace = (liveIsThisRow ? live!.workspace : "") || ctx?.workspace || "";
  return {
    name,
    workspace,
    indexPath: ctx?.indexPath,
    summaryIndex: liveIsThisRow ? (live!.summaryIndex ?? undefined) : undefined,
    source: liveIsThisRow && live!.workspace ? "live" : ctx?.workspace ? "run" : "none",
  };
}
