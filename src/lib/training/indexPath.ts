/**
 * 存档导入时「检索/聚类索引在哪」的**唯一**决策 —— 抽成纯函数,因为它的失败是**静默**的。
 *
 * ## 为什么它必须能被测试驱动(S126 §E2E-M1)
 *
 * 这条链的每一环单看都「没报错」,合起来是一次无声的降级:
 *  1. `get_slot_export_context` 失败被前端 `catch` 掉 ⇒ 工作区路径塌成 `""`;
 *  2. 用 `""` 拼出的候选路径当然不存在 ⇒ 返回「没有索引」;
 *  3. `import_model` 收到 `index_file: None` ⇒ **整个 `WARN_INDEX_MISSING` 分支被跳过**
 *     (那个分支只在「用户明确指名了一个文件、而它不在」时才说话);
 *  4. 它转而在**存档文件旁边**(`<run>/weights/`)自动探测 `.npy` / `.index` / `added_*.index`,
 *     而 RVC 的索引在 `<run>/total_fea.npy` ⇒ 三条全落空,`warnings` 里**一个字都不 push**;
 *  5. 于是模型装上了、没有检索矩阵、**无警告无 CODE 无 toast**,唯一症状是音色相似度下降。
 *
 * ⛔ 第 5 步**听不出来**(没有 A/B 判不出,判出来也归因不到索引上)。所以这件事**不能**留给
 * 实机窗口去「看一眼」——它在盘上不过是一个文件在不在,必须由判据来答。
 *
 * ## 「不知道」不是「没有」
 *
 * 关键的修正在这里:工作区未知时,旧代码与「确实没有索引」给出**同一个答案**(`undefined`)。
 * 它俩的正确处置完全不同 —— 前者必须让用户知道,后者是正常的。所以这个函数返回的是一个
 * **可区分的结果**,而不是一个可空字符串。
 *
 * ⚠ 顺带堵掉一处真实的荒唐:`""` 拼出来的 `\total_fea.npy` 在 Windows 上是**当前盘符的根**,
 * 探测它既无意义又有极小概率命中别人的文件。工作区未知时**不探测**。
 */

/** 声码器没有索引/聚类伴生物;探测只会捞到别的后端的遗留(红队 A16)。 */
const BACKENDS_WITHOUT_INDEX = ["vocoder"];

export type IndexResolution =
  /** 明确知道用哪个:Rust 的槽/run 上下文给的,或本次实时 run 的 summary。 */
  | { kind: "explicit"; path: string }
  /** 工作区里探到的。 */
  | { kind: "probed"; path: string }
  /** 这个后端/这份产物**本来就没有**索引 —— 正常状态,不需要提示。 */
  | { kind: "none" }
  /** ⛔ **我们不知道该去哪找**(工作区上下文没拿到)。装出来的模型可能缺检索矩阵,
   *  而缺了是**听不出来**的 ⇒ 调用方**必须**把它变成用户看得见的一句话。 */
  | { kind: "unknownWorkspace" };

export interface ResolveIndexInput {
  /** 存档段当前作用的后端(`rvc` / `sovits` / `sovits_v2` / `vocoder` / `sovits_diff`)。 */
  backend: string;
  /** 这一行所属 run 的目录。`""` = 上下文没拿到 —— **不是**「根目录」。 */
  workspace: string;
  /** 实时 run 的 `summary.index`,**仅当本段就是那个 run** 时可信。 */
  summaryIndex?: string | null;
  /** Rust 侧 `get_slot_export_context` 的答案(它自己 `is_file()` 探过)。 */
  ctxIndexPath?: string | null;
  /** 注入的存在性探针(生产用 tauri plugin-fs 的 `exists`)。 */
  exists: (path: string) => Promise<boolean>;
}

/** 候选路径:**故意**用反斜杠拼,和生产代码原样一致(Windows 专有,别在这里「顺手规范化」——
 *  规范化会让判据与真实拼接产生分叉,而分叉正是这条链出问题的地方)。 */
function candidates(backend: string, workspace: string): string[] {
  if (backend === "rvc") return [`${workspace}\\total_fea.npy`];
  return [`${workspace}\\cluster\\kmeans_10000.pt`, `${workspace}\\cluster\\0.index_vectors.npy`];
}

export async function resolveArchiveIndex(input: ResolveIndexInput): Promise<IndexResolution> {
  const { backend, workspace, summaryIndex, ctxIndexPath, exists } = input;
  // 明确指名的优先,而且**不再探测** —— 指名了就是指名了,再去猜是另一种静默错配。
  const named = summaryIndex ?? ctxIndexPath ?? undefined;
  if (named) return { kind: "explicit", path: named };
  if (BACKENDS_WITHOUT_INDEX.includes(backend)) return { kind: "none" };
  // ⛔ 工作区未知 ⇒ 说「不知道」,不说「没有」。
  if (!workspace) return { kind: "unknownWorkspace" };
  for (const cand of candidates(backend, workspace)) {
    if (await exists(cand)) return { kind: "probed", path: cand };
  }
  return { kind: "none" };
}

/** 传给 `import_model` 的 `indexPath`。**只有真的知道一个路径时才给** ——
 *  给一个不存在的路径会让 Rust 认为「用户选的,别再自动找」,那比不给更糟。 */
export function indexPathArg(r: IndexResolution): string | undefined {
  return r.kind === "explicit" || r.kind === "probed" ? r.path : undefined;
}

/** 需要让用户看见的 CODE(走既有的 `backendError` 漏斗),没有就是 null。 */
export function indexWarningCode(r: IndexResolution): string | null {
  return r.kind === "unknownWorkspace" ? "WARN_INDEX_CONTEXT_UNKNOWN" : null;
}
