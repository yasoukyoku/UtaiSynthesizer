/**
 * 「把表单放回这个槽已经在的地方」—— 纯函数版。
 *
 * ## 为什么它必须是一等公民而不是一个组件内的闭包
 *
 * `resume_lock.rs` 的模块头把**四处**列为必须互相同意的地方:`try_start` 的守卫、运行段的
 * 开始前对话框、参数页的只读渲染,以及**项目页的表单还原** —— 也就是这里。前三处都有闸
 * (`resume_lock` 的表驱动单测 + `resumeLockParity`),这一处此前既没有闸,也没有办法有:
 * 它是 `ProjectDetail` 里的一个闭包,vitest 不做组件测试,所以**没有任何东西驱动得动它**。
 *
 * 而它漏掉的恰好是最贵的一档。锁表分两档:`locked`(续训会被拒)与 `costly`(允许改,但会
 * 给数据集重新指纹化 ⇒ 下一次运行重跑预处理)。旧版这里**只还原 locked 那一档**,理由是
 * 「不还原就会被拒」—— 于是 costly 那一档从来没有被还原过,而 costly 的定义正是
 * 「**它决定用哪个预处理池**」。后果不是一次拒绝(那是响亮的),是:
 *
 * * `sovitsLoudnorm` 折进 `extract_cache_fp_text`(`utai_train/sovits/pipeline.py`)⇒ 表单送回
 *   默认的 `false`,`open_pool` 按内容选不到那个池,**铸一个新池并重跑整份预处理**,痕迹只有
 *   一行 `logger.info`;顺带 `manifest["loudnorm"]` 被这次请求覆盖,连**记录**都没了。
 * * `augCopies` 送回 0 ⇒ `augment_slices` 的 stale 清扫按 `idx > copies` 把**全部** `_aug*` 切片
 *   连同 `.soft.pt` / `.f0.npy` / `.spec.pt` / `.vol.npy` 就地删掉(它的 docstring 明写
 *   「knob-down 必须完全去增强」),续训在缩水后的数据集上接着练。
 *
 * store 没有持久化(全仓无 `persist(`),所以**冷启动**之后每一个 costly 字段都是默认值;
 * 同一会话里 `enterProject` 也只清 `modelName`,其余原样带进下一个项目。
 *
 * ## ★ 它同时是 §F2⒝ ④d 的结构前提
 *
 * ④d 要把 `|aug=<n>` / `|sr=<x>` 写进**池身份**。池身份是「盘上那堆产物属于谁」的定义,
 * 而定义它的输入如果在每次「继续训练」时都被表单忘掉,那这个身份就不是身份。
 * 迁移器给存量池补的 `<n>` 只有在「下一次请求会送回同一个值」时才有意义 —— 这个文件就是
 * 那句话为真的地方。
 *
 * ## 编辑规矩
 *
 * 加一行 costly 到 `resume_lock.rs` ⇒ 必须在 [`POOL_FORM_FIELDS`] 里给它一个表单字段,
 * 或者在 [`NO_FORM_CONTROL`] 里说明它为什么没有。`formForSlot.test.ts` 两头对拍,漏一个就红。
 */

import type { TrainingBackend, TrainingFormConfig, WorkspaceInfo } from "../../store/training";

/**
 * 锁表的 `costly` id → 参数页上承载它的那个表单字段,按 backend。
 *
 * ⛔ 这张表是**声明**,不是文档:`formForSlot.test.ts` 用 `lockedFieldIds(b, "costly")` 逐个
 * backend 对拍它的键集,再逐个字段驱动 [`formForSlot`] 断言它真的被还原。所以它不会悄悄过期。
 *
 * ⚠ `sovits_diff` 的 `augCopies` 只在**没有宿主主模型**时是这个 run 自己的选择(diff-first);
 * 有宿主时 Rust 的 `eff_aug_copies` 从 manifest 继承,参数页连输入框都不渲染。见 [`formForSlot`]。
 */
export const POOL_FORM_FIELDS: Record<TrainingBackend, Record<string, keyof TrainingFormConfig>> = {
  rvc: { augCopies: "augCopies" },
  sovits: { loudnorm: "sovitsLoudnorm", augCopies: "sovitsAugCopies" },
  // v2 与 sovits **共用** sovits* 那组表单字段(两张卡不同时存在),而它同样送 loudnorm 且
  // 同样折进 fp_text ⇒ 它的池身份也由这两个值决定。
  sovits_v2: { loudnorm: "sovitsLoudnorm", augCopies: "sovitsAugCopies" },
  sovits_diff: { augCopies: "diffAugCopies" },
  vocoder: { augCopies: "vocAugCopies" },
};

/**
 * 锁表里**故意**没有表单字段的 costly id。
 *
 * `dataset` = 数据集本身,它归数据页管(增删音频),参数页上没有、也不该有一个控件代表它。
 * 把它写在这里而不是让 [`POOL_FORM_FIELDS`] 静默漏掉,是为了让「漏了」与「不该有」分得开。
 */
export const NO_FORM_CONTROL: ReadonlySet<string> = new Set(["dataset"]);

/**
 * 这个槽的 manifest 里有没有「上一次是怎么练的」。
 *
 * 与 `resumeWouldBeGuarded` 同源(`version` / `sample_rate` 是一起写的),理由也一样:manifest
 * 在 worker 开始预处理**之前**就写好了,所以一个练到一半被停掉、还没有 `G_*.pth` 的槽同样
 * 有话要说。用 `has_main_progress` 当判据会在那个窗口里把表单打回默认值。
 *
 * 反过来,槽**没跑过**时必须一个字段都不还原:那时 `aug_copies` 是 0、`loudnorm` 是 null,
 * 「还原」它们等于把用户刚在参数页填好的值清掉。
 *
 * ★S142 笔 1 一度把它导出:那时 `TrainingPage.tsx` 的参数页把同一条规则**又手抄了一份**
 * (第三份 —— `resumeWouldBeGuarded` 的非 diff 臂是第二份),而三份里只有这一份带着上面那段
 * 理由,所以让代价提示那一层转调它。
 * ★S142 笔 3 **收回了那个 export**:代价提示改成直接问盘(`WorkspaceInfo.has_preprocessing`,
 * 由 Rust 的 `slot_has_preprocessing` 算),第三份手抄件**整个不存在了** ⇒ 这里不再有仓内
 * 第二个消费者,而一个没有消费者的公开 API 就是下一次漂移的入口。
 */
function hasManifest(info: WorkspaceInfo | null | undefined): info is WorkspaceInfo {
  return !!info && info.exists && (info.version !== "" || info.sample_rate !== "");
}

/**
 * 把表单放回这个槽(这个 run)已经在的地方。
 *
 * 还原两档:
 * * **locked** —— 不还原就会在开始训练时被 Rust 拒(`RESUME_PARAMS_MISMATCH` /
 *   `RESUME_VOL_EMBEDDING_MISMATCH` / `RESUME_KSTEP_MISMATCH`)。默认值在这里**不是无害的**:
 *   `sampleRate` 默认 48k 而槽可能是 40k,`sovitsVersion` 默认 4.1 而槽可能是 4.0。
 * * **costly** —— 不还原不会被拒,而是**静默换池重跑**(见文件头)。
 *
 * `pin` 是「再训一个」传的版本选择:它会清空这个槽,所以版本由用户当场选定而不是沿用 manifest,
 * 而 `volEmbedding` 这种「跟着版本走」的值不再回填。costly 那一档在重训路径上**照样还原** ——
 * 那时它不是约束而是**默认配方**:重训最常见的意图是「同样的配方再练一遍」,而把它打回
 * 出厂默认(响度归一化关、增强 0)从来不是用户的意思。
 *
 * @param info 这个 **run** 的 `WorkspaceInfo`(`slot_info(.., run_id)`);`null`/未跑过 = 不还原。
 */
export function formForSlot(
  backend: TrainingBackend,
  info: WorkspaceInfo | null | undefined,
  pin?: "4.1" | "4.0",
): Partial<TrainingFormConfig> {
  const known = hasManifest(info) ? info : null;

  // ---- costly:决定用哪个预处理池的那些值 -------------------------------------------------
  const pool: Partial<TrainingFormConfig> = {};
  if (known) {
    const fields = POOL_FORM_FIELDS[backend];
    // `loudnorm` 三态:只有盘上真的回答了才动表单(null = 不知道 ⇒ 用户的值不动)。
    if (fields.loudnorm && known.loudnorm !== null) {
      (pool as Record<string, unknown>)[fields.loudnorm] = known.loudnorm;
    }
    // 浅扩散有宿主时份数是**继承**来的,参数页显示的是一行只读文字而不是输入框 ——
    // 往一个不存在的控件里回填一个后端会忽略的数字,只会让 `diffAugCopies` 在切回
    // diff-first 的槽时带着别人的值。
    const augIsThisRunsChoice = backend !== "sovits_diff" || !known.has_main_progress;
    if (fields.augCopies && augIsThisRunsChoice) {
      (pool as Record<string, unknown>)[fields.augCopies] = known.aug_copies;
    }
  }

  // ---- locked:不还原就会被拒的那些值 -----------------------------------------------------
  if (backend === "rvc") {
    const v = known?.version === "v1" || known?.version === "v2" ? known.version : undefined;
    const sr =
      known?.sample_rate === "32k" || known?.sample_rate === "40k" || known?.sample_rate === "48k"
        ? known.sample_rate
        : undefined;
    return { ...pool, ...(v ? { version: v } : {}), ...(sr ? { sampleRate: sr } : {}) };
  }
  if (backend === "sovits") {
    const manifest = known?.version === "4.1" || known?.version === "4.0" ? known.version : undefined;
    const v = pin ?? manifest ?? "4.1";
    return {
      ...pool,
      sovitsVersion: v,
      // 响度嵌入 is 4.1-only and is baked into the graph AND the wire inputs.
      ...(v === "4.1" && !pin && known?.vol_embedding != null
        ? { sovitsVolEmbedding: known.vol_embedding }
        : {}),
    };
  }
  if (backend === "sovits_diff") {
    // k_step_max 一旦有扩散进度就被锁死(`RESUME_KSTEP_MISMATCH`);判据与 `resumeWouldBeGuarded`
    // 对这个 backend 用的是同一个信号(扩散自己的进度,不是宿主 manifest)。
    return { ...pool, ...(known && known.diff_steps > 0 ? { diffKStepMax: known.diff_k_step_max } : {}) };
  }
  // sovits_v2 / vocoder 的 version 与 sample_rate 是固定标记,没有 locked 项要还原。
  return pool;
}
