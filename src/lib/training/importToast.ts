/**
 * 导入 / 附加之后**弹什么**的全部决策 —— 档位、行、以及弹几条。
 *
 * ## 为什么要把它从组件里搬出来(§E2E-M6 的前端那一跳 + §E2E-M7,S141)
 *
 * 后端把「能恢复的失败」报成 **warning** 而不是 error(`WARN_INDEX_MISSING` 最典型,
 * §F9 的 `WARN_DIFFUSION_VOCODER_CUSTOM` 是那条**被真实用户报过**的路)。这些 warning
 * 的整条呈现链此前活在 `TrainingPage.tsx` 的三个 async 闭包里,没有可导出的决策函数 ⇒
 * vitest 驱不动 ⇒ M6 的 Rust 半边做完了、而「它到底有没有到达用户」零判据。
 *
 * ## ⛔ 记忆里那条阴性对照是**错的**,别照它写
 *
 * §E2E-M7 原文写「无 warning 时**不弹**」。三条路在无 warning 时**都弹一条** toast ——
 * 照原文写会写出一条永远红的断言,然后被判成「假红」耸肩带过(S129 铁律点名的那类)。
 * 真正的判据落在 **档位与追加行**:无 warning ⇒ `success` 且**零追加行**。
 *
 * ## 为什么是四个函数而不是一个
 *
 * 三条路的漏斗**本来就不同**,而这一笔不改行为:
 * * 导入(单条)把 warning 折进**同一条** toast(改档 + 追加行);
 * * 批量另有一档 `error`(有失败时,warning **也要一起呈现**,不能被失败吃掉);
 * * 附加浅扩散是**先无条件弹一条 success,再每条 warning 各弹一条 info**。
 * 硬把它们合成一个函数就是在这一笔里偷偷改 UX。⇒ 形状留原样,判据分别钉。
 */

export type ToastLevel = "success" | "info" | "error";

export interface ToastCall {
  text: string;
  level: ToastLevel;
}

/**
 * 后端报的 warning ＋ 前端那条「我们不知道该去哪找索引」必须走**同一个漏斗**(§E2E-M1)。
 *
 * ⛔ 否则它与「确实没有索引」在界面上长得一模一样,而两者的后果完全不同 ——
 * 那正是 M1 那一轮买回来的真缺陷(两种情况此前被合并成同一个 `undefined`)。
 */
export function collectWarningCodes(
  outcome: { warnings?: string[] } | null | undefined,
  frontendWarn?: string | null,
): string[] {
  return [...(outcome?.warnings ?? []), ...(frontendWarn ? [frontendWarn] : [])];
}

/**
 * 单条导入(以及任何「一条 toast 说完」的路):档位由**有没有追加行**决定。
 *
 * ⚠ 无 warning 时它**仍然弹**一条 —— 用户需要知道「装上了」。判据钉的是
 * `level === "success" && 追加行为空`,不是「没有 toast」。
 */
export function importToast(base: string, warnLines: readonly string[]): ToastCall {
  return warnLines.length > 0
    ? { text: `${base}\n${warnLines.join("\n")}`, level: "info" }
    : { text: base, level: "success" };
}

export interface BatchImportToastInput {
  /** 全部成功时的那句话。 */
  doneText: string;
  /** 有失败时的那句话(带成功数/总数)。 */
  partialText: string;
  failed: readonly string[];
  warns: readonly string[];
}

/**
 * 批量导入的三档。
 *
 * ⛔ 有失败时,**warning 也要一起呈现**:一条「索引没装上」不会因为同一批里另有一条失败
 * 就变得不重要,而失败与 warning 分属两个不同的补救动作。
 */
export function batchImportToast(a: BatchImportToastInput): ToastCall {
  if (a.failed.length > 0) {
    return {
      text: `${a.partialText}\n${[...a.failed, ...a.warns].join("\n")}`,
      level: "error",
    };
  }
  if (a.warns.length > 0) {
    return { text: `${a.doneText}\n${a.warns.join("\n")}`, level: "info" };
  }
  return { text: a.doneText, level: "success" };
}

/**
 * 附加浅扩散:**先**一条 success,**再**每条 warning 各一条 info。
 *
 * ⚠ 它与导入是**两个不同的漏斗**,而记忆里一度把它归成「无 warning 时不弹」的那一类 ——
 * 实际上它无 warning 时也弹恰好一条。这里返回一个**数组**,就是为了让「弹几条」本身
 * 成为可断言的事实:§F9 那条 vocoder 提示走的正是这条路。
 */
export function attachToasts(base: string, warnLines: readonly string[]): ToastCall[] {
  return [
    { text: base, level: "success" as const },
    ...warnLines.map((w) => ({ text: w, level: "info" as const })),
  ];
}
