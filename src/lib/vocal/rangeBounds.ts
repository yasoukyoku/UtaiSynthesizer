/** S146e — 编辑「可用域 / 舒适区」两个边界的**唯一**一份逻辑。
 *
 *  为什么单开一个纯模块:这两个旋钮现在有**两个**入口(资源管理器的 VoiceRangeRow、
 *  人声侧栏的音域扩展栏),而仓里**根本没有组件渲染测试**(vitest.config.ts 是 node 环境、
 *  include 只收 `*.test.ts`)⇒ 放在 TSX 里的逻辑一条判据都不会红。所以夹取、载荷构造、
 *  还原语义全部落在这里,TSX 只做壳。
 *
 *  ⛔ 后端 `validate_range_record`(vocal_range.rs)硬性要求
 *  `u_lo ≤ u_hi ∧ c_lo ≤ c_hi ∧ c_lo ≥ u_lo ∧ c_hi ≤ u_hi`,违反就是 `RANGE_INVALID` ——
 *  而这条错误今天在 UI 上**完全不可见**(无 catch、无文案)。收窄 usable 时若不同时把
 *  comfort 收进去,用户会得到一次静默失败。⇒ 两个边界只能在**同一个载荷**里一次写完,
 *  这就是 `boundsPayload` 存在的理由,也是它不接受「只改一个」的理由。 */
import { clampComfort, MIN_COMFORT_SPAN, type SpeakerRangeRecord } from "./rangeTest";

export type Bounds = [number, number];

/** 一次编辑的完整结果 —— 两个边界永远成对出现,不许分开落盘。 */
export interface RangeBoundsEdit {
  usable: Bounds;
  comfort: Bounds;
}

/** 扫描当初量出来的可用域。
 *
 *  老记录(S146e 之前)没有这一列 —— 那时 `usable` 就**是**扫描的答案(没有任何 UI 能改它),
 *  所以缺列时拿 `usable` 顶替是事实陈述,不是猜测。 */
export function autoUsable(sp: SpeakerRangeRecord): Bounds {
  const a = (sp as SpeakerRangeRecord & { usable_auto?: Bounds }).usable_auto;
  return a && a.length === 2 ? [a[0], a[1]] : [sp.usable[0], sp.usable[1]];
}

/** 把用户提的一对边界夹成后端一定收得下的形状。
 *
 *  ⛔ `usable` 夹在**扫描量出来的**可用域之内:往外拉没有意义 —— `slot_singable` 是
 *  「在 usable 内 ∧ 扫描说这个格子能唱」,扫描没量过的格子永远不能唱,所以一个能拉到
 *  扫描范围之外的旋钮就是一个用户看得见、拖得动、而什么也不会发生的旋钮。 */
export function clampBounds(sp: SpeakerRangeRecord, usable: Bounds, comfort: Bounds): RangeBoundsEdit {
  const [aLo, aHi] = autoUsable(sp);
  let lo = Math.round(Math.min(Math.max(usable[0], aLo), aHi));
  let hi = Math.round(Math.min(Math.max(usable[1], aLo), aHi));
  if (hi < lo) [lo, hi] = [hi, lo];
  // 可用域至少要放得下一个合法的舒适区,否则下一行的 clampComfort 会被迫产出一个
  // 逃出 usable 的区间,后端当场 RANGE_INVALID。
  if (hi - lo < MIN_COMFORT_SPAN) {
    if (aHi - aLo < MIN_COMFORT_SPAN) [lo, hi] = [aLo, aHi]; // 扫描本身就窄:照抄,别造假
    else if (hi + (MIN_COMFORT_SPAN - (hi - lo)) <= aHi) hi = lo + MIN_COMFORT_SPAN;
    else lo = hi - MIN_COMFORT_SPAN;
  }
  // ⛔ S146f: comfort is clamped into the SCAN's band, **not** into the edited `usable`.
  // The old clamp had a concrete cost the user hit: dragging the ceiling to 74 silently pulled
  // their comfort from 79 down to 74, after which no landing could ever reach it and the knob
  // stopped doing anything (every rescue logged the fallback). The two knobs are orthogonal now
  // — `usable` says WHICH notes to rescue, `comfort` says WHERE they land — so neither may move
  // the other. The backend agrees (validate_range_record bounds comfort by usable_auto ∪ usable).
  return { usable: [lo, hi], comfort: clampComfort([aLo, aHi], [Math.round(comfort[0]), Math.round(comfort[1])]) };
}

/** 构造一次 `set_model_vocal_range` 的完整 speaker 记录。
 *
 *  ⚠ 顺带把 `usable_auto` 补写进去(缺列时 = 今天的 usable,见 `autoUsable`)。没有它,
 *  用户压低可用上界之后就**再也回不去**了 —— 而这两个旋钮按用户的说法是要来回调的。 */
export function boundsPayload(sp: SpeakerRangeRecord, usable: Bounds, comfort: Bounds): SpeakerRangeRecord {
  const edit = clampBounds(sp, usable, comfort);
  return { ...sp, usable_auto: autoUsable(sp), usable: edit.usable, comfort: edit.comfort } as SpeakerRangeRecord;
}

/** 「还原」= 回到扫描量出来的两个边界。comfort_auto 可能比新 usable 宽 ⇒ 一并夹取。 */
export function autoBounds(sp: SpeakerRangeRecord): RangeBoundsEdit {
  const a = autoUsable(sp);
  return clampBounds(sp, a, sp.comfort_auto ?? a);
}

/** 从一个模型条目里取出某位歌手的音域记录。
 *
 *  ⚠ 记录是**逐歌手**的:`config.vocal_range.speakers[String(id)]`。借别的歌手的记录来用
 *  是主动错误 —— 合训出来的歌手音域本来就不同,那正是合训的意义。 */
export function speakerRecordOf(
  config: unknown,
  speakerId: number | null | undefined,
): SpeakerRangeRecord | null {
  const rec = (config as { vocal_range?: { speakers?: Record<string, SpeakerRangeRecord> } } | undefined)?.vocal_range;
  return rec?.speakers?.[String(speakerId ?? 0)] ?? null;
}

/** 这份记录的两个边界有没有被人手动动过(决定「还原」按钮要不要出现)。 */
export function boundsAreEdited(sp: SpeakerRangeRecord): boolean {
  const a = autoBounds(sp);
  return (
    sp.usable[0] !== a.usable[0] ||
    sp.usable[1] !== a.usable[1] ||
    sp.comfort[0] !== a.comfort[0] ||
    sp.comfort[1] !== a.comfort[1]
  );
}
