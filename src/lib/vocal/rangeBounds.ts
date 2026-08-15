/** 音域记录里那**两个边界**的全部逻辑 —— 读、夹、写、还原,只有这一份。
 *
 *  可用范围 = 哪些音要救(在它之内的照谱直接唱)
 *  目标范围 = 被救的音落在哪里(磁盘键名仍是 `comfort`;改键名会作废所有现存记录)
 *
 *  ⛔ **为什么全部挤在一个文件里**:这两个边界有三类消费者(资源管理器、人声侧栏、
 *  渲染前的预览),规则却只有一套。S146f 把「目标范围由扫描带约束、不由 usable 约束」
 *  改了一半 —— `clampBounds` 改了、读侧的 `targetRange` 漏了 —— 用户当场撞上:
 *  存进去 79、读回来 74。那是这套规则散落在两个文件里的直接代价。
 *  ⇒ 规则只写一次(`fitsIn`),读与写共用它;要改规矩,改这个文件。
 *
 *  ⚠ 这里也是**唯一**能被判据覆盖的地方:仓里没有组件渲染测试(vitest.config.ts 是
 *  node 环境、include 只收 `*.test.ts`),放进 TSX 的逻辑一条都不会红。TSX 只做壳。 */
import { MIN_COMFORT_SPAN } from "../../store/voice-models";
import type { SpeakerRangeRecord } from "./rangeTest";

export { MIN_COMFORT_SPAN };

export type Bounds = [number, number];

/** 一次编辑的完整结果 —— 两个边界永远成对出现。
 *  ⛔ 不许分开落盘:后端 `validate_range_record` 同时校验两者,只写一半会 `RANGE_INVALID`,
 *  而这条错误今天在 UI 上完全不可见(无 catch、无文案)。 */
export interface RangeBoundsEdit {
  usable: Bounds;
  comfort: Bounds;
}

/** ⭐ 扫描量出来的可用域 —— **一切边界判定的参照系**(Rust 侧叫 `SpeakerRange::reach`)。
 *
 *  老记录(S146e 之前)没有 `usable_auto` 这一列:那时 `usable` 就**是**扫描的答案
 *  (没有任何 UI 改得动它)⇒ 缺列时拿 `usable` 顶替是事实陈述,不是猜测。
 *  ⚠ 取并集:手改过的 sidecar 可能写出比 `usable` 更窄的 `usable_auto`,并集保证这一层
 *  只可能放宽、绝不会悄悄砍掉已有的行为。 */
export function scanBand(sp: SpeakerRangeRecord): Bounds {
  const a = (sp as SpeakerRangeRecord & { usable_auto?: Bounds }).usable_auto;
  if (!a || a.length !== 2) return [sp.usable[0], sp.usable[1]];
  return [Math.min(a[0], sp.usable[0]), Math.max(a[1], sp.usable[1])];
}

/** 一个区间在参照系里站不站得住 —— **夹取与读侧愈合共用的唯一判据**。 */
function fitsIn(band: Bounds | undefined, within: Bounds): boolean {
  return !!band && band[1] - band[0] >= MIN_COMFORT_SPAN && band[0] >= within[0] && band[1] <= within[1];
}

/** 把一对边界夹进参照系,并保证最小跨度(退化时向还放得下的一侧撑开)。 */
function clampInto(band: Bounds, within: Bounds): Bounds {
  let lo = Math.round(Math.min(Math.max(Math.min(band[0], band[1]), within[0]), within[1]));
  let hi = Math.round(Math.min(Math.max(Math.max(band[0], band[1]), within[0]), within[1]));
  if (hi - lo < MIN_COMFORT_SPAN) {
    if (within[1] - within[0] < MIN_COMFORT_SPAN) return [within[0], within[1]]; // 参照系本身就窄:照抄,别造假
    hi = Math.min(within[1], lo + MIN_COMFORT_SPAN);
    lo = Math.max(within[0], hi - MIN_COMFORT_SPAN);
  }
  return [lo, hi];
}

/** 用户提的一对边界 → 后端一定收得下的形状。
 *
 *  ⛔ 两个边界都夹在**扫描带**里,而且**互不牵连**。旧写法把 comfort 夹进编辑后的 usable,
 *  代价用户实测过:把可用上限拖到 74,目标范围被一起拖到 74,此后没有任何落点够得着它,
 *  那个旋钮就悄悄不做事了(每一组都在打回退行)。
 *  ⛔ usable 拉不出扫描带:扫描没量过的格子永远唱不出来,一个拖得动而什么也不会发生的
 *  旋钮比没有更糟。 */
export function clampBounds(sp: SpeakerRangeRecord, usable: Bounds, comfort: Bounds): RangeBoundsEdit {
  const band = scanBand(sp);
  return { usable: clampInto(usable, band), comfort: clampInto(comfort, band) };
}

/** ⭐ 渲染层实际瞄准的**目标范围** —— 镜像 Rust 的读侧愈合
 *  (`vocal_range.rs::speaker_range`:站不住的存值 → `comfort_auto` → 兜底整条扫描带)。
 *  UI 显示与滑条播种必须用这个,不是原始存值。 */
export function targetRange(sp: SpeakerRangeRecord): Bounds {
  const band = scanBand(sp);
  if (fitsIn(sp.comfort, band)) return sp.comfort;
  if (fitsIn(sp.comfort_auto, band)) return sp.comfort_auto;
  return band;
}

/** 「还原」的目标 = 扫描量出来的那一对。 */
export function autoBounds(sp: SpeakerRangeRecord): RangeBoundsEdit {
  const band = scanBand(sp);
  return clampBounds(sp, band, sp.comfort_auto ?? band);
}

/** 这份记录的两个边界有没有被人动过(决定「还原」按钮亮不亮)。 */
export function boundsAreEdited(sp: SpeakerRangeRecord): boolean {
  const a = autoBounds(sp);
  return (["usable", "comfort"] as const).some(
    (k) => sp[k][0] !== a[k][0] || sp[k][1] !== a[k][1],
  );
}

/** 构造一次 `set_model_vocal_range` 的完整 speaker 记录。
 *  ⚠ 顺带补写 `usable_auto`,否则用户压低可用上限之后**再也回不去**(只能重测)。 */
export function boundsPayload(sp: SpeakerRangeRecord, usable: Bounds, comfort: Bounds): SpeakerRangeRecord {
  const edit = clampBounds(sp, usable, comfort);
  return { ...sp, usable_auto: scanBand(sp), usable: edit.usable, comfort: edit.comfort };
}

/** 从模型条目里取出某位歌手的音域记录。
 *  ⚠ 记录是**逐歌手**的:借别的歌手的来用是主动错误 —— 合训出来的歌手音域本来就不同。 */
export function speakerRecordOf(
  config: unknown,
  speakerId: number | null | undefined,
): SpeakerRangeRecord | null {
  const rec = (config as { vocal_range?: { speakers?: Record<string, SpeakerRangeRecord> } } | undefined)?.vocal_range;
  return rec?.speakers?.[String(speakerId ?? 0)] ?? null;
}
