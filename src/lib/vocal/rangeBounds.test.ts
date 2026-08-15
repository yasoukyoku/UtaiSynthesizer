/** S146e —— 两个边界旋钮的判据。
 *
 *  ⚠ 这是**唯一**能红的地方:仓里没有组件渲染测试(vitest.config.ts = node 环境,
 *  include 只收 `*.test.ts`),所以放进 TSX 的逻辑一条都验不到。凡是这两个旋钮的规则,
 *  必须落在 rangeBounds.ts 里,并在这里被钉住。 */
import { describe, it, expect } from "vitest";
import { MIN_COMFORT_SPAN, autoBounds, boundsAreEdited, boundsPayload, clampBounds, scanBand, targetRange } from "./rangeBounds";
import type { SpeakerRangeRecord } from "./rangeTest";
import type { Bounds } from "./rangeBounds";

function rec(over: Partial<SpeakerRangeRecord> = {}): SpeakerRangeRecord {
  return {
    usable: [36, 80],
    comfort: [36, 79],
    comfort_auto: [36, 79],
    semitones: {},
    tested_at: "2026-08-14",
    ...over,
  } as SpeakerRangeRecord;
}

describe("后端那道闸(comfort ⊆ 扫描量出来的可用域)永远满足", () => {
  // ⛔ S146f:这道闸从「comfort ⊆ usable」改成了「comfort ⊆ usable_auto ∪ usable」。
  // 改口的理由是一次真实退化:用户把可用上限拖到 74,旧夹取**把他的 comfort 从 79 一起
  // 拖到 74**,此后没有任何落点够得着它 ⇒ 那个旋钮悄悄不做事了(每一组都在打回退行)。
  // 拆分之后两者正交:usable 说「哪些音要救」,comfort 说「落点去哪」,谁也不许动谁。
  it("⭐ 收窄可用上界【不再】把目标范围一起拖下去", () => {
    const sp = rec({ usable_auto: [36, 80] } as Partial<SpeakerRangeRecord>);
    const e = clampBounds(sp, [36, 60], [36, 79]);
    expect(e.usable).toEqual([36, 60]);
    expect(e.comfort).toEqual([36, 79]);
  });

  it("抬高可用下界同样不动目标范围", () => {
    const sp = rec({ usable_auto: [36, 80] } as Partial<SpeakerRangeRecord>);
    expect(clampBounds(sp, [60, 80], [36, 79]).comfort).toEqual([36, 79]);
  });

  it("穷举:任何一对提议都产出后端收得下的形状(comfort ⊆ 扫描带)", () => {
    const sp = rec({ usable_auto: [36, 80] } as Partial<SpeakerRangeRecord>);
    const [aLo, aHi] = scanBand(sp);
    for (let ul = 20; ul <= 100; ul += 7)
      for (let uh = 20; uh <= 100; uh += 11)
        for (let cl = 20; cl <= 100; cl += 13)
          for (let ch = 20; ch <= 100; ch += 17) {
            const e = clampBounds(sp, [ul, uh], [cl, ch]);
            expect(e.usable[0]).toBeLessThanOrEqual(e.usable[1]);
            expect(e.comfort[0]).toBeLessThanOrEqual(e.comfort[1]);
            expect(e.comfort[0]).toBeGreaterThanOrEqual(aLo);
            expect(e.comfort[1]).toBeLessThanOrEqual(aHi);
            expect(e.usable[0]).toBeGreaterThanOrEqual(aLo);
            expect(e.usable[1]).toBeLessThanOrEqual(aHi);
          }
  });

  it("退化的可用范围被撑到最小跨度,而目标范围不受牵连", () => {
    const sp = rec({ usable_auto: [36, 80] } as Partial<SpeakerRangeRecord>);
    const e = clampBounds(sp, [70, 70], [36, 79]);
    expect(e.usable[1] - e.usable[0]).toBeGreaterThanOrEqual(MIN_COMFORT_SPAN);
    expect(e.comfort).toEqual([36, 79]);
  });

  it("目标范围可以【高于】救援线 —— 这正是拆分之后要允许的组合", () => {
    // 「74 以上的音都救,但落点别高过 79」是一句合法且有用的话。
    const sp = rec({ usable_auto: [36, 80] } as Partial<SpeakerRangeRecord>);
    const e = clampBounds(sp, [36, 74], [36, 79]);
    expect(e.usable[1]).toBe(74);
    expect(e.comfort[1]).toBe(79);
  });
});

describe("可用域夹在扫描量出来的范围之内", () => {
  it("拉不出扫描的上界", () => {
    // 拉出去没有意义:slot_singable = 在 usable 内 ∧ 扫描说这个格子能唱,
    // 而扫描没量过的格子永远不能唱 ⇒ 那会是一个拖得动、但什么也不会发生的旋钮。
    expect(clampBounds(rec(), [36, 96], [36, 79]).usable[1]).toBe(80);
  });

  it("老记录(没有 usable_auto)拿今天的 usable 当扫描答案", () => {
    expect(scanBand(rec())).toEqual([36, 80]);
  });

  it("已经被压低过的记录,还原回的是扫描值而不是当前值", () => {
    const sp = rec({ usable: [36, 60], comfort: [36, 58], usable_auto: [36, 80] } as Partial<SpeakerRangeRecord>);
    expect(scanBand(sp)).toEqual([36, 80]);
    expect(autoBounds(sp).usable).toEqual([36, 80]);
    // ⛔ 反向对照:没有 usable_auto 时「还原」只能回到当前值 —— 这正是必须补这一列的理由。
    const legacy = rec({ usable: [36, 60], comfort: [36, 58] });
    expect(autoBounds(legacy).usable).toEqual([36, 60]);
  });
});

describe("落盘载荷", () => {
  it("两个边界在同一份载荷里,而且顺带补上 usable_auto", () => {
    const p = boundsPayload(rec(), [36, 60], [36, 79]) as SpeakerRangeRecord & { usable_auto: [number, number] };
    expect(p.usable).toEqual([36, 60]);
    expect(p.comfort).toEqual([36, 79]); // S146f: 不再被 usable 拖走
    expect(p.usable_auto).toEqual([36, 80]);
  });

  it("不碰记录里的其他任何东西(扫描数据必须原样活下来)", () => {
    const sp = rec({ semitones: { "70": [1, 1, -1, 0.2] }, semitones_onset: { "70": [2, 0.95] }, scan_version: 3 });
    const p = boundsPayload(sp, [36, 60], [36, 58]);
    expect(p.semitones).toBe(sp.semitones);
    expect(p.semitones_onset).toBe(sp.semitones_onset);
    expect(p.scan_version).toBe(3);
    expect(p.comfort_auto).toEqual(sp.comfort_auto);
    expect(p.tested_at).toBe(sp.tested_at);
  });

  it("二次编辑不会把 usable_auto 覆盖成被压低后的值", () => {
    // ⛔ 这条是那种一次编辑看不出来、第二次才炸的形状:若 boundsPayload 每次都拿当前
    // usable 去写 usable_auto,用户压两次之后就再也回不到扫描值了。
    const once = boundsPayload(rec(), [36, 60], [36, 58]);
    const twice = boundsPayload(once, [36, 50], [36, 48]) as SpeakerRangeRecord & { usable_auto: [number, number] };
    expect(twice.usable_auto).toEqual([36, 80]);
    expect(autoBounds(twice).usable).toEqual([36, 80]);
  });
});

describe("「已被手动改过」的判定", () => {
  it("新鲜记录 = 没改过", () => {
    expect(boundsAreEdited(rec())).toBe(false);
  });

  it("压低可用上界 = 改过", () => {
    expect(boundsAreEdited(boundsPayload(rec(), [36, 60], [36, 58]))).toBe(true);
  });

  it("只动目标范围也算改过", () => {
    expect(boundsAreEdited(boundsPayload(rec(), [36, 80], [50, 70]))).toBe(true);
  });

  it("还原之后回到「没改过」", () => {
    const edited = boundsPayload(rec(), [36, 60], [36, 58]);
    const a = autoBounds(edited);
    expect(boundsAreEdited(boundsPayload(edited, a.usable, a.comfort))).toBe(false);
  });
});

describe("targetRange —— 读侧愈合(镜像 Rust 的 speaker_range)", () => {
  // ⛔ 用户实测撞上的那个 bug 就在这条上:把可用上限拖到 D5(74)、目标上限设到 G5(79),
  // 存盘完全正确,但读回来时旧判据拿 `usable` 判有效性 ⇒ 79 被判「逃出可用域」⇒ 退回
  // comfort_auto(也超)⇒ 兜底成 usable ⇒ 显示与下一次 OK 全变成 74。**存对了、读错了**。
  it("⭐ 目标范围高于可用范围时,读回来必须原样活着", () => {
    const sp = rec({
      usable: [36, 74], comfort: [36, 79], comfort_auto: [36, 79], usable_auto: [36, 80],
    } as Partial<SpeakerRangeRecord>);
    expect(targetRange(sp)).toEqual([36, 79]);
  });

  it("站不住的存值退回 comfort_auto,再退回整条扫描带", () => {
    const mk = (comfort: Bounds, auto: Bounds) =>
      rec({ usable: [42, 70], comfort, comfort_auto: auto } as Partial<SpeakerRangeRecord>);
    expect(targetRange(mk([42, 42], [42, 70]))).toEqual([42, 70]); // 退化 → auto
    expect(targetRange(mk([42, 42], [50, 52]))).toEqual([42, 70]); // auto 也退化 → 扫描带
    expect(targetRange(mk([45, 60], [42, 70]))).toEqual([45, 60]); // 站得住的存值胜出
  });

  it("参照系是扫描带,不是 usable —— 换个说法钉同一条", () => {
    const legacy = rec({ usable: [42, 70], comfort: [45, 60], comfort_auto: [42, 70] } as Partial<SpeakerRangeRecord>);
    expect(scanBand(legacy)).toEqual([42, 70]); // 无 usable_auto ⇒ 参照系 = usable,行为逐字照旧
    const split = rec({
      usable: [42, 55], comfort: [45, 68], comfort_auto: [42, 70], usable_auto: [42, 70],
    } as Partial<SpeakerRangeRecord>);
    expect(targetRange(split)).toEqual([45, 68]); // 目标可以整段落在 usable 之外
  });
});

describe("最小跨度", () => {
  it("退化的区间被撑开到最小跨度,且不越出参照系", () => {
    // ⚠ 参照系是 usable_auto ∪ usable,所以两列都得钉住,否则量的是别的区间。
    const sp = rec({ usable: [42, 70], usable_auto: [42, 70] } as Partial<SpeakerRangeRecord>);
    expect(clampBounds(sp, [42, 42], [42, 42]).comfort).toEqual([42, 42 + MIN_COMFORT_SPAN]);
    expect(clampBounds(sp, [70, 70], [70, 70]).comfort).toEqual([70 - MIN_COMFORT_SPAN, 70]);
  });

  it("提议写反了会被排序,而不是被拒绝", () => {
    const sp = rec({ usable: [42, 70], usable_auto: [42, 70] } as Partial<SpeakerRangeRecord>);
    expect(clampBounds(sp, [60, 50], [60, 50]).usable).toEqual([50, 60]);
  });

  it("参照系本身就比最小跨度窄时照抄它,不造假", () => {
    const sp = rec({ usable: [60, 63], comfort: [61, 61], usable_auto: [60, 63] } as Partial<SpeakerRangeRecord>);
    expect(clampBounds(sp, [61, 61], [61, 61]).comfort).toEqual([60, 63]);
  });
});
