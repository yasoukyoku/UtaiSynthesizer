import { describe, expect, it } from "vitest";
import type { RunDetail, SlotDetail, WorkspaceInfo } from "../../store/training";
import { pickDiffHost, prepPoolLine, slotStarted, startedRun, visibleRuns } from "./slotRows";

/**
 * §E2E-M3 / M4 / M5(S141)—— 一个槽画成什么样。
 *
 * 这五个决策此前是 `ProjectDetail.tsx` 组件体内的内联表达式,vitest **结构上**够不着,
 * 所以它们只有源码在守。下面分三层:
 *
 * ⑴ **具名用例** —— 每一条都造出「它的反面能不能通过」那个形状(backlog 的规矩)。
 * ⑵ **差分模糊** —— 把搬家**之前**那几行原样抄成 `legacy*`,和搬家之后的模块逐个输入对拍。
 *    这是 S91 定的那把尺子:声明「纯提取、行为逐位不变」时该给的证据形状是差分,不是
 *    「我读了代码,它们看起来一样」。⛔ `legacy*` 抄自组件里的原文,不是抄自新模块 ——
 *    否则它只是在证明文件等于它自己(S108 那条「对拍的两边必须有一边是我改不动的东西」)。
 * ⑶ **语料自检** —— 模糊的输入集必须真的走到有分辨力的形状(带主模型的 run **不是**第一个、
 *    零个 run、`id === ""` 的伪造行……),否则它跑一万次也只是把同一格重复了一万次。
 */

const INFO: WorkspaceInfo = {
  exists: true,
  family: "sovits",
  version: "",
  sample_rate: "",
  has_main_progress: false,
  diff_steps: 0,
  best_resume_step: null,
  diff_best_resume_step: null,
  aug_copies: 0,
  loudnorm: null,
  has_dataset: false,
  vol_embedding: null,
  n_speakers: 1,
  speakers: [],
  diff_k_step_max: 0,
};

function mkRun(
  id: string,
  over: Omit<Partial<RunDetail>, "info"> & { info?: Partial<WorkspaceInfo> } = {},
): RunDetail {
  const { info, ...rest } = over;
  return {
    id,
    modelName: "",
    info: { ...INFO, ...info },
    resumeStep: null,
    hasResumePoint: false,
    ckptCount: 0,
    ckptBytes: 0,
    ...rest,
  };
}

function mkSlot(runs: RunDetail[], prep: { count?: number; bytes?: number } = {}): SlotDetail {
  return {
    family: "sovits",
    runs,
    bytes: 0,
    ckptCount: 0,
    ckptBytes: 0,
    prepPoolCount: prep.count ?? 0,
    prepPoolBytes: prep.bytes ?? 0,
  };
}

// ─── ⑵ 搬家之前的原文(ProjectDetail.tsx :701 / :716 / :719 / :520-528 / :831-836)──────
const legacyStartedRun = (r: RunDetail) => r.hasResumePoint || r.info.has_main_progress;
const legacyVisibleRuns = (runs: RunDetail[]) => runs.filter((r) => r.id !== "" || legacyStartedRun(r));
const legacyStarted = (runs: RunDetail[]) => legacyVisibleRuns(runs).length > 0;
const legacyDiffHost = (sovitsSlot: SlotDetail | undefined) =>
  sovitsSlot?.runs.find((r) => r.info.has_main_progress) ?? sovitsSlot?.runs[0];
const legacyPinned = (diffHost: RunDetail | undefined): "4.1" | "4.0" | undefined =>
  // ⚠ 原文里这里是 `diffHost.info.version`(无 `!`):组件里 `diffHost` 是一个 const,TS 能从
  // 条件里把它收窄成非 undefined;搬成函数参数之后收窄没了,所以补一个 `!`。
  // 这是**类型层的记账**,不是行为改动 —— 条件为真时它必然非空。
  diffHost?.info.version === "4.1" || diffHost?.info.version === "4.0"
    ? diffHost!.info.version
    : undefined;
const legacyDiffSteps = (diffHost: RunDetail | undefined) => diffHost?.info.diff_steps ?? 0;
const legacyPrepShow = (slot: SlotDetail | undefined) => (slot?.prepPoolCount ?? 0) > 0;
const legacyPrepCount = (slot: SlotDetail | undefined) => slot?.prepPoolCount ?? 0;
const legacyPrepBytes = (slot: SlotDetail | undefined) => slot?.prepPoolBytes ?? 0;

describe("visibleRuns / slotStarted(§E2E-M3)", () => {
  it("★ 一个【铸了但没练成】的真 run 必须画得出来", () => {
    // ④e 修的正是这一条:`run_manifest.json` 在 spawn 之前就写好,随后的切片/f0/特征要跑
    // 几小时 —— 中途崩/强停留下的就是这个形状:有目录、有 manifest、没有断点、没有主模型。
    // 它此前被 `runs.filter(startedRun)` 藏掉,而它是一个满强度的冻结源(manifest 里有
    // speakers ⇒ 锁住整个项目的数据集),卡片却写着「尚未开始」。
    const born = mkRun("raaa");
    const runs = [born];
    expect(startedRun(born)).toBe(false);
    expect(visibleRuns(runs).map((r) => r.id)).toEqual(["raaa"]);
    expect(slotStarted(runs)).toBe(true);
  });

  it("★ 反面:后端在【零 run】时补的那条 id === \"\" 伪造行,没练出东西就不该画", () => {
    // 它寻址的是**槽根**,不是某个 run —— 那种槽本来就没有 run 可挑。这条与上一条一起,
    // 才把「真 run 一律画」和「伪造行按练没练过滤」分开;少任何一条,
    // `r.id !== "" ||` 或 `|| startedRun(r)` 其中一半就可以整个删掉而不变红。
    expect(visibleRuns([mkRun("")])).toEqual([]);
    expect(slotStarted([mkRun("")])).toBe(false);
    // 同一条伪造行,练出过东西就要画
    expect(visibleRuns([mkRun("", { hasResumePoint: true })]).length).toBe(1);
    expect(visibleRuns([mkRun("", { info: { has_main_progress: true } })]).length).toBe(1);
  });

  it("startedRun 的两个条件各自成立就够(它们不是同一件事)", () => {
    expect(startedRun(mkRun("r", { hasResumePoint: true }))).toBe(true);
    expect(startedRun(mkRun("r", { info: { has_main_progress: true } }))).toBe(true);
    expect(startedRun(mkRun("r"))).toBe(false);
  });

  it("空槽:没有 run ⇒ 没有行,也不算开始过", () => {
    expect(visibleRuns([])).toEqual([]);
    expect(slotStarted([])).toBe(false);
  });
});

describe("pickDiffHost(§E2E-M4)", () => {
  it("★ 带主模型的那个 run 被选中 —— 而它必须【不是】id 字典序第一个", () => {
    // ⛔ 这一条的夹具是承重的:只放一个带主模型的 run 的话,`find(...)` 与 `runs[0]`
    // 返回**同一个对象**,把 find 换成 [0] 的变异结构上不可见(S125/S126 那族装饰性判据)。
    const slot = mkSlot([
      mkRun("raaa"),
      mkRun("rbbb", { info: { has_main_progress: true, version: "4.1", diff_steps: 700 } }),
    ]);
    const c = pickDiffHost(slot);
    expect(c.host?.id).toBe("rbbb");
    expect(c.source).toBe("main-progress");
    expect(c.pinnedVersion).toBe("4.1");
    expect(c.steps).toBe(700);
  });

  it("★ 一个 run 且它带主模型时,答案要自己说出【出处】", () => {
    // 今天生产上的常态就是这一格,而它正是 find 与 [0] 无法区分的那一格 ——
    // `source` 存在的唯一理由就是让这一格仍然有分辨力。
    const c = pickDiffHost(mkSlot([mkRun("raaa", { info: { has_main_progress: true } })]));
    expect(c.host?.id).toBe("raaa");
    expect(c.source).toBe("main-progress");
    const d = pickDiffHost(mkSlot([mkRun("raaa")]));
    expect(d.host?.id).toBe("raaa");
    expect(d.source).toBe("fallback-first");
  });

  it("零个带主模型 ⇒ 回落到第一个,且版本不 pin", () => {
    const c = pickDiffHost(mkSlot([mkRun("raaa", { info: { version: "4.1" } }), mkRun("rbbb")]));
    expect(c.host?.id).toBe("raaa");
    expect(c.source).toBe("fallback-first");
    expect(c.withMainProgress).toBe(0);
    // ⚠ 回落到的那个 run 自己带 4.1 ⇒ 仍然 pin。这是**今天的行为**,钉住它是为了
    // 让 B2-⑤ 改它的时候必须看见自己在改什么,不是因为它显然对。
    expect(c.pinnedVersion).toBe("4.1");
  });

  it("★ 两个 run 都带主模型 ⇒ 歧义必须变成一个【可断言的事实】", () => {
    // ⛔★★S133:此前这里有一句注释说「这条 find 是肯定事实,不是挑第一个」,而 ④e 的 flip
    // 让它变成了假话。行为在 B2-⑤ 之前保持现状(仍然按 list_runs 的字典序取第一个),
    // 但歧义本身从「一句注释」升级成一个数 —— 注释不会红,这个数会。
    const c = pickDiffHost(
      mkSlot([
        mkRun("raaa", { info: { has_main_progress: true, version: "4.0" } }),
        mkRun("rbbb", { info: { has_main_progress: true, version: "4.1" } }),
      ]),
    );
    expect(c.withMainProgress).toBe(2);
    expect(c.host?.id).toBe("raaa");
    expect(c.pinnedVersion).toBe("4.0");
  });

  it("槽不存在 / 槽里没有 run ⇒ 没有宿主,而且这两种要与「有宿主」分得开", () => {
    for (const slot of [undefined, mkSlot([])]) {
      const c = pickDiffHost(slot);
      expect(c.host).toBeUndefined();
      expect(c.source).toBe("none");
      expect(c.pinnedVersion).toBeUndefined();
      expect(c.steps).toBe(0);
    }
  });

  it("只有 4.1 / 4.0 会被 pin —— 别的版本串一律 undefined", () => {
    for (const v of ["", "v1", "v2", "40k", "4.10", "4"]) {
      expect(pickDiffHost(mkSlot([mkRun("r", { info: { version: v } })])).pinnedVersion)
        .toBeUndefined();
    }
    for (const v of ["4.1", "4.0"] as const) {
      expect(pickDiffHost(mkSlot([mkRun("r", { info: { version: v } })])).pinnedVersion).toBe(v);
    }
  });
});

describe("prepPoolLine(§E2E-M5 的渲染那一跳)", () => {
  it("零份 ⇒ 不画;有池 ⇒ 画,而且两个数来自同一个读数", () => {
    expect(prepPoolLine(mkSlot([], { count: 0, bytes: 0 })).show).toBe(false);
    // ⚠ 「零份却有字节」是盘上真会出现的形状(池目录读不动时 Rust 会报错,但一个
    //   刚被清空的槽是 0/0);判据钉的是 show 只由 count 决定。
    expect(prepPoolLine(mkSlot([], { count: 0, bytes: 999 })).show).toBe(false);
    const l = prepPoolLine(mkSlot([], { count: 2, bytes: 3000 }));
    expect(l).toEqual({ show: true, count: 2, bytes: 3000 });
  });

  it("槽不存在 ⇒ 不画,而不是画一行「0 份」", () => {
    expect(prepPoolLine(undefined)).toEqual({ show: false, count: 0, bytes: 0 });
  });
});

// ─── ⑵⑶ 差分模糊 + 语料自检 ───────────────────────────────────────────────────────
describe("★ 搬家前后逐个输入对拍(纯提取的证据形状)", () => {
  it("2000 个随机槽上,新模块与搬家前的表达式给出同一个答案", () => {
    // 固定种子的 LCG:随机但**可复现**——一条只在某次运行里红的判据没人查得动。
    let seed = 20260812;
    const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
    // `!`:下标恒在范围内(rnd() ∈ [0,1) ⇒ floor 落在 0..len-1,取模只是保险)。
    const pick = <T,>(xs: readonly T[]): T => xs[Math.floor(rnd() * xs.length) % xs.length]!;

    const IDS = ["", "raaa", "rbbb", "rccc"] as const;
    const VERSIONS = ["", "4.0", "4.1", "v1", "v2", "40k"] as const;

    let sawFindDiffersFromFirst = 0;
    let sawEmpty = 0;
    let sawFakeRow = 0;
    let sawTwoMain = 0;

    for (let i = 0; i < 2000; i++) {
      const n = Math.floor(rnd() * 4); // 0..3 个 run
      const runs: RunDetail[] = [];
      for (let k = 0; k < n; k++) {
        runs.push(
          mkRun(pick(IDS), {
            hasResumePoint: rnd() < 0.5,
            info: {
              has_main_progress: rnd() < 0.4,
              version: pick(VERSIONS),
              diff_steps: Math.floor(rnd() * 3) * 700,
            },
          }),
        );
      }
      const slot: SlotDetail | undefined =
        rnd() < 0.1
          ? undefined
          : mkSlot(runs, { count: Math.floor(rnd() * 3), bytes: Math.floor(rnd() * 5000) });

      const mains = runs.filter((r) => r.info.has_main_progress);
      if (mains.length > 0 && runs[0] !== mains[0]) sawFindDiffersFromFirst++;
      if (n === 0) sawEmpty++;
      if (runs.some((r) => r.id === "")) sawFakeRow++;
      if (mains.length > 1) sawTwoMain++;

      const got = pickDiffHost(slot);
      const want = legacyDiffHost(slot);
      expect(got.host).toBe(want);
      expect(got.pinnedVersion).toBe(legacyPinned(want));
      expect(got.steps).toBe(legacyDiffSteps(want));

      expect(visibleRuns(runs)).toEqual(legacyVisibleRuns(runs));
      expect(slotStarted(runs)).toBe(legacyStarted(runs));
      for (const r of runs) expect(startedRun(r)).toBe(legacyStartedRun(r));

      const line = prepPoolLine(slot);
      expect(line.show).toBe(legacyPrepShow(slot));
      expect(line.count).toBe(legacyPrepCount(slot));
      expect(line.bytes).toBe(legacyPrepBytes(slot));
    }

    // ⑶ 语料自检:上面那 2000 次里,真的走到过有分辨力的形状吗?
    // 少了这一段,一个只生成「单 run、带主模型」的生成器会让整个对拍变成一格重复 2000 次。
    expect(sawFindDiffersFromFirst).toBeGreaterThan(50);
    expect(sawEmpty).toBeGreaterThan(50);
    expect(sawFakeRow).toBeGreaterThan(50);
    expect(sawTwoMain).toBeGreaterThan(50);
  });
});
