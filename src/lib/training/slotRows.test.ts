import { describe, expect, it } from "vitest";
import type { RunDetail, SlotDetail, WorkspaceInfo } from "../../store/training";
import {
  foldRunRows,
  newRunNameProblem,
  pickDiffHost,
  prepPoolLine,
  slotStarted,
  sortRunRows,
  startedRun,
  visibleRuns,
} from "./slotRows";

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
  has_preprocessing: false,
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

  it("★★S144 —— 正在跑的那一行永远画得出来:新槽第一次训练时它是【唯一】的证据", () => {
    // 机理逐条可查:新槽整场会话都是 layout 0(建槽没人写 slot.json、`migrate_layouts` 只在
    // 开机跑)⇒ `run_dir_for_start` 答**槽根**、不建 `runs/` ⇒ `list_runs` 恒空 ⇒ 后端补
    // `id: ""` 的伪造行,而 `snapshot.run_id` 同样是 `""`(`run_id_of` 对槽根就这么答)。
    // 第一个 `G_*.pth` 落盘要几十分钟,在那之前两个条件都假 ⇒ 这一行此前被过滤掉,
    // 而训练**正在跑** ⇒ 屏幕上是「未开始」+ 一颗槽级「开始训练」。
    const rows = [mkRun("")];
    expect(visibleRuns(rows, "")).toHaveLength(1);
    expect(slotStarted(rows, "")).toBe(true);
    // 阴性对照 ⑴ 没有训练在跑(`liveRunIdFor` 答 null)⇒ 逐字节回到今天的行为
    expect(visibleRuns(rows, null)).toEqual([]);
    expect(slotStarted(rows, null)).toBe(false);
    // 阴性对照 ⑵ 跑的是**别的** run ⇒ 不许拿它救这一行。
    // ⛔ 这一格是「`=== liveRunId`」与「liveRunId 非 null 就放行」之间**唯一**分得开的输入。
    expect(visibleRuns(rows, "rbbb")).toEqual([]);
    expect(slotStarted(rows, "rbbb")).toBe(false);
  });

  it("★ 那条豁免是**并集**:真 run 不许因为 liveRunId 而变得不画", () => {
    // ⛔ 防的是把新加的 `||` 写成 `&&` 那一族 —— 它在上面那条用例上完全不可见
    //    (伪造行两半都假时 `&&` 与 `||` 给同一个答案)。
    const real = [mkRun("raaa")];
    expect(visibleRuns(real, null).map((r) => r.id)).toEqual(["raaa"]);
    expect(visibleRuns(real, "rzzz").map((r) => r.id)).toEqual(["raaa"]);
    expect(visibleRuns(real, "raaa").map((r) => r.id)).toEqual(["raaa"]);
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

describe("foldRunRows(S141:run 多了才收)", () => {
  const runs = (n: number) => Array.from({ length: n }, (_, i) => mkRun(`r${i}`));

  it("★ 少量 run 时必须逐条照画 —— 用户原话「现在这样直接显示确实很清楚」", () => {
    // 阈值以内一条都不许收,连那行「还有 N 个」都不该出现(它自己也占一行)。
    for (const n of [0, 1, 2]) {
      const f = foldRunRows(runs(n), false);
      expect(f.rows.length, `${n} 条时不该折`).toBe(n);
      expect(f.hidden).toBe(0);
    }
  });

  it("★ 超过阈值才收,而且【收起来的条数要说得出来】", () => {
    const f = foldRunRows(runs(5), false);
    expect(f.rows.map((r) => r.id)).toEqual(["r0", "r1"]);
    expect(f.hidden, "「还有 N 个」的那个 N 必须是真的被收起来的条数").toBe(3);
  });

  it("展开之后一条不少 —— 折叠不许变成「看不见的行」", () => {
    const f = foldRunRows(runs(5), true);
    expect(f.rows.length).toBe(5);
    expect(f.hidden, "展开态还报 hidden>0,那行「还有 N 个」会和已经展开的行同时出现").toBe(0);
  });

  it("画出来的 + 收起来的 == 全部(任何 n 与 limit)", () => {
    // ⛔ 这条是防「切片切错一位」:off-by-one 不会让上面任何一条红,但会静默丢掉一个 run。
    for (let n = 0; n <= 8; n++) {
      for (let limit = 1; limit <= 4; limit++) {
        const f = foldRunRows(runs(n), false, limit);
        expect(f.rows.length + f.hidden, `n=${n} limit=${limit}`).toBe(n);
      }
    }
  });

  it("不改动传进来的数组(它是 store 里的那一份)", () => {
    // ⛔ S143:这一条原本只断言 `src.length` —— 那是一条**装饰性判据**:一次原地 `sort()`
    //    **不改长度**,从它底下原样穿过去。而 `slot.runs` 同时被 `pickDiffHost`(读 `runs[0]`
    //    当回落)与 store 里那一份共用 ⇒ 原地排会静默改掉浅扩散训练进哪个 run 目录。
    //    ⇒ 断言改成**顺序与元素身份**。
    const src = runs(5);
    const before = src.map((r) => r.id);
    foldRunRows(src, false);
    expect(src.map((r) => r.id), "入参被改动了").toEqual(before);
  });
});

describe("§E2E-M25 ⑴ —— 折叠不许藏起正在跑的那条(复合:sortRunRows → foldRunRows)", () => {
  /** ⛔⛔ 这一格的形状是承重的,而且它是 S143 侦察的对抗核验者点名的那条:
   *
   *  ⑴ 单独喂 `foldRunRows` 一份**已排序**的行去断言「正在跑的还在」是一句**恒真的话**——
   *     排序器已经把它放在 index 0 了,把折叠的 `slice` 换成任何取头部的写法它都绿,
   *     甚至把置顶整个删掉也绿(因为夹具是排完序才喂进来的)。那是 S128 的 L9 同族。
   *  ⑵ 所以输入必须是**未排序**的,而且正在跑的那一行要落在 `limit` **之后** ——
   *     这样「排序没做」与「折叠切错了」两种坏法都会让它红。
   *  ⑶ 而且要**成对**断言:其余行仍然被折起来(`hidden` 正确)。只断言「它在里面」的话,
   *     一个「有 liveRunId 就整段展开」的实现会通过,而那等于把折叠功能静默删掉。 */
  const unsorted = () => [
    mkRun("ra1", { modelName: "alpha" }),
    mkRun("rb2", { modelName: "beta" }),
    mkRun("rc3", { modelName: "gamma" }),
    mkRun("rd4", { modelName: "zzz-live" }), // 名字排**最后**,而且 index 3 > limit(=2)
  ];

  it("★★★ 正在跑的那一行,在一份【未排序】且它排在折叠线之后的输入上,仍然被画出来", () => {
    const rows = unsorted();
    // 夹具前提:不排序时它确实会被切掉(否则这条判据从出生就没有分辨力)。
    expect(foldRunRows(rows, false).rows.map((r) => r.id), "夹具前提:不排序时它会被折进去")
      .not.toContain("rd4");

    const fold = foldRunRows(sortRunRows(rows, "rd4"), false);
    expect(fold.rows.map((r) => r.id), "正在跑的那一行被折叠藏起来了").toContain("rd4");
    // ⑶ 成对:其余的仍然收着 —— 否则「有人在跑就整段展开」也能通过。
    expect(fold.hidden, "折叠功能被静默删掉了(什么都没收起来)").toBe(2);
    expect(fold.rows.length).toBe(2);
    // 而它必须在**第一位**(它是这一屏最该看见的东西)。
    expect(fold.rows[0]?.id).toBe("rd4");
  });

  it("★ 没有 run 在跑时,折叠逐字节回到今天的样子", () => {
    // 用户拍板「阈值以内逐像素不变」;这一条把它扩到「没人在跑时,排序也不许改变可见的那两行」。
    const rows = unsorted();
    const fold = foldRunRows(sortRunRows(rows, null), false);
    expect(fold.rows.map((r) => r.id)).toEqual(["ra1", "rb2"]);
    expect(fold.hidden).toBe(2);
  });
});

describe("sortRunRows(§E2E-M25 ⑷:正在跑的置顶,其余按名字)", () => {
  /** ⛔ 名字序与 id 序**故意相反**:全部同名(现成的 `runs(n)` 工装 `modelName` 恒 `""`)时,
   *  「按名字排」是一个**恒等映射**,`rows => rows` 也能通过。这份语料是让那件事有分辨力的
   *  最小形状 —— 而它正是 S143 侦察的对抗核验者点名的那一格。 */
  const zoo = () => [
    mkRun("ra1", { modelName: "zeta" }),
    mkRun("rb2", { modelName: "alpha" }),
    mkRun("rc3", { modelName: "mid" }),
  ];

  it("★★ 按名字排 —— 而这份语料的名字序与 id 序正好相反", () => {
    // 期望是**手写的字面数组**,不是拿实现的比较器算出来的(S108:对拍的两边必须有一边
    // 是我改不动的东西)。
    expect(sortRunRows(zoo(), null).map((r) => r.id)).toEqual(["rb2", "rc3", "ra1"]);
  });

  it("★★ 正在跑的置顶 —— 而它的名字排在**最后**", () => {
    // ⛔ 承重:若夹具里 live run 的名字本来就排第一,「置顶」与「按名字」给出同一个答案,
    //    一个**完全没有置顶逻辑**的实现照样通过。
    expect(sortRunRows(zoo(), "ra1").map((r) => r.id)).toEqual(["ra1", "rb2", "rc3"]);
    // …反方向也要有一格能杀:一个只置顶、不排序的实现在这里会给 ["rb2","ra1","rc3"]。
    expect(sortRunRows(zoo(), "rb2").map((r) => r.id)).toEqual(["rb2", "rc3", "ra1"]);
  });

  it("★★ 未迁移槽:`\"\"` 是一个合法 id,置顶要认它", () => {
    // 真值判断(`liveRunId && a.id === liveRunId`)在这一格上会静默失效。
    const rows = [mkRun("ra1", { modelName: "zeta" }), mkRun("", { modelName: "zzz" })];
    expect(sortRunRows(rows, "").map((r) => r.id)).toEqual(["", "ra1"]);
    // 而 `null`(没人在跑)时它就是一条普通的行,按名字走。
    expect(sortRunRows(rows, null).map((r) => r.id)).toEqual(["ra1", ""]);
  });

  it("★ 没起过名的排在后面(它还没练完过,不是用户在找的东西)", () => {
    const rows = [mkRun("ra1"), mkRun("rb2", { modelName: "alpha" }), mkRun("rc3")];
    expect(sortRunRows(rows, null).map((r) => r.id)).toEqual(["rb2", "ra1", "rc3"]);
  });

  it("★ 同名时按 id —— 同名是**可达状态**(改名那条路今天没有同名闸)", () => {
    // ⛔ 没有这条 tie-break,两条同名 run 的先后就靠 `Array#sort` 的稳定性,
    //    而那是一条没人声明过的巧合;输入顺序一变答案就变。
    const a = [mkRun("rz9", { modelName: "same" }), mkRun("ra1", { modelName: "same" })];
    const b = [mkRun("ra1", { modelName: "same" }), mkRun("rz9", { modelName: "same" })];
    expect(sortRunRows(a, null).map((r) => r.id)).toEqual(["ra1", "rz9"]);
    expect(sortRunRows(b, null).map((r) => r.id)).toEqual(["ra1", "rz9"]);
  });

  it("★ 数字按人读的方式排(run2 在 run10 前面)", () => {
    const rows = [mkRun("ra1", { modelName: "run10" }), mkRun("rb2", { modelName: "run2" })];
    expect(sortRunRows(rows, null).map((r) => r.id)).toEqual(["rb2", "ra1"]);
  });

  it("★★ 不改动传进来的数组 —— 断言的是**顺序**,不是长度", () => {
    // ⛔ 这一条是全场最容易写成装饰件的:`expect(src.length)` 对原地 `sort()` 完全瞎,
    //    而原地排会同时污染 `pickDiffHost` 读的那一份与 store 里那一份。
    const src = zoo();
    const before = src.map((r) => r.id);
    const out = sortRunRows(src, "ra1");
    expect(src.map((r) => r.id), "入参被原地排序了").toEqual(before);
    expect(out).not.toBe(src);
    // …而且返回的是**同一批对象**(不是复制品):行的身份要能被 `=== r.id` 之外的东西认出来。
    expect(new Set(out).size).toBe(3);
    for (const r of out) expect(src).toContain(r);
  });

  it("★★★ 把排过序的行喂给 `pickDiffHost` **会**换掉浅扩散的宿主 —— 这条钉的是那个危险是真的", () => {
    // ⛔⛔ 这一条第一版我写的是「排序不改变 pickDiffHost 的答案」,而**它当场红了,红得对**:
    //    排序确实会换掉宿主(ra1 → rc3)。⇒ 那个期望是我自己编的(S128 §4⒞ 那一族)。
    //
    //    真正该钉的是**反过来的那件事**:这个危险是**真的**,所以
    //    `rowIdentityWiring` 里那道「`pickDiffHost(` 的实参必须是后端原序那一份」的源码闸
    //    **不是装饰**。一条声称「两者同解」的判据反而会给人一个安全的错觉,而且它在
    //    `withMainProgress <= 1` 的夹具上恒真 —— 那正是本仓反复买过的空判据形状。
    //
    //    ⚠ 后果不是观感:宿主决定**浅扩散训练写进哪个 run 目录**、用谁的名字
    //    (`askRunName` → `hps.name` → `weights/<slug>*`)、还原谁的表单(`formForSlot` 的
    //    `k_step_max`/`aug`)—— 三条都要几小时后才看得出来,而全程无声。
    const rows = [
      mkRun("ra1", { modelName: "zeta", info: { has_main_progress: true } }),
      mkRun("rb2", { modelName: "alpha" }),
      mkRun("rc3", { modelName: "mid", info: { has_main_progress: true } }),
    ];
    const raw = pickDiffHost(mkSlot(rows));
    expect(raw.withMainProgress, "夹具前提:必须两个都有主模型,一个的话 find 与 [0] 同解").toBe(2);
    expect(raw.host?.id, "夹具前提:后端原序下宿主是 ra1").toBe("ra1");

    const sorted = pickDiffHost(mkSlot(sortRunRows(rows, "rb2")));
    expect(
      sorted.host?.id,
      "排序后喂进去竟然给同一个宿主 —— 那条源码闸就成了装饰件,这份夹具失去了分辨力",
    ).not.toBe(raw.host?.id);

    // …而只要**不**把排序结果喂给它(今天的接线),它逐位不变。这一半是那道源码闸守的。
    expect(pickDiffHost(mkSlot(rows)).host?.id).toBe("ra1");
    expect(rows.map((r) => r.id), "排序把入参改了 —— 那会静默污染这一份").toEqual([
      "ra1",
      "rb2",
      "rc3",
    ]);
  });
});

describe("newRunNameProblem 的『排除自己』那一档(§E2E-M25 笔 5 · 改名)", () => {
  /** ⛔ 这一档是给**改名**用的:不排除自己的话,「把名字改成它自己已有的那个」会被自己挡住。
   *  而夹具必须**两个 run、名字互不相同** —— 每槽一个 run 时,「跳过自己」与「谁也不跳」
   *  给出同一个答案,那半个条件结构上不可见。 */
  const two = () => [
    mkRun("ra1", { modelName: "初号机" }),
    mkRun("rb2", { modelName: "零号机" }),
  ];

  it("★ 改成自己已有的名字 ⇒ 放行;撞上兄弟 ⇒ 拒", () => {
    expect(newRunNameProblem("初号机", two(), "ra1")).toBeNull();
    expect(newRunNameProblem("初号机", two(), "rb2")).toBe("taken");
    // 不传第三个参数(「再训一个」那条路)⇒ 谁的名字都不许撞,包括这两个。
    expect(newRunNameProblem("初号机", two())).toBe("taken");
  });

  it("★★ `\"\"` 是一个合法 run id —— 拿它当「不排除」会让未迁移槽的唯一那一行被跳过", () => {
    // 未迁移槽:唯一那一行的 id 就是空串。改它自己的名字必须放行。
    const rows = [mkRun("", { modelName: "初号机" })];
    expect(newRunNameProblem("初号机", rows, "")).toBeNull();
    // …而 `null`(不排除任何人)时它就是一个占用中的名字。
    expect(newRunNameProblem("初号机", rows, null)).toBe("taken");
    expect(newRunNameProblem("初号机", rows)).toBe("taken");
  });

  it("★ 两边都 trim(S141 §E2E-M24 买到的那条)", () => {
    expect(newRunNameProblem("  初号机  ", two(), "rb2")).toBe("taken");
    expect(newRunNameProblem("  初号机  ", two(), "ra1")).toBeNull();
  });
});

describe("newRunNameProblem(§E2E-M24)", () => {
  const named = (id: string, modelName: string) => mkRun(id, { modelName });

  it("★ 与同槽已有 run 重名必须被拒 —— 那不是不方便,是产物前缀撞车", () => {
    // 名字 ⇒ slug ⇒ `weights/<slug>*` 与 `audition/<slug>_*`。两个 run 同名之后,
    // `plan_cleanup` 的 `installed_stem` 按 file_stem 判「还装着」,会把**另一个** run 的
    // 快照也判成 StillInstalled 而永久保留。
    const runs = [named("raaa", "歌姫"), named("rbbb", "歌姫2")];
    expect(newRunNameProblem("歌姫", runs)).toBe("taken");
    expect(newRunNameProblem("歌姫3", runs)).toBe(null);
  });

  it("★ 只差首尾空格的名字**不是**新名字(落库写的就是 trim 之后的串)", () => {
    // ⛔ 两边都要 trim:只 trim 一边的话 ` 歌姫 ` 会顺利通过,然后落成 `歌姫` —— 撞车照旧,
    //    而对话框亲口说过没问题。
    expect(newRunNameProblem("  歌姫  ", [named("raaa", "歌姫")])).toBe("taken");
    expect(newRunNameProblem("歌姫", [named("raaa", "  歌姫  ")])).toBe("taken");
  });

  it("空 / 全空白 ⇒ empty,而且它与 taken 是两条不同的路", () => {
    expect(newRunNameProblem("", [])).toBe("empty");
    expect(newRunNameProblem("   ", [])).toBe("empty");
    // 两档必须分得开:文案不同(一条说「名字不能为空」,一条说「这个名字已经被占了」)
    expect(newRunNameProblem("x", [named("raaa", "x")])).toBe("taken");
  });

  it("没有名字的 run(铸了但没练成)不占名额", () => {
    expect(newRunNameProblem("歌姫", [mkRun("raaa"), named("rbbb", "别的")])).toBe(null);
  });

  it("空槽 ⇒ 任何非空名字都行", () => {
    expect(newRunNameProblem("歌姫", [])).toBe(null);
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

      // ⚠★S144 诚实边界:这两条对拍钉的是**默认那一档**(`liveRunId` 缺省 = `null`)。
      //   S144 给两个函数追加的「正在跑的那一行永远画」是**新行为**,搬家前的表达式里没有它,
      //   所以它结构上在这条差分的**覆盖之外** —— 它的判据是上面那两条专门的用例。
      //   ⛔ 别为了「让差分也盖住它」去改 `legacy*`:那会把纯提取的证据变成自证(sortRunRows
      //   头注的硬约束 1 是同一条)。
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
