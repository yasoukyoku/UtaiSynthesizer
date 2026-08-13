/**
 * `liveRun.ts` 的判据(§E2E-M25 笔 1)。
 *
 * ⛔⛔ 这个文件里**每一格夹具都是承重的**,而且它们不是「多写几条更保险」——
 * 每一格恰好对应一种「实现写错了、而全部其余夹具都答同一个答案」的形状:
 *
 * | 夹具 | 它是唯一能杀掉的那个错法 |
 * |---|---|
 * | `state: "completed"` 而 run_id 仍指着这一行 | `state !== "idle"`(仓里有两份现成抄本会诱人这么写)⇒ 跑完之后按钮永久禁死 |
 * | 另一个项目在跑 | 忘了比 `project_id` ⇒ 给别人的 run 贴徽章 |
 * | `backend: "sovits_diff"` 而 family = `sovits` | 写成 `snap.backend === family` —— 它在**所有单 backend 的夹具上与正确写法同值** |
 * | `run_id: ""` 且在跑(未迁移槽) | 把空串当「没有」⇒ 那一行永远/从不被认出来 |
 * | 两个 run | 每槽恒一个 run 时,「是这一行」与「有 run 在跑」返回同一个答案 |
 * | 试听在飞而训练没在跑 | 删除跟错了谓词(跟成「有训练在跑」)⇒ 最常见的那一格仍然白点一次 |
 *
 * ⚠ 这几格全部来自 S143 侦察的**对抗核验者**,而不是我写先验时想到的 —— 先验里那份判据
 * 会在「每槽一个 run、backend 恒等于 family、project 恒等于本页」的语料上全绿。
 */

import { describe, expect, it } from "vitest";
import {
  diffusionIsLive,
  isRunningState,
  liveRunIdFor,
  runRowActions,
  trainingIsLive,
  type LiveFacts,
} from "./liveRun";

const PID = "p_11111111";
const snap = (over: Partial<LiveFacts> = {}): LiveFacts => ({
  state: "running",
  project_id: PID,
  backend: "sovits",
  run_id: "ra11111111a1",
  ...over,
});

describe("isRunningState / trainingIsLive(全局、跨项目)", () => {
  it("★ 只有 starting 与 running 算在跑 —— `!== \"idle\"` 会把跑完的算进来", () => {
    expect(isRunningState("starting")).toBe(true);
    expect(isRunningState("running")).toBe(true);
    // ⛔ 这四格是那条错法的全部落点:快照不在训练结束时清(只有「清空结果」清),
    //    所以这些状态下 workspace/run_id 仍然指着刚练完的那一行。
    for (const s of ["idle", "completed", "stopped", "error"]) {
      expect(isRunningState(s), `${s} 不是「正在跑」`).toBe(false);
    }
  });

  it("★ 跨项目:别的项目在训练时,这里一样是「有训练在跑」", () => {
    // 后端的训练互斥是进程级单槽 ⇒ 前端这一档**不许**按项目过滤。
    expect(trainingIsLive(snap({ project_id: "p_99999999" }), false)).toBe(true);
    expect(trainingIsLive(snap({ project_id: "p_99999999", backend: "rvc" }), false)).toBe(true);
  });

  it("★★ 起训那段窗口:后端已经在拒了,而快照还没说话", () => {
    // `running` 的 compare_exchange 在 `state = "starting"` **450 行之前**就置位,中间还有
    // 真搬目录的 `migrate_one_slot` ⇒ 只看 state 的禁用在这段窗口里是开着的。
    expect(trainingIsLive(snap({ state: "idle" }), true)).toBe(true);
    expect(trainingIsLive(snap({ state: "completed" }), true)).toBe(true);
    // 而没有在飞的 start 时,completed 就是 completed。
    expect(trainingIsLive(snap({ state: "completed" }), false)).toBe(false);
  });
});

describe("liveRunIdFor(本项目、本槽、逐行)", () => {
  it("正常一格:本项目本槽在跑 ⇒ 答出那个 run id", () => {
    expect(liveRunIdFor(snap(), PID, "sovits")).toBe("ra11111111a1");
  });

  it("★★ 跑完了不算 —— 而 run_id 还指着这一行", () => {
    // 这一格是 `state !== "idle"` 那个错法的唯一看守。
    for (const s of ["completed", "stopped", "error", "idle"]) {
      expect(liveRunIdFor(snap({ state: s }), PID, "sovits"), `state=${s}`).toBeNull();
    }
  });

  it("★★ 别的项目在跑 ⇒ 本页的行一个都不算(与 trainingIsLive 正好相反)", () => {
    expect(liveRunIdFor(snap({ project_id: "p_99999999" }), PID, "sovits")).toBeNull();
    // …而同一份快照在全局那个谓词上是 true。两条规矩必须在**同一份输入**上给出不同答案,
    // 否则「合成一个谓词」这件事在判据上是看不见的。
    expect(trainingIsLive(snap({ project_id: "p_99999999" }), false)).toBe(true);
  });

  it("★★ 浅扩散跑在 sovits 槽里 —— `snap.backend === family` 会在这里答错", () => {
    // `sovits_diff` 不是一个 family:它的产物写在 sovits 槽的那个 run 目录下。
    expect(liveRunIdFor(snap({ backend: "sovits_diff" }), PID, "sovits")).toBe("ra11111111a1");
    // 而它当然不该点亮别的槽。
    expect(liveRunIdFor(snap({ backend: "sovits_diff" }), PID, "rvc")).toBeNull();
    expect(liveRunIdFor(snap({ backend: "sovits_diff" }), PID, "sovits_v2")).toBeNull();
  });

  it("不是这个槽 ⇒ null(而不是「随便哪个槽都算」)", () => {
    expect(liveRunIdFor(snap({ backend: "rvc" }), PID, "sovits")).toBeNull();
    expect(liveRunIdFor(snap({ backend: "rvc" }), PID, "rvc")).toBe("ra11111111a1");
  });

  it("★★ 未迁移槽:run_id 是空串,而那是【肯定事实】不是「没有」", () => {
    // 槽根就是那个 run(layout ≤2),`RunDetail.id` 给那一行的值也正是 `""`。
    expect(liveRunIdFor(snap({ run_id: "" }), PID, "sovits")).toBe("");
    // ⛔ 而它必须与「没有 run 在跑」分得开 —— 后者是 null,不是 ""。
    expect(liveRunIdFor(snap({ run_id: "", state: "idle" }), PID, "sovits")).toBeNull();
    expect(liveRunIdFor(snap({ run_id: "", state: "completed" }), PID, "sovits")).toBeNull();
  });

  it("★ 两个 run 的槽里,只有一行对得上", () => {
    // ⛔ 每槽恒一个 run 时,「是这一行」与「有 run 在跑」返回同一个答案 ——
    //    这一格是让那两件事分开的最小形状(`rowIdentity.ts` 头注为同一条付过账)。
    const live = liveRunIdFor(snap({ run_id: "rb22222222b2" }), PID, "sovits");
    const rows = ["ra11111111a1", "rb22222222b2"];
    expect(rows.map((id) => id === live)).toEqual([false, true]);
  });
});

describe("diffusionIsLive(浅扩散不是一个 family,要单独问)", () => {
  it("★★ 主模型在练**不算**浅扩散在练 —— 而它们跑在同一个 run 目录里", () => {
    // ⛔ 这一格是这个函数存在的全部理由:`liveRunIdFor(..., "sovits")` 对两者答案相同
    //    (浅扩散就住在那个 sovits run 里),而两张卡是两张卡 —— 主模型在练时给浅扩散卡
    //    贴「训练中」是一句假话。
    expect(diffusionIsLive(snap({ backend: "sovits_diff" }), PID)).toBe(true);
    expect(diffusionIsLive(snap({ backend: "sovits" }), PID)).toBe(false);
    // …而槽那一层对两者都答「有 run 在跑」,这正是分不开的那一半。
    expect(liveRunIdFor(snap({ backend: "sovits_diff" }), PID, "sovits")).not.toBeNull();
    expect(liveRunIdFor(snap({ backend: "sovits" }), PID, "sovits")).not.toBeNull();
  });

  it("别的项目 / 跑完了 ⇒ 不算", () => {
    expect(diffusionIsLive(snap({ backend: "sovits_diff", project_id: "p_9" }), PID)).toBe(false);
    expect(diffusionIsLive(snap({ backend: "sovits_diff", state: "completed" }), PID)).toBe(false);
  });
});

describe("runRowActions(四颗按钮跟的不是同一个谓词)", () => {
  const base = {
    blocked: false,
    inFlight: false,
    trainingLive: false,
    anyTaskLive: false,
    hasRunId: true,
  };

  it("闲着的时候四颗全能点", () => {
    const g = runRowActions(base);
    expect([g.cont.disabled, g.retrain.disabled, g.del.disabled, g.rename.disabled]).toEqual([
      false,
      false,
      false,
      false,
    ]);
    expect([g.cont.reason, g.del.reason]).toEqual(["", ""]);
  });

  it("★★ 有训练在跑:继续/再训禁掉,而删除/改名跟的是**更宽**的那个谓词", () => {
    // 训练也是一种长任务,所以真实世界里两个标志会一起为真;这里故意拆开,
    // 是为了让「删除跟错了谓词」这个错法有一格能杀。
    const g = runRowActions({ ...base, trainingLive: true, anyTaskLive: true });
    expect(g.cont.reason).toBe("training");
    expect(g.retrain.reason).toBe("training");
    expect(g.del.reason).toBe("tasks");
    expect(g.rename.reason).toBe("tasks");
  });

  it("★★ 试听/渲染/分离在飞而训练没在跑 —— 删除与改名仍然要禁", () => {
    // ⛔ 这一格是「删除只禁正在跑的那一行」那个做法的唯一看守:后端 `running_tasks_of` 把
    //    training / separation / render / audition / 全部 active_tasks 一起算,而试听恰恰是
    //    同一个训练页发起的。禁得比后端窄 = 用户在最常见的那一格上照样白点一次。
    const g = runRowActions({ ...base, anyTaskLive: true });
    expect(g.del.reason).toBe("tasks");
    expect(g.rename.reason).toBe("tasks");
    // …而这一格**不该**挡住继续训练:后端那道闸(`TRAINING_ALREADY_RUNNING`)问的是
    // 「有没有训练在跑」。禁得比后端严 = 试听一段音频就开不了训。
    // ⚠ 断言消息不是装饰:探针要能分辨「红在这一条」与「红在上面那条」,而注释不进输出。
    expect(g.cont.disabled, "试听在飞时继续训练被挡住了 —— 这一档禁得比后端还严").toBe(false);
    expect(g.retrain.disabled, "试听在飞时再训一个被挡住了 —— 这一档禁得比后端还严").toBe(
      false,
    );
  });

  it("★ 项目被标记时,报的是「被标记」而不是「有训练在跑」", () => {
    // 顺序是承重的:说「有训练在跑」会把人引去等一场永远等不到的训练。
    const g = runRowActions({ ...base, blocked: true, trainingLive: true, anyTaskLive: true });
    expect([g.cont.reason, g.retrain.reason, g.del.reason, g.rename.reason]).toEqual([
      "flagged",
      "flagged",
      "flagged",
      "flagged",
    ]);
  });

  it("本页自己那次 invoke 在飞 ⇒ 只挡改动类,不挡开训", () => {
    const g = runRowActions({ ...base, inFlight: true });
    expect(g.del.reason).toBe("inflight");
    expect(g.rename.reason).toBe("inflight");
    expect(g.cont.disabled).toBe(false);
  });

  it("★ 未迁移槽那一行没有 run id ⇒ 删除如实答「不可用」(画不画由调用点决定)", () => {
    const g = runRowActions({ ...base, hasRunId: false });
    expect(g.del.disabled).toBe(true);
    // 而它**不该**因此挡住改名或继续训练 —— 未迁移槽照样改得了名、续得了训。
    expect(g.rename.disabled).toBe(false);
    expect(g.cont.disabled).toBe(false);
  });

  it("★★ 语料自检:上面这些用例真的走到过四种不同的 reason", () => {
    // ⛔ 一个把 reason 全部塌成 `""` / 全部塌成同一个串的实现,如果没有这一条,
    //    只靠 `disabled` 那几条断言是活得下去的(它们大多只看布尔)。
    const seen = new Set<string>();
    for (const over of [
      {},
      { blocked: true },
      { trainingLive: true },
      { anyTaskLive: true },
      { inFlight: true },
    ]) {
      const g = runRowActions({ ...base, ...over });
      for (const k of [g.cont, g.retrain, g.del, g.rename]) seen.add(k.reason);
    }
    expect(seen, "四种 reason 没有全部被走到 —— 上面的用例覆盖不到某一档").toEqual(
      new Set(["", "flagged", "training", "tasks", "inflight"]),
    );
  });
});
