/**
 * ★★S144(用户实机报的)——「数据这一步满足了吗」这条谓词,**尤其是浅扩散那一格**。
 *
 * ## 为什么这份文件在这一场才出现
 *
 * `trainingDataOk` 有**四个**消费点(项目页的路由 `nextSegFor`、向导的 `step3Ok`、数据页的
 * 「下一步」、运行段的开始守卫),而它此前**一条判据都没有**。于是一条把浅扩散的入口
 * **结构性地焊死**的规则从 S78 活到了 S144,只有实机窗口看得见:
 * 点浅扩散的「开始训练」⇒ 被扔进数据段 ⇒ **导入多少音频都出不来下一步**。
 *
 * 机理:`diffPoolReady` 要求 `info.family === "sovits"`,而那个字段读的是 **run manifest**;
 * 一个**从没训练过主模型**的槽没有 manifest ⇒ 恒 `""` ⇒ 谓词恒假。而它当时是浅扩散那一格的
 * **唯一**分支(`return diffPoolReady(...)`),所以「没有主模型先训扩散」——后端明写受支持的
 * **diff-first** ——在界面上做不到。
 *
 * ⛔ 这里每一条都在钉**两件不同的事**:⑴ 捷径(免导入直训)还在;⑵ 捷径**不成立时**回落到
 * 通用规则,而不是拒绝。少任何一半,这条谓词就会退回它坏掉的那种形状之一。
 */
import { describe, expect, it } from "vitest";
import { trainingDataOk, type DatasetSummary, type WorkspaceInfo } from "./training";

/** 一份**平铺**(单歌手)的项目数据集 —— 用户导入完音频之后的常态。 */
function flatDs(over: Partial<DatasetSummary> = {}): DatasetSummary {
  return {
    files: 12,
    bytes: 1234,
    datasetDir: "D:/x/dataset",
    speakers: [],
    entries: [],
    groups: [],
    orderKnown: true,
    ...over,
  };
}

/** 一个**从没训练过**的 sovits 槽:目录可能在(池/数据),但没有 run manifest ⇒ `family` 是 `""`。 */
function freshSlotInfo(over: Partial<WorkspaceInfo> = {}): WorkspaceInfo {
  return {
    exists: true,
    family: "",
    version: "",
    sample_rate: "",
    has_main_progress: false,
    diff_steps: 0,
    best_resume_step: null,
    diff_best_resume_step: null,
    aug_copies: 0,
    loudnorm: null,
    has_dataset: true,
    has_preprocessing: false,
    vol_embedding: null,
    n_speakers: 1,
    speakers: [],
    diff_k_step_max: 0,
    ...over,
  };
}

/** 宿主槽**已经训练过** sovits ⇒ 捷径(免导入直训)成立的那一格。 */
const trainedHost = freshSlotInfo({ family: "sovits", has_main_progress: true });

describe("trainingDataOk —— 浅扩散(§E2E,S144 实机买回来的)", () => {
  it("★★ diff-first:槽从没训练过,但项目有一份平铺数据集 ⇒ **可以进下一步**", () => {
    // 这一条就是用户报的那个屏幕。⛔ 它此前是 `false`,而四个消费点用的是同一个谓词 ⇒
    // 路由把人扔进数据段、数据段的「下一步」又永远不亮。
    expect(trainingDataOk("sovits_diff", flatDs(), freshSlotInfo())).toBe(true);
    // `diffWsInfo` 还没被写过(探针失败/首次进入)也一样 —— 它不该是**必要**条件
    expect(trainingDataOk("sovits_diff", flatDs(), null)).toBe(true);
  });

  it("★ 捷径仍然在:宿主槽已经训练过 ⇒ **不必**导入任何音频", () => {
    // 免导入直训(S41 共享池)。⛔ 与上一条成对:只有上一条时,一个「把 diffPoolReady 整个
    // 删掉」的实现也全绿;只有这一条时,坏掉的那种形状(捷径当唯一入口)也全绿。
    expect(trainingDataOk("sovits_diff", null, trainedHost)).toBe(true);
    expect(trainingDataOk("sovits_diff", flatDs({ files: 0 }), trainedHost)).toBe(true);
  });

  it("★ 没有数据、也没有可复用的宿主 ⇒ 仍然要先给它数据", () => {
    expect(trainingDataOk("sovits_diff", null, freshSlotInfo())).toBe(false);
    expect(trainingDataOk("sovits_diff", flatDs({ files: 0 }), freshSlotInfo())).toBe(false);
  });

  it("★★ 多歌手数据集 ⇒ 浅扩散不许进(后端会拒,而在这里挡住才不会走完向导才被拒)", () => {
    const multi = flatDs({ speakers: ["a", "b"] });
    expect(trainingDataOk("sovits_diff", multi, freshSlotInfo())).toBe(false);
    // ⚠ 但捷径那一格**照旧**:宿主槽的 `n_speakers <= 1` 是 `diffPoolReady` 自己在管的,
    //   项目数据集长什么样与那条捷径无关。
    expect(trainingDataOk("sovits_diff", multi, trainedHost)).toBe(true);
    // 而宿主自己是多歌手时,捷径必须关掉
    expect(
      trainingDataOk("sovits_diff", multi, freshSlotInfo({ family: "sovits", n_speakers: 2 })),
    ).toBe(false);
  });
});

describe("trainingDataOk —— 另外四个 backend 逐字节不变(S144 的回归网)", () => {
  it("平铺数据集喂得动每一个 backend", () => {
    for (const b of ["rvc", "sovits", "sovits_v2", "vocoder"]) {
      expect(trainingDataOk(b, flatDs(), null), b).toBe(true);
    }
  });

  it("★ 多歌手数据集只喂得动**共训**的那三个 —— 而这条区分不许被浅扩散那一格带跑", () => {
    const multi = flatDs({ speakers: ["a", "b"] });
    for (const b of ["rvc", "sovits", "sovits_v2"]) {
      expect(trainingDataOk(b, multi, null), b).toBe(true);
    }
    // 声码器的 loader 假设单歌手 ⇒ 与浅扩散同一条规则
    expect(trainingDataOk("vocoder", multi, null)).toBe(false);
  });

  it("★ 没有数据就是没有数据 —— 而且**宿主槽的捷径不许溢出到别的 backend**", () => {
    for (const b of ["rvc", "sovits", "sovits_v2", "vocoder"]) {
      expect(trainingDataOk(b, null, null), b).toBe(false);
      // ⛔ 这一格是承重的:`diffPoolReady` 内部第一条就是 `backend === "sovits_diff"`,
      //    把它删掉的话,一个训练过的 sovits 槽会让**每个** backend 都不必导入数据。
      expect(trainingDataOk(b, null, trainedHost), b).toBe(false);
    }
  });
});
