/**
 * §F2⒝ 批 2 ④d 笔 0 —— 「继续训练」把表单放回槽在的地方,**包括决定用哪个预处理池的那些值**。
 *
 * ⛔ 这个文件存在的第一个理由是结构性的:`resume_lock.rs` 的模块头把「项目页的表单还原」列为
 * 必须与守卫一致的四处之一,而它此前是一个组件内闭包 —— vitest 不做组件测试,所以那一处
 * **没有任何东西驱动得动**。把它抽成纯函数就是为了让这里存在。
 *
 * ⛔ 第二个理由是自证陷阱:一条「还原了 loudnorm」的断言,**单独**会被一个恒返回 `true` 的
 * 实现满足;一条「不知道时不还原」的断言,单独会被一个**什么都不返回**的实现满足。所以
 * 三态(true / false / 未知)是**成对**断言的,少任何一条,剩下的都能被常量骗过。
 */

import { describe, expect, it } from "vitest";
import { formForSlot, NO_FORM_CONTROL, POOL_FORM_FIELDS } from "./formForSlot";
import { lockedFieldIds } from "../resumeLock";
import type { TrainingBackend, TrainingFormConfig, WorkspaceInfo } from "../../store/training";

/** 一个**跑过**的槽。`version` 非空 = manifest 存在(它写在 worker 开始预处理之前)。 */
function ws(over: Partial<WorkspaceInfo> = {}): WorkspaceInfo {
  return {
    exists: true,
    family: "sovits",
    version: "4.1",
    sample_rate: "44k",
    has_main_progress: true,
    diff_steps: 0,
    best_resume_step: null,
    diff_best_resume_step: null,
    aug_copies: 0,
    loudnorm: null,
    has_dataset: true,
    // 一个**跑过**的槽必然已经有池 —— 这份夹具的主题就是「盘上有东西回答」。
    has_preprocessing: true,
    vol_embedding: null,
    n_speakers: 1,
    speakers: [],
    diff_k_step_max: 0,
    ...over,
  };
}

const ALL_BACKENDS: TrainingBackend[] = ["rvc", "sovits", "sovits_v2", "sovits_diff", "vocoder"];

describe("formForSlot —— costly 档(决定用哪个预处理池的值)", () => {
  it("sovits:还原 loudnorm 与增强份数,而不是把它们打回出厂默认", () => {
    const got = formForSlot("sovits", ws({ loudnorm: true, aug_copies: 2 }));
    expect(got.sovitsLoudnorm).toBe(true);
    expect(got.sovitsAugCopies).toBe(2);
  });

  // ★ 这一对必须同时存在。只有上面那条,一个 `sovitsLoudnorm: true` 的常量实现就能全绿;
  //   只有下面那条,一个「什么都不还原」的实现也能全绿。分开它们的是**同一个字段的两个值**。
  it("sovits:槽是【关着响度归一化】建的 ⇒ 复选框放回关,而不是留着用户的开", () => {
    const got = formForSlot("sovits", ws({ loudnorm: false }));
    expect(got).toHaveProperty("sovitsLoudnorm");
    expect(got.sovitsLoudnorm).toBe(false);
  });

  it("★盘上没有东西回答(loudnorm=null)⇒ 一个字都不许写,用户当前的值不动", () => {
    const got = formForSlot("sovits", ws({ loudnorm: null }));
    expect(got).not.toHaveProperty("sovitsLoudnorm");
  });

  it("★槽从没跑过(manifest 不存在)⇒ 一个 costly 字段都不还原", () => {
    // 阴性对照:同一份 info 只把 version/sample_rate 抹掉。若实现改成「无条件还原」,
    // 它会把用户刚在参数页填好的 aug=3 清成 0 —— 这条就是钉那件事的。
    const never = ws({ version: "", sample_rate: "", aug_copies: 0, loudnorm: false });
    const got = formForSlot("sovits", never);
    expect(got).not.toHaveProperty("sovitsAugCopies");
    expect(got).not.toHaveProperty("sovitsLoudnorm");
  });

  it("sovits_v2:同样送 loudnorm、同样折进 fp_text ⇒ 同样要还原(此前这条返回空对象)", () => {
    const got = formForSlot("sovits_v2", ws({ family: "sovits_v2", version: "4.0-v2", loudnorm: true, aug_copies: 1 }));
    expect(got.sovitsLoudnorm).toBe(true);
    expect(got.sovitsAugCopies).toBe(1);
  });

  it("vocoder:还原它自己那个增强字段(此前这条也返回空对象)", () => {
    const got = formForSlot("vocoder", ws({ family: "vocoder", version: "nsf_hifigan", aug_copies: 3 }));
    expect(got.vocAugCopies).toBe(3);
    // 阴性对照:别把它写进 sovits 那组共享字段里去
    expect(got).not.toHaveProperty("sovitsAugCopies");
  });

  it("rvc:costly 与 locked 一起还原,互不吞并", () => {
    const got = formForSlot("rvc", ws({ family: "rvc", version: "v2", sample_rate: "40k", aug_copies: 2 }));
    expect(got).toMatchObject({ version: "v2", sampleRate: "40k", augCopies: 2 });
  });

  it("sovits_diff:diff-first(槽里没有主模型)时份数是这个 run 自己的选择 ⇒ 还原", () => {
    const got = formForSlot("sovits_diff", ws({ has_main_progress: false, aug_copies: 2 }));
    expect(got.diffAugCopies).toBe(2);
  });

  it("★sovits_diff:有宿主主模型时份数是【继承】的,参数页连输入框都不渲染 ⇒ 不许回填", () => {
    const got = formForSlot("sovits_diff", ws({ has_main_progress: true, aug_copies: 2 }));
    expect(got).not.toHaveProperty("diffAugCopies");
  });
});

describe("formForSlot —— locked 档(不还原就会被 Rust 拒的值)", () => {
  it("sovits:版本与响度嵌入", () => {
    const got = formForSlot("sovits", ws({ version: "4.1", vol_embedding: false }));
    expect(got).toMatchObject({ sovitsVersion: "4.1", sovitsVolEmbedding: false });
  });

  it("sovits:4.0 的槽不许被默认的 4.1 顶掉", () => {
    expect(formForSlot("sovits", ws({ version: "4.0" })).sovitsVersion).toBe("4.0");
  });

  it("sovits_diff:有扩散进度才还原 kStepMax(与后端 guard 同一个信号)", () => {
    expect(formForSlot("sovits_diff", ws({ diff_steps: 4000, diff_k_step_max: 200 })).diffKStepMax).toBe(200);
    expect(formForSlot("sovits_diff", ws({ diff_steps: 0, diff_k_step_max: 200 }))).not.toHaveProperty(
      "diffKStepMax",
    );
  });

  it("★重训(pin):版本由用户当场选定、响度嵌入不回填,但 costly 那一档【照样】还原", () => {
    const got = formForSlot("sovits", ws({ version: "4.1", vol_embedding: true, loudnorm: true, aug_copies: 2 }), "4.0");
    expect(got.sovitsVersion).toBe("4.0");
    expect(got).not.toHaveProperty("sovitsVolEmbedding");
    // 重训会清空这个槽,所以这两个值不是约束而是**默认配方**——把它们打回出厂默认
    // (响度归一化关、增强 0)从来不是「再练一遍」的意思。
    expect(got).toMatchObject({ sovitsLoudnorm: true, sovitsAugCopies: 2 });
  });

  it("null info(槽还不存在)⇒ 只给出表单必须有的那个版本默认值,不编造别的", () => {
    expect(formForSlot("sovits", null)).toEqual({ sovitsVersion: "4.1" });
    expect(formForSlot("rvc", null)).toEqual({});
    expect(formForSlot("vocoder", undefined)).toEqual({});
  });
});

/**
 * ★★ 棘轮:锁表加一行 `costly` 而没人决定它怎么还原 ⇒ 这里红。
 *
 * 这是本文件最重要的两条。手抄一份「要还原哪些字段」的清单是**没有闸**的 —— 它正是 ④d 之前
 * 那份清单的下场(它只抄了 locked 那一档,而漏掉的一档决定用哪个池)。所以清单必须由锁表
 * **驱动**,并且「声明了」还要被证明「真的还原了」。
 */
describe("★棘轮:锁表 ↔ 表单字段", () => {
  it("每个 backend 的每一条 costly 规则,要么有表单字段,要么明写它为什么没有", () => {
    for (const backend of ALL_BACKENDS) {
      for (const id of lockedFieldIds(backend, "costly")) {
        const declared = id in POOL_FORM_FIELDS[backend] || NO_FORM_CONTROL.has(id);
        expect(declared, `${backend}.${id} 没有落点:给它一个表单字段,或写进 NO_FORM_CONTROL`).toBe(
          true,
        );
      }
    }
  });

  it("反向:声明了表单字段的每一条,必须真的出现在 formForSlot 的返回值里", () => {
    /** id → (给一个能与默认值区分开的 info,期望的值)。加一个新的 costly id 时这里也要加。 */
    const probes: Record<string, { info: Partial<WorkspaceInfo>; expect: unknown }> = {
      loudnorm: { info: { loudnorm: true }, expect: true },
      augCopies: { info: { aug_copies: 3, has_main_progress: false }, expect: 3 },
      // ★S144 —— ⚠ 必须显式写一个**在值域里**的采样率:`ws()` 的默认是 `"44k"`,而
      // `formForSlot` 的 rvc 臂只认 `"32k"|"40k"|"48k"` ⇒ 用默认值探针会得到 `undefined`,
      // 而那条红说的是「声明了却没还原」——一句与被测性质无关的假话。
      sampleRate: { info: { sample_rate: "40k" }, expect: "40k" },
    };
    for (const backend of ALL_BACKENDS) {
      for (const [id, field] of Object.entries(POOL_FORM_FIELDS[backend])) {
        const probe = probes[id];
        expect(probe, `${backend}.${id} 没有探针:加一个,否则这条声明没人验`).toBeTruthy();
        const got = formForSlot(backend, ws(probe!.info)) as Record<string, unknown>;
        expect(got[field as keyof TrainingFormConfig], `${backend}.${id} 声明了 ${field} 却没还原它`).toBe(
          probe!.expect,
        );
      }
    }
  });

  it("NO_FORM_CONTROL 里的每一条都必须真的在某个 backend 的锁表里(别攒死条目)", () => {
    const all = new Set<string>();
    for (const backend of ALL_BACKENDS) for (const id of lockedFieldIds(backend, "costly")) all.add(id);
    for (const id of NO_FORM_CONTROL) expect(all.has(id), `${id} 已经不在锁表里了`).toBe(true);
  });
});
