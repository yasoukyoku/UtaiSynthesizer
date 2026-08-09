/**
 * §E2E-M1 的前端一半 —— 「装出来的模型带不带检索矩阵」这条**静默**链的决策层。
 *
 * ⛔ 这些用例存在的理由不是覆盖率:它们全都是**耳朵判不了**的事实。缺检索矩阵的症状是音色
 * 相似度下降,没有 A/B 判不出来,判出来也归因不到索引上 —— 而它在盘上不过是一个文件在不在。
 * 用实机窗口「看一眼」查它,等于没有判据(见 memory 的 feedback_test_in_tauri_window 头一节)。
 *
 * ★ 每一条都写了它**对应哪一种真实故障**。探针是注入的,所以这里的每个 `exists` 回答都是我
 * 声明的事实,而不是从被测代码里回读出来的(自证)。
 */
import { describe, it, expect } from "vitest";
import {
  resolveArchiveIndex,
  indexPathArg,
  indexWarningCode,
  type IndexResolution,
} from "./indexPath";

/** 一个只认识给定路径集合的探针。**大小写与分隔符原样比对** —— 生产代码拼的就是反斜杠,
 *  在这里「顺手规范化」会让判据比真实拼接更宽容,而宽容正是这条链出问题的地方。 */
const probe = (present: string[]) => {
  const seen: string[] = [];
  const exists = async (p: string) => {
    seen.push(p);
    return present.includes(p);
  };
  return { exists, seen };
};

const RUN = "D:\\data\\training\\p_1\\rvc\\runs\\rfeedfacefeed";

describe("存档导入的索引解析", () => {
  it("上下文给了明确路径 ⇒ 用它,而且不再探测", async () => {
    const { exists, seen } = probe([]);
    const r = await resolveArchiveIndex({
      backend: "rvc",
      workspace: RUN,
      ctxIndexPath: `${RUN}\\total_fea.npy`,
      exists,
    });
    expect(r).toEqual({ kind: "explicit", path: `${RUN}\\total_fea.npy` });
    // 指名了就是指名了:再去猜是另一种静默错配(把别的 run 的索引装给这个模型)
    expect(seen).toEqual([]);
  });

  it("★明确路径压过一切:工作区未知也照用(有路径就不需要工作区)", async () => {
    // 变异探针买回来的:原来所有「有明确路径」的用例里工作区都是非空的,于是判据**分不开**
    // 「指名优先」与「指名 + 工作区都要有才算数」。这条把语义钉死 —— 已经拿到路径了,
    // 工作区拿没拿到与它无关。
    const { exists, seen } = probe([]);
    const r = await resolveArchiveIndex({
      backend: "rvc",
      workspace: "",
      ctxIndexPath: "D:\\somewhere\\total_fea.npy",
      exists,
    });
    expect(r).toEqual({ kind: "explicit", path: "D:\\somewhere\\total_fea.npy" });
    expect(indexWarningCode(r)).toBeNull();
    expect(seen).toEqual([]);
  });

  it("实时 run 的 summary.index 压过上下文", async () => {
    const { exists } = probe([]);
    const r = await resolveArchiveIndex({
      backend: "rvc",
      workspace: RUN,
      summaryIndex: "D:\\live\\total_fea.npy",
      ctxIndexPath: `${RUN}\\total_fea.npy`,
      exists,
    });
    expect(r).toEqual({ kind: "explicit", path: "D:\\live\\total_fea.npy" });
  });

  it("没有明确路径但工作区里探得到 ⇒ probed", async () => {
    const { exists, seen } = probe([`${RUN}\\total_fea.npy`]);
    const r = await resolveArchiveIndex({ backend: "rvc", workspace: RUN, exists });
    expect(r).toEqual({ kind: "probed", path: `${RUN}\\total_fea.npy` });
    expect(seen).toEqual([`${RUN}\\total_fea.npy`]);
  });

  it("工作区里确实没有 ⇒ none(这是正常状态,不该打扰用户)", async () => {
    const { exists } = probe([]);
    expect(await resolveArchiveIndex({ backend: "rvc", workspace: RUN, exists })).toEqual({
      kind: "none",
    });
  });

  it("sovits 家族按 kmeans 优先、再退到检索向量", async () => {
    const S = "D:\\data\\training\\p_1\\sovits\\runs\\rcafebabecafe";
    const both = probe([`${S}\\cluster\\kmeans_10000.pt`, `${S}\\cluster\\0.index_vectors.npy`]);
    expect(await resolveArchiveIndex({ backend: "sovits", workspace: S, exists: both.exists })).toEqual(
      { kind: "probed", path: `${S}\\cluster\\kmeans_10000.pt` },
    );
    const onlyVectors = probe([`${S}\\cluster\\0.index_vectors.npy`]);
    expect(
      await resolveArchiveIndex({ backend: "sovits", workspace: S, exists: onlyVectors.exists }),
    ).toEqual({ kind: "probed", path: `${S}\\cluster\\0.index_vectors.npy` });
  });

  it("声码器从不探测(探到的只会是别的后端的遗留)", async () => {
    const { exists, seen } = probe([`${RUN}\\total_fea.npy`]);
    expect(await resolveArchiveIndex({ backend: "vocoder", workspace: RUN, exists })).toEqual({
      kind: "none",
    });
    expect(seen).toEqual([]);
  });

  // ── ★★ 这条是 M1 的正主:S126 修掉的那条静默链的**判据** ────────────────────────────
  it("⛔工作区未知(上下文没拿到)⇒ 说【不知道】,不说【没有】,而且一次都不探测", async () => {
    const { exists, seen } = probe([]);
    const r = await resolveArchiveIndex({ backend: "rvc", workspace: "", exists });
    expect(r).toEqual({ kind: "unknownWorkspace" });
    // 旧代码会去探 `\total_fea.npy` —— 在 Windows 上那是**当前盘符的根**
    expect(seen).toEqual([]);
    // 而两者的处置**完全不同**:
    expect(indexPathArg(r)).toBeUndefined(); // 不许给 Rust 一个编出来的路径
    expect(indexWarningCode(r)).toBe("WARN_INDEX_CONTEXT_UNKNOWN"); // 必须让用户看见
  });

  it("★『不知道』与『没有』必须给出不同的用户可见结果", async () => {
    const { exists } = probe([]);
    const unknown = await resolveArchiveIndex({ backend: "rvc", workspace: "", exists });
    const none = await resolveArchiveIndex({ backend: "rvc", workspace: RUN, exists });
    // 这一条如果红了,就是两种状态又被合并成了一个 `undefined` —— 正是 S126 之前的形状
    expect(indexWarningCode(unknown)).not.toBe(indexWarningCode(none));
    expect(indexWarningCode(none)).toBeNull();
  });

  it("给 Rust 的实参只在真的知道路径时才有值", async () => {
    const cases: [IndexResolution, string | undefined][] = [
      [{ kind: "explicit", path: "a" }, "a"],
      [{ kind: "probed", path: "b" }, "b"],
      [{ kind: "none" }, undefined],
      [{ kind: "unknownWorkspace" }, undefined],
    ];
    for (const [r, want] of cases) expect(indexPathArg(r)).toBe(want);
  });
});
