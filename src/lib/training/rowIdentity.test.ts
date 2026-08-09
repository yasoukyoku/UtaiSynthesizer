import { describe, expect, it } from "vitest";
import { resolveRowIdentity } from "./rowIdentity";

/**
 * §F2⒝ 批 2 ④b —— 一行的名字与工作区必须来自**同一次**解析。
 *
 * ⛔ 自证陷阱(这一族在本仓反复出现):今天每槽恒一个 run,所以「实时读数」与「这一行自己那个
 * run 的读数」**是同一个字符串** —— 任何只在 N=1 形状上写的用例,在保护生效与失效两个世界里
 * 返回同一个答案。所以下面每一条要害用例都**造出两个 run**,让两条来源给出**不同的**字符串。
 */

const RUN_A = "C:\\d\\training\\proj\\rvc\\runs\\ra111111111a";
const RUN_B = "C:\\d\\training\\proj\\rvc\\runs\\rb222222222b";

describe("resolveRowIdentity", () => {
  it("takes the live run's readings for a row that structurally has no run id", () => {
    const r = resolveRowIdentity({
      runId: null,
      ctx: { modelName: "槽级旧名", workspace: RUN_A, indexPath: null },
      live: { modelName: "本次训练名", workspace: RUN_A, summaryIndex: "C:\\x\\total_fea.npy" },
      fallbackName: "表单名",
    });
    expect(r).toMatchObject({
      name: "本次训练名",
      workspace: RUN_A,
      summaryIndex: "C:\\x\\total_fea.npy",
      source: "live",
    });
  });

  it("★ a row that names ANOTHER run must not borrow the live run's name, workspace or index", () => {
    const r = resolveRowIdentity({
      runId: "rb222222222b",
      ctx: { modelName: "run B 的名字", workspace: RUN_B, indexPath: null },
      // the live run is A — a different directory, a different name, and its own index product
      live: { modelName: "run A 的名字", workspace: RUN_A, summaryIndex: "C:\\A\\total_fea.npy" },
      fallbackName: "表单名",
    });
    expect(r.name).toBe("run B 的名字");
    expect(r.workspace).toBe(RUN_B);
    // ⛔ the sharp one: run A's index must NOT travel onto run B's import
    expect(r.summaryIndex).toBeUndefined();
    expect(r.source).toBe("run");
  });

  it("★ still uses the live readings when the row's run IS the live run (a positive fact: equal paths)", () => {
    const r = resolveRowIdentity({
      runId: "ra111111111a",
      ctx: { modelName: "盘上写的名字", workspace: RUN_A, indexPath: null },
      live: { modelName: "本次训练名", workspace: RUN_A, summaryIndex: "C:\\A\\total_fea.npy" },
      fallbackName: "表单名",
    });
    // ⚠ this and the case above differ ONLY in whether the two workspaces are equal — that is
    // the whole predicate, and a rule keyed on "does the row have a run id" cannot tell them apart
    expect(r.source).toBe("live");
    expect(r.name).toBe("本次训练名");
    expect(r.summaryIndex).toBe("C:\\A\\total_fea.npy");
  });

  it("does not treat the live run as this row when the live workspace is unknown", () => {
    const r = resolveRowIdentity({
      runId: "rb222222222b",
      ctx: { modelName: "run B", workspace: RUN_B, indexPath: null },
      live: { modelName: "运行中", workspace: "", summaryIndex: "C:\\A\\total_fea.npy" },
      fallbackName: "表单名",
    });
    expect(r.source).toBe("run");
    expect(r.summaryIndex).toBeUndefined();
    expect(r.workspace).toBe(RUN_B);
  });

  it("falls back to the slot context in the standalone archive view (no live identity)", () => {
    const r = resolveRowIdentity({
      runId: null,
      ctx: { modelName: "槽里冻结的名字", workspace: RUN_A, indexPath: "C:\\A\\total_fea.npy" },
      live: null,
      fallbackName: "表单名",
    });
    expect(r).toMatchObject({
      name: "槽里冻结的名字",
      workspace: RUN_A,
      indexPath: "C:\\A\\total_fea.npy",
      source: "run",
    });
    expect(r.summaryIndex).toBeUndefined();
  });

  it("★ reports an unknown workspace as unknown instead of inventing one", () => {
    // `get_slot_export_context` threw and the caller has no live run — the S126 degraded state.
    // It must stay `""` so `resolveArchiveIndex` can answer `unknownWorkspace` rather than
    // probing the drive root, and the name must fall back to the form rather than to "".
    const r = resolveRowIdentity({
      runId: "rb222222222b",
      ctx: null,
      live: null,
      fallbackName: "表单名",
    });
    expect(r.workspace).toBe("");
    expect(r.name).toBe("表单名");
    expect(r.source).toBe("none");
  });

  it("keeps the live NAME even when only the context knows the workspace", () => {
    // a live run whose snapshot has a name but no workspace yet (early in a start)
    const r = resolveRowIdentity({
      runId: null,
      ctx: { modelName: "盘上写的名字", workspace: RUN_A, indexPath: null },
      live: { modelName: "本次训练名", workspace: "" },
      fallbackName: "表单名",
    });
    expect(r.name).toBe("本次训练名");
    expect(r.workspace).toBe(RUN_A);
    // the workspace came from the context, so `source` must say so — otherwise a reader would
    // conclude the live run answered both
    expect(r.source).toBe("run");
  });
});
