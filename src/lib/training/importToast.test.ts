import { describe, expect, it } from "vitest";
import {
  attachToasts,
  batchImportToast,
  collectWarningCodes,
  importToast,
} from "./importToast";

/**
 * §E2E-M6(前端那一跳)/ §E2E-M7(S141)—— 后端的 warning 到底有没有到达用户。
 *
 * ⛔ 记忆里 M7 那格的阴性对照写的是「无 warning 时**不弹**」,而**三条路在无 warning 时
 * 都弹一条 toast**。照原文写会写出一条永远红的断言,然后被判成「假红」耸肩带过 ——
 * S129 铁律点名的那类。⇒ 判据落在**档位与追加行**,不是「有没有 toast」。
 */

const VOCODER = "WARN_DIFFUSION_VOCODER_CUSTOM";
const INDEX_MISSING = "WARN_INDEX_MISSING";
const INDEX_UNKNOWN = "WARN_INDEX_CONTEXT_UNKNOWN";

describe("collectWarningCodes(§E2E-M1 的同一个漏斗)", () => {
  it("★ 前端那条「不知道去哪找索引」与后端的 warning 走同一个漏斗", () => {
    // M1 那一轮买回来的真缺陷:【不知道该去哪找】与【确实没有】此前被合并成同一个
    // `undefined`。两者的补救动作完全不同,所以它们必须都能到达用户,而且是同一条路。
    expect(collectWarningCodes({ warnings: [INDEX_MISSING] }, INDEX_UNKNOWN)).toEqual([
      INDEX_MISSING,
      INDEX_UNKNOWN,
    ]);
  });

  it("两个来源各自单独出现时也要在", () => {
    expect(collectWarningCodes({ warnings: [VOCODER] }, null)).toEqual([VOCODER]);
    expect(collectWarningCodes({ warnings: [] }, INDEX_UNKNOWN)).toEqual([INDEX_UNKNOWN]);
    expect(collectWarningCodes(undefined, INDEX_UNKNOWN)).toEqual([INDEX_UNKNOWN]);
  });

  it("都没有 ⇒ 空,而且空串不算一条 warning", () => {
    expect(collectWarningCodes(null, null)).toEqual([]);
    expect(collectWarningCodes({}, undefined)).toEqual([]);
    // `indexWarningCode` 在「没有可报的」时给的是 null;一个空串混进去会变成一行空白追加行
    expect(collectWarningCodes({ warnings: [] }, "")).toEqual([]);
  });
});

describe("importToast(§E2E-M7:单条导入)", () => {
  it("★ 无 warning ⇒ success 且【零追加行】——⛔ 不是「不弹」", () => {
    const r = importToast("已导入 歌姫", []);
    expect(r.level).toBe("success");
    expect(r.text).toBe("已导入 歌姫");
    expect(r.text).not.toContain("\n");
  });

  it("★ 有 warning ⇒ 档位必须变 info,而且那条码要真的出现在文本里", () => {
    // 「档位变了」与「内容到了」是两条断言:只改档位而不追加行,用户看见的是一条
    // 语气不同的成功提示,而不知道索引没装上。
    const r = importToast("已导入 歌姫", ["检索索引没找到"]);
    expect(r.level).toBe("info");
    expect(r.text).toBe("已导入 歌姫\n检索索引没找到");
  });

  it("多条 warning 各占一行", () => {
    const r = importToast("已导入", ["a", "b"]);
    expect(r.text.split("\n")).toEqual(["已导入", "a", "b"]);
    expect(r.level).toBe("info");
  });
});

describe("batchImportToast(§E2E-M7:批量)", () => {
  it("全成功且无 warning ⇒ success,一行", () => {
    const r = batchImportToast({ doneText: "导入 3 个", partialText: "x", failed: [], warns: [] });
    expect(r).toEqual({ text: "导入 3 个", level: "success" });
  });

  it("有 warning 无失败 ⇒ info + 追加行", () => {
    const r = batchImportToast({
      doneText: "导入 3 个",
      partialText: "x",
      failed: [],
      warns: ["歌姫: 索引没找到"],
    });
    expect(r.level).toBe("info");
    expect(r.text).toBe("导入 3 个\n歌姫: 索引没找到");
  });

  it("★ 有失败 ⇒ error,而 warning【也要一起呈现】", () => {
    // ⛔ 这一条是承重的:一条「索引没装上」不会因为同一批里另有一条失败就变得不重要,
    // 而失败与 warning 分属两个不同的补救动作。把 warns 从 error 那一档漏掉是静默的。
    const r = batchImportToast({
      doneText: "x",
      partialText: "导入 2/3",
      failed: ["歌姫B: 磁盘满"],
      warns: ["歌姫A: 索引没找到"],
    });
    expect(r.level).toBe("error");
    expect(r.text).toContain("歌姫B: 磁盘满");
    expect(r.text).toContain("歌姫A: 索引没找到");
    expect(r.text.split("\n")[0]).toBe("导入 2/3");
  });

  it("失败排在 warning 前面(先说坏消息)", () => {
    const r = batchImportToast({
      doneText: "x",
      partialText: "p",
      failed: ["F1", "F2"],
      warns: ["W1"],
    });
    expect(r.text.split("\n")).toEqual(["p", "F1", "F2", "W1"]);
  });
});

describe("attachToasts(§E2E-M6 的前端那一跳:附加浅扩散)", () => {
  it("★ 无 warning 时也弹【恰好一条】success —— 它与导入不是同一个漏斗", () => {
    // ⚠ 记忆一度把这条路归成「无 warning 时不弹」。实测它无条件先弹一条 success。
    const r = attachToasts("已附加 歌姫", []);
    expect(r).toEqual([{ text: "已附加 歌姫", level: "success" }]);
  });

  it("★ §F9 那条 vocoder 提示走的正是这条路:它必须【另开】一条 info", () => {
    // 用户报的那个案例:社区分享的浅扩散没带它自己微调过的声码器。提示是**一次性的、
    // 在导入那一刻**,错过就没有第二次 —— 所以「它有没有被弹出来」必须是可断言的。
    const r = attachToasts("已附加 歌姫", ["这个浅扩散是对着另一个声码器训练的"]);
    expect(r.length).toBe(2);
    expect(r[0]?.level).toBe("success");
    expect(r[1]).toEqual({ text: "这个浅扩散是对着另一个声码器训练的", level: "info" });
  });

  it("N 条 warning ⇒ N+1 条 toast(这是它与导入唯一的形状差别,钉住它)", () => {
    expect(attachToasts("base", ["a", "b", "c"]).length).toBe(4);
    expect(attachToasts("base", ["a", "b", "c"]).filter((x) => x.level === "info").length).toBe(3);
  });
});
