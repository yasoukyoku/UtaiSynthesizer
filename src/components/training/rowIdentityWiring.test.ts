/**
 * 第三条边:**`resolveRowIdentity` 是对的** 与 **存档段真的按行用了它** 是两条断言,
 * 而 vitest 不做组件测试 ⇒ 后者没有任何东西看得见。这份闸就是那个看得见的东西。
 *
 * 它防的具体坏法(§F2⒝ 批 2 ④b,侦察的完整性批评员判为**原子**的那一组):
 * 名字按行解析、而试听的工作区仍取槽级那一个标量 ⇒ 一行标着 run B 的名字、
 * 放出来的却是 run A 的缓存。缓存命中只看 `<dir>/model.json` 在不在,
 * `workspace_is_a_slot` 那道结构闸对**兄弟 run** 也放行 ⇒ 无异常、无 CODE、无 toast,
 * 症状只是「这个存档听起来像那个存档」—— 耳朵判不出来的一类。
 *
 * ⛔ 扫源码前先把**整行注释**抹成等长空白:裸子串搜索连注释一起搜,于是任何一条含有锚点
 * 字面量的注释都会替真正的调用点满足断言(同一条教训在 Rust 侧的源序棘轮上刚修过)。
 */
import { describe, it, expect } from "vitest";

type NodeFs = { readFileSync(p: string, enc: string): string };
const importFs = (): Promise<NodeFs> => {
  const spec = "node:fs";
  return import(/* @vite-ignore */ spec) as Promise<NodeFs>;
};

const FILE = "src/components/training/TrainingPage.tsx";
/** ★§F2⒝ ④e —— per-run 的按钮(改名、继续、再训一个、S133 的删除)住在**这个**文件里,
 *  而这份闸此前只扫 `FILE` ⇒ 对它们零覆盖。 */
const PROJECT_DETAIL = "src/components/training/ProjectDetail.tsx";

/** 整行 `//` 注释抹成等长空白(行数与相对顺序不变),这样锚点只可能落在**代码**上。 */
function codeOnly(src: string): string {
  return src
    .split("\n")
    .map((l) => (l.trimStart().startsWith("//") || l.trimStart().startsWith("*") || l.trimStart().startsWith("/*") ? " ".repeat(l.length) : l))
    .join("\n");
}

describe("archive rows resolve their identity per row", () => {
  it("★ every audition render is handed THIS ROW's workspace, not a slot-level scalar", async () => {
    const fs = await importFs();
    const code = codeOnly(fs.readFileSync(FILE, "utf8"));
    for (const cmd of [
      "render_audition_vocoder",
      "render_audition_diffusion",
      "render_audition_voice",
    ]) {
      const at = code.indexOf(`"${cmd}"`);
      expect(at, `${cmd} invoke not found in ${FILE}`).toBeGreaterThan(0);
      // ⛔ the window MUST stop at this invoke's own closing `});`. A fixed-width window is the
      // defect this file exists to prevent, one level up: the first version took 500 characters,
      // which swallowed the NEXT invoke — so breaking the vocoder call was satisfied by the
      // diffusion call's argument and the probe reported GREEN. (Caught by mutation W1.)
      const end = code.indexOf("});", at);
      expect(end, `${cmd} invoke has no closing brace`).toBeGreaterThan(at);
      const window = code.slice(at, end);
      expect(
        window,
        `${cmd} must pass the row-resolved workspace — a slot-level one renders run B's row ` +
          `from run A's audition cache, and the cache hit predicate is only「model.json 在不在」`,
      ).toContain("workspace: auditionWs");
    }
  });

  it("★ the suggested import name is never taken from a slot-level scalar", async () => {
    const fs = await importFs();
    const code = codeOnly(fs.readFileSync(FILE, "utf8"));
    // the old shape: one slot-level `exportName` / `exportWorkspace` const, read by every row
    expect(
      /\bconst exportName\b/.test(code) || /\bconst exportWorkspace\b/.test(code),
      "the slot-level export identity is back: two runs of one slot then propose the SAME model " +
        "name, and import_model replaces by name with no dialog",
    ).toBe(false);
    // every CALL to suggestedName must pass the row-resolved name as its second argument
    const calls = [...code.matchAll(/suggestedName\(([^)]*)\)/g)].filter(
      (m) => !code.slice(Math.max(0, m.index! - 40), m.index!).includes("const suggestedName ="),
    );
    expect(calls.length, "no suggestedName call sites found — the anchor moved").toBeGreaterThan(0);
    for (const m of calls) {
      expect(m[1], `suggestedName call without a per-row name: ${m[0]}`).toContain(",");
    }
  });

  it("★ a same-name install is never silent — on either import path", async () => {
    // Why this is a training-page gate at all: `import_model` replaces by NAME, and a saved
    // project remembers its voice by that same string (`Track.voiceModel`, re-bound at load by
    // `modelPathHeal`). So a silent replace does not merely lose a model — it swaps the singer in
    // projects the user has not opened in months. ④ is the batch that makes two runs of one slot
    // propose the same name, so the ask has to exist before that lands.
    const fs = await importFs();
    const code = codeOnly(fs.readFileSync(FILE, "utf8"));

    const single = code.indexOf("const importCkpt = async (");
    expect(single, "importCkpt is gone").toBeGreaterThan(0);
    const singleInvoke = code.indexOf('"import_model"', single);
    expect(singleInvoke).toBeGreaterThan(single);
    expect(
      code.slice(single, singleInvoke),
      "the single-row import reaches import_model without asking about a same-name replace",
    ).toContain("confirmReplaceInstalled(name)");

    const batch = code.indexOf("const importSelected = async (");
    expect(batch, "importSelected is gone").toBeGreaterThan(0);
    const batchInvoke = code.indexOf('"import_model"', batch);
    expect(batchInvoke).toBeGreaterThan(batch);
    expect(
      code.slice(batch, batchInvoke),
      "the batch import's confirmation does not say which rows would replace an installed model",
    ).toContain("importReplaceBody");
  });

  /** 一次 `invoke(` 调用的**实参窗口**:从命令字面量往后按括号配平截断。
   *
   *  ⛔ 不许用定宽窗口,也不许用 `indexOf("});")` —— 前者是本文件 W1 变异抓到的那条(窗口吞掉
   *  下一个 invoke,于是断言被**别人**的实参满足);后者对写成
   *  `invoke(\n  "cmd",\n  { … },\n)` 的调用点根本找不到自己的收尾,会一路跑到几百行之后。 */
  function callWindow(code: string, cmd: string, from: number): [number, string] | null {
    const at = code.indexOf(`"${cmd}"`, from);
    if (at < 0) return null;
    let depth = 1; // 命令字面量已经在 invoke 的括号里
    for (let i = at; i < code.length; i++) {
      if (code[i] === "(") depth++;
      else if (code[i] === ")") {
        depth--;
        if (depth === 0) return [at, code.slice(at, i)];
      }
    }
    return [at, code.slice(at)];
  }

  it("★ every run-aware probe passes a runId — without it a second run makes the slot untrainable", async () => {
    // ⛔★★§F2⒝ ④e —— S132 的 flip 让「再训一个」真的铸出第二个 run,而这四处探针都在问
    // 「这个**槽**练到哪了」。`resolve_run_dir(None)` 对多于一个 run **拒绝作答**
    // (`RUN_AMBIGUOUS`),于是:
    //   · 两处 onStart 探针是 fail-closed ⇒ **训练根本起不来**,而且文案让用户「重启应用」——
    //     重启改变不了 run 数,那是一条死路。「继续训练」与「再训一个」两条路一起死。
    //   · 页根 effect 的 catch 把它吞成 `setSlotInfo(null)` ⇒ 续训锁全部解除、
    //     「会重跑预处理」的提示整体消失 ⇒ 用户改一个 costly 字段,下次运行静默换池、重跑几小时。
    // 这道闸钉的是**参数**,因为漏传是一次静默的省略,不会让任何别的判据变红。
    const fs = await importFs();
    const seen: Record<string, number> = {};
    // ⚠ **两个**文件。此前这份闸的 `FILE` 写死了 TrainingPage,而 per-run 的按钮住在
    //   ProjectDetail —— 一道看不见新入口的闸,对新入口是零覆盖。
    for (const file of [FILE, PROJECT_DETAIL]) {
      const code = codeOnly(fs.readFileSync(file, "utf8"));
      for (const cmd of ["get_training_slot_info", "get_slot_export_context"]) {
        let from = 0;
        for (;;) {
          const hit = callWindow(code, cmd, from);
          if (!hit) break;
          const [at, window] = hit;
          seen[cmd] = (seen[cmd] ?? 0) + 1;
          // 自检:一个吞掉了下一个调用的窗口会被**别人**的实参满足 —— 那正是 W1。
          expect(window.length, `${file}: ${cmd} 的实参窗口长得不像一次调用(${window.length} 字符)`)
            .toBeLessThan(1500);
          expect(
            window,
            `${file}: ${cmd} 少传了 runId —— 两个 run 之后它问的是一个没有答案的问题(RUN_AMBIGUOUS)`,
          ).toContain("runId");
          from = at + 1;
        }
      }
    }
    // 锚点还在(命令改名/调用点被删时不许静默变绿)
    expect(seen["get_training_slot_info"], "槽状态探针一个都没找到 —— 锚点漂了").toBeGreaterThanOrEqual(4);
    expect(seen["get_slot_export_context"], "导出上下文探针一个都没找到 —— 锚点漂了").toBeGreaterThanOrEqual(2);
  });

  it("★ a failed probe shows the mapped CODE, not a raw Rust string", async () => {
    // `String(e)` 会把 `RUN_AMBIGUOUS: 2 runs in D:\…` 原样糊到用户脸上,而那条 CODE 的三语文案
    // **早就在 backendError.ts 的 CODE_KEYS 里**——只是没被用上。
    const fs = await importFs();
    const code = codeOnly(fs.readFileSync(FILE, "utf8"));
    const calls = [...code.matchAll(/t\("training\.probeFailed",\s*\{\s*err:\s*([^}]*)\}/g)];
    expect(calls.length, "probeFailed 的调用点一个都没找到 —— 锚点漂了").toBeGreaterThanOrEqual(2);
    for (const m of calls) {
      expect(m[1], `probeFailed 直接吐了裸 Rust 串:${m[0]}`).toContain("backendErrorMessage");
    }
  });

  it("the wiring probe can actually see the file it claims to scan", async () => {
    // ⚠ 自检:一个读不到文件、或把整份源码都抹成空白的探针,上面两条会**为错误的原因**变绿。
    const fs = await importFs();
    const raw = fs.readFileSync(FILE, "utf8");
    const code = codeOnly(raw);
    expect(raw.length).toBeGreaterThan(50_000);
    expect(code.length).toBe(raw.length);
    expect(code).toContain("const rowIdentityFor");
    expect(code.replace(/\s/g, "").length).toBeGreaterThan(raw.replace(/\s/g, "").length * 0.5);
  });
});
