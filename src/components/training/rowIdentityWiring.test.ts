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

  it("★ S141 §E2E-M3/M4/M5 —— 槽卡片的决策留在纯模块里,不许再长回组件体", async () => {
    // 纯模块那半有 13 条判据(`lib/training/slotRows.test.ts`),但它们证明的是**函数是对的**。
    // 「这张卡真的在用那个函数」是另一条断言,而 vitest 不做组件测试 ⇒ 只有这道闸看得见。
    // 具体的坏法:有人在组件里就地写回一句 `runs.filter(startedRun)`(那正是 ④e 修掉的那条),
    // 纯模块的 13 条判据**全部照绿** —— 因为屏幕根本不再经过它们。
    const fs = await importFs();
    const code = codeOnly(fs.readFileSync(PROJECT_DETAIL, "utf8"));

    expect(
      code,
      "ProjectDetail 不再从 lib/training/slotRows 取决策 —— 那几个决策要么被删了,要么被就地写回了",
    ).toContain("lib/training/slotRows");
    // 搬走的那几个形状不许在这个文件里复活。⚠ 只钉**具体的那几个形状**,不是「不许出现
    // .filter(」—— 一条过宽的禁令会在下一次正常改动时误红,然后被人放宽成没有。
    //
    // ⛔ 这一组必须排在下面那条「调用点还在吗」**之前**(S108:具体的排前面,兜底的排最后)。
    // 把一句 `visibleRuns(runs)` 就地写回 `runs.filter(startedRun)` 会让**两组同时**为真,
    // 而先红的那一组决定下一个人读到的是哪句话:「它被写回组件体了」是诊断,
    // 「调用点不见了」只是症状。实测 R1 第一版正是红在症状上。
    const reInlined: [RegExp, string][] = [
      [/runs\.filter\(/, "画哪几行的过滤又被写回组件体了(④e 修的正是这一条)"],
      [/\.runs\.find\(/, "浅扩散宿主的挑选又被写回组件体了"],
      [/prepPoolCount/, "「预处理 N 份」的字段又被组件直接读了 —— show 与 count 会分头取值"],
      [
        /hasResumePoint\s*\|\|\s*\w+\.info\.has_main_progress/,
        "「练出过东西没有」这个谓词又出现了第二份手抄件(S141 刚收编掉一份)",
      ],
    ];
    for (const [re, why] of reInlined) {
      expect(re.test(code), why).toBe(false);
    }

    // 兜底:决策整个消失(不是被写回,而是被删掉/改名)也要红。
    for (const fn of ["visibleRuns(", "slotStarted(", "pickDiffHost(", "prepPoolLine("]) {
      expect(code, `${fn} 的调用点不见了`).toContain(fn);
    }
  });

  it("★ S141 §E2E-M6/M7 —— 三条导入路的 toast 决策留在纯模块里,不许再长回闭包", async () => {
    // 后端把「能恢复的失败」报成 warning(`WARN_INDEX_MISSING` / §F9 的
    // `WARN_DIFFUSION_VOCODER_CUSTOM`)。整条呈现链此前活在三个 async 闭包里,没有可导出的
    // 决策函数 ⇒ M6 的 Rust 半边做完了,而「它到底有没有到达用户」零判据。
    const fs = await importFs();
    const code = codeOnly(fs.readFileSync(FILE, "utf8"));

    expect(code, "TrainingPage 不再从 lib/training/importToast 取决策").toContain(
      "lib/training/importToast",
    );

    // 具体的形状排前面(S108)——「它被写回闭包了」是诊断,「调用点不见了」只是症状。
    const reInlined: [RegExp, string][] = [
      [
        /outcome\?\.warnings \?\? \[\]/,
        "又直接摊开了后端 warning —— 前端那条「不知道去哪找索引」就会从这条路上掉出去(§E2E-M1)",
      ],
      [
        /warns\.length > 0\s*\n?\s*\?/,
        "单条导入的档位三元又被写回闭包了(它决定 warning 是 info 还是被 success 盖住)",
      ],
      [
        /failed\.length > 0\s*\)\s*\{/,
        "批量导入的三档又被写回闭包了 —— 有失败时 warning 会不会一起呈现就没人守了",
      ],
    ];
    for (const [re, why] of reInlined) {
      expect(re.test(code), why).toBe(false);
    }

    // 兜底:决策整个消失也要红。⚠ `collectWarningCodes` 必须**三条路都在用**(单条 / 批量 /
    // 附加)—— 少一条就是那条路上的 warning 又变回了「后端说了但界面没说」。
    for (const fn of ["attachToasts(", "importToast(", "batchImportToast("]) {
      expect(code, `${fn} 的调用点不见了`).toContain(fn);
    }
    expect(
      [...code.matchAll(/collectWarningCodes\(/g)].length,
      "collectWarningCodes 的调用点少于三处 —— 三条导入路里有一条又不走同一个漏斗了",
    ).toBeGreaterThanOrEqual(3);
  });

  it("★ S141 —— 带着「再训一个」的新名字走到开始对话框时,续训那一档必须先说清它放弃了什么", async () => {
    // 实机第一次开窗口撞到的那一条:用户点「再训一个」、给新 run 起名 `run2-rvc`,然后在这个
    // 对话框里改主意选了「从最佳存档继续」⇒ **没有铸出新 run**,而那个新名字被写进了**旧 run**
    // 的 run.json(后端那一半已由 `training::name_to_persist` 修掉)。
    // 这道闸守的是另一半:**那个决定必须在屏幕上说出来**。它是源码闸,因为这段活在组件的
    // async 闭包里,vitest 驱不动 —— 而「文案在不在」恰恰是 i18n 那两道结构闸看不见的东西。
    const fs = await importFs();
    const code = codeOnly(fs.readFileSync(FILE, "utf8"));
    const at = code.indexOf('t("training.confirmExistBody"');
    expect(at, "开始训练那个「已存在」对话框的锚点漂了").toBeGreaterThan(0);
    // 只看这次 showConfirm 调用的正文构造,别一路扫到下一个对话框去(W1 那条血训)。
    const window = code.slice(at, code.indexOf("buttons:", at));
    expect(window.length, "正文窗口长得不像一次调用").toBeLessThan(800);
    expect(
      window,
      "续训那一档不再对「刚起了新名字」的情况说明它会放弃那个新 run —— 用户点下去之后,\
       屏幕上唯一的变化是名字,而产物仍然是旧 run 的",
    ).toContain("retrainIntentResumeWarn");
    expect(
      window,
      "那句提醒不再受「是不是从再训一个进来的」约束 —— 无条件挂上去会让普通续训也读到一句\
       与它无关的警告,而一条到处都亮的警告等于没有警告",
    ).toContain("wantsRetrain");
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

    // ⚠ 同一条自检也要盖住**第二个**被扫的文件:S141 那条闸里有四个 `not.toMatch`,
    // 而一个读空 / 全抹白的 ProjectDetail 会让它们**全部为错误的原因变绿**
    // (「否定式断言 + 空输入 = 恒真」是本仓反复买过的形状)。
    const pdRaw = fs.readFileSync(PROJECT_DETAIL, "utf8");
    const pdCode = codeOnly(pdRaw);
    expect(pdRaw.length).toBeGreaterThan(20_000);
    expect(pdCode.length).toBe(pdRaw.length);
    expect(pdCode).toContain("const dataHasDependents");
    expect(pdCode.replace(/\s/g, "").length).toBeGreaterThan(pdRaw.replace(/\s/g, "").length * 0.5);
  });
});
