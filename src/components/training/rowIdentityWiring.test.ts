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
import { poolCostFieldIds } from "../../lib/training/costlyNote";
import { NO_FORM_CONTROL, POOL_FORM_FIELDS } from "../../lib/training/formForSlot";
import type { TrainingBackend } from "../../store/training";

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
    for (const fn of ["visibleRuns(", "slotStarted(", "pickDiffHost(", "prepPoolLine(", "foldRunRows("]) {
      expect(code, `${fn} 的调用点不见了`).toContain(fn);
    }
    // ⛔ S141:折叠是**纯观感**,不许改变「这个槽开始过没有」。`slotStarted` 必须吃**全部**
    // 过滤后的行,而不是折叠之后剩下的那几行 —— 否则收起来之后卡片会写「尚未开始」,
    // 而它下面明明还挂着 run(④e 修掉的正是这类自相矛盾的一屏)。
    expect(
      /slotStarted\(runs\)/.test(code),
      "`slotStarted` 不再吃全部行 —— 折叠一旦影响到它,收起来的槽会自称「尚未开始」",
    ).toBe(true);
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

  it("★ S142 §E2E-M11 —— 每一条走进参数页的路,都必须先把表单放回这个 run 已经在的地方", async () => {
    // §E2E-M11 那条腿(`store/trainingStartPayload.test.ts`)钉的是
    // `formForSlot → updateConfig → start_training 的载荷`,而它自己**写了**那第一行 ——
    // 组件哪天改成内联一份,腿照样全绿。这道闸就是那一半。
    //
    // ⛔ 坏法是可判定的:不还原 costly 那一档**不会被拒**,而是静默换池、把切片与全部伴生
    // 特征重跑一遍(几小时),并把 manifest 里的记录一起覆盖掉。S128 笔 0 修的正是它。
    const fs = await importFs();
    const code = codeOnly(fs.readFileSync(PROJECT_DETAIL, "utf8"));

    expect(code, "ProjectDetail 不再从 lib/training/formForSlot 取决策").toContain(
      "lib/training/formForSlot",
    );

    // 具体形状排前面(S108):「同一条规则的第五份副本又长回来了」是诊断。
    expect(
      /diff_steps\s*>\s*0/.test(code),
      "ProjectDetail 又自己写了一份 `diff_steps > 0` —— 那正是 S128 收编掉的第五份副本",
    ).toBe(false);

    // ★ 三条路(继续训练 / 再训一个 / 浅扩散)都以 `setRoute({ seg: nextSegFor(` 收尾,
    //   而它**前面最近的那个** `updateConfig({` 里必须摊开 formFor —— 少一条就是那条路上
    //   的表单被打回出厂默认。
    const hops = [...code.matchAll(/setRoute\(\{ seg: nextSegFor\(/g)];
    expect(hops.length, "进参数页的路一条都没找到 —— 锚点漂了,这条闸什么也没看").toBe(3);
    for (const h of hops) {
      const before = code.slice(0, h.index);
      const at = before.lastIndexOf("updateConfig({");
      expect(at, "这条路上根本没有 updateConfig —— 表单没有被放回任何地方").toBeGreaterThan(0);
      const window = before.slice(at);
      expect(window.length, "窗口是空的").toBeGreaterThan(20);
      expect(
        window,
        "这条进参数页的路没有摊开 formFor —— costly 那一档会被打回默认,下一次运行静默换池重跑",
      ).toContain("...formFor");
    }
  });

  it("★ S142 §E2E-M10 —— 代价提示还在屏幕上,而且每个控件都挂着它自己的那一条", async () => {
    // ⛔ 这条排在纯模块判据**之前**的理由:在它之前,把整段 costly 提示删光,全仓一条判据都
    // 不会红 —— i18n 的对拍闸只钉结构(三语键集合/占位符/非空串),死键闭合只管 `backend.*`,
    // 所以 `training.resumeCostly` 会以「三语齐全但没有人用」的姿态活下去;而 vitest.config.ts
    // 明写不做渲染。⇒ 先给这个功能一条**存在性**判据,再谈它的分支覆盖。
    const fs = await importFs();
    const code = codeOnly(fs.readFileSync(FILE, "utf8"));

    expect(code, "TrainingPage 不再从 lib/training/costlyNote 取决策").toContain(
      "lib/training/costlyNote",
    );

    // 具体形状排前面(S108):「被写回组件体了」是诊断,「调用点不见了」只是症状。
    const reInlined: [RegExp, string][] = [
      [
        /lockedFieldIds\([^)]*"costly"/,
        "costly 那一档的集合又在组件体里现算了 —— 而那一跳的判据(尤其 scope 那一半)全在纯模块上",
      ],
      [
        /poolInvalidatingIds\(/,
        "又直接调 poolInvalidatingIds 了 —— 两半分开求交正是 S128 那条等价变异能藏身的形状",
      ],
      [
        // ★S142 笔 3:代价提示不许再被「这是不是重训」门住。池是**槽级、跨 run 共享**的,
        // 重训不清空整槽(S132 的 flip)⇒ 那道门一关,唯一能改这些字段的那条路上代价就没人说了。
        // ⚠ 这句只有源码闸说得出来:`poolAtStake` 不再收 `retrainIntent`,是**类型层**的事实,
        // 单测对它一个字也说不上(变异 C4 当场证明了那条单测是装饰件)。
        /poolAtStake\([^)]*retrainIntent/,
        "代价提示又被 retrainIntent 门住了 —— 重训路径上池还在、代价还在,而屏幕会重新变哑",
      ],
    ];
    for (const [re, why] of reInlined) {
      expect(re.test(code), why).toBe(false);
    }

    // ★ 逐 backend、逐字段:**这个控件与下一个控件之间**必须挂着它自己的那一条提示。
    //   期望完全由锁表 + `POOL_FORM_FIELDS` 推出来 —— 锁表哪天多一行 costly+pool 的可编辑
    //   字段,这里会点名说「哪个 backend 的哪个控件没挂」,而那正是今天没有任何东西看得见的坏法。
    const backends: TrainingBackend[] = ["rvc", "sovits", "sovits_v2", "sovits_diff", "vocoder"];
    const wantedKeys = new Set<string>();
    for (const backend of backends) {
      for (const id of poolCostFieldIds(backend, true)) {
        if (NO_FORM_CONTROL.has(id)) continue; // `dataset` 归数据页,参数页没有控件代表它
        const key = POOL_FORM_FIELDS[backend][id];
        expect(key, `${backend}/${id} 在 POOL_FORM_FIELDS 里没有表单字段`).toBeTruthy();
        wantedKeys.add(key as string);

        const at = code.indexOf(`updateConfig({ ${key}`);
        expect(at, `${backend}/${id}:找不到写 ${key} 的控件 —— 锚点漂了,这条闸什么也没看`).toBeGreaterThan(0);
        const rest = code.slice(at + 1);
        const next = rest.indexOf("updateConfig({");
        const window = rest.slice(0, next < 0 ? rest.length : next);
        expect(window.length, `${backend}/${id}:窗口是空的`).toBeGreaterThan(20);
        expect(
          window,
          `${backend}/${id}:${key} 这个控件旁边没有代价提示 —— 改它会静默换池、重跑整份预处理`,
        ).toContain(`costlyNote("${id}")`);
      }
    }

    // 反过来:多出来的挂点也要有人问一句(挂在一个**不**换池的字段上 = 屏幕在撒谎)。
    // 期望值由上面那张表推出,不是手写的常量。
    expect(
      [...code.matchAll(/\{costlyNote\("/g)].length,
      "costlyNote 的挂点数与锁表推出来的控件数对不上",
    ).toBe(wantedKeys.size);
  });

  it("★ S141 —— 「再训一个」这条路不许再弹第三个对话框问「模型已存在」", async () => {
    // 用户实机**连报两次**的那一条。他已经在项目页答过两次(「重训这个架构」的 danger 框、
    // 「给这个新 run 起个名字」),而走到开始训练时还会撞上第三个框,标题写着**「模型已存在」**、
    // 正文把**新**名字说成「已存在名为 X 的模型/训练记录」——**那句话是假的**:这个分支的条件是
    // `wsExists`,问的是 `config.runId` 指的那个【旧】run 有没有产物,与输入的名字毫无关系。
    // 用户原话:「无论我输入什么训练名 我选『再训一个』的时候都会提示已存在模型/训练记录」。
    //
    // ⛔ 它也不再是一道「擦除同意」:`wipe_confirmed` 在后端只有一个消费点,且要求
    // `backend == "sovits_diff"`,而 `retrainIntent` 只由 `retrainFamily` 置位、它只传 family
    // ⇒ 结构上到不了那个分支。⇒ 这一问没有任何东西要被同意,只剩噪音与一句假话。
    //
    // 这道闸是**源码闸**:这段活在组件的 async 闭包里,vitest 驱不动;而「弹不弹」恰恰是
    // i18n 那两道结构闸看不见的东西。
    const fs = await importFs();
    const code = codeOnly(fs.readFileSync(FILE, "utf8"));
    // ⚠ 锚点取 `wantsRetrain` 的**绑定处**,不是某个文案键 —— 早一版锚在
    // `t("training.confirmExistBody"` 上,而我随后在它**前面**加了分支,窗口起点就被自己的
    // 改动挤走、看不见新加的那一支(当场红)。**锚点要选不会被正文改动推走的东西。**
    const at = code.indexOf("const wantsRetrain = useTrainingStore.getState().retrainIntent;");
    expect(at, "开始训练那个「已存在」对话框的锚点漂了").toBeGreaterThan(0);
    const window = code.slice(at, code.indexOf("buttons:", at));
    // ⚠ 量**去掉空白之后**的长度:`codeOnly` 把注释抹成等长空白,原始长度会随注释一起涨。
    const dense = window.replace(/\s/g, "").length;
    expect(dense, `窗口长得不像一次调用(去空白 ${dense} 字符)`).toBeLessThan(900);
    expect(dense, "窗口是空的 —— 锚点之后立刻就是 buttons:,这道闸什么也没看").toBeGreaterThan(20);

    // ⑴ 「再训一个」必须**短路**在弹框之前。
    expect(
      /if \(wantsRetrain\)\s*\{[\s\S]{0,200}fresh = true;/.test(window),
      "「再训一个」这条路又落进那个『模型已存在』对话框了 —— 用户已经答过两次,而这一问的标题" +
        "对他刚起的新名字是假话,且它已经没有任何东西要征求同意(wipe_confirmed 只对 sovits_diff 有效)",
    ).toBe(true);
    // ⑵ 而且**不许**在那条路上顺手把擦除同意也给了:后端今天读不到它,但将来分类若变,
    //    「没同意过」是 fail-closed 的方向。
    expect(
      /if \(wantsRetrain\)\s*\{[\s\S]{0,200}wipeConfirmed = true/.test(window),
      "「再训一个」这条路自作主张地把 wipeConfirmed 置成了 true —— 那是把一道 fail-closed 的" +
        "保险默认打开",
    ).toBe(false);
    // ⑶ 兜底:另一条路(普通续训/重训)那个对话框仍然在,别把它一起删了。
    expect(window, "「已存在」对话框整个不见了 —— 普通续训那条路失去了它唯一的选择入口")
      .toContain("training.confirmExistBody");
  });

  it("★★ S143 §E2E-M25 ⑶ —— 四颗 per-run 按钮的 disabled 真的接到了那个决策上", async () => {
    // ⛔ 这一条是这一笔的**第一条判据**,而且它必须是存在性/接线闸,不是分支覆盖:
    //    纯模块那 17 条腿吃的是参数,它们对「组件根本没在用这个函数」零分辨力;而 S142 刚为
    //    同一个形状付过账(整段代价提示可以被删光而全仓零红)。
    //
    // ⛔ 它读的是 `PROJECT_DETAIL` 不是 `FILE` —— 照 S142 那道 M10 闸的形状抄一份会写出一条
    //    读**错文件**的闸,它会因为「TrainingPage 里当然没有这些新形状」而以错误的理由绿。
    const fs = await importFs();
    const code = codeOnly(fs.readFileSync(PROJECT_DETAIL, "utf8"));

    expect(code, "决策模块不再被引用 —— 四颗按钮又变回裸的布尔了").toContain(
      'from "../../lib/training/liveRun"',
    );
    expect(code, "行级门禁不再被调用").toContain("runRowActions({");

    // ★ 逐颗按钮:`disabled=` 里必须出现它自己那一档,而不是任何一个裸标志。
    //   期望是**由那四个字段名推出来的**,不是手抄一串 —— 手抄的那一份打错字会两边一起绿。
    for (const which of ["cont", "retrain", "del", "rename"] as const) {
      expect(code, `${which} 那颗按钮的 disabled 没有接到 runRowActions 上`).toContain(
        `disabled={gates.${which}.disabled}`,
      );
      // …而且它得说出为什么(一颗禁着却不解释的按钮只是把困惑换了个位置)。
      expect(code, `${which} 那颗按钮禁着却不说为什么`).toContain(`gateTitle(gates.${which}`);
    }

    // ⛔ 被换掉的那个**具体**写法不许回来:删除按钮此前是 `blocked || busy`,而那个 `busy`
    //    是「本页有一次 invoke 在飞」,与「有没有东西在跑」毫无关系 —— 它读起来完全像一道
    //    忙碌保护,实际只防重复点击。
    // ⚠ 这里**没有**写成「run 行里不许出现任何 `disabled={blocked}`」:第一版就是那样,而它
    //    当场红了,红得对 —— 同一行里的「存档 X GB」是一条只读导航,`blocked` 对它是正确的。
    //    ⇒ 诚实边界:这条闸看得见「四颗按钮被改回裸标志」,看不见「有人新加了第五颗按钮而忘了
    //    给它门禁」。后者今天没有便宜的判据,写在这里免得下一个人以为它被盖住了。
    expect(code, "删除按钮又变回了 `blocked || busy` —— 那个 busy 只防重复点击,不是「有东西在跑」")
      .not.toContain("disabled={blocked || busy}");

    // ★★ 死路的另外两个入口:槽级「开始训练」与浅扩散卡。它们与 run 行那两颗走**同一条路**
    //    (updateConfig → setRoute → 运行段),而有训练在跑时预启动卡结构上根本不渲染
    //    (它整块在 `snapshot.state === "idle"` 里面)⇒ 用户会落在**另一个 run 的实时进度**上,
    //    没有开始按钮、没有解释,而且要等那个 run 跑完再点「清空结果」才回得去。
    //    ⇒ 五颗按钮必须同一条门禁;只堵 run 行那两颗 = 把同一条死路留了三个入口。
    const starts = [...code.matchAll(/disabled=\{slotStart\.disabled\}/g)];
    expect(
      starts.length,
      "槽级/浅扩散的「开始训练」没有全部接到同一条门禁上 —— 那条死路还留着入口",
    ).toBe(4);

    // ★ 订阅本身:载体接进来了、而这一页仍然收不到快照变化 ⇒ 徽章/禁用会对着一份挂载时的
    //   旧数据工作,**而所有纯模块判据照绿**。这是这一笔唯一能看见「它有没有真的活起来」的地方。
    expect(code, "这一页又不订阅实时快照了 —— 训练开始/结束时它连重渲染都不会发生").toContain(
      "useTrainingStore((s) => s.snapshot.state)",
    );
    // ⛔ 而且必须是**标量** selector:订整个 snapshot 会让整片卡片墙按训练步频率重渲染。
    expect(code, "订了整个 snapshot 对象 —— store 每一步都换一个新对象").not.toMatch(
      /useTrainingStore\(\(s\) => s\.snapshot\)/,
    );

    // ★ 「有没有任何长任务在跑」必须真的去问后端那份清单(镜像 `running_tasks_of` 的粗粒度);
    //   只按训练禁 = 前端比后端更宽,而那正好落空在最常见的一格上。
    expect(code, "删除/改名不再镜像后端那份长任务清单").toContain('invoke<string[]>("running_tasks")');
  });

  it("★★ S143 §E2E-M25 ⑶ —— 那四句禁用文案三语齐全,而且真的被显示出来", async () => {
    // ⛔ 为什么这条必须存在:`training.*` **没有死键闭合**(parity.test.ts 那道闸硬编只扫
    //    `backend.*`),而全仓没有任何东西把源码里的 `t("...")` 与键集合对上 ⇒ 把 key 打错一个
    //    字母,界面上会原样显示那串 key,而 vitest / cargo / M20 全绿。
    const fs = await importFs();
    // 期望**由那张表推出**,不是手抄字面串(手抄 = 同一个串在两边各写一遍,打错两遍一起绿)。
    const { GATE_REASON_KEYS } = await import("../../lib/training/liveRun");
    expect(GATE_REASON_KEYS.length, "禁用原因表读空了").toBe(4);

    for (const lang of ["zh", "en", "ja"]) {
      const json = JSON.parse(fs.readFileSync(`src/i18n/${lang}.json`, "utf8")) as Record<
        string,
        Record<string, string>
      >;
      for (const key of GATE_REASON_KEYS) {
        const [ns, k] = key.split(".") as [string, string];
        const v = json[ns]?.[k];
        expect(typeof v, `${lang}.json 缺 ${key}`).toBe("string");
        expect((v ?? "").length, `${lang}.json 的 ${key} 是空串`).toBeGreaterThan(1);
      }
    }

    // …而且它们真的走到了屏幕上:决策层给键、组件层显示它。
    const code = codeOnly(fs.readFileSync(PROJECT_DETAIL, "utf8"));
    expect(code, "禁用原因没有被翻译成文案 —— 那四个键会以「三语齐全但没人用」的姿态活着")
      .toContain("gateReasonKey(g.reason)");
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
