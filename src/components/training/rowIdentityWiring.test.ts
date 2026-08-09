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
