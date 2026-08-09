/**
 * `WorkspaceInfo` 的 Rust ↔ TS 字段对拍(§F2⒝ 批 2 ④d 笔 0)。
 *
 * ## 为什么这条边需要一道闸
 *
 * `WorkspaceInfo` 是「这个 run 当初是怎么练的」唯一的过江通道:Rust 从 `run_manifest.json`
 * (与 ④d 起的池指纹)读出事实,前端拿它把参数页的表单放回槽在的地方。两侧的字段表是**各写
 * 一份的手抄本** —— Rust 是 `#[derive(Serialize)]` 的 struct,TS 是一个 `interface`,中间没有
 * 代码生成,`tsc` 也不会知道后端多送了什么。
 *
 * 于是漂移的形态是**静默的**:Rust 加一个字段而 TS 忘了声明 ⇒ 读它恒为 `undefined` ⇒ 还原逻辑
 * 一声不响地跳过那个字段。这正是 ④d 笔 0 在修的那类缺陷本身(表单忘了一个决定池身份的值 ⇒
 * 静默换池重跑),所以让它有可能以「加字段时忘了另一半」的形式再回来,是不能接受的。
 *
 * ## 它证明什么、不证明什么
 *
 * 证明:**字段名集合逐个相同**,而且两侧都还是 snake_case(这个 struct 故意没有 `rename_all`,
 * 见它自己的 doc)。
 * 不证明:类型对不对。`aug_copies: string` 这种错这里看不出来 —— 那是另一种漂移,而这一种
 * (整个字段消失成 `undefined`)是唯一会**静默改变行为**的一种。
 *
 * ⚠ 两侧都是从源码解析出来的,所以「解析器读空 ⇒ 两个空集相等 ⇒ 全绿」是这条闸自己最可能的
 * 死法。下面第一条用例就是它的**存活证明**:锚点字段 + 数量下限,缺一条这道闸就不算活着。
 */

import { describe, expect, it } from "vitest";

// 前端 tsconfig 无 @types/node —— 同 resumeLockParity:用「变量说明符的动态 import」拿 fs。
type NodeFs = { readFileSync(p: string, enc: string): string };
const importFs = (): Promise<NodeFs> => {
  const spec = "node:fs";
  return import(/* @vite-ignore */ spec) as Promise<NodeFs>;
};

const RS = "src-tauri/src/training/mod.rs";
const TS = "src/store/training.ts";

/** 取一个声明的花括号体:从 `head` 那一行之后,到第一行**顶格**的 `}`。两边的这两个声明都在
 *  文件顶层,所以顶格的 `}` 就是它的收尾 —— 而这比「数括号」更难被注释里的括号骗到。 */
function declBody(src: string, head: string): string {
  const at = src.indexOf(head);
  expect(at, `找不到声明:${head}`).toBeGreaterThanOrEqual(0);
  const rest = src.slice(at + head.length);
  const end = rest.search(/^\}/m);
  expect(end, `${head} 没有顶格收尾`).toBeGreaterThan(0);
  return rest.slice(0, end);
}

/** Rust 的 `pub <name>: <ty>,`。doc 注释与属性行不匹配这个形状,所以自动被略过。 */
function rustFields(src: string): string[] {
  return [...declBody(src, "pub struct WorkspaceInfo {").matchAll(/^\s*pub ([a-z0-9_]+):/gm)].map(
    (m) => m[1]!,
  );
}

/** TS 的 `<name>: <ty>;`。同样只匹配字段行,注释行匹配不上。 */
function tsFields(src: string): string[] {
  return [
    ...declBody(src, "export interface WorkspaceInfo {").matchAll(/^\s{2}([a-z0-9_]+)\??:/gm),
  ].map((m) => m[1]!);
}

describe("WorkspaceInfo Rust ↔ TS 对拍", () => {
  it("★存活证明:两侧解析器都真的读到了东西(否则这道闸是两个空集相等)", async () => {
    const fs = await importFs();
    const rust = rustFields(fs.readFileSync(RS, "utf8"));
    const ts = tsFields(fs.readFileSync(TS, "utf8"));
    for (const [side, got] of [
      ["rust", rust],
      ["ts", ts],
    ] as const) {
      // 锚点:三个字段各代表一类 —— 最老的一个、④d 新加的那个、以及一个 Option/null 三态的。
      for (const anchor of ["exists", "aug_copies", "loudnorm", "vol_embedding"]) {
        expect(got, `${side} 解析器没读到 ${anchor}`).toContain(anchor);
      }
      expect(got.length, `${side} 只解析出 ${got.length} 个字段,解析器多半读空了`).toBeGreaterThan(10);
      expect(new Set(got).size, `${side} 有重复字段名`).toBe(got.length);
    }
  });

  it("字段名集合完全一致", async () => {
    const fs = await importFs();
    expect(new Set(tsFields(fs.readFileSync(TS, "utf8")))).toEqual(
      new Set(rustFields(fs.readFileSync(RS, "utf8"))),
    );
  });

  it("★这个 struct 不许长出 rename_all —— 一长出来上面那条对拍就在比错的东西", async () => {
    const src = (await importFs()).readFileSync(RS, "utf8");
    const head = src.lastIndexOf("#[derive", src.indexOf("pub struct WorkspaceInfo {"));
    const attrs = src.slice(head, src.indexOf("pub struct WorkspaceInfo {"));
    expect(attrs).not.toMatch(/rename_all/);
  });
});
