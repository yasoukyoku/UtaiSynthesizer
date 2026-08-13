/**
 * `TrainingSnapshot` 的 Rust ↔ TS 字段对拍(§F2⒝-B2-⑤ / §E2E-M25 笔 0)。
 *
 * ## 为什么这条边到今天才有闸,而它是四件事的**上游**
 *
 * `WorkspaceInfo` 早有一份同形的对拍(见隔壁 `workspaceInfoParity.test.ts`),而**实时快照这条边
 * 一直是裸的**:两侧各写一份手抄本(Rust 是 `#[derive(Serialize)]` 的 struct,TS 是 `interface`),
 * 中间没有代码生成,`tsc` 不会知道后端多送了什么,`cargo` 不会知道前端少抄了什么。
 *
 * M25 要往这条边上加**一个新字段**(`run_id` —— 「哪个 run 正在跑」的载体),而它有四个消费者
 * (折叠保底 / 说出在跑的是谁 / 禁按钮 / 排序置顶)。漂移的形态因此是**四件一起静默失效**:
 * Rust 加了、TS 忘了 ⇒ 读出来恒 `undefined` ⇒ 每一行都答「不是我」⇒ 徽章不出现、不置顶、
 * 不禁用,**而 cargo 与 vitest 全绿**。这正是「判据看起来在测、其实测的是别的东西」的上游形态:
 * 下游那四条判据可以条条都对,而它们喂进去的载体从来没有过江。
 *
 * ★ 这道闸落地时做了一次**自然实验**:先加 Rust 那一侧、不动 TS,直接跑它 —— 它**正确地红了**
 * (18 vs 17)。加一个字段的时候顺手让对拍先红一次,买到的就是「这道闸没坏」。
 *
 * ## 它证明什么、不证明什么
 *
 * 证明:**字段名集合逐个相同**,而且这个 struct 仍然没有 `rename_all`(所以两侧都是 snake_case)。
 * 不证明:类型对不对,也不证明**值**对不对 —— 「`run_id` 送的是不是这次训练真正落在的那个 run」
 * 由 Rust 侧 `trun::run_id_of` 的行为判据 + `try_start` 那道源序棘轮回答,不在这里。
 *
 * ⚠ 两侧都是从源码解析出来的 ⇒「解析器读空 ⇒ 两个空集相等 ⇒ 全绿」是这条闸自己最可能的死法。
 * 第一条用例就是它的**存活证明**。⛔ 那个数量下限**必须按今天真实的字段数定**,照抄隔壁那份的
 * `> 10` 会让这个 18 个字段的 struct 少掉 7 个也照绿。
 */

import { describe, expect, it } from "vitest";

// 前端 tsconfig 无 @types/node —— 同 workspaceInfoParity:用「变量说明符的动态 import」拿 fs。
type NodeFs = { readFileSync(p: string, enc: string): string };
const importFs = (): Promise<NodeFs> => {
  const spec = "node:fs";
  return import(/* @vite-ignore */ spec) as Promise<NodeFs>;
};

const RS = "src-tauri/src/training/mod.rs";
const TS = "src/store/training.ts";

/** 取一个声明的花括号体:从 `head` 那一行之后,到第一行**顶格**的 `}`。两边这两个声明都在文件
 *  顶层,所以顶格的 `}` 就是它的收尾 —— 比「数括号」更难被注释里的括号骗到。 */
function declBody(src: string, head: string): string {
  const at = src.indexOf(head);
  expect(at, `找不到声明:${head}`).toBeGreaterThanOrEqual(0);
  const rest = src.slice(at + head.length);
  const end = rest.search(/^\}/m);
  expect(end, `${head} 没有顶格收尾`).toBeGreaterThan(0);
  return rest.slice(0, end);
}

/** Rust 的 `pub <name>: <ty>,`。doc 注释与属性行不匹配这个形状,自动被略过。 */
function rustFields(src: string): string[] {
  return [...declBody(src, "pub struct TrainingSnapshot {").matchAll(/^\s*pub ([a-z0-9_]+):/gm)].map(
    (m) => m[1]!,
  );
}

/** TS 的 `<name>: <ty>;` / `<name>?: <ty>;`。同样只匹配字段行。 */
function tsFields(src: string): string[] {
  return [
    ...declBody(src, "export interface TrainingSnapshot {").matchAll(/^\s{2}([a-z0-9_]+)\??:/gm),
  ].map((m) => m[1]!);
}

describe("TrainingSnapshot Rust ↔ TS 对拍(M25 载体的上游闸)", () => {
  it("★存活证明:两侧解析器都真的读到了东西(否则这道闸是两个空集相等)", async () => {
    const fs = await importFs();
    const rust = rustFields(fs.readFileSync(RS, "utf8"));
    const ts = tsFields(fs.readFileSync(TS, "utf8"));
    for (const [side, got] of [
      ["rust", rust],
      ["ts", ts],
    ] as const) {
      // 锚点各代表一类:最老的一个 · `#[serde(default)]` 的一个 · `skip_serializing_if` 的一个
      // (⚠ `warnings` 在健康 run 的线上**根本没有这个键** —— 所以这道闸只能按【源码声明】做,
      //  按「实际收到的 JSON 键」做会在这里假红)· 以及 M25 这一笔加的那个。
      for (const anchor of ["backend", "project_id", "warnings", "run_id"]) {
        expect(got, `${side} 解析器没读到 ${anchor}`).toContain(anchor);
      }
      // ⛔ 下限按今天真实的 18 个字段定,不是照抄隔壁那份的 10。
      expect(
        got.length,
        `${side} 只解析出 ${got.length} 个字段,解析器多半读空了`,
      ).toBeGreaterThan(15);
      expect(new Set(got).size, `${side} 有重复字段名`).toBe(got.length);
    }
  });

  it("字段名集合完全一致 —— 少抄一个,四个消费者会一起静默答「谁都没在跑」", async () => {
    const fs = await importFs();
    expect(new Set(tsFields(fs.readFileSync(TS, "utf8")))).toEqual(
      new Set(rustFields(fs.readFileSync(RS, "utf8"))),
    );
  });

  it("★这个 struct 不许长出 rename_all —— 一长出来上面那条对拍就在比错的东西", async () => {
    const src = (await importFs()).readFileSync(RS, "utf8");
    const head = src.lastIndexOf("#[derive", src.indexOf("pub struct TrainingSnapshot {"));
    const attrs = src.slice(head, src.indexOf("pub struct TrainingSnapshot {"));
    expect(attrs).not.toMatch(/rename_all/);
  });

  it("★`run_id` 必须是**必填**字段 —— 声明成可选就把 tsc 那道免费的闸关掉了", async () => {
    // ⛔ 这一条是这道闸里唯一**不被 tsc 蕴含**的断言,写清楚它买的是什么:
    //
    //    `IDLE_SNAPSHOT`(「什么都没在跑」时四个消费者真正吃到的那个对象)只列**必填**字段 ——
    //    6 个 optional 的按设计不写,那是对的。于是 tsc 只有在字段是必填时才会因为漏写而红。
    //    把 `run_id` 声明成 `run_id?: string`,tsc 立刻放行,而消费者读到的是 `undefined`;
    //    随后每个人都会顺手写 `snap.run_id ?? ""`,而 `""` 在这条边上是【槽根就是那个 run】
    //    这个**肯定事实** ⇒ 未迁移槽的唯一那一行被判成「正在训练」,空闲时也是。
    //    ⇒ 载体这一格上,「可选」不是一个更宽松的选择,是一个把两种含义合并掉的选择。
    //
    // ⚠ 落地时实测过这一条不是空的:把 `run_id: string` 改成 `run_id?: string` ⇒ 本条红,
    //   而 tsc 与其余三条全绿。
    const src = (await importFs()).readFileSync(TS, "utf8");
    const decl = declBody(src, "export interface TrainingSnapshot {");
    expect(decl, "接口里没有 run_id").toMatch(/^ {2}run_id:/m);
    expect(decl, "run_id 被声明成了可选").not.toMatch(/^ {2}run_id\?:/m);

    // 并且它真的落进了那份运行时字面量(而不是只活在声明里)。
    const idle = declBody(src, "export const IDLE_SNAPSHOT: TrainingSnapshot = {");
    const keys = [...idle.matchAll(/^\s{2}([a-z0-9_]+):/gm)].map((m) => m[1]!);
    expect(keys.length, "IDLE_SNAPSHOT 解析读空了").toBeGreaterThan(8);
    // 接口的**必填**字段(可选的按设计不列)必须一个不少地出现在这个字面量里。
    const required = [...decl.matchAll(/^ {2}([a-z0-9_]+):/gm)].map((m) => m[1]!);
    expect(required.length, "必填字段解析读空了").toBeGreaterThan(8);
    expect(
      new Set(keys),
      "IDLE_SNAPSHOT 与接口的必填字段对不上 —— 少一个键,「什么都没在跑」那一档读到的是 undefined",
    ).toEqual(new Set(required));
  });
});
