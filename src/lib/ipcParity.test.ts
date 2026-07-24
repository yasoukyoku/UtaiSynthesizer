/**
 * IPC 契约对拍(S76 批 4 新增)。
 *
 * Tauri 的 `invoke` 是**字符串路由**:改一个命令名、删一个命令、漏改一个调用点,tsc 和
 * cargo 都一个字都不会说,直到用户在真窗口上点到那颗按钮才炸。S76 批 4 一口气动了六个调用点
 * 和四条命令(两条删除、两条改签名),这条 gate 就是那次改动的回归钉。
 *
 * 两个方向都要钉:
 *  · 前端调了后端没有 ⇒ 运行时 "Command xxx not found";
 *  · 后端注册了没人调 ⇒ 死命令。批 3 就这样留下了 `delete_training_workspace` —— 一条守卫比
 *    替代者弱一档、只能由 IPC 名字触达的**破坏性**命令,在仓库里躺了整整一批。
 */
import { describe, it, expect } from "vitest";

// 前端 tsconfig 无 @types/node(include=src 全量过 tsc)。用「变量说明符的动态 import」拿
// fs/path —— TS 对非字面量说明符不做模块解析(无 TS2307),运行时由 node 原生解析。
// (同款写法见 src/lib/vocal/e1CrossDump.test.ts)
type NodeFs = {
  readFileSync(p: string, enc: string): string;
  readdirSync(p: string, o: { withFileTypes: true }): { name: string; isDirectory(): boolean }[];
};
const importFs = (): Promise<NodeFs> => {
  const spec = "node:fs";
  return import(/* @vite-ignore */ spec) as Promise<NodeFs>;
};

const SRC = "src";
const LIB_RS = "src-tauri/src/lib.rs";

/** 把注释挖空(保留长度与换行,行号不变)。必须先做这一步:仓库里有好几处注释在讨论
 *  「某某命令」并原样写出调用形式,直接扫源码会把注释里的名字当成真调用 —— 那会让这条
 *  gate 报假红(比漏报更糟:假红的 gate 迟早被人关掉)。
 *
 *  逐字符走,同时跟踪字符串/模板/正则以外的状态;不处理正则字面量里的 `//`,那在本仓不出现。 */
function stripComments(src: string): string {
  const out = src.split("");
  let i = 0;
  const blank = (from: number, to: number) => {
    for (let k = from; k < to && k < out.length; k++) if (out[k] !== "\n") out[k] = " ";
  };
  while (i < src.length) {
    const c = src[i];
    const d = src[i + 1];
    if (c === '"' || c === "'" || c === "`") {
      const q = c;
      i++;
      while (i < src.length && src[i] !== q) {
        if (src[i] === "\\") i++;
        i++;
      }
      i++;
    } else if (c === "/" && d === "/") {
      let j = i;
      while (j < src.length && src[j] !== "\n") j++;
      blank(i, j);
      i = j;
    } else if (c === "/" && d === "*") {
      let j = i + 2;
      while (j < src.length && !(src[j] === "*" && src[j + 1] === "/")) j++;
      blank(i, Math.min(j + 2, src.length));
      i = j + 2;
    } else {
      i++;
    }
  }
  return out.join("");
}

/** 前端调用点。**不能**用惰性泛型正则:`invoke` 后面跟跨行/嵌套泛型时,`<[\s\S]*?>` 会一路
 *  吃到很远的地方,把中间的调用点整段吞掉(实测能一次漏报好几条)。所以从 `invoke` 往后按
 *  尖括号配平跳过泛型参数,再取紧跟其后的字符串字面量。 */
function collectInvoked(src: string): string[] {
  const out: string[] = [];
  for (let i = src.indexOf("invoke"); i >= 0; i = src.indexOf("invoke", i + 1)) {
    // 必须是独立标识符(别把 reinvoke / invokeLater 认成它)。**成员访问要认**:
    // `core.invoke("x")` 是完全合法的写法,把 `.` 算进标识符字符会让整条调用消失
    // —— 那正是这条 gate 假绿的方式(它调了不存在的命令也查不出来)。
    const before = src[i - 1] ?? "";
    if (/[A-Za-z0-9_$]/.test(before)) continue;
    let j = i + "invoke".length;
    if (/[A-Za-z0-9_$]/.test(src[j] ?? "")) continue;
    while (/\s/.test(src[j] ?? "")) j++;
    if (src[j] === "<") {
      let depth = 0;
      for (; j < src.length; j++) {
        if (src[j] === "<") depth++;
        else if (src[j] === ">") {
          // 箭头函数的 `>` 不是泛型闭合:`invoke<Record<string, () => void>>("x")` 会被算成
          // 多闭了一层,于是扫描点落在字符串之后,整条调用被漏掉。
          if (src[j - 1] === "=") continue;
          depth--;
          if (depth === 0) {
            j++;
            break;
          }
        }
      }
      while (/\s/.test(src[j] ?? "")) j++;
    }
    if (src[j] !== "(") continue;
    // 取**第一个实参**里的全部字符串字面量,而不是「紧跟其后必须是一个字面量」:命令名有时是
    // 三元表达式(engine.ts 的 `isRvc ? "run_rvc" : "run_sovits"`、exitFlow.ts 的
    // `mode === "restart" ? "restart_app" : "quit_app"`)。只认第一个字面量会把这四条真命令
    // 判成「注册了没人调」—— 我第一版就是这么误报的。
    // (命令名若是一个变量,这里查不出来;那种写法本仓没有,出现了就得在这里补。)
    j++;
    let depth = 0;
    let prev = "("; // 上一个有意义的字符 —— 字面量只有落在「命令位」才算数
    for (; j < src.length; j++) {
      const c = src[j];
      if (/\s/.test(c ?? "")) continue;
      if (c === "(" || c === "[" || c === "{") {
        depth++;
        prev = c;
      } else if (c === ")" || c === "]" || c === "}") {
        if (depth === 0) break;
        depth--;
        prev = c;
      } else if (c === "," && depth === 0) {
        break;
      } else if (c === '"' || c === "'" || c === "`") {
        const q = c;
        let k = j + 1;
        while (k < src.length && src[k] !== q) {
          if (src[k] === "\\") k++;
          k++;
        }
        // 命令位 = 实参开头,或三元的某个分支。`mode === "restart" ? …` 里的 "restart" 是
        // 比较用的字面量,不是命令名 —— 不加这道判断就会报「restart 未注册」的假红。
        if (prev === "(" || prev === "?" || prev === ":") {
          const name = src.slice(j + 1, k);
          if (/^[a-z_][a-z0-9_]*$/.test(name)) out.push(name);
        }
        j = k;
        prev = '"';
      } else {
        prev = c ?? "";
      }
    }
  }
  return out;
}

/** lib.rs 的 `generate_handler![...]` 里注册的命令名(路径末段)。 */
function collectRegistered(src: string): string[] {
  const start = src.indexOf("generate_handler![");
  if (start < 0) throw new Error("generate_handler! not found in lib.rs");
  let depth = 0;
  let end = start;
  for (let i = src.indexOf("[", start); i < src.length; i++) {
    if (src[i] === "[") depth++;
    else if (src[i] === "]") {
      depth--;
      if (depth === 0) {
        end = i;
        break;
      }
    }
  }
  return src
    .slice(start, end)
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s && !s.startsWith("//"))
    .map((s) => s.split("::").pop()!.trim())
    .filter((s) => /^[a-z_][a-z0-9_]*$/.test(s));
}

describe("IPC 契约", () => {
  /** 扫描器自己的回归钉。这三种写法本仓现在都有(或随时会有),而**每一种都曾经被整条吞掉**
   *  —— 一条扫不到东西的 gate 会安静地全绿。 */
  it("扫描器认得出真实世界里的 invoke 写法", () => {
    const cases: [string, string[]][] = [
      ['await invoke("plain_one", { a: 1 });', ["plain_one"]],
      ['invoke<Foo>("with_generic")', ["with_generic"]],
      ['invoke<Array<{ kind: string }>>("nested_generic")', ["nested_generic"]],
      ['invoke<Record<string, () => void>>("arrow_in_generic", {})', ["arrow_in_generic"]],
      ['core.invoke("member_form")', ["member_form"]],
      ['invoke(isRvc ? "ternary_a" : "ternary_b", {})', ["ternary_a", "ternary_b"]],
      ['invoke(mode === "restart" ? "cmd_a" : "cmd_b")', ["cmd_a", "cmd_b"]],
      ['reinvoke("not_a_command")', []],
      ['invokeLater("not_a_command")', []],
      ['invoke(dynamicName, {})', []],
    ];
    for (const [src, want] of cases) {
      expect({ src, got: collectInvoked(stripComments(src)) }).toEqual({ src, got: want });
    }
    // 注释里写出的调用形式**不能**算数,否则这条 gate 会因为一句说明而报假红
    expect(collectInvoked(stripComments('// see invoke("documented_only")\nconst a = 1;'))).toEqual([]);
    expect(collectInvoked(stripComments('/* invoke("in_block_comment") */'))).toEqual([]);
    // …但字符串里的 `//` 不是注释,挖空它会把后面的真代码一起吃掉
    expect(collectInvoked(stripComments('const u = "https://x"; invoke("after_url");'))).toEqual([
      "after_url",
    ]);
  });

  it("前端 invoke 的每个命令都在 lib.rs 注册,且没有无人调用的注册命令", async () => {
    const fs = await importFs();
    const files: string[] = [];
    const scan = (dir: string) => {
      for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
        const p = `${dir}/${e.name}`;
        if (e.isDirectory()) scan(p);
        // 测试文件不是 app 代码(这个文件自己就在注释里写了 invoke 的形状)
        else if (/\.tsx?$/.test(e.name) && !/\.test\.tsx?$/.test(e.name)) files.push(p);
      }
    };
    scan(SRC);

    const invokedBy = new Map<string, string[]>();
    for (const f of files) {
      for (const name of collectInvoked(stripComments(fs.readFileSync(f, "utf8")))) {
        if (!invokedBy.has(name)) invokedBy.set(name, []);
        invokedBy.get(name)!.push(f);
      }
    }
    const registered = new Set(collectRegistered(fs.readFileSync(LIB_RS, "utf8")));

    // 自检:两边都不该是空的,否则这条 gate 会在什么都没扫到的情况下「全绿」
    expect(registered.size).toBeGreaterThan(50);
    expect(invokedBy.size).toBeGreaterThan(50);

    const notRegistered = [...invokedBy.keys()].filter((n) => !registered.has(n)).sort();
    expect(notRegistered).toEqual([]);

    const neverInvoked = [...registered].filter((n) => !invokedBy.has(n)).sort();
    expect(neverInvoked).toEqual([]);
  });
});
