/**
 * 续训锁的跨语言对拍 —— same trick as `ipcParity`: read the Rust source and require the TS
 * mirror to describe exactly the same fields.
 *
 * Why it has to exist: the AUTHORITY is `resume_lock.rs` (it is what `try_start` enforces), but
 * the parameters page must know the answer without an `invoke`, so a mirror is unavoidable.
 * A mirror nobody checks is just a second opinion — and the failure is silent in both
 * directions: a field the UI thinks is free but the guard refuses (a dialog promising 续训 and
 * a start that rejects it), or a field the UI locks that the guard would have allowed (an
 * editable knob rendered dead, with no way to discover why).
 *
 * The Rust side already drives its table THROUGH its guard (`every_locked_field_refuses_and_
 * every_costly_field_does_not`), so pinning the mirror to the table pins it to the guard.
 */
import { describe, expect, it } from "vitest";
import {
  resumeLockedFields,
  resumeWouldBeGuarded,
  type LockedField,
  type LockScope,
} from "./resumeLock";

// 前端 tsconfig 无 @types/node —— 同 ipcParity:用「变量说明符的动态 import」拿 fs
// (TS 对非字面量说明符不做模块解析,运行时由 node 原生解析)。
type NodeFs = { readFileSync(p: string, enc: string): string };
const importFs = (): Promise<NodeFs> => {
  const spec = "node:fs";
  return import(/* @vite-ignore */ spec) as Promise<NodeFs>;
};

const RS = "src-tauri/src/training/resume_lock.rs";
const BACKENDS = ["rvc", "sovits", "sovits_v2", "sovits_diff", "vocoder"];

/** 去掉 Rust 一行里的注释,**但不碰字符串字面量里的 `//`**。
 *
 *  ⛔ 这不是整洁,是承重的:一条内容含 `locked("x", …)` 字样的注释会**凭空造出一行**,
 *  于是这道闸开始比较一个不存在的规则。S127 在源序棘轮上真踩过这个形状。 */
function stripComment(line: string): string {
  let inStr = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i]!;
    if (c === '"' && line[i - 1] !== "\\") inStr = !inStr;
    else if (!inStr && c === "/" && line[i + 1] === "/") return line.slice(0, i);
  }
  return line;
}

type Frame = { applies: boolean | "unknown"; what: string };

/**
 * 逐行走 Rust 的 `resume_locked_fields`,带一个条件栈,按「这一行处在哪个条件里」判归属。
 *
 * ⛔ 上一版是「两条硬编的开头行 + 一个抓条件块的正则 + 两条硬编的结尾行」。它有两个**实测**
 * 盲区,而 ④d 恰好同时踩中两个:
 * ⒜ 在**顶层**新增一条无条件的 `v.push(...)` —— 它完全看不见(开头和结尾都是抄进来的);
 * ⒝ 把 `augCopies` / `dataset` 从无条件改成按 family 条件化 —— 它照样报成无条件。
 * 现在没有任何一行是抄进来的:每一条 row 都由它当时所处的条件栈决定。
 *
 * ⛔ 它仍然**故意死板**:只认这个函数今天写成的那几种行形状。任何一种它不认识的改写都会
 * `expect.fail` 而不是少读几行 —— 少读几行的表现恰恰是「两个集合恰好相等」,也就是全绿。
 */
function rustTable(src: string, backend: string): LockedField[] {
  const body = src.slice(
    src.indexOf("pub fn resume_locked_fields"),
    src.indexOf("/// Everything the guard needs to know"),
  );
  expect(body.length).toBeGreaterThan(200);

  const out: LockedField[] = [];
  const stack: Frame[] = [];
  const evalCond = (cond: string): boolean | "unknown" => {
    let m = /^matches!\(\s*backend\s*,(.+)\)$/.exec(cond);
    if (m) return [...m[1]!.matchAll(/"(\w+)"/g)].some((x) => x[1] === backend);
    m = /^backend\s*==\s*"(\w+)"$/.exec(cond);
    if (m) return m[1] === backend;
    m = /^backend\s*!=\s*"(\w+)"$/.exec(cond);
    if (m) return m[1] !== backend;
    return "unknown";
  };

  for (const raw of body.split("\n")) {
    const t = stripComment(raw).trim();
    if (!t) continue;

    // ① 这一行上的 row —— 按**当前**条件栈计。
    for (const m of t.matchAll(/\b(locked|costly)\(\s*"(\w+)"([^)]*)\)/g)) {
      if (stack.some((f) => f.applies === "unknown")) {
        expect.fail(`解析器读不懂这个条件,却在它里面撞见了一行 row:${stack.map((f) => f.what).join(" / ")}`);
      }
      if (!stack.every((f) => f.applies === true)) continue;
      const sm = /LockScope::(\w+)/.exec(m[3] ?? "");
      expect(sm, `${m[2]}: scope 必须是一个 LockScope:: 字面量(算出来的变量这里读不出)`).toBeTruthy();
      out.push({
        id: m[2]!,
        tier: m[1] === "locked" ? "locked" : "costly",
        scope: sm![1]!.toLowerCase() as LockScope,
      });
    }

    // ② 括号 —— 只认这个函数今天写成的那几种形状。
    if (/^pub fn resume_locked_fields/.test(t)) {
      stack.push({ applies: true, what: "fn" });
      continue;
    }
    if (/^\}\s*else\s*\{$/.test(t)) {
      stack.pop();
      // else 分支的条件是「上一个的反面」——今天没有一条 row 住在里面,所以不去解它;
      // 真有人往里放一行,上面那句 `expect.fail` 会当场说出来。
      stack.push({ applies: "unknown", what: "else" });
      continue;
    }
    if (/^\}[;,]?$/.test(t)) {
      expect(stack.length, "多余的 }").toBeGreaterThan(0);
      stack.pop();
      continue;
    }
    let m = /^(?:let\s+\w+\s*=\s*)?if\s+(.+?)\s*\{$/.exec(t);
    if (m) {
      stack.push({ applies: evalCond(m[1]!.trim()), what: m[1]! });
      continue;
    }
    if (/^match\s+backend\s*\{$/.test(t)) {
      stack.push({ applies: true, what: "match" });
      continue;
    }
    m = /^"(\w+)"\s*=>\s*\{$/.exec(t);
    if (m) {
      stack.push({ applies: m[1] === backend, what: `arm ${m[1]}` });
      continue;
    }
    if (/^_\s*=>\s*\{\}$/.test(t)) continue;
    const bal = (t.match(/\{/g)?.length ?? 0) - (t.match(/\}/g)?.length ?? 0);
    expect(bal, `解析器不认识这一行的括号,它会读错归属:${t}`).toBe(0);
  }
  expect(stack.length, "条件栈没有收干净 —— 解析器与源码的形状对不上了").toBe(0);
  return out;
}

describe("续训锁 Rust ↔ TS 对拍", () => {
  it("每个后端的锁定字段、档位与 scope 完全一致", async () => {
    const src = (await importFs()).readFileSync(RS, "utf8");
    for (const b of BACKENDS) {
      const rust = rustTable(src, b);
      const ts = resumeLockedFields(b);
      // order is not part of the contract; the SET of (id, tier, scope) is.
      // ★ scope 进这把钥匙是 ④d 笔 1 加的:少了它,「两侧都有 loudnorm 但一侧说它是 run 级」
      // 这种漂移是全绿的,而参数页的代价提示与 ④d 的池身份不变量都读那一列。
      const key = (f: LockedField) => `${f.id}:${f.tier}:${f.scope}`;
      expect(new Set(ts.map(key)), `backend ${b}`).toEqual(new Set(rust.map(key)));
      expect(ts.length, `backend ${b} has a duplicate row`).toBe(new Set(ts.map(key)).size);
    }
  });

  it("自检:解析真的读到了东西(否则上面的相等是两个空集)", async () => {
    const src = (await importFs()).readFileSync(RS, "utf8");
    // sovits is the richest backend — if the parser silently matched nothing, this fails
    const rust = rustTable(src, "sovits");
    expect(rust.map((f) => f.id).sort()).toEqual([
      "augCopies",
      "dataset",
      "loudnorm",
      "sampleRate",
      "speakerCount",
      "speakerSet",
      "version",
      "volEmbedding",
    ]);
    expect(rustTable(src, "vocoder").map((f) => f.id).sort()).toEqual([
      "augCopies",
      "dataset",
      "sampleRate",
      "version",
    ]);
    // ★ scope 也要有存活证明:一个把 scope 一律读成 "run" 的解析器,会让上面那条相等
    // 变成「两侧都错得一模一样」。这三条各代表一种 scope。
    const scopeOf = (rows: LockedField[], id: string) => rows.find((f) => f.id === id)?.scope;
    expect(scopeOf(rust, "version")).toBe("both");
    expect(scopeOf(rust, "volEmbedding")).toBe("run");
    expect(scopeOf(rust, "loudnorm")).toBe("pool");
    expect(scopeOf(rustTable(src, "rvc"), "version")).toBe("run");
  });

  it("★条件化一行、或在顶层新增一行,解析器都必须看得见(上一版这两条是瞎的)", async () => {
    const src = (await importFs()).readFileSync(RS, "utf8");
    // ⒜ 顶层新增一条无条件 row ⇒ 每个 backend 都要多出它。
    const added = src.replace(
      'v.push(costly("dataset", LockScope::Pool));',
      'v.push(costly("dataset", LockScope::Pool));\n    v.push(costly("zzzProbe", LockScope::Run));',
    );
    expect(added, "锚点没命中,这条自检什么也没测").not.toBe(src);
    for (const b of BACKENDS) {
      expect(rustTable(added, b).map((f) => f.id)).toContain("zzzProbe");
    }
    // ⒝ 把一条无条件 row 条件化 ⇒ 只有那个 family 还有它。
    const gated = src.replace(
      'v.push(costly("augCopies", LockScope::Pool));',
      'if backend == "rvc" {\n        v.push(costly("augCopies", LockScope::Pool));\n    }',
    );
    expect(gated).not.toBe(src);
    expect(rustTable(gated, "rvc").map((f) => f.id)).toContain("augCopies");
    for (const b of BACKENDS.filter((x) => x !== "rvc")) {
      expect(rustTable(gated, b).map((f) => f.id), b).not.toContain("augCopies");
    }
    // ⒞ 一条**内容恰好是一行 row** 的整行注释,不许被读成一行。
    const decoy = src.replace(
      'v.push(costly("dataset", LockScope::Pool));',
      '// v.push(costly("zzzDecoy", LockScope::Run));\n    v.push(costly("dataset", LockScope::Pool));',
    );
    expect(decoy).not.toBe(src);
    expect(rustTable(decoy, "sovits").map((f) => f.id)).not.toContain("zzzDecoy");
  });
});

describe("resumeWouldBeGuarded —— 镜像后端 check_resume_locks(审查 S78)", () => {
  type SlotLike = {
    exists: boolean;
    has_main_progress: boolean;
    diff_steps: number;
    version: string;
    sample_rate: string;
  };
  const slot = (over: Partial<SlotLike>): SlotLike => ({
    exists: true,
    has_main_progress: false,
    diff_steps: 0,
    version: "",
    sample_rate: "",
    ...over,
  });

  it("空槽 / 不存在的槽不锁", () => {
    expect(resumeWouldBeGuarded("rvc", null)).toBe(false);
    expect(
      resumeWouldBeGuarded("rvc", slot({ exists: false, version: "v2", sample_rate: "40k" })),
    ).toBe(false);
  });

  it("★预处理阶段停训(manifest 有版本、无 checkpoint)仍锁 —— 否则 UI 让改、后端拒", () => {
    // 后端守卫看 manifest 的 version/sample_rate,而 manifest 早于 worker 写入;
    // has_main_progress=false 的窗口若不锁,用户改了 sampleRate 选续训就会被 RESUME_PARAMS_MISMATCH 拒。
    expect(
      resumeWouldBeGuarded("rvc", slot({ has_main_progress: false, version: "v2", sample_rate: "40k" })),
    ).toBe(true);
    expect(
      resumeWouldBeGuarded("sovits", slot({ has_main_progress: false, version: "4.1", sample_rate: "44k" })),
    ).toBe(true);
  });

  it("已有 checkpoint 的槽当然锁", () => {
    expect(
      resumeWouldBeGuarded("rvc", slot({ has_main_progress: true, version: "v2", sample_rate: "40k" })),
    ).toBe(true);
  });

  it("有目录但 manifest 无版本(pre-S37 / 只建了 dataset)不锁 —— 与后端 fail-open 一致", () => {
    expect(resumeWouldBeGuarded("rvc", slot({ exists: true }))).toBe(false);
  });

  it("sovits_diff 用 diff_steps 而非 manifest 版本(版本由主模型钉)", () => {
    expect(
      resumeWouldBeGuarded("sovits_diff", slot({ diff_steps: 0, version: "4.1", sample_rate: "44k" })),
    ).toBe(false);
    expect(
      resumeWouldBeGuarded("sovits_diff", slot({ diff_steps: 12, version: "4.1", sample_rate: "44k" })),
    ).toBe(true);
  });
});
