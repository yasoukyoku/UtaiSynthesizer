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
import { resumeLockedFields, type LockTier } from "./resumeLock";

// 前端 tsconfig 无 @types/node —— 同 ipcParity:用「变量说明符的动态 import」拿 fs
// (TS 对非字面量说明符不做模块解析,运行时由 node 原生解析)。
type NodeFs = { readFileSync(p: string, enc: string): string };
const importFs = (): Promise<NodeFs> => {
  const spec = "node:fs";
  return import(/* @vite-ignore */ spec) as Promise<NodeFs>;
};

const RS = "src-tauri/src/training/resume_lock.rs";
const BACKENDS = ["rvc", "sovits", "sovits_v2", "sovits_diff", "vocoder"];

/** Evaluate the Rust `resume_locked_fields` body by reading its rows and their guard
 *  conditions. Deliberately literal — it parses the shape the function is written in, so a
 *  rewrite into some cleverer form fails LOUDLY here rather than silently passing. */
function rustTable(src: string, backend: string): { id: string; tier: LockTier }[] {
  const body = src.slice(
    src.indexOf("pub fn resume_locked_fields"),
    src.indexOf("/// Everything the guard needs to know"),
  );
  expect(body.length).toBeGreaterThan(200);
  const out: { id: string; tier: LockTier }[] = [];
  // the two unconditional rows (`locked("version", ver_code)` …)
  expect(body).toMatch(/let mut v = vec!\[locked\("version",\s*ver_code\),\s*locked\("sampleRate",\s*ver_code\)\]/);
  out.push({ id: "version", tier: "locked" }, { id: "sampleRate", tier: "locked" });

  // conditional rows: a `backend == "x"` / `matches!(backend, "a" | "b")` block followed by pushes
  const blocks = [...body.matchAll(/(?:if\s+(.+?)\s*\{|"(\w+)"\s*=>\s*\{)([\s\S]*?)\n\s{4}\}/g)];
  for (const [, cond, armBackend, inner] of blocks) {
    const test = armBackend ?? cond ?? "";
    const body2 = inner ?? "";
    let applies: boolean;
    if (armBackend) applies = armBackend === backend;
    else if (test.startsWith("matches!")) {
      applies = [...test.matchAll(/"(\w+)"/g)].some((m) => m[1] === backend);
    } else if (/backend\s*!=\s*"(\w+)"/.test(test)) {
      applies = RegExp.$1 !== backend;
    } else if (/backend\s*==\s*"(\w+)"/.test(test)) {
      applies = RegExp.$1 === backend;
    } else continue;
    if (!applies) continue;
    for (const m of body2.matchAll(/\b(locked|costly)\("(\w+)"/g)) {
      out.push({ id: m[2]!, tier: m[1] === "locked" ? "locked" : "costly" });
    }
  }
  // the unconditional trailing row
  expect(body).toMatch(/v\.push\(costly\("dataset"\)\);/);
  out.push({ id: "dataset", tier: "costly" });
  return out;
}

describe("续训锁 Rust ↔ TS 对拍", () => {
  it("每个后端的锁定字段与档位完全一致", async () => {
    const src = (await importFs()).readFileSync(RS, "utf8");
    for (const b of BACKENDS) {
      const rust = rustTable(src, b);
      const ts = resumeLockedFields(b);
      // order is not part of the contract; the SET of (id, tier) is
      const key = (f: { id: string; tier: LockTier }) => `${f.id}:${f.tier}`;
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
  });
});
