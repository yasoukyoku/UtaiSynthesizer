/**
 * 三语平行铁律的机器把关(S76 批 4 新增)。
 *
 * 在此之前这条铁律**完全靠人工**:仓库里没有任何 i18n 校验脚本、没有 CI,vitest 的其它测试
 * 一律把 i18n mock 掉,而 i18next 没做 TypeScript 类型增强 —— 于是「漏一语」「拼错 key」
 * 「插值占位符只改了一种语言」全都零报警,只会在界面上原样显示 key 字符串或吐出 `{{name}}`。
 * 每次发版前手工跑一段 node 一次性脚本不是纪律,是运气。
 *
 * 这里只钉**结构**,不碰译文内容:
 *  1. 三语叶子键集合完全相同(键数相同还不够——同数不同集是真实会发生的);
 *  2. 同一个 key 的 `{{占位符}}` 集合三语一致(少一个 = 界面上少一半信息,多一个 = 直接漏出大括号);
 *  3. 每个 key 的值都是非空字符串(空串会静默显示成空白,比缺键更难查);
 *  4. `backend.*` 与 backendError.ts 的 CODE 表双向闭合(CODE 有条目却无文案 = 用户看到裸 CODE;
 *     文案无 CODE = 死键)。这条闭合性在 S76 勘查里被确认当前成立,值得保持。
 */
import { describe, it, expect } from "vitest";
import zh from "./zh.json";
import en from "./en.json";
import ja from "./ja.json";
import { CODE_KEYS } from "../lib/backendError";

type Tree = { [k: string]: string | Tree };

/** 叶子路径 → 值。i18n JSON 是嵌套对象(不是扁平点号 key),所以必须递归展开。 */
function leaves(t: Tree, prefix = ""): Map<string, string> {
  const out = new Map<string, string>();
  for (const [k, v] of Object.entries(t)) {
    if (v && typeof v === "object") {
      for (const [ik, iv] of leaves(v as Tree, `${prefix}${k}.`)) out.set(ik, iv);
    } else {
      out.set(`${prefix}${k}`, v as string);
    }
  }
  return out;
}

/** i18next 的插值占位符。Settings.tsx 的内联 L() 表用的是单花括号 `{name}` + 手工 replace,
 *  那是另一个域,不归这里管——只认双花括号,免得把 UI 文案里的普通括号当成占位符。 */
function placeholders(s: string): string[] {
  return [...s.matchAll(/\{\{\s*([A-Za-z0-9_]+)[^}]*\}\}/g)].map((m) => m[1]!).sort();
}

const LANGS: [string, Map<string, string>][] = [
  ["zh", leaves(zh as Tree)],
  ["en", leaves(en as Tree)],
  ["ja", leaves(ja as Tree)],
];

describe("i18n 三语平行", () => {
  it("三语的叶子键集合完全相同", () => {
    const [[, base]] = LANGS as [[string, Map<string, string>]];
    const baseKeys = [...base.keys()].sort();
    for (const [lang, m] of LANGS.slice(1)) {
      const keys = [...m.keys()].sort();
      const missing = baseKeys.filter((k) => !m.has(k));
      const extra = keys.filter((k) => !base.has(k));
      expect({ lang, missing, extra }).toEqual({ lang, missing: [], extra: [] });
    }
  });

  it("同一个 key 的插值占位符三语一致", () => {
    const [[, base]] = LANGS as [[string, Map<string, string>]];
    const bad: string[] = [];
    for (const [key, zhVal] of base) {
      const want = placeholders(zhVal).join(",");
      for (const [lang, m] of LANGS.slice(1)) {
        const v = m.get(key);
        if (v === undefined) continue; // 上一条测试负责报缺键
        const got = placeholders(v).join(",");
        if (got !== want) bad.push(`${key}: zh[${want}] vs ${lang}[${got}]`);
      }
    }
    expect(bad).toEqual([]);
  });

  it("没有写成单花括号的占位符", () => {
    // 本仓有**两套**插值语法并存:i18n JSON 走 i18next 的 {{name}},而 Settings.tsx 的内联
    // L() 表走单花括号 {name} + 手工 replace。写混了两边都不报错 —— i18next 会把 {name}
    // 原样显示给用户,replace 则永远匹配不上。这条只管 JSON 这一侧。
    const single = /(?<!\{)\{\s*[A-Za-z0-9_]+\s*\}(?!\})/;
    const bad: string[] = [];
    for (const [lang, m] of LANGS) {
      for (const [key, v] of m) {
        if (single.test(v)) bad.push(`${lang}:${key} = ${v}`);
      }
    }
    expect(bad).toEqual([]);
  });

  it("没有空文案", () => {
    const bad: string[] = [];
    for (const [lang, m] of LANGS) {
      for (const [key, v] of m) {
        if (typeof v !== "string" || v.trim() === "") bad.push(`${lang}:${key}`);
      }
    }
    expect(bad).toEqual([]);
  });

  it("backendError.ts 的每个 CODE 都指向一条真实存在的文案", () => {
    // 注意:CODE 的文案 key **不一定**是 `backend.<CODE>` —— 有一批刻意复用了别处的既有文案
    // (APP_BUSY → common.busyRetry 等),所以只能按条目自己声明的 key 校验。按前缀猜会得到
    // 一份假的「缺失」清单。
    const [[, base]] = LANGS as [[string, Map<string, string>]];
    const dangling = Object.entries(CODE_KEYS)
      .filter(([, e]) => !base.has(e.key))
      .map(([code, e]) => `${code} → ${e.key}`)
      .sort();
    expect(dangling).toEqual([]);
  });

  it("backend.* 里没有无人引用的死键", () => {
    const used = new Set(Object.values(CODE_KEYS).map((e) => e.key));
    const [[, base]] = LANGS as [[string, Map<string, string>]];
    const dead = [...base.keys()].filter((k) => k.startsWith("backend.") && !used.has(k)).sort();
    expect(dead).toEqual([]);
  });
});
