import { describe, it, expect, beforeEach, vi } from "vitest";

// vitest 跑在 node 环境(见 vitest.config.ts),没有 localStorage —— 而 settings.ts 的
// loadSetting/saveSetting 是 try/catch 静默降级的:不补一个 Storage,写入会被无声吞掉、
// 读取永远拿 fallback,这里每条断言都会在测一个空存储,等于什么都没测。
// 只在本文件补,不动 vitest.config.ts(那会影响整个套件的环境)。
class MemoryStorage {
  private map = new Map<string, string>();
  get length(): number { return this.map.size; }
  clear(): void { this.map.clear(); }
  getItem(k: string): string | null { return this.map.get(k) ?? null; }
  setItem(k: string, v: string): void { this.map.set(k, String(v)); }
  removeItem(k: string): void { this.map.delete(k); }
  key(i: number): string | null { return Array.from(this.map.keys())[i] ?? null; }
}
vi.stubGlobal("localStorage", new MemoryStorage());

import {
  listNodeParamPresets,
  saveNodeParamPreset,
  deleteNodeParamPreset,
  MAX_PRESETS_PER_TYPE,
} from "./nodeParamPresets";

const KEY = "utai.nodeParamPresets";

beforeEach(() => {
  localStorage.clear();
});

// 前置自检:上面的 stub 真的生效了吗?如果 settings 还在走降级分支,下面所有用例
// 都会"通过得毫无意义",所以先把这条路验死。
describe("测试前置条件", () => {
  it("内存 Storage 生效 —— 写进去读得回来", () => {
    saveNodeParamPreset("selfcheck", "x", { a: 1 });
    expect(localStorage.getItem(KEY)).not.toBeNull();
    expect(listNodeParamPresets("selfcheck")).toHaveLength(1);
  });
});

describe("saveNodeParamPreset", () => {
  it("存一条之后能按类型读回来", () => {
    expect(saveNodeParamPreset("rvc", "我的音色", { pitch: 2, index: 0.7 })).toBe(true);
    const list = listNodeParamPresets("rvc");
    expect(list).toHaveLength(1);
    expect(list[0]?.name).toBe("我的音色");
    expect(list[0]?.params).toEqual({ pitch: 2, index: 0.7 });
  });

  it("按节点类型分桶 —— 别的类型读不到", () => {
    saveNodeParamPreset("rvc", "A", { pitch: 1 });
    expect(listNodeParamPresets("soVits")).toEqual([]);
  });

  it("同名视为覆盖,不产生重复项", () => {
    saveNodeParamPreset("rvc", "A", { pitch: 1 });
    saveNodeParamPreset("rvc", "A", { pitch: 9 });
    const list = listNodeParamPresets("rvc");
    expect(list).toHaveLength(1);
    expect(list[0]?.params).toEqual({ pitch: 9 });
  });

  it("名称两端空白被裁掉;纯空白名拒绝保存", () => {
    expect(saveNodeParamPreset("rvc", "  A  ", { pitch: 1 })).toBe(true);
    expect(listNodeParamPresets("rvc")[0]?.name).toBe("A");
    expect(saveNodeParamPreset("rvc", "   ", { pitch: 1 })).toBe(false);
    expect(listNodeParamPresets("rvc")).toHaveLength(1);
  });

  it("到达上限后新名字被拒,但同名覆盖仍然允许", () => {
    for (let i = 0; i < MAX_PRESETS_PER_TYPE; i++) {
      expect(saveNodeParamPreset("rvc", `p${i}`, { pitch: i })).toBe(true);
    }
    expect(saveNodeParamPreset("rvc", "overflow", { pitch: 0 })).toBe(false);
    expect(saveNodeParamPreset("rvc", "p0", { pitch: 999 })).toBe(true);
    expect(listNodeParamPresets("rvc")).toHaveLength(MAX_PRESETS_PER_TYPE);
  });

  it("不可序列化的键被丢掉,其余键照常保留", () => {
    // JSON 往返会把函数 / undefined 丢掉。这里刻意不整条拒绝:真实 params 里混进一个
    // undefined 就让用户存不了,比静默丢掉那个键糟糕得多。
    expect(saveNodeParamPreset("rvc", "mixed", { pitch: 3, run: () => {}, gone: undefined })).toBe(true);
    expect(listNodeParamPresets("rvc")[0]?.params).toEqual({ pitch: 3 });
  });

  it("循环引用无法序列化 → 拒绝保存,不写坏存储", () => {
    const circular: Record<string, unknown> = { pitch: 1 };
    circular.self = circular;
    expect(saveNodeParamPreset("rvc", "loop", circular)).toBe(false);
    expect(listNodeParamPresets("rvc")).toEqual([]);
  });
});

describe("listNodeParamPresets", () => {
  it("未存过的类型 → 空数组", () => {
    expect(listNodeParamPresets("rvc")).toEqual([]);
  });

  it("按名称排序,顺序不随保存先后变化", () => {
    saveNodeParamPreset("rvc", "C", {});
    saveNodeParamPreset("rvc", "A", {});
    saveNodeParamPreset("rvc", "B", {});
    expect(listNodeParamPresets("rvc").map((p) => p.name)).toEqual(["A", "B", "C"]);
  });

  it("存储里是坏数据时兜底为空,不抛异常", () => {
    localStorage.setItem(KEY, "{ not json");
    expect(listNodeParamPresets("rvc")).toEqual([]);
    localStorage.setItem(KEY, JSON.stringify([1, 2, 3]));
    expect(listNodeParamPresets("rvc")).toEqual([]);
    localStorage.setItem(KEY, JSON.stringify({ rvc: "nope" }));
    expect(listNodeParamPresets("rvc")).toEqual([]);
  });

  it("桶里混进形状不对的条目时只过滤掉坏的那条", () => {
    localStorage.setItem(KEY, JSON.stringify({ rvc: [{ name: "good", params: { a: 1 }, updatedAt: 1 }, null, { params: {} }] }));
    const list = listNodeParamPresets("rvc");
    expect(list).toHaveLength(1);
    expect(list[0]?.name).toBe("good");
  });
});

describe("deleteNodeParamPreset", () => {
  it("删掉指定项,其余保留", () => {
    saveNodeParamPreset("rvc", "A", { pitch: 1 });
    saveNodeParamPreset("rvc", "B", { pitch: 2 });
    deleteNodeParamPreset("rvc", "A");
    expect(listNodeParamPresets("rvc").map((p) => p.name)).toEqual(["B"]);
  });

  it("删到空之后不在存储里留空桶", () => {
    saveNodeParamPreset("rvc", "A", { pitch: 1 });
    deleteNodeParamPreset("rvc", "A");
    const raw = JSON.parse(localStorage.getItem(KEY) ?? "{}") as Record<string, unknown>;
    expect("rvc" in raw).toBe(false);
  });

  it("删不存在的名字是无操作,不影响别的类型", () => {
    saveNodeParamPreset("soVits", "keep", { a: 1 });
    deleteNodeParamPreset("rvc", "ghost");
    deleteNodeParamPreset("soVits", "ghost");
    expect(listNodeParamPresets("soVits").map((p) => p.name)).toEqual(["keep"]);
  });
});
