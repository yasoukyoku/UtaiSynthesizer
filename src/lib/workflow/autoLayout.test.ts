import { describe, it, expect } from "vitest";
import { computeAutoLayout, COLUMN_GAP, ROW_GAP, COMPONENT_GAP } from "./autoLayout";
import type { LayoutNodeInput, LayoutEdgeInput } from "./autoLayout";

const W = 240;
const H = 120;
const PITCH = W + COLUMN_GAP;

function n(id: string, x = 0, y = 0, measured?: { width?: number; height?: number }): LayoutNodeInput {
  return { id, type: "gain", position: { x, y }, ...(measured ? { measured } : {}) };
}
function e(source: string, target: string): LayoutEdgeInput {
  return { source, target };
}
function at(
  r: ReturnType<typeof computeAutoLayout>,
  id: string,
): { x: number; y: number } {
  const p = r.positions.get(id);
  if (!p) throw new Error(`no position for ${id}`);
  return p;
}

describe("computeAutoLayout · 分层", () => {
  it("链式图排成从左到右的列,列距与模板一致", () => {
    const r = computeAutoLayout([n("a"), n("b"), n("c")], [e("a", "b"), e("b", "c")]);
    expect(at(r, "a")).toEqual({ x: 0, y: 0 });
    expect(at(r, "b")).toEqual({ x: PITCH, y: 0 });
    expect(at(r, "c")).toEqual({ x: PITCH * 2, y: 0 });
    expect(PITCH).toBe(320);
  });

  it("用最长路径定层:捷径边不会把汇聚节点拉回前面一列", () => {
    // a→b→c 同时 a→c。c 必须在第 2 列(最长路径),不是第 1 列。
    const r = computeAutoLayout([n("a"), n("b"), n("c")], [e("a", "b"), e("b", "c"), e("a", "c")]);
    expect(at(r, "c").x).toBe(PITCH * 2);
  });

  it("同层多个节点按实测高度往下堆,不重叠", () => {
    const r = computeAutoLayout(
      [n("src"), n("t1", 0, 0, { width: W, height: 300 }), n("t2", 0, 999, { width: W, height: 80 })],
      [e("src", "t1"), e("src", "t2")],
    );
    const t1 = at(r, "t1");
    const t2 = at(r, "t2");
    expect(t1.x).toBe(PITCH);
    expect(t2.x).toBe(PITCH);
    // t1 高 300 → t2 必须落在 300 + ROW_GAP 之后,而不是固定行距。
    expect(t2.y).toBe(t1.y + 300 + ROW_GAP);
  });

  it("列宽取该列最宽的节点,宽卡片不会压到下一列", () => {
    const r = computeAutoLayout(
      [n("a", 0, 0, { width: 280, height: H }), n("b")],
      [e("a", "b")],
    );
    expect(at(r, "b").x).toBe(280 + COLUMN_GAP);
  });

  it("同列顺序按上游 y 的重心排,保留用户原有的上下关系", () => {
    // t2 的上游在上面、t1 的上游在下面 → 整理后 t2 应该排在 t1 上面。
    const r = computeAutoLayout(
      [n("hi", 0, 0), n("lo", 0, 500), n("t1"), n("t2")],
      [e("lo", "t1"), e("hi", "t2")],
    );
    expect(at(r, "t2").y).toBeLessThan(at(r, "t1").y);
  });

  it("参考 y 相同时退回原数组顺序 —— 结果稳定可复现", () => {
    const nodes = [n("src"), n("x"), n("y"), n("z")];
    const edges = [e("src", "x"), e("src", "y"), e("src", "z")];
    const a = computeAutoLayout(nodes, edges);
    const b = computeAutoLayout(nodes, edges);
    expect(at(a, "x").y).toBeLessThan(at(a, "y").y);
    expect(at(a, "y").y).toBeLessThan(at(a, "z").y);
    expect([...a.positions]).toEqual([...b.positions]);
  });
});

describe("computeAutoLayout · 容错(parseWorkflowGraph 会抛错的那些中间态)", () => {
  it("空图返回空结果,不抛错", () => {
    const r = computeAutoLayout([], []);
    expect(r.positions.size).toBe(0);
    expect(r.moved).toBe(0);
  });

  it("成环不抛错,每个节点仍然拿到坐标", () => {
    const r = computeAutoLayout([n("a"), n("b"), n("c")], [e("a", "b"), e("b", "c"), e("c", "a")]);
    expect(r.positions.size).toBe(3);
    for (const id of ["a", "b", "c"]) {
      expect(Number.isFinite(at(r, id).x)).toBe(true);
      expect(Number.isFinite(at(r, id).y)).toBe(true);
    }
  });

  it("半截连线(一端已删)被忽略,不影响其余节点分层", () => {
    const r = computeAutoLayout([n("a"), n("b")], [e("a", "b"), e("ghost", "b"), e("a", "ghost")]);
    expect(r.positions.size).toBe(2);
    expect(at(r, "b").x).toBe(PITCH);
  });

  it("自环被忽略,节点留在第 0 列", () => {
    const r = computeAutoLayout([n("a")], [e("a", "a")]);
    expect(at(r, "a")).toEqual({ x: 0, y: 0 });
  });

  it("重复连线不会把节点越推越远", () => {
    const r = computeAutoLayout([n("a"), n("b")], [e("a", "b"), e("a", "b"), e("a", "b")]);
    expect(at(r, "b").x).toBe(PITCH);
  });

  it("没有输入/输出节点的裸图也能整理(parseWorkflowGraph 在这会抛错)", () => {
    const r = computeAutoLayout([n("g1"), n("g2")], [e("g1", "g2")]);
    expect(r.positions.size).toBe(2);
  });
});

describe("computeAutoLayout · 连通分量", () => {
  it("互不相连的子图上下分开,不在同一列里交错", () => {
    const r = computeAutoLayout(
      [n("a1"), n("a2"), n("b1"), n("b2")],
      [e("a1", "a2"), e("b1", "b2")],
    );
    // 两张子图的第 0 列都在 x=0,但 y 必须分开。
    expect(at(r, "a1").x).toBe(0);
    expect(at(r, "b1").x).toBe(0);
    expect(at(r, "b1").y).toBe(at(r, "a1").y + H + COMPONENT_GAP);
  });

  it("孤立节点也参与排布,不会被丢在原地", () => {
    const r = computeAutoLayout([n("a"), n("b"), n("lonely", 9999, 9999)], [e("a", "b")]);
    expect(r.positions.has("lonely")).toBe(true);
    expect(at(r, "lonely").x).toBe(0);
    expect(at(r, "lonely").y).toBeGreaterThan(at(r, "a").y);
  });
});

describe("computeAutoLayout · moved 计数(调用方据此跳过无意义写入)", () => {
  it("已经是整理后的图 → moved 为 0", () => {
    const first = computeAutoLayout([n("a"), n("b")], [e("a", "b")]);
    const settled = [
      n("a", at(first, "a").x, at(first, "a").y),
      n("b", at(first, "b").x, at(first, "b").y),
    ];
    expect(computeAutoLayout(settled, [e("a", "b")]).moved).toBe(0);
  });

  it("有节点要动 → moved 只数真的动了的那些", () => {
    const r = computeAutoLayout([n("a"), n("b", 77, 88)], [e("a", "b")]);
    expect(r.moved).toBe(1);
  });

  it("重复整理是幂等的", () => {
    const nodes = [n("a", 10, 20), n("b", 500, 30), n("c", 5, 600)];
    const edges = [e("a", "b"), e("b", "c")];
    const one = computeAutoLayout(nodes, edges);
    const applied = nodes.map((x) => n(x.id, at(one, x.id).x, at(one, x.id).y));
    const two = computeAutoLayout(applied, edges);
    expect(two.moved).toBe(0);
    expect([...two.positions]).toEqual([...one.positions]);
  });
});

describe("computeAutoLayout · 纯函数 / 起点", () => {
  it("不改传进来的节点数组", () => {
    const nodes = [n("a", 11, 22), n("b", 33, 44)];
    computeAutoLayout(nodes, [e("a", "b")]);
    expect(nodes[0]?.position).toEqual({ x: 11, y: 22 });
    expect(nodes[1]?.position).toEqual({ x: 33, y: 44 });
  });

  it("origin 平移整张图", () => {
    const r = computeAutoLayout([n("a"), n("b")], [e("a", "b")], { x: 100, y: 50 });
    expect(at(r, "a")).toEqual({ x: 100, y: 50 });
    expect(at(r, "b")).toEqual({ x: 100 + PITCH, y: 50 });
  });

  it("measured 缺失或非法(0/NaN)时退回默认尺寸", () => {
    const r = computeAutoLayout(
      [n("a", 0, 0, { width: 0, height: Number.NaN }), n("b")],
      [e("a", "b")],
    );
    expect(at(r, "b").x).toBe(PITCH);
  });
});
