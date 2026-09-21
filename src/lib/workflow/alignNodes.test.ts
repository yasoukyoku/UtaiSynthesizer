import { describe, it, expect } from "vitest";
import {
  alignNodes,
  distributeNodes,
  MIN_ALIGN_NODES,
  MIN_DISTRIBUTE_NODES,
  type AlignMode,
} from "./alignNodes";
import type { LayoutNodeInput } from "./autoLayout";

function n(
  id: string,
  x: number,
  y: number,
  size?: { width: number; height: number },
): LayoutNodeInput {
  return { id, position: { x, y }, ...(size ? { measured: size } : {}) };
}

describe("alignNodes", () => {
  it("returns nothing for fewer than MIN_ALIGN_NODES", () => {
    expect(MIN_ALIGN_NODES).toBe(2);
    expect(alignNodes([], "left").moved).toBe(0);
    expect(alignNodes([n("a", 10, 20)], "left").moved).toBe(0);
  });

  it("aligns left to the selection's leftmost edge", () => {
    const r = alignNodes([n("a", 100, 0), n("b", 40, 50), n("c", 70, 90)], "left");
    expect(r.moved).toBe(2);
    expect(r.positions.get("a")?.x).toBe(40);
    expect(r.positions.get("c")?.x).toBe(40);
    // b was already at the baseline, so it must not be reported as moved.
    expect(r.positions.has("b")).toBe(false);
  });

  it("preserves the off-axis coordinate when aligning horizontally", () => {
    const r = alignNodes([n("a", 100, 33), n("b", 40, 50)], "left");
    expect(r.positions.get("a")).toEqual({ x: 40, y: 33 });
  });

  it("aligns right by trailing edge, honouring per-node widths", () => {
    // a spans 0..100, b spans 40..340 → rightmost edge is 340.
    const r = alignNodes(
      [n("a", 0, 0, { width: 100, height: 50 }), n("b", 40, 60, { width: 300, height: 50 })],
      "right",
    );
    expect(r.positions.get("a")?.x).toBe(240);
    expect(r.positions.has("b")).toBe(false);
  });

  it("centers horizontally on the selection bounding box", () => {
    // Bounding box 0..300 → center 150. Width-100 node lands at 100.
    const r = alignNodes(
      [n("a", 0, 0, { width: 300, height: 50 }), n("b", 0, 60, { width: 100, height: 50 })],
      "hcenter",
    );
    expect(r.positions.get("b")?.x).toBe(100);
    expect(r.positions.has("a")).toBe(false);
  });

  it("aligns top / bottom / vcenter on the vertical axis", () => {
    const nodes = [
      n("a", 0, 0, { width: 100, height: 100 }),
      n("b", 0, 200, { width: 100, height: 40 }),
    ];
    expect(alignNodes(nodes, "top").positions.get("b")?.y).toBe(0);
    // Bounding box 0..240 → bottom edge 240, so b's top is 200 (already there).
    const bottom = alignNodes(nodes, "bottom");
    expect(bottom.positions.get("a")?.y).toBe(140);
    // center 120 → a (h=100) at 70, b (h=40) at 100.
    const vcenter = alignNodes(nodes, "vcenter");
    expect(vcenter.positions.get("a")?.y).toBe(70);
    expect(vcenter.positions.get("b")?.y).toBe(100);
  });

  it("falls back to default size when measured is absent", () => {
    // Default width 240: a spans 0..240, b spans 100..340 → right edge 340.
    const r = alignNodes([n("a", 0, 0), n("b", 100, 0)], "right");
    expect(r.positions.get("a")?.x).toBe(100);
  });

  it("ignores a zero or negative measured size", () => {
    const r = alignNodes(
      [n("a", 0, 0, { width: 0, height: 0 }), n("b", 100, 0, { width: 240, height: 120 })],
      "right",
    );
    expect(r.positions.get("a")?.x).toBe(100);
  });

  it("is idempotent", () => {
    const nodes = [n("a", 100, 0), n("b", 40, 50), n("c", 70, 90)];
    const once = alignNodes(nodes, "left");
    const applied = nodes.map((node) => {
      const p = once.positions.get(node.id);
      return p ? { ...node, position: p } : node;
    });
    expect(alignNodes(applied, "left").moved).toBe(0);
  });

  it("does not mutate the input array or its nodes", () => {
    const nodes = [n("a", 100, 0), n("b", 40, 50)];
    const snapshot = JSON.stringify(nodes);
    alignNodes(nodes, "left");
    expect(JSON.stringify(nodes)).toBe(snapshot);
  });

  it("reports moved === 0 when everything already sits on the baseline", () => {
    const r = alignNodes([n("a", 50, 0), n("b", 50, 80)], "left");
    expect(r.moved).toBe(0);
    expect(r.positions.size).toBe(0);
  });

  it("handles every align mode without throwing", () => {
    const modes: AlignMode[] = ["left", "right", "hcenter", "top", "bottom", "vcenter"];
    for (const m of modes) {
      expect(() => alignNodes([n("a", 0, 0), n("b", 30, 40)], m)).not.toThrow();
    }
  });
});

describe("distributeNodes", () => {
  it("requires at least MIN_DISTRIBUTE_NODES", () => {
    expect(MIN_DISTRIBUTE_NODES).toBe(3);
    expect(distributeNodes([n("a", 0, 0), n("b", 100, 0)], "horizontal").moved).toBe(0);
  });

  it("equalises horizontal gaps and pins the outer nodes", () => {
    // Span 0..700, three 100-wide nodes → 400 free space over 2 gaps = 200 each.
    const r = distributeNodes(
      [
        n("a", 0, 0, { width: 100, height: 50 }),
        n("b", 150, 0, { width: 100, height: 50 }),
        n("c", 600, 0, { width: 100, height: 50 }),
      ],
      "horizontal",
    );
    expect(r.positions.has("a")).toBe(false);
    expect(r.positions.has("c")).toBe(false);
    expect(r.positions.get("b")?.x).toBe(300);
  });

  it("equalises gaps between differently sized nodes", () => {
    // Span 0..600. Sizes 100+200+100=400 → 200 free over 2 gaps = 100 each.
    const r = distributeNodes(
      [
        n("a", 0, 0, { width: 100, height: 50 }),
        n("b", 120, 0, { width: 200, height: 50 }),
        n("c", 500, 0, { width: 100, height: 50 }),
      ],
      "horizontal",
    );
    expect(r.positions.get("b")?.x).toBe(200);
  });

  it("distributes vertically and leaves x untouched", () => {
    const r = distributeNodes(
      [
        n("a", 7, 0, { width: 100, height: 100 }),
        n("b", 9, 120, { width: 100, height: 100 }),
        n("c", 11, 700, { width: 100, height: 100 }),
      ],
      "vertical",
    );
    // Span 0..800, sizes 300 → 500 over 2 gaps = 250. b lands at 100+250 = 350.
    expect(r.positions.get("b")).toEqual({ x: 9, y: 350 });
  });

  it("sorts by axis position rather than array order", () => {
    const r = distributeNodes(
      [
        n("c", 600, 0, { width: 100, height: 50 }),
        n("a", 0, 0, { width: 100, height: 50 }),
        n("b", 150, 0, { width: 100, height: 50 }),
      ],
      "horizontal",
    );
    expect(r.positions.get("b")?.x).toBe(300);
    expect(r.positions.has("a")).toBe(false);
    expect(r.positions.has("c")).toBe(false);
  });

  it("clamps the gap to zero when nodes overflow the span", () => {
    // Span 0..150 but sizes total 300 → negative gap, clamped to 0 (nodes butt together).
    const r = distributeNodes(
      [
        n("a", 0, 0, { width: 100, height: 50 }),
        n("b", 10, 0, { width: 100, height: 50 }),
        n("c", 50, 0, { width: 100, height: 50 }),
      ],
      "horizontal",
    );
    expect(r.positions.get("b")?.x).toBe(100);
    expect(r.positions.get("c")?.x).toBe(200);
  });

  it("is idempotent", () => {
    const nodes = [
      n("a", 0, 0, { width: 100, height: 50 }),
      n("b", 150, 0, { width: 100, height: 50 }),
      n("c", 600, 0, { width: 100, height: 50 }),
    ];
    const once = distributeNodes(nodes, "horizontal");
    const applied = nodes.map((node) => {
      const p = once.positions.get(node.id);
      return p ? { ...node, position: p } : node;
    });
    expect(distributeNodes(applied, "horizontal").moved).toBe(0);
  });

  it("does not mutate the input array order", () => {
    const nodes = [n("c", 600, 0), n("a", 0, 0), n("b", 150, 0)];
    distributeNodes(nodes, "horizontal");
    expect(nodes.map((x) => x.id)).toEqual(["c", "a", "b"]);
  });
});
