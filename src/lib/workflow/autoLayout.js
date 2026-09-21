// 一键整理画布(Opt 9):按数据流把节点排成从左到右的分层图。
//
// 刻意**不复用** lib/workflow/graph.ts 的 parseWorkflowGraph —— 它在「没有输入节点 / 多个输入
// 节点 / 没有输出节点 / 成环」时直接抛错,而这四种情况全都是用户边搭边改时的正常中间态。整理
// 按钮必须在任何时刻都能按,所以这里自己做一遍容错遍历:半截连线、自环、重复连线、孤立节点、
// 成环都只是被「读作没有层级信息」,不会报错、不会拒绝服务。
//
// 尺寸优先用 ReactFlow v12 渲染后回填的 node.measured;拿不到就退回默认值。节点卡片高度差得很
// 多(Opt 1 之后还有 280px 宽的宽卡),固定行距会让高卡片压到下一个,所以按实测高度累加排布。
/** 列间距(在上一列最宽节点的右边缘之后再留这么多)。配合 240 默认宽度 = 320 列距,和 templates.ts 手写坐标同一口径。 */
export const COLUMN_GAP = 80;
/** 同列相邻节点的垂直间距。 */
export const ROW_GAP = 40;
/** 互不相连的子图之间的垂直间距。 */
export const COMPONENT_GAP = 120;
const DEFAULT_NODE_W = 240;
const DEFAULT_NODE_H = 120;
const MOVE_EPSILON = 0.5;
function sizeOf(n) {
    const w = n.measured?.width;
    const h = n.measured?.height;
    return {
        w: typeof w === "number" && Number.isFinite(w) && w > 0 ? w : DEFAULT_NODE_W,
        h: typeof h === "number" && Number.isFinite(h) && h > 0 ? h : DEFAULT_NODE_H,
    };
}
/** 丢掉给不出层级信息的边:半截边(一端已删)、自环、以及同一对节点之间的重复连线。 */
function normalizeEdges(ids, edges) {
    const seen = new Set();
    const out = [];
    for (const e of edges) {
        if (!ids.has(e.source) || !ids.has(e.target))
            continue;
        if (e.source === e.target)
            continue;
        const key = `${e.source}\u0000${e.target}`;
        if (seen.has(key))
            continue;
        seen.add(key);
        out.push({ source: e.source, target: e.target });
    }
    return out;
}
/**
 * 最长路径分层。先跑 Kahn 拿一个拓扑序,**环里的节点按原顺序补在序列末尾** —— 这样既保证一定
 * 终止,又不用像 parseWorkflowGraph 那样成环就抛错;代价只是环内层级退化成「按出现顺序」。
 */
function assignLayers(order, preds, succs, edges) {
    const indeg = new Map();
    for (const id of order)
        indeg.set(id, 0);
    for (const e of edges)
        indeg.set(e.target, (indeg.get(e.target) ?? 0) + 1);
    const queue = order.filter((id) => (indeg.get(id) ?? 0) === 0);
    const topo = [];
    const settled = new Set();
    while (queue.length > 0) {
        const id = queue.shift();
        if (settled.has(id))
            continue;
        settled.add(id);
        topo.push(id);
        for (const next of succs.get(id) ?? []) {
            const left = (indeg.get(next) ?? 1) - 1;
            indeg.set(next, left);
            if (left === 0)
                queue.push(next);
        }
    }
    // 成环时 Kahn 会剩一批节点出不来 —— 按原数组顺序补上,绝不抛错。
    for (const id of order)
        if (!settled.has(id))
            topo.push(id);
    const layer = new Map();
    for (const id of topo) {
        let lv = 0;
        for (const p of preds.get(id) ?? []) {
            const pl = layer.get(p);
            // 环内的前驱可能还没定层(undefined),跳过即可,不参与取最大。
            if (pl !== undefined && pl + 1 > lv)
                lv = pl + 1;
        }
        layer.set(id, lv);
    }
    return layer;
}
/** 弱连通分量(忽略方向)—— 让互不相连的几张子图上下分开摆,而不是在同一列里交错。 */
function findComponents(ids, preds, succs) {
    const seen = new Set();
    const comps = [];
    for (const start of ids) {
        if (seen.has(start))
            continue;
        const comp = [];
        const stack = [start];
        seen.add(start);
        while (stack.length > 0) {
            const id = stack.pop();
            comp.push(id);
            for (const nb of [...(preds.get(id) ?? []), ...(succs.get(id) ?? [])]) {
                if (!seen.has(nb)) {
                    seen.add(nb);
                    stack.push(nb);
                }
            }
        }
        comps.push(comp);
    }
    return comps;
}
/**
 * 算出整理后的坐标。纯函数:只返回新坐标,不碰任何状态,也不改传进来的数组 —— 调用方自己决定
 * 是否写回(走 setNodes 就会被 sigOfGraph 捕获成一步撤销)。
 *
 * 排布规则:
 *  - x:按层级(最长路径)分列,列宽取该列最宽节点,列间留 COLUMN_GAP。
 *  - y:同列内按「上游节点的平均 y」排序(重心法,减少连线交叉),再按实测高度依次往下堆。
 *  - 每个弱连通分量单独摆一段,分量之间留 COMPONENT_GAP;分量内部仍然共享全局列坐标,所以
 *    所有子图的同一层级是对齐的。
 *  - 起点固定在 (origin.x, origin.y),默认 (0, 0),和模板里手写的坐标同一口径。
 */
export function computeAutoLayout(nodes, edges, origin = { x: 0, y: 0 }) {
    const positions = new Map();
    if (nodes.length === 0)
        return { positions, moved: 0 };
    const ids = nodes.map((n) => n.id);
    const idSet = new Set(ids);
    const byId = new Map(nodes.map((n) => [n.id, n]));
    const clean = normalizeEdges(idSet, edges);
    const preds = new Map();
    const succs = new Map();
    for (const id of ids) {
        preds.set(id, []);
        succs.set(id, []);
    }
    for (const e of clean) {
        succs.get(e.source).push(e.target);
        preds.get(e.target).push(e.source);
    }
    const layer = assignLayers(ids, preds, succs, clean);
    // 列宽/列 x:全局统一,这样不同分量的同层节点左对齐。
    const colWidth = new Map();
    for (const n of nodes) {
        const lv = layer.get(n.id) ?? 0;
        const { w } = sizeOf(n);
        if (w > (colWidth.get(lv) ?? 0))
            colWidth.set(lv, w);
    }
    const colX = new Map();
    {
        const levels = [...colWidth.keys()].sort((a, b) => a - b);
        let x = origin.x;
        for (const lv of levels) {
            colX.set(lv, x);
            x += (colWidth.get(lv) ?? DEFAULT_NODE_W) + COLUMN_GAP;
        }
    }
    // 重心排序用的参考 y:上游节点当前的 y 平均值(没有上游就用自己当前的 y)。用「整理前」的坐标
    // 做参考,是为了让整理结果尽量贴近用户已有的上下布局,不至于整完面目全非。
    const refY = (id) => {
        const ps = preds.get(id) ?? [];
        if (ps.length === 0)
            return byId.get(id)?.position.y ?? 0;
        let sum = 0;
        let cnt = 0;
        for (const p of ps) {
            const py = byId.get(p)?.position.y;
            if (typeof py === "number") {
                sum += py;
                cnt += 1;
            }
        }
        return cnt === 0 ? (byId.get(id)?.position.y ?? 0) : sum / cnt;
    };
    const orderIndex = new Map(ids.map((id, i) => [id, i]));
    let cursorY = origin.y;
    for (const comp of findComponents(ids, preds, succs)) {
        const rows = new Map();
        for (const id of comp) {
            const lv = layer.get(id) ?? 0;
            const bucket = rows.get(lv);
            if (bucket)
                bucket.push(id);
            else
                rows.set(lv, [id]);
        }
        let compBottom = cursorY;
        for (const [lv, bucket] of rows) {
            bucket.sort((a, b) => {
                const d = refY(a) - refY(b);
                // 参考 y 相同(常见:多个新加的节点都在同一位置)时退回原数组顺序,保证结果稳定可复现。
                if (Math.abs(d) > 0.001)
                    return d;
                return (orderIndex.get(a) ?? 0) - (orderIndex.get(b) ?? 0);
            });
            let y = cursorY;
            for (const id of bucket) {
                const n = byId.get(id);
                if (!n)
                    continue;
                positions.set(id, { x: colX.get(lv) ?? origin.x, y });
                y += sizeOf(n).h + ROW_GAP;
            }
            if (y - ROW_GAP > compBottom)
                compBottom = y - ROW_GAP;
        }
        cursorY = compBottom + COMPONENT_GAP;
    }
    let moved = 0;
    for (const n of nodes) {
        const p = positions.get(n.id);
        if (!p)
            continue;
        if (Math.abs(p.x - n.position.x) > MOVE_EPSILON || Math.abs(p.y - n.position.y) > MOVE_EPSILON)
            moved += 1;
    }
    return { positions, moved };
}
