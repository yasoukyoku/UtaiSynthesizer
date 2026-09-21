// 节点对齐 / 均匀分布:多选节点后把它们按某条基准线对齐,或在首尾之间等间距铺开。
//
// 和 autoLayout.ts 的分工:autoLayout 重排**整张图**(按数据流分层),这里只动**用户选中的
// 那几个**节点,且完全不看连线 —— 对齐是纯几何操作,用户既然手动选了这几个,就按选中集合办事。
//
// 同样是纯函数 + 只返回「真的变了」的坐标,理由和 autoLayout 一致:位置不进 sigOfGraph,写回
// 越少越好,moved === 0 时调用方可以直接跳过 setNodes。
//
// 尺寸同样优先用 ReactFlow v12 回填的 node.measured —— 右对齐/居中/水平分布都依赖真实宽高,
// 拿固定值会让宽窄不一的卡片(Opt 1 之后有 280px 宽卡)对不齐。
const DEFAULT_NODE_W = 240;
const DEFAULT_NODE_H = 120;
const MOVE_EPSILON = 0.5;
/** 对齐/分布至少要 2 个节点才有意义;分布实际需要 3 个(首尾固定,只有中间的会动)。 */
export const MIN_ALIGN_NODES = 2;
export const MIN_DISTRIBUTE_NODES = 3;
function sizeOf(n) {
    const w = n.measured?.width;
    const h = n.measured?.height;
    return {
        w: typeof w === "number" && Number.isFinite(w) && w > 0 ? w : DEFAULT_NODE_W,
        h: typeof h === "number" && Number.isFinite(h) && h > 0 ? h : DEFAULT_NODE_H,
    };
}
/** 把结果收敛成「只含真的移动了的节点」,并给出计数。 */
function finalize(nodes, next) {
    const positions = new Map();
    let moved = 0;
    for (const n of nodes) {
        const p = next.get(n.id);
        if (!p)
            continue;
        if (Math.abs(p.x - n.position.x) > MOVE_EPSILON || Math.abs(p.y - n.position.y) > MOVE_EPSILON) {
            positions.set(n.id, p);
            moved += 1;
        }
    }
    return { positions, moved };
}
/**
 * 按基准线对齐。基准线取自**选中集合本身**的包围盒(而不是画布原点):
 *  - left / right / hcenter:分别对齐到集合最左边缘、最右边缘、水平中心
 *  - top / bottom / vcenter:同理,纵向
 *
 * 右对齐和居中都按各自的实测宽高换算,所以宽度不同的卡片是**边缘**对齐,不是坐标对齐。
 */
export function alignNodes(nodes, mode) {
    if (nodes.length < MIN_ALIGN_NODES)
        return { positions: new Map(), moved: 0 };
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const n of nodes) {
        const { w, h } = sizeOf(n);
        if (n.position.x < minX)
            minX = n.position.x;
        if (n.position.x + w > maxX)
            maxX = n.position.x + w;
        if (n.position.y < minY)
            minY = n.position.y;
        if (n.position.y + h > maxY)
            maxY = n.position.y + h;
    }
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const next = new Map();
    for (const n of nodes) {
        const { w, h } = sizeOf(n);
        let { x, y } = n.position;
        switch (mode) {
            case "left":
                x = minX;
                break;
            case "right":
                x = maxX - w;
                break;
            case "hcenter":
                x = cx - w / 2;
                break;
            case "top":
                y = minY;
                break;
            case "bottom":
                y = maxY - h;
                break;
            case "vcenter":
                y = cy - h / 2;
                break;
        }
        next.set(n.id, { x, y });
    }
    return finalize(nodes, next);
}
/**
 * 均匀分布:保持**首尾两个节点不动**,让中间节点的间隙(gap,即边缘到边缘的空隙)相等。
 *
 * 刻意按 gap 均分而不是按坐标均分 —— 后者在卡片宽高不一致时看着是歪的。所以先算出「首尾之间
 * 的可用空间减去所有节点自身尺寸」,再把剩余空间平均分给 n-1 个缝隙。总尺寸超过可用空间时
 * (节点太挤)gap 会是负数,这里夹到 0,退化成首尾之间依次紧贴排列 —— 不报错。
 */
export function distributeNodes(nodes, mode) {
    if (nodes.length < MIN_DISTRIBUTE_NODES)
        return { positions: new Map(), moved: 0 };
    const horizontal = mode === "horizontal";
    // 按主轴起点排序;起点相同时按另一轴兜底,保证结果稳定可复现。
    const sorted = [...nodes].sort((a, b) => {
        const d = horizontal ? a.position.x - b.position.x : a.position.y - b.position.y;
        if (Math.abs(d) > 0.001)
            return d;
        return horizontal ? a.position.y - b.position.y : a.position.x - b.position.x;
    });
    const first = sorted[0];
    const last = sorted[sorted.length - 1];
    const sizeOn = (n) => (horizontal ? sizeOf(n).w : sizeOf(n).h);
    const startOf = (n) => (horizontal ? n.position.x : n.position.y);
    const spanStart = startOf(first);
    const spanEnd = startOf(last) + sizeOn(last);
    const totalSize = sorted.reduce((sum, n) => sum + sizeOn(n), 0);
    const gap = Math.max(0, (spanEnd - spanStart - totalSize) / (sorted.length - 1));
    const next = new Map();
    let cursor = spanStart;
    for (const n of sorted) {
        next.set(n.id, horizontal ? { x: cursor, y: n.position.y } : { x: n.position.x, y: cursor });
        cursor += sizeOn(n) + gap;
    }
    return finalize(nodes, next);
}
