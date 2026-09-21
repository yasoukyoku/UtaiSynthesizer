import { useAppStore } from "../../store/app";
import { useProjectStore } from "../../store/project";
import { TimeAxis } from "../timeAxis";
import { PIXELS_PER_TICK } from "../../lib/constants";
import { computeTotalTicks } from "../../lib/trackLayout";
/**
 * 横向缩放 fit-to-content：把 zoom 调成"能一眼看到整首时间轴"的倍率。
 * 在工程 打开/恢复/新建 完成后调用一次(不在渲染期订阅，避免和用户缩放打架)。
 * 空工程 → 复位为 1×；宽度未知或内容为 0 → 不动。clamp 到 app 允许的 0.1–10。
 */
export function fitTimelineToContent() {
    const s = useAppStore.getState();
    const tracks = useProjectStore.getState().tracks;
    if (tracks.length === 0) {
        s.setZoom(1);
        s.setScroll(0, s.scrollY);
        return;
    }
    const ts = useProjectStore.getState().timeSignature;
    const axis = TimeAxis.global(ts[0], ts[1]);
    const contentPx = computeTotalTicks(tracks, axis) * PIXELS_PER_TICK;
    const cw = s.canvasWidth;
    if (!cw || !contentPx)
        return;
    const z = Math.max(0.1, Math.min(10, cw / contentPx));
    s.setZoom(z);
    s.setScroll(0, s.scrollY);
}
