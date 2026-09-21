import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Handle, Position, useInternalNode } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { useWorkflowStore } from "../../../store/workflow";
import { useAppStore } from "../../../store/app";
import { t18 } from "../../../lib/models/msst-catalog";
import { NODE_DAMAGE, DAMAGE_ICONS } from "../../../lib/workflow/damage";
import { rfTypeToWfType } from "../../../lib/workflow/rfTypes";
import { inputPortKind, outputPortKind } from "../../../lib/workflow/ports";
import { NodeOutputPreview } from "./NodeOutputPreview";
import { NodeAnnotationBadge } from "../NodeAnnotation";
import "./NodeShell.css";
export function NodeShell({ nodeId, label, icon, color, inputs = 1, outputs = 1, outputLabels, children, onRunNode, runTitle, noRunButton, width }) {
    const { i18n } = useTranslation();
    const outCount = outputLabels ? outputLabels.length : outputs;
    const segmentId = useAppStore((s) => s.workflowSegmentId);
    // A pending split-mid-render LINK target mirrors its render SOURCE's node badges (same node ids — the
    // graph was copied on split) and LOCKS its run buttons: it doesn't run its own copy of the nodes (the
    // source's single render feeds both halves via RenderLinkWatcher). renderLinks is empty in the common
    // case → statusSeg === segmentId, locked === false (the extra subscription only re-renders on a link
    // appearing/clearing, which is rare).
    const linkedSource = useWorkflowStore((s) => (segmentId ? s.renderLinks[segmentId] : undefined));
    const statusSeg = linkedSource ?? segmentId;
    const locked = linkedSource !== undefined;
    // Subscribe to THIS node's status/progress/error specifically (primitive selectors) so a progress
    // tick on one node doesn't re-render every other node card on the canvas.
    const status = useWorkflowStore((s) => (statusSeg && nodeId ? s.nodeStatuses[statusSeg]?.[nodeId] : undefined)) ?? "idle";
    const progress = useWorkflowStore((s) => (statusSeg && nodeId ? s.nodeProgress[statusSeg]?.[nodeId] : undefined)) ?? 0;
    const error = useWorkflowStore((s) => (statusSeg && nodeId ? s.nodeErrors[statusSeg]?.[nodeId] : undefined));
    const singleNodeRunner = useWorkflowStore((s) => s.singleNodeRunner);
    const resolvedRunNode = onRunNode ?? (nodeId && singleNodeRunner ? () => singleNodeRunner(nodeId) : undefined);
    // 规划 6-4/6-5: 借内部节点反查域类型 → 损伤等级（徽标）；data.bypass 是持久图属性
    // （旁通态与是否在跑无关），status === "bypassed" 则是本轮运行的预标/终标 —— 两者都算旁通。
    const internal = useInternalNode(nodeId ?? "");
    const wfType = internal?.type ? rfTypeToWfType[internal.type] : undefined;
    const damage = wfType ? NODE_DAMAGE[wfType] : undefined;
    const bypassed = internal?.data?.bypass === true || status === "bypassed";
    const annotation = internal?.data?.annotation;
    const annotationEditor = useWorkflowStore((s) => s.annotationEditor);
    const statusClass = [status !== "idle" ? `wf-node-${status}` : "", bypassed ? "wf-node-bypassed" : ""]
        .filter(Boolean)
        .join(" ");
    return (_jsxs("div", { className: `wf-node ${statusClass}`, style: { "--node-color": color, ...(width ? { "--node-width": `${width}px` } : {}) }, children: [_jsxs("div", { className: "wf-node-header", children: [_jsx("span", { className: "wf-node-icon", children: icon }), _jsx("span", { className: "wf-node-label", children: label }), _jsx("div", { className: "wf-node-header-spacer" }), damage && damage.level > 0 && (_jsx("span", { className: "wf-node-damage", title: t18({ zh: `保真度等级 ${damage.level} · 单级 SNR ≈ ${damage.snrDb}dB`, en: `Fidelity tier ${damage.level} · per-node SNR ≈ ${damage.snrDb}dB`, ja: `忠実度レベル ${damage.level} · 単体 SNR ≈ ${damage.snrDb}dB` }, i18n.language), children: DAMAGE_ICONS[damage.level] })), bypassed && (_jsx("span", { className: "wf-node-bypass-badge", title: t18({ zh: "旁通中 —— 引擎跳过此节点，输入原样透传", en: "Bypassed — engine skips this node, input passes through", ja: "バイパス中 —— エンジンはこのノードをスキップし入力をそのまま通す" }, i18n.language), children: "\u23F8" })), annotation && annotationEditor && nodeId && (_jsx(NodeAnnotationBadge, { annotation: annotation, onClick: () => annotationEditor(nodeId) })), status === "running" && progress > 0 && (_jsxs("span", { className: "wf-node-pct", children: [Math.round(progress * 100), "%"] })), status === "running" && _jsx("span", { className: "wf-node-pulse" }), status === "completed" && _jsx("span", { className: "wf-node-done", children: "OK" }), status === "degraded" && _jsx("span", { className: "wf-node-warn", title: error ?? undefined, children: "!" }), status === "error" && _jsx("span", { className: "wf-node-err", title: error, children: "!!" }), resolvedRunNode && !noRunButton && !locked && !bypassed && status !== "running" && (_jsx("button", { className: "wf-node-run-btn", onClick: (e) => { e.stopPropagation(); resolvedRunNode(); }, title: runTitle ?? t18({ zh: "运行此节点", en: "Run this node", ja: "このノードを実行" }, i18n.language), children: ">" }))] }), status === "running" && progress > 0 && (_jsx("div", { className: "wf-node-progress", children: _jsx("div", { className: "wf-node-progress-fill", style: { width: `${Math.round(progress * 100)}%` } }) })), children && _jsx("div", { className: "wf-node-body", children: children }), nodeId && status !== "running" && (_jsx(NodeOutputPreview, { statusSeg: statusSeg, nodeId: nodeId, outputLabels: outputLabels })), Array.from({ length: inputs }).map((_, i) => {
                const portType = wfType ? inputPortKind(wfType, i) : "any";
                return (_jsx(Handle, { type: "target", position: Position.Left, id: `in-${i}`, style: { top: `${((i + 1) / (inputs + 1)) * 100}%` }, "data-port-type": portType }, `in-${i}`));
            }), Array.from({ length: outCount }).map((_, i) => {
                const portType = wfType ? outputPortKind(wfType, i) : "any";
                return (_jsx(Handle, { type: "source", position: Position.Right, id: `out-${i}`, style: { top: `${((i + 1) / (outCount + 1)) * 100}%` }, "data-port-type": portType }, `out-${i}`));
            }), outputLabels && (_jsx("div", { className: "wf-node-out-labels", children: outputLabels.map((lbl, i) => (_jsx("span", { className: "wf-out-label", style: { top: `${((i + 1) / (outCount + 1)) * 100}%` }, children: lbl }, i))) }))] }));
}
