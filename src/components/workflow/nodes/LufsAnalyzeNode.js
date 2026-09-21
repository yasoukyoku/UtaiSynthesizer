import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";
export function LufsAnalyzeNode(props) {
    const { t } = useTranslation();
    const [params, updateParams] = useNodeParams(props);
    const targetLufs = params.targetLufs ?? -16;
    return (_jsx(NodeShell, { nodeId: props.id, label: t("workflow.nodeLufsAnalyze"), icon: "\uD83D\uDCCA", color: "#8b5cf6", inputs: 1, outputs: 2, outputLabels: [t("workflow.lufsAnalyzeOutAudio"), t("workflow.lufsAnalyzeOutReport")], children: _jsx("div", { className: "sep-node-body", children: _jsxs("div", { className: "sep-params", children: [_jsx(ParamSlider, { label: t("workflow.lufsTargetRef"), title: t("workflow.lufsTargetRefTitle"), min: -24, max: -9, step: 0.5, value: targetLufs, format: (v) => `${v.toFixed(1)} LUFS`, onChange: (v) => updateParams({ targetLufs: v }) }), _jsx("div", { style: { fontSize: 10, color: "var(--text-muted)", padding: "4px 0 0 4px" }, children: t("workflow.lufsAnalyzeHint") })] }) }) }));
}
