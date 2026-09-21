import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSliderWithUnit } from "./ParamSliderWithUnit";
export function LufsNormalizeNode(props) {
    const { t } = useTranslation();
    const [params, updateParams] = useNodeParams(props);
    const targetLufs = params.targetLufs ?? -16;
    return (_jsx(NodeShell, { nodeId: props.id, label: t("workflow.nodeLufsNormalize"), icon: "\uD83D\uDCE2", color: "#14b8a6", inputs: 1, outputs: 1, outputLabels: ["audio"], children: _jsx("div", { className: "sep-node-body", children: _jsxs("div", { className: "sep-params", children: [_jsx(ParamSliderWithUnit, { label: t("workflow.lufsTarget"), title: t("workflow.lufsTargetTitle"), unitType: "lufs", min: -24, max: -9, step: 0.5, value: targetLufs, onChange: (v) => updateParams({ targetLufs: v }) }), _jsx("div", { style: { fontSize: 10, color: "var(--text-muted)", padding: "4px 0 0 4px" }, children: t("workflow.lufsHint") })] }) }) }));
}
