import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSliderWithUnit } from "./ParamSliderWithUnit";
export function StereoWidthNode(props) {
    const { t } = useTranslation();
    const [params, updateParams] = useNodeParams(props);
    const width = Math.max(0, Math.min(1.5, params.width ?? 1));
    return (_jsx(NodeShell, { nodeId: props.id, label: t("workflow.nodeStereoWidth"), icon: "\uD83C\uDFA7", color: "#a3e635", inputs: 1, outputs: 1, outputLabels: ["audio"], children: _jsx("div", { className: "sep-node-body", children: _jsxs("div", { className: "sep-params", children: [_jsx(ParamSliderWithUnit, { label: t("workflow.stereoWidthAmount"), unitType: "ratio", title: t("workflow.stereoWidthHint"), min: 0, max: 1.5, step: 0.05, value: width, onChange: (v) => updateParams({ width: v }) }), _jsx("div", { style: { fontSize: 10, color: "var(--text-muted)", padding: "4px 0 0 4px" }, children: t("workflow.stereoWidthHint") })] }) }) }));
}
