import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";
export function MergeNode(props) {
    const { t } = useTranslation();
    const [params, updateParams] = useNodeParams(props);
    const inputs = Math.max(2, Math.min(8, Math.round(params.inputs ?? 2)));
    return (_jsx(NodeShell, { nodeId: props.id, label: t("workflow.nodeMerge"), icon: "\uD83E\uDDF2", color: "#fb923c", inputs: inputs, outputs: 1, outputLabels: ["mix"], children: _jsx("div", { className: "sep-node-body", children: _jsxs("div", { className: "sep-params", children: [_jsx(ParamSlider, { label: t("workflow.mergeInputs"), title: t("workflow.mergeInputsTitle"), min: 2, max: 8, step: 1, value: inputs, format: (v) => `${v}`, onChange: (v) => updateParams({ inputs: Math.round(v) }) }), _jsx("div", { style: { fontSize: 10, color: "var(--text-muted)", padding: "4px 0 0 4px" }, children: t("workflow.mergeHint") })] }) }) }));
}
