import { jsx as _jsx } from "react/jsx-runtime";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";
export function SplitNode(props) {
    const { t } = useTranslation();
    const [params, updateParams] = useNodeParams(props);
    const outputs = Math.max(1, Math.min(8, Math.round(params.outputs ?? 2)));
    return (_jsx(NodeShell, { nodeId: props.id, label: t("workflow.nodeSplit"), icon: "\u2702", color: "#94a3b8", inputs: 1, outputs: outputs, outputLabels: Array.from({ length: outputs }, (_, i) => `${i + 1}`), children: _jsx("div", { className: "sep-node-body", children: _jsx("div", { className: "sep-params", children: _jsx(ParamSlider, { label: t("workflow.splitOutputs"), title: t("workflow.splitOutputsTitle"), min: 1, max: 8, step: 1, value: outputs, format: (v) => `${v}`, onChange: (v) => updateParams({ outputs: Math.round(v) }) }) }) }) }));
}
