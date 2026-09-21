import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";
const GRID_KEYS = ["symGrid8", "symGrid16"];
export function SwingQuantizeNode(props) {
    const { t } = useTranslation();
    const [params, updateParams] = useNodeParams(props);
    const gridIdx = params.grid ?? 0;
    const swing = params.swing ?? 55;
    const quantize = params.quantize ?? 0;
    return (_jsx(NodeShell, { nodeId: props.id, label: t("workflow.nodeSwingQuantize"), icon: "\uD83E\uDD41", color: "#a3e635", inputs: 1, outputs: 1, outputLabels: ["notes"], children: _jsx("div", { className: "sep-node-body", children: _jsxs("div", { className: "sep-params", children: [_jsxs("div", { className: "sep-label-row", children: [_jsx("span", { className: "sep-label", children: t("workflow.symGrid") }), _jsx("select", { className: "sep-select", value: String(gridIdx), onChange: (e) => updateParams({ grid: Number(e.target.value) }), children: GRID_KEYS.map((k, i) => (_jsx("option", { value: i, children: t(`workflow.${k}`) }, k))) })] }), _jsx(ParamSlider, { label: t("workflow.symSwing"), min: 0, max: 100, step: 1, value: swing, onChange: (v) => updateParams({ swing: v }) }), _jsx(ParamSlider, { label: t("workflow.symQuantize"), min: 0, max: 100, step: 1, value: quantize, onChange: (v) => updateParams({ quantize: v }) }), _jsx("div", { style: { fontSize: 10, color: "#888", padding: "4px 0 0 4px" }, children: t("workflow.symSwingHint") })] }) }) }));
}
