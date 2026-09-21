import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";
const MODE_KEYS = ["symModePush", "symModeLayBack", "symModeSyncopate", "symModeSparse"];
export function RhythmVariationNode(props) {
    const { t } = useTranslation();
    const [params, updateParams] = useNodeParams(props);
    const mode = params.mode ?? 0;
    const amount = params.amount ?? 50;
    return (_jsx(NodeShell, { nodeId: props.id, label: t("workflow.nodeRhythmVariation"), icon: "\uD83D\uDD79", color: "#e879f9", inputs: 1, outputs: 1, outputLabels: ["notes"], children: _jsx("div", { className: "sep-node-body", children: _jsxs("div", { className: "sep-params", children: [_jsxs("div", { className: "sep-label-row", children: [_jsx("span", { className: "sep-label", children: t("workflow.symMode") }), _jsx("select", { className: "sep-select", value: String(mode), onChange: (e) => updateParams({ mode: Number(e.target.value) }), children: MODE_KEYS.map((k, i) => (_jsx("option", { value: i, children: t(`workflow.${k}`) }, k))) })] }), _jsx(ParamSlider, { label: t("workflow.symAmount"), min: 0, max: 100, step: 1, value: amount, onChange: (v) => updateParams({ amount: v }) }), _jsx("div", { style: { fontSize: 10, color: "#888", padding: "4px 0 0 4px" }, children: t("workflow.symRhythmVarHint") })] }) }) }));
}
