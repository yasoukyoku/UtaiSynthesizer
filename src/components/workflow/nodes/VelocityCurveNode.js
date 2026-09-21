import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";
const CURVE_KEYS = ["symCurveCresc", "symCurveDecresc", "symCurveArch", "symCurveCustom"];
export function VelocityCurveNode(props) {
    const { t } = useTranslation();
    const [params, updateParams] = useNodeParams(props);
    const curveIdx = params.curve ?? 2;
    const intensity = params.intensity ?? 60;
    const customShape = params.customShape ?? "";
    return (_jsx(NodeShell, { nodeId: props.id, label: t("workflow.nodeVelocityCurve"), icon: "\uD83D\uDCC8", color: "#f472b6", inputs: 1, outputs: 1, outputLabels: ["notes"], children: _jsx("div", { className: "sep-node-body", children: _jsxs("div", { className: "sep-params", children: [_jsxs("div", { className: "sep-label-row", children: [_jsx("span", { className: "sep-label", children: t("workflow.symCurve") }), _jsx("select", { className: "sep-select", value: String(curveIdx), onChange: (e) => updateParams({ curve: Number(e.target.value) }), children: CURVE_KEYS.map((k, i) => (_jsx("option", { value: i, children: t(`workflow.${k}`) }, k))) })] }), _jsx(ParamSlider, { label: t("workflow.symIntensity"), min: 0, max: 100, step: 1, value: intensity, onChange: (v) => updateParams({ intensity: v }) }), curveIdx === 3 && (_jsxs("div", { className: "sep-label-row", children: [_jsx("span", { className: "sep-label", children: t("workflow.symCustomShape") }), _jsx("textarea", { className: "sep-select", rows: 2, style: { resize: "none", fontSize: 10 }, value: customShape, placeholder: "20, 60, 100, 40", onChange: (e) => updateParams({ customShape: e.target.value }) })] })), _jsx("div", { style: { fontSize: 10, color: "#888", padding: "4px 0 0 4px" }, children: t("workflow.symVelCurveHint") })] }) }) }));
}
