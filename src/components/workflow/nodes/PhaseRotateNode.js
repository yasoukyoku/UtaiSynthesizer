import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";
export function PhaseRotateNode(props) {
    const { t } = useTranslation();
    const [params, updateParams] = useNodeParams(props);
    const strength = Math.max(0, Math.min(1, params.strength ?? 0));
    return (_jsx(NodeShell, { nodeId: props.id, label: t("workflow.nodePhaseRotate"), icon: "\uD83C\uDF00", color: "#c084fc", inputs: 1, outputs: 1, outputLabels: ["audio"], children: _jsx("div", { className: "sep-node-body", children: _jsxs("div", { className: "sep-params", children: [_jsx(ParamSlider, { label: t("workflow.phaseRotateStrength"), title: t("workflow.phaseRotateHint"), min: 0, max: 1, step: 0.125, value: strength, format: (v) => `${Math.round(v * 8)} / 8`, onChange: (v) => updateParams({ strength: v }) }), _jsx("div", { style: { fontSize: 10, color: "var(--text-muted)", padding: "4px 0 0 4px" }, children: t("workflow.phaseRotateHint") })] }) }) }));
}
