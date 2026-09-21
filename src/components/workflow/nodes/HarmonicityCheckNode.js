import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";
export function HarmonicityCheckNode(props) {
    const { t } = useTranslation();
    const [params, updateParams] = useNodeParams(props);
    const f0Hz = Math.max(0, Math.min(1000, params.f0Hz ?? 0));
    return (_jsx(NodeShell, { nodeId: props.id, label: t("workflow.nodeHarmonicityCheck"), icon: "\uD83D\uDD2C", color: "#4ade80", inputs: 1, outputs: 2, outputLabels: [t("workflow.analysisOutAudio"), t("workflow.analysisOutReport")], children: _jsx("div", { className: "sep-node-body", children: _jsxs("div", { className: "sep-params", children: [_jsx(ParamSlider, { label: t("workflow.harmonicityTargetF0"), title: t("workflow.harmonicityHint"), min: 0, max: 1000, step: 5, value: f0Hz, format: (v) => `${Math.round(v)} Hz`, onChange: (v) => updateParams({ f0Hz: v }) }), _jsx("div", { style: { fontSize: 10, color: "var(--text-muted)", padding: "4px 0 0 4px" }, children: t("workflow.harmonicityHint") })] }) }) }));
}
