import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";
export function MelodyReharmNode(props) {
    const { t } = useTranslation();
    const [params, updateParams] = useNodeParams(props);
    const density = params.density ?? 40;
    const seed = params.seed ?? 0;
    return (_jsx(NodeShell, { nodeId: props.id, label: t("workflow.nodeMelodyReharm"), icon: "\uD83C\uDFBC", color: "#34d399", inputs: 1, outputs: 1, outputLabels: ["notes"], children: _jsx("div", { className: "sep-node-body", children: _jsxs("div", { className: "sep-params", children: [_jsx(ParamSlider, { label: t("workflow.symDensity"), min: 0, max: 100, step: 1, value: density, onChange: (v) => updateParams({ density: v }) }), _jsx(ParamSlider, { label: t("workflow.symSeed"), min: 0, max: 4095, step: 1, value: seed, onChange: (v) => updateParams({ seed: v }) }), _jsx("div", { style: { fontSize: 10, color: "#888", padding: "4px 0 0 4px" }, children: t("workflow.symReharmHint") })] }) }) }));
}
