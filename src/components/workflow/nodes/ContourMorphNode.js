import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";
const CLIMAX_KEYS = ["symOff", "symOn"];
export function ContourMorphNode(props) {
    const { t } = useTranslation();
    const [params, updateParams] = useNodeParams(props);
    const strength = params.strength ?? 60;
    const keepClimax = params.keepClimax ?? 1;
    const seed = params.seed ?? 0;
    return (_jsx(NodeShell, { nodeId: props.id, label: t("workflow.nodeContourMorph"), icon: "\uD83C\uDFA2", color: "#f97316", inputs: 1, outputs: 1, outputLabels: ["notes"], children: _jsx("div", { className: "sep-node-body", children: _jsxs("div", { className: "sep-params", children: [_jsx(ParamSlider, { label: t("workflow.symStrength"), min: 0, max: 100, step: 1, value: strength, onChange: (v) => updateParams({ strength: v }) }), _jsxs("div", { className: "sep-label-row", children: [_jsx("span", { className: "sep-label", children: t("workflow.symKeepClimax") }), _jsx("select", { className: "sep-select", value: String(keepClimax), onChange: (e) => updateParams({ keepClimax: Number(e.target.value) }), children: CLIMAX_KEYS.map((k, i) => (_jsx("option", { value: i, children: t(`workflow.${k}`) }, k))) })] }), _jsx(ParamSlider, { label: t("workflow.symSeed"), min: 0, max: 4095, step: 1, value: seed, onChange: (v) => updateParams({ seed: v }) }), _jsx("div", { style: { fontSize: 10, color: "#888", padding: "4px 0 0 4px" }, children: t("workflow.symContourHint") })] }) }) }));
}
