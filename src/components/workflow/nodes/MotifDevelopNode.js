import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";
const TECH_KEYS = [
    "symTechSequence", "symTechInvert", "symTechRetrograde",
    "symTechAugment", "symTechDiminish", "symTechMixed",
];
export function MotifDevelopNode(props) {
    const { t } = useTranslation();
    const [params, updateParams] = useNodeParams(props);
    const technique = params.technique ?? 5;
    const motifBars = params.motifBars ?? 2;
    const seed = params.seed ?? 0;
    return (_jsx(NodeShell, { nodeId: props.id, label: t("workflow.nodeMotifDevelop"), icon: "\uD83E\uDDEC", color: "#c084fc", inputs: 1, outputs: 1, outputLabels: ["notes"], children: _jsx("div", { className: "sep-node-body", children: _jsxs("div", { className: "sep-params", children: [_jsxs("div", { className: "sep-label-row", children: [_jsx("span", { className: "sep-label", children: t("workflow.symTechnique") }), _jsx("select", { className: "sep-select", value: String(technique), onChange: (e) => updateParams({ technique: Number(e.target.value) }), children: TECH_KEYS.map((k, i) => (_jsx("option", { value: i, children: t(`workflow.${k}`) }, k))) })] }), _jsx(ParamSlider, { label: t("workflow.symMotifBars"), min: 2, max: 4, step: 1, value: motifBars, format: (v) => `${v}`, onChange: (v) => updateParams({ motifBars: v }) }), _jsx(ParamSlider, { label: t("workflow.symSeed"), min: 0, max: 4095, step: 1, value: seed, onChange: (v) => updateParams({ seed: v }) }), _jsx("div", { style: { fontSize: 10, color: "#888", padding: "4px 0 0 4px" }, children: t("workflow.symMotifHint") })] }) }) }));
}
