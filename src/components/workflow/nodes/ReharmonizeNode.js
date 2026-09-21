import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";
const STRAT_KEYS = ["symStratDiatonic", "symStratBorrowed", "symStratTritone", "symStratExtension"];
export function ReharmonizeNode(props) {
    const { t } = useTranslation();
    const [params, updateParams] = useNodeParams(props);
    const strategy = params.strategy ?? 0;
    const density = params.density ?? 50;
    const seed = params.seed ?? 0;
    return (_jsx(NodeShell, { nodeId: props.id, label: t("workflow.nodeReharmonize"), icon: "\uD83C\uDFB7", color: "#22d3ee", inputs: 1, outputs: 2, outputLabels: ["chords", "info"], children: _jsx("div", { className: "sep-node-body", children: _jsxs("div", { className: "sep-params", children: [_jsxs("div", { className: "sep-label-row", children: [_jsx("span", { className: "sep-label", children: t("workflow.symStrategy") }), _jsx("select", { className: "sep-select", value: String(strategy), onChange: (e) => updateParams({ strategy: Number(e.target.value) }), children: STRAT_KEYS.map((k, i) => (_jsx("option", { value: i, children: t(`workflow.${k}`) }, k))) })] }), _jsx(ParamSlider, { label: t("workflow.symDensity"), min: 0, max: 100, step: 1, value: density, onChange: (v) => updateParams({ density: v }) }), _jsx(ParamSlider, { label: t("workflow.symSeed"), min: 0, max: 4095, step: 1, value: seed, onChange: (v) => updateParams({ seed: v }) }), _jsx("div", { style: { fontSize: 10, color: "#888", padding: "4px 0 0 4px" }, children: t("workflow.symReharmChordHint") })] }) }) }));
}
