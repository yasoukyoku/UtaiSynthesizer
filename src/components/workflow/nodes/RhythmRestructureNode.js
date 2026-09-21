import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";
export function RhythmRestructureNode(props) {
    const { t } = useTranslation();
    const [params, updateParams] = useNodeParams(props);
    const strength = params.strength ?? 60;
    const beatsPerBar = params.beatsPerBar ?? 4;
    const seed = params.seed ?? 0;
    return (_jsx(NodeShell, { nodeId: props.id, label: t("workflow.nodeRhythmRestructure"), icon: "\uD83E\uDD41", color: "#fbbf24", inputs: 1, outputs: 1, outputLabels: ["notes"], children: _jsx("div", { className: "sep-node-body", children: _jsxs("div", { className: "sep-params", children: [_jsx(ParamSlider, { label: t("workflow.symStrength"), min: 0, max: 100, step: 1, value: strength, onChange: (v) => updateParams({ strength: v }) }), _jsx(ParamSlider, { label: t("workflow.symBeatsPerBar"), min: 2, max: 7, step: 1, value: beatsPerBar, onChange: (v) => updateParams({ beatsPerBar: v }) }), _jsx(ParamSlider, { label: t("workflow.symSeed"), min: 0, max: 4095, step: 1, value: seed, onChange: (v) => updateParams({ seed: v }) }), _jsx("div", { style: { fontSize: 10, color: "#888", padding: "4px 0 0 4px" }, children: t("workflow.symRhythmRestrHint") })] }) }) }));
}
