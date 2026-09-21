import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";
export function MidiHumanizeNode(props) {
    const { t } = useTranslation();
    const [params, updateParams] = useNodeParams(props);
    const timingMs = params.timingMs ?? 12;
    const velocityJitter = params.velocityJitter ?? 8;
    const seed = params.seed ?? 0;
    return (_jsx(NodeShell, { nodeId: props.id, label: t("workflow.nodeMidiHumanize"), icon: "\uD83C\uDFB9", color: "#38bdf8", inputs: 1, outputs: 1, outputLabels: ["notes"], children: _jsx("div", { className: "sep-node-body", children: _jsxs("div", { className: "sep-params", children: [_jsx(ParamSlider, { label: t("workflow.symTimingMs"), min: 0, max: 50, step: 1, value: timingMs, format: (v) => `${v} ms`, onChange: (v) => updateParams({ timingMs: v }) }), _jsx(ParamSlider, { label: t("workflow.symVelJitter"), min: 0, max: 30, step: 1, value: velocityJitter, onChange: (v) => updateParams({ velocityJitter: v }) }), _jsx(ParamSlider, { label: t("workflow.symSeed"), min: 0, max: 4095, step: 1, value: seed, onChange: (v) => updateParams({ seed: v }) }), _jsx("div", { style: { fontSize: 10, color: "#888", padding: "4px 0 0 4px" }, children: t("workflow.symHumanizeHint") })] }) }) }));
}
