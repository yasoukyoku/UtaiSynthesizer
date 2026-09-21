import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";
const OP_KEYS = ["symOpRepeatTail", "symOpDropTail", "symOpTransposeTail", "symOpLengthenTail"];
export function StructureEditNode(props) {
    const { t } = useTranslation();
    const [params, updateParams] = useNodeParams(props);
    const op = params.op ?? 0;
    const sectionBars = params.sectionBars ?? 4;
    const semitones = params.semitones ?? 2;
    return (_jsx(NodeShell, { nodeId: props.id, label: t("workflow.nodeStructureEdit"), icon: "\uD83C\uDFD7", color: "#facc15", inputs: 1, outputs: 2, outputLabels: ["chords", "info"], children: _jsx("div", { className: "sep-node-body", children: _jsxs("div", { className: "sep-params", children: [_jsxs("div", { className: "sep-label-row", children: [_jsx("span", { className: "sep-label", children: t("workflow.symOp") }), _jsx("select", { className: "sep-select", value: String(op), onChange: (e) => updateParams({ op: Number(e.target.value) }), children: OP_KEYS.map((k, i) => (_jsx("option", { value: i, children: t(`workflow.${k}`) }, k))) })] }), _jsx(ParamSlider, { label: t("workflow.symSectionBars"), min: 1, max: 8, step: 1, value: sectionBars, format: (v) => `${v}`, onChange: (v) => updateParams({ sectionBars: v }) }), _jsx(ParamSlider, { label: t("workflow.symSemitones"), min: -12, max: 12, step: 1, value: semitones, format: (v) => (v > 0 ? `+${v}` : `${v}`), onChange: (v) => updateParams({ semitones: v }) }), _jsx("div", { style: { fontSize: 10, color: "#888", padding: "4px 0 0 4px" }, children: t("workflow.symStructureHint") })] }) }) }));
}
