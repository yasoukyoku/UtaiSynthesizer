import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";
export function BreathPlannerNode(props) {
    const { t } = useTranslation();
    const [params, updateParams] = useNodeParams(props);
    const minGapBeats = params.minGapBeats ?? 1;
    const lyrics = params.lyrics ?? "";
    return (_jsx(NodeShell, { nodeId: props.id, label: t("workflow.nodeBreathPlanner"), icon: "\uD83D\uDCA8", color: "#60a5fa", inputs: 1, outputs: 2, outputLabels: ["input", "breath"], children: _jsx("div", { className: "sep-node-body", children: _jsxs("div", { className: "sep-params", children: [_jsx(ParamSlider, { label: t("workflow.symMinGap"), min: 0.5, max: 4, step: 0.5, value: minGapBeats, format: (v) => `${v}`, onChange: (v) => updateParams({ minGapBeats: v }) }), _jsxs("div", { className: "sep-label-row", children: [_jsx("span", { className: "sep-label", children: t("workflow.symLyrics") }), _jsx("textarea", { className: "sep-select", rows: 2, style: { resize: "none", fontSize: 10 }, value: lyrics, onChange: (e) => updateParams({ lyrics: e.target.value }) })] }), _jsx("div", { style: { fontSize: 10, color: "#888", padding: "4px 0 0 4px" }, children: t("workflow.symBreathHint") })] }) }) }));
}
