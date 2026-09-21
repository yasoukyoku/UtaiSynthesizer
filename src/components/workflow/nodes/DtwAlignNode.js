import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";
export function DtwAlignNode(props) {
    const { t } = useTranslation();
    const [params, updateParams] = useNodeParams(props);
    const band = Math.max(0, Math.min(64, Math.round(params.band ?? 0)));
    return (_jsx(NodeShell, { nodeId: props.id, label: t("workflow.nodeDtwAlign"), icon: "\uD83E\uDDED", color: "#f472b6", inputs: 2, outputs: 2, outputLabels: [t("workflow.analysisOutAudio"), t("workflow.analysisOutReport")], children: _jsx("div", { className: "sep-node-body", children: _jsxs("div", { className: "sep-params", children: [_jsx(ParamSlider, { label: t("workflow.dtwBand"), title: t("workflow.dtwHint"), min: 0, max: 64, step: 1, value: band, format: (v) => `${Math.round(v)}`, onChange: (v) => updateParams({ band: Math.round(v) }) }), _jsx("div", { style: { fontSize: 10, color: "var(--text-muted)", padding: "4px 0 0 4px" }, children: t("workflow.dtwHint") })] }) }) }));
}
