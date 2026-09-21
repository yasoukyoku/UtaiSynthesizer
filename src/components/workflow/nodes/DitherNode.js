import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
export function DitherNode(props) {
    const { t } = useTranslation();
    const [params, updateParams] = useNodeParams(props);
    const ditherType = Math.max(0, Math.min(2, Math.round(params.ditherType ?? 2)));
    return (_jsx(NodeShell, { nodeId: props.id, label: t("workflow.nodeDither"), icon: "\uD83C\uDF9A", color: "#64748b", inputs: 1, outputs: 1, outputLabels: ["16-bit"], children: _jsx("div", { className: "sep-node-body", children: _jsxs("div", { className: "sep-params", children: [_jsxs("div", { className: "sep-label-row", children: [_jsx("span", { className: "sep-label", children: t("workflow.ditherType") }), _jsxs("select", { className: "sep-select", value: ditherType, onChange: (e) => updateParams({ ditherType: Number(e.target.value) }), children: [_jsx("option", { value: 0, children: t("workflow.ditherNone") }), _jsx("option", { value: 1, children: "TPDF" }), _jsxs("option", { value: 2, children: ["TPDF + ", t("workflow.ditherShaped")] })] })] }), _jsx("div", { style: { fontSize: 10, color: "var(--text-muted)", padding: "4px 0 0 4px" }, children: t("workflow.ditherHint") })] }) }) }));
}
