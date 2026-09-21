import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useCallback } from "react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
const selStyle = {
    marginTop: 2,
    padding: "4px 6px",
    borderRadius: 6,
    border: "1px solid var(--border-subtle)",
    background: "var(--bg-surface)",
    color: "var(--text-primary)",
    fontSize: 12,
};
const labelStyle = {
    display: "flex",
    flexDirection: "column",
    gap: 2,
    fontSize: 12,
    color: "var(--text-secondary)",
};
export function MelodySimilarityNode(props) {
    const { t } = useTranslation();
    const [params, updateParams] = useNodeParams(props);
    const threshold = params.threshold ?? 35;
    const handleThresholdChange = useCallback((e) => {
        const val = Math.max(0, Math.min(100, parseInt(e.target.value, 10) || 35));
        updateParams({ threshold: val });
    }, [updateParams]);
    return (_jsx(NodeShell, { nodeId: props.id, label: t("workflow.nodeMelodySimilarity"), icon: "\uD83D\uDD0D", color: "#f59e0b", inputs: 2, outputs: 2, outputLabels: [t("workflow.analysisOutMidi"), t("workflow.analysisOutReport")], children: _jsxs("div", { className: "sep-node-body", children: [_jsxs("label", { style: labelStyle, children: [_jsx("span", { children: t("workflow.melodySimilarityThreshold") }), _jsxs("div", { style: { display: "flex", alignItems: "center", gap: 6 }, children: [_jsx("input", { type: "range", min: "0", max: "100", value: threshold, onChange: handleThresholdChange, style: { flex: 1, minWidth: 100 } }), _jsx("input", { type: "number", value: threshold, onChange: handleThresholdChange, style: { ...selStyle, width: 50, textAlign: "center" }, min: "0", max: "100" }), _jsx("span", { style: { fontSize: 11, color: "var(--text-muted)" }, children: "%" })] })] }), _jsx("div", { style: { fontSize: 10, color: "var(--text-muted)", padding: "4px 0 0 0" }, children: t("workflow.melodySimilarityHint") })] }) }));
}
