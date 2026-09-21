import { jsx as _jsx } from "react/jsx-runtime";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
export function ComplianceCheckNode(props) {
    const { t } = useTranslation();
    return (_jsx(NodeShell, { nodeId: props.id, label: t("workflow.nodeComplianceCheck"), icon: "\uD83E\uDE7A", color: "#22c55e", inputs: 1, outputs: 2, outputLabels: [t("workflow.complianceOutAudio"), t("workflow.complianceOutReport")], children: _jsx("div", { className: "sep-node-body", children: _jsx("div", { style: { fontSize: 10, color: "var(--text-muted)", padding: "4px 0 0 4px" }, children: t("workflow.complianceHint") }) }) }));
}
