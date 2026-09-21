import { jsx as _jsx } from "react/jsx-runtime";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
export function DcRemoveNode(props) {
    const { t } = useTranslation();
    useNodeParams(props);
    return (_jsx(NodeShell, { nodeId: props.id, label: t("workflow.nodeDcRemove"), icon: "\uD83E\uDDF9", color: "#94a3b8", inputs: 1, outputs: 1, outputLabels: ["audio"], children: _jsx("div", { className: "sep-node-body", children: _jsx("div", { className: "sep-params", children: _jsx("div", { style: { fontSize: 10, color: "var(--text-muted)", padding: "4px 0 0 4px" }, children: t("workflow.dcRemoveHint") }) }) }) }));
}
