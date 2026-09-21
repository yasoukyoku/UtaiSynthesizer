import { jsx as _jsx } from "react/jsx-runtime";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
export function SpectralCompareNode(props) {
    const { t } = useTranslation();
    useNodeParams(props);
    return (_jsx(NodeShell, { nodeId: props.id, label: t("workflow.nodeSpectralCompare"), icon: "\u2696", color: "#a78bfa", inputs: 2, outputs: 2, outputLabels: [t("workflow.analysisOutAudio"), t("workflow.analysisOutReport")], children: _jsx("div", { className: "sep-node-body", children: _jsx("div", { style: { fontSize: 10, color: "var(--text-muted)", padding: "4px 0 0 4px" }, children: t("workflow.spectralCompareHint") }) }) }));
}
