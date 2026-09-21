import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSliderWithUnit } from "./ParamSliderWithUnit";
export function SaturateNode(props) {
    const { t } = useTranslation();
    const [params, updateParams] = useNodeParams(props);
    const drive = Math.max(0, Math.min(1, params.drive ?? 0.05));
    return (_jsx(NodeShell, { nodeId: props.id, label: t("workflow.nodeSaturate"), icon: "\uD83D\uDD25", color: "#f97316", inputs: 1, outputs: 1, outputLabels: ["audio"], children: _jsx("div", { className: "sep-node-body", children: _jsxs("div", { className: "sep-params", children: [_jsx(ParamSliderWithUnit, { label: t("workflow.saturateDrive"), unitType: "percent", title: t("workflow.saturateDriveTitle"), min: 0, max: 1, step: 0.01, value: drive, onChange: (v) => updateParams({ drive: v }) }), _jsx("div", { style: { fontSize: 10, color: "var(--text-muted)", padding: "4px 0 0 4px" }, children: t("workflow.saturateHint") })] }) }) }));
}
