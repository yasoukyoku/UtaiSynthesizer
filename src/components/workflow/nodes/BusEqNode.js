import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSliderWithUnit } from "./ParamSliderWithUnit";
const GAIN_BANDS = [
    { key: "eqGain0", labelKey: "workflow.eqBandLow", titleKey: "workflow.eqBandGainTitle" },
    { key: "eqGain1", labelKey: "workflow.eqBandMid", titleKey: "workflow.eqBandGainTitle" },
    { key: "eqGain2", labelKey: "workflow.eqBandMidHigh", titleKey: "workflow.eqBandGainTitle" },
    { key: "eqGain3", labelKey: "workflow.eqBandHigh", titleKey: "workflow.eqBandGainTitle" },
];
export function BusEqNode(props) {
    const { t } = useTranslation();
    const [params, updateParams] = useNodeParams(props);
    const eqHpf = (params.eqHpf ?? 1) > 0 ? 1 : 0;
    return (_jsx(NodeShell, { nodeId: props.id, label: t("workflow.nodeBusEq"), icon: "\uD83C\uDF9A", color: "#38bdf8", inputs: 1, outputs: 1, outputLabels: ["audio"], children: _jsx("div", { className: "sep-node-body", children: _jsxs("div", { className: "sep-params", children: [_jsxs("div", { className: "sep-label-row", children: [_jsx("span", { className: "sep-label", children: t("workflow.eqBandHpf") }), _jsxs("select", { className: "sep-select", value: eqHpf, title: t("workflow.eqBandGainTitle"), onChange: (e) => updateParams({ eqHpf: Number(e.target.value) }), children: [_jsx("option", { value: 0, children: t("workflow.eqOff") }), _jsx("option", { value: 1, children: t("workflow.eqOn") })] })] }), GAIN_BANDS.map((b) => (_jsx(ParamSliderWithUnit, { label: t(b.labelKey), unitType: "db", title: t(b.titleKey), min: -12, max: 12, step: 0.5, value: (params[b.key] ?? 0), onChange: (v) => updateParams({ [b.key]: v }) }, b.key))), _jsx("div", { style: { fontSize: 10, color: "var(--text-muted)", padding: "4px 0 0 4px" }, children: t("workflow.busEqHint") })] }) }) }));
}
