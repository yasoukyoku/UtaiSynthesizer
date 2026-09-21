import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSliderWithUnit } from "./ParamSliderWithUnit";
export function SpeedShiftNode(props) {
    const [params, updateParams] = useNodeParams(props);
    const rate = params.rate ?? 1.0;
    return (_jsx(NodeShell, { nodeId: props.id, label: "\u53D8\u901F", icon: "\u23F1", color: "#06b6d4", inputs: 1, outputs: 1, children: _jsx("div", { className: "sep-node-body", children: _jsxs("div", { className: "sep-params", children: [_jsx(ParamSliderWithUnit, { label: "\u64AD\u653E\u901F\u5EA6", unitType: "ratio", title: "Signalsmith \u9891\u8C31\u65F6\u95F4\u62C9\u4F38 \u2014 \u53D8\u901F\u4F46\u4FDD\u97F3\u9AD8. 0.5x = \u534A\u901F, 2.0x = \u500D\u901F (\u97F3\u9AD8\u4E0D\u53D8). \u8303\u56F4 0.25x ~ 4.0x.", min: 0.25, max: 4.0, step: 0.05, value: rate, onChange: (v) => updateParams({ rate: v }) }), _jsx("div", { style: { fontSize: 10, color: "#888", padding: "4px 0 0 4px" }, children: "Rust Signalsmith \u00B7 \u9891\u8C31\u65F6\u95F4\u62C9\u4F38 \u00B7 \u4FDD\u97F3\u9AD8" })] }) }) }));
}
