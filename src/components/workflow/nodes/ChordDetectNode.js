import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
export function ChordDetectNode(props) {
    const [params, updateParams] = useNodeParams(props);
    const beatsPerBar = params.beatsPerBar ?? 4;
    return (_jsx(NodeShell, { nodeId: props.id, label: "\u548C\u5F26\u8BC6\u522B", icon: "\uD83C\uDFBC", color: "#a78bfa", inputs: 1, outputs: 2, children: _jsx("div", { className: "sep-node-body", children: _jsxs("div", { className: "sep-params", children: [_jsxs("div", { className: "sep-label-row", children: [_jsx("span", { className: "sep-label", children: "\u6BCF\u62CD\u7A97\u6570" }), _jsxs("select", { className: "sep-select", value: beatsPerBar, onChange: (e) => updateParams({ beatsPerBar: Number(e.target.value) }), children: [_jsx("option", { value: 4, children: "4/4 \u62CD (\u9ED8\u8BA4)" }), _jsx("option", { value: 3, children: "3/4 \u62CD" }), _jsx("option", { value: 6, children: "6/8 \u62CD" })] })] }), _jsx("div", { style: { fontSize: 10, color: "var(--text-muted)", padding: "4px 0 0 4px" }, children: "MIDI \u97F3\u7B26\u9A71\u52A8 \u00B7 analyzeChords \u00B7 \u786E\u5B9A\u6027\u8F93\u51FA" }), _jsx("div", { style: { fontSize: 10, color: "var(--text-muted)", padding: "2px 0 0 4px" }, children: "\uD83D\uDCA1 \u63A5 MIDI File In \u6216 AMT \u8F6C\u8C31\u8282\u70B9" })] }) }) }));
}
