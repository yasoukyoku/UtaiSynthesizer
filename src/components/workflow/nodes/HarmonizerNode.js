import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
const CHORD_STYLES = ["POP_STANDARD", "POP_COMPLEX", "DARK", "RANDB", "NOCONSTRAINT"];
const STYLE_ZH = {
    POP_STANDARD: "标准流行", POP_COMPLEX: "丰富流行", DARK: "暗黑", RANDB: "随机变化", NOCONSTRAINT: "自由分析",
};
export function HarmonizerNode(props) {
    const [params, updateParams] = useNodeParams(props);
    const chordStyle = params.chordStyle ?? "POP_STANDARD";
    const chordsPerBar = params.chordsPerBar ?? 1;
    const key = params.key ?? "auto";
    return (_jsx(NodeShell, { nodeId: props.id, label: "\u548C\u58F0\u5C42", icon: "\uD83C\uDFB6", color: "#06b6d4", inputs: 1, outputs: 2, children: _jsx("div", { className: "sep-node-body", children: _jsxs("div", { className: "sep-params", children: [_jsxs("div", { className: "sep-label-row", children: [_jsx("span", { className: "sep-label", children: "\u548C\u5F26\u98CE\u683C" }), _jsx("select", { className: "sep-select", value: chordStyle, onChange: (e) => updateParams({ chordStyle: e.target.value }), children: CHORD_STYLES.map((s) => _jsx("option", { value: s, children: STYLE_ZH[s] }, s)) })] }), _jsxs("div", { className: "sep-label-row", children: [_jsx("span", { className: "sep-label", children: "\u6BCF\u5C0F\u8282\u548C\u5F26\u6570" }), _jsxs("select", { className: "sep-select", value: chordsPerBar, onChange: (e) => updateParams({ chordsPerBar: Number(e.target.value) }), children: [_jsx("option", { value: 1, children: "1 (\u6574\u5C0F\u8282)" }), _jsx("option", { value: 2, children: "2 (\u534A\u5C0F\u8282)" })] })] }), _jsxs("div", { className: "sep-label-row", children: [_jsx("span", { className: "sep-label", children: "\u8C03\u6027" }), _jsx("input", { type: "text", className: "sep-input", style: { width: 80 }, value: key, placeholder: "auto", onChange: (e) => updateParams({ key: e.target.value }) })] }), _jsx("div", { style: { fontSize: 10, color: "#888", padding: "4px 0 0 4px" }, children: "\u7EAF\u524D\u7AEF \u00B7 MIDI-SAG \u548C\u58F0\u5316\u5F15\u64CE \u00B7 \u786E\u5B9A\u6027\u8F93\u51FA" })] }) }) }));
}
