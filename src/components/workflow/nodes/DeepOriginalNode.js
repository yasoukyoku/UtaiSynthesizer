import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";
export function DeepOriginalNode(props) {
    const [params, updateParams] = useNodeParams(props);
    const semitones = params.semitones ?? 3;
    const rate = params.rate ?? 1.0;
    const style = params.style ?? "trap";
    const enableRVC = params.enableRVC ?? false;
    const randomSeed = params.randomSeed ?? 0;
    const autoInstruments = params.instruments ?? ["drums", "bass", "piano", "chords", "melody"];
    const STYLES = [
        { v: "trap", label: "Trap (常用)" },
        { v: "lofi", label: "Lo-Fi" },
        { v: "techno", label: "Techno" },
        { v: "edm", label: "电子 EDM" },
        { v: "pop", label: "流行" },
        { v: "kpop", label: "K-Pop" },
        { v: "chinese", label: "中国风" },
        { v: "reggae", label: "雷鬼" },
        { v: "ambient", label: "氛围" },
        { v: "rock", label: "摇滚" },
    ];
    const ALL_INSTRUMENTS = [
        { v: "drums", label: "🥁 鼓" },
        { v: "bass", label: "🎸 贝斯" },
        { v: "piano", label: "🎹 钢琴" },
        { v: "chords", label: "🪟 Pad" },
        { v: "melody", label: "✨ AI Lead" },
    ];
    const toggleInst = (v) => {
        if (autoInstruments.includes(v)) {
            updateParams({ instruments: autoInstruments.filter((x) => x !== v) });
        }
        else {
            updateParams({ instruments: [...autoInstruments, v] });
        }
    };
    return (_jsx(NodeShell, { nodeId: props.id, label: "\u26A1\u6DF1\u5EA6\u539F\u521B", icon: "\u2728", color: "#f59e0b", inputs: 1, outputs: 5, children: _jsx("div", { className: "sep-node-body", children: _jsxs("div", { className: "sep-params", children: [_jsx("div", { style: { fontSize: 11, color: "var(--color-warning)", padding: "0 0 6px 2px" }, children: "\u4E00\u952E\u94FE: \u97F3\u9891\u2192\u53D8\u901F\u2192\u53D8\u8C03\u2192AMT\u2192\u548C\u5F26\u2192\u7F16\u66F2" }), _jsx(ParamSlider, { label: "\u53D8\u8C03 (\u534A\u97F3)", title: "\u628A\u539F\u66F2\u79FB\u8C03. \u00B13/4/5/7 \u534A\u97F3\u901A\u5E38\u4E0D\u4F1A\u88AB\u8BC6\u522B\u4E3A\u539F\u66F2", min: -12, max: 12, step: 1, value: semitones, format: (v) => (v > 0 ? `+${v}` : `${v}`), onChange: (v) => updateParams({ semitones: v }) }), _jsx(ParamSlider, { label: "\u53D8\u901F (x)", title: "\u64AD\u653E\u901F\u7387. 0.8x / 1.25x \u6539\u8282\u594F\u4E0D\u8C03\u5F0F", min: 0.7, max: 1.5, step: 0.05, value: rate, format: (v) => `${v.toFixed(2)}x`, onChange: (v) => updateParams({ rate: v }) }), _jsx(ParamSlider, { label: "\uD83C\uDFB2 \u968F\u673A\u79CD\u5B50", title: "0 = \u786E\u5B9A\u6027 (\u540C\u8F93\u5165\u540C\u8F93\u51FA). \u6539\u503C = \u6BCF\u6B21\u751F\u6210\u4E0D\u540C\u65CB\u5F8B/\u8282\u594F\u53D8\u4F53", min: 0, max: 9999, step: 1, value: randomSeed, format: (v) => (v === 0 ? "🎯 确定" : `#${v}`), onChange: (v) => updateParams({ randomSeed: v }) }), _jsxs("div", { className: "sep-label-row", children: [_jsx("span", { className: "sep-label", children: "\u7F16\u66F2\u98CE\u683C" }), _jsx("select", { className: "sep-select", value: style, onChange: (e) => updateParams({ style: e.target.value }), children: STYLES.map((s) => _jsx("option", { value: s.v, children: s.label }, s.v)) })] }), _jsx("div", { style: { fontSize: 10, color: "var(--text-muted)", padding: "4px 0 2px 2px" }, children: "\u4E50\u5668\u8F68:" }), _jsx("div", { style: { display: "flex", flexWrap: "wrap", gap: 3, padding: "0 2px" }, children: ALL_INSTRUMENTS.map((inst) => {
                            const on = autoInstruments.includes(inst.v);
                            return (_jsx("button", { type: "button", className: `sep-inst-chip ${on ? "on" : ""}`, onClick: () => toggleInst(inst.v), style: {
                                    fontSize: 10,
                                    padding: "2px 6px",
                                    borderRadius: 6,
                                    border: `1px solid ${on ? "#f59e0b" : "var(--border-strong)"}`,
                                    background: on ? "#78350f" : "transparent",
                                    color: on ? "var(--color-warning)" : "var(--text-muted)",
                                    cursor: "pointer",
                                    fontFamily: "inherit",
                                }, children: inst.label }, inst.v));
                        }) }), _jsxs("label", { className: "sep-checkbox-row", style: { marginTop: 6 }, children: [_jsx("input", { type: "checkbox", checked: enableRVC, onChange: (e) => updateParams({ enableRVC: e.target.checked }) }), _jsx("span", { style: { fontSize: 11 }, children: "RVC \u6362\u58F0 (\u7528\u4F60\u8BAD\u7EC3\u7684\u97F3\u8272\u66FF\u539F\u5531)" })] }), _jsx("div", { style: { fontSize: 10, color: "var(--color-error)", padding: "4px 0 0 4px" }, children: "\u26A0\uFE0F \u9700\u8981 Tauri \u684C\u9762\u7248 + AMT \u6A21\u578B" })] }) }) }));
}
