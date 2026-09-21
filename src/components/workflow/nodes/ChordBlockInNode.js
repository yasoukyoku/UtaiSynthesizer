import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
/** Chord Block In — undefined. */
export function ChordBlockInNode(props) {
    const [params, updateParams] = useNodeParams(props);
    const chords = params.chords ?? "C | Am | F | G";
    const bpm = params.bpm ?? 120;
    const PRESETS = [
        { name: "🎵 流行", v: "C | Am | F | G" },
        { name: "🌧 伤感", v: "Am | F | C | G" },
        { name: "🖤 情感", v: "Am | F | C | E" },
        { name: "🎸 摇滚", v: "E | A | B | E" },
        { name: "🌊 Lo-Fi", v: "Dm | Bb | F | C" },
        { name: "🎩 Jazz", v: "Dm7 | G7 | Cmaj7" },
        { name: "💜 vi-IV-I-V", v: "Am | F | C | G" },
    ];
    return (_jsx(NodeShell, { nodeId: props.id, label: "Chord Block In", icon: "\uD83C\uDFBC", color: "#eab308", inputs: 0, outputs: 1, width: 280, children: _jsx("div", { className: "sep-node-body", children: _jsxs("div", { className: "sep-params", children: [_jsx("div", { style: { fontSize: 10, color: "#eab308", padding: "0 0 6px 2px" }, children: "\u624B\u52A8\u548C\u5F26 \u2192 \u81EA\u52A8\u751F\u6210\u65CB\u5F8B/\u7F16\u66F2" }), _jsx("textarea", { value: chords, onChange: (e) => updateParams({ chords: e.target.value }), rows: 2, style: { width: "100%", fontSize: 12, padding: 6, borderRadius: 6, border: "1px solid #eab308", background: "#1c1917", color: "#fde68a", fontFamily: "monospace", resize: "none", boxSizing: "border-box" }, placeholder: "C | Am | F | G" }), _jsx("div", { style: { display: "flex", gap: 3, flexWrap: "wrap", marginTop: 4 }, children: PRESETS.map(p => (_jsx("button", { onClick: () => updateParams({ chords: p.v }), type: "button", style: { fontSize: 9, padding: "2px 5px", borderRadius: 4, border: "1px solid #a16207", background: "#451a03", color: "#fbbf24", cursor: "pointer", fontFamily: "inherit" }, children: p.name }, p.v))) }), _jsx("div", { style: { display: "flex", gap: 6, marginTop: 6 }, children: _jsxs("label", { style: { fontSize: 10, color: "var(--text-primary)" }, children: ["BPM ", _jsx("input", { type: "number", min: 40, max: 220, value: bpm, onChange: (e) => updateParams({ bpm: +e.target.value }), style: { width: 50, marginLeft: 4, padding: 2, borderRadius: 4, border: "1px solid var(--border-strong)", background: "var(--bg-deep)", color: "var(--text-primary)" } })] }) })] }) }) }));
}
