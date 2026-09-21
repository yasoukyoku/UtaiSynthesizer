import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
const STYLES = [
    "pop", "rock", "jazz", "edm", "latin", "funk", "rnb", "ballad", "country",
    "lofi", "trap", "reggae", "bossanova", "kpop", "chinese", "ambient", "techno",
];
const STYLE_ZH = {
    pop: "流行", rock: "摇滚", jazz: "爵士", edm: "电子", latin: "拉丁",
    funk: "放克", rnb: "R&B", ballad: "抒情", country: "乡村",
    lofi: "Lo-Fi", trap: "Trap", reggae: "雷鬼", bossanova: "Bossa Nova",
    kpop: "K-Pop", chinese: "中国风", ambient: "氛围", techno: "Techno",
};
const MOODS = ["neutral", "happy", "sad", "energetic", "chill", "dark"];
const MOOD_ZH = {
    neutral: "中性", happy: "开心", sad: "悲伤", energetic: "活力", chill: "放松", dark: "暗黑",
};
// 与引擎 autoArrange 的 tracks 表逐位对齐；末位是和弦摘要（chords 端口，内联 JSON）。
const OUT_LABELS = [
    "🥁 Drums", "🎸 Bass", "🎹 Piano", "🪕 GuitarArp", "🎶 Strum", "🎼 E.Piano",
    "🎻 Strings", "🪟 Pad", "🎛 SynthPad", "💠 Pluck", "✨ Lead", "🎵 Chords",
];
export function AutoArrangeNode(props) {
    const [params, updateParams] = useNodeParams(props);
    const style = params.style ?? "pop";
    const mood = params.mood ?? "neutral";
    const autoReplace = params.autoReplace ?? true;
    return (_jsx(NodeShell, { nodeId: props.id, label: "AI \u81EA\u52A8\u7F16\u66F2", icon: "\uD83E\uDD41", color: "#ec4899", inputs: 1, outputLabels: OUT_LABELS, children: _jsx("div", { className: "sep-node-body", children: _jsxs("div", { className: "sep-params", children: [_jsxs("div", { className: "sep-label-row", children: [_jsx("span", { className: "sep-label", children: "\u98CE\u683C" }), _jsx("select", { className: "sep-select", value: style, onChange: (e) => updateParams({ style: e.target.value }), children: STYLES.map((s) => _jsx("option", { value: s, children: STYLE_ZH[s] }, s)) })] }), _jsxs("div", { className: "sep-label-row", children: [_jsx("span", { className: "sep-label", children: "\u60C5\u7EEA" }), _jsx("select", { className: "sep-select", value: mood, onChange: (e) => updateParams({ mood: e.target.value }), children: MOODS.map((m) => _jsx("option", { value: m, children: MOOD_ZH[m] }, m)) })] }), _jsxs("label", { className: "sep-checkbox-row", children: [_jsx("input", { type: "checkbox", checked: autoReplace, onChange: (e) => updateParams({ autoReplace: e.target.checked }) }), _jsx("span", { children: "\u518D\u751F\u6210\u65F6\u81EA\u52A8\u66FF\u6362\u65E7\u8F68" })] }), _jsx("div", { style: { fontSize: 10, color: "#888", padding: "4px 0 0 4px" }, children: "\u7EAF\u524D\u7AEF \u00B7 22 \u79CD\u98CE\u683C \u00B7 11 \u7C7B\u4E50\u5668\u81EA\u7531\u7EC4\u5408 \u00B7 \u786E\u5B9A\u6027\u8F93\u51FA" })] }) }) }));
}
