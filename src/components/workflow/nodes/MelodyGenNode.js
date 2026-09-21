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
const MOODS = ["neutral", "happy", "sad", "energetic", "chill", "epic"];
const MOOD_ZH = {
    neutral: "中性", happy: "开心", sad: "悲伤", energetic: "活力", chill: "放松", epic: "史诗",
};
export function MelodyGenNode(props) {
    const [params, updateParams] = useNodeParams(props);
    const style = params.style ?? "pop";
    const mood = params.mood ?? "neutral";
    return (_jsx(NodeShell, { nodeId: props.id, label: "AI \u65CB\u5F8B", icon: "\uD83C\uDFB5", color: "#10b981", inputs: 1, outputs: 2, children: _jsx("div", { className: "sep-node-body", children: _jsxs("div", { className: "sep-params", children: [_jsxs("div", { className: "sep-label-row", children: [_jsx("span", { className: "sep-label", children: "\u98CE\u683C" }), _jsx("select", { className: "sep-select", value: style, onChange: (e) => updateParams({ style: e.target.value }), children: STYLES.map((s) => _jsx("option", { value: s, children: STYLE_ZH[s] }, s)) })] }), _jsxs("div", { className: "sep-label-row", children: [_jsx("span", { className: "sep-label", children: "\u60C5\u7EEA" }), _jsx("select", { className: "sep-select", value: mood, onChange: (e) => updateParams({ mood: e.target.value }), children: MOODS.map((m) => _jsx("option", { value: m, children: MOOD_ZH[m] }, m)) })] }), _jsx("div", { style: { fontSize: 10, color: "#888", padding: "4px 0 0 4px" }, children: "\u7EAF\u524D\u7AEF \u00B7 \u548C\u5F26\u9A71\u52A8\u786E\u5B9A\u6027 seed \u00B7 \u540C\u8F93\u5165\u540C\u8F93\u51FA" })] }) }) }));
}
