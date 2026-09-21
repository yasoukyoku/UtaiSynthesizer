import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
/** MIDI File In — undefined. */
export function MidiFileInNode(props) {
    const [params, updateParams] = useNodeParams(props);
    const filePath = params.filePath ?? "";
    return (_jsx(NodeShell, { nodeId: props.id, label: "MIDI File In", icon: "\uD83C\uDFB9", color: "#a855f7", inputs: 0, outputs: 1, children: _jsx("div", { className: "sep-node-body", children: _jsxs("div", { className: "sep-params", children: [_jsx("div", { style: { fontSize: 10, color: "#a855f7", padding: "0 0 6px 2px" }, children: "\u5BFC\u5165 .mid \u6587\u4EF6 \u2192 \u9001\u5165\u4E0B\u6E38" }), _jsx("button", { onClick: () => {
                            if (typeof window !== "undefined" && typeof window !== "undefined" && window.__TAURI__) {
                                import("@tauri-apps/plugin-dialog").then(d => d.open({ filters: [{ name: "MIDI", extensions: ["mid", "midi"] }] }))
                                    .then(p => { if (p)
                                    updateParams({ filePath: String(p) }); }).catch(() => { });
                            }
                        }, style: { fontSize: 11, padding: "6px 10px", borderRadius: 6, border: "1px solid #a855f7", background: "#2e1065", color: "#c4b5fd", cursor: "pointer", width: "100%", fontFamily: "inherit" }, children: filePath ? "📄 " + filePath.split(/[\\/]/).pop() : "📂 选择 MIDI 文件..." }), !filePath && _jsx("div", { style: { fontSize: 10, color: "var(--color-error)", padding: "4px 0 0 2px" }, children: "\u26A0\uFE0F \u9700\u8981\u5148\u9009\u62E9\u6587\u4EF6\u518D\u8FD0\u884C" })] }) }) }));
}
