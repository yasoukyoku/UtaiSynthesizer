import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { songNodeColor, songTextAreaStyle } from "./SongNodeShared";
const STRUCT_TAGS = ["[Verse]", "[Chorus]", "[Bridge]", "[Outro]", "[Instrumental]"];
export function SongLyricsNode(props) {
    const { t } = useTranslation();
    const [params, updateParams] = useNodeParams(props);
    const text = params.text ?? "";
    return (_jsx(NodeShell, { nodeId: props.id, label: t("songNode.lyrics"), icon: "\u270D", color: songNodeColor, inputs: 0, outputs: 1, outputLabels: [t("songNode.portLyrics")], width: 280, children: _jsx("div", { className: "sep-node-body", children: _jsxs("div", { className: "sep-params", children: [_jsx("div", { style: { display: "flex", gap: 3, flexWrap: "wrap", paddingBottom: 4 }, children: STRUCT_TAGS.map((tag) => (_jsx("button", { type: "button", onClick: () => updateParams({ text: text ? `${text}\n${tag}\n` : `${tag}\n` }), className: "nodrag", style: { fontSize: 9, padding: "2px 5px", borderRadius: 4, border: "1px solid #9d174d", background: "#500724", color: "#f9a8d4", cursor: "pointer", fontFamily: "inherit" }, children: tag }, tag))) }), _jsx("textarea", { value: text, onChange: (e) => updateParams({ text: e.target.value }), rows: 6, className: "nodrag", style: songTextAreaStyle, placeholder: t("songNode.lyricsPlaceholder") })] }) }) }));
}
