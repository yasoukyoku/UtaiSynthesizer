import { jsx as _jsx } from "react/jsx-runtime";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { songNodeColor } from "./SongNodeShared";
export function SongSheetNode(props) {
    const { t } = useTranslation();
    const [params, updateParams] = useNodeParams(props);
    return (_jsx(NodeShell, { nodeId: props.id, label: t("songNode.sheet"), icon: "\uD83C\uDFB9", color: songNodeColor, inputs: 1, outputLabels: [t("songNode.portAbc"), t("songNode.portMidi")], children: _jsx("div", { className: "sep-node-body", children: _jsx("div", { className: "sep-params", children: _jsx("textarea", { value: params.prompt ?? "", onChange: (e) => updateParams({ prompt: e.target.value }), rows: 2, className: "nodrag", style: { width: "100%", fontSize: 11, padding: 6, borderRadius: 6, border: "1px solid #ec4899", background: "#1c1917", color: "#fbcfe8", resize: "none", boxSizing: "border-box" }, placeholder: t("songNode.sheetPromptPlaceholder") }) }) }) }));
}
