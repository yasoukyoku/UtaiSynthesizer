import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { songNodeColor } from "./SongNodeShared";
function SecInput({ value, onChange }) {
    return (_jsx("input", { type: "number", min: 0, step: 0.5, className: "nodrag", value: value, onChange: (e) => onChange(Math.max(0, parseFloat(e.target.value) || 0)), style: { width: 56 } }));
}
export function SongRepaintNode(props) {
    const { t } = useTranslation();
    const [params, updateParams] = useNodeParams(props);
    return (_jsx(NodeShell, { nodeId: props.id, label: t("songNode.repaint"), icon: "\uD83D\uDD8C", color: songNodeColor, inputs: 2, outputs: 1, children: _jsx("div", { className: "sep-node-body", children: _jsxs("div", { className: "sep-params", children: [_jsxs("div", { className: "sep-param-row", children: [_jsx("label", { children: t("songNode.startSec") }), _jsx(SecInput, { value: params.repaintStart ?? 0, onChange: (v) => updateParams({ repaintStart: v }) })] }), _jsxs("div", { className: "sep-param-row", children: [_jsx("label", { children: t("songNode.endSec") }), _jsx(SecInput, { value: params.repaintEnd ?? 0, onChange: (v) => updateParams({ repaintEnd: v }) })] }), _jsxs("div", { className: "sep-param-row", children: [_jsx("label", { children: t("songNode.songName") }), _jsx("input", { type: "text", className: "nodrag", value: params.songName ?? "", onChange: (e) => updateParams({ songName: e.target.value }), style: { width: 96 } })] })] }) }) }));
}
