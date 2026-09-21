import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider, formatRatio } from "./ParamSlider";
import { songNodeColor } from "./SongNodeShared";
export function SongCoverNode(props) {
    const { t } = useTranslation();
    const [params, updateParams] = useNodeParams(props);
    return (_jsx(NodeShell, { nodeId: props.id, label: t("songNode.cover"), icon: "\uD83C\uDFA4", color: songNodeColor, inputs: 3, outputs: 1, children: _jsx("div", { className: "sep-node-body", children: _jsxs("div", { className: "sep-params", children: [_jsxs("div", { className: "sep-param-row", children: [_jsx("label", { children: t("songNode.songName") }), _jsx("input", { type: "text", className: "nodrag", value: params.songName ?? "", onChange: (e) => updateParams({ songName: e.target.value }), style: { width: 110 } })] }), _jsx(ParamSlider, { label: t("songNode.strength"), title: t("songNode.strengthTip"), min: 0, max: 1, step: 0.05, value: params.coverStrength ?? 0.5, onChange: (v) => updateParams({ coverStrength: v }), format: formatRatio })] }) }) }));
}
