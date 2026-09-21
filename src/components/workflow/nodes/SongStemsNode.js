import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { DEMUCS_STEM_KINDS } from "../../../lib/models/song-tasks";
import { trackClassLabel, songNodeColor } from "./SongNodeShared";
export function SongStemsNode(props) {
    const { t } = useTranslation();
    const [params, updateParams] = useNodeParams(props);
    return (_jsx(NodeShell, { nodeId: props.id, label: t("songNode.stems"), icon: "\uD83C\uDF9A", color: songNodeColor, inputs: 1, outputLabels: DEMUCS_STEM_KINDS.map(trackClassLabel), children: _jsx("div", { className: "sep-node-body", children: _jsx("div", { className: "sep-params", children: _jsxs("div", { className: "sep-param-row", children: [_jsx("label", { children: t("songNode.songName") }), _jsx("input", { type: "text", className: "nodrag", value: params.songName ?? "", onChange: (e) => updateParams({ songName: e.target.value }), style: { width: 96 } })] }) }) }) }));
}
