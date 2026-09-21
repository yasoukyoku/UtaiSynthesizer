import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { TrackClassChips, songNodeColor } from "./SongNodeShared";
export function SongLegoNode(props) {
    const { t } = useTranslation();
    const [params, updateParams] = useNodeParams(props);
    const trackName = params.trackName ?? "guitar";
    return (_jsx(NodeShell, { nodeId: props.id, label: t("songNode.lego"), icon: "\uD83E\uDDF1", color: songNodeColor, inputs: 2, outputs: 1, children: _jsx("div", { className: "sep-node-body", children: _jsxs("div", { className: "sep-params", children: [_jsx("div", { className: "sep-param-row", style: { marginBottom: 2 }, children: _jsx("label", { children: t("songNode.legoTrack") }) }), _jsx(TrackClassChips, { value: [trackName], multi: false, onChange: (next) => updateParams({ trackName: next[0] }) })] }) }) }));
}
