import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { TrackClassChips, trackClassLabel, songNodeColor } from "./SongNodeShared";
export function SongCompleteNode(props) {
    const { t } = useTranslation();
    const [params, updateParams] = useNodeParams(props);
    const classes = params.trackClasses ?? ["drums", "bass", "guitar"];
    return (_jsx(NodeShell, { nodeId: props.id, label: t("songNode.complete"), icon: "\uD83E\uDDE9", color: songNodeColor, inputs: 2, outputLabels: [t("songNode.portMix"), ...classes.map(trackClassLabel)], children: _jsx("div", { className: "sep-node-body", children: _jsxs("div", { className: "sep-params", children: [_jsx("div", { className: "sep-param-row", style: { marginBottom: 2 }, children: _jsx("label", { children: t("songNode.trackClasses") }) }), _jsx(TrackClassChips, { value: classes, multi: true, onChange: (next) => updateParams({ trackClasses: next }) })] }) }) }));
}
