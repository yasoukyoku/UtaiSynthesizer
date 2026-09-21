import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { songNodeColor, songTextAreaStyle } from "./SongNodeShared";
export function SongPromptNode(props) {
    const { t } = useTranslation();
    const [params, updateParams] = useNodeParams(props);
    const text = params.text ?? "";
    const presets = [
        { key: "presetPop", v: "pop, upbeat, catchy chorus, modern drums, warm bass" },
        { key: "presetRock", v: "rock, driving guitar, powerful drums, energetic" },
        { key: "presetBallad", v: "ballad, emotional, piano, strings, slow tempo" },
        { key: "presetElectronic", v: "electronic, synth lead, punchy beat, 128 bpm" },
        { key: "presetFolk", v: "folk, acoustic guitar, gentle, storytelling" },
    ];
    return (_jsx(NodeShell, { nodeId: props.id, label: t("songNode.prompt"), icon: "\uD83C\uDFA8", color: songNodeColor, inputs: 0, outputs: 1, outputLabels: [t("songNode.portPrompt")], width: 280, children: _jsx("div", { className: "sep-node-body", children: _jsxs("div", { className: "sep-params", children: [_jsx("div", { style: { display: "flex", gap: 3, flexWrap: "wrap", paddingBottom: 4 }, children: presets.map((p) => (_jsx("button", { type: "button", onClick: () => updateParams({ text: p.v }), className: "nodrag", style: { fontSize: 9, padding: "2px 5px", borderRadius: 4, border: "1px solid #9f1239", background: "#4c0519", color: "#fda4af", cursor: "pointer", fontFamily: "inherit" }, children: t(`songNode.${p.key}`) }, p.key))) }), _jsx("textarea", { value: text, onChange: (e) => updateParams({ text: e.target.value }), rows: 4, className: "nodrag", style: songTextAreaStyle, placeholder: t("songNode.promptPlaceholder") })] }) }) }));
}
