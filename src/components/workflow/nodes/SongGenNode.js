import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
// P2-14 songGen 歌曲模型核心节点（规划 11.2/11.3）：
// 0:歌词文本(可空) 1:提示词文本(可空) 2:参考音频(可空) → 0:整首 1:MIDI 2:字幕 3+i:分轨。
// 歌词/提示词优先取连线（songLyrics/songPrompt），未连线时用节点内文本框。
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";
import { ACE_TRACK_CLASSES } from "../../../lib/models/song-tasks";
import { trackClassLabel, songNodeColor, songTextAreaStyle } from "./SongNodeShared";
export function SongGenNode(props) {
    const { t } = useTranslation();
    const [params, updateParams] = useNodeParams(props);
    const [showAdv, setShowAdv] = useState(false);
    const model = params.model ?? "acestep-v1.5";
    const isYue2 = model.startsWith("yue2");
    const songTask = params.songTask ?? "generate";
    const wantStems = params.wantStems ?? false;
    const lyricsConn = params.lyrics ?? "";
    const promptConn = params.prompt ?? "";
    // 输出端口：0=整首 1=MIDI 2=LRC；勾选分轨后 3+i 按轨种展开（与引擎映射一致）
    const outLabels = [
        t("songNode.portMix"),
        t("songNode.portMidi"),
        t("songNode.portLrc"),
        ...(wantStems ? ACE_TRACK_CLASSES.map(trackClassLabel) : []),
    ];
    return (_jsx(NodeShell, { nodeId: props.id, label: t("songNode.gen"), icon: "\uD83C\uDFB5", color: songNodeColor, inputs: 3, outputLabels: outLabels, width: 280, children: _jsx("div", { className: "sep-node-body", children: _jsxs("div", { className: "sep-params", children: [_jsxs("div", { className: "sep-param-row", children: [_jsx("label", { children: t("songNode.model") }), _jsxs("select", { value: model, onChange: (e) => updateParams({ model: e.target.value }), className: "nodrag", children: [_jsx("option", { value: "acestep-v1.5", children: "ACE-Step v1.5" }), _jsx("option", { value: "yue2-3b", children: "YuE2-3B" })] })] }), _jsxs("div", { className: "sep-param-row", children: [_jsx("label", { children: t("songNode.genTask") }), _jsxs("select", { value: songTask, onChange: (e) => updateParams({ songTask: e.target.value }), className: "nodrag", children: [_jsx("option", { value: "generate", children: t("songNode.taskWrite") }), _jsx("option", { value: "instrumental", children: t("songNode.taskInst") })] })] }), _jsx(ParamSlider, { label: t("songNode.duration"), min: 0, max: 300, step: 5, value: params.durationSec ?? 0, onChange: (v) => updateParams({ durationSec: v }), format: (v) => (v === 0 ? t("songNode.durationAuto") : `${v}s`) }), _jsxs("div", { className: "sep-param-row", children: [_jsx("label", { children: t("songNode.seed") }), _jsx("input", { type: "number", min: 0, step: 1, className: "nodrag", value: params.seed ?? 0, onChange: (e) => updateParams({ seed: Math.max(0, parseInt(e.target.value) || 0) }) })] }), _jsxs("div", { className: "sep-param-row", children: [_jsx("label", { children: t("songNode.outputs") }), _jsxs("span", { className: "nodrag", style: { display: "flex", gap: 6 }, children: [_jsxs("label", { style: { display: "flex", gap: 2, alignItems: "center", cursor: "pointer" }, children: [t("songNode.wantStems"), _jsx("input", { type: "checkbox", checked: wantStems, onChange: (e) => updateParams({ wantStems: e.target.checked }) })] }), _jsxs("label", { style: { display: "flex", gap: 2, alignItems: "center", cursor: "pointer" }, children: ["MIDI", _jsx("input", { type: "checkbox", checked: params.wantMidi ?? false, onChange: (e) => updateParams({ wantMidi: e.target.checked }) })] }), _jsxs("label", { style: { display: "flex", gap: 2, alignItems: "center", cursor: "pointer" }, children: ["LRC", _jsx("input", { type: "checkbox", checked: params.wantLrc ?? false, onChange: (e) => updateParams({ wantLrc: e.target.checked }) })] })] })] }), _jsx("textarea", { value: lyricsConn, onChange: (e) => updateParams({ lyrics: e.target.value }), rows: 2, className: "nodrag", style: songTextAreaStyle, placeholder: t("songNode.lyricsPlaceholder") }), _jsx("textarea", { value: promptConn, onChange: (e) => updateParams({ prompt: e.target.value }), rows: 2, className: "nodrag", style: songTextAreaStyle, placeholder: t("songNode.promptPlaceholder") }), _jsxs("button", { type: "button", className: "nodrag", onClick: () => setShowAdv((s) => !s), style: { fontSize: 10, background: "none", border: "none", color: "#f9a8d4", cursor: "pointer", textAlign: "left", padding: 0 }, children: [showAdv ? "▾" : "▸", " ", t("songNode.advanced")] }), showAdv && (isYue2 ? (_jsxs("div", { className: "sep-param-row", children: [_jsx("label", { children: "CoT" }), _jsxs("select", { value: (params.cot ?? "full"), onChange: (e) => updateParams({ cot: e.target.value }), className: "nodrag", children: [_jsx("option", { value: "full", children: "full" }), _jsx("option", { value: "off", children: "off" })] })] })) : (_jsxs(_Fragment, { children: [_jsx(ParamSlider, { label: "guidance", min: 0, max: 15, step: 0.5, value: params.guidanceScale ?? 0, onChange: (v) => updateParams({ guidanceScale: v || undefined }), format: (v) => (v === 0 ? t("songNode.default") : v.toFixed(1)) }), _jsx(ParamSlider, { label: "steps", min: 0, max: 100, step: 5, value: params.numInferenceSteps ?? 0, onChange: (v) => updateParams({ numInferenceSteps: v || undefined }), format: (v) => (v === 0 ? t("songNode.default") : String(v)) })] })))] }) }) }));
}
