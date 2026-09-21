import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useCallback, useState } from "react";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { useTranslation } from "react-i18next";
import { MUSCRIPTOR_INSTRUMENTS, MUSCRIPTOR_ZH_LABELS } from "../../../lib/models/muscriptor-instruments";
const MIDI_MODES = [
    { value: "smart", labelKey: "amt.modeSmart", multi: true },
    { value: "vocal_split", labelKey: "amt.modeVocalSplit", multi: true },
    { value: "six_stem_split", labelKey: "amt.modeSixStem", multi: true },
    { value: "piano_transkun", labelKey: "amt.modePiano", multi: false },
    { value: "piano_transkun_v2_aug", labelKey: "amt.modePianoAug", multi: false },
    { value: "piano_aria_amt", labelKey: "amt.modeAria", multi: false },
    { value: "piano_bytedance_pedal", labelKey: "amt.modeBytedance", multi: false },
];
const BACKENDS = [
    { value: "yourmt3", labelKey: "amt.backendYourmt3" },
    { value: "miros", labelKey: "amt.backendMiros" },
    { value: "muscriptor", labelKey: "amt.backendMuscriptor" },
];
const QUANTIZE_GRIDS = ["off", "1/4", "1/8", "1/16", "1/32", "1/64"];
const selStyle = {
    marginTop: 2,
    padding: "4px 6px",
    borderRadius: 6,
    border: "1px solid var(--border-subtle)",
    background: "var(--bg-surface)",
    color: "var(--text-primary)",
    fontSize: 12,
};
const labelStyle = {
    display: "flex",
    flexDirection: "column",
    gap: 2,
    fontSize: 12,
    color: "var(--text-secondary)",
};
export function AmtMidiNode(props) {
    const { t } = useTranslation();
    const [params, updateParams] = useNodeParams(props);
    const midiMode = params.midiMode ?? "smart";
    const useGpu = params.useGpu ?? true;
    const backend = params.backend ?? "yourmt3";
    const quantizeGrid = params.quantizeGrid ?? "off";
    // 轨道布局：multi_track=全轨道(按乐器拆分为多条)，single_track=单轨道(合并)
    const midiTrackMode = params.midiTrackMode ?? "multi_track";
    // MuScriptor 专属：输出乐器(勾选) + 分段衔接方式
    const muscriptorInstruments = params.muscriptorInstruments ?? [];
    // Python contract: only "official" | "telknet" (never the legacy "sustain_connect").
    const muscriptorChain = params.muscriptorChain ?? "official";
    const handleModeChange = useCallback((mode) => updateParams({ midiMode: mode }), [updateParams]);
    const handleBackendChange = useCallback((b) => updateParams({ backend: b }), [updateParams]);
    const handleQuantizeChange = useCallback((g) => updateParams({ quantizeGrid: g }), [updateParams]);
    const handleGpuToggle = useCallback(() => updateParams({ useGpu: !useGpu }), [updateParams, useGpu]);
    const handleTrackModeChange = useCallback((m) => updateParams({ midiTrackMode: m }), [updateParams]);
    const handleChainChange = useCallback((c) => updateParams({ muscriptorChain: c }), [updateParams]);
    const toggleInstrument = useCallback((id) => {
        const next = muscriptorInstruments.includes(id)
            ? muscriptorInstruments.filter((x) => x !== id)
            : [...muscriptorInstruments, id];
        updateParams({ muscriptorInstruments: next });
    }, [updateParams, muscriptorInstruments]);
    const selectAllInstruments = useCallback(() => updateParams({ muscriptorInstruments: [...MUSCRIPTOR_INSTRUMENTS] }), [updateParams]);
    const currentMode = MIDI_MODES.find((m) => m.value === midiMode) ?? MIDI_MODES[0];
    const showBackend = currentMode.multi;
    const isMuscriptor = showBackend && backend === "muscriptor";
    // 「输出乐器」默认折叠 — 节点面板空间有限，展开后才勾选/取消
    const [instrumentsOpen, setInstrumentsOpen] = useState(false);
    return (_jsx(NodeShell, { nodeId: props.id, label: t("amt.nodeLabel"), icon: "\u266A", color: "#a855f7", inputs: 1, outputLabels: [t("amt.midiOutput")], children: _jsxs("div", { className: "amt-node-body", style: {
                padding: 8,
                display: "flex",
                flexDirection: "column",
                gap: 8,
                minWidth: 200,
            }, children: [_jsxs("label", { style: labelStyle, children: [t("amt.modeLabel"), _jsx("select", { value: midiMode, onChange: (e) => handleModeChange(e.target.value), style: selStyle, children: MIDI_MODES.map((m) => (_jsx("option", { value: m.value, children: t(m.labelKey) }, m.value))) })] }), showBackend && (_jsxs("label", { style: labelStyle, children: [t("amt.backendLabel"), _jsx("select", { value: backend, onChange: (e) => handleBackendChange(e.target.value), style: selStyle, children: BACKENDS.map((b) => (_jsx("option", { value: b.value, children: t(b.labelKey) }, b.value))) })] })), _jsxs("label", { style: labelStyle, children: [t("amt.quantizeLabel"), _jsx("select", { value: quantizeGrid, onChange: (e) => handleQuantizeChange(e.target.value), style: selStyle, children: QUANTIZE_GRIDS.map((g) => (_jsx("option", { value: g, children: g === "off" ? t("amt.quantizeOff") : g }, g))) })] }), _jsxs("label", { style: labelStyle, children: [t("amt.trackLayout"), _jsxs("select", { value: midiTrackMode, onChange: (e) => handleTrackModeChange(e.target.value), style: selStyle, children: [_jsx("option", { value: "multi_track", children: t("amt.trackLayoutAll") }), _jsx("option", { value: "single_track", children: t("amt.trackLayoutSingle") })] })] }), isMuscriptor && (_jsxs(_Fragment, { children: [_jsxs("label", { style: labelStyle, children: [t("amt.processingChain"), _jsxs("select", { value: muscriptorChain, onChange: (e) => handleChainChange(e.target.value), style: selStyle, children: [_jsx("option", { value: "official", children: t("amt.chainOfficial") }), _jsx("option", { value: "telknet", children: t("amt.chainTelknet", "Telk-Net 增强 (实验)") })] })] }), _jsxs("div", { style: { ...labelStyle, gap: 4 }, children: [_jsxs("div", { style: { display: "flex", alignItems: "center", justifyContent: "space-between", cursor: "pointer", userSelect: "none" }, onClick: () => setInstrumentsOpen((o) => !o), children: [_jsxs("span", { children: [instrumentsOpen ? "▼" : "▶", " ", t("amt.nodeInstruments"), _jsxs("span", { style: { color: "var(--text-tertiary)", fontSize: 10, marginLeft: 4 }, children: ["(", muscriptorInstruments.length, "/", MUSCRIPTOR_INSTRUMENTS.length, ")"] })] }), _jsx("button", { type: "button", onClick: (e) => { e.stopPropagation(); selectAllInstruments(); }, style: {
                                                fontSize: 10,
                                                padding: "1px 6px",
                                                borderRadius: 4,
                                                border: "1px solid var(--border-subtle)",
                                                background: "transparent",
                                                color: "var(--text-secondary)",
                                                cursor: "pointer",
                                            }, title: "\u5168\u9009", children: t("common.selectAll") })] }), instrumentsOpen && (_jsx("div", { style: { display: "flex", flexWrap: "wrap", gap: 4, maxHeight: 140, overflowY: "auto" }, children: MUSCRIPTOR_INSTRUMENTS.map((id) => (_jsxs("label", { style: {
                                            display: "flex",
                                            alignItems: "center",
                                            gap: 3,
                                            fontSize: 10,
                                            padding: "2px 6px",
                                            borderRadius: 10,
                                            border: "1px solid var(--border-subtle)",
                                            cursor: "pointer",
                                            background: muscriptorInstruments.includes(id)
                                                ? "rgba(168,85,247,0.25)"
                                                : "transparent",
                                        }, children: [_jsx("input", { type: "checkbox", checked: muscriptorInstruments.includes(id), onChange: () => toggleInstrument(id), style: { accentColor: "#a855f7" } }), MUSCRIPTOR_ZH_LABELS[id] ?? id] }, id))) }))] })] })), _jsxs("label", { style: {
                        display: "flex",
                        alignItems: "center",
                        gap: 6,
                        fontSize: 12,
                        color: "var(--text-secondary)",
                        cursor: "pointer",
                    }, children: [_jsx("input", { type: "checkbox", checked: useGpu, onChange: handleGpuToggle }), t("amt.useGpu")] }), _jsxs("div", { style: {
                        fontSize: 11,
                        color: "var(--text-tertiary)",
                        lineHeight: 1.4,
                        marginTop: 4,
                    }, children: [t("amt.modeDesc"), ": ", t(currentMode.labelKey)] })] }) }));
}
