import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useCallback, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { useSoundfontStore } from "../../../store/soundfont";
const SAMPLE_RATES = [44100, 48000];
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
export function SoundfontRenderNode(props) {
    const { t } = useTranslation();
    const [params, updateParams] = useNodeParams(props);
    const fonts = useSoundfontStore((s) => s.fonts);
    const refresh = useSoundfontStore((s) => s.refresh);
    useEffect(() => {
        if (fonts.length === 0)
            void refresh();
    }, [fonts.length, refresh]);
    const fontId = params.fontId ?? "";
    const presetId = params.presetId ?? "";
    const backend = params.backend ?? "builtin";
    const sampleRate = params.sampleRate ?? 44100;
    const font = fonts.find((f) => f.id === fontId);
    const presets = font?.presets ?? [];
    const handleFontChange = useCallback((id) => {
        const next = fonts.find((f) => f.id === id);
        updateParams({ fontId: id, presetId: next?.presets[0]?.id ?? "" });
    }, [fonts, updateParams]);
    const handlePresetChange = useCallback((id) => updateParams({ presetId: id }), [updateParams]);
    const handleBackendChange = useCallback((b) => updateParams({ backend: b }), [updateParams]);
    const handleSampleRateChange = useCallback((sr) => updateParams({ sampleRate: Number(sr) }), [updateParams]);
    return (_jsx(NodeShell, { nodeId: props.id, label: t("workflow.nodeSoundfontRender"), icon: "\uD83C\uDFBB", color: "#2dd4bf", inputs: 1, outputLabels: [t("workflow.sfRenderAudioOut")], children: _jsxs("div", { style: {
                padding: 8,
                display: "flex",
                flexDirection: "column",
                gap: 8,
                minWidth: 200,
            }, children: [fonts.length === 0 ? (_jsx("div", { style: { fontSize: 11, color: "var(--text-tertiary)", lineHeight: 1.4 }, children: t("workflow.sfRenderNoFont") })) : (_jsxs(_Fragment, { children: [_jsxs("label", { style: labelStyle, children: [t("workflow.sfRenderFont"), _jsxs("select", { value: fontId, onChange: (e) => handleFontChange(e.target.value), style: selStyle, children: [_jsx("option", { value: "", children: t("workflow.sfRenderPickFont") }), fonts.map((f) => (_jsxs("option", { value: f.id, children: [f.name, " (", f.format.toUpperCase(), ")"] }, f.id)))] })] }), _jsxs("label", { style: labelStyle, children: [t("workflow.sfRenderPreset"), _jsxs("select", { value: presetId, onChange: (e) => handlePresetChange(e.target.value), style: selStyle, disabled: presets.length === 0, children: [_jsx("option", { value: "", children: t("workflow.sfRenderPickPreset") }), presets.map((p) => (_jsx("option", { value: p.id, children: p.name }, p.id)))] })] })] })), _jsxs("label", { style: labelStyle, children: [t("workflow.sfRenderBackend"), _jsxs("select", { value: backend, onChange: (e) => handleBackendChange(e.target.value), style: selStyle, children: [_jsx("option", { value: "builtin", children: t("workflow.sfRenderBackendBuiltin") }), _jsx("option", { value: "fluidsynth", children: t("workflow.sfRenderBackendFluidsynth") })] })] }), _jsxs("label", { style: labelStyle, children: [t("workflow.sfRenderSampleRate"), _jsx("select", { value: String(sampleRate), onChange: (e) => handleSampleRateChange(e.target.value), style: selStyle, children: SAMPLE_RATES.map((sr) => (_jsxs("option", { value: sr, children: [sr, " Hz"] }, sr))) })] }), _jsx("div", { style: { fontSize: 11, color: "var(--text-tertiary)", lineHeight: 1.4 }, children: t("workflow.sfRenderDesc") })] }) }));
}
