import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useTranslation } from "react-i18next";
import { useAmtStore } from "../../../store/amt";
import { useAppStore } from "../../../store/app";
import { t18 } from "../../../lib/models/msst-catalog";
import "../../common/ConfirmDialog.css";
const QUANTIZE_GRIDS = ["off", "1/4", "1/8", "1/16", "1/32", "1/64"];
export function MidiWorkbench({ nodeId, onClose }) {
    const { t, i18n } = useTranslation();
    const lang = i18n.language;
    const run = useAmtStore((s) => s.runs[nodeId]);
    const setRun = useAmtStore((s) => s.setRun);
    const [meta, setMeta] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [quantizeGrid, setQuantizeGrid] = useState("off");
    const [customBpm, setCustomBpm] = useState("");
    const [rerunning, setRerunning] = useState(false);
    const [reProgress, setReProgress] = useState(null);
    const midiPath = run?.midiPath ?? "";
    const loadMeta = useCallback(async () => {
        if (!midiPath)
            return;
        setLoading(true);
        setError(null);
        try {
            const m = await invoke("amt_midi_metadata", { midiPath });
            setMeta(m);
        }
        catch (e) {
            setError(String(e));
        }
        finally {
            setLoading(false);
        }
    }, [midiPath]);
    useEffect(() => {
        void loadMeta();
    }, [loadMeta, midiPath]);
    const rerun = useCallback(async (opts) => {
        if (!run)
            return;
        setRerunning(true);
        setReProgress(0);
        setError(null);
        const outDir = midiPath.substring(0, midiPath.lastIndexOf("\\")) ||
            midiPath.substring(0, midiPath.lastIndexOf("/")) || ".";
        try {
            const res = await invoke("run_amt_midi", {
                audioPath: run.audioPath,
                midiMode: run.mode,
                transcriptionBackend: ["smart", "vocal_split", "six_stem_split"].includes(run.mode)
                    ? run.backend
                    : null,
                yourmt3Model: null,
                muscriptorModel: null,
                midiTrackMode: null,
                tempoMode: opts.customBpm != null ? "fixed_manual" : "fixed_auto",
                customBpm: opts.customBpm,
                quantizeNotes: opts.quantizeGrid !== "off",
                quantizeGrid: opts.quantizeGrid === "off" ? null : opts.quantizeGrid,
                useGpu: true,
                gpuDevice: 0,
                outputDir: outDir,
                nodeId,
            });
            setRun({
                ...run,
                midiPath: res.midi_path,
                totalNotes: res.total_notes,
                processingTimeSecs: res.processing_time_secs,
            });
            setReProgress(1);
        }
        catch (e) {
            setError(String(e));
        }
        finally {
            setRerunning(false);
            setTimeout(() => setReProgress(null), 800);
        }
    }, [run, midiPath, nodeId, setRun]);
    // Export EVERY track of the merged MIDI as its own .mid into a user-chosen folder.
    // Fallback: if the merged file has a single track, one file is still written into that folder.
    const handleExportFolder = useCallback(async () => {
        if (!midiPath)
            return;
        const { open } = await import("@tauri-apps/plugin-dialog");
        let dir = await open({ directory: true, title: t("amt.exportFolderTitle") });
        if (!dir)
            return;
        // multi-selection off → returns a single string; normalize an array just in case.
        if (Array.isArray(dir))
            dir = dir[0];
        if (typeof dir !== "string")
            return;
        try {
            const written = await invoke("export_midi_tracks_to_folder", {
                midiPath,
                folder: dir,
            });
            useAppStore
                .getState()
                .showToast(t18({ zh: `已导出 ${written.length} 个轨道`, en: `Exported ${written.length} track(s)`, ja: `${written.length} トラックをエクスポート` }, lang), "success");
        }
        catch (e) {
            useAppStore.getState().showToast(String(e), "error");
        }
    }, [midiPath, lang, t]);
    const fmtSize = (b) => {
        if (b < 1024)
            return `${b} B`;
        if (b < 1024 * 1024)
            return `${(b / 1024).toFixed(1)} KB`;
        return `${(b / 1024 / 1024).toFixed(2)} MB`;
    };
    return (_jsx("div", { className: "confirm-overlay", onMouseDown: onClose, children: _jsxs("div", { className: "confirm-dialog", role: "dialog", "aria-modal": "true", onMouseDown: (e) => e.stopPropagation(), style: { maxWidth: 560 }, children: [_jsx("div", { className: "confirm-title", children: t("amt.workbenchTitle") }), loading && _jsx("div", { className: "confirm-body", children: t("amt.loadingMeta") }), error && (_jsx("div", { className: "confirm-body", style: { color: "var(--danger)" }, children: error })), meta && (_jsxs("div", { className: "confirm-body", style: { display: "flex", flexDirection: "column", gap: 10 }, children: [_jsxs("div", { style: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, fontSize: 13 }, children: [_jsxs("div", { children: [_jsxs("span", { style: { color: "var(--text-tertiary)" }, children: [t("amt.totalNotes"), ": "] }), _jsx("b", { children: meta.total_notes })] }), _jsxs("div", { children: [_jsxs("span", { style: { color: "var(--text-tertiary)" }, children: [t("amt.tracks"), ": "] }), _jsx("b", { children: meta.track_count })] }), _jsxs("div", { children: [_jsxs("span", { style: { color: "var(--text-tertiary)" }, children: [t("amt.bpm"), ": "] }), _jsx("b", { children: meta.bpm != null ? meta.bpm.toFixed(1) : "—" })] }), _jsxs("div", { children: [_jsxs("span", { style: { color: "var(--text-tertiary)" }, children: [t("amt.timeSig"), ": "] }), _jsx("b", { children: meta.time_signature ? `${meta.time_signature[0]}/${meta.time_signature[1]}` : "—" })] }), _jsxs("div", { children: [_jsx("span", { style: { color: "var(--text-tertiary)" }, children: "PPQ: " }), _jsx("b", { children: meta.ppq })] }), _jsxs("div", { children: [_jsxs("span", { style: { color: "var(--text-tertiary)" }, children: [t("amt.fileSize"), ": "] }), _jsx("b", { children: fmtSize(meta.size_bytes) })] })] }), _jsx("div", { style: { maxHeight: 160, overflowY: "auto", border: "1px solid var(--border-subtle)", borderRadius: 6, padding: 6 }, children: meta.tracks.map((tr, i) => (_jsxs("div", { style: { display: "flex", justifyContent: "space-between", fontSize: 12, padding: "2px 4px" }, children: [_jsx("span", { style: { color: "var(--text-secondary)" }, children: tr.name }), _jsxs("span", { children: [tr.note_count, " ", t("amt.notes")] })] }, i))) }), _jsxs("div", { style: { borderTop: "1px solid var(--border-subtle)", paddingTop: 10 }, children: [_jsx("div", { style: { fontSize: 13, fontWeight: 600, marginBottom: 6 }, children: t("amt.requantize") }), _jsxs("div", { style: { display: "flex", gap: 8, alignItems: "center" }, children: [_jsx("select", { value: quantizeGrid, onChange: (e) => setQuantizeGrid(e.target.value), style: { flex: 1, padding: "6px 8px", borderRadius: 6, border: "1px solid var(--border-subtle)", background: "var(--bg-surface)", color: "var(--text-primary)" }, children: QUANTIZE_GRIDS.map((g) => (_jsx("option", { value: g, children: g === "off" ? t("amt.quantizeOff") : g }, g))) }), _jsx("button", { className: "confirm-btn neutral", disabled: rerunning, onClick: () => void rerun({ quantizeGrid }), children: t("amt.apply") })] })] }), _jsxs("div", { children: [_jsx("div", { style: { fontSize: 13, fontWeight: 600, marginBottom: 6 }, children: t("amt.retempo") }), _jsxs("div", { style: { display: "flex", gap: 8, alignItems: "center" }, children: [_jsx("input", { type: "number", min: 20, max: 300, step: 0.1, placeholder: t("amt.bpmPlaceholder"), value: customBpm, onChange: (e) => setCustomBpm(e.target.value), style: { flex: 1, padding: "6px 8px", borderRadius: 6, border: "1px solid var(--border-subtle)", background: "var(--bg-surface)", color: "var(--text-primary)" } }), _jsx("button", { className: "confirm-btn neutral", disabled: rerunning || !customBpm, onClick: () => void rerun({ customBpm: customBpm ? parseFloat(customBpm) : null }), children: t("amt.apply") }), _jsx("button", { className: "confirm-btn neutral", disabled: rerunning, onClick: () => void rerun({ customBpm: null }), title: t("amt.resetTempo"), children: t("amt.auto") })] })] }), reProgress != null && (_jsx("div", { style: { fontSize: 12, color: "var(--accent)" }, children: rerunning ? t("amt.processing") : t("amt.done") }))] })), _jsxs("div", { className: "confirm-buttons", children: [_jsx("button", { className: "confirm-btn neutral", onClick: onClose, children: t18({ zh: "关闭", en: "Close", ja: "閉じる" }, lang) }), _jsx("button", { className: "confirm-btn primary", onClick: () => void handleExportFolder(), disabled: !midiPath, children: t("amt.exportAll") })] })] }) }));
}
