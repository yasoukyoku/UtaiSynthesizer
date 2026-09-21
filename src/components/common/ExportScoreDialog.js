import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useMemo, useRef, useState } from "react";
import { save as saveDialog } from "@tauri-apps/plugin-dialog";
import { useTranslation } from "react-i18next";
import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import { runScoreExport, scoreExportableTracks } from "../../lib/vocal/exportScore";
import { backendErrorMessage } from "../../lib/backendError";
import "./ConfirmDialog.css";
// S63 — Score-export dialog (File → Export Score): format + which vocal tracks. Multi-track ust becomes
// a zip (one .ust per track — the format is single-track); ustx/midi hold all tracks in one file.
// Same style-joining posture as ExportAudioDialog (confirm-* shell, settings-source-opt pills,
// training-check-row checkboxes — the house angular controls, no native widgets invented here).
const FORMATS = ["ustx", "ust", "midi"];
/** EXPORT_SCORE_* lives in THE shared backend mapper (backendError.ts, incl. the ": detail" suffix
 *  convention) — this is just the unknown-message fallback. */
function scoreErrorMessage(e, t) {
    return backendErrorMessage(e) ?? `${t("export.errFailed")}: ${e instanceof Error ? e.message : String(e)}`;
}
export function ExportScoreDialog({ onClose }) {
    const { t } = useTranslation();
    const tracks = useProjectStore((s) => s.tracks);
    const projectName = useProjectStore((s) => s.name);
    const choices = useMemo(() => scoreExportableTracks(tracks), [tracks]);
    const [format, setFormat] = useState("ustx");
    const [picked, setPicked] = useState(() => new Set(choices.map((c) => c.trackId)));
    const [busy, setBusy] = useState(false);
    // Busy guard on every dismiss path (audit): closing mid-invoke would unmount with the Rust write
    // still running — the completion toast would fire with no dialog and a re-open could double-run.
    const busyRef = useRef(busy);
    busyRef.current = busy;
    useEffect(() => {
        const onKey = (e) => {
            e.stopPropagation();
            if (e.key === "Escape" && !busyRef.current)
                onClose();
        };
        window.addEventListener("keydown", onKey, true);
        return () => window.removeEventListener("keydown", onKey, true);
    }, [onClose]);
    const toggle = (id) => setPicked((prev) => {
        const next = new Set(prev);
        if (next.has(id))
            next.delete(id);
        else
            next.add(id);
        return next;
    });
    const chosen = choices.filter((c) => picked.has(c.trackId));
    const handleExport = async () => {
        if (busy || chosen.length === 0)
            return;
        // ust is single-track-per-file: several tracks land in one zip at the chosen location.
        const ext = format === "midi" ? "mid" : format === "ust" && chosen.length > 1 ? "zip" : format;
        const base = (projectName || "export").replace(/[<>:"/\\|?*]/g, "_");
        const outPath = await saveDialog({
            title: t("export.scoreTitle"),
            defaultPath: `${base}.${ext}`,
            filters: [{ name: ext.toUpperCase(), extensions: [ext] }],
        });
        if (!outPath || typeof outPath !== "string")
            return;
        setBusy(true);
        try {
            await runScoreExport(format, outPath, chosen.map((c) => c.trackId));
            const name = outPath.replace(/^.*[\\/]/, "");
            useAppStore.getState().showToast(`${t("export.done")} · ${name}`, "success");
            onClose();
        }
        catch (e) {
            useAppStore.getState().showToast(scoreErrorMessage(e, t), "error");
        }
        finally {
            setBusy(false);
        }
    };
    return (_jsx("div", { className: "confirm-overlay", onMouseDown: busy ? undefined : onClose, children: _jsxs("div", { className: "confirm-dialog", role: "dialog", "aria-modal": "true", onMouseDown: (e) => e.stopPropagation(), children: [_jsx("div", { className: "confirm-title", children: t("export.scoreTitle") }), _jsxs("div", { style: { marginBottom: 10 }, children: [_jsx("div", { style: { fontSize: "var(--font-size-sm)", color: "var(--text-secondary)", marginBottom: 4 }, children: t("export.format") }), _jsx("div", { style: { display: "flex", flexWrap: "wrap", gap: 4 }, children: FORMATS.map((f) => (_jsxs("label", { className: `settings-source-opt ${format === f ? "active" : ""}`, children: [_jsx("input", { type: "radio", name: "exp-score-fmt", checked: format === f, disabled: busy, onChange: () => setFormat(f) }), _jsx("span", { children: f === "midi" ? "MIDI" : `.${f}` })] }, f))) }), format === "ust" && chosen.length > 1 && (_jsx("div", { style: { fontSize: "var(--font-size-sm)", color: "var(--text-secondary)", marginTop: 4 }, children: t("export.ustZipNote") }))] }), _jsxs("div", { style: { marginBottom: 14 }, children: [_jsx("div", { style: { fontSize: "var(--font-size-sm)", color: "var(--text-secondary)", marginBottom: 4 }, children: t("export.tracks") }), _jsx("div", { style: { maxHeight: 200, overflowY: "auto" }, children: choices.map((c) => (_jsxs("label", { className: "training-check-row", style: { display: "flex", alignItems: "center", gap: 8, padding: "3px 0" }, children: [_jsx("input", { type: "checkbox", checked: picked.has(c.trackId), disabled: busy, onChange: () => toggle(c.trackId) }), _jsx("span", { style: { color: "var(--text-primary)" }, children: c.name }), _jsxs("span", { style: { color: "var(--text-secondary)", fontSize: "var(--font-size-sm)" }, children: ["(", c.noteCount, ")"] })] }, c.trackId))) })] }), _jsxs("div", { className: "confirm-buttons", children: [_jsx("button", { className: "confirm-btn neutral", disabled: busy, onClick: onClose, children: t("common.cancel") }), _jsx("button", { className: "confirm-btn primary", disabled: busy || chosen.length === 0, onClick: () => void handleExport(), children: t("export.start") })] })] }) }));
}
