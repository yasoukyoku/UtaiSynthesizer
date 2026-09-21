import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useRef, useState } from "react";
import { save as saveDialog } from "@tauri-apps/plugin-dialog";
import { invoke } from "@tauri-apps/api/core";
import { useTranslation } from "react-i18next";
import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import { useAudioStore } from "../../store/audio";
import { useWorkflowStore } from "../../store/workflow";
import { useVoiceModelStore } from "../../store/voice-models";
import { runAudioExport } from "../../lib/audio/exportAudio";
import { backendErrorMessage, isBusyError } from "../../lib/backendError";
import "./ConfirmDialog.css";
// S63 — Audio-export dialog (File → Export Audio). Styling deliberately JOINS existing selector groups
// instead of forking new ones (NO-duplication): the modal shell reuses ConfirmDialog's confirm-* frame,
// the option pills reuse Settings' settings-source-opt radio-label group. The mixdown itself lives in
// lib/audio/exportAudio.ts (render) + Rust export_audio.rs (encode) — this file is pure form + phases.
const FORMATS = ["wav", "flac", "mp3", "ogg", "opus", "m4a"];
const RATES = [44100, 48000];
const WAV_DEPTHS = ["16", "24", "32f"];
const FLAC_DEPTHS = ["16", "24"];
const BITRATES = [128, 160, 192, 256, 320];
const isLossy = (f) => f === "mp3" || f === "ogg" || f === "opus" || f === "m4a";
/** Map the export pipeline's stable codes (frontend renderMixdown EXPORT_* + Rust encode codes) to a
 *  localized line — payload-first is irrelevant here (no user-content payloads), so plain substring
 *  checks then delegation to THE shared backend mapper (the vocalRenderErrorMessage pattern). */
function exportErrorMessage(e, t) {
    const msg = e instanceof Error ? e.message : String(e);
    const local = {
        EXPORT_VOCALS_FAILED: "export.errVocals",
        EXPORT_INSTRUMENTS_FAILED: "export.errInstruments",
        EXPORT_VOCALS_UNRENDERED: "export.errUnrendered",
        EXPORT_SOURCE_LOADING: "export.errLoading",
        EXPORT_SOURCE_MISSING: "export.errMissing",
        EXPORT_DECODE_FAIL: "export.errDecode",
        EXPORT_TOO_LONG: "export.errTooLong",
        EXPORT_EMPTY: "export.errEmpty",
    };
    for (const code of Object.keys(local)) {
        if (msg.includes(code))
            return t(local[code]);
    }
    return backendErrorMessage(e) ?? `${t("export.errFailed")}: ${msg}`;
}
export function ExportAudioDialog({ onClose }) {
    const { t } = useTranslation();
    const projectName = useProjectStore((s) => s.name);
    const [format, setFormat] = useState("wav");
    const [rate, setRate] = useState(44100);
    const [depth, setDepth] = useState("16");
    const [bitrate, setBitrate] = useState(320);
    /** §user 母带处理：EQ 打磨 + 温和压缩 + -1dBFS 前瞻限制，导出即发行级响度。 */
    const [mastering, setMastering] = useState(true);
    const [phase, setPhase] = useState(null);
    const [busy, setBusy] = useState(false);
    const abortRef = useRef(false);
    // Same live gates as Settings' cacheCleanBlocked family — an export must not overlap a job that is
    // mid-write into the render cache / holds the voice guard (the dirty-vocal pre-render needs it free).
    const isPlaying = useAudioStore((s) => s.isPlaying);
    const vocalRenderActive = useAppStore((s) => s.vocalRenderActive);
    const anyWorkflowRunning = useWorkflowStore((s) => Object.values(s.executions).some((e) => e.status === "running"));
    const midiExtracting = useAppStore((s) => Object.keys(s.midiExtracting).length > 0);
    const rangeTesting = useVoiceModelStore((s) => Object.keys(s.rangeTesting).length > 0);
    const decoding = useAudioStore((s) => s.loadingPaths.length > 0);
    const blocked = isPlaying || vocalRenderActive || anyWorkflowRunning || midiExtracting || rangeTesting || decoding;
    // Own the keyboard while open (the ConfirmDialog capture pattern) — Esc closes unless mid-export.
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
    const depths = format === "wav" ? WAV_DEPTHS : format === "flac" ? FLAC_DEPTHS : [];
    const pickFormat = (f) => {
        setFormat(f);
        // Keep the depth valid for the new format (flac has no 32f).
        if (f === "flac" && depth === "32f")
            setDepth("24");
    };
    const handleExport = async () => {
        if (busy || blocked)
            return;
        const base = (projectName || "export").replace(/[<>:"/\\|?*]/g, "_");
        const outPath = await saveDialog({
            title: t("export.audioTitle"),
            defaultPath: `${base}.${format}`,
            filters: [{ name: format.toUpperCase(), extensions: [format] }],
        });
        if (!outPath || typeof outPath !== "string")
            return;
        setBusy(true);
        abortRef.current = false;
        try {
            const params = {
                outPath,
                format,
                sampleRate: rate,
                bitDepth: depth,
                mastering,
                bitrateKbps: bitrate,
            };
            const res = await runAudioExport(params, setPhase, () => abortRef.current);
            if (res.cancelled) {
                useAppStore.getState().showToast(t("export.cancelled"), "info");
                return;
            }
            const name = outPath.replace(/^.*[\\/]/, "");
            useAppStore.getState().showToast(`${t("export.done")} · ${name}`, "success");
            if (res.peak > 1)
                useAppStore.getState().showToast(t("export.clipWarn"), "info");
            onClose();
        }
        catch (e) {
            useAppStore.getState().showToast(exportErrorMessage(e, t), isBusyError(e) ? "info" : "error");
        }
        finally {
            setBusy(false);
            setPhase(null);
        }
    };
    const handleCancel = () => {
        if (!busy)
            return onClose();
        abortRef.current = true;
        // cancel_voice ONLY during the vocal pre-render phase (the Toolbar pattern — aborts the in-flight
        // GPU render). Past that phase the latch is not ours to pull: a manual sidebar render started while
        // we're in mix/encode would be killed by a stray global cancel (the flag alone ends us between stages).
        if (phase?.kind === "vocals")
            void invoke("cancel_voice").catch(() => { });
    };
    const phaseLine = !phase
        ? null
        : phase.kind === "vocals"
            ? `${t("export.phaseVocals")} (${phase.total})`
            : phase.kind === "instruments"
                ? `${t("export.phaseInstruments")} (${phase.total})`
                : phase.kind === "mix"
                    ? t("export.phaseMix")
                    : t("export.phaseEncode");
    const mixFrac = phase?.kind === "mix" ? phase.frac : phase?.kind === "encode" ? 1 : 0;
    const optRow = (label, children) => (_jsxs("div", { style: { marginBottom: 10 }, children: [_jsx("div", { style: { fontSize: "var(--font-size-sm)", color: "var(--text-secondary)", marginBottom: 4 }, children: label }), _jsx("div", { style: { display: "flex", flexWrap: "wrap", gap: 4 }, children: children })] }));
    const pill = (name, active, label, onPick) => (_jsxs("label", { className: `settings-source-opt ${active ? "active" : ""}`, children: [_jsx("input", { type: "radio", name: name, checked: active, disabled: busy, onChange: onPick }), _jsx("span", { children: label })] }, label));
    return (_jsx("div", { className: "confirm-overlay", onMouseDown: busy ? undefined : onClose, children: _jsxs("div", { className: "confirm-dialog", role: "dialog", "aria-modal": "true", onMouseDown: (e) => e.stopPropagation(), children: [_jsx("div", { className: "confirm-title", children: t("export.audioTitle") }), optRow(t("export.format"), FORMATS.map((f) => pill("exp-fmt", format === f, f.toUpperCase(), () => pickFormat(f)))), optRow(t("export.sampleRate"), RATES.map((r) => pill("exp-rate", rate === r, `${r / 1000} kHz`, () => setRate(r)))), depths.length > 0 &&
                    optRow(t("export.bitDepth"), depths.map((d) => pill("exp-depth", depth === d, d === "32f" ? "32-bit float" : `${d}-bit`, () => setDepth(d)))), isLossy(format) &&
                    optRow(t("export.bitrate"), BITRATES.map((b) => pill("exp-rate-k", bitrate === b, `${b} kbps`, () => setBitrate(b)))), _jsxs("label", { className: `settings-source-opt ${mastering ? "active" : ""}`, style: { display: "inline-flex", marginBottom: 12, cursor: "pointer" }, title: t("export.masteringTip") || "三段 EQ 打磨 + 温和压缩 + 前瞻限制器（-1 dBFS 天花板）：导出即发行级响度，不再需要外部母带软件", children: [_jsx("input", { type: "checkbox", checked: mastering, disabled: busy, onChange: (e) => setMastering(e.target.checked), style: { marginRight: 6 } }), _jsxs("span", { children: ["\uD83C\uDF9A ", t("export.mastering") || "母带处理（EQ + 压缩 + 限制器）"] })] }), blocked && !busy && (_jsx("div", { className: "confirm-body", style: { marginBottom: 12, color: "var(--accent-tertiary)" }, children: isPlaying ? t("export.blockedPlaying") : t("common.busyRetry") })), busy && (_jsxs("div", { style: { margin: "6px 0 12px" }, children: [_jsx("div", { style: { fontSize: "var(--font-size-sm)", color: "var(--text-secondary)", marginBottom: 5 }, children: phaseLine }), _jsx("div", { style: { height: 4, background: "var(--bg-base)", border: "1px solid var(--border-strong)" }, children: _jsx("div", { style: {
                                    height: "100%",
                                    width: `${Math.round(mixFrac * 100)}%`,
                                    background: "var(--accent-primary)",
                                    transition: "width 0.12s linear",
                                } }) })] })), _jsxs("div", { className: "confirm-buttons", children: [_jsx("button", { className: "confirm-btn neutral", disabled: busy && phase?.kind === "encode", onClick: handleCancel, children: t("common.cancel") }), _jsx("button", { className: "confirm-btn primary", disabled: busy || blocked, onClick: () => void handleExport(), children: t("export.start") })] })] }) }));
}
