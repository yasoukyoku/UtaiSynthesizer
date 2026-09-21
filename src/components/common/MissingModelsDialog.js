import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect } from "react";
import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useTranslation } from "react-i18next";
import { useAppStore } from "../../store/app";
import { useMsstModelStore } from "../../store/msst-models";
import { hfBaseForMirror } from "../../lib/models/msst-catalog";
import { backendErrorMessage } from "../../lib/backendError";
import "./ConfirmDialog.css";
import "./MissingModelsDialog.css";
/**
 * S66 pre-run model dialog — the "don't make users guess" surface. The workflow/vocal
 * preflight (engine.collectMissingModels / preflightVocalModels) aborts the run and opens
 * this instead of letting MSST_MODEL_NOT_CONVERTED / AUX_FILE_MISSING explode mid-run.
 * Each row carries its own one-click action:
 *   - unconverted MSST model → convert (serial app-wide via the Rust convert slot)
 *   - model file not installed → open the resource manager
 *   - core asset pack missing → start the download (progress lives in Settings → Model Assets)
 * The user re-runs after the fixes — the dialog deliberately does NOT auto-restart the run.
 */
export function MissingModelsDialog() {
    const items = useAppStore((s) => s.missingModels);
    const close = useAppStore((s) => s.closeMissingModels);
    const { t } = useTranslation();
    const installed = useMsstModelStore((s) => s.installed);
    const downloading = useMsstModelStore((s) => s.downloading);
    const error = useMsstModelStore((s) => s.error);
    const convertPrecision = useMsstModelStore((s) => s.convertPrecision);
    const fetchInstalled = useMsstModelStore((s) => s.fetchInstalled);
    const mirror = useMsstModelStore((s) => s.mirror);
    const [auxStarted, setAuxStarted] = useState(false);
    // Fresh conversion state for the rows (the store may never have fetched this session).
    useEffect(() => {
        if (items)
            void fetchInstalled();
    }, [items, fetchInstalled]);
    useEffect(() => {
        if (!items)
            return;
        const onKey = (e) => {
            e.stopPropagation();
            if (e.key === "Escape") {
                e.preventDefault();
                close();
            }
        };
        window.addEventListener("keydown", onKey, true);
        return () => window.removeEventListener("keydown", onKey, true);
    }, [items, close]);
    if (!items)
        return null;
    const anyConverting = Object.values(downloading).some((d) => d.stage === "converting");
    return (_jsx("div", { className: "confirm-overlay", onMouseDown: close, children: _jsxs("div", { className: "confirm-dialog mm-dialog", role: "dialog", "aria-modal": "true", onMouseDown: (e) => e.stopPropagation(), children: [_jsx("div", { className: "confirm-title", children: t("missingModels.title") }), _jsx("div", { className: "confirm-body", children: t("missingModels.body") }), _jsx("div", { className: "mm-list", children: items.map((it, i) => {
                        if (it.kind === "msstConvert") {
                            const entry = installed.find((m) => m.filename === it.filename);
                            const done = !!entry && (entry.has_onnx || entry.has_fp16);
                            const converting = it.filename ? downloading[it.filename]?.stage === "converting" : false;
                            return (_jsxs("div", { className: "mm-row", children: [_jsx("span", { className: "mm-label", title: it.label, children: it.label }), done ? (_jsx("span", { className: "mm-status mm-done", children: t("missingModels.converted") })) : converting ? (_jsx("span", { className: "mm-status", children: t("missingModels.converting") })) : (_jsx("button", { className: "mm-btn", disabled: anyConverting, onClick: () => {
                                            if (it.filename)
                                                void convertPrecision(it.filename, it.precision, it.architecture);
                                        }, children: t("missingModels.convert") }))] }, i));
                        }
                        if (it.kind === "msstMissing") {
                            return (_jsxs("div", { className: "mm-row", children: [_jsx("span", { className: "mm-label", title: it.label, children: it.label }), _jsx("span", { className: "mm-status", children: t("missingModels.notInstalled") }), _jsx("button", { className: "mm-btn", onClick: () => {
                                            if (!useAppStore.getState().modelManagerOpen)
                                                useAppStore.getState().toggleModelManager();
                                        }, children: t("missingModels.openManager") })] }, i));
                        }
                        // auxPack — label 即 pack id(aux-inference / aux-autotune,S73 泛化;
                        // 显示文案按 pack 选,下载走同一 download_asset_pack 漏斗)
                        const packId = it.label || "aux-inference";
                        return (_jsxs("div", { className: "mm-row", children: [_jsx("span", { className: "mm-label", children: packId === "aux-autotune" ? t("missingModels.autotunePack") : t("startup.compAux") }), auxStarted ? (_jsx("span", { className: "mm-status", children: t("missingModels.downloadStarted") })) : (_jsx("button", { className: "mm-btn", onClick: () => {
                                        setAuxStarted(true);
                                        void invoke("download_asset_pack", {
                                            id: packId,
                                            hfBase: hfBaseForMirror(mirror),
                                        }).catch(() => {
                                            /* busy/cancel/fail surface in Settings → Model Assets */
                                        });
                                    }, children: t("missingModels.download") }))] }, i));
                    }) }), error && _jsx("div", { className: "mm-error", children: backendErrorMessage(error) ?? error }), _jsx("div", { className: "confirm-body mm-hint", children: t("missingModels.hint") }), _jsx("div", { className: "confirm-buttons", children: _jsx("button", { className: "confirm-btn neutral", onClick: close, children: t("missingModels.close") }) })] }) }));
}
