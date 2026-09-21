import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { open as openUrl } from "@tauri-apps/plugin-shell";
import { useAppStore } from "../../store/app";
import { useSoundfontStore, RECOMMENDED_SOUNDFONTS } from "../../store/soundfont";
import { useAmtModelStore, setupAmtDownloadListener } from "../../store/amt-models";
import { AMT_CATALOG } from "../../lib/models/amt-catalog";
import { t18 } from "../../lib/models/msst-catalog";
import { useFloatingPanel } from "../../lib/useFloatingPanel";
import { PanelResizeHandles } from "../common/PanelResizeHandles";
import { backendErrorMessage } from "../../lib/backendError";
import "./SoundfontManager.css";
function formatSize(bytes) {
    if (!bytes || bytes <= 0)
        return "—";
    if (bytes < 1024 * 1024)
        return `${Math.max(1, Math.round(bytes / 1024))} KB`;
    if (bytes < 1024 * 1024 * 1024)
        return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}
/** 可一键下载并自动安装的通用音源（来自 AMT 目录，architecture = soundfont）。 */
const DOWNLOADABLE_SOUNDFONTS = AMT_CATALOG.filter((e) => e.architecture === "soundfont");
export function SoundfontManager() {
    const { t, i18n } = useTranslation();
    const lang = i18n.language;
    const close = useCallback(() => useAppStore.getState().toggleSoundfontManager(), []);
    const { fonts, scanning, importingPaths, refresh, importFont, removeFont } = useSoundfontStore();
    const { installed: amtInstalled, downloading: amtProgress, fetchInstalled: fetchAmtInstalled, downloadEntry: downloadSoundfont, } = useAmtModelStore();
    // 稳定 action 从 getState 取,避免整 store 订阅(toast/滚动等任意变化都会重渲染本面板)。
    const { showToast, showConfirm } = useAppStore.getState();
    const [tab, setTab] = useState("installed");
    const [confirmDelete, setConfirmDelete] = useState(null);
    const { style, startDrag, startResize } = useFloatingPanel({
        storageKey: "muno.soundfontManagerRect",
        initial: () => ({ x: 160, y: 120, w: 620, h: Math.round(window.innerHeight * 0.75) }),
        minW: 520,
        minH: 360,
    });
    // 打开即扫一次 + 拉取可下载音源的安装状态
    useEffect(() => {
        void refresh();
        void fetchAmtInstalled();
        void setupAmtDownloadListener();
    }, [refresh, fetchAmtInstalled]);
    const handleImport = async () => {
        try {
            const { open } = await import("@tauri-apps/plugin-dialog");
            const picked = await open({
                multiple: true,
                filters: [
                    { name: "SoundFont", extensions: ["sf2", "sf3"] },
                    { name: "SFZ", extensions: ["sfz"] },
                ],
                title: t("soundfont.importDialogTitle"),
            });
            if (!picked)
                return;
            const paths = Array.isArray(picked) ? picked : [picked];
            for (const p of paths) {
                try {
                    const font = await importFont(p);
                    if (font)
                        showToast(t("soundfont.imported", { name: font.name }), "success");
                }
                catch (e) {
                    showToast(backendErrorMessage(e) ?? String(e), "error");
                }
            }
        }
        catch (e) {
            showToast(backendErrorMessage(e) ?? String(e), "error");
        }
    };
    const handleImportDir = async () => {
        try {
            const { open } = await import("@tauri-apps/plugin-dialog");
            const dir = await open({ directory: true, multiple: false, title: t("soundfont.importDirTitle") });
            if (!dir || Array.isArray(dir))
                return;
            try {
                const font = await importFont(dir);
                if (font)
                    showToast(t("soundfont.imported", { name: font.name }), "success");
            }
            catch (e) {
                showToast(backendErrorMessage(e) ?? String(e), "error");
            }
        }
        catch (e) {
            showToast(backendErrorMessage(e) ?? String(e), "error");
        }
    };
    const handleDelete = async (font) => {
        setConfirmDelete(font.id);
        const choice = await showConfirm({
            title: t("soundfont.deleteTitle"),
            body: t("soundfont.deleteBody", { name: font.name }),
            buttons: [
                { id: "cancel", label: t("common.cancel") },
                { id: "delete", label: t("soundfont.deleteConfirm"), kind: "danger" },
            ],
        });
        setConfirmDelete(null);
        if (choice !== "delete")
            return;
        const ok = await removeFont(font.id);
        if (ok)
            showToast(t("soundfont.deleted", { name: font.name }), "success");
        else
            showToast(t("soundfont.deleteFailed"), "error");
    };
    const handleOpenFolder = async () => {
        try {
            const { invoke } = await import("@tauri-apps/api/core");
            await invoke("open_soundfonts_dir");
        }
        catch (e) {
            showToast(backendErrorMessage(e) ?? String(e), "error");
        }
    };
    const handleOpenLink = async (url) => {
        try {
            await openUrl(url);
        }
        catch {
            /* openUrl 权限失败时静默 */
        }
    };
    const handleRescan = () => void refresh();
    return (_jsxs("aside", { className: "soundfont-manager", style: style, children: [_jsxs("div", { className: "panel-header", onMouseDown: startDrag, children: [_jsx("span", { className: "panel-title", children: t("soundfont.title") }), _jsx("button", { className: "panel-close", onClick: close, children: "X" })] }), _jsxs("div", { className: "sfm-tabs", children: [_jsxs("button", { className: `sfm-tab ${tab === "installed" ? "active" : ""}`, onClick: () => setTab("installed"), children: [t("soundfont.tabInstalled"), fonts.length > 0 && _jsx("span", { className: "sfm-tab-count", children: fonts.length })] }), _jsxs("button", { className: `sfm-tab ${tab === "download" ? "active" : ""}`, onClick: () => setTab("download"), children: [t("soundfont.tabDownload"), _jsx("span", { className: "sfm-tab-count sfm-tab-primary", children: DOWNLOADABLE_SOUNDFONTS.length + RECOMMENDED_SOUNDFONTS.length })] })] }), _jsxs("div", { className: "sfm-body", children: [tab === "installed" && (_jsxs("div", { className: "sfm-tab-pane", children: [_jsxs("div", { className: "sfm-toolbar", children: [_jsx("button", { className: "sfm-btn primary", onClick: handleImport, disabled: importingPaths.length > 0, children: t("soundfont.importFile") }), _jsx("button", { className: "sfm-btn", onClick: handleImportDir, disabled: importingPaths.length > 0, children: t("soundfont.importDir") }), _jsx("button", { className: "sfm-btn", onClick: handleOpenFolder, children: t("soundfont.openFolder") }), _jsx("button", { className: "sfm-btn", onClick: handleRescan, disabled: scanning, children: scanning ? t("soundfont.scanning") : t("soundfont.rescan") })] }), _jsx("div", { className: "sfm-list", children: fonts.length === 0 && !scanning ? (_jsxs("div", { className: "sfm-empty", children: [_jsx("p", { children: t("soundfont.empty") }), _jsx("p", { className: "sfm-hint", children: t("soundfont.emptyHint") }), _jsxs("button", { className: "sfm-btn primary", style: { marginTop: 12 }, onClick: () => setTab("download"), children: ["\u2192 ", t("soundfont.tabDownload")] })] })) : (fonts.map((f) => (_jsxs("div", { className: `sfm-font-card ${confirmDelete === f.id ? "confirming" : ""}`, children: [_jsxs("div", { className: "sfm-font-main", children: [_jsx("span", { className: `sfm-format ${f.format}`, children: f.format.toUpperCase() }), _jsx("span", { className: "sfm-font-name", title: f.id, children: f.name }), _jsx("span", { className: "sfm-size", children: formatSize(f.sizeBytes) })] }), _jsx("div", { className: "sfm-font-sub", children: f.presets.length > 0 ? (_jsx("span", { className: "sfm-presets", title: f.presets.map((p) => p.name).join(" · "), children: t("soundfont.presetCount", { count: f.presets.length }) })) : (_jsx("span", { className: "sfm-presets none", children: t("soundfont.noPresets") })) }), _jsx("button", { className: "sfm-delete", onClick: () => void handleDelete(f), title: t("soundfont.deleteTitle"), children: "\u2715" })] }, f.id)))) })] })), tab === "download" && (_jsxs("div", { className: "sfm-tab-pane sfm-download-pane", children: [_jsxs("div", { className: "sfm-section", children: [_jsxs("div", { className: "sfm-section-title", children: [_jsx("span", { className: "sfm-section-dot" }), t("soundfont.oneClick")] }), _jsx("div", { className: "sfm-section-hint", children: t("soundfont.recommendedNote") }), _jsx("div", { className: "sfm-rec-grid", children: DOWNLOADABLE_SOUNDFONTS.map((entry) => (_jsx(AmtSoundfontCard, { entry: entry, lang: lang, isInstalled: !!amtInstalled.find((m) => m.id === entry.id)?.is_available, dl: amtProgress[entry.id], onDownload: () => void downloadSoundfont(entry) }, entry.id))) })] }), _jsxs("div", { className: "sfm-section sfm-section-sfz", children: [_jsxs("div", { className: "sfm-section-title", children: [_jsx("span", { className: "sfm-section-dot sfm-section-dot-sfz" }), t("soundfont.sfzSection")] }), _jsx("div", { className: "sfm-section-hint", children: t("soundfont.sfzSectionHint") }), _jsx("div", { className: "sfm-rec-grid", children: RECOMMENDED_SOUNDFONTS.map((r) => {
                                            // 根据名称模糊匹配已导入的音源（不区分大小写）
                                            const lower = (s) => s.toLowerCase();
                                            const matched = fonts.find((f) => lower(f.name).includes(lower(r.name).substring(0, 8))
                                                || r.matchKeywords.some((kw) => lower(f.id).includes(lower(kw))));
                                            return (_jsx(SfzSoundfontCard, { rec: r, isImported: !!matched, onOpenLink: () => void handleOpenLink(r.url), onImportDir: handleImportDir, purposeLabel: t(r.purposeKey) }, r.name));
                                        }) })] })] }))] }), _jsx(PanelResizeHandles, { start: startResize })] }));
}
/** 可一键下载 SF2 音源卡片（来自 AMT_CATALOG，复用 amt-models 的下载/进度） */
function AmtSoundfontCard({ entry, lang, isInstalled, dl, onDownload, }) {
    const { t } = useTranslation();
    const isDownloading = !!dl;
    return (_jsxs("div", { className: `sfm-rec-card ${isInstalled ? "installed" : ""} ${isDownloading ? "downloading" : ""}`, children: [_jsxs("div", { className: "sfm-rec-card-header", children: [_jsx("span", { className: "sfm-rec-name", children: t18(entry.name, lang) }), _jsx("span", { className: "sfm-rec-format", children: entry.architecture.toUpperCase() })] }), _jsx("p", { className: "sfm-rec-desc", children: t18(entry.description, lang) }), entry.newbie_tip && (_jsxs("div", { className: "sfm-rec-tip", children: ["\uD83D\uDCA1 ", t18(entry.newbie_tip, lang)] })), _jsxs("div", { className: "sfm-rec-meta", children: [_jsx("span", { children: formatSize(entry.fileSize) }), entry.rating && (_jsxs("span", { className: "sfm-rec-rating", children: ["★".repeat(entry.rating), "☆".repeat(5 - entry.rating)] })), entry.ui_features && (_jsx("span", { children: entry.ui_features.map((f) => t18(f, lang)).join(" / ") }))] }), isDownloading && (_jsx("div", { className: "sfm-rec-progress", children: dl.stage === "extract" ? (_jsxs(_Fragment, { children: [_jsx("div", { className: "sfm-rec-progress-bar", style: { width: "100%" } }), _jsx("span", { className: "sfm-rec-progress-text", children: t("soundfont.extracting") })] })) : (_jsxs(_Fragment, { children: [_jsx("div", { className: "sfm-rec-progress-bar", style: { width: dl.total > 0 ? `${(dl.downloaded / dl.total) * 100}%` : "0%" } }), _jsxs("span", { className: "sfm-rec-progress-text", children: [formatSize(dl.downloaded), " / ", dl.total > 0 ? formatSize(dl.total) : "..."] })] })) })), _jsx("button", { className: "sfm-rec-btn", disabled: isDownloading || isInstalled, onClick: onDownload, children: isInstalled
                    ? `✓ ${t("soundfont.installed")}`
                    : isDownloading
                        ? t("soundfont.downloading")
                        : t("soundfont.download") })] }));
}
/** SFZ 目录音源卡片（需手动下载后导入） */
function SfzSoundfontCard({ rec, isImported, onOpenLink, onImportDir, purposeLabel, }) {
    const { t } = useTranslation();
    return (_jsxs("div", { className: `sfm-rec-card sfz ${isImported ? "installed" : ""}`, children: [_jsxs("div", { className: "sfm-rec-card-header", children: [_jsx("span", { className: "sfm-rec-name", children: rec.name }), _jsx("span", { className: "sfm-rec-format sfz", children: "SFZ" })] }), _jsxs("div", { className: "sfm-rec-desc sfz-desc", children: [_jsx("span", { className: "sfz-purpose-badge", children: purposeLabel }), _jsxs("span", { className: "sfz-meta-text", children: [rec.size, " \u00B7 ", rec.license] })] }), _jsx("div", { className: "sfm-rec-actions sfz-actions", children: isImported ? (_jsxs("span", { className: "sfm-rec-installed-tag", children: ["\u2713 ", t("soundfont.importedStatus")] })) : (_jsxs(_Fragment, { children: [_jsx("button", { className: "sfm-rec-btn sfz-download", onClick: onOpenLink, children: t("soundfont.sourceDownload") }), _jsx("button", { className: "sfm-rec-btn sfz-import", onClick: onImportDir, children: t("soundfont.importDir") })] })) })] }));
}
