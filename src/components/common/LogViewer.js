import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useTranslation } from "react-i18next";
import { useLogStore } from "../../store/logs";
import { useFloatingPanel } from "../../lib/useFloatingPanel";
import { loadSetting, saveSetting } from "../../lib/settings";
import { PanelResizeHandles } from "./PanelResizeHandles";
import "./LogViewer.css";
/** S67 readability: the log body font is user-adjustable (A- / A+), persisted. */
const FONT_MIN = 9;
const FONT_MAX = 16;
const FONT_DEFAULT = 12;
const LEVEL_FILTERS = ["ALL", "ERROR", "WARN", "INFO", "DEBUG"];
const LEVEL_KEY = {
    ALL: "all", ERROR: "error", WARN: "warn", INFO: "info", DEBUG: "debug",
};
export function LogViewer({ onClose }) {
    const { t } = useTranslation();
    const { entries, logDir, startPolling, stopPolling } = useLogStore();
    const [filter, setFilter] = useState("ALL");
    const [search, setSearch] = useState("");
    const [autoScroll, setAutoScroll] = useState(true);
    const listRef = useRef(null);
    const { style, startDrag, startResize } = useFloatingPanel({
        storageKey: "utai.logViewerRect",
        initial: () => ({ x: 128, y: 108, w: 480, h: Math.round(window.innerHeight * 0.6) }),
        minW: 360,
        minH: 240,
    });
    const [fontSize, setFontSize] = useState(() => loadSetting("utai.logFontSize", FONT_DEFAULT));
    const bumpFont = (d) => setFontSize((f) => {
        const n = Math.min(FONT_MAX, Math.max(FONT_MIN, f + d));
        saveSetting("utai.logFontSize", n);
        return n;
    });
    useEffect(() => {
        startPolling();
        return () => stopPolling();
    }, [startPolling, stopPolling]);
    useEffect(() => {
        if (autoScroll && listRef.current) {
            listRef.current.scrollTop = listRef.current.scrollHeight;
        }
    }, [entries.length, autoScroll]);
    const filtered = entries.filter((e) => {
        if (filter !== "ALL" && e.level !== filter)
            return false;
        if (search && !e.message.toLowerCase().includes(search.toLowerCase()) &&
            !e.module.toLowerCase().includes(search.toLowerCase()))
            return false;
        return true;
    });
    const handleCopy = () => {
        const text = filtered
            .map((e) => `[${e.timestamp}] ${e.level} ${e.module}: ${e.message}`)
            .join("\n");
        navigator.clipboard.writeText(text);
    };
    const handleScroll = () => {
        if (!listRef.current)
            return;
        const { scrollTop, scrollHeight, clientHeight } = listRef.current;
        setAutoScroll(scrollHeight - scrollTop - clientHeight < 40);
    };
    return (_jsxs("aside", { className: "log-viewer", style: style, children: [_jsxs("div", { className: "panel-header", onMouseDown: startDrag, children: [_jsx("span", { className: "panel-title", children: t("log.title") }), _jsx("button", { className: "panel-close", onClick: onClose, children: "X" })] }), _jsxs("div", { className: "log-toolbar", children: [_jsx("div", { className: "log-filters", children: LEVEL_FILTERS.map((lvl) => (_jsx("button", { className: filter === lvl ? "active" : "", onClick: () => setFilter(lvl), children: t(`log.${LEVEL_KEY[lvl]}`) }, lvl))) }), _jsx("input", { type: "text", className: "log-search", placeholder: t("log.search"), value: search, onChange: (e) => setSearch(e.target.value) }), _jsx("button", { className: "log-copy-btn log-icon-btn", onClick: () => bumpFont(-1), title: t("log.fontSmaller"), children: _jsx(ZoomIcon, { plus: false }) }), _jsx("button", { className: "log-copy-btn log-icon-btn", onClick: () => bumpFont(1), title: t("log.fontLarger"), children: _jsx(ZoomIcon, { plus: true }) }), _jsx("button", { className: "log-copy-btn", onClick: handleCopy, title: t("log.copyTitle"), children: t("log.copy") })] }), _jsxs("div", { className: "log-entries", ref: listRef, onScroll: handleScroll, style: { fontSize }, children: [filtered.map((entry, i) => (_jsx(LogLine, { entry: entry }, i))), filtered.length === 0 && (_jsx("div", { className: "log-empty", children: t("log.empty") }))] }), _jsxs("div", { className: "log-footer", children: [_jsxs("span", { className: "log-count mono", children: [filtered.length, " / ", entries.length] }), _jsx("span", { className: "log-dir mono", title: logDir, children: logDir }), _jsx("button", { className: "log-copy-btn log-icon-btn log-open-btn", onClick: () => void invoke("open_log_dir").catch(() => { }), title: t("log.openDir"), children: _jsx("svg", { width: "11", height: "11", viewBox: "0 0 12 12", "aria-hidden": "true", children: _jsx("path", { d: "M1 2.5h3.5l1 1.5H11v5.5H1z", fill: "none", stroke: "currentColor" }) }) })] }), _jsx(PanelResizeHandles, { start: startResize })] }));
}
/** Magnifier +/- for the font-size buttons (§user: reads as "zoom", not "append text"). */
function ZoomIcon({ plus }) {
    return (_jsxs("svg", { width: "11", height: "11", viewBox: "0 0 12 12", "aria-hidden": "true", children: [_jsx("circle", { cx: "5", cy: "5", r: "3.6", fill: "none", stroke: "currentColor" }), _jsx("path", { d: "M7.8 7.8 L11 11", stroke: "currentColor" }), _jsx("path", { d: "M3.4 5 H6.6", stroke: "currentColor" }), plus && _jsx("path", { d: "M5 3.4 V6.6", stroke: "currentColor" })] }));
}
function LogLine({ entry }) {
    const levelClass = `log-level-${entry.level.toLowerCase()}`;
    const time = entry.timestamp.split("T")[1]?.substring(0, 12) ?? "";
    return (_jsxs("div", { className: `log-line ${levelClass}`, children: [_jsx("span", { className: "log-time", children: time }), _jsx("span", { className: `log-lvl ${levelClass}`, children: entry.level.charAt(0) }), _jsx("span", { className: "log-module", children: entry.module }), _jsx("span", { className: "log-msg", children: entry.message })] }));
}
