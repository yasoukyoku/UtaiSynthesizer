import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useTranslation } from "react-i18next";
import { useLogStore, type LogEntry } from "../../store/logs";
import { useFloatingPanel } from "../../lib/useFloatingPanel";
import { loadSetting, saveSetting } from "../../lib/settings";
import { PanelResizeHandles } from "./PanelResizeHandles";
import "./LogViewer.css";

/** S67 readability: the log body font is user-adjustable (A- / A+), persisted. */
const FONT_MIN = 9;
const FONT_MAX = 16;
const FONT_DEFAULT = 12;

const LEVEL_FILTERS = ["ALL", "ERROR", "WARN", "INFO", "DEBUG"] as const;
const LEVEL_KEY: Record<string, string> = {
  ALL: "all", ERROR: "error", WARN: "warn", INFO: "info", DEBUG: "debug",
};

export function LogViewer({ onClose }: { onClose: () => void }) {
  const { t } = useTranslation();
  const { entries, logDir, startPolling, stopPolling } = useLogStore();
  const [filter, setFilter] = useState<string>("ALL");
  const [search, setSearch] = useState("");
  const [autoScroll, setAutoScroll] = useState(true);
  const listRef = useRef<HTMLDivElement>(null);
  const { style, startDrag, startResize } = useFloatingPanel({
    storageKey: "utai.logViewerRect",
    initial: () => ({ x: 128, y: 108, w: 480, h: Math.round(window.innerHeight * 0.6) }),
    minW: 360,
    minH: 240,
  });
  const [fontSize, setFontSize] = useState(() => loadSetting("utai.logFontSize", FONT_DEFAULT));
  const bumpFont = (d: number) =>
    setFontSize((f) => {
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
    if (filter !== "ALL" && e.level !== filter) return false;
    if (search && !e.message.toLowerCase().includes(search.toLowerCase()) &&
        !e.module.toLowerCase().includes(search.toLowerCase())) return false;
    return true;
  });

  const handleCopy = () => {
    const text = filtered
      .map((e) => `[${e.timestamp}] ${e.level} ${e.module}: ${e.message}`)
      .join("\n");
    navigator.clipboard.writeText(text);
  };

  const handleScroll = () => {
    if (!listRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = listRef.current;
    setAutoScroll(scrollHeight - scrollTop - clientHeight < 40);
  };

  return (
    <aside className="log-viewer" style={style}>
      <div className="panel-header" onMouseDown={startDrag}>
        <span className="panel-title">{t("log.title")}</span>
        <button className="panel-close" onClick={onClose}>X</button>
      </div>

      <div className="log-toolbar">
        <div className="log-filters">
          {LEVEL_FILTERS.map((lvl) => (
            <button
              key={lvl}
              className={filter === lvl ? "active" : ""}
              onClick={() => setFilter(lvl)}
            >
              {t(`log.${LEVEL_KEY[lvl]}`)}
            </button>
          ))}
        </div>
        <input
          type="text"
          className="log-search"
          placeholder={t("log.search")}
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <button className="log-copy-btn log-icon-btn" onClick={() => bumpFont(-1)} title={t("log.fontSmaller")}>
          <ZoomIcon plus={false} />
        </button>
        <button className="log-copy-btn log-icon-btn" onClick={() => bumpFont(1)} title={t("log.fontLarger")}>
          <ZoomIcon plus />
        </button>
        <button className="log-copy-btn" onClick={handleCopy} title={t("log.copyTitle")}>
          {t("log.copy")}
        </button>
      </div>

      <div className="log-entries" ref={listRef} onScroll={handleScroll} style={{ fontSize }}>
        {filtered.map((entry, i) => (
          <LogLine key={i} entry={entry} />
        ))}
        {filtered.length === 0 && (
          <div className="log-empty">{t("log.empty")}</div>
        )}
      </div>

      <div className="log-footer">
        <span className="log-count mono">{filtered.length} / {entries.length}</span>
        <span className="log-dir mono" title={logDir}>{logDir}</span>
        <button
          className="log-copy-btn log-icon-btn log-open-btn"
          onClick={() => void invoke("open_log_dir").catch(() => {})}
          title={t("log.openDir")}
        >
          <svg width="11" height="11" viewBox="0 0 12 12" aria-hidden="true">
            <path d="M1 2.5h3.5l1 1.5H11v5.5H1z" fill="none" stroke="currentColor" />
          </svg>
        </button>
      </div>
      <PanelResizeHandles start={startResize} />
    </aside>
  );
}

/** Magnifier +/- for the font-size buttons (§user: reads as "zoom", not "append text"). */
function ZoomIcon({ plus }: { plus: boolean }) {
  return (
    <svg width="11" height="11" viewBox="0 0 12 12" aria-hidden="true">
      <circle cx="5" cy="5" r="3.6" fill="none" stroke="currentColor" />
      <path d="M7.8 7.8 L11 11" stroke="currentColor" />
      <path d="M3.4 5 H6.6" stroke="currentColor" />
      {plus && <path d="M5 3.4 V6.6" stroke="currentColor" />}
    </svg>
  );
}

function LogLine({ entry }: { entry: LogEntry }) {
  const levelClass = `log-level-${entry.level.toLowerCase()}`;
  const time = entry.timestamp.split("T")[1]?.substring(0, 12) ?? "";

  return (
    <div className={`log-line ${levelClass}`}>
      <span className="log-time">{time}</span>
      <span className={`log-lvl ${levelClass}`}>{entry.level.charAt(0)}</span>
      <span className="log-module">{entry.module}</span>
      <span className="log-msg">{entry.message}</span>
    </div>
  );
}
