import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useLogStore, type LogEntry } from "../../store/logs";
import { useDraggable } from "../../lib/useDraggable";
import "./LogViewer.css";

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
  const { pos, startDrag } = useDraggable(() => ({ x: 128, y: 108 }));

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
    <aside className="log-viewer" style={{ left: pos.x, top: pos.y }}>
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
        <button className="log-copy-btn" onClick={handleCopy} title={t("log.copyTitle")}>
          {t("log.copy")}
        </button>
      </div>

      <div className="log-entries" ref={listRef} onScroll={handleScroll}>
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
      </div>
    </aside>
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
