// HistoryPanel — S10 撤销/重做面板：列出历史状态（最新在上），点击任意一行跳回该状态。
// 数据来自 history.ts 的只读 API（historyStepList / jumpToHistoryStep），行内文字复用撤销横幅
// 的 history.* i18n 标签，语言一致。

import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { historyStepList, jumpToHistoryStep, type HistoryStep } from "../../store/history";
import "./HistoryPanel.css";

export function HistoryPanel({ onClose }: { onClose: () => void }) {
  const { t } = useTranslation();
  // 打开时读取一次；点击跳转后关闭面板（大幅跳转不必逐帧刷新）。
  const [steps] = useState<HistoryStep[]>(() => historyStepList());

  // Esc 或点遮罩关闭
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const jump = (idx: number) => {
    jumpToHistoryStep(idx);
    onClose();
  };

  // 最新在上
  const ordered = [...steps].reverse();

  return (
    <div className="history-backdrop" onMouseDown={onClose} onContextMenu={(e) => { e.preventDefault(); onClose(); }}>
      <div className="history-panel" onMouseDown={(e) => e.stopPropagation()}>
        <div className="history-header">
          <span className="history-title">{t("history.panelTitle")}</span>
          <span className="history-count">{ordered.length - 1}</span>
        </div>
        <div className="history-list">
          {ordered.map((s) => (
            <button
              key={s.idx}
              type="button"
              className={`history-row ${s.isCurrent ? "current" : ""}`}
              disabled={s.isCurrent}
              onClick={() => jump(s.idx)}
              title={s.isCurrent ? t("history.current") : ""}
            >
              <span className="history-badge">{s.idx + 1}</span>
              <span className="history-label">{t(`history.${s.labelKey}`)}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}