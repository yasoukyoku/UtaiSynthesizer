import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
// HistoryPanel — S10 撤销/重做面板：列出历史状态（最新在上），点击任意一行跳回该状态。
// 数据来自 history.ts 的只读 API（historyStepList / jumpToHistoryStep），行内文字复用撤销横幅
// 的 history.* i18n 标签，语言一致。
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { historyStepList, jumpToHistoryStep } from "../../store/history";
import "./HistoryPanel.css";
export function HistoryPanel({ onClose }) {
    const { t } = useTranslation();
    // 打开时读取一次；点击跳转后关闭面板（大幅跳转不必逐帧刷新）。
    const [steps] = useState(() => historyStepList());
    // Esc 或点遮罩关闭
    useEffect(() => {
        const onKey = (e) => {
            if (e.key === "Escape")
                onClose();
        };
        window.addEventListener("keydown", onKey);
        return () => window.removeEventListener("keydown", onKey);
    }, [onClose]);
    const jump = (idx) => {
        jumpToHistoryStep(idx);
        onClose();
    };
    // 最新在上
    const ordered = [...steps].reverse();
    return (_jsx("div", { className: "history-backdrop", onMouseDown: onClose, onContextMenu: (e) => { e.preventDefault(); onClose(); }, children: _jsxs("div", { className: "history-panel", onMouseDown: (e) => e.stopPropagation(), children: [_jsxs("div", { className: "history-header", children: [_jsx("span", { className: "history-title", children: t("history.panelTitle") }), _jsx("span", { className: "history-count", children: ordered.length - 1 })] }), _jsx("div", { className: "history-list", children: ordered.map((s) => (_jsxs("button", { type: "button", className: `history-row ${s.isCurrent ? "current" : ""}`, disabled: s.isCurrent, onClick: () => jump(s.idx), title: s.isCurrent ? t("history.current") : "", children: [_jsx("span", { className: "history-badge", children: s.idx + 1 }), _jsx("span", { className: "history-label", children: t(`history.${s.labelKey}`) })] }, s.idx))) })] }) }));
}
