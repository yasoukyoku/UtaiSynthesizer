import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect } from "react";
import { useTranslation } from "react-i18next";
import { t18 } from "../../lib/models/msst-catalog";
import "./ShortcutsDialog.css";
const S11 = (zh, en, ja, keys) => [zh, en, ja, keys];
/** 快捷键帮助面板 —— 汇总常用键盘操作。Esc / 点击背景关闭。 */
export function ShortcutsDialog({ onClose }) {
    const { i18n } = useTranslation();
    const lang = i18n.language;
    useEffect(() => {
        const onKey = (e) => {
            if (e.key === "Escape")
                onClose();
        };
        window.addEventListener("keydown", onKey);
        return () => window.removeEventListener("keydown", onKey);
    }, [onClose]);
    const ROWS = [
        S11("保存工程", "Save project", "プロジェクトを保存", "Ctrl / ⌘ + S"),
        S11("另存为", "Save as", "名前を付けて保存", "Ctrl / ⌘ + Shift + S"),
        S11("打开工程", "Open project", "開く", "Ctrl / ⌘ + O"),
        S11("新建工程", "New project", "新規", "Ctrl / ⌘ + N"),
        S11("撤销", "Undo", "元に戻す", "Ctrl / ⌘ + Z"),
        S11("重做", "Redo", "やり直し", "Ctrl / ⌘ + Y"),
        S11("全选", "Select all", "すべて選択", "Ctrl / ⌘ + A"),
        S11("复制", "Copy", "コピー", "Ctrl / ⌘ + C"),
        S11("剪切", "Cut", "カット", "Ctrl / ⌘ + X"),
        S11("粘贴", "Paste", "ペースト", "Ctrl / ⌘ + V"),
        S11("删除所选", "Delete selection", "選択を削除", "Delete"),
        S11("切片 / 分割", "Slice clip", "分割", "Ctrl / ⌘ + K"),
        S11("横向缩放", "Zoom horizontally", "横方向ズーム", "滑块 / 滚轮"),
        S11("纵向缩放", "Zoom vertically", "縦方向ズーム", "Alt + 滚轮"),
        S11("播放 / 暂停", "Play / Pause", "再生 / 一時停止", "空格"),
    ];
    return (_jsx("div", { className: "shortcuts-overlay", onClick: onClose, children: _jsxs("div", { className: "shortcuts-panel", onClick: (e) => e.stopPropagation(), children: [_jsxs("div", { className: "shortcuts-head", children: [_jsx("span", { className: "shortcuts-title", children: t18({ zh: "快捷键帮助", en: "Keyboard Shortcuts", ja: "ショートカット" }, lang) }), _jsx("button", { className: "shortcuts-close", onClick: onClose, children: "\u2715" })] }), _jsx("div", { className: "shortcuts-body", children: ROWS.map(([zh, en, ja, keys]) => (_jsxs("div", { className: "shortcuts-row", children: [_jsx("span", { className: "shortcuts-desc", children: t18({ zh, en, ja }, lang) }), _jsx("span", { className: "shortcuts-keys", children: keys.split(" / ").map((k) => (_jsx("kbd", { children: k }, k))) })] }, `${zh}-${keys}`))) }), _jsx("div", { className: "shortcuts-foot", children: t18({ zh: "按 Esc 或点击空白处关闭", en: "Press Esc or click outside to close", ja: "Esc キーまたは外側クリックで閉じます" }, lang) })] }) }));
}
