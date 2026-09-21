import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useRef, useState } from "react";
import "./ContextMenu.css";
export function ContextMenu({ x, y, items, onClose }) {
    const ref = useRef(null);
    const [pos, setPos] = useState({ left: x, top: y });
    useEffect(() => {
        const el = ref.current;
        if (!el)
            return;
        const rect = el.getBoundingClientRect();
        let left = x;
        let top = y;
        if (left + rect.width > window.innerWidth)
            left = window.innerWidth - rect.width - 4;
        if (top + rect.height > window.innerHeight)
            top = window.innerHeight - rect.height - 4;
        if (left < 0)
            left = 4;
        if (top < 0)
            top = 4;
        setPos({ left, top });
        // items.length: a menu whose items arrive ASYNC (the vocal-track singer picker fetches the model
        // list after opening) grows after the first clamp ran — re-clamp so it never extends off-screen.
    }, [x, y, items.length]);
    useEffect(() => {
        const handle = (e) => {
            if (ref.current && !ref.current.contains(e.target)) {
                onClose();
            }
        };
        const handleKey = (e) => {
            if (e.key === "Escape")
                onClose();
        };
        document.addEventListener("mousedown", handle);
        document.addEventListener("keydown", handleKey);
        return () => {
            document.removeEventListener("mousedown", handle);
            document.removeEventListener("keydown", handleKey);
        };
    }, [onClose]);
    return (_jsx("div", { className: "ctx-menu", ref: ref, style: pos, children: items.map((item, i) => "type" in item && item.type === "divider" ? (_jsx("div", { className: "ctx-divider", role: "separator" }, i)) : "type" in item && item.type === "submenu" ? (_jsx(SubMenuItem, { item: item, onClose: onClose }, i)) : (_jsxs("button", { className: `ctx-item ${item.danger ? "ctx-danger" : ""} ${item.active ? "ctx-active" : ""}`, disabled: item.disabled, title: item.title, onClick: () => {
                item.onClick();
                onClose();
            }, children: [_jsxs("span", { className: "ctx-label", children: [item.icon ? _jsx("span", { className: "ctx-icon", "aria-hidden": "true", children: item.icon }) : null, item.label] }), item.shortcut && _jsx("span", { className: "ctx-shortcut mono", children: item.shortcut })] }, i))) }));
}
// ── 二级子菜单 ─────────────────────────────────────────────────────────────
// 悬停展开；右侧无空间时翻到左侧；面板超高内部滚动（复用 .ctx-menu 滚动样式）。
// 延迟关闭：离开触发项/面板有 150ms 宽限，穿过缝隙不闪断。
function SubMenuItem({ item, onClose }) {
    const wrapRef = useRef(null);
    const [open, setOpen] = useState(false);
    const [flip, setFlip] = useState(false);
    const closeTimer = useRef(null);
    const openNow = () => {
        if (closeTimer.current !== null) {
            window.clearTimeout(closeTimer.current);
            closeTimer.current = null;
        }
        const el = wrapRef.current;
        if (el) {
            const rect = el.getBoundingClientRect();
            setFlip(rect.right + 200 > window.innerWidth && rect.left > 200);
        }
        setOpen(true);
    };
    const closeSoon = () => {
        if (closeTimer.current !== null)
            window.clearTimeout(closeTimer.current);
        closeTimer.current = window.setTimeout(() => setOpen(false), 150);
    };
    useEffect(() => () => {
        if (closeTimer.current !== null)
            window.clearTimeout(closeTimer.current);
    }, []);
    return (_jsxs("div", { ref: wrapRef, className: "ctx-submenu-wrap", onMouseEnter: openNow, onMouseLeave: closeSoon, children: [_jsxs("button", { className: `ctx-item ctx-submenu ${item.danger ? "ctx-danger" : ""} ${open ? "ctx-open" : ""}`, disabled: item.disabled, title: item.title, onClick: () => (open ? setOpen(false) : openNow()), children: [_jsxs("span", { className: "ctx-label", children: [item.icon ? _jsx("span", { className: "ctx-icon", "aria-hidden": "true", children: item.icon }) : null, item.label] }), _jsx("span", { className: "ctx-arrow", "aria-hidden": "true", children: "\u25B8" })] }), open && !item.disabled && (_jsx("div", { className: `ctx-menu ctx-submenu-panel ${flip ? "ctx-flip-left" : ""}`, children: item.items.map((sub, i) => "type" in sub && sub.type === "divider" ? (_jsx("div", { className: "ctx-divider", role: "separator" }, i)) : "type" in sub && sub.type === "submenu" ? (_jsx(SubMenuItem, { item: sub, onClose: onClose }, i)) : (_jsxs("button", { className: `ctx-item ${sub.danger ? "ctx-danger" : ""} ${sub.active ? "ctx-active" : ""}`, disabled: sub.disabled, title: sub.title, onClick: () => {
                        sub.onClick();
                        onClose();
                    }, children: [_jsxs("span", { className: "ctx-label", children: [sub.icon ? _jsx("span", { className: "ctx-icon", "aria-hidden": "true", children: sub.icon }) : null, sub.label] }), sub.shortcut && _jsx("span", { className: "ctx-shortcut mono", children: sub.shortcut })] }, i))) }))] }));
}
