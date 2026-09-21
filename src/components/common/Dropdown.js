import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * Custom themed dropdown replacing native <select> (house style: square corners,
 * theme tokens, self-drawn arrow + options panel — the WebView2 native popup
 * ignores the app theme entirely). Keyboard: Enter/Space/ArrowDown open,
 * arrows move, Enter selects, Esc closes; click-outside closes.
 */
import { useEffect, useRef, useState } from "react";
import "./Dropdown.css";
export function Dropdown({ value, options, onChange, className, }) {
    const [open, setOpen] = useState(false);
    const [hover, setHover] = useState(-1);
    const rootRef = useRef(null);
    const selectedIdx = options.findIndex((o) => o.value === value);
    const selected = options[selectedIdx];
    useEffect(() => {
        if (!open)
            return;
        const onDocDown = (e) => {
            if (rootRef.current && !rootRef.current.contains(e.target)) {
                setOpen(false);
            }
        };
        document.addEventListener("pointerdown", onDocDown, true);
        return () => document.removeEventListener("pointerdown", onDocDown, true);
    }, [open]);
    const openPanel = () => {
        setHover(selectedIdx >= 0 ? selectedIdx : 0);
        setOpen(true);
    };
    const commit = (idx) => {
        const opt = options[idx];
        if (opt?.disabled)
            return; // stays open — a dead click must not read as "accepted"
        if (opt)
            onChange(opt.value);
        setOpen(false);
    };
    /** Arrow keys SKIP disabled entries — landing on one and pressing Enter would read as broken. */
    const moveHover = (step) => setHover((h) => {
        for (let i = h + step; i >= 0 && i < options.length; i += step) {
            if (!options[i]?.disabled)
                return i;
        }
        return h;
    });
    const onKeyDown = (e) => {
        if (!open) {
            if (e.key === "Enter" || e.key === " " || e.key === "ArrowDown") {
                e.preventDefault();
                openPanel();
            }
            return;
        }
        if (e.key === "Escape") {
            e.preventDefault();
            setOpen(false);
        }
        else if (e.key === "ArrowDown") {
            e.preventDefault();
            moveHover(1);
        }
        else if (e.key === "ArrowUp") {
            e.preventDefault();
            moveHover(-1);
        }
        else if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            commit(hover);
        }
        else if (e.key === "Tab") {
            setOpen(false);
        }
    };
    return (_jsxs("div", { className: `ut-dropdown ${className ?? ""}`, ref: rootRef, children: [_jsxs("button", { type: "button", className: `ut-dropdown-trigger ${open ? "open" : ""}`, onClick: () => (open ? setOpen(false) : openPanel()), onKeyDown: onKeyDown, "aria-haspopup": "listbox", "aria-expanded": open, children: [_jsx("span", { className: "ut-dropdown-label", children: selected?.label ?? "" }), _jsx("span", { className: "ut-dropdown-arrow", children: open ? "▴" : "▾" })] }), open && (_jsx("div", { className: "ut-dropdown-panel", role: "listbox", children: options.map((o, i) => (_jsx("div", { role: "option", "aria-selected": i === selectedIdx, "aria-disabled": o.disabled || undefined, title: o.title ?? o.label, className: `ut-dropdown-option ${i === selectedIdx ? "selected" : ""} ${i === hover ? "hover" : ""} ${o.disabled ? "disabled" : ""}`, onPointerEnter: () => !o.disabled && setHover(i), onPointerDown: (e) => {
                        // pointerdown (not click): commit before the outside-click
                        // closer sees the event, and before focus shifts
                        e.preventDefault();
                        commit(i);
                    }, children: o.label }, String(o.value)))) }))] }));
}
