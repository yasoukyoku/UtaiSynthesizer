import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useState } from "react";
import { useAppStore } from "../../store/app";
import "./HistoryBanner.css";
/**
 * Transient top-right corner banner. Tells the user what just happened (undo/redo of WHAT, save/load
 * confirmation, …) without disturbing the view/flow — replaced the old reveal-on-undo viewport scroll.
 * A single element that re-triggers in place on each event (keyed on the store `seq`); the inner block
 * is keyed by `seq` so the slide-in + countdown line restart on a rapid retrigger (no stacking).
 */
const KIND_COLOR = {
    undo: "var(--accent-primary)",
    redo: "var(--accent-secondary)",
    save: "var(--color-success)",
    load: "var(--accent-tertiary)",
    info: "var(--accent-primary)",
};
const svg = { viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 2, strokeLinecap: "round", strokeLinejoin: "round" };
function Icon({ kind }) {
    switch (kind) {
        case "undo":
            return _jsxs("svg", { ...svg, children: [_jsx("path", { d: "M9 14 4 9l5-5" }), _jsx("path", { d: "M4 9h11a5 5 0 0 1 0 10h-3" })] });
        case "redo":
            return _jsxs("svg", { ...svg, children: [_jsx("path", { d: "m15 14 5-5-5-5" }), _jsx("path", { d: "M20 9H9a5 5 0 0 0 0 10h3" })] });
        case "save":
            return _jsxs("svg", { ...svg, children: [_jsx("path", { d: "M12 3v12" }), _jsx("path", { d: "m7 10 5 5 5-5" }), _jsx("path", { d: "M5 21h14" })] });
        case "load":
            return _jsxs("svg", { ...svg, children: [_jsx("path", { d: "M12 21V9" }), _jsx("path", { d: "m7 14 5-5 5 5" }), _jsx("path", { d: "M5 3h14" })] });
        default:
            return _jsx("svg", { ...svg, children: _jsx("circle", { cx: "12", cy: "12", r: "9" }) });
    }
}
export function HistoryBanner() {
    const banner = useAppStore((s) => s.banner);
    const [shown, setShown] = useState(false);
    useEffect(() => {
        if (!banner)
            return;
        setShown(true);
        const id = setTimeout(() => setShown(false), 1800);
        return () => clearTimeout(id);
    }, [banner?.seq]);
    if (!banner)
        return null;
    return (_jsx("div", { className: `app-banner ${shown ? "show" : ""}`, style: { ["--ab-color"]: KIND_COLOR[banner.kind] }, "aria-live": "polite", children: _jsxs("div", { className: "app-banner-inner", children: [_jsx("span", { className: "app-banner-icon", children: _jsx(Icon, { kind: banner.kind }) }), _jsx("span", { className: "app-banner-text", children: banner.message })] }, banner.seq) }));
}
