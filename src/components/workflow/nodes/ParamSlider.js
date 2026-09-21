import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import "./NodeShell.css";
/**
 * Shared thin-bar slider row for workflow-node params (label + range + value readout) — ONE
 * source of truth for the markup that was previously pasted per-param inside SeparationNode.
 * The `sep-*` classes in NodeShell.css carry the S24 slider-specificity fix (`.sep-overlap`
 * prefix beats the global `input[type="range"]` thumb): reuse them, never copy the CSS.
 */
export function ParamSlider({ label, title, min, max, step, value, onChange, format, disabled }) {
    return (_jsxs("div", { className: "sep-param-row", children: [_jsx("label", { title: title, children: label }), _jsxs("span", { className: "sep-overlap nodrag", children: [_jsx("input", { className: "sep-overlap-range nodrag", type: "range", min: min, max: max, step: step, value: value, disabled: disabled, onPointerDown: (e) => e.stopPropagation(), onChange: (e) => onChange(parseFloat(e.target.value)) }), _jsx("span", { className: "sep-overlap-val", children: format ? format(value) : String(value) })] })] }));
}
/** Two-decimal readout for 0..1 ratio sliders. */
export function formatRatio(v) {
    return v.toFixed(2);
}
