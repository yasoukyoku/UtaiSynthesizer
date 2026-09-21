import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import "./NodeShell.css";
export function formatWithUnit(value, type) {
    switch (type) {
        case "semitones": {
            const octaves = Math.floor(Math.abs(value) / 12);
            const remainder = Math.abs(value) % 12;
            const sign = value >= 0 ? "+" : "-";
            if (octaves > 0 && remainder === 0) {
                return {
                    value,
                    unit: `${sign}${octaves} 八度`,
                    description: `${sign}${value} 半音`,
                };
            }
            else if (octaves > 0) {
                return {
                    value,
                    unit: `${sign}${octaves}八度${remainder}半音`,
                    description: `${sign}${value} 半音`,
                };
            }
            else {
                return {
                    value,
                    unit: `${sign}${Math.abs(value)} 半音`,
                };
            }
        }
        case "db":
            return {
                value,
                unit: `${value >= 0 ? "+" : ""}${value.toFixed(1)} dB`,
                description: value > 0 ? "增益" : value < 0 ? "衰减" : "无变化",
            };
        case "lufs":
            return {
                value,
                unit: `${value.toFixed(1)} LUFS`,
                description: value >= -14 ? "响度较高" : value >= -23 ? "标准响度" : "响度较低",
            };
        case "ms":
            if (value >= 1000) {
                return {
                    value,
                    unit: `${(value / 1000).toFixed(2)} 秒`,
                    description: `${value} ms`,
                };
            }
            return {
                value,
                unit: `${value} ms`,
            };
        case "hz":
            if (value >= 1000) {
                return {
                    value,
                    unit: `${(value / 1000).toFixed(2)} kHz`,
                    description: `${value} Hz`,
                };
            }
            return {
                value,
                unit: `${value} Hz`,
            };
        case "percent":
            return {
                value,
                unit: `${(value * 100).toFixed(0)}%`,
                description: value === 0 ? "关闭" : value === 1 ? "完全" : undefined,
            };
        case "ratio":
            return {
                value,
                unit: value.toFixed(2),
                description: value === 0 ? "最小" : value === 1 ? "最大" : undefined,
            };
        case "samples":
            if (value >= 48000) {
                return {
                    value,
                    unit: `${(value / 48000).toFixed(2)} 秒`,
                    description: `${value} 采样点 @48kHz`,
                };
            }
            return {
                value,
                unit: `${value} 采样`,
            };
        case "bars":
            return {
                value,
                unit: `${value.toFixed(1)} 小节`,
                description: value === 0 ? "起点" : undefined,
            };
        default:
            return { value, unit: String(value) };
    }
}
export function ParamSliderWithUnit({ label, title, min, max, step, value, onChange, unitType, disabled }) {
    const formatted = formatWithUnit(value, unitType);
    const displayTitle = title || formatted.description;
    return (_jsxs("div", { className: "sep-param-row", children: [_jsx("label", { title: displayTitle, children: label }), _jsxs("span", { className: "sep-overlap nodrag", children: [_jsx("input", { className: "sep-overlap-range nodrag", type: "range", min: min, max: max, step: step, value: value, disabled: disabled, onPointerDown: (e) => e.stopPropagation(), onChange: (e) => onChange(parseFloat(e.target.value)) }), _jsx("span", { className: "sep-overlap-val", title: formatted.description, children: formatted.unit })] })] }));
}
