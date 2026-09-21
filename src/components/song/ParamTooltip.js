import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * 参数悬停提示组件
 * 显示：推荐值 + 效果说明 + 示例场景
 */
import { useState, useRef } from "react";
import "./ParamTooltip.css";
export function ParamTooltip({ paramName, recommendedValue, description, example, children, }) {
    const [visible, setVisible] = useState(false);
    const [position, setPosition] = useState({ x: 0, y: 0 });
    const containerRef = useRef(null);
    const tooltipRef = useRef(null);
    const updatePosition = () => {
        const container = containerRef.current;
        const tooltip = tooltipRef.current;
        if (!container || !tooltip)
            return;
        const rect = container.getBoundingClientRect();
        const tooltipRect = tooltip.getBoundingClientRect();
        let x = rect.left;
        let y = rect.bottom + 8;
        if (x + tooltipRect.width > window.innerWidth - 10) {
            x = Math.max(10, window.innerWidth - tooltipRect.width - 10);
        }
        if (y + tooltipRect.height > window.innerHeight - 10) {
            y = rect.top - tooltipRect.height - 8;
        }
        setPosition({ x, y });
    };
    const handleMouseEnter = () => {
        updatePosition();
        setVisible(true);
    };
    const handleMouseLeave = () => setVisible(false);
    return (_jsxs("div", { ref: containerRef, className: "param-tooltip-wrapper", onMouseEnter: handleMouseEnter, onMouseLeave: handleMouseLeave, children: [children, _jsxs("div", { ref: tooltipRef, className: `param-tooltip-content ${visible ? "is-visible" : ""}`, style: { left: position.x, top: position.y }, role: "tooltip", children: [_jsxs("div", { className: "param-tooltip-header", children: [_jsx("span", { className: "param-tooltip-name", children: paramName }), recommendedValue !== undefined && recommendedValue !== "" && (_jsxs("span", { className: "param-tooltip-recommended", children: ["\u63A8\u8350: ", recommendedValue] }))] }), description && (_jsx("div", { className: "param-tooltip-description", children: description })), example && (_jsxs("div", { className: "param-tooltip-example", children: [_jsx("span", { className: "param-tooltip-example-label", children: "\uD83D\uDCA1 \u793A\u4F8B\u573A\u666F" }), example] }))] })] }));
}
