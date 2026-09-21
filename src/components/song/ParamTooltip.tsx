/**
 * 参数悬停提示组件
 * 显示：推荐值 + 效果说明 + 示例场景
 */

import { useCallback, useRef, useState } from "react";
import "./ParamTooltip.css";

export interface ParamTooltipProps {
  paramName: string;
  recommendedValue?: string | number;
  description: string;
  example?: string;
  children: React.ReactNode;
}

export function ParamTooltip({
  paramName,
  recommendedValue,
  description,
  example,
  children,
}: ParamTooltipProps) {
  const [visible, setVisible] = useState(false);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);

  const updatePosition = useCallback((clientX: number, clientY: number) => {
    const tooltip = tooltipRef.current;
    if (!tooltip) return;

    const margin = 12;
    const gap = 14;
    const tooltipRect = tooltip.getBoundingClientRect();
    const rightX = clientX + gap;
    const leftX = clientX - tooltipRect.width - gap;
    const bottomY = clientY + gap;
    const topY = clientY - tooltipRect.height - gap;
    const x = rightX + tooltipRect.width <= window.innerWidth - margin
      ? rightX
      : Math.max(margin, leftX);
    const y = bottomY + tooltipRect.height <= window.innerHeight - margin
      ? bottomY
      : Math.max(margin, topY);

    setPosition({
      x: Math.min(Math.max(margin, x), window.innerWidth - tooltipRect.width - margin),
      y: Math.min(Math.max(margin, y), window.innerHeight - tooltipRect.height - margin),
    });
  }, []);

  const handlePointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    if (visible) updatePosition(event.clientX, event.clientY);
  };

  const handleMouseEnter = (event: React.MouseEvent<HTMLDivElement>) => {
    setVisible(true);
    requestAnimationFrame(() => updatePosition(event.clientX, event.clientY));
  };

  const handleMouseLeave = () => setVisible(false);

  return (
    <div
      ref={containerRef}
      className="param-tooltip-wrapper"
      onMouseEnter={handleMouseEnter}
      onPointerMove={handlePointerMove}
      onMouseLeave={handleMouseLeave}
    >
      {children}
      <div
        ref={tooltipRef}
        className={`param-tooltip-content ${visible ? "is-visible" : ""}`}
        style={{ left: position.x, top: position.y }}
        role="tooltip"
      >
        <div className="param-tooltip-header">
          <span className="param-tooltip-name">{paramName}</span>
          {recommendedValue !== undefined && recommendedValue !== "" && (
            <span className="param-tooltip-recommended">
              推荐: {recommendedValue}
            </span>
          )}
        </div>
        {description && (
          <div className="param-tooltip-description">{description}</div>
        )}
        {example && (
          <div className="param-tooltip-example">
            <span className="param-tooltip-example-label">💡 示例场景</span>
            {example}
          </div>
        )}
      </div>
    </div>
  );
}
