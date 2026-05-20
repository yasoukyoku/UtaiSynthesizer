import { useEffect, useRef } from "react";
import "./ContextMenu.css";

export interface MenuItem {
  label: string;
  shortcut?: string;
  danger?: boolean;
  disabled?: boolean;
  onClick: () => void;
}

interface Props {
  x: number;
  y: number;
  items: MenuItem[];
  onClose: () => void;
}

export function ContextMenu({ x, y, items, onClose }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handle = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        onClose();
      }
    };
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("mousedown", handle);
    document.addEventListener("keydown", handleKey);
    return () => {
      document.removeEventListener("mousedown", handle);
      document.removeEventListener("keydown", handleKey);
    };
  }, [onClose]);

  // Keep menu within viewport
  const style: React.CSSProperties = {
    left: x,
    top: y,
  };

  return (
    <div className="ctx-menu" ref={ref} style={style}>
      {items.map((item, i) => (
        <button
          key={i}
          className={`ctx-item ${item.danger ? "ctx-danger" : ""}`}
          disabled={item.disabled}
          onClick={() => {
            item.onClick();
            onClose();
          }}
        >
          <span className="ctx-label">{item.label}</span>
          {item.shortcut && <span className="ctx-shortcut mono">{item.shortcut}</span>}
        </button>
      ))}
    </div>
  );
}
