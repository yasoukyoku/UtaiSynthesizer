import { useEffect, useRef, useState } from "react";
import "./ContextMenu.css";

export interface MenuItem {
  label: string;
  shortcut?: string;
  danger?: boolean;
  disabled?: boolean;
  /** The CURRENT selection in a picker-style menu — highlighted in the accent color (still clickable;
   *  re-picking is a harmless no-op). Never use `disabled` for "current": gray reads as "unavailable". */
  active?: boolean;
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
  const [pos, setPos] = useState({ left: x, top: y });

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    let left = x;
    let top = y;
    if (left + rect.width > window.innerWidth) left = window.innerWidth - rect.width - 4;
    if (top + rect.height > window.innerHeight) top = window.innerHeight - rect.height - 4;
    if (left < 0) left = 4;
    if (top < 0) top = 4;
    setPos({ left, top });
    // items.length: a menu whose items arrive ASYNC (the vocal-track singer picker fetches the model
    // list after opening) grows after the first clamp ran — re-clamp so it never extends off-screen.
  }, [x, y, items.length]);

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

  return (
    <div className="ctx-menu" ref={ref} style={pos}>
      {items.map((item, i) => (
        <button
          key={i}
          className={`ctx-item ${item.danger ? "ctx-danger" : ""} ${item.active ? "ctx-active" : ""}`}
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
