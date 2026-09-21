import { useEffect, useRef, useState } from "react";
import "./ContextMenu.css";

export type MenuItem =
  | { type?: "item"; label: string; icon?: string; shortcut?: string; danger?: boolean;
      disabled?: boolean; active?: boolean; title?: string; onClick: () => void }
  | { type: "divider" }
  | { type: "submenu"; label: string; icon?: string; disabled?: boolean; danger?: boolean;
      title?: string; items: MenuItem[] };   // 二级子菜单（悬停展开）

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
      {items.map((item, i) =>
        "type" in item && item.type === "divider" ? (
          <div key={i} className="ctx-divider" role="separator" />
        ) : "type" in item && item.type === "submenu" ? (
          <SubMenuItem key={i} item={item} onClose={onClose} />
        ) : (
          <button
            key={i}
            className={`ctx-item ${item.danger ? "ctx-danger" : ""} ${item.active ? "ctx-active" : ""}`}
            disabled={item.disabled}
            title={item.title}
            onClick={() => {
              item.onClick();
              onClose();
            }}
          >
            <span className="ctx-label">{item.icon ? <span className="ctx-icon" aria-hidden="true">{item.icon}</span> : null}{item.label}</span>
            {item.shortcut && <span className="ctx-shortcut mono">{item.shortcut}</span>}
          </button>
        )
      )}
    </div>
  );
}

// ── 二级子菜单 ─────────────────────────────────────────────────────────────
// 悬停展开；右侧无空间时翻到左侧；面板超高内部滚动（复用 .ctx-menu 滚动样式）。
// 延迟关闭：离开触发项/面板有 150ms 宽限，穿过缝隙不闪断。

function SubMenuItem({ item, onClose }: { item: Extract<MenuItem, { type: "submenu" }>; onClose: () => void }) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [flip, setFlip] = useState(false);
  const closeTimer = useRef<number | null>(null);

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
    if (closeTimer.current !== null) window.clearTimeout(closeTimer.current);
    closeTimer.current = window.setTimeout(() => setOpen(false), 150);
  };
  useEffect(() => () => {
    if (closeTimer.current !== null) window.clearTimeout(closeTimer.current);
  }, []);

  return (
    <div
      ref={wrapRef}
      className="ctx-submenu-wrap"
      onMouseEnter={openNow}
      onMouseLeave={closeSoon}
    >
      <button
        className={`ctx-item ctx-submenu ${item.danger ? "ctx-danger" : ""} ${open ? "ctx-open" : ""}`}
        disabled={item.disabled}
        title={item.title}
        onClick={() => (open ? setOpen(false) : openNow())}
      >
        <span className="ctx-label">{item.icon ? <span className="ctx-icon" aria-hidden="true">{item.icon}</span> : null}{item.label}</span>
        <span className="ctx-arrow" aria-hidden="true">▸</span>
      </button>
      {open && !item.disabled && (
        <div className={`ctx-menu ctx-submenu-panel ${flip ? "ctx-flip-left" : ""}`}>
          {item.items.map((sub, i) =>
            "type" in sub && sub.type === "divider" ? (
              <div key={i} className="ctx-divider" role="separator" />
            ) : "type" in sub && sub.type === "submenu" ? (
              <SubMenuItem key={i} item={sub} onClose={onClose} />
            ) : (
              <button
                key={i}
                className={`ctx-item ${sub.danger ? "ctx-danger" : ""} ${sub.active ? "ctx-active" : ""}`}
                disabled={sub.disabled}
                title={sub.title}
                onClick={() => {
                  sub.onClick();
                  onClose();
                }}
              >
                <span className="ctx-label">{sub.icon ? <span className="ctx-icon" aria-hidden="true">{sub.icon}</span> : null}{sub.label}</span>
                {sub.shortcut && <span className="ctx-shortcut mono">{sub.shortcut}</span>}
              </button>
            )
          )}
        </div>
      )}
    </div>
  );
}
