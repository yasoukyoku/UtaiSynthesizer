/**
 * Custom themed dropdown replacing native <select> (house style: square corners,
 * theme tokens, self-drawn arrow + options panel — the WebView2 native popup
 * ignores the app theme entirely). Keyboard: Enter/Space/ArrowDown open,
 * arrows move, Enter selects, Esc closes; click-outside closes.
 */
import { useEffect, useRef, useState } from "react";
import "./Dropdown.css";

export interface DropdownOption<T extends string | number> {
  value: T;
  label: string;
  /** S75: shown but not choosable — the entry must stay VISIBLE (a user who knows they own that
   *  device must not think we lost it) while "you can select it" keeps implying "it works". */
  disabled?: boolean;
  /** Native tooltip. Option rows are nowrap + ellipsis and the panel is only as wide as the
   *  trigger, so a label carrying a REASON gets truncated — and reasons put the actionable half
   *  ("install it under Settings → …") at the end, which is exactly what gets cut (S75 review). */
  title?: string;
}

export function Dropdown<T extends string | number>({
  value,
  options,
  onChange,
  className,
}: {
  value: T;
  options: DropdownOption<T>[];
  onChange: (v: T) => void;
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  const [hover, setHover] = useState<number>(-1);
  const rootRef = useRef<HTMLDivElement>(null);

  const selectedIdx = options.findIndex((o) => o.value === value);
  const selected = options[selectedIdx];

  useEffect(() => {
    if (!open) return;
    const onDocDown = (e: PointerEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
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

  const commit = (idx: number) => {
    const opt = options[idx];
    if (opt?.disabled) return; // stays open — a dead click must not read as "accepted"
    if (opt) onChange(opt.value);
    setOpen(false);
  };

  /** Arrow keys SKIP disabled entries — landing on one and pressing Enter would read as broken. */
  const moveHover = (step: 1 | -1) =>
    setHover((h) => {
      for (let i = h + step; i >= 0 && i < options.length; i += step) {
        if (!options[i]?.disabled) return i;
      }
      return h;
    });

  const onKeyDown = (e: React.KeyboardEvent) => {
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
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      moveHover(1);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      moveHover(-1);
    } else if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      commit(hover);
    } else if (e.key === "Tab") {
      setOpen(false);
    }
  };

  return (
    <div className={`ut-dropdown ${className ?? ""}`} ref={rootRef}>
      <button
        type="button"
        className={`ut-dropdown-trigger ${open ? "open" : ""}`}
        onClick={() => (open ? setOpen(false) : openPanel())}
        onKeyDown={onKeyDown}
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <span className="ut-dropdown-label">{selected?.label ?? ""}</span>
        <span className="ut-dropdown-arrow">{open ? "▴" : "▾"}</span>
      </button>
      {open && (
        <div className="ut-dropdown-panel" role="listbox">
          {options.map((o, i) => (
            <div
              key={String(o.value)}
              role="option"
              aria-selected={i === selectedIdx}
              aria-disabled={o.disabled || undefined}
              title={o.title ?? o.label}
              className={`ut-dropdown-option ${i === selectedIdx ? "selected" : ""} ${
                i === hover ? "hover" : ""
              } ${o.disabled ? "disabled" : ""}`}
              onPointerEnter={() => !o.disabled && setHover(i)}
              onPointerDown={(e) => {
                // pointerdown (not click): commit before the outside-click
                // closer sees the event, and before focus shifts
                e.preventDefault();
                commit(i);
              }}
            >
              {o.label}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
