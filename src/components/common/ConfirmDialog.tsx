import { useEffect, useRef, useState } from "react";
import { useAppStore } from "../../store/app";
import "./ConfirmDialog.css";

/**
 * App-styled modal confirm dialog — replaces the native `ask()` popup (which looked out of place). Driven
 * by the app-store `confirm` request (set via `showConfirm(...)`, which resolves with the chosen button
 * id, or "" on Esc/backdrop dismiss). The keyboard is OWNED while open (capture + stopPropagation) so
 * background shortcuts (Ctrl+S/Z/…) don't fire underneath. Enter triggers the `primary` button if there
 * is one (never a `danger` button — destructive actions require an explicit click).
 *
 * TEXT-INPUT mode (`confirm.input` set, e.g. the "new group" prompt): an input renders between body and
 * buttons; the primary button / Enter resolves with the TRIMMED VALUE instead of the button id, blocked
 * (with an inline error) while empty or `input.invalid(value)` returns a message. Typing still works
 * despite the capture-phase stopPropagation — text insertion is a browser DEFAULT ACTION (only
 * preventDefault would block it), and the controlled value updates via the `input` event.
 */
export function ConfirmDialog() {
  const confirm = useAppStore((s) => s.confirm);
  const [value, setValue] = useState("");
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Fresh input state per dialog (seq bumps on every showConfirm).
  useEffect(() => {
    setValue(confirm?.input?.initial ?? "");
    setError(null);
  }, [confirm?.seq, confirm?.input?.initial]);

  // Attempt to commit the input value via the primary action; returns the resolution or null if blocked.
  const commitInput = () => {
    if (!confirm?.input) return null;
    const v = value.trim();
    const err = v === "" ? "" : confirm.input.invalid?.(v) ?? null; // "" = silently blocked (empty)
    if (v === "" || err !== null) {
      setError(err || null);
      inputRef.current?.focus();
      return null;
    }
    return v;
  };
  const commitRef = useRef(commitInput);
  commitRef.current = commitInput;

  useEffect(() => {
    if (!confirm) return;
    const onKey = (e: KeyboardEvent) => {
      e.stopPropagation(); // the dialog owns the keyboard while open
      if (e.key === "Escape") {
        e.preventDefault();
        confirm.resolve("");
      } else if (e.key === "Enter") {
        e.preventDefault();
        const primary = confirm.buttons.find((b) => b.kind === "primary");
        if (!primary) return;
        if (confirm.input) {
          const v = commitRef.current();
          if (v !== null) confirm.resolve(v);
        } else {
          confirm.resolve(primary.id);
        }
      }
    };
    window.addEventListener("keydown", onKey, true); // capture: intercept before App's global handlers
    return () => window.removeEventListener("keydown", onKey, true);
  }, [confirm]);

  if (!confirm) return null;
  return (
    <div className="confirm-overlay" onMouseDown={() => confirm.resolve("")}>
      <div
        className="confirm-dialog"
        key={confirm.seq}
        role="dialog"
        aria-modal="true"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="confirm-title">{confirm.title}</div>
        {confirm.body && <div className="confirm-body">{confirm.body}</div>}
        {confirm.input && (
          <div className="confirm-input-row">
            <input
              ref={inputRef}
              className="confirm-input"
              type="text"
              autoFocus
              value={value}
              placeholder={confirm.input.placeholder}
              onChange={(e) => {
                setValue(e.target.value);
                setError(null);
              }}
            />
            {error && <div className="confirm-input-error">{error}</div>}
          </div>
        )}
        <div className="confirm-buttons">
          {confirm.buttons.map((b) => (
            <button
              key={b.id}
              className={`confirm-btn ${b.kind ?? "neutral"}`}
              onClick={() => {
                if (confirm.input && b.kind === "primary") {
                  const v = commitRef.current();
                  if (v !== null) confirm.resolve(v);
                } else if (confirm.input && b.kind !== "danger") {
                  confirm.resolve(""); // input mode: non-primary neutral buttons read as cancel
                } else {
                  confirm.resolve(b.id);
                }
              }}
            >
              {b.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
