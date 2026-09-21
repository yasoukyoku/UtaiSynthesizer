import { useEffect, useRef, useState } from "react";
import "./FaderValue.css";

interface Props {
  value: number;
  min: number;
  max: number;
  /** 显示文本格式化(如 "+6.0 dB" 或 "L50");编辑态输入框内则是原始数值字符串。 */
  format: (v: number) => string;
  /** 提交新数值(已 clamp 到 [min,max])。 */
  onCommit: (v: number) => void;
  onGestureStart?: () => void;
  onGestureEnd?: () => void;
}

/** 单击即可编辑的数值显示框:显示态是一个按钮(展示 fader 数值),点击后变成输入框,
 *  可输入整数或小数;回车/失焦提交, Esc 取消。提交会通过 onCommit 联动拖动条。 */
export function FaderValue({ value, min, max, format, onCommit, onGestureStart, onGestureEnd }: Props) {
  const [editing, setEditing] = useState(false);
  const [text, setText] = useState(String(value));
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editing) {
      setText(String(value));
      requestAnimationFrame(() => inputRef.current?.select());
    }
  }, [editing, value]);

  const startEdit = () => {
    setEditing(true);
    onGestureStart?.();
  };

  const end = (commit: boolean) => {
    if (commit) {
      const n = parseFloat(text);
      if (Number.isFinite(n)) onCommit(Math.max(min, Math.min(max, n)));
    }
    setEditing(false);
    onGestureEnd?.();
  };

  if (editing) {
    return (
      <input
        ref={inputRef}
        className="fader-val-input"
        type="number"
        step="any"
        value={text}
        autoFocus
        onChange={(e) => setText(e.target.value)}
        onBlur={() => end(true)}
        onKeyDown={(e) => {
          e.stopPropagation();
          if (e.key === "Enter") end(true);
          else if (e.key === "Escape") end(false);
        }}
        onClick={(e) => e.stopPropagation()}
      />
    );
  }

  return (
    <button
      className="fader-val"
      title="点击输入数值"
      onClick={(e) => {
        e.stopPropagation();
        startEdit();
      }}
    >
      {format(value)}
    </button>
  );
}