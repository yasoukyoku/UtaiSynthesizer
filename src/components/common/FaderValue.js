import { jsx as _jsx } from "react/jsx-runtime";
import { useEffect, useRef, useState } from "react";
import "./FaderValue.css";
/** 单击即可编辑的数值显示框:显示态是一个按钮(展示 fader 数值),点击后变成输入框,
 *  可输入整数或小数;回车/失焦提交, Esc 取消。提交会通过 onCommit 联动拖动条。 */
export function FaderValue({ value, min, max, format, onCommit, onGestureStart, onGestureEnd }) {
    const [editing, setEditing] = useState(false);
    const [text, setText] = useState(String(value));
    const inputRef = useRef(null);
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
    const end = (commit) => {
        if (commit) {
            const n = parseFloat(text);
            if (Number.isFinite(n))
                onCommit(Math.max(min, Math.min(max, n)));
        }
        setEditing(false);
        onGestureEnd?.();
    };
    if (editing) {
        return (_jsx("input", { ref: inputRef, className: "fader-val-input", type: "number", step: "any", value: text, autoFocus: true, onChange: (e) => setText(e.target.value), onBlur: () => end(true), onKeyDown: (e) => {
                e.stopPropagation();
                if (e.key === "Enter")
                    end(true);
                else if (e.key === "Escape")
                    end(false);
            }, onClick: (e) => e.stopPropagation() }));
    }
    return (_jsx("button", { className: "fader-val", title: "\u70B9\u51FB\u8F93\u5165\u6570\u503C", onClick: (e) => {
            e.stopPropagation();
            startEdit();
        }, children: format(value) }));
}
