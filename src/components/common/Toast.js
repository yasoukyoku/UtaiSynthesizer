import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useAppStore } from "../../store/app";
import "./Toast.css";
export function ToastContainer() {
    const toasts = useAppStore((s) => s.toasts);
    const dismiss = useAppStore((s) => s.dismissToast);
    if (toasts.length === 0)
        return null;
    return (_jsx("div", { className: "toast-container", children: toasts.map((t) => (_jsxs("div", { className: `toast toast-${t.type}`, onClick: () => dismiss(t.id), children: [_jsx("span", { className: "toast-icon", children: t.type === "error" ? "!" : t.type === "success" ? "+" : "i" }), _jsx("span", { className: "toast-msg", children: t.message }), _jsx("button", { className: "toast-close", onClick: () => dismiss(t.id), children: "x" })] }, t.id))) }));
}
