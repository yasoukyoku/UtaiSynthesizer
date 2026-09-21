import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Component } from "react";
import i18n from "../../i18n";
import "./ConfirmDialog.css";
export class ErrorBoundary extends Component {
    constructor(props) {
        super(props);
        Object.defineProperty(this, "reset", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: () => {
                this.setState({ error: null, errorInfo: null });
            }
        });
        this.state = { error: null, errorInfo: null };
    }
    static getDerivedStateFromError(error) {
        return { error };
    }
    componentDidCatch(error, errorInfo) {
        console.error("ErrorBoundary 捕获到渲染错误:", error, errorInfo);
        this.setState({ errorInfo });
    }
    render() {
        const { error, errorInfo } = this.state;
        const { children, fallback } = this.props;
        if (error) {
            if (fallback)
                return fallback(error, this.reset);
            const stack = errorInfo?.componentStack || error.stack || "";
            const showDetails = stack.length > 0;
            return (_jsx("div", { className: "confirm-overlay", children: _jsxs("div", { className: "confirm-dialog", role: "alert", "aria-live": "assertive", children: [_jsx("div", { className: "confirm-title", children: i18n.t("common.errorBoundaryTitle") }), _jsx("div", { className: "confirm-body", children: i18n.t("common.errorBoundaryBody") }), showDetails && (_jsxs("details", { style: { marginBottom: "16px" }, children: [_jsx("summary", { style: {
                                        fontFamily: "var(--font-mono)",
                                        fontSize: "var(--font-size-sm)",
                                        color: "var(--text-secondary)",
                                        cursor: "pointer",
                                        marginBottom: "8px",
                                    }, children: i18n.t("common.errorBoundaryDetails") }), _jsxs("pre", { style: {
                                        fontFamily: "var(--font-mono)",
                                        fontSize: "var(--font-size-xs)",
                                        color: "var(--text-tertiary)",
                                        background: "var(--bg-base)",
                                        padding: "10px",
                                        maxHeight: "200px",
                                        overflowY: "auto",
                                        whiteSpace: "pre-wrap",
                                        wordBreak: "break-word",
                                    }, children: [error.message, "\n\n", stack] })] })), _jsx("div", { className: "confirm-buttons", children: _jsx("button", { className: "confirm-btn primary", onClick: this.reset, children: i18n.t("common.errorBoundaryReload") }) })] }) }));
        }
        return children;
    }
}
