import { Component, ReactNode, ErrorInfo } from "react";
import i18n from "../../i18n";
import "./ConfirmDialog.css";

interface Props {
  children: ReactNode;
  fallback?: (error: Error, reset: () => void) => ReactNode;
}

interface State {
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("ErrorBoundary 捕获到渲染错误:", error, errorInfo);
    this.setState({ errorInfo });
  }

  reset = () => {
    this.setState({ error: null, errorInfo: null });
  };

  render() {
    const { error, errorInfo } = this.state;
    const { children, fallback } = this.props;

    if (error) {
      if (fallback) return fallback(error, this.reset);

      const stack = errorInfo?.componentStack || error.stack || "";
      const showDetails = stack.length > 0;

      return (
        <div className="confirm-overlay">
          <div className="confirm-dialog" role="alert" aria-live="assertive">
            <div className="confirm-title">{i18n.t("common.errorBoundaryTitle")}</div>
            <div className="confirm-body">{i18n.t("common.errorBoundaryBody")}</div>
            {showDetails && (
              <details style={{ marginBottom: "16px" }}>
                <summary
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: "var(--font-size-sm)",
                    color: "var(--text-secondary)",
                    cursor: "pointer",
                    marginBottom: "8px",
                  }}
                >
                  {i18n.t("common.errorBoundaryDetails")}
                </summary>
                <pre
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: "var(--font-size-xs)",
                    color: "var(--text-tertiary)",
                    background: "var(--bg-base)",
                    padding: "10px",
                    maxHeight: "200px",
                    overflowY: "auto",
                    whiteSpace: "pre-wrap",
                    wordBreak: "break-word",
                  }}
                >
                  {error.message}
                  {"\n\n"}
                  {stack}
                </pre>
              </details>
            )}
            <div className="confirm-buttons">
              <button className="confirm-btn primary" onClick={this.reset}>
                {i18n.t("common.errorBoundaryReload")}
              </button>
            </div>
          </div>
        </div>
      );
    }

    return children;
  }
}
