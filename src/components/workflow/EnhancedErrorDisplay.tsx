import { useState } from "react";
import { useTranslation } from "react-i18next";
import "./EnhancedErrorDisplay.css";

interface ErrorSuggestion {
  title: string;
  description: string;
  action?: {
    label: string;
    onClick: () => void;
  };
}

interface Props {
  error: string;
  nodeType: string;
  onDismiss: () => void;
}

export function EnhancedErrorDisplay({ error, nodeType, onDismiss }: Props) {
  const { t } = useTranslation();
  const [expanded, setExpanded] = useState(false);
  
  const suggestions = generateSuggestions(error, nodeType, t);
  const severity = classifyErrorSeverity(error);

  return (
    <div className={`enhanced-error-display severity-${severity}`}>
      <div className="enhanced-error-header">
        <span className="enhanced-error-icon">{getSeverityIcon(severity)}</span>
        <div className="enhanced-error-title-section">
          <h4>{getSeverityLabel(severity, t)}</h4>
          <p className="enhanced-error-summary">{getErrorSummary(error)}</p>
        </div>
        <button className="enhanced-error-close" onClick={onDismiss}>✕</button>
      </div>

      {suggestions.length > 0 && (
        <div className="enhanced-error-suggestions">
          <div className="enhanced-error-suggestions-header">
            <span className="enhanced-error-suggestions-icon">💡</span>
            <span>{t("workflow.error.suggestionsTitle")}</span>
          </div>
          {suggestions.map((suggestion, index) => (
            <div key={index} className="enhanced-error-suggestion-item">
              <div className="enhanced-error-suggestion-content">
                <strong>{suggestion.title}</strong>
                <p>{suggestion.description}</p>
              </div>
              {suggestion.action && (
                <button 
                  className="enhanced-error-suggestion-action"
                  onClick={suggestion.action.onClick}
                >
                  {suggestion.action.label}
                </button>
              )}
            </div>
          ))}
        </div>
      )}

      <div className="enhanced-error-details">
        <button 
          className="enhanced-error-details-toggle"
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? "▼" : "▶"} {t("workflow.error.technicalDetails")}
        </button>
        {expanded && (
          <pre className="enhanced-error-details-content">{error}</pre>
        )}
      </div>
    </div>
  );
}

function getSeverityIcon(severity: "error" | "warning" | "info"): string {
  switch (severity) {
    case "error": return "❌";
    case "warning": return "⚠️";
    case "info": return "ℹ️";
  }
}

function getSeverityLabel(severity: "error" | "warning" | "info", t: any): string {
  switch (severity) {
    case "error": return t("workflow.error.severityError");
    case "warning": return t("workflow.error.severityWarning");
    case "info": return t("workflow.error.severityInfo");
  }
}

function classifyErrorSeverity(error: string): "error" | "warning" | "info" {
  const lowerError = error.toLowerCase();
  if (lowerError.includes("warning") || lowerError.includes("degraded")) {
    return "warning";
  }
  if (lowerError.includes("info") || lowerError.includes("notice")) {
    return "info";
  }
  return "error";
}

function getErrorSummary(error: string): string {
  const lines = error.split("\n");
  const firstLine = lines[0] || "";
  return firstLine.substring(0, 100) + (firstLine.length > 100 ? "..." : "");
}

function generateSuggestions(error: string, nodeType: string, t: any): ErrorSuggestion[] {
  const suggestions: ErrorSuggestion[] = [];
  const lowerError = error.toLowerCase();

  if (lowerError.includes("model") && lowerError.includes("not found")) {
    suggestions.push({
      title: t("workflow.error.suggestion.modelNotFound.title"),
      description: t("workflow.error.suggestion.modelNotFound.description"),
      action: {
        label: t("workflow.error.suggestion.modelNotFound.action"),
        onClick: () => {
          console.log("Navigate to model settings");
        },
      },
    });
  }

  if (lowerError.includes("gpu") || lowerError.includes("cuda") || lowerError.includes("out of memory")) {
    suggestions.push({
      title: t("workflow.error.suggestion.gpuMemory.title"),
      description: t("workflow.error.suggestion.gpuMemory.description"),
    });
  }

  if (lowerError.includes("file") && (lowerError.includes("not found") || lowerError.includes("missing"))) {
    suggestions.push({
      title: t("workflow.error.suggestion.fileNotFound.title"),
      description: t("workflow.error.suggestion.fileNotFound.description"),
    });
  }

  if (lowerError.includes("connection") || lowerError.includes("network") || lowerError.includes("timeout")) {
    suggestions.push({
      title: t("workflow.error.suggestion.connection.title"),
      description: t("workflow.error.suggestion.connection.description"),
    });
  }

  if (nodeType === "rvc" || nodeType === "sovits") {
    if (lowerError.includes("index")) {
      suggestions.push({
        title: t("workflow.error.suggestion.voiceIndex.title"),
        description: t("workflow.error.suggestion.voiceIndex.description"),
      });
    }
  }

  if (suggestions.length === 0) {
    suggestions.push({
      title: t("workflow.error.suggestion.generic.title"),
      description: t("workflow.error.suggestion.generic.description"),
    });
  }

  return suggestions;
}
