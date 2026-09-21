import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import "./EnhancedErrorDisplay.css";
export function EnhancedErrorDisplay({ error, nodeType, onDismiss }) {
    const { t } = useTranslation();
    const [expanded, setExpanded] = useState(false);
    const suggestions = generateSuggestions(error, nodeType, t);
    const severity = classifyErrorSeverity(error);
    return (_jsxs("div", { className: `enhanced-error-display severity-${severity}`, children: [_jsxs("div", { className: "enhanced-error-header", children: [_jsx("span", { className: "enhanced-error-icon", children: getSeverityIcon(severity) }), _jsxs("div", { className: "enhanced-error-title-section", children: [_jsx("h4", { children: getSeverityLabel(severity, t) }), _jsx("p", { className: "enhanced-error-summary", children: getErrorSummary(error) })] }), _jsx("button", { className: "enhanced-error-close", onClick: onDismiss, children: "\u2715" })] }), suggestions.length > 0 && (_jsxs("div", { className: "enhanced-error-suggestions", children: [_jsxs("div", { className: "enhanced-error-suggestions-header", children: [_jsx("span", { className: "enhanced-error-suggestions-icon", children: "\uD83D\uDCA1" }), _jsx("span", { children: t("workflow.error.suggestionsTitle") })] }), suggestions.map((suggestion, index) => (_jsxs("div", { className: "enhanced-error-suggestion-item", children: [_jsxs("div", { className: "enhanced-error-suggestion-content", children: [_jsx("strong", { children: suggestion.title }), _jsx("p", { children: suggestion.description })] }), suggestion.action && (_jsx("button", { className: "enhanced-error-suggestion-action", onClick: suggestion.action.onClick, children: suggestion.action.label }))] }, index)))] })), _jsxs("div", { className: "enhanced-error-details", children: [_jsxs("button", { className: "enhanced-error-details-toggle", onClick: () => setExpanded(!expanded), children: [expanded ? "▼" : "▶", " ", t("workflow.error.technicalDetails")] }), expanded && (_jsx("pre", { className: "enhanced-error-details-content", children: error }))] })] }));
}
function getSeverityIcon(severity) {
    switch (severity) {
        case "error": return "❌";
        case "warning": return "⚠️";
        case "info": return "ℹ️";
    }
}
function getSeverityLabel(severity, t) {
    switch (severity) {
        case "error": return t("workflow.error.severityError");
        case "warning": return t("workflow.error.severityWarning");
        case "info": return t("workflow.error.severityInfo");
    }
}
function classifyErrorSeverity(error) {
    const lowerError = error.toLowerCase();
    if (lowerError.includes("warning") || lowerError.includes("degraded")) {
        return "warning";
    }
    if (lowerError.includes("info") || lowerError.includes("notice")) {
        return "info";
    }
    return "error";
}
function getErrorSummary(error) {
    const lines = error.split("\n");
    const firstLine = lines[0] || "";
    return firstLine.substring(0, 100) + (firstLine.length > 100 ? "..." : "");
}
function generateSuggestions(error, nodeType, t) {
    const suggestions = [];
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
