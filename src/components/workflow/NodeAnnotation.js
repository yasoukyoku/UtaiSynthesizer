import { jsxs as _jsxs, jsx as _jsx } from "react/jsx-runtime";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import "./NodeAnnotation.css";
export function NodeAnnotationEditor({ annotation, onSave, onDelete, onClose }) {
    const { t } = useTranslation();
    const [content, setContent] = useState(annotation.content);
    const [color, setColor] = useState(annotation.color || "yellow");
    const colors = [
        { value: "yellow", label: t("workflow.annotation.colorYellow"), color: "#FCD34D" },
        { value: "blue", label: t("workflow.annotation.colorBlue"), color: "#60A5FA" },
        { value: "green", label: t("workflow.annotation.colorGreen"), color: "#4ADE80" },
        { value: "red", label: t("workflow.annotation.colorRed"), color: "#F87171" },
        { value: "purple", label: t("workflow.annotation.colorPurple"), color: "#C084FC" },
    ];
    const handleSave = () => {
        if (content.trim()) {
            onSave(content, color);
            onClose();
        }
    };
    return (_jsx("div", { className: "annotation-editor-overlay", onClick: onClose, children: _jsxs("div", { className: "annotation-editor", onClick: (e) => e.stopPropagation(), children: [_jsxs("div", { className: "annotation-editor-header", children: [_jsxs("h3", { children: ["\uD83D\uDCDD ", t("workflow.annotation.title")] }), _jsx("button", { className: "annotation-editor-close", onClick: onClose, children: "\u2715" })] }), _jsx("textarea", { className: "annotation-editor-textarea", value: content, onChange: (e) => setContent(e.target.value), placeholder: t("workflow.annotation.placeholder"), autoFocus: true }), _jsxs("div", { className: "annotation-editor-colors", children: [_jsxs("label", { children: [t("workflow.annotation.colorLabel"), ":"] }), _jsx("div", { className: "annotation-color-picker", children: colors.map((c) => (_jsx("button", { className: `annotation-color-btn ${color === c.value ? "selected" : ""}`, style: { backgroundColor: c.color }, onClick: () => setColor(c.value), title: c.label }, c.value))) })] }), _jsxs("div", { className: "annotation-editor-footer", children: [_jsx("button", { className: "annotation-delete-btn", onClick: onDelete, children: t("workflow.annotation.deleteButton") }), _jsxs("div", { className: "annotation-editor-actions", children: [_jsx("button", { className: "annotation-cancel-btn", onClick: onClose, children: t("workflow.annotation.cancelButton") }), _jsx("button", { className: "annotation-save-btn", onClick: handleSave, disabled: !content.trim(), children: t("workflow.annotation.saveButton") })] })] })] }) }));
}
export function NodeAnnotationBadge({ annotation, onClick }) {
    const colorMap = {
        yellow: "#FCD34D",
        blue: "#60A5FA",
        green: "#4ADE80",
        red: "#F87171",
        purple: "#C084FC",
    };
    return (_jsx("div", { className: "node-annotation-badge", style: { backgroundColor: colorMap[annotation.color || "yellow"] }, onClick: onClick, title: annotation.content, children: "\uD83D\uDCDD" }));
}
