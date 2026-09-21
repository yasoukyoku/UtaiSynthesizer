import { useState } from "react";
import { useTranslation } from "react-i18next";
import type { NodeAnnotation as NodeAnnotationType } from "../../lib/workflow/nodeAnnotations";
import "./NodeAnnotation.css";

interface Props {
  annotation: NodeAnnotationType;
  onSave: (content: string, color?: NodeAnnotationType["color"]) => void;
  onDelete: () => void;
  onClose: () => void;
}

export function NodeAnnotationEditor({ annotation, onSave, onDelete, onClose }: Props) {
  const { t } = useTranslation();
  const [content, setContent] = useState(annotation.content);
  const [color, setColor] = useState<NodeAnnotationType["color"]>(annotation.color || "yellow");

  const colors: Array<{ value: NodeAnnotationType["color"]; label: string; color: string }> = [
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

  return (
    <div className="annotation-editor-overlay" onClick={onClose}>
      <div className="annotation-editor" onClick={(e) => e.stopPropagation()}>
        <div className="annotation-editor-header">
          <h3>📝 {t("workflow.annotation.title")}</h3>
          <button className="annotation-editor-close" onClick={onClose}>✕</button>
        </div>

        <textarea
          className="annotation-editor-textarea"
          value={content}
          onChange={(e) => setContent(e.target.value)}
          placeholder={t("workflow.annotation.placeholder")}
          autoFocus
        />

        <div className="annotation-editor-colors">
          <label>{t("workflow.annotation.colorLabel")}:</label>
          <div className="annotation-color-picker">
            {colors.map((c) => (
              <button
                key={c.value}
                className={`annotation-color-btn ${color === c.value ? "selected" : ""}`}
                style={{ backgroundColor: c.color }}
                onClick={() => setColor(c.value)}
                title={c.label}
              />
            ))}
          </div>
        </div>

        <div className="annotation-editor-footer">
          <button className="annotation-delete-btn" onClick={onDelete}>
            {t("workflow.annotation.deleteButton")}
          </button>
          <div className="annotation-editor-actions">
            <button className="annotation-cancel-btn" onClick={onClose}>
              {t("workflow.annotation.cancelButton")}
            </button>
            <button 
              className="annotation-save-btn" 
              onClick={handleSave}
              disabled={!content.trim()}
            >
              {t("workflow.annotation.saveButton")}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

interface BadgeProps {
  annotation: NodeAnnotationType;
  onClick: () => void;
}

export function NodeAnnotationBadge({ annotation, onClick }: BadgeProps) {
  const colorMap: Record<string, string> = {
    yellow: "#FCD34D",
    blue: "#60A5FA",
    green: "#4ADE80",
    red: "#F87171",
    purple: "#C084FC",
  };

  return (
    <div 
      className="node-annotation-badge" 
      style={{ backgroundColor: colorMap[annotation.color || "yellow"] }}
      onClick={onClick}
      title={annotation.content}
    >
      📝
    </div>
  );
}
