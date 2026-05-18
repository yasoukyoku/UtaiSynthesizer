import { useTranslation } from "react-i18next";
import "./NodePalette.css";

interface Props {
  onAddNode: (type: string, label: string) => void;
}

const nodeCategories = [
  {
    category: "workflow.catVoice",
    nodes: [
      { type: "rvc", label: "RVC", icon: "🎤", color: "#39c5bb" },
      { type: "sovits", label: "SoVITS", icon: "✨", color: "#8b5cf6" },
    ],
  },
  {
    category: "workflow.catEffects",
    nodes: [
      { type: "pitchShift", label: "Pitch Shift", icon: "↕", color: "#fbbf24" },
      { type: "formantShift", label: "Formant Shift", icon: "🔄", color: "#f97316" },
      { type: "audioEnhance", label: "Audio Enhance", icon: "💎", color: "#a78bfa" },
    ],
  },
  {
    category: "workflow.catSeparation",
    nodes: [
      { type: "msst", label: "MSST Separate", icon: "🔀", color: "#ec4899" },
    ],
  },
  {
    category: "workflow.catIO",
    nodes: [
      { type: "audioOutput", label: "Output", icon: "🔊", color: "#4ade80" },
    ],
  },
];

export function NodePalette({ onAddNode }: Props) {
  const { t } = useTranslation();

  return (
    <aside className="node-palette">
      <div className="palette-title">{t("workflow.nodes")}</div>
      {nodeCategories.map((cat) => (
        <div key={cat.category} className="palette-category">
          <div className="palette-category-title">{t(cat.category)}</div>
          {cat.nodes.map((node) => (
            <button
              key={node.type}
              className="palette-node"
              onClick={() => onAddNode(node.type, node.label)}
              style={{ "--node-color": node.color } as React.CSSProperties}
            >
              <span className="palette-node-icon">{node.icon}</span>
              <span className="palette-node-label">{node.label}</span>
            </button>
          ))}
        </div>
      ))}
    </aside>
  );
}
