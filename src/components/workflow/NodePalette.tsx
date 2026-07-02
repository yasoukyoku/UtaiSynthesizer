import { useCallback, useRef, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { useMsstModelStore } from "../../store/msst-models";
import { useAppStore } from "../../store/app";
import { MSST_CATALOG, ALL_CATEGORIES, CATEGORY_LABELS, CATEGORY_COLORS, t18 } from "../../lib/models/msst-catalog";
import { OUTPUT_NODE_COLOR } from "../../lib/constants";
import "./NodePalette.css";

interface Props {
  onAddNode: (type: string, label: string, extraParams?: Record<string, unknown>) => void;
  onDropNode?: (type: string, label: string, clientX: number, clientY: number, extraParams?: Record<string, unknown>) => void;
}

export function NodePalette({ onAddNode, onDropNode }: Props) {
  const { t, i18n } = useTranslation();
  const lang = i18n.language;
  const installed = useMsstModelStore((s) => s.installed);
  const installedFiles = new Set(installed.map((m) => m.filename));
  const toggleModelManager = useAppStore((s) => s.toggleModelManager);

  const availableCategories = ALL_CATEGORIES.filter((cat) =>
    MSST_CATALOG.some((e) => e.category === cat && installedFiles.has(e.filename)),
  );

  const dragRef = useRef<{ type: string; label: string; extraParams?: Record<string, unknown> } | null>(null);
  const ghostRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!dragRef.current || !ghostRef.current) return;
      ghostRef.current.style.left = `${e.clientX - 40}px`;
      ghostRef.current.style.top = `${e.clientY - 12}px`;
    };
    const onUp = (e: MouseEvent) => {
      if (!dragRef.current) return;
      const { type, label, extraParams } = dragRef.current;
      dragRef.current = null;
      if (ghostRef.current) {
        ghostRef.current.remove();
        ghostRef.current = null;
      }
      document.body.style.cursor = "";
      if (onDropNode) {
        onDropNode(type, label, e.clientX, e.clientY, extraParams);
      }
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    return () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
  }, [onDropNode]);

  const startDrag = useCallback((e: React.MouseEvent, type: string, label: string, extraParams?: Record<string, unknown>) => {
    e.preventDefault();
    dragRef.current = { type, label, extraParams };
    document.body.style.cursor = "grabbing";
    const ghost = document.createElement("div");
    ghost.className = "palette-drag-ghost";
    ghost.textContent = label;
    ghost.style.left = `${e.clientX - 40}px`;
    ghost.style.top = `${e.clientY - 12}px`;
    document.body.appendChild(ghost);
    ghostRef.current = ghost;
  }, []);

  return (
    <aside className="node-palette">
      <div className="palette-title">{t("workflow.nodes")}</div>

      <div className="palette-category">
        <div className="palette-category-title">{t("workflow.catVoice")}</div>
        <PaletteItem type="rvc" label="RVC" icon="R" color="#39c5bb" onAdd={onAddNode} onDrag={startDrag} />
        <PaletteItem type="sovits" label="SoVITS" icon="S" color="#8b5cf6" onAdd={onAddNode} onDrag={startDrag} />
      </div>

      <div className="palette-category">
        <div className="palette-category-title">{t("workflow.catEffects")}</div>
        <PaletteItem type="effects" label={t18({ zh: "效果器", en: "Effects", ja: "エフェクト" }, lang)} icon="FX" color="#fbbf24" onAdd={onAddNode} onDrag={startDrag} />
      </div>

      <div className="palette-category">
        <div className="palette-category-title">{t("workflow.catSeparation")}</div>
        {availableCategories.map((cat) => (
          <PaletteItem
            key={cat}
            type="separation"
            label={t18(CATEGORY_LABELS[cat], lang)}
            icon={t18(CATEGORY_LABELS[cat], lang).charAt(0)}
            color={CATEGORY_COLORS[cat]}
            extraParams={{ category: cat }}
            onAdd={onAddNode}
            onDrag={startDrag}
          />
        ))}
        {availableCategories.length === 0 && (
          <span className="palette-empty">
            {t18({ zh: "未安装分离模型", en: "No separation models", ja: "分離モデル未インストール" }, lang)}
          </span>
        )}
        <button className="palette-manage-btn" onClick={toggleModelManager}>
          {t18({ zh: "管理模型...", en: "Manage models...", ja: "モデル管理..." }, lang)}
        </button>
      </div>

      <div className="palette-category">
        <div className="palette-category-title">{t("workflow.catIO")}</div>
        <PaletteItem type="audioOutput" label={t("workflow.nodeOutput")} icon=">" color={OUTPUT_NODE_COLOR} onAdd={onAddNode} onDrag={startDrag} />
      </div>
    </aside>
  );
}

interface PaletteItemProps {
  type: string;
  label: string;
  icon: string;
  color: string;
  extraParams?: Record<string, unknown>;
  onAdd: (type: string, label: string, extraParams?: Record<string, unknown>) => void;
  onDrag: (e: React.MouseEvent, type: string, label: string, extraParams?: Record<string, unknown>) => void;
}

function PaletteItem({ type, label, icon, color, extraParams, onAdd, onDrag }: PaletteItemProps) {
  return (
    <div
      className="palette-node"
      role="button"
      tabIndex={0}
      onClick={() => onAdd(type, label, extraParams)}
      onMouseDown={(e) => { if (e.button === 0) onDrag(e, type, label, extraParams); }}
      style={{ "--node-color": color } as React.CSSProperties}
    >
      <span className="palette-node-icon">[{icon}]</span>
      <span className="palette-node-label">{label}</span>
    </div>
  );
}
