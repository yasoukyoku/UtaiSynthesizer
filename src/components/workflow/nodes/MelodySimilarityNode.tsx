import { useCallback } from "react";
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";

const selStyle: React.CSSProperties = {
  marginTop: 2,
  padding: "4px 6px",
  borderRadius: 6,
  border: "1px solid var(--border-subtle)",
  background: "var(--bg-surface)",
  color: "var(--text-primary)",
  fontSize: 12,
};

const labelStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 2,
  fontSize: 12,
  color: "var(--text-secondary)",
};

export function MelodySimilarityNode(props: NodeProps) {
  const { t } = useTranslation();
  const [params, updateParams] = useNodeParams(props);

  const threshold = (params.threshold as number) ?? 35;

  const handleThresholdChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const val = Math.max(0, Math.min(100, parseInt(e.target.value, 10) || 35));
      updateParams({ threshold: val });
    },
    [updateParams],
  );

  return (
    <NodeShell
      nodeId={props.id}
      label={t("workflow.nodeMelodySimilarity")}
      icon="🔍"
      color="#f59e0b"
      inputs={2}
      outputs={2}
      outputLabels={[t("workflow.analysisOutMidi"), t("workflow.analysisOutReport")]}
    >
      <div className="sep-node-body">
        <label style={labelStyle}>
          <span>{t("workflow.melodySimilarityThreshold")}</span>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <input
              type="range"
              min="0"
              max="100"
              value={threshold}
              onChange={handleThresholdChange}
              style={{ flex: 1, minWidth: 100 }}
            />
            <input
              type="number"
              value={threshold}
              onChange={handleThresholdChange}
              style={{ ...selStyle, width: 50, textAlign: "center" }}
              min="0"
              max="100"
            />
            <span style={{ fontSize: 11, color: "var(--text-muted)" }}>%</span>
          </div>
        </label>
        <div style={{ fontSize: 10, color: "var(--text-muted)", padding: "4px 0 0 0" }}>
          {t("workflow.melodySimilarityHint")}
        </div>
      </div>
    </NodeShell>
  );
}
