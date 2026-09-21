/**
 * BreathPlannerNode — 换气规划 (P2-11, 规划 2-11).
 * 引擎: engine.ts case "breathPlanner" — planBreathPoints: 音符流标注换气点;
 * 端口0 = 原样透传 (标注层不改音符), 端口1 = 换气点 JSON.
 */
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";

export function BreathPlannerNode(props: NodeProps) {
  const { t } = useTranslation();
  const [params, updateParams] = useNodeParams(props);
  const minGapBeats = (params.minGapBeats as number) ?? 1;
  const lyrics = (params.lyrics as string) ?? "";

  return (
    <NodeShell
      nodeId={props.id}
      label={t("workflow.nodeBreathPlanner")}
      icon="💨"
      color="#60a5fa"
      inputs={1}
      outputs={2}
      outputLabels={["input", "breath"]}
    >
      <div className="sep-node-body">
        <div className="sep-params">
          <ParamSlider
            label={t("workflow.symMinGap")}
            min={0.5} max={4} step={0.5} value={minGapBeats}
            format={(v) => `${v}`}
            onChange={(v) => updateParams({ minGapBeats: v })}
          />
          <div className="sep-label-row">
            <span className="sep-label">{t("workflow.symLyrics")}</span>
            <textarea
              className="sep-select"
              rows={2}
              style={{ resize: "none", fontSize: 10 }}
              value={lyrics}
              onChange={(e) => updateParams({ lyrics: e.target.value })}
            />
          </div>
          <div style={{ fontSize: 10, color: "#888", padding: "4px 0 0 4px" }}>
            {t("workflow.symBreathHint")}
          </div>
        </div>
      </div>
    </NodeShell>
  );
}
