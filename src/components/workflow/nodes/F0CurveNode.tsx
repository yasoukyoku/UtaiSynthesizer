/**
 * F0CurveNode — 基频轨迹 (Phase 5-2, 分析可视化族).
 * 引擎: engine.ts case "f0Curve" — 音频原样透传 (端口 0),
 *       F0 轨迹报告 JSON 输出到端口 1.
 * 后端: track_f0 — 逐帧基频检测 ( voiced / unvoiced 标记 ).
 */
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";

export function F0CurveNode(props: NodeProps) {
  const { t } = useTranslation();
  useNodeParams(props);

  return (
    <NodeShell
      nodeId={props.id}
      label={t("workflow.nodeF0Curve")}
      icon="〰"
      color="#22d3ee"
      inputs={1}
      outputs={2}
      outputLabels={[t("workflow.analysisOutAudio"), t("workflow.analysisOutReport")]}
    >
      <div className="sep-node-body">
        <div style={{ fontSize: 10, color: "var(--text-muted)", padding: "4px 0 0 4px" }}>
          {t("workflow.f0CurveHint")}
        </div>
      </div>
    </NodeShell>
  );
}
