/**
 * TimbreMetricsNode — 音色量规 (Phase 5-3, 分析可视化族).
 * 引擎: engine.ts case "timbreMetrics" — 音频原样透传 (端口 0),
 *       音色指标报告 JSON 输出到端口 1.
 * 后端: analyze_timbre — 频谱质心 / 滚降 / 平坦度 / 频段能量分布.
 */
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";

export function TimbreMetricsNode(props: NodeProps) {
  const { t } = useTranslation();
  useNodeParams(props);

  return (
    <NodeShell
      nodeId={props.id}
      label={t("workflow.nodeTimbreMetrics")}
      icon="🎨"
      color="#fbbf24"
      inputs={1}
      outputs={2}
      outputLabels={[t("workflow.analysisOutAudio"), t("workflow.analysisOutReport")]}
    >
      <div className="sep-node-body">
        <div style={{ fontSize: 10, color: "var(--text-muted)", padding: "4px 0 0 4px" }}>
          {t("workflow.timbreHint")}
        </div>
      </div>
    </NodeShell>
  );
}
