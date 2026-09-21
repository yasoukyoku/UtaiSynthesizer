/**
 * AbCompareNode — A/B 对比直通 (Phase 5-7, 分析可视化族).
 * 引擎: engine.ts case "abCompare" — 纯前端零处理: A→端口 0, B→端口 1.
 * 用途: 并排监听两条链路, 搭配输出预览逐端口试听对比.
 */
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";

export function AbCompareNode(props: NodeProps) {
  const { t } = useTranslation();
  useNodeParams(props);

  return (
    <NodeShell
      nodeId={props.id}
      label={t("workflow.nodeAbCompare")}
      icon="🔀"
      color="#fb923c"
      inputs={2}
      outputs={2}
      outputLabels={[t("workflow.analysisOutAudio"), t("workflow.analysisOutB")]}
    >
      <div className="sep-node-body">
        <div style={{ fontSize: 10, color: "var(--text-muted)", padding: "4px 0 0 4px" }}>
          {t("workflow.abCompareHint")}
        </div>
      </div>
    </NodeShell>
  );
}
