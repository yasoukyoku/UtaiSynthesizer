/**
 * SpectralCompareNode — 频谱对比 (Phase 5-5, 分析可视化族).
 * 引擎: engine.ts case "spectralCompare" — A 原样透传 (端口 0),
 *       log-mel 相似度报告 JSON 输出到端口 1; B 接端口 1 (对比输入).
 * 后端: compare_spectra — log-mel 余弦 / L2 / 频段能量差 (A−B).
 */
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";

export function SpectralCompareNode(props: NodeProps) {
  const { t } = useTranslation();
  useNodeParams(props);

  return (
    <NodeShell
      nodeId={props.id}
      label={t("workflow.nodeSpectralCompare")}
      icon="⚖"
      color="#a78bfa"
      inputs={2}
      outputs={2}
      outputLabels={[t("workflow.analysisOutAudio"), t("workflow.analysisOutReport")]}
    >
      <div className="sep-node-body">
        <div style={{ fontSize: 10, color: "var(--text-muted)", padding: "4px 0 0 4px" }}>
          {t("workflow.spectralCompareHint")}
        </div>
      </div>
    </NodeShell>
  );
}
