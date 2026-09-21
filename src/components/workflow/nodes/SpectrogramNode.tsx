/**
 * SpectrogramNode — 语谱图 (Phase 5-1, 分析可视化族).
 * 引擎: engine.ts case "spectrogram" — 音频原样透传 (端口 0),
 *       Inferno PNG artifact 落盘缓存目录 (端口 1), 几何报告 JSON (端口 2).
 * 后端: analyze_spectrogram — STFT → log 幅度 → Inferno 调色 PNG.
 */
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";

export function SpectrogramNode(props: NodeProps) {
  const { t } = useTranslation();
  useNodeParams(props);

  return (
    <NodeShell
      nodeId={props.id}
      label={t("workflow.nodeSpectrogram")}
      icon="🌊"
      color="#38bdf8"
      inputs={1}
      outputs={3}
      outputLabels={[t("workflow.analysisOutAudio"), t("workflow.analysisOutPng"), t("workflow.analysisOutReport")]}
    >
      <div className="sep-node-body">
        <div style={{ fontSize: 10, color: "var(--text-muted)", padding: "4px 0 0 4px" }}>
          {t("workflow.spectrogramHint")}
        </div>
      </div>
    </NodeShell>
  );
}
