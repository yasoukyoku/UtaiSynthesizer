/**
 * HarmonicityCheckNode — 谐波健康检查 (Phase 5-4, 分析可视化族).
 * 引擎: engine.ts case "harmonicityCheck" — 音频原样透传 (端口 0),
 *       谐波 8 指标报告 JSON 输出到端口 1; f0Hz=0 → 自动检测基频.
 * 后端: analyze_harmonicity — 二次谐波验证 / 谐波能量比 / 噪声底等.
 */
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";

export function HarmonicityCheckNode(props: NodeProps) {
  const { t } = useTranslation();
  const [params, updateParams] = useNodeParams(props);
  const f0Hz = Math.max(0, Math.min(1000, (params.f0Hz as number) ?? 0));

  return (
    <NodeShell
      nodeId={props.id}
      label={t("workflow.nodeHarmonicityCheck")}
      icon="🔬"
      color="#4ade80"
      inputs={1}
      outputs={2}
      outputLabels={[t("workflow.analysisOutAudio"), t("workflow.analysisOutReport")]}
    >
      <div className="sep-node-body">
        <div className="sep-params">
          <ParamSlider
            label={t("workflow.harmonicityTargetF0")}
            title={t("workflow.harmonicityHint")}
            min={0} max={1000} step={5} value={f0Hz}
            format={(v) => `${Math.round(v)} Hz`}
            onChange={(v) => updateParams({ f0Hz: v })}
          />
          <div style={{ fontSize: 10, color: "var(--text-muted)", padding: "4px 0 0 4px" }}>
            {t("workflow.harmonicityHint")}
          </div>
        </div>
      </div>
    </NodeShell>
  );
}
