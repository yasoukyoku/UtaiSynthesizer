/**
 * DtwAlignNode — DTW 对齐对比 (Phase 5-6, 分析可视化族).
 * 引擎: engine.ts case "dtwAlign" — A 原样透传 (端口 0),
 *       DTW 报告 JSON 输出到端口 1; B 接端口 1 (对比输入).
 * 后端: dtw_compare — Sakoe-Chiba 带约束的动态时间规整.
 */
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";

export function DtwAlignNode(props: NodeProps) {
  const { t } = useTranslation();
  const [params, updateParams] = useNodeParams(props);
  const band = Math.max(0, Math.min(64, Math.round((params.band as number) ?? 0)));

  return (
    <NodeShell
      nodeId={props.id}
      label={t("workflow.nodeDtwAlign")}
      icon="🧭"
      color="#f472b6"
      inputs={2}
      outputs={2}
      outputLabels={[t("workflow.analysisOutAudio"), t("workflow.analysisOutReport")]}
    >
      <div className="sep-node-body">
        <div className="sep-params">
          <ParamSlider
            label={t("workflow.dtwBand")}
            title={t("workflow.dtwHint")}
            min={0} max={64} step={1} value={band}
            format={(v) => `${Math.round(v)}`}
            onChange={(v) => updateParams({ band: Math.round(v) })}
          />
          <div style={{ fontSize: 10, color: "var(--text-muted)", padding: "4px 0 0 4px" }}>
            {t("workflow.dtwHint")}
          </div>
        </div>
      </div>
    </NodeShell>
  );
}
