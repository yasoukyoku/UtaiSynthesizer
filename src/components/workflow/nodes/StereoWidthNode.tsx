/**
 * StereoWidthNode — 立体声宽度 (Phase 4-2, 配方 F ③).
 * 引擎: engine.ts case "stereoWidth" — M/S 矩阵, 0=单声道 / 1=原样 / 上限 1.5.
 * 后端: apply_stereo_width — 中侧处理, Rust 侧二次钳位兜底并上报 capped.
 */
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSliderWithUnit } from "./ParamSliderWithUnit";

export function StereoWidthNode(props: NodeProps) {
  const { t } = useTranslation();
  const [params, updateParams] = useNodeParams(props);
  const width = Math.max(0, Math.min(1.5, (params.width as number) ?? 1));

  return (
    <NodeShell
      nodeId={props.id}
      label={t("workflow.nodeStereoWidth")}
      icon="🎧"
      color="#a3e635"
      inputs={1}
      outputs={1}
      outputLabels={["audio"]}
    >
      <div className="sep-node-body">
        <div className="sep-params">
          <ParamSliderWithUnit
            label={t("workflow.stereoWidthAmount")}
            unitType="ratio"
            title={t("workflow.stereoWidthHint")}
            min={0} max={1.5} step={0.05} value={width}
            onChange={(v) => updateParams({ width: v })}
          />
          <div style={{ fontSize: 10, color: "var(--text-muted)", padding: "4px 0 0 4px" }}>
            {t("workflow.stereoWidthHint")}
          </div>
        </div>
      </div>
    </NodeShell>
  );
}
