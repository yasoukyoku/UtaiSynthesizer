/**
 * SaturateNode — 饱和激励 (Phase 4-3, 配方 F ④).
 * 引擎: engine.ts case "saturate" — tanh 归一化软削波, drive 0=旁通.
 * 后端: apply_saturate — 输入钳位 [-1,1] 后过 tanh, 输出永不超 0dBFS.
 */
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSliderWithUnit } from "./ParamSliderWithUnit";

export function SaturateNode(props: NodeProps) {
  const { t } = useTranslation();
  const [params, updateParams] = useNodeParams(props);
  const drive = Math.max(0, Math.min(1, (params.drive as number) ?? 0.05));

  return (
    <NodeShell
      nodeId={props.id}
      label={t("workflow.nodeSaturate")}
      icon="🔥"
      color="#f97316"
      inputs={1}
      outputs={1}
      outputLabels={["audio"]}
    >
      <div className="sep-node-body">
        <div className="sep-params">
          <ParamSliderWithUnit
            label={t("workflow.saturateDrive")}
            unitType="percent"
            title={t("workflow.saturateDriveTitle")}
            min={0} max={1} step={0.01} value={drive}
            onChange={(v) => updateParams({ drive: v })}
          />
          <div style={{ fontSize: 10, color: "var(--text-muted)", padding: "4px 0 0 4px" }}>
            {t("workflow.saturateHint")}
          </div>
        </div>
      </div>
    </NodeShell>
  );
}
