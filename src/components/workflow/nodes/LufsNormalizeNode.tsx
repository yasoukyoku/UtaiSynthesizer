/**
 * LufsNormalizeNode — 响度归一 (EBU R128).
 * 引擎: engine.ts case "lufsNormalize" — 测量 integrated LUFS 后按目标施加线性增益.
 * 后端: normalize_to_lufs — ebur128 测量 + 归一化输出.
 */
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSliderWithUnit } from "./ParamSliderWithUnit";

export function LufsNormalizeNode(props: NodeProps) {
  const { t } = useTranslation();
  const [params, updateParams] = useNodeParams(props);
  const targetLufs = (params.targetLufs as number) ?? -16;

  return (
    <NodeShell
      nodeId={props.id}
      label={t("workflow.nodeLufsNormalize")}
      icon="📢"
      color="#14b8a6"
      inputs={1}
      outputs={1}
      outputLabels={["audio"]}
    >
      <div className="sep-node-body">
        <div className="sep-params">
          <ParamSliderWithUnit
            label={t("workflow.lufsTarget")}
            title={t("workflow.lufsTargetTitle")}
            unitType="lufs"
            min={-24} max={-9} step={0.5} value={targetLufs}
            onChange={(v) => updateParams({ targetLufs: v })}
          />
          <div style={{ fontSize: 10, color: "var(--text-muted)", padding: "4px 0 0 4px" }}>
            {t("workflow.lufsHint")}
          </div>
        </div>
      </div>
    </NodeShell>
  );
}
