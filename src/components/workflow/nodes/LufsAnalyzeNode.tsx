import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";

export function LufsAnalyzeNode(props: NodeProps) {
  const { t } = useTranslation();
  const [params, updateParams] = useNodeParams(props);
  const targetLufs = (params.targetLufs as number) ?? -16;

  return (
    <NodeShell
      nodeId={props.id}
      label={t("workflow.nodeLufsAnalyze")}
      icon="📊"
      color="#8b5cf6"
      inputs={1}
      outputs={2}
      outputLabels={[t("workflow.lufsAnalyzeOutAudio"), t("workflow.lufsAnalyzeOutReport")]}
    >
      <div className="sep-node-body">
        <div className="sep-params">
          <ParamSlider
            label={t("workflow.lufsTargetRef")}
            title={t("workflow.lufsTargetRefTitle")}
            min={-24} max={-9} step={0.5} value={targetLufs}
            format={(v) => `${v.toFixed(1)} LUFS`}
            onChange={(v) => updateParams({ targetLufs: v })}
          />
          <div style={{ fontSize: 10, color: "var(--text-muted)", padding: "4px 0 0 4px" }}>
            {t("workflow.lufsAnalyzeHint")}
          </div>
        </div>
      </div>
    </NodeShell>
  );
}
