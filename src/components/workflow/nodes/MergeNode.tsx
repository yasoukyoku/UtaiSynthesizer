/**
 * MergeNode — 多路音频混音合并.
 * 引擎: engine.ts case "merge" — 把 N 路输入求和混成一路.
 * 后端: mix_audio_files — 单声道自动升为立体声, 短输入补静音,
 *       输出未钳位的 32-bit Float WAV (保留完整动态余量, 峰值写入运行报告).
 */
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";

export function MergeNode(props: NodeProps) {
  const { t } = useTranslation();
  const [params, updateParams] = useNodeParams(props);
  const inputs = Math.max(2, Math.min(8, Math.round((params.inputs as number) ?? 2)));

  return (
    <NodeShell
      nodeId={props.id}
      label={t("workflow.nodeMerge")}
      icon="🧲"
      color="#fb923c"
      inputs={inputs}
      outputs={1}
      outputLabels={["mix"]}
    >
      <div className="sep-node-body">
        <div className="sep-params">
          <ParamSlider
            label={t("workflow.mergeInputs")}
            title={t("workflow.mergeInputsTitle")}
            min={2} max={8} step={1} value={inputs}
            format={(v) => `${v}`}
            onChange={(v) => updateParams({ inputs: Math.round(v) })}
          />
          <div style={{ fontSize: 10, color: "var(--text-muted)", padding: "4px 0 0 4px" }}>
            {t("workflow.mergeHint")}
          </div>
        </div>
      </div>
    </NodeShell>
  );
}
