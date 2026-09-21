/**
 * SplitNode — 输入分片/多路扇出.
 * 引擎: engine.ts case "split" — 把 primaryInput 原样复制到 N 个输出端口.
 * 不做任何 DSP, 纯粹是路由工具 (把一路信号喂给多个下游消费者).
 */
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";

export function SplitNode(props: NodeProps) {
  const { t } = useTranslation();
  const [params, updateParams] = useNodeParams(props);
  const outputs = Math.max(1, Math.min(8, Math.round((params.outputs as number) ?? 2)));

  return (
    <NodeShell
      nodeId={props.id}
      label={t("workflow.nodeSplit")}
      icon="✂"
      color="#94a3b8"
      inputs={1}
      outputs={outputs}
      outputLabels={Array.from({ length: outputs }, (_, i) => `${i + 1}`)}
    >
      <div className="sep-node-body">
        <div className="sep-params">
          <ParamSlider
            label={t("workflow.splitOutputs")}
            title={t("workflow.splitOutputsTitle")}
            min={1} max={8} step={1} value={outputs}
            format={(v) => `${v}`}
            onChange={(v) => updateParams({ outputs: Math.round(v) })}
          />
        </div>
      </div>
    </NodeShell>
  );
}
