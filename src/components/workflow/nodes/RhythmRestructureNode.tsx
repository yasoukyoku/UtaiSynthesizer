/**
 * RhythmRestructureNode — 节奏重构 (P2-4b, 规划 2-4).
 * 引擎: engine.ts case "rhythmRestructure" — rhythmRestructure: 小节对齐的节奏重排, 音高序列不变.
 */
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";

export function RhythmRestructureNode(props: NodeProps) {
  const { t } = useTranslation();
  const [params, updateParams] = useNodeParams(props);
  const strength = (params.strength as number) ?? 60;
  const beatsPerBar = (params.beatsPerBar as number) ?? 4;
  const seed = (params.seed as number) ?? 0;

  return (
    <NodeShell
      nodeId={props.id}
      label={t("workflow.nodeRhythmRestructure")}
      icon="🥁"
      color="#fbbf24"
      inputs={1}
      outputs={1}
      outputLabels={["notes"]}
    >
      <div className="sep-node-body">
        <div className="sep-params">
          <ParamSlider
            label={t("workflow.symStrength")}
            min={0} max={100} step={1} value={strength}
            onChange={(v) => updateParams({ strength: v })}
          />
          <ParamSlider
            label={t("workflow.symBeatsPerBar")}
            min={2} max={7} step={1} value={beatsPerBar}
            onChange={(v) => updateParams({ beatsPerBar: v })}
          />
          <ParamSlider
            label={t("workflow.symSeed")}
            min={0} max={4095} step={1} value={seed}
            onChange={(v) => updateParams({ seed: v })}
          />
          <div style={{ fontSize: 10, color: "#888", padding: "4px 0 0 4px" }}>
            {t("workflow.symRhythmRestrHint")}
          </div>
        </div>
      </div>
    </NodeShell>
  );
}
