/**
 * MidiHumanizeNode — 演奏人性化 (P2-1, 规划 2-1).
 * 引擎: engine.ts case "midiHumanize" — humanizeNotes: 毫秒级时值/力度微扰, 种子可复现.
 */
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";

export function MidiHumanizeNode(props: NodeProps) {
  const { t } = useTranslation();
  const [params, updateParams] = useNodeParams(props);
  const timingMs = (params.timingMs as number) ?? 12;
  const velocityJitter = (params.velocityJitter as number) ?? 8;
  const seed = (params.seed as number) ?? 0;

  return (
    <NodeShell
      nodeId={props.id}
      label={t("workflow.nodeMidiHumanize")}
      icon="🎹"
      color="#38bdf8"
      inputs={1}
      outputs={1}
      outputLabels={["notes"]}
    >
      <div className="sep-node-body">
        <div className="sep-params">
          <ParamSlider
            label={t("workflow.symTimingMs")}
            min={0} max={50} step={1} value={timingMs}
            format={(v) => `${v} ms`}
            onChange={(v) => updateParams({ timingMs: v })}
          />
          <ParamSlider
            label={t("workflow.symVelJitter")}
            min={0} max={30} step={1} value={velocityJitter}
            onChange={(v) => updateParams({ velocityJitter: v })}
          />
          <ParamSlider
            label={t("workflow.symSeed")}
            min={0} max={4095} step={1} value={seed}
            onChange={(v) => updateParams({ seed: v })}
          />
          <div style={{ fontSize: 10, color: "#888", padding: "4px 0 0 4px" }}>
            {t("workflow.symHumanizeHint")}
          </div>
        </div>
      </div>
    </NodeShell>
  );
}
