/**
 * MelodyReharmNode — 旋律级度替换 (P2-4a, 规划 2-4).
 * 引擎: engine.ts case "melodyReharm" — degreeSwap: 和弦内音/调式音级等度替换, 保持轮廓与可唱性.
 */
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";

export function MelodyReharmNode(props: NodeProps) {
  const { t } = useTranslation();
  const [params, updateParams] = useNodeParams(props);
  const density = (params.density as number) ?? 40;
  const seed = (params.seed as number) ?? 0;

  return (
    <NodeShell
      nodeId={props.id}
      label={t("workflow.nodeMelodyReharm")}
      icon="🎼"
      color="#34d399"
      inputs={1}
      outputs={1}
      outputLabels={["notes"]}
    >
      <div className="sep-node-body">
        <div className="sep-params">
          <ParamSlider
            label={t("workflow.symDensity")}
            min={0} max={100} step={1} value={density}
            onChange={(v) => updateParams({ density: v })}
          />
          <ParamSlider
            label={t("workflow.symSeed")}
            min={0} max={4095} step={1} value={seed}
            onChange={(v) => updateParams({ seed: v })}
          />
          <div style={{ fontSize: 10, color: "#888", padding: "4px 0 0 4px" }}>
            {t("workflow.symReharmHint")}
          </div>
        </div>
      </div>
    </NodeShell>
  );
}
