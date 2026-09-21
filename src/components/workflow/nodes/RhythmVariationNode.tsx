/**
 * RhythmVariationNode — 节奏变奏 (P2-6, 规划 2-6).
 * 引擎: engine.ts case "rhythmVariation" — varyRhythm: 提前/拖后/切分/留白, 起音位置平移.
 */
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";

const MODE_KEYS = ["symModePush", "symModeLayBack", "symModeSyncopate", "symModeSparse"];

export function RhythmVariationNode(props: NodeProps) {
  const { t } = useTranslation();
  const [params, updateParams] = useNodeParams(props);
  const mode = (params.mode as number) ?? 0;
  const amount = (params.amount as number) ?? 50;

  return (
    <NodeShell
      nodeId={props.id}
      label={t("workflow.nodeRhythmVariation")}
      icon="🕹"
      color="#e879f9"
      inputs={1}
      outputs={1}
      outputLabels={["notes"]}
    >
      <div className="sep-node-body">
        <div className="sep-params">
          <div className="sep-label-row">
            <span className="sep-label">{t("workflow.symMode")}</span>
            <select
              className="sep-select"
              value={String(mode)}
              onChange={(e) => updateParams({ mode: Number(e.target.value) })}
            >
              {MODE_KEYS.map((k, i) => (
                <option key={k} value={i}>{t(`workflow.${k}`)}</option>
              ))}
            </select>
          </div>
          <ParamSlider
            label={t("workflow.symAmount")}
            min={0} max={100} step={1} value={amount}
            onChange={(v) => updateParams({ amount: v })}
          />
          <div style={{ fontSize: 10, color: "#888", padding: "4px 0 0 4px" }}>
            {t("workflow.symRhythmVarHint")}
          </div>
        </div>
      </div>
    </NodeShell>
  );
}
