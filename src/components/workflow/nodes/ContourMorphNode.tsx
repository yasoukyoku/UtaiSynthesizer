/**
 * ContourMorphNode — 旋律轮廓变形 (P2-4c, 规划 2-4 验收核心).
 * 引擎: engine.ts case "contourMorph" — contourMorph: 相似度收敛带 35–45% (60% 强度),
 * 每音程独立概率重构, 可保护乐句最高点. 验收: melodySimilarity ∈ [0.35, 0.45].
 */
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";

const CLIMAX_KEYS = ["symOff", "symOn"];

export function ContourMorphNode(props: NodeProps) {
  const { t } = useTranslation();
  const [params, updateParams] = useNodeParams(props);
  const strength = (params.strength as number) ?? 60;
  const keepClimax = (params.keepClimax as number) ?? 1;
  const seed = (params.seed as number) ?? 0;

  return (
    <NodeShell
      nodeId={props.id}
      label={t("workflow.nodeContourMorph")}
      icon="🎢"
      color="#f97316"
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
          <div className="sep-label-row">
            <span className="sep-label">{t("workflow.symKeepClimax")}</span>
            <select
              className="sep-select"
              value={String(keepClimax)}
              onChange={(e) => updateParams({ keepClimax: Number(e.target.value) })}
            >
              {CLIMAX_KEYS.map((k, i) => (
                <option key={k} value={i}>{t(`workflow.${k}`)}</option>
              ))}
            </select>
          </div>
          <ParamSlider
            label={t("workflow.symSeed")}
            min={0} max={4095} step={1} value={seed}
            onChange={(v) => updateParams({ seed: v })}
          />
          <div style={{ fontSize: 10, color: "#888", padding: "4px 0 0 4px" }}>
            {t("workflow.symContourHint")}
          </div>
        </div>
      </div>
    </NodeShell>
  );
}
