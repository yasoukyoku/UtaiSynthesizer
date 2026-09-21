/**
 * MotifDevelopNode — 动机发展 (P2-4d, 规划 2-4).
 * 引擎: engine.ts case "motifDevelop" — motifDevelop: 截取开头动机, 按模进/倒影/逆行/增值/减值/混合发展.
 */
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";

const TECH_KEYS = [
  "symTechSequence", "symTechInvert", "symTechRetrograde",
  "symTechAugment", "symTechDiminish", "symTechMixed",
];

export function MotifDevelopNode(props: NodeProps) {
  const { t } = useTranslation();
  const [params, updateParams] = useNodeParams(props);
  const technique = (params.technique as number) ?? 5;
  const motifBars = (params.motifBars as number) ?? 2;
  const seed = (params.seed as number) ?? 0;

  return (
    <NodeShell
      nodeId={props.id}
      label={t("workflow.nodeMotifDevelop")}
      icon="🧬"
      color="#c084fc"
      inputs={1}
      outputs={1}
      outputLabels={["notes"]}
    >
      <div className="sep-node-body">
        <div className="sep-params">
          <div className="sep-label-row">
            <span className="sep-label">{t("workflow.symTechnique")}</span>
            <select
              className="sep-select"
              value={String(technique)}
              onChange={(e) => updateParams({ technique: Number(e.target.value) })}
            >
              {TECH_KEYS.map((k, i) => (
                <option key={k} value={i}>{t(`workflow.${k}`)}</option>
              ))}
            </select>
          </div>
          <ParamSlider
            label={t("workflow.symMotifBars")}
            min={2} max={4} step={1} value={motifBars}
            format={(v) => `${v}`}
            onChange={(v) => updateParams({ motifBars: v })}
          />
          <ParamSlider
            label={t("workflow.symSeed")}
            min={0} max={4095} step={1} value={seed}
            onChange={(v) => updateParams({ seed: v })}
          />
          <div style={{ fontSize: 10, color: "#888", padding: "4px 0 0 4px" }}>
            {t("workflow.symMotifHint")}
          </div>
        </div>
      </div>
    </NodeShell>
  );
}
