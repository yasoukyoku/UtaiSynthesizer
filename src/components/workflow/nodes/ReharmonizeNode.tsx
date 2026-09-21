/**
 * ReharmonizeNode — 和声重配 (P2-5, 规划 2-5).
 * 引擎: engine.ts case "reharmonize" — reharmonizeSegments:
 * 端口0 = chordBlock (可回喂 melodyGen/harmonizer), 端口1 = chordDetect 兼容分段.
 */
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";

const STRAT_KEYS = ["symStratDiatonic", "symStratBorrowed", "symStratTritone", "symStratExtension"];

export function ReharmonizeNode(props: NodeProps) {
  const { t } = useTranslation();
  const [params, updateParams] = useNodeParams(props);
  const strategy = (params.strategy as number) ?? 0;
  const density = (params.density as number) ?? 50;
  const seed = (params.seed as number) ?? 0;

  return (
    <NodeShell
      nodeId={props.id}
      label={t("workflow.nodeReharmonize")}
      icon="🎷"
      color="#22d3ee"
      inputs={1}
      outputs={2}
      outputLabels={["chords", "info"]}
    >
      <div className="sep-node-body">
        <div className="sep-params">
          <div className="sep-label-row">
            <span className="sep-label">{t("workflow.symStrategy")}</span>
            <select
              className="sep-select"
              value={String(strategy)}
              onChange={(e) => updateParams({ strategy: Number(e.target.value) })}
            >
              {STRAT_KEYS.map((k, i) => (
                <option key={k} value={i}>{t(`workflow.${k}`)}</option>
              ))}
            </select>
          </div>
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
            {t("workflow.symReharmChordHint")}
          </div>
        </div>
      </div>
    </NodeShell>
  );
}
