/**
 * VelocityCurveNode — 力度曲线 (P2-2, 规划 2-2).
 * 引擎: engine.ts case "velocityCurve" — applyVelocityCurve: 渐强/渐弱/拱形/自定义采样包络.
 */
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";

const CURVE_KEYS = ["symCurveCresc", "symCurveDecresc", "symCurveArch", "symCurveCustom"];

export function VelocityCurveNode(props: NodeProps) {
  const { t } = useTranslation();
  const [params, updateParams] = useNodeParams(props);
  const curveIdx = (params.curve as number) ?? 2;
  const intensity = (params.intensity as number) ?? 60;
  const customShape = (params.customShape as string) ?? "";

  return (
    <NodeShell
      nodeId={props.id}
      label={t("workflow.nodeVelocityCurve")}
      icon="📈"
      color="#f472b6"
      inputs={1}
      outputs={1}
      outputLabels={["notes"]}
    >
      <div className="sep-node-body">
        <div className="sep-params">
          <div className="sep-label-row">
            <span className="sep-label">{t("workflow.symCurve")}</span>
            <select
              className="sep-select"
              value={String(curveIdx)}
              onChange={(e) => updateParams({ curve: Number(e.target.value) })}
            >
              {CURVE_KEYS.map((k, i) => (
                <option key={k} value={i}>{t(`workflow.${k}`)}</option>
              ))}
            </select>
          </div>
          <ParamSlider
            label={t("workflow.symIntensity")}
            min={0} max={100} step={1} value={intensity}
            onChange={(v) => updateParams({ intensity: v })}
          />
          {curveIdx === 3 && (
            <div className="sep-label-row">
              <span className="sep-label">{t("workflow.symCustomShape")}</span>
              <textarea
                className="sep-select"
                rows={2}
                style={{ resize: "none", fontSize: 10 }}
                value={customShape}
                placeholder="20, 60, 100, 40"
                onChange={(e) => updateParams({ customShape: e.target.value })}
              />
            </div>
          )}
          <div style={{ fontSize: 10, color: "#888", padding: "4px 0 0 4px" }}>
            {t("workflow.symVelCurveHint")}
          </div>
        </div>
      </div>
    </NodeShell>
  );
}
