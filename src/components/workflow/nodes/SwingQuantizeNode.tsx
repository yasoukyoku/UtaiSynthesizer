/**
 * SwingQuantizeNode — Swing 律动 + 网格量化 (P2-3, 规划 2-3).
 * 引擎: engine.ts case "swingQuantize" — swingQuantizeNotes: 奇数网格位延迟 + 吸附量化.
 */
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";

const GRID_KEYS = ["symGrid8", "symGrid16"];

export function SwingQuantizeNode(props: NodeProps) {
  const { t } = useTranslation();
  const [params, updateParams] = useNodeParams(props);
  const gridIdx = (params.grid as number) ?? 0;
  const swing = (params.swing as number) ?? 55;
  const quantize = (params.quantize as number) ?? 0;

  return (
    <NodeShell
      nodeId={props.id}
      label={t("workflow.nodeSwingQuantize")}
      icon="🥁"
      color="#a3e635"
      inputs={1}
      outputs={1}
      outputLabels={["notes"]}
    >
      <div className="sep-node-body">
        <div className="sep-params">
          <div className="sep-label-row">
            <span className="sep-label">{t("workflow.symGrid")}</span>
            <select
              className="sep-select"
              value={String(gridIdx)}
              onChange={(e) => updateParams({ grid: Number(e.target.value) })}
            >
              {GRID_KEYS.map((k, i) => (
                <option key={k} value={i}>{t(`workflow.${k}`)}</option>
              ))}
            </select>
          </div>
          <ParamSlider
            label={t("workflow.symSwing")}
            min={0} max={100} step={1} value={swing}
            onChange={(v) => updateParams({ swing: v })}
          />
          <ParamSlider
            label={t("workflow.symQuantize")}
            min={0} max={100} step={1} value={quantize}
            onChange={(v) => updateParams({ quantize: v })}
          />
          <div style={{ fontSize: 10, color: "#888", padding: "4px 0 0 4px" }}>
            {t("workflow.symSwingHint")}
          </div>
        </div>
      </div>
    </NodeShell>
  );
}
