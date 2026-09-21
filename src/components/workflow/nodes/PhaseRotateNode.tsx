/**
 * PhaseRotateNode — 全通相位旋转 (Phase 4-5, 配方 F ⑤).
 * 引擎: engine.ts case "phaseRotate" — strength 0=旁通, 每步 1/8 = 1 个全通段.
 * 后端: apply_phase_rotate — 8 级 2 阶全通级联 (60Hz-9kHz, Q=0.7), 仅做峰值余量工具.
 */
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";

export function PhaseRotateNode(props: NodeProps) {
  const { t } = useTranslation();
  const [params, updateParams] = useNodeParams(props);
  const strength = Math.max(0, Math.min(1, (params.strength as number) ?? 0));

  return (
    <NodeShell
      nodeId={props.id}
      label={t("workflow.nodePhaseRotate")}
      icon="🌀"
      color="#c084fc"
      inputs={1}
      outputs={1}
      outputLabels={["audio"]}
    >
      <div className="sep-node-body">
        <div className="sep-params">
          <ParamSlider
            label={t("workflow.phaseRotateStrength")}
            title={t("workflow.phaseRotateHint")}
            min={0} max={1} step={0.125} value={strength}
            format={(v) => `${Math.round(v * 8)} / 8`}
            onChange={(v) => updateParams({ strength: v })}
          />
          <div style={{ fontSize: 10, color: "var(--text-muted)", padding: "4px 0 0 4px" }}>
            {t("workflow.phaseRotateHint")}
          </div>
        </div>
      </div>
    </NodeShell>
  );
}
