/**
 * BusEqNode — 5 段母线 EQ (Phase 4-1, 配方 F ②).
 * 引擎: engine.ts case "busEq" — hpf30 / lowShelf120 / peak500 / peak2.5k / highShelf10k.
 * 后端: apply_bus_eq — RBJ cookbook 双二阶级联, link-stereo (左右共用系数, 不破坏声像).
 */
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSliderWithUnit } from "./ParamSliderWithUnit";

const GAIN_BANDS = [
  { key: "eqGain0", labelKey: "workflow.eqBandLow", titleKey: "workflow.eqBandGainTitle" },
  { key: "eqGain1", labelKey: "workflow.eqBandMid", titleKey: "workflow.eqBandGainTitle" },
  { key: "eqGain2", labelKey: "workflow.eqBandMidHigh", titleKey: "workflow.eqBandGainTitle" },
  { key: "eqGain3", labelKey: "workflow.eqBandHigh", titleKey: "workflow.eqBandGainTitle" },
];

export function BusEqNode(props: NodeProps) {
  const { t } = useTranslation();
  const [params, updateParams] = useNodeParams(props);
  const eqHpf = ((params.eqHpf as number) ?? 1) > 0 ? 1 : 0;

  return (
    <NodeShell
      nodeId={props.id}
      label={t("workflow.nodeBusEq")}
      icon="🎚"
      color="#38bdf8"
      inputs={1}
      outputs={1}
      outputLabels={["audio"]}
    >
      <div className="sep-node-body">
        <div className="sep-params">
          <div className="sep-label-row">
            <span className="sep-label">{t("workflow.eqBandHpf")}</span>
            <select
              className="sep-select"
              value={eqHpf}
              title={t("workflow.eqBandGainTitle")}
              onChange={(e) => updateParams({ eqHpf: Number(e.target.value) })}
            >
              <option value={0}>{t("workflow.eqOff")}</option>
              <option value={1}>{t("workflow.eqOn")}</option>
            </select>
          </div>
          {GAIN_BANDS.map((b) => (
            <ParamSliderWithUnit
              key={b.key}
              label={t(b.labelKey)}
              unitType="db"
              title={t(b.titleKey)}
              min={-12} max={12} step={0.5} value={((params[b.key] as number) ?? 0)}
              onChange={(v) => updateParams({ [b.key]: v })}
            />
          ))}
          <div style={{ fontSize: 10, color: "var(--text-muted)", padding: "4px 0 0 4px" }}>
            {t("workflow.busEqHint")}
          </div>
        </div>
      </div>
    </NodeShell>
  );
}
