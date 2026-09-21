/**
 * StructureEditNode — 曲式编辑 (P2-7, 规划 2-7).
 * 引擎: engine.ts case "structureEdit" — editStructure: 尾部反复/删除/移调/延长;
 * 端口0 = chordBlock, 端口1 = chordDetect 兼容分段.
 */
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";

const OP_KEYS = ["symOpRepeatTail", "symOpDropTail", "symOpTransposeTail", "symOpLengthenTail"];

export function StructureEditNode(props: NodeProps) {
  const { t } = useTranslation();
  const [params, updateParams] = useNodeParams(props);
  const op = (params.op as number) ?? 0;
  const sectionBars = (params.sectionBars as number) ?? 4;
  const semitones = (params.semitones as number) ?? 2;

  return (
    <NodeShell
      nodeId={props.id}
      label={t("workflow.nodeStructureEdit")}
      icon="🏗"
      color="#facc15"
      inputs={1}
      outputs={2}
      outputLabels={["chords", "info"]}
    >
      <div className="sep-node-body">
        <div className="sep-params">
          <div className="sep-label-row">
            <span className="sep-label">{t("workflow.symOp")}</span>
            <select
              className="sep-select"
              value={String(op)}
              onChange={(e) => updateParams({ op: Number(e.target.value) })}
            >
              {OP_KEYS.map((k, i) => (
                <option key={k} value={i}>{t(`workflow.${k}`)}</option>
              ))}
            </select>
          </div>
          <ParamSlider
            label={t("workflow.symSectionBars")}
            min={1} max={8} step={1} value={sectionBars}
            format={(v) => `${v}`}
            onChange={(v) => updateParams({ sectionBars: v })}
          />
          <ParamSlider
            label={t("workflow.symSemitones")}
            min={-12} max={12} step={1} value={semitones}
            format={(v) => (v > 0 ? `+${v}` : `${v}`)}
            onChange={(v) => updateParams({ semitones: v })}
          />
          <div style={{ fontSize: 10, color: "#888", padding: "4px 0 0 4px" }}>
            {t("workflow.symStructureHint")}
          </div>
        </div>
      </div>
    </NodeShell>
  );
}
