/**
 * DitherNode — 抖动量化 (母带最后一环).
 * 引擎: engine.ts case "dither" — 对浮点音频做确定性抖动并量化为 16-bit WAV.
 * 后端: apply_dither — 0=无 / 1=TPDF / 2=TPDF+二阶噪声整形, xorshift64 固定种子 (结果可复现).
 */
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";

export function DitherNode(props: NodeProps) {
  const { t } = useTranslation();
  const [params, updateParams] = useNodeParams(props);
  const ditherType = Math.max(0, Math.min(2, Math.round((params.ditherType as number) ?? 2)));

  return (
    <NodeShell
      nodeId={props.id}
      label={t("workflow.nodeDither")}
      icon="🎚"
      color="#64748b"
      inputs={1}
      outputs={1}
      outputLabels={["16-bit"]}
    >
      <div className="sep-node-body">
        <div className="sep-params">
          <div className="sep-label-row">
            <span className="sep-label">{t("workflow.ditherType")}</span>
            <select
              className="sep-select"
              value={ditherType}
              onChange={(e) => updateParams({ ditherType: Number(e.target.value) })}
            >
              <option value={0}>{t("workflow.ditherNone")}</option>
              <option value={1}>TPDF</option>
              <option value={2}>TPDF + {t("workflow.ditherShaped")}</option>
            </select>
          </div>
          <div style={{ fontSize: 10, color: "var(--text-muted)", padding: "4px 0 0 4px" }}>
            {t("workflow.ditherHint")}
          </div>
        </div>
      </div>
    </NodeShell>
  );
}
