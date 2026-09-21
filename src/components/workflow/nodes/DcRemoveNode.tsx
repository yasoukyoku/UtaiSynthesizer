/**
 * DcRemoveNode — 直流偏移去除 (Phase 4 配方 F ①, 母带链第一环).
 * 引擎: engine.ts case "dcRemove" — 测量并移除直流, 防止余量损失与咔哒噪声.
 * 后端: remove_dc — measure + remove (report 返回前后 DC 值).
 */
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";

export function DcRemoveNode(props: NodeProps) {
  const { t } = useTranslation();
  useNodeParams(props);

  return (
    <NodeShell
      nodeId={props.id}
      label={t("workflow.nodeDcRemove")}
      icon="🧹"
      color="#94a3b8"
      inputs={1}
      outputs={1}
      outputLabels={["audio"]}
    >
      <div className="sep-node-body">
        <div className="sep-params">
          <div style={{ fontSize: 10, color: "var(--text-muted)", padding: "4px 0 0 4px" }}>
            {t("workflow.dcRemoveHint")}
          </div>
        </div>
      </div>
    </NodeShell>
  );
}
