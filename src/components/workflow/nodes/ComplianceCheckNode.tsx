/**
 * ComplianceCheckNode — 母带合规体检.
 * 引擎: engine.ts case "complianceCheck" — 音频原样透传 (端口 0),
 *       合规报告 JSON 输出到端口 1.
 * 后端: compliance_check — 集成响度 (LUFS) / 真峰 (dBTP) / 削波段数 /
 *       相位相关性 / 直流偏移 / 首尾静音时长.
 */
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";

export function ComplianceCheckNode(props: NodeProps) {
  const { t } = useTranslation();

  return (
    <NodeShell
      nodeId={props.id}
      label={t("workflow.nodeComplianceCheck")}
      icon="🩺"
      color="#22c55e"
      inputs={1}
      outputs={2}
      outputLabels={[t("workflow.complianceOutAudio"), t("workflow.complianceOutReport")]}
    >
      <div className="sep-node-body">
        <div style={{ fontSize: 10, color: "var(--text-muted)", padding: "4px 0 0 4px" }}>
          {t("workflow.complianceHint")}
        </div>
      </div>
    </NodeShell>
  );
}
