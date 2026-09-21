import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import type { Node } from "@xyflow/react";
import { NODE_PORTS, inputPortKind, outputPortKind, portsCompatible } from "../../lib/workflow/ports";
import type { PortKind } from "../../lib/workflow/ports";
import { rfTypeToWfType } from "../../lib/workflow/rfTypes";
import "./SmartConnectionHelper.css";

interface Props {
  nodes: Node[];
  sourceNode?: string;
  sourceHandle?: string;
}

const PORT_COLORS: Record<PortKind, string> = {
  audio: "#4CAF50",
  midi: "#2196F3",
  chords: "#FF9800",
  lyrics: "#E91E63",
  report: "#9C27B0",
  any: "#9E9E9E",
};

export function SmartConnectionHelper({ nodes, sourceNode, sourceHandle }: Props) {
  const { t } = useTranslation();

  const hint = useMemo(() => {
    if (!sourceNode || !sourceHandle) return null;

    const source = nodes.find((n) => n.id === sourceNode);
    if (!source) return null;

    const sourceWfType = rfTypeToWfType[source.type ?? ""];
    if (!sourceWfType) return null;

    const sourcePort = Number.parseInt(sourceHandle.replace(/^out-/, ""), 10);
    if (!Number.isFinite(sourcePort)) return null;

    const sourcePortType = outputPortKind(sourceWfType, sourcePort);

    // 统计能接住这根线的入端口数。走 NODE_PORTS 权威表(与 isValidConnection 同源),
    // 口径必须和真正的落线校验一致 —— 否则提示"3 个兼容"而实际只能落 1 根,比没提示更糟。
    let count = 0;
    for (const node of nodes) {
      if (node.id === sourceNode) continue;
      const wfType = rfTypeToWfType[node.type ?? ""];
      if (!wfType) continue;
      const inputs = NODE_PORTS[wfType]?.inputs.length ?? 0;
      for (let port = 0; port < inputs; port++) {
        if (portsCompatible(sourcePortType, inputPortKind(wfType, port))) count++;
      }
    }

    return { sourcePortType, count };
  }, [nodes, sourceNode, sourceHandle]);

  if (!hint) return null;

  return (
    <div className="smart-connection-overlay">
      <div className="smart-connection-hint">
        <span
          className="smart-connection-port-indicator"
          style={{ backgroundColor: PORT_COLORS[hint.sourcePortType] }}
        >
          {t(`workflow.portKind.${hint.sourcePortType}`)}
        </span>
        <span className="smart-connection-arrow">→</span>
        <span className="smart-connection-text">
          {hint.count > 0
            ? t("workflow.smartConnection.compatibleCount", { count: hint.count })
            : t("workflow.smartConnection.noCompatible")}
        </span>
      </div>
    </div>
  );
}
