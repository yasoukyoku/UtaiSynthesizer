import type { NodeProps } from "@xyflow/react";
import { NodeShell } from "./NodeShell";

export function AudioOutputNode(_props: NodeProps) {
  return (
    <NodeShell label="Output" icon="🔊" color="#4ade80" inputs={1} outputs={0}>
      <span style={{ fontSize: "10px", color: "var(--text-muted)" }}>→ Arrangement lane</span>
    </NodeShell>
  );
}
