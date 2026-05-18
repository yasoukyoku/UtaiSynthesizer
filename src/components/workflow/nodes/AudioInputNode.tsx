import type { NodeProps } from "@xyflow/react";
import { NodeShell } from "./NodeShell";

export function AudioInputNode(_props: NodeProps) {
  return (
    <NodeShell label="Audio Input" icon="🎵" color="#60a5fa" inputs={0} outputs={1}>
      <span style={{ fontSize: "10px", color: "var(--text-muted)" }}>Segment audio source</span>
    </NodeShell>
  );
}
