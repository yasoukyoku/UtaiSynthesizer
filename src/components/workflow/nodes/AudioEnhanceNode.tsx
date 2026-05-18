import type { NodeProps } from "@xyflow/react";
import { NodeShell } from "./NodeShell";

export function AudioEnhanceNode(_props: NodeProps) {
  return (
    <NodeShell label="Audio Enhance" icon="💎" color="#a78bfa" inputs={1} outputs={1}>
      <span style={{ fontSize: "10px", color: "var(--text-muted)" }}>
        NSF-HiFiGAN post-processing
      </span>
    </NodeShell>
  );
}
