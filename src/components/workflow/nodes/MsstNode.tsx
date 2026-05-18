import type { NodeProps } from "@xyflow/react";
import { NodeShell } from "./NodeShell";

export function MsstNode(_props: NodeProps) {
  return (
    <NodeShell label="MSST Separate" icon="🔀" color="#ec4899" inputs={1} outputs={4}>
      <label>Model</label>
      <select>
        <option value="htdemucs">HTDemucs (4-stem)</option>
        <option value="vocal">Vocal/Inst (2-stem)</option>
      </select>
      <span style={{ fontSize: "9px", color: "var(--text-muted)" }}>
        Out: Vocal / Drums / Bass / Other
      </span>
    </NodeShell>
  );
}
