import type { NodeProps } from "@xyflow/react";
import { NodeShell } from "./NodeShell";

export function RvcNode(props: NodeProps) {
  return (
    <NodeShell nodeId={props.id} label="RVC" icon="[R]" color="#39c5bb" inputs={1} outputs={1}>
      <label>Model</label>
      <select>
        <option value="">Select voice...</option>
      </select>
      <label>Pitch (st)</label>
      <input type="number" defaultValue={0} min={-24} max={24} step={1} />
    </NodeShell>
  );
}
