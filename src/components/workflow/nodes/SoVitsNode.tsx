import type { NodeProps } from "@xyflow/react";
import { NodeShell } from "./NodeShell";

export function SoVitsNode(_props: NodeProps) {
  return (
    <NodeShell label="SoVITS" icon="✨" color="#8b5cf6" inputs={1} outputs={1}>
      <label>Model</label>
      <select>
        <option value="">Select voice...</option>
      </select>
      <label>Pitch (st)</label>
      <input type="number" defaultValue={0} min={-24} max={24} step={1} />
      <label>
        <input type="checkbox" defaultChecked /> Shallow Diffusion
      </label>
    </NodeShell>
  );
}
