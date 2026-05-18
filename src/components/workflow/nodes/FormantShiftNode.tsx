import type { NodeProps } from "@xyflow/react";
import { NodeShell } from "./NodeShell";

export function FormantShiftNode(_props: NodeProps) {
  return (
    <NodeShell label="Formant Shift" icon="🔄" color="#f97316" inputs={1} outputs={1}>
      <label>Ratio</label>
      <input type="number" defaultValue={1.0} min={0.5} max={2.0} step={0.05} />
      <label>Vocoder</label>
      <select>
        <option value="world">WORLD (fast)</option>
        <option value="nsf">NSF-HiFiGAN (quality)</option>
      </select>
    </NodeShell>
  );
}
