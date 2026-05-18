import type { NodeProps } from "@xyflow/react";
import { NodeShell } from "./NodeShell";

export function PitchShiftNode(_props: NodeProps) {
  return (
    <NodeShell label="Pitch Shift" icon="↕" color="#fbbf24" inputs={1} outputs={1}>
      <label>Semitones</label>
      <input type="number" defaultValue={0} min={-24} max={24} step={0.5} />
      <label>Vocoder</label>
      <select>
        <option value="world">WORLD (fast)</option>
        <option value="nsf">NSF-HiFiGAN (quality)</option>
      </select>
    </NodeShell>
  );
}
