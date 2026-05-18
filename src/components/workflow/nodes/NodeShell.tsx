import { Handle, Position } from "@xyflow/react";
import type { ReactNode } from "react";
import "./NodeShell.css";

interface Props {
  label: string;
  icon: string;
  color: string;
  inputs?: number;
  outputs?: number;
  children?: ReactNode;
}

export function NodeShell({ label, icon, color, inputs = 1, outputs = 1, children }: Props) {
  return (
    <div className="wf-node" style={{ "--node-color": color } as React.CSSProperties}>
      <div className="wf-node-header">
        <span className="wf-node-icon">{icon}</span>
        <span className="wf-node-label">{label}</span>
      </div>
      {children && <div className="wf-node-body">{children}</div>}
      {Array.from({ length: inputs }).map((_, i) => (
        <Handle
          key={`in-${i}`}
          type="target"
          position={Position.Left}
          id={`in-${i}`}
          style={{ top: `${((i + 1) / (inputs + 1)) * 100}%` }}
        />
      ))}
      {Array.from({ length: outputs }).map((_, i) => (
        <Handle
          key={`out-${i}`}
          type="source"
          position={Position.Right}
          id={`out-${i}`}
          style={{ top: `${((i + 1) / (outputs + 1)) * 100}%` }}
        />
      ))}
    </div>
  );
}
