import { Handle, Position } from "@xyflow/react";
import type { ReactNode } from "react";
import { useWorkflowStore, type NodeStatus } from "../../../store/workflow";
import { useAppStore } from "../../../store/app";
import "./NodeShell.css";

interface Props {
  nodeId?: string;
  label: string;
  icon: string;
  color: string;
  inputs?: number;
  outputs?: number;
  outputLabels?: string[];
  children?: ReactNode;
  onRunNode?: () => void;
}

export function NodeShell({ nodeId, label, icon, color, inputs = 1, outputs = 1, outputLabels, children, onRunNode }: Props) {
  const outCount = outputLabels ? outputLabels.length : outputs;
  const segmentId = useAppStore((s) => s.workflowSegmentId);
  const nodeStatuses = useWorkflowStore((s) => s.nodeStatuses);
  const nodeErrors = useWorkflowStore((s) => s.nodeErrors);
  const singleNodeRunner = useWorkflowStore((s) => s.singleNodeRunner);

  const resolvedRunNode = onRunNode ?? (nodeId && singleNodeRunner ? () => singleNodeRunner(nodeId) : undefined);

  const status: NodeStatus = (segmentId && nodeId ? nodeStatuses[segmentId]?.[nodeId] : undefined) ?? "idle";
  const error = segmentId && nodeId ? nodeErrors[segmentId]?.[nodeId] : undefined;

  const statusClass = status !== "idle" ? `wf-node-${status}` : "";

  return (
    <div className={`wf-node ${statusClass}`} style={{ "--node-color": color } as React.CSSProperties}>
      <div className="wf-node-header">
        <span className="wf-node-icon">{icon}</span>
        <span className="wf-node-label">{label}</span>
        <div className="wf-node-header-spacer" />
        {status === "running" && <span className="wf-node-pulse" />}
        {status === "completed" && <span className="wf-node-done">OK</span>}
        {status === "error" && <span className="wf-node-err" title={error}>!!</span>}
        {resolvedRunNode && status !== "running" && (
          <button className="wf-node-run-btn" onClick={(e) => { e.stopPropagation(); resolvedRunNode(); }} title="Run this node">
            &gt;
          </button>
        )}
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
      {Array.from({ length: outCount }).map((_, i) => (
        <Handle
          key={`out-${i}`}
          type="source"
          position={Position.Right}
          id={`out-${i}`}
          style={{ top: `${((i + 1) / (outCount + 1)) * 100}%` }}
        />
      ))}
      {outputLabels && (
        <div className="wf-node-out-labels">
          {outputLabels.map((lbl, i) => (
            <span
              key={i}
              className="wf-out-label"
              style={{ top: `${((i + 1) / (outCount + 1)) * 100}%` }}
            >
              {lbl}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
