import { Handle, Position } from "@xyflow/react";
import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { useWorkflowStore, type NodeStatus } from "../../../store/workflow";
import { useAppStore } from "../../../store/app";
import { t18 } from "../../../lib/models/msst-catalog";
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
  /** Tooltip for the run button. Output nodes set this to label it "deposit" rather than "run". */
  runTitle?: string;
  /** Suppress the per-node run button entirely (Output nodes auto-deposit — no manual run/deposit). */
  noRunButton?: boolean;
}

export function NodeShell({ nodeId, label, icon, color, inputs = 1, outputs = 1, outputLabels, children, onRunNode, runTitle, noRunButton }: Props) {
  const { i18n } = useTranslation();
  const outCount = outputLabels ? outputLabels.length : outputs;
  const segmentId = useAppStore((s) => s.workflowSegmentId);
  // A pending split-mid-render LINK target mirrors its render SOURCE's node badges (same node ids — the
  // graph was copied on split) and LOCKS its run buttons: it doesn't run its own copy of the nodes (the
  // source's single render feeds both halves via RenderLinkWatcher). renderLinks is empty in the common
  // case → statusSeg === segmentId, locked === false (the extra subscription only re-renders on a link
  // appearing/clearing, which is rare).
  const linkedSource = useWorkflowStore((s) => (segmentId ? s.renderLinks[segmentId] : undefined));
  const statusSeg = linkedSource ?? segmentId;
  const locked = linkedSource !== undefined;
  // Subscribe to THIS node's status/progress/error specifically (primitive selectors) so a progress
  // tick on one node doesn't re-render every other node card on the canvas.
  const status: NodeStatus = useWorkflowStore((s) => (statusSeg && nodeId ? s.nodeStatuses[statusSeg]?.[nodeId] : undefined)) ?? "idle";
  const progress = useWorkflowStore((s) => (statusSeg && nodeId ? s.nodeProgress[statusSeg]?.[nodeId] : undefined)) ?? 0;
  const error = useWorkflowStore((s) => (statusSeg && nodeId ? s.nodeErrors[statusSeg]?.[nodeId] : undefined));
  const singleNodeRunner = useWorkflowStore((s) => s.singleNodeRunner);

  const resolvedRunNode = onRunNode ?? (nodeId && singleNodeRunner ? () => singleNodeRunner(nodeId) : undefined);

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
        {resolvedRunNode && !noRunButton && !locked && status !== "running" && (
          <button className="wf-node-run-btn" onClick={(e) => { e.stopPropagation(); resolvedRunNode(); }} title={runTitle ?? t18({ zh: "运行此节点", en: "Run this node", ja: "このノードを実行" }, i18n.language)}>
            &gt;
          </button>
        )}
      </div>
      {status === "running" && progress > 0 && (
        <div className="wf-node-progress">
          <div className="wf-node-progress-fill" style={{ width: `${Math.round(progress * 100)}%` }} />
        </div>
      )}
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
