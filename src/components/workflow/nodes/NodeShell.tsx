import { Handle, Position, useInternalNode } from "@xyflow/react";
import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { useWorkflowStore, type NodeStatus } from "../../../store/workflow";
import { useAppStore } from "../../../store/app";
import { t18 } from "../../../lib/models/msst-catalog";
import { NODE_DAMAGE, DAMAGE_ICONS } from "../../../lib/workflow/damage";
import { rfTypeToWfType } from "../../../lib/workflow/rfTypes";
import { inputPortKind, outputPortKind } from "../../../lib/workflow/ports";
import { NodeOutputPreview } from "./NodeOutputPreview";
import { NodeAnnotationBadge } from "../NodeAnnotation";
import type { NodeAnnotation } from "../../../lib/workflow/nodeAnnotations";
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
  /** Explicit card width in px. Omit to keep the default content-driven sizing (180–260px).
   *  Set it only for cards whose body genuinely needs more room (textareas, preset button grids). */
  width?: number;
}

export function NodeShell({ nodeId, label, icon, color, inputs = 1, outputs = 1, outputLabels, children, onRunNode, runTitle, noRunButton, width }: Props) {
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

  // 规划 6-4/6-5: 借内部节点反查域类型 → 损伤等级（徽标）；data.bypass 是持久图属性
  // （旁通态与是否在跑无关），status === "bypassed" 则是本轮运行的预标/终标 —— 两者都算旁通。
  const internal = useInternalNode(nodeId ?? "");
  const wfType = internal?.type ? rfTypeToWfType[internal.type] : undefined;
  const damage = wfType ? NODE_DAMAGE[wfType] : undefined;
  const bypassed = (internal?.data as { bypass?: boolean } | undefined)?.bypass === true || status === "bypassed";
  const annotation = (internal?.data as { annotation?: NodeAnnotation } | undefined)?.annotation;
  const annotationEditor = useWorkflowStore((s) => s.annotationEditor);

  const statusClass = [status !== "idle" ? `wf-node-${status}` : "", bypassed ? "wf-node-bypassed" : ""]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={`wf-node ${statusClass}`} style={{ "--node-color": color, ...(width ? { "--node-width": `${width}px` } : {}) } as React.CSSProperties}>
      <div className="wf-node-header">
        <span className="wf-node-icon">{icon}</span>
        <span className="wf-node-label">{label}</span>
        <div className="wf-node-header-spacer" />
        {damage && damage.level > 0 && (
          <span
            className="wf-node-damage"
            title={t18({ zh: `保真度等级 ${damage.level} · 单级 SNR ≈ ${damage.snrDb}dB`, en: `Fidelity tier ${damage.level} · per-node SNR ≈ ${damage.snrDb}dB`, ja: `忠実度レベル ${damage.level} · 単体 SNR ≈ ${damage.snrDb}dB` }, i18n.language)}
          >
            {DAMAGE_ICONS[damage.level]}
          </span>
        )}
        {bypassed && (
          <span className="wf-node-bypass-badge" title={t18({ zh: "旁通中 —— 引擎跳过此节点，输入原样透传", en: "Bypassed — engine skips this node, input passes through", ja: "バイパス中 —— エンジンはこのノードをスキップし入力をそのまま通す" }, i18n.language)}>
            ⏸
          </span>
        )}
        {annotation && annotationEditor && nodeId && (
          <NodeAnnotationBadge
            annotation={annotation}
            onClick={() => annotationEditor(nodeId)}
          />
        )}
        {status === "running" && progress > 0 && (
          <span className="wf-node-pct">{Math.round(progress * 100)}%</span>
        )}
        {status === "running" && <span className="wf-node-pulse" />}
        {status === "completed" && <span className="wf-node-done">OK</span>}
        {status === "degraded" && <span className="wf-node-warn" title={error ?? undefined}>!</span>}
        {status === "error" && <span className="wf-node-err" title={error}>!!</span>}
        {resolvedRunNode && !noRunButton && !locked && !bypassed && status !== "running" && (
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
      {/* S66 output audition: hidden while running (the paths are being replaced). */}
      {nodeId && status !== "running" && (
        <NodeOutputPreview statusSeg={statusSeg} nodeId={nodeId} outputLabels={outputLabels} />
      )}
      {Array.from({ length: inputs }).map((_, i) => {
        const portType = wfType ? inputPortKind(wfType, i) : "any";
        return (
          <Handle
            key={`in-${i}`}
            type="target"
            position={Position.Left}
            id={`in-${i}`}
            style={{ top: `${((i + 1) / (inputs + 1)) * 100}%` }}
            data-port-type={portType}
          />
        );
      })}
      {Array.from({ length: outCount }).map((_, i) => {
        const portType = wfType ? outputPortKind(wfType, i) : "any";
        return (
          <Handle
            key={`out-${i}`}
            type="source"
            position={Position.Right}
            id={`out-${i}`}
            style={{ top: `${((i + 1) / (outCount + 1)) * 100}%` }}
            data-port-type={portType}
          />
        );
      })}
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
