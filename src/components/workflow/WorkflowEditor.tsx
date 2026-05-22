import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  addEdge,
  useNodesState,
  useEdgesState,
  type Connection,
  type Edge,
  type Node,
  type NodeTypes,
  type ReactFlowInstance,
  type NodeMouseHandler,
  BackgroundVariant,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { AudioInputNode } from "./nodes/AudioInputNode";
import { AudioOutputNode } from "./nodes/AudioOutputNode";
import { RvcNode } from "./nodes/RvcNode";
import { SoVitsNode } from "./nodes/SoVitsNode";
import { SeparationNode } from "./nodes/SeparationNode";
import { EffectsNode } from "./nodes/EffectsNode";
import { NodePalette } from "./NodePalette";
import { useProjectStore } from "../../store/project";
import { useWorkflowStore } from "../../store/workflow";
import { useMsstModelStore } from "../../store/msst-models";
import { executeWorkflow, executeSingleNode } from "../../lib/workflow/engine";
import { clearBufferCache, loadAudioBuffer } from "../../lib/audio/playback";
import { ContextMenu, type MenuItem } from "../common/ContextMenu";
import { useTranslation } from "react-i18next";
import type { Workflow, WorkflowNode as WfNode, WorkflowConnection, WorkflowNodeType } from "../../types/project";
import "./WorkflowEditor.css";

const nodeTypes: NodeTypes = {
  audioInput: AudioInputNode,
  audioOutput: AudioOutputNode,
  rvc: RvcNode,
  sovits: SoVitsNode,
  separation: SeparationNode,
  effects: EffectsNode,
  // Legacy — kept for loading old workflows
  msst: SeparationNode,
  pitchShift: EffectsNode,
  formantShift: EffectsNode,
  audioEnhance: EffectsNode,
};

const rfTypeToWfType: Record<string, WorkflowNodeType> = {
  audioInput: "input",
  audioOutput: "output",
  rvc: "rvc",
  sovits: "sovits",
  separation: "msstSeparation",
  effects: "pitchShift",
  msst: "msstSeparation",
  pitchShift: "pitchShift",
  formantShift: "formantShift",
  audioEnhance: "audioEnhance",
};

const wfTypeToRfType: Record<string, string> = {
  input: "audioInput",
  output: "audioOutput",
  rvc: "rvc",
  sovits: "sovits",
  msstSeparation: "separation",
  pitchShift: "effects",
  formantShift: "effects",
  audioEnhance: "effects",
};

interface Props {
  segmentId: string;
  onClose: () => void;
}

let nodeCounter = 0;

function workflowToReactFlow(wf: Workflow): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = wf.nodes.map((n, i) => ({
    id: n.id,
    type: wfTypeToRfType[n.nodeType] ?? n.nodeType,
    position: { x: n.position.x, y: n.position.y },
    data: { label: n.nodeType, params: n.params },
    deletable: n.nodeType !== "input" && n.nodeType !== "output",
    zIndex: i,
  }));
  const edges: Edge[] = wf.connections.map((c, i) => ({
    id: `e-${i}`,
    source: c.fromNode,
    sourceHandle: `out-${c.fromPort}`,
    target: c.toNode,
    targetHandle: `in-${c.toPort}`,
    animated: true,
  }));
  return { nodes, edges };
}

function reactFlowToWorkflow(nodes: Node[], edges: Edge[]): Workflow {
  const wfNodes: WfNode[] = nodes.map((n) => ({
    id: n.id,
    nodeType: rfTypeToWfType[n.type ?? ""] ?? (n.type as WorkflowNodeType),
    position: { x: n.position.x, y: n.position.y },
    params: (n.data?.params as Record<string, unknown>) ?? {},
  }));
  const wfConns: WorkflowConnection[] = edges.map((e) => ({
    fromNode: e.source,
    fromPort: parseInt(e.sourceHandle?.replace("out-", "") ?? "0", 10),
    toNode: e.target,
    toPort: parseInt(e.targetHandle?.replace("in-", "") ?? "0", 10),
  }));
  return { nodes: wfNodes, connections: wfConns };
}

const defaultNodes: Node[] = [
  {
    id: "input-1",
    type: "audioInput",
    position: { x: 50, y: 200 },
    data: { label: "Audio In" },
    deletable: false,
    zIndex: 0,
  },
  {
    id: "output-1",
    type: "audioOutput",
    position: { x: 600, y: 200 },
    data: { label: "Output", params: { laneLabel: "Main" } },
    deletable: false,
    zIndex: 1,
  },
];

export function WorkflowEditor({ segmentId, onClose }: Props) {
  const { t } = useTranslation();
  const { tracks, setProcessedOutputs } = useProjectStore();
  const updateTrackRef = useRef(useProjectStore.getState().updateTrack);
  updateTrackRef.current = useProjectStore.getState().updateTrack;
  const executionState = useWorkflowStore((s) => s.executions[segmentId]);

  useEffect(() => {
    useMsstModelStore.getState().fetchModelsDir();
    useMsstModelStore.getState().fetchInstalled();
  }, []);

  const segment = tracks
    .flatMap((tr) => tr.segments.map((s) => ({ trackId: tr.id, seg: s })))
    .find((x) => x.seg.id === segmentId);

  const initialData = segment?.seg.workflow
    ? workflowToReactFlow(segment.seg.workflow)
    : { nodes: defaultNodes, edges: [] as Edge[] };

  const [nodes, setNodes, onNodesChange] = useNodesState(initialData.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialData.edges);
  const saveTimer = useRef<ReturnType<typeof setTimeout>>(undefined);
  const segTrackIdRef = useRef(segment?.trackId);
  segTrackIdRef.current = segment?.trackId;
  const [rfInstance, setRfInstance] = useState<ReactFlowInstance | null>(null);
  const [nodeCtx, setNodeCtx] = useState<{ x: number; y: number; nodeId: string } | null>(null);

  useEffect(() => {
    const trackId = segTrackIdRef.current;
    if (!trackId) return;
    clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(() => {
      const wf = reactFlowToWorkflow(nodes, edges);
      const currentTracks = useProjectStore.getState().tracks;
      const track = currentTracks.find((tr) => tr.id === trackId);
      if (!track) return;
      const updatedSegments = track.segments.map((s) =>
        s.id === segmentId ? { ...s, workflow: wf } : s,
      );
      updateTrackRef.current(trackId, { segments: updatedSegments });
    }, 300);
    return () => clearTimeout(saveTimer.current);
  }, [nodes, edges, segmentId]);

  const onConnect = useCallback(
    (connection: Connection) => {
      setEdges((eds) => addEdge({ ...connection, animated: true }, eds));
    },
    [setEdges],
  );

  const onAddNode = useCallback(
    (type: string, label: string, extraParams?: Record<string, unknown>) => {
      nodeCounter++;
      const id = `${type}-${crypto.randomUUID().slice(0, 8)}`;
      const defaultParams: Record<string, unknown> = { ...(extraParams ?? {}) };
      if (type === "audioOutput") defaultParams.laneLabel = "Output";
      const newNode: Node = {
        id,
        type,
        position: { x: 300 + Math.random() * 100, y: 150 + Math.random() * 100 },
        data: { label, params: defaultParams },
        zIndex: nodeCounter + 100,
      };
      setNodes((nds) => [...nds, newNode]);
    },
    [setNodes],
  );

  const onDropNode = useCallback(
    (type: string, label: string, clientX: number, clientY: number, extraParams?: Record<string, unknown>) => {
      if (!rfInstance) return;
      // Check if drop is within the canvas area
      const canvasEl = document.querySelector(".workflow-canvas");
      if (!canvasEl) return;
      const rect = canvasEl.getBoundingClientRect();
      if (clientX < rect.left || clientX > rect.right || clientY < rect.top || clientY > rect.bottom) return;

      const position = rfInstance.screenToFlowPosition({ x: clientX, y: clientY });
      nodeCounter++;
      const id = `${type}-${crypto.randomUUID().slice(0, 8)}`;
      const defaultParams: Record<string, unknown> = { ...(extraParams ?? {}) };
      if (type === "audioOutput") defaultParams.laneLabel = "Output";
      const newNode: Node = {
        id,
        type,
        position,
        data: { label, params: defaultParams },
        zIndex: nodeCounter + 100,
      };
      setNodes((nds) => [...nds, newNode]);
    },
    [setNodes, rfInstance],
  );


  const onNodeContextMenu: NodeMouseHandler = useCallback((event, node) => {
    event.preventDefault();
    if (node.deletable === false) return;
    setNodeCtx({ x: event.clientX, y: event.clientY, nodeId: node.id });
  }, []);

  const handleDeleteNode = useCallback((nodeId: string) => {
    setNodes((nds) => nds.filter((n) => n.id !== nodeId));
    setEdges((eds) => eds.filter((e) => e.source !== nodeId && e.target !== nodeId));
    setNodeCtx(null);
  }, [setNodes, setEdges]);

  const nodeCtxItems: MenuItem[] = nodeCtx ? [
    { label: t("toolbar.delete"), shortcut: "Del", danger: true, onClick: () => handleDeleteNode(nodeCtx.nodeId) },
  ] : [];

  const handleExecute = useCallback(async () => {
    const trackId = segTrackIdRef.current;
    if (!segment || !trackId) return;
    const wf = reactFlowToWorkflow(nodes, edges);
    try {
      const outputs = await executeWorkflow(segmentId, segment.seg, wf);
      for (const out of outputs) {
        clearBufferCache(out.audioPath);
        loadAudioBuffer(out.audioPath);
      }
      setProcessedOutputs(trackId, segmentId, outputs);
      if (!useProjectStore.getState().tracks.find(t => t.id === trackId)?.expanded) {
        useProjectStore.getState().toggleTrackExpanded(trackId);
      }
    } catch (err) {
      if (err instanceof Error && err.message === "Cancelled") return;
      console.error("Workflow execution failed:", err);
    }
  }, [segmentId, segment, nodes, edges, setProcessedOutputs]);

  const handleCancel = useCallback(() => {
    useWorkflowStore.getState().cancelExecution(segmentId);
    invoke("cancel_separation").catch(() => {});
  }, [segmentId]);

  const nodesRef = useRef(nodes);
  nodesRef.current = nodes;
  const edgesRef = useRef(edges);
  edgesRef.current = edges;
  const segmentRef = useRef(segment);
  segmentRef.current = segment;

  const handleRunSingleNode = useCallback((nodeId: string) => {
    const seg = segmentRef.current;
    if (!seg) return;
    const wf = reactFlowToWorkflow(nodesRef.current, edgesRef.current);
    executeSingleNode(segmentId, seg.seg, wf, nodeId).catch((err) => {
      console.error("Single node execution failed:", err);
    });
  }, [segmentId]);

  useEffect(() => {
    useWorkflowStore.getState().registerSingleNodeRunner(handleRunSingleNode);
    return () => useWorkflowStore.getState().registerSingleNodeRunner(null);
  }, [handleRunSingleNode]);

  const isRunning = executionState?.status === "running";

  return (
    <div className="workflow-editor">
      <div className="workflow-header">
        <span className="workflow-title">
          {t("workflow.title")} — {segmentId.slice(0, 8)}
        </span>
        <div className="workflow-header-actions">
          {executionState?.status === "error" && (
            <span className="wf-error">{executionState.error}</span>
          )}
          {isRunning ? (
            <button className="wf-run-btn running" onClick={handleCancel}>
              Stop
            </button>
          ) : (
            <button className="wf-run-btn" onClick={handleExecute}>
              Run
            </button>
          )}
          <button className="workflow-close" onClick={onClose}>
            x
          </button>
        </div>
      </div>
      <div className="workflow-body">
        <NodePalette onAddNode={onAddNode} onDropNode={onDropNode} />
        <div className="workflow-canvas">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeContextMenu={onNodeContextMenu}
            onPaneClick={() => setNodeCtx(null)}
            onNodeClick={() => setNodeCtx(null)}
            onMoveStart={() => setNodeCtx(null)}
            onInit={setRfInstance}
            nodeTypes={nodeTypes}
            fitView
            deleteKeyCode="Delete"
            defaultEdgeOptions={{
              style: { stroke: "var(--accent-primary)", strokeWidth: 2 },
              animated: true,
            }}
            proOptions={{ hideAttribution: true }}
          >
            <Background
              variant={BackgroundVariant.Dots}
              gap={20}
              size={1}
              color="rgba(57, 197, 187, 0.1)"
            />
            <Controls
              showInteractive={false}
              style={{ background: "var(--bg-panel)", borderColor: "var(--border-default)" }}
            />
            <MiniMap
              style={{ background: "var(--bg-base)" }}
              nodeColor="var(--accent-primary-dim)"
              maskColor="rgba(13, 18, 32, 0.7)"
            />
          </ReactFlow>
          {nodeCtx && <ContextMenu x={nodeCtx.x} y={nodeCtx.y} items={nodeCtxItems} onClose={() => setNodeCtx(null)} />}
        </div>
      </div>
    </div>
  );
}
