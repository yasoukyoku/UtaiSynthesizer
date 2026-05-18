import { useCallback } from "react";
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
  BackgroundVariant,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { AudioInputNode } from "./nodes/AudioInputNode";
import { AudioOutputNode } from "./nodes/AudioOutputNode";
import { RvcNode } from "./nodes/RvcNode";
import { SoVitsNode } from "./nodes/SoVitsNode";
import { PitchShiftNode } from "./nodes/PitchShiftNode";
import { FormantShiftNode } from "./nodes/FormantShiftNode";
import { AudioEnhanceNode } from "./nodes/AudioEnhanceNode";
import { MsstNode } from "./nodes/MsstNode";
import { NodePalette } from "./NodePalette";
import { useTranslation } from "react-i18next";
import "./WorkflowEditor.css";

const nodeTypes: NodeTypes = {
  audioInput: AudioInputNode,
  audioOutput: AudioOutputNode,
  rvc: RvcNode,
  sovits: SoVitsNode,
  pitchShift: PitchShiftNode,
  formantShift: FormantShiftNode,
  audioEnhance: AudioEnhanceNode,
  msst: MsstNode,
};

interface Props {
  segmentId: string;
  onClose: () => void;
}

const defaultNodes: Node[] = [
  {
    id: "input-1",
    type: "audioInput",
    position: { x: 50, y: 200 },
    data: { label: "Audio In" },
    deletable: false,
  },
  {
    id: "output-1",
    type: "audioOutput",
    position: { x: 600, y: 200 },
    data: { label: "Output" },
    deletable: false,
  },
];

const defaultEdges: Edge[] = [];

export function WorkflowEditor({ segmentId, onClose }: Props) {
  const { t } = useTranslation();
  const [nodes, setNodes, onNodesChange] = useNodesState(defaultNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(defaultEdges);

  const onConnect = useCallback(
    (connection: Connection) => {
      setEdges((eds) => addEdge({ ...connection, animated: true }, eds));
    },
    [setEdges]
  );

  const onAddNode = useCallback(
    (type: string, label: string) => {
      const id = `${type}-${Date.now()}`;
      const newNode: Node = {
        id,
        type,
        position: { x: 300 + Math.random() * 100, y: 150 + Math.random() * 100 },
        data: { label },
      };
      setNodes((nds) => [...nds, newNode]);
    },
    [setNodes]
  );

  return (
    <div className="workflow-editor">
      <div className="workflow-header">
        <span className="workflow-title">
          {t("workflow.title")} — {segmentId}
        </span>
        <button className="workflow-close" onClick={onClose}>
          ✕
        </button>
      </div>
      <div className="workflow-body">
        <NodePalette onAddNode={onAddNode} />
        <div className="workflow-canvas">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            nodeTypes={nodeTypes}
            fitView
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
        </div>
      </div>
    </div>
  );
}
