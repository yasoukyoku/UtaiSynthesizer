import { useCallback, useEffect, useRef, useState, type CSSProperties } from "react";
import { invoke } from "@tauri-apps/api/core";
import { flushAutosaveNow } from "../../lib/project/autosave";
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
  type EdgeMouseHandler,
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
import { useAudioStore } from "../../store/audio";
import { useWorkflowStore, type NodeStatus } from "../../store/workflow";
import { useMsstModelStore } from "../../store/msst-models";
import { useAppStore } from "../../store/app";
import { setUndoScope, useHistoryStore } from "../../store/history";
import { DEFAULT_OUTPUT_GROUP } from "../../lib/constants";
import i18n from "../../i18n";
import { executeWorkflow, executeSingleNode, collectCachedPaths, loadCachedOutput, outputLanes, rehydrateRenderState, planDetachGroup, type CachedPath } from "../../lib/workflow/engine";
import { logToBackend } from "../../lib/log";
import { clearBufferCache, loadAudioBuffer } from "../../lib/audio/playback";
import { ContextMenu, type MenuItem } from "../common/ContextMenu";
import { useTranslation } from "react-i18next";
import type { Workflow, WorkflowNode as WfNode, WorkflowConnection, WorkflowNodeType, ProcessedOutput } from "../../types/project";
import "./WorkflowEditor.css";

const nodeTypes: NodeTypes = {
  audioInput: AudioInputNode,
  audioOutput: AudioOutputNode,
  rvc: RvcNode,
  sovits: SoVitsNode,
  separation: SeparationNode,
  effects: EffectsNode,
  // Legacy — kept for loading old workflows. Only "msst" can actually reach ReactFlow: the palette has
  // never emitted the raw effect names, and wfTypeToRfType maps every legacy wf nodeType to "effects"
  // before they get here.
  msst: SeparationNode,
};

const rfTypeToWfType: Record<string, WorkflowNodeType> = {
  audioInput: "input",
  audioOutput: "output",
  rvc: "rvc",
  sovits: "sovits",
  separation: "msstSeparation",
  effects: "pitchShift",
  msst: "msstSeparation",
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
  /** Set by the split container so the docked panel takes a fixed (resizable) height. */
  style?: CSSProperties;
}

let nodeCounter = 0;

function workflowToReactFlow(wf: Workflow): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = wf.nodes.map((n, i) => ({
    id: n.id,
    type: wfTypeToRfType[n.nodeType] ?? n.nodeType,
    position: { x: n.position.x, y: n.position.y },
    data: { label: n.nodeType, params: n.params },
    deletable: n.nodeType !== "input", // Output nodes are deletable now (live deposit — deleting one drops its lanes)
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

/** Structural signature of the node graph for undo diffing — EXCLUDES selection/dimensions (which
 *  must not create undo steps), includes node type/position/params and edge wiring. */
function sigOfGraph(nodes: Node[], edges: Edge[]): string {
  const ns = nodes
    // Node POSITION is intentionally EXCLUDED — moving a node around the canvas is not a meaningful edit and
    // must not create an undo step (position is still persisted via the debounced save, just not undoable).
    .map((n) => `${n.id}:${n.type}:${JSON.stringify(n.data?.params ?? {})}`)
    .sort()
    .join("|");
  const es = edges.map((e) => `${e.source}.${e.sourceHandle}>${e.target}.${e.targetHandle}`).sort().join("|");
  return `${ns}||${es}`;
}

/** Describe (i18n key under "history.") the node-graph op that transforms from→to, for the banner. */
function describeNodeDelta(from: { nodes: Node[]; edges: Edge[] }, to: { nodes: Node[]; edges: Edge[] }): string {
  if (to.nodes.length > from.nodes.length) return "nodeAdd";
  if (to.nodes.length < from.nodes.length) return "nodeRemove";
  if (to.edges.length > from.edges.length) return "nodeConnect";
  if (to.edges.length < from.edges.length) return "nodeDisconnect";
  const fById = new Map(from.nodes.map((n) => [n.id, n]));
  for (const tn of to.nodes) {
    const fn = fById.get(tn.id);
    if (!fn) return "nodeEdit";
    if (Math.round(fn.position.x) !== Math.round(tn.position.x) || Math.round(fn.position.y) !== Math.round(tn.position.y)) return "nodeMove";
    if (JSON.stringify(fn.data?.params ?? {}) !== JSON.stringify(tn.data?.params ?? {})) return "nodeParam";
  }
  return "nodeEdit";
}

function announceNode(from: { nodes: Node[]; edges: Edge[] }, to: { nodes: Node[]; edges: Edge[] }, kind: "undo" | "redo") {
  const verb = i18n.t(kind === "undo" ? "history.undone" : "history.redone");
  useAppStore.getState().showBanner(`${verb} · ${i18n.t(`history.${describeNodeDelta(from, to)}`)}`, kind);
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
    data: { label: "Output", params: { laneLabel: DEFAULT_OUTPUT_GROUP } },
    deletable: true,
    zIndex: 1,
  },
];

export function WorkflowEditor({ segmentId, onClose, style }: Props) {
  const { t } = useTranslation();
  // Narrow selectors (NOT the whole store): otherwise WorkflowEditor re-renders on every project-store
  // change incl. the playhead during playback → the reconcile effect churns at frame rate.
  const tracks = useProjectStore((s) => s.tracks);
  const mergeProcessedOutputs = useProjectStore((s) => s.mergeProcessedOutputs);
  const updateTrackRef = useRef(useProjectStore.getState().updateTrack);
  updateTrackRef.current = useProjectStore.getState().updateTrack;
  const executionState = useWorkflowStore((s) => s.executions[segmentId]);
  // A pending split-mid-render LINK target mirrors its render SOURCE for display (the source's single render
  // feeds both halves via RenderLinkWatcher) and is LOCKED — it must not run its own copy of the nodes
  // (that would start a rejected/duplicate render). renderLocked drives the read-through + the run lock.
  const linkedSource = useWorkflowStore((s) => s.renderLinks[segmentId]);
  const linkedExec = useWorkflowStore((s) => (linkedSource ? s.executions[linkedSource] : undefined));
  const effExec = linkedExec ?? executionState;
  const renderLocked = linkedSource !== undefined;
  // The reconciler (below) re-runs when this segment's render cache changes (a node just finished).
  const nodeOutputsForSegment = useWorkflowStore((s) => s.nodeOutputs[segmentId]);
  // Focus-based Ctrl+Z / edit-key ownership (the panel is co-visible with the tracks now): clicking or
  // focusing anywhere in the editor claims the "workflow" pane; the track area reclaims "timeline".
  const setActivePane = useAppStore((s) => s.setActivePane);
  const activePane = useAppStore((s) => s.activePane);
  const focusEditor = useCallback(() => setActivePane("workflow"), [setActivePane]);

  useEffect(() => {
    useMsstModelStore.getState().fetchModelsDir();
    useMsstModelStore.getState().fetchInstalled();
  }, []);

  // On open, if this segment was rendered in a PRIOR session (persisted processedOutputs) but the runtime
  // render cache is cold (project just loaded/autoloaded), rebuild the cache + node badges from the saved
  // deposits — so the separation/render nodes show completed and deleting an Output edge then reconnecting
  // it re-deposits from cache instead of forcing a full re-separation of audio that already exists. Runs
  // once per open (the panel is keyed by segmentId → it remounts on every segment switch). A pending
  // split-mid-render LINK target is skipped (RenderLinkWatcher owns its lanes until the source settles).
  useEffect(() => {
    if (useWorkflowStore.getState().renderLinks[segmentId]) return;
    const seg = useProjectStore.getState().tracks.flatMap((t) => t.segments).find((s) => s.id === segmentId);
    if (seg) rehydrateRenderState(segmentId, seg);
  }, [segmentId]);

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
  const [edgeCtx, setEdgeCtx] = useState<{ x: number; y: number; edgeId: string } | null>(null);

  // Live refs (used by single-node run + the modal-local undo capture / drag coalescing).
  const nodesRef = useRef(nodes);
  nodesRef.current = nodes;
  const edgesRef = useRef(edges);
  edgesRef.current = edges;

  // --- Modal-local undo/redo for the node graph -----------------------------
  // The workflow editor OWNS Ctrl+Z while open (registered via setUndoScope below) — a self-contained
  // snapshot stack of {nodes, edges} that is independent of the timeline history. Selection/dimension
  // changes are excluded; a node DRAG is coalesced into one step (dragStart→dragStop); add/delete/
  // connect/param edits are one step each. A successful RUN is a COMMIT BARRIER that CLEARS the stack
  // — so pressing undo right after a render does nothing, and you can never undo across a render. This
  // is the agreed "a render is a commit" rule (audio-track node workflows only; vocal-track rendering
  // will get its own, more complex rule later). The persisted segment.workflow is deliberately kept
  // OUT of the timeline undo (excluded from its meaningful diff), so node editing never makes a
  // timeline step.
  const lpPast = useRef<{ nodes: Node[]; edges: Edge[] }[]>([]);
  const lpFuture = useRef<{ nodes: Node[]; edges: Edge[] }[]>([]);
  const commitRef = useRef<{ nodes: Node[]; edges: Edge[] }>({ nodes: initialData.nodes, edges: initialData.edges });
  const lastLocalSig = useRef(sigOfGraph(initialData.nodes, initialData.edges));
  const draggingNodeRef = useRef(false);
  const applyingLocalRef = useRef(false);

  const captureLocal = useCallback(() => {
    if (draggingNodeRef.current) return; // mid-drag frames coalesce; recorded on dragStop
    const sig = sigOfGraph(nodesRef.current, edgesRef.current);
    if (sig === lastLocalSig.current) {
      // No STRUCTURAL change (e.g. a pure node MOVE — position is out of the signature, so dragging a node
      // is not an undo step, or a selection-only change). Still refresh the baseline graph so a LATER real
      // edit's undo doesn't snap nodes back to a stale position.
      commitRef.current = { nodes: nodesRef.current, edges: edgesRef.current };
      return;
    }
    lpPast.current.push(commitRef.current);
    if (lpPast.current.length > 100) lpPast.current.shift();
    lpFuture.current = [];
    commitRef.current = { nodes: nodesRef.current, edges: edgesRef.current };
    lastLocalSig.current = sig;
  }, []);

  const applyLocal = useCallback((snap: { nodes: Node[]; edges: Edge[] }) => {
    applyingLocalRef.current = true; // consumed by the capture effect so undo/redo doesn't re-record
    // Node POSITION is not undoable: overlay each SURVIVING node's CURRENT position onto the restored
    // structure, so undoing/redoing a graph edit never yanks nodes back to where they sat at capture time
    // (the reported "Ctrl+Z reverts my node move together with the real op"). A node RE-ADDED by the
    // undo/redo has no current position, so it keeps the snapshot's.
    const curPos = new Map(nodesRef.current.map((n) => [n.id, n.position]));
    const nodes = snap.nodes.map((n) => {
      const pos = curPos.get(n.id);
      return pos ? { ...n, position: pos } : n;
    });
    setNodes(nodes);
    setEdges(snap.edges);
    commitRef.current = { nodes, edges: snap.edges };
    lastLocalSig.current = sigOfGraph(nodes, snap.edges);
  }, [setNodes, setEdges]);

  const localUndo = useCallback(() => {
    if (draggingNodeRef.current || applyingLocalRef.current) return; // not mid-drag / re-entrant
    if (lpPast.current.length === 0) return;
    const cur = commitRef.current;
    lpFuture.current.push(cur);
    const before = lpPast.current.pop()!;
    applyLocal(before);
    announceNode(before, cur, "undo"); // the undone node op transformed before→cur
  }, [applyLocal]);

  const localRedo = useCallback(() => {
    if (draggingNodeRef.current || applyingLocalRef.current) return;
    if (lpFuture.current.length === 0) return;
    const cur = commitRef.current;
    lpPast.current.push(cur);
    const after = lpFuture.current.pop()!;
    applyLocal(after);
    announceNode(cur, after, "redo");
  }, [applyLocal]);

  const onNodeDragStart = useCallback(() => { draggingNodeRef.current = true; }, []);
  const onNodeDragStop = useCallback(() => { draggingNodeRef.current = false; captureLocal(); }, [captureLocal]);

  // Auto-capture node/edge edits (add / delete / connect / param) as undo steps. Mid-drag frames and
  // undo/redo applies are skipped (drag is captured on dragStop instead).
  useEffect(() => {
    if (applyingLocalRef.current) { applyingLocalRef.current = false; return; }
    captureLocal();
  }, [nodes, edges, captureLocal]);

  // Claim Ctrl+Z / Ctrl+Y while the editor is open; release on close so the timeline regains it.
  // canUndo/canRedo read the local stacks so the Edit menu's enablement matches this scope.
  useEffect(() => {
    setUndoScope({
      undo: localUndo,
      redo: localRedo,
      canUndo: () => lpPast.current.length > 0,
      canRedo: () => lpFuture.current.length > 0,
    });
    return () => setUndoScope(null);
  }, [localUndo, localRedo]);

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
      if (type === "audioOutput") defaultParams.laneLabel = DEFAULT_OUTPUT_GROUP;
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
      if (type === "audioOutput") defaultParams.laneLabel = DEFAULT_OUTPUT_GROUP;
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


  // A node is "busy" while it is running or queued in the active run. Deleting it mid-render would
  // only drop the UI while the backend job keeps going — so we block deletion of busy nodes.
  const isNodeBusy = useCallback((nodeId: string) => {
    // Only guard during a LIVE run. Once it settles (completed / error / cancelled) the node frees
    // up — otherwise a stale "waiting" left behind by a cancel or partial failure would make the
    // downstream nodes permanently un-deletable until a full re-run.
    if (useWorkflowStore.getState().executions[segmentId]?.status !== "running") return false;
    // Output nodes are NEVER busy: deleting one mid-run only drops its deposit intent (the backend job
    // runs on the upstream nodes + the dispatch-time graph snapshot), so they stay freely deletable —
    // even while showing the blue "depositing" pulse. Only the upstream RENDER path locks.
    if (nodesRef.current.find((n) => n.id === nodeId)?.type === "audioOutput") return false;
    const st = useWorkflowStore.getState().nodeStatuses[segmentId]?.[nodeId];
    return st === "running" || st === "waiting";
  }, [segmentId]);

  const closeMenus = useCallback(() => { setNodeCtx(null); setEdgeCtx(null); }, []);

  // An edge whose TARGET is an Output node is never a render-path edge — deleting it only changes what
  // gets deposited, never the in-flight backend job — so it stays freely deletable even mid-run.
  const isOutputEdge = useCallback((edge: { target: string }) =>
    nodesRef.current.find((n) => n.id === edge.target)?.type === "audioOutput", []);

  const onNodeContextMenu: NodeMouseHandler = useCallback((event, node) => {
    event.preventDefault();
    if (node.deletable === false) return;
    if (isNodeBusy(node.id)) return; // 正在执行/排队的节点不弹删除菜单
    setEdgeCtx(null);
    setNodeCtx({ x: event.clientX, y: event.clientY, nodeId: node.id });
  }, [isNodeBusy]);

  const onEdgeContextMenu: EdgeMouseHandler = useCallback((event, edge) => {
    event.preventDefault();
    if (!isOutputEdge(edge) && (isNodeBusy(edge.source) || isNodeBusy(edge.target))) return; // lock render-path edges only; edges INTO an Output stay deletable
    setNodeCtx(null);
    setEdgeCtx({ x: event.clientX, y: event.clientY, edgeId: edge.id });
  }, [isNodeBusy, isOutputEdge]);

  const handleDeleteNode = useCallback((nodeId: string) => {
    setNodes((nds) => nds.filter((n) => n.id !== nodeId));
    setEdges((eds) => eds.filter((e) => e.source !== nodeId && e.target !== nodeId));
    setNodeCtx(null);
  }, [setNodes, setEdges]);

  const handleDeleteEdge = useCallback((edgeId: string) => {
    setEdges((eds) => eds.filter((e) => e.id !== edgeId));
    setEdgeCtx(null);
  }, [setEdges]);

  // Veto deletion (Delete key OR programmatic) of busy nodes. IO nodes are already non-deletable
  // (deletable:false) so ReactFlow filters them out before this runs; we only guard busy ones.
  const onBeforeDelete = useCallback(
    async ({ nodes: delNodes, edges: delEdges }: { nodes: Node[]; edges: Edge[] }) => {
      // A node is locked while running/queued. An EDGE is locked if EITHER endpoint is — this covers
      // both "delete the busy node (xyflow auto-adds its edges to delEdges)" AND "user right-clicks /
      // selects just that edge". Locking the wires of the about-to-render path stops the graph being
      // changed out from under the in-flight backend job (which runs on the dispatch-time snapshot).
      const allowedNodes = delNodes.filter((n) => !isNodeBusy(n.id));
      const deletedIds = new Set(allowedNodes.map((n) => n.id));
      // An edge MUST be removed if EITHER endpoint is being removed — never leave a dangling edge
      // (a deleted-source edge poisons parseWorkflowGraph → "contains a cycle" and bricks the graph).
      // Otherwise it's the user deleting just the edge: allowed only if neither endpoint is busy.
      const allowedEdges = delEdges.filter(
        (e) => deletedIds.has(e.source) || deletedIds.has(e.target) || isOutputEdge(e) || (!isNodeBusy(e.source) && !isNodeBusy(e.target)),
      );
      if (allowedNodes.length === delNodes.length && allowedEdges.length === delEdges.length) {
        return { nodes: delNodes, edges: delEdges };
      }
      if (allowedNodes.length === 0 && allowedEdges.length === 0) return false;
      return { nodes: allowedNodes, edges: allowedEdges };
    },
    [isNodeBusy, isOutputEdge],
  );

  const nodeCtxItems: MenuItem[] = nodeCtx ? [
    // "Detach": only for a MULTI-input Output node — splits it into one Output (group) per inbound edge.
    ...(nodesRef.current.find((n) => n.id === nodeCtx.nodeId)?.type === "audioOutput" &&
    edgesRef.current.filter((e) => e.target === nodeCtx.nodeId).length >= 2
      ? [{
          label: t("workflow.detachGroup"),
          onClick: () => {
            useAppStore.getState().requestLaneDetach(segmentId, nodeCtx.nodeId);
            setNodeCtx(null);
          },
        }]
      : []),
    { label: t("toolbar.delete"), shortcut: "Del", danger: true, onClick: () => handleDeleteNode(nodeCtx.nodeId) },
  ] : [];

  const edgeCtxItems: MenuItem[] = edgeCtx ? [
    { label: t("workflow.deleteConnection"), shortcut: "Del", danger: true, onClick: () => handleDeleteEdge(edgeCtx.edgeId) },
  ] : [];

  const handleExecute = useCallback(async () => {
    const trackId = segTrackIdRef.current;
    if (!segment || !trackId) return;
    if (useWorkflowStore.getState().renderLinks[segmentId]) return; // linked half: its render is the source's
    const wf = reactFlowToWorkflow(nodes, edges);
    // Barrier floor captured at DISPATCH so edits made DURING the async render stay undoable down to —
    // but not past — this point.
    const dispatchPastLen = lpPast.current.length;
    try {
      // The RECONCILER (running live during + after the run) owns the track lanes: at run start it places
      // loading placeholders for the connected Output lanes, then deposits each lane the moment its branch
      // finishes (executeWorkflow caches per node → the reconcile effect fires), and cleans the
      // placeholders of branches that never ran. So handleExecute no longer touches processedOutputs — it
      // only drives the render + the node-graph undo barrier.
      // Re-run with edited params overwrites node outputs at the SAME deterministic path, so the live
      // reconciler's path-equality check would KEEP the stale deposited lane (stale waveform + playback
      // buffer). Invalidate this segment's prior deposit + buffers UP FRONT, so each branch re-decodes
      // fresh as it finishes during the run (no stale lane, and no post-run flash-out-and-back).
      const segBefore = useProjectStore.getState().tracks.find((t) => t.id === trackId)?.segments.find((s) => s.id === segmentId);
      const stale = new Set<string>();
      for (const o of segBefore?.processedOutputs ?? []) {
        if (o.loading) continue;
        clearBufferCache(o.audioPath);
        if (o.outputNodeId) stale.add(o.outputNodeId);
      }
      for (const outId of stale) useProjectStore.getState().removeProcessedOutputsForNode(trackId, segmentId, outId);
      const outputs = await executeWorkflow(segmentId, segment.seg, wf);
      if (outputs.length === 0) {
        // Nothing reached an Output node (none connected, or no stems produced) — say so, don't sit silent.
        useAppStore.getState().showToast(i18n.t("workflow.noOutputs"), "error");
        return;
      }
      flushAutosaveNow(); // a render is a commit — snapshot to disk NOW (don't wait for the 1.5s debounce)
      // A render is a COMMIT BARRIER for the node-graph undo: drop the pre-render history (and any redo
      // branch) but KEEP edits made while the render was in flight undoable.
      lpPast.current = lpPast.current.slice(Math.min(dispatchPastLen, lpPast.current.length));
      lpFuture.current = [];
    } catch (err) {
      // Live-deposit model: branches that FINISHED stay on the track (what rendered, rendered); the
      // reconciler removes the loading placeholders of branches that didn't. Just drop stuck badges here.
      useWorkflowStore.getState().clearPendingStatuses(segmentId);
      if (err instanceof Error && err.message === "Cancelled") return;
      console.error("Workflow execution failed:", err);
    }
  }, [segmentId, segment, nodes, edges]);

  const handleCancel = useCallback(() => {
    useWorkflowStore.getState().cancelExecution(segmentId);
    useWorkflowStore.getState().clearPendingStatuses(segmentId); // 立即清掉蓝/黄框（后端随后停止）
    invoke("cancel_separation").catch(() => {});
  }, [segmentId]);

  const segmentRef = useRef(segment);
  segmentRef.current = segment;

  // Flush a pending debounced save on unmount so closing fast doesn't drop the last <300ms of edits
  // (the save effect's cleanup only clears the timer — this writes the final graph through).
  useEffect(() => {
    return () => {
      clearTimeout(saveTimer.current);
      const trackId = segTrackIdRef.current;
      if (!trackId) return;
      const wf = reactFlowToWorkflow(nodesRef.current, edgesRef.current);
      const track = useProjectStore.getState().tracks.find((tr) => tr.id === trackId);
      if (!track) return;
      updateTrackRef.current(trackId, {
        segments: track.segments.map((s) => (s.id === segmentId ? { ...s, workflow: wf } : s)),
      });
    };
  }, [segmentId]);

  const handleRunSingleNode = useCallback((nodeId: string) => {
    const seg = segmentRef.current;
    if (!seg) return;
    if (useWorkflowStore.getState().renderLinks[segmentId]) return; // linked half: its render is the source's
    const wf = reactFlowToWorkflow(nodesRef.current, edgesRef.current);
    // Invalidation is PATH-driven: every run writes into a fresh run dir (engine ensureRunDir), so each
    // node this run actually executes — the clicked node AND any upstream re-run as an uncached/sparse
    // dependency — emits NEW output paths, and the reconciler's path-inequality check re-decodes every
    // deposited lane they feed (including lanes of OTHER Output nodes fed by a re-run upstream, which a
    // clicked-node-only invalidation used to leave stale). Nodes reused from a dense cache keep their
    // old paths → their lanes are untouched.
    executeSingleNode(segmentId, seg.seg, wf, nodeId).catch((err) => {
      console.error("Single node execution failed:", err);
    });
  }, [segmentId]);

  // --- Output AUTO-DEPOSIT reconciler ----------------------------------------
  // Output nodes are LIVE: connecting a rendered edge into one auto-deposits its audio onto the track,
  // disconnecting the edge or deleting the node removes it — there is no manual button. ONE reconciler
  // makes each Output node's track lanes MATCH its current inbound edges, so it stays correct under
  // connect / disconnect / node-delete / a node finishing a render / even Ctrl+Z on an edge (all change
  // the inputs the effect below watches). It only writes processedOutputs (a non-undoable overlay) so it
  // NEVER clears the redo stack. Removal is driven by edge STRUCTURE, not render-cache freshness, so
  // reopening a saved segment (cold cache) never wipes its persisted lanes.
  // DEBOUNCED playback reschedule: rapid lane changes (a multi-branch run depositing one-by-one, or
  // several removals) coalesce into ONE stop+replay instead of restarting playback per change —
  // restarting repeatedly mid-playback is what caused the audible "ghosting"/stutter.
  const rescheduleTimerRef = useRef<ReturnType<typeof setTimeout>>(undefined);
  const scheduleReschedule = useCallback(() => {
    clearTimeout(rescheduleTimerRef.current);
    rescheduleTimerRef.current = setTimeout(() => {
      if (useAudioStore.getState().isPlaying) useAudioStore.getState().bumpSchedule();
    }, 140);
  }, []);

  const reconcilingRef = useRef<Set<string>>(new Set());
  const pendingReconcileRef = useRef<Set<string>>(new Set());
  const reconcileSigRef = useRef(""); // last (structure+cache+status) sig — skip no-op effect re-fires
  const reconcileOutputNodeRef = useRef<(id: string) => Promise<void>>(async () => {});

  const reconcileOutputNode = useCallback(async (outputNodeId: string) => {
    const trackId = segTrackIdRef.current;
    if (!trackId || !segmentRef.current) return;
    // One reconcile per node at a time; coalesce a change that lands mid-flight into one re-run after.
    if (reconcilingRef.current.has(outputNodeId)) { pendingReconcileRef.current.add(outputNodeId); return; }
    reconcilingRef.current.add(outputNodeId);
    const setStatus = (s: NodeStatus) => useWorkflowStore.getState().setNodeStatus(segmentId, outputNodeId, s);
    const segOutputs = () =>
      useProjectStore.getState().tracks.find((t) => t.id === trackId)
        ?.segments.find((s) => s.id === segmentId)?.processedOutputs ?? [];
    const reschedule = scheduleReschedule; // debounced — coalesces rapid lane changes into one replay
    try {
      const wf = reactFlowToWorkflow(nodesRef.current, edgesRef.current);
      const lanes = outputLanes(wf, outputNodeId);                            // structural lanes [{laneId, laneLabel, group}]
      const cached = new Map(collectCachedPaths(segmentId, outputNodeId, wf).paths.map((p) => [p.laneId, p] as const));
      const mine = segOutputs().filter((o) => o.outputNodeId === outputNodeId);
      const deposited = new Map(mine.filter((o) => !o.loading).map((o) => [o.laneId, o] as const));
      // "Is a render in flight for THIS segment's lanes?" — for a split-mid-render LINKED half the render
      // runs on its SOURCE (it has no execution of its own), so read the source's status via renderLinks.
      // Without this a linked half was always runActive=false → connected/new lanes hit "idle → no lane"
      // (loading never showed / carried placeholders got wiped), which is why the reconciler used to skip
      // linked halves entirely. Reading the source's status lets the reconciler run NORMALLY on a linked
      // half: add placeholders for newly-connected lanes, keep the connected ones, prune the deleted ones.
      const wfNow = useWorkflowStore.getState();
      const runActive = wfNow.executions[wfNow.renderLinks[segmentId] ?? segmentId]?.status === "running";

      // Build the DESIRED on-track set for this node + the list of lanes needing a (re)decode.
      const target: ProcessedOutput[] = [];
      const toDecode: CachedPath[] = [];
      for (const { laneId, laneLabel, group } of lanes) {
        const c = cached.get(laneId);
        const dep = deposited.get(laneId);
        if (c && (!dep || dep.audioPath !== c.audioPath)) {
          toDecode.push(c);
          target.push({ laneId, laneLabel: c.laneLabel, group: c.group, audioPath: c.audioPath, totalDurationMs: 0, loading: true, outputNodeId });
        } else if (dep) {
          // unchanged, or cold-cache persisted → KEEP (refresh the display label/group if changed — use
          // the STRUCTURAL label so a rename propagates even with a cold cache where c is undefined)
          target.push(laneLabel !== dep.laneLabel || group !== dep.group ? { ...dep, laneLabel, group } : dep);
        } else if (runActive) {
          // connected but its source hasn't rendered YET during a live run → loading placeholder so the
          // user sees the lane "coming" (covers connecting a stem mid-run).
          target.push({ laneId, laneLabel, group, audioPath: `__pending_${laneId}`, totalDurationMs: 0, loading: true, outputNodeId });
        }
        // else: uncached + idle → no lane (deposits once its source renders)
      }
      const anyLoading = target.some((o) => o.loading);
      const sig = (arr: ProcessedOutput[]) =>
        arr.map((o) => `${o.laneId}|${o.audioPath}|${o.laneLabel}|${o.group ?? ""}|${o.loading ? 1 : 0}`).sort().join(",");
      const changed = sig(target) !== sig(mine);

      if (!changed) {
        const want: NodeStatus = anyLoading ? "running" : target.length > 0 ? "completed" : "idle";
        if (useWorkflowStore.getState().nodeStatuses[segmentId]?.[outputNodeId] !== want) setStatus(want);
        return;
      }
      if (target.length === 0) {
        useProjectStore.getState().removeProcessedOutputsForNode(trackId, segmentId, outputNodeId);
        setStatus("idle");
        reschedule();
        return;
      }
      // Show the placeholders/keeps IMMEDIATELY (instant feedback), then decode + replace.
      mergeProcessedOutputs(trackId, segmentId, target);
      if (!useProjectStore.getState().tracks.find((t) => t.id === trackId)?.expanded) {
        useProjectStore.getState().toggleTrackExpanded(trackId);
      }
      setStatus(anyLoading ? "running" : "completed");
      reschedule();
      if (toDecode.length > 0) {
        const decoded = new Map<string, ProcessedOutput>();
        for (const p of toDecode) {
          clearBufferCache(p.audioPath);
          loadAudioBuffer(p.audioPath);
          decoded.set(p.laneId, await loadCachedOutput(p));
        }
        // The node may have been deleted while we were decoding — drop its lanes, don't deposit a phantom.
        if (!nodesRef.current.some((n) => n.id === outputNodeId)) {
          useProjectStore.getState().removeProcessedOutputsForNode(trackId, segmentId, outputNodeId);
          reschedule();
          return;
        }
        const finalTarget = target.map((o) => decoded.get(o.laneId) ?? o); // decoded replace placeholders; pending stay
        mergeProcessedOutputs(trackId, segmentId, finalTarget);
        setStatus(finalTarget.some((o) => o.loading) ? "running" : "completed");
        reschedule();
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      logToBackend("error", `Output auto-deposit failed: ${msg}`);
      setStatus("error");
      useAppStore.getState().showToast(i18n.t("workflow.depositFailed"), "error");
      // Drop this node's still-loading placeholder(s): the change-sig would otherwise rebuild an
      // IDENTICAL placeholder next reconcile and early-return → a lane stuck "loading" forever. Keep its
      // finished (non-loading) lanes; a later trigger can retry the decode.
      const keep = segOutputs().filter((o) => o.outputNodeId === outputNodeId && !o.loading);
      useProjectStore.getState().removeProcessedOutputsForNode(trackId, segmentId, outputNodeId);
      if (keep.length > 0) mergeProcessedOutputs(trackId, segmentId, keep);
      reschedule();
    } finally {
      reconcilingRef.current.delete(outputNodeId);
      if (pendingReconcileRef.current.has(outputNodeId)) {
        pendingReconcileRef.current.delete(outputNodeId);
        void reconcileOutputNodeRef.current(outputNodeId);
      }
    }
  }, [segmentId, mergeProcessedOutputs, scheduleReschedule]);
  reconcileOutputNodeRef.current = reconcileOutputNode;

  // Reconcile every Output node whenever the graph (nodes/edges), this segment's render cache, or the run
  // state changes — INCLUDING DURING a live run, so each Output lane deposits the moment its branch
  // finishes (and shows a loading placeholder before then). Mutating the project store inside does NOT
  // re-trigger this effect (its deps are graph + workflow-store state), so there is no feedback loop.
  useEffect(() => {
    const trackId = segTrackIdRef.current;
    if (!trackId || !segmentRef.current) return;
    const outputIds = nodesRef.current.filter((n) => n.type === "audioOutput").map((n) => n.id);
    const outSet = new Set(outputIds);
    // Idempotency guard: only act when something that affects deposits changed — the Output-node set, the
    // edges INTO them, this segment's render cache, and the run status. Without it the effect re-runs on
    // unrelated re-renders (progress ticks, playback) and the per-frame reconcile floods the log + CPU.
    const wf = useWorkflowStore.getState();
    // A split-mid-render LINKED half reconciles NORMALLY (no early-return): its per-node reconcile reads
    // `runActive` from the SOURCE's execution (see reconcileOutputNode), so during the render it correctly
    // shows loading placeholders for connected/newly-added lanes, keeps them, and prunes deleted ones — all
    // LIVE. It won't wrongly deposit from a cold cache (that path needs cached paths this segment lacks until
    // RenderLinkWatcher clones + headless-deposits on settle). This replaced the old "skip linked entirely"
    // early-return, which made loading never update on a linked half — the regression the user hit.
    const cache = wf.nodeOutputs[segmentId] ?? {};
    const sig = JSON.stringify({
      o: [...outputIds].sort(),
      e: edgesRef.current.filter((e) => outSet.has(e.target)).map((e) => `${e.source}:${e.sourceHandle}>${e.target}:${e.targetHandle}`).sort(),
      c: Object.keys(cache).sort().map((k) => `${k}=${(cache[k] ?? []).join(",")}`),
      s: wf.executions[segmentId]?.status ?? "",
      // The Output nodes' GROUP param: a rename must re-reconcile (the KEEP branch relabels the deposited
      // lanes in place) — without this term the rename only landed on the NEXT unrelated sig change,
      // looking broken then applying as a surprise later. Output nodes only — other nodes' param edits
      // (RVC sliders etc.) must NOT churn the reconciler.
      l: nodesRef.current
        .filter((n) => n.type === "audioOutput")
        .map((n) => `${n.id}=${((n.data as { params?: Record<string, unknown> }).params?.laneLabel as string) ?? ""}`)
        .sort(),
      // Upstream stem names feeding the outputs: the structural laneLabel is `group · stem`, and the
      // stem comes from the FEEDER's stemLabels param (changes with a model/preset switch, no run) —
      // without this the relabel is just as late as a group rename used to be.
      t: edgesRef.current
        .filter((e) => outSet.has(e.target))
        .map((e) => {
          const port = parseInt(e.sourceHandle?.replace("out-", "") ?? "0", 10);
          const stems = (nodesRef.current.find((n) => n.id === e.source)?.data as { params?: Record<string, unknown> } | undefined)
            ?.params?.stemLabels as string[] | undefined;
          return `${e.source}:${port}=${stems?.[port] ?? ""}`;
        })
        .sort(),
    });
    if (sig === reconcileSigRef.current) return;
    reconcileSigRef.current = sig;
    // Orphan cleanup: a deleted Output node is gone from the graph, so the per-node loop won't see it —
    // drop any track lanes whose producing node no longer exists (+ refresh live playback).
    const seg = useProjectStore.getState().tracks.find((t) => t.id === trackId)?.segments.find((s) => s.id === segmentId);
    for (const o of seg?.processedOutputs ?? []) {
      if (o.outputNodeId && !outSet.has(o.outputNodeId)) {
        useProjectStore.getState().removeProcessedOutputsForNode(trackId, segmentId, o.outputNodeId);
        scheduleReschedule();
      }
    }
    for (const id of outputIds) void reconcileOutputNodeRef.current(id);
    // `linkedSource` (renderLinks[segmentId]) IS a dependency: the per-node reconcile derives runActive
    // from the SOURCE while linked, so the effect must re-fire when the link RESOLVES (unlinkRender →
    // linkedSource becomes undefined) — the sig gate alone wouldn't re-run it, and the orphan cleanup /
    // final reconcile for edits made while linked would only land on the next unrelated change.
  }, [nodes, edges, nodeOutputsForSegment, executionState?.status, linkedSource, segmentId, reconcileOutputNode, scheduleReschedule]);

  // --- Output-group DETACH ("ungroup") -----------------------------------------
  // Requested via app-store `pendingLaneDetach` (from THIS editor's node menu, or from a timeline lane
  // right-click that opened this editor first) — ONE code path: split the multi-input Output node into
  // one single-edge Output per inbound edge (stem-named groups), rewriting the deposited lanes IN PLACE
  // (same audio, no re-decode). The graph edit lands in the NODE-GRAPH undo stack (auto-capture); undoing
  // it restores the old node and the reconciler converges the lanes back (laneOps' old key is kept).
  // The store half runs under history.runSilent — laneOps/laneControls are in the timeline meaningfulSig,
  // and a machine bookkeeping write must not push a phantom timeline step / wash the redo stack.
  const pendingDetach = useAppStore((s) => s.pendingLaneDetach);
  useEffect(() => {
    // LIVE read (not the subscribed closure value): under React.StrictMode the mount effect runs twice
    // with the SAME closure — the synchronous clear below makes the second invocation read null and
    // no-op, where a closure check would double-apply the detach (duplicate nodes/edges/lanes,
    // review-caught HIGH). The subscription (`pendingDetach`) exists only to re-fire this effect.
    const pending = useAppStore.getState().pendingLaneDetach;
    if (!pending || pending.segmentId !== segmentId) return;
    useAppStore.getState().clearLaneDetach();
    const trackId = segTrackIdRef.current;
    if (!trackId || !segmentRef.current) return;
    const wf = reactFlowToWorkflow(nodesRef.current, edgesRef.current);
    const plan = planDetachGroup(wf, pending.outputNodeId);
    if (!plan) return;
    nodeCounter++;
    const zBase = nodeCounter + 100;
    const newRfNodes: Node[] = [
      ...nodesRef.current.filter((n) => n.id !== plan.oldNodeId),
      ...plan.newNodes.map((nn, i) => ({
        id: nn.id,
        type: "audioOutput",
        position: nn.position,
        data: { label: "output", params: { laneLabel: nn.group } },
        deletable: true,
        zIndex: zBase + i,
      })),
    ];
    const newRfEdges: Edge[] = [
      ...edgesRef.current.filter((e) => e.target !== plan.oldNodeId),
      ...plan.newNodes.map((nn) => ({
        id: `e-detach-${nn.id}`,
        source: nn.edge.fromNode,
        sourceHandle: `out-${nn.edge.fromPort}`,
        target: nn.id,
        targetHandle: "in-0",
        animated: true,
      })),
    ];
    setNodes(newRfNodes);
    setEdges(newRfEdges);
    // Persist the post-detach graph to segment.workflow SYNCHRONOUSLY (same write the debounced save
    // does, which alone would lag ~300ms): applyLaneDetach rewrites the deposited lanes to the new node
    // ids IMMEDIATELY, and a save/autosave/split landing in the debounce window would otherwise capture
    // lanes referencing nodes that exist in no persisted graph — on reload the orphan cleanup would
    // silently delete every detached (already-rendered) lane. workflow is history-excluded → no undo step.
    {
      const wfAfter = reactFlowToWorkflow(newRfNodes, newRfEdges);
      const track = useProjectStore.getState().tracks.find((tr) => tr.id === trackId);
      if (track) {
        updateTrackRef.current(trackId, {
          segments: track.segments.map((s) => (s.id === segmentId ? { ...s, workflow: wfAfter } : s)),
        });
      }
    }
    useHistoryStore.getState().runSilent(() =>
      useProjectStore.getState().applyLaneDetach(trackId, segmentId, plan.oldNodeId, plan.mapping),
    );
    // The lane selection (set by the lane right-click that requested this) points at the REMOVED node —
    // remap it onto the first detached group so the gold cue + Ctrl+K/Delete keep targeting a real lane
    // (a stale selection would lazily clear and mis-route Ctrl+K to a whole-segment split).
    const sel = useAppStore.getState().selectedLane;
    if (sel && sel.segmentId === segmentId && sel.outputNodeId === plan.oldNodeId && plan.mapping.length > 0) {
      useAppStore.getState().selectLane(trackId, segmentId, plan.mapping[0]!.newNodeId, sel.clipIndex);
    }
    scheduleReschedule();
  }, [pendingDetach, segmentId, setNodes, setEdges, scheduleReschedule]);

  // Clicking an Output node selects its GROUP's sub-lanes on the track (the many-to-one bridge): all
  // rows of that group in the open segment light up gold — same selection the lane click sets. Only
  // when the group actually has deposited lanes; expands the track so the cue is visible. activePane
  // stays "workflow" (selectLane doesn't touch it), so Delete keeps deleting NODES, not lane pieces.
  const onNodeClick: NodeMouseHandler = useCallback(
    (_e, node) => {
      closeMenus();
      if (node.type !== "audioOutput") return;
      const trackId = segTrackIdRef.current;
      if (!trackId) return;
      const ps = useProjectStore.getState();
      const track = ps.tracks.find((tr) => tr.id === trackId);
      const seg = track?.segments.find((sg) => sg.id === segmentId);
      if (!track || !seg?.processedOutputs?.some((o) => o.outputNodeId === node.id && !o.loading)) return;
      if (!track.expanded) ps.toggleTrackExpanded(trackId);
      useAppStore.getState().selectLane(trackId, segmentId, node.id, 0);
    },
    [closeMenus, segmentId],
  );

  useEffect(() => {
    useWorkflowStore.getState().registerSingleNodeRunner(handleRunSingleNode);
    return () => useWorkflowStore.getState().registerSingleNodeRunner(null);
  }, [handleRunSingleNode]);

  const isRunning = effExec?.status === "running";

  return (
    <div className="workflow-editor" style={style} onPointerDownCapture={focusEditor} onFocusCapture={focusEditor}>
      <div className="workflow-header">
        <div className="workflow-header-left">
          <button className="workflow-close" onClick={onClose}>
            x
          </button>
          <span className="workflow-title">
            {t("workflow.title")} — {segmentId.slice(0, 8)}
          </span>
        </div>
        <div className="workflow-header-actions">
          {effExec?.status === "error" && (
            <span className="wf-error">{effExec.error}</span>
          )}
          {renderLocked ? (
            // Linked half: the source half owns the single shared render — show its state, don't offer a
            // Run/Stop here (stop it from the source half). Prevents a rejected/duplicate render. Literal
            // text to match the adjacent Run/Stop buttons (also literal).
            <span className="wf-run-btn running" title="Rendering — the source half owns this render; stop it there">
              Rendering…
            </span>
          ) : isRunning ? (
            <button className="wf-run-btn running" onClick={handleCancel}>
              Stop
            </button>
          ) : (
            <button className="wf-run-btn" onClick={handleExecute}>
              Run
            </button>
          )}
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
            onNodeDragStart={onNodeDragStart}
            onNodeDragStop={onNodeDragStop}
            onNodeContextMenu={onNodeContextMenu}
            onEdgeContextMenu={onEdgeContextMenu}
            onBeforeDelete={onBeforeDelete}
            onPaneClick={closeMenus}
            onNodeClick={onNodeClick}
            onMoveStart={closeMenus}
            onInit={setRfInstance}
            nodeTypes={nodeTypes}
            fitView
            deleteKeyCode={activePane === "workflow" ? "Delete" : null}
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
          {edgeCtx && <ContextMenu x={edgeCtx.x} y={edgeCtx.y} items={edgeCtxItems} onClose={() => setEdgeCtx(null)} />}
        </div>
      </div>
    </div>
  );
}
