import { create } from "zustand";

export type NodeStatus = "idle" | "waiting" | "running" | "completed" | "error";

export interface ExecutionState {
  status: "idle" | "running" | "completed" | "error";
  currentNodeId?: string;
  progress?: number;
  error?: string;
  cancelled?: boolean;
  /** Dispatch-time snapshot of the node ids this run will EXECUTE (non-IO; for a single-node run, the
   *  chain up to the target). The reconciler's pending placeholders key on membership — a lane whose
   *  feeder is not in the running snapshot will never be fed by it, so it must not show "loading". */
  participants?: string[];
}

/** A segment's portable render state (S61 copy/paste): the output CACHE always, node badges +
 *  execution only when the run was SETTLED (the cloneSegmentState gate — a running execution must
 *  never be duplicated onto another id, it would be a phantom nothing drives). Self-contained, so a
 *  clipboard can hold it long after the source segment is deleted. */
export interface SegmentRenderSnapshot {
  nodeOutputs?: Record<string, string[]>;
  nodeStatuses?: Record<string, NodeStatus>;
  nodeProgress?: Record<string, number>;
  nodeErrors?: Record<string, string>;
  execution?: ExecutionState;
}

interface WorkflowStore {
  executions: Record<string, ExecutionState>;
  nodeStatuses: Record<string, Record<string, NodeStatus>>;
  nodeOutputs: Record<string, Record<string, string[]>>;
  nodeProgress: Record<string, Record<string, number>>;
  nodeErrors: Record<string, Record<string, string>>;
  /** Split-mid-render inheritance: newSegmentId -> sourceSegmentId. When a segment is split WHILE its
   *  render is in flight, the new half is linked here; RenderLinkWatcher mirrors the single ongoing
   *  render's final lanes onto it once the source settles (the render itself is one global job — see the
   *  S25 render-lifecycle map). */
  renderLinks: Record<string, string>;
  singleNodeRunner: ((nodeId: string) => void) | null;

  registerSingleNodeRunner: (fn: ((nodeId: string) => void) | null) => void;
  startExecution: (segmentId: string, participants: string[]) => void;
  updateProgress: (segmentId: string, nodeId: string, progress: number) => void;
  completeExecution: (segmentId: string) => void;
  failExecution: (segmentId: string, error: string) => void;
  cancelExecution: (segmentId: string) => void;
  isCancelled: (segmentId: string) => boolean;
  clearExecution: (segmentId: string) => void;
  /** Copy a segment's render state (output CACHE always; node badges + execution only for a SETTLED run)
   *  onto a NEW segment id — used when a segment is SPLIT so the new half "remembers" it was rendered
   *  (reconnecting an Output re-deposits from the cache instead of forcing a full re-run). A RUNNING
   *  source is NOT cloned wholesale: cloning a 'running' execution would create a phantom that nothing
   *  drives (stuck spinner + blocks the quit/busy warning), so only its partial cache rides along. */
  cloneSegmentState: (fromId: string, toId: string) => void;
  /** Capture a segment's render state as a portable snapshot (see SegmentRenderSnapshot). null when
   *  there is nothing to carry. Same settled gate as cloneSegmentState (single source: clone = snapshot
   *  + install). */
  snapshotSegmentState: (segmentId: string) => SegmentRenderSnapshot | null;
  /** Install a previously captured snapshot under a (new) segment id — the paste half of copy/paste.
   *  Overwrites nothing when the snapshot is empty. */
  installSegmentState: (segmentId: string, snap: SegmentRenderSnapshot) => void;
  /** Warm the runtime render CACHE + mark nodes "completed" for a segment IN ONE update, reconstructed
   *  from its PERSISTED processedOutputs after a project load/autoload (see engine.rehydrateRenderState).
   *  No-op if the cache is already warm (a live / just-run session), so it never clobbers real run state. */
  hydrateRenderState: (segmentId: string, nodeOutputs: Record<string, string[]>, completedNodeIds: string[]) => void;
  setNodeOutputs: (segmentId: string, nodeId: string, paths: string[]) => void;
  /** Drop the runtime output cache for a whole segment (nodeId omitted) or ONE node. Called at the start of
   *  a (re-)run so the live reconciler can't re-deposit a stale path a run is about to overwrite in place. */
  clearNodeOutputs: (segmentId: string, nodeId?: string) => void;

  setNodeStatus: (segmentId: string, nodeId: string, status: NodeStatus) => void;
  setNodeProgress: (segmentId: string, nodeId: string, progress: number) => void;
  setNodeError: (segmentId: string, nodeId: string, error: string) => void;
  clearNodeStatuses: (segmentId: string) => void;
  clearPendingStatuses: (segmentId: string) => void;
  linkRender: (toId: string, fromId: string) => void;
  unlinkRender: (toId: string) => void;
}

export const useWorkflowStore = create<WorkflowStore>((set, get) => ({
  executions: {},
  nodeStatuses: {},
  nodeOutputs: {},
  nodeProgress: {},
  nodeErrors: {},
  renderLinks: {},
  singleNodeRunner: null,

  registerSingleNodeRunner: (fn) => set({ singleNodeRunner: fn }),
  startExecution: (segmentId, participants) =>
    set((s) => ({
      executions: {
        ...s.executions,
        [segmentId]: { status: "running", progress: 0, participants },
      },
    })),

  updateProgress: (segmentId, nodeId, progress) =>
    set((s) => ({
      executions: {
        ...s.executions,
        // Preserve the dispatch snapshot — progress ticks rebuild the entry wholesale.
        [segmentId]: { status: "running", currentNodeId: nodeId, progress, participants: s.executions[segmentId]?.participants },
      },
    })),

  completeExecution: (segmentId) =>
    set((s) => ({
      executions: {
        ...s.executions,
        [segmentId]: { status: "completed" },
      },
    })),

  failExecution: (segmentId, error) =>
    set((s) => ({
      executions: {
        ...s.executions,
        [segmentId]: { status: "error", error },
      },
    })),

  /** Flag the run cancelled but KEEP status "running" — the backend hasn't actually stopped yet. The
   *  engine loop polls isCancelled(), fires the Rust cancel flags, and SETTLES the execution
   *  (failExecution) only when the in-flight invoke really returns. The old behavior flipped to a
   *  settled status here, so the UI read "stopped" seconds before the backend obeyed: the Run button
   *  reverted, badges vanished, and the still-working node looked frozen (S62b user report). UIs show
   *  the interim as "cancelling" via status==="running" && cancelled. */
  cancelExecution: (segmentId) =>
    set((s) => {
      const cur = s.executions[segmentId];
      if (!cur) return {};
      return { executions: { ...s.executions, [segmentId]: { ...cur, cancelled: true } } };
    }),

  isCancelled: (segmentId) => {
    const exec = get().executions[segmentId];
    return exec?.cancelled === true;
  },

  clearExecution: (segmentId) =>
    set((s) => {
      const { [segmentId]: _, ...rest } = s.executions;
      return { executions: rest };
    }),

  cloneSegmentState: (fromId, toId) => {
    const snap = get().snapshotSegmentState(fromId);
    if (snap) get().installSegmentState(toId, snap);
  },

  snapshotSegmentState: (segmentId) => {
    const s = get();
    const outs = s.nodeOutputs[segmentId];
    const exec = s.executions[segmentId];
    const statuses = s.nodeStatuses[segmentId];
    const progress = s.nodeProgress[segmentId];
    const errors = s.nodeErrors[segmentId];
    // SETTLED = there is an execution and it is not still running. Only then are the node badges +
    // execution status meaningful to carry (a running source would leave the new id with stuck
    // waiting/running badges + a phantom execution nothing drives). The output CACHE always rides along
    // (harmless, and what the reconciler reads to re-deposit). New inner Records so later per-node writes
    // to either id don't alias; the path arrays inside are read-only and may stay shared.
    const settled = exec !== undefined && exec.status !== "running";
    // Strip the dispatch-time participant roster: it describes ONE run of ONE segment id and is only
    // consulted while status==="running" (never true for a snapshot), so carrying it to a paste/split
    // copy would just be a stale roster waiting to confuse a future reader.
    const { participants: _roster, ...execRest } = exec ?? {};
    const snap: SegmentRenderSnapshot = {
      ...(outs ? { nodeOutputs: { ...outs } } : {}),
      ...(settled && statuses ? { nodeStatuses: { ...statuses } } : {}),
      ...(settled && progress ? { nodeProgress: { ...progress } } : {}),
      ...(settled && errors ? { nodeErrors: { ...errors } } : {}),
      ...(settled ? { execution: execRest as ExecutionState } : {}),
    };
    return Object.keys(snap).length > 0 ? snap : null;
  },

  installSegmentState: (segmentId, snap) =>
    set((s) => ({
      nodeOutputs: snap.nodeOutputs ? { ...s.nodeOutputs, [segmentId]: { ...snap.nodeOutputs } } : s.nodeOutputs,
      nodeStatuses: snap.nodeStatuses ? { ...s.nodeStatuses, [segmentId]: { ...snap.nodeStatuses } } : s.nodeStatuses,
      nodeProgress: snap.nodeProgress ? { ...s.nodeProgress, [segmentId]: { ...snap.nodeProgress } } : s.nodeProgress,
      nodeErrors: snap.nodeErrors ? { ...s.nodeErrors, [segmentId]: { ...snap.nodeErrors } } : s.nodeErrors,
      executions: snap.execution ? { ...s.executions, [segmentId]: { ...snap.execution } } : s.executions,
    })),

  hydrateRenderState: (segmentId, nodeOutputs, completedNodeIds) =>
    set((s) => {
      const warm = s.nodeOutputs[segmentId];
      if (warm && Object.keys(warm).length > 0) return {}; // already warm — don't clobber a live run
      const statuses: Record<string, NodeStatus> = { ...(s.nodeStatuses[segmentId] ?? {}) };
      for (const id of completedNodeIds) statuses[id] = "completed";
      return {
        nodeOutputs: { ...s.nodeOutputs, [segmentId]: nodeOutputs },
        nodeStatuses: { ...s.nodeStatuses, [segmentId]: statuses },
      };
    }),

  setNodeOutputs: (segmentId, nodeId, paths) =>
    set((s) => ({
      nodeOutputs: {
        ...s.nodeOutputs,
        [segmentId]: { ...(s.nodeOutputs[segmentId] ?? {}), [nodeId]: paths },
      },
    })),

  clearNodeOutputs: (segmentId, nodeId) =>
    set((s) => {
      const seg = s.nodeOutputs[segmentId];
      if (!seg) return {};
      if (nodeId === undefined) {
        const { [segmentId]: _drop, ...rest } = s.nodeOutputs;
        return { nodeOutputs: rest };
      }
      if (!(nodeId in seg)) return {};
      const { [nodeId]: _dropNode, ...restNodes } = seg;
      return { nodeOutputs: { ...s.nodeOutputs, [segmentId]: restNodes } };
    }),

  setNodeStatus: (segmentId, nodeId, status) =>
    set((s) => ({
      nodeStatuses: {
        ...s.nodeStatuses,
        [segmentId]: { ...(s.nodeStatuses[segmentId] ?? {}), [nodeId]: status },
      },
    })),

  setNodeProgress: (segmentId, nodeId, progress) =>
    set((s) => ({
      nodeProgress: {
        ...s.nodeProgress,
        [segmentId]: { ...(s.nodeProgress[segmentId] ?? {}), [nodeId]: progress },
      },
    })),

  setNodeError: (segmentId, nodeId, error) =>
    set((s) => ({
      nodeErrors: {
        ...s.nodeErrors,
        [segmentId]: { ...(s.nodeErrors[segmentId] ?? {}), [nodeId]: error },
      },
    })),

  clearNodeStatuses: (segmentId) =>
    set((s) => {
      const { [segmentId]: _a, ...restStatuses } = s.nodeStatuses;
      const { [segmentId]: _b, ...restErrors } = s.nodeErrors;
      const { [segmentId]: _c, ...restProgress } = s.nodeProgress;
      return { nodeStatuses: restStatuses, nodeErrors: restErrors, nodeProgress: restProgress };
    }),

  // After a run settles (cancel / failure), drop running+waiting badges so nodes don't stay stuck
  // blue/yellow — but KEEP completed (green) and error (red) so the user still sees what finished/failed.
  clearPendingStatuses: (segmentId) =>
    set((s) => {
      const cur = s.nodeStatuses[segmentId];
      if (!cur) return {};
      const next: Record<string, NodeStatus> = {};
      for (const [id, st] of Object.entries(cur)) {
        if (st === "completed" || st === "error") next[id] = st;
      }
      return { nodeStatuses: { ...s.nodeStatuses, [segmentId]: next } };
    }),

  linkRender: (toId, fromId) =>
    set((s) => ({ renderLinks: { ...s.renderLinks, [toId]: fromId } })),

  unlinkRender: (toId) =>
    set((s) => {
      const { [toId]: _, ...rest } = s.renderLinks;
      return { renderLinks: rest };
    }),
}));
