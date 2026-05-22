import { create } from "zustand";

export type NodeStatus = "idle" | "waiting" | "running" | "completed" | "error";

export interface ExecutionState {
  status: "idle" | "running" | "completed" | "error";
  currentNodeId?: string;
  progress?: number;
  error?: string;
  cancelled?: boolean;
}

interface WorkflowStore {
  executions: Record<string, ExecutionState>;
  nodeStatuses: Record<string, Record<string, NodeStatus>>;
  nodeOutputs: Record<string, Record<string, string[]>>;
  nodeErrors: Record<string, Record<string, string>>;
  singleNodeRunner: ((nodeId: string) => void) | null;

  registerSingleNodeRunner: (fn: ((nodeId: string) => void) | null) => void;
  startExecution: (segmentId: string) => void;
  updateProgress: (segmentId: string, nodeId: string, progress: number) => void;
  completeExecution: (segmentId: string) => void;
  failExecution: (segmentId: string, error: string) => void;
  cancelExecution: (segmentId: string) => void;
  isCancelled: (segmentId: string) => boolean;
  clearExecution: (segmentId: string) => void;
  setNodeOutputs: (segmentId: string, nodeId: string, paths: string[]) => void;

  setNodeStatus: (segmentId: string, nodeId: string, status: NodeStatus) => void;
  setNodeError: (segmentId: string, nodeId: string, error: string) => void;
  clearNodeStatuses: (segmentId: string) => void;
}

export const useWorkflowStore = create<WorkflowStore>((set, get) => ({
  executions: {},
  nodeStatuses: {},
  nodeOutputs: {},
  nodeErrors: {},
  singleNodeRunner: null,

  registerSingleNodeRunner: (fn) => set({ singleNodeRunner: fn }),
  startExecution: (segmentId) =>
    set((s) => ({
      executions: {
        ...s.executions,
        [segmentId]: { status: "running", progress: 0 },
      },
    })),

  updateProgress: (segmentId, nodeId, progress) =>
    set((s) => ({
      executions: {
        ...s.executions,
        [segmentId]: { status: "running", currentNodeId: nodeId, progress },
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

  cancelExecution: (segmentId) =>
    set((s) => ({
      executions: {
        ...s.executions,
        [segmentId]: { ...s.executions[segmentId]!, status: "error", error: "Cancelled", cancelled: true },
      },
    })),

  isCancelled: (segmentId) => {
    const exec = get().executions[segmentId];
    return exec?.cancelled === true;
  },

  clearExecution: (segmentId) =>
    set((s) => {
      const { [segmentId]: _, ...rest } = s.executions;
      return { executions: rest };
    }),

  setNodeOutputs: (segmentId, nodeId, paths) =>
    set((s) => ({
      nodeOutputs: {
        ...s.nodeOutputs,
        [segmentId]: { ...(s.nodeOutputs[segmentId] ?? {}), [nodeId]: paths },
      },
    })),

  setNodeStatus: (segmentId, nodeId, status) =>
    set((s) => ({
      nodeStatuses: {
        ...s.nodeStatuses,
        [segmentId]: { ...(s.nodeStatuses[segmentId] ?? {}), [nodeId]: status },
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
      return { nodeStatuses: restStatuses, nodeErrors: restErrors };
    }),
}));
