import { create } from "zustand";
export const useWorkflowStore = create((set, get) => ({
    executions: {},
    nodeStatuses: {},
    nodeOutputs: {},
    nodeProgress: {},
    nodeErrors: {},
    renderLinks: {},
    singleNodeRunner: null,
    annotationEditor: null,
    registerSingleNodeRunner: (fn) => set({ singleNodeRunner: fn }),
    registerAnnotationEditor: (fn) => set({ annotationEditor: fn }),
    startExecution: (segmentId, participants) => set((s) => ({
        executions: {
            ...s.executions,
            [segmentId]: { status: "running", progress: 0, participants },
        },
    })),
    updateProgress: (segmentId, nodeId, progress) => set((s) => ({
        executions: {
            ...s.executions,
            // Preserve the dispatch snapshot — progress ticks rebuild the entry wholesale.
            // `cancelled` MUST ride along: the engine loop calls updateProgress at the TOP of every node
            // and polls isCancelled() a few lines later with no await in between, so dropping the flag here
            // erased every Stop that landed while a node was running. Purely-frontend nodes (the symbolic
            // family / autoArrange / split / dsp) never re-check it themselves, so the run drove to the end
            // and settled as "completed" — a Stop the user watched do nothing.
            [segmentId]: {
                status: "running",
                currentNodeId: nodeId,
                progress,
                participants: s.executions[segmentId]?.participants,
                ...(s.executions[segmentId]?.cancelled ? { cancelled: true } : {}),
            },
        },
    })),
    completeExecution: (segmentId) => set((s) => ({
        executions: {
            ...s.executions,
            [segmentId]: { status: "completed" },
        },
    })),
    failExecution: (segmentId, error) => set((s) => ({
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
    cancelExecution: (segmentId) => set((s) => {
        const cur = s.executions[segmentId];
        if (!cur)
            return {};
        return { executions: { ...s.executions, [segmentId]: { ...cur, cancelled: true } } };
    }),
    isCancelled: (segmentId) => {
        const exec = get().executions[segmentId];
        return exec?.cancelled === true;
    },
    clearExecution: (segmentId) => set((s) => {
        const { [segmentId]: _, ...rest } = s.executions;
        return { executions: rest };
    }),
    cloneSegmentState: (fromId, toId) => {
        if (!fromId || !toId) {
            console.warn('[Workflow] cloneSegmentState called with invalid id', { fromId, toId });
            return;
        }
        const snap = get().snapshotSegmentState(fromId);
        if (snap)
            get().installSegmentState(toId, snap);
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
        const snap = {
            ...(outs ? { nodeOutputs: { ...outs } } : {}),
            ...(settled && statuses ? { nodeStatuses: { ...statuses } } : {}),
            ...(settled && progress ? { nodeProgress: { ...progress } } : {}),
            ...(settled && errors ? { nodeErrors: { ...errors } } : {}),
            ...(settled ? { execution: execRest } : {}),
        };
        return Object.keys(snap).length > 0 ? snap : null;
    },
    installSegmentState: (segmentId, snap) => {
        if (!segmentId) {
            console.warn('[Workflow] installSegmentState called with invalid segmentId');
            return;
        }
        if (!snap)
            return;
        set((s) => {
            const updates = {};
            if (snap.nodeOutputs)
                updates.nodeOutputs = { ...s.nodeOutputs, [segmentId]: snap.nodeOutputs };
            if (snap.nodeStatuses)
                updates.nodeStatuses = { ...s.nodeStatuses, [segmentId]: snap.nodeStatuses };
            if (snap.nodeProgress)
                updates.nodeProgress = { ...s.nodeProgress, [segmentId]: snap.nodeProgress };
            if (snap.nodeErrors)
                updates.nodeErrors = { ...s.nodeErrors, [segmentId]: snap.nodeErrors };
            if (snap.execution)
                updates.executions = { ...s.executions, [segmentId]: snap.execution };
            return updates;
        });
    },
    hydrateRenderState: (segmentId, nodeOutputs, completedNodeIds) => set((s) => {
        const warm = s.nodeOutputs[segmentId];
        if (warm && Object.keys(warm).length > 0)
            return {}; // already warm — don't clobber a live run
        const statuses = { ...(s.nodeStatuses[segmentId] ?? {}) };
        for (const id of completedNodeIds)
            statuses[id] = "completed";
        return {
            nodeOutputs: { ...s.nodeOutputs, [segmentId]: nodeOutputs },
            nodeStatuses: { ...s.nodeStatuses, [segmentId]: statuses },
        };
    }),
    setNodeOutputs: (segmentId, nodeId, paths) => set((s) => ({
        nodeOutputs: {
            ...s.nodeOutputs,
            [segmentId]: { ...(s.nodeOutputs[segmentId] ?? {}), [nodeId]: paths },
        },
    })),
    clearNodeOutputs: (segmentId, nodeId) => set((s) => {
        const seg = s.nodeOutputs[segmentId];
        if (!seg)
            return {};
        if (nodeId === undefined) {
            const { [segmentId]: _drop, ...rest } = s.nodeOutputs;
            return { nodeOutputs: rest };
        }
        if (!(nodeId in seg))
            return {};
        const { [nodeId]: _dropNode, ...restNodes } = seg;
        return { nodeOutputs: { ...s.nodeOutputs, [segmentId]: restNodes } };
    }),
    setNodeStatus: (segmentId, nodeId, status) => set((s) => ({
        nodeStatuses: {
            ...s.nodeStatuses,
            [segmentId]: { ...(s.nodeStatuses[segmentId] ?? {}), [nodeId]: status },
        },
    })),
    setNodeProgress: (segmentId, nodeId, progress) => set((s) => ({
        nodeProgress: {
            ...s.nodeProgress,
            [segmentId]: { ...(s.nodeProgress[segmentId] ?? {}), [nodeId]: progress },
        },
    })),
    setNodeError: (segmentId, nodeId, error) => set((s) => ({
        nodeErrors: {
            ...s.nodeErrors,
            [segmentId]: { ...(s.nodeErrors[segmentId] ?? {}), [nodeId]: error },
        },
    })),
    clearNodeStatuses: (segmentId) => set((s) => {
        const { [segmentId]: _a, ...restStatuses } = s.nodeStatuses;
        const { [segmentId]: _b, ...restErrors } = s.nodeErrors;
        const { [segmentId]: _c, ...restProgress } = s.nodeProgress;
        return { nodeStatuses: restStatuses, nodeErrors: restErrors, nodeProgress: restProgress };
    }),
    // After a run settles (cancel / failure), drop running+waiting badges so nodes don't stay stuck
    // blue/yellow — but KEEP completed (green) and error (red) so the user still sees what finished/failed.
    clearPendingStatuses: (segmentId) => set((s) => {
        const cur = s.nodeStatuses[segmentId];
        if (!cur)
            return {};
        const next = {};
        for (const [id, st] of Object.entries(cur)) {
            // 规划 6-3: degraded（兜底透传）与 bypassed（用户旁通）都是「终态事实」，
            // 与 completed/error 一样要在失败/取消清扫中幸存，否则旁通徽标一闪即逝。
            if (st === "completed" || st === "error" || st === "degraded" || st === "bypassed")
                next[id] = st;
        }
        return { nodeStatuses: { ...s.nodeStatuses, [segmentId]: next } };
    }),
    linkRender: (toId, fromId) => set((s) => ({ renderLinks: { ...s.renderLinks, [toId]: fromId } })),
    unlinkRender: (toId) => set((s) => {
        const { [toId]: _, ...rest } = s.renderLinks;
        return { renderLinks: rest };
    }),
}));
