import { create } from "zustand";
export const useAmtStore = create((set) => ({
    runs: {},
    setRun: (ctx) => set((s) => ({ runs: { ...s.runs, [ctx.nodeId]: ctx } })),
    clearRun: (nodeId) => set((s) => {
        const runs = { ...s.runs };
        delete runs[nodeId];
        return { runs };
    }),
    workbenchOpen: {},
    openWorkbench: (nodeId) => set((s) => ({ workbenchOpen: { ...s.workbenchOpen, [nodeId]: true } })),
    closeWorkbench: (nodeId) => set((s) => {
        const workbenchOpen = { ...s.workbenchOpen };
        delete workbenchOpen[nodeId];
        return { workbenchOpen };
    }),
}));
