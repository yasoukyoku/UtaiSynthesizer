import { create } from "zustand";

/** Context captured when an AMT node finishes, so the MIDI workbench can
 *  re-run transcription with different quantize/tempo settings without the
 *  user having to rewire the graph. */
export interface AmtRunContext {
  nodeId: string;
  audioPath: string;
  midiPath: string;
  mode: string;
  backend: string;
  totalNotes: number | null;
  processingTimeSecs: number;
}

interface AmtStore {
  runs: Record<string, AmtRunContext>;
  setRun: (ctx: AmtRunContext) => void;
  clearRun: (nodeId: string) => void;
  /** Workbench open state keyed by nodeId. */
  workbenchOpen: Record<string, boolean>;
  openWorkbench: (nodeId: string) => void;
  closeWorkbench: (nodeId: string) => void;
}

export const useAmtStore = create<AmtStore>((set) => ({
  runs: {},
  setRun: (ctx) => set((s) => ({ runs: { ...s.runs, [ctx.nodeId]: ctx } })),
  clearRun: (nodeId) =>
    set((s) => {
      const runs = { ...s.runs };
      delete runs[nodeId];
      return { runs };
    }),
  workbenchOpen: {},
  openWorkbench: (nodeId) =>
    set((s) => ({ workbenchOpen: { ...s.workbenchOpen, [nodeId]: true } })),
  closeWorkbench: (nodeId) =>
    set((s) => {
      const workbenchOpen = { ...s.workbenchOpen };
      delete workbenchOpen[nodeId];
      return { workbenchOpen };
    }),
}));
