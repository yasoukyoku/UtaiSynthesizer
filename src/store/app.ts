import { create } from "zustand";

interface AppState {
  trainingPanelOpen: boolean;
  settingsOpen: boolean;
  activeTrackId: string | null;
  workflowSegmentId: string | null;
  zoom: number;
  scrollX: number;
  scrollY: number;

  toggleTrainingPanel: () => void;
  setActiveTrack: (id: string | null) => void;
  openWorkflow: (segmentId: string) => void;
  closeWorkflow: () => void;
  setZoom: (zoom: number) => void;
  setScroll: (x: number, y: number) => void;
}

export const useAppStore = create<AppState>((set) => ({
  trainingPanelOpen: false,
  settingsOpen: false,
  activeTrackId: null,
  workflowSegmentId: null,
  zoom: 1.0,
  scrollX: 0,
  scrollY: 0,

  toggleTrainingPanel: () =>
    set((s) => ({ trainingPanelOpen: !s.trainingPanelOpen })),
  setActiveTrack: (id) => set({ activeTrackId: id }),
  openWorkflow: (segmentId) => set({ workflowSegmentId: segmentId }),
  closeWorkflow: () => set({ workflowSegmentId: null }),
  setZoom: (zoom) => set({ zoom: Math.max(0.1, Math.min(10, zoom)) }),
  setScroll: (x, y) => set({ scrollX: x, scrollY: y }),
}));
