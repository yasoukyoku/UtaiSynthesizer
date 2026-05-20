import { create } from "zustand";

interface SegmentSelection {
  trackId: string;
  segmentId: string;
}

interface AppState {
  trainingPanelOpen: boolean;
  settingsOpen: boolean;
  activeTrackId: string | null;
  selectedSegment: SegmentSelection | null;
  workflowSegmentId: string | null;
  zoom: number;
  scrollX: number;
  scrollY: number;

  toggleTrainingPanel: () => void;
  setActiveTrack: (id: string | null) => void;
  selectSegment: (trackId: string, segmentId: string) => void;
  clearSelection: () => void;
  openWorkflow: (segmentId: string) => void;
  closeWorkflow: () => void;
  setZoom: (zoom: number) => void;
  setScroll: (x: number, y: number) => void;
}

export const useAppStore = create<AppState>((set) => ({
  trainingPanelOpen: false,
  settingsOpen: false,
  activeTrackId: null,
  selectedSegment: null,
  workflowSegmentId: null,
  zoom: 1.0,
  scrollX: 0,
  scrollY: 0,

  toggleTrainingPanel: () =>
    set((s) => ({ trainingPanelOpen: !s.trainingPanelOpen })),
  setActiveTrack: (id) => set({ activeTrackId: id }),
  selectSegment: (trackId, segmentId) =>
    set({ selectedSegment: { trackId, segmentId }, activeTrackId: trackId }),
  clearSelection: () => set({ selectedSegment: null }),
  openWorkflow: (segmentId) => set({ workflowSegmentId: segmentId }),
  closeWorkflow: () => set({ workflowSegmentId: null }),
  setZoom: (zoom) => set({ zoom: Math.max(0.1, Math.min(10, zoom)) }),
  setScroll: (x, y) => set({ scrollX: x, scrollY: y }),
}));
