import { create } from "zustand";

interface SegmentSelection {
  trackId: string;
  segmentId: string;
}

interface ToastState {
  message: string;
  type: "error" | "info" | "success";
  id: number;
}

interface AppState {
  trainingPanelOpen: boolean;
  modelManagerOpen: boolean;
  logViewerOpen: boolean;
  settingsOpen: boolean;
  toggleSettings: () => void;
  activeTrackId: string | null;
  selectedSegment: SegmentSelection | null;
  workflowSegmentId: string | null;
  zoom: number;
  scrollX: number;
  scrollY: number;
  canvasWidth: number;
  toasts: ToastState[];

  toggleTrainingPanel: () => void;
  toggleModelManager: () => void;
  toggleLogViewer: () => void;
  setActiveTrack: (id: string | null) => void;
  selectSegment: (trackId: string, segmentId: string) => void;
  clearSelection: () => void;
  openWorkflow: (segmentId: string) => void;
  closeWorkflow: () => void;
  setZoom: (zoom: number) => void;
  setScroll: (x: number, y: number) => void;
  setCanvasWidth: (w: number) => void;
  showToast: (message: string, type?: "error" | "info" | "success") => void;
  dismissToast: (id: number) => void;
}

export const useAppStore = create<AppState>((set) => ({
  trainingPanelOpen: false,
  modelManagerOpen: false,
  logViewerOpen: false,
  settingsOpen: false,
  activeTrackId: null,
  selectedSegment: null,
  workflowSegmentId: null,
  zoom: 1.0,
  scrollX: 0,
  scrollY: 0,
  canvasWidth: 800,
  toasts: [],

  toggleTrainingPanel: () =>
    set((s) => ({ trainingPanelOpen: !s.trainingPanelOpen })),
  toggleModelManager: () =>
    set((s) => ({ modelManagerOpen: !s.modelManagerOpen })),
  toggleLogViewer: () =>
    set((s) => ({ logViewerOpen: !s.logViewerOpen })),
  toggleSettings: () =>
    set((s) => ({ settingsOpen: !s.settingsOpen })),
  setActiveTrack: (id) => set({ activeTrackId: id }),
  selectSegment: (trackId, segmentId) =>
    set({ selectedSegment: { trackId, segmentId }, activeTrackId: trackId }),
  clearSelection: () => set({ selectedSegment: null }),
  openWorkflow: (segmentId) => set({ workflowSegmentId: segmentId }),
  closeWorkflow: () => set({ workflowSegmentId: null }),
  setZoom: (zoom) => set({ zoom: Math.max(0.1, Math.min(10, zoom)) }),
  setScroll: (x, y) => set({ scrollX: x, scrollY: y }),
  setCanvasWidth: (w) => set({ canvasWidth: w }),
  showToast: (message, type = "error") => {
    const id = Date.now();
    set((s) => ({ toasts: [...s.toasts, { message, type, id }] }));
    setTimeout(() => {
      set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) }));
    }, 5000);
  },
  dismissToast: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
}));
