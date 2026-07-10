import { create } from "zustand";
import { loadSetting, saveSetting } from "../lib/settings";

interface SegmentSelection {
  trackId: string;
  segmentId: string;
}

/** A selected sub-lane GROUP within a segment (P3). Output nodes are many-to-one: `outputNodeId` selects
 *  ALL lanes fanned into that Output node. `clipIndex` = the piece under the cursor at click time — the
 *  target for Ctrl+K slice / Delete. Distinct from segment selection; drives lane trim/slice/delete. */
interface LaneSelection {
  trackId: string;
  segmentId: string;
  outputNodeId: string;
  clipIndex: number;
}

interface ToastState {
  message: string;
  type: "error" | "info" | "success";
  id: number;
}

export interface ConfirmButton {
  id: string;
  label: string;
  /** Visual emphasis — "primary" = accent, "danger" = destructive; omit for a neutral button. */
  kind?: "primary" | "danger";
}

/** Optional text-input mode for the confirm dialog (e.g. "new group" name prompt). When present, the
 *  PRIMARY button (and Enter) resolves with the TRIMMED input value instead of the button id; a
 *  non-empty value is always required, `invalid` adds extra validation (returns an error message to
 *  show, or null when ok). Cancel/Esc/backdrop still resolve "". */
export interface ConfirmInput {
  placeholder?: string;
  initial?: string;
  invalid?: (value: string) => string | null;
}

interface ConfirmRequest {
  title: string;
  body: string;
  buttons: ConfirmButton[];
  input?: ConfirmInput;
  resolve: (id: string) => void;
  seq: number;
}

interface AppState {
  trainingPageOpen: boolean;
  modelManagerOpen: boolean;
  logViewerOpen: boolean;
  settingsOpen: boolean;
  toggleSettings: () => void;
  activeTrackId: string | null;
  /** Primary/anchor selection (drives Toolbar split/delete). */
  selectedSegment: SegmentSelection | null;
  /** Full multi-selection set (Ctrl+click adds/removes; drives highlight + move-together). */
  selectedSegments: SegmentSelection[];
  /** Selected sub-lane group (P3), or null. Non-null makes Ctrl+K/Delete act on the lane, not the segment. */
  selectedLane: LaneSelection | null;
  workflowSegmentId: string | null;
  /** ② The notes segment whose VOCAL (piano-roll) editor is docked at the bottom, or null. Mutually
   *  exclusive with workflowSegmentId — the bottom dock shows one editor at a time (a segment is either a
   *  notes part or an audioClip). Mirrors workflowSegmentId (§9.6). */
  vocalSegmentId: string | null;
  /** A requested Output-group DETACH ("ungroup") waiting for that segment's workflow editor to perform
   *  it — the editor is the ONE code path (its graph state + local undo own the op); a timeline
   *  right-click first opens the editor, then this hands the request over. Consumed on mount/change. */
  pendingLaneDetach: { segmentId: string; outputNodeId: string } | null;
  /** Which pane owns Ctrl+Z and the Delete/Ctrl+K edit keys: the track timeline, the bottom-docked
   *  workflow editor, or the bottom-docked vocal (piano-roll) editor. Set on pointer/focus into each pane. */
  activePane: "timeline" | "workflow" | "vocal";
  /** Height (px) of the bottom workflow panel when open; persisted across sessions. */
  workflowPanelHeight: number;
  /** ② Height (px) of the bottom vocal-editor panel when open; persisted (own value, §9.0). */
  vocalPanelHeight: number;
  zoom: number;
  /** Vertical zoom — scales track display height (header + lanes). */
  vZoom: number;
  scrollX: number;
  scrollY: number;
  canvasWidth: number;
  canvasHeight: number;
  /** While dragging a NEW track in between existing ones, the placeholder gap to open (so both the
   *  canvas and the track-header column show an empty slot at `index`, `count` rows tall). */
  ghostInsert: { index: number; count: number } | null;
  /** Snap dragged/resized clips to other clips' edges + the playhead. */
  snapSegments: boolean;
  /** Snap the playhead (drag on ruler / arrangement) to clip edges. */
  snapPlayhead: boolean;
  /** ② A vocal (score→singing) render is in flight — GLOBAL single-flight: only one at a time, since the
   *  shared ORT engine + release_gpu_sessions_except would make concurrent renders evict each other's
   *  session mid-inference. Gates the Render button everywhere + backs the throw-guard in vocalRender.ts. */
  vocalRenderActive: boolean;
  toasts: ToastState[];
  /** Transient corner banner (undo/redo info, save/load confirmation, …). `seq` bumps each time so a
   *  rapid retrigger updates the same single banner in place (no stacking, no viewport jump). */
  banner: { message: string; kind: BannerKind; seq: number } | null;
  /** A pending styled confirm dialog (replaces the native `ask` popup). null = nothing shown. */
  confirm: ConfirmRequest | null;

  toggleTrainingPage: () => void;
  toggleModelManager: () => void;
  toggleLogViewer: () => void;
  setActiveTrack: (id: string | null) => void;
  selectSegment: (trackId: string, segmentId: string) => void;
  toggleSegment: (trackId: string, segmentId: string) => void;
  /** Select a sub-lane group (all lanes of `outputNodeId`) in a segment, with the clicked piece index. */
  selectLane: (trackId: string, segmentId: string, outputNodeId: string, clipIndex: number) => void;
  clearSelection: () => void;
  openWorkflow: (segmentId: string) => void;
  closeWorkflow: () => void;
  /** ② Open the vocal (piano-roll) editor on a notes segment; closes any open workflow editor (§9.6). */
  openVocalEditor: (segmentId: string) => void;
  closeVocalEditor: () => void;
  requestLaneDetach: (segmentId: string, outputNodeId: string) => void;
  clearLaneDetach: () => void;
  setActivePane: (pane: "timeline" | "workflow" | "vocal") => void;
  setWorkflowPanelHeight: (h: number) => void;
  setVocalPanelHeight: (h: number) => void;
  setZoom: (zoom: number) => void;
  setVZoom: (vZoom: number) => void;
  setScroll: (x: number, y: number) => void;
  setCanvasWidth: (w: number) => void;
  setCanvasHeight: (h: number) => void;
  setGhostInsert: (g: { index: number; count: number } | null) => void;
  toggleSnapSegments: () => void;
  toggleSnapPlayhead: () => void;
  setVocalRenderActive: (v: boolean) => void;
  showToast: (message: string, type?: "error" | "info" | "success") => void;
  dismissToast: (id: number) => void;
  showBanner: (message: string, kind: BannerKind) => void;
  /** Show a styled confirm dialog; resolves with the chosen button id, or "" if dismissed (Esc/backdrop).
   *  With `input` set, the primary button/Enter resolves the trimmed input VALUE instead (see ConfirmInput). */
  showConfirm: (opts: { title: string; body: string; buttons: ConfirmButton[]; input?: ConfirmInput }) => Promise<string>;
}

export type BannerKind = "undo" | "redo" | "save" | "load" | "info";

export const useAppStore = create<AppState>((set, get) => ({
  trainingPageOpen: false,
  modelManagerOpen: false,
  logViewerOpen: false,
  settingsOpen: false,
  activeTrackId: null,
  selectedSegment: null,
  selectedSegments: [],
  selectedLane: null,
  workflowSegmentId: null,
  vocalSegmentId: null,
  pendingLaneDetach: null,
  activePane: "timeline",
  workflowPanelHeight: loadSetting("utai.workflowPanelHeight", 460),
  vocalPanelHeight: loadSetting("utai.vocalPanelHeight", 460),
  zoom: 1.0,
  vZoom: 1.0,
  scrollX: 0,
  scrollY: 0,
  canvasWidth: 800,
  canvasHeight: 600,
  ghostInsert: null,
  snapSegments: loadSetting("utai.snapSegments", true),
  snapPlayhead: loadSetting("utai.snapPlayhead", true),
  vocalRenderActive: false,
  toasts: [],
  banner: null,
  confirm: null,

  toggleTrainingPage: () =>
    set((s) => ({ trainingPageOpen: !s.trainingPageOpen })),
  toggleModelManager: () =>
    set((s) => ({ modelManagerOpen: !s.modelManagerOpen })),
  toggleLogViewer: () =>
    set((s) => ({ logViewerOpen: !s.logViewerOpen })),
  toggleSettings: () =>
    set((s) => ({ settingsOpen: !s.settingsOpen })),
  setActiveTrack: (id) => set({ activeTrackId: id }),
  selectSegment: (trackId, segmentId) =>
    set({
      selectedSegment: { trackId, segmentId },
      selectedSegments: [{ trackId, segmentId }],
      selectedLane: null,
      activeTrackId: trackId,
    }),
  toggleSegment: (trackId, segmentId) =>
    set((s) => {
      const exists = s.selectedSegments.some((x) => x.trackId === trackId && x.segmentId === segmentId);
      const next = exists
        ? s.selectedSegments.filter((x) => !(x.trackId === trackId && x.segmentId === segmentId))
        : [...s.selectedSegments, { trackId, segmentId }];
      return {
        selectedSegments: next,
        selectedSegment: exists ? (next[next.length - 1] ?? null) : { trackId, segmentId },
        selectedLane: null,
        activeTrackId: trackId,
      };
    }),
  // Selecting a lane anchors the parent as `selectedSegment` (so Ctrl+K/Delete + the Split button have a
  // coherent fallback target) but does NOT add it to `selectedSegments` — otherwise the parent segment
  // ALSO lit up gold, competing with the sub-lane's own gold highlight and reading as confusing. Only the
  // sub-lane group is cued (via selectedLane); the non-null selectedLane routes the edit to the lane.
  selectLane: (trackId, segmentId, outputNodeId, clipIndex) =>
    set({
      selectedLane: { trackId, segmentId, outputNodeId, clipIndex },
      selectedSegment: { trackId, segmentId },
      selectedSegments: [],
      activeTrackId: trackId,
    }),
  clearSelection: () => set({ selectedSegment: null, selectedSegments: [], selectedLane: null }),
  // Opening either bottom-dock editor closes the OTHER (the dock shows one at a time) so activePane, the
  // divider cue, and undo routing can never point at a hidden editor (§9.6 exclusivity).
  openWorkflow: (segmentId) => set({ workflowSegmentId: segmentId, vocalSegmentId: null, activePane: "workflow" }),
  closeWorkflow: () => set({ workflowSegmentId: null, activePane: "timeline" }),
  openVocalEditor: (segmentId) => set({ vocalSegmentId: segmentId, workflowSegmentId: null, activePane: "vocal" }),
  closeVocalEditor: () => set({ vocalSegmentId: null, activePane: "timeline" }),
  requestLaneDetach: (segmentId, outputNodeId) => set({ pendingLaneDetach: { segmentId, outputNodeId } }),
  clearLaneDetach: () => set({ pendingLaneDetach: null }),
  setActivePane: (pane) => set((s) => (s.activePane === pane ? s : { activePane: pane })),
  setWorkflowPanelHeight: (h) => set({ workflowPanelHeight: h }),
  setVocalPanelHeight: (h) => set({ vocalPanelHeight: h }),
  setZoom: (zoom) => set({ zoom: Math.max(0.1, Math.min(10, zoom)) }),
  setVZoom: (vZoom) => set({ vZoom: Math.max(0.6, Math.min(3, vZoom)) }),
  setScroll: (x, y) => set({ scrollX: x, scrollY: y }),
  setCanvasWidth: (w) => set({ canvasWidth: w }),
  setCanvasHeight: (h) => set({ canvasHeight: h }),
  setGhostInsert: (g) => set({ ghostInsert: g }),
  toggleSnapSegments: () =>
    set((s) => {
      const v = !s.snapSegments;
      saveSetting("utai.snapSegments", v);
      return { snapSegments: v };
    }),
  toggleSnapPlayhead: () =>
    set((s) => {
      const v = !s.snapPlayhead;
      saveSetting("utai.snapPlayhead", v);
      return { snapPlayhead: v };
    }),
  setVocalRenderActive: (v) => set({ vocalRenderActive: v }),
  showToast: (message, type = "error") => {
    const id = Date.now();
    set((s) => ({ toasts: [...s.toasts, { message, type, id }] }));
    setTimeout(() => {
      set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) }));
    }, 5000);
  },
  dismissToast: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
  showBanner: (message, kind) =>
    set((s) => ({ banner: { message, kind, seq: (s.banner?.seq ?? 0) + 1 } })),
  showConfirm: (opts) =>
    new Promise<string>((resolve) => {
      // Capture the prior seq BEFORE settling the previous dialog: its resolve() nulls `confirm`, so
      // reading `s.confirm?.seq` afterwards always yielded 1 — a stacked dialog then reused the prior
      // dialog's keyed input state (ConfirmDialog remounts on `key={seq}`).
      const prevSeq = get().confirm?.seq ?? 0;
      // Settle any already-open dialog as dismissed first, so its awaiter never hangs (e.g. the native
      // window-close button firing onCloseRequested while a New/Open discard dialog is still up).
      get().confirm?.resolve("");
      set({
        confirm: {
          title: opts.title,
          body: opts.body,
          buttons: opts.buttons,
          input: opts.input,
          seq: prevSeq + 1,
          resolve: (id: string) => {
            set({ confirm: null });
            resolve(id);
          },
        },
      });
    }),
}));
