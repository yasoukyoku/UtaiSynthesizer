import { useRef, useEffect, useState, useCallback, lazy, Suspense } from "react";
import { DawView } from "./DawView";
import { WorkflowEditor } from "../workflow/WorkflowEditor";
import { VocalEditor } from "./VocalEditor";
import { TrackInspectorPanel } from "./TrackInspectorPanel";
import { VirtualPiano } from "./VirtualPiano";
import { ErrorBoundary } from "../common/ErrorBoundary";
import { useAppStore } from "../../store/app";
import { useProjectStore } from "../../store/project";
import { saveSetting } from "../../lib/settings";
import "./DawWorkflowSplit.css";

// AmtResultPanel is heavy (129KB bundle); lazy-load so the base DAW shell stays lean
// when no conversion result is on screen.
const AmtResultPanel = lazy(() =>
  import("../amt/AmtResultPanel").then((m) => ({ default: m.AmtResultPanel })),
);

// SynthV/OpenUTAU-style vertical split: the track arrangement (DawView) on top, and ONE per-segment editor
// docked as a RESIZABLE bottom panel — the node workflow editor (audioClip segment) OR the ② vocal
// piano-roll editor (notes segment). They are mutually exclusive (the app store opens one and closes the
// other), so a single divider + clamp serves whichever is open; each keeps its OWN persisted height. The
// track area REALLY shrinks when the panel opens/grows, so DawView's ResizeObserver chain self-corrects.

const MIN_PANEL = 160; // the editor never smaller than this
const MIN_TRACKS = 120; // the track area never smaller than this
const DIVIDER_H = 6;

export function DawWorkflowSplit() {
  const workflowSegmentId = useAppStore((s) => s.workflowSegmentId);
  const vocalSegmentId = useAppStore((s) => s.vocalSegmentId);
  const amtResult = useAppStore((s) => s.amtResult);
  const inspectorTrackId = useAppStore((s) => s.inspectorTrackId);
  const closeWorkflow = useAppStore((s) => s.closeWorkflow);
  const closeVocalEditor = useAppStore((s) => s.closeVocalEditor);
  const closeAmtResult = useAppStore((s) => s.closeAmtResult);
  const activePane = useAppStore((s) => s.activePane); // drives the active-pane cue on the divider
  const workflowPanelHeight = useAppStore((s) => s.workflowPanelHeight);
  const vocalPanelHeight = useAppStore((s) => s.vocalPanelHeight);
  const amtPanelHeight = useAppStore((s) => s.amtPanelHeight);
  const setWorkflowPanelHeight = useAppStore((s) => s.setWorkflowPanelHeight);
  const setVocalPanelHeight = useAppStore((s) => s.setVocalPanelHeight);
  const setAmtPanelHeight = useAppStore((s) => s.setAmtPanelHeight);
  const tracks = useProjectStore((s) => s.tracks);
  const splitRef = useRef<HTMLDivElement>(null);
  const [splitH, setSplitH] = useState(0); // measured container height; 0 = not yet measured
  const dragRef = useRef<{ startY: number; startH: number; splitH: number } | null>(null);

  const isVocal = vocalSegmentId != null;
  const isWorkflow = workflowSegmentId != null;
  const isAmt = amtResult != null;
  const panelOpen = isWorkflow || isVocal || isAmt;
  const panelHeight = isVocal ? vocalPanelHeight : isAmt ? amtPanelHeight : workflowPanelHeight;
  const setPanelHeight = isVocal
    ? setVocalPanelHeight
    : isAmt
      ? setAmtPanelHeight
      : setWorkflowPanelHeight;

  // Close a docked editor if its segment vanishes out from under it (deleting it / its track / a load).
  useEffect(() => {
    if (workflowSegmentId && !tracks.some((t) => t.segments.some((s) => s.id === workflowSegmentId))) closeWorkflow();
    if (vocalSegmentId && !tracks.some((t) => t.segments.some((s) => s.id === vocalSegmentId))) closeVocalEditor();
  }, [tracks, workflowSegmentId, vocalSegmentId, closeWorkflow, closeVocalEditor]);

  // Measure the split container so we can CLAMP the rendered panel height to what fits, WITHOUT mutating
  // the stored preference (see the earlier one-way-ratchet fix).
  useEffect(() => {
    const el = splitRef.current;
    if (!el) return;
    const measure = () => setSplitH(el.clientHeight);
    measure();
    const ob = new ResizeObserver(measure);
    ob.observe(el);
    return () => ob.disconnect();
  }, []);

  const maxPanel = Math.max(MIN_PANEL, splitH - MIN_TRACKS - DIVIDER_H);
  const effectiveHeight = splitH > 0 ? Math.min(panelHeight, maxPanel) : panelHeight;

  const endDrag = useCallback((e: React.PointerEvent) => {
    if (!dragRef.current) return;
    dragRef.current = null;
    (e.currentTarget as Element).releasePointerCapture?.(e.pointerId);
    // Persist the height of WHICHEVER editor is open (read live so a mid-drag pane swap can't cross wires).
    const st = useAppStore.getState();
    if (st.vocalSegmentId != null) saveSetting("utai.vocalPanelHeight", st.vocalPanelHeight);
    else if (st.amtResult != null) saveSetting("utai.amtPanelHeight", st.amtPanelHeight);
    else saveSetting("utai.workflowPanelHeight", st.workflowPanelHeight);
  }, []);

  const onDividerDown = useCallback(
    (e: React.PointerEvent) => {
      e.preventDefault();
      dragRef.current = { startY: e.clientY, startH: panelHeight, splitH: splitRef.current?.clientHeight ?? 0 };
      (e.currentTarget as Element).setPointerCapture(e.pointerId);
    },
    [panelHeight],
  );

  const onDividerMove = useCallback(
    (e: React.PointerEvent) => {
      const d = dragRef.current;
      if (!d) return;
      if (e.buttons === 0) { endDrag(e); return; }
      const dy = e.clientY - d.startY; // dragging UP (dy<0) grows the bottom panel
      const max = Math.max(MIN_PANEL, d.splitH - MIN_TRACKS - DIVIDER_H);
      setPanelHeight(Math.max(MIN_PANEL, Math.min(max, d.startH - dy)));
    },
    [setPanelHeight, endDrag],
  );

  return (
    <div className={`daw-split pane-${activePane}${panelOpen ? " panel-open" : ""}`} ref={splitRef}>
      <DawView />
      {panelOpen && (
        <>
          <div
            className="workflow-divider"
            onPointerDown={onDividerDown}
            onPointerMove={onDividerMove}
            onPointerUp={endDrag}
            onPointerCancel={endDrag}
            onLostPointerCapture={endDrag}
          />
          {/* key={segmentId/amtResult.trackId}: switching targets REMOUNTS the editor so its view/undo/save scope resets. */}
          {isVocal ? (
            <ErrorBoundary key={`vocal-eb-${vocalSegmentId}`}>
              <VocalEditor key={vocalSegmentId} segmentId={vocalSegmentId!} onClose={closeVocalEditor} style={{ height: effectiveHeight }} />
            </ErrorBoundary>
          ) : isAmt && amtResult ? (
            <ErrorBoundary key={`amt-eb-${amtResult.trackId}`}>
              <Suspense fallback={null}>
                <AmtResultPanel key={`amt-${amtResult.trackId}`} result={amtResult} onClose={closeAmtResult} style={{ height: effectiveHeight }} />
              </Suspense>
            </ErrorBoundary>
          ) : (
            <ErrorBoundary key={`workflow-eb-${workflowSegmentId}`}>
              <WorkflowEditor key={workflowSegmentId} segmentId={workflowSegmentId!} onClose={closeWorkflow} style={{ height: effectiveHeight }} />
            </ErrorBoundary>
          )}
        </>
      )}
      {/* Studio Pro 风格底部 docked Inspector 面板 */}
      {inspectorTrackId && <TrackInspectorPanel />}
      {/* 🎹 虚拟钢琴: 固定在整屏最底部 — 无论工作流/MIDI编辑/Inspector 是否打开都始终垫底 */}
      <VirtualPiano />
    </div>
  );
}
